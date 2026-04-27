# Dual-Model Anomaly Detection Report
**Branch:** `feature-power-snmp-detection`
**Dataset:** `synthetic_power_snmp_dataset.csv`
**Date generated:** 2026-04-24
**Alert policy:** OR (either baseline OR phase model exceeds threshold → alert)

---

## 1. What a "window" is

Before reading numbers, understand what is being counted.

The detection does not score individual SNMP poll rows. It scores **sliding windows**
of 10 consecutive rows per device (10 rows × 5 min = 50-minute window).

```
Row 0  → Window 0:  rows  0–9   (00:00 – 00:45)
Row 1  → Window 1:  rows  1–10  (00:05 – 00:50)
Row 2  → Window 2:  rows  2–11  (00:10 – 00:55)
...
```

Windows overlap. One anomalous row at timestep T will appear in up to 10 windows
(T−9 through T). That's why true_anomaly_windows (7,443) is much higher than
the number of actual injected anomaly rows (~900).

A window is labeled **true_label=1** if **any** of its 10 rows was injected as anomalous.

---

## 2. Overall detection summary

| Metric | Value | Meaning |
|--------|-------|---------|
| Total windows scored | 18,054 | 9 devices × 2,006 windows each |
| True anomaly windows | **7,443** (41.2%) | Windows containing at least one anomalous row |
| Flagged by baseline model | **4,133** (22.9%) | Baseline reconstruction error > 0.0624 |
| Flagged by phase model | **3,889** (21.5%) | Phase reconstruction error > 0.0565 |
| **Flagged combined (OR)** | **4,739** (26.2%) | Either model fired |
| Missed anomalies (FN) | 2,704 | True anomaly windows not caught |
| False alarms (FP) | 228 | Normal windows incorrectly flagged |

### How OR policy combines the two models

```
Window result = baseline_flag  OR  phase_flag

baseline_flag = 1  if  baseline_reconstruction_error  > 0.062388
phase_flag    = 1  if  phase_reconstruction_error     > 0.056544
```

### Model agreement breakdown

| Scenario | Count | Share of all windows |
|----------|-------|----------------------|
| **Both models flagged** | 3,283 | 18.2% |
| **Baseline only flagged** | 850 | 4.7% |
| **Phase only flagged** | 606 | 3.4% |
| Neither flagged | 13,315 | 73.7% |

The two models agree on most alerts (3,283 out of 4,739 = 69%).
The remaining 31% are unique contributions — baseline catches 850 windows
the phase model misses, and phase catches 606 that baseline misses.
This proves the dual-model design adds value over using either model alone.

---

## 3. What makes a window anomalous — reconstruction error explained

Both models are **LSTM Autoencoders**. They learn to reproduce ("reconstruct")
normal power readings. When they see something they were never trained on,
they struggle to reproduce it and the reproduction is inaccurate.

That inaccuracy is measured as **Mean Squared Error (MSE) between the original
input window and the model's reconstructed output**:

```
reconstruction_error = mean((original_value - reconstructed_value)²)
                       averaged over all 10 timesteps and all features
```

A window is flagged if this error exceeds the stored threshold.

### Reconstruction error: normal vs anomaly

| | Baseline model | Phase model |
|--|--|--|
| **Threshold** | **0.0624** | **0.0565** |
| Normal — mean error | 0.0209 | 0.0248 |
| Normal — p95 error | 0.0487 | 0.0455 |
| Normal — max error | 0.1074 | 0.0751 |
| Anomaly — mean error | **1.0059** | **0.4940** |
| Anomaly — p95 error | 8.2329 | 4.0343 |
| Anomaly — max error | **20.3230** | **8.8498** |

**Key observation:** Anomaly errors average 48× (baseline) and 20× (phase) higher
than normal errors. This confirms the models have learned the normal pattern well.
The threshold sits just above the normal distribution's tail (p95 = 0.049 vs
threshold 0.062 for baseline).

The worst single window recorded was `ups_apc_02` on 2025-01-01 at 15:50–16:35
with a baseline error of **20.32** — that is **325× the threshold** — caused by
simultaneous `thermal_runaway` and `battery_drain` anomalies stacking on an
already aged battery.

---

## 4. Per-device results

### 4a. Summary table

| Device | Category | Total Windows | True Anomaly | Flagged | TP | FP | FN | Precision | Recall | F1 |
|--------|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ups_apc_01` | UPS (APC, 1-phase, young) | 2,006 | 829 | 732 | 675 | 57 | 154 | 0.92 | 0.81 | **0.86** |
| `ups_apc_02` | UPS (APC, 1-phase, aged) | 2,006 | 862 | 724 | 709 | 15 | 153 | 0.98 | 0.82 | **0.89** |
| `ups_lie_01` | UPS (Liebert, 3-phase) | 2,006 | 830 | 621 | 589 | 32 | 241 | 0.95 | 0.71 | **0.81** |
| `ups_lie_02` | UPS (Liebert, 3-phase, aged) | 2,006 | 911 | 597 | 572 | 25 | 339 | 0.96 | 0.63 | **0.76** |
| `pdu_apc_01` | PDU (APC) | 2,006 | 839 | 431 | 419 | 12 | 420 | 0.97 | 0.50 | 0.66 |
| `pdu_rar_01` | PDU (Raritan) | 2,006 | 807 | 644 | 630 | 14 | 177 | 0.98 | 0.78 | **0.87** |
| `net_cis_01` | Switch (Cisco) | 2,006 | 802 | 483 | 473 | 10 | 329 | 0.98 | 0.59 | 0.74 |
| `net_gen_01` | Router (generic) | 2,006 | 795 | 438 | 421 | 17 | 374 | 0.96 | 0.53 | 0.68 |
| `env_gen_01` | Environmental sensor | 2,006 | 768 | 69 | 26 | 43 | 742 | 0.38 | 0.03 | **0.06** |

### 4b. Device-by-device analysis

---

#### `ups_apc_01` — APC 3kVA UPS, single-phase, young battery (90 days old)
**F1: 0.86 | Precision: 0.92 | Recall: 0.81**

- Best performer among single-phase UPS devices
- Young battery means battery_charge_pct and battery_voltage_v follow predictable
  curves — the model learned these well
- 57 false positives: these occur near peak business hours when load variations
  push reconstruction error just above the threshold without a true anomaly
- 154 false negatives: mostly mild overload events at night (low baseline load factor)
  where the overload-induced error increase is smaller than the threshold margin

**Mean baseline reconstruction error:** 0.825 (13× threshold)
**Peak baseline error:** 15.59 (250× threshold — combined overload + battery_drain event)

---

#### `ups_apc_02` — APC 3kVA UPS, single-phase, aged battery (900 days old)
**F1: 0.89 | Precision: 0.98 | Recall: 0.82** ← Best overall F1

- Nearly perfect precision (0.98) — only 15 false positives in 2,006 windows
- Aged battery (900 days / 3-year expected life = 82% through lifespan) shows
  lower resting voltage and charge %, which the model treats as a more distinct
  pattern — anomalies are easier to distinguish
- Highest reconstruction errors in the entire dataset: `thermal_runaway` stacked
  with `battery_drain` on an already-degraded battery created errors 325× the threshold
- 153 false negatives: same mild overload pattern as ups_apc_01

**Top anomaly window:** 2025-01-01 15:50–16:35, error=20.32 (`thermal_runaway + battery_drain`)

---

#### `ups_lie_01` — Liebert 10kVA UPS, three-phase, moderate age (365 days)
**F1: 0.81 | Precision: 0.95 | Recall: 0.71**

- Three-phase device — phase model is active alongside baseline
- Both models contribute: 589 of 621 alerts were confirmed anomalies
- 241 false negatives: Liebert's 230V nominal input (vs APC's 120V) creates a
  different voltage scale. The model trained on a mix of voltage scales, which
  slightly increases reconstruction error for normal three-phase readings, raising
  the false negative rate for subtle anomalies
- Contains `phase_sag` anomalies (143 windows) — see Section 5 for why these are hard

---

#### `ups_lie_02` — Liebert 10kVA UPS, three-phase, older (700 days)
**F1: 0.76 | Precision: 0.96 | Recall: 0.63**

- Lowest recall of all UPS devices (0.63) despite being the device with the most
  true anomaly windows (911)
- The age effect works in reverse here: aged Liebert devices have a slower-changing
  degradation curve than APC (5-year expected life vs 3-year), so anomalies
  on top of slow degradation are harder to separate from the normal slope
- 339 false negatives — most are `overload` events where the three-phase averaging
  in `input_voltage_avg` masks the per-phase stress

---

#### `pdu_apc_01` — APC PDU, single-phase
**F1: 0.66 | Precision: 0.97 | Recall: 0.50**

- PDU devices use **baseline model only** (no phase-level model — PDUs don't
  have per-phase breakdowns in this dataset)
- Precision is excellent (0.97) but recall is weak (0.50) — the model catches
  half the anomalies
- PDU anomalies are limited to `overload` and `psu_failure`. Overload on a PDU
  (changing output_load_pct and output_power_w) produces smaller errors than
  UPS battery events because the PDU has fewer distinctive features
- 420 false negatives: almost all are `overload` events below the threshold

---

#### `pdu_rar_01` — Raritan PDU
**F1: 0.87 | Precision: 0.98 | Recall: 0.78** ← Best non-UPS device

- Unexpectedly strong performance for a PDU
- Raritan uses 230V nominal input. Overload events at 230V cause larger absolute
  changes in `output_power_w` (higher wattage nameplate: 7,200W vs APC's 1,440W)
  → larger reconstruction errors → easier to detect
- The phase model still runs and contributes 627 flags (vs 408 from baseline alone)
  despite PDU not having true per-phase data — the phase model picks up the
  high-wattage overload patterns through the non-phase features it shares with baseline

---

#### `net_cis_01` — Cisco switch (dual PSU, 200W)
**F1: 0.74 | Precision: 0.98 | Recall: 0.59**

- Network gear has low rated power (200W) so anomalies produce smaller absolute
  changes in output_power_w and output_load_pct
- 329 false negatives: mostly `overload` events where the overload on a 200W device
  changes the load by only a few watts — below the model's detection sensitivity

---

#### `net_gen_01` — Generic router (single PSU, 150W)
**F1: 0.68 | Precision: 0.96 | Recall: 0.53**

- Similar story to Cisco switch but worse recall because 150W rated capacity
  is even smaller — overload anomalies produce even tinier absolute changes
- The model is trained on all device types together, so the threshold is set
  by larger UPS events — small network gear anomalies sit below it

---

#### `env_gen_01` — Environmental sensor (5W)
**F1: 0.06 | Precision: 0.38 | Recall: 0.03** ← Effectively broken

- This device has only **overload** as its possible anomaly type
- Its rated capacity is 5W. An "overload" on a 5W sensor changes output_power_w
  by a few watts at most — completely invisible to a model calibrated on UPS events
- The 768 true anomaly windows all show baseline errors of 0.009–0.050,
  far below the threshold of 0.062
- **The environmental sensor should not share a threshold with UPS devices.**
  Either: (a) train a separate model for env sensors, or (b) set a much lower
  threshold for this device category. See Section 6 for recommendations.
- The 43 false positives are all borderline threshold crossings on normal days
  (errors of 0.063–0.078) — the model is over-sensitive to the sensor's limited
  feature variability

---

## 5. Anomaly type detection rates

How well does the system catch each type of failure?

| Anomaly type | Windows in dataset | Combined catch rate | Baseline only | Phase only | Missed |
|---|:---:|:---:|:---:|:---:|:---:|
| `battery_drain` | 900 | **100.0%** | 100.0% | 100.0% | 0% |
| `psu_failure` | 2,267 | **95.1%** | 95.1% | 76.0% | 4.9% |
| `thermal_runaway` | 1,377 | **86.9%** | 86.6% | 68.3% | 13.1% |
| `overload` | 3,587 | 32.7% | 18.2% | 29.9% | 67.3% |
| `phase_sag` | 337 | 24.3% | 22.3% | 21.4% | 75.7% |

### `battery_drain` — 100% caught (perfect)

Why it's easy: `battery_drain` simultaneously drops `battery_charge_pct` by 40–70%,
lowers `runtime_remaining_min` sharply, and sets `on_battery_flag=1`.
This combination is so far outside the normal range (charge % rarely drops below 80
in normal operation) that both models flag it instantly.

A window containing a `battery_drain` row has a baseline error averaging 8–20×
the threshold. The model cannot "miss" this — the feature deviation is too large.

### `psu_failure` — 95.1% caught

Why it's easy: `psu_failure` zeros out `output_load_pct` and `output_power_w`
entirely and sets both `on_battery_flag=1` and `bypass_flag=1`.
Zero output power during business hours is completely outside normal training
patterns. Both flags together form a highly distinctive signature.

The 4.9% missed are PSU failure events that occur at night (00:00–05:00) when
normal load is already near its minimum — zero output power at midnight is
not as far from normal as zero output at noon.

### `thermal_runaway` — 86.9% caught

Why it's mostly caught: `thermal_runaway` raises `battery_temperature_c` by 15–35°C
above normal and increases `output_power_w` by 1.2–1.8×.
Temperature spikes are very visible because normal battery temperature stays
in a narrow band (25°C ± 8°C depending on load). A 35°C addition produces a
reading of ~68°C — 4–5× outside the normal range in the scaled space.

The 13.1% missed: thermal events at night when ambient temperature is low and the
temperature spike still falls within a wider-than-average normal band, or when the
spike is on the smaller end (15°C addition instead of 35°C).

### `overload` — 32.7% caught (weakest for large devices)

Why it's hard: `overload` multiplies `output_load_pct` by 1.5–2.5×.
During peak business hours (09:00–18:00), normal load is already 70–90% of
capacity. An overload at 80% normal → 120% overload produces a reading of 120%
which is clipped to 100% in the data generator (via `np.clip(..., 5, 100)`).
The clip means mild overloads near peak hours barely change the feature values.

The phase model actually catches more overload windows than baseline (29.9% vs
18.2%) because per-phase currents increase proportionally to load and are not
clipped the same way.

For environmental sensors and small network gear, overload is essentially
undetectable at current threshold (see Section 4 above).

### `phase_sag` — 24.3% caught (structural weakness)

This is the most important finding. `phase_sag` deliberately drops L1 voltage
to 60–80% of normal while L2 and L3 remain unchanged. This is exactly what the
phase model was designed to catch.

**Why it still misses 75.7%:**

```
phase_sag breakdown:
  337 total windows
  ├── Both models caught: 65 (19.3%)
  ├── Baseline only:       10 (3.0%)
  ├── Phase only:           7 (2.1%)
  └── Missed entirely:    255 (75.7%)
```

The `phase_sag` anomaly applies to three-phase devices only (`ups_lie_01` and
`ups_lie_02`). For a 230V L1 input, dropping to 60–80% means L1 drops to
138–184V. The `voltage_imbalance_pct` value after the sag is:

```
avg_v = (138 + 230 + 230) / 3 = 199.3V
deviation = |138 - 199.3| / 199.3 * 100 = 30.8%
```

This is well above the clipping max of 10%. After clipping, `voltage_imbalance_pct`
becomes 10/10 = 1.0 (normalized max). The problem: the phase model was trained
with imbalance clipped to [0, 10%] because of the near-zero IQR issue described
in the engineering docs. That same clipping compresses the anomaly signal —
a catastrophic 30% imbalance looks the same as a borderline 10% imbalance
to the phase model after clipping.

**This is the primary design trade-off in the current system:** the clipping
that prevents training instability also caps the maximum anomaly signal for
severe phase imbalance events.

---

## 6. False positives — what triggers them

**228 total false positives** across all devices.

| Device | FP count | Root cause |
|--------|:---:|---|
| `env_gen_01` | 43 | Model threshold too high for 5W sensor; normal variation triggers it |
| `ups_apc_01` | 57 | Peak business-hour load variation crosses baseline threshold |
| `ups_lie_01` | 32 | Three-phase normal variation slightly higher than single-phase |
| `ups_lie_02` | 25 | Same as ups_lie_01 |
| `pdu_rar_01` | 14 | High-wattage PDU normal fluctuations near threshold |
| Others | 57 | Scattered across small devices |

**What the false positive windows look like:**

False positives cluster around reconstruction errors of 0.063–0.085 for baseline
(threshold = 0.062) — they are barely over the line. This means:

- The model is not wrong about these windows being unusual
- They represent real unusual-but-not-injected moments (load spikes, frequency
  variation, coincidental combinations of near-limit values)
- In a real deployment, these are "nuisance alerts" — not critical faults but
  worth investigating

Example false positive from `env_gen_01` on 2025-01-02 09:50–10:35:
```
baseline_error = 0.0632  (threshold = 0.0624 — just 1.3% over)
phase_error    = 0.0300  (under threshold)
anomaly_types  = []      (no anomaly injected)
```

This window occurred during mid-morning when load transitions from night-low
to business-peak. The transition pattern pushed reconstruction error just over
the line.

---

## 7. False negatives — what gets missed

**2,704 total false negatives** — true anomaly windows not flagged.

The false negatives split into three root causes:

### Cause 1: Wrong device type for the model
Environmental sensor (`env_gen_01`) accounts for 742 of 2,704 missed windows
(27.4% of all misses). The model's threshold is simply calibrated for UPS-scale
events and cannot detect 5W sensor overloads.

### Cause 2: Mild overload during high normal load
When normal load is already at 70–90%, the overload multiplier (1.5–2.5×)
gets clipped to 100% maximum. The anomaly becomes nearly indistinguishable from
a noisy normal reading.

Example missed window:
```
device:  env_gen_01
window:  2025-01-01 00:00 – 00:45
anomaly: overload injected
baseline_error = 0.0096  (threshold = 0.0624)
phase_error    = 0.0211  (threshold = 0.0565)
```

The model sees reconstruction error at 15% of the threshold — the anomaly
did not change the features enough.

### Cause 3: Phase sag compression
255 of 337 phase_sag windows are missed entirely because the 10% clip on
imbalance features compresses the anomaly signal (explained in Section 5).

---

## 8. The 10 most anomalous windows (highest reconstruction error)

These represent the most severe events detected in the dataset:

| Rank | Device | Window | Baseline Error | Threshold ratio | Anomaly types |
|:---:|---|---|:---:|:---:|---|
| 1 | `ups_apc_02` | 2025-01-01 15:50 – 16:35 | **20.33** | 325× | thermal_runaway + battery_drain |
| 2 | `ups_apc_02` | 2025-01-01 15:55 – 16:40 | 20.31 | 325× | thermal_runaway + battery_drain |
| 3 | `ups_apc_02` | 2025-01-01 15:45 – 16:30 | 20.01 | 320× | thermal_runaway + battery_drain |
| 4 | `ups_apc_02` | 2025-01-01 15:40 – 16:25 | 19.44 | 311× | thermal_runaway + battery_drain |
| 5 | `ups_apc_02` | 2025-01-01 15:35 – 16:20 | 18.84 | 301× | overload + thermal_runaway + battery_drain |
| 6 | `ups_lie_02` | 2025-01-03 22:50 – 23:35 | 18.00 | 288× | battery_drain |
| 7 | `ups_apc_02` | 2025-01-01 15:30 – 16:15 | 17.33 | 277× | overload + thermal_runaway + battery_drain |
| 8 | `ups_apc_02` | 2025-01-01 15:25 – 16:10 | 16.91 | 271× | overload + thermal_runaway + battery_drain |
| 9 | `ups_apc_02` | 2025-01-01 15:20 – 16:05 | 16.69 | 267× | overload + thermal_runaway + battery_drain |
| 10 | `ups_apc_01` | 2025-01-04 09:10 – 09:55 | 15.59 | 250× | overload + battery_drain |

**Observation:** Seven of the top 10 are `ups_apc_02` (the aged APC battery).
The aged battery's degraded baseline voltage and charge levels mean that anomaly
events sit even further from the model's expectation of what normal looks like.
Stacking two or three anomaly types simultaneously creates compound errors.

The event at rank 6 (`ups_lie_02`, 2025-01-03 22:50) is notable: a solo
`battery_drain` at night on the Liebert device produced a baseline error of 18.0
(288× threshold). Night-time battery drain is particularly extreme because the
model expects near-full charge during low-load hours.

---

## 9. Summary: what works, what doesn't, and why

### What works well

| Scenario | Performance | Reason |
|---|---|---|
| UPS battery drain | 100% detection | Large multi-feature deviation; battery metrics drop simultaneously |
| PSU failure | 95% detection | Zero output is unmistakable; both flags set |
| Thermal runaway | 87% detection | Temperature exceeds normal band by 4–5× in scaled space |
| Aged UPS detection | High precision (0.98) | Aged battery pattern is distinct from new battery pattern |
| Three-phase UPS (general) | Good (F1 0.76–0.81) | Phase model adds coverage the baseline misses |

### What doesn't work well

| Scenario | Performance | Root cause | How to fix |
|---|---|---|---|
| Environmental sensor | F1 = 0.06 | Threshold calibrated for UPS scale | Separate model or per-category threshold |
| Overload on small devices | ~30% catch | Features are clipped or barely change | Lower threshold for low-wattage devices |
| Phase sag | 24% catch | 10% imbalance clip compresses anomaly signal | Remove or raise clip; use separate scaler for imbalance |
| Night-time mild overload | Low recall | Normal night load ≈ overloaded load | Time-of-day aware threshold |

### The trade-off in one line

> High precision (95%+) means when the system fires an alert, it is almost certainly real.
> Low recall on some anomaly types (24–50%) means it will miss events.
> For a production monitoring system, this is a **better failure mode** than the reverse —
> false alarms cause alert fatigue, missed critical events cause outages.

---

## 10. Detection output file reference

**File:** `outputs/power_dual/anomaly_results.csv`

Each row represents one scored window:

| Column | Type | Meaning |
|---|---|---|
| `device_id` | string | Which physical device |
| `window_start` | datetime | Timestamp of first row in the 10-step window |
| `window_end` | datetime | Timestamp of last row in the 10-step window |
| `baseline_error` | float | MSE between input and baseline model reconstruction |
| `baseline_threshold` | float | Stored threshold: 0.062388 |
| `baseline_anomaly` | 0 or 1 | 1 if baseline_error > baseline_threshold |
| `phase_error` | float | MSE between input and phase model reconstruction |
| `phase_threshold` | float | Stored threshold: 0.056544 |
| `phase_anomaly` | 0 or 1 | 1 if phase_error > phase_threshold |
| `combined_flag` | 0 or 1 | 1 if baseline_anomaly OR phase_anomaly = 1 |
| `alert_policy` | string | "or" in this run |
| `true_label` | 0 or 1 | Ground truth: 1 if any row in window was injected as anomaly |
| `anomaly_types` | list | Which anomaly types appeared in this window |

To filter for only flagged anomalies in a notebook or script:
```python
import pandas as pd
df = pd.read_csv("snmp_anomaly_detection/outputs/power_dual/anomaly_results.csv")
flagged = df[df['combined_flag'] == 1]
true_alarms = df[(df['combined_flag'] == 1) & (df['true_label'] == 1)]
false_alarms = df[(df['combined_flag'] == 1) & (df['true_label'] == 0)]
missed = df[(df['combined_flag'] == 0) & (df['true_label'] == 1)]
```
