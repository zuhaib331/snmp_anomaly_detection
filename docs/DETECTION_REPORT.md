# Dual-Model Anomaly Detection Report
**Branch:** `feature-power-snmp-detection`
**Dataset:** `synthetic_power_snmp_dataset.csv`
**Date generated:** 2026-04-27
**Alert policy:** OR (any of: baseline LSTM, phase LSTM, overload rule → alert)

---

## 1. What a "window" is

The detection scores **sliding windows** of 20 consecutive rows per device (20 rows × 5 min = 100-minute window).

```
Row 0  → Window 0:  rows  0–19   (00:00 – 01:35)
Row 1  → Window 1:  rows  1–20   (00:05 – 01:40)
Row 2  → Window 2:  rows  2–21   (00:10 – 01:45)
...
```

Windows overlap. One anomalous row at timestep T will appear in up to 20 windows.
A window is labeled **true_label=1** if **any** of its 20 rows was injected as anomalous.

---

## 2. How the three-signal scorer works

```
                       ┌──────────────────────────────────┐
                       │   For each 20-step window        │
                       └──────────────────────────────────┘
                                        │
               ┌────────────────────────┼─────────────────────────┐
               │                        │                          │
               ▼                        ▼                          ▼
  ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────────┐
  │  Baseline LSTM AE  │   │  Phase LSTM AE     │   │  Rule: load_pct>100 │
  │  18 features       │   │  29 features       │   │  (zero-FP overload) │
  │  per-category thr. │   │  per-category thr. │   │                     │
  └──────────┬─────────┘   └──────────┬─────────┘   └──────────┬──────────┘
             │                        │                          │
             └────────────────────────┴──────────────────────────┘
                                        │  OR
                                        ▼
                                   final_flag
```

**Per-category thresholds** — each device category has its own reconstruction error
baseline so categories with low natural variance (env, pdu) are not swamped by the
UPS-dominated global threshold:

| Category | Baseline threshold | Global fallback |
|---|---|---|
| `env` | 0.112 | 0.193 |
| `network` | 0.172 | 0.193 |
| `pdu` | 0.113 | 0.193 |
| `ups` | 0.219 | 0.193 |

**Rule-based overload supplement** — `output_load_pct > 100` is physically impossible
during normal operation (synthetic data caps normal at 100%). Any reading above 100
immediately sets `overload_rule_flag=1` with zero false positives.

---

## 3. Overall detection summary

| Metric | Value | Meaning |
|---|---|---|
| Total windows scored | **17,964** | 9 devices × 1,996 windows each (seq_len=20) |
| True anomaly windows | **11,575** (64.4%) | Windows containing at least one anomalous row |
| Flagged by LSTM (combined) | **7,491** | Either baseline or phase model above threshold |
| Flagged by overload rule | **3,786** | output_load_pct > 100 in any window timestep |
| **Flagged final (OR all)** | **9,511** (53.0%) | Any of the three signals fired |
| True positives (TP) | **9,436** | Correctly detected anomaly windows |
| False positives (FP) | **75** | Normal windows incorrectly flagged |
| False negatives (FN) | **2,139** | Missed anomaly windows |

---

## 4. Precision / Recall / F1

### Overall

| Metric | Value |
|---|---|
| **Precision** | **0.992** |
| **Recall** | **0.815** |
| **F1 Score** | **0.895** |
| **Accuracy** | **0.877** |

### Before vs after improvements (2026-04-27)

| Metric | Before | After | Delta |
|---|---|---|---|
| Precision | 0.988 | **0.992** | +0.004 |
| Recall | 0.445 | **0.815** | **+0.370** |
| F1 Score | 0.613 | **0.895** | **+0.282** |
| Accuracy | 0.772 | **0.877** | +0.105 |

---

## 5. Per-device breakdown

| Device | Category | Vendor | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| env_gen_01 | env | generic | 881 | 13 | 544 | 0.985 | 0.618 | 0.760 |
| net_cis_01 | network | cisco | 1131 | 0 | 154 | 1.000 | 0.880 | 0.936 |
| net_gen_01 | network | generic | 1199 | 0 | 134 | 1.000 | 0.899 | 0.947 |
| pdu_apc_01 | pdu | apc | 958 | 32 | 165 | 0.968 | 0.853 | 0.907 |
| pdu_rar_01 | pdu | raritan | 1044 | 30 | 290 | 0.972 | 0.783 | 0.867 |
| ups_apc_01 | ups | apc | 1148 | 0 | 147 | 1.000 | 0.886 | 0.940 |
| ups_apc_02 | ups | apc | 1175 | 0 | 158 | 1.000 | 0.881 | 0.937 |
| ups_lie_01 | ups | liebert | 901 | 0 | 339 | 1.000 | 0.727 | 0.842 |
| ups_lie_02 | ups | liebert | 999 | 0 | 208 | 1.000 | 0.828 | 0.906 |

---

## 6. Per anomaly type recall

| Anomaly type | Total windows | Detected | Missed | Recall |
|---|---|---|---|---|
| `battery_drain` | 1,274 | **1,274** | 0 | **1.000** |
| `psu_failure` | 3,393 | **3,393** | 0 | **1.000** |
| `thermal_runaway` | 1,164 | **1,103** | 61 | **0.948** |
| `overload` | 4,985 | **3,257** | 1,728 | **0.653** |
| `phase_sag` | 759 | **409** | 350 | **0.539** |

---

## 7. False positive rate

| Device | Normal windows | FP | FPR |
|---|---|---|---|
| env_gen_01 | 571 | 13 | 0.023 |
| net_cis_01 | 711 | 0 | 0.000 |
| net_gen_01 | 663 | 0 | 0.000 |
| pdu_apc_01 | 873 | 32 | 0.037 |
| pdu_rar_01 | 662 | 30 | 0.045 |
| ups_apc_01 | 701 | 0 | 0.000 |
| ups_apc_02 | 663 | 0 | 0.000 |
| ups_lie_01 | 756 | 0 | 0.000 |
| ups_lie_02 | 789 | 0 | 0.000 |

**Overall FPR: 75 / 6,389 = 1.2%**

---

## 8. What drove the improvements

### 1. New delta features (`temperature_delta`, `output_load_delta`)
Added to `BASELINE_UPS_FEATURES`. Both use signed-log1p compression to handle
extreme spikes. These give the LSTM explicit rate-of-change signals:

- `temperature_delta` rises 10× faster during `thermal_runaway` than normal
  → thermal_runaway recall: **0.386 → 0.948**
- `output_load_delta` spikes at overload onset, reinforcing the LSTM signal

### 2. Longer sequence window (10 → 20 steps)
Changed `seq_len` from 10 to 20 (50 min → 100 min context). Gradual anomalies
like `overload` and `thermal_runaway` now accumulate more reconstruction error
across the window before being scored.

### 3. Lower threshold multiplier (3.0 → 2.0)
`threshold_std_multiplier` dropped from 3.0 to 2.0. Pre-change FPR was 0.4%,
leaving room for more sensitive detection without unacceptable false alarm rates.

### 4. Per-category thresholds
After training, per-category reconstruction errors are computed on normal training
data and saved to `baseline_metadata.json` / `phase_metadata.json` as
`per_category_thresholds`. Inference uses the device's category threshold instead
of the global one. `env_gen_01` gets threshold 0.112 vs global 0.193.
→ env_gen_01 recall: **0.000 → 0.618** (completely blind before)

### 5. Rule-based overload supplement
`output_load_pct > 100` is physically impossible during normal operation.
The rule fires with zero false positives.
→ `overload` recall: **0.082 → 0.653**

---

## 9. Remaining gaps

| Gap | Root cause | Possible fix |
|---|---|---|
| `overload` recall 0.653 | Normal load variance (std=25%) overlaps with moderate overload (65–100% range). Rule only catches events above 100%. | Supervised overload classifier; or per-device adaptive load threshold |
| `phase_sag` recall 0.539 | Small voltage dips (2–5V) over short duration. Still below phase model threshold for many events. | Dedicated phase-sag detector using deviation from rolling mean voltage per phase |
| `ups_lie_01` recall 0.727 | 3-phase Liebert UPS — phase imbalance features near-zero during many faults | More Liebert-specific fault patterns in training data |
