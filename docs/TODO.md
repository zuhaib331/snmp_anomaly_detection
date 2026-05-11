# SNMP Anomaly Detection — Task Backlog

<!-- Managed list — update Status field as work progresses. -->
<!-- Priority order: F9 → F10 → B2 → A1 → B3 → F11 → F12 → F13 → F14 → F15 → F5/F6 → E1 → E2 → E3 → E4 → E5 → E6 → E7 → C1 → E8 → D1/D2/D3/D4 -->
<!-- Done: F1, F2, F3, F4, F7, F8, F9, F10, B1, B2, B3, E1, E2, E3, E4, E5, E6, E7, F11 -->

<!-- ================================================================ -->
<!-- F-series: Code review bug fixes (must resolve before feature work) -->
<!-- ================================================================ -->

## F1 — Fix `TrainingConfig.threshold_std_multiplier` regression in network pipeline
**Status:** Done — 2026-04-29  
**Priority:** Critical — silent breakage of existing network pipeline threshold; no test to catch it  
**File:** [config.py:150](../snmp_anomaly_detection/config.py#L150)

### Issue
`TrainingConfig.threshold_std_multiplier` was changed from `3.0` to `2.0` in this branch. `TrainingConfig` is used by the **network** pipeline's `train_model.py`, not only the power pipeline. Lowering to 2σ raises false-positive rates on network anomaly detection without any network-side evaluation.

### Fix
Revert `TrainingConfig.threshold_std_multiplier` back to `3.0`. Power models already use `PowerTrainingConfig` which has its own independent `threshold_std_multiplier = 3.0`, so no power-pipeline change is needed.

---

## F2 — Fix validation scaler data leakage in `train_baseline_power.py`
**Status:** Done — 2026-04-29  
**Priority:** High — validation split is scaled with its own fitted scaler, not the training scaler  
**File:** [train_baseline_power.py:710](../snmp_anomaly_detection/training/train_baseline_power.py#L710)

### Issue
```python
scaled_val, _ = scale_features(val_df, paths, paths.power_outputs_dir / "baseline_val_scaler.pkl")
```
`scale_features` always calls `fit_transform`, so a new `RobustScaler` is fitted on the validation set instead of transforming with the training scaler. The saved `baseline_val_scaler.pkl` is never loaded anywhere (dead artifact). Validation loss therefore reflects a different feature space than inference.

### Fix
Return the fitted scaler from the training call and apply `.transform()` directly on val data:
```python
scaled_train, scaler = scale_features(train_df, paths)
val_feature_cols = [c for c in BASELINE_UPS_FEATURES if c in val_df.columns]
val_out = val_df.copy()
val_out[val_feature_cols] = scaler.transform(val_out[val_feature_cols])
```
Remove the `baseline_val_scaler.pkl` artifact path.

---

## F3 — Fix `preprocess-power` producing sequences without delta features
**Status:** Done — 2026-04-29  
**Priority:** High — `X_train.npy` from `preprocess-power` has 14 features; training uses 16  
**File:** [power_features.py:727](../snmp_anomaly_detection/preprocessing/power_features.py#L727)

### Issue
`run_power_feature_engineering` (invoked by `preprocess-power`) calls:
```
aggregate_phase_metrics → apply_log1p_skewed → filter_normal_rows → scale_features
```
It never calls `add_delta_features`, so the saved `X_train.npy` lacks `runtime_delta` and `battery_charge_delta`. Training scripts call `add_delta_features` independently and don't load `X_train.npy`, so models are trained correctly — but the `preprocess-power` output is silently wrong and misleading.

### Fix
Add `df = add_delta_features(df)` after `apply_log1p_skewed` in `run_power_feature_engineering`.

---

## F4 — Fix `derive_rul_labels` ignoring device install age
**Status:** Done — 2026-04-30 (code fix applied; **RUL model must be retrained** — run `python3 -m snmp_anomaly_detection train-battery-rul` to apply the label fix)  
**Priority:** High — RUL labels for new and end-of-life batteries are identical at dataset start  
**File:** [battery_features.py:52](../snmp_anomaly_detection/preprocessing/battery_features.py#L52)

### Issue
```python
age_days = i * _INTERVAL_MINUTES / 1440
rul = max(life_days - age_days, 0.0)
```
Age is always reset to 0 at dataset start for every device. `ups_apc_01` (90 days old) and `ups_apc_02` (900 days old, near end-of-life) receive identical RUL labels. The RUL model cannot learn to differentiate battery age from health signals — it only observes intra-dataset aging (~7 days), not the years of accumulated degradation.

### Fix
Persist `install_age_days` from `PowerDeviceProfile` as a column in the synthetic CSV during `build_power_dataset`. In `derive_rul_labels`, use it as the starting age offset:
```python
install_age = group["install_age_days"].iloc[0] if "install_age_days" in group.columns else 0.0
age_days = install_age + i * _INTERVAL_MINUTES / 1440
```

---

## F8 — Fix Battery RUL confidence intervals (MC Dropout CI coverage is 0%)
**Status:** Done — 2026-04-30 (dropout 0.1→0.3, mc_samples 30→50, retrained after F4 label fix; see F10 for remaining test-split issue)  
**Priority:** High — `rul_metrics.json` shows `ci_coverage_95pct: 0.0`; all 4 inference predictions cluster at ~173–175 days regardless of device state  
**File:** [train_battery_rul.py](../snmp_anomaly_detection/training/train_battery_rul.py), [rul_eval.py](../snmp_anomaly_detection/evaluation/rul_eval.py)

### Issue

Two separate problems compound each other:

**1. `dropout=0.1` is too small for MC Dropout to produce meaningful uncertainty.**  
MC Dropout requires a dropout rate of at least 0.2–0.3 to generate variance across forward passes. At 0.1, all samples from the same input collapse to nearly the same value — the resulting CI is ≈ ±0 wide relative to the true label spread (0–1459 days). This explains CI coverage of 0.0%.

**2. RUL model must be retrained after F4 before evaluating these metrics.**  
F4 fixed the RUL label generation (install age offset). The current `rul_metrics.json` and `rul_predictions.json` are from the pre-F4 model where all devices started aging from day 0, producing identical-looking labels that caused the model to learn the dataset mean (~173 days) rather than device-specific health trajectories.

### Fix

**Step 1 — Retrain (prerequisite):**
```bash
python3 -m snmp_anomaly_detection generate-power-data
python3 -m snmp_anomaly_detection train-battery-rul
```
Evaluate metrics. If `mae_days` drops below ~100 after retraining, F4 was the root cause and CI coverage may improve too.

**Step 2 — Raise dropout if CI coverage is still < 0.50 after retraining:**
In [config.py](../snmp_anomaly_detection/config.py), update `BatteryRULConfig`:
```python
dropout: float = 0.3   # was 0.1; MC Dropout requires ≥ 0.2 to produce useful variance
mc_samples: int = 50   # increase from default if fewer
```

**Step 3 — Verify inference sets `model.train()` for MC sampling:**
In `rul_eval.py`, confirm that the MC Dropout loop calls `model.train()` before sampling, not `model.eval()`. Calling `model.eval()` disables dropout, making all samples identical.
```python
model.train()   # must be train() not eval() for MC Dropout
with torch.no_grad():
    samples = torch.stack([model(x_tensor) for _ in range(mc_samples)])
```

### Validation
After retraining and fix:
- `mae_days` should be < 50 days  
- `ci_coverage_95pct` should be > 0.80  
- Predictions for a near-end-of-life device (install age > 1000 days) should be < 100 days; for a fresh device (install age < 30 days) should be > 1000 days

---

## F10 — Fix RUL test split: per-device temporal holdout instead of global last-15%
**Status:** Done — 2026-04-30  
**Priority:** Medium — global split dumps all sequences from one device into test set, producing misleading metrics; per-device predictions are already accurate (4/5 devices within 15 days)  
**File:** [train_battery_rul.py](../snmp_anomaly_detection/training/train_battery_rul.py)

### Issue

`build_rul_sequences` concatenates sequences from all UPS devices in `unique()` order. The global 70/15/15 split then puts the **last device's entire sequence** into the test set. After retraining (F8 + F4):

- Test set true RUL range: **1318–1323 days** (all from `ups_lie_03`, 5-day spread)
- Model predictions for that slice: **1152–1489 days** (model cannot distinguish a 500-day-old Liebert from a 365-day-old one in a 2-hour window)
- Result: test MAE = 144 days, CI coverage = 0.7% — both are artifacts of the split, not real model quality

Per-device predictions show the model is actually working: 4 of 5 devices are predicted within 15 days of ground truth. `ups_lie_03` is the outlier (off by ~155 days) because it looks nearly identical to fresher Liebert devices at the 5-min SNMP polling scale.

### Fix

In `train_battery_rul.py`, split **per device** before concatenating:

```python
train_seqs, val_seqs, test_seqs = [], [], []
train_lbls, val_lbls, test_lbls = [], [], []

for device_id in ups_df["device_id"].unique():
    dev = ups_df[ups_df["device_id"] == device_id]
    seqs, lbls = _build_device_sequences(dev, feature_cols, seq_len)
    n = len(seqs)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)
    train_seqs.append(seqs[:t_end]);  train_lbls.append(lbls[:t_end])
    val_seqs.append(seqs[t_end:v_end]); val_lbls.append(lbls[t_end:v_end])
    test_seqs.append(seqs[v_end:]);  test_lbls.append(lbls[v_end:])

x_train = np.concatenate(train_seqs); y_train = np.concatenate(train_lbls)
x_val   = np.concatenate(val_seqs);   y_val   = np.concatenate(val_lbls)
x_test  = np.concatenate(test_seqs);  y_test  = np.concatenate(test_lbls)
```

After this change the test set spans all 5 UPS devices' final windows (RUL from ~189 days to ~1454 days), giving a representative evaluation.

### Expected improvement
- Test MAE: 144 days → target < 50 days
- CI coverage: 0.7% → target > 0.80

---

## F9 — Capability registry for model routing (phase model + RUL gating)
**Status:** Done — 2026-04-30  
**Priority:** High — phase model runs on all 16 devices but produces `flagged_phase: 0` for every PDU/network/env window; wastes inference time and pollutes OR policy with dead signal. Hardcoding `if category == "ups"` in inference code does not scale when new device types are onboarded.  
**Files:** [config.py](../snmp_anomaly_detection/config.py), [dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py), [battery_features.py](../snmp_anomaly_detection/preprocessing/battery_features.py), [rul_eval.py](../snmp_anomaly_detection/evaluation/rul_eval.py)

### Issue

The phase model's `PHASE_LEVEL_FEATURES` include three-phase voltage columns (`input_voltage_l1/l2/l3`, `voltage_imbalance_pct`, `current_skew_pct`) that PDU/network/env devices do not have. The scorer currently runs the phase model on all device categories and fills missing features with 0. This produces reconstruction errors that are meaningless, yet consistently stays below the UPS-calibrated threshold — resulting in `flagged_phase: 0` for all non-UPS windows while still burning compute.

Confirmed from `detection_summary.json`: all 4 PDU, 3 network, and 1 env device show `flagged_phase: 0` across 100% of their windows.

**Why a hardcoded `if device_category == "ups":` is not enough:**
The same category-to-model coupling exists in three places today (`dual_model_scorer.py`, `battery_features.py`, `rul_eval.py`). Hardcoding the string in each file means every new device type (e.g. a 3-phase PDU or a generator) requires hunting down and updating multiple files. This will be missed.

### Fix

**Step 1 — Add a capability registry to `config.py` (single source of truth)**

```python
# Which device categories are eligible for each model.
# To onboard a new category: add it here only — no inference code changes needed.
PHASE_MODEL_CATEGORIES: frozenset[str] = frozenset({"ups"})
BATTERY_RUL_CATEGORIES: frozenset[str] = frozenset({"ups"})
BASELINE_MODEL_CATEGORIES: frozenset[str] = frozenset({"ups", "pdu", "network", "env"})
```

**Step 2 — Update `dual_model_scorer.py` to read from config**

```python
from snmp_anomaly_detection.config import PHASE_MODEL_CATEGORIES

if device_category in PHASE_MODEL_CATEGORIES:
    phase_error = _score_phase_window(window_features, phase_model, phase_scaler)
    phase_anomaly = int(phase_error > phase_threshold)
else:
    phase_error = 0.0
    phase_anomaly = 0
```

Set `phase_top_features` to `[]` and `phase_timestep_errors` to `[]` for non-phase-capable rows in the output CSV.

**Step 3 — Update `battery_features.py` and `rul_eval.py` to read from config**

Replace all `df["device_category"] == "ups"` filters with:
```python
from snmp_anomaly_detection.config import BATTERY_RUL_CATEGORIES
ups_df = df[df["device_category"].isin(BATTERY_RUL_CATEGORIES)].copy()
```

**Future path (ties into A1):** When device onboarding is built, replace these frozensets with capability flags on `PowerDeviceProfile` (`has_three_phase: bool`, `has_battery: bool`). The registry is the right abstraction for now; flags are the right long-term answer once A1 infrastructure exists.

### Validation
```bash
python3 -m snmp_anomaly_detection detect-power-csv
# Confirm: non-UPS rows have phase_error=0.0, phase_anomaly=0
# Confirm: UPS rows still have non-zero phase scores
# Confirm: inference wall-clock time drops (one forward pass instead of two for non-UPS)
# Confirm: adding "pdu" to PHASE_MODEL_CATEGORIES in config.py alone is sufficient
#          to enable phase scoring for PDUs — no inference file changes required
```

---

## F5 — Expose imbalance constants as public API in `phase_features.py`
**Status:** Superseded by F11  
**Priority:** Low — resolved as a side effect of F11: `IMBALANCE_COLS` and `IMBALANCE_CLIP_MAX` will be public constants in `scalar_transforms.py`, imported by all callers  
**Files:** [dual_model_scorer.py:194](../snmp_anomaly_detection/inference/dual_model_scorer.py#L194), [power_stream_processor.py:371](../snmp_anomaly_detection/inference/power_stream_processor.py#L371)

No separate action needed — close this when F11 is merged.

---

## F6 — Stop `process_event` from mutating its `PowerEvent` argument
**Status:** Superseded by F12  
**Priority:** Low — resolved as a side effect of F12: `EventPreprocessor.process()` copies `event.feature_values` before modifying it, so the caller's dict is never touched  
**File:** [power_stream_processor.py:498](../snmp_anomaly_detection/inference/power_stream_processor.py#L498)

No separate action needed — close this when F12 is merged.

---

## F11 — Extract `preprocessing/scalar_transforms.py`: single source of truth for all feature math
**Status:** Done — 2026-05-11  
**Priority:** High — root cause of the CSV/Kafka preprocessing gap; every future transport will diverge again without this foundation  
**Depends on:** Nothing  
**Files:** [preprocessing/power_features.py](../snmp_anomaly_detection/preprocessing/power_features.py), [preprocessing/phase_features.py](../snmp_anomaly_detection/preprocessing/phase_features.py)

### Why this task exists

The CSV inference path and Kafka inference path currently have **two separate implementations of the same preprocessing logic**. The Kafka path (`_apply_preprocessing` in `power_stream_processor.py`) was written by hand as a copy of the pandas functions, with comments like `# Mirrors apply_log1p_skewed from power_features.py`. That copy has already drifted in two places (F11 originally: missing `aggregate_phase_metrics`; F12 originally: missing `enrich_imbalance_features`). Any future transport (MQTT, ZMQ, HTTP push) would require a third copy.

This task extracts the core math into a transport-agnostic module so there is exactly one implementation that everything else calls.

### What to build

**New file: `preprocessing/scalar_transforms.py`**

Pure functions only — no pandas, no torch, only numpy. Each function takes a plain `dict` (feature values) and mutates it in-place (or takes additional arguments for stateful steps). This matches the streaming path's natural interface and is equally usable by a DataFrame row iteration.

```python
# preprocessing/scalar_transforms.py

import numpy as np

# ── Step 1: phase voltage aggregation ──────────────────────────────────────
_PHASE_V_COLS = ("input_voltage_l1", "input_voltage_l2", "input_voltage_l3")

def aggregate_phase_voltages(fv: dict) -> None:
    """Recompute input_voltage_v = mean(L1, L2, L3) when phase columns present.
    Must run before normalize_absolute so input_voltage_dev_pct uses the averaged value.
    Matches aggregate_phase_metrics() in power_features.py.
    """
    if all(c in fv for c in _PHASE_V_COLS):
        fv["input_voltage_v"] = sum(fv[c] for c in _PHASE_V_COLS) / 3.0

# ── Step 2: B2 vendor-agnostic normalization ────────────────────────────────
def normalize_absolute(
    fv: dict,
    rated_capacity_w: float,
    nominal_voltage_v: float,
    rated_battery_v: float,
) -> None:
    """Replace absolute V/A/W features with vendor-agnostic ratios/deviations.
    Matches normalize_absolute_features() in power_features.py.
    """
    cap  = max(rated_capacity_w, 1.0)
    nomv = max(nominal_voltage_v, 1.0)
    rated_current = cap / nomv
    fv["output_current_ratio"]   = fv.get("output_current_a", 0.0) / max(rated_current, 0.01)
    fv["input_voltage_dev_pct"]  = (fv.get("input_voltage_v",  nomv) - nomv) / nomv * 100.0
    fv["output_voltage_dev_pct"] = (fv.get("output_voltage_v", nomv) - nomv) / nomv * 100.0
    if rated_battery_v > 0:
        rated_bv = max(rated_battery_v, 1.0)
        fv["battery_voltage_ratio"] = fv.get("battery_voltage_v", 0.0) / rated_bv
        rated_bc = cap / rated_bv / 10.0
        fv["battery_current_ratio"] = fv.get("battery_current_a", 0.0) / max(rated_bc, 0.01)
    else:
        fv["battery_voltage_ratio"] = 0.0
        fv["battery_current_ratio"] = 0.0

# ── Step 3: log1p on skewed columns ────────────────────────────────────────
_LOG1P_COLS = ("runtime_remaining_min",)

def apply_log1p(fv: dict) -> None:
    """log1p transform on skewed columns. Must run before compute_deltas so
    runtime_delta is the diff of log1p(runtime), matching training order.
    """
    for col in _LOG1P_COLS:
        if col in fv:
            fv[col] = float(np.log1p(max(fv[col], 0.0)))

# ── Step 4a: signed-log1p delta features ───────────────────────────────────
_DELTA_PAIRS = (
    ("runtime_remaining_min",  "runtime_delta"),
    ("battery_temperature_c",  "temperature_delta"),
    ("output_load_pct",        "output_load_delta"),
)

def compute_deltas(fv: dict, prev: dict) -> None:
    """Per-device signed_log1p delta. prev is the feature dict from the previous event."""
    for src, tgt in _DELTA_PAIRS:
        cur = fv.get(src, 0.0)
        if prev:
            diff = cur - prev.get(src, cur)
            fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
        else:
            fv[tgt] = 0.0

# ── Step 4b: voltage drop delta features (B3) ──────────────────────────────
_VOLT_DROP_PAIRS = (
    ("input_voltage_l1", "voltage_drop_delta_l1"),
    ("input_voltage_l2", "voltage_drop_delta_l2"),
    ("input_voltage_l3", "voltage_drop_delta_l3"),
)

def compute_volt_drop_deltas(fv: dict, prev: dict) -> None:
    """Negative-only voltage diffs — only drops carry phase sag signal."""
    for src, tgt in _VOLT_DROP_PAIRS:
        cur = fv.get(src, 0.0)
        if prev:
            diff = min(cur - prev.get(src, cur), 0.0)
            fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
        else:
            fv[tgt] = 0.0

# ── Step 5: imbalance features ──────────────────────────────────────────────
_PHASE_V_COLS = ("input_voltage_l1", "input_voltage_l2", "input_voltage_l3")
_PHASE_I_COLS = ("input_current_l1", "input_current_l2", "input_current_l3")

def compute_imbalance(fv: dict) -> None:
    """Recompute voltage_imbalance_pct and current_skew_pct from raw phase columns
    using the NEMA MG-1 formula. Overwrites any pre-computed value from the collector
    so the formula is always identical to training.
    Matches enrich_imbalance_features() in phase_features.py.
    """
    if all(c in fv for c in _PHASE_V_COLS):
        vs  = [fv[c] for c in _PHASE_V_COLS]
        avg = sum(vs) / 3.0
        fv["voltage_imbalance_pct"] = (max(abs(v - avg) for v in vs) / max(avg, 1e-6)) * 100.0
    if all(c in fv for c in _PHASE_I_COLS):
        cs  = [fv[c] for c in _PHASE_I_COLS]
        avg = sum(cs) / 3.0
        fv["current_skew_pct"] = (max(abs(c - avg) for c in cs) / max(avg, 1e-6)) * 100.0

# Public constants used by EventPreprocessor and scalar wrappers in power_features.py
ALL_DELTA_SOURCES = tuple(s for s, _ in _DELTA_PAIRS) + tuple(s for s, _ in _VOLT_DROP_PAIRS)
IMBALANCE_CLIP_MAX: float = 10.0
IMBALANCE_COLS: tuple[str, ...] = ("voltage_imbalance_pct", "current_skew_pct")
```

### Update pandas wrappers to call scalar functions

`aggregate_phase_metrics`, `normalize_absolute_features`, `apply_log1p_skewed`, `add_delta_features` in `power_features.py` and `enrich_imbalance_features` in `phase_features.py` should each call the matching scalar function from `scalar_transforms.py` rather than containing their own implementation. This guarantees the math is identical — the pandas wrappers just apply the scalar function row-by-row or vectorise where safe.

The imbalance constants `_IMBALANCE_COLS` and `_IMBALANCE_CLIP_MAX` currently scattered across `phase_features.py`, `dual_model_scorer.py`, and `power_stream_processor.py` are replaced by `IMBALANCE_COLS` and `IMBALANCE_CLIP_MAX` from `scalar_transforms.py` — resolving **F5** as a side effect.

### Validation
```bash
python3 -m compileall snmp_anomaly_detection   # syntax check
python3 -m snmp_anomaly_detection detect-power-csv
# Confirm anomaly_results.csv matches pre-refactor output exactly (no behaviour change in this step)
```

---

## F12 — Create `inference/event_preprocessor.py`: shared preprocessing contract for all transports
**Status:** Done — 2026-05-11  
**Priority:** High — the interface that every transport (Kafka, CSV, MQTT, ZMQ, …) calls; without it each transport remains a hand-rolled copy  
**Depends on:** F11 (scalar_transforms.py must exist first)  
**Files:** [inference/event_preprocessor.py](../snmp_anomaly_detection/inference/event_preprocessor.py) ← new file

### What to build

A stateful class that takes one `PowerEvent` at a time and applies all preprocessing steps in the canonical order by calling functions from `scalar_transforms.py`. This is the single preprocessing contract — no transport writes feature-engineering logic directly.

```python
# inference/event_preprocessor.py
import dataclasses
from snmp_anomaly_detection.preprocessing import scalar_transforms as T

class EventPreprocessor:
    """Stateful per-device preprocessing for single-event inference.

    Maintains per-device previous-row state for delta features.
    Call process() once per event in arrival order per device.
    Call reset_device() when a device reconnects after a gap to avoid
    stale deltas from the previous session.
    """

    def __init__(self):
        self._prev: dict[str, dict] = {}   # device_id → last post-log1p feature values

    def process(self, event: PowerEvent) -> PowerEvent:
        fv = dict(event.feature_values)    # never mutate the caller's dict (fixes F6)

        T.aggregate_phase_voltages(fv)                      # step 1
        T.normalize_absolute(                               # step 2
            fv,
            event.rated_capacity_w,
            event.nominal_voltage_v,
            event.rated_battery_v,
        )
        T.apply_log1p(fv)                                   # step 3
        prev = self._prev.get(event.device_id, {})
        T.compute_deltas(fv, prev)                          # step 4a
        T.compute_volt_drop_deltas(fv, prev)                # step 4b
        self._prev[event.device_id] = {                     # persist for next event
            s: fv.get(s, 0.0) for s in T.ALL_DELTA_SOURCES
        }
        T.compute_imbalance(fv)                             # step 5

        return dataclasses.replace(event, feature_values=fv)

    def reset_device(self, device_id: str) -> None:
        """Clear per-device delta state. Call when a device reconnects."""
        self._prev.pop(device_id, None)
```

Because `process()` copies `event.feature_values` before modifying it, this also resolves **F6** (PowerEvent mutation) as a side effect.

### Verification property

Given the same sequence of raw rows for a device, `EventPreprocessor.process()` applied row-by-row must produce feature dicts identical to what `power_features.py` pandas functions produce when applied to the whole DataFrame. This can be verified with a simple test:

```python
# pseudo-test
preprocessor = EventPreprocessor()
for _, row in device_df.iterrows():
    event = PowerEvent.from_csv_row(row)
    processed = preprocessor.process(event)
    # processed.feature_values should match the corresponding row
    # in the pandas-preprocessed DataFrame for every feature in BASELINE_UPS_FEATURES
```

### Validation
```bash
python3 -m compileall snmp_anomaly_detection
# No behaviour change yet — EventPreprocessor exists but is not wired into any path.
# Verify by running detect-power-csv and confirm output is unchanged.
```

---

## F13 — Migrate both inference paths to `EventPreprocessor`; delete `_apply_preprocessing`
**Status:** Done — 2026-05-11  
**Priority:** High — closes the CSV/Kafka gap permanently; after this step adding a new transport costs zero preprocessing work  
**Depends on:** F12 (EventPreprocessor must exist first)  
**Files:** [inference/power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py), [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py)

### What to change

**`power_stream_processor.py`**

1. Remove `_apply_preprocessing()` entirely — all ~60 lines of hand-rolled feature math.
2. Remove the local constants that duplicated `scalar_transforms.py`: `_LOG1P_COLS`, `_DELTA_SOURCES`, `_DELTA_TARGETS`, `_VOLT_DROP_SOURCES`, `_VOLT_DROP_TARGETS`.
3. Construct one `EventPreprocessor` in `PowerStreamProcessor.__init__()`:
   ```python
   self._preprocessor = EventPreprocessor()
   ```
4. Replace the `self._apply_preprocessing(event)` call in `process_event()` with:
   ```python
   event = self._preprocessor.process(event)
   ```
5. Wire `reset_device()` into the existing device-disconnect/reset path so delta state is cleared on reconnect.

**`dual_model_scorer.py`**

1. Remove the four pandas preprocessing calls (lines 206–210): `aggregate_phase_metrics`, `normalize_absolute_features`, `apply_log1p_skewed`, `add_delta_features`, `enrich_imbalance_features`.
2. Construct an `EventPreprocessor` once before the device loop.
3. Convert each DataFrame row to a `PowerEvent`, call `preprocessor.process()`, then extract the feature vector — instead of bulk-transforming the entire DataFrame first.

The imbalance clipping block in `dual_model_scorer.py` (lines 231–234) is also deleted — it now lives inside `scalar_transforms.compute_imbalance()`.

### Why iterating rows is acceptable for CSV

`dual_model_scorer.py` already iterates per-device inside a Python loop. The vectorised pandas preprocessing was an optimisation for bulk transforms but introduced the divergence. For the dataset sizes in scope (≤50K rows), per-row Python is fast enough; the LSTM forward passes dominate wall-clock time by orders of magnitude.

### Validation
```bash
# Run full pipeline before and after — outputs must be byte-identical for CSV path
python3 -m snmp_anomaly_detection detect-power-csv
diff outputs/power_dual/anomaly_results_before.csv outputs/power_dual/anomaly_results.csv

# Run Kafka path with synthetic producer
python3 -m snmp_anomaly_detection detect-power-kafka
# Verify: for the same device row, Kafka and CSV now produce identical feature values
# (compare input_voltage_dev_pct, voltage_imbalance_pct, delta features per device)
```

---

## F14 — Extract shared `DualModelScorer` and `RollingWindowBuffer`; remove scoring duplication
**Status:** Done — 2026-05-11  
**Priority:** Medium — same scoring/windowing logic exists in both `dual_model_scorer.py` and `power_stream_processor.py`; a change to thresholding or alert policy requires two edits  
**Depends on:** F13 (inference paths must use EventPreprocessor before refactoring scoring)  
**Files:** [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py), [inference/power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py)

### Current duplication

| Logic | dual_model_scorer.py | power_stream_processor.py |
|---|---|---|
| LSTM window scoring + MSE exclusion | `_score_window_detailed()` | `_score_detailed()` |
| IForest scoring + ratio guard | inline in window loop | `_score_iforest_row()` |
| Overload rule | inline | inline |
| OR/AND alert policy | inline | inline |
| Per-category threshold lookup | inline | inline |

Any change to alert policy (E7 state machine, threshold tuning) currently requires editing both files. This is the same copy-problem as preprocessing.

### What to build

**`inference/scorer.py`** — a stateless scoring class that both transports inject:

```python
class DualModelScorer:
    def __init__(self, baseline_model, phase_model, iforest_models,
                 baseline_scaler, phase_scaler,
                 baseline_meta, phase_meta, iforest_meta,
                 config: InferenceConfig):
        ...

    def score_window(
        self,
        device_id: str,
        device_category: str,
        baseline_window: np.ndarray,     # (seq_len, n_baseline_features)
        phase_window: np.ndarray | None, # (seq_len, n_phase_features) or None
        iforest_row: np.ndarray,         # (n_baseline_features,) — current row
        raw_load_pct: float,
        timestamps: list[str],
    ) -> PowerScoringResult:
        ...
```

**`inference/rolling_buffer.py`** — extracted from `PowerStreamProcessor`, used by any transport that needs windowing:

```python
class RollingWindowBuffer:
    def push(self, device_id: str, vector: list[float]) -> np.ndarray | None:
        """Add one vector. Returns a (seq_len, n_feat) window when full, else None."""
```

### Result

Both `dual_model_scorer.py` (CSV transport) and `power_stream_processor.py` (Kafka transport) become thin wrappers: load models → construct `EventPreprocessor` + `DualModelScorer` → iterate events.

### Validation
```bash
python3 -m snmp_anomaly_detection detect-power-csv   # output unchanged
python3 -m snmp_anomaly_detection detect-power-kafka  # output unchanged
```

---

## F15 — Reduce transport files to thin adapters: wire format only, no logic
**Status:** Pending  
**Priority:** Medium — once F13 and F14 are done, the transport files should contain only deserialisation; this task removes the last remnants of duplicated logic  
**Depends on:** F14  
**Files:** [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py), [streaming/detect_power_kafka.py](../snmp_anomaly_detection/streaming/detect_power_kafka.py)

### Target shape of each transport file

```python
# streaming/detect_power_kafka.py  (after F15 — ~50 lines)
preprocessor = EventPreprocessor()
scorer       = DualModelScorer.from_artifacts(paths)
consumer     = KafkaConsumer(...)

for msg in consumer:
    event = PowerEvent.from_kafka(msg.value)          # deserialise
    event = preprocessor.process(event)               # preprocess
    result = scorer.score_window(...)                 # score
    write_output(result)                              # emit

# inference/csv_transport.py  (after F15 — ~50 lines, replaces dual_model_scorer.py main())
preprocessor = EventPreprocessor()
scorer       = DualModelScorer.from_artifacts(paths)

for row in pd.read_csv(path).itertuples():
    event = PowerEvent.from_csv_row(row)              # deserialise
    event = preprocessor.process(event)               # preprocess
    result = scorer.score_window(...)                 # score
    results.append(result)
```

Adding a new transport — MQTT, ZMQ, gRPC, HTTP webhook — requires only a `from_wire()` deserialiser and a consumer loop. No preprocessing logic, no scoring logic, no windowing logic. The transport is ~50 lines.

### Validation
```bash
# All three commands must produce identical results to pre-F15 baselines
python3 -m snmp_anomaly_detection detect-power-csv
python3 -m snmp_anomaly_detection detect-power-kafka
python3 -m compileall snmp_anomaly_detection
```

---

## F7 — Replace `battery_voltage_v > 0` UPS proxy with `device_category == "ups"`
**Status:** Done — 2026-04-30  
**Priority:** Low — voltage can be 0 during a fault, causing UPS to be silently excluded from RUL scoring  
**File:** [rul_eval.py:103](../snmp_anomaly_detection/evaluation/rul_eval.py#L103)

### Fix
```python
# Before
ups_df = df[df["battery_voltage_v"] > 0].copy()
# After
ups_df = df[df["device_category"] == "ups"].copy()
```

---

## B1 — Per-device online threshold in PowerStreamProcessor
**Status:** Done — 2026-04-29  
**Priority:** High — fixes current FPs, scales to millions of devices, no retraining required  
**Depends on:** Nothing (can start immediately)

### What to build
Add `DeviceErrorStats` dataclass to [power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py):

```python
@dataclass
class DeviceErrorStats:
    calibration_errors: list[float]  # collected during cold-start
    calibrated: bool                 # False until N windows seen
    rolling_mean: float
    rolling_var: float               # Welford online variance
    n_windows: int                   # total windows scored
```

### Behaviour
1. **Cold-start**: first `N_CALIBRATION_WINDOWS` (e.g. 50) windows per device collect errors silently — no alerts raised. Per-category threshold used as safety net for extreme values only (overload rule still active).
2. **Calibration complete**: `threshold[device] = mean(calibration_errors) + k × std(...)`
3. **Live**: each normal window (`error ≤ 2.5 × threshold`) updates rolling stats via Welford algorithm. Anomalous windows are locked out from stats update.
4. **Fallback chain**: device threshold → per-category threshold → global threshold.

### Also required
- Load `per_category_thresholds` from `baseline_metadata.json` into `PowerStreamProcessor` (already computed at training time, currently unused in the streaming path).
- Persist `DeviceErrorStats` to disk (JSON) so state survives consumer restarts.
- Log when a device transitions from calibration → live.

### Validation
```bash
# Smoke test — expect zero alerts
python3 -m snmp_anomaly_detection produce-power-kafka-test \
  --use-training-profiles --anomaly-probability 0.0

# Random fleet smoke test — also expect zero alerts after calibration window
python3 -m snmp_anomaly_detection produce-power-kafka-test \
  --anomaly-probability 0.0
```

---

## B2 — Make the model vendor-agnostic: replace absolute features with normalized features
**Status:** Done — 2026-05-04  
**Priority:** High — current `detection_summary.json` shows 3 devices with 100% false-positive rates and 8117+ total FP windows; root cause is a structural design flaw, not missing vendor profiles  
**Depends on:** B1 (done)

**Results after implementation:**
- Overall FP rate: 0.1% (target < 5%) ✅
- 230V device FP rate: 0.0–0.2% (was 100% before B2) ✅
- Recall: 88.3% (target > 60%) ✅
- Baseline UPS threshold: 0.338 (was 43.366 before removing battery_charge_delta) ✅
- RUL validation MAE: 8.83 days ✅

**Key fix during testing:** `battery_charge_delta` was removed from `BASELINE_UPS_FEATURES` (and `PHASE_LEVEL_FEATURES` by extension). Charge changes ~0.02%/step are noise after RobustScaler; the LSTM reconstructed it with MSE=2.37 on normal data, inflating the UPS threshold to 43. `battery_charge_pct` sequence gives the LSTM the trend implicitly.  
**MIB analysis confirmed (2026-04-30):** All three normalization values (`rated_capacity_w`, `nominal_voltage_v`, `rated_battery_v`) are auto-discoverable via SNMP for APC and Liebert GP devices. For RFC 1628-only devices, `rated_battery_v` has no OID and requires manual entry at onboarding. See A1 for the full OID resolution and unit conversion layer.

### Root cause (why adding more vendor profiles is the wrong fix)

The model inputs (`BASELINE_UPS_FEATURES`) do not include `vendor` or `device_id` — but they do include features whose **absolute values are determined by device physics**, which in turn depends on vendor-chosen capacity and voltage level:

| Feature in model | Why it is device-specific |
|---|---|
| `output_power_w` | APC 3kW UPS: 0–3000 W. Liebert 10kW UPS: 0–10000 W. Completely different ranges. |
| `output_current_a` | Derived from power ÷ voltage. Varies by both capacity and voltage standard (120V vs 230V). |
| `input_voltage_v` | 120 V for APC, 230 V for Liebert/Raritan — hardcoded in `dataset_builder.py:223`. |
| `output_voltage_v` | Same as above. |
| `battery_voltage_v` | Varies by battery string design: 12V, 48V, 240V depending on UPS model. |
| `battery_current_a` | Derived from battery voltage and capacity — device-specific. |

The `RobustScaler` is fitted on these absolute values from the training devices (APC 3kW at 120V and Liebert 10kW at 230V). When a new device with different capacity or voltage appears, its feature values land outside the scaler's learned range — reconstruction error is permanently above threshold even on perfectly normal windows. **This is why adding "more vendor profiles" is not the fix**: every new vendor would require retraining, and in production you cannot know vendor ahead of time.

The real fix is to **replace absolute-value features with ratio/deviation features that mean the same thing regardless of device size or voltage level**. Once features are normalized, a 120V APC 3kW and a 230V Eaton 10kW look the same to the model when both are healthy. The model becomes category-scoped (UPS vs PDU), not vendor-scoped.

### Step 1 — Add device registration fields to the CSV (dataset_builder.py)

Two values drive all normalization. They must be written to every CSV row so `power_features.py` can use them at preprocessing time. In [dataset_builder.py:415](../snmp_anomaly_detection/data/dataset_builder.py#L415), add to the `row` dict:

```python
"rated_capacity_w":   profile.rated_capacity_w,
"nominal_voltage_v":  230.0 if profile.vendor in ("liebert", "raritan") else 120.0,
"rated_battery_v":    profile.battery_ah * 12.0 if profile.battery_ah > 0 else 0.0,
```

In production these values come from MIB discovery or device registration in the entity management tool — they are entered once when the device is onboarded, not inferred from vendor.

### Step 2 — New function in power_features.py: `normalize_absolute_features()`

Add this function to [power_features.py](../snmp_anomaly_detection/preprocessing/power_features.py), called **before** `apply_log1p_skewed`:

```python
def normalize_absolute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace device-specific absolute values with device-agnostic ratios/deviations.

    All outputs are dimensionless and have the same meaning across vendors,
    voltage standards (120V / 230V), and capacity classes (1kW – 20kW).
    """
    df = df.copy()
    cap  = df["rated_capacity_w"].clip(lower=1.0)
    nomv = df["nominal_voltage_v"].clip(lower=1.0)

    # output_power_w → already captured by output_load_pct (0-100%); drop it
    df.drop(columns=["output_power_w"], errors="ignore", inplace=True)

    # output_current_a → ratio relative to rated current at nominal voltage
    rated_current = cap / nomv
    df["output_current_ratio"] = df["output_current_a"] / rated_current.clip(lower=0.01)

    # input_voltage_v, output_voltage_v → deviation from nominal in %
    df["input_voltage_dev_pct"]  = (df["input_voltage_v"]  - nomv) / nomv * 100.0
    df["output_voltage_dev_pct"] = (df["output_voltage_v"] - nomv) / nomv * 100.0

    # battery_voltage_v → ratio to rated battery string voltage (0 for non-UPS)
    rated_bv = df["rated_battery_v"].clip(lower=1.0)
    df["battery_voltage_ratio"] = df["battery_voltage_v"] / rated_bv
    df.loc[df["rated_battery_v"] == 0, "battery_voltage_ratio"] = 0.0

    # battery_current_a → ratio to rated discharge current (capacity / voltage / 10h rate)
    rated_bc = cap / rated_bv.clip(lower=1.0) / 10.0
    df["battery_current_ratio"] = df["battery_current_a"] / rated_bc.clip(lower=0.01)
    df.loc[df["rated_battery_v"] == 0, "battery_current_ratio"] = 0.0

    return df
```

Drop the registration columns before passing to the scaler — they are not model inputs:
```python
df.drop(columns=["rated_capacity_w", "nominal_voltage_v", "rated_battery_v"],
        errors="ignore", inplace=True)
```

### Step 3 — Update BASELINE_UPS_FEATURES in config.py

Replace the six absolute-value features with their normalized equivalents:

```python
BASELINE_UPS_FEATURES: tuple[str, ...] = (
    "battery_charge_pct",         # unchanged — already 0-100%
    "battery_voltage_ratio",      # replaces battery_voltage_v
    "battery_current_ratio",      # replaces battery_current_a
    "battery_temperature_c",      # unchanged — universal scale (°C)
    "runtime_remaining_min",      # unchanged — log1p applied; anomaly signal is the drop
    "on_battery_status",          # unchanged — binary
    "battery_replace_status",     # unchanged — binary
    "input_voltage_dev_pct",      # replaces input_voltage_v
    "input_frequency_hz",         # unchanged — 50/60 Hz, universal
    "output_voltage_dev_pct",     # replaces output_voltage_v
    "output_current_ratio",       # replaces output_current_a
    "output_load_pct",            # unchanged — already 0-100%
    "output_frequency_hz",        # unchanged — 50/60 Hz, universal
    # output_power_w is DROPPED — redundant with output_load_pct
    "runtime_delta",
    "battery_charge_delta",
    "temperature_delta",
    "output_load_delta",
)
```

`PHASE_LEVEL_FEATURES` keeps all phase columns unchanged — they are already expressed as voltages/currents whose anomaly signal is imbalance between phases (relative), and their deviation features derive from `input_voltage_dev_pct` computed above.

### Step 4 — Replace vendor-specific profiles with a capacity/voltage parameter sweep

The training dataset no longer needs vendor-named profiles. Replace `_DEFAULT_POWER_PROFILES` with a sweep across the parameter space that matters:

| Parameter | Values to cover |
|---|---|
| UPS capacity | 1 kW, 2 kW, 3 kW, 5 kW, 7.5 kW, 10 kW, 15 kW |
| Voltage standard | 120 V, 230 V |
| Phase count | 1-phase, 3-phase |
| Battery age | young (≤180 days), middle (180–720 days), aged (≥720 days) |
| PDU capacity | 1.4 kW, 3.6 kW, 7.2 kW, 14.4 kW |

This produces ~30–40 synthetic devices that cover the full production range without any vendor dependency. New vendors added to the entity management system will fall within this parameter space automatically.

Remove the vendor string from `PowerDeviceProfile` or keep it for logging only — it must not drive any data generation logic. The only inputs that should matter for normalization are `rated_capacity_w`, `nominal_voltage_v`, `phase_count`, and `battery_ah`.

### Step 5 — Update `_generate_normal_row` to use nominal_voltage_v, not vendor check

In [dataset_builder.py:223](../snmp_anomaly_detection/data/dataset_builder.py#L223), replace:
```python
# Before
nominal_v = 230.0 if profile.vendor in ("liebert", "raritan") else 120.0
```
with a field on `PowerDeviceProfile`:
```python
# After — nominal_voltage_v is an explicit field, not inferred from vendor
nominal_v = profile.nominal_voltage_v
```

Add `nominal_voltage_v: float = 120.0` to `PowerDeviceProfile`. Each profile in the sweep sets this explicitly.

### Retraining pipeline (run in order after all code changes)

```bash
python3 -m snmp_anomaly_detection generate-power-data
python3 -m snmp_anomaly_detection preprocess-power
python3 -m snmp_anomaly_detection train-power-baseline
python3 -m snmp_anomaly_detection train-power-phase
python3 -m snmp_anomaly_detection train-battery-rul
python3 -m snmp_anomaly_detection detect-power-csv
```

### Validation

```bash
# 1. Check normalized feature ranges — all should be in [-3, 3] for normal windows
python3 - <<'EOF'
import pandas as pd, numpy as np
df = pd.read_csv("snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv")
norm_cols = ["output_current_ratio","input_voltage_dev_pct","output_voltage_dev_pct",
             "battery_voltage_ratio","battery_current_ratio"]
print(df[df["anomaly"]==0][norm_cols].describe().round(3))
EOF

# 2. After detect-power-csv, precision should be >90% and FP rate <5% on normal windows
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("snmp_anomaly_detection/outputs/power_dual/anomaly_results.csv")
normal = df[df["true_label"]==0]
print("FP rate on normal windows:", normal["final_flag"].mean().round(3))
EOF
```

This also unblocks the `threshold_std_multiplier` recall improvement (currently 42.7% recall, `PowerTrainingConfig.threshold_std_multiplier=3.0`): lowering to 2.0 is the right call, but only evaluate it **after** FP rate is confirmed < 5% on normal windows from the new normalized model.

---

## A1 — SNMP OID Adapter Layer: vendor-agnostic OID resolution and unit normalization
**Status:** Pending  
**Priority:** High — without this layer, feature engineering (B2) cannot run against real devices; unit conversion bugs (RFC1628 ÷10 scaling, APC TimeTicks) will silently corrupt all features  
**Depends on:** B2 (feature schema must be finalised before OID mappings can be written)

### Why this task exists

The MIB analysis (2026-04-30) confirmed that the same physical measurement is exposed under **different OID paths and in different units** depending on the vendor and MIB in use. Without an explicit resolution and conversion layer, the pipeline will silently ingest wrong values. Examples:

| Measurement | RFC 1628 unit | APC unit | Liebert unit | Action needed |
|---|---|---|---|---|
| `battery_voltage_v` | **0.1 V** | Volts | Volts | RFC1628 path → ÷ 10 |
| `battery_current_a` | **0.1 A** | Amps | Amps | RFC1628 path → ÷ 10 |
| `input_frequency_hz` | **0.1 Hz** | Hz | Hz | RFC1628 path → ÷ 10 |
| `output_frequency_hz` | **0.1 Hz** | Hz | Hz | RFC1628 path → ÷ 10 |
| `output_current_a` | **0.1 A** | Amps | Amps | RFC1628 path → ÷ 10 |
| `runtime_remaining_min` | minutes | **TimeTicks (1/100 s)** | minutes | APC path → ÷ 6000 |
| `rated_capacity_w` | Watts | **kVA** | Watts / VA | APC path → × 1000 (VA), then × 0.9 ≈ W |

These are **silent corruptions** — SNMP returns a valid integer, but the magnitude is 10× wrong. There is no error to catch.

### What to build

**1. OID resolution config — `snmp_anomaly_detection/config/oid_map.yaml`**

One YAML file per feature, ordered by polling priority (try standard RFC 1628 first; fall back to vendor-proprietary if the standard OID returns 0 or error). Format:

```yaml
features:
  rated_capacity_w:
    - oid: "1.3.6.1.2.1.33.1.9.6"       # RFC1628 upsConfigOutputPower
      scale: 1.0
      unit: watts
    - oid: "1.3.6.1.4.1.318.1.1.1.4.2.6" # APC upsAdvOutputKVACapacity
      scale: 1000.0                        # kVA → VA (use as W approximation)
      unit: kva
    - oid: "1.3.6.1.4.1.476.1.42.3.5.7.6" # Liebert GP lgpPwrTopMaximumFrameCapacity
      scale: 1.0
      unit: va
    - oid: "1.3.6.1.4.1.476.1.1.1.1.1.9.4" # Liebert UPS lcUpsNominalOutputWatts
      scale: 1.0
      unit: watts

  nominal_voltage_v:
    - oid: "1.3.6.1.2.1.33.1.9.1"         # RFC1628 upsConfigInputVoltage
      scale: 1.0
    - oid: "1.3.6.1.4.1.318.1.1.1.3.2.7"  # APC upsAdvInputNominalVoltage
      scale: 1.0
    - oid: "1.3.6.1.4.1.476.1.1.1.1.1.9.2" # Liebert UPS lcUpsNominalInputVoltage
      scale: 1.0
    # Liebert GP: table walk required — handled in code, not YAML

  rated_battery_v:
    - oid: "1.3.6.1.4.1.318.1.1.1.2.2.7"   # APC upsAdvBatteryNominalVoltage ✓
      scale: 1.0
    - oid: "1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.1" # Liebert GP lgpPwrDcMeasurementPointNomVolts ✓
      scale: 1.0
    # RFC1628 and Liebert legacy UPS MIB: no OID → fallback = manual_entry

  battery_voltage_v:
    - oid: "1.3.6.1.2.1.33.1.2.5"           # RFC1628 upsBatteryVoltage — 0.1 V units
      scale: 0.1                              # ÷10 to get Volts
    - oid: "1.3.6.1.4.1.318.1.1.1.2.2.8"    # APC upsAdvBatteryActualVoltage — whole Volts
      scale: 1.0
    - oid: "1.3.6.1.4.1.318.1.1.1.2.3.2"    # APC hi-res upsHighPrecBatteryActualVoltage — 0.1 V
      scale: 0.1
    - oid: "1.3.6.1.4.1.476.1.1.1.1.1.2.3"  # Liebert UPS lcUpsBatVoltage — whole Volts
      scale: 1.0

  runtime_remaining_min:
    - oid: "1.3.6.1.2.1.33.1.2.3"            # RFC1628 — minutes
      scale: 1.0
    - oid: "1.3.6.1.4.1.318.1.1.1.2.2.3"     # APC — TimeTicks (1/100 s) → ÷6000
      scale: 0.000166667
    - oid: "1.3.6.1.4.1.476.1.1.1.1.1.2.1"   # Liebert UPS — minutes
      scale: 1.0
    - oid: "1.3.6.1.4.1.476.1.42.3.5.1.18"   # Liebert GP — minutes (65535 = unavailable)
      scale: 1.0

  # ... (full list for all 17 features follows the same pattern)
```

**2. `rated_battery_v` fallback strategy for RFC1628-only devices**

When no OID returns a usable value and no manual entry is provided, set `rated_battery_v = 0` and disable `battery_voltage_ratio` and `battery_current_ratio` from the feature vector for that device. The model still runs on the remaining 15 features. Log a warning at onboarding time.

```python
if device.rated_battery_v == 0:
    features["battery_voltage_ratio"] = 0.0
    features["battery_current_ratio"] = 0.0
    # these two columns are masked out; scaler was trained with 0.0 for non-UPS devices
```

**3. `on_battery_status` derivation (different polling pattern per vendor)**

| MIB | Method |
|---|---|
| RFC1628 | Poll `upsOutputSource` (1.3.6.1.2.1.33.1.4.1); value `5` = on battery |
| APC | Poll `upsBasicOutputStatus` (1.3.6.1.4.1.318.1.1.1.4.1.1); value `3` = on battery |
| Liebert GP | Poll `lgpPwrOutputToLoadOnInverter` (1.3.6.1.4.1.476.1.42.3.5.3.7); value `1` = yes |
| Liebert UPS | Walk `lcUpsAlarmTable`; check for `lcUpsAlarmOnBattery` descriptor |

**4. `battery_replace_status` derivation**

| MIB | Method |
|---|---|
| RFC1628 | Walk `upsAlarmTable` (1.3.6.1.2.1.33.1.6.2); flag set if `upsAlarmBatteryBad` OID present |
| APC | Poll `upsAdvBatteryReplaceIndicator` (.2.2.4); value `2` = needs replacing ← preferred, direct scalar |
| Liebert GP | Poll `lgpPwrBatteryCapacityStatus` (.1.26); value `3` (low) or `4` (depleted) = flag |
| Liebert UPS | Walk alarm table; check for `lcUpsAlarmBatteryBad` |

**5. Liebert GP two-step discovery**

The GP Power MIB uses a generic measurement-point table. Before polling live values, walk `lgpPwrMeasurementPointTable` once at onboarding to discover point IDs (input row vs. output row), then use those IDs as the first index `M` when polling `lgpPwrLineMeasurementTable` for per-phase readings.

### New features unlocked by the MIB analysis (add to BASELINE_UPS_FEATURES if universal coverage confirmed)

| Feature | OID source | Coverage | Value for anomaly detection |
|---|---|---|---|
| `bypass_flag` | RFC1628 `upsOutputSource`==4; APC status==6/9/10; Liebert GP `lgpPwrOutputToLoadOnBypass` | All vendors | Already exists as a column in synthetic data — now wire to real OIDs |
| `charger_fault_flag` | RFC1628 alarm `upsAlarmChargerFailed`; APC `upsAdvBatteryNumOfBadBattPacks`>0; Liebert GP `lgpPwrBatteryChargeStatus` fault enum | All vendors | Charger failure precedes battery drain by hours — strong early warning |
| `transfer_count_delta` | Liebert GP `lgpPwrTransferCount`; RFC1628 `upsInputLineBads` (cumulative counter) | Liebert GP direct; others indirect | Rate of input-to-battery transfers is a power quality degradation signal |
| `discharge_cycles` | Liebert GP `lgpPwrBatteryDischargeCount` | Liebert GP only | Cumulative discharge cycles — strongest available predictor for RUL model |

`discharge_cycles` from Liebert GP should be added to `BATTERY_RUL_FEATURES` if Liebert is a primary vendor. `bypass_flag` and `charger_fault_flag` should be added to `BASELINE_UPS_FEATURES` as they are universally available.

**THD (`lgpPwrLineMeasurementCurrentTHD`) is Liebert GP only — do not add to universal feature set. Mark as a Liebert-specific enrichment feature.**

### File to create
```
snmp_anomaly_detection/config/oid_map.yaml   ← new file
snmp_anomaly_detection/collection/oid_resolver.py  ← new file; loads YAML, tries OIDs in order, applies scale
```

---

## B3 — Phase sag recall and Liebert 3-phase accuracy
**Status:** Done — 2026-05-04  
**Priority:** Medium — improves phase model recall for voltage fault types; fixes Liebert-specific FPs  
**Depends on:** B2 (done) and F9 (done)

### Results after implementation
- UPS phase sag recall: **100%** (was ~25%) ✅
- Phase FP rate on normal windows: **0.1%** (no regression) ✅
- Overall FP rate: **0.1%** ✅
- `PHASE_LEVEL_FEATURES`: 29 → 32 features (3 new voltage drop delta columns)
- Liebert audit confirmed: L1/L2/L3 correctly at ~230V (mean=230.00V, std=1.01V, imbalance 0.01–1.72%) — no data generator change needed

### What was implemented

**Issue 1 — Phase sag recall:** Added `voltage_drop_delta_l1/l2/l3` features using Option A (preferred):
- [power_features.py](../snmp_anomaly_detection/preprocessing/power_features.py): added drop delta block inside `add_delta_features()` — clips per-device voltage diffs to `upper=0` (only drops produce signal) then applies signed_log1p
- [config.py](../snmp_anomaly_detection/config.py): added the 3 new columns to `PHASE_LEVEL_FEATURES`
- [power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py): added `_VOLT_DROP_SOURCES/_TARGETS` constants and matching computation in `_apply_preprocessing()` so live Kafka inference stays in sync with batch training

**Issue 2 — Liebert audit:** Confirmed clean. No separate fault label needed — Liebert phase sag variance is within normal model tolerance.

---

## C1 — ClickHouse storage integration
**Status:** Pending  
**Priority:** Medium — production readiness, replaces .jsonl flat file outputs  
**Depends on:** B1 and B2 complete (streaming path must be stable before wiring storage)

### Why
Replace `.jsonl` flat file outputs with a queryable, production-grade columnar store. Enables Grafana dashboards for live anomaly rates, RUL trends, and per-device heatmaps. Also stores raw telemetry for future retraining without manual dataset rebuilds.

### Steps
1. Stand up ClickHouse via Docker for dev:
   ```bash
   docker run -d --name clickhouse -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server
   ```
2. Create `snmp_raw` and `anomaly_windows` tables — MergeTree engine, partitioned by month, ordered by `(device_id, timestamp)`.
3. Set up Kafka Materialized View on `snmp-live-events` and `snmp-power-events` for automatic raw ingestion (no code change to producers).
4. Modify [detect_kafka.py](../snmp_anomaly_detection/streaming/detect_kafka.py) to write scored windows to ClickHouse via `clickhouse-connect` Python client.
5. Do the same for [detect_power_kafka.py](../snmp_anomaly_detection/streaming/detect_power_kafka.py) (power pipeline results).
6. Connect Grafana to ClickHouse — dashboards: anomaly rate per device, RUL trend, per-category heatmap.

### New dependency
```
clickhouse-connect>=0.7
```

---

<!-- ================================================================ -->
<!-- E-series: Isolation Forest second model                          -->
<!-- ================================================================ -->

## E1 — Isolation Forest as a real second detection model
**Status:** Done — 2026-05-05  
**Priority:** High — the NOC dashboard already shows an "Isolation Forest" panel as if it is live; currently no IF model exists anywhere in the pipeline  
**Depends on:** B1 and B2 complete (streaming path and training data should be stable before adding a third scoring signal)

### Why
The LSTM Autoencoder detects temporal anomalies well (gradual drifts, ramp-ups) but can miss point anomalies — sudden single-timestep spikes that look normal in a sequence context. Isolation Forest is a strong complement: it scores individual feature vectors in isolation, independent of sequence order, making it sensitive to exactly the fault types LSTM under-detects (instantaneous voltage spikes, sudden load jumps).

### What to build

**1. Training — new file `training/train_iforest_power.py`**

```python
from sklearn.ensemble import IsolationForest
import joblib

# Train one IF model per device category on normal (anomaly==0) rows only.
# Use BASELINE_UPS_FEATURES (flat, no sequence needed).
# contamination="auto" — revisit after smoke testing.
models: dict[str, IsolationForest] = {}
for category, group in df[df["anomaly"] == 0].groupby("device_category"):
    X = scaler.transform(group[BASELINE_UPS_FEATURES])
    clf = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    clf.fit(X)
    models[category] = clf

joblib.dump(models, paths.power_dual_outputs_dir / "iforest_models.pkl")
```

Save per-category score thresholds (1st percentile of training scores) to `iforest_metadata.json` alongside the `.pkl`.

**2. Artifact path — add to `config.py` `ProjectPaths`**

```python
iforest_model_file: Path = field(
    default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_models.pkl"
)
iforest_metadata_file: Path = field(
    default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_metadata.json"
)
```

**3. CLI command — add to `main.py`**

```python
"train-power-iforest": train_iforest_main,
```

**4. Scoring — extend `DualModelResult` in `dual_model_scorer.py`**

Add two fields to the dataclass:
```python
iforest_score: float    # raw anomaly score (more negative = more anomalous)
iforest_flag: int       # 1 if score < per-category 1st-percentile threshold
```

In `score_windows()`, load `iforest_models.pkl` once at startup. For each window, take the **last timestep's** feature vector (most recent observation), run `clf.score_samples(X)`, store the score, and threshold against `iforest_metadata.json`.

Update `final_flag`:
```python
final_flag = combined_flag OR iforest_flag OR overload_rule_flag
```

**5. Output schema — update `anomaly_results.csv` / `.jsonl`**

Add `iforest_score` and `iforest_flag` columns so D1 (dashboard chart) can plot the IF score line alongside the LSTM line.

### Artifact layout
```
outputs/power_dual/
  iforest_models.pkl        # dict[category → IsolationForest]
  iforest_metadata.json     # {"ups": {"threshold": -0.12, ...}, ...}
```

### Validation
```bash
# Train IF models (run after train-power-baseline)
python3 -m snmp_anomaly_detection train-power-iforest

# Smoke test — normal traffic, expect zero IF flags
python3 -m snmp_anomaly_detection detect-power-csv

# Verify schema and flag rate
python3 -c "
import pandas as pd
df = pd.read_csv('snmp_anomaly_detection/outputs/anomaly_results.csv')
print(df[['device_id','iforest_score','iforest_flag','baseline_anomaly','final_flag']].head(20))
print('IF flag rate:', df['iforest_flag'].mean())
"
```

Target: IF flag rate < 0.02 on normal traffic; ≥ 0.70 on injected anomaly rows.

---

## E2 — Per-category feature masks for Isolation Forest
**Status:** Done — 2026-05-05  
**Priority:** High — 117 of 118 `iforest_only` events are false positives; root cause is battery features (always 0.0 for PDU/network/env) and `output_frequency_hz` (grid noise) being fed to IF for all categories  
**Depends on:** E1 (done)  
**Files:** [training/train_iforest_power.py](../snmp_anomaly_detection/training/train_iforest_power.py), [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py)

### Issue

The LSTM already excludes `output_frequency_hz` from MSE for non-UPS categories (`_NON_UPS_MSE_EXCLUDE`) and never scores phase features on PDU/network/env. IF has no equivalent — it trains and scores on all 15 `BASELINE_UPS_FEATURES` for every category, including:

| Feature | Problem for non-UPS |
|---|---|
| `battery_voltage_ratio` | Always exactly 0.0 for PDU/network/env — adds noise to isolation trees |
| `battery_current_ratio` | Always exactly 0.0 — same |
| `on_battery_status` | Always 0 — same |
| `battery_replace_status` | Always 0 — same |
| `output_frequency_hz` | Grid noise (~50/60 Hz ± tiny variation) — already excluded from LSTM MSE for non-UPS |

These zero/noise features produce meaningless splits in the IF trees, shifting the score distribution and making the threshold unreliable.

### Fix

Add `IF_CATEGORY_FEATURES` to `config.py`:

```python
# Features to exclude from IF scoring per category.
# Battery features are always 0 for non-UPS — useless for isolation and shift score distribution.
_BATTERY_ONLY_FEATURES: frozenset[str] = frozenset({
    "battery_voltage_ratio", "battery_current_ratio",
    "on_battery_status", "battery_replace_status",
})

IF_EXCLUDED_FEATURES: dict[str, frozenset[str]] = {
    "ups":     frozenset(),
    "pdu":     _BATTERY_ONLY_FEATURES | {"output_frequency_hz"},
    "network": _BATTERY_ONLY_FEATURES | {"output_frequency_hz"},
    "env":     _BATTERY_ONLY_FEATURES | {"output_frequency_hz"},
}
```

In `train_iforest_power.py`, use per-category feature lists:
```python
from snmp_anomaly_detection.config import IF_EXCLUDED_FEATURES

excluded = IF_EXCLUDED_FEATURES.get(str(category), frozenset())
cat_features = [f for f in feature_cols if f not in excluded]
X = scaler.transform(group[cat_features])
# Save cat_features in metadata so scoring uses the same list
metadata[str(category)]["feature_cols"] = cat_features
```

In `dual_model_scorer.py`, load `feature_cols` from metadata per category and use it when calling `clf.score_samples()`.

### Validation
After retraining IF and re-running detection:
- `iforest_only` FP count should drop from 117 to < 10
- Recall on anomaly windows should improve (battery noise no longer dilutes anomaly signal)

---

## E3 — Score all window timesteps, take minimum (most anomalous point)
**Status:** Done — 2026-05-06  
**Priority:** High — current implementation scores only the last timestep; faults that peak mid-window are missed entirely, keeping IF recall at 3–16%  
**Depends on:** E2 (feature masks should be in place first so min-score is computed on clean features)  
**File:** [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py)

### Issue

```python
# Current — only last timestep
last_vec = baseline_vals[i + baseline_seq_len - 1 : i + baseline_seq_len]
if_score = float(clf.score_samples(last_vec)[0])
```

Most faults build up over several timesteps and peak in the middle of the window, not at the end. Scoring only the last timestep means IF misses the peak in those cases. For a 20-timestep window with a fault at timestep 8, the anomaly signal is completely ignored.

### Fix

Score all timesteps in the baseline window and take the **minimum** (most negative = most anomalous):

```python
# After — all timesteps, most anomalous wins
window_vecs = baseline_vals[i : i + baseline_seq_len]   # (seq_len, n_feat)
all_scores = clf.score_samples(window_vecs)              # (seq_len,)
if_score = float(all_scores.min())
if_peak_idx = int(np.argmin(all_scores))
if_peak_timestep = win_timestamps[if_peak_idx]
```

Store `if_peak_timestep` in `DualModelResult` and surface it in the `iforest` sub-dict of `anomaly_windows_detail.json` (same pattern as `baseline_peak_timestep`).

### Expected improvement
IF recall on UPS anomaly windows: 3.1% → target > 30%

---

## E4 — Raise IF score_ratio minimum to filter borderline flags
**Status:** Done — 2026-05-06  
**Priority:** Medium — 117 FP events have score_ratio 1.00–1.10; a minimum ratio guard removes these without retraining  
**Depends on:** E2 and E3 (implement feature masks and all-timestep scoring first; recalibrate threshold after)  
**File:** [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py)

### Issue

Current FP score ratios are tightly clustered at 1.00–1.10 (mean=1.02). A real anomaly should produce score ratios of 1.5+. The 1st-percentile threshold is a good starting point but is not conservative enough when training data is small (env: 1,769 samples, network: 3,673 samples).

### Fix

Add a minimum `score_ratio` guard in `dual_model_scorer.py`:

```python
IF_MIN_SCORE_RATIO: float = 1.15   # must be at least 15% below threshold to flag
```

```python
if_flag = int(if_score < if_threshold and
              (if_threshold != 0) and
              (if_score / if_threshold) >= IF_MIN_SCORE_RATIO)
```

Add `IF_MIN_SCORE_RATIO` to `config.py` so it can be tuned per deployment without code changes.

### Note
This is a tuning fix on top of E2+E3. Re-evaluate the right ratio value after E2 and E3 are implemented — the distribution of FP ratios will shift once battery features are excluded and all-timestep scoring is in place.

---

## E5 — IForest CSV performance: batch score per device, not per window
**Status:** Done — 2026-05-06  
**Priority:** Medium — `detect-power-csv` is noticeably slow after adding IForest; root cause is ~42,000 individual `score_samples()` calls instead of 21  
**Depends on:** E2 (feature masks should be in place first so batch call uses the right columns)  
**File:** [inference/dual_model_scorer.py](../snmp_anomaly_detection/inference/dual_model_scorer.py)

### Issue

The current scorer calls `clf.score_samples()` once per window, inside the stride-1 loop:

```python
# Called ~2000 times per device, ~42,000 times total
last_vec = baseline_vals[i + baseline_seq_len - 1 : i + baseline_seq_len]
if_score = float(clf.score_samples(last_vec)[0])
```

sklearn's `IsolationForest.score_samples()` has significant per-call overhead — it traverses all 200 trees for every call. With stride=1 and 21 devices averaging ~2000 rows each, this is **~42,000 × 200 = 8.4 million tree traversals**, all with individual Python function call overhead.

The LSTM calls are equally frequent but PyTorch operations are compiled and GPU-friendly. sklearn's IForest does not share that advantage.

### Fix

**Step 1 — Pre-score all rows for a device in one batch call before the window loop:**

```python
# Before the window loop, once per device
all_if_scores: np.ndarray | None = None
clf = iforest_models.get(device_category)
if clf is not None:
    cat_feature_cols = iforest_metadata.get(device_category, {}).get("feature_cols", baseline_cols)
    all_if_scores = clf.score_samples(baseline_vals[:, [baseline_cols.index(c) for c in cat_feature_cols]])
    # shape: (n_rows,) — one score per row, computed in a single vectorized call

# Inside the window loop, just index — no sklearn call
if all_if_scores is not None:
    if_score = float(all_if_scores[i + baseline_seq_len - 1])
    if_flag = int(if_score < if_threshold)
```

This reduces **~42,000 sklearn calls → 21 calls** (one per device). sklearn can also use its internal parallelism more effectively when scoring a batch.

**Step 2 (optional) — Reduce n_estimators from 200 → 100 in training:**

In [train_iforest_power.py:47](../snmp_anomaly_detection/training/train_iforest_power.py#L47):
```python
clf = IsolationForest(n_estimators=100, random_state=42, contamination="auto")
```

200 estimators is double the sklearn default. Halving it halves scoring time with negligible precision impact — especially relevant since the model's precision problems are not caused by too few trees.

### Expected improvement
`detect-power-csv` wall-clock time: should drop by 60–80% for the IForest scoring portion.

---

## E6 — IForest streaming: decouple from LSTM buffer, score on every incoming row
**Status:** Done — 2026-05-07  
**Priority:** Medium — in streaming mode IForest only fires when a full 20-row LSTM window is ready; it is unnecessarily blocked by the LSTM buffer requirement when it only needs 1 row  
**Depends on:** E2 (feature masks), E5 (batch scoring pattern understood)  
**File:** [inference/power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py)

### Issue

IForest scores individual feature vectors — it has no concept of sequences or time windows. But in the current implementation, IForest lives inside the LSTM window loop in `dual_model_scorer.py`. This means in streaming, IForest only runs when a device's rolling buffer has accumulated a full 20-row window and flushes to the LSTM.

The result:
- A sudden voltage spike at row 3 of a fresh device buffer will **not** be scored by IForest until 17 more rows arrive
- IForest runs at the LSTM's cadence (every `seq_len` rows), not at its natural cadence (every SNMP poll)
- This defeats the purpose of having a point-anomaly detector that is supposed to be faster than the LSTM

The two models have fundamentally different latency requirements:

| Model | Needs | Natural cadence |
|---|---|---|
| LSTM Autoencoder | 20-row sequence | Every `seq_len` rows (slow) |
| Isolation Forest | 1 row (current reading) | Every SNMP poll (fast) |

### Fix

In `PowerStreamProcessor.process_event()`, call IForest **immediately on the new row** before the buffer check, independently of the LSTM:

```python
def process_event(self, event: PowerEvent) -> ScoringResult | None:
    # Step 1 — IForest scores the current row immediately (no buffer needed)
    if_score = 0.0
    if_flag = 0
    clf = self.iforest_models.get(event.device_category)
    if clf is not None:
        raw = np.array([[event.feature_values.get(c, 0.0) for c in self.iforest_feature_cols]])
        scaled = self.baseline_scaler.transform(raw)
        if_score = float(clf.score_samples(scaled)[0])
        if_threshold = self.iforest_thresholds.get(event.device_category, -0.5)
        if_flag = int(if_score < if_threshold)

    # Step 2 — add row to rolling buffer
    self._buffers[event.device_id].append(event)

    # Step 3 — LSTM scores only when buffer is full
    if len(self._buffers[event.device_id]) < self.seq_len:
        # Buffer not ready — IForest result still emittable if flagged
        if if_flag:
            return ScoringResult(iforest_only=True, if_score=if_score, ...)
        return None

    # Step 4 — full window available: run LSTM, combine with IForest result
    window = list(self._buffers[event.device_id])
    lstm_result = self._score_lstm_window(window)
    return self._combine(lstm_result, if_score, if_flag)
```

### Why this matters for production

In a real datacenter, a sudden voltage transient (e.g. utility switching event) lasts 1–3 SNMP polls (~5–15 minutes at 5-min polling). The LSTM will not see this event until 20 polls later — 100 minutes after it happened. IForest should catch it immediately. Decoupling them means IForest can raise a fast early warning while the LSTM confirms or clears the alert when its window fills.

### Also required
- Load `iforest_models.pkl` and `iforest_metadata.json` in `PowerStreamProcessor.__init__()` (same pattern as baseline/phase model loading)
- Expose `iforest_feature_cols` per category from metadata (needed after E2 adds per-category feature masks)

---

## E7 — Two-stage alert lifecycle: SUSPECTED → CONFIRMED / CLEARED
**Status:** Done — 2026-05-07  
**Priority:** High — without this, E6's fast IF alerts are stateless; the NOC sees a stream of `iforest_only` flags with no way to know which ones LSTM later agreed with or retracted  
**Depends on:** E6 (IF must be decoupled from LSTM buffer first)  
**File:** [inference/power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py)

### Why this task exists

After E6, IF fires immediately on every incoming row and LSTM fires once the 20-row buffer fills. In steady state (post cold-start) both score on every row and their results are available simultaneously — combining them is straightforward. The problem is **temporal misalignment** in two scenarios:

1. **Cold start / device reconnect** — buffer has fewer than 20 rows; IF can flag but LSTM is silent. An `iforest_only` result is emitted with no follow-up.
2. **Steady state borderline flags** — IF fires at row N; LSTM scores the window ending at row N and may agree or disagree. Without state, each scoring call is independent and there is no "LSTM confirmed this earlier IF alert" signal.

The NOC needs to know the difference between:
- IF fired, LSTM confirmed → **high-confidence alarm** (act now)
- IF fired, LSTM cleared → **false positive** (auto-resolve, learning signal)
- IF fired, LSTM still pending → **early warning** (watch, don't page yet)

### What to build

**1. `AlertState` enum in `power_stream_processor.py`**

```python
from enum import Enum

class AlertState(str, Enum):
    SUSPECTED  = "suspected"   # IF fired, LSTM hasn't weighed in yet
    CONFIRMED  = "confirmed"   # both models agreed
    LSTM_ONLY  = "lstm_only"   # LSTM flagged, IF didn't (gradual drift)
    CLEARED    = "cleared"     # IF fired but LSTM did not confirm within window
```

**2. Per-device pending alert tracker in `PowerStreamProcessor`**

```python
@dataclass
class PendingAlert:
    if_score: float
    if_flag_row: int          # absolute row index when IF fired
    if_peak_timestep: str
    if_top_features: list[str]
    confirmed: bool = False
```

Add `_pending_alerts: dict[str, PendingAlert | None]` to `PowerStreamProcessor.__init__()`, keyed by `device_id`.

**3. State machine logic in `process_event()`**

```
IF fires at row N:
    → store PendingAlert(device_id, if_score, row=N)
    → emit ScoringResult(alert_state=SUSPECTED, ...)  # fast early warning

LSTM scores window ending at row N (or later):
    → if pending alert exists for device and if_flag_row is within this window:
        if lstm_flag:
            emit ScoringResult(alert_state=CONFIRMED, ...)   # upgrade
            clear pending alert
        else:
            emit ScoringResult(alert_state=CLEARED, ...)     # retract
            clear pending alert
    → if lstm_flag but no pending IF alert:
        emit ScoringResult(alert_state=LSTM_ONLY, ...)

Confirmation window timeout:
    if pending alert is older than N_CONFIRMATION_WINDOWS (e.g. 3) without LSTM verdict:
        emit CLEARED and discard
```

**4. `ScoringResult` schema update**

Add `alert_state: AlertState` field. Replace the boolean `iforest_only` flag (E6) with `alert_state` — `iforest_only=True` maps to `SUSPECTED`.

**5. Kafka output schema**

Add `"alert_state"` to the emitted JSON alongside existing fields. The NOC dashboard reads this field to set severity:

| `alert_state` | NOC display | Colour |
|---|---|---|
| `suspected` | Early warning | Amber |
| `confirmed` | High-confidence alarm | Red |
| `lstm_only` | Anomaly detected | Orange |
| `cleared`   | Auto-resolved | Grey |

### Key design notes

- In **steady state** (buffer already full), IF and LSTM score the same row simultaneously. `CONFIRMED` is emitted in the same `process_event()` call — zero extra latency vs. today.
- The `SUSPECTED` → `CONFIRMED/CLEARED` transition only adds latency during **cold start** (up to 20 polling cycles = ~100 min at 5-min polling). This is acceptable — you get an early warning immediately and confirmation shortly after.
- `CLEARED` events are valuable as a **false-positive feedback signal** for future IF threshold tuning (E4).
- `N_CONFIRMATION_WINDOWS = 3` (tunable in `config.py`) prevents orphaned `SUSPECTED` alerts from accumulating if a device stops sending data mid-window.

### Validation

```bash
# After E6 + E7 are implemented, run Kafka dry-run and check alert_state distribution
python3 -m snmp_anomaly_detection detect-power-kafka
# Expect: normal traffic → no SUSPECTED or CONFIRMED alerts after calibration
# Expect: injected spike → SUSPECTED at row N, CONFIRMED at row N+k (k ≤ seq_len)
# Expect: transient spike (1-2 rows) → SUSPECTED then CLEARED within one window
```

---

## E8 — Separate telemetry stream from alert lifecycle stream
**Status:** Pending  
**Priority:** Medium — do this before wiring a real NOC; not urgent while C1 and real devices are still missing  
**Depends on:** C1 (ClickHouse), D1 (NOC data wiring) — the two-stream split only becomes necessary when there are two real consumers  
**File:** [inference/power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py), [streaming/detect_power_kafka.py](../snmp_anomaly_detection/streaming/detect_power_kafka.py)

### Why this task exists

E7 added `alert_state` directly onto `PowerScoringResult`, which mixes two concerns into one stream:

1. **Telemetry** — every scored window, all error values and scores → should go to ClickHouse (C1) → Grafana dashboards
2. **Alert lifecycle** — state changes the NOC must act on → should go to NOC / PagerDuty

Symptoms of the current design that will cause problems in production:

| Problem | Effect |
|---------|--------|
| Normal windows carry `alert_state=LSTM_ONLY` | LSTM_ONLY is not an alert state — it means "no alert"; the field is misleading |
| CLEARED emitted for every IF-only steady-state fire | If IF fires 50 times without LSTM confirming, 50 CLEARED records are written — very noisy |
| Multiple SUSPECTED emitted per device during cold start | NOC sees 3 ambers, only 1 gets resolved; 2 stay open forever |
| Alert logic lives inside the scoring record | Forces every consumer to understand alert lifecycle instead of just displaying scores |

### What to build

**1. New `AlertEvent` dataclass (separate from `PowerScoringResult`)**

```python
@dataclass
class AlertEvent:
    alert_id: str           # stable UUID per device per incident (generated at OPENED)
    device_id: str
    device_category: str
    transition: str         # "opened" | "confirmed" | "closed"
    trigger: str            # "iforest" | "lstm" | "both" | "timeout"
    peak_if_score: float
    if_fire_count: int      # how many times IF fired before resolution
    opened_at: datetime
    resolved_at: datetime | None
    window_start: str
    window_end: str
```

**2. `PowerStreamProcessor.process_event()` returns a tuple**

```python
def process_event(self, event: PowerEvent) -> tuple[PowerScoringResult | None, AlertEvent | None]:
    ...
    return scoring_result, alert_event  # alert_event is None when no state change
```

`ScoringResult` has no `alert_state` field — it just has scores and flags.  
`AlertEvent` is only emitted when the lifecycle state actually changes (opened, confirmed, closed).

**3. Cold-start: suppress duplicate SUSPECTED events**

When IF fires multiple times during cold start (before the LSTM buffer fills):
- First fire: emit `AlertEvent(transition="opened")`, store `PendingAlert`
- Subsequent fires: update `PendingAlert.peak_if_score` and increment `PendingAlert.if_fire_count` silently — no new `AlertEvent`
- At LSTM resolution: emit `AlertEvent(transition="confirmed" or "closed")` with `if_fire_count` reflecting all fires

This gives the NOC: 1 OPENED → 1 CONFIRMED or CLOSED. Never orphaned ambers.

**4. `detect_power_kafka.py` routes by stream**

```python
scoring_result, alert_event = processor.process_event(event)

if scoring_result:
    # Write to ClickHouse telemetry table (all windows)
    write_to_clickhouse(scoring_result)

if alert_event:
    # Write to NOC / alert table (state changes only)
    write_alert_event(alert_event)
    print(f"[{alert_event.transition.upper()}] {alert_event.device_id} ...")
```

### When to implement

After C1 (ClickHouse storage) and D1 (NOC data wiring) are started — that is when two real consumers exist and the split becomes necessary rather than optional. Implementing E8 before C1 would mean redesigning the interface a second time.

---

<!-- ================================================================ -->
<!-- D-series: NOC Alarms Realtime dashboard (NOC Alarms Realtime.html) -->
<!-- ================================================================ -->

## D1 — Wire charts to real pipeline data
**Status:** Pending  
**Priority:** High — both chart panels are empty; the dashboard is effectively non-functional without them  
**Depends on:** C1 (ClickHouse) OR interim: poll `anomaly_results.csv` / `.jsonl` files directly

### What to fix
Two chart areas render blank in the current HTML:

1. **Anomaly Score Trend (24h)** — LSTM and IForest score lines over time. Should read from `snmp_anomaly_detection/outputs/anomaly_results.csv` (or `power_dual/anomaly_windows_detail.json`) and plot `reconstruction_error` (LSTM) and iforest score per timestamp.
2. **Alarm Trends (1H / 6H / 24H)** — alarm count bucketed by severity over time. Source: same output files, grouped by `anomaly_type` and binned into time buckets matching the selected range button.

Interim approach (before C1): use a `setInterval` fetch of the local `.jsonl`/`.csv` via a small Python Flask dev server, or convert outputs to a static JSON that the HTML polls.
Long-term: replace polling with a ClickHouse HTTP query (port 8123) once C1 is done.

---

## D2 — Fix AI Anomalies counter sync
**Status:** Pending  
**Priority:** High — UI shows inconsistent state: IForest panel says "3 Outliers Detected", Combined AI Alerts says "2 High Confidence", but the top-right "AI Anomalies" tile shows 0  
**Depends on:** Nothing (pure frontend bug)

### What to fix
The top summary tiles (All Alarms, Critical, Major, Minor, Warning, Cleared, AI Anomalies) are not reading from the same data source as the AI Anomaly Detection sub-panel. Ensure a single shared state object drives both. The "AI Anomalies" tile count should equal the number of rows where `dual_model_flag == 1` (or equivalent) in the active time window.

---

## D3 — Populate "Recent AI-Detected Anomalies" table
**Status:** Pending  
**Priority:** Medium — section renders empty; no anomaly rows visible  
**Depends on:** D1 (data source must be wired first)

### What to build
Read the latest N anomaly rows from the data source (`.jsonl` or ClickHouse) and render them in the table with columns:

| Column | Source field |
|---|---|
| Timestamp | `timestamp` |
| Device ID | `device_id` |
| Category | `device_category` |
| Anomaly Type | `anomaly_type` |
| LSTM Error | `reconstruction_error` |
| Phase Error | `phase_error` (if present) |
| Confidence | derived from dual-model OR policy |
| Status badge | Critical / Investigating / Action Required |

Support the "View All" link — either paginate in-page or open a filtered view.

---

## D4 — Implement unfinished UI actions
**Status:** Pending  
**Priority:** Low — buttons exist but are non-functional  
**Depends on:** D1 (need data before actions make sense)

### Items
| Control | What it should do |
|---|---|
| **Acknowledge** button | Mark selected alarm rows as acknowledged; persist state (localStorage or backend endpoint) |
| **Auto-Rules (0)** badge | Open a rule-builder panel to define suppress/escalate rules on alarm fields; count reflects active rules |
| **Download** icon | Export current filtered alarm table to CSV |
| **Sound Off** toggle | Play an alert tone (Web Audio API) when a new Critical/AI anomaly arrives; respect the toggle state |
| **Query Builder** | Parse structured filter expressions and apply them to the alarm list (e.g. `device_id:ups_apc_01 AND severity:critical`) |
| **Source / Impact toggles** | Already rendered — wire them to filter the alarm list by source and impact dimensions |
