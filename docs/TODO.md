# SNMP Anomaly Detection — Task Backlog

<!-- Managed list — update Status field as work progresses. -->
<!-- Priority order: F9 → F10 → B2 → A1 → B3 → F5/F6 → E1 → C1 → D1/D2/D3/D4 -->
<!-- Done: F1, F2, F3, F4, F7, F8, B1 -->

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
**Status:** Pending  
**Priority:** Low — private names (`_IMBALANCE_COLS`, `_IMBALANCE_CLIP_MAX`) used across two inference modules  
**Files:** [dual_model_scorer.py:194](../snmp_anomaly_detection/inference/dual_model_scorer.py#L194), [power_stream_processor.py:371](../snmp_anomaly_detection/inference/power_stream_processor.py#L371)

### Fix
Rename `_IMBALANCE_COLS` → `IMBALANCE_COLS` and `_IMBALANCE_CLIP_MAX` → `IMBALANCE_CLIP_MAX` in `phase_features.py`. Update both import sites.

---

## F6 — Stop `process_event` from mutating its `PowerEvent` argument
**Status:** Pending  
**Priority:** Low — in-place mutation of caller's dict is surprising; breaks replay/retry  
**File:** [power_stream_processor.py:498](../snmp_anomaly_detection/inference/power_stream_processor.py#L498)

### Issue
```python
event.feature_values[col] = (
    min(max(event.feature_values[col], 0.0), _IMBALANCE_CLIP_MAX)
    / _IMBALANCE_CLIP_MAX
)
```
The imbalance clipping overwrites the original event values. A caller that logs the raw event before scoring will see silently modified values.

### Fix
Apply the clipping transform to a local dict copy before building `p_raw`:
```python
clipped = dict(event.feature_values)
for col in IMBALANCE_COLS:
    if col in clipped:
        clipped[col] = min(max(clipped[col], 0.0), IMBALANCE_CLIP_MAX) / IMBALANCE_CLIP_MAX
p_raw = np.array([[clipped.get(c, 0.0) for c in PHASE_LEVEL_FEATURES]])
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
**Status:** Pending  
**Priority:** High — current `detection_summary.json` shows 3 devices with 100% false-positive rates and 8117+ total FP windows; root cause is a structural design flaw, not missing vendor profiles  
**Depends on:** B1 (done)  
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
**Status:** Pending  
**Priority:** Medium — improves phase model recall for voltage fault types; fixes Liebert-specific FPs  
**Depends on:** B2 (Liebert single-phase profiles must be in training data first) and F9 (phase model must be gated to UPS-only first)

### Current confirmed numbers (from `detection_summary.json`)
- Phase sag detection rate: **~25%** (confirmed by evaluation)
- Phase model fires on UPS devices only — all 4 PDU/network/env devices show `flagged_phase: 0` across every window, even though the scorer runs the phase model on them. This is expected once F9 is done (gating), but confirms the phase model provides zero value for non-UPS today.

### Issue 1 — Phase sag recall (voltage features under-weighted in global MSE)

The phase model uses a single scalar MSE threshold per device category. Phase sag faults only disturb `input_voltage_l1`, `voltage_imbalance_pct`, and `output_voltage_v` — a small subset of the 29 `PHASE_LEVEL_FEATURES`. Their contribution to MSE is diluted by the 26 unaffected features, causing missed detections.

Two options (pick one):

**Option A — `voltage_drop_delta` feature (preferred):** Add a per-phase rate-of-change feature that amplifies the sag signal before MSE:

```python
# In power_features.py, after log1p transforms:
df["voltage_drop_delta_l1"] = df["input_voltage_l1"].diff().clip(upper=0)  # negative = drop
df["voltage_drop_delta_l2"] = df["input_voltage_l2"].diff().clip(upper=0)
df["voltage_drop_delta_l3"] = df["input_voltage_l3"].diff().clip(upper=0)
```

Add the three new columns to `PHASE_LEVEL_FEATURES` in [config.py](../snmp_anomaly_detection/config.py). Retrain phase model only (baseline model is unaffected).

**Option B — per-feature MSE weights:** Apply a weight vector in `dual_model_scorer.py` that up-weights voltage columns before computing the scalar error. Lower implementation risk but harder to tune.

### Issue 2 — Liebert 3-phase `voltage_imbalance_pct` / `current_skew_pct` correctness

The imbalance/skew formulas in [dataset_builder.py:387-398](../snmp_anomaly_detection/data/dataset_builder.py#L387) are vendor-agnostic. Liebert devices report per-phase voltages under different OID paths (Emerson/Liebert MIB vs. APC POWERNET-MIB), so raw values may arrive in different units or scale. Two actions needed:

1. **Audit**: confirm that `input_voltage_l1/l2/l3` in the dataset correctly reflects 230V-nominal Liebert readings at normal load (should be ~228–232 V per phase, not ~120 V).
2. **Separate fault label (if warranted)**: if Liebert imbalance behaves differently under `phase_sag` (e.g. higher baseline variance), add a `liebert_phase_sag` anomaly type in `_anomaly_types_for_category` and a corresponding `_inject_anomaly` branch so the model sees realistic Liebert fault signatures during training.

### Validation
```bash
# After Option A retraining — check phase sag recall is ≥ 0.80
python3 -m snmp_anomaly_detection evaluate-power-baseline   # compare phase F1 before/after

# Liebert-only audit
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv")
lie = df[df["vendor"] == "liebert"]
print(lie[["input_voltage_l1","input_voltage_l2","input_voltage_l3","voltage_imbalance_pct"]].describe())
EOF
```

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
**Status:** Pending  
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
