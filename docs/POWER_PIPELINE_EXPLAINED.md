# Power SNMP Pipeline — Full Technical Explanation

This document explains every file, every design choice, and every number in the
`feature-power-snmp-detection` branch so you can confidently modify it yourself.

---

## Table of Contents

1. [The big picture — what was built and why](#1-the-big-picture)
2. [Project configuration — config.py](#2-project-configuration)
3. [Data generation — dataset_builder.py](#3-data-generation)
4. [Vendor OID mapping — vendor_oid_map.py](#4-vendor-oid-mapping)
5. [Feature engineering — power_features.py](#5-feature-engineering-baseline)
6. [Phase feature engineering — phase_features.py](#6-feature-engineering-phase-level)
7. [Battery RUL features — battery_features.py](#7-battery-rul-feature-engineering)
8. [Models — lstm_autoencoder.py and battery_rul.py](#8-the-models)
9. [Training — train_baseline_power.py](#9-training-baseline-model)
10. [Training — train_phase_model.py](#10-training-phase-model)
11. [Training — train_battery_rul.py](#11-training-battery-rul-model)
12. [Evaluation — time_split.py and power_eval.py](#12-evaluation)
13. [Detection — dual_model_scorer.py](#13-detection-dual-model-scorer)
14. [Streaming — power_stream_processor.py](#14-streaming-processor)
15. [Where to change things — modification guide](#15-where-to-change-things)

---

## 1. The big picture

### What existed before (master branch)

The master branch monitors generic network devices — routers, switches, firewalls.
It collects 5 metrics per device: `cpu`, `memory`, `in_octets`, `out_octets`, `errors`.
One LSTM Autoencoder learns what "normal" traffic looks like, and flags deviations.

### What this branch adds

Power equipment has completely different failure modes. A UPS doesn't have
`cpu` or `in_octets`. It has battery charge, voltage per phase, thermal state,
and runtime remaining. A failing battery looks totally different from a network spike.

So this branch trains **separate models, with separate features, for power equipment**.

### Three models, three jobs

```
┌─────────────────────────────────────────────────────────────┐
│                     POWER SNMP DATA                         │
│         (UPS, PDU, network gear, sensors)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
  ┌───────────────┐     ┌──────────────────┐
  │   Baseline    │     │   Phase-level    │
  │ LSTM AutoEnc  │     │  LSTM AutoEnc    │
  │  10 features  │     │  21 features     │
  │ (aggregated)  │     │ (per L1/L2/L3)   │
  └──────┬────────┘     └───────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    │
              ┌─────▼──────┐
              │ Dual scorer │  ← OR / AND policy
              └─────┬──────┘
                    │
                Anomaly alert

  ┌───────────────────────────┐
  │   Battery RUL LSTM Regr.  │  ← separate, independent model
  │        6 features          │
  │  output: days until repl.  │
  └───────────────────────────┘
```

- **Baseline model**: catches global UPS health problems (overload, battery drain, thermal)
- **Phase model**: catches asymmetric single-phase failures that average out in the baseline
- **RUL model**: predicts how many days before a battery needs replacing (forecasting, not detection)

---

## 2. Project configuration

**File:** `snmp_anomaly_detection/config.py`

This file is the single source of truth for feature lists, directory paths,
and hyperparameters. Every other file imports from here instead of hardcoding.

### Feature lists

```python
BASELINE_UPS_FEATURES = (
    "battery_charge_pct",      # 0–100%: state of charge
    "battery_voltage_v",       # volts: open-circuit voltage of battery pack
    "battery_temperature_c",   # celsius: battery cell temperature
    "runtime_remaining_min",   # minutes at current load until shutdown
    "input_voltage_avg",       # average of L1+L2+L3 input voltages
    "input_frequency_hz",      # AC supply frequency (normally 50 or 60 Hz)
    "output_load_pct",         # % of rated capacity currently being used
    "output_power_w",          # actual watts being drawn by the load
    "on_battery_flag",         # 1 = running on battery, 0 = on mains
    "bypass_flag",             # 1 = bypassed (maintenance mode or fault)
)
```

These 10 features go into the **baseline model**. They describe the UPS as a whole,
ignoring which individual phase is doing what.

```python
PHASE_LEVEL_FEATURES = BASELINE_UPS_FEATURES + (
    "input_voltage_l1",        # Line 1 input voltage
    "input_voltage_l2",        # Line 2 input voltage
    "input_voltage_l3",        # Line 3 input voltage
    "input_current_l1",        # Amps drawn on Line 1
    "input_current_l2",
    "input_current_l3",
    "output_current_l1",       # Amps being supplied on Line 1
    "output_current_l2",
    "output_current_l3",
    "voltage_imbalance_pct",   # NEMA formula: max deviation / avg × 100
    "current_skew_pct",        # same formula for currents
)
```

These 21 features go into the **phase model**. They include the per-phase breakdown
so the model can detect when one phase sags while the others are normal.

**IMPORTANT:** `PHASE_LEVEL_FEATURES` is a superset of `BASELINE_UPS_FEATURES`.
This caused a critical bug during development — more on that in Section 13.

```python
BATTERY_RUL_FEATURES = (
    "battery_charge_pct",
    "battery_voltage_v",
    "battery_temperature_c",
    "runtime_remaining_min",
    "charge_rate",             # % points per 5-minute interval (positive = charging)
    "discharge_cycles_approx", # estimated full cycles since installation
)
```

These 6 features go into the **RUL regression model**. They describe battery health trends.

### Paths

`ProjectPaths` is a frozen dataclass. All file locations are defined here.

| Property | Points to |
|----------|-----------|
| `power_dataset_file` | `data/synthetic_power_snmp_dataset.csv` |
| `power_outputs_dir` | `outputs/power/baseline/` |
| `power_phase_outputs_dir` | `outputs/power/phase/` |
| `battery_rul_outputs_dir` | `outputs/power/battery_rul/` |
| `power_dual_outputs_dir` | `outputs/power/dual/` |

**To change where files are saved:** edit `ProjectPaths` in `config.py`.

### Hyperparameters

`PowerTrainingConfig` controls training for all three models:

```python
batch_size = 64       # sequences per gradient step
epochs = 30           # training passes over the data
learning_rate = 1e-3  # Adam optimizer starting rate
hidden_size = 64      # LSTM hidden units
latent_size = 32      # bottleneck dimension (autoencoder only)
dropout = 0.1         # used in RUL model + MC Dropout at inference
threshold_std_multiplier = 3.0  # threshold = mean_error + 3 × std_error
```

`PowerInferenceConfig` controls detection behavior:

```python
alert_policy = "or"               # "or" = either model triggers = alert
                                  # "and" = both models must trigger = alert
rul_urgent_days = 14              # advisory = "urgent" below this
rul_warn_days = 30                # advisory = "warn" below this
compound_alert_window_minutes = 10  # UPS+PDU correlation window
```

---

## 3. Data generation

**File:** `snmp_anomaly_detection/power/data/dataset_builder.py`
**Command:** `python3 main.py generate-power-data`

Since real UPS/PDU SNMP data is not available, we generate synthetic data that
follows realistic physical patterns. The output is 18,144 rows across 9 devices.

### Device profiles

Nine devices are defined in `_DEFAULT_POWER_PROFILES`. Each is a `PowerDeviceProfile`:

```python
PowerDeviceProfile(
    device_id="ups_apc_01",
    device_category="ups",           # ups | pdu | network | env
    vendor="apc",
    phase_count=1,                   # 1 = single-phase, 3 = three-phase
    rated_capacity_w=3000.0,         # nameplate watts
    battery_ah=7.2,                  # amp-hours (0 for non-UPS)
    battery_expected_life_years=3.0,
    install_age_days=90.0,           # simulated age at dataset start
)
```

The 9 devices cover all four categories:
- `ups_apc_01` / `ups_apc_02` — single-phase APC UPS (young vs. aged battery)
- `ups_lie_01` / `ups_lie_02` — three-phase Liebert UPS
- `pdu_apc_01` / `pdu_rar_01` — PDUs (no battery, no phase-level model)
- `net_cis_01` / `net_gen_01` — network gear
- `env_gen_01` — environmental sensor

**To add a new device type:** add a new entry to `_DEFAULT_POWER_PROFILES` with your
device's real nameplate values. Then re-run `generate-power-data`.

### How each row is built

Every 5 minutes, for each device, `_build_power_row()` is called. It:

1. **Calculates load factor** using `_circadian_load_factor()` — simulates business-hours
   pattern. Load peaks around 09:00–18:00 and drops at night. Add random noise.

2. **Calculates battery state** (UPS only):
   - `charge_delta`: +0.02 if lightly loaded (charging), -0.01 if heavily loaded (discharging)
   - Applies `_battery_degradation_factor()` — older batteries hold less charge
   - Derives voltage from charge % using lead-acid open-circuit voltage formula
   - Estimates runtime from battery energy ÷ current load (85% inverter efficiency)

3. **Calculates phase voltages** using `_phase_voltages()`:
   - Single-phase: L1=L2=L3 (all the same with tiny noise)
   - Three-phase: each phase has independent ±1V noise around nominal

4. **Calculates per-phase currents** using P = V × I × PF (power factor = 0.95)

5. **Calculates imbalance** — NEMA formula: max deviation from average ÷ average × 100%

### Anomaly injection

With 5% probability per timestep, an anomaly is injected by `_inject_power_anomaly()`:

| Anomaly type | What it does | Which devices get it |
|---|---|---|
| `battery_drain` | Drops charge by 40–70%, sets on_battery_flag | UPS only |
| `overload` | Multiplies load by 1.5–2.5× | All |
| `phase_sag` | Drops L1 voltage to 60–80% of normal | Three-phase UPS only |
| `thermal_runaway` | Adds 15–35°C to battery temperature | UPS, network |
| `psu_failure` | Zeros output, sets on_battery + bypass flags | All |

Note: `phase_sag` on a single-phase device is replaced by `overload` (sag isn't
meaningful when all three phases are the same number).

### How to modify the data

| What to change | Where to change it |
|---|---|
| Number of days | `PowerDatasetConfig.total_points` (2016 = 7 days at 5-min intervals) |
| Anomaly rate | `PowerDatasetConfig.anomaly_probability` (0.05 = 5%) |
| Add a device | Append to `_DEFAULT_POWER_PROFILES` in `dataset_builder.py` |
| Change anomaly severity | Edit `_inject_power_anomaly()` multiplier values |
| Use real data | Replace `build_power_dataset()` with a CSV loader from your SNMP poller |

---

## 4. Vendor OID mapping

**File:** `snmp_anomaly_detection/power/data/vendor_oid_map.py`

SNMP OID names differ by vendor. APC calls battery charge
`upsAdvBatteryCapacity`. Liebert calls it `lgpPwrMeasurementOutputLoadPercent`.
RFC 1628 (the standard) calls it `upsBatteryCapacity`.

The OID map translates all of these to the same canonical name: `battery_charge_pct`.

```python
APC_OID_MAP = {
    "upsAdvBatteryCapacity":       "battery_charge_pct",
    "upsAdvBatteryActualVoltage":  "battery_voltage_v",
    "upsAdvBatteryTemperature":    "battery_temperature_c",
    ...
}
```

### How to use it

```python
from snmp_anomaly_detection.power.data.vendor_oid_map import resolve_to_canonical

canonical = resolve_to_canonical("apc", "upsAdvBatteryCapacity")
# returns: "battery_charge_pct"
```

Returns `None` for unmapped or internal fields (those starting with `_`).

### Status flag decoding

Some OIDs return integer codes that need decoding:

```python
from snmp_anomaly_detection.power.data.vendor_oid_map import decode_apc_status_flags

flags = decode_apc_status_flags(8)
# returns: {"on_battery": True, "low_battery": False, "replace_battery": False, ...}
```

**To add a new vendor:** add a new `VENDOR_OID_MAP` dict following the same pattern,
then add it to the `_ALL_MAPS` lookup inside `resolve_to_canonical()`.

---

## 5. Feature engineering — baseline

**File:** `snmp_anomaly_detection/power/preprocessing/power_features.py`
**Command:** `python3 main.py preprocess-power`

This file prepares data for the **baseline model** training.

### What it does step by step

**Step 1 — Load dataset**
```python
df = load_power_dataset(paths)
# reads synthetic_power_snmp_dataset.csv
# sorts by device_id, then timestamp
```

**Step 2 — Recompute input_voltage_avg**
```python
df = aggregate_phase_metrics(df)
# input_voltage_avg = mean(l1, l2, l3)
# This re-derives the aggregate from per-phase columns
```

**Step 3 — Log1p transform**
```python
df = apply_log1p_skewed(df)
# Applies log(1 + x) to runtime_remaining_min and output_power_w
```

Why log1p? These two columns are right-skewed — most values cluster near 0
or a small range, but anomalies can produce extreme spikes (runtime=999 or power=0).
Log1p compresses the scale so the model doesn't get dominated by those extremes.

**Step 4 — Filter normal rows only**
```python
train_df = filter_normal_rows(df)
# Keeps only rows where anomaly == 0
# LSTM Autoencoders must be trained on normal data only
```

Why train only on normal? An autoencoder learns to reconstruct what it was trained on.
If you train it on anomalies too, it learns to reconstruct those as well, and the
reconstruction error for anomalies won't be higher than for normal data.

**Step 5 — Scale with RobustScaler**
```python
scaled_df, scaler = scale_features(train_df, paths)
# Fits RobustScaler on the 10 BASELINE_UPS_FEATURES
# Saves scaler to outputs/power/baseline/baseline_scaler.pkl
```

RobustScaler uses median and IQR instead of mean and std. This makes it resistant
to outliers — a single temperature spike doesn't ruin the scaler for all other rows.

**Step 6 — Build sequences**
```python
sequences = build_baseline_sequences(scaled_df, seq_len=10)
# For each device, creates overlapping windows of 10 timesteps
# shape: (num_sequences, 10, 10) = (windows, time_steps, features)
```

Why sequences? LSTMs need time-ordered inputs. Instead of feeding one row at a time,
we feed a sliding window of 10 consecutive readings (= 50 minutes of data).

**Output:** `outputs/power/baseline/X_train.npy` — shape (N, 10, 10)

---

## 6. Feature engineering — phase level

**File:** `snmp_anomaly_detection/power/preprocessing/phase_features.py`

This prepares data for the **phase model**, which uses 21 features including
per-phase L1/L2/L3 readings.

### The imbalance clipping problem (critical)

**`voltage_imbalance_pct`** and **`current_skew_pct`** are near-zero for most
single-phase devices (all three "phases" are identical, so deviation = 0).

RobustScaler divides by IQR (interquartile range). When IQR ≈ 0 (because 99%
of values are 0%), dividing produces infinity or values like 10^14.

This caused the phase model training loss to explode to 3.4 × 10²⁵ in testing.

**The fix:**
```python
_IMBALANCE_CLIP_MAX = 10.0   # physical max: >5% is abnormal, >10% is fault

for col in ("voltage_imbalance_pct", "current_skew_pct"):
    df[col] = df[col].clip(0.0, 10.0) / 10.0
    # This puts imbalance in [0, 1] range before RobustScaler
```

By clipping to 10% (a physically meaningful fault ceiling) and dividing by 10,
we normalize imbalance to [0, 1] manually — RobustScaler then sees a reasonable
IQR and doesn't divide by near-zero.

### Imbalance calculation

```python
def compute_voltage_imbalance_pct(df):
    avg = mean(l1, l2, l3)
    max_deviation = max(|l1-avg|, |l2-avg|, |l3-avg|)
    return max_deviation / avg * 100
```

This is the NEMA MG-1 standard formula. It measures the worst single-phase
voltage deviation from the three-phase average, expressed as a percentage.
Values above 5% indicate a problem; above 10% is a fault condition.

### Gradient clipping

The phase model training also uses gradient clipping:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This caps the gradient magnitude at 1.0 during backpropagation. Without it,
even small numerical instabilities in the 21-feature space can cause weight
updates that are too large, destabilizing training.

The learning rate for the phase model is also set to `learning_rate × 0.1`
(10× smaller than baseline) for the same stability reason.

---

## 7. Battery RUL feature engineering

**File:** `snmp_anomaly_detection/power/preprocessing/battery_features.py`

This prepares data for the **RUL regression model**.

### What is RUL?

Remaining Useful Life = number of days until the battery needs replacement.
A new battery in a 3-year UPS starts at RUL = 1095 days (3 × 365).
As it ages, RUL counts down toward 0.

### How RUL labels are derived

For synthetic data, we don't have real degradation measurements. So we compute
RUL analytically from the device's age:

```python
def derive_rul_labels(df):
    for each device:
        life_days = vendor_expected_life * 365   # apc=3yr, liebert=5yr
        for row i in device:
            age_at_row = i * (5 minutes / 1440 min_per_day)
            rul = max(life_days - age_at_row, 0)
```

This is a simple countdown. In real deployment, you'd replace this with
actual capacity measurements from a battery management system.

### Sequence length

RUL uses `seq_len=24` (vs 10 for anomaly detection). That's 24 × 5 min = 2 hours.
The model needs a longer lookback to detect trends (gradual voltage drop,
increasing temperature) rather than sudden spikes.

### Why RUL accuracy is poor on synthetic data

The synthetic dataset covers 7 days. Within a 3-year battery lifespan,
7 days is an almost flat line — RUL changes by at most 7 days across the
entire dataset. The model has almost no slope to learn from.

Real RUL prediction needs months to years of data showing genuine degradation.

---

## 8. The models

### LSTM Autoencoder (`models/lstm_autoencoder.py`)

This model is inherited unchanged from master. It works for any number of features
because `input_size` is passed as a constructor argument.

**Architecture:**
```
Input: (batch, seq_len, input_size)
  → LSTM Encoder: hidden_size units
    → Repeat last hidden state for seq_len steps
      → LSTM Decoder: hidden_size units
        → Linear layer: hidden_size → input_size
Output: (batch, seq_len, input_size)  ← reconstruction
```

**How anomaly detection works:**
1. Train the model on normal-only sequences
2. At inference, feed a new sequence
3. Compare output (reconstruction) to input (original)
4. Compute mean squared error between input and output
5. If error > threshold → anomaly

The idea: a model trained only on normal patterns can reconstruct normal data
well but struggles to reconstruct anomalous patterns → high reconstruction error.

**Threshold:**
```python
threshold = mean(train_errors) + 3 × std(train_errors)
```

This is set so that ~99.7% of normal sequences fall below the threshold
(if errors were normally distributed). The `threshold_std_multiplier=3.0`
in config controls this. Increase it to reduce false positives; decrease
it to catch more anomalies (at the cost of more false alarms).

### Battery RUL Model (`models/battery_rul.py`)

**Architecture:**
```
Input: (batch, seq_len=24, input_size=6)
  → LSTM: 2 layers, hidden_size=64
    → take last hidden state h_n[-1]: (batch, 64)
      → Dropout(0.1)
        → Linear(64 → 1): scalar RUL prediction
Output: (batch,) — predicted days until replacement
```

This is a regression model, not an autoencoder. It predicts a number (days),
not a reconstruction.

**MC Dropout:**
During inference, dropout is kept active (by calling `model.train()` instead
of `model.eval()`). Running 30 forward passes with different dropout masks
gives 30 slightly different predictions. The spread of those predictions gives
a confidence interval.

```python
predictions = [model(x) for _ in range(30)]  # 30 stochastic passes
mean = average(predictions)
std = stdev(predictions)
confidence_interval = [mean - 2×std, mean + 2×std]  # ~95%
```

---

## 9. Training — baseline model

**File:** `snmp_anomaly_detection/power/training/train_baseline_power.py`
**Command:** `python3 main.py train-power-baseline`

### Training loop

```
Load dataset → preprocess → time-based split → scale → build sequences
                                                           ↓
                                                    LSTMAutoencoder
                                                    Adam optimizer
                                                    MSE loss
                                                    30 epochs
                                                           ↓
                                               compute threshold from train errors
                                                           ↓
                                               save model.pt + metadata.json
```

The time-based split is done **after** loading, **before** scaling. This is
intentional — the scaler is fitted only on training data. If you fit it on
the full dataset, information from future (test) data would leak into the
training distribution.

### Threshold setting

```python
train_errors = compute reconstruction errors on all training sequences
threshold = mean(train_errors) + 3 × std(train_errors)
```

**To change threshold sensitivity:**
- Increase `threshold_std_multiplier` in `PowerTrainingConfig` → fewer alerts, lower recall
- Decrease it → more alerts, higher recall
- Alternatively, run `evaluate-power-baseline` which shows how P/R/F1 change at the current threshold

### Output files

| File | Contains |
|---|---|
| `outputs/power/baseline/baseline_model.pt` | PyTorch model weights |
| `outputs/power/baseline/baseline_metadata.json` | input_size, threshold, training history |
| `outputs/power/baseline/baseline_scaler.pkl` | fitted RobustScaler (used at inference) |

---

## 10. Training — phase model

**File:** `snmp_anomaly_detection/power/training/train_phase_model.py`
**Command:** `python3 main.py train-power-phase`

Same structure as baseline training with two differences:

1. **Features**: uses 21 `PHASE_LEVEL_FEATURES` instead of 10 `BASELINE_UPS_FEATURES`
2. **Training stability**:
   - Learning rate is `config.learning_rate × 0.1` (10× smaller)
   - Gradient clipping: `max_norm=1.0` after each backward pass

These stability measures were added because the imbalance features, even after
clipping, still create a higher-dimensional space with more potential for
gradient instability than the simpler baseline feature set.

---

## 11. Training — battery RUL model

**File:** `snmp_anomaly_detection/power/training/train_battery_rul.py`
**Command:** `python3 main.py train-battery-rul`

```
Load dataset → log1p → derive RUL labels → scale features → build sequences + labels
                                                                      ↓
                                                          BatteryRULModel
                                                          MSE loss (regression)
                                                          30 epochs
                                                                      ↓
                                                        save model.pt + metadata.json
                                                        save X_test_rul.npy / y_test_rul.npy
```

Key difference from anomaly models: this is **supervised regression**. Each sequence
has a corresponding RUL label (a number of days). The model minimizes mean squared
error between predicted days and actual days.

The dataset is split 70/15/15 on the raw sequence index (already chronological).
Only UPS devices (those with `battery_voltage_v > 0`) are used — PDUs and network
gear don't have batteries.

---

## 12. Evaluation

### Time-based split (`evaluation/time_split.py`)

```python
def time_based_split(df, train_frac=0.70, val_frac=0.15):
    sorted_df = df.sort_by(timestamp)
    # First 70% → training
    # Next 15% → validation
    # Last 15% → test
```

**Why time-based?** If you shuffle rows randomly, your training set contains
readings from the same day as your test set. The model can learn from "future"
patterns during training, making test accuracy artificially high.

With time-based split: training uses earliest data, testing uses latest data —
same as real deployment where you train on historical data and predict on new data.

### Baseline evaluation (`evaluation/power_eval.py`)

**Command:** `python3 main.py evaluate-power-baseline`

1. Loads baseline model + scaler
2. Takes the held-out test split (last 15% of data chronologically)
3. Applies the trained scaler to transform test features
4. Builds sequences from the test data (including anomaly rows this time)
5. Scores each sequence — reconstruction error vs threshold
6. Computes Precision, Recall, F1

**Window-level labeling:** a sequence window is labeled anomalous if **any** of its
10 timesteps contains an anomaly. So if one row in a 10-row window is flagged,
the whole window gets label=1.

**Current results:**
```
Precision: 0.956  (when it flags an anomaly, it's right 95.6% of the time)
Recall:    0.547  (it catches 54.7% of all actual anomalies)
F1:        0.696
```

The low recall is expected: the baseline model misses anomalies that are subtle
at the aggregate level (e.g., a phase_sag that barely changes the average voltage).
The phase model catches those.

### RUL evaluation (`evaluation/rul_eval.py`)

**Command:** `python3 main.py predict-battery-rul`

Two functions:

**`evaluate_rul()`** — computes MAE/RMSE/MAPE on held-out test sequences

**`predict_rul_per_device()`** — takes the most recent window from each UPS device
and produces a per-device prediction with MC Dropout confidence intervals:

```
ups_apc_01: RUL=1092d CI=[1087, 1097] [ok]
ups_apc_02: RUL=1079d CI=[1074, 1084] [ok]
```

The advisory thresholds:
```python
URGENT_DAYS = 14   # replacement needed within 2 weeks
WARN_DAYS   = 30   # replacement needed within 1 month
```

---

## 13. Detection — dual model scorer

**File:** `snmp_anomaly_detection/power/inference/dual_model_scorer.py`
**Command:** `python3 main.py detect-power-csv`

This is the main detection entry point for batch/CSV-based detection.

### What it does

For each device, for each sliding window of data:

1. **Load both models** (baseline and phase) with their saved scalers and thresholds
2. **Scale separately** — this is critical
3. **Score each window** through both models
4. **Combine flags** according to alert policy

### The critical scaling bug (now fixed)

`PHASE_LEVEL_FEATURES` includes all of `BASELINE_UPS_FEATURES` (it's a superset).

If you do this (wrong):
```python
device_df[phase_cols] = phase_scaler.transform(device_df[phase_cols])
# Now device_df has phase-scaled values in the baseline columns too

baseline_vals = device_df[baseline_cols].values
# These are now PHASE-SCALED, not BASELINE-SCALED
# Baseline model sees wrong scale → reconstruction error = 300,000 vs threshold 0.06
# Result: every single window flagged as anomaly
```

The fix — scale into separate arrays:
```python
baseline_vals = baseline_scaler.transform(device_df[baseline_cols])
phase_vals = phase_scaler.transform(phase_df[phase_cols])
# Each scaler's output goes to its own numpy array
# Neither overwrites the other
```

### Alert policy

```python
if policy == "or":
    combined = int(baseline_flag OR phase_flag)   # either model triggers = alert
else:
    combined = int(baseline_flag AND phase_flag)  # both must trigger = alert
```

- `"or"` policy: higher recall (catches more anomalies), more false positives
- `"and"` policy: higher precision (fewer false alerts), misses some anomalies

Change `alert_policy` in `PowerInferenceConfig` to switch.

### Output

`outputs/power/dual/anomaly_results.csv` — one row per detection window:

| Column | Meaning |
|---|---|
| `device_id` | which device |
| `window_start` / `window_end` | timestamps of the 10-step window |
| `baseline_error` | reconstruction error from baseline model |
| `baseline_threshold` | the stored threshold for comparison |
| `baseline_anomaly` | 1 if baseline flagged, 0 if not |
| `phase_error` | reconstruction error from phase model |
| `phase_anomaly` | 1 if phase flagged |
| `combined_flag` | final decision (OR or AND of the two) |
| `true_label` | ground truth from dataset (for evaluation) |
| `anomaly_types` | which anomaly types appeared in this window |

---

## 14. Streaming processor

**File:** `snmp_anomaly_detection/power/inference/power_stream_processor.py`
**Commands:** `python3 main.py detect-power-kafka` / `produce-power-kafka-test`

This extends the master branch's streaming pattern to handle power events.

### PowerEvent

```python
@dataclass
class PowerEvent:
    timestamp: datetime
    device_id: str
    device_category: str        # ups | pdu | network | env
    vendor: str
    feature_values: dict[str, float]   # feature name → value
```

Each Kafka message is parsed into a `PowerEvent`. The `feature_values` dict
holds all available SNMP readings by canonical name.

### Routing by device category

```python
if event.device_category == "ups":
    # Score through BOTH models
    baseline_flag = baseline_model(baseline_features)
    phase_flag = phase_model(phase_features)
    combined = baseline_flag OR phase_flag
else:
    # PDU, network, env — baseline only (no phase-level model for these)
    baseline_flag = baseline_model(baseline_features)
    combined = baseline_flag
```

### Per-device rolling window

Each device maintains its own buffer of the last N readings (N = seq_len).
When the buffer is full, scoring runs. New events push out old ones.

```python
class PowerDeviceWindowManager:
    # separate buffer per device_id
    # deque with maxlen=seq_len
    def push(device_id, vector):
        buffer.append(vector)
        if len(buffer) == seq_len:
            return np.array(buffer)   # ready to score
        return None                   # not enough data yet
```

### Compound alerts

If a UPS anomaly and a PDU anomaly both fire within 10 minutes of each other,
a `COMPOUND_ALERT` is emitted linking both device IDs.

```python
# After detecting a UPS anomaly:
for (pdu_id, pdu_ts) in recent_pdu_anomalies:
    if |ups_ts - pdu_ts| <= 10 minutes:
        emit COMPOUND_ALERT(ups_id, pdu_id)
```

This catches scenarios like: PDU overloaded → UPS kicks in → both anomalous simultaneously.

### Kafka topic

- **Consumer topic:** `snmp-power-events`
- **Producer** (`produce_power_kafka_test.py`) replays from the synthetic CSV dataset
- Each message is JSON with all feature values + device metadata
- Stop either consumer or producer with `Ctrl+C`

---

## 15. Where to change things

### I want to change the number of features

Edit `BASELINE_UPS_FEATURES` or `PHASE_LEVEL_FEATURES` in `config.py`.
Then re-run `generate-power-data`, `preprocess-power`, `train-power-baseline`,
`train-power-phase`, `detect-power-csv` in order — the model files are
incompatible between feature counts.

### I want to add a new anomaly type

Edit `_inject_power_anomaly()` in `dataset_builder.py` to add a new `elif` branch.
Add the new type name to `_anomaly_types_for_category()` for the relevant categories.
Re-run `generate-power-data` and retrain.

### I want to change the detection threshold

**Option A — directly:** edit `threshold_std_multiplier` in `PowerTrainingConfig`
in `config.py`, then retrain. Higher value = fewer alerts.

**Option B — post-hoc:** after training, edit `baseline_metadata.json` or
`phase_metadata.json` and change the `"threshold"` value directly.
No retraining needed. Useful for quick experiments.

### I want to use a different scaler

Replace `RobustScaler` with `StandardScaler` or `MinMaxScaler` in
`power_features.py::scale_features()` and `phase_features.py::scale_phase_features()`.
`RobustScaler` was chosen because it's resistant to outliers.

### I want to use real SNMP data

1. Replace `build_power_dataset()` with a function that reads your real CSV
   (with the same column names as `BASELINE_UPS_FEATURES` + `PHASE_LEVEL_FEATURES`)
2. The rest of the pipeline works unchanged because everything downstream
   reads from the CSV, not from the generator
3. Make sure to set `anomaly=0` for all rows initially — add true labels
   later from incident records if available
4. Retrain all models on the real data

### I want to change the LSTM architecture

Edit `LSTMAutoencoder` in `models/lstm_autoencoder.py`. Change `hidden_size`
and `latent_size` in `PowerTrainingConfig` in `config.py`.

### I want to change alert policy

Change `alert_policy` in `PowerInferenceConfig` in `config.py`:
- `"or"` → flag if either model fires (higher recall)
- `"and"` → flag only if both models fire (higher precision)

### I want to change RUL advisory thresholds

Edit `URGENT_DAYS` and `WARN_DAYS` in `models/battery_rul.py`.

### I want to add a new vendor

1. Add a new OID map dict in `data/vendor_oid_map.py`
2. Add an entry to the `_ALL_MAPS` lookup in `resolve_to_canonical()`
3. Add a `PowerDeviceProfile` entry in `dataset_builder.py` with `vendor="your_vendor"`
4. Re-generate data and retrain

---

## Key numbers summary

| Model | Features | Seq len | Hidden | Threshold | F1 |
|---|---|---|---|---|---|
| Baseline LSTM Autoencoder | 10 | 10 | 64 | 0.0624 | 0.696 |
| Phase LSTM Autoencoder | 21 | 10 | 64 | 0.0565 | — (see dual) |
| Dual (OR policy) | 10 + 21 | 10 | — | — | **0.741** |
| Battery RUL LSTM | 6 | 24 | 64 | — | MAE ~1647d* |

*RUL MAE is high because synthetic data covers only 7 days. Real accuracy needs months of data.

---

## Data flow diagram

```
generate-power-data
  └─ dataset_builder.py → synthetic_power_snmp_dataset.csv (18,144 rows)

preprocess-power
  └─ power_features.py
       load CSV → aggregate_phase_metrics → apply_log1p → filter_normal
       → RobustScaler (fit on train only) → sliding windows
       → outputs/power/baseline/X_train.npy

train-power-baseline
  └─ train_baseline_power.py
       load X_train.npy → LSTMAutoencoder → 30 epochs
       → compute threshold → save model.pt + metadata.json + scaler.pkl

train-power-phase
  └─ train_phase_model.py
       load CSV → log1p → enrich_imbalance → filter_normal
       → clip imbalance [0,10%] → RobustScaler → sliding windows
       → LSTMAutoencoder (21 features) → save phase_model.pt

train-battery-rul
  └─ train_battery_rul.py
       load CSV → log1p → derive_rul_labels → RobustScaler
       → build_rul_sequences (UPS only) → BatteryRULModel (regression)
       → save rul_model.pt

evaluate-power-baseline
  └─ power_eval.py → p1_metrics.json (P=0.956, R=0.547, F1=0.696)

predict-battery-rul
  └─ rul_eval.py → rul_predictions.json (per-device RUL + advisory)

detect-power-csv
  └─ dual_model_scorer.py
       load both models + scalers
       for each device, for each window:
         baseline_vals = baseline_scaler.transform(baseline_cols)  ← separate arrays!
         phase_vals = phase_scaler.transform(phase_cols)
         b_flag = baseline_error > baseline_threshold
         p_flag = phase_error > phase_threshold
         combined = b_flag OR p_flag
       → outputs/power/dual/anomaly_results.csv

detect-power-kafka  (live)
  └─ power_stream_processor.py
       Kafka consumer → PowerEvent → route by device_category
       → rolling buffer per device → score when full → emit result
```
