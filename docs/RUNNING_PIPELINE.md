# Running the Project

This document explains how to run each module separately and how to run the full SNMP anomaly detection pipeline end to end.

> **Branch guide:**
> - `master` / `feature-snmp-v2-interface-roadmap` — generic network SNMP (cpu, memory, octets, errors)
> - `feature-power-snmp-detection` — power equipment SNMP (UPS, PDU, network gear PSU, environmental sensors) — see [Section 9](#9-power-snmp-pipeline-feature-power-snmp-detection-branch)

## 1. Project Type

This is a Python project. There is no separate build or compile step like Java or C++.

What you need instead:
- install dependencies
- run the required module or pipeline step

Optional syntax check:

```bash
python3 -m compileall snmp_anomaly_detection
```

This only checks that the Python files compile successfully to bytecode.

## 2. Environment Setup

From the project root:

```bash
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have the dependencies installed, you can skip the virtual environment setup.

## 3. Main Entry Point

The project exposes a single CLI entry point:

```bash
python3 main.py --help
```

Available steps:

```bash
# Network SNMP pipeline
python3 main.py generate-data
python3 main.py preprocess
python3 main.py train
python3 main.py detect
python3 main.py detect-csv
python3 main.py detect-kafka-dry
python3 main.py detect-kafka
python3 main.py produce-kafka-test-data

# Power SNMP pipeline (feature-power-snmp-detection branch)
python3 main.py generate-power-data
python3 main.py preprocess-power
python3 main.py train-power-baseline
python3 main.py train-power-phase
python3 main.py train-battery-rul
python3 main.py train-power-iforest
python3 main.py evaluate-power-baseline
python3 main.py predict-battery-rul
python3 main.py detect-power-csv
python3 main.py detect-power-kafka
python3 main.py produce-power-kafka-test
```

## 4. Run Each Module Separately

You can run the pipeline steps one by one through `main.py` or directly as Python modules.

### A. Generate synthetic dataset

Using main entry point:

```bash
python3 main.py generate-data
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.data.dataset_builder
```

Output:
- dataset CSV is saved to `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`

### B. Preprocess dataset

This step:
- loads the dataset
- keeps normal rows for training
- scales feature columns
- creates sequence windows
- saves train/test arrays and scaler

Using main entry point:

```bash
python3 main.py preprocess
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.preprocessing.feature_engineering
```

Outputs:
- `snmp_anomaly_detection/utils/scaler.pkl`
- `snmp_anomaly_detection/outputs/X_train.npy`
- `snmp_anomaly_detection/outputs/X_test.npy`
- `snmp_anomaly_detection/outputs/y_train.npy`
- `snmp_anomaly_detection/outputs/y_test.npy`

### C. Train the model

This step:
- loads training arrays
- trains the LSTM autoencoder
- computes the anomaly threshold
- saves model artifacts

Using main entry point:

```bash
python3 main.py train
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.training.train_model
```

Outputs:
- `snmp_anomaly_detection/outputs/lstm_autoencoder.pth`
- `snmp_anomaly_detection/outputs/model_metadata.json`

### D. Run anomaly detection

This step:
- loads the dataset
- loads scaler and trained model
- creates detection windows
- scores each window
- saves anomaly detection outputs

Using main entry point:

```bash
python3 main.py detect
```

Explicit CSV replay mode:

```bash
python3 main.py detect-csv
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.inference.detect_anomalies
python3 -m snmp_anomaly_detection.inference.csv_replay
```

Outputs:
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`

### E. Run Kafka dry ingestion

This step:
- connects to Kafka
- consumes messages from the hardcoded topic
- decodes JSON payloads
- normalizes them into the shared event shape
- does not run anomaly scoring yet

Using main entry point:

```bash
python3 main.py detect-kafka-dry
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.streaming.detect_kafka_dry
```

Notes:
- current v1 topic is hardcoded in code
- `kafka-python` must be installed
- stop the dry consumer with `Ctrl+C`

### F. Publish Kafka test data

This step:
- generates fresh synthetic SNMP events in memory
- creates continuous per-device telemetry for a shared topic
- injects anomalies probabilistically for testing
- publishes messages in the same JSON schema expected by the Kafka consumer

Using main entry point:

```bash
python3 main.py produce-kafka-test-data
python3 main.py produce-kafka-test-data --device-count 20 --sleep-seconds 0.05
python3 main.py produce-kafka-test-data --max-messages 200 --anomaly-probability 0.10
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.streaming.produce_kafka_test_data
```

Notes:
- the producer sends to the same hardcoded topic used by `detect-kafka-dry`
- messages include: `timestamp`, `device_id`, `cpu`, `memory`, `in_octets`, `out_octets`, `errors`, `anomaly`
- default behavior is continuous streaming until `Ctrl+C`
- detailed test-generator guide: `docs/streaming/KAFKA_TEST_DATA_GENERATOR.md`

### G. Run Kafka live detection

This step:
- connects to Kafka
- validates incoming messages
- normalizes valid payloads into shared events
- builds per-device rolling windows
- applies live micro-batching
- prints live anomaly results locally

Using main entry point:

```bash
python3 main.py detect-kafka
```

Using module directly:

```bash
python3 -m snmp_anomaly_detection.streaming.detect_kafka
```

Notes:
- current v1 topic is hardcoded in code
- live results are printed to the terminal
- live results are also written locally to `snmp_anomaly_detection/outputs/kafka_live_results.jsonl`
- anomalous live windows are also written locally to `snmp_anomaly_detection/outputs/kafka_live_anomaly_windows.jsonl`
- stop the live consumer with `Ctrl+C`

## 5. Run the Full Pipeline End to End

Run these commands in order from the project root:

```bash
python3 main.py generate-data
python3 main.py preprocess
python3 main.py train
python3 main.py detect
```

This executes the full pipeline:
1. create synthetic SNMP data
2. preprocess and scale data
3. train the LSTM autoencoder
4. run anomaly detection and save results

## 6. Minimal Detection-Only Flow

If the dataset, scaler, and model are already available, you only need:

```bash
python3 main.py detect
```

Required existing files:
- `snmp_anomaly_detection/utils/scaler.pkl`
- `snmp_anomaly_detection/outputs/lstm_autoencoder.pth`
- `snmp_anomaly_detection/outputs/model_metadata.json`
- dataset CSV in one of the supported locations

Supported dataset lookup order:
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- `synthetic_snmp_dataset.csv`
- `featureEngineering/synthetic_snmp_dataset.csv`

## 7. Important Output Locations

Main outputs are stored here:

- dataset: `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- scaler: `snmp_anomaly_detection/utils/scaler.pkl`
- training arrays: `snmp_anomaly_detection/outputs/`
- model: `snmp_anomaly_detection/outputs/lstm_autoencoder.pth`
- metadata: `snmp_anomaly_detection/outputs/model_metadata.json`
- detection CSV: `snmp_anomaly_detection/outputs/anomaly_results.csv`
- anomaly windows JSON: `snmp_anomaly_detection/outputs/anomaly_windows.json`

## 8. Quick Verification

To check that the package imports correctly:

```bash
python3 -m compileall snmp_anomaly_detection
```

To test the full working flow:

```bash
python3 main.py generate-data
python3 main.py preprocess
python3 main.py train
python3 main.py detect
```

If detection completes successfully, the final result files should appear in:

```text
snmp_anomaly_detection/outputs/
```

---

## 9. Power SNMP Pipeline (`feature-power-snmp-detection` branch)

### What this branch does

This branch adds anomaly detection and forecasting for **power equipment** monitored via SNMP:

| Equipment | Examples | SNMP MIBs |
|-----------|----------|-----------|
| UPS (single-phase) | APC SUA3000 | PowerNet-MIB |
| UPS (three-phase) | Liebert GXT 10kVA | LIEBERT-GP-POWER-MIB, RFC 1628 UPS-MIB |
| PDU | APC AP7900, Raritan PX2 | rPDU MIB, PX2-MIB |
| Network gear PSU | Generic switches (dual PSU) | ENTITY-MIB |
| Environmental sensors | Temp/humidity | SENSOR-MIB |

It adds **11 new CLI commands** on top of the original 8 and trains **four separate models**:

1. **Baseline LSTM Autoencoder** — 16 vendor-agnostic features (absolute voltages/currents replaced by ratio/deviation columns to remove 120 V vs 230 V bias) plus delta features (`runtime_delta`, `temperature_delta`, `output_load_delta`)
2. **Phase-level LSTM Autoencoder** — 30 features: all 16 baseline features plus per-phase L1/L2/L3 voltages and currents, voltage/current imbalance percentages, and per-phase voltage-drop deltas (`voltage_drop_delta_l1/l2/l3`) added by B3 to fix phase-sag recall
3. **Battery RUL LSTM Regression** — predicts days until battery replacement from charge rate, discharge cycles, and battery health trends
4. **Isolation Forest** — per-device-category anomaly detector trained on normal-only baseline features; battery features masked out for non-UPS categories to avoid constant-zero corruption of isolation splits

The four signals combine as a **four-signal scorer**:
- Baseline LSTM reconstruction error vs per-category threshold
- Phase LSTM reconstruction error vs per-category threshold (UPS only)
- Rule-based overload flag (`output_load_pct > 100` → zero false positives)
- Isolation Forest score vs per-category threshold with ratio guard (`IF_MIN_SCORE_RATIO = 1.05`)

`final_flag = baseline_flag OR phase_flag OR overload_rule_flag OR iforest_flag`

### Quick start (full pipeline)

```bash
git checkout feature-power-snmp-detection
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation

# Phase 1 — generate synthetic power dataset
python3 main.py generate-power-data

# Phase 2 — preprocess + train + evaluate baseline UPS model
python3 main.py preprocess-power
python3 main.py train-power-baseline
python3 main.py evaluate-power-baseline

# Phase 3 — train phase-level model + Isolation Forest + run four-signal detection on CSV
python3 main.py train-power-phase
python3 main.py train-power-iforest
python3 main.py detect-power-csv

# Phase 4 — train battery RUL model + generate per-device predictions
python3 main.py train-battery-rul
python3 main.py predict-battery-rul

# Phase 5 — live Kafka streaming (requires Kafka running on localhost:9092)
python3 main.py detect-power-kafka                                                             # terminal 1
python3 main.py produce-power-kafka-test --use-training-profiles --anomaly-probability 0.0    # terminal 2: smoke test (expect zero alerts)
python3 main.py produce-power-kafka-test --use-training-profiles --seed 42                    # terminal 2: full detection test
```

### Command reference

#### Phase 1 — Data generation

```bash
python3 main.py generate-power-data
```

Generates a synthetic multi-vendor power SNMP dataset with 21 device profiles across four categories, 5-minute intervals, and probabilistic anomaly injection (~5% rate).

Output:
- `snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv` — 42,336 rows (~7 days per device), 21 devices

Device profiles: 14 UPS (generic/apc/eaton/liebert, 1–15 kW, 120 V and 230 V), 4 PDU (apc/raritan), 2 network PSU (cisco/generic), 1 environmental sensor

Anomaly types injected: `battery_drain`, `overload`, `phase_sag`, `thermal_runaway`, `psu_failure`

#### Phase 2 — Baseline UPS model

```bash
python3 main.py preprocess-power
```

Loads the power dataset, applies log1p to skewed columns (`runtime_remaining_min`, `output_power_w`), fits a RobustScaler on normal-only rows, and builds sliding-window sequences.

Outputs:
- `snmp_anomaly_detection/outputs/power_baseline/baseline_scaler.pkl`
- `snmp_anomaly_detection/outputs/power_baseline/X_train.npy`

```bash
python3 main.py train-power-baseline
```

Trains a 16-feature LSTM Autoencoder (hidden=64, latent=32, 100 epochs, seq_len=20).
Threshold is `mean + 2.0×std` of training reconstruction errors on normal sequences.
Also computes **per-device-category thresholds** (env/network/pdu/ups) saved in metadata.

> **B2 normalization:** absolute voltage/current columns replaced by vendor-agnostic ratio/deviation features (`battery_voltage_ratio`, `battery_current_ratio`, `input_voltage_dev_pct`, `output_voltage_dev_pct`, `output_current_ratio`). `output_power_w` and `battery_charge_delta` were dropped. This eliminates 230 V vs 120 V false positives without any changes to the model architecture.

Outputs:
- `snmp_anomaly_detection/outputs/power_baseline/baseline_model.pt`
- `snmp_anomaly_detection/outputs/power_baseline/baseline_metadata.json`

```bash
python3 main.py evaluate-power-baseline
```

Runs the baseline model on the held-out test split and reports Precision, Recall, F1.

Output:
- `snmp_anomaly_detection/outputs/power_baseline/p1_metrics.json`

Achieved metrics on synthetic dataset (single-model baseline only):

| Metric | Value |
|--------|-------|
| Threshold (global) | 0.193 |
| Per-category thresholds | env=0.112, network=0.172, pdu=0.113, ups=0.219 |

#### Phase 3 — Phase-level model + dual detection

```bash
python3 main.py train-power-phase
```

Trains a 30-feature LSTM Autoencoder on per-phase L1/L2/L3 metrics, voltage/current imbalance percentages, and delta features (runtime, temperature, load).
Also saves per-category thresholds to `phase_metadata.json`.

> **B3 addition:** `voltage_drop_delta_l1/l2/l3` (clipped to ≤0 — captures only voltage drops, not rises) were added to amplify the phase-sag signal that was diluted in global MSE. This brought phase_sag recall from ~25% → 100% on the synthetic dataset.

> **Important:** `voltage_imbalance_pct` and `current_skew_pct` are clipped to `[0%, 10%]` before RobustScaler. Single-phase devices produce near-zero IQR for these columns which causes RobustScaler to overflow. Clipping to the physical fault ceiling (`10%`) prevents this.

Outputs:
- `snmp_anomaly_detection/outputs/power_phase/phase_model.pt`
- `snmp_anomaly_detection/outputs/power_phase/phase_metadata.json`
- `snmp_anomaly_detection/outputs/power_phase/phase_scaler.pkl`

```bash
python3 main.py train-power-iforest
```

Trains one `IsolationForest` (`n_estimators=200`) per device category on normal-only baseline-scaled rows.
Battery-specific features (`battery_voltage_ratio`, `battery_current_ratio`, `on_battery_status`, `battery_replace_status`) and `output_frequency_hz` are **excluded** for non-UPS categories — constant zeros in these columns corrupt isolation-tree splits and inflate scores.

Outputs:
- `snmp_anomaly_detection/outputs/power_dual/iforest_models.pkl` — per-category fitted models
- `snmp_anomaly_detection/outputs/power_dual/iforest_metadata.json` — per-category thresholds and feature column lists

Borderline IF flags are suppressed by a ratio guard (`IF_MIN_SCORE_RATIO = 1.05`): the IF score must be at least 5% below the threshold — filters false positives that cluster at ratio 1.00–1.10 on normal data.

> **E5: batch pre-scoring** — all rows for a device are scored in one `score_samples()` call before the window loop, so the loop only slices the pre-computed array rather than making individual sklearn calls per window.

```bash
python3 main.py detect-power-csv
```

Runs the **four-signal scorer** on the full CSV dataset:
1. Baseline LSTM — per-category threshold
2. Phase LSTM — per-category threshold (UPS devices only)
3. Rule: `output_load_pct > 100` → `overload_rule_flag=1` (zero FP)
4. Isolation Forest — per-category threshold with `IF_MIN_SCORE_RATIO` ratio guard

`final_flag = combined_flag OR overload_rule_flag OR iforest_flag`

> **Implementation note:** `PHASE_LEVEL_FEATURES ⊃ BASELINE_UPS_FEATURES`. Scaling must be done into **separate numpy arrays** — not the same dataframe — to prevent the phase scaler from overwriting the baseline-scaled columns.

Outputs:
- `snmp_anomaly_detection/outputs/power_dual/anomaly_results.csv` — per-window scores including `final_flag`
- `snmp_anomaly_detection/outputs/power_dual/detection_summary.json`
- `snmp_anomaly_detection/outputs/power_dual/anomaly_windows_detail.json` — per-timestep breakdown

Achieved metrics (three-signal OR, 2026-04-27, **pre-B2/B3**):

> **Note:** these metrics predate B2 (vendor-agnostic normalization) and B3 (voltage drop deltas). Re-run `detect-power-csv` after retraining both models to get updated numbers. Phase_sag recall in particular is expected to reach ~100% after B3.

| Metric | Value |
|--------|-------|
| **Precision** | **0.992** |
| **Recall** | **0.815** |
| **F1** | **0.895** |
| Windows flagged (final) | 9,511 / 17,964 |
| True anomaly windows | 11,575 |
| False positive rate | 1.2% |

Per anomaly type (pre-B2/B3):

| Type | Recall (pre-B2/B3) | Notes |
|---|---|---|
| battery_drain | 1.000 | Unchanged |
| psu_failure | 1.000 | Unchanged |
| thermal_runaway | 0.948 | Unchanged |
| overload | 0.653 | Normal load std=25% overlaps moderate overload; rule only fires above 100% |
| phase_sag | 0.539 → **~1.000** | Fixed by B3 voltage drop deltas |

#### Phase 4 — Battery RUL forecasting

```bash
python3 main.py train-battery-rul
```

Trains an LSTM regression model (hidden=64, 2 layers, MC Dropout=0.1) to predict remaining battery life in days. Loss: MSE on RUL labels derived from synthetic install-age data.

Outputs:
- `snmp_anomaly_detection/outputs/battery_rul/rul_model.pt`
- `snmp_anomaly_detection/outputs/battery_rul/rul_scaler.pkl`

```bash
python3 main.py predict-battery-rul
```

Generates per-device RUL predictions with 95% confidence intervals (30 MC Dropout forward passes). Issues maintenance advisories:
- `urgent` — predicted RUL < 14 days
- `warn` — predicted RUL < 30 days
- `ok` — predicted RUL >= 30 days

Output:
- `snmp_anomaly_detection/outputs/battery_rul/rul_predictions.json`
- `snmp_anomaly_detection/outputs/battery_rul/rul_metrics.json`

> **Limitation:** With only 7 days of synthetic data (2,016 timesteps per device), RUL labels vary by at most ~7 days within any training window. MAE of ~1,647 days is expected on this dataset. Meaningful RUL prediction requires months to years of historical battery telemetry.

#### Phase 5 — Kafka streaming

**Prerequisites:** Kafka broker running on `localhost:9092`. Topic: `snmp-power-events`.

Run the consumer and producer in separate terminals:

```bash
# Terminal 1 — start live dual-model detection
python3 main.py detect-power-kafka
```

```bash
# Terminal 2 — smoke test first (no anomalies, expect zero alerts)
python3 main.py produce-power-kafka-test --use-training-profiles --anomaly-probability 0.0

# Full detection test (anomalies injected, reproducible)
python3 main.py produce-power-kafka-test --use-training-profiles --seed 42

# Stress test (higher anomaly rate)
python3 main.py produce-power-kafka-test --use-training-profiles --anomaly-probability 0.15 --seed 99

# Random fleet (for exploring generalisation — expect some FPs until model is retrained on wider data)
python3 main.py produce-power-kafka-test --device-count 16 --num-events 800 --seed 42
```

**Testing strategy — always run in this order:**

1. **Smoke test** (`--use-training-profiles --anomaly-probability 0.0`): zero alerts expected. Any alert is a false positive and indicates a model or preprocessing issue.
2. **Full detection test** (`--use-training-profiles --seed 42`): alerts should appear only on injected anomaly windows. Compare `final_flag` against `expected_label` in the JSONL output.
3. **Stress test** (`--anomaly-probability 0.15`): verifies the alert pipeline fires reliably at higher injection rates.

Producer options:

| Flag | Default | Description |
|------|---------|-------------|
| `--use-training-profiles` | off | Use the exact 21 training device profiles instead of random ones. **Required for smoke testing** — guarantees all feature values are in-distribution for the fitted scaler. Ignores `--device-count`. |
| `--device-count` | 12 | Number of randomly generated devices. Ignored when `--use-training-profiles` is set. |
| `--num-events` | 600 | Total main-phase events to produce. A warmup batch of 20 events per device is always prepended. |
| `--sleep-seconds` | 0.05 | Delay between messages (~20 msg/s). |
| `--anomaly-probability` | 0.05 | Per-event anomaly injection probability. Set to `0.0` for smoke testing. |
| `--seed` | random | Random seed for reproducible profiles and data. |

**Why `--use-training-profiles` matters:** random profiles can generate device configurations (e.g. a 10 000 W UPS at 120 V) whose feature values — particularly `output_current_a` — fall outside the range seen during training, causing every window to score above the reconstruction threshold even with no anomaly injected. Training profiles are guaranteed in-distribution.

With random profiles (no flag), the fleet is distributed across categories with weighted sampling:

| Category | Phase options | Vendors |
|----------|--------------|---------|
| `ups` | 1-phase or 3-phase | apc, liebert, generic |
| `pdu` | 1-phase or 3-phase | apc, raritan, generic |
| `network` | 1-phase only | cisco, generic |
| `env` | 1-phase only | generic |

**Device routing** in the stream processor:
- `ups` → baseline LSTM + phase LSTM (dual scoring)
- `pdu`, `network`, `env` → baseline LSTM only

**Compound alerts**: when a UPS anomaly and a PDU anomaly occur within 10 minutes of each other, a `COMPOUND_ALERT` is emitted correlating both devices.

**Runtime file updates**: all three report files are rewritten after every scored window — you can open them in a viewer while the consumer is running and see results accumulate in real time. On `Ctrl+C` shutdown, a final verbose summary is also printed to the console.

Outputs written to `snmp_anomaly_detection/outputs/power_dual/`:

| File | Description |
|------|-------------|
| `anomaly_results.csv` | Per-window scores: device, timestamps, errors, thresholds, all flags — identical schema to `detect-power-csv` |
| `detection_summary.json` | Aggregate counts by device / vendor / category, collapsed event count |
| `anomaly_windows_detail.json` | Per-event detail with full timestep error sequences, peak timestep, top contributing features — same format as `detect-power-csv` |
| `kafka_power_results.jsonl` | Append-only raw stream log (one JSON line per scored window, written in real time) |

> **Note:** `true_label` and `anomaly_types` are always `0` / `[]` in Kafka output since ground truth is not available at inference time.

### Output file map

```text
snmp_anomaly_detection/
├── data/
│   ├── synthetic_power_snmp_dataset.csv       # generated power dataset
│   └── vendor_oid_map.py                      # APC/Liebert/RFC1628/Raritan OID→canonical mapping
├── outputs/
│   ├── power_baseline/
│   │   ├── baseline_model.pt                  # trained baseline LSTM
│   │   ├── baseline_scaler.pkl                # RobustScaler fitted on normal rows
│   │   ├── baseline_metadata.json             # input_size, hidden_size, threshold
│   │   └── p1_metrics.json                    # Precision/Recall/F1
│   ├── power_phase/
│   │   ├── phase_model.pt                     # trained phase LSTM
│   │   ├── phase_scaler.pkl                   # phase RobustScaler (with imbalance clipping)
│   │   └── phase_metadata.json
│   ├── power_dual/
│   │   ├── anomaly_results.csv                # per-window scores — CSV and Kafka, identical schema
│   │   ├── detection_summary.json             # aggregate counts by device/vendor/category
│   │   ├── anomaly_windows_detail.json        # per-event timestep breakdown + top features
│   │   ├── kafka_power_results.jsonl          # raw Kafka stream log (append-only, runtime)
│   │   ├── iforest_models.pkl                 # per-category IsolationForest models (E1/E2)
│   │   └── iforest_metadata.json             # per-category thresholds + feature column lists
│   └── battery_rul/
│       ├── rul_model.pt                       # RUL regression LSTM
│       ├── rul_scaler.pkl
│       ├── rul_predictions.json               # per-device RUL + advisory
│       └── rul_metrics.json                   # MAE/RMSE/MAPE
├── preprocessing/
│   ├── power_features.py                      # baseline feature engineering
│   └── phase_features.py                      # phase L1/L2/L3 + imbalance features
├── training/
│   ├── train_baseline_power.py
│   ├── train_phase_model.py
│   └── train_battery_rul.py
├── inference/
│   ├── dual_model_scorer.py                   # combined baseline+phase detection
│   └── power_stream_processor.py             # Kafka stream routing by device_category
└── streaming/
    ├── detect_power_kafka.py                  # Kafka consumer entry point
    └── produce_power_kafka_test.py            # Kafka test producer
```

### Feature sets

**Baseline UPS features (16):**
Vendor-agnostic health signals: `battery_charge_pct`, `battery_voltage_ratio` *(v / rated_battery_v)*, `battery_current_ratio` *(a / rated_discharge_current)*, `battery_temperature_c`, `runtime_remaining_min`, `on_battery_status`, `battery_replace_status`, `input_voltage_dev_pct` *((v − nominal) / nominal × 100)*, `input_frequency_hz`, `output_voltage_dev_pct`, `output_current_ratio` *(a / (rated_w / nominal_v))*, `output_load_pct`, `output_frequency_hz`
Rate-of-change (signed-log1p compressed): `runtime_delta`, `temperature_delta`, `output_load_delta`

> `output_power_w` and `battery_charge_delta` were removed in B2 — redundant/noise after normalization.

**Phase-level features (30):** all 16 baseline features + `input_voltage_l1/l2/l3`, `input_current_l1/l2/l3`, `output_current_l1/l2/l3`, `voltage_imbalance_pct`, `current_skew_pct`, `voltage_drop_delta_l1/l2/l3` *(B3 — clipped to ≤0)*

**Battery RUL features (7):** `battery_charge_pct`, `battery_voltage_v`, `battery_current_a`, `battery_temperature_c`, `runtime_remaining_min`, `charge_rate`, `discharge_cycles_approx`

### Known issues and limitations (updated 2026-05-08)

| Issue | Details |
|-------|---------|
| overload recall ~0.65 | Normal load variance (std=25%) overlaps moderate overload (65–100%). Rule only fires above 100%. Improving requires tighter load-spike features or a narrower normal distribution. |
| phase_sag recall | **Fixed by B3** — `voltage_drop_delta_l1/l2/l3` bring recall from ~25% → ~100% on the synthetic dataset. |
| RUL accuracy | Synthetic 7-day window: MAE ~1,647 days. Meaningful RUL needs months of real battery telemetry. |
| IF auto-calibration | `IF_MIN_SCORE_RATIO = 1.05` is a fixed stop-gap. Per-category calibration from FP/TP gap (E4b) is planned but blocked until real device data (A1) is available. |
| Kafka Phase 5 | Streaming code complete but requires a running Kafka broker. |
| Metric table stale | The precision/recall/F1 table above is pre-B2/B3. Retrain and re-detect to get updated numbers. |

