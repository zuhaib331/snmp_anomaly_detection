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
python3 main.py generate-data
python3 main.py preprocess
python3 main.py train
python3 main.py detect
python3 main.py detect-csv
python3 main.py detect-kafka-dry
python3 main.py detect-kafka
python3 main.py produce-kafka-test-data
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

It adds **10 new CLI commands** on top of the original 8 and trains **three separate models**:

1. **Baseline LSTM Autoencoder** — 10 aggregated UPS health metrics (battery %, voltage, temperature, runtime, input voltage/frequency, output load/power, battery flag, bypass flag)
2. **Phase-level LSTM Autoencoder** — 21 features including per-phase L1/L2/L3 voltages and currents plus voltage and current imbalance percentages
3. **Battery RUL LSTM Regression** — predicts days until battery replacement from charge rate, discharge cycles, and battery health trends

The baseline and phase models run together as a **dual-model scorer** with an OR alert policy (either model flags → alert).

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

# Phase 3 — train phase-level model + run dual detection on CSV
python3 main.py train-power-phase
python3 main.py detect-power-csv

# Phase 4 — train battery RUL model + generate per-device predictions
python3 main.py train-battery-rul
python3 main.py predict-battery-rul

# Phase 5 — live Kafka streaming (requires Kafka running)
python3 main.py produce-power-kafka-test   # terminal 1
python3 main.py detect-power-kafka         # terminal 2
```

### Command reference

#### Phase 1 — Data generation

```bash
python3 main.py generate-power-data
```

Generates a synthetic multi-vendor power SNMP dataset with 9 device profiles, 5-minute intervals, and probabilistic anomaly injection (~5% rate).

Output:
- `snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv` — 18,144 rows, 9 devices

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

Trains a 10-feature LSTM Autoencoder (hidden=64, latent=32, 30 epochs). Threshold is set at the 95th percentile of validation reconstruction errors on normal sequences.

Outputs:
- `snmp_anomaly_detection/outputs/power_baseline/baseline_model.pt`
- `snmp_anomaly_detection/outputs/power_baseline/baseline_metadata.json`

```bash
python3 main.py evaluate-power-baseline
```

Runs the baseline model on the held-out test split and reports Precision, Recall, F1.

Output:
- `snmp_anomaly_detection/outputs/power_baseline/p1_metrics.json`

Achieved metrics on synthetic dataset:

| Metric | Value |
|--------|-------|
| Precision | 0.956 |
| Recall | 0.547 |
| F1 | 0.696 |
| Threshold | 0.0624 |

#### Phase 3 — Phase-level model + dual detection

```bash
python3 main.py train-power-phase
```

Trains a 21-feature LSTM Autoencoder on per-phase L1/L2/L3 metrics plus voltage/current imbalance percentages.

> **Important:** `voltage_imbalance_pct` and `current_skew_pct` are clipped to `[0%, 10%]` before RobustScaler. Single-phase devices produce near-zero IQR for these columns which causes RobustScaler to overflow. Clipping to the physical fault ceiling (`10%`) prevents this.

Outputs:
- `snmp_anomaly_detection/outputs/power_phase/phase_model.pt`
- `snmp_anomaly_detection/outputs/power_phase/phase_metadata.json`
- `snmp_anomaly_detection/outputs/power_phase/phase_scaler.pkl`

```bash
python3 main.py detect-power-csv
```

Runs the **dual-model scorer** on the full CSV dataset. Both models score every window independently; the combined flag uses OR policy (either model above threshold → alert).

> **Implementation note:** `PHASE_LEVEL_FEATURES ⊃ BASELINE_UPS_FEATURES`. Scaling must be done into **separate numpy arrays** — not the same dataframe — to prevent the phase scaler from overwriting the baseline-scaled columns.

Outputs:
- `snmp_anomaly_detection/outputs/power_dual/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/power_dual/detection_summary.json`

Achieved metrics (OR policy):

| Metric | Value |
|--------|-------|
| F1 | 0.741 |
| Precision | 0.953 |
| Windows flagged | 4,739 / 18,054 |
| True anomaly windows | 7,443 |

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

```bash
# Terminal 1 — produce 500 synthetic power events to Kafka topic snmp-power-events
python3 main.py produce-power-kafka-test

# Terminal 2 — consume and run dual-model detection live
python3 main.py detect-power-kafka
```

The stream processor routes events by `device_category`:
- `ups` → baseline model + phase model (dual scoring)
- `pdu`, `network`, `env` → baseline model only

**Compound alerts**: when a UPS anomaly and a PDU anomaly occur within 10 minutes of each other, a `COMPOUND_ALERT` is emitted correlating both devices.

Requires Kafka running locally on `localhost:9092`. Topic: `snmp-power-events`.

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
│   │   ├── anomaly_results.csv                # per-window dual-model scores
│   │   └── detection_summary.json             # aggregate counts by model
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

**Baseline UPS features (10):** `battery_charge_pct`, `battery_voltage_v`, `battery_temperature_c`, `runtime_remaining_min`, `input_voltage_avg`, `input_frequency_hz`, `output_load_pct`, `output_power_w`, `on_battery_flag`, `bypass_flag`

**Phase-level features (21):** all baseline features + `input_voltage_l1/l2/l3`, `input_current_l1/l2/l3`, `output_current_l1/l2/l3`, `voltage_imbalance_pct`, `current_skew_pct`

**Battery RUL features (6):** `battery_charge_pct`, `battery_voltage_v`, `battery_temperature_c`, `runtime_remaining_min`, `charge_rate`, `discharge_cycles_approx`

### Known issues and limitations

| Issue | Details |
|-------|---------|
| RUL accuracy requires real data | Synthetic 7-day window gives MAE ~1,647 days; need months of real battery telemetry |
| Phase model instability | Near-zero IQR on imbalance features for single-phase devices causes overflow; fixed by clipping to [0, 10%] before RobustScaler |
| Scaler overwrite bug (fixed) | `PHASE_LEVEL_FEATURES ⊃ BASELINE_UPS_FEATURES` — must scale into separate arrays, not the same df |
| Kafka Phase 5 not testable without Kafka | Streaming code is complete but requires a running Kafka broker |
