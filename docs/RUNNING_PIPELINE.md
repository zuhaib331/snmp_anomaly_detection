# Running the Project

This document explains how to run the current SNMP anomaly detection pipeline after `P0` completion and `F1` implementation.

The current validated baseline is:
- richer synthetic SNMP-like dataset
- cumulative counters
- interface-scoped windows
- derived-rate feature set:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`
- validated artifact directory:
  `f1_baseline_v1`

## 1. Environment Setup

From the project root:

```bash
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional syntax check:

```bash
python3 -m compileall snmp_anomaly_detection
```

## 2. Main Entry Point

The project exposes one CLI entry point:

```bash
python3 main.py --help
```

Available steps:

```bash
python3 main.py generate-data
python3 main.py preprocess
python3 main.py train
python3 main.py evaluate-baseline
python3 main.py detect
python3 main.py detect-csv
python3 main.py detect-kafka-dry
python3 main.py detect-kafka
python3 main.py produce-kafka-test-data
```

## 3. Current Data And Feature Flow

The current synthetic dataset now includes:
- `device_id`
- `timestamp`
- `interface`
- `cpu`
- `memory`
- cumulative `in_octets`
- cumulative `out_octets`
- cumulative `errors`
- packet counters
- discard counters
- interface speed
- interface admin and oper status
- anomaly metadata

The active `F1` model does not use all columns yet.
It currently trains on:

- `cpu`
- `memory`
- `in_rate`
- `out_rate`
- `error_rate`

These are derived during preprocessing from cumulative counters using elapsed time between polls.

## 4. Artifact Directory Concept

Preprocessing and training artifacts are saved in:

```text
snmp_anomaly_detection/artifacts/<artifact_dir_name>/
```

Typical files inside one artifact directory:

- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `scaler.pkl`
- `lstm_autoencoder.pth`
- `model_metadata.json`

Current meaningful artifact directories:

- `baseline_preprocess_ready_v1`
  preprocess-only historical reference
- `f1_baseline_v1`
  current validated `F1` baseline

Rule:
- use the same artifact directory name for `preprocess`, `train`, and `detect` when they belong to the same run

## 5. Recommended Branch Usage

Recommended git branch split:

- `demo-current-pipeline-results`
  use this when you want to show a stable snapshot to your boss
- `feature-snmp-v2-interface-roadmap`
  use this for roadmap implementation and experimentation

Recommended artifact split:

- official current baseline:
  `f1_baseline_v1`
- future demo artifact example:
  `boss_demo_approved_v1`
- future experiment artifact example:
  `v2_f2_interface_health_v1`

This gives two layers of safety:
- branch keeps code stable
- artifact directory keeps model and scaler stable

## 6. Run The Current Validated F1 Baseline

Run these commands in order from the project root:

```bash
python3 main.py generate-data
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
python3 main.py train --artifact-dir-name f1_baseline_v1
python3 main.py detect --artifact-dir-name f1_baseline_v1
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

This does the following:
1. generates the richer synthetic SNMP dataset
2. derives rate features and saves arrays plus scaler
3. trains the LSTM autoencoder on the `F1` feature set
4. runs anomaly detection using the same validated artifact set
5. saves the `P1` time split and baseline metrics report

## 7. Run Each Step Separately

### A. Generate synthetic dataset

```bash
python3 main.py generate-data
```

What this step does:
- creates a richer synthetic SNMP-like dataset
- uses cumulative counters instead of per-row raw traffic values
- includes interface-level metadata and anomaly context

Output:
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`

### B. Build `P0` verification reports

Dataset inventory:

```bash
python3 -m snmp_anomaly_detection.data.dataset_inventory --output-file snmp_anomaly_detection/outputs/p0_dataset_inventory.json
```

Data quality audit:

```bash
python3 -m snmp_anomaly_detection.data.data_quality_audit --output-file snmp_anomaly_detection/outputs/p0_data_quality_audit.json
```

What these steps do:
- confirm dataset shape and required columns
- confirm interface readiness
- verify duplicate, gap, and negative-delta behavior

Outputs:
- `snmp_anomaly_detection/outputs/p0_dataset_inventory.json`
- `snmp_anomaly_detection/outputs/p0_data_quality_audit.json`

### C. Preprocess dataset for `F1`

```bash
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
```

Optional `T2` preprocessing examples:

```bash
python3 main.py preprocess --artifact-dir-name t2_standard_v1 --scaler-name standard
python3 main.py preprocess --artifact-dir-name t2_robust_v1 --scaler-name robust
python3 main.py preprocess --artifact-dir-name t2_log1p_standard_v1 --scaler-name standard --log1p-features in_rate out_rate error_rate
```

What this step does:
- loads the dataset
- derives `in_rate`, `out_rate`, and `error_rate`
- builds a repeatable time-based train, validation, and test split
- filters normal rows for training and test artifacts
- skips invalid warm-up and reset rows
- fits the scaler on train-period normal rows only
- can use `minmax`, `standard`, or `robust` scaling
- can optionally apply `log1p` to selected heavy-tailed features before scaling
- applies the saved scaler to both train and test periods
- creates sequence windows per `device_id + interface` inside each split boundary
- saves train/test arrays, scaler, and preprocessing metadata into the selected artifact directory

Outputs:
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/X_train.npy`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/X_test.npy`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/y_train.npy`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/y_test.npy`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/scaler.pkl`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/preprocessing_metadata.json`

### D. Train the `F1` model

```bash
python3 main.py train --artifact-dir-name f1_baseline_v1
```

What this step does:
- loads training arrays from the selected artifact directory
- trains the LSTM autoencoder
- computes the anomaly threshold from reconstruction error
- saves model and metadata into the same artifact directory

Outputs:
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/lstm_autoencoder.pth`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/model_metadata.json`

### E. Run anomaly detection

```bash
python3 main.py detect --artifact-dir-name f1_baseline_v1
```

Explicit CSV replay mode:

```bash
python3 main.py detect-csv --artifact-dir-name f1_baseline_v1
```

What this step does:
- loads the dataset
- loads scaler, model, and metadata from the selected artifact directory
- replays events through interface-scoped windowing
- derives rate features online using elapsed-time logic
- scores anomaly windows
- saves output files

Outputs:
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`

Current validated result shape includes:
- `device_id`
- `interface`
- `stream_id`
- `top_error_feature`
- `feature_error_in_rate`
- `feature_error_out_rate`
- `feature_error_error_rate`

### F. Run `P1` baseline evaluation

```bash
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

What this step does:
- replays the dataset using the selected artifact set
- defines a repeatable time-based split for the current dataset
- computes baseline window-level metrics from replay output
- saves per-split and per-interface reporting artifacts for future comparison

Outputs:
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_time_split.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_baseline_metrics.json`

Current saved `P1` split for `f1_baseline_v1`:
- train:
  `2025-01-01 00:00:00` to `2025-01-05 20:35:00`
- validation:
  `2025-01-05 20:40:00` to `2025-01-06 21:35:00`
- test:
  `2025-01-06 21:40:00` to `2025-01-07 22:35:00`

Current saved `P1` summary for `f1_baseline_v1`:
- windows:
  `29850`
- actual anomaly windows:
  `277`
- predicted anomaly windows:
  `1821`
- precision:
  `0.0923`
- recall:
  `0.6065`
- false positive rate:
  `0.0559`

### G. Run Kafka dry ingestion

```bash
python3 main.py detect-kafka-dry
```

Notes:
- consumes Kafka messages from the hardcoded topic
- validates and normalizes payloads
- does not perform anomaly scoring
- stop with `Ctrl+C`

### H. Publish Kafka test data

```bash
python3 main.py produce-kafka-test-data
python3 main.py produce-kafka-test-data --device-count 20 --sleep-seconds 0.05
python3 main.py produce-kafka-test-data --max-messages 200 --anomaly-probability 0.10
```

Notes:
- the live synthetic Kafka producer now also includes `interface`
- payloads use cumulative counters, matching the `F1` online-rate logic better than before

### I. Run Kafka live detection

```bash
python3 main.py detect-kafka --artifact-dir-name f1_baseline_v1
```

Notes:
- live scoring loads model and scaler from the selected artifact directory
- live scoring uses the same elapsed-time rate logic as CSV replay
- local JSONL outputs are written to:
  `snmp_anomaly_detection/outputs/kafka_live_results.jsonl`
  `snmp_anomaly_detection/outputs/kafka_live_anomaly_windows.jsonl`
- stop with `Ctrl+C`

## 8. Minimal Detection-Only Flow

If the dataset, scaler, and model are already available, you only need:

```bash
python3 main.py detect --artifact-dir-name f1_baseline_v1
```

Required existing files inside that artifact directory:
- `scaler.pkl`
- `lstm_autoencoder.pth`
- `model_metadata.json`

Supported dataset lookup order:
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- `synthetic_snmp_dataset.csv`
- `featureEngineering/synthetic_snmp_dataset.csv`

## 9. Inspect The Current Validated Baseline

Check the active feature set and threshold:

```bash
python3 - <<'PY'
import json
meta = json.load(open('snmp_anomaly_detection/artifacts/f1_baseline_v1/model_metadata.json'))
print('feature_columns:', meta['feature_config']['feature_columns'])
print('threshold:', meta['threshold'])
PY
```

Check the latest anomaly detection summary:

```bash
python3 - <<'PY'
import pandas as pd
results = pd.read_csv('snmp_anomaly_detection/outputs/anomaly_results.csv')
print('total_windows:', len(results))
print('predicted_anomalies:', int(results['predicted_anomaly'].sum()))
print('columns:', results.columns.tolist())
PY
```

## 10. Demo-Safe Workflow

If you need a stable demo for your boss:

1. Switch to the demo branch:

```bash
git switch demo-current-pipeline-results
```

2. Create or reuse a dedicated demo artifact directory:

```bash
python3 main.py preprocess --artifact-dir-name boss_demo_approved_v1
python3 main.py train --artifact-dir-name boss_demo_approved_v1
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

3. For later demos, rerun only detection if the artifact set is already frozen:

```bash
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

Do not overwrite `f1_baseline_v1` if you want to preserve the validated `F1` baseline exactly as-is.

## 11. Quick Verification

To verify the Python files compile:

```bash
python3 -m compileall snmp_anomaly_detection
```

To verify the full current `F1` baseline flow:

```bash
python3 main.py generate-data
python3 -m snmp_anomaly_detection.data.dataset_inventory --output-file snmp_anomaly_detection/outputs/p0_dataset_inventory.json
python3 -m snmp_anomaly_detection.data.data_quality_audit --output-file snmp_anomaly_detection/outputs/p0_data_quality_audit.json
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
python3 main.py train --artifact-dir-name f1_baseline_v1
python3 main.py detect --artifact-dir-name f1_baseline_v1
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

If detection completes successfully, check:

- `snmp_anomaly_detection/artifacts/f1_baseline_v1/`
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_time_split.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_baseline_metrics.json`
