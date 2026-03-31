# Running the Project

This document explains how to run the SNMP anomaly detection pipeline after the move to versioned artifact directories.

The main idea is:
- code version is controlled by the git branch
- model, scaler, and training arrays are controlled by the artifact directory

That separation is important because it lets you:
- keep a stable demo artifact set for presentations
- continue `v2` work without overwriting the demo model

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
python3 main.py detect
python3 main.py detect-csv
python3 main.py detect-kafka-dry
python3 main.py detect-kafka
python3 main.py produce-kafka-test-data
```

## 3. Artifact Directory Concept

Preprocessing and training artifacts are now saved in:

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

Examples of good artifact names:

- `boss_demo_approved_v1`
- `demo_trained_pipeline_v1`
- `baseline_preprocess_ready_v1`
- `v2_f1_rate_features_v1`
- `v2_t1_clean_training_v1`

Rule:
- use the same artifact directory name for `preprocess`, `train`, and `detect` when they belong to the same run

## 4. Recommended Branch Usage

Recommended git branch split:

- `demo-current-pipeline-results`
  use this when you want to show the current stable pipeline to your boss
- `feature-snmp-v2-interface-roadmap`
  use this for roadmap implementation and experiments

Recommended artifact split:

- demo artifact example: `boss_demo_approved_v1`
- experiment artifact example: `v2_f1_rate_features_v1`

This gives two layers of safety:
- branch keeps demo code stable
- artifact directory keeps demo model and scaler stable

## 5. Run the Full Baseline Pipeline

Run these commands in order from the project root:

```bash
python3 main.py generate-data
python3 main.py preprocess --artifact-dir-name boss_demo_approved_v1
python3 main.py train --artifact-dir-name boss_demo_approved_v1
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

This does the following:
1. generate synthetic SNMP data
2. preprocess it and save arrays plus scaler
3. train the LSTM autoencoder and save model plus metadata
4. run anomaly detection using that exact saved artifact set

## 6. Run Each Step Separately

### A. Generate synthetic dataset

```bash
python3 main.py generate-data
```

Output:
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`

### B. Preprocess dataset

```bash
python3 main.py preprocess --artifact-dir-name boss_demo_approved_v1
```

What this step does:
- loads the dataset
- keeps normal rows for training
- scales feature columns
- creates sequence windows
- saves train/test arrays and scaler into the selected artifact directory

Outputs:
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/X_train.npy`
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/X_test.npy`
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/y_train.npy`
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/y_test.npy`
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/scaler.pkl`

### C. Train the model

```bash
python3 main.py train --artifact-dir-name boss_demo_approved_v1
```

What this step does:
- loads training arrays from the selected artifact directory
- trains the LSTM autoencoder
- computes the anomaly threshold
- saves model and metadata into the same artifact directory

Outputs:
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/lstm_autoencoder.pth`
- `snmp_anomaly_detection/artifacts/boss_demo_approved_v1/model_metadata.json`

### D. Run anomaly detection

```bash
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

Explicit CSV replay mode:

```bash
python3 main.py detect-csv --artifact-dir-name boss_demo_approved_v1
```

What this step does:
- loads the dataset
- loads scaler, model, and metadata from the selected artifact directory
- creates detection windows
- scores each window
- saves anomaly detection outputs

Outputs:
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`

### E. Run Kafka dry ingestion

```bash
python3 main.py detect-kafka-dry
```

Notes:
- consumes Kafka messages from the hardcoded topic
- validates and normalizes payloads
- does not perform anomaly scoring
- stop with `Ctrl+C`

### F. Publish Kafka test data

```bash
python3 main.py produce-kafka-test-data
python3 main.py produce-kafka-test-data --device-count 20 --sleep-seconds 0.05
python3 main.py produce-kafka-test-data --max-messages 200 --anomaly-probability 0.10
```

### G. Run Kafka live detection

```bash
python3 main.py detect-kafka --artifact-dir-name boss_demo_approved_v1
```

Notes:
- live scoring loads model and scaler from the selected artifact directory
- local JSONL outputs are still written to:
  `snmp_anomaly_detection/outputs/kafka_live_results.jsonl`
  `snmp_anomaly_detection/outputs/kafka_live_anomaly_windows.jsonl`
- stop with `Ctrl+C`

## 7. Minimal Detection-Only Flow

If the dataset, scaler, and model are already available, you only need:

```bash
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

Required existing files inside that artifact directory:
- `scaler.pkl`
- `lstm_autoencoder.pth`
- `model_metadata.json`

Supported dataset lookup order:
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- `synthetic_snmp_dataset.csv`
- `featureEngineering/synthetic_snmp_dataset.csv`

## 8. Demo-Safe Workflow

If you need a stable demo for your boss:

1. Switch to the demo branch:

```bash
git switch demo-current-pipeline-results
```

2. Run or reuse the demo artifact set:

```bash
python3 main.py preprocess --artifact-dir-name boss_demo_approved_v1
python3 main.py train --artifact-dir-name boss_demo_approved_v1
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

3. For later demos, rerun only detection if the artifact set is already frozen:

```bash
python3 main.py detect --artifact-dir-name boss_demo_approved_v1
```

This prevents retraining in `v2` work from silently changing the demo model.

## 9. Experiment Workflow For V2

When doing roadmap work, use a new artifact name for each meaningful milestone.

Example:

```bash
python3 main.py preprocess --artifact-dir-name v2_f1_rate_features_v1
python3 main.py train --artifact-dir-name v2_f1_rate_features_v1
python3 main.py detect --artifact-dir-name v2_f1_rate_features_v1
```

Do not overwrite a demo artifact directory with experiment runs.

## 10. Important Output Locations

Main locations:

- dataset:
  `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- artifact directories:
  `snmp_anomaly_detection/artifacts/`
- detection CSV:
  `snmp_anomaly_detection/outputs/anomaly_results.csv`
- anomaly windows JSON:
  `snmp_anomaly_detection/outputs/anomaly_windows.json`
- Kafka live results:
  `snmp_anomaly_detection/outputs/kafka_live_results.jsonl`
- Kafka live anomaly windows:
  `snmp_anomaly_detection/outputs/kafka_live_anomaly_windows.jsonl`

## 11. Quick Verification

To verify the Python files compile:

```bash
python3 -m compileall snmp_anomaly_detection
```

To verify the full baseline flow:

```bash
python3 main.py generate-data
python3 main.py preprocess --artifact-dir-name verification_run_v1
python3 main.py train --artifact-dir-name verification_run_v1
python3 main.py detect --artifact-dir-name verification_run_v1
```

If detection completes successfully, check:

- `snmp_anomaly_detection/artifacts/verification_run_v1/`
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`
