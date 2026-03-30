# Running the Project

This document explains how to run each module separately and how to run the full SNMP anomaly detection pipeline end to end.

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
