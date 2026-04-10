# SNMP Anomaly Detection Demo

Demo date: 2026-04-07

## 1. Opening Summary

Today we are showing the current SNMP anomaly detection pipeline that we have built so far.

In simple words, the project can now:

- generate a synthetic SNMP-like dataset
- validate dataset quality before training
- derive SNMP rate-based features from cumulative counters
- train an LSTM autoencoder model
- detect anomalous SNMP windows
- evaluate baseline performance
- compare multiple candidate artifacts against the validated baseline
- support offline detection, CSV replay, and Kafka-oriented live detection entry points

Current validated artifact:

```text
f1_baseline_v1
```

Current active model features:

```text
cpu, memory, in_rate, out_rate, error_rate
```

Current model:

```text
LSTM autoencoder
```

## 2. What We Have Achieved

### P0: Dataset Readiness

We first prepared and validated the dataset before training.

What we achieved:

- created a richer synthetic SNMP dataset
- added device-level and interface-level fields
- used cumulative counters instead of only direct traffic values
- added counters such as `in_octets`, `out_octets`, `errors`, packet counters, and discard counters
- added interface information such as `interface`, speed, admin status, and operational status
- added anomaly metadata and counter reset information
- built dataset inventory and data quality audit reports

Evidence from the current dataset inventory:

- rows: `30000`
- columns: `20`
- devices: `15`
- interface count: `4`
- timeline: `2025-01-01 00:00:00` to `2025-01-07 22:35:00`
- per-interface ready devices: `15`
- per-device fallback devices: `0`

Evidence from the current data quality audit:

- duplicate rows: `0`
- duplicate keys: `0`
- timestamp parse failures: `0`
- out-of-order groups: `0`
- polling gap groups: `0`
- negative counter deltas: `0`

Demo point:

```text
We do not start model training blindly. We first confirm the synthetic SNMP data is usable for interface-level anomaly detection.
```

### F1: Derived-Rate Feature Baseline

After dataset validation, we improved the feature flow.

What we achieved:

- derived `in_rate` from cumulative `in_octets`
- derived `out_rate` from cumulative `out_octets`
- derived `error_rate` from cumulative `errors`
- calculated rates per `device_id + interface`
- used elapsed poll time between SNMP samples
- skipped invalid warm-up and reset rows
- created interface-scoped sequence windows
- trained on the derived feature set:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`

Demo point:

```text
This makes the model closer to real SNMP behavior because counters are cumulative in real SNMP systems.
```

### Model Training And Detection

We trained an LSTM autoencoder for anomaly detection.

Current model metadata:

- input size: `5`
- hidden size: `64`
- latent size: `32`
- sequence length: `10`
- epochs: `20`
- learning rate: `0.001`
- threshold: `0.014308651676401496`

How it works:

- the model learns normal SNMP behavior from normal windows
- at detection time, the model reconstructs the input window
- high reconstruction error means the window is suspicious
- the threshold is based on reconstruction error from normal training windows

Demo point:

```text
The model is not using a fixed rule like CPU > 90. It learns normal multi-feature behavior and flags windows whose pattern is unusual.
```

### P1: Baseline Evaluation

We evaluated the validated baseline artifact `f1_baseline_v1`.

Overall result:

| Metric | Value |
| --- | ---: |
| Windows evaluated | `29850` |
| Actual anomalies | `277` |
| Predicted anomalies | `1821` |
| True positives | `168` |
| False positives | `1653` |
| False negatives | `109` |
| Precision | `0.0923` |
| Recall | `0.6065` |
| False positive rate | `0.0559` |

Test split result:

| Metric | Value |
| --- | ---: |
| Test windows | `4500` |
| Actual anomalies | `44` |
| Predicted anomalies | `198` |
| True positives | `19` |
| False positives | `179` |
| False negatives | `25` |
| Precision | `0.0960` |
| Recall | `0.4318` |
| False positive rate | `0.0402` |

Demo point:

```text
The baseline is useful as a working starting point. It catches many anomalies, but precision and false positives still need improvement.
```

### P1.1: Candidate Comparison

We also compared candidate artifacts against the current baseline.

Compared artifacts:

- `t1_candidate_v1`
- `t2_standard_v1`
- `t2_robust_v1`
- `t2_log1p_standard_v1`
- `t2_log1p_robust_v1`

Current conclusion:

```text
None of the first candidate artifacts clearly replaces f1_baseline_v1 yet.
```

Why:

- several candidates increased predicted anomalies and false positives
- some candidates reduced recall
- `t2_log1p_robust_v1` improved test recall, but also increased false positives and false positive rate
- therefore, `f1_baseline_v1` remains the validated artifact for the demo

Short comparison summary:

| Artifact | Overall Precision | Overall Recall | Overall FPR | Test Recall | Test FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f1_baseline_v1` | `0.0923` | `0.6065` | `0.0559` | `0.4318` | `0.0402` |
| `t1_candidate_v1` | `0.0896` | `0.6029` | `0.0574` | `0.4091` | `0.0415` |
| `t2_standard_v1` | `0.0843` | `0.5848` | `0.0595` | `0.4091` | `0.0478` |
| `t2_robust_v1` | `0.0808` | `0.5596` | `0.0596` | `0.3864` | `0.0550` |
| `t2_log1p_standard_v1` | `0.0765` | `0.5704` | `0.0645` | `0.4318` | `0.0554` |
| `t2_log1p_robust_v1` | `0.0819` | `0.6029` | `0.0633` | `0.4773` | `0.0568` |

Demo point:

```text
We now have repeatable comparison reports, so we can test improvements safely instead of guessing which model is better.
```

### Difference Between Artifacts

Use this section if someone asks why we have multiple artifact folders.

Short answer:

```text
Each artifact directory is one saved experiment. It contains the scaler, train/test arrays, trained model, metadata, and evaluation outputs for that specific run.
```

What stays mostly the same across these artifacts:

- same dataset source:
  `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- same active feature set:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`
- same sequence length:
  `10`
- same model type:
  LSTM autoencoder
- same model size:
  input size `5`, hidden size `64`, latent size `32`
- same training epoch count:
  `20`
- same time split style:
  train `70%`, validation `15%`, test `15%`

What changes between artifacts:

| Artifact | Purpose | Main Difference | Current Decision |
| --- | --- | --- | --- |
| `f1_baseline_v1` | Validated baseline | First stable derived-rate baseline, using the default preprocessing/scaling path | Keep as current demo baseline |
| `t1_candidate_v1` | T1 candidate | Tests cleaner time-based preprocessing and training/evaluation metadata against the baseline | Did not beat baseline |
| `t2_standard_v1` | T2 scaler candidate | Uses `StandardScaler` during preprocessing | Did not beat baseline |
| `t2_robust_v1` | T2 scaler candidate | Uses `RobustScaler` during preprocessing | Did not beat baseline |
| `t2_log1p_standard_v1` | T2 transform plus scaler candidate | Applies `log1p` to `in_rate`, `out_rate`, `error_rate`, then uses `StandardScaler` | Did not beat baseline |
| `t2_log1p_robust_v1` | T2 transform plus scaler candidate | Applies `log1p` to `in_rate`, `out_rate`, `error_rate`, then uses `RobustScaler` | Did not beat baseline |

How to explain `f1_baseline_v1`:

```text
This is our stable reference artifact. We compare all new experiments against it. For today's demo, this is the safest artifact to use.
```

How to explain `t1_candidate_v1`:

```text
This candidate tests the T1 direction: cleaner time-based training and evaluation. It added preprocessing metadata and explicit split boundaries, but its metrics were slightly worse than the baseline, so we did not promote it.
```

How to explain `t2_*` artifacts:

```text
These candidates test preprocessing choices. Standard scaling, robust scaling, and log1p transformations were tried because SNMP traffic rates can be skewed or bursty. The first T2 experiments did not improve the overall tradeoff enough to replace the baseline.
```

Important metadata note:

```text
The T2 preprocessing metadata records the actual scaler and log1p choices, but some T2 model metadata still shows old/default values. This is already identified as a cleanup task before we treat T2 artifacts as fully clean experiment records.
```

Simple decision summary:

```text
We keep f1_baseline_v1 for the demo because it has the best current overall balance. The candidate artifacts are useful evidence that we are experimenting in a controlled way, but they are not promoted yet.
```

## 3. Demo Flow

Use this order during the demo.

### Step 1: Go To Project Directory

```bash
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation
```

Explain:

```text
This is the main project directory. All commands are run from here.
```

### Step 2: Activate Environment

```bash
source .venv/bin/activate
```

If the environment is not created yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Explain:

```text
The virtual environment keeps Python dependencies isolated for this project.
```

### Step 3: Show Available Pipeline Commands

```bash
python3 main.py --help
```

Mention the important steps:

- `generate-data`
- `preprocess`
- `train`
- `detect`
- `evaluate-baseline`
- `compare-artifacts`
- `detect-csv`
- `detect-kafka-dry`
- `detect-kafka`

Explain:

```text
The project now has one main CLI entry point, and each pipeline stage is available as a command.
```

### Step 4: Generate Synthetic SNMP Dataset

```bash
python3 main.py generate-data
```

Output:

```text
snmp_anomaly_detection/data/synthetic_snmp_dataset.csv
```

Explain:

```text
This creates a synthetic SNMP-like dataset with devices, interfaces, cumulative counters, traffic counters, interface status, and anomaly labels.
```

### Step 5: Run Dataset Inventory

```bash
python3 -m snmp_anomaly_detection.data.dataset_inventory --output-file snmp_anomaly_detection/outputs/p0_dataset_inventory.json
```

Output:

```text
snmp_anomaly_detection/outputs/p0_dataset_inventory.json
```

Explain:

```text
This confirms the dataset shape, available columns, device count, interface readiness, and whether the data is ready for per-interface modeling.
```

### Step 6: Run Data Quality Audit

```bash
python3 -m snmp_anomaly_detection.data.data_quality_audit --output-file snmp_anomaly_detection/outputs/p0_data_quality_audit.json
```

Output:

```text
snmp_anomaly_detection/outputs/p0_data_quality_audit.json
```

Explain:

```text
This checks duplicates, timestamp quality, polling gaps, ordering problems, missing required fields, and negative counter deltas.
```

### Step 7: Preprocess Data For Baseline

```bash
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
```

Output directory:

```text
snmp_anomaly_detection/artifacts/f1_baseline_v1/
```

Important files:

- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `scaler.pkl`

Explain:

```text
Preprocessing derives rate features, scales the feature values, creates train and test windows, and saves repeatable artifacts.
```

### Step 8: Train The LSTM Autoencoder

```bash
python3 main.py train --artifact-dir-name f1_baseline_v1
```

Important outputs:

- `snmp_anomaly_detection/artifacts/f1_baseline_v1/lstm_autoencoder.pth`
- `snmp_anomaly_detection/artifacts/f1_baseline_v1/model_metadata.json`

Explain:

```text
Training creates the LSTM autoencoder model and saves the anomaly threshold and model configuration.
```

### Step 9: Run Detection

```bash
python3 main.py detect --artifact-dir-name f1_baseline_v1
```

Important outputs:

- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`

Explain:

```text
Detection loads the saved scaler and trained model, scores the windows, and writes anomaly results.
```

### Step 10: Evaluate Baseline

```bash
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

Important outputs:

- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_baseline_metrics.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_time_split.json`

Explain:

```text
Evaluation reports train, validation, and test performance using the saved baseline artifact.
```

### Step 11: Compare Candidate Artifacts

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1
```

Important output:

```text
snmp_anomaly_detection/outputs/f1_baseline_v1_vs_5_candidates_p1_1_comparison.json
```

Explain:

```text
This compares each candidate against the stable baseline and shows whether precision, recall, false positives, and false positive rate improved or regressed.
```

### Step 12: Show Kafka Live Features

Kafka support is now part of the project CLI.

Kafka commands:

- `detect-kafka-dry`
- `detect-kafka`
- `produce-kafka-test-data`

Current Kafka topic:

```text
snmp-live-events
```

Current Kafka bootstrap server default:

```text
localhost:9092
```

Current Kafka message schema:

- `timestamp`
- `device_id`
- `interface`
- `cpu`
- `memory`
- `in_octets`
- `out_octets`
- `errors`
- optional `anomaly`
- optional `anomaly_type`

What we achieved for Kafka:

- added Kafka consumer support
- added Kafka test-data producer support
- added input validation for live JSON messages
- added normalization from Kafka payloads into the shared event format
- reused the same `EventProcessor`, window manager, and live micro-batching logic
- added Kafka dry ingestion to verify connectivity without model scoring
- added Kafka live detection to score windows using the saved model artifact
- added local JSONL outputs for live Kafka results and anomaly windows

Kafka dry ingestion flow:

```text
Kafka topic
-> consume JSON message
-> validate required fields
-> normalize into event shape
-> print per-device message counts
```

Kafka live detection flow:

```text
Kafka topic
-> consume JSON message
-> validate and normalize
-> derive live rate features
-> build rolling windows per device/interface
-> micro-batch ready windows
-> score with LSTM autoencoder
-> write local live results
```

Current micro-batching defaults:

- batch size: `8`
- max wait: `50 ms`

Live Kafka output files:

- `snmp_anomaly_detection/outputs/kafka_live_results.jsonl`
- `snmp_anomaly_detection/outputs/kafka_live_anomaly_windows.jsonl`

Safe Kafka demo option:

```bash
python3 main.py detect-kafka-dry
```

In another terminal:

```bash
python3 main.py produce-kafka-test-data --max-messages 50 --sleep-seconds 0.1 --anomaly-probability 0.05 --seed 7
```

Explain:

```text
This verifies that the project can receive Kafka messages, validate the schema, normalize them, and keep device-level message counts without running model scoring.
```

Full Kafka scoring option:

```bash
python3 main.py detect-kafka --artifact-dir-name f1_baseline_v1
```

In another terminal:

```bash
python3 main.py produce-kafka-test-data --max-messages 100 --sleep-seconds 0.1 --anomaly-probability 0.05 --seed 7
```

Explain:

```text
This uses the validated f1_baseline_v1 model and scaler to score live Kafka events. Results are printed and also saved locally as JSONL files.
```

Important Kafka demo note:

```text
Kafka requires a local Kafka broker running at localhost:9092. If Kafka is not running during the demo, use the CSV/offline flow and explain that the Kafka code path is implemented but depends on broker availability.
```

## 4. Suggested Demo Script

Use this simple speaking script.

```text
We started by building a synthetic SNMP anomaly detection pipeline.

The first milestone was data readiness. We created a richer SNMP-like dataset with devices, interfaces, cumulative counters, interface status, and anomaly metadata. Then we added P0 checks to verify dataset quality before training.

Next, we improved the feature pipeline. Instead of directly depending on raw cumulative counters, we derive rate-based features such as in_rate, out_rate, and error_rate. This is important because real SNMP counters are cumulative, so rate calculation is closer to real network monitoring behavior.

After preprocessing, we create interface-scoped windows and train an LSTM autoencoder. The model learns normal SNMP behavior and flags windows that have high reconstruction error.

Our current validated artifact is f1_baseline_v1. It uses five features: cpu, memory, in_rate, out_rate, and error_rate.

We also added evaluation and artifact comparison. This means every experiment can now be compared against the same baseline using saved metrics instead of manually guessing.

We also added Kafka support for live-style SNMP events. The dry Kafka command verifies message consumption and schema normalization, while the live Kafka command can use the saved model artifact to score rolling windows from Kafka input.

The baseline currently detects a meaningful number of anomalies, but it also has false positives. So the next work is focused on improving metadata tracking, threshold and window tuning, adding richer interface health features, and eventually correlating anomalies with operational events.
```

## 5. Current Limitations

Be transparent about what is not finished yet:

- the current dataset is synthetic, not real production SNMP data
- real known-normal and maintenance labels are not yet available
- richer interface health fields exist in the data but are not all active in the model yet
- precision is still low, so false positives need improvement
- T2 metadata propagation needs cleanup so scaler and `log1p` choices are always recorded consistently
- Kafka live demo requires a running Kafka broker at `localhost:9092`
- Kafka output topics are not implemented yet; current live Kafka results are printed and saved locally
- event correlation is not implemented yet

Suggested wording:

```text
This is a working baseline, not the final production model. The key achievement is that the full loop now exists: data generation, validation, preprocessing, training, detection, evaluation, and experiment comparison.
```

## 6. What We Are Going To Do Further

Recommended next steps:

1. Fix T2 metadata propagation
2. Run threshold and windowing experiments against `f1_baseline_v1`
3. Improve the false-positive and recall tradeoff
4. Add richer interface health features such as utilization, discards, packets, status changes, and reset context
5. Add real known-normal and maintenance labels when available
6. Keep per-interface modeling as the main direction and per-device as fallback
7. Improve CSV replay and Kafka live scoring demos
8. Add event correlation so anomalies can be explained with nearby operational events
9. Add Kafka output topics after the first live Kafka scoring path is stable

Future goal:

```text
Move from only "this SNMP window is abnormal" toward "this interface is abnormal, this is the likely reason, and these nearby events may explain it."
```

## 7. Safe Demo Command List

If time is short, use only these commands:

```bash
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation
source .venv/bin/activate
python3 main.py --help
python3 main.py detect --artifact-dir-name f1_baseline_v1
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1
```

Why this is safe:

```text
It uses the already validated baseline artifact and focuses the demo on current results instead of retraining live.
```

Optional Kafka demo, if Kafka is running:

```bash
python3 main.py detect-kafka-dry
python3 main.py produce-kafka-test-data --max-messages 50 --sleep-seconds 0.1 --anomaly-probability 0.05 --seed 7
```

## 8. Full Rebuild Command List

If you want to show the complete pipeline from scratch:

```bash
cd /home/aircod/gitlab/SNMP-DATA/Dataset_creation
source .venv/bin/activate
python3 main.py generate-data
python3 -m snmp_anomaly_detection.data.dataset_inventory --output-file snmp_anomaly_detection/outputs/p0_dataset_inventory.json
python3 -m snmp_anomaly_detection.data.data_quality_audit --output-file snmp_anomaly_detection/outputs/p0_data_quality_audit.json
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
python3 main.py train --artifact-dir-name f1_baseline_v1
python3 main.py detect --artifact-dir-name f1_baseline_v1
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

Note:

```text
For a live demo, prefer the safe command list unless there is enough time to retrain.
```
