# Running the Project

This document explains how to run the current SNMP anomaly detection pipeline after `P0`, `F1`, `P2`, `F2`, `F3`, `P3`, `C1`, `C2`, and `C3` implementation.

The current recommended artifacts are:

- official rich-feature baseline:
  `f3_t3_seq10_p995_v1`
- stable simple baseline:
  `f1_baseline_v1`
- low-noise rich-feature option:
  `f3_t3_seq10_p999_v1`

The official rich-feature baseline uses:
- richer synthetic SNMP-like dataset
- cumulative counters
- interface-scoped windows
- `F3` feature profile with `35` features
- threshold mode:
  percentile `99.5`
- recommended artifact directory:
  `f3_t3_seq10_p995_v1`

Keep `f1_baseline_v1` when you need the simpler historical `F1` baseline for comparison.

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
python3 main.py report-p2
python3 main.py preprocess
python3 main.py train
python3 main.py train-iforest
python3 main.py evaluate-baseline
python3 main.py evaluate-iforest
python3 main.py compare-artifacts
python3 main.py report-p3
python3 main.py normalize-events
python3 main.py correlate-events
python3 main.py explain-anomalies
python3 main.py detect
python3 main.py detect-iforest
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

The stable simple `F1` baseline trains on:

- `cpu`
- `memory`
- `in_rate`
- `out_rate`
- `error_rate`

These are derived during preprocessing from cumulative counters using elapsed time between polls.

The official rich-feature baseline, `f3_t3_seq10_p995_v1`, uses the `F3` feature profile. It includes the `F2` interface-capacity and health features plus rolling context features for traffic, error, and utilization behavior.

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
- `t1_candidate_v1`
  first `T1` synthetic candidate reviewed against the baseline
- `t2_standard_v1`
  first `T2` StandardScaler candidate
- `t2_robust_v1`
  first `T2` RobustScaler candidate
- `t2_log1p_standard_v1`
  first `T2` log1p plus StandardScaler candidate
- `t2_log1p_robust_v1`
  first `T2` log1p plus RobustScaler candidate
- `f2_candidate_v1`
  first `F2` candidate using richer interface-capacity and health features
- `f2_t2_standard_v1`
  `F2` candidate with StandardScaler
- `f2_t2_robust_v1`
  `F2` candidate with RobustScaler
- `f2_t2_log1p_standard_v1`
  `F2` candidate with log1p plus StandardScaler
- `f2_t2_log1p_robust_v1`
  `F2` candidate with log1p plus RobustScaler
- `f3_candidate_v1`
  first contextual rolling-feature candidate
- `f3_t3_seq10_p995_v1`
  official recommended rich-feature baseline
- `f3_t3_seq10_p999_v1`
  stricter low-noise rich-feature option

Rule:
- use the same artifact directory name for `preprocess`, `train`, and `detect` when they belong to the same run

## 5. Recommended Branch Usage

Recommended git branch split:

- `demo-current-pipeline-results`
  use this when you want to show a stable snapshot to your boss
- `feature-snmp-v2-interface-roadmap`
  use this for roadmap implementation and experimentation

Recommended artifact split:

- official rich-feature baseline:
  `f3_t3_seq10_p995_v1`
- stable simple baseline:
  `f1_baseline_v1`
- low-noise rich-feature option:
  `f3_t3_seq10_p999_v1`
- future demo artifact example:
  `boss_demo_approved_v1`
- future experiment artifact example:
  `v2_f2_interface_health_v1`

This gives two layers of safety:
- branch keeps code stable
- artifact directory keeps model and scaler stable

## 6. Run The Current Recommended Rich-Feature Baseline

Run these commands in order from the project root:

```bash
python3 main.py generate-data
python3 main.py preprocess --artifact-dir-name f3_t3_seq10_p995_v1 --feature-profile f3
python3 main.py train --artifact-dir-name f3_t3_seq10_p995_v1 --threshold-mode percentile --threshold-percentile 99.5
python3 main.py detect --artifact-dir-name f3_t3_seq10_p995_v1
python3 main.py evaluate-baseline --artifact-dir-name f3_t3_seq10_p995_v1
python3 main.py report-p3
python3 main.py normalize-events
python3 main.py correlate-events
python3 main.py explain-anomalies
```

This does the following:
1. generates the richer synthetic SNMP dataset
2. derives rate, interface-health, and contextual rolling features
3. trains the LSTM autoencoder on the `F3` feature set
4. runs anomaly detection using the same rich-feature artifact set
5. saves the `P1` time split and baseline metrics report
6. prepares the first event source for correlation
7. normalizes event samples into the shared C1 schema
8. correlates anomaly windows with nearby events
9. adds deterministic C3 explanations to correlated anomaly windows

To rerun the stable simple baseline instead, use:

```bash
python3 main.py preprocess --artifact-dir-name f1_baseline_v1
python3 main.py train --artifact-dir-name f1_baseline_v1
python3 main.py detect --artifact-dir-name f1_baseline_v1
python3 main.py evaluate-baseline --artifact-dir-name f1_baseline_v1
```

Optional candidate comparison, once candidate `P1` metrics reports already exist:

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1
```

Current rich-feature comparison:

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name f3_t3_seq10_p995_v1 --candidate-artifact-dir-name f3_t3_seq10_p999_v1
```

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
python3 main.py preprocess --artifact-dir-name t2_log1p_robust_v1 --scaler-name robust --log1p-features in_rate out_rate error_rate
```

Optional `F2` preprocessing examples:

```bash
python3 main.py preprocess --artifact-dir-name f2_candidate_v1 --feature-profile f2
python3 main.py preprocess --artifact-dir-name f2_t2_standard_v1 --feature-profile f2 --scaler-name standard
python3 main.py preprocess --artifact-dir-name f2_t2_robust_v1 --feature-profile f2 --scaler-name robust
python3 main.py preprocess --artifact-dir-name f2_t2_log1p_standard_v1 --feature-profile f2 --scaler-name standard --log1p-features in_rate out_rate error_rate packet_rate_in packet_rate_out discard_rate_in discard_rate_out
python3 main.py preprocess --artifact-dir-name f2_t2_log1p_robust_v1 --feature-profile f2 --scaler-name robust --log1p-features in_rate out_rate error_rate packet_rate_in packet_rate_out discard_rate_in discard_rate_out
```

What this step does:
- loads the dataset
- derives `in_rate`, `out_rate`, and `error_rate`
- can also derive `F2` features such as utilization, packet/discard rates, `in_out_ratio`, and interface state values
- builds a repeatable time-based train, validation, and test split
- filters normal rows for training and test artifacts
- skips invalid warm-up and reset rows
- fits the scaler on train-period normal rows only
- supports named feature profiles such as `baseline`, `f2`, and `f3`
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

### D. Run `P2` feature readiness reporting

```bash
python3 main.py report-p2
```

What this step does:
- scans the dataset for the richer interface-level source fields needed before `F2`
- builds a capability matrix for each `device_id + interface` stream
- documents whether `per-interface` can remain the main analysis scope
- writes the extended feature schema note needed for `F2` and `F3`

Outputs:
- `snmp_anomaly_detection/outputs/p2_interface_capability_matrix.json`
- `snmp_anomaly_detection/outputs/p2_extended_feature_schema.json`

Current saved `P2` result:
- all `15` streams have full source-field coverage
- all `15` devices currently recommend `per_interface`
- current live schema additions required for `F2`:
  `none`

### E. Train the `F1` model

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

### F. Run anomaly detection

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

### G. Run `P1` baseline evaluation

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

### H. Run `P1.1` artifact comparison

Run this after candidate artifacts already have saved `P1` metrics reports.
The comparison step reads the existing `*_p1_baseline_metrics.json` files and does not rerun inference.

If you need to recreate candidate `P1` reports first, use the same artifact name across `preprocess`, `train`, and `evaluate-baseline`.
For example:

```bash
python3 main.py preprocess --artifact-dir-name t1_candidate_v1
python3 main.py train --artifact-dir-name t1_candidate_v1
python3 main.py evaluate-baseline --artifact-dir-name t1_candidate_v1

python3 main.py preprocess --artifact-dir-name t2_standard_v1 --scaler-name standard
python3 main.py train --artifact-dir-name t2_standard_v1
python3 main.py evaluate-baseline --artifact-dir-name t2_standard_v1

python3 main.py preprocess --artifact-dir-name t2_robust_v1 --scaler-name robust
python3 main.py train --artifact-dir-name t2_robust_v1
python3 main.py evaluate-baseline --artifact-dir-name t2_robust_v1

python3 main.py preprocess --artifact-dir-name t2_log1p_standard_v1 --scaler-name standard --log1p-features in_rate out_rate error_rate
python3 main.py train --artifact-dir-name t2_log1p_standard_v1
python3 main.py evaluate-baseline --artifact-dir-name t2_log1p_standard_v1

python3 main.py preprocess --artifact-dir-name t2_log1p_robust_v1 --scaler-name robust --log1p-features in_rate out_rate error_rate
python3 main.py train --artifact-dir-name t2_log1p_robust_v1
python3 main.py evaluate-baseline --artifact-dir-name t2_log1p_robust_v1
```

Current note:
- T2 candidate preprocessing metadata records scaler and `log1p` choices correctly
- T2 `model_metadata.json` now mirrors those preprocessing choices
- the current comparison report should have no scaler or `log1p` metadata consistency warnings

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1
```

Shorter example for comparing only one candidate:

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1
```

What this step does:
- loads the saved `P1` metrics report for the baseline artifact
- loads the saved `P1` metrics report for each candidate artifact
- computes candidate-minus-baseline deltas for overall metrics
- computes candidate-minus-baseline deltas for train, validation, and test splits
- includes saved top-interface false-positive regression and improvement highlights
- checks whether model metadata agrees with preprocessing metadata for scaler and `log1p` choices
- writes a repeatable `P1.1` comparison report

Output:
- `snmp_anomaly_detection/outputs/f1_baseline_v1_vs_5_candidates_p1_1_comparison.json`

How to read the report:
- positive precision delta is better
- positive recall delta is better
- negative false positive rate delta is better
- `recommendation: keep_baseline` means the candidate should not replace `f1_baseline_v1`
- metadata warnings mean a candidate should not be trusted as a clean experiment artifact until fixed

Current saved `P1.1` result:
- `t1_candidate_v1`:
  precision delta `-0.0027`, recall delta `-0.0036`, false positive rate delta `+0.0015`, recommendation `keep_baseline`
- `t2_standard_v1`:
  precision delta `-0.0080`, recall delta `-0.0217`, false positive rate delta `+0.0036`, recommendation `keep_baseline`
- `t2_robust_v1`:
  precision delta `-0.0115`, recall delta `-0.0469`, false positive rate delta `+0.0038`, recommendation `keep_baseline`
- `t2_log1p_standard_v1`:
  precision delta `-0.0157`, recall delta `-0.0361`, false positive rate delta `+0.0086`, recommendation `keep_baseline`
- `t2_log1p_robust_v1`:
  precision delta `-0.0103`, recall delta `-0.0036`, false positive rate delta `+0.0074`, recommendation `keep_baseline`

Current decision:
- keep `f1_baseline_v1` as the stable simple baseline for historical comparison
- use `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline
- T2 metadata propagation is fixed for scaler and `log1p` tracking
- continue later experiments against the saved `P1` comparison workflow

### I. Run first `F2` candidate flow

Run this after `P2` if you want to test the richer interface-capacity and health feature set.

```bash
python3 main.py preprocess --artifact-dir-name f2_candidate_v1 --feature-profile f2
python3 main.py train --artifact-dir-name f2_candidate_v1
python3 main.py evaluate-baseline --artifact-dir-name f2_candidate_v1
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name f2_candidate_v1
```

What this step does:
- switches preprocessing from the `F1` baseline feature set to the richer `F2` feature profile
- trains and evaluates one first `F2` candidate artifact
- compares it directly against `f1_baseline_v1`

Current saved `F2` candidate result:
- `f2_candidate_v1`:
  precision `0.0890`, recall `0.6137`, false positive rate `0.0589`
- compared with `f1_baseline_v1`:
  precision delta `-0.0033`, recall delta `+0.0072`, false positive rate delta `+0.0030`
- current recommendation:
  `review_tradeoff`

Current saved `F2 + T2` comparison result:
- `f2_t2_standard_v1`:
  recommendation `keep_baseline`
- `f2_t2_robust_v1`:
  recommendation `keep_baseline`
- `f2_t2_log1p_standard_v1`:
  recommendation `review_tradeoff`
- `f2_t2_log1p_robust_v1`:
  recommendation `review_tradeoff`

### J. Run `P3` event-source readiness reporting

Run this after the dataset exists. It prepares the first event source for later C1/C2/C3 correlation work.

```bash
python3 main.py report-p3
```

What this step does:
- selects SNMP traps / interface-state events as the first correlation source
- writes the normalized event schema used by C1
- checks timestamp alignment assumptions
- generates a synthetic sample event JSONL file from the current SNMP dataset

Outputs:
- `snmp_anomaly_detection/outputs/p3_event_source_selection.json`
- `snmp_anomaly_detection/outputs/p3_normalized_event_schema.json`
- `snmp_anomaly_detection/outputs/p3_timestamp_alignment_notes.json`
- `snmp_anomaly_detection/outputs/p3_sample_normalized_events.jsonl`

### K. Run `C1` event normalization

Run this after `P3`.

```bash
python3 main.py normalize-events
```

Optional real event input:

```bash
python3 main.py normalize-events --input-file path/to/events.jsonl
python3 main.py normalize-events --input-file path/to/events.csv
python3 main.py normalize-events --input-file path/to/events.json
```

What this step does:
- reads CSV, JSON, or JSONL event records
- normalizes event fields into the shared `P3` schema
- maps source event names into standard event types
- rejects malformed timestamps and missing `device_id` records safely
- preserves `interface` when available and falls back to `per_device` scope when missing

Outputs:
- `snmp_anomaly_detection/outputs/c1_normalized_events.jsonl`
- `snmp_anomaly_detection/outputs/c1_rejected_events.jsonl`
- `snmp_anomaly_detection/outputs/c1_normalization_summary.json`

### L. Run `C2` event correlation

Run this after anomaly detection and C1 normalization.

```bash
python3 main.py correlate-events
```

Optional explicit inputs:

```bash
python3 main.py correlate-events --anomaly-windows-file snmp_anomaly_detection/outputs/anomaly_windows.json --normalized-events-file snmp_anomaly_detection/outputs/c1_normalized_events.jsonl
```

What this step does:
- reads anomaly windows from detection
- reads normalized C1 events
- matches events around anomaly `window_end`
- uses a deterministic `10` minute lookback and `5` minute lookahead window
- ranks matches by device/interface match, proximity, severity, and top-feature relationship

Outputs:
- `snmp_anomaly_detection/outputs/c2_correlated_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c2_correlation_summary.json`

Current saved C2 result:
- anomaly windows:
  `1581`
- anomalies with correlated events:
  `254`
- anomalies without correlated events:
  `1327`
- selected correlated events:
  `329`

### M. Run `C3` anomaly explanation

Run this after C2.

```bash
python3 main.py explain-anomalies
```

Optional explicit input:

```bash
python3 main.py explain-anomalies --correlated-anomaly-windows-file snmp_anomaly_detection/outputs/c2_correlated_anomaly_windows.json
```

What this step does:
- reads C2 correlated anomaly windows
- adds one deterministic `c3_explanation` object per anomaly window
- produces "likely related" explanations when a correlated event exists
- produces model-only explanations when no event matched the C2 window
- preserves the raw C2 evidence alongside the explanation

Outputs:
- `snmp_anomaly_detection/outputs/c3_explained_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c3_explanation_summary.json`

Current saved C3 result:
- explained anomaly windows:
  `1581`
- event-backed explanations:
  `254`
- model-only explanations:
  `1327`
- high-confidence explanations:
  `142`
- medium-confidence explanations:
  `89`

### N. Run `T4` Isolation Forest baseline

Run this after the source feature/preprocessing artifact exists. The current recommended source artifact is `f3_t3_seq10_p995_v1`.

```bash
python3 main.py train-iforest --artifact-dir-name iforest_f3_v1 --source-artifact-dir-name f3_t3_seq10_p995_v1
python3 main.py detect-iforest --artifact-dir-name iforest_f3_v1
python3 main.py evaluate-iforest --artifact-dir-name iforest_f3_v1
```

What this step does:
- trains Isolation Forest on the same F3 engineered sequence windows used by the LSTM baseline
- flattens each `sequence_length x feature_count` window into one tabular vector
- saves a standalone Isolation Forest artifact
- replays the CSV input through the same feature/window construction path
- writes standalone scoring output and P1-style metrics

Outputs:
- `snmp_anomaly_detection/artifacts/iforest_f3_v1/isolation_forest.pkl`
- `snmp_anomaly_detection/artifacts/iforest_f3_v1/isolation_forest_metadata.json`
- `snmp_anomaly_detection/outputs/iforest_f3_v1_iforest_results.csv`
- `snmp_anomaly_detection/outputs/iforest_f3_v1_iforest_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/iforest_f3_v1_iforest_p1_time_split.json`
- `snmp_anomaly_detection/outputs/iforest_f3_v1_iforest_p1_baseline_metrics.json`

Current saved T4 result:
- source artifact:
  `f3_t3_seq10_p995_v1`
- Isolation Forest artifact:
  `iforest_f3_v1`
- training windows:
  `20648`
- evaluated windows:
  `29850`
- predicted anomaly windows:
  `811`
- precision:
  `0.0210`
- recall:
  `0.0614`
- false positive rate:
  `0.0268`

Current interpretation:
- Isolation Forest is much quieter than the LSTM baseline
- it misses most labeled anomalies in the current synthetic dataset
- do not replace `f3_t3_seq10_p995_v1` with `iforest_f3_v1`
- do not build an LSTM + Isolation Forest ensemble until more T4 tuning or comparison is done

### O. Run Kafka dry ingestion

```bash
python3 main.py detect-kafka-dry
```

Notes:
- consumes Kafka messages from the hardcoded topic
- validates and normalizes payloads
- does not perform anomaly scoring
- stop with `Ctrl+C`

### P. Publish Kafka test data

```bash
python3 main.py produce-kafka-test-data
python3 main.py produce-kafka-test-data --device-count 20 --sleep-seconds 0.05
python3 main.py produce-kafka-test-data --max-messages 200 --anomaly-probability 0.10
```

Notes:
- the live synthetic Kafka producer now also includes `interface`
- the live synthetic Kafka producer now also includes packet counters, discard counters, interface speed, and interface status
- payloads use cumulative counters, matching the `F1` online-rate logic better than before

### Q. Run Kafka live detection

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
python3 main.py report-p3
python3 main.py normalize-events
python3 main.py correlate-events
python3 main.py explain-anomalies
```

If detection and explanation complete successfully, check:

- `snmp_anomaly_detection/artifacts/f1_baseline_v1/`
- `snmp_anomaly_detection/outputs/anomaly_results.csv`
- `snmp_anomaly_detection/outputs/anomaly_windows.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_time_split.json`
- `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_baseline_metrics.json`
- `snmp_anomaly_detection/outputs/c1_normalized_events.jsonl`
- `snmp_anomaly_detection/outputs/c2_correlated_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c3_explained_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c3_explanation_summary.json`

If candidate `P1` metrics reports already exist, also verify the comparison report:

```bash
python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1
```

Expected comparison output:

- `snmp_anomaly_detection/outputs/f1_baseline_v1_vs_5_candidates_p1_1_comparison.json`
