# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SNMP anomaly detection pipeline for network and power equipment. Two parallel pipelines share the same codebase:
- **Network pipeline** (main branch): 5-metric detection on routers/switches/firewalls.
- **Power pipeline** (feature-power-snmp-detection): dual-model detection on UPS/PDU/network PSUs/env sensors, plus battery RUL forecasting.

All commands are run from the repo root as `python3 snmp_anomaly_detection/main.py <command>` (or equivalently `python3 -m snmp_anomaly_detection <command>` from within the package).

## Common Commands

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Syntax check (no pytest setup)
python3 -m compileall snmp_anomaly_detection

# Power pipeline — run in order
python3 snmp_anomaly_detection/main.py generate-power-data
python3 snmp_anomaly_detection/main.py preprocess-power
python3 snmp_anomaly_detection/main.py train-power-baseline
python3 snmp_anomaly_detection/main.py train-power-phase
python3 snmp_anomaly_detection/main.py train-battery-rul
python3 snmp_anomaly_detection/main.py detect-power-csv
python3 snmp_anomaly_detection/main.py detect-power-kafka     # requires local Kafka

# Network pipeline
python3 snmp_anomaly_detection/main.py generate-data
python3 snmp_anomaly_detection/main.py preprocess
python3 snmp_anomaly_detection/main.py train
python3 snmp_anomaly_detection/main.py detect-csv
python3 snmp_anomaly_detection/main.py detect-kafka-dry       # validates parsing without models
```

## Architecture

### Data Flow (Power Pipeline)

```
synthetic_power_snmp_dataset.csv   (40 devices × 2000 rows)
         ↓  data/dataset_builder.py
power_features.py                  log1p → delta features → RobustScaler → sequences
         ↓
  ┌──────────────────────┬──────────────────────┐
  │  Baseline model      │  Phase model          │
  │  18 BASELINE_UPS_    │  29 PHASE_LEVEL_      │
  │  FEATURES            │  FEATURES             │
  │  LSTM AE (seq=20)    │  LSTM AE (seq=20)     │
  └──────────┬───────────┴──────────┬────────────┘
             └──────────┬───────────┘
             dual_model_scorer.py    OR policy + overload rule
                        ↓
             anomaly_results.csv / kafka_live_results.jsonl
```

The phase model's feature set is a strict **superset** of the baseline set. If you add/reorder features in either list in `config.py`, both models must be retrained together.

### Key Source Files

| File | Role |
|---|---|
| `config.py` | Single source of truth for all feature lists, hyperparameter dataclasses, device categories, and artifact paths |
| `main.py` | CLI dispatcher — maps command strings to pipeline functions |
| `data/dataset_builder.py` | Synthetic data generation with realistic fault injection per device category |
| `preprocessing/power_features.py` | Feature engineering: log1p transforms, delta (rate-of-change) features, per-category scaling |
| `training/train_baseline_power.py` | Trains `baseline_scaler.pkl` + `baseline_metadata.json` + baseline LSTM |
| `training/train_phase_model.py` | Trains `phase_scaler.pkl` + `phase_metadata.json` + phase LSTM |
| `training/train_battery_rul.py` | Trains `rul_model.pt` — regression LSTM with MC Dropout for confidence intervals |
| `inference/dual_model_scorer.py` | Loads both models + scalers, scores batches, applies OR/AND policy + overload rule |
| `inference/power_stream_processor.py` | Per-device rolling window manager for Kafka micro-batching |
| `streaming/detect_power_kafka.py` | KafkaConsumer loop feeding `power_stream_processor` |

### Model Architecture

**LSTM Autoencoder** (both baseline and phase models):
- Encoder: `LSTM(input_size → hidden=64)` → `Linear(hidden → latent=32)`
- Decoder: `LSTM(latent → hidden=64)` → `Linear(hidden → input_size)`
- Anomaly threshold: **99.9th percentile** of per-category training reconstruction errors (not mean+kσ — dataset has heavy-tailed distributions)
- Thresholds stored per device category in `*_metadata.json`

**BatteryRULModel**: 2-layer LSTM with `dropout=0.1` → `Linear(hidden → 1)`, outputs days until battery replacement.

### Fault Injection Constraints

- Fault events use **additive injection** on normal baselines (not multiplicative).
- Overload events require the final `output_load_pct` to stay ≤ 100% (except for intentional overload fault type which exceeds it).
- `phase_sag` faults require ≥ 4 three-phase devices in the dataset; single-phase-only configs will silently skip this fault type.
- `output_frequency_hz` is excluded from MSE for PDU/network/env device categories — grid noise creates false positives. UPS devices retain it.

### Kafka Details

- Bootstrap servers: `localhost:9092` (hardcoded in streaming modules).
- Topics: `snmp-power-events` (power), `snmp-events` (network).
- Micro-batch flushes at `batch_size=8` windows OR `max_wait_ms=50` ms, whichever comes first.
- `detect-kafka-dry` consumes and parses without loading models — use to validate topic/message format.

### Artifacts Layout

Model artifacts are written to `snmp_anomaly_detection/outputs/` by default (path overridable via `config.py`). Experiment snapshots accumulate in `snmp_anomaly_detection/artifacts/` — these are not auto-cleaned.

## Critical Invariants

- **Delta features are computed on log-scaled values**, not raw. `runtime_delta` is the difference of `log1p(runtime_remaining_min)`, not of the raw minutes. Maintain this order in `power_features.py`.
- **Time-based train/test split** — no shuffle anywhere in the pipeline. Test set is always the most recent 15% of rows per device to prevent temporal leakage.
- **Torch is a lazy import** — `preprocessing/` and `data/` modules must not import torch at module level. Only `training/` and `inference/` modules import it.
