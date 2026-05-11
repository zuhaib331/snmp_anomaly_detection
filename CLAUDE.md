# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SNMP anomaly detection pipeline for network and power equipment. Two parallel pipelines share the same codebase:
- **Network pipeline** (main branch): 5-metric detection on routers/switches/firewalls.
- **Power pipeline** (feature-power-snmp-detection): triple-model detection (2× LSTM AE + Isolation Forest) on UPS/PDU/network PSUs/env sensors, plus battery RUL forecasting and a two-stage alert lifecycle.

All commands are run from the repo root as `python3 -m snmp_anomaly_detection <command>`.

---

## Common Commands

```bash
# Environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Syntax check (no pytest setup)
python3 -m compileall snmp_anomaly_detection

# Power pipeline — run in order
python3 -m snmp_anomaly_detection generate-power-data
python3 -m snmp_anomaly_detection preprocess-power
python3 -m snmp_anomaly_detection train-power-baseline
python3 -m snmp_anomaly_detection train-power-phase
python3 -m snmp_anomaly_detection train-power-iforest
python3 -m snmp_anomaly_detection train-battery-rul
python3 -m snmp_anomaly_detection detect-power-csv
python3 -m snmp_anomaly_detection detect-power-kafka     # requires local Kafka

# Network pipeline
python3 -m snmp_anomaly_detection generate-data
python3 -m snmp_anomaly_detection preprocess
python3 -m snmp_anomaly_detection train
python3 -m snmp_anomaly_detection detect-csv
python3 -m snmp_anomaly_detection detect-kafka-dry       # validates parsing without models
```

---

## Architecture

### Data Flow (Power Pipeline)

```
synthetic_power_snmp_dataset.csv   (40 devices × 2000 rows)
         ↓  data/dataset_builder.py
         ↓
preprocessing/scalar_transforms.py  ← single source of truth for all feature math
         ↓  (called by pandas wrappers for batch AND by EventPreprocessor for streaming)
power_features.py:
  aggregate_phase_metrics → normalize_absolute_features → apply_log1p_skewed
  → add_delta_features (+ volt_drop_deltas B3) → RobustScaler → sequences
phase_features.py:
  same + enrich_imbalance_features (NEMA MG-1 voltage_imbalance_pct, current_skew_pct)
         ↓
  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │  Baseline model      │  Phase model          │  Isolation Forest    │
  │  15 BASELINE_UPS_    │  29 PHASE_LEVEL_      │  per-category        │
  │  FEATURES            │  FEATURES             │  feature masks       │
  │  LSTM AE (seq=20)    │  LSTM AE (seq=20)     │  200 estimators      │
  │  UPS+PDU+network+env │  UPS only             │  score_samples min   │
  └──────────┬───────────┴──────────┬────────────┴──────────┬───────────┘
             └──────────────────────┴──────────────────────-─┘
                     dual_model_scorer.py (CSV) OR
                     power_stream_processor.py (Kafka — two-stage alert lifecycle)
                                   ↓
                    anomaly_results.csv / kafka_live_results.jsonl
```

- Phase model's feature set is a strict **superset** of baseline. Adding/reordering in `config.py` requires retraining both models together.
- IForest is independent of sequence length — scores flat feature vectors per row with per-category feature masks.

---

## Module Map

### `config.py` — single source of truth

All feature lists, hyperparameter dataclasses, capability registries, and artifact paths live here.

| Symbol | What it is |
|---|---|
| `BASELINE_UPS_FEATURES` | 15 features fed to the baseline LSTM AE (all device categories) |
| `PHASE_LEVEL_FEATURES` | 29 features fed to the phase LSTM AE (UPS only); superset of baseline |
| `BATTERY_RUL_FEATURES` | 7 features fed to the BatteryRULModel |
| `BASELINE_MODEL_CATEGORIES` | `{"ups", "pdu", "network", "env"}` |
| `PHASE_MODEL_CATEGORIES` | `{"ups"}` |
| `BATTERY_RUL_CATEGORIES` | `{"ups"}` |
| `IF_EXCLUDED_FEATURES` | Per-category dict of features to mask from IForest (battery zeros + freq noise) |
| `IF_MIN_SCORE_RATIO` | `1.05` — minimum `score / threshold` ratio to raise an IF flag |
| `ProjectPaths` | All artifact file paths; overridable at runtime |
| `PowerTrainingConfig` | LSTM hyperparams (hidden=64, latent=32, seq_len=20, epochs=50) |
| `BatteryRULConfig` | RUL hyperparams (dropout=0.3, mc_samples=50, hidden=64, layers=2) |

### `data/dataset_builder.py` — synthetic data generation

Generates `synthetic_power_snmp_dataset.csv` with 40 devices across 4 categories and 2000 rows each. Injects 8 realistic fault types per device category using **additive injection** on normal baselines:

| Category | Devices | Fault types |
|---|---|---|
| UPS | 16 | battery_drain, overload, phase_sag (3-phase only), battery_replace, overheating, input_spike, on_battery, charger_fail |
| PDU | 8 | overload, input_spike, output_fault |
| Network | 10 | power_fluctuation, overload |
| Env | 6 | power_loss, overload |

Device registration fields written to every CSV row: `rated_capacity_w`, `nominal_voltage_v`, `rated_battery_v`. These drive B2 vendor-agnostic normalization.

`phase_sag` faults require ≥ 4 three-phase devices in the dataset; single-phase configs silently skip this fault type.

### `preprocessing/scalar_transforms.py` — F11: canonical feature math (NEW)

**Single source of truth** for all feature-engineering logic. Pure functions, numpy only, no pandas. Operates on plain Python dicts so any transport (Kafka, MQTT, ZMQ, CSV) can call the same functions.

| Function | What it does |
|---|---|
| `aggregate_phase_voltages(fv)` | Recomputes `input_voltage_v = mean(L1, L2, L3)` — must run before `normalize_absolute` |
| `normalize_absolute(fv, cap, nomv, bv)` | Replaces absolute V/A/W with vendor-agnostic ratios (B2) |
| `apply_log1p(fv)` | log1p on `runtime_remaining_min` — must run before `compute_deltas` |
| `compute_deltas(fv, prev)` | Signed log1p deltas for runtime, temperature, load |
| `compute_volt_drop_deltas(fv, prev)` | Negative-only voltage diffs (B3) — amplifies phase sag signal |
| `compute_imbalance(fv)` | NEMA MG-1 `voltage_imbalance_pct` and `current_skew_pct` |

Key constants exported: `PHASE_V_COLS`, `PHASE_I_COLS`, `LOG1P_COLS`, `DELTA_PAIRS`, `VOLT_DROP_PAIRS`, `ALL_DELTA_SOURCES`, `IMBALANCE_COLS`, `IMBALANCE_CLIP_MAX`.

Pandas wrappers in `power_features.py` and `phase_features.py` call or mirror these functions for batch training. The Kafka streaming path (`power_stream_processor._apply_preprocessing`) duplicates this math — **that duplication is the outstanding gap** (F12 → F13 will close it by introducing `EventPreprocessor`).

### `preprocessing/power_features.py` — baseline batch preprocessing

Pandas batch path for the baseline model:
1. `aggregate_phase_metrics` — mean of L1/L2/L3 → `input_voltage_v`
2. `normalize_absolute_features` — B2 ratio/deviation features, drops registration columns
3. `apply_log1p_skewed` — log1p on `runtime_remaining_min`
4. `add_delta_features` — signed log1p deltas + B3 voltage drop deltas
5. `scale_features` — RobustScaler, saved to `baseline_scaler.pkl`
6. `build_baseline_sequences` — per-device sliding windows (seq_len=20)

### `preprocessing/phase_features.py` — phase batch preprocessing

Superset of power_features, adds:
- `enrich_imbalance_features` — recomputes `voltage_imbalance_pct`/`current_skew_pct` using NEMA MG-1 formula from raw phase columns
- `scale_phase_features` — clips imbalance cols to `[0, IMBALANCE_CLIP_MAX]` before RobustScaler (near-zero IQR guard)
- `build_phase_sequences` — same sliding window logic for the 29-feature phase set

Backward-compatible private aliases `_IMBALANCE_COLS` and `_IMBALANCE_CLIP_MAX` remain in this file until F13 removes them.

### `preprocessing/battery_features.py` — RUL label generation

Computes `rul_days` label for each UPS row using `install_age_days` from the device profile as the starting offset (F4 fix). Approximates `discharge_cycles_approx` from charge/discharge state transitions. Filters to `BATTERY_RUL_CATEGORIES` using the config registry (F9).

### `training/train_baseline_power.py`

Trains the baseline LSTM AE:
1. Loads dataset, runs full preprocessing pipeline (aggregate → normalize → log1p → delta → scale)
2. Per-device temporal split: 70% train / 15% val / 15% test (no shuffle — temporal order preserved)
3. Validation scaled with training scaler (F2 fix — no leakage)
4. Threshold: **99.9th percentile** per device category from training reconstruction errors (not mean+kσ)
5. Saves: `baseline_model.pt`, `baseline_scaler.pkl`, `baseline_metadata.json`

`baseline_metadata.json` schema:
```json
{
  "per_category_thresholds": {"ups": 0.338, "pdu": ..., ...},
  "feature_columns": [...],
  "seq_len": 20,
  "non_ups_mse_exclude": ["output_frequency_hz"]
}
```

`output_frequency_hz` is excluded from MSE for PDU/network/env categories — grid noise produces false positives. UPS retains it.

### `training/train_phase_model.py`

Same pattern as baseline but operates on `PHASE_LEVEL_FEATURES` (29 features, UPS only). Saves `phase_model.pt`, `phase_scaler.pkl`, `phase_metadata.json`.

### `training/train_iforest_power.py` — E1/E2: Isolation Forest

Trains one `IsolationForest` per device category on normal rows only:
- Feature masks per category from `IF_EXCLUDED_FEATURES` (excludes battery zeros and freq noise for non-UPS)
- `n_estimators=200`, `contamination="auto"`, `random_state=42`
- Threshold: 1st percentile of training scores per category
- Saves: `iforest_models.pkl` (dict: category → model), `iforest_metadata.json` (thresholds + feature lists per category)

### `training/train_battery_rul.py`

Trains `BatteryRULModel` (2-layer LSTM + dropout + linear):
- Input: `BATTERY_RUL_FEATURES` (7 features), `seq_len=20`
- Per-device temporal split (F10 fix) so test set covers all 5 UPS devices' final windows
- MC Dropout inference: `model.train()` mode, 50 samples, mean ± 1.96σ = 95% CI
- `dropout=0.3` (raised from 0.1 in F8 to get usable CI variance)
- Saves: `rul_model.pt`, `rul_metadata.json`

### `inference/dual_model_scorer.py` — CSV inference path

Batch inference for CSV replay. Loads all three models + scalers, runs full pandas preprocessing pipeline, then scores per-device windows:

| Step | What happens |
|---|---|
| Preprocessing | aggregate → normalize → log1p → delta → enrich_imbalance → scale |
| IF pre-score | `score_samples` on all rows per device in one batch call (E5), takes window min (E3) |
| Baseline LSTM | `_score_window_detailed`: reconstructs window, computes per-category MSE excluding `output_frequency_hz` for non-UPS |
| Phase LSTM | Only for `PHASE_MODEL_CATEGORIES`; skipped for PDU/network/env (F9) |
| Alert policy | OR policy: flag if any model flags; overload rule: `output_load_pct > 98` always flags |
| IF guard | IF flag only raised if `abs(score) / abs(threshold) >= IF_MIN_SCORE_RATIO` (E4) |

Output: `anomaly_results.csv` + `anomaly_windows_detail.json` + `detection_summary.json`.

`DualModelResult` fields: `device_id`, `device_category`, `timestamp`, `baseline_error`, `baseline_anomaly`, `phase_error`, `phase_anomaly`, `iforest_score`, `iforest_flag`, `overload_flag`, `final_flag`, `true_label`, `baseline_top_features`, `phase_top_features`, `if_peak_timestep`.

### `inference/power_stream_processor.py` — Kafka inference path

Per-device stateful streaming processor. Key classes:

**`PowerDeviceWindowManager`**: rolling deque buffer per device, yields `(seq_len, n_features)` numpy array when full.

**`DeviceErrorStats`**: Welford online stats for per-device adaptive thresholding (B1). Cold-start: first 50 windows collected silently. Threshold = `mean + k×std` from calibration errors, falling back to per-category → global threshold.

**`AlertState`** (E7): `SUSPECTED | CONFIRMED | LSTM_ONLY | CLEARED`

**`PendingAlert`** (E7): tracks IF fires awaiting LSTM confirmation. Cleared after `N_CONFIRMATION_WINDOWS` (default 3) without confirmation.

**`PowerStreamProcessor.process_event()`** flow (E6 + E7):
1. Preprocess event via `_apply_preprocessing()` (hand-rolled — **pending replacement by F12/F13**)
2. IF scores current row immediately — no buffer needed (E6 decoupling)
3. Push row to rolling buffer
4. If buffer not full: emit `SUSPECTED` result if IF fired, else `None`
5. If buffer full: LSTM scores window; combine with pending IF result; emit `CONFIRMED`/`CLEARED`/`LSTM_ONLY`

Preprocessing is delegated to `EventPreprocessor` (F12/F13). Scoring is delegated to `DualModelScorer` (F14). Rolling window buffers use `RollingWindowBuffer` (F14).

### `inference/events.py` — `PowerEvent` dataclass

Frozen dataclass carrying one SNMP event: `device_id`, `device_category`, `timestamp`, `feature_values: dict`, `rated_capacity_w`, `nominal_voltage_v`, `rated_battery_v`, `anomaly`. Methods: `get_baseline_vector()`, `get_phase_vector()`.

### `models/lstm_autoencoder.py` — LSTM Autoencoder

```
Encoder: LSTM(input_size → hidden=64) → Linear(hidden → latent=32)
Decoder: LSTM(latent → hidden=64) → Linear(hidden → input_size)
```

Reconstruction MSE per feature, summed (with category-specific exclusions) → anomaly score.

### `models/battery_rul.py` — BatteryRULModel

```
2× LSTM layers(hidden=64, dropout=0.3) → Linear(64 → 1)
```

Outputs `rul_days` (regression). MC Dropout used at inference: `model.train()` + 50 forward passes → mean prediction + 95% CI.

### `evaluation/power_eval.py` — detection metrics

Computes precision, recall, F1, FP rate from `anomaly_results.csv`. Reads `detection_summary.json` for per-device breakdown.

### `evaluation/rul_eval.py` — RUL metrics

MAE, `ci_coverage_95pct` (fraction of ground-truth labels inside predicted 95% CI). Filters to `BATTERY_RUL_CATEGORIES` via config registry (F7/F9 fix — no longer uses `battery_voltage_v > 0` proxy).

### `streaming/detect_power_kafka.py` — Kafka consumer loop

`KafkaConsumer` on topic `snmp-power-events`, feeds `PowerStreamProcessor`. Micro-batch: flushes at `batch_size=8` windows OR `max_wait_ms=50` ms. Writes scored results to `kafka_live_results.jsonl`.

### `streaming/produce_power_kafka_test.py` — test data producer

Replays `synthetic_power_snmp_dataset.csv` into Kafka. Supports `--anomaly-probability` flag and `--use-training-profiles` to reproduce exact training conditions for calibration testing.

### `data/vendor_oid_map.py` — MIB analysis reference

Documents OID paths and unit scales for APC, Liebert GP, Liebert UPS, and RFC 1628 MIBs. Used as reference for A1 (OID adapter layer). Key unit-conversion facts:
- RFC 1628: `battery_voltage_v` in 0.1 V units (÷10), `runtime_remaining_min` in TimeTicks for APC (÷6000)
- APC: `rated_capacity_w` in kVA (×1000)
- `rated_battery_v` has no RFC 1628 OID — requires manual entry or vendor-proprietary MIB

---

## Feature Engineering Detail

### BASELINE_UPS_FEATURES (15 features)
```
battery_charge_pct, battery_voltage_ratio, battery_current_ratio,
battery_temperature_c, runtime_remaining_min, on_battery_status,
battery_replace_status, input_voltage_dev_pct, input_frequency_hz,
output_voltage_dev_pct, output_current_ratio, output_load_pct,
output_frequency_hz, runtime_delta, temperature_delta, output_load_delta
```
Note: `battery_charge_delta` was removed — charge changes ~0.02%/step are noise after RobustScaler and inflated the UPS threshold to 43× (removed in B2). `output_power_w` dropped — redundant with `output_load_pct`.

### PHASE_LEVEL_FEATURES (29 features)
All 15 baseline features plus:
```
input_voltage_l1/l2/l3, input_current_l1/l2/l3, output_current_l1/l2/l3,
voltage_imbalance_pct, current_skew_pct,
voltage_drop_delta_l1/l2/l3  (B3 — negative-only voltage diffs)
```

### Preprocessing order (must be maintained)
1. `aggregate_phase_voltages` — mean L1/L2/L3 → `input_voltage_v`
2. `normalize_absolute` — compute ratios/deviations from registration fields, drop registration columns
3. `apply_log1p` — compress `runtime_remaining_min`
4. `compute_deltas` — signed log1p deltas (on log-scaled values — **critical invariant**)
5. `compute_volt_drop_deltas` — B3 negative-only voltage diffs
6. `compute_imbalance` — NEMA MG-1 voltage/current imbalance

---

## Alert Lifecycle (E7)

```
IF fires at row N:
  → PendingAlert stored for device
  → ScoringResult(alert_state=SUSPECTED) emitted immediately  ← fast early warning

LSTM scores window ending at row N+k (k ≤ seq_len):
  → IF pending? lstm_flag?
      YES + YES → CONFIRMED (high-confidence, both models agreed)
      YES + NO  → CLEARED (auto-resolved, IF was FP or transient)
      NO  + YES → LSTM_ONLY (gradual drift, no early warning)

Timeout: PendingAlert older than N_CONFIRMATION_WINDOWS → CLEARED
```

| alert_state | Meaning | NOC action |
|---|---|---|
| `suspected` | IF fired, LSTM pending | Early warning (amber) |
| `confirmed` | Both models agree | High-confidence alarm (red) |
| `lstm_only` | LSTM flagged, IF missed | Gradual drift alarm (orange) |
| `cleared` | IF fired, LSTM retracted | Auto-resolve (grey) |

---

## Artifact Layout

```
snmp_anomaly_detection/outputs/
  power_dual/
    baseline_model.pt           # LSTM AE weights
    baseline_scaler.pkl         # RobustScaler fitted on training data
    baseline_metadata.json      # per-category thresholds, feature_columns, seq_len
    phase_model.pt
    phase_scaler.pkl
    phase_metadata.json
    iforest_models.pkl          # dict[category → IsolationForest]
    iforest_metadata.json       # per-category thresholds + feature_cols
    anomaly_results.csv         # per-window scores (CSV inference)
    anomaly_windows_detail.json # top contributing features per window
    detection_summary.json      # per-device precision/recall/FP breakdown
    kafka_live_results.jsonl    # streaming inference output
  device_stats/                 # B1 Welford per-device calibration state (JSON per device)
  battery_rul/
    rul_model.pt
    rul_metadata.json
    rul_predictions.json
    rul_metrics.json
  power_phase/
    X_train_phase.npy
    phase_scaler.pkl
    preprocess_phase_meta.json
  X_train.npy                   # baseline sequences
  preprocess_meta.json
snmp_anomaly_detection/artifacts/  # experiment snapshots (not auto-cleaned)
```

---

## Kafka Details

- Bootstrap servers: `localhost:9092` (hardcoded in streaming modules)
- Topics: `snmp-power-events` (power), `snmp-events` (network)
- Micro-batch: `batch_size=8` windows OR `max_wait_ms=50` ms, whichever first
- `detect-kafka-dry` consumes and parses without loading models — validates topic/message format

---

## Critical Invariants

- **Delta features are computed on log-scaled values**, not raw. `runtime_delta` is the diff of `log1p(runtime_remaining_min)`, not of raw minutes. Step 3 (log1p) must always precede step 4 (deltas).
- **Time-based train/test split everywhere** — no shuffle. Test set is always the most recent rows per device to prevent temporal leakage.
- **Torch is a lazy import** — `preprocessing/` and `data/` modules must not import torch at module level. Only `training/` and `inference/` modules import it.
- **Additive fault injection** — faults are added to normal baselines; overload fault type is the only one allowed to push `output_load_pct > 100`.
- **Capability registry gates model routing** — use `PHASE_MODEL_CATEGORIES`, `BATTERY_RUL_CATEGORIES` from `config.py`, never hardcode category strings in inference code.
- **Threshold: 99.9th percentile** per category, not mean+kσ — the error distribution is heavy-tailed.
- **`output_frequency_hz` excluded from MSE** for PDU/network/env. UPS keeps it. Same exclusion applied to IF via `IF_EXCLUDED_FEATURES`.
- **scalar_transforms.py is the canonical math** — the pandas wrappers in `power_features.py` and `phase_features.py` must stay in sync with it. If you change a formula in one place, update both.

---

## Features Built (Chronological)

| ID | Feature | Status |
|---|---|---|
| F1 | Fix `threshold_std_multiplier` regression in network pipeline | Done 2026-04-29 |
| F2 | Fix validation scaler data leakage (`fit_transform` on val set) | Done 2026-04-29 |
| F3 | Fix `preprocess-power` missing delta features in `X_train.npy` | Done 2026-04-29 |
| F4 | Fix RUL labels ignoring device install age (`install_age_days` offset) | Done 2026-04-30 |
| F7 | Fix `rul_eval` using `battery_voltage_v > 0` as UPS proxy | Done 2026-04-30 |
| F8 | Fix Battery RUL CI coverage (dropout 0.1→0.3, mc_samples 30→50) | Done 2026-04-30 |
| F9 | Capability registry for model routing (phase model + RUL gating) | Done 2026-04-30 |
| F10 | Per-device temporal test split for RUL (not global last-15%) | Done 2026-04-30 |
| B1 | Per-device Welford online threshold in `PowerStreamProcessor` | Done 2026-04-29 |
| B2 | Vendor-agnostic normalization: replace absolute V/A/W with ratios | Done 2026-05-04 |
| B3 | Phase sag recall: `voltage_drop_delta_l1/l2/l3` features (25%→100%) | Done 2026-05-04 |
| E1 | Isolation Forest as real third detection model | Done 2026-05-05 |
| E2 | Per-category feature masks for IF (battery zeros, freq noise) | Done 2026-05-05 |
| E3 | Score all window timesteps, take minimum (peak anomaly wins) | Done 2026-05-06 |
| E4 | IF score_ratio guard (`IF_MIN_SCORE_RATIO`) to filter borderline FPs | Done 2026-05-06 |
| E5 | Batch pre-score all IF rows per device before window loop | Done 2026-05-06 |
| E6 | Decouple IF from LSTM buffer: IF scores on every incoming row | Done 2026-05-07 |
| E7 | Two-stage alert lifecycle: SUSPECTED → CONFIRMED / CLEARED | Done 2026-05-07 |
| F11 | Extract `scalar_transforms.py`: single source of truth for feature math | Done 2026-05-11 |
| F12 | Create `EventPreprocessor` in `inference/event_preprocessor.py` | Done 2026-05-11 |
| F13 | Migrate both inference paths to `EventPreprocessor`; delete `_apply_preprocessing` | Done 2026-05-11 |
| F14 | Extract shared `DualModelScorer` + `RollingWindowBuffer` into `inference/model_scorer.py` | Done 2026-05-11 |
| F16 | Fix type mismatch: `to_dual_result()` `str` → `list[str]` for top-feature fields | Done 2026-05-11 |
| F17 | Add `iforest_peak_timestep` + `iforest_top_features` to `PowerScoringResult` | Done 2026-05-11 |
| F15 | Reduce transport files to thin adapters (`PowerEvent.from_dict`, `format_power_alert`) | Done 2026-05-11 |

---

## Pending Work (priority order)

| ID | Feature | Depends on |
|---|---|---|
| ~~F16~~ | ~~Fix type mismatch: `to_dual_result()` passes `str` for `list[str]` top-feature fields~~ | Done 2026-05-11 |
| ~~F17~~ | ~~Add `iforest_peak_timestep` + `iforest_top_features` to `PowerScoringResult`~~ | Done 2026-05-11 |
| F15 | Reduce transport files to thin adapters | Done 2026-05-11 |
| A1 | SNMP OID adapter layer: vendor-agnostic OID resolution + unit normalization | B2 (done) |
| C1 | ClickHouse storage integration (replace .jsonl outputs) | B1, B2 done |
| E8 | Separate telemetry stream from alert lifecycle stream | C1, D1 |
| D1–D4 | NOC dashboard wiring (charts, counters, table, actions) | C1 |

F5 (expose imbalance constants) and F6 (stop mutating PowerEvent) are superseded by F11 and F12 respectively.
