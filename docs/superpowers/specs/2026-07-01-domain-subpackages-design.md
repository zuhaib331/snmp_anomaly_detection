# Design: Domain-Subpackage Restructure (lighter 80/20 split)

**Date:** 2026-07-01
**Branch:** `refactor/domain-subpackages` (off `basemodel`)
**Status:** Design — pending user review

---

## 1. Purpose

`snmp_anomaly_detection` holds two independent detection pipelines — **network** (5-metric:
cpu/memory/in_octets/out_octets/errors) and **power** (UPS/PDU/PSU/env: LSTM autoencoder +
Isolation Forest + battery RUL). Today they are organized **by layer** (`data/`, `preprocessing/`,
`inference/`, …) with files distinguished only by a `power_*` naming convention. As the power
pipeline grows, this strains: the prefix discipline gets harder to hold, a few files co-locate both
pipelines, and trained artifacts for the two pipelines are interleaved under one `outputs/` tree.

This refactor reorganizes the package **by domain** so each pipeline is a self-contained unit, and
separates trained artifacts the same way. It is **internal-only**: the CLI surface
(`python3 -m snmp_anomaly_detection <step>`) and all runtime behavior are unchanged.

### Why now / why safe
An import audit confirmed **zero cross-imports** between the pipelines. Exactly **two** code modules
are imported by both sides — `config.py` and `models/lstm_autoencoder.py` — and both stay shared.
So this is a mechanical move-and-rewrite, not a dependency untangling.

### Non-goals (explicitly out of scope)
- No logic or behavior changes; no model retraining beyond regenerating relocated artifacts.
- No content rewrites of docs (e.g. the stale "15-feature / phase model" drift in `CLAUDE.md`) —
  only **path references** are updated.
- No file renames (e.g. `power_features.py` keeps its `power_` prefix). Renaming is a possible
  later cleanup.
- No `shared/` subpackage and no splitting of `config.py` — those belong to the rejected "full
  split" alternative (see §8).

---

## 2. Target structure

```
snmp_anomaly_detection/
├── __init__.py, __main__.py            ← SHARED, unchanged
├── main.py                             ← SHARED CLI router (import paths updated)
├── config.py                           ← SHARED, stays WHOLE & in place; only ProjectPaths VALUES edited
├── models/
│   ├── __init__.py
│   └── lstm_autoencoder.py             ← SHARED generic model, stays
│
├── network/                            ← NEW subpackage
│   ├── __init__.py
│   ├── data/dataset_builder.py         ← network half of the split
│   ├── preprocessing/feature_engineering.py
│   ├── training/train_model.py
│   ├── inference/
│   │   ├── __init__.py                 ← relocated network re-export block (was inference/__init__.py)
│   │   ├── core.py, csv_replay.py, detect_anomalies.py,
│   │   ├── event_processor.py, live_microbatch.py, window_manager.py
│   │   └── events.py                   ← NormalizedEvent
│   └── streaming/
│       ├── detect_kafka.py, detect_kafka_dry.py,
│       └── kafka_source.py, produce_kafka_test_data.py
│
└── power/                              ← NEW subpackage
    ├── __init__.py
    ├── data/dataset_builder.py         ← power half of the split
    ├── data/vendor_oid_map.py
    ├── preprocessing/power_features.py, phase_features.py,
    │                 scalar_transforms.py, battery_features.py
    ├── models/battery_rul.py
    ├── training/train_baseline_power.py, train_iforest_power.py,
    │            train_battery_rul.py, train_phase_model.py
    ├── inference/
    │   ├── dual_model_scorer.py, event_preprocessor.py,
    │   ├── model_scorer.py, power_stream_processor.py
    │   └── events.py                   ← PowerEvent
    ├── streaming/detect_power_kafka.py, produce_power_kafka_test.py
    └── evaluation/power_eval.py, rul_eval.py, time_split.py
```

- Layer subdirectories (`inference/`, `preprocessing/`, …) are **kept inside each domain** to
  preserve the existing mental model and limit import surprises.
- Every new package directory gets an `__init__.py`.
- `utils/` is **removed** (see §5).

---

## 3. Complete file-move map

### Stay at top level (shared) — not moved
`__init__.py`, `__main__.py`, `main.py`, `config.py`, `models/__init__.py`,
`models/lstm_autoencoder.py`.

### Move to `network/` (via `git mv`, preserves history)
| From | To |
|---|---|
| `preprocessing/feature_engineering.py` | `network/preprocessing/feature_engineering.py` |
| `training/train_model.py` | `network/training/train_model.py` |
| `inference/core.py` | `network/inference/core.py` |
| `inference/csv_replay.py` | `network/inference/csv_replay.py` |
| `inference/detect_anomalies.py` | `network/inference/detect_anomalies.py` |
| `inference/event_processor.py` | `network/inference/event_processor.py` |
| `inference/live_microbatch.py` | `network/inference/live_microbatch.py` |
| `inference/window_manager.py` | `network/inference/window_manager.py` |
| `streaming/detect_kafka.py` | `network/streaming/detect_kafka.py` |
| `streaming/detect_kafka_dry.py` | `network/streaming/detect_kafka_dry.py` |
| `streaming/kafka_source.py` | `network/streaming/kafka_source.py` |
| `streaming/produce_kafka_test_data.py` | `network/streaming/produce_kafka_test_data.py` |

### Move to `power/` (via `git mv`)
| From | To |
|---|---|
| `data/vendor_oid_map.py` | `power/data/vendor_oid_map.py` |
| `preprocessing/power_features.py` | `power/preprocessing/power_features.py` |
| `preprocessing/phase_features.py` | `power/preprocessing/phase_features.py` |
| `preprocessing/scalar_transforms.py` | `power/preprocessing/scalar_transforms.py` |
| `preprocessing/battery_features.py` | `power/preprocessing/battery_features.py` |
| `models/battery_rul.py` | `power/models/battery_rul.py` |
| `training/train_baseline_power.py` | `power/training/train_baseline_power.py` |
| `training/train_iforest_power.py` | `power/training/train_iforest_power.py` |
| `training/train_battery_rul.py` | `power/training/train_battery_rul.py` |
| `training/train_phase_model.py` | `power/training/train_phase_model.py` |
| `inference/dual_model_scorer.py` | `power/inference/dual_model_scorer.py` |
| `inference/event_preprocessor.py` | `power/inference/event_preprocessor.py` |
| `inference/model_scorer.py` | `power/inference/model_scorer.py` |
| `inference/power_stream_processor.py` | `power/inference/power_stream_processor.py` |
| `streaming/detect_power_kafka.py` | `power/streaming/detect_power_kafka.py` |
| `streaming/produce_power_kafka_test.py` | `power/streaming/produce_power_kafka_test.py` |
| `evaluation/power_eval.py` | `power/evaluation/power_eval.py` |
| `evaluation/rul_eval.py` | `power/evaluation/rul_eval.py` |
| `evaluation/time_split.py` | `power/evaluation/time_split.py` |

### Files that must be SPLIT (not just moved)

**`data/dataset_builder.py` (465 lines) → two files**
- Network half → `network/data/dataset_builder.py`:
  `DatasetConfig`, `generate_normal_pattern`, `inject_anomaly`, `build_dataset`, `save_dataset`,
  network `main()`.
- Power half → `power/data/dataset_builder.py`: everything from `PowerDeviceProfile` onward
  (`PowerDatasetConfig`, `_circadian_load_factor`, …, `build_power_dataset`, `save_power_dataset`).
- Consumer check (confirmed): network consumers (`produce_kafka_test_data.py`) use only network
  functions; power consumers use only power functions. No consumer needs both halves.

**`inference/events.py` (106 lines) → two files**
- `NormalizedEvent` → `network/inference/events.py` (no internal config import needed).
- `PowerEvent` → `power/inference/events.py` (keeps `from snmp_anomaly_detection.config import
  BASELINE_UPS_FEATURES`).
- Confirmed: no file imports both event types.

### Removed
- `inference/__init__.py` — its 40-line re-export block is **network-only** and is **not imported
  package-level anywhere** (all internal imports use full `.inference.submodule` paths). Relocate
  the block to `network/inference/__init__.py` to preserve the public API for external/notebook
  users; the old top-level `inference/` package is dissolved.
- `utils/` — see §5.
- Empty top-level `data/`, `preprocessing/`, `training/`, `streaming/`, `evaluation/` packages are
  dissolved once their files move (their `__init__.py` files are empty).

---

## 4. Import-rewrite rules

| Import pattern | Action |
|---|---|
| `from snmp_anomaly_detection.config import …` | **UNCHANGED** (config stays at top — the 80/20 win; all 27 importers' lines untouched) |
| `from snmp_anomaly_detection.models.lstm_autoencoder import …` | **UNCHANGED** (stays shared) |
| `…models.battery_rul …` | → `…power.models.battery_rul …` |
| network-specific `…inference.X`, `…preprocessing.feature_engineering`, `…training.train_model`, `…streaming.{detect_kafka,detect_kafka_dry,kafka_source,produce_kafka_test_data}`, `…data.dataset_builder` (network fns) | prepend `network.` |
| power-specific `…inference.{dual_model_scorer,event_preprocessor,model_scorer,power_stream_processor}`, `…preprocessing.{power_features,phase_features,scalar_transforms,battery_features}`, `…training.{train_baseline_power,train_iforest_power,train_battery_rul,train_phase_model}`, `…streaming.{detect_power_kafka,produce_power_kafka_test}`, `…evaluation.{power_eval,rul_eval,time_split}`, `…data.{vendor_oid_map,dataset_builder (power fns)}` | prepend `power.` |
| `…inference.events import NormalizedEvent` | → `network.inference.events` |
| `…inference.events import PowerEvent` | → `power.inference.events` |
| `main.py` router imports | repoint network steps to `network.*`, power steps to `power.*` |

Top-level `__init__.py` re-exports only config classes (`FeatureEngineeringConfig`,
`InferenceConfig`, `ProjectPaths`, `TrainingConfig`) — **unchanged** since config stays put.

---

## 5. `utils/` removal

`utils/` currently holds only `scaler.pkl` (the network MinMaxScaler) and an empty `__init__.py`.
Section 6 relocates `scaler.pkl` into `outputs/network/`. After that:
- Delete the `utils/` directory.
- Remove `ProjectPaths.utils_dir` and the `utils_dir` entry in `ProjectPaths.ensure_directories()`.

---

## 6. Output / artifact separation (`ProjectPaths` value edits only)

`config.py` stays in place; only the **path string values** in `ProjectPaths` change (no import-line
impact). Every training/inference module reads paths from `ProjectPaths`, so relocations propagate
automatically. Existing on-disk artifacts are relocated and are regenerated on next train.

### New `outputs/` layout
```
outputs/
├── network/
│   ├── lstm_autoencoder.pth            (was outputs/lstm_autoencoder.pth)
│   ├── model_metadata.json             (was outputs/model_metadata.json)
│   ├── scaler.pkl                      (was utils/scaler.pkl)
│   ├── X_train.npy, X_test.npy, y_train.npy, y_test.npy   (was outputs/*.npy)
│   ├── anomaly_results.csv             (was outputs/anomaly_results.csv)
│   ├── anomaly_windows.json            (was outputs/anomaly_windows.json)
│   ├── kafka_live_results.jsonl        (was outputs/kafka_live_results.jsonl)
│   └── kafka_live_anomaly_windows.jsonl
└── power/
    ├── baseline/                       (was outputs/power_baseline/)
    ├── dual/                           (was outputs/power_dual/)
    ├── phase/                          (was outputs/power_phase/)
    └── battery_rul/                    (was outputs/battery_rul/)
```

### `ProjectPaths` field changes
- `scaler_file`: `utils/scaler.pkl` → `outputs/network/scaler.pkl`
- `x_train_file`, `x_test_file`, `y_train_file`, `y_test_file`: `outputs/*.npy` → `outputs/network/*.npy`
- `model_file`: `outputs/lstm_autoencoder.pth` → `outputs/network/lstm_autoencoder.pth`
- `model_metadata_file`: → `outputs/network/model_metadata.json`
- `anomaly_results_file`, `anomaly_windows_file`: → `outputs/network/…`
- `kafka_live_results_file`, `kafka_live_anomaly_windows_file`: → `outputs/network/…`
- `power_outputs_dir`: `outputs/power_baseline` → `outputs/power/baseline`
- `power_phase_outputs_dir`: `outputs/power_phase` → `outputs/power/phase`
- `battery_rul_outputs_dir`: `outputs/battery_rul` → `outputs/power/battery_rul`
- `power_dual_outputs_dir`: `outputs/power_dual` → `outputs/power/dual`
- `device_stats_file`, `iforest_model_file`, `iforest_metadata_file`: rebase onto `outputs/power/dual/…`
- Add `outputs/network` to `ensure_directories()`; update `ensure_power_directories()` to the new
  `outputs/power/*` dirs.

---

## 7. Documentation updates (path references only)

Update moved module paths and `outputs/` paths in the 9 docs that reference them; **no content
rewrites**:
`CLAUDE.md` (module map + Artifact Layout paths), `config/migrations/SYSTEM_DESIGN_HANDOFF.md`
(seed-model paths `outputs/power_baseline` → `outputs/power/baseline`, `outputs/power_dual` →
`outputs/power/dual`, and `streaming/detect_power_kafka.py` → `power/streaming/detect_power_kafka.py`),
`SYSTEM_DESIGN_HANDOFF.md` (root copy), `docs/model_report.md`, `docs/PLAN_INDEX.md`,
`docs/POWER_PIPELINE_EXPLAINED.md`, `docs/RUNNING_PIPELINE.md`, `docs/streaming/PHASE_4_KAFKA_LIVE.md`,
`docs/TODO.md`.

A few additional docs reference `outputs/power_baseline` / `outputs/power_dual` paths (e.g.
`docs/DETECTION_REPORT.md`, `docs/PLAN_PHASE_1_DATABASE.md`); the `git grep` audit in §9 is the
authoritative backstop and must come back clean across **all** docs, not just the 9 listed.

Explicitly **not** changed: the stale 15-feature / phase-model narrative in `CLAUDE.md`.

---

## 8. Kafka config: transport stays shared, topic names live with their domain

`KafkaConfig` mixes two different concerns: shared **transport/connection** settings and one
**power-specific topic name**.

- **Transport settings stay shared** in `KafkaConfig`: `bootstrap_servers`, `consumer_group_id`,
  `poll_timeout_ms`, `micro_batch_size`, `micro_batch_max_wait_ms`, `save_local_results`. Both
  pipelines hit the same broker with the same batching tuning — duplicating these would be wrong.
- **Topic names belong with their domain.** Two of three already do: `HARDCODED_INPUT_TOPIC`
  (`snmp-live-events`) lives in network `kafka_source.py`; `POWER_KAFKA_TOPIC` (`snmp-power-events`)
  lives in power `detect_power_kafka.py`. The outlier is `power_alerts_topic`
  (`snmp-power-anomaly-windows`), which sits in the shared `KafkaConfig` but is read by exactly one
  file — `detect_power_kafka.py`, which already defines `POWER_KAFKA_TOPIC` locally.

**Action:** move `power_alerts_topic` out of `KafkaConfig` into
`power/streaming/detect_power_kafka.py` as a module-level constant beside `POWER_KAFKA_TOPIC`, and
update its single reader (`detect_power_kafka.py:48`, `kafka_config.power_alerts_topic` → the new
local constant). After this, `KafkaConfig` is pure shared transport config with zero domain-specific
fields. One field, one file.

Note (out of scope): both pipelines currently share the default `consumer_group_id`. That is a
behavior/semantics question, not structure, and is intentionally left unchanged by this refactor.

### `config.py` otherwise stays whole — rejected alternative (full split)
Move `config.py` into `shared/` and split it three ways, move `main.py` into `shared/`, prepend
`shared.` everywhere. Cleanest end state but pays the full 27-importer config churn for marginal
benefit. Rejected in favor of the lighter split.

---

## 9. Workflow & verification

1. Branch `refactor/domain-subpackages` off `basemodel`; `git pull` first.
2. Create `network/` and `power/` package trees with `__init__.py` files.
3. `git mv` all moved files (preserves history); split `dataset_builder.py` and `events.py`;
   relocate the `inference/__init__.py` re-export block; remove `utils/`.
4. Rewrite imports per §4; edit `ProjectPaths` values per §6; update docs per §7.
5. **Verify** (env currently has no deps installed):
   - `python -m venv .venv && pip install -r requirements.txt`
   - `python -m compileall snmp_anomaly_detection` — syntax.
   - **Import smoke test** — a script that imports every module in its new location; this is the
     check that actually catches a broken import path (compileall does not resolve imports).
   - `git grep` audit — zero references to old module paths or old `outputs/` paths remain.
6. Merge to `basemodel` only once all checks are green.

### Success criteria
- `network/` and `power/` subpackages exist; no top-level `data/preprocessing/training/streaming/
  evaluation/utils` packages remain; `config.py`, `main.py`, `models/lstm_autoencoder.py` unchanged
  in location.
- Every module imports cleanly in its new location (smoke test passes); `compileall` passes.
- `git grep` finds no stale module-path or `outputs/`-path references in code or docs.
- CLI surface unchanged: `python3 -m snmp_anomaly_detection <step>` lists and runs all existing steps.
- Trained artifacts land under `outputs/network/` and `outputs/power/*` respectively.
```
