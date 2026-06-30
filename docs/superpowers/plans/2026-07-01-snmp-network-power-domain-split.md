# SNMP Network/Power Domain-Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `snmp_anomaly_detection` by domain into `network/` and `power/` subpackages (shared `config.py`, `main.py`, `models/lstm_autoencoder.py` stay at top level), and separate trained artifacts into `outputs/network/` and `outputs/power/`.

**Architecture:** Pure internal restructure — no behavior change. Files move via `git mv`; internal imports are rewritten with scoped `sed` passes plus a handful of explicit edits for the two split files (`dataset_builder.py`, `events.py`) and the router (`main.py`). `config.py` stays whole and in place, so its 27 importers' import lines are untouched; only `ProjectPaths` *values* change for the artifact relocation.

**Tech Stack:** Python 3, PyTorch, scikit-learn, pandas, kafka-python. No pytest in this project.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-01-domain-subpackages-design.md` — this plan implements it.
- **Branch:** `snmp-network-power-domain-split`, created off `basemodel`.
- **No behavior change.** The CLI surface (`python3 -m snmp_anomaly_detection <step>`) and all step names stay identical.
- **Shared stays at top, unchanged location:** `config.py`, `main.py`, `__main__.py`, `__init__.py`, `models/__init__.py`, `models/lstm_autoencoder.py`.
- **`config.py` stays whole.** Never split it. `from snmp_anomaly_detection.config import …` lines are NEVER rewritten. Only `ProjectPaths` field *values* and the removal of `KafkaConfig.power_alerts_topic` change inside it.
- **`models.lstm_autoencoder` imports are NEVER rewritten** (shared). Only `models.battery_rul` moves.
- **Filenames unchanged** (e.g. `power_features.py` keeps its prefix). No renames.
- **Preserve history:** move with `git mv`, not delete+create (except the two split files, where one half is necessarily a new file).
- **Torch-lazy-import invariant:** `network/data/`, `network/preprocessing/`, `power/data/`, `power/preprocessing/` modules must NOT import torch at module level (only `training/` and `inference/` may). Do not introduce torch imports while moving.
- **No pytest.** Each task is verified by: `python -m compileall`, targeted module imports (using the venv built in Task 1), and `git grep` audits. There are no unit tests to write.
- **Verification python:** after Task 1, always invoke the venv interpreter at `.venv/Scripts/python.exe` (Windows Git Bash) so torch/sklearn/kafka resolve.
- **Old paths that must end up with ZERO `git grep` hits in code:** `snmp_anomaly_detection.inference.`, `snmp_anomaly_detection.preprocessing.`, `snmp_anomaly_detection.training.`, `snmp_anomaly_detection.streaming.`, `snmp_anomaly_detection.evaluation.`, `snmp_anomaly_detection.data.` (except where rewritten to `.network.`/`.power.`), and `utils/scaler.pkl`.

---

## File Structure (target)

```
snmp_anomaly_detection/
├── __init__.py, __main__.py, main.py     (shared; main.py imports rewritten)
├── config.py                             (shared whole; ProjectPaths values + KafkaConfig edited)
├── models/__init__.py, models/lstm_autoencoder.py   (shared)
├── network/
│   ├── __init__.py
│   ├── data/__init__.py, data/dataset_builder.py            (network half)
│   ├── preprocessing/__init__.py, preprocessing/feature_engineering.py
│   ├── training/__init__.py, training/train_model.py
│   ├── inference/__init__.py  (relocated re-export block)
│   │   core.py, csv_replay.py, detect_anomalies.py,
│   │   event_processor.py, live_microbatch.py, window_manager.py, events.py (NormalizedEvent)
│   └── streaming/__init__.py
│       detect_kafka.py, detect_kafka_dry.py, kafka_source.py, produce_kafka_test_data.py
└── power/
    ├── __init__.py
    ├── data/__init__.py, data/dataset_builder.py (power half), data/vendor_oid_map.py
    ├── preprocessing/__init__.py, power_features.py, phase_features.py, scalar_transforms.py, battery_features.py
    ├── models/__init__.py, battery_rul.py
    ├── training/__init__.py, train_baseline_power.py, train_iforest_power.py, train_battery_rul.py, train_phase_model.py
    ├── inference/__init__.py, dual_model_scorer.py, event_preprocessor.py, model_scorer.py, power_stream_processor.py, events.py (PowerEvent)
    ├── streaming/__init__.py, detect_power_kafka.py, produce_power_kafka_test.py
    └── evaluation/__init__.py, power_eval.py, rul_eval.py, time_split.py
```

`utils/` is removed. Old top-level `data/ preprocessing/ training/ streaming/ evaluation/ inference/` packages are dissolved once emptied.

---

## Task 1: Branch, environment, and baseline

**Files:**
- Modify: `.gitignore` (ensure `.venv/` ignored)

**Interfaces:**
- Produces: a `.venv` with all deps; a recorded baseline list of CLI step names used as the parity check in Task 5/8.

- [ ] **Step 1: Create the branch off basemodel**

```bash
cd /d/Aircod/snmp_anomaly_detection
git checkout basemodel && git pull
git checkout -b snmp-network-power-domain-split
```

- [ ] **Step 2: Ensure `.venv/` is gitignored**

Check `.gitignore` contains a line `.venv/`. If absent, add it:

```bash
grep -qxF '.venv/' .gitignore || printf '.venv/\n' >> .gitignore
```

- [ ] **Step 3: Create venv and install dependencies**

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
Expected: installs numpy, pandas, scikit-learn, joblib, torch, kafka-python. The torch wheel is large; allow several minutes.

- [ ] **Step 4: Baseline — compileall passes on current code**

```bash
.venv/Scripts/python.exe -m compileall -q snmp_anomaly_detection && echo "COMPILE OK"
```
Expected: `COMPILE OK`.

- [ ] **Step 5: Baseline — record the current CLI step list**

```bash
.venv/Scripts/python.exe -c "import snmp_anomaly_detection.main as m; print('\n'.join(sorted(m.build_parser()._actions[1].choices)))" > /tmp/cli_steps_baseline.txt
cat /tmp/cli_steps_baseline.txt
```
Expected: the full sorted list of steps (`detect`, `detect-csv`, `detect-kafka`, `detect-kafka-dry`, `generate-data`, `generate-power-data`, `preprocess`, `preprocess-power`, `produce-kafka-test-data`, `produce-power-kafka-test`, `train`, `train-battery-rul`, `train-power-baseline`, `train-power-iforest`, `train-power-phase`, …). Keep this file — Task 5 and Task 8 compare against it.

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: branch setup and venv gitignore for domain split"
```

---

## Task 2: Scaffold the `network/` and `power/` package trees

**Files:**
- Create: `network/__init__.py` and one per subpackage; `power/__init__.py` and one per subpackage.

**Interfaces:**
- Produces: empty importable package dirs that Tasks 3–4 move files into.

- [ ] **Step 1: Create all package directories with empty `__init__.py`**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
for d in network network/data network/preprocessing network/training network/inference network/streaming \
         power power/data power/preprocessing power/training power/models power/inference power/streaming power/evaluation; do
  mkdir -p "$d"; : > "$d/__init__.py"
done
```

- [ ] **Step 2: Verify the trees compile**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe -m compileall -q snmp_anomaly_detection && echo "COMPILE OK"
```
Expected: `COMPILE OK`.

- [ ] **Step 3: Commit**

```bash
git add snmp_anomaly_detection/network snmp_anomaly_detection/power
git commit -m "refactor: scaffold network/ and power/ subpackage trees"
```

---

## Task 3: Migrate the network pipeline

**Files:**
- Move (git mv): `preprocessing/feature_engineering.py`, `training/train_model.py`, `inference/{core,csv_replay,detect_anomalies,event_processor,live_microbatch,window_manager}.py`, `streaming/{detect_kafka,detect_kafka_dry,kafka_source,produce_kafka_test_data}.py` → under `network/`.
- Split: `data/dataset_builder.py` → `network/data/dataset_builder.py` (network half).
- Split: `inference/events.py` → `network/inference/events.py` (`NormalizedEvent`).
- Create: `network/inference/__init__.py` (relocated re-export block).

**Interfaces:**
- Produces: importable `snmp_anomaly_detection.network.*` modules. `main.py` is temporarily broken until Task 5 — do NOT import the top-level package or run the CLI in this task; verify only `network.*` modules.

- [ ] **Step 1: git mv the straightforward network files**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
git mv preprocessing/feature_engineering.py network/preprocessing/feature_engineering.py
git mv training/train_model.py            network/training/train_model.py
git mv inference/core.py                  network/inference/core.py
git mv inference/csv_replay.py            network/inference/csv_replay.py
git mv inference/detect_anomalies.py      network/inference/detect_anomalies.py
git mv inference/event_processor.py       network/inference/event_processor.py
git mv inference/live_microbatch.py       network/inference/live_microbatch.py
git mv inference/window_manager.py        network/inference/window_manager.py
git mv streaming/detect_kafka.py          network/streaming/detect_kafka.py
git mv streaming/detect_kafka_dry.py      network/streaming/detect_kafka_dry.py
git mv streaming/kafka_source.py          network/streaming/kafka_source.py
git mv streaming/produce_kafka_test_data.py network/streaming/produce_kafka_test_data.py
```

- [ ] **Step 2: Split `dataset_builder.py` — network half**

```bash
git mv data/dataset_builder.py network/data/dataset_builder.py
```
Then edit `network/data/dataset_builder.py`: **delete everything from the line `@dataclass(frozen=True)` immediately above `class PowerDeviceProfile:` to the end of file.** Keep the header imports (lines 1–11) and the network symbols: `DatasetConfig`, `generate_normal_pattern`, `inject_anomaly`, `build_dataset`, `save_dataset`, and the network `main()`. (The power half is recreated in Task 4 from git history.)

- [ ] **Step 3: Split `events.py` — network `NormalizedEvent`**

```bash
git mv inference/events.py network/inference/events.py
```
Then replace the entire contents of `network/inference/events.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedEvent:
    timestamp: Any
    device_id: str
    cpu: float
    memory: float
    in_octets: float
    out_octets: float
    errors: float
    anomaly: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "cpu": self.cpu,
            "memory": self.memory,
            "in_octets": self.in_octets,
            "out_octets": self.out_octets,
            "errors": self.errors,
            "anomaly": self.anomaly,
        }
```

- [ ] **Step 4: Relocate the network re-export block into `network/inference/__init__.py`**

Write `network/inference/__init__.py` with the re-export block (moved from the old `inference/__init__.py`), repointing every path to `network.inference.*`:

```python
from snmp_anomaly_detection.network.inference.core import (
    InferenceArtifacts,
    WindowScore,
    load_inference_artifacts,
    scale_window,
    score_window,
    score_windows,
)
from snmp_anomaly_detection.network.inference.csv_replay import detect_csv_replay
from snmp_anomaly_detection.network.inference.event_processor import (
    EventProcessor,
    ProcessedWindow,
)
from snmp_anomaly_detection.network.inference.events import NormalizedEvent
from snmp_anomaly_detection.network.inference.live_microbatch import (
    LiveMicroBatchProcessor,
    MicroBatchConfig,
    PreparedWindow,
)
from snmp_anomaly_detection.network.inference.window_manager import DeviceWindowManager, ReadyWindow

__all__ = [
    "DeviceWindowManager",
    "EventProcessor",
    "InferenceArtifacts",
    "LiveMicroBatchProcessor",
    "MicroBatchConfig",
    "NormalizedEvent",
    "PreparedWindow",
    "ProcessedWindow",
    "ReadyWindow",
    "WindowScore",
    "detect_csv_replay",
    "load_inference_artifacts",
    "scale_window",
    "score_window",
    "score_windows",
]
```
(Confirm the imported symbol names against the moved `network/inference/core.py` and `live_microbatch.py` — they must match the originals exactly. If any name differs from the list above, use the actual name from the source file.)

- [ ] **Step 5: Replace the now-stale old `inference/__init__.py` with empty**

The old `inference/` package still holds power files until Task 4. Empty its `__init__.py` so it does not import moved modules:

```bash
: > inference/__init__.py
```

- [ ] **Step 6: Rewrite network-internal import paths (scoped to the network/ tree)**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
NET="snmp_anomaly_detection.network"
declare -a MAP=(
  "snmp_anomaly_detection.preprocessing.feature_engineering=$NET.preprocessing.feature_engineering"
  "snmp_anomaly_detection.training.train_model=$NET.training.train_model"
  "snmp_anomaly_detection.inference.core=$NET.inference.core"
  "snmp_anomaly_detection.inference.csv_replay=$NET.inference.csv_replay"
  "snmp_anomaly_detection.inference.detect_anomalies=$NET.inference.detect_anomalies"
  "snmp_anomaly_detection.inference.event_processor=$NET.inference.event_processor"
  "snmp_anomaly_detection.inference.live_microbatch=$NET.inference.live_microbatch"
  "snmp_anomaly_detection.inference.window_manager=$NET.inference.window_manager"
  "snmp_anomaly_detection.inference.events=$NET.inference.events"
  "snmp_anomaly_detection.streaming.detect_kafka=$NET.streaming.detect_kafka"
  "snmp_anomaly_detection.streaming.detect_kafka_dry=$NET.streaming.detect_kafka_dry"
  "snmp_anomaly_detection.streaming.kafka_source=$NET.streaming.kafka_source"
  "snmp_anomaly_detection.streaming.produce_kafka_test_data=$NET.streaming.produce_kafka_test_data"
  "snmp_anomaly_detection.data.dataset_builder=$NET.data.dataset_builder"
)
for pair in "${MAP[@]}"; do
  old="${pair%%=*}"; new="${pair##*=}"
  grep -rl --include=*.py "$old" network | xargs -r sed -i "s/${old//./\\.}/$new/g"
done
echo "network import rewrite done"
```
Note: this is scoped to the `network/` tree only, so it cannot touch `config`, `models.lstm_autoencoder`, power files, or `main.py`.

- [ ] **Step 7: Verify every network module imports cleanly**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe - <<'PY'
import importlib, pkgutil
import snmp_anomaly_detection.network as net
failed = []
for m in pkgutil.walk_packages(net.__path__, net.__name__ + "."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        failed.append((m.name, repr(e)))
if failed:
    print("FAILURES:"); [print(" ", n, e) for n, e in failed]
else:
    print("NETWORK IMPORTS OK")
import sys; sys.exit(1 if failed else 0)
PY
```
Expected: `NETWORK IMPORTS OK`.

- [ ] **Step 8: Confirm no old network paths remain in the network tree**

```bash
cd /d/Aircod/snmp_anomaly_detection
git grep -nE "snmp_anomaly_detection\.(inference|preprocessing|training|streaming)\.(core|csv_replay|detect_anomalies|event_processor|live_microbatch|window_manager|events|feature_engineering|train_model|detect_kafka|detect_kafka_dry|kafka_source|produce_kafka_test_data)" -- snmp_anomaly_detection/network || echo "CLEAN"
```
Expected: `CLEAN`.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: migrate network pipeline into network/ subpackage"
```

---

## Task 4: Migrate the power pipeline

**Files:**
- Move (git mv): `preprocessing/{power_features,phase_features,scalar_transforms,battery_features}.py`, `models/battery_rul.py`, `training/{train_baseline_power,train_iforest_power,train_battery_rul,train_phase_model}.py`, `inference/{dual_model_scorer,event_preprocessor,model_scorer,power_stream_processor}.py`, `streaming/{detect_power_kafka,produce_power_kafka_test}.py`, `evaluation/{power_eval,rul_eval,time_split}.py`, `data/vendor_oid_map.py` → under `power/`.
- Create: `power/data/dataset_builder.py` (power half, from git history), `power/inference/events.py` (`PowerEvent`).
- Modify: `config.py` (remove `KafkaConfig.power_alerts_topic`), `power/streaming/detect_power_kafka.py` (add local `POWER_ALERTS_TOPIC`).

**Interfaces:**
- Consumes: `network/` is already migrated (Task 3).
- Produces: importable `snmp_anomaly_detection.power.*`. `main.py` still broken until Task 5; verify only `power.*` here.

- [ ] **Step 1: git mv the straightforward power files**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
git mv preprocessing/power_features.py     power/preprocessing/power_features.py
git mv preprocessing/phase_features.py     power/preprocessing/phase_features.py
git mv preprocessing/scalar_transforms.py  power/preprocessing/scalar_transforms.py
git mv preprocessing/battery_features.py   power/preprocessing/battery_features.py
git mv models/battery_rul.py               power/models/battery_rul.py
git mv training/train_baseline_power.py    power/training/train_baseline_power.py
git mv training/train_iforest_power.py     power/training/train_iforest_power.py
git mv training/train_battery_rul.py       power/training/train_battery_rul.py
git mv training/train_phase_model.py       power/training/train_phase_model.py
git mv inference/dual_model_scorer.py      power/inference/dual_model_scorer.py
git mv inference/event_preprocessor.py     power/inference/event_preprocessor.py
git mv inference/model_scorer.py           power/inference/model_scorer.py
git mv inference/power_stream_processor.py  power/inference/power_stream_processor.py
git mv streaming/detect_power_kafka.py     power/streaming/detect_power_kafka.py
git mv streaming/produce_power_kafka_test.py power/streaming/produce_power_kafka_test.py
git mv evaluation/power_eval.py            power/evaluation/power_eval.py
git mv evaluation/rul_eval.py              power/evaluation/rul_eval.py
git mv evaluation/time_split.py            power/evaluation/time_split.py
git mv data/vendor_oid_map.py              power/data/vendor_oid_map.py
```

- [ ] **Step 2: Recreate the power half of `dataset_builder.py`**

```bash
git show HEAD~1:snmp_anomaly_detection/data/dataset_builder.py > power/data/dataset_builder.py
```
(`HEAD~1` is the network-migration commit, which still had the full original file before Task 3 trimmed it — verify the dump contains both halves; if not, use the commit hash prior to Task 3's split.) Then edit `power/data/dataset_builder.py`: **delete the network section** — remove `DatasetConfig`, `generate_normal_pattern`, `inject_anomaly`, `build_dataset`, `save_dataset`, and the network `main()` (everything between the header imports and `class PowerDeviceProfile:`). Keep the header imports (lines 1–11) and everything from `class PowerDeviceProfile:` to end of file.

- [ ] **Step 3: Create `power/inference/events.py` with `PowerEvent`**

Write `power/inference/events.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES

# Fields that every wire-format payload must carry.
POWER_REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "device_category",
    "vendor",
)

# Raw (pre-normalization) columns carried in feature_values so EventPreprocessor
# can compute vendor-agnostic derived features.
# These are NOT model inputs — they are consumed during preprocessing.
POWER_RAW_COLS: tuple[str, ...] = (
    "input_voltage_v",
    "output_voltage_v",
    "battery_voltage_v",
)


@dataclass
class PowerEvent:
    """One SNMP event from any transport (Kafka, MQTT, CSV replay, …)."""
    timestamp: datetime
    device_id: str
    device_category: str
    vendor: str
    feature_values: dict[str, float]   # canonical feature name → value
    phase_count: int = 1               # 1 (single-phase) or 3 (three-phase)
    # Device registration constants for normalization.
    # Set from device registration / SNMP discovery at onboarding time.
    nominal_voltage_v: float = 120.0   # nominal input voltage (120 or 230)
    rated_battery_v: float = 0.0       # battery string voltage (0 for non-UPS)
    session_reset: bool = False        # clears stale per-device delta state on new producer run

    @classmethod
    def from_dict(cls, payload: dict) -> PowerEvent | None:
        """Construct from a raw wire-format dict (Kafka, MQTT, HTTP webhook, …).

        Returns None if a required field is missing or the timestamp is unparseable.
        Every feature column defaults to 0.0 when absent from the payload.
        """
        import pandas as pd  # lazy — keeps events.py free of pandas at module level

        for required in POWER_REQUIRED_FIELDS:
            if required not in payload:
                return None
        try:
            ts = pd.to_datetime(payload["timestamp"])
            if pd.isna(ts):
                return None
        except Exception:
            return None

        all_cols = set(BASELINE_UPS_FEATURES) | set(POWER_RAW_COLS)
        feature_values: dict[str, float] = {}
        for col in all_cols:
            try:
                feature_values[col] = float(payload.get(col, 0.0))
            except (TypeError, ValueError):
                feature_values[col] = 0.0

        return cls(
            timestamp=ts.to_pydatetime(),
            device_id=str(payload["device_id"]),
            device_category=str(payload.get("device_category", "ups")),
            vendor=str(payload.get("vendor", "liebert")),
            feature_values=feature_values,
            phase_count=int(payload.get("phase_count", 1)),
            nominal_voltage_v=float(payload.get("nominal_voltage_v", 120.0)),
            rated_battery_v=float(payload.get("rated_battery_v", 0.0)),
            session_reset=bool(payload.get("_device_reset", False)),
        )

    def get_baseline_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in BASELINE_UPS_FEATURES]
```

- [ ] **Step 4: Move `power_alerts_topic` out of `KafkaConfig`**

In `config.py`, delete the line inside `class KafkaConfig`:
```python
    power_alerts_topic: str = "snmp-power-anomaly-windows"
```
In `power/streaming/detect_power_kafka.py`, add a module-level constant next to `POWER_KAFKA_TOPIC`:
```python
POWER_ALERTS_TOPIC = "snmp-power-anomaly-windows"
```
and change its single reader (was `alerts_topic = kafka_config.power_alerts_topic`) to:
```python
    alerts_topic = POWER_ALERTS_TOPIC
```

- [ ] **Step 5: Rewrite power-internal import paths (scoped to the power/ tree)**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
PWR="snmp_anomaly_detection.power"
declare -a MAP=(
  "snmp_anomaly_detection.preprocessing.power_features=$PWR.preprocessing.power_features"
  "snmp_anomaly_detection.preprocessing.phase_features=$PWR.preprocessing.phase_features"
  "snmp_anomaly_detection.preprocessing.scalar_transforms=$PWR.preprocessing.scalar_transforms"
  "snmp_anomaly_detection.preprocessing.battery_features=$PWR.preprocessing.battery_features"
  "snmp_anomaly_detection.models.battery_rul=$PWR.models.battery_rul"
  "snmp_anomaly_detection.training.train_baseline_power=$PWR.training.train_baseline_power"
  "snmp_anomaly_detection.training.train_iforest_power=$PWR.training.train_iforest_power"
  "snmp_anomaly_detection.training.train_battery_rul=$PWR.training.train_battery_rul"
  "snmp_anomaly_detection.training.train_phase_model=$PWR.training.train_phase_model"
  "snmp_anomaly_detection.inference.dual_model_scorer=$PWR.inference.dual_model_scorer"
  "snmp_anomaly_detection.inference.event_preprocessor=$PWR.inference.event_preprocessor"
  "snmp_anomaly_detection.inference.model_scorer=$PWR.inference.model_scorer"
  "snmp_anomaly_detection.inference.power_stream_processor=$PWR.inference.power_stream_processor"
  "snmp_anomaly_detection.streaming.detect_power_kafka=$PWR.streaming.detect_power_kafka"
  "snmp_anomaly_detection.streaming.produce_power_kafka_test=$PWR.streaming.produce_power_kafka_test"
  "snmp_anomaly_detection.evaluation.power_eval=$PWR.evaluation.power_eval"
  "snmp_anomaly_detection.evaluation.rul_eval=$PWR.evaluation.rul_eval"
  "snmp_anomaly_detection.evaluation.time_split=$PWR.evaluation.time_split"
  "snmp_anomaly_detection.data.vendor_oid_map=$PWR.data.vendor_oid_map"
  "snmp_anomaly_detection.data.dataset_builder=$PWR.data.dataset_builder"
  "snmp_anomaly_detection.inference.events=$PWR.inference.events"
)
for pair in "${MAP[@]}"; do
  old="${pair%%=*}"; new="${pair##*=}"
  grep -rl --include=*.py "$old" power | xargs -r sed -i "s/${old//./\\.}/$new/g"
done
echo "power import rewrite done"
```
Note: scoped to `power/` only — cannot touch `config`, `models.lstm_autoencoder`, network files, or `main.py`.

- [ ] **Step 6: Remove the now-empty old top-level packages**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
git rm -r data preprocessing training streaming evaluation inference
```
Expected: each removed dir contained only an `__init__.py` (and the already-moved files). If `git rm` reports a non-`__init__.py` file remaining, STOP — a file was missed in the move; investigate before continuing.

- [ ] **Step 7: Verify every power module imports cleanly**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe - <<'PY'
import importlib, pkgutil
import snmp_anomaly_detection.power as pwr
failed = []
for m in pkgutil.walk_packages(pwr.__path__, pwr.__name__ + "."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        failed.append((m.name, repr(e)))
if failed:
    print("FAILURES:"); [print(" ", n, e) for n, e in failed]
else:
    print("POWER IMPORTS OK")
import sys; sys.exit(1 if failed else 0)
PY
```
Expected: `POWER IMPORTS OK`.

- [ ] **Step 8: Confirm no `power_alerts_topic` reference remains in config**

```bash
cd /d/Aircod/snmp_anomaly_detection
git grep -n "power_alerts_topic" -- snmp_anomaly_detection || echo "CLEAN"
```
Expected: `CLEAN` (the only references should now be the new `POWER_ALERTS_TOPIC` constant, which this grep does not match).

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: migrate power pipeline into power/ subpackage; move power_alerts_topic to power streaming"
```

---

## Task 5: Rewrite the `main.py` router and verify full-package import

**Files:**
- Modify: `snmp_anomaly_detection/main.py` (all internal import paths).

**Interfaces:**
- Consumes: migrated `network.*` and `power.*` modules.
- Produces: a fully importable top-level package with an unchanged CLI step list.

- [ ] **Step 1: Rewrite the network-step imports in `main.py`**

Apply these exact path changes in `main.py`:
- `from snmp_anomaly_detection.inference.csv_replay import main as detect_csv_main` → `from snmp_anomaly_detection.network.inference.csv_replay import main as detect_csv_main`
- `from snmp_anomaly_detection.inference.detect_anomalies import main as detect_main` → `…network.inference.detect_anomalies…`
- `from snmp_anomaly_detection.preprocessing.feature_engineering import (main as feature_engineering_main,)` → `…network.preprocessing.feature_engineering…`
- `from snmp_anomaly_detection.streaming.detect_kafka import main as detect_kafka_main` → `…network.streaming.detect_kafka…`
- `from snmp_anomaly_detection.streaming.detect_kafka_dry import main as detect_kafka_dry_main` → `…network.streaming.detect_kafka_dry…`
- `from snmp_anomaly_detection.streaming.produce_kafka_test_data import (main as produce_kafka_test_data_main,)` → `…network.streaming.produce_kafka_test_data…`
- `from snmp_anomaly_detection.training.train_model import main as train_main` → `…network.training.train_model…`
- network `dataset_builder` import: `from snmp_anomaly_detection.data.dataset_builder import main as build_dataset_main` → `…network.data.dataset_builder…`

- [ ] **Step 2: Rewrite the power-step imports in `main.py`**

Inside `_import_power_steps()`:
- `from snmp_anomaly_detection.data.dataset_builder import (PowerDatasetConfig, build_power_dataset, save_power_dataset,)` → `from snmp_anomaly_detection.power.data.dataset_builder import (…)`
- `from snmp_anomaly_detection.preprocessing.power_features import main as preprocess_power_main` → `…power.preprocessing.power_features…`
- `from snmp_anomaly_detection.training.train_baseline_power import main as train_baseline_main` → `…power.training.train_baseline_power…`
- `from snmp_anomaly_detection.training.train_battery_rul import main as train_rul_main` → `…power.training.train_battery_rul…`
- `from snmp_anomaly_detection.training.train_iforest_power import main as train_iforest_main` → `…power.training.train_iforest_power…`
- `from snmp_anomaly_detection.evaluation.power_eval import main as evaluate_baseline_main` → `…power.evaluation.power_eval…`
- `from snmp_anomaly_detection.evaluation.rul_eval import main as evaluate_rul_main` → `…power.evaluation.rul_eval…`
- `from snmp_anomaly_detection.inference.dual_model_scorer import main as detect_power_csv_main` → `…power.inference.dual_model_scorer…`
- `from snmp_anomaly_detection.streaming.detect_power_kafka import main as detect_power_kafka_main` → `…power.streaming.detect_power_kafka…`
- `from snmp_anomaly_detection.streaming.produce_power_kafka_test import (main as produce_power_kafka_test_main,)` → `…power.streaming.produce_power_kafka_test…`

- [ ] **Step 3: Verify the WHOLE package imports**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe - <<'PY'
import importlib, pkgutil
import snmp_anomaly_detection as pkg
failed = []
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        failed.append((m.name, repr(e)))
if failed:
    print("FAILURES:"); [print(" ", n, e) for n, e in failed]
else:
    print("ALL IMPORTS OK")
import sys; sys.exit(1 if failed else 0)
PY
```
Expected: `ALL IMPORTS OK`.

- [ ] **Step 4: Verify the CLI step list is unchanged vs baseline**

```bash
.venv/Scripts/python.exe -c "import snmp_anomaly_detection.main as m; print('\n'.join(sorted(m.build_parser()._actions[1].choices)))" > /tmp/cli_steps_after.txt
diff /tmp/cli_steps_baseline.txt /tmp/cli_steps_after.txt && echo "CLI STEPS UNCHANGED"
```
Expected: `CLI STEPS UNCHANGED` (empty diff).

- [ ] **Step 5: Commit**

```bash
git add snmp_anomaly_detection/main.py
git commit -m "refactor: repoint main.py router to network/ and power/ subpackages"
```

---

## Task 6: Reorganize output artifacts via `ProjectPaths`

**Files:**
- Modify: `snmp_anomaly_detection/config.py` (`ProjectPaths` field values, `ensure_directories`, `ensure_power_directories`; remove `utils_dir`).
- Remove: `snmp_anomaly_detection/utils/` (directory + `__init__.py` + `scaler.pkl`).

**Interfaces:**
- Consumes: `ProjectPaths` is the single source of artifact paths read by all training/inference code.
- Produces: artifacts resolve under `outputs/network/` and `outputs/power/{baseline,dual,phase,battery_rul}`.

- [ ] **Step 1: Edit `ProjectPaths` network artifact paths**

In `config.py`, change these field default values (keep field names identical):
- `scaler_file`: `PACKAGE_ROOT / "utils" / "scaler.pkl"` → `PACKAGE_ROOT / "outputs" / "network" / "scaler.pkl"`
- `x_train_file`: `… / "outputs" / "X_train.npy"` → `… / "outputs" / "network" / "X_train.npy"`
- `x_test_file`, `y_train_file`, `y_test_file`: same — insert `"network"` before the filename.
- `model_file`: `… / "outputs" / "lstm_autoencoder.pth"` → `… / "outputs" / "network" / "lstm_autoencoder.pth"`
- `model_metadata_file`: → `… / "outputs" / "network" / "model_metadata.json"`
- `anomaly_results_file`: → `… / "outputs" / "network" / "anomaly_results.csv"`
- `anomaly_windows_file`: → `… / "outputs" / "network" / "anomaly_windows.json"`
- `kafka_live_results_file`: → `… / "outputs" / "network" / "kafka_live_results.jsonl"`
- `kafka_live_anomaly_windows_file`: → `… / "outputs" / "network" / "kafka_live_anomaly_windows.jsonl"`

- [ ] **Step 2: Edit `ProjectPaths` power artifact paths**

- `power_outputs_dir`: `… / "outputs" / "power_baseline"` → `… / "outputs" / "power" / "baseline"`
- `power_phase_outputs_dir`: `… / "outputs" / "power_phase"` → `… / "outputs" / "power" / "phase"`
- `battery_rul_outputs_dir`: `… / "outputs" / "battery_rul"` → `… / "outputs" / "power" / "battery_rul"`
- `power_dual_outputs_dir`: `… / "outputs" / "power_dual"` → `… / "outputs" / "power" / "dual"`
- `device_stats_file`: `… / "outputs" / "power_dual" / "device_stats.json"` → `… / "outputs" / "power" / "dual" / "device_stats.json"`
- `iforest_model_file`: → `… / "outputs" / "power" / "dual" / "iforest_models.pkl"`
- `iforest_metadata_file`: → `… / "outputs" / "power" / "dual" / "iforest_metadata.json"`

- [ ] **Step 3: Update directory-creation helpers and remove `utils_dir`**

In `ProjectPaths`:
- Remove the `utils_dir` field.
- In `ensure_directories()`, replace the `(self.data_dir, self.outputs_dir, self.utils_dir)` tuple with `(self.data_dir, self.outputs_dir, self.outputs_dir / "network")`.
- In `ensure_power_directories()`, ensure the tuple lists the new dirs: `self.power_outputs_dir, self.power_phase_outputs_dir, self.battery_rul_outputs_dir, self.power_dual_outputs_dir` (values now resolve under `outputs/power/`).

- [ ] **Step 4: Remove the `utils/` directory**

```bash
cd /d/Aircod/snmp_anomaly_detection/snmp_anomaly_detection
git rm -r utils
```

- [ ] **Step 5: Verify the new paths resolve and `utils_dir` is gone**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe - <<'PY'
from snmp_anomaly_detection.config import ProjectPaths
p = ProjectPaths()
assert p.model_file.parent.name == "network", p.model_file
assert p.scaler_file.parent.name == "network", p.scaler_file
assert p.power_dual_outputs_dir.parent.name == "power", p.power_dual_outputs_dir
assert p.power_outputs_dir.name == "baseline", p.power_outputs_dir
assert not hasattr(p, "utils_dir"), "utils_dir still present"
p.ensure_directories(); p.ensure_power_directories()
print("PATHS OK:", p.model_file)
print("        ", p.power_dual_outputs_dir)
PY
```
Expected: `PATHS OK:` lines under `outputs/network` and `outputs/power/dual`, no assertion error.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: separate artifacts into outputs/network and outputs/power; remove utils/"
```

---

## Task 7: Update documentation path references

**Files:**
- Modify (path references only, no content rewrites): `CLAUDE.md`, `config/migrations/SYSTEM_DESIGN_HANDOFF.md`, `SYSTEM_DESIGN_HANDOFF.md`, `docs/model_report.md`, `docs/PLAN_INDEX.md`, `docs/POWER_PIPELINE_EXPLAINED.md`, `docs/RUNNING_PIPELINE.md`, `docs/streaming/PHASE_4_KAFKA_LIVE.md`, `docs/TODO.md`, `docs/DETECTION_REPORT.md`, `docs/PLAN_PHASE_1_DATABASE.md`.

**Interfaces:**
- Produces: docs whose module/artifact paths match the new layout. The grep audit in Step 3 is authoritative.

- [ ] **Step 1: Rewrite module paths in docs**

For each doc above, update any reference to a moved module to its new path. Apply these substitutions (module paths, both `snmp_anomaly_detection.X` and `snmp_anomaly_detection/X` slash form):
- power streaming/inference/training/eval/preprocessing modules → `…/power/…` (e.g. `snmp_anomaly_detection/streaming/detect_power_kafka.py` → `snmp_anomaly_detection/power/streaming/detect_power_kafka.py`)
- network modules → `…/network/…`

Run a guided pass:
```bash
cd /d/Aircod/snmp_anomaly_detection
git grep -nE "snmp_anomaly_detection[./](data|preprocessing|training|inference|streaming|evaluation|utils)[./]" -- '*.md' | grep -vE "/(network|power)/"
```
Edit each reported line to insert the correct `network/` or `power/` segment (use the same domain mapping as Tasks 3–4). `config.py`, `main.py`, `models/lstm_autoencoder.py` references stay unchanged.

- [ ] **Step 2: Rewrite artifact (`outputs/`) paths in docs**

Update artifact path references to the new layout:
```bash
git grep -lE "outputs/power_baseline|outputs/power_dual|outputs/power_phase|outputs/battery_rul|utils/scaler\.pkl" -- '*.md' \
  | xargs -r sed -i -e 's#outputs/power_baseline#outputs/power/baseline#g' \
                    -e 's#outputs/power_dual#outputs/power/dual#g' \
                    -e 's#outputs/power_phase#outputs/power/phase#g' \
                    -e 's#outputs/battery_rul#outputs/power/battery_rul#g' \
                    -e 's#utils/scaler\.pkl#outputs/network/scaler.pkl#g'
```
Also update the handoff's seed-model references (`SYSTEM_DESIGN_HANDOFF.md`, `config/migrations/SYSTEM_DESIGN_HANDOFF.md`) so `outputs/power_baseline/` and `outputs/power_dual/` read `outputs/power/baseline/` and `outputs/power/dual/`, and `streaming/detect_power_kafka.py` reads `power/streaming/detect_power_kafka.py` (covered by Steps 1–2 substitutions; verify in Step 3).

Do NOT touch the stale 15-feature / phase-model narrative in `CLAUDE.md` — out of scope.

- [ ] **Step 3: Grep audit — no stale module or artifact paths remain anywhere**

```bash
cd /d/Aircod/snmp_anomaly_detection
echo "--- code+docs module-path audit ---"
git grep -nE "snmp_anomaly_detection[./](data|preprocessing|training|inference|streaming|evaluation|utils)[./]" \
  | grep -vE "[./](network|power)[./]" \
  | grep -vE "snmp_anomaly_detection[./](config|main|models)" || echo "MODULE PATHS CLEAN"
echo "--- artifact-path audit ---"
git grep -nE "outputs/power_baseline|outputs/power_dual|outputs/power_phase|outputs/battery_rul|utils/scaler\.pkl" || echo "ARTIFACT PATHS CLEAN"
```
Expected: `MODULE PATHS CLEAN` and `ARTIFACT PATHS CLEAN`. Investigate and fix any reported line. (The `models` exclusion preserves the legitimate `models/lstm_autoencoder.py` and `models/battery_rul` — note `power/models/battery_rul` already contains `power/` so it passes.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs: update module and artifact path references for domain split"
```

---

## Task 8: Final verification gate

**Files:** none (verification only).

**Interfaces:**
- Consumes: the fully migrated package.
- Produces: green confirmation across compile, import, CLI, and grep.

- [ ] **Step 1: compileall**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe -m compileall -q snmp_anomaly_detection && echo "COMPILE OK"
```
Expected: `COMPILE OK`.

- [ ] **Step 2: Full import smoke test**

```bash
.venv/Scripts/python.exe - <<'PY'
import importlib, pkgutil
import snmp_anomaly_detection as pkg
failed = []
count = 0
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    count += 1
    try:
        importlib.import_module(m.name)
    except Exception as e:
        failed.append((m.name, repr(e)))
print(f"checked {count} modules")
if failed:
    print("FAILURES:"); [print(" ", n, e) for n, e in failed]
else:
    print("ALL IMPORTS OK")
import sys; sys.exit(1 if failed else 0)
PY
```
Expected: `ALL IMPORTS OK`.

- [ ] **Step 3: CLI parity**

```bash
.venv/Scripts/python.exe -c "import snmp_anomaly_detection.main as m; print('\n'.join(sorted(m.build_parser()._actions[1].choices)))" > /tmp/cli_steps_final.txt
diff /tmp/cli_steps_baseline.txt /tmp/cli_steps_final.txt && echo "CLI STEPS UNCHANGED"
```
Expected: `CLI STEPS UNCHANGED`.

- [ ] **Step 4: End-to-end smoke of one cheap step per pipeline (no torch needed)**

```bash
cd /d/Aircod/snmp_anomaly_detection
.venv/Scripts/python.exe -m snmp_anomaly_detection generate-data
ls snmp_anomaly_detection/data/synthetic_snmp_dataset.csv && echo "NETWORK GEN OK"
.venv/Scripts/python.exe -m snmp_anomaly_detection generate-power-data
ls snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv && echo "POWER GEN OK"
```
Expected: both datasets generated; `NETWORK GEN OK` and `POWER GEN OK`. (These exercise the split `dataset_builder` halves through the real CLI without needing torch.)

- [ ] **Step 5: Final grep audit (authoritative)**

```bash
git grep -nE "snmp_anomaly_detection[./](data|preprocessing|training|inference|streaming|evaluation|utils)[./]" \
  | grep -vE "[./](network|power)[./]" \
  | grep -vE "snmp_anomaly_detection[./](config|main|models)" || echo "ALL CLEAN"
```
Expected: `ALL CLEAN`.

- [ ] **Step 6: Final commit (if Step 4 produced regenerated artifacts that should not be tracked)**

Confirm `outputs/` and generated CSVs are gitignored (they were not tracked before). If `git status` is clean, no commit needed. Otherwise:
```bash
git status
# if only ignored artifacts changed → nothing to commit
```

---

## Self-Review

**Spec coverage:**
- §2 target structure → Tasks 2–5. ✓
- §3 file-move map (network) → Task 3; (power) → Task 4. ✓
- §3 `dataset_builder.py` split → Task 3 Step 2 + Task 4 Step 2. ✓
- §3 `events.py` split → Task 3 Step 3 + Task 4 Step 3. ✓
- §3 `inference/__init__.py` relocation → Task 3 Steps 4–5. ✓
- §3 remove emptied old packages → Task 4 Step 6. ✓
- §4 import-rewrite rules (config/lstm untouched) → Tasks 3 Step 6, 4 Step 5 (scoped), 5 (main.py). ✓
- §5 remove `utils/` → Task 6 Steps 3–4. ✓
- §6 outputs separation via ProjectPaths → Task 6 Steps 1–2; ensure_dirs → Step 3. ✓
- §7 docs path-only updates → Task 7. ✓
- §8 `power_alerts_topic` move → Task 4 Step 4. ✓
- §9 workflow + verification (branch, venv, compileall, import smoke, grep) → Tasks 1, 3, 4, 5, 8. ✓
- §9 success criteria (CLI parity, artifacts under new dirs) → Task 5 Step 4, Task 6 Step 5, Task 8. ✓

No spec requirement is left without a task.

**Placeholder scan:** No TBD/TODO/"handle appropriately" steps; the two file splits and all import edits show exact code or exact substitutions.

**Type consistency:** `NormalizedEvent` and `PowerEvent` reproduce the original field/method signatures verbatim; `POWER_ALERTS_TOPIC` is defined in Task 4 Step 4 and referenced only there; `ProjectPaths` field names are preserved (only values change), so its 27 importers stay valid.
