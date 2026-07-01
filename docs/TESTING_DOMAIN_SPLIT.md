# Testing Guide — `snmp-network-power-domain-split`

How to verify the network/power domain-split refactor on branch
`snmp-network-power-domain-split`. The refactor is **pure structural** (no behavior
change): the CLI surface is identical, and every module was moved by domain into
`network/` and `power/` subpackages with shared code (`config.py`, `main.py`,
`models/lstm_autoencoder.py`) kept at the top level. Artifacts now write to
`outputs/network/` and `outputs/power/`.

This guide has two parts: **structural verification** (proves the move is correct and
imports resolve) and **functional testing** (runs the pipelines end-to-end).

> Platform note: commands below use **Git Bash** on Windows and the project virtualenv
> interpreter `.venv/Scripts/python.exe`. On macOS/Linux, activate the venv and use
> `python` instead.

---

## 0. Setup

```bash
cd /d/Aircod/snmp_anomaly_detection
git checkout snmp-network-power-domain-split
git pull

# Create the virtualenv and install dependencies (once).
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `torch`, `kafka-python`.
The `torch` wheel is large — first install takes several minutes.

> This project has **no pytest suite**. "Testing" a structural refactor means proving
> every module still imports from its new location, the CLI is unchanged, and the
> pipelines still run — not unit tests. The checks below are that proof.

---

## 1. Structural verification

### 1.1 Byte-compile the whole package

```bash
.venv/Scripts/python.exe -m compileall -q snmp_anomaly_detection && echo "COMPILE OK"
```
**Expected:** `COMPILE OK` (no syntax errors anywhere).

### 1.2 Import smoke test — every module resolves from its new path

This is the key test for the refactor: it imports every module and reports failures.
It skips `__main__` (which runs argparse and would exit) and catches `SystemExit`.

```bash
.venv/Scripts/python.exe - <<'PY'
import importlib, pkgutil
import snmp_anomaly_detection as pkg
failed, count = [], 0
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    if m.name.endswith(".__main__"):
        continue
    count += 1
    try:
        importlib.import_module(m.name)
    except BaseException as e:
        failed.append((m.name, type(e).__name__, str(e)[:90]))
print(f"checked {count} modules")
for n, t, e in failed:
    print("  FAIL", n, "->", t, e)
print(f"{len(failed)} failures")
PY
```

**Expected:** `checked 53 modules` and **exactly 2 failures**, both pre-existing:

```
  FAIL snmp_anomaly_detection.power.preprocessing.phase_features -> ImportError cannot import name 'PHASE_V_COLS' ...
  FAIL snmp_anomaly_detection.power.training.train_phase_model   -> ImportError cannot import name 'aggregate_phase_metrics' ...
2 failures
```

> ⚠️ These two failures are **NOT caused by this branch.** They are dead files left
> from the earlier phase-model removal and fail identically on the pre-refactor commit
> `f677e4e`. To prove that, see §1.5. **Any OTHER import failure is a real regression.**

### 1.3 CLI surface is unchanged

The set of CLI steps must be identical to before the refactor.

```bash
.venv/Scripts/python.exe -c "import snmp_anomaly_detection.main as m; print('\n'.join(sorted(m.build_parser()._actions[1].choices)))"
```
**Expected — exactly these 19 steps:**
```
detect
detect-csv
detect-kafka
detect-kafka-dry
detect-power-csv
detect-power-kafka
evaluate-power-baseline
generate-data
generate-power-data
predict-battery-rul
preprocess
preprocess-power
produce-kafka-test-data
produce-power-kafka-test
train
train-battery-rul
train-power-baseline
train-power-iforest
train-power-phase
```

### 1.4 Path audits — no stale module or artifact paths remain

```bash
# Module paths: nothing should point at the old flat layout (config/main/models excepted).
git grep -nE "snmp_anomaly_detection[./](data|preprocessing|training|inference|streaming|evaluation|utils)[./]" -- ':!docs/superpowers/' \
  | grep -vE "[./](network|power)[./]" \
  | grep -vE "snmp_anomaly_detection[./](config|main|models)" || echo "MODULE PATHS CLEAN"

# Artifact paths: no old outputs/ layout references.
git grep -nE "outputs/power_baseline|outputs/power_dual|outputs/power_phase|outputs/battery_rul|utils/scaler\.pkl" -- ':!docs/superpowers/' || echo "ARTIFACT PATHS CLEAN"
```
**Expected:** `MODULE PATHS CLEAN` and `ARTIFACT PATHS CLEAN`.

> The `docs/superpowers/` design + plan docs are excluded on purpose — they document the
> migration itself using old→new path notation.

### 1.5 (Optional) Prove the 2 phase failures are pre-existing

```bash
git stash -u
git switch --detach f677e4e
.venv/Scripts/python.exe -c "import snmp_anomaly_detection.preprocessing.phase_features" 2>&1 | tail -1
git switch snmp-network-power-domain-split
git stash pop 2>/dev/null || true
```
**Expected:** the same `ImportError: cannot import name 'PHASE_V_COLS'` on the pre-refactor
commit — confirming the branch only *moved* the dead file, it did not break it.

---

## 2. Functional testing — run the pipelines end-to-end

Run steps in order (generate → preprocess → train → detect). Detection loads the trained
artifacts, so training must come first.

### 2.1 Network pipeline (routers/switches/firewalls — 5 metrics)

```bash
.venv/Scripts/python.exe -m snmp_anomaly_detection generate-data
.venv/Scripts/python.exe -m snmp_anomaly_detection preprocess
.venv/Scripts/python.exe -m snmp_anomaly_detection train
.venv/Scripts/python.exe -m snmp_anomaly_detection detect-csv
```
**Expected artifacts (new layout):**
- `snmp_anomaly_detection/data/synthetic_snmp_dataset.csv`
- `snmp_anomaly_detection/outputs/network/X_train.npy`, `scaler.pkl`
- `snmp_anomaly_detection/outputs/network/lstm_autoencoder.pth`, `model_metadata.json`
- `snmp_anomaly_detection/outputs/network/anomaly_results.csv`

Quick check that outputs landed under `network/`:
```bash
ls snmp_anomaly_detection/outputs/network/
```

### 2.2 Power pipeline (UPS/PDU/PSU/env — LSTM AE + Isolation Forest + battery RUL)

```bash
.venv/Scripts/python.exe -m snmp_anomaly_detection generate-power-data
.venv/Scripts/python.exe -m snmp_anomaly_detection preprocess-power
.venv/Scripts/python.exe -m snmp_anomaly_detection train-power-baseline
.venv/Scripts/python.exe -m snmp_anomaly_detection train-power-iforest
.venv/Scripts/python.exe -m snmp_anomaly_detection train-battery-rul
.venv/Scripts/python.exe -m snmp_anomaly_detection detect-power-csv
.venv/Scripts/python.exe -m snmp_anomaly_detection evaluate-power-baseline
.venv/Scripts/python.exe -m snmp_anomaly_detection predict-battery-rul
```
**Expected artifacts (new layout):**
- `snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv`
- `snmp_anomaly_detection/outputs/power/baseline/` — `baseline_model.pt`, `baseline_scaler.pkl`, `baseline_metadata.json`
- `snmp_anomaly_detection/outputs/power/dual/` — `iforest_models.pkl`, `anomaly_results.csv`, `detection_summary.json`
- `snmp_anomaly_detection/outputs/power/battery_rul/` — `rul_model.pt`, `rul_metrics.json`

Quick check:
```bash
ls snmp_anomaly_detection/outputs/power/baseline/ snmp_anomaly_detection/outputs/power/dual/ snmp_anomaly_detection/outputs/power/battery_rul/
```

> **Do NOT run `train-power-phase`.** The phase model was removed; this step is a
> known no-op / broken (tracked as a separate follow-up). It appears in the CLI list
> only because of a pre-existing dispatch gap.

### 2.3 (Optional) Streaming paths — require a local Kafka broker

Only if you have Kafka running on `localhost:9092`:
```bash
.venv/Scripts/python.exe -m snmp_anomaly_detection detect-kafka-dry     # validates parsing, no models
.venv/Scripts/python.exe -m snmp_anomaly_detection detect-kafka         # network live detection
.venv/Scripts/python.exe -m snmp_anomaly_detection detect-power-kafka   # power live detection
```
`detect-kafka-dry` needs no models and is the safest smoke test of the streaming path.

---

## 3. Pass / fail criteria

The branch **passes** when:

| Check | Pass condition |
|---|---|
| §1.1 compileall | prints `COMPILE OK` |
| §1.2 import smoke | 53 modules checked; **exactly** the 2 pre-existing phase failures, nothing else |
| §1.3 CLI steps | exactly the 19 steps listed (unchanged vs pre-refactor) |
| §1.4 path audits | `MODULE PATHS CLEAN` and `ARTIFACT PATHS CLEAN` |
| §2.1 network run | all 4 steps succeed; artifacts under `outputs/network/` |
| §2.2 power run | all steps succeed; artifacts under `outputs/power/{baseline,dual,battery_rul}` |

The branch **fails** if any import beyond the 2 known phase modules breaks, the CLI step
list differs, a path audit reports a stale path, or any pipeline step errors out.

---

## 4. Known pre-existing issues (not introduced by this branch)

These exist on `basemodel` too and are tracked as separate follow-ups — do not treat them
as regressions of this refactor:

1. `power/preprocessing/phase_features.py` and `power/training/train_phase_model.py` fail
   to import (dead files from the phase-model removal).
2. `train-power-phase` is listed in the CLI but has no working handler.
3. `CLAUDE.md` / `POWER_PIPELINE_EXPLAINED.md` still describe the removed phase model and
   the old 15-feature set; the code is now a 13-feature baseline + Isolation Forest.
