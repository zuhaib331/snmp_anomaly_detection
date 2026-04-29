# SNMP Anomaly Detection — Task Backlog

<!-- Managed list — update Status field as work progresses. -->
<!-- Priority order: F1 → F2 → F3 → F4 → F5/F6/F7 → B1 → B2 → B3 → E1 → C1 → D1/D2/D3/D4 -->

<!-- ================================================================ -->
<!-- F-series: Code review bug fixes (must resolve before feature work) -->
<!-- ================================================================ -->

## F1 — Fix `TrainingConfig.threshold_std_multiplier` regression in network pipeline
**Status:** Done — 2026-04-29  
**Priority:** Critical — silent breakage of existing network pipeline threshold; no test to catch it  
**File:** [config.py:150](../snmp_anomaly_detection/config.py#L150)

### Issue
`TrainingConfig.threshold_std_multiplier` was changed from `3.0` to `2.0` in this branch. `TrainingConfig` is used by the **network** pipeline's `train_model.py`, not only the power pipeline. Lowering to 2σ raises false-positive rates on network anomaly detection without any network-side evaluation.

### Fix
Revert `TrainingConfig.threshold_std_multiplier` back to `3.0`. Power models already use `PowerTrainingConfig` which has its own independent `threshold_std_multiplier = 3.0`, so no power-pipeline change is needed.

---

## F2 — Fix validation scaler data leakage in `train_baseline_power.py`
**Status:** Done — 2026-04-29  
**Priority:** High — validation split is scaled with its own fitted scaler, not the training scaler  
**File:** [train_baseline_power.py:710](../snmp_anomaly_detection/training/train_baseline_power.py#L710)

### Issue
```python
scaled_val, _ = scale_features(val_df, paths, paths.power_outputs_dir / "baseline_val_scaler.pkl")
```
`scale_features` always calls `fit_transform`, so a new `RobustScaler` is fitted on the validation set instead of transforming with the training scaler. The saved `baseline_val_scaler.pkl` is never loaded anywhere (dead artifact). Validation loss therefore reflects a different feature space than inference.

### Fix
Return the fitted scaler from the training call and apply `.transform()` directly on val data:
```python
scaled_train, scaler = scale_features(train_df, paths)
val_feature_cols = [c for c in BASELINE_UPS_FEATURES if c in val_df.columns]
val_out = val_df.copy()
val_out[val_feature_cols] = scaler.transform(val_out[val_feature_cols])
```
Remove the `baseline_val_scaler.pkl` artifact path.

---

## F3 — Fix `preprocess-power` producing sequences without delta features
**Status:** Done — 2026-04-29  
**Priority:** High — `X_train.npy` from `preprocess-power` has 14 features; training uses 16  
**File:** [power_features.py:727](../snmp_anomaly_detection/preprocessing/power_features.py#L727)

### Issue
`run_power_feature_engineering` (invoked by `preprocess-power`) calls:
```
aggregate_phase_metrics → apply_log1p_skewed → filter_normal_rows → scale_features
```
It never calls `add_delta_features`, so the saved `X_train.npy` lacks `runtime_delta` and `battery_charge_delta`. Training scripts call `add_delta_features` independently and don't load `X_train.npy`, so models are trained correctly — but the `preprocess-power` output is silently wrong and misleading.

### Fix
Add `df = add_delta_features(df)` after `apply_log1p_skewed` in `run_power_feature_engineering`.

---

## F4 — Fix `derive_rul_labels` ignoring device install age
**Status:** Pending  
**Priority:** High — RUL labels for new and end-of-life batteries are identical at dataset start  
**File:** [battery_features.py:52](../snmp_anomaly_detection/preprocessing/battery_features.py#L52)

### Issue
```python
age_days = i * _INTERVAL_MINUTES / 1440
rul = max(life_days - age_days, 0.0)
```
Age is always reset to 0 at dataset start for every device. `ups_apc_01` (90 days old) and `ups_apc_02` (900 days old, near end-of-life) receive identical RUL labels. The RUL model cannot learn to differentiate battery age from health signals — it only observes intra-dataset aging (~7 days), not the years of accumulated degradation.

### Fix
Persist `install_age_days` from `PowerDeviceProfile` as a column in the synthetic CSV during `build_power_dataset`. In `derive_rul_labels`, use it as the starting age offset:
```python
install_age = group["install_age_days"].iloc[0] if "install_age_days" in group.columns else 0.0
age_days = install_age + i * _INTERVAL_MINUTES / 1440
```

---

## F5 — Expose imbalance constants as public API in `phase_features.py`
**Status:** Pending  
**Priority:** Low — private names (`_IMBALANCE_COLS`, `_IMBALANCE_CLIP_MAX`) used across two inference modules  
**Files:** [dual_model_scorer.py:194](../snmp_anomaly_detection/inference/dual_model_scorer.py#L194), [power_stream_processor.py:371](../snmp_anomaly_detection/inference/power_stream_processor.py#L371)

### Fix
Rename `_IMBALANCE_COLS` → `IMBALANCE_COLS` and `_IMBALANCE_CLIP_MAX` → `IMBALANCE_CLIP_MAX` in `phase_features.py`. Update both import sites.

---

## F6 — Stop `process_event` from mutating its `PowerEvent` argument
**Status:** Pending  
**Priority:** Low — in-place mutation of caller's dict is surprising; breaks replay/retry  
**File:** [power_stream_processor.py:498](../snmp_anomaly_detection/inference/power_stream_processor.py#L498)

### Issue
```python
event.feature_values[col] = (
    min(max(event.feature_values[col], 0.0), _IMBALANCE_CLIP_MAX)
    / _IMBALANCE_CLIP_MAX
)
```
The imbalance clipping overwrites the original event values. A caller that logs the raw event before scoring will see silently modified values.

### Fix
Apply the clipping transform to a local dict copy before building `p_raw`:
```python
clipped = dict(event.feature_values)
for col in IMBALANCE_COLS:
    if col in clipped:
        clipped[col] = min(max(clipped[col], 0.0), IMBALANCE_CLIP_MAX) / IMBALANCE_CLIP_MAX
p_raw = np.array([[clipped.get(c, 0.0) for c in PHASE_LEVEL_FEATURES]])
```

---

## F7 — Replace `battery_voltage_v > 0` UPS proxy with `device_category == "ups"`
**Status:** Pending  
**Priority:** Low — voltage can be 0 during a fault, causing UPS to be silently excluded from RUL scoring  
**File:** [rul_eval.py:103](../snmp_anomaly_detection/evaluation/rul_eval.py#L103)

### Fix
```python
# Before
ups_df = df[df["battery_voltage_v"] > 0].copy()
# After
ups_df = df[df["device_category"] == "ups"].copy()
```

---

## B1 — Per-device online threshold in PowerStreamProcessor
**Status:** Done — 2026-04-29  
**Priority:** High — fixes current FPs, scales to millions of devices, no retraining required  
**Depends on:** Nothing (can start immediately)

### What to build
Add `DeviceErrorStats` dataclass to [power_stream_processor.py](../snmp_anomaly_detection/inference/power_stream_processor.py):

```python
@dataclass
class DeviceErrorStats:
    calibration_errors: list[float]  # collected during cold-start
    calibrated: bool                 # False until N windows seen
    rolling_mean: float
    rolling_var: float               # Welford online variance
    n_windows: int                   # total windows scored
```

### Behaviour
1. **Cold-start**: first `N_CALIBRATION_WINDOWS` (e.g. 50) windows per device collect errors silently — no alerts raised. Per-category threshold used as safety net for extreme values only (overload rule still active).
2. **Calibration complete**: `threshold[device] = mean(calibration_errors) + k × std(...)`
3. **Live**: each normal window (`error ≤ 2.5 × threshold`) updates rolling stats via Welford algorithm. Anomalous windows are locked out from stats update.
4. **Fallback chain**: device threshold → per-category threshold → global threshold.

### Also required
- Load `per_category_thresholds` from `baseline_metadata.json` into `PowerStreamProcessor` (already computed at training time, currently unused in the streaming path).
- Persist `DeviceErrorStats` to disk (JSON) so state survives consumer restarts.
- Log when a device transitions from calibration → live.

### Validation
```bash
# Smoke test — expect zero alerts
python3 -m snmp_anomaly_detection produce-power-kafka-test \
  --use-training-profiles --anomaly-probability 0.0

# Random fleet smoke test — also expect zero alerts after calibration window
python3 -m snmp_anomaly_detection produce-power-kafka-test \
  --anomaly-probability 0.0
```

---

## B2 — Expand training profiles and retrain
**Status:** Pending  
**Priority:** Medium — improves cold-start quality for new/unseen devices  
**Depends on:** B1 must be completed first

### Why
B1 fixes FPs for calibrated devices. But during the cold-start calibration window, a brand-new device (never seen in training) falls back to the per-category threshold. If that threshold is poorly calibrated for the device's actual capacity/phase/vendor, it can cause FPs during the first ~50 windows. B2 makes the per-category fallback threshold more robust.

### What to build
Expand `_DEFAULT_POWER_PROFILES` in [dataset_builder.py](../snmp_anomaly_detection/data/dataset_builder.py) to cover configurations not currently in training:

| Missing profile | Why it matters |
|---|---|
| `generic` UPS, single-phase, 120V, 3000/6000/10000 W | Largest FP source in random fleet tests |
| Single-phase Liebert UPS | All training Liebert are 3-phase — causes phase model FPs |
| 3-phase APC PDU | All training APC PDU are single-phase |
| Higher-wattage APC UPS at 120V (6000/9000 W) | `output_current_a` goes OOD for scaler |

### Retraining pipeline
```bash
python3 -m snmp_anomaly_detection generate-power-data
python3 -m snmp_anomaly_detection preprocess-power
python3 -m snmp_anomaly_detection train-power-baseline
python3 -m snmp_anomaly_detection train-power-phase
python3 -m snmp_anomaly_detection evaluate-power-baseline
```

### Validation
```bash
# Random fleet smoke test with no --use-training-profiles flag
# Per-category cold-start threshold should now be clean for diverse devices
python3 -m snmp_anomaly_detection produce-power-kafka-test \
  --anomaly-probability 0.0 --seed 42
```

---

## B3 — Phase sag recall and Liebert 3-phase accuracy
**Status:** Pending  
**Priority:** Medium — improves phase model recall for voltage fault types; fixes Liebert-specific FPs  
**Depends on:** B2 (Liebert single-phase profiles must be in training data first)

### Issue 1 — Phase sag recall (voltage features under-weighted in global MSE)

The phase model uses a single scalar MSE threshold per device category. Phase sag faults only disturb `input_voltage_l1`, `voltage_imbalance_pct`, and `output_voltage_v` — a small subset of the 29 `PHASE_LEVEL_FEATURES`. Their contribution to MSE is diluted by the 26 unaffected features, causing missed detections.

Two options (pick one):

**Option A — `voltage_drop_delta` feature (preferred):** Add a per-phase rate-of-change feature that amplifies the sag signal before MSE:

```python
# In power_features.py, after log1p transforms:
df["voltage_drop_delta_l1"] = df["input_voltage_l1"].diff().clip(upper=0)  # negative = drop
df["voltage_drop_delta_l2"] = df["input_voltage_l2"].diff().clip(upper=0)
df["voltage_drop_delta_l3"] = df["input_voltage_l3"].diff().clip(upper=0)
```

Add the three new columns to `PHASE_LEVEL_FEATURES` in [config.py](../snmp_anomaly_detection/config.py). Retrain phase model only (baseline model is unaffected).

**Option B — per-feature MSE weights:** Apply a weight vector in `dual_model_scorer.py` that up-weights voltage columns before computing the scalar error. Lower implementation risk but harder to tune.

### Issue 2 — Liebert 3-phase `voltage_imbalance_pct` / `current_skew_pct` correctness

The imbalance/skew formulas in [dataset_builder.py:387-398](../snmp_anomaly_detection/data/dataset_builder.py#L387) are vendor-agnostic. Liebert devices report per-phase voltages under different OID paths (Emerson/Liebert MIB vs. APC POWERNET-MIB), so raw values may arrive in different units or scale. Two actions needed:

1. **Audit**: confirm that `input_voltage_l1/l2/l3` in the dataset correctly reflects 230V-nominal Liebert readings at normal load (should be ~228–232 V per phase, not ~120 V).
2. **Separate fault label (if warranted)**: if Liebert imbalance behaves differently under `phase_sag` (e.g. higher baseline variance), add a `liebert_phase_sag` anomaly type in `_anomaly_types_for_category` and a corresponding `_inject_anomaly` branch so the model sees realistic Liebert fault signatures during training.

### Validation
```bash
# After Option A retraining — check phase sag recall is ≥ 0.80
python3 -m snmp_anomaly_detection evaluate-power-baseline   # compare phase F1 before/after

# Liebert-only audit
python3 - <<'EOF'
import pandas as pd
df = pd.read_csv("snmp_anomaly_detection/data/synthetic_power_snmp_dataset.csv")
lie = df[df["vendor"] == "liebert"]
print(lie[["input_voltage_l1","input_voltage_l2","input_voltage_l3","voltage_imbalance_pct"]].describe())
EOF
```

---

## C1 — ClickHouse storage integration
**Status:** Pending  
**Priority:** Medium — production readiness, replaces .jsonl flat file outputs  
**Depends on:** B1 and B2 complete (streaming path must be stable before wiring storage)

### Why
Replace `.jsonl` flat file outputs with a queryable, production-grade columnar store. Enables Grafana dashboards for live anomaly rates, RUL trends, and per-device heatmaps. Also stores raw telemetry for future retraining without manual dataset rebuilds.

### Steps
1. Stand up ClickHouse via Docker for dev:
   ```bash
   docker run -d --name clickhouse -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server
   ```
2. Create `snmp_raw` and `anomaly_windows` tables — MergeTree engine, partitioned by month, ordered by `(device_id, timestamp)`.
3. Set up Kafka Materialized View on `snmp-live-events` and `snmp-power-events` for automatic raw ingestion (no code change to producers).
4. Modify [detect_kafka.py](../snmp_anomaly_detection/streaming/detect_kafka.py) to write scored windows to ClickHouse via `clickhouse-connect` Python client.
5. Do the same for [detect_power_kafka.py](../snmp_anomaly_detection/streaming/detect_power_kafka.py) (power pipeline results).
6. Connect Grafana to ClickHouse — dashboards: anomaly rate per device, RUL trend, per-category heatmap.

### New dependency
```
clickhouse-connect>=0.7
```

---

<!-- ================================================================ -->
<!-- E-series: Isolation Forest second model                          -->
<!-- ================================================================ -->

## E1 — Isolation Forest as a real second detection model
**Status:** Pending  
**Priority:** High — the NOC dashboard already shows an "Isolation Forest" panel as if it is live; currently no IF model exists anywhere in the pipeline  
**Depends on:** B1 and B2 complete (streaming path and training data should be stable before adding a third scoring signal)

### Why
The LSTM Autoencoder detects temporal anomalies well (gradual drifts, ramp-ups) but can miss point anomalies — sudden single-timestep spikes that look normal in a sequence context. Isolation Forest is a strong complement: it scores individual feature vectors in isolation, independent of sequence order, making it sensitive to exactly the fault types LSTM under-detects (instantaneous voltage spikes, sudden load jumps).

### What to build

**1. Training — new file `training/train_iforest_power.py`**

```python
from sklearn.ensemble import IsolationForest
import joblib

# Train one IF model per device category on normal (anomaly==0) rows only.
# Use BASELINE_UPS_FEATURES (flat, no sequence needed).
# contamination="auto" — revisit after smoke testing.
models: dict[str, IsolationForest] = {}
for category, group in df[df["anomaly"] == 0].groupby("device_category"):
    X = scaler.transform(group[BASELINE_UPS_FEATURES])
    clf = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    clf.fit(X)
    models[category] = clf

joblib.dump(models, paths.power_dual_outputs_dir / "iforest_models.pkl")
```

Save per-category score thresholds (1st percentile of training scores) to `iforest_metadata.json` alongside the `.pkl`.

**2. Artifact path — add to `config.py` `ProjectPaths`**

```python
iforest_model_file: Path = field(
    default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_models.pkl"
)
iforest_metadata_file: Path = field(
    default_factory=lambda: PACKAGE_ROOT / "outputs" / "power_dual" / "iforest_metadata.json"
)
```

**3. CLI command — add to `main.py`**

```python
"train-power-iforest": train_iforest_main,
```

**4. Scoring — extend `DualModelResult` in `dual_model_scorer.py`**

Add two fields to the dataclass:
```python
iforest_score: float    # raw anomaly score (more negative = more anomalous)
iforest_flag: int       # 1 if score < per-category 1st-percentile threshold
```

In `score_windows()`, load `iforest_models.pkl` once at startup. For each window, take the **last timestep's** feature vector (most recent observation), run `clf.score_samples(X)`, store the score, and threshold against `iforest_metadata.json`.

Update `final_flag`:
```python
final_flag = combined_flag OR iforest_flag OR overload_rule_flag
```

**5. Output schema — update `anomaly_results.csv` / `.jsonl`**

Add `iforest_score` and `iforest_flag` columns so D1 (dashboard chart) can plot the IF score line alongside the LSTM line.

### Artifact layout
```
outputs/power_dual/
  iforest_models.pkl        # dict[category → IsolationForest]
  iforest_metadata.json     # {"ups": {"threshold": -0.12, ...}, ...}
```

### Validation
```bash
# Train IF models (run after train-power-baseline)
python3 -m snmp_anomaly_detection train-power-iforest

# Smoke test — normal traffic, expect zero IF flags
python3 -m snmp_anomaly_detection detect-power-csv

# Verify schema and flag rate
python3 -c "
import pandas as pd
df = pd.read_csv('snmp_anomaly_detection/outputs/anomaly_results.csv')
print(df[['device_id','iforest_score','iforest_flag','baseline_anomaly','final_flag']].head(20))
print('IF flag rate:', df['iforest_flag'].mean())
"
```

Target: IF flag rate < 0.02 on normal traffic; ≥ 0.70 on injected anomaly rows.

---

<!-- ================================================================ -->
<!-- D-series: NOC Alarms Realtime dashboard (NOC Alarms Realtime.html) -->
<!-- ================================================================ -->

## D1 — Wire charts to real pipeline data
**Status:** Pending  
**Priority:** High — both chart panels are empty; the dashboard is effectively non-functional without them  
**Depends on:** C1 (ClickHouse) OR interim: poll `anomaly_results.csv` / `.jsonl` files directly

### What to fix
Two chart areas render blank in the current HTML:

1. **Anomaly Score Trend (24h)** — LSTM and IForest score lines over time. Should read from `snmp_anomaly_detection/outputs/anomaly_results.csv` (or `power_dual/anomaly_windows_detail.json`) and plot `reconstruction_error` (LSTM) and iforest score per timestamp.
2. **Alarm Trends (1H / 6H / 24H)** — alarm count bucketed by severity over time. Source: same output files, grouped by `anomaly_type` and binned into time buckets matching the selected range button.

Interim approach (before C1): use a `setInterval` fetch of the local `.jsonl`/`.csv` via a small Python Flask dev server, or convert outputs to a static JSON that the HTML polls.
Long-term: replace polling with a ClickHouse HTTP query (port 8123) once C1 is done.

---

## D2 — Fix AI Anomalies counter sync
**Status:** Pending  
**Priority:** High — UI shows inconsistent state: IForest panel says "3 Outliers Detected", Combined AI Alerts says "2 High Confidence", but the top-right "AI Anomalies" tile shows 0  
**Depends on:** Nothing (pure frontend bug)

### What to fix
The top summary tiles (All Alarms, Critical, Major, Minor, Warning, Cleared, AI Anomalies) are not reading from the same data source as the AI Anomaly Detection sub-panel. Ensure a single shared state object drives both. The "AI Anomalies" tile count should equal the number of rows where `dual_model_flag == 1` (or equivalent) in the active time window.

---

## D3 — Populate "Recent AI-Detected Anomalies" table
**Status:** Pending  
**Priority:** Medium — section renders empty; no anomaly rows visible  
**Depends on:** D1 (data source must be wired first)

### What to build
Read the latest N anomaly rows from the data source (`.jsonl` or ClickHouse) and render them in the table with columns:

| Column | Source field |
|---|---|
| Timestamp | `timestamp` |
| Device ID | `device_id` |
| Category | `device_category` |
| Anomaly Type | `anomaly_type` |
| LSTM Error | `reconstruction_error` |
| Phase Error | `phase_error` (if present) |
| Confidence | derived from dual-model OR policy |
| Status badge | Critical / Investigating / Action Required |

Support the "View All" link — either paginate in-page or open a filtered view.

---

## D4 — Implement unfinished UI actions
**Status:** Pending  
**Priority:** Low — buttons exist but are non-functional  
**Depends on:** D1 (need data before actions make sense)

### Items
| Control | What it should do |
|---|---|
| **Acknowledge** button | Mark selected alarm rows as acknowledged; persist state (localStorage or backend endpoint) |
| **Auto-Rules (0)** badge | Open a rule-builder panel to define suppress/escalate rules on alarm fields; count reflects active rules |
| **Download** icon | Export current filtered alarm table to CSV |
| **Sound Off** toggle | Play an alert tone (Web Audio API) when a new Critical/AI anomaly arrives; respect the toggle state |
| **Query Builder** | Parse structured filter expressions and apply them to the alarm list (e.g. `device_id:ups_apc_01 AND severity:critical`) |
| **Source / Impact toggles** | Already rendered — wire them to filter the alarm list by source and impact dimensions |
