# SNMP Live Scoring Service — Architecture Improvement Notes
> Written for Claude Code. Work through these one by one before moving to production.
> Context: `detect_power_kafka.py` + `event_preprocessor.py` + alert state machine pipeline.

---

## Issue 1 — Delta computation ignores time gaps  
**Severity: HIGH**  
**File: `event_preprocessor.py` → `compute_deltas()`**

### Problem
`runtime_delta = log1p(now) - log1p(prev)` assumes the previous row arrived
recently. If a device goes quiet for several minutes and then sends a burst,
the delta spans the entire silent gap and looks like a massive spike — triggering
a false positive anomaly.

### Fix
Add a `MAX_DELTA_GAP_SECONDS` guard before computing any delta. If the time gap
between the current event and the previous row exceeds the threshold, zero out
all deltas for that row rather than computing a meaningless value.

```python
# event_preprocessor.py — inside compute_deltas()

MAX_DELTA_GAP_SECONDS = 120  # tune per your poll interval

time_gap = (current_timestamp - prev_timestamp).total_seconds()

if time_gap > MAX_DELTA_GAP_SECONDS:
    # gap too large — delta is meaningless, zero everything out
    feature_values["runtime_delta"]        = 0.0
    feature_values["temperature_delta"]    = 0.0
    feature_values["output_load_delta"]    = 0.0
    feature_values["voltage_drop_delta_L1"] = 0.0
    feature_values["voltage_drop_delta_L2"] = 0.0
    feature_values["voltage_drop_delta_L3"] = 0.0
    # also flag this row so downstream can optionally skip it
    feature_values["delta_gap_flag"] = 1
else:
    feature_values["delta_gap_flag"]       = 0
    feature_values["runtime_delta"]        = log1p(now_runtime) - log1p(prev_runtime)
    feature_values["temperature_delta"]    = now_temp - prev_temp
    feature_values["output_load_delta"]    = now_load - prev_load
    # voltage drop deltas — negative only (phase sag amplification)
    for phase in ["L1", "L2", "L3"]:
        raw_delta = now_voltage[phase] - prev_voltage[phase]
        feature_values[f"voltage_drop_delta_{phase}"] = min(raw_delta, 0.0)
```

### Notes
- Tune `MAX_DELTA_GAP_SECONDS` to roughly 2–3× your normal poll interval.
- Consider also adding `delta_gap_flag` as a feature so the model can learn
  that a gap-following row is less trustworthy.

---

## Issue 2 — RollingWindowBuffer is lost on service restart  
**Severity: MEDIUM**  
**File: wherever `RollingWindowBuffer` is defined + `detect_power_kafka.py`**

### Problem
The per-device rolling buffer (deque of 20 scaled rows) lives entirely in memory.
On any restart — deploy, crash, OOM kill — every device loses its buffer.
All devices go back to the calibration phase. Worse, if the service restarts
during an active anomaly event, the LSTM confirmation window is lost entirely.

### Fix
Periodically checkpoint buffer state to disk (or Redis for multi-instance setups).
On startup, attempt to restore from checkpoint, discarding stale ones.

```python
# rolling_window_buffer.py — add these two methods

import pickle
from pathlib import Path
from datetime import datetime, timedelta

CHECKPOINT_DIR = Path("checkpoints/buffers")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(self, device_id: str):
    path = CHECKPOINT_DIR / f"{device_id}.pkl"
    state = {
        "buffer":     list(self.buffer[device_id]),
        "saved_at":   datetime.utcnow().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_checkpoint(self, device_id: str, max_age_seconds: int = 300) -> bool:
    path = CHECKPOINT_DIR / f"{device_id}.pkl"
    if not path.exists():
        return False
    with open(path, "rb") as f:
        state = pickle.load(f)
    saved_at = datetime.fromisoformat(state["saved_at"])
    if (datetime.utcnow() - saved_at).total_seconds() > max_age_seconds:
        path.unlink()   # stale — discard
        return False
    self.buffer[device_id].extend(state["buffer"])
    return True
```

```python
# detect_power_kafka.py — in the main consume loop

EVENT_COUNT = 0

for message in consumer:
    result = process_event(message)
    EVENT_COUNT += 1

    # checkpoint every 50 events per device
    if EVENT_COUNT % 50 == 0:
        buffer.save_checkpoint(device_id=result.device_id)
```

### Notes
- For a single-instance deployment, disk checkpoints are fine.
- For multi-instance (horizontal scaling), replace with Redis:
  `HSET buffer:{device_id} rows <serialised>`
- `max_age_seconds=300` means a buffer older than 5 minutes is discarded on
  restore — prevents replaying stale data as if it were current.

---

## Issue 3 — PendingAlert has no TTL / expiry  
**Severity: MEDIUM**  
**File: alert state machine (wherever `PendingAlert` is stored)**

### Problem
When Isolation Forest fires but the LSTM rolling window is not yet full, a
`PendingAlert` is stored for that device. The state machine waits up to
`N_CONFIRMATION_WINDOWS` LSTM windows for confirmation. But if the device
goes quiet after IF fires — no more events arrive — the `PendingAlert` sits
in memory indefinitely. Over time this leaks memory and produces stale alerts
when the device eventually reconnects.

### Fix
Add a `created_at` timestamp and `TTL_SECONDS` to `PendingAlert`. Check expiry
before applying pending alert logic in the state machine.

```python
# events.py or wherever PendingAlert is defined

from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PendingAlert:
    if_score:            float
    if_threshold:        float
    if_peak_timestep:    int
    created_at:          datetime = field(default_factory=datetime.utcnow)
    lstm_windows_pending: int     = 0
    TTL_SECONDS:         int      = 300   # 5 minutes — tune as needed

    def is_expired(self) -> bool:
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.TTL_SECONDS
```

```python
# alert state machine — at the top of the evaluation block

pending = pending_alerts.get(device_id)

if pending and pending.is_expired():
    # device went quiet after IF fired — quietly discard
    del pending_alerts[device_id]
    pending = None
    # optionally log: "PendingAlert expired for {device_id}"
```

### Notes
- `TTL_SECONDS = 300` (5 min) is a safe starting point. If your LSTM window
  fills in ~2 min at normal poll rates, 5 min gives enough headroom.
- Consider emitting a low-severity `EXPIRED` alert state so operators know
  an unconfirmed IF flag was seen for a device.

---

## Issue 4 — IF-only result is CLEARED, hiding short-lived spikes  
**Severity: MEDIUM**  
**File: alert state machine**

### Problem
Current logic:
```
if_flag=1, lstm_flag=0 (LSTM window complete, did not confirm)
→ CLEARED  (treats IF as a false positive)
```
This means Isolation Forest can **never** produce a confirmed alert on its own.
A genuine single-event spike — a voltage transient, a brief overload — that
disappears before the LSTM window fills will be silently CLEARED. In power
monitoring, a one-poll-cycle spike is often exactly what you need to catch.

### Fix
Introduce a `CONFIRMED_IF_ONLY` state instead of `CLEARED` for this case.
Keep `CLEARED` only for when IF fired during a pending window and the
subsequent LSTM window also did not flag.

```python
# alert state machine — revise the "NO pending alert, fresh evaluation" branch

if if_flag and lstm_flagged:
    alert_state = AlertState.CONFIRMED           # both agree — high confidence

elif lstm_flagged and not if_flag:
    alert_state = AlertState.LSTM_ONLY           # sequence anomaly, no point event

elif if_flag and not lstm_flagged:
    # IF saw something, LSTM did not confirm over the full window
    # Do NOT clear — emit as medium-severity
    alert_state      = AlertState.CONFIRMED_IF_ONLY
    effective_if_flag = 1                        # still counts in final_flag

else:
    alert_state = AlertState.CLEARED             # nothing flagged

# Add to AlertState enum:
# CONFIRMED_IF_ONLY = "CONFIRMED_IF_ONLY"
```

```python
# format_power_alert() — add a line for the new state
# "CONFIRMED_IF_ONLY [IF] [ups] dev01 @ ts ... (LSTM did not confirm)"
```

### Notes
- `CONFIRMED_IF_ONLY` should carry a lower severity in your alerting system
  than `CONFIRMED` (both models agree).
- This change means IF alone CAN raise an alert — review your alert
  volume before deploying to make sure IF thresholds are well-tuned.

---

## Issue 5 — Compound alert check is O(n) scan  
**Severity: LOW**  
**File: compound alert check section of `detect_power_kafka.py`**

### Problem
Scanning "recent PDU anomalies within ±5 min" iterates over a list or recent
results. At high event rates from thousands of devices this becomes a bottleneck
as the recent-anomalies list grows.

### Fix
Replace the list scan with an O(1) dict keyed by device_id. Only store the
latest anomaly timestamp per device — that is all you need for the ±5 min check.

```python
# detect_power_kafka.py — module-level state

from collections import defaultdict
from datetime import datetime, timedelta

# {device_id: (timestamp, category)}
recent_anomaly_index: dict[str, tuple[datetime, str]] = {}

COMPOUND_WINDOW_SECONDS = 300  # ±5 minutes

def update_anomaly_index(device_id: str, category: str,
                          timestamp: datetime):
    recent_anomaly_index[device_id] = (timestamp, category)

def check_compound_alert(device_id: str, category: str,
                          timestamp: datetime) -> tuple[bool, str | None]:
    peer_category = "pdu" if category == "ups" else "ups"
    for did, (ts, cat) in recent_anomaly_index.items():
        if did == device_id:
            continue
        if cat != peer_category:
            continue
        if abs((timestamp - ts).total_seconds()) <= COMPOUND_WINDOW_SECONDS:
            return True, did
    return False, None

# Call update_anomaly_index() whenever final_flag=1
# Periodically prune stale entries (e.g. every 1000 events):
def prune_anomaly_index(cutoff_seconds: int = 600):
    now = datetime.utcnow()
    stale = [did for did, (ts, _) in recent_anomaly_index.items()
             if (now - ts).total_seconds() > cutoff_seconds]
    for did in stale:
        del recent_anomaly_index[did]
```

### Notes
- The inner loop is still O(devices with recent anomalies) but in practice
  that set is small (anomalies are rare). True O(1) would require a separate
  index per category — overkill unless you have 10k+ simultaneous anomalies.
- Add `prune_anomaly_index()` call every ~1000 events to prevent unbounded growth.

---

## Issue 6 — No model hot-reload mechanism  
**Severity: MEDIUM**  
**File: `detect_power_kafka.py` + new `api.py`**

### Problem
The Isolation Forest and LSTM models are loaded once at service startup from
disk. When the ML training service retrains and deploys new models, the scoring
service keeps using the old ones until it is manually restarted. A restart means
losing all rolling buffers (see Issue 2) and causes a detection gap.

### Fix
Add a lightweight FastAPI reload endpoint. The ML training DAG calls this
endpoint after deploying new model files — no restart needed.

```python
# live_scoring/api.py  (new file — run alongside the Kafka consumer)

from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading

app = FastAPI()
scorer_ref = {}   # mutable reference shared with consumer thread

@app.post("/reload-model")
def reload_model():
    from models.iso_forest_scorer import IsoForestScorer
    from models.lstm_scorer import LSTMScorer

    new_iso  = IsoForestScorer.load_latest()
    new_lstm = LSTMScorer.load_latest()

    # atomic swap — consumer thread sees new models on next event
    scorer_ref["iso_forest"] = new_iso
    scorer_ref["lstm"]       = new_lstm

    version = new_iso.version   # or read from a VERSION file
    return {"status": "reloaded", "model_version": version}

@app.get("/health")
def health():
    return {"status": "ok",
            "model_version": scorer_ref.get("iso_forest", {}).version}
```

```python
# In DAG 2 — deploy_models task, after writing model files:

import requests

requests.post(
    "http://live-scoring-service:8000/reload-model",
    timeout=10,
)
```

```python
# docker-compose.yml or startup script — run both together:
# uvicorn live_scoring.api:app --host 0.0.0.0 --port 8000 &
# python detect_power_kafka.py
```

### Notes
- The `scorer_ref` dict gives a thread-safe way to swap model references
  because Python dict assignment is atomic at the GIL level.
- For true zero-downtime swap, load the new model fully before replacing
  the reference — never partially replace.
- Log the model version in every `PowerScoringResult` so you can audit
  which model produced which alert.

---

## Summary table

| # | Issue | Severity | File(s) to change |
|---|-------|----------|-------------------|
| 1 | Delta ignores time gaps | HIGH | `event_preprocessor.py` |
| 2 | Buffer lost on restart | MEDIUM | `rolling_window_buffer.py`, `detect_power_kafka.py` |
| 3 | PendingAlert no TTL | MEDIUM | `PendingAlert` dataclass, state machine |
| 4 | IF-only → CLEARED hides spikes | MEDIUM | alert state machine, `AlertState` enum |
| 5 | Compound alert O(n) scan | LOW | `detect_power_kafka.py` |
| 6 | No model hot-reload | MEDIUM | new `live_scoring/api.py`, DAG 2 deploy task |

---

## Suggested work order for Claude Code

1. **Issue 1 first** — it affects data quality going into both models. Fix before tuning thresholds.
2. **Issue 4 next** — a design decision that changes alert semantics. Decide on `CONFIRMED_IF_ONLY` before adding more alert consumers.
3. **Issue 3** — straightforward dataclass change, low risk.
4. **Issue 2** — add checkpointing once the above are stable.
5. **Issue 6** — add hot-reload when you are ready to wire up the full three-service deployment.
6. **Issue 5** — last, it is a performance optimisation not a correctness fix.
