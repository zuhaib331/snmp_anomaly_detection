"""Real-time power anomaly detection stream processor.

Extends the master branch's EventProcessor + DeviceWindowManager pattern to
support power events with device-category-based model routing:
  - UPS devices → dual model (baseline + phase)
  - PDU / network / env devices → baseline model only

Compound alert: when a UPS anomaly and a PDU anomaly occur within
PowerInferenceConfig.compound_alert_window_minutes of each other, a
COMPOUND_ALERT is emitted.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    PHASE_LEVEL_FEATURES,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.preprocessing.phase_features import (
    _IMBALANCE_COLS,
    _IMBALANCE_CLIP_MAX,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch


@dataclass
class PowerEvent:
    timestamp: datetime
    device_id: str
    device_category: str
    vendor: str
    feature_values: dict[str, float]   # canonical feature name → value

    def get_baseline_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in BASELINE_UPS_FEATURES]

    def get_phase_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in PHASE_LEVEL_FEATURES]


@dataclass
class PowerScoringResult:
    device_id: str
    device_category: str
    window_end: datetime
    baseline_anomaly: int
    phase_anomaly: int
    combined_flag: int
    compound_alert: bool
    compound_alert_peer: str = ""


class PowerDeviceWindowManager:
    """Per-device rolling buffer identical in design to the master DeviceWindowManager."""

    def __init__(self, seq_len: int) -> None:
        self._seq_len = seq_len
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))

    def push(self, device_id: str, vector: list[float]) -> np.ndarray | None:
        buf = self._buffers[device_id]
        buf.append(vector)
        if len(buf) == self._seq_len:
            return np.array(buf)
        return None


class PowerStreamProcessor:
    def __init__(
        self,
        inference_config: PowerInferenceConfig | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for live power detection.")
        self._config = inference_config or PowerInferenceConfig()
        self._paths = paths or ProjectPaths()
        self._load_models()
        self._baseline_windows = PowerDeviceWindowManager(self._baseline_meta["seq_len"])
        self._phase_windows = PowerDeviceWindowManager(self._phase_meta["seq_len"])
        # Recent anomaly timestamps for compound alert correlation
        self._recent_ups_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)
        self._recent_pdu_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)

    def _load_models(self) -> None:
        p = self._paths

        with open(p.power_outputs_dir / "baseline_metadata.json") as f:
            self._baseline_meta = json.load(f)
        self._baseline_model = LSTMAutoencoder(
            input_size=self._baseline_meta["input_size"],
            hidden_size=self._baseline_meta["hidden_size"],
            latent_size=self._baseline_meta["latent_size"],
        )
        self._baseline_model.load_state_dict(
            torch.load(p.power_outputs_dir / "baseline_model.pt", map_location="cpu")
        )
        self._baseline_model.eval()
        self._baseline_scaler = joblib.load(p.power_outputs_dir / "baseline_scaler.pkl")
        self._baseline_threshold: float = self._baseline_meta["threshold"]

        with open(p.power_phase_outputs_dir / "phase_metadata.json") as f:
            self._phase_meta = json.load(f)
        self._phase_model = LSTMAutoencoder(
            input_size=self._phase_meta["input_size"],
            hidden_size=self._phase_meta["hidden_size"],
            latent_size=self._phase_meta["latent_size"],
        )
        self._phase_model.load_state_dict(
            torch.load(p.power_phase_outputs_dir / "phase_model.pt", map_location="cpu")
        )
        self._phase_model.eval()
        self._phase_scaler = joblib.load(p.power_phase_outputs_dir / "phase_scaler.pkl")
        self._phase_threshold: float = self._phase_meta["threshold"]

    def _score(self, model, window: np.ndarray, threshold: float) -> tuple[float, int]:
        loss_fn = torch.nn.MSELoss(reduction="none")
        tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            recon = model(tensor)
            error = float(loss_fn(recon, tensor).mean().item())
        return error, int(error > threshold)

    def _check_compound_alert(
        self, category: str, device_id: str, ts: datetime
    ) -> tuple[bool, str]:
        window = timedelta(minutes=self._config.compound_alert_window_minutes)
        if category == "ups":
            for (peer_id, peer_ts) in self._recent_pdu_anomalies:
                if abs(ts - peer_ts) <= window:
                    return True, peer_id
        elif category == "pdu":
            for (peer_id, peer_ts) in self._recent_ups_anomalies:
                if abs(ts - peer_ts) <= window:
                    return True, peer_id
        return False, ""

    def process_event(self, event: PowerEvent) -> PowerScoringResult | None:
        # Scale baseline vector
        b_raw = np.array([event.get_baseline_vector()])
        b_scaled = self._baseline_scaler.transform(b_raw)[0].tolist()
        b_window = self._baseline_windows.push(event.device_id, b_scaled)

        p_window = None
        if event.device_category == "ups":
            # Apply same imbalance clipping used during phase model training
            for col in _IMBALANCE_COLS:
                if col in event.feature_values:
                    event.feature_values[col] = (
                        min(max(event.feature_values[col], 0.0), _IMBALANCE_CLIP_MAX)
                        / _IMBALANCE_CLIP_MAX
                    )
            p_raw = np.array([event.get_phase_vector()])
            p_scaled = self._phase_scaler.transform(p_raw)[0].tolist()
            p_window = self._phase_windows.push(event.device_id, p_scaled)

        # Only score when both windows are ready (for UPS) or just baseline (other)
        if b_window is None:
            return None
        if event.device_category == "ups" and p_window is None:
            return None

        _, b_flag = self._score(self._baseline_model, b_window, self._baseline_threshold)
        p_flag = 0
        if event.device_category == "ups" and p_window is not None:
            _, p_flag = self._score(self._phase_model, p_window, self._phase_threshold)

        policy = self._config.alert_policy
        combined = int(b_flag or p_flag) if policy == "or" else int(b_flag and p_flag)

        # Compound alert tracking
        compound, peer = False, ""
        if combined:
            compound, peer = self._check_compound_alert(
                event.device_category, event.device_id, event.timestamp
            )
            if event.device_category == "ups":
                self._recent_ups_anomalies.append((event.device_id, event.timestamp))
            elif event.device_category == "pdu":
                self._recent_pdu_anomalies.append((event.device_id, event.timestamp))

        return PowerScoringResult(
            device_id=event.device_id,
            device_category=event.device_category,
            window_end=event.timestamp,
            baseline_anomaly=b_flag,
            phase_anomaly=p_flag,
            combined_flag=combined,
            compound_alert=compound,
            compound_alert_peer=peer,
        )
