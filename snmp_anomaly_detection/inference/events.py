from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, PHASE_LEVEL_FEATURES


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


@dataclass
class PowerEvent:
    """One SNMP event from any transport (Kafka, MQTT, CSV replay, …)."""
    timestamp: datetime
    device_id: str
    device_category: str
    vendor: str
    feature_values: dict[str, float]   # canonical feature name → value
    phase_count: int = 1               # 1 (single-phase) or 3 (three-phase)
    # Device registration constants for B2 normalization.
    # Set from device registration / SNMP discovery at onboarding time.
    # If rated_capacity_w == 0, normalization falls back to raw values (graceful degradation).
    rated_capacity_w: float = 0.0      # nameplate power rating in Watts
    nominal_voltage_v: float = 120.0   # nominal input voltage (120 or 230)
    rated_battery_v: float = 0.0       # battery string voltage (0 for non-UPS)
    session_reset: bool = False        # clears stale per-device delta state on new producer run
    true_label: int = 0                # ground-truth flag from test producer (0 in production)
    anomaly_type: str = "none"         # ground-truth type from test producer ("none" in production)

    def get_baseline_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in BASELINE_UPS_FEATURES]

    def get_phase_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in PHASE_LEVEL_FEATURES]
