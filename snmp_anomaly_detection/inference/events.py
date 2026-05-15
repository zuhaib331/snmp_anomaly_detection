from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
