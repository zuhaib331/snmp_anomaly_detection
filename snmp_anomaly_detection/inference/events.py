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
