from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedEvent:
    timestamp: Any
    device_id: str
    interface: str | None
    cpu: float
    memory: float
    in_octets: float
    out_octets: float
    errors: float
    in_ucast_pkts: float | None = None
    out_ucast_pkts: float | None = None
    in_discards: float | None = None
    out_discards: float | None = None
    interface_speed_mbps: float | None = None
    interface_admin_status: int | None = None
    interface_oper_status: int | None = None
    counter_reset: int = 0
    anomaly: int = 0

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_id": self.device_id,
            "interface": self.interface,
            "cpu": self.cpu,
            "memory": self.memory,
            "in_octets": self.in_octets,
            "out_octets": self.out_octets,
            "errors": self.errors,
            "in_ucast_pkts": self.in_ucast_pkts,
            "out_ucast_pkts": self.out_ucast_pkts,
            "in_discards": self.in_discards,
            "out_discards": self.out_discards,
            "interface_speed_mbps": self.interface_speed_mbps,
            "interface_admin_status": self.interface_admin_status,
            "interface_oper_status": self.interface_oper_status,
            "counter_reset": self.counter_reset,
            "anomaly": self.anomaly,
        }
