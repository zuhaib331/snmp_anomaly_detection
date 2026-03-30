from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from snmp_anomaly_detection.config import FeatureEngineeringConfig


@dataclass(frozen=True)
class ReadyWindow:
    device_id: str
    device_window_index: int
    window_start: str
    window_end: str
    source_anomaly_label: int
    values: np.ndarray
    records: list[dict[str, Any]]


class DeviceWindowManager:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.feature_columns = list(config.feature_columns)
        self._buffers: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.config.sequence_length)
        )
        self._window_counts: dict[str, int] = defaultdict(int)

    def add_event(self, event: Mapping[str, Any]) -> ReadyWindow | None:
        device_id = str(event["device_id"])
        buffer = self._buffers[device_id]
        normalized_event = dict(event)
        buffer.append(normalized_event)

        if len(buffer) < self.config.sequence_length:
            return None

        device_window_index = self._window_counts[device_id]
        self._window_counts[device_id] += 1

        window_records = list(buffer)
        values = np.array(
            [
                [float(record[feature_name]) for feature_name in self.feature_columns]
                for record in window_records
            ],
            dtype=float,
        )

        return ReadyWindow(
            device_id=device_id,
            device_window_index=device_window_index,
            window_start=str(window_records[0]["timestamp"]),
            window_end=str(window_records[-1]["timestamp"]),
            source_anomaly_label=int(window_records[-1].get("anomaly", 0)),
            values=values,
            records=window_records,
        )
