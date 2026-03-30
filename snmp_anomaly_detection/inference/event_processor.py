from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import FeatureEngineeringConfig
from snmp_anomaly_detection.inference.core import (
    InferenceArtifacts,
    WindowScore,
    score_window,
    score_windows,
)
from snmp_anomaly_detection.inference.events import NormalizedEvent
from snmp_anomaly_detection.inference.window_manager import DeviceWindowManager, ReadyWindow


@dataclass(frozen=True)
class ProcessedWindow:
    ready_window: ReadyWindow
    score: WindowScore

    def to_result_record(self, config: FeatureEngineeringConfig) -> dict[str, object]:
        result = {
            "device_id": self.ready_window.device_id,
            "device_window_index": self.ready_window.device_window_index,
            "window_start": self.ready_window.window_start,
            "window_end": self.ready_window.window_end,
            "source_anomaly_label": self.ready_window.source_anomaly_label,
            "sequence_length": config.sequence_length,
            "reconstruction_error": self.score.reconstruction_error,
            "threshold": self.score.threshold,
            "error_margin": self.score.error_margin,
            "predicted_anomaly": self.score.predicted_anomaly,
            "top_error_feature": self.score.top_error_feature,
            "top_error_timestep_offset": self.score.top_error_timestep_offset,
            "detection_basis": self.score.detection_basis,
        }

        for feature_name in config.feature_columns:
            result[f"feature_error_{feature_name}"] = self.score.feature_errors[feature_name]

        return result


@dataclass(frozen=True)
class PreparedWindow:
    ready_window: ReadyWindow

    def score(
        self,
        artifacts: InferenceArtifacts,
        config: FeatureEngineeringConfig,
    ) -> ProcessedWindow:
        score = score_window(
            self.ready_window.values,
            artifacts=artifacts,
            config=config,
        )
        return ProcessedWindow(ready_window=self.ready_window, score=score)


class EventProcessor:
    def __init__(
        self,
        artifacts: InferenceArtifacts,
        config: FeatureEngineeringConfig | None = None,
        window_manager: DeviceWindowManager | None = None,
    ):
        self.config = config or FeatureEngineeringConfig()
        self.artifacts = artifacts
        self.window_manager = window_manager or DeviceWindowManager(self.config)
        self.feature_columns = list(self.config.feature_columns)

    def _scale_event(self, event: NormalizedEvent) -> NormalizedEvent:
        feature_frame = pd.DataFrame(
            [
                {
                    feature_name: getattr(event, feature_name)
                    for feature_name in self.feature_columns
                }
            ]
        )
        scaled_values = self.artifacts.scaler.transform(feature_frame)[0]
        scaled_map = {
            feature_name: float(scaled_values[index])
            for index, feature_name in enumerate(self.feature_columns)
        }
        return NormalizedEvent(
            timestamp=event.timestamp,
            device_id=event.device_id,
            cpu=scaled_map["cpu"],
            memory=scaled_map["memory"],
            in_octets=scaled_map["in_octets"],
            out_octets=scaled_map["out_octets"],
            errors=scaled_map["errors"],
            anomaly=event.anomaly,
        )

    def prepare_event(self, event: NormalizedEvent) -> PreparedWindow | None:
        scaled_event = self._scale_event(event)
        ready_window = self.window_manager.add_event(scaled_event.to_record())
        if ready_window is None:
            return None

        return PreparedWindow(
            ready_window=ready_window,
        )

    def process_event(self, event: NormalizedEvent) -> ProcessedWindow | None:
        prepared_window = self.prepare_event(event)
        if prepared_window is None:
            return None
        return prepared_window.score(self.artifacts, self.config)

    def process_prepared_windows(
        self,
        prepared_windows: list[PreparedWindow],
    ) -> list[ProcessedWindow]:
        if not prepared_windows:
            return []

        scaled_windows = np.array(
            [prepared_window.ready_window.values for prepared_window in prepared_windows],
            dtype=float,
        )
        scores = score_windows(
            scaled_windows,
            artifacts=self.artifacts,
            config=self.config,
        )
        return [
            ProcessedWindow(
                ready_window=prepared_window.ready_window,
                score=score,
            )
            for prepared_window, score in zip(prepared_windows, scores)
        ]
