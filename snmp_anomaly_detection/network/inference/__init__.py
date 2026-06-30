"""Inference and anomaly detection entrypoints."""

from snmp_anomaly_detection.network.inference.csv_replay import detect_csv_replay
from snmp_anomaly_detection.network.inference.core import (
    InferenceArtifacts,
    WindowScore,
    load_inference_artifacts,
    scale_window,
    score_window,
    score_windows,
)
from snmp_anomaly_detection.network.inference.event_processor import (
    EventProcessor,
    PreparedWindow,
    ProcessedWindow,
)
from snmp_anomaly_detection.network.inference.events import NormalizedEvent
from snmp_anomaly_detection.network.inference.live_microbatch import (
    LiveMicroBatchProcessor,
    MicroBatchConfig,
)
from snmp_anomaly_detection.network.inference.window_manager import DeviceWindowManager, ReadyWindow

__all__ = [
    "DeviceWindowManager",
    "EventProcessor",
    "InferenceArtifacts",
    "LiveMicroBatchProcessor",
    "MicroBatchConfig",
    "NormalizedEvent",
    "PreparedWindow",
    "ProcessedWindow",
    "ReadyWindow",
    "WindowScore",
    "detect_csv_replay",
    "load_inference_artifacts",
    "scale_window",
    "score_window",
    "score_windows",
]
