from __future__ import annotations

import pandas as pd

from snmp_anomaly_detection.config import FeatureEngineeringConfig
from snmp_anomaly_detection.inference.event_processor import ProcessedWindow


def build_result_record(
    processed_window: ProcessedWindow,
    config: FeatureEngineeringConfig,
    source: str,
) -> dict[str, object]:
    record = processed_window.to_result_record(config)
    record["source"] = source
    return record


def build_anomaly_window_record(
    processed_window: ProcessedWindow,
    config: FeatureEngineeringConfig,
    source: str,
    window_records: list[dict[str, object]],
) -> dict[str, object]:
    result_record = build_result_record(processed_window, config, source)

    return {
        "source": source,
        "device_id": result_record["device_id"],
        "device_window_index": result_record["device_window_index"],
        "window_start": result_record["window_start"],
        "window_end": result_record["window_end"],
        "sequence_length": result_record["sequence_length"],
        "predicted_anomaly": result_record["predicted_anomaly"],
        "source_anomaly_label": result_record["source_anomaly_label"],
        "reconstruction_error": result_record["reconstruction_error"],
        "threshold": result_record["threshold"],
        "error_margin": result_record["error_margin"],
        "top_error_feature": result_record["top_error_feature"],
        "top_error_timestep_offset": result_record["top_error_timestep_offset"],
        "detection_basis": result_record["detection_basis"],
        "feature_error_cpu": result_record["feature_error_cpu"],
        "feature_error_memory": result_record["feature_error_memory"],
        "feature_error_in_octets": result_record["feature_error_in_octets"],
        "feature_error_out_octets": result_record["feature_error_out_octets"],
        "feature_error_errors": result_record["feature_error_errors"],
        "window_records": window_records,
        "window_summary": {
            "record_count": len(window_records),
            "first_timestamp": result_record["window_start"],
            "last_timestamp": result_record["window_end"],
            "device_id": result_record["device_id"],
        },
    }


def empty_results_frame(config: FeatureEngineeringConfig) -> pd.DataFrame:
    results = pd.DataFrame()
    results["source"] = pd.Series(dtype=str)
    results["device_id"] = pd.Series(dtype=str)
    results["device_window_index"] = pd.Series(dtype=int)
    results["window_start"] = pd.Series(dtype=str)
    results["window_end"] = pd.Series(dtype=str)
    results["source_anomaly_label"] = pd.Series(dtype=int)
    results["sequence_length"] = pd.Series(dtype=int)
    results["reconstruction_error"] = pd.Series(dtype=float)
    results["threshold"] = pd.Series(dtype=float)
    results["error_margin"] = pd.Series(dtype=float)
    results["predicted_anomaly"] = pd.Series(dtype=int)
    results["top_error_feature"] = pd.Series(dtype=str)
    results["top_error_timestep_offset"] = pd.Series(dtype=int)
    results["detection_basis"] = pd.Series(dtype=str)
    for feature_name in config.feature_columns:
        results[f"feature_error_{feature_name}"] = pd.Series(dtype=float)
    return results
