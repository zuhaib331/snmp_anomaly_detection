from __future__ import annotations

import json

import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.inference.core import (
    load_inference_artifacts,
    scale_feature_frame,
    score_windows,
)
from snmp_anomaly_detection.inference.window_manager import DeviceWindowManager
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


def build_detection_sequences(
    dataframe: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sequences = []
    metadata = []
    manager = DeviceWindowManager(config)

    for _, row in dataframe.iterrows():
        ready_window = manager.add_event(row.to_dict())
        if ready_window is None:
            continue

        sequences.append(ready_window.values)
        metadata.append(
            {
                "device_id": ready_window.device_id,
                "device_window_index": ready_window.device_window_index,
                "window_start": ready_window.window_start,
                "window_end": ready_window.window_end,
                "source_anomaly_label": ready_window.source_anomaly_label,
            }
        )

    if not sequences:
        return np.empty((0, config.sequence_length, len(config.feature_columns))), metadata

    return np.array(sequences), metadata


def _build_window_export(
    original_dataframe: pd.DataFrame,
    results: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> list[dict[str, object]]:
    anomaly_windows: list[dict[str, object]] = []
    feature_columns = list(config.feature_columns)

    anomalous_rows = results[results["predicted_anomaly"] == 1]
    if anomalous_rows.empty:
        return anomaly_windows

    for device_id in original_dataframe["device_id"].unique():
        device_frame = original_dataframe[
            original_dataframe["device_id"] == device_id
        ].reset_index(drop=True)
        device_predictions = anomalous_rows[anomalous_rows["device_id"] == device_id]

        for _, row in device_predictions.iterrows():
            start_index = int(row["device_window_index"])
            end_index = start_index + config.sequence_length
            window_frame = device_frame.iloc[start_index:end_index].copy()

            anomaly_windows.append(
                {
                    "device_id": device_id,
                    "device_window_index": start_index,
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "predicted_anomaly": int(row["predicted_anomaly"]),
                    "source_anomaly_label": int(row["source_anomaly_label"]),
                    "reconstruction_error": float(row["reconstruction_error"]),
                    "threshold": float(row["threshold"]),
                    "error_margin": float(row["error_margin"]),
                    "top_error_feature": row["top_error_feature"],
                    "top_error_timestep_offset": int(row["top_error_timestep_offset"]),
                    "detection_basis": row["detection_basis"],
                    "window_records": window_frame[
                        ["timestamp", "device_id", *feature_columns, "anomaly"]
                    ].to_dict(orient="records"),
                }
            )

    return anomaly_windows


def detect_anomalies(
    paths: ProjectPaths | None = None,
    config: FeatureEngineeringConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    config = config or FeatureEngineeringConfig()
    inference_config = inference_config or InferenceConfig()

    dataframe = load_dataset(paths=paths)
    artifacts = load_inference_artifacts(paths=paths, config=config)

    scaled = scale_feature_frame(dataframe, artifacts.scaler, config)

    sequences, sequence_metadata = build_detection_sequences(scaled, config)
    scores = score_windows(sequences, artifacts, config)

    results = pd.DataFrame(sequence_metadata)
    results["sequence_length"] = config.sequence_length

    if scores:
        results["reconstruction_error"] = [score.reconstruction_error for score in scores]
        results["threshold"] = [score.threshold for score in scores]
        results["error_margin"] = [score.error_margin for score in scores]
        results["predicted_anomaly"] = [score.predicted_anomaly for score in scores]
        results["top_error_feature"] = [score.top_error_feature for score in scores]
        results["top_error_timestep_offset"] = [
            score.top_error_timestep_offset for score in scores
        ]
        results["detection_basis"] = [score.detection_basis for score in scores]

        for feature_name in config.feature_columns:
            results[f"feature_error_{feature_name}"] = [
                score.feature_errors[feature_name] for score in scores
            ]
    else:
        results["reconstruction_error"] = pd.Series(dtype=float)
        results["threshold"] = pd.Series(dtype=float)
        results["error_margin"] = pd.Series(dtype=float)
        results["predicted_anomaly"] = pd.Series(dtype=int)
        results["top_error_feature"] = pd.Series(dtype=str)
        results["top_error_timestep_offset"] = pd.Series(dtype=int)
        results["detection_basis"] = pd.Series(dtype=str)
        for feature_name in config.feature_columns:
            results[f"feature_error_{feature_name}"] = pd.Series(dtype=float)

    anomaly_windows = _build_window_export(dataframe, results, config)

    if inference_config.save_results:
        results.to_csv(paths.anomaly_results_file, index=False)
        with open(paths.anomaly_windows_file, 'w', encoding='utf-8') as file:
            json.dump(anomaly_windows, file, indent=2, default=str)

    predicted_count = int(results["predicted_anomaly"].sum()) if not results.empty else 0
    print(f"Detection results saved to: {paths.anomaly_results_file}")
    print(f"Anomaly windows saved to: {paths.anomaly_windows_file}")
    print(f"Total windows evaluated: {len(results)}")
    print(f"Predicted anomalies: {predicted_count}")
    return results


def main() -> None:
    detect_anomalies()


if __name__ == "__main__":
    main()
