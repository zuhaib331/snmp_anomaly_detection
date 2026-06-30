from __future__ import annotations

import json

import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.network.inference.core import load_inference_artifacts
from snmp_anomaly_detection.network.inference.event_processor import EventProcessor
from snmp_anomaly_detection.network.inference.events import NormalizedEvent
from snmp_anomaly_detection.network.preprocessing.feature_engineering import load_dataset


def normalize_csv_row(row: pd.Series) -> NormalizedEvent:
    return NormalizedEvent(
        timestamp=row["timestamp"],
        device_id=str(row["device_id"]),
        cpu=float(row["cpu"]),
        memory=float(row["memory"]),
        in_octets=float(row["in_octets"]),
        out_octets=float(row["out_octets"]),
        errors=float(row["errors"]),
        anomaly=int(row.get("anomaly", 0)),
    )


def replay_csv_events(dataframe: pd.DataFrame) -> list[NormalizedEvent]:
    return [normalize_csv_row(row) for _, row in dataframe.iterrows()]


def _empty_results_frame(config: FeatureEngineeringConfig) -> pd.DataFrame:
    results = pd.DataFrame()
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


def detect_csv_replay(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
    config: FeatureEngineeringConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    config = config or FeatureEngineeringConfig()
    inference_config = inference_config or InferenceConfig()

    dataframe = load_dataset(input_file=input_file, paths=paths)
    artifacts = load_inference_artifacts(paths=paths, config=config)
    processor = EventProcessor(artifacts=artifacts, config=config)

    prepared_windows = []
    for event in replay_csv_events(dataframe):
        prepared_window = processor.prepare_event(event)
        if prepared_window is None:
            continue
        prepared_windows.append(prepared_window)

    processed_windows = processor.process_prepared_windows(prepared_windows)
    result_records: list[dict[str, object]] = []
    for processed_window in processed_windows:
        result_records.append(processed_window.to_result_record(config))

    results = (
        pd.DataFrame(result_records)
        if result_records
        else _empty_results_frame(config)
    )
    anomaly_windows = _build_window_export(dataframe, results, config)

    if inference_config.save_results:
        results.to_csv(paths.anomaly_results_file, index=False)
        with open(paths.anomaly_windows_file, "w", encoding="utf-8") as file:
            json.dump(anomaly_windows, file, indent=2, default=str)

    predicted_count = int(results["predicted_anomaly"].sum()) if not results.empty else 0
    print(f"Detection results saved to: {paths.anomaly_results_file}")
    print(f"Anomaly windows saved to: {paths.anomaly_windows_file}")
    print(f"Total windows evaluated: {len(results)}")
    print(f"Predicted anomalies: {predicted_count}")
    return results


def main() -> None:
    detect_csv_replay()


if __name__ == "__main__":
    main()
