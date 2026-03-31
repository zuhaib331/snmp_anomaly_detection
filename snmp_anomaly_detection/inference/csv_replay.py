from __future__ import annotations

import argparse
import json

import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.inference.core import load_inference_artifacts
from snmp_anomaly_detection.inference.event_processor import EventProcessor
from snmp_anomaly_detection.inference.events import NormalizedEvent
from snmp_anomaly_detection.inference.output_schema import (
    build_anomaly_window_record,
    build_result_record,
    empty_results_frame,
)
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


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


def _build_window_export(
    original_dataframe: pd.DataFrame,
    processed_windows: list,
    config: FeatureEngineeringConfig,
) -> list[dict[str, object]]:
    anomaly_windows: list[dict[str, object]] = []
    feature_columns = list(config.feature_columns)
    if not processed_windows:
        return anomaly_windows

    for processed_window in processed_windows:
        if processed_window.score.predicted_anomaly != 1:
            continue

        device_id = processed_window.ready_window.device_id
        device_frame = original_dataframe[
            original_dataframe["device_id"] == device_id
        ].reset_index(drop=True)
        start_index = processed_window.ready_window.device_window_index
        end_index = start_index + config.sequence_length
        window_frame = device_frame.iloc[start_index:end_index].copy()
        window_records = window_frame[
            ["timestamp", "device_id", *feature_columns, "anomaly"]
        ].to_dict(orient="records")

        anomaly_windows.append(
            build_anomaly_window_record(
                processed_window=processed_window,
                config=config,
                source="csv-replay",
                window_records=window_records,
            )
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
        result_records.append(
            build_result_record(
                processed_window=processed_window,
                config=config,
                source="csv-replay",
            )
        )

    results = (
        pd.DataFrame(result_records)
        if result_records
        else empty_results_frame(config)
    )
    anomaly_windows = _build_window_export(dataframe, processed_windows, config)

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
    parser = argparse.ArgumentParser(description="Replay a CSV file through anomaly detection.")
    parser.add_argument(
        "--input-file",
        help="Optional CSV file to replay. Defaults to the configured dataset file.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        help="Named artifact directory under snmp_anomaly_detection/artifacts/ to load.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path to load model, scaler, and metadata from.",
    )
    args = parser.parse_args()

    default_paths = ProjectPaths()
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or default_paths.artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    detect_csv_replay(input_file=args.input_file, paths=paths)


if __name__ == "__main__":
    main()
