from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.inference.core import (
    build_feature_config_from_artifact_metadata,
    scale_window,
)
from snmp_anomaly_detection.inference.csv_replay import normalize_csv_row
from snmp_anomaly_detection.inference.window_manager import DeviceWindowManager, ReadyWindow
from snmp_anomaly_detection.models.isolation_forest import (
    feature_deviation_profile,
    flatten_sequences,
)
from snmp_anomaly_detection.preprocessing.derived_features import OnlineRateFeatureBuilder
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


@dataclass(frozen=True)
class IsolationForestArtifacts:
    model: Any
    scaler: Any
    metadata: dict[str, Any]
    source_paths: ProjectPaths
    feature_config: FeatureEngineeringConfig
    threshold: float
    flattened_mean: np.ndarray
    flattened_std: np.ndarray


@dataclass(frozen=True)
class IsolationForestScore:
    anomaly_score: float
    threshold: float
    score_margin: float
    predicted_anomaly: int
    top_error_feature: str
    top_error_timestep_offset: int
    detection_basis: str
    feature_deviations: dict[str, float]


@dataclass(frozen=True)
class IsolationForestProcessedWindow:
    ready_window: ReadyWindow
    score: IsolationForestScore

    def to_result_record(self, config: FeatureEngineeringConfig) -> dict[str, object]:
        result: dict[str, object] = {
            "source": "csv-replay-iforest",
            "device_id": self.ready_window.device_id,
            "interface": self.ready_window.interface,
            "stream_id": self.ready_window.stream_id,
            "device_window_index": self.ready_window.device_window_index,
            "window_start": self.ready_window.window_start,
            "window_end": self.ready_window.window_end,
            "source_anomaly_label": self.ready_window.source_anomaly_label,
            "sequence_length": config.sequence_length,
            "iforest_anomaly_score": self.score.anomaly_score,
            "iforest_threshold": self.score.threshold,
            "iforest_score_margin": self.score.score_margin,
            "predicted_anomaly": self.score.predicted_anomaly,
            "top_error_feature": self.score.top_error_feature,
            "top_error_timestep_offset": self.score.top_error_timestep_offset,
            "detection_basis": self.score.detection_basis,
        }
        for feature_name in config.feature_columns:
            result[f"feature_deviation_{feature_name}"] = self.score.feature_deviations[
                feature_name
            ]
        return result


def _load_iforest_metadata(paths: ProjectPaths) -> dict[str, Any]:
    with open(paths.iforest_metadata_file, encoding="utf-8") as file:
        return json.load(file)


def _source_paths_from_metadata(metadata: dict[str, Any]) -> ProjectPaths:
    source_artifact_dir = metadata.get("source_artifact_dir")
    source_artifact_dir_name = metadata.get("source_artifact_dir_name")
    return ProjectPaths(
        artifact_dir_name=source_artifact_dir_name or ProjectPaths().artifact_dir_name,
        artifact_dir_override=source_artifact_dir,
    )


def load_iforest_artifacts(paths: ProjectPaths) -> IsolationForestArtifacts:
    metadata = _load_iforest_metadata(paths)
    source_paths = _source_paths_from_metadata(metadata)
    feature_config = build_feature_config_from_artifact_metadata(
        FeatureEngineeringConfig(),
        {"feature_config": metadata.get("feature_config", {})},
    )
    scaler_path = source_paths.scaler_file if source_paths.scaler_file.exists() else source_paths.legacy_scaler_file

    return IsolationForestArtifacts(
        model=joblib.load(paths.iforest_model_file),
        scaler=joblib.load(scaler_path),
        metadata=metadata,
        source_paths=source_paths,
        feature_config=feature_config,
        threshold=float(metadata["threshold"]),
        flattened_mean=np.array(metadata["flattened_feature_mean"], dtype=float),
        flattened_std=np.array(metadata["flattened_feature_std"], dtype=float),
    )


def _prepare_windows(
    dataframe: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> list[ReadyWindow]:
    feature_builder = OnlineRateFeatureBuilder()
    window_manager = DeviceWindowManager(config)
    ready_windows: list[ReadyWindow] = []
    for _, row in dataframe.iterrows():
        normalized_event = normalize_csv_row(row)
        derived = feature_builder.transform(normalized_event)
        if derived is None:
            continue
        ready_window = window_manager.add_event(derived.record)
        if ready_window is not None:
            ready_windows.append(ready_window)
    return ready_windows


def _build_detection_basis(
    anomaly_score: float,
    threshold: float,
    top_error_feature: str,
    top_error_timestep_offset: int,
) -> str:
    margin = anomaly_score - threshold
    if anomaly_score > threshold:
        return (
            f"Isolation Forest anomaly_score {anomaly_score:.6f} exceeded threshold "
            f"{threshold:.6f} by {margin:.6f}; highest scaled deviation from "
            f"{top_error_feature} at window step {top_error_timestep_offset}"
        )
    return (
        f"Isolation Forest anomaly_score {anomaly_score:.6f} stayed below threshold "
        f"{threshold:.6f}; highest scaled deviation from {top_error_feature} "
        f"at window step {top_error_timestep_offset}"
    )


def score_iforest_windows(
    ready_windows: list[ReadyWindow],
    artifacts: IsolationForestArtifacts,
) -> list[IsolationForestProcessedWindow]:
    if not ready_windows:
        return []

    config = artifacts.feature_config
    feature_columns = list(config.feature_columns)
    scaled_windows = np.array(
        [
            scale_window(
                ready_window.values,
                scaler=artifacts.scaler,
                config=config,
            )
            for ready_window in ready_windows
        ],
        dtype=float,
    )
    flat_windows = flatten_sequences(scaled_windows)
    anomaly_scores = -artifacts.model.score_samples(flat_windows)
    predictions = artifacts.model.predict(flat_windows)

    processed_windows: list[IsolationForestProcessedWindow] = []
    for index, ready_window in enumerate(ready_windows):
        feature_deviations, timestep_deviations = feature_deviation_profile(
            scaled_windows[index],
            artifacts.flattened_mean,
            artifacts.flattened_std,
        )
        top_feature_index = int(feature_deviations.argmax())
        top_timestep_index = int(timestep_deviations.argmax())
        top_error_feature = feature_columns[top_feature_index]
        anomaly_score = float(anomaly_scores[index])
        score = IsolationForestScore(
            anomaly_score=anomaly_score,
            threshold=artifacts.threshold,
            score_margin=float(anomaly_score - artifacts.threshold),
            predicted_anomaly=int(predictions[index] == -1),
            top_error_feature=top_error_feature,
            top_error_timestep_offset=top_timestep_index,
            detection_basis=_build_detection_basis(
                anomaly_score=anomaly_score,
                threshold=artifacts.threshold,
                top_error_feature=top_error_feature,
                top_error_timestep_offset=top_timestep_index,
            ),
            feature_deviations={
                feature_name: float(feature_deviations[feature_index])
                for feature_index, feature_name in enumerate(feature_columns)
            },
        )
        processed_windows.append(
            IsolationForestProcessedWindow(
                ready_window=ready_window,
                score=score,
            )
        )
    return processed_windows


def _build_anomaly_windows(
    processed_windows: list[IsolationForestProcessedWindow],
    config: FeatureEngineeringConfig,
) -> list[dict[str, object]]:
    anomaly_windows: list[dict[str, object]] = []
    for processed_window in processed_windows:
        if processed_window.score.predicted_anomaly != 1:
            continue
        result_record = processed_window.to_result_record(config)
        anomaly_windows.append(
            {
                **result_record,
                "window_records": processed_window.ready_window.records,
                "window_summary": {
                    "record_count": len(processed_window.ready_window.records),
                    "first_timestamp": result_record["window_start"],
                    "last_timestamp": result_record["window_end"],
                    "device_id": result_record["device_id"],
                    "interface": result_record["interface"],
                },
            }
        )
    return anomaly_windows


def detect_isolation_forest_csv(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
    inference_config: InferenceConfig | None = None,
) -> pd.DataFrame:
    paths = paths or ProjectPaths(artifact_dir_name="iforest_f3_v1")
    inference_config = inference_config or InferenceConfig()
    paths.ensure_directories()

    artifacts = load_iforest_artifacts(paths)
    dataframe = load_dataset(input_file=input_file, paths=artifacts.source_paths)
    ready_windows = _prepare_windows(dataframe, artifacts.feature_config)
    processed_windows = score_iforest_windows(ready_windows, artifacts)
    result_records = [
        processed_window.to_result_record(artifacts.feature_config)
        for processed_window in processed_windows
    ]
    results = pd.DataFrame(result_records)
    anomaly_windows = _build_anomaly_windows(processed_windows, artifacts.feature_config)

    if inference_config.save_results:
        results.to_csv(paths.iforest_results_file, index=False)
        with open(paths.iforest_anomaly_windows_file, "w", encoding="utf-8") as file:
            json.dump(anomaly_windows, file, indent=2, default=str)

    predicted_count = int(results["predicted_anomaly"].sum()) if not results.empty else 0
    print(f"Isolation Forest results saved to: {paths.iforest_results_file}")
    print(f"Isolation Forest anomaly windows saved to: {paths.iforest_anomaly_windows_file}")
    print(f"Total windows evaluated: {len(results)}")
    print(f"Predicted anomalies: {predicted_count}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Isolation Forest anomaly detection on CSV input.")
    parser.add_argument(
        "--input-file",
        help="Optional CSV file to score. Defaults to the configured dataset file.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        default="iforest_f3_v1",
        help="Isolation Forest artifact directory under snmp_anomaly_detection/artifacts/.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit Isolation Forest artifact directory path.",
    )
    args = parser.parse_args()

    detect_isolation_forest_csv(
        input_file=args.input_file,
        paths=ProjectPaths(
            artifact_dir_name=args.artifact_dir_name,
            artifact_dir_override=args.artifact_dir,
        ),
    )


if __name__ == "__main__":
    main()
