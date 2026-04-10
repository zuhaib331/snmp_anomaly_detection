from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    EvaluationConfig,
    FeatureEngineeringConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.evaluation.time_split import (
    TimeSplitDefinition,
    apply_time_split_labels,
    build_time_split_definition,
)
from snmp_anomaly_detection.preprocessing.derived_features import (
    derive_rate_features_dataframe,
)
from snmp_anomaly_detection.preprocessing.pipeline import (
    PreprocessingPipeline,
    available_scaler_names,
)


@dataclass(frozen=True)
class FeatureEngineeringArtifacts:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass(frozen=True)
class PreprocessingMetadata:
    split_definition: TimeSplitDefinition
    scaler_name: str
    log1p_features: tuple[str, ...]
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    train_normal_row_count: int
    validation_normal_row_count: int
    test_normal_row_count: int
    x_train_shape: tuple[int, ...]
    x_test_shape: tuple[int, ...]


def load_dataset(input_file: str | None = None, paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    if input_file is not None:
        dataset_path = Path(input_file)
        if not dataset_path.is_absolute():
            dataset_path = paths.repo_root / input_file
    else:
        legacy_candidates = (
            paths.dataset_file,
            paths.repo_root / "synthetic_snmp_dataset.csv",
            paths.repo_root / "featureEngineering" / "synthetic_snmp_dataset.csv",
        )
        dataset_path = next((path for path in legacy_candidates if path.exists()), paths.dataset_file)

    dataframe = pd.read_csv(dataset_path)
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    sort_columns = ["device_id"]
    if "interface" in dataframe.columns:
        sort_columns.append("interface")
    sort_columns.append("timestamp")
    return dataframe.sort_values(by=sort_columns).reset_index(drop=True)


def filter_normal_rows(
    dataframe: pd.DataFrame, config: FeatureEngineeringConfig
) -> pd.DataFrame:
    filtered = dataframe[dataframe["anomaly"] == config.normal_label].copy()
    if "elapsed_seconds" in filtered.columns:
        filtered = filtered[filtered["elapsed_seconds"] > 0]
    if "reset_detected" in filtered.columns:
        filtered = filtered[filtered["reset_detected"] == 0]
    return filtered.copy()


def fit_scaler(
    dataframe: pd.DataFrame,
    config: FeatureEngineeringConfig,
    paths: ProjectPaths,
) -> PreprocessingPipeline:
    scaler = PreprocessingPipeline(
        feature_columns=tuple(config.feature_columns),
        scaler_name=config.scaler_name,
        log1p_features=tuple(config.log1p_features),
    )
    scaler.fit(dataframe)

    if config.save_scaler:
        joblib.dump(scaler, paths.scaler_file)

    return scaler


def transform_features(
    dataframe: pd.DataFrame,
    scaler: PreprocessingPipeline,
    config: FeatureEngineeringConfig,
) -> pd.DataFrame:
    _ = config
    return scaler.transform(dataframe)


def create_sequences(values: np.ndarray, sequence_length: int) -> np.ndarray:
    sequences = []
    for index in range(len(values) - sequence_length):
        sequences.append(values[index : index + sequence_length])
    return np.array(sequences)


def build_device_sequences(
    dataframe: pd.DataFrame, config: FeatureEngineeringConfig
) -> np.ndarray:
    chunks = []
    feature_columns = list(config.feature_columns)
    group_columns = ["device_id"]
    if "interface" in dataframe.columns:
        group_columns.append("interface")

    for _, device_frame in dataframe.groupby(group_columns, dropna=False):
        device_values = device_frame[feature_columns].values
        device_sequences = create_sequences(device_values, config.sequence_length)

        if len(device_sequences) > 0:
            chunks.append(device_sequences)

    if not chunks:
        feature_count = len(feature_columns)
        return np.empty((0, config.sequence_length, feature_count))

    return np.concatenate(chunks, axis=0)


def save_artifacts(artifacts: FeatureEngineeringArtifacts, paths: ProjectPaths) -> None:
    np.save(paths.x_train_file, artifacts.x_train)
    np.save(paths.x_test_file, artifacts.x_test)
    np.save(paths.y_train_file, artifacts.y_train)
    np.save(paths.y_test_file, artifacts.y_test)


def build_preprocessing_metadata(
    labeled_frame: pd.DataFrame,
    normal_labeled_frame: pd.DataFrame,
    split_definition: TimeSplitDefinition,
    config: FeatureEngineeringConfig,
    artifacts: FeatureEngineeringArtifacts,
) -> PreprocessingMetadata:
    split_counts = labeled_frame["split"].value_counts().to_dict()
    normal_split_counts = normal_labeled_frame["split"].value_counts().to_dict()
    return PreprocessingMetadata(
        split_definition=split_definition,
        scaler_name=config.scaler_name,
        log1p_features=tuple(config.log1p_features),
        train_row_count=int(split_counts.get("train", 0)),
        validation_row_count=int(split_counts.get("validation", 0)),
        test_row_count=int(split_counts.get("test", 0)),
        train_normal_row_count=int(normal_split_counts.get("train", 0)),
        validation_normal_row_count=int(normal_split_counts.get("validation", 0)),
        test_normal_row_count=int(normal_split_counts.get("test", 0)),
        x_train_shape=tuple(int(size) for size in artifacts.x_train.shape),
        x_test_shape=tuple(int(size) for size in artifacts.x_test.shape),
    )


def save_preprocessing_metadata(
    metadata: PreprocessingMetadata,
    paths: ProjectPaths,
) -> None:
    with open(paths.preprocessing_metadata_file, "w", encoding="utf-8") as file:
        json.dump(asdict(metadata), file, indent=2)


def run_feature_engineering(
    input_file: str | None = None,
    config: FeatureEngineeringConfig | None = None,
    paths: ProjectPaths | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> FeatureEngineeringArtifacts:
    config = config or FeatureEngineeringConfig()
    paths = paths or ProjectPaths()
    evaluation_config = evaluation_config or EvaluationConfig()
    paths.ensure_directories()

    dataframe = load_dataset(input_file=input_file, paths=paths)
    dataframe = derive_rate_features_dataframe(dataframe)
    split_definition = build_time_split_definition(dataframe, evaluation_config)
    labeled_frame = apply_time_split_labels(dataframe, split_definition)
    normal_labeled_frame = filter_normal_rows(labeled_frame, config)

    train_frame = normal_labeled_frame[normal_labeled_frame["split"] == "train"].copy()
    test_frame = normal_labeled_frame[normal_labeled_frame["split"] == "test"].copy()

    if train_frame.empty:
        raise ValueError("Time-based train split produced no normal rows for preprocessing.")

    scaler = fit_scaler(train_frame, config, paths)
    scaled_train_frame = transform_features(train_frame, scaler, config)
    scaled_test_frame = transform_features(test_frame, scaler, config)

    x_train = build_device_sequences(scaled_train_frame, config)
    x_test = build_device_sequences(scaled_test_frame, config)
    artifacts = FeatureEngineeringArtifacts(
        x_train=x_train,
        x_test=x_test,
        y_train=x_train.copy(),
        y_test=x_test.copy(),
    )
    save_artifacts(artifacts, paths)
    preprocessing_metadata = build_preprocessing_metadata(
        labeled_frame=labeled_frame,
        normal_labeled_frame=normal_labeled_frame,
        split_definition=split_definition,
        config=config,
        artifacts=artifacts,
    )
    save_preprocessing_metadata(preprocessing_metadata, paths)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNMP feature engineering.")
    parser.add_argument(
        "--artifact-dir-name",
        help="Named artifact directory under snmp_anomaly_detection/artifacts/.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path to use for saved arrays and scaler.",
    )
    parser.add_argument(
        "--scaler-name",
        choices=available_scaler_names(),
        help="Scaler to fit on training-period normal rows.",
    )
    parser.add_argument(
        "--log1p-features",
        nargs="*",
        default=None,
        help="Optional feature names to transform with log1p before scaling.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="Sliding window length used for sequence generation.",
    )
    args = parser.parse_args()

    default_config = FeatureEngineeringConfig()
    config = FeatureEngineeringConfig(
        sequence_length=args.sequence_length or default_config.sequence_length,
        scaler_name=args.scaler_name or default_config.scaler_name,
        log1p_features=tuple(args.log1p_features)
        if args.log1p_features is not None
        else default_config.log1p_features,
    )
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or ProjectPaths().artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    artifacts = run_feature_engineering(paths=paths, config=config)
    print("Feature engineering completed successfully.")
    print(f"X_train shape: {artifacts.x_train.shape}")
    print(f"X_test shape: {artifacts.x_test.shape}")
    print(f"Artifact directory: {paths.artifact_dir}")
    print(f"Scaler: {paths.scaler_file}")
    print(f"Preprocessing metadata: {paths.preprocessing_metadata_file}")


if __name__ == "__main__":
    main()
