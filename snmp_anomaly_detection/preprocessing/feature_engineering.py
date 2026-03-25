from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from snmp_anomaly_detection.config import FeatureEngineeringConfig, ProjectPaths


@dataclass(frozen=True)
class FeatureEngineeringArtifacts:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


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
    return dataframe.sort_values(by=["device_id", "timestamp"]).reset_index(drop=True)


def filter_normal_rows(
    dataframe: pd.DataFrame, config: FeatureEngineeringConfig
) -> pd.DataFrame:
    return dataframe[dataframe["anomaly"] == config.normal_label].copy()


def scale_features(
    dataframe: pd.DataFrame,
    config: FeatureEngineeringConfig,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    feature_columns = list(config.feature_columns)
    scaled = dataframe.copy()
    scaled[feature_columns] = scaler.fit_transform(scaled[feature_columns])

    if config.save_scaler:
        joblib.dump(scaler, paths.scaler_file)

    return scaled, scaler


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

    for device_id in dataframe["device_id"].unique():
        device_frame = dataframe[dataframe["device_id"] == device_id]
        device_values = device_frame[feature_columns].values
        device_sequences = create_sequences(device_values, config.sequence_length)

        if len(device_sequences) > 0:
            chunks.append(device_sequences)

    if not chunks:
        feature_count = len(feature_columns)
        return np.empty((0, config.sequence_length, feature_count))

    return np.concatenate(chunks, axis=0)


def split_sequences(
    sequences: np.ndarray, config: FeatureEngineeringConfig
) -> FeatureEngineeringArtifacts:
    split_index = int(config.train_split * len(sequences))
    x_train = sequences[:split_index]
    x_test = sequences[split_index:]
    return FeatureEngineeringArtifacts(
        x_train=x_train,
        x_test=x_test,
        y_train=x_train.copy(),
        y_test=x_test.copy(),
    )


def save_artifacts(artifacts: FeatureEngineeringArtifacts, paths: ProjectPaths) -> None:
    np.save(paths.x_train_file, artifacts.x_train)
    np.save(paths.x_test_file, artifacts.x_test)
    np.save(paths.y_train_file, artifacts.y_train)
    np.save(paths.y_test_file, artifacts.y_test)


def run_feature_engineering(
    input_file: str | None = None,
    config: FeatureEngineeringConfig | None = None,
    paths: ProjectPaths | None = None,
) -> FeatureEngineeringArtifacts:
    config = config or FeatureEngineeringConfig()
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    dataframe = load_dataset(input_file=input_file, paths=paths)
    train_frame = filter_normal_rows(dataframe, config)
    scaled_frame, _ = scale_features(train_frame, config, paths)
    sequences = build_device_sequences(scaled_frame, config)
    artifacts = split_sequences(sequences, config)
    save_artifacts(artifacts, paths)
    return artifacts


def main() -> None:
    artifacts = run_feature_engineering()
    print("Feature engineering completed successfully.")
    print(f"X_train shape: {artifacts.x_train.shape}")
    print(f"X_test shape: {artifacts.x_test.shape}")
    print(f"Scaler: {ProjectPaths().scaler_file}")


if __name__ == "__main__":
    main()
