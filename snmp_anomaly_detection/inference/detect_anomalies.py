from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for detection. Install torch, then run `python3 main.py detect`."
        )


def load_model_metadata(paths: ProjectPaths) -> dict:
    with open(paths.model_metadata_file, "r", encoding="utf-8") as file:
        return json.load(file)


def build_detection_sequences(
    dataframe: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sequences = []
    metadata = []
    feature_columns = list(config.feature_columns)

    for device_id in dataframe["device_id"].unique():
        device_frame = dataframe[dataframe["device_id"] == device_id].reset_index(drop=True)
        values = device_frame[feature_columns].values

        for index in range(len(values) - config.sequence_length):
            window = values[index : index + config.sequence_length]
            sequences.append(window)
            end_row = device_frame.iloc[index + config.sequence_length - 1]
            metadata.append(
                {
                    "device_id": device_id,
                    "device_window_index": index,
                    "window_start": str(device_frame.iloc[index]["timestamp"]),
                    "window_end": str(end_row["timestamp"]),
                    "source_anomaly_label": int(end_row["anomaly"]),
                }
            )

    if not sequences:
        return np.empty((0, config.sequence_length, len(feature_columns))), metadata

    return np.array(sequences), metadata


def run_model(model, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(sequences) == 0:
        return np.array([]), np.array([]), np.array([])

    inputs = torch.tensor(sequences, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        reconstructed = model(inputs)
        squared_error = (reconstructed - inputs) ** 2
        sequence_errors = torch.mean(squared_error, dim=(1, 2))
        feature_errors = torch.mean(squared_error, dim=1)
        timestep_errors = torch.mean(squared_error, dim=2)

    return (
        sequence_errors.cpu().numpy(),
        feature_errors.cpu().numpy(),
        timestep_errors.cpu().numpy(),
    )


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
    _require_torch()
    paths = paths or ProjectPaths()
    config = config or FeatureEngineeringConfig()
    inference_config = inference_config or InferenceConfig()

    dataframe = load_dataset(paths=paths)
    scaler = joblib.load(paths.scaler_file)
    metadata = load_model_metadata(paths)

    feature_columns = list(config.feature_columns)
    scaled = dataframe.copy()
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])

    sequences, sequence_metadata = build_detection_sequences(scaled, config)
    model = LSTMAutoencoder(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        latent_size=metadata["latent_size"],
    )
    state_dict = torch.load(paths.model_file, map_location="cpu")
    model.load_state_dict(state_dict)

    reconstruction_errors, feature_errors, timestep_errors = run_model(model, sequences)
    threshold = float(metadata["threshold"])

    results = pd.DataFrame(sequence_metadata)
    results["sequence_length"] = config.sequence_length
    results["reconstruction_error"] = reconstruction_errors
    results["threshold"] = threshold
    results["error_margin"] = results["reconstruction_error"] - results["threshold"]
    results["predicted_anomaly"] = (
        results["reconstruction_error"] > results["threshold"]
    ).astype(int)

    if len(results) > 0:
        top_feature_indices = feature_errors.argmax(axis=1)
        top_timestep_indices = timestep_errors.argmax(axis=1)

        results["top_error_feature"] = [
            feature_columns[index] for index in top_feature_indices
        ]
        results["top_error_timestep_offset"] = top_timestep_indices.astype(int)

        for feature_index, feature_name in enumerate(feature_columns):
            results[f"feature_error_{feature_name}"] = feature_errors[:, feature_index]

        results["detection_basis"] = results.apply(
            lambda row: (
                f"error {row['reconstruction_error']:.6f} exceeded threshold "
                f"{row['threshold']:.6f} by {row['error_margin']:.6f}; "
                f"highest contribution from {row['top_error_feature']} at "
                f"window step {int(row['top_error_timestep_offset'])}"
            )
            if row["predicted_anomaly"] == 1
            else (
                f"error {row['reconstruction_error']:.6f} stayed below threshold "
                f"{row['threshold']:.6f}; highest contribution from "
                f"{row['top_error_feature']} at window step "
                f"{int(row['top_error_timestep_offset'])}"
            ),
            axis=1,
        )
    else:
        results["top_error_feature"] = pd.Series(dtype=str)
        results["top_error_timestep_offset"] = pd.Series(dtype=int)
        results["detection_basis"] = pd.Series(dtype=str)

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
