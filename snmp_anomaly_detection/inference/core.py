from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import FeatureEngineeringConfig, ProjectPaths
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch


@dataclass(frozen=True)
class InferenceArtifacts:
    scaler: Any
    model: LSTMAutoencoder
    metadata: dict[str, Any]
    threshold: float


@dataclass(frozen=True)
class WindowScore:
    reconstruction_error: float
    threshold: float
    error_margin: float
    predicted_anomaly: int
    top_error_feature: str
    top_error_timestep_offset: int
    detection_basis: str
    feature_errors: dict[str, float]


def require_torch_for_inference() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for detection. Install torch, then run `python3 main.py detect`."
        )


def load_model_metadata(paths: ProjectPaths) -> dict[str, Any]:
    with open(paths.model_metadata_file, "r", encoding="utf-8") as file:
        return json.load(file)


def load_inference_artifacts(
    paths: ProjectPaths | None = None,
    config: FeatureEngineeringConfig | None = None,
) -> InferenceArtifacts:
    require_torch_for_inference()
    paths = paths or ProjectPaths()
    config = config or FeatureEngineeringConfig()
    metadata = load_model_metadata(paths)

    model = LSTMAutoencoder(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        latent_size=metadata["latent_size"],
    )
    state_dict = torch.load(paths.model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return InferenceArtifacts(
        scaler=joblib.load(paths.scaler_file),
        model=model,
        metadata=metadata,
        threshold=float(metadata["threshold"]),
    )


def scale_feature_frame(
    dataframe: pd.DataFrame,
    scaler: Any,
    config: FeatureEngineeringConfig,
) -> pd.DataFrame:
    feature_columns = list(config.feature_columns)
    scaled = dataframe.copy()
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])
    return scaled


def scale_window(
    window: np.ndarray,
    scaler: Any,
    config: FeatureEngineeringConfig,
) -> np.ndarray:
    feature_columns = list(config.feature_columns)
    frame = pd.DataFrame(window, columns=feature_columns)
    scaled_frame = scale_feature_frame(frame, scaler, config)
    return scaled_frame[feature_columns].to_numpy(dtype=float)


def _run_model(model, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(sequences) == 0:
        return np.array([]), np.array([]), np.array([])

    inputs = torch.tensor(sequences, dtype=torch.float32)
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


def build_detection_basis(
    reconstruction_error: float,
    threshold: float,
    top_error_feature: str,
    top_error_timestep_offset: int,
) -> str:
    error_margin = reconstruction_error - threshold
    if reconstruction_error > threshold:
        return (
            f"error {reconstruction_error:.6f} exceeded threshold "
            f"{threshold:.6f} by {error_margin:.6f}; "
            f"highest contribution from {top_error_feature} at "
            f"window step {top_error_timestep_offset}"
        )

    return (
        f"error {reconstruction_error:.6f} stayed below threshold "
        f"{threshold:.6f}; highest contribution from "
        f"{top_error_feature} at window step {top_error_timestep_offset}"
    )


def score_windows(
    sequences: np.ndarray,
    artifacts: InferenceArtifacts,
    config: FeatureEngineeringConfig,
) -> list[WindowScore]:
    if len(sequences) == 0:
        return []

    feature_columns = list(config.feature_columns)
    reconstruction_errors, feature_errors, timestep_errors = _run_model(
        artifacts.model,
        sequences,
    )

    scores: list[WindowScore] = []
    for index, reconstruction_error in enumerate(reconstruction_errors):
        top_feature_index = int(feature_errors[index].argmax())
        top_timestep_index = int(timestep_errors[index].argmax())
        top_error_feature = feature_columns[top_feature_index]
        threshold = artifacts.threshold
        feature_error_map = {
            feature_name: float(feature_errors[index][feature_index])
            for feature_index, feature_name in enumerate(feature_columns)
        }

        scores.append(
            WindowScore(
                reconstruction_error=float(reconstruction_error),
                threshold=threshold,
                error_margin=float(reconstruction_error - threshold),
                predicted_anomaly=int(reconstruction_error > threshold),
                top_error_feature=top_error_feature,
                top_error_timestep_offset=top_timestep_index,
                detection_basis=build_detection_basis(
                    reconstruction_error=float(reconstruction_error),
                    threshold=threshold,
                    top_error_feature=top_error_feature,
                    top_error_timestep_offset=top_timestep_index,
                ),
                feature_errors=feature_error_map,
            )
        )

    return scores


def score_window(
    window: np.ndarray,
    artifacts: InferenceArtifacts,
    config: FeatureEngineeringConfig,
) -> WindowScore:
    scores = score_windows(np.expand_dims(window, axis=0), artifacts, config)
    if not scores:
        raise ValueError("Expected one window to score, but received an empty input.")
    return scores[0]
