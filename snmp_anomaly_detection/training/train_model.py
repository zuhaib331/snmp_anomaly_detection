from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np

from snmp_anomaly_detection.config import FeatureEngineeringConfig, ProjectPaths, TrainingConfig
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for training. Install torch, then run `python3 main.py train`."
        )


def load_training_arrays(paths: ProjectPaths) -> tuple[np.ndarray, np.ndarray]:
    x_train_path = paths.x_train_file if paths.x_train_file.exists() else paths.legacy_x_train_file
    y_train_path = paths.y_train_file if paths.y_train_file.exists() else paths.legacy_y_train_file
    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    return x_train, y_train


def load_preprocessing_metadata(paths: ProjectPaths) -> dict[str, Any]:
    if not paths.preprocessing_metadata_file.exists():
        return {}
    with open(paths.preprocessing_metadata_file, "r", encoding="utf-8") as file:
        return json.load(file)


def build_feature_config_from_preprocessing_metadata(
    feature_config: FeatureEngineeringConfig,
    preprocessing_metadata: dict[str, Any],
) -> FeatureEngineeringConfig:
    scaler_name = preprocessing_metadata.get("scaler_name") or feature_config.scaler_name
    log1p_features = (
        preprocessing_metadata.get("log1p_features")
        if preprocessing_metadata.get("log1p_features") is not None
        else feature_config.log1p_features
    )
    return FeatureEngineeringConfig(
        sequence_length=feature_config.sequence_length,
        feature_columns=tuple(
            preprocessing_metadata.get("feature_columns", feature_config.feature_columns)
        ),
        train_split=feature_config.train_split,
        normal_label=feature_config.normal_label,
        save_scaler=feature_config.save_scaler,
        scaler_name=scaler_name,
        log1p_features=tuple(log1p_features),
    )


def build_dataloader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
):
    features = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(features, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )


def compute_reconstruction_errors(model, tensor_data, loss_fn):
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor_data)
        per_timestep_loss = loss_fn(reconstructed, tensor_data)
        return per_timestep_loss.mean(dim=(1, 2)).cpu().numpy()


def compute_anomaly_threshold(
    train_errors: np.ndarray,
    config: TrainingConfig,
) -> float:
    if config.threshold_mode == "stddev":
        return float(
            train_errors.mean() + config.threshold_std_multiplier * train_errors.std()
        )
    if config.threshold_mode == "percentile":
        return float(np.percentile(train_errors, config.threshold_percentile))
    raise ValueError(f"Unsupported threshold_mode: {config.threshold_mode}")


def save_training_artifacts(
    model,
    paths: ProjectPaths,
    config: TrainingConfig,
    feature_config: FeatureEngineeringConfig,
    preprocessing_metadata: dict[str, Any],
    input_size: int,
    threshold: float,
    train_loss_history: list[float],
    num_training_sequences: int,
) -> None:
    torch.save(model.state_dict(), paths.model_file)
    metadata = {
        "artifact_dir": str(paths.artifact_dir),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "latent_size": config.latent_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "threshold": threshold,
        "feature_config": asdict(feature_config),
        "preprocessing_config": {
            "scaler_name": preprocessing_metadata.get("scaler_name"),
            "log1p_features": preprocessing_metadata.get("log1p_features"),
        },
        "threshold_config": {
            "mode": config.threshold_mode,
            "std_multiplier": config.threshold_std_multiplier,
            "percentile": config.threshold_percentile,
        },
        "training_config": asdict(config),
        "preprocessing_metadata_file": str(paths.preprocessing_metadata_file),
        "preprocessing_metadata_exists": paths.preprocessing_metadata_file.exists(),
        "num_training_sequences": num_training_sequences,
        "train_loss_history": train_loss_history,
    }
    with open(paths.model_metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def train_model(
    paths: ProjectPaths | None = None,
    config: TrainingConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
) -> dict[str, float]:
    _require_torch()
    paths = paths or ProjectPaths()
    config = config or TrainingConfig()
    feature_config = feature_config or FeatureEngineeringConfig()
    paths.ensure_directories()

    preprocessing_metadata = load_preprocessing_metadata(paths)
    feature_config = build_feature_config_from_preprocessing_metadata(
        feature_config,
        preprocessing_metadata,
    )

    x_train, y_train = load_training_arrays(paths)
    input_size = x_train.shape[2]

    dataloader = build_dataloader(x_train, y_train, config)
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()
    train_loss_history = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch_inputs)
            loss = loss_fn(reconstructed, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = batch_inputs.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

        epoch_loss = running_loss / max(sample_count, 1)
        train_loss_history.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss:.6f}")

    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    detailed_loss_fn = torch.nn.MSELoss(reduction="none")
    train_errors = compute_reconstruction_errors(model, train_tensor, detailed_loss_fn)
    threshold = compute_anomaly_threshold(train_errors, config)

    save_training_artifacts(
        model=model,
        paths=paths,
        config=config,
        feature_config=feature_config,
        preprocessing_metadata=preprocessing_metadata,
        input_size=input_size,
        threshold=threshold,
        train_loss_history=train_loss_history,
        num_training_sequences=int(len(x_train)),
    )

    summary = {
        "final_train_loss": train_loss_history[-1],
        "threshold": threshold,
        "num_training_sequences": float(len(x_train)),
    }
    print(f"Model saved to: {paths.model_file}")
    print(f"Metadata saved to: {paths.model_metadata_file}")
    print(f"Artifact directory: {paths.artifact_dir}")
    print(f"Anomaly threshold: {threshold:.6f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SNMP anomaly model.")
    parser.add_argument(
        "--artifact-dir-name",
        help="Named artifact directory under snmp_anomaly_detection/artifacts/.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path to use for training artifacts.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("stddev", "percentile"),
        help="Threshold rule to derive from training reconstruction errors.",
    )
    parser.add_argument(
        "--threshold-std-multiplier",
        type=float,
        help="K value for the threshold rule mean + K * std.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        help="Percentile to use when threshold mode is percentile.",
    )
    args = parser.parse_args()

    default_paths = ProjectPaths()
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or default_paths.artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    default_config = TrainingConfig()
    config = TrainingConfig(
        batch_size=default_config.batch_size,
        epochs=default_config.epochs,
        learning_rate=default_config.learning_rate,
        hidden_size=default_config.hidden_size,
        latent_size=default_config.latent_size,
        threshold_mode=args.threshold_mode or default_config.threshold_mode,
        threshold_std_multiplier=(
            args.threshold_std_multiplier
            if args.threshold_std_multiplier is not None
            else default_config.threshold_std_multiplier
        ),
        threshold_percentile=(
            args.threshold_percentile
            if args.threshold_percentile is not None
            else default_config.threshold_percentile
        ),
    )
    train_model(paths=paths, config=config)


if __name__ == "__main__":
    main()
