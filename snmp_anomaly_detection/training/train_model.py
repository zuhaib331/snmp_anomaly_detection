from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime

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


def save_training_artifacts(
    model,
    paths: ProjectPaths,
    config: TrainingConfig,
    feature_config: FeatureEngineeringConfig,
    input_size: int,
    threshold: float,
    train_loss_history: list[float],
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
        "training_config": asdict(config),
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
    threshold = float(
        train_errors.mean() + config.threshold_std_multiplier * train_errors.std()
    )

    save_training_artifacts(
        model=model,
        paths=paths,
        config=config,
        feature_config=feature_config,
        input_size=input_size,
        threshold=threshold,
        train_loss_history=train_loss_history,
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
    args = parser.parse_args()

    default_paths = ProjectPaths()
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or default_paths.artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    train_model(paths=paths)


if __name__ == "__main__":
    main()
