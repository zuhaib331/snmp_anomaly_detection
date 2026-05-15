"""Train LSTM Autoencoder on baseline aggregated UPS health metrics."""
from __future__ import annotations

import json

import numpy as np

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, ProjectPaths, PowerTrainingConfig
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.evaluation.time_split import time_based_split
from snmp_anomaly_detection.preprocessing.power_features import (
    run_power_feature_engineering,
    load_power_dataset,
    normalize_absolute_features,
    apply_log1p_skewed,
    add_delta_features,
    filter_normal_rows,
    scale_features,
    build_baseline_sequences,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install torch and re-run.")


def _build_dataloader(sequences: np.ndarray, config: PowerTrainingConfig):
    tensor = torch.tensor(sequences, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor, tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def _compute_errors(model, tensor_data) -> np.ndarray:
    loss_fn = torch.nn.MSELoss(reduction="none")
    model.eval()
    with torch.no_grad():
        recon = model(tensor_data)
        return loss_fn(recon, tensor_data).mean(dim=(1, 2)).cpu().numpy()


def train_baseline_power(
    seq_len: int | None = None,
    config: PowerTrainingConfig | None = None,
    paths: ProjectPaths | None = None,
) -> dict:
    _require_torch()
    config = config or PowerTrainingConfig()
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()
    seq_len = seq_len if seq_len is not None else config.seq_len

    # Build time-split sequences using normal-only training partition
    df = load_power_dataset(paths)
    df = normalize_absolute_features(df)
    df = apply_log1p_skewed(df)
    df = add_delta_features(df)
    normal_df = filter_normal_rows(df)
    train_df, val_df, _ = time_based_split(normal_df)
    scaled_train, train_scaler = scale_features(train_df, paths)
    val_feature_cols = [c for c in BASELINE_UPS_FEATURES if c in val_df.columns]
    scaled_val = val_df.copy()
    scaled_val[val_feature_cols] = train_scaler.transform(scaled_val[val_feature_cols])

    x_train = build_baseline_sequences(scaled_train, seq_len)
    x_val = build_baseline_sequences(scaled_val, seq_len)
    input_size = x_train.shape[2]

    dataloader = _build_dataloader(x_train, config)
    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()
    history = []

    for epoch in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        epoch_loss = total / max(count, 1)
        history.append(epoch_loss)
        print(f"[baseline] epoch {epoch+1}/{config.epochs} loss={epoch_loss:.6f}")

    # Threshold from training reconstruction errors
    loss_fn_no_reduce = torch.nn.MSELoss(reduction="none")
    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_errors = _compute_errors(model, train_tensor)
    # Percentile-based threshold is robust to heavy-tailed error distributions
    # (e.g. Liebert 3-phase UPS devices whose normal variance makes mean+k*std too tight).
    threshold = float(np.percentile(train_errors, 99.9))

    # Per-feature reconstruction stats on normal training data (mean + std for z-score attribution)
    feature_names = [c for c in BASELINE_UPS_FEATURES if c in scaled_train.columns]
    model.eval()
    with torch.no_grad():
        recon = model(train_tensor)
        per_seq_feat = loss_fn_no_reduce(recon, train_tensor).mean(dim=1).cpu().numpy()  # (n_seq, n_feat)
    per_feat_mean = per_seq_feat.mean(axis=0)
    per_feat_std  = per_seq_feat.std(axis=0)
    normal_feature_errors = {f: round(float(e), 8) for f, e in zip(feature_names, per_feat_mean)}
    normal_feature_error_stds = {f: round(float(e), 8) for f, e in zip(feature_names, per_feat_std)}

    # Per-category thresholds — each device category has its own reconstruction error range.
    # Categories with lower normal errors (env, network) get a tighter threshold so soft
    # anomalies (overload on env device) are not swamped by the global UPS-dominated baseline.
    per_category_thresholds: dict[str, float] = {}
    if "device_category" in scaled_train.columns:
        for cat, cat_df in scaled_train.groupby("device_category"):
            x_cat = build_baseline_sequences(cat_df, seq_len)
            if len(x_cat) > 0:
                cat_tensor = torch.tensor(x_cat, dtype=torch.float32)
                cat_errors = _compute_errors(model, cat_tensor)
                per_category_thresholds[str(cat)] = round(
                    float(np.percentile(cat_errors, 99.9)), 8
                )

    # Val errors for reporting
    val_tensor = torch.tensor(x_val, dtype=torch.float32)
    val_errors = _compute_errors(model, val_tensor)

    model_path = paths.power_outputs_dir / "baseline_model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "model_type": "lstm_autoencoder_baseline",
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "latent_size": config.latent_size,
        "seq_len": seq_len,
        "epochs": config.epochs,
        "threshold": threshold,
        "train_loss_history": history,
        "val_mean_error": float(val_errors.mean()),
        "feature_names": feature_names,
        "normal_feature_errors": normal_feature_errors,
        "normal_feature_error_stds": normal_feature_error_stds,
        "per_category_thresholds": per_category_thresholds,
    }
    with open(paths.power_outputs_dir / "baseline_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Baseline model saved: {model_path}")
    print(f"Anomaly threshold: {threshold:.6f}")
    return meta


def main() -> None:
    train_baseline_power()
