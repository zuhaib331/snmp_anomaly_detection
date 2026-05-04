"""Train LSTM Autoencoder on raw per-phase L1/L2/L3 features (phase-level model)."""
from __future__ import annotations

import json

import numpy as np

from snmp_anomaly_detection.config import PHASE_LEVEL_FEATURES, ProjectPaths, PowerTrainingConfig
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.evaluation.time_split import time_based_split
from snmp_anomaly_detection.preprocessing.power_features import (
    load_power_dataset,
    aggregate_phase_metrics,
    normalize_absolute_features,
    apply_log1p_skewed,
    add_delta_features,
    filter_normal_rows,
)
from snmp_anomaly_detection.preprocessing.phase_features import (
    enrich_imbalance_features,
    scale_phase_features,
    build_phase_sequences,
    _IMBALANCE_COLS,
    _IMBALANCE_CLIP_MAX,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install torch and re-run.")


def train_phase_model(
    seq_len: int | None = None,
    config: PowerTrainingConfig | None = None,
    paths: ProjectPaths | None = None,
) -> dict:
    _require_torch()
    config = config or PowerTrainingConfig()
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()
    seq_len = seq_len if seq_len is not None else config.seq_len

    df = load_power_dataset(paths)
    df = aggregate_phase_metrics(df)
    df = normalize_absolute_features(df)
    df = apply_log1p_skewed(df)
    df = add_delta_features(df)
    df = enrich_imbalance_features(df)
    normal_df = filter_normal_rows(df)
    train_df, _, _ = time_based_split(normal_df)
    scaled_train, _ = scale_phase_features(train_df, paths)

    x_train = build_phase_sequences(scaled_train, seq_len)
    input_size = x_train.shape[2]

    train_tensor = torch.tensor(x_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * 0.1)
    loss_fn = torch.nn.MSELoss()
    loss_fn_no_reduce = torch.nn.MSELoss(reduction="none")
    history = []

    for epoch in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        epoch_loss = total / max(count, 1)
        history.append(epoch_loss)
        print(f"[phase] epoch {epoch+1}/{config.epochs} loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        recon = model(train_tensor)
        per_window = loss_fn_no_reduce(recon, train_tensor).mean(dim=(1, 2)).cpu().numpy()
    # Percentile-based threshold is robust to heavy-tailed error distributions
    # (e.g. Liebert 3-phase UPS devices whose normal variance makes mean+k*std too tight).
    threshold = float(np.percentile(per_window, 99.9))

    # Per-feature reconstruction stats on normal training data (mean + std for z-score attribution)
    feature_names = [c for c in PHASE_LEVEL_FEATURES if c in scaled_train.columns]
    per_seq_feat = loss_fn_no_reduce(recon, train_tensor).mean(dim=1).cpu().numpy()  # (n_seq, n_feat)
    per_feat_mean = per_seq_feat.mean(axis=0)
    per_feat_std  = per_seq_feat.std(axis=0)
    normal_feature_errors = {f: round(float(e), 8) for f, e in zip(feature_names, per_feat_mean)}
    normal_feature_error_stds = {f: round(float(e), 8) for f, e in zip(feature_names, per_feat_std)}

    # Per-category thresholds — mirrors baseline training logic
    per_category_thresholds: dict[str, float] = {}
    if "device_category" in scaled_train.columns:
        loss_fn_cat = torch.nn.MSELoss(reduction="none")
        model.eval()
        for cat, cat_df in scaled_train.groupby("device_category"):
            # Apply same imbalance clipping used during normal phase scaling
            cat_df2 = cat_df.copy()
            for col in _IMBALANCE_COLS:
                if col in cat_df2.columns:
                    cat_df2[col] = cat_df2[col].clip(0.0, _IMBALANCE_CLIP_MAX) / _IMBALANCE_CLIP_MAX
            x_cat = build_phase_sequences(cat_df2, seq_len)
            if len(x_cat) > 0:
                cat_tensor = torch.tensor(x_cat, dtype=torch.float32)
                with torch.no_grad():
                    recon_cat = model(cat_tensor)
                    cat_errors = loss_fn_cat(recon_cat, cat_tensor).mean(dim=(1, 2)).cpu().numpy()
                per_category_thresholds[str(cat)] = round(
                    float(np.percentile(cat_errors, 99.9)), 8
                )

    model_path = paths.power_phase_outputs_dir / "phase_model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "model_type": "lstm_autoencoder_phase",
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "latent_size": config.latent_size,
        "seq_len": seq_len,
        "epochs": config.epochs,
        "threshold": threshold,
        "train_loss_history": history,
        "feature_names": feature_names,
        "normal_feature_errors": normal_feature_errors,
        "normal_feature_error_stds": normal_feature_error_stds,
        "per_category_thresholds": per_category_thresholds,
    }
    with open(paths.power_phase_outputs_dir / "phase_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Phase model saved: {model_path}")
    print(f"Phase anomaly threshold: {threshold:.6f}")
    return meta


def main() -> None:
    train_phase_model()
