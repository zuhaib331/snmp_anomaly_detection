"""Train the BatteryRULModel LSTM regression for Remaining Useful Life prediction."""
from __future__ import annotations

import json

import numpy as np

from snmp_anomaly_detection.config import ProjectPaths, PowerTrainingConfig
from snmp_anomaly_detection.power.models.battery_rul import BatteryRULModel, torch
from snmp_anomaly_detection.power.preprocessing.power_features import load_power_dataset, apply_log1p_skewed
from snmp_anomaly_detection.power.preprocessing.battery_features import (
    derive_rul_labels,
    scale_battery_features,
    build_per_device_rul_split,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install torch and re-run.")


def train_battery_rul(
    seq_len: int = 24,
    config: PowerTrainingConfig | None = None,
    paths: ProjectPaths | None = None,
) -> dict:
    _require_torch()
    config = config or PowerTrainingConfig()
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    # Per-device 70/15/15 split (F10) — each UPS device contributes windows to
    # all three sets so test metrics reflect the full device population.
    df = load_power_dataset(paths)
    df = apply_log1p_skewed(df)
    df = derive_rul_labels(df)
    scaled_df, _ = scale_battery_features(df, paths)

    split = build_per_device_rul_split(scaled_df, seq_len=seq_len)
    x_train, y_train = split["x_train"], split["y_train"]
    x_val,   y_val   = split["x_val"],   split["y_val"]
    x_test,  y_test  = split["x_test"],  split["y_test"]

    if len(x_train) == 0:
        raise RuntimeError("No RUL sequences generated. Run generate-power-data first.")

    # Normalize labels to [0, 1] so MSE loss operates on a unit scale.
    # Scale factor is saved to metadata so inference can denormalize outputs.
    rul_label_max = float(np.concatenate([y_train, y_val, y_test]).max())
    y_train_norm = y_train / rul_label_max
    y_val_norm = y_val / rul_label_max

    input_size = x_train.shape[2]
    model = BatteryRULModel(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=2,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_tensor_x = torch.tensor(x_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_norm, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    history = []
    for epoch in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        epoch_loss = total / max(count, 1)
        history.append(epoch_loss)
        print(f"[RUL] epoch {epoch+1}/{config.epochs} loss={epoch_loss:.4f}")

    # Validation MAE — denormalize predictions back to days before reporting
    model.eval()
    with torch.no_grad():
        val_preds_norm = model(torch.tensor(x_val, dtype=torch.float32)).cpu().numpy()
        val_preds_days = val_preds_norm * rul_label_max
        val_mae = float(np.abs(val_preds_days - y_val).mean())

    model_path = paths.battery_rul_outputs_dir / "rul_model.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "model_type": "battery_rul_lstm",
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "num_layers": 2,
        "dropout": config.dropout,
        "seq_len": seq_len,
        "epochs": config.epochs,
        "rul_label_max": rul_label_max,
        "val_mae_days": val_mae,
        "train_loss_history": history,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
    }
    with open(paths.battery_rul_outputs_dir / "rul_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save test arrays for rul_eval
    np.save(paths.battery_rul_outputs_dir / "X_test_rul.npy", x_test)
    np.save(paths.battery_rul_outputs_dir / "y_test_rul.npy", y_test)

    print(f"RUL model saved: {model_path}")
    print(f"Validation MAE: {val_mae:.2f} days")
    return meta


def main() -> None:
    train_battery_rul()
