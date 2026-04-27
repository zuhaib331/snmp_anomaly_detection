"""Train the BatteryRULModel LSTM regression for Remaining Useful Life prediction."""
from __future__ import annotations

import json

import numpy as np

from snmp_anomaly_detection.config import ProjectPaths, PowerTrainingConfig
from snmp_anomaly_detection.models.battery_rul import BatteryRULModel, torch
from snmp_anomaly_detection.evaluation.time_split import time_based_split
from snmp_anomaly_detection.preprocessing.battery_features import run_battery_feature_engineering


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

    # Build sequences and labels
    result = run_battery_feature_engineering(seq_len=seq_len, paths=paths)
    sequences = result["sequences"]
    labels = result["labels"]

    if len(sequences) == 0:
        raise RuntimeError("No RUL sequences generated. Run generate-power-data first.")

    # Time-based split on flattened index (sequences are already ordered chronologically)
    n = len(sequences)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    x_train, y_train = sequences[:train_end], labels[:train_end]
    x_val, y_val = sequences[train_end:val_end], labels[train_end:val_end]
    x_test, y_test = sequences[val_end:], labels[val_end:]

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
    train_tensor_y = torch.tensor(y_train, dtype=torch.float32)
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

    # Validation MAE
    model.eval()
    with torch.no_grad():
        val_preds = model(torch.tensor(x_val, dtype=torch.float32)).cpu().numpy()
        val_mae = float(np.abs(val_preds - y_val).mean())

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
