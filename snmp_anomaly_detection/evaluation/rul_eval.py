"""Evaluation and per-device RUL prediction with MC Dropout confidence intervals."""
from __future__ import annotations

import json
from dataclasses import asdict

import joblib
import numpy as np

from snmp_anomaly_detection.config import BATTERY_RUL_FEATURES, ProjectPaths
from snmp_anomaly_detection.models.battery_rul import BatteryRULModel, RULPrediction, torch
from snmp_anomaly_detection.preprocessing.power_features import (
    apply_log1p_skewed,
    load_power_dataset,
)
from snmp_anomaly_detection.preprocessing.battery_features import derive_rul_labels


_MC_SAMPLES: int = 30   # forward passes for MC Dropout confidence intervals


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for RUL evaluation.")


def _load_rul_model(paths: ProjectPaths) -> tuple["BatteryRULModel", dict]:
    with open(paths.battery_rul_outputs_dir / "rul_metadata.json") as f:
        meta = json.load(f)
    model = BatteryRULModel(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(
        torch.load(paths.battery_rul_outputs_dir / "rul_model.pt", map_location="cpu")
    )
    return model, meta


def _mc_predict(model: "BatteryRULModel", x: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Run n_samples stochastic forward passes (dropout active) to get mean and std."""
    model.train()   # keep dropout active during MC inference
    preds = []
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(tensor).cpu().numpy())
    arr = np.stack(preds, axis=0)   # (n_samples, batch)
    return arr.mean(axis=0), arr.std(axis=0)


def evaluate_rul(paths: ProjectPaths | None = None) -> dict:
    _require_torch()
    paths = paths or ProjectPaths()

    model, meta = _load_rul_model(paths)
    scaler = joblib.load(paths.battery_rul_outputs_dir / "rul_scaler.pkl")

    x_test = np.load(paths.battery_rul_outputs_dir / "X_test_rul.npy")
    y_test = np.load(paths.battery_rul_outputs_dir / "y_test_rul.npy")

    if len(x_test) == 0:
        print("No test sequences. Run train-battery-rul first.")
        return {}

    rul_label_max = meta.get("rul_label_max", 1.0)
    raw_means, raw_stds = _mc_predict(model, x_test, _MC_SAMPLES)
    means = raw_means * rul_label_max
    stds = raw_stds * rul_label_max

    mae = float(np.abs(means - y_test).mean())
    rmse = float(np.sqrt(((means - y_test) ** 2).mean()))
    mape = float((np.abs((means - y_test) / np.clip(y_test, 1, None)) * 100).mean())
    # Coverage: fraction where true RUL is within [mean-2std, mean+2std]
    lo = means - 2 * stds
    hi = means + 2 * stds
    coverage = float(((y_test >= lo) & (y_test <= hi)).mean())

    metrics = {
        "mae_days": mae,
        "rmse_days": rmse,
        "mape_pct": mape,
        "ci_coverage_95pct": coverage,
        "num_test_sequences": len(x_test),
    }
    with open(paths.battery_rul_outputs_dir / "rul_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"MAE:      {mae:.2f} days")
    print(f"RMSE:     {rmse:.2f} days")
    print(f"MAPE:     {mape:.2f}%")
    print(f"Coverage: {coverage:.1%}")
    return metrics


def predict_rul_per_device(paths: ProjectPaths | None = None) -> list[RULPrediction]:
    """Generate one RUL prediction per UPS device from the latest data window."""
    _require_torch()
    paths = paths or ProjectPaths()

    model, meta = _load_rul_model(paths)
    scaler = joblib.load(paths.battery_rul_outputs_dir / "rul_scaler.pkl")
    seq_len = meta["seq_len"]
    rul_label_max = meta.get("rul_label_max", 1.0)

    df = load_power_dataset(paths)
    df = apply_log1p_skewed(df)
    df = derive_rul_labels(df)

    feature_cols = [c for c in BATTERY_RUL_FEATURES if c in df.columns]
    ups_df = df[df["device_category"] == "ups"].copy()
    ups_df[feature_cols] = scaler.transform(ups_df[feature_cols])

    predictions: list[RULPrediction] = []
    for device_id in ups_df["device_id"].unique():
        device_df = ups_df[ups_df["device_id"] == device_id]
        if len(device_df) < seq_len:
            continue
        window = device_df[feature_cols].values[-seq_len:]
        mean_pred, std_pred = _mc_predict(model, window[np.newaxis], _MC_SAMPLES)
        rul_days = float(np.clip(mean_pred[0] * rul_label_max, 0, None))
        ci_half = float(2 * std_pred[0] * rul_label_max)

        pred = RULPrediction(
            device_id=device_id,
            timestamp=str(device_df["timestamp"].iloc[-1]),
            predicted_rul_days=round(rul_days, 1),
            confidence_interval_lo=round(max(rul_days - ci_half, 0), 1),
            confidence_interval_hi=round(rul_days + ci_half, 1),
            replacement_advisory="ok",   # __post_init__ will overwrite
        )
        predictions.append(pred)

    out_path = paths.battery_rul_outputs_dir / "rul_predictions.json"
    with open(out_path, "w") as f:
        json.dump([asdict(p) for p in predictions], f, indent=2)

    for p in predictions:
        status = {"urgent": "URGENT", "warn": "WARN", "ok": "ok"}[p.replacement_advisory]
        print(f"{p.device_id}: RUL={p.predicted_rul_days:.0f}d CI=[{p.confidence_interval_lo:.0f}, {p.confidence_interval_hi:.0f}] [{status}]")

    print(f"\nSaved: {out_path}")
    return predictions


def main() -> None:
    paths = ProjectPaths()
    evaluate_rul(paths)
    predict_rul_per_device(paths)
