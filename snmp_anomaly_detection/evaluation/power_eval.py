"""Precision / Recall / F1 evaluation for the baseline and phase-level power models."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, ProjectPaths
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.evaluation.time_split import time_based_split
from snmp_anomaly_detection.preprocessing.power_features import (
    aggregate_phase_metrics,
    apply_log1p_skewed,
    create_sequences,
    load_power_dataset,
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for evaluation.")


def _load_model_and_meta(model_path, meta_path) -> tuple:
    with open(meta_path) as f:
        meta = json.load(f)
    model = LSTMAutoencoder(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        latent_size=meta["latent_size"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, meta


def _score_sequences(model, sequences: np.ndarray) -> np.ndarray:
    loss_fn = torch.nn.MSELoss(reduction="none")
    tensor = torch.tensor(sequences, dtype=torch.float32)
    with torch.no_grad():
        recon = model(tensor)
        return loss_fn(recon, tensor).mean(dim=(1, 2)).cpu().numpy()


def _build_test_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequences, labels) for the test split — includes anomaly rows."""
    all_seqs, all_labels = [], []
    for device_id in df["device_id"].unique():
        device_df = df[df["device_id"] == device_id]
        values = device_df[feature_cols].values
        labels = device_df["anomaly"].values
        for i in range(len(values) - seq_len):
            all_seqs.append(values[i : i + seq_len])
            # Label is 1 if any timestep in the window is anomalous
            all_labels.append(int(labels[i : i + seq_len].max()))
    seqs = np.array(all_seqs) if all_seqs else np.empty((0, seq_len, len(feature_cols)))
    labs = np.array(all_labels, dtype=int)
    return seqs, labs


def _pr_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_baseline(
    paths: ProjectPaths | None = None,
) -> dict:
    _require_torch()
    paths = paths or ProjectPaths()

    model_path = paths.power_outputs_dir / "baseline_model.pt"
    meta_path = paths.power_outputs_dir / "baseline_metadata.json"
    scaler_path = paths.power_outputs_dir / "baseline_scaler.pkl"

    model, meta = _load_model_and_meta(model_path, meta_path)
    threshold = meta["threshold"]
    seq_len = meta["seq_len"]
    scaler = joblib.load(scaler_path)

    df = load_power_dataset(paths)
    df = aggregate_phase_metrics(df)
    df = apply_log1p_skewed(df)
    _, _, test_df = time_based_split(df)

    feature_cols = [c for c in BASELINE_UPS_FEATURES if c in test_df.columns]
    test_df = test_df.copy()
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    sequences, labels = _build_test_sequences(test_df, feature_cols, seq_len)
    if len(sequences) == 0:
        print("No test sequences available.")
        return {}

    errors = _score_sequences(model, sequences)

    # Sweep thresholds on val set for optimal F1, then evaluate on test
    # (For simplicity with synthetic data we use the training threshold directly)
    y_pred = (errors > threshold).astype(int)
    metrics = _pr_f1(labels, y_pred)
    metrics["threshold_used"] = threshold
    metrics["num_test_sequences"] = len(sequences)
    metrics["anomaly_rate_actual"] = float(labels.mean())
    metrics["anomaly_rate_predicted"] = float(y_pred.mean())

    out_path = paths.power_outputs_dir / "p1_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"Results:   {out_path}")
    return metrics


def main() -> None:
    evaluate_baseline()
