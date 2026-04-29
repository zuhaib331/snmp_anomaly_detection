"""Feature engineering for the baseline UPS health model.

Loads the power SNMP dataset, derives aggregated metrics, applies scaling,
and builds fixed-length sequences per device for LSTM Autoencoder training.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, ProjectPaths


LOG1P_COLS: tuple[str, ...] = ("runtime_remaining_min", "output_power_w")


def load_power_dataset(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    df = pd.read_csv(paths.power_dataset_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device rate-of-change features.

    Must be called AFTER apply_log1p_skewed so runtime_remaining_min is already
    compressed. signed_log1p squashes extreme delta spikes during anomalies
    (e.g. battery_charge_pct dropping 97 pts in one step, output_load_pct jumping to 120).
    temperature_delta and output_load_delta give the LSTM early warning on gradual anomalies.
    """
    df = df.copy()
    runtime_diff  = df.groupby("device_id")["runtime_remaining_min"].diff().fillna(0.0)
    charge_diff   = df.groupby("device_id")["battery_charge_pct"].diff().fillna(0.0)
    temp_diff     = df.groupby("device_id")["battery_temperature_c"].diff().fillna(0.0)
    load_diff     = df.groupby("device_id")["output_load_pct"].diff().fillna(0.0)
    # signed_log1p: preserves direction, compresses magnitude
    df["runtime_delta"]        = np.sign(runtime_diff)  * np.log1p(np.abs(runtime_diff))
    df["battery_charge_delta"] = np.sign(charge_diff)   * np.log1p(np.abs(charge_diff))
    df["temperature_delta"]    = np.sign(temp_diff)     * np.log1p(np.abs(temp_diff))
    df["output_load_delta"]    = np.sign(load_diff)     * np.log1p(np.abs(load_diff))
    return df


def aggregate_phase_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute input_voltage_v as the mean of per-phase columns when present."""
    phase_cols = ["input_voltage_l1", "input_voltage_l2", "input_voltage_l3"]
    if all(c in df.columns for c in phase_cols):
        df = df.copy()
        df["input_voltage_v"] = df[phase_cols].mean(axis=1)
    return df


def apply_log1p_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to right-skewed columns to reduce scale variance."""
    df = df.copy()
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def filter_normal_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["anomaly"] == 0].copy()


def scale_features(
    df: pd.DataFrame,
    paths: ProjectPaths,
    scaler_path: Path | None = None,
) -> tuple[pd.DataFrame, RobustScaler]:
    scaler_path = scaler_path or paths.power_outputs_dir / "baseline_scaler.pkl"
    feature_cols = [c for c in BASELINE_UPS_FEATURES if c in df.columns]
    scaler = RobustScaler()
    out = df.copy()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    joblib.dump(scaler, scaler_path)
    return out, scaler


def create_sequences(values: np.ndarray, seq_len: int) -> np.ndarray:
    seqs = [values[i : i + seq_len] for i in range(len(values) - seq_len)]
    return np.array(seqs) if seqs else np.empty((0, seq_len, values.shape[1]))


def build_baseline_sequences(
    df: pd.DataFrame,
    seq_len: int = 10,
) -> np.ndarray:
    feature_cols = [c for c in BASELINE_UPS_FEATURES if c in df.columns]
    chunks = []
    for device_id in df["device_id"].unique():
        device_df = df[df["device_id"] == device_id]
        seqs = create_sequences(device_df[feature_cols].values, seq_len)
        if len(seqs):
            chunks.append(seqs)
    if not chunks:
        return np.empty((0, seq_len, len(feature_cols)))
    return np.concatenate(chunks, axis=0)


def run_power_feature_engineering(
    seq_len: int = 10,
    paths: ProjectPaths | None = None,
) -> dict[str, np.ndarray]:
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    df = load_power_dataset(paths)
    df = aggregate_phase_metrics(df)
    df = apply_log1p_skewed(df)
    df = add_delta_features(df)
    train_df = filter_normal_rows(df)
    scaled_df, _ = scale_features(train_df, paths)
    sequences = build_baseline_sequences(scaled_df, seq_len)

    x_train_path = paths.power_outputs_dir / "X_train.npy"
    np.save(x_train_path, sequences)

    meta = {
        "seq_len": seq_len,
        "feature_columns": [c for c in BASELINE_UPS_FEATURES if c in df.columns],
        "num_sequences": len(sequences),
        "shape": list(sequences.shape),
    }
    with open(paths.power_outputs_dir / "preprocess_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"X_train": sequences}


def main() -> None:
    result = run_power_feature_engineering()
    print(f"Baseline sequences shape: {result['X_train'].shape}")
