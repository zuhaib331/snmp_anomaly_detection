"""Feature engineering for the baseline UPS health model.

Loads the power SNMP dataset, derives vendor-agnostic metrics, applies scaling,
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
from snmp_anomaly_detection.preprocessing.scalar_transforms import (
    LOG1P_COLS,
    DELTA_PAIRS,
)


def load_power_dataset(paths: ProjectPaths | None = None) -> pd.DataFrame:
    paths = paths or ProjectPaths()
    df = pd.read_csv(paths.power_dataset_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)


def normalize_absolute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace absolute V features with vendor-agnostic ratios and deviations.

    Registration fields required in the dataset CSV:
      nominal_voltage_v — drives input/output voltage deviation features
      rated_battery_v   — drives battery_voltage_ratio (0 for non-UPS)

    These columns are dropped after use (not model inputs).
    Call BEFORE apply_log1p_skewed so ratios are computed on raw values.
    """
    df = df.copy()
    nomv = df["nominal_voltage_v"].clip(lower=1.0)

    df["input_voltage_dev_pct"]  = (df["input_voltage_v"]  - nomv) / nomv * 100.0
    df["output_voltage_dev_pct"] = (df["output_voltage_v"] - nomv) / nomv * 100.0

    rated_bv = df["rated_battery_v"].clip(lower=1.0)
    df["battery_voltage_ratio"] = df["battery_voltage_v"] / rated_bv
    df.loc[df["rated_battery_v"] == 0, "battery_voltage_ratio"] = 0.0

    df.drop(
        columns=["nominal_voltage_v", "rated_battery_v", "rated_capacity_va",
                 "input_voltage_v", "output_voltage_v", "battery_voltage_v",
                 "input_current_a", "output_current_a", "output_power_w"],
        errors="ignore",
        inplace=True,
    )
    return df


def apply_log1p_skewed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-device rate-of-change features.

    Must be called AFTER apply_log1p_skewed so runtime_remaining_min is already
    log-compressed. signed_log1p squashes extreme delta spikes during anomalies.
    """
    df = df.copy()
    for src, tgt in DELTA_PAIRS:
        diff = df.groupby("device_id")[src].diff().fillna(0.0)
        df[tgt] = np.sign(diff) * np.log1p(np.abs(diff))
    return df


def filter_normal_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly" in df.columns:
        return df[df["anomaly"] == 0].copy()
    return df.copy()


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
    df = normalize_absolute_features(df)
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
