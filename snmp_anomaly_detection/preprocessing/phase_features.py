"""Feature engineering for the phase-level anomaly model.

Builds sequences from raw per-phase L1/L2/L3 voltage and current columns
plus the derived imbalance metrics, without aggregating across phases.
This allows the model to detect asymmetric single-phase failures that the
aggregated baseline model cannot see.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from snmp_anomaly_detection.config import PHASE_LEVEL_FEATURES, ProjectPaths
from snmp_anomaly_detection.preprocessing.power_features import (
    apply_log1p_skewed,
    load_power_dataset,
    filter_normal_rows,
    create_sequences,
)


def compute_voltage_imbalance_pct(
    df: pd.DataFrame,
    l1_col: str = "input_voltage_l1",
    l2_col: str = "input_voltage_l2",
    l3_col: str = "input_voltage_l3",
) -> pd.Series:
    """NEMA MG-1 voltage imbalance: max deviation from average / average * 100."""
    avg = df[[l1_col, l2_col, l3_col]].mean(axis=1)
    max_dev = (df[[l1_col, l2_col, l3_col]].subtract(avg, axis=0)).abs().max(axis=1)
    return (max_dev / avg.replace(0, np.nan) * 100).fillna(0.0)


def compute_current_skew_pct(
    df: pd.DataFrame,
    l1_col: str = "input_current_l1",
    l2_col: str = "input_current_l2",
    l3_col: str = "input_current_l3",
) -> pd.Series:
    """Current skew: same formula as voltage imbalance applied to currents."""
    avg = df[[l1_col, l2_col, l3_col]].mean(axis=1)
    max_dev = (df[[l1_col, l2_col, l3_col]].subtract(avg, axis=0)).abs().max(axis=1)
    return (max_dev / avg.replace(0, np.nan) * 100).fillna(0.0)


def enrich_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute voltage_imbalance_pct and current_skew_pct from raw phase columns."""
    phase_v_cols = ["input_voltage_l1", "input_voltage_l2", "input_voltage_l3"]
    phase_i_cols = ["input_current_l1", "input_current_l2", "input_current_l3"]
    df = df.copy()
    if all(c in df.columns for c in phase_v_cols):
        df["voltage_imbalance_pct"] = compute_voltage_imbalance_pct(df)
    if all(c in df.columns for c in phase_i_cols):
        df["current_skew_pct"] = compute_current_skew_pct(df)
    return df


# Physical max for imbalance features — >5% is already abnormal; >10% is fault level.
# Clipping prevents near-zero IQR from destroying RobustScaler for these columns.
_IMBALANCE_CLIP_MAX: float = 10.0
_IMBALANCE_COLS: tuple[str, ...] = ("voltage_imbalance_pct", "current_skew_pct")


def scale_phase_features(
    df: pd.DataFrame,
    paths: ProjectPaths,
    scaler_path: Path | None = None,
) -> tuple[pd.DataFrame, RobustScaler]:
    scaler_path = scaler_path or paths.power_phase_outputs_dir / "phase_scaler.pkl"
    feature_cols = [c for c in PHASE_LEVEL_FEATURES if c in df.columns]
    out = df.copy()
    # Clip imbalance columns before scaling so near-zero IQR doesn't cause overflow
    for col in _IMBALANCE_COLS:
        if col in out.columns:
            out[col] = out[col].clip(0.0, _IMBALANCE_CLIP_MAX) / _IMBALANCE_CLIP_MAX
    scaler = RobustScaler()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    joblib.dump(scaler, scaler_path)
    return out, scaler


def build_phase_sequences(
    df: pd.DataFrame,
    seq_len: int = 10,
) -> np.ndarray:
    feature_cols = [c for c in PHASE_LEVEL_FEATURES if c in df.columns]
    chunks = []
    for device_id in df["device_id"].unique():
        device_df = df[df["device_id"] == device_id]
        seqs = create_sequences(device_df[feature_cols].values, seq_len)
        if len(seqs):
            chunks.append(seqs)
    if not chunks:
        return np.empty((0, seq_len, len(feature_cols)))
    return np.concatenate(chunks, axis=0)


def run_phase_feature_engineering(
    seq_len: int = 10,
    paths: ProjectPaths | None = None,
) -> dict[str, np.ndarray]:
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    df = load_power_dataset(paths)
    df = apply_log1p_skewed(df)
    df = enrich_imbalance_features(df)
    train_df = filter_normal_rows(df)
    scaled_df, _ = scale_phase_features(train_df, paths)
    sequences = build_phase_sequences(scaled_df, seq_len)

    np.save(paths.power_phase_outputs_dir / "X_train_phase.npy", sequences)

    feature_cols = [c for c in PHASE_LEVEL_FEATURES if c in df.columns]
    meta = {
        "seq_len": seq_len,
        "feature_columns": feature_cols,
        "num_sequences": len(sequences),
        "shape": list(sequences.shape),
    }
    with open(paths.power_phase_outputs_dir / "preprocess_phase_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"X_train_phase": sequences}
