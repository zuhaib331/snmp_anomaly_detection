"""Feature engineering and RUL label derivation for battery health prediction.

RUL (Remaining Useful Life) is the number of days until the battery reaches
end-of-life (capacity degraded to ~50% of rated). The label is derived
analytically from the synthetic device profile metadata embedded in the dataset.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from snmp_anomaly_detection.config import BATTERY_RUL_FEATURES, BATTERY_RUL_CATEGORIES, ProjectPaths
from snmp_anomaly_detection.preprocessing.power_features import (
    apply_log1p_skewed,
    load_power_dataset,
    create_sequences,
)

# Expected battery life per vendor (years) — used to compute RUL labels
_VENDOR_LIFE_YEARS: dict[str, float] = {
    "apc": 3.0,
    "liebert": 5.0,
    "generic": 3.0,
}

# interval in minutes (must match PowerDatasetConfig.interval_minutes)
_INTERVAL_MINUTES: int = 5


def derive_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a rul_days column computed from install_age_days embedded in device_id.

    For synthetic data, the age at each timestep is reconstructed from
    the row index within each device group and the polling interval.
    """
    df = df.copy()
    rul_rows = []
    for device_id, group in df.groupby("device_id"):
        vendor = group["vendor"].iloc[0]
        life_years = _VENDOR_LIFE_YEARS.get(str(vendor).lower(), 3.0)
        life_days = life_years * 365
        install_age = float(group["install_age_days"].iloc[0]) if "install_age_days" in group.columns else 0.0
        for i, (idx, _row) in enumerate(group.iterrows()):
            age_days = install_age + i * _INTERVAL_MINUTES / 1440
            rul = max(life_days - age_days, 0.0)
            rul_rows.append((idx, rul))

    rul_series = pd.Series(dict(rul_rows), name="rul_days")
    df["rul_days"] = rul_series
    return df


def build_rul_sequences(
    df: pd.DataFrame,
    seq_len: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequences, rul_labels) where each label is the RUL at the END of the window.

    Uses only UPS devices (those with non-zero battery_ah proxy: battery_voltage_v > 0).
    """
    feature_cols = [c for c in BATTERY_RUL_FEATURES if c in df.columns]
    ups_df = df[df["device_category"].isin(BATTERY_RUL_CATEGORIES)].copy()

    all_seqs, all_labels = [], []
    for device_id in ups_df["device_id"].unique():
        device_df = ups_df[ups_df["device_id"] == device_id]
        values = device_df[feature_cols].values
        rul_vals = device_df["rul_days"].values
        for i in range(len(values) - seq_len):
            all_seqs.append(values[i : i + seq_len])
            all_labels.append(rul_vals[i + seq_len - 1])  # label at window end

    seqs = np.array(all_seqs) if all_seqs else np.empty((0, seq_len, len(feature_cols)))
    labels = np.array(all_labels, dtype=np.float32)
    return seqs, labels


def build_per_device_rul_split(
    df: pd.DataFrame,
    seq_len: int = 24,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict:
    """Per-device 70/15/15 temporal split — every UPS contributes to all three sets.

    The global last-15% split dumps one device's entire sequence into the test
    set, producing misleading metrics. Splitting per device first then
    concatenating ensures the test set spans all devices (F10).
    """
    feature_cols = [c for c in BATTERY_RUL_FEATURES if c in df.columns]
    ups_df = df[df["device_category"].isin(BATTERY_RUL_CATEGORIES)].copy()

    train_seqs, val_seqs, test_seqs = [], [], []
    train_lbls, val_lbls, test_lbls = [], [], []

    for device_id in ups_df["device_id"].unique():
        device_df = ups_df[ups_df["device_id"] == device_id]
        values = device_df[feature_cols].values
        rul_vals = device_df["rul_days"].values

        seqs, lbls = [], []
        for i in range(len(values) - seq_len):
            seqs.append(values[i : i + seq_len])
            lbls.append(rul_vals[i + seq_len - 1])

        if not seqs:
            continue

        seqs_arr = np.array(seqs)
        lbls_arr = np.array(lbls, dtype=np.float32)
        n = len(seqs_arr)
        t_end = int(n * train_frac)
        v_end = int(n * (train_frac + val_frac))

        train_seqs.append(seqs_arr[:t_end])
        val_seqs.append(seqs_arr[t_end:v_end])
        test_seqs.append(seqs_arr[v_end:])
        train_lbls.append(lbls_arr[:t_end])
        val_lbls.append(lbls_arr[t_end:v_end])
        test_lbls.append(lbls_arr[v_end:])

    return {
        "x_train": np.concatenate(train_seqs),
        "y_train": np.concatenate(train_lbls),
        "x_val":   np.concatenate(val_seqs),
        "y_val":   np.concatenate(val_lbls),
        "x_test":  np.concatenate(test_seqs),
        "y_test":  np.concatenate(test_lbls),
    }


def scale_battery_features(
    df: pd.DataFrame,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, RobustScaler]:
    scaler_path = paths.battery_rul_outputs_dir / "rul_scaler.pkl"
    feature_cols = [c for c in BATTERY_RUL_FEATURES if c in df.columns]
    scaler = RobustScaler()
    out = df.copy()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    joblib.dump(scaler, scaler_path)
    return out, scaler


def run_battery_feature_engineering(
    seq_len: int = 24,
    paths: ProjectPaths | None = None,
) -> dict:
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    df = load_power_dataset(paths)
    df = apply_log1p_skewed(df)
    df = derive_rul_labels(df)
    scaled_df, _ = scale_battery_features(df, paths)
    sequences, labels = build_rul_sequences(scaled_df, seq_len)

    np.save(paths.battery_rul_outputs_dir / "X_rul.npy", sequences)
    np.save(paths.battery_rul_outputs_dir / "y_rul.npy", labels)

    feature_cols = [c for c in BATTERY_RUL_FEATURES if c in df.columns]
    meta = {
        "seq_len": seq_len,
        "feature_columns": feature_cols,
        "num_sequences": len(sequences),
        "shape": list(sequences.shape),
        "label_min": float(labels.min()) if len(labels) else 0,
        "label_max": float(labels.max()) if len(labels) else 0,
    }
    with open(paths.battery_rul_outputs_dir / "rul_preprocess_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"sequences": sequences, "labels": labels}
