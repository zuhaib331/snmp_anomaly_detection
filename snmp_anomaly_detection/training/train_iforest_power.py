"""Train per-category Isolation Forest on normal BASELINE_UPS_FEATURES rows.

Complements the LSTM Autoencoder by scoring individual feature vectors in
isolation — catches point anomalies (sudden spikes) that the sequence-based
LSTM dilutes across its window.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, IF_EXCLUDED_FEATURES, ProjectPaths
from snmp_anomaly_detection.preprocessing.power_features import (
    load_power_dataset,
    normalize_absolute_features,
    apply_log1p_skewed,
    add_delta_features,
    filter_normal_rows,
)


def train_iforest_power(paths: ProjectPaths | None = None) -> dict:
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    df = load_power_dataset(paths)
    df = normalize_absolute_features(df)
    df = apply_log1p_skewed(df)
    df = add_delta_features(df)
    normal_df = filter_normal_rows(df)

    # Reuse the baseline scaler so IF sees the same feature space as the LSTM.
    # baseline_scaler.pkl must exist — run train-power-baseline first.
    scaler = joblib.load(paths.power_outputs_dir / "baseline_scaler.pkl")
    all_feature_cols = [c for c in BASELINE_UPS_FEATURES if c in normal_df.columns]

    models: dict[str, IsolationForest] = {}
    metadata: dict[str, dict] = {}

    for category, group in normal_df.groupby("device_category"):
        excluded = IF_EXCLUDED_FEATURES.get(str(category), frozenset())
        cat_feature_cols = [f for f in all_feature_cols if f not in excluded]
        col_indices = [all_feature_cols.index(f) for f in cat_feature_cols]
        X_all = scaler.transform(group[all_feature_cols])
        X = X_all[:, col_indices]
        clf = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
        clf.fit(X)
        scores = clf.score_samples(X)
        # 1st-percentile threshold: flag bottom 1% of training scores as anomalous.
        threshold = float(np.percentile(scores, 1))
        models[str(category)] = clf
        metadata[str(category)] = {
            "threshold": round(threshold, 8),
            "n_train_samples": int(len(X)),
            "score_mean": round(float(scores.mean()), 8),
            "score_std": round(float(scores.std()), 8),
            "feature_cols": cat_feature_cols,
        }
        print(f"  [{category}]  n={len(X):6d}  n_features={len(cat_feature_cols)}"
              f"  threshold={threshold:.4f}  score_mean={scores.mean():.4f}")

    joblib.dump(models, paths.iforest_model_file)
    with open(paths.iforest_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nIF models:    {paths.iforest_model_file}")
    print(f"IF metadata:  {paths.iforest_metadata_file}")
    return metadata


def main() -> None:
    train_iforest_power()
