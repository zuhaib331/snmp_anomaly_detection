from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    IsolationForestConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.inference.core import resolve_feature_config_for_artifact
from snmp_anomaly_detection.models.isolation_forest import flatten_sequences


def _load_training_sequences(paths: ProjectPaths) -> np.ndarray:
    x_train_path = paths.x_train_file if paths.x_train_file.exists() else paths.legacy_x_train_file
    return np.load(x_train_path)


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "min": float(scores.min()) if len(scores) else 0.0,
        "max": float(scores.max()) if len(scores) else 0.0,
        "mean": float(scores.mean()) if len(scores) else 0.0,
        "std": float(scores.std()) if len(scores) else 0.0,
        "p95": float(np.percentile(scores, 95)) if len(scores) else 0.0,
        "p99": float(np.percentile(scores, 99)) if len(scores) else 0.0,
    }


def train_isolation_forest(
    paths: ProjectPaths | None = None,
    source_paths: ProjectPaths | None = None,
    config: IsolationForestConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
) -> dict[str, Any]:
    paths = paths or ProjectPaths(artifact_dir_name="iforest_f3_v1")
    source_paths = source_paths or ProjectPaths(artifact_dir_name="f3_t3_seq10_p995_v1")
    config = config or IsolationForestConfig()
    paths.ensure_directories()

    feature_config = feature_config or resolve_feature_config_for_artifact(paths=source_paths)
    x_train = _load_training_sequences(source_paths)
    x_train_flat = flatten_sequences(x_train)

    model = IsolationForest(
        n_estimators=config.n_estimators,
        contamination=config.contamination,
        max_samples=config.max_samples,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    model.fit(x_train_flat)

    anomaly_scores = -model.score_samples(x_train_flat)
    threshold = float(-model.offset_)
    flattened_mean = x_train_flat.mean(axis=0)
    flattened_std = x_train_flat.std(axis=0)

    joblib.dump(model, paths.iforest_model_file)
    metadata: dict[str, Any] = {
        "artifact_dir": str(paths.artifact_dir),
        "source_artifact_dir_name": source_paths.artifact_dir.name,
        "source_artifact_dir": str(source_paths.artifact_dir),
        "source_scaler_file": str(source_paths.scaler_file),
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model_type": "IsolationForest",
        "feature_config": asdict(feature_config),
        "training_config": asdict(config),
        "input_shape": [int(size) for size in x_train.shape],
        "flattened_input_size": int(x_train_flat.shape[1]) if x_train_flat.ndim == 2 else 0,
        "threshold": threshold,
        "threshold_source": "negative IsolationForest offset_ so higher anomaly_score means more anomalous",
        "train_anomaly_score_summary": _score_summary(anomaly_scores),
        "flattened_feature_mean": flattened_mean.tolist(),
        "flattened_feature_std": flattened_std.tolist(),
    }
    with open(paths.iforest_metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    summary = {
        "artifact_dir": str(paths.artifact_dir),
        "model_file": str(paths.iforest_model_file),
        "metadata_file": str(paths.iforest_metadata_file),
        "source_artifact_dir": str(source_paths.artifact_dir),
        "num_training_sequences": int(len(x_train)),
        "flattened_input_size": metadata["flattened_input_size"],
        "threshold": threshold,
    }
    print(f"Isolation Forest model saved to: {paths.iforest_model_file}")
    print(f"Isolation Forest metadata saved to: {paths.iforest_metadata_file}")
    print(f"Source artifact directory: {source_paths.artifact_dir}")
    print(f"Training sequences: {len(x_train)}")
    print(f"Anomaly score threshold: {threshold:.6f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Isolation Forest baseline.")
    parser.add_argument(
        "--artifact-dir-name",
        default="iforest_f3_v1",
        help="Artifact directory name for the Isolation Forest model.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path for the Isolation Forest model.",
    )
    parser.add_argument(
        "--source-artifact-dir-name",
        default="f3_t3_seq10_p995_v1",
        help="Existing LSTM/preprocessing artifact directory to read X_train and scaler metadata from.",
    )
    parser.add_argument(
        "--source-artifact-dir",
        help="Explicit source artifact directory path.",
    )
    parser.add_argument("--n-estimators", type=int, default=IsolationForestConfig.n_estimators)
    parser.add_argument("--contamination", default=str(IsolationForestConfig.contamination))
    parser.add_argument("--random-state", type=int, default=IsolationForestConfig.random_state)
    args = parser.parse_args()

    contamination: float | str
    contamination = "auto" if args.contamination == "auto" else float(args.contamination)
    train_isolation_forest(
        paths=ProjectPaths(
            artifact_dir_name=args.artifact_dir_name,
            artifact_dir_override=args.artifact_dir,
        ),
        source_paths=ProjectPaths(
            artifact_dir_name=args.source_artifact_dir_name,
            artifact_dir_override=args.source_artifact_dir,
        ),
        config=IsolationForestConfig(
            n_estimators=args.n_estimators,
            contamination=contamination,
            random_state=args.random_state,
        ),
    )


if __name__ == "__main__":
    main()
