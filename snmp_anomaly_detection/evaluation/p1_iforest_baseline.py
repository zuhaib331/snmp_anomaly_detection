from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from snmp_anomaly_detection.config import (
    EvaluationConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.evaluation.p1_baseline import (
    _classification_metrics,
    _group_summary,
    _timeline_profile,
)
from snmp_anomaly_detection.evaluation.time_split import (
    build_time_split_definition,
    label_timestamp,
)
from snmp_anomaly_detection.inference.detect_isolation_forest import (
    detect_isolation_forest_csv,
    load_iforest_artifacts,
)
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


def build_iforest_metrics_report(
    dataset_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    paths: ProjectPaths,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    artifacts = load_iforest_artifacts(paths)
    split_definition = build_time_split_definition(dataset_frame, evaluation_config)
    results = results_frame.copy()
    results["window_end"] = pd.to_datetime(results["window_end"])
    results["window_start"] = pd.to_datetime(results["window_start"])
    results["split"] = results["window_end"].apply(
        lambda timestamp: label_timestamp(timestamp, split_definition)
    )

    split_metrics = {
        split_name: _classification_metrics(split_frame)
        for split_name, split_frame in results.groupby("split", dropna=False)
    }

    return {
        "stage": "T4",
        "model_type": "IsolationForest",
        "artifact_dir_name": paths.artifact_dir.name,
        "artifact_dir": str(paths.artifact_dir),
        "source_artifact_dir_name": artifacts.metadata.get("source_artifact_dir_name"),
        "source_artifact_dir": artifacts.metadata.get("source_artifact_dir"),
        "dataset_file": str(artifacts.source_paths.dataset_file),
        "evaluation_config": asdict(evaluation_config),
        "feature_columns": list(artifacts.feature_config.feature_columns),
        "time_split_definition": asdict(split_definition),
        "artifact_usage": {
            "iforest_model_file": str(paths.iforest_model_file),
            "iforest_model_exists": paths.iforest_model_file.exists(),
            "iforest_metadata_file": str(paths.iforest_metadata_file),
            "iforest_metadata_exists": paths.iforest_metadata_file.exists(),
            "source_scaler_file": str(artifacts.source_paths.scaler_file),
            "source_scaler_exists": artifacts.source_paths.scaler_file.exists(),
            "source_x_train_file": str(artifacts.source_paths.x_train_file),
            "source_x_train_exists": artifacts.source_paths.x_train_file.exists(),
        },
        "training_metadata": {
            "training_config": artifacts.metadata.get("training_config", {}),
            "input_shape": artifacts.metadata.get("input_shape"),
            "flattened_input_size": artifacts.metadata.get("flattened_input_size"),
            "threshold": artifacts.metadata.get("threshold"),
            "train_anomaly_score_summary": artifacts.metadata.get(
                "train_anomaly_score_summary", {}
            ),
        },
        "dataset_timeline_profile": _timeline_profile(dataset_frame, split_definition),
        "overall_metrics": _classification_metrics(results),
        "metrics_by_split": split_metrics,
        "top_devices_by_false_positives": _group_summary(
            results,
            group_columns=["device_id"],
            top_n=evaluation_config.report_top_n,
        ),
        "top_interfaces_by_false_positives": _group_summary(
            results,
            group_columns=["device_id", "interface"],
            top_n=evaluation_config.report_top_n,
        ),
        "report_notes": [
            "Window-level metrics use source_anomaly_label from replay output as baseline truth.",
            "Time split assignment is based on window_end to preserve temporal ordering.",
            "Isolation Forest is evaluated as a standalone T4 baseline, not as an LSTM ensemble.",
        ],
    }


def evaluate_iforest_baseline(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    paths = paths or ProjectPaths(artifact_dir_name="iforest_f3_v1")
    evaluation_config = evaluation_config or EvaluationConfig()
    paths.ensure_directories()

    artifacts = load_iforest_artifacts(paths)
    dataset_frame = load_dataset(input_file=input_file, paths=artifacts.source_paths)
    results_frame = detect_isolation_forest_csv(
        input_file=input_file,
        paths=paths,
        inference_config=InferenceConfig(save_results=True),
    )
    report = build_iforest_metrics_report(
        dataset_frame=dataset_frame,
        results_frame=results_frame,
        paths=paths,
        evaluation_config=evaluation_config,
    )

    with open(paths.iforest_p1_time_split_file, "w", encoding="utf-8") as file:
        json.dump(report["time_split_definition"], file, indent=2)
    with open(paths.iforest_p1_baseline_metrics_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Isolation Forest P1 time split saved to: {paths.iforest_p1_time_split_file}")
    print(
        "Isolation Forest P1 metrics saved to: "
        f"{paths.iforest_p1_baseline_metrics_file}"
    )
    print(
        "Isolation Forest P1 summary: "
        f"windows={report['overall_metrics']['window_count']} "
        f"predicted_anomalies={report['overall_metrics']['predicted_anomalies']} "
        f"precision={report['overall_metrics']['precision']:.4f} "
        f"recall={report['overall_metrics']['recall']:.4f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Isolation Forest T4 baseline.")
    parser.add_argument(
        "--input-file",
        help="Optional CSV file to score. Defaults to the configured dataset file.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        default="iforest_f3_v1",
        help="Isolation Forest artifact directory under snmp_anomaly_detection/artifacts/.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit Isolation Forest artifact directory path.",
    )
    args = parser.parse_args()

    evaluate_iforest_baseline(
        input_file=args.input_file,
        paths=ProjectPaths(
            artifact_dir_name=args.artifact_dir_name,
            artifact_dir_override=args.artifact_dir,
        ),
    )


if __name__ == "__main__":
    main()
