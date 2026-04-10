from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from snmp_anomaly_detection.config import (
    EvaluationConfig,
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.evaluation.time_split import (
    TimeSplitDefinition,
    apply_time_split_labels,
    build_time_split_definition,
    label_timestamp,
)
from snmp_anomaly_detection.inference.core import resolve_feature_config_for_artifact
from snmp_anomaly_detection.inference.csv_replay import detect_csv_replay
from snmp_anomaly_detection.preprocessing.feature_engineering import load_dataset


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_metrics(dataframe: pd.DataFrame) -> dict[str, float | int]:
    actual = dataframe["source_anomaly_label"].astype(int)
    predicted = dataframe["predicted_anomaly"].astype(int)

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    return {
        "window_count": int(len(dataframe)),
        "actual_anomalies": int(actual.sum()),
        "predicted_anomalies": int(predicted.sum()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _safe_divide(true_positive, true_positive + false_positive),
        "recall": _safe_divide(true_positive, true_positive + false_negative),
        "false_positive_rate": _safe_divide(false_positive, false_positive + true_negative),
        "predicted_anomaly_rate": _safe_divide(int(predicted.sum()), len(dataframe)),
    }


def _group_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    top_n: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    grouped = dataframe.groupby(group_columns, dropna=False)
    for group_key, group_frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        summary = {column: value for column, value in zip(group_columns, group_key)}
        summary.update(_classification_metrics(group_frame))
        summary["top_error_feature"] = (
            group_frame["top_error_feature"].mode(dropna=True).iloc[0]
            if not group_frame["top_error_feature"].mode(dropna=True).empty
            else None
        )
        summaries.append(summary)

    return sorted(
        summaries,
        key=lambda item: (
            int(item["false_positive"]),
            int(item["predicted_anomalies"]),
            str(item[group_columns[0]]),
        ),
        reverse=True,
    )[:top_n]


def _timeline_profile(dataframe: pd.DataFrame, split_definition: TimeSplitDefinition) -> dict[str, object]:
    labeled = apply_time_split_labels(dataframe, split_definition)

    profile: dict[str, object] = {
        "maintenance_label_available": False,
        "note": "Current synthetic dataset exposes anomaly labels but no separate maintenance label.",
        "rows_by_split": {},
    }

    for split_name, split_frame in labeled.groupby("split", dropna=False):
        profile["rows_by_split"][split_name] = {
            "row_count": int(len(split_frame)),
            "known_normal_rows": int((split_frame["anomaly"] == 0).sum()),
            "suspicious_rows": int((split_frame["anomaly"] == 1).sum()),
            "maintenance_rows": 0,
        }

    return profile


def build_baseline_metrics_report(
    dataset_frame: pd.DataFrame,
    results_frame: pd.DataFrame,
    paths: ProjectPaths,
    feature_config: FeatureEngineeringConfig,
    evaluation_config: EvaluationConfig,
    split_definition: TimeSplitDefinition,
) -> dict[str, object]:
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
        "stage": "P1",
        "artifact_dir_name": paths.artifact_dir.name,
        "artifact_dir": str(paths.artifact_dir),
        "dataset_file": str(paths.dataset_file),
        "evaluation_config": asdict(evaluation_config),
        "feature_columns": list(feature_config.feature_columns),
        "time_split_definition": asdict(split_definition),
        "artifact_usage": {
            "scaler_file": str(paths.scaler_file),
            "scaler_exists": paths.scaler_file.exists(),
            "x_train_file": str(paths.x_train_file),
            "x_train_exists": paths.x_train_file.exists(),
            "x_test_file": str(paths.x_test_file),
            "x_test_exists": paths.x_test_file.exists(),
            "model_file": str(paths.model_file),
            "model_exists": paths.model_file.exists(),
            "model_metadata_file": str(paths.model_metadata_file),
            "model_metadata_exists": paths.model_metadata_file.exists(),
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
            "Maintenance periods are not separately labeled in the current synthetic dataset.",
        ],
    }


def evaluate_p1_baseline(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict[str, object]:
    paths = paths or ProjectPaths()
    feature_config = resolve_feature_config_for_artifact(paths=paths, config=feature_config)
    evaluation_config = evaluation_config or EvaluationConfig()
    paths.ensure_directories()

    dataset_frame = load_dataset(input_file=input_file, paths=paths)
    split_definition = build_time_split_definition(dataset_frame, evaluation_config)
    results_frame = detect_csv_replay(
        input_file=input_file,
        paths=paths,
        config=feature_config,
        inference_config=InferenceConfig(save_results=True),
    )
    report = build_baseline_metrics_report(
        dataset_frame=dataset_frame,
        results_frame=results_frame,
        paths=paths,
        feature_config=feature_config,
        evaluation_config=evaluation_config,
        split_definition=split_definition,
    )

    with open(paths.p1_time_split_file, "w", encoding="utf-8") as file:
        json.dump(asdict(split_definition), file, indent=2)
    with open(paths.p1_baseline_metrics_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"P1 time split saved to: {paths.p1_time_split_file}")
    print(f"P1 baseline metrics saved to: {paths.p1_baseline_metrics_file}")
    print(
        "P1 baseline summary: "
        f"windows={report['overall_metrics']['window_count']} "
        f"predicted_anomalies={report['overall_metrics']['predicted_anomalies']} "
        f"precision={report['overall_metrics']['precision']:.4f} "
        f"recall={report['overall_metrics']['recall']:.4f}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1 baseline evaluation and reporting.")
    parser.add_argument(
        "--input-file",
        help="Optional CSV file to score. Defaults to the configured dataset file.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        help="Named artifact directory under snmp_anomaly_detection/artifacts/ to evaluate.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path to evaluate.",
    )
    args = parser.parse_args()

    default_paths = ProjectPaths()
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or default_paths.artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    evaluate_p1_baseline(input_file=args.input_file, paths=paths)


if __name__ == "__main__":
    main()
