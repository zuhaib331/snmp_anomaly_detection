from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from snmp_anomaly_detection.config import ProjectPaths


METRIC_KEYS = (
    "window_count",
    "actual_anomalies",
    "predicted_anomalies",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "false_positive_rate",
    "predicted_anomaly_rate",
)


def _load_json(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required P1 report does not exist: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _metrics_file(paths: ProjectPaths, artifact_dir_name: str) -> Path:
    artifact_paths = ProjectPaths(
        artifact_dir_name=artifact_dir_name,
        package_root=paths.package_root,
        repo_root=paths.repo_root,
        data_dir=paths.data_dir,
        outputs_dir=paths.outputs_dir,
        utils_dir=paths.utils_dir,
        artifacts_root_dir=paths.artifacts_root_dir,
        dataset_file=paths.dataset_file,
        anomaly_results_file=paths.anomaly_results_file,
        anomaly_windows_file=paths.anomaly_windows_file,
        kafka_live_results_file=paths.kafka_live_results_file,
        kafka_live_anomaly_windows_file=paths.kafka_live_anomaly_windows_file,
    )
    return artifact_paths.p1_baseline_metrics_file


def _artifact_dir(paths: ProjectPaths, artifact_dir_name: str) -> Path:
    return paths.artifacts_root_dir / artifact_dir_name


def _load_artifact_metadata(paths: ProjectPaths, artifact_dir_name: str) -> dict[str, Any]:
    artifact_dir = _artifact_dir(paths, artifact_dir_name)
    model_metadata_file = artifact_dir / "model_metadata.json"
    preprocessing_metadata_file = artifact_dir / "preprocessing_metadata.json"

    model_metadata = _load_json(model_metadata_file) if model_metadata_file.exists() else {}
    preprocessing_metadata = (
        _load_json(preprocessing_metadata_file) if preprocessing_metadata_file.exists() else {}
    )

    model_feature_config = model_metadata.get("feature_config", {})
    model_scaler = model_feature_config.get("scaler_name")
    preprocessing_scaler = preprocessing_metadata.get("scaler_name")
    model_log1p = model_feature_config.get("log1p_features")
    preprocessing_log1p = preprocessing_metadata.get("log1p_features")

    consistency_warnings: list[str] = []
    if preprocessing_scaler is not None and model_scaler != preprocessing_scaler:
        consistency_warnings.append(
            "model_metadata feature_config.scaler_name does not match "
            "preprocessing_metadata.scaler_name"
        )
    if preprocessing_log1p is not None and model_log1p != preprocessing_log1p:
        consistency_warnings.append(
            "model_metadata feature_config.log1p_features does not match "
            "preprocessing_metadata.log1p_features"
        )

    return {
        "artifact_dir": str(artifact_dir),
        "model_metadata_file": str(model_metadata_file),
        "model_metadata_exists": model_metadata_file.exists(),
        "preprocessing_metadata_file": str(preprocessing_metadata_file),
        "preprocessing_metadata_exists": preprocessing_metadata_file.exists(),
        "model_scaler_name": model_scaler,
        "preprocessing_scaler_name": preprocessing_scaler,
        "model_log1p_features": model_log1p,
        "preprocessing_log1p_features": preprocessing_log1p,
        "consistency_warnings": consistency_warnings,
    }


def _metric_delta(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> dict[str, dict[str, float | int | None]]:
    deltas: dict[str, dict[str, float | int | None]] = {}
    for key in METRIC_KEYS:
        baseline_value = baseline_metrics.get(key)
        candidate_value = candidate_metrics.get(key)
        delta = None
        if isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)):
            delta = candidate_value - baseline_value
        deltas[key] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
    return deltas


def _split_metric_delta(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    baseline_splits = baseline_report.get("metrics_by_split", {})
    candidate_splits = candidate_report.get("metrics_by_split", {})
    split_names = sorted(set(baseline_splits) | set(candidate_splits))

    return {
        split_name: _metric_delta(
            baseline_splits.get(split_name, {}),
            candidate_splits.get(split_name, {}),
        )
        for split_name in split_names
    }


def _summary_key(summary: dict[str, Any]) -> str:
    interface = summary.get("interface")
    if interface is None:
        return str(summary.get("device_id"))
    return f"{summary.get('device_id')}::{interface}"


def _compare_top_interfaces(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    top_n: int,
) -> dict[str, Any]:
    baseline_items = {
        _summary_key(item): item for item in baseline_report.get("top_interfaces_by_false_positives", [])
    }
    candidate_items = {
        _summary_key(item): item
        for item in candidate_report.get("top_interfaces_by_false_positives", [])
    }

    compared: list[dict[str, Any]] = []
    for key in sorted(set(baseline_items) | set(candidate_items)):
        baseline_item = baseline_items.get(key, {})
        candidate_item = candidate_items.get(key, {})
        compared.append(
            {
                "interface_key": key,
                "device_id": candidate_item.get("device_id", baseline_item.get("device_id")),
                "interface": candidate_item.get("interface", baseline_item.get("interface")),
                "baseline_false_positive": baseline_item.get("false_positive", 0),
                "candidate_false_positive": candidate_item.get("false_positive", 0),
                "false_positive_delta": int(candidate_item.get("false_positive", 0))
                - int(baseline_item.get("false_positive", 0)),
                "baseline_predicted_anomalies": baseline_item.get("predicted_anomalies", 0),
                "candidate_predicted_anomalies": candidate_item.get("predicted_anomalies", 0),
                "candidate_top_error_feature": candidate_item.get("top_error_feature"),
                "baseline_top_error_feature": baseline_item.get("top_error_feature"),
            }
        )

    regressions = sorted(compared, key=lambda item: item["false_positive_delta"], reverse=True)[:top_n]
    improvements = sorted(compared, key=lambda item: item["false_positive_delta"])[:top_n]

    return {
        "comparison_scope": "saved top_interfaces_by_false_positives lists",
        "scope_note": (
            "This compares the top interface summaries saved in each P1 report. "
            "It is not a full per-interface recomputation from raw replay rows."
        ),
        "largest_false_positive_regressions": regressions,
        "largest_false_positive_improvements": improvements,
    }


def _recommendation(overall_delta: dict[str, dict[str, float | int | None]]) -> str:
    precision_delta = overall_delta["precision"]["delta"]
    recall_delta = overall_delta["recall"]["delta"]
    fpr_delta = overall_delta["false_positive_rate"]["delta"]

    if precision_delta is None or recall_delta is None or fpr_delta is None:
        return "review_missing_metrics"
    if precision_delta >= 0 and recall_delta >= 0 and fpr_delta <= 0:
        return "promote_candidate"
    if recall_delta > 0 and fpr_delta > 0:
        return "review_tradeoff"
    return "keep_baseline"


def _compare_candidate(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    paths: ProjectPaths,
    top_n: int,
) -> dict[str, Any]:
    candidate_name = candidate_report["artifact_dir_name"]
    overall_delta = _metric_delta(
        baseline_report["overall_metrics"],
        candidate_report["overall_metrics"],
    )
    return {
        "candidate_artifact": candidate_name,
        "candidate_metrics_file": str(_metrics_file(paths, candidate_name)),
        "artifact_metadata": _load_artifact_metadata(paths, candidate_name),
        "overall_delta": overall_delta,
        "split_deltas": _split_metric_delta(baseline_report, candidate_report),
        "interface_false_positive_highlights": _compare_top_interfaces(
            baseline_report,
            candidate_report,
            top_n,
        ),
        "recommendation": _recommendation(overall_delta),
    }


def compare_p1_artifacts(
    baseline_artifact_dir_name: str,
    candidate_artifact_dir_names: list[str],
    paths: ProjectPaths | None = None,
    output_file: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    baseline_report = _load_json(_metrics_file(paths, baseline_artifact_dir_name))
    candidate_reports = [
        _load_json(_metrics_file(paths, candidate_name))
        for candidate_name in candidate_artifact_dir_names
    ]

    report = {
        "stage": "P1.1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "baseline_artifact": baseline_artifact_dir_name,
        "baseline_metrics_file": str(_metrics_file(paths, baseline_artifact_dir_name)),
        "baseline_artifact_metadata": _load_artifact_metadata(paths, baseline_artifact_dir_name),
        "candidate_artifacts": list(candidate_artifact_dir_names),
        "comparison_notes": [
            "Deltas are candidate minus baseline.",
            "Positive precision and recall deltas are better.",
            "Negative false_positive_rate and false_positive deltas are better.",
            "Interface highlights use the top interface summaries saved in P1 reports.",
        ],
        "comparisons": [
            _compare_candidate(
                baseline_report=baseline_report,
                candidate_report=candidate_report,
                paths=paths,
                top_n=top_n,
            )
            for candidate_report in candidate_reports
        ],
    }

    if output_file:
        report_file = Path(output_file)
        if not report_file.is_absolute():
            report_file = paths.repo_root / report_file
    else:
        joined_candidates = (
            candidate_artifact_dir_names[0]
            if len(candidate_artifact_dir_names) == 1
            else f"{len(candidate_artifact_dir_names)}_candidates"
        )
        report_file = paths.outputs_dir / (
            f"{baseline_artifact_dir_name}_vs_{joined_candidates}_p1_1_comparison.json"
        )

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"P1.1 comparison saved to: {report_file}")
    for comparison in report["comparisons"]:
        delta = comparison["overall_delta"]
        print(
            "P1.1 comparison summary: "
            f"candidate={comparison['candidate_artifact']} "
            f"precision_delta={delta['precision']['delta']:+.4f} "
            f"recall_delta={delta['recall']['delta']:+.4f} "
            f"fpr_delta={delta['false_positive_rate']['delta']:+.4f} "
            f"recommendation={comparison['recommendation']}"
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P1.1 artifact-to-artifact comparison.")
    parser.add_argument(
        "--baseline-artifact-dir-name",
        default="f1_baseline_v1",
        help="Baseline artifact directory name used to locate its saved P1 metrics report.",
    )
    parser.add_argument(
        "--candidate-artifact-dir-name",
        action="append",
        required=True,
        help="Candidate artifact directory name. Repeat this option to compare multiple candidates.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional comparison report path. Defaults to snmp_anomaly_detection/outputs/.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top interface regression/improvement highlights to include.",
    )
    args = parser.parse_args()

    compare_p1_artifacts(
        baseline_artifact_dir_name=args.baseline_artifact_dir_name,
        candidate_artifact_dir_names=args.candidate_artifact_dir_name,
        output_file=args.output_file,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
