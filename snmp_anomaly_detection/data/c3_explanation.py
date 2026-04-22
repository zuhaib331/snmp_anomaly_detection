from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from snmp_anomaly_detection.config import ProjectPaths


C3_HIGH_CONFIDENCE_SCORE = 150
C3_MEDIUM_CONFIDENCE_SCORE = 120

FEATURE_LABELS: dict[str, str] = {
    "cpu": "CPU behavior",
    "memory": "memory behavior",
    "in_rate": "inbound traffic rate",
    "out_rate": "outbound traffic rate",
    "error_rate": "interface error rate",
    "discard_rate_in": "inbound discard rate",
    "discard_rate_out": "outbound discard rate",
    "utilization_in_pct": "inbound utilization",
    "utilization_out_pct": "outbound utilization",
    "packet_rate_in": "inbound packet rate",
    "packet_rate_out": "outbound packet rate",
    "interface_oper_status": "interface operational status",
    "interface_admin_status": "interface administrative status",
    "burst_indicator": "traffic burst behavior",
}

EVENT_LABELS: dict[str, str] = {
    "admin_down": "administrative shutdown",
    "counter_reset": "counter reset",
    "device_reboot": "device reboot",
    "high_error_rate": "high error-rate event",
    "interface_down": "interface-down event",
    "interface_up": "interface-up event",
    "link_flap": "link-flap event",
    "oper_down": "operational-down event",
    "unknown_event": "nearby event",
}


def _resolve_path(path_value: str | None, default_path: Path, paths: ProjectPaths) -> Path:
    path = Path(path_value) if path_value else default_path
    if not path.is_absolute():
        path = paths.repo_root / path
    return path


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}.")
    return [record for record in payload if isinstance(record, dict)]


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _feature_base(feature_name: Any) -> str:
    text = _clean_string(feature_name)
    for suffix in (
        "_rolling_mean",
        "_rolling_std",
        "_zscore",
        "_trend",
    ):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _feature_label(feature_name: Any) -> str:
    base_feature = _feature_base(feature_name)
    return FEATURE_LABELS.get(base_feature, base_feature or "unknown feature")


def _event_label(event_type: Any) -> str:
    event_name = _clean_string(event_type) or "unknown_event"
    return EVENT_LABELS.get(event_name, event_name.replace("_", " "))


def _time_phrase(minutes_from_window_end: Any) -> str:
    try:
        minutes = float(minutes_from_window_end)
    except (TypeError, ValueError):
        return "near the anomaly window"

    rounded_minutes = int(round(abs(minutes)))
    unit = "minute" if rounded_minutes == 1 else "minutes"
    if minutes < 0:
        return f"{rounded_minutes} {unit} before the anomaly window ended"
    if minutes > 0:
        return f"{rounded_minutes} {unit} after the anomaly window ended"
    return "at the anomaly window end"


def _confidence_from_event(best_event: dict[str, Any] | None) -> str:
    if best_event is None:
        return "low"
    score = int(best_event.get("correlation_score", 0) or 0)
    if score >= C3_HIGH_CONFIDENCE_SCORE:
        return "high"
    if score >= C3_MEDIUM_CONFIDENCE_SCORE:
        return "medium"
    return "low"


def _entity_label(anomaly: dict[str, Any]) -> str:
    device_id = _clean_string(anomaly.get("device_id")) or "unknown device"
    interface = _clean_string(anomaly.get("interface"))
    if interface:
        return f"{device_id} {interface}"
    return device_id


def _build_explanation(anomaly: dict[str, Any]) -> dict[str, Any]:
    correlation = anomaly.get("c2_correlation", {})
    if not isinstance(correlation, dict):
        correlation = {}

    best_event = correlation.get("best_event")
    if not isinstance(best_event, dict):
        best_event = None

    top_feature = _clean_string(anomaly.get("top_error_feature")) or "unknown_feature"
    feature_label = _feature_label(top_feature)
    entity = _entity_label(anomaly)
    evidence = [
        f"top_model_error_feature={top_feature}",
        f"reconstruction_error={float(anomaly.get('reconstruction_error', 0.0) or 0.0):.6f}",
        f"threshold={float(anomaly.get('threshold', 0.0) or 0.0):.6f}",
    ]

    if best_event is None:
        explanation = (
            f"Anomaly on {entity} has no matched normalized event in the C2 correlation "
            f"window. Model evidence points most strongly to {feature_label}."
        )
        return {
            "explanation": explanation,
            "explanation_type": "model_only",
            "confidence": "low",
            "has_correlated_event": False,
            "top_error_feature": top_feature,
            "related_event_type": None,
            "related_event_severity": None,
            "related_event_time_delta_minutes": None,
            "evidence": evidence,
        }

    event_type = _clean_string(best_event.get("event_type")) or "unknown_event"
    event_severity = _clean_string(best_event.get("severity")) or "unknown"
    time_phrase = _time_phrase(best_event.get("minutes_from_window_end"))
    event_label = _event_label(event_type)
    confidence = _confidence_from_event(best_event)
    explanation = (
        f"Anomaly on {entity} is likely related to a {event_label} "
        f"{time_phrase}. Top anomaly feature: {top_feature}."
    )
    evidence.extend(
        [
            f"matched_event_type={event_type}",
            f"matched_event_severity={event_severity}",
            f"matched_event_score={int(best_event.get('correlation_score', 0) or 0)}",
            f"matched_event_time_delta_minutes={float(best_event.get('minutes_from_window_end', 0.0) or 0.0):.2f}",
        ]
    )

    return {
        "explanation": explanation,
        "explanation_type": "correlated_event",
        "confidence": confidence,
        "has_correlated_event": True,
        "top_error_feature": top_feature,
        "related_event_type": event_type,
        "related_event_severity": event_severity,
        "related_event_time_delta_minutes": best_event.get("minutes_from_window_end"),
        "related_event_correlation_score": best_event.get("correlation_score"),
        "evidence": evidence,
    }


def explain_correlated_anomaly_windows(
    correlated_anomaly_windows_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    input_path = _resolve_path(
        correlated_anomaly_windows_file,
        paths.c2_correlated_anomaly_windows_file,
        paths,
    )
    correlated_windows = _read_json_file(input_path)

    explained_windows: list[dict[str, Any]] = []
    explanation_type_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    top_feature_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()

    for anomaly in correlated_windows:
        explanation = _build_explanation(anomaly)
        enriched = dict(anomaly)
        enriched["c3_explanation"] = explanation
        explained_windows.append(enriched)

        explanation_type_counts.update([str(explanation["explanation_type"])])
        confidence_counts.update([str(explanation["confidence"])])
        top_feature_counts.update([str(explanation["top_error_feature"])])
        if explanation.get("related_event_type"):
            event_type_counts.update([str(explanation["related_event_type"])])

    summary: dict[str, Any] = {
        "stage": "C3",
        "status": "completed",
        "correlated_anomaly_windows_file": str(input_path),
        "output_files": {
            "explained_anomaly_windows": str(paths.c3_explained_anomaly_windows_file),
            "explanation_summary": str(paths.c3_explanation_summary_file),
        },
        "input_counts": {
            "correlated_anomaly_window_count": len(correlated_windows),
        },
        "explanation_counts": {
            "explained_anomaly_window_count": len(explained_windows),
            "with_correlated_event": int(explanation_type_counts.get("correlated_event", 0)),
            "model_only": int(explanation_type_counts.get("model_only", 0)),
        },
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "explanation_type_counts": dict(sorted(explanation_type_counts.items())),
        "top_error_feature_counts": dict(top_feature_counts.most_common(10)),
        "related_event_type_counts": dict(sorted(event_type_counts.items())),
        "explanation_policy": {
            "method": "deterministic post-processing of C2 correlation output",
            "certainty_rule": "uses likely-related wording and preserves raw C2 evidence",
            "high_confidence_min_correlation_score": C3_HIGH_CONFIDENCE_SCORE,
            "medium_confidence_min_correlation_score": C3_MEDIUM_CONFIDENCE_SCORE,
        },
    }

    with open(paths.c3_explained_anomaly_windows_file, "w", encoding="utf-8") as file:
        json.dump(explained_windows, file, indent=2)
    with open(paths.c3_explanation_summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary, explained_windows


def _print_c3_summary(summary: dict[str, Any]) -> None:
    counts = summary["explanation_counts"]
    print(
        "C3 summary: "
        f"explained={counts['explained_anomaly_window_count']} "
        f"with_correlated_event={counts['with_correlated_event']} "
        f"model_only={counts['model_only']}"
    )
    print(f"Explained anomaly windows: {summary['output_files']['explained_anomaly_windows']}")
    print(f"Explanation summary: {summary['output_files']['explanation_summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add deterministic C3 explanations to C2 correlated anomaly windows."
    )
    parser.add_argument(
        "--correlated-anomaly-windows-file",
        help="C2 correlated anomaly windows JSON file. Defaults to C2 output.",
    )
    args = parser.parse_args()

    summary, _ = explain_correlated_anomaly_windows(
        correlated_anomaly_windows_file=args.correlated_anomaly_windows_file,
        paths=ProjectPaths(),
    )
    _print_c3_summary(summary)


if __name__ == "__main__":
    main()
