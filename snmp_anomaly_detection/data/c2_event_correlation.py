from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths


C2_LOOKBACK_MINUTES = 10
C2_LOOKAHEAD_MINUTES = 5
C2_MAX_EVENTS_PER_ANOMALY = 5
C2_SEVERITY_RANK: dict[str, int] = {
    "critical": 30,
    "warning": 20,
    "info": 10,
}
C2_FEATURE_EVENT_HINTS: dict[str, set[str]] = {
    "error_rate": {"high_error_rate", "interface_down", "oper_down", "link_flap"},
    "discard_rate_in": {"high_error_rate", "interface_down", "oper_down", "link_flap"},
    "discard_rate_out": {"high_error_rate", "interface_down", "oper_down", "link_flap"},
    "interface_oper_status": {"interface_down", "interface_up", "oper_down", "link_flap"},
    "interface_admin_status": {"admin_down", "interface_down", "interface_up"},
    "burst_indicator": {"high_error_rate", "link_flap", "interface_down"},
    "in_rate": {"interface_down", "interface_up", "link_flap"},
    "out_rate": {"interface_down", "interface_up", "link_flap"},
    "utilization_in_pct": {"interface_down", "interface_up", "link_flap"},
    "utilization_out_pct": {"interface_down", "interface_up", "link_flap"},
}


def _resolve_path(path_value: str | None, default_path: Path, paths: ProjectPaths) -> Path:
    path = Path(path_value) if path_value else default_path
    if not path.is_absolute():
        path = paths.repo_root / path
    return path


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}.")
    return [record for record in payload if isinstance(record, dict)]


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                records.append(payload)
    return records


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


def _feature_related_event_bonus(top_error_feature: Any, event_type: str) -> int:
    base_feature = _feature_base(top_error_feature)
    related_events = C2_FEATURE_EVENT_HINTS.get(base_feature, set())
    return 20 if event_type in related_events else 0


def _minutes_between(event_time: pd.Timestamp, anchor_time: pd.Timestamp) -> float:
    return float((event_time - anchor_time).total_seconds() / 60.0)


def _match_key_score(anomaly: dict[str, Any], event: dict[str, Any]) -> tuple[int, str] | None:
    anomaly_device = _clean_string(anomaly.get("device_id"))
    event_device = _clean_string(event.get("device_id"))
    if not anomaly_device or anomaly_device != event_device:
        return None

    anomaly_interface = _clean_string(anomaly.get("interface"))
    event_interface = _clean_string(event.get("interface"))
    event_scope = _clean_string(event.get("analysis_scope"))

    if anomaly_interface and event_interface and anomaly_interface == event_interface:
        return 100, "device_id+interface"
    if not event_interface or event_scope == "per_device":
        return 60, "device_id"
    return None


def _proximity_score(minutes_from_window_end: float) -> int:
    max_window_minutes = max(C2_LOOKBACK_MINUTES, C2_LOOKAHEAD_MINUTES)
    distance = abs(minutes_from_window_end)
    score = 20 * max(0.0, 1.0 - (distance / max_window_minutes))
    return int(round(score))


def _build_correlated_event(
    anomaly: dict[str, Any],
    event: dict[str, Any],
    window_end: pd.Timestamp,
) -> dict[str, Any] | None:
    event_time = pd.to_datetime(event.get("timestamp"), errors="coerce")
    if pd.isna(event_time):
        return None

    minutes_from_window_end = _minutes_between(event_time, window_end)
    if minutes_from_window_end < -C2_LOOKBACK_MINUTES or minutes_from_window_end > C2_LOOKAHEAD_MINUTES:
        return None

    key_score = _match_key_score(anomaly, event)
    if key_score is None:
        return None

    match_score, match_scope = key_score
    event_type = _clean_string(event.get("event_type")) or "unknown_event"
    severity = _clean_string(event.get("severity")) or "warning"
    severity_score = C2_SEVERITY_RANK.get(severity, 0)
    feature_bonus = _feature_related_event_bonus(anomaly.get("top_error_feature"), event_type)
    proximity_score = _proximity_score(minutes_from_window_end)
    correlation_score = match_score + severity_score + feature_bonus + proximity_score

    reasons = [
        f"matched by {match_scope}",
        f"event_time_delta_minutes={minutes_from_window_end:.2f}",
        f"severity={severity}",
    ]
    if feature_bonus:
        reasons.append(f"event_type relates to top_error_feature={anomaly.get('top_error_feature')}")

    return {
        "timestamp": event.get("timestamp"),
        "device_id": event.get("device_id"),
        "interface": event.get("interface"),
        "analysis_scope": event.get("analysis_scope"),
        "source_type": event.get("source_type"),
        "event_type": event_type,
        "severity": severity,
        "message": event.get("message"),
        "minutes_from_window_end": minutes_from_window_end,
        "match_scope": match_scope,
        "correlation_score": correlation_score,
        "correlation_reasons": reasons,
        "extra": event.get("extra", {}),
    }


def _index_events_by_match_key(
    normalized_events: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    interface_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    device_index: dict[str, list[dict[str, Any]]] = {}

    for event in normalized_events:
        device_id = _clean_string(event.get("device_id"))
        if not device_id:
            continue
        interface = _clean_string(event.get("interface"))
        if interface:
            interface_index.setdefault((device_id, interface), []).append(event)
        else:
            device_index.setdefault(device_id, []).append(event)

    return interface_index, device_index


def _candidate_events_for_anomaly(
    anomaly: dict[str, Any],
    interface_index: dict[tuple[str, str], list[dict[str, Any]]],
    device_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    device_id = _clean_string(anomaly.get("device_id"))
    interface = _clean_string(anomaly.get("interface"))
    candidates: list[dict[str, Any]] = []
    if device_id and interface:
        candidates.extend(interface_index.get((device_id, interface), []))
    if device_id:
        candidates.extend(device_index.get(device_id, []))
    return candidates


def correlate_anomaly_windows(
    anomaly_windows_file: str | None = None,
    normalized_events_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    anomaly_path = _resolve_path(anomaly_windows_file, paths.anomaly_windows_file, paths)
    events_path = _resolve_path(normalized_events_file, paths.c1_normalized_events_file, paths)

    anomaly_windows = _read_json_file(anomaly_path)
    normalized_events = _read_jsonl_file(events_path)
    interface_event_index, device_event_index = _index_events_by_match_key(normalized_events)
    correlated_windows: list[dict[str, Any]] = []

    unmatched_count = 0
    matched_count = 0
    total_matches = 0
    event_type_counts: Counter[str] = Counter()
    match_scope_counts: Counter[str] = Counter()

    for anomaly in anomaly_windows:
        window_end = pd.to_datetime(anomaly.get("window_end"), errors="coerce")
        correlated_events: list[dict[str, Any]] = []
        if not pd.isna(window_end):
            for event in _candidate_events_for_anomaly(
                anomaly,
                interface_event_index,
                device_event_index,
            ):
                correlated_event = _build_correlated_event(anomaly, event, window_end)
                if correlated_event is not None:
                    correlated_events.append(correlated_event)

        correlated_events.sort(
            key=lambda event: (
                -int(event["correlation_score"]),
                abs(float(event["minutes_from_window_end"])),
                str(event["timestamp"]),
                str(event["event_type"]),
            )
        )
        selected_events = correlated_events[:C2_MAX_EVENTS_PER_ANOMALY]
        best_event = selected_events[0] if selected_events else None
        has_correlated_event = best_event is not None

        if has_correlated_event:
            matched_count += 1
        else:
            unmatched_count += 1

        total_matches += len(selected_events)
        event_type_counts.update(event["event_type"] for event in selected_events)
        match_scope_counts.update(event["match_scope"] for event in selected_events)

        enriched = dict(anomaly)
        enriched["c2_correlation"] = {
            "has_correlated_event": has_correlated_event,
            "matched_event_count": len(selected_events),
            "best_event": best_event,
            "correlated_events": selected_events,
            "correlation_window": {
                "anchor": "window_end",
                "lookback_minutes": C2_LOOKBACK_MINUTES,
                "lookahead_minutes": C2_LOOKAHEAD_MINUTES,
            },
            "ranking_policy": [
                "prefer device_id+interface over device_id-only",
                "prefer closer event timestamps",
                "prefer higher severity",
                "prefer event types related to top_error_feature",
            ],
        }
        correlated_windows.append(enriched)

    summary: dict[str, Any] = {
        "stage": "C2",
        "status": "completed",
        "anomaly_windows_file": str(anomaly_path),
        "normalized_events_file": str(events_path),
        "output_files": {
            "correlated_anomaly_windows": str(paths.c2_correlated_anomaly_windows_file),
            "correlation_summary": str(paths.c2_correlation_summary_file),
        },
        "correlation_window": {
            "anchor": "window_end",
            "lookback_minutes": C2_LOOKBACK_MINUTES,
            "lookahead_minutes": C2_LOOKAHEAD_MINUTES,
        },
        "ranking_policy": {
            "same_device_and_interface_score": 100,
            "same_device_only_score": 60,
            "severity_scores": C2_SEVERITY_RANK,
            "feature_related_event_bonus": 20,
            "max_proximity_score": 20,
            "max_events_per_anomaly": C2_MAX_EVENTS_PER_ANOMALY,
        },
        "input_counts": {
            "anomaly_window_count": len(anomaly_windows),
            "normalized_event_count": len(normalized_events),
            "interface_event_key_count": len(interface_event_index),
            "device_only_event_key_count": len(device_event_index),
        },
        "correlation_counts": {
            "anomalies_with_correlated_events": matched_count,
            "anomalies_without_correlated_events": unmatched_count,
            "selected_correlated_event_count": total_matches,
        },
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "match_scope_counts": dict(sorted(match_scope_counts.items())),
    }

    with open(paths.c2_correlated_anomaly_windows_file, "w", encoding="utf-8") as file:
        json.dump(correlated_windows, file, indent=2)
    with open(paths.c2_correlation_summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary, correlated_windows


def _print_c2_summary(summary: dict[str, Any]) -> None:
    counts = summary["correlation_counts"]
    print(
        "C2 summary: "
        f"anomalies={summary['input_counts']['anomaly_window_count']} "
        f"events={summary['input_counts']['normalized_event_count']} "
        f"matched={counts['anomalies_with_correlated_events']} "
        f"unmatched={counts['anomalies_without_correlated_events']}"
    )
    print(f"Correlated anomaly windows: {summary['output_files']['correlated_anomaly_windows']}")
    print(f"Correlation summary: {summary['output_files']['correlation_summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate anomaly windows with normalized C1 events.")
    parser.add_argument(
        "--anomaly-windows-file",
        help="Anomaly windows JSON file. Defaults to snmp_anomaly_detection/outputs/anomaly_windows.json.",
    )
    parser.add_argument(
        "--normalized-events-file",
        help="Normalized events JSONL file. Defaults to C1 normalized events output.",
    )
    args = parser.parse_args()

    summary, _ = correlate_anomaly_windows(
        anomaly_windows_file=args.anomaly_windows_file,
        normalized_events_file=args.normalized_events_file,
        paths=ProjectPaths(),
    )
    _print_c2_summary(summary)


if __name__ == "__main__":
    main()
