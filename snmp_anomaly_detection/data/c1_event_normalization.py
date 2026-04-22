from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths
from snmp_anomaly_detection.data.p3_event_readiness import (
    P3_EVENT_TYPES,
    P3_SCHEMA_VERSION,
    P3_SOURCE_TYPE,
    P3_SYNTHETIC_SOURCE_TYPE,
)


C1_SCHEMA_VERSION = P3_SCHEMA_VERSION
C1_DEFAULT_SOURCE_TYPE = P3_SOURCE_TYPE
C1_ALLOWED_SEVERITIES: tuple[str, ...] = ("info", "warning", "critical")
C1_EVENT_TYPE_ALIASES: dict[str, str] = {
    "interface_down": "interface_down",
    "link_down": "interface_down",
    "ifdown": "interface_down",
    "if_down": "interface_down",
    "operstatusdown": "interface_down",
    "oper_down": "oper_down",
    "interface_up": "interface_up",
    "link_up": "interface_up",
    "ifup": "interface_up",
    "if_up": "interface_up",
    "operstatusup": "interface_up",
    "admin_down": "admin_down",
    "administratively_down": "admin_down",
    "shutdown": "admin_down",
    "link_flap": "link_flap",
    "flap": "link_flap",
    "counter_reset": "counter_reset",
    "counterreset": "counter_reset",
    "device_reboot": "device_reboot",
    "warmstart": "device_reboot",
    "warm_start": "device_reboot",
    "coldstart": "device_reboot",
    "cold_start": "device_reboot",
    "high_error_rate": "high_error_rate",
    "error_burst": "high_error_rate",
    "errors_high": "high_error_rate",
    "unknown_event": "unknown_event",
}
C1_SEVERITY_BY_EVENT_TYPE: dict[str, str] = {
    "interface_down": "critical",
    "admin_down": "critical",
    "oper_down": "critical",
    "link_flap": "warning",
    "counter_reset": "warning",
    "device_reboot": "warning",
    "high_error_rate": "warning",
    "interface_up": "info",
    "unknown_event": "warning",
}


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _first_present(row: dict[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in row and _clean_string(row[alias]):
            return row[alias]
    return None


def _normalize_key(value: Any) -> str:
    text = _clean_string(value)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return text.lower().replace("-", "_").replace(" ", "_").replace(".", "_")


def _normalize_event_type(value: Any, row: dict[str, Any]) -> str:
    explicit_type = C1_EVENT_TYPE_ALIASES.get(_normalize_key(value))
    if explicit_type:
        return explicit_type

    anomaly_type = C1_EVENT_TYPE_ALIASES.get(_normalize_key(row.get("anomaly_type")))
    if anomaly_type:
        return anomaly_type

    trap_name = C1_EVENT_TYPE_ALIASES.get(_normalize_key(row.get("trap_name")))
    if trap_name:
        return trap_name

    oper_status = _clean_string(row.get("interface_oper_status") or row.get("oper_status"))
    admin_status = _clean_string(row.get("interface_admin_status") or row.get("admin_status"))
    counter_reset = _clean_string(row.get("counter_reset"))

    if admin_status and admin_status not in {"1", "up"}:
        return "admin_down"
    if oper_status and oper_status not in {"1", "up"}:
        return "interface_down"
    if counter_reset in {"1", "true", "True"}:
        return "counter_reset"
    return "unknown_event"


def _normalize_severity(value: Any, event_type: str) -> str:
    severity = _normalize_key(value)
    if severity in C1_ALLOWED_SEVERITIES:
        return severity
    return C1_SEVERITY_BY_EVENT_TYPE.get(event_type, "warning")


def _parse_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = _clean_string(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_extra": text}
    return parsed if isinstance(parsed, dict) else {"raw_extra": parsed}


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if pd.isna(value) if isinstance(value, float) else False:
        return None
    return value


def _read_jsonl(input_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open(input_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "__c1_read_error": f"invalid_json: {exc.msg}",
                    "__raw_line": stripped,
                }
                continue
            if not isinstance(payload, dict):
                yield line_number, {
                    "__c1_read_error": "jsonl_record_is_not_an_object",
                    "__raw_line": payload,
                }
                continue
            yield line_number, payload


def _read_csv(input_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with open(input_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line_number, row in enumerate(reader, start=2):
            yield line_number, dict(row)


def read_event_rows(input_file: str | None, paths: ProjectPaths) -> tuple[Path, list[tuple[int, dict[str, Any]]]]:
    input_path = Path(input_file) if input_file else paths.p3_sample_normalized_events_file
    if not input_path.is_absolute():
        input_path = paths.repo_root / input_path

    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return input_path, list(_read_jsonl(input_path))
    if suffix == ".json":
        with open(input_path, encoding="utf-8") as file:
            payload = json.load(file)
        rows = payload if isinstance(payload, list) else payload.get("events", [])
        normalized_rows: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(rows, start=1):
            if isinstance(row, dict):
                normalized_rows.append((index, row))
            else:
                normalized_rows.append(
                    (
                        index,
                        {
                            "__c1_read_error": "json_record_is_not_an_object",
                            "__raw_line": row,
                        },
                    )
                )
        return input_path, normalized_rows
    if suffix == ".csv":
        return input_path, list(_read_csv(input_path))
    raise ValueError(f"Unsupported event input format: {input_path.suffix}. Expected CSV, JSON, or JSONL.")


def normalize_event_row(row: dict[str, Any], source_line: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    timestamp_raw = _first_present(row, ("timestamp", "event_time", "time", "created_at"))
    timestamp = pd.to_datetime(timestamp_raw, errors="coerce")
    device_id = _clean_string(_first_present(row, ("device_id", "device", "hostname", "host", "node")))

    if pd.isna(timestamp):
        return None, {
            "source_line": source_line,
            "reason": "missing_or_malformed_timestamp",
            "raw_event": row,
        }
    if not device_id:
        return None, {
            "source_line": source_line,
            "reason": "missing_device_id",
            "raw_event": row,
        }

    interface = _clean_string(_first_present(row, ("interface", "if_name", "ifName", "ifDescr", "port"))) or None
    event_type = _normalize_event_type(
        _first_present(row, ("event_type", "trap_type", "trap_name", "name", "event_name", "anomaly_type")),
        row,
    )
    if event_type not in P3_EVENT_TYPES:
        event_type = "unknown_event"

    source_type = _clean_string(row.get("source_type")) or C1_DEFAULT_SOURCE_TYPE
    if source_type == P3_SYNTHETIC_SOURCE_TYPE:
        source_type = P3_SOURCE_TYPE
    severity = _normalize_severity(row.get("severity"), event_type)
    message = _clean_string(row.get("message")) or f"{event_type} on {device_id} {interface or '<device>'}."
    extra = _parse_extra(row.get("extra"))
    extra.update(
        {
            "c1_source_line": source_line,
            "c1_original_event_type": _clean_string(
                _first_present(row, ("event_type", "trap_type", "trap_name", "name", "event_name", "anomaly_type"))
            )
            or None,
        }
    )

    normalized = {
        "timestamp": timestamp.isoformat(sep=" "),
        "device_id": device_id,
        "interface": interface,
        "analysis_scope": "per_interface" if interface else "per_device",
        "source_type": source_type,
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "extra": {key: _json_safe(value) for key, value in extra.items()},
    }
    return normalized, None


def normalize_events(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    input_path, rows = read_event_rows(input_file, paths)
    normalized_events: list[dict[str, Any]] = []
    rejected_events: list[dict[str, Any]] = []

    for source_line, row in rows:
        if "__c1_read_error" in row:
            rejected_events.append(
                {
                    "source_line": source_line,
                    "reason": row["__c1_read_error"],
                    "raw_event": row.get("__raw_line"),
                }
            )
            continue
        normalized_event, rejected_event = normalize_event_row(row, source_line)
        if normalized_event is not None:
            normalized_events.append(normalized_event)
        if rejected_event is not None:
            rejected_events.append(rejected_event)

    event_type_counts = Counter(event["event_type"] for event in normalized_events)
    severity_counts = Counter(event["severity"] for event in normalized_events)
    analysis_scope_counts = Counter(event["analysis_scope"] for event in normalized_events)
    rejection_counts = Counter(event["reason"] for event in rejected_events)

    summary: dict[str, Any] = {
        "stage": "C1",
        "schema_version": C1_SCHEMA_VERSION,
        "status": "completed",
        "input_file": str(input_path),
        "output_files": {
            "normalized_events": str(paths.c1_normalized_events_file),
            "rejected_events": str(paths.c1_rejected_events_file),
            "normalization_summary": str(paths.c1_normalization_summary_file),
        },
        "row_count": len(rows),
        "normalized_event_count": len(normalized_events),
        "rejected_event_count": len(rejected_events),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "analysis_scope_counts": dict(sorted(analysis_scope_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "deterministic_mapping": {
            "accepted_input_formats": ["csv", "json", "jsonl"],
            "standard_event_types": list(P3_EVENT_TYPES),
            "severity_by_event_type": C1_SEVERITY_BY_EVENT_TYPE,
            "missing_interface_action": "keep with analysis_scope=per_device",
            "unknown_event_action": "map to unknown_event",
        },
    }

    with open(paths.c1_normalized_events_file, "w", encoding="utf-8") as file:
        for event in normalized_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    with open(paths.c1_rejected_events_file, "w", encoding="utf-8") as file:
        for event in rejected_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")
    with open(paths.c1_normalization_summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary, normalized_events, rejected_events


def _print_c1_summary(summary: dict[str, Any]) -> None:
    print(
        "C1 summary: "
        f"rows={summary['row_count']} "
        f"normalized={summary['normalized_event_count']} "
        f"rejected={summary['rejected_event_count']}"
    )
    print(f"Normalized events: {summary['output_files']['normalized_events']}")
    print(f"Rejected events: {summary['output_files']['rejected_events']}")
    print(f"Normalization summary: {summary['output_files']['normalization_summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize C1 event samples into the P3 shared schema.")
    parser.add_argument(
        "--input-file",
        help="CSV, JSON, or JSONL event input. Defaults to the P3 sample normalized events file.",
    )
    args = parser.parse_args()

    summary, _, _ = normalize_events(input_file=args.input_file, paths=ProjectPaths())
    _print_c1_summary(summary)


if __name__ == "__main__":
    main()
