from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterable

import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths
from snmp_anomaly_detection.data.p0_utils import (
    add_timestamp_column,
    interface_series,
    load_raw_dataset,
)


P3_SCHEMA_VERSION = "p3_normalized_event_v1"
P3_SOURCE_TYPE = "snmp_trap_interface_state"
P3_SYNTHETIC_SOURCE_TYPE = "synthetic_snmp_trap_interface_state"
P3_REQUIRED_SOURCE_FIELDS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "interface",
    "interface_admin_status",
    "interface_oper_status",
)
P3_OPTIONAL_SOURCE_FIELDS: tuple[str, ...] = (
    "anomaly_type",
    "counter_reset",
    "errors",
)
P3_EVENT_TYPES: tuple[str, ...] = (
    "interface_down",
    "interface_up",
    "admin_down",
    "oper_down",
    "link_flap",
    "counter_reset",
    "device_reboot",
    "high_error_rate",
    "unknown_event",
)


def _clean_interface(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value).strip()
    return text or None


def _analysis_scope(interface_name: str | None) -> str:
    return "per_interface" if interface_name else "per_device"


def _safe_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return default
    return int(numeric)


def _event_record(
    *,
    timestamp: pd.Timestamp,
    device_id: object,
    interface_name: str | None,
    event_type: str,
    severity: str,
    message: str,
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        "timestamp": timestamp.isoformat(sep=" "),
        "device_id": str(device_id),
        "interface": interface_name,
        "analysis_scope": _analysis_scope(interface_name),
        "source_type": P3_SYNTHETIC_SOURCE_TYPE,
        "event_type": event_type,
        "severity": severity,
        "message": message,
        "extra": extra,
    }


def _source_field_coverage(dataframe: pd.DataFrame) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = {}
    row_count = len(dataframe)
    for field_name in (*P3_REQUIRED_SOURCE_FIELDS, *P3_OPTIONAL_SOURCE_FIELDS):
        present = field_name in dataframe.columns
        non_null_rows = int(dataframe[field_name].notna().sum()) if present else 0
        coverage[field_name] = {
            "present": present,
            "required_for_p3": field_name in P3_REQUIRED_SOURCE_FIELDS,
            "non_null_rows": non_null_rows,
            "coverage_ratio": float(non_null_rows / row_count) if row_count else 0.0,
        }
    return coverage


def build_event_source_selection(
    dataframe: pd.DataFrame,
    dataset_path: Path,
) -> dict[str, object]:
    field_coverage = _source_field_coverage(dataframe)
    missing_required_fields = [
        field_name
        for field_name in P3_REQUIRED_SOURCE_FIELDS
        if not field_coverage[field_name]["present"]
    ]
    interface_values = interface_series(dataframe) if "interface" in dataframe.columns else pd.Series(dtype="object")

    return {
        "stage": "P3",
        "status": "completed",
        "selected_event_source": {
            "source_type": P3_SOURCE_TYPE,
            "first_implementation_source_type": P3_SYNTHETIC_SOURCE_TYPE,
            "data_mode": "synthetic_from_existing_snmp_dataset",
            "dataset_path": str(dataset_path),
            "reason": (
                "SNMP traps and interface state events naturally preserve device_id, "
                "often preserve interface, and explain link/state/reset/high-error causes."
            ),
        },
        "accepted_real_source_formats_for_c1": ["csv", "jsonl"],
        "supported_standard_event_types": list(P3_EVENT_TYPES),
        "source_field_coverage": field_coverage,
        "readiness": {
            "usable_for_c1": not missing_required_fields,
            "missing_required_fields": missing_required_fields,
            "device_id_available": "device_id" in dataframe.columns,
            "interface_identifier_available": bool(len(interface_values) and interface_values.ne("").any()),
            "timestamp_available": "timestamp" in dataframe.columns,
        },
        "next_step": "Use this source selection and schema to implement C1 normalization.",
    }


def build_normalized_event_schema() -> dict[str, object]:
    return {
        "stage": "P3",
        "schema_version": P3_SCHEMA_VERSION,
        "status": "completed",
        "event_schema": [
            {
                "field_name": "timestamp",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "notes": "ISO-like timestamp parsed with the same pandas timestamp parser used for SNMP rows.",
            },
            {
                "field_name": "device_id",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "notes": "Primary device matching key.",
            },
            {
                "field_name": "interface",
                "dtype": "string",
                "required": False,
                "nullable": True,
                "notes": "Preferred second matching key when present.",
            },
            {
                "field_name": "analysis_scope",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "allowed_values": ["per_interface", "per_device"],
                "notes": "Preserves whether correlation should use device+interface or device-only matching.",
            },
            {
                "field_name": "source_type",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "allowed_values": [P3_SOURCE_TYPE, P3_SYNTHETIC_SOURCE_TYPE],
                "notes": "Identifies the external evidence family.",
            },
            {
                "field_name": "event_type",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "allowed_values": list(P3_EVENT_TYPES),
                "notes": "Standard event type for deterministic downstream correlation.",
            },
            {
                "field_name": "severity",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "allowed_values": ["info", "warning", "critical"],
                "notes": "Normalized operational severity.",
            },
            {
                "field_name": "message",
                "dtype": "string",
                "required": True,
                "nullable": False,
                "notes": "Human-readable explanation of the normalized event.",
            },
            {
                "field_name": "extra",
                "dtype": "object",
                "required": True,
                "nullable": False,
                "notes": "Source-specific fields retained for auditability.",
            },
        ],
        "matching_key_policy": {
            "preferred": ["device_id", "interface"],
            "fallback": ["device_id"],
            "fallback_condition": "Use device-only matching when interface is missing or empty.",
        },
        "timestamp_policy": {
            "parser": "pandas.to_datetime(errors='coerce')",
            "timezone_assumption": "same timezone as the SNMP dataset unless the event source explicitly provides one",
            "malformed_or_missing_timestamp_action": "reject during C1 normalization",
        },
        "draft_c2_correlation_window": {
            "lookback_minutes_before_anomaly_window_end": 10,
            "lookahead_minutes_after_anomaly_window_end": 5,
        },
    }


def iter_sample_normalized_events(dataframe: pd.DataFrame) -> Iterable[dict[str, object]]:
    sorted_frame = dataframe.sort_values(["device_id", "interface_normalized", "timestamp_parsed"])
    grouped = sorted_frame.groupby(["device_id", "interface_normalized"], dropna=False)

    for (_, _), group_frame in grouped:
        previous_oper_status: int | None = None
        previous_admin_status: int | None = None

        for _, row in group_frame.iterrows():
            timestamp = row["timestamp_parsed"]
            if pd.isna(timestamp):
                continue

            device_id = row.get("device_id", "")
            interface_name = _clean_interface(row.get("interface_normalized"))
            anomaly_type = str(row.get("anomaly_type", "normal") or "normal")
            admin_status = _safe_int(row.get("interface_admin_status"), default=1)
            oper_status = _safe_int(row.get("interface_oper_status"), default=1)
            counter_reset = _safe_int(row.get("counter_reset"), default=0)

            if admin_status != 1:
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="admin_down",
                    severity="critical",
                    message=f"Administrative state is down on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "interface_admin_status": admin_status,
                        "anomaly_type": anomaly_type,
                    },
                )
            elif previous_admin_status is not None and previous_admin_status != 1 and admin_status == 1:
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="interface_up",
                    severity="info",
                    message=f"Administrative state returned up on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "previous_interface_admin_status": previous_admin_status,
                        "interface_admin_status": admin_status,
                    },
                )

            if anomaly_type == "link_down" or oper_status != 1:
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="interface_down",
                    severity="critical",
                    message=f"Interface operational state is down on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "interface_oper_status": oper_status,
                        "anomaly_type": anomaly_type,
                    },
                )
            elif previous_oper_status is not None and previous_oper_status != 1 and oper_status == 1:
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="interface_up",
                    severity="info",
                    message=f"Interface operational state returned up on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "previous_interface_oper_status": previous_oper_status,
                        "interface_oper_status": oper_status,
                    },
                )

            if counter_reset == 1:
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="counter_reset",
                    severity="warning",
                    message=f"Counter reset detected on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "counter_reset": counter_reset,
                    },
                )

            if anomaly_type == "error_burst":
                yield _event_record(
                    timestamp=timestamp,
                    device_id=device_id,
                    interface_name=interface_name,
                    event_type="high_error_rate",
                    severity="warning",
                    message=f"High error-rate event detected on {device_id} {interface_name or '<device>'}.",
                    extra={
                        "source_row_index": int(row.name),
                        "errors": _safe_int(row.get("errors"), default=0),
                        "anomaly_type": anomaly_type,
                    },
                )

            previous_admin_status = admin_status
            previous_oper_status = oper_status


def build_timestamp_alignment_notes(
    dataframe: pd.DataFrame,
    sample_events: list[dict[str, object]],
    dataset_path: Path,
) -> dict[str, object]:
    valid_timestamps = dataframe["timestamp_parsed"].dropna()
    sample_event_timestamps = pd.to_datetime(
        [event["timestamp"] for event in sample_events],
        errors="coerce",
    )
    snmp_timestamp_values = set(valid_timestamps.astype("int64"))
    aligned_event_count = int(
        sum(
            int(timestamp.value) in snmp_timestamp_values
            for timestamp in sample_event_timestamps.dropna()
        )
    )

    poll_deltas = (
        valid_timestamps.sort_values().drop_duplicates().diff().dropna().dt.total_seconds()
    )
    common_poll_interval_seconds = (
        float(poll_deltas.mode().iloc[0]) if not poll_deltas.empty else None
    )

    interface_present_count = sum(1 for event in sample_events if event["interface"])
    return {
        "stage": "P3",
        "status": "completed",
        "dataset_path": str(dataset_path),
        "timestamp_parser": "pandas.to_datetime(errors='coerce')",
        "timezone_assumption": "same timezone as the SNMP dataset; synthetic source timestamps are naive like the SNMP rows",
        "snmp_timestamp_quality": {
            "row_count": int(len(dataframe)),
            "valid_timestamp_rows": int(valid_timestamps.shape[0]),
            "invalid_timestamp_rows": int(dataframe["timestamp_parsed"].isna().sum()),
            "timestamp_min": str(valid_timestamps.min()) if not valid_timestamps.empty else None,
            "timestamp_max": str(valid_timestamps.max()) if not valid_timestamps.empty else None,
            "common_poll_interval_seconds": common_poll_interval_seconds,
        },
        "sample_event_timestamp_quality": {
            "sample_event_count": int(len(sample_events)),
            "valid_event_timestamp_count": int(sample_event_timestamps.notna().sum()),
            "aligned_to_snmp_poll_timestamp_count": aligned_event_count,
            "alignment_ratio": float(aligned_event_count / len(sample_events)) if sample_events else 0.0,
        },
        "matching_key_quality": {
            "events_with_device_id": int(sum(1 for event in sample_events if event["device_id"])),
            "events_with_interface": int(interface_present_count),
            "per_interface_event_ratio": float(interface_present_count / len(sample_events)) if sample_events else 0.0,
            "fallback_policy": "Use device_id only when interface is missing.",
        },
        "c1_rejection_policy_preview": {
            "missing_or_malformed_timestamp": "reject",
            "missing_device_id": "reject",
            "missing_interface": "keep with analysis_scope=per_device",
            "unknown_event_type": "map to unknown_event",
        },
    }


def run_p3_event_readiness(
    input_file: str | None = None,
    max_sample_events: int = 100,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    raw_dataframe, dataset_path = load_raw_dataset(input_file=input_file, paths=paths)
    dataframe = add_timestamp_column(raw_dataframe)
    dataframe["interface_normalized"] = interface_series(dataframe)

    source_selection = build_event_source_selection(dataframe, dataset_path)
    normalized_schema = build_normalized_event_schema()
    sample_events = list(islice(iter_sample_normalized_events(dataframe), max_sample_events))
    timestamp_alignment_notes = build_timestamp_alignment_notes(
        dataframe,
        sample_events,
        dataset_path,
    )

    with open(paths.p3_event_source_selection_file, "w", encoding="utf-8") as file:
        json.dump(source_selection, file, indent=2)
    with open(paths.p3_normalized_event_schema_file, "w", encoding="utf-8") as file:
        json.dump(normalized_schema, file, indent=2)
    with open(paths.p3_timestamp_alignment_notes_file, "w", encoding="utf-8") as file:
        json.dump(timestamp_alignment_notes, file, indent=2)
    with open(paths.p3_sample_normalized_events_file, "w", encoding="utf-8") as file:
        for event in sample_events:
            file.write(json.dumps(event, sort_keys=True) + "\n")

    return source_selection, normalized_schema, timestamp_alignment_notes, sample_events


def _print_p3_summary(
    source_selection: dict[str, object],
    timestamp_alignment_notes: dict[str, object],
    sample_events: list[dict[str, object]],
    paths: ProjectPaths,
) -> None:
    readiness = source_selection["readiness"]
    alignment = timestamp_alignment_notes["sample_event_timestamp_quality"]
    print(
        "P3 summary: "
        f"usable_for_c1={readiness['usable_for_c1']} "
        f"sample_events={len(sample_events)} "
        f"alignment_ratio={alignment['alignment_ratio']:.3f}"
    )
    print(f"Event source selection: {paths.p3_event_source_selection_file}")
    print(f"Normalized event schema: {paths.p3_normalized_event_schema_file}")
    print(f"Timestamp alignment notes: {paths.p3_timestamp_alignment_notes_file}")
    print(f"Sample normalized events: {paths.p3_sample_normalized_events_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P3 event-source readiness outputs.")
    parser.add_argument(
        "--input-file",
        help="Optional dataset path. Defaults to the configured dataset lookup order.",
    )
    parser.add_argument(
        "--max-sample-events",
        type=int,
        default=100,
        help="Maximum synthetic normalized events to write to the sample JSONL output.",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    source_selection, _, timestamp_alignment_notes, sample_events = run_p3_event_readiness(
        input_file=args.input_file,
        max_sample_events=max(args.max_sample_events, 0),
        paths=paths,
    )
    _print_p3_summary(source_selection, timestamp_alignment_notes, sample_events, paths)


if __name__ == "__main__":
    main()
