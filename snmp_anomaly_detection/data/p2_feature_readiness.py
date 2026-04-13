from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from snmp_anomaly_detection.config import FeatureEngineeringConfig, ProjectPaths
from snmp_anomaly_detection.data.p0_utils import (
    add_timestamp_column,
    interface_series,
    load_raw_dataset,
)
from snmp_anomaly_detection.inference.events import NormalizedEvent
from snmp_anomaly_detection.streaming.kafka_source import REQUIRED_KAFKA_FIELDS


P2_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "interface_speed": ("interface_speed_mbps", "interface_speed", "if_speed"),
    "packet_counter_in": ("in_ucast_pkts", "packets_in"),
    "packet_counter_out": ("out_ucast_pkts", "packets_out"),
    "discard_counter_in": ("in_discards", "discard_in"),
    "discard_counter_out": ("out_discards", "discard_out"),
    "interface_admin_status": ("interface_admin_status",),
    "interface_oper_status": ("interface_oper_status",),
}

P2_REQUIRED_CAPABILITIES: tuple[str, ...] = tuple(P2_SOURCE_ALIASES.keys())


def _resolve_selected_column(
    dataframe: pd.DataFrame,
    aliases: tuple[str, ...],
) -> str | None:
    for column_name in aliases:
        if column_name in dataframe.columns:
            return column_name
    return None


def _safe_bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _column_summary(dataframe: pd.DataFrame, aliases: tuple[str, ...]) -> dict[str, object]:
    available_aliases = [column_name for column_name in aliases if column_name in dataframe.columns]
    selected_column = _resolve_selected_column(dataframe, aliases)
    if selected_column is None:
        return {
            "selected_column": None,
            "available_aliases": available_aliases,
            "column_present": False,
            "non_null_rows": 0,
            "coverage_ratio": 0.0,
            "positive_rows": 0,
            "positive_ratio": 0.0,
            "unique_non_null_values": 0,
        }

    series = dataframe[selected_column]
    numeric_series = pd.to_numeric(series, errors="coerce")
    non_null_rows = int(series.notna().sum())
    positive_rows = int((numeric_series > 0).sum()) if numeric_series.notna().any() else 0
    row_count = len(dataframe)

    return {
        "selected_column": selected_column,
        "available_aliases": available_aliases,
        "column_present": True,
        "non_null_rows": non_null_rows,
        "coverage_ratio": float(non_null_rows / row_count) if row_count else 0.0,
        "positive_rows": positive_rows,
        "positive_ratio": float(positive_rows / row_count) if row_count else 0.0,
        "unique_non_null_values": int(series.dropna().nunique()),
    }


def _series_ready(series: pd.Series, require_positive: bool = False) -> bool:
    if series.isna().any():
        return False
    if require_positive:
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.isna().any():
            return False
        return _safe_bool((numeric_series > 0).all())
    return True


def _capability_flags(group_frame: pd.DataFrame) -> tuple[dict[str, bool], dict[str, str | None]]:
    flags: dict[str, bool] = {}
    source_columns: dict[str, str | None] = {}

    for capability_name, aliases in P2_SOURCE_ALIASES.items():
        selected_column = _resolve_selected_column(group_frame, aliases)
        source_columns[capability_name] = selected_column
        if selected_column is None:
            flags[capability_name] = False
            continue

        require_positive = capability_name == "interface_speed"
        flags[capability_name] = _series_ready(
            group_frame[selected_column],
            require_positive=require_positive,
        )

    return flags, source_columns


def build_interface_capability_matrix(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, object]:
    paths = paths or ProjectPaths()
    raw_dataframe, dataset_path = load_raw_dataset(input_file=input_file, paths=paths)
    dataframe = add_timestamp_column(raw_dataframe)
    dataframe["interface_normalized"] = interface_series(dataframe)

    for capability_name, aliases in P2_SOURCE_ALIASES.items():
        selected_column = _resolve_selected_column(dataframe, aliases)
        if selected_column is None:
            continue
        if capability_name == "interface_speed":
            dataframe[selected_column] = pd.to_numeric(dataframe[selected_column], errors="coerce")

    feature_source_summary = {
        capability_name: _column_summary(dataframe, aliases)
        for capability_name, aliases in P2_SOURCE_ALIASES.items()
    }

    grouped = dataframe.groupby(["device_id", "interface_normalized"], dropna=False)
    stream_records: list[dict[str, object]] = []
    full_feature_stream_count = 0
    fallback_stream_count = 0

    for (device_id, interface_name), group_frame in grouped:
        capability_flags, source_columns = _capability_flags(group_frame)
        missing_capabilities = [
            capability_name
            for capability_name in P2_REQUIRED_CAPABILITIES
            if not capability_flags[capability_name]
        ]
        has_named_interface = interface_name not in {"", "nan", "None"}
        full_feature_coverage = has_named_interface and not missing_capabilities
        recommended_scope = "per_interface" if full_feature_coverage else "per_device"
        if full_feature_coverage:
            full_feature_stream_count += 1
        else:
            fallback_stream_count += 1

        stream_records.append(
            {
                "device_id": str(device_id),
                "interface": str(interface_name),
                "stream_id": f"{device_id}::{interface_name}",
                "row_count": int(len(group_frame)),
                "timestamp_min": str(group_frame["timestamp_parsed"].min()),
                "timestamp_max": str(group_frame["timestamp_parsed"].max()),
                "has_named_interface": has_named_interface,
                "recommended_scope": recommended_scope,
                "full_per_interface_feature_coverage": full_feature_coverage,
                "missing_capabilities": missing_capabilities,
                "capabilities": capability_flags,
                "source_columns": source_columns,
            }
        )

    device_records: list[dict[str, object]] = []
    for device_id, device_frame in dataframe.groupby("device_id", dropna=False):
        device_streams = [
            record for record in stream_records if record["device_id"] == str(device_id)
        ]
        full_stream_count = sum(
            1 for record in device_streams if record["full_per_interface_feature_coverage"]
        )
        fallback_streams = [
            record["interface"]
            for record in device_streams
            if not record["full_per_interface_feature_coverage"]
        ]
        recommended_scope = (
            "per_interface" if full_stream_count == len(device_streams) and device_streams else "per_device"
        )
        device_records.append(
            {
                "device_id": str(device_id),
                "interface_count": int(len(device_streams)),
                "fully_covered_interface_count": int(full_stream_count),
                "fallback_interfaces": fallback_streams,
                "recommended_scope": recommended_scope,
            }
        )

    return {
        "stage": "P2",
        "dataset_path": str(dataset_path),
        "analysis_scope_policy": {
            "primary_scope": "per_interface",
            "fallback_scope": "per_device",
        },
        "summary": {
            "row_count": int(len(dataframe)),
            "device_count": int(dataframe["device_id"].nunique()) if "device_id" in dataframe.columns else 0,
            "stream_count": int(len(stream_records)),
            "full_feature_stream_count": int(full_feature_stream_count),
            "fallback_stream_count": int(fallback_stream_count),
            "full_feature_device_count": int(
                sum(1 for record in device_records if record["recommended_scope"] == "per_interface")
            ),
            "fallback_device_count": int(
                sum(1 for record in device_records if record["recommended_scope"] == "per_device")
            ),
        },
        "feature_source_summary": feature_source_summary,
        "device_scope_recommendations": device_records,
        "stream_capability_matrix": stream_records,
    }


def build_extended_feature_schema_note(
    capability_matrix: dict[str, object],
) -> dict[str, object]:
    baseline_features = list(FeatureEngineeringConfig().feature_columns)
    current_live_fields = sorted(NormalizedEvent.__dataclass_fields__.keys())
    current_kafka_required_fields = list(REQUIRED_KAFKA_FIELDS)

    schema_fields = [
        {
            "field_name": "interface_speed_mbps",
            "category": "capacity_source",
            "dtype": "float",
            "required_for_features": ["utilization_in_pct", "utilization_out_pct"],
            "offline_aliases": list(P2_SOURCE_ALIASES["interface_speed"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "interface_speed_mbps" in current_live_fields,
            "notes": "Must be positive for safe utilization calculations.",
        },
        {
            "field_name": "in_ucast_pkts",
            "category": "packet_counter_source",
            "dtype": "float",
            "required_for_features": ["packet_rate_in"],
            "offline_aliases": list(P2_SOURCE_ALIASES["packet_counter_in"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "in_ucast_pkts" in current_live_fields,
            "notes": "Cumulative counter; derive rate per device_id + interface.",
        },
        {
            "field_name": "out_ucast_pkts",
            "category": "packet_counter_source",
            "dtype": "float",
            "required_for_features": ["packet_rate_out"],
            "offline_aliases": list(P2_SOURCE_ALIASES["packet_counter_out"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "out_ucast_pkts" in current_live_fields,
            "notes": "Cumulative counter; derive rate per device_id + interface.",
        },
        {
            "field_name": "in_discards",
            "category": "discard_counter_source",
            "dtype": "float",
            "required_for_features": ["discard_rate_in"],
            "offline_aliases": list(P2_SOURCE_ALIASES["discard_counter_in"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "in_discards" in current_live_fields,
            "notes": "Cumulative counter; derive rate per device_id + interface.",
        },
        {
            "field_name": "out_discards",
            "category": "discard_counter_source",
            "dtype": "float",
            "required_for_features": ["discard_rate_out"],
            "offline_aliases": list(P2_SOURCE_ALIASES["discard_counter_out"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "out_discards" in current_live_fields,
            "notes": "Cumulative counter; derive rate per device_id + interface.",
        },
        {
            "field_name": "interface_admin_status",
            "category": "state_source",
            "dtype": "int",
            "required_for_features": ["interface_admin_status"],
            "offline_aliases": list(P2_SOURCE_ALIASES["interface_admin_status"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "interface_admin_status" in current_live_fields,
            "notes": "Preserve as raw state value for explanation and optional training use.",
        },
        {
            "field_name": "interface_oper_status",
            "category": "state_source",
            "dtype": "int",
            "required_for_features": ["interface_oper_status"],
            "offline_aliases": list(P2_SOURCE_ALIASES["interface_oper_status"]),
            "required_in_live_event": True,
            "currently_in_live_event_schema": "interface_oper_status" in current_live_fields,
            "notes": "Preserve as raw state value for explanation and optional training use.",
        },
        {
            "field_name": "utilization_in_pct",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "100 * (in_rate * 8) / (interface_speed_mbps * 1_000_000)",
            "required_for_features": ["utilization_in_pct"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Derived after rate generation; clamp or null-handle when speed is unavailable.",
        },
        {
            "field_name": "utilization_out_pct",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "100 * (out_rate * 8) / (interface_speed_mbps * 1_000_000)",
            "required_for_features": ["utilization_out_pct"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Derived after rate generation; clamp or null-handle when speed is unavailable.",
        },
        {
            "field_name": "packet_rate_in",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "delta(in_ucast_pkts) / elapsed_seconds",
            "required_for_features": ["packet_rate_in"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Reset-aware cumulative-counter rate.",
        },
        {
            "field_name": "packet_rate_out",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "delta(out_ucast_pkts) / elapsed_seconds",
            "required_for_features": ["packet_rate_out"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Reset-aware cumulative-counter rate.",
        },
        {
            "field_name": "discard_rate_in",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "delta(in_discards) / elapsed_seconds",
            "required_for_features": ["discard_rate_in"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Reset-aware cumulative-counter rate.",
        },
        {
            "field_name": "discard_rate_out",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "delta(out_discards) / elapsed_seconds",
            "required_for_features": ["discard_rate_out"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Reset-aware cumulative-counter rate.",
        },
        {
            "field_name": "in_out_ratio",
            "category": "derived_f2_feature",
            "dtype": "float",
            "formula": "in_rate / max(out_rate, epsilon)",
            "required_for_features": ["in_out_ratio"],
            "required_in_live_event": False,
            "currently_in_live_event_schema": False,
            "notes": "Useful for asymmetry detection; should be epsilon-safe.",
        },
    ]

    live_gap_fields = [
        field["field_name"]
        for field in schema_fields
        if field["required_in_live_event"] and not field["currently_in_live_event_schema"]
    ]

    return {
        "stage": "P2",
        "goal": "Define the safe extended interface-level schema needed before F2 and F3.",
        "current_validated_baseline_features": baseline_features,
        "recommended_f2_candidate_features": [
            "cpu",
            "memory",
            "in_rate",
            "out_rate",
            "error_rate",
            "utilization_in_pct",
            "utilization_out_pct",
            "discard_rate_in",
            "discard_rate_out",
            "packet_rate_in",
            "packet_rate_out",
            "in_out_ratio",
            "interface_oper_status",
            "interface_admin_status",
        ],
        "current_live_input_schema": {
            "normalized_event_fields": current_live_fields,
            "kafka_required_fields": current_kafka_required_fields,
        },
        "required_live_input_additions_for_f2": live_gap_fields,
        "extended_interface_row_schema": schema_fields,
        "future_output_expectations": {
            "result_records_should_preserve": [
                "device_id",
                "interface",
                "stream_id",
                "analysis_scope",
                "top_error_feature",
                "feature_error_<feature_name>",
            ],
            "anomaly_window_records_should_preserve": [
                "window_records[].interface_speed_mbps",
                "window_records[].interface_admin_status",
                "window_records[].interface_oper_status",
                "window_records[].packet_rate_in",
                "window_records[].packet_rate_out",
                "window_records[].discard_rate_in",
                "window_records[].discard_rate_out",
                "window_records[].utilization_in_pct",
                "window_records[].utilization_out_pct",
                "window_records[].in_out_ratio",
            ],
        },
        "implementation_notes": [
            "Offline and live paths must derive the same F2 features from the same source columns.",
            "Per-interface remains the default path; use per-device only when the required interface-level fields are missing.",
            "Current synthetic data has full source-field coverage for every stream in the baseline dataset.",
            "Current live event and Kafka payload schemas now include the source fields needed for F2 feature derivation.",
        ],
        "capability_summary": capability_matrix["summary"],
    }


def run_p2_feature_readiness(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    paths = paths or ProjectPaths()
    paths.ensure_directories()

    capability_matrix = build_interface_capability_matrix(input_file=input_file, paths=paths)
    schema_note = build_extended_feature_schema_note(capability_matrix)

    with open(paths.p2_interface_capability_matrix_file, "w", encoding="utf-8") as file:
        json.dump(capability_matrix, file, indent=2)
    with open(paths.p2_extended_feature_schema_file, "w", encoding="utf-8") as file:
        json.dump(schema_note, file, indent=2)

    return capability_matrix, schema_note


def _print_p2_summary(
    capability_matrix: dict[str, object],
    schema_note: dict[str, object],
    paths: ProjectPaths,
) -> None:
    summary = capability_matrix["summary"]
    print(
        "P2 summary: "
        f"streams={summary['stream_count']} "
        f"full_feature_streams={summary['full_feature_stream_count']} "
        f"fallback_streams={summary['fallback_stream_count']} "
        f"full_feature_devices={summary['full_feature_device_count']} "
        f"fallback_devices={summary['fallback_device_count']}"
    )
    print(f"Capability matrix: {paths.p2_interface_capability_matrix_file}")
    print(f"Extended feature schema: {paths.p2_extended_feature_schema_file}")
    live_additions = schema_note["required_live_input_additions_for_f2"]
    print(
        "Live schema additions required for F2: "
        + (", ".join(live_additions) if live_additions else "none")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the P2 interface capability matrix and schema note.")
    parser.add_argument(
        "--input-file",
        help="Optional dataset path. Defaults to the configured dataset lookup order.",
    )
    parser.add_argument(
        "--capability-output-file",
        help="Optional JSON file path to override the default capability matrix output.",
    )
    parser.add_argument(
        "--schema-output-file",
        help="Optional JSON file path to override the default extended schema output.",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    capability_matrix, schema_note = run_p2_feature_readiness(
        input_file=args.input_file,
        paths=paths,
    )

    if args.capability_output_file:
        capability_output_path = Path(args.capability_output_file)
        if not capability_output_path.is_absolute():
            capability_output_path = paths.repo_root / capability_output_path
        capability_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(capability_output_path, "w", encoding="utf-8") as file:
            json.dump(capability_matrix, file, indent=2)

    if args.schema_output_file:
        schema_output_path = Path(args.schema_output_file)
        if not schema_output_path.is_absolute():
            schema_output_path = paths.repo_root / args.schema_output_file
        schema_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_output_path, "w", encoding="utf-8") as file:
            json.dump(schema_note, file, indent=2)

    _print_p2_summary(capability_matrix, schema_note, paths)


if __name__ == "__main__":
    main()
