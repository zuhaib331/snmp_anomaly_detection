from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths
from snmp_anomaly_detection.data.p0_utils import (
    DEFAULT_COUNTER_COLUMNS,
    DEFAULT_REQUIRED_COLUMNS,
    add_timestamp_column,
    interface_series,
    load_raw_dataset,
)


def _group_keys(dataframe: pd.DataFrame) -> list[str]:
    keys = ["device_id"]
    if "interface" in dataframe.columns:
        keys.append("interface")
    return keys


def _build_group_id(group_keys: list[str], group_value: object) -> str:
    if not isinstance(group_value, tuple):
        group_value = (group_value,)
    return ", ".join(
        f"{key}={value}" for key, value in zip(group_keys, group_value, strict=False)
    )


def _expected_interval_seconds(group_frame: pd.DataFrame) -> float | None:
    deltas = (
        group_frame["timestamp_parsed"]
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
    )
    if deltas.empty:
        return None
    return float(deltas.median())


def build_data_quality_audit(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, object]:
    raw_dataframe, dataset_path = load_raw_dataset(input_file=input_file, paths=paths)
    dataframe = add_timestamp_column(raw_dataframe)
    if "interface" in dataframe.columns:
        dataframe["interface"] = interface_series(dataframe)

    duplicate_row_count = int(dataframe.duplicated().sum())
    duplicate_key_count = 0
    key_columns = [
        column for column in ("device_id", "interface", "timestamp") if column in dataframe.columns
    ]
    if key_columns:
        duplicate_key_count = int(dataframe.duplicated(subset=key_columns).sum())

    required_missing_value_counts = {
        column: int(dataframe[column].isna().sum()) if column in dataframe.columns else None
        for column in DEFAULT_REQUIRED_COLUMNS
    }

    counter_negative_deltas = {counter: 0 for counter in DEFAULT_COUNTER_COLUMNS if counter in dataframe.columns}
    out_of_order_group_count = 0
    groups_with_gaps = 0
    groups_with_irregular_intervals = 0
    sample_group_issues: list[dict[str, object]] = []

    group_keys = _group_keys(dataframe)
    grouped = dataframe.groupby(group_keys, dropna=False) if all(
        key in dataframe.columns for key in group_keys
    ) else []

    for group_value, group_frame in grouped:
        group_id = _build_group_id(group_keys, group_value)
        issues: list[str] = []

        original_order = group_frame["timestamp_parsed"]
        original_deltas = original_order.diff().dt.total_seconds()
        if (original_deltas.dropna() < 0).any():
            out_of_order_group_count += 1
            issues.append("out_of_order_timestamps")

        sorted_frame = group_frame.sort_values("timestamp_parsed").copy()
        interval_seconds = (
            sorted_frame["timestamp_parsed"].diff().dt.total_seconds().dropna()
        )
        expected_interval = _expected_interval_seconds(sorted_frame)
        if expected_interval and not interval_seconds.empty:
            tolerance = max(1.0, expected_interval * 0.1)
            irregular_count = int((interval_seconds.sub(expected_interval).abs() > tolerance).sum())
            gap_count = int((interval_seconds > expected_interval * 1.5).sum())
            if irregular_count > 0:
                groups_with_irregular_intervals += 1
                issues.append(f"irregular_intervals={irregular_count}")
            if gap_count > 0:
                groups_with_gaps += 1
                issues.append(f"polling_gaps={gap_count}")

        for counter_name in counter_negative_deltas:
            negative_deltas = int((sorted_frame[counter_name].diff().dropna() < 0).sum())
            counter_negative_deltas[counter_name] += negative_deltas
            if negative_deltas > 0:
                issues.append(f"negative_{counter_name}_deltas={negative_deltas}")

        if issues and len(sample_group_issues) < 20:
            sample_group_issues.append(
                {
                    "group": group_id,
                    "row_count": int(len(group_frame)),
                    "issues": issues,
                }
            )

    audit = {
        "dataset_path": str(dataset_path),
        "summary": {
            "row_count": int(len(dataframe)),
            "duplicate_row_count": duplicate_row_count,
            "duplicate_key_count": duplicate_key_count,
            "timestamp_parse_failures": int(dataframe["timestamp_parsed"].isna().sum()),
            "out_of_order_group_count": out_of_order_group_count,
            "groups_with_polling_gaps": groups_with_gaps,
            "groups_with_irregular_intervals": groups_with_irregular_intervals,
        },
        "required_missing_value_counts": required_missing_value_counts,
        "counter_negative_delta_counts": counter_negative_deltas,
        "sample_group_issues": sample_group_issues,
    }
    return audit


def _print_audit(audit: dict[str, object]) -> None:
    summary = audit["summary"]
    print(f"Dataset: {audit['dataset_path']}")
    print(
        "Quality summary: "
        f"rows={summary['row_count']} "
        f"duplicate_rows={summary['duplicate_row_count']} "
        f"duplicate_keys={summary['duplicate_key_count']} "
        f"timestamp_parse_failures={summary['timestamp_parse_failures']} "
        f"out_of_order_groups={summary['out_of_order_group_count']} "
        f"groups_with_gaps={summary['groups_with_polling_gaps']} "
        f"groups_with_irregular_intervals={summary['groups_with_irregular_intervals']}"
    )
    print(
        "Negative deltas: "
        + ", ".join(
            f"{counter}={count}"
            for counter, count in audit["counter_negative_delta_counts"].items()
        )
    )
    missing_counts = audit["required_missing_value_counts"]
    print(
        "Missing required values: "
        + ", ".join(f"{column}={count}" for column, count in missing_counts.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a P0 data quality audit report.")
    parser.add_argument(
        "--input-file",
        help="Optional dataset path. Defaults to the configured dataset lookup order.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional JSON file path to save the audit report.",
    )
    args = parser.parse_args()

    audit = build_data_quality_audit(input_file=args.input_file)
    _print_audit(audit)

    if args.output_file:
        output_path = Path(args.output_file)
        if not output_path.is_absolute():
            output_path = ProjectPaths().repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(audit, file, indent=2)
        print(f"Saved audit report to: {output_path}")


if __name__ == "__main__":
    main()
