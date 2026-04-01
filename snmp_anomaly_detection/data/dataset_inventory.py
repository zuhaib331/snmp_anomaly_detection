from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from snmp_anomaly_detection.config import ProjectPaths
from snmp_anomaly_detection.data.p0_utils import (
    DEFAULT_OPTIONAL_COLUMNS,
    DEFAULT_REQUIRED_COLUMNS,
    add_timestamp_column,
    interface_series,
    load_raw_dataset,
)


@dataclass(frozen=True)
class DeviceInventoryRecord:
    device_id: str
    row_count: int
    interface_count: int
    missing_interface_rows: int
    has_timestamp_gaps: bool
    recommended_scope: str


def build_inventory(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> dict[str, object]:
    raw_dataframe, dataset_path = load_raw_dataset(input_file=input_file, paths=paths)
    dataframe = add_timestamp_column(raw_dataframe)
    interfaces = interface_series(dataframe)
    dataframe = dataframe.assign(interface_normalized=interfaces)

    missing_required = [
        column for column in DEFAULT_REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    available_optional = [
        column for column in DEFAULT_OPTIONAL_COLUMNS if column in dataframe.columns
    ]

    device_records: list[DeviceInventoryRecord] = []
    per_interface_ready = 0
    per_device_fallback = 0

    if {"device_id", "timestamp_parsed"}.issubset(dataframe.columns):
        grouped = dataframe.groupby("device_id", dropna=False)
    else:
        grouped = []

    for device_id, device_frame in grouped:
        normalized_interfaces = (
            device_frame["interface_normalized"]
            if "interface_normalized" in device_frame.columns
            else []
        )
        non_empty_interfaces = sorted(
            {
                interface_name
                for interface_name in normalized_interfaces
                if interface_name not in {"", "nan", "None"}
            }
        )
        missing_interface_rows = int(
            normalized_interfaces.isin({"", "nan", "None"}).sum()
        )
        sorted_frame = device_frame.sort_values("timestamp_parsed")
        has_timestamp_gaps = bool(sorted_frame["timestamp_parsed"].isna().any())
        recommended_scope = (
            "per_interface"
            if len(non_empty_interfaces) > 0 and missing_interface_rows == 0
            else "per_device"
        )
        if recommended_scope == "per_interface":
            per_interface_ready += 1
        else:
            per_device_fallback += 1

        device_records.append(
            DeviceInventoryRecord(
                device_id=str(device_id),
                row_count=int(len(device_frame)),
                interface_count=len(non_empty_interfaces),
                missing_interface_rows=missing_interface_rows,
                has_timestamp_gaps=has_timestamp_gaps,
                recommended_scope=recommended_scope,
            )
        )

    timestamp_min = None
    timestamp_max = None
    if "timestamp_parsed" in dataframe.columns and not dataframe["timestamp_parsed"].dropna().empty:
        timestamp_min = str(dataframe["timestamp_parsed"].min())
        timestamp_max = str(dataframe["timestamp_parsed"].max())

    inventory = {
        "dataset_path": str(dataset_path),
        "summary": {
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns) - int("timestamp_parsed" in dataframe.columns) - int("interface_normalized" in dataframe.columns)),
            "device_count": int(dataframe["device_id"].nunique()) if "device_id" in dataframe.columns else 0,
            "interface_count": int(
                len(
                    {
                        interface_name
                        for interface_name in interfaces.tolist()
                        if interface_name not in {"", "nan", "None"}
                    }
                )
            ),
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "per_interface_ready_devices": per_interface_ready,
            "per_device_fallback_devices": per_device_fallback,
        },
        "columns": {
            "available": [column for column in raw_dataframe.columns.tolist()],
            "missing_required": missing_required,
            "available_optional": available_optional,
        },
        "feature_readiness": {
            "can_derive_in_rate": "in_octets" in dataframe.columns,
            "can_derive_out_rate": "out_octets" in dataframe.columns,
            "can_derive_error_rate": "errors" in dataframe.columns,
        },
        "device_inventory": [asdict(record) for record in device_records],
    }
    return inventory


def _print_inventory(inventory: dict[str, object]) -> None:
    summary = inventory["summary"]
    columns = inventory["columns"]
    readiness = inventory["feature_readiness"]

    print(f"Dataset: {inventory['dataset_path']}")
    print(
        "Summary: "
        f"rows={summary['row_count']} "
        f"devices={summary['device_count']} "
        f"interfaces={summary['interface_count']} "
        f"per_interface_ready={summary['per_interface_ready_devices']} "
        f"per_device_fallback={summary['per_device_fallback_devices']}"
    )
    print(
        "Time range: "
        f"{summary['timestamp_min']} -> {summary['timestamp_max']}"
    )
    print(f"Available columns: {', '.join(columns['available'])}")
    if columns["missing_required"]:
        print(f"Missing required columns: {', '.join(columns['missing_required'])}")
    else:
        print("Missing required columns: none")
    print(
        "Rate readiness: "
        f"in_rate={readiness['can_derive_in_rate']} "
        f"out_rate={readiness['can_derive_out_rate']} "
        f"error_rate={readiness['can_derive_error_rate']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a P0 dataset inventory report.")
    parser.add_argument(
        "--input-file",
        help="Optional dataset path. Defaults to the configured dataset lookup order.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional JSON file path to save the inventory report.",
    )
    args = parser.parse_args()

    inventory = build_inventory(input_file=args.input_file)
    _print_inventory(inventory)

    if args.output_file:
        output_path = Path(args.output_file)
        if not output_path.is_absolute():
            output_path = ProjectPaths().repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(inventory, file, indent=2)
        print(f"Saved inventory report to: {output_path}")


if __name__ == "__main__":
    main()
