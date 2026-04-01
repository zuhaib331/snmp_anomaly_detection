from __future__ import annotations

from pathlib import Path

import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths


DEFAULT_REQUIRED_COLUMNS = (
    "device_id",
    "timestamp",
    "interface",
    "in_octets",
    "out_octets",
    "errors",
)
DEFAULT_OPTIONAL_COLUMNS = (
    "cpu",
    "memory",
    "vendor",
    "device_type",
    "if_speed",
    "interface_speed",
    "interface_speed_mbps",
    "interface_admin_status",
    "interface_oper_status",
    "in_ucast_pkts",
    "out_ucast_pkts",
    "discard_in",
    "discard_out",
    "in_discards",
    "out_discards",
    "packets_in",
    "packets_out",
    "anomaly_type",
    "counter_reset",
    "anomaly",
)
DEFAULT_COUNTER_COLUMNS = (
    "in_octets",
    "out_octets",
    "errors",
    "in_ucast_pkts",
    "out_ucast_pkts",
    "in_discards",
    "out_discards",
)


def resolve_dataset_path(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> Path:
    paths = paths or ProjectPaths()
    if input_file is not None:
        dataset_path = Path(input_file)
        if not dataset_path.is_absolute():
            dataset_path = paths.repo_root / input_file
        return dataset_path

    candidates = (
        paths.dataset_file,
        paths.repo_root / "synthetic_snmp_dataset.csv",
        paths.repo_root / "featureEngineering" / "synthetic_snmp_dataset.csv",
    )
    return next((candidate for candidate in candidates if candidate.exists()), paths.dataset_file)


def load_raw_dataset(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
) -> tuple[pd.DataFrame, Path]:
    dataset_path = resolve_dataset_path(input_file=input_file, paths=paths)
    dataframe = pd.read_csv(dataset_path)
    return dataframe, dataset_path


def add_timestamp_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    enriched = dataframe.copy()
    enriched["timestamp_parsed"] = pd.to_datetime(
        enriched["timestamp"],
        errors="coerce",
    )
    return enriched


def interface_series(dataframe: pd.DataFrame) -> pd.Series:
    if "interface" not in dataframe.columns:
        return pd.Series([""] * len(dataframe), index=dataframe.index, dtype="object")
    return dataframe["interface"].fillna("").astype(str).str.strip()
