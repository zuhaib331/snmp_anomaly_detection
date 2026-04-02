from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from snmp_anomaly_detection.config import EvaluationConfig


@dataclass(frozen=True)
class TimeSplitDefinition:
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    total_unique_timestamps: int


def _to_iso8601(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat(sep=" ")


def build_time_split_definition(
    dataframe: pd.DataFrame,
    config: EvaluationConfig,
) -> TimeSplitDefinition:
    if round(
        config.train_fraction + config.validation_fraction + config.test_fraction,
        10,
    ) != 1.0:
        raise ValueError("EvaluationConfig fractions must sum to 1.0")

    timestamps = pd.Index(pd.to_datetime(dataframe["timestamp"])).sort_values().unique()
    if len(timestamps) < 3:
        raise ValueError("At least 3 unique timestamps are required to build a P1 time split.")

    train_end_index = max(int(len(timestamps) * config.train_fraction) - 1, 0)
    validation_end_index = max(
        int(len(timestamps) * (config.train_fraction + config.validation_fraction)) - 1,
        train_end_index + 1,
    )
    validation_end_index = min(validation_end_index, len(timestamps) - 2)
    test_start_index = validation_end_index + 1

    train_start = timestamps[0]
    train_end = timestamps[train_end_index]
    validation_start = timestamps[train_end_index + 1]
    validation_end = timestamps[validation_end_index]
    test_start = timestamps[test_start_index]
    test_end = timestamps[-1]

    return TimeSplitDefinition(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        train_start=_to_iso8601(train_start),
        train_end=_to_iso8601(train_end),
        validation_start=_to_iso8601(validation_start),
        validation_end=_to_iso8601(validation_end),
        test_start=_to_iso8601(test_start),
        test_end=_to_iso8601(test_end),
        total_unique_timestamps=len(timestamps),
    )


def label_timestamp(timestamp: pd.Timestamp, split_definition: TimeSplitDefinition) -> str:
    timestamp = pd.Timestamp(timestamp)
    if timestamp <= pd.Timestamp(split_definition.train_end):
        return "train"
    if timestamp <= pd.Timestamp(split_definition.validation_end):
        return "validation"
    return "test"


def apply_time_split_labels(
    dataframe: pd.DataFrame,
    split_definition: TimeSplitDefinition,
) -> pd.DataFrame:
    labeled = dataframe.copy()
    labeled["split"] = labeled["timestamp"].apply(
        lambda timestamp: label_timestamp(pd.Timestamp(timestamp), split_definition)
    )
    return labeled
