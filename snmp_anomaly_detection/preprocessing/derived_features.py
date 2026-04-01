from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = ("device_id", "interface")


def _resolve_group_keys(dataframe: pd.DataFrame) -> list[str]:
    keys = ["device_id"]
    if "interface" in dataframe.columns:
        keys.append("interface")
    return keys


def derive_rate_features_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    derived = dataframe.copy()
    derived["timestamp"] = pd.to_datetime(derived["timestamp"], errors="coerce")
    group_keys = _resolve_group_keys(derived)
    derived = derived.sort_values(group_keys + ["timestamp"]).reset_index(drop=True)

    grouped = derived.groupby(group_keys, dropna=False)
    elapsed_seconds = grouped["timestamp"].diff().dt.total_seconds()
    counter_reset_flag = (
        derived["counter_reset"].fillna(0).astype(int)
        if "counter_reset" in derived.columns
        else pd.Series(0, index=derived.index, dtype=int)
    )

    for source_column, feature_name in (
        ("in_octets", "in_rate"),
        ("out_octets", "out_rate"),
        ("errors", "error_rate"),
    ):
        deltas = grouped[source_column].diff()
        invalid_mask = (elapsed_seconds.isna()) | (elapsed_seconds <= 0) | (deltas < 0) | (
            counter_reset_flag == 1
        )
        safe_elapsed = elapsed_seconds.where(~invalid_mask)
        safe_deltas = deltas.where(~invalid_mask)
        derived[feature_name] = (safe_deltas / safe_elapsed).fillna(0.0)

    derived["cpu"] = derived["cpu"].astype(float)
    derived["memory"] = derived["memory"].astype(float)
    derived["analysis_scope"] = (
        "per_interface" if "interface" in derived.columns else "per_device"
    )
    derived["reset_detected"] = counter_reset_flag.astype(int)
    derived["elapsed_seconds"] = elapsed_seconds.fillna(0.0)
    return derived


@dataclass(frozen=True)
class OnlineDerivedRecord:
    record: dict[str, Any]


class OnlineRateFeatureBuilder:
    def __init__(self) -> None:
        self._previous_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def _build_key(self, event: Any) -> tuple[str, str]:
        interface = getattr(event, "interface", None)
        return str(event.device_id), "" if interface is None else str(interface)

    def transform(self, event: Any) -> OnlineDerivedRecord | None:
        event_time = pd.to_datetime(event.timestamp, errors="coerce")
        if pd.isna(event_time):
            return None

        key = self._build_key(event)
        previous = self._previous_by_key.get(key)
        current = {
            "timestamp": event_time,
            "device_id": str(event.device_id),
            "interface": None if key[1] == "" else key[1],
            "cpu": float(event.cpu),
            "memory": float(event.memory),
            "in_octets": float(event.in_octets),
            "out_octets": float(event.out_octets),
            "errors": float(event.errors),
            "anomaly": int(getattr(event, "anomaly", 0)),
            "analysis_scope": "per_interface" if key[1] else "per_device",
        }

        self._previous_by_key[key] = current
        if previous is None:
            return None

        elapsed_seconds = float((event_time - previous["timestamp"]).total_seconds())
        if elapsed_seconds <= 0:
            return None

        in_delta = current["in_octets"] - previous["in_octets"]
        out_delta = current["out_octets"] - previous["out_octets"]
        error_delta = current["errors"] - previous["errors"]
        if in_delta < 0 or out_delta < 0 or error_delta < 0:
            current["reset_detected"] = 1
            return None

        current["in_rate"] = in_delta / elapsed_seconds
        current["out_rate"] = out_delta / elapsed_seconds
        current["error_rate"] = error_delta / elapsed_seconds
        current["elapsed_seconds"] = elapsed_seconds
        current["reset_detected"] = 0
        return OnlineDerivedRecord(record=current)
