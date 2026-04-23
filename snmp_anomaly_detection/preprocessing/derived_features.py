from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


GROUP_KEYS = ("device_id", "interface")
EPSILON = 1e-9
STATUS_DEFAULT_UP = 1.0
ROLLING_CONTEXT_WINDOW = 12
F3_CONTEXT_BASE_FEATURES = (
    "in_rate",
    "out_rate",
    "error_rate",
    "utilization_in_pct",
    "utilization_out_pct",
)


def _resolve_group_keys(dataframe: pd.DataFrame) -> list[str]:
    keys = ["device_id"]
    if "interface" in dataframe.columns:
        keys.append("interface")
    return keys


def _resolve_numeric_series(
    dataframe: pd.DataFrame,
    aliases: tuple[str, ...],
    default_value: float | None = None,
) -> pd.Series:
    for column_name in aliases:
        if column_name in dataframe.columns:
            return pd.to_numeric(dataframe[column_name], errors="coerce")
    if default_value is None:
        return pd.Series(np.nan, index=dataframe.index, dtype=float)
    return pd.Series(default_value, index=dataframe.index, dtype=float)


def _derive_counter_rate(
    grouped,
    source_series: pd.Series,
    elapsed_seconds: pd.Series,
    counter_reset_flag: pd.Series,
) -> pd.Series:
    deltas = grouped[source_series.name].diff() if source_series.name in grouped.obj.columns else source_series.groupby(grouped.grouper).diff()
    invalid_mask = (
        elapsed_seconds.isna()
        | (elapsed_seconds <= 0)
        | source_series.isna()
        | deltas.isna()
        | (deltas < 0)
        | (counter_reset_flag == 1)
    )
    safe_elapsed = elapsed_seconds.where(~invalid_mask)
    safe_deltas = deltas.where(~invalid_mask)
    return (safe_deltas / safe_elapsed).fillna(0.0)


def _derive_counter_rate_from_column(
    dataframe: pd.DataFrame,
    grouped,
    source_column: str,
    elapsed_seconds: pd.Series,
    counter_reset_flag: pd.Series,
) -> pd.Series:
    source_series = pd.to_numeric(dataframe[source_column], errors="coerce")
    deltas = grouped[source_column].diff()
    invalid_mask = (
        elapsed_seconds.isna()
        | (elapsed_seconds <= 0)
        | source_series.isna()
        | deltas.isna()
        | (deltas < 0)
        | (counter_reset_flag == 1)
    )
    safe_elapsed = elapsed_seconds.where(~invalid_mask)
    safe_deltas = deltas.where(~invalid_mask)
    return (safe_deltas / safe_elapsed).fillna(0.0)


def _derive_utilization(rate_series: pd.Series, interface_speed_mbps: pd.Series) -> pd.Series:
    speed_bps = interface_speed_mbps * 1_000_000.0
    valid_speed = speed_bps.where(speed_bps > 0)
    return ((rate_series * 8.0) / valid_speed * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_contextual_rolling_features(
    dataframe: pd.DataFrame,
    group_keys: list[str],
    rolling_window: int = ROLLING_CONTEXT_WINDOW,
) -> pd.DataFrame:
    enriched = dataframe.copy()
    grouped = enriched.groupby(group_keys, dropna=False)

    for feature_name in F3_CONTEXT_BASE_FEATURES:
        history = grouped[feature_name].shift(1)
        rolling = history.groupby([enriched[key] for key in group_keys], dropna=False)
        rolling_mean = rolling.transform(
            lambda values: values.rolling(rolling_window, min_periods=2).mean()
        )
        rolling_std = rolling.transform(
            lambda values: values.rolling(rolling_window, min_periods=2).std()
        )
        previous_value = grouped[feature_name].shift(1)
        safe_std = rolling_std.where(rolling_std > EPSILON)

        enriched[f"{feature_name}_rolling_mean"] = rolling_mean.fillna(0.0)
        enriched[f"{feature_name}_rolling_std"] = rolling_std.fillna(0.0)
        enriched[f"{feature_name}_zscore"] = (
            ((enriched[feature_name] - rolling_mean) / safe_std)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        enriched[f"{feature_name}_trend"] = (
            enriched[feature_name] - previous_value
        ).fillna(0.0)

    zscore_columns = [f"{feature_name}_zscore" for feature_name in F3_CONTEXT_BASE_FEATURES]
    enriched["burst_indicator"] = (
        enriched[zscore_columns].abs().max(axis=1) >= 3.0
    ).astype(float)
    return enriched


def derive_rate_features_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    derived = dataframe.copy()
    derived["timestamp"] = pd.to_datetime(derived["timestamp"], errors="coerce")
    group_keys = _resolve_group_keys(derived)
    derived = derived.sort_values(group_keys + ["timestamp"]).reset_index(drop=True)

    grouped = derived.groupby(group_keys, dropna=False)
    elapsed_seconds = grouped["timestamp"].diff().dt.total_seconds()
    counter_reset_flag = (
        pd.to_numeric(derived["counter_reset"], errors="coerce").fillna(0).astype(int)
        if "counter_reset" in derived.columns
        else pd.Series(0, index=derived.index, dtype=int)
    )

    base_counter_columns = (
        ("in_octets", "in_rate"),
        ("out_octets", "out_rate"),
        ("errors", "error_rate"),
    )
    for source_column, feature_name in base_counter_columns:
        derived[feature_name] = _derive_counter_rate_from_column(
            dataframe=derived,
            grouped=grouped,
            source_column=source_column,
            elapsed_seconds=elapsed_seconds,
            counter_reset_flag=counter_reset_flag,
        )

    packet_in_column = next(
        (column_name for column_name in ("in_ucast_pkts", "packets_in") if column_name in derived.columns),
        None,
    )
    packet_out_column = next(
        (column_name for column_name in ("out_ucast_pkts", "packets_out") if column_name in derived.columns),
        None,
    )
    discard_in_column = next(
        (column_name for column_name in ("in_discards", "discard_in") if column_name in derived.columns),
        None,
    )
    discard_out_column = next(
        (column_name for column_name in ("out_discards", "discard_out") if column_name in derived.columns),
        None,
    )

    derived["packet_rate_in"] = (
        _derive_counter_rate_from_column(
            dataframe=derived,
            grouped=grouped,
            source_column=packet_in_column,
            elapsed_seconds=elapsed_seconds,
            counter_reset_flag=counter_reset_flag,
        )
        if packet_in_column is not None
        else pd.Series(0.0, index=derived.index, dtype=float)
    )
    derived["packet_rate_out"] = (
        _derive_counter_rate_from_column(
            dataframe=derived,
            grouped=grouped,
            source_column=packet_out_column,
            elapsed_seconds=elapsed_seconds,
            counter_reset_flag=counter_reset_flag,
        )
        if packet_out_column is not None
        else pd.Series(0.0, index=derived.index, dtype=float)
    )
    derived["discard_rate_in"] = (
        _derive_counter_rate_from_column(
            dataframe=derived,
            grouped=grouped,
            source_column=discard_in_column,
            elapsed_seconds=elapsed_seconds,
            counter_reset_flag=counter_reset_flag,
        )
        if discard_in_column is not None
        else pd.Series(0.0, index=derived.index, dtype=float)
    )
    derived["discard_rate_out"] = (
        _derive_counter_rate_from_column(
            dataframe=derived,
            grouped=grouped,
            source_column=discard_out_column,
            elapsed_seconds=elapsed_seconds,
            counter_reset_flag=counter_reset_flag,
        )
        if discard_out_column is not None
        else pd.Series(0.0, index=derived.index, dtype=float)
    )

    interface_speed_mbps = _resolve_numeric_series(
        derived,
        ("interface_speed_mbps", "interface_speed", "if_speed"),
        default_value=0.0,
    )
    derived["utilization_in_pct"] = _derive_utilization(derived["in_rate"], interface_speed_mbps)
    derived["utilization_out_pct"] = _derive_utilization(derived["out_rate"], interface_speed_mbps)
    derived["in_out_ratio"] = (
        derived["in_rate"] / np.maximum(derived["out_rate"].to_numpy(dtype=float), EPSILON)
    )
    derived["interface_speed_mbps"] = interface_speed_mbps.fillna(0.0)
    derived["interface_admin_status"] = _resolve_numeric_series(
        derived,
        ("interface_admin_status",),
        default_value=STATUS_DEFAULT_UP,
    ).fillna(STATUS_DEFAULT_UP)
    derived["interface_oper_status"] = _resolve_numeric_series(
        derived,
        ("interface_oper_status",),
        default_value=STATUS_DEFAULT_UP,
    ).fillna(STATUS_DEFAULT_UP)
    derived["cpu"] = pd.to_numeric(derived["cpu"], errors="coerce").fillna(0.0)
    derived["memory"] = pd.to_numeric(derived["memory"], errors="coerce").fillna(0.0)
    derived["analysis_scope"] = (
        "per_interface" if "interface" in derived.columns else "per_device"
    )
    derived["reset_detected"] = counter_reset_flag.astype(int)
    derived["elapsed_seconds"] = elapsed_seconds.fillna(0.0)
    derived = _add_contextual_rolling_features(derived, group_keys=group_keys)
    return derived


@dataclass(frozen=True)
class OnlineDerivedRecord:
    record: dict[str, Any]


class OnlineRateFeatureBuilder:
    def __init__(self) -> None:
        self._previous_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        self._history_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _build_key(self, event: Any) -> tuple[str, str]:
        interface = getattr(event, "interface", None)
        return str(event.device_id), "" if interface is None else str(interface)

    def transform(self, event: Any) -> OnlineDerivedRecord | None:
        event_time = pd.to_datetime(event.timestamp, errors="coerce")
        if pd.isna(event_time):
            return None

        key = self._build_key(event)
        current = {
            "timestamp": event_time,
            "device_id": str(event.device_id),
            "interface": None if key[1] == "" else key[1],
            "cpu": float(event.cpu),
            "memory": float(event.memory),
            "in_octets": float(event.in_octets),
            "out_octets": float(event.out_octets),
            "errors": float(event.errors),
            "in_ucast_pkts": None if getattr(event, "in_ucast_pkts", None) is None else float(event.in_ucast_pkts),
            "out_ucast_pkts": None if getattr(event, "out_ucast_pkts", None) is None else float(event.out_ucast_pkts),
            "in_discards": None if getattr(event, "in_discards", None) is None else float(event.in_discards),
            "out_discards": None if getattr(event, "out_discards", None) is None else float(event.out_discards),
            "interface_speed_mbps": None
            if getattr(event, "interface_speed_mbps", None) is None
            else float(event.interface_speed_mbps),
            "interface_admin_status": float(
                getattr(event, "interface_admin_status", STATUS_DEFAULT_UP)
                if getattr(event, "interface_admin_status", None) is not None
                else STATUS_DEFAULT_UP
            ),
            "interface_oper_status": float(
                getattr(event, "interface_oper_status", STATUS_DEFAULT_UP)
                if getattr(event, "interface_oper_status", None) is not None
                else STATUS_DEFAULT_UP
            ),
            "counter_reset": int(getattr(event, "counter_reset", 0) or 0),
            "anomaly": int(getattr(event, "anomaly", 0)),
            "analysis_scope": "per_interface" if key[1] else "per_device",
        }

        previous = self._previous_by_key.get(key)
        self._previous_by_key[key] = current
        if previous is None:
            return None

        elapsed_seconds = float((event_time - previous["timestamp"]).total_seconds())
        if elapsed_seconds <= 0:
            return None

        if current["counter_reset"] == 1:
            current["reset_detected"] = 1
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

        for source_name, feature_name in (
            ("in_ucast_pkts", "packet_rate_in"),
            ("out_ucast_pkts", "packet_rate_out"),
            ("in_discards", "discard_rate_in"),
            ("out_discards", "discard_rate_out"),
        ):
            current_value = current[source_name]
            previous_value = previous.get(source_name)
            if current_value is None or previous_value is None:
                current[feature_name] = 0.0
                continue
            delta = current_value - previous_value
            if delta < 0:
                current["reset_detected"] = 1
                return None
            current[feature_name] = delta / elapsed_seconds

        interface_speed_mbps = current["interface_speed_mbps"] or 0.0
        if interface_speed_mbps > 0:
            current["utilization_in_pct"] = (current["in_rate"] * 8.0) / (
                interface_speed_mbps * 1_000_000.0
            ) * 100.0
            current["utilization_out_pct"] = (current["out_rate"] * 8.0) / (
                interface_speed_mbps * 1_000_000.0
            ) * 100.0
        else:
            current["utilization_in_pct"] = 0.0
            current["utilization_out_pct"] = 0.0
        current["in_out_ratio"] = current["in_rate"] / max(current["out_rate"], EPSILON)
        current["elapsed_seconds"] = elapsed_seconds
        current["reset_detected"] = 0
        history = self._history_by_key.setdefault(key, [])
        self._add_online_contextual_features(current, history)
        history.append(current.copy())
        if len(history) > ROLLING_CONTEXT_WINDOW:
            del history[:-ROLLING_CONTEXT_WINDOW]
        return OnlineDerivedRecord(record=current)

    def _add_online_contextual_features(
        self,
        current: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        for feature_name in F3_CONTEXT_BASE_FEATURES:
            previous_values = [
                float(record.get(feature_name, 0.0))
                for record in history[-ROLLING_CONTEXT_WINDOW:]
            ]
            current_value = float(current.get(feature_name, 0.0))

            if len(previous_values) >= 2:
                mean_value = float(np.mean(previous_values))
                std_value = float(np.std(previous_values, ddof=1))
            else:
                mean_value = 0.0
                std_value = 0.0

            previous_value = previous_values[-1] if previous_values else current_value
            zscore = (current_value - mean_value) / std_value if std_value > EPSILON else 0.0

            current[f"{feature_name}_rolling_mean"] = mean_value
            current[f"{feature_name}_rolling_std"] = std_value
            current[f"{feature_name}_zscore"] = zscore
            current[f"{feature_name}_trend"] = current_value - previous_value

        current["burst_indicator"] = float(
            max(
                abs(float(current[f"{feature_name}_zscore"]))
                for feature_name in F3_CONTEXT_BASE_FEATURES
            )
            >= 3.0
        )
