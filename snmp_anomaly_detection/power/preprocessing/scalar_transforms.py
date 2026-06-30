"""Scalar (single-event) preprocessing transforms for power SNMP features.

These are the canonical implementations of every feature-engineering step.
They operate on plain Python dicts (feature_values) rather than DataFrames,
making them usable by any transport (Kafka, MQTT, ZMQ, CSV replay, …) without
importing pandas.

The pandas wrappers in power_features.py apply the same math using vectorised
DataFrame operations for training-time batch processing.
Both must stay in sync with these definitions — if you change a formula here,
update the corresponding pandas wrapper too, and vice versa.
"""
from __future__ import annotations

import numpy as np


# ── Column-name constants ────────────────────────────────────────────────────

# Columns transformed by log1p before delta computation.
LOG1P_COLS: tuple[str, ...] = ("runtime_remaining_min",)

# (source_column, target_delta_column) pairs for signed_log1p deltas.
DELTA_PAIRS: tuple[tuple[str, str], ...] = (
    ("runtime_remaining_min", "runtime_delta"),
    ("battery_temperature_c", "temperature_delta"),
    ("output_load_pct",       "output_load_delta"),
)

# All source columns that need a previous-row value for delta computation.
ALL_DELTA_SOURCES: tuple[str, ...] = tuple(src for src, _ in DELTA_PAIRS)


# ── Scalar transform functions ───────────────────────────────────────────────
# Each function takes a feature dict and mutates it in-place.
# Call them in the order listed; the ordering matches the training pipeline.

def normalize_absolute(
    fv: dict,
    nominal_voltage_v: float,
    rated_battery_v: float,
) -> None:
    """Replace absolute V features with vendor-agnostic ratios and deviations.

    Matches normalize_absolute_features() in power_features.py.
    Call BEFORE apply_log1p so ratios are computed on raw values.

    Registration fields used (not model inputs):
      nominal_voltage_v — drives input/output voltage deviation features
      rated_battery_v   — drives battery_voltage_ratio
    """
    nomv = max(nominal_voltage_v, 1.0)

    fv["input_voltage_dev_pct"]  = (fv.get("input_voltage_v",  nomv) - nomv) / nomv * 100.0
    fv["output_voltage_dev_pct"] = (fv.get("output_voltage_v", nomv) - nomv) / nomv * 100.0

    if rated_battery_v > 0:
        rated_bv = max(rated_battery_v, 1.0)
        fv["battery_voltage_ratio"] = fv.get("battery_voltage_v", 0.0) / rated_bv
    else:
        fv["battery_voltage_ratio"] = 0.0


def apply_log1p(fv: dict) -> None:
    """log1p transform on right-skewed columns.

    Must run BEFORE compute_deltas so runtime_delta is the diff of
    log1p(runtime), matching the training order.
    Matches apply_log1p_skewed() in power_features.py.
    """
    for col in LOG1P_COLS:
        if col in fv:
            fv[col] = float(np.log1p(max(fv[col], 0.0)))


def compute_deltas(fv: dict, prev: dict) -> None:
    """Per-device signed_log1p delta features.

    prev is the post-log1p feature dict from the previous event for the same
    device. Pass an empty dict (or omit) for the first event — deltas are 0.
    Matches add_delta_features() in power_features.py.
    """
    for src, tgt in DELTA_PAIRS:
        cur = fv.get(src, 0.0)
        if prev:
            diff = cur - prev.get(src, cur)
            fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
        else:
            fv[tgt] = 0.0
