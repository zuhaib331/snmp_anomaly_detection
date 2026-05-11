"""Scalar (single-event) preprocessing transforms for power SNMP features.

These are the canonical implementations of every feature-engineering step.
They operate on plain Python dicts (feature_values) rather than DataFrames,
making them usable by any transport (Kafka, MQTT, ZMQ, CSV replay, …) without
importing pandas.

The pandas wrappers in power_features.py and phase_features.py apply the same
math using vectorised DataFrame operations for training-time batch processing.
Both must stay in sync with these definitions — if you change a formula here,
update the corresponding pandas wrapper too, and vice versa.
"""
from __future__ import annotations

import numpy as np


# ── Column-name constants ────────────────────────────────────────────────────
# Shared by both the scalar functions below and the pandas wrappers.

PHASE_V_COLS: tuple[str, ...] = (
    "input_voltage_l1",
    "input_voltage_l2",
    "input_voltage_l3",
)

PHASE_I_COLS: tuple[str, ...] = (
    "input_current_l1",
    "input_current_l2",
    "input_current_l3",
)

# Columns transformed by log1p before delta computation.
LOG1P_COLS: tuple[str, ...] = ("runtime_remaining_min",)

# (source_column, target_delta_column) pairs for signed_log1p deltas.
DELTA_PAIRS: tuple[tuple[str, str], ...] = (
    ("runtime_remaining_min", "runtime_delta"),
    ("battery_charge_pct",    "battery_charge_delta"),
    ("battery_temperature_c", "temperature_delta"),
    ("output_load_pct",       "output_load_delta"),
)

# Voltage drop deltas (B3) — only negative diffs kept; rises clamped to 0.
VOLT_DROP_PAIRS: tuple[tuple[str, str], ...] = (
    ("input_voltage_l1", "voltage_drop_delta_l1"),
    ("input_voltage_l2", "voltage_drop_delta_l2"),
    ("input_voltage_l3", "voltage_drop_delta_l3"),
)

# All source columns that need a previous-row value for delta computation.
ALL_DELTA_SOURCES: tuple[str, ...] = (
    tuple(src for src, _ in DELTA_PAIRS)
    + tuple(src for src, _ in VOLT_DROP_PAIRS)
)

# Physical max for imbalance features (>5 % abnormal, >10 % fault level).
# Clipping prevents near-zero IQR from overflowing RobustScaler.
IMBALANCE_CLIP_MAX: float = 10.0
IMBALANCE_COLS: tuple[str, ...] = ("voltage_imbalance_pct", "current_skew_pct")


# ── Scalar transform functions ───────────────────────────────────────────────
# Each function takes a feature dict and mutates it in-place.
# Call them in the order listed; the ordering matches the training pipeline.

def aggregate_phase_voltages(fv: dict) -> None:
    """Recompute input_voltage_v = mean(L1, L2, L3) when phase columns present.

    Must run before normalize_absolute so input_voltage_dev_pct is derived from
    the phase-averaged value — matching aggregate_phase_metrics() in power_features.py.
    """
    if all(c in fv for c in PHASE_V_COLS):
        fv["input_voltage_v"] = sum(fv[c] for c in PHASE_V_COLS) / 3.0


def normalize_absolute(
    fv: dict,
    rated_capacity_w: float,
    nominal_voltage_v: float,
    rated_battery_v: float,
) -> None:
    """Replace absolute V/A/W features with vendor-agnostic ratios and deviations.

    Matches normalize_absolute_features() in power_features.py.
    Call BEFORE apply_log1p so ratios are computed on raw values.
    """
    cap  = max(rated_capacity_w, 1.0)
    nomv = max(nominal_voltage_v, 1.0)

    rated_current = cap / nomv
    fv["output_current_ratio"]   = fv.get("output_current_a",  0.0) / max(rated_current, 0.01)
    fv["input_voltage_dev_pct"]  = (fv.get("input_voltage_v",  nomv) - nomv) / nomv * 100.0
    fv["output_voltage_dev_pct"] = (fv.get("output_voltage_v", nomv) - nomv) / nomv * 100.0

    if rated_battery_v > 0:
        rated_bv = max(rated_battery_v, 1.0)
        fv["battery_voltage_ratio"] = fv.get("battery_voltage_v", 0.0) / rated_bv
        rated_bc = cap / rated_bv / 10.0
        fv["battery_current_ratio"] = fv.get("battery_current_a", 0.0) / max(rated_bc, 0.01)
    else:
        fv["battery_voltage_ratio"] = 0.0
        fv["battery_current_ratio"] = 0.0


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


def compute_volt_drop_deltas(fv: dict, prev: dict) -> None:
    """Voltage drop delta features (B3) — negative diffs only, rises clamped to 0.

    Amplifies phase sag signal without diluting it with recovery rises.
    Matches the drop-delta block in add_delta_features() in power_features.py.
    """
    for src, tgt in VOLT_DROP_PAIRS:
        cur = fv.get(src, 0.0)
        if prev:
            diff = min(cur - prev.get(src, cur), 0.0)
            fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
        else:
            fv[tgt] = 0.0


def compute_imbalance(fv: dict) -> None:
    """Recompute voltage_imbalance_pct and current_skew_pct from raw phase columns.

    Uses the NEMA MG-1 formula: max deviation from mean / mean * 100.
    Overwrites any pre-computed value so the formula is always identical to
    training, regardless of what the upstream collector sent.
    Matches compute_voltage_imbalance_pct / compute_current_skew_pct in phase_features.py.
    """
    if all(c in fv for c in PHASE_V_COLS):
        vs  = [fv[c] for c in PHASE_V_COLS]
        avg = sum(vs) / 3.0
        fv["voltage_imbalance_pct"] = (
            max(abs(v - avg) for v in vs) / max(avg, 1e-6) * 100.0
        )
    if all(c in fv for c in PHASE_I_COLS):
        cs  = [fv[c] for c in PHASE_I_COLS]
        avg = sum(cs) / 3.0
        fv["current_skew_pct"] = (
            max(abs(c - avg) for c in cs) / max(avg, 1e-6) * 100.0
        )
