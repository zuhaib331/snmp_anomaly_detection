"""EventPreprocessor: single-event, stateful feature preprocessor for streaming inference.

Applies all feature-engineering steps from scalar_transforms.py in the canonical
training order.  Per-device delta state is owned here — callers do not need to
track previous-row values themselves.

Usage::

    preprocessor = EventPreprocessor()
    preprocessor.preprocess(event)   # mutates event.feature_values in-place
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from snmp_anomaly_detection.power.preprocessing.scalar_transforms import (
    ALL_DELTA_SOURCES,
    apply_log1p,
    compute_deltas,
    normalize_absolute,
)

if TYPE_CHECKING:
    from snmp_anomaly_detection.power.inference.events import PowerEvent


class EventPreprocessor:
    """Stateful per-device feature preprocessor for streaming inference.

    Applies the canonical preprocessing steps from scalar_transforms.py
    in training order.  Session resets (event.session_reset=True) clear
    stale delta state for the affected device before processing.
    """

    def __init__(self) -> None:
        self._prev_raw: dict[str, dict[str, float]] = {}

    def reset_device(self, device_id: str) -> None:
        """Clear stale delta state for one device."""
        self._prev_raw.pop(device_id, None)

    def preprocess(self, event: PowerEvent) -> None:
        """Apply all feature-engineering steps in training order.

        Mutates event.feature_values in-place.  Steps:
          1. normalize_absolute — raw V → vendor-agnostic ratios/deviations
          2. apply_log1p        — log1p on runtime_remaining_min
          3. compute_deltas     — signed log1p deltas (post-log1p values)
        """
        if event.session_reset:
            self.reset_device(event.device_id)

        fv = event.feature_values
        prev = self._prev_raw.get(event.device_id)

        normalize_absolute(
            fv,
            event.nominal_voltage_v,
            event.rated_battery_v,
        )

        apply_log1p(fv)

        compute_deltas(fv, prev or {})

        self._prev_raw[event.device_id] = {
            src: fv.get(src, 0.0) for src in ALL_DELTA_SOURCES
        }
