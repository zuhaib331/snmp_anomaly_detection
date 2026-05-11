"""EventPreprocessor: single-event, stateful feature preprocessor for streaming inference.

Applies all feature-engineering steps from scalar_transforms.py in the canonical
training order.  Per-device delta state is owned here — callers do not need to
track previous-row values themselves.

Usage::

    preprocessor = EventPreprocessor()
    preprocessor.preprocess(event)   # mutates event.feature_values in-place

F12: this class bridges scalar_transforms (F11) to the streaming inference path.
F13 migrated both inference paths to use this class; _apply_preprocessing() removed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from snmp_anomaly_detection.preprocessing.scalar_transforms import (
    ALL_DELTA_SOURCES,
    IMBALANCE_CLIP_MAX,
    IMBALANCE_COLS,
    aggregate_phase_voltages,
    apply_log1p,
    compute_deltas,
    compute_imbalance,
    compute_volt_drop_deltas,
    normalize_absolute,
)

if TYPE_CHECKING:
    from snmp_anomaly_detection.inference.events import PowerEvent


class EventPreprocessor:
    """Stateful per-device feature preprocessor for streaming inference.

    Applies the six canonical preprocessing steps from scalar_transforms.py
    in the training order.  Session resets (event.session_reset=True) clear
    stale delta state for the affected device before processing.

    Call preprocess_phase(event) after preprocess(event) when routing to the
    phase model — it clips imbalance columns before the phase scaler.
    """

    def __init__(self) -> None:
        # Post-log1p feature values from the previous event, keyed by device_id.
        # Used to compute signed deltas and voltage-drop deltas.
        self._prev_raw: dict[str, dict[str, float]] = {}

    def reset_device(self, device_id: str) -> None:
        """Clear stale delta state for one device.

        Call this on a producer session restart to prevent a spurious large
        delta spike on the first scored window of the new session.
        """
        self._prev_raw.pop(device_id, None)

    def preprocess(self, event: PowerEvent) -> None:
        """Apply all feature-engineering steps in training order.

        Mutates event.feature_values in-place.  Steps mirror the batch pipeline:
          1. aggregate_phase_voltages — mean(L1,L2,L3) → input_voltage_v
          2. normalize_absolute       — absolute V/A/W → ratios/deviations (B2)
          3. apply_log1p              — log1p on runtime_remaining_min
          4. compute_deltas           — signed log1p deltas (post-log1p values)
          5. compute_volt_drop_deltas — negative-only phase voltage diffs (B3)
          6. compute_imbalance        — NEMA MG-1 voltage/current imbalance
        """
        if event.session_reset:
            self.reset_device(event.device_id)

        fv = event.feature_values
        prev = self._prev_raw.get(event.device_id)

        # Step 1: must run before normalize_absolute so input_voltage_dev_pct
        # is derived from the phase-averaged value (3-phase UPS only; no-op otherwise).
        aggregate_phase_voltages(fv)

        # Step 2: skip gracefully when device registration data is absent.
        if event.rated_capacity_w > 0:
            normalize_absolute(
                fv,
                event.rated_capacity_w,
                event.nominal_voltage_v,
                event.rated_battery_v,
            )

        # Step 3: must precede deltas — runtime_delta tracks log-scale differences.
        apply_log1p(fv)

        # Steps 4 + 5: pass empty dict for the first event (produces zero deltas).
        compute_deltas(fv, prev or {})
        compute_volt_drop_deltas(fv, prev or {})

        # Step 6: no-op for single-phase or non-phase devices (column presence guard
        # is inside compute_imbalance).
        compute_imbalance(fv)

        # Persist post-log1p values for the next event's delta computation.
        self._prev_raw[event.device_id] = {
            src: fv.get(src, 0.0) for src in ALL_DELTA_SOURCES
        }

    def preprocess_phase(self, event: PowerEvent) -> None:
        """Clip imbalance features into the [0, 1] range expected by the phase scaler.

        Call after preprocess() when routing this event to the phase model.
        Mutates event.feature_values in-place.  Safe to call on non-phase devices —
        columns absent from feature_values are silently skipped.
        """
        fv = event.feature_values
        for col in IMBALANCE_COLS:
            if col in fv:
                fv[col] = min(max(fv[col], 0.0), IMBALANCE_CLIP_MAX) / IMBALANCE_CLIP_MAX
