"""Real-time power anomaly detection stream processor.

Stateful per-device processor for power SNMP events with device-category-based
model routing:
  - UPS devices → dual model (baseline LSTM + phase LSTM) + IForest
  - PDU / network / env devices → baseline LSTM + IForest only

Uses EventPreprocessor (F12/F13), DualModelScorer, and RollingWindowBuffer (F14).

Compound alert: when a UPS anomaly and a PDU anomaly occur within
PowerInferenceConfig.compound_alert_window_minutes of each other, a
COMPOUND_ALERT is emitted.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    PHASE_LEVEL_FEATURES,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.models.lstm_autoencoder import torch
from snmp_anomaly_detection.inference.events import PowerEvent
from snmp_anomaly_detection.inference.event_preprocessor import EventPreprocessor
from snmp_anomaly_detection.inference.model_scorer import (
    DualModelResult,
    DualModelScorer,
    RollingWindowBuffer,
    NON_UPS_MSE_EXCLUDE as _NON_UPS_MSE_EXCLUDE,
)


class AlertState(str, Enum):
    SUSPECTED = "suspected"  # IF fired, LSTM hasn't scored yet (cold-start early warning)
    CONFIRMED = "confirmed"  # both IF and LSTM flagged
    LSTM_ONLY = "lstm_only"  # LSTM flagged without IF, or normal window (no flag)
    CLEARED   = "cleared"    # IF fired but LSTM did not confirm within n_confirmation_windows


@dataclass
class PendingAlert:
    if_score: float
    if_threshold: float
    if_peak_timestep: str
    lstm_windows_pending: int = 0  # LSTM windows scored since IF fired without LSTM confirmation


@dataclass
class DeviceErrorStats:
    """Per-device, per-model calibration state for B1 online threshold."""
    calibration_errors: list  # collected during cold-start; cleared after calibration
    calibrated: bool           # True once n_calibration_windows normal errors observed
    rolling_mean: float        # Welford running mean (post-calibration)
    rolling_m2: float          # Welford M2 accumulator (sum of squared deviations)
    n_windows: int             # total windows used in rolling stats


@dataclass
class PowerScoringResult:
    device_id: str
    device_category: str
    vendor: str
    phase_count: int
    window_start: str
    window_end: datetime
    window_timestamps: list[str]
    # Baseline model
    baseline_anomaly: int
    baseline_error: float
    baseline_threshold: float
    baseline_peak_timestep: str
    baseline_top_features: str
    baseline_timestep_errors: list[float]
    # Phase model
    phase_anomaly: int
    phase_error: float
    phase_threshold: float
    phase_peak_timestep: str
    phase_top_features: str
    phase_timestep_errors: list[float]
    # Combined flags — mirrors DualModelResult
    combined_flag: int
    overload_rule_flag: int   # output_load_pct > 100 in any window timestep
    final_flag: int           # combined_flag OR overload_rule_flag OR iforest_flag
    alert_policy: str
    compound_alert: bool
    compound_alert_peer: str = ""
    calibration_status: str = "live"  # "calibrating" (cold-start) or "live" (device threshold active)
    # Ground-truth fields — unavailable at inference time; kept for schema parity
    true_label: int = 0
    anomaly_types: list[str] = field(default_factory=list)
    # IForest fields (E6: scored independently on every incoming row)
    iforest_score: float = 0.0
    iforest_threshold: float = 0.0
    iforest_flag: int = 0
    # E7: two-stage alert lifecycle
    alert_state: AlertState = AlertState.LSTM_ONLY


class PowerStreamProcessor:
    def __init__(
        self,
        inference_config: PowerInferenceConfig | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for live power detection.")
        self._config = inference_config or PowerInferenceConfig()
        self._paths = paths or ProjectPaths()
        self._scorer = DualModelScorer(self._paths)
        self._baseline_windows = RollingWindowBuffer(self._scorer.baseline_meta["seq_len"])
        self._phase_windows = RollingWindowBuffer(self._scorer.phase_meta["seq_len"])
        # Per-device timestamp buffer — aligned with baseline window size
        self._ts_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        # Per-device raw output_load_pct buffer for overload rule (unscaled, pre-log1p)
        self._load_pct_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        self._preprocessor = EventPreprocessor()
        # Per-device ground-truth label buffers (aligned with baseline window)
        self._true_label_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        self._anomaly_type_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        # Recent anomaly timestamps for compound alert correlation
        self._recent_ups_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)
        self._recent_pdu_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)
        # B1 — per-device online threshold calibration
        self._n_cal = self._config.n_calibration_windows
        self._thresh_k = self._config.online_threshold_k
        self._safety_mult = self._config.calibration_safety_multiplier
        self._device_stats: dict[str, dict[str, DeviceErrorStats]] = {}
        self._windows_since_save: int = 0
        self._load_device_stats()
        # E7 — two-stage alert lifecycle
        self._pending_alerts: dict[str, PendingAlert | None] = {}
        self._n_confirmation_windows: int = self._config.n_confirmation_windows

    # ------------------------------------------------------------------
    # B1 — per-device online threshold helpers
    # ------------------------------------------------------------------

    def _get_stats(self, device_id: str, model_key: str) -> DeviceErrorStats:
        if device_id not in self._device_stats:
            self._device_stats[device_id] = {}
        if model_key not in self._device_stats[device_id]:
            self._device_stats[device_id][model_key] = DeviceErrorStats(
                calibration_errors=[], calibrated=False,
                rolling_mean=0.0, rolling_m2=0.0, n_windows=0,
            )
        return self._device_stats[device_id][model_key]

    def _get_device_threshold(
        self, device_id: str, model_key: str, category: str
    ) -> tuple[float, bool]:
        """Return (threshold, is_calibrated).

        Fallback chain: device threshold → per-category threshold × safety_mult → global × safety_mult.
        During cold-start the safety multiplier keeps the threshold high enough to suppress
        normal OOD reconstruction noise without masking catastrophic faults.
        """
        stats = self._get_stats(device_id, model_key)
        if stats.calibrated:
            std = (stats.rolling_m2 / max(stats.n_windows - 1, 1)) ** 0.5
            return stats.rolling_mean + self._thresh_k * std, True
        meta = self._scorer.baseline_meta if model_key == "baseline" else self._scorer.phase_meta
        global_thresh = meta["threshold"]
        cat_thresh = meta.get("per_category_thresholds", {}).get(category, global_thresh)
        return cat_thresh * self._safety_mult, False

    def _update_stats(self, device_id: str, model_key: str, error: float, category: str) -> None:
        """Update DeviceErrorStats for one scored window using Welford online algorithm.

        Anomalous windows (clearly above current threshold) are locked out so a real fault
        cannot inflate the device's normal baseline.
        """
        stats = self._get_stats(device_id, model_key)
        meta = self._scorer.baseline_meta if model_key == "baseline" else self._scorer.phase_meta
        global_thresh = meta["threshold"]
        cat_thresh = meta.get("per_category_thresholds", {}).get(category, global_thresh)

        if not stats.calibrated:
            # During cold-start: skip windows that are clearly fault-driven
            if error > self._safety_mult * cat_thresh:
                return
            stats.calibration_errors.append(error)
            if len(stats.calibration_errors) >= self._n_cal:
                errors = stats.calibration_errors
                mean = sum(errors) / len(errors)
                m2 = sum((e - mean) ** 2 for e in errors)
                stats.calibrated = True
                stats.rolling_mean = mean
                stats.rolling_m2 = m2
                stats.n_windows = len(errors)
                stats.calibration_errors = []  # free memory
                device_thresh = mean + self._thresh_k * (m2 / max(len(errors) - 1, 1)) ** 0.5
                print(
                    f"[B1] {device_id}/{model_key} calibrated — "
                    f"mean={mean:.6f}  threshold={device_thresh:.6f}"
                )
                self._save_device_stats()
        else:
            # Live: lock out windows that are anomalous (> 2.5× current device threshold)
            std = (stats.rolling_m2 / max(stats.n_windows - 1, 1)) ** 0.5
            live_thresh = stats.rolling_mean + self._thresh_k * std
            if error > 2.5 * live_thresh:
                return
            # Welford online update
            stats.n_windows += 1
            delta = error - stats.rolling_mean
            stats.rolling_mean += delta / stats.n_windows
            delta2 = error - stats.rolling_mean
            stats.rolling_m2 += delta * delta2

    def _load_device_stats(self) -> None:
        stats_file = self._paths.device_stats_file
        if not stats_file.exists():
            return
        with open(stats_file) as f:
            raw = json.load(f)
        for device_id, models in raw.items():
            self._device_stats[device_id] = {}
            for model_key, s in models.items():
                self._device_stats[device_id][model_key] = DeviceErrorStats(
                    calibration_errors=s["calibration_errors"],
                    calibrated=s["calibrated"],
                    rolling_mean=s["rolling_mean"],
                    rolling_m2=s["rolling_m2"],
                    n_windows=s["n_windows"],
                )

    def _save_device_stats(self) -> None:
        stats_file = self._paths.device_stats_file
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        raw: dict = {}
        for device_id, models in self._device_stats.items():
            raw[device_id] = {}
            for model_key, stats in models.items():
                raw[device_id][model_key] = {
                    "calibration_errors": stats.calibration_errors,
                    "calibrated": stats.calibrated,
                    "rolling_mean": stats.rolling_mean,
                    "rolling_m2": stats.rolling_m2,
                    "n_windows": stats.n_windows,
                }
        with open(stats_file, "w") as f:
            json.dump(raw, f)

    def _make_suspected_result(
        self, event: PowerEvent, if_score: float, if_threshold: float, if_flag: int
    ) -> PowerScoringResult:
        """Minimal SUSPECTED result emitted when IF fires before the LSTM buffer is full."""
        ts_str = str(event.timestamp)
        true_labels = list(self._true_label_buffers[event.device_id])
        anomaly_types = list(self._anomaly_type_buffers[event.device_id])
        return PowerScoringResult(
            device_id=event.device_id,
            device_category=event.device_category,
            vendor=event.vendor,
            phase_count=event.phase_count,
            window_start=ts_str,
            window_end=event.timestamp,
            window_timestamps=[ts_str],
            baseline_anomaly=0,
            baseline_error=0.0,
            baseline_threshold=0.0,
            baseline_peak_timestep="",
            baseline_top_features="",
            baseline_timestep_errors=[],
            phase_anomaly=0,
            phase_error=0.0,
            phase_threshold=0.0,
            phase_peak_timestep="",
            phase_top_features="",
            phase_timestep_errors=[],
            combined_flag=0,
            overload_rule_flag=0,
            final_flag=1,
            alert_policy=self._config.alert_policy,
            compound_alert=False,
            calibration_status="calibrating",
            true_label=int(any(true_labels)),
            anomaly_types=list({t for t in anomaly_types if t and t != "none"}),
            iforest_score=round(if_score, 6),
            iforest_threshold=round(if_threshold, 6),
            iforest_flag=if_flag,
            alert_state=AlertState.SUSPECTED,
        )

    def _check_compound_alert(
        self, category: str, device_id: str, ts: datetime
    ) -> tuple[bool, str]:
        window = timedelta(minutes=self._config.compound_alert_window_minutes)
        if category == "ups":
            for (peer_id, peer_ts) in self._recent_pdu_anomalies:
                if abs(ts - peer_ts) <= window:
                    return True, peer_id
        elif category == "pdu":
            for (peer_id, peer_ts) in self._recent_ups_anomalies:
                if abs(ts - peer_ts) <= window:
                    return True, peer_id
        return False, ""

    def process_event(self, event: PowerEvent) -> PowerScoringResult | None:
        if event.session_reset:
            self._true_label_buffers.pop(event.device_id, None)
            self._anomaly_type_buffers.pop(event.device_id, None)
        self._preprocessor.preprocess(event)

        # Track timestamp, ground-truth labels, and raw output_load_pct for overload rule
        self._ts_buffers[event.device_id].append(str(event.timestamp))
        self._true_label_buffers[event.device_id].append(event.true_label)
        self._anomaly_type_buffers[event.device_id].append(event.anomaly_type)
        self._load_pct_buffers[event.device_id].append(
            event.feature_values.get("output_load_pct", 0.0)
        )

        # sklearn scalers are order-dependent, not name-dependent — numpy is fine here
        b_raw = np.array([event.get_baseline_vector()])
        b_scaled = self._scorer.baseline_scaler.transform(b_raw)[0].tolist()

        # E6: score IF on the current row immediately — no buffer needed
        if_score, if_threshold, if_flag = self._scorer.score_iforest_row(b_scaled, event.device_category)

        b_window = self._baseline_windows.push(event.device_id, b_scaled)

        p_window = None
        if event.device_category == "ups":
            self._preprocessor.preprocess_phase(event)
            p_raw = np.array([event.get_phase_vector()])
            p_scaled = self._scorer.phase_scaler.transform(p_raw)[0].tolist()
            p_window = self._phase_windows.push(event.device_id, p_scaled)

        # Only score LSTM when windows are full; IF already fired above regardless
        if b_window is None or (event.device_category == "ups" and p_window is None):
            if if_flag:
                # E7: store pending alert, emit SUSPECTED early warning
                self._pending_alerts[event.device_id] = PendingAlert(
                    if_score=if_score,
                    if_threshold=if_threshold,
                    if_peak_timestep=str(event.timestamp),
                )
                return self._make_suspected_result(event, if_score, if_threshold, if_flag)
            return None

        win_timestamps = list(self._ts_buffers[event.device_id])
        win_load_pct = list(self._load_pct_buffers[event.device_id])
        win_true_label = int(any(self._true_label_buffers[event.device_id]))
        win_anomaly_types = list({
            t for t in self._anomaly_type_buffers[event.device_id]
            if t and t != "none"
        })

        # B1 — resolve per-device thresholds (falls back to category × safety_mult during cold-start)
        b_thresh, b_calibrated = self._get_device_threshold(
            event.device_id, "baseline", event.device_category
        )
        p_thresh, p_calibrated = self._get_device_threshold(
            event.device_id, "phase", event.device_category
        )
        if event.device_category == "ups":
            cal_status = "live" if (b_calibrated and p_calibrated) else "calibrating"
        else:
            cal_status = "live" if b_calibrated else "calibrating"

        mse_excl = _NON_UPS_MSE_EXCLUDE if event.device_category != "ups" else frozenset()
        b_detail = self._scorer.score_window(
            "baseline", b_window, list(BASELINE_UPS_FEATURES), mse_excl
        )

        # Phase scoring: UPS only; other categories get zero-filled placeholders
        if event.device_category == "ups" and p_window is not None:
            p_detail = self._scorer.score_window(
                "phase", p_window, list(PHASE_LEVEL_FEATURES), mse_excl
            )
        else:
            p_detail = {
                "mean_error": 0.0,
                "timestep_errors": [], "peak_timestep_idx": 0, "top_features": [],
            }

        # Update per-device stats (Welford); anomalous windows are locked out inside _update_stats
        self._update_stats(event.device_id, "baseline", b_detail["mean_error"], event.device_category)
        if event.device_category == "ups":
            self._update_stats(event.device_id, "phase", p_detail["mean_error"], event.device_category)

        # Periodic save — on calibration transitions _save_device_stats is called immediately;
        # here we flush remaining state every 100 windows to bound data loss on crash
        self._windows_since_save += 1
        if self._windows_since_save >= 100:
            self._save_device_stats()
            self._windows_since_save = 0

        b_flag = int(b_detail["mean_error"] > b_thresh)
        p_flag = int(p_detail["mean_error"] > p_thresh)
        policy = self._config.alert_policy
        combined = int(b_flag or p_flag) if policy == "or" else int(b_flag and p_flag)

        # Rule-based overload: output_load_pct > 100 is physically impossible during normal op
        overload_rule = int(bool(win_load_pct) and max(win_load_pct) > self._config.overload_load_pct_threshold)

        # E7: two-stage alert lifecycle state machine
        lstm_flagged = bool(combined or overload_rule)
        pending = self._pending_alerts.get(event.device_id)
        if pending is not None:
            # Pending SUSPECTED alert from cold-start — resolve it now
            if lstm_flagged or if_flag:
                # LSTM confirms (or IF fires again) → CONFIRMED
                alert_state = AlertState.CONFIRMED
                effective_if_flag = 1
                self._pending_alerts[event.device_id] = None
            else:
                pending.lstm_windows_pending += 1
                if pending.lstm_windows_pending >= self._n_confirmation_windows:
                    # Timeout — auto-clear the pending SUSPECTED alert
                    alert_state = AlertState.CLEARED
                    effective_if_flag = 0
                    self._pending_alerts[event.device_id] = None
                else:
                    # Still within confirmation window — keep SUSPECTED
                    alert_state = AlertState.SUSPECTED
                    effective_if_flag = 1  # pending IF still active
        else:
            effective_if_flag = if_flag
            if if_flag and lstm_flagged:
                alert_state = AlertState.CONFIRMED
            elif lstm_flagged:
                alert_state = AlertState.LSTM_ONLY
            elif if_flag:
                # IF fired in steady state but LSTM didn't confirm → immediate CLEARED
                alert_state = AlertState.CLEARED
                effective_if_flag = 0
            else:
                alert_state = AlertState.LSTM_ONLY  # normal window, no flag

        # CLEARED is a retraction — suppress the IF contribution from final_flag
        final_flag = int(combined or overload_rule or effective_if_flag)

        # Compound alert tracking
        compound, peer = False, ""
        if final_flag:
            compound, peer = self._check_compound_alert(
                event.device_category, event.device_id, event.timestamp
            )
            if event.device_category == "ups":
                self._recent_ups_anomalies.append((event.device_id, event.timestamp))
            elif event.device_category == "pdu":
                self._recent_pdu_anomalies.append((event.device_id, event.timestamp))

        b_peak_idx = b_detail["peak_timestep_idx"]
        p_peak_idx = p_detail["peak_timestep_idx"]

        return PowerScoringResult(
            device_id=event.device_id,
            device_category=event.device_category,
            vendor=event.vendor,
            phase_count=event.phase_count,
            window_start=win_timestamps[0] if win_timestamps else "",
            window_end=event.timestamp,
            window_timestamps=win_timestamps,
            baseline_anomaly=b_flag,
            baseline_error=round(b_detail["mean_error"], 6),
            baseline_threshold=round(b_thresh, 6),
            baseline_peak_timestep=win_timestamps[b_peak_idx] if win_timestamps else "",
            baseline_top_features=",".join(b_detail["top_features"]),
            baseline_timestep_errors=b_detail["timestep_errors"],
            phase_anomaly=p_flag,
            phase_error=round(p_detail["mean_error"], 6),
            phase_threshold=round(p_thresh, 6),
            phase_peak_timestep=(
                win_timestamps[p_peak_idx]
                if win_timestamps and p_detail["timestep_errors"] else ""
            ),
            phase_top_features=",".join(p_detail["top_features"]),
            phase_timestep_errors=p_detail["timestep_errors"],
            combined_flag=combined,
            overload_rule_flag=overload_rule,
            final_flag=final_flag,
            alert_policy=policy,
            compound_alert=compound,
            compound_alert_peer=peer,
            calibration_status=cal_status,
            true_label=win_true_label,
            anomaly_types=win_anomaly_types,
            iforest_score=round(if_score, 6),
            iforest_threshold=round(if_threshold, 6),
            iforest_flag=effective_if_flag,
            alert_state=alert_state,
        )


def to_dual_result(r: PowerScoringResult) -> DualModelResult:
    """Convert a streaming PowerScoringResult to DualModelResult for unified report saving."""
    return DualModelResult(
        device_id=r.device_id,
        device_category=r.device_category,
        vendor=r.vendor,
        phase_count=r.phase_count,
        window_start=r.window_start,
        window_end=str(r.window_end),
        baseline_error=r.baseline_error,
        baseline_threshold=r.baseline_threshold,
        baseline_anomaly=r.baseline_anomaly,
        phase_error=r.phase_error,
        phase_threshold=r.phase_threshold,
        phase_anomaly=r.phase_anomaly,
        combined_flag=r.combined_flag,
        overload_rule_flag=r.overload_rule_flag,
        final_flag=r.final_flag,
        alert_policy=r.alert_policy,
        window_timestamps=r.window_timestamps,
        baseline_timestep_errors=r.baseline_timestep_errors,
        baseline_peak_timestep=r.baseline_peak_timestep,
        baseline_top_features=r.baseline_top_features,
        phase_timestep_errors=r.phase_timestep_errors,
        phase_peak_timestep=r.phase_peak_timestep,
        phase_top_features=r.phase_top_features,
        iforest_score=r.iforest_score,
        iforest_threshold=r.iforest_threshold,
        iforest_flag=r.iforest_flag,
    )
