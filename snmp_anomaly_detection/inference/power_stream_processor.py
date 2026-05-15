"""Real-time power anomaly detection stream processor.

Stateful per-device processor for power SNMP events.
Models: Baseline LSTM Autoencoder + Isolation Forest.

Uses EventPreprocessor, DualModelScorer, and RollingWindowBuffer from model_scorer.py.
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
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    LSTM_ONLY = "lstm_only"
    CLEARED   = "cleared"


@dataclass
class PendingAlert:
    if_score: float
    if_threshold: float
    if_peak_timestep: str
    lstm_windows_pending: int = 0


@dataclass
class DeviceErrorStats:
    calibration_errors: list
    calibrated: bool
    rolling_mean: float
    rolling_m2: float
    n_windows: int


@dataclass
class PowerScoringResult:
    device_id: str
    device_category: str
    vendor: str
    phase_count: int
    window_start: str
    window_end: datetime
    window_timestamps: list[str]
    baseline_anomaly: int
    baseline_error: float
    baseline_threshold: float
    baseline_peak_timestep: str
    baseline_top_features: list[str]
    baseline_timestep_errors: list[float]
    overload_rule_flag: int
    final_flag: int
    alert_policy: str
    compound_alert: bool
    compound_alert_peer: str = ""
    calibration_status: str = "live"
    iforest_score: float = 0.0
    iforest_threshold: float = 0.0
    iforest_flag: int = 0
    iforest_peak_timestep: str = ""
    iforest_top_features: list[str] = field(default_factory=list)
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
        self._ts_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        self._load_pct_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._scorer.baseline_meta["seq_len"])
        )
        self._preprocessor = EventPreprocessor()
        self._recent_ups_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)
        self._recent_pdu_anomalies: deque[tuple[str, datetime]] = deque(maxlen=100)
        self._n_cal = self._config.n_calibration_windows
        self._thresh_k = self._config.online_threshold_k
        self._safety_mult = self._config.calibration_safety_multiplier
        self._device_stats: dict[str, DeviceErrorStats] = {}
        self._windows_since_save: int = 0
        self._load_device_stats()
        self._pending_alerts: dict[str, PendingAlert | None] = {}
        self._n_confirmation_windows: int = self._config.n_confirmation_windows

    def _get_stats(self, device_id: str) -> DeviceErrorStats:
        if device_id not in self._device_stats:
            self._device_stats[device_id] = DeviceErrorStats(
                calibration_errors=[], calibrated=False,
                rolling_mean=0.0, rolling_m2=0.0, n_windows=0,
            )
        return self._device_stats[device_id]

    def _get_device_threshold(self, device_id: str, category: str) -> tuple[float, bool]:
        stats = self._get_stats(device_id)
        if stats.calibrated:
            std = (stats.rolling_m2 / max(stats.n_windows - 1, 1)) ** 0.5
            return stats.rolling_mean + self._thresh_k * std, True
        meta = self._scorer.baseline_meta
        global_thresh = meta["threshold"]
        cat_thresh = meta.get("per_category_thresholds", {}).get(category, global_thresh)
        return cat_thresh * self._safety_mult, False

    def _update_stats(self, device_id: str, error: float, category: str) -> None:
        stats = self._get_stats(device_id)
        meta = self._scorer.baseline_meta
        global_thresh = meta["threshold"]
        cat_thresh = meta.get("per_category_thresholds", {}).get(category, global_thresh)

        if not stats.calibrated:
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
                stats.calibration_errors = []
                device_thresh = mean + self._thresh_k * (m2 / max(len(errors) - 1, 1)) ** 0.5
                print(f"[B1] {device_id}/baseline calibrated — mean={mean:.6f}  threshold={device_thresh:.6f}")
                self._save_device_stats()
        else:
            std = (stats.rolling_m2 / max(stats.n_windows - 1, 1)) ** 0.5
            live_thresh = stats.rolling_mean + self._thresh_k * std
            if error > 2.5 * live_thresh:
                return
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
        for device_id, s in raw.items():
            self._device_stats[device_id] = DeviceErrorStats(
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
        for device_id, stats in self._device_stats.items():
            raw[device_id] = {
                "calibration_errors": stats.calibration_errors,
                "calibrated": stats.calibrated,
                "rolling_mean": stats.rolling_mean,
                "rolling_m2": stats.rolling_m2,
                "n_windows": stats.n_windows,
            }
        with open(stats_file, "w") as f:
            json.dump(raw, f)

    def _make_suspected_result(
        self,
        event: PowerEvent,
        if_score: float,
        if_threshold: float,
        if_flag: int,
        if_top_features: list[str],
    ) -> PowerScoringResult:
        ts_str = str(event.timestamp)
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
            baseline_top_features=[],
            baseline_timestep_errors=[],
            overload_rule_flag=0,
            final_flag=1,
            alert_policy=self._config.alert_policy,
            compound_alert=False,
            calibration_status="calibrating",
            iforest_score=round(if_score, 6),
            iforest_threshold=round(if_threshold, 6),
            iforest_flag=if_flag,
            iforest_peak_timestep=ts_str,
            iforest_top_features=if_top_features,
            alert_state=AlertState.SUSPECTED,
        )

    def _check_compound_alert(self, category: str, device_id: str, ts: datetime) -> tuple[bool, str]:
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
            pass
        self._preprocessor.preprocess(event)

        self._ts_buffers[event.device_id].append(str(event.timestamp))
        self._load_pct_buffers[event.device_id].append(
            event.feature_values.get("output_load_pct", 0.0)
        )

        b_raw = np.array([event.get_baseline_vector()])
        b_scaled = self._scorer.baseline_scaler.transform(b_raw)[0].tolist()

        if_score, if_threshold, if_flag = self._scorer.score_iforest_row(b_scaled, event.device_category)
        iforest_peak_timestep = str(event.timestamp) if if_flag else ""
        if_top_features: list[str] = (
            self._scorer.get_iforest_top_features(
                np.array(b_scaled), event.device_category, list(BASELINE_UPS_FEATURES)
            )
            if if_flag else []
        )

        b_window = self._baseline_windows.push(event.device_id, b_scaled)

        if b_window is None:
            if if_flag:
                self._pending_alerts[event.device_id] = PendingAlert(
                    if_score=if_score,
                    if_threshold=if_threshold,
                    if_peak_timestep=str(event.timestamp),
                )
                return self._make_suspected_result(event, if_score, if_threshold, if_flag, if_top_features)
            return None

        win_timestamps = list(self._ts_buffers[event.device_id])
        win_load_pct = list(self._load_pct_buffers[event.device_id])

        b_thresh, b_calibrated = self._get_device_threshold(event.device_id, event.device_category)
        cal_status = "live" if b_calibrated else "calibrating"

        mse_excl = _NON_UPS_MSE_EXCLUDE if event.device_category != "ups" else frozenset()
        b_detail = self._scorer.score_window(b_window, list(BASELINE_UPS_FEATURES), mse_excl)

        self._update_stats(event.device_id, b_detail["mean_error"], event.device_category)

        self._windows_since_save += 1
        if self._windows_since_save >= 100:
            self._save_device_stats()
            self._windows_since_save = 0

        b_flag = int(b_detail["mean_error"] > b_thresh)
        overload_rule = int(bool(win_load_pct) and max(win_load_pct) > self._config.overload_load_pct_threshold)

        lstm_flagged = bool(b_flag or overload_rule)
        pending = self._pending_alerts.get(event.device_id)
        if pending is not None:
            if lstm_flagged or if_flag:
                alert_state = AlertState.CONFIRMED
                effective_if_flag = 1
                self._pending_alerts[event.device_id] = None
            else:
                pending.lstm_windows_pending += 1
                if pending.lstm_windows_pending >= self._n_confirmation_windows:
                    alert_state = AlertState.CLEARED
                    effective_if_flag = 0
                    self._pending_alerts[event.device_id] = None
                else:
                    alert_state = AlertState.SUSPECTED
                    effective_if_flag = 1
        else:
            effective_if_flag = if_flag
            if if_flag and lstm_flagged:
                alert_state = AlertState.CONFIRMED
            elif lstm_flagged:
                alert_state = AlertState.LSTM_ONLY
            elif if_flag:
                alert_state = AlertState.CLEARED
                effective_if_flag = 0
            else:
                alert_state = AlertState.LSTM_ONLY

        final_flag = int(b_flag or overload_rule or effective_if_flag)

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
            baseline_top_features=b_detail["top_features"],
            baseline_timestep_errors=b_detail["timestep_errors"],
            overload_rule_flag=overload_rule,
            final_flag=final_flag,
            alert_policy=self._config.alert_policy,
            compound_alert=compound,
            compound_alert_peer=peer,
            calibration_status=cal_status,
            iforest_score=round(if_score, 6),
            iforest_threshold=round(if_threshold, 6),
            iforest_flag=effective_if_flag,
            iforest_peak_timestep=iforest_peak_timestep,
            iforest_top_features=if_top_features,
            alert_state=alert_state,
        )


def format_power_alert(result: PowerScoringResult) -> str | None:
    state = result.alert_state
    if state == AlertState.SUSPECTED:
        return (
            f"SUSPECTED [IF] [{result.device_category}] "
            f"{result.device_id} @ {result.window_start} "
            f"if_score={result.iforest_score:.4f} < thresh={result.iforest_threshold:.4f}"
        )
    if state == AlertState.CLEARED:
        return (
            f"CLEARED [{result.device_category}] "
            f"{result.device_id} @ {result.window_start} "
            "(IF retracted — LSTM did not confirm)"
        )
    if result.final_flag:
        prefix = "COMPOUND " if result.compound_alert else ""
        msg = (
            f"{prefix}{state.value.upper()} [{result.device_category}] "
            f"{result.device_id} "
            f"{result.window_start} → {result.window_end} "
            f"baseline={result.baseline_anomaly} "
            f"(err={result.baseline_error:.4f} > {result.baseline_threshold:.4f}"
            + (f", top=[{','.join(result.baseline_top_features)}]" if result.baseline_top_features else "")
            + f") if={result.iforest_flag}"
        )
        if result.iforest_flag:
            msg += f" (score={result.iforest_score:.4f}, top=[{','.join(result.iforest_top_features)}])"
        if result.compound_alert:
            msg += f"  peer={result.compound_alert_peer}"
        return msg
    return None


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
        overload_rule_flag=r.overload_rule_flag,
        iforest_flag=r.iforest_flag,
        final_flag=r.final_flag,
        alert_policy=r.alert_policy,
        window_timestamps=r.window_timestamps,
        baseline_timestep_errors=r.baseline_timestep_errors,
        baseline_peak_timestep=r.baseline_peak_timestep,
        baseline_top_features=r.baseline_top_features,
        iforest_score=r.iforest_score,
        iforest_threshold=r.iforest_threshold,
        iforest_peak_timestep=r.iforest_peak_timestep,
        iforest_top_features=r.iforest_top_features,
    )
