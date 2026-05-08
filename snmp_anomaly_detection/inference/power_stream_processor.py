"""Real-time power anomaly detection stream processor.

Extends the master branch's EventProcessor + DeviceWindowManager pattern to
support power events with device-category-based model routing:
  - UPS devices → dual model (baseline + phase)
  - PDU / network / env devices → baseline model only

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

import joblib
import numpy as np

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    IF_MIN_SCORE_RATIO,
    PHASE_LEVEL_FEATURES,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.preprocessing.phase_features import (
    _IMBALANCE_COLS,
    _IMBALANCE_CLIP_MAX,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch

# Mirrors apply_log1p_skewed from power_features.py (output_power_w dropped by B2 normalization)
_LOG1P_COLS: tuple[str, ...] = ("runtime_remaining_min",)

# Mirrors add_delta_features: source columns → target delta column names.
# runtime_remaining_min is tracked *after* log1p (same as training order).
_DELTA_SOURCES: tuple[str, ...] = (
    "runtime_remaining_min",
    "battery_charge_pct",
    "battery_temperature_c",
    "output_load_pct",
)
_DELTA_TARGETS: tuple[str, ...] = (
    "runtime_delta",
    "battery_charge_delta",
    "temperature_delta",
    "output_load_delta",
)

# B3: per-phase voltage drop deltas — only negative diffs (drops) kept.
# Mirrors the clip(upper=0) branch added to add_delta_features() in power_features.py.
_VOLT_DROP_SOURCES: tuple[str, ...] = (
    "input_voltage_l1",
    "input_voltage_l2",
    "input_voltage_l3",
)
_VOLT_DROP_TARGETS: tuple[str, ...] = (
    "voltage_drop_delta_l1",
    "voltage_drop_delta_l2",
    "voltage_drop_delta_l3",
)

# Features excluded from attribution for ALL device categories.
_ATTRIBUTION_EXCLUDE: frozenset[str] = frozenset({
    "input_frequency_hz",
    "output_frequency_hz",
    "on_battery_status",       # transient binary flag — self-evident, not a diagnostic cause
    "battery_replace_status",  # persistent firmware flag — RUL model covers this separately
})

# output_frequency_hz MSE is excluded for non-UPS categories (pdu/network/env).
# For UPS it is the inverter output frequency — a real fault signal.
# For PDU/network/env it is ambient line frequency (grid noise) that causes FPs.
_NON_UPS_MSE_EXCLUDE: frozenset[str] = frozenset({"output_frequency_hz"})

_EPS = 1e-9


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
class PowerEvent:
    timestamp: datetime
    device_id: str
    device_category: str
    vendor: str
    feature_values: dict[str, float]   # canonical feature name → value
    phase_count: int = 1               # 1 (single-phase) or 3 (three-phase)
    # Device registration constants for B2 normalization.
    # Set these from device registration / SNMP discovery at onboarding time.
    # If rated_capacity_w == 0, normalization falls back to raw values (graceful degradation).
    rated_capacity_w: float = 0.0      # nameplate power rating in Watts
    nominal_voltage_v: float = 120.0   # nominal input voltage (120 or 230)
    rated_battery_v: float = 0.0       # battery string voltage (0 for non-UPS)
    session_reset: bool = False        # clears stale per-device delta state on new producer run
    true_label: int = 0                # ground-truth flag from test producer (0 in production)
    anomaly_type: str = "none"         # ground-truth type from test producer ("none" in production)

    def get_baseline_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in BASELINE_UPS_FEATURES]

    def get_phase_vector(self) -> list[float]:
        return [self.feature_values.get(c, 0.0) for c in PHASE_LEVEL_FEATURES]


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


class PowerDeviceWindowManager:
    """Per-device rolling buffer identical in design to the master DeviceWindowManager."""

    def __init__(self, seq_len: int) -> None:
        self._seq_len = seq_len
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))

    def push(self, device_id: str, vector: list[float]) -> np.ndarray | None:
        buf = self._buffers[device_id]
        buf.append(vector)
        if len(buf) == self._seq_len:
            return np.array(buf)
        return None


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
        self._load_models()
        self._baseline_windows = PowerDeviceWindowManager(self._baseline_meta["seq_len"])
        self._phase_windows = PowerDeviceWindowManager(self._phase_meta["seq_len"])
        # Per-device timestamp buffer — aligned with baseline window size
        self._ts_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._baseline_meta["seq_len"])
        )
        # Per-device raw output_load_pct buffer for overload rule (unscaled, pre-log1p)
        self._load_pct_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._baseline_meta["seq_len"])
        )
        # Per-device last-seen values for delta feature computation
        self._prev_raw: dict[str, dict[str, float]] = {}
        # Per-device ground-truth label buffers (aligned with baseline window)
        self._true_label_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._baseline_meta["seq_len"])
        )
        self._anomaly_type_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._baseline_meta["seq_len"])
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

    def _load_models(self) -> None:
        p = self._paths

        with open(p.power_outputs_dir / "baseline_metadata.json") as f:
            self._baseline_meta = json.load(f)
        self._baseline_model = LSTMAutoencoder(
            input_size=self._baseline_meta["input_size"],
            hidden_size=self._baseline_meta["hidden_size"],
            latent_size=self._baseline_meta["latent_size"],
        )
        self._baseline_model.load_state_dict(
            torch.load(p.power_outputs_dir / "baseline_model.pt", map_location="cpu")
        )
        self._baseline_model.eval()
        self._loss_fn = torch.nn.MSELoss(reduction="none")
        self._baseline_scaler = joblib.load(p.power_outputs_dir / "baseline_scaler.pkl")
        self._baseline_threshold: float = self._baseline_meta["threshold"]
        self._baseline_cat_thresholds: dict[str, float] = self._baseline_meta.get("per_category_thresholds", {})
        self._baseline_normal_errors: dict[str, float] = self._baseline_meta.get("normal_feature_errors", {})
        self._baseline_normal_stds: dict[str, float] = self._baseline_meta.get("normal_feature_error_stds", {})

        with open(p.power_phase_outputs_dir / "phase_metadata.json") as f:
            self._phase_meta = json.load(f)
        self._phase_model = LSTMAutoencoder(
            input_size=self._phase_meta["input_size"],
            hidden_size=self._phase_meta["hidden_size"],
            latent_size=self._phase_meta["latent_size"],
        )
        self._phase_model.load_state_dict(
            torch.load(p.power_phase_outputs_dir / "phase_model.pt", map_location="cpu")
        )
        self._phase_model.eval()
        self._phase_scaler = joblib.load(p.power_phase_outputs_dir / "phase_scaler.pkl")
        self._phase_threshold: float = self._phase_meta["threshold"]
        self._phase_cat_thresholds: dict[str, float] = self._phase_meta.get("per_category_thresholds", {})
        self._phase_normal_errors: dict[str, float] = self._phase_meta.get("normal_feature_errors", {})
        self._phase_normal_stds: dict[str, float] = self._phase_meta.get("normal_feature_error_stds", {})

        # E6: IForest loaded for immediate per-row scoring (decoupled from LSTM buffer)
        self._iforest_models: dict = {}
        self._iforest_thresholds: dict[str, float] = {}
        self._iforest_feature_cols: dict[str, list[str]] = {}
        if p.iforest_model_file.exists() and p.iforest_metadata_file.exists():
            self._iforest_models = joblib.load(p.iforest_model_file)
            with open(p.iforest_metadata_file) as f:
                _if_meta = json.load(f)
            self._iforest_thresholds = {cat: v["threshold"] for cat, v in _if_meta.items()}
            self._iforest_feature_cols = {
                cat: v.get("feature_cols", list(BASELINE_UPS_FEATURES))
                for cat, v in _if_meta.items()
            }
        # Pre-compute per-category column index lists once so _score_iforest_row
        # doesn't recompute them on every incoming message.
        _b_col_list = list(BASELINE_UPS_FEATURES)
        self._iforest_col_indices: dict[str, list[int]] = {
            cat: [_b_col_list.index(c) for c in cols if c in _b_col_list]
            for cat, cols in self._iforest_feature_cols.items()
        }

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
        global_thresh = self._baseline_threshold if model_key == "baseline" else self._phase_threshold
        cat_thresholds = self._baseline_cat_thresholds if model_key == "baseline" else self._phase_cat_thresholds
        cat_thresh = cat_thresholds.get(category, global_thresh)
        return cat_thresh * self._safety_mult, False

    def _update_stats(self, device_id: str, model_key: str, error: float, category: str) -> None:
        """Update DeviceErrorStats for one scored window using Welford online algorithm.

        Anomalous windows (clearly above current threshold) are locked out so a real fault
        cannot inflate the device's normal baseline.
        """
        stats = self._get_stats(device_id, model_key)
        global_thresh = self._baseline_threshold if model_key == "baseline" else self._phase_threshold
        cat_thresholds = self._baseline_cat_thresholds if model_key == "baseline" else self._phase_cat_thresholds
        cat_thresh = cat_thresholds.get(category, global_thresh)

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

    # ------------------------------------------------------------------
    # E6 — IForest helpers
    # ------------------------------------------------------------------

    def _score_iforest_row(
        self, b_scaled: list[float], device_category: str
    ) -> tuple[float, float, int]:
        """Score a single scaled baseline row with IForest.

        Returns (if_score, if_threshold, if_flag). Returns (0.0, 0.0, 0) when no
        IF model is available for this category.
        """
        clf = self._iforest_models.get(device_category)
        if clf is None:
            return 0.0, 0.0, 0
        _col_indices = self._iforest_col_indices.get(device_category, list(range(len(BASELINE_UPS_FEATURES))))
        _vec = np.array([[b_scaled[j] for j in _col_indices]])
        score = float(clf.score_samples(_vec)[0])
        threshold = self._iforest_thresholds.get(device_category, -0.5)
        score_ratio = (score / threshold) if threshold != 0 else 0.0
        flag = int(score < threshold and score_ratio >= IF_MIN_SCORE_RATIO)
        return score, threshold, flag

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

    # ------------------------------------------------------------------

    def _apply_preprocessing(self, event: PowerEvent) -> None:
        """Mirror the training transforms applied before scaling.

        Replicates, in order:
          1. normalize_absolute_features — replace absolute V/A/W with vendor-agnostic ratios
          2. apply_log1p_skewed          — log1p on runtime_remaining_min
          3. add_delta_features          — signed_log1p of per-device diff for the 4 delta cols
        """
        # A new producer session signals that _prev_raw carries state from a different
        # run. Clearing it prevents a large delta spike on the first scored window.
        if event.session_reset:
            self._prev_raw.pop(event.device_id, None)
            self._true_label_buffers.pop(event.device_id, None)
            self._anomaly_type_buffers.pop(event.device_id, None)

        fv = event.feature_values

        # Step 1: B2 normalization — mirrors normalize_absolute_features() from power_features.py.
        # Requires rated_capacity_w > 0 (set from device registration at onboarding).
        # Gracefully skips normalization if registration data is absent (rated_capacity_w == 0).
        cap = max(event.rated_capacity_w, 1.0)
        nomv = max(event.nominal_voltage_v, 1.0)
        rated_bv = event.rated_battery_v

        if event.rated_capacity_w > 0:
            rated_current = cap / nomv
            fv["output_current_ratio"] = fv.get("output_current_a", 0.0) / max(rated_current, 0.01)
            fv["input_voltage_dev_pct"] = (fv.get("input_voltage_v", nomv) - nomv) / nomv * 100.0
            fv["output_voltage_dev_pct"] = (fv.get("output_voltage_v", nomv) - nomv) / nomv * 100.0

            if rated_bv > 0:
                fv["battery_voltage_ratio"] = fv.get("battery_voltage_v", 0.0) / max(rated_bv, 1.0)
                rated_bc = cap / max(rated_bv, 1.0) / 10.0
                fv["battery_current_ratio"] = fv.get("battery_current_a", 0.0) / max(rated_bc, 0.01)
            else:
                fv["battery_voltage_ratio"] = 0.0
                fv["battery_current_ratio"] = 0.0

        # Step 2: log1p on skewed cols (runtime is log1p'd before delta is computed, same as training)
        for col in _LOG1P_COLS:
            if col in fv:
                fv[col] = float(np.log1p(max(fv[col], 0.0)))

        # Step 3: per-device signed_log1p delta
        prev = self._prev_raw.get(event.device_id)
        for src, tgt in zip(_DELTA_SOURCES, _DELTA_TARGETS):
            cur = fv.get(src, 0.0)
            if prev is not None:
                diff = cur - prev.get(src, cur)
                fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
            else:
                fv[tgt] = 0.0  # first event for this device — no previous value

        # B3: voltage drop deltas — only negative diffs kept (rises clamped to 0).
        for src, tgt in zip(_VOLT_DROP_SOURCES, _VOLT_DROP_TARGETS):
            cur = fv.get(src, 0.0)
            if prev is not None:
                diff = min(cur - prev.get(src, cur), 0.0)
                fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
            else:
                fv[tgt] = 0.0

        # Persist current (post-log1p) values for the next event from this device
        all_prev_srcs = list(_DELTA_SOURCES) + list(_VOLT_DROP_SOURCES)
        self._prev_raw[event.device_id] = {src: fv.get(src, 0.0) for src in all_prev_srcs}

    def _score_detailed(
        self,
        model,
        window: np.ndarray,
        threshold: float,
        feature_names: list[str],
        normal_errors: dict[str, float],
        normal_stds: dict[str, float],
        mse_exclude: frozenset[str] = frozenset(),
    ) -> dict:
        """Score one window and return reconstruction error diagnostics.

        Returns mean_error, per-timestep errors, flag, peak timestep index,
        and top-3 features by z-score surprise — matching dual_model_scorer.py.
        """
        tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            recon = model(tensor)
            errors = self._loss_fn(recon, tensor).squeeze(0).cpu().numpy()  # (seq_len, n_feat)

        if mse_exclude:
            active_cols = [j for j, f in enumerate(feature_names) if f not in mse_exclude]
            errors_active = errors[:, active_cols]
        else:
            errors_active = errors

        timestep_errors = errors_active.mean(axis=1)   # (seq_len,)
        feature_errors = errors.mean(axis=0)            # (n_feat,) — full, for attribution
        mean_error = float(errors_active.mean())

        # Z-score surprise: std-devs above normal reconstruction error per feature
        surprise = np.clip(
            [
                (feature_errors[j] - normal_errors.get(f, 0.0))
                / (normal_stds.get(f, _EPS) + _EPS)
                for j, f in enumerate(feature_names)
            ],
            0.0, None,
        )
        for j, f in enumerate(feature_names):
            if f in attribution_exclude:
                surprise[j] = 0.0

        peak_idx = int(np.argmax(timestep_errors))
        top_features = [
            feature_names[j]
            for j in np.argsort(surprise)[::-1][:3]
            if j < len(feature_names) and surprise[j] > 0.0
        ]
        return {
            "mean_error": mean_error,
            "flag": int(mean_error > threshold),
            "timestep_errors": [round(float(e), 6) for e in timestep_errors],
            "peak_timestep_idx": peak_idx,
            "top_features": ",".join(top_features),
        }

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
        # Apply log1p + delta features before scaling (mirrors training preprocessing)
        self._apply_preprocessing(event)

        # Track timestamp, ground-truth labels, and raw output_load_pct for overload rule
        self._ts_buffers[event.device_id].append(str(event.timestamp))
        self._true_label_buffers[event.device_id].append(event.true_label)
        self._anomaly_type_buffers[event.device_id].append(event.anomaly_type)
        self._load_pct_buffers[event.device_id].append(
            event.feature_values.get("output_load_pct", 0.0)
        )

        # sklearn scalers are order-dependent, not name-dependent — numpy is fine here
        b_raw = np.array([event.get_baseline_vector()])
        b_scaled = self._baseline_scaler.transform(b_raw)[0].tolist()

        # E6: score IF on the current row immediately — no buffer needed
        if_score, if_threshold, if_flag = self._score_iforest_row(b_scaled, event.device_category)

        b_window = self._baseline_windows.push(event.device_id, b_scaled)

        p_window = None
        if event.device_category == "ups":
            # Apply same imbalance clipping used during phase model training
            for col in _IMBALANCE_COLS:
                if col in event.feature_values:
                    event.feature_values[col] = (
                        min(max(event.feature_values[col], 0.0), _IMBALANCE_CLIP_MAX)
                        / _IMBALANCE_CLIP_MAX
                    )
            p_raw = np.array([event.get_phase_vector()])
            p_scaled = self._phase_scaler.transform(p_raw)[0].tolist()
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
        b_detail = self._score_detailed(
            self._baseline_model, b_window, b_thresh,
            list(BASELINE_UPS_FEATURES),
            self._baseline_normal_errors, self._baseline_normal_stds,
            mse_excl,
        )

        # Phase scoring: UPS only; other categories get zero-filled placeholders
        if event.device_category == "ups" and p_window is not None:
            p_detail = self._score_detailed(
                self._phase_model, p_window, p_thresh,
                list(PHASE_LEVEL_FEATURES),
                self._phase_normal_errors, self._phase_normal_stds,
                mse_excl,
            )
        else:
            p_detail = {
                "mean_error": 0.0, "flag": 0,
                "timestep_errors": [], "peak_timestep_idx": 0, "top_features": "",
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

        b_flag = b_detail["flag"]
        p_flag = p_detail["flag"]
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
            baseline_top_features=b_detail["top_features"],
            baseline_timestep_errors=b_detail["timestep_errors"],
            phase_anomaly=p_flag,
            phase_error=round(p_detail["mean_error"], 6),
            phase_threshold=round(p_thresh, 6),
            phase_peak_timestep=(
                win_timestamps[p_peak_idx]
                if win_timestamps and p_detail["timestep_errors"] else ""
            ),
            phase_top_features=p_detail["top_features"],
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
