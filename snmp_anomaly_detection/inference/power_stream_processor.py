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

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    PHASE_LEVEL_FEATURES,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.preprocessing.phase_features import (
    _IMBALANCE_COLS,
    _IMBALANCE_CLIP_MAX,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch

# Mirrors apply_log1p_skewed from power_features.py
_LOG1P_COLS: tuple[str, ...] = ("runtime_remaining_min", "output_power_w")

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

# Features excluded from attribution — confirmed zero ground-truth deviation
_ATTRIBUTION_EXCLUDE: frozenset[str] = frozenset({"input_frequency_hz"})

# output_frequency_hz MSE is excluded for non-UPS categories (pdu/network/env).
# For UPS it is the inverter output frequency — a real fault signal.
# For PDU/network/env it is ambient line frequency (grid noise) that causes FPs.
_NON_UPS_MSE_EXCLUDE: frozenset[str] = frozenset({"output_frequency_hz"})

_EPS = 1e-9


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
    final_flag: int           # combined_flag OR overload_rule_flag
    alert_policy: str
    compound_alert: bool
    compound_alert_peer: str = ""
    calibration_status: str = "live"  # "calibrating" (cold-start) or "live" (device threshold active)
    # Ground-truth fields — unavailable at inference time; kept for schema parity
    true_label: int = 0
    anomaly_types: list[str] = field(default_factory=list)


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

    def _apply_preprocessing(self, event: PowerEvent) -> None:
        """Mirror the training transforms applied before scaling.

        Replicates, in order:
          1. apply_log1p_skewed  — log1p on runtime_remaining_min and output_power_w
          2. add_delta_features  — signed_log1p of per-device diff for the 4 delta cols
        """
        # A new producer session signals that _prev_raw carries state from a different
        # run. Clearing it prevents a large delta spike on the first scored window.
        if event.session_reset:
            self._prev_raw.pop(event.device_id, None)
            self._true_label_buffers.pop(event.device_id, None)
            self._anomaly_type_buffers.pop(event.device_id, None)

        fv = event.feature_values

        # Step 1: log1p on skewed cols (runtime is log1p'd before delta is computed, same as training)
        for col in _LOG1P_COLS:
            if col in fv:
                fv[col] = float(np.log1p(max(fv[col], 0.0)))

        # Step 2: per-device signed_log1p delta
        prev = self._prev_raw.get(event.device_id)
        for src, tgt in zip(_DELTA_SOURCES, _DELTA_TARGETS):
            cur = fv.get(src, 0.0)
            if prev is not None:
                diff = cur - prev.get(src, cur)
                fv[tgt] = float(np.sign(diff) * np.log1p(abs(diff)))
            else:
                fv[tgt] = 0.0  # first event for this device — no previous value

        # Persist current (post-log1p) values for the next event from this device
        self._prev_raw[event.device_id] = {src: fv.get(src, 0.0) for src in _DELTA_SOURCES}

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
        loss_fn = torch.nn.MSELoss(reduction="none")
        tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            recon = model(tensor)
            errors = loss_fn(recon, tensor).squeeze(0).cpu().numpy()  # (seq_len, n_feat)

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
            if f in _ATTRIBUTION_EXCLUDE:
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

        # Scale baseline vector — use DataFrame to match column names the scaler was fitted with
        b_raw = pd.DataFrame([event.get_baseline_vector()], columns=list(BASELINE_UPS_FEATURES))
        b_scaled = self._baseline_scaler.transform(b_raw)[0].tolist()
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
            p_raw = pd.DataFrame([event.get_phase_vector()], columns=list(PHASE_LEVEL_FEATURES))
            p_scaled = self._phase_scaler.transform(p_raw)[0].tolist()
            p_window = self._phase_windows.push(event.device_id, p_scaled)

        # Only score when both windows are ready (for UPS) or just baseline (other)
        if b_window is None:
            return None
        if event.device_category == "ups" and p_window is None:
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
        overload_rule = int(bool(win_load_pct) and max(win_load_pct) > 100.0)
        final_flag = int(combined or overload_rule)

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
        )
