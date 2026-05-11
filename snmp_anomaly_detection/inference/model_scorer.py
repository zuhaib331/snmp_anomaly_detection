"""Shared model loading, window scoring, rolling buffer, and CSV detection for power anomaly detection.

F14: DualModelScorer and RollingWindowBuffer extracted from dual_model_scorer.py (CSV path)
and power_stream_processor.py (Kafka path) so that adding a model or changing a scoring
formula is a one-file change.

F15: DualModelResult, run_csv_detection, and save_dual_results moved here from
dual_model_scorer.py so both transport files share a single definition.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    IF_MIN_SCORE_RATIO,
    PHASE_LEVEL_FEATURES,
    PHASE_MODEL_CATEGORIES,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch

# Features excluded from attribution for ALL device categories.
ATTRIBUTION_EXCLUDE: frozenset[str] = frozenset({
    "input_frequency_hz",
    "output_frequency_hz",
    "on_battery_status",
    "battery_replace_status",
})

# output_frequency_hz MSE excluded for non-UPS (grid noise FP).
NON_UPS_MSE_EXCLUDE: frozenset[str] = frozenset({"output_frequency_hz"})

_EPS = 1e-9


class RollingWindowBuffer:
    """Per-device rolling deque buffer; yields (seq_len, n_features) array when full.

    Drop-in replacement for PowerDeviceWindowManager in power_stream_processor.py.
    """

    def __init__(self, seq_len: int) -> None:
        self._seq_len = seq_len
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=seq_len))

    def push(self, device_id: str, vector: list[float]) -> np.ndarray | None:
        buf = self._buffers[device_id]
        buf.append(vector)
        if len(buf) == self._seq_len:
            return np.array(buf)
        return None


@dataclass
class _ModelArtifacts:
    baseline_model: Any
    baseline_scaler: Any
    baseline_meta: dict
    phase_model: Any
    phase_scaler: Any
    phase_meta: dict
    iforest_models: dict
    iforest_thresholds: dict
    iforest_feature_cols: dict
    iforest_col_indices: dict


def _load_artifacts(paths: ProjectPaths) -> _ModelArtifacts:
    if torch is None:
        raise ImportError("PyTorch is required for detection.")

    with open(paths.power_outputs_dir / "baseline_metadata.json") as f:
        baseline_meta = json.load(f)
    baseline_model = LSTMAutoencoder(
        input_size=baseline_meta["input_size"],
        hidden_size=baseline_meta["hidden_size"],
        latent_size=baseline_meta["latent_size"],
    )
    baseline_model.load_state_dict(
        torch.load(paths.power_outputs_dir / "baseline_model.pt", map_location="cpu")
    )
    baseline_model.eval()
    baseline_scaler = joblib.load(paths.power_outputs_dir / "baseline_scaler.pkl")

    with open(paths.power_phase_outputs_dir / "phase_metadata.json") as f:
        phase_meta = json.load(f)
    phase_model = LSTMAutoencoder(
        input_size=phase_meta["input_size"],
        hidden_size=phase_meta["hidden_size"],
        latent_size=phase_meta["latent_size"],
    )
    phase_model.load_state_dict(
        torch.load(paths.power_phase_outputs_dir / "phase_model.pt", map_location="cpu")
    )
    phase_model.eval()
    phase_scaler = joblib.load(paths.power_phase_outputs_dir / "phase_scaler.pkl")

    iforest_models: dict = {}
    iforest_thresholds: dict = {}
    iforest_feature_cols: dict = {}
    iforest_col_indices: dict = {}
    if paths.iforest_model_file.exists() and paths.iforest_metadata_file.exists():
        iforest_models = joblib.load(paths.iforest_model_file)
        with open(paths.iforest_metadata_file) as f:
            _if_meta = json.load(f)
        iforest_thresholds = {cat: v["threshold"] for cat, v in _if_meta.items()}
        iforest_feature_cols = {
            cat: v.get("feature_cols", list(BASELINE_UPS_FEATURES))
            for cat, v in _if_meta.items()
        }
        _b_col_list = list(BASELINE_UPS_FEATURES)
        iforest_col_indices = {
            cat: [_b_col_list.index(c) for c in cols if c in _b_col_list]
            for cat, cols in iforest_feature_cols.items()
        }

    return _ModelArtifacts(
        baseline_model=baseline_model,
        baseline_scaler=baseline_scaler,
        baseline_meta=baseline_meta,
        phase_model=phase_model,
        phase_scaler=phase_scaler,
        phase_meta=phase_meta,
        iforest_models=iforest_models,
        iforest_thresholds=iforest_thresholds,
        iforest_feature_cols=iforest_feature_cols,
        iforest_col_indices=iforest_col_indices,
    )


class DualModelScorer:
    """Loads all power model artifacts and exposes unified window-scoring methods.

    Both the CSV replay path (dual_model_scorer.py) and the Kafka streaming path
    (power_stream_processor.py) use this class so that scoring logic lives in one place.
    """

    def __init__(self, paths: ProjectPaths | None = None) -> None:
        self._arts = _load_artifacts(paths or ProjectPaths())
        self._loss_fn = torch.nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    @property
    def baseline_meta(self) -> dict:
        return self._arts.baseline_meta

    @property
    def phase_meta(self) -> dict:
        return self._arts.phase_meta

    @property
    def baseline_scaler(self) -> Any:
        return self._arts.baseline_scaler

    @property
    def phase_scaler(self) -> Any:
        return self._arts.phase_scaler

    @property
    def iforest_feature_cols(self) -> dict:
        return self._arts.iforest_feature_cols

    # ------------------------------------------------------------------ #
    # LSTM window scoring                                                  #
    # ------------------------------------------------------------------ #

    def score_window(
        self,
        model_key: str,
        window: np.ndarray,
        feature_names: list[str],
        mse_exclude: frozenset[str] = frozenset(),
    ) -> dict:
        """Score one (seq_len, n_features) window.

        model_key: "baseline" or "phase"

        Returns dict with keys:
          mean_error (float), timestep_errors (list[float]),
          peak_timestep_idx (int), top_features (list[str])
        """
        arts = self._arts
        if model_key == "baseline":
            model = arts.baseline_model
            normal_errors = arts.baseline_meta.get("normal_feature_errors", {})
            normal_stds = arts.baseline_meta.get("normal_feature_error_stds", {})
        else:
            model = arts.phase_model
            normal_errors = arts.phase_meta.get("normal_feature_errors", {})
            normal_stds = arts.phase_meta.get("normal_feature_error_stds", {})

        tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            recon = model(tensor)
            errors = self._loss_fn(recon, tensor).squeeze(0).cpu().numpy()

        if mse_exclude:
            active_cols = [j for j, f in enumerate(feature_names) if f not in mse_exclude]
            errors_active = errors[:, active_cols]
        else:
            errors_active = errors

        timestep_errors = errors_active.mean(axis=1)
        feature_errors = errors.mean(axis=0)

        surprise = np.clip(
            [
                (feature_errors[j] - normal_errors.get(f, 0.0))
                / (normal_stds.get(f, _EPS) + _EPS)
                for j, f in enumerate(feature_names)
            ],
            0.0, None,
        )
        for j, f in enumerate(feature_names):
            if f in ATTRIBUTION_EXCLUDE:
                surprise[j] = 0.0

        peak_idx = int(np.argmax(timestep_errors))
        top_features = [
            feature_names[j]
            for j in np.argsort(surprise)[::-1][:3]
            if j < len(feature_names) and surprise[j] > 0.0
        ]
        return {
            "mean_error": float(errors_active.mean()),
            "timestep_errors": [round(float(e), 6) for e in timestep_errors],
            "peak_timestep_idx": peak_idx,
            "top_features": top_features,
        }

    # ------------------------------------------------------------------ #
    # IForest scoring                                                      #
    # ------------------------------------------------------------------ #

    def score_iforest_row(
        self,
        b_scaled: list[float],
        device_category: str,
    ) -> tuple[float, float, int]:
        """Score a single scaled baseline row. Returns (score, threshold, flag)."""
        clf = self._arts.iforest_models.get(device_category)
        if clf is None:
            return 0.0, 0.0, 0
        col_indices = self._arts.iforest_col_indices.get(
            device_category, list(range(len(BASELINE_UPS_FEATURES)))
        )
        vec = np.array([[b_scaled[j] for j in col_indices]])
        score = float(clf.score_samples(vec)[0])
        threshold = self._arts.iforest_thresholds.get(device_category, -0.5)
        score_ratio = (score / threshold) if threshold != 0 else 0.0
        flag = int(score < threshold and score_ratio >= IF_MIN_SCORE_RATIO)
        return score, threshold, flag

    def score_iforest_batch(
        self,
        baseline_vals: np.ndarray,
        device_category: str,
    ) -> tuple[np.ndarray | None, float, list[int], list[str]]:
        """Batch-score all rows for one device (E5 optimisation).

        Returns (scores_or_None, threshold, col_indices, cat_cols).
        scores is a (n_rows,) float array; None when no model exists for this category.
        """
        clf = self._arts.iforest_models.get(device_category)
        if clf is None:
            return None, 0.0, [], []
        col_indices = self._arts.iforest_col_indices.get(
            device_category, list(range(len(BASELINE_UPS_FEATURES)))
        )
        cat_cols = self._arts.iforest_feature_cols.get(
            device_category, list(BASELINE_UPS_FEATURES)
        )
        threshold = self._arts.iforest_thresholds.get(device_category, -0.5)
        scores = clf.score_samples(baseline_vals[:, col_indices])
        return scores, threshold, col_indices, cat_cols


# ---------------------------------------------------------------------------
# DualModelResult — shared result type for both CSV and Kafka inference paths
# ---------------------------------------------------------------------------

@dataclass
class DualModelResult:
    device_id: str
    device_category: str        # ups | pdu | network | env
    vendor: str                 # apc | liebert | raritan | cisco | generic
    phase_count: int            # 1 or 3
    window_start: str
    window_end: str
    baseline_error: float
    baseline_threshold: float
    baseline_anomaly: int
    phase_error: float
    phase_threshold: float
    phase_anomaly: int
    combined_flag: int          # determined by alert policy (OR/AND of LSTM signals)
    overload_rule_flag: int     # rule: output_load_pct > 100 in any window timestep
    final_flag: int             # combined_flag OR overload_rule_flag
    alert_policy: str
    # Per-sequence diagnostics
    window_timestamps: list[str] = field(default_factory=list)
    baseline_timestep_errors: list[float] = field(default_factory=list)
    baseline_peak_timestep: str = ""
    baseline_top_features: list[str] = field(default_factory=list)
    phase_timestep_errors: list[float] = field(default_factory=list)
    phase_peak_timestep: str = ""
    phase_top_features: list[str] = field(default_factory=list)
    iforest_score: float = 0.0
    iforest_threshold: float = 0.0
    iforest_flag: int = 0
    iforest_peak_timestep: str = ""
    iforest_top_features: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSV detection — batch replay of synthetic_power_snmp_dataset.csv
# ---------------------------------------------------------------------------

def run_csv_detection(
    inference_config: PowerInferenceConfig | None = None,
    paths: ProjectPaths | None = None,
) -> list[DualModelResult]:
    from snmp_anomaly_detection.inference.event_preprocessor import EventPreprocessor
    from snmp_anomaly_detection.inference.events import PowerEvent
    from snmp_anomaly_detection.preprocessing.power_features import load_power_dataset

    inference_config = inference_config or PowerInferenceConfig()
    paths = paths or ProjectPaths()

    scorer = DualModelScorer(paths)
    baseline_meta = scorer.baseline_meta
    phase_meta = scorer.phase_meta
    baseline_seq_len = baseline_meta["seq_len"]
    phase_seq_len = phase_meta["seq_len"]
    baseline_cat_thresholds = baseline_meta.get("per_category_thresholds", {})
    phase_cat_thresholds = phase_meta.get("per_category_thresholds", {})

    df = load_power_dataset(paths)
    preprocessor = EventPreprocessor()

    results: list[DualModelResult] = []
    policy = inference_config.alert_policy

    for device_id in df["device_id"].unique():
        device_df = df[df["device_id"] == device_id].copy()

        device_category = str(device_df["device_category"].iloc[0]) if "device_category" in device_df.columns else "unknown"
        vendor = str(device_df["vendor"].iloc[0]) if "vendor" in device_df.columns else "unknown"
        phase_count = int(device_df["phase_count"].iloc[0]) if "phase_count" in device_df.columns else 1

        b_threshold = baseline_cat_thresholds.get(device_category, baseline_meta["threshold"])
        p_threshold = phase_cat_thresholds.get(device_category, phase_meta["threshold"])

        timestamps = device_df["timestamp"].astype(str).values
        raw_load_pct = device_df["output_load_pct"].values if "output_load_pct" in device_df.columns else None

        preprocessor.reset_device(device_id)
        numeric_cols = [c for c in device_df.select_dtypes(include=[np.number]).columns if c != "anomaly"]
        preprocessed_rows: list[dict] = []
        phase_rows: list[dict] = []
        for _, row in device_df.iterrows():
            event = PowerEvent(
                timestamp=row["timestamp"],
                device_id=device_id,
                device_category=device_category,
                vendor=vendor,
                phase_count=phase_count,
                feature_values={col: float(row[col]) for col in numeric_cols if pd.notna(row[col])},
                rated_capacity_w=float(row.get("rated_capacity_w", 0.0)),
                nominal_voltage_v=float(row.get("nominal_voltage_v", 120.0)),
                rated_battery_v=float(row.get("rated_battery_v", 0.0)),
            )
            preprocessor.preprocess(event)
            preprocessed_rows.append(dict(event.feature_values))
            if device_category in PHASE_MODEL_CATEGORIES:
                preprocessor.preprocess_phase(event)
            phase_rows.append(dict(event.feature_values))

        preprocessed_df = pd.DataFrame(preprocessed_rows)
        phase_df = pd.DataFrame(phase_rows)
        baseline_cols = [c for c in BASELINE_UPS_FEATURES if c in preprocessed_df.columns]
        phase_cols = [c for c in PHASE_LEVEL_FEATURES if c in phase_df.columns]

        baseline_vals = scorer.baseline_scaler.transform(preprocessed_df[baseline_cols])
        phase_vals = scorer.phase_scaler.transform(phase_df[phase_cols])

        device_if_scores, device_if_threshold, device_if_col_indices, device_if_cat_cols = (
            scorer.score_iforest_batch(baseline_vals, device_category)
        )

        seq_len = max(baseline_seq_len, phase_seq_len)
        for i in range(len(device_df) - seq_len):
            b_win = baseline_vals[i : i + baseline_seq_len]
            p_win = phase_vals[i : i + phase_seq_len]
            win_timestamps = timestamps[i : i + seq_len].tolist()

            mse_excl = NON_UPS_MSE_EXCLUDE if device_category != "ups" else frozenset()
            b_detail = scorer.score_window("baseline", b_win, baseline_cols, mse_excl)

            if device_category in PHASE_MODEL_CATEGORIES:
                p_detail = scorer.score_window("phase", p_win, phase_cols, mse_excl)
                p_err = p_detail["mean_error"]
                p_flag = int(p_err > p_threshold)
            else:
                p_detail = {"mean_error": 0.0, "peak_timestep_idx": 0, "top_features": [], "timestep_errors": []}
                p_err = 0.0
                p_flag = 0

            b_err = b_detail["mean_error"]
            b_flag = int(b_err > b_threshold)
            combined = int(b_flag or p_flag) if policy == "or" else int(b_flag and p_flag)

            if_score = 0.0
            if_threshold = 0.0
            if_flag = 0
            if_peak_timestep = ""
            if_top_features: list[str] = []
            if device_if_scores is not None:
                win_scores = device_if_scores[i : i + baseline_seq_len]
                if_peak_idx = int(np.argmin(win_scores))
                if_score = float(win_scores[if_peak_idx])
                if_peak_timestep = win_timestamps[if_peak_idx]
                if_threshold = device_if_threshold
                score_ratio = (if_score / if_threshold) if if_threshold != 0 else 0.0
                if_flag = int(if_score < if_threshold and score_ratio >= IF_MIN_SCORE_RATIO)
                abs_devs = np.abs(baseline_vals[i + if_peak_idx])
                top_idx = np.argsort(abs_devs)[::-1]
                if_top_features = [
                    baseline_cols[j]
                    for j in top_idx[:5]
                    if abs_devs[j] > 0.5
                    and baseline_cols[j] not in ATTRIBUTION_EXCLUDE
                    and baseline_cols[j] in device_if_cat_cols
                ][:3]

            overload_rule = 0
            if raw_load_pct is not None:
                win_load = raw_load_pct[i : i + seq_len]
                if win_load.max() > inference_config.overload_load_pct_threshold:
                    overload_rule = 1

            final = int(combined or overload_rule or if_flag)
            b_peak_idx = b_detail["peak_timestep_idx"]
            p_peak_idx = p_detail["peak_timestep_idx"]

            results.append(DualModelResult(
                device_id=device_id,
                device_category=device_category,
                vendor=vendor,
                phase_count=phase_count,
                window_start=win_timestamps[0],
                window_end=win_timestamps[-1],
                baseline_error=round(b_err, 6),
                baseline_threshold=b_threshold,
                baseline_anomaly=b_flag,
                phase_error=round(p_err, 6),
                phase_threshold=p_threshold,
                phase_anomaly=p_flag,
                combined_flag=combined,
                overload_rule_flag=overload_rule,
                final_flag=final,
                alert_policy=policy,
                window_timestamps=win_timestamps,
                baseline_timestep_errors=b_detail["timestep_errors"],
                baseline_peak_timestep=win_timestamps[b_peak_idx],
                baseline_top_features=b_detail["top_features"],
                phase_timestep_errors=p_detail["timestep_errors"],
                phase_peak_timestep=win_timestamps[p_peak_idx],
                phase_top_features=p_detail["top_features"],
                iforest_score=round(if_score, 6),
                iforest_threshold=round(if_threshold, 6),
                iforest_flag=if_flag,
                iforest_peak_timestep=if_peak_timestep,
                iforest_top_features=if_top_features,
            ))

    return results


# ---------------------------------------------------------------------------
# Reporting helpers and save_dual_results
# ---------------------------------------------------------------------------

def _collapse_to_events(flagged: list[DualModelResult]) -> list[DualModelResult]:
    """Merge overlapping flagged windows per device into one event (highest combined error)."""
    if not flagged:
        return []

    events: list[DualModelResult] = []
    by_device: dict[str, list[DualModelResult]] = {}
    for r in flagged:
        by_device.setdefault(r.device_id, []).append(r)

    for rows in by_device.values():
        group: list[DualModelResult] = [rows[0]]
        for prev, curr in zip(rows, rows[1:]):
            if pd.Timestamp(curr.window_start) <= pd.Timestamp(prev.window_end):
                group.append(curr)
            else:
                events.append(max(group, key=lambda r: r.baseline_error + r.phase_error))
                group = [curr]
        events.append(max(group, key=lambda r: r.baseline_error + r.phase_error))

    return events


def _triggered_by(r: DualModelResult) -> str:
    if r.overload_rule_flag:
        return "overload_rule"
    if r.baseline_anomaly and r.phase_anomaly:
        return "both_models"
    if r.baseline_anomaly:
        return "baseline_model"
    if r.phase_anomaly:
        return "phase_model"
    if r.iforest_flag:
        return "iforest_only"
    return "unknown"


def _severity(max_ratio: float) -> str:
    if max_ratio >= 3.0:
        return "critical"
    if max_ratio >= 1.5:
        return "high"
    return "medium"


def save_dual_results(results: list[DualModelResult], paths: ProjectPaths, *, silent: bool = False) -> None:
    from dataclasses import asdict
    paths.ensure_power_directories()

    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    csv_path = paths.power_dual_outputs_dir / "anomaly_results.csv"
    df.to_csv(csv_path, index=False)

    by_device: dict = {}
    for r in results:
        entry = by_device.setdefault(r.device_id, {
            "device_id": r.device_id,
            "device_category": r.device_category,
            "vendor": r.vendor,
            "phase_count": r.phase_count,
            "total_windows": 0,
            "flagged_baseline": 0,
            "flagged_phase": 0,
            "flagged_combined": 0,
            "flagged_iforest": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_baseline"] += r.baseline_anomaly
        entry["flagged_phase"] += r.phase_anomaly
        entry["flagged_overload_rule"] = entry.get("flagged_overload_rule", 0) + r.overload_rule_flag
        entry["flagged_combined"] += r.combined_flag
        entry["flagged_iforest"] += r.iforest_flag
        entry["flagged_final"] = entry.get("flagged_final", 0) + r.final_flag

    by_vendor: dict = {}
    for r in results:
        entry = by_vendor.setdefault(r.vendor, {
            "vendor": r.vendor,
            "total_windows": 0,
            "flagged_final": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_final"] += r.final_flag

    by_category: dict = {}
    for r in results:
        entry = by_category.setdefault(r.device_category, {
            "device_category": r.device_category,
            "total_windows": 0,
            "flagged_iforest": 0,
            "flagged_final": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_iforest"] += r.iforest_flag
        entry["flagged_final"] += r.final_flag

    flagged = [r for r in results if r.final_flag == 1]
    events = _collapse_to_events(flagged)

    summary = {
        "alert_policy": results[0].alert_policy if results else "or",
        "total_windows": len(results),
        "flagged_by_baseline": sum(r.baseline_anomaly for r in results),
        "flagged_by_phase": sum(r.phase_anomaly for r in results),
        "flagged_by_overload_rule": sum(r.overload_rule_flag for r in results),
        "flagged_by_iforest": sum(r.iforest_flag for r in results),
        "flagged_combined": sum(r.combined_flag for r in results),
        "flagged_final": sum(r.final_flag for r in results),
        "total_anomaly_events": len(events),
        "by_device": list(by_device.values()),
        "by_vendor": list(by_vendor.values()),
        "by_category": list(by_category.values()),
    }
    with open(paths.power_dual_outputs_dir / "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rul_map: dict = {}
    rul_path = paths.battery_rul_outputs_dir / "rul_predictions.json"
    if rul_path.exists():
        with open(rul_path) as f:
            for p in json.load(f):
                rul_map[p["device_id"]] = p

    detail_rows = []
    for window_number, r in enumerate(flagged, start=1):
        b_ratio = round(r.baseline_error / r.baseline_threshold, 2) if r.baseline_threshold else 0.0
        p_ratio = round(r.phase_error / r.phase_threshold, 2) if r.phase_threshold else 0.0

        rul_entry = None
        if r.device_category == "ups" and r.device_id in rul_map:
            p = rul_map[r.device_id]
            rul_entry = {
                "predicted_days": p["predicted_rul_days"],
                "confidence_interval_lo": p["confidence_interval_lo"],
                "confidence_interval_hi": p["confidence_interval_hi"],
                "advisory": p["replacement_advisory"],
                "as_of": p["timestamp"],
            }

        _total_ts = len(r.window_timestamps)
        b_breach_ts = [
            ts for ts, err in zip(r.window_timestamps, r.baseline_timestep_errors)
            if err > r.baseline_threshold
        ]
        p_breach_ts = [
            ts for ts, err in zip(r.window_timestamps, r.phase_timestep_errors)
            if err > r.phase_threshold
        ]

        detail_rows.append({
            "window_number": window_number,
            "device_id": r.device_id,
            "device_category": r.device_category,
            "vendor": r.vendor,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "triggered_by": _triggered_by(r),
            "alert_policy": r.alert_policy,
            "severity": _severity(max(b_ratio, p_ratio)),
            "baseline": {
                "error": r.baseline_error,
                "threshold": r.baseline_threshold,
                "sequence_accuracy_pct": round((_total_ts - len(b_breach_ts)) / _total_ts * 100, 1) if _total_ts else 0.0,
                "flagged": bool(r.baseline_anomaly),
                "peak_timestep": r.baseline_peak_timestep,
                "top_features": r.baseline_top_features,
                "breach_timesteps": b_breach_ts,
            },
            "phase": {
                "error": r.phase_error,
                "threshold": r.phase_threshold,
                "sequence_accuracy_pct": round((_total_ts - len(p_breach_ts)) / _total_ts * 100, 1) if _total_ts else 0.0,
                "flagged": bool(r.phase_anomaly),
                "peak_timestep": r.phase_peak_timestep,
                "top_features": r.phase_top_features,
                "breach_timesteps": p_breach_ts,
            },
            "iforest": {
                "score": r.iforest_score,
                "threshold": r.iforest_threshold,
                "score_ratio": round(r.iforest_score / r.iforest_threshold, 2) if r.iforest_threshold else 0.0,
                "health_pct": round(max(0.0, (1 - abs(r.iforest_score / r.iforest_threshold))) * 100, 1) if r.iforest_threshold else 0.0,
                "flagged": bool(r.iforest_flag),
                "peak_timestep": r.iforest_peak_timestep,
                "top_features": r.iforest_top_features,
            },
            "rul": rul_entry,
        })
    detail_path = paths.power_dual_outputs_dir / "anomaly_windows_detail.json"
    with open(detail_path, "w") as f:
        json.dump(detail_rows, f, indent=2)

    if not silent:
        print(f"Results:        {csv_path}")
        print(f"Window detail:  {detail_path}")
        print(f"Windows flagged (LSTM):    {summary['flagged_combined']} / {summary['total_windows']}")
        print(f"Windows flagged (rule):    {summary['flagged_by_overload_rule']} / {summary['total_windows']}")
        print(f"Windows flagged (final):   {summary['flagged_final']} / {summary['total_windows']}")
        print(f"Anomaly events (deduped):  {summary['total_anomaly_events']}")
        print("\nFlagged by device:")
        for d in summary["by_device"]:
            print(f"  {d['device_id']:15s} [{d['device_category']:7s} / {d['vendor']:8s}]  flagged={d.get('flagged_final',0):4d} / {d['total_windows']}")

        if flagged:
            print(f"\nSample flagged windows (first 5 of {len(flagged)}):")
            for r in flagged[:5]:
                print(f"\n  [{r.device_id}] {r.window_start} → {r.window_end}")
                if r.baseline_anomaly:
                    print(f"    baseline: error={r.baseline_error:.6f} > thresh={r.baseline_threshold:.6f}")
                    print(f"             peak_at={r.baseline_peak_timestep}  top_features={r.baseline_top_features}")
                if r.phase_anomaly:
                    print(f"    phase:    error={r.phase_error:.6f} > thresh={r.phase_threshold:.6f}")
                    print(f"             peak_at={r.phase_peak_timestep}  top_features={r.phase_top_features}")
