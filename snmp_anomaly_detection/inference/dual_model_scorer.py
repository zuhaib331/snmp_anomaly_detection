"""Dual-model scorer: combines baseline LSTM and phase-level LSTM for power anomaly detection.

Alert policies:
  or  — either model exceeds its threshold → alert (maximizes recall)
  and — both models must exceed threshold → alert (maximizes precision)

Also serves as the entry point for detect-power-csv CLI command.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import (
    BASELINE_UPS_FEATURES,
    PHASE_LEVEL_FEATURES,
    PHASE_MODEL_CATEGORIES,
    IF_MIN_SCORE_RATIO,
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.preprocessing.power_features import (
    load_power_dataset,
    aggregate_phase_metrics,
    normalize_absolute_features,
    apply_log1p_skewed,
    add_delta_features,
    create_sequences,
)
from snmp_anomaly_detection.preprocessing.phase_features import (
    enrich_imbalance_features,
    _IMBALANCE_COLS,
    _IMBALANCE_CLIP_MAX,
)


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
    true_label: int = 0
    anomaly_types: list[str] = field(default_factory=list)
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


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for detection.")


def _load_model(model_path: Path, meta: dict) -> "LSTMAutoencoder":
    model = LSTMAutoencoder(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        latent_size=meta["latent_size"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# Features excluded from attribution — confirmed zero ground-truth deviation
_ATTRIBUTION_EXCLUDE: frozenset[str] = frozenset({"input_frequency_hz"})

# output_frequency_hz is the UPS inverter output frequency — a real fault signal for UPS.
# For PDU/network/env it is ambient line frequency (grid noise), not a device health indicator.
# Exclude its MSE contribution for non-UPS categories to prevent scaler-amplified false positives.
_NON_UPS_MSE_EXCLUDE: frozenset[str] = frozenset({"output_frequency_hz"})


def _score_window_detailed(
    model,
    window: np.ndarray,
    feature_names: list[str],
    normal_feature_errors: dict[str, float],
    normal_feature_error_stds: dict[str, float],
    mse_exclude: frozenset[str] = frozenset(),
) -> dict:
    loss_fn = torch.nn.MSELoss(reduction="none")
    tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)  # (1, seq_len, n_feat)
    with torch.no_grad():
        recon = model(tensor)
        errors = loss_fn(recon, tensor).squeeze(0).cpu().numpy()  # (seq_len, n_feat)

    # Active columns exclude mse_exclude features from mean_error and timestep_errors.
    # feature_errors keeps all columns so attribution z-scores remain accurate.
    if mse_exclude:
        active_cols = [j for j, f in enumerate(feature_names) if f not in mse_exclude]
        errors_active = errors[:, active_cols]
    else:
        errors_active = errors

    timestep_errors = errors_active.mean(axis=1)   # (seq_len,)
    feature_errors = errors.mean(axis=0)            # (n_feat,) — full, for attribution

    # Z-score surprise: how many std-devs above the normal reconstruction error
    # Handles high-variance features correctly; clipped to 0 (only care about worse-than-normal)
    _EPS = 1e-9
    surprise = np.clip(
        [
            (feature_errors[j] - normal_feature_errors.get(f, 0.0))
            / (normal_feature_error_stds.get(f, _EPS) + _EPS)
            for j, f in enumerate(feature_names)
        ],
        0.0, None,
    )

    # Fix 1: zero out known-noise features so they never rank as top cause
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
        "mean_error": float(errors_active.mean()),
        "timestep_errors": [round(float(e), 6) for e in timestep_errors],
        "peak_timestep_idx": peak_idx,
        "top_features": top_features,
    }


def run_dual_detection(
    inference_config: PowerInferenceConfig | None = None,
    paths: ProjectPaths | None = None,
) -> list[DualModelResult]:
    _require_torch()
    inference_config = inference_config or PowerInferenceConfig()
    paths = paths or ProjectPaths()

    # Load baseline artifacts
    with open(paths.power_outputs_dir / "baseline_metadata.json") as f:
        baseline_meta = json.load(f)
    baseline_model = _load_model(paths.power_outputs_dir / "baseline_model.pt", baseline_meta)
    baseline_scaler = joblib.load(paths.power_outputs_dir / "baseline_scaler.pkl")
    baseline_threshold = baseline_meta["threshold"]
    baseline_seq_len = baseline_meta["seq_len"]
    baseline_normal_errors = baseline_meta.get("normal_feature_errors", {})
    baseline_normal_stds   = baseline_meta.get("normal_feature_error_stds", {})
    baseline_cat_thresholds = baseline_meta.get("per_category_thresholds", {})

    # Load phase artifacts
    with open(paths.power_phase_outputs_dir / "phase_metadata.json") as f:
        phase_meta = json.load(f)
    phase_model = _load_model(paths.power_phase_outputs_dir / "phase_model.pt", phase_meta)
    phase_scaler = joblib.load(paths.power_phase_outputs_dir / "phase_scaler.pkl")
    phase_threshold = phase_meta["threshold"]
    phase_seq_len = phase_meta["seq_len"]
    phase_normal_errors = phase_meta.get("normal_feature_errors", {})
    phase_normal_stds   = phase_meta.get("normal_feature_error_stds", {})
    phase_cat_thresholds = phase_meta.get("per_category_thresholds", {})

    # Load IF artifacts (optional — scoring is skipped gracefully if not trained yet)
    iforest_models: dict = {}
    iforest_thresholds: dict = {}
    iforest_feature_cols: dict = {}
    if paths.iforest_model_file.exists() and paths.iforest_metadata_file.exists():
        iforest_models = joblib.load(paths.iforest_model_file)
        with open(paths.iforest_metadata_file) as f:
            _if_meta = json.load(f)
        iforest_thresholds = {cat: v["threshold"] for cat, v in _if_meta.items()}
        # Per-category feature lists saved at training time (E2).
        # Falls back to all baseline_cols for models trained before E2.
        iforest_feature_cols = {cat: v["feature_cols"] for cat, v in _if_meta.items() if "feature_cols" in v}

    # Load and preprocess dataset
    df = load_power_dataset(paths)
    df = aggregate_phase_metrics(df)
    df = normalize_absolute_features(df)
    df = apply_log1p_skewed(df)
    df = add_delta_features(df)
    df = enrich_imbalance_features(df)

    baseline_cols = [c for c in BASELINE_UPS_FEATURES if c in df.columns]
    phase_cols = [c for c in PHASE_LEVEL_FEATURES if c in df.columns]

    results: list[DualModelResult] = []
    policy = inference_config.alert_policy

    for device_id in df["device_id"].unique():
        device_df = df[df["device_id"] == device_id].copy()

        # Pull device identity fields from the first row
        device_category = str(device_df["device_category"].iloc[0]) if "device_category" in device_df.columns else "unknown"
        vendor = str(device_df["vendor"].iloc[0]) if "vendor" in device_df.columns else "unknown"
        phase_count = int(device_df["phase_count"].iloc[0]) if "phase_count" in device_df.columns else 1

        # Use per-category threshold when available (tighter for env/network, wider for ups)
        b_threshold = baseline_cat_thresholds.get(device_category, baseline_threshold)
        p_threshold = phase_cat_thresholds.get(device_category, phase_threshold)

        # Apply imbalance clipping before phase scaling (same as training transform)
        phase_df = device_df.copy()
        for col in _IMBALANCE_COLS:
            if col in phase_df.columns:
                phase_df[col] = phase_df[col].clip(0.0, _IMBALANCE_CLIP_MAX) / _IMBALANCE_CLIP_MAX

        # Scale baseline and phase columns separately to avoid overwrite
        # (PHASE_LEVEL_FEATURES ⊃ BASELINE_UPS_FEATURES, so they must not share a df)
        baseline_vals = baseline_scaler.transform(device_df[baseline_cols])
        phase_vals = phase_scaler.transform(phase_df[phase_cols])

        timestamps = device_df["timestamp"].astype(str).values
        labels = device_df["anomaly"].values
        anomaly_types = device_df.get("anomaly_type", pd.Series(["none"] * len(device_df))).values
        # Raw output_load_pct (unscaled) — used for rule-based overload detection.
        # Normal operation: output_load_pct ≤ 100.  Any reading > 100 is definitively overload.
        raw_load_pct = device_df["output_load_pct"].values if "output_load_pct" in device_df.columns else None

        # Pre-score all rows for this device in one batch call so the window loop
        # only needs to slice and index — avoids ~2000 individual sklearn calls per device.
        clf = iforest_models.get(device_category)
        device_if_scores: np.ndarray | None = None
        device_if_threshold = 0.0
        device_if_col_indices: list[int] = []
        device_if_cat_cols: list[str] = []
        if clf is not None:
            _cat_cols = iforest_feature_cols.get(device_category, baseline_cols)
            _col_indices = [baseline_cols.index(c) for c in _cat_cols]
            device_if_scores = clf.score_samples(baseline_vals[:, _col_indices])
            device_if_threshold = iforest_thresholds.get(device_category, -0.5)
            device_if_col_indices = _col_indices
            device_if_cat_cols = _cat_cols

        seq_len = max(baseline_seq_len, phase_seq_len)
        for i in range(len(device_df) - seq_len):
            b_win = baseline_vals[i : i + baseline_seq_len]
            p_win = phase_vals[i : i + phase_seq_len]
            win_timestamps = timestamps[i : i + seq_len].tolist()

            mse_excl = _NON_UPS_MSE_EXCLUDE if device_category != "ups" else frozenset()
            b_detail = _score_window_detailed(baseline_model, b_win, baseline_cols, baseline_normal_errors, baseline_normal_stds, mse_excl)

            if device_category in PHASE_MODEL_CATEGORIES:
                p_detail = _score_window_detailed(phase_model, p_win, phase_cols, phase_normal_errors, phase_normal_stds, mse_excl)
                p_err = p_detail["mean_error"]
                p_flag = int(p_err > p_threshold)
            else:
                p_detail = {"mean_error": 0.0, "peak_timestep_idx": 0, "top_features": [], "timestep_errors": []}
                p_err = 0.0
                p_flag = 0

            b_err = b_detail["mean_error"]
            b_flag = int(b_err > b_threshold)

            if policy == "or":
                combined = int(b_flag or p_flag)
            else:
                combined = int(b_flag and p_flag)

            # Isolation Forest: slice pre-computed per-row scores for this window,
            # take the minimum (most anomalous timestep).
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
                # Attribution: full (unfiltered) vector at the peak timestep.
                abs_devs = np.abs(baseline_vals[i + if_peak_idx])
                top_idx = np.argsort(abs_devs)[::-1]
                if_top_features = [
                    baseline_cols[j]
                    for j in top_idx[:5]
                    if abs_devs[j] > 0.5
                    and baseline_cols[j] not in _ATTRIBUTION_EXCLUDE
                    and baseline_cols[j] in device_if_cat_cols
                ][:3]

            # Rule-based overload: output_load_pct > 100 is physically impossible during normal
            # operation (synthetic data caps normal at 100). Zero false-positive supplement.
            overload_rule = 0
            if raw_load_pct is not None:
                win_load = raw_load_pct[i : i + seq_len]
                if win_load.max() > 100.0:
                    overload_rule = 1

            final = int(combined or overload_rule or if_flag)

            true_label = int(labels[i : i + seq_len].max())
            seen_types = list({t for t in anomaly_types[i : i + seq_len] if t != "none"})

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
                true_label=true_label,
                anomaly_types=seen_types,
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


def _collapse_to_events(flagged: list[DualModelResult]) -> list[DualModelResult]:
    """Merge overlapping flagged windows per device into one event.

    With stride=1 a single anomaly produces ~seq_len consecutive flagged windows.
    This collapses them into one representative (highest combined error) so that
    summary counts and the detail JSON reflect real events, not window counts.
    """
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
    paths.ensure_power_directories()
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    csv_path = paths.power_dual_outputs_dir / "anomaly_results.csv"
    df.to_csv(csv_path, index=False)

    # --- Per-device breakdown ---
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
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_baseline"] += r.baseline_anomaly
        entry["flagged_phase"] += r.phase_anomaly
        entry["flagged_overload_rule"] = entry.get("flagged_overload_rule", 0) + r.overload_rule_flag
        entry["flagged_combined"] += r.combined_flag
        entry["flagged_iforest"] += r.iforest_flag
        entry["flagged_final"] = entry.get("flagged_final", 0) + r.final_flag
        entry["true_anomaly_windows"] += r.true_label

    # --- Per-vendor breakdown ---
    by_vendor: dict = {}
    for r in results:
        entry = by_vendor.setdefault(r.vendor, {
            "vendor": r.vendor,
            "total_windows": 0,
            "flagged_final": 0,
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_final"] += r.final_flag
        entry["true_anomaly_windows"] += r.true_label

    # --- Per-category breakdown ---
    by_category: dict = {}
    for r in results:
        entry = by_category.setdefault(r.device_category, {
            "device_category": r.device_category,
            "total_windows": 0,
            "flagged_iforest": 0,
            "flagged_final": 0,
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_iforest"] += r.iforest_flag
        entry["flagged_final"] += r.final_flag
        entry["true_anomaly_windows"] += r.true_label

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
        "true_anomaly_windows": sum(r.true_label for r in results),
        "total_anomaly_events": len(events),
        "true_anomaly_events": sum(r.true_label for r in events),
        "by_device": list(by_device.values()),
        "by_vendor": list(by_vendor.values()),
        "by_category": list(by_category.values()),
    }
    with open(paths.power_dual_outputs_dir / "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Load RUL predictions for UPS join ---
    rul_map: dict = {}
    rul_path = paths.battery_rul_outputs_dir / "rul_predictions.json"
    if rul_path.exists():
        with open(rul_path) as f:
            for p in json.load(f):
                rul_map[p["device_id"]] = p

    # --- Flagged event detail JSON (one entry per real anomaly event, duplicates collapsed) ---
    detail_rows = []
    for event_number, r in enumerate(events, start=1):
        duration_minutes = int(
            (pd.Timestamp(r.window_end) - pd.Timestamp(r.window_start)).total_seconds() / 60
        )
        b_ratio = round(r.baseline_error / r.baseline_threshold, 2) if r.baseline_threshold else 0.0
        p_ratio = round(r.phase_error / r.phase_threshold, 2) if r.phase_threshold else 0.0
        ground_truth_type = r.anomaly_types[0] if r.anomaly_types else None

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

        detail_rows.append({
            "event_number": event_number,
            "device_id": r.device_id,
            "device_category": r.device_category,
            "vendor": r.vendor,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "duration_minutes": duration_minutes,
            "triggered": bool(r.final_flag),
            "triggered_by": _triggered_by(r),
            "alert_policy": r.alert_policy,
            "severity": _severity(max(b_ratio, p_ratio)),
            "ground_truth_type": ground_truth_type,
            "true_positive": bool(r.final_flag) and ground_truth_type is not None,
            "baseline": {
                "error": r.baseline_error,
                "threshold": r.baseline_threshold,
                "error_ratio": b_ratio,
                "flagged": bool(r.baseline_anomaly),
                "peak_timestep": r.baseline_peak_timestep,
                "top_features": r.baseline_top_features,
                "breach_timesteps": [
                    ts for ts, err in zip(r.window_timestamps, r.baseline_timestep_errors)
                    if err > r.baseline_threshold
                ],
            },
            "phase": {
                "error": r.phase_error,
                "threshold": r.phase_threshold,
                "error_ratio": p_ratio,
                "flagged": bool(r.phase_anomaly),
                "peak_timestep": r.phase_peak_timestep,
                "top_features": r.phase_top_features,
                "breach_timesteps": [
                    ts for ts, err in zip(r.window_timestamps, r.phase_timestep_errors)
                    if err > r.phase_threshold
                ],
            },
            "iforest": {
                "score": r.iforest_score,
                "threshold": r.iforest_threshold,
                "score_ratio": round(r.iforest_score / r.iforest_threshold, 2) if r.iforest_threshold else 0.0,
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
        print(f"Event detail:   {detail_path}")
        print(f"Windows flagged (LSTM):    {summary['flagged_combined']} / {summary['total_windows']}")
        print(f"Windows flagged (rule):    {summary['flagged_by_overload_rule']} / {summary['total_windows']}")
        print(f"Windows flagged (final):   {summary['flagged_final']} / {summary['total_windows']}")
        print(f"Anomaly events (deduped):  {summary['total_anomaly_events']}  (true={summary['true_anomaly_events']})")
        print("\nFlagged by device:")
        for d in summary["by_device"]:
            print(f"  {d['device_id']:15s} [{d['device_category']:7s} / {d['vendor']:8s}]  flagged={d.get('flagged_final',0):4d} / {d['total_windows']}")

        if events:
            print(f"\nSample anomaly events (first 5 of {len(events)}):")
            for r in events[:5]:
                print(f"\n  [{r.device_id}] {r.window_start} → {r.window_end}  types={r.anomaly_types}")
                if r.baseline_anomaly:
                    print(f"    baseline: error={r.baseline_error:.6f} > thresh={r.baseline_threshold:.6f}")
                    print(f"             peak_at={r.baseline_peak_timestep}  top_features={r.baseline_top_features}")
                if r.phase_anomaly:
                    print(f"    phase:    error={r.phase_error:.6f} > thresh={r.phase_threshold:.6f}")
                    print(f"             peak_at={r.phase_peak_timestep}  top_features={r.phase_top_features}")


def main() -> None:
    paths = ProjectPaths()
    results = run_dual_detection(paths=paths)
    save_dual_results(results, paths)
