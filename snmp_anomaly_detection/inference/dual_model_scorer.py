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
    PowerInferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.models.lstm_autoencoder import LSTMAutoencoder, torch
from snmp_anomaly_detection.preprocessing.power_features import (
    aggregate_phase_metrics,
    apply_log1p_skewed,
    add_delta_features,
    load_power_dataset,
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
    combined_flag: int          # determined by alert policy
    alert_policy: str
    true_label: int = 0
    anomaly_types: list[str] = field(default_factory=list)
    # Per-sequence diagnostics
    window_timestamps: list[str] = field(default_factory=list)
    baseline_timestep_errors: list[float] = field(default_factory=list)
    baseline_peak_timestep: str = ""
    baseline_top_features: str = ""
    phase_timestep_errors: list[float] = field(default_factory=list)
    phase_peak_timestep: str = ""
    phase_top_features: str = ""


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


def _score_window_detailed(
    model,
    window: np.ndarray,
    feature_names: list[str],
    normal_feature_errors: dict[str, float],
    normal_feature_error_stds: dict[str, float],
) -> dict:
    loss_fn = torch.nn.MSELoss(reduction="none")
    tensor = torch.tensor(window[np.newaxis], dtype=torch.float32)  # (1, seq_len, n_feat)
    with torch.no_grad():
        recon = model(tensor)
        errors = loss_fn(recon, tensor).squeeze(0).cpu().numpy()  # (seq_len, n_feat)

    timestep_errors = errors.mean(axis=1)   # (seq_len,)
    feature_errors = errors.mean(axis=0)    # (n_feat,)

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
        "mean_error": float(errors.mean()),
        "timestep_errors": [round(float(e), 6) for e in timestep_errors],
        "peak_timestep_idx": peak_idx,
        "top_features": ",".join(top_features),
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

    # Load phase artifacts
    with open(paths.power_phase_outputs_dir / "phase_metadata.json") as f:
        phase_meta = json.load(f)
    phase_model = _load_model(paths.power_phase_outputs_dir / "phase_model.pt", phase_meta)
    phase_scaler = joblib.load(paths.power_phase_outputs_dir / "phase_scaler.pkl")
    phase_threshold = phase_meta["threshold"]
    phase_seq_len = phase_meta["seq_len"]
    phase_normal_errors = phase_meta.get("normal_feature_errors", {})
    phase_normal_stds   = phase_meta.get("normal_feature_error_stds", {})

    # Load and preprocess dataset
    df = load_power_dataset(paths)
    df = aggregate_phase_metrics(df)
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

        seq_len = max(baseline_seq_len, phase_seq_len)
        for i in range(len(device_df) - seq_len):
            b_win = baseline_vals[i : i + baseline_seq_len]
            p_win = phase_vals[i : i + phase_seq_len]
            win_timestamps = timestamps[i : i + seq_len].tolist()

            b_detail = _score_window_detailed(baseline_model, b_win, baseline_cols, baseline_normal_errors, baseline_normal_stds)
            p_detail = _score_window_detailed(phase_model, p_win, phase_cols, phase_normal_errors, phase_normal_stds)

            b_err = b_detail["mean_error"]
            p_err = p_detail["mean_error"]
            b_flag = int(b_err > baseline_threshold)
            p_flag = int(p_err > phase_threshold)

            if policy == "or":
                combined = int(b_flag or p_flag)
            else:
                combined = int(b_flag and p_flag)

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
                baseline_threshold=baseline_threshold,
                baseline_anomaly=b_flag,
                phase_error=round(p_err, 6),
                phase_threshold=phase_threshold,
                phase_anomaly=p_flag,
                combined_flag=combined,
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
            ))

    return results


def save_dual_results(results: list[DualModelResult], paths: ProjectPaths) -> None:
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
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_baseline"] += r.baseline_anomaly
        entry["flagged_phase"] += r.phase_anomaly
        entry["flagged_combined"] += r.combined_flag
        entry["true_anomaly_windows"] += r.true_label

    # --- Per-vendor breakdown ---
    by_vendor: dict = {}
    for r in results:
        entry = by_vendor.setdefault(r.vendor, {
            "vendor": r.vendor,
            "total_windows": 0,
            "flagged_combined": 0,
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_combined"] += r.combined_flag
        entry["true_anomaly_windows"] += r.true_label

    # --- Per-category breakdown ---
    by_category: dict = {}
    for r in results:
        entry = by_category.setdefault(r.device_category, {
            "device_category": r.device_category,
            "total_windows": 0,
            "flagged_combined": 0,
            "true_anomaly_windows": 0,
        })
        entry["total_windows"] += 1
        entry["flagged_combined"] += r.combined_flag
        entry["true_anomaly_windows"] += r.true_label

    summary = {
        "alert_policy": results[0].alert_policy if results else "or",
        "total_windows": len(results),
        "flagged_by_baseline": sum(r.baseline_anomaly for r in results),
        "flagged_by_phase": sum(r.phase_anomaly for r in results),
        "flagged_combined": sum(r.combined_flag for r in results),
        "true_anomaly_windows": sum(r.true_label for r in results),
        "by_device": list(by_device.values()),
        "by_vendor": list(by_vendor.values()),
        "by_category": list(by_category.values()),
    }
    with open(paths.power_dual_outputs_dir / "detection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Flagged window detail JSON ---
    flagged = [r for r in results if r.combined_flag == 1]
    detail_rows = []
    for r in flagged:
        detail_rows.append({
            "device_id": r.device_id,
            "device_category": r.device_category,
            "vendor": r.vendor,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "anomaly_types": r.anomaly_types,
            "alert_policy": r.alert_policy,
            "baseline": {
                "error": r.baseline_error,
                "threshold": r.baseline_threshold,
                "flagged": bool(r.baseline_anomaly),
                "peak_timestep": r.baseline_peak_timestep,
                "top_features": r.baseline_top_features,
                "sequence": [
                    {"timestamp": ts, "error": err}
                    for ts, err in zip(r.window_timestamps, r.baseline_timestep_errors)
                ],
            },
            "phase": {
                "error": r.phase_error,
                "threshold": r.phase_threshold,
                "flagged": bool(r.phase_anomaly),
                "peak_timestep": r.phase_peak_timestep,
                "top_features": r.phase_top_features,
                "sequence": [
                    {"timestamp": ts, "error": err}
                    for ts, err in zip(r.window_timestamps, r.phase_timestep_errors)
                ],
            },
        })
    detail_path = paths.power_dual_outputs_dir / "anomaly_windows_detail.json"
    with open(detail_path, "w") as f:
        json.dump(detail_rows, f, indent=2)

    print(f"Results:        {csv_path}")
    print(f"Window detail:  {detail_path}")
    print(f"Windows flagged (combined): {summary['flagged_combined']} / {summary['total_windows']}")
    print("\nFlagged by device:")
    for d in summary["by_device"]:
        print(f"  {d['device_id']:15s} [{d['device_category']:7s} / {d['vendor']:8s}]  flagged={d['flagged_combined']:4d} / {d['total_windows']}")

    if flagged:
        print(f"\nSample anomaly windows (first 5 of {len(flagged)}):")
        for r in flagged[:5]:
            print(f"\n  [{r.device_id}] {r.window_start} → {r.window_end}  types={r.anomaly_types}")
            if r.baseline_anomaly:
                print(f"    baseline: error={r.baseline_error:.6f} > thresh={r.baseline_threshold:.6f}")
                print(f"             peak_at={r.baseline_peak_timestep}  top_features=[{r.baseline_top_features}]")
                print(f"             sequence_errors={r.baseline_timestep_errors}")
            if r.phase_anomaly:
                print(f"    phase:    error={r.phase_error:.6f} > thresh={r.phase_threshold:.6f}")
                print(f"             peak_at={r.phase_peak_timestep}  top_features=[{r.phase_top_features}]")
                print(f"             sequence_errors={r.phase_timestep_errors}")


def main() -> None:
    paths = ProjectPaths()
    results = run_dual_detection(paths=paths)
    save_dual_results(results, paths)
