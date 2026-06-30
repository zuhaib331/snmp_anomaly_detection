"""CSV replay adapter for power anomaly detection.

Thin transport layer: load dataset → score → save results.
All logic lives in model_scorer.py; this file is the CLI entry point.
"""
from __future__ import annotations

from snmp_anomaly_detection.config import ProjectPaths
from snmp_anomaly_detection.power.inference.model_scorer import (
    DualModelResult,
    run_csv_detection,
    save_dual_results,
)

__all__ = ["DualModelResult", "run_csv_detection", "save_dual_results"]


def main() -> None:
    paths = ProjectPaths()
    results = run_csv_detection(paths=paths)
    save_dual_results(results, paths)
