from __future__ import annotations

import argparse

import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.inference.csv_replay import detect_csv_replay


def detect_anomalies(
    input_file: str | None = None,
    paths: ProjectPaths | None = None,
    config: FeatureEngineeringConfig | None = None,
    inference_config: InferenceConfig | None = None,
) -> pd.DataFrame:
    return detect_csv_replay(
        input_file=input_file,
        paths=paths,
        config=config,
        inference_config=inference_config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNMP anomaly detection on CSV input.")
    parser.add_argument(
        "--input-file",
        help="Optional CSV file to score. Defaults to the configured dataset file.",
    )
    parser.add_argument(
        "--artifact-dir-name",
        help="Named artifact directory under snmp_anomaly_detection/artifacts/ to load.",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Explicit artifact directory path to load model, scaler, and metadata from.",
    )
    args = parser.parse_args()

    default_paths = ProjectPaths()
    paths = ProjectPaths(
        artifact_dir_name=args.artifact_dir_name or default_paths.artifact_dir_name,
        artifact_dir_override=args.artifact_dir,
    )
    detect_anomalies(input_file=args.input_file, paths=paths)


if __name__ == "__main__":
    main()
