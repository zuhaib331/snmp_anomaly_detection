from __future__ import annotations

import pandas as pd

from snmp_anomaly_detection.config import (
    FeatureEngineeringConfig,
    InferenceConfig,
    ProjectPaths,
)
from snmp_anomaly_detection.network.inference.csv_replay import detect_csv_replay


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
    detect_anomalies()


if __name__ == "__main__":
    main()
