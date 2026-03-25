from __future__ import annotations

import argparse

from snmp_anomaly_detection.data.dataset_builder import main as build_dataset_main
from snmp_anomaly_detection.inference.detect_anomalies import main as detect_main
from snmp_anomaly_detection.preprocessing.feature_engineering import (
    main as feature_engineering_main,
)
from snmp_anomaly_detection.training.train_model import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNMP anomaly detection pipeline")
    parser.add_argument(
        "step",
        choices=["generate-data", "preprocess", "train", "detect"],
        help="Pipeline step to execute.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.step == "generate-data":
        build_dataset_main()
    elif args.step == "preprocess":
        feature_engineering_main()
    elif args.step == "train":
        train_main()
    elif args.step == "detect":
        detect_main()


if __name__ == "__main__":
    main()
