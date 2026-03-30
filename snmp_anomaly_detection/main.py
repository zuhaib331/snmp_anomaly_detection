from __future__ import annotations

import argparse
import sys

from snmp_anomaly_detection.data.dataset_builder import main as build_dataset_main
from snmp_anomaly_detection.inference.csv_replay import main as detect_csv_main
from snmp_anomaly_detection.inference.detect_anomalies import main as detect_main
from snmp_anomaly_detection.preprocessing.feature_engineering import (
    main as feature_engineering_main,
)
from snmp_anomaly_detection.streaming.detect_kafka import main as detect_kafka_main
from snmp_anomaly_detection.streaming.detect_kafka_dry import main as detect_kafka_dry_main
from snmp_anomaly_detection.streaming.produce_kafka_test_data import (
    main as produce_kafka_test_data_main,
)
from snmp_anomaly_detection.training.train_model import main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNMP anomaly detection pipeline")
    parser.add_argument(
        "step",
        choices=[
            "generate-data",
            "preprocess",
            "train",
            "detect",
            "detect-csv",
            "detect-kafka-dry",
            "detect-kafka",
            "produce-kafka-test-data",
        ],
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "step_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to the selected step.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    forwarded_argv = [sys.argv[0], *args.step_args]

    if args.step == "generate-data":
        build_dataset_main()
    elif args.step == "preprocess":
        feature_engineering_main()
    elif args.step == "train":
        train_main()
    elif args.step == "detect":
        detect_main()
    elif args.step == "detect-csv":
        detect_csv_main()
    elif args.step == "detect-kafka-dry":
        detect_kafka_dry_main()
    elif args.step == "detect-kafka":
        detect_kafka_main()
    elif args.step == "produce-kafka-test-data":
        sys.argv = forwarded_argv
        produce_kafka_test_data_main()


if __name__ == "__main__":
    main()
