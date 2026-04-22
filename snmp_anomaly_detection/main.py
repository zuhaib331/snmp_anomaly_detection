from __future__ import annotations

import argparse
import sys

from snmp_anomaly_detection.data.c1_event_normalization import main as normalize_events_main
from snmp_anomaly_detection.data.c2_event_correlation import main as correlate_events_main
from snmp_anomaly_detection.data.c3_explanation import main as explain_anomalies_main
from snmp_anomaly_detection.data.dataset_builder import main as build_dataset_main
from snmp_anomaly_detection.data.p2_feature_readiness import main as report_p2_main
from snmp_anomaly_detection.data.p3_event_readiness import main as report_p3_main
from snmp_anomaly_detection.evaluation.p1_1_comparison import main as compare_p1_main
from snmp_anomaly_detection.evaluation.p1_baseline import main as evaluate_p1_main
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
            "report-p2",
            "preprocess",
            "train",
            "evaluate-baseline",
            "compare-artifacts",
            "report-p3",
            "normalize-events",
            "correlate-events",
            "explain-anomalies",
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
    sys.argv = forwarded_argv

    if args.step == "generate-data":
        build_dataset_main()
    elif args.step == "report-p2":
        report_p2_main()
    elif args.step == "preprocess":
        feature_engineering_main()
    elif args.step == "train":
        train_main()
    elif args.step == "evaluate-baseline":
        evaluate_p1_main()
    elif args.step == "compare-artifacts":
        compare_p1_main()
    elif args.step == "report-p3":
        report_p3_main()
    elif args.step == "normalize-events":
        normalize_events_main()
    elif args.step == "correlate-events":
        correlate_events_main()
    elif args.step == "explain-anomalies":
        explain_anomalies_main()
    elif args.step == "detect":
        detect_main()
    elif args.step == "detect-csv":
        detect_csv_main()
    elif args.step == "detect-kafka-dry":
        detect_kafka_dry_main()
    elif args.step == "detect-kafka":
        detect_kafka_main()
    elif args.step == "produce-kafka-test-data":
        produce_kafka_test_data_main()


if __name__ == "__main__":
    main()
