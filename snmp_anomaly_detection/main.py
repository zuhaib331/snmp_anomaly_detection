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

# Power pipeline steps (imported lazily to avoid hard dependency on torch at startup)


def _import_power_steps() -> dict:
    from snmp_anomaly_detection.data.dataset_builder import (
        PowerDatasetConfig,
        build_power_dataset,
        save_power_dataset,
    )
    from snmp_anomaly_detection.preprocessing.power_features import main as preprocess_power_main
    from snmp_anomaly_detection.training.train_baseline_power import main as train_baseline_main
    from snmp_anomaly_detection.training.train_phase_model import main as train_phase_main
    from snmp_anomaly_detection.training.train_battery_rul import main as train_rul_main
    from snmp_anomaly_detection.evaluation.power_eval import main as evaluate_baseline_main
    from snmp_anomaly_detection.evaluation.rul_eval import main as evaluate_rul_main
    from snmp_anomaly_detection.inference.dual_model_scorer import main as detect_power_csv_main
    from snmp_anomaly_detection.streaming.detect_power_kafka import main as detect_power_kafka_main
    from snmp_anomaly_detection.streaming.produce_power_kafka_test import (
        main as produce_power_kafka_test_main,
    )

    def _generate_power_data() -> None:
        config = PowerDatasetConfig()
        df = build_power_dataset(config)
        path = save_power_dataset(df)
        print(f"Power dataset saved: {path}")
        print(f"Rows: {len(df)}, Devices: {df['device_id'].nunique()}")
        print(df[["device_id", "device_category", "vendor", "anomaly"]].value_counts("anomaly"))

    return {
        "generate-power-data": _generate_power_data,
        "preprocess-power": preprocess_power_main,
        "train-power-baseline": train_baseline_main,
        "train-power-phase": train_phase_main,
        "train-battery-rul": train_rul_main,
        "evaluate-power-baseline": evaluate_baseline_main,
        "predict-battery-rul": evaluate_rul_main,
        "detect-power-csv": detect_power_csv_main,
        "detect-power-kafka": detect_power_kafka_main,
        "produce-power-kafka-test": produce_power_kafka_test_main,
    }


_POWER_STEPS = (
    "generate-power-data",
    "preprocess-power",
    "train-power-baseline",
    "train-power-phase",
    "train-battery-rul",
    "evaluate-power-baseline",
    "predict-battery-rul",
    "detect-power-csv",
    "detect-power-kafka",
    "produce-power-kafka-test",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SNMP anomaly detection pipeline")
    parser.add_argument(
        "step",
        choices=[
            # Original network pipeline
            "generate-data",
            "preprocess",
            "train",
            "detect",
            "detect-csv",
            "detect-kafka-dry",
            "detect-kafka",
            "produce-kafka-test-data",
            # Power domain pipeline
            *_POWER_STEPS,
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

    # Original network pipeline
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

    # Power domain pipeline
    elif args.step in _POWER_STEPS:
        power_steps = _import_power_steps()
        power_steps[args.step]()


if __name__ == "__main__":
    main()
