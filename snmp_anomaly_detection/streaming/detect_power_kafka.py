"""Live power anomaly detection from Kafka topic `snmp-power-events`."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from snmp_anomaly_detection.config import KafkaConfig, PowerInferenceConfig, ProjectPaths
from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, PHASE_LEVEL_FEATURES
from snmp_anomaly_detection.inference.power_stream_processor import PowerEvent, PowerStreamProcessor

try:
    from kafka import KafkaConsumer
except ImportError:
    KafkaConsumer = None  # type: ignore

POWER_KAFKA_TOPIC = "snmp-power-events"

POWER_REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "device_category",
    "vendor",
)


def _require_kafka() -> None:
    if KafkaConsumer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. "
            "Install it and then run `python main.py detect-power-kafka`."
        )


def _parse_power_event(payload: dict) -> PowerEvent | None:
    for field in POWER_REQUIRED_FIELDS:
        if field not in payload:
            return None
    try:
        ts = pd.to_datetime(payload["timestamp"])
        if pd.isna(ts):
            return None
    except Exception:
        return None

    feature_values: dict[str, float] = {}
    all_feature_cols = set(BASELINE_UPS_FEATURES) | set(PHASE_LEVEL_FEATURES)
    for col in all_feature_cols:
        try:
            feature_values[col] = float(payload.get(col, 0.0))
        except (TypeError, ValueError):
            feature_values[col] = 0.0

    return PowerEvent(
        timestamp=ts.to_pydatetime(),
        device_id=str(payload["device_id"]),
        device_category=str(payload.get("device_category", "ups")),
        vendor=str(payload.get("vendor", "generic")),
        feature_values=feature_values,
    )


def run_power_kafka_detection(
    kafka_config: KafkaConfig | None = None,
    inference_config: PowerInferenceConfig | None = None,
    paths: ProjectPaths | None = None,
) -> None:
    _require_kafka()
    kafka_config = kafka_config or KafkaConfig()
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()

    processor = PowerStreamProcessor(inference_config=inference_config, paths=paths)

    consumer = KafkaConsumer(
        POWER_KAFKA_TOPIC,
        bootstrap_servers=list(kafka_config.bootstrap_servers),
        group_id="snmp-power-anomaly-detection",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    results_file = paths.power_dual_outputs_dir / "kafka_power_results.jsonl"
    print(f"Listening on topic: {POWER_KAFKA_TOPIC}")
    print(f"Results: {results_file}")

    with open(results_file, "a") as out_f:
        try:
            while True:
                polled = consumer.poll(timeout_ms=kafka_config.poll_timeout_ms)
                for _, records in polled.items():
                    for message in records:
                        event = _parse_power_event(message.value)
                        if event is None:
                            continue
                        result = processor.process_event(event)
                        if result is None:
                            continue
                        row = asdict(result)
                        row["window_end"] = str(row["window_end"])
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                        if result.combined_flag:
                            prefix = "COMPOUND " if result.compound_alert else ""
                            print(
                                f"{prefix}ANOMALY [{result.device_category}] "
                                f"{result.device_id} @ {result.window_end} "
                                f"baseline={result.baseline_anomaly} phase={result.phase_anomaly}"
                            )
        finally:
            consumer.close()


def main() -> None:
    run_power_kafka_detection()
