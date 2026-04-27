"""Produce synthetic power SNMP events to the Kafka topic `snmp-power-events`.

Generates realistic multi-device power events (UPS, PDU, network, env)
including occasional injected anomalies for end-to-end testing.
"""
from __future__ import annotations

import json
import time

from snmp_anomaly_detection.config import KafkaConfig
from snmp_anomaly_detection.data.dataset_builder import (
    PowerDatasetConfig,
    build_power_dataset,
)
from snmp_anomaly_detection.streaming.detect_power_kafka import POWER_KAFKA_TOPIC

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None  # type: ignore


_DEFAULT_EVENTS_TO_PRODUCE = 500
_INTER_EVENT_SLEEP_S = 0.05   # 50ms between events = ~20 msg/s


def _require_kafka() -> None:
    if KafkaProducer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. "
            "Install it and then run `python main.py produce-power-kafka-test`."
        )


def produce_power_test_events(
    num_events: int = _DEFAULT_EVENTS_TO_PRODUCE,
    kafka_config: KafkaConfig | None = None,
) -> None:
    _require_kafka()
    kafka_config = kafka_config or KafkaConfig()

    producer = KafkaProducer(
        bootstrap_servers=list(kafka_config.bootstrap_servers),
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    # Generate a small dataset and replay its rows
    config = PowerDatasetConfig(total_points=max(num_events // 9, 100))  # 9 devices
    df = build_power_dataset(config)
    rows = df.to_dict("records")

    produced = 0
    for row in rows[:num_events]:
        message = {k: v for k, v in row.items() if k != "anomaly_type"}
        producer.send(POWER_KAFKA_TOPIC, value=message, key=str(row["device_id"]).encode())
        produced += 1
        if produced % 50 == 0:
            print(f"Produced {produced}/{num_events} events to {POWER_KAFKA_TOPIC}")
        time.sleep(_INTER_EVENT_SLEEP_S)

    producer.flush()
    producer.close()
    print(f"Done. {produced} power events sent to {POWER_KAFKA_TOPIC}.")


def main() -> None:
    produce_power_test_events()
