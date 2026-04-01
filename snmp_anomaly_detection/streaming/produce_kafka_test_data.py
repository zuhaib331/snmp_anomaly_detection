from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np

from snmp_anomaly_detection.config import KafkaConfig
from snmp_anomaly_detection.data.dataset_builder import (
    build_device_profile,
    generate_normal_pattern,
    inject_anomaly,
    pick_anomaly_type,
)
from snmp_anomaly_detection.streaming.kafka_source import HARDCODED_INPUT_TOPIC

try:
    from kafka import KafkaProducer
except ImportError:  # pragma: no cover - lets non-Kafka commands run without dependency installed
    KafkaProducer = None


@dataclass(frozen=True)
class SyntheticStreamConfig:
    device_count: int = 15
    max_messages: int | None = None
    sleep_seconds: float = 0.1
    anomaly_probability: float = 0.02
    timestamp_step_seconds: int = 5
    seed: int | None = None


@dataclass
class DeviceState:
    device_id: str
    interface: str
    device_type: str
    timestep: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cumulative_in_octets: float = 0.0
    cumulative_out_octets: float = 0.0
    cumulative_errors: int = 0


def _require_kafka() -> None:
    if KafkaProducer is None:
        raise ImportError(
            "Kafka producer support requires `kafka-python`. Install dependencies, then run "
            "`python3 main.py produce-kafka-test-data`."
        )


def _build_producer(config: KafkaConfig | None = None):
    _require_kafka()
    config = config or KafkaConfig()
    return KafkaProducer(
        bootstrap_servers=list(config.bootstrap_servers),
        key_serializer=lambda value: value.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def _build_device_states(config: SyntheticStreamConfig) -> list[DeviceState]:
    base_time = datetime.now(timezone.utc)
    device_types = ("Router", "Switch", "Firewall")
    return [
        DeviceState(
            device_id=f"dev_{index}",
            interface=f"eth{random.randint(0, 3)}",
            device_type=random.choice(device_types),
            timestep=index,
            timestamp=base_time + timedelta(seconds=index),
        )
        for index in range(config.device_count)
    ]


def _build_payload(device_state: DeviceState, config: SyntheticStreamConfig) -> dict[str, object]:
    profile = build_device_profile(device_state.device_type)
    traffic = generate_normal_pattern(device_state.timestep) * profile.traffic_scale
    cpu = float(np.clip(np.random.normal(profile.base_cpu, 8), 0, 100))
    memory = float(np.clip(np.random.normal(profile.base_memory, 10), 0, 100))
    in_octets_increment = float(max(traffic * random.uniform(800, 1200), 0))
    out_octets_increment = float(max(traffic * random.uniform(700, 1100), 0))
    errors_increment = int(np.random.poisson(1))
    anomaly_type = "normal"

    anomaly = 0
    if random.random() < config.anomaly_probability:
        anomaly_type = pick_anomaly_type()
        if anomaly_type == "traffic_spike":
            in_octets_increment *= random.uniform(2.5, 4.5)
            out_octets_increment *= random.uniform(2.0, 4.0)
            cpu = float(np.clip(cpu + random.uniform(8, 20), 0, 100))
        elif anomaly_type == "traffic_drop":
            in_octets_increment *= random.uniform(0.02, 0.20)
            out_octets_increment *= random.uniform(0.02, 0.20)
        elif anomaly_type == "error_burst":
            errors_increment += random.randint(20, 120)
        elif anomaly_type == "cpu_spike":
            cpu = float(np.clip(inject_anomaly(cpu), 0, 100))
        elif anomaly_type == "memory_spike":
            memory = float(np.clip(inject_anomaly(memory), 0, 100))
        elif anomaly_type == "link_down":
            in_octets_increment = 0.0
            out_octets_increment = 0.0
            errors_increment += random.randint(1, 10)
        anomaly = 1

    in_octets_increment = max(in_octets_increment, 0.0)
    out_octets_increment = max(out_octets_increment, 0.0)
    errors_increment = max(errors_increment, 0)

    device_state.cumulative_in_octets += in_octets_increment
    device_state.cumulative_out_octets += out_octets_increment
    device_state.cumulative_errors += errors_increment

    payload = {
        "timestamp": device_state.timestamp.isoformat(),
        "device_id": device_state.device_id,
        "interface": device_state.interface,
        "cpu": round(cpu, 2),
        "memory": round(memory, 2),
        "in_octets": round(device_state.cumulative_in_octets, 2),
        "out_octets": round(device_state.cumulative_out_octets, 2),
        "errors": device_state.cumulative_errors,
        "anomaly": anomaly,
        "anomaly_type": anomaly_type,
    }

    device_state.timestep += 1
    device_state.timestamp += timedelta(seconds=config.timestamp_step_seconds)
    return payload


def iter_synthetic_messages(config: SyntheticStreamConfig | None = None) -> Iterator[dict[str, object]]:
    config = config or SyntheticStreamConfig()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    device_states = _build_device_states(config)
    sent_count = 0

    while True:
        for device_state in device_states:
            if config.max_messages is not None and sent_count >= config.max_messages:
                return

            yield _build_payload(device_state, config)
            sent_count += 1


def produce_kafka_test_data(
    kafka_config: KafkaConfig | None = None,
    stream_config: SyntheticStreamConfig | None = None,
) -> int:
    kafka_config = kafka_config or KafkaConfig()
    stream_config = stream_config or SyntheticStreamConfig()
    producer = _build_producer(kafka_config)
    sent_count = 0

    print(f"Publishing live synthetic SNMP events to Kafka topic: {HARDCODED_INPUT_TOPIC}")
    print(f"Bootstrap servers: {', '.join(kafka_config.bootstrap_servers)}")
    print(f"Stream config: {asdict(stream_config)}")
    print("Press Ctrl+C to stop.")

    try:
        for payload in iter_synthetic_messages(stream_config):
            future = producer.send(
                HARDCODED_INPUT_TOPIC,
                key=str(payload["device_id"]),
                value=payload,
            )
            future.get(timeout=10)
            sent_count += 1
            print(
                "sent="
                f"{sent_count} device_id={payload['device_id']} "
                f"timestamp={payload['timestamp']} anomaly={payload['anomaly']}"
            )

            if stream_config.sleep_seconds > 0:
                time.sleep(stream_config.sleep_seconds)
    except KeyboardInterrupt:
        print("Kafka producer stopped by user.")
    finally:
        producer.flush()
        producer.close()

    print(f"Total Kafka messages sent: {sent_count}")
    return sent_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fresh synthetic SNMP events continuously and publish them to Kafka.",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=15,
        help="Number of simulated devices publishing into the shared topic.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after sending this many messages. Default is continuous streaming.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Delay between Kafka messages to simulate a live stream.",
    )
    parser.add_argument(
        "--anomaly-probability",
        type=float,
        default=0.02,
        help="Probability that a generated message is labeled and shaped as an anomaly.",
    )
    parser.add_argument(
        "--timestamp-step-seconds",
        type=int,
        default=5,
        help="Logical per-device time increment applied after each generated event.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible synthetic streams.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stream_config = SyntheticStreamConfig(
        device_count=args.device_count,
        max_messages=args.max_messages,
        sleep_seconds=args.sleep_seconds,
        anomaly_probability=args.anomaly_probability,
        timestamp_step_seconds=args.timestamp_step_seconds,
        seed=args.seed,
    )
    produce_kafka_test_data(stream_config=stream_config)


if __name__ == "__main__":
    main()
