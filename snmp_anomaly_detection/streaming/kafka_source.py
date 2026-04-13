from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd

from snmp_anomaly_detection.config import KafkaConfig
from snmp_anomaly_detection.inference.events import NormalizedEvent

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:  # pragma: no cover - lets non-Kafka commands run without dependency installed
    KafkaConsumer = None

    class KafkaError(Exception):  # pragma: no cover
        """Fallback Kafka error when kafka-python is not installed."""


HARDCODED_INPUT_TOPIC = "snmp-live-events"


@dataclass(frozen=True)
class KafkaMessage:
    topic: str
    partition: int
    offset: int
    key: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class KafkaValidationResult:
    is_valid: bool
    errors: tuple[str, ...]


REQUIRED_KAFKA_FIELDS: tuple[str, ...] = (
    "timestamp",
    "device_id",
    "interface",
    "cpu",
    "memory",
    "in_octets",
    "out_octets",
    "errors",
)
OPTIONAL_NUMERIC_KAFKA_FIELDS: tuple[str, ...] = (
    "in_ucast_pkts",
    "out_ucast_pkts",
    "in_discards",
    "out_discards",
    "interface_speed_mbps",
    "interface_admin_status",
    "interface_oper_status",
    "counter_reset",
)


def _require_kafka() -> None:
    if KafkaConsumer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. Install dependencies, then run "
            "`python3 main.py detect-kafka-dry`."
        )


def _decode_message_key(raw_key: bytes | None) -> str | None:
    if raw_key is None:
        return None
    return raw_key.decode("utf-8")


def validate_kafka_payload(payload: Any) -> KafkaValidationResult:
    errors: list[str] = []

    if not isinstance(payload, dict):
        return KafkaValidationResult(
            is_valid=False,
            errors=("payload must be a JSON object",),
        )

    for field_name in REQUIRED_KAFKA_FIELDS:
        if field_name not in payload:
            errors.append(f"missing required field `{field_name}`")

    if errors:
        return KafkaValidationResult(is_valid=False, errors=tuple(errors))

    try:
        parsed_timestamp = pd.to_datetime(payload["timestamp"])
        if pd.isna(parsed_timestamp):
            errors.append("timestamp is invalid")
    except Exception:
        errors.append("timestamp is invalid")

    try:
        device_id = str(payload["device_id"]).strip()
        if not device_id:
            errors.append("device_id is empty")
    except Exception:
        errors.append("device_id is invalid")

    numeric_fields = ("cpu", "memory", "in_octets", "out_octets", "errors")
    for field_name in numeric_fields:
        try:
            value = float(payload[field_name])
            if pd.isna(value):
                errors.append(f"`{field_name}` is invalid")
        except Exception:
            errors.append(f"`{field_name}` is invalid")

    for field_name in OPTIONAL_NUMERIC_KAFKA_FIELDS:
        if field_name not in payload or payload[field_name] is None:
            continue
        try:
            value = float(payload[field_name])
            if pd.isna(value):
                errors.append(f"`{field_name}` is invalid")
        except Exception:
            errors.append(f"`{field_name}` is invalid")

    return KafkaValidationResult(
        is_valid=not errors,
        errors=tuple(errors),
    )


def normalize_kafka_payload(payload: dict[str, Any]) -> NormalizedEvent:
    validation = validate_kafka_payload(payload)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.errors))

    timestamp = pd.to_datetime(payload["timestamp"])
    return NormalizedEvent(
        timestamp=timestamp,
        device_id=str(payload["device_id"]),
        interface=str(payload["interface"]) if payload.get("interface") is not None else None,
        cpu=float(payload["cpu"]),
        memory=float(payload["memory"]),
        in_octets=float(payload["in_octets"]),
        out_octets=float(payload["out_octets"]),
        errors=float(payload["errors"]),
        in_ucast_pkts=(
            None if payload.get("in_ucast_pkts") is None else float(payload["in_ucast_pkts"])
        ),
        out_ucast_pkts=(
            None if payload.get("out_ucast_pkts") is None else float(payload["out_ucast_pkts"])
        ),
        in_discards=(
            None if payload.get("in_discards") is None else float(payload["in_discards"])
        ),
        out_discards=(
            None if payload.get("out_discards") is None else float(payload["out_discards"])
        ),
        interface_speed_mbps=(
            None
            if payload.get("interface_speed_mbps") is None
            else float(payload["interface_speed_mbps"])
        ),
        interface_admin_status=(
            None
            if payload.get("interface_admin_status") is None
            else int(float(payload["interface_admin_status"]))
        ),
        interface_oper_status=(
            None
            if payload.get("interface_oper_status") is None
            else int(float(payload["interface_oper_status"]))
        ),
        counter_reset=int(float(payload.get("counter_reset", 0) or 0)),
        anomaly=int(payload.get("anomaly", 0)),
    )


def build_consumer(config: KafkaConfig | None = None):
    _require_kafka()
    config = config or KafkaConfig()
    return KafkaConsumer(
        HARDCODED_INPUT_TOPIC,
        bootstrap_servers=list(config.bootstrap_servers),
        group_id=config.consumer_group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )


def iter_kafka_messages(config: KafkaConfig | None = None) -> Iterator[KafkaMessage]:
    config = config or KafkaConfig()
    consumer = build_consumer(config)
    try:
        while True:
            polled_records = consumer.poll(timeout_ms=config.poll_timeout_ms)
            for topic_partition, records in polled_records.items():
                for message in records:
                    yield KafkaMessage(
                        topic=topic_partition.topic,
                        partition=topic_partition.partition,
                        offset=message.offset,
                        key=_decode_message_key(message.key),
                        payload=message.value,
                    )
    finally:
        consumer.close()
