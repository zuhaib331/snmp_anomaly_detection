from __future__ import annotations

from collections import Counter

from snmp_anomaly_detection.config import KafkaConfig
from snmp_anomaly_detection.network.streaming.kafka_source import (
    HARDCODED_INPUT_TOPIC,
    iter_kafka_messages,
    normalize_kafka_payload,
    validate_kafka_payload,
)


def detect_kafka_dry(config: KafkaConfig | None = None) -> int:
    config = config or KafkaConfig()
    message_count = 0
    invalid_message_count = 0
    device_counts: Counter[str] = Counter()

    print(f"Starting Kafka dry ingestion on hardcoded topic: {HARDCODED_INPUT_TOPIC}")
    print(f"Bootstrap servers: {', '.join(config.bootstrap_servers)}")
    print("Press Ctrl+C to stop.")

    try:
        for message in iter_kafka_messages(config):
            validation = validate_kafka_payload(message.payload)
            if not validation.is_valid:
                invalid_message_count += 1
                print(
                    "invalid_message="
                    f"{invalid_message_count} topic={message.topic} "
                    f"partition={message.partition} offset={message.offset} "
                    f"errors={' | '.join(validation.errors)}"
                )
                continue

            event = normalize_kafka_payload(message.payload)
            message_count += 1
            device_counts[event.device_id] += 1
            print(
                "message="
                f"{message_count} topic={message.topic} partition={message.partition} "
                f"offset={message.offset} device_id={event.device_id} "
                f"timestamp={event.timestamp} device_seen={device_counts[event.device_id]}"
            )
    except KeyboardInterrupt:
        print("Kafka dry ingestion stopped by user.")

    print(f"Total Kafka messages processed: {message_count}")
    print(f"Total invalid Kafka messages skipped: {invalid_message_count}")
    if device_counts:
        print("Per-device message counts:")
        for device_id, count in sorted(device_counts.items()):
            print(f"- {device_id}: {count}")
    return message_count


def main() -> None:
    detect_kafka_dry()


if __name__ == "__main__":
    main()
