from __future__ import annotations

import json
from collections import Counter

from snmp_anomaly_detection.config import FeatureEngineeringConfig, KafkaConfig, ProjectPaths
from snmp_anomaly_detection.network.inference.core import load_inference_artifacts
from snmp_anomaly_detection.network.inference.event_processor import EventProcessor, ProcessedWindow
from snmp_anomaly_detection.network.inference.live_microbatch import (
    LiveMicroBatchProcessor,
    MicroBatchConfig,
)
from snmp_anomaly_detection.network.streaming.kafka_source import (
    HARDCODED_INPUT_TOPIC,
    _decode_message_key,
    build_consumer,
    normalize_kafka_payload,
    validate_kafka_payload,
)


def _build_live_result_record(
    processed_window: ProcessedWindow,
    feature_config: FeatureEngineeringConfig,
) -> dict[str, object]:
    record = processed_window.to_result_record(feature_config)
    record["source"] = "kafka-live"
    return record


def _build_live_anomaly_window_record(
    processed_window: ProcessedWindow,
    feature_config: FeatureEngineeringConfig,
) -> dict[str, object]:
    result_record = _build_live_result_record(processed_window, feature_config)
    ready_window = processed_window.ready_window

    return {
        "source": "kafka-live",
        "device_id": result_record["device_id"],
        "device_window_index": result_record["device_window_index"],
        "window_start": result_record["window_start"],
        "window_end": result_record["window_end"],
        "sequence_length": result_record["sequence_length"],
        "predicted_anomaly": result_record["predicted_anomaly"],
        "source_anomaly_label": result_record["source_anomaly_label"],
        "reconstruction_error": result_record["reconstruction_error"],
        "threshold": result_record["threshold"],
        "error_margin": result_record["error_margin"],
        "top_error_feature": result_record["top_error_feature"],
        "top_error_timestep_offset": result_record["top_error_timestep_offset"],
        "detection_basis": result_record["detection_basis"],
        "feature_error_cpu": result_record["feature_error_cpu"],
        "feature_error_memory": result_record["feature_error_memory"],
        "feature_error_in_octets": result_record["feature_error_in_octets"],
        "feature_error_out_octets": result_record["feature_error_out_octets"],
        "feature_error_errors": result_record["feature_error_errors"],
        "window_records": ready_window.records,
        "window_summary": {
            "record_count": len(ready_window.records),
            "first_timestamp": result_record["window_start"],
            "last_timestamp": result_record["window_end"],
            "device_id": result_record["device_id"],
        },
    }


def _print_processed_window(
    processed_window: ProcessedWindow,
    feature_config: FeatureEngineeringConfig,
) -> None:
    record = _build_live_result_record(processed_window, feature_config)
    score = processed_window.score
    ready_window = processed_window.ready_window
    print(
        "live_result "
        f"device_id={record['device_id']} "
        f"window_index={ready_window.device_window_index} "
        f"window_start={record['window_start']} "
        f"window_end={record['window_end']} "
        f"predicted_anomaly={record['predicted_anomaly']} "
        f"reconstruction_error={score.reconstruction_error:.6f} "
        f"threshold={record['threshold']:.6f} "
        f"top_error_feature={record['top_error_feature']} "
        f"top_error_timestep_offset={record['top_error_timestep_offset']}"
    )


def detect_kafka(
    kafka_config: KafkaConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    kafka_config = kafka_config or KafkaConfig()
    feature_config = feature_config or FeatureEngineeringConfig()
    paths = paths or ProjectPaths()

    artifacts = load_inference_artifacts(paths=paths, config=feature_config)
    event_processor = EventProcessor(artifacts=artifacts, config=feature_config)
    micro_batch_processor = LiveMicroBatchProcessor(
        event_processor=event_processor,
        config=MicroBatchConfig(
            batch_size=kafka_config.micro_batch_size,
            max_wait_ms=kafka_config.micro_batch_max_wait_ms,
        ),
    )

    valid_message_count = 0
    invalid_message_count = 0
    emitted_result_count = 0
    device_counts: Counter[str] = Counter()

    print(f"Starting Kafka live detection on hardcoded topic: {HARDCODED_INPUT_TOPIC}")
    print(f"Bootstrap servers: {', '.join(kafka_config.bootstrap_servers)}")
    print(
        "Micro-batching: "
        f"batch_size={kafka_config.micro_batch_size} "
        f"max_wait_ms={kafka_config.micro_batch_max_wait_ms}"
    )
    if kafka_config.save_local_results:
        print(f"Local live result file: {paths.kafka_live_results_file}")
        print(f"Local live anomaly window file: {paths.kafka_live_anomaly_windows_file}")
    print("Press Ctrl+C to stop.")

    paths.ensure_directories()
    result_file = None
    anomaly_window_file = None
    if kafka_config.save_local_results:
        result_file = open(paths.kafka_live_results_file, "w", encoding="utf-8")
        anomaly_window_file = open(
            paths.kafka_live_anomaly_windows_file,
            "w",
            encoding="utf-8",
        )

    def persist_processed_window(processed_window: ProcessedWindow) -> None:
        _print_processed_window(processed_window, feature_config)
        if result_file is not None:
            json.dump(
                _build_live_result_record(processed_window, feature_config),
                result_file,
                default=str,
            )
            result_file.write("\n")
            result_file.flush()

        if (
            anomaly_window_file is not None
            and processed_window.score.predicted_anomaly == 1
        ):
            json.dump(
                _build_live_anomaly_window_record(processed_window, feature_config),
                anomaly_window_file,
                default=str,
            )
            anomaly_window_file.write("\n")
            anomaly_window_file.flush()

    consumer = build_consumer(kafka_config)
    try:
        while True:
            polled_records = consumer.poll(timeout_ms=kafka_config.poll_timeout_ms)

            if not polled_records:
                flushed_windows = micro_batch_processor.poll()
                emitted_result_count += len(flushed_windows)
                for processed_window in flushed_windows:
                    persist_processed_window(processed_window)
                continue

            for topic_partition, records in polled_records.items():
                for message in records:
                    payload = message.value
                    validation = validate_kafka_payload(payload)
                    if not validation.is_valid:
                        invalid_message_count += 1
                        print(
                            "invalid_message="
                            f"{invalid_message_count} topic={topic_partition.topic} "
                            f"partition={topic_partition.partition} offset={message.offset} "
                            f"key={_decode_message_key(message.key)} "
                            f"errors={' | '.join(validation.errors)}"
                        )
                        continue

                    event = normalize_kafka_payload(payload)
                    valid_message_count += 1
                    device_counts[event.device_id] += 1

                    flushed_windows = micro_batch_processor.handle_event(event)
                    emitted_result_count += len(flushed_windows)

                    print(
                        "message="
                        f"{valid_message_count} topic={topic_partition.topic} "
                        f"partition={topic_partition.partition} offset={message.offset} "
                        f"device_id={event.device_id} timestamp={event.timestamp} "
                        f"device_seen={device_counts[event.device_id]} "
                        f"flushed_windows={len(flushed_windows)}"
                    )

                    for processed_window in flushed_windows:
                        persist_processed_window(processed_window)
    except KeyboardInterrupt:
        print("Kafka live detection stopped by user.")
    finally:
        remaining_windows = micro_batch_processor.flush()
        emitted_result_count += len(remaining_windows)
        for processed_window in remaining_windows:
            persist_processed_window(processed_window)
        consumer.close()
        if result_file is not None:
            result_file.close()
        if anomaly_window_file is not None:
            anomaly_window_file.close()

    print(f"Total Kafka messages processed: {valid_message_count}")
    print(f"Total invalid Kafka messages skipped: {invalid_message_count}")
    print(f"Total live results emitted: {emitted_result_count}")
    if device_counts:
        print("Per-device message counts:")
        for device_id, count in sorted(device_counts.items()):
            print(f"- {device_id}: {count}")

    return emitted_result_count


def main() -> None:
    detect_kafka()


if __name__ == "__main__":
    main()
