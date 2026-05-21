"""Live power anomaly detection from Kafka topic `snmp-power-events`.

Thin transport adapter: deserialise → process → emit.
All logic lives in inference/power_stream_processor.py and inference/events.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from snmp_anomaly_detection.config import KafkaConfig, PowerInferenceConfig, ProjectPaths
from snmp_anomaly_detection.inference.events import PowerEvent
from snmp_anomaly_detection.inference.model_scorer import build_window_detail, save_dual_results
from snmp_anomaly_detection.inference.power_stream_processor import (
    AlertState,
    PowerStreamProcessor,
    format_power_alert,
    to_dual_result,
)

try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError:
    KafkaConsumer = None  # type: ignore
    KafkaProducer = None  # type: ignore

POWER_KAFKA_TOPIC = "snmp-power-events"


def _require_kafka() -> None:
    if KafkaConsumer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. "
            "Install it and then run `python -m snmp_anomaly_detection detect-power-kafka`."
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

    alerts_topic = kafka_config.power_alerts_topic
    processor = PowerStreamProcessor(inference_config=inference_config, paths=paths)
    consumer = KafkaConsumer(
        POWER_KAFKA_TOPIC,
        bootstrap_servers=list(kafka_config.bootstrap_servers),
        group_id="snmp-power-anomaly-detection",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    alert_producer = KafkaProducer(
        bootstrap_servers=list(kafka_config.bootstrap_servers),
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    jsonl_file = paths.power_dual_outputs_dir / "kafka_power_results.jsonl"
    print(f"Listening on topic : {POWER_KAFKA_TOPIC}")
    print(f"Publishing alerts  : {alerts_topic}")
    print(f"Streaming results  : {jsonl_file}")
    print(f"Reports saved to   : {paths.power_dual_outputs_dir}/ on shutdown\n")

    accumulated_dual = []
    flagged_count = 0
    with open(jsonl_file, "a") as out_f:
        try:
            while True:
                polled = consumer.poll(timeout_ms=kafka_config.poll_timeout_ms)
                for _, records in polled.items():
                    for message in records:
                        event = PowerEvent.from_dict(message.value)
                        if event is None:
                            continue
                        result = processor.process_event(event)
                        if result is None:
                            continue

                        row = asdict(result)
                        row["window_end"] = str(row["window_end"])
                        out_f.write(json.dumps(row) + "\n")

                        alert_str = format_power_alert(result)
                        if alert_str:
                            print(alert_str)

                        if result.alert_state == AlertState.SUSPECTED:
                            continue  # provisional — skip accumulation until confirmed

                        dual = to_dual_result(result)
                        accumulated_dual.append(dual)

                        if dual.final_flag:
                            flagged_count += 1
                            detail = build_window_detail(dual, window_number=flagged_count)
                            alert_producer.send(
                                alerts_topic,
                                value=detail,
                                key=dual.device_id.encode(),
                            )

                        if len(accumulated_dual) % 50 == 0:
                            out_f.flush()
                        if len(accumulated_dual) % 100 == 0:
                            save_dual_results(accumulated_dual, paths, silent=True)
        finally:
            consumer.close()
            alert_producer.flush()
            alert_producer.close()
            if accumulated_dual:
                print(f"\nFinal session report ({len(accumulated_dual)} windows scored):")
                save_dual_results(accumulated_dual, paths)
            else:
                print("No windows scored this session — skipping report generation.")


def main() -> None:
    run_power_kafka_detection()
