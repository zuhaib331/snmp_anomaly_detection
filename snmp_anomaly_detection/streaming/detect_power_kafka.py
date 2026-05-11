"""Live power anomaly detection from Kafka topic `snmp-power-events`."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from snmp_anomaly_detection.config import KafkaConfig, PowerInferenceConfig, ProjectPaths
from snmp_anomaly_detection.config import BASELINE_UPS_FEATURES, PHASE_LEVEL_FEATURES
from snmp_anomaly_detection.inference.model_scorer import DualModelResult, save_dual_results
from snmp_anomaly_detection.inference.events import PowerEvent
from snmp_anomaly_detection.inference.power_stream_processor import (
    AlertState,
    PowerScoringResult,
    PowerStreamProcessor,
    to_dual_result,
)

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

# Raw source columns needed by EventPreprocessor for B2 normalization.
# Not in BASELINE_UPS_FEATURES / PHASE_LEVEL_FEATURES (those hold the post-normalization names),
# but these must be in feature_values so normalize_absolute() can compute the ratios.
_B2_RAW_COLS: tuple[str, ...] = (
    "input_voltage_v",
    "output_voltage_v",
    "output_current_a",
    "battery_voltage_v",
    "battery_current_a",
)


def _require_kafka() -> None:
    if KafkaConsumer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. "
            "Install it and then run `python -m snmp_anomaly_detection detect-power-kafka`."
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
    all_feature_cols = set(BASELINE_UPS_FEATURES) | set(PHASE_LEVEL_FEATURES) | set(_B2_RAW_COLS)
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
        phase_count=int(payload.get("phase_count", 1)),
        rated_capacity_w=float(payload.get("rated_capacity_w", 0.0)),
        nominal_voltage_v=float(payload.get("nominal_voltage_v", 120.0)),
        rated_battery_v=float(payload.get("rated_battery_v", 0.0)),
        session_reset=bool(payload.get("_device_reset", False)),
        true_label=int(payload.get("expected_label", 0)),
        anomaly_type=str(payload.get("expected_anomaly_type", "none")),
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

    jsonl_file = paths.power_dual_outputs_dir / "kafka_power_results.jsonl"
    print(f"Listening on topic: {POWER_KAFKA_TOPIC}")
    print(f"Streaming results : {jsonl_file}")
    print(f"Reports saved to  : {paths.power_dual_outputs_dir}/ on shutdown\n")

    accumulated_dual: list[DualModelResult] = []

    with open(jsonl_file, "a") as out_f:
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

                        if result.alert_state == AlertState.SUSPECTED:
                            print(
                                f"SUSPECTED [IF] [{result.device_category}] "
                                f"{result.device_id} @ {result.window_start} "
                                f"if_score={result.iforest_score:.4f} < thresh={result.iforest_threshold:.4f}"
                            )
                            continue  # provisional — skip accumulation until confirmed

                        if result.alert_state == AlertState.CLEARED:
                            print(
                                f"CLEARED [{result.device_category}] "
                                f"{result.device_id} @ {result.window_start} "
                                f"(IF retracted — LSTM did not confirm)"
                            )

                        accumulated_dual.append(to_dual_result(result))

                        if len(accumulated_dual) % 50 == 0:
                            out_f.flush()

                        if len(accumulated_dual) % 100 == 0:
                            save_dual_results(accumulated_dual, paths, silent=True)

                        if result.final_flag and result.alert_state != AlertState.CLEARED:
                            state_label = result.alert_state.value.upper()
                            prefix = "COMPOUND " if result.compound_alert else ""
                            print(
                                f"{prefix}{state_label} [{result.device_category}] "
                                f"{result.device_id} "
                                f"{result.window_start} → {result.window_end} "
                                f"baseline={result.baseline_anomaly} "
                                f"(err={result.baseline_error:.4f} > {result.baseline_threshold:.4f}"
                                + (f", top=[{result.baseline_top_features}]" if result.baseline_top_features else "")
                                + f") phase={result.phase_anomaly}"
                                + (
                                    f" (err={result.phase_error:.4f} > {result.phase_threshold:.4f}"
                                    + (f", top=[{result.phase_top_features}]" if result.phase_top_features else "")
                                    + ")"
                                    if result.phase_anomaly else ""
                                )
                                + (f" if={result.iforest_flag}"
                                    + (f" (score={result.iforest_score:.4f})" if result.iforest_flag else "")
                                )
                                + (f"  peer={result.compound_alert_peer}" if result.compound_alert else "")
                            )
        finally:
            consumer.close()
            if accumulated_dual:
                print(f"\nFinal session report ({len(accumulated_dual)} windows scored):")
                save_dual_results(accumulated_dual, paths)
            else:
                print("No windows scored this session — skipping report generation.")


def main() -> None:
    run_power_kafka_detection()
