"""Produce synthetic power SNMP events to the Kafka topic `snmp-power-events`.

Generates a randomised fleet of devices (UPS, PDU, network, env) covering
single-phase and three-phase variants with injected anomalies, for
end-to-end testing of the power detection pipeline.

Improvements over v1
--------------------
* Capacity ratings constrained to training-data distribution so output_power_w
  stays in-distribution for the fitted scaler — eliminates OOD false positives.
* Events interleaved by poll cycle across all devices, mirroring real SNMP
  polling where every device is queried each interval.
* Guaranteed-normal warmup of seq_len events per device sent first so the
  per-device rolling buffer fills before any anomaly data arrives.
* Wall-clock timestamps replace synthetic 2025-01-01 values so alert times
  match the actual test run and compound-alert timing is real.
* `expected_label` (0/1) and `expected_anomaly_type` included in every message
  so TP/FP/FN can be computed offline from the JSONL output.
* First event per device carries `_device_reset=true` so the consumer resets
  per-device delta state left over from any previous producer run.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone

from snmp_anomaly_detection.config import KafkaConfig, PowerTrainingConfig
from snmp_anomaly_detection.data.dataset_builder import (
    PowerDatasetConfig,
    PowerDeviceProfile,
    _DEFAULT_POWER_PROFILES,
    build_power_dataset,
)
from snmp_anomaly_detection.streaming.detect_power_kafka import POWER_KAFKA_TOPIC

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None  # type: ignore


_DEFAULT_DEVICE_COUNT = 12
_DEFAULT_NUM_EVENTS = 600
_DEFAULT_SLEEP_S = 0.05
_DEFAULT_ANOMALY_PROB = 0.05

# Must match the baseline model's seq_len so warmup fills the buffer exactly once.
_WARMUP_SEQ_LEN: int = PowerTrainingConfig().seq_len

# Minimum points generated per device regardless of --num-events.
# 300 covers a full circadian cycle (288 × 5-min steps = 24 h).
_MIN_POINTS_PER_DEVICE = 300

# Capacity ranges constrained to values seen in training profiles.
# Keeps output_power_w in-distribution for the fitted scaler and prevents OOD FPs.
# Training profiles: UPS 3000/10000 W, PDU 1440/7200/14400 W, network 150/200 W, env 5 W.
_CAPACITY_RANGE_BY_CATEGORY: dict[str, tuple[float, float]] = {
    "ups":     (2000.0, 12000.0),
    "pdu":     (1200.0, 15000.0),
    "network": (100.0,    250.0),
    "env":     (3.0,       10.0),
}

_VENDORS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "ups":     ("apc", "liebert", "generic"),
    "pdu":     ("apc", "raritan", "generic"),
    "network": ("cisco", "generic"),
    "env":     ("generic",),
}

# Weighted category distribution: UPS is most common in real DC environments
_CATEGORIES: tuple[str, ...] = ("ups", "pdu", "network", "env")
_CATEGORY_WEIGHTS: tuple[int, ...] = (5, 3, 2, 1)


def _require_kafka() -> None:
    if KafkaProducer is None:
        raise ImportError(
            "Kafka support requires `kafka-python`. "
            "Install it and then run `python -m snmp_anomaly_detection produce-power-kafka-test`."
        )


def _random_profiles(device_count: int, rng: random.Random) -> list[PowerDeviceProfile]:
    """Generate `device_count` randomised power device profiles.

    Always seeds with one device of each category before filling the remainder
    randomly, so every run exercises the full pipeline regardless of device_count.
    Capacity ratings are constrained to the training distribution.
    """
    base = list(_CATEGORIES)
    while len(base) < device_count:
        base.append(rng.choices(_CATEGORIES, weights=_CATEGORY_WEIGHTS, k=1)[0])
    rng.shuffle(base)
    assigned = base[:device_count]

    profiles: list[PowerDeviceProfile] = []
    for idx, category in enumerate(assigned):
        vendor = rng.choice(_VENDORS_BY_CATEGORY[category])

        if category in ("ups", "pdu"):
            phase_count = rng.choices([1, 3], weights=[2, 1], k=1)[0]
        else:
            phase_count = 1

        cap_lo, cap_hi = _CAPACITY_RANGE_BY_CATEGORY[category]
        rated_capacity_w = round(rng.uniform(cap_lo, cap_hi), 1)

        if category == "ups":
            battery_ah = rng.choice([7.2, 12.0, 20.0, 40.0])
            battery_life_years = round(rng.uniform(2.0, 5.0), 1)
            install_age_days = round(rng.uniform(0.0, battery_life_years * 365 * 0.9), 1)
        else:
            battery_ah = 0.0
            battery_life_years = 3.0
            install_age_days = 0.0

        profiles.append(PowerDeviceProfile(
            device_id=f"{category}_{vendor}_{idx + 1:02d}",
            device_category=category,
            vendor=vendor,
            phase_count=phase_count,
            rated_capacity_w=rated_capacity_w,
            battery_ah=battery_ah,
            battery_expected_life_years=battery_life_years,
            install_age_days=install_age_days,
        ))

    return profiles


def _interleave_by_poll_cycle(df) -> list[dict]:
    """Sort rows by (synthetic timestamp, device_id) to mirror real SNMP polling order.

    Real SNMP pollers query all devices each interval, so the stream looks like:
      [dev1_t0, dev2_t0, ..., devN_t0, dev1_t1, dev2_t1, ..., devN_t1, ...]
    not device-by-device as the dataset builder emits them.
    """
    return df.sort_values(["timestamp", "device_id"]).to_dict("records")


def _build_message(row: dict, is_warmup: bool, first_for_device: bool) -> dict:
    """Convert a dataset row into a Kafka message payload."""
    message = {k: v for k, v in row.items() if k not in ("anomaly", "anomaly_type")}
    # Replace synthetic 2025-01-01 timestamp with wall-clock time so alert
    # timestamps match the actual test run and compound-alert timing is accurate.
    message["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    # Evaluation fields: allows TP/FP/FN computation from JSONL output offline.
    message["expected_label"] = int(row.get("anomaly", 0))
    message["expected_anomaly_type"] = str(row.get("anomaly_type", "none"))
    message["is_warmup"] = is_warmup
    # Signals the consumer to reset per-device delta state from any prior run.
    if first_for_device:
        message["_device_reset"] = True
    return message


def produce_power_test_events(
    num_events: int = _DEFAULT_NUM_EVENTS,
    device_count: int = _DEFAULT_DEVICE_COUNT,
    sleep_seconds: float = _DEFAULT_SLEEP_S,
    anomaly_probability: float = _DEFAULT_ANOMALY_PROB,
    seed: int | None = None,
    kafka_config: KafkaConfig | None = None,
    use_training_profiles: bool = False,
) -> None:
    _require_kafka()
    kafka_config = kafka_config or KafkaConfig()
    rng = random.Random(seed)

    if use_training_profiles:
        profiles = list(_DEFAULT_POWER_PROFILES)
        print(f"Using {len(profiles)} training profiles (in-distribution, guaranteed no OOD FPs):")
    else:
        profiles = _random_profiles(device_count, rng)
        print(f"Generated {len(profiles)} random device profiles:")
    for p in profiles:
        phase_label = f"{p.phase_count}-phase"
        batt = f"  batt={p.battery_ah}Ah  age={p.install_age_days:.0f}d" if p.battery_ah else ""
        print(f"  {p.device_id:28s}  {p.device_category:8s}  {p.vendor:8s}  {phase_label}  {p.rated_capacity_w:.0f}W{batt}")

    # --- Warmup: guaranteed-normal events so per-device buffers fill cleanly ---
    # seq_len warmup events per device means:
    #   - No anomaly data is present while the rolling buffer is filling.
    #   - Per-device delta state (_prev_raw) is stable before scoring starts.
    #   - The _device_reset signal on the first warmup event clears stale state
    #     from any previous producer run.
    warmup_config = PowerDatasetConfig(
        total_points=_WARMUP_SEQ_LEN + 5,  # +5 ensures head(seq_len) is always satisfiable
        anomaly_probability=0.0,
    )
    warmup_df = build_power_dataset(warmup_config, profiles=profiles)
    warmup_rows = (
        warmup_df
        .sort_values(["timestamp", "device_id"])
        .groupby("device_id", sort=False)
        .head(_WARMUP_SEQ_LEN)
        .sort_values(["timestamp", "device_id"])
        .to_dict("records")
    )

    # --- Main dataset with anomaly injection, interleaved by poll cycle ---
    # total_points raised to _MIN_POINTS_PER_DEVICE so the circadian pattern
    # has room to fully express before the pool runs dry.
    total_points = max(num_events // max(len(profiles), 1), _MIN_POINTS_PER_DEVICE)
    dataset_config = PowerDatasetConfig(
        total_points=total_points,
        anomaly_probability=anomaly_probability,
    )
    main_df = build_power_dataset(dataset_config, profiles=profiles)
    main_rows = _interleave_by_poll_cycle(main_df)[:num_events]

    expected_anomalies = sum(1 for r in main_rows if r.get("anomaly", 0))
    total_to_send = len(warmup_rows) + len(main_rows)
    print(
        f"\nSending {len(warmup_rows)} warmup + {len(main_rows)} main events "
        f"({expected_anomalies} expected anomaly timesteps) → {POWER_KAFKA_TOPIC}"
    )

    producer = KafkaProducer(
        bootstrap_servers=list(kafka_config.bootstrap_servers),
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    seen_devices: set[str] = set()
    produced = 0

    try:
        for is_warmup, rows in ((True, warmup_rows), (False, main_rows)):
            phase_label = "warmup" if is_warmup else "main"
            for row in rows:
                device_id = str(row["device_id"])
                first_for_device = device_id not in seen_devices
                seen_devices.add(device_id)

                message = _build_message(row, is_warmup=is_warmup, first_for_device=first_for_device)
                producer.send(POWER_KAFKA_TOPIC, value=message, key=device_id.encode())
                produced += 1
                if produced % 50 == 0:
                    print(f"  [{phase_label}] {produced}/{total_to_send}")
                time.sleep(sleep_seconds)
    finally:
        producer.flush()
        producer.close()

    print(f"Done. {produced} power events sent to {POWER_KAFKA_TOPIC}.")
    print(f"  warmup={len(warmup_rows)}  main={len(main_rows)}  expected_anomaly_timesteps={expected_anomalies}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce randomised synthetic power SNMP events to Kafka for end-to-end testing.",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=_DEFAULT_DEVICE_COUNT,
        help=(
            f"Number of randomly generated devices (default: {_DEFAULT_DEVICE_COUNT}). "
            "Always includes at least one UPS, PDU, network, and env device."
        ),
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=_DEFAULT_NUM_EVENTS,
        help=(
            f"Total main-phase Kafka messages to produce (default: {_DEFAULT_NUM_EVENTS}). "
            f"A warmup batch of {_WARMUP_SEQ_LEN} events per device is always prepended."
        ),
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=_DEFAULT_SLEEP_S,
        help=f"Delay between messages in seconds (default: {_DEFAULT_SLEEP_S} ≈ 20 msg/s).",
    )
    parser.add_argument(
        "--anomaly-probability",
        type=float,
        default=_DEFAULT_ANOMALY_PROB,
        help=f"Per-event anomaly injection probability (default: {_DEFAULT_ANOMALY_PROB}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible device profiles and data (default: random).",
    )
    parser.add_argument(
        "--use-training-profiles",
        action="store_true",
        default=False,
        help=(
            "Use the exact 11 device profiles from training instead of randomly generated ones. "
            "Guarantees all feature values are in-distribution for the fitted scaler — use this "
            "for smoke tests (--anomaly-probability 0.0) to verify zero false positives. "
            "Ignores --device-count when set."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    produce_power_test_events(
        num_events=args.num_events,
        device_count=args.device_count,
        sleep_seconds=args.sleep_seconds,
        anomaly_probability=args.anomaly_probability,
        seed=args.seed,
        use_training_profiles=args.use_training_profiles,
    )


if __name__ == "__main__":
    main()
