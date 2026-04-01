from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from snmp_anomaly_detection.config import ProjectPaths


@dataclass(frozen=True)
class DatasetConfig:
    num_devices: int = 15
    vendors: tuple[str, ...] = ("Cisco", "Dell", "Fortigate")
    device_types: tuple[str, ...] = ("Router", "Switch", "Firewall")
    start_time: datetime = datetime(2025, 1, 1, 0, 0, 0)
    interval_minutes: int = 5
    total_points: int = 2000
    anomaly_probability: float = 0.01
    counter_reset_probability: float = 0.0


@dataclass(frozen=True)
class DeviceProfile:
    base_cpu: float
    base_memory: float
    traffic_scale: float
    interface_speed_mbps: int


def build_device_profile(device_type: str) -> DeviceProfile:
    profiles = {
        "Router": DeviceProfile(
            base_cpu=42.0,
            base_memory=58.0,
            traffic_scale=1.4,
            interface_speed_mbps=1000,
        ),
        "Switch": DeviceProfile(
            base_cpu=28.0,
            base_memory=46.0,
            traffic_scale=1.9,
            interface_speed_mbps=1000,
        ),
        "Firewall": DeviceProfile(
            base_cpu=52.0,
            base_memory=64.0,
            traffic_scale=1.1,
            interface_speed_mbps=1000,
        ),
    }
    return profiles.get(
        device_type,
        DeviceProfile(
            base_cpu=40.0,
            base_memory=55.0,
            traffic_scale=1.0,
            interface_speed_mbps=1000,
        ),
    )


def generate_normal_pattern(timestep: int) -> float:
    day_cycle = np.sin(2 * np.pi * (timestep % 288) / 288)
    base = 50 + 30 * day_cycle
    noise = np.random.normal(0, 5)
    return max(base + noise, 0)


def inject_anomaly(value: float) -> float:
    anomaly_type = random.choice(["spike", "drop", "flat"])

    if anomaly_type == "spike":
        return value * random.uniform(2, 5)
    if anomaly_type == "drop":
        return value * random.uniform(0, 0.2)
    if anomaly_type == "flat":
        return 0
    return value


def pick_anomaly_type() -> str:
    return random.choices(
        population=[
            "traffic_spike",
            "traffic_drop",
            "error_burst",
            "link_down",
            "cpu_spike",
            "memory_spike",
        ],
        weights=[0.30, 0.18, 0.18, 0.10, 0.12, 0.12],
        k=1,
    )[0]


def bytes_per_packet_for_device(device_type: str) -> float:
    baseline = {
        "Router": 900.0,
        "Switch": 750.0,
        "Firewall": 1100.0,
    }.get(device_type, 850.0)
    return max(np.random.normal(baseline, baseline * 0.12), 256.0)


def build_dataset(config: DatasetConfig | None = None) -> pd.DataFrame:
    config = config or DatasetConfig()

    devices = []
    for index in range(config.num_devices):
        devices.append(
            {
                "device_id": f"dev_{index}",
                "vendor": random.choice(config.vendors),
                "device_type": random.choice(config.device_types),
                "interface": f"eth{random.randint(0, 3)}",
            }
        )

    rows = []
    for device in devices:
        current_time = config.start_time
        profile = build_device_profile(device["device_type"])
        cumulative_in_octets = 0.0
        cumulative_out_octets = 0.0
        cumulative_errors = 0
        cumulative_in_ucast_pkts = 0
        cumulative_out_ucast_pkts = 0
        cumulative_in_discards = 0
        cumulative_out_discards = 0
        oper_status = 1
        admin_status = 1

        for timestep in range(config.total_points):
            traffic = generate_normal_pattern(timestep) * profile.traffic_scale
            traffic = max(traffic, 0)
            cpu = np.clip(np.random.normal(profile.base_cpu, 8), 0, 100)
            memory = np.clip(np.random.normal(profile.base_memory, 10), 0, 100)

            in_octets_increment = traffic * random.uniform(800, 1200)
            out_octets_increment = traffic * random.uniform(700, 1100)
            errors_increment = int(np.random.poisson(1))
            bytes_per_packet = bytes_per_packet_for_device(device["device_type"])
            in_packets_increment = int(max(in_octets_increment / bytes_per_packet, 0))
            out_packets_increment = int(max(out_octets_increment / bytes_per_packet, 0))
            in_discards_increment = int(np.random.poisson(0.05))
            out_discards_increment = int(np.random.poisson(0.05))
            anomaly_type = "normal"

            is_anomaly = 0
            if random.random() < config.anomaly_probability:
                anomaly_type = pick_anomaly_type()
                if anomaly_type == "traffic_spike":
                    in_octets_increment *= random.uniform(2.5, 4.5)
                    out_octets_increment *= random.uniform(2.0, 4.0)
                    cpu = float(np.clip(cpu + random.uniform(8, 20), 0, 100))
                elif anomaly_type == "traffic_drop":
                    in_octets_increment *= random.uniform(0.02, 0.20)
                    out_octets_increment *= random.uniform(0.02, 0.20)
                    cpu = float(np.clip(cpu - random.uniform(5, 15), 0, 100))
                elif anomaly_type == "error_burst":
                    errors_increment += random.randint(20, 120)
                    in_discards_increment += random.randint(5, 40)
                    out_discards_increment += random.randint(5, 30)
                    cpu = float(np.clip(cpu + random.uniform(5, 12), 0, 100))
                elif anomaly_type == "link_down":
                    oper_status = 2
                    in_octets_increment = 0.0
                    out_octets_increment = 0.0
                    in_packets_increment = 0
                    out_packets_increment = 0
                    in_discards_increment += random.randint(0, 2)
                    out_discards_increment += random.randint(0, 2)
                    errors_increment += random.randint(1, 10)
                elif anomaly_type == "cpu_spike":
                    cpu = float(np.clip(inject_anomaly(cpu), 0, 100))
                elif anomaly_type == "memory_spike":
                    memory = float(np.clip(inject_anomaly(memory), 0, 100))

                if anomaly_type != "link_down":
                    oper_status = 1
                admin_status = 1
                bytes_per_packet = bytes_per_packet_for_device(device["device_type"])
                in_packets_increment = int(max(in_octets_increment / bytes_per_packet, 0))
                out_packets_increment = int(max(out_octets_increment / bytes_per_packet, 0))
                is_anomaly = 1
            else:
                oper_status = 1
                admin_status = 1

            in_octets_increment = max(in_octets_increment, 0)
            out_octets_increment = max(out_octets_increment, 0)
            errors_increment = max(errors_increment, 0)
            in_packets_increment = max(in_packets_increment, 0)
            out_packets_increment = max(out_packets_increment, 0)
            in_discards_increment = max(in_discards_increment, 0)
            out_discards_increment = max(out_discards_increment, 0)

            counter_reset = 0
            if random.random() < config.counter_reset_probability:
                cumulative_in_octets = float(in_octets_increment)
                cumulative_out_octets = float(out_octets_increment)
                cumulative_errors = int(errors_increment)
                cumulative_in_ucast_pkts = int(in_packets_increment)
                cumulative_out_ucast_pkts = int(out_packets_increment)
                cumulative_in_discards = int(in_discards_increment)
                cumulative_out_discards = int(out_discards_increment)
                counter_reset = 1
            else:
                cumulative_in_octets += float(in_octets_increment)
                cumulative_out_octets += float(out_octets_increment)
                cumulative_errors += int(errors_increment)
                cumulative_in_ucast_pkts += int(in_packets_increment)
                cumulative_out_ucast_pkts += int(out_packets_increment)
                cumulative_in_discards += int(in_discards_increment)
                cumulative_out_discards += int(out_discards_increment)

            rows.append(
                {
                    "timestamp": current_time,
                    "device_id": device["device_id"],
                    "vendor": device["vendor"],
                    "device_type": device["device_type"],
                    "interface": device["interface"],
                    "cpu": round(cpu, 2),
                    "memory": round(memory, 2),
                    "in_octets": round(cumulative_in_octets, 2),
                    "out_octets": round(cumulative_out_octets, 2),
                    "errors": cumulative_errors,
                    "in_ucast_pkts": cumulative_in_ucast_pkts,
                    "out_ucast_pkts": cumulative_out_ucast_pkts,
                    "in_discards": cumulative_in_discards,
                    "out_discards": cumulative_out_discards,
                    "interface_speed_mbps": profile.interface_speed_mbps,
                    "interface_admin_status": admin_status,
                    "interface_oper_status": oper_status,
                    "anomaly": is_anomaly,
                    "anomaly_type": anomaly_type,
                    "counter_reset": counter_reset,
                }
            )
            current_time += timedelta(minutes=config.interval_minutes)

    return pd.DataFrame(rows)


def save_dataset(
    dataframe: pd.DataFrame, output_path: str | None = None, paths: ProjectPaths | None = None
) -> str:
    paths = paths or ProjectPaths()
    paths.ensure_directories()
    target = paths.dataset_file if output_path is None else paths.package_root / output_path
    dataframe.to_csv(target, index=False)
    return str(target)


def main() -> None:
    dataframe = build_dataset()
    output_path = save_dataset(dataframe)
    print(f"Dataset generated: {output_path}")
    print(dataframe.head())


if __name__ == "__main__":
    main()
