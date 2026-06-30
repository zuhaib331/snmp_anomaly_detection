from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
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
        for timestep in range(config.total_points):
            traffic = generate_normal_pattern(timestep)
            cpu = np.clip(np.random.normal(40, 10), 0, 100)
            memory = np.clip(np.random.normal(60, 15), 0, 100)

            in_octets = traffic * random.uniform(800, 1200)
            out_octets = traffic * random.uniform(700, 1100)
            errors = np.random.poisson(1)

            is_anomaly = 0
            if random.random() < config.anomaly_probability:
                in_octets = inject_anomaly(in_octets)
                out_octets = inject_anomaly(out_octets)
                cpu = inject_anomaly(cpu)
                errors = int(inject_anomaly(errors + 1))
                is_anomaly = 1

            rows.append(
                {
                    "timestamp": current_time,
                    "device_id": device["device_id"],
                    "vendor": device["vendor"],
                    "device_type": device["device_type"],
                    "interface": device["interface"],
                    "cpu": round(cpu, 2),
                    "memory": round(memory, 2),
                    "in_octets": round(in_octets, 2),
                    "out_octets": round(out_octets, 2),
                    "errors": errors,
                    "anomaly": is_anomaly,
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


