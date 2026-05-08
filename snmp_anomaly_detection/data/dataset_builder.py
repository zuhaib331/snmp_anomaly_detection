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


# ---------------------------------------------------------------------------
# Power SNMP dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerDeviceProfile:
    """Physical characteristics of a power device used during synthetic generation."""
    device_id: str
    device_category: str           # ups | pdu | network | env
    vendor: str                    # for logging only — does not drive data generation logic
    phase_count: int               # 1 or 3
    rated_capacity_w: float        # nameplate rating in Watts
    nominal_voltage_v: float       # 120.0 (US/Japan) or 230.0 (Europe/Asia) — explicit, not inferred from vendor
    battery_ah: float              # Amp-hour capacity (UPS only; 0 for non-UPS)
    rated_battery_v: float         # nominal battery string voltage (UPS only; 0 for non-UPS)
    battery_expected_life_years: float = 3.0
    install_age_days: float = 0.0  # simulated age at dataset start


@dataclass(frozen=True)
class PowerDatasetConfig:
    start_time: datetime = field(default_factory=lambda: datetime(2025, 1, 1, 0, 0, 0))
    interval_minutes: int = 5
    total_points: int = 2016       # ~7 days at 5-min intervals
    anomaly_probability: float = 0.05


# Parameter sweep across capacity × voltage × phase × battery-age space.
# Vendor field is kept for logging/identification only — it does not drive any
# data-generation logic.  All physics parameters are explicit on the profile.
# rated_battery_v: 24V (≤3 kW), 48V (5 kW), 96V (7.5 kW), 120V (10 kW 3Φ), 192V (15 kW 3Φ)
_DEFAULT_POWER_PROFILES: list[PowerDeviceProfile] = [
    # ── UPS: 1 kW / 120 V / single-phase ──────────────────────────────────
    PowerDeviceProfile("ups_1kw_120v_a", "ups", "generic", 1, 1000.0, 120.0,  7.2, 24.0, 3.0,   90.0),
    PowerDeviceProfile("ups_1kw_120v_b", "ups", "generic", 1, 1000.0, 120.0,  7.2, 24.0, 3.0,  720.0),
    # ── UPS: 3 kW / 120 V / single-phase ──────────────────────────────────
    PowerDeviceProfile("ups_3kw_120v_a", "ups", "apc",     1, 3000.0, 120.0,  7.2, 24.0, 3.0,   90.0),
    PowerDeviceProfile("ups_3kw_120v_b", "ups", "apc",     1, 3000.0, 120.0,  7.2, 24.0, 3.0,  365.0),
    PowerDeviceProfile("ups_3kw_120v_c", "ups", "apc",     1, 3000.0, 120.0,  7.2, 24.0, 3.0,  900.0),
    # ── UPS: 5 kW / 120 V / single-phase ──────────────────────────────────
    PowerDeviceProfile("ups_5kw_120v_a", "ups", "apc",     1, 5000.0, 120.0, 18.0, 48.0, 4.0,  180.0),
    # ── UPS: 5 kW / 230 V / single-phase ──────────────────────────────────
    PowerDeviceProfile("ups_5kw_230v_a", "ups", "eaton",   1, 5000.0, 230.0, 18.0, 48.0, 4.0,  200.0),
    # ── UPS: 7.5 kW / 230 V / single-phase ───────────────────────────────
    PowerDeviceProfile("ups_7k5_230v_a", "ups", "eaton",   1, 7500.0, 230.0, 20.0, 96.0, 5.0,  150.0),
    PowerDeviceProfile("ups_7k5_230v_b", "ups", "eaton",   1, 7500.0, 230.0, 20.0, 96.0, 5.0,  800.0),
    # ── UPS: 10 kW / 230 V / three-phase ─────────────────────────────────
    PowerDeviceProfile("ups_10kw_230v_a", "ups", "liebert", 3, 10000.0, 230.0, 40.0, 120.0, 5.0, 365.0),
    PowerDeviceProfile("ups_10kw_230v_b", "ups", "liebert", 3, 10000.0, 230.0, 40.0, 120.0, 5.0, 700.0),
    PowerDeviceProfile("ups_10kw_230v_c", "ups", "liebert", 3, 10000.0, 230.0, 40.0, 120.0, 5.0, 500.0),
    # ── UPS: 15 kW / 230 V / three-phase ─────────────────────────────────
    PowerDeviceProfile("ups_15kw_230v_a", "ups", "liebert", 3, 15000.0, 230.0, 40.0, 192.0, 5.0,  180.0),
    PowerDeviceProfile("ups_15kw_230v_b", "ups", "liebert", 3, 15000.0, 230.0, 40.0, 192.0, 5.0, 1000.0),
    # ── PDU: parameter sweep ──────────────────────────────────────────────
    PowerDeviceProfile("pdu_1k4_120v", "pdu", "apc",     1,  1440.0, 120.0, 0.0, 0.0, 0.0, 0.0),
    PowerDeviceProfile("pdu_3k6_120v", "pdu", "raritan", 1,  3600.0, 120.0, 0.0, 0.0, 0.0, 0.0),
    PowerDeviceProfile("pdu_7k2_230v", "pdu", "raritan", 1,  7200.0, 230.0, 0.0, 0.0, 0.0, 0.0),
    PowerDeviceProfile("pdu_14k_230v", "pdu", "raritan", 3, 14400.0, 230.0, 0.0, 0.0, 0.0, 0.0),
    # ── Network PSUs ──────────────────────────────────────────────────────
    PowerDeviceProfile("net_200w_120v", "network", "cisco",   1, 200.0, 120.0, 0.0, 0.0, 0.0, 0.0),
    PowerDeviceProfile("net_150w_120v", "network", "generic", 1, 150.0, 120.0, 0.0, 0.0, 0.0, 0.0),
    # ── Environmental sensor ──────────────────────────────────────────────
    PowerDeviceProfile("env_5w_120v", "env", "generic", 1, 5.0, 120.0, 0.0, 0.0, 0.0, 0.0),
]


def _circadian_load_factor(timestep: int, interval_minutes: int) -> float:
    """Return a 0–1 load factor following a business-hours circadian pattern."""
    minutes_in_day = 24 * 60
    minute_of_day = (timestep * interval_minutes) % minutes_in_day
    # Peak: 09:00–18:00 (minutes 540–1080), valley: 01:00–05:00
    angle = 2 * math.pi * (minute_of_day - 540) / minutes_in_day
    base = 0.55 + 0.35 * math.sin(angle - math.pi / 2)
    return float(np.clip(base + np.random.normal(0, 0.03), 0.1, 1.0))


def _battery_voltage_from_charge(charge_pct: float, rated_battery_v: float = 24.0) -> float:
    """Approximate lead-acid string voltage (V) from charge % and rated string voltage.

    Scales linearly from 83.3% of rated (depleted) to 110% of rated (full),
    matching the classic 24V pack behaviour: 20.0 V → 26.4 V at rated_battery_v=24.
    """
    return rated_battery_v * (0.833 + 0.267 * (charge_pct / 100.0))


def _runtime_estimate(battery_ah: float, charge_pct: float, load_w: float, voltage_v: float = 24.0) -> float:
    """Estimate runtime in minutes from battery state and current load."""
    if load_w <= 0 or battery_ah <= 0:
        return 999.0
    # Energy available = capacity * charge% * voltage; load in Wh
    energy_wh = battery_ah * (charge_pct / 100.0) * voltage_v
    runtime_h = energy_wh / load_w * 0.85   # 85% inverter efficiency
    return float(np.clip(runtime_h * 60, 0.0, 999.0))


def _battery_degradation_factor(install_age_days: float, expected_life_years: float) -> float:
    """Return 0–1 capacity factor (1=new, approaching 0 as battery ages)."""
    age_fraction = install_age_days / (expected_life_years * 365)
    return float(np.clip(1.0 - 0.5 * age_fraction, 0.1, 1.0))


def _phase_voltages(nominal_v: float, phase_count: int) -> tuple[float, float, float]:
    """Return (l1, l2, l3) voltages; for single-phase l2==l3==l1."""
    l1 = nominal_v + np.random.normal(0, 1.0)
    if phase_count == 1:
        return (l1, l1, l1)
    # Three-phase: small independent variation per phase
    l2 = nominal_v + np.random.normal(0, 1.2)
    l3 = nominal_v + np.random.normal(0, 1.2)
    return (l1, l2, l3)


def _inject_power_anomaly(
    row: dict,
    anomaly_type: str,
    profile: PowerDeviceProfile,
    event: "_AnomalyEvent | None" = None,
) -> dict:
    """Mutate row in-place with a named power anomaly pattern.

    event: pre-sampled event parameters (add_load, max_temp_rise, drain_amount).
    Overload uses additive injection so that even low-background timesteps produce
    clearly above-normal load values, giving the LSTM a strong learnable signature.
    All other fault types apply full-strength injection uniformly throughout the event.
    """
    nominal_v = profile.nominal_voltage_v

    if anomaly_type == "battery_drain":
        drain = event.drain_amount if event else random.uniform(40, 70)
        row["battery_charge_pct"] = float(np.clip(row["battery_charge_pct"] - drain, 0, 100))
        row["runtime_remaining_min"] = _runtime_estimate(
            profile.battery_ah, row["battery_charge_pct"], row["output_power_w"]
        )
        row["on_battery_status"] = 1.0
        row["battery_current_a"] = -float(row["output_power_w"] / max(row["battery_voltage_v"], 1.0))

    elif anomaly_type == "overload":
        # Additive injection: background load + fixed addition guarantees load is always
        # significantly above normal regardless of the circadian baseline at injection time.
        add_load = event.add_load if event else random.uniform(40, 70)
        row["output_load_pct"] = float(np.clip(row["output_load_pct"] + add_load, 0, 120))
        row["output_power_w"] = float(row["output_load_pct"] / 100.0 * profile.rated_capacity_w)
        load_ratio = row["output_load_pct"] / max(row["output_load_pct"] - add_load, 1.0)
        for ph in ("l1", "l2", "l3"):
            if f"output_current_{ph}" in row:
                row[f"output_current_{ph}"] *= load_ratio
        row["output_current_a"] = float(row["output_power_w"] / max(row.get("output_voltage_v", nominal_v) * 0.95, 1.0))

    elif anomaly_type == "phase_sag":
        row["input_voltage_l1"] = float(row["input_voltage_l1"] * random.uniform(0.6, 0.8))
        avg = (row["input_voltage_l1"] + row["input_voltage_l2"] + row["input_voltage_l3"]) / 3
        if avg > 0:
            row["voltage_imbalance_pct"] = (
                max(abs(row["input_voltage_l1"] - avg),
                    abs(row["input_voltage_l2"] - avg),
                    abs(row["input_voltage_l3"] - avg)) / avg * 100
            )
        row["output_voltage_v"] = float(row.get("output_voltage_v", nominal_v) * random.uniform(0.92, 0.98))

    elif anomaly_type == "thermal_runaway":
        temp_rise = event.max_temp_rise if event else random.uniform(15, 35)
        row["battery_temperature_c"] = float(row["battery_temperature_c"] + temp_rise)
        row["output_power_w"] = float(row["output_power_w"] * random.uniform(1.2, 1.8))
        row["output_current_a"] = float(row["output_power_w"] / max(row.get("output_voltage_v", nominal_v) * 0.95, 1.0))

    elif anomaly_type == "psu_failure":
        row["output_load_pct"] = 0.0
        row["output_power_w"] = 0.0
        row["output_current_a"] = 0.0
        row["output_voltage_v"] = 0.0
        row["output_frequency_hz"] = 0.0
        row["battery_current_a"] = 0.0
        row["on_battery_status"] = 1.0
        row["bypass_flag"] = 1.0

    return row


def _anomaly_types_for_category(category: str) -> list[str]:
    if category == "ups":
        return ["battery_drain", "overload", "phase_sag", "thermal_runaway", "psu_failure"]
    if category == "pdu":
        return ["overload", "phase_sag", "psu_failure"]
    if category == "network":
        return ["overload", "thermal_runaway", "psu_failure"]
    return ["overload"]


# Event duration ranges (timesteps) per anomaly type.
# Overload and thermal_runaway are intentionally long so the LSTM sees the ramp-up.
_EVENT_DURATION_RANGE: dict[str, tuple[int, int]] = {
    "overload":        (30, 60),
    "thermal_runaway": (24, 48),
    "battery_drain":   (12, 24),
    "phase_sag":       (10, 20),
    "psu_failure":     (6,  12),
}
_AVG_EVENT_DURATION = 25  # approximate mean across types, used for rate calibration


@dataclass(frozen=True)
class _AnomalyEvent:
    start: int
    end: int                    # exclusive
    anomaly_type: str
    # Event-level parameters sampled once so every timestep in the event is consistent.
    # Overload uses additive load (pp) rather than a multiplier so that even low-background
    # timesteps produce clearly above-normal load and the LSTM learns a strong ramp signature.
    add_load: float = 0.0       # overload: load percentage points to add (constant per step)
    max_temp_rise: float = 0.0  # thermal_runaway: temperature rise applied every event step
    drain_amount: float = 0.0   # battery_drain: charge subtraction applied every event step


def _generate_anomaly_events(
    total_points: int,
    anomaly_probability: float,
    device_category: str,
    phase_count: int,
) -> list[_AnomalyEvent]:
    """Pre-generate non-overlapping multi-timestep anomaly events for one device.

    The event start probability is scaled by _AVG_EVENT_DURATION so that total
    anomaly timesteps ≈ total_points * anomaly_probability, preserving the overall
    anomaly fraction seen during training.
    """
    event_probability = anomaly_probability / _AVG_EVENT_DURATION
    events: list[_AnomalyEvent] = []
    t = 0
    while t < total_points:
        if random.random() < event_probability:
            anomaly_type = random.choice(_anomaly_types_for_category(device_category))
            if anomaly_type == "phase_sag" and phase_count == 1:
                anomaly_type = "overload"
            min_dur, max_dur = _EVENT_DURATION_RANGE[anomaly_type]
            duration = random.randint(min_dur, max_dur)
            end = min(t + duration, total_points)
            events.append(_AnomalyEvent(
                start=t,
                end=end,
                anomaly_type=anomaly_type,
                add_load=random.uniform(55, 75) if anomaly_type == "overload" else 0.0,
                max_temp_rise=random.uniform(15, 35) if anomaly_type == "thermal_runaway" else 0.0,
                drain_amount=random.uniform(40, 70) if anomaly_type == "battery_drain" else 0.0,
            ))
            # Skip ahead past the event plus a minimum quiet gap
            t = end + random.randint(6, 24)
        else:
            t += 1
    return events


def _build_power_row(
    profile: PowerDeviceProfile,
    timestamp: datetime,
    timestep: int,
    config: PowerDatasetConfig,
    charge_pct: float,
    discharge_cycles: float,
) -> tuple[dict, float]:
    """Build one power SNMP row; returns the row dict and updated charge_pct."""
    load_factor = _circadian_load_factor(timestep, config.interval_minutes)
    load_pct = float(np.clip(load_factor * 100 + np.random.normal(0, 2), 5, 100))
    output_power_w = load_pct / 100.0 * profile.rated_capacity_w

    # Battery metrics (UPS only; PDU/network/env get neutral values)
    deg = 1.0  # capacity degradation factor — overwritten for UPS below
    charge_delta = 0.0
    if profile.battery_ah > 0:
        deg = _battery_degradation_factor(
            profile.install_age_days + timestep * config.interval_minutes / 1440,
            profile.battery_expected_life_years,
        )
        # Slow trickle charge during low-load periods; slight discharge otherwise
        charge_delta = 0.02 if load_pct < 40 else -0.01
        charge_pct = float(np.clip(charge_pct + charge_delta + np.random.normal(0, 0.05), 0, 100))
        effective_charge = charge_pct * deg
        batt_voltage = _battery_voltage_from_charge(effective_charge, profile.rated_battery_v)
        batt_temp = 25.0 + load_factor * 8 + np.random.normal(0, 0.5)
        runtime_min = _runtime_estimate(profile.battery_ah, effective_charge, output_power_w, profile.rated_battery_v)
    else:
        charge_pct = 100.0
        batt_voltage = 0.0
        batt_temp = 20.0 + np.random.normal(0, 0.3)
        runtime_min = 0.0

    # Nominal input voltage comes from the explicit profile field (not inferred from vendor)
    nominal_input_v = profile.nominal_voltage_v
    l1, l2, l3 = _phase_voltages(nominal_input_v, profile.phase_count)
    avg_v = (l1 + l2 + l3) / 3
    vol_imbalance = (max(abs(l1 - avg_v), abs(l2 - avg_v), abs(l3 - avg_v)) / avg_v * 100
                     if avg_v > 0 else 0.0)

    # Per-phase currents (P = V * I * PF; assume PF=0.95)
    pf = 0.95
    phase_power = output_power_w / profile.phase_count
    i_l1 = phase_power / (l1 * pf) if l1 > 0 else 0.0
    i_l2 = phase_power / (l2 * pf) if l2 > 0 else 0.0
    i_l3 = phase_power / (l3 * pf) if l3 > 0 else 0.0
    avg_i = (i_l1 + i_l2 + i_l3) / 3
    cur_skew = (max(abs(i_l1 - avg_i), abs(i_l2 - avg_i), abs(i_l3 - avg_i)) / avg_i * 100
                if avg_i > 0 else 0.0)

    # charge_rate: percentage points per interval (positive = charging)
    charge_rate = charge_delta + np.random.normal(0, 0.02)

    # --- New canonical features ---
    # battery_current_a: positive = trickle charging (mains), negative = discharging (on battery)
    battery_current_a = float(profile.battery_ah * 0.02 + np.random.normal(0, 0.05)) if profile.battery_ah > 0 else 0.0
    # battery_replace_status: 1 when degradation drops below 60% of rated capacity
    battery_replace_status = 1.0 if (profile.battery_ah > 0 and deg < 0.6) else 0.0
    # output_voltage_v: inverter/bypass output voltage (close to nominal with slight noise)
    output_voltage_v = float(nominal_input_v + np.random.normal(0, 1.0))
    # output_frequency_hz: output AC frequency (inverter tracks input; slight independent noise)
    output_frequency_hz = float(50.0 + np.random.normal(0, 0.02))
    # output_current_a: aggregate AC output current = P / (V * PF)
    output_current_a = float(output_power_w / max(output_voltage_v * pf, 1.0))

    row: dict = {
        "timestamp": timestamp,
        "device_id": profile.device_id,
        "device_category": profile.device_category,
        "vendor": profile.vendor,
        "phase_count": profile.phase_count,
        # Device registration constants — used by normalize_absolute_features() in preprocessing.
        # In production these come from SNMP discovery or device registration (not computed here).
        "rated_capacity_w": profile.rated_capacity_w,
        "nominal_voltage_v": profile.nominal_voltage_v,
        "rated_battery_v": profile.rated_battery_v,
        # BASELINE_UPS_FEATURES — canonical names
        "battery_charge_pct": round(charge_pct, 2),
        "battery_voltage_v": round(batt_voltage, 3),
        "battery_current_a": round(battery_current_a, 3),
        "battery_temperature_c": round(batt_temp, 2),
        "runtime_remaining_min": round(runtime_min, 1),
        "on_battery_status": 0.0,
        "battery_replace_status": battery_replace_status,
        "input_voltage_v": round(avg_v, 2),
        "input_frequency_hz": round(50.0 + np.random.normal(0, 0.02), 3),
        "output_voltage_v": round(output_voltage_v, 2),
        "output_current_a": round(output_current_a, 3),
        "output_load_pct": round(load_pct, 2),
        "output_frequency_hz": round(output_frequency_hz, 3),
        "output_power_w": round(output_power_w, 1),
        # metadata (not a model feature)
        "bypass_flag": 0.0,
        # PHASE_LEVEL_FEATURES additions
        "input_voltage_l1": round(l1, 2),
        "input_voltage_l2": round(l2, 2),
        "input_voltage_l3": round(l3, 2),
        "input_current_l1": round(i_l1, 3),
        "input_current_l2": round(i_l2, 3),
        "input_current_l3": round(i_l3, 3),
        "output_current_l1": round(i_l1, 3),
        "output_current_l2": round(i_l2, 3),
        "output_current_l3": round(i_l3, 3),
        "voltage_imbalance_pct": round(vol_imbalance, 3),
        "current_skew_pct": round(cur_skew, 3),
        # BATTERY_RUL_FEATURES extras
        "charge_rate": round(charge_rate, 4),
        "discharge_cycles_approx": round(discharge_cycles, 2),
        # Device metadata — used by derive_rul_labels; not a model feature
        "install_age_days": profile.install_age_days,
        # Label
        "anomaly": 0,
        "anomaly_type": "none",
    }
    return row, charge_pct


def build_power_dataset(
    config: PowerDatasetConfig | None = None,
    profiles: list[PowerDeviceProfile] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic multi-vendor power SNMP dataset."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    config = config or PowerDatasetConfig()
    profiles = profiles or _DEFAULT_POWER_PROFILES

    rows: list[dict] = []
    for profile in profiles:
        charge_pct = random.uniform(85, 100)
        discharge_cycles = profile.install_age_days / 180.0
        current_time = config.start_time

        # Pre-generate multi-timestep events so overload/thermal_runaway have gradual ramps
        events = _generate_anomaly_events(
            config.total_points,
            config.anomaly_probability,
            profile.device_category,
            profile.phase_count,
        )
        # Build a fast lookup: timestep → event
        timestep_to_event: dict[int, _AnomalyEvent] = {}
        for ev in events:
            for t in range(ev.start, ev.end):
                timestep_to_event[t] = ev

        for timestep in range(config.total_points):
            row, charge_pct = _build_power_row(
                profile, current_time, timestep, config, charge_pct, discharge_cycles
            )

            if timestep in timestep_to_event:
                ev = timestep_to_event[timestep]
                row = _inject_power_anomaly(row, ev.anomaly_type, profile, ev)
                row["anomaly"] = 1
                row["anomaly_type"] = ev.anomaly_type

            rows.append(row)
            current_time += timedelta(minutes=config.interval_minutes)

    return pd.DataFrame(rows)


def save_power_dataset(
    dataframe: pd.DataFrame,
    output_path: str | None = None,
    paths: ProjectPaths | None = None,
) -> str:
    paths = paths or ProjectPaths()
    paths.ensure_power_directories()
    target = paths.power_dataset_file if output_path is None else paths.package_root / output_path
    dataframe.to_csv(target, index=False)
    return str(target)


if __name__ == "__main__":
    main()
