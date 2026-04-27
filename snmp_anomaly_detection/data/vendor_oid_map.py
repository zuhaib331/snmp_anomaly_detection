"""
Canonical OID-to-feature mapping for multi-vendor UPS, PDU, and power devices.

Maps vendor-specific SNMP OID names to canonical feature names so that the
training pipeline is decoupled from vendor OID changes.  The canonical names
align with BASELINE_UPS_FEATURES and PHASE_LEVEL_FEATURES in config.py.

Supported vendors:
  - APC        (PowerNet-MIB, enterprise OID 1.3.6.1.4.1.318)
  - Liebert    (LIEBERT-GP-POWER-MIB, enterprise OID 1.3.6.1.4.1.476)
  - RFC 1628   (standard UPS-MIB, any compliant UPS)
  - Raritan    (PX2-MIB, PDU devices)
  - Generic    (ENTITY-MIB / SENSOR-MIB for network gear + environment)
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# APC PowerNet-MIB OID name -> canonical feature name
# Covers: APC SUA series (single-phase), SMT series, SRT series
# ---------------------------------------------------------------------------
APC_OID_MAP: dict[str, str] = {
    # Battery metrics
    "upsAdvBatteryCapacity": "battery_charge_pct",
    "upsAdvBatteryActualVoltage": "battery_voltage_v",
    "upsAdvBatteryTemperature": "battery_temperature_c",
    "upsAdvBatteryRunTimeRemaining": "runtime_remaining_min",  # unit: timeticks -> convert /6000
    "upsAdvBatteryNumOfBadBattPacks": "battery_bad_packs_count",
    # Input metrics
    "upsAdvInputLineVoltage": "input_voltage_avg",
    "upsAdvInputFrequency": "input_frequency_hz",
    # Output metrics
    "upsAdvOutputLoad": "output_load_pct",
    "upsAdvOutputActivePower": "output_power_w",
    "upsAdvOutputVoltage": "output_voltage_v",
    # Status flags (integer encoded)
    "upsAdvOutputStatus": "_apc_output_status_raw",   # decode separately
    "upsBasicOutputStatus": "_apc_basic_status_raw",
    # Environmental (NBES module)
    "uioSensorStatusTemperatureC": "temperature_c",
    "uioSensorStatusHumidity": "humidity_pct",
}

# APC upsAdvOutputStatus values that map to on_battery_flag=1
APC_ON_BATTERY_STATUS_CODES: frozenset[int] = frozenset({2, 3})   # onBattery, onBatteryLow
APC_BYPASS_STATUS_CODES: frozenset[int] = frozenset({5, 7, 8})    # onBypass variants


# ---------------------------------------------------------------------------
# Liebert LIEBERT-GP-POWER-MIB -> canonical feature name
# Covers: Liebert GXT series (single and three-phase)
# ---------------------------------------------------------------------------
LIEBERT_OID_MAP: dict[str, str] = {
    # Battery
    "lgpPwrBatteryCapacityRemaining": "battery_charge_pct",
    "lgpPwrBatteryVoltage": "battery_voltage_v",           # unit: 0.1V -> /10
    "lgpPwrBatteryTemperature": "battery_temperature_c",   # unit: 0.1°C -> /10
    "lgpPwrBatteryEstimatedTime": "runtime_remaining_min", # unit: seconds -> /60
    "lgpPwrNumberFailedBatteryModules": "battery_bad_packs_count",
    # Input (three-phase raw)
    "lgpPwrMeasurementVoltagePhase1": "input_voltage_l1",
    "lgpPwrMeasurementVoltagePhase2": "input_voltage_l2",
    "lgpPwrMeasurementVoltagePhase3": "input_voltage_l3",
    "lgpPwrMeasurementCurrentPhase1": "input_current_l1",
    "lgpPwrMeasurementCurrentPhase2": "input_current_l2",
    "lgpPwrMeasurementCurrentPhase3": "input_current_l3",
    "lgpPwrInputFrequency": "input_frequency_hz",
    # Output
    "lgpPwrOutputPowerWatts": "output_power_w",
    "lgpPwrMeasurementLoadPct": "output_load_pct",
    "lgpPwrOutputCurrentPhase1": "output_current_l1",
    "lgpPwrOutputCurrentPhase2": "output_current_l2",
    "lgpPwrOutputCurrentPhase3": "output_current_l3",
    # Status
    "lgpPwrOnBattery": "on_battery_flag",    # 1=yes, 2=no (normalize to 0/1)
    "lgpPwrBypassMode": "bypass_flag",       # same encoding
    # Environmental (integrated sensor)
    "lgpEnvTemperature": "temperature_c",
    "lgpEnvHumidity": "humidity_pct",
}

# Liebert boolean OID: value=1 means "yes" (flag=1), value=2 means "no" (flag=0)
LIEBERT_BOOL_OID_YES_VALUE: int = 1


# ---------------------------------------------------------------------------
# RFC 1628 UPS-MIB -> canonical feature name
# Compliant with any standards-based UPS
# ---------------------------------------------------------------------------
RFC1628_OID_MAP: dict[str, str] = {
    # Battery group (1.3.6.1.2.1.33.1.2)
    "upsBatteryStatus": "_rfc_battery_status_raw",   # decode separately
    "upsEstimatedChargeRemaining": "battery_charge_pct",
    "upsBatteryVoltage": "battery_voltage_v",         # unit: 0.1V -> /10
    "upsBatteryTemperature": "battery_temperature_c",
    "upsEstimatedMinutesRemaining": "runtime_remaining_min",
    # Input group (1.3.6.1.2.1.33.1.3) — tabular, indexed by phase
    "upsInputVoltage": "input_voltage_avg",   # scalar for single-phase
    "upsInputCurrent": "input_current_l1",    # 0.1A -> /10
    "upsInputFrequency": "input_frequency_hz",  # 0.1Hz -> /10
    # Output group (1.3.6.1.2.1.33.1.4)
    "upsOutputVoltage": "output_voltage_v",
    "upsOutputCurrent": "output_current_l1",
    "upsOutputPower": "output_power_w",
    "upsOutputPercentLoad": "output_load_pct",
    "upsOutputFrequency": "output_frequency_hz",
}

# RFC 1628 upsBatteryStatus values: 1=unknown, 2=batteryNormal, 3=batteryLow, 4=batteryDepleted
RFC1628_ON_BATTERY_STATUS_CODES: frozenset[int] = frozenset({3, 4})


# ---------------------------------------------------------------------------
# Raritan PX2-MIB -> canonical feature name  (PDU devices)
# ---------------------------------------------------------------------------
RARITAN_OID_MAP: dict[str, str] = {
    "pduRMSCurrent": "pdu_total_current_a",
    "pduActivePower": "pdu_total_power_w",
    "pduApparentPower": "pdu_apparent_power_va",
    "pduPowerFactor": "pdu_power_factor",
    "outletActivePower": "outlet_power_w",      # per-outlet (indexed)
    "outletRMSCurrent": "outlet_current_a",
    "outletSwitchingState": "outlet_state",      # 0=off, 1=on
    "pduOverCurrentProtectorStatus": "pdu_overload_flag",
    "inletSensor": "pdu_inlet_voltage_v",
}


# ---------------------------------------------------------------------------
# Generic ENTITY-MIB / SENSOR-MIB (network gear + environmental)
# ---------------------------------------------------------------------------
GENERIC_OID_MAP: dict[str, str] = {
    # ENTITY-MIB power (RFC 6933)
    "entPhysicalPowerAllocated": "psu_allocated_mw",  # milliwatts
    "entPhysicalPowerUsed": "psu_used_mw",
    # Cisco ENTITY-FRU-CONTROL-MIB
    "cefcFRUPowerOperStatus": "psu_oper_status",   # 1=on, 2=off, 3=failed
    "cefcFRUPowerAdminStatus": "psu_admin_status",
    # SENSOR-MIB (RFC 3433)
    "entPhySensorValue": "sensor_value_raw",
    "entPhySensorType": "sensor_type_raw",
    # Temperature sensors (common OIDs)
    "ciscoEnvMonTemperatureStatusValue": "temperature_c",
    "lmTempSensorsValue": "temperature_c",          # unit: millidegrees C -> /1000
    # Humidity
    "lmMiscSensorsValue": "humidity_pct",
}


# ---------------------------------------------------------------------------
# Unified resolver
# ---------------------------------------------------------------------------

_ALL_MAPS: dict[str, dict[str, str]] = {
    "apc": APC_OID_MAP,
    "liebert": LIEBERT_OID_MAP,
    "rfc1628": RFC1628_OID_MAP,
    "raritan": RARITAN_OID_MAP,
    "generic": GENERIC_OID_MAP,
}


def resolve_to_canonical(vendor: str, raw_oid_name: str) -> Optional[str]:
    """Return canonical feature name for a vendor OID name, or None if unknown.

    Canonical names beginning with '_' are vendor-specific raw values that
    require additional decoding (e.g., status enums) and should not be fed
    directly into the model.
    """
    oid_map = _ALL_MAPS.get(vendor.lower())
    if oid_map is None:
        return None
    canonical = oid_map.get(raw_oid_name)
    if canonical and canonical.startswith("_"):
        return None   # internal raw field, not directly usable
    return canonical


def list_vendor_features(vendor: str) -> list[str]:
    """Return all canonical feature names exposed by a vendor (excludes raw fields)."""
    oid_map = _ALL_MAPS.get(vendor.lower(), {})
    return [v for v in oid_map.values() if not v.startswith("_")]


def decode_apc_status_flags(output_status: int) -> dict[str, int]:
    """Decode APC upsAdvOutputStatus integer into on_battery_flag and bypass_flag."""
    return {
        "on_battery_flag": int(output_status in APC_ON_BATTERY_STATUS_CODES),
        "bypass_flag": int(output_status in APC_BYPASS_STATUS_CODES),
    }


def decode_liebert_bool(value: int) -> int:
    """Normalize Liebert boolean OID (1=yes → 1, 2=no → 0)."""
    return 1 if value == LIEBERT_BOOL_OID_YES_VALUE else 0


def decode_rfc1628_battery_flags(battery_status: int) -> dict[str, int]:
    """Decode RFC 1628 upsBatteryStatus into on_battery_flag."""
    return {
        "on_battery_flag": int(battery_status in RFC1628_ON_BATTERY_STATUS_CODES),
        "bypass_flag": 0,  # not tracked by RFC 1628
    }
