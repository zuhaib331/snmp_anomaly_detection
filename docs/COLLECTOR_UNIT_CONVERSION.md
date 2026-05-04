# SNMP Collector — Unit Conversion Contract

This document is for whoever builds or maintains the SNMP collector/producer that publishes
power device telemetry to the `snmp-power-events` Kafka topic.

The anomaly detection pipeline (`PowerStreamProcessor`) expects all values to arrive in
**SI units with canonical feature names**. It performs no OID resolution and no unit
conversion internally. If raw SNMP values are published without conversion, the model
receives silently wrong inputs — there is no error or warning raised.

---

## Kafka message format

Each message on `snmp-power-events` must deserialize into a `PowerEvent`. The relevant
fields that require attention from the collector are:

```json
{
  "timestamp": "2026-05-04T10:00:00",
  "device_id": "ups_rack1_floor2",
  "device_category": "ups",
  "vendor": "apc",
  "phase_count": 1,
  "rated_capacity_w": 3000.0,
  "nominal_voltage_v": 120.0,
  "rated_battery_v": 24.0,
  "feature_values": {
    "battery_charge_pct": 94.5,
    "battery_voltage_v": 25.8,
    "battery_current_a": 1.2,
    "battery_temperature_c": 27.3,
    "runtime_remaining_min": 42.0,
    "on_battery_status": 0.0,
    "battery_replace_status": 0.0,
    "input_voltage_v": 120.3,
    "input_frequency_hz": 60.0,
    "output_voltage_v": 120.1,
    "output_current_a": 15.6,
    "output_load_pct": 62.4,
    "output_frequency_hz": 60.0,
    "output_power_w": 1872.0,
    "input_voltage_l1": 120.3,
    "input_voltage_l2": 120.3,
    "input_voltage_l3": 120.3,
    "input_current_l1": 5.2,
    "input_current_l2": 5.2,
    "input_current_l3": 5.2,
    "output_current_l1": 5.2,
    "output_current_l2": 5.2,
    "output_current_l3": 5.2,
    "voltage_imbalance_pct": 0.08,
    "current_skew_pct": 0.0
  }
}
```

---

## Measurement features — OID sources and required conversions

### Battery

| Feature name | Expected unit | RFC 1628 OID | RFC 1628 raw unit | Conversion | APC OID | APC raw unit | Conversion | Liebert GP OID | Liebert raw unit |
|---|---|---|---|---|---|---|---|---|---|
| `battery_charge_pct` | % (0–100) | `1.3.6.1.2.1.33.1.2.4` `upsBatteryCapacity` | % | none | `1.3.6.1.4.1.318.1.1.1.2.2.1` `upsAdvBatteryCapacity` | % | none | `1.3.6.1.4.1.476.1.42.3.5.2.1` `lgpPwrBatteryCapacityPct` | % | none |
| `battery_voltage_v` | Volts | `1.3.6.1.2.1.33.1.2.5` `upsBatteryVoltage` | **0.1 V** | **÷ 10** | `1.3.6.1.4.1.318.1.1.1.2.2.8` `upsAdvBatteryActualVoltage` | Volts | none | `1.3.6.1.4.1.476.1.1.1.1.1.2.3` `lcUpsBatVoltage` | Volts | none |
| `battery_current_a` | Amps | `1.3.6.1.2.1.33.1.2.6` `upsBatteryCurrent` | **0.1 A** | **÷ 10** | `1.3.6.1.4.1.318.1.1.1.2.2.9` `upsAdvBatteryActualCurrent` | Amps | none | `1.3.6.1.4.1.476.1.1.1.1.1.2.4` `lcUpsBatCurrent` | Amps | none |
| `battery_temperature_c` | °C | `1.3.6.1.2.1.33.1.2.7` `upsBatteryTemperature` | °C | none | `1.3.6.1.4.1.318.1.1.1.2.2.2` `upsAdvBatteryTemperature` | °C | none | `1.3.6.1.4.1.476.1.1.1.1.1.2.5` `lcUpsBatTemperature` | °C | none |
| `runtime_remaining_min` | Minutes | `1.3.6.1.2.1.33.1.2.3` `upsEstimatedMinutesRemaining` | Minutes | none | `1.3.6.1.4.1.318.1.1.1.2.2.3` `upsAdvBatteryRunTimeRemaining` | **TimeTicks (1/100 s)** | **÷ 6000** | `1.3.6.1.4.1.476.1.42.3.5.1.18` `lgpPwrBatteryEstimatedRunTime` | Minutes | none |

**APC `runtime_remaining_min` example:**
Raw SNMP value = `2520000` TimeTicks → `2520000 ÷ 6000 = 420 minutes`

**RFC 1628 `battery_voltage_v` example:**
Raw SNMP value = `240` → `240 ÷ 10 = 24.0 V`

---

### Input power

| Feature name | Expected unit | RFC 1628 OID | RFC 1628 raw unit | Conversion | APC OID | APC raw unit | Conversion |
|---|---|---|---|---|---|---|---|
| `input_voltage_v` | Volts | `1.3.6.1.2.1.33.1.3.3.1.3` `upsInputVoltage` (table, index 1) | Volts | none | `1.3.6.1.4.1.318.1.1.1.3.2.1` `upsAdvInputLineVoltage` | Volts | none |
| `input_frequency_hz` | Hz | `1.3.6.1.2.1.33.1.3.3.1.2` `upsInputFrequency` (table, index 1) | **0.1 Hz** | **÷ 10** | `1.3.6.1.4.1.318.1.1.1.3.2.4` `upsAdvInputFrequency` | Hz | none |

**Note:** For three-phase devices, `input_voltage_v` should be the mean of L1, L2, L3.
The preprocessing pipeline recomputes this automatically when per-phase columns are present —
but the per-phase values themselves must still be in Volts.

---

### Output power

| Feature name | Expected unit | RFC 1628 OID | RFC 1628 raw unit | Conversion | APC OID | APC raw unit | Conversion |
|---|---|---|---|---|---|---|---|
| `output_voltage_v` | Volts | `1.3.6.1.2.1.33.1.4.4.1.2` `upsOutputVoltage` (table) | Volts | none | `1.3.6.1.4.1.318.1.1.1.4.2.1` `upsAdvOutputVoltage` | Volts | none |
| `output_current_a` | Amps | `1.3.6.1.2.1.33.1.4.4.1.3` `upsOutputCurrent` (table) | **0.1 A** | **÷ 10** | `1.3.6.1.4.1.318.1.1.1.4.2.4` `upsAdvOutputCurrent` | Amps | none |
| `output_load_pct` | % (0–100) | `1.3.6.1.2.1.33.1.4.4.1.5` `upsOutputPercentLoad` (table) | % | none | `1.3.6.1.4.1.318.1.1.1.4.2.3` `upsAdvOutputLoad` | % | none |
| `output_frequency_hz` | Hz | `1.3.6.1.2.1.33.1.4.2` `upsOutputFrequency` | **0.1 Hz** | **÷ 10** | `1.3.6.1.4.1.318.1.1.1.4.2.2` `upsAdvOutputFrequency` | Hz | none |
| `output_power_w` | Watts | `1.3.6.1.2.1.33.1.4.4.1.4` `upsOutputPower` (table) | Watts | none | `1.3.6.1.4.1.318.1.1.1.4.2.8` `upsAdvOutputActivePower` | Watts | none |

---

### Per-phase values (three-phase UPS only)

Per-phase columns are used by the **phase model** only. For single-phase devices, send
`input_voltage_l1 = input_voltage_l2 = input_voltage_l3 = input_voltage_v`.

| Feature name | Expected unit | RFC 1628 OID | Notes |
|---|---|---|---|
| `input_voltage_l1` | Volts | `upsInputVoltage` table row 1 | Volts — no conversion |
| `input_voltage_l2` | Volts | `upsInputVoltage` table row 2 | Volts — no conversion |
| `input_voltage_l3` | Volts | `upsInputVoltage` table row 3 | Volts — no conversion |
| `input_current_l1` | Amps | `upsInputCurrent` table row 1 (`1.3.6.1.2.1.33.1.3.3.1.4`) | **RFC1628: 0.1 A → ÷ 10** |
| `input_current_l2` | Amps | `upsInputCurrent` table row 2 | **RFC1628: 0.1 A → ÷ 10** |
| `input_current_l3` | Amps | `upsInputCurrent` table row 3 | **RFC1628: 0.1 A → ÷ 10** |
| `output_current_l1` | Amps | `upsOutputCurrent` table row 1 | **RFC1628: 0.1 A → ÷ 10** |
| `output_current_l2` | Amps | `upsOutputCurrent` table row 2 | **RFC1628: 0.1 A → ÷ 10** |
| `output_current_l3` | Amps | `upsOutputCurrent` table row 3 | **RFC1628: 0.1 A → ÷ 10** |
| `voltage_imbalance_pct` | % | Computed by collector | `max(|L1-avg|, |L2-avg|, |L3-avg|) / avg * 100` |
| `current_skew_pct` | % | Computed by collector | `max(|I1-avg|, |I2-avg|, |I3-avg|) / avg * 100` |

`voltage_imbalance_pct` and `current_skew_pct` are **not polled directly** — compute them
from the per-phase values after conversion.

---

## Registration constants (send once at device onboarding, resend on every message)

These three values drive the B2 normalization step inside the pipeline. Without them,
the 5 normalized features (`battery_voltage_ratio`, `battery_current_ratio`,
`input_voltage_dev_pct`, `output_voltage_dev_pct`, `output_current_ratio`) silently
become 0.0 and the model receives corrupted input.

| Field | Expected unit | How to obtain | Notes |
|---|---|---|---|
| `rated_capacity_w` | Watts | See table below | Nameplate power rating |
| `nominal_voltage_v` | Volts | See table below | 120.0 (US/Japan) or 230.0 (EU/Asia) |
| `rated_battery_v` | Volts | See table below | Battery string voltage: 24 / 48 / 96 / 120 / 192 V |

### `rated_capacity_w` OID sources

| Vendor / MIB | OID | Raw unit | Conversion |
|---|---|---|---|
| RFC 1628 | `1.3.6.1.2.1.33.1.9.6` `upsConfigOutputPower` | Watts | none |
| APC PowerNet | `1.3.6.1.4.1.318.1.1.1.4.2.6` `upsAdvConfigRatedOutputVoltage` — **use kVA OID instead:** `1.3.6.1.4.1.318.1.1.12.2.2.1.1.4` `rPDULoadDevMaxPhaseLoad` | kVA | **× 1000** |
| Liebert GP | `1.3.6.1.4.1.476.1.42.3.5.7.6` `lgpPwrTopMaximumFrameCapacity` | VA | × 1.0 (treat VA ≈ W) |
| Liebert UPS | `1.3.6.1.4.1.476.1.1.1.1.1.9.4` `lcUpsNominalOutputWatts` | Watts | none |
| **Fallback** | None available | — | **Manual entry at onboarding** |

### `nominal_voltage_v` OID sources

| Vendor / MIB | OID | Notes |
|---|---|---|
| RFC 1628 | `1.3.6.1.2.1.33.1.9.1` `upsConfigInputVoltage` | Volts, direct |
| APC PowerNet | `1.3.6.1.4.1.318.1.1.1.3.2.7` `upsAdvInputNominalVoltage` | Volts, direct |
| Liebert UPS | `1.3.6.1.4.1.476.1.1.1.1.1.9.2` `lcUpsNominalInputVoltage` | Volts, direct |
| Liebert GP | Table walk required — see note below | — |
| **Fallback** | 120 or 230 based on country/region | Manual entry |

**Liebert GP note:** Walk `lgpPwrMeasurementPointTable` at onboarding to find the input
measurement point index, then read `lgpPwrLineMeasurementNominalVoltage` from that row.
This is a one-time discovery step, not a per-poll operation.

### `rated_battery_v` OID sources

| Vendor / MIB | OID | Notes |
|---|---|---|
| APC PowerNet | `1.3.6.1.4.1.318.1.1.1.2.2.7` `upsAdvBatteryNominalVoltage` | Volts, direct |
| Liebert GP | `1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.1` `lgpPwrDcMeasurementPointNomVolts` | Volts, direct |
| RFC 1628 | **No OID available** | Manual entry required |
| Liebert legacy UPS MIB | **No OID available** | Manual entry required |

**Fallback values by typical UPS capacity:**

| UPS rated capacity | Typical battery string voltage |
|---|---|
| ≤ 3 kW | 24 V |
| 5 kW | 48 V |
| 7.5 kW | 96 V |
| 10 kW (3-phase) | 120 V |
| 15 kW (3-phase) | 192 V |

If `rated_battery_v` cannot be determined, send `0.0`. The pipeline will set
`battery_voltage_ratio = 0.0` and `battery_current_ratio = 0.0` for that device
(graceful degradation — 15 of 17 features still work correctly).

---

## Derived binary flags

These are not raw SNMP counters — they require status interpretation per vendor.

### `on_battery_status`

Send `1.0` when the UPS is running on battery, `0.0` when on mains.

| Vendor / MIB | OID | Condition for `1.0` |
|---|---|---|
| RFC 1628 | `1.3.6.1.2.1.33.1.4.1` `upsOutputSource` | Value = `5` (battery) |
| APC PowerNet | `1.3.6.1.4.1.318.1.1.1.4.1.1` `upsBasicOutputStatus` | Value = `3` (onBattery) |
| Liebert GP | `1.3.6.1.4.1.476.1.42.3.5.3.7` `lgpPwrOutputToLoadOnInverter` | Value = `1` (true) |
| Liebert legacy UPS | Walk `lcUpsAlarmTable`; check descriptor string for `lcUpsAlarmOnBattery` | String match |

### `battery_replace_status`

Send `1.0` when the battery needs replacement, `0.0` otherwise.

| Vendor / MIB | OID | Condition for `1.0` |
|---|---|---|
| RFC 1628 | Walk `1.3.6.1.2.1.33.1.6.2` `upsAlarmTable` | `upsAlarmBatteryBad` OID present in table |
| APC PowerNet | `1.3.6.1.4.1.318.1.1.1.2.2.4` `upsAdvBatteryReplaceIndicator` | Value = `2` (batteryNeedsReplacing) |
| Liebert GP | `1.3.6.1.4.1.476.1.42.3.5.2.1` `lgpPwrBatteryCapacityStatus` | Value = `3` (low) or `4` (depleted) |
| Liebert legacy UPS | Walk `lcUpsAlarmTable` | `lcUpsAlarmBatteryBad` descriptor present |

---

## Non-UPS devices (PDU, network PSU, environmental sensor)

For PDU, network, and env devices, battery-related fields do not apply.
Send the following fixed values:

```json
"rated_battery_v": 0.0,
"feature_values": {
  "battery_charge_pct": 100.0,
  "battery_voltage_v": 0.0,
  "battery_current_a": 0.0,
  "battery_temperature_c": 20.0,
  "runtime_remaining_min": 0.0,
  "on_battery_status": 0.0,
  "battery_replace_status": 0.0
}
```

PDU-specific OIDs for load monitoring (Raritan / APC RPDU):
- `output_load_pct`: Raritan `1.3.6.1.4.1.13742.6.5.2.3.1.6` `measurementsUnitActivePower`
  (convert Watts → pct using `rated_capacity_w`)
- `output_current_a`: Raritan `1.3.6.1.4.1.13742.6.5.2.3.1.4` `measurementsUnitCurrent`
  (0.001 A units → **÷ 1000**)

---

## Quick reference — all conversions at a glance

| Feature | MIB | Raw unit | Multiply by |
|---|---|---|---|
| `battery_voltage_v` | RFC 1628 | 0.1 V | **0.1** |
| `battery_current_a` | RFC 1628 | 0.1 A | **0.1** |
| `input_frequency_hz` | RFC 1628 | 0.1 Hz | **0.1** |
| `output_frequency_hz` | RFC 1628 | 0.1 Hz | **0.1** |
| `output_current_a` | RFC 1628 | 0.1 A | **0.1** |
| `input_current_l1/l2/l3` | RFC 1628 | 0.1 A | **0.1** |
| `output_current_l1/l2/l3` | RFC 1628 | 0.1 A | **0.1** |
| `runtime_remaining_min` | APC | TimeTicks (1/100 s) | **0.000166667** (÷ 6000) |
| `rated_capacity_w` | APC | kVA | **1000** |
| `output_current_a` | Raritan PDU | 0.001 A | **0.001** |
| Everything else | Any | already in SI | **1.0** |

All other features (battery_charge_pct, output_load_pct, voltage values from APC/Liebert,
temperature, binary flags) arrive in SI units directly and require no conversion.

---

## Validation checklist before going live

Run these sanity checks on the first 100 messages from a new device before enabling
live scoring:

- [ ] `battery_voltage_v` is in range 10–300 V (not 100–3000 which signals missing ÷10)
- [ ] `input_frequency_hz` is 49–51 or 59–61 Hz (not 490–510 which signals missing ÷10)
- [ ] `runtime_remaining_min` is 0–999 (not > 100,000 which signals raw TimeTicks)
- [ ] `rated_capacity_w` is in range 100–50,000 W (not 1–50 which signals kVA not converted)
- [ ] `output_current_a` is < 1000 A (not 1,000–10,000 which signals missing ÷10)
- [ ] `rated_battery_v` is one of: 0, 24, 48, 96, 120, 192 (other values suggest a unit error)
- [ ] `on_battery_status` is 0.0 or 1.0 only (not the raw enum integer)
- [ ] `battery_replace_status` is 0.0 or 1.0 only

If any check fails, the collector has a unit conversion bug for that OID path.
Fix it at the collector — do not route the device to the anomaly detection pipeline
until the check passes.
