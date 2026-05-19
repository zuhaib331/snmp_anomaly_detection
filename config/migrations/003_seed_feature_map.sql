-- Migration 003: feature_map seed data — Vertiv UPS (LIEBERT-GP-POWER-MIB)
-- Target: local PostgreSQL, database snmp_anomaly_detection_db (running on your machine)
-- Run after 001_create_tables.sql.
--
-- Connect before running:
--   psql -d snmp_anomaly_detection_db -U <your_local_superuser>
-- UPSERT — safe to re-run. ON CONFLICT updates feature_name, scaling_factor,
-- is_metadata, is_enum, required, and updated_at.
--
-- Rules enforced here:
--   - All OIDs stored WITHOUT leading dot (strip if present before inserting)
--   - feature_name values match BASELINE_UPS_FEATURES in config.py / scoring service
--   - scaling_factor=0.01 for power factor OIDs (Liebert stores as integer 0-100)
--   - is_enum=true rows: use enum_value (not value) as the numeric ML feature
--   - is_metadata=true rows: go into envelope metadata{}, NOT into feature_values{}
--
-- Expected result after this migration:
--   SELECT COUNT(*) FROM feature_map;                         -- 21
--   SELECT COUNT(*) FROM feature_map WHERE is_metadata=false; -- 16
--   SELECT COUNT(*) FROM feature_map WHERE is_metadata=true;  -- 5
--   SELECT COUNT(*) FROM feature_map WHERE scaling_factor=0.01; -- 2
--   SELECT COUNT(*) FROM feature_map WHERE oid LIKE '.%';     -- 0 (no leading dots)

INSERT INTO feature_map
    (vendor, category, device_type, kafka_topic, mib_module, oid,
     oid_name_pattern, feature_name, value_type, is_enum, required,
     is_metadata, scaling_factor, unit, notes)
VALUES

-- ── ML Features (go into feature_values{}) ───────────────────────────────────

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.1.1',
 'lgpPwrLineMeasurementCurrent.1.1',
 'input_current_l1_a', 'Integer', false, true, false, 1.0, 'amps',
 'Input line current L1 in amps'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.3.1',
 'lgpPwrLineMeasurementCurrent.3.1',
 'output_current_l1_a', 'Integer', false, true, false, 1.0, 'amps',
 'Output line current L1 in amps'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.8.3.1',
 'lgpPwrLineMeasurementVA.3.1',
 'output_apparent_power_va', 'Integer', false, true, false, 1.0, 'va',
 'Output apparent power in VA'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.9.3.1',
 'lgpPwrLineMeasurementTruePower.3.1',
 'output_true_power_w', 'Integer', false, true, false, 1.0, 'watts',
 'Output true power in watts'),

-- scaling_factor=0.01: Liebert stores power factor as integer 0-100.
-- Value 95 → 0.95, value 74 → 0.74 after scaling.
('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.1.1',
 'lgpPwrLineMeasurementPowerFactor.1.1',
 'input_power_factor', 'Integer', false, true, false, 0.01, 'ratio',
 'Input power factor — raw value 95 = 0.95 after scaling'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.3.1',
 'lgpPwrLineMeasurementPowerFactor.3.1',
 'output_power_factor', 'Integer', false, true, false, 0.01, 'ratio',
 'Output power factor — raw value 74 = 0.74 after scaling'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.16.1.1',
 'lgpPwrLineMeasurementMaxVolts.1.1',
 'input_voltage_max_v', 'Integer', false, false, false, 1.0, 'volts',
 'Maximum input voltage recorded'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.17.1.1',
 'lgpPwrLineMeasurementMinVolts.1.1',
 'input_voltage_min_v', 'Integer', false, false, false, 1.0, 'volts',
 'Minimum input voltage recorded'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.19.3.1',
 'lgpPwrLineMeasurementPercentLoad.3.1',
 'output_load_pct', 'Integer', false, true, false, 1.0, 'percent',
 'Output load as percentage of rated capacity'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.1.1',
 'lgpPwrLineMeasurementVolts.1.1',
 'input_voltage_l1_v', 'Integer', false, true, false, 1.0, 'volts',
 'Input line voltage L1'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.2.1',
 'lgpPwrLineMeasurementVolts.2.1',
 'input_voltage_l2_v', 'Integer', false, false, false, 1.0, 'volts',
 'Input line voltage L2 — bypass line'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.3.1',
 'lgpPwrLineMeasurementVolts.3.1',
 'output_voltage_l1_v', 'Integer', false, true, false, 1.0, 'volts',
 'Output line voltage L1'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.4.1',
 'lgpPwrDcMeasurementPointVolts.1',
 'battery_voltage_v', 'Integer', false, true, false, 1.0, 'volts',
 'Battery DC voltage'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.5.1',
 'lgpPwrDcMeasurementPointCurrent.1',
 'battery_current_a', 'Integer', false, true, false, 1.0, 'amps',
 'Battery DC current — 0 is valid when on mains'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.6.1',
 'lgpPwrDcMeasurementPointNomVolts.1',
 'battery_nominal_voltage_v', 'Integer', false, false, false, 1.0, 'volts',
 'Battery nominal voltage — semi-static device registration field'),

-- is_enum=true: use enum_value (integer) as the ML feature, not raw value string.
-- enum_value=1 means "on inverter", enum_value=2 means "no".
('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.3.7.0',
 'lgpPwrOutputToLoadOnInverter.0',
 'on_inverter_flag', 'Integer', true, true, false, 1.0, 'bool',
 'Enum: 1=yes on inverter, 2=no. Use enum_value as feature.'),

-- ── Metadata (go into metadata{} envelope, NOT into feature_values{}) ────────
-- All OID-type rows are structural/pointer OIDs — never cast as floats.
-- Store value_resolved (e.g. "lgpPwrSource1Input") as metadata string.

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.1',
 'lgpPwrMeasurementPoint.1.1',
 'meas_point_1_1_ref', 'OID', false, false, true, 1.0, NULL,
 'Structural reference OID pointing to measurement source'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.2',
 'lgpPwrMeasurementPoint.1.2',
 'meas_point_1_2_ref', 'OID', false, false, true, 1.0, NULL,
 'Structural reference OID'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.2.1',
 'lgpPwrMeasurementPoint.2.1',
 'meas_point_bypass_ref', 'OID', false, false, true, 1.0, NULL,
 'Structural reference OID pointing to bypass measurement'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.2.1',
 'lgpPwrDcMeasurementPointId.1',
 'dc_point_battery_ref', 'OID', false, false, true, 1.0, NULL,
 'Points to lgpPwrMeasBattery — structural'),

('vertiv', 'power', 'ups', 'snmp-events-power', 'LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.3.1',
 'lgpPwrDcMeasurementPointSubID.1',
 'dc_point_sub_id', 'Integer', false, false, true, 1.0, NULL,
 'Sub-index integer — structural, not a measurement')

ON CONFLICT (vendor, category, oid) DO UPDATE SET
    feature_name   = EXCLUDED.feature_name,
    scaling_factor = EXCLUDED.scaling_factor,
    is_metadata    = EXCLUDED.is_metadata,
    is_enum        = EXCLUDED.is_enum,
    required       = EXCLUDED.required,
    updated_at     = now();
