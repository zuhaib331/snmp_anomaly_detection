# Phase 1 — Database Foundations
> Complete this phase fully before touching any Python code.
> Reference document: SYSTEM_DESIGN_HANDOFF.md Sections 2, 12, 13

---

## Goal of this phase

Stand up PostgreSQL and ClickHouse with all tables, indexes, partitioning, and seed data
in place. When this phase is done, both databases are ready to receive real data.
No service code is written in this phase.

---

## Database names

| Engine | Database name |
|---|---|
| PostgreSQL (local) | `snmp_anomaly_detection_db` |
| Remote VM (192.168.25.47) | `snmp_engine` — read-only, `devices` table only |
| ClickHouse (local) | confirm before running migration 004 |

Connect to local PostgreSQL:
```bash
psql -d snmp_anomaly_detection_db -U <your_local_superuser>
```

---

## Prerequisites — confirm before starting

- [ ] Local PostgreSQL is running: `psql -d snmp_anomaly_detection_db -U <superuser> -c "SELECT 1;"`
- [ ] Local ClickHouse is running and reachable
- [ ] You have a superuser connection to `snmp_anomaly_detection_db` for DDL
- [ ] Remote VM (192.168.25.47) `snmp_engine` is reachable and you have SELECT on `devices`
- [ ] Check if pg_partman is available: `SELECT * FROM pg_extension WHERE extname = 'pg_partman';`
      Record the result — it changes how migration 002 is written

---

## Step 1 — Create migration 001: snmp_polling and feature_map tables

File: `config/migrations/001_create_tables.sql`

**What to create vs what already exists:**
```
devices         → ALREADY EXISTS in production. Do NOT touch. Verify access only.
snmp_polling    → does NOT exist. Create it here as a partitioned table.
feature_map     → does NOT exist. Create it here.
```

### 1a — devices table

**This table already exists in your production database. Do NOT run a CREATE TABLE for it.**

Instead, verify it is accessible and has the expected columns:

```sql
-- Run this to confirm your existing table has all required columns
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'devices'
ORDER BY column_name;
```

Expected columns to confirm present:
```
active          boolean
community       text
created_at      timestamp with time zone
device_id       text
dimensions      jsonb
ip              text
mib_config      jsonb
output_config   jsonb
polling_config  jsonb
port            integer
retries         integer
snmp_version    text
tags            jsonb
timeout_ms      integer
trap_config     jsonb
updated_at      timestamp with time zone
v3_config       jsonb
```

Also verify the JSONB structure is accessible:
```sql
-- Must return a data_topic value for at least one active device
SELECT
    device_id,
    output_config->>'data_topic'   AS data_topic,
    polling_config->>'operation'   AS operation,
    jsonb_array_length(polling_config->'oids') AS oid_count,
    tags->>'vendor'                AS vendor,
    dimensions->>'tenant'          AS tenant,
    active
FROM devices
WHERE active = true
  AND (polling_config->>'enabled')::boolean = true
LIMIT 3;
```

If any value returns null unexpectedly, check the JSONB key names match exactly.

### 1b — snmp_polling table (partitioned)

IMPORTANT: declare as partitioned table from the start.
Never create as a plain table and add partitioning later — it requires a full rebuild.

```sql
CREATE TABLE IF NOT EXISTS snmp_polling (
    id                  BIGSERIAL,
    poll_id             TEXT,
    device_id           TEXT,
    agentip             TEXT,
    ip                  TEXT,
    engine_node_id      TEXT,
    mib_module          TEXT,
    oid                 TEXT,
    oid_name            TEXT,
    operation           TEXT,
    operation_status    TEXT,
    requested_oids      TEXT[],
    result_count        INTEGER,
    access              TEXT,
    status              TEXT,
    syntax              TEXT,
    value               TEXT,
    value_type          TEXT,
    value_resolved      TEXT,
    display_value       TEXT,
    enum_label          TEXT,
    enum_value          INTEGER,
    rack                TEXT,
    site_id             INTEGER,
    tags                JSONB,
    tenant              TEXT,
    timestamp           BIGINT,
    timestamp_ms        BIGINT,
    type                TEXT DEFAULT 'polling',
    v                   INTEGER,
    weight              FLOAT,
    is_production       BOOLEAN,
    inserted_at         TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (id, inserted_at)   -- partition key must be in PK
) PARTITION BY RANGE (inserted_at);
```

Create indexes on the parent table (they propagate to partitions automatically):

```sql
CREATE INDEX IF NOT EXISTS idx_snmp_polling_poll_id
    ON snmp_polling (poll_id);

CREATE INDEX IF NOT EXISTS idx_snmp_polling_device_ts
    ON snmp_polling (device_id, timestamp_ms);

CREATE INDEX IF NOT EXISTS idx_snmp_polling_inserted
    ON snmp_polling (inserted_at);

CREATE INDEX IF NOT EXISTS idx_snmp_polling_inserted_device
    ON snmp_polling (inserted_at, device_id);

CREATE INDEX IF NOT EXISTS idx_snmp_polling_oid
    ON snmp_polling (oid);
```

### 1c — feature_map table

```sql
CREATE TABLE IF NOT EXISTS feature_map (
    id                  SERIAL PRIMARY KEY,
    vendor              TEXT NOT NULL,
    category            TEXT NOT NULL,
    device_type         TEXT NOT NULL,
    kafka_topic         TEXT NOT NULL,
    mib_module          TEXT,
    oid                 TEXT NOT NULL,
    oid_name_pattern    TEXT,
    feature_name        TEXT NOT NULL,
    value_type          TEXT NOT NULL,
    is_enum             BOOLEAN DEFAULT false,
    required            BOOLEAN DEFAULT false,
    is_metadata         BOOLEAN DEFAULT false,
    scaling_factor      FLOAT DEFAULT 1.0,
    unit                TEXT,
    notes               TEXT,
    is_active           BOOLEAN DEFAULT true,
    updated_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (vendor, category, oid)
);

CREATE INDEX IF NOT EXISTS idx_feature_map_vendor_category
    ON feature_map (vendor, category);

CREATE INDEX IF NOT EXISTS idx_feature_map_updated
    ON feature_map (updated_at);

CREATE TRIGGER feature_map_updated_at
    BEFORE UPDATE ON feature_map
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
```

---

## Step 2 — Create migration 002: partition helper functions + initial partitions

File: `config/migrations/002_partition_helpers.sql`

**Migration numbering used across ALL documents:**
```
001_create_tables.sql          PostgreSQL tables + indexes + triggers
002_partition_helpers.sql      Partition helper functions + initial partitions
003_seed_feature_map.sql       Vertiv UPS feature_map seed rows
004_clickhouse_tables.sql      ClickHouse snmp_features, scores, quarantine
```
This matches SYSTEM_DESIGN_HANDOFF.md Section 14 build order. Do not renumber.

Create partitions for today, yesterday, and the next 7 days so the system
starts with headroom. Adjust dates to match your go-live date.

```sql
-- Pattern: one partition per day
-- FOR VALUES FROM ('YYYY-MM-DD') TO ('YYYY-MM-DD+1')
-- Create enough partitions to cover today + 7 days forward

CREATE TABLE IF NOT EXISTS snmp_polling_y2025m01d01
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2025-01-01 00:00:00+00') TO ('2025-01-02 00:00:00+00');

-- Repeat for each day you need
-- Claude Code: generate these dynamically for the week of go-live
```

Helper function to create a partition for any given date:

```sql
CREATE OR REPLACE FUNCTION create_snmp_partition(target_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    start_date TIMESTAMPTZ;
    end_date TIMESTAMPTZ;
BEGIN
    partition_name := 'snmp_polling_y' ||
                      to_char(target_date, 'YYYY') || 'm' ||
                      to_char(target_date, 'MM') || 'd' ||
                      to_char(target_date, 'DD');
    start_date := target_date::TIMESTAMPTZ AT TIME ZONE 'UTC';
    end_date   := (target_date + 1)::TIMESTAMPTZ AT TIME ZONE 'UTC';

    IF NOT EXISTS (
        SELECT FROM pg_tables WHERE tablename = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF snmp_polling
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        RAISE NOTICE 'Created partition: %', partition_name;
    ELSE
        RAISE NOTICE 'Partition already exists: %', partition_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Helper to drop a partition by date (used by DAG 1 retention task)
CREATE OR REPLACE FUNCTION drop_snmp_partition(target_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
BEGIN
    partition_name := 'snmp_polling_y' ||
                      to_char(target_date, 'YYYY') || 'm' ||
                      to_char(target_date, 'MM') || 'd' ||
                      to_char(target_date, 'DD');
    EXECUTE format('DROP TABLE IF EXISTS %I', partition_name);
    RAISE NOTICE 'Dropped partition: %', partition_name;
END;
$$ LANGUAGE plpgsql;
```

If pg_partman is available, use it instead of the above functions:

```sql
-- Only run this if pg_partman is available
SELECT partman.create_parent(
    p_parent_table  => 'public.snmp_polling',
    p_control       => 'inserted_at',
    p_type          => 'range',
    p_interval      => '1 day',
    p_premake       => 7,
    p_retention     => '7 days',
    p_retention_keep_table => false
);
```

---

## Step 3 — Create migration 003: feature_map seed data

File: `config/migrations/003_seed_feature_map.sql`
Note: this is file 003 — not 002. See numbering table in Step 2.

CRITICAL RULES for this file:
- All OIDs stored WITHOUT leading dot
- feature_name values MUST match BASELINE_UPS_FEATURES in detect_power_kafka.py exactly
- Verify the feature names against the scoring service before inserting
- scaling_factor=0.01 for power factor OIDs (Liebert stores as 0-100 integer)

```sql
-- Vertiv UPS — LIEBERT-GP-POWER-MIB
-- vendor=vertiv, category=power, device_type=ups, kafka_topic=snmp-events-power

INSERT INTO feature_map
    (vendor, category, device_type, kafka_topic, mib_module, oid,
     oid_name_pattern, feature_name, value_type, is_enum, required,
     is_metadata, scaling_factor, unit, notes)
VALUES

-- ── ML Features (go into feature_values) ──────────────────────────

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.1.1',
 'lgpPwrLineMeasurementCurrent.1.1',
 'input_current_l1_a','Integer',false,true,false,1.0,'amps',
 'Input line current L1 in amps'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.3.1',
 'lgpPwrLineMeasurementCurrent.3.1',
 'output_current_l1_a','Integer',false,true,false,1.0,'amps',
 'Output line current L1 in amps'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.8.3.1',
 'lgpPwrLineMeasurementVA.3.1',
 'output_apparent_power_va','Integer',false,true,false,1.0,'va',
 'Output apparent power in VA'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.9.3.1',
 'lgpPwrLineMeasurementTruePower.3.1',
 'output_true_power_w','Integer',false,true,false,1.0,'watts',
 'Output true power in watts'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.1.1',
 'lgpPwrLineMeasurementPowerFactor.1.1',
 'input_power_factor','Integer',false,true,false,0.01,'ratio',
 'Input power factor — raw value 95 = 0.95 after scaling'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.3.1',
 'lgpPwrLineMeasurementPowerFactor.3.1',
 'output_power_factor','Integer',false,true,false,0.01,'ratio',
 'Output power factor — raw value 74 = 0.74 after scaling'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.16.1.1',
 'lgpPwrLineMeasurementMaxVolts.1.1',
 'input_voltage_max_v','Integer',false,false,false,1.0,'volts',
 'Maximum input voltage recorded'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.17.1.1',
 'lgpPwrLineMeasurementMinVolts.1.1',
 'input_voltage_min_v','Integer',false,false,false,1.0,'volts',
 'Minimum input voltage recorded'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.19.3.1',
 'lgpPwrLineMeasurementPercentLoad.3.1',
 'output_load_pct','Integer',false,true,false,1.0,'percent',
 'Output load as percentage of rated capacity'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.1.1',
 'lgpPwrLineMeasurementVolts.1.1',
 'input_voltage_l1_v','Integer',false,true,false,1.0,'volts',
 'Input line voltage L1'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.2.1',
 'lgpPwrLineMeasurementVolts.2.1',
 'input_voltage_l2_v','Integer',false,false,false,1.0,'volts',
 'Input line voltage L2 — bypass line'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.3.1',
 'lgpPwrLineMeasurementVolts.3.1',
 'output_voltage_l1_v','Integer',false,true,false,1.0,'volts',
 'Output line voltage L1'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.4.1',
 'lgpPwrDcMeasurementPointVolts.1',
 'battery_voltage_v','Integer',false,true,false,1.0,'volts',
 'Battery DC voltage'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.5.1',
 'lgpPwrDcMeasurementPointCurrent.1',
 'battery_current_a','Integer',false,true,false,1.0,'amps',
 'Battery DC current — 0 is valid when on mains'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.6.1',
 'lgpPwrDcMeasurementPointNomVolts.1',
 'battery_nominal_voltage_v','Integer',false,false,false,1.0,'volts',
 'Battery nominal voltage — semi-static device registration field'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.3.7.0',
 'lgpPwrOutputToLoadOnInverter.0',
 'on_inverter_flag','Integer',true,true,false,1.0,'bool',
 'Enum: 1=yes on inverter, 2=no. Use enum_value as feature.'),

-- ── Metadata (go into metadata{}, NOT feature_values) ─────────────

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.1',
 'lgpPwrMeasurementPoint.1.1',
 'meas_point_1_1_ref','OID',false,false,true,1.0,NULL,
 'Structural reference OID pointing to measurement source'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.2',
 'lgpPwrMeasurementPoint.1.2',
 'meas_point_1_2_ref','OID',false,false,true,1.0,NULL,
 'Structural reference OID'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.2.1',
 'lgpPwrMeasurementPoint.2.1',
 'meas_point_bypass_ref','OID',false,false,true,1.0,NULL,
 'Structural reference OID pointing to bypass measurement'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.2.1',
 'lgpPwrDcMeasurementPointId.1',
 'dc_point_battery_ref','OID',false,false,true,1.0,NULL,
 'Points to lgpPwrMeasBattery — structural'),

('vertiv','power','ups','snmp-events-power','LIEBERT-GP-POWER-MIB',
 '1.3.6.1.4.1.476.1.42.3.5.2.4.1.3.1',
 'lgpPwrDcMeasurementPointSubID.1',
 'dc_point_sub_id','Integer',false,false,true,1.0,NULL,
 'Sub-index integer — structural, not a measurement')

ON CONFLICT (vendor, category, oid) DO UPDATE SET
    feature_name   = EXCLUDED.feature_name,
    scaling_factor = EXCLUDED.scaling_factor,
    is_metadata    = EXCLUDED.is_metadata,
    is_enum        = EXCLUDED.is_enum,
    required       = EXCLUDED.required,
    updated_at     = now();
```

---

## Step 4 — Create ClickHouse tables

File: `config/migrations/004_clickhouse_tables.sql`
Note: this is file 004. Run against ClickHouse (not PostgreSQL).

```sql
-- snmp_features: main ML training table
CREATE TABLE IF NOT EXISTS snmp_features (
    device_id           String,
    device_type         LowCardinality(String),
    category            LowCardinality(String),
    vendor              LowCardinality(String),
    tenant              LowCardinality(String),
    site_id             UInt32,
    rack                String,
    is_production       UInt8,
    timestamp           DateTime,
    timestamp_ms        UInt64,
    poll_id             String,
    feature_values      String,
    missing_features    Array(String),
    pipeline_version    String,
    batch_id            String,
    source              LowCardinality(String),
    inserted_at         DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY (category, toYYYYMM(timestamp))
ORDER BY (device_id, timestamp)
TTL timestamp + INTERVAL 2 YEAR;

-- snmp_anomaly_scores: written by live scoring services
CREATE TABLE IF NOT EXISTS snmp_anomaly_scores (
    device_id               String,
    device_type             LowCardinality(String),
    category                LowCardinality(String),
    scored_at               DateTime,
    timestamp_ms            UInt64,
    poll_id                 String,
    lstm_recon_error        Nullable(Float64),
    lstm_threshold          Nullable(Float64),
    iso_forest_score        Float64,
    iso_forest_threshold    Float64,
    is_anomaly              UInt8,
    alert_state             LowCardinality(String),
    anomaly_reason          String,
    top_features            Array(String),
    compound_alert          UInt8,
    compound_peer           String,
    model_version           String,
    calibration_status      LowCardinality(String)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(scored_at)
ORDER BY (device_id, scored_at);

-- snmp_quarantine: bad rows from ETL DQ gate
CREATE TABLE IF NOT EXISTS snmp_quarantine (
    device_id       String,
    raw_json        String,
    failure_reason  String,
    batch_id        String,
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (device_id, created_at)
TTL created_at + INTERVAL 90 DAY;
```

---

## Step 5 — Verify existing device data is readable

Do not insert test data. Your devices table already has real devices.
Run these queries to confirm the pipeline can read what it needs.

```sql
-- Check active devices with complete config
SELECT
    device_id,
    output_config->>'data_topic'           AS kafka_topic,
    polling_config->>'operation'           AS operation,
    jsonb_array_length(polling_config->'oids') AS oid_count,
    tags->>'vendor'                        AS vendor,
    dimensions->>'tenant'                  AS tenant,
    dimensions->>'site_id'                 AS site_id,
    active
FROM devices
WHERE active = true
  AND (polling_config->>'enabled')::boolean = true
ORDER BY device_id
LIMIT 10;
```

Expected: at least one row returned with non-null kafka_topic, vendor, and tenant.

```sql
-- Confirm updated_at is present and recent (config watcher depends on this)
SELECT device_id, updated_at
FROM devices
WHERE active = true
ORDER BY updated_at DESC
LIMIT 5;
```

```sql
-- Confirm no device has a null data_topic (would cause forwarder to skip it)
SELECT COUNT(*) AS devices_missing_topic
FROM devices
WHERE active = true
  AND (output_config->>'data_topic') IS NULL;
-- Should return 0
```

---

## Step 6 — Verification checklist

Run devices checks against the **remote VM** (`psql -h 192.168.25.47 -d snmp_engine`).
Run all other checks against **local** (`psql -d snmp_anomaly_detection_db`).

```sql
-- [remote: snmp_engine] Confirm devices table is readable
SELECT COUNT(*) FROM devices WHERE active = true;   -- must be >= 1

-- Confirm JSONB fields are accessible
SELECT COUNT(*) FROM devices
WHERE active = true
  AND (output_config->>'data_topic') IS NOT NULL;         -- must equal active device count

SELECT COUNT(*) FROM devices
WHERE active = true
  AND (polling_config->>'enabled')::boolean = true;       -- must be >= 1

-- [local: snmp_anomaly_detection_db] snmp_polling checks
SELECT COUNT(*) FROM snmp_polling LIMIT 1;                -- must succeed (table exists)

-- [local: snmp_anomaly_detection_db] feature_map checks
SELECT COUNT(*) FROM feature_map;                         -- must be >= 21
SELECT COUNT(*) FROM feature_map WHERE is_metadata = false; -- must be 16
SELECT COUNT(*) FROM feature_map WHERE is_metadata = true;  -- must be 5
SELECT COUNT(*) FROM feature_map WHERE scaling_factor = 0.01; -- must be 2

-- Check OIDs have no leading dots
SELECT COUNT(*) FROM feature_map WHERE oid LIKE '.%';     -- must be 0

-- Check partition exists for today
SELECT tablename FROM pg_tables
WHERE tablename LIKE 'snmp_polling_%'
ORDER BY tablename;

-- [local: ClickHouse] ClickHouse checks
SELECT count() FROM snmp_features;
SELECT count() FROM snmp_anomaly_scores;
SELECT count() FROM snmp_quarantine;
```

---

## Deliverables — phase 1 complete when

- [ ] Existing `devices` table verified readable — active devices return non-null kafka_topic
- [ ] `config/migrations/001_create_tables.sql` written — creates snmp_polling + feature_map only
- [ ] `config/migrations/002_partition_helpers.sql` written and applied
- [ ] `config/migrations/003_seed_feature_map.sql` written and applied
- [ ] `config/migrations/004_clickhouse_tables.sql` written and applied
- [ ] All verification queries pass
- [ ] feature_map has exactly 21 rows (16 features + 5 metadata)
- [ ] Zero OIDs in feature_map have a leading dot
- [ ] snmp_polling partition exists for today

**Do not proceed to Phase 2 until every item above is checked.**
