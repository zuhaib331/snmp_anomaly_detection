# SNMP Anomaly Detection System — Complete Design Handoff
> This document is the single source of truth for Claude Code.
> Read it fully before writing any code. Every decision here was made deliberately.

---

## 1. What we are building

An enterprise SNMP anomaly detection system that:
- Collects SNMP polling data from thousands of devices (UPS, PDU, routers, switches, sensors)
- Detects anomalies in real time using Isolation Forest (per event) and LSTM Autoencoder (per sequence window)
- Stores clean feature data in ClickHouse for ML model training and reporting
- Routes events to the correct scoring service based on device category
- Retrains models automatically when new data accumulates

---

## 2. PostgreSQL — three tables, single source of truth

Three tables matter. One (`devices`) already exists in your production database — read from it only. Two (`snmp_polling` and `feature_map`) must be created as part of Phase 1 migrations.

### Table 1 — devices

This table already exists in your production PostgreSQL database. Do NOT recreate it.
Read from it. Never write to it from ETL or live forwarder.

Confirmed real column names and types:
```
device_id       TEXT                   e.g. "idrac3-6-15-6"
ip              TEXT                   e.g. "192.168.3.6"
snmp_version    TEXT                   "v2c" or "v3"
community       TEXT                   SNMP community string
v3_config       JSONB                  null if v2c
tags            JSONB                  {"site": "...", "vendor": "dell-idrac"}
dimensions      JSONB                  {"site_id": 102, "rack": "R15", "tenant": "core-net-2",
                                        "is_production": true, "weight": 1.05, "agentip": "..."}
timeout_ms      INTEGER
retries         INTEGER
port            INTEGER                default 161
polling_config  JSONB                  see structure below
output_config   JSONB                  see structure below
trap_config     JSONB                  not used by ETL or forwarder
mib_config      JSONB                  not used by ETL or forwarder
active          BOOLEAN                true = process this device
created_at      TIMESTAMPTZ
updated_at      TIMESTAMPTZ            watched by config watcher for change detection
```

**polling_config JSONB structure (confirmed from real data):**
```json
{
  "_v": 1,
  "oids": [
    ".1.3.6.1.4.1.674.10892.5.4.700.20.1.6.1.1",
    ".1.3.6.1.4.1.674.10892.5.4.700.20.1.6.1.2"
  ],
  "port": 161,
  "enabled": true,
  "interval": 40,
  "operation": "GET",
  "non_repeaters": 0,
  "max_repetitions": 25
}
```

Key fields used by the pipeline:
- `polling_config->>'enabled'`  — skip device if false
- `polling_config->'oids'`      — array of OIDs this device polls
- `polling_config->>'operation'` — "GET", "GETBULK", or "WALK"
- `polling_config->>'interval'`  — seconds between polls

**IMPORTANT — polling_config.oids contains specific OIDs, not subtrees.**
Unlike the design assumption, real devices use specific OIDs directly (e.g.
`.1.3.6.1.4.1.674.10892.5.4.700.20.1.6.1.1`) not base subtrees for walks.
For GET operations, each poll produces exactly `len(oids)` rows in snmp_polling.
The assembler uses `result_count` from snmp_polling rows to know when a poll is complete.

**output_config JSONB structure (confirmed from real data):**
```json
{
  "_v": 1,
  "data_topic": "snmp.output.idrac3-6-15-6",
  "error_topic": "snmp.errors.idrac3-6-15-6"
}
```

Key fields used by the pipeline:
- `output_config->>'data_topic'` — Kafka topic the live forwarder subscribes to

**How to read these fields in SQL (config watcher queries):**
```sql
-- Get all active devices with their Kafka topics
SELECT
    device_id,
    ip,
    snmp_version,
    community,
    tags->>'vendor'                        AS vendor,
    dimensions->>'tenant'                  AS tenant,
    dimensions->>'site_id'                 AS site_id,
    dimensions->>'rack'                    AS rack,
    dimensions->>'is_production'           AS is_production,
    polling_config->'oids'                 AS polling_oids,
    polling_config->>'operation'           AS polling_operation,
    polling_config->>'interval'            AS polling_interval,
    output_config->>'data_topic'           AS output_data_topic,
    output_config->>'error_topic'          AS output_error_topic,
    updated_at
FROM devices
WHERE active = true
  AND (polling_config->>'enabled')::boolean = true;
```

Change detection: watch `updated_at`. Both the live forwarder and ETL service refresh
their in-memory config when `updated_at` changes on any row.

Rule: if a device_id is not in devices, ignore all its polling data. No silent collection.

### Table 2 — snmp_polling

Raw polling results. One row per OID per poll. Written by the SNMP poller engine.

Key fields based on confirmed sample data:
```
id                  BIGSERIAL PRIMARY KEY
poll_id             TEXT                   UUID shared by ALL rows from one GETBULK response
                                           e.g. "343412a5-196f-48a5-98d9-40e5eeacf478"
                                           THIS IS THE PRIMARY GROUPING KEY — see note below
device_id           TEXT                   matches devices.device_id
agentip             TEXT
ip                  TEXT
engine_node_id      TEXT
mib_module          TEXT                   e.g. "LIEBERT-GP-POWER-MIB"
oid                 TEXT                   full child OID as written by the SNMP poller
                                           may include a leading dot e.g. ".1.3.6.1.4.1.476..."
                                           strip leading dot before feature_map lookup
oid_name            TEXT                   MIB-resolved name e.g. "lgpPwrLineMeasurementVA.3.1"
operation           TEXT                   "GET", "GETBULK", "WALK"
operation_status    TEXT                   "ok", "error", "timeout"
requested_oids      TEXT[]                 base OIDs that triggered this poll
result_count        INTEGER                total OID rows in this poll response
                                           use this to know when a poll is fully assembled
access              TEXT                   "read-only", "read-write"
status              TEXT                   "current", "deprecated"
syntax              TEXT                   MIB syntax e.g. "Integer32", "OBJECT IDENTIFIER"
value               TEXT                   raw string value
value_type          TEXT                   "Integer", "Gauge32", "Counter32", "OctetString", "OID"
value_resolved      TEXT                   optional MIB-resolved OID label e.g. "lgpPwrSource1Input"
display_value       TEXT                   human-readable display (present on enum rows)
enum_label          TEXT                   e.g. "yes", "no", "normal", "fault"
                                           present only on enumerated INTEGER OIDs
enum_value          INTEGER                numeric value of the enum (same as value when enum)
rack                TEXT
site_id             INTEGER
tags                JSONB
tenant              TEXT
timestamp           BIGINT                 unix seconds
timestamp_ms        BIGINT                 unix milliseconds
type                TEXT                   "polling"
v                   INTEGER                schema version
weight              FLOAT
is_production       BOOLEAN
inserted_at         TIMESTAMPTZ DEFAULT now()
```

**PRIMARY GROUPING KEY: `poll_id`**
All OID rows from one GETBULK/WALK response share the same `poll_id`.
`result_count` tells the assembler exactly how many rows to expect for this poll.
The assembler is complete when it has received `result_count` rows for a given `poll_id`.

Do NOT use `(device_id, timestamp_ms)` as the grouping key.
`timestamp_ms` is identical across all rows of the same poll in GETBULK
but could theoretically differ in multi-packet responses. `poll_id` is authoritative.

Fallback for missing poll_id: if poll_id is null (older data or GET operations),
fall back to grouping by `(device_id, timestamp_ms)` with a 2-second window.

**OID value_type categories — handling rules:**

1. Numeric types (Integer, Integer32, Gauge32, Counter32, TimeTicks):
   → cast to float, apply scaling_factor from feature_map
   → check against sentinel values

2. OctetString:
   → carry as metadata string if is_metadata=true in feature_map
   → attempt float cast if feature_map value_type is numeric (some vendors encode numbers as strings)

3. OID type (value_type="OID"):
   → these are pointer/reference OIDs — structural metadata, NOT ML features
   → use value_resolved (e.g. "lgpPwrSource1Input") as metadata string
   → NEVER attempt to cast OID-type values as floats

4. Enum/INTEGER with enum_label present:
   → use enum_value (integer) as the feature value for ML
   → store enum_label in metadata for human readability
   → example: lgpPwrOutputToLoadOnInverter — enum_value=1 means "on inverter"

Sentinel values to treat as NULL (not real readings):
- 4294967295  (0xFFFFFFFF — Gauge32/Counter32 max, means "unavailable")
- 2147483647  (0x7FFFFFFF — Integer max)
- 65535, 255  (smaller type maxes used as sentinels by some vendors)
- 0 on voltage/current OIDs where 0 is physically impossible — treat as suspect,
  flag in missing_features but do not hard-drop (0 may be valid on battery current)

Indexes required:
```sql
CREATE INDEX idx_snmp_polling_poll_id          ON snmp_polling (poll_id);
CREATE INDEX idx_snmp_polling_device_ts        ON snmp_polling (device_id, timestamp_ms);
CREATE INDEX idx_snmp_polling_inserted         ON snmp_polling (inserted_at);
CREATE INDEX idx_snmp_polling_inserted_device  ON snmp_polling (inserted_at, device_id);
CREATE INDEX idx_snmp_polling_oid              ON snmp_polling (oid);
```

Note on `idx_snmp_polling_inserted_device`: this composite index is critical for the ETL
batch query which filters on both `inserted_at > watermark` AND `device_id IN (...)`.
The single `inserted_at` index alone is not selective enough as the table grows to billions
of rows. Do not remove this composite index.

**Retention strategy — daily partitioning:**

`snmp_polling` grows at roughly 86 million rows/day at 1000 devices × 25 OIDs × 144 polls/hour.
Without retention the table becomes unmanageable within weeks. Use PostgreSQL declarative
range partitioning by `inserted_at` day.

```sql
-- migration 001: declare as partitioned table
CREATE TABLE snmp_polling (
    -- all columns as documented above --
) PARTITION BY RANGE (inserted_at);

-- daily partition — one per day, created in advance
CREATE TABLE snmp_polling_y2025m01d15
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2025-01-15 00:00:00+00') TO ('2025-01-16 00:00:00+00');
```

Retention window: 7 days. Enough for ETL watermark to recover from any outage and for the
live forwarder to reprocess after a crash. Beyond 7 days the data is already in ClickHouse
snmp_features and no longer needed in PostgreSQL.

Partition management — add two tasks at the TOP of DAG 1 (before extract):
1. `create_tomorrows_partition` — creates next day's partition if it does not exist
2. `drop_old_partitions` — drops any partition with data older than 7 days

```sql
-- drop example (DAG task runs this dynamically for partitions > 7 days old)
DROP TABLE IF EXISTS snmp_polling_y2025m01d07;
```

If `pg_partman` extension is available on your PostgreSQL instance, use it — it handles
partition creation and retention automatically without the Airflow tasks.
Check availability: `SELECT * FROM pg_extension WHERE extname = 'pg_partman';`

### Table 3 — feature_map

Maps specific child OIDs to ML feature names. Scoped per vendor + category.
This is Config B — the analytical config that tells the pipeline which OIDs are features.

```
id                  SERIAL PRIMARY KEY
vendor              TEXT                   e.g. "vertiv", "apc", "cisco"
category            TEXT                   e.g. "power", "network", "environment"
device_type         TEXT                   e.g. "ups", "pdu", "router", "switch"
kafka_topic         TEXT                   e.g. "snmp-events-power"
mib_module          TEXT                   e.g. "LIEBERT-GP-POWER-MIB" — for documentation
oid                 TEXT                   full child OID — ALWAYS stored WITHOUT leading dot
                                           e.g. "1.3.6.1.4.1.476.1.42.3.5.2.3.1.8.3.1"
                                           Normalise on insert: strip leading dot if present.
                                           The assembler strips leading dots from snmp_polling OIDs
                                           before lookup. Both sides must match — no leading dots anywhere.
oid_name_pattern    TEXT                   MIB name pattern for reference e.g. "lgpPwrLineMeasurementVolts"
feature_name        TEXT                   canonical ML feature name (vendor-agnostic)
value_type          TEXT                   expected type for casting: "Integer", "Gauge32", "OctetString", "enum"
is_enum             BOOLEAN DEFAULT false  if true, use enum_value field as numeric feature
required            BOOLEAN                if true, missing = flag in missing_features[]
is_metadata         BOOLEAN DEFAULT false  if true, carry in envelope metadata but NOT in feature_values
scaling_factor      FLOAT DEFAULT 1.0      multiply raw value by this (e.g. 0.01 for percent stored as integer)
unit                TEXT                   e.g. "volts", "amps", "watts", "percent", "va"
notes               TEXT                   human notes on what this OID means
is_active           BOOLEAN DEFAULT true
updated_at          TIMESTAMPTZ DEFAULT now()
```

Vendor-agnostic feature names: the same concept uses the same feature_name across vendors.
Example: output load percentage is `output_load_pct` for both Vertiv and APC —
even though their OIDs are completely different.

This is what makes the scoring service vendor-agnostic. It always sees the same feature names.

**Confirmed seed rows for Vertiv UPS (from real polling data — LIEBERT-GP-POWER-MIB):**

ML features (go into feature_values{}):
```
vendor=vertiv, category=power, device_type=ups, kafka_topic=snmp-events-power

OID                                              feature_name                  type     required  scaling  unit
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.1.1          input_current_l1_a            Integer  true      1.0      amps
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.6.3.1          output_current_l1_a           Integer  true      1.0      amps
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.8.3.1          output_apparent_power_va      Integer  true      1.0      va
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.9.3.1          output_true_power_w           Integer  true      1.0      watts
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.1.1         input_power_factor            Integer  true      0.01     ratio    (value 95 → 0.95)
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.14.3.1         output_power_factor           Integer  true      0.01     ratio    (value 74 → 0.74)
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.16.1.1         input_voltage_max_v           Integer  false     1.0      volts
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.17.1.1         input_voltage_min_v           Integer  false     1.0      volts
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.19.3.1         output_load_pct               Integer  true      1.0      percent
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.1.1         input_voltage_l1_v            Integer  true      1.0      volts
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.2.1         input_voltage_l2_v            Integer  false     1.0      volts    (bypass line)
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.20.3.1         output_voltage_l1_v           Integer  true      1.0      volts
.1.3.6.1.4.1.476.1.42.3.5.2.4.1.4.1            battery_voltage_v             Integer  true      1.0      volts
.1.3.6.1.4.1.476.1.42.3.5.2.4.1.5.1            battery_current_a             Integer  true      1.0      amps
.1.3.6.1.4.1.476.1.42.3.5.2.4.1.6.1            battery_nominal_voltage_v     Integer  false     1.0      volts    (device registration — semi-static)
.1.3.6.1.4.1.476.1.42.3.5.3.7.0                on_inverter_flag              enum     true      1.0      bool     (is_enum=true, enum_value=1 means "yes")
```

Metadata (go into metadata{} envelope, NOT feature_values):
```
OID                                              feature_name (metadata key)   type       notes
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.1          meas_point_1_1_ref            OID        structural reference OID — value_type=OID rows
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.1.2          meas_point_1_2_ref            OID        structural reference OID
.1.3.6.1.4.1.476.1.42.3.5.2.3.1.3.2.1          meas_point_bypass_ref         OID        structural reference OID
.1.3.6.1.4.1.476.1.42.3.5.2.4.1.2.1            dc_point_battery_ref          OID        points to lgpPwrMeasBattery
.1.3.6.1.4.1.476.1.42.3.5.2.4.1.3.1            dc_point_sub_id               Integer    sub-index, structural
```

Note on OID-type rows: any row with value_type="OID" is a structural/pointer OID.
They contain OID values like ".1.3.6.1.4.1.476.1.42.3.5.2.1.1" (lgpPwrSource1Input).
These are NEVER numeric features. Mark is_metadata=true for all OID-type rows.

Note on scaling: Liebert encodes power factor as integer 0-100 representing 0.00-1.00.
Apply scaling_factor=0.01 so the model sees 0.95 not 95.

---

## 3. The two core services

### Service A — Live Event Forwarder

**What it is:** An always-on Python process. Never sleeps. Restarts immediately on crash.

**Data source: Kafka (hybrid approach — Option A)**
The SNMP poller already publishes raw poll results to a per-device Kafka topic
(confirmed from devices: `output.data_topic: "snmp.output.{device_id}"`).
The live forwarder consumes directly from these Kafka topics.

PostgreSQL is NOT used for raw event reading in this service.
PostgreSQL is used ONLY for reading devices and feature_map (config management).

Why Kafka for live path:
- No SELECT loop polling — Kafka consumer offset handles position automatically
- Kafka resumes from exact offset on crash/restart — no last_seen_id state file needed
- Eliminates read pressure on snmp_polling table during high-throughput writes
- Kafka consumer lag is a standard observable metric — better operational visibility
- Naturally parallelisable — add consumer instances to the same group_id to scale

The SNMP poller writes to BOTH Kafka AND PostgreSQL (snmp_polling table).
Both paths are fully independent. Option A chosen deliberately:
- Kafka → live forwarder → scoring services  (sub-second path)
- PostgreSQL → ETL batch service → ClickHouse  (15-minute path)

**What it does:**
1. On startup:
   - Read devices and feature_map from PostgreSQL into memory. Build two indexes:
     - device_index: device_id → {vendor, category, device_type, kafka_topic,
                                   raw_kafka_topic, dimensions}
     - feature_index: (vendor, category) → {oid → {feature_name, value_type,
                                                     required, is_metadata,
                                                     scaling_factor, is_enum}}
   - Subscribe to Kafka topics for all active devices:
     topics = [devices.output_data_topic for each active device]
     e.g. ["snmp.output.Vertiv-US-3.39", "snmp.output.APC-US-3.40", ...]
     group_id = "snmp-live-forwarder"

2. Main loop — Kafka consumer poll:
   ```
   messages = consumer.poll(timeout_ms=100, max_records=2000)
   ```
   - If 0 messages → loop immediately (Kafka handles the wait internally)
   - Process all returned messages in order
   - Commit offsets after successful processing of each batch

3. For each message (one OID row per message, same shape as snmp_polling row):
   - Parse JSON payload
   - Normalise OID: strip leading dot before any lookup
   - Check device_id is in device_index. If not → skip silently, log once per unknown device_id
   - If value_type is "OID" → structural pointer row. Keep only if is_metadata=true. Skip otherwise.
   - Buffer row under key: poll_id (primary) or (device_id, floor(timestamp_ms/2000)*2000) (fallback)

4. After buffering each row, check completion for all open poll groups:
   - Poll is complete when: buffered row count for poll_id == result_count
   - Poll is stale when: oldest row in group is > 3 seconds old (partial poll — drop and flush)
   - On complete OR stale → assemble, extract features, publish to routed Kafka topic

5. Feature extraction per assembled poll:
   - For each OID in feature_index for this device's vendor+category:
     - If is_enum=true → use enum_value cast to float
     - If value_type="OID" and is_metadata=true → carry value_resolved string in metadata{}
     - Otherwise → cast value to float, apply scaling_factor
     - If OID absent or sentinel value → null in feature_values, add to missing_features[]
   - Build RoutedEvent envelope (see Section 4)

6. Publish RoutedEvent to routed Kafka topic:
   - Topic = kafka_topic from feature_index for this category (e.g. snmp-events-power)
   - Key = device_id (guarantees per-device ordering within partitions)
   - acks=all for durability

7. Config refresh: every 60 seconds, check devices and feature_map
   WHERE updated_at > last_config_check. Reload only changed rows.
   On new device registered:
     - Add to device_index
     - Subscribe to its new raw Kafka topic (consumer.subscribe() with updated topic list)
     - Starts being processed immediately
   On device deactivated:
     - Remove from device_index
     - Update subscription list (unsubscribe from its raw topic)

**Scaling — when one instance is not enough:**
Add more consumer instances to the same Kafka consumer group_id.
Kafka automatically distributes topic partitions across instances.
No manual sharding or coordination needed — Kafka handles it.
Each instance processes a subset of device topics.

**What it does NOT do:**
- Does not read from snmp_polling table in PostgreSQL
- Does not write to ClickHouse
- Does not run ML models
- Does not score anomalies
- Does NOT use PostgreSQL LISTEN/NOTIFY
- Does NOT use a short-poll SELECT loop against PostgreSQL

**Failure behaviour:**
- If Kafka (source) is unavailable: consumer blocks waiting for broker — reconnects automatically
- If Kafka (destination routed topics) is unavailable: buffer assembled RoutedEvents in bounded
  in-memory queue (max 10,000). Drop oldest if queue fills. Log dropped count as metric.
- On crash and restart: Kafka consumer resumes from last committed offset automatically.
  No state file needed. At most one batch of messages (max_records=2000) may be reprocessed.
  Kafka deduplication by poll_id handles duplicates on the scoring service side.
- If PostgreSQL (config) is unavailable on startup: retry with exponential backoff.
  If config was loaded before crash: use cached config and log warning.

**Deployment:** Docker container, restart=always

### Service B — Batch ETL (Airflow DAG 1)

**What it is:** An Airflow DAG. Runs every 15 minutes. Job = feed ClickHouse for ML training.

**What it does:**
1. On each run:
   a. Refresh config: reload changed devices and feature_map rows
   b. Read watermark: last successfully processed timestamp_ms (stored in Airflow Variable)
   c. Query snmp_polling:
      SELECT rows WHERE device_id IN (active devices) AND oid IN (configured OIDs)
      AND inserted_at > watermark ORDER BY inserted_at ASC LIMIT batch_size
   d. Group rows by poll_id (primary) or (device_id, floor(timestamp_ms/2000)*2000) (fallback)
      → assembled polls. This is identical logic to the live forwarder — use shared/assembly/poll_assembler.py
   e. For each assembled poll:
      - Extract features using feature_map
      - Run data quality gate (type check, sentinel detection, missing OID flagging)
      - Separate passed rows from quarantine rows
   f. Bulk insert passed rows → ClickHouse snmp_features
   g. Bulk insert quarantine rows → ClickHouse snmp_quarantine
   h. Update watermark to max(inserted_at) of processed rows
   i. Emit metrics: rows extracted, passed, quarantined, write duration
   j. Signal Airflow Dataset → triggers ML training DAG if enough new data

2. max_active_runs=1 — never overlap
3. Retry 3 times with exponential backoff on failure
4. On failure: watermark not updated → next run reprocesses from same point

**What it does NOT do:**
- Does not publish to Kafka
- Does not score anomalies
- Does not read from Kafka

---

## 4. The RoutedEvent envelope

Both services produce this exact same shape. The scoring service deserialises this.

```json
{
  "device_id":          "Vertiv-US-3.39",
  "device_type":        "ups",
  "category":           "power",
  "vendor":             "vertiv",
  "tenant":             "ups-core",
  "site_id":            105,
  "rack":               "R15",
  "is_production":      true,
  "timestamp":          1778745695,
  "timestamp_ms":       1778745695710,
  "poll_id":            "343412a5-196f-48a5-98d9-40e5eeacf478",
  "poll_result_count":  25,
  "feature_values": {
    "input_current_l1_a":       2.0,
    "output_current_l1_a":      3.0,
    "output_apparent_power_va": 600.0,
    "output_true_power_w":      450.0,
    "input_power_factor":       0.95,
    "output_power_factor":      0.74,
    "input_voltage_max_v":      238.0,
    "input_voltage_min_v":      0.0,
    "output_load_pct":          12.0,
    "input_voltage_l1_v":       215.0,
    "input_voltage_l2_v":       214.0,
    "output_voltage_l1_v":      208.0,
    "battery_voltage_v":        218.0,
    "battery_current_a":        0.0,
    "battery_nominal_voltage_v": 192.0,
    "on_inverter_flag":         1.0
  },
  "metadata": {
    "meas_point_1_1_ref":    "lgpPwrSource1Input",
    "meas_point_bypass_ref": "lgpPwrMeasBypass",
    "dc_point_battery_ref":  "lgpPwrMeasBattery"
  },
  "missing_features":   [],
  "pipeline_version":   "1.0.0",
  "source":             "live_forwarder"
}
```

Rules:
- `feature_values` contains only non-metadata OIDs — all cast to float
- Enum rows: use enum_value cast to float (e.g. on_inverter_flag = 1.0 means "yes")
- OID-type rows (value_type="OID"): carry value_resolved string in metadata, never in feature_values
- Sentinel values (4294967295 etc.) become null in feature_values, name added to missing_features[]
- scaling_factor from feature_map is applied before writing to feature_values
  (e.g. input_power_factor raw=95 × 0.01 = 0.95)
- `metadata` contains is_metadata=true OIDs — for context, not for ML
- `missing_features` lists required OIDs that were absent or sentinel — for DQ tracking
- `poll_id` preserved for audit and deduplication in ClickHouse
- `source` is "live_forwarder" or "etl_batch" — for audit

---

## 5. Kafka topics

There are two distinct sets of Kafka topics in this system. They serve different purposes.

**Tier 1 — Raw per-device topics (written by SNMP poller, read by live forwarder):**
```
snmp.output.{device_id}     one topic per registered device
                             e.g. snmp.output.Vertiv-US-3.39
                             e.g. snmp.output.APC-US-3.40
                             written by: SNMP poller engine
                             consumed by: live forwarder (group_id = snmp-live-forwarder)
                             message shape: raw OID row JSON (same schema as snmp_polling table)
                             retention: 24 hours (live forwarder processes in seconds)
                             partitions: 1 per topic (one device, ordered events)
```

The topic name comes from `devices.output_data_topic` — already confirmed in sample config.
When a new device is registered, its raw topic is created automatically by the SNMP poller.
The live forwarder subscribes to the new topic on its next config refresh (within 60 seconds).

**Tier 2 — Routed category topics (written by live forwarder, read by scoring services):**
```
snmp-events-power            ups, pdu devices
snmp-events-network          router, switch, firewall devices
snmp-events-environment      sensor, hvac, pdu_env devices
```

One topic per category. Add a new row to feature_map with a new kafka_topic value
and that category is automatically routed — no code changes needed.
Message shape: RoutedEvent envelope (see Section 4) — fully assembled, feature-extracted.

**Kafka config — Tier 1 (raw topics):**
```
acks = all
replication_factor = 1 (dev) / 3 (prod)
retention = 24 hours
key = device_id
partitions = 1 per topic
auto.create.topics.enable = true  (poller creates topics on first publish)
```

**Kafka config — Tier 2 (routed topics):**
```
acks = all
replication_factor = 1 (dev) / 3 (prod)
retention = 48 hours  (longer — scoring service may be restarted)
key = device_id  (preserves per-device ordering within partitions)
partitions = 6 per topic  (allows 6 parallel scoring service consumers)
```

---

## 6. ClickHouse tables

### snmp_features — main ML training table

```sql
CREATE TABLE snmp_features (
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
    feature_values      String,        -- JSON string of feature_values dict
    missing_features    Array(String),
    pipeline_version    String,
    batch_id            String,
    inserted_at         DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY (category, toYYYYMM(timestamp))
ORDER BY (device_id, timestamp)
TTL timestamp + INTERVAL 2 YEAR;
```

### snmp_anomaly_scores — written by live scoring services

```sql
CREATE TABLE snmp_anomaly_scores (
    device_id               String,
    device_type             LowCardinality(String),
    category                LowCardinality(String),
    scored_at               DateTime,
    timestamp_ms            UInt64,
    lstm_recon_error        Nullable(Float64),
    lstm_threshold          Nullable(Float64),
    iso_forest_score        Float64,
    iso_forest_threshold    Float64,
    is_anomaly              UInt8,
    alert_state             String,    -- SUSPECTED/CONFIRMED/LSTM_ONLY/CLEARED
    anomaly_reason          String,
    top_features            Array(String),
    compound_alert          UInt8,
    compound_peer           String,
    model_version           String,
    calibration_status      String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(scored_at)
ORDER BY (device_id, scored_at);
```

### snmp_quarantine — bad rows from ETL

```sql
CREATE TABLE snmp_quarantine (
    device_id       String,
    raw_json        String,
    failure_reason  String,
    batch_id        String,
    created_at      DateTime DEFAULT now()
)
ENGINE = MergeTree()
ORDER BY (device_id, created_at);
```

---

## 7. ML Training Service (Airflow DAG 2)

Triggered automatically by Airflow Dataset when DAG 1 writes new data to ClickHouse.

Steps:
1. Check data volume: only retrain if new rows since last training >= min_rows_for_retrain
2. Build training dataset: query ClickHouse snmp_features for last N days, save as Parquet
3. Train Isolation Forest and LSTM Autoencoder in parallel
4. Evaluate: if F1 < min_f1_threshold → fail task, do not deploy, alert team
5. Deploy: write new model files to shared model storage volume
6. Reload: POST /reload-model to each live scoring service
7. Log: write training metrics to ClickHouse

Model storage: shared Docker volume mounted read-write by ML training, read-only by scoring services.

---

## 8. Live Scoring Services

One service per category. Each is a separate Docker container.
All share the same codebase — category is passed as an environment variable.

**Existing codebase location:** `snmp_anomaly_detection/power/streaming/detect_power_kafka.py`
**Consumer topic confirmed:** `snmp-events-power` — already correct, no change needed.
**Model paths:** defined in `snmp_anomaly_detection/config.py` — read from there always.
**Trained seed models:** `snmp_anomaly_detection/outputs/power/baseline/` and `power/dual/`

Each service:
- Consumes from its Kafka topic (e.g. snmp-events-power) — already wired
- Deserialises RoutedEvent envelope — reads from `feature_values` dict
- Runs Isolation Forest immediately on feature_values → score per event
- Maintains RollingWindowBuffer per device (maxlen=20)
- When buffer full: runs LSTM Autoencoder on (20, n_features) window
- Runs alert state machine (SUSPECTED → CONFIRMED → CLEARED / LSTM_ONLY)
- Runs compound alert check (UPS + PDU anomaly within ±5 min)
- Runs Welford online threshold update
- Writes PowerScoringResult to ClickHouse snmp_anomaly_scores
- Exposes FastAPI endpoint: POST /reload-model, GET /health, GET /ready

**Minimal changes required to existing detect_power_kafka.py:**
- Add `reload_models()` — hot-swap without restarting consumer
- Add `is_ready()` — for /ready health endpoint
- Both must read paths from `config.py` — never hardcode

See SNMP_SCORING_IMPROVEMENTS.md for six known issues to fix before production.

---

## 9. Monitoring and alerting

Each service must expose metrics and have defined alert thresholds.
Without this the system is blind in production. Implement before go-live.

**Live forwarder — key metrics to expose:**

| Metric | Type | Alert when |
|---|---|---|
| `forwarder_kafka_consumer_lag` | Gauge per topic | > 5000 messages behind |
| `forwarder_messages_processed_per_sec` | Gauge | Drops to 0 for > 30 seconds |
| `forwarder_kafka_output_buffer_size` | Gauge | > 5000 (output Kafka unreachable) |
| `forwarder_unknown_devices_total` | Counter | Spikes (new devices not registered) |
| `forwarder_stale_polls_total` | Counter | Sustained rise (device dropping OID rows) |
| `forwarder_assembly_lag_ms` | Histogram | p99 > 3000ms |
| `forwarder_pg_reconnects_total` | Counter | Any non-zero (PostgreSQL config instability) |
| `forwarder_active_topic_count` | Gauge | Drops unexpectedly (device deregistered) |

**Batch ETL — key metrics to expose:**

| Metric | Type | Alert when |
|---|---|---|
| `etl_rows_extracted` | Gauge (per run) | 0 rows for 2+ consecutive runs |
| `etl_rows_written_clickhouse` | Gauge (per run) | < 50% of extracted rows |
| `etl_quarantine_rate_pct` | Gauge (per run) | > 5% (data quality degrading) |
| `etl_run_duration_seconds` | Gauge | > 600s (approaching next schedule) |
| `etl_watermark_lag_seconds` | Gauge | > 1800s (30 min behind) |

**Live scoring services — key metrics to expose:**

| Metric | Type | Alert when |
|---|---|---|
| `scoring_kafka_consumer_lag` | Gauge | > 10000 messages (scoring falling behind) |
| `scoring_events_per_sec` | Gauge | Drops to 0 (Kafka topic empty or consumer broken) |
| `scoring_anomalies_confirmed_total` | Counter | Spike > 2× baseline in 5 min |
| `scoring_buffer_fill_rate` | Gauge | Drops suddenly (devices going offline) |
| `scoring_model_version` | Label/Info | Changes unexpectedly (reload called unintentionally) |

**Implementation:** expose all metrics via Prometheus `/metrics` endpoint on each service.
Use `prometheus-client` Python library. Wire to Grafana for dashboards.

**Minimum health check endpoints (required on every service):**
```
GET /health
→ 200 if service is running and PostgreSQL/Kafka connections are alive
→ 503 if any dependency is unreachable

GET /ready
→ 200 if service has loaded config and is actively processing
→ 503 if still initialising (e.g. config not yet loaded)
```

Docker Compose and Kubernetes readiness probes must use `/ready`, not `/health`.
A service can be healthy (process running) but not ready (config still loading).

**Alerting rules (wire to PagerDuty or Slack):**
- `forwarder_kafka_consumer_lag` > 5000 for > 60s → immediate alert
- `etl_quarantine_rate_pct` > 5% → warning
- `scoring_kafka_consumer_lag` > 10000 → warning
- Any service `/health` returning 503 for > 30s → immediate alert
- `scoring_anomalies_confirmed_total` spike > 2× baseline → notify on-call

---

## 10. Folder structure

```
project_root/                         ← snmp_anomaly_detection/ git repo
│
├── CLAUDE.md                         ← SYSTEM_DESIGN_HANDOFF.md content here
├── docker-compose.yml
├── .env.template                     ← never commit .env
│
├── docs/                             ← all plan documents live here
│
├── config/migrations/
│   ├── 001_create_tables.sql         ← snmp_polling + feature_map (devices already exists)
│   ├── 002_partition_helpers.sql
│   ├── 003_seed_feature_map.sql
│   └── 004_clickhouse_tables.sql
│
├── shared/                           ← imported by ALL services — never duplicated
│   ├── constants.py                  ← sentinel values, normalise_oid(), thresholds
│   ├── models/
│   │   └── routed_event.py           ← RoutedEvent dataclass
│   ├── assembly/
│   │   ├── poll_assembler.py         ← single implementation used by both services
│   │   └── event_builder.py          ← builds RoutedEvent, applies scaling
│   └── config/
│       └── pg_config_watcher.py      ← ONE copy — imported by live_forwarder AND etl
│
├── live_forwarder/                   ← Service A
│   ├── Dockerfile
│   ├── supervisord.conf              ← runs main.py + api.py together
│   ├── main.py                       ← entry point: LiveForwarder.run()
│   └── routing/
│       └── kafka_publisher.py        ← publishes RoutedEvent to snmp-events-{category}
│   (no config/ here — imports shared/config/pg_config_watcher.py)
│
├── live_scoring/                     ← Docker build context for scoring container
│   ├── Dockerfile
│   ├── supervisord.conf              ← runs detect_power_kafka.py + api.py together
│   ├── api.py                        ← FastAPI /reload-model /health /ready /metrics
│   └── requirements.txt
│   (mounts snmp_anomaly_detection/ — scoring logic stays in existing package)
│
├── etl/                              ← Service B (Airflow DAG 1 + DAG 2)
│   ├── requirements.txt
│   ├── dags/
│   │   ├── snmp_etl_pipeline.py      ← DAG 1
│   │   └── snmp_ml_retrain.py        ← DAG 2, imports from snmp_anomaly_detection/power/training/
│   ├── extract/
│   │   └── pg_polling_extractor.py
│   ├── quality/
│   │   ├── dq_gate.py
│   │   └── quarantine_writer.py
│   └── load/
│       └── ch_writer.py
│   (no config/ here — imports shared/config/pg_config_watcher.py)
│
├── scripts/
│   └── create_kafka_topics.sh
│
├── tests/
│   └── produce_test_poll.py
│
└── snmp_anomaly_detection/           ← existing package — minimal changes only
    ├── config.py                     ← model paths — read always, never change
    ├── power/streaming/
    │   └── detect_power_kafka.py     ← add reload_models() + is_ready() only
    ├── power/training/                ← DAG 2 imports directly from here
    │   ├── train_baseline_power.py
    │   └── train_iforest_power.py
    ├── power/evaluation/              ← DAG 2 imports directly from here
    │   └── power_eval.py
    ├── outputs/                      ← existing trained models = seed models
    │   ├── power/baseline/
    │   └── power/dual/
    └── ... everything else unchanged
```

---

## 11. Docker Compose — startup order and volume spec

This section is mandatory. Claude Code must implement docker-compose.yml exactly as specified here.
Missing a `depends_on` causes silent startup failures that are hard to debug.

**Startup order (strict — do not change):**
```
1. postgres          — must be healthy before anything else starts
2. zookeeper         — must be healthy before kafka
3. kafka             — must be healthy before live-forwarder and airflow-scheduler
4. clickhouse        — must be healthy before airflow-scheduler and live-scoring-*
5. airflow-init      — runs once, must EXIT CLEANLY before airflow-scheduler starts
6. airflow-scheduler — depends on: postgres (healthy), airflow-init (completed)
7. airflow-webserver — depends on: airflow-scheduler (started)
8. live-forwarder    — depends on: kafka (healthy), postgres (healthy for config)
                       NOTE: live-forwarder reads RAW events from Kafka, config from PostgreSQL
9. live-scoring-power — depends on: kafka (healthy), clickhouse (healthy)
   NOTE: on first boot, the scoring service bootstrap copies seed models from
         snmp_anomaly_detection/outputs/ into model_storage volume automatically.
         No manual seed model copy needed.
```

**Shared volumes (critical — get these wrong and model reload breaks):**
```
model_storage:
  mounted READ-WRITE by: airflow-scheduler (DAG 2 deploys new models here)
  mounted READ-ONLY  by: live-scoring-power, live-scoring-network, live-scoring-env
  All scoring services and airflow-scheduler must mount the SAME named volume.

buffer_checkpoints:
  mounted by: live-scoring-power (and other scoring services)
  persists rolling window buffer state across container restarts
```

Note: forwarder_state volume is NOT needed in the hybrid approach.
Kafka consumer offsets are managed by the Kafka broker automatically.

Note: seed models come from `snmp_anomaly_detection/outputs/` — the bootstrap
function in the scoring service copies them to model_storage on first boot.
No separate seed/ directory needed.

**Restart policies:**
```
live-forwarder:       restart: always   — must never stay down
live-scoring-*:       restart: always   — must never stay down
airflow-scheduler:    restart: unless-stopped
postgres:             restart: unless-stopped
kafka:                restart: unless-stopped
clickhouse:           restart: unless-stopped
airflow-init:         restart: no       — runs once only
```

**Health check requirements (every service must have one):**
```
postgres:     pg_isready -U ${POSTGRES_USER}
kafka:        kafka-broker-api-versions --bootstrap-server localhost:9092
clickhouse:   clickhouse-client --query "SELECT 1"
live-forwarder:    curl -f http://localhost:8001/health
live-scoring-*:    curl -f http://localhost:8000/health
airflow-webserver: curl -f http://localhost:8080/health
```

---

## 12. Shared assembly logic — important rule

Three things must live in `shared/` and never be duplicated:

**`shared/assembly/poll_assembler.py` and `shared/assembly/event_builder.py`**
Both `live_forwarder` and `etl` import from here. Never copy these files.

**`shared/config/pg_config_watcher.py`**
Both `live_forwarder` and `etl` import this. One file, two importers.
Never create `live_forwarder/config/pg_config_watcher.py` or
`etl/config/pg_config_watcher.py` — those paths must not exist.

**`shared/constants.py`** must contain the OID normalisation function used by both:

```python
def normalise_oid(oid: str) -> str:
    """Strip leading dot. Always call this before feature_map lookup."""
    return oid.lstrip(".")
```

---

## 13. Database migration — feature_map seed data

The feature_map table must be seeded with OID mappings before any data flows.
Vertiv UPS seed data is fully confirmed from real polling data — use it exactly.

Build the seed file at `config/migrations/003_seed_feature_map.sql`.

The 16 confirmed Vertiv UPS feature OIDs are documented in Section 2 Table 3 above.
Seed all of them. The ML feature OIDs go into feature_values. The metadata OIDs go into metadata.
All OIDs in the seed SQL must be stored WITHOUT a leading dot (normalised form).

The scoring service (detect_power_kafka.py) uses feature names from `BASELINE_UPS_FEATURES`.
The names in the seed data above MUST match those exactly:
- output_load_pct
- input_voltage_l1_v
- output_voltage_l1_v
- battery_voltage_v
- battery_current_a
- on_inverter_flag
- output_true_power_w
- output_apparent_power_va
- input_power_factor
- output_power_factor
(Check detect_power_kafka.py BASELINE_UPS_FEATURES list and align — these names are the contract)

---

## 14. Build order for Claude Code

Work in this sequence. Do not skip ahead.

1. Database migrations — create snmp_polling (partitioned) and feature_map tables.
   devices table already exists — do NOT recreate it. Verify it is accessible first.
2. Seed feature_map — Vertiv UPS OIDs (no leading dots, scaling_factor confirmed)
3. Shared library — RoutedEvent dataclass, poll_assembler, event_builder, normalise_oid()
4. Live forwarder — pg_config_watcher, Kafka consumer loop (snmp.output.* topics), kafka_publisher, /health + /metrics
5. Test live forwarder end to end — publish sample OID rows to snmp.output.Vertiv-US-3.39
   Kafka topic using a test producer, verify RoutedEvent appears on snmp-events-power
6. Batch ETL — DAG 1 with partition management tasks, pg_polling_extractor, dq_gate, ch_writer
7. Test ETL end to end, verify ClickHouse rows match live forwarder output
8. Wire live scoring service to consume from snmp-events-power
9. Add /reload-model + /health + /ready + /metrics endpoints to live scoring service
10. DAG 2 — ML training pipeline
11. Docker Compose — implement exactly per Section 11 startup order and volume spec
12. Fix six issues from SNMP_SCORING_IMPROVEMENTS.md

---

## 15. Key rules — do not violate these

- A device not in devices must never be processed. Drop silently, log as unknown.
- A device where `active = false` or `polling_config->>'enabled' = false` must be skipped.
- Read devices using JSONB operators — never assume flat columns for polling/output fields.
- `output_config->>'data_topic'` is the Kafka topic to subscribe to for each device.
- `polling_config->'oids'` contains the specific OIDs polled — may have leading dots, normalise before feature_map lookup.
- Sentinel values (4294967295, 2147483647, 65535, 255) must become null, never be used as features.
- OID-type rows (value_type="OID") must NEVER be cast to float. They are structural pointers. Metadata only.
- Enum rows: always use enum_value (integer) as the numeric feature, not the raw value string.
- scaling_factor from feature_map must always be applied before writing to feature_values.
- Always call normalise_oid() before any feature_map lookup — strip the leading dot.
- OIDs in feature_map seed SQL must be stored without leading dot. Enforce on insert.
- poll_id is the primary grouping key. Never group by timestamp alone when poll_id is present.
- result_count tells the assembler when a poll is complete. Use it. Do not guess.
- The live forwarder reads RAW events from Kafka (snmp.output.{device_id} topics).
- The live forwarder reads CONFIG from PostgreSQL (devices + feature_map tables).
- The live forwarder must NEVER read from snmp_polling table.
- The live forwarder must NEVER write to ClickHouse.
- Kafka consumer offsets are committed after successful batch processing — never before.
- When a new device is registered, subscribe to its raw Kafka topic within 60 seconds.
- Do NOT add a NOTIFY trigger to snmp_polling. It is not used by any service in this architecture.
- snmp_polling must be created as a partitioned table. Never as a plain table.
- Partition management (create tomorrow, drop old) runs at the top of DAG 1 before extract.
- poll_assembler and event_builder must live in shared/ — never duplicated in either service.
- Kafka Tier 1 topics (raw per-device): 24-hour retention, 1 partition each.
- Kafka Tier 2 topics (routed category): 48-hour retention, 6 partitions each.
- Kafka messages must be keyed by device_id — never publish without a key.
- The batch ETL reads from snmp_polling in PostgreSQL — never from Kafka.
- The batch ETL must never publish to Kafka.
- ClickHouse inserts must be batched — never insert one row at a time.
- The watermark in DAG 1 must only update after a successful ClickHouse write.
- Model deployment in DAG 2 must call /reload-model on all scoring services after writing model files.
- Config refresh in both services must be incremental — only reload rows where updated_at changed.
- feature_name values in feature_map seed data must match BASELINE_UPS_FEATURES in detect_power_kafka.py exactly.
- Every service must expose /health, /ready, and /metrics. Docker Compose health checks use /health.
