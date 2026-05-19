-- Migration 004: ClickHouse tables
-- Target: local ClickHouse (running on your machine — NOT PostgreSQL).
--
-- Connect before running:
--   clickhouse-client --database <your_local_ch_database>
--
-- All three tables use idempotent CREATE TABLE IF NOT EXISTS.

-- ── snmp_features ────────────────────────────────────────────────────────────
-- Main ML training table. Written by the ETL batch service (15-minute cadence).
-- feature_values is a JSON string — parsed by the feature assembler before scoring.
-- ReplacingMergeTree deduplicates by (device_id, timestamp) keeping the row
-- with the latest inserted_at.
-- TTL: 2 years. ClickHouse enforces this at merge time.

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

-- ── snmp_anomaly_scores ──────────────────────────────────────────────────────
-- Written by live scoring services (detect_power_kafka.py and network equivalent).
-- One row per scored window. alert_state matches the E7 two-stage lifecycle:
--   suspected | confirmed | lstm_only | cleared
-- compound_alert + compound_peer: reserved for cross-device correlation (Phase 3+).

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

-- ── snmp_quarantine ──────────────────────────────────────────────────────────
-- Receives bad rows rejected by the ETL data-quality gate.
-- raw_json preserves the original message for inspection and replay.
-- failure_reason is a short code: "missing_required_features", "sentinel_value",
--   "parse_error", "unknown_device", etc.
-- TTL: 90 days — long enough for investigation, short enough to not accumulate forever.

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
