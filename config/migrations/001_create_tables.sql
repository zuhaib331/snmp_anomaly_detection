-- Migration 001: snmp_polling (partitioned) + feature_map
-- Target: local PostgreSQL, database snmp_anomaly_detection_db (running on your machine)
-- Run as a superuser (DDL + trigger creation required).
--
-- Connect before running:
--   psql -d snmp_anomaly_detection_db -U <your_local_superuser>
--
-- NOTE: devices table lives on the remote VM (192.168.25.47/snmp_engine).
-- This file does NOT touch it. Read access to that table is handled separately
-- via a remote connection (SELECT only).
-- The snmp_polling and feature_map tables created here are LOCAL.

-- ── Utility function (touch_updated_at) ─────────────────────────────────────
-- Used by the feature_map trigger below.
-- CREATE OR REPLACE is idempotent; safe to re-run.

CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ── Table: snmp_polling (partitioned by inserted_at, one partition per day) ─
-- IMPORTANT: declared as partitioned from the start.
-- Never convert a plain table to a partitioned one — it requires a full rebuild.
-- PRIMARY KEY must include the partition column (inserted_at).

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

-- Indexes on the parent table propagate automatically to each partition.
-- idx_snmp_polling_inserted_device is critical: the ETL batch query filters
-- on both inserted_at > watermark AND device_id IN (...). The single
-- inserted_at index is not selective enough at billions of rows per day.

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

-- ── Table: feature_map ───────────────────────────────────────────────────────
-- Maps specific child OIDs to vendor-agnostic ML feature names.
-- OIDs are stored WITHOUT a leading dot. Strip on insert if present.
-- The assembler strips leading dots from snmp_polling OIDs before lookup —
-- both sides must match.

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
