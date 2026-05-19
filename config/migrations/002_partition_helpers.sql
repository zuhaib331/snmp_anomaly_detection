-- Migration 002: partition helper functions + initial daily partitions
-- Target: local PostgreSQL, database snmp_anomaly_detection_db (running on your machine)
-- Run after 001_create_tables.sql.
--
-- Connect before running:
--   psql -d snmp_anomaly_detection_db -U <your_local_superuser>
--
-- Today: 2026-05-15. Partitions cover yesterday + today + 7 days forward (9 total).

-- ── Helper: create_snmp_partition ───────────────────────────────────────────
-- Creates a single daily partition for snmp_polling.
-- Called by DAG 1 create_tomorrows_partition task each day.
-- Idempotent: skips if partition already exists.

CREATE OR REPLACE FUNCTION create_snmp_partition(target_date DATE)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    start_date TIMESTAMPTZ;
    end_date   TIMESTAMPTZ;
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

-- ── Helper: drop_snmp_partition ──────────────────────────────────────────────
-- Drops a daily partition. Called by DAG 1 drop_old_partitions task.
-- Retention window: 7 days. Partitions older than 7 days are dropped.
-- Data is already in ClickHouse snmp_features by the time a partition is dropped.

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

-- ── Initial partitions ───────────────────────────────────────────────────────
-- Covers: yesterday (2026-05-14) + today (2026-05-15) + 7 days forward.
-- All timestamps are UTC midnight boundaries.

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d14
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-14 00:00:00+00') TO ('2026-05-15 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d15
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-15 00:00:00+00') TO ('2026-05-16 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d16
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-16 00:00:00+00') TO ('2026-05-17 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d17
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-17 00:00:00+00') TO ('2026-05-18 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d18
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-18 00:00:00+00') TO ('2026-05-19 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d19
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-19 00:00:00+00') TO ('2026-05-20 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d20
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-20 00:00:00+00') TO ('2026-05-21 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d21
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-21 00:00:00+00') TO ('2026-05-22 00:00:00+00');

CREATE TABLE IF NOT EXISTS snmp_polling_y2026m05d22
    PARTITION OF snmp_polling
    FOR VALUES FROM ('2026-05-22 00:00:00+00') TO ('2026-05-23 00:00:00+00');

-- ── pg_partman alternative ───────────────────────────────────────────────────
-- If pg_partman is available on your PostgreSQL instance, replace the manual
-- partition blocks above with:
--
--   SELECT partman.create_parent(
--       p_parent_table  => 'public.snmp_polling',
--       p_control       => 'inserted_at',
--       p_type          => 'range',
--       p_interval      => '1 day',
--       p_premake       => 7,
--       p_retention     => '7 days',
--       p_retention_keep_table => false
--   );
--
-- pg_partman handles creation and retention automatically, removing the need
-- for DAG 1 partition management tasks.
-- Check availability: SELECT * FROM pg_extension WHERE extname = 'pg_partman';
