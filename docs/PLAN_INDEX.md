# SNMP Anomaly Detection System — Implementation Plan
> Read SYSTEM_DESIGN_HANDOFF.md before reading any plan document.
> The handoff is the single source of truth. The plan documents tell you how to build it.

---

## How to use these documents

Each phase produces a working, testable system state before the next phase begins.
Never start a phase before the previous phase's deliverables are fully checked.
Each plan document is self-contained — Claude Code can work through one at a time.

---

## The five phases

| Phase | Document | What you build | Gate to next phase |
|---|---|---|---|
| 1 | PLAN_PHASE_1_DATABASE.md | PostgreSQL tables, ClickHouse tables, feature_map seed data | All SQL verification queries pass |
| 2 | PLAN_PHASE_2_LIVE_FORWARDER.md | Shared library, live event forwarder service | RoutedEvent verified on snmp-events-power |
| 3 | PLAN_PHASE_3_ETL_SERVICE.md | Batch ETL DAG 1, ClickHouse writer | ClickHouse populated from real data |
| 4 | PLAN_PHASE_4_SCORING_ML.md | Scoring service wired, model reload, DAG 2 | Full scoring loop end to end |
| 5 | PLAN_PHASE_5_PRODUCTION.md | Docker Compose, six scoring fixes, load testing | All tests pass, system stable |

---

## System architecture summary

```
SNMP Poller
    │
    ├── writes to ──► Kafka snmp.output.{device_id}  ──► Live Forwarder
    │                                                          │
    │                   reads config from ◄── PostgreSQL       │
    │                   devices + feature_map            │
    │                                                          │ assembles + routes
    └── writes to ──► PostgreSQL snmp_polling                  ▼
                            │                       Kafka snmp-events-{category}
                            │                                  │
                            ▼                                  ▼
                    ETL Service (DAG 1)              Live Scoring Service
                    every 15 minutes                 IF + LSTM — always on
                            │                                  │
                            ▼                                  ▼
                    ClickHouse                        ClickHouse
                    snmp_features                     snmp_anomaly_scores
                            │
                            ▼
                    ML Training (DAG 2)
                    triggered by new data
                            │
                            ▼ deploys models + calls /reload-model
                    Live Scoring Service
```

---

## Files produced by each phase

**Phase 1:**
- `config/migrations/001_create_tables.sql`  ← snmp_polling + feature_map only
- `config/migrations/002_partition_helpers.sql`
- `config/migrations/003_seed_feature_map.sql`
- `config/migrations/004_clickhouse_tables.sql`

**Phase 2:**
- `shared/constants.py`
- `shared/models/routed_event.py`
- `shared/assembly/poll_assembler.py`
- `shared/assembly/event_builder.py`
- `shared/config/pg_config_watcher.py`  ← ONE copy, imported by both services
- `live_forwarder/main.py`
- `live_forwarder/routing/kafka_publisher.py`
- `live_forwarder/Dockerfile`
- `live_forwarder/supervisord.conf`
- `scripts/create_kafka_topics.sh`
- `tests/produce_test_poll.py`

**Phase 3:**
- `etl/requirements.txt`
- `etl/extract/pg_polling_extractor.py`
- `etl/quality/dq_gate.py`
- `etl/quality/quarantine_writer.py`
- `etl/load/ch_writer.py`
- `etl/dags/snmp_etl_pipeline.py`

**Phase 4:**
- `live_scoring/requirements.txt`
- `live_scoring/api.py`
- `live_scoring/Dockerfile`
- `live_scoring/supervisord.conf`
- `snmp_anomaly_detection/training/dataset_builder.py`  ← new file in existing package
- `snmp_anomaly_detection/training/deployer.py`         ← new file in existing package
- `etl/dags/snmp_ml_retrain.py`
- Updates to `snmp_anomaly_detection/streaming/detect_power_kafka.py`
  (add reload_models() + is_ready() only)

**Phase 5:**
- `docker-compose.yml`
- `.env.template`
- Updates to `snmp_anomaly_detection/inference/event_preprocessor.py` (delta gap fix)
- Updates to alert state machine (CONFIRMED_IF_ONLY)
- Updates to `PendingAlert` dataclass (TTL)
- Updates to rolling window buffer (checkpointing)
- Updates to compound alert check (O(1) lookup)

---

## Key rules to never violate

These come from SYSTEM_DESIGN_HANDOFF.md Section 15.
If any of these are violated, the system will have silent bugs.

1. All OIDs in feature_map stored WITHOUT leading dot
2. normalise_oid() called before every feature_map lookup
3. poll_id is the primary grouping key — never timestamp alone
4. Live forwarder reads Kafka topic from `output_config->>'data_topic'` — not a flat column
5. Filter active devices with `active = true AND (polling_config->>'enabled')::boolean = true`
6. ETL NEVER publishes to Kafka
7. snmp_polling MUST be a partitioned table
8. ClickHouse inserts ALWAYS batched
9. Watermark only updates after successful ClickHouse write
10. BASELINE_UPS_FEATURES in detect_power_kafka.py must match feature_map seed names

---

## Estimated effort per phase

| Phase | Complexity | Estimated sessions |
|---|---|---|
| 1 — Database | Low | 1 |
| 2 — Live forwarder | Medium | 2–3 |
| 3 — ETL service | Medium | 2–3 |
| 4 — Scoring + ML | High | 3–4 |
| 5 — Production | Medium | 2–3 |

Total: approximately 10–14 focused working sessions.
