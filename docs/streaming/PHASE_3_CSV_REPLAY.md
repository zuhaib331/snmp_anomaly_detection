# Phase 3: CSV Replay Mode

## Objective

Formalize CSV replay as an explicit input source that uses the same downstream event-processing flow intended for Kafka.

## Scope

- Add a normalized event structure for replayable input records.
- Add a reusable event processor that accepts normalized events and emits scored windows.
- Add a dedicated CSV replay adapter that converts dataframe rows into normalized events.
- Keep file-based replay outputs in the existing anomaly result files.

## Implementation Changes

- Add `snmp_anomaly_detection.network.inference.events`.
- Add `snmp_anomaly_detection.network.inference.event_processor`.
- Add `snmp_anomaly_detection.network.inference.csv_replay`.
- Make `detect` a compatibility wrapper around CSV replay mode.
- Add `detect-csv` as an explicit CLI mode for replay/testing.

## Inputs and Outputs

Inputs:
- CSV file with:
  - `timestamp`
  - `device_id`
  - `cpu`
  - `memory`
  - `in_octets`
  - `out_octets`
  - `errors`
  - optional `anomaly`

Outputs:
- `anomaly_results.csv`
- `anomaly_windows.json`
- normalized event-by-event processing through the shared pipeline

## Acceptance Criteria

- `detect-csv` works without Kafka.
- CSV rows are normalized into events before processing.
- CSV replay uses the same downstream event processor that Kafka mode can use later.
- Replay results remain deterministic for the same input file.

## Out of Scope

- Kafka consumer implementation
- schema validation hardening
- output topic publishing
- device state expiration
