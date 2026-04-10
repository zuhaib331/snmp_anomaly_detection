# Phased Plan for Hybrid CSV + Kafka Anomaly Detection

## Summary
Split the work into small implementation plans so we can build and verify one piece at a time. The sequence is ordered to minimize risk: first extract reusable inference logic, then introduce per-device rolling windows, then formalize CSV replay on top of the same event pipeline, then add Kafka ingestion from a single mixed-device topic, and finally harden outputs and validation.

## Phase 1: Core Inference Refactor
Goal: separate model scoring from CSV-only batch code.

- Extract reusable functions for:
  - loading scaler/model/threshold
  - scaling one event or one window
  - scoring one window
  - building anomaly result metadata
- Keep current training flow unchanged.
- Keep current detection behavior working during refactor.
- Deliverable:
  - one inference core that accepts a single `(sequence_length, feature_count)` window and returns anomaly results

Acceptance:
- New scorer returns the same reconstruction error and anomaly flag as the current implementation for the same window.
- Existing saved model artifacts still work.
- Status: completed

## Phase 2: Device Window Manager
Goal: support per-device rolling windows independent of input source.

- Add an in-memory state manager keyed by `device_id`.
- Each device gets its own fixed-length rolling buffer.
- No result until buffer reaches full sequence length.
- After warm-up, each new event for that device emits one overlapping window.
- Update the current CSV detection path to create windows through this manager.

Deliverable:
- a reusable component that accepts normalized events and emits ready-to-score windows

Acceptance:
- Interleaved events from multiple devices stay isolated.
- Window creation is correct for each device.
- CSV detection continues to run successfully after switching to rolling device windows.
- Status: completed

## Phase 3: CSV Replay Mode
Goal: formalize CSV as an explicit replay source that uses the same event-processing pipeline intended for live Kafka input.

- Introduce a reusable event-processing pipeline after the input source.
- Define a normalized input event schema shared by CSV replay and Kafka ingestion.
- Move row-to-event normalization into a dedicated CSV replay adapter.
- Feed normalized CSV events into the device window manager and scorer.
- Keep CSV outputs (`anomaly_results.csv`, `anomaly_windows.json`) as the replay-mode outputs.

Deliverable:
- `detect-csv` mode for testing and replay

Acceptance:
- CSV replay works without Kafka.
- Results are deterministic for the same input file.
- Mixed-device CSV input produces correct per-device results.
- CSV replay and future Kafka mode share the same downstream event-processing flow.

## Phase 4: Kafka Live Mode
Goal: consume production events from one shared Kafka topic.

- Add Kafka consumer configuration.
- Consume one topic only.
- Extract `device_id` from each message and segregate in application code.
- Pass events into the same window manager and scorer used by CSV replay.

Deliverable:
- `detect-kafka` mode for live anomaly detection

Acceptance:
- One mixed-device topic can be consumed continuously.
- Device-specific windows are created correctly.
- Anomalies are produced as live results.

## Phase 5: Output and Result Handling
Goal: make both modes produce consistent results.

- Define one normalized output schema for anomaly results.
- CSV mode writes file outputs.
- Kafka mode logs results and optionally publishes to an output topic.
- Include explanation fields:
  - reconstruction error
  - threshold
  - predicted anomaly
  - top error feature
  - top error timestep

Deliverable:
- consistent result records across both modes

Acceptance:
- Same schema is used in CSV and Kafka modes.
- Output is easy to compare across environments.

## Phase 6: Validation and Hardening
Goal: make the pipeline safe enough for regular use.

- Add validation for missing or malformed fields.
- Add tests for:
  - per-device segregation
  - warm-up behavior
  - scoring parity
  - malformed records
- Document known v1 limitations:
  - no retraining in stream
  - no out-of-order correction
  - no state expiry for inactive devices

Deliverable:
- baseline tests and usage documentation

Acceptance:
- Clear error handling exists for bad input.
- Core scenarios are covered by tests.

## Documentation Structure
Create and maintain the work as a small set of docs, one per phase.

- `docs/streaming/PHASE_1_CORE_REFACTOR.md`
- `docs/streaming/PHASE_2_DEVICE_WINDOWS.md`
- `docs/streaming/PHASE_3_CSV_REPLAY.md`
- `docs/streaming/PHASE_4_KAFKA_LIVE.md`
- `docs/streaming/PHASE_5_OUTPUTS.md`
- `docs/streaming/PHASE_6_VALIDATION.md`

Each phase doc should contain:
- objective
- scope
- implementation changes
- inputs/outputs
- acceptance criteria
- out-of-scope items

## Assumptions
- We will implement one phase at a time and validate before moving on.
- Training remains offline and unchanged for now.
- Kafka mode is for live anomaly detection only.
- CSV mode is the testing/replay path.
- Single-topic mixed-device Kafka input is a fixed requirement.

## Current Architecture After Phase 2

Current detection flow:

```text
CSV dataset
  -> load dataframe
  -> scale features
  -> iterate rows sequentially
  -> route each row by device_id into DeviceWindowManager
  -> emit one ready window per device after warm-up
  -> score window with shared inference core
  -> save anomaly_results.csv / anomaly_windows.json
```

Training flow remains unchanged:

```text
CSV dataset
  -> preprocess
  -> create train/test arrays
  -> train LSTM autoencoder
  -> save scaler/model/threshold
```

## Remaining Gap Before Kafka

After Phase 2, the project is already using rolling per-device windows for CSV detection, but it still lacks:

- an explicit reusable event processor abstraction
- a dedicated CSV replay mode/adapter
- a normalized event schema shared by CSV and Kafka sources
- Kafka consumer integration
