# Phase 4: Kafka Live Mode

This document breaks Phase 4 into practical implementation steps for the current project.

## Goal

Consume live events from Kafka and pass them into the existing anomaly detection pipeline.

For the first implementation:
- use one hardcoded Kafka input topic
- do not implement dynamic topic switching
- do not implement output Kafka topics yet

Dynamic topic consumption can be added later after the first live pipeline is stable.

## Stage 1: Kafka Foundation

Add Kafka support to the project configuration and runtime.

### Tasks

- Add Kafka-related settings in `snmp_anomaly_detection/config.py`
- Keep only the required settings for v1:
  - `bootstrap_servers`
  - `consumer_group_id`
  - `poll_timeout_ms`
  - `micro_batch_size`
  - `micro_batch_max_wait_ms`
- Use one hardcoded input topic in code for Phase 4 v1
- Do not add dynamic topic update handling yet

### Expected outcome

The project can connect to Kafka with a simple, stable input configuration.

## Stage 2: Kafka Dry Ingestion

Build Kafka ingestion without anomaly scoring first.

### Tasks

- Create a Kafka consumer module, for example:
  - `snmp_anomaly_detection/network/streaming/kafka_source.py`
- Subscribe to the one hardcoded topic
- Read messages continuously
- Decode JSON payloads
- Extract `device_id`
- Convert each message into `NormalizedEvent`
- Add a dry-run command:
  - `detect-kafka-dry`

### Expected outcome

The project can consume Kafka messages and normalize them successfully without running the model.

## Stage 3: Input Validation

Validate incoming Kafka messages before connecting them to anomaly detection.

### Tasks

- Verify each Kafka message contains:
  - `timestamp`
  - `device_id`
  - `cpu`
  - `memory`
  - `in_octets`
  - `out_octets`
  - `errors`
- Skip malformed messages safely
- Log validation failures clearly
- Confirm interleaved devices are being separated correctly

### Expected outcome

Kafka data can be trusted enough to feed the live detection pipeline.

## Stage 4: Connect the Existing Pipeline

Connect Kafka ingestion to the current event-based detection flow.

### Reused components

- `NormalizedEvent`
- `EventProcessor`
- `DeviceWindowManager`
- `LiveMicroBatchProcessor`

### Runtime flow

```text
hardcoded Kafka topic
-> consume message
-> normalize
-> EventProcessor
-> DeviceWindowManager
-> LiveMicroBatchProcessor
-> scored anomaly result
```

### Tasks

- Add live command:
  - `detect-kafka`
- Feed Kafka events into the shared event processor
- Use the existing live micro-batching layer for low-latency scoring

### Expected outcome

The project can detect anomalies from live Kafka events using the same core logic as CSV replay.

## Stage 5: Result Handling

Start with low-risk result output.

### Tasks

- Log anomaly results locally
- Optionally write local debug output if needed
- Keep output Kafka topics out of scope for the first version
- Include these fields in live results:
  - `device_id`
  - `window_start`
  - `window_end`
  - `reconstruction_error`
  - `threshold`
  - `predicted_anomaly`
  - `top_error_feature`
  - `top_error_timestep_offset`

### Expected outcome

Live anomalies are visible and easy to verify without adding extra Kafka complexity.

## Stage 6: Operational Safety

Use conservative defaults for the first live rollout.

### Defaults

- `micro_batch_size = 8`
- `micro_batch_max_wait_ms = 50`

### Constraints

- Fail fast if Kafka config is missing
- No runtime topic reload in v1
- No dynamic topic switching in v1
- No retraining in live mode
- No out-of-order event correction yet

## Stage 7: Testing Order

Follow this sequence during implementation:

1. Test `detect-kafka-dry` with sample producer messages
2. Confirm JSON decoding works
3. Confirm normalization works
4. Confirm multi-device event segregation works
5. Confirm rolling windows are produced correctly
6. Enable `detect-kafka`
7. Compare Kafka-fed results with CSV replay on a controlled sample

## v1 Scope Boundary

Included in Phase 4 v1:
- one hardcoded Kafka input topic
- live message consumption
- event normalization
- device segregation
- rolling window creation
- micro-batch scoring
- local live result output

Not included in Phase 4 v1:
- dynamic topic consumption
- runtime topic switching
- output Kafka topic
- stream retraining
- out-of-order handling

## Success Criteria

Phase 4 v1 is successful when:

- Kafka consumer reads from the hardcoded topic reliably
- live events are normalized correctly
- device-specific windows are formed correctly
- anomalies are scored from Kafka input
- latency stays near-real-time using live micro-batching
