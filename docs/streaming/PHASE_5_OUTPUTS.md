# Phase 5: Output and Result Handling

## Objective

Make CSV replay and Kafka live detection produce the same core result schema.

## Scope

- Introduce a shared output schema builder for result records.
- Reuse the same anomaly-window record shape across CSV and Kafka.
- Keep CSV file outputs in place.
- Keep Kafka local JSONL outputs in place.

## Implementation Changes

- Add `snmp_anomaly_detection.inference.output_schema`.
- Use one shared `build_result_record(...)` helper in both CSV and Kafka paths.
- Use one shared `build_anomaly_window_record(...)` helper in both CSV and Kafka paths.
- Keep source-specific sinks, but unify record contents.

## Expected Outcome

- CSV replay result rows and Kafka live result rows are directly comparable.
- Anomaly-window exports across both modes share the same main fields.
- The output format is easier to inspect and validate.
