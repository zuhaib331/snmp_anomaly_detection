# Phase 2: Device Window Manager

## Objective

Add per-device rolling window state so detection can work from sequential events instead of only full grouped dataframes.

## Scope

- Add an in-memory window manager keyed by `device_id`.
- Keep an independent rolling buffer for each device.
- Emit a ready window once a device has collected `sequence_length` events.
- Reuse the same scorer from Phase 1 for emitted windows.
- Update CSV detection to build windows through the window manager.

## Implementation Changes

- Add `snmp_anomaly_detection.network.inference.window_manager`.
- Introduce a reusable `DeviceWindowManager` that accepts one event at a time.
- Introduce a `ReadyWindow` structure carrying metadata and feature values for scoring.
- Update the CSV detection path to iterate rows sequentially and emit windows from the manager.

## Inputs and Outputs

Inputs:
- normalized event records with:
  - `timestamp`
  - `device_id`
  - feature columns
  - optional `anomaly`

Outputs:
- one ready-to-score window per device whenever the rolling buffer is full
- window metadata:
  - `device_id`
  - `device_window_index`
  - `window_start`
  - `window_end`
  - `source_anomaly_label`

## Acceptance Criteria

- Interleaved device events remain isolated.
- Each device produces windows independently.
- CSV detection works using the window manager.
- Existing result files are still generated successfully.

## Out of Scope

- Kafka consumer integration
- output topic publishing
- malformed event validation
- inactive device state expiry
