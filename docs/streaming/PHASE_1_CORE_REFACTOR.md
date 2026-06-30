# Phase 1: Core Inference Refactor

## Objective

Separate model/scaler loading and per-window anomaly scoring from the CSV-only batch detection flow.

## Scope

- Create a reusable inference core for loading trained artifacts.
- Support scaling a full feature frame or a single window with the saved scaler.
- Support scoring one window or many windows with the saved LSTM autoencoder.
- Keep the existing training flow unchanged.
- Keep the existing `detect` command working against CSV input.

## Implementation Changes

- Add `snmp_anomaly_detection.network.inference.core` as the shared inference module.
- Move artifact loading logic into the shared module.
- Move reconstruction scoring and anomaly explanation logic into the shared module.
- Update the existing batch detector to call the shared module instead of duplicating scoring logic.

## Inputs and Outputs

Inputs:
- saved scaler artifact
- saved model artifact
- saved metadata with anomaly threshold
- one sequence window shaped `(sequence_length, feature_count)` or a batch of windows

Outputs:
- reconstruction error
- threshold
- anomaly flag
- top contributing feature
- top contributing timestep
- per-feature error values
- detection summary text

## Acceptance Criteria

- Scoring the same window through the shared core matches the existing batch detector behavior.
- Existing saved artifacts remain compatible.
- Current CSV detection still produces anomaly output files successfully.

## Out of Scope

- Kafka consumption
- per-device rolling window state
- CSV replay through event-by-event processing
- input validation hardening
