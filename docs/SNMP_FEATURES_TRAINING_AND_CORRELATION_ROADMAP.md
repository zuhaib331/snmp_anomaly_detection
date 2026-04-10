# SNMP Features, Training, and Correlation Roadmap

## Purpose

This document is the working roadmap for the SNMP anomaly project.
It is written to answer four simple questions:

- where we are now
- what we have already achieved
- what must be done before the next phase
- what we should do next

The goal is to improve three areas in a controlled order:

- better SNMP features
- better model training and evaluation
- better correlation with operational events

## One-Page Summary

Current project status in plain language:

- the SNMP anomaly pipeline works end to end
- interface-scoped rolling windows are working
- CSV replay and Kafka live scoring share the same downstream logic
- `P0` is completed
- `F1` is completed
- `P1` is completed for the synthetic baseline
- `P1.1` is completed for the first synthetic candidate comparison set
- `T1` first candidate has been reviewed against the baseline
- `T2` first scaler and transformation comparisons have been generated
- `T3` first threshold and windowing comparisons have been generated
- the validated artifact set is `f1_baseline_v1`
- the best `T3` review candidate so far is `t3_seq10_std3_5_v1`
- the next recommended work is to complete `P2`, then continue to `F2` and `F3`

Current validated baseline:

- active feature set:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`
- sequence length:
  `10`
- model:
  LSTM autoencoder
- threshold source:
  reconstruction error on normal training windows
- validated artifact directory:
  `f1_baseline_v1`
- validated threshold:
  `0.014308651676401496`
- latest replay result:
  `29850` windows evaluated, `1821` predicted anomalies

## Why This Roadmap Exists

The current pipeline already answers this question well:

- "Does this SNMP window look abnormal?"

It does not yet answer these questions well:

- "Why did this happen?"
- "Is this a traffic spike, congestion issue, interface problem, or status change?"
- "Which nearby event likely explains the anomaly?"
- "How do we reduce false positives?"

This roadmap closes that gap in phases, so we can improve the system without losing the stable baseline.

## Current Scope Decision

The project now follows one primary production direction:

- main scope = `per-interface`
- fallback scope = `per-device` only when interface telemetry is not available

Why this decision matters:

- traffic rates, utilization, discards, and interface state are more meaningful at interface level
- alerts are easier to triage when we know the affected interface
- correlation with future event sources is easier when anomalies already point to a specific interface
- we should not silently mix interface-level and device-level behavior into the same baseline

Rule for the roadmap:

- keep `per-interface` as the main path
- keep `per-device` as a clearly labeled fallback path

## What We Have Already Achieved

### Completed Work

#### Stage `P0`: Synthetic baseline readiness

What `P0` was for:

- verify the dataset is usable for interface-level anomaly work
- audit sequential quality before derived-rate features

What we achieved:

- dataset inventory tooling exists
- data quality audit tooling exists
- baseline synthetic dataset slice was prepared
- cumulative counters were corrected so default synthetic data no longer produces negative deltas
- richer synthetic fields were added:
  packet counters, discard counters, interface speed, interface status, anomaly metadata, counter reset flags

Current evidence:

- outputs are saved under `snmp_anomaly_detection/outputs/`
- dataset quality is auditable before training

#### Phase `F1`: Derived-rate feature baseline

What `F1` was for:

- move from raw counter emphasis to rate-based signals

What we achieved:

- preprocessing derives:
  `in_rate`, `out_rate`, `error_rate`
- derived rates are computed per `device_id + interface`
- elapsed poll time is used in the rate calculations
- reset-aware logic is used to avoid false spikes
- offline preprocessing, CSV replay, and Kafka live scoring use the same core rate logic
- outputs now carry `interface` and `stream_id`
- the active model trains on:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`

Validated baseline produced by `F1`:

- artifact directory:
  `f1_baseline_v1`
- threshold:
  `0.014308651676401496`
- replay summary:
  `29850` windows, `1821` predicted anomalies

#### Stage `P1.1`: Artifact comparison reporting

What `P1.1` was for:

- compare candidate artifacts against the validated baseline using saved `P1` reports
- make improvement or regression review repeatable

What we achieved:

- added `compare-artifacts` as a pipeline step
- generated a comparison report for `t1_candidate_v1` and the first T2 candidate set
- confirmed that none of the first candidates should replace `f1_baseline_v1`
- surfaced and resolved metadata consistency warnings for T2 scaler and `log1p` tracking

### What Is Still Missing

Even after `F1`, the project still has these gaps:

- real known-normal and maintenance labels are not yet available for training exclusion
- per-interface thresholding is not yet implemented
- richer interface health features are present in data but not yet active in the model
- contextual rolling features are not yet added
- correlation with non-SNMP events is not yet implemented

## Roadmap Status

This is the simplest view of the roadmap:

- `P0`: completed
- `F1`: completed
- `P1`: completed
- `P1.1`: completed
- `T1`: first synthetic candidate reviewed; real known-normal/maintenance label support still pending
- `T2`: first comparison completed; metadata propagation fixed
- `T3`: first comparison completed; keep `f1_baseline_v1`, review `t3_seq10_std3_5_v1` if lower alert noise is preferred
- `P2`: in progress
- `F2`: pending
- `F3`: pending
- `P3`: pending
- `C1`: pending
- `C2`: pending
- `C3`: pending

## What We Should Do Next

The next best step is:

- complete `P2`
- then start `F2`
- then continue to `F3`

Why this is the right order:

- `F1` already gave us a stable feature baseline
- `P1` and `P1.1` now give us a repeatable evaluation and comparison foundation
- the first `T1` and `T2` candidates did not beat `f1_baseline_v1`
- `T3` has now been tested, and none of the first candidates clearly beats `f1_baseline_v1`
- `t3_seq10_std3_5_v1` is the best lower-noise review candidate, but it is still a tradeoff rather than a strict replacement
- the next meaningful gains are more likely to come from richer labels and richer features than from simple threshold tuning alone

In short:

- first strengthen evaluation
- then compare preprocessing choices
- then add artifact-to-artifact comparison support for repeatable experiment reviews
- then tune threshold and windowing
- then complete `P2`
- then add richer features
- then add event correlation

## Phase Order At A Glance

The roadmap is organized into four kinds of work:

- prerequisite stages:
  `P0`, `P1`, `P2`, `P3`
- feature phases:
  `F1`, `F2`, `F3`
- training phases:
  `T1`, `T2`, `T3`, `T4`
- correlation phases:
  `C1`, `C2`, `C3`

Simple order:

1. prepare baseline data and checks
2. improve features
3. improve training and evaluation
4. improve explanation by event correlation

## Prerequisites By Stage

The prerequisite stages are not the same as feature phases.
They are the preparation steps that make later phases safe and measurable.

### Stage `P0`: Before `F1`

Goal:

- make sure the synthetic dataset is trustworthy enough for derived-rate work

Prerequisites for `P0`:

- stable `device_id`, `timestamp`, and `interface`
- enough counters to derive rate features
- enough continuity to inspect gaps, duplicates, and ordering problems

Expected outputs:

- dataset inventory
- data quality audit
- baseline dataset selection

Current status:

- completed

What was achieved:

- inventory and audit tooling were added
- baseline synthetic slice was frozen for comparison
- cumulative counter behavior was improved
- richer interface-related fields were added to the synthetic dataset

### Stage `P1`: Before `T1` and `T2`

Goal:

- make evaluation reproducible before we change training behavior or scaler choices

Why this stage matters:

- `T1` and `T2` depend on trustworthy measurement
- without this stage, we cannot compare later changes honestly

Prerequisites for `P1`:

- define a repeatable time-based train/validation/test split
- identify known-normal periods, suspicious periods, and maintenance windows where possible
- define the baseline evaluation workflow
- save the current baseline metrics
- confirm versioned artifact usage for scaler, training arrays, model, and metadata

Expected outputs:

- written time-split definition
- baseline metrics report
- named baseline artifact usage confirmation

Current status:

- completed for the synthetic baseline

What we achieved in `P1`:

- versioned artifact directory exists:
  `f1_baseline_v1`
- repeatable time split was defined and saved
- baseline metrics report was generated and saved
- artifact usage for scaler, arrays, model, and metadata is now recorded in the report
- replay outputs are now tied to a documented baseline evaluation workflow

Current `P1` outputs:

- time split file:
  `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_time_split.json`
- baseline metrics file:
  `snmp_anomaly_detection/outputs/f1_baseline_v1_p1_baseline_metrics.json`

Current saved time split:

- train:
  `2025-01-01 00:00:00` to `2025-01-05 20:35:00`
- validation:
  `2025-01-05 20:40:00` to `2025-01-06 21:35:00`
- test:
  `2025-01-06 21:40:00` to `2025-01-07 22:35:00`

Current saved baseline metrics summary:

- windows evaluated:
  `29850`
- actual anomaly windows:
  `277`
- predicted anomaly windows:
  `1821`
- precision:
  `0.0923`
- recall:
  `0.6065`
- false positive rate:
  `0.0559`

Follow-on improvement completed after `P1`:

- `P1.1` comparison mode was added for artifact-to-artifact evaluation
- it compares `f1_baseline_v1` against candidate artifacts using saved `P1` metrics reports
- it reports overall metric deltas, split-wise deltas, and saved top-interface false-positive highlights
- it also flags artifact metadata consistency issues between model metadata and preprocessing metadata

Current `P1.1` output:

- comparison file:
  `snmp_anomaly_detection/outputs/f1_baseline_v1_vs_5_candidates_p1_1_comparison.json`
- command:
  `python3 main.py compare-artifacts --baseline-artifact-dir-name f1_baseline_v1 --candidate-artifact-dir-name t1_candidate_v1 --candidate-artifact-dir-name t2_standard_v1 --candidate-artifact-dir-name t2_robust_v1 --candidate-artifact-dir-name t2_log1p_standard_v1 --candidate-artifact-dir-name t2_log1p_robust_v1`
- current result:
  all compared candidates recommend `keep_baseline`

### Stage `P2`: Before `F2` and `F3`

Goal:

- confirm which richer interface-centric features are truly available and safe to use

Why this stage matters:

- `F2` and `F3` should be based on confirmed data availability, not assumptions

Prerequisites for `P2`:

- confirm availability of interface speed or capacity
- confirm availability of packet counters
- confirm availability of discard counters
- confirm availability of admin and oper status
- document which devices support full `per-interface` feature coverage
- document which devices need `per-device` fallback
- define the schema fields needed for extended interface-level feature rows and outputs

Expected outputs:

- interface capability matrix
- extended feature schema note

Current status:

- in progress

What is now implemented:

- preprocessing supports configurable scaler selection
- optional `log1p` preprocessing can be applied to selected heavy-tailed features
- preprocessing choice is saved into artifact metadata for repeatable experiments

Recommended first comparisons:

- `minmax`
- `standard`
- `robust`
- `log1p + standard`
- `log1p + robust`

What is already in place to help `P2`:

- the synthetic dataset already includes packet counters
- the synthetic dataset already includes discard counters
- the synthetic dataset already includes interface speed
- the synthetic dataset already includes interface state fields

### Stage `P3`: Before `C1`, `C2`, and `C3`

Goal:

- prepare the first usable non-SNMP event source for anomaly correlation

Why this stage matters:

- correlation quality depends heavily on timestamp alignment and matching keys

Prerequisites for `P3`:

- choose the first non-SNMP event source
- verify timestamp alignment quality between SNMP and that source
- define normalized event schema fields
- confirm whether interface identifiers are available in that event source

Expected outputs:

- first event source selection
- normalized event schema draft
- timestamp alignment notes

Current status:

- pending

## Feature Roadmap

### Phase `F1`: Derived rates

Goal:

- convert cumulative counters into rate-based behavior signals

Target features:

- `in_rate`
- `out_rate`
- `error_rate`
- optional `cpu_delta`
- optional `memory_delta`

Acceptance criteria:

- rate features are computed correctly from sequential data
- rate features stay keyed by `device_id + interface`
- elapsed seconds are used in normalization
- counter resets do not create false spikes
- polling gaps and irregular intervals are handled explicitly
- CSV replay and Kafka paths carry the same feature logic

Current status:

- completed

What we achieved:

- `in_rate`, `out_rate`, and `error_rate` are implemented
- the active training baseline uses the new rate features
- replay and live scoring both use reset-aware elapsed-time logic

Important note:

- richer fields needed for `F2` already exist in the dataset, but they are not active model features yet

### Phase `F2`: Interface capacity and health features

Goal:

- tell the difference between busy links, congested links, failing links, and shut links

Target features:

- `utilization_in_pct`
- `utilization_out_pct`
- `discard_rate_in`
- `discard_rate_out`
- `packet_rate_in`
- `packet_rate_out`
- `in_out_ratio`
- `interface_oper_status`
- `interface_admin_status`

Why this phase matters:

- raw rate alone cannot explain whether a link is healthy relative to capacity
- discards and state changes provide stronger operational meaning

Acceptance criteria:

- utilization is calculated using interface speed
- discard and packet features work in offline and live paths
- interface state is preserved as a first-class feature for explanation and optional training use

Current status:

- pending

Prerequisite before starting:

- complete `P2`

### Phase `F3`: Contextual rolling features

Goal:

- measure whether current behavior is unusual for that specific interface

Target features:

- rolling mean
- rolling standard deviation
- z-score or standardized deviation
- short-term trend slope
- burst indicator

Why this phase matters:

- a value can be normal for one interface and abnormal for another

Acceptance criteria:

- features can be computed online without future leakage
- they improve validation results against the same baseline used for `F2` and `T2`

Current status:

- pending

Prerequisite before starting:

- complete `P2`

## Training Roadmap

### Phase `T1`: Clean training data and time-based evaluation

Goal:

- make training and evaluation closer to real operational conditions

Planned improvements:

- train on known-normal periods where possible
- exclude suspicious periods and maintenance windows where possible
- split train and test by time instead of only by row order
- report metrics in a repeatable way
- report false positives by `device_id` and `interface`

Why this phase matters:

- if anomalous behavior leaks into training, the model learns it as normal
- if evaluation is weak, later improvements cannot be trusted

Acceptance criteria:

- training and testing use clear time boundaries
- metrics are reported and saved
- false positives are reviewed per interface
- a documented baseline run exists for future comparison

Current status:

- first synthetic candidate reviewed; still in progress for real known-normal and maintenance labeling

What is now implemented:

- preprocessing now builds a repeatable time split before sequence generation
- the scaler is fit on train-period normal data only
- saved `X_train` and `X_test` arrays now follow explicit time boundaries instead of row-order slicing
- preprocessing metadata is saved with split boundaries and row counts for the selected artifact
- `t1_candidate_v1` exists and has been compared against `f1_baseline_v1` using `P1.1`

Current `T1` comparison result:

- `t1_candidate_v1` did not beat `f1_baseline_v1`
- overall precision delta:
  `-0.0027`
- overall recall delta:
  `-0.0036`
- overall false positive rate delta:
  `+0.0015`
- current recommendation:
  keep `f1_baseline_v1` as the validated baseline

What still remains:

- use real known-normal and maintenance labels when available instead of synthetic anomaly-only filtering
- rerun candidate training when richer operational labels become available

Prerequisite before starting:

- completed

### Phase `T2`: Improve scaling and transformations

Goal:

- make preprocessing more stable for real traffic distributions

Candidate comparisons:

- `MinMaxScaler` vs `StandardScaler`
- `MinMaxScaler` vs `RobustScaler`
- `log1p` on heavy-tailed traffic features

Why this phase matters:

- traffic and error bursts are often skewed
- poor scaling can hide useful variation or overreact to outliers

Acceptance criteria:

- scaler choice is documented and justified by validation results
- feature distributions are checked before training
- outliers do not collapse the useful range of normal samples
- saved preprocessing artifacts are versioned and reused consistently

Current status:

- in progress

What is now implemented:

- first candidate artifacts were generated for:
  `standard`, `robust`, `log1p + standard`, and `log1p + robust`
- these candidates were compared against `f1_baseline_v1` using `P1.1`
- none of the first T2 candidates should be promoted over `f1_baseline_v1`
- training metadata propagation now records scaler and `log1p` choices from preprocessing metadata
- current T2 `model_metadata.json` files now match their saved preprocessing metadata

Current T2 comparison result:

- `t2_standard_v1`:
  precision delta `-0.0080`, recall delta `-0.0217`, false positive rate delta `+0.0036`
- `t2_robust_v1`:
  precision delta `-0.0115`, recall delta `-0.0469`, false positive rate delta `+0.0038`
- `t2_log1p_standard_v1`:
  precision delta `-0.0157`, recall delta `-0.0361`, false positive rate delta `+0.0086`
- `t2_log1p_robust_v1`:
  precision delta `-0.0103`, recall delta `-0.0036`, false positive rate delta `+0.0074`

What still remains:

- keep `f1_baseline_v1` as the validated baseline unless a future candidate improves the measured tradeoff
- move to `T3` threshold and windowing experiments

Prerequisite before starting:

- completed

### Phase `T3`: Tune windowing and threshold strategy

Goal:

- improve detection reliability without unnecessary complexity

Things to tune:

- `sequence_length`
- threshold rule
- global threshold vs per-interface threshold
- percentile threshold vs `mean + K * std`

Acceptance criteria:

- threshold method is selected from measured results
- detection delay and false positives are reported
- chosen sequence length fits the poll interval and operational need
- final threshold logic is documented clearly for inference and replay

Current status:

- first comparison completed

What is now implemented:

- CLI support now exists for:
  `--sequence-length`
- training CLI now supports:
  `--threshold-mode`, `--threshold-std-multiplier`, and `--threshold-percentile`
- replay, evaluation, and Kafka inference now load saved artifact feature settings automatically
- the first `T3` experiment set was run against `f1_baseline_v1`

First `T3` candidates reviewed:

- `t3_seq10_std2_5_v1`
- `t3_seq10_std3_5_v1`
- `t3_seq10_p995_v1`
- `t3_seq10_p999_v1`
- `t3_seq_8_std3_v1`
- `t3_seq_12_std3_v1`
- `t3_seq_15_std3_v1`

Current result in simple words:

- `t3_seq10_std3_5_v1` is the best lower-noise `T3` candidate so far
- `t3_seq10_p999_v1` is the strictest useful threshold-only candidate
- `t3_seq10_std2_5_v1` improves recall but creates too many extra false positives
- sequence-length variants `8`, `12`, and `15` were too noisy to promote
- no tested `T3` candidate clearly beats `f1_baseline_v1` across precision, recall, and false positive rate together

Current recommendation:

- keep `f1_baseline_v1` as the validated default artifact
- review `t3_seq10_std3_5_v1` only if lower alert noise is preferred over a small recall drop

What still remains:

- add per-interface threshold experiments if we continue `T3`
- document the preferred strict-review candidate separately from the validated default baseline

### Phase `T4`: Compare model baselines

Goal:

- confirm whether the LSTM autoencoder is still the best practical choice

Candidate comparisons:

- current LSTM autoencoder
- GRU autoencoder
- temporal convolution or 1D CNN autoencoder
- simple baseline such as Isolation Forest on engineered summary features

Why this phase matters:

- stronger features plus a simpler model may be better operationally

Acceptance criteria:

- comparisons use the same feature set and time split
- one baseline is selected using both quality and operational simplicity

Current status:

- pending

## Correlation Roadmap

### Phase `C1`: Normalize one non-SNMP event source

Goal:

- standardize external evidence into one event format

Candidate sources:

- syslog
- SNMP traps
- interface state events
- NetFlow summaries
- sFlow summaries

Common fields:

- `timestamp`
- `device_id`
- `interface`
- `analysis_scope`
- `source_type`
- `event_type`
- `severity`
- `message`
- `extra`

Acceptance criteria:

- at least one source can be normalized into the shared schema
- malformed events are rejected safely
- schema versioning is defined

Current status:

- pending

Prerequisite before starting:

- complete `P3`

### Phase `C2`: Correlate anomalies with nearby events

Goal:

- enrich anomaly output with likely nearby operational context

Planned behavior:

- match anomalies with nearby normalized events
- match by time window
- match by `device_id`
- match by `interface` when available

Acceptance criteria:

- anomaly outputs can include correlated events
- the correlation window is documented
- unmatched anomalies are still handled safely

Current status:

- pending

Prerequisite before starting:

- complete `P3`

### Phase `C3`: Add basic explanation output

Goal:

- move from "abnormal window" toward "likely explanation"

Planned behavior:

- combine top error feature with nearby correlated events
- produce a short explanation summary
- support interface-level triage first

Acceptance criteria:

- anomaly outputs include a basic explanation field
- explanation logic is deterministic and easy to inspect

Current status:

- pending

Prerequisite before starting:

- complete `P3`

## Minimum Practical Starting Point

The roadmap can move productively when we have:

- SNMP data with `device_id`, `timestamp`, and enough counters to derive traffic and error rates
- interface-level identifiers for the main production path
- a reproducible offline training and replay workflow
- one event source that can later be normalized for correlation
- saved baseline evaluation results for comparison

## Working Rule For Future Updates

Whenever a phase changes state, update this document in four places:

- `Roadmap Status`
- the relevant stage or phase section
- `What We Have Already Achieved`
- `What We Should Do Next`

This will keep the roadmap easy to follow for both implementation and review.
