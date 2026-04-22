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
- `P2` is completed for the synthetic dataset
- `F2` first extended-feature candidate has been generated and reviewed
- `F3` first contextual rolling-feature candidate has been generated and reviewed
- `F3` threshold tuning produced the official recommended rich-feature baseline
- the stable simple baseline remains `f1_baseline_v1`
- the official recommended rich-feature baseline is `f3_t3_seq10_p995_v1`
- the best `T3` review candidate so far is `t3_seq10_std3_5_v1`
- the stricter low-noise feature-rich candidate is `f3_t3_seq10_p999_v1`
- `P3` is completed for SNMP traps / interface state events as the first correlation source
- `C1` is completed for SNMP trap / interface-state event normalization using the `P3` schema
- `C2` is completed for deterministic matching between anomaly windows and normalized events
- `C3` is completed for deterministic explanation output from C2 correlated anomalies
- the next recommended work is to add Isolation Forest as a separate experimental model baseline under `T4`, not as an immediate replacement for the LSTM path
- any LSTM + Isolation Forest ensemble should wait until Isolation Forest has its own repeatable metrics report

Current recommended artifacts:

- official rich-feature baseline:
  `f3_t3_seq10_p995_v1`
- stable simple baseline:
  `f1_baseline_v1`
- low-noise rich-feature option:
  `f3_t3_seq10_p999_v1`

Current stable simple baseline:

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

Current rich-feature baseline:

- active feature profile:
  `f3`
- feature count:
  `35`
- sequence length:
  `10`
- model:
  LSTM autoencoder
- threshold rule:
  percentile `99.5`
- recommended artifact directory:
  `f3_t3_seq10_p995_v1`
- latest replay result:
  `29850` windows evaluated, `1816` predicted anomalies
- summary versus `f1_baseline_v1`:
  `+14` true positives, `-19` false positives, higher precision, higher recall, and lower false positive rate

## Why This Roadmap Exists

The current pipeline already answers this question well:

- "Does this SNMP window look abnormal?"

It does not yet answer these questions well:

- "Why did this happen?"
- "Is this a traffic spike, congestion issue, interface problem, or status change?"
- "Which nearby event likely explains the anomaly?"
- "How do we reduce false positives?"

This roadmap closes that gap in phases, so we can improve the system without losing the stable baseline.

## Model Boundary Decision

The LSTM autoencoder remains responsible for anomaly detection.

Current detection model:

- recommended artifact:
  `f3_t3_seq10_p995_v1`
- model type:
  LSTM autoencoder
- job:
  decide whether an SNMP sequence window is anomalous

`P3`, `C1`, `C2`, and `C3` do not require a new ML model initially.
They should run after anomaly detection as deterministic correlation and explanation logic.

Isolation Forest is a planned model-baseline experiment, but it should come after
`C3` so the current LSTM-based detection and explanation path is completed first.
It should initially be implemented as a separate artifact and evaluation path, not
as a replacement for `f3_t3_seq10_p995_v1`.

Simple boundary:

- LSTM:
  "this SNMP window is abnormal"
- correlation:
  "this nearby event may explain why"

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

#### Stage `P2`: Extended interface feature readiness

What `P2` was for:

- confirm which richer interface-centric fields are available before using them in `F2` and `F3`
- decide whether each stream can stay on the primary `per-interface` path or needs `per-device` fallback
- define the extended feature row and output schema for capacity, packet, discard, and state features

What we achieved:

- generated the interface capability matrix
- generated the extended feature schema note
- confirmed full synthetic dataset coverage for:
  interface speed, packet counters, discard counters, admin status, and oper status
- confirmed all `15` synthetic streams can use the primary `per-interface` path
- confirmed no synthetic streams currently require `per-device` fallback

Current evidence:

- capability matrix:
  `snmp_anomaly_detection/outputs/p2_interface_capability_matrix.json`
- extended feature schema:
  `snmp_anomaly_detection/outputs/p2_extended_feature_schema.json`

#### Phase `F2`: First extended interface feature candidate

What `F2` was for:

- add capacity-aware and interface-health features to distinguish busy links, congested links, failing links, and shut links

What we achieved:

- generated the first `F2` candidate artifact:
  `f2_candidate_v1`
- active candidate features now include:
  `utilization_in_pct`, `utilization_out_pct`, `discard_rate_in`, `discard_rate_out`,
  `packet_rate_in`, `packet_rate_out`, `in_out_ratio`, `interface_oper_status`,
  and `interface_admin_status`
- compared `f2_candidate_v1` against `f1_baseline_v1` with the repeatable `P1.1` workflow

Current `F2` candidate result:

- precision:
  `0.0890`, down from baseline `0.0923`
- recall:
  `0.6137`, up from baseline `0.6065`
- false positive rate:
  `0.0589`, up from baseline `0.0559`
- false positives:
  `1741`, up from baseline `1653`

Current recommendation:

- do not promote `f2_candidate_v1` over `f1_baseline_v1`
- keep `f1_baseline_v1` as the stable simple baseline
- use `F2` as the feature foundation for `F3` contextual rolling-feature work

#### Phase `F3`: First contextual rolling-feature candidate

What `F3` was for:

- add per-interface context so the model can see whether current traffic is unusual for that specific stream

What we achieved:

- generated the first `F3` candidate artifact:
  `f3_candidate_v1`
- added contextual features for key traffic and utilization signals:
  rolling mean, rolling standard deviation, z-score, trend, and a burst indicator
- trained and evaluated the candidate against the same synthetic baseline workflow
- compared `f3_candidate_v1` against `f1_baseline_v1` with `P1.1`

Current `F3` candidate result:

- precision:
  `0.0899`, down from baseline `0.0923`
- recall:
  `0.6823`, up from baseline `0.6065`
- false positive rate:
  `0.0647`, up from baseline `0.0559`
- false positives:
  `1913`, up from baseline `1653`
- recommendation:
  `review_tradeoff`

Current recommendation:

- do not promote `f3_candidate_v1` over `f1_baseline_v1`
- keep `f1_baseline_v1` as the stable simple baseline
- treat `F3` as a recall-improving candidate that needs stricter thresholding or feature ablation before promotion

#### Phase `F3/T3`: Feature-rich threshold tuning

What this work was for:

- keep the richer `F3` feature set but reduce the extra false positives from the first `F3` run

What we achieved:

- generated and reviewed three threshold-tuned `F3` candidates:
  `f3_t3_seq10_std3_5_v1`, `f3_t3_seq10_p995_v1`, and `f3_t3_seq10_p999_v1`
- compared all three against `f1_baseline_v1` with `P1.1`

Current result:

- `f3_t3_seq10_p995_v1` is the best balanced feature-rich candidate:
  precision `0.1002`, recall `0.6570`, false positive rate `0.0553`
- compared with `f1_baseline_v1`, it adds `14` true positives, reduces false positives by `19`, and keeps predicted anomalies nearly flat
- `f3_t3_seq10_p999_v1` is the best low-noise feature-rich candidate:
  precision `0.1075`, recall `0.6137`, false positive rate `0.0477`
- compared with `f1_baseline_v1`, it reduces false positives by `242` but only adds `2` true positives
- `f3_t3_seq10_std3_5_v1` preserves the strongest recall gain but still adds `86` false positives

Current recommendation:

- use `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline
- review `f3_t3_seq10_p999_v1` if lower alert noise is the stronger operational priority
- keep `f1_baseline_v1` as the stable simple baseline for historical comparison

### What Is Still Missing

Even after `F1`, `P2`, and the first `F2` and `F3` candidates, the project still has these gaps:

- real known-normal and maintenance labels are not yet available for training exclusion
- per-interface thresholding is not yet implemented
- real-data validation is still needed before treating `f3_t3_seq10_p995_v1` as production-ready beyond the synthetic dataset
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
- `P2`: completed for the synthetic dataset
- `F2`: first candidate generated and reviewed; not promoted over `f1_baseline_v1`
- `F3`: first candidate generated and reviewed; raw `F3` candidate not promoted
- `F3/T3`: threshold tuning completed; `f3_t3_seq10_p995_v1` is the official recommended rich-feature baseline, and `f3_t3_seq10_p999_v1` is the low-noise rich-feature option
- `P3`: completed; selected SNMP traps / interface state events and generated schema, timestamp alignment, and sample normalized event outputs
- `C1`: completed; normalized SNMP trap / interface-state samples into the `P3` shared schema and emits rejected-event records
- `C2`: completed; correlated anomaly windows with normalized `C1` events using a deterministic time/key/score policy
- `C3`: completed; added deterministic explanation output after `C2`
- `T4`: planned next; compare model baselines, including Isolation Forest, using the same feature set and time split

## What We Should Do Next

The next best step is:

- implement Isolation Forest as a separate `T4` experimental baseline and compare it against the LSTM reports
- keep `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline
- keep `f1_baseline_v1` as the stable simple baseline for historical comparison
- keep `f3_t3_seq10_p999_v1` as the low-noise rich-feature option
- only consider combined LSTM + Isolation Forest alerting after the standalone Isolation Forest metrics are saved and reviewed
- then validate explanations against real known-normal and maintenance-labeled data when available

Optional alternate next step:

- if the immediate priority is production hardening, validate `f3_t3_seq10_p995_v1` against real known-normal and maintenance-labeled data before moving to correlation

Why this is the right order:

- `F1` already gave us a stable feature baseline
- `P1` and `P1.1` now give us a repeatable evaluation and comparison foundation
- the first `T1` and `T2` candidates did not beat `f1_baseline_v1`
- `T3` has now been tested, and none of the first candidates clearly beats `f1_baseline_v1`
- `t3_seq10_std3_5_v1` is the best lower-noise review candidate, but it is still a tradeoff rather than a strict replacement
- `P2` confirmed the synthetic dataset has full source coverage for extended interface features
- the first `F2` candidate improved recall slightly but increased false positives, so it should not replace the validated baseline
- the first `F3` candidate improved recall more strongly but also increased false positives, so it should not replace the validated baseline
- threshold tuning fixed the main `F3` noise issue for `f3_t3_seq10_p995_v1`, which improved precision, recall, and false positive rate versus `f1_baseline_v1`
- the next meaningful gains are more likely to come from richer labels and richer features than from simple threshold tuning alone

In short:

- first strengthen evaluation
- then compare preprocessing choices
- then add artifact-to-artifact comparison support for repeatable experiment reviews
- then tune threshold and windowing
- then close out `P2` feature-readiness checks
- then add and review richer interface features with `F2`
- then add contextual rolling features with `F3`
- then review `F3` thresholding or feature ablation if recall-oriented behavior is desired
- then decide whether to promote a feature-rich `F3/T3` candidate
- then define `P3` event-source and schema requirements
- then normalize one event source with `C1`
- then correlate anomalies with nearby events with `C2`
- then add deterministic explanations with `C3`
- then compare model baselines with `T4`, starting with Isolation Forest as a standalone experiment

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

- completed for the synthetic dataset

What is now implemented:

- interface capability matrix generation is complete
- extended feature schema documentation is complete
- current synthetic data has full source-field coverage for every stream
- current live event and Kafka payload schemas include the source fields needed for `F2` feature derivation
- all `15` synthetic streams are recommended for the primary `per-interface` path
- no synthetic streams currently require `per-device` fallback

Current `P2` outputs:

- interface capability matrix:
  `snmp_anomaly_detection/outputs/p2_interface_capability_matrix.json`
- extended feature schema note:
  `snmp_anomaly_detection/outputs/p2_extended_feature_schema.json`

What `P2` confirmed:

- the synthetic dataset already includes packet counters
- the synthetic dataset already includes discard counters
- the synthetic dataset already includes interface speed
- the synthetic dataset already includes interface state fields
- the required `F2` live input additions list is empty for the current schema

### Stage `P3`: Before `C1`, `C2`, and `C3`

Goal:

- prepare the first usable non-SNMP event source for anomaly correlation

Why this stage matters:

- correlation quality depends heavily on timestamp alignment and matching keys
- the anomaly model can detect abnormal SNMP behavior, but event correlation needs separate evidence with compatible timestamps and identifiers
- `P3` keeps the correlation work deterministic and inspectable before any future root-cause model is considered

Prerequisites for `P3`:

- choose the first non-SNMP event source
- verify timestamp alignment quality between SNMP and that source
- define normalized event schema fields
- confirm whether interface identifiers are available in that event source

Recommended first event source:

- SNMP traps / interface state events

Why this is the recommended first source:

- it naturally maps to `device_id`
- it often maps to `interface`
- it explains common network anomaly causes such as link down, link up, oper down, admin down, link flaps, counter resets, device reboot, and high-error events

Acceptable first implementation data:

- real SNMP trap or interface-state event samples, if available
- synthetic trap/interface-state event samples generated from the existing synthetic dataset, if real samples are not yet available

Draft normalized event schema:

- `timestamp`
- `device_id`
- `interface`
- `analysis_scope`
- `source_type`
- `event_type`
- `severity`
- `message`
- `extra`

Timestamp rules:

- parse event timestamps with the same timestamp parser used for SNMP data
- assume the same timezone as the SNMP dataset unless the event source explicitly provides timezone
- normalize malformed or missing timestamps as rejected events
- record timestamp quality notes before correlation is implemented

Matching-key rules:

- prefer `device_id + interface`
- fall back to `device_id` only when interface is missing
- preserve `analysis_scope` so downstream output can distinguish `per_interface` from `per_device` evidence

Draft correlation window for later `C2`:

- look from `10` minutes before anomaly `window_end`
- through `5` minutes after anomaly `window_end`

Expected outputs:

- first event source selection
- normalized event schema draft
- timestamp alignment notes
- matching-key policy note
- first sample normalized events file

Current status:

- completed for the synthetic dataset

What was achieved:

- selected SNMP traps / interface state events as the first event source
- defined the first normalized event schema for deterministic correlation
- generated timestamp alignment notes using the same timestamp parser as SNMP data
- generated a synthetic sample normalized events file from existing interface-state and error-burst records
- confirmed sample events preserve `device_id`, `interface`, and `analysis_scope`

Recommended `P3` output files:

- `snmp_anomaly_detection/outputs/p3_event_source_selection.json`
- `snmp_anomaly_detection/outputs/p3_normalized_event_schema.json`
- `snmp_anomaly_detection/outputs/p3_timestamp_alignment_notes.json`
- `snmp_anomaly_detection/outputs/p3_sample_normalized_events.jsonl`

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

- richer fields needed for `F2` now exist in the dataset and are active in `F2` candidate artifacts, but they are not yet part of the validated default artifact

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

- first candidate generated and reviewed

What is now implemented:

- `F2` feature profile can train with extended interface features
- first `F2` candidate artifact exists:
  `f2_candidate_v1`
- `F2` candidate metrics were saved through the repeatable `P1` evaluation workflow
- `F2` candidate was compared against `f1_baseline_v1` through `P1.1`

Current `F2` candidate feature set:

- `cpu`
- `memory`
- `in_rate`
- `out_rate`
- `error_rate`
- `utilization_in_pct`
- `utilization_out_pct`
- `discard_rate_in`
- `discard_rate_out`
- `packet_rate_in`
- `packet_rate_out`
- `in_out_ratio`
- `interface_oper_status`
- `interface_admin_status`

Current `F2` comparison result:

- `f2_candidate_v1` improves recall from `0.6065` to `0.6137`
- precision drops from `0.0923` to `0.0890`
- false positive rate rises from `0.0559` to `0.0589`
- false positives rise from `1653` to `1741`

Current recommendation:

- keep `f1_baseline_v1` as the stable simple baseline
- do not promote `f2_candidate_v1` yet
- use the `F2` feature profile as the foundation for `F3`

What still remains:

- test whether contextual rolling features can make the extended feature set more useful
- review whether specific `F2` features should be excluded, transformed, or weighted differently
- keep using `f1_baseline_v1` as the comparison baseline until a candidate improves the measured tradeoff

Prerequisite before starting:

- completed: `P2`

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

- first candidate generated and reviewed

What is now implemented:

- first `F3` candidate artifact exists:
  `f3_candidate_v1`
- contextual features were added for:
  `in_rate`, `out_rate`, `error_rate`, `utilization_in_pct`, and `utilization_out_pct`
- each contextual base feature now has:
  rolling mean, rolling standard deviation, z-score, and trend
- a `burst_indicator` feature is included
- offline preprocessing and CSV replay both derive the contextual fields
- threshold-tuned `F3` candidates were generated for:
  `stddev 3.5`, `percentile 99.5`, and `percentile 99.9`

Current `F3` comparison result:

- `f3_candidate_v1` improves recall from `0.6065` to `0.6823`
- precision drops from `0.0923` to `0.0899`
- false positive rate rises from `0.0559` to `0.0647`
- false positives rise from `1653` to `1913`
- recommendation is `review_tradeoff`

Current `F3/T3` threshold-tuning result:

- `f3_t3_seq10_std3_5_v1`:
  precision `0.0980`, recall `0.6823`, false positive rate `0.0588`
- `f3_t3_seq10_p995_v1`:
  precision `0.1002`, recall `0.6570`, false positive rate `0.0553`
- `f3_t3_seq10_p999_v1`:
  precision `0.1075`, recall `0.6137`, false positive rate `0.0477`

Current recommendation:

- keep `f1_baseline_v1` as the stable simple baseline
- do not promote raw `f3_candidate_v1`
- use `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline
- use `f3_t3_seq10_p999_v1` as the stricter low-noise rich-feature option

Prerequisite before starting:

- completed: `P2`

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

- keep `f1_baseline_v1` as the stable simple baseline
- review `t3_seq10_std3_5_v1` only if lower alert noise is preferred over a small recall drop

What still remains:

- add per-interface threshold experiments if we continue `T3`
- document the preferred strict-review candidate separately from the validated default baseline

### Phase `T4`: Compare model baselines

Goal:

- confirm whether the LSTM autoencoder is still the best practical choice
- add Isolation Forest as a separate experimental baseline after `C3` is implemented

Candidate comparisons:

- current LSTM autoencoder
- GRU autoencoder
- temporal convolution or 1D CNN autoencoder
- simple baseline such as Isolation Forest on engineered summary features

Why this phase matters:

- stronger features plus a simpler model may be better operationally

Acceptance criteria:

- comparisons use the same feature set and time split
- Isolation Forest has its own artifact, scoring output, and metrics report before any ensemble logic is considered
- one baseline is selected using both quality and operational simplicity

Current status:

- planned after `C3`
- Isolation Forest is not yet implemented

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
- normalized events preserve `device_id`
- normalized events preserve `interface` when available
- parser behavior is deterministic and easy to inspect

Current status:

- completed for SNMP trap / interface-state event samples

Prerequisite before starting:

- complete `P3` done

Recommended first `C1` implementation:

- normalize SNMP trap / interface-state events into the `P3` shared schema

Expected behavior:

- read raw event samples from CSV or JSONL
- parse timestamps
- map source-specific event names into standard event types
- assign severity consistently
- emit rejected-event records or warnings for malformed rows
- save normalized JSONL output

Suggested standard event types:

- `interface_down`
- `interface_up`
- `admin_down`
- `oper_down`
- `link_flap`
- `counter_reset`
- `device_reboot`
- `high_error_rate`
- `unknown_event`

Expected output files:

- `snmp_anomaly_detection/outputs/c1_normalized_events.jsonl`
- `snmp_anomaly_detection/outputs/c1_rejected_events.jsonl`
- `snmp_anomaly_detection/outputs/c1_normalization_summary.json`

What was achieved:

- added deterministic CSV, JSON, and JSONL event normalization
- maps common source event names and SNMP trap names into the shared standard event types
- rejects malformed timestamps and missing `device_id` rows safely
- preserves `interface` when available and falls back to `analysis_scope=per_device` when missing
- generated the first `C1` normalized event output from the `P3` sample normalized events

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
- matching prefers exact `device_id + interface`
- matching falls back to `device_id` only when interface is unavailable
- correlation score or reason is deterministic

Current status:

- completed for the current anomaly-window output and `C1` normalized events

Prerequisite before starting:

- complete `P3`
- complete `C1`

Recommended first `C2` implementation:

- correlate anomaly windows from the current recommended rich-feature artifact:
  `f3_t3_seq10_p995_v1`
- use normalized events produced by `C1`
- match by event time near anomaly `window_end`

Initial correlation window:

- `10` minutes before anomaly `window_end`
- `5` minutes after anomaly `window_end`

Suggested deterministic ranking:

- same `device_id + interface` match ranks highest
- same `device_id` without interface ranks lower
- closer timestamp ranks higher
- higher severity ranks higher
- event types related to the top error feature rank higher when applicable

Expected output files:

- `snmp_anomaly_detection/outputs/c2_correlated_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c2_correlation_summary.json`

What was achieved:

- added deterministic anomaly-to-event matching around anomaly `window_end`
- uses the documented `10` minute lookback and `5` minute lookahead correlation window
- prefers exact `device_id + interface` matches and falls back to device-only event evidence when interface is unavailable
- ranks matches by match scope, timestamp proximity, severity, and top-feature/event-type relationship
- preserves unmatched anomalies with empty correlation context
- generated the first `C2` correlated anomaly-window output from `anomaly_windows.json` and `c1_normalized_events.jsonl`

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
- explanations include whether the anomaly had a matched event
- explanations include the top model error feature
- explanations avoid claiming certainty; use "likely related" wording

Current status:

- completed

What was achieved:

- added deterministic C3 explanation output from C2 correlated anomaly windows
- each anomaly window now includes a `c3_explanation` object
- explanations preserve raw C2 correlation evidence and avoid certainty claims
- explanations distinguish event-backed explanations from model-only explanations
- generated the first C3 outputs from the current C2 file:
  `1581` explained windows, `254` with correlated events, and `1327` model-only explanations

Prerequisite before starting:

- complete `P3`
- complete `C1`
- complete `C2`

Recommended first `C3` implementation:

- combine model output from `f3_t3_seq10_p995_v1` with `C2` correlated events
- add one explanation string per anomaly window
- preserve raw correlation evidence alongside the explanation

Example explanation:

- `Anomaly on dev_6 eth2 is likely related to interface_down event 3 minutes before the anomaly. Top anomaly feature: error_rate.`

Expected output files:

- `snmp_anomaly_detection/outputs/c3_explained_anomaly_windows.json`
- `snmp_anomaly_detection/outputs/c3_explanation_summary.json`

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
