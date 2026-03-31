# SNMP Features, Training, and Correlation Roadmap

## Objective

Define a phased plan to improve:
- SNMP feature quality
- anomaly model training quality
- post-detection correlation with operational events

This document is intended to guide the next stage of the project after the current SNMP-only anomaly pipeline.

## Why This Document Exists

The current project already proves an important baseline:
- per-device rolling windows work
- an LSTM autoencoder can score SNMP windows
- CSV replay and Kafka live scoring share the same downstream pipeline

That is a good foundation, but it is still a narrow anomaly detector.

Today the project mainly answers:
- "Does this SNMP window look abnormal?"

It does not yet answer well:
- "Why did this happen?"
- "Which network event likely caused it?"
- "How can we reduce false positives from normal but unusual traffic changes?"

This roadmap closes that gap in phases.

## Current State

The current model uses these feature columns:
- `cpu`
- `memory`
- `in_octets`
- `out_octets`
- `errors`

Current sequence settings:
- `sequence_length = 10`
- one rolling window per device after warm-up

Current model:
- LSTM autoencoder
- trained mostly on synthetic SNMP-like data
- threshold derived from reconstruction error on normal training windows

Current strengths:
- simple and understandable
- good for pipeline validation
- suitable for first anomaly experiments

Current limitations:
- relies heavily on raw values instead of stronger derived features
- uses synthetic data patterns that are simpler than production telemetry
- does not model interface capacity or utilization
- does not yet use discards, packet counters, or interface state
- does not yet correlate anomalies with syslog, traps, flow summaries, or status events
- cannot yet explain likely root cause beyond "top error feature"

## Prerequisites

Before this roadmap can be followed effectively, the project should have a minimum set of data, schema, and evaluation foundations in place.

### Data Prerequisites

- stable `device_id` values across SNMP collection, training data, replay, and live scoring
- reliable `interface` identifiers for the primary `per-interface` production path
- timestamped sequential SNMP poll data with enough continuity to compute rates safely
- interface metadata where available, including capacity or speed and interface state
- at least one usable non-SNMP event source for later correlation work, such as syslog, traps, or interface state events

### Pipeline Prerequisites

- handling for counter resets, wraparound, polling gaps, and out-of-order records
- ability to compute rates using actual elapsed poll time between observations
- the same feature engineering behavior available in both offline dataset generation and Kafka live scoring
- support for `analysis_scope` labeling so `per-interface` and `per-device` fallback records do not get mixed silently
- versioned preprocessing artifacts so training and inference use the same scaler and transformations

### Training And Evaluation Prerequisites

- a repeatable time-based train/validation/test split process
- at least some known-normal or reviewable baseline periods for cleaner unsupervised training
- a reproducible evaluation workflow that reports false positives, detection delay, and per-interface performance
- baseline metrics saved before major feature or model changes so improvements can be measured honestly

### Correlation Prerequisites

- timestamps from SNMP and non-SNMP sources that are aligned well enough for time-window matching
- a normalized event schema or a clear path to create one
- enough event context to match by `device_id`, and by `interface` where available

### Fallback Readiness

- if some devices do not expose interface-level telemetry, the pipeline must support a labeled `per-device` fallback path
- fallback records should use `analysis_scope = per_device`
- fallback outputs should use `interface = null` or an equivalent explicit empty value
- fallback training and thresholding should remain separate from the main `per-interface` production baseline

### Minimum Practical Starting Point

At minimum, the roadmap can begin productively when the project has:
- SNMP data with `device_id`, timestamp, and enough counters to derive traffic and error rates
- interface-level identifiers for the devices intended for the main production path
- a reproducible offline training and replay workflow
- one event source that can later be normalized for correlation
- saved baseline evaluation results for comparison across phases

## Design Principles

The next version should follow these principles:

- Keep SNMP anomaly scoring as the core detector first.
- Improve features before replacing the model.
- Prefer derived rates and utilization over raw cumulative counters.
- Standardize the roadmap on `per-interface` analysis so feature engineering, training, and correlation stay aligned with production operations.
- Add explanation through event correlation before attempting full multi-source joint training.
- Preserve the current shared event-processing architecture where possible.
- Roll out changes in phases so accuracy, complexity, and operational risk can be evaluated separately.

## Primary Analysis Scope

This roadmap adopts one explicit production-first scope decision:
- primary anomaly scope = `per-interface`
- `per-device` handling is fallback compatibility only for datasets that do not yet expose interface-level telemetry

Why this matters:
- rate features, utilization, and discards are most actionable at interface level
- correlation output is easier to interpret when anomalies point to a concrete interface
- production triage is faster when alerts identify the affected interface directly
- training samples and thresholds should not mix interface-level and device-level semantics invisibly

Implementation note:
- every dataset, feature schema, model artifact, and anomaly output should carry `analysis_scope = per_interface` by default
- any temporary `per_device` fallback path should be labeled explicitly and kept outside the main production baseline

## Fallback Strategy For Limited Visibility Devices

Some production devices may not expose usable interface-level telemetry.
The roadmap should still support them, but without weakening the main `per-interface` design.

Fallback rule:
- if interface-level counters and state are available, use `per-interface` analysis
- if interface-level visibility is missing or incomplete, use a labeled `per-device` fallback path

Fallback expectations:
- fallback records must carry `analysis_scope = per_device`
- fallback anomaly outputs should use `interface = null` or an equivalent explicit empty value
- fallback training samples should be evaluated separately from the main `per-interface` production baseline
- fallback thresholding should be tuned independently because device-level variability differs from interface-level variability
- event correlation for fallback devices should match on `device_id` when no reliable interface key exists

Recommended fallback features:
- `cpu`
- `memory`
- total `in_rate`
- total `out_rate`
- total `error_rate`
- device-level health or status events

Operational goal:
- keep production coverage for limited-visibility devices
- avoid mixing weaker device-level semantics into the primary interface-first model

## What We Should Improve First

Highest-value improvements:
- derive better SNMP features
- retrain on those features
- tighten validation and thresholding
- correlate detected anomalies with nearby events

Lower-priority improvements for later:
- multi-source joint model training
- dynamic per-device adaptive models
- online retraining in stream

## Feature Engineering Roadmap

### Phase F1: Replace Raw Counter Emphasis With Derived Rates

#### Objective

Convert raw SNMP counters into more meaningful rate-based signals.

#### Why

Raw counters and absolute values are often weaker than change-over-time features.

Example:
- raw `in_octets` mostly reflects accumulated volume or scale
- `in_rate` better reflects traffic behavior between polls

#### Target Features

- `in_rate`
- `out_rate`
- `error_rate`
- optional `cpu_delta`
- optional `memory_delta`

#### Notes

- Compute deltas per `device_id` and `interface`.
- Convert deltas into rates using elapsed poll time, not just successive counter difference.
- Guard against poll jitter so delayed polls do not look like traffic spikes.
- Handle counter resets and wraparound safely.
- Drop or repair negative deltas caused by resets, restarts, or bad ordering.

#### Acceptance Criteria

- Derived rate features are created correctly from sequential SNMP data.
- Rate features are generated at interface granularity and remain keyed by `device_id` plus `interface`.
- Rate features are normalized by elapsed seconds between polls.
- Counter resets do not produce false spikes.
- Polling gaps and irregular intervals are handled explicitly and tested.
- Existing CSV replay and Kafka paths can carry the new feature set.

### Phase F2: Add Interface Capacity and Health Features

#### Objective

Capture whether traffic is high relative to link capacity and whether the interface itself is unhealthy.

#### Target Features

- `utilization_in_pct`
- `utilization_out_pct`
- `discard_rate_in`
- `discard_rate_out`
- `packet_rate_in`
- `packet_rate_out`
- `in_out_ratio`
- `interface_oper_status`
- `interface_admin_status`

#### Why

These features improve distinction between:
- busy but healthy links
- congested links
- failing links
- administratively shut links

#### Acceptance Criteria

- Utilization is calculated using interface speed rather than raw traffic alone.
- Discards and packet features are available in both offline and live paths.
- Interface state is preserved as a first-class production feature for anomaly explanation and optional training use.

### Phase F3: Add Contextual Rolling Features

#### Objective

Measure deviation from each interface's recent history, not only absolute values.

#### Target Features

- rolling mean
- rolling standard deviation
- z-score or standardized deviation
- short-term trend slope
- burst indicator

#### Why

Many anomalies are interface-relative:
- 40% utilization may be normal on one interface
- the same traffic level may be normal on one uplink and abnormal on another

#### Acceptance Criteria

- Features can be computed online from recent windows without future leakage.
- Interface-relative deviation features improve validation metrics against the F2/T2 baseline on the same time split.

## Training Roadmap

### Phase T1: Clean Training Data and Time-Based Evaluation

#### Objective

Train the model on more realistic normal behavior and evaluate it in a production-like way.

#### Improvements

- train only on known-normal periods where possible
- exclude maintenance windows and known incidents
- split train/test by time rather than random order
- track precision, recall, and false positive rate
- report metrics by `device_id` and `interface`

#### Why

If anomalous periods leak into the normal training set, the model learns them as acceptable behavior.

#### Acceptance Criteria

- Training and testing use clear time boundaries.
- Metrics are reported in addition to saved model artifacts.
- False positives are reviewed per interface, with device-level rollups as secondary reporting.
- A documented baseline run exists so later phases can be compared against the same evaluation slice.

### Phase T2: Improve Scaling and Transformations

#### Objective

Make preprocessing more stable for real telemetry distributions.

#### Candidates

- compare `MinMaxScaler` with `StandardScaler`
- compare `MinMaxScaler` with `RobustScaler`
- apply `log1p` on heavy-tailed traffic features before scaling

#### Why

Traffic rates and error bursts are often skewed and can dominate learning if scaling is too brittle.

#### Acceptance Criteria

- scaler choice is documented and justified by validation results
- feature distributions are checked before training
- large outliers do not collapse the useful range of normal samples
- the saved preprocessing artifact is versioned and reused consistently by offline and live scoring

### Phase T3: Tune Windowing and Threshold Strategy

#### Objective

Find a more reliable detection setup without overcomplicating the model.

#### Things to Tune

- `sequence_length`
- threshold rule
- per-interface vs global threshold
- percentile threshold vs `mean + K * std`

#### Why

Some anomalies are short spikes, others are slow drifts. One window length and one threshold may not fit all behaviors.

#### Acceptance Criteria

- threshold method is selected from measured validation results
- detection delay and false positives are reported
- chosen sequence length matches poll interval and operational need
- the final threshold strategy is documented with exact selection logic so inference and replay use the same rule

### Phase T4: Compare Baseline Models Before Major Complexity

#### Objective

Validate whether the current LSTM autoencoder remains the best practical choice.

#### Candidate Comparisons

- current LSTM autoencoder
- GRU autoencoder
- 1D CNN or temporal convolution autoencoder
- simpler baseline on engineered summary features, such as Isolation Forest

#### Why

A simpler model with stronger features can outperform a more complex model with weak features.

#### Acceptance Criteria

- comparisons are run on the same feature set and time split
- one baseline is selected for operational simplicity and accuracy
- the selected model is justified with both quality metrics and operational cost considerations

## Event Correlation Roadmap

### Phase C1: Add a Normalized Event Schema

#### Objective

Standardize non-SNMP evidence into one event format.

#### Candidate Sources

- syslog
- SNMP traps
- interface state events
- NetFlow summaries
- sFlow summaries

#### Common Fields

- `timestamp`
- `device_id`
- `interface`
- `analysis_scope`
- `source_type`
- `event_type`
- `severity`
- `message`
- `extra`

#### Why

A shared schema makes correlation simple even when raw sources are very different.

#### Acceptance Criteria

- at least one non-SNMP source can be normalized into the shared event format
- malformed events are rejected safely
- the normalized schema is versioned so downstream correlation logic can evolve safely

### Phase C2: Correlate Detected Anomalies With Nearby Events

#### Objective

Attach likely supporting evidence to each detected anomaly window.

#### Input

SNMP anomaly result with:
- `device_id`
- `interface`
- `window_start`
- `window_end`
- `top_error_feature`
- `predicted_anomaly`

#### Correlation Rule

For each anomaly:
- find events from the same `device_id`
- require the same `interface` when interface context exists in both anomaly and event records
- search within a configurable time buffer around the anomaly window
- rank matched events by proximity and relevance

#### Example Time Buffer

- from `window_start - 5 minutes`
- to `window_end + 5 minutes`

#### Example Relevance Hints

- traffic anomalies prefer flow summaries, link events, trap evidence
- CPU or memory anomalies prefer syslog and control-plane events
- error or discard anomalies prefer interface events and fault traps

#### Acceptance Criteria

- anomaly results can include a supporting evidence list
- evidence ranking is explainable and deterministic
- correlation behavior is evaluated on a small backtest set or reviewed incident sample using a metric such as precision at top-k evidence items

### Phase C3: Produce Enriched Anomaly Output

#### Objective

Move from bare anomaly scores to operator-friendly incident hints.

#### Desired Output Shape

- anomaly metadata
- schema version
- analysis scope
- top abnormal SNMP features
- correlated events
- likely cause summary
- evidence confidence or correlation score

#### Example Outcome

"Traffic anomaly detected on device `R1`; likely related to uplink state change based on `linkDown` trap and interface-down syslog."

#### Acceptance Criteria

- enriched results are saved in a stable schema
- operators can understand why the anomaly likely happened without manual log hunting
- enriched outputs are versioned so replay, live scoring, and downstream consumers stay aligned

## Important Things We Should Not Forget

These are easy to miss but materially affect accuracy and trustworthiness:

- Counter resets:
  raw counters can restart after device reboot or interface reset.

- Polling gaps:
  missing data must be handled explicitly or rates become misleading.

- Out-of-order events:
  both SNMP and external event streams may arrive late or out of order.

- Time synchronization:
  event correlation is weak if source timestamps are not aligned.

- Poll interval variability:
  rates should use actual elapsed time because fixed-interval assumptions break under jitter or delayed polling.

- Interface-level scope:
  this roadmap assumes production anomaly detection is interface-first, with device-level views used mainly for aggregation and fallback compatibility.

- Device role differences:
  routers, switches, firewalls, and servers may need separate baselines.

- Maintenance suppression:
  planned changes should not poison training data or alert volume.

- Seasonality:
  business-hour traffic and overnight traffic may require different baselines.

- Label quality:
  synthetic labels are fine for pipeline testing but not enough for production confidence.

- Explainability:
  operators will trust the system more if results include evidence, not just a score.

## Recommended Phase Order

Recommended implementation sequence:

1. Phase F1
2. Phase T1
3. Phase T2
4. Phase F2
5. Phase T3
6. Phase C1
7. Phase C2
8. Phase C3
9. Phase F3
10. Phase T4

This order keeps the highest-value improvements early:
- stronger SNMP features first
- cleaner training second
- explanation and correlation third

## Scope Boundaries

Included in this roadmap:
- stronger SNMP features
- improved offline training workflow
- validation and threshold improvements
- post-detection event correlation
- enriched anomaly outputs

Not included in the first implementation:
- online retraining in Kafka live mode
- full multi-source joint model training
- dynamic topic orchestration
- long-term incident management workflows

## Suggested Deliverables By Phase

- Phase F1:
  updated feature engineering module, derived-rate dataset support, and poll-interval-aware rate tests
- Phase T1:
  time-split evaluation notes, training metrics, and a documented baseline comparison run
- Phase F2:
  extended schema for interface health and utilization
- Phase C1:
  normalized and versioned non-SNMP event schema
- Phase C2:
  anomaly-to-event correlation module plus reviewed correlation backtest examples
- Phase C3:
  versioned enriched anomaly result export format

## Success Criteria

This roadmap is successful when:

- anomaly detection uses stronger SNMP-derived features than raw counters alone at interface granularity
- training and evaluation reflect realistic operating conditions
- false positives are reduced compared with the current baseline
- anomaly results explain not only what was abnormal, but also what likely caused it
- the system remains understandable enough to extend phase by phase

## Next Step

The best next implementation step is:
- Phase F1: derive `in_rate`, `out_rate`, and `error_rate`

That gives the project a stronger data foundation before larger model or correlation changes.
