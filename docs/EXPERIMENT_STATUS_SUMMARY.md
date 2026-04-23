# SNMP Experiment Status Summary

This document explains, in simple words, what we have tried so far and which model setup is the best choice right now.

## Recommended Choice Right Now

Recommended default model setup:

- use `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline

Why:

- it is feature-rich and improves precision, recall, and false positive rate versus `f1_baseline_v1`
- it adds `14` true positives while reducing false positives by `19`
- it keeps total predicted anomalies nearly flat:
  `1821` for `f1_baseline_v1` versus `1816` for `f3_t3_seq10_p995_v1`

Stable simple baseline:

- keep `f1_baseline_v1` as the historical/simple comparison baseline

Low-noise rich-feature option:

- use `f3_t3_seq10_p999_v1` if the priority is fewer false positives

Best `T3` review candidate if we want a stricter alert policy:

- `t3_seq10_std3_5_v1`

Why:

- it reduces false positives from `1653` to `1508`
- it improves precision from `0.0923` to `0.0975`
- the recall drop is small compared with the baseline:
  `0.6065` to `0.5884`

Simple decision:

- choose `f3_t3_seq10_p995_v1` for the best balanced rich-feature setup
- choose `f3_t3_seq10_p999_v1` if lower alert noise matters more than recall gain
- use `f1_baseline_v1` only when you need the simpler historical baseline

## What The Project Uses

Across these experiments, the main model type is still the same:

- model:
  LSTM autoencoder
- stable simple baseline features:
  `cpu`, `memory`, `in_rate`, `out_rate`, `error_rate`
- official rich-feature baseline:
  `f3_t3_seq10_p995_v1`
- official rich-feature profile:
  `F3`, with `35` features
- dataset:
  synthetic SNMP dataset with interface-level rolling windows

What changed across experiments:

- training and evaluation rules
- scaler and feature transformation choices
- threshold strategy
- sequence length

## Baseline: `F1`

Artifact:

- `f1_baseline_v1`

What it is:

- the first stable derived-rate baseline
- uses the current default feature set and default threshold style
- acts as the reference point for all later comparisons

Overall metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f1_baseline_v1` | `0.0923` | `0.6065` | `0.0559` | `1653` | `168` |

Simple reading:

- catches a useful number of anomalies
- still has too many false positives
- remains the stable simple baseline, but is no longer the official recommended rich-feature setup

## Official Rich-Feature Baseline: `f3_t3_seq10_p995_v1`

Artifact:

- `f3_t3_seq10_p995_v1`

What it is:

- the best balanced rich-feature model so far
- uses the `F3` feature profile with interface health, utilization, packet/discard rates, rolling context, z-scores, trends, and burst indication
- uses sequence length `10`
- uses percentile threshold `99.5`

Overall metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f3_t3_seq10_p995_v1` | `0.1002` | `0.6570` | `0.0553` | `1634` | `182` |

Compared with `f1_baseline_v1`:

- precision improved from `0.0923` to `0.1002`
- recall improved from `0.6065` to `0.6570`
- false positive rate improved from `0.0559` to `0.0553`
- false positives decreased from `1653` to `1634`
- true positives increased from `168` to `182`

Decision:

- make this the official recommended rich-feature baseline

## Low-Noise Rich-Feature Option: `f3_t3_seq10_p999_v1`

Artifact:

- `f3_t3_seq10_p999_v1`

What it is:

- the stricter rich-feature option when fewer false positives matter most
- uses the same `F3` feature profile, but with percentile threshold `99.9`

Overall metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f3_t3_seq10_p999_v1` | `0.1075` | `0.6137` | `0.0477` | `1411` | `170` |

Compared with `f1_baseline_v1`:

- precision improved from `0.0923` to `0.1075`
- recall improved slightly from `0.6065` to `0.6137`
- false positives dropped from `1653` to `1411`

Decision:

- use this when the priority is lowest alert noise, not maximum recall

## `T1`: Cleaner Training And Time-Based Evaluation

Artifact:

- `t1_candidate_v1`

What we changed:

- cleaner time-based preprocessing and evaluation flow
- clearer split handling and saved metadata

Why we tested it:

- to make training and evaluation closer to real operations

Overall metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t1_candidate_v1` | `0.0896` | `0.6029` | `0.0574` | `1697` | `167` |

Result in simple words:

- very close to baseline
- slightly worse precision
- slightly worse recall
- slightly worse false positive rate

Decision:

- do not promote over `f1_baseline_v1`

## `T2`: Scaling And Transformation Experiments

These experiments kept the same general model, but changed preprocessing.

### `t2_standard_v1`

What it tested:

- `StandardScaler` instead of the baseline scaling choice

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_standard_v1` | `0.0843` | `0.5848` | `0.0595` | `1760` | `162` |

Simple result:

- worse than baseline

### `t2_robust_v1`

What it tested:

- `RobustScaler` for heavier-tailed traffic behavior

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_robust_v1` | `0.0808` | `0.5596` | `0.0596` | `1764` | `155` |

Simple result:

- worse than baseline

### `t2_log1p_standard_v1`

What it tested:

- `log1p` on heavy-tailed rate features plus `StandardScaler`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_log1p_standard_v1` | `0.0765` | `0.5704` | `0.0645` | `1907` | `158` |

Simple result:

- worse than baseline

### `t2_log1p_robust_v1`

What it tested:

- `log1p` on heavy-tailed rate features plus `RobustScaler`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t2_log1p_robust_v1` | `0.0819` | `0.6029` | `0.0633` | `1871` | `167` |

Simple result:

- recall stayed close to baseline
- false positives got worse
- still not a better default

Decision for `T2`:

- none of the first `T2` candidates beat `f1_baseline_v1`

## `T3`: Threshold And Windowing Experiments

These experiments tested whether changing the threshold rule or sequence length could improve the alert tradeoff.

### Threshold-only experiments at sequence length `10`

#### `t3_seq10_std2_5_v1`

What it tested:

- same window size as baseline
- looser threshold:
  `mean + 2.5 * std`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq10_std2_5_v1` | `0.0843` | `0.6715` | `0.0683` | `2021` | `186` |

Simple result:

- catches more anomalies
- creates too many extra false alarms
- interesting for recall, but too noisy as a default

#### `t3_seq10_std3_5_v1`

What it tested:

- same window size as baseline
- stricter threshold:
  `mean + 3.5 * std`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq10_std3_5_v1` | `0.0975` | `0.5884` | `0.0510` | `1508` | `163` |

Simple result:

- fewer false positives than baseline
- better precision than baseline
- slightly lower recall than baseline

Decision:

- best `T3` tradeoff candidate so far
- good review option if we want cleaner alerts

#### `t3_seq10_p995_v1`

What it tested:

- same window size as baseline
- percentile threshold at `99.5`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq10_p995_v1` | `0.0941` | `0.5921` | `0.0534` | `1578` | `164` |

Simple result:

- slightly fewer false alarms than baseline
- slightly better precision than baseline
- slightly lower recall than baseline

Decision:

- useful conservative option
- not as strong as `t3_seq10_std3_5_v1`

#### `t3_seq10_p999_v1`

What it tested:

- same window size as baseline
- stricter percentile threshold at `99.9`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq10_p999_v1` | `0.1013` | `0.5668` | `0.0471` | `1393` | `157` |

Simple result:

- cleanest threshold-only candidate
- strongest precision among the tested threshold variants
- lower recall than baseline and lower than `t3_seq10_std3_5_v1`

Decision:

- strong option if low alert noise matters most
- too strict to replace the baseline by default

### Sequence-length experiments

#### `t3_seq_8_std3_v1`

What it tested:

- shorter sequence length:
  `8`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq_8_std3_v1` | `0.0113` | `0.9531` | `0.7838` | `23179` | `264` |

Simple result:

- catches almost everything
- flags almost everything too
- unusably noisy

Decision:

- reject

#### `t3_seq_12_std3_v1`

What it tested:

- longer sequence length:
  `12`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq_12_std3_v1` | `0.0477` | `0.7437` | `0.1390` | `4110` | `206` |

Simple result:

- recall improves
- false positives rise far too much

Decision:

- reject as default

#### `t3_seq_15_std3_v1`

What it tested:

- longer sequence length:
  `15`

Metrics:

| Artifact | Precision | Recall | False Positive Rate | False Positives | True Positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `t3_seq_15_std3_v1` | `0.0256` | `0.8087` | `0.2879` | `8515` | `224` |

Simple result:

- recall improves a lot
- false positives become far too high

Decision:

- reject as default

## Final Ranking In Simple Words

### Best balanced default

1. `f1_baseline_v1`

Why:

- still the most balanced overall
- no tested candidate clearly beats it across precision, recall, and false positive rate

### Best lower-noise review candidate

2. `t3_seq10_std3_5_v1`

Why:

- better precision than baseline
- lower false positive rate than baseline
- only a small recall drop

### Best strict conservative option

3. `t3_seq10_p999_v1`

Why:

- lowest false positive rate among the useful `T3` threshold candidates
- highest precision among those threshold-only candidates
- recall drop is more noticeable than `t3_seq10_std3_5_v1`

### Candidates to reject for default use

- `t3_seq_8_std3_v1`
- `t3_seq_12_std3_v1`
- `t3_seq_15_std3_v1`

Why:

- they produce too many false positives
- they are not operationally practical as default alerting setups

## Suggested Next Practical Step

Use this path:

- treat `f3_t3_seq10_p995_v1` as the official recommended rich-feature baseline
- keep `f1_baseline_v1` as the stable simple comparison baseline
- start `P3` event-source selection for correlation

If alert-noise reduction is the main priority, use:

- `f3_t3_seq10_p999_v1`
