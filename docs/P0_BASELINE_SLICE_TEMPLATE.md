# P0 Baseline Slice Definition Template

Use this template to define the fixed baseline dataset slice for Stage `P0`.
The goal is to freeze one comparison slice before feature or model improvements begin.

## 1. Baseline Identity

- baseline name:
- owner:
- date created:
- artifact directory name:
- source dataset path:

## 2. Scope

- primary analysis scope: `per_interface` or `per_device`
- fallback devices included: yes or no
- target device types:
- target vendors:

## 3. Included Time Range

- training start:
- training end:
- validation start:
- validation end:
- test start:
- test end:

## 4. Excluded Time Ranges

List any windows that should be excluded from training or evaluation.

| start | end | reason |
| --- | --- | --- |
|  |  |  |

Common reasons:
- maintenance window
- known incident
- replay artifact
- missing telemetry
- timestamp corruption

## 5. Dataset Readiness Summary

- stable `device_id` confirmed: yes or no
- stable `interface` confirmed for primary path: yes or no
- timestamp continuity acceptable: yes or no
- duplicate rows reviewed: yes or no
- counter reset behavior reviewed: yes or no
- enough counters to derive `in_rate`, `out_rate`, `error_rate`: yes or no

## 6. Device Classification

| device_id | recommended scope | reason |
| --- | --- | --- |
|  | `per_interface` or `per_device` |  |

## 7. Event Correlation Readiness

- first non-SNMP event source selected:
- timestamp alignment reviewed: yes or no
- interface key available in event source: yes or no

## 8. Baseline Evaluation Plan

- evaluation dataset:
- current feature set:
- current sequence length:
- current threshold method:
- metrics to record:

Recommended minimum metrics:
- false positive count
- false positive rate
- per-device or per-interface anomaly counts
- detection delay where relevant

## 9. Baseline Deliverables

Attach or link the outputs produced during `P0`.

- dataset inventory report:
- data quality audit report:
- baseline metrics report:
- notes on fallback devices:

## 10. Approval To Start F1

Mark `ready` only when all items below are complete.

- [ ] dataset inventory completed
- [ ] data quality audit completed
- [ ] baseline slice frozen
- [ ] fallback devices identified
- [ ] enough counters available for derived-rate features
- [ ] artifact directory name chosen for the baseline run

## 11. Notes

Add anything important that could affect `F1`, `T1`, or later correlation work.
