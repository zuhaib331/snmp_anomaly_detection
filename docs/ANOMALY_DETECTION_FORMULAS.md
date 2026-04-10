# Anomaly Detection Formulas and Calculations

This document explains how anomaly detection works in this project, how each sequence window is scored, and how the anomaly threshold is set.

## 1. Core Idea

The model used in this project is an **LSTM Autoencoder**.

Its job is:
- take a sequence window as input
- reconstruct the same sequence window
- compare the reconstructed output with the original input
- assign an anomaly score based on reconstruction error

If the model was trained only on normal behavior, then:
- **normal windows** are reconstructed well, so error is low
- **abnormal windows** are reconstructed poorly, so error is high

## 2. What Is One Input Window?

The model does not usually evaluate one row by itself.
It evaluates a **sequence window** of fixed length.

In this project:
- `sequence_length = 10`
- `feature_columns = [cpu, memory, in_octets, out_octets, errors]`

So one model input has shape:

```text
(10, 5)
```

Meaning:
- 10 timestamps
- 5 features at each timestamp

Example window:

```text
[
  [cpu, memory, in_octets, out_octets, errors],
  [cpu, memory, in_octets, out_octets, errors],
  ... 10 rows total ...
]
```

During training with batches, the full model input becomes:

```text
(batch_size, sequence_length, number_of_features)
```

Example:

```text
(64, 10, 5)
```

## 3. Reconstruction Process

Let the original input window be:

```text
X
```

Let the reconstructed output from the autoencoder be:

```text
X_hat
```

The model tries to make:

```text
X_hat ≈ X
```

If the window looks normal, reconstruction is close.
If the window looks unusual, reconstruction becomes worse.

## 4. Reconstruction Error Formula

The anomaly score is based on **Mean Squared Error (MSE)**.

For each element in the window:

```text
difference = X - X_hat
```

Square it:

```text
squared_error = (X - X_hat)^2
```

Then average across all timesteps and all features:

```text
Reconstruction Error = mean((X - X_hat)^2)
```

This gives one scalar value for each sequence window.

## 5. Expanded Formula

If:
- `T` = number of timesteps in a sequence
- `F` = number of features
- `X[t, f]` = original value at timestep `t` and feature `f`
- `X_hat[t, f]` = reconstructed value at timestep `t` and feature `f`

Then the reconstruction error for one window is:

```text
Error = (1 / (T * F)) * Σ Σ (X[t, f] - X_hat[t, f])^2
```

Where:
- the first summation is over timesteps
- the second summation is over features

In this project:
- `T = 10`
- `F = 5`

So the error is averaged over `10 * 5 = 50` values.

## 6. Example of Error Calculation

Suppose one small part of a window has these original and reconstructed values:

```text
Original:      [0.80, 0.60, 0.20]
Reconstructed: [0.75, 0.62, 0.30]
```

Step 1: subtract

```text
[0.80 - 0.75, 0.60 - 0.62, 0.20 - 0.30]
= [0.05, -0.02, -0.10]
```

Step 2: square

```text
[0.0025, 0.0004, 0.0100]
```

Step 3: average

```text
(0.0025 + 0.0004 + 0.0100) / 3 = 0.0043
```

That average becomes the error score for that example.

For the real model, the same calculation is done across all values in the full sequence window.

## 7. Feature-Level and Timestep-Level Error

The system also calculates additional explanations.

### Feature-level error

This shows which feature contributed most to the window error.

Formula:

```text
Feature Error[f] = mean over timesteps of (X[t, f] - X_hat[t, f])^2
```

This is how the project derives fields like:
- `feature_error_cpu`
- `feature_error_memory`
- `feature_error_in_octets`
- `feature_error_out_octets`
- `feature_error_errors`

The highest one becomes:

```text
top_error_feature
```

### Timestep-level error

This shows which position inside the window contributed most.

Formula:

```text
Timestep Error[t] = mean over features of (X[t, f] - X_hat[t, f])^2
```

The highest one becomes:

```text
top_error_timestep_offset
```

If the value is `7`, it means the 8th point inside the 10-step window contributed the most error.

## 8. How Threshold Is Set

The threshold is calculated from **normal training data only**.

Why?
Because the autoencoder is trained on normal behavior, so normal reconstruction errors show what “expected” error looks like.

### Step-by-step threshold calculation

1. Train the model on normal windows only.
2. Run the trained model on those same training windows.
3. Compute reconstruction error for each normal training window.
4. Collect all these errors into a list:

```text
train_errors = [e1, e2, e3, ..., en]
```

5. Compute:

```text
mean_error = mean(train_errors)
std_error = standard_deviation(train_errors)
```

6. Set threshold as:

```text
threshold = mean_error + K * std_error
```

In the current project:

```text
K = threshold_std_multiplier = 3.0
```

So the actual formula is:

```text
threshold = mean(train_errors) + 3 * std(train_errors)
```

## 9. Why This Threshold Works

Normal windows usually have reconstruction errors near the average training error.

If a new window produces an error much larger than the normal range, it is treated as suspicious.

So:
- low error means the model recognizes the pattern as normal
- high error means the model struggles to reconstruct it, so it may be anomalous

## 10. Final Detection Rule

Once a new sequence window is scored, the project uses this rule:

```text
if reconstruction_error > threshold:
    predicted_anomaly = 1
else:
    predicted_anomaly = 0
```

Meaning:
- `1` = anomaly
- `0` = normal

## 11. Error Margin

The project also computes:

```text
error_margin = reconstruction_error - threshold
```

Interpretation:
- `error_margin > 0` means anomaly score crossed the threshold
- `error_margin < 0` means the score stayed below threshold

This is useful to understand how strongly a window was flagged.

## 12. What Is Saved in Output Files?

### `anomaly_results.csv`

Each row represents one scored sequence window.

Important columns:
- `device_id`
- `device_window_index`
- `window_start`
- `window_end`
- `sequence_length`
- `reconstruction_error`
- `threshold`
- `error_margin`
- `predicted_anomaly`
- `top_error_feature`
- `top_error_timestep_offset`
- `detection_basis`

### `anomaly_windows.json`

This file stores the actual raw 10-record window for each detected anomaly.

It helps answer:
- which sequence was flagged
- what values were inside that sequence
- which feature contributed most
- why the system marked it as anomalous

## 13. Example Detection Decision

Suppose a window gets:

```text
reconstruction_error = 0.0724
threshold = 0.0169
```

Then:

```text
error_margin = 0.0724 - 0.0169 = 0.0555
```

Since:

```text
0.0724 > 0.0169
```

The result is:

```text
predicted_anomaly = 1
```

If another window gets:

```text
reconstruction_error = 0.0101
threshold = 0.0169
```

Then:

```text
0.0101 < 0.0169
```

So:

```text
predicted_anomaly = 0
```

## 14. Parameters That Affect Detection

These parameters strongly affect scoring and threshold behavior:

### `sequence_length`
Number of timestamps inside one window.
Larger windows capture more temporal behavior.

### `hidden_size`
Controls model capacity in the LSTM.

### `latent_size`
Controls how much the sequence is compressed.

### `epochs`
How long the model trains.

### `learning_rate`
How fast weights are updated.

### `threshold_std_multiplier`
Controls anomaly sensitivity.

Example:
- smaller multiplier -> more sensitive detector
- larger multiplier -> stricter detector

## 15. Summary

The anomaly detection pipeline works as follows:

1. Build a sequence window of 10 timesteps.
2. Pass it through the trained LSTM autoencoder.
3. Reconstruct the same input window.
4. Compute Mean Squared Error between original and reconstructed sequence.
5. Use that error as the anomaly score.
6. Compare it against the threshold.
7. If the score is above the threshold, mark the window as anomalous.

In one line:

```text
Anomaly Score = mean((Original Window - Reconstructed Window)^2)
```

and

```text
Threshold = mean(train_errors) + 3 * std(train_errors)
```

and final decision:

```text
if score > threshold -> anomaly
else -> normal
```
