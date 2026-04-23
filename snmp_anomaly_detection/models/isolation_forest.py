from __future__ import annotations

import numpy as np


def flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    if sequences.ndim != 3:
        raise ValueError(
            "Isolation Forest expects 3D sequence input shaped "
            "(window_count, sequence_length, feature_count)."
        )
    return sequences.reshape(sequences.shape[0], -1)


def feature_deviation_profile(
    sequence: np.ndarray,
    flattened_mean: np.ndarray,
    flattened_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flattened = sequence.reshape(-1)
    safe_std = np.where(flattened_std > 1e-12, flattened_std, 1.0)
    deviation = np.abs((flattened - flattened_mean) / safe_std).reshape(sequence.shape)
    return deviation.mean(axis=0), deviation.max(axis=1)
