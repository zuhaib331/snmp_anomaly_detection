"""Time-based train/val/test split — no shuffle to prevent leakage."""
from __future__ import annotations

import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df chronologically into train / val / test partitions.

    Rows are sorted by timestamp before splitting so that the test set always
    contains the most recent data and there is no temporal leakage.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = sorted_df.iloc[:train_end].copy()
    val = sorted_df.iloc[train_end:val_end].copy()
    test = sorted_df.iloc[val_end:].copy()
    return train, val, test
