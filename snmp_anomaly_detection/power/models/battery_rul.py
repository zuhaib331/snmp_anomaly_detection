"""LSTM regression model for battery Remaining Useful Life (RUL) prediction.

Architecture:
  - Encoder: LSTM (seq_len × feature_count → hidden_size)
  - Dropout: applied at inference for MC Dropout confidence intervals
  - Head: Linear(hidden_size → 1) → predicted days_to_replacement

RULPrediction is the structured output returned by the inference path.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn

    class BatteryRULModel(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]          # shape: (batch, hidden_size)
            out = self.dropout(last_hidden)
            return self.head(out).squeeze(-1)  # shape: (batch,)

except ImportError:
    torch = None  # type: ignore

    class BatteryRULModel:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for BatteryRULModel.")


# Advisory thresholds (days)
URGENT_DAYS: int = 14
WARN_DAYS: int = 30


@dataclass
class RULPrediction:
    device_id: str
    timestamp: str
    predicted_rul_days: float
    confidence_interval_lo: float
    confidence_interval_hi: float
    replacement_advisory: str      # "urgent" | "warn" | "ok"

    def __post_init__(self) -> None:
        if self.predicted_rul_days <= URGENT_DAYS:
            object.__setattr__(self, "replacement_advisory", "urgent")
        elif self.predicted_rul_days <= WARN_DAYS:
            object.__setattr__(self, "replacement_advisory", "warn")
        else:
            object.__setattr__(self, "replacement_advisory", "ok")
