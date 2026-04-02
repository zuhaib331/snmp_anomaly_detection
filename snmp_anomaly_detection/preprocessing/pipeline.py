from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def available_scaler_names() -> tuple[str, ...]:
    return ("minmax", "standard", "robust")


def build_scaler(scaler_name: str) -> Any:
    normalized_name = scaler_name.strip().lower()
    if normalized_name == "minmax":
        return MinMaxScaler()
    if normalized_name == "standard":
        return StandardScaler()
    if normalized_name == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unsupported scaler `{scaler_name}`. "
        f"Expected one of: {', '.join(available_scaler_names())}."
    )


@dataclass
class PreprocessingPipeline:
    feature_columns: tuple[str, ...]
    scaler_name: str = "minmax"
    log1p_features: tuple[str, ...] = ()
    scaler: Any | None = None

    def __post_init__(self) -> None:
        missing_features = sorted(set(self.log1p_features) - set(self.feature_columns))
        if missing_features:
            raise ValueError(
                "log1p_features must be a subset of feature_columns. "
                f"Unexpected values: {', '.join(missing_features)}"
            )
        if self.scaler is None:
            self.scaler = build_scaler(self.scaler_name)

    def fit(self, dataframe: pd.DataFrame) -> "PreprocessingPipeline":
        transformed_features = self._transform_feature_frame(
            dataframe[list(self.feature_columns)]
        )
        self.scaler.fit(transformed_features)
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        feature_frame = self._transform_feature_frame(
            transformed[list(self.feature_columns)]
        )
        transformed[list(self.feature_columns)] = self.scaler.transform(feature_frame)
        return transformed

    def _transform_feature_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        transformed = dataframe.copy()
        for feature_name in self.log1p_features:
            # SNMP rates are expected to be non-negative; clipping protects log1p
            # against occasional negative values caused by imperfect upstream data.
            clipped_values = np.maximum(transformed[feature_name].to_numpy(dtype=float), 0.0)
            transformed[feature_name] = np.log1p(clipped_values)
        return transformed
