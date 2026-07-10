from __future__ import annotations

import pandas as pd

from ml_pipeline.config import FEATURE_COLUMNS, TARGET_COLUMN


def validate_training_data(data: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if data.empty:
        raise ValueError("Training data is empty.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    unique_targets = set(data[TARGET_COLUMN].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("Target column must contain binary labels 0 and 1.")

    if data[FEATURE_COLUMNS].isna().mean().max() > 0.5:
        raise ValueError("At least one feature column has more than 50% missing values.")
