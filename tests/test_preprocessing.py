from __future__ import annotations

from ml_pipeline.config import FEATURE_COLUMNS
from ml_pipeline.data_loader import generate_sample_churn_data
from ml_pipeline.preprocessing import build_preprocessor


def test_preprocessor_transforms_features():
    data = generate_sample_churn_data(n_samples=25)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(data[FEATURE_COLUMNS])

    assert transformed.shape[0] == 25
    assert transformed.shape[1] >= len(FEATURE_COLUMNS)
