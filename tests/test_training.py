from __future__ import annotations

from pathlib import Path

import mlflow

from ml_pipeline.data_loader import generate_sample_churn_data
from ml_pipeline.train import train_pipeline


def test_training_pipeline_creates_model_artifacts(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    generate_sample_churn_data(n_samples=120).to_csv(data_path, index=False)

    model_path = tmp_path / "best_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr("ml_pipeline.model_registry.MODEL_PATH", model_path)
    monkeypatch.setattr("ml_pipeline.model_registry.METRICS_PATH", metrics_path)

    mlflow_dir = tmp_path / "mlruns"
    result = train_pipeline(data_path=str(data_path), tracking_uri=mlflow_dir.as_uri())

    assert result["best_model"] in {"logistic_regression", "random_forest"}
    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert result["metrics"]["f1"] >= 0.0

    assert result["split_sizes"] == {"train": 72, "validation": 24, "test": 24}
    selected = max(result["validation_comparison"], key=lambda name: result["validation_comparison"][name]["f1"])
    assert result["best_model"] == selected
    assert "roc_auc" in result["baseline_test_metrics"]
