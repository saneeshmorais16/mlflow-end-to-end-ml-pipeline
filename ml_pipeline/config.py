from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "best_model.joblib"))
METRICS_PATH = MODELS_DIR / "metrics.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "sample-churn-classification"
RANDOM_STATE = 42
TARGET_COLUMN = "churn"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_calls",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "has_online_security",
    "is_senior_citizen",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
