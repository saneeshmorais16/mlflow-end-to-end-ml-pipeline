from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml_pipeline.config import MODEL_PATH


class ModelNotReadyError(RuntimeError):
    """Raised when a trained model artifact is not available yet."""


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise ModelNotReadyError(
            f"Model artifact not found at {path}. Run `python run_training.py` first."
        )
    return joblib.load(path)


def predict_churn(payload: dict) -> dict:
    model = load_model()
    frame = pd.DataFrame([payload])
    prediction = int(model.predict(frame)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][1])

    return {
        "prediction": prediction,
        "label": "churn" if prediction == 1 else "not_churn",
        "churn_probability": probability,
    }
