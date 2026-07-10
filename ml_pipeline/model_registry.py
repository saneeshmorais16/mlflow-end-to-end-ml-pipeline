from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml_pipeline.config import METRICS_PATH, MODEL_PATH, MODELS_DIR


def save_best_model(model, metrics: dict | None = None, model_path: str | Path | None = None) -> Path:
    path = Path(model_path) if model_path else MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def save_metrics(metrics: dict, metrics_path: str | Path | None = None) -> Path:
    path = Path(metrics_path) if metrics_path else METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_model_card(model_name: str, metrics: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    card_path = MODELS_DIR / "MODEL_CARD.md"
    lines = [
        "# Model Card",
        "",
        f"Selected model: `{model_name}`",
        "",
        "This model is trained on generated sample churn data for demonstration.",
        "It is not intended for real customer decisions.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(f"- {key}: {value:.4f}" for key, value in metrics.items() if isinstance(value, float))
    card_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return card_path
