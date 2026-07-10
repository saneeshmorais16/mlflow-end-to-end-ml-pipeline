from __future__ import annotations

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_pipeline.config import FEATURE_COLUMNS, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_STATE, TARGET_COLUMN
from ml_pipeline.data_loader import load_data
from ml_pipeline.evaluate import evaluate_classifier
from ml_pipeline.model_registry import save_best_model, save_metrics, save_model_card
from ml_pipeline.preprocessing import build_preprocessor
from ml_pipeline.validation import validate_training_data


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_pipeline(data_path: str | None = None, tracking_uri: str | None = None) -> dict:
    active_tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
    if active_tracking_uri:
        mlflow.set_tracking_uri(active_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    data = load_data(data_path)
    validate_training_data(data)

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_result: dict | None = None
    for model_name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            pipeline.fit(x_train, y_train)
            metrics = evaluate_classifier(pipeline, x_test, y_test)
            params = estimator.get_params()
            mlflow.log_params({f"model__{key}": value for key, value in params.items() if isinstance(value, (str, int, float, bool)) or value is None})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            result = {
                "model_name": model_name,
                "model": pipeline,
                "metrics": metrics,
            }
            if best_result is None or metrics.get("f1", 0.0) > best_result["metrics"].get("f1", 0.0):
                best_result = result

    if best_result is None:
        raise RuntimeError("Training did not produce a model.")

    model_path = save_best_model(best_result["model"])
    metrics_path = save_metrics(
        {
            "best_model": best_result["model_name"],
            **best_result["metrics"],
        }
    )
    model_card_path = save_model_card(best_result["model_name"], best_result["metrics"])

    return {
        "best_model": best_result["model_name"],
        "metrics": best_result["metrics"],
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_card_path": str(model_card_path),
    }
