# MLflow End-to-End ML Pipeline

[Verification record](docs/VERIFICATION.md) · [Measured evaluation](docs/EVALUATION.md) · [Local demo](docs/DEMO.md)


This repository is a clean, portfolio-ready starter project for an end-to-end machine learning workflow. It trains a sample customer churn classifier, tracks experiments with MLflow, saves the best local model artifact, and exposes predictions through a FastAPI service.

The dataset is generated synthetically in code. It is intended for demonstrating engineering workflow, testing, and model lifecycle practices without committing private or sensitive data.

## Why It Matters

Many ML projects fail to show the operational pieces around a model. This project demonstrates a reproducible path from data loading through validation, preprocessing, training, evaluation, experiment tracking, inference, Docker packaging, and automated tests.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- MLflow
- FastAPI
- Docker
- pytest
- GitHub Actions

## Core Features

- Generated safe sample churn dataset.
- Data validation and preprocessing pipeline.
- Logistic Regression and Random Forest training.
- MLflow experiment tracking.
- Best-model artifact saving.
- FastAPI health and prediction endpoints.
- Docker support.
- Automated tests with GitHub Actions.

## ML Pipeline Architecture

```text
Generated sample data
  -> validation checks
  -> preprocessing with scikit-learn pipelines
  -> train/test split
  -> Logistic Regression and Random Forest training
  -> metrics logged to MLflow
  -> best model saved locally
  -> FastAPI prediction endpoint
```

## Setup

Use Python 3.12. Run these commands from the repository root in a fresh virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Train the Model

```bash
python run_training.py
```

Training uses generated sample churn data by default. The best model is saved to `models/best_model.joblib`, which is intentionally ignored by Git.

## Run MLflow

Start the MLflow UI locally:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Local MLflow runs are written to `mlruns/`, which is ignored by Git.

## Run the FastAPI App

Train the model first, then run:

```bash
uvicorn app.main:app --reload
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## Example Prediction Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 6,
    "monthly_charges": 95.5,
    "total_charges": 573.0,
    "contract_type": "month-to-month",
    "internet_service": "fiber",
    "payment_method": "electronic_check",
    "support_calls": 3,
    "has_online_security": false,
    "is_senior_citizen": true
  }'
```

Example response:

```json
{
  "prediction": 1,
  "label": "churn",
  "churn_probability": 0.82
}
```

The exact probability depends on the model produced by the latest local training run.

## Evaluation Metrics

The training pipeline computes:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

Metrics are logged to MLflow and written locally to `models/metrics.json`. Generated metrics files are local artifacts and should be reviewed before publishing.

## Docker

Build the image:

```bash
docker build -t mlflow-end-to-end-ml-pipeline .
```

Run the container:

```bash
docker run -p 8000:8000 mlflow-end-to-end-ml-pipeline
```

The container expects a trained model artifact to be present in `models/` at build time or mounted at runtime.

## Recorded local demo

An executed inference request and reproducible prediction output are included in [the demo guide](docs/DEMO.md).

## Honest Project Status

This is a portfolio starter project, not a production deployment. It demonstrates the structure of an end-to-end ML workflow with generated non-sensitive data.

## Tests

```bash
pytest
```

GitHub Actions runs the same test suite on pushes and pull requests.

## Responsible AI and Limitations

- This project uses synthetic sample data and should not be used for real customer decisions.
- The model is designed to demonstrate workflow, not to provide production-grade churn predictions.
- No fairness, bias, drift, or robustness analysis has been completed beyond basic validation and tests.
- Any real-world use would require appropriate data governance, monitoring, explainability, privacy review, and domain validation.

## Future Improvements

- Add formal data contracts with stricter schema validation.
- Add model explainability with SHAP or permutation importance.
- Add a remote MLflow tracking backend and staged model promotion.
- Add deployment manifests for a chosen cloud platform.
- Add monitoring for prediction quality and data drift.

## No-Secrets Note

Do not commit API keys, `.env` files, tokens, credentials, personal data, database files, logs, generated PDFs, local MLflow runs, model binaries, or private datasets. Use `.env.example` only for placeholder configuration names.
