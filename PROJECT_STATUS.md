# Project Status

## Current Build Status

Starter implementation complete locally. GitHub Actions is configured to run tests on push and pull requests.

## Completed Features

- Generated non-sensitive churn classification dataset.
- Data loading and validation modules.
- Preprocessing pipeline for numeric and categorical features.
- Train/test split and model training.
- Logistic Regression and Random Forest model comparison.
- Evaluation metrics: accuracy, precision, recall, F1, and ROC-AUC.
- MLflow experiment tracking for parameters, metrics, and model artifacts.
- Best model saved locally after training.
- FastAPI endpoints for health checks and prediction.
- Dockerfile for local container builds.
- Pytest coverage for preprocessing, training, health, and prediction endpoints.
- GitHub Actions workflow for tests.

## Remaining Improvements

- Add richer data drift and schema validation.
- Add model registry promotion stages using a remote MLflow backend.
- Add model explainability reports.
- Add monitoring and deployment infrastructure if a real deployment target is chosen.

## Testing Status

Latest local check:

- `pytest`: 4 passed
- `python run_training.py`: completed and selected Random Forest on generated sample data

Run locally with:

```bash
pytest
```

## What Needs Checking Before Making Public

- Confirm no `.env` files, credentials, logs, generated reports, private datasets, or local MLflow runs are tracked.
- Confirm generated model binaries are excluded unless intentionally published through a release artifact.
- Confirm README metrics match the latest local training run if metrics are updated.
- Confirm the repository does not claim production deployment.

## Files That Must Never Be Committed

- `.env`
- `.env.*`
- API keys, tokens, credentials, or private configuration files
- `secrets/`
- `credentials/`
- Private datasets
- Database files such as `*.db`, `*.sqlite`, and `*.sqlite3`
- Logs such as `*.log`
- Generated PDFs
- Local MLflow runs in `mlruns/`
- Local model binaries such as `models/*.pkl` and `models/*.joblib`
