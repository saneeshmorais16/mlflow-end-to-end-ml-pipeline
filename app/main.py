from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.inference import ModelNotReadyError, predict_churn
from app.schemas import ChurnPredictionRequest, ChurnPredictionResponse

app = FastAPI(
    title="MLflow End-to-End ML Pipeline API",
    description="FastAPI inference service for the sample churn classifier.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest) -> dict:
    try:
        return predict_churn(request.model_dump())
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
