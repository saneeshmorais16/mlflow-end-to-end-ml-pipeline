from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint(monkeypatch):
    class DummyModel:
        def predict(self, frame: pd.DataFrame):
            assert len(frame) == 1
            return [1]

        def predict_proba(self, frame: pd.DataFrame):
            return [[0.2, 0.8]]

    monkeypatch.setattr("app.inference.load_model", lambda: DummyModel())

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "tenure_months": 6,
            "monthly_charges": 95.5,
            "total_charges": 573.0,
            "contract_type": "month-to-month",
            "internet_service": "fiber",
            "payment_method": "electronic_check",
            "support_calls": 3,
            "has_online_security": False,
            "is_senior_citizen": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["label"] == "churn"
    assert body["churn_probability"] == 0.8
