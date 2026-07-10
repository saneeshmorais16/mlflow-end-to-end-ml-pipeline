from __future__ import annotations

from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str = Field(..., examples=["month-to-month"])
    internet_service: str = Field(..., examples=["fiber"])
    payment_method: str = Field(..., examples=["electronic_check"])
    support_calls: int = Field(..., ge=0, le=20)
    has_online_security: bool
    is_senior_citizen: bool


class ChurnPredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float | None = None
