from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.config import RAW_DATA_DIR, RANDOM_STATE


def generate_sample_churn_data(n_samples: int = 600, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    tenure = rng.integers(1, 73, size=n_samples)
    monthly_charges = rng.normal(72, 22, size=n_samples).clip(20, 140).round(2)
    total_charges = (tenure * monthly_charges * rng.normal(1.0, 0.08, size=n_samples)).clip(20).round(2)
    support_calls = rng.poisson(1.4, size=n_samples).clip(0, 12)

    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_samples,
        p=[0.55, 0.28, 0.17],
    )
    internet_service = rng.choice(["dsl", "fiber", "none"], size=n_samples, p=[0.35, 0.5, 0.15])
    payment_method = rng.choice(
        ["electronic_check", "credit_card", "bank_transfer"],
        size=n_samples,
        p=[0.42, 0.33, 0.25],
    )
    has_online_security = rng.choice([True, False], size=n_samples, p=[0.46, 0.54])
    is_senior_citizen = rng.choice([True, False], size=n_samples, p=[0.18, 0.82])

    risk_score = (
        -1.2
        + 0.035 * support_calls
        + 0.018 * (monthly_charges - 70)
        - 0.035 * (tenure - 24)
        + np.where(contract_type == "month-to-month", 1.0, -0.45)
        + np.where(internet_service == "fiber", 0.45, -0.15)
        + np.where(payment_method == "electronic_check", 0.35, -0.1)
        + np.where(has_online_security, -0.45, 0.25)
        + np.where(is_senior_citizen, 0.25, 0.0)
    )
    probability = 1 / (1 + np.exp(-risk_score))
    churn = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "support_calls": support_calls,
            "has_online_security": has_online_security,
            "is_senior_citizen": is_senior_citizen,
            "churn": churn,
        }
    )


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        return generate_sample_churn_data()

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def write_sample_data(path: str | Path | None = None) -> Path:
    output_path = Path(path) if path else RAW_DATA_DIR / "sample_churn.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_churn_data().to_csv(output_path, index=False)
    return output_path
