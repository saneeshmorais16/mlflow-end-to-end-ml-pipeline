# Local inference demo

Use Python 3.12, create and activate a virtual environment, then:

```sh
pip install -r requirements.txt
python -m pytest -q
python -m evaluation.run
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`, expand `POST /predict`, choose **Try it out**, and paste `api_example.request` from [the measured output](../evaluation/results.json). Execute it to obtain the saved model's prediction. The example is synthetic; its score is not a real customer risk estimate.


No public hosted API is claimed. Training produces ignored model artefacts locally; do not upload private datasets or model binaries.

## Recorded API exchange

Request:

```json
{
  "tenure_months": 6,
  "monthly_charges": 95.5,
  "total_charges": 573.0,
  "contract_type": "month-to-month",
  "internet_service": "fiber",
  "payment_method": "electronic_check",
  "support_calls": 3,
  "has_online_security": false,
  "is_senior_citizen": true
}
```

Actual response:

```json
{
  "prediction": 1,
  "label": "churn",
  "churn_probability": 0.8506308043808044
}
```
