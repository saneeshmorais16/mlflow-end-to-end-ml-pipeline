# Synthetic churn evaluation

Run `python -m evaluation.run` from the repository root after installing requirements. Raw results, package versions, and an actual API prediction are in [results.json](../evaluation/results.json).

## Dataset and method

The existing generator creates 600 synthetic records with seed 42. No customer data is used. Stratified partitions contain 360 training, 120 validation, and 120 test rows. Preprocessing is fitted on training rows only. Logistic regression and random forest are compared by **validation F1**; the winner is then evaluated on the untouched test partition, without refitting. A most-frequent-class dummy classifier provides a test baseline.

The previous implementation selected the winner using test scores. This has been corrected; the regression test verifies selection against validation metrics.

| Held-out metric | Majority baseline | Selected random forest |
| --- | ---: | ---: |
| Accuracy | 0.6750 | 0.7083 |
| F1 | 0.0000 | 0.5070 |
| ROC-AUC | 0.5000 | 0.7404 |
| Recall | 0.0000 | 0.4615 |

Validation F1: logistic regression 0.5246; random forest 0.5397. The selection rule was fixed before interpreting the test result.

## Decisions and limitations

- Separate validation and test sets avoid selecting the model on the reported final score.
- A scikit-learn pipeline keeps preprocessing consistent between training and inference.
- MLflow records candidate validation metrics and model artefacts; the JSON report preserves final test results.
- The selected model improves accuracy only modestly over the majority baseline and misses more than half of positive cases at its default threshold.
- This is one synthetic dataset and one split, not cross-validation or evidence of real customer performance. No confidence interval, calibration, fairness, or deployment validation is claimed.
- The local API was exercised with the saved model. Docker execution was not verified because Docker is unavailable in the audit environment.
