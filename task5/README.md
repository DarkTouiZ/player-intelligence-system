# Task 5 - Account Security Monitoring

Task 5 detects suspicious or anomalous player accounts without supervised labels.

## Notebook

- `task5_anomaly_detection.ipynb`

## What the Notebook Covers

- Loads `test.csv`.
- Groups and cleans account, activity, payment, device, and behavior-related features.
- Engineers additional security and behavior indicators.
- Scales numeric features for anomaly models.
- Trains unsupervised detectors, including Isolation Forest and One-Class SVM.
- Uses PCA and visualization to inspect anomaly separation.
- Combines anomaly signals with a soft score averaging ensemble.
- Produces an `is_anomaly` output column.

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Modeling Notes

The notebook sets an expected anomaly proportion with `CONTAMINATION = 0.10`. Tune this value if the deployment context expects a materially different suspicious-account rate.
