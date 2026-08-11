# Results

This directory receives generated outputs when `python src/train.py` is executed.

Expected generated files include:

- `metrics_test.csv` — held-out test-set metrics
- `cross_validation.csv` — fold-level cross-validation scores
- `roc_curve.png` — ROC curves for both classifiers
- `confusion_matrices.png` — confusion matrices
- `logistic_regression.joblib` — trained Logistic Regression pipeline
- `svm.joblib` — trained SVM pipeline

No experimental result is claimed in this repository until the training script has actually been executed.
