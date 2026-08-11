from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset():
    """Fetch the official UCI Phishing Websites dataset (ID 327)."""
    dataset = fetch_ucirepo(id=327)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # UCI's target is numeric; convert it explicitly and validate.
    y = pd.to_numeric(y, errors="raise").astype(int)

    # Keep only numeric predictors. The UCI dataset is documented as integer-valued.
    X = X.apply(pd.to_numeric, errors="raise")

    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 features from UCI dataset 327, got {X.shape[1]}.")

    if len(X) != 11055:
        raise ValueError(f"Expected 11,055 rows from UCI dataset 327, got {len(X)}.")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("Unexpected missing values were found.")

    return X, y


def build_models():
    # The imputer is included for pipeline robustness, although UCI reports no
    # missing values. StandardScaler is fit inside each training split.
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("preprocess", preprocessing),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_test_set(models, X_train, X_test, y_train, y_test):
    rows = []
    roc_data = {}
    cms = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1_Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, y_prob),
            }
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))
        cms[name] = confusion_matrix(y_test, y_pred)

        filename = (
            "logistic_regression.joblib"
            if name == "Logistic Regression"
            else "svm.joblib"
        )
        joblib.dump(model, RESULTS / filename)

    return pd.DataFrame(rows), roc_data, cms


def cross_validate_models(models, X, y):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    all_rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        for fold_idx in range(5):
            all_rows.append(
                {
                    "Model": name,
                    "Fold": fold_idx + 1,
                    "Accuracy": scores["test_accuracy"][fold_idx],
                    "Precision": scores["test_precision"][fold_idx],
                    "Recall": scores["test_recall"][fold_idx],
                    "F1_Score": scores["test_f1"][fold_idx],
                    "ROC_AUC": scores["test_roc_auc"][fold_idx],
                }
            )

    return pd.DataFrame(all_rows)


def save_roc_plot(roc_data):
    plt.figure(figsize=(8, 6))

    for name, (fpr, tpr, auc_value) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.4f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Phishing Website Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "roc_curve.png", dpi=200)
    plt.close()


def save_confusion_matrices(cms):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (name, cm) in zip(axes, cms.items()):
        image = ax.imshow(cm)
        ax.set_title(name)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def main():
    print("Loading UCI dataset 327...")
    X, y = load_dataset()

    print(f"Dataset shape: {X.shape}")
    print(f"Target values: {sorted(y.unique().tolist())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    print("\nEvaluating held-out test set...")
    test_metrics, roc_data, cms = evaluate_test_set(
        models, X_train, X_test, y_train, y_test
    )
    test_metrics.to_csv(RESULTS / "metrics_test.csv", index=False)

    print("\nHeld-out test results:")
    print(test_metrics.to_string(index=False))

    print("\nRunning 5-fold stratified cross-validation...")
    cv_metrics = cross_validate_models(models, X, y)
    cv_metrics.to_csv(RESULTS / "cross_validation.csv", index=False)

    summary = (
        cv_metrics.groupby("Model")
        [["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]]
        .agg(["mean", "std"])
        .round(6)
    )
    summary.to_csv(RESULTS / "cross_validation_summary.csv")

    # Because the same stratified folds are used for both models, a paired
    # t-test is appropriate for comparing fold-level accuracy values.
    lr_acc = cv_metrics.loc[
        cv_metrics["Model"] == "Logistic Regression", "Accuracy"
    ].to_numpy()
    svm_acc = cv_metrics.loc[
        cv_metrics["Model"] == "SVM", "Accuracy"
    ].to_numpy()

    t_stat, p_value = ttest_rel(svm_acc, lr_acc)

    statistical_result = pd.DataFrame(
        [
            {
                "Comparison": "SVM vs Logistic Regression",
                "Test": "Paired t-test on 5-fold accuracy",
                "t_statistic": t_stat,
                "p_value": p_value,
            }
        ]
    )
    statistical_result.to_csv(RESULTS / "statistical_comparison.csv", index=False)

    save_roc_plot(roc_data)
    save_confusion_matrices(cms)

    print("\nCross-validation mean ± std:")
    print(summary)

    print("\nPaired t-test on fold-level accuracy:")
    print(f"t = {t_stat:.6f}, p = {p_value:.6f}")

    print("\nGenerated files:")
    for path in sorted(RESULTS.iterdir()):
        if path.is_file():
            print(f" - {path.name}")


if __name__ == "__main__":
    main()
