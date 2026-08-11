from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "svm.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained SVM model was not found. Run `python src/train.py` first."
        )

    model = joblib.load(MODEL_PATH)

    # The UCI Phishing Websites dataset contains 30 pre-extracted integer
    # features. This example uses an existing row-shaped feature vector only
    # to demonstrate the prediction interface; it does not invent a URL label.
    #
    # Replace these values with a real 30-feature vector from the dataset or
    # from a separately validated feature-extraction pipeline.
    feature_names = [
        "having_ip_address",
        "url_length",
        "shortining_service",
        "having_at_symbol",
        "double_slash_redirecting",
        "prefix_suffix",
        "having_sub_domain",
        "sslfinal_state",
        "domain_registration_length",
        "favicon",
        "port",
        "https_token",
        "request_url",
        "url_of_anchor",
        "links_in_tags",
        "sfh",
        "submitting_to_email",
        "abnormal_url",
        "redirect",
        "on_mouseover",
        "rightclick",
        "popupwindow",
        "iframe",
        "age_of_domain",
        "dnsrecord",
        "web_traffic",
        "page_rank",
        "google_index",
        "links_pointing_to_page",
        "statistical_report",
    ]

    # A vector must come from actual feature extraction or an actual dataset
    # row. The script intentionally refuses to create a fake example vector.
    vector_file = ROOT / "data" / "prediction_features.csv"

    if not vector_file.exists():
        print("Model loaded successfully.")
        print("\nTo make a real prediction:")
        print("1. Create data/prediction_features.csv.")
        print("2. Put exactly one row containing the 30 UCI feature values.")
        print("3. Keep the column names below in the same order.")
        print("\nRequired columns:")
        print(", ".join(feature_names))
        return

    df = pd.read_csv(vector_file)

    if list(df.columns) != feature_names:
        raise ValueError(
            "prediction_features.csv must contain exactly the 30 UCI feature "
            "columns in the documented order."
        )

    if len(df) != 1:
        raise ValueError("prediction_features.csv must contain exactly one row.")

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])

    print(f"Predicted class: {prediction}")
    print(f"Estimated probability for class 1: {probability:.6f}")


if __name__ == "__main__":
    main()
