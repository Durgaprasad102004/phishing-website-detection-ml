# Phishing Website Detection Using Machine Learning

A reproducible machine-learning project for binary classification of phishing and legitimate websites.

## Project objective

The system trains and compares two supervised classification algorithms:

- Logistic Regression
- Support Vector Machine (SVM)

The project uses the **Phishing Websites** dataset from the UCI Machine Learning Repository. The dataset contains 11,055 instances and 30 integer-valued features, with no missing values reported by UCI.

The dataset is **not copied into this repository**. The training script downloads it through the official `ucimlrepo` package when the project is run.

## Dataset source

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/327/phishing

Citation:

Mohammad, R., & McCluskey, L. (2012). *Phishing Websites*. UCI Machine Learning Repository. DOI: 10.24432/C51W2X.

Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Technologies

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- SciPy
- matplotlib
- ucimlrepo
- joblib

## Repository structure

```text
phishing-website-detection-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── train.py
│   └── predict.py
└── results/
    └── README.md
```

## How to run

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train and evaluate

```bash
python src/train.py
```

The script:

1. Fetches dataset ID 327 from UCI.
2. Separates features and target.
3. Performs a stratified train/test split.
4. Standardizes numeric features using statistics learned only from the training set.
5. Trains Logistic Regression and SVM models.
6. Calculates Accuracy, Precision, Recall, F1-score and ROC-AUC on the held-out test set.
7. Runs 5-fold stratified cross-validation for both models.
8. Saves a metrics CSV, ROC curve, confusion matrices and trained models under `results/`.

The exact results are generated when the code is executed. No performance numbers are hard-coded in this repository.

### 4. Predict from a feature vector

After training:

```bash
python src/predict.py
```

The prediction script loads the saved SVM pipeline and demonstrates the expected 30-feature input format. It does **not** claim to extract features from a URL automatically; the supplied dataset contains pre-extracted website features.

## Important reproducibility note

This repository deliberately does not invent experimental results. Run `python src/train.py` to generate the actual results from the UCI dataset and the code in this repository.

## Scope

This is a research/academic machine-learning classifier. A prediction from the model should not be treated as a guarantee that a website is safe or malicious. Real-world deployment would require current data, continuous validation and additional security controls.
