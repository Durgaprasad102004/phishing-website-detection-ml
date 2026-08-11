# Dataset

The project uses the **Phishing Websites** dataset from the UCI Machine Learning Repository, dataset ID 327.

Official source:

https://archive.ics.uci.edu/dataset/327/phishing

UCI reports:

- 11,055 instances
- 30 features
- Integer feature values
- No missing values
- Classification task

The dataset is intentionally not duplicated in this repository. `src/train.py` retrieves it through the `ucimlrepo` package so the repository does not redistribute a copy of the dataset.

The UCI dataset license is CC BY 4.0. Please retain the dataset citation when using the project.
