[README.md](https://github.com/user-attachments/files/33625509/README.md)
# Banking Fraud Detection with XGBoost

A classification capstone project that compares seven machine-learning models for identifying fraudulent transactions in PaySim mobile-money data. The primary focus is the minority fraud class: recall, precision, F1, and precision-recall average precision (AP), rather than accuracy alone.

> **Data note:** The notebook reads a local SQLite database. Do not commit or upload `classification.db` or the source CSV. Download the public dataset from Kaggle and recreate the local database as described below.

## Problem - Dataset

Financial transaction fraud is rare relative to legitimate activity. A model that predicts every transaction as legitimate can achieve high accuracy while missing nearly all fraud. This project evaluates how well classifiers identify the fraud class and explores the precision-recall trade-off.

The notebook uses the `fraud_detection` table from a local `classification.db`. It has **6,362,620 transactions**, **11 original columns**, and **8,213 fraud records** (about **0.129%**). The source is the synthetic **PaySim** mobile-money dataset, distributed as [Synthetic Financial Datasets for Fraud Detection on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). PaySim simulates mobile-money transactions based on aggregated transaction logs; it is not a set of real customer transactions. See the original [PaySim paper](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.html) for the simulator background.

Original fields include transaction step and type, amount, sender and recipient IDs and balances, `isFraud` (target), and `isFlaggedFraud` (a simulator-provided rule flag).

## Approach

### Feature engineering and preprocessing

- Converted simulation step into an hour-of-step feature; the notebook also derived a day-of-week feature, then dropped it during feature selection.
- Imputed missing transaction type as `Unknown`; imputed missing origin balance after the transaction from origin balance and amount; imputed missing recipient starting balance with the dataset median.
- One-hot encoded transaction type.
- Replaced raw account IDs with origin-customer and recipient-merchant prefix flags.
- Added balance-flow features: origin/destination balance changes, transaction-balance discrepancies, and amount ratios.
- Dropped `newbalanceOrig`, `newbalanceDest`, `dest_amount_ratio`, and `day_of_week` after correlation analysis.
- Used a stratified 70/30 train-test split (`random_state=0`), median-imputed remaining missing values, and MinMax scaling.

### Class imbalance and models

Fraud is approximately **0.129%** of the data (roughly 1 fraud to 773 non-fraud transactions). The notebook compares:

- Logistic Regression
- K-Nearest Neighbors (KNN; distance weighting, trained on a stratified 1% sample of the training data)
- Linear Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost
- Neural Network / MLP

Imbalance handling varied by model: XGBoost uses `scale_pos_weight`; SVM, Decision Tree, and Random Forest use `class_weight='balanced'`; the neural network uses class weights; and KNN uses distance weighting plus a stratified training sample. The notebook also evaluates XGBoost probabilities at a 0.85 decision threshold.

## Results

The following values are transcribed from the notebook's stratified test-set output. Precision, recall, and F1 are for the **fraud class** at each model's default classification threshold (0.50 for the neural network). AP is reported in the notebook's precision-recall summary.

| Model | Fraud precision | Fraud recall | Fraud F1 | PR average precision (AP) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3953 | 0.0613 | 0.1061 | 0.1741 |
| K-Nearest Neighbors | **0.6872** | 0.3129 | **0.4300** | 0.3742 |
| Support Vector Machine | 0.0176 | 0.9968 | 0.0345 | 0.2954 |
| Decision Tree | 0.0366 | 0.9935 | 0.0706 | 0.5794 |
| Random Forest | 0.0788 | 0.9927 | 0.1460 | 0.9010 |
| **XGBoost** | 0.1517 | **0.9976** | 0.2633 | **0.9361** |
| Neural Network (MLP) | 0.0195 | 0.9968 | 0.0382 | 0.5627 |

![Fraud-class precision, recall, and F1 by model](assets/model-comparison.svg)

At the notebook's XGBoost threshold of **0.85**, the test-set fraud-class precision was **0.34**, recall **0.99**, and F1 **0.50** (rounded in the classification report). This threshold result was selected and evaluated on the same test set; treat it as exploratory, and choose a threshold on validation data before a final untouched test evaluation.

**Interpretation:** XGBoost has the strongest reported PR average precision and the highest default-threshold fraud recall. KNN has the highest default-threshold fraud precision and F1, but only 31.29% recall. These models reflect different alert-volume trade-offs; the project does not establish a universally best model for a real bank's operating costs.

### Important evaluation caveats

- The notebook computes XGBoost `scale_pos_weight` using the displayed test-set class counts. Recompute it from `y_train` only to avoid using test-label information during training.
- The 0.85 threshold was evaluated on the same test split used for model comparison. Use a separate validation set to tune the threshold, then evaluate once on an untouched test set.
- The notebook calculates the missing `oldbalanceDest` median before the train-test split. Fit preprocessing statistics on training data only.
- PaySim is synthetic. Results on this dataset do not establish performance on live banking transactions.
- Review feature availability and leakage risk before deployment. In particular, `isFlaggedFraud` is a simulator-provided detection flag, and balance-derived features may reflect how PaySim generates fraud cases.
- The notebook uses a random transaction split; a chronological split and account/group-aware validation would better test future-period generalization.
- The dataset is highly imbalanced, so accuracy is not a useful headline metric by itself. Report false positives per 1,000 legitimate transactions and the operational review load alongside fraud recall.

## How to run

### 1. Get the data locally

Download the PaySim CSV from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it under `data/`. Do not commit the CSV or the generated SQLite database. Add rules such as these to `.gitignore`:

```gitignore
data/
*.db
*.sqlite
```

### 2. Install notebook dependencies

Use a virtual environment and install the versions appropriate for your Python environment:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow scipy
```

The notebook reads a SQLite table, so create `classification.db` and import the downloaded CSV into a table named `fraud_detection`. For a large CSV, import in chunks:

```python
import sqlite3
import pandas as pd

csv_path = "data/PS_20174392719_1491204439457_log.csv"  # update to the downloaded filename
with sqlite3.connect("data/classification.db") as conn:
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        chunk.to_sql(
            "fraud_detection", conn,
            if_exists="replace" if first else "append",
            index=False,
        )
        first = False
```

### 3. Point the notebook at your local database and run it

In the notebook's SQLite connection cell, update the database path to `data/classification.db`. Then launch Jupyter:

```bash
jupyter notebook Classification_Capstone_project.ipynb
```

The notebook compares models, prints classification reports, and saves chart images in the current working directory. Full execution processes over 6.3 million rows and can require substantial RAM and CPU time.

## Key learnings / limitations

- Accuracy can obscure failure on a rare class; compare fraud precision, recall, F1, and PR average precision, and decide thresholds using the cost of missed fraud versus investigation workload.
- XGBoost achieved the highest reported AP (0.9361) and very high recall, but at its default threshold it had only 0.1517 fraud precision. The threshold-adjusted result improved precision in the same test set, so it needs independent validation.
- Feature engineering from balance changes and account-type signals can strongly affect results. Validate that each feature exists at decision time and does not encode target-generation rules.
- PaySim is synthetic research data, not evidence of production bank performance. A real deployment would need representative, time-aware validation, calibrated probabilities, monitoring, privacy controls, and human review.
- The current notebook is an experiment, not a production pipeline: data paths are local, preprocessing is not packaged with a saved model, and training/evaluation are coupled in one notebook.

## Dataset citation

Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. 28th European Modeling and Simulation Symposium. [Proceedings page](https://www.msc-les.org/proceedings/emss/emss2016/emss2016_249.html).
