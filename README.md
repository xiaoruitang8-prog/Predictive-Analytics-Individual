# Predictive Analytics — Customer Churn Classification

MSIN0097 Individual Coursework

## Task
Binary classification of customer churn (`Exited` column) using the **Churn Modelling** dataset.

## Data access

A **200-row stratified sample** is included in the repository at
`data/sample_customer_churn.csv` so that scripts can run out of the box for
a quick preview.

For the **full dataset** (10 000 rows):

1. Download the CSV from Kaggle (licence: CC0 — Public Domain):
   <https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data>
2. Place the file at **exactly**: `data/customer_churn.csv`

The full file is excluded via `.gitignore`; the sample is committed.

The data loader (`src/data_loader.py`) reads the full dataset first. If it is
missing, it falls back to the sample with a warning that results are
preview-only. If neither file exists, a clear error is raised with the Kaggle
link and required paths.

Sanity check — run from the repo root:

```bash
python -c "from src.data_loader import load_churn_data; df = load_churn_data(); print(f'Loaded {len(df)} rows, {len(df.columns)} columns')"
```

## Repository structure

```
data/               <- sample_customer_churn.csv (committed) + customer_churn.csv (git-ignored)
src/
  data_loader.py    <- Shared data-loading utility (file check + error msg)
  01_eda.py         <- Exploratory data analysis
  02_modelling.py   <- Pipeline, model selection, final evaluation
  run_all.py        <- Single entry point to reproduce everything
outputs/            <- Plots, metrics, saved models (git-ignored heavy files)
appendix/           <- Agent log and decision register for the report
requirements.txt    <- Pinned Python dependencies
```

## How to reproduce

```bash
pip install -r requirements.txt
# Works immediately with the sample; place customer_churn.csv in data/ for full results
python src/run_all.py
```

All random seeds are fixed to `42`. Outputs are saved to `outputs/`.

## Methodology
- Stratified 60/20/20 train-validation-test split
- Single scikit-learn `Pipeline` with `ColumnTransformer` (no leakage)
- Model selection on validation set (DummyClassifier baseline, Logistic Regression, HistGradientBoosting, MLP with early stopping)
- Final evaluation on held-out test set (reported once)
- Metrics: ROC-AUC, PR-AUC, Recall@top-20%, F1, confusion matrix
