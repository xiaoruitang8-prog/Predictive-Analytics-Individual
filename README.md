# Predictive Analytics — Customer Churn Classification

MSIN0097 Individual Coursework

## Task
Binary classification of customer churn (`Exited` column) using the **Churn Modelling** dataset.

## Data access

The dataset is **not included** in this repository (excluded via `.gitignore`).

To set up the data:

1. Download the Excel file from Kaggle:
   <https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data>
2. Rename the file (if needed) to **exactly** `Churn_Modelling.xlsx`.
3. Place it at: `data/Churn_Modelling.xlsx`

Sanity check — run this from the repo root:

```bash
python -c "import pandas as pd; df = pd.read_excel('data/Churn_Modelling.xlsx'); print(f'Loaded {len(df)} rows, {len(df.columns)} columns')"
```

If the file is missing, every script will raise a clear error with download instructions.

## Repository structure

```
data/               <- Place Churn_Modelling.xlsx here (git-ignored)
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
# Place Churn_Modelling.xlsx in data/ first
python src/run_all.py
```

All random seeds are fixed to `42`. Outputs are saved to `outputs/`.

## Methodology
- Stratified 60/20/20 train-validation-test split
- Single scikit-learn `Pipeline` with `ColumnTransformer` (no leakage)
- Model selection on validation set (DummyClassifier baseline, Logistic Regression, HistGradientBoosting, MLP with early stopping)
- Final evaluation on held-out test set (reported once)
- Metrics: ROC-AUC, PR-AUC, Recall@top-20%, F1, confusion matrix
