# Predictive Analytics — Customer Churn Classification

MSIN0097 Individual Coursework

## Task
Binary classification of customer churn (`Exited` column) using the **Churn Modelling** dataset.

## Repository structure

```
data/               <- Place Churn Modelling.xlsx here
src/
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
python src/run_all.py
```

All random seeds are fixed to `42`. Outputs are saved to `outputs/`.

## Methodology
- Stratified 60/20/20 train-validation-test split
- Single scikit-learn `Pipeline` with `ColumnTransformer` (no leakage)
- Model selection on validation set (Logistic Regression, Random Forest, Gradient Boosting)
- Final evaluation on held-out test set (reported once)
- Metrics: ROC-AUC, PR-AUC, Recall@top-20%, F1, confusion matrix
