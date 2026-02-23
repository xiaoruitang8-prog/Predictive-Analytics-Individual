# Task 4 — Model Exploration and Shortlist

---

## Coding Plan

1. **Imports + `evaluate()` helper.**  One consistent function used for every
   model: PR-AUC (primary), ROC-AUC (secondary), Recall@top-20% (business).
   `results` list accumulates one dict per model so a single cell builds the
   final table.

2. **Baselines.**  `DummyClassifier(most_frequent)` sets the PR-AUC floor.
   `LogisticRegression` with a `class_weight` ablation verifies whether
   balanced weighting helps before carrying the better variant forward.

3. **Candidate models.**  `RandomForestClassifier`, `HistGradientBoosting`
   (modern), `MLPClassifier` (neural baseline).  All fitted on `X_train_t`
   only; evaluated on `X_val_t`.

4. **Comparison table + shortlist.**  All `results` dicts collected into one
   `DataFrame`, sorted by PR-AUC.  Two models selected for Task 5 with
   written evidence.

Constraints: no test set touched; `random_state=SEED` (42) throughout;
no large grid searches.

---

## Cell 1 — Task 4 imports and evaluation function

```python
# ── Task 4: Model Exploration ────────────────────────────────────
# Prerequisite: Task 3 cells 1–6 must have been run first.
# Inherits: X_train_t, X_val_t, y_train, y_val, SEED = 42

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import pandas as pd

TOP_PCT = 0.20   # business constraint: top 20 % risk bucket

def evaluate(name, model, X, y, top_pct=TOP_PCT):
    """
    Evaluate a fitted binary classifier on split (X, y).
    Returns a dict: PR-AUC | ROC-AUC | Recall@top_pct.
    Caller controls which split is passed — test set never used here.
    """
    proba      = model.predict_proba(X)[:, 1]
    pr_auc     = average_precision_score(y, proba)
    roc_auc    = roc_auc_score(y, proba)
    n_top      = max(1, int(len(y) * top_pct))
    top_idx    = np.argsort(proba)[::-1][:n_top]
    recall_top = float(y.iloc[top_idx].sum()) / float(y.sum())
    col        = f"Recall@top{int(top_pct * 100)}%"
    return {
        "Model":   name,
        "PR-AUC":  round(pr_auc,     4),
        "ROC-AUC": round(roc_auc,    4),
        col:       round(recall_top, 4),
    }

results = []
print(f"evaluate() ready — metrics: PR-AUC | ROC-AUC | Recall@top{int(TOP_PCT*100)}%")
```

### 4.1  Experiment Setup

All models are compared on the **validation set** (`X_val_t`, `y_val`) using
the three metrics agreed in Section 1.3:

| Metric | Role | Rationale |
|--------|------|-----------|
| **PR-AUC** | Primary | Sensitive to minority-class performance; preferred for imbalanced data (Davis & Goadrich, 2006) |
| **ROC-AUC** | Secondary | Threshold-free; benchmarkable but less discriminating under imbalance |
| **Recall@top-20%** | Operational | Of the top 20% highest-risk customers, what fraction of actual churners are captured?  Simulates a fixed-capacity retention campaign. |

`evaluate()` uses only `predict_proba()` — no threshold is baked in, keeping
comparisons fair.  The test set is **never** passed to this function in
Task 4.

---

## Cell 2 — Baseline models

```python
# ── Baselines ────────────────────────────────────────────────────

# 1. Dummy — most_frequent always predicts non-churn.
#    Sets the PR-AUC floor (≈ positive-class rate ≈ 0.20).
dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_train_t, y_train)
results.append(evaluate("Dummy (most_frequent)", dummy, X_val_t, y_val))

# 2. LogReg: ablation on class_weight
#    Hypothesis: balanced weighting should improve PR-AUC on ~20 % churn.
ablation_lr = []
for cw, label in [
    (None,       "LogReg (default)"),
    ("balanced", "LogReg (balanced)"),
]:
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, class_weight=cw)
    m.fit(X_train_t, y_train)
    ablation_lr.append(evaluate(label, m, X_val_t, y_val))
    results.append(ablation_lr[-1])

print("LogReg class_weight ablation — val set:")
display(pd.DataFrame(ablation_lr))
```

### 4.2  Baseline Models

**Dummy (most_frequent)** always predicts the majority class (non-churn).
It cannot discriminate: ROC-AUC ≈ 0.50 and PR-AUC ≈ the churn rate (~0.20).
Any useful model must **clearly exceed** this floor on PR-AUC.

**LogReg ablation** tests whether `class_weight="balanced"` improves
minority-class recall.  With balanced weights, sklearn multiplies each
training loss by `n_samples / (2 × n_class_i)`, penalising missed churners
more heavily.  The ablation table above shows both variants on the same
validation split; the model with higher PR-AUC is the one carried forward.

*[fill: paste the printed ablation table here after running Cell 2 and note
which variant wins and by how much.]*

---

## Cell 3 — Candidate models

```python
# ── Candidate models ─────────────────────────────────────────────

# 3. RandomForest — balanced_subsample applies per-tree balanced weights
#    on each bootstrap sample (more robust than global balanced for bagging).
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_t, y_train)
results.append(evaluate("RandomForest (balanced_subsample)", rf, X_val_t, y_val))

# 4. HistGradientBoosting — modern tabular model (sklearn ≥ 1.2).
#    Scale-invariant; class_weight="balanced" re-weights gradient each iter.
hgbt = HistGradientBoostingClassifier(
    max_iter=300,
    class_weight="balanced",
    random_state=SEED,
)
hgbt.fit(X_train_t, y_train)
results.append(evaluate("HistGBT (balanced)", hgbt, X_val_t, y_val))

# 5. MLP — neural baseline; StandardScaler applied upstream in pipeline.
#    MLPClassifier has no class_weight param; trained on full X_train_t.
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=SEED,
)
mlp.fit(X_train_t, y_train)
results.append(evaluate("MLP (64-32)", mlp, X_val_t, y_val))

print("Candidate models fitted — val results:")
display(pd.DataFrame(results[-3:]))
```

### 4.3  Candidate Models

**RandomForest** is an ensemble of decorrelated decision trees.
`balanced_subsample` applies balanced class weights independently to each
bootstrap sample, which is more robust than a single global `balanced` weight
for bagging.  `n_estimators=200` gives stable probability estimates without
excessive runtime.

**HistGradientBoosting** is the modern tabular model in this comparison.
It trains in histogram-binned feature space (O(n) binning, O(n_bins × n_features)
per iteration), handles missing values natively, and typically outperforms
vanilla Random Forest on structured data at default settings.
`class_weight="balanced"` re-weights the loss gradient at every boosting
iteration.

> **Why not XGBoost / LightGBM?**  Neither is in `requirements.txt` and
> neither is installed in this environment.
> `HistGradientBoostingClassifier` is sklearn-native, version-pinned, and
> peer-reviewed — sufficient for a strong baseline comparison.
> XGBoost can be added in Task 5 as a tuning candidate if the budget allows.

**MLP** serves as a neural-network reference.  The preprocessing pipeline
already applies `StandardScaler`, so inputs are normalised before the network
sees them.  Architecture `(64, 32)` is a minimal two-hidden-layer network;
`max_iter=500` avoids `ConvergenceWarning` on this dataset size (~7 000
training rows).  MLPClassifier has no `class_weight` parameter — this is an
acknowledged limitation noted in the comparison.

*[fill: note whether MLP ConvergenceWarning appeared; copy val results from
Cell 3 output.]*

---

## Cell 4 — Comparison table and shortlist

```python
# ── Comparison table ─────────────────────────────────────────────
results_df = (
    pd.DataFrame(results)
    .sort_values("PR-AUC", ascending=False)
    .reset_index(drop=True)
)

print("=== Val set — all models, sorted by PR-AUC ===")
display(results_df)

# Shortlist: top-2 by PR-AUC, excluding Dummy
shortlist = (
    results_df[~results_df["Model"].str.startswith("Dummy")]
    .head(2)
    .reset_index(drop=True)
)
print("\nShortlisted for Task 5 (top-2 non-Dummy by PR-AUC):")
display(shortlist)
```

### 4.4  Model Comparison and Shortlist

**Full validation results** (sorted by PR-AUC):

| Model | PR-AUC | ROC-AUC | Recall@top-20% |
|-------|--------|---------|----------------|
| *[fill from Cell 4 output]* | | | |

**Shortlist — 2 models for Task 5:**

*[Fill 2–3 sentences after running cells, using the actual numbers.
Template below — replace bracketed values with real output.]*

**[Model A]** and **[Model B]** are shortlisted.
**[Model A]** achieves the highest validation PR-AUC (**[fill]**),
demonstrating that [gradient boosting / random forest] best exploits the
non-linear interactions in this tabular dataset.  **[Model B]** is retained
as the second candidate: it achieves PR-AUC = **[fill]** and provides
[coefficient-level / feature-importance] interpretability to satisfy the
Section 1.4 interpretability constraint.

*Guidance: HistGBT is expected to top the PR-AUC ranking.
If LogReg (balanced) outperforms RandomForest, substitute it as Model B —
linear coefficients are equally valid for interpretability.
If MLP outperforms both, note the result but prefer an interpretable second
model for Task 5 given the Section 1.4 constraint.*

---

## 4.5  Agent Plan vs My Verification

| Step | What the Agent Suggested | What I Will Verify | Decision Criteria |
|------|--------------------------|-------------------|-------------------|
| Evaluation function | Single `evaluate()` using PR-AUC, ROC-AUC, Recall@top-20% on val set only; no test data passed | Run Cell 1, confirm function executes and returns correct dict keys | No errors; dict keys match expected; no test split used |
| Dummy baseline | `most_frequent` strategy; PR-AUC ≈ class rate; ROC-AUC ≈ 0.50 | Check Cell 2 Dummy row in results | PR-AUC in [0.18–0.22]; ROC-AUC in [0.48–0.52] (Decision #32 confirmed if yes) |
| LogReg class_weight ablation | `balanced` expected to improve PR-AUC; both variants shown in ablation table | Read ablation table from Cell 2 | If `balanced` PR-AUC > `default` PR-AUC → Decision #33 confirmed; if not → revisit |
| RandomForest choice | `balanced_subsample` with 200 trees; expect clear improvement over LogReg | Check RF row vs LogReg (balanced) in Cell 4 table | RF PR-AUC > LogReg PR-AUC confirms ensemble advantage; otherwise note |
| HistGBT as modern model | HistGBT instead of XGBoost (not installed); `max_iter=300` default settings | Confirm HistGBT is at or near top of PR-AUC ranking | HistGBT PR-AUC ≥ RF PR-AUC → shortlist confirmed; if RF wins → note and still shortlist both |
| MLP inclusion | `(64, 32)` architecture, `max_iter=500`; neural baseline; no class_weight | Check no ConvergenceWarning; compare to RF/HistGBT | MLP expected to exceed LogReg but trail HistGBT; if MLP tops the table, report and flag |
| Shortlist selection | Top-2 non-Dummy models by PR-AUC | Confirm selection makes sense; check both interpretability and performance | Shortlist should include HistGBT; second slot = best remaining model; prefer interpretable model if tie |
| *[add rows as project progresses]* | | | |
