# Task 4 — Model Exploration and Shortlist

---

## Coding Plan

1. **4.1 Shared evaluation setup.**  Define `recall_precision_top()` and a
   single `evaluate()` helper returning four consistent metrics — PR-AUC,
   ROC-AUC, Recall@top-20%, Precision@top-20%.  A `results` list
   accumulates one dict per model so every cell feeds the same table.

2. **4.2 Baselines.**  `DummyClassifier (most_frequent)` sets the PR-AUC
   floor.  `LogisticRegression` (default settings) is the linear baseline —
   confirms features carry real signal before moving to ensembles.

3. **4.3 Tree ensemble and modern tabular model.**
   `RandomForestClassifier` (optional tree ensemble) and
   `HistGradientBoostingClassifier` (modern tabular approach, sklearn-native).
   Both trained on `X_train_t`, evaluated on `X_val_t`.

4. **4.4 Full comparison, operating-rule demo, and shortlist.**  One sorted
   table of all models.  A small controlled comparison: HistGBT with
   threshold 0.5 vs top-20% ranking, showing why ranking is more relevant
   for a fixed-capacity retention campaign.  Shortlist top-2 non-Dummy
   models for Task 5.

Constraints: **no hyperparameter search** in Task 4 — all models use
sensible defaults.  Test set **never touched**.  `random_state=SEED`
throughout.  Tuning is deferred entirely to Task 5.

---

## Cell 1 — 4.1 Shared evaluation setup

```python
# ── Task 4: Shared evaluation setup ──────────────────────────────────────────
# Prerequisites: Task 3 cells 1–6 must have run first.
# Inherits: X_train_t, X_val_t, y_train, y_val, SEED = 42

from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics      import (average_precision_score, roc_auc_score,
                                   recall_score, precision_score)
import numpy  as np
import pandas as pd

TOP_PCT = 0.20   # business constraint: retain top 20% highest-risk customers

# ── helper: ranking-based recall and precision ────────────────────────────────
def recall_precision_top(y, proba, top_pct=TOP_PCT):
    """Flag the top_pct highest-probability customers as positive.
    Returns (recall, precision) for that fixed-size bucket."""
    n_top   = max(1, int(len(y) * top_pct))
    top_idx = np.argsort(proba)[::-1][:n_top]
    tp      = float(y.iloc[top_idx].sum())
    return round(tp / float(y.sum()), 4), round(tp / float(n_top), 4)

# ── shared evaluation function ────────────────────────────────────────────────
def evaluate(name, model, X, y):
    """Evaluate a fitted classifier.  Uses predict_proba (not predict)
    so PR-AUC integrates the full curve.  Test set is NEVER passed here."""
    proba     = model.predict_proba(X)[:, 1]
    pr_auc    = round(average_precision_score(y, proba), 4)
    roc_auc   = round(roc_auc_score(y, proba), 4)
    rec, prec = recall_precision_top(y, proba)
    return {"Model": name, "PR-AUC": pr_auc, "ROC-AUC": roc_auc,
            "Recall@top20%": rec, "Precision@top20%": prec}

results = []   # accumulates one dict per model
print("evaluate() ready — 4 metrics: PR-AUC | ROC-AUC | Recall@top20% | Precision@top20%")
```

### 4.1  Shared Evaluation Setup

All models are scored on the **validation set** (`X_val_t`, `y_val`) using
four metrics:

| Metric | Role | Why this metric |
|--------|------|-----------------|
| **PR-AUC** | Primary | Best single number for imbalanced data — conditions on the positive class, so true negatives cannot inflate the score (Davis & Goadrich 2006) |
| **ROC-AUC** | Secondary | Threshold-free discrimination benchmark; useful but less sensitive under class imbalance |
| **Recall@top-20 %** | Business | Of all actual churners, what fraction falls in our top-20 % risk bucket? Directly models the retention campaign's capacity constraint |
| **Precision@top-20 %** | Business | Of the customers we flag, what fraction are genuine churners? High precision = fewer wasted interventions |

`evaluate()` always uses `predict_proba()[:, 1]` — never binary `predict()`.
This keeps model rankings fair and ensures PR-AUC integrates the full curve.

---

## Cell 2 — 4.2 Baselines

```python
# ── 4.2 Baselines ─────────────────────────────────────────────────────────────

# 1. Dummy — always predicts the majority class (non-churn).
#    Expected: PR-AUC ≈ churn prevalence (~0.20), ROC-AUC ≈ 0.50.
dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_train_t, y_train)
results.append(evaluate("Dummy (most_frequent)", dummy, X_val_t, y_val))

# 2. LogisticRegression — linear baseline with default settings.
#    Confirms features carry real signal before trying ensembles.
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
lr.fit(X_train_t, y_train)
results.append(evaluate("LogReg", lr, X_val_t, y_val))

print("── Baselines (validation set) ──")
display(pd.DataFrame(results))
```

### 4.2  Baselines

**Dummy (most\_frequent)** always predicts the majority class (Stay).
PR-AUC = 0.2040 (≈ churn prevalence), ROC-AUC = 0.50.  This is the
no-skill floor — every useful model must clearly exceed it.

**LogReg** jumps to PR-AUC = 0.5068 and ROC-AUC = 0.7846, confirming
the features carry real predictive signal and that even a simple linear
model ranks customers well above chance.  The sharp improvement over
Dummy (PR-AUC +0.30) establishes that there is meaningful signal in the
features; the question is whether non-linear models can exploit it
further.

---

## Cell 3 — 4.3 Tree ensemble and modern tabular model

```python
# ── 4.3 Tree ensemble + modern model ─────────────────────────────────────────

# 3. RandomForest — balanced_subsample re-weights each bootstrap sample.
#    200 trees for stable probability estimates; n_jobs=-1 for speed.
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_t, y_train)
results.append(evaluate("RandomForest", rf, X_val_t, y_val))

# 4. HistGradientBoosting — modern sklearn-native tabular model.
#    Histogram-binned O(n) splits; handles missing values natively.
#    class_weight="balanced" re-weights the gradient each iteration.
#    Default hyperparameters only — tuning is deferred to Task 5.
hgbt = HistGradientBoostingClassifier(
    max_iter=300,
    class_weight="balanced",
    random_state=SEED,
)
hgbt.fit(X_train_t, y_train)
results.append(evaluate("HistGBT", hgbt, X_val_t, y_val))

print("── Tree / modern models (validation set) ──")
display(pd.DataFrame(results[-2:]))
```

### 4.3  Tree Ensemble and Modern Tabular Model

**RandomForest** uses `balanced_subsample`, which re-weights classes
independently per bootstrap sample — more robust than a single global weight
for bagged ensembles.  200 trees give stable probability estimates on
~7 000 training rows.

**HistGradientBoosting** is the modern tabular approach in this comparison.
Histogram binning makes each boosting round O(n\_bins × n\_features);
`class_weight="balanced"` re-weights the gradient at every iteration.
All hyperparameters are sklearn defaults (except `max_iter=300` and
`class_weight`); tuning is deferred to Task 5 to keep this section a pure
model-selection exercise.

> **Why not XGBoost / LightGBM?**  Neither is in `requirements.txt`.
> `HistGradientBoostingClassifier` (sklearn ≥ 1.0) is the sklearn-native
> equivalent, avoiding an extra dependency.

*[fill: which model has the highest PR-AUC at this stage?  Does HistGBT
beat RandomForest?  Are both clearly above the Dummy floor?]*

---

## Cell 4 — 4.4 Full comparison, operating-rule demo, and shortlist

```python
# ── 4.4 Comparison table ──────────────────────────────────────────────────────
results_df = (
    pd.DataFrame(results)
    .sort_values("PR-AUC", ascending=False)
    .reset_index(drop=True)
)
print("=== Validation results — all models, sorted by PR-AUC ===")
display(results_df)

# ── Controlled comparison: operating rule (HistGBT) ──────────────────────────
# Why top-20% ranking beats a fixed 0.5 threshold for a capacity-constrained
# retention campaign: the campaign can contact exactly 20% of customers,
# so ranking fills every slot whereas threshold 0.5 may flag far fewer.
proba_hgbt = hgbt.predict_proba(X_val_t)[:, 1]

# (a) Threshold 0.5 — may under-flag on a 20% minority class
pred_05 = (proba_hgbt >= 0.5).astype(int)
n_flag_05 = int(pred_05.sum())

# (b) Top-20% ranking — always flags exactly n_top customers
n_top     = max(1, int(len(y_val) * TOP_PCT))
top_arr   = np.zeros(len(y_val), dtype=int)
top_arr[np.argsort(proba_hgbt)[::-1][:n_top]] = 1

rule_rows = []
for tag, preds in [("Threshold 0.5", pred_05), ("Top-20% ranking", top_arr)]:
    rule_rows.append({
        "Decision rule":  tag,
        "Flagged":        int(preds.sum()),
        "Recall":         round(recall_score(y_val, preds), 4),
        "Precision":      round(precision_score(y_val, preds, zero_division=0), 4),
    })
rule_df = pd.DataFrame(rule_rows)

print("\n── Operating-rule comparison (HistGBT, validation set) ──")
display(rule_df)
print(f"(Threshold 0.5 flags {n_flag_05} customers vs "
      f"top-20% always flags {n_top})")

# ── Shortlist ─────────────────────────────────────────────────────────────────
shortlist = (
    results_df[~results_df["Model"].str.startswith("Dummy")]
    .head(2)
    .reset_index(drop=True)
)
print("\n=== Shortlisted for Task 5 (top-2 non-Dummy by PR-AUC) ===")
display(shortlist)
```

### 4.4  Full Comparison and Operating-Rule Demo

**Validation results** (all models, one shared `evaluate()`, sorted by
PR-AUC):

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| *[fill from Cell 4 output]* | | | | |
| | | | | |
| | | | | |
| | | | | |

**Operating-rule comparison (HistGBT, validation set):**

| Decision rule | Flagged | Recall | Precision |
|---------------|---------|--------|-----------|
| Threshold 0.5 | *[fill]* | *[fill]* | *[fill]* |
| Top-20% ranking | *[fill]* | *[fill]* | *[fill]* |

The 0.5 threshold is calibrated for 50/50 class balance.  On a ~20 %
minority class it typically under-flags — leaving campaign slots empty
and churners uncaught.  The top-20 % ranking rule always fills every
available slot, matching the campaign's capacity constraint.

*[fill: how many more customers does top-20 % flag than threshold 0.5?
Does recall increase?  Is precision acceptable?]*

---

## 4.5  Shortlist Decision

*Markdown cell — fill after running all four cells.*

---

**Shortlisted for Task 5: RandomForest and HistGBT**

On the validation set, **RandomForest** leads with PR-AUC = **0.6957**
and **HistGBT** follows closely at **0.6952** — both roughly 0.19 points
above LogReg.  Both tree ensembles capture over 60 % of churners in the
top-20 % bucket, making them the clear candidates for tuning in Task 5.

LogReg (PR-AUC = 0.5068) is well above the Dummy floor but substantially
behind the tree models — the non-linear interactions that trees capture
(e.g. Age × NumOfProducts) cannot be recovered by a linear decision
boundary.

Both models were selected on **validation metrics only**; the test set
remains untouched for Task 5.

---

## 4.6  My Agent vs My Verification

| Step | What My Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Metric set | Agent initially proposed 3 metrics (PR-AUC, ROC-AUC, Recall@top-20%). I requested adding Precision@top-20% as a fourth metric to measure campaign cost-efficiency | I confirmed that Precision@top-20% is mechanically linked to Recall@top-20% under a fixed-bucket rule but tells a different business story (wasted interventions vs churner coverage). Updated Section 1.3 accordingly |
| Evaluation function | Defined `evaluate()` with 4 metrics and `recall_precision_top()` helper; uses `predict_proba` not `predict`; test set never passed in Task 4 | [fill: confirm Cell 1 runs without error; check dict keys match column names in the comparison table] |
| Baseline (Dummy) | `DummyClassifier(most_frequent)` fitted and evaluated; sets PR-AUC floor | [fill: confirm PR-AUC ≈ 0.20 and ROC-AUC ≈ 0.50 — if not, something is wrong] |
| Baseline (LogReg) | LogReg with default settings as linear baseline; confirms features carry signal above Dummy floor | PR-AUC = 0.5068 — sharp jump over Dummy (0.2040), confirming real signal exists. Establishes the linear ceiling that tree models must beat |
| Model set | Agent proposed Dummy + LogReg + RF + HistGBT (4 models). Dummy → LogReg → RF + HistGBT gives a clean no-skill → linear → ensemble → boosting progression | [fill: confirm all 4 models fit without error; all PR-AUCs above Dummy floor] |
| RandomForest | `balanced_subsample`, 200 trees, default depth; no tuning | [fill: PR-AUC above Dummy and LogReg? Or between them?] |
| HistGBT (modern) | `HistGradientBoostingClassifier` with `class_weight="balanced"`, default hyperparameters, no tuning | [fill: confirm this is the highest PR-AUC model; note that hyperparameters are untuned — tuning happens in Task 5] |
| Operating-rule demo | Threshold 0.5 vs top-20 % ranking for HistGBT; showed flagged count, recall, precision | [fill: does top-20 % improve recall? How many more customers are flagged? Is this the right rule for the retention campaign?] |
| Shortlist | Top-2 non-Dummy models by PR-AUC; 4.5 has evidence-based text with fill placeholders | [fill: do you agree with the two shortlisted models? Is there a reason to prefer a different second model?] |
| No tuning in Task 4 | Agent correctly deferred all hyperparameter tuning to Task 5 — Task 4 is a pure model-selection exercise with default parameters | I confirmed: no `RandomizedSearchCV` or `GridSearchCV` calls in Task 4 cells |
| *[add rows as needed]* | | |
