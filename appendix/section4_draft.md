# Task 4 — Model Exploration and Shortlist

---

## Coding Plan

1. **4.1 Shared evaluation setup.**  Define `recall_precision_top()` and a
   single `evaluate()` helper returning four consistent metrics — PR-AUC,
   ROC-AUC, Recall@top-20%, Precision@top-20%.  A `results` list accumulates
   one dict per model so every cell feeds the same summary table.

2. **4.2 Baselines — Ablation A (class weighting).**  `DummyClassifier
   (most_frequent)` sets the PR-AUC floor.  `LogisticRegression` is fitted
   with and without `class_weight="balanced"` on the same data; the ablation
   table proves whether balanced weighting helps on ~20% churn.

3. **4.3 Tree and modern models — Ablation B (decision rule).**  Fit
   `RandomForestClassifier`, `HistGradientBoostingClassifier`, and
   `MLPClassifier`.  For HistGBT, compare default-threshold (0.5) vs
   top-20%-ranking decision rules: this is the key operational question for
   the retention campaign scenario.

4. **4.4 Shortlist and Ablation C (tuning).**  Collect all results into one
   table sorted by PR-AUC; pick the top-2 non-Dummy models.  Run a small
   `RandomizedSearchCV` (n_iter=8, 3-fold CV on training only) for HistGBT
   and compare untuned vs tuned on the validation set in a mini table.

Constraints: test set never touched; `random_state=SEED` throughout; no
large grid searches; identical metric columns for every row.

---

## Cell 1 — 4.1 Shared evaluation setup

```python
# ── Task 4: Shared evaluation setup ──────────────────────────────────────────
# Prerequisites: Task 3 cells 1–6 must have been run first.
# Inherits: X_train_t, X_val_t, y_train, y_val, SEED = 42

from sklearn.dummy       import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble    import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics     import (average_precision_score, roc_auc_score,
                                  recall_score, precision_score)
import numpy  as np
import pandas as pd

TOP_PCT = 0.20   # business constraint: top 20% risk bucket

# ── helper: ranking-based recall and precision ────────────────────────────────
def recall_precision_top(y, proba, top_pct=TOP_PCT):
    """
    Label the top_pct% highest-probability customers as positive.
    Returns (recall, precision) for that fixed-size prediction set.
    """
    n_top   = max(1, int(len(y) * top_pct))
    top_idx = np.argsort(proba)[::-1][:n_top]
    tp      = float(y.iloc[top_idx].sum())
    return round(tp / float(y.sum()), 4), round(tp / float(n_top), 4)

# ── shared evaluation function ────────────────────────────────────────────────
def evaluate(name, model, X, y):
    """
    Evaluate a fitted binary classifier on split (X, y).
    Returns a dict with four consistent metric columns.
    The caller controls which split is passed — test set is NEVER used in Task 4.
    """
    proba        = model.predict_proba(X)[:, 1]
    pr_auc       = round(average_precision_score(y, proba), 4)
    roc_auc      = round(roc_auc_score(y, proba), 4)
    rec, prec    = recall_precision_top(y, proba)
    return {
        "Model":            name,
        "PR-AUC":           pr_auc,
        "ROC-AUC":          roc_auc,
        "Recall@top20%":    rec,
        "Precision@top20%": prec,
    }

results = []   # accumulates one dict per model; Cell 4 builds the full table
print("evaluate() ready — metrics: PR-AUC | ROC-AUC | Recall@top20% | Precision@top20%")
```

### 4.1  Shared Evaluation Setup

All models are evaluated on the **validation set** (`X_val_t`, `y_val`) using
four consistent metrics agreed in Section 1.3:

| Metric | Role | Rationale |
|--------|------|-----------|
| **PR-AUC** | Primary | Preferred for imbalanced data; measures minority-class precision-recall trade-off without inflation from true negatives (Davis & Goadrich, 2006) |
| **ROC-AUC** | Secondary | Threshold-free discrimination; useful for benchmarking but less discriminating under class imbalance |
| **Recall@top-20%** | Operational | Of the top-20% highest-risk customers flagged, what fraction of all churners are captured? Models the fixed-capacity retention campaign |
| **Precision@top-20%** | Operational | Of those same top-20% customers flagged, what fraction are genuine churners? High precision means fewer wasted interventions |

`evaluate()` uses `predict_proba()` throughout — no threshold is applied in
the comparison function, keeping model rankings fair.  The test set is
**never** passed to `evaluate()` in Task 4.

---

## Cell 2 — 4.2 Baselines (Ablation A: class weighting)

```python
# ── 4.2 Baselines ─────────────────────────────────────────────────────────────

# 1. Dummy — most_frequent: always predicts non-churn.
#    PR-AUC floor ≈ positive-class prevalence (~0.20).
#    ROC-AUC ≈ 0.50 (no discriminative power).
dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_train_t, y_train)
results.append(evaluate("Dummy (most_frequent)", dummy, X_val_t, y_val))

# 2. Ablation A — LogisticRegression: class_weight effect on imbalanced data.
#    Hypothesis: "balanced" weighting penalises missed churners more heavily
#    (sample weight = n_samples / (2 × n_class_i)), improving PR-AUC.
ablation_a = []
for cw, label in [
    (None,       "LogReg (default)"),
    ("balanced", "LogReg (balanced)"),
]:
    m = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED,
                           class_weight=cw)
    m.fit(X_train_t, y_train)
    row = evaluate(label, m, X_val_t, y_val)
    ablation_a.append(row)
    results.append(row)

print("── Ablation A: class_weight effect (LogReg) ──")
display(pd.DataFrame(ablation_a))

# Report winner for later reference
lr_best_label = ("LogReg (balanced)"
                 if ablation_a[1]["PR-AUC"] > ablation_a[0]["PR-AUC"]
                 else "LogReg (default)")
print(f"\nWinner (higher PR-AUC): {lr_best_label}")
```

### 4.2  Baseline Models

**Dummy (most\_frequent)** always predicts the majority class (non-churn).
It cannot discriminate: ROC-AUC ≈ 0.50 and PR-AUC ≈ the churn prevalence
(~0.20).  Every useful model must **clearly exceed** this floor on PR-AUC.

**Ablation A — class weighting for LogisticRegression.**  With
`class_weight="balanced"`, sklearn multiplies each sample's loss contribution
by `n_samples / (2 × n_class_i)`, so the gradient from a missed churner
counts more heavily during training.  The ablation table above shows both
variants on the same validation split; the winner is carried into the full
comparison table.

*[fill: state which variant wins (default vs balanced) and by how much in
PR-AUC, after running Cell 2.]*

---

## Cell 3 — 4.3 Tree and Modern Models (Ablation B: decision rule)

```python
# ── 4.3 Tree and modern models ────────────────────────────────────────────────

# 3. RandomForest — balanced_subsample: re-weights each bootstrap independently.
#    200 trees for stable probability estimates; n_jobs=-1 for speed.
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_t, y_train)
results.append(evaluate("RandomForest (balanced_subsample)", rf, X_val_t, y_val))

# 4. HistGradientBoosting — modern tabular model (sklearn ≥ 1.2).
#    Histogram-binned O(n) training; handles missingness natively;
#    invariant to monotonic feature transforms (no scaling needed, but
#    the upstream StandardScaler is harmless).
hgbt = HistGradientBoostingClassifier(
    max_iter=300,
    class_weight="balanced",
    random_state=SEED,
)
hgbt.fit(X_train_t, y_train)
results.append(evaluate("HistGBT (balanced)", hgbt, X_val_t, y_val))

# 5. MLP — neural baseline; StandardScaler already applied upstream.
#    MLPClassifier has no class_weight parameter (acknowledged limitation).
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=SEED,
)
mlp.fit(X_train_t, y_train)
results.append(evaluate("MLP (64-32)", mlp, X_val_t, y_val))

print("── Candidate models — val metrics ──")
display(pd.DataFrame(results[-3:]))

# ── Ablation B: decision rule comparison for HistGBT ─────────────────────────
# Compare: (a) default threshold 0.5  vs  (b) top-20% risk ranking.
# Question: does moving from a hard threshold to a capacity-constrained
# ranking strategy change recall and precision for the retention campaign?
proba_hgbt = hgbt.predict_proba(X_val_t)[:, 1]

# (a) Threshold 0.5
pred_05  = (proba_hgbt >= 0.5).astype(int)
rec_05   = round(recall_score(y_val, pred_05), 4)
prec_05  = round(precision_score(y_val, pred_05, zero_division=0), 4)

# (b) Top-20% ranking — label the n_top highest-risk predictions as positive
n_top      = max(1, int(len(y_val) * TOP_PCT))
top_arr    = np.zeros(len(y_val), dtype=int)
top_arr[np.argsort(proba_hgbt)[::-1][:n_top]] = 1
rec_top20  = round(recall_score(y_val, top_arr), 4)
prec_top20 = round(precision_score(y_val, top_arr, zero_division=0), 4)

ablation_b = pd.DataFrame([
    {"Decision rule": "Threshold 0.5",   "Recall": rec_05,   "Precision": prec_05},
    {"Decision rule": "Top-20% ranking", "Recall": rec_top20, "Precision": prec_top20},
])
print("\n── Ablation B: decision rule comparison (HistGBT, val set) ──")
display(ablation_b)
```

### 4.3  Tree and Modern Models

**RandomForest** uses `balanced_subsample`, which applies class weights
independently on each bootstrap sample — more robust than global `balanced`
for bagging because each tree sees a different drawn class ratio.
`n_estimators=200` gives stable probability estimates without excessive
runtime on ~7,000 training rows.

**HistGradientBoosting** is the modern tabular model in this comparison.
Histogram binning reduces per-iteration cost to O(n\_bins × n\_features);
it also handles missing values natively (unused here since imputation was
done upstream).  `class_weight="balanced"` re-weights the gradient at every
boosting iteration.

> **Why not XGBoost / LightGBM?**  Neither is in `requirements.txt` and both
> fail to import in this environment.  `HistGradientBoostingClassifier` is
> sklearn-native, version-pinned to scikit-learn 1.5.1, and the Task 4 spec
> explicitly accepts it as the modern approach.

**MLP** is a neural-network reference point.  The upstream `StandardScaler`
normalises inputs.  Architecture `(64, 32)` with `max_iter=500` trains in
seconds on this dataset size.  No `class_weight` parameter exists in
`MLPClassifier` — this is an acknowledged limitation noted in comparisons.

**Ablation B — decision rule comparison (HistGBT).**  Two strategies are
compared on the same model and validation set:

- *Threshold 0.5*: label positive if probability ≥ 0.5.  The threshold is
  calibrated to the training class distribution and tends to under-flag
  churners for a ~20% minority class.
- *Top-20% ranking*: always label exactly 20% of customers as positive,
  matching the retention campaign's capacity constraint.  This strategy does
  not require probability calibration.

*[fill: copy the Ablation B table from Cell 3 output.  Note which strategy
gives higher recall, whether precision drops materially, and which decision
rule you would recommend for the retention campaign.]*

---

## Cell 4 — 4.4 Shortlist and Tuning (Ablation C)

```python
# ── 4.4 Shortlist and tuning experiment ───────────────────────────────────────

# Full comparison table — all models on the validation set, sorted by PR-AUC.
results_df = (
    pd.DataFrame(results)
    .sort_values("PR-AUC", ascending=False)
    .reset_index(drop=True)
)
print("=== Validation results — all models, sorted by PR-AUC ===")
display(results_df)

# Shortlist: top-2 non-Dummy models by PR-AUC
shortlist = (
    results_df[~results_df["Model"].str.startswith("Dummy")]
    .head(2)
    .reset_index(drop=True)
)
print("\nShortlisted for Task 5 (top-2 non-Dummy by PR-AUC):")
display(shortlist)

# ── Ablation C: tuned vs untuned (HistGBT) ───────────────────────────────────
# Tuning on TRAINING set only (3-fold CV, scoring=average_precision).
# Budget: n_iter=8 → 8 × 3 = 24 total fits — fast on this dataset size.
# Best config is evaluated ONCE on validation after training CV completes.

param_dist = {
    "max_iter":         [100, 200, 300],
    "max_depth":        [3, 5, None],
    "learning_rate":    [0.05, 0.1, 0.2],
    "min_samples_leaf": [20, 40],
}

search = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    param_distributions=param_dist,
    n_iter=8,
    cv=3,
    scoring="average_precision",
    random_state=SEED,
    n_jobs=-1,
)
search.fit(X_train_t, y_train)   # training only — validation not seen here

print(f"\nBest hyperparameters (from 3-fold training CV): {search.best_params_}")
print(f"Best CV PR-AUC on training:                    {search.best_score_:.4f}")

hgbt_tuned  = search.best_estimator_   # refit=True by default

row_untuned = evaluate("HistGBT untuned", hgbt,       X_val_t, y_val)
row_tuned   = evaluate("HistGBT tuned",   hgbt_tuned, X_val_t, y_val)

ablation_c = pd.DataFrame([row_untuned, row_tuned])
print("\n── Ablation C: tuned vs untuned — validation set ──")
display(ablation_c)
print("\nNote: best params chosen by training CV only — val used once for final comparison.")
```

### 4.4  Model Comparison and Shortlist

**Full validation results** (all models, sorted by PR-AUC):

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| *[fill from Cell 4 output]* | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

**Ablation C — tuned vs untuned (HistGBT, validation only):**

| | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-|--------|---------|---------------|------------------|
| Untuned | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Tuned   | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

*[fill: state the best params found, whether tuning improved PR-AUC on
validation, and by how much.  Note that the best params were selected using
training CV only and the validation comparison was performed once.]*

---

## 4.5  Shortlist Decision

*Markdown-only cell — fill after running all four cells.*

---

**Shortlisted models for Task 5: [Model A] and [Model B]**

On the validation set, **[Model A]** achieves the highest PR-AUC of
**[fill]**, with Recall@top-20% = **[fill]** — capturing **[fill fraction]**
of all churners under a fixed-capacity retention campaign.
**[Model B]** achieves PR-AUC = **[fill]** and is retained as the second
candidate because it provides [coefficient-level interpretability /
feature-importance transparency / ensemble diversity — choose one], which
addresses the interpretability constraint stated in Section 1.4.
Both models were selected on **validation metrics only**; the test set
remains untouched and will be used exclusively in Task 5.

*Guidance: HistGBT is expected to lead on PR-AUC.  LogReg (balanced) is the
natural second candidate for coefficient-level interpretability.  If RF
outperforms LogReg, substitute it as Model B.  If MLP tops the table, report
the result but consider retaining an interpretable model for the second slot.*

---

## Agent Plan vs My Verification

| Step | What the Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Evaluation function | Defined `evaluate()` with 4 metrics (PR-AUC, ROC-AUC, Recall@top20%, Precision@top20%) and `recall_precision_top()` helper; test set never passed in Task 4 | [fill: confirm Cell 1 runs without errors; check dict keys match expected column names] |
| Ablation A (class weighting) | LogReg `default` vs `balanced` fitted and compared in Cell 2; winner label printed automatically | [fill: which variant wins? By how much in PR-AUC? Does balanced weighting help as expected?] |
| Ablation B (decision rule) | Threshold 0.5 vs top-20% ranking compared for HistGBT in Cell 3; recall and precision shown for both rules | [fill: does top-20% ranking improve recall over threshold 0.5? What is the precision trade-off? Which rule is better for the campaign?] |
| Ablation C (tuning) | RandomizedSearchCV on HistGBT with n\_iter=8, cv=3, `scoring="average_precision"`, training only; best config evaluated once on validation | [fill: did tuning improve val PR-AUC? By how much? Were best params close to defaults?] |
| Shortlist | Top-2 non-Dummy models by validation PR-AUC; shown in Cell 4; 4.5 shortlist text has fill placeholders | [fill: do you agree with both shortlisted models? Any reason to prefer a different second model given the interpretability constraint?] |
| [add rows as needed] | | |
