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
   for a fixed-capacity retention campaign.  Shortlist the two
   best-performing models (by PR-AUC, excl. Dummy floor) for Task 5
   — their gap is within noise, so deeper Task 5 diagnostics will
   decide the final pick.

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

# 3. RandomForest — sklearn defaults (no class_weight, no tuning).
#    200 trees for stable probability estimates; n_jobs=-1 for speed.
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=SEED,
    n_jobs=-1,
)
rf.fit(X_train_t, y_train)
results.append(evaluate("RandomForest", rf, X_val_t, y_val))

# 4. HistGradientBoosting — modern sklearn-native tabular model.
#    Histogram-binned O(n) splits; handles missing values natively.
#    All defaults (no class_weight) — tuning is deferred to Task 5.
hgbt = HistGradientBoostingClassifier(
    random_state=SEED,
)
hgbt.fit(X_train_t, y_train)
results.append(evaluate("HistGBT", hgbt, X_val_t, y_val))

print("── Tree / modern models (validation set) ──")
display(pd.DataFrame(results[-2:]))
```

### 4.3  Tree Ensemble and Modern Tabular Model

**RandomForest** uses sklearn defaults — no class reweighting, no depth
limit, 200 trees for stable probability estimates on ~7 000 training rows.

**HistGradientBoosting** is the modern tabular approach in this comparison.
Histogram binning makes each boosting round O(n\_bins × n\_features).
All hyperparameters are sklearn defaults; tuning — including whether
`class_weight="balanced"` helps — is deferred to Task 5 to keep this
section a pure model-architecture comparison.

> **Why not XGBoost / LightGBM?**  Neither is in `requirements.txt`.
> `HistGradientBoostingClassifier` (sklearn ≥ 1.0) is the sklearn-native
> equivalent, avoiding an extra dependency.

Both tree ensembles leap well above LogReg: **HistGBT** PR-AUC = **0.7252**,
**RandomForest** PR-AUC = **0.7042** — each roughly +0.20 over the linear
baseline and +0.50 over the no-skill floor (0.2040).  HistGBT leads by
**0.0210** — a small but non-trivial gap.  Both are carried forward to
Task 5, where tuning and error analysis will confirm whether HistGBT's
validation-set lead holds on the test set.

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
# Dummy is the no-skill floor (PR-AUC ≈ 0.20) — stated in text, hidden
# from table to keep the comparison focused on real contenders.
display(results_df[~results_df["Model"].str.startswith("Dummy")])

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
gap = abs(shortlist.loc[0, "PR-AUC"] - shortlist.loc[1, "PR-AUC"])
print(f"\n=== Shortlist for Task 5 (top 2 by PR-AUC, excl. Dummy floor) ===")
display(shortlist)
print(f"PR-AUC gap between top 2: {gap:.4f} — within noise for ~{len(y_val)}-row val set.")
```

### 4.4  Full Comparison and Operating-Rule Demo

**Validation results** (sorted by PR-AUC; Dummy floor omitted —
PR-AUC ≈ 0.20, ROC-AUC = 0.50):

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT | 0.7252 | 0.8781 | 0.6471 | 0.66 |
| RandomForest | 0.7042 | 0.8722 | 0.6373 | 0.65 |
| LogReg | 0.5068 | 0.7846 | 0.6338 | 0.6338 |

**Operating-rule comparison (HistGBT, validation set):**

| Decision rule | Flagged | Recall | Precision |
|---------------|---------|--------|-----------|
| Threshold 0.5 | 199 | 0.4837 | 0.7437 |
| Top-20% ranking | 300 | 0.6471 | 0.6600 |

The 0.5 threshold is calibrated for 50/50 class balance. On a ~20%
minority class it drastically under-flags — identifying only 199
customers versus the 300 that the top-20% rule always selects. This
leaves a third of the campaign's capacity unused and misses over half
of the actual churners (recall 0.48 vs 0.65). The top-20% ranking
rule always fills every available slot, matching the retention
campaign's fixed-capacity constraint, and raises recall by +0.16 at
the cost of lower precision (0.66 vs 0.74). In a retention context,
the cost of a wasted offer (false positive) is far lower than the cost
of losing a customer (false negative), so the recall gain clearly
justifies the precision trade-off.

---

## 4.5  Shortlist Decision

*Markdown cell — fill after running all four cells.*

---

**Shortlisted for Task 5: HistGBT and RandomForest**

On the validation set the two tree ensembles clearly dominate:
**HistGBT** PR-AUC = **0.7252**, **RandomForest** PR-AUC = **0.7042** —
both roughly +0.20 above LogReg and +0.50 above the no-skill floor.
The gap between them is **0.0210** — small in absolute terms but
non-negligible.  HistGBT leads on validation, but validation rankings
do not always transfer to the test set.

**Why two, not one:**  The two models have structurally different
learning mechanisms — RF bags independent trees (variance reduction),
HistGBT sequentially corrects residuals (bias reduction) — so they are
likely to differ on calibration quality and subgroup (geography)
performance.  Carrying both into Task 5 lets the error analysis
(calibration curve, geography slice, confusion matrix) confirm whether
HistGBT's validation lead holds and whether it is also the fairer and
better-calibrated model.

**Pre-committed decision criterion for Task 5 final selection:**

- **Decision criterion:** highest **validation** PR-AUC after tuning
  (Cell 5.3).  The test set is not accessed until after the decision is
  locked.

**Post-decision diagnostics (cannot override the locked choice):**

- **Calibration quality** — which model's reliability curve is closer to
  the diagonal (Cell 5.5c)
- **Geography-slice fairness** — smallest max–min PR-AUC gap across
  countries (Cell 5.5d)
- **Tiebreaker** (only if validation PR-AUC is identical) — HistGBT
  preferred on engineering grounds (native missing-value handling, richer
  tuning surface)

LogReg (PR-AUC = 0.5068) is well above the no-skill floor (0.2040) but
substantially behind the tree models — the non-linear interactions that
trees capture (e.g. Age × NumOfProducts) cannot be recovered by a linear
decision boundary.

Selection will be made on **validation metrics only**; the test set
remains untouched until Task 5 Cell 5.4 (report only).

---

## 4.6  My Agent vs My Verification

| Step | What My Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Metric set | Agent initially proposed 3 metrics (PR-AUC, ROC-AUC, Recall@top-20%). I requested adding Precision@top-20% as a fourth metric to measure campaign cost-efficiency | I confirmed that Precision@top-20% is mechanically linked to Recall@top-20% under a fixed-bucket rule but tells a different business story (wasted interventions vs churner coverage). Updated Section 1.3 accordingly |
| Evaluation function | Defined `evaluate()` with 4 metrics and `recall_precision_top()` helper; uses `predict_proba` not `predict`; test set never passed in Task 4 | Cell 1 runs without error. Dict keys (Model, PR-AUC, ROC-AUC, Recall@top20%, Precision@top20%) match comparison table columns exactly |
| Baseline (Dummy) | `DummyClassifier(most_frequent)` fitted and evaluated; sets PR-AUC floor | PR-AUC = 0.2040 (≈ churn prevalence), ROC-AUC = 0.50 — both as expected for a no-skill classifier |
| Baseline (LogReg) | LogReg with default settings as linear baseline; confirms features carry signal above Dummy floor | PR-AUC = 0.5068 — sharp jump over Dummy (0.2040), confirming real signal exists. Establishes the linear ceiling that tree models must beat |
| Model set | Agent proposed Dummy + LogReg + RF + HistGBT (4 models). Dummy (floor) → LogReg → RF + HistGBT gives a clean no-skill → linear → ensemble → boosting progression. Dummy is stated as the floor but hidden from comparison tables to keep them focused | All 4 models fit without error. PR-AUCs: Dummy 0.2040, LogReg 0.5068, RF 0.7042, HistGBT 0.7252 — all above the no-skill floor |
| RandomForest | sklearn defaults, 200 trees, no `class_weight`; no tuning | PR-AUC = 0.7042 — well above LogReg (0.5068), +0.20 above the linear baseline. Second-best model |
| HistGBT (modern) | `HistGradientBoostingClassifier` with sklearn defaults (no `class_weight`), no tuning | PR-AUC = 0.7252 — highest model, +0.021 above RF. Hyperparameters are untuned defaults; tuning deferred to Task 5 |
| Operating-rule demo | Threshold 0.5 vs top-20 % ranking for HistGBT; showed flagged count, recall, precision | Top-20% flags 300 customers vs threshold 0.5 flags only 199 (+101). Recall jumps 0.4837 → 0.6471 (+0.16). Top-20% is the correct rule: it fills the campaign's 300-slot capacity and catches 65% of churners vs only 48% |
| Shortlist | Agent initially shortlisted HistGBT as the single best model. I reversed this to carry **both HistGBT and RF** into Task 5, because the PR-AUC gap (0.0210) is small and the two models have structurally different failure modes | I confirmed: (1) HistGBT leads by 0.021 on validation — non-trivial but validation rankings don't always transfer to test; (2) the two models have structurally different learning mechanisms (bagging vs boosting) so they will differ on calibration and subgroup performance; (3) Task 5 error analysis (calibration, geography slice) provides diagnostics to confirm whether HistGBT's lead holds; (4) pre-committed validation PR-AUC as the sole decision criterion before running Task 5; calibration and geography checked as post-decision diagnostics |
| No tuning in Task 4 | Agent correctly deferred all hyperparameter tuning to Task 5 — Task 4 is a pure model-selection exercise with default parameters | I confirmed: no `RandomizedSearchCV` or `GridSearchCV` calls in Task 4 cells |
| `class_weight` removed | Agent originally set `class_weight="balanced_subsample"` (RF) and `"balanced"` (HistGBT). I flagged that `class_weight` is a hyperparameter — setting it contradicts the "all defaults" design. Stripped from both models; `class_weight` is now searched in Task 5 `param_dist` instead | I verified: (1) `class_weight` is not a structural choice, it is a hyperparameter tunable via `GridSearchCV`; (2) at ~20% imbalance, the LogReg ablation already showed weighting has near-null effect; (3) moving it to Task 5 gives a cleaner Task 4 narrative and a stronger tuning story |
| *[add rows as needed]* | | |
