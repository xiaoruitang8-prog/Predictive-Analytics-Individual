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

4. **4.4 Full comparison and shortlist.**  One sorted table of all models.
   Interpret all four metrics across architectures.  Shortlist the single
   best-performing model (HistGBT, by PR-AUC) for tuning — it leads on
   all four metrics at defaults, so carrying a second model adds complexity
   without changing the outcome.  The operating-rule comparison (ranking vs.
   threshold 0.5) is deferred to a later section.

Constraints: **no hyperparameter search** in Task 4 — all models use
sensible defaults.  Test set **never touched**.  `random_state=SEED`
throughout.  Tuning is deferred to a subsequent task.

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

All models are scored on the **validation set** using four metrics:
PR-AUC (primary — best single number for imbalanced data), ROC-AUC
(secondary discrimination check), Recall@top-20 % (churner coverage
within the campaign's fixed capacity), and Precision@top-20 %
(wasted-intervention rate).  `evaluate()` uses `predict_proba()`, not
`predict()`, so PR-AUC integrates the full precision–recall curve.  The
test set is not used in Task 4.

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

**Dummy (most\_frequent)** always predicts Stay, so every customer
receives the same score and the model cannot distinguish churners from
non-churners.  Its PR-AUC of 0.2040 (≈ churn prevalence) and ROC-AUC
of 0.50 define the no-skill floor that every real model must beat.
Because the Dummy assigns identical probabilities to all customers, its
Recall@top-20 % and Precision@top-20 % reflect pure chance: the top-20 %
bucket is effectively a random sample, so it captures roughly 20 % of
churners — far too few to justify a targeted campaign.

**LogReg** (default settings) raises PR-AUC to 0.5068 — a +0.30 jump
that confirms the features carry genuine linear signal.  Its ROC-AUC of
0.7846 shows strong overall discrimination: the model assigns higher
probabilities to churners than to stayers in roughly 78 % of randomly
drawn churner–stayer pairs.  However, on the business metrics that
matter most for a capacity-constrained campaign, LogReg's advantage over
Dummy is surprisingly narrow: Recall@top-20 % = 0.6338 and
Precision@top-20 % = 0.6338, meaning the top-ranked fifth of customers
captures about 63 % of actual churners and roughly two in three flagged
customers are genuine churners.  These figures set the linear ceiling;
the question is whether non-linear models can push the ranking quality
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
#    All defaults (no class_weight) — tuning is deferred.
hgbt = HistGradientBoostingClassifier(
    random_state=SEED,
)
hgbt.fit(X_train_t, y_train)
results.append(evaluate("HistGBT", hgbt, X_val_t, y_val))

print("── Tree / modern models (validation set) ──")
display(pd.DataFrame(results[-2:]))
```

### 4.3  Tree Ensemble and Modern Tabular Model

**RandomForest** (200 trees, sklearn defaults) and
**HistGradientBoosting** (sklearn-native histogram-based boosting,
defaults) are both fitted on training data and evaluated on validation.
All hyperparameters are untuned defaults; tuning is deferred to a later
task to keep this section a pure architecture comparison.  RF uses 200
trees — a standard default for stable probability estimates.

Both tree ensembles leap well above the linear baseline on every metric.
On PR-AUC — the primary measure of ranking quality across the full
precision–recall trade-off — HistGBT reaches **0.7252** and RF
**0.7042**, each roughly +0.20 above LogReg's 0.5068.  This confirms
that the non-linear interaction structure in the data (e.g. Age ×
Geography, Balance × NumOfProducts) rewards tree-based architectures.
ROC-AUC tells a consistent story: HistGBT achieves 0.8781 and RF
0.8722, indicating that both models correctly order churner–stayer pairs
roughly 87 % of the time — a substantial improvement over LogReg's
0.7846 and well into the "good discrimination" range.

Turning to the business-level metrics that translate directly to
campaign performance: HistGBT's Recall@top-20 % of 0.6471 means that
if the bank contacts the top-ranked fifth of customers, the campaign
would reach nearly two-thirds of all actual churners — a meaningful
improvement over LogReg's 0.6338 and a dramatic one over the Dummy
baseline.  RF is close behind at 0.6373.  Meanwhile, Precision@top-20 %
(HistGBT 0.6600, RF 0.6500) shows that roughly two in three customers
flagged for retention would indeed have churned, keeping the
wasted-intervention rate at about one in three — an acceptable cost
given that missing a churner is typically more expensive than one
unnecessary retention call.

---

## Cell 4 — 4.4 Full comparison and shortlist

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

# ── Shortlist ─────────────────────────────────────────────────────────────────
# HistGBT leads on all four metrics — shortlist it alone for tuning.
SHORTLISTED = "HistGBT"
print(f"\n=== Shortlisted for tuning: {SHORTLISTED} ===")
print("Rationale: HistGBT leads on all four metrics at defaults; "
      "no second model is close enough to justify a two-model tuning run.")
```

### 4.4  Full Comparison and Shortlist

**Validation results** (sorted by PR-AUC; Dummy floor omitted —
PR-AUC ≈ 0.20, ROC-AUC = 0.50):

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT | 0.7252 | 0.8781 | 0.6471 | 0.6600 |
| RandomForest | 0.7042 | 0.8722 | 0.6373 | 0.6500 |
| LogReg | 0.5068 | 0.7846 | 0.6338 | 0.6338 |

All models are evaluated under a top-20 % ranking rule, reflecting the
campaign's fixed capacity; the formal operating-rule comparison
(ranking vs. threshold 0.5) is deferred to a later section.

The table reveals a clear two-tier structure.  On the primary metric,
PR-AUC, both tree ensembles (HistGBT 0.7252, RF 0.7042) sit roughly
+0.20 above LogReg (0.5068), indicating that non-linear models produce
substantially better precision–recall trade-offs across the full
ranking.  ROC-AUC confirms this pattern from a different angle: HistGBT
(0.8781) and RF (0.8722) both exceed 0.87, meaning each model correctly
orders a randomly drawn churner above a randomly drawn stayer roughly
87 % of the time — a nine-percentage-point lead over LogReg's 0.7846.
The consistency between PR-AUC and ROC-AUC is reassuring: both summary
measures agree that the tree ensembles discriminate much better than the
linear baseline, even though PR-AUC is more sensitive to performance on
the minority class.

The business-level metrics are more compressed across models, which is
expected: all three contenders produce reasonable rankings, but the
top-20 % bucket can only improve so much.  HistGBT's Recall@top-20 %
of 0.6471 means its top-ranked fifth captures nearly two-thirds of all
churners — a 1.3-percentage-point edge over LogReg's 0.6338 and a
1.0-point edge over RF's 0.6373.  In absolute terms, HistGBT's top
bucket catches roughly 99 churners out of 153 in validation (versus
RF's 97 and LogReg's 97), so the practical gap is a handful of
customers.  However, when scaled to the full customer base, even a
small improvement in recall compounds — each additional correctly
identified churner is one more retention intervention that could prevent
revenue loss.  Precision@top-20 % (HistGBT 0.6600, RF 0.6500, LogReg
0.6338) tells the cost side of the story: at HistGBT's precision, about
one in three flagged customers would not actually churn, representing
wasted outreach.  This false-alarm rate is acceptable for a retention
campaign where the cost of contacting a non-churner is low relative to
the cost of losing a genuine churner.

HistGBT leads on all four metrics: PR-AUC (+0.021 over RF), ROC-AUC
(+0.006), Recall@top-20 % (+0.010), and Precision@top-20 % (+0.010).
While the margins are individually modest on a ~1,500-row validation
set, the fact that HistGBT wins on every measure — both the full-curve
summaries and the business-level operating-point metrics — gives
consistent evidence rather than a single noisy signal.  **HistGBT is
shortlisted alone for tuning.**  Carrying RandomForest forward would add
complexity (doubling the tuning budget) without a realistic prospect of
overtaking HistGBT, since RF trails on every metric at defaults.  The
decision criterion for the final lock remains pre-committed: validation
PR-AUC after tuning.

---

## 4.5  Shortlist Decision

HistGBT is carried forward as the sole candidate for tuning.  It leads
on all four validation metrics at defaults; no second model is close
enough to justify a two-model tuning run.  The final lock criterion
remains **validation PR-AUC after tuning**.  Calibration and geography
checks serve as post-decision diagnostics only.

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
| HistGBT (modern) | `HistGradientBoostingClassifier` with sklearn defaults (no `class_weight`), no tuning | PR-AUC = 0.7252 — highest model, +0.021 above RF. Hyperparameters are untuned defaults; tuning deferred to a subsequent task |
| Operating-rule demo | Agent originally included a threshold-0.5 vs top-20 % comparison in Task 4. I moved it to a later section, where it belongs — Task 4's role is architecture comparison, not operating-rule selection | Task 4.4 now contains a one-sentence deferral. The full three-rule comparison (top-20 % ranking, threshold ≈ 20 %, threshold 0.5) with recall/precision numbers lives in a dedicated operating-rule section |
| Shortlist | Agent initially shortlisted HistGBT as the single best model | I confirmed HistGBT leads on all four metrics at defaults (PR-AUC +0.021, ROC-AUC +0.006, Recall +0.010, Precision +0.010 over RF). Consistent advantage across every measure justifies a single-model shortlist — carrying RF forward would double the tuning budget without a realistic prospect of changing the outcome |
| No tuning in Task 4 | Agent correctly deferred all hyperparameter tuning — Task 4 is a pure model-selection exercise with default parameters | I confirmed: no `RandomizedSearchCV` or `GridSearchCV` calls in Task 4 cells |
| `class_weight` removed | Agent originally set `class_weight="balanced_subsample"` (RF) and `"balanced"` (HistGBT). I flagged that `class_weight` is a hyperparameter — setting it contradicts the "all defaults" design. Stripped from both models; `class_weight` is now searched in the tuning `param_dist` instead | I verified: (1) `class_weight` is not a structural choice, it is a hyperparameter tunable via `GridSearchCV`; (2) at ~20% imbalance, the LogReg ablation already showed weighting has near-null effect; (3) deferring it to the tuning task gives a cleaner Task 4 narrative and a stronger tuning story |
| `n_estimators=200` | Agent set RF to 200 trees without formal justification beyond "stable probability estimates" | I verified: (1) 200 is a standard sklearn community default for medium-sized datasets; (2) the tuning `param_dist` searches over `[100, 200, 300]`, so the choice is validated empirically during tuning |
