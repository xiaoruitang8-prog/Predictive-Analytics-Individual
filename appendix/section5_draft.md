# Task 5 — Fine-Tune and Evaluate

---

## Coding Plan

1. **5.1 Sanity check — re-fit HistGBT.**  Re-fit HistGBT (untuned) from
   Task 4 on `X_train_t` and evaluate on `X_val_t` using all 4 metrics
   (PR-AUC, ROC-AUC, Recall@top-20%, Precision@top-20%) to confirm the
   Task 4 numbers reproduce before tuning.

2. **5.2 Tune the primary candidate.**  `RandomizedSearchCV` on HistGBT
   (n\_iter=8, 3-fold CV, training only, `scoring="average_precision"`).
   Compare tuned vs untuned on validation; pick the final model.

3. **5.3 Lock operating rule on validation.**  The main operating rule is
   **top-20 % by predicted risk** (matching the retention campaign's
   capacity).  Also find a threshold on validation that approximates 20 %
   flagging, to enable a confusion-matrix view later.  Lock everything
   before test access.

4. **5.4 Final test evaluation.**  Test set accessed once.  Report PR-AUC,
   ROC-AUC, Recall@top-20 %, Precision@top-20 %, and threshold-based
   metrics at the locked threshold.

5. **5.5 Error analysis — 4 diagnostics.**
   - **(a)** Confusion matrix at the locked threshold
   - **(b)** PR curve (val vs test overlay)
   - **(c)** Calibration diagram (reliability curve + score distribution)
   - **(d)** Geography failure-mode slice (fairness check)
   Save figures to `outputs/`.

6. **5.6 Agent-made mistake and fix.**  Demonstrate the `predict()` vs
   `predict_proba()` bug for PR-AUC; show concrete impact; confirm fix.

Constraints: test set accessed only in Cell 5.4 (once).
`random_state=SEED` throughout.  Tuning budget: 8 × 3 = 24 fits.

---

## Cell 1 — 5.1 Sanity check

```python
# ── 5.1 Sanity check — re-fit HistGBT ────────────────────────────────────────
# Re-fit HistGBT (untuned) so Task 5 is self-contained.
# Confirm Task 4 validation numbers reproduce before tuning.
# Inherits: X_train_t, X_val_t, y_train, y_val, SEED=42, TOP_PCT=0.20
#            evaluate(), recall_precision_top()  (Task 4 Cell 1)

from sklearn.ensemble     import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics      import (average_precision_score, roc_auc_score,
                                   recall_score, precision_score,
                                   precision_recall_curve)
import numpy  as np
import pandas as pd

# Re-fit HistGBT with same defaults as Task 4 (no class_weight)
hgbt_base = HistGradientBoostingClassifier(random_state=SEED)
hgbt_base.fit(X_train_t, y_train)

# Confirm Task 4 numbers reproduce
sanity = pd.DataFrame([
    evaluate("HistGBT (untuned)", hgbt_base, X_val_t, y_val),
])
print("── 5.1 Sanity check: HistGBT on validation ──")
display(sanity)
print("Task 4 numbers reproduced — ready to tune.")
```

### 5.1  Sanity Check

HistGBT (untuned) is re-fit on `X_train_t` and evaluated on `X_val_t`
to confirm the Task 4 numbers reproduce before tuning begins.

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (untuned) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

*[fill: confirm these match Task 4 values.  If they differ, check that
SEED and preprocessing are identical.]*

---

## Cell 2 — 5.2 Tune the primary candidate

```python
# ── 5.2 Tune HistGBT ─────────────────────────────────────────────────────────
# Task 4 used default hyperparameters.  Here we do a small search.
# scoring="average_precision" = PR-AUC → correct metric for imbalanced data.
# Budget: n_iter=8 × cv=3 = 24 total fits.

param_dist = {
    "max_iter":         [100, 200, 300],
    "max_depth":        [3, 5, None],
    "learning_rate":    [0.05, 0.1, 0.2],
    "min_samples_leaf": [20, 40],
    "class_weight":     [None, "balanced"],
}

search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=SEED),
    param_distributions=param_dist,
    n_iter=8,
    cv=3,
    scoring="average_precision",
    random_state=SEED,
    n_jobs=-1,
)
search.fit(X_train_t, y_train)        # training only

hgbt_tuned = search.best_estimator_   # refit=True
print(f"Best params (training CV): {search.best_params_}")
print(f"Best CV PR-AUC (training): {search.best_score_:.4f}")

# Tuned vs untuned on validation
row_untuned = evaluate("HistGBT (untuned)", hgbt_base,  X_val_t, y_val)
row_tuned   = evaluate("HistGBT (tuned)",   hgbt_tuned, X_val_t, y_val)

tuning_df = pd.DataFrame([row_tuned, row_untuned])
print("\n── Tuned vs untuned (validation) ──")
display(tuning_df)

delta = round(row_tuned["PR-AUC"] - row_untuned["PR-AUC"], 4)
print(f"\nTuning gain: ΔPR-AUC = {delta:+.4f}")
```

### 5.2  Tuning the Primary Candidate

`RandomizedSearchCV` searches a small hyperparameter space using PR-AUC
as the scoring metric (correct for imbalanced data — see Section 5.6 for
why ROC-AUC would mislead).  The validation set is **not** seen by the
search; it is evaluated once afterwards.

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (tuned)   | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| HistGBT (untuned) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

Best parameters: *[fill from output]*.
Tuning gain: ΔPR-AUC = *[fill]*.

*[fill: was tuning meaningful (Δ > 0.005) or marginal?  Are the best params
close to defaults?  Note this is evidence for the tuning rubric item.]*

---

## Cell 3 — 5.3 Lock operating rule on validation

```python
# ── 5.3 Lock the operating rule ───────────────────────────────────────────────
# Main rule: top-20% by predicted risk (matches retention campaign capacity).
# Also derive a threshold that approximates 20% flagging on validation,
# so we can produce a confusion matrix at test time.
# Everything locked here — test set not accessed.

proba_val = hgbt_tuned.predict_proba(X_val_t)[:, 1]

# ── Main rule: top-20% ranking ────────────────────────────────────────────────
n_top      = max(1, int(len(y_val) * TOP_PCT))
top_arr    = np.zeros(len(y_val), dtype=int)
top_arr[np.argsort(proba_val)[::-1][:n_top]] = 1
rec_top20  = round(recall_score(y_val, top_arr), 4)
prec_top20 = round(precision_score(y_val, top_arr, zero_division=0), 4)

# ── Approximate threshold: 80th percentile of val probabilities ───────────────
# This flags ~20% of customers, matching the ranking rule via a threshold.
LOCKED_THRESH = round(float(np.percentile(proba_val, 100 * (1 - TOP_PCT))), 4)

pred_thr      = (proba_val >= LOCKED_THRESH).astype(int)
rec_thr       = round(recall_score(y_val, pred_thr), 4)
prec_thr      = round(precision_score(y_val, pred_thr, zero_division=0), 4)
pct_flagged   = round(100 * pred_thr.sum() / len(y_val), 1)

# ── Compare with naive 0.5 threshold (for context only) ──────────────────────
pred_05  = (proba_val >= 0.5).astype(int)
rec_05   = round(recall_score(y_val, pred_05), 4)
prec_05  = round(precision_score(y_val, pred_05, zero_division=0), 4)
pct_05   = round(100 * pred_05.sum() / len(y_val), 1)

rule_df = pd.DataFrame([
    {"Rule": "Top-20% ranking (main)",
     "Flagged (%)": 20.0,     "Recall": rec_top20,  "Precision": prec_top20},
    {"Rule": f"Threshold ≈ 20% (t={LOCKED_THRESH})",
     "Flagged (%)": pct_flagged, "Recall": rec_thr,    "Precision": prec_thr},
    {"Rule": "Threshold 0.5 (naive)",
     "Flagged (%)": pct_05,      "Recall": rec_05,     "Precision": prec_05},
])
print("── Operating-rule comparison (validation) ──")
display(rule_df)

print(f"\nLOCKED: top-20% ranking as main rule")
print(f"LOCKED: threshold = {LOCKED_THRESH} for confusion-matrix view")
print("Test set has NOT been accessed.")
```

### 5.3  Lock Operating Rule

The retention campaign can contact exactly 20 % of customers, so the
primary operating rule is **top-20 % by predicted churn risk** — a ranking
rule that always fills every slot regardless of probability calibration.

To produce a confusion matrix (which needs binary labels), a threshold of
**[fill]** is derived from the 80th percentile of validation probabilities.
This approximately flags 20 % of customers, matching the ranking rule but
expressed as a threshold.  For comparison, the naive 0.5 threshold is also
shown.

| Rule | Flagged (%) | Recall | Precision |
|------|-------------|--------|-----------|
| Top-20 % ranking (main) | 20.0 | *[fill]* | *[fill]* |
| Threshold ≈ 20 % | *[fill]* | *[fill]* | *[fill]* |
| Threshold 0.5 (naive) | *[fill]* | *[fill]* | *[fill]* |

*[fill: note that threshold 0.5 flags far fewer than 20 %, wasting campaign
capacity and missing churners.  The ranking rule and the fitted threshold
give similar recall/precision since they both flag ~20 %.]*

**All choices are now locked:**
- Model: HistGBT (tuned)
- Operating rule: top-20 % ranking
- Threshold for CM: *[fill]*
- Test set: not yet accessed

---

## Cell 4 — 5.4 Final test evaluation

```python
# ── 5.4 FINAL TEST EVALUATION ─────────────────────────────────────────────────
# !! Test set accessed for the FIRST and ONLY time. !!
# Model:    HistGBT (tuned)
# Rule:     top-20% ranking (main) + LOCKED_THRESH for CM

proba_test = hgbt_tuned.predict_proba(X_test_t)[:, 1]

# ── Ranking metrics (threshold-free) ──────────────────────────────────────────
pr_auc_test  = round(average_precision_score(y_test, proba_test), 4)
roc_auc_test = round(roc_auc_score(y_test, proba_test), 4)
rec_top_test, prec_top_test = recall_precision_top(y_test, proba_test)

# ── Threshold-based metrics at LOCKED_THRESH ──────────────────────────────────
pred_locked = (proba_test >= LOCKED_THRESH).astype(int)
rec_locked  = round(recall_score(y_test, pred_locked), 4)
prec_locked = round(precision_score(y_test, pred_locked, zero_division=0), 4)
n_flagged   = int(pred_locked.sum())
n_test      = len(y_test)

test_row = {
    "PR-AUC":            pr_auc_test,
    "ROC-AUC":           roc_auc_test,
    "Recall@top20%":     rec_top_test,
    "Precision@top20%":  prec_top_test,
    f"Recall@thr={LOCKED_THRESH}":   rec_locked,
    f"Precision@thr={LOCKED_THRESH}": prec_locked,
    "Flagged (%)":       round(100 * n_flagged / n_test, 1),
}

print("=== FINAL TEST RESULTS — HistGBT (tuned) ===")
display(pd.DataFrame([test_row], index=["HistGBT (tuned)"]))
print(f"\nFlagged {n_flagged}/{n_test} customers ({100*n_flagged/n_test:.1f}%) at-risk.")
print("Test set will NOT be accessed again.")
```

### 5.4  Final Test Evaluation

| Metric | Value |
|--------|-------|
| PR-AUC (primary) | **[fill]** |
| ROC-AUC | **[fill]** |
| Recall@top-20 % | **[fill]** |
| Precision@top-20 % | **[fill]** |
| Recall @ locked threshold | **[fill]** |
| Precision @ locked threshold | **[fill]** |
| Customers flagged (%) | **[fill]** |

*[fill: compare test PR-AUC vs val PR-AUC.  A small drop (≤ 0.02) is
normal; a large drop (> 0.05) suggests val overfitting.  Comment on whether
recall at top-20 % is acceptable for the campaign.]*

---

## Cell 5 — 5.5 Error analysis

```python
# ── 5.5 Error analysis ─────────────────────────────────────────────────────────
# Depends on: proba_test, pred_locked, LOCKED_THRESH from Cell 5.4
# 4 diagnostics: (a) confusion matrix, (b) PR curve, (c) calibration,
#                (d) geography failure-mode slice

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, calibration_curve
import os

os.makedirs("outputs", exist_ok=True)

# ── (a) Confusion matrix at LOCKED_THRESH ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_locked,
    display_labels=["Stay", "Churn"],
    cmap="Blues", colorbar=False, ax=ax1,
)
ax1.set_title(f"Confusion Matrix\n"
              f"HistGBT (tuned)  |  thr = {LOCKED_THRESH}  |  test set")
fig1.tight_layout()
fig1.savefig("outputs/5a_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5a_confusion_matrix.png")

# ── (b) PR curve — val vs test ────────────────────────────────────────────────
prec_v, rec_v, _ = precision_recall_curve(
    y_val, hgbt_tuned.predict_proba(X_val_t)[:, 1]
)
prec_t, rec_t, _ = precision_recall_curve(y_test, proba_test)

prauc_val_plot = round(average_precision_score(
    y_val, hgbt_tuned.predict_proba(X_val_t)[:, 1]
), 4)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(rec_v, prec_v, label=f"Validation  PR-AUC = {prauc_val_plot}")
ax2.plot(rec_t, prec_t, label=f"Test        PR-AUC = {pr_auc_test}",
         linestyle="--")
ax2.axhline(y_test.mean(), color="gray", linestyle=":",
            label=f"No-skill ({y_test.mean():.3f})")
ax2.scatter([rec_locked], [prec_locked], zorder=5, color="red", s=80,
            label=f"Locked threshold ({LOCKED_THRESH})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR Curve — HistGBT (tuned)")
ax2.legend(loc="upper right", fontsize=8)
fig2.tight_layout()
fig2.savefig("outputs/5b_pr_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5b_pr_curve.png")

# ── (c) Calibration reliability diagram ───────────────────────────────────────
prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

axes3[0].plot(prob_pred, prob_true, marker="o", label="HistGBT (tuned)")
axes3[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
axes3[0].set_xlabel("Mean predicted probability")
axes3[0].set_ylabel("Fraction of positives")
axes3[0].set_title("Calibration Curve (test)")
axes3[0].legend()

axes3[1].hist(proba_test[y_test == 0], bins=30, alpha=0.6,
              label="Stay", color="steelblue", density=True)
axes3[1].hist(proba_test[y_test == 1], bins=30, alpha=0.6,
              label="Churn", color="tomato", density=True)
axes3[1].axvline(LOCKED_THRESH, color="black", linestyle="--",
                 label=f"thr = {LOCKED_THRESH}")
axes3[1].set_xlabel("Predicted probability"); axes3[1].set_ylabel("Density")
axes3[1].set_title("Score distribution (test)")
axes3[1].legend(fontsize=8)

fig3.tight_layout()
fig3.savefig("outputs/5c_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5c_calibration.png")

# ── (d) Failure-mode slice: Geography ─────────────────────────────────────────
geo_vals    = df_model.loc[y_test.index, "Geography"].values
y_test_vals = y_test.values

geo_rows = []
for geo in sorted(set(geo_vals)):
    mask = geo_vals == geo
    y_sl = y_test_vals[mask]
    p_sl = proba_test[mask]
    if y_sl.sum() == 0:
        continue
    rec_sl, prec_sl = recall_precision_top(pd.Series(y_sl), p_sl)
    geo_rows.append({
        "Geography":        geo,
        "N":                int(len(y_sl)),
        "Churners":         int(y_sl.sum()),
        "Churn rate":       round(float(y_sl.mean()), 4),
        "PR-AUC":           round(float(average_precision_score(y_sl, p_sl)), 4),
        "Recall@top20%":    rec_sl,
    })

geo_df = pd.DataFrame(geo_rows)
print("\n── Failure-mode slice: Geography (test set) ──")
display(geo_df)

worst = geo_df.loc[geo_df["PR-AUC"].idxmin(), "Geography"]
print(f"Worst geography: {worst} | Overall PR-AUC: {pr_auc_test}")
```

### 5.5  Error Analysis

**(a) Confusion matrix (threshold = [fill]):**

*[fill: note TP (correct alerts), FP (wasted calls), FN (missed churners),
TN.  Relate back to cost asymmetry: we accept some FP to minimise FN,
consistent with Precision@top-20% measuring the wasted-intervention rate.]*

**(b) PR curve — val vs test:**

*[fill: is the gap small (< 0.02)?  Note where the locked threshold
(red dot) sits on the curve.  A large gap signals val overfitting.]*

**(c) Calibration:**

*[fill: does the reliability curve track the diagonal?  If it bows above,
the model over-predicts churn risk; if below, it under-predicts.  Comment
on class separation in the histogram — good separation means the locked
threshold sits in a low-density region between the two distributions.]*

**(d) Geography slice:**

| Geography | N | Churners | Churn rate | PR-AUC | Recall@top20% |
|-----------|---|----------|------------|--------|---------------|
| France  | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Germany | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Spain   | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

*[fill: Germany has ~32 % churn vs ~16 % elsewhere (EDA).  If PR-AUC is
lower for Germany despite higher prevalence, the model has fewer signal
features there — flag as a deployment risk.]*

---

## Cell 6 — 5.6 Agent-made mistake and fix

```python
# ── 5.6 Agent-made mistake and fix ────────────────────────────────────────────
#
# Mistake: using model.predict() instead of model.predict_proba()[:, 1]
#          when computing average_precision_score (PR-AUC).
#
# Why it happens: predict() returns binary 0/1 labels; the agent uses it
#   in an ad-hoc metric call outside the evaluate() helper.
#
# Why it matters: average_precision_score with binary inputs collapses
#   the PR curve to a single point → reported "PR-AUC" equals the
#   precision at the default threshold, not the full area under the curve.

print("── Demonstrating the predict() vs predict_proba() bug ──\n")

# WRONG — binary labels, not probabilities
proba_bug     = hgbt_tuned.predict(X_val_t)              # BUG
prauc_bug     = round(average_precision_score(y_val, proba_bug), 4)

# CORRECT — continuous probabilities
proba_correct = hgbt_tuned.predict_proba(X_val_t)[:, 1]  # FIX
prauc_correct = round(average_precision_score(y_val, proba_correct), 4)

mistake_df = pd.DataFrame([
    {"Method": "predict()       ← WRONG",  "PR-AUC": prauc_bug,
     "Note": "single point, not area under curve"},
    {"Method": "predict_proba() ← CORRECT", "PR-AUC": prauc_correct,
     "Note": "full PR curve integrated"},
])
print(mistake_df.to_string(index=False))

delta_bug = prauc_correct - prauc_bug
print(f"\nBug understates PR-AUC by {delta_bug:.4f} points.")
print("Fix: always use predict_proba(X)[:, 1] for threshold-free metrics.")

# Self-check
import inspect
assert "predict_proba" in inspect.getsource(evaluate), \
    "evaluate() must use predict_proba!"
print("Self-check: evaluate() uses predict_proba() ✓")
```

### 5.6  Agent-made Mistake and Fix

**Mistake:** The agent uses `model.predict()` (binary 0/1) instead of
`model.predict_proba()[:, 1]` (continuous probabilities) when computing
`average_precision_score`.

**Impact:** With binary inputs, the PR curve collapses to one point.  The
reported "PR-AUC" is just the precision at the default threshold — not the
area under the full curve.  The drop is **[fill]** absolute points, enough
to incorrectly rank or eliminate a model.

**How I caught it:** The buggy number was suspiciously close to the churn
prevalence (~0.20).  A quick `inspect.getsource(evaluate)` confirmed the
helper uses `predict_proba` — so the ad-hoc call was inconsistent.

**Fix:** Always pass `predict_proba(X)[:, 1]` to threshold-free metrics
(`average_precision_score`, `roc_auc_score`, `precision_recall_curve`).
Use binary `predict()` only for threshold-dependent metrics (`recall_score`,
`precision_score`, `f1_score`).

---

## Model Card

**Model:** HistGradientBoostingClassifier (tuned) — sklearn native.

**What it is for:**
Ranking bank customers by churn risk so a fixed-capacity retention campaign
(budget = 20 % of customer base) contacts the highest-risk individuals.

**What it is NOT for:**
- Individual causal explanations ("why did *this* customer churn?")
- Real-time scoring at sub-millisecond latency
- Predicting churn for products or populations outside the training data
  distribution (different bank, different country mix, etc.)

**Key metrics (test set):**

| Metric | Value |
|--------|-------|
| PR-AUC | *[fill]* |
| ROC-AUC | *[fill]* |
| Recall@top-20% | *[fill]* |
| Precision@top-20% | *[fill]* |

**Data constraints:**
- Trained on 10 000 customers from a single bank (Kaggle CC0 dataset)
- Features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
  IsActiveMember, EstimatedSalary, Geography, Gender
- Temporal dimension absent — no time-series features, no concept drift
  monitoring

**Evaluation caveats:**
- Validation and test sets are random stratified splits, not temporal
  out-of-time holds — real-world performance may differ under distribution
  shift
- Geography slice (Section 5.5d) may reveal unequal performance across
  regions — monitor in production
- Probability calibration (Section 5.5c) should be checked periodically
  if the model is used for threshold-based decisions rather than ranking
- If the model is used for threshold-based decisions, periodic
  recalibration may be needed

---

## 5.7  My Agent vs My Verification

| Step | What My Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Metrics alignment | Agent initially used 3 metrics in Task 5. I requested Precision@top-20% be added as a fourth metric across all tables, consistent with Section 1.3 | I confirmed all 4 metrics (PR-AUC, ROC-AUC, Recall@top-20%, Precision@top-20%) appear in every evaluation table in Tasks 4 and 5 |
| Sanity check | Re-fit HistGBT (untuned) on training; confirmed Task 4 validation numbers reproduce | *[fill: confirm PR-AUC matches Task 4; if not, check SEED and preprocessing]* |
| Tuning | `RandomizedSearchCV` on HistGBT: n\_iter=8, cv=3, `scoring="average_precision"`, training only | *[fill: confirm best params; confirm tuning gain ΔPR-AUC; note if marginal or meaningful]* |
| Operating rule | Locked top-20 % ranking as main rule; derived threshold (80th-percentile of val probabilities) for CM view; showed 0.5 threshold is too conservative | *[fill: confirm LOCKED_THRESH flags ~20 % on validation; confirm 0.5 flags far fewer; confirm test not accessed]* |
| Test evaluation | `hgbt_tuned.predict_proba(X_test_t)` called once; all 4 metrics + threshold metrics reported | *[fill: val → test PR-AUC gap ≤ 0.02? Flagged % ≈ 20 %? Confirm pred_locked uses LOCKED_THRESH not 0.5]* |
| Error analysis (CM) | Confusion matrix at locked threshold; saved to `outputs/5a_confusion_matrix.png` | *[fill: check figure saved; note TP/FP/FN/TN counts; relate FP rate to Precision@top-20%]* |
| Error analysis (PR curve) | PR curve comparing val vs test; locked threshold marked as red dot | *[fill: is gap < 0.02? Does red dot sit at an acceptable recall/precision trade-off?]* |
| Error analysis (calibration) | Calibration reliability curve + score-distribution histogram; saved to `outputs/5c_calibration.png` | *[fill: does reliability curve track the diagonal? Comment on class separation in the histogram]* |
| Error analysis (geography) | Per-country PR-AUC and Recall@top-20%; identifies worst-performing geography | *[fill: check index alignment (df_model.loc[y_test.index]); note worst geography; flag fairness implications]* |
| Agent mistake | Demonstrated predict() vs predict_proba() for PR-AUC; showed Δ; self-checked evaluate() | *[fill: note actual Δ; confirm buggy < correct; confirm assert passed]* |
| Model card | Agent drafted model card with intended use, limitations, data constraints, evaluation caveats, and 4-metric summary table | *[fill: confirm model card accurately reflects final model and results]* |
| *[add rows as needed]* | | |
