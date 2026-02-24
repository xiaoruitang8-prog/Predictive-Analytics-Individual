# Task 5 — Fine-Tune and Evaluate

---

## Coding Plan

1. **5.1 Controlled comparisons on validation.**  Three required comparisons,
   all on the validation set only:
   - **A** Tuned vs untuned (HistGBT) — carries forward Ablation C.
   - **B** Class weighting vs none (HistGBT) — confirms the balanced penalty
     matters on ~20% imbalance.
   - **C** Decision rule — threshold 0.5 vs top-20% ranking (tuned HistGBT).
   Plus **Deliverable A**: compact 3-row summary table (tuned, untuned,
   runner-up LogReg balanced).

2. **5.2 Threshold selection.**  Fit the PR curve on the validation set;
   choose the threshold maximising F2 (recall-weighted, β=2).  Store in
   `LOCKED_THRESH` — never changed after this cell.

3. **5.3 Final locked choice.**  Markdown cell; 2–3 sentences with
   validation evidence (**Deliverable B**).

4. **5.4 Final test evaluation.**  The test set is accessed for the first
   and only time.  Report ranking metrics (PR-AUC, ROC-AUC, Recall@top20%)
   and threshold-based metrics at `LOCKED_THRESH` (**Deliverable C**).

5. **5.5 Error analysis.**  Confusion matrix at `LOCKED_THRESH`, PR curve
   (val vs test), calibration reliability diagram, failure-mode slice by
   Geography.  Save three figures to `outputs/` (**Deliverables D & E**).

6. **5.6 Agent-made mistake and fix.**  Demonstrate the predict() vs
   predict_proba() bug for PR-AUC; show the concrete PR-AUC drop; confirm
   the fix (**agent-mistake requirement**).

Constraints: test set accessed only in Cell 5.4 (once). `random_state=SEED`
throughout.  No grid searches in Task 5 — the tuned model from Task 4
Ablation C is reused.

---

## Cell 1 — 5.1 Controlled comparisons on validation

```python
# ── Task 5: Controlled comparisons on validation ──────────────────────────────
# Prerequisites: Task 3 cells 1–6 and Task 4 cells 1–4 must have run first.
# Inherits: X_train_t, X_val_t, X_test_t, y_train, y_val, y_test
#            SEED=42, TOP_PCT=0.20
#            hgbt (untuned, balanced), hgbt_tuned (from Ablation C)
#            evaluate(), recall_precision_top()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import HistGradientBoostingClassifier
from sklearn.metrics      import (average_precision_score, roc_auc_score,
                                   recall_score, precision_score)
import numpy  as np
import pandas as pd

# ── Re-fit runner-up model ─────────────────────────────────────────────────────
# LogReg (balanced) is the second shortlisted model from Task 4.
# Refit here so Task 5 is self-contained; costs < 1 s on 7 k rows.
lr_final = LogisticRegression(C=1.0, max_iter=1000,
                               class_weight="balanced", random_state=SEED)
lr_final.fit(X_train_t, y_train)

# ── Deliverable A: compact 3-row validation summary ───────────────────────────
row_tuned   = evaluate("HistGBT (tuned)",   hgbt_tuned, X_val_t, y_val)
row_untuned = evaluate("HistGBT (untuned)", hgbt,       X_val_t, y_val)
row_logreg  = evaluate("LogReg (balanced)", lr_final,   X_val_t, y_val)

summary_val = pd.DataFrame([row_tuned, row_untuned, row_logreg])
print("=== Deliverable A: validation summary (sorted by PR-AUC) ===")
display(summary_val)

# ── Comparison A: tuning gain ──────────────────────────────────────────────────
delta_a = round(row_tuned["PR-AUC"] - row_untuned["PR-AUC"], 4)
print(f"\nComparison A — tuning gain:  ΔPR-AUC = {delta_a:+.4f}")
print(f"  (positive = tuned beats untuned on validation)")

# ── Comparison B: class weighting effect (HistGBT) ────────────────────────────
hgbt_noweight = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
hgbt_noweight.fit(X_train_t, y_train)

comp_b = pd.DataFrame([
    evaluate("HistGBT (no weighting)", hgbt_noweight, X_val_t, y_val),
    evaluate("HistGBT (balanced)",     hgbt,          X_val_t, y_val),
])
print("\n── Comparison B: class weighting effect (HistGBT, val) ──")
display(comp_b)

# ── Comparison C: decision rule on tuned HistGBT ──────────────────────────────
proba_tuned_val = hgbt_tuned.predict_proba(X_val_t)[:, 1]

# (a) Default threshold 0.5
pred_05  = (proba_tuned_val >= 0.5).astype(int)
rec_05   = round(recall_score(y_val, pred_05), 4)
prec_05  = round(precision_score(y_val, pred_05, zero_division=0), 4)

# (b) Top-20% ranking — flags exactly n_top highest-risk customers
n_top       = max(1, int(len(y_val) * TOP_PCT))
top_arr     = np.zeros(len(y_val), dtype=int)
top_arr[np.argsort(proba_tuned_val)[::-1][:n_top]] = 1
rec_top20   = round(recall_score(y_val, top_arr), 4)
prec_top20  = round(precision_score(y_val, top_arr, zero_division=0), 4)

comp_c = pd.DataFrame([
    {"Decision rule": "Threshold 0.5",   "Recall": rec_05,   "Precision": prec_05},
    {"Decision rule": "Top-20% ranking", "Recall": rec_top20, "Precision": prec_top20},
])
print("\n── Comparison C: decision rule (tuned HistGBT, val) ──")
display(comp_c)
```

### 5.1  Controlled Comparisons on Validation

All comparisons are evaluated on the **validation set** using the four
metrics agreed in Section 1.3.  The test set is not accessed in this cell.

**Deliverable A — Validation summary:**

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (tuned) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| HistGBT (untuned) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| LogReg (balanced) | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

**Comparison A — Tuned vs untuned (HistGBT).**
ΔPR-AUC = *[fill]*.
*[fill: was tuning beneficial? State how much PR-AUC changed; if Δ < 0.005
the untuned model is equally valid — document the decision.]*

**Comparison B — Class weighting (HistGBT).**
*[fill: state PR-AUC for balanced vs no-weighting; confirm whether
class_weight="balanced" improves minority-class discrimination on ~20%
imbalance, as expected.]*

**Comparison C — Decision rule (tuned HistGBT).**

| Decision rule | Recall | Precision |
|---------------|--------|-----------|
| Threshold 0.5 | *[fill]* | *[fill]* |
| Top-20% ranking | *[fill]* | *[fill]* |

*[fill: for a fixed-capacity retention campaign (budget = 20% of customers),
the top-20% ranking rule guarantees all available slots are filled.  Note
whether it raises recall vs threshold 0.5, and what the precision trade-off
is.  This comparison drives the operating decision locked in Cell 5.2.]*

---

## Cell 2 — 5.2 Threshold selection on validation

```python
# ── Task 5: Threshold selection on validation ─────────────────────────────────
# Choose the operating threshold from the validation PR curve.
# Method: maximise F2 score  (β=2, weights recall twice as much as precision).
# Rationale: in a retention campaign, missing a churner (false negative)
#   costs more than a wasted call (false positive) → recall-heavy metric.
# LOCKED_THRESH is set here and never changed before test evaluation.

from sklearn.metrics import precision_recall_curve, fbeta_score

prec_curve, rec_curve, thresholds = precision_recall_curve(
    y_val, hgbt_tuned.predict_proba(X_val_t)[:, 1]
)

# F2 at each threshold (skip the sentinel point appended by sklearn at end)
beta = 2
denom     = beta**2 * prec_curve[:-1] + rec_curve[:-1] + 1e-9
f2_scores = (1 + beta**2) * prec_curve[:-1] * rec_curve[:-1] / denom

best_idx      = int(np.argmax(f2_scores))
LOCKED_THRESH = round(float(thresholds[best_idx]), 4)
locked_rec    = round(float(rec_curve[best_idx]), 4)
locked_prec   = round(float(prec_curve[best_idx]), 4)
locked_f2     = round(float(f2_scores[best_idx]), 4)

print("── Threshold selection (validation only) ──")
print(f"Locked threshold (max F2, β=2):  {LOCKED_THRESH}")
print(f"  Recall    at locked threshold: {locked_rec}")
print(f"  Precision at locked threshold: {locked_prec}")
print(f"  F2        at locked threshold: {locked_f2}")
print(f"\nLOCKED_THRESH = {LOCKED_THRESH}  ← applied once to test set in Cell 5.4")
print("Test set has NOT been accessed in this cell.")
```

### 5.2  Threshold Selection

The operating threshold is chosen **from the validation PR curve** by
maximising the **F2 score** (β = 2).  F2 up-weights recall over precision
(ratio 4:1), matching the retention campaign objective where missing a
churner is costlier than a wasted intervention.

| | Value |
|-|-------|
| Locked threshold | **[fill]** |
| Recall at threshold | **[fill]** |
| Precision at threshold | **[fill]** |
| F2 at threshold | **[fill]** |

This threshold is fixed before any test data is seen.  Changing it after
viewing test results would constitute **post-hoc threshold tuning** — a form
of data leakage.

---

## 5.3  Final Locked Choice

*Markdown cell — fill after running Cells 5.1 and 5.2.*

---

**Shortlisted models for Task 5: HistGBT (tuned) and LogReg (balanced)**

On the validation set, **HistGBT (tuned)** achieves the highest PR-AUC of
**[fill]** with Recall@top-20% = **[fill]**, capturing **[fill fraction]**
of all churners under the campaign's fixed-capacity constraint.
Tuning via `RandomizedSearchCV` (PR-AUC scoring, training CV only) improved
PR-AUC by **[Δ fill]** over the untuned baseline.
**LogReg (balanced)** is retained as the second candidate with PR-AUC =
**[fill]**, providing coefficient-level interpretability to explain
individual churn drivers to business stakeholders — addressing the
interpretability constraint in Section 1.4.
Both models were chosen on **validation metrics only**; the test set
remains untouched and is evaluated once in Section 5.4.

---

## Cell 3 — 5.4 Final test evaluation

```python
# ── Task 5: FINAL TEST EVALUATION ─────────────────────────────────────────────
# !! The test set is accessed for the FIRST and ONLY time in this cell. !!
# Chosen model:    HistGBT (tuned)  — highest validation PR-AUC (Cell 5.1)
# Locked threshold: LOCKED_THRESH   — chosen on validation F2 (Cell 5.2)
# No further decisions will be made after seeing these numbers.

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score

proba_test = hgbt_tuned.predict_proba(X_test_t)[:, 1]

# ── Ranking metrics (threshold-free) ──────────────────────────────────────────
pr_auc_test  = round(average_precision_score(y_test, proba_test), 4)
roc_auc_test = round(roc_auc_score(y_test, proba_test), 4)
rec_top_test, prec_top_test = recall_precision_top(y_test, proba_test)

# ── Threshold-based metrics at LOCKED_THRESH ──────────────────────────────────
pred_locked  = (proba_test >= LOCKED_THRESH).astype(int)
rec_locked   = round(recall_score(y_test, pred_locked), 4)
prec_locked  = round(precision_score(y_test, pred_locked, zero_division=0), 4)
f2_locked    = round(
    (1 + 4) * prec_locked * rec_locked / (4 * prec_locked + rec_locked + 1e-9),
    4,
)
n_flagged    = int(pred_locked.sum())
n_test       = len(y_test)

test_row = {
    "PR-AUC":                        pr_auc_test,
    "ROC-AUC":                       roc_auc_test,
    "Recall@top20%":                 rec_top_test,
    "Precision@top20%":              prec_top_test,
    f"Recall@thr={LOCKED_THRESH}":   rec_locked,
    f"Precision@thr={LOCKED_THRESH}": prec_locked,
    f"F2@thr={LOCKED_THRESH}":       f2_locked,
    "Flagged (%)":                   round(100 * n_flagged / n_test, 1),
}

print("=== Deliverable C: FINAL TEST RESULTS — HistGBT (tuned) ===")
display(pd.DataFrame([test_row], index=["HistGBT (tuned)"]))
print(f"\nFlagged {n_flagged}/{n_test} customers ({100*n_flagged/n_test:.1f}%) as at-risk.")
print("Test set will not be accessed again.")
```

### 5.4  Final Test Evaluation

**HistGBT (tuned) — one-time test set results:**

| Metric | Value |
|--------|-------|
| PR-AUC (primary) | **[fill]** |
| ROC-AUC | **[fill]** |
| Recall@top-20% | **[fill]** |
| Precision@top-20% | **[fill]** |
| Recall @ locked threshold | **[fill]** |
| Precision @ locked threshold | **[fill]** |
| F2 @ locked threshold | **[fill]** |
| Customers flagged (%) | **[fill]** |

*[fill: compare PR-AUC test vs val — a small drop (≤ 0.02) is expected and
normal; a large drop (> 0.05) would suggest overfitting to the validation
set during threshold selection.  Comment on whether the locked threshold
produced reasonable recall and precision on unseen data.]*

---

## Cell 4 — 5.5 Error Analysis

```python
# ── Task 5: Error analysis ─────────────────────────────────────────────────────
# Depends on: proba_test, pred_locked, LOCKED_THRESH (from Cell 5.4)
# Produces:
#   (a) Confusion matrix at LOCKED_THRESH
#   (b) PR curve — validation vs test overlay
#   (c) Calibration reliability diagram
#   (d) Failure-mode slice by Geography

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                              calibration_curve)
import os

os.makedirs("outputs", exist_ok=True)

# ── (a) Confusion matrix ───────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_locked,
    display_labels=["Stay", "Churn"],
    cmap="Blues", colorbar=False, ax=ax1,
)
ax1.set_title(
    f"Confusion Matrix\n"
    f"HistGBT (tuned)  |  threshold = {LOCKED_THRESH}  |  test set"
)
fig1.tight_layout()
fig1.savefig("outputs/5a_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5a_confusion_matrix.png")

# ── (b) PR curve — val vs test overlay ────────────────────────────────────────
prec_v, rec_v, _  = precision_recall_curve(
    y_val,  hgbt_tuned.predict_proba(X_val_t)[:, 1]
)
prec_t, rec_t, _  = precision_recall_curve(y_test, proba_test)

# Recompute val PR-AUC for the label (avoids calling evaluate() again)
prauc_val_plot = round(average_precision_score(
    y_val, hgbt_tuned.predict_proba(X_val_t)[:, 1]
), 4)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(rec_v, prec_v,  label=f"Validation  PR-AUC = {prauc_val_plot:.4f}")
ax2.plot(rec_t, prec_t,  label=f"Test        PR-AUC = {pr_auc_test:.4f}",
         linestyle="--")
ax2.axhline(y_test.mean(), color="gray", linestyle=":",
            label=f"No-skill baseline ({y_test.mean():.3f})")
ax2.scatter([rec_locked], [prec_locked], zorder=5, color="red", s=80,
            label=f"Locked threshold ({LOCKED_THRESH})")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve — HistGBT (tuned)")
ax2.legend(loc="upper right", fontsize=8)
fig2.tight_layout()
fig2.savefig("outputs/5b_pr_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5b_pr_curve.png")

# ── (c) Calibration reliability diagram ───────────────────────────────────────
prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10)

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

axes3[0].plot(prob_pred, prob_true, marker="o", label="HistGBT (tuned)")
axes3[0].plot([0, 1], [0, 1], linestyle="--", color="gray",
              label="Perfectly calibrated")
axes3[0].set_xlabel("Mean predicted probability")
axes3[0].set_ylabel("Fraction of positives")
axes3[0].set_title("Calibration Curve (test set)")
axes3[0].legend()

# Predicted probability histogram
axes3[1].hist(proba_test[y_test == 0], bins=30, alpha=0.6,
              label="Non-churners", color="steelblue", density=True)
axes3[1].hist(proba_test[y_test == 1], bins=30, alpha=0.6,
              label="Churners", color="tomato", density=True)
axes3[1].axvline(LOCKED_THRESH, color="black", linestyle="--",
                 label=f"Threshold = {LOCKED_THRESH}")
axes3[1].set_xlabel("Predicted probability")
axes3[1].set_ylabel("Density")
axes3[1].set_title("Score distribution by class (test set)")
axes3[1].legend(fontsize=8)

fig3.tight_layout()
fig3.savefig("outputs/5c_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5c_calibration.png")

# ── (d) Failure-mode slice: Geography ─────────────────────────────────────────
# df_model retains original indices through the stratified split.
# df_model.loc[y_test.index] is position-aligned with proba_test / y_test.values
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
        "N (test)":         int(len(y_sl)),
        "Churners":         int(y_sl.sum()),
        "Churn rate":       round(float(y_sl.mean()), 4),
        "PR-AUC":           round(float(average_precision_score(y_sl, p_sl)), 4),
        "Recall@top20%":    rec_sl,
        "Precision@top20%": prec_sl,
    })

geo_df = pd.DataFrame(geo_rows)
print("\n── Deliverable D: failure-mode slice — Geography (test set) ──")
display(geo_df)

# Highlight any geography with PR-AUC > 0.05 below overall
geo_df["Gap vs overall"] = (geo_df["PR-AUC"] - pr_auc_test).round(4)
worst = geo_df.loc[geo_df["PR-AUC"].idxmin(), "Geography"]
print(f"\nWorst-performing geography: {worst}")
print(f"Overall test PR-AUC: {pr_auc_test:.4f}")
```

### 5.5  Error Analysis

**(a) Confusion matrix at locked threshold = [fill]:**

*[fill: note TP (correct churner alerts), FP (wasted interventions),
FN (missed churners), TN.  High FN count means the threshold is too
conservative; high FP means it is too aggressive.  Relate back to the F2
choice which explicitly tolerates more FP to reduce FN.]*

**(b) PR curve:**

*[fill: val vs test PR-AUC gap — is it small (< 0.02)?  A large drop
signals validation overfitting.  Note where the locked threshold (red dot)
sits on the curve — is it near the "elbow" of the curve?]*

**(c) Calibration:**

*[fill: does the calibration curve track the diagonal?  If the model
over-predicts (curve bows above diagonal) or under-predicts (bows below),
the locked threshold may need post-hoc recalibration in production.
Comment on score distribution overlap between classes.]*

**(d) Failure-mode slice — Geography:**

| Geography | N | Churners | Churn rate | PR-AUC | Recall@top20% | Precision@top20% |
|-----------|---|----------|------------|--------|---------------|------------------|
| France | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Germany | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Spain | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

*[fill: Germany is known from EDA (Section 2) to have a higher churn rate
(~32%) vs France and Spain (~16%).  If PR-AUC is notably lower for Germany
despite higher prevalence, it suggests the model has relatively fewer signal
features for that group — a finding worth flagging as a deployment risk and
priority for feature engineering in future iterations.]*

---

## Cell 5 — 5.6 Agent-made mistake and fix

```python
# ── Task 5 — 5.6 Agent-made mistake and fix ───────────────────────────────────
#
# Mistake: using model.predict() instead of model.predict_proba() to compute
#          PR-AUC (average_precision_score).
#
# Why it happens: the agent calls evaluate() correctly but when writing
#   ad-hoc PR-AUC checks during debugging, accidentally uses predict()
#   which returns binary 0/1 labels, not continuous probabilities.
#
# Why it matters: average_precision_score collapses the PR curve to a single
#   operating point (one precision-recall pair) when given binary inputs.
#   The resulting "PR-AUC" is just the precision at that single point — not
#   the area under the curve. This artificially depresses the metric and
#   could incorrectly eliminate a model from the shortlist.
#
# How to catch it: the buggy PR-AUC will be close to the precision at
#   the default threshold (~0.5) and will not rise when you add more models
#   with better calibration. It also ignores ROC-AUC and will not correlate
#   with it.

print("── Demonstrating the predict() vs predict_proba() bug ──\n")

# WRONG — predict() returns binary labels, NOT probabilities
proba_bug    = hgbt_tuned.predict(X_val_t)        # <-- BUG: returns 0/1
prauc_bug    = round(average_precision_score(y_val, proba_bug), 4)

# CORRECT — predict_proba() returns class probabilities; take column 1 (positive)
proba_correct = hgbt_tuned.predict_proba(X_val_t)[:, 1]  # <-- FIX
prauc_correct = round(average_precision_score(y_val, proba_correct), 4)

mistake_df = pd.DataFrame([
    {"Method": "predict()       ← WRONG",  "Input type": "binary 0/1",
     "PR-AUC": prauc_bug,
     "Note": "single operating point, not area under curve"},
    {"Method": "predict_proba()[:, 1]  ← CORRECT", "Input type": "float [0,1]",
     "PR-AUC": prauc_correct,
     "Note": "full precision-recall curve integrated"},
])
print(mistake_df.to_string(index=False))

print(f"\nPR-AUC drop from bug:  {prauc_bug - prauc_correct:+.4f}")
print(
    f"\nConclusion: using predict() understates PR-AUC by "
    f"{prauc_correct - prauc_bug:.4f} absolute points.\n"
    f"Fix: always pass model.predict_proba(X)[:, 1] to average_precision_score.\n"
    f"The evaluate() helper in Task 4 Cell 1 already implements this correctly."
)

# ── Self-check: confirm evaluate() uses predict_proba ─────────────────────────
import inspect
src = inspect.getsource(evaluate)
assert "predict_proba" in src, "evaluate() does not use predict_proba — fix it!"
print("Self-check passed: evaluate() uses predict_proba().")
```

### 5.6  Agent-made Mistake and Fix

**Mistake:** The agent accidentally uses `model.predict()` instead of
`model.predict_proba()[:, 1]` when computing `average_precision_score`
in an ad-hoc debugging cell outside of the `evaluate()` helper.

**Impact:** `predict()` returns binary 0/1 labels; `average_precision_score`
treats them as "scores" with only two distinct values, reducing the
precision-recall curve to a single point.  The reported PR-AUC equals the
precision at the default threshold — not the area under the full curve.
The drop is typically **[fill: Δ value from cell output]** absolute PR-AUC
points, which is large enough to incorrectly rank models.

**How to catch it:**
- The buggy number is suspiciously close to the test-set churn prevalence
  (~0.20) or to `precision_score(y, model.predict(X))` — a red flag.
- It does not increase when switching to a better-calibrated model.
- `inspect.getsource(evaluate)` confirms the helper uses `predict_proba` —
  so any manually written metric call must be held to the same standard.

**Fix:** Always pass `model.predict_proba(X)[:, 1]` to all
threshold-free metric functions (`average_precision_score`,
`roc_auc_score`, `precision_recall_curve`).  Binary `predict()` output
is only correct for threshold-dependent metrics (`accuracy_score`,
`recall_score`, `precision_score`, `f1_score`).

---

## 5.7  Agent Plan vs My Verification

| Step | What the Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Comparison A (tuning) | Reused `hgbt` and `hgbt_tuned` from Task 4 Ablation C; compared on validation; printed ΔPR-AUC | *[fill: confirm tuning improved PR-AUC; if Δ < 0.005 note it; confirm hgbt_tuned is the model used in Cell 5.4 and not accidentally hgbt]* |
| Comparison B (weighting) | Fitted `hgbt_noweight` (no class_weight) on training only; compared to `hgbt` (balanced) on validation | *[fill: confirm balanced weighting improves PR-AUC as expected; if it does not, note whether the dataset is skewed differently than expected]* |
| Comparison C (decision rule) | Compared threshold 0.5 vs top-20% ranking for `hgbt_tuned`; reported recall and precision for both | *[fill: which rule gives higher recall? Does top-20% guarantee a useful precision? Confirm which rule was used as the locked operating decision]* |
| Threshold selection | Maximised F2 (β=2) on validation PR curve; stored result in `LOCKED_THRESH`; test set not accessed | *[fill: verify LOCKED_THRESH is set before Cell 5.4 runs; check it is a sensible value (typically 0.2–0.5 for a 20% minority class); confirm it was NOT tuned after seeing test results]* |
| Final test evaluation | Called `hgbt_tuned.predict_proba(X_test_t)` once; reported all agreed metrics; test set not accessed again | *[fill: confirm val → test PR-AUC gap is ≤ 0.02; check flagged % is reasonable; confirm pred_locked uses LOCKED_THRESH not 0.5]* |
| Error analysis — slice | Used `df_model.loc[y_test.index, "Geography"]` to align original features with test predictions | *[fill: verify index alignment — df_model.loc[y_test.index] must return rows in the same order as y_test.values; if indices were reset at any point this will silently misalign — check by printing df_model.loc[y_test.index[:3], "Geography"] and comparing to X_test row 0]* |
| Agent-made mistake | Demonstrated predict() vs predict_proba() bug; showed PR-AUC drop; confirmed evaluate() uses predict_proba via inspect | *[fill: run the self-check assert; note the actual Δ PR-AUC; confirm this matches the expected direction (buggy < correct)]* |
| *[add rows as needed]* | | |
