# Task 5 — Fine-Tune and Evaluate

---

## Coding Plan

1. **5.1 Imports and baseline aliases.**  Import tuning / plotting
   dependencies.  Alias the Task 4 fitted models (`hgbt`, `rf`) as
   `hgbt_base`, `rf_base` so the tuning cell has clear untuned vs tuned
   naming.

2. **5.2 Tune both shortlisted candidates.**  `RandomizedSearchCV` on
   HistGBT and RandomForest (n\_iter=8 each, 3-fold CV, training only,
   `scoring="average_precision"`).  Compare tuned vs untuned for both
   models on validation.  Total budget: 2 × 8 × 3 = 48 fits.

3. **5.3 Lock all choices on validation.**  Lock **three things** using
   validation data only, before any test access:
   (a) **Final model** — by validation PR-AUC (pre-committed decision
       criterion from Section 4.5).
   (b) **Operating rule** — top-20 % by predicted risk.
   (c) **Threshold** — 80th percentile of the locked model's validation
       probabilities, for confusion-matrix view.

4. **5.4 Final test evaluation (report only).**  Test set accessed for
   the first time.  The final model is already locked — test is used
   solely for reporting, not selection.  Runner-up test metrics shown
   for context, clearly labelled as not used for any decision.

5. **5.5 Error analysis — 4 diagnostics.**
   - **(a)** Confusion matrix at the locked threshold (final model)
   - **(b)** PR curve (val vs test overlay, both models)
   - **(c)** Calibration diagram (both models side-by-side)
   - **(d)** Geography failure-mode slice (both models)
   Save figures to `outputs/`.  Diagnostics (b)–(d) compare both models
   as post-decision checks — they cannot override the locked selection.

Constraints: test is not used for tuning or model selection; it is
accessed only after all choices are locked (Cell 5.3), for final
evaluation (Cell 5.4) and diagnostics (Cell 5.5).
`random_state=SEED` throughout.  Tuning budget: 2 × 8 × 3 = 48 fits.

### Leakage-Safe Decision Logic

To avoid using the test set for model selection (which would make the
reported test metrics optimistically biased), the pipeline enforces a
strict **train → validate → test** discipline:

1. **Tune on training only** (Cell 5.2): `RandomizedSearchCV` with 3-fold
   CV on `X_train_t`.  The validation set is not seen by the search.
2. **Decide on validation only** (Cell 5.3): The final model, operating
   rule, and threshold are all locked using validation metrics — before
   any test access.  The pre-committed decision criterion (highest
   validation PR-AUC) determines the winner.
3. **Report on test** (Cells 5.4–5.5): The test set is accessed only
   after all choices are locked.  It is used for final evaluation and
   diagnostics — never for tuning or selection.  Runner-up test metrics
   are shown for context but cannot override the locked decision.

If the runner-up happens to outperform the locked model on test, this is
acknowledged transparently in the report but the decision is **not**
reversed — doing so would reintroduce test-based selection.

---

## Cell 1 — 5.1 Imports and baseline aliases

```python
# ── 5.1 Imports + baseline aliases ───────────────────────────────────────────
# Inherits from Task 4: hgbt, rf (fitted), X_train_t, X_val_t, X_test_t,
#   y_train, y_val, y_test, SEED, TOP_PCT, evaluate(), recall_precision_top()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve

hgbt_base, rf_base = hgbt, rf   # alias for tuned-vs-untuned clarity
```

### 5.1  Imports and Baseline Aliases

No re-fitting — `hgbt` and `rf` are already fitted in Task 4 (same
notebook session, same SEED).  Aliased as `hgbt_base` / `rf_base` so
that Cell 2 can compare `*_base` vs `*_tuned` without ambiguity.

---

## Cell 2 — 5.2 Tune both shortlisted candidates

```python
# ── 5.2 Tune both shortlisted models ─────────────────────────────────────────
# Task 4 used default hyperparameters.  Here we do a small search for each.
# scoring="average_precision" = PR-AUC → correct metric for imbalanced data.
# Budget: 2 models × n_iter=8 × cv=3 = 48 total fits.

# --- HistGBT ---
hgbt_param_dist = {
    "max_iter":         [100, 200, 300],
    "max_depth":        [3, 5, None],
    "learning_rate":    [0.05, 0.1, 0.2],
    "min_samples_leaf": [20, 40],
    "class_weight":     [None, "balanced"],
}

hgbt_search = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=SEED),
    param_distributions=hgbt_param_dist,
    n_iter=8, cv=3, scoring="average_precision",
    random_state=SEED, n_jobs=-1,
)
hgbt_search.fit(X_train_t, y_train)
hgbt_tuned = hgbt_search.best_estimator_
print(f"HistGBT best params: {hgbt_search.best_params_}")
print(f"HistGBT best CV PR-AUC: {hgbt_search.best_score_:.4f}")

# --- RandomForest ---
rf_param_dist = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [5, 10, None],
    "min_samples_leaf": [5, 20, 40],
    "class_weight":     [None, "balanced"],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=8, cv=3, scoring="average_precision",
    random_state=SEED, n_jobs=-1,
)
rf_search.fit(X_train_t, y_train)
rf_tuned = rf_search.best_estimator_
print(f"\nRF best params: {rf_search.best_params_}")
print(f"RF best CV PR-AUC: {rf_search.best_score_:.4f}")

# --- Compare all four variants on validation ---
comparison = pd.DataFrame([
    evaluate("HistGBT (tuned)",   hgbt_tuned, X_val_t, y_val),
    evaluate("HistGBT (untuned)", hgbt_base,  X_val_t, y_val),
    evaluate("RF (tuned)",        rf_tuned,   X_val_t, y_val),
    evaluate("RF (untuned)",      rf_base,    X_val_t, y_val),
]).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
print("\n── Tuned vs untuned — both models (validation) ──")
display(comparison)

delta_hgbt = round(
    evaluate("HistGBT (tuned)", hgbt_tuned, X_val_t, y_val)["PR-AUC"]
    - evaluate("HistGBT (untuned)", hgbt_base, X_val_t, y_val)["PR-AUC"], 4)
delta_rf = round(
    evaluate("RF (tuned)", rf_tuned, X_val_t, y_val)["PR-AUC"]
    - evaluate("RF (untuned)", rf_base, X_val_t, y_val)["PR-AUC"], 4)
print(f"\nTuning gain — HistGBT: ΔPR-AUC = {delta_hgbt:+.4f}")
print(f"Tuning gain — RF:     ΔPR-AUC = {delta_rf:+.4f}")
```

### 5.2  Tuning Both Shortlisted Candidates

`RandomizedSearchCV` with 3-fold CV on **training data only**, scoring
on `average_precision` (= PR-AUC).  Eight random combinations per model,
48 fits total.  The validation set is not seen by the search; it is
evaluated once afterwards.

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (tuned)   | 0.7324 | 0.8888 | 0.6471 | 0.6600 |
| RF (tuned)        | 0.7269 | 0.8825 | 0.6503 | 0.6633 |
| HistGBT (untuned) | 0.7252 | 0.8781 | 0.6471 | 0.6600 |
| RF (untuned)      | 0.7042 | 0.8722 | 0.6373 | 0.6500 |

**PR-AUC (primary).** HistGBT's tuning gain is marginal (ΔPR-AUC =
+0.007), confirming that sklearn's defaults were already near-optimal
for this dataset.  RF benefits more substantially (ΔPR-AUC = +0.023),
largely because the search found `min_samples_leaf=5`, which allows
finer splits and reduces the under-fitting that hampered the default
configuration.  After tuning, the gap between the two models narrows to
just 0.005 (HistGBT 0.7324 vs RF 0.7269) — well within the range that
validation noise on a ~1,500-row set could reverse.

**ROC-AUC (secondary discrimination).** Tuning lifts both models'
ROC-AUC into the high-0.88 range (HistGBT 0.8888, RF 0.8825), up from
0.8781 and 0.8722 respectively.  This means the tuned models correctly
rank a random churner above a random stayer roughly 89 % of the time —
a one-percentage-point improvement that, while modest, confirms the
tuning search improved general discrimination and did not merely
overfit to the PR-AUC objective.

**Recall@top-20 % and Precision@top-20 % (business metrics).**
Interestingly, tuning barely moves HistGBT's top-20 % metrics: recall
remains at 0.6471 and precision at 0.6600, identical to the untuned
values.  This suggests HistGBT's ranking of the highest-risk customers
was already well-calibrated at the top of the distribution; tuning
improved the full-curve summary (PR-AUC) by refining discrimination in
the middle of the risk spectrum, not at the critical top tail.  RF's
picture is slightly different: recall rises from 0.6373 to 0.6503 and
precision from 0.6500 to 0.6633, indicating that tuning genuinely
improved RF's ranking of the highest-risk customers.  After tuning, RF
actually edges ahead of HistGBT on both business metrics (Recall
0.6503 vs 0.6471; Precision 0.6633 vs 0.6600), though the differences
are less than one percentage point.  This divergence between PR-AUC
(where HistGBT leads) and the top-20 % metrics (where RF leads) is not
contradictory: PR-AUC integrates the entire precision–recall curve,
while the top-20 % metrics reflect performance at a single operating
point.

Best parameters: HistGBT `{max_iter:300, max_depth:3, lr:0.05,
min_samples_leaf:20, class_weight:None}`; RF `{n_estimators:200,
min_samples_leaf:5, max_depth:None, class_weight:None}`.

---

## Cell 3 — 5.3 Lock all choices on validation

```python
# ── 5.3 Lock all choices on validation ────────────────────────────────────────
# BEFORE any test access, lock:
#   (a) Final model — by validation PR-AUC (pre-committed decision criterion)
#   (b) Operating rule — top-20% by predicted risk
#   (c) Threshold — 80th percentile of val probabilities for CM view

# ── (a) Lock final model ─────────────────────────────────────────────────────
hgbt_val_prauc = evaluate("HistGBT (tuned)", hgbt_tuned, X_val_t, y_val)["PR-AUC"]
rf_val_prauc   = evaluate("RF (tuned)",      rf_tuned,   X_val_t, y_val)["PR-AUC"]

FINAL_MODEL_NAME = "HistGBT (tuned)" if hgbt_val_prauc >= rf_val_prauc else "RF (tuned)"
final_model      = hgbt_tuned if hgbt_val_prauc >= rf_val_prauc else rf_tuned
runner_up_name   = "RF (tuned)" if FINAL_MODEL_NAME == "HistGBT (tuned)" else "HistGBT (tuned)"
runner_up_model  = rf_tuned    if FINAL_MODEL_NAME == "HistGBT (tuned)" else hgbt_tuned

print(f"LOCKED final model: {FINAL_MODEL_NAME}")
print(f"  Val PR-AUC: {FINAL_MODEL_NAME} = {max(hgbt_val_prauc, rf_val_prauc):.4f}  "
      f"vs  {runner_up_name} = {min(hgbt_val_prauc, rf_val_prauc):.4f}")
print(f"  Decision: pre-committed criterion (highest val PR-AUC)")

# ── (b) Operating rule: top-20% ranking ───────────────────────────────────────
proba_val  = final_model.predict_proba(X_val_t)[:, 1]
n_top      = max(1, int(len(y_val) * TOP_PCT))
top_arr    = np.zeros(len(y_val), dtype=int)
top_arr[np.argsort(proba_val)[::-1][:n_top]] = 1
rec_top20  = round(recall_score(y_val, top_arr), 4)
prec_top20 = round(precision_score(y_val, top_arr, zero_division=0), 4)

# ── (c) Threshold: 80th percentile of val probabilities ──────────────────────
LOCKED_THRESH = round(float(np.percentile(proba_val, 100 * (1 - TOP_PCT))), 4)

pred_thr    = (proba_val >= LOCKED_THRESH).astype(int)
rec_thr     = round(recall_score(y_val, pred_thr), 4)
prec_thr    = round(precision_score(y_val, pred_thr, zero_division=0), 4)
pct_flagged = round(100 * pred_thr.sum() / len(y_val), 1)

# ── Naive 0.5 baseline (for context) ─────────────────────────────────────────
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
print(f"\n── Operating-rule comparison ({FINAL_MODEL_NAME} on validation) ──")
display(rule_df)

print(f"\n{'='*60}")
print(f"ALL CHOICES LOCKED (validation only — test not accessed):")
print(f"  Final model:    {FINAL_MODEL_NAME}")
print(f"  Runner-up:      {runner_up_name}")
print(f"  Operating rule: top-20% ranking")
print(f"  CM threshold:   {LOCKED_THRESH}")
print(f"{'='*60}")
```

### 5.3  Lock All Choices on Validation

Three choices are locked using **validation data only**, before any test
access:

**(a) Final model.**  HistGBT (tuned) leads with val PR-AUC = 0.7324 vs
RF = 0.7269.  The pre-committed criterion is "highest validation PR-AUC",
so HistGBT is locked.  RF is carried forward for diagnostic comparison
only — it cannot override this decision.

**(b) Operating rule.**  The campaign contacts exactly 20 % of customers.
A top-20 % ranking fills every slot regardless of calibration.  Threshold
0.5 flags only 12.3 % — wasting capacity and dropping recall from 0.6471
to 0.4673.

| Rule | Flagged (%) | Recall | Precision |
|------|-------------|--------|-----------|
| Top-20 % ranking (main) | 20.0 | 0.6471 | 0.6600 |
| Threshold ≈ 20 % (t=0.3235) | 20.0 | 0.6471 | 0.6600 |
| Threshold 0.5 (naive) | 12.3 | 0.4673 | 0.7772 |

**(c) Threshold for confusion matrix.**  The 80th percentile of the
locked model's validation probabilities gives threshold 0.3235, which
flags exactly 20 % — consistent with the ranking rule.

**All choices locked.  Test set: not yet accessed.**

---

## Cell 4 — 5.4 Final test evaluation (report only)

```python
# ── 5.4 FINAL TEST EVALUATION (report only) ──────────────────────────────────
# All choices locked in Cell 5.3 using validation data only.
# Test set accessed here for the FIRST time — for reporting, not selection.

# Both models' probabilities (needed for Cell 5.5 diagnostics)
proba_test_hgbt = hgbt_tuned.predict_proba(X_test_t)[:, 1]
proba_test_rf   = rf_tuned.predict_proba(X_test_t)[:, 1]
proba_test      = proba_test_hgbt if "HistGBT" in FINAL_MODEL_NAME else proba_test_rf

# ── Official result: locked final model ───────────────────────────────────────
test_final = evaluate(FINAL_MODEL_NAME, final_model, X_test_t, y_test)
print("=== OFFICIAL TEST RESULT — locked final model ===")
display(pd.DataFrame([test_final]))

pr_auc_test   = test_final["PR-AUC"]
roc_auc_test  = test_final["ROC-AUC"]
rec_top_test  = test_final["Recall@top20%"]
prec_top_test = test_final["Precision@top20%"]

# ── Threshold-based metrics at LOCKED_THRESH ──────────────────────────────────
pred_locked = (proba_test >= LOCKED_THRESH).astype(int)
rec_locked  = round(recall_score(y_test, pred_locked), 4)
prec_locked = round(precision_score(y_test, pred_locked, zero_division=0), 4)
n_flagged   = int(pred_locked.sum())
n_test      = len(y_test)

print(f"\nFlagged {n_flagged}/{n_test} ({100*n_flagged/n_test:.1f}%) "
      f"at locked threshold {LOCKED_THRESH}")

# ── Runner-up (for context only — NOT used for selection) ─────────────────────
test_runner = evaluate(runner_up_name, runner_up_model, X_test_t, y_test)
print("\n── For context: runner-up on test (decision already locked) ──")
display(pd.DataFrame([test_runner]))

# ── Val → test stability check ───────────────────────────────────────────────
val_prauc_final  = max(hgbt_val_prauc, rf_val_prauc)
val_prauc_runner = min(hgbt_val_prauc, rf_val_prauc)
print(f"\nVal → test PR-AUC shift:")
print(f"  {FINAL_MODEL_NAME}: {val_prauc_final:.4f} → {pr_auc_test:.4f} "
      f"(Δ = {pr_auc_test - val_prauc_final:+.4f})")
print(f"  {runner_up_name}:  {val_prauc_runner:.4f} → {test_runner['PR-AUC']:.4f} "
      f"(Δ = {test_runner['PR-AUC'] - val_prauc_runner:+.4f})")
print("\nTest is not used for any decisions — all choices were locked in Cell 5.3.")
```

### 5.4  Final Test Evaluation (Report Only)

The final model was locked in Cell 5.3 on validation evidence.  Test is
accessed here **solely for reporting** — no selection or threshold tuning.

**Official result — locked final model (HistGBT (tuned)):**

| Metric | Value |
|--------|-------|
| PR-AUC | **0.7344** |
| ROC-AUC | **0.8737** |
| Recall@top-20% | **0.6426** |
| Precision@top-20% | **0.6533** |
| Customers flagged at threshold 0.3235 | **296 / 1500 (19.7 %)** |

**Runner-up on test (for context — decision already locked):**

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| RF (tuned) | 0.7079 | 0.8605 | 0.6197 | 0.6300 |

**Interpreting the test metrics.**  On the primary metric, HistGBT
achieves a test PR-AUC of 0.7344, indicating that the model maintains
strong ranking quality across the full precision–recall trade-off on
unseen data.  This is well above the no-skill baseline of approximately
0.20 (churn prevalence) and closely matches the validation figure of
0.7324, suggesting the model has not overfit to the training data.
ROC-AUC on test is 0.8737, meaning the model correctly ranks a randomly
drawn churner above a randomly drawn stayer roughly 87 % of the time —
consistent with the validation ROC-AUC of 0.8888, though slightly lower,
which is normal for a held-out set.

The business-level metrics translate these rankings into campaign
outcomes.  Recall@top-20 % of 0.6426 means that if the bank contacts
its top-ranked fifth of customers, the campaign reaches approximately
64 % of all actual churners in the test set — a strong result given that
the 20 % budget imposes a hard ceiling on how many churners can
possibly be captured.  Precision@top-20 % of 0.6533 indicates that
roughly two in three customers flagged for retention are genuine
churners, keeping the wasted-intervention rate at about one in three.
For a retention campaign where the cost of a false alarm (one
unnecessary phone call) is low relative to the cost of missing a
genuine churner (lost revenue), this trade-off is operationally
acceptable.

The runner-up RF trails on every metric: PR-AUC 0.7079 (−0.027),
ROC-AUC 0.8605 (−0.013), Recall@top-20 % 0.6197 (−0.023), and
Precision@top-20 % 0.6300 (−0.023).  The gap is wider on test than it
was on validation, suggesting RF's ranking generalises less stably.

**Val → test stability:**

| Model | Val PR-AUC | Test PR-AUC | Δ |
|-------|-----------|------------|---|
| HistGBT (tuned) | 0.7324 | 0.7344 | +0.0020 |
| RF (tuned) | 0.7269 | 0.7079 | −0.0190 |

HistGBT is remarkably stable (Δ = +0.002, essentially flat), which is
the ideal outcome: it suggests the model's ranking ability transfers
reliably to new data.  RF drops by 0.019 — on the edge of the ≤ 0.02
normal-range threshold but not alarming in isolation.  Crucially,
HistGBT leads on both validation and test across all four metrics, so
the locked decision is confirmed — no conflict between validation
ranking and test ranking.

---

## Cell 5 — 5.5 Error analysis

```python
# ── 5.5 Error analysis ─────────────────────────────────────────────────────────
# Depends on: proba_test_hgbt, proba_test_rf, proba_test, pred_locked,
#             LOCKED_THRESH, FINAL_MODEL_NAME, runner_up_name  (Cells 5.3–5.4)
# 4 diagnostics: (a) confusion matrix, (b) PR curve, (c) calibration,
#                (d) geography failure-mode slice
# The final model was locked on validation in Cell 5.3.
# Diagnostics (b)–(d) show both models for understanding — they are
# post-hoc consistency checks, NOT selection criteria.

import matplotlib.pyplot as plt
from sklearn.metrics     import ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import os

os.makedirs("outputs", exist_ok=True)

# ── (a) Confusion matrix at LOCKED_THRESH (final model only) ─────────────────
fig1, ax1 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred_locked,
    display_labels=["Stay", "Churn"],
    cmap="Blues", colorbar=False, ax=ax1,
)
ax1.set_title(f"Confusion Matrix\n"
              f"{FINAL_MODEL_NAME}  |  thr = {LOCKED_THRESH}  |  test set")
fig1.tight_layout()
fig1.savefig("outputs/5a_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5a_confusion_matrix.png")

# ── (b) PR curve — both models on test ────────────────────────────────────────
prec_h, rec_h, _ = precision_recall_curve(y_test, proba_test_hgbt)
prec_r, rec_r, _ = precision_recall_curve(y_test, proba_test_rf)

prauc_hgbt = round(average_precision_score(y_test, proba_test_hgbt), 4)
prauc_rf   = round(average_precision_score(y_test, proba_test_rf), 4)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(rec_h, prec_h, label=f"HistGBT (tuned)  PR-AUC = {prauc_hgbt}")
ax2.plot(rec_r, prec_r, label=f"RF (tuned)       PR-AUC = {prauc_rf}",
         linestyle="--")
ax2.axhline(y_test.mean(), color="gray", linestyle=":",
            label=f"No-skill ({y_test.mean():.3f})")
ax2.scatter([rec_locked], [prec_locked], zorder=5, color="red", s=80,
            label=f"Locked threshold ({LOCKED_THRESH})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR Curve — both shortlisted models (test)")
ax2.legend(loc="upper right", fontsize=8)
fig2.tight_layout()
fig2.savefig("outputs/5b_pr_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5b_pr_curve.png")

# ── (c) Calibration — both models side-by-side ───────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

for proba, label, color in [
    (proba_test_hgbt, "HistGBT (tuned)", "tab:blue"),
    (proba_test_rf,   "RF (tuned)",      "tab:orange"),
]:
    pt, pp = calibration_curve(y_test, proba, n_bins=10)
    axes3[0].plot(pp, pt, marker="o", label=label, color=color)
axes3[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
axes3[0].set_xlabel("Mean predicted probability")
axes3[0].set_ylabel("Fraction of positives")
axes3[0].set_title("Calibration Curve (test)")
axes3[0].legend(fontsize=8)

# Score-distribution histogram — final model only (for clarity)
axes3[1].hist(proba_test[y_test == 0], bins=30, alpha=0.6,
              label="Stay", color="steelblue", density=True)
axes3[1].hist(proba_test[y_test == 1], bins=30, alpha=0.6,
              label="Churn", color="tomato", density=True)
axes3[1].axvline(LOCKED_THRESH, color="black", linestyle="--",
                 label=f"thr = {LOCKED_THRESH}")
axes3[1].set_xlabel("Predicted probability"); axes3[1].set_ylabel("Density")
axes3[1].set_title(f"Score distribution — {FINAL_MODEL_NAME} (test)")
axes3[1].legend(fontsize=8)

fig3.tight_layout()
fig3.savefig("outputs/5c_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5c_calibration.png")

# ── (d) Failure-mode slice: Geography — both models ──────────────────────────
geo_vals    = df_model.loc[y_test.index, "Geography"].values
y_test_vals = y_test.values

geo_rows = []
for model_name, proba_m in [("HistGBT", proba_test_hgbt),
                              ("RF",      proba_test_rf)]:
    for geo in sorted(set(geo_vals)):
        mask = geo_vals == geo
        y_sl = y_test_vals[mask]
        p_sl = proba_m[mask]
        if y_sl.sum() == 0:
            continue
        rec_sl, prec_sl = recall_precision_top(pd.Series(y_sl), p_sl)
        geo_rows.append({
            "Model":            model_name,
            "Geography":        geo,
            "N":                int(len(y_sl)),
            "Churners":         int(y_sl.sum()),
            "Churn rate":       round(float(y_sl.mean()), 4),
            "PR-AUC":           round(float(average_precision_score(y_sl, p_sl)), 4),
            "Recall@top20%":    rec_sl,
        })

geo_df = pd.DataFrame(geo_rows)
print("\n── Failure-mode slice: Geography (test set, both models) ──")
display(geo_df.pivot_table(index="Geography",
                            columns="Model",
                            values=["PR-AUC", "Recall@top20%"],
                            aggfunc="first"))

# ── Post-decision diagnostics ─────────────────────────────────────────────────
# The final model was ALREADY locked on validation PR-AUC (Cell 5.3).
# The checks below are post-decision diagnostics — they CANNOT change the decision.
# Diagnostic 1: Calibration — which model's reliability curve is closer to diagonal?
# (Visual comparison from plot above — document in report text)
# Diagnostic 2: Geography fairness — smallest max–min PR-AUC gap across countries
for model_name, proba_m in [("HistGBT", proba_test_hgbt),
                              ("RF",      proba_test_rf)]:
    sub = geo_df[geo_df["Model"] == model_name]
    gap = sub["PR-AUC"].max() - sub["PR-AUC"].min()
    print(f"{model_name} geography PR-AUC gap (max-min): {gap:.4f}")

print(f"\n=== FINAL MODEL (locked in Cell 5.3): {FINAL_MODEL_NAME} ===")
```

### 5.5  Error Analysis

**(a) Confusion matrix (threshold = 0.3235, final model = HistGBT (tuned)):**

At threshold 0.3235 the model flags 296 / 1500 test customers (19.7 %).
Read TP, FP, FN, TN from the saved figure (`outputs/5a_confusion_matrix.png`).
The campaign accepts some FP (wasted retention calls) to minimise FN
(missed churners) — this trade-off is captured by Precision@top-20 %
(0.6533): roughly 1 in 3 flagged customers is a false alarm, which is
acceptable given that missing a churner is costlier than one unnecessary
call.

**(b) PR curve — both models on test:**

HistGBT (PR-AUC = 0.7344) and RF (PR-AUC = 0.7079) are plotted together.
HistGBT's curve sits above RF's across most recall levels, consistent with
the 0.027 PR-AUC gap.  The locked threshold (0.3235) is marked as a red
dot.  Both curves are well above the no-skill baseline (~0.20).  Refer to
`outputs/5b_pr_curve.png` for the visual.

**(c) Calibration — both models:**

Refer to `outputs/5c_calibration.png`.  This is a post-decision
diagnostic — it cannot change the locked choice.  Note which model's
reliability curve tracks the diagonal more closely, and whether one
over-predicts (bows above) or under-predicts (bows below).  In the
score-distribution histogram, good class separation means the locked
threshold (0.3235) sits in a low-density region between the Stay and
Churn distributions.

**(d) Geography slice — both models:**

| Geography | HistGBT PR-AUC | RF PR-AUC | HistGBT Recall@top20% | RF Recall@top20% |
|-----------|---------------|-----------|----------------------|-----------------|
| France  | 0.6587 | 0.6211 | 0.6119 | 0.6119 |
| Germany | 0.8166 | 0.7980 | 0.5702 | 0.5526 |
| Spain   | 0.7528 | 0.7471 | 0.7193 | 0.7018 |

HistGBT geography PR-AUC gap (max−min): 0.1579.
RF geography PR-AUC gap (max−min): 0.1769.

HistGBT has the smaller gap (0.1579 vs 0.1769) — fairer across
geographies.  Germany has the highest PR-AUC (0.8166) but the lowest
Recall@top-20 % (0.5702) — a ceiling effect: with ~32 % churners and
only 20 % flagged, maximum recall ≈ 0.625.  Spain (~16 % churn)
achieves the highest recall (0.7193) because the budget captures a
larger share of its smaller churner pool.  In deployment, Germany's
higher miss rate should be monitored.

**Post-decision diagnostics (cannot override the locked choice):**

| Check | Favours |
|-------|---------|
| Decision criterion (val PR-AUC) | HistGBT (0.7324 vs 0.7269) |
| Calibration quality | (see calibration plot) |
| Geography fairness | HistGBT (gap 0.158 vs 0.177) |

The geography diagnostic confirms the locked choice.  Even if one
diagnostic favoured RF, the decision would not be reversed — it was
pre-committed on validation to avoid test-based selection.

**Final model (locked): HistGBT (tuned)**

---

### 5.6  Agent-Made Mistake: Test-Based Selection Leak

**Mistake.**  The agent's original Cell 5.4 code selected the final model
by sorting on **test-set** PR-AUC (`test_results.sort_values("PR-AUC").iloc[0]`).
This made the reported test metrics also the selection criteria — so
they were optimistically biased.  If RF happened to beat HistGBT on test
despite losing on validation, the agent would have switched and reported
RF's test numbers as "final" — a textbook data-leakage risk.

**How I caught it.**  While reviewing Cell 5.4 I noticed that
`FINAL_MODEL_NAME` was set *after* test access, not before.  The pipeline
was supposed to follow a strict train → validate → test discipline, but
the agent skipped the "decide on validation" step entirely.

**Fix.**  I restructured the pipeline: Cell 5.3 now locks the final model
on **validation** PR-AUC before any test access.  Cell 5.4 is pure
reporting — it displays test metrics for the already-locked model and the
runner-up, but cannot change the decision.  This ensures the reported
test metrics are unbiased estimates of real-world performance.

---

## Model Card

**Model:** HistGradientBoostingClassifier (tuned) — sklearn native.
Selected on validation PR-AUC after tuning (Cell 5.3); test set used only
for final reporting and diagnostics (Cells 5.4–5.5).

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
| PR-AUC | 0.7344 |
| ROC-AUC | 0.8737 |
| Recall@top-20% | 0.6426 |
| Precision@top-20% | 0.6533 |

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
| Shortlist scope | Agent initially designed Task 5 for a single model (HistGBT only). I requested carrying both HistGBT and RF into Task 5, because the PR-AUC gap from Task 4 was < 0.01 (noise) and the two models have structurally different failure modes | I confirmed: (1) tuning budget remains small (2 × 8 × 3 = 48 fits on 7k rows); (2) pre-committed validation PR-AUC as the decision criterion before seeing Task 5 results; (3) error analysis diagnostics (calibration, geography) now compare both models side-by-side as post-decision checks |
| Baseline aliases | Agent originally re-fit both models from scratch (redundant — same notebook session, same SEED). I replaced with a one-line alias (`hgbt_base, rf_base = hgbt, rf`) — no re-fitting needed | I confirmed: aliased models produce the same val PR-AUC as Task 4 (HistGBT 0.7252, RF 0.7042) since they are the same objects in memory |
| Tuning | `RandomizedSearchCV` on both models: n\_iter=8 each, cv=3, `scoring="average_precision"`, training only | I confirmed: HistGBT best params `{min_samples_leaf:20, max_iter:300, max_depth:3, lr:0.05, class_weight:None}`, ΔPR-AUC = +0.0072 (marginal). RF best params `{n_estimators:200, min_samples_leaf:5, max_depth:None, class_weight:None}`, ΔPR-AUC = +0.0227 (meaningful). `max_iter=300` is at grid boundary for HistGBT but marginal gain makes extension unnecessary. HistGBT leads on val PR-AUC (0.7324 vs 0.7269) but gap is within noise |
| Test-based selection leak | Agent originally selected the final model by **test** PR-AUC in Cell 5.4 (`test_results.iloc[0]`), making the reported test metrics optimistically biased. I restructured: Cell 5.3 now locks the final model on **validation** PR-AUC (pre-committed criterion); Cell 5.4 is pure reporting. Cell 5.5 diagnostics are reframed as post-decision checks that cannot override the locked decision | I confirmed: (1) no test data is accessed before Cell 5.4; (2) `FINAL_MODEL_NAME` is set in Cell 5.3 using `hgbt_val_prauc` vs `rf_val_prauc`; (3) Cell 5.4 cannot change the decision; (4) wording changed from "test accessed once" to accurate description |
| Operating rule | Locked top-20 % ranking as main rule; derived threshold from locked model's 80th-percentile val probabilities for CM view; showed 0.5 threshold is too conservative | I confirmed: LOCKED_THRESH = 0.3235 flags exactly 20.0 % on validation with Recall 0.6471 and Precision 0.6600 — identical to the top-20 % ranking. Threshold 0.5 flags only 12.3 % (Recall 0.4673) — wastes 7.7 pp of campaign capacity. Test set was NOT accessed in Cell 5.3 |
| Test evaluation | Locked final model evaluated on test (report only); runner-up shown for context, clearly labelled | I confirmed: HistGBT test PR-AUC = 0.7344 (val 0.7324, Δ = +0.002 — remarkably stable). RF test PR-AUC = 0.7079 (val 0.7269, Δ = −0.019 — within normal range). Runner-up RF does **not** beat the locked model on test (0.7079 < 0.7344) — no conflict. Decision is NOT switched |
| Error analysis (CM) | Confusion matrix at locked threshold for final model; saved to `outputs/5a_confusion_matrix.png` | I confirmed figure saved. Threshold 0.3235 flags 296/1500 (19.7 %) on test. Read TP/FP/FN/TN from the saved figure. Precision@top-20 % = 0.6533 → roughly 1 in 3 flagged customers is a false alarm, acceptable given the asymmetric cost of missing churners |
| Error analysis (PR curve) | PR curve comparing both models on test; locked threshold marked as red dot | I confirmed: HistGBT curve sits above RF across most recall levels, consistent with the 0.027 test PR-AUC gap. Both curves well above the no-skill baseline (~0.20). Red dot marks locked threshold. Figure saved to `outputs/5b_pr_curve.png` |
| Error analysis (calibration) | Calibration reliability curves for both models on same axes + score histogram for final model; saved to `outputs/5c_calibration.png` | Post-decision diagnostic — cannot change the locked choice. Assess from figure which model tracks the diagonal more closely. Score-distribution histogram shows class separation at locked threshold 0.3235 |
| Error analysis (geography) | Per-country PR-AUC and Recall@top-20% for both models; computes max–min PR-AUC gap per model | HistGBT gap = 0.1579, RF gap = 0.1769 → HistGBT is fairer. Post-decision diagnostic — cannot change the locked choice, but consistent with it. Germany has highest PR-AUC (0.8166) but lowest Recall@top-20 % (0.5702) — ceiling effect from 32 % churn rate vs 20 % budget. Spain best recall (0.7193) |
| Final selection | Final model locked on **validation** PR-AUC in Cell 5.3. Calibration and geography shown as post-decision diagnostics on test in Cell 5.5 — they cannot override | I confirmed: geography diagnostic favours HistGBT (smaller gap). HistGBT also wins on test PR-AUC (0.7344 vs 0.7079). Post-decision diagnostics are consistent with the locked choice. **Final model: HistGBT (tuned)** |
| Model card | Agent drafted model card with intended use, limitations, data constraints, evaluation caveats, and 4-metric summary table | I confirmed: model card now correctly states HistGradientBoostingClassifier (tuned), selected on validation PR-AUC, test for reporting only. Metrics filled: PR-AUC 0.7344, ROC-AUC 0.8737, Recall@top-20% 0.6426, Precision@top-20% 0.6533 |
| `calibration_curve` import error (Cell 5.5) | Agent wrote `from sklearn.metrics import ConfusionMatrixDisplay, calibration_curve` — importing `calibration_curve` from `sklearn.metrics`. This raises `ImportError` in recent sklearn versions because `calibration_curve` lives in `sklearn.calibration`, not `sklearn.metrics` | I caught the `ImportError` at runtime, identified the correct module (`sklearn.calibration`), and split the import into two lines: `from sklearn.metrics import ConfusionMatrixDisplay` and `from sklearn.calibration import calibration_curve`. Cell 5.5 now runs without error |
| `MallocStackLogging` warnings (Cell 5.2) | Agent used `n_jobs=-1` in both `RandomizedSearchCV` calls, which spawns parallel worker processes via `joblib`. On macOS, each child process emits `MallocStackLogging: can't turn off malloc stack logging because it was not enabled` | I confirmed these are **harmless macOS system-level messages** — they do not affect tuning results, model parameters, or metrics. No code change needed. Setting `os.environ["MallocStackLogging"] = "0"` before the cell suppresses them cosmetically if desired |
| Stale "final pick on test set" print (Cell 5.3) | Agent's Cell 5.3 printed "Both models still carried forward — final pick on test set" — leftover from the original test-based selection design. This contradicts the validation-only selection logic: the final model is already locked in Cell 5.3, not chosen on test | I caught this at runtime. The print statement was a remnant of the pre-revision code (before the test-based selection leak was fixed). Corrected to "ALL CHOICES LOCKED (validation only — test not accessed)" in the current draft |
| Stale "Step 1 / Steps 2–3" language (Cell 5.4) | Agent's Cell 5.4 printed "Step 1 (PR-AUC): winner = HistGBT (tuned) (Steps 2–3 — calibration + geography — evaluated in Cell 5.5)" — implying selection happens on test and using step-numbering that confuses decision criteria with diagnostics | I caught this at runtime. The language was a remnant of the "4-step decision rule" framing. Current draft code says "OFFICIAL TEST RESULT — locked final model" and does not use step numbering. Calibration and geography are now called "post-decision diagnostics" throughout |
