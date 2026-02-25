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
   (a) **Final model** — by validation PR-AUC (pre-committed criterion).
   (b) **Operating rule** — top-20 % by predicted risk.
   (c) **Threshold** — 80th percentile of the locked model's validation
       probabilities, for confusion-matrix view.
   Runner-up is noted for comparison; it does not appear in later cells.

4. **5.4 Final test evaluation.**  Test set accessed for the first time.
   Only the locked final model is evaluated.

5. **5.5 Error analysis — 4 diagnostics (final model only).**
   - **(a)** Confusion matrix at the locked threshold
   - **(b)** PR curve — validation vs test overlay (generalisation check)
   - **(c)** Calibration diagram + score distribution
   - **(d)** Geography failure-mode slice
   Save figures to `outputs/`.

Constraints: test is not used for tuning or model selection; it is
accessed only after all choices are locked (Cell 5.3).
`random_state=SEED` throughout.  Tuning budget: 2 × 8 × 3 = 48 fits.

### Leakage-Safe Decision Logic

The pipeline enforces a strict **train → validate → test** order:

1. **Tune on training only** (Cell 5.2): `RandomizedSearchCV` with 3-fold
   CV on `X_train_t`.  The validation set is not seen by the search.
2. **Decide on validation only** (Cell 5.3): The final model, operating
   rule, and threshold are all locked using validation metrics — before
   any test access.
3. **Report on test** (Cells 5.4–5.5): The test set is accessed only
   after all choices are locked, for evaluation and diagnostics.

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
48 fits total.

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (tuned)   | 0.7324 | 0.8888 | 0.6471 | 0.6600 |
| RF (tuned)        | 0.7269 | 0.8825 | 0.6503 | 0.6633 |
| HistGBT (untuned) | 0.7252 | 0.8781 | 0.6471 | 0.6600 |
| RF (untuned)      | 0.7042 | 0.8722 | 0.6373 | 0.6500 |

**PR-AUC (primary).**  HistGBT's tuning gain is marginal (ΔPR-AUC =
+0.007), confirming sklearn's defaults were already near-optimal.  RF
benefits more (ΔPR-AUC = +0.023), largely from `min_samples_leaf=5`
allowing finer splits.  After tuning, the gap narrows to 0.005 — within
validation noise on ~1,500 rows.

**ROC-AUC (secondary).**  Both models reach the high-0.88 range after
tuning, a one-point improvement confirming the search improved general
discrimination without overfitting to the PR-AUC objective.

**Business metrics (Recall / Precision@top-20 %).**  Tuning barely moves
HistGBT's top-20 % metrics (recall stays 0.6471, precision 0.6600),
suggesting its ranking at the top tail was already well-calibrated;
the gain came from refining discrimination in the middle of the risk
spectrum.  RF's recall rises from 0.6373 to 0.6503 and precision from
0.6500 to 0.6633 — edging ahead of HistGBT on both business metrics,
though by less than one percentage point.  This divergence (HistGBT leads
PR-AUC; RF leads top-20 %) is not contradictory: PR-AUC integrates the
entire curve, while top-20 % reflects a single operating point.

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

print(f"LOCKED final model: {FINAL_MODEL_NAME}")
print(f"  Val PR-AUC: {FINAL_MODEL_NAME} = {max(hgbt_val_prauc, rf_val_prauc):.4f}  "
      f"vs  {runner_up_name} = {min(hgbt_val_prauc, rf_val_prauc):.4f}")
print(f"  Criterion: highest validation PR-AUC (pre-committed)")

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
print(f"  Operating rule: top-20% ranking")
print(f"  CM threshold:   {LOCKED_THRESH}")
print(f"{'='*60}")
```

### 5.3  Lock All Choices on Validation

Three choices are locked using **validation data only**, before any test
access:

**(a) Final model.**  HistGBT (tuned) leads with val PR-AUC = 0.7324 vs
RF = 0.7269.  The pre-committed criterion is "highest validation PR-AUC",
so HistGBT is locked.  The runner-up is not carried into later cells.

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

## Cell 4 — 5.4 Final test evaluation

```python
# ── 5.4 FINAL TEST EVALUATION ────────────────────────────────────────────────
# All choices locked in Cell 5.3. Test accessed here for the first time.

proba_test = final_model.predict_proba(X_test_t)[:, 1]

# ── Official result ──────────────────────────────────────────────────────────
test_final = evaluate(FINAL_MODEL_NAME, final_model, X_test_t, y_test)
print("=== OFFICIAL TEST RESULT ===")
display(pd.DataFrame([test_final]))

pr_auc_test   = test_final["PR-AUC"]
roc_auc_test  = test_final["ROC-AUC"]
rec_top_test  = test_final["Recall@top20%"]
prec_top_test = test_final["Precision@top20%"]

# ── Threshold-based metrics ──────────────────────────────────────────────────
pred_locked = (proba_test >= LOCKED_THRESH).astype(int)
rec_locked  = round(recall_score(y_test, pred_locked), 4)
prec_locked = round(precision_score(y_test, pred_locked, zero_division=0), 4)
n_flagged   = int(pred_locked.sum())
n_test      = len(y_test)

print(f"\nFlagged {n_flagged}/{n_test} ({100*n_flagged/n_test:.1f}%) "
      f"at locked threshold {LOCKED_THRESH}")

# ── Val → test stability ────────────────────────────────────────────────────
val_prauc_final = max(hgbt_val_prauc, rf_val_prauc)
print(f"\nVal → test PR-AUC: {val_prauc_final:.4f} → {pr_auc_test:.4f} "
      f"(Δ = {pr_auc_test - val_prauc_final:+.4f})")
```

### 5.4  Final Test Evaluation

| Metric | Value |
|--------|-------|
| PR-AUC | **0.7344** |
| ROC-AUC | **0.8737** |
| Recall@top-20% | **0.6426** |
| Precision@top-20% | **0.6533** |
| Customers flagged at threshold 0.3235 | **296 / 1500 (19.7 %)** |

**Ranking quality.**  Test PR-AUC of 0.7344 is well above the no-skill
baseline (~0.20) and closely matches validation (0.7324, Δ = +0.002),
indicating the model has not overfit.  ROC-AUC of 0.8737 means the model
correctly ranks a random churner above a random stayer roughly 87 % of
the time.

**Business impact.**  Recall@top-20 % of 0.6426 means the campaign
reaches ~64 % of actual churners by contacting the top-ranked fifth.
Precision@top-20 % of 0.6533 indicates roughly two in three flagged
customers are genuine churners — keeping the wasted-intervention rate
at about one in three.  For a retention campaign where a false alarm
(one unnecessary call) costs far less than a missed churner (lost
revenue), this trade-off is operationally acceptable.

**Stability.**  Val → test PR-AUC shift is +0.002, essentially flat —
the model's ranking generalises reliably.

---

## Cell 5 — 5.5 Error analysis

```python
# ── 5.5 Error analysis — final model only ─────────────────────────────────────
# 4 diagnostics on the locked final model:
#   (a) confusion matrix, (b) PR curve (val vs test),
#   (c) calibration, (d) geography failure-mode slice

import matplotlib.pyplot as plt
from sklearn.metrics     import ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import os

os.makedirs("outputs", exist_ok=True)

# ── (a) Confusion matrix at LOCKED_THRESH ────────────────────────────────────
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

# ── (b) PR curve — validation vs test (generalisation check) ─────────────────
prec_val_curve, rec_val_curve, _ = precision_recall_curve(y_val, proba_val)
prec_test_curve, rec_test_curve, _ = precision_recall_curve(y_test, proba_test)

prauc_val  = round(average_precision_score(y_val, proba_val), 4)
prauc_test = round(average_precision_score(y_test, proba_test), 4)

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(rec_val_curve, prec_val_curve,
         label=f"Validation  PR-AUC = {prauc_val}")
ax2.plot(rec_test_curve, prec_test_curve,
         label=f"Test  PR-AUC = {prauc_test}", linestyle="--")
ax2.axhline(y_test.mean(), color="gray", linestyle=":",
            label=f"No-skill ({y_test.mean():.3f})")
ax2.scatter([rec_locked], [prec_locked], zorder=5, color="red", s=80,
            label=f"Locked threshold ({LOCKED_THRESH})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title(f"PR Curve — {FINAL_MODEL_NAME} (val vs test)")
ax2.legend(loc="upper right", fontsize=8)
fig2.tight_layout()
fig2.savefig("outputs/5b_pr_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5b_pr_curve.png")

# ── (c) Calibration + score distribution ─────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

pt, pp = calibration_curve(y_test, proba_test, n_bins=10)
axes3[0].plot(pp, pt, marker="o", label=FINAL_MODEL_NAME, color="tab:blue")
axes3[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
axes3[0].set_xlabel("Mean predicted probability")
axes3[0].set_ylabel("Fraction of positives")
axes3[0].set_title(f"Calibration Curve — {FINAL_MODEL_NAME} (test)")
axes3[0].legend(fontsize=8)

axes3[1].hist(proba_test[y_test == 0], bins=30, alpha=0.6,
              label="Stay", color="steelblue", density=True)
axes3[1].hist(proba_test[y_test == 1], bins=30, alpha=0.6,
              label="Churn", color="tomato", density=True)
axes3[1].axvline(LOCKED_THRESH, color="black", linestyle="--",
                 label=f"thr = {LOCKED_THRESH}")
axes3[1].set_xlabel("Predicted probability"); axes3[1].set_ylabel("Density")
axes3[1].set_title(f"Score Distribution — {FINAL_MODEL_NAME} (test)")
axes3[1].legend(fontsize=8)

fig3.tight_layout()
fig3.savefig("outputs/5c_calibration.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: outputs/5c_calibration.png")

# ── (d) Failure-mode slice: Geography ────────────────────────────────────────
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
print(f"\n── Failure-mode slice: Geography — {FINAL_MODEL_NAME} (test) ──")
display(geo_df)

gap = geo_df["PR-AUC"].max() - geo_df["PR-AUC"].min()
print(f"PR-AUC gap across geographies (max−min): {gap:.4f}")

print(f"\n=== FINAL MODEL: {FINAL_MODEL_NAME} ===")
```

### 5.5  Error Analysis

**(a) Confusion matrix (threshold = 0.3235):**

At threshold 0.3235 the model flags 296 / 1500 test customers (19.7 %).
Precision@top-20 % = 0.6533 means roughly 1 in 3 flagged customers is a
false alarm — acceptable given that missing a churner costs more than one
unnecessary retention call.

**(b) PR curve — validation vs test:**

The two curves nearly overlap (val PR-AUC 0.7324, test 0.7344),
confirming the model's precision–recall trade-off generalises.  The
locked threshold (red dot) sits in the steep part of the curve where
small recall gains cost meaningful precision — consistent with the
top-20 % operating point chosen in Cell 5.3.

**(c) Calibration + score distribution:**

Assess from `outputs/5c_calibration.png`: does the reliability curve
track the diagonal, or does it bow above (under-confidence) or below
(over-confidence)?  In the score-distribution histogram, the locked
threshold (0.3235) should sit in a low-density gap between the Stay
and Churn distributions, indicating good class separation.

**(d) Geography slice:**

| Geography | N | Churners | Churn rate | PR-AUC | Recall@top20% |
|-----------|---|----------|------------|--------|---------------|
| France  | ... | ... | ... | 0.6587 | 0.6119 |
| Germany | ... | ... | ... | 0.8166 | 0.5702 |
| Spain   | ... | ... | ... | 0.7528 | 0.7193 |

PR-AUC gap across geographies: 0.1579.

Germany has the highest PR-AUC (0.8166) but the lowest Recall@top-20 %
(0.5702) — a ceiling effect: with ~32 % churn rate and only 20 % of
customers flagged, maximum achievable recall is ~0.625.  Spain (~16 %
churn) achieves the highest recall (0.7193) because the budget captures
a larger share of its smaller churner pool.  France sits in between on
both metrics.  In deployment, Germany's higher miss rate should be
monitored; a geography-specific budget allocation could improve equity.

---

### 5.6  Agent-Made Mistake: Test-Based Selection Leak

**Mistake.**  The agent's original Cell 5.4 code selected the final model
by sorting on **test-set** PR-AUC (`test_results.sort_values("PR-AUC").iloc[0]`).
This made the reported test metrics also the selection criteria — so
they were optimistically biased.

**How I caught it.**  While reviewing Cell 5.4 I noticed that
`FINAL_MODEL_NAME` was set *after* test access, not before.  The pipeline
was supposed to follow a strict train → validate → test discipline, but
the agent skipped the "decide on validation" step entirely.

**Fix.**  I restructured the pipeline: Cell 5.3 now locks the final model
on **validation** PR-AUC before any test access.  Cell 5.4 is pure
reporting — it displays test metrics for the already-locked model but
cannot change the decision.  This ensures the reported test metrics are
unbiased estimates of real-world performance.

---

## 5.7  My Agent vs My Verification

| Step | What My Agent Did | What I Verified or Corrected |
|------|--------------------|------------------------------|
| Test-based selection leak | Agent selected the final model by **test** PR-AUC in Cell 5.4 — making reported metrics optimistically biased | I restructured: Cell 5.3 locks the final model on **validation** PR-AUC before any test access. Cell 5.4 is pure reporting |
| `calibration_curve` import | Agent imported `calibration_curve` from `sklearn.metrics` — wrong module | I caught the `ImportError` at runtime and moved the import to `from sklearn.calibration import calibration_curve` |
| Stale print statements | Cell 5.3 printed "final pick on test set"; Cell 5.4 used "Step 1 / Steps 2–3" numbering from a discarded multi-step selection design | I replaced with accurate language: "ALL CHOICES LOCKED (validation only)" and "OFFICIAL TEST RESULT" |
| Metrics alignment | Agent initially used 3 metrics in Task 5 (no Precision@top-20%) | I added Precision@top-20% as a fourth metric across all tables, consistent with Section 1.3 |
| Shortlist → single model flow | Agent originally designed Task 5 for a single model; I expanded to tune both shortlisted models and compare on validation before locking | I confirmed: tuning budget stays small (48 fits), pre-committed criterion selects the winner, error analysis uses only the locked model |
| Tuning | `RandomizedSearchCV` on both models: n\_iter=8, cv=3, training only | I confirmed: HistGBT ΔPR-AUC = +0.007 (marginal), RF ΔPR-AUC = +0.023 (meaningful). `max_iter=300` at grid boundary but marginal gain makes extension unnecessary |
| Baseline aliases | Agent originally re-fit both models from scratch (redundant) | I replaced with `hgbt_base, rf_base = hgbt, rf` — no re-fitting needed, same objects in memory |
