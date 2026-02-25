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
   (a) **Final model** — by validation PR-AUC (Step 1 of pre-committed
       rule from Section 4.5).
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
   to justify the final selection per the pre-committed rule.

6. **5.6 Agent-made mistake and fix.**  Demonstrate the `predict()` vs
   `predict_proba()` bug for PR-AUC; show concrete impact; confirm fix.

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
   any test access.  Step 1 of the pre-committed decision rule (highest
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

`RandomizedSearchCV` searches a small hyperparameter space for each
shortlisted model using PR-AUC as the scoring metric (correct for
imbalanced data — see Section 5.6 for why ROC-AUC would mislead).
The validation set is **not** seen by either search; it is evaluated
once afterwards.

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| HistGBT (tuned)   | 0.7324 | 0.8888 | 0.6471 | 0.6600 |
| RF (tuned)        | 0.7269 | 0.8825 | 0.6503 | 0.6633 |
| HistGBT (untuned) | 0.7252 | 0.8781 | 0.6471 | 0.6600 |
| RF (untuned)      | 0.7042 | 0.8722 | 0.6373 | 0.6500 |

**HistGBT best parameters:** `min_samples_leaf=20, max_iter=300,
max_depth=3, learning_rate=0.05, class_weight=None`.
**RF best parameters:** `n_estimators=200, min_samples_leaf=5,
max_depth=None, class_weight=None`.

Tuning gain — HistGBT: ΔPR-AUC = +0.0072 (marginal).
Tuning gain — RF: ΔPR-AUC = +0.0227 (meaningful, > 0.005 threshold).

HistGBT's gain is marginal — the defaults were already near-optimal
(`max_depth=3`, `learning_rate=0.05` are conservative regularisation
values typical for boosting).  RF benefits more: the search found
`min_samples_leaf=5` (vs the sklearn default of 1), which slightly
reduces variance without over-smoothing.

**Boundary check:** `max_iter=300` is the grid maximum for HistGBT —
a larger budget might squeeze out another fraction, but the marginal
gain is already small so extending is unlikely to help.  RF's
`max_depth=None` (fully grown trees) was also in the grid, so the
search could have preferred a shallower depth but did not.

On validation PR-AUC, HistGBT (tuned) leads at 0.7324 vs RF (tuned)
0.7269 — a gap of 0.0055, still within noise.  The final selection
uses test metrics (Cell 5.4), not validation.

---

## Cell 3 — 5.3 Lock all choices on validation

```python
# ── 5.3 Lock all choices on validation ────────────────────────────────────────
# BEFORE any test access, lock:
#   (a) Final model — by validation PR-AUC (Step 1 of pre-committed rule)
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
print(f"  Decision: Step 1 of pre-committed rule (highest val PR-AUC)")

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

Three choices are locked here using **validation data only**, before any
test access:

**(a) Final model — validation PR-AUC (Step 1 of pre-committed rule).**
HistGBT (tuned) leads with val PR-AUC = 0.7324 vs RF (tuned) = 0.7269.
The gap (0.0055) is small but nonzero.  Because the pre-committed rule
(Section 4.5) specifies "highest PR-AUC" as Step 1, HistGBT is locked as
the final model.  The runner-up (RF) is carried forward only for
diagnostic comparison — it cannot override this decision.

**(b) Operating rule — top-20 % ranking.**
The retention campaign can contact exactly 20 % of customers, so the
primary rule is **top-20 % by predicted churn risk** — a ranking rule that
fills every slot regardless of probability calibration.

**(c) Threshold for confusion matrix.**
A threshold of *[fill]* is derived from the 80th percentile of the locked
model's validation probabilities.  This approximately flags 20 % of
customers, giving a confusion-matrix view consistent with the ranking rule.

| Rule | Flagged (%) | Recall | Precision |
|------|-------------|--------|-----------|
| Top-20 % ranking (main) | 20.0 | *[fill]* | *[fill]* |
| Threshold ≈ 20 % | *[fill]* | *[fill]* | *[fill]* |
| Threshold 0.5 (naive) | *[fill]* | *[fill]* | *[fill]* |

*[fill: note that threshold 0.5 flags far fewer than 20 %, wasting campaign
capacity and missing churners.  The ranking rule and the fitted threshold
give similar recall/precision since they both flag ~20 %.]*

**All choices are now locked:**
- Final model: *[fill: HistGBT (tuned) or RF (tuned)]*
- Operating rule: top-20 % ranking
- Threshold for CM: *[fill]*
- Test set: **not yet accessed**

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

**Official result — locked final model (*[fill: name]*):**

| Metric | Value |
|--------|-------|
| PR-AUC | **[fill]** |
| ROC-AUC | **[fill]** |
| Recall@top-20% | **[fill]** |
| Precision@top-20% | **[fill]** |
| Recall @ locked threshold | **[fill]** |
| Precision @ locked threshold | **[fill]** |
| Customers flagged (%) | **[fill]** |

**Runner-up on test (for context — decision already locked):**

| Model | PR-AUC | ROC-AUC | Recall@top20% | Precision@top20% |
|-------|--------|---------|---------------|------------------|
| *[fill]* | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

**Val → test stability:**
*[fill: report the ΔPR-AUC for each model.  A shift ≤ 0.02 is normal;
> 0.05 suggests validation overfitting.  If the runner-up happens to
beat the locked model on test, note this but do **not** switch — that
would be test-based selection.]*

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
from sklearn.metrics import ConfusionMatrixDisplay, calibration_curve
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

# ── Post-hoc consistency checks (Steps 2–3 of pre-committed rule) ────────────
# The final model was ALREADY locked on validation (Cell 5.3, Step 1).
# Steps 2–3 are shown here as diagnostics — they CANNOT change the decision.
# Step 2: Calibration — which model's reliability curve is closer to diagonal?
# (Visual comparison from plot above — document in report text)
# Step 3: Geography fairness — smallest max–min PR-AUC gap across countries
for model_name, proba_m in [("HistGBT", proba_test_hgbt),
                              ("RF",      proba_test_rf)]:
    sub = geo_df[geo_df["Model"] == model_name]
    gap = sub["PR-AUC"].max() - sub["PR-AUC"].min()
    print(f"{model_name} geography PR-AUC gap (max-min): {gap:.4f}")

print(f"\n=== FINAL MODEL (locked in Cell 5.3): {FINAL_MODEL_NAME} ===")
```

### 5.5  Error Analysis

**(a) Confusion matrix (threshold = [fill], final model = [fill]):**

*[fill: note TP (correct alerts), FP (wasted calls), FN (missed churners),
TN.  Relate back to cost asymmetry: we accept some FP to minimise FN,
consistent with Precision@top-20% measuring the wasted-intervention rate.]*

**(b) PR curve — both models on test:**

*[fill: do the two curves largely overlap?  Which model dominates in the
high-recall region (where the retention campaign operates)?  Is the gap
consistent with the PR-AUC numbers from Cell 5.4?]*

**(c) Calibration — both models:**

*[fill: which model's reliability curve tracks the diagonal more closely?
If one bows above (over-predicts) and the other bows below (under-predicts),
note this.  This is Step 2 of the pre-committed decision rule.  Comment
on class separation in the score-distribution histogram — good separation
means the locked threshold sits in a low-density region.]*

**(d) Geography slice — both models:**

| Geography | HistGBT PR-AUC | RF PR-AUC | HistGBT Recall@top20% | RF Recall@top20% |
|-----------|---------------|-----------|----------------------|-----------------|
| France  | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Germany | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| Spain   | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

HistGBT geography PR-AUC gap (max−min): *[fill]*.
RF geography PR-AUC gap (max−min): *[fill]*.

*[fill: which model has the smaller gap (fairer)?  This is Step 3 of the
pre-committed decision rule.  Germany has ~32 % churn vs ~16 % elsewhere
(EDA) — if PR-AUC is lower for Germany despite higher prevalence, flag
as a deployment risk.]*

**Post-hoc consistency check (decision already locked in Cell 5.3):**

The final model was locked on validation PR-AUC before test access.
Steps 2–3 are reported below as diagnostics — they cannot override the
locked decision.

| Step | Criterion | Applied on | Favours |
|------|-----------|------------|---------|
| 1 | PR-AUC (decision criterion) | **validation** (Cell 5.3) | *[fill]* |
| 2 | Calibration quality (diagnostic) | test (Cell 5.5c) | *[fill]* |
| 3 | Geography fairness (diagnostic) | test (Cell 5.5d) | *[fill]* |

*[fill: do Steps 2–3 confirm or contradict the locked choice?  If they
confirm, note the convergence.  If they contradict, acknowledge this
transparently but explain that the decision was pre-committed on
validation to avoid test-based selection.]*

**Final model (locked): [fill]**

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
proba_bug     = final_model.predict(X_val_t)              # BUG
prauc_bug     = round(average_precision_score(y_val, proba_bug), 4)

# CORRECT — continuous probabilities
proba_correct = final_model.predict_proba(X_val_t)[:, 1]  # FIX
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

**Model:** *[fill: HistGradientBoostingClassifier or RandomForestClassifier]*
(tuned) — sklearn native.  Selected by pre-committed decision rule
(Section 4.5) applied to test-set metrics and error-analysis diagnostics.

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
| Shortlist scope | Agent initially designed Task 5 for a single model (HistGBT only). I requested carrying both HistGBT and RF into Task 5, because the PR-AUC gap from Task 4 was < 0.01 (noise) and the two models have structurally different failure modes | I confirmed: (1) tuning budget remains small (2 × 8 × 3 = 48 fits on 7k rows); (2) pre-committed a 4-step decision rule before seeing Task 5 results; (3) error analysis diagnostics (calibration, geography) now compare both models side-by-side to justify the final pick |
| Baseline aliases | Agent originally re-fit both models from scratch (redundant — same notebook session, same SEED). I replaced with a one-line alias (`hgbt_base, rf_base = hgbt, rf`) — no re-fitting needed | I confirmed: aliased models produce the same val PR-AUC as Task 4 (HistGBT 0.7252, RF 0.7042) since they are the same objects in memory |
| Tuning | `RandomizedSearchCV` on both models: n\_iter=8 each, cv=3, `scoring="average_precision"`, training only | I confirmed: HistGBT best params `{min_samples_leaf:20, max_iter:300, max_depth:3, lr:0.05, class_weight:None}`, ΔPR-AUC = +0.0072 (marginal). RF best params `{n_estimators:200, min_samples_leaf:5, max_depth:None, class_weight:None}`, ΔPR-AUC = +0.0227 (meaningful). `max_iter=300` is at grid boundary for HistGBT but marginal gain makes extension unnecessary. HistGBT leads on val PR-AUC (0.7324 vs 0.7269) but gap is within noise |
| Test-based selection leak | Agent originally selected the final model by **test** PR-AUC in Cell 5.4 (`test_results.iloc[0]`), making the reported test metrics optimistically biased. I restructured: Cell 5.3 now locks the final model on **validation** PR-AUC (Step 1); Cell 5.4 is pure reporting. Cell 5.5 diagnostics are reframed as post-hoc consistency checks that cannot override the locked decision | I confirmed: (1) no test data is accessed before Cell 5.4; (2) `FINAL_MODEL_NAME` is set in Cell 5.3 using `hgbt_val_prauc` vs `rf_val_prauc`; (3) Cell 5.4 cannot change the decision; (4) wording changed from "test accessed once" to accurate description |
| Operating rule | Locked top-20 % ranking as main rule; derived threshold from locked model's 80th-percentile val probabilities for CM view; showed 0.5 threshold is too conservative | *[fill: confirm LOCKED_THRESH flags ~20 % on validation; confirm 0.5 flags far fewer; confirm test not accessed]* |
| Test evaluation | Locked final model evaluated on test (report only); runner-up shown for context, clearly labelled | *[fill: report val → test PR-AUC shift for each model; note if runner-up beats locked model on test — if so, do NOT switch]* |
| Error analysis (CM) | Confusion matrix at locked threshold for final model; saved to `outputs/5a_confusion_matrix.png` | *[fill: check figure saved; note TP/FP/FN/TN counts; relate FP rate to Precision@top-20%]* |
| Error analysis (PR curve) | PR curve comparing both models on test; locked threshold marked as red dot | *[fill: do curves overlap? Which dominates in high-recall region? Is gap consistent with PR-AUC numbers?]* |
| Error analysis (calibration) | Calibration reliability curves for both models on same axes + score histogram for final model; saved to `outputs/5c_calibration.png` | *[fill: which model tracks the diagonal more closely? This is Step 2 of the decision rule. Comment on class separation in the histogram]* |
| Error analysis (geography) | Per-country PR-AUC and Recall@top-20% for both models; computes max–min PR-AUC gap per model | *[fill: which model has smaller gap (fairer)? This is Step 3 of the decision rule. Note worst geography for each; flag fairness implications]* |
| Final selection | Final model locked on **validation** PR-AUC in Cell 5.3 (Step 1 of pre-committed rule). Steps 2–3 shown as post-hoc diagnostics on test in Cell 5.5 — they cannot override | *[fill: confirm Steps 2–3 are consistent with the locked choice; if they contradict, acknowledge transparently; state final model name]* |
| Agent mistake | Demonstrated predict() vs predict_proba() for PR-AUC; showed Δ; self-checked evaluate() | *[fill: note actual Δ; confirm buggy < correct; confirm assert passed]* |
| Model card | Agent drafted model card with intended use, limitations, data constraints, evaluation caveats, and 4-metric summary table | *[fill: confirm model card accurately reflects final model and results]* |
| *[add rows as needed]* | | |
