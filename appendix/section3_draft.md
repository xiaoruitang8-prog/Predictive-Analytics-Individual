# 3  Prepare the Data for Modelling

## 3A  Split Discipline

A **stratified 60 / 20 / 20 train-validation-test split** is applied using
`sklearn.model_selection.train_test_split` with `random_state=42`.
Stratification ensures each split preserves the original class balance of
`Exited`, preventing the minority class from being under- or over-represented
in any partition.

| Split | Purpose | Fraction | Rows (full dataset) |
|-------|---------|----------|---------------------|
| Train | Fit preprocessing and model parameters | 60% | **[train_n]** |
| Validation | Tune hyperparameters, compare models | 20% | **[val_n]** |
| Test | Final held-out evaluation (reported once) | 20% | **[test_n]** |

**Leakage prevention.**  The `ColumnTransformer` is **fit only on the
training split**.  Validation and test data are transformed using the
statistics (means, standard deviations, category vocabularies) learned from
training data only.  This mirrors production conditions where future data
is unseen at training time.

**Reproducibility.**  Split indices are saved to `outputs/split_indices.json`
so that Task 4 can reload the exact same partition.  The fitted preprocessor
is saved to `outputs/preprocessor.joblib`.

---

## 3B  Feature Engineering

### HasBalance indicator

A binary feature `HasBalance` (1 if `Balance > 0`, else 0) is added before
splitting.

**Rationale.**  EDA (Plot 1, Plot 5, Section 2D Action #3) revealed that
approximately **[zero_bal_pct]%** of customers have exactly zero balance,
creating a bimodal distribution.  Without an explicit indicator, tree-based
models must find the split point at zero on their own (possible but
inefficient), while distance-based models (Logistic Regression, MLP) are
distorted by the zero-mass spike.

**Verification.**
```python
# Run after loading data and adding engineered features:
print((df["Balance"] == 0).mean())      # should match [zero_bal_pct]%
print(df["HasBalance"].value_counts())   # counts for 0 and 1
# Cross-check: HasBalance==0 rows should have Balance==0
assert (df.loc[df["HasBalance"] == 0, "Balance"] == 0).all()
```

**Leakage risk.**  This is a deterministic transform — no statistics are
learned from the data.  It is safe to apply before splitting because the
same formula is applied identically at inference time.

---

## 3C  Preprocessing Pipeline

The preprocessing uses a single `sklearn.compose.ColumnTransformer` with
two sub-pipelines:

| Sub-pipeline | Columns | Steps | Notes |
|-------------|---------|-------|-------|
| Numeric | `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `HasBalance` | `SimpleImputer(strategy="median")` then `StandardScaler()` | Median imputation is robust to outliers (Age, Balance have long tails). StandardScaler centres and scales to unit variance. |
| Categorical | `Geography`, `Gender` | `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown="ignore"` produces all-zero columns for unseen categories at transform time, preventing errors. |

`remainder="drop"` ensures any unlisted column (including the target) is
excluded from the feature matrix.

**Output features after transformation (fill after running):**
> **[n_output_features]** total:
> - **[n_numeric]** scaled numeric features
> - **[n_ohe]** one-hot-encoded columns from Geography and Gender
>
> Feature names: `[feature_names_list]`

---

## 3D  Data Validation Checks

The following checks run automatically before modelling via
`preprocessing.validate_data()`.  Critical failures raise `ValueError`;
non-critical issues are logged in the report.

| # | Check | Expected Result | Status |
|---|-------|----------------|--------|
| 1 | Expected columns exist | All of `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Geography`, `Gender`, `Exited` present | **[pass/fail]** |
| 2 | Target is binary | `Exited` values are exactly `{0, 1}` | **[pass/fail]** |
| 3 | ID columns not in features | `RowNumber`, `CustomerId`, `Surname` absent (dropped by loader) | **[pass/fail]** |
| 4 | Missing values | Count of nulls per column; total missing | **[n_missing_total]** |
| 5 | Duplicate rows | Count of exact duplicate rows | **[n_dupes]** |
| 6 | Age range | `min >= 0`; expected range roughly 18–92 | **[age_min]** – **[age_max]** |
| 7 | CreditScore range | Expected range roughly 350–850 | **[cs_min]** – **[cs_max]** |
| 8 | Class balance per split | Churn rate consistent across train/val/test (within ~1 pp of overall rate) | See table below |

**Class balance per split (fill after running):**

| Split | N | Class 0 | Class 1 | Churn Rate |
|-------|---|---------|---------|------------|
| Train | **[train_n]** | **[train_0]** | **[train_1]** | **[train_rate]** |
| Val | **[val_n]** | **[val_0]** | **[val_1]** | **[val_rate]** |
| Test | **[test_n]** | **[test_0]** | **[test_1]** | **[test_rate]** |

---

## 3E  How to Reproduce

From the repo root:

```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Run Task 3 preprocessing
python src/03_preprocess.py
```

Expected outputs in `outputs/`:
- `split_indices.json` — row indices for train, val, test
- `preprocessor.joblib` — fitted ColumnTransformer (fit on train only)
- `validation_report.json` — all checks + class balance + feature names

---

## 3F  Agent Plan vs. My Verification

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Preprocessing module | Agent created `src/preprocessing.py` with feature constants, `add_engineered_features()`, `build_preprocessor()`, `stratified_split()`, `validate_data()`, `split_class_balance()` (Log #19, Decision #19) | [fill after reviewing code and running script] |
| Entry point script | Agent created `src/03_preprocess.py` that loads data, validates, engineers features, splits, fits preprocessor on train only, saves all outputs (Log #19) | [fill after running `python src/03_preprocess.py` and checking outputs] |
| HasBalance feature | Agent added `HasBalance = (Balance > 0).astype(int)` as a deterministic engineered feature, applied before splitting (Log #19, Decision #19) | [fill: verify with `(df["Balance"] == 0).mean()` and cross-check assertion] |
| Validation checks | Agent implemented 7 checks covering columns, target, IDs, missing values, duplicates, Age range, CreditScore range (Log #19) | [fill: confirm all checks pass on full dataset; check validation_report.json] |
| Section 3 draft | Agent drafted 3A–3F with placeholders | [fill all `[placeholders]` after running script on full dataset] |
| *[add rows as project progresses]* | | |
