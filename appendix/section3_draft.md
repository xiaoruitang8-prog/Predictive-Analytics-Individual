# 3  Prepare the Data for Modelling

## 3A  Split Discipline

A **stratified 70 / 15 / 15 train-validation-test split** is applied using
`sklearn.model_selection.train_test_split` with `random_state=42`.
Stratification on `Exited` ensures each split preserves the original class
balance.

| Split | Purpose | Fraction | Rows (full dataset) |
|-------|---------|----------|---------------------|
| Train | Fit preprocessing and model parameters | 70% | **[train_n]** |
| Validation | Tune hyperparameters, compare models | 15% | **[val_n]** |
| Test | Final held-out evaluation (reported once) | 15% | **[test_n]** |

**Two-step procedure.**  First split: train (70%) vs temp (30%).  Second
split: temp into val (50% of 30% = 15%) and test (50% of 30% = 15%).  Both
calls use `random_state=42`.

**Leakage prevention.**  The `ColumnTransformer` is **fit only on the
training split**.  Validation and test data are transformed using the
statistics (means, standard deviations, category vocabularies) learned from
training data only.  This mirrors production conditions where future data
is unseen at training time.

**Reproducibility.**  No output files are saved.  Task 4 reproduces the
exact same partition by calling `stratified_split()` with the same seed.

---

## 3B  Preprocessing Pipeline

The preprocessing uses a single `sklearn.compose.ColumnTransformer` with
two sub-pipelines:

| Sub-pipeline | Columns | Steps | Notes |
|-------------|---------|-------|-------|
| Numeric | `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` | `SimpleImputer(strategy="median")` then `StandardScaler()` | Median imputation is defensive (dataset has no missing values, but future data might). StandardScaler centres and scales to unit variance — needed for LogReg and MLP; tree-based models are scale-invariant. |
| Categorical | `Geography`, `Gender` | `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown="ignore"` produces all-zero columns for unseen categories at transform time. No ordinal encoding used. |

`remainder="drop"` ensures any unlisted column (including the target) is
excluded from the feature matrix.

**Identifier columns.**  `RowNumber`, `CustomerId`, `Surname` are dropped
by `load_churn_data(drop_ids=True)` before any processing.

**Output features after transformation (fill after running):**
> **[n_output_features]** total:
> - **[n_numeric]** scaled numeric features
> - **[n_ohe]** one-hot-encoded columns from Geography and Gender
>
> Feature names: `[feature_names_list]`

---

## 3C  Data Validation Checks

The following checks run automatically via `preprocessing.validate_data()`
and print results to stdout.  Critical failures raise `ValueError`.

| # | Check | Expected Result | Actual Result |
|---|-------|----------------|---------------|
| 1 | Missing values per column and total | 0 missing (EDA confirmed) | **[fill]** |
| 2 | Duplicate rows | 0 duplicates | **[fill]** |
| 3 | Age range (min/max) | Roughly 18–92 | **[fill]** |
| 4 | CreditScore range (min/max) | Roughly 350–850 | **[fill]** |
| 5 | Balance range (min/max) | 0 to ~250k | **[fill]** |
| 6 | Target is binary | Values exactly {0, 1} | **[fill]** |
| 7 | ID columns not in features | RowNumber, CustomerId, Surname absent | **[fill]** |
| 8 | Overall class balance | ~20% churn | **[fill]** |

**Class balance per split (fill after running):**

| Split | N | Class 0 | Class 1 | Churn Rate |
|-------|---|---------|---------|------------|
| Train | **[train_n]** | **[train_0]** | **[train_1]** | **[train_rate]** |
| Val | **[val_n]** | **[val_0]** | **[val_1]** | **[val_rate]** |
| Test | **[test_n]** | **[test_0]** | **[test_1]** | **[test_rate]** |

---

## 3D  How to Reproduce

From the repo root:

```bash
pip install -r requirements.txt
python src/03_preprocess.py
```

All results are printed to stdout.  No output files are created.

---

## 3E  Agent Plan vs. My Verification

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Initial Task 3 (v1) | Agent created preprocessing with 60/20/20 split, HasBalance engineered feature, and saved three output files (split_indices.json, preprocessor.joblib, validation_report.json) (Log #19, Decision #19) | I reviewed and decided the setup was over-engineered: HasBalance is unnecessary for a minimal rubric-aligned submission, 70/15/15 gives more training data, and output files add clutter when Task 4 can call the same functions |
| Revised Task 3 (v2) | Agent rewrote to 70/15/15 split, removed HasBalance, removed all output file saving, added Balance range check, simplified validation to print-only (Log #20, Decision #20) | [fill after running `python src/03_preprocess.py` and reviewing printed output] |
| log1p for Balance | Agent recommended skipping log1p: HistGradientBoosting is monotonic-invariant, LogReg/MLP get StandardScaler, and adding FunctionTransformer for one column adds pipeline complexity for uncertain gain | [fill: agree/disagree with rationale] |
| Section 3 draft | Agent drafted 3A–3E with placeholders | [fill all `[placeholders]` after running script on full dataset] |
| *[add rows as project progresses]* | | |
