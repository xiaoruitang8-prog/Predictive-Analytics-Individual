# Task 3 — Prepare the Data for Modelling

---

## Coding Plan

1. **Imports.** One cell with the sklearn imports for splitting, imputing,
   scaling, and encoding.

2. **Create `df_model`.** Copy `df` into `df_model`, dropping the three
   identifier columns.  `df` is never overwritten so EDA cells stay
   re-runnable.

3. **Validate before splitting.**  Single validation cell on `df_model`:
   missing values, duplicates, target check, range checks, churn rate,
   plus two pitfall prints (Balance == 0 fraction, NumOfProducts counts).

4. **Stratified split.**  70 / 15 / 15 via two `train_test_split` calls
   with `random_state=42`.  Print split sizes and churn rate per split.

5. **Preprocessing pipeline.**  `ColumnTransformer` with numeric and
   categorical sub-pipelines.  `fit_transform` on training only.

6. **No outputs saved.**  Task 4 re-runs these cells with the same seed.

---

## Cell 1 — Task 3 imports

```python
# ── Task 3: Data Preparation ────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
```

*(No report text — imports only.)*

---

## Cell 2 — Create df_model (drop identifiers)

```python
# Keep df (the raw 14-column dataframe from EDA) untouched so that
# all EDA cells above remain re-runnable without side-effects.
# df_model is the modelling-ready dataframe with identifiers removed.

ID_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET  = "Exited"

df_model = df.drop(columns=[c for c in ID_COLS if c in df.columns])

NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL = ["Geography", "Gender"]

print(f"df_model shape: {df_model.shape}")
print(f"Columns: {list(df_model.columns)}")
```

*(No separate report text — column drop is mentioned in 3C below.)*

---

## Cell 3 — Data validation checks

```python
# ── Pre-split validation ────────────────────────────────────────
# One cell, all checks.  Confirms the dataset is clean and documents
# two distributional quirks relevant to modelling pitfalls.

# 1. Missing values
missing_total = df_model.isnull().sum().sum()
print(f"1. Total missing values: {missing_total}")

# 2. Duplicate rows
n_dupes = df_model.duplicated().sum()
print(f"2. Duplicate rows:       {n_dupes}")

# 3. Target is binary
print(f"3. Target unique values: {sorted(df_model[TARGET].unique())}")

# 4. Range checks
print(f"4. Age range:            {df_model['Age'].min()} – {df_model['Age'].max()}")
print(f"   CreditScore range:    {df_model['CreditScore'].min()} – {df_model['CreditScore'].max()}")
print(f"   Balance range:        {df_model['Balance'].min():.2f} – {df_model['Balance'].max():.2f}")

# 5. Overall class balance
churn_rate = df_model[TARGET].mean()
print(f"5. Overall churn rate:   {churn_rate:.4f}")

# ── Modelling pitfalls ──────────────────────────────────────────
# 6. Zero-balance fraction — Balance has a large spike at exactly 0,
#    creating a bimodal distribution that may affect distance-based
#    models and tree split points.
zero_bal_frac = (df_model["Balance"] == 0).mean()
print(f"\n6. Fraction Balance == 0:  {zero_bal_frac:.4f}")

# 7. NumOfProducts distribution — categories 3 and 4 have very few
#    rows, which can cause unstable estimates and noisy splits.
print(f"7. NumOfProducts value counts:")
print(df_model["NumOfProducts"].value_counts().sort_index().to_string())
```

### Report text — 3B  Data Validation and Modelling Pitfalls

Before splitting, the following checks confirm the dataset is modelling-ready.

| # | Check | Result |
|---|-------|--------|
| 1 | Missing values | **[fill]** (expect 0) |
| 2 | Duplicate rows | **[fill]** (expect 0) |
| 3 | Target values | **[fill]** (expect {0, 1}) |
| 4 | Age range | **[fill]** (expect 18–92) |
| 5 | CreditScore range | **[fill]** (expect 350–850) |
| 6 | Balance range | **[fill]** (expect 0–~250 k) |
| 7 | Overall churn rate | **[fill]** (expect ~0.2037) |

**Modelling pitfalls identified:**

- **Zero-balance spike.**  **[fill]%** of rows have `Balance == 0`,
  creating a bimodal distribution (visible in the EDA histogram, Plot 1).
  StandardScaler will centre this spike but cannot remove the bimodality.
  Tree-based models handle this naturally; for linear models, a binary
  `HasBalance` indicator could be added in future iterations, but it is
  omitted here to keep the baseline pipeline minimal.

- **NumOfProducts rare categories.**  Products 1 and 2 dominate, while
  products **[fill]** and **[fill]** have very few rows (**[fill]** and
  **[fill]** respectively).  These small groups can cause unstable
  cross-validation folds and noisy tree splits.  The current pipeline
  treats `NumOfProducts` as numeric (scaled), which sidesteps the
  rare-category problem.  If treated categorically in future work,
  grouping 3+ into a single bin would be advisable.

---

## Cell 4 — Stratified 70 / 15 / 15 split

```python
# ── Stratified split ────────────────────────────────────────────
# Two-step procedure with the same seed for reproducibility:
#   Step 1: 70% train  vs  30% temp
#   Step 2: 50/50 temp → 15% val + 15% test
SEED = 42

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
)

# Confirm sizes and stratification
for name, sy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    n = len(sy)
    n1 = int(sy.sum())
    print(f"  {name:5s}: n={n:5d}, churn={n1:4d}, rate={n1/n:.4f}")
```

### Report text — 3A  Split Discipline

A **stratified 70 / 15 / 15 train-validation-test split** is applied using
`sklearn.model_selection.train_test_split` with `random_state=42`.
Stratification on `Exited` preserves the original class balance in every
split.

| Split | Purpose | Rows |
|-------|---------|------|
| Train | Fit preprocessing and model parameters | **[train_n]** |
| Validation | Tune hyperparameters, compare models | **[val_n]** |
| Test | Final held-out evaluation (reported once) | **[test_n]** |

The split uses a **two-step procedure**: first 70% train vs 30% temp, then
temp is split 50/50 into validation and test.  Both calls use
`random_state=42`.

**Leakage prevention.**  The `ColumnTransformer` is **fit only on the
training split**.  Validation and test data are transformed using the
statistics (means, standard deviations, category vocabularies) learned from
training data only.

**Class balance per split:**

| Split | N | Churned | Churn Rate |
|-------|---|---------|------------|
| Train | **[fill]** | **[fill]** | **[fill]** |
| Val   | **[fill]** | **[fill]** | **[fill]** |
| Test  | **[fill]** | **[fill]** | **[fill]** |

All three splits show a churn rate within 1 percentage point of the overall
rate, confirming that stratification worked correctly.

---

## Cell 5 — Preprocessing pipeline

```python
# ── Preprocessing pipeline ──────────────────────────────────────
# Numeric:     median imputer (defensive) → StandardScaler
# Categorical: most-frequent imputer → OneHotEncoder
# fit_transform on TRAINING data only; transform val and test.

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,      NUMERIC),
        ("cat", categorical_pipe,   CATEGORICAL),
    ],
    remainder="drop",
)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out().tolist()
print(f"Output features ({len(feature_names)}):")
for fn in feature_names:
    print(f"  {fn}")
print(f"\nShapes — Train: {X_train_t.shape}  Val: {X_val_t.shape}  Test: {X_test_t.shape}")
```

### Report text — 3C  Preprocessing Pipeline

A single `sklearn.compose.ColumnTransformer` applies two sub-pipelines:

| Sub-pipeline | Columns | Steps | Rationale |
|-------------|---------|-------|-----------|
| Numeric | CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | MedianImputer → StandardScaler | Median imputation is defensive (no nulls exist, but future data might).  StandardScaler is needed for LogReg and MLP; tree models are scale-invariant. |
| Categorical | Geography, Gender | MostFrequentImputer → OneHotEncoder | `handle_unknown="ignore"` produces all-zero rows for unseen categories.  No ordinal encoding used. |

`remainder="drop"` excludes any unlisted column.  Identifier columns were
already removed when creating `df_model`.

**Output features:** **[fill]** total — **[fill]** scaled numeric +
**[fill]** one-hot-encoded (Geography × 3, Gender × 2).

---

## 3D  Agent Plan vs. My Verification

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Initial Task 3 (v1) | Agent created `src/preprocessing.py` with 60/20/20 split, HasBalance engineered feature, and saved three output files (split_indices.json, preprocessor.joblib, validation_report.json) (Log #19, Decision #19) | I reviewed and decided the setup was over-engineered: HasBalance is unnecessary for a minimal rubric-aligned submission, 70/15/15 gives more training data, and output files add clutter when Task 4 can call the same functions |
| Revised Task 3 (v2) | Agent rewrote to 70/15/15 split, removed HasBalance, removed all output file saving, added Balance range check, simplified validation to print-only (Log #20, Decision #20) | [fill after running notebook cells] |
| df naming | Agent initially used `df = df.drop(...)`, silently overwriting the EDA dataframe | I requested `df_model` to preserve `df` (the raw EDA dataframe) so all Task 2 cells remain re-runnable without side-effects |
| Validation structure | Agent had validation checks in four places (Cell 4, Cell 5, report table 3B, checklist 3D).  Redundant checklist section 3D repeated 3B almost verbatim | I merged all validation into one cell and one report table, removed the duplicate checklist section |
| Pitfall prints | Agent added Balance == 0 fraction and NumOfProducts value counts as two extra prints in the validation cell, referenced in report text as modelling pitfalls | [fill: confirm output values match report text] |
| log1p for Balance | Agent recommended skipping log1p: HistGradientBoosting is monotonic-invariant, LogReg/MLP get StandardScaler, and adding FunctionTransformer for one column adds pipeline complexity for uncertain gain | [fill: agree/disagree with rationale] |
| Notebook cells (final) | Agent provided 5 self-contained cells (imports, df_model, validation, split, pipeline) with no src imports, consistent with notebook style from Task 2 | [fill: confirm cells run, outputs match expected values] |
| *[add rows as project progresses]* | | |
