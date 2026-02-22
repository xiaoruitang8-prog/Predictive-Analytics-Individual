# 3  Prepare the Data for Modelling

> **How to use this document.**  Each section below has three parts:
> - **Report text** — the narrative for your coursework submission
> - **Notebook cell** — the exact code to paste into your Jupyter notebook
> - **Verify** — what to check after running the cell
>
> Run the cells in order.  Fill in `[placeholders]` with actual values from
> the notebook output.

---

## 3A  Split Discipline

### Report text

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

### Notebook cells

**Cell 1 — Task 3 imports** (add after last EDA cell)

```python
# Task 3: Data Preparation
# These imports are used for splitting, preprocessing, and building the pipeline.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
```

**Cell 2 — Define columns and drop identifiers**

```python
# Target column for binary classification.
TARGET = "Exited"

# Identifier columns carry no predictive signal and must not enter the model.
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

# Drop identifiers from the working dataframe.
# This is deferred from EDA (Task 2) so that the raw dataframe was available
# for exploration with all 14 columns intact.
df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

# Feature lists after dropping IDs.
NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL = ["Geography", "Gender"]

print(f"Columns after dropping IDs: {list(df.columns)}")
print(f"Shape: {df.shape}")
```

**Cell 3 — Stratified 70/15/15 split**

```python
# --- Stratified Train / Validation / Test Split ---
# Two-step procedure:
#   Step 1: 70% train vs 30% temp
#   Step 2: split temp 50/50 into 15% val + 15% test
# Stratify on Exited so each split preserves the class balance.
SEED = 42

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Step 1: train (70%) vs temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED,
)

# Step 2: val (50% of 30% = 15%) vs test (50% of 30% = 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED,
)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
```

### Verify (after running cells 1–3)

- [ ] Columns after dropping IDs: 11 (8 numeric + 2 categorical + 1 target)
- [ ] Split sizes: **[train_n]** / **[val_n]** / **[test_n]** (expect 7000 / 1500 / 1500)
- [ ] No ID columns in `df.columns`

---

## 3B  Data Validation Checks

### Report text

The following checks run in the notebook before modelling to confirm the
dataset is clean.

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

### Notebook cells

**Cell 4 — Data validation checks**

```python
# --- Data Validation Checks ---
# These checks confirm the dataset is clean before modelling.

# 1. Missing values per column and total
missing = df.isnull().sum()
print("Missing values per column:")
print(missing.to_string())
print(f"Total missing: {missing.sum()}\n")

# 2. Duplicate rows
n_dupes = df.duplicated().sum()
print(f"Duplicate rows: {n_dupes}\n")

# 3. Range checks for key numeric features
print("Range checks:")
print(f"  Age:         {df['Age'].min()} – {df['Age'].max()}")
print(f"  CreditScore: {df['CreditScore'].min()} – {df['CreditScore'].max()}")
print(f"  Balance:     {df['Balance'].min():.2f} – {df['Balance'].max():.2f}\n")

# 4. Target values — must be exactly {0, 1}
print(f"Target unique values: {sorted(df[TARGET].unique())}")

# 5. Overall class balance
vc = df[TARGET].value_counts()
print(f"Class balance: {vc.to_dict()}")
print(f"Churn rate: {vc[1] / len(df):.4f}")
```

**Cell 5 — Class balance per split**

```python
# Confirm stratification preserved the churn rate across all splits.
for name, split_y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    n = len(split_y)
    n1 = int(split_y.sum())
    n0 = n - n1
    print(f"  {name:5s}: n={n:5d}, class_0={n0:4d}, class_1={n1:4d}, "
          f"churn_rate={n1/n:.4f}")
```

### Verify (after running cells 4–5)

- [ ] Total missing = 0
- [ ] Duplicate rows = 0
- [ ] Age range: 18–92
- [ ] CreditScore range: 350–850
- [ ] Balance range: 0.00–250898.09
- [ ] Target values: [0, 1]
- [ ] Churn rate ~0.2037 in all three splits (within 1 pp of each other)

---

## 3C  Preprocessing Pipeline

### Report text

The preprocessing uses a single `sklearn.compose.ColumnTransformer` with
two sub-pipelines:

| Sub-pipeline | Columns | Steps | Notes |
|-------------|---------|-------|-------|
| Numeric | `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` | `SimpleImputer(strategy="median")` then `StandardScaler()` | Median imputation is defensive (dataset has no missing values, but future data might). StandardScaler centres and scales to unit variance — needed for LogReg and MLP; tree-based models are scale-invariant. |
| Categorical | `Geography`, `Gender` | `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")` | `handle_unknown="ignore"` produces all-zero columns for unseen categories at transform time. No ordinal encoding used. |

`remainder="drop"` ensures any unlisted column (including the target) is
excluded from the feature matrix.

**Identifier columns.**  `RowNumber`, `CustomerId`, `Surname` were dropped
in Cell 2 before splitting.

**Output features after transformation (fill after running):**
> **[n_output_features]** total:
> - **[n_numeric]** scaled numeric features
> - **[n_ohe]** one-hot-encoded columns from Geography and Gender
>
> Feature names: `[feature_names_list]`

### Notebook cell

**Cell 6 — Build and fit preprocessing pipeline**

```python
# --- Preprocessing Pipeline ---
# Single ColumnTransformer with two sub-pipelines:
#   Numeric:     median imputer (defensive — no nulls in this dataset) + StandardScaler
#   Categorical: most-frequent imputer + OneHotEncoder
#
# fit_transform on training data ONLY to prevent data leakage.
# Validation and test sets are transformed using training statistics.

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ],
    remainder="drop",  # drops any unlisted column (safety net)
)

# Fit on train, transform all three splits.
X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

# Show output feature names and shapes.
feature_names = preprocessor.get_feature_names_out().tolist()
print(f"Output features ({len(feature_names)}):")
for fn in feature_names:
    print(f"  {fn}")
print(f"\nTrain: {X_train_t.shape}  Val: {X_val_t.shape}  Test: {X_test_t.shape}")
```

### Verify (after running cell 6)

- [ ] Output features: 13 total (8 numeric + 3 Geography + 2 Gender)
- [ ] Train shape: (7000, 13)
- [ ] Val shape: (1500, 13)
- [ ] Test shape: (1500, 13)
- [ ] `fit_transform` called on `X_train` only; `transform` on val and test

---

## 3D  Complete Verification Checklist

After running all six cells, confirm every item below:

| # | Check | Expected | Pass? |
|---|-------|----------|-------|
| 1 | No missing values | total = 0 | |
| 2 | No duplicates | count = 0 | |
| 3 | Age range | 18–92 | |
| 4 | CreditScore range | 350–850 | |
| 5 | Balance range | 0.00–250898.09 | |
| 6 | Target values | {0, 1} only | |
| 7 | ID columns absent | RowNumber, CustomerId, Surname not in df | |
| 8 | Split sizes | 7000 / 1500 / 1500 | |
| 9 | Churn rate per split | all ~0.2037 | |
| 10 | Output features | 13 (8 num + 5 OHE) | |
| 11 | Preprocessor fit on train only | fit_transform(X_train), transform(X_val), transform(X_test) | |
| 12 | No output files created | no JSON, no joblib | |

---

## 3E  Agent Plan vs. My Verification

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Initial Task 3 (v1) | Agent created `src/preprocessing.py` with 60/20/20 split, HasBalance engineered feature, and saved three output files (split_indices.json, preprocessor.joblib, validation_report.json) (Log #19, Decision #19) | I reviewed and decided the setup was over-engineered: HasBalance is unnecessary for a minimal rubric-aligned submission, 70/15/15 gives more training data, and output files add clutter when Task 4 can call the same functions |
| Revised Task 3 (v2) | Agent rewrote `src/` to 70/15/15 split, removed HasBalance, removed all output file saving, added Balance range check, simplified validation to print-only (Log #20, Decision #20) | [fill after running notebook cells] |
| Notebook cells | Agent provided 6 self-contained notebook cells matching the `src/` logic but written inline (no `src` imports), consistent with the notebook style established in Task 2 (Decision #12) | [fill: confirm cells run, outputs match expected values, paste into notebook] |
| log1p for Balance | Agent recommended skipping log1p: HistGradientBoosting is monotonic-invariant, LogReg/MLP get StandardScaler, and adding FunctionTransformer for one column adds pipeline complexity for uncertain gain | [fill: agree/disagree with rationale] |
| Section 3 draft | Agent drafted 3A–3E as integrated plan (report text + notebook code + verification per section) | [fill all `[placeholders]` after running notebook on full dataset] |
| *[add rows as project progresses]* | | |
