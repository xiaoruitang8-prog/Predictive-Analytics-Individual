# Task 3 — Prepare the Data for Modelling

---

## Coding Plan

1. **Imports.** sklearn splitting, imputing, scaling, encoding; numpy for
   post-preprocessing NaN checks.

2. **Create `df_raw` and `df_model`.** Copy `df` to `df_raw`; drop
   RowNumber, CustomerId, Surname to create `df_model`.  Neither overwrites
   `df`, so EDA cells stay re-runnable.

3. **Pre-split validation (A).** Seven integrity checks on `df_model`:
   missing values, duplicates, target binary, Age/CreditScore/Balance
   ranges, overall churn rate.  Plus two pitfall prints (Balance == 0
   fraction, NumOfProducts value counts).

4. **Stratified split.** 70 / 15 / 15 via two `train_test_split` calls
   with `random_state=42`.  Print sizes and churn rate per split.

5. **Preprocessing pipeline.** Single `ColumnTransformer`: numeric
   (MedianImputer → StandardScaler), categorical (MostFrequentImputer →
   OneHotEncoder).  `fit_transform` on X_train only.

6. **Post-preprocessing validation (B).** Four checks: row counts match
   y splits, no NaNs, feature count correct, `handle_unknown="ignore"`
   works for unseen categories.

7. **No files saved.** Task 4 re-runs these deterministic cells.

---

## Cell 1 — Task 3 imports

```python
# ── Task 3: Data Preparation ────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
```

*(No report text — imports only.)*

---

## Cell 2 — Create df_raw and df_model

```python
# df is the 14-column dataframe loaded in the EDA preamble.
# df_raw preserves a read-only copy; df_model drops identifiers.
# Neither overwrites df, so all EDA cells remain re-runnable.

ID_COLS     = ["RowNumber", "CustomerId", "Surname"]
TARGET      = "Exited"
NUMERIC     = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL = ["Geography", "Gender"]

df_raw   = df.copy()
df_model = df_raw.drop(columns=ID_COLS)

print(f"df_raw:   {df_raw.shape}")
print(f"df_model: {df_model.shape}")
print(f"Columns:  {list(df_model.columns)}")
```

*(No separate report text — column drop is documented in 3.3.)*

---

## Cell 3 — Pre-split validation

```python
# ── Pre-split integrity checks (A) ─────────────────────────────
# All checks run on df_model before any splitting or transformation.

# 1. Missing values
print(f"1. Missing values:       {df_model.isnull().sum().sum()}")

# 2. Duplicate rows
print(f"2. Duplicate rows:       {df_model.duplicated().sum()}")

# 3. Target is binary
print(f"3. Target unique values: {sorted(df_model[TARGET].unique())}")

# 4. Key ranges
print(f"4. Age range:            {df_model['Age'].min()} – {df_model['Age'].max()}")
print(f"   CreditScore range:    {df_model['CreditScore'].min()} – {df_model['CreditScore'].max()}")
print(f"   Balance range:        {df_model['Balance'].min():.2f} – {df_model['Balance'].max():.2f}")

# 5. Overall churn rate
print(f"5. Overall churn rate:   {df_model[TARGET].mean():.4f}")

# ── Modelling pitfalls ──────────────────────────────────────────
# 6. Zero-balance fraction
print(f"\n6. Balance == 0 fraction: {(df_model['Balance'] == 0).mean():.4f}")

# 7. NumOfProducts distribution
print(f"7. NumOfProducts counts:")
print(df_model["NumOfProducts"].value_counts().sort_index().to_string())
```

### 3.1  Data Validation and Modelling Pitfalls

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
  StandardScaler centres this spike but cannot remove the bimodality.
  Tree-based models handle this naturally; for linear models, a binary
  `HasBalance` indicator could be added in future iterations, but is
  omitted here to keep the baseline pipeline minimal.

- **NumOfProducts rare categories.**  Products 1 and 2 dominate; products
  **[fill]** and **[fill]** have very few rows (**[fill]** and **[fill]**
  respectively).  Small groups can cause noisy tree splits.  The pipeline
  treats `NumOfProducts` as numeric (scaled), sidestepping the
  rare-category problem.

---

## Cell 4 — Stratified 70 / 15 / 15 split

```python
# ── Stratified split ────────────────────────────────────────────
# Two-step procedure:  70% train vs 30% temp → 50/50 → 15% val + 15% test
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
    print(f"  {name:5s}: n={len(sy):5d}, churn={int(sy.sum()):4d}, rate={sy.mean():.4f}")
```

### 3.2  Split Discipline

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

preprocessor = ColumnTransformer([
    ("num", numeric_pipe,    NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
], remainder="drop")

X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out().tolist()
print(f"Output features ({len(feature_names)}):")
for fn in feature_names:
    print(f"  {fn}")
print(f"\nShapes — Train: {X_train_t.shape}  Val: {X_val_t.shape}  Test: {X_test_t.shape}")
```

### 3.3  Preprocessing Pipeline

A single `sklearn.compose.ColumnTransformer` applies two sub-pipelines:

| Sub-pipeline | Columns | Steps | Rationale |
|-------------|---------|-------|-----------|
| Numeric | CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary | MedianImputer → StandardScaler | Median imputation is defensive (no nulls exist, but guards against future data).  StandardScaler is needed for LogReg and MLP; tree models are scale-invariant. |
| Categorical | Geography, Gender | MostFrequentImputer → OneHotEncoder | `handle_unknown="ignore"` produces all-zero rows for unseen categories.  No ordinal encoding used. |

`remainder="drop"` excludes any unlisted column.  Identifier columns
(RowNumber, CustomerId, Surname) were already removed when creating
`df_model`.

**Output features:** **[fill]** total — **[fill]** scaled numeric +
**[fill]** one-hot-encoded (Geography × 3, Gender × 2).

**Post-preprocessing validation (Cell 6)** confirms: row counts match the
y-splits, no NaN values in transformed arrays, feature count equals the
expected total, and `handle_unknown="ignore"` correctly produces all-zero
Geography columns for an unseen category.

---

## Cell 6 — Post-preprocessing checks

```python
# ── Post-preprocessing validation (B) ──────────────────────────

# 1. Row counts match y splits
assert X_train_t.shape[0] == len(y_train), "Train row mismatch"
assert X_val_t.shape[0]   == len(y_val),   "Val row mismatch"
assert X_test_t.shape[0]  == len(y_test),  "Test row mismatch"
print(f"1. Row counts match: Train={X_train_t.shape[0]}, Val={X_val_t.shape[0]}, Test={X_test_t.shape[0]}")

# 2. No NaNs in transformed arrays
print(f"2. NaNs — Train: {np.isnan(X_train_t).sum()}, Val: {np.isnan(X_val_t).sum()}, Test: {np.isnan(X_test_t).sum()}")

# 3. Feature count
n_ohe = len(feature_names) - len(NUMERIC)
print(f"3. Features: {X_train_t.shape[1]} ({len(NUMERIC)} numeric + {n_ohe} OHE = {len(feature_names)})")

# 4. handle_unknown test — unseen category → all-zero OHE columns
test_row = X_val.iloc[[0]].copy()
test_row["Geography"] = "Atlantis"
test_out = preprocessor.transform(test_row)
geo_cols = test_out[0, len(NUMERIC):len(NUMERIC) + 3]
print(f"4. handle_unknown: unseen 'Atlantis' → Geography OHE = {geo_cols}")
print(f"   Expected: [0. 0. 0.] (all-zero for unseen category)")
```

*(Checks are documented in Section 3.3 above.)*

---

## 3.4  Agent Plan vs. My Verification

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Initial Task 3 (v1) | Agent created `src/preprocessing.py` with 60/20/20 split, HasBalance engineered feature, and saved three output files (Log #19, Decision #19) | I reviewed and decided the setup was over-engineered: HasBalance unnecessary, 70/15/15 gives more training data, output files add clutter |
| Revised Task 3 (v2) | Agent rewrote to 70/15/15 split, removed HasBalance, removed file saving, added Balance range check, print-only validation (Log #20, Decision #20) | Confirmed — all 6 cells run without error. Split sizes, churn rates, and feature names printed as expected. No saved output files. |
| df naming (v2 → v3) | Agent initially used `df = df.drop(...)`, overwriting the EDA dataframe (Log #21). Revised to `df_model` only (Decision #21). In v3 I requested a further revision to `df_raw` + `df_model` (Log #25, Decision #26) | `df_raw = df.copy()` preserves the full 14-column dataset under an explicit name; `df_model = df_raw.drop(columns=ID_COLS)` drops identifiers. `df` (EDA preamble) is never overwritten. Supersedes Decision #21 |
| Gender encoding | Agent recommended keeping OHE without `drop="first"` — collinear columns harmless with regularisation, aids interpretability (Log #22, Decision #22) | I asked whether manual binary encoding was needed; agent's recommendation correct — no change required |
| Validation staging (v2 → v3) | v2 had all checks in one pre-split cell (Log #23, Decision #23). In v3 I specified two-stage validation (Log #25, Decision #26) | Stage A (Cell 3): 7 pre-split integrity checks + 2 pitfall prints on `df_model`. Stage B (Cell 6): 4 post-preprocessing checks on transformed arrays. Supersedes single-stage approach |
| Pitfall prints | Agent added Balance == 0 fraction and NumOfProducts counts in Cell 3 (Log #23, Decision #24) | [fill: copy exact printed fractions and counts from Cell 3 output into Section 3.1 report text] |
| log1p for Balance | Agent recommended skipping log1p: HistGBT is monotonic-invariant, LogReg/MLP get StandardScaler (Log #20, Decision #20) | Agreed — StandardScaler is sufficient. Adding a `FunctionTransformer(np.log1p)` step would add complexity with no measurable gain given the model set. |
| Post-preprocessing checks (new in v3) | Agent added Cell 6 with 4 checks: row counts match y splits, no NaNs, feature count, handle_unknown test with unseen "Atlantis" (Log #25, Decision #26) | [fill: confirm all 4 checks pass; note actual Geography OHE output for "Atlantis" row (expect [0. 0. 0.])] |
| Draft structure | Agent initially used three-part layout (Log #21, #23); later interleaved report text under each cell (Log #24, Decision #25) | I requested interleaved layout so the document reads top-to-bottom |
| Notebook cells (v3 final) | Agent provided 6 self-contained cells: (1) imports, (2) df_raw + df_model, (3) pre-split validation, (4) split, (5) pipeline, (6) post-preprocessing checks. No src imports (Log #25, Decision #26) | Confirmed — all 6 cells self-contained and run in order. [fill: note any deviations from the draft code] |
| Section numbering | Agent used letter-suffixed labels 3A, 3B, 3C, 3D throughout draft (Log #28, Decision #29) | I requested decimal numbering consistent with Section 2: 3B→3.1 (Data Validation), 3A→3.2 (Split Discipline), 3C→3.3 (Preprocessing Pipeline), 3D→3.4 (Agent Plan). All internal cross-references updated. |
| Out-of-order section numbers | Agent assigned 3.1 to Split and 3.2 to Validation, causing 3.2 to appear before 3.1 in the document (Log #29, Decision #30) | Identified and fixed: sections renumbered to match notebook execution order — Cell 3 (validation) → 3.1, Cell 4 (split) → 3.2, Cell 5 (pipeline) → 3.3. Document now reads sequentially. |
| *[add rows as project progresses]* | | |
