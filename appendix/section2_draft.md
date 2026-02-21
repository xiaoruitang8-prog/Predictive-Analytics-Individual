# 2  Explore the Data to Gain Insights

## 2A  EDA Approach

Before any modelling or preprocessing, a visual EDA was conducted to
understand the dataset's structure, distributions, class balance, and
potential data-quality issues.  All analysis was performed on the **raw
14-column dataframe** (no ID-column drop — that is deferred to Task 3,
data preparation).  Each plot cell explicitly selects the columns it needs
so that identifier columns (`RowNumber`, `CustomerId`, `Surname`) do not
pollute numeric or categorical analyses.

**Scope.**  Ten visual checks organised into eight plot groups:

| # | Plot / Check | Question It Answers | Output File |
|---|-------------|---------------------|-------------|
| 1 | Target class balance | How imbalanced is `Exited`? | `outputs/eda_01_target_balance.png` |
| 2 | Missingness check | Are there any null or missing values? | `outputs/eda_02_missingness.png` |
| 3 | Numeric distributions | What do the numeric features' distributions look like? Any skew, outliers, or unexpected ranges? | `outputs/eda_03_numeric_distributions.png` |
| 4 | Churn rate by Geography | Does churn rate differ across countries? | `outputs/eda_04_churn_by_geography.png` |
| 5 | Churn rate by Gender | Does churn rate differ by gender? | `outputs/eda_05_churn_by_gender.png` |
| 6 | Zero-balance segment | What fraction of customers have zero balance, and does that segment churn differently? | `outputs/eda_06_zero_balance.png` |
| 7 | Correlation heatmap | Which numeric features are correlated with each other or with the target? | `outputs/eda_07_correlation_heatmap.png` |
| 8 | Boxplots by churn | How do numeric feature distributions differ between churners and non-churners? | `outputs/eda_08_boxplots_by_churn.png` |
| 9 | Leakage check | Does any single feature near-perfectly predict the target (suspiciously high correlation or AUC)? | `outputs/eda_09_leakage_check.png` |
| 10 | NumOfProducts vs churn | Does the number of products held relate to churn in a non-linear way? | `outputs/eda_10_products_churn.png` |

**Tooling.**  All plots use `matplotlib` and `seaborn` in a Jupyter
notebook, saved to `outputs/` at 150 dpi for inclusion as coursework
evidence.  No `src/` imports are used — each cell is self-contained
(see Decision Register #12).

---

## 2B  Visual EDA Results

Each subsection below states (i) what to look at in the plot,
(ii) a simple dataframe check to validate the visual, and
(iii) an interpretation template with **placeholders** to be filled
after running the notebook on the full dataset.

### 2B.1  Plot 1 — Target Class Balance

*What to look at.*  Bar heights and percentage labels for class 0
(retained) vs class 1 (churned).  The split reveals the degree of class
imbalance, which directly affects metric choice and sampling strategy.

*Dataframe validation.*
```python
df["Exited"].value_counts(normalize=True)
```
Confirm the percentages on the bars match the output of the command above.

*Interpretation (fill after running).*
> The dataset contains **[N_rows]** customers.  **[retained_pct]%** are
> retained (`Exited = 0`) and **[churn_pct]%** churned (`Exited = 1`).
> This confirms a moderately imbalanced dataset, justifying the use of
> PR-AUC as the primary metric (Section 1C).

---

### 2B.2  Plot 2 — Missingness Check

*What to look at.*  A bar chart or heatmap showing null counts per column.
Any column with non-zero missing values needs a handling strategy before
modelling.

*Dataframe validation.*
```python
df.isnull().sum()
```
Confirm counts match the visual and that the total matches
`df.isnull().sum().sum()`.

*Interpretation (fill after running).*
> **[n_missing_total]** missing values were found across all columns.
> [If zero: "The dataset is complete — no imputation is required."
> If non-zero: "Column(s) **[col_names]** have **[n_missing]** missing
> values (**[pct_missing]%**).  Handling strategy: [drop / impute]."]

---

### 2B.3  Plot 3 — Numeric Distributions

*What to look at.*  Histograms (or KDE plots) for each numeric feature:
`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`,
`NumOfProducts`.  Look for skewness, multi-modality, unexpected ranges,
and potential outliers.

*Dataframe validation.*
```python
df[["CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary"]].describe()
```
Confirm min/max/mean align with what the histograms show.

*Interpretation (fill after running).*
> - `Age`: ranges from **[age_min]** to **[age_max]**, roughly
>   **[age_shape]**-shaped.
> - `Balance`: **[zero_bal_pct]%** of values are exactly zero, creating a
>   spike at zero; the remaining values are approximately
>   **[bal_shape]**-distributed.
> - `CreditScore`: ranges **[cs_min]**–**[cs_max]**, approximately normal.
> - `Tenure`: approximately **[tenure_shape]**-distributed across
>   0–**[tenure_max]** years.
> - `EstimatedSalary`: appears roughly **[salary_shape]**-distributed
>   between **[salary_min]** and **[salary_max]**.
> - `NumOfProducts`: discrete, most customers hold **[modal_products]**
>   product(s).

---

### 2B.4  Plot 4 — Churn Rate by Geography

*What to look at.*  Grouped or stacked bar chart showing churn rate per
country (`France`, `Germany`, `Spain`).  Look for countries with
disproportionately high or low churn.

*Dataframe validation.*
```python
df.groupby("Geography")["Exited"].mean()
```
Confirm the rates shown on the plot match the groupby output.

*Interpretation (fill after running).*
> Churn rates by geography: France **[fr_rate]%**, Germany
> **[de_rate]%**, Spain **[es_rate]%**.
> **[highest_country]** has the highest churn rate at **[highest_rate]%**,
> roughly **[comparison]** the rate in **[lowest_country]**.
> This suggests geography is a meaningful feature for the model.

---

### 2B.5  Plot 5 — Churn Rate by Gender

*What to look at.*  Bar chart of churn rate for `Male` vs `Female`.

*Dataframe validation.*
```python
df.groupby("Gender")["Exited"].mean()
```

*Interpretation (fill after running).*
> Female churn rate: **[f_rate]%**; male churn rate: **[m_rate]%**.
> **[higher_gender]** customers churn at a **[diff_description]** higher
> rate.  This is a potential fairness consideration (Section 1C).

---

### 2B.6  Plot 6 — Zero-Balance Segment

*What to look at.*  Compare churn rate for customers with `Balance == 0`
vs `Balance > 0`.  A large zero-balance segment with a different churn
profile may warrant a binary indicator feature in preprocessing.

*Dataframe validation.*
```python
df.groupby(df["Balance"] == 0)["Exited"].mean()
```

*Interpretation (fill after running).*
> **[zero_bal_n]** customers (**[zero_bal_pct]%**) have zero balance.
> Their churn rate is **[zero_churn]%** vs **[nonzero_churn]%** for
> non-zero balances.
> [If notable difference: "A binary `HasBalance` feature may help the
> model distinguish these two groups."]

---

### 2B.7  Plot 7 — Correlation Heatmap

*What to look at.*  Pearson correlation matrix of numeric features
(explicitly excluding `RowNumber`, `CustomerId`).  Look for (a) strongly
correlated feature pairs that may cause multicollinearity and (b)
features with the highest absolute correlation to `Exited`.

*Dataframe validation.*
```python
df[["CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary", "Exited"]].corr()
```

*Interpretation (fill after running).*
> - Strongest feature–target correlation: `[top_corr_feature]`
>   (r = **[top_corr_val]**).
> - Strongest inter-feature correlation: `[top_pair_a]` – `[top_pair_b]`
>   (r = **[top_pair_val]**).
> - [If all inter-feature correlations are weak: "No strong
>   multicollinearity detected — all pairwise |r| < [threshold]."]

---

### 2B.8  Plot 8 — Boxplots by Churn

*What to look at.*  Side-by-side boxplots of each numeric feature split
by `Exited` (0 vs 1).  Look for distributional shifts: features where
the median, IQR, or outlier pattern differs between churners and
non-churners are likely informative predictors.

*Dataframe validation.*
```python
df.groupby("Exited")[["CreditScore", "Age", "Tenure", "Balance",
                       "NumOfProducts", "EstimatedSalary"]].median()
```

*Interpretation (fill after running).*
> Features with visible distributional shift between classes:
> - `Age`: churners have a **[direction]** median
>   (**[churn_median]** vs **[retained_median]**).
> - `Balance`: **[balance_observation]**.
> - `NumOfProducts`: **[products_observation]**.
> - Features with little visible difference:
>   **[flat_features]** — these may contribute less to the model.

---

### 2B.9  Plot 9 — Leakage Check

*What to look at.*  A bar chart of absolute correlation (or univariate
AUC) of every feature with `Exited`.  Any feature with suspiciously high
predictive power (e.g., |r| > 0.8 or AUC > 0.95) may indicate data
leakage — i.e., the feature encodes information that would not be
available at prediction time.

*Dataframe validation.*
```python
df[["CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
    "HasCrCard", "IsActiveMember", "Exited"]].corr()["Exited"].abs().sort_values(ascending=False)
```

*Interpretation (fill after running).*
> - Highest absolute correlation with `Exited`: `[leak_feature]`
>   (|r| = **[leak_val]**).
> - [If all |r| < 0.5: "No feature shows suspiciously high correlation
>   with the target.  Leakage is unlikely."]
> - [If any |r| > 0.8: "**Warning**: `[feature]` has |r| = [val] with
>   `Exited`.  Investigate whether this feature would be available at
>   prediction time."]

---

### 2B.10  Plot 10 — NumOfProducts vs Churn

*What to look at.*  Bar chart of churn rate by number of products held
(1, 2, 3, 4).  Look for non-linear patterns — e.g., customers with
very few or very many products may churn at higher rates.

*Dataframe validation.*
```python
df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"])
```

*Interpretation (fill after running).*
> - Customers with **[n_products_low]** product(s): churn rate
>   **[rate_low]%** (n = **[count_low]**).
> - Customers with **[n_products_high]** products: churn rate
>   **[rate_high]%** (n = **[count_high]**).
> - [If non-linear: "The relationship is non-linear — churn is
>   **[pattern_description]**.  This favours tree-based models over
>   plain logistic regression."]

---

## 2C  Data-Quality Issues and Modelling Pitfalls

The following are framed as **checks to perform**, not claims.  Each item
must be confirmed or ruled out by running the notebook on the full
dataset.

| # | Check | How to Verify | Status |
|---|-------|--------------|--------|
| 1 | **Class imbalance**: is the positive class (churners) a minority? | `df["Exited"].value_counts(normalize=True)` | [confirmed / not confirmed] |
| 2 | **Missing values**: are there any nulls? | `df.isnull().sum().sum()` | [confirmed / not confirmed] |
| 3 | **Zero-balance spike**: does `Balance` have a large mass at zero? | `(df["Balance"] == 0).mean()` | [confirmed / not confirmed] |
| 4 | **ID columns present**: do `RowNumber`, `CustomerId`, `Surname` remain in the raw dataframe? | `df.columns.tolist()` | [confirmed — must be dropped in Task 3] |
| 5 | **`Surname` cardinality**: high-cardinality string column that would need encoding or removal | `df["Surname"].nunique()` | [confirmed / not confirmed] |
| 6 | **`NumOfProducts` rare categories**: are there products = 3 or 4 with very few rows? | `df["NumOfProducts"].value_counts()` | [confirmed / not confirmed] |
| 7 | **No leakage features**: does any feature have |r| > 0.8 with `Exited`? | correlation check (Plot 9) | [confirmed / not confirmed] |
| 8 | **Geography imbalance**: are the three countries represented roughly equally? | `df["Geography"].value_counts()` | [confirmed / not confirmed] |
| 9 | **Age outliers**: are there extreme ages (e.g., < 18 or > 90)? | `df["Age"].describe()` | [confirmed / not confirmed] |
| 10 | **CreditScore range**: does it fall within typical bounds (300–850)? | `df["CreditScore"].describe()` | [confirmed / not confirmed] |
| 11 | **Tenure range**: is 0 a valid value or does it indicate missing data? | `df["Tenure"].value_counts().sort_index()` | [confirmed / not confirmed] |
| 12 | **EstimatedSalary distribution**: is it approximately uniform (synthetic data artefact)? | histogram shape (Plot 3) | [confirmed / not confirmed] |

---

## 2D  Actions Taken or Planned

These are high-level actions arising from EDA findings.  Implementation
details belong in Task 3 (data preparation) and Task 4 (modelling).

| # | Issue Identified | Action | When |
|---|-----------------|--------|------|
| 1 | Class imbalance (~[churn_pct]% positive) | Use PR-AUC as primary metric; consider class weighting or threshold tuning in modelling | Task 4 |
| 2 | ID columns (`RowNumber`, `CustomerId`, `Surname`) carry no predictive signal | Drop before modelling | Task 3 |
| 3 | Zero-balance spike in `Balance` | Consider adding a binary `HasBalance` indicator feature | Task 3 |
| 4 | `NumOfProducts` rare categories (3, 4) with very few rows | Monitor for instability in cross-validation; consider grouping 3+ into one bin | Task 3 |
| 5 | Geography has different churn rates | Ensure one-hot or ordinal encoding preserves this signal | Task 3 |
| 6 | `Gender` churn-rate gap | Include `Gender` as a feature; monitor fairness metrics post-modelling | Task 4 |
| 7 | `Age` distributional shift between classes | Likely an important predictor — ensure it is scaled for models that need it (LogReg, MLP) | Task 3 |
| 8 | No missing values (if confirmed) | No imputation step required in pipeline | Task 3 |
| 9 | `EstimatedSalary` possibly uniform | Low predictive power expected — keep in model but note if feature importance is near zero | Task 4 |
| 10 | No leakage detected (if confirmed) | No features to remove for leakage reasons | — |

---

## 2E  Agent Plan vs. My Verification

The table below documents which parts of this EDA were drafted by the
AI coding agent and what I personally verified or corrected by running
the notebook and cross-checking against the raw data.

| Step | What the Agent Did | What I Verified / Corrected | Evidence |
|------|--------------------|-----------------------------|----------|
| EDA plan and plot list | Agent proposed 10-plot EDA scope with questions, output filenames, and validation checks (Log #9, Decision #10) | I reviewed the plot list, confirmed it covers the key questions, and agreed to the scope. I rejected agent's suggestion to drop ID columns in the preamble — deferred to Task 3 (Decision #11) | Appendix: Agent Log #9, #10; Decision Register #10, #11 |
| Notebook preamble | Agent drafted preamble with `src` imports and ID drop | I rewrote the preamble as standalone code (no `src` dependency) with `df.describe()` only; ID drop deferred to Task 3 (Decision #11, #12) | Appendix: Agent Log #10, #11; Decision Register #11, #12 |
| Plot 1 code (class balance) | Agent provided initial snippet using `src` imports | I rewrote Plot 1 as standalone code with inline `TARGET`, percentage labels, human-readable tick labels. Verified bar heights match `df["Exited"].value_counts()` | Appendix: Agent Log #11; Decision Register #12; screenshot `[screenshot_plot1]`; commit `[hash_plot1]` |
| Plots 2–10 code | Agent provided code snippets for all remaining plots | I [ran each cell / adapted each snippet] and verified outputs against dataframe checks listed in Section 2B | Screenshots: `[screenshot_plot2]` … `[screenshot_plot10]`; commit `[hash_plots]` |
| Section 2 draft (this document) | Agent drafted 2A–2E with placeholders | I filled all `[placeholders]` after running the full notebook; cross-checked every percentage and count against `value_counts()`, `groupby().mean()`, and `.corr()` outputs | Appendix: Agent Log #12; commit `[hash_section2_final]` |
| Data-quality checks (2C) | Agent listed 12 checks as hypotheses | I confirmed or ruled out each check by running the validation commands in the notebook | Section 2C status column filled in; notebook cell outputs |
| *[add rows as project progresses]* | | | |
