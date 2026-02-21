# 2  Explore the Data to Gain Insights

## 2A  EDA Approach

Before any modelling or preprocessing, a visual EDA was conducted to
understand the dataset's structure, distributions, class balance, and
potential data-quality issues.  All analysis was performed on the **raw
14-column dataframe** (no ID-column drop — that is deferred to Task 3,
data preparation).  Each plot cell explicitly selects the columns it needs
so that identifier columns (`RowNumber`, `CustomerId`, `Surname`) do not
pollute numeric or categorical analyses.

**Scope.**  Six visual checks, ordered to match the coursework rubric
sequence (distributions → missingness → leakage risks → class imbalance →
outliers):

| # | Plot / Check | Question It Answers | Output File |
|---|-------------|---------------------|-------------|
| 1 | Numeric distributions | What do the continuous features look like? Any skew, zero-spikes, or unexpected ranges? | `outputs/eda_01_numeric_distributions.png` |
| 2 | Missingness check | Are there any null or missing values? | `outputs/eda_02_missingness.png` |
| 3 | Correlation heatmap (+ leakage check) | Which features correlate with each other or the target? Does any feature have suspiciously high predictive power? | `outputs/eda_03_correlation_heatmap.png` |
| 4 | Target class balance | How imbalanced is `Exited`? | `outputs/eda_04_target_balance.png` |
| 5 | Churn rate by category | Does churn rate differ across Geography or Gender? | `outputs/eda_05_churn_by_category.png` |
| 6 | Boxplots by churn | How do numeric feature distributions and outliers differ between churners and non-churners? | `outputs/eda_06_boxplots_by_churn.png` |

**What was consolidated.**  The original 10-plot plan (Decision #10)
included separate Geography / Gender churn-rate plots (merged into Plot 5),
a standalone zero-balance segment chart (the spike is visible in Plot 1),
a separate leakage bar chart (absorbed into Plot 3's heatmap + print
output), and a NumOfProducts-vs-churn plot (patterns surfaced via
`describe()` and the heatmap).  See Decision #15.

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

### 2B.1  Plot 1 — Numeric Distributions

*What to look at.*  Histograms for each continuous feature:
`CreditScore`, `Age`, `Tenure`, `Balance`, `EstimatedSalary`.
Look for skewness, multi-modality, unexpected ranges, and potential
outliers.  The zero-balance spike in `Balance` is visible here
(no separate plot needed).

*Dataframe validation.*
```python
df[["CreditScore", "Age", "Tenure", "Balance",
    "EstimatedSalary"]].describe()
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

---

### 2B.2  Plot 2 — Missingness Check

*What to look at.*  A horizontal bar chart showing null counts per column.
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

### 2B.3  Plot 3 — Correlation Heatmap (+ Leakage Check)

*What to look at.*  Pearson correlation matrix of all numeric features
plus `Exited` (explicitly excluding `RowNumber`, `CustomerId`).  Look for
(a) strongly correlated feature pairs that may cause multicollinearity,
(b) features with the highest absolute correlation to `Exited`, and
(c) any feature with |r| > 0.8 with the target, which would suggest
data leakage.  A printed sorted list of |correlation with Exited|
appears below the heatmap for quick inspection.

*Dataframe validation.*
```python
df[["CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"]].corr()["Exited"].abs().sort_values(ascending=False)
```

*Interpretation (fill after running).*
> - Strongest feature–target correlation: `[top_corr_feature]`
>   (r = **[top_corr_val]**).
> - Strongest inter-feature correlation: `[top_pair_a]` – `[top_pair_b]`
>   (r = **[top_pair_val]**).
> - [If all inter-feature |r| < threshold: "No strong multicollinearity
>   detected — all pairwise |r| < [threshold]."]
> - **Leakage check**: highest |r| with `Exited` is **[leak_val]**.
>   [If < 0.5: "No feature shows suspiciously high correlation with the
>   target.  Leakage is unlikely."]
>   [If > 0.8: "**Warning**: `[feature]` has |r| = [val].  Investigate
>   whether this feature would be available at prediction time."]

---

### 2B.4  Plot 4 — Target Class Balance

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

### 2B.5  Plot 5 — Churn Rate by Category

*What to look at.*  Two side-by-side bar charts: churn rate by
`Geography` (France, Germany, Spain) and churn rate by `Gender`
(Male, Female).  Look for categories with disproportionately high or low
churn.

*Dataframe validation.*
```python
df.groupby("Geography")["Exited"].mean()
df.groupby("Gender")["Exited"].mean()
```
Confirm the rates shown on each subplot match the groupby outputs.

*Interpretation (fill after running).*
> **Geography:** France **[fr_rate]%**, Germany **[de_rate]%**,
> Spain **[es_rate]%**.  **[highest_country]** has the highest churn rate
> at **[highest_rate]%**, roughly **[comparison]** the rate in
> **[lowest_country]**.
>
> **Gender:** Female **[f_rate]%**, Male **[m_rate]%**.
> **[higher_gender]** customers churn at a **[diff_description]** higher
> rate.  This is a potential fairness consideration (Section 1C).

---

### 2B.6  Plot 6 — Boxplots by Churn

*What to look at.*  Side-by-side boxplots of each continuous feature split
by `Exited` (0 vs 1).  Look for distributional shifts: features where
the median, IQR, or outlier pattern differs between churners and
non-churners are likely informative predictors.  Outlier dots beyond
the whiskers flag extreme values.

*Dataframe validation.*
```python
df.groupby("Exited")[["CreditScore", "Age", "Tenure", "Balance",
                       "EstimatedSalary"]].median()
```

*Interpretation.*

> **Age — clearest separation.**  The churned group has a noticeably higher
> median age and a higher interquartile range than the retained group,
> confirming that churn risk increases with age.  Many data points sit above
> the upper whisker in both groups (especially retained), indicating a long
> upper tail with genuine outliers (ages above ~70).  The distributional
> shift suggests churn may change across life stages — tree-based models or
> splines will capture this non-linearity better than a purely linear term.
>
> **Balance — wide spread, zero-inflated.**  Both groups show a very wide
> IQR, with the churned group sitting slightly higher overall.  Extreme high
> values appear above the whiskers, confirming that Balance is heavy-tailed.
> Critically, a large cluster of zero-balance customers is visible as
> outlier dots near 0, creating a bimodal distribution (the zero-spike is
> also visible in Plot 1).  This zero-inflated pattern needs special
> treatment — a binary `HasBalance` indicator should be added to prevent
> the zero mass from distorting tree splits and distance-based models.
>
> **CreditScore — weak discriminator.**  Distributions overlap
> substantially between churned and retained customers, indicating limited
> discriminatory power on its own.  A small number of unusually low scores
> (~400 and below) appear as outlier dots, but the medians and IQRs are
> nearly identical across both classes.
>
> **EstimatedSalary — near-identical distributions.**  The boxplots are
> virtually indistinguishable between churned and retained customers,
> confirming that EstimatedSalary is unlikely to be a strong predictor of
> churn.  The spread is wide but roughly uniform with minimal outlier
> behaviour compared to Balance.
>
> **Summary of outliers identified in this plot:**
>
> | Feature | Outlier Pattern | Severity |
> |---------|----------------|----------|
> | Age | Many unusually high ages above the upper whisker (~70+) | Moderate — may inflate variance in linear models |
> | Balance | Very large balances above the upper whisker; zero-balance cluster acts as structural outlier | High — heavy tail + zero-inflation affect multiple model families |
> | CreditScore | Small number of very low scores (~400) below the lower whisker | Low — few points, weak class separation |
> | EstimatedSalary | Wide spread but less extreme outlier behaviour than Balance | Low — near-uniform, minimal impact expected |

---

## 2C  Data-Quality Issues and Modelling Pitfalls

The following are framed as **checks to perform**, not claims.  Each item
must be confirmed or ruled out by running the notebook on the full
dataset.

| # | Check | How to Verify | Status |
|---|-------|--------------|--------|
| 1 | **Class imbalance**: is the positive class (churners) a minority? | `df["Exited"].value_counts(normalize=True)` — see Plot 4 | [confirmed / not confirmed] |
| 2 | **Missing values**: are there any nulls? | `df.isnull().sum().sum()` — see Plot 2 | [confirmed / not confirmed] |
| 3 | **Zero-balance spike**: does `Balance` have a large mass at zero? | `(df["Balance"] == 0).mean()` — visible in Plot 1 | [confirmed / not confirmed] |
| 4 | **ID columns present**: do `RowNumber`, `CustomerId`, `Surname` remain in the raw dataframe? | `df.columns.tolist()` | [confirmed — must be dropped in Task 3] |
| 5 | **`Surname` cardinality**: high-cardinality string column that would need encoding or removal | `df["Surname"].nunique()` | [confirmed / not confirmed] |
| 6 | **`NumOfProducts` rare categories**: are there products = 3 or 4 with very few rows? | `df["NumOfProducts"].value_counts()` | [confirmed / not confirmed] |
| 7 | **No leakage features**: does any feature have |r| > 0.8 with `Exited`? | correlation print output below Plot 3 | [confirmed / not confirmed] |
| 8 | **Geography imbalance**: are the three countries represented roughly equally? | `df["Geography"].value_counts()` — see Plot 5 | [confirmed / not confirmed] |
| 9 | **Age outliers**: are there extreme ages (e.g., < 18 or > 90)? | `df["Age"].describe()` — see Plot 6 | **Confirmed** — many values above ~70 visible beyond the upper whisker in both classes; robust scaling or winsorisation recommended for linear models |
| 10 | **CreditScore range**: does it fall within typical bounds (300–850)? | `df["CreditScore"].describe()` — see Plot 1 | [confirmed / not confirmed] |
| 11 | **Tenure range**: is 0 a valid value or does it indicate missing data? | `df["Tenure"].value_counts().sort_index()` | [confirmed / not confirmed] |
| 12 | **EstimatedSalary distribution**: is it approximately uniform (synthetic data artefact)? | histogram shape in Plot 1; boxplots nearly identical across classes in Plot 6 | **Confirmed** — roughly uniform with near-identical churned/retained distributions; low predictive power expected |

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
| 7 | `Age` distributional shift between classes + upper-tail outliers | Likely the strongest single predictor; apply robust scaling or winsorisation for LogReg/MLP; use tree-based models or splines to capture non-linear life-stage effects | Task 3 / 4 |
| 8 | No missing values (if confirmed) | No imputation step required in pipeline | Task 3 |
| 9 | `EstimatedSalary` roughly uniform, near-identical across classes | Low predictive power expected — keep in model but note if feature importance is near zero; may add noise, so regularisation or feature selection should be applied | Task 4 |
| 10 | No leakage detected (if confirmed) | No features to remove for leakage reasons | — |
| 11 | `Balance` heavy-tailed with extreme high values | Apply `log1p` transform or robust scaling to prevent logistic regression and distance-based models from being pulled by extreme balance values | Task 3 |
| 12 | `CreditScore` low-end outliers (~400) with weak class separation | Monitor feature importance; consider robust scaling but expect limited contribution on its own | Task 3 / 4 |
| 13 | Overlapping features (`CreditScore`, `EstimatedSalary`) may add noise | Apply regularisation (L1/L2) or feature selection to prevent low-signal features from degrading model performance | Task 4 |

---

## 2E  Agent Plan vs. My Verification

The table below documents which parts of this EDA were drafted by the
AI coding agent and what I personally verified or corrected by running
the notebook and cross-checking against the raw data.

| Step | What the Agent Did | What I Verified / Corrected | Evidence |
|------|--------------------|-----------------------------|----------|
| EDA plan and plot list | Agent initially proposed 10-plot scope (Log #9, Decision #10). User requested consolidation to 6 concise plots in rubric order (Decision #15) | I reviewed the consolidated plot list, confirmed it covers all rubric areas (distributions, missingness, leakage, class imbalance, outliers), and agreed to the 6-plot scope | Appendix: Agent Log #9, #15; Decision Register #10, #15 |
| Notebook preamble | Agent drafted preamble with `src` imports and ID drop | I rewrote the preamble as standalone code (no `src` dependency) with `df.describe()` only; ID drop deferred to Task 3 (Decision #11, #12) | Appendix: Agent Log #10, #11; Decision Register #11, #12 |
| Plot code (all 6 plots) | Agent provided code snippets for all plots | I [ran each cell / adapted each snippet] and verified outputs against dataframe checks listed in Section 2B | Screenshots: `[screenshot_plot1]` … `[screenshot_plot6]`; commit `[hash_plots]` |
| Section 2 draft (this document) | Agent drafted 2A–2E with placeholders | I filled all `[placeholders]` after running the full notebook; cross-checked every percentage and count against `value_counts()`, `groupby().mean()`, and `.corr()` outputs | Appendix: Agent Log #12, #15; commit `[hash_section2_final]` |
| Data-quality checks (2C) | Agent listed 12 checks as hypotheses | I confirmed or ruled out each check by running the validation commands in the notebook | Section 2C status column filled in; notebook cell outputs |
| *[add rows as project progresses]* | | | |
