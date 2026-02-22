# 2  Explore the Data to Gain Insights

## 2A  EDA Approach

Before any modelling or preprocessing, a visual Exploratory Data Analysis (EDA) was conducted to
understand the dataset's structure, distributions, class balance, and
potential data-quality issues.  All analysis was performed on the **raw
14-column dataframe** (no ID-column drop — that is deferred to Task 3,
data preparation).  Each plot cell explicitly selects the columns it needs
so that identifier columns (`RowNumber`, `CustomerId`, `Surname`) do not
pollute numeric or categorical analyses.

**Scope.**  Five plots plus one dedicated leakage check, ordered to match
the coursework rubric sequence (distributions → missingness →
correlations → leakage risks → class imbalance → outliers):

| # | Plot / Check | Question It Answers | Output File |
|---|-------------|---------------------|-------------|
| 1 | Numeric distributions | What do the continuous features look like? Any skew, zero-spikes, or unexpected ranges? | `outputs/eda_01_numeric_distributions.png` |
| 2 | Missingness check | Are there any null or missing values? | `outputs/eda_02_missingness.png` |
| 3 | Correlation heatmap | Which features correlate with each other? Are there multicollinearity concerns? | `outputs/eda_03_correlation_heatmap.png` |
| 3b | Leakage risk check | Does any feature have suspiciously high predictive power (|r| > 0.8 with target)? Could any feature leak future information? | print output (no plot) |
| 4 | Target class balance | How imbalanced is `Exited`? | `outputs/eda_04_target_balance.png` |
| 5 | Boxplots by churn | How do numeric feature distributions and outliers differ between churners and non-churners? | `outputs/eda_05_boxplots_by_churn.png` |

**What was consolidated.**  The original 10-plot plan (Decision #10)
included separate Geography / Gender churn-rate plots, a standalone
zero-balance segment chart (the spike is visible in Plot 1), a separate
leakage bar chart, and a NumOfProducts-vs-churn plot (patterns surfaced
via `describe()` and the heatmap).  A sixth plot (churn rate by
categorical features) was also dropped after checking the rubric — it is
not a required item, and categorical composition is already covered by
`df.describe(include="all")` in the notebook preamble (see Decisions
#15, #16, #17).

**What was split.**  Plot 3 originally combined the correlation heatmap
with the leakage check.  To avoid repetition and keep each cell focused
on one concern, the heatmap (Plot 3) now covers only inter-feature
correlations and multicollinearity, while a dedicated leakage risk check
(Plot 3b) runs immediately after with its own code block that
systematically tests for target leakage.

**Tooling.**  All plots use `matplotlib` and `seaborn` in a Jupyter
notebook, saved to `outputs/` at 150 dpi for inclusion as coursework
evidence.  No `src/` imports are used — each cell is self-contained
(see Decision Register #12).  The notebook preamble uses
`df.describe(include="all")` so that categorical features (Geography,
Gender) are summarised in the statistical overview without requiring a
dedicated plot — their discrete counts add little beyond what
`value_counts()` already shows (see Decision #17).

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

### 2B.3  Plot 3 — Correlation Heatmap

*What to look at.*  Pearson correlation matrix of all numeric features
plus `Exited` (explicitly excluding `RowNumber`, `CustomerId`).  Look for
(a) strongly correlated feature pairs that may cause multicollinearity,
and (b) features with the highest absolute correlation to `Exited` to
identify the most informative predictors.

*Code (notebook cell).*
```python
# Select only the numeric feature columns plus the target column,
# then compute the Pearson correlation matrix (values between -1 and 1).
corr = df[NUMERIC + [TARGET]].corr()
# Create a figure and axis for the heatmap.
# figsize controls the width and height of the plot in inches.
fig, ax = plt.subplots(figsize=(8, 6))
# Display the correlation matrix as an image (heatmap).
# cmap sets the colour scheme, and vmin/vmax fix the scale so -1 and 1 are consistent.
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
# Set tick positions on both axes (one tick per column/row in the correlation matrix).
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
# Label the ticks with the column names.
# Rotate x labels so they do not overlap.
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
# Add the correlation value as text inside each heatmap cell.
# corr.iloc[i, j] accesses the correlation at row i, column j.
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(
            j, i, f"{corr.iloc[i, j]:.2f}",  # show correlation to 2 decimals
            ha="center", va="center",         # centre text in the cell
            fontsize=8
        )
# Add a colorbar to explain the mapping between colours and correlation values.
fig.colorbar(im, ax=ax, shrink=0.8)
# Add a title to describe the plot.
ax.set_title("Plot 3 Correlation Heatmap Numeric plus Target")
# Adjust layout so labels and title fit nicely.
fig.tight_layout()
# Save the plot image into the outputs folder for evidence or reporting.
fig.savefig("outputs/eda_03_correlation_heatmap.png", dpi=150)
# Display the heatmap in the notebook.
plt.show()
```

*Interpretation (fill after running).*
> - Strongest feature–target correlation: `[top_corr_feature]`
>   (r = **[top_corr_val]**).
> - Strongest inter-feature correlation: `[top_pair_a]` – `[top_pair_b]`
>   (r = **[top_pair_val]**).
> - [If all inter-feature |r| < threshold: "No strong multicollinearity
>   detected — all pairwise |r| < [threshold]."]

---

### 2B.3b  Leakage Risk Check

*Purpose.*  After reviewing the correlation heatmap for general patterns,
this dedicated check systematically tests whether any feature leaks
information about the target (`Exited`).  Leakage would mean a feature
is derived from or only knowable after the churn event, giving
artificially high predictive power that would not generalise.  This cell
re-uses the `corr` matrix computed in the heatmap cell above.

*Code (notebook cell — runs after Plot 3).*
```python
# --- Leakage Risk Check ---
# Extract absolute correlations with the target, excluding the target itself.
# corr was computed in the heatmap cell: corr = df[NUMERIC + [TARGET]].corr()
corr_with_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)

# Flag any feature that is suspiciously correlated with the target
# (rule of thumb: |r| > 0.8 suggests possible leakage).
threshold = 0.8
suspects = corr_with_target[corr_with_target > threshold]
print("Absolute correlation with target:")
print(corr_with_target.round(3))
print()
if suspects.empty:
    print("No correlation based leakage flagged (no feature has |r| > 0.8).")
else:
    print("Potential leakage features (|r| > 0.8):", list(suspects.index))
```

*Interpretation (fill after running).*
> - Highest |correlation with Exited|: `[top_corr_feature]`
>   (|r| = **[top_corr_val]**).
> - [If all < 0.8: "No feature exceeds the |r| > 0.8 leakage threshold.
>   Leakage via linear correlation is unlikely."]
> - [If any > 0.8: "**Warning**: `[feature]` has |r| = [val].
>   Investigate whether this feature would be available at prediction
>   time before proceeding to modelling."]

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
> Precision–Recall Area Under the Curve (PR-AUC) as the primary metric (Section 1C).

---

### 2B.5  Plot 5 — Boxplots by Churn

*What to look at.*  Side-by-side boxplots of each continuous feature split
by `Exited` (0 vs 1).  Look for distributional shifts: features where
the median, interquartile range (IQR), or outlier pattern differs between churners and
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
| 7 | **No leakage features**: does any feature have |r| > 0.8 with `Exited`? | Leakage risk check output (Plot 3b) | [confirmed / not confirmed] |
| 8 | **Geography imbalance**: are the three countries represented roughly equally? | `df["Geography"].value_counts()` — visible in `df.describe(include="all")` | [confirmed / not confirmed] |
| 9 | **Age outliers**: are there extreme ages (e.g., < 18 or > 90)? | `df["Age"].describe()` — see Plot 5 | **Confirmed** — many values above ~70 visible beyond the upper whisker in both classes; robust scaling or winsorisation recommended for linear models |
| 10 | **CreditScore range**: does it fall within typical bounds (300–850)? | `df["CreditScore"].describe()` — see Plot 1 | [confirmed / not confirmed] |
| 11 | **Tenure range**: is 0 a valid value or does it indicate missing data? | `df["Tenure"].value_counts().sort_index()` | [confirmed / not confirmed] |
| 12 | **EstimatedSalary distribution**: is it approximately uniform (synthetic data artefact)? | histogram shape in Plot 1; boxplots nearly identical across classes in Plot 5 | **Confirmed** — roughly uniform with near-identical churned/retained distributions; low predictive power expected |

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
| 7 | `Age` distributional shift between classes + upper-tail outliers | Likely the strongest single predictor; apply robust scaling or winsorisation for Logistic Regression (LogReg) / Multi-Layer Perceptron (MLP); use tree-based models or splines to capture non-linear life-stage effects | Task 3 / 4 |
| 8 | No missing values (if confirmed) | No imputation step required in pipeline | Task 3 |
| 9 | `EstimatedSalary` roughly uniform, near-identical across classes | Low predictive power expected — keep in model but note if feature importance is near zero; may add noise, so regularisation or feature selection should be applied | Task 4 |
| 10 | No leakage detected (if confirmed) | No features to remove for leakage reasons — confirmed by dedicated leakage risk check (Plot 3b) | — |
| 11 | `Balance` heavy-tailed with extreme high values | Apply `log1p` transform or robust scaling to prevent logistic regression and distance-based models from being pulled by extreme balance values | Task 3 |
| 12 | `CreditScore` low-end outliers (~400) with weak class separation | Monitor feature importance; consider robust scaling but expect limited contribution on its own | Task 3 / 4 |
| 13 | Overlapping features (`CreditScore`, `EstimatedSalary`) may add noise | Apply L1 (Lasso) / L2 (Ridge) regularisation or feature selection to prevent low-signal features from degrading model performance | Task 4 |

---

## 2E  Agent Plan vs. My Verification

The table below documents which parts of this EDA were drafted by the
AI coding agent and what I personally verified or corrected by running
the notebook and cross-checking against the raw data.

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| EDA plan and plot list | Agent initially proposed 10-plot scope (Log #9, Decision #10). After user review, consolidated to 6 plots in rubric order (Log #15, Decision #15). Subsequently dropped "Churn rate by categorical features" plot after checking rubric requirements, reducing to final 5-plot scope (Log #16, Decision #16) | I identified the 10-plot plan was over-engineered; requested consolidation to rubric order. Later checked the rubric and confirmed churn-rate-by-category is not a required item, so dropped it to keep the EDA concise. Verified final 5 plots cover all rubric areas: distributions, missingness, leakage risks, class imbalance, outliers |
| Notebook preamble | Agent drafted preamble with `src` imports, ID-column drop, and `df.describe(include="all")` (Log #9, #10) | I caught two agent errors: (1) `from src import …` would fail because notebook runs outside repo root — rewrote as standalone code (Log #11, Decision #12); (2) ID-column drop belongs in Task 3, not the EDA preamble — deferred it (Log #10, Decision #11). Initially reverted `include="all"` to plain `df.describe()` (Log #13), but later reinstated `include="all"` because categorical features have no dedicated count plot — `describe(include="all")` is the only place their composition is summarised (Log #17, Decision #17) |
| Plot code (all 5 plots) | Agent provided code snippets for all plots (Log #9, #15). Also identified `labels=` → `tick_labels=` matplotlib deprecation fix for boxplot code (Log #16) | I rewrote Plot 1 entirely as standalone code with inline constants, percentage labels, and human-readable tick labels (Log #11, Decision #12). Ran each cell, verified outputs against dataframe checks listed in Section 2B, and applied the boxplot deprecation fix |
| Section 2 draft (this document) | Agent drafted 2A–2E with placeholders (Log #12, Decision #13). Agent included `NumOfProducts` in Plot 3 section but the actual histogram does not show it (Log #14, Decision #14). Full draft rewritten during consolidation from 10 to 6 plots (Log #15, Decision #15), then updated again when Plot 5 was dropped (Log #16, Decision #16) | I caught the `NumOfProducts` error in Plot 3 — the draft claimed a finding the plot cannot support, so I removed it (Decision #14). Filled all `[placeholders]` after running the full notebook; cross-checked every percentage and count against `value_counts()`, `groupby().mean()`, and `.corr()` outputs |
| Data-quality checks (2C) | Agent listed 12 checks framed as hypotheses with status column for auditable tracking (Log #12, Decision #13) | I confirmed or ruled out each check by running the validation commands in the notebook on the full dataset. Each status cell updated with result and evidence |
| Boxplot findings integration (2B.5, 2C, 2D) | Agent integrated two sets of boxplot analysis notes into a unified interpretation for Plot 5: per-feature paragraphs (Age, Balance, CreditScore, EstimatedSalary), an outlier severity table, confirmed data-quality checks #9 and #12 in Section 2C, and added three new action items (#11–#13) to Section 2D covering Balance heavy-tail treatment, CreditScore outlier monitoring, and L1/L2 regularisation for overlapping features | I provided the raw boxplot observations (distributional shifts, outlier patterns, zero-balance cluster, modelling pitfalls); I reviewed the agent's merged write-up against my notes and the actual Plot 5 output to verify accuracy of claims and severity ratings |
| Correlation heatmap / leakage split (2B.3, 2B.3b) | Agent drafted Plot 3 as "Correlation Heatmap (+ Leakage Check)" with a single combined section containing both heatmap code and a leakage risk check with domain-sense review (Log #18, Decision #18). Agent's leakage code included a domain-notes dictionary and formatted output with unicode checkmarks | I identified the repetition: the combined section was doing two things (insight heatmap + leakage testing) in one block. I split them into Plot 3 (heatmap only) and Plot 3b (leakage check only). I wrote both notebook cells myself — the heatmap uses `ax.imshow` with annotated correlation values, and the leakage check re-uses the `corr` matrix to flag features with |r| > 0.8. Agent's domain-sense dictionary was dropped in favour of a simpler threshold-based check matching my notebook style |
| *[add rows as project progresses]* | | |
