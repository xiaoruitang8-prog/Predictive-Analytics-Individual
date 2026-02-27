# 2  Explore the Data to Gain Insights

Before any modelling or preprocessing, a visual Exploratory Data Analysis
(EDA) was conducted to understand the dataset's structure, distributions,
class balance, and potential data-quality issues.  All analysis was
performed on the **raw 14-column dataframe** (no ID-column drop — that is
deferred to Task 3, data preparation).  Each plot cell explicitly selects
the columns it needs so that identifier columns (`RowNumber`, `CustomerId`,
`Surname`) do not pollute numeric or categorical analyses.

**Preamble: summary statistics.**  The notebook begins with
`df.describe(include="all")` to provide a statistical overview of all
14 columns — numeric ranges, counts, means, and categorical composition
(top, freq, unique).  This summary table sets context for the six visual
checks that follow.

**Scope.**  Six visual checks ordered to build understanding
progressively — starting with class balance to set evaluation context,
then data quality, then feature exploration, ending with leakage as a
final sanity check:

| Section | Check | Why This Order | Output File |
|---------|-------|----------------|-------------|
| 2.1 | Class imbalance | Sets context for evaluation metrics and split discipline | `outputs/eda_01_class_balance.png` |
| 2.2 | Missingness | Tells you if cleaning or imputation is needed before anything else | `outputs/eda_02_missingness.png` |
| 2.3 | Distributions | Shows skew, zero-inflation, tails for continuous features | `outputs/eda_03_distributions.png` |
| 2.4 | Categorical churn rates | Gives segment insights and fairness flags | `outputs/eda_04_churn_by_category.png` |
| 2.5 | Outliers | Highlights extreme values that may affect models | `outputs/eda_05_boxplots_by_churn.png` |
| 2.6 | Leakage risks | Final sanity check — IDs removed, no suspicious feature | `outputs/eda_06_correlation_heatmap.png` + print |

**What was consolidated.**  The original 10-plot plan (Decision #10)
included separate Geography / Gender churn-rate plots, a standalone
zero-balance segment chart (spike visible in Section 2.3), a separate
leakage bar chart, and a NumOfProducts-vs-churn plot (patterns surfaced
via `describe()` and the heatmap).  The categorical churn-rate plot was
initially dropped after checking the rubric (Decisions #15, #16), but
later reinstated as a 4-panel bar chart in Section 2.4 — the visual
format makes categorical differences immediately clear, whereas
`df.describe(include="all")` only shows counts, not churn behaviour
(Decision #27).

**What was reordered.**  The agent's original ordering followed
distributions → missingness → correlations → leakage → categories →
class balance → outliers.  I reorganised to the current flow
(Decision #28): class imbalance first (sets evaluation context), then
missingness (cleaning needs), distributions (feature shapes), categorical
rates (segment insights), outliers (modelling concerns), and leakage last
(final sanity check).  Numbering changed from letter-based (2A, 2B.1)
to decimal (2.1, 2.2) for clarity.

**Tooling.**  All plots use `matplotlib` and `seaborn` in a Jupyter
notebook, saved to `outputs/` at 150 dpi for inclusion as coursework
evidence.  No `src/` imports are used — each cell is self-contained
(see Decision #12).  The notebook preamble uses `df.describe(include="all")`
to summarise categorical composition (counts, top, freq); Section 2.4
then shows how churn *rates* differ across those same categories
(Decision #27).

---

## 2.1  Class Imbalance

*Why first.*  Knowing the class split up front sets context for every
subsequent analysis — it determines metric choice (PR-AUC over accuracy),
split discipline (stratification), and how to read churn-rate plots in
Section 2.4.

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
> Precision–Recall Area Under the Curve (PR-AUC) as the primary metric
> (Section 1.3).

---

## 2.2  Missingness

*Why here.*  Before examining distributions, we need to know whether the
data is complete — missing values would affect every downstream plot and
model.

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

## 2.3  Distributions

*Why here.*  With class balance and completeness established, we now
examine the shape of each continuous feature — skew, zero-inflation, and
tails inform preprocessing choices (scaling strategy, outlier handling).

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

> Figure 2 shows the distributions of five continuous numerical features,
> highlighting heterogeneity across customers and several modelling
> pitfalls.
>
> **Age** is right-skewed with a long upper tail: a small group of older
> customers (above ~70) sit well beyond the bulk of the distribution.
> These extreme values may make linear models more sensitive to outliers,
> pulling coefficients and inflating variance.
>
> **Balance** follows a bimodal distribution — **[zero_bal_pct]%** of
> values are exactly zero, creating a substantial mass at the left edge,
> while the remaining values form a roughly normal upper spread.  A
> single linear term cannot represent this two-regime structure and is
> likely to underfit.
>
> **Tenure**, **CreditScore**, and **EstimatedSalary** are approximately
> uniform.  Uniformity implies weak marginal signal in isolation —
> no single value range strongly predicts churn on its own — and suggests
> a greater reliance on feature interactions.  Models that evaluate
> features independently (e.g., naïve Bayes) may therefore underestimate
> their contribution.
>
> Finally, these five features differ substantially in scale (e.g.,
> Balance in the tens of thousands vs. Tenure in single digits).  Without
> consistent scaling, scale-sensitive models (logistic regression, MLP,
> KNN) may face optimisation difficulties, and their coefficients or
> distances become difficult to compare.

---

## 2.4  Categorical Churn Rates

*Why here.*  After understanding numeric feature shapes, we examine how
churn *behaviour* differs across categorical features — providing segment
insights and flagging potential fairness concerns (Geography, Gender).

*What to look at.*  Four side-by-side bar charts showing churn rate (%)
for each level of `Geography`, `Gender`, `IsActiveMember`, and
`HasCrCard`.  Look for categories with substantially higher or lower
churn rates.  `df.describe(include="all")` (notebook preamble) shows
category counts and composition; this plot shows how churn *behaviour*
differs across categories — an insight that `describe()` cannot provide.

*Code (notebook cell).*
```python
cat_cols = ["Geography", "Gender", "IsActiveMember", "HasCrCard"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 4))
for ax, col in zip(axes, cat_cols):
    rates = df.groupby(col)[TARGET].mean() * 100
    bars = ax.bar(rates.index.astype(str), rates.values, color="teal")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9)
    ax.set_xlabel(col)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"Churn by {col}")
fig.suptitle("Plot 4 — Churn Rate by Categorical Features", y=1.02,
             fontsize=13)
fig.tight_layout()
fig.savefig("outputs/eda_04_churn_by_category.png", dpi=150)
plt.show()
```

*Interpretation (fill after running).*
> - **Geography:** Germany has the highest churn rate (**[fill]%**) —
>   roughly double France (**[fill]%**) and Spain (**[fill]%**).
>   This is the strongest categorical signal and justifies including
>   Geography as a one-hot-encoded feature.
> - **Gender:** Female customers churn at **[fill]%** vs **[fill]%** for
>   male — a notable gap.
> - **IsActiveMember:** Inactive members churn at **[fill]%** vs
>   **[fill]%** for active members — approximately a 2× difference,
>   making this a strong binary predictor.
> - **HasCrCard:** [Describe whether credit card ownership shows a
>   meaningful churn rate difference or not.]

---

## 2.5  Outliers

*Why here.*  With distributions and categorical patterns understood, we
now look for extreme values that may affect model performance, and examine
how feature distributions shift between churners and non-churners.

*What to look at.*  Side-by-side boxplots of each continuous feature split
by `Exited` (0 vs 1).  Look for distributional shifts: features where the
median, IQR, or outlier pattern differs between churners and non-churners
are likely informative predictors.  Outlier dots beyond the whiskers flag
extreme values.

*Dataframe validation.*
```python
df.groupby("Exited")[["CreditScore", "Age", "Tenure", "Balance",
                       "EstimatedSalary"]].median()
```

*Interpretation.*

> **Age — clearest separation.**  The churned group has a noticeably
> higher median age and IQR than the retained group, confirming that churn
> risk increases with age.  Many data points sit above the upper whisker in
> both groups, indicating a long upper tail with genuine outliers (ages
> above ~70).  The distributional shift suggests churn may change across
> life stages — tree-based models will capture this non-linearity better
> than a purely linear term.
>
> **Balance — wide spread, zero-inflated.**  Both groups show a very wide
> IQR, with the churned group sitting slightly higher overall.  Extreme
> high values appear above the whiskers, confirming that Balance is
> heavy-tailed.  A large cluster of zero-balance customers is visible as
> outlier dots near 0, creating a bimodal distribution (also visible in
> Section 2.3).  This zero-inflated pattern may warrant special treatment
> (see Section 2.7, action #3).
>
> **CreditScore — weak discriminator.**  Distributions overlap
> substantially between churned and retained customers, indicating limited
> discriminatory power on its own.  A small number of unusually low scores
> (~400) appear as outlier dots, but medians and IQRs are nearly identical
> across both classes.
>
> **EstimatedSalary — near-identical distributions.**  Boxplots are
> virtually indistinguishable between churned and retained customers,
> confirming that EstimatedSalary is unlikely to be a strong predictor.
> The spread is wide but roughly uniform with minimal outlier behaviour.
>
> **Outlier severity summary:**
>
> | Feature | Outlier Pattern | Severity |
> |---------|----------------|----------|
> | Age | Many unusually high ages above the upper whisker (~70+) | Moderate — may inflate variance in linear models |
> | Balance | Very large balances above the upper whisker; zero-balance cluster acts as structural outlier | High — heavy tail + zero-inflation affect multiple model families |
> | CreditScore | Small number of very low scores (~400) below the lower whisker | Low — few points, weak class separation |
> | EstimatedSalary | Wide spread but less extreme outlier behaviour than Balance | Low — near-uniform, minimal impact expected |

---

## 2.6  Leakage Risks

*Why last.*  This is a final sanity check before proceeding to modelling.
After understanding distributions and patterns, we verify that no feature
leaks information about the target and that identifier columns are excluded.

Two notebook cells:
1. **Correlation heatmap** — visual overview of inter-feature correlations
   and feature–target relationships.
2. **Leakage threshold check** — systematic test for |r| > 0.8 with `Exited`.

### Correlation Heatmap

*What to look at.*  Pearson correlation matrix of all numeric features
plus `Exited` (excluding `RowNumber`, `CustomerId`).  Look for
(a) strongly correlated feature pairs (multicollinearity risk), and
(b) features with the highest absolute correlation to `Exited`.

*Code (notebook cell).*
```python
corr = df[NUMERIC + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Plot 3 Correlation Heatmap Numeric plus Target")
fig.tight_layout()
fig.savefig("outputs/eda_06_correlation_heatmap.png", dpi=150)
plt.show()
```

*Interpretation (fill after running).*
> - Strongest feature–target correlation: `[top_corr_feature]`
>   (r = **[top_corr_val]**).
> - Strongest inter-feature correlation: `[top_pair_a]` – `[top_pair_b]`
>   (r = **[top_pair_val]**).
> - [If all inter-feature |r| < threshold: "No strong multicollinearity
>   detected — all pairwise |r| < [threshold]."]

### Leakage Threshold Check

*Purpose.*  Systematically tests whether any feature has suspiciously high
correlation with the target — which would indicate data leakage.

*Code (notebook cell — runs after heatmap).*
```python
corr_with_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
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

## 2.7  Data-Quality Issues and Actions

### Checks

The following are framed as **checks to perform**, not claims.  Each item
must be confirmed or ruled out by running the notebook on the full dataset.

| # | Check | How to Verify | Status |
|---|-------|--------------|--------|
| 1 | **Class imbalance**: is the positive class (churners) a minority? | `df["Exited"].value_counts(normalize=True)` — see Section 2.1 | [confirmed / not confirmed] |
| 2 | **Missing values**: are there any nulls? | `df.isnull().sum().sum()` — see Section 2.2 | [confirmed / not confirmed] |
| 3 | **Zero-balance spike**: does `Balance` have a large mass at zero? | `(df["Balance"] == 0).mean()` — visible in Section 2.3 | [confirmed / not confirmed] |
| 4 | **ID columns present**: do `RowNumber`, `CustomerId`, `Surname` remain in the raw dataframe? | `df.columns.tolist()` | [confirmed — must be dropped in Task 3] |
| 5 | **`Surname` cardinality**: high-cardinality string column that would need encoding or removal | `df["Surname"].nunique()` | [confirmed / not confirmed] |
| 6 | **`NumOfProducts` rare categories**: are there products = 3 or 4 with very few rows? | `df["NumOfProducts"].value_counts()` | [confirmed / not confirmed] |
| 7 | **No leakage features**: does any feature have |r| > 0.8 with `Exited`? | Leakage threshold check — see Section 2.6 | [confirmed / not confirmed] |
| 8 | **Geography imbalance**: are the three countries represented roughly equally? | `df["Geography"].value_counts()` — visible in `df.describe(include="all")` and Section 2.4 | [confirmed / not confirmed] |
| 9 | **Age outliers**: are there extreme ages (e.g., < 18 or > 90)? | `df["Age"].describe()` — see Section 2.5 | **Confirmed** — many values above ~70 visible beyond the upper whisker; robust scaling or winsorisation recommended for linear models |
| 10 | **CreditScore range**: does it fall within typical bounds (300–850)? | `df["CreditScore"].describe()` — see Section 2.3 | [confirmed / not confirmed] |
| 11 | **Tenure range**: is 0 a valid value or does it indicate missing data? | `df["Tenure"].value_counts().sort_index()` | [confirmed / not confirmed] |
| 12 | **EstimatedSalary distribution**: is it approximately uniform (synthetic data artefact)? | histogram shape in Section 2.3; boxplots nearly identical across classes in Section 2.5 | **Confirmed** — roughly uniform with near-identical churned/retained distributions; low predictive power expected |

### Actions

High-level actions arising from EDA findings.  Implementation details
belong in Task 3 (data preparation) and Task 4 (modelling).

| # | Issue Identified | Action | When |
|---|-----------------|--------|------|
| 1 | Class imbalance (~[churn_pct]% positive) | Use PR-AUC as primary metric; consider class weighting or threshold tuning in modelling | Task 4 |
| 2 | ID columns (`RowNumber`, `CustomerId`, `Surname`) carry no predictive signal | Drop before modelling | Task 3 |
| 3 | Zero-balance spike in `Balance` | Consider adding a binary `HasBalance` indicator feature | Task 3 |
| 4 | `NumOfProducts` rare categories (3, 4) with very few rows | Monitor for instability in cross-validation; consider grouping 3+ into one bin | Task 3 |
| 5 | Geography has different churn rates (see Section 2.4) | Ensure one-hot encoding preserves this signal | Task 3 |
| 6 | `Gender` churn-rate gap (see Section 2.4) | Include `Gender` as a feature; monitor fairness metrics post-modelling | Task 4 |
| 7 | `Age` distributional shift between classes + upper-tail outliers | Likely the strongest single predictor; apply robust scaling or winsorisation for LogReg / MLP; tree-based models handle non-linearity naturally | Task 3 / 4 |
| 8 | No missing values (if confirmed) | No imputation step required in pipeline | Task 3 |
| 9 | `EstimatedSalary` roughly uniform, near-identical across classes | Low predictive power expected — keep in model but note if feature importance is near zero | Task 4 |
| 10 | No leakage detected (if confirmed) | No features to remove for leakage reasons — confirmed by Section 2.6 | — |
| 11 | `Balance` heavy-tailed with extreme high values | Apply `log1p` transform or robust scaling to prevent linear/distance-based models from being pulled by extreme values | Task 3 |
| 12 | `CreditScore` low-end outliers (~400) with weak class separation | Monitor feature importance; consider robust scaling but expect limited contribution on its own | Task 3 / 4 |
| 13 | Overlapping features (`CreditScore`, `EstimatedSalary`) may add noise | Apply L1 / L2 regularisation or feature selection to prevent low-signal features from degrading performance | Task 4 |
| 14 | `IsActiveMember` and `HasCrCard` churn-rate differences (see Section 2.4) | Include both as features; `IsActiveMember` shows ~2× churn-rate gap — strong binary predictor | Task 3 |

---

## 2.8  Agent Plan vs. My Verification

The table below documents which parts of this EDA were drafted by the
AI coding agent and what I personally verified or corrected by running
the notebook and cross-checking against the raw data.

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| EDA plan and plot list | Agent initially proposed 10-plot scope (Log #9, Decision #10). Consolidated to 6 plots in rubric order (Log #15, Decision #15). Subsequently dropped "Churn rate by categorical features" after checking rubric — not a required item — reducing to 5-plot scope (Log #16, Decision #16). Later suggested a printed table as a minimal alternative | I identified the 10-plot plan was over-engineered; requested consolidation. Decided a bar-chart plot is more effective than a printed table — reinstated as Section 2.4 (4-panel bar chart), restoring the EDA to 6 checks (Decision #27). Verified final 6 checks cover all rubric areas: distributions, missingness, leakage risks, class imbalance, outliers, plus categorical churn behaviour |
| Notebook preamble | Agent drafted preamble with `src` imports, ID-column drop, and `df.describe(include="all")` (Log #9, #10) | Caught two errors: (1) `from src import …` fails outside repo root — rewrote as standalone code (Decision #12); (2) ID-column drop belongs in Task 3, not the EDA preamble (Decision #11). Initially reverted `include="all"` to plain `df.describe()` (Log #13), but reinstated `include="all"` because it is the only place categorical composition is summarised, complementing Section 2.4 (Decision #17) |
| Plot code | Agent provided code snippets for all plots (Log #9, #15). Identified `labels=` → `tick_labels=` matplotlib deprecation fix for boxplot code (Log #16) | Rewrote the class balance plot entirely as standalone code with inline constants and percentage labels (Decision #12). Wrote the categorical churn rates plot (Section 2.4) myself as a 4-panel bar chart (Decision #27). Ran each cell and verified outputs against dataframe checks |
| Section 2 draft structure | Agent drafted original with letter-based numbering (2A, 2B.1, 2B.2 etc.) and a different ordering: distributions → missingness → correlations → leakage → categories → class balance → outliers (Log #12, Decision #13). Redrafted multiple times as scope changed (Logs #15, #16, #26) | Caught the `NumOfProducts` error — draft claimed a finding the distributions plot cannot support (Decision #14). Filled all `[placeholders]` after running the full notebook |
| Data-quality checks (Section 2.7) | Agent listed 12 checks framed as hypotheses with a status column for auditable tracking (Log #12, Decision #13) | Confirmed or ruled out each check by running the validation commands on the full dataset. Updated each status cell with result and evidence |
| Outlier analysis (Section 2.5) | Agent integrated boxplot observations into unified per-feature paragraphs and an outlier severity table. Added three new actions (#11–#13) to Section 2.7 covering Balance heavy-tail, CreditScore monitoring, and L1/L2 regularisation | I provided the raw boxplot observations; reviewed the agent's merged write-up against my notes and the actual plot output to verify accuracy of claims and severity ratings |
| Leakage section (Section 2.6) | Agent drafted heatmap and leakage check as a single combined section, then split them but added an over-engineered domain-sense dictionary (Log #18, Decision #18) | I identified the repetition and split them into two focused cells. I wrote both notebook cells myself — heatmap uses `ax.imshow` with annotated correlation values; leakage check re-uses `corr` and flags |r| > 0.8. Dropped agent's domain-sense dictionary in favour of simpler threshold-based check |
| Categorical churn rate plot (Section 2.4) | Agent proposed this in the 10-plot plan, then dropped it after rubric review (Decision #16). Later suggested a printed table as a minimal alternative (Log #26) | I decided a bar-chart plot is clearer than a printed table — the visual format makes category-level churn-rate gaps immediately obvious. Wrote the 4-panel code myself (Decision #27) |
| **EDA ordering and numbering** | **Agent used letter-based numbering (2A, 2B.1 etc.) and a different section order through all drafts** (Logs #12, #15, #16, #26) | **I reorganised to a more logical analytical flow** (Decision #28): class imbalance first (sets evaluation context) → missingness (cleaning needs) → distributions (feature shapes) → categorical rates (segment insights) → outliers (modelling concerns) → leakage (final sanity check). Changed numbering from letter-based (2A, 2B.1) to decimal (2.1, 2.2). Merged old data-quality checks and actions into single Section 2.7 |
| *[add rows as project progresses]* | | |
