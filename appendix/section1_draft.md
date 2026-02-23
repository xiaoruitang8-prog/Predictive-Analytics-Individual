# 1  Obtain a Dataset and Frame the Predictive Problem

## 1.1  Dataset and Problem Framing

This project uses the **Customer Churn Dataset**
([Kaggle source, CC0 licence](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data)),
distributed as a CSV file (`customer_churn.csv`).
The dataset contains **[N rows]** records of retail-banking customers across
three European markets (France, Germany, Spain).
Each row represents **one customer at a single point in time** and includes
demographic attributes (age, gender, geography), account-level features
(credit score, balance, number of products, tenure), and behavioural
indicators (active-member flag, credit-card ownership).
Three identifier columns (`RowNumber`, `CustomerId`, `Surname`) are dropped
before modelling as they carry no predictive signal.

**Data access.**
A **200-row stratified sample** is included in the repository at
`data/sample_customer_churn.csv` so that all scripts can run out of the box
for a quick preview.
The **full dataset** (10 000 rows) is not committed; to reproduce the final
results, download the CSV from the
[Kaggle page](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data)
and place it at exactly `data/customer_churn.csv`.
Only the full file is excluded via `.gitignore`; the sample is committed.

The shared utility `src/data_loader.py` reads the full dataset first
(`data/customer_churn.csv`). If that file is missing it falls back to the
sample (`data/sample_customer_churn.csv`) and emits a warning that results
are **preview only**. If neither file is found, a clear error is raised with
the Kaggle link and both required paths.

## 1.2  Target and Prediction Type

| Item | Value |
|------|-------|
| Target column | `Exited` |
| Prediction type | Binary classification |
| Positive class (`Exited = 1`) | Customer **did churn** — they closed their account or left the bank |
| Negative class (`Exited = 0`) | Customer **retained** — they remained an active customer |
| Class balance | **[churn_rate]%** positive / **[retained_rate]%** negative — the dataset is imbalanced toward the majority (retained) class |

The business goal is to **predict which customers are most likely to churn**
so that the bank can intervene with targeted retention offers before
the customer leaves.

## 1.3  Success Metrics and Constraints

### Metrics

| Role | Metric | Why |
|------|--------|-----|
| **Primary** | **PR-AUC** (area under the Precision–Recall curve) | With an imbalanced dataset, PR-AUC gives a more informative summary of performance across thresholds than ROC-AUC, because it is sensitive to false-positive inflation in the minority class (Davis & Goadrich, 2006). |
| **Secondary** | **ROC-AUC** | A widely reported threshold-free metric that allows comparison with published benchmarks. Less sensitive to class imbalance than PR-AUC but still useful as a complementary view. |
| **Operational** | **Recall @ top 20%** | Simulates a realistic business constraint: if the retention team can contact only the top 20% of customers ranked by predicted churn probability, what fraction of actual churners does the model capture? This directly measures the model's usefulness under limited intervention capacity. |

### Constraints

1. **Interpretability.** Stakeholders need to understand *why* a customer is flagged. At minimum, feature-importance rankings and, ideally, per-prediction explanations should be available.
2. **Fairness.** The model should not systematically under-serve customers by `Geography` or `Gender`. Disparate false-negative rates across protected groups would mean some segments receive fewer retention offers.
3. **Cost asymmetry of errors.** A false negative (missing a churner) is more costly than a false positive (offering retention to a loyal customer). Metric choice and threshold selection should reflect this asymmetry.
4. **Runtime.** Training and inference must complete within minutes on a standard laptop, ruling out approaches that require GPU infrastructure.
5. **Reproducibility.** Fixed random seeds (`42`), pinned dependencies (`requirements.txt`), and a single entry-point script (`src/run_all.py`) must allow anyone to reproduce results end-to-end.

## 1.4  Assumptions and Limitations

1. **Cross-sectional snapshot.** The dataset captures customers at a single point in time. We assume this snapshot is representative of the population the model would score in production, but we cannot model temporal dynamics (e.g., trends in churn rate over time).
2. **No temporal leakage check possible.** Without a date column, we cannot verify that features were recorded *before* the churn event. We proceed on the assumption that all features are legitimately available at prediction time.
3. **Stationarity.** We assume the relationship between features and churn is stable. In practice, economic conditions, product changes, or competitor actions could cause distribution shift.
4. **Feature completeness.** The dataset contains only **[n_features]** features. Important predictors available to a real bank — transaction frequency, customer-service call logs, recent product changes — are absent, likely limiting model performance.
5. **Geographic scope.** Only three countries (France, Germany, Spain) are represented. Results may not generalise to other markets.
6. **Label reliability.** We treat `Exited` as ground truth. We assume it was recorded accurately and that the definition of "churn" is consistent across all rows.
7. **No cost matrix available.** We discuss cost asymmetry qualitatively (Section 1.3) but do not have actual monetary values for false positives vs. false negatives, so threshold optimisation is illustrative rather than business-calibrated.
8. **Sample size.** At **[N rows]** rows and a **[churn_rate]%** churn rate, the minority class contains approximately **[n_churners]** examples. This is adequate for the models considered but limits the precision of metrics estimated on validation and test splits.

## 1.5  Agent Plan vs. My Verification

The table below documents which parts of this section were drafted by the
AI coding agent and what I personally verified or corrected.

| Step | What the Agent Did | What I Verified / Corrected |
|------|--------------------|-----------------------------|
| Project scaffolding | Created repo structure, README, `.gitignore`, `requirements.txt` | Reviewed all files; corrected filename inconsistency (`Churn Modelling` → `Churn_Modelling`); required dummy baseline + MLP in model set; required `.xlsx` gitignore |
| Data access setup | Created `src/data_loader.py` with full → sample fallback and Kaggle link; committed 200-row sample CSV; updated README | Ran sanity-check command; tested full-load, sample-fallback, and missing-file error paths; confirmed full CSV not tracked in git, sample committed |
| Section 1 draft | Drafted this markdown (1.1–1.5) with placeholders | Filled in all `[placeholders]` after running notebook; verified metric rationale against lecture notes; checked assumptions against dataset documentation |
| *[add rows as project progresses]* | | |
