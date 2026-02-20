# Decision Register

| # | Decision | Alternatives Considered | Rationale | Agent Suggested? | I Agreed? | Notes |
|---|----------|------------------------|-----------|-----------------|-----------|-------|
| 1 | Stratified 60/20/20 split | 70/15/15, 80/10/10, k-fold only | 60/20/20 gives a reasonably sized validation set for stable metric estimates and a proper held-out test set; stratification preserves class balance across all three sets | Yes | (fill in) | |
| 2 | Use single sklearn Pipeline with ColumnTransformer | Separate preprocessing scripts, pandas-based transforms | Pipeline prevents train-test leakage by fitting transformers only on training data; ColumnTransformer handles mixed types cleanly | Yes | (fill in) | |
| 3 | Metrics: ROC-AUC, PR-AUC, Recall@top-20% | Accuracy, F1 only | Dataset is imbalanced (~20% churn); PR-AUC is more informative than ROC-AUC for imbalanced data; Recall@top-20% simulates a business scenario of targeting the most likely churners | Yes | (fill in) | |
| 4 | Models: LogReg, RF, GBM | SVM, Neural Net, XGBoost | These three span linear to ensemble methods, are interpretable, and run fast. XGBoost is similar to GBM but adds a dependency; SVM and NNs are harder to interpret for a 2000-word report | Yes | (fill in) | |
| 5 | | | | | | |
