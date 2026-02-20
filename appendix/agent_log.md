# Agent Usage Log

| # | Date | Task Given to Agent | Agent Output Summary | Verified By Me? | Errors Found | Correction Made |
|---|------|---------------------|----------------------|-----------------|--------------|-----------------|
| 1 | 2026-02-20 | Set up project scaffolding and propose plan for churn classification coursework | Agent created directory structure, requirements.txt, README, .gitignore, and proposed 7-phase plan | Yes | Three issues: (1) README used `Churn Modelling.xlsx` (space) instead of `Churn_Modelling.xlsx` (underscore); (2) Model set excluded dummy baseline and MLP neural net, limiting diversity; (3) `.gitignore` did not exclude `data/*.xlsx` and README lacked data access instructions | (1) Standardised filename to `Churn_Modelling.xlsx` in README; (2) Revised model set to Dummy, LogReg, HistGradientBoosting, MLP; (3) Added `data/*.xlsx` to .gitignore and "Data access" section to README |
| 2 | 2026-02-20 | Apply Step-0 corrections A, B, C per user review | Updated README (filename consistency + data access section), .gitignore (data/*.xlsx), decision register (revised decision #4, added decision #5), agent log (documented errors) | Pending | | |
| 3 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
