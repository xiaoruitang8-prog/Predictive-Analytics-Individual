"""
Task 3 — Preprocessing utilities for the churn classification project.

Provides:
  - Feature-column constants
  - Stratified 70/15/15 train-validation-test split
  - ColumnTransformer-based preprocessing pipeline
  - Data validation checks (print-only, no output files)

Leakage discipline:
  - The preprocessor is fit ONLY on the training split
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import ID_COLS, TARGET

# ── Feature column constants ─────────────────────────────────────

NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]

SEED = 42


# ── Stratified split ─────────────────────────────────────────────


def stratified_split(df, target=TARGET, seed=SEED):
    """Stratified 70/15/15 train-validation-test split.

    Two-step split:
      1. train (70%) vs temp (30%)
      2. temp -> val (50% of 30% = 15%) vs test (50% of 30% = 15%)
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[target], random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[target], random_state=seed,
    )
    return train_df, val_df, test_df


# ── Preprocessing pipeline ───────────────────────────────────────


def build_preprocessor():
    """Single ColumnTransformer: numeric (imputer + scaler) + categorical (OHE).

    Numeric:  SimpleImputer(median) -> StandardScaler
    Categorical:  SimpleImputer(most_frequent) -> OneHotEncoder

    handle_unknown="ignore" produces all-zero columns for unseen categories.
    remainder="drop" excludes the target and any unlisted columns.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ── Data validation ──────────────────────────────────────────────


def validate_data(df):
    """Run validation checks and print results.

    Checks:
      1. Missing values per column and total
      2. Duplicate rows
      3. Range checks: Age, CreditScore, Balance (min/max)
      4. Target is binary {0, 1}
      5. ID columns not in features
      6. Overall class balance

    Raises ValueError on critical failures.
    """
    print("--- Data Validation ---")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # 1. Missing values
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    print(f"\nMissing values (total): {total_missing}")
    if total_missing > 0:
        for col, n in missing[missing > 0].items():
            print(f"  {col}: {n}")

    # 2. Duplicates
    n_dupes = int(df.duplicated().sum())
    print(f"Duplicate rows: {n_dupes}")

    # 3. Range checks
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    cs_min, cs_max = int(df["CreditScore"].min()), int(df["CreditScore"].max())
    bal_min, bal_max = float(df["Balance"].min()), float(df["Balance"].max())
    print(f"Age range: {age_min} – {age_max}")
    print(f"CreditScore range: {cs_min} – {cs_max}")
    print(f"Balance range: {bal_min:.2f} – {bal_max:.2f}")
    if age_min < 0:
        raise ValueError(f"Negative age: {age_min}")

    # 4. Target binary
    target_vals = set(df[TARGET].unique())
    if not target_vals.issubset({0, 1}):
        raise ValueError(f"Target has unexpected values: {target_vals}")
    print(f"Target values: {sorted(target_vals)}")

    # 5. ID columns
    id_found = [c for c in ID_COLS if c in df.columns]
    if id_found:
        raise ValueError(f"ID columns still present: {id_found}")
    print(f"ID columns in features: none (correct)")

    # 6. Overall class balance
    vc = df[TARGET].value_counts()
    churn_rate = vc.get(1, 0) / len(df)
    print(f"Overall class balance: {vc.to_dict()}, churn rate: {churn_rate:.4f}")

    print("--- All checks passed ---\n")


def print_split_balance(train_df, val_df, test_df, target=TARGET):
    """Print class balance for each split."""
    print("--- Class Balance Per Split ---")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = len(split_df)
        n1 = int(split_df[target].sum())
        n0 = n - n1
        print(f"  {name:5s}: n={n:5d}, class_0={n0:4d}, class_1={n1:4d}, "
              f"churn_rate={n1/n:.4f}")
    print()
