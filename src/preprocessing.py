"""
Task 3 — Preprocessing utilities for the churn classification project.

Provides:
  - Feature-column constants
  - Feature engineering (HasBalance indicator)
  - Stratified 60/20/20 train-validation-test split
  - ColumnTransformer-based preprocessing pipeline
  - Data validation checks

Leakage discipline:
  - The preprocessor is fit ONLY on the training split
  - Deterministic feature engineering (HasBalance) is applied before
    splitting — no statistics are learned, so no leakage risk
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import ID_COLS, TARGET

# ── Feature column constants ─────────────────────────────────────
# These match the columns remaining after load_churn_data(drop_ids=True).
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

ENGINEERED_FEATURES = ["HasBalance"]

ALL_FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES


# ── Feature engineering ──────────────────────────────────────────


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic engineered features.

    Currently adds:
      - HasBalance: 1 if Balance > 0, else 0

    Rationale (EDA Section 2D, Action #3):
      ~36% of customers have exactly zero balance, creating a bimodal
      distribution visible in Plot 1 and Plot 5.  A binary indicator
      lets tree-based models split on zero vs non-zero cleanly, and
      prevents the zero-mass from distorting distance-based models.

    This is a deterministic transform (no learned statistics),
    so it is safe to apply before train-test splitting.
    """
    df = df.copy()
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    return df


# ── Stratified split ─────────────────────────────────────────────


def stratified_split(
    df: pd.DataFrame,
    target: str = TARGET,
    seed: int = 42,
    train_size: float = 0.60,
    val_size: float = 0.20,
    test_size: float = 0.20,
):
    """Stratified 60/20/20 train-validation-test split.

    Returns (train_df, val_df, test_df) with original indices preserved.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    # First split: train (60%) vs temp (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        stratify=df[target],
        random_state=seed,
    )

    # Second split: val (50% of 40% = 20%) vs test (50% of 40% = 20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df[target],
        random_state=seed,
    )

    return train_df, val_df, test_df


# ── Preprocessing pipeline ───────────────────────────────────────


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features.

    Numeric pipeline:  SimpleImputer(median) -> StandardScaler
    Categorical pipeline:  SimpleImputer(most_frequent) -> OneHotEncoder

    The HasBalance engineered feature is included in the numeric list.
    OneHotEncoder uses handle_unknown="ignore" so that unseen categories
    at transform time produce all-zero columns instead of raising.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ── Data validation ──────────────────────────────────────────────


def validate_data(df: pd.DataFrame, stage: str = "raw") -> dict:
    """Run data validation checks and return a report dict.

    Checks (raises ValueError on critical failures):
      1. Expected columns exist
      2. Target is binary (0, 1)
      3. ID columns not present in feature set
      4. Missing values per column
      5. Duplicate rows
      6. Age range check
      7. CreditScore range check
    """
    report = {"stage": stage, "n_rows": len(df), "checks": {}}

    # 1. Expected columns
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    present = set(df.columns)
    missing_cols = sorted(expected - present)
    report["checks"]["expected_columns_missing"] = missing_cols
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # 2. Target is binary
    target_vals = sorted(df[TARGET].unique().tolist())
    report["checks"]["target_values"] = target_vals
    if not set(target_vals).issubset({0, 1}):
        raise ValueError(f"Target has unexpected values: {target_vals}")

    # 3. ID columns not in features
    id_cols_found = [c for c in ID_COLS if c in df.columns]
    report["checks"]["id_columns_present"] = id_cols_found
    if id_cols_found:
        raise ValueError(
            f"ID columns still present: {id_cols_found}. "
            f"Use load_churn_data(drop_ids=True)."
        )

    # 4. Missing values
    missing = df.isnull().sum()
    missing_dict = {col: int(v) for col, v in missing.items() if v > 0}
    report["checks"]["missing_values"] = missing_dict
    report["checks"]["total_missing"] = int(missing.sum())

    # 5. Duplicates
    n_dupes = int(df.duplicated().sum())
    report["checks"]["duplicate_rows"] = n_dupes

    # 6. Age range
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    report["checks"]["age_range"] = [age_min, age_max]
    if age_min < 0:
        raise ValueError(f"Negative age found: {age_min}")

    # 7. CreditScore range
    cs_min, cs_max = int(df["CreditScore"].min()), int(df["CreditScore"].max())
    report["checks"]["credit_score_range"] = [cs_min, cs_max]

    return report


def split_class_balance(train_df, val_df, test_df, target=TARGET):
    """Report class balance across splits."""
    balance = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df[target].value_counts().to_dict()
        total = len(split_df)
        balance[name] = {
            "n": total,
            "class_0": int(counts.get(0, 0)),
            "class_1": int(counts.get(1, 0)),
            "churn_rate": round(counts.get(1, 0) / total, 4),
        }
    return balance
