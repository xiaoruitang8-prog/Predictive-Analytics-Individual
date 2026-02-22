"""
Task 3 entry point — Data validation, splitting, and preprocessing.

Usage (from repo root):
    python src/03_preprocess.py

Outputs (saved to outputs/):
    - split_indices.json       — train/val/test row indices for reproducibility
    - preprocessor.joblib      — fitted ColumnTransformer
    - validation_report.json   — data checks + class balance per split
"""

import json
import os
import sys

# Allow imports from repo root when running as `python src/03_preprocess.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src.data_loader import TARGET, load_churn_data
from src.preprocessing import (
    ALL_FEATURES,
    ENGINEERED_FEATURES,
    add_engineered_features,
    build_preprocessor,
    split_class_balance,
    stratified_split,
    validate_data,
)

SEED = 42
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load data (IDs already dropped) ──────────────────────
    print("=" * 60)
    print("TASK 3 — Data Validation, Splitting, and Preprocessing")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    df = load_churn_data(drop_ids=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 2. Validate raw data ────────────────────────────────────
    print("\n[2/6] Running validation checks on raw data...")
    report = validate_data(df, stage="raw")
    for check, value in report["checks"].items():
        print(f"  {check}: {value}")
    print("  All critical checks passed.")

    # ── 3. Feature engineering ──────────────────────────────────
    print("\n[3/6] Adding engineered features...")
    df = add_engineered_features(df)
    print(f"  Added: {ENGINEERED_FEATURES}")
    print(f"  HasBalance value counts:")
    print(f"    {df['HasBalance'].value_counts().to_dict()}")

    # ── 4. Stratified split ─────────────────────────────────────
    print("\n[4/6] Splitting data (60/20/20, stratified, seed=42)...")
    train_df, val_df, test_df = stratified_split(df, seed=SEED)

    balance = split_class_balance(train_df, val_df, test_df)
    report["split_balance"] = balance
    for name, info in balance.items():
        print(f"  {name}: n={info['n']}, "
              f"class_0={info['class_0']}, class_1={info['class_1']}, "
              f"churn_rate={info['churn_rate']:.4f}")

    # Save split indices
    split_indices = {
        "train": train_df.index.tolist(),
        "val": val_df.index.tolist(),
        "test": test_df.index.tolist(),
    }
    indices_path = os.path.join(OUTPUT_DIR, "split_indices.json")
    with open(indices_path, "w") as f:
        json.dump(split_indices, f)
    print(f"  Split indices saved to {indices_path}")

    # ── 5. Fit preprocessor on TRAIN only ───────────────────────
    print("\n[5/6] Fitting preprocessor on training split only...")
    preprocessor = build_preprocessor()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()
    report["preprocessor"] = {
        "n_output_features": len(feature_names),
        "feature_names": feature_names,
        "train_shape": list(X_train_processed.shape),
        "val_shape": list(X_val_processed.shape),
        "test_shape": list(X_test_processed.shape),
    }
    print(f"  Output features ({len(feature_names)}):")
    for fn in feature_names:
        print(f"    {fn}")

    # Save fitted preprocessor
    preprocessor_path = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Fitted preprocessor saved to {preprocessor_path}")

    # ── 6. Save validation report ───────────────────────────────
    print("\n[6/6] Saving validation report...")
    report_path = os.path.join(OUTPUT_DIR, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Validation report saved to {report_path}")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 3 preprocessing complete.")
    print(f"  Train: {X_train_processed.shape}")
    print(f"  Val:   {X_val_processed.shape}")
    print(f"  Test:  {X_test_processed.shape}")
    print(f"\nOutputs saved to {OUTPUT_DIR}/:")
    print(f"  - split_indices.json")
    print(f"  - preprocessor.joblib")
    print(f"  - validation_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
