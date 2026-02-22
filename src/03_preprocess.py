"""
Task 3 entry point — Data validation, splitting, and preprocessing.

Usage (from repo root):
    python src/03_preprocess.py

All results are printed to stdout.  No output files are saved.
Task 4 reproduces the same split by calling stratified_split() with seed=42.
"""

import os
import sys

# Allow imports from repo root when running as `python src/03_preprocess.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import TARGET, load_churn_data
from src.preprocessing import (
    build_preprocessor,
    print_split_balance,
    stratified_split,
    validate_data,
)


def main():
    print("=" * 60)
    print("TASK 3 — Data Validation, Splitting, and Preprocessing")
    print("=" * 60)

    # 1. Load data (IDs dropped by loader)
    print("\n[1] Loading data...")
    df = load_churn_data(drop_ids=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # 2. Validate
    print("[2] Validation checks...")
    validate_data(df)

    # 3. Stratified split 70/15/15
    print("[3] Splitting (70/15/15, stratified, seed=42)...")
    train_df, val_df, test_df = stratified_split(df)
    print_split_balance(train_df, val_df, test_df)

    # 4. Fit preprocessor on train only, transform all splits
    print("[4] Fitting preprocessor on train split only...")
    preprocessor = build_preprocessor()

    X_train = train_df.drop(columns=[TARGET])
    X_val = val_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"Output features ({len(feature_names)}):")
    for fn in feature_names:
        print(f"  {fn}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Preprocessing complete.")
    print(f"  Train: {X_train_t.shape}")
    print(f"  Val:   {X_val_t.shape}")
    print(f"  Test:  {X_test_t.shape}")
    print(f"  No output files saved (Task 4 calls the same functions).")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
