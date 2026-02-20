"""
Shared data-loading utility for the churn classification project.

Every script that needs the dataset should call load_churn_data()
from this module. It checks for the file and raises a clear error
with download instructions if it is missing.
"""

import os
import sys
import pandas as pd

# ── constants ──────────────────────────────────────────────────────
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/shantanudhakadd/"
    "bank-customer-churn-prediction/data"
)
# Relative to the repo root (scripts are run from the repo root)
DATA_PATH = os.path.join("data", "Churn_Modelling.xlsx")

ID_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"


def _missing_file_message() -> str:
    return (
        f"\n{'=' * 64}\n"
        f"ERROR: Dataset file not found.\n\n"
        f"  Expected file : {DATA_PATH}\n"
        f"  Expected name : Churn_Modelling.xlsx\n"
        f"  Expected folder: data/\n\n"
        f"To fix this:\n"
        f"  1. Download the Excel file from Kaggle:\n"
        f"     {KAGGLE_URL}\n"
        f"  2. Rename it (if needed) to exactly: Churn_Modelling.xlsx\n"
        f"  3. Place it at: {DATA_PATH}\n\n"
        f"Then re-run, e.g.:\n"
        f"  python src/run_all.py\n"
        f"{'=' * 64}\n"
    )


def load_churn_data(drop_ids: bool = True) -> pd.DataFrame:
    """Load the churn dataset from data/Churn_Modelling.xlsx.

    Parameters
    ----------
    drop_ids : bool
        If True (default), drop RowNumber, CustomerId, Surname.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.isfile(DATA_PATH):
        print(_missing_file_message(), file=sys.stderr)
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. "
            f"See error message above for download instructions."
        )

    df = pd.read_excel(DATA_PATH)

    if drop_ids:
        cols_to_drop = [c for c in ID_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df
