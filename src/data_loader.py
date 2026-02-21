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
    "https://www.kaggle.com/datasets/anandshaw2001/"
    "customer-churn-dataset/data"
)
# Relative to the repo root (scripts are run from the repo root)
DATA_PATH_XLSX = os.path.join("data", "Churn_Modelling.xlsx")
DATA_PATH_CSV = os.path.join("data", "Churn_Modelling.csv")
DATA_PATH = DATA_PATH_XLSX  # preferred format

ID_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"


def _missing_file_message() -> str:
    return (
        f"\n{'=' * 64}\n"
        f"ERROR: Dataset file not found.\n\n"
        f"  Looked for (in order):\n"
        f"    1. {DATA_PATH_XLSX}\n"
        f"    2. {DATA_PATH_CSV}\n\n"
        f"  Expected name : Churn_Modelling.xlsx  (or .csv)\n"
        f"  Expected folder: data/\n\n"
        f"To fix this:\n"
        f"  1. Download from Kaggle (licence: CC0):\n"
        f"     {KAGGLE_URL}\n"
        f"  2. Place the file as:\n"
        f"       {DATA_PATH_XLSX}   (preferred)\n"
        f"     or:\n"
        f"       {DATA_PATH_CSV}    (also accepted)\n\n"
        f"Then re-run, e.g.:\n"
        f"  python src/run_all.py\n"
        f"{'=' * 64}\n"
    )


def load_churn_data(drop_ids: bool = True) -> pd.DataFrame:
    """Load the churn dataset from data/Churn_Modelling.xlsx (or .csv).

    Tries XLSX first, then falls back to CSV.

    Parameters
    ----------
    drop_ids : bool
        If True (default), drop RowNumber, CustomerId, Surname.

    Returns
    -------
    pd.DataFrame
    """
    if os.path.isfile(DATA_PATH_XLSX):
        df = pd.read_excel(DATA_PATH_XLSX)
    elif os.path.isfile(DATA_PATH_CSV):
        df = pd.read_csv(DATA_PATH_CSV)
    else:
        print(_missing_file_message(), file=sys.stderr)
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH_XLSX}' or '{DATA_PATH_CSV}'. "
            f"See error message above for download instructions."
        )

    if drop_ids:
        cols_to_drop = [c for c in ID_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df
