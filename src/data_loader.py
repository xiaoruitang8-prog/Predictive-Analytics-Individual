"""
Shared data-loading utility for the churn classification project.

Every script that needs the dataset should call load_churn_data()
from this module. It reads the full dataset first; if missing, it
falls back to the committed sample with a warning. If neither file
exists it raises a clear error with download instructions.
"""

import os
import sys
import warnings
import pandas as pd

# ── constants ──────────────────────────────────────────────────────
KAGGLE_URL = (
    "https://www.kaggle.com/datasets/anandshaw2001/"
    "customer-churn-dataset/data"
)
# Relative to the repo root (scripts are run from the repo root)
DATA_PATH = os.path.join("data", "customer_churn.csv")
SAMPLE_PATH = os.path.join("data", "sample_customer_churn.csv")

ID_COLS = ["RowNumber", "CustomerId", "Surname"]
TARGET = "Exited"


def _missing_file_message() -> str:
    return (
        f"\n{'=' * 64}\n"
        f"ERROR: No dataset file found.\n\n"
        f"  Looked for (in order):\n"
        f"    1. {DATA_PATH}          (full, 10 000 rows)\n"
        f"    2. {SAMPLE_PATH}  (sample, 200 rows)\n\n"
        f"To fix this:\n"
        f"  1. Download the CSV from Kaggle (licence: CC0):\n"
        f"     {KAGGLE_URL}\n"
        f"  2. Place it at exactly: {DATA_PATH}\n\n"
        f"Or restore the sample that ships with the repo:\n"
        f"  git checkout -- {SAMPLE_PATH}\n\n"
        f"Then re-run, e.g.:\n"
        f"  python src/run_all.py\n"
        f"{'=' * 64}\n"
    )


def load_churn_data(drop_ids: bool = True) -> pd.DataFrame:
    """Load the churn dataset, preferring the full file over the sample.

    Resolution order:
      1. data/customer_churn.csv        (full dataset)
      2. data/sample_customer_churn.csv (200-row sample, with warning)

    Parameters
    ----------
    drop_ids : bool
        If True (default), drop RowNumber, CustomerId, Surname.

    Returns
    -------
    pd.DataFrame
    """
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    elif os.path.isfile(SAMPLE_PATH):
        warnings.warn(
            f"Full dataset not found at '{DATA_PATH}'. "
            f"Falling back to the 200-row sample at '{SAMPLE_PATH}'. "
            f"Results are PREVIEW ONLY — download the full CSV from "
            f"{KAGGLE_URL} and place it at '{DATA_PATH}' for real results.",
            stacklevel=2,
        )
        df = pd.read_csv(SAMPLE_PATH)
    else:
        print(_missing_file_message(), file=sys.stderr)
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}' or '{SAMPLE_PATH}'. "
            f"See error message above for download instructions."
        )

    if drop_ids:
        cols_to_drop = [c for c in ID_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df
