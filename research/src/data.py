"""Load and split the UCI Bank Marketing data."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bank_marketing_raw.csv"


def load_data():
    """Load cached/source data and encode the target as no=0, yes=1."""
    if RAW_DATA_PATH.exists():
        raw = pd.read_csv(RAW_DATA_PATH)
        y_raw = raw.pop("y").astype("string").str.strip().str.lower()
        X = raw
    else:
        dataset = fetch_ucirepo(id=222)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.iloc[:, 0].astype("string").str.strip().str.lower()

        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = X.copy()
        raw["y"] = y_raw
        raw.to_csv(RAW_DATA_PATH, index=False, encoding="utf-8-sig")

    # duration is recorded during the call and would leak information unavailable
    # when deciding whether to contact a customer.
    X = X.drop(columns=["duration"], errors="ignore")

    y = y_raw.map({"no": 0, "yes": 1})
    if y.isna().any():
        unknown_labels = sorted(y_raw[y.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected target labels in Bank Marketing data: {unknown_labels}")

    y = y.astype("int8").rename("y")
    return X, y


def split_data(X, y):
    """Create the fixed, stratified 70/30 train/test split used in all runs."""
    return train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
