"""Small, explicit helpers for CSV inputs and outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def read_csv_checked(path: str | Path, *, required_columns: Iterable[str] = (), **kwargs: object) -> pd.DataFrame:
    """Read a CSV and raise a useful error when required columns are absent."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV input not found: {csv_path}")

    table = pd.read_csv(csv_path, **kwargs)
    missing = sorted(set(required_columns).difference(table.columns))
    if missing:
        raise ValueError(f"CSV input {csv_path} is missing required columns: {missing}")
    return table


def write_csv(table: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Write a CSV, creating its parent directory, and return its path."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False, **kwargs)
    return csv_path
