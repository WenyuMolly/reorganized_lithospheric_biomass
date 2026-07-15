"""Shared paths, run-directory, and tabular I/O helpers.

This package intentionally contains only generic file-system and table helpers.
Scientific preprocessing, modelling, and biomass integration remain in their
respective domain packages.
"""

from .paths import (
    DATA_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    data_path,
    figure_path,
    run_path,
)
from .runs import create_run_id, get_run_id, run_directory
from .tables import read_csv_checked, write_csv

__all__ = [
    "DATA_DIR",
    "FIGURES_DIR",
    "PROJECT_ROOT",
    "RUNS_DIR",
    "create_run_id",
    "data_path",
    "figure_path",
    "get_run_id",
    "read_csv_checked",
    "run_directory",
    "run_path",
    "write_csv",
]
