#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "oceanic" / "stratified_oceanic_cellcount_estimation.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: stratified_cellcount.py [--exclude-shallow | --include-shallow] [--n-draws N] [--seed SEED] [--output-dir DIR]")
    print("\nRuns the stratified log10 bootstrap oceanic cell-count workflow.")
    sys.exit(0)

os.chdir(PROJECT_ROOT)
runpy.run_path(str(TARGET), run_name="__main__")
