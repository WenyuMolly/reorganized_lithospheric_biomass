#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "oceanic" / "stratified_oceanic_cellcount_estimation.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: stratified_cellcount.py [--exclude-shallow | --include-shallow] [--n-draws N] [--seed SEED] [--output-dir DIR] [--z122-scenario {mc,low,base,high}]")
    print("\nRuns the stratified log10 bootstrap oceanic cell-count workflow.")
    sys.exit(0)

if "--output-dir" not in sys.argv[1:]:
    run_id = os.environ.get("BIOMASS_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = "with_shallow" if "--include-shallow" in sys.argv[1:] else "without_shallow"
    sys.argv.extend([
        "--output-dir",
        str(PROJECT_ROOT / "runs" / "oceanic" / run_id / "stratified_log10_bootstrap" / dataset),
    ])

os.chdir(PROJECT_ROOT)
runpy.run_path(str(TARGET), run_name="__main__")
