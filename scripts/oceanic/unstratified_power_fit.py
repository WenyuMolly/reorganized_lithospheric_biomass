#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "oceanic" / "unstratified_power_fit.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: unstratified_power_fit.py [--exclude-shallow | --include-shallow] [--n-draws N] [--seed SEED] [--output-dir DIR]")
    print("\nRuns the unstratified oceanic depth-power-law Monte Carlo workflow.")
    sys.exit(0)

os.chdir(PROJECT_ROOT)
runpy.run_path(str(TARGET), run_name="__main__")
