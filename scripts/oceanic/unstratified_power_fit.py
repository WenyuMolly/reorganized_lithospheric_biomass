#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "oceanic" / "unstratified_power_fit.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: unstratified_power_fit.py [--exclude-shallow | --include-shallow] [--n-draws N] [--seed SEED] [--output-dir DIR] [--z122-scenario {mc,low,base,high}]")
    print("\nRuns the unstratified oceanic depth-power-law Monte Carlo workflow.")
    sys.exit(0)

if "--output-dir" not in sys.argv[1:]:
    run_id = os.environ.get("BIOMASS_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = "with_shallow" if "--include-shallow" in sys.argv[1:] else "without_shallow"
    sys.argv.extend([
        "--output-dir",
        str(PROJECT_ROOT / "runs" / "oceanic" / run_id / "unstratified_power_law" / dataset),
    ])

os.chdir(PROJECT_ROOT)
runpy.run_path(str(TARGET), run_name="__main__")
