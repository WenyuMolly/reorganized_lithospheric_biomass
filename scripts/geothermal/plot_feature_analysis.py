#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "geothermal" / "plot_feature_analysis.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        "usage: plot_feature_analysis.py [--domain {oceanic,continental}] "
        "[--train-x PATH] [--test-x PATH] [--train-y PATH] [--test-y PATH] "
        "[--model PATH] [--out-csv PATH] [--out-fig PATH]"
    )
    print("\nBuilds gain plus LOFO feature-importance diagnostics from a trained XGBoost model.")
    sys.exit(0)

runpy.run_path(str(TARGET), run_name="__main__")
