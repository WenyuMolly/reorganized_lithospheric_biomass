#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys

from biomass.io import PROJECT_ROOT

TARGET = PROJECT_ROOT / "src" / "biomass" / "geothermal" / "baseline_xgboost.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        "usage: baseline_xgboost.py [--Attempt ATTEMPT] [--Run RUN] "
        "[--run_type {train,test}] [--if_inference] [--data_path DATA_PATH] "
        "[--is_land] [--params_algorithm {grid,random,None}]"
    )
    print("\nRuns the XGBoost geothermal-gradient training, testing, or inference workflow.")
    sys.exit(0)

(PROJECT_ROOT / "runs" / "geothermal").mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT / "runs" / "geothermal")
runpy.run_path(str(TARGET), run_name="__main__")
