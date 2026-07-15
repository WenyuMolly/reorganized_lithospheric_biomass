#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy

from biomass.io import PROJECT_ROOT

TARGET = PROJECT_ROOT / "src" / "biomass" / "preprocessing" / "mast_get_by_cds.py"

(PROJECT_ROOT / "data" / "raw" / "mast").mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT / "data" / "raw" / "mast")
runpy.run_path(str(TARGET), run_name="__main__")
