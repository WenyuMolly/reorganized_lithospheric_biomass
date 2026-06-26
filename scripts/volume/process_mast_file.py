#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "src" / "biomass" / "volume" / "process_mast_file.py"

runpy.run_path(str(TARGET), run_name="__main__")
