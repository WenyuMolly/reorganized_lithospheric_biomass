#!/usr/bin/env python3
from __future__ import annotations

import runpy

from biomass.io import PROJECT_ROOT

TARGET = PROJECT_ROOT / "src" / "biomass" / "volume" / "habitable_volume.py"

runpy.run_path(str(TARGET), run_name="__main__")
