#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys

from biomass.io import PROJECT_ROOT

TARGET = PROJECT_ROOT / "src" / "biomass" / "oceanic" / "oceanic_celldensity_violin.py"

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: plot_cell_density_violin.py")
    print("\nBuilds the oceanic cell-density shallow-sensitivity violin plot.")
    sys.exit(0)

runpy.run_path(str(TARGET), run_name="__main__")
