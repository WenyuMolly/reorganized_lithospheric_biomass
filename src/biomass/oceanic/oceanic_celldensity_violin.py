import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================================
# FILE
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INFILE = PROJECT_ROOT / "data/raw/oceanic/oceanic_cell_densities.xlsx"
OUTDIR = PROJECT_ROOT / "figures/generated"
OUTDIR.mkdir(parents=True, exist_ok=True)

COL_LAYER = "Rock Domain"
COL_DENSITY = "Cell Count"              # original units may be cells/g for some refs
COL_DEPTH = "Depth for Power Fit"
COL_REF = "Reference"                   # needed for unit conversion

# ==========================================================
# LOAD
# ==========================================================
df = pd.read_excel(INFILE)
df = df[[COL_LAYER, COL_DENSITY, COL_DEPTH, COL_REF]].dropna()

# clean strings to avoid matching issues
df[COL_LAYER] = df[COL_LAYER].astype(str).str.strip()
df[COL_REF]   = df[COL_REF].astype(str).str.strip()

# ==========================================================
# UNIT CONVERSION
# Convert from cells/g to cells/cm^3 using material densities when possible
# Santelli et al. (2008) basalt glass: 2.77 g/cm^3 from Lima et al. (2020)
# Jacobson Meyers et al. (2014) pillow basalt: 2.90 g/cm^3 from Moore (2001)
# ==========================================================
def convert_to_cells_per_cm3(row) -> float:
    ref = str(row[COL_REF]).lower()
    val = float(row[COL_DENSITY])

    if not np.isfinite(val) or val <= 0:
        return np.nan

    if "santelli" in ref:
        return val * 2.77
    if ("meyers" in ref) or ("jacobson" in ref):
        return val * 2.90

    # Default: assume already cells/cm^3 (or unknown, keep as-is)
    return val

df["Cell Count (cm^3)"] = df.apply(convert_to_cells_per_cm3, axis=1)
df = df.dropna(subset=["Cell Count (cm^3)"])
df = df[df["Cell Count (cm^3)"] > 0]

# Use converted density for log scale
df["log_density"] = np.log10(df["Cell Count (cm^3)"])
df["shallow_flag"] = df[COL_DEPTH] < 0.3

layer_order = [
    "Upper Crust",
    "Middle Crust",
    "Lower Crust",
    "Mantle",
]
df = df[df[COL_LAYER].isin(layer_order)]

# ==========================================================
# STYLE (Science Advances grade)
# ==========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)

# ----------------------------------------------------------
# 1. Full distribution (layer-specific pastel palette)
# ----------------------------------------------------------
layer_palette = {
    "Upper Crust": "#f6d6dc",
    "Middle Crust": "#d9f0dc",
    "Lower Crust": "#d6e9f8",
    "Mantle": "#ead9c8",
}

sns.violinplot(
    data=df,
    x="log_density",
    y=COL_LAYER,
    order=layer_order,
    palette=layer_palette,
    inner=None,
    linewidth=0.8,
    cut=0,
    ax=ax
)

# soften edges
for artist in ax.collections:
    artist.set_edgecolor("#7f7f7f")
    artist.set_alpha(0.95)

# ----------------------------------------------------------
# 2. Median comparison
# ----------------------------------------------------------
for i, layer in enumerate(layer_order):
    sub_full = df[df[COL_LAYER] == layer]
    sub_filtered = sub_full[sub_full["shallow_flag"] == False]

    if len(sub_full) == 0:
        continue

    med_full = sub_full["log_density"].median()
    med_filtered = sub_filtered["log_density"].median()

    ax.plot([med_full, med_full],
            [i - 0.28, i + 0.28],
            color="#333333",
            linewidth=1.1,
            zorder=3)

    ax.scatter(med_filtered, i,
               color="#3b6c8e",
               s=45,
               zorder=4)

# ----------------------------------------------------------
# 3. Shallow sample highlight
# ----------------------------------------------------------
shallow = df[df["shallow_flag"]]

ax.scatter(
    shallow["log_density"],
    shallow[COL_LAYER],
    facecolors="none",
    edgecolors="#e08214",
    linewidths=1.1,
    s=50,
    zorder=5,
)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0],
           marker="o",
           color="none",
           markerfacecolor="#3b6c8e",
           markersize=6,
           label="Median (Dataset A)"),

    Line2D([0], [0],
           color="#333333",
           lw=1.1,
           label="Median (Dataset B)"),

    Line2D([0], [0],
           marker="o",
           color="#e08214",
           markerfacecolor="none",
           markersize=7,
           lw=0,
           label="Shallow samples (< 0.3 mbsf)")
]

ax.legend(
    handles=legend_elements,
    loc="center right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#aaaaaa",
    borderpad=0.6,
    handlelength=1.6,
    fontsize=8.5
)

# ----------------------------------------------------------
# Labels
# ----------------------------------------------------------
ax.set_xlabel(r"log$_{10}$ cell density (cells cm$^{-3}$)",
              fontsize=12,
              labelpad=6)
ax.set_ylabel("")
ax.tick_params(axis="both", which="major", length=4, width=0.8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUTDIR / "Panel_E_shallow_sensitivity.png", dpi=600)
plt.savefig(OUTDIR / "Panel_E_shallow_sensitivity.pdf")
plt.show()
