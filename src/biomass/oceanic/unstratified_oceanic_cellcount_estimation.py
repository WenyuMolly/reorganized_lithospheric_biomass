# oceanic_cellcount_bootstrap_mc_unstrat_log10.py
"""
Oceanic subsurface cell count estimation (UNSTRATIFIED log10-bootstrap)
=======================================================================
- All oceanic crust cell counts are pooled together (no stratification).
- Cell densities are log10-transformed; bootstrap resampling is performed
  in log10 space; then results are converted back via 10**draws.
- Monte Carlo also accounts for per-grid depth uncertainty (maxdepth_sd).
- Volumes computed from ECM1 (DLy, TLy) with 1°×1° area.

NEW:
- Add continental-style depth bins: [0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (km),
  and automatically extend the last edge up to max(maxdepth + 3*maxdepth_sd).
- Output global-by-bin totals across draws (matrix + summary with mean/median/2.5%/97.5%).
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- Parameters ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser(description="Run unstratified oceanic log10-bootstrap biomass estimates.")
dataset_group = parser.add_mutually_exclusive_group()
dataset_group.add_argument("--exclude-shallow", action="store_true", help="Exclude shallow/seawater-contacted samples.")
dataset_group.add_argument("--include-shallow", action="store_true", help="Include shallow/seawater-contacted samples.")
parser.add_argument("--n-draws", type=int, default=1000, help="Number of Monte Carlo draws.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
args = parser.parse_args()

N_DRAWS = args.n_draws
SEED = args.seed
EXCLUDE_SHALLOW = not args.include_shallow
OUTDIR = args.output_dir or (
    PROJECT_ROOT / "runs/oceanic/unstratified_log10_bootstrap_results_by_depth_without_shallow"
    if EXCLUDE_SHALLOW
    else PROJECT_ROOT / "runs/oceanic/unstratified_log10_bootstrap_results_by_depth_with_shallow"
)
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Load data -----------------
inference_df = pd.read_csv(
    PROJECT_ROOT / "runs/volume/submitted/results/inference_and_depth_to_122.0_calculation_oceanic.csv"
)
ecm_df = pd.read_csv(PROJECT_ROOT / "data/raw/oceanic/ecm/ECM1.txt", sep="\t", skiprows=0)
cell_data = pd.read_excel(PROJECT_ROOT / "data/raw/oceanic/oceanic_cell_densities.xlsx")

# ---------------- Unit conversion -----------
def to_cells_per_cm3(row: pd.Series) -> float:
    """Convert cells/g to cells/cm^3 for specified references; otherwise pass through."""
    ref = str(row.get("Reference", "")).lower()
    x   = row["Cell Count"]
    if "santelli" in ref:
        return float(x) * 2.77  # g/cm^3
    if "meyers" in ref or "jacobson" in ref:
        return float(x) * 2.90
    return float(x)

cell_data["Cell Count (cm^3)"] = cell_data.apply(to_cells_per_cm3, axis=1)
cell_data = cell_data[cell_data["Cell Count (cm^3)"] > 0]

# Optionally remove shallow and seawater-contacted samples.
EXCLUDE_REFS = ["santelli", "jacobson", "meyers"]  # case-insensitive
if EXCLUDE_SHALLOW:
    cell_data = cell_data[cell_data["Depth for Power Fit"] > 0.3]
    ref_str = cell_data["Reference"].astype(str).str.lower()
    mask_excl = ref_str.str.contains("|".join(EXCLUDE_REFS), na=False)
    cell_data = cell_data.loc[~mask_excl].copy()
cell_data.to_csv(f"{OUTDIR}/data_for_analysis.csv", index=False)

# Build log10 pool (unstratified -> one pool for all domains)
pool_log10 = np.log10(
    pd.to_numeric(cell_data["Cell Count (cm^3)"], errors="coerce").dropna().values
)
pool_log10 = pool_log10[np.isfinite(pool_log10)]
if pool_log10.size == 0:
    raise RuntimeError("No valid log10 cell-count values remain for bootstrap.")

# ---------------- ECM1 merge ----------------
ecm_df.columns = [
    'Numb','Lon','Lat','Hcc','Sed','Hc','Type',
    'DLy1','DLy2','DLy3','TLy1','TLy2','TLy3',
    'Vp1','Vp2','Vp3','Vs1','Vs2','Vs3',
    'Vpn','Vsn','Rho1','Rho2','Rho3','Rhon'
]
ecm_clean = ecm_df[['Lon','Lat','Sed','DLy1','DLy2','DLy3','TLy1','TLy2','TLy3']].copy()

inference_df['lat_rounded'] = inference_df['lat'].round(1)
inference_df['lon_rounded'] = inference_df['lon'].round(1)
ecm_clean['lat_rounded']    = ecm_clean['Lat'].round(1)
ecm_clean['lon_rounded']    = ecm_clean['Lon'].round(1)

merged = pd.merge(inference_df, ecm_clean, on=['lat_rounded','lon_rounded'], how="left")

# ---------------- Depth bins (auto-extended) ---------------
BASE_BINS_KM = np.array([0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

maxdepth = pd.to_numeric(merged.get("maxdepth", np.nan), errors="coerce").values
sd = pd.to_numeric(merged.get("maxdepth_sd", 0.0), errors="coerce").fillna(0.0).values \
     if hasattr(merged.get("maxdepth_sd", None), "values") else np.zeros_like(maxdepth)

upper_conservative = np.nanmax(maxdepth + 3.0 * sd) if maxdepth.size else BASE_BINS_KM[-1]
if not np.isfinite(upper_conservative):
    upper_conservative = np.nanmax(maxdepth) if np.isfinite(np.nanmax(maxdepth)) else BASE_BINS_KM[-1]

if upper_conservative > BASE_BINS_KM[-1] + 1e-9:
    DEPTH_BINS_KM = np.concatenate([BASE_BINS_KM, [float(upper_conservative)]])
else:
    DEPTH_BINS_KM = BASE_BINS_KM.copy()

NBINS = len(DEPTH_BINS_KM) - 1

# ---------------- Helpers -------------------
def grid_area_cm2(lat_deg: float) -> float:
    """Approximate 1°×1° cell area in cm²."""
    lat_rad = np.radians(lat_deg)
    d_cm = 11_132_000.0  # 111.32 km in cm
    return d_cm * d_cm * np.cos(lat_rad)

def add_segment_to_bins(z0_km: float, z1_km: float, density_cm3: float,
                        area_cm2: float, bins_km: np.ndarray, acc_vec: np.ndarray):
    """
    Add cells from a vertical segment [z0, z1] (km) with constant density (cells/cm^3).
    Cells = density * volume; volume for thickness Δz is (Δz[km] * 1e5 [cm/km]) * area_cm2.
    Accumulate overlap with each depth bin into acc_vec (length = nbins).
    """
    if not (np.isfinite(z0_km) and np.isfinite(z1_km) and np.isfinite(density_cm3) and np.isfinite(area_cm2)):
        return
    if density_cm3 <= 0:
        return
    z0 = float(z0_km); z1 = float(z1_km)
    if z1 <= z0:
        return
    per_km_cells = density_cm3 * area_cm2 * 1e5  # cells per km-thickness
    for j in range(len(bins_km) - 1):
        b0, b1 = bins_km[j], bins_km[j+1]
        overlap = max(0.0, min(z1, b1) - max(z0, b0))
        if overlap > 0.0:
            acc_vec[j] += per_km_cells * overlap

rng = np.random.default_rng(SEED)
all_draws = []                 # per-draw per-grid table (kept as before)
by_depth_all = np.zeros((N_DRAWS, NBINS), dtype=float)  # per-draw global by-depth totals

# ---------------- Monte Carlo ---------------
for draw_idx in range(N_DRAWS):
    recs = []
    by_depth_vec = np.zeros(NBINS, dtype=float)

    for _, r in merged.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])

        # Perturb isotherm depth (km)
        z_mean = float(r.get("maxdepth", np.nan))
        z_sd   = float(r.get("maxdepth_sd", 0.0) or 0.0)
        z_pert = rng.normal(loc=z_mean, scale=z_sd) if (np.isfinite(z_mean) and z_sd > 0) else z_mean
        if not np.isfinite(z_pert) or z_pert <= 0:
            z_pert = z_mean
        if not np.isfinite(z_pert) or z_pert <= 0:
            # If still invalid, skip this grid
            continue

        sed = float(r.get('Sed', 0.0) or 0.0)
        d1 = max(float(r.get('DLy1', 0.0) or 0.0) - sed, 0.0)
        d2 = max(float(r.get('DLy2', 0.0) or 0.0) - sed, 0.0)
        d3 = max(float(r.get('DLy3', 0.0) or 0.0) - sed, 0.0)
        t1 = float(r.get('TLy1', 0.0) or 0.0)
        t2 = float(r.get('TLy2', 0.0) or 0.0)
        t3 = float(r.get('TLy3', 0.0) or 0.0)

        # Layer thicknesses within z_pert (km)
        u = m = l = mn = 0.0
        if z_pert <= d1:
            u = z_pert
        elif z_pert <= d2:
            u = t1
            m = z_pert - d1
        elif z_pert <= d3:
            u = t1
            m = t2
            l = z_pert - d2
        else:
            u, m, l = t1, t2, t3
            mn = max(0.0, z_pert - d3)

        # Horizontal area (cm²)
        A_cm2 = grid_area_cm2(lat)
        km2cm = 1e5

        # Volumes (cm³) for legacy per-layer outputs
        vols = {
            "Upper Crust": u * km2cm * A_cm2,
            "Middle Crust": m * km2cm * A_cm2,
            "Lower Crust": l * km2cm * A_cm2,
            "Mantle": mn * km2cm * A_cm2
        }

        # Unstratified: draw four densities independently from the same pool
        draws_log10 = rng.choice(pool_log10, size=4, replace=True)
        draws_lin   = 10.0 ** draws_log10
        uc_d, mc_d, lc_d, mn_d = draws_lin.tolist()

        # Per-layer cell counts (legacy outputs)
        counts = {
            "Upper Crust": vols["Upper Crust"] * uc_d,
            "Middle Crust": vols["Middle Crust"] * mc_d,
            "Lower Crust": vols["Lower Crust"] * lc_d,
            "Mantle": vols["Mantle"] * mn_d
        }
        counts["Total"] = sum(counts.values())

        # --- NEW: accumulate into global depth bins ---
        # Segment tops/bottoms in basement coordinates (km)
        # UC: [0, u]; MC: [d1, d1+m]; LC: [d2, d2+l]; Mn: [d3, d3+mn]
        if u  > 0: add_segment_to_bins(0.0,         u,          uc_d, A_cm2, DEPTH_BINS_KM, by_depth_vec)
        if m  > 0: add_segment_to_bins(d1,          d1 + m,     mc_d, A_cm2, DEPTH_BINS_KM, by_depth_vec)
        if l  > 0: add_segment_to_bins(d2,          d2 + l,     lc_d, A_cm2, DEPTH_BINS_KM, by_depth_vec)
        if mn > 0: add_segment_to_bins(d3,          d3 + mn,    mn_d, A_cm2, DEPTH_BINS_KM, by_depth_vec)

        recs.append({
            "lat": lat, "lon": lon,
            "depth_draw_km": z_pert,
            **counts
        })

    all_draws.append(pd.DataFrame(recs))
    by_depth_all[draw_idx, :] = by_depth_vec

# ---------------- Per-grid summaries (legacy) ----------------
layers = ["Upper Crust", "Middle Crust", "Lower Crust", "Mantle", "Total"]
summary = all_draws[0][["lat", "lon"]].copy()

for L in layers + ["depth_draw_km"]:
    stack = np.stack([df[L].values for df in all_draws])
    summary[f"{L} Mean"]   = stack.mean(axis=0)
    summary[f"{L} Std"]    = stack.std(axis=0, ddof=0)
    summary[f"{L} 2.5%"]   = np.percentile(stack,  2.5, axis=0)
    summary[f"{L} 50%"]    = np.percentile(stack, 50.0, axis=0)
    summary[f"{L} 97.5%"]  = np.percentile(stack, 97.5, axis=0)

summary.to_csv(f"{OUTDIR}/oceanic_cell_counts_bootstrap_mc_unstrat_log10.csv", index=False)

# ---------------- Global totals (legacy) ----------------
totals = []
for L in layers:
    arr = np.array([df[L].sum() for df in all_draws], dtype=float)
    totals.append({
        "Layer": L,
        "Mean":   float(arr.mean()),
        "Median": float(np.percentile(arr, 50.0)),
        "2.5%":   float(np.percentile(arr,  2.5)),
        "97.5%":  float(np.percentile(arr, 97.5)),
        "Std":    float(arr.std(ddof=0))
    })
pd.DataFrame(totals).to_csv(
    f"{OUTDIR}/oceanic_cell_totals_bootstrap_mc_unstrat_log10.csv",
    index=False
)

# ---------------- NEW: global by-depth outputs ----------------
# 1) Matrix: rows = bins, columns = iter_0001..N
by_depth_matrix = pd.DataFrame({
    "depth_top_km": DEPTH_BINS_KM[:-1],
    "depth_bot_km": DEPTH_BINS_KM[1:]
})
for k in range(N_DRAWS):
    by_depth_matrix[f"iter_{k+1:04d}"] = by_depth_all[k, :]
by_depth_matrix.to_csv(
    f"{OUTDIR}/oceanic_cellcount_by_depth_matrix_unstrat_log10.csv", index=False
)

# 2) Summary across draws per bin
bin_summary_rows = []
for j in range(NBINS):
    vals = by_depth_all[:, j]
    bin_summary_rows.append({
        "depth_top_km": DEPTH_BINS_KM[j],
        "depth_bot_km": DEPTH_BINS_KM[j+1],
        "Mean":   float(np.mean(vals)),
        "Median": float(np.median(vals)),
        "2.5%":   float(np.percentile(vals,  2.5)),
        "97.5%":  float(np.percentile(vals, 97.5)),
        "Std":    float(np.std(vals, ddof=0))
    })
pd.DataFrame(bin_summary_rows).to_csv(
    f"{OUTDIR}/oceanic_cellcount_by_depth_summary_unstrat_log10.csv", index=False
)

print("Done. Results written to:", OUTDIR)
print("Depth bins used (km):", DEPTH_BINS_KM.tolist())
