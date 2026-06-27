# oceanic_cellcount_montecarlo_bootstrap.py
# Monte Carlo with depth-bin aggregation; depth bins auto-extend to max zmax

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mode

# ---------------------------- Config ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

parser = argparse.ArgumentParser(description="Run stratified oceanic log10-bootstrap biomass estimates.")
dataset_group = parser.add_mutually_exclusive_group()
dataset_group.add_argument("--exclude-shallow", action="store_true", help="Exclude shallow/seawater-contacted samples.")
dataset_group.add_argument("--include-shallow", action="store_true", help="Include shallow/seawater-contacted samples.")
parser.add_argument("--n-draws", type=int, default=1000, help="Number of Monte Carlo draws.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
parser.add_argument(
    "--z122-scenario",
    choices=["mc", "low", "base", "high"],
    default="mc",
    help="Depth treatment: mc preserves original maxdepth/maxdepth_sd sampling; low/base/high use maxdepth-sd, maxdepth, or maxdepth+sd.",
)
args = parser.parse_args()

EXCLUDE_SHALLOW = not args.include_shallow
OUTDIR = args.output_dir or (
    PROJECT_ROOT / "runs/oceanic/stratified_log10_results_by_depth_without_shallow"
    if EXCLUDE_SHALLOW
    else PROJECT_ROOT / "runs/oceanic/stratified_log10_results_by_depth_with_shallow"
)
os.makedirs(OUTDIR, exist_ok=True)

res_deg = 1.0       # grid resolution in degrees (default 1° × 1°)
n_draws = args.n_draws      # number of Monte Carlo draws
seed = args.seed            # RNG seed

# Base depth-bin edges in km (same as continental R scripts)
BASE_BINS_KM = np.array([0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

# ---------------------------- Load data ----------------------------
inference_df = pd.read_csv(PROJECT_ROOT / "runs/volume/submitted/results/inference_and_depth_to_122.0_calculation_oceanic.csv")
ecm_df = pd.read_csv(PROJECT_ROOT / "data/raw/oceanic/ecm/ECM1.txt", sep="\t", skiprows=0)
cell_data = pd.read_excel(PROJECT_ROOT / "data/raw/oceanic/oceanic_cell_densities.xlsx")

# Convert from cells/g to cells/cm³ using material densities when possible
def convert_to_cells_per_cm3(row):
    ref = str(row["Reference"]).lower()
    if "santelli" in ref:
        return row["Cell Count"] * 2.77  # g/cm³, from Lima et al. (2020)
    elif "meyers" in ref or "jacobson" in ref:
        return row["Cell Count"] * 2.90  # g/cm³, from Moore (2001)
    else:
        return row["Cell Count"]

cell_data["Cell Count (cm^3)"] = cell_data.apply(convert_to_cells_per_cm3, axis=1)
cell_data = cell_data[cell_data["Cell Count (cm^3)"] > 0]

# Optionally exclude references likely influenced by seawater contact.
EXCLUDE_REFS = ["santelli", "jacobson", "meyers"]  # case-insensitive
if EXCLUDE_SHALLOW:
    ref_str = cell_data["Reference"].astype(str).str.lower()
    mask_excl = ref_str.str.contains("|".join(EXCLUDE_REFS), na=False)
    excluded = cell_data.loc[mask_excl].copy()
    kept = cell_data.loc[~mask_excl].copy()
    excluded.to_csv(f"{OUTDIR}/excluded_refs.csv", index=False)
    cell_data = kept.reset_index(drop=True)
    cell_data = cell_data[cell_data["Depth for Power Fit"] > 0.3]
cell_data.to_csv(f"{OUTDIR}/data_for_analysis.csv", index=False)

# Clean ECM columns
ecm_df.columns = [
    "Numb","Lon","Lat","Hcc","Sed","Hc","Type",
    "DLy1","DLy2","DLy3","TLy1","TLy2","TLy3",
    "Vp1","Vp2","Vp3","Vs1","Vs2","Vs3",
    "Vpn","Vsn","Rho1","Rho2","Rho3","Rhon"
]
ecm_cleaned = ecm_df[["Lon","Lat","Sed","DLy1","DLy2","DLy3","TLy1","TLy2","TLy3"]].copy()

# Spatial merge via rounded coordinates (0.1°)
inference_df["lat_rounded"] = inference_df["lat"].round(1)
inference_df["lon_rounded"] = inference_df["lon"].round(1)
ecm_cleaned["lat_rounded"]   = ecm_cleaned["Lat"].round(1)
ecm_cleaned["lon_rounded"]   = ecm_cleaned["Lon"].round(1)
merged_df = pd.merge(inference_df, ecm_cleaned, on=["lat_rounded","lon_rounded"], how="left")

# ---------------------------- Build dynamic depth bins ----------------------------
# Use a conservative upper bound: maxdepth + 3*maxdepth_sd (falls back to maxdepth if sd missing)
maxdepth = pd.to_numeric(merged_df.get("maxdepth", pd.Series(dtype=float)), errors="coerce").to_numpy()
sd = pd.to_numeric(merged_df.get("maxdepth_sd", pd.Series(0.0, index=merged_df.index)), errors="coerce").fillna(0.0).to_numpy()
upper_nominal = np.nanmax(maxdepth) if maxdepth.size else BASE_BINS_KM[-1]
upper_conservative = np.nanmax(maxdepth + 3.0 * sd) if maxdepth.size else BASE_BINS_KM[-1]
upper_km = float(np.nanmax([BASE_BINS_KM[-1], upper_nominal, upper_conservative]))

DEPTH_BINS_KM = BASE_BINS_KM.copy()
if upper_km > DEPTH_BINS_KM[-1] + 1e-12:
    # Append one last edge so the final bin becomes [last_base_edge, upper_km]
    DEPTH_BINS_KM = np.concatenate([DEPTH_BINS_KM, [upper_km]])

NBINS = len(DEPTH_BINS_KM) - 1
print(f"[INFO] Depth bins (km): {DEPTH_BINS_KM.tolist()}")

# ---------------------------- Bootstrap pool (log10) ----------------------------
domains = ["Upper Crust","Middle Crust","Lower Crust","Mantle"]
log_bootstrap_pool = {}
for dom in domains:
    vals = (
        cell_data.loc[cell_data["Rock Domain"] == dom, "Cell Count (cm^3)"]
        .astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
    )
    vals = vals[vals > 0]
    log_bootstrap_pool[dom] = np.log10(vals) if vals.size > 0 else np.array([], dtype=float)

def sample_log_bootstrap_linear(domain: str, rng: np.random.Generator) -> float:
    """Draw one cell density in linear space, using bootstrap in log10 space."""
    log_vals = log_bootstrap_pool.get(domain, np.array([], dtype=float))
    if log_vals.size == 0:
        return 0.0
    s = rng.choice(log_vals)  # bootstrap in log space
    return float(10.0 ** s)

def grid_area_cm2(lat_deg: float, res_deg: float = 1.0) -> float:
    """Approximate grid cell area (cm²) for a given latitude and resolution in degrees."""
    m_per_deg = 111_320.0
    dx = m_per_deg * res_deg * np.cos(np.radians(lat_deg))
    dy = m_per_deg * res_deg
    area_m2 = dx * dy
    return area_m2 * 1e4  # m² → cm²

def draw_z122_depth_km(row: pd.Series, rng: np.random.Generator, scenario: str) -> float:
    """Return z122 depth for MC or deterministic uncertainty-bound scenarios."""
    z_mean = float(row.get("maxdepth", np.nan))
    z_sd = float(row.get("maxdepth_sd", 0.0) or 0.0)
    z_sd = max(z_sd, 0.0) if np.isfinite(z_sd) else 0.0
    if scenario == "low":
        return float(max(z_mean - z_sd, 0.0)) if np.isfinite(z_mean) else np.nan
    elif scenario == "base":
        return float(z_mean)
    elif scenario == "high":
        return float(z_mean + z_sd) if np.isfinite(z_mean) else np.nan
    else:
        z = rng.normal(loc=z_mean, scale=z_sd) if (np.isfinite(z_mean) and z_sd > 0) else z_mean
        z = z_mean if (not np.isfinite(z) or z <= 0) else z
        return z_mean if (not np.isfinite(z) or z <= 0) else float(z)

def accumulate_bins(u, m, l, mn, ucc, mcc, lcc, mncc, area_cm2, depth_bins_km):
    """
    Slice per-layer contributions into depth bins.
    u, m, l, mn  : thicknesses (km) of UC/MC/LC/Mantle within zmax
    *_cc         : sampled cell densities (cells/cm³) for each layer
    area_cm2     : cell area in cm²
    depth_bins_km: np.array of bin edges in km
    Returns: np.ndarray of length NBINS with counts per bin
    """
    # Build cumulative depth segments [start_km, end_km, density]
    segs = []
    start = 0.0
    if u > 0:  segs.append((start, start+u,   ucc)); start += u
    if m > 0:  segs.append((start, start+m,   mcc)); start += m
    if l > 0:  segs.append((start, start+l,   lcc)); start += l
    if mn > 0: segs.append((start, start+mn, mncc)); start += mn
    zmax = start

    km_to_cm = 1e5
    out = np.zeros(len(depth_bins_km)-1, dtype=float)

    for b in range(len(depth_bins_km)-1):
        b0, b1 = depth_bins_km[b], depth_bins_km[b+1]
        if b0 >= zmax:  # nothing deeper than zmax
            break
        for s0, s1, dens in segs:
            # overlap of [s0,s1] with [b0,b1] but also truncated at zmax
            lo = max(b0, s0)
            hi = min(b1, s1, zmax)
            if hi > lo and dens > 0:
                thickness_km = (hi - lo)
                vol_cm3 = thickness_km * km_to_cm * area_cm2
                out[b] += vol_cm3 * dens
    return out

# ---------------------------- Monte Carlo simulation ----------------------------
rng = np.random.default_rng(seed)
all_samples = []                 # list of per-draw DataFrames (per-grid metrics)
by_depth_totals_by_draw = []     # list of np.array length NBINS (global totals per bin per draw)

for _ in range(n_draws):
    records = []
    depth_bin_sums = np.zeros(NBINS, dtype=float)  # global totals per bin for this draw

    for _, row in merged_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        maxdepth = row["maxdepth"]                 # km
        perturbed_depth = draw_z122_depth_km(row, rng, args.z122_scenario)

        # ECM crustal layer depths (km), subtracting sediment thickness
        sed = float(row.get("Sed", 0.0))
        d1 = max(float(row.get("DLy1", 0.0)) - sed, 0.0)
        d2 = max(float(row.get("DLy2", 0.0)) - sed, 0.0)
        d3 = max(float(row.get("DLy3", 0.0)) - sed, 0.0)
        t1 = float(row.get("TLy1", 0.0))
        t2 = float(row.get("TLy2", 0.0))
        t3 = float(row.get("TLy3", 0.0))

        # Grid area (cm²)
        area_cm2 = grid_area_cm2(lat, res_deg=res_deg)

        # Layer thickness within perturbed depth (km)
        u = m = l = mn = 0.0
        if perturbed_depth <= d1:
            u = perturbed_depth
        elif perturbed_depth <= d2:
            u = t1
            m = perturbed_depth - d1
        elif perturbed_depth <= d3:
            u = t1
            m = t2
            l = perturbed_depth - d2
        else:
            u, m, l = t1, t2, t3
            mn = max(0.0, perturbed_depth - d3)

        # Sample cell densities for each rock domain (cells/cm³)
        # (use the shared rng to keep results reproducible)
        ucc  = sample_log_bootstrap_linear("Upper Crust", rng)
        mcc  = sample_log_bootstrap_linear("Middle Crust", rng)
        lcc  = sample_log_bootstrap_linear("Lower Crust", rng)
        mncc = sample_log_bootstrap_linear("Mantle", rng)

        # Bin-wise accumulation
        bin_counts = accumulate_bins(u, m, l, mn, ucc, mcc, lcc, mncc, area_cm2, DEPTH_BINS_KM)
        depth_bin_sums += bin_counts

        # Also keep layer-wise totals (not depth-binned) for compatibility
        km_to_cm = 1e5
        uc_vol = u  * km_to_cm * area_cm2
        mc_vol = m  * km_to_cm * area_cm2
        lc_vol = l  * km_to_cm * area_cm2
        mn_vol = mn * km_to_cm * area_cm2

        uc = uc_vol * ucc
        mc = mc_vol * mcc
        lc = lc_vol * lcc
        mt = mn_vol * mncc
        tot_cell = uc + mc + lc + mt

        records.append({
            "lat": lat, "lon": lon,
            "Perturbed Depth": perturbed_depth, "Max Depth": maxdepth,
            "Upper Crust Depth": u, "Middle Crust Depth": m,
            "Lower Crust Depth": l, "Mantle Depth": mn,
            "Sampled UCC": ucc, "Sampled MCC": mcc,
            "Sampled LCC": lcc, "Sampled MNCC": mncc,
            "Upper Crust": uc, "Middle Crust": mc, "Lower Crust": lc, "Mantle": mt,
            "Total in Each Cell": tot_cell
        })

    all_samples.append(pd.DataFrame(records))
    by_depth_totals_by_draw.append(depth_bin_sums)

# ---------------------------- Per-grid summaries ----------------------------
summary_layers = [
    "Perturbed Depth","Upper Crust Depth","Middle Crust Depth","Lower Crust Depth","Mantle Depth",
    "Sampled UCC","Sampled MCC","Sampled LCC","Sampled MNCC",
    "Upper Crust","Middle Crust","Lower Crust","Mantle","Total in Each Cell"
]
summary_df = all_samples[0][["lat","lon"]].copy()

for layer in summary_layers:
    stack = np.stack([df[layer].values for df in all_samples])  # shape: (n_draws, n_cells)
    summary_df[f"{layer} Mean"]  = stack.mean(axis=0)
    summary_df[f"{layer} Std"]   = stack.std(axis=0, ddof=0)
    summary_df[f"{layer} 2.5%"]  = np.percentile(stack,  2.5, axis=0)
    summary_df[f"{layer} 50%"]   = np.percentile(stack, 50.0, axis=0)  # median
    summary_df[f"{layer} 97.5%"] = np.percentile(stack, 97.5, axis=0)

summary_df.to_csv(f"{OUTDIR}/oceanic_cell_counts_bootstrap_mc.csv", index=False)

# ---------------------------- Global totals by layer (matrix + summary) ----------------------------
layers_for_totals = ["Upper Crust","Middle Crust","Lower Crust","Mantle"]
layer_totals_by_draw = np.zeros((n_draws, len(layers_for_totals)), dtype=float)
for i, df in enumerate(all_samples):
    layer_totals_by_draw[i, 0] = df["Upper Crust"].sum()
    layer_totals_by_draw[i, 1] = df["Middle Crust"].sum()
    layer_totals_by_draw[i, 2] = df["Lower Crust"].sum()
    layer_totals_by_draw[i, 3] = df["Mantle"].sum()

by_layer_matrix = pd.DataFrame({"Layer": layers_for_totals})
for k in range(n_draws):
    by_layer_matrix[f"iter_{k+1:04d}"] = layer_totals_by_draw[k, :]
by_layer_matrix.to_csv(f"{OUTDIR}/oceanic_cellcount_by_layer_matrix.csv", index=False)

by_layer_summary = []
for j, layer in enumerate(layers_for_totals):
    vals = layer_totals_by_draw[:, j]
    by_layer_summary.append({
        "Layer": layer,
        "Mean":   float(np.mean(vals)),
        "Median": float(np.median(vals)),
        "2.5%":   float(np.percentile(vals,  2.5)),
        "97.5%":  float(np.percentile(vals, 97.5)),
        "Std":    float(np.std(vals, ddof=0))
    })
pd.DataFrame(by_layer_summary).to_csv(f"{OUTDIR}/oceanic_cellcount_by_layer_summary.csv", index=False)

# ---------------------------- Global totals by depth bin ----------------------------
by_depth_totals_by_draw = np.stack(by_depth_totals_by_draw)  # shape: (n_draws, NBINS)

# Matrix: rows are bins, columns are iter_0001..N
depth_matrix = pd.DataFrame({
    "depth_top_km": DEPTH_BINS_KM[:-1],
    "depth_bot_km": DEPTH_BINS_KM[1:]
})
for k in range(n_draws):
    depth_matrix[f"iter_{k+1:04d}"] = by_depth_totals_by_draw[k, :]
depth_matrix.to_csv(f"{OUTDIR}/oceanic_cellcount_by_depth_matrix.csv", index=False)

# Summary: Mean/Median/2.5%/97.5%/Std per bin
depth_summary = pd.DataFrame({
    "depth_top_km": DEPTH_BINS_KM[:-1],
    "depth_bot_km": DEPTH_BINS_KM[1:],
    "Mean":   by_depth_totals_by_draw.mean(axis=0),
    "Median": np.percentile(by_depth_totals_by_draw, 50.0, axis=0),
    "2.5%":   np.percentile(by_depth_totals_by_draw,  2.5, axis=0),
    "97.5%":  np.percentile(by_depth_totals_by_draw, 97.5, axis=0),
    "Std":    by_depth_totals_by_draw.std(axis=0, ddof=0)
})
depth_summary.to_csv(f"{OUTDIR}/oceanic_cellcount_by_depth_summary.csv", index=False)

# ---------------------------- Global totals (all fields) ----------------------------
total_list = []
for layer in summary_layers:
    vals = np.array([df[layer].sum() for df in all_samples])
    total_list.append({
        "Layer": layer,
        "Total Mean":   float(vals.mean()),
        "Total Median": float(np.percentile(vals, 50.0)),
        "Total Std":    float(vals.std(ddof=0)),
        "Total 2.5%":   float(np.percentile(vals,  2.5)),
        "Total 97.5%":  float(np.percentile(vals, 97.5))
    })
pd.DataFrame(total_list).to_csv(f"{OUTDIR}/oceanic_cell_totals_bootstrap_mc.csv", index=False)

# ---------------------------- Optional diagnostics across draws ----------------------------
dist_list = []
for layer in summary_layers:
    val_medians = np.array([df[layer].median()        for df in all_samples])
    val_means   = np.array([df[layer].mean()          for df in all_samples])
    val_std     = np.array([df[layer].std()           for df in all_samples])
    val_25p     = np.array([df[layer].quantile(0.025) for df in all_samples])
    val_975p    = np.array([df[layer].quantile(0.975) for df in all_samples])

    all_values  = np.concatenate([df[layer].values for df in all_samples])
    try:
        mode_result = mode(all_values, nan_policy="omit", keepdims=False)
    except TypeError:
        mode_result = mode(all_values, nan_policy="omit")
    most_freq       = np.atleast_1d(mode_result.mode)[0]  if np.size(mode_result.mode)  else np.nan
    most_freq_count = np.atleast_1d(mode_result.count)[0] if np.size(mode_result.count) else np.nan

    dist_list.append({
        "Layer": layer,
        "Mean of means":        float(val_means.mean()),
        "Median of medians":    float(val_medians.mean()),
        "Mean Std":             float(val_std.mean()),
        "Mean 2.5%":            float(val_25p.mean()),
        "Mean 97.5%":           float(val_975p.mean()),
        "Most Frequent":        float(most_freq),
        "Most Frequent Count":  float(most_freq_count)
    })

pd.DataFrame(dist_list).to_csv(f"{OUTDIR}/oceanic_cellcount_sampling_distributions.csv", index=False)

print(f"[INFO] Depth bins used -> {DEPTH_BINS_KM.tolist()}")
print(f"[INFO] Saved per-grid summary -> {OUTDIR}/oceanic_cell_counts_bootstrap_mc.csv")
print(f"[INFO] Saved layer x iter matrix -> {OUTDIR}/oceanic_cellcount_by_layer_matrix.csv")
print(f"[INFO] Saved by-layer summary -> {OUTDIR}/oceanic_cellcount_by_layer_summary.csv")
print(f"[INFO] Saved depth-bin matrix -> {OUTDIR}/oceanic_cellcount_by_depth_matrix.csv")
print(f"[INFO] Saved depth-bin summary -> {OUTDIR}/oceanic_cellcount_by_depth_summary.csv")
print(f"[INFO] Saved global totals -> {OUTDIR}/oceanic_cell_totals_bootstrap_mc.csv")
print(f"[INFO] Saved diagnostics -> {OUTDIR}/oceanic_cellcount_sampling_distributions.csv")
