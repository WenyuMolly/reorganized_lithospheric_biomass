# -*- coding: utf-8 -*-
"""
Unstratified power-law fit (pooled observations) + Monte Carlo estimation
with stratified (Upper Crust/Middle Crust/Lower Crust/Mantle) volumes
+ fixed depth bins (continental-style, auto-extended)
=========================================================================

Summary
-------
- Fit ONE global power-law in log10-space to all oceanic observations (pooled).
- During MC prediction, compute ECM1 layer volumes per grid, evaluate the SAME
  model on each layer, and integrate to cell counts.
- NEW: also accumulate global cell counts into fixed depth bins
  [0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] km, automatically extending the
  last edge up to max(maxdepth + 3*maxdepth_sd) if needed.

Outputs
-------
- results/unstratified_power_law_mc_without_shallow/domain_power_fit_global.csv
- results/unstratified_power_law_mc_without_shallow/domain_power_fit_global.json
- results/unstratified_power_law_mc_without_shallow/oceanic_cell_counts_power_mc.csv
  (per-grid summary, per-layer mean/std/quantiles)
- results/unstratified_power_law_mc_without_shallow/oceanic_cell_totals_power_mc.csv
  (global totals per layer & total, across MC draws)
- results/unstratified_power_law_mc_without_shallow/oceanic_cellcount_by_depth_matrix_unstrat_power.csv
  (global-by-depth matrix: rows=bins, cols=iter_0001..)
- results/unstratified_power_law_mc_without_shallow/oceanic_cellcount_by_depth_summary_unstrat_power.csv
  (global-by-depth summary with Mean/Median/2.5%/97.5%/Std)
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

# ---------------------- Config ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "runs/oceanic/unstratified_power_law_mc_by_depth_with_shallow"

RES_DEG = 1.0           # grid resolution (deg)
N_DRAWS = 1000          # Monte-Carlo draws
SEED    = 42

# Optionally exclude seafloor-exposed references (e.g., Santelli 2008, Jacobson Meyers 2014)
EXCLUDE_THESE = False
EXCLUDE_REFS = ["santelli", "jacobson", "meyers"]   # case-insensitive

# Lower bound for depth to avoid log10(0) and z^b singularities
Z_MIN_KM = 1e-3

# ECM stratification names (consistent with your volume calculation)
DOMAINS = ["Upper Crust", "Middle Crust", "Lower Crust", "Mantle"]

# Defensive caps to avoid numeric blow-up in MC
DENSITY_CLIP_MAX = 1e9       # cells/cm^3 (hard ceiling)
VOLUME_CM3_CLIP_MAX = 1e22   # cm^3 per cell per layer
CELLCOUNT_CLIP_MAX = 1e35    # cells per cell per layer
EXTREME_CELLCOUNT_WARN = 1e33

DEBUG_FIRST_DRAW_ONLY = True
DEBUG_FIRST_N_CELLS = 5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unstratified oceanic power-law biomass estimates.")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--exclude-shallow", action="store_true", help="Exclude shallow/seawater-contacted samples.")
    dataset_group.add_argument("--include-shallow", action="store_true", help="Include shallow/seawater-contacted samples.")
    parser.add_argument("--n-draws", type=int, default=N_DRAWS, help="Number of Monte Carlo draws.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    return parser.parse_args()

# ---------------------- Helpers ---------------------

def grid_area_cm2(lat_deg: float, res_deg: float = 1.0) -> float:
    """
    Return res_deg×res_deg grid area in cm^2 at latitude.
    Earth approx: 1 deg ~ 111.32 km. Area = (111.32 km * cos(lat)) * (111.32 km) * res_deg^2.
    """
    m_per_deg = 111_320.0
    dx = m_per_deg * res_deg * np.cos(np.radians(lat_deg))
    dy = m_per_deg * res_deg
    area_m2 = dx * dy
    return area_m2 * 1e4  # m^2 -> cm^2

def convert_cells_per_g_to_cm3(row: pd.Series) -> float:
    """
    Return cell density in cells/cm^3.
    - For Santelli et al. (2008): cells/g -> ×2.77 g/cm^3 (Lima et al., 2020).
    - For Jacobson Meyers et al. (2014): cells/g -> ×2.90 g/cm^3 (Moore, 2001).
    - Otherwise: 'Cell Count' assumed already in cells/cm^3.
    """
    ref = str(row.get("Reference", "")).lower()
    val = float(row.get("Cell Count", np.nan))
    if not np.isfinite(val):
        return np.nan
    if "santelli" in ref:
        return val * 2.77
    if ("meyers" in ref) or ("jacobson" in ref):
        return val * 2.90
    return val

def detect_depth_column_and_to_km(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Detect observation depth column: "Depth for Power Fit" or "Depth for the fit".
    Values are in meters → convert to km if median > 10.
    Returns: depth_km, unit_from ('m' or 'km').
    """
    candidates = ["Depth for Power Fit", "Depth for the fit"]
    col_found = None
    for c in candidates:
        if c in df.columns:
            col_found = c
            break
    if col_found is None:
        raise RuntimeError(f"Expected depth column not found; tried: {candidates}")

    depth_raw = pd.to_numeric(df[col_found], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(depth_raw)
    if np.isnan(med):
        raise RuntimeError("Depth column contains no valid numeric values.")
    if med > 10:  # likely meters
        return depth_raw / 1000.0, "m"
    else:
        return depth_raw.copy(), "km"

def is_constant_depth(depths: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Return True if all depths are numerically almost the same."""
    d = np.asarray(depths, dtype=float)
    d = d[np.isfinite(d)]
    if d.size <= 1:
        return True
    return np.allclose(d, d[0], rtol=rtol, atol=atol)

def safe_log10(x: np.ndarray, min_pos: float = 1e-12) -> np.ndarray:
    """Compute log10 with clipping at min_pos to avoid -inf."""
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, min_pos, None))

def geometric_mean(values: np.ndarray) -> float:
    """Geometric mean using log10, ignoring non-positive and non-finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(10.0 ** (np.log10(v).mean()))

def fit_power_law_log10(depth_km: np.ndarray,
                        dens_cm3: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
    """
    Fit log10(density) = a + b*log10(depth_km).
    Return a, b, cov (2x2), s_e (residual std in log10), r2 (on log10-space).
    """
    z = np.clip(depth_km.astype(float), Z_MIN_KM, None)
    y = dens_cm3.astype(float)
    lx = np.log10(z)
    ly = np.log10(y)

    p, cov_ba = np.polyfit(lx, ly, deg=1, cov=True)  # p[0]=b, p[1]=a
    b, a = float(p[0]), float(p[1])
    yhat = a + b*lx
    resid = ly - yhat
    s_e = float(np.sqrt((resid**2).sum() / max(len(ly)-2, 1)))
    ss_res = float((resid**2).sum())
    ss_tot = float(((ly - ly.mean())**2).sum())
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    cov_ab = cov_ba[::-1, ::-1]
    return a, b, cov_ab, s_e, r2

def fit_unstratified_power_or_constant(depth_km: np.ndarray,
                                       dens_cm3: np.ndarray) -> Dict[str, Any]:
    """
    Fit ONE global model for all pooled data. If depth is almost constant or too few points,
    fallback to constant model (A = GM of y).
    Returns a dictionary with unified keys for MC.
    """
    x = np.asarray(depth_km, dtype=float)
    y = np.asarray(dens_cm3, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    n = int(x.size)

    if n < 3 or is_constant_depth(x):
        A = geometric_mean(y)
        ly = safe_log10(y)
        s_e = float(np.std(ly, ddof=1)) if ly.size > 1 else 0.0
        return {
            "model": "constant",
            "n": n,
            "A": float(A),
            "a": float(np.log10(max(A, 1e-12))),  # unify interface
            "b": 0.0,
            "cov": None,
            "s_e": s_e,
            "r2_log10": np.nan
        }
    a, b, cov, s_e, r2 = fit_power_law_log10(x, y)
    return {
        "model": "power",
        "n": n,
        "a": float(a),
        "b": float(b),
        "cov": cov.tolist(),
        "s_e": float(s_e),
        "r2_log10": float(r2),
        "A": float(10**a)   # for reference
    }

def layer_mean_density_from_params_km(A: float, B: float,
                                      z_top_km: float, z_bot_km: float) -> float:
    """
    Average of y(z) = A * z^B over [z_top_km, z_bot_km] (z in km).
    Handles B = -1 by log integration; if Δz→0, return midpoint value.
    """
    z0 = max(float(z_top_km), Z_MIN_KM)
    z1 = max(float(z_bot_km), z0 + 1e-12)
    L = z1 - z0
    if abs(B + 1.0) < 1e-10:
        return float(A * np.log(z1/z0) / L)
    return float(A * (z1**(B+1) - z0**(B+1)) / ((B+1)*L))

# --- NEW: fixed depth bins utilities ----------------------------------

def add_powerlaw_segment_to_bins(A: float, B: float,
                                 seg_top_km: float, seg_bot_km: float,
                                 area_cm2: float,
                                 bins_km: np.ndarray,
                                 acc_vec: np.ndarray):
    """
    Accumulate cells from a vertical segment [seg_top_km, seg_bot_km] into fixed bins.
    For each overlap with bin [b0,b1], compute layer-mean density over that sub-interval
    via the (A,B) model, multiply by volume = (Δz[km]*1e5)*area_cm2, and add to acc_vec[j].
    """
    if not (np.isfinite(A) and np.isfinite(B) and np.isfinite(seg_top_km) and np.isfinite(seg_bot_km)):
        return
    if area_cm2 <= 0:
        return
    z0 = float(seg_top_km)
    z1 = float(seg_bot_km)
    if z1 <= z0:
        return
    per_km_vol_cm3 = area_cm2 * 1e5  # cm^3 per km-thickness
    for j in range(len(bins_km) - 1):
        b0, b1 = bins_km[j], bins_km[j+1]
        top = max(z0, b0)
        bot = min(z1, b1)
        dz = bot - top
        if dz > 0:
            y_avg = layer_mean_density_from_params_km(A, B, top, bot)
            if not np.isfinite(y_avg) or y_avg <= 0:
                continue
            acc_vec[j] += y_avg * per_km_vol_cm3 * dz

# ---------------------- Main -------------------------------------------

def main():
    global OUT_DIR, N_DRAWS, SEED, EXCLUDE_THESE

    args = parse_args()
    N_DRAWS = args.n_draws
    SEED = args.seed
    EXCLUDE_THESE = args.exclude_shallow
    OUT_DIR = args.output_dir or (
        PROJECT_ROOT / "runs/oceanic/unstratified_power_law_mc_by_depth_without_shallow"
        if EXCLUDE_THESE
        else PROJECT_ROOT / "runs/oceanic/unstratified_power_law_mc_by_depth_with_shallow"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load data
    inference_df = pd.read_csv(PROJECT_ROOT / "runs/volume/submitted/results/inference_and_depth_to_122.0_calculation_oceanic.csv")
    ecm_df = pd.read_csv(PROJECT_ROOT / "data/raw/oceanic/ecm/ECM1.txt", sep="\t", skiprows=0)
    cell_df = pd.read_excel(PROJECT_ROOT / "data/raw/oceanic/oceanic_cell_densities.xlsx")

    # 2) Prepare observation dataset
    cell_df["Cell Density (cm^3)"] = cell_df.apply(convert_cells_per_g_to_cm3, axis=1)
    if EXCLUDE_THESE:
        ref_str = cell_df["Reference"].astype(str).str.lower()
        mask_excl = ref_str.str.contains("|".join(EXCLUDE_REFS), na=False)
        excluded = cell_df.loc[mask_excl].copy()
        excluded.to_csv(os.path.join(OUT_DIR, "excluded_refs.csv"), index=False)
        cell_df = cell_df.loc[~mask_excl].copy()
        cell_df = cell_df[cell_df["Depth for Power Fit"] > 0.3]

    depth_km, unit_from = detect_depth_column_and_to_km(cell_df)
    cell_df["Depth_km_for_fit"] = depth_km
    print(f"[INFO] Observation depth column detected ({unit_from}) and converted to km.")

    # Keep required + depth_km_for_fit
    req_cols = ["Reference", "Rock Domain", "Depth for Power Fit", "Cell Density (cm^3)", "Depth_km_for_fit"]
    missing = [c for c in req_cols if c not in cell_df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in oceanic_cell_densities.xlsx: {missing}")

    cell_df = cell_df[req_cols].replace([np.inf, -np.inf], np.nan).dropna()
    cell_df = cell_df[(cell_df["Depth_km_for_fit"] > 0) & (cell_df["Cell Density (cm^3)"] > 0)]

    # 3) Fit ONE global (unstratified) model
    model_global = fit_unstratified_power_or_constant(
        cell_df["Depth_km_for_fit"].values.astype(float),
        cell_df["Cell Density (cm^3)"].values.astype(float)
    )
    # Empirical cap from all data (99.5% quantile)
    cap_val = float(np.nanquantile(cell_df["Cell Density (cm^3)"].values, 0.995))
    model_global["cap_value"] = cap_val

    # Save fitted model (CSV + JSON)
    pd.DataFrame([{
        "n_points": model_global["n"],
        "model": model_global["model"],
        "A (if constant)": model_global["A"] if model_global["model"]=="constant" else np.nan,
        "a (log10-intercept)": model_global["a"],
        "b (slope)": model_global["b"],
        "s_e_log10": model_global["s_e"],
        "r2_log10": model_global["r2_log10"],
        "obs_99.5%_cap_cells_cm3": cap_val
    }]).to_csv(os.path.join(OUT_DIR, "domain_power_fit_global.csv"), index=False)

    with open(os.path.join(OUT_DIR, "domain_power_fit_global.json"), "w", encoding="utf-8") as f:
        json.dump(model_global, f, indent=2)

    if model_global["model"] == "power":
        print(f"[OK] Global model: n={model_global['n']}, A≈{10**model_global['a']:.3e}, "
              f"B={model_global['b']:.3f}, s_e={model_global['s_e']:.3f}, cap≈{cap_val:.3e}")
    else:
        print(f"[OK] Global model: n={model_global['n']}, constant A={model_global['A']:.3e}, "
              f"s_e={model_global['s_e']:.3f}, cap≈{cap_val:.3e}")

    # 4) Clean ECM & merge to grids
    ecm_df.columns = [
        'Numb','Lon','Lat','Hcc','Sed','Hc','Type',
        'DLy1','DLy2','DLy3','TLy1','TLy2','TLy3',
        'Vp1','Vp2','Vp3','Vs1','Vs2','Vs3',
        'Vpn','Vsn','Rho1','Rho2','Rho3','Rhon'
    ]
    ecm_cleaned = ecm_df[['Lon','Lat','Sed','DLy1','DLy2','DLy3','TLy1','TLy2','TLy3']].copy()

    inference_df["lat_rounded"] = inference_df["lat"].round(1)
    inference_df["lon_rounded"] = inference_df["lon"].round(1)
    ecm_cleaned["lat_rounded"] = ecm_cleaned["Lat"].round(1)
    ecm_cleaned["lon_rounded"] = ecm_cleaned["Lon"].round(1)
    merged_df = pd.merge(inference_df, ecm_cleaned, on=["lat_rounded","lon_rounded"], how="left")

    # 4.1) Build depth bins (continental-style, auto-extended to data max)
    BASE_BINS_KM = np.array([0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    maxdepth_arr = pd.to_numeric(merged_df.get("maxdepth", np.nan), errors="coerce").values
    sd_series = merged_df.get("maxdepth_sd", 0.0)
    if hasattr(sd_series, "values"):
        sd_arr = pd.to_numeric(sd_series, errors="coerce").fillna(0.0).values
    else:
        sd_arr = np.zeros_like(maxdepth_arr)
    upper_cons = np.nanmax(maxdepth_arr + 3.0 * sd_arr) if maxdepth_arr.size else BASE_BINS_KM[-1]
    if not np.isfinite(upper_cons):
        upper_cons = np.nanmax(maxdepth_arr) if np.isfinite(np.nanmax(maxdepth_arr)) else BASE_BINS_KM[-1]
    if upper_cons > BASE_BINS_KM[-1] + 1e-9:
        DEPTH_BINS_KM = np.concatenate([BASE_BINS_KM, [float(upper_cons)]])
    else:
        DEPTH_BINS_KM = BASE_BINS_KM.copy()
    NBINS = len(DEPTH_BINS_KM) - 1
    print("[INFO] Depth bins (km):", DEPTH_BINS_KM.tolist())

    # 5) Monte Carlo (stratified by ECM layers, but using the SAME global model)
    rng = np.random.default_rng(SEED)
    all_samples = []
    did_debug_print_for_draw0 = False

    # NEW: store global-by-depth totals per draw
    by_depth_all = np.zeros((N_DRAWS, NBINS), dtype=float)

    for draw_idx in range(N_DRAWS):
        records = []
        by_depth_vec = np.zeros(NBINS, dtype=float)

        for cell_idx, (_, row) in enumerate(merged_df.iterrows()):
            lat, lon = float(row["lat"]), float(row["lon"])
            maxdepth = float(row["maxdepth"])
            maxdepth_sd = float(row.get("maxdepth_sd", 0.0) or 0.0)

            # Depth perturbation
            if maxdepth_sd > 0:
                perturbed = rng.normal(loc=maxdepth, scale=maxdepth_sd)
                perturbed_depth = max(perturbed, 0.0) if np.isfinite(perturbed) else maxdepth
                if perturbed_depth <= 0:
                    perturbed_depth = maxdepth
            else:
                perturbed_depth = maxdepth

            # Basement coordinates (subtract Sed)
            sed = float(row.get("Sed", 0.0))
            z_b = max(perturbed_depth, 0.0)

            # Basement layer boundaries
            d1c = max(float(row["DLy1"]) - sed, 0.0)
            d2c = max(float(row["DLy2"]) - sed, 0.0)
            d3c = max(float(row["DLy3"]) - sed, 0.0)

            t1, t2, t3 = float(row["TLy1"]), float(row["TLy2"]), float(row["TLy3"])

            # Layer thickness in basement km
            u = m = l = mn = 0.0
            if z_b <= 0.0:
                pass
            elif z_b <= d1c:
                u = z_b
            elif z_b <= d2c:
                u = t1
                m = z_b - d1c
            elif z_b <= d3c:
                u = t1
                m = t2
                l = z_b - d2c
            else:
                u, m, l = t1, t2, t3
                mn = max(0.0, z_b - d3c)

            # Horizontal area and layer volumes (cm^3) (still used for per-grid layer outputs)
            area_cm2 = grid_area_cm2(lat, res_deg=RES_DEG)
            km_to_cm = 1e5
            uc_vol = min(u  * km_to_cm * area_cm2, VOLUME_CM3_CLIP_MAX)
            mc_vol = min(m  * km_to_cm * area_cm2, VOLUME_CM3_CLIP_MAX)
            lc_vol = min(l  * km_to_cm * area_cm2, VOLUME_CM3_CLIP_MAX)
            mn_vol = min(mn * km_to_cm * area_cm2, VOLUME_CM3_CLIP_MAX)

            # Sample coefficients/residual from global model
            s_e = float(model_global["s_e"])
            cap_here = float(model_global.get("cap_value", np.inf))

            if model_global["model"] == "power":
                a_mu = float(model_global["a"])
                b_mu = float(model_global["b"])
                cov = np.array(model_global["cov"], dtype=float)
                ab = rng.multivariate_normal(mean=[a_mu, b_mu], cov=cov, check_valid="ignore")
                a_s, b_s = float(ab[0]), float(ab[1])
                eps = rng.normal(loc=0.0, scale=s_e)
                A = (10.0 ** a_s) * (10.0 ** eps)
                B = b_s
            else:
                # constant model
                A0 = float(model_global["A"])
                eps = rng.normal(loc=0.0, scale=s_e)
                A = A0 * (10.0 ** eps)
                B = 0.0

            # For each ECM layer, compute average density over the layer and multiply by layer volume
            counts = {}
            for dom, thick, top, v_here in [
                ("Upper Crust", u,  0.0, uc_vol),
                ("Middle Crust", m, d1c, mc_vol),
                ("Lower Crust", l, d2c, lc_vol),
                ("Mantle",      mn, d3c, mn_vol),
            ]:
                if thick <= 0.0:
                    counts[dom] = 0.0
                    continue
                z0_km = float(top)
                z1_km = z0_km + float(thick)
                y_avg = layer_mean_density_from_params_km(A, B, z0_km, z1_km)

                # defensive clipping
                if not np.isfinite(y_avg) or y_avg < 0.0:
                    y_avg = 0.0
                if y_avg > cap_here:
                    y_avg = cap_here
                if y_avg > DENSITY_CLIP_MAX:
                    y_avg = DENSITY_CLIP_MAX

                cells = y_avg * v_here
                if not np.isfinite(cells) or cells < 0:
                    cells = 0.0
                if cells > CELLCOUNT_CLIP_MAX:
                    cells = CELLCOUNT_CLIP_MAX

                counts[dom] = cells

                # --- NEW: accumulate into global depth bins by subdividing the layer
                add_powerlaw_segment_to_bins(A, B, z0_km, z1_km, area_cm2, DEPTH_BINS_KM, by_depth_vec)

            total_cells = counts["Upper Crust"] + counts["Middle Crust"] + counts["Lower Crust"] + counts["Mantle"]

            # Optional debug print
            if DEBUG_FIRST_DRAW_ONLY and (draw_idx == 0) and (cell_idx < DEBUG_FIRST_N_CELLS) and (not did_debug_print_for_draw0):
                print("\n[DEBUG] Draw0 Cell#{} @ (lat={:.2f}, lon={:.2f})".format(cell_idx, lat, lon))
                print("        area_cm2 = {:.3e}".format(area_cm2))
                print("        maxdepth={:.3f} km, maxdepth_sd={:.3f} km, Sed={:.3f} km, z_b={:.3f} km".format(
                    maxdepth, maxdepth_sd, sed, z_b))
                print("        d1c={:.3f}, d2c={:.3f}, d3c={:.3f}".format(d1c, d2c, d3c))
                print("        thicknesses: U={:.3f}, M={:.3f}, L={:.3f}, Mn={:.3f}".format(u, m, l, mn))
                print("        [Global model] a={:.3f}, b={:.3f}, s_e={:.3f}, cap≈{:.3e}".format(
                    model_global["a"], model_global["b"], model_global["s_e"], cap_here))
                print("        cells (UC,MC,LC,Mn): {:.3e}, {:.3e}, {:.3e}, {:.3e}, TOTAL={:.3e}".format(
                    counts["Upper Crust"], counts["Middle Crust"], counts["Lower Crust"], counts["Mantle"], total_cells))
                did_debug_print_for_draw0 = True

            if total_cells > EXTREME_CELLCOUNT_WARN:
                print("[WARN] Extreme cellcount at (lat={:.2f}, lon={:.2f}): TOTAL={:.3e}".format(lat, lon, total_cells))

            records.append({
                "lat": lat, "lon": lon,
                "Perturbed Depth": perturbed_depth, "Max Depth": maxdepth,
                "Sediment": sed,
                "Basement Depth z_b": z_b,
                "Upper Crust Depth": u, "Middle Crust Depth": m,
                "Lower Crust Depth": l, "Mantle Depth": mn,
                "Upper Crust": counts["Upper Crust"],
                "Middle Crust": counts["Middle Crust"],
                "Lower Crust": counts["Lower Crust"],
                "Mantle": counts["Mantle"],
                "Total in Each Cell": total_cells
            })

        all_samples.append(pd.DataFrame(records))
        by_depth_all[draw_idx, :] = by_depth_vec

    # 6) Per-grid summary (mean/std/quantiles) for layers and total
    summary_layers = [
        "Perturbed Depth", "Sediment", "Basement Depth z_b",
        "Upper Crust Depth", "Middle Crust Depth", "Lower Crust Depth", "Mantle Depth",
        "Upper Crust", "Middle Crust", "Lower Crust", "Mantle",
        "Total in Each Cell"
    ]
    summary_df = all_samples[0][["lat", "lon"]].copy()
    for layer in summary_layers:
        stack = np.stack([df[layer].values for df in all_samples])  # [n_draws, n_cells]
        summary_df[layer + " Mean"]   = stack.mean(axis=0)
        summary_df[layer + " Std"]    = stack.std(axis=0, ddof=1)
        summary_df[layer + " 2.5%"]   = np.percentile(stack, 2.5, axis=0)
        summary_df[layer + " 50%"]    = np.percentile(stack, 50.0, axis=0)
        summary_df[layer + " 97.5%"]  = np.percentile(stack, 97.5, axis=0)

    out_grid = os.path.join(OUT_DIR, "oceanic_cell_counts_power_mc.csv")
    summary_df.to_csv(out_grid, index=False)
    print(f"[OK] Grid summary saved: {out_grid}")

    # 7) Global totals across draws (per layer and total)
    total_rows = []
    for layer in ["Upper Crust","Middle Crust","Lower Crust","Mantle","Total in Each Cell"]:
        vals = np.array([df[layer].sum() for df in all_samples], dtype=float)
        total_rows.append({
            "Layer": layer,
            "Total Mean": float(vals.mean()),
            "Total Median": float(np.percentile(vals, 50.0)),
            "Total Std": float(vals.std(ddof=1)),
            "Total 2.5%": float(np.percentile(vals, 2.5)),
            "Total 97.5%": float(np.percentile(vals, 97.5)),
            "N_draws": int(len(vals))
        })

    out_tot = os.path.join(OUT_DIR, "oceanic_cell_totals_power_mc.csv")
    pd.DataFrame(total_rows).to_csv(out_tot, index=False)
    print(f"[OK] Global totals saved: {out_tot}")

    # 8) NEW: Global by-depth outputs (matrix + summary)
    by_depth_matrix = pd.DataFrame({
        "depth_top_km": DEPTH_BINS_KM[:-1],
        "depth_bot_km": DEPTH_BINS_KM[1:]
    })
    for k in range(N_DRAWS):
        by_depth_matrix[f"iter_{k+1:04d}"] = by_depth_all[k, :]
    by_depth_matrix.to_csv(
        os.path.join(OUT_DIR, "oceanic_cellcount_by_depth_matrix_unstrat_power.csv"),
        index=False
    )

    bin_summary_rows = []
    for j in range(len(DEPTH_BINS_KM) - 1):
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
        os.path.join(OUT_DIR, "oceanic_cellcount_by_depth_summary_unstrat_power.csv"),
        index=False
    )

    print("[OK] By-depth matrix & summary saved.")

if __name__ == "__main__":
    main()
