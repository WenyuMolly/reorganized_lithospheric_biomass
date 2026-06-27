# -*- coding: utf-8 -*-
"""
Power-law (depth-controlled) oceanic cell-count estimation with Monte Carlo
+ depth-bin outputs auto-extended to cover max zmax (≈ maxdepth + 3σ)
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List

# ---------------------- Configuration ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "runs/oceanic/stratified_power_law_mc_by_depth_with_shallow"

RES_DEG = 1.0          # grid area resolution in degrees
N_DRAWS = 1000         # Monte Carlo iterations
SEED = 42

EXCLUDE_THESE = False
EXCLUDE_REFS = ["santelli", "jacobson", "meyers"]  # lowercased match

Z_MIN_KM = 1e-3        # avoid log singularities at z=0

DOMAINS = ["Upper Crust", "Middle Crust", "Lower Crust", "Mantle"]

# Depth bins template (km). Will auto-extend the top edge if needed.
BASE_BINS_KM = [0, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Sanity/clip guards (optional)
DENSITY_CLIP_MAX = 1e9
VOLUME_CM3_CLIP_MAX = 1e22
CELLCOUNT_CLIP_MAX = 1e35
EXTREME_CELLCOUNT_WARN = 1e33

DEBUG_FIRST_DRAW_ONLY = False
DEBUG_FIRST_N_CELLS = 5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified oceanic power-law biomass estimates.")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--exclude-shallow", action="store_true", help="Exclude shallow/seawater-contacted samples.")
    dataset_group.add_argument("--include-shallow", action="store_true", help="Include shallow/seawater-contacted samples.")
    parser.add_argument("--n-draws", type=int, default=N_DRAWS, help="Number of Monte Carlo draws.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--z122-scenario",
        choices=["mc", "low", "base", "high"],
        default="mc",
        help="Depth treatment: mc preserves original maxdepth/maxdepth_sd sampling; low/base/high use maxdepth-sd, maxdepth, or maxdepth+sd.",
    )
    return parser.parse_args()

# ---------------------- Helpers ---------------------------------------

def grid_area_cm2(lat_deg: float, res_deg: float = 1.0) -> float:
    """Approximate 1°×1° cell area in m² then convert to cm²."""
    m_per_deg = 111_320.0
    dx = m_per_deg * res_deg * np.cos(np.radians(lat_deg))
    dy = m_per_deg * res_deg
    area_m2 = dx * dy
    return area_m2 * 1e4  # m² -> cm²

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

def convert_cells_per_g_to_cm3(row: pd.Series) -> float:
    """Convert cells/g to cells/cm³ if needed; otherwise pass-through."""
    ref = str(row.get("Reference", "")).lower()
    val = float(row["Cell Count"])
    if "santelli" in ref:
        return val * 2.77
    elif "meyers" in ref or "jacobson" in ref:
        return val * 2.90
    else:
        return val

def detect_depth_column_and_to_km(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Detect oceanic observation depth column and convert meters->km if needed.
    Accepts 'Depth for Power Fit' or 'Depth for the fit'.
    """
    candidates = ["Depth for Power Fit", "Depth for the fit"]
    found = next((c for c in candidates if c in df.columns), None)
    if not found:
        raise RuntimeError(f"Expected depth column not found; tried: {candidates}")

    depth = pd.to_numeric(df[found], errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(depth)
    if np.isnan(med):
        raise RuntimeError("Depth column contains no valid numeric values.")
    if med > 10:  # very likely meters
        return depth / 1000.0, "m"
    return depth.copy(), "km"

def is_constant_depth(depths: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    d = np.asarray(depths, dtype=float)
    d = d[np.isfinite(d)]
    if d.size <= 1:
        return True
    return np.allclose(d, d[0], rtol=rtol, atol=atol)

def safe_log10(x, min_pos=1e-12):
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, min_pos, None))

def geometric_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return np.nan
    return float(10.0 ** (np.log10(v).mean()))

def fit_power_law_log10(depth: np.ndarray, dens_cm3: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
    """Fit log10(y) = a + b*log10(z). Return a, b, cov(2×2), s_e (log10), r2."""
    z = np.clip(depth.astype(float), Z_MIN_KM, None)
    y = dens_cm3.astype(float)
    lx = np.log10(z)
    ly = np.log10(y)
    p, cov_ba = np.polyfit(lx, ly, deg=1, cov=True)  # p[0]=b, p[1]=a
    b, a = float(p[0]), float(p[1])
    yhat = a + b * lx
    resid = ly - yhat
    s_e = float(np.sqrt((resid ** 2).sum() / max(len(ly) - 2, 1)))
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    cov_ab = cov_ba[::-1, ::-1]
    return a, b, cov_ab, s_e, r2

def fit_power_or_constant(depth: np.ndarray,
                          dens_cm3: np.ndarray,
                          domain_name: str) -> Dict[str, Any]:
    """
    Fit power law if feasible; otherwise fallback to constant (A=GM(y), b=0).
    Returns a dict with fields for MC sampling.
    """
    x = np.asarray(depth, dtype=float)
    y = np.asarray(dens_cm3, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[m]; y = y[m]
    n = x.size

    if n < 3 or is_constant_depth(x):
        A = geometric_mean(y)
        ly = safe_log10(y)
        s_e = float(np.std(ly, ddof=1)) if ly.size > 1 else 0.0
        return {
            "model": "constant",
            "domain": domain_name,
            "n": int(n),
            "A": float(A),
            "a": float(np.log10(max(A, 1e-12))),
            "b": 0.0,
            "cov": None,
            "s_e": s_e,
            "r2_log10": np.nan
        }

    a, b, cov, s_e, r2 = fit_power_law_log10(x, y)
    return {
        "model": "power",
        "domain": domain_name,
        "n": int(n),
        "a": float(a),
        "b": float(b),
        "cov": cov.tolist(),
        "s_e": float(s_e),
        "r2_log10": float(r2),
        "A": float(10.0 ** a)
    }

def layer_mean_density_from_params_km(A: float, B: float,
                                      z_top_km: float, z_bot_km: float) -> float:
    """
    Average of y(z)=A*z^B over [z_top_km, z_bot_km] (z in km).
    """
    z0 = max(float(z_top_km), Z_MIN_KM)
    z1 = max(float(z_bot_km), z0 + 1e-12)
    L = z1 - z0
    if abs(B + 1.0) < 1e-10:
        return float(A * np.log(z1 / z0) / L)
    return float(A * (z1 ** (B + 1.0) - z0 ** (B + 1.0)) / ((B + 1.0) * L))

def segment_cells_from_params(A: float, B: float,
                              z0_km: float, z1_km: float,
                              area_cm2: float,
                              cap_density: float) -> float:
    """
    Compute cells for a depth segment [z0, z1] using averaged density and volume.
    """
    if z1_km <= z0_km:
        return 0.0
    y_avg = layer_mean_density_from_params_km(A, B, z0_km, z1_km)
    if not np.isfinite(y_avg) or y_avg < 0:
        y_avg = 0.0
    y_avg = min(y_avg, cap_density, DENSITY_CLIP_MAX)
    km_to_cm = 1e5
    vol_cm3 = (z1_km - z0_km) * km_to_cm * area_cm2
    # vol_cm3 = min(vol_cm3, VOLUME_CM3_CLIP_MAX)  # (optional) clamp
    cells = y_avg * vol_cm3
    if not np.isfinite(cells) or cells < 0:
        cells = 0.0
    return min(cells, CELLCOUNT_CLIP_MAX)

# ---------------------- Main pipeline ----------------------------------

def main():
    global OUT_DIR, N_DRAWS, SEED, EXCLUDE_THESE

    args = parse_args()
    N_DRAWS = args.n_draws
    SEED = args.seed
    EXCLUDE_THESE = args.exclude_shallow
    OUT_DIR = args.output_dir or (
        PROJECT_ROOT / "runs/oceanic/stratified_power_law_mc_by_depth_without_shallow"
        if EXCLUDE_THESE
        else PROJECT_ROOT / "runs/oceanic/stratified_power_law_mc_by_depth_with_shallow"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Load grids and observations
    inference_df = pd.read_csv(PROJECT_ROOT / "runs/volume/submitted/results/inference_and_depth_to_122.0_calculation_oceanic.csv")
    ecm_df = pd.read_csv(PROJECT_ROOT / "data/raw/oceanic/ecm/ECM1.txt", sep="\t", skiprows=0)
    cell_df = pd.read_excel(PROJECT_ROOT / "data/raw/oceanic/oceanic_cell_densities.xlsx")

    # 2) Prepare cell dataset
    cell_df["Cell Density (cm^3)"] = cell_df.apply(convert_cells_per_g_to_cm3, axis=1)
    if EXCLUDE_THESE:
        ref_str = cell_df["Reference"].astype(str).str.lower()
        mask_excl = ref_str.str.contains("|".join(EXCLUDE_REFS), na=False)
        kept = cell_df.loc[~mask_excl].copy()
        kept = kept[kept.get("Depth for Power Fit", np.inf) > 0.3]  # remove very shallow samples
        kept.to_csv(os.path.join(OUT_DIR, "included_data.csv"), index=False)
        cell_df = kept

    depth_km, unit_from = detect_depth_column_and_to_km(cell_df)
    cell_df["Depth_km_for_fit"] = depth_km

    # Keep required columns and clean
    req_cols = ["Reference", "Rock Domain", "Depth_km_for_fit", "Cell Density (cm^3)"]
    if missing := [c for c in req_cols if c not in cell_df.columns]:
        raise RuntimeError(f"Missing columns in oceanic_cell_densities.xlsx: {missing}")

    cell_df = cell_df[req_cols].replace([np.inf, -np.inf], np.nan).dropna()
    cell_df = cell_df[(cell_df["Depth_km_for_fit"] > 0) & (cell_df["Cell Density (cm^3)"] > 0)]

    # 3) Fit power-law per domain
    fit_rows = []
    domain_models: Dict[str, Dict[str, Any]] = {}
    for dom in DOMAINS:
        sub = cell_df.loc[cell_df["Rock Domain"] == dom].copy()
        if sub.empty:
            print(f"[WARN] {dom}: no data; skip.")
            continue
        model = fit_power_or_constant(
            sub["Depth_km_for_fit"].to_numpy(float),
            sub["Cell Density (cm^3)"].to_numpy(float),
            domain_name=dom
        )
        cap_val = float(np.nanquantile(sub["Cell Density (cm^3)"].to_numpy(float), 0.995))
        model["cap_value"] = cap_val
        domain_models[dom] = model

        fit_rows.append({
            "Domain": dom,
            "n_points": model["n"],
            "model": model["model"],
            "A (if constant)": model["A"] if model["model"] == "constant" else np.nan,
            "a (log10-intercept)": model["a"],
            "b (slope)": model["b"],
            "s_e_log10": model["s_e"],
            "r2_log10": model["r2_log10"],
            "obs_99.5%_cap_cells_cm3": cap_val
        })

        if model["model"] == "power":
            print(f"[OK] {dom}: n={model['n']}, A≈{10**model['a']:.3e}, B={model['b']:.3f}, s_e={model['s_e']:.3f}")
        else:
            print(f"[OK] {dom}: n={model['n']}, constant A={model['A']:.3e}, s_e={model['s_e']:.3f}")

    if not domain_models:
        raise RuntimeError("No domain had a successful fit; cannot proceed.")

    pd.DataFrame(fit_rows).to_csv(os.path.join(OUT_DIR, "domain_power_fits.csv"), index=False)
    with open(os.path.join(OUT_DIR, "domain_power_fits.json"), "w", encoding="utf-8") as f:
        json.dump(domain_models, f, indent=2)

    # 4) Clean ECM and merge to grids
    ecm_df.columns = [
        'Numb','Lon','Lat','Hcc','Sed','Hc','Type',
        'DLy1','DLy2','DLy3','TLy1','TLy2','TLy3',
        'Vp1','Vp2','Vp3','Vs1','Vs2','Vs3',
        'Vpn','Vsn','Rho1','Rho2','Rho3','Rhon'
    ]
    ecm_cleaned = ecm_df[['Lon','Lat','Sed','DLy1','DLy2','DLy3','TLy1','TLy2','TLy3']].copy()

    inference_df['lat_rounded'] = inference_df['lat'].round(1)
    inference_df['lon_rounded'] = inference_df['lon'].round(1)
    ecm_cleaned['lat_rounded'] = ecm_cleaned['Lat'].round(1)
    ecm_cleaned['lon_rounded'] = ecm_cleaned['Lon'].round(1)
    merged_df = pd.merge(inference_df, ecm_cleaned, on=['lat_rounded','lon_rounded'], how='left')

    # 5) Auto-extend depth bins to the conservative upper bound of zmax
    maxdepth = merged_df.get("maxdepth", pd.Series(dtype=float)).to_numpy(dtype=float)
    maxdepth_sd = merged_df.get("maxdepth_sd", pd.Series(0.0, index=merged_df.index)).to_numpy(dtype=float)
    if maxdepth.size:
        upper_conservative = np.nanmax(maxdepth + 3.0 * np.clip(maxdepth_sd, 0.0, None))
    else:
        upper_conservative = BASE_BINS_KM[-1]
    if not np.isfinite(upper_conservative):
        upper_conservative = BASE_BINS_KM[-1]

    bins_km: List[float] = list(BASE_BINS_KM)
    if upper_conservative > bins_km[-1] + 1e-12:
        # round up to nearest 0.1 km
        bins_km.append(float(np.ceil(upper_conservative * 10.0) / 10.0))
    bins_km = sorted(set(bins_km))
    n_bins = len(bins_km) - 1
    print(f"[INFO] Depth bins (km): {bins_km}  (n_bins={n_bins})")

    # 6) Monte Carlo
    rng = np.random.default_rng(SEED)
    all_samples = []                 # per-draw per-grid records (for legacy outputs)
    totals_by_depth = np.zeros((N_DRAWS, n_bins), dtype=float)  # per-draw totals by depth bin

    did_debug_print_for_draw0 = False

    for draw_idx in range(N_DRAWS):
        records = []
        bin_totals_this_draw = np.zeros(n_bins, dtype=float)

        for cell_idx, (_, row) in enumerate(merged_df.iterrows()):
            lat, lon = float(row['lat']), float(row['lon'])
            maxdepth_km = float(row['maxdepth'])
            z_b = draw_z122_depth_km(row, rng, args.z122_scenario)
            z_b = max(z_b, 0.0)

            # Basement coordinates (subtract sediment)
            sed = float(row.get('Sed', 0.0))

            d1c = max(float(row['DLy1']) - sed, 0.0)
            d2c = max(float(row['DLy2']) - sed, 0.0)
            d3c = max(float(row['DLy3']) - sed, 0.0)

            t1 = float(row['TLy1']); t2 = float(row['TLy2']); t3 = float(row['TLy3'])

            # Layer thicknesses inside z_b
            u = m = l = mn = 0.0
            if z_b <= 0.0:
                pass
            elif z_b <= d1c:
                u = z_b
            elif z_b <= d2c:
                u = t1; m = z_b - d1c
            elif z_b <= d3c:
                u = t1; m = t2; l = z_b - d2c
            else:
                u, m, l = t1, t2, t3
                mn = max(0.0, z_b - d3c)

            area_cm2 = grid_area_cm2(lat, res_deg=RES_DEG)
            km_to_cm = 1e5

            # Sample per-domain parameters ONCE per cell (reuse for bins and for layer totals)
            sampled_params: Dict[str, Tuple[float, float, float]] = {}
            for dom in DOMAINS:
                if dom not in domain_models:
                    continue
                mod = domain_models[dom]
                s_e = float(mod["s_e"])
                cap_here = float(mod.get("cap_value", np.inf))

                if mod["model"] == "power":
                    a_mu = float(mod["a"]); b_mu = float(mod["b"])
                    cov = np.array(mod["cov"], dtype=float)
                    ab = rng.multivariate_normal(mean=[a_mu, b_mu], cov=cov, check_valid="ignore")
                    a_s, b_s = float(ab[0]), float(ab[1])
                    eps = rng.normal(loc=0.0, scale=s_e)  # log10 residual
                    A = (10.0 ** a_s) * (10.0 ** eps)
                    B = b_s
                else:
                    A0 = float(mod["A"])
                    eps = rng.normal(loc=0.0, scale=s_e)
                    A = A0 * (10.0 ** eps)
                    B = 0.0
                sampled_params[dom] = (A, B, cap_here)

            # Domain-wise totals (legacy outputs)
            uc_vol = u  * km_to_cm * area_cm2
            mc_vol = m  * km_to_cm * area_cm2
            lc_vol = l  * km_to_cm * area_cm2
            mn_vol = mn * km_to_cm * area_cm2

            counts = {"Upper Crust": 0.0, "Middle Crust": 0.0, "Lower Crust": 0.0, "Mantle": 0.0}
            if u > 0 and "Upper Crust" in sampled_params:
                A,B,cap = sampled_params["Upper Crust"]
                y_avg = layer_mean_density_from_params_km(A, B, 0.0, u)
                y_avg = min(max(y_avg, 0.0), domain_models["Upper Crust"]["cap_value"], DENSITY_CLIP_MAX)
                counts["Upper Crust"] = min(y_avg * uc_vol, CELLCOUNT_CLIP_MAX)

            if m > 0 and "Middle Crust" in sampled_params:
                A,B,cap = sampled_params["Middle Crust"]
                y_avg = layer_mean_density_from_params_km(A, B, d1c, d1c + m)
                y_avg = min(max(y_avg, 0.0), domain_models["Middle Crust"]["cap_value"], DENSITY_CLIP_MAX)
                counts["Middle Crust"] = min(y_avg * mc_vol, CELLCOUNT_CLIP_MAX)

            if l > 0 and "Lower Crust" in sampled_params:
                A,B,cap = sampled_params["Lower Crust"]
                y_avg = layer_mean_density_from_params_km(A, B, d2c, d2c + l)
                y_avg = min(max(y_avg, 0.0), domain_models["Lower Crust"]["cap_value"], DENSITY_CLIP_MAX)
                counts["Lower Crust"] = min(y_avg * lc_vol, CELLCOUNT_CLIP_MAX)

            if mn > 0 and "Mantle" in sampled_params:
                A,B,cap = sampled_params["Mantle"]
                y_avg = layer_mean_density_from_params_km(A, B, d3c, d3c + mn)
                y_avg = min(max(y_avg, 0.0), domain_models["Mantle"]["cap_value"], DENSITY_CLIP_MAX)
                counts["Mantle"] = min(y_avg * mn_vol, CELLCOUNT_CLIP_MAX)

            total_cells = sum(counts.values())

            # ---- Depth-bin accumulation (split by domain boundaries inside each bin) ----
            # domain boundaries in basement coordinates
            boundaries = [0.0, d1c, d2c, d3c]  # UC | MC | LC | Mn
            for b in range(n_bins):
                b_top = bins_km[b]
                b_bot = bins_km[b+1]
                if b_top >= z_b:  # no more contribution beyond basement depth
                    break
                seg0 = max(b_top, 0.0)
                seg1 = min(b_bot, z_b)
                if seg1 <= seg0:
                    continue

                # Cut this [seg0, seg1] by domain boundaries
                cut_points = [seg0, seg1] + [x for x in boundaries if seg0 < x < seg1]
                cut_points = sorted(set([float(x) for x in cut_points]))
                bin_sum = 0.0

                for s0, s1 in zip(cut_points[:-1], cut_points[1:]):
                    if s1 <= s0:
                        continue
                    # Decide which domain this sub-segment belongs to
                    if s0 >= d3c:
                        dom = "Mantle"
                    elif s0 >= d2c:
                        dom = "Lower Crust"
                    elif s0 >= d1c:
                        dom = "Middle Crust"
                    else:
                        dom = "Upper Crust"

                    if dom not in sampled_params:
                        continue
                    A,B,cap = sampled_params[dom]
                    bin_sum += segment_cells_from_params(A, B, s0, s1, area_cm2, cap)

                bin_totals_this_draw[b] += bin_sum

            # ---- Optional debug print ----
            if DEBUG_FIRST_DRAW_ONLY and (draw_idx == 0) and (cell_idx < DEBUG_FIRST_N_CELLS) and (not did_debug_print_for_draw0):
                print("\n[DEBUG] Draw0 Cell#{} @ (lat={:.2f}, lon={:.2f})".format(cell_idx, lat, lon))
                print("        z_b={:.3f} km; d1c={:.3f}, d2c={:.3f}, d3c={:.3f}".format(z_b, d1c, d2c, d3c))
                print("        thicknesses: U={:.3f}, M={:.3f}, L={:.3f}, Mn={:.3f}".format(u, m, l, mn))
                did_debug_print_for_draw0 = True

            if total_cells > EXTREME_CELLCOUNT_WARN:
                print("[WARN] Extreme cellcount at (lat={:.2f}, lon={:.2f}): TOTAL={:.3e}".format(lat, lon, total_cells))

            # Keep legacy per-grid record (domain totals)
            records.append({
                "lat": lat, "lon": lon,
                "Perturbed Depth": z_b, "Max Depth": maxdepth_km,
                "Sediment": sed,
                "Upper Crust Depth": u, "Middle Crust Depth": m,
                "Lower Crust Depth": l, "Mantle Depth": mn,
                "Upper Crust Cell Counts": counts["Upper Crust"],
                "Middle Crust Cell Counts": counts["Middle Crust"],
                "Lower Crust Cell Counts": counts["Lower Crust"],
                "Mantle Cell Counts": counts["Mantle"],
                "Total in Each Cell": total_cells
            })

        # end loop over cells
        all_samples.append(pd.DataFrame(records))
        totals_by_depth[draw_idx, :] = bin_totals_this_draw

    # 7) Per-grid summary (legacy)
    summary_layers = [
        "Perturbed Depth", "Sediment",
        "Upper Crust Depth", "Middle Crust Depth", "Lower Crust Depth", "Mantle Depth",
        "Upper Crust Cell Counts", "Middle Crust Cell Counts", "Lower Crust Cell Counts", "Mantle Cell Counts",
        "Total in Each Cell"
    ]
    summary_df = all_samples[0][["lat","lon"]].copy()
    for layer in summary_layers:
        stack = np.stack([df[layer].values for df in all_samples])  # [N_DRAWS, N_CELLS]
        summary_df[f"{layer} Mean"]   = stack.mean(axis=0)
        summary_df[f"{layer} Std"]    = stack.std(axis=0, ddof=1)
        summary_df[f"{layer} 2.5%"]   = np.percentile(stack, 2.5, axis=0)
        summary_df[f"{layer} 50%"]    = np.percentile(stack, 50.0, axis=0)
        summary_df[f"{layer} 97.5%"]  = np.percentile(stack, 97.5, axis=0)

    out_grid = os.path.join(OUT_DIR, "oceanic_cell_counts_power_mc.csv")
    summary_df.to_csv(out_grid, index=False)
    print(f"[OK] Grid summary saved: {out_grid}")

    # 8) Global totals across draws (legacy domain totals)
    total_rows = []
    for layer in ["Upper Crust Cell Counts","Middle Crust Cell Counts","Lower Crust Cell Counts","Mantle Cell Counts","Total in Each Cell"]:
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

    # 9) NEW: by-depth matrix (bins × draws) and summary
    # Matrix: rows=bins, cols=iter_0001..; plus depth_top_km/depth_bot_km
    by_depth_matrix = pd.DataFrame({
        "depth_top_km": bins_km[:-1],
        "depth_bot_km": bins_km[1:]
    })
    for k in range(N_DRAWS):
        by_depth_matrix[f"iter_{k+1:04d}"] = totals_by_depth[k, :]  # each column = one draw, per bin
    out_bydepth_mat = os.path.join(OUT_DIR, "oceanic_cellcount_by_depth_matrix.csv")
    by_depth_matrix.to_csv(out_bydepth_mat, index=False)

    # Summary across draws, per bin
    by_depth_mean   = totals_by_depth.mean(axis=0)
    by_depth_median = np.percentile(totals_by_depth, 50.0, axis=0)
    by_depth_lo     = np.percentile(totals_by_depth, 2.5, axis=0)
    by_depth_hi     = np.percentile(totals_by_depth, 97.5, axis=0)
    by_depth_std    = totals_by_depth.std(axis=0, ddof=1)

    by_depth_summary = pd.DataFrame({
        "depth_top_km": bins_km[:-1],
        "depth_bot_km": bins_km[1:],
        "total_mean":   by_depth_mean,
        "total_median": by_depth_median,
        "total_lo95":   by_depth_lo,
        "total_hi95":   by_depth_hi,
        "total_std":    by_depth_std
    })
    out_bydepth_sum = os.path.join(OUT_DIR, "oceanic_cellcount_by_depth_summary.csv")
    by_depth_summary.to_csv(out_bydepth_sum, index=False)

    print(f"[OK] By-depth matrix saved: {out_bydepth_mat}")
    print(f"[OK] By-depth summary saved: {out_bydepth_sum}")

if __name__ == "__main__":
    main()
