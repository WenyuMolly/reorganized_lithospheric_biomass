#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "runs/oceanic/geothermal_z122_uncertainty/without_shallow"
TABLE_DIR = PROJECT_ROOT / "results/tables"

METHODS = {
    "stratified-log10": {
        "script": PROJECT_ROOT / "scripts/oceanic/stratified_cellcount.py",
        "by_depth_matrix": "oceanic_cellcount_by_depth_matrix.csv",
        "grid_summary": "oceanic_cell_counts_bootstrap_mc.csv",
        "totals_by_draw": "oceanic_cell_totals_by_draw.csv",
        "totals_summary": "oceanic_cell_totals_bootstrap_mc.csv",
        "total_row": "Total in Each Cell",
        "total_columns": {
            "mean": "Total Mean",
            "median": "Total Median",
            "p2_5": "Total 2.5%",
            "p97_5": "Total 97.5%",
        },
        "depth_std_col": "Perturbed Depth Std",
    },
    "unstratified-log10": {
        "script": PROJECT_ROOT / "scripts/oceanic/unstratified_cellcount.py",
        "by_depth_matrix": "oceanic_cellcount_by_depth_matrix_unstrat_log10.csv",
        "grid_summary": "oceanic_cell_counts_bootstrap_mc_unstrat_log10.csv",
        "totals_by_draw": "oceanic_cell_totals_by_draw.csv",
        "totals_summary": "oceanic_cell_totals_bootstrap_mc_unstrat_log10.csv",
        "total_row": "Total",
        "total_columns": {
            "mean": "Mean",
            "median": "Median",
            "p2_5": "2.5%",
            "p97_5": "97.5%",
        },
        "depth_std_col": "depth_draw_km Std",
    },
    "stratified-power": {
        "script": PROJECT_ROOT / "scripts/oceanic/stratified_power_fit.py",
        "by_depth_matrix": "oceanic_cellcount_by_depth_matrix.csv",
        "grid_summary": "oceanic_cell_counts_power_mc.csv",
        "totals_by_draw": "oceanic_cell_totals_by_draw.csv",
        "totals_summary": "oceanic_cell_totals_power_mc.csv",
        "total_row": "Total in Each Cell",
        "total_columns": {
            "mean": "Total Mean",
            "median": "Total Median",
            "p2_5": "Total 2.5%",
            "p97_5": "Total 97.5%",
        },
        "depth_std_col": "Perturbed Depth Std",
    },
    "unstratified-power": {
        "script": PROJECT_ROOT / "scripts/oceanic/unstratified_power_fit.py",
        "by_depth_matrix": "oceanic_cellcount_by_depth_matrix_unstrat_power.csv",
        "grid_summary": "oceanic_cell_counts_power_mc.csv",
        "totals_by_draw": "oceanic_cell_totals_by_draw.csv",
        "totals_summary": "oceanic_cell_totals_power_mc.csv",
        "total_row": "Total in Each Cell",
        "total_columns": {
            "mean": "Total Mean",
            "median": "Total Median",
            "p2_5": "Total 2.5%",
            "p97_5": "Total 97.5%",
        },
        "depth_std_col": "Perturbed Depth Std",
    },
}

SCENARIOS = {
    "low": "max(maxdepth - maxdepth_sd, 0)",
    "base": "maxdepth",
    "high": "maxdepth + maxdepth_sd",
}

METHOD_MIN = 4.94e27
METHOD_MAX = 15.55e27
MAIN_SHALLOW_EXCLUDED_TOTALS = {
    "stratified-log10": 9.35e27,
    "unstratified-log10": 4.94e27,
    "stratified-power": 15.55e27,
    "unstratified-power": 9.05e27,
}
VALIDATION_RTOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and summarize shallow-excluded oceanic geothermal z122 uncertainty "
            "using the existing oceanic biomass MC workflows."
        )
    )
    method_choices = ["all", *METHODS.keys()]
    parser.add_argument("--method", choices=method_choices, default="all", help="Oceanic biomass method to run.")
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip a scenario run when its by-depth matrix already exists.")
    return parser.parse_args()


def selected_methods(method: str) -> list[str]:
    return list(METHODS) if method == "all" else [method]


def reusable_scenario_outputs(config: dict[str, object], out_dir: Path, n_draws: int) -> bool:
    matrix_path = out_dir / str(config["by_depth_matrix"])
    totals_by_draw_path = out_dir / str(config["totals_by_draw"])
    totals_summary_path = out_dir / str(config["totals_summary"])
    if not (matrix_path.exists() and totals_by_draw_path.exists() and totals_summary_path.exists()):
        return False

    try:
        totals_by_draw = pd.read_csv(totals_by_draw_path, usecols=["iter"])
    except Exception:
        return False
    return len(totals_by_draw) == n_draws


def run_scenario(method: str, scenario: str, n_draws: int, seed: int, reuse_existing: bool) -> Path:
    config = METHODS[method]
    script = config["script"]
    if not script.exists():
        raise FileNotFoundError(f"Could not locate oceanic biomass script for {method}: {script}")

    out_dir = RUN_ROOT / method / scenario
    if reuse_existing and reusable_scenario_outputs(config, out_dir, n_draws):
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--exclude-shallow",
        "--z122-scenario",
        scenario,
        "--n-draws",
        str(n_draws),
        "--seed",
        str(seed),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return out_dir


def draw_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("iter_")]


def depth_bin_rows(matrix: pd.DataFrame, matrix_path: Path) -> pd.Series:
    required = {"depth_top_km", "depth_bot_km"}
    missing = sorted(required - set(matrix.columns))
    if missing:
        raise RuntimeError(
            f"By-depth matrix is missing required depth-bin columns {missing}: {matrix_path}"
        )

    non_iter_cols = [col for col in matrix.columns if not col.startswith("iter_")]
    total_like = matrix[non_iter_cols].astype(str).apply(
        lambda col: col.str.contains("total|subtotal|cumul", case=False, na=False)
    )
    if bool(total_like.any(axis=None)):
        raise RuntimeError(
            "By-depth matrix appears to contain total/subtotal/cumulative rows; "
            f"refusing to sum all rows because that would double-count: {matrix_path}"
        )

    top = pd.to_numeric(matrix["depth_top_km"], errors="coerce")
    bot = pd.to_numeric(matrix["depth_bot_km"], errors="coerce")
    valid = top.notna() & bot.notna() & (bot > top)
    if not bool(valid.all()):
        bad_count = int((~valid).sum())
        raise RuntimeError(
            f"By-depth matrix contains {bad_count} non-bin row(s); expected only rows with "
            f"finite depth_top_km and depth_bot_km where bottom > top: {matrix_path}"
        )
    return valid


def read_matrix_draw_totals(method: str, out_dir: Path) -> tuple[pd.Series, list[str]]:
    matrix_path = out_dir / METHODS[method]["by_depth_matrix"]
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing by-depth matrix: {matrix_path}")

    matrix = pd.read_csv(matrix_path)
    iter_cols = draw_columns(matrix)
    if not iter_cols:
        raise RuntimeError(f"No iter_XXXX columns found in {matrix_path}")

    valid_rows = depth_bin_rows(matrix, matrix_path)
    totals = matrix.loc[valid_rows, iter_cols].sum(axis=0)
    totals.index = iter_cols
    return totals.astype(float), iter_cols


def read_global_totals_by_draw(method: str, out_dir: Path) -> tuple[pd.Series, list[str]]:
    totals_path = out_dir / METHODS[method]["totals_by_draw"]
    if not totals_path.exists():
        raise FileNotFoundError(f"Missing global totals by-draw file: {totals_path}")

    totals = pd.read_csv(totals_path)
    required = {"iter", "Total"}
    missing = sorted(required - set(totals.columns))
    if missing:
        raise RuntimeError(f"Global totals by-draw file is missing columns {missing}: {totals_path}")

    iter_ids = totals["iter"].astype(str).tolist()
    if not iter_ids:
        raise RuntimeError(f"No draw rows found in global totals by-draw file: {totals_path}")
    if len(iter_ids) != len(set(iter_ids)):
        raise RuntimeError(f"Duplicate iter ids found in global totals by-draw file: {totals_path}")

    values = pd.to_numeric(totals["Total"], errors="coerce")
    if values.isna().any():
        raise RuntimeError(f"Non-numeric Total value found in global totals by-draw file: {totals_path}")

    series = pd.Series(values.to_numpy(dtype=float), index=iter_ids)
    return series, iter_ids


def read_global_total_summary(method: str, out_dir: Path) -> dict[str, float]:
    config = METHODS[method]
    totals_path = out_dir / config["totals_summary"]
    if not totals_path.exists():
        raise FileNotFoundError(f"Missing global totals summary: {totals_path}")

    totals = pd.read_csv(totals_path)
    if "Layer" not in totals.columns:
        raise RuntimeError(f"Global totals summary has no Layer column: {totals_path}")

    row = totals.loc[totals["Layer"].astype(str).eq(config["total_row"])]
    if row.empty:
        raise RuntimeError(
            f"Could not find total row {config['total_row']!r} in global totals summary: {totals_path}"
        )
    row = row.iloc[0]
    columns = config["total_columns"]
    missing = sorted(set(columns.values()) - set(totals.columns))
    if missing:
        raise RuntimeError(f"Global totals summary is missing columns {missing}: {totals_path}")

    return {
        key: float(row[col])
        for key, col in columns.items()
    }


def summarize_draws(values: pd.Series | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def base_uses_deterministic_maxdepth(method: str, base_out_dir: Path) -> bool:
    config = METHODS[method]
    grid_path = base_out_dir / config["grid_summary"]
    if not grid_path.exists():
        return False
    grid = pd.read_csv(grid_path)
    col = config["depth_std_col"]
    if col not in grid.columns:
        return False
    return bool(np.nanmax(np.abs(pd.to_numeric(grid[col], errors="coerce"))) < 1e-10)


def validation_issue(
    method: str,
    totals_by_scenario: dict[str, pd.Series],
    matrix_totals_by_scenario: dict[str, pd.Series],
    summary_by_scenario: dict[str, dict[str, float]],
    columns_by_scenario: dict[str, list[str]],
    output_dirs: dict[str, Path],
) -> str | None:
    missing = [
        scenario for scenario, out_dir in output_dirs.items()
        if not (out_dir / METHODS[method]["by_depth_matrix"]).exists()
    ]
    if missing:
        return f"output mismatch: missing by-depth matrix for {', '.join(missing)}"

    missing_by_draw = [
        scenario for scenario, out_dir in output_dirs.items()
        if not (out_dir / METHODS[method]["totals_by_draw"]).exists()
    ]
    if missing_by_draw:
        return f"output mismatch: missing global totals by-draw file for {', '.join(missing_by_draw)}"

    base_cols = columns_by_scenario["base"]
    for scenario in ("low", "high"):
        if columns_by_scenario[scenario] != base_cols:
            return f"output mismatch: {scenario} draw columns differ from base"

    if not base_uses_deterministic_maxdepth(method, output_dirs["base"]):
        return "random sampling mismatch: base scenario does not appear to use deterministic maxdepth"

    for scenario, by_draw_totals in totals_by_scenario.items():
        by_draw_mean = float(by_draw_totals.mean())
        summary_mean = float(summary_by_scenario[scenario]["mean"])
        if not np.isclose(by_draw_mean, summary_mean, rtol=VALIDATION_RTOL, atol=0.0):
            pct = (by_draw_mean - summary_mean) / summary_mean * 100.0
            return (
                f"by-draw/global-total mismatch for {scenario}: per-draw totals mean differs "
                f"from global totals summary by {pct:.2f}%"
            )

    for scenario, matrix_totals in matrix_totals_by_scenario.items():
        matrix_mean = float(matrix_totals.mean())
        summary_mean = float(summary_by_scenario[scenario]["mean"])
        if not np.isclose(matrix_mean, summary_mean, rtol=0.01, atol=0.0):
            pct = (matrix_mean - summary_mean) / summary_mean * 100.0
            print(
                f"[WARN] {method} {scenario}: by-depth matrix mean differs from "
                f"global totals summary by {pct:.2f}%"
            )

    low_mean = float(totals_by_scenario["low"].mean())
    base_mean = float(totals_by_scenario["base"].mean())
    high_mean = float(totals_by_scenario["high"].mean())
    if not (high_mean >= base_mean >= low_mean):
        if columns_by_scenario["low"] != columns_by_scenario["base"] or columns_by_scenario["high"] != columns_by_scenario["base"]:
            return "random sampling mismatch: scenario draw columns are not aligned"
        return "numerical clipping or integration mismatch: mean cells are not monotonic with z122"

    return None


def relation_to_method_spread(percent: float) -> str:
    if percent < 50.0:
        return "smaller than"
    if percent <= 150.0:
        return "comparable to"
    return "larger than"


def main() -> None:
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    scenario_rows = []
    paired_rows = []
    manuscript_rows = []

    for method in selected_methods(args.method):
        output_dirs = {
            scenario: run_scenario(method, scenario, args.n_draws, args.seed, args.reuse_existing)
            for scenario in ("low", "base", "high")
        }

        totals_by_scenario = {}
        matrix_totals_by_scenario = {}
        summary_by_scenario = {}
        columns_by_scenario = {}
        scenario_start_idx = len(scenario_rows)
        for scenario in ("low", "base", "high"):
            totals, iter_cols = read_global_totals_by_draw(method, output_dirs[scenario])
            matrix_totals, matrix_iter_cols = read_matrix_draw_totals(method, output_dirs[scenario])
            summary_stats = read_global_total_summary(method, output_dirs[scenario])
            matrix_stats = summarize_draws(matrix_totals)
            totals_by_scenario[scenario] = totals
            matrix_totals_by_scenario[scenario] = matrix_totals
            summary_by_scenario[scenario] = summary_stats
            columns_by_scenario[scenario] = iter_cols
            matrix_mean = matrix_stats["mean"]
            summary_mean = summary_stats["mean"]
            if matrix_iter_cols != iter_cols:
                raise RuntimeError(f"By-depth matrix draw ids differ from totals-by-draw ids for {method} {scenario}")
            scenario_rows.append({
                "method": method,
                "scenario": scenario,
                "z122_definition": SCENARIOS[scenario],
                "n_draws": len(iter_cols),
                "cells_mean": summary_stats["mean"],
                "cells_median": summary_stats["median"],
                "cells_p2_5": summary_stats["p2_5"],
                "cells_p97_5": summary_stats["p97_5"],
                "matrix_cells_mean": matrix_mean,
                "matrix_vs_global_total_percent": (matrix_mean - summary_mean) / summary_mean * 100.0,
                "expected_main_shallow_excluded_mean": (
                    MAIN_SHALLOW_EXCLUDED_TOTALS[method] if scenario == "base" else np.nan
                ),
                "base_vs_expected_main_percent": (
                    (summary_mean - MAIN_SHALLOW_EXCLUDED_TOTALS[method])
                    / MAIN_SHALLOW_EXCLUDED_TOTALS[method]
                    * 100.0
                    if scenario == "base"
                    else np.nan
                ),
                "source_output_dir": str(output_dirs[scenario]),
            })

        low = totals_by_scenario["low"]
        base = totals_by_scenario["base"]
        high = totals_by_scenario["high"]

        paired_start_idx = len(paired_rows)
        paired_metrics = {
            "low_vs_base_percent": (low - base) / base * 100.0,
            "high_vs_base_percent": (high - base) / base * 100.0,
            "high_low_spread_percent_of_base": (high - low) / base * 100.0,
            "high_low_fold": high / low,
        }
        for metric, values in paired_metrics.items():
            stats = summarize_draws(values)
            paired_rows.append({
                "method": method,
                "metric": metric,
                "mean": stats["mean"],
                "median": stats["median"],
                "p2_5": stats["p2_5"],
                "p97_5": stats["p97_5"],
            })

        cells_low_mean = float(summary_by_scenario["low"]["mean"])
        cells_base_mean = float(summary_by_scenario["base"]["mean"])
        cells_high_mean = float(summary_by_scenario["high"]["mean"])
        method_spread = METHOD_MAX - METHOD_MIN
        thermal_spread = cells_high_mean - cells_low_mean
        thermal_vs_method = thermal_spread / method_spread * 100.0
        validation = validation_issue(
            method,
            totals_by_scenario,
            matrix_totals_by_scenario,
            summary_by_scenario,
            columns_by_scenario,
            output_dirs,
        )
        base_vs_main_pct = (
            (cells_base_mean - MAIN_SHALLOW_EXCLUDED_TOTALS[method])
            / MAIN_SHALLOW_EXCLUDED_TOTALS[method]
            * 100.0
        )
        if abs(base_vs_main_pct) > 1.0:
            print(
                f"[WARN] {method}: deterministic z122 base differs from latest main "
                f"Shallow-Excluded total by {base_vs_main_pct:.2f}%"
            )
        validation_status = "passed" if validation is None else "failed"
        validation_message = "" if validation is None else validation
        for row in scenario_rows[scenario_start_idx:]:
            row["validation_status"] = validation_status
            row["validation_issue"] = validation_message
        for row in paired_rows[paired_start_idx:]:
            row["validation_status"] = validation_status
            row["validation_issue"] = validation_message

        if validation is None:
            relation = relation_to_method_spread(thermal_vs_method)
            interpretation = (
                "Using grid-cell z122 uncertainty represented by maxdepth +/- maxdepth_sd, "
                f"the {method} oceanic biomass estimate varies from {cells_low_mean:.3e} to "
                f"{cells_high_mean:.3e} cells around a baseline of {cells_base_mean:.3e} cells. "
                f"This corresponds to a high-low spread of {thermal_spread / cells_base_mean * 100.0:.2f}% "
                f"of the baseline estimate, or {cells_high_mean / cells_low_mean:.2f}-fold. "
                f"The thermal-boundary spread is {thermal_vs_method:.2f}% of the spread across the four "
                f"preferred oceanic extrapolation methods, indicating that geothermal-model uncertainty is "
                f"{relation} extrapolation-method uncertainty."
            )
        else:
            interpretation = f"Validation failed ({validation}); sensitivity results should not be interpreted."

        manuscript_rows.append({
            "method": method,
            "n_draws": len(columns_by_scenario["base"]),
            "cells_low_mean": cells_low_mean,
            "cells_base_mean": cells_base_mean,
            "cells_high_mean": cells_high_mean,
            "low_vs_base_percent_from_means": (cells_low_mean - cells_base_mean) / cells_base_mean * 100.0,
            "high_vs_base_percent_from_means": (cells_high_mean - cells_base_mean) / cells_base_mean * 100.0,
            "high_low_percent_of_base_from_means": thermal_spread / cells_base_mean * 100.0,
            "high_low_fold_from_means": cells_high_mean / cells_low_mean,
            "method_min": METHOD_MIN,
            "method_max": METHOD_MAX,
            "method_spread": method_spread,
            "thermal_spread_vs_method_spread_percent": thermal_vs_method,
            "interpretation": interpretation,
        })

    scenario_out = TABLE_DIR / "oceanic_geothermal_z122_uncertainty_by_scenario.csv"
    paired_out = TABLE_DIR / "oceanic_geothermal_z122_uncertainty_paired_summary.csv"
    manuscript_out = TABLE_DIR / "oceanic_geothermal_z122_uncertainty_manuscript_summary.csv"

    pd.DataFrame(scenario_rows).to_csv(scenario_out, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_out, index=False)
    pd.DataFrame(manuscript_rows).to_csv(manuscript_out, index=False)

    print(f"[OK] Wrote {scenario_out}")
    print(f"[OK] Wrote {paired_out}")
    print(f"[OK] Wrote {manuscript_out}")
    for row in manuscript_rows:
        if row["interpretation"].startswith("Validation failed"):
            print(f"[WARN] {row['method']}: {row['interpretation']}")


if __name__ == "__main__":
    main()
