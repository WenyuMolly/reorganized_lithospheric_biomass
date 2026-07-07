"""Helpers for R-based continental biomass estimation workflows.

The underlying computations are performed by R scripts located in
`scripts/continental/`. This module provides Python utilities to locate inputs,
run those R scripts, check the R environment, and collect result files for
downstream Python analysis.

Available workflows:
- modified_magnabosco: Updated depth/temperature fits with merged gradient data
- original_magnabosco: Original Magnabosco et al. methodology for comparison

Both workflows include:
- Depth-based power-law fits
- Temperature-based linear fits
- GLM (elastic net) fits
- Crust-specific fits (CSF)
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VALID_WORKFLOWS = ("modified_magnabosco", "original_magnabosco")

DEFAULT_SCRIPTS = {
    "modified_magnabosco": [
        "Depth_and_Temperature_Fits_wenyu.R",
        "Depth_and_Temperature_GLM_wenyu.R",
        "Crust_Specific_Fits_wenyu.R",
    ],
    "original_magnabosco": [
        "Depth_and_Temperature_Fits_origin.R",
        "Depth_and_Temperature_GLM_origin.R",
        "Crust_Specific_Fits_origin.R",
    ],
}

RESULT_PATTERNS = {
    "bootstrap_total_biomass": ["*CSF_bootstrap_total_biomass*.csv"],
    "bootstrap_grid_cell": ["*CSF_bootstrap_grid_cell*.csv"],
    "bootstrap_parameters": ["*CSF_bootstrap_model_parameters*.csv"],
    "bootstrap_rsquare": ["*CSF_bootstrap*rsquare*.csv"],
    "bootstrap_rdata": ["*CSF_full_bootstrap_results*.RData"],
    "glm_biomass": ["glm*.biomass.csv", "origin_glm*.biomass.csv", "glm*_biomass*.csv"],
    "glm_error": ["glm*.error.csv", "origin_glm*.error.csv", "glm*_error*.csv"],
    "glm_grid_result": ["glm*GridResult*.csv", "origin_glm*GridResult*.csv"],
    "glm_rsquare": ["glm*.rsq.csv"],
    "glm_parameters": ["glm*.parameters.csv"],
    "depth_model_biomass": ["*depth_model_total_biomass*.csv", "lm*.biomass*.csv"],
    "depth_model_error": ["*depth_model_error*.csv", "*tdepth_model_error*.csv", "lm*.error*.csv"],
    "depth_model_grid_result": ["*depth_model_grid_result*.csv", "lm*GridResult*.csv"],
    "depth_model_parameters": ["*depth_model_parameters*.csv", "lm*parameters*.csv"],
    "temperature_model_biomass": ["*temperature_model_total_biomass*.csv"],
    "temperature_model_error": ["*temperature_model_error*.csv"],
    "temperature_model_grid_result": ["*temperature_model_grid_result*.csv"],
    "temperature_model_parameters": ["*temperature_model_parameters*.csv"],
    "by_depth_matrix": ["*by_depth_matrix*.csv"],
    "by_depth_summary": ["*by_depth_summary*.csv"],
}


def validate_workflow(workflow: str) -> str:
    """Validate and return a continental workflow name."""
    if workflow not in VALID_WORKFLOWS:
        raise ValueError(f"Invalid workflow '{workflow}'. Must be one of: {list(VALID_WORKFLOWS)}")
    return workflow


def get_script_dir(workflow: str) -> Path:
    """Return the directory containing R scripts for the specified workflow.

    Parameters
    ----------
    workflow : str
        Either "modified_magnabosco" or "original_magnabosco".

    Returns
    -------
    Path
        Directory containing the R scripts.
    """
    workflow = validate_workflow(workflow)
    return PROJECT_ROOT / "scripts" / "continental" / workflow


def get_input_dir(workflow: str) -> Path:
    """Return the input data directory for the specified workflow.

    Parameters
    ----------
    workflow : str
        Either "modified_magnabosco" or "original_magnabosco".

    Returns
    -------
    Path
        Input data directory under `data/processed/continental/`.
    """
    workflow = validate_workflow(workflow)
    return PROJECT_ROOT / "data" / "processed" / "continental" / workflow


def get_output_dir(workflow: str, base_dir: str = "latest") -> Path:
    """Return the output directory for the specified workflow.

    Parameters
    ----------
    workflow : str
        Either "modified_magnabosco" or "original_magnabosco".
    base_dir : str, optional
        Output subdirectory name. Default is "latest" for new runs.
        Use "submitted" for reference outputs.

    Returns
    -------
    Path
        Output directory under `runs/continental/{base_dir}/`.
    """
    workflow = validate_workflow(workflow)
    return PROJECT_ROOT / "runs" / "continental" / base_dir / workflow


def run_r_script(
    script_name: str,
    workflow: str = "modified_magnabosco",
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a continental biomass R script.

    Parameters
    ----------
    script_name : str
        Name of the R script file (e.g., "Depth_and_Temperature_Fits_wenyu.R").
    workflow : str, optional
        Either "modified_magnabosco" or "original_magnabosco".
    timeout : int, optional
        Maximum execution time in seconds. None means no limit.
    check : bool, optional
        If True, raise CalledProcessError on non-zero exit code.

    Returns
    -------
    subprocess.CompletedProcess
        Result of the R script execution.

    Raises
    ------
    FileNotFoundError
        If the R script does not exist.
    subprocess.CalledProcessError
        If the R script fails and check=True.

    Notes
    -----
    The R scripts define their own input/output paths relative to the project
    root. This wrapper invokes the script without changing those paths.
    """
    script_dir = get_script_dir(workflow)
    script_path = script_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    cmd = ["Rscript", str(script_path)]

    warnings.warn(
        f"Running R script {script_name}. "
        "This may take several minutes for bootstrap iterations.",
        UserWarning
    )

    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def run_workflow(
    workflow: str = "modified_magnabosco",
    scripts: list[str] | None = None,
    strict: bool = True,
) -> dict[str, subprocess.CompletedProcess | subprocess.CalledProcessError]:
    """Run all or selected scripts for a continental biomass workflow.

    Parameters
    ----------
    workflow : str, optional
        Either "modified_magnabosco" or "original_magnabosco".
    scripts : list[str], optional
        List of script names to run. If None, run all available scripts.
    strict : bool, optional
        If True, missing script names raise FileNotFoundError. If False, missing
        scripts are skipped with a warning.

    Returns
    -------
    dict[str, subprocess.CompletedProcess | subprocess.CalledProcessError]
        Mapping from script name to execution result or failed process error.

    Examples
    --------
    Run all modified workflow scripts:

    >>> from biomass.continental import run_workflow
    >>> results = run_workflow("modified_magnabosco")

    Run only depth-temperature fits:

    >>> results = run_workflow(
    ...     "modified_magnabosco",
    ...     scripts=["Depth_and_Temperature_Fits_wenyu.R"]
    ... )
    """
    workflow = validate_workflow(workflow)
    script_dir = get_script_dir(workflow)

    if scripts is None:
        scripts = DEFAULT_SCRIPTS[workflow]

    results = {}

    for script_name in scripts:
        script_path = script_dir / script_name
        if not script_path.exists():
            if strict:
                raise FileNotFoundError(f"R script not found: {script_path}")
            warnings.warn(f"Script not found, skipping: {script_path}", UserWarning)
            continue

        print(f"[INFO] Running {script_name}...")
        try:
            result = run_r_script(script_name, workflow=workflow)
            results[script_name] = result
            print(f"[OK] {script_name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {script_name} failed with exit code {e.returncode}")
            print(f"        stderr: {e.stderr[:500]}")
            results[script_name] = e

    return results


def load_results(
    workflow: str = "modified_magnabosco",
    base_dir: str = "submitted",
) -> dict[str, Path | list[Path]]:
    """Locate result files for a completed workflow.

    Parameters
    ----------
    workflow : str, optional
        Either "modified_magnabosco" or "original_magnabosco".
    base_dir : str, optional
        Output subdirectory name. Default "submitted" for reference outputs.

    Returns
    -------
    dict[str, Path | list[Path]]
        Mapping from result file description to one or more file paths.

    Examples
    --------
    List all submitted modified workflow results:

    >>> from biomass.continental import load_results
    >>> files = load_results("modified_magnabosco", "submitted")
    >>> for name, path in files.items():
    ...     print(f"{name}: {path}")
    """
    output_dir = get_output_dir(workflow, base_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    found_files = {}

    for name, patterns in RESULT_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(output_dir.glob(pattern))
        matches = sorted(set(matches))
        if matches:
            found_files[name] = matches[0] if len(matches) == 1 else matches

    all_outputs = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".rdata"}
    )
    if all_outputs:
        found_files["all_outputs"] = all_outputs

    return found_files


def check_r_available() -> bool:
    """Check if R and required packages are available.

    Returns
    -------
    bool
        True if R is installed and accessible.
    """
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_required_r_packages() -> list[str]:
    """Check if required R packages are installed.

    Returns
    -------
    list[str]
        List of missing package names.
    """
    required_packages = [
        "foreach",
        "doParallel",
        "glmnet",
        "fields",
        "nlstools",
        "ggplot2",
    ]

    missing = []

    for pkg in required_packages:
        try:
            result = subprocess.run(
                ["Rscript", "-e", f"library({pkg})"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                missing.append(pkg)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(pkg)

    return missing


# Convenience exports
__all__ = [
    "run_workflow",
    "run_r_script",
    "load_results",
    "get_script_dir",
    "get_input_dir",
    "get_output_dir",
    "check_r_available",
    "check_required_r_packages",
    "validate_workflow",
]
