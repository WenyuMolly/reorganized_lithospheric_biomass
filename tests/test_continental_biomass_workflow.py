from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_continental_workflow_scripts_use_project_data_and_runs_directories():
    scripts = [
        PROJECT_ROOT / "scripts/continental/modified_magnabosco/Depth_and_Temperature_Fits_wenyu.R",
        PROJECT_ROOT / "scripts/continental/modified_magnabosco/Depth_and_Temperature_GLM_wenyu.R",
        PROJECT_ROOT / "scripts/continental/modified_magnabosco/Crust_Specific_Fits_wenyu.R",
        PROJECT_ROOT / "scripts/continental/original_magnabosco/Depth_and_Temperature_Fits_origin.R",
        PROJECT_ROOT / "scripts/continental/original_magnabosco/Depth_and_Temperature_GLM_origin.R",
        PROJECT_ROOT / "scripts/continental/original_magnabosco/Crust_Specific_Fits_origin.R",
    ]

    for script in scripts:
        text = script.read_text()
        assert "data/processed/continental" in text or "data/raw/continental" in text
        assert 'Sys.getenv("BIOMASS_RUN_ID"' in text
        assert 'file.path(project_root, "runs", "continental"' in text
        assert "dir.create(output_dir" in text
        assert "r_session_info.txt" in text


def test_continental_inputs_and_reference_outputs_are_available():
    expected_inputs = [
        PROJECT_ROOT / "data/raw/continental/cores_with_PCR.csv",
        PROJECT_ROOT / "data/processed/continental/modified_magnabosco/metadata_with_merged_depth_and_gradient.csv",
        PROJECT_ROOT / "data/processed/continental/modified_magnabosco/cores_with_gradient_filled.csv",
        PROJECT_ROOT / "data/processed/continental/original_magnabosco/metadata_by_grid.csv",
        PROJECT_ROOT / "data/processed/continental/original_magnabosco/cores_with_PCR.csv",
    ]

    for path in expected_inputs:
        assert path.exists()
        assert path.stat().st_size > 0

    reference_root = PROJECT_ROOT / "runs/continental/submitted/2026-07-18_reference_summaries"
    expected_reference_outputs = [
        reference_root / "modified_magnabosco/CSF_bootstrap_total_biomass_and_mse.csv",
        reference_root / "modified_magnabosco/temperature_model_total_biomass.csv",
        reference_root / "original_magnabosco/Origin_CSF_bootstrap_total_biomass_and_mse.csv",
        reference_root / "original_magnabosco/Origin_temperature_model_total_biomass.csv",
    ]
    for path in expected_reference_outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_continental_environment_check_lists_required_packages():
    check_script = PROJECT_ROOT / "scripts/continental/check_environment.R"
    text = check_script.read_text()

    for package in ("foreach", "doParallel", "glmnet", "fields", "nlstools", "ggplot2"):
        assert f'"{package}"' in text


def test_continental_python_wrapper_is_available():
    wrapper = PROJECT_ROOT / "scripts/continental/run_workflow.py"
    text = wrapper.read_text()

    assert "run_workflow(" in text
    assert "--workflow" in text
    assert "--run-id" in text
