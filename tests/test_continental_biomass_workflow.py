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
        assert "runs/continental/latest" in text
        assert "dir.create(output_dir" in text


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

    assert (PROJECT_ROOT / "runs/continental/submitted/modified_magnabosco").exists()
    assert (PROJECT_ROOT / "runs/continental/submitted/original_magnabosco").exists()
