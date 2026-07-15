# Lithospheric Biomass Code

Code and data products for estimating rock-hosted lithospheric biomass from geothermal-gradient reconstructions, 122 °C habitable-depth calculations, and continental/oceanic cell-density extrapolations.

## Licence

Project-authored code, documentation, and project-authored data are released
under the [Creative Commons Attribution 4.0 International License](LICENSE).
Third-party data and model inputs retain their original licences and citation
requirements; see [`data/README.md`](data/README.md).

## Repository Layout

```text
src/biomass/                         Importable Python source code
src/biomass/preprocessing/           Reusable intermediate-data preparation utilities
scripts/geothermal/                  Command-line entry points for XGBoost geothermal-gradient models
scripts/preprocessing/               Command-line entry points for intermediate-data preparation
scripts/volume/                      Command-line entry points for habitable-volume steps
scripts/oceanic/                     Command-line entry points for oceanic biomass steps
scripts/continental/                 R entry points for continental biomass workflows
data/raw/                            Immutable input data
data/processed/                      Curated and intermediate inputs
runs/                                Generated tabular outputs
figures/benchmarks/                  Reference figures
figures/generated/                   Regenerated figures
tests/                               Regression tests
```

Reference outputs are stored under `runs/*/submitted/`. New long-running
analyses are written to timestamped folders under `runs/` by default. Set
`BIOMASS_RUN_ID` to group multiple commands into one named run directory.

## Installation

The Python environment is managed with `uv`:

```bash
uv sync
```

Run Python commands through the managed environment:

```bash
uv run python scripts/preprocessing/tab_file_processor.py --help
```

### R Installation (Required for Continental Biomass)

The Python `uv` environment does not install R or R packages. Install R and the
packages required by the Magnabosco continental scripts before running the
continental workflows in Step 6. Verify the local R environment with:

```bash
Rscript scripts/continental/check_environment.R
```

The scripts require `foreach`, `doParallel`, `glmnet`, `fields`, `nlstools`,
and `ggplot2`. Each completed continental run saves `r_session_info.txt` with
the R version and loaded-package provenance alongside its numerical outputs.

## Workflow

### 1. Process Mean Annual Surface Temperature

Optional ERA5 download, if `data/raw/mast/era5_2024_monthly.nc` is not already
present and CDS credentials are configured:

```bash
uv run python scripts/preprocessing/mast_get_by_cds.py
```

```bash
uv run python scripts/preprocessing/process_mast_file.py \
  --input data/raw/mast/era5_2024_monthly.nc \
  --regridded-output data/processed/mast/global_mean_temperature_1deg.csv
```

The generated `data/processed/mast/global_mean_temperature_1deg.csv` is used as the surface-temperature input for the habitable-volume calculation.
The raw ERA5 NetCDF is expected at `data/raw/mast/era5_2024_monthly.nc`; the processing step also writes an intermediate full-resolution table,
`data/processed/mast/global_mean_temperature_2024.csv`.

### 2. Geothermal-Gradient Models

Pre-trained geothermal-gradient models are expected at:

```text
runs/geothermal/1stAttempt/oceanic_final/myModel1st.model
runs/geothermal/1stAttempt/continental_final/myModel1st.model
```

If these files are present, skip directly to inference in Step 3. Retrain only
when you need to regenerate the model files.
The geothermal wrapper runs from `runs/geothermal/`, so the commands below use
paths relative to that directory for `--data_path` and model-file arguments.

Optional oceanic retraining command:

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --Run oceanic_final \
  --run_type train \
  --params_algorithm random \
  --data_path ../../data/raw/geothermal_model_final_data/split_ocean_1x1.csv
```

Optional continental retraining command:

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --Run continental_final \
  --run_type train \
  --params_algorithm random \
  --is_land \
  --data_path ../../data/raw/geothermal_model_final_data/split_ocean_1x1.csv
```

Training outputs are written under `runs/geothermal/1stAttempt/`.

### 3. Run Geothermal-Gradient Inference

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --if_inference \
  --data_path ../../data/raw/geothermal_model_final_data/split_ocean_1x1.csv \
  --omodel_path 1stAttempt/oceanic_final/myModel1st.model \
  --cmodel_path 1stAttempt/continental_final/myModel1st.model
```

This writes `runs/geothermal/1stAttempt/total_oceanic.csv` and `runs/geothermal/1stAttempt/total_continental.csv`.

### 4. Calculate Habitable Lithospheric Volume

Optional: set a run id before running volume and biomass workflows.

```bash
export BIOMASS_RUN_ID=$(date +%Y%m%d_%H%M%S)
```

```bash
uv run python scripts/volume/habitable_volume.py \
  --continental_file runs/geothermal/1stAttempt/total_continental.csv \
  --oceanic_file runs/geothermal/1stAttempt/total_oceanic.csv \
  --mast_file data/processed/mast/global_mean_temperature_1deg.csv \
  --temperature 122
```

New volume outputs are written to `runs/volume/$BIOMASS_RUN_ID/`, or to a new
timestamped folder under `runs/volume/` when `BIOMASS_RUN_ID` is unset.
Reference volume outputs are stored in `runs/volume/submitted/results/`.
The oceanic biomass scripts in Step 5 read the reference/submitted volume table
from `runs/volume/submitted/results/` by default. To use a newly generated
volume run, replace or copy the corresponding
`inference_and_depth_to_122.0_calculation_oceanic.csv` into that submitted
results directory before running the oceanic biomass scripts.

### 5. Run Oceanic Lithospheric Biomass Estimates

Optional intermediate-data preparation: convert PANGAEA TAB files. This is not
required when using the curated `data/raw/oceanic/oceanic_cell_densities.xlsx`
workbook directly.

```bash
uv run python scripts/preprocessing/tab_file_processor.py \
  --input-dir data/raw/oceanic/pangaea_exp357/tab_files \
  --output data/processed/oceanic/pangaea_exp357_cell_abundance_merged_corrected.csv \
  --write-individual \
  --individual-output-dir data/processed/oceanic/pangaea_exp357_csv_files
```

Always pass either `--exclude-shallow` or `--include-shallow` when reproducing
the manuscript oceanic biomass workflows; do not rely on script defaults.
`--exclude-shallow` removes the shallow/seawater-contacted reference set used in
the manuscript sensitivity comparison, whereas `--include-shallow` keeps those
observations.

Log10 bootstrap estimates without shallow/seawater-contacted samples:

```bash
uv run python scripts/oceanic/unstratified_cellcount.py --exclude-shallow
uv run python scripts/oceanic/stratified_cellcount.py --exclude-shallow
```

Log10 bootstrap estimates with shallow/seawater-contacted samples:

```bash
uv run python scripts/oceanic/unstratified_cellcount.py --include-shallow
uv run python scripts/oceanic/stratified_cellcount.py --include-shallow
```

Power-law estimates without shallow/seawater-contacted samples:

```bash
uv run python scripts/oceanic/unstratified_power_fit.py --exclude-shallow
uv run python scripts/oceanic/stratified_power_fit.py --exclude-shallow
```

Power-law estimates with shallow/seawater-contacted samples:

```bash
uv run python scripts/oceanic/unstratified_power_fit.py --include-shallow
uv run python scripts/oceanic/stratified_power_fit.py --include-shallow
```

Oceanic geothermal z122 uncertainty sensitivity for both shallow-excluded and
shallow-included datasets:

```bash
uv run python scripts/sensitivity/summarize_oceanic_geothermal_z122_uncertainty.py \
  --dataset both \
  --method all \
  --n-draws 1000 \
  --seed 42
```

This runs the existing stratified/unstratified log10-bootstrap and power-law MC
scripts with `--z122-scenario low`, `base`, and `high`. The wrapper defaults to
`--dataset without-shallow` if no dataset is specified. Outputs are written
under `runs/oceanic/geothermal_z122_uncertainty/$BIOMASS_RUN_ID/`, or under a
new timestamped folder when `BIOMASS_RUN_ID` is unset. Avoid `--reuse-existing`
for formal reruns unless the existing scenario folders are known to match the
intended code, inputs, seed, and draw count.

Oceanic cell-density sensitivity figure:

```bash
uv run python scripts/oceanic/plot_cell_density_violin.py
```

Oceanic scripts read `data/raw/oceanic/oceanic_cell_densities.xlsx`, `data/raw/oceanic/ecm/ECM1.txt`, and the volume table in `runs/volume/submitted/results/`.

### 6. Run Continental Lithospheric Biomass Estimates

The Python wrapper runs every R script in a workflow with one shared,
timestamped run identifier. Modified continental workflow:

```bash
uv run python scripts/continental/run_workflow.py \
  --workflow modified_magnabosco
```

Original comparison workflow:

```bash
uv run python scripts/continental/run_workflow.py \
  --workflow original_magnabosco
```

Pass `--run-id NAME` to assign a meaningful shared output name. The resulting
files are written to `runs/continental/<run-id>/<workflow>/`. The underlying
R scripts can still be called directly, but then set `BIOMASS_RUN_ID` first to
keep all scripts in the same output directory.

## Data Notes

- See [`data/README.md`](data/README.md) for data provenance, citation, and
  third-party redistribution notes.
- `data/raw/mast/` contains source files used to prepare mean annual surface temperature.
- `data/raw/oceanic/` contains oceanic cell-density inputs, ECM1 layer information, and PANGAEA TAB files.
- `data/raw/continental/` contains continental cell-count input data.
- `data/processed/` contains curated inputs used directly by downstream scripts.
- `runs/` contains generated tabular outputs and reference result tables.

## Tests

Run the Python tests with:

```bash
uv run pytest
```

The test suite includes lightweight workflow checks for geothermal-gradient model training/inference, habitable-volume calculation, oceanic biomass fitting utilities, and continental biomass script/data organization.
