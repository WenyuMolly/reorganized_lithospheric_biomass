# Lithospheric Biomass Code

Code and data products for estimating rock-hosted lithospheric biomass from geothermal-gradient reconstructions, 122 degree C habitable-depth calculations, and continental/oceanic cell-density extrapolations.

## Repository Layout

```text
src/biomass/                         Importable Python source code
scripts/geothermal/                  Command-line entry points for XGBoost geothermal-gradient models
scripts/volume/                      Command-line entry points for MAST and habitable-volume steps
scripts/oceanic/                     Command-line entry points for oceanic biomass steps
scripts/continental/                 R entry points for continental biomass workflows
data/raw/                            Immutable input data
data/processed/                      Curated and intermediate inputs
runs/                                Generated tabular outputs
figures/benchmarks/                  Reference figures
figures/generated/                   Regenerated figures
tests/                               Regression tests
```

Reference outputs are stored under `runs/*/submitted/`. New analyses should be written to `runs/*/latest/` or to a timestamped folder under `runs/`.

## Installation

The Python environment is managed with `uv`:

```bash
uv sync
```

Run Python commands through the managed environment:

```bash
uv run python scripts/oceanic/tab_file_processor.py --help
```

The continental biomass workflows are R scripts. They were developed for R 4.x and use packages imported by the individual scripts, including `foreach`, `doParallel`, `glmnet`, `fields`, `nlstools`, and `ggplot2`.

## Workflow

### 1. Process Mean Annual Surface Temperature

```bash
uv run python scripts/volume/process_mast_file.py \
  --input data/raw/mast/era5_2024_monthly.nc \
  --regridded-output data/processed/mast/global_mean_temperature_1deg.csv
```

The generated `data/processed/mast/global_mean_temperature_1deg.csv` is used as the surface-temperature input for the habitable-volume calculation.

### 2. Train Geothermal-Gradient Models

Oceanic model:

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --Run oceanic_final \
  --run_type train \
  --params_algorithm random \
  --data_path data/raw/geothermal_model_final_data/split_ocean_1x1.csv
```

Continental model:

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --Run continental_final \
  --run_type train \
  --params_algorithm random \
  --is_land \
  --data_path data/raw/geothermal_model_final_data/split_ocean_1x1.csv
```

Training outputs are written under `runs/geothermal/1stAttempt/`.

### 3. Run Geothermal-Gradient Inference

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --if_inference \
  --data_path data/raw/geothermal_model_final_data/split_ocean_1x1.csv \
  --omodel_path 1stAttempt/oceanic_final/myModel1st.model \
  --cmodel_path 1stAttempt/continental_final/myModel1st.model
```

This writes `runs/geothermal/1stAttempt/total_oceanic.csv` and `runs/geothermal/1stAttempt/total_continental.csv`.

### 4. Calculate Habitable Lithospheric Volume

```bash
uv run python scripts/volume/habitable_volume.py \
  --continental_file runs/geothermal/1stAttempt/total_continental.csv \
  --oceanic_file runs/geothermal/1stAttempt/total_oceanic.csv \
  --mast_file data/processed/mast/global_mean_temperature_1deg.csv \
  --temperature 122 \
  --output_dir runs/volume/latest
```

Reference volume outputs are stored in `runs/volume/submitted/results/`.

### 5. Run Oceanic Biomass Estimates

Convert PANGAEA TAB files:

```bash
uv run python scripts/oceanic/tab_file_processor.py \
  --input-dir data/raw/oceanic/pangaea_exp357/tab_files \
  --output data/processed/oceanic/pangaea_exp357_cell_abundance_merged_corrected.csv \
  --write-individual \
  --individual-output-dir data/processed/oceanic/pangaea_exp357_csv_files
```

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

Oceanic cell-density sensitivity figure:

```bash
uv run python scripts/oceanic/plot_cell_density_violin.py
```

Oceanic scripts read `data/raw/oceanic/oceanic_cell_densities.xlsx`, `data/raw/oceanic/ecm/ECM1.txt`, and the volume table in `runs/volume/submitted/results/`.

### 6. Run Continental Biomass Estimates

Modified continental workflow:

```bash
Rscript scripts/continental/modified_magnabosco/Depth_and_Temperature_Fits_wenyu.R
Rscript scripts/continental/modified_magnabosco/Depth_and_Temperature_GLM_wenyu.R
Rscript scripts/continental/modified_magnabosco/Crust_Specific_Fits_wenyu.R
```

Original comparison workflow:

```bash
Rscript scripts/continental/original_magnabosco/Depth_and_Temperature_Fits_origin.R
Rscript scripts/continental/original_magnabosco/Depth_and_Temperature_GLM_origin.R
Rscript scripts/continental/original_magnabosco/Crust_Specific_Fits_origin.R
```

Continental scripts read from `data/processed/continental/` and write rerun outputs to `runs/continental/latest/`.

## Data Notes

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
