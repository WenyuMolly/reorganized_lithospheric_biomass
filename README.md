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
tests/                               Common Python tests
```

New long-running analyses are written to timestamped folders under `runs/` by
default. Set `BIOMASS_RUN_ID` to group multiple commands into one named run
directory.

Reference outputs are stored under `runs/*/submitted/`.

Set a run id before running volume and biomass workflows when several commands
should write to the same run directory:

```bash
export BIOMASS_RUN_ID=$(date +%Y%m%d_%H%M%S)
```


## Installation

The Python environment is managed with `uv`:

```bash
uv sync
```

Run Python commands through the managed environment:

```bash
uv run python scripts/preprocessing/tab_file_processor.py --help
```

### R Installation (Required for Continental Lithospheric Biomass)

The Python `uv` environment does not install R or R packages. Install R and the
packages required by the Magnabosco continental scripts before running the
continental workflows in Step 5. Verify the local R environment with:

```bash
Rscript scripts/continental/check_environment.R
```

The scripts require `foreach`, `doParallel`, `glmnet`, `fields`, `nlstools`,
and `ggplot2`. Each completed continental run saves `r_session_info.txt` with
the R version and loaded-package provenance alongside its numerical outputs.

## Workflow


### 1. Geothermal-Gradient Models

Pre-trained geothermal-gradient models are stored at:

```text
runs/geothermal/1stAttempt/oceanic_final/myModel1st.model
runs/geothermal/1stAttempt/continental_final/myModel1st.model
```

You can skip directly to inference in Step 2. Retrain only when you need to
regenerate the model files.

### 2. Run Geothermal-Gradient Inference

```bash
uv run python scripts/geothermal/baseline_xgboost.py \
  --Attempt 1st \
  --if_inference \
  --data_path ../../data/raw/geothermal_model_final_data/split_ocean_1x1.csv \
  --omodel_path 1stAttempt/oceanic_final/myModel1st.model \
  --cmodel_path 1stAttempt/continental_final/myModel1st.model
```

This writes `runs/geothermal/1stAttempt/total_oceanic.csv` and `runs/geothermal/1stAttempt/total_continental.csv`.

### 3. Calculate Habitable Lithospheric Volume

```bash
uv run python scripts/volume/habitable_volume.py \
  --continental_file runs/geothermal/1stAttempt/total_continental.csv \
  --oceanic_file runs/geothermal/1stAttempt/total_oceanic.csv \
  --mast_file data/processed/mast/global_mean_temperature_1deg.csv \
  --temperature 122
```

New volume outputs are written to `runs/volume/$BIOMASS_RUN_ID/`, or to a new
timestamped folder under `runs/volume/` when `BIOMASS_RUN_ID` is unset.

### 4. Run Oceanic Lithospheric Biomass Estimates

Always pass either `--exclude-shallow` or `--include-shallow` when reproducing
the manuscript oceanic biomass workflows.

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

Oceanic scripts read `data/raw/oceanic/oceanic_cell_densities.xlsx`, `data/raw/oceanic/ecm/ECM1.txt`, and the volume table in `runs/volume/submitted/results/`.

### 5. Run Continental Lithospheric Biomass Estimates

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

Use `--run-id NAME` to assign one shared output identifier to all R scripts in
the selected workflow. This option takes precedence over `BIOMASS_RUN_ID`.

If `--run-id` is omitted, the wrapper uses `BIOMASS_RUN_ID` when it is set;
otherwise it creates one timestamp automatically. Outputs are written to:

`runs/continental/<run-id>/<workflow>/`

When running an R script directly rather than through the Python wrapper, set
`BIOMASS_RUN_ID` yourself to keep separately invoked scripts in the same output
directory.


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

The common test suite covers geothermal-gradient model training/inference,
run-directory and CSV I/O helpers, habitable-volume calculation, and core
oceanic biomass fitting utilities. Continental R workflows are validated by
running their documented workflow commands in an R-enabled environment.

### Optional: Process Mean Annual Surface Temperature

The habitable-volume calculation requires
`data/processed/mast/global_mean_temperature_1deg.csv`. It is included with
the repository. To regenerate it from the supplied ERA5 input:

```bash
uv run python scripts/preprocessing/process_mast_file.py \
  --input data/raw/mast/era5_2024_monthly.nc \
  --regridded-output data/processed/mast/global_mean_temperature_1deg.csv
```

The optional ERA5 download command is `uv run python
scripts/preprocessing/mast_get_by_cds.py`; it requires configured CDS
credentials.
