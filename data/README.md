# Data provenance and redistribution

This directory contains immutable source inputs (`raw/`) and curated inputs
used by the published workflows (`processed/`). Generated model, volume, and
biomass outputs belong under `runs/`, not under this directory.

## Source inputs

- `raw/mast/era5_2024_monthly.nc` is the ERA5 monthly 2 m air-temperature
  input used to create the annual mean surface-temperature product. It is
  obtained through the Copernicus Climate Data Store; users must follow the
  applicable Copernicus licence and citation requirements.
- `raw/mast/CRU_mean_temperature_mon_1x1_global_2019_v4.03.nc` is a CRU
  climate input retained for the legacy workflow. Users must follow the
  applicable CRU data terms and citation requirements.
- `raw/oceanic/ecm/ECM1.txt` is the Earth Crustal Model input used for
  oceanic layer geometry. Field definitions are provided in the adjacent
  `README_ECM1.txt`; users should cite and comply with the terms of the
  original ECM1 distribution.
- `raw/oceanic/pangaea_exp357/tab_files/` contains Expedition 357 data files
  downloaded from PANGAEA. Each TAB file retains its source DOI, attribution,
  and licence in the header. For example, the included files identify a
  CC-BY-3.0 licence.
- `raw/oceanic/oceanic_cell_densities.xlsx` and
  `raw/continental/cores_with_PCR.csv` are curated literature-derived cell
  density inputs. Citations are retained in their source tables where
  available.
- `raw/geothermal_model_final_data/split_ocean_1x1.csv` is the gridded
  predictor table used by the geothermal-model workflow.

## Curated inputs

`processed/continental/` and `processed/mast/` contain the curated tables
consumed directly by the continental and habitable-volume workflows. They are
included so that the documented workflows can be rerun without regenerating
every intermediate product.

## Redistribution note

Project-authored material is released under CC BY 4.0, but several inputs above
are third-party data products. The repository-level licence does not replace
their individual data licences or citation requirements. Confirm the
redistribution terms for ECM1, ERA5, CRU, and every curated literature dataset
before publishing a public archive.
