# Data provenance and redistribution

This directory contains immutable source inputs (`raw/`) and curated inputs
used by the published workflows (`processed/`). Generated model, volume, and
biomass outputs belong under `runs/`, not under this directory.

## Source inputs

- `raw/mast/era5_2024_monthly.nc` is the ERA5 monthly 2 m air-temperature
  input used to create the annual mean surface-temperature product. It is
  obtained through the Copernicus Climate Data Store. Cite: Hersbach, H. et
  al. (2020), The ERA5 global reanalysis, *Quarterly Journal of the Royal
  Meteorological Society*, 146, 1999-2049,
  https://doi.org/10.1002/qj.3803; and the ERA5 monthly averaged data on
  single levels dataset, https://doi.org/10.24381/cds.f17050d7. Users must
  also follow the applicable Copernicus licence and citation requirements.
- `raw/oceanic/ecm/ECM1.txt` is the Earth Crustal Model input used for
  oceanic layer geometry. Field definitions are provided in the adjacent
  `README_ECM1.txt`. Cite: Mooney, W. D., Barrera-Lopez, C., Suarez, M. G.,
  and Castelblanco, M. A. (2023), Earth Crustal Model 1 (ECM1): A 1 degree x
  1 degree global seismic and density model, *Earth-Science Reviews*, 243,
  104493, https://doi.org/10.1016/j.earscirev.2023.104493. Users should also
  comply with the terms of the original ECM1 distribution.
- `raw/oceanic/pangaea_exp357/tab_files/` contains Expedition 357 data files
  downloaded from PANGAEA. Each TAB file retains its source DOI, attribution,
  and licence in the header. For example, the included files identify a
  CC-BY-3.0 licence.
- `raw/oceanic/oceanic_cell_densities.xlsx` is a curated literature-derived
  cell-density input. The `links` column records the source information for
  each observation.
- `raw/continental/cores_with_PCR.csv` is the curated continental
  literature-derived cell-density input used by the Magnabosco workflows.
- `raw/geothermal_model_final_data/split_ocean_1x1.csv` is the gridded
  predictor table used by the geothermal-model workflow. The original source
  for each predictor is listed in Table 2 of the manuscript.

## Curated inputs

`processed/continental/` and `processed/mast/` contain the curated tables
consumed directly by the continental and habitable volume workflows. In
particular, `processed/mast/global_mean_temperature_2024.csv` is the annual
mean ERA5 table and `processed/mast/global_mean_temperature_1deg.csv` is its
1 degree regridded version used by the habitable volume workflow. These files
are included so that the documented workflows can be rerun without regenerating
every intermediate product.

## Redistribution note

Project-authored material is released under CC BY 4.0, but several inputs above
are third-party data products. The repository level licence does not replace
their individual data licences or citation requirements. Confirm the
redistribution terms for ECM1, ERA5, and every curated literature dataset
before publishing a public archive.
