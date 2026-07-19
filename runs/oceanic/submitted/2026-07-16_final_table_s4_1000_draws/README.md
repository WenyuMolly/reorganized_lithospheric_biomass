# Final Oceanic Table S4 Reference Outputs

This directory is the curated, machine-readable reference output for the
revision's oceanic Table S4. It was assembled on 2026-07-16 from eight
1,000-draw Monte Carlo total-summary files supplied in
`oceanic_biomass_results_temp/`.

## Contents

- `oceanic_biomass_global_method_comparison.csv` is the canonical numeric
  table, in cells.
- `oceanic_biomass_table_s4_style_x10e27_cells.csv` is the Table S4 display
  table, in units of `1e27` cells, with means and 95% percentile intervals.
- `source_totals/` contains the eight source total-summary CSV files used to
  build the two tables.
- `source_totals_manifest.csv` maps every method/dataset pair to its supplied
  source path and SHA-256 checksum.

## Methods and datasets

The table contains stratified and unstratified bootstrap (log10) estimates and
stratified and unstratified power-law estimates, for Shallow-Excluded and
Shallow-Included cell-density datasets. All source summaries report 1,000
draws.

The Shallow-Excluded total means, in `1e27` cells, are 9.347 (stratified
bootstrap), 4.937 (unstratified bootstrap), 15.547 (stratified power-law), and
9.048 (unstratified power-law). These are the reference values underlying the
rounded main-text Table S4 totals of 9.35, 4.94, 15.55, and 9.05.
