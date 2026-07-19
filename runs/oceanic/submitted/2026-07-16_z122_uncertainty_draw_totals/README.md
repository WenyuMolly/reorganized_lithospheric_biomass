# Oceanic z122 uncertainty reference output

This directory is the compact, versioned reference output for the oceanic
geothermal z122 uncertainty analysis. It was curated from the 24 scenario
runs supplied on 2026-07-16: two cell-density datasets (`without_shallow`,
`with_shallow`), four biomass methods, and three z122 surfaces (`low`,
`base`, `high`).

## Contents

- `draw_totals/`: 24 files of 1,000 paired Monte Carlo totals. These are the
  data needed to recompute scenario means, intervals, and paired low/base/high
  changes without retaining per-grid or depth-bin matrices.
- `total_summaries/`: the 24 source global-total summary files copied from the
  scenario runs.
- `oceanic_geothermal_z122_scenario_summary.csv`: total-cell summaries
  recalculated directly from the archived draw totals.
- `oceanic_geothermal_z122_paired_summary.csv`: paired changes between the
  aligned low, base, and high draws for each dataset and method.
- `source_manifest.csv`: original relative path and SHA-256 checksum for every
  archived file.

## Interpretation boundary

For each z122 scenario, the z122 surface is fixed at the selected low, base,
or high value per grid cell; the biomass-model Monte Carlo draws remain in the
calculation. The `base` scenario therefore represents the deterministic
baseline z122 surface, not the main Table S4 workflow, which additionally
draws z122 depth from each grid cell's `maxdepth_sd`. A base result can thus
differ from Table S4 without indicating a calculation error.

The obsolete top-level `tables/` summaries from the temporary source directory
are deliberately not archived because they predate the corrected
draw-total-based aggregation. Use only the tables in this directory for the
revision.
