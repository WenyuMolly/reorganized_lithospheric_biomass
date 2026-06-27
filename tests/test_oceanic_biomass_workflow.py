from __future__ import annotations

import numpy as np
import pandas as pd

from biomass.oceanic.stratified_power_fit import (
    fit_power_law_log10 as fit_stratified_power_law_log10,
    fit_power_or_constant,
)
from biomass.oceanic.unstratified_power_fit import (
    convert_cells_per_g_to_cm3,
    detect_depth_column_and_to_km,
    draw_z122_depth_km,
    fit_power_law_log10 as fit_unstratified_power_law_log10,
    fit_unstratified_power_or_constant,
    grid_area_cm2,
)


def test_oceanic_density_unit_conversion_and_depth_detection():
    santelli = pd.Series({"Reference": "Santelli et al. 2008", "Cell Count": 100.0})
    meyers = pd.Series({"Reference": "Jacobson Meyers et al. 2014", "Cell Count": 100.0})
    direct = pd.Series({"Reference": "Other", "Cell Count": 100.0})

    assert convert_cells_per_g_to_cm3(santelli) == 277.0
    assert convert_cells_per_g_to_cm3(meyers) == 290.0
    assert convert_cells_per_g_to_cm3(direct) == 100.0

    depth_km, unit = detect_depth_column_and_to_km(
        pd.DataFrame({"Depth for Power Fit": [100.0, 200.0, 300.0]})
    )
    assert unit == "m"
    np.testing.assert_allclose(depth_km.to_numpy(), [0.1, 0.2, 0.3])


def test_oceanic_power_law_fits_return_finite_parameters():
    depth = np.array([0.1, 0.3, 0.8, 1.5, 3.0])
    density = np.array([1e5, 7e4, 2e4, 8e3, 2e3])

    unstratified = fit_unstratified_power_or_constant(depth, density)
    stratified = fit_power_or_constant(depth, density, domain_name="Upper Crust")

    assert unstratified["model"] == "power"
    assert stratified["model"] == "power"
    assert np.isfinite(unstratified["A"])
    assert np.isfinite(stratified["A"])
    assert grid_area_cm2(0.0) > grid_area_cm2(60.0)


def test_oceanic_power_law_covariance_matches_returned_a_b_order():
    depth = np.array([0.1, 0.3, 0.8, 1.5, 3.0])
    density = np.array([1e5, 7e4, 2e4, 8e3, 2e3])
    lx = np.log10(depth)
    ly = np.log10(density)
    _, cov_ba = np.polyfit(lx, ly, deg=1, cov=True)
    expected_cov_ab = cov_ba[::-1, ::-1]

    _, _, unstratified_cov, _, _ = fit_unstratified_power_law_log10(depth, density)
    _, _, stratified_cov, _, _ = fit_stratified_power_law_log10(depth, density)

    np.testing.assert_allclose(unstratified_cov, expected_cov_ab)
    np.testing.assert_allclose(stratified_cov, expected_cov_ab)


def test_oceanic_z122_scenarios_use_standard_deviation_bounds():
    row = pd.Series({"maxdepth": 2.0, "maxdepth_sd": 0.5})
    shallow_row = pd.Series({"maxdepth": 0.4, "maxdepth_sd": 0.8})
    rng = np.random.default_rng(42)

    assert draw_z122_depth_km(row, rng, "low") == 1.5
    assert draw_z122_depth_km(row, rng, "base") == 2.0
    assert draw_z122_depth_km(row, rng, "high") == 2.5
    assert draw_z122_depth_km(shallow_row, rng, "low") == 0.0
