from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd

from biomass.volume.habitable_volume import lithoVolume
from biomass.preprocessing.process_mast_file import _lat_to_1deg_center, _lon_to_1deg_center


def test_mast_regrid_centers_do_not_shift_longitude_by_180_degrees():
    assert _lon_to_1deg_center(0.0) == 0.5
    assert _lon_to_1deg_center(0.25) == 0.5
    assert _lon_to_1deg_center(179.75) == 179.5
    assert _lon_to_1deg_center(180.0) == -179.5
    assert _lon_to_1deg_center(359.75) == -0.5


def test_mast_regrid_latitude_centers_clip_poles_to_valid_1deg_grid():
    assert _lat_to_1deg_center(90.0) == 89.5
    assert _lat_to_1deg_center(89.75) == 89.5
    assert _lat_to_1deg_center(-89.75) == -89.5
    assert _lat_to_1deg_center(-90.0) == -89.5


def test_habitable_volume_calculation_writes_expected_outputs(tmp_path):
    gradient_file = tmp_path / "gradients.csv"
    mast_file = tmp_path / "mast.csv"
    output_dir = tmp_path / "volume_outputs"

    pd.DataFrame(
        {
            "lat": [0.5, 1.5],
            "lon": [10.5, 11.5],
            "gradient": [30.0, 40.0],
        }
    ).to_csv(gradient_file, index=False)

    pd.DataFrame(
        {
            "Latitude": [0.5, 1.5],
            "Longitude": [10.5, 11.5],
            "Mean_Temperature_C": [15.0, 12.0],
        }
    ).to_csv(mast_file, index=False)

    calculator = lithoVolume(Namespace())
    volume = calculator.calcutor(
        resolution=1.0,
        gradient_file=gradient_file,
        mast_file=mast_file,
        temperature=122.0,
        domain="continental",
        output_dir=output_dir,
    )

    output_table = output_dir / "inference_and_depth_to_122.0_calculation_continental.csv"
    assert volume > 0
    assert output_table.exists()
    assert (output_dir / "continental_habitable_volume_result.txt").exists()

    result = pd.read_csv(output_table)
    assert {"maxdepth", "maxdepth_sd", "volume"}.issubset(result.columns)
    assert np.isfinite(result["volume"]).all()
    assert (result["maxdepth"] > 0).all()
