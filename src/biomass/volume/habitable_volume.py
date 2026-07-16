# -*- coding: UTF-8 -*-
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from biomass.io import PROJECT_ROOT
from biomass.preprocessing.process_mast_file import _lat_to_1deg_center, _lon_to_1deg_center

# resolution is not certain
# drop elevation in the features

class lithoVolume:
    '''calculate the subsurface habitable volume
       assuming that the gradient is a constant at each certain grid
    '''

    def __init__(self,args):
        self.args = args
        self.rmse_g = {'continental': 8.656,   # °C km-1
                       'oceanic':     28.304}  # °C km-1

    def calcutor(self, resolution, gradient_file, mast_file, temperature, domain, output_dir):
        if domain not in self.rmse_g:
            raise ValueError(f"Unknown lithospheric domain: {domain}")

        rmse = self.rmse_g[domain] # °C km-1
        df = pd.read_csv(gradient_file)
        required_gradient_columns = {"lat", "lon", "gradient"}
        missing_gradient_columns = required_gradient_columns.difference(df.columns)
        if missing_gradient_columns:
            raise ValueError(
                f"Gradient file {gradient_file} is missing columns: "
                f"{sorted(missing_gradient_columns)}"
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gradient = df['gradient'].astype(float) + 0.1

        # Clip gradient to 1st and 99th percentile
        lower = np.percentile(gradient, 1)
        upper = np.percentile(gradient, 99)
        print(f"Clipping gradient between {lower:.3f} and {upper:.3f}")
        gradient = np.clip(gradient, lower, upper)  

        if domain == 'continental':
            df_mast = pd.read_csv(mast_file)
            required_mast_columns = {"Latitude", "Longitude", "Mean_Temperature_C"}
            missing_mast_columns = required_mast_columns.difference(df_mast.columns)
            if missing_mast_columns:
                raise ValueError(
                    f"MAST file {mast_file} is missing columns: {sorted(missing_mast_columns)}"
                )

            # Both sources are treated as 1 degree grid cells. Normalising here
            # avoids fragile float equality and reconciles 0--360 with -180--180 longitude.
            mast_lookup = df_mast.assign(
                _mast_lat=df_mast["Latitude"].map(_lat_to_1deg_center),
                _mast_lon=df_mast["Longitude"].map(_lon_to_1deg_center),
            )
            mast_lookup = (
                mast_lookup.groupby(["_mast_lat", "_mast_lon"], as_index=False)["Mean_Temperature_C"]
                .mean()
                .rename(columns={"Mean_Temperature_C": "_mast_temperature"})
            )
            df["_mast_lat"] = df["lat"].map(_lat_to_1deg_center)
            df["_mast_lon"] = df["lon"].map(_lon_to_1deg_center)
            df = df.merge(mast_lookup, on=["_mast_lat", "_mast_lon"], how="left", validate="many_to_one")
            unmatched = df["_mast_temperature"].isna()
            if unmatched.any():
                examples = df.loc[unmatched, ["lat", "lon", "_mast_lat", "_mast_lon"]].head(5)
                raise ValueError(
                    f"No MAST temperature for {int(unmatched.sum())} continental grid cells. "
                    f"Example coordinates:\n{examples.to_string(index=False)}"
                )
            surface_temperature = df["_mast_temperature"].to_numpy(dtype=float)
        else:
            surface_temperature = np.full(len(df), 4.0, dtype=float)

        gradient = np.asarray(gradient, dtype=float)
        latitudes = df["lat"].to_numpy(dtype=float)
        lon_len = 111.32 * np.abs(np.cos(np.radians(latitudes))) * resolution
        square_km2 = lon_len * 111.32 * resolution
        depth_km = (temperature - surface_temperature) / gradient
        depth_sd = (temperature - surface_temperature) / gradient**2 * rmse

        df["maxdepth"] = depth_km
        df["maxdepth_sd"] = depth_sd
        df["volume"] = square_km2 * depth_km
        volume_sum = float(df["volume"].sum())
        df.drop(columns=["_mast_lat", "_mast_lon", "_mast_temperature"], errors="ignore", inplace=True)

        df.to_csv(output_dir / ("inference_and_depth_to_%.1f_calculation_%s.csv" % (temperature, domain)), index=False)
        print('The %s lithospheric volume is %.5f km^3' % (domain, volume_sum))
        with open(output_dir / f"{domain}_habitable_volume_result.txt", "w") as f:
            f.write(f'The {domain} lithospheric volume is {volume_sum:.5f} km^3\n')

        return volume_sum

def parse_opt():
    run_id = os.environ.get("BIOMASS_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=float, default=1, help='the resolution of bin method')
    parser.add_argument('--continental_file', type=str, default=str(PROJECT_ROOT / 'runs/geothermal/1stAttempt/total_continental.csv'), help='the path of continental gradient file')
    parser.add_argument('--oceanic_file', type=str, default=str(PROJECT_ROOT / 'runs/geothermal/1stAttempt/total_oceanic.csv'), help='the path of oceanic gradient file')
    parser.add_argument('--temperature', type=float, default=122, help='the extreme temperature of life')
    parser.add_argument('--mast_file', type=str, default=str(PROJECT_ROOT / 'data/processed/mast/global_mean_temperature_1deg.csv'), help='the path of mast file')
    parser.add_argument('--output_dir', type=str, default=str(PROJECT_ROOT / 'runs/volume' / run_id), help='directory for generated volume outputs')
    return parser.parse_known_args()[0]

def earth_propotion(value):
    '''in percent form,
       the earth volume is about 1.082*10**12
    '''
    propotion = value/(1.082*(10**12))
    return propotion*100

def crust_propotion(value):
    '''crust volume is calculated by moho depth
    '''
    propotion = value/(12476861831.976618)
    return propotion*100

def main():
    args = parse_opt()
    resolution = args.resolution
    continental_file = args.continental_file
    oceanic_file = args.oceanic_file
    temperature = args.temperature  
    litho_vol = lithoVolume(args)
    mast_file = args.mast_file
    output_dir = args.output_dir

    con_volume = litho_vol.calcutor(resolution, continental_file, mast_file, temperature, 'continental', output_dir)
    con_pro_earth = earth_propotion(con_volume)
    con_pro_crust = crust_propotion(con_volume)

    oce_volume = litho_vol.calcutor(resolution, oceanic_file, mast_file, temperature, 'oceanic', output_dir)
    oce_pro_earth = earth_propotion(oce_volume)
    oce_pro_crust = crust_propotion(oce_volume)
    text = np.vstack((con_volume, con_pro_earth, con_pro_crust, oce_volume, oce_pro_earth, oce_pro_crust )).T
    np.savetxt(Path(output_dir) / ('1deglithospheric_volume_%.2f.txt'%args.temperature), text, fmt='%.5f', 
            header='con_volume, con_pro_earth(%), con_pro_crust(%), oce_volume, oce_pro_earth(%), oce_pro_crust(%)')


if __name__ == '__main__':
    main()
