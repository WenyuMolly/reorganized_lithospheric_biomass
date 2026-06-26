# -*- coding: UTF-8 -*-
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
        volume_sum = 0
        rmse = self.rmse_g[domain] # °C km-1
        df = pd.read_csv(gradient_file)
        df_mast = pd.read_csv(mast_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df['maxdepth'] = np.nan
        df['maxdepth_sd'] = np.nan
        df['volume'] = np.nan
        gradient = df['gradient'].map(lambda x: x+0.1)

        # Clip gradient to 1st and 99th percentile
        lower = np.percentile(gradient, 1)
        upper = np.percentile(gradient, 99)
        print(f"Clipping gradient between {lower:.3f} and {upper:.3f}")
        gradient = np.clip(gradient, lower, upper)  

        for i, g in enumerate(gradient):
            lat, lon = df.loc[i, ['lat', 'lon']]
            lonLen = 111.32*(abs(math.cos(math.radians(df.loc[i,'lat']))))*resolution
            square_km2  = lonLen*111.32*resolution
            mast = df_mast.loc[
                (df_mast['Latitude'] == lat) & (df_mast['Longitude'] == lon),
                'Mean_Temperature_C'
            ].iloc[0]
            T0 = mast if domain == 'continental' else 4.0
            depth_km = (temperature - T0) / g
            # propagate gradient RMSE to depth SD
            depth_sd = (temperature - T0) / g**2 * rmse
            
            df.at[i, 'maxdepth'] = depth_km
            df.at[i, 'maxdepth_sd'] = depth_sd
            cell_vol = square_km2 * depth_km
            df.at[i, 'volume']  = cell_vol
            volume_sum += cell_vol

        df.to_csv(output_dir / ("inference_and_depth_to_%.1f_calculation_%s.csv" % (temperature, domain)), index=False)
        print('The %s lithospheric volume is %.5f km^3' % (domain, volume_sum))
        with open(output_dir / f"{domain}_habitable_volume_result.txt", "w") as f:
            f.write(f'The {domain} lithospheric volume is {volume_sum:.5f} km^3\n')

        return volume_sum

def parse_opt():
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=float, default=1, help='the resolution of bin method')
    parser.add_argument('--continental_file', type=str, default=str(project_root / 'runs/geothermal/1stAttempt/total_continental.csv'), help='the path of continental gradient file')
    parser.add_argument('--oceanic_file', type=str, default=str(project_root / 'runs/geothermal/1stAttempt/total_oceanic.csv'), help='the path of oceanic gradient file')
    parser.add_argument('--temperature', type=float, default=122, help='the extreme temperature of life')
    parser.add_argument('--mast_file', type=str, default=str(project_root / 'data/processed/mast/global_mean_temperature_1deg.csv'), help='the path of mast file')
    parser.add_argument('--output_dir', type=str, default=str(project_root / 'runs/volume/latest'), help='directory for generated volume outputs')
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
