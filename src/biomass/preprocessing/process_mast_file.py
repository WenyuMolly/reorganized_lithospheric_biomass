# -*- coding: UTF-8 -*-

import argparse
from pathlib import Path

from biomass.io import PROJECT_ROOT

SCRIPT_DIR = Path(__file__).resolve().parent


def _temperature_var_name(ds):
    """Return the expected ERA5 2 m temperature variable name."""
    if "t2m" in ds.data_vars:
        return "t2m"
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise RuntimeError(f"Could not identify temperature variable. Available variables: {list(ds.data_vars)}")


def _time_dim_name(data_array):
    """Return the time dimension used by CDS/ERA5 files."""
    for candidate in ("valid_time", "time"):
        if candidate in data_array.dims:
            return candidate
    raise RuntimeError(f"Could not identify time dimension. Available dimensions: {data_array.dims}")


def _to_celsius(data_array, source_var):
    units = str(source_var.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin"}:
        return data_array - 273.15
    return data_array


def _lon_to_1deg_center(value):
    """Map longitude to 1 degree cell center in [-179.5, 179.5]."""
    import numpy as np

    lon = ((float(value) + 180.0) % 360.0) - 180.0
    return float(np.floor(lon) + 0.5)


def _lat_to_1deg_center(value):
    """Map latitude to 1 degree cell center in [-89.5, 89.5]."""
    import numpy as np

    return float(np.clip(np.floor(float(value)) + 0.5, -89.5, 89.5))


def plot_mean_temperature(file_path=PROJECT_ROOT / "data/raw/mast/era5_2024_monthly.nc", output_png=PROJECT_ROOT / "figures/generated/era5_2024_monthly.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from mpl_toolkits.basemap import Basemap

    ds = xr.open_dataset(file_path)
    variable_name = _temperature_var_name(ds)
    time_dim = _time_dim_name(ds[variable_name])
    mean_temp = ds[variable_name].mean(dim=time_dim)
    mean_temp_celsius = _to_celsius(mean_temp, ds[variable_name])

    # Extract lat/lon
    lat = ds["latitude"].values
    lon = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon = lon[order]
    mean_temp_celsius = mean_temp_celsius.isel(longitude=order)

    # Create global map
    fig, ax = plt.subplots(figsize=(12, 6))
    m = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax)
    m.drawcoastlines()
    m.drawcountries()

    # Convert to 2D grid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    sc = m.pcolormesh(lon_grid, lat_grid, mean_temp_celsius, cmap="coolwarm", shading="auto", latlon=True)

    # Add colorbar
    cbar = plt.colorbar(sc, orientation="horizontal", pad=0.05)
    cbar.set_label("Annual Mean 2m Temperature (°C)")

    # Set title
    plt.title("Global Annual 2m Temperature (Land + Ocean) - 2024")

    # Show plot
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png)

    # Close dataset
    ds.close()


# def plot_mean_oceanic_temperature():
#     # Load the NetCDF file
#     ds = xr.open_dataset("oras5_2024_bottom_temperature.nc")

#     # Check available variables
#     print(ds)

#     # Extract temperature variable (adjust based on dataset structure)
#     variable_name = "thetao"  # ORAS5 uses "thetao" for seawater temperature
#     bottom_temp = ds[variable_name].mean(dim="time")  # Compute annual mean

#     # Extract lat/lon
#     lat = ds["latitude"].values
#     lon = ds["longitude"].values

#     # Create global map
#     fig, ax = plt.subplots(figsize=(12, 6))
#     m = Basemap(projection="cyl", llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax)
#     m.drawcoastlines()
#     m.drawcountries()

#     # Convert to 2D grid
#     lon_grid, lat_grid = np.meshgrid(lon, lat)
#     sc = m.pcolormesh(lon_grid, lat_grid, bottom_temp, cmap="coolwarm", shading="auto", latlon=True)

#     # Add colorbar
#     cbar = plt.colorbar(sc, orientation="horizontal", pad=0.05)
#     cbar.set_label("Annual Mean Bottom Seawater Temperature (°C)")

#     # Set title
#     plt.title("Global Annual Mean Seafloor Temperature - 2024")

#     # Show plot
#     plt.show()

#     # Close dataset
#     ds.close()


def save_mean_data(file_path=PROJECT_ROOT / "data/raw/mast/era5_2024_monthly.nc", output_csv=PROJECT_ROOT / "data/processed/mast/global_mean_temperature_2024.csv"):
    import xarray as xr

    ds = xr.open_dataset(file_path)
    variable_name = _temperature_var_name(ds)
    time_dim = _time_dim_name(ds[variable_name])
    mean_temp = ds[variable_name].mean(dim=time_dim)
    mean_temp_celsius = _to_celsius(mean_temp, ds[variable_name])

    df = (
        mean_temp_celsius
        .to_dataframe(name="Mean_Temperature_C")
        .reset_index()
        .rename(columns={"latitude": "Latitude", "longitude": "Longitude"})
    )
    df = df[["Latitude", "Longitude", "Mean_Temperature_C"]]

    # Save to CSV file
    csv_filename = Path(output_csv)
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_filename, index=False)

    # Confirm CSV file saved successfully
    print(f"CSV file saved as: {csv_filename}")

    # Close dataset
    ds.close()
    
def regrid_mast(input_csv=PROJECT_ROOT / "data/processed/mast/global_mean_temperature_2024.csv", output_file=PROJECT_ROOT / "data/processed/mast/global_mean_temperature_1deg.csv"):
    import pandas as pd

    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Apply rounding function to latitude and longitude
    df["Lat_1deg"] = df["Latitude"].apply(_lat_to_1deg_center)
    df["Lon_1deg"] = df["Longitude"].apply(_lon_to_1deg_center)

    # Aggregate data by averaging temperatures within each 1° x 1° grid cell
    df_agg = df.groupby(["Lat_1deg", "Lon_1deg"])["Mean_Temperature_C"].mean().reset_index()

    # Rename columns for clarity
    df_agg.rename(columns={"Lat_1deg": "Latitude", "Lon_1deg": "Longitude"}, inplace=True)
    df_agg = df_agg.sort_values(["Latitude", "Longitude"]).reset_index(drop=True)

    # Save the processed data to a new CSV file
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_file, index=False)

    print(f"Processing complete. The regridded data has been saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process ERA5 mean annual surface temperature to a 1 degree CSV grid.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data/raw/mast/era5_2024_monthly.nc"), help="Input ERA5 NetCDF file.")
    parser.add_argument("--mean-output", default=str(PROJECT_ROOT / "data/processed/mast/global_mean_temperature_2024.csv"), help="Intermediate full-resolution annual-mean CSV.")
    parser.add_argument("--regridded-output", default=str(PROJECT_ROOT / "data/processed/mast/global_mean_temperature_1deg.csv"), help="Output 1 degree CSV used by habitable_volume.py.")
    parser.add_argument("--plot-output", default=str(PROJECT_ROOT / "figures/generated/era5_2024_monthly.png"), help="Output PNG for annual mean temperature.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip map plotting.")
    return parser.parse_args()


if __name__ == "__main__":
    # # Load the NetCDF file

    # # Print dataset summary
    # print("Dataset Information:")
    # print(ds)
    args = parse_args()
    save_mean_data(args.input, args.mean_output)
    if not args.skip_plot:
        plot_mean_temperature(args.input, args.plot_output)
    regrid_mast(args.mean_output, args.regridded_output)
