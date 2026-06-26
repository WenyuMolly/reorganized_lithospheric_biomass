# -*- coding: UTF-8 -*-

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent


def plot_mean_temperature(file_path=PROJECT_ROOT / "data/raw/mast/era5_2024_monthly.nc", output_png=PROJECT_ROOT / "figures/generated/era5_2024_monthly.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from mpl_toolkits.basemap import Basemap

    ds = xr.open_dataset(file_path)
    # Extract the correct temperature variable
    variable_name = "t2m"  # This is the correct variable name for 2m temperature

    # Compute the mean over time (use "valid_time" instead of "time")
    mean_temp = ds[variable_name].mean(dim="valid_time")  # Fixed error

    # Convert Kelvin to Celsius
    mean_temp_celsius = mean_temp - 273.15

    # Extract lat/lon
    lat = ds["latitude"].values
    lon = ds["longitude"].values

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
    import numpy as np
    import pandas as pd
    import xarray as xr

    ds = xr.open_dataset(file_path)

    # Extract the correct temperature variable
    variable_name = "t2m"  # Change based on your dataset (e.g., "t2m" for ERA5)
    mean_temp = ds[variable_name].mean(dim="valid_time")  # Compute the annual mean

    # Convert Kelvin to Celsius if necessary
    if "units" in ds[variable_name].attrs and ds[variable_name].attrs["units"] == "K":
        mean_temp_celsius = mean_temp - 273.15
    else:
        mean_temp_celsius = mean_temp  # Already in Celsius

    # Extract latitude and longitude values
    lat = ds["latitude"].values  # Change "lat" to "latitude" if needed
    lon = ds["longitude"].values  # Change "lon" to "longitude" if needed

    # Create a DataFrame with latitude, longitude, and mean temperature
    df = pd.DataFrame({
        "Latitude": np.repeat(lat, len(lon)),  # Repeat latitudes for each longitude
        "Longitude": np.tile(lon, len(lat)),   # Tile longitudes for each latitude
        "Mean_Temperature_C": mean_temp_celsius.values.flatten()  # Flatten the temperature values
    })

    # Save to CSV file
    csv_filename = Path(output_csv)
    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_filename, index=False)

    # Confirm CSV file saved successfully
    print(f"CSV file saved as: {csv_filename}")

    # Close dataset
    ds.close()
    
def regrid_mast(input_csv=PROJECT_ROOT / "data/processed/mast/global_mean_temperature_2024.csv", output_file=PROJECT_ROOT / "data/processed/mast/global_mean_temperature_1deg.csv"):
    import numpy as np
    import pandas as pd

    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Function to round values to the nearest 1° center grid (e.g., -89.5, -88.5, ..., 89.5)
    def round_to_1deg_lat_grid(value):
        return np.floor(value) + 0.5  # Centers values at the correct 1° resolution grid

    def round_to_1deg_lon_grid(value):
        return np.floor(value) + 0.5 - 180 # Centers values at the correct 1° resolution grid

    # Apply rounding function to latitude and longitude
    df["Lat_1deg"] = df["Latitude"].apply(round_to_1deg_lat_grid)
    df["Lon_1deg"] = df["Longitude"].apply(round_to_1deg_lon_grid)

    # Aggregate data by averaging temperatures within each 1° x 1° grid cell
    df_agg = df.groupby(["Lat_1deg", "Lon_1deg"])["Mean_Temperature_C"].mean().reset_index()

    # Rename columns for clarity
    df_agg.rename(columns={"Lat_1deg": "Latitude", "Lon_1deg": "Longitude"}, inplace=True)

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
