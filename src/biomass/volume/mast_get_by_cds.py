import cdsapi

# dataset = "insitu-gridded-observations-global-and-regional"
# request = {
#     "origin": "cru",
#     "region": "global",
#     "variable": ["temperature"],
#     "statistic": ["mean"],
#     "time_aggregation": "monthly",
#     "horizontal_aggregation": ["1_x_1"],
#     "year": ["2019"],
#     "version": ["v4_03"]
# }

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()

# import cdsapi

# # Initialize CDS API client
# client = cdsapi.Client()

# # Define request parameters
# dataset = "insitu-gridded-observations-global-and-regional"

# request = {
#     "product_type": "temperature",
#     "variable": "surface_air_temperature",
#     "year": "2024",  # Change to the desired year
#     "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],  # All months
#     "grid_resolution": "1x1",  # 1-degree resolution
#     "statistic": "annual_mean",  # Get yearly mean instead of monthly
#     "format": "netcdf",  # Output format: NetCDF
# }

# # Download the dataset
# client.retrieve(dataset, request, "cds_2024_global_surface_temperature.nc")


# import cdsapi

# client = cdsapi.Client()

# client.retrieve(
#     "reanalysis-era5-single-levels-monthly-means",  # Use ERA5 monthly means
#     {
#         "variable": ["2m_temperature"],  # Near-surface air temperature
#         "year": "2024",  # Change to the desired year
#         "month": [f"{m:02d}" for m in range(1, 13)],  # Request all 12 months
#         "time": "00:00",
#         "format": "netcdf",
#     },
#     "era5_2024_monthly.nc"
# )
import cdsapi

# Initialize CDS API client
client = cdsapi.Client()

# Request bottom seawater temperature from ORAS5
client.retrieve(
    "reanalysis-ora5",  # ORAS5 Ocean Reanalysis dataset
    {
        "variable": "sea_water_potential_temperature",  # Temperature variable
        "depth": "bottom",  # Select bottom layer
        "year": "2024",  # Year of interest
        "month": [f"{m:02d}" for m in range(1, 13)],  # Request all 12 months
        "time": "00:00",  # Midnight data
        "format": "netcdf",  # Download format
    },
    "oras5_2024_bottom_temperature.nc"  # Save file as NetCDF
)

