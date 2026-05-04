
import os
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin

import pandas as pd, numpy as np
import geopandas as gpd

from shapely.geometry import Point
from rasterstats import zonal_stats
import rasterio


def download_all_csv(folder_name='data'):
    url = "https://opendata.apps.mow.vlaanderen.be/fietstellingen/index.html"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)
    csv_links = [urljoin(url, link["href"]) for link in links if link["href"].endswith(".csv")]
    if not csv_links:
        print("No CSV files found.")
        return
    print(f"Found {len(csv_links)} CSV files.")
    for csv_url in csv_links:
        file_name = os.path.join(folder_name, os.path.basename(csv_url))
        print(f"Downloading {csv_url} -> {file_name}")
        r = requests.get(csv_url)
        with open(file_name, "wb") as f:
            f.write(r.content)
    print("Download complete.")

def time_of_day(dt):
    hour = dt.hour
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'

def df2gdf(df, lat_col='latitude', long_col='longtitude', crs="EPSG:4326"):
    """
    Convert a DataFrame to a GeoDataFrame with points from latitude and longitude columns.

    Args:
        df: Input dataframe.
        lat_col (str): Name of latitude column.
        long_col (str): Name of longitude column.
        crs (str): CRS string.

    Returns:
        gpd.GeoDataFrame: Resulting GeoDataFrame.
    """
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[long_col], df[lat_col]),
        crs=crs
    )

def compute_site_population_buffers(
    df_sites,
    raster_path="data/bel_pop_2025_CN_100m_R2025A_v1.tif",
    rad_values=[1000, 5000, 10000],
    long_col="longtitude",
    lat_col="latitude",
    buffer_crs="EPSG:31370",
    ):
    """
    Compute site population sums within specified buffer radii for given sites.
    
    Parameters:
    - df_sites: DataFrame with 'longtitude' and 'latitude' columns at minimum
    - raster_path: path to population raster file
    - rad_values: list of buffer radii in meters
    - buffer_crs: CRS string for buffer (meters; default is Belgian CRS)
    
    Returns:
    - GeoDataFrame with population columns added
    """

    # Create sites as GeoDataFrame in WGS84
    gdf_sites = gpd.GeoDataFrame(
        df_sites.copy(),
        geometry=[Point(xy) for xy in zip(df_sites[long_col], df_sites[lat_col])],
        crs="EPSG:4326", # crs='EPSG:4326' means coordinates are in WGS84 latitude/longitude
    )

    # Determine raster CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    for rad in rad_values:
        # Buffer in meters with projection suited for distances
        gdf_sites_m = gdf_sites.to_crs(buffer_crs)
        gdf_buf_m = gdf_sites_m.copy()
        gdf_buf_m["geometry"] = gdf_buf_m.geometry.buffer(rad)

        # Reproject buffer to raster's CRS
        gdf_buf_for_raster = gdf_buf_m.to_crs(raster_crs)
        stats = zonal_stats(
            gdf_buf_for_raster.geometry,
            raster_path,
            stats=["sum"],
            all_touched=False,
        )
        gdf_sites[f"pop_{int(rad/1000)}km"] = [
            0 if s.get("sum") is None else float(s["sum"]) for s in stats
        ]

    return gdf_sites
