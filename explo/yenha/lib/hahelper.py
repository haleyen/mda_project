import os
import requests

from bs4 import BeautifulSoup
from urllib.parse import urljoin

import pandas as pd, numpy as np
import geopandas as gpd

from shapely.geometry import Point
from rasterstats import zonal_stats
import rasterio
import holidays

PATH_EXTRA_DATA     = "../../data/extra/"
PATH_TRAFFIC        = "../../data/traffic/"
FILEPATH_RICH       = "../../data/richtingen.csv"
FILEPATH_SITES      = "../../data/sites.csv"

HOME = "/Users/ponddie/Documents/py/mda_project/explo/yenha"
FILEPATH_FTS_POP    = HOME + '/data/fts_population.csv'
FILEPATH_FTS_WTHER  = HOME + '/data/fts_weather.csv'

TRAFFIC_COLS   = ["site_id", "direction", "type", "start_time", "end_time", "traffic"]
SITES_COLS  = ["site_id", "site_nr", "longitude", "latitude", "site_name", "domain", "road_id", "district", "municipality", "interval", "installation_date"]
RICH_COLS   = ["site_id", "direction", "name"]


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

# def time_of_day(dt):
#     hour = dt.hour
#     if 0 <= hour < 6:
#         return 'night'
#     elif 6 <= hour < 12:
#         return 'morning'
#     elif 12 <= hour < 18:
#         return 'afternoon'
#     else:
#         return 'evening'

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
        crs="EPSG:4326", # crs='EPSG:4326' means coordinates are in WGS84 latitude/longitude (in meters)
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

########################################################################
### FEATURE ENGINEERING ###

def compute_traffic_15min_features(df_raw):
    """
    Compute both within-hour 15-min traffic features and calendar+cyclical features.
    Returns a DataFrame with all features concatenated to the original df.
    Input: raw traffic file
    """

    df1 = df_raw.copy().query('type=="FIETSERS"')
    df1["start_time"] = pd.to_datetime(df1["start_time"])
    df1["date"] = df1["start_time"].dt.strftime("%Y-%m-%d")
    df1["hour"] = df1["start_time"].dt.hour
    df1["datetime"] = (pd.to_datetime(df1["date"]) + pd.to_timedelta(df1["hour"], unit="h"))
    df1["quarter"] = df1["start_time"].dt.minute // 15

    traffic_pivot = (
        df1.pivot_table(index=["site_id", "datetime", "date", "hour", "direction"], columns="quarter", values="traffic", aggfunc="sum")
        .reset_index()
        .rename(columns={0: "traffic_00_15", 1: "traffic_15_30", 2: "traffic_30_45", 3: "traffic_45_60"})
        )

    df1_hourly = (
        df1.groupby(["site_id", "datetime", "date", "hour", "direction"], as_index=False)
        .agg(count=("traffic", "sum"))
        .rename(columns={"count": "traffic"})
        )

    df1_hourly = (
        df1_hourly.merge(traffic_pivot, on=["site_id", "datetime", "date", "hour", "direction"], how="left")
        .sort_values(["site_id", "datetime"]).reset_index(drop=True)
        )

    df2 = df1_hourly.copy()
    cols = ["traffic_00_15", "traffic_15_30", "traffic_30_45", "traffic_45_60"]
    traffic_15min_mean = df2[cols].mean(axis=1, skipna=True)
    traffic_15min_std = df2[cols].std(axis=1, skipna=True)
    traffic_15min_min = df2[cols].min(axis=1, skipna=True)
    traffic_15min_max = df2[cols].max(axis=1, skipna=True)

    traffic_00_15_pct = df2["traffic_00_15"] / df2["traffic"].replace(0, np.nan)
    traffic_15_30_pct = df2["traffic_15_30"] / df2["traffic"].replace(0, np.nan)
    traffic_30_45_pct = df2["traffic_30_45"] / df2["traffic"].replace(0, np.nan)
    traffic_45_60_pct = df2["traffic_45_60"] / df2["traffic"].replace(0, np.nan)

    df2["traffic_15m_mean"] = traffic_15min_mean
    df2["traffic_15m_std"] = traffic_15min_std
    df2["traffic_15m_min"] = traffic_15min_min
    df2["traffic_15m_max"] = traffic_15min_max
    df2["traffic_00_15_pct"] = traffic_00_15_pct
    df2["traffic_15_30_pct"] = traffic_15_30_pct
    df2["traffic_30_45_pct"] = traffic_30_45_pct
    df2["traffic_45_60_pct"] = traffic_45_60_pct

    return df2

def _time_of_day(hour):
    if   0  <= hour <  5: return "night"
    elif 5  <= hour <  7: return "early_morning"
    elif 7  <= hour <  9: return "morning_rush"
    elif 9  <= hour < 12: return "mid_morning"
    elif 12 <= hour < 14: return "lunch"
    elif 14 <= hour < 16: return "afternoon"
    elif 16 <= hour < 19: return "evening_rush"
    elif 19 <= hour < 22: return "evening"
    else:                 return "late_night"

def compute_datetime_features(df_hour, col_datetime="datetime"):
    """Calendar + cyclical features at hourly resolution (no 15-min slot)."""
    df1 = df_hour.copy()

    t = df1[col_datetime]
    df1["day"] = t.dt.day
    df1["month"] = t.dt.month
    df1["year"] = t.dt.year
    df1["dow"] = t.dt.dayofweek  # 0=Mon, 6=Sun
    # df1["day_name"] = t.dt.day_name()
    df1["woy"] = t.dt.isocalendar().week.astype(int)
    df1["hour"] = t.dt.hour

    df1["is_weekend"] = (df1["dow"] >= 5).astype(int)
    df1["is_weekday"] = (df1["dow"] < 5).astype(int)
    df1["is_monday"] = (df1["dow"] == 0).astype(int)
    df1["is_friday"] = (df1["dow"] == 4).astype(int)

    df1["time_of_day"] = df1["hour"].map(_time_of_day)
    df1["is_morning_rush"] = ((df1["hour"] >= 7) & (df1["hour"] < 9)).astype(int)
    df1["is_afternoon_rush"] = ((df1["hour"] >= 16) & (df1["hour"] <19)).astype(int)
    df1["is_rush_hour"] = (df1["is_morning_rush"] | df1["is_afternoon_rush"]).astype(int)
    df1["is_lunch_hour"] = ((df1["hour"] >= 12) & (df1["hour"] < 14)).astype(int)
    df1["is_night"]  = ((df1["hour"] >= 22) | (df1["hour"] < 5)).astype(int)
    df1["is_business_hours"] = ((df1["hour"] >= 8) & (df1["hour"] < 18) & (df1["is_weekday"] == 1)).astype(int)

    be_holidays = holidays.BE(years=list(range(2019, 2027)))
    holiday_dict = {pd.to_datetime(k).date(): v for k, v in be_holidays.items()}
    df1["is_public_holiday"] = t.dt.date.apply(lambda d: int(d in holiday_dict))
    df1["public_holiday"] = t.dt.date.apply(lambda d: holiday_dict.get(d, None)).astype('str')

    # # Cyclic (hourly series — no slot_15m / slot_sin / slot_cos)
    # df1["hour_sin"] = np.sin(2 * np.pi * df1["hour"] / 24)
    # df1["hour_cos"] = np.cos(2 * np.pi * df1["hour"] / 24)
    # df1["dow_sin"] = np.sin(2 * np.pi * df1["dow"] / 7)
    # df1["dow_cos"] = np.cos(2 * np.pi * df1["dow"] / 7)
    # df1["month_sin"] = np.sin(2 * np.pi * (df1["month"] - 1) / 12)
    # df1["month_cos"] = np.cos(2 * np.pi * (df1["month"] - 1) / 12)
    return df1[[x for x in df1 if not x.startswith('traffic') and x!='time_of_day']]

def compute_site_features(df_raw):
    df_sites = pd.read_csv(FILEPATH_SITES, header=None, names=SITES_COLS)
    df = df_raw.merge(df_sites, on="site_id", how="left")

    df["installation_date"] = pd.to_datetime(df["installation_date"])
    df["site_sensor_age"] = (df["datetime"] - df["installation_date"]).dt.days

    # A: Motorway / Highway, N: National / Regional Road, R: Ring Road, T:Tertiary / Connecting Road
    df["site_is_road_tunnel"] = df["road_id"].astype(str).str.startswith("T").astype(int)
    df["site_is_road_national"] = df["road_id"].astype(str).str.startswith("N").astype(int)
    df["site_is_road_ring"] = df["road_id"].astype(str).str.startswith("R").astype(int)
    df["site_is_road_motorway"] = df["road_id"].astype(str).str.startswith("A").astype(int)

    df["site_is_inbound"] = (df["direction"] == "IN").astype(int)
    df["site_is_outbound"] = (df["direction"] == "OUT").astype(int)
    return df[['site_id', 'direction', 'date', 'hour', 'datetime']+[x for x in df.columns if x.startswith('site') and x not in ('site_nr', 'site_name','site_id')]]