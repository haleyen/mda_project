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

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))

PATH_EXTRA_DATA       = os.path.join(BASE_DIR, "../../../data/extra/")
PATH_TRAFFIC          = os.path.join(BASE_DIR, "../../../data/traffic/")
FILEPATH_RICH         = os.path.join(BASE_DIR, "../../../data/richtingen.csv")
FILEPATH_SITES        = os.path.join(BASE_DIR, "../../../data/sites.csv")

FILEPATH_FTS_POP      = os.path.join(BASE_DIR, "../data/fts_population.csv")
FILEPATH_FTS_WEATHER  = os.path.join(BASE_DIR, "../data/fts_weather.csv")

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
    df = df_raw.copy().query('type=="FIETSERS"')
    df["start_time"] = pd.to_datetime(df["start_time"])
    # df["month"] = df["start_time"].dt.strftime("%Y-%m")
    # df["date"] = df["start_time"].dt.strftime("%Y-%m-%d")
    # df["hour"] = df["start_time"].dt.hour

    df["datehour"] = (pd.to_datetime(df["start_time"].dt.strftime("%Y-%m-%d")) + pd.to_timedelta(df["start_time"].dt.hour, unit="h"))
    df["quarter"] = df["start_time"].dt.minute // 15

    df_hourly = (
        df.pivot_table(
            index=["site_id", "datehour"],
            columns=["quarter", "direction"],
            values="traffic",
            aggfunc="sum"
        )
    )

    quarter_map = {0: "00_15", 1: "15_30", 2: "30_45", 3: "45_60"}
    df_hourly.columns = [
        f"traffic_{str(direction).lower()}_{quarter_map[q]}"
        for q, direction in df_hourly.columns
    ]
    df_hourly = df_hourly.reset_index()
    df_hourly["traffic_in"] = df_hourly[[col for col in df_hourly.columns if col.startswith("traffic_in_")]].sum(axis=1)
    df_hourly["traffic_out"] = df_hourly[[col for col in df_hourly.columns if col.startswith("traffic_out_")]].sum(axis=1)

    for d in ["in", "out"]:
        traffic_cols = [col for col in df_hourly.columns if col.startswith(f"traffic_{d}_")]
        df_hourly[f"traffic_{d}_15m_mean"] = df_hourly[traffic_cols].mean(axis=1, skipna=True)
        df_hourly[f"traffic_{d}_15m_std"] = df_hourly[traffic_cols].std(axis=1, skipna=True)
        df_hourly[f"traffic_{d}_15m_min"] = df_hourly[traffic_cols].min(axis=1, skipna=True)
        df_hourly[f"traffic_{d}_15m_max"] = df_hourly[traffic_cols].max(axis=1, skipna=True)

    return df_hourly

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

def compute_datetime_features(df_raw):
    """Calendar + cyclical features at hourly resolution (no 15-min slot)."""
    df = df_raw.copy()
    df = df.query('type=="FIETSERS"')
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["datehour"] = (pd.to_datetime(df["start_time"].dt.strftime("%Y-%m-%d")) + pd.to_timedelta(df["start_time"].dt.hour, unit="h"))

    df1 = df[["site_id", "datehour"]].drop_duplicates()

    t = df1["datehour"]
    # df1["date"] = t.dt.day
    # df1["month"] = t.dt.month
    # df1["year"] = t.dt.year
    df1["hour"] = t.dt.hour
    df1["dow"] = t.dt.dayofweek  # 0=Mon, 6=Sun
    # df1["day_name"] = t.dt.day_name()
    # df1["woy"] = t.dt.isocalendar().week.astype(int)

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

    df2 = df1[[x for x in df1 if not x.startswith('traffic') and x not in ['hour','time_of_day', 'dow']]]
    df2 = df2.rename(columns={col: f"dt_{col}" for col in [col for col in df2.columns if col not in ("site_id", "datehour")]})
    return df2
    
def compute_site_features(df_raw):
    df_sites = pd.read_csv(FILEPATH_SITES, header=None, names=SITES_COLS)

    df = df_raw.copy()
    df = df.query('type=="FIETSERS"')
    df["start_time"] = pd.to_datetime(df["start_time"])
    # df["month"] = df["start_time"].dt.strftime("%Y-%m")
    # df["date"] = df["start_time"].dt.normalize()#.strftime("%Y-%m-%d")
    # df["hour"] = df["start_time"].dt.hour
    df["datehour"] = (pd.to_datetime(df["start_time"].dt.strftime("%Y-%m-%d")) + pd.to_timedelta(df["start_time"].dt.hour, unit="h"))

    df1 = df[["site_id", "datehour"]].drop_duplicates()
    df1 = df1.merge(df_sites, on="site_id", how="left")

    df1["installation_date"] = pd.to_datetime(df1["installation_date"])
    df1["site_sensor_age"] = (df1["datehour"] - df1["installation_date"]).dt.days

    # A: Motorway / Highway, N: National / Regional Road, R: Ring Road, T:Tertiary / Connecting Road
    df1["site_is_road_tunnel"] = df1["road_id"].astype(str).str.startswith("T").astype(int)
    df1["site_is_road_national"] = df1["road_id"].astype(str).str.startswith("N").astype(int)
    df1["site_is_road_ring"] = df1["road_id"].astype(str).str.startswith("R").astype(int)
    df1["site_is_road_motorway"] = df1["road_id"].astype(str).str.startswith("A").astype(int)

    return df1[["site_id", "datehour"]+[x for x in df1.columns if x.startswith('site') and x not in ('site_nr', 'site_name','site_id')]]


########################################################################
### MODELING ###

def split_traintest(df, list_features, target):
    d = df.copy()#.dropna(subset=cols)
    X = d[list_features]  # keep as DataFrame
    y = d[target].astype(float)
    return X, y