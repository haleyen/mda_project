import time
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import holidays

from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

import math
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer


######## FEATURE ENGINEERING ########
def compute_neighbor_traffic_lag_hour_features(df_traffic, df_site):
    RADII_M = [200,500,1000,2000]
    RADII_KM = [r / 1000 for r in RADII_M]   # [0.2, 0.5, 1.0, 2.0]

    sites = df_site.copy()
    sites = sites[["site_id", "latitude", "longtitude"]].drop_duplicates("site_id").copy()
    sites = sites.sort_values("site_id").reset_index(drop=True)

    ids = sites["site_id"].to_numpy()
    lat = sites["latitude"].to_numpy()
    lon = sites["longtitude"].to_numpy()

    # Haversine for all pairs (vectorized)
    R = 6371.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (np.sin(dlat / 2) ** 2
        + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(dlon / 2) ** 2)
    dist = 2 * R * np.arcsin(np.sqrt(a))

    # Fill diagonal with infinity so that sites don't consider themselves as neighbors
    np.fill_diagonal(dist, np.inf)

    dist_df = pd.DataFrame(dist, index=ids, columns=ids)

    # radius_maps[radius_m][site_id] = [list site within that radius]
    radius_maps = {r_m: {} for r_m in RADII_M}
    for site in ids:
        ordered = dist_df.loc[site].sort_values()
        for r_m, r_km in zip(RADII_M, RADII_KM):
            radius_maps[r_m][site] = ordered[ordered <= r_km].index.tolist()


    df_traffic_cleaned = df_traffic.copy()
    df_traffic_cleaned = df_traffic_cleaned[df_traffic_cleaned["type"] == "FIETSERS"]
    df_traffic_cleaned['count'] = df_traffic_cleaned['count'].fillna(0)
    df_traffic_cleaned["site_id"]   = df_traffic_cleaned["site_id"].astype("int32")
    df_traffic_cleaned["from"] = pd.to_datetime(df_traffic_cleaned["from"])
    df_traffic_cleaned["datehour"] = pd.to_datetime(df_traffic_cleaned["from"]).dt.floor("h")

    df_traffic_hourly = (
        df_traffic_cleaned
        .groupby(["site_id", "datehour"], as_index=False)["count"]
        .sum()
        .rename(columns={'count':'traffic_total'})
        .reset_index(drop=True)
    )
    traffic_wide = df_traffic_hourly.pivot_table(index="datehour",columns="site_id",values="traffic_total",aggfunc="sum").sort_index()

    lag_dfs = []
    for LAG in [1, 24, 168]:
        traffic_wide_lag = traffic_wide.shift(LAG)
        nan_template = pd.Series(np.nan, index=traffic_wide_lag.index)
        parts = []
        for site in ids:
            row_dict = {"site_id": site, "datehour": traffic_wide_lag.index}
            for r_m in RADII_M:
                rad_cols = [s for s in radius_maps[r_m][site] if s in traffic_wide_lag.columns]
                prefix = f"nb_r{r_m}m_lag{LAG}h"
                if rad_cols:
                    sub = traffic_wide_lag[rad_cols]
                    row_dict[f"{prefix}_mean"] = sub.mean(axis=1).values
                    row_dict[f"{prefix}_max"]  = sub.max(axis=1).values
                    row_dict[f"{prefix}_sum"]  = sub.sum(axis=1).values
                    row_dict[f"{prefix}_std"]  = sub.std(axis=1).values
                else:
                    for stat in ("mean", "max", "sum", "std"):
                        row_dict[f"{prefix}_{stat}"] = nan_template.values
            parts.append(pd.DataFrame(row_dict))
        lag_dfs.append(pd.concat(parts, ignore_index=True))

    df_fts_neighbor_lag = lag_dfs[0]
    for df_lag in lag_dfs[1:]:
        df_fts_neighbor_lag = df_fts_neighbor_lag.merge(
            df_lag.drop(columns=["site_id", "datehour"], errors="ignore")
            if False else df_lag,  # merge full second df
            on=["site_id", "datehour"],
            how="outer",
        )
    df_fts_neighbor_lag.columns = [col.replace('168h', '7d') for col in df_fts_neighbor_lag.columns]
    df_fts_neighbor_lag = df_fts_neighbor_lag[['site_id','datehour'] + [x for x in df_fts_neighbor_lag if x.startswith('nb')]]
    df_fts_neighbor_lag = df_fts_neighbor_lag.loc[:, df_fts_neighbor_lag.isna().mean() <= 0.7]

    return df_fts_neighbor_lag

def compute_neighbor_traffic_lag_15m_features(df_traffic, df_site):
    RADII_M = [200,500,1000,2000]
    RADII_KM = [r / 1000 for r in RADII_M]

    sites = df_site.copy()
    sites = sites[["site_id", "latitude", "longtitude"]].drop_duplicates("site_id").copy()
    sites = sites.sort_values("site_id").reset_index(drop=True)

    ids = sites["site_id"].to_numpy()
    lat = sites["latitude"].to_numpy()
    lon = sites["longtitude"].to_numpy()

    # Haversine for all pairs (vectorized)
    R = 6371.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (np.sin(dlat / 2) ** 2
        + np.cos(lat_rad)[:, None] * np.cos(lat_rad)[None, :] * np.sin(dlon / 2) ** 2)
    dist = 2 * R * np.arcsin(np.sqrt(a))

    # Fill diagonal with infinity so that sites don't consider themselves as neighbors
    np.fill_diagonal(dist, np.inf)
    dist_df = pd.DataFrame(dist, index=ids, columns=ids)

    # radius_maps[radius_m][site_id] = [list site within that radius]
    radius_maps = {r_m: {} for r_m in RADII_M}
    for site in ids:
        ordered = dist_df.loc[site].sort_values()
        for r_m, r_km in zip(RADII_M, RADII_KM):
            radius_maps[r_m][site] = ordered[ordered <= r_km].index.tolist()


    df_traffic_cleaned = df_traffic.copy()
    df_traffic_cleaned = df_traffic_cleaned[df_traffic_cleaned["type"] == "FIETSERS"]
    df_traffic_cleaned['count'] = df_traffic_cleaned['count'].fillna(0)
    df_traffic_cleaned["site_id"]   = df_traffic_cleaned["site_id"].astype("int32")
    df_traffic_cleaned["from"] = pd.to_datetime(df_traffic_cleaned["from"])
    df_traffic_cleaned["datehour"] = pd.to_datetime(df_traffic_cleaned["from"]).dt.floor("h")


    # Step 1: 15-min FIETSERS traffic, summed across directions 
    df_traffic_fiets_15min = (
        df_traffic_cleaned
        .groupby(["site_id", "from"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total_fiets_15min"})
    )

    # Step 2: Wide matrix (rows = 15-min timestamp, cols = site_id)
    traffic_wide_15min = df_traffic_fiets_15min.pivot_table(
        index="from",
        columns="site_id",
        values="total_fiets_15min",
        aggfunc="sum",
    ).sort_index()
    traffic_wide_15min.index.name = "ts"

    # Step 3: Shift by 1 period (15 min) → previous 15-min traffic 
    # Row at ts=T in shifted table = original traffic at ts=(T-15min)
    traffic_wide_15min_shifted = traffic_wide_15min.shift(1)

    # Step 4: Keep only on-the-hour rows to align with datehour 
    # Row at ts=14:00 (minute==0) after shift = traffic from 13:45-14:00
    traffic_wide_prev15 = traffic_wide_15min_shifted[
        traffic_wide_15min_shifted.index.minute == 0
    ].copy()
    traffic_wide_prev15.index.name = "datehour"


    # Step 5: For each site and datehour, compute neighbor features based on previous 15-min traffic
    parts_15m = []
    nan_tpl = pd.Series(np.nan, index=traffic_wide_prev15.index)

    for site in ids:
        row = {"site_id":  site, "datehour": traffic_wide_prev15.index,}

        for r_m in RADII_M:
            rad_cols = [s for s in radius_maps[r_m][site] if s in traffic_wide_prev15.columns]
            pfx = f"nb_r{r_m}m_lag15m"

            if rad_cols:
                sub = traffic_wide_prev15[rad_cols]
                row[f"{pfx}_mean"] = sub.mean(axis=1).values
                row[f"{pfx}_max"]  = sub.max(axis=1).values
                row[f"{pfx}_sum"]  = sub.sum(axis=1).values
                row[f"{pfx}_std"]  = sub.std(axis=1).values
            else:
                for stat in ("mean", "max", "sum", "std"):
                    row[f"{pfx}_{stat}"] = nan_tpl.values
        parts_15m.append(pd.DataFrame(row))

    df_fts_neighbor_lag15m = pd.concat(parts_15m, ignore_index=True)
    df_fts_neighbor_lag15m = df_fts_neighbor_lag15m[['site_id','datehour'] + [x for x in df_fts_neighbor_lag15m if x.startswith('nb')]]
    df_fts_neighbor_lag15m = df_fts_neighbor_lag15m.loc[:, df_fts_neighbor_lag15m.isna().mean() <= 0.7]

    return df_fts_neighbor_lag15m
 
def compute_weather_features(df_weather, lags_h = [1, 2, 3, 6]):
    df = df_weather.copy()
    df["datehour"] = pd.to_datetime(df["datetime"])
    df = df.drop(columns=['latitude', 'longitude', 'datetime']).rename(columns={x: f"wt_{x}" for x in df.columns if x not in ['site_id', 'latitude', 'longitude','datehour']})


    df["datehour"] = pd.to_datetime(df["datehour"])
    df = df.sort_values(["site_id", "datehour"])

    grp = df.groupby("site_id")

    # ── Plain lags
    lag_cols = ["wt_precipitation", "wt_rain", "wt_temperature_2m", "wt_wind_speed_10m", "wt_cloud_cover", "wt_snowfall"]
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lags_h:
            df[f"{col}_lag{lag}h"] = grp[col].shift(lag)

    # ── Rolling aggregations (sum & mean, using only past values)
    rolling_windows = {
        "wt_precipitation": [3, 6, 12],   # accumulated rain last N hours
        "wt_temperature_2m": [3, 24],      # temp trend
    }
    for col, windows in rolling_windows.items():
        if col not in df.columns:
            continue
        for w in windows:
            # shift(1) → exclude current hour, use only past
            base = grp[col].shift(1)
            df[f"{col}_rolling_sum_{w}h"]  = base.transform(
                lambda x: x.rolling(w, min_periods=1).sum()
            )
            df[f"{col}_rolling_mean_{w}h"] = base.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )

    # ── Derived lag features
    if "wt_temperature_2m" in df.columns:
        # Temperature trend: warming or cooling?
        df["wt_temp_change_1h"] = df["wt_temperature_2m"] - grp["wt_temperature_2m"].shift(1)
        df["wt_temp_change_3h"] = df["wt_temperature_2m"] - grp["wt_temperature_2m"].shift(3)

    if "wt_precipitation" in df.columns:
        # Binary: was it raining in each of the last 3 hours?
        for lag in [1, 2, 3]:
            df[f"wt_was_raining_lag{lag}h"] = (
                grp["wt_precipitation"].shift(lag) > 0.5
            ).astype(int)

        # Consecutive rainy hours (deterrence buildup)
        is_raining = (grp["wt_precipitation"].shift(1) > 0.5).astype(int)
        df["wt_consec_rain_hours"] = (
            is_raining
            .groupby(df["site_id"])
            .transform(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
        )

        # Flag: first dry hour after rain (pent-up demand spike)
        df["wt_first_dry_after_rain"] = (
            (grp["wt_precipitation"].shift(1) > 0.5) &   # was raining 1h ago
            (df["wt_precipitation"] <= 0.5)               # now dry
        ).astype(int)

    return df

def compute_population_features(df_site, tif_path):
    def population_estimate(lat, lon, radius_m, tif_path):
        with rasterio.open(tif_path) as src:
            METRIC_CRS = "EPSG:3035"  
            # 1. Covert (lat, lon) to m for accurate buffering
            to_metric = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
            x_m, y_m = to_metric.transform(lon, lat)

            # 2. Create circle with radius_m meters
            circle = Point(x_m, y_m).buffer(radius_m)

            # 3. Transform circle to raster's CRS for masking
            to_raster = Transformer.from_crs(METRIC_CRS, src.crs, always_xy=True)
            circle_in_raster = transform(to_raster.transform, circle)

            # 4. Cut raster by circle, mask NoData automatically
            out, _ = mask(src, [circle_in_raster], crop=True, filled=False)

            total_pop = float(out.sum())                       # total population in the circle
            area_km2  = math.pi * (radius_m ** 2) / 1e6        # area of the circle in km²
            density   = total_pop / area_km2                   # population density in people/km²
            return total_pop, density, area_km2

    rows = []
    for loc in df_site.itertuples():
        name, lat, lon = loc.site_id, loc.latitude, loc.longtitude
        row = {"site_id": name, "latitude": lat, "longitude": lon}
        for r in (500, 1000, 5000):
            pop, dens, area = population_estimate(lat, lon, r, tif_path)
            row[f"pop_{r}m"] = round(pop, 1)
            row[f"density_{r}m"] = round(dens, 1)
        rows.append(row)

    df_fts_population = pd.DataFrame(rows)
    df_fts_population = df_fts_population[['site_id'] + [x for x in df_fts_population if x.startswith('pop') or x.startswith('density')]]
    return df_fts_population

def compute_datetime_features(df_traffic):
    """Calendar + cyclical features at hourly resolution (no 15-min slot)."""
    df_traffic_cleaned = df_traffic.copy()
    df_traffic_cleaned = df_traffic_cleaned.query('type=="FIETSERS"')
    df_traffic_cleaned["from"] = pd.to_datetime(df_traffic_cleaned["from"])
    df_traffic_cleaned["datehour"] = (pd.to_datetime(df_traffic_cleaned["from"].dt.strftime("%Y-%m-%d")) + pd.to_timedelta(df_traffic_cleaned["from"].dt.hour, unit="h"))
    df_traffic_cleaned["datehour"] = df_traffic_cleaned["from"].dt.floor("h")

    dt = df_traffic_cleaned[["site_id", "datehour"]].drop_duplicates().copy()
    dt["year"]      = dt["datehour"].dt.year
    dt["month"]     = dt["datehour"].dt.month
    dt["day"]       = dt["datehour"].dt.day
    dt["hour"]      = dt["datehour"].dt.hour
    dt["dayofweek"] = dt["datehour"].dt.dayofweek

    # Weekday / weekend
    dt["dt_is_weekend"] = dt["dayofweek"].isin([5, 6]).astype(int)
    dt["dt_is_weekday"] = 1 - dt["dt_is_weekend"]

    # Specific days
    dt["dt_is_monday"] = (dt["dayofweek"] == 0).astype(int)
    dt["dt_is_friday"] = (dt["dayofweek"] == 4).astype(int)

    # Hour-of-day buckets
    dt["dt_is_morning_rush"]   = dt["hour"].isin([7, 8, 9]).astype(int)
    dt["dt_is_afternoon_rush"] = dt["hour"].isin([16, 17, 18]).astype(int)
    dt["dt_is_rush_hour"]      = (dt["dt_is_morning_rush"] | dt["dt_is_afternoon_rush"]).astype(int)
    dt["dt_is_lunch_hour"]     = dt["hour"].isin([12, 13]).astype(int)
    dt["dt_is_night"]          = dt["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    dt["dt_is_business_hours"] = (dt["hour"].between(9, 17) & (dt["dt_is_weekday"] == 1)).astype(int)

    # Belgian pulic holiday
    bel_holiday_dates = list(holidays.Belgium(years=range(2024, 2027)).keys())
    dt["dt_is_holiday"] = dt["datehour"].dt.normalize().isin(bel_holiday_dates).astype(int)

    df2 = dt[['site_id','datehour', "year", "month", "day", "hour", "dayofweek"] + [x for x in dt if x.startswith('dt')]]

    return df2

def compute_site_features(df_traffic, df_site):
    df_traffic_cleaned = df_traffic.copy()
    df_traffic_cleaned = df_traffic_cleaned.query('type=="FIETSERS"')
    df_traffic_cleaned["from"] = pd.to_datetime(df_traffic_cleaned["from"])
    df_traffic_cleaned["datehour"] = (pd.to_datetime(df_traffic_cleaned["from"].dt.strftime("%Y-%m-%d")) + pd.to_timedelta(df_traffic_cleaned["from"].dt.hour, unit="h"))
    df_traffic_cleaned["datehour"] = df_traffic_cleaned["from"].dt.floor("h")


    sites = df_site.copy()
    sites["install_date"] = pd.to_datetime(sites["date_installed"], errors="coerce")
    sites["site_is_road_tunnel"]   = sites["road_nbr"].astype(str).str.upper().str.startswith("T", na=False).astype(int)
    sites["site_is_road_national"] = sites["road_nbr"].astype(str).str.upper().str.startswith("N", na=False).astype(int)
    sites["site_is_road_ring"]     = sites["road_nbr"].astype(str).str.upper().str.startswith("R", na=False).astype(int)
    sites["site_is_road_motorway"] = sites["road_nbr"].astype(str).str.upper().str.startswith("A", na=False).astype(int)

    df_fts_site = (
        df_traffic_cleaned[["site_id", "datehour"]]
        .drop_duplicates()
        .merge(
            sites[["site_id", "install_date"] + [x for x in sites if x.startswith('site_is')]],
            on="site_id", how="left",
        )
    )

    df_fts_site["site_sensor_age"] = ((df_fts_site["datehour"] - df_fts_site["install_date"]).dt.days / 365)
    df_fts_site = df_fts_site[["site_id", "datehour"] + [x for x in df_fts_site if x.startswith('site_is')]]

    return df_fts_site
    
def compute_infra_features(df_infra, df_site):
    df_infra00 = df_infra.copy() 
    df_site00 = df_site.copy()

    df_infra00 = df_infra00.rename(columns={'site_id':'site_nr'}).merge(df_site00[['site_id','site_nr']], on='site_nr', how='left')
    df_fts_infra = df_infra00.drop(columns=['sensor_age_days','has_cycleway'])

    poi_types = ['shop', 'education', 'hotel', 'hospital']
    for dist in ['250m', '500m', '1000m']:
        cols = [f'poi_{typ}_{dist}' for typ in poi_types]
        cols_present = [c for c in cols if c in df_fts_infra.columns]
        df_fts_infra[f'poi_{dist}'] = df_fts_infra[cols_present].sum(axis=1)

    df_fts_infra['road_category'] = df_fts_infra['road_category_en'].replace({
        'not_applicable': 'others',
        'high_capacity_road': 'others',
        'expressway_limited_access': 'others',
    })

    road_category_dummies = pd.get_dummies(df_fts_infra['road_category'], prefix='road_category', dtype=int)
    df_fts_infra = pd.concat([df_infra00, road_category_dummies], axis=1)
    df_fts_infra = df_fts_infra.rename(columns={c: f'road_{c}' for c in ['length_m', 'dist_to_segment_m', 'bike_lane_width_m', 'has_cycleway', 'bike_width_imputed']})
    df_fts_infra = df_fts_infra[['site_id'] + [x for x in df_fts_infra if (x.startswith('road') or x.startswith('poi')) and pd.api.types.is_numeric_dtype(df_fts_infra[x])]]

    return df_fts_infra

######## MODELING ########
def split_traintest(df, list_features, target):
    d = df.copy()
    X = d[list_features]
    y = d[target].astype(float)
    return X, y

def score_predictions(y_true, pred):
    pred = np.clip(pred, 0, None)
    return {
        "MAE":  mean_absolute_error(y_true, pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, pred)),
    }

def run_models(dmodels, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate all models in dmodels, return dict_model with results.
    Added 4 models: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor.
    Each output contains the number of features ('nfeatures' key).
    Adds validation MAE and RMSE to the output.
    """
    dict_model = {}
    for name, model in tqdm(dmodels.items(), total=len(dmodels), desc="Training models"):
        t0 = time.time()
        nfeatures = X_train.shape[1]

        if "LinearRegression" in name or "GradientBoosting" in name:
            X_tr, X_te, X_val = X_train.fillna(0), X_test.fillna(0), X_val.fillna(0)
            model.fit(X_tr, y_train)
            pred = model.predict(X_te)
            pred_val = model.predict(X_val)
        elif any(k in name for k in ["DecisionTree", "RandomForest", "HistGradientBoosting"]):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred_val = model.predict(X_val)
        elif "XGBoost" in name:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred = model.predict(X_test)
            pred_val = model.predict(X_val)
        elif "LightGBM" in name:
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], 
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
            )
            pred = model.predict(X_test)
            pred_val = model.predict(X_val)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            pred_val = model.predict(X_val)

        scores = score_predictions(y_test, pred)
        valid_scores = score_predictions(y_val, pred_val)

        if "HistGradientBoosting" in name or "GradientBoostingRegressor" in name:
            best_iter = model.n_iter_
        else:
            best_iter = getattr(model, "best_iteration_", getattr(model, "best_iteration", "N/A"))
        print(f"{name:<20s} - Test MAE: {scores['MAE']:8.3f} | Valid MAE: {valid_scores['MAE']:8.3f} | Iter: {str(best_iter):>5s} | nfeatures: {nfeatures:>4d}")
      
        dict_model[name] = {
            "model": model,
            "nfeatures": nfeatures,
            "mae": scores["MAE"],
            "rmse": scores["RMSE"],
            "valid_mae": valid_scores["MAE"],
            "valid_rmse": valid_scores["RMSE"],
            "time": time.time() - t0,
            "best_iteration": best_iter
        }
    return dict_model

def stepwise_feature_selection_lgb(
    mode,
    list_feature_all,
    base_model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    model_params=None,
):
    if model_params is None:
        model_params = {}
    if mode not in ["add", "remove"]:
        raise ValueError("mode must be either 'add' or 'remove'")
    model_class = lgb.LGBMRegressor
    log_dict = {}

    # 1. Extract initial features from the base model
    if hasattr(base_model, "feature_names_in_"):
        current_features = list(base_model.feature_names_in_)
    else:
        raise AttributeError("base_model must have .feature_names_in_ attribute.")

    # 2. Establish initial baseline (on validation set)
    X_valid_base = X_valid[current_features]
    base_preds = base_model.predict(X_valid_base)
    curr_mae = mean_absolute_error(y_valid, base_preds)

    log_dict['model_base'] = {
        'features': list(current_features),
        'MAE': curr_mae,
        'model': base_model
    }

    # 3. Determine candidates based on mode
    if mode == "add":
        candidates = [f for f in list_feature_all if f not in current_features]
    else:  # remove
        candidates = list(current_features)

    step_iter = 1

    # 4. Single pass through candidates
    for feat in candidates:
        # Construct the feature set
        if mode == "add":
            try_features = current_features + [feat]
        else:  # remove
            if len(current_features) <= 1:
                break
            try_features = [f for f in current_features if f != feat]
        
        # Train and evaluate on training/validation splits
        model_try = model_class(**model_params)
        model_try.fit(X_train[try_features], y_train)
        preds_try = model_try.predict(X_valid[try_features])
        mae_try = mean_absolute_error(y_valid, preds_try)

        action = "Add" if mode == "add" else "Remove"
        print(f"{action}: {feat:50s} | MAE: {mae_try:.5f} | current MAE: {curr_mae:.5f}")

        # If MAE improves (decreases), update current set and baseline
        if mae_try < curr_mae:
            status = "ADDED" if mode == "add" else "REMOVED"
            print(f"  => [{status}]: {feat:50s} | New Base MAE: {mae_try:.5f}")
            
            current_features = try_features
            curr_mae = mae_try

            log_dict[f'model_{mode}_{step_iter}'] = {
                'features': list(current_features),
                'MAE': curr_mae,
                'model': model_try,
                'feature_changed': feat
            }
            step_iter += 1

    return current_features, log_dict  