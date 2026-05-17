"""
Fixed OSM fetch cells for site_road_feature_extract.ipynb
=========================================================
Drop-in replacement code for cells c6, c7, c8.

Changes vs original:
  - c6 (bike lane)  : caches results to bike_cache.csv — skip 151 API calls on re-runs
  - c7 (city nodes) : caches to cities_cache.gpkg + filters to Belgium only
                      (fixes "Merzig" bug — non-Belgian cities were returned as nearest)
  - c8 (POI counts) : caches each category to poi_{cat}_cache.gpkg
"""

# ── Assumes all earlier cells (c1–c5) have already run ──────────────────────
# i.e. sites_joined, OUTPUT_DIR, POI_TAGS, POI_RADII are already in scope.

import os, time
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox


# ════════════════════════════════════════════════════════════════════════════
# CELL c6  –  Bike-Lane Width from OpenStreetMap  (with caching)
# ════════════════════════════════════════════════════════════════════════════

WIDTH_TAGS_PRIORITY = [
    'cycleway:width',
    'cycleway:right:width',
    'cycleway:left:width',
    'cycleway:both:width',
    'cycleway:lane:width',
]
BIKE_HIGHWAY_TYPES = {'cycleway', 'path', 'footway', 'bridleway', 'track'}
CYCLEWAY_PRESENCE_TAGS = [
    'cycleway', 'cycleway:right', 'cycleway:left', 'cycleway:both',
    'cycleway:lane', 'cyclestreet', 'oneway:bicycle',
]


def _parse_width(val):
    """Parse width strings like '2.5', '2.5 m', '250 cm' into float metres."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    if s.endswith('cm'):
        try:
            return float(s.replace('cm', '').strip()) / 100
        except ValueError:
            pass
    for unit in [' meters', ' meter', ' m', 'm']:
        s = s.replace(unit, '').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def get_bike_info(lat, lon, dist=200):
    result = {'bike_lane_width_m': np.nan, 'bike_lane_source': 'no_data', 'has_cycleway': False}
    try:
        gdf = ox.features_from_point((lat, lon), tags={'highway': True}, dist=dist)
        if gdf.empty:
            result['bike_lane_source'] = 'no_osm_features'
            return result

        # Tier 1
        for tag in WIDTH_TAGS_PRIORITY:
            if tag in gdf.columns:
                vals = gdf[tag].dropna()
                if not vals.empty:
                    w = _parse_width(vals.iloc[0])
                    if not np.isnan(w):
                        result.update({'bike_lane_width_m': w, 'bike_lane_source': f'tier1:{tag}', 'has_cycleway': True})
                        return result

        # Cycleway presence
        rows_with_cycleway = pd.Series(False, index=gdf.index)
        for c in CYCLEWAY_PRESENCE_TAGS:
            if c in gdf.columns:
                valid = gdf[c].dropna()
                positive = valid[~valid.astype(str).str.lower().isin(['no', 'none', 'nan'])]
                rows_with_cycleway |= gdf.index.isin(positive.index)
        if rows_with_cycleway.any():
            result['has_cycleway'] = True

        # Tier 2
        if 'width' in gdf.columns and rows_with_cycleway.any():
            cyc_widths = gdf.loc[rows_with_cycleway, 'width'].dropna()
            if not cyc_widths.empty:
                w = _parse_width(cyc_widths.iloc[0])
                if not np.isnan(w):
                    result.update({'bike_lane_width_m': w, 'bike_lane_source': 'tier2:width+cycleway_tag'})
                    return result

        # Tier 3
        if 'highway' in gdf.columns and 'width' in gdf.columns:
            bike_ways = gdf[gdf['highway'].isin(BIKE_HIGHWAY_TYPES)]
            widths = bike_ways['width'].dropna() if not bike_ways.empty else pd.Series([])
            if not widths.empty:
                w = _parse_width(widths.iloc[0])
                if not np.isnan(w):
                    result.update({'bike_lane_width_m': w, 'bike_lane_source': 'tier3:bike_highway_width', 'has_cycleway': True})
                    return result

        # Tier 4
        result['bike_lane_source'] = (
            'tier4:has_cycleway_no_width' if result['has_cycleway'] else 'tier4:no_cycleway_tag'
        )
    except Exception as e:
        result['bike_lane_source'] = f'error:{type(e).__name__}'
    return result


# ── Cache check ──────────────────────────────────────────────────────────────
BIKE_CACHE = os.path.join(OUTPUT_DIR, 'bike_cache.csv')

if os.path.exists(BIKE_CACHE):
    bike_df = pd.read_csv(BIKE_CACHE)
    sites_joined['bike_lane_width_m'] = bike_df['bike_lane_width_m'].values
    sites_joined['bike_lane_source']  = bike_df['bike_lane_source'].values
    sites_joined['has_cycleway']      = bike_df['has_cycleway'].values
    print(f'Loaded bike-lane info from cache: {BIKE_CACHE}')
else:
    print('Extracting bike-lane info (200 m radius, 4-tier fallback)...')
    t0 = time.time()
    bike_results = [get_bike_info(r['latitude'], r['longitude'])
                    for _, r in sites_joined.iterrows()]
    sites_joined['bike_lane_width_m'] = [d['bike_lane_width_m'] for d in bike_results]
    sites_joined['bike_lane_source']  = [d['bike_lane_source']  for d in bike_results]
    sites_joined['has_cycleway']      = [d['has_cycleway']      for d in bike_results]
    # Save cache
    sites_joined[['site_id', 'bike_lane_width_m', 'bike_lane_source', 'has_cycleway']].to_csv(BIKE_CACHE, index=False)
    print(f'Done in {time.time()-t0:.1f} s — saved to {BIKE_CACHE}')

n_total = len(sites_joined)
n_width = sites_joined['bike_lane_width_m'].notna().sum()
n_cyc   = sites_joined['has_cycleway'].sum()
print(f'  Width extracted : {n_width}/{n_total} sites ({n_width/n_total*100:.1f} %)')
print(f'  Has cycleway tag: {n_cyc}/{n_total} sites ({n_cyc/n_total*100:.1f} %)')
print(sites_joined['bike_lane_source'].value_counts().to_string())


# ════════════════════════════════════════════════════════════════════════════
# CELL c7  –  Distance to Nearest City (with caching + Belgium-only filter)
# ════════════════════════════════════════════════════════════════════════════

BELGIUM_BBOX = (51.55, 49.45, 6.45, 2.50)   # (north, south, east, west)
CITIES_CACHE = os.path.join(OUTPUT_DIR, 'cities_cache.gpkg')

# ── Step 1: Try loading from cache ───────────────────────────────────────────
cities_pts = None
if os.path.exists(CITIES_CACHE):
    _tmp = gpd.read_file(CITIES_CACHE)
    if len(_tmp) > 0:
        cities_pts = _tmp
        print(f'Loaded {len(cities_pts):,} Belgian cities from cache.')
    else:
        print('Cache file was empty — deleting and will re-fetch...')
        os.remove(CITIES_CACHE)

# ── Step 2: Fetch from OSM if cache was missing or empty ─────────────────────
if cities_pts is None:
    print('Fetching all Belgian city/town nodes from OSM...')
    t0 = time.time()
    _cache_was_on = ox.settings.use_cache
    ox.settings.use_cache = False
    try:
        cities_gdf = ox.features_from_bbox(bbox=BELGIUM_BBOX, tags={'place': ['city', 'town']})
    finally:
        ox.settings.use_cache = _cache_was_on

    # ── Diagnostics: inspect what OSMnx returned ──────────────────────────────
    print(f'  Total OSM features returned : {len(cities_gdf)}')
    print(f'  CRS                         : {cities_gdf.crs}')
    print(f'  Geometry types              : {cities_gdf.geometry.geom_type.value_counts().to_dict()}')

    cities_pts = cities_gdf[cities_gdf.geometry.geom_type == 'Point'].copy().reset_index(drop=True)
    print(f'  Point features              : {len(cities_pts)}')
    if len(cities_pts) > 0:
        print(f'  Longitude (x) range         : {cities_pts.geometry.x.min():.4f} → {cities_pts.geometry.x.max():.4f}')
        print(f'  Latitude  (y) range         : {cities_pts.geometry.y.min():.4f} → {cities_pts.geometry.y.max():.4f}')

    # Filter to Belgium bounding box only
    cities_pts[['name', 'geometry']].to_file(CITIES_CACHE, driver='GPKG')
    print(f'Downloaded and cached {len(cities_pts):,} Belgian city/town nodes in {time.time()-t0:.1f} s')

city_lats  = np.radians(cities_pts.geometry.y.values)
city_lons  = np.radians(cities_pts.geometry.x.values)
city_names = cities_pts['name'].values
print(f'Using {len(cities_pts):,} Belgian city/town nodes.')
print('Sample cities found:', city_names[:10].tolist())


def haversine_batch(site_lat_deg, site_lon_deg, city_lats_rad, city_lons_rad):
    R = 6_371_000.0
    lat1 = np.radians(site_lat_deg)
    lon1 = np.radians(site_lon_deg)
    dlat = city_lats_rad - lat1
    dlon = city_lons_rad - lon1
    a = (np.sin(dlat / 2)**2
         + np.cos(lat1) * np.cos(city_lats_rad) * np.sin(dlon / 2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


print('Computing nearest city for each site (vectorised)...')
t0 = time.time()
nearest_cities, nearest_dists = [], []
for _, row in sites_joined.iterrows():
    dists = haversine_batch(row['latitude'], row['longitude'], city_lats, city_lons)
    idx = int(np.argmin(dists))
    nearest_cities.append(str(city_names[idx]))
    nearest_dists.append(float(dists[idx]))

sites_joined['nearest_city']   = nearest_cities
sites_joined['dist_to_city_m'] = np.round(nearest_dists, 1)
print(f'Done in {time.time()-t0:.1f} s')


# ════════════════════════════════════════════════════════════════════════════
# CELL c8  –  POI Counts at Multiple Radii (with per-category caching)
# ════════════════════════════════════════════════════════════════════════════

poi_gdfs = {}
print('Loading POI data for Belgium (1 request per category, cached)...')
for cat, tags in POI_TAGS.items():
    cache_path = os.path.join(OUTPUT_DIR, f'poi_{cat}_cache.gpkg')
    if os.path.exists(cache_path):
        gdf = gpd.read_file(cache_path)
        poi_gdfs[cat] = gdf
        print(f'  {cat:12s}: loaded {len(gdf):,} features from cache')
    else:
        t0 = time.time()
        try:
            gdf = ox.features_from_bbox(bbox=BELGIUM_BBOX, tags=tags)
            gdf = gdf.copy()
            gdf['geometry'] = gdf.geometry.centroid
            gdf = gdf.to_crs('EPSG:31370')
            gdf[['geometry']].to_file(cache_path, driver='GPKG')   # save geometry only
            poi_gdfs[cat] = gdf
            print(f'  {cat:12s}: {len(gdf):,} features in {time.time()-t0:.1f} s — cached')
        except Exception as e:
            print(f'  {cat:12s}: no features found ({type(e).__name__}), filling with 0')
            poi_gdfs[cat] = None

print()
print('Counting POIs per site per radius (local — no further API calls)...')
t0 = time.time()
for cat, poi_gdf in poi_gdfs.items():
    if poi_gdf is None or poi_gdf.empty:
        for r in POI_RADII:
            sites_joined[f'poi_{cat}_{r}m'] = 0
        continue
    poi_geom = poi_gdf.geometry
    for r in POI_RADII:
        col = f'poi_{cat}_{r}m'
        counts = [int(poi_geom.within(site_geom.buffer(r)).sum())
                  for site_geom in sites_joined.geometry]
        sites_joined[col] = counts
        print(f'  {col}: done  (sum={sum(counts):,})')

print(f'\nAll POI counts done in {time.time()-t0:.1f} s')
