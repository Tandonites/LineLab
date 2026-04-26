"""
Feature Data Fetcher
=====================
Pulls two external datasets needed for the MTA ridership model:

  1. ACS 5-year census estimates (population + job proxy) at census tract level
     Source: Census Bureau API  ->  ./raw/census_tracts.geojson
                                    ./raw/census_population.csv

  2. OSM bus stop locations across NYC
     Source: OpenStreetMap via osmnx  ->  ./raw/bus_stops.geojson

  3. Joins both datasets to the parsed station list and writes
     a combined feature table ready for XGBoost training:
                                    ./raw/station_features.csv

Usage:
  # First-time: get a free Census API key at https://api.census.gov/data/key_signup.html
  python fetch_features.py --census-key YOUR_KEY_HERE

  # If you already ran it and just want to rebuild station_features.csv:
  python fetch_features.py --census-key YOUR_KEY --skip-download

Requirements:
  pip install census osmnx geopandas requests shapely pandas
"""

import os
import json
import time
import argparse
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from shapely.geometry import Point, shape
from census import Census

RAW_DIR    = Path("./raw")
OUTPUT_DIR = Path("./output")   # where mta_parser.py wrote stations.json

# NYC bounding box (used for OSM query)
NYC_BBOX = (40.4774, -74.2591, 40.9176, -73.7004)  # S, W, N, E

# ACS variables we want
# B01003_001E = total population
# B08301_001E = total commuters (proxy for job-adjacent demand)
# B19013_001E = median household income
ACS_VARS = {
    "B01003_001E": "population",
    "B08301_001E": "commuters",
    "B19013_001E": "median_income",
}


# ─────────────────────────────────────────────────────────────
# 1. Census ACS Data
# ─────────────────────────────────────────────────────────────

def fetch_census(api_key: str):
    """
    Pull ACS 5-year estimates for all census tracts in NYC's 5 counties.
    Saves:
      raw/census_population.csv   — flat table per tract
      raw/census_tracts.geojson   — tract polygons with population attached
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    c = Census(api_key)

    # NYC counties by FIPS:
    # 061 = Manhattan, 047 = Brooklyn, 081 = Queens, 005 = Bronx, 085 = Staten Island
    nyc_counties = ["061", "047", "081", "005", "085"]
    STATE_FIPS   = "36"  # New York State

    all_rows = []
    var_list = list(ACS_VARS.keys())

    print("Fetching ACS 5-year estimates (2022)...")
    for county in nyc_counties:
        print(f"  County FIPS {county}...", end=" ", flush=True)
        try:
            results = c.acs5.state_county_tract(
                fields   = ["NAME"] + var_list,
                state_fips  = STATE_FIPS,
                county_fips = county,
                tract    = Census.ALL,
                year     = 2022,
            )
            for r in results:
                row = {
                    "geoid":   f"{STATE_FIPS}{county}{r['tract']}",
                    "name":    r.get("NAME", ""),
                    "county":  county,
                    "tract":   r["tract"],
                }
                for api_col, friendly_col in ACS_VARS.items():
                    val = r.get(api_col, -1)
                    row[friendly_col] = int(val) if val and int(float(val)) >= 0 else 0
                all_rows.append(row)
            print(f"{len(results)} tracts")
        except Exception as e:
            print(f"FAILED: {e}")

    df_census = pd.DataFrame(all_rows)
    csv_path  = RAW_DIR / "census_population.csv"
    df_census.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_census):,} tracts -> {csv_path}")

    # Download tract geometries from Census TIGER/Line shapefiles via the GeoAPI
    print("\nFetching tract geometries from Census GeoAPI...")
    geoid_list = []
    for county in nyc_counties:
        url = (
            f"https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
            f"tigerWMS_ACS2022/MapServer/8/query"
            f"?where=STATE='36'+AND+COUNTY='{county}'"
            f"&outFields=GEOID,NAME&f=geojson&returnGeometry=true"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            gdf_county = gpd.read_file(resp.text if isinstance(resp.text, str) else resp.content)
            geoid_list.append(gdf_county)
            print(f"  County {county}: {len(gdf_county)} tracts")
            time.sleep(0.3)
        except Exception as e:
            print(f"  County {county} geometry FAILED: {e}")

    if geoid_list:
        gdf_tracts = pd.concat(geoid_list, ignore_index=True)
        # Standardize GEOID column name
        for col in gdf_tracts.columns:
            if "geoid" in col.lower():
                gdf_tracts = gdf_tracts.rename(columns={col: "GEOID"})
                break
        # Merge population data
        gdf_tracts["GEOID"] = gdf_tracts["GEOID"].astype(str).str.strip()
        df_census["geoid"]  = df_census["geoid"].astype(str).str.strip()
        gdf_merged = gdf_tracts.merge(df_census, left_on="GEOID", right_on="geoid", how="left")

        # Compute population density (people per sq km)
        gdf_merged = gdf_merged.to_crs(epsg=32618)   # UTM zone 18N, meters
        gdf_merged["area_sqkm"]  = gdf_merged.geometry.area / 1e6
        gdf_merged["pop_density"] = (gdf_merged["population"] / gdf_merged["area_sqkm"]).round(1)
        gdf_merged = gdf_merged.to_crs(epsg=4326)     # back to WGS84

        geojson_path = RAW_DIR / "census_tracts.geojson"
        gdf_merged.to_file(geojson_path, driver="GeoJSON")
        print(f"Saved tract geometries -> {geojson_path}")
        return gdf_merged
    else:
        print("No tract geometries retrieved — skipping GeoJSON output.")
        # Return a lightweight GeoDataFrame from the CSV alone (no geometry)
        return gpd.GeoDataFrame(df_census)


# ─────────────────────────────────────────────────────────────
# 2. OSM Bus Stops
# ─────────────────────────────────────────────────────────────

def fetch_bus_stops():
    """
    Pull all bus stops in NYC from OpenStreetMap.
    Saves: raw/bus_stops.geojson
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFetching NYC bus stops from OpenStreetMap...")
    print("  (This may take 1-2 minutes depending on OSM server load)")

    tags = {"highway": "bus_stop"}

    try:
        gdf_stops = ox.features_from_bbox(
            bbox=NYC_BBOX,
            tags=tags,
        )
    except Exception as e:
        print(f"  osmnx features_from_bbox failed: {e}")
        print("  Trying Overpass API directly...")
        gdf_stops = _fetch_bus_stops_overpass()

    # Keep only point geometries (some OSM bus stops are ways/areas)
    gdf_stops = gdf_stops[gdf_stops.geometry.geom_type == "Point"].copy()

    # Keep useful columns
    keep_cols = ["geometry"]
    for col in ["name", "ref", "network", "operator", "route_ref"]:
        if col in gdf_stops.columns:
            keep_cols.append(col)
    gdf_stops = gdf_stops[keep_cols].reset_index(drop=True)
    gdf_stops = gdf_stops.to_crs(epsg=4326)

    out_path = RAW_DIR / "bus_stops.geojson"
    gdf_stops.to_file(out_path, driver="GeoJSON")
    print(f"Saved {len(gdf_stops):,} bus stops -> {out_path}")
    return gdf_stops


def _fetch_bus_stops_overpass() -> gpd.GeoDataFrame:
    """Fallback: query Overpass API directly if osmnx fails."""
    query = """
    [out:json][timeout:60];
    node["highway"="bus_stop"]
      (40.4774,-74.2591,40.9176,-73.7004);
    out body;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=90,
    )
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    records = []
    for el in elements:
        if el.get("type") == "node":
            records.append({
                "geometry": Point(el["lon"], el["lat"]),
                "name":     el.get("tags", {}).get("name", ""),
                "ref":      el.get("tags", {}).get("ref", ""),
            })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf


# ─────────────────────────────────────────────────────────────
# 3. Join to Station List → station_features.csv
# ─────────────────────────────────────────────────────────────

def build_station_features(gdf_census: gpd.GeoDataFrame, gdf_stops: gpd.GeoDataFrame):
    """
    For each subway station from mta_parser output, compute:
      - pop_density_500m   : population density of the tract the station sits in
      - population_500m    : sum of population in tracts intersecting 500m buffer
      - bus_stops_250m     : # of bus stops within 250m
      - bus_stops_500m     : # of bus stops within 500m
    Saves: raw/station_features.csv
    """
    station_json = OUTPUT_DIR / "stations.json"
    if not station_json.exists():
        print(f"\n[warn] {station_json} not found — run mta_parser.py first.")
        return

    with open(station_json) as f:
        stations = json.load(f)

    df_stations = pd.DataFrame(stations)
    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["lon"], df_stations["lat"]),
        crs="EPSG:4326",
    )

    # Project everything to a meter-based CRS for distance calculations
    CRS_M = "EPSG:32618"
    gdf_stations_m = gdf_stations.to_crs(CRS_M)
    gdf_stops_m    = gdf_stops.to_crs(CRS_M) if not gdf_stops.empty else gdf_stops

    has_census_geo = (
        not gdf_census.empty
        and hasattr(gdf_census, "geometry")
        and gdf_census.geometry.notna().any()
    )
    if has_census_geo:
        gdf_census_m = gdf_census.to_crs(CRS_M)

    print(f"\nBuilding feature table for {len(gdf_stations):,} stations...")

    rows = []
    for idx, station in gdf_stations_m.iterrows():
        pt = station.geometry
        row = {
            "station_complex_id":   station["station_complex_id"],
            "name":                 station["name"],
            "borough":              station["borough"],
            "lat":                  station["lat"],
            "lon":                  station["lon"],
            "lines":                ",".join(station["lines"]) if isinstance(station["lines"], list) else station["lines"],
            "num_lines":            len(station["lines"]) if isinstance(station["lines"], list) else 1,
            "total_ridership":      station.get("total_ridership", 0),
            "avg_ridership_per_hour": station.get("avg_ridership_per_hour", 0),
        }

        # ── Census features ───────────────────────────────────────────────
        if has_census_geo:
            buf_500 = pt.buffer(500)

            # Which tract does this station fall in?
            containing = gdf_census_m[gdf_census_m.geometry.contains(pt)]
            if not containing.empty:
                row["pop_density_tract"] = containing.iloc[0].get("pop_density", 0)
                row["median_income"]     = containing.iloc[0].get("median_income", 0)
                row["commuters_tract"]   = containing.iloc[0].get("commuters", 0)
            else:
                row["pop_density_tract"] = 0
                row["median_income"]     = 0
                row["commuters_tract"]   = 0

            # Population sum within 500m buffer (weighted by overlap fraction)
            overlapping = gdf_census_m[gdf_census_m.geometry.intersects(buf_500)].copy()
            if not overlapping.empty:
                overlapping["overlap_area"] = overlapping.geometry.intersection(buf_500).area
                overlapping["tract_area"]   = overlapping.geometry.area
                overlapping["weight"]       = overlapping["overlap_area"] / overlapping["tract_area"]
                pop_col = "population" if "population" in overlapping.columns else None
                row["population_500m"] = int(
                    (overlapping["weight"] * overlapping[pop_col]).sum()
                ) if pop_col else 0
            else:
                row["population_500m"] = 0
        else:
            row["pop_density_tract"] = None
            row["median_income"]     = None
            row["commuters_tract"]   = None
            row["population_500m"]   = None

        # ── Bus stop features ─────────────────────────────────────────────
        if not gdf_stops_m.empty:
            dists = gdf_stops_m.geometry.distance(pt)
            row["bus_stops_250m"] = int((dists <= 250).sum())
            row["bus_stops_500m"] = int((dists <= 500).sum())
        else:
            row["bus_stops_250m"] = 0
            row["bus_stops_500m"] = 0

        rows.append(row)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(gdf_stations)} stations processed...")

    df_feat = pd.DataFrame(rows)

    out_path = RAW_DIR / "station_features.csv"
    df_feat.to_csv(out_path, index=False)
    print(f"\nSaved station feature table -> {out_path}")
    print(f"Shape: {df_feat.shape}")
    print("\nColumn summary:")
    print(df_feat.describe(include="all").T[["count","mean","min","max"]].to_string())
    return df_feat


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fetch census + OSM feature data for MTA model")
    p.add_argument("--census-key",    required=True,
                   help="Census API key from https://api.census.gov/data/key_signup.html")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip re-downloading raw files; just rebuild station_features.csv")
    p.add_argument("--skip-census",   action="store_true",
                   help="Skip census fetch (use if you already have census_tracts.geojson)")
    p.add_argument("--skip-osm",      action="store_true",
                   help="Skip OSM fetch (use if you already have bus_stops.geojson)")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Census ────────────────────────────────────────────────
    census_geojson = RAW_DIR / "census_tracts.geojson"
    census_csv     = RAW_DIR / "census_population.csv"

    if args.skip_download or args.skip_census:
        if census_geojson.exists():
            print(f"Loading existing {census_geojson}...")
            gdf_census = gpd.read_file(census_geojson)
        elif census_csv.exists():
            print(f"Loading existing {census_csv} (no geometry)...")
            gdf_census = gpd.GeoDataFrame(pd.read_csv(census_csv))
        else:
            print("[warn] No census file found — skipping census features.")
            gdf_census = gpd.GeoDataFrame()
    else:
        gdf_census = fetch_census(args.census_key)

    # ── Bus stops ─────────────────────────────────────────────
    stops_path = RAW_DIR / "bus_stops.geojson"

    if args.skip_download or args.skip_osm:
        if stops_path.exists():
            print(f"Loading existing {stops_path}...")
            gdf_stops = gpd.read_file(stops_path)
        else:
            print("[warn] No bus stops file found — skipping OSM features.")
            gdf_stops = gpd.GeoDataFrame()
    else:
        gdf_stops = fetch_bus_stops()

    # ── Join to stations ──────────────────────────────────────
    build_station_features(gdf_census, gdf_stops)

    print("\nDone. Files written to ./raw/:")
    for f in sorted(RAW_DIR.glob("*")):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<35} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
