"""
fetch_features.py
==================
Pulls two external datasets needed for the MTA ridership model:

  1. ACS 5-year census estimates (population + job proxy) at census tract level
     Source: Census Bureau API  ->  ./raw/census_tracts.geojson
                                    ./raw/census_population.csv

  2. OSM bus stop locations across NYC
     Source: OpenStreetMap via osmnx / Overpass API  ->  ./raw/bus_stops.geojson

  3. Joins both to the parsed station list and writes a combined feature table:
                                    ./raw/station_features.csv

Usage:
  # Set CENSUS_API_KEY in .env (get a free key at https://api.census.gov/data/key_signup.html)
  python3 fetch_features.py

  # Or pass key directly
  python3 fetch_features.py --census-key YOUR_KEY

  # Already have raw files? Just rebuild station_features.csv
  python3 fetch_features.py --skip-census --skip-osm

  # Skip everything except the join step
  python3 fetch_features.py --skip-download

Requirements:
  pip install census osmnx geopandas requests shapely pandas python-dotenv
"""

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
from pathlib import Path
from shapely.geometry import Point
from census import Census
from requests import RequestException

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RAW_DIR    = Path("./raw")
OUTPUT_DIR = Path("./processed")   # where mta_parser.py wrote stations.json

# NYC bounding box  (S, W, N, E)
NYC_BBOX = (40.4774, -74.2591, 40.9176, -73.7004)

# ACS variables:
#   B01003_001E = total population
#   B08301_001E = total commuters (job-adjacent demand proxy)
#   B19013_001E = median household income
ACS_VARS = {
    "B01003_001E": "population",
    "B08301_001E": "commuters",
    "B19013_001E": "median_income",
}


# ─────────────────────────────────────────────────────────────
# 1. Census ACS Data
# ─────────────────────────────────────────────────────────────

def fetch_census(api_key: str) -> gpd.GeoDataFrame:
    """
    Pull ACS 5-year estimates for all census tracts in NYC's 5 counties.
    Saves:
      raw/census_population.csv  — flat table per tract
      raw/census_tracts.geojson  — tract polygons with population attached
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    c = Census(api_key)

    # NYC county FIPS: Manhattan=061, Brooklyn=047, Queens=081, Bronx=005, SI=085
    nyc_counties = ["061", "047", "081", "005", "085"]
    STATE_FIPS   = "36"

    all_rows = []
    var_list = list(ACS_VARS.keys())

    print("Fetching ACS 5-year estimates (2022)...")
    for county in nyc_counties:
        print(f"  County FIPS {county}...", end=" ", flush=True)
        try:
            results = c.acs5.state_county_tract(
                fields      = ["NAME"] + var_list,
                state_fips  = STATE_FIPS,
                county_fips = county,
                tract       = Census.ALL,
                year        = 2022,
            )
            for r in results:
                row = {
                    "geoid":  f"{STATE_FIPS}{county}{r['tract']}",
                    "name":   r.get("NAME", ""),
                    "county": county,
                    "tract":  r["tract"],
                }
                for api_col, friendly_col in ACS_VARS.items():
                    val = r.get(api_col, -1)
                    try:
                        row[friendly_col] = int(float(val)) if float(val) >= 0 else 0
                    except (TypeError, ValueError):
                        row[friendly_col] = 0
                all_rows.append(row)
            print(f"{len(results)} tracts")
        except Exception as e:
            print(f"FAILED: {e}")

    df_census = pd.DataFrame(all_rows)
    csv_path  = RAW_DIR / "census_population.csv"
    df_census.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df_census):,} tracts -> {csv_path}")

    # Download tract geometries from Census TIGER/Line via GeoAPI
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
            gdf_county = gpd.read_file(resp.text)
            geoid_list.append(gdf_county)
            print(f"  County {county}: {len(gdf_county)} tracts")
            time.sleep(0.3)
        except Exception as e:
            print(f"  County {county} geometry FAILED: {e}")

    if not geoid_list:
        print("No tract geometries retrieved — returning CSV only.")
        return gpd.GeoDataFrame(df_census)

    gdf_tracts = pd.concat(geoid_list, ignore_index=True)

    # Standardize GEOID column name (varies by API version)
    for col in gdf_tracts.columns:
        if col.upper() == "GEOID":
            gdf_tracts = gdf_tracts.rename(columns={col: "GEOID"})
            break

    gdf_tracts["GEOID"]     = gdf_tracts["GEOID"].astype(str).str.strip()
    df_census["geoid"]      = df_census["geoid"].astype(str).str.strip()
    gdf_merged = gdf_tracts.merge(df_census, left_on="GEOID", right_on="geoid", how="left")

    # Compute population density (people per sq km)
    gdf_merged = gdf_merged.to_crs(epsg=32618)
    gdf_merged["area_sqkm"]   = gdf_merged.geometry.area / 1e6
    gdf_merged["pop_density"] = (
        gdf_merged["population"] / gdf_merged["area_sqkm"].replace(0, float("nan"))
    ).round(1).fillna(0)
    gdf_merged = gdf_merged.to_crs(epsg=4326)

    geojson_path = RAW_DIR / "census_tracts.geojson"
    gdf_merged.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved tract geometries -> {geojson_path}")
    return gdf_merged


# ─────────────────────────────────────────────────────────────
# 2. OSM Bus Stops
# ─────────────────────────────────────────────────────────────

def fetch_bus_stops() -> gpd.GeoDataFrame:
    """
    Pull all bus stops in NYC from OpenStreetMap.
    Tries osmnx first, then falls back to direct Overpass API.
    Saves: raw/bus_stops.geojson
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFetching NYC bus stops from OpenStreetMap...")
    print("  (This may take 1-2 minutes depending on OSM server load)")

    gdf_stops = None

    # Try osmnx first
    try:
        gdf_stops = ox.features_from_bbox(bbox=NYC_BBOX, tags={"highway": "bus_stop"})
        print(f"  osmnx returned {len(gdf_stops):,} features")
    except Exception as e:
        print(f"  osmnx failed: {e}")
        print("  Falling back to direct Overpass API...")
        gdf_stops = _fetch_bus_stops_overpass()

    if gdf_stops is None or gdf_stops.empty:
        print("  [warn] No bus stops retrieved.")
        return gpd.GeoDataFrame()

    # Keep only point geometries
    gdf_stops = gdf_stops[gdf_stops.geometry.geom_type == "Point"].copy()

    # Keep useful columns only
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
    """
    Query Overpass API directly for NYC bus stops.
    Tries multiple public mirrors in order.
    """
    # Primary query: classic OSM tagging for bus stops.
    # Fallback query: includes public_transport=platform + bus=yes used in some areas.
    queries = [
        (
            "[out:json][timeout:120];"
            "("
            'node["highway"="bus_stop"](40.4774,-74.2591,40.9176,-73.7004);'
            ");"
            "out body;"
        ),
        (
            "[out:json][timeout:120];"
            "("
            'node["highway"="bus_stop"](40.4774,-74.2591,40.9176,-73.7004);'
            'node["public_transport"="platform"]["bus"="yes"](40.4774,-74.2591,40.9176,-73.7004);'
            ");"
            "out body;"
        ),
    ]

    # overpass-api.de has been returning 406 — private.coffee (ex-kumi.systems)
    # and mail.ru are reliable fallbacks
    mirrors = [
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    for q_idx, query in enumerate(queries, start=1):
        print(f"  Overpass query {q_idx}/{len(queries)}...")
        for url in mirrors:
            print(f"  Trying {url}...")
            for attempt in range(1, 4):
                try:
                    # Pass form data as dict; requests handles encoding safely.
                    resp = requests.post(
                        url,
                        data={"data": query},
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=(12, 75),
                    )
                    resp.raise_for_status()
                    elements = resp.json().get("elements", [])

                    records = []
                    for el in elements:
                        if el.get("type") == "node" and "lat" in el and "lon" in el:
                            records.append({
                                "geometry": Point(el["lon"], el["lat"]),
                                "name":     el.get("tags", {}).get("name", ""),
                                "ref":      el.get("tags", {}).get("ref", ""),
                                "operator": el.get("tags", {}).get("operator", ""),
                            })

                    if records:
                        print(f"  Got {len(records):,} stops from {url}")
                        df = pd.DataFrame(records).drop_duplicates(subset=["name", "ref", "geometry"])
                        return gpd.GeoDataFrame(df, crs="EPSG:4326")

                    print("  0 results from this mirror/query combo.")
                    break

                except RequestException as e:
                    if attempt == 3:
                        print(f"  Failed ({url}) after 3 attempts: {e}")
                    else:
                        backoff = 1.5 * attempt
                        print(f"  Attempt {attempt}/3 failed ({e}); retrying in {backoff:.1f}s...")
                        time.sleep(backoff)
                except ValueError as e:
                    print(f"  Non-JSON response from {url}: {e}")
                    break

    print("  [warn] All Overpass mirrors failed. Returning empty GeoDataFrame.")
    return gpd.GeoDataFrame()


# ─────────────────────────────────────────────────────────────
# 3. Join to Station List → station_features.csv
# ─────────────────────────────────────────────────────────────

def build_station_features(gdf_census: gpd.GeoDataFrame, gdf_stops: gpd.GeoDataFrame):
    """
    For each subway station in output/stations.json, compute:
      - pop_density_tract  : pop density of the tract the station sits in
      - population_500m    : area-weighted population within 500m
      - median_income      : median HH income of containing tract
      - commuters_tract    : commuter count of containing tract
      - bus_stops_250m     : # OSM bus stops within 250m
      - bus_stops_500m     : # OSM bus stops within 500m
    Saves: raw/station_features.csv
    """
    station_json = OUTPUT_DIR / "stations.json"
    if not station_json.exists():
        print(f"\n[error] {station_json} not found.")
        print("  Run mta_parser.py first:")
        print("  python3 mta_parser.py --input raw/MTA_Subway_Hourly_Ridership__*.csv --output-dir output --skip-hourly")
        return None

    print(f"\nLoading stations from {station_json}...")
    with open(station_json) as f:
        stations = json.load(f)

    df_stations = pd.DataFrame(stations)
    gdf_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=gpd.points_from_xy(df_stations["lon"], df_stations["lat"]),
        crs="EPSG:4326",
    )

    CRS_M = "EPSG:32618"  # UTM 18N, meters — accurate for NYC
    gdf_stations_m = gdf_stations.to_crs(CRS_M)

    has_stops = not gdf_stops.empty and "geometry" in gdf_stops.columns
    gdf_stops_m = gdf_stops.to_crs(CRS_M) if has_stops else None

    has_census = (
        not gdf_census.empty
        and hasattr(gdf_census, "geometry")
        and "geometry" in gdf_census.columns
        and gdf_census.geometry.notna().any()
    )
    gdf_census_m = gdf_census.to_crs(CRS_M) if has_census else None

    print(f"Building features for {len(gdf_stations):,} stations...")
    print(f"  Census geometry available: {has_census}")
    print(f"  Bus stops available:       {has_stops} ({len(gdf_stops):,} stops)" if has_stops else "  Bus stops available:       False")

    rows = []
    total = len(gdf_stations_m)

    for i, (_, station) in enumerate(gdf_stations_m.iterrows()):
        pt = station.geometry

        row = {
            "station_complex_id":     station["station_complex_id"],
            "name":                   station["name"],
            "borough":                station["borough"],
            "lat":                    station["lat"],
            "lon":                    station["lon"],
            "lines":                  ",".join(station["lines"]) if isinstance(station["lines"], list) else station["lines"],
            "num_lines":              len(station["lines"]) if isinstance(station["lines"], list) else 1,
            "total_ridership":        station.get("total_ridership", 0),
            "avg_ridership_per_hour": station.get("avg_ridership_per_hour", 0),
        }

        # ── Census features ───────────────────────────────────
        if has_census:
            buf_500 = pt.buffer(500)

            # Tract containing this station
            containing = gdf_census_m[gdf_census_m.geometry.contains(pt)]
            if not containing.empty:
                t = containing.iloc[0]
                row["pop_density_tract"] = float(t.get("pop_density", 0) or 0)
                row["median_income"]     = int(t.get("median_income") or 0) if pd.notna(t.get("median_income")) else 0
                row["commuters_tract"]   = int(t.get("commuters") or 0) if pd.notna(t.get("commuters")) else 0
            else:
                row["pop_density_tract"] = 0.0
                row["median_income"]     = 0
                row["commuters_tract"]   = 0

            # Area-weighted population within 500m buffer
            overlapping = gdf_census_m[gdf_census_m.geometry.intersects(buf_500)].copy()
            if not overlapping.empty and "population" in overlapping.columns:
                overlapping["overlap_frac"] = (
                    overlapping.geometry.intersection(buf_500).area
                    / overlapping.geometry.area.replace(0, float("nan"))
                ).fillna(0)
                row["population_500m"] = int(
                    (overlapping["overlap_frac"] * overlapping["population"].fillna(0)).sum()
                )
            else:
                row["population_500m"] = 0
        else:
            row["pop_density_tract"] = None
            row["median_income"]     = None
            row["commuters_tract"]   = None
            row["population_500m"]   = None

        # ── Bus stop features ─────────────────────────────────
        if has_stops:
            dists = gdf_stops_m.geometry.distance(pt)
            row["bus_stops_250m"] = int((dists <= 250).sum())
            row["bus_stops_500m"] = int((dists <= 500).sum())
        else:
            row["bus_stops_250m"] = 0
            row["bus_stops_500m"] = 0

        rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} stations processed...")

    df_feat = pd.DataFrame(rows)

    out_path = RAW_DIR / "station_features.csv"
    df_feat.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  shape={df_feat.shape}")
    print("\nFeature summary:")
    numeric = df_feat.select_dtypes(include="number")
    print(numeric.describe().round(1).T[["count", "mean", "min", "max"]].to_string())
    return df_feat


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fetch census + OSM features for MTA model")
    p.add_argument(
        "--census-key",
        default=os.getenv("CENSUS_API_KEY"),
        help="Census API key (or set CENSUS_API_KEY in .env). "
             "Get one free at https://api.census.gov/data/key_signup.html",
    )
    p.add_argument("--skip-download", action="store_true",
                   help="Skip all downloads; just rebuild station_features.csv from existing raw files")
    p.add_argument("--skip-census",   action="store_true",
                   help="Skip census fetch (use existing census_tracts.geojson / census_population.csv)")
    p.add_argument("--skip-osm",      action="store_true",
                   help="Skip OSM fetch (use existing bus_stops.geojson)")
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
            print("[warn] No census file found — census features will be null.")
            gdf_census = gpd.GeoDataFrame()
    else:
        if not args.census_key:
            print("Error: Census API key required. Pass --census-key or set CENSUS_API_KEY in .env")
            sys.exit(1)
        gdf_census = fetch_census(args.census_key)

    # ── Bus stops ─────────────────────────────────────────────
    stops_path = RAW_DIR / "bus_stops.geojson"

    if args.skip_download or args.skip_osm:
        if stops_path.exists():
            print(f"Loading existing {stops_path}...")
            gdf_stops = gpd.read_file(stops_path)
        else:
            print("[warn] No bus stops file found — bus stop features will be 0.")
            gdf_stops = gpd.GeoDataFrame()
    else:
        gdf_stops = fetch_bus_stops()

    # ── Join → station_features.csv ───────────────────────────
    build_station_features(gdf_census, gdf_stops)

    print("\nDone. Files in ./raw/:")
    for f in sorted(RAW_DIR.glob("*")):
        print(f"  {f.name:<40} {f.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()