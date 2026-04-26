"""
Census Data Parser
==================
Parses NYC census tract datasets into model-friendly tabular outputs.

Inputs (defaults):
  raw/census_tracts.geojson
  raw/census_population.csv

This parser is dependency-light (Python stdlib only), so it works even when
geopandas/shapely/pandas are unavailable in the active environment.

Outputs (in --output-dir):
  census_tract_features.csv   - normalized per-tract feature table
  census_borough_summary.csv  - borough-level aggregated metrics
  census_summary.json         - high-level totals and QA counters

Usage:
  python3 census_parser.py
  python3 census_parser.py --input-geojson raw/census_tracts.geojson --input-csv raw/census_population.csv
  python3 census_parser.py --output-dir processed
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

COUNTY_TO_BOROUGH = {
    "005": "Bronx",
    "047": "Brooklyn",
    "061": "Manhattan",
    "081": "Queens",
    "085": "Staten Island",
}


def to_float(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def normalize_geoid(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    return s.zfill(11 if len(s) <= 11 else len(s))


def geoid_parts(geoid: str) -> Tuple[str, str, str]:
    # GEOID format for tract: SSCCCCTTTTTT (11 chars)
    g = normalize_geoid(geoid)
    if len(g) >= 11:
        state = g[:2]
        county = g[2:5]
        tract = g[5:11]
        return state, county, tract
    return "", "", ""


def polygon_bbox(geometry: dict) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (min_lon, min_lat, max_lon, max_lat) from GeoJSON Polygon/MultiPolygon geometry."""
    if not geometry:
        return (None, None, None, None)

    coords = geometry.get("coordinates")
    gtype = geometry.get("type")
    if not coords or not gtype:
        return (None, None, None, None)

    points: List[Tuple[float, float]] = []

    try:
        if gtype == "Polygon":
            # Polygon: [ring[ [lon,lat], ... ], ...]
            for ring in coords:
                for pt in ring:
                    if isinstance(pt, list) and len(pt) >= 2:
                        points.append((float(pt[0]), float(pt[1])))
        elif gtype == "MultiPolygon":
            # MultiPolygon: [polygon[ring[[lon,lat], ...], ...], ...]
            for poly in coords:
                for ring in poly:
                    for pt in ring:
                        if isinstance(pt, list) and len(pt) >= 2:
                            points.append((float(pt[0]), float(pt[1])))
    except (TypeError, ValueError):
        return (None, None, None, None)

    if not points:
        return (None, None, None, None)

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return (min(lons), min(lats), max(lons), max(lats))


def read_population_csv(path: Path) -> Dict[str, dict]:
    by_geoid: Dict[str, dict] = {}
    if not path.exists():
        return by_geoid

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geoid = normalize_geoid(row.get("geoid", ""))
            if not geoid:
                continue

            by_geoid[geoid] = {
                "geoid": geoid,
                "name": (row.get("name") or "").strip(),
                "county": (row.get("county") or "").strip().zfill(3),
                "tract": (row.get("tract") or "").strip(),
                "population": to_int(row.get("population"), 0),
                "commuters": to_int(row.get("commuters"), 0),
                "median_income": to_int(row.get("median_income"), 0),
            }

    return by_geoid


TRACT_FIELDS = [
    "geoid", "state", "county", "borough", "tract", "name",
    "population", "commuters", "median_income", "area_sqkm", "pop_density",
    "commuter_ratio", "centroid_lat", "centroid_lon",
    "bbox_min_lat", "bbox_min_lon", "bbox_max_lat", "bbox_max_lon",
]


def parse_census(geojson_path: Path, csv_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    pop_lookup = read_population_csv(csv_path)

    if not geojson_path.exists() and not pop_lookup:
        raise FileNotFoundError("No census input files found. Provide at least one of geojson/csv.")

    rows = []
    missing_geoid = 0
    rows_from_geojson = 0

    if geojson_path.exists():
        with geojson_path.open("r", encoding="utf-8") as f:
            geo = json.load(f)

        for feat in geo.get("features", []):
            rows_from_geojson += 1
            props = feat.get("properties", {}) or {}
            geoid = normalize_geoid(props.get("GEOID") or props.get("geoid") or "")
            if not geoid:
                missing_geoid += 1
                continue

            state, county_from_geoid, tract_from_geoid = geoid_parts(geoid)
            csv_row = pop_lookup.get(geoid, {})

            county = (
                str(props.get("county") or "").strip().zfill(3)
                or str(csv_row.get("county") or "").strip().zfill(3)
                or county_from_geoid
            )
            tract = (
                str(props.get("tract") or "").strip()
                or str(csv_row.get("tract") or "").strip()
                or tract_from_geoid
            )
            name = (
                str(props.get("name") or "").strip()
                or str(csv_row.get("name") or "").strip()
                or str(props.get("NAME") or "").strip()
            )

            population = to_int(props.get("population"), to_int(csv_row.get("population"), 0))
            commuters = to_int(props.get("commuters"), to_int(csv_row.get("commuters"), 0))
            median_income = to_int(props.get("median_income"), to_int(csv_row.get("median_income"), 0))
            area_sqkm = to_float(props.get("area_sqkm"), 0.0)
            pop_density = to_float(props.get("pop_density"), 0.0)

            # If density missing but area+population present, derive density.
            if pop_density <= 0 and area_sqkm > 0 and population > 0:
                pop_density = round(population / area_sqkm, 1)

            min_lon, min_lat, max_lon, max_lat = polygon_bbox(feat.get("geometry") or {})

            rows.append(
                {
                    "geoid": geoid,
                    "state": state,
                    "county": county,
                    "borough": COUNTY_TO_BOROUGH.get(county, "Unknown"),
                    "tract": tract,
                    "name": name,
                    "population": population,
                    "commuters": commuters,
                    "median_income": median_income,
                    "area_sqkm": round(area_sqkm, 6) if area_sqkm else 0.0,
                    "pop_density": round(pop_density, 1),
                    "centroid_lat": round((min_lat + max_lat) / 2, 6) if min_lat is not None and max_lat is not None else None,
                    "centroid_lon": round((min_lon + max_lon) / 2, 6) if min_lon is not None and max_lon is not None else None,
                    "bbox_min_lat": min_lat,
                    "bbox_min_lon": min_lon,
                    "bbox_max_lat": max_lat,
                    "bbox_max_lon": max_lon,
                }
            )

    # Add CSV-only GEOIDs that were absent from geojson.
    seen_geoids = {r["geoid"] for r in rows}
    for geoid, c in pop_lookup.items():
        if geoid in seen_geoids:
            continue
        state, county_from_geoid, tract_from_geoid = geoid_parts(geoid)
        county = (c.get("county") or county_from_geoid or "").zfill(3)
        tract = c.get("tract") or tract_from_geoid
        rows.append(
            {
                "geoid": geoid,
                "state": state,
                "county": county,
                "borough": COUNTY_TO_BOROUGH.get(county, "Unknown"),
                "tract": tract,
                "name": c.get("name", ""),
                "population": to_int(c.get("population"), 0),
                "commuters": to_int(c.get("commuters"), 0),
                "median_income": to_int(c.get("median_income"), 0),
                "area_sqkm": 0.0,
                "pop_density": 0.0,
                "centroid_lat": None,
                "centroid_lon": None,
                "bbox_min_lat": None,
                "bbox_min_lon": None,
                "bbox_max_lat": None,
                "bbox_max_lon": None,
            }
        )

    if not rows:
        raise RuntimeError("Parser produced 0 rows. Check input files.")

    # Keep one row per geoid (prefer rows with area info and non-zero population).
    best_by_geoid: Dict[str, dict] = {}
    for row in rows:
        geoid = row["geoid"]
        rank = (2 if to_float(row.get("area_sqkm"), 0.0) > 0 else 0) + (
            1 if to_int(row.get("population"), 0) > 0 else 0
        )
        current = best_by_geoid.get(geoid)
        if current is None:
            row["_rank"] = rank
            best_by_geoid[geoid] = row
        else:
            if rank > current.get("_rank", -1):
                row["_rank"] = rank
                best_by_geoid[geoid] = row

    tract_rows = []
    for geoid in sorted(best_by_geoid.keys()):
        r = best_by_geoid[geoid]
        population = to_int(r.get("population"), 0)
        commuters = to_int(r.get("commuters"), 0)
        r["commuter_ratio"] = round((commuters / population), 4) if population > 0 else 0.0
        r.pop("_rank", None)
        tract_rows.append(r)

    tract_out = output_dir / "census_tract_features.csv"
    with tract_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACT_FIELDS)
        w.writeheader()
        for row in tract_rows:
            w.writerow({k: row.get(k) for k in TRACT_FIELDS})

    borough_stats: Dict[str, dict] = {}
    for r in tract_rows:
        b = r.get("borough") or "Unknown"
        s = borough_stats.setdefault(
            b,
            {
                "borough": b,
                "tract_count": 0,
                "population_total": 0,
                "commuters_total": 0,
                "median_income_values": [],
                "area_sqkm_total": 0.0,
            },
        )
        s["tract_count"] += 1
        s["population_total"] += to_int(r.get("population"), 0)
        s["commuters_total"] += to_int(r.get("commuters"), 0)
        inc = to_int(r.get("median_income"), 0)
        if inc > 0:
            s["median_income_values"].append(inc)
        s["area_sqkm_total"] += to_float(r.get("area_sqkm"), 0.0)

    borough_rows = []
    for borough in sorted(borough_stats.keys()):
        s = borough_stats[borough]
        population_total = s["population_total"]
        commuters_total = s["commuters_total"]
        area_total = s["area_sqkm_total"]
        median_values = s["median_income_values"]
        borough_rows.append(
            {
                "borough": borough,
                "tract_count": s["tract_count"],
                "population_total": population_total,
                "commuters_total": commuters_total,
                "median_income_median": int(statistics.median(median_values)) if median_values else 0,
                "area_sqkm_total": round(area_total, 6),
                "pop_density_weighted": round((population_total / area_total), 1) if area_total > 0 else 0.0,
                "commuter_ratio": round((commuters_total / population_total), 4) if population_total > 0 else 0.0,
            }
        )

    borough_out = output_dir / "census_borough_summary.csv"
    with borough_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "borough",
                "tract_count",
                "population_total",
                "commuters_total",
                "median_income_median",
                "area_sqkm_total",
                "pop_density_weighted",
                "commuter_ratio",
            ],
        )
        w.writeheader()
        w.writerows(borough_rows)

    summary = {
        "inputs": {
            "geojson": str(geojson_path),
            "csv": str(csv_path),
            "geojson_rows": rows_from_geojson,
            "csv_rows": len(pop_lookup),
        },
        "quality": {
            "missing_geoid_in_geojson": missing_geoid,
            "output_tract_rows": int(len(tract_rows)),
            "output_borough_rows": int(len(borough_rows)),
        },
        "totals": {
            "population_total": int(sum(to_int(r.get("population"), 0) for r in tract_rows)),
            "commuters_total": int(sum(to_int(r.get("commuters"), 0) for r in tract_rows)),
            "area_sqkm_total": float(round(sum(to_float(r.get("area_sqkm"), 0.0) for r in tract_rows), 4)),
        },
    }

    summary_out = output_dir / "census_summary.json"
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[1] census_tract_features.csv  -> {tract_out}")
    print(f"[2] census_borough_summary.csv -> {borough_out}")
    print(f"[3] census_summary.json        -> {summary_out}")
    print(f"Rows: {len(tract_rows):,} tracts across {len(borough_rows)} borough groups")


def main():
    p = argparse.ArgumentParser(description="Parse NYC census inputs into tract and borough feature tables")
    p.add_argument("--input-geojson", default="raw/census_tracts.geojson")
    p.add_argument("--input-csv", default="raw/census_population.csv")
    p.add_argument("--output-dir", default="processed")
    args = p.parse_args()

    parse_census(Path(args.input_geojson), Path(args.input_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
