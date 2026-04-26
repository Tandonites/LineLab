"""
MTA Bus Hourly Ridership Parser
===============================
Parses MTA bus ridership CSV files into model-friendly outputs.

It is intentionally tolerant of schema differences by supporting common
column aliases (e.g., stop_id vs station_complex_id, lat vs latitude).

Expected (or alias-compatible) fields:
  transit_timestamp, transit_mode, stop_id, stop_name, borough,
  payment_method, fare_class_category, ridership, transfers,
  latitude, longitude, route_id, trip_id, direction_id, stop_sequence

Outputs (in --output-dir):
  bus_stops.json              - unique stops with coords, routes, borough
  bus_ridership_daily.csv     - ridership per stop per day
  bus_ridership_hourly.csv    - cleaned hourly records (optional)
  bus_route_summary.csv       - per-route totals
  bus_hourly_patterns.json    - demand matrix: route_group x hour x day_of_week
  bus_network_graph.json      - inferred stop graph for simulation

Usage:
  python3 bus_parser.py --input raw/MTA_Bus_Hourly_Ridership.csv --output-dir processed
  python3 bus_parser.py --input raw/data.csv --max-rows 1000000 --skip-hourly
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import Optional


# ── Data Model ───────────────────────────────────────────────

@dataclass
class BusStop:
    stop_id: str
    name: str
    routes: list
    borough: str
    lat: float
    lon: float
    total_ridership: int = 0
    total_transfers: int = 0
    record_count: int = 0


# ── Helpers ───────────────────────────────────────────────────

COLUMN_ALIASES = {
    "timestamp": ["transit_timestamp", "timestamp", "datetime", "date_time"],
    "mode": ["transit_mode", "mode"],
    "stop_id": ["stop_id", "stop_code", "stopid", "station_complex_id"],
    "stop_name": ["stop_name", "stop", "station_complex", "stop_description"],
    "borough": ["borough", "boro"],
    "payment": ["payment_method", "payment"],
    "fare_class": ["fare_class_category", "fare_class"],
    "ridership": ["ridership", "riders", "boardings"],
    "transfers": ["transfers"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng"],
    "route": ["route_id", "route", "line", "route_short_name", "routes"],
    "trip_id": ["trip_id", "trip"],
    "direction": ["direction_id", "direction"],
    "stop_sequence": ["stop_sequence", "sequence", "seq"],
}


def pick(row: dict, logical_name: str, default: str = "") -> str:
    for col in COLUMN_ALIASES[logical_name]:
        if col in row and row[col] not in (None, ""):
            return str(row[col]).strip()
    return default


def parse_timestamp(ts: str) -> Optional[datetime]:
    for fmt in (
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_routes(route_raw: str, stop_name: str) -> list:
    routes = []

    # Direct route field first: "B12", "M1,M2", "B12|B13", etc.
    if route_raw:
        for tok in re.split(r"[|,;/ ]+", route_raw):
            tok = tok.strip().upper()
            if tok:
                routes.append(tok)

    # Also try parsing stop names like: "Flatbush Av (B41,B44)"
    match = re.search(r"\(([^)]+)\)", stop_name or "")
    if match:
        for tok in match.group(1).split(","):
            tok = tok.strip().upper()
            if tok:
                routes.append(tok)

    # Stable unique preserving order
    out = []
    seen = set()
    for r in routes:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def route_group(route: str) -> str:
    r = route.upper().strip()
    if r.startswith("BX"):
        return "bronx"
    if r.startswith("BM"):
        return "brooklyn_express"
    if r.startswith("QM"):
        return "queens_express"
    if r.startswith("SIM"):
        return "staten_express"
    if r.startswith("M"):
        return "manhattan"
    if r.startswith("B"):
        return "brooklyn"
    if r.startswith("Q"):
        return "queens"
    if r.startswith("S"):
        return "staten"
    if r.startswith("X"):
        return "express"
    return "other"


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = radians(lat1), radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(p1) * cos(p2) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


# ── Core Parser ───────────────────────────────────────────────

def parse(input_path: str, max_rows: int, skip_hourly: bool, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    stops = {}
    hourly_records = []
    daily = defaultdict(lambda: {"ridership": 0, "transfers": 0, "count": 0})
    route_stats = defaultdict(lambda: {"ridership": 0, "transfers": 0, "stops": set()})
    hourly_pattern = defaultdict(int)

    # For network graph inference:
    # If trip_id+stop_sequence exists we build true consecutive edges;
    # otherwise we fall back to sparse geo-neighbor edges per route.
    route_trip_seq = defaultdict(lambda: defaultdict(list))

    print(f"Parsing: {input_path}")
    print(f"Max rows: {max_rows:,} | Skip hourly CSV: {skip_hourly}")
    print("-" * 50)

    row_count = 0
    skipped = 0
    missing_required = 0

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row_count >= max_rows:
                print(f"  [cap] Reached {max_rows:,} rows.")
                break

            try:
                ts = parse_timestamp(pick(row, "timestamp"))
                if ts is None:
                    skipped += 1
                    continue

                mode = pick(row, "mode").lower()
                if mode and "bus" not in mode:
                    # If transit_mode exists and is not bus, ignore row.
                    continue

                stop_id = pick(row, "stop_id")
                stop_name = pick(row, "stop_name")
                lat = to_float(pick(row, "lat"), default=float("nan"))
                lon = to_float(pick(row, "lon"), default=float("nan"))
                if not stop_id or not stop_name or lat != lat or lon != lon:
                    missing_required += 1
                    continue

                borough = pick(row, "borough")
                payment = pick(row, "payment")
                fare_class = pick(row, "fare_class")
                ridership = to_int(pick(row, "ridership"), default=0)
                transfers = to_int(pick(row, "transfers"), default=0)

                route_raw = pick(row, "route")
                routes = parse_routes(route_raw, stop_name)
                main_route = routes[0] if routes else "UNKNOWN"

                hour = ts.hour
                dow = ts.weekday()
                date_str = ts.strftime("%Y-%m-%d")
                grp = route_group(main_route)

                if stop_id not in stops:
                    stops[stop_id] = BusStop(
                        stop_id=stop_id,
                        name=stop_name,
                        routes=routes,
                        borough=borough,
                        lat=lat,
                        lon=lon,
                    )
                s = stops[stop_id]

                # Merge routes seen across rows.
                if routes:
                    merged = []
                    seen_routes = set()
                    for r in s.routes + routes:
                        if r and r not in seen_routes:
                            seen_routes.add(r)
                            merged.append(r)
                    s.routes = merged

                s.total_ridership += ridership
                s.total_transfers += transfers
                s.record_count += 1

                key = (stop_id, date_str)
                daily[key]["ridership"] += ridership
                daily[key]["transfers"] += transfers
                daily[key]["count"] += 1

                for r in routes if routes else ["UNKNOWN"]:
                    route_stats[r]["ridership"] += ridership
                    route_stats[r]["transfers"] += transfers
                    route_stats[r]["stops"].add(stop_id)

                hourly_pattern[(grp, hour, dow)] += ridership

                if not skip_hourly:
                    hourly_records.append({
                        "timestamp": ts.isoformat(),
                        "hour": hour,
                        "day_of_week": dow,
                        "is_weekend": int(dow >= 5),
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "route": main_route,
                        "route_group": grp,
                        "routes": ",".join(routes),
                        "borough": borough,
                        "payment": payment,
                        "fare_class": fare_class,
                        "ridership": ridership,
                        "transfers": transfers,
                        "lat": lat,
                        "lon": lon,
                    })

                trip_id = pick(row, "trip_id")
                seq_raw = pick(row, "stop_sequence")
                if trip_id and seq_raw and routes:
                    seq = to_int(seq_raw, default=-1)
                    if seq >= 0:
                        for r in routes:
                            route_trip_seq[r][trip_id].append((seq, stop_id))

                row_count += 1
                if row_count % 100_000 == 0:
                    print(f"  Processed {row_count:,} rows...")

            except (ValueError, KeyError):
                skipped += 1

    print(f"\nDone. Parsed: {row_count:,} | Skipped: {skipped:,}")
    print(f"Missing required stop fields: {missing_required:,}")
    print(f"Unique stops: {len(stops):,} | Unique routes: {len(route_stats):,}")

    # 1) bus_stops.json
    stop_list = []
    for s in stops.values():
        d = asdict(s)
        d["avg_ridership_per_hour"] = round(s.total_ridership / max(s.record_count, 1), 2)
        stop_list.append(d)
    stop_list.sort(key=lambda x: x["total_ridership"], reverse=True)
    out = os.path.join(output_dir, "bus_stops.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stop_list, f, indent=2)
    print(f"\n[1] bus_stops.json         -> {out}")

    # 2) bus_ridership_daily.csv
    out = os.path.join(output_dir, "bus_ridership_daily.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stop_id", "stop_name", "borough", "routes", "lat", "lon",
            "date", "ridership", "transfers", "hourly_records",
        ])
        for (sid, date), vals in sorted(daily.items()):
            s = stops.get(sid)
            if s:
                w.writerow([
                    sid, s.name, s.borough, ",".join(s.routes), s.lat, s.lon,
                    date, vals["ridership"], vals["transfers"], vals["count"],
                ])
    print(f"[2] bus_ridership_daily.csv -> {out}")

    # 3) bus_route_summary.csv
    out = os.path.join(output_dir, "bus_route_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route", "route_group", "total_ridership", "total_transfers", "stop_count"])
        for route, stats in sorted(route_stats.items(), key=lambda x: -x[1]["ridership"]):
            w.writerow([
                route,
                route_group(route),
                stats["ridership"],
                stats["transfers"],
                len(stats["stops"]),
            ])
    print(f"[3] bus_route_summary.csv   -> {out}")

    # 4) bus_hourly_patterns.json
    patterns = {}
    for (grp, hour, dow), total in hourly_pattern.items():
        patterns.setdefault(grp, {})[f"h{hour:02d}_d{dow}"] = total
    out = os.path.join(output_dir, "bus_hourly_patterns.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(patterns, f, indent=2)
    print(f"[4] bus_hourly_patterns.json -> {out}")

    # 5) bus_network_graph.json
    nodes = [
        {
            "id": s.stop_id,
            "name": s.name,
            "lat": s.lat,
            "lon": s.lon,
            "borough": s.borough,
            "routes": s.routes,
            "ridership": s.total_ridership,
        }
        for s in stops.values()
    ]

    edges = []
    seen = set()

    # Preferred: consecutive stops from trip_id + stop_sequence
    for route, trip_map in route_trip_seq.items():
        for _, seq_list in trip_map.items():
            seq_list.sort(key=lambda x: x[0])
            for i in range(1, len(seq_list)):
                a, b = seq_list[i - 1][1], seq_list[i][1]
                if a == b:
                    continue
                u, v = min(a, b), max(a, b)
                key = (u, v, route)
                if key not in seen:
                    seen.add(key)
                    edges.append({"source": u, "target": v, "route": route, "inferred": "trip_sequence"})

    # Fallback when no sequence data was present for a route: geo-neighbor chain
    routes_without_sequence = [r for r in route_stats.keys() if r not in route_trip_seq]
    for route in routes_without_sequence:
        stop_ids = sorted(route_stats[route]["stops"])
        if len(stop_ids) < 2:
            continue

        route_points = []
        for sid in stop_ids:
            s = stops.get(sid)
            if s:
                route_points.append((sid, s.lat, s.lon))

        if len(route_points) < 2:
            continue

        # Build a sparse nearest-neighbor walk to avoid dense all-pairs graph.
        remaining = {sid for sid, _, _ in route_points}
        current = route_points[0][0]
        remaining.remove(current)
        coord = {sid: (lat, lon) for sid, lat, lon in route_points}

        while remaining:
            lat1, lon1 = coord[current]
            nxt = min(
                remaining,
                key=lambda sid: haversine_m(lat1, lon1, coord[sid][0], coord[sid][1]),
            )
            u, v = min(current, nxt), max(current, nxt)
            key = (u, v, route)
            if key not in seen:
                seen.add(key)
                edges.append({"source": u, "target": v, "route": route, "inferred": "geo_neighbor"})
            current = nxt
            remaining.remove(nxt)

    out = os.path.join(output_dir, "bus_network_graph.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
    print(f"[5] bus_network_graph.json  -> {out}")

    # 6) bus_ridership_hourly.csv (optional)
    if not skip_hourly and hourly_records:
        out = os.path.join(output_dir, "bus_ridership_hourly.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hourly_records[0].keys())
            w.writeheader()
            w.writerows(hourly_records)
        print(f"[6] bus_ridership_hourly.csv -> {out}  ({len(hourly_records):,} rows)")

    print(f"\nAll outputs in: {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MTA Bus Hourly Ridership Parser")
    p.add_argument("--input", required=True, help="Path to bus ridership CSV")
    p.add_argument("--max-rows", type=int, default=5_000_000)
    p.add_argument("--skip-hourly", action="store_true")
    p.add_argument("--output-dir", default="./processed")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    parse(args.input, args.max_rows, args.skip_hourly, args.output_dir)


if __name__ == "__main__":
    main()
