"""
MTA Subway Hourly Ridership Parser
====================================
Exact schema: transit_timestamp, transit_mode, station_complex_id,
station_complex, borough, payment_method, fare_class_category,
ridership, transfers, latitude, longitude, Georeference

Outputs (in ./output/):
  stations.json        - unique stations with coords, lines, borough
  ridership_daily.csv  - ridership per station per day
  ridership_hourly.csv - full cleaned hourly records (optional, large)
  line_summary.csv     - per subway line totals
  hourly_patterns.json - demand matrix: line_group x hour x day_of_week
  network_graph.json   - adjacency graph for new-line simulation

Usage:
    python3 mta_parser.py --input raw/MTA_Subway_Hourly_Ridership__2020-2024_20260426.csv
    python3 mta_parser.py --input data.csv --max-rows 1000000 --skip-hourly
"""

import csv
import json
import re
import os
import sys
import argparse
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional


# ── Data Model ───────────────────────────────────────────────

@dataclass
class Station:
    station_complex_id: str
    name: str
    lines: list
    borough: str
    lat: float
    lon: float
    total_ridership: int = 0
    total_transfers: int = 0
    record_count: int = 0


# ── Helpers ───────────────────────────────────────────────────

def parse_lines(station_complex: str) -> tuple:
    """
    '111 St (J)'              -> ('111 St', ['J'])
    '23 St (F,M)'             -> ('23 St', ['F', 'M'])
    'Canal St (J,N,Q,R,W,Z,6)' -> ('Canal St', ['J','N','Q','R','W','Z','6'])
    """
    # Some station names contain parenthetical descriptors before the
    # route list, e.g. "Cathedral Pkwy (110 St) (1)".
    # The final parenthetical token is the route list.
    matches = list(re.finditer(r'\(([^)]+)\)', station_complex))
    if matches:
        last_match = matches[-1]
        lines = [l.strip() for l in last_match.group(1).split(',') if l.strip()]
        name  = station_complex[:last_match.start()].strip()
    else:
        lines = []
        name  = station_complex.strip()
    return name, lines


def parse_timestamp(ts: str) -> Optional[datetime]:
    for fmt in ('%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_numeric(value: str) -> float:
    """Parse numeric strings that may include thousands separators."""
    cleaned = (value or '').strip().replace(',', '')
    if cleaned == '':
        return 0.0
    return float(cleaned)


LINE_GROUPS = {
    'A':'blue','C':'blue','E':'blue',
    'B':'orange','D':'orange','F':'orange','M':'orange',
    'N':'yellow','Q':'yellow','R':'yellow','W':'yellow',
    '1':'red','2':'red','3':'red',
    '4':'green','5':'green','6':'green',
    'J':'brown','Z':'brown',
    'L':'grey','G':'lime','7':'purple','S':'shuttle',
}

def line_group(lines: list) -> str:
    for l in lines:
        if l in LINE_GROUPS:
            return LINE_GROUPS[l]
    return 'other'


# ── Core Parser ───────────────────────────────────────────────

def parse(input_path: str, max_rows: int, skip_hourly: bool, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    stations       = {}
    hourly_records = []
    daily          = defaultdict(lambda: {'ridership': 0, 'transfers': 0, 'count': 0})
    line_stats     = defaultdict(lambda: {'ridership': 0, 'transfers': 0, 'stations': set()})
    hourly_pattern = defaultdict(int)

    print(f"Parsing: {input_path}")
    print(f"Max rows: {max_rows:,} | Skip hourly CSV: {skip_hourly}")
    print("-" * 50)

    row_count = 0
    skipped   = 0

    with open(input_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row_count >= max_rows:
                print(f"  [cap] Reached {max_rows:,} rows.")
                break
            try:
                ts = parse_timestamp(row['transit_timestamp'])
                if ts is None:
                    skipped += 1
                    continue

                station_id  = row['station_complex_id'].strip()
                station_raw = row['station_complex'].strip()
                borough     = row['borough'].strip()
                payment     = row['payment_method'].strip()
                fare_class  = row['fare_class_category'].strip()
                ridership   = int(parse_numeric(row['ridership']))
                transfers   = int(parse_numeric(row['transfers']))
                lat         = parse_numeric(row['latitude'])
                lon         = parse_numeric(row['longitude'])

                station_name, lines = parse_lines(station_raw)
                hour     = ts.hour
                dow      = ts.weekday()
                date_str = ts.strftime('%Y-%m-%d')
                grp      = line_group(lines)

                if station_id not in stations:
                    stations[station_id] = Station(
                        station_complex_id=station_id,
                        name=station_name, lines=lines,
                        borough=borough, lat=lat, lon=lon
                    )
                s = stations[station_id]
                s.total_ridership += ridership
                s.total_transfers += transfers
                s.record_count    += 1

                key = (station_id, date_str)
                daily[key]['ridership'] += ridership
                daily[key]['transfers'] += transfers
                daily[key]['count']     += 1

                for line in lines:
                    line_stats[line]['ridership'] += ridership
                    line_stats[line]['transfers'] += transfers
                    line_stats[line]['stations'].add(station_id)

                hourly_pattern[(grp, hour, dow)] += ridership

                if not skip_hourly:
                    hourly_records.append({
                        'timestamp':    ts.isoformat(),
                        'hour':         hour,
                        'day_of_week':  dow,
                        'is_weekend':   int(dow >= 5),
                        'station_id':   station_id,
                        'station_name': station_name,
                        'lines':        ','.join(lines),
                        'line_group':   grp,
                        'borough':      borough,
                        'payment':      payment,
                        'fare_class':   fare_class,
                        'ridership':    ridership,
                        'transfers':    transfers,
                        'lat':          lat,
                        'lon':          lon,
                    })

                row_count += 1
                if row_count % 100_000 == 0:
                    print(f"  Processed {row_count:,} rows...")

            except (ValueError, KeyError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [warn] Row {row_count}: {e}")

    print(f"\nDone. Parsed: {row_count:,} | Skipped: {skipped:,}")
    print(f"Unique stations: {len(stations):,} | Unique lines: {len(line_stats):,}")

    # 1. stations.json
    station_list = []
    for s in stations.values():
        d = asdict(s)
        d['avg_ridership_per_hour'] = round(s.total_ridership / max(s.record_count, 1), 2)
        station_list.append(d)
    station_list.sort(key=lambda x: x['total_ridership'], reverse=True)
    out = os.path.join(output_dir, 'stations.json')
    with open(out, 'w') as f:
        json.dump(station_list, f, indent=2)
    print(f"\n[1] stations.json         -> {out}")

    # 2. ridership_daily.csv
    out = os.path.join(output_dir, 'ridership_daily.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['station_id','station_name','borough','lines','lat','lon',
                    'date','ridership','transfers','hourly_records'])
        for (sid, date), vals in sorted(daily.items()):
            s = stations.get(sid)
            if s:
                w.writerow([sid, s.name, s.borough, ','.join(s.lines),
                            s.lat, s.lon, date,
                            vals['ridership'], vals['transfers'], vals['count']])
    print(f"[2] ridership_daily.csv   -> {out}")

    # 3. line_summary.csv
    out = os.path.join(output_dir, 'line_summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['line','line_group','total_ridership','total_transfers','station_count'])
        for line, stats in sorted(line_stats.items(), key=lambda x: -x[1]['ridership']):
            w.writerow([line, line_group([line]),
                        stats['ridership'], stats['transfers'], len(stats['stations'])])
    print(f"[3] line_summary.csv      -> {out}")

    # 4. hourly_patterns.json
    patterns = {}
    for (grp, hour, dow), total in hourly_pattern.items():
        patterns.setdefault(grp, {})[f'h{hour:02d}_d{dow}'] = total
    out = os.path.join(output_dir, 'hourly_patterns.json')
    with open(out, 'w') as f:
        json.dump(patterns, f, indent=2)
    print(f"[4] hourly_patterns.json  -> {out}")

    # 5. network_graph.json
    nodes = [
        {'id': s.station_complex_id, 'name': s.name,
         'lat': s.lat, 'lon': s.lon, 'borough': s.borough,
         'lines': s.lines, 'ridership': s.total_ridership}
        for s in stations.values()
    ]
    line_to_sids = defaultdict(list)
    for s in stations.values():
        for line in s.lines:
            line_to_sids[line].append(s.station_complex_id)

    edges = []
    seen  = set()
    for line, sids in line_to_sids.items():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = min(sids[i], sids[j]), max(sids[i], sids[j])
                key  = (a, b, line)
                if key not in seen:
                    seen.add(key)
                    edges.append({'source': a, 'target': b, 'line': line})

    out = os.path.join(output_dir, 'network_graph.json')
    with open(out, 'w') as f:
        json.dump({'nodes': nodes, 'edges': edges}, f, indent=2)
    print(f"[5] network_graph.json    -> {out}")

    # 6. ridership_hourly.csv (optional)
    if not skip_hourly and hourly_records:
        out = os.path.join(output_dir, 'ridership_hourly.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hourly_records[0].keys())
            w.writeheader()
            w.writerows(hourly_records)
        print(f"[6] ridership_hourly.csv  -> {out}  ({len(hourly_records):,} rows)")

    print(f"\nAll outputs in: {output_dir}/")


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='MTA Hourly Ridership Parser')
    p.add_argument('--input',       required=True)
    p.add_argument('--max-rows',    type=int, default=5_000_000)
    p.add_argument('--skip-hourly', action='store_true')
    p.add_argument('--output-dir',  default='./output')
    args = p.parse_args()
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    parse(args.input, args.max_rows, args.skip_hourly, args.output_dir)

if __name__ == '__main__':
    main()