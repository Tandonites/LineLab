from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Literal
import csv
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIONS_PATH = ROOT_DIR / 'data' / 'processed' / 'stations.json'
LINE_SUMMARY_PATH = ROOT_DIR / 'data' / 'processed' / 'line_summary.csv'


class StationInput(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    is_new: bool = False


class SimulationRequest(BaseModel):
    stations: list[StationInput] = Field(min_length=2)


class AffectedLine(BaseModel):
    line: str
    delta_pct: float


class AffectedStation(BaseModel):
    station_id: str
    name: str
    ridership_delta: int
    ridership_delta_pct: float


class SimulationResponse(BaseModel):
    new_line_ridership: int
    peak_hour_ridership: int
    operational_cost_daily: int
    affected_lines: list[AffectedLine]
    affected_stations: list[AffectedStation]


class StationFeature(BaseModel):
    station_complex_id: str
    name: str
    lines: list[str]
    lat: float
    lon: float
    total_ridership: int = 0


app = FastAPI(title='Highball Backend', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    d_lat = radians(b_lat - a_lat)
    d_lon = radians(b_lon - a_lon)
    lat1 = radians(a_lat)
    lat2 = radians(b_lat)
    term = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    return 6371 * 2 * asin(sqrt(term))


def route_length_km(route: list[StationInput]) -> float:
    total = 0.0
    for i in range(1, len(route)):
        a = route[i - 1]
        b = route[i]
        total += haversine_km(a.lat, a.lon, b.lat, b.lon)
    return total


def load_station_features() -> list[StationFeature]:
    if not STATIONS_PATH.exists():
        return []

    raw = json.loads(STATIONS_PATH.read_text(encoding='utf-8'))
    stations: list[StationFeature] = []
    for row in raw:
        stations.append(
            StationFeature(
                station_complex_id=str(row.get('station_complex_id')),
                name=row.get('name', 'Unknown'),
                lines=[str(line) for line in row.get('lines', [])],
                lat=float(row.get('lat', 0.0)),
                lon=float(row.get('lon', 0.0)),
                total_ridership=int(row.get('total_ridership', 0) or 0),
            )
        )
    return stations


def load_line_totals() -> dict[str, int]:
    if not LINE_SUMMARY_PATH.exists():
        return {}

    totals: dict[str, int] = {}
    with LINE_SUMMARY_PATH.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            line = row.get('line')
            total = row.get('total_ridership')
            if not line or not total:
                continue
            try:
                totals[line] = int(total)
            except ValueError:
                continue
    return totals


STATION_FEATURES = load_station_features()
LINE_TOTALS = load_line_totals()


def proximity_score(station: StationFeature, drawn: list[StationInput]) -> float:
    min_distance = min(
        haversine_km(station.lat, station.lon, point.lat, point.lon) for point in drawn
    )
    return max(0.0, 1 - min_distance / 3.8)


def signed_delta(station_id: str, positive_ids: set[str]) -> Literal[-1, 1]:
    if station_id in positive_ids:
        return 1
    bucket = sum(ord(ch) for ch in station_id) % 7
    return 1 if bucket in {0, 1} else -1


def simulate(payload: SimulationRequest) -> SimulationResponse:
    drawn = payload.stations
    if len(drawn) < 2:
        raise HTTPException(status_code=400, detail='Need at least two stations.')

    line_km = max(1.0, route_length_km(drawn))
    new_stop_count = sum(1 for station in drawn if station.is_new)

    new_line_ridership = int(12000 + line_km * 7600 + new_stop_count * 1800)
    peak_hour_ridership = int(new_line_ridership * 0.19)
    operational_cost_daily = int(line_km * 165000 + len(drawn) * 20000)

    if not STATION_FEATURES:
        return SimulationResponse(
            new_line_ridership=new_line_ridership,
            peak_hour_ridership=peak_hour_ridership,
            operational_cost_daily=operational_cost_daily,
            affected_lines=[],
            affected_stations=[],
        )

    positive_station_ids = {station.id for station in drawn if not station.is_new}

    candidates: list[tuple[StationFeature, int, float]] = []
    for station in STATION_FEATURES:
        score = proximity_score(station, drawn)
        if score <= 0:
            continue

        base = max(180, int(station.total_ridership * 0.04 * score))
        sign = signed_delta(station.station_complex_id, positive_station_ids)
        delta = base * sign

        if station.total_ridership > 0:
            delta_pct = (delta / station.total_ridership) * 100
        else:
            delta_pct = 0.0

        candidates.append((station, delta, delta_pct))

    candidates.sort(key=lambda item: abs(item[1]), reverse=True)
    top = candidates[:30]

    line_delta_sum: dict[str, int] = {}
    for station, delta, _ in top:
        for line in station.lines:
            line_delta_sum[line] = line_delta_sum.get(line, 0) + delta

    affected_lines: list[AffectedLine] = []
    for line, delta in line_delta_sum.items():
        baseline = LINE_TOTALS.get(line, max(1, new_line_ridership))
        delta_pct = (delta / baseline) * 100
        affected_lines.append(AffectedLine(line=line, delta_pct=round(delta_pct, 2)))

    affected_lines.sort(key=lambda item: abs(item.delta_pct), reverse=True)

    affected_stations = [
        AffectedStation(
            station_id=station.station_complex_id,
            name=station.name,
            ridership_delta=delta,
            ridership_delta_pct=round(delta_pct, 2),
        )
        for station, delta, delta_pct in top
    ]

    return SimulationResponse(
        new_line_ridership=new_line_ridership,
        peak_hour_ridership=peak_hour_ridership,
        operational_cost_daily=operational_cost_daily,
        affected_lines=affected_lines[:8],
        affected_stations=affected_stations,
    )


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/api/simulate', response_model=SimulationResponse)
def simulate_route(payload: SimulationRequest) -> SimulationResponse:
    return simulate(payload)
