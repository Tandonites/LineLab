from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Literal
import csv
import json

import networkx as nx
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
try:
    from predict import predict_new_line as _predict_new_line
    _PREDICT_AVAILABLE = True
except Exception:
    try:
        from src.predict import predict_new_line as _predict_new_line
        _PREDICT_AVAILABLE = True
    except Exception:
        _PREDICT_AVAILABLE = False
        _predict_new_line = None
from fastapi.middleware.cors import CORSMiddleware
from networkx.readwrite import json_graph
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIONS_PATH = ROOT_DIR / 'data' / 'processed' / 'stations.json'
LINE_SUMMARY_PATH = ROOT_DIR / 'data' / 'processed' / 'line_summary.csv'
TIME_GRAPH_PATH = ROOT_DIR / 'data' / 'processed' / 'mta_time_graph.json'
MODELS_DIR = ROOT_DIR / 'data' / 'models'
STATION_FEATURES_CSV = ROOT_DIR / 'data' / 'raw' / 'station_features.csv'


class StationInput(BaseModel):
    id: str
    name: str
    lat: float
    lon: float
    is_new: bool = False


class SimulationRequest(BaseModel):
    train_service: Literal['local', 'express'] = 'local'
    stations: list[StationInput] = Field(min_length=2)


class AffectedLine(BaseModel):
    line: str
    delta_pct: float


class AffectedStation(BaseModel):
    station_id: str
    name: str
    ridership_delta: int
    ridership_delta_pct: float


class RouteComparison(BaseModel):
    available: bool
    existing_route_label: str
    origin_name: str
    destination_name: str
    first_train: str
    transfer_station: str | None
    second_train: str | None
    existing_travel_minutes: int
    new_route_minutes: int
    time_saved_minutes: int


class SimulationResponse(BaseModel):
    new_line_ridership: int
    peak_hour_ridership: int
    operational_cost_daily: int
    affected_lines: list[AffectedLine]
    affected_stations: list[AffectedStation]
    route_comparison: RouteComparison | None


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


# ── ML model loading ──────────────────────────────────────────────────────────

def load_ml_models() -> tuple[xgb.Booster | None, xgb.Booster | None, dict]:
    meta: dict = {
        'feature_columns': [],
        'log_target_daily': True,
        'log_target_peak': False,
        'fallback_peak_factor': 0.10,
    }
    rid_model: xgb.Booster | None = None
    pf_model: xgb.Booster | None = None

    feat_path = MODELS_DIR / 'feature_columns.json'
    if feat_path.exists():
        try:
            meta = json.loads(feat_path.read_text(encoding='utf-8'))
        except (ValueError, KeyError):
            pass

    rid_path = MODELS_DIR / 'ridership_model.json'
    if rid_path.exists():
        try:
            m = xgb.Booster()
            m.load_model(str(rid_path))
            rid_model = m
        except Exception:
            rid_model = None

    pf_path = MODELS_DIR / 'peak_factor_model.json'
    if pf_path.exists():
        try:
            m = xgb.Booster()
            m.load_model(str(pf_path))
            pf_model = m
        except Exception:
            pf_model = None

    return rid_model, pf_model, meta


def load_station_feature_lookup() -> dict[str, dict]:
    """Load station_features.csv into {station_complex_id: row_dict}."""
    lookup: dict[str, dict] = {}
    if not STATION_FEATURES_CSV.exists():
        return lookup
    with STATION_FEATURES_CSV.open(newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            sid = str(row.get('station_complex_id', '')).strip()
            if sid:
                lookup[sid] = row
    return lookup


RIDERSHIP_MODEL, PEAK_FACTOR_MODEL, MODEL_META = load_ml_models()
STATION_FEATURE_LOOKUP = load_station_feature_lookup()

BOROUGH_COLS = ['boro_Bronx', 'boro_Brooklyn', 'boro_Manhattan', 'boro_Queens', 'boro_Staten Island']
BOROUGH_MAP = {
    'bronx': 'boro_Bronx',
    'brooklyn': 'boro_Brooklyn',
    'manhattan': 'boro_Manhattan',
    'queens': 'boro_Queens',
    'staten island': 'boro_Staten Island',
    'staten': 'boro_Staten Island',
}


def _row_to_vec(row: dict, lat: float, lon: float, num_lines: int) -> dict[str, float]:
    """Convert a station_features row into the flat feature dict the model expects."""

    def flt(key: float, default: float = 0.0) -> float:
        try:
            v = row.get(key, default)
            return float(v) if v not in (None, '', 'nan') else default
        except (ValueError, TypeError):
            return default

    boro_raw = str(row.get('borough', '')).strip().lower()
    features: dict[str, float] = {
        'num_lines': float(num_lines),
        'lat': lat,
        'lon': lon,
        'pop_density_tract': flt('pop_density_tract'),
        'population_500m': flt('population_500m'),
        'median_income': flt('median_income'),
        'commuters_tract': flt('commuters_tract'),
        'bus_stops_250m': flt('bus_stops_250m'),
        'bus_stops_500m': flt('bus_stops_500m'),
    }
    for col in BOROUGH_COLS:
        features[col] = 0.0
    mapped = BOROUGH_MAP.get(boro_raw)
    if mapped:
        features[mapped] = 1.0
    return features


def _interpolate_features(lat: float, lon: float, num_lines: int) -> dict[str, float]:
    """For new stations with no lookup entry, interpolate from 3 nearest known stations."""
    if not STATION_FEATURE_LOOKUP:
        return {c: 0.0 for c in (['num_lines', 'lat', 'lon', 'pop_density_tract',
                                   'population_500m', 'median_income', 'commuters_tract',
                                   'bus_stops_250m', 'bus_stops_500m'] + BOROUGH_COLS)}

    neighbours = sorted(
        STATION_FEATURE_LOOKUP.values(),
        key=lambda r: haversine_km(lat, lon, float(r.get('lat') or 0), float(r.get('lon') or 0)),
    )[:3]

    merged: dict[str, float] = {}
    for key in ('pop_density_tract', 'population_500m', 'median_income',
                'commuters_tract', 'bus_stops_250m', 'bus_stops_500m'):
        vals = []
        for r in neighbours:
            try:
                v = float(r.get(key) or 0)
                vals.append(v)
            except (ValueError, TypeError):
                pass
        merged[key] = float(np.mean(vals)) if vals else 0.0

    # Pick majority borough from neighbours
    boro_counts: dict[str, int] = {}
    for r in neighbours:
        b = str(r.get('borough', '')).strip()
        boro_counts[b] = boro_counts.get(b, 0) + 1
    dominant_boro = max(boro_counts, key=boro_counts.get) if boro_counts else ''

    merged['borough'] = dominant_boro
    vec = _row_to_vec(merged, lat, lon, num_lines)
    return vec


def build_feature_vector(station: 'StationInput', num_lines: int) -> list[float]:
    """Build the ordered feature vector for one station."""
    feature_cols: list[str] = MODEL_META.get('feature_columns', [])
    if not feature_cols:
        return []

    row = STATION_FEATURE_LOOKUP.get(station.id)
    if row is not None:
        feat = _row_to_vec(row, station.lat, station.lon, num_lines)
    else:
        feat = _interpolate_features(station.lat, station.lon, num_lines)

    return [feat.get(col, 0.0) for col in feature_cols]


def predict_ridership_ml(stations: list['StationInput']) -> tuple[int, int]:
    """
    Use the trained XGBoost model to predict total new-line daily ridership and
    peak-hour ridership.  Falls back to heuristic if model is unavailable.
    """
    feature_cols: list[str] = MODEL_META.get('feature_columns', [])

    if RIDERSHIP_MODEL is None or not feature_cols:
        return None, None  # caller will use heuristic

    rows = []
    for st in stations:
        num_lines = 1  # new line adds at least 1 line to this stop
        vec = build_feature_vector(st, num_lines)
        if len(vec) != len(feature_cols):
            return None, None
        rows.append(vec)

    X = np.array(rows, dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    preds = RIDERSHIP_MODEL.predict(dmat)

    if MODEL_META.get('log_target_daily', True):
        preds = np.expm1(preds)

    # New-line total ridership = sum of per-station predictions scaled by a
    # transfer-uplift factor (new lines capture a share of existing demand +
    # induced demand ~15 %).
    total_daily = int(round(float(np.sum(preds)) * 1.15))

    # Peak-hour fraction
    if PEAK_FACTOR_MODEL is not None:
        pf_preds = PEAK_FACTOR_MODEL.predict(dmat)
        peak_factor = float(np.mean(pf_preds))
    else:
        peak_factor = float(MODEL_META.get('fallback_peak_factor', 0.10))

    peak_hourly = int(round(total_daily * peak_factor))
    return total_daily, peak_hourly


def load_time_graph() -> nx.Graph | None:
    if not TIME_GRAPH_PATH.exists():
        return None
    try:
        raw = json.loads(TIME_GRAPH_PATH.read_text(encoding='utf-8'))
        return json_graph.node_link_graph(raw)
    except (ValueError, TypeError):
        return None


TIME_GRAPH = load_time_graph()


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


def estimate_new_route_minutes(route: list[StationInput], train_service: str) -> int:
    line_km = max(0.5, route_length_km(route))
    avg_speed_kmh = 28 if train_service == 'local' else 40
    dwell_minutes = 0.7 * max(0, len(route) - 1)
    in_motion_minutes = (line_km / avg_speed_kmh) * 60
    return max(3, int(round(in_motion_minutes + dwell_minutes)))


def nearest_graph_node(graph: nx.Graph, lat: float, lon: float) -> str | None:
    best_node: str | None = None
    best_dist = float('inf')
    for node_id, attrs in graph.nodes(data=True):
        node_lat = attrs.get('lat')
        node_lon = attrs.get('lon')
        if node_lat is None or node_lon is None:
            continue
        try:
            dist = haversine_km(lat, lon, float(node_lat), float(node_lon))
        except (TypeError, ValueError):
            continue
        if dist < best_dist:
            best_dist = dist
            best_node = str(node_id)
    return best_node


def estimate_existing_travel_minutes(route: list[StationInput]) -> int | None:
    if TIME_GRAPH is None:
        return None

    origin = route[0]
    destination = route[-1]
    start_node = nearest_graph_node(TIME_GRAPH, origin.lat, origin.lon)
    end_node = nearest_graph_node(TIME_GRAPH, destination.lat, destination.lon)
    if not start_node or not end_node:
        return None

    try:
        seconds = nx.dijkstra_path_length(TIME_GRAPH, start_node, end_node, weight='weight')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    minutes = int(round(float(seconds) / 60.0))
    return max(1, minutes)


def build_route_comparison(payload: SimulationRequest, route: list[StationInput]) -> RouteComparison:
    origin_name = route[0].name
    destination_name = route[-1].name
    new_route_minutes = estimate_new_route_minutes(route, payload.train_service)
    existing_minutes = estimate_existing_travel_minutes(route)

    if existing_minutes is None:
        existing_minutes = int(round(new_route_minutes * 1.35 + 4))
        available = False
        first_train = 'Current service'
        existing_label = 'Estimated current network trip'
    else:
        available = True
        first_train = 'Current service'
        existing_label = 'Current network fastest route'

    return RouteComparison(
        available=available,
        existing_route_label=existing_label,
        origin_name=origin_name,
        destination_name=destination_name,
        first_train=first_train,
        transfer_station=None,
        second_train=None,
        existing_travel_minutes=existing_minutes,
        new_route_minutes=new_route_minutes,
        time_saved_minutes=max(0, existing_minutes - new_route_minutes),
    )


def simulate(payload: SimulationRequest) -> SimulationResponse:
    drawn = payload.stations
    if len(drawn) < 2:
        raise HTTPException(status_code=400, detail='Need at least two stations.')

    line_km = max(1.0, route_length_km(drawn))
    new_stop_count = sum(1 for station in drawn if station.is_new)
    operational_cost_daily = int(line_km * 165000 + len(drawn) * 20000)
    route_comparison = build_route_comparison(payload, drawn)

    # Use the full predict module (geospatial feature extraction + time-graph redistribution)
    if _PREDICT_AVAILABLE:
        station_dicts = [
            {'id': st.id, 'name': st.name, 'lat': st.lat, 'lon': st.lon, 'is_new': st.is_new}
            for st in drawn
        ]
        pred = _predict_new_line(station_dicts)
        return SimulationResponse(
            new_line_ridership=pred['new_line_ridership'],
            peak_hour_ridership=pred['peak_hour_ridership'],
            operational_cost_daily=operational_cost_daily,
            affected_lines=[
                AffectedLine(line=a['line'], delta_pct=a['delta_pct'])
                for a in pred['affected_lines']
            ],
            affected_stations=[
                AffectedStation(
                    station_id=a['station_id'],
                    name=a['name'],
                    ridership_delta=a['ridership_delta'],
                    ridership_delta_pct=a['ridership_delta_pct'],
                )
                for a in pred['affected_stations']
            ],
            route_comparison=route_comparison,
        )

    # Fallback: inline ML model (no geopandas, no time-graph redistribution)
    ml_ridership, ml_peak = predict_ridership_ml(drawn)
    if ml_ridership is not None:
        new_line_ridership = ml_ridership
        peak_hour_ridership = ml_peak
    else:
        new_line_ridership = int(12000 + line_km * 7600 + new_stop_count * 1800)
        peak_hour_ridership = int(new_line_ridership * 0.10)

    if not STATION_FEATURES:
        return SimulationResponse(
            new_line_ridership=new_line_ridership,
            peak_hour_ridership=peak_hour_ridership,
            operational_cost_daily=operational_cost_daily,
            affected_lines=[],
            affected_stations=[],
            route_comparison=route_comparison,
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
        delta_pct = (delta / station.total_ridership * 100) if station.total_ridership > 0 else 0.0
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
        route_comparison=route_comparison,
    )


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/api/simulate', response_model=SimulationResponse)
def simulate_route(payload: SimulationRequest) -> SimulationResponse:
    return simulate(payload)
