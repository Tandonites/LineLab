from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Literal
import csv
import json
import logging

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
RIDERSHIP_MONTHLY_PATH = ROOT_DIR / 'data' / 'processed' / 'ridership_monthly.csv'
MONTHLY_COST_LABELS_PATH = ROOT_DIR / 'data' / 'raw' / 'monthly_operating_cost.csv'
MODELS_DIR = ROOT_DIR / 'data' / 'models'
STATION_FEATURES_CSV = ROOT_DIR / 'data' / 'raw' / 'station_features.csv'
GTFS_TRIPS_PATH = ROOT_DIR / 'data' / 'gtfs_subway' / 'trips.txt'
TIME_MODEL_PATH = ROOT_DIR / 'data' / 'models' / 'time_model.joblib'


logger = logging.getLogger(__name__)


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
    is_walking_only: bool
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
    operational_cost_monthly: int
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

DEFAULT_VALID_LINES = {
    '1', '2', '3', '4', '5', '6', '6X', '7', '7X',
    'A', 'B', 'C', 'D', 'E', 'F', 'FS', 'FX', 'G', 'GS', 'H',
    'J', 'L', 'M', 'N', 'Q', 'R', 'SI', 'SIR', 'S', 'W', 'Z',
}


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


def load_system_service_stats() -> dict[str, float]:
    """Load system-level baseline stats used by the cost formula."""
    stats = {
        'line_count': 20.0,
        'station_count_total': 400.0,
        'baseline_ridership_total': 100_000_000.0,
        'baseline_transfers_total': 10_000_000.0,
        'baseline_transfer_ratio': 0.10,
    }
    if not LINE_SUMMARY_PATH.exists():
        return stats

    try:
        import pandas as pd  # local import to avoid changing top-level dependency behavior

        df = pd.read_csv(LINE_SUMMARY_PATH)
        if not df.empty:
            rid = pd.to_numeric(df.get('total_ridership', 0), errors='coerce').fillna(0.0)
            trn = pd.to_numeric(df.get('total_transfers', 0), errors='coerce').fillna(0.0)
            stn = pd.to_numeric(df.get('station_count', 0), errors='coerce').fillna(0.0)
            baseline_ridership_total = float(rid.sum())
            baseline_transfers_total = float(trn.sum())
            station_count_total = float(stn.sum())
            line_count = float(len(df))
            stats.update({
                'line_count': line_count,
                'station_count_total': station_count_total,
                'baseline_ridership_total': baseline_ridership_total,
                'baseline_transfers_total': baseline_transfers_total,
                'baseline_transfer_ratio': baseline_transfers_total / max(1.0, baseline_ridership_total),
            })
    except Exception:
        pass
    return stats


def load_cost_formula_params() -> dict[str, float]:
    """
    Calibrate a simple monthly cost formula from historical labels.

    Formula basis:
      monthly_cost ~= b0 + b1 * monthly_ridership + b2 * transfers_proxy
    """
    # Safe defaults if historical files are unavailable.
    params: dict[str, float] = {
        'intercept': 12_000_000.0,
        'coef_ridership': 7.5,
        'coef_transfers_proxy': 2.0,
        'n_obs': 0.0,
    }

    if not RIDERSHIP_MONTHLY_PATH.exists() or not MONTHLY_COST_LABELS_PATH.exists():
        return params

    try:
        import pandas as pd  # local import to avoid changing top-level dependency behavior

        rid = pd.read_csv(RIDERSHIP_MONTHLY_PATH)
        lab = pd.read_csv(MONTHLY_COST_LABELS_PATH)
        if 'month' not in rid.columns or 'month' not in lab.columns:
            return params

        rid_col = 'monthly_ridership' if 'monthly_ridership' in rid.columns else 'ridership'
        if rid_col not in rid.columns or 'monthly_operating_cost' not in lab.columns:
            return params

        rid = rid[['month', rid_col]].copy()
        lab = lab[['month', 'monthly_operating_cost']].copy()
        rid['month'] = pd.to_datetime(rid['month'], errors='coerce').dt.to_period('M').astype(str)
        lab['month'] = pd.to_datetime(lab['month'], errors='coerce').dt.to_period('M').astype(str)
        rid[rid_col] = pd.to_numeric(rid[rid_col], errors='coerce').fillna(0.0)
        lab['monthly_operating_cost'] = pd.to_numeric(lab['monthly_operating_cost'], errors='coerce')

        merged = rid.merge(lab, on='month', how='inner').dropna(subset=['monthly_operating_cost'])
        if merged.empty:
            return params

        transfer_ratio = float(SYSTEM_SERVICE_STATS.get('baseline_transfer_ratio', 0.10))
        line_count = float(SYSTEM_SERVICE_STATS.get('line_count', 20.0))
        x1 = merged[rid_col].astype(float)
        y = merged['monthly_operating_cost'].astype(float)

        valid = x1 > 0
        if valid.any():
            unit_cost = (y[valid] / x1[valid]).replace([np.inf, -np.inf], np.nan).dropna()
        else:
            unit_cost = np.array([])

        if len(unit_cost) > 0:
            # System-wide average includes central overhead; use a reduced marginal factor for a single new corridor.
            marginal_unit_cost = float(np.median(unit_cost)) * 0.62
            params['coef_ridership'] = float(np.clip(marginal_unit_cost, 3.0, 20.0))
            params['coef_transfers_proxy'] = float(params['coef_ridership'] * transfer_ratio * 0.45)

        per_line_monthly = float(y.mean()) / max(1.0, line_count)
        params['intercept'] = float(np.clip(per_line_monthly * 0.28, 5_000_000.0, 35_000_000.0))
        params['n_obs'] = float(len(merged))
    except Exception:
        pass

    return params

def load_valid_route_ids() -> set[str]:
    """Load canonical route IDs from GTFS trips; fallback to defaults."""
    valid = set(DEFAULT_VALID_LINES)
    if not GTFS_TRIPS_PATH.exists():
        return valid

    with GTFS_TRIPS_PATH.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rid = str(row.get('route_id', '')).strip()
            if rid:
                valid.add(rid)
    return valid


STATION_FEATURES = load_station_features()
LINE_TOTALS = load_line_totals()
VALID_ROUTE_IDS = load_valid_route_ids()
SYSTEM_SERVICE_STATS = load_system_service_stats()
COST_FORMULA_PARAMS = load_cost_formula_params()


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


def load_cost_ml_artifacts() -> tuple[xgb.Booster | None, list[str]]:
    model_path = MODELS_DIR / 'cost_model.json'
    feat_path = MODELS_DIR / 'cost_feature_columns.json'
    if not model_path.exists() or not feat_path.exists():
        return None, []

    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
    except Exception:
        return None, []

    try:
        payload = json.loads(feat_path.read_text(encoding='utf-8'))
        cols = payload.get('feature_columns', [])
        cols = [str(c) for c in cols if c]
    except Exception:
        cols = []

    return model, cols


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


def load_time_route_predictor():
    """Load the ML time model + parser. If model is missing, train and persist one."""
    try:
        from data.time_predict import MODEL_PATH as TP_MODEL_PATH
        from data.time_predict import load_model as tp_load_model
        from data.time_predict import parse_and_predict_route as tp_parse_and_predict_route
        from data.time_predict import save_model as tp_save_model
        from data.time_predict import train_model as tp_train_model
    except Exception as exc:
        logger.warning('Time predictor import unavailable: %s', exc)
        return None, None

    try:
        model_path = Path(TP_MODEL_PATH)
        if model_path.exists():
            model = tp_load_model()
        else:
            logger.info('time_model.joblib not found; training new time model...')
            model = tp_train_model()
            tp_save_model(model)
        return model, tp_parse_and_predict_route
    except Exception as exc:
        logger.warning('Failed to initialize time predictor model: %s', exc)
        return None, None


RIDERSHIP_MODEL, PEAK_FACTOR_MODEL, MODEL_META = load_ml_models()
COST_ML_MODEL, COST_ML_FEATURE_COLS = load_cost_ml_artifacts()
STATION_FEATURE_LOOKUP = load_station_feature_lookup()
TIME_ROUTE_MODEL, TIME_ROUTE_PARSER = load_time_route_predictor()

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

WALK_SPEED_MS = 1.4  # metres per second


def _calculate_existing_trip(route: list[StationInput]) -> tuple[float, bool]:
    """Replicate calculate_current_time logic using the already-loaded TIME_GRAPH.

    Returns (best_time_seconds, is_walking_only).
    Walking-only is True when there are no existing stations in the route, or
    when straight-line walking is faster than the transit option.
    """
    start = route[0]
    end = route[-1]

    dist_km = haversine_km(start.lat, start.lon, end.lat, end.lon)
    pure_walk_seconds = (dist_km * 1000.0) / WALK_SPEED_MS

    if TIME_GRAPH is None:
        return pure_walk_seconds, True

    # Two-pointer: find first and last non-new stations
    lo, hi = 0, len(route) - 1
    while lo < len(route) and route[lo].is_new:
        lo += 1
    while hi >= 0 and route[hi].is_new:
        hi -= 1

    if lo > hi:
        return pure_walk_seconds, True

    existing_start = route[lo]
    existing_end = route[hi]

    # Walking time from new endpoints to nearest existing station in route
    walk_seconds = 0.0
    if start.is_new:
        walk_seconds += haversine_km(start.lat, start.lon, existing_start.lat, existing_start.lon) * 1000.0 / WALK_SPEED_MS
    if end.is_new:
        walk_seconds += haversine_km(end.lat, end.lon, existing_end.lat, existing_end.lon) * 1000.0 / WALK_SPEED_MS

    # Name-based node lookup (mirrors find_path.py approach)
    src_nodes = [nid for nid, d in TIME_GRAPH.nodes(data=True)
                 if existing_start.name in str(d.get('name', ''))]
    dst_nodes = [nid for nid, d in TIME_GRAPH.nodes(data=True)
                 if existing_end.name in str(d.get('name', ''))]

    best_transit = float('inf')
    for src in src_nodes:
        for dst in dst_nodes:
            try:
                t = nx.dijkstra_path_length(TIME_GRAPH, src, dst, weight='weight')
                if t < best_transit:
                    best_transit = t
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if best_transit == float('inf'):
        return pure_walk_seconds, True

    transit_total = best_transit + walk_seconds
    is_walking_only = pure_walk_seconds < transit_total
    return min(pure_walk_seconds, transit_total), is_walking_only


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
    if TIME_ROUTE_MODEL is not None and TIME_ROUTE_PARSER is not None:
        try:
            stations_payload = [
                {
                    'station_complex_id': st.id,
                    'name': st.name,
                    'lat': st.lat,
                    'lon': st.lon,
                }
                for st in route
            ]
            pred = TIME_ROUTE_PARSER(stations_payload, train_service, TIME_ROUTE_MODEL, verbose=False)
            total_minutes = float(pred.get('total_minutes', 0.0))
            if np.isfinite(total_minutes) and total_minutes > 0:
                return max(1, int(round(total_minutes)))
        except Exception:
            # Fall through to deterministic geometry-based estimate.
            pass

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

    existing_seconds, is_walking_only = _calculate_existing_trip(route)
    existing_minutes_raw = existing_seconds / 60.0

    # The time graph encodes pure in-motion seconds (no wait, no dwell, no
    # platform walking).  A 1.6× factor brings it in line with real-world
    # door-to-door trip times (headway + dwell + access).
    # Walking-only times are not scaled because they already represent real
    # elapsed wall-clock time at walking speed.
    TRANSIT_REALISM_FACTOR = 1.6

    # If the graph returned a real path, mark as available
    all_new = all(st.is_new for st in route)
    available = not all_new and not is_walking_only

    if is_walking_only:
        existing_minutes = int(round(existing_minutes_raw))
        existing_label = 'Walking only (no faster transit route found)'
        first_train = 'Walk'
    else:
        existing_minutes = int(round(existing_minutes_raw * TRANSIT_REALISM_FACTOR))
        existing_label = 'Current network fastest route'
        first_train = 'Current service'

    # Fallback: if seconds came back as pure walk and it seems unreasonably low, use heuristic
    if existing_minutes < 1:
        existing_minutes = int(round(new_route_minutes * 1.35 + 4))
        available = False
        is_walking_only = False
        existing_label = 'Estimated current network trip'
        first_train = 'Current service'

    return RouteComparison(
        available=available,
        is_walking_only=is_walking_only,
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


def _predict_monthly_cost_ml_check(
    *,
    line_km: float,
    stop_count: int,
    monthly_ridership: float,
    monthly_transfers_proxy: float,
    target_tph: float,
    train_service: Literal['local', 'express'],
    monthly_formula: float,
) -> int | None:
    """Optional ML sanity check for cost (formula remains authoritative)."""
    if COST_ML_MODEL is None or not COST_ML_FEATURE_COLS:
        return None

    feature_map: dict[str, float] = {
        'line_length_km': float(line_km),
        'station_count': float(stop_count),
        'monthly_ridership': float(monthly_ridership),
        'transfer_proxy': float(monthly_transfers_proxy),
        'is_express': 1.0 if train_service == 'express' else 0.0,
        'service_frequency_proxy': float(target_tph),
        'route_complexity': float(line_km * stop_count),
        # No real future lag values exist for a hypothetical line; anchor to formula baseline.
        'target_lag_1': float(monthly_formula),
        'target_lag_3_avg': float(monthly_formula),
    }

    row = [float(feature_map.get(col, 0.0)) for col in COST_ML_FEATURE_COLS]
    try:
        dmat = xgb.DMatrix(np.array([row], dtype=float), feature_names=COST_ML_FEATURE_COLS)
        pred = float(COST_ML_MODEL.predict(dmat)[0])
    except Exception:
        return None

    if not np.isfinite(pred):
        return None
    return int(round(max(0.0, pred)))


def predict_operating_costs(
    route: list[StationInput],
    train_service: Literal['local', 'express'],
    new_line_ridership: int,
    peak_hour_ridership: int,
) -> tuple[int, int]:
    """
    Estimate daily and monthly operating costs.

    Primary path: calibrated formula from historical monthly cost labels.
    Secondary guardrails: geometry/service adjustments and bounded fallback.
    """
    line_km = max(1.0, route_length_km(route))
    stop_count = max(2, len(route))

    # Convert predicted demand to monthly for formula features.
    monthly_ridership = float(max(0, new_line_ridership)) * 30.4
    transfer_ratio = float(SYSTEM_SERVICE_STATS.get('baseline_transfer_ratio', 0.10))
    monthly_transfers_proxy = monthly_ridership * transfer_ratio

    # Calibrated baseline (data-driven, interpretable)
    b0 = float(COST_FORMULA_PARAMS.get('intercept', 250_000_000.0))
    b1 = float(COST_FORMULA_PARAMS.get('coef_ridership', 3.5))
    b2 = float(COST_FORMULA_PARAMS.get('coef_transfers_proxy', 1.2))
    monthly_formula = b0 + b1 * monthly_ridership + b2 * monthly_transfers_proxy

    # Service intensity from demand: higher peak demand implies higher frequency.
    cap_per_train = 1100 if train_service == 'local' else 1300
    target_tph = max(4.0, min(30.0, peak_hour_ridership / max(1, cap_per_train)))
    service_multiplier = max(0.85, min(1.45, 1.0 + 0.018 * (target_tph - 8.0)))

    # Geometry complexity factor: longer line and more stops imply higher cost.
    geometry_baseline_km = 12.0
    geometry_factor = max(0.70, min(1.35, 0.88 + 0.12 * (line_km / geometry_baseline_km)))
    stop_factor = max(0.80, min(1.25, 0.90 + 0.02 * stop_count))

    # Express corridors generally cost more per km due to service profile.
    pattern_factor = 1.08 if train_service == 'express' else 1.0

    monthly_adjusted = monthly_formula * service_multiplier * geometry_factor * stop_factor * pattern_factor

    # Conservative fallback to avoid extreme outputs when calibration is unstable.
    monthly_fallback = (line_km * 165000 + stop_count * 20000) * 30.4

    # Blend toward calibrated formula while preserving robust behavior.
    monthly = 0.80 * monthly_adjusted + 0.20 * monthly_fallback

    # Secondary check: lightly nudge toward ML only when both estimates agree closely.
    ml_monthly = _predict_monthly_cost_ml_check(
        line_km=line_km,
        stop_count=stop_count,
        monthly_ridership=monthly_ridership,
        monthly_transfers_proxy=monthly_transfers_proxy,
        target_tph=target_tph,
        train_service=train_service,
        monthly_formula=monthly_formula,
    )
    if ml_monthly is not None:
        ratio = ml_monthly / max(1.0, monthly)
        if 0.70 <= ratio <= 1.30:
            monthly = 0.90 * monthly + 0.10 * ml_monthly

    monthly = int(round(max(20_000_000, monthly)))
    daily = int(round(monthly / 30.4))
    return daily, monthly


def simulate(payload: SimulationRequest) -> SimulationResponse:
    drawn = payload.stations
    if len(drawn) < 2:
        raise HTTPException(status_code=400, detail='Need at least two stations.')

    new_stop_count = sum(1 for station in drawn if station.is_new)
    route_comparison = build_route_comparison(payload, drawn)

    # Use the full predict module (geospatial feature extraction + time-graph redistribution)
    if _PREDICT_AVAILABLE:
        station_dicts = [
            {'id': st.id, 'name': st.name, 'lat': st.lat, 'lon': st.lon, 'is_new': st.is_new}
            for st in drawn
        ]
        pred = _predict_new_line(station_dicts)
        operational_cost_daily, operational_cost_monthly = predict_operating_costs(
            drawn,
            payload.train_service,
            int(pred['new_line_ridership']),
            int(pred['peak_hour_ridership']),
        )
        return SimulationResponse(
            new_line_ridership=pred['new_line_ridership'],
            peak_hour_ridership=pred['peak_hour_ridership'],
            operational_cost_daily=operational_cost_daily,
            operational_cost_monthly=operational_cost_monthly,
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
        line_km = max(1.0, route_length_km(drawn))
        new_line_ridership = int(12000 + line_km * 7600 + new_stop_count * 1800)
        peak_hour_ridership = int(new_line_ridership * 0.10)

    operational_cost_daily, operational_cost_monthly = predict_operating_costs(
        drawn,
        payload.train_service,
        new_line_ridership,
        peak_hour_ridership,
    )

    if not STATION_FEATURES:
        return SimulationResponse(
            new_line_ridership=new_line_ridership,
            peak_hour_ridership=peak_hour_ridership,
            operational_cost_daily=operational_cost_daily,
            operational_cost_monthly=operational_cost_monthly,
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
            if line not in VALID_ROUTE_IDS:
                continue
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
        operational_cost_monthly=operational_cost_monthly,
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
