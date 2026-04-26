"""
predict.py
============
Inference module imported by the FastAPI backend.

Top-level function:
  predict_new_line(stations: list[dict]) -> dict

Returns:
  {
    "new_line_ridership":  int,
    "peak_hour_ridership": int,
    "affected_lines":      [{"line": str, "delta_pct": float}, ...],
    "affected_stations":   [{"station_id": str, "name": str,
                             "ridership_delta": int, "ridership_delta_pct": float}, ...]
  }

Artifact paths (resolved relative to this file's location):
  ../data/models/ridership_model.json
  ../data/models/peak_factor_model.json   (optional)
  ../data/models/feature_columns.json
  ../data/raw/census_tracts.geojson
  ../data/raw/bus_stops.geojson
  ../data/processed/stations.json
  ../data/processed/mta_time_graph.json
  ../data/gtfs_subway/trips.txt
  ../data/gtfs_subway/stop_times.txt
  ../data/gtfs_subway/stops.txt
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from networkx.readwrite import json_graph
from shapely.geometry import Point

# ── Path constants ────────────────────────────────────────────────────────────
_SRC_DIR   = Path(__file__).resolve().parent
DATA_DIR   = _SRC_DIR.parent / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
GTFS_DIR   = DATA_DIR / "gtfs_subway"

# ── Tunable assumptions ───────────────────────────────────────────────────────
ATTR_FRAC            = 0.65   # fraction of station ridership credited to the new line
TRAVEL_SPEED_KMH     = 30     # avg subway speed including dwell
DWELL_PER_STOP_SEC   = 36     # seconds added per intermediate stop for new-line edges
TRANSFER_PENALTY_SEC = 300    # seconds penalty per line transfer (5 min)
GRAVITY_SAMPLE       = 400    # number of O-D pairs for redistribution estimation

FIPS_TO_BOROUGH = {
    "061": "Manhattan", "047": "Brooklyn",
    "081": "Queens",    "005": "Bronx", "085": "Staten Island",
}
BOROUGH_COLS = [
    "boro_Bronx", "boro_Brooklyn", "boro_Manhattan",
    "boro_Queens", "boro_Staten Island",
]
_BOROUGH_KEY = {
    "bronx": "boro_Bronx", "brooklyn": "boro_Brooklyn",
    "manhattan": "boro_Manhattan", "queens": "boro_Queens",
    "staten island": "boro_Staten Island",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Module-level state loaded once at import
# ─────────────────────────────────────────────────────────────────────────────

def _load_models() -> tuple[xgb.Booster, Optional[xgb.Booster], dict]:
    meta = {
        "feature_columns": [],
        "log_target_daily": True,
        "log_target_peak": False,
        "fallback_peak_factor": 0.10,
    }
    feat_path = MODELS_DIR / "feature_columns.json"
    if feat_path.exists():
        meta = json.loads(feat_path.read_text(encoding="utf-8"))

    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "ridership_model.json"))

    pf_booster: Optional[xgb.Booster] = None
    pf_path = MODELS_DIR / "peak_factor_model.json"
    if pf_path.exists():
        pf_booster = xgb.Booster()
        pf_booster.load_model(str(pf_path))

    return booster, pf_booster, meta


def _load_time_graph() -> nx.Graph:
    """Load mta_time_graph.json and patch the missing parent↔directional edges.

    GTFS transfers.txt uses parent stop IDs (e.g. "127") while stop_times uses
    directional IDs (e.g. "127N", "127S").  The raw graph therefore has 459
    disconnected components.  We add zero-weight boarding edges between each
    directional node and its parent so that transfers work correctly.
    """
    with open(PROC_DIR / "mta_time_graph.json", encoding="utf-8") as fh:
        data = json.load(fh)
    G = json_graph.node_link_graph(data)

    # Add parent↔directional boarding edges (0 s — already at the platform)
    for nid in list(G.nodes()):
        sid = str(nid)
        if sid and sid[-1] in ("N", "S"):
            parent = sid[:-1]
            if parent in G.nodes and not G.has_edge(sid, parent):
                G.add_edge(sid, parent, weight=0)

    return G


def _load_stop_line_map() -> dict[str, list[str]]:
    """
    Build {stop_id: [route_id, ...]} from GTFS trips + stop_times.
    Uses parent stop IDs (no N/S suffix) as well as directional IDs.
    """
    trips     = pd.read_csv(GTFS_DIR / "trips.txt",      usecols=["trip_id", "route_id"])
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=["trip_id", "stop_id"])
    merged = stop_times.merge(trips, on="trip_id")

    result: dict[str, set] = defaultdict(set)
    for sid, route in zip(merged["stop_id"].astype(str), merged["route_id"].astype(str)):
        result[sid].add(route)
        # Also register the parent stop (strip trailing N / S)
        parent = sid.rstrip("NS")
        if parent != sid:
            result[parent].add(route)

    return {k: sorted(v) for k, v in result.items()}


def _load_stations() -> pd.DataFrame:
    with open(PROC_DIR / "stations.json", encoding="utf-8") as fh:
        return pd.DataFrame(json.load(fh))


def _build_complex_to_stop(
    df_stations: pd.DataFrame,
    G: nx.Graph,
    main_nodes: set,
) -> dict[str, str]:
    """
    Map each station_complex_id to the nearest GTFS node that has coordinates.
    After patching parent↔directional edges the graph is nearly fully connected,
    so any node with lat/lon is a valid candidate.  We prefer directional nodes
    (ends in N or S) over parent stubs when distances tie.
    """
    candidate_nodes = [
        (nid, data)
        for nid, data in G.nodes(data=True)
        if nid in main_nodes and data.get("lat") is not None
    ]
    if not candidate_nodes:
        candidate_nodes = [(nid, data) for nid, data in G.nodes(data=True)
                           if data.get("lat") is not None]

    mapping: dict[str, str] = {}
    for _, row in df_stations.iterrows():
        cid  = str(row["station_complex_id"])
        slat = float(row["lat"])
        slon = float(row["lon"])
        best_id, best_dist = None, float("inf")
        for nid, nd in candidate_nodes:
            d = _haversine_km(slat, slon, float(nd["lat"]), float(nd["lon"]))
            if d < best_dist:
                best_dist = d
                best_id = nid
        if best_id is not None:
            mapping[cid] = str(best_id)
    return mapping


def _load_geo() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    census = gpd.read_file(RAW_DIR / "census_tracts.geojson").to_crs("EPSG:32618")

    bus_stops_path = RAW_DIR / "bus_stops.geojson"
    if bus_stops_path.exists():
        stops = gpd.read_file(bus_stops_path).to_crs("EPSG:32618")
    else:
        # No bus stop geometry available — return an empty GeoDataFrame
        stops = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:32618"))

    return census, stops


class _State:
    """All heavy artifacts loaded once at module import."""
    __slots__ = (
        "booster", "pf_booster", "meta",
        "G", "stop_line_map",
        "df_stations", "complex_to_stop",
        "census", "bus_stops",
        "main_nodes",
    )

    def __init__(self) -> None:
        self.booster, self.pf_booster, self.meta = _load_models()
        self.G                                   = _load_time_graph()
        self.stop_line_map                       = _load_stop_line_map()
        # After patching parent↔directional edges the graph is ~1 component;
        # keep all nodes with lat/lon as "main" candidates
        self.main_nodes: set = {
            nid for nid, data in self.G.nodes(data=True)
            if data.get("lat") is not None
        }
        self.df_stations                         = _load_stations()
        self.complex_to_stop                     = _build_complex_to_stop(
            self.df_stations, self.G, self.main_nodes
        )
        self.census, self.bus_stops              = _load_geo()


_S = _State()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_features(
    lat: float,
    lon: float,
    num_lines: int,
    borough: Optional[str] = None,
) -> dict[str, float]:
    pt_wgs = Point(lon, lat)
    pt_m   = gpd.GeoSeries([pt_wgs], crs="EPSG:4326").to_crs("EPSG:32618").iloc[0]

    f: dict[str, float] = {"num_lines": float(num_lines), "lat": lat, "lon": lon}

    # Census tract features
    containing = _S.census[_S.census.geometry.contains(pt_m)]
    if not containing.empty:
        t = containing.iloc[0]
        f["pop_density_tract"] = _safe_float(t.get("pop_density"))
        f["median_income"]     = _safe_float(t.get("median_income"))
        f["commuters_tract"]   = _safe_float(t.get("commuters"))
        if borough is None:
            geoid = str(t.get("GEOID") or "")
            if len(geoid) >= 5:
                borough = FIPS_TO_BOROUGH.get(geoid[2:5])
    else:
        f["pop_density_tract"] = 0.0
        f["median_income"]     = 0.0
        f["commuters_tract"]   = 0.0

    # 500 m population buffer
    buf         = pt_m.buffer(500)
    overlapping = _S.census[_S.census.geometry.intersects(buf)].copy()
    if not overlapping.empty and "population" in overlapping.columns:
        areas = overlapping.geometry.area.replace(0, np.nan)
        frac  = (overlapping.geometry.intersection(buf).area / areas).fillna(0.0)
        f["population_500m"] = float((frac * overlapping["population"].fillna(0)).sum())
    else:
        f["population_500m"] = 0.0

    # Bus stops within radius
    if not _S.bus_stops.empty:
        d = _S.bus_stops.geometry.distance(pt_m)
        f["bus_stops_250m"] = float((d <= 250).sum())
        f["bus_stops_500m"] = float((d <= 500).sum())
    else:
        f["bus_stops_250m"] = 0.0
        f["bus_stops_500m"] = 0.0

    # Borough one-hot
    for col in BOROUGH_COLS:
        f[col] = 0.0
    if borough:
        key = _BOROUGH_KEY.get(borough.strip().lower())
        if key:
            f[key] = 1.0

    return f


# ─────────────────────────────────────────────────────────────────────────────
# ML prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _predict_one(features: dict, booster: xgb.Booster, log_target: bool) -> float:
    cols = _S.meta["feature_columns"]
    row  = np.array([[_safe_float(features.get(c)) for c in cols]], dtype=np.float32)
    d    = xgb.DMatrix(row, feature_names=cols)
    p    = float(booster.predict(d)[0])
    return float(np.expm1(p)) if log_target else p


def _predict_line_ridership(
    stations: list[dict],
) -> tuple[int, int, list[dict]]:
    """
    Predict per-station ridership with the XGBoost model, attribute ATTR_FRAC
    of each station's total to the new line, then sum across all stops.

    Returns (total_daily, total_peak_hour, per_station_breakdown).
    """
    booster    = _S.booster
    pf_booster = _S.pf_booster
    log_t      = _S.meta.get("log_target_daily", True)
    fallback_pf = _S.meta.get("fallback_peak_factor", 0.10)
    df_s       = _S.df_stations

    total_daily = 0.0
    total_peak  = 0.0
    breakdown   = []

    for st in stations:
        sid    = str(st.get("id", ""))
        is_new = bool(st.get("is_new", False))

        if is_new:
            feats = _extract_features(
                lat=float(st["lat"]),
                lon=float(st["lon"]),
                num_lines=1,
            )
        else:
            row = df_s[df_s["station_complex_id"].astype(str) == sid]
            if row.empty:
                continue
            r = row.iloc[0]
            current_lines = r.get("lines") or []
            n_after = (len(current_lines) if isinstance(current_lines, list) else 1) + 1
            feats = _extract_features(
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                num_lines=n_after,
                borough=str(r.get("borough") or ""),
            )

        daily_pred = max(0.0, _predict_one(feats, booster, log_t))

        if pf_booster is not None:
            pf = max(0.02, min(0.30, _predict_one(feats, pf_booster, False)))
        else:
            pf = fallback_pf

        attr_daily = daily_pred * ATTR_FRAC
        attr_peak  = attr_daily * pf

        total_daily += attr_daily
        total_peak  += attr_peak
        breakdown.append({
            "station_id":       sid,
            "is_new":           is_new,
            "predicted_daily":  int(round(daily_pred)),
            "attributed_daily": int(round(attr_daily)),
            "peak_factor":      round(pf, 3),
        })

    return int(round(total_daily)), int(round(total_peak)), breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Graph utilities — mta_time_graph aware
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_graph_node(lat: float, lon: float) -> Optional[str]:
    """Find the closest main-component node in the time graph by haversine distance."""
    best_id, best_dist = None, float("inf")
    for nid in _S.main_nodes:
        data = _S.G.nodes[nid]
        nlat = data.get("lat")
        nlon = data.get("lon")
        if nlat is None or nlon is None:
            continue
        d = _haversine_km(lat, lon, float(nlat), float(nlon))
        if d < best_dist:
            best_dist = d
            best_id = nid
    return str(best_id) if best_id is not None else None


def _station_graph_node(st: dict) -> Optional[str]:
    """
    Resolve a station dict to its node ID in the time graph.
    Prefers provided lat/lon for accurate nearest-node lookup.
    Falls back to complex_id mapping when coordinates are absent.
    """
    lat = st.get("lat")
    lon = st.get("lon")
    if lat is not None and lon is not None:
        return _nearest_graph_node(float(lat), float(lon))
    # No coordinates — fall back to pre-built complex_id → graph node mapping
    sid = str(st.get("id", ""))
    return _S.complex_to_stop.get(sid)


def _edge_lines(G: nx.Graph, u: str, v: str) -> list[str]:
    """Return the lines (route_ids) serving the edge (u, v)."""
    lines_u = set(_S.stop_line_map.get(str(u), []))
    lines_v = set(_S.stop_line_map.get(str(v), []))
    common  = lines_u & lines_v
    if "NEW" in (G.get_edge_data(u, v) or {}).get("line", ""):
        return ["NEW"]
    return sorted(common) if common else sorted(lines_u | lines_v)


def _build_graph_with_new_line(stations: list[dict]) -> tuple[nx.Graph, list[str]]:
    """
    Return a copy of the time graph with the drawn line's edges added.
    New station nodes are inserted at their nearest existing node location
    (for routing purposes) or as new nodes (if truly novel geography).
    """
    G = _S.G.copy()

    node_ids: list[str] = []
    for st in stations:
        if st.get("is_new", False):
            sid = "new_" + str(st.get("id", "")).replace(" ", "_")
            if sid not in G.nodes:
                G.add_node(
                    sid,
                    id=sid,
                    name=st.get("name", "New Station"),
                    lat=float(st["lat"]),
                    lon=float(st["lon"]),
                )
            node_ids.append(sid)
        else:
            nid = _station_graph_node(st)
            node_ids.append(nid if nid else _nearest_graph_node(
                float(st["lat"]), float(st["lon"])
            ))

    for i in range(len(node_ids) - 1):
        a, b = node_ids[i], node_ids[i + 1]
        if a is None or b is None or a not in G.nodes or b not in G.nodes:
            continue
        la   = float(G.nodes[a].get("lat", 0))
        lo_a = float(G.nodes[a].get("lon", 0))
        lb   = float(G.nodes[b].get("lat", 0))
        lo_b = float(G.nodes[b].get("lon", 0))
        dist_km    = _haversine_km(la, lo_a, lb, lo_b)
        travel_sec = (dist_km / TRAVEL_SPEED_KMH) * 3600 + DWELL_PER_STOP_SEC
        G.add_edge(a, b, line="NEW", weight=travel_sec, dist_km=dist_km)

    return G, node_ids


def _path_weight_with_penalty(G: nx.Graph, path: list[str]) -> float:
    """
    Sum edge weights along the path and add TRANSFER_PENALTY_SEC whenever
    the dominant line changes between consecutive segments.
    """
    total     = 0.0
    last_line = None
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v) or {}
        if isinstance(next(iter(data.values()), None), dict):
            data = min(data.values(), key=lambda e: e.get("weight", float("inf")))
        total += float(data.get("weight", 0))
        lines = _edge_lines(G, u, v)
        cur_line = lines[0] if lines else None
        if last_line is not None and cur_line != last_line:
            total += TRANSFER_PENALTY_SEC
        last_line = cur_line
    return total


def _shortest(G: nx.Graph, src: str, dst: str) -> tuple[float, list[str]]:
    """Dijkstra shortest path; returns (weight, path) or (inf, [])."""
    try:
        length, path = nx.single_source_dijkstra(G, src, dst, weight="weight")
        return float(length), path
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
        return float("inf"), []


def _redistribution(
    stations: list[dict],
    new_line_daily: int,
) -> tuple[list[dict], list[dict]]:
    """
    Gravity-model O-D sampling.  For pairs where adding the new line creates a
    faster route, shift proportional ridership away from the old path.

    Two sampling phases:
    1. Random ridership-weighted sampling across all stations.
    2. Corridor-targeted sampling: explicitly tests pairs near the new line's
       geographic endpoints to capture cross-corridor redistribution.
    """
    G_old            = _S.G
    G_new, new_nodes = _build_graph_with_new_line(stations)

    df_s        = _S.df_stations[_S.df_stations["total_ridership"] > 0].copy()
    complex_ids = df_s["station_complex_id"].astype(str).values
    riderships  = df_s["total_ridership"].values.astype(float)
    station_nodes = [_S.complex_to_stop.get(c) for c in complex_ids]

    weights    = riderships / riderships.sum()
    rng        = np.random.default_rng(42)
    origin_idx = rng.choice(len(complex_ids), size=GRAVITY_SAMPLE * 2, replace=True, p=weights)
    dest_idx   = rng.choice(len(complex_ids), size=GRAVITY_SAMPLE * 2, replace=True, p=weights)

    line_loss:        dict[str, float] = defaultdict(float)
    station_loss:     dict[str, float] = defaultdict(float)
    line_baseline:    dict[str, float] = defaultdict(float)
    station_baseline: dict[str, float] = defaultdict(float)

    new_node_set = set(new_nodes) | {
        "new_" + str(s.get("id", "")).replace(" ", "_")
        for s in stations if s.get("is_new", False)
    }

    def _eval_pair(src_node: str, dst_node: str) -> None:
        """Evaluate one O-D pair and update loss/baseline accumulators."""
        if src_node is None or dst_node is None or src_node == dst_node:
            return
        old_w, old_path = _shortest(G_old, src_node, dst_node)
        if old_w == float("inf") or len(old_path) < 2:
            return
        for u, v in zip(old_path[:-1], old_path[1:]):
            for ln in _edge_lines(G_old, u, v):
                line_baseline[ln] += 1.0
        for n in old_path:
            station_baseline[n] += 1.0
        new_w, new_path = _shortest(G_new, src_node, dst_node)
        if new_w == float("inf"):
            return
        new_path_lines: set[str] = set()
        for u, v in zip(new_path[:-1], new_path[1:]):
            new_path_lines.update(_edge_lines(G_new, u, v))
        if "NEW" not in new_path_lines or new_w >= old_w - 30:
            return
        switch_frac = min(1.0, (old_w - new_w) / old_w)
        for u, v in zip(old_path[:-1], old_path[1:]):
            for ln in _edge_lines(G_old, u, v):
                if ln != "NEW":
                    line_loss[ln] += switch_frac
        for n in old_path:
            if n not in new_node_set:
                station_loss[n] += switch_frac

    # Phase 1: random ridership-weighted sampling
    evaluated = 0
    for oi, di in zip(origin_idx, dest_idx):
        if evaluated >= GRAVITY_SAMPLE:
            break
        sn, dn = station_nodes[oi], station_nodes[di]
        if sn is None or dn is None or sn == dn:
            continue
        evaluated += 1
        _eval_pair(sn, dn)

    # Phase 2: corridor-targeted sampling
    CORRIDOR_RADIUS_KM = 5.0
    CORRIDOR_SAMPLE    = GRAVITY_SAMPLE

    def _node_latlon(nid: str) -> Optional[tuple[float, float]]:
        nd = G_new.nodes.get(nid, {})
        lat, lon = nd.get("lat"), nd.get("lon")
        return (float(lat), float(lon)) if lat is not None and lon is not None else None

    start_ll = _node_latlon(new_nodes[0])
    end_ll   = _node_latlon(new_nodes[-1])

    if start_ll and end_ll:
        near_start: list[int] = []
        near_end:   list[int] = []
        for idx, sn in enumerate(station_nodes):
            if sn is None:
                continue
            nd   = G_old.nodes.get(sn, {})
            slat = nd.get("lat")
            slon = nd.get("lon")
            if slat is None:
                continue
            slat, slon = float(slat), float(slon)
            if _haversine_km(start_ll[0], start_ll[1], slat, slon) <= CORRIDOR_RADIUS_KM:
                near_start.append(idx)
            if _haversine_km(end_ll[0], end_ll[1], slat, slon) <= CORRIDOR_RADIUS_KM:
                near_end.append(idx)

        if near_start and near_end:
            corridor_pairs: list[tuple[int, int]] = []
            for a in near_start:
                for b in near_end:
                    corridor_pairs.append((a, b))
            for a in near_end:
                for b in near_start:
                    corridor_pairs.append((a, b))
            rng2 = np.random.default_rng(99)
            if len(corridor_pairs) > CORRIDOR_SAMPLE:
                chosen = rng2.choice(len(corridor_pairs), size=CORRIDOR_SAMPLE, replace=False)
                corridor_pairs = [corridor_pairs[i] for i in chosen]
            for oi, di in corridor_pairs:
                _eval_pair(station_nodes[oi], station_nodes[di])

    attributed_total = sum(station_loss.values())
    scale = (new_line_daily / attributed_total) if attributed_total > 0 else 0.0

    # ── Affected lines ────────────────────────────────────────────────────────
    affected_lines: list[dict] = []
    for ln, loss in line_loss.items():
        baseline = line_baseline.get(ln, 0)
        if baseline == 0:
            continue
        delta_pct = max(-25.0, -100.0 * loss / baseline)
        affected_lines.append({"line": ln, "delta_pct": round(delta_pct, 2)})
    affected_lines.sort(key=lambda x: x["delta_pct"])
    affected_lines = affected_lines[:8]

    # ── Affected stations ─────────────────────────────────────────────────────
    df_lookup = _S.df_stations.copy()
    df_lookup.index = df_lookup["station_complex_id"].astype(str)
    stop_to_complex: dict[str, str] = {v: k for k, v in _S.complex_to_stop.items()}

    affected_stations: list[dict] = []
    for graph_node, loss in station_loss.items():
        loss_riders = loss * scale
        if loss_riders < 50:
            continue
        cid = stop_to_complex.get(str(graph_node))
        if cid is None or cid not in df_lookup.index:
            continue
        row       = df_lookup.loc[cid]
        baseline  = max(float(row.get("total_ridership", 0) or 0), 1.0)
        delta     = -loss_riders
        delta_pct = max(-30.0, 100.0 * delta / baseline)
        affected_stations.append({
            "station_id":          cid,
            "name":                str(row.get("name", "")),
            "ridership_delta":     int(round(delta)),
            "ridership_delta_pct": round(delta_pct, 2),
        })
    affected_stations.sort(key=lambda x: x["ridership_delta"])
    affected_stations = affected_stations[:25]

    return affected_lines, affected_stations


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_line(stations: list[dict]) -> dict:
    """
    Args:
      stations: list of dicts in route order, each with:
        id (str)   — station_complex_id for existing; any unique string for new
        name (str)
        lat (float)
        lon (float)
        is_new (bool)

    Returns dict matching the SimulationResponse shape expected by the API.
    """
    if not stations or len(stations) < 2:
        return {
            "new_line_ridership":  0,
            "peak_hour_ridership": 0,
            "affected_lines":      [],
            "affected_stations":   [],
        }

    daily, peak, _ = _predict_line_ridership(stations)
    affected_lines, affected_stations = _redistribution(stations, daily)

    return {
        "new_line_ridership":  daily,
        "peak_hour_ridership": peak,
        "affected_lines":      affected_lines,
        "affected_stations":   affected_stations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_stations = [
        {"id": "611", "name": "Times Sq-42 St",    "lat": 40.75731,  "lon": -73.98676, "is_new": False},
        {"id": "new_0", "name": "New Mid Stop",    "lat": 40.76200,  "lon": -73.97500, "is_new": True},
        {"id": "120", "name": "96 St (1/2/3)",     "lat": 40.78430,  "lon": -73.97881, "is_new": False},
    ]
    print("Test stations:")
    for s in test_stations:
        print(f"  {'[new]' if s['is_new'] else '     '} {s['name']}")
    print()
    result = predict_new_line(test_stations)
    print(json.dumps(result, indent=2))
