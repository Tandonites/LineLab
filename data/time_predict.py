# time_predict.py
import pandas as pd
import joblib
import os
import math
from pathlib import Path
from xgboost import XGBRegressor

try:
    from .borough_parser import load_borough_polygons, get_borough
except ImportError:
    from borough_parser import load_borough_polygons, get_borough

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAINING_DATA_PATH = ROOT_DIR / "data" / "processed" / "time_training_data.csv"
EXPRESS_MULTIPLIER = 1.25  # express trains are 1.25x faster than local
MODEL_PATH = ROOT_DIR / "data" / "models" / "time_model.joblib"

# compute mean speed dynamically from training data as default for new stations
_df = pd.read_csv(TRAINING_DATA_PATH)
MEAN_SPEED_MS = (_df["distance_m"] / _df["travel_time_seconds"]).mean()


_BOROUGH_POLYGONS = None


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _get_borough_polygons():
    global _BOROUGH_POLYGONS
    if _BOROUGH_POLYGONS is None:
        _BOROUGH_POLYGONS = load_borough_polygons()
    return _BOROUGH_POLYGONS

# encode boroughs as integers since XGBoost requires numeric features
BOROUGH_ENCODING = {
    "Manhattan": 0,
    "Brooklyn": 1,
    "Queens": 2,
    "Bronx": 3,
    "Staten Island": 4,
    "Unknown": 5
}

def train_model():
    df = pd.read_csv(TRAINING_DATA_PATH)

    df["borough_u"] = df["borough_u"].map(BOROUGH_ENCODING)
    df["borough_v"] = df["borough_v"].map(BOROUGH_ENCODING)

    # drop any rows with missing values
    df = df.dropna(subset=["borough_u", "borough_v", "distance_m", "travel_time_seconds", "speed_ms"])

    X = df[["distance_m", "borough_u", "borough_v", "speed_ms"]]
    y = df["travel_time_seconds"]

    model = XGBRegressor()
    model.fit(X, y)

    return model

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    return joblib.load(MODEL_PATH)

def predict_time(lat1, lon1, lat2, lon2, borough_u, borough_v, train_service, model):
    dist = haversine(lat1, lon1, lat2, lon2)

    b_u = BOROUGH_ENCODING.get(borough_u, 5)
    b_v = BOROUGH_ENCODING.get(borough_v, 5)

    features = pd.DataFrame([{
        "distance_m": dist,
        "borough_u": b_u,
        "borough_v": b_v,
        "speed_ms": MEAN_SPEED_MS  # default to mean speed since new stations have no historical data
    }])

    predicted = model.predict(features)[0]

    # apply express multiplier if user selected express service
    if train_service == "express":
        predicted = predicted / EXPRESS_MULTIPLIER

    return predicted

# Taking list of station objects from frontend and predicts total travel time along new route
def parse_and_predict_route(stations, train_service, model, verbose=True):
    
    total_seconds = 0.0
    segment_times = []

    for i in range(len(stations) - 1):
        u = stations[i]
        v = stations[i + 1]

        # extract coordinates and borough from each station object
        lat1, lon1 = u["lat"], u["lon"]
        lat2, lon2 = v["lat"], v["lon"]
        borough_u = u.get("borough") or get_borough(lat1, lon1, _get_borough_polygons())
        borough_v = v.get("borough") or get_borough(lat2, lon2, _get_borough_polygons())

        # predict time for this segment
        seg_time = predict_time(lat1, lon1, lat2, lon2, borough_u, borough_v, train_service, model)

        # log each segment for debugging
        u_label = u["name"] if u["name"] else f"New Station ({lat1}, {lon1})"
        v_label = v["name"] if v["name"] else f"New Station ({lat2}, {lon2})"
        if verbose:
            print(f"Segment {u_label} -> {v_label}: {int(seg_time // 60)}m {int(seg_time % 60)}s")

        segment_times.append({
            "from": u_label,
            "to": v_label,
            "travel_time_seconds": seg_time
        })

        total_seconds += seg_time

    if verbose:
        print(f"Total predicted time: {int(total_seconds // 60)}m {int(total_seconds % 60)}s")

    return {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "segments": segment_times
    }

