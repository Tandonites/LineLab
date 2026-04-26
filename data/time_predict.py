# time_predict.py
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from borough_parser import load_borough_polygons, get_borough
from timegraph_parser import haversine

TRAINING_DATA_PATH = "data/processed/time_training_data.csv"
EXPRESS_MULTIPLIER = 1.25  # express trains are 1.25x faster than local
MODEL_PATH = "data/processed/time_model.joblib"

# compute mean speed dynamically from training data as default for new stations
_df = pd.read_csv(TRAINING_DATA_PATH)
MEAN_SPEED_MS = (_df["distance_m"] / _df["travel_time_seconds"]).mean()

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
def parse_and_predict_route(stations, train_service, model):
    
    total_seconds = 0.0
    segment_times = []

    for i in range(len(stations) - 1):
        u = stations[i]
        v = stations[i + 1]

        # extract coordinates and borough from each station object
        lat1, lon1 = u["lat"], u["lon"]
        lat2, lon2 = v["lat"], v["lon"]
        borough_u = u["borough"]
        borough_v = v["borough"]

        # predict time for this segment
        seg_time = predict_time(lat1, lon1, lat2, lon2, borough_u, borough_v, train_service, model)

        # log each segment for debugging
        u_label = u["name"] if u["name"] else f"New Station ({lat1}, {lon1})"
        v_label = v["name"] if v["name"] else f"New Station ({lat2}, {lon2})"
        print(f"Segment {u_label} -> {v_label}: {int(seg_time // 60)}m {int(seg_time % 60)}s")

        segment_times.append({
            "from": u_label,
            "to": v_label,
            "travel_time_seconds": seg_time
        })

        total_seconds += seg_time

    print(f"Total predicted time: {int(total_seconds // 60)}m {int(total_seconds % 60)}s")

    return {
        "total_seconds": total_seconds,
        "total_minutes": total_seconds / 60,
        "segments": segment_times
    }

if __name__ == "__main__":
    # load existing model if available, otherwise train and save a new one
    if os.path.exists(MODEL_PATH):
        print("Loading existing model")
        model = load_model()
    else:
        print("Training new model")
        model = train_model()
        save_model(model)

    # test parse_and_predict_route with a sample stations array
    test_stations = [
        {"station_complex_id": "", "name": "Jackson Heights-Roosevelt Ave", "borough": "Queens", "lat": 40.7466, "lon": -73.8912},
        {"station_complex_id": "", "name": "Elmhurst-Queens Blvd", "borough": "Queens", "lat": 40.7411, "lon": -73.8893},
        {"station_complex_id": "", "name": "Maspeth-Grand Ave", "borough": "Queens", "lat": 40.7297, "lon": -73.8821},
        {"station_complex_id": "", "name": "Eliot Ave", "borough": "Queens", "lat": 40.7203, "lon": -73.8837},
        {"station_complex_id": "", "name": "Metropolitan Ave", "borough": "Queens", "lat": 40.7118, "lon": -73.8893},
        {"station_complex_id": "", "name": "Myrtle Ave", "borough": "Queens", "lat": 40.7003, "lon": -73.8943},
        {"station_complex_id": "", "name": "Wilson Ave", "borough": "Brooklyn", "lat": 40.6888, "lon": -73.9042},
        {"station_complex_id": "", "name": "Atlantic Ave", "borough": "Brooklyn", "lat": 40.6766, "lon": -73.9038},
        {"station_complex_id": "", "name": "Sutter Ave", "borough": "Brooklyn", "lat": 40.6685, "lon": -73.9025},
        {"station_complex_id": "", "name": "Livonia Ave", "borough": "Brooklyn", "lat": 40.6641, "lon": -73.9026},
        {"station_complex_id": "", "name": "Linden Blvd", "borough": "Brooklyn", "lat": 40.6586, "lon": -73.9028},
        {"station_complex_id": "", "name": "Remsen Ave", "borough": "Brooklyn", "lat": 40.6521, "lon": -73.9103},
        {"station_complex_id": "", "name": "Utica Ave", "borough": "Brooklyn", "lat": 40.6413, "lon": -73.9284},
        {"station_complex_id": "", "name": "Brooklyn College-Flatbush Ave", "borough": "Brooklyn", "lat": 40.6327, "lon": -73.9477},
        {"station_complex_id": "", "name": "East 16th St", "borough": "Brooklyn", "lat": 40.6299, "lon": -73.9616},
        {"station_complex_id": "", "name": "McDonald Ave", "borough": "Brooklyn", "lat": 40.6259, "lon": -73.9762},
        {"station_complex_id": "", "name": "New Utrecht Ave", "borough": "Brooklyn", "lat": 40.6191, "lon": -73.9995},
        {"station_complex_id": "", "name": "Eighth Ave", "borough": "Brooklyn", "lat": 40.6288, "lon": -74.0116},
        {"station_complex_id": "", "name": "Brooklyn Army Terminal", "borough": "Brooklyn", "lat": 40.6451, "lon": -74.0242},
    ]

    parse_and_predict_route(test_stations, "express", model)
