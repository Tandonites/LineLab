# borough_parser.py
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import Point

BOROUGH_DATA_PATH = "data/raw/Borough_Boundaries_20260426.csv"

def load_borough_polygons():
    df = pd.read_csv(BOROUGH_DATA_PATH)
    boroughs = []
    for _, row in df.iterrows():
        polygon = loads(row["the_geom"])
        boroughs.append((row["BoroName"], polygon))
    return boroughs

def get_borough(lat, lon, boroughs):
    point = Point(lon, lat)
    for name, polygon in boroughs:
        if polygon.contains(point):
            return name
    return "Unknown"