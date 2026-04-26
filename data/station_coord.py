import pandas as pd

stops = pd.read_csv("data/gtfs_subway/stops.txt")
stations = [
    "14 St-Union Sq",
    "23 St",
    "Times Sq-42 St",
    "42 St-Port Authority Bus Terminal",
    "Jay St-MetroTech",
    "Fulton St",
    "Jackson Hts-Roosevelt Av",
    "74 St-Broadway",
    "Howard Beach-JFK Airport",
    "Broad Channel",
    "Stillwell Av",
    "Kings Hwy"
]

for name in stations:
    match = stops[stops["stop_name"].str.contains(name, na=False)].iloc[0]
    print(f"{match['stop_name']}: {match['stop_lat']}, {match['stop_lon']}")