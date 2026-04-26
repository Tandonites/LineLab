import json
from api import haversine_km, StationInput
from find_path import find_path

WALK_SPEED_MS = 1.4

def parse_stations(api_response: str) -> tuple[bool, list[StationInput]]:
    data = json.loads(api_response)

    stations = data.get("stations", [])

    station_objects = [
        StationInput(
            id=station["id"],
            name=station["name"],
            lat=station["lat"],
            lon=station["lon"],
            is_new=station["is_new"]
        )
        for station in stations
    ]

    return station_objects

# returns (current time in seconds, is walking-only route)
def calculate_current_time(stations: list[StationInput]) -> tuple[float, bool]:
    start = stations[0]
    end = stations[-1]

    # pure walking time
    total_dist_km = haversine_km(start.lat, start.lon, end.lat, end.lon)
    pure_walk_time = (total_dist_km * 1000) / WALK_SPEED_MS

    # two-pointer traversal to find the first and last existing stations
    lo, hi = 0, len(stations) - 1

    while lo < len(stations) and stations[lo].is_new:
        lo += 1
    while hi >= 0 and stations[hi].is_new:
        hi -= 1

    # no existing stations, can only walk
    if lo > hi:
        return pure_walk_time, True

    existing_start = stations[lo]
    existing_end = stations[hi]

    # walking distance between new endpoints and their nearest existing stations
    walk_time = 0
    if start.is_new:
        walk_dist = haversine_km(start.lat, start.lon, existing_start.lat, existing_start.lon) * 1000
        walk_time += walk_dist / WALK_SPEED_MS

    if end.is_new:
        walk_dist = haversine_km(end.lat, end.lon, existing_end.lat, existing_end.lon) * 1000
        walk_time += walk_dist / WALK_SPEED_MS

    # run Dijkstra's on the existing station segment
    best_time, _ = find_path(existing_start.name, existing_end.name)
    transit_time = best_time + walk_time

    return min(pure_walk_time, transit_time), pure_walk_time < transit_time

# testing, creates StationInput from list
if __name__ == "__main__":

    # end of Q line + walking up 7 blocks
    Q_ext = [
        StationInput(id="476", name="86 St", lat=40.77789, lon=-73.95179, is_new=False),
        StationInput(id="475", name="96 St", lat=40.784317, lon=-73.94715, is_new=False),
        StationInput(id="1476", name="103 St", lat=40.788755, lon=-73.943821, is_new=True)
    ]

    # # walk laterally, shorter to walk then take 1 down to columbus then B/C up
    # lateral = [
    #     StationInput(id="618", name="14 St", lat=40.740894, lon=-74.00169, is_new=False),
    #     StationInput(id="405", name="23 St", lat=40.739864, lon=-73.9866, is_new=False),
    # ]

    best_time, walking_only = calculate_current_time(Q_ext)
    print(f"time: {int(best_time // 60)}m {int(best_time % 60)}s", "walking-only" if walking_only else "using transit")
