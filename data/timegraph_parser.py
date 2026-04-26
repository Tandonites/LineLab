import json
import csv
import math
import networkx as nx
from networkx.readwrite import json_graph
from borough_parser import load_borough_polygons, get_borough


# for accurately calculating distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000 # Earth's radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

with open("data/processed/mta_time_graph.json") as f:
    data = json.load(f)

G = json_graph.node_link_graph(data)
boroughs = load_borough_polygons()

rows = []
for u, v, edge_data in G.edges(data=True):
    u_data = G.nodes[u]
    v_data = G.nodes[v]

    # skip if either node is missing coordinates
    if not all(k in u_data for k in ["lat", "lon"]) or \
       not all(k in v_data for k in ["lat", "lon"]):
        continue

    dist = haversine(u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"])
    speed = dist / edge_data["weight"] if edge_data["weight"] > 0 else 0

    rows.append({
        "stop_id_u": u,
        "lat_u": u_data["lat"],
        "lon_u": u_data["lon"],
        "stop_id_v": v,
        "lat_v": v_data["lat"],
        "lon_v": v_data["lon"],
        "distance_m": dist,
        "travel_time_seconds": edge_data["weight"],
        "borough_u": get_borough(u_data["lat"], u_data["lon"], boroughs),
        "borough_v": get_borough(v_data["lat"], v_data["lon"], boroughs),
        "speed_ms": speed
    })

with open("data/processed/time_training_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted {len(rows)} edges")