import pandas as pd
import networkx as nx
import json
from collections import defaultdict
from networkx.readwrite import json_graph

stops = pd.read_csv("data/gtfs_subway/stops.txt")
stop_times = pd.read_csv("data/gtfs_subway/stop_times.txt")
transfers = pd.read_csv("data/gtfs_subway/transfers.txt")

stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

TRANSFER_PENALTY = 60  # time on top of transfer time given

def gtfs_time_to_seconds(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

G = nx.Graph()


for _, row in stops.iterrows():
    G.add_node(row["stop_id"], name=row["stop_name"],
               lat=row["stop_lat"], lon=row["stop_lon"])

edge_times = defaultdict(list)

# for building out weights, taking average time for more realistic ridership data
for trip_id, group in stop_times.groupby("trip_id"):
    rows = group.reset_index(drop=True)
    for i in range(len(rows) - 1):
        u = rows.loc[i,   "stop_id"]
        v = rows.loc[i+1, "stop_id"]
        dep = gtfs_time_to_seconds(rows.loc[i,   "departure_time"])
        arr = gtfs_time_to_seconds(rows.loc[i+1, "arrival_time"])
        edge_times[(u, v)].append(arr - dep)

for (u, v), times in edge_times.items():
    G.add_edge(u, v, weight=sum(times) / len(times))
    
# Connect directional platform nodes to their parent station node so that
# transfer edges (which use parent IDs) are reachable from line chains.
PLATFORM_ACCESS_SECONDS = 60
for _, row in stops.iterrows():
    child = str(row["stop_id"])
    parent = row.get("parent_station")
    if pd.isna(parent) or not parent:
        continue
    parent = str(parent)
    if not G.has_node(parent) or not G.has_node(child):
        continue
    if G.has_edge(parent, child):
        G[parent][child]["weight"] = min(G[parent][child]["weight"], PLATFORM_ACCESS_SECONDS)
    else:
        G.add_edge(parent, child, weight=PLATFORM_ACCESS_SECONDS)

for _, row in transfers.iterrows():
    u, v = row["from_stop_id"], row["to_stop_id"]
    if u == v:
        continue

    # min_transfer_time is in seconds; default to 120s if missing
    t = row.get("min_transfer_time", 120)
    if pd.isna(t):
        t = 120
    t = int(t) + TRANSFER_PENALTY

    if G.has_edge(u, v):
        G[u][v]["weight"] = min(G[u][v]["weight"], t)
    else:
        G.add_edge(u, v, weight=t)

data = json_graph.node_link_data(G)
with open("data/processed/mta_time_graph.json", "w") as f:
    json.dump(data, f)
