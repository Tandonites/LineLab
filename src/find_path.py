import networkx as nx
import json
from networkx.readwrite import json_graph

JSON_GRAPH_FILE = "data/processed/mta_time_graph.json"

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def find_stop_ids(G, name):
    return [nid for nid, d in G.nodes(data=True) if name in d["name"]]

# passthrough using names
def shortest_path_name(G, src_name, dst_name):
    srcs = find_stop_ids(G, src_name)
    dsts = find_stop_ids(G, dst_name)
    return shortest_path(G, srcs, dsts)

# def shortest_path_obj(G, src_obj, dst_obj):
    # srcs = [nid for nid, node in G.nodes(data=True) if nid.startswith(src_obj.id)]
    # dsts = [nid for nid, node in G.nodes(data=True) if nid.startswith(dst_obj.id)]
    # return shortest_path(G, srcs, dsts)

def shortest_path(G, src_ids, dst_ids):
    best_time, best_path = float("inf"), None

    for src in src_ids:
        for dst in dst_ids:
            try:
                t = nx.dijkstra_path_length(G, src, dst, weight="weight")
                if t < best_time:
                    best_time = t
                    best_path = nx.dijkstra_path(G, src, dst, weight="weight")
            except nx.NetworkXNoPath:
                continue

    return best_time, best_path

def find_path(src_name, dst_name, display=False) -> tuple[float, ]:
    G = load_graph(JSON_GRAPH_FILE)
    best_time, best_path = shortest_path_name(G, src_name, dst_name)

    if display:
        print(f"Estimated time: {int(best_time // 60)}m {int(best_time % 60)}s")
        print(f"Path: {[G.nodes[s]['name'] for s in best_path]}")

    return best_time, best_path
