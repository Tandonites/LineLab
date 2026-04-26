import networkx as nx
import json
import sys
from networkx.readwrite import json_graph

JSON_GRAPH_FILE = "data/processed/mta_time_graph.json"

def load_graph(path):
    with open(path) as f:
        data = json.load(f)
    return json_graph.node_link_graph(data)

def find_stop_ids(G, name):
    return [nid for nid, d in G.nodes(data=True) if name in d["name"]]

def shortest_path(G, src_name, dst_name):

    srcs = find_stop_ids(G, src_name)
    dsts = find_stop_ids(G, dst_name)

    best_time, best_path = float("inf"), None

    for src in srcs:
        for dst in dsts:
            try:
                t = nx.dijkstra_path_length(G, src, dst, weight="weight")
                if t < best_time:
                    best_time = t
                    best_path = nx.dijkstra_path(G, src, dst, weight="weight")
            except nx.NetworkXNoPath:
                continue

    return best_time, best_path

if __name__ == "__main__":
    src_name = sys.argv[1]
    dst_name = sys.argv[2]

    G = load_graph(JSON_GRAPH_FILE)
    best_time, best_path = shortest_path(G, src_name, dst_name)

    print(f"Estimated time: {int(best_time // 60)}m {int(best_time % 60)}s")
    print(f"Path: {[G.nodes[s]['name'] for s in best_path]}")
