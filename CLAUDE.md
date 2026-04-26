# NYC Transit New Line Predictor

## Project overview

A tool that identifies transit gaps in NYC, proposes hypothetical new subway corridors, and models the cascading effects on existing line ridership and newly viable trips. The baseline case is a Queens-Brooklyn direct connection — a corridor that has been publicly debated for years (the BQX) but never backed by rigorous demand modeling.

The core pitch: the MTA and city planners make expensive infrastructure decisions without a fast, interactive tool to model demand shifts. We built that tool.

## Problem statement

There is no direct subway connection between Queens and Brooklyn without going through Manhattan. Riders traveling between the two boroughs — say, from Astoria to Park Slope — must transfer through Manhattan, adding significant time and crowding existing lines like the 7 and N. The MTA has no publicly available tool to model what a direct Queens-Brooklyn line would do to ridership across the network.

## What the tool does

1. Visualizes the existing subway network and identifies transit desert zones
2. Lets the user draw or select a hypothetical new line corridor
3. Computes shortest paths for all OD (origin-destination) pairs on the existing network
4. Re-routes those paths with the new line added and diffs the results
5. Outputs:
   - Ridership diverted away from existing lines (e.g. how much the 7 and N lose)
   - Newly viable trips — OD pairs where travel time drops enough to make the trip worth taking for the first time
   - Rough cost estimate based on line length and per-mile construction benchmarks
6. Displays everything on an interactive map

## Technical pipeline

### Stage 1 — Network graph
- Parse MTA GTFS static feed into a NetworkX directed graph
- Nodes = stations, edges = track segments weighted by travel time
- Join MTA ridership data to station nodes as demand weights

### Stage 2 — OD demand matrix
- Process Census LODES OD flow data into a demand matrix (where people travel between)
- This tells us where people *want* to go, not just where they currently go

### Stage 3 — Path substitution model
- Run Dijkstra/A* shortest path for each OD pair on the existing network
- Add hypothetical new line edges to the graph
- Recompute shortest paths and diff against baseline
- Count diverted OD pairs per existing line → ridership delta
- Count newly viable trips (OD pairs where new path is meaningfully shorter)

### Stage 4 — Cost estimation
- Calculate new line length from geometry
- Apply per-mile construction cost benchmarks (NYC subway costs are well-documented public data)
- Output rough total cost estimate

### Stage 5 — Visualization
- Interactive map built with Folium or Plotly
- Layers: existing network, hypothetical new line, ridership shift heatmap, newly viable trips overlay
- Simple controls for judges to interact with

## Data sources

| Dataset | Source | Purpose |
|---|---|---|
| MTA GTFS static feed | mta.info/developers | Network graph (stops, routes, shapes) |
| MTA ridership (Oct 1–10 2024) | MTA Open Data portal | Station-level demand weights |
| Census LODES OD flows | lehd.ces.census.gov | Origin-destination demand matrix |
| Borough boundaries GeoJSON | NYC Open Data | Map background layer |

## Tech stack

- Python (pandas, geopandas, networkx, gtfs-kit)
- Folium or Plotly for interactive map
- Simple REST API to connect backend to frontend
- Frontend: interactive map UI with controls

## Key differentiators

- **Generative, not reactive** — proposes what should exist, not just analyzing what does
- **Cascade modeling** — shows second-order effects on existing lines, not just the new line in isolation
- **Induced demand** — newly viable trips captures demand the current network suppresses entirely
- **Actionable output** — gives planners the information layer they currently lack, framed as "here's the data you need to make this decision"

## Baseline demo

Queens-Brooklyn direct connection. Judges can see exactly how much demand the 7 and N currently absorb from would-be Queens-Brooklyn travelers, how a direct line would redistribute that ridership, and how many new trips become viable that weren't before.
