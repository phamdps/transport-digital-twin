import os
import osmnx as ox
import networkx as nx
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
GRAPH_FILENAME = "network_topology.graphml"

def compute_accessibility_isochrones():
    """
    Computes travel-time isochrones (reachable areas within 3, 6, and 9 minutes) 
    from a central point in San Francisco using graph network topology.
    """
    graph_path = PROCESSED_DATA_DIR / GRAPH_FILENAME
    
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}. Please complete Level 1 first.")
        return

    print(f"Loading network topology from {graph_path}...")
    G = ox.load_graphml(graph_path)

    # Assume an average urban vehicle speed of 30 km/h (8.33 meters per second)
    average_speed_mps = 30 * 1000 / 3600 

    print("Computing edge travel times based on network geometry and speed assumptions...")
    for u, v, data in G.edges(data=True):
        length_m = data.get('length', 10.0) # default fallback length
        # Travel time in seconds = distance / speed
        data['travel_time'] = length_m / average_speed_mps

    # Choose a center point (e.g., Union Square, San Francisco)
    center_coords = (37.7879, -122.4075)
    center_node = ox.distance.nearest_nodes(G, X=center_coords[1], Y=center_coords[0])

    print(f"📍 Center Node selected: {center_node} near Union Square")

    # Define time horizons in seconds (3 mins, 6 mins, 9 mins)
    time_horizons = [180, 360, 540]
    horizon_labels = ["3 Minutes", "6 Minutes", "9 Minutes"]

    print("\n⏱️ CALCULATING ACCESSIBILITY ISOCHRONES:")
    for seconds, label in zip(time_horizons, horizon_labels):
        # Extract ego subgraph bounded by travel time limit
        subgraph = nx.ego_graph(G, center_node, radius=seconds, distance='travel_time')
        
        reachable_nodes = len(subgraph.nodes())
        reachable_edges = len(subgraph.edges())
        
        print(f"   - Within {label}:")
        print(f"     * Reachable Intersections (Nodes): {reachable_nodes}")
        print(f"     * Reachable Street Segments (Edges): {reachable_edges}")

    print("\n✅ Isochrone accessibility metrics computed successfully!")
    print("   Digital twin can now evaluate service catchments and spatial equity across SF neighborhoods.")

if __name__ == "__main__":
    compute_accessibility_isochrones()
