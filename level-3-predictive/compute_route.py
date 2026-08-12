import os
import osmnx as ox
import networkx as nx
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
GRAPH_FILENAME = "network_topology.graphml"

def compute_shortest_path():
    """
    Computes the shortest path between two coordinates in the San Francisco network topology.
    """
    graph_path = PROCESSED_DATA_DIR / GRAPH_FILENAME
    
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}. Please complete Level 1 first.")
        return

    print(f"Loading network topology from {graph_path}...")
    G = ox.load_graphml(graph_path)

    # Define Origin and Destination coordinates in San Francisco
    # Example: Financial District (Origin) to Oracle Park (Destination)
    origin_coords = (37.7937, -122.3965)  # (lat, lon)
    destination_coords = (37.7786, -122.3893)

    print("Mapping coordinates to nearest network graph nodes...")
    origin_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
    destination_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])

    print(f"Computing shortest path between nodes {origin_node} and {destination_node}...")
    
    try:
        # Compute shortest path using edge lengths
        path = nx.shortest_path(G, source=origin_node, target=destination_node, weight="length")
        
        # Calculate total distance in meters
        total_length_m = 0
        for u, v in zip(path[:-1], path[1:]):
            # Handle multi-edge graphs by taking the minimum length edge between u and v
            edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('length', 0))
            total_length_m += edge_data.get('length', 0)

        print(f"✅ Route successfully calculated!")
        print(f"   - Total Path Nodes: {len(path)}")
        print(f"   - Total Distance: {total_length_m:.2f} meters ({total_length_m / 1000:.2f} km)")

        return path

    except nx.NetworkXNoPath:
        print("❌ Error: No valid path found between the specified origin and destination.")
        return None

if __name__ == "__main__":
    compute_shortest_path()
