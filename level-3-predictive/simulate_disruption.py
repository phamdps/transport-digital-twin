import os
import osmnx as ox
import networkx as nx
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
GRAPH_FILENAME = "network_topology.graphml"

def run_disruption_scenario():
    """
    Simulates a road closure along a baseline travel route and computes 
    the dynamic digital twin rerouting response.
    """
    graph_path = PROCESSED_DATA_DIR / GRAPH_FILENAME
    
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}. Please complete Level 1 first.")
        return

    print(f"Loading network topology from {graph_path}...")
    G = ox.load_graphml(graph_path)

    # Define Origin and Destination coordinates in San Francisco
    origin_coords = (37.7937, -122.3965)  # Financial District
    destination_coords = (37.7786, -122.3893)  # Oracle Park

    print("Mapping coordinates to network nodes...")
    origin_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
    destination_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])

    # 1. Compute baseline shortest path
    baseline_path = nx.shortest_path(G, source=origin_node, target=destination_node, weight="length")
    
    baseline_length = 0
    for u, v in zip(baseline_path[:-1], baseline_path[1:]):
        edge_data = min(G.get_edge_data(u, v).values(), key=lambda x: x.get('length', 0))
        baseline_length += edge_data.get('length', 0)

    print(f"\n📊 Baseline Route Computed:")
    print(f"   - Total Path Nodes: {len(baseline_path)}")
    print(f"   - Baseline Distance: {baseline_length:.2f} meters ({baseline_length / 1000:.2f} km)")

    # 2. Simulate Disruption: Block a critical street link right in the middle of the baseline path
    if len(baseline_path) > 4:
        mid_idx = len(baseline_path) // 2
        u_block, v_block = baseline_path[mid_idx], baseline_path[mid_idx + 1]
        
        print(f"\n🚨 SIMULATING DISRUPTION EVENT:")
        print(f"   - Road closure injected between nodes: {u_block} -> {v_block}")

        # Create a modified copy of the graph with the edge removed
        G_disrupted = G.copy()
        if G_disrupted.has_edge(u_block, v_block):
            G_disrupted.remove_edge(u_block, v_block)
        if G_disrupted.has_edge(v_block, u_block):
            G_disrupted.remove_edge(v_block, u_block)

        # 3. Re-compute path on the disrupted network topology
        try:
            disrupted_path = nx.shortest_path(G_disrupted, source=origin_node, target=destination_node, weight="length")
            
            disrupted_length = 0
            for u, v in zip(disrupted_path[:-1], disrupted_path[1:]):
                edge_data = min(G_disrupted.get_edge_data(u, v).values(), key=lambda x: x.get('length', 0))
                disrupted_length += edge_data.get('length', 0)

            print(f"\n🔄 Alternative Rerouted Path Computed:")
            print(f"   - Rerouted Nodes: {len(disrupted_path)}")
            print(f"   - Rerouted Distance: {disrupted_length:.2f} meters ({disrupted_length / 1000:.2f} km)")
            
            detour_diff = disrupted_length - baseline_length
            print(f"   - ⚠️ Scenario Impact: Detour adds +{detour_diff:.2f} meters (+{(detour_diff/baseline_length)*100:.1f}%)")

        except nx.NetworkXNoPath:
            print("❌ Error: Disruption caused a complete network partition (no alternative route found).")
    else:
        print("Baseline path too short to simulate an internal network disruption.")

if __name__ == "__main__":
    run_disruption_scenario()
