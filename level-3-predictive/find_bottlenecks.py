import os
import osmnx as ox
import networkx as nx
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
GRAPH_FILENAME = "network_topology.graphml"

def identify_bottlenecks():
    """
    Computes betweenness centrality across the San Francisco network topology 
    to pinpoint critical vulnerability chokepoints and arterial lifelines.
    """
    graph_path = PROCESSED_DATA_DIR / GRAPH_FILENAME
    
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}. Please complete Level 1 first.")
        return

    print(f"Loading network topology from {graph_path}...")
    G = ox.load_graphml(graph_path)

    # Convert to undirected graph for global structural flow analysis
    print("Preparing network topology for centrality calculations...")
    G_undirected = G.to_undirected()

    # Extract the largest connected component to ensure mathematical continuity
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    subgraph = G_undirected.subgraph(largest_cc).copy()

    print(f"Analyzing {subgraph.number_of_nodes()} intersections across the San Francisco network...")
    
    # Compute betweenness centrality using a randomized sample for high performance
    # (k parameter samples a subset of nodes to calculate path flows efficiently)
    sample_size = min(150, subgraph.number_of_nodes())
    centrality_scores = nx.betweenness_centrality(
        subgraph, 
        k=sample_size, 
        weight='length', 
        seed=42
    )

    # Sort nodes by highest centrality score
    sorted_critical_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n🚨 TOP 5 CRITICAL VULNERABILITY CHOKE POINTS IN SAN FRANCISCO:")
    for i, (node_id, score) in enumerate(sorted_critical_nodes[:5], 1):
        node_data = subgraph.nodes[node_id]
        lat, lon = node_data.get('y', 0.0), node_data.get('x', 0.0)
        print(f"   {i}. Intersection Node ID: {node_id}")
        print(f"      - Coordinates: ({lat:.4f}, {lon:.4f})")
        print(f"      - Criticality Score: {score:.5f}")

    print("\n✅ Network vulnerability profiling complete!")
    print("   Planners can use these metrics to prioritize structural reinforcement, emergency response positioning, or congestion pricing.")

if __name__ == "__main__":
    identify_bottlenecks()
