import os
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, GRAPH_FILENAME

def validate_and_visualize_network():
    """
    Validates network topology and generates a baseline static map for Level 1.
    """
    graph_path = os.path.join(PROCESSED_DATA_DIR, GRAPH_FILENAME)
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}. Please run extract_spatial_data.py first.")
        return

    print(f"Loading network topology from {graph_path}...")
    graph = ox.load_graphml(graph_path)

    # Step 1.3: Structural Validation
    print("\n--- Network Topology Validation ---")
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    is_weakly_connected = nx.is_weakly_connected(graph)
    
    print(f"  - Total Intersections (Nodes): {num_nodes}")
    print(f"  - Total Road Segments (Edges): {num_edges}")
    print(f"  - Network is Weakly Connected: {is_weakly_connected}")
    
    if not is_weakly_connected:
        components = list(nx.weakly_connected_components(graph))
        print(f"  - Notice: Network contains {len(components)} disconnected components.")

    # Step 1.4: Baseline Static Map Visualization
    print("\n--- Generating Baseline Visualization ---")
    fig, ax = ox.plot_graph(
        graph, 
        show=False, 
        close=False, 
        node_size=0, 
        edge_color="#00ffcc", 
        edge_linewidth=0.6, 
        bgcolor="#111111"
    )
    
    plot_output_path = os.path.join(PROCESSED_DATA_DIR, "network_baseline.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Baseline map successfully saved to {plot_output_path}")

if __name__ == "__main__":
    validate_and_visualize_network()
