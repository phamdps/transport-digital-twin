import os
import osmnx as ox
from config import PROCESSED_DATA_DIR, GRAPH_FILENAME, BASE_DIR

def generate_interactive_map():
    """
    Generates an interactive HTML map of the street network using GeoPandas explore 
    and CartoDB dark_matter tiles, saving directly to the results directory.
    """
    graph_path = os.path.join(PROCESSED_DATA_DIR, GRAPH_FILENAME)
    
    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}. Please run extract_spatial_data.py first.")
        return

    print(f"Loading network topology from {graph_path}...")
    graph = ox.load_graphml(graph_path)

    print("Converting graph to GeoDataFrame and generating interactive map...")
    # Extract edges GeoDataFrame from the graph
    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    # Generate interactive map using GeoPandas explore with dark mode tiles
    m = edges.explore(
        tiles="cartodbdarkmatter",
        color="#00ffcc",
        style_kwds={"weight": 1.5, "opacity": 0.8}
    )

    # Define results directory and ensure it exists
    results_dir = BASE_DIR / "results"
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "network_interactive.html")
    m.save(output_path)
    
    print(f"✅ Interactive map successfully saved to {output_path}")

if __name__ == "__main__":
    generate_interactive_map()
