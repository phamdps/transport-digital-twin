import os
import osmnx as ox
import geopandas as gpd
from config import TARGET_PLACE, NETWORK_TYPE, SIMPLIFY_NETWORK, PROCESSED_DATA_DIR, NODES_FILENAME, EDGES_FILENAME, GRAPH_FILENAME

def extract_network():
    """
    Downloads spatial network data for the configured place and exports 
    nodes, edges, and topology graph files.
    """
    print(f"Fetching network for: {TARGET_PLACE} (network type: {NETWORK_TYPE})...")
    
    # Download the street network graph from OpenStreetMap
    graph = ox.graph_from_place(TARGET_PLACE, network_type=NETWORK_TYPE, simplify=SIMPLIFY_NETWORK)
    
    # Convert graph elements to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    
    print(f"Extraction complete:")
    print(f"  - Nodes (Intersections): {len(nodes)}")
    print(f"  - Edges (Road Segments): {len(edges)}")
    
    # Ensure output directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Export spatial files
    nodes_path = os.path.join(PROCESSED_DATA_DIR, NODES_FILENAME)
    edges_path = os.path.join(PROCESSED_DATA_DIR, EDGES_FILENAME)
    graph_path = os.path.join(PROCESSED_DATA_DIR, GRAPH_FILENAME)
    
    nodes.to_file(nodes_path, driver="GeoJSON")
    edges.to_file(edges_path, driver="GeoJSON")
    ox.save_graphml(graph, filepath=graph_path)
    
    print(f"Successfully saved assets to {PROCESSED_DATA_DIR}/")

if __name__ == "__main__":
    extract_network()
