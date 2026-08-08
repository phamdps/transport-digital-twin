from pathlib import Path

# Target Urban Area Configuration
TARGET_PLACE = "San Francisco, California, USA"

# Network Types to Extract from OpenStreetMap
NETWORK_TYPE = "drive"

# Simplification settings for the graph topology
SIMPLIFY_NETWORK = True

# Output Paths (Safely anchored relative to the project root)
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

NODES_FILENAME = "network_nodes.geojson"
EDGES_FILENAME = "network_edges.geojson"
GRAPH_FILENAME = "network_topology.graphml"
