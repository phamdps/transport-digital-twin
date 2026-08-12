import os
import random
import time
import json
import osmnx as ox
from pathlib import Path

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
REALTIME_DATA_DIR = BASE_DIR / "data" / "realtime"
GRAPH_FILENAME = "network_topology.graphml"

def run_simulation():
    """
    Simulates live public transit vehicle telemetry moving across the San Francisco network topology.
    """
    os.makedirs(REALTIME_DATA_DIR, exist_ok=True)
    graph_path = PROCESSED_DATA_DIR / GRAPH_FILENAME
    
    if not graph_path.exists():
        print(f"Error: Graph not found at {graph_path}. Please complete Level 1 first.")
        return

    print(f"Loading network topology from {graph_path}...")
    G = ox.load_graphml(graph_path)
    
    nodes = list(G.nodes(data=True))
    if not nodes:
        print("Error: No nodes found in the network graph.")
        return

    # Initialize a fleet of simulated vehicles
    num_vehicles = 15
    vehicles = []
    for i in range(num_vehicles):
        start_node = random.choice(nodes)
        vehicles.append({
            "vehicle_id": f"SFMTA-BUS-{200 + i}",
            "current_node": start_node[0],
            "lat": start_node[1]['y'],
            "lon": start_node[1]['x'],
            "speed_kmh": random.uniform(12.0, 40.0)
        })

    print(f"🚀 Initialized live simulation stream with {num_vehicles} active vehicles.")
    print("Broadcasting telemetry frames every 5 seconds. Press Ctrl+C to exit.\n")

    try:
        while True:
            timestamp = time.time()
            telemetry_batch = []
            
            for v in vehicles:
                # Find valid connected street intersections
                neighbors = list(G.neighbors(v["current_node"]))
                if neighbors:
                    next_node = random.choice(neighbors)
                    node_data = G.nodes[next_node]
                    v["current_node"] = next_node
                    v["lat"] = node_data['y']
                    v["lon"] = node_data['x']
                
                telemetry_batch.append({
                    "timestamp": timestamp,
                    "vehicle_id": v["vehicle_id"],
                    "latitude": v["lat"],
                    "longitude": v["lon"],
                    "speed_kmh": round(v["speed_kmh"], 2)
                })

            # Write out snapshot to our operational streaming buffer
            output_file = REALTIME_DATA_DIR / "latest_vehicles.json"
            with open(output_file, "w") as f:
                json.dump(telemetry_batch, f, indent=2)

            print(f"[{time.strftime('%H:%M:%S')}] Published telemetry snapshot: {len(telemetry_batch)} vehicles updated.")
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Live vehicle stream simulation terminated safely.")

if __name__ == "__main__":
    run_simulation()
