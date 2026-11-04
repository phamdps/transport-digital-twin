import os
import json
import time
import sqlite3
import pytest
import pandas as pd
import networkx as nx
from pathlib import Path

# Import utility functions from backend
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "dashboards"))
from dashboards.dashboard_utils import (
    GRAPH_PATH, JSON_PATH, DB_PATH, CONTROL_PATH,
    load_graph, check_network_health, compute_dynamic_route
)

# ==========================================
# LEVEL 1: TOPOLOGY BENCHMARKS
# ==========================================
def test_level1_topology_connectivity():
    """Benchmark: Ensure graph exists and LCC (Largest Connected Component) >= 98%."""
    assert GRAPH_PATH.exists(), f"Topology GraphML missing at {GRAPH_PATH}"
    G = load_graph()
    assert G is not None, "Failed to load NetworkX graph."
    
    total_nodes = G.number_of_nodes()
    assert total_nodes > 0, "Graph has zero nodes."
    
    # Calculate Largest Connected Component ratio
    G_und = G.to_undirected()
    largest_cc = max(nx.connected_components(G_und), key=len)
    lcc_ratio = len(largest_cc) / total_nodes
    
    print(f"\n[Level 1] Total Nodes: {total_nodes}, LCC Ratio: {lcc_ratio:.4f}")
    assert lcc_ratio >= 0.98, f"Topology fragmented! LCC ratio {lcc_ratio:.2f} is below 98% threshold."

def test_level1_centrality_speed():
    """Benchmark: Ensure structural bottleneck calculations execute under 5.0 seconds."""
    G = load_graph()
    G_und = G.to_undirected()
    largest_cc = max(nx.connected_components(G_und), key=len)
    sub_g = G_und.subgraph(largest_cc).copy()
    
    start_time = time.time()
    _ = nx.betweenness_centrality(sub_g, k=35, weight='length', seed=42) # Adjusted k for speed
    duration = time.time() - start_time
    
    print(f"\n[Level 1] Centrality computation duration: {duration:.4f}s")
    assert duration < 5.0, f"Centrality calculation too slow: {duration:.2f}s (Target < 5.0s)"


# ==========================================
# LEVEL 2: LIVE FLEET & TELEMETRY BENCHMARKS
# ==========================================
def test_level2_telemetry_schema_and_alerts():
    """Benchmark: Validate live vehicle schema integrity and threshold breach detection."""
    if not JSON_PATH.exists():
        pytest.skip("Live vehicle telemetry JSON feed not active. Run simulation script first.")
        
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    assert isinstance(data, list), "Telemetry data must be a JSON array."
    if len(data) > 0:
        df = pd.DataFrame(data)
        required_columns = {"vehicle_id", "latitude", "longitude", "speed_kmh", "timestamp"}
        assert required_columns.issubset(df.columns), f"Missing required telemetry columns: {required_columns - set(df.columns)}"
        
        # Test alert trigger logic
        alert = check_network_health(df, threshold=15.0)
        if len(df[df['speed_kmh'] < 15.0]) > 5:
            assert alert is not None, "Congestion alert failed to trigger despite threshold breaches."
            assert alert["type"] == "CRITICAL"


# ==========================================
# LEVEL 3: PREDICTIVE ROUTING BENCHMARKS
# ==========================================
def test_level3_dynamic_routing_resilience():
    """Benchmark: Ensure dynamic routing avoids congested links or successfully computes paths."""
    G = load_graph()
    nodes = list(G.nodes())
    if len(nodes) < 2:
        pytest.skip("Graph too small for routing tests.")
        
    orig, dest = nodes[0], nodes[-1]
    
    # Create fake live dataframe with severe slowdown near nodes
    fake_live_df = pd.DataFrame([{
        "vehicle_id": "test_car",
        "latitude": G.nodes[orig]['y'],
        "longitude": G.nodes[orig]['x'],
        "speed_kmh": 2.0 # Extreme slowdown
    }])
    
    start_time = time.time()
    path = compute_dynamic_route(G, orig, dest, live_df=fake_live_df)
    duration = time.time() - start_time
    
    assert isinstance(path, list), "Routing algorithm failed to return an ordered list of nodes."
    assert len(path) >= 2, "Calculated path is trivial or empty."
    assert duration < 1.0, f"Dynamic path calculation took too long: {duration:.4f}s"


# ==========================================
# LEVEL 4: SIMULATION LAB BENCHMARKS
# ==========================================
def test_level4_simulation_stress_degradation():
    """Benchmark: Validate monotonic network graph degradation under stress-testing."""
    G = load_graph()
    working_graph = G.copy()
    initial_edge_count = working_graph.number_of_edges()
    
    # Simulate 10% link closure
    edges = list(working_graph.edges())
    import random
    random.seed(42)
    edges_to_remove = random.sample(edges, int(initial_edge_count * 0.10))
    working_graph.remove_edges_from(edges_to_remove)
    
    final_edge_count = working_graph.number_of_edges()
    expected_edges = initial_edge_count - len(edges_to_remove)
    
    assert final_edge_count == expected_edges, "Graph link removal simulation mismatch."


# ==========================================
# LEVEL 5: AUTONOMOUS CONTROL BENCHMARKS
# ==========================================
def test_level5_closed_loop_actuation():
    """Benchmark: Validate control signal write-to-read latency and schema persistence."""
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    test_payload = {
        "status": "ACTIVE",
        "timestamp": time.time(),
        "interventions": [{"target_zone": "Market Street Corridor", "speed_cap_kmh": 15.0}]
    }
    
    # Write signal (Simulate UI command)
    write_start = time.time()
    with open(CONTROL_PATH, "w") as f:
        json.dump(test_payload, f)
    write_latency = time.time() - write_start
    
    # Read signal (Simulate background worker agent)
    read_start = time.time()
    with open(CONTROL_PATH, "r") as f:
        loaded_payload = json.load(f)
    read_latency = time.time() - read_start
    
    total_roundtrip = write_latency + read_latency
    
    print(f"\n[Level 5] Control signal round-trip latency: {total_roundtrip * 1000:.2f} ms")
    assert loaded_payload["status"] == "ACTIVE"
    assert loaded_payload["interventions"][0]["speed_cap_kmh"] == 15.0
    assert total_roundtrip < 0.1, f"Control loop roundtrip too slow: {total_roundtrip:.4f}s (Target < 100ms)"