#!/usr/bin/env python3
"""
Level 4: Comprehensive Simulation & What-If Scenario Analysis
Target: San Francisco Network Topology (GraphML-driven simulation)
"""

import os
import networkx as nx
import random
import json

class NetworkSimulationTwin:
    def __init__(self, graphml_path: str):
        self.graphml_path = graphml_path
        if not os.path.exists(graphml_path):
            raise FileNotFoundError(f"Network topology not found at {graphml_path}")
        print(f"📥 Loading network topology from {graphml_path}...")
        self.graph = nx.read_graphml(graphml_path)
        
        # Ensure edge weights (like 'length') are cast to float to prevent type errors during routing
        for u, v, data in self.graph.edges(data=True):
            if 'length' in data:
                try:
                    data['length'] = float(data['length'])
                except (ValueError, TypeError):
                    data['length'] = 1.0
            else:
                data['length'] = 1.0
                
        print(f"✅ Loaded network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def run_what_if_disruption(self, sample_source_target: tuple = None, removal_fraction: float = 0.05):
        """
        Executes a What-If Disruption Scenario: Removes a fraction of vital edges 
        (simulating road closures or transit strikes) and measures network degradation.
        """
        print(f"\n🧪 [LEVEL 4 SIMULATION] Initializing What-If Disruption Scenario...")
        
        # Backup original graph state
        working_graph = self.graph.copy()
        edges = list(working_graph.edges())
        
        # Select target edges to disable
        num_to_remove = max(1, int(len(edges) * removal_fraction))
        disrupted_edges = random.sample(edges, num_to_remove)
        
        print(f"⚠️ Simulating sudden closure of {num_to_remove} critical network corridors...")
        working_graph.remove_edges_from(disrupted_edges)
        
        # Pick sample nodes for route analysis if not provided
        nodes = list(working_graph.nodes())
        if len(nodes) >= 2:
            source, target = sample_source_target if sample_source_target else (nodes[0], nodes[min(10, len(nodes)-1)])
            
            try:
                # Compute alternative path post-disruption using the numeric 'length' weight
                path = nx.shortest_path(working_graph, source=source, target=target, weight='length')
                print(f"🔀 Alternative Route computed successfully between {source} and {target} ({len(path)} nodes).")
            except nx.NetworkXNoPath:
                print(f"❌ Critical Failure: Network disconnected between {source} and {target} due to disruption!")

        # Calculate network structural metrics post-disruption
        is_connected = nx.is_weakly_connected(working_graph) if working_graph.is_directed() else nx.is_connected(working_graph)
        
        results = {
            "scenario": "link_closure",
            "edges_removed": num_to_remove,
            "network_fully_connected": is_connected,
            "remaining_nodes": working_graph.number_of_nodes(),
            "remaining_edges": working_graph.number_of_edges()
        }
        
        print(f"📊 Scenario Output Metrics: {json.dumps(results, indent=2)}")
        return results

if __name__ == "__main__":
    topology_path = "../data/processed/network_topology.graphml"
    simulator = NetworkSimulationTwin(topology_path)
    simulator.run_what_if_disruption(removal_fraction=0.03)