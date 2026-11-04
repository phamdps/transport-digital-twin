import json
import pytest
from pathlib import Path

from twins.model_router import TransportModelRouter
from twins.nlp_simulator import NaturalLanguageIncidentParser
from twins.agentic_controller import AutonomousAgentController


def test_twins_nlp_graph_mutation():
    """Test that NLP parser successfully loads the real NetworkX graph and mutates edges."""
    prompt = "Simulate a 20% capacity reduction disaster near Market Street."
    result = NaturalLanguageIncidentParser.simulate_incident_from_prompt(prompt)
    
    assert result["status"] == "SUCCESS"
    assert result["incident_zone"] == "Market Street Corridor"
    assert result["edges_closed"] > 0
    assert 0.0 <= result["network_lcc_health_ratio"] <= 1.0


def test_twins_agentic_telemetry_loop():
    """Test that the agentic controller reads live telemetry data and persists control signals."""
    controller = AutonomousAgentController()
    result = controller.evaluate_and_act()
    
    assert "status" in result
    assert "mean_fleet_speed_kmh" in result
    assert controller.control_path.exists(), "Control signal file must be written to disk."