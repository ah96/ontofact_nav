"""FastAPI endpoint smoke tests, including JSON-safety of the why-not response
when the actual path is infeasible (cost_delta unbounded)."""

from __future__ import annotations

from fastapi.testclient import TestClient

import ontofact_nav.api as api


def _client() -> TestClient:
    # The context manager triggers the startup event that builds the scenario.
    return TestClient(api.app)


def test_health_lists_agents():
    with _client() as c:
        assert c.get("/health").status_code == 200
        agents = c.get("/agents").json()["agents"]
        assert "delivery_bot" in agents and "cargo_bot" in agents


def test_navigate_feasible():
    with _client() as c:
        r = c.post("/navigate", json={
            "start": "entrance", "goal": "icu",
            "agent_name": "delivery_bot", "k_alternatives": 3,
        })
        assert r.status_code == 200
        assert r.json()["is_feasible"] is True


def test_why_not_feasible_actual_is_json_safe():
    with _client() as c:
        r = c.post("/why-not", json={
            "start": "entrance", "goal": "icu", "agent_name": "delivery_bot",
            "alt_nodes": ["entrance", "lobby", "icu"],
        })
        assert r.status_code == 200
        assert isinstance(r.json()["is_achievable"], bool)


def test_hospital_scenario_builder_loads_real_world():
    # ONTOFACT_SCENARIO=hospital must load the actual hospital world (3 robots,
    # icu_main node) — previously it silently fell back to the 4-node demo.
    api._build_hospital_scenario()
    try:
        assert set(api._state["agents"]) == {"delivery_bot", "cargo_bot", "legged_bot"}
        assert "icu_main" in api._state["graph"].nodes
    finally:
        api._build_demo_scenario()   # restore default state for the other tests


def test_why_not_infeasible_actual_does_not_crash():
    # cargo_bot cannot reach icu → actual path infeasible → cost_delta unbounded.
    # The response must serialise (cost_delta = None), not raise
    # "Out of range float values are not JSON compliant".
    with _client() as c:
        r = c.post("/why-not", json={
            "start": "entrance", "goal": "icu", "agent_name": "cargo_bot",
            "alt_nodes": ["entrance", "lobby", "icu"],
        })
        assert r.status_code == 200
        assert r.json()["cost_delta"] is None
