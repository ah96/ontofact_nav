"""Tests for graph data structures, validation, A*, and k-shortest paths."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from ontofact_nav import (
    AStarPlanner,
    NavEdge,
    NavigationGraph,
    NavNode,
    NavPath,
)


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

def test_navnode_rejects_empty_id():
    with pytest.raises(ValidationError):
        NavNode("", (0.0, 0.0))


def test_navnode_rejects_non_finite_position():
    with pytest.raises(ValidationError):
        NavNode("n", (math.inf, 0.0))


def test_navedge_rejects_non_positive_distance(make_space):
    with pytest.raises(ValidationError):
        NavEdge("a", "b", make_space(), distance=0.0)


def test_navpath_rejects_negative_cost():
    with pytest.raises(ValidationError):
        NavPath(nodes=[], edges=[], total_cost=-1.0, affordance_results=[])


def test_navnode_equality_and_hash_by_id():
    assert NavNode("x", (0.0, 0.0)) == NavNode("x", (9.0, 9.0))
    assert len({NavNode("x", (0.0, 0.0)), NavNode("x", (1.0, 1.0))}) == 1


def test_infeasible_sentinel():
    p = NavPath.infeasible()
    assert p.is_feasible is False
    assert p.total_cost == math.inf
    assert p.nodes == []


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_add_edge_is_bidirectional_by_default(make_space):
    g = NavigationGraph()
    g.add_node(NavNode("a", (0.0, 0.0)))
    g.add_node(NavNode("b", (1.0, 0.0)))
    g.add_edge(NavEdge("a", "b", make_space(), distance=1.0))
    assert [e.to_id for e in g.neighbors("a")] == ["b"]
    assert [e.to_id for e in g.neighbors("b")] == ["a"]   # reverse added


def test_neighbors_of_unknown_node_is_empty(make_space):
    assert NavigationGraph().neighbors("ghost") == []


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

def _linear_graph(make_space):
    """s --10-- m --10-- g, all clear corridors."""
    g = NavigationGraph()
    g.add_node(NavNode("s", (0.0, 0.0)))
    g.add_node(NavNode("m", (10.0, 0.0)))
    g.add_node(NavNode("g", (20.0, 0.0)))
    g.add_edge(NavEdge("s", "m", make_space(), distance=10.0))
    g.add_edge(NavEdge("m", "g", make_space(), distance=10.0))
    return g


def test_astar_finds_linear_path(make_space, reasoner, agent):
    planner = AStarPlanner(_linear_graph(make_space), reasoner)
    path = planner.find_path("s", "g", agent)
    assert path.is_feasible
    assert path.nodes == ["s", "m", "g"]
    assert path.total_cost == pytest.approx(20.0)


def test_astar_unknown_endpoints_are_infeasible(make_space, reasoner, agent):
    planner = AStarPlanner(_linear_graph(make_space), reasoner)
    assert planner.find_path("s", "nowhere", agent).is_feasible is False
    assert planner.find_path("nowhere", "g", agent).is_feasible is False


def _diamond_graph(make_space):
    """Two routes s->g: via 'x' (clear) and via 'y' (crowded, pricier)."""
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("x", (5.0, 5.0)),
                     ("y", (5.0, -5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("s", "x", make_space(), distance=7.07))
    g.add_edge(NavEdge("x", "g", make_space(), distance=7.07))
    g.add_edge(NavEdge("s", "y", make_space(crowd_density=0.5), distance=7.07))
    g.add_edge(NavEdge("y", "g", make_space(), distance=7.07))
    return g


def test_astar_prefers_cheaper_route(make_space, reasoner, agent):
    planner = AStarPlanner(_diamond_graph(make_space), reasoner)
    path = planner.find_path("s", "g", agent)
    assert path.nodes == ["s", "x", "g"]   # avoids the crowded 'y' corridor


# ---------------------------------------------------------------------------
# k-shortest paths (Yen / NetworkX)
# ---------------------------------------------------------------------------

def test_k_paths_returns_sorted_alternatives(make_space, reasoner, agent):
    planner = AStarPlanner(_diamond_graph(make_space), reasoner)
    paths = planner.find_k_paths("s", "g", agent, k=2)
    assert len(paths) == 2
    assert paths[0].nodes == ["s", "x", "g"]
    assert paths[0].total_cost <= paths[1].total_cost


def test_k_paths_excludes_impassable_edges(make_space, reasoner, agent):
    # Same diamond as above, but the 'y' route's second leg is a locked door,
    # so the only feasible route is via 'x'.
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("x", (5.0, 5.0)),
                     ("y", (5.0, -5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("s", "x", make_space(), distance=7.07))
    g.add_edge(NavEdge("x", "g", make_space(), distance=7.07))
    g.add_edge(NavEdge("s", "y", make_space(), distance=7.07))
    g.add_edge(NavEdge("y", "g", make_space(door_state="locked"), distance=7.07))

    planner = AStarPlanner(g, reasoner)
    paths = planner.find_k_paths("s", "g", agent, k=5)
    assert paths, "expected at least the clear route"
    assert all("y" not in p.nodes for p in paths)


# ---------------------------------------------------------------------------
# evaluate_sequence (used by the counterfactual engine)
# ---------------------------------------------------------------------------

def test_evaluate_sequence_costs_explicit_route(make_space, reasoner, agent):
    planner = AStarPlanner(_linear_graph(make_space), reasoner)
    path = planner.evaluate_sequence(["s", "m", "g"], agent)
    assert path.is_feasible
    assert path.total_cost == pytest.approx(20.0)


def test_evaluate_sequence_missing_edge_is_infeasible(make_space, reasoner, agent):
    planner = AStarPlanner(_linear_graph(make_space), reasoner)
    assert planner.evaluate_sequence(["s", "g"], agent).is_feasible is False


def test_evaluate_sequence_single_node_is_trivially_feasible(make_space, reasoner, agent):
    planner = AStarPlanner(_linear_graph(make_space), reasoner)
    path = planner.evaluate_sequence(["s"], agent)
    assert path.is_feasible
    assert path.total_cost == 0.0
