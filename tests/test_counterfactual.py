"""Tests for the counterfactual engine and the OntofactNavigator façade."""

from __future__ import annotations

import math

import pytest

from ontofact_nav import (
    AStarPlanner,
    NavEdge,
    NavigationGraph,
    NavNode,
    OntofactNavigator,
)
from ontofact_nav.counterfactual import CounterfactualEngine, PropertyChange


# ---------------------------------------------------------------------------
# PropertyChange classification
# ---------------------------------------------------------------------------

def _change(prop):
    return PropertyChange(
        individual_name="seg", property_name=prop,
        original_value=0, counterfactual_value=1,
    )


@pytest.mark.parametrize("prop, actionable", [
    ("door_state", True),
    ("crowd_density", True),
    ("illumination", True),
    ("restricted", True),
    ("width", False),
    ("slope_angle", False),
])
def test_is_actionable(prop, actionable):
    assert _change(prop).is_actionable() is actionable


@pytest.mark.parametrize("prop, level", [
    ("width", "high"),
    ("slope_angle", "high"),
    ("restricted", "medium"),
    ("is_accessible", "medium"),
    ("door_state", "low"),
    ("crowd_density", "low"),
])
def test_effort_levels(prop, level):
    assert _change(prop).effort() == level


# ---------------------------------------------------------------------------
# A world where the direct route is blocked by a door the robot can't open,
# with a clear detour via 'via'.
# ---------------------------------------------------------------------------

@pytest.fixture
def blocked_world(onto, make_space, make_agent):
    robot = make_agent("bot", can_open_doors=False)
    g = NavigationGraph()
    for nid, pos in [("start", (0.0, 0.0)), ("via", (5.0, 5.0)), ("goal", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("start", "goal", make_space(name="seg_direct", door_state="closed"),
                       distance=10.0))
    g.add_edge(NavEdge("start", "via", make_space(name="seg_sv"), distance=7.07))
    g.add_edge(NavEdge("via", "goal", make_space(name="seg_vg"), distance=7.07))
    return onto, g, robot


def test_explain_why_not_finds_actionable_door_fix(blocked_world, reasoner):
    onto, graph, robot = blocked_world
    planner = AStarPlanner(graph, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)

    actual = planner.find_path("start", "goal", robot)
    assert actual.nodes == ["start", "via", "goal"]   # took the detour

    cf = engine.explain_why_not(actual, ["start", "goal"], robot)
    assert cf.changes, "expected a fix for the blocked direct route"
    assert any(c.property_name == "door_state" for c in cf.changes)
    assert cf.is_achievable is True
    assert cf.actionable_changes()
    assert math.isfinite(cf.cf_cost)          # opening the door makes it passable
    assert cf.cf_cost < actual.total_cost     # ...and cheaper than the detour


def test_explain_why_not_optimal_alternative_needs_no_changes(make_space, reasoner, make_agent, onto):
    robot = make_agent("bot")
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("x", (5.0, 5.0)),
                     ("y", (5.0, -5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    g.add_edge(NavEdge("s", "x", make_space(name="sx"), distance=7.07))
    g.add_edge(NavEdge("x", "g", make_space(name="xg"), distance=7.07))
    g.add_edge(NavEdge("s", "y", make_space(name="sy", crowd_density=0.4), distance=7.07))
    g.add_edge(NavEdge("y", "g", make_space(name="yg"), distance=7.07))

    planner = AStarPlanner(g, reasoner)
    engine = CounterfactualEngine(reasoner, planner, onto.individuals)
    actual = planner.find_path("s", "g", robot)

    cf = engine.explain_why_not(actual, ["s", "y", "g"], robot)
    assert cf.changes == []              # the costlier-but-feasible route needs no fix
    assert cf.is_achievable is False
    assert "optimal" in cf.explanation.lower()


# ---------------------------------------------------------------------------
# OntofactNavigator integration
# ---------------------------------------------------------------------------

def test_navigator_navigate_and_why_not(blocked_world):
    onto, graph, robot = blocked_world
    nav = OntofactNavigator(onto, graph)

    path, explanation = nav.navigate("start", "goal", robot, k_alternatives=3)
    assert path.is_feasible
    assert path.nodes == ["start", "via", "goal"]
    assert explanation is not None

    cf = nav.query_why_not("start", "goal", robot, ["start", "goal"])
    assert cf.is_achievable is True
    assert any(c.property_name == "door_state" for c in cf.changes)


def test_navigator_reports_infeasible_when_no_route(make_space, make_agent, onto):
    robot = make_agent("bot", can_open_doors=False)
    g = NavigationGraph()
    g.add_node(NavNode("a", (0.0, 0.0)))
    g.add_node(NavNode("b", (10.0, 0.0)))
    # Only connection is a locked door — impassable for everyone.
    g.add_edge(NavEdge("a", "b", make_space(name="seg_locked", door_state="locked"),
                       distance=10.0))
    nav = OntofactNavigator(onto, g)

    path, explanation = nav.navigate("a", "b", robot)
    assert path.is_feasible is False
    assert explanation is None
