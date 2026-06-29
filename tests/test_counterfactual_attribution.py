"""Issue 2 — honest attribution: is_achievable reflects whether the ACTIONABLE
changes alone flip the decision, and the advertised saving is attributed to the
exact actionable world (not to structural construction)."""

from __future__ import annotations

import math

from ontofact_nav import AStarPlanner, NavEdge, NavigationGraph, NavNode
from ontofact_nav.affordance import AffordanceReasoner
from ontofact_nav.counterfactual import CounterfactualEngine, PropertyChange

CLEAR = dict(width=2.0, height=2.5, surface_type="smooth",
             illumination=1.0, is_accessible=True)


def _engine_and_world(onto, robot, direct_props, detour_distance=12.0):
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("via", (6.0, 6.0)), ("g", (12.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    direct = onto.create("seg_direct", "Corridor", **direct_props)
    sv = onto.create("seg_sv", "Corridor", **CLEAR)
    vg = onto.create("seg_vg", "Corridor", **CLEAR)
    g.add_edge(NavEdge("s", "g",   direct, distance=10.0))
    g.add_edge(NavEdge("s", "via", sv,     distance=detour_distance))
    g.add_edge(NavEdge("via", "g", vg,     distance=detour_distance))
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    actual = planner.find_path("s", "g", robot)
    return engine, actual


def test_structural_savings_not_achievable(onto, make_agent):
    # Alt blocked only by a too-narrow corridor (width = non-actionable).
    robot = make_agent("bot", robot_width=1.0, min_clearance=0.15)
    engine, actual = _engine_and_world(onto, robot, dict(CLEAR, width=0.9))
    cf = engine.explain_why_not(actual, ["s", "g"], robot)

    assert {c.property_name for c in cf.changes} == {"width"}
    assert cf.cost_delta > 0                      # widening WOULD make it cheaper
    assert cf.is_achievable is False              # ...but not by operator action
    assert cf.actionable_only_changes == []
    assert math.isinf(cf.actionable_only_cost)
    assert cf.actionable_only_delta == 0.0


def test_actionable_attribution_matches(onto, make_agent):
    # Alt blocked only by a closed door (robot can't open) — fully actionable.
    robot = make_agent("bot", can_open_doors=False)
    engine, actual = _engine_and_world(onto, robot, dict(CLEAR, door_state="closed"))
    cf = engine.explain_why_not(actual, ["s", "g"], robot)

    assert cf.is_achievable is True
    assert {c.property_name for c in cf.actionable_only_changes} == {"door_state"}
    # The advertised saving is computed from the exact actionable world.
    assert cf.actionable_only_cost == cf.cf_cost
    assert cf.actionable_only_delta == cf.cost_delta


def test_mixed_blockers_not_misattributed(onto, make_agent):
    # The exact old bug: a door (actionable) AND an over-steep slope (structural)
    # both block the route.  Opening the door alone is NOT enough, so even though
    # an actionable change appears in `changes`, is_achievable must be False —
    # proving it is no longer `any(actionable)`.
    robot = make_agent("bot", mobility_type="wheeled",
                       max_slope_angle=8.0, can_open_doors=False)
    engine, actual = _engine_and_world(onto, robot, dict(
        CLEAR, door_state="closed", slope_angle=18.0,
    ))
    cf = engine.explain_why_not(actual, ["s", "g"], robot)

    assert {c.property_name for c in cf.changes} == {"door_state", "slope_angle"}
    assert any(c.is_actionable() for c in cf.changes)   # the door is actionable
    assert cf.is_achievable is False                    # ...yet not achievable alone
    assert cf.actionable_only_changes == []
    assert math.isinf(cf.actionable_only_cost)


def test_hidden_structural_blocker_makes_unachievable(onto, make_agent):
    # Regression for the truncation bug's effect on attribution: edge 1 is
    # hazardous (actionable), edge 2 is too narrow (structural). The old code
    # truncated at edge 1, never saw the width blocker, and wrongly reported the
    # route operator-achievable. With the full path visible, is_achievable=False.
    robot = make_agent("bot", robot_width=1.0, min_clearance=0.15)
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("n1", (1.0, 0.0)), ("g", (2.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    seg0 = onto.create("seg0", "Corridor", **dict(CLEAR, is_hazardous=True))
    seg1 = onto.create("seg1", "Corridor", **dict(CLEAR, width=0.9))   # < required 1.3
    detour = onto.create("detour", "Corridor", **CLEAR)
    g.add_edge(NavEdge("s", "n1", seg0, distance=1.0), bidirectional=False)
    g.add_edge(NavEdge("n1", "g", seg1, distance=1.0), bidirectional=False)
    g.add_edge(NavEdge("s", "g", detour, distance=5.0), bidirectional=False)
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    actual = planner.find_path("s", "g", robot)

    cf = engine.explain_why_not(actual, ["s", "n1", "g"], robot)
    assert {c.property_name for c in cf.changes} == {"is_hazardous", "width"}
    assert any(c.is_actionable() for c in cf.changes)   # the hazard fix is actionable
    assert cf.is_achievable is False                    # ...but width is structural
    assert cf.actionable_only_changes == []
    assert math.isinf(cf.actionable_only_cost)


def test_is_actionable_and_effort_are_orthogonal():
    def mk(prop):
        return PropertyChange("seg", prop, 0, 1)
    # Actionable yet medium effort (policy change) — the two axes are independent.
    assert mk("restricted").is_actionable() is True
    assert mk("restricted").effort() == "medium"
    assert mk("is_accessible").is_actionable() is True
    assert mk("is_accessible").effort() == "medium"
    # Non-actionable and high effort (construction).
    assert mk("width").is_actionable() is False
    assert mk("width").effort() == "high"
    # Actionable and low effort (runtime action).
    assert mk("door_state").is_actionable() is True
    assert mk("door_state").effort() == "low"
