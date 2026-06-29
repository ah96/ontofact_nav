"""Issue 1 — the counterfactual is genuinely MINIMAL: it returns the smallest
subset of changes that flips the decision, not every applicable fix."""

from __future__ import annotations

import math

from ontofact_nav import AStarPlanner, NavEdge, NavigationGraph, NavNode
from ontofact_nav.affordance import AffordanceReasoner
from ontofact_nav.counterfactual import CounterfactualEngine

CLEAR = dict(width=2.0, height=2.5, surface_type="smooth",
             illumination=1.0, is_accessible=True)


def _engine_and_world(onto, robot, direct_props, detour_distance=12.0):
    """s --direct-- g (direct_props) and a clear detour s --via-- g."""
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
    return engine, planner, actual


def test_minimal_picks_only_the_door_not_soft_costs(onto, make_agent):
    # Direct edge is blocked by a closed door (robot can't open) and is ALSO
    # crowded and dark.  The OLD engine dumped all three fixes; the minimal
    # engine returns only the door (opening it alone makes the route preferred).
    # `risk_certified=True` so the crowded+dark edge is NOT a hard HighRiskZone
    # block here — keeping crowd/illumination as the *unnecessary* soft costs
    # this test is about (HighRiskZone gating is covered in test_classification).
    robot = make_agent("bot", can_open_doors=False, risk_certified=True)
    engine, _, actual = _engine_and_world(onto, robot, dict(
        CLEAR, door_state="closed", crowd_density=0.6, illumination=0.2,
    ))
    assert actual.nodes == ["s", "via", "g"]
    cf = engine.explain_why_not(actual, ["s", "g"], robot)
    assert {c.property_name for c in cf.changes} == {"door_state"}
    assert math.isfinite(cf.cf_cost) and cf.cf_cost < actual.total_cost


def test_minimal_excludes_unnecessary_soft_fix(onto, make_agent):
    # Three independent TRAVERSABLE blockers (hazard, restricted, closed door)
    # are ALL required; the dark-illumination soft fix is NOT — so it must be
    # excluded from the minimal set.
    robot = make_agent("bot", can_open_doors=False)
    engine, _, actual = _engine_and_world(onto, robot, dict(
        CLEAR, is_hazardous=True, restricted=True,
        door_state="closed", illumination=0.2,
    ))
    cf = engine.explain_why_not(actual, ["s", "g"], robot)
    props = {c.property_name for c in cf.changes}
    assert props == {"is_hazardous", "restricted", "door_state"}
    assert "illumination" not in props


def test_large_pool_fallback_handles_jointly_required(onto, make_agent):
    # > cf_max_pool (12) independently-blocking edges that must ALL be fixed
    # jointly. The old forward-greedy fallback returned changes=[] / cf_cost=inf
    # here (no single fix improves an all-infeasible path); the backward-
    # elimination fallback finds the full jointly-required set.
    robot = make_agent("bot")
    g = NavigationGraph()
    g.add_node(NavNode("n0", (0.0, 0.0)))
    seq = ["n0"]
    N = 13                                          # 13 hazardous edges → pool 13 > 12
    for i in range(N):
        nxt = f"n{i + 1}"
        g.add_node(NavNode(nxt, (float(i + 1), 0.0)))
        seg = onto.create(f"seg{i}", "Corridor", **dict(CLEAR, is_hazardous=True))
        g.add_edge(NavEdge(seq[-1], nxt, seg, distance=1.0), bidirectional=False)
        seq.append(nxt)
    detour = onto.create("detour", "Corridor", **CLEAR)
    g.add_edge(NavEdge("n0", seq[-1], detour, distance=100.0), bidirectional=False)

    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    actual = planner.find_path("n0", seq[-1], robot)
    assert actual.nodes == ["n0", seq[-1]]          # hazardous chain blocked → detour

    cf = engine.explain_why_not(actual, seq, robot)
    assert len(cf.changes) == N                     # all hazards cleared (NOT empty)
    assert all(c.property_name == "is_hazardous" for c in cf.changes)
    assert math.isfinite(cf.cf_cost) and cf.cf_cost < actual.total_cost
    assert cf.is_achievable is True                 # clearing hazards is actionable
    # Independent end-to-end check that the reported set really flips the decision.
    cost = engine._evaluate_subset(cf.alternative_path, robot, cf.changes)
    assert math.isfinite(cost) and cost <= actual.total_cost


def _two_hop_world(onto, robot, seg0_props, seg1_props):
    """s --seg0-- n1 --seg1-- g, plus a clear s --detour-- g (distance 5)."""
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("n1", (1.0, 0.0)), ("g", (2.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    seg0 = onto.create("seg0", "Corridor", **dict(CLEAR, **seg0_props))
    seg1 = onto.create("seg1", "Corridor", **dict(CLEAR, **seg1_props))
    detour = onto.create("detour", "Corridor", **CLEAR)
    g.add_edge(NavEdge("s", "n1", seg0, distance=1.0), bidirectional=False)
    g.add_edge(NavEdge("n1", "g", seg1, distance=1.0), bidirectional=False)
    g.add_edge(NavEdge("s", "g", detour, distance=5.0), bidirectional=False)
    r = AffordanceReasoner()
    planner = AStarPlanner(g, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    return engine, planner.find_path("s", "g", robot)


def test_minimal_sees_blockers_on_every_edge(onto, make_agent):
    # Regression for the truncation bug: a 2-edge alternative whose FIRST edge is
    # hazardous and SECOND has a closed door. evaluate_sequence must expose BOTH
    # edges so the minimal set fixes both — the old code truncated at the first
    # impassable edge and reported only {is_hazardous} with an understated cost.
    robot = make_agent("bot", can_open_doors=False)
    engine, actual = _two_hop_world(
        onto, robot, {"is_hazardous": True}, {"door_state": "closed"})
    assert actual.nodes == ["s", "g"]            # double-blocked route → detour
    cf = engine.explain_why_not(actual, ["s", "n1", "g"], robot)
    assert {c.property_name for c in cf.changes} == {"is_hazardous", "door_state"}
    assert math.isclose(cf.cf_cost, 2.0)         # BOTH edges counted, not just seg0
    assert cf.cf_cost < actual.total_cost


def test_two_blockers_both_required_and_subset_minimal(onto, make_agent):
    # Direct edge blocked by BOTH a closed door (can't open) and being too
    # narrow.  Both are necessary; removing either leaves it infeasible.
    robot = make_agent("bot", robot_width=1.0, min_clearance=0.15, can_open_doors=False)
    engine, _, actual = _engine_and_world(onto, robot, dict(
        CLEAR, width=0.9, door_state="closed",
    ))
    cf = engine.explain_why_not(actual, ["s", "g"], robot)
    assert {c.property_name for c in cf.changes} == {"door_state", "width"}
    # Subset-minimality: dropping any single change must break the goal.
    for c in cf.changes:
        rest = [x for x in cf.changes if x is not c]
        cost = engine._evaluate_subset(cf.alternative_path, robot, rest)
        assert (not math.isfinite(cost)) or cost > actual.total_cost
