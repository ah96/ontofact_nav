"""SPARQL-driven classification layer — proves the RDF/SPARQL inference genuinely
drives planning, stays correct under counterfactuals, and is memoization-cheap."""

from __future__ import annotations

import math

import pytest

from ontofact_nav import (
    AStarPlanner,
    AffordanceType,
    Classifier,
    DerivationRule,
    NavCostConfig,
    NavEdge,
    NavigationGraph,
    NavNode,
    build_default_rules,
)
from ontofact_nav.affordance import AffordanceReasoner
from ontofact_nav.counterfactual import CounterfactualEngine

CLEAR = dict(width=2.0, height=2.5, surface_type="smooth",
             illumination=1.0, is_accessible=True)


def _diamond(onto, direct_slope):
    """s --direct(Ramp, slope)-- g (dist 10) + clear detour s --via-- g (7.07×2)."""
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("via", (5.0, 5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    direct = onto.create("direct", "Ramp", **dict(CLEAR, slope_angle=direct_slope))
    sv = onto.create("sv", "Corridor", **CLEAR)
    vg = onto.create("vg", "Corridor", **CLEAR)
    g.add_edge(NavEdge("s", "g", direct, distance=10.0), bidirectional=False)
    g.add_edge(NavEdge("s", "via", sv, distance=7.07), bidirectional=False)
    g.add_edge(NavEdge("via", "g", vg, distance=7.07), bidirectional=False)
    return g


# A wheeled robot whose max_slope_angle (20) EXCEEDS the ramp slope (18), so the
# agent-relative slope_within_limit does NOT block — isolating the SPARQL-inferred
# "Steep" category (slope > 15) as the sole blocker.
def _perm_wheeled(make_agent):
    return make_agent("perm", mobility_type="wheeled", max_slope_angle=20.0)


# ---------------------------------------------------------------------------
# (a) Inferred category (numeric SPARQL inference) changes the A* plan
# ---------------------------------------------------------------------------

def test_inferred_steep_reroutes_planner(onto, make_agent):
    r = AffordanceReasoner()
    agent = _perm_wheeled(make_agent)
    # slope 18 > 15 ⇒ SPARQL infers "Steep" ⇒ wheeled blocked ⇒ reroute.
    path = AStarPlanner(_diamond(onto, 18.0), r).find_path("s", "g", agent)
    assert path.nodes == ["s", "via", "g"]
    assert AffordanceType.CLIMBABLE not in set(
        r.compute(onto.individual("direct"), agent).affordances)


def test_not_steep_keeps_direct_path(onto, make_agent):
    r = AffordanceReasoner()
    agent = _perm_wheeled(make_agent)
    # slope 10 ≤ 15 ⇒ not "Steep" ⇒ direct ramp is fine and cheaper.
    path = AStarPlanner(_diamond(onto, 10.0), r).find_path("s", "g", agent)
    assert path.nodes == ["s", "g"]


# ---------------------------------------------------------------------------
# (b) Counterfactual that changes the property flips the inference + route
# ---------------------------------------------------------------------------

def test_counterfactual_lowers_slope_below_steep_threshold(onto, make_agent):
    r = AffordanceReasoner()
    agent = _perm_wheeled(make_agent)
    graph = _diamond(onto, 18.0)
    planner = AStarPlanner(graph, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    actual = planner.find_path("s", "g", agent)            # detour
    cf = engine.explain_why_not(actual, ["s", "g"], agent)
    # The minimal fix reduces slope below the Steep threshold; the clone, classified
    # from its *current* view, is no longer Steep ⇒ the direct route becomes feasible.
    assert "slope_angle" in {c.property_name for c in cf.changes}
    assert math.isfinite(cf.cf_cost) and cf.cf_cost < actual.total_cost


# ---------------------------------------------------------------------------
# (c) sparql_select over the classified live graph shows the derived type
# ---------------------------------------------------------------------------

def test_classify_materialises_derived_type_for_sparql(onto):
    onto.create("steep1", "Ramp", slope_angle=20.0)
    onto.create("flat1", "Corridor", slope_angle=2.0)
    added = onto.classify(build_default_rules(NavCostConfig()))
    assert added >= 1
    rows = [str(r.s) for r in onto.sparql_select("SELECT ?s WHERE { ?s a nav:Steep }")]
    assert any(s.endswith("steep1") for s in rows)
    assert not any(s.endswith("flat1") for s in rows)


# ---------------------------------------------------------------------------
# (d) Disabling the derivation rules changes planning (proves load-bearing)
# ---------------------------------------------------------------------------

def test_clearing_rules_makes_steep_traversable(onto, make_agent):
    r = AffordanceReasoner()
    agent = _perm_wheeled(make_agent)
    ramp = onto.create("ramp", "Ramp", **dict(CLEAR, slope_angle=18.0))
    assert r.navigation_cost(ramp, agent, 10.0)[0] == math.inf   # Steep blocks
    r.classifier.set_rules(())                                   # delete derivations
    assert math.isfinite(r.navigation_cost(ramp, agent, 10.0)[0])  # gate gone ⇒ passable


# ---------------------------------------------------------------------------
# (e) Memoization: pure function, bounded distinct evaluations
# ---------------------------------------------------------------------------

def test_memoization_is_pure_and_bounded():
    clf = Classifier(NavCostConfig())
    clf._evals = 0
    first = clf.derived_categories({"slope_angle": 18.0}, {})
    for _ in range(100):
        assert clf.derived_categories({"slope_angle": 18.0}, {}) == first   # determinism
    assert clf._evals == 1                                  # one real SPARQL run
    clf.derived_categories({"slope_angle": 5.0}, {})        # different input → new eval
    assert clf._evals == 2


def test_navigate_collapses_calls_to_few_evals(onto, make_agent):
    # A counterfactual-heavy navigate must not run SPARQL per hot-path call.
    r = AffordanceReasoner()
    agent = _perm_wheeled(make_agent)
    graph = _diamond(onto, 18.0)
    planner = AStarPlanner(graph, r)
    engine = CounterfactualEngine(r, planner, onto.individuals)
    r.classifier.clear_cache(); r.classifier._evals = 0
    actual = planner.find_path("s", "g", agent)
    engine.explain_why_not(actual, ["s", "g"], agent)
    assert r.classifier._evals <= 12        # bounded distinct evaluations, not 1000s


# ---------------------------------------------------------------------------
# (f) Combined-condition category HighRiskZone (a conjunction)
# ---------------------------------------------------------------------------

def test_high_risk_zone_blocks_uncertified_only(onto, make_space, make_agent):
    r = AffordanceReasoner()
    risky = make_space(crowd_density=0.6, illumination=0.2)     # crowded AND dark
    uncertified = make_agent("u")                              # risk_certified absent ⇒ False
    certified = make_agent("c", risk_certified=True)
    assert r.navigation_cost(risky, uncertified, 10.0)[0] == math.inf
    assert math.isfinite(r.navigation_cost(risky, certified, 10.0)[0])


@pytest.mark.parametrize("crowd, illum", [(0.6, 1.0), (0.1, 0.2)])
def test_neither_disjunct_alone_is_a_risk_zone(onto, make_space, make_agent, crowd, illum):
    # Crowded-but-lit and dark-but-empty are NOT HighRiskZones — proves the AND.
    r = AffordanceReasoner()
    space = make_space(crowd_density=crowd, illumination=illum)
    assert math.isfinite(r.navigation_cost(space, make_agent("u"), 10.0)[0])


# ---------------------------------------------------------------------------
# (g) The layer CAN do agent-relative (joint) classification — we chose not to
#     migrate the slope limit, but the mechanism supports it.
# ---------------------------------------------------------------------------

def test_classify_does_not_type_the_class_node(onto):
    # The Staircase CLASS carries a mirrored default slope_angle=30 triple; classify()
    # must type only INDIVIDUALS, not the class node itself, as nav:Steep.
    added = onto.classify(build_default_rules(NavCostConfig()))   # no individuals yet
    steep = [str(r.s) for r in onto.sparql_select("SELECT ?s WHERE { ?s a nav:Steep }")]
    assert steep == []          # the Staircase class node is NOT typed Steep
    assert added == 0


def test_classifier_rejects_undeclared_filter_property():
    # A rule whose SPARQL reads a property it does not declare would yield a stale/
    # incomplete cache key — the classifier must reject it at construction.
    bad = DerivationRule(
        category="Bad", source_props=("slope_angle",),
        sparql_ask=("ASK { ?s nav:slope_angle ?sl ; nav:illumination ?i . "
                    "FILTER(?sl > 1 && ?i < 1) }"),
    )
    with pytest.raises(ValueError):
        Classifier(NavCostConfig(), rules=(bad,))


def test_cache_key_distinguishes_bool_from_int():
    # True == 1 and hash-collide in Python, but classify differently in SPARQL.
    rule = DerivationRule(category="Flag", source_props=("v",),
                          sparql_ask="ASK { ?s nav:v ?v . FILTER(?v > 0.5) }")
    clf = Classifier(NavCostConfig(), rules=(rule,))
    assert clf.derived_categories({"v": True}, {}) == frozenset()          # bool: not > 0.5
    assert clf.derived_categories({"v": 1}, {}) == frozenset({"Flag"})     # int: NOT served stale


def test_joint_rule_does_agent_relative_classification():
    joint = DerivationRule(
        category="TooSteepForAgent",
        source_props=("slope_angle",), agent_props=("max_slope_angle",), scope="joint",
        sparql_ask=("ASK { ?s nav:slope_angle ?sl . ?a nav:max_slope_angle ?m . "
                    "FILTER(?sl > ?m) }"),
    )
    clf = Classifier(NavCostConfig(), rules=(joint,))
    assert clf.derived_categories({"slope_angle": 18.0}, {"max_slope_angle": 8.0}) == frozenset({"TooSteepForAgent"})
    assert clf.derived_categories({"slope_angle": 18.0}, {"max_slope_angle": 25.0}) == frozenset()
