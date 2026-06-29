"""Issue 3 — the ontology is load-bearing: the class hierarchy and class-level
defaults genuinely change affordance/planning outcomes, and live mutations sync
to the RDF graph (while clones stay isolated)."""

from __future__ import annotations

import math

from ontofact_nav import (
    AffordanceType,
    AStarPlanner,
    NavEdge,
    NavigationGraph,
    NavNode,
)
from ontofact_nav.affordance import AffordanceReasoner

# Identical EXPLICIT properties; the only difference between worlds is the class.
SAME = dict(width=2.0, height=2.5, surface_type="smooth",
            illumination=1.0, is_accessible=True)


# ---------------------------------------------------------------------------
# Class membership changes the planning outcome
# ---------------------------------------------------------------------------

def test_changing_class_alone_changes_planning(onto, make_agent):
    r = AffordanceReasoner()
    agent = make_agent("w", mobility_type="wheeled", max_slope_angle=8.0)
    corridor  = onto.create("as_corridor",  "Corridor",  **SAME)
    staircase = onto.create("as_staircase", "Staircase", **SAME)   # no slope_angle
    # Same explicit properties, different class → different feasibility.
    assert math.isfinite(r.navigation_cost(corridor, agent, 10.0)[0])
    assert r.navigation_cost(staircase, agent, 10.0)[0] == math.inf


def _diamond(onto, direct_cls):
    """s --10-- g direct (class=direct_cls); s --via-- g detour (Corridors)."""
    g = NavigationGraph()
    for nid, pos in [("s", (0.0, 0.0)), ("via", (5.0, 5.0)), ("g", (10.0, 0.0))]:
        g.add_node(NavNode(nid, pos))
    direct = onto.create(f"direct_{direct_cls}", direct_cls, **SAME)
    sv     = onto.create(f"sv_{direct_cls}",     "Corridor", **SAME)
    vg     = onto.create(f"vg_{direct_cls}",     "Corridor", **SAME)
    g.add_edge(NavEdge("s", "g",   direct, distance=10.0))
    g.add_edge(NavEdge("s", "via", sv,     distance=7.07))
    g.add_edge(NavEdge("via", "g", vg,     distance=7.07))
    return g


def test_class_change_reroutes_planner(onto, make_agent):
    r = AffordanceReasoner()
    agent = make_agent("w", mobility_type="wheeled", max_slope_angle=8.0)
    # Direct corridor (cost 10) beats the 14.14 detour.
    p1 = AStarPlanner(_diamond(onto, "Corridor"), r).find_path("s", "g", agent)
    assert p1.nodes == ["s", "g"]
    # Same geometry but the direct passage is a Staircase → wheeled robot reroutes.
    p2 = AStarPlanner(_diamond(onto, "Staircase"), r).find_path("s", "g", agent)
    assert p2.nodes == ["s", "via", "g"]


# ---------------------------------------------------------------------------
# Class-level inherited defaults
# ---------------------------------------------------------------------------

def test_class_default_inherited(onto):
    st = onto.create("st1", "Staircase")        # no explicit slope/surface
    assert st.get("slope_angle") == 30.0
    assert st.get("surface_type") == "rough"


def test_explicit_property_overrides_class_default(onto):
    st = onto.create("st2", "Staircase", slope_angle=2.0)
    assert st.get("slope_angle") == 2.0          # instance wins over class default


def test_corridor_has_no_inherited_slope(onto):
    c = onto.create("c1", "Corridor")
    assert c.get("slope_angle", 0.0) == 0.0      # no Staircase default leaks in


# ---------------------------------------------------------------------------
# Class-conditioned rule (independent of numeric slope)
# ---------------------------------------------------------------------------

def test_class_conditioned_rule_blocks_staircase_for_wheeled(onto, make_agent):
    r = AffordanceReasoner()
    # Explicit FLAT slope, so the numeric slope rules would grant CLIMBABLE —
    # but the class-conditioned rule still blocks it for a wheeled robot.
    flat_stair = onto.create("flat_stair", "Staircase", slope_angle=0.0, **SAME)
    wheeled = make_agent("w", mobility_type="wheeled")
    legged  = make_agent("l", mobility_type="legged", max_slope_angle=40.0)
    assert r.compute(flat_stair, wheeled).missing(AffordanceType.CLIMBABLE)
    assert r.compute(flat_stair, legged).has(AffordanceType.CLIMBABLE)


# ---------------------------------------------------------------------------
# RDF graph reflects class defaults + live mutations (clones stay isolated)
# ---------------------------------------------------------------------------

def test_class_default_mirrored_to_rdf(onto):
    rows = list(onto.sparql_select(
        "SELECT ?s WHERE { nav:Staircase nav:slope_angle ?s }"
    ))
    assert any(float(r.s) == 30.0 for r in rows)


def test_set_syncs_live_individual_to_rdf(onto):
    onto.create("door1", "Doorway", door_state="closed", width=1.0)
    onto.individual("door1").set("door_state", "open")
    rows = list(onto.sparql_select("SELECT ?d WHERE { nav:door1 nav:door_state ?d }"))
    assert [str(r.d) for r in rows] == ["open"]


def test_clone_mutation_does_not_touch_rdf(onto):
    ind = onto.create("door2", "Doorway", door_state="closed")
    clone = ind.clone()
    clone.set("door_state", "open")              # clone is graph-less
    rows = list(onto.sparql_select("SELECT ?d WHERE { nav:door2 nav:door_state ?d }"))
    assert [str(r.d) for r in rows] == ["closed"]   # live graph untouched
