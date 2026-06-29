"""
ontofact_nav/scenarios/warehouse.py
=====================================
Warehouse navigation scenario.

Purpose
-------
Demonstrates affordance rules and counterfactuals in an industrial environment
with mobility-type differentiation:
  - Wet floor: high friction cost (SurfaceType.WET)
  - Dark inspection zone: OBSERVABLE blocked → navigation uncertainty penalty
  - Steep ramp (18°): CLIMBABLE blocked for wheeled robots (max 8°)
  - Forklift: wide body restricts it to main-aisle routes only
  - Tracked robot: high slope tolerance allows it to use the 18° ramp

Environment layout (top-down, not to scale)
--------------------------------------------
[loading_bay] ── [main_aisle] ── [cross_aisle] ── [storage_A]
                      │               │
                 [narrow_aisle]  [ramp_zone] ── [mezzanine]
                      │
                 [wet_floor_zone]
                      │
                 [inspection_zone]  ← dark (illumination=0.15)

Key constraints:
  seg_main_narrow    : width=0.85 m  → too narrow for forklift (1.4 m body)
  seg_wet_inspect    : illumination=0.15 → OBSERVABLE blocked (+1.8 cost)
  seg_cross_ramp     : slope_angle=18° → CLIMBABLE blocked for wheeled (max 8°)
                       but accessible for tracked (max 25°)

Robots
------
  picker_bot   : 0.6 m wide, wheeled, no arm  — standard warehouse picker
  forklift_bot : 1.4 m wide, wheeled, no arm  — heavy-load transporter
  tracked_bot  : 0.8 m wide, tracked, no arm  — rough-terrain workhorse

Scenarios
---------
  1. picker_bot  → storage_A  (direct main aisle — no special constraints)
  2. tracked_bot → mezzanine  (via steep ramp — only tracked can climb 18°)
  3. "Why can't picker_bot use the ramp?" (slope 18° > 8° limit; counterfactual)
  4. forklift_bot → storage_A (same route as picker_bot — wide corridors OK)
"""

from __future__ import annotations

import math

from ontofact_nav import (
    OntofactNavigator,
    build_navigation_ontology,
    NavigationGraph,
    NavNode,
    NavEdge,
    DoorState,
    MobilityType,
    SurfaceType,
)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------

def build_warehouse_world():
    """
    Construct the warehouse ontology, navigation graph, and robot agents.

    Returns
    -------
    (onto, graph, agents_dict)
    """
    onto  = build_navigation_ontology()
    graph = NavigationGraph()

    # ── Waypoint nodes ──────────────────────────────────────────────────────
    # Y-axis goes downward (loading bay at 0,0; inspection zone at bottom).
    node_data = [
        #  (node_id,           (x,    y),   label)
        ("loading_bay",      ( 0.0,  0.0), "Loading Bay"),
        ("main_aisle",       (10.0,  0.0), "Main Aisle Junction"),
        ("cross_aisle",      (20.0,  0.0), "Cross Aisle"),
        ("storage_A",        (30.0,  0.0), "Storage Zone A"),
        ("narrow_aisle",     (10.0, -8.0), "Narrow Aisle"),
        ("wet_floor_zone",   (10.0,-16.0), "Wet Floor Zone"),
        ("inspection_zone",  (10.0,-24.0), "Dark Inspection Zone"),
        ("ramp_zone",        (20.0, -8.0), "Ramp to Mezzanine"),
        ("mezzanine",        (20.0,-16.0), "Mezzanine Level"),
    ]
    for nid, pos, label in node_data:
        graph.add_node(NavNode(node_id=nid, position=pos, label=label))

    # ── Space individuals ────────────────────────────────────────────────────

    def space(name, class_name, **props):
        """Shorthand factory for space individuals."""
        return onto.create(name, class_name, **props)

    def dist(a, b):
        """Euclidean distance between two graph nodes (metres)."""
        na, nb = graph.nodes[a], graph.nodes[b]
        return math.hypot(
            na.position[0] - nb.position[0],
            na.position[1] - nb.position[1],
        )

    # loading_bay → main_aisle
    # Designated emergency route (discount applied).  Wide enough for all robots.
    bay_main = space(
        "seg_bay_main", "Corridor",
        width=4.0,  height=5.0,  length=10.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.85,
        obstacle_density=0.05,
        crowd_density=0.1,
        is_accessible=True,  is_hazardous=False,  restricted=False,
        emergency_route=True,   # ← preferred entry/exit route
    )

    # main_aisle → cross_aisle
    # Spacious main corridor — wide enough for forklift.
    main_cross = space(
        "seg_main_cross", "Corridor",
        width=3.5,  height=4.5,  length=10.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.88,
        obstacle_density=0.05,
        crowd_density=0.05,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # cross_aisle → storage_A
    cross_storA = space(
        "seg_cross_storA", "Corridor",
        width=3.0,  height=4.0,  length=10.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.80,
        obstacle_density=0.1,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # main_aisle → narrow_aisle  ← NARROW — blocks forklift_bot
    # Width=0.85 m is insufficient for forklift (1.4 + 2×0.25 = 1.9 m required).
    # picker_bot (0.6 + 2×0.10 = 0.8 m required) can just barely fit.
    # ROUGH surface adds friction cost of 0.6.
    main_narrow = space(
        "seg_main_narrow", "Corridor",
        width=0.85,  height=3.0,  length=8.0,  # tight for most robots
        surface_type=SurfaceType.ROUGH.value,   # cost adder: +0.6
        illumination=0.75,
        obstacle_density=0.1,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # narrow_aisle → wet_floor_zone  ← WET FLOOR — high friction cost
    # Surface cost adder = 1.4 (highest in the table).
    # Passable but slow — robot must reduce speed on wet surface.
    narrow_wet = space(
        "seg_narrow_wet", "Corridor",
        width=2.5,  height=3.0,  length=8.0,
        surface_type=SurfaceType.WET.value,     # cost adder: +1.4
        illumination=0.70,
        obstacle_density=0.0,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # wet_floor_zone → inspection_zone  ← DARK — OBSERVABLE blocked
    # illumination=0.15 < 0.25 threshold → OBSERVABLE rule fires → cost += 1.8
    # The robot can still navigate (TRAVERSABLE/PASSABLE not affected) but must
    # move more cautiously due to sensor uncertainty.
    wet_inspect = space(
        "seg_wet_inspect", "Corridor",
        width=2.5,  height=3.0,  length=8.0,
        surface_type=SurfaceType.ROUGH.value,
        illumination=0.15,   # ← below 0.25 threshold → OBSERVABLE blocked
        obstacle_density=0.0,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # cross_aisle → ramp_zone  ← STEEP RAMP — CLIMBABLE blocked for wheeled
    # slope_angle=18° > 8° max for wheeled robots → CLIMBABLE blocked → cost=inf
    # tracked_bot has max_slope_angle=25° → 18° ≤ 25° → CLIMBABLE granted
    # The block_climbable_wheeled_on_stairs rule also fires (18° > 15° threshold)
    # as an additional safety guard for wheeled robots.
    cross_ramp = space(
        "seg_cross_ramp", "Ramp",
        width=2.5,  height=4.0,  length=8.5,
        slope_angle=18.0,          # ← steep: blocked for wheeled, fine for tracked
        surface_type=SurfaceType.ROUGH.value,
        illumination=0.80,
        obstacle_density=0.0,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # ramp_zone → mezzanine  (landing platform — flat and wide)
    ramp_mezz = space(
        "seg_ramp_mezz", "Corridor",
        width=3.0,  height=3.0,  length=1.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.90,
        obstacle_density=0.0,
        crowd_density=0.0,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # ── Graph edges ──────────────────────────────────────────────────────────
    edges = [
        #  (from_node,       to_node,            space_individual, distance)
        ("loading_bay",   "main_aisle",       bay_main,    dist("loading_bay",   "main_aisle")),
        ("main_aisle",    "cross_aisle",      main_cross,  dist("main_aisle",    "cross_aisle")),
        ("cross_aisle",   "storage_A",        cross_storA, dist("cross_aisle",   "storage_A")),
        ("main_aisle",    "narrow_aisle",     main_narrow, dist("main_aisle",    "narrow_aisle")),
        ("narrow_aisle",  "wet_floor_zone",   narrow_wet,  dist("narrow_aisle",  "wet_floor_zone")),
        ("wet_floor_zone","inspection_zone",  wet_inspect, dist("wet_floor_zone","inspection_zone")),
        ("cross_aisle",   "ramp_zone",        cross_ramp,  dist("cross_aisle",   "ramp_zone")),
        ("ramp_zone",     "mezzanine",        ramp_mezz,   dist("ramp_zone",     "mezzanine")),
    ]
    for from_id, to_id, individual, distance in edges:
        graph.add_edge(
            NavEdge(from_id=from_id, to_id=to_id,
                    space_individual=individual, distance=distance),
            bidirectional=True,
        )

    # ── Robot agents ─────────────────────────────────────────────────────────

    # Standard warehouse picker — narrow, can fit in most places
    picker_bot = onto.create(
        "picker_bot", "Robot",
        robot_width    = 0.6,
        robot_height   = 1.8,
        min_clearance  = 0.10,   # tight clearance for narrow aisles
        max_slope_angle= 8.0,    # ← wheeled limit; cannot climb 18° ramp
        mobility_type  = MobilityType.WHEELED.value,
        can_open_doors = False,
        has_arm        = True,   # for picking tasks (not used in navigation here)
        battery_level  = 0.90,
        max_speed      = 1.5,
    )

    # Heavy-load forklift — very wide body, restricted to main aisles
    forklift_bot = onto.create(
        "forklift_bot", "Robot",
        robot_width    = 1.4,    # requires 1.4 + 2×0.25 = 1.9 m clearance
        robot_height   = 3.5,    # tall — limited by warehouse ceiling height
        min_clearance  = 0.25,
        max_slope_angle= 5.0,
        mobility_type  = MobilityType.WHEELED.value,
        can_open_doors = False,
        has_arm        = False,
        battery_level  = 0.95,
        max_speed      = 0.6,    # heavy load → slow
    )

    # Tracked rough-terrain robot — can handle 18° ramp, moderately wide
    tracked_bot = onto.create(
        "tracked_bot", "Robot",
        robot_width    = 0.8,
        robot_height   = 1.5,
        min_clearance  = 0.15,
        max_slope_angle= 25.0,   # ← high tolerance; 18° ramp is fine
        mobility_type  = MobilityType.TRACKED.value,
        can_open_doors = False,
        has_arm        = False,
        battery_level  = 0.80,
        max_speed      = 0.8,
    )

    return onto, graph, {
        "picker_bot":   picker_bot,
        "forklift_bot": forklift_bot,
        "tracked_bot":  tracked_bot,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run() -> None:
    """Run all four warehouse navigation scenarios."""
    onto, graph, agents = build_warehouse_world()

    onto.print_schema()

    nav = OntofactNavigator(onto, graph)
    DIVIDER = "\n" + "▓" * 72

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 1 — picker_bot → storage_A
    #
    # Expected: loading_bay → main_aisle → cross_aisle → storage_A
    # Straight main-aisle route; no special constraints for picker_bot.
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  [WAREHOUSE] picker_bot: loading_bay → storage_A")
    print(DIVIDER)
    _, exp = nav.navigate(
        "loading_bay", "storage_A",
        agents["picker_bot"],
        k_alternatives=3,
    )
    if exp:
        nav.print_report(exp)

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 2 — tracked_bot → mezzanine
    #
    # Expected: loading_bay → main_aisle → cross_aisle → ramp_zone → mezzanine
    # tracked_bot has max_slope_angle=25° → CLIMBABLE granted for 18° ramp.
    # picker_bot / forklift_bot would fail here (max 8° / 5° respectively).
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  [WAREHOUSE] tracked_bot: loading_bay → mezzanine")
    print(DIVIDER)
    _, exp2 = nav.navigate(
        "loading_bay", "mezzanine",
        agents["tracked_bot"],
        k_alternatives=3,
    )
    if exp2:
        nav.print_report(exp2)

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 3 — "Why can't picker_bot use the ramp?"
    #
    # Expected: slope_angle=18° > max_slope_angle=8° for wheeled robot
    # Counterfactual: reducing slope to 7° (installing a shallower ramp)
    # would make the route feasible at cost 27.24.
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  [WAREHOUSE] 'Why can't picker_bot use the ramp to mezzanine?'")
    print(DIVIDER)
    cf = nav.query_why_not(
        "loading_bay", "mezzanine",
        agents["picker_bot"],
        # Explicit node sequence: straight route via the ramp
        alt_nodes=["loading_bay", "main_aisle", "cross_aisle",
                   "ramp_zone", "mezzanine"],
    )
    print(f"\n  {cf.explanation}\n")
    for ch in cf.changes:
        tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
        print(f"  • [{tag}] {ch.description()}")
        print(f"    {ch.rationale}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 4 — forklift_bot → storage_A
    #
    # Expected: same main-aisle route as picker_bot (loading_bay → … → storage_A)
    # Forklift body (1.4 m) fits in all main corridors (3.0–4.0 m wide).
    # Would be blocked from narrow_aisle (0.85 m) but that route is not optimal.
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  [WAREHOUSE] forklift_bot: loading_bay → storage_A")
    print(DIVIDER)
    path4, exp4 = nav.navigate(
        "loading_bay", "storage_A",
        agents["forklift_bot"],
        k_alternatives=3,
    )
    if exp4:
        nav.print_report(exp4)
    else:
        print(f"  Path: {path4.summary()}")


if __name__ == "__main__":
    run()
