"""
ontofact_nav/scenarios/hospital.py
===================================
Hospital floor-plan navigation scenario.

Purpose
-------
Demonstrates the full framework pipeline in a realistic setting:
  - Multi-robot differentiation (arm vs. no-arm, wide vs. narrow body)
  - Door-opening affordance (delivery_bot can open ICU door; cargo_bot cannot)
  - Width-based affordance blocking (narrow corridor blocks cargo_bot)
  - Access restriction (staff corridor blocks all robots without permission)
  - Live world mutation (simulate propping the ICU door open)
  - Per-robot improvement roadmap (what infrastructure changes unlock routes)

Environment layout (top-down, not to scale)
--------------------------------------------
                    [elev_lobby] ──── [floor2]
                         │
[entrance] ── [lobby] ── [corridor_a] ── [icu_entrance] ── [icu_main]
                                │
                           [corridor_b] ── [staff_corridor]
                                               │
                                          (→ icu_main via staff door)

Key physical constraints:
  • seg_corrA_ICUent   : width=0.9 m  (passable for 0.6 m robot; NOT for 1.1 m cargo_bot)
  • seg_ICUent_ICUmain : door=CLOSED  (openable only if robot has can_open_doors=True)
  • seg_staff_corridor : restricted=True (inaccessible to non-authorised robots)

Robots
------
  delivery_bot : 0.6 m wide, can_open_doors=True, has_arm=True  → full access
  cargo_bot    : 1.1 m wide, can_open_doors=False                → blocked by width + door
  legged_bot   : 0.65 m wide, can_open_doors=False               → blocked by door only

Scenarios
---------
  1. delivery_bot: entrance → icu_main  (standard navigatin with door opening)
  2. cargo_bot   : entrance → icu_main  (infeasible; per-route counterfactual analysis)
  3. legged_bot  : entrance → icu_main  (infeasible; single door blocker)
  4. Ad-hoc query: "Why not use the staff corridor?"
  5. World mutation: open ICU door → cost drops from 35.04 → 32.54
  6. Improvement roadmap for cargo_bot
"""

from __future__ import annotations

import sys
import os

# Allow running this script directly (python3 scenarios/hospital.py)
# by adding the project root to the Python path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

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

def build_hospital_world():
    """
    Construct the hospital ontology, navigation graph, and robot agents.

    Returns
    -------
    (onto, graph, agents_dict)
      onto        : Ontology populated with space individuals
      graph       : NavigationGraph with nodes and bidirectional edges
      agents_dict : {name → OntologyIndividual} for the three robot types
    """
    onto  = build_navigation_ontology()
    graph = NavigationGraph()

    # ── Waypoint nodes ──────────────────────────────────────────────────────
    # Each node represents a location in the hospital.  The (x, y) position
    # is used only by the Euclidean heuristic in A* — it does not need to
    # correspond to a real-world coordinate system, just preserve relative
    # distances.
    node_data = [
        #  (node_id,          (x,    y),   label)
        ("entrance",       ( 0.0,  0.0), "Main Entrance"),
        ("lobby",          ( 8.0,  0.0), "Main Lobby"),
        ("corridor_a",     (16.0,  0.0), "Corridor A Junction"),
        ("icu_entrance",   (24.0,  0.0), "ICU Entrance"),
        ("icu_main",       (32.0,  0.0), "ICU Main"),
        ("corridor_b",     (16.0,  8.0), "Corridor B Junction"),
        ("staff_corridor", (24.0,  8.0), "Staff Corridor"),
        ("elev_lobby",     ( 8.0,  8.0), "Elevator Lobby"),
        ("floor2",         ( 8.0, 16.0), "Floor 2 (elevator access)"),
    ]
    for nid, pos, label in node_data:
        graph.add_node(NavNode(node_id=nid, position=pos, label=label))

    # ── Space individuals (passage properties) ──────────────────────────────
    # Each call to onto.create() registers a space individual that describes
    # the physical characteristics of the passage traversed along one edge.
    # Properties directly drive the affordance rules in affordance.py.

    def space(name, class_name, **props):
        """Shorthand factory for space individuals."""
        return onto.create(name, class_name, **props)

    # entrance → lobby
    # Designated emergency route: gets a 0.75× cost multiplier (preferred path).
    # Wide, well-lit, lightly crowded — no constraints for any robot.
    ent_lob = space(
        "seg_entrance_lobby", "Corridor",
        width=3.5,  height=3.0,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.95,
        obstacle_density=0.05,
        crowd_density=0.2,
        is_accessible=True,  is_hazardous=False,  restricted=False,
        emergency_route=True,   # ← cost discount applied here
    )

    # lobby → corridor_a
    # Standard hospital corridor.  Moderate crowd due to patient traffic.
    lob_cora = space(
        "seg_lobby_corrA", "Corridor",
        width=2.8,  height=2.8,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.90,
        obstacle_density=0.10,
        crowd_density=0.3,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # corridor_a → icu_entrance  ← NARROW — key constraint for cargo_bot
    # Width=0.9 m is exactly the minimum for delivery_bot (0.6 + 2×0.15 = 0.9).
    # cargo_bot requires 1.1 + 2×0.20 = 1.5 m → PASSABLE blocked → cost = inf.
    cora_icuent = space(
        "seg_corrA_ICUent", "Corridor",
        width=0.9,  height=2.6,  length=8.0,  # 0.9 m width — tight squeeze
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.85,
        obstacle_density=0.0,
        crowd_density=0.1,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # icu_entrance → icu_main  ← CLOSED DOOR — key constraint for no-arm robots
    # door_state=CLOSED means:
    #   • delivery_bot (can_open_doors=True) → OPENABLE granted → cost += 2.5
    #   • cargo_bot / legged_bot (can_open_doors=False) → OPENABLE blocked → cost = inf
    icuent_icum = space(
        "seg_ICUent_ICUmain", "Doorway",
        width=2.0,  height=2.5,  length=2.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.95,
        obstacle_density=0.0,
        crowd_density=0.0,
        door_state=DoorState.CLOSED.value,   # ← security door, not locked
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # corridor_a ↔ corridor_b
    # Cross-corridor connecting the two main routes.  Bidirectional.
    cora_corb = space(
        "seg_corrA_corrB", "Corridor",
        width=2.5,  height=2.8,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.88,
        obstacle_density=0.05,
        crowd_density=0.2,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # corridor_b → icu_entrance
    # The longer detour route — 12+ more cost units than the direct path.
    corb_icuent = space(
        "seg_corrB_ICUent", "Corridor",
        width=2.5,  height=2.8,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.90,
        obstacle_density=0.0,
        crowd_density=0.15,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # lobby ↔ elevator lobby
    lob_elev = space(
        "seg_lobby_elevLobby", "Corridor",
        width=2.5,  height=2.8,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.92,
        obstacle_density=0.0,
        crowd_density=0.1,
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # elevator lobby → floor 2  (elevator cabin)
    # Door is OPEN by default (elevator is waiting).  Wide enough for all robots.
    elev_f2 = space(
        "seg_elevator", "Elevator",
        width=2.2,  height=2.5,  length=1.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.95,
        obstacle_density=0.0,
        crowd_density=0.0,
        door_state=DoorState.OPEN.value,  # elevator doors currently open
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # corridor_b → staff_corridor  ← RESTRICTED — blocks all robots by default
    # restricted=True triggers the block_traversable_restricted rule, making
    # this space impassable regardless of robot capabilities.
    # Counterfactual scenario 4 shows what change would unlock this route.
    corb_staff = space(
        "seg_staff_corridor", "Corridor",
        width=2.0,  height=2.8,  length=8.0,
        surface_type=SurfaceType.TILED.value,
        illumination=0.90,
        obstacle_density=0.0,
        crowd_density=0.05,
        is_accessible=True,  is_hazardous=False,
        restricted=True,   # ← staff-only access; blocks all robots
    )

    # staff_corridor → icu_main  (direct staff access — door is open)
    staff_icum = space(
        "seg_staff_ICUmain", "Doorway",
        width=2.0,  height=2.5,  length=2.0,
        surface_type=SurfaceType.SMOOTH.value,
        illumination=0.95,
        obstacle_density=0.0,
        crowd_density=0.0,
        door_state=DoorState.OPEN.value,  # staff-side door is open
        is_accessible=True,  is_hazardous=False,  restricted=False,
    )

    # ── Graph edges ──────────────────────────────────────────────────────────
    # Distance is computed as Euclidean distance between node positions.
    # All edges are bidirectional so the robot can navigate in both directions.

    def dist(a, b):
        """Euclidean distance between two node positions in the graph."""
        na, nb = graph.nodes[a], graph.nodes[b]
        return math.hypot(
            na.position[0] - nb.position[0],
            na.position[1] - nb.position[1],
        )

    edges = [
        #  (from_node,      to_node,          space_individual, distance)
        ("entrance",    "lobby",          ent_lob,     dist("entrance",    "lobby")),
        ("lobby",       "corridor_a",     lob_cora,    dist("lobby",       "corridor_a")),
        ("corridor_a",  "icu_entrance",   cora_icuent, dist("corridor_a",  "icu_entrance")),
        ("icu_entrance","icu_main",        icuent_icum, dist("icu_entrance","icu_main")),
        ("corridor_a",  "corridor_b",     cora_corb,   dist("corridor_a",  "corridor_b")),
        ("corridor_b",  "icu_entrance",   corb_icuent, dist("corridor_b",  "icu_entrance")),
        ("lobby",       "elev_lobby",     lob_elev,    dist("lobby",       "elev_lobby")),
        ("elev_lobby",  "floor2",         elev_f2,     dist("elev_lobby",  "floor2")),
        ("corridor_b",  "staff_corridor", corb_staff,  dist("corridor_b",  "staff_corridor")),
        ("staff_corridor","icu_main",     staff_icum,  dist("staff_corridor","icu_main")),
    ]
    for from_id, to_id, individual, distance in edges:
        graph.add_edge(
            NavEdge(from_id=from_id, to_id=to_id,
                    space_individual=individual, distance=distance),
            bidirectional=True,
        )

    # ── Robot agents ──────────────────────────────────────────────────────────
    # Each robot has different physical capabilities that interact with the
    # affordance rules to produce different navigation behaviours.

    # Standard delivery robot — narrow body, manipulation arm, can open doors
    delivery_bot = onto.create(
        "delivery_bot", "Robot",
        robot_width    = 0.6,   # fits through 0.9 m corridor (with clearance)
        robot_height   = 1.4,
        min_clearance  = 0.15,  # required side buffer; total = 0.6 + 2×0.15 = 0.9 m
        max_slope_angle= 8.0,
        mobility_type  = MobilityType.WHEELED.value,
        can_open_doors = True,  # ← can traverse closed (unlocked) doors
        has_arm        = True,
        battery_level  = 0.85,
        max_speed      = 1.2,
    )

    # Wide cargo robot — no arm, cannot open doors, too wide for narrow corridor
    cargo_bot = onto.create(
        "cargo_bot", "Robot",
        robot_width    = 1.1,   # requires 1.1 + 2×0.20 = 1.5 m → fails 0.9 m corridor
        robot_height   = 1.6,
        min_clearance  = 0.20,
        max_slope_angle= 5.0,
        mobility_type  = MobilityType.WHEELED.value,
        can_open_doors = False,  # ← cannot open ICU door
        has_arm        = False,
        battery_level  = 0.95,
        max_speed      = 0.8,
    )

    # Legged robot — narrow enough to fit, but no arm (blocked by ICU door)
    legged_bot = onto.create(
        "legged_bot", "Robot",
        robot_width    = 0.65,
        robot_height   = 1.2,
        min_clearance  = 0.10,
        max_slope_angle= 30.0,  # high slope tolerance (can climb stairs)
        mobility_type  = MobilityType.LEGGED.value,
        can_open_doors = False,  # ← blocked by same door as cargo_bot
        has_arm        = False,
        battery_level  = 0.72,
        max_speed      = 1.0,
    )

    return onto, graph, {
        "delivery_bot": delivery_bot,
        "cargo_bot":    cargo_bot,
        "legged_bot":   legged_bot,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run() -> None:
    """Run all six hospital navigation scenarios."""
    onto, graph, agents = build_hospital_world()

    # Print the class hierarchy to show the ontology structure
    onto.print_schema()

    nav = OntofactNavigator(onto, graph)

    DIVIDER = "\n" + "█" * 72   # visual separator between scenarios

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 1 — delivery_bot: entrance → icu_main
    #
    # Expected behaviour:
    #   • Takes the direct route via corridor_a (narrow but fits)
    #   • Opens the closed ICU door (has arm) → cost += 2.5
    #   • Emergency route discount applied to entrance→lobby segment
    #   • Counterfactual identifies corridor_b detour as 12.4 cost units costlier
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 1 — delivery_bot: entrance → icu_main")
    print(DIVIDER)

    path, exp = nav.navigate(
        start="entrance",
        goal="icu_main",
        agent=agents["delivery_bot"],
        k_alternatives=3,
    )
    if exp:
        nav.print_report(exp)
    else:
        print(f"  No feasible path found. Path summary: {path.summary()}")

    # Ad-hoc "why not" query: operator asks why the robot didn't take the
    # longer detour route via corridor_b instead of the direct path.
    # Expected answer: "the chosen route is already cheaper — no changes needed."
    print("\n  USER QUERY: 'Why didn't the robot go via the longer"
          " corridor_b detour instead?'\n")
    cf = nav.query_why_not(
        start="entrance",
        goal="icu_main",
        agent=agents["delivery_bot"],
        # Explicit node sequence for the detour path
        alt_nodes=["entrance", "lobby", "corridor_a", "corridor_b",
                   "icu_entrance", "icu_main"],
    )
    print(f"  Counterfactual: {cf.explanation}\n")
    if cf.changes:
        print("  Changes that would unlock this route:")
        for ch in cf.changes:
            tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
            print(f"    • [{tag}] {ch.description()}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 2 — cargo_bot (wide body, no arm): entrance → icu_main
    #
    # Expected behaviour:
    #   • ALL paths to icu_main are infeasible for cargo_bot because:
    #     - Direct route: seg_corrA_ICUent is too narrow (0.9 m < 1.5 m required)
    #     - Detour route: ICU door is CLOSED and cargo_bot cannot open it
    #     - Staff corridor: restricted (and ICU door still blocks the final step)
    #   • Counterfactual analysis identifies the specific blocker on each route
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 2 — cargo_bot (wide body, no arm): entrance → icu_main")
    print(DIVIDER)

    path2, exp2 = nav.navigate(
        start="entrance",
        goal="icu_main",
        agent=agents["cargo_bot"],
        k_alternatives=3,
    )
    if exp2:
        nav.print_report(exp2)
    else:
        print(f"  No feasible path for cargo_bot. Analysing blockers...\n")
        # Explain each candidate route to identify what is blocking it
        for alt in [
            ["entrance", "lobby", "corridor_a", "icu_entrance", "icu_main"],
            ["entrance", "lobby", "corridor_a", "corridor_b", "icu_entrance", "icu_main"],
        ]:
            cf2 = nav.query_why_not("entrance", "icu_main", agents["cargo_bot"], alt)
            print(f"  Route {' → '.join(alt)}:")
            print(f"  {cf2.explanation}\n")
            for ch in cf2.changes:
                tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
                print(f"    • [{tag}] {ch.description()}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 3 — legged_bot: entrance → icu_main
    #
    # Expected behaviour:
    #   • legged_bot fits through the narrow corridor (0.65 m body)
    #   • Blocked only by the ICU door (no arm → cannot open it)
    #   • Counterfactual: opening the door would make route feasible at cost 32.54
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 3 — legged_bot: entrance → icu_main")
    print(DIVIDER)

    path3, exp3 = nav.navigate(
        start="entrance",
        goal="icu_main",
        agent=agents["legged_bot"],
        k_alternatives=3,
    )
    if exp3:
        nav.print_report(exp3)
    else:
        print(f"  [No feasible path found for legged_bot]")
        # Single counterfactual to identify the specific blocker
        cf3 = nav.query_why_not(
            start="entrance",
            goal="icu_main",
            agent=agents["legged_bot"],
            alt_nodes=["entrance", "lobby", "corridor_a", "icu_entrance", "icu_main"],
        )
        print(f"\n  Counterfactual: {cf3.explanation}")
        for ch in cf3.changes:
            print(f"    • {ch.description()}  [{ch.rationale}]")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 4 — Ad-hoc query: "Why not the staff corridor shortcut?"
    #
    # Expected behaviour:
    #   • Staff corridor is restricted → TRAVERSABLE blocked → cost = inf
    #   • Counterfactual: set restricted=False → route costs 33.27 (saves 1.78)
    #   • Change is actionable (medium effort — admin/policy change)
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 4 — 'Why not use the staff corridor shortcut?'")
    print(DIVIDER)

    cf_staff = nav.query_why_not(
        start="entrance",
        goal="icu_main",
        agent=agents["delivery_bot"],
        # Route via the staff corridor (shorter but restricted)
        alt_nodes=[
            "entrance", "lobby", "corridor_a", "corridor_b",
            "staff_corridor", "icu_main",
        ],
    )
    print(f"\n  Explanation:\n  {cf_staff.explanation}\n")
    print("  Required changes to unlock staff-corridor route:")
    for ch in cf_staff.changes:
        tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
        print(f"    • [{tag}] {ch.description()}")
        print(f"      Rationale : {ch.rationale}")
        print(f"      Effort    : {ch.effort()}")

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 5 — Live world simulation: prop the ICU door open
    #
    # Simulates an operator physically propping the ICU door open (e.g. for
    # a maintenance window).  We mutate the ontology individual directly, then
    # replan to show the cost reduction (no door-opening penalty: -2.5 units).
    #
    # After the scenario, the door is restored to CLOSED so subsequent
    # scenarios see the original world.
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 5 — Counterfactual simulation: ICU door NOW OPEN")
    print("               (simulates an operator propping the door open)")
    print(DIVIDER)

    # Mutate the live ontology individual (direct world-state change)
    icu_door = onto.individual("seg_ICUent_ICUmain")
    icu_door.set("door_state", DoorState.OPEN.value)

    path5, exp5 = nav.navigate(
        start="entrance",
        goal="icu_main",
        agent=agents["delivery_bot"],
        k_alternatives=3,
    )
    if exp5:
        nav.print_report(exp5)

    # Restore the door to closed (so scenario 6 sees the original world)
    icu_door.set("door_state", DoorState.CLOSED.value)

    # ══════════════════════════════════════════════════════════════════════════
    # SCENARIO 6 — Improvement roadmap for cargo_bot
    #
    # Answers the question: "What infrastructure changes would allow cargo_bot
    # to reach the ICU?"  Generates per-route counterfactuals when the base
    # path is infeasible, providing an actionable prioritised roadmap.
    # ══════════════════════════════════════════════════════════════════════════
    print(DIVIDER)
    print("  SCENARIO 6 — Improvement roadmap for cargo_bot")
    print("               (which world changes unlock which routes?)")
    print(DIVIDER)

    _, exp6 = nav.navigate(
        start="entrance",
        goal="icu_main",
        agent=agents["cargo_bot"],
        k_alternatives=4,
    )
    if exp6 and exp6.recommendations:
        print("\n  Operator recommendations to improve cargo_bot reachability:\n")
        for rec in exp6.recommendations:
            print(f"    {rec}")
    else:
        # Base path infeasible — generate route-specific analysis manually
        print("  Base path infeasible for cargo_bot — generating route-specific analysis:")
        for route_name, alt in [
            ("direct (via narrow corridor)",
             ["entrance", "lobby", "corridor_a", "icu_entrance", "icu_main"]),
            ("detour (via corridor_b)",
             ["entrance", "lobby", "corridor_a", "corridor_b",
              "icu_entrance", "icu_main"]),
        ]:
            cf6 = nav.query_why_not("entrance", "icu_main", agents["cargo_bot"], alt)
            print(f"\n  Route [{route_name}]:")
            for ch in cf6.changes:
                tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
                print(f"    [{tag}] {ch.description()}")
                print(f"           {ch.rationale}")
                print(f"           Effort: {ch.effort()}")


if __name__ == "__main__":
    run()
