"""
ontofact_nav/domain.py
======================
Navigation domain schema: symbolic constants (enums) and the factory function
that constructs the OWL-style ontology used throughout the framework.

Why separate schema from the engine (ontology.py)?
  ontology.py is a generic, domain-agnostic engine.  domain.py is the
  application-specific vocabulary.  This separation means you can swap in a
  different domain (manufacturing, outdoor terrain, …) without touching the
  core engine.

Adding properties to this schema does NOT automatically make them usable by the
affordance engine.  You must also add matching AffordanceRules in affordance.py
and (optionally) matching counterfactual-change logic in counterfactual.py.
"""

from __future__ import annotations

from enum import Enum

from .ontology import Ontology


# ---------------------------------------------------------------------------
# Symbolic constants — used by the affordance engine, counterfactual engine,
# and scenario builders so string literals never appear in rule conditions.
# ---------------------------------------------------------------------------

class AffordanceType(Enum):
    """
    Ecological affordances relevant to robot navigation (Gibson, 1979).

    Each value represents one type of action possibility that a space may
    (or may not) offer to a given robot.  Affordances are not stored on
    individuals — they are *inferred* by the AffordanceReasoner from raw
    ontology properties.

    TRAVERSABLE  — Robot is permitted to enter and move through the space.
                   Blocked by: hazards, restricted areas, inaccessibility.
    PASSABLE     — Robot's body (width + clearance) fits within the space.
                   Blocked by: corridors narrower than robot + 2×clearance,
                               ceilings lower than robot height.
    CLIMBABLE    — The slope angle is within what the robot's drive system
                   can handle.  Blocked by slopes > max_slope_angle.
    OPENABLE     — The robot can open a closed (but not locked) door.
                   Only relevant for spaces that explicitly have a door.
    OBSERVABLE   — Space is well-lit enough for reliable sensor readings.
                   Blocked by illumination ≤ 0.25 (25% of maximum).
    AVOIDABLE    — This space can be excluded from the navigation plan.
                   Always granted — the planner can always route around.
    """
    TRAVERSABLE  = "traversable"
    PASSABLE     = "passable"
    CLIMBABLE    = "climbable"
    OPENABLE     = "openable"
    OBSERVABLE   = "observable"
    AVOIDABLE    = "avoidable"


class SurfaceType(Enum):
    """
    Floor surface materials, ordered roughly by friction / navigation cost.

    Used by the AffordanceReasoner cost formula to add a surface-friction
    penalty on top of the Euclidean edge distance.  Wet surfaces have the
    highest penalty (1.4) because they risk slipping; grass is moderate (0.7).
    """
    SMOOTH   = "smooth"     # cost adder: 0.0
    TILED    = "tiled"      # cost adder: 0.0
    CARPETED = "carpeted"   # cost adder: 0.3
    ROUGH    = "rough"      # cost adder: 0.6
    WET      = "wet"        # cost adder: 1.4  (highest — slip risk)
    GRAVEL   = "gravel"     # cost adder: 0.9
    GRASS    = "grass"      # cost adder: 0.7


class DoorState(Enum):
    """
    State of a door blocking entry to a space.

    Only spaces that explicitly carry the 'door_state' property are subject
    to door-related affordance rules.  A corridor with no 'door_state' key
    is treated as if it has no door at all.

    OPEN   — Robot passes through freely (no cost penalty).
    CLOSED — Robot CAN pass if it has the OPENABLE affordance (+2.5 cost).
    LOCKED — Robot CANNOT pass regardless of capabilities (impassable).
    """
    OPEN   = "open"
    CLOSED = "closed"
    LOCKED = "locked"


class MobilityType(Enum):
    """
    Robot drive system type.

    Used in affordance rules (e.g., wheeled robots cannot climb stairs
    steeper than ~8° regardless of max_slope_angle setting).

    WHEELED — Standard differential-drive or omnidirectional wheel base.
    LEGGED  — Bipedal or quadrupedal walking robot.
    TRACKED — Continuous track (tank-style) for rough terrain.
    AERIAL  — Flying robot (ignores floor properties entirely).
    """
    WHEELED = "wheeled"
    LEGGED  = "legged"
    TRACKED = "tracked"
    AERIAL  = "aerial"


# ---------------------------------------------------------------------------
# Domain ontology factory
# ---------------------------------------------------------------------------

def build_navigation_ontology() -> Ontology:
    """
    Construct and return the navigation domain ontology.

    This function is the single source of truth for:
      - The class hierarchy (20 classes)
      - The property vocabulary (22 data properties)

    It does NOT create any individual instances — that is the responsibility
    of the scenario builders (see scenarios/hospital.py and scenarios/warehouse.py).

    Class hierarchy
    ───────────────
    Thing
    ├── PhysicalEntity
    │   ├── Space
    │   │   ├── IndoorSpace
    │   │   │   ├── Room
    │   │   │   ├── Corridor
    │   │   │   ├── Staircase
    │   │   │   ├── Elevator
    │   │   │   └── Doorway
    │   │   ├── OutdoorSpace
    │   │   │   └── OutdoorPath
    │   │   └── Ramp            (can be indoor or outdoor)
    │   ├── PhysicalObject
    │   │   ├── Door
    │   │   └── Obstacle
    │   └── Agent
    │       └── Robot
    └── AbstractEntity
        ├── Affordance          (not instantiated — used as a marker)
        └── NavigationAction    (not instantiated — reserved for future use)

    Why is Ramp a sibling of IndoorSpace / OutdoorSpace rather than a child?
      A ramp can exist indoors (accessible ramp beside stairs) or outdoors
      (loading dock approach).  Making it a child of Space avoids the need
      for multiple inheritance.
    """
    onto = Ontology("NavigationOntology")

    # ── Top-level taxonomy ──────────────────────────────────────────────────
    # "Thing" is the universal superclass (mirroring owl:Thing).
    onto.defclass("Thing")
    onto.defclass("PhysicalEntity", parent_name="Thing",
                  description="Any physical entity in the environment")
    onto.defclass("AbstractEntity", parent_name="Thing",
                  description="Non-physical conceptual entities")

    # ── Space taxonomy ──────────────────────────────────────────────────────
    # Space is the key class for navigation — all navigable regions subtype it.
    onto.defclass("Space",        parent_name="PhysicalEntity",
                  description="Any navigable region of the environment")
    onto.defclass("IndoorSpace",  parent_name="Space",
                  description="Enclosed, climate-controlled spaces")
    onto.defclass("OutdoorSpace", parent_name="Space",
                  description="Open-air navigable areas")

    # Indoor sub-spaces — each has different typical properties
    onto.defclass("Room",      parent_name="IndoorSpace",
                  description="General-purpose enclosed room")
    onto.defclass("Corridor",  parent_name="IndoorSpace",
                  description="Narrow linear passage connecting spaces")
    onto.defclass("Staircase", parent_name="IndoorSpace",
                  description="Stepped passage between floors")
    onto.defclass("Elevator",  parent_name="IndoorSpace",
                  description="Mechanical vertical transport unit")
    onto.defclass("Doorway",   parent_name="IndoorSpace",
                  description="Short threshold passage with a door")

    onto.defclass("OutdoorPath", parent_name="OutdoorSpace",
                  description="Paved or unpaved path in an outdoor area")
    onto.defclass("Ramp",        parent_name="Space",
                  description="Sloped passage; may be indoors or outdoors")

    # ── Physical objects ────────────────────────────────────────────────────
    onto.defclass("PhysicalObject", parent_name="PhysicalEntity",
                  description="Discrete physical object (not a space)")
    onto.defclass("Door",     parent_name="PhysicalObject")
    onto.defclass("Obstacle", parent_name="PhysicalObject")

    # ── Agents ─────────────────────────────────────────────────────────────
    # Agent is kept general so non-robot agents could be added later.
    onto.defclass("Agent", parent_name="PhysicalEntity",
                  description="Autonomous or semi-autonomous acting entity")
    onto.defclass("Robot", parent_name="Agent",
                  description="Mobile robot navigating the environment")

    # ── Abstract entities ───────────────────────────────────────────────────
    onto.defclass("Affordance",       parent_name="AbstractEntity",
                  description="Action possibility inferred from space + agent properties")
    onto.defclass("NavigationAction", parent_name="AbstractEntity",
                  description="Abstract navigation action (reserved for future use)")

    # ── Space data properties ───────────────────────────────────────────────
    # These properties characterise individual spaces (corridors, rooms, etc.)
    # and are the raw inputs consumed by the affordance rule engine.

    space_props = [
        # Geometry — used by the PASSABLE and CLIMBABLE affordance rules
        ("width",            float, "Passable width in metres"),
        ("height",           float, "Overhead clearance in metres"),
        ("length",           float, "Length of the passage in metres"),
        ("slope_angle",      float, "Slope angle in degrees (0 = flat)"),

        # Surface — used by the friction cost formula in AffordanceReasoner
        ("surface_type",     str,   "Floor surface (SurfaceType enum value)"),

        # Visibility — used by the OBSERVABLE affordance rule
        ("illumination",     float, "Illumination level in [0, 1]; 1.0 = fully lit"),

        # Occupancy — used as soft cost multipliers (not hard blockers)
        ("obstacle_density", float, "Fraction of floor blocked by obstacles [0, 1]"),
        ("crowd_density",    float, "Human crowd density [0, 1]"),

        # Door — used by the OPENABLE affordance rule (only if key present)
        ("door_state",       str,   "DoorState enum value: open | closed | locked"),

        # Access control — used by the TRAVERSABLE affordance rules
        ("is_accessible",    bool,  "Whether the space is currently accessible"),
        ("is_hazardous",     bool,  "Whether the space contains active hazards"),
        ("restricted",       bool,  "Staff-only / restricted access area"),

        # Route type — used for the emergency-route cost discount
        ("emergency_route",  bool,  "Designated emergency exit / route"),
    ]
    for name, rtype, desc in space_props:
        onto.defproperty(name, domain="Space", range_type=rtype, description=desc)

    # ── Robot data properties ───────────────────────────────────────────────
    # These properties characterise the robot and are compared against space
    # properties inside affordance rule conditions.

    robot_props = [
        # Body dimensions — compared against space geometry for PASSABLE
        ("robot_width",     float, "Robot body width in metres"),
        ("robot_height",    float, "Robot body height in metres"),

        # Required side clearance on each side (collision avoidance buffer)
        # Total required width = robot_width + 2 × min_clearance
        ("min_clearance",   float, "Required side clearance per side in metres"),

        # Drive system limits — used by CLIMBABLE rule
        ("max_slope_angle", float, "Maximum navigable slope angle in degrees"),
        ("mobility_type",   str,   "MobilityType enum value"),

        # Manipulation capabilities — used by OPENABLE rule
        ("can_open_doors",  bool,  "Whether robot can open closed (unlocked) doors"),
        ("has_arm",         bool,  "Whether robot has a manipulation arm"),

        # Operational status (available for extension; not used in current rules)
        ("battery_level",   float, "Current battery charge [0, 1]"),
        ("max_speed",       float, "Maximum speed in m/s"),
    ]
    for name, rtype, desc in robot_props:
        onto.defproperty(name, domain="Robot", range_type=rtype, description=desc)

    return onto
