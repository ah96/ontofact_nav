"""
ontofact_nav/config.py
======================
Single source of truth for the framework's numeric policy.

`NavCostConfig` centralises every previously-hard-coded constant — affordance
thresholds, soft-cost weights, geometric fallbacks, and counterfactual targets —
**plus** the derived feasibility predicates that the rule engine, the cost
function, and the counterfactual search all share.  Co-locating the constants and
the predicates means the geometry/threshold formulas exist in exactly one place
(addresses both the "magic numbers" and the "duplicated passability logic" issues).

Two kinds of numeric fields:
  THRESHOLD / GEOMETRY → consumed by rule conditions; they affect *feasibility*.
  WEIGHT / PENALTY     → consumed only by the soft-cost section of
                         ``navigation_cost``; they never affect feasibility.

The dataclass is ``frozen`` so a single shared instance cannot drift mid-run.
``surface_cost`` uses ``default_factory`` so the dict is per-instance even though
the dataclass is frozen.  Defaults reproduce the framework's historical numeric
behaviour exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .domain import DoorState, MobilityType, SurfaceType


@dataclass(frozen=True)
class NavCostConfig:
    """Numeric policy + derived feasibility predicates (see module docstring)."""

    # ── Geometric fallbacks (for incomplete individuals) ─────────────────────
    default_space_width:    float = 10.0
    default_space_height:   float = 3.0
    default_robot_width:    float = 0.5
    default_robot_height:   float = 1.5
    default_min_clearance:  float = 0.15
    clearance_buffer_sides: int   = 2     # required width = robot_width + 2 × clearance

    # ── Slope ────────────────────────────────────────────────────────────────
    default_slope_angle:       float = 0.0
    default_max_slope_angle:   float = 8.0
    wheeled_stair_slope_limit: float = 15.0   # wheels fail on stair-like angles

    # ── Illumination ─────────────────────────────────────────────────────────
    default_illumination: float = 1.0
    observable_threshold: float = 0.25        # illumination > this → OBSERVABLE

    # ── Occupancy ────────────────────────────────────────────────────────────
    default_crowd_density:    float = 0.0
    default_obstacle_density: float = 0.0
    crowd_cost_weight:        float = 4.0     # high weight: crowds are slow/risky
    obstacle_cost_weight:     float = 2.5
    crowd_block_threshold:    float = 0.5     # counterfactual: "very crowded"

    # ── Door ─────────────────────────────────────────────────────────────────
    door_opening_penalty: float = 2.5         # CLOSED + OPENABLE soft cost

    # ── Visibility / slope soft cost ─────────────────────────────────────────
    visibility_penalty: float = 1.8           # added when OBSERVABLE missing
    slope_cost_coeff:   float = 0.06          # per-degree gentle gradient cost

    # ── Route preference ─────────────────────────────────────────────────────
    emergency_discount: float = 0.75          # 25 % off designated routes
    cost_floor:         float = 0.01          # guard against zero-cost edges

    # ── Counterfactual targets / search ──────────────────────────────────────
    cf_crowd_target:        float = 0.2
    cf_illumination_target: float = 0.6
    cf_width_safety_margin: float = 0.1
    cf_slope_margin:        float = 1.0
    cf_problematic_factor:  float = 1.5
    cf_problematic_buffer:  float = 2.5       # unified with door_opening_penalty
                                              # (was a separate hard-coded 2.0)
    cf_max_pool:            int   = 12        # exhaustive-search cap before fallback

    # ── Surface friction table (cost adders over the base edge distance) ─────
    surface_cost: Dict[str, float] = field(default_factory=lambda: {
        SurfaceType.SMOOTH.value:   0.0,
        SurfaceType.TILED.value:    0.0,
        SurfaceType.CARPETED.value: 0.3,
        SurfaceType.ROUGH.value:    0.6,
        SurfaceType.WET.value:      1.4,   # slip risk → extra caution
        SurfaceType.GRAVEL.value:   0.9,
        SurfaceType.GRASS.value:    0.7,
    })

    # ── Derived SSOT predicate helpers ───────────────────────────────────────
    # These are the ONLY definitions of the geometry/threshold formulas.  Both
    # the affordance grant/block rules AND the counterfactual candidate generator
    # call them, so the two can never drift apart.

    def required_width(self, a: Dict[str, Any]) -> float:
        """Total width a robot needs: body width + one clearance buffer per side."""
        return (a.get("robot_width", self.default_robot_width)
                + self.clearance_buffer_sides
                * a.get("min_clearance", self.default_min_clearance))

    def fits_width(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        return e.get("width", self.default_space_width) >= self.required_width(a)

    def fits_height(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        return (e.get("height", self.default_space_height)
                >= a.get("robot_height", self.default_robot_height))

    def slope_within_limit(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        return (e.get("slope_angle", self.default_slope_angle)
                <= a.get("max_slope_angle", self.default_max_slope_angle))

    def wheeled_on_stairs(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """Wheeled robots cannot climb stair-like angles regardless of their limit."""
        return (e.get("slope_angle", self.default_slope_angle) > self.wheeled_stair_slope_limit
                and a.get("mobility_type", MobilityType.WHEELED.value)
                    == MobilityType.WHEELED.value)

    def lit_enough(self, e: Dict[str, Any]) -> bool:
        return e.get("illumination", self.default_illumination) > self.observable_threshold

    def can_open_closed_door(self, e: Dict[str, Any], a: Dict[str, Any]) -> bool:
        """Robot can cross a CLOSED (non-locked) door iff it can open doors.

        Single source for the door-crossing condition: reused by the OPENABLE
        grant rule and by the TRAVERSABLE block rule that folds the door
        hard-block into the affordance set.
        """
        return (e.get("door_state") == DoorState.CLOSED.value
                and a.get("can_open_doors", False))
