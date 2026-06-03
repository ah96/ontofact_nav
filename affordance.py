"""
ontofact_nav/affordance.py
==========================
Affordance inference engine based on forward-chaining rules.

Theoretical basis
-----------------
James Gibson (1979) defined an *affordance* as the set of action possibilities
that the environment offers to an animal (or agent).  For robot navigation the
relevant affordances are: can the robot *traverse* this space? does it *fit*?
can it *climb* a slope? can it *open* a door?

Implementation pattern
----------------------
Each ``AffordanceRule`` is a triple (condition, affordance, explanation):
  - *condition*     — a callable (entity_props, agent_props) → bool
  - *affordance*    — which AffordanceType this rule grants or blocks
  - *explanation*   — a format string for human-readable output

The ``AffordanceReasoner`` evaluates ALL grant-rules and ALL block-rules.
Net affordances = { a | some grant-rule fires for a }
               − { a | some block-rule fires for a }

This open-world / monotone semantics means:
  - Adding a rule never silently removes others.
  - Conflicting rules produce the union of grants minus the union of blocks.
  - A strong block (e.g. locked door) always wins over a grant.

Navigation cost
---------------
``navigation_cost()`` converts the net affordance set into a scalar edge
weight for A*.  Missing TRAVERSABLE or PASSABLE → cost = inf (impassable).
Soft penalties (surface friction, crowd, illumination) add to the base distance.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .domain import AffordanceType, DoorState, MobilityType, SurfaceType
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Rule & result data structures
# ---------------------------------------------------------------------------

@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AffordanceRule:
    """
    A single forward-chaining inference rule.

    Attributes
    ----------
    name                 : str — unique identifier (used in debug output)
    affordance           : AffordanceType — which affordance this grants/blocks
    condition            : callable(entity_props: dict, agent_props: dict) → bool
                           Receives the raw property dicts, NOT the Individual
                           objects, to keep rule code terse.
    explanation_template : str — Python str.format template; may reference
                           {entity} (individual name) and any property key.
    priority             : int — reserved for future ordered evaluation
    """
    name:                 str
    affordance:           AffordanceType
    condition:            Callable[[Dict[str, Any], Dict[str, Any]], bool]
    explanation_template: str
    priority:             int = 0

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("AffordanceRule name must not be empty")
        return v

    @field_validator("priority")
    @classmethod
    def _priority_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"priority must be >= 0, got {v}")
        return v


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class AffordanceResult:
    """
    The computed affordance profile for a (space, agent) pair.

    Returned by ``AffordanceReasoner.compute()`` and stored alongside each
    edge in a NavPath for post-hoc explanation generation.

    Attributes
    ----------
    entity_name        : name of the space individual
    affordances        : net affordances (granted minus blocked)
    blocked_affordances: affordances that were granted but then blocked
    explanations       : {affordance → reason it was granted}
    blocking_reasons   : {affordance → reason it was blocked}
    confidence         : placeholder for probabilistic extensions (always 1.0)
    """
    entity_name:         str
    affordances:         List[AffordanceType]
    blocked_affordances: List[AffordanceType]
    explanations:        Dict[AffordanceType, str]
    blocking_reasons:    Dict[AffordanceType, str]
    confidence:          float = 1.0

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    def has(self, a: AffordanceType) -> bool:
        """Check membership in the net affordance set."""
        return a in self.affordances

    def missing(self, a: AffordanceType) -> bool:
        """True if this affordance is not in the net set."""
        return a not in self.affordances


# ---------------------------------------------------------------------------
# Affordance reasoner
# ---------------------------------------------------------------------------

class AffordanceReasoner:
    """
    Infers affordances of a navigation space for a specific robot agent.

    Maintains two lists of AffordanceRule:
      grant_rules — add an affordance when fired
      block_rules — remove an affordance when fired

    Call ``add_grant()`` / ``add_block()`` to extend the rule set at runtime
    (e.g. to add domain-specific constraints without subclassing).
    """

    def __init__(self) -> None:
        self.grant_rules: List[AffordanceRule] = []
        self.block_rules: List[AffordanceRule] = []
        self._register_default_rules()

    # ------------------------------------------------------------------
    # Rule registration helpers
    # ------------------------------------------------------------------

    def add_grant(self, rule: AffordanceRule) -> None:
        """Register a new grant rule (appended; no priority sorting yet)."""
        self.grant_rules.append(rule)

    def add_block(self, rule: AffordanceRule) -> None:
        """Register a new block rule."""
        self.block_rules.append(rule)

    # ------------------------------------------------------------------
    # Built-in rule set
    # ------------------------------------------------------------------

    def _register_default_rules(self) -> None:
        """
        Register the standard navigation affordance rules.

        Rules are grouped by affordance type for readability.  Each group
        has one or more grant rules followed by one or more block rules.

        Rule naming convention: <action>_<affordance>_<condition_summary>
        """

        # ── TRAVERSABLE ────────────────────────────────────────────────────
        # A space is traversable if it is accessible, non-hazardous, and not
        # restricted.  All three flags must be true; any single false value
        # blocks the affordance via a dedicated block rule.

        self.grant_rules.append(AffordanceRule(
            name="traversable_if_accessible_and_safe",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, _a: (
                e.get("is_accessible", True)
                and not e.get("is_hazardous", False)
                and not e.get("restricted", False)
            ),
            explanation_template=(
                "{entity} is accessible, non-hazardous, and not restricted"
            ),
        ))

        # Three separate block rules (one per failure mode) so the blocking
        # reason in AffordanceResult is specific enough to drive counterfactuals.
        self.block_rules.append(AffordanceRule(
            name="block_traversable_hazardous",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, _a: e.get("is_hazardous", False),
            explanation_template="{entity} is marked hazardous",
        ))
        self.block_rules.append(AffordanceRule(
            name="block_traversable_restricted",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, _a: e.get("restricted", False),
            explanation_template="{entity} is a restricted/staff-only area",
        ))
        self.block_rules.append(AffordanceRule(
            name="block_traversable_inaccessible",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, _a: not e.get("is_accessible", True),
            explanation_template="{entity} is marked not accessible",
        ))

        # ── PASSABLE ───────────────────────────────────────────────────────
        # Depends on the robot's body dimensions relative to the space.
        # Required width = robot_width + 2 × min_clearance (one buffer per side).
        # Required height = robot_height.

        self.grant_rules.append(AffordanceRule(
            name="passable_if_wide_enough",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: (
                e.get("width",  10.0) >= a.get("robot_width", 0.5)
                                        + 2 * a.get("min_clearance", 0.15)
                and e.get("height", 3.0) >= a.get("robot_height", 1.5)
            ),
            explanation_template=(
                "{entity} width ≥ robot_width + 2×clearance; "
                "height ≥ robot height"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_passable_too_narrow",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: (
                e.get("width", 10.0) < a.get("robot_width", 0.5)
                                      + 2 * a.get("min_clearance", 0.15)
            ),
            explanation_template=(
                "{entity} width < robot_width + 2×min_clearance "
                "(robot does not fit)"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_passable_too_low",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: (
                e.get("height", 3.0) < a.get("robot_height", 1.5)
            ),
            explanation_template=(
                "{entity} clearance < robot height (robot does not fit)"
            ),
        ))

        # ── CLIMBABLE ──────────────────────────────────────────────────────
        # The slope angle of the space must be within the robot's drive-system
        # limit (max_slope_angle).  An additional hard block prevents wheeled
        # robots from climbing stair-like slopes (> 15°) even if max_slope_angle
        # were set permissively, because wheels lose traction on steps.

        self.grant_rules.append(AffordanceRule(
            name="climbable_slope_within_limit",
            affordance=AffordanceType.CLIMBABLE,
            condition=lambda e, a: (
                e.get("slope_angle", 0.0) <= a.get("max_slope_angle", 8.0)
            ),
            explanation_template=(
                "{entity} slope ≤ agent max slope — within drive-system capability"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_climbable_steep_slope",
            affordance=AffordanceType.CLIMBABLE,
            condition=lambda e, a: (
                e.get("slope_angle", 0.0) > a.get("max_slope_angle", 8.0)
            ),
            explanation_template=(
                "{entity} slope exceeds agent max_slope_angle — drive system limit"
            ),
        ))
        # Wheeled robots fail on stair-like angles even if max_slope_angle is
        # set high, because individual steps create vertical obstacles for wheels.
        self.block_rules.append(AffordanceRule(
            name="block_climbable_wheeled_on_stairs",
            affordance=AffordanceType.CLIMBABLE,
            condition=lambda e, a: (
                e.get("slope_angle", 0.0) > 15.0
                and a.get("mobility_type", MobilityType.WHEELED.value)
                   == MobilityType.WHEELED.value
            ),
            explanation_template=(
                "Wheeled robot cannot climb stair-like slope > 15° in {entity}"
            ),
        ))

        # ── OPENABLE ───────────────────────────────────────────────────────
        # OPENABLE is ONLY relevant for spaces that explicitly carry a
        # 'door_state' property set to CLOSED or LOCKED.  Corridors without
        # a door are unaffected.  This avoids spurious "openable blocked"
        # warnings for every doorless corridor (a common false-positive in
        # naive implementations).

        def _has_closed_or_locked_door(e: dict) -> bool:
            """True only when the space has an explicit, non-open door."""
            return e.get("door_state") in (
                DoorState.CLOSED.value, DoorState.LOCKED.value
            )

        self.grant_rules.append(AffordanceRule(
            name="openable_if_can_open_and_not_locked",
            affordance=AffordanceType.OPENABLE,
            condition=lambda e, a: (
                _has_closed_or_locked_door(e)                   # only if a door exists
                and a.get("can_open_doors", False)              # robot has capability
                and e.get("door_state") != DoorState.LOCKED.value  # door is not locked
            ),
            explanation_template=(
                "Robot can open doors and {entity} door is closed (not locked)"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_openable_locked",
            affordance=AffordanceType.OPENABLE,
            condition=lambda e, _a: (
                e.get("door_state") == DoorState.LOCKED.value
            ),
            explanation_template="{entity} door is locked — cannot open",
        ))
        self.block_rules.append(AffordanceRule(
            name="block_openable_no_capability",
            affordance=AffordanceType.OPENABLE,
            condition=lambda e, a: (
                _has_closed_or_locked_door(e)       # only fires when door exists
                and not a.get("can_open_doors", False)
            ),
            explanation_template=(
                "Robot lacks door-opening capability and {entity} door is "
                "{door_state}"
            ),
        ))

        # ── OBSERVABLE ─────────────────────────────────────────────────────
        # Low illumination degrades sensor reliability (cameras, LiDAR in
        # dusty/dark conditions).  Threshold 0.25 is a design choice — below
        # this the robot cannot localise with sufficient confidence.
        # OBSERVABLE does not hard-block traversal; it adds a cost penalty (1.8)
        # to account for slower, more cautious navigation under uncertainty.

        self.grant_rules.append(AffordanceRule(
            name="observable_if_lit",
            affordance=AffordanceType.OBSERVABLE,
            condition=lambda e, _a: e.get("illumination", 1.0) > 0.25,
            explanation_template=(
                "{entity} illumination > 0.25 — sufficient for reliable sensing"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_observable_dark",
            affordance=AffordanceType.OBSERVABLE,
            condition=lambda e, _a: e.get("illumination", 1.0) <= 0.25,
            explanation_template=(
                "{entity} too dark (illumination ≤ 0.25) — localisation uncertainty"
            ),
        ))

        # ── AVOIDABLE ──────────────────────────────────────────────────────
        # Any space can in principle be excluded from the plan by routing
        # around it.  This affordance is always granted and never blocked —
        # it exists so the explanation layer can describe avoidance decisions.

        self.grant_rules.append(AffordanceRule(
            name="always_avoidable",
            affordance=AffordanceType.AVOIDABLE,
            condition=lambda _e, _a: True,
            explanation_template="{entity} can be excluded from the navigation plan",
        ))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def compute(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
    ) -> AffordanceResult:
        """
        Evaluate all grant-rules and block-rules for this (space, agent) pair.

        Algorithm
        ---------
        1. Evaluate every grant-rule; collect (affordance → explanation) for fires.
        2. Evaluate every block-rule; collect (affordance → reason) for fires.
        3. Net = granted.keys() − blocked.keys()

        Edge cases
        ----------
        - If a rule's format-template has keys absent from the property dict,
          we catch the exception and fall back to a partial format.  This keeps
          the engine robust against incomplete ontology individuals.
        """
        e = entity.properties
        a = agent.properties

        granted: Dict[AffordanceType, str] = {}
        blocked: Dict[AffordanceType, str] = {}

        # Evaluate grant rules
        for rule in self.grant_rules:
            try:
                if rule.condition(e, a):
                    # Merge entity and agent props for template interpolation.
                    # Entity props take precedence so {width} refers to the
                    # space's width, not some agent property with the same name.
                    explanation = rule.explanation_template.format(
                        entity=entity.name, **e, **a
                    )
                    granted[rule.affordance] = explanation
            except (KeyError, TypeError):
                # Template referenced a missing property — still record the grant
                granted[rule.affordance] = rule.explanation_template.replace(
                    "{entity}", entity.name
                )

        # Evaluate block rules
        for rule in self.block_rules:
            try:
                if rule.condition(e, a):
                    reason = rule.explanation_template.format(
                        entity=entity.name, **e, **a
                    )
                    blocked[rule.affordance] = reason
            except (KeyError, TypeError):
                blocked[rule.affordance] = rule.explanation_template.replace(
                    "{entity}", entity.name
                )

        # Net affordances = granted minus blocked
        net = [af for af in granted if af not in blocked]

        return AffordanceResult(
            entity_name         = entity.name,
            affordances         = net,
            blocked_affordances = list(blocked.keys()),
            explanations        = {af: granted[af] for af in net},
            blocking_reasons    = blocked,
        )

    # ------------------------------------------------------------------
    # Cost derivation
    # ------------------------------------------------------------------

    # Surface friction cost adders (in cost-distance units added to edge length)
    # Calibrated so that a WET floor adds ~14% overhead over a 10 m corridor.
    _SURFACE_COST: Dict[str, float] = {
        SurfaceType.SMOOTH.value:   0.0,
        SurfaceType.TILED.value:    0.0,
        SurfaceType.CARPETED.value: 0.3,
        SurfaceType.ROUGH.value:    0.6,
        SurfaceType.WET.value:      1.4,   # slip risk → extra caution → slower
        SurfaceType.GRAVEL.value:   0.9,
        SurfaceType.GRASS.value:    0.7,
    }

    def navigation_cost(
        self,
        entity:        OntologyIndividual,
        agent:         OntologyIndividual,
        base_distance: float = 1.0,
    ) -> Tuple[float, AffordanceResult]:
        """
        Translate the affordance profile into a scalar A* edge cost.

        Returns ``(cost, AffordanceResult)``.
        ``cost == math.inf`` means the edge is impassable for this agent.

        Cost formula
        ------------
        cost = base_distance                    (Euclidean metres)
             + door_opening_penalty             (2.5 if closed but openable)
             + surface_friction                 (0.0–1.4)
             + crowd_density  × 4.0             (high weight: safety concern)
             + obstacle_density × 2.5
             + visibility_penalty               (1.8 if OBSERVABLE missing)
             + slope_angle × 0.06               (gentle gradient cost)
             × emergency_discount               (× 0.75 for preferred routes)

        The multiplier of 4.0 for crowd is intentionally high — navigating
        through dense crowds is both slow and risky for the robot.
        """
        result = self.compute(entity, agent)
        af = set(result.affordances)
        e  = entity.properties
        a  = agent.properties

        # ── Hard blockers → impassable ──────────────────────────────────────
        # TRAVERSABLE missing: space is hazardous, restricted, or inaccessible.
        if AffordanceType.TRAVERSABLE not in af:
            return math.inf, result

        # PASSABLE missing: robot body does not fit.
        if AffordanceType.PASSABLE not in af:
            return math.inf, result

        # Door check: locked is always impassable; closed requires OPENABLE.
        door_state = e.get("door_state", DoorState.OPEN.value)
        if door_state == DoorState.LOCKED.value:
            return math.inf, result
        if door_state == DoorState.CLOSED.value and AffordanceType.OPENABLE not in af:
            return math.inf, result

        # Slope check: if slope exceeds limit, CLIMBABLE will be absent.
        slope = e.get("slope_angle", 0.0)
        if slope > a.get("max_slope_angle", 8.0):
            if AffordanceType.CLIMBABLE not in af:
                return math.inf, result

        # ── Soft cost contributions ─────────────────────────────────────────
        cost = base_distance

        # Door-opening adds time cost (robot must stop, engage manipulator)
        if door_state == DoorState.CLOSED.value and AffordanceType.OPENABLE in af:
            cost += 2.5

        # Surface friction (material resistance / slip precaution)
        surface = e.get("surface_type", SurfaceType.SMOOTH.value)
        cost += self._SURFACE_COST.get(surface, 0.0)

        # Crowd — high weight because crowds slow the robot significantly
        cost += e.get("crowd_density",    0.0) * 4.0
        # Obstacles — moderate weight (robot can weave around scattered obstacles)
        cost += e.get("obstacle_density", 0.0) * 2.5

        # Low visibility adds a navigation-uncertainty penalty
        if AffordanceType.OBSERVABLE not in af:
            cost += 1.8

        # Gradual slope adds mild extra cost (motor load)
        if slope > 0.0:
            cost += slope * 0.06

        # Emergency-designated routes are preferred — 25% discount
        if e.get("emergency_route", False):
            cost *= 0.75

        return max(cost, 0.01), result   # guard against zero-cost edges
