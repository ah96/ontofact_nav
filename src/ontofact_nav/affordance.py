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

from .classification import Classifier
from .config import NavCostConfig
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
    requires_class       : Optional[str] — if set, the rule only applies to
                           individuals whose ontology class is, or descends
                           from, this class name.  Lets a rule depend on the
                           ontology hierarchy (e.g. "a Staircase is not
                           climbable for wheeled robots") rather than on raw
                           properties alone.  ``None`` ⇒ applies to every space.
    requires_category    : Optional[str] — if set, the rule only applies to
                           individuals that belong to this SPARQL-inferred
                           category (see classification.py), e.g. "Steep" or
                           "HighRiskZone".  Lets a rule depend on a numeric
                           inference the ontology computes via SPARQL.
    """
    name:                 str
    affordance:           AffordanceType
    condition:            Callable[[Dict[str, Any], Dict[str, Any]], bool]
    explanation_template: str
    priority:             int = 0
    requires_class:       Optional[str] = None
    requires_category:    Optional[str] = None   # SPARQL-inferred category gate

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

    All numeric thresholds and the geometry/feasibility predicates come from an
    injected :class:`NavCostConfig` (defaults reproduce historical behaviour),
    so the rule conditions and the cost formula share one source of truth.
    """

    def __init__(self, config: Optional[NavCostConfig] = None) -> None:
        self.config = config or NavCostConfig()
        # SPARQL-driven numeric classifier (memoized).  Infers space categories
        # such as "Steep" / "HighRiskZone" that gate the category-conditioned
        # rules below — see classification.py.
        self.classifier = Classifier(self.config)
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

        Every geometric / threshold condition delegates to a ``NavCostConfig``
        predicate (``cfg.fits_width`` etc.), so the rule engine and the cost
        function evaluate the *same* formula — there is no second copy of the
        passability logic to drift out of sync.

        Hard barriers that used to be re-checked inside ``navigation_cost``
        (locked / unopenable doors, over-steep slopes) are folded in here as
        block rules on TRAVERSABLE / CLIMBABLE.  The cost function therefore
        decides feasibility purely from the resulting affordance set.

        Rule naming convention: <action>_<affordance>_<condition_summary>
        """
        cfg = self.config

        # ── TRAVERSABLE ────────────────────────────────────────────────────
        # A space is traversable if it is accessible, non-hazardous, not
        # restricted, AND not blocked by an impassable door (locked, or closed
        # with no way for this robot to open it).  Any single failure mode is a
        # dedicated block rule so the counterfactual engine sees a specific reason.

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
        # Door hard-barriers folded into TRAVERSABLE (single source of truth):
        # a locked door is impassable; a closed door is impassable unless this
        # robot can open it.  cfg.can_open_closed_door is the SAME predicate the
        # OPENABLE grant rule uses, so the two cannot disagree.
        self.block_rules.append(AffordanceRule(
            name="block_traversable_door_locked",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, _a: e.get("door_state") == DoorState.LOCKED.value,
            explanation_template="{entity} door is locked — hard barrier, not traversable",
        ))
        self.block_rules.append(AffordanceRule(
            name="block_traversable_door_closed_uncrossable",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, a: (
                e.get("door_state") == DoorState.CLOSED.value
                and not cfg.can_open_closed_door(e, a)
            ),
            explanation_template="{entity} door is closed and robot cannot open it",
        ))
        # Combined-condition category (SPARQL-inferred): a "HighRiskZone" is
        # crowded AND poorly lit — a conjunction the per-property rules don't
        # express (crowd is a soft cost; low illumination only blocks OBSERVABLE).
        # A robot without risk certification may not enter such a zone.
        self.block_rules.append(AffordanceRule(
            name="block_traversable_high_risk_uncertified",
            affordance=AffordanceType.TRAVERSABLE,
            requires_category="HighRiskZone",
            condition=lambda e, a: not a.get("risk_certified", False),
            explanation_template=(
                "{entity} is a HighRiskZone (crowded + dark); "
                "robot lacks risk certification"
            ),
        ))

        # ── PASSABLE ───────────────────────────────────────────────────────
        # Robot body vs. space geometry.  Required width / height live in
        # NavCostConfig (cfg.fits_width / cfg.fits_height).

        self.grant_rules.append(AffordanceRule(
            name="passable_if_wide_enough",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: cfg.fits_width(e, a) and cfg.fits_height(e, a),
            explanation_template=(
                "{entity} width ≥ robot_width + 2×clearance; "
                "height ≥ robot height"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_passable_too_narrow",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: not cfg.fits_width(e, a),
            explanation_template=(
                "{entity} width < robot_width + 2×min_clearance "
                "(robot does not fit)"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_passable_too_low",
            affordance=AffordanceType.PASSABLE,
            condition=lambda e, a: not cfg.fits_height(e, a),
            explanation_template=(
                "{entity} clearance < robot height (robot does not fit)"
            ),
        ))

        # ── CLIMBABLE ──────────────────────────────────────────────────────
        # Slope within the robot's drive-system limit, plus a wheeled-on-stairs
        # hard block.  Because flat spaces always satisfy slope_within_limit,
        # CLIMBABLE is granted for every ordinary corridor — so "CLIMBABLE
        # absent" is an exact signal that a steep-slope/stairs block fired, and
        # the cost function can treat CLIMBABLE membership as the slope gate.

        self.grant_rules.append(AffordanceRule(
            name="climbable_slope_within_limit",
            affordance=AffordanceType.CLIMBABLE,
            condition=lambda e, a: cfg.slope_within_limit(e, a),
            explanation_template=(
                "{entity} slope ≤ agent max slope — within drive-system capability"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_climbable_steep_slope",
            affordance=AffordanceType.CLIMBABLE,
            condition=lambda e, a: not cfg.slope_within_limit(e, a),
            explanation_template=(
                "{entity} slope exceeds agent max_slope_angle — drive system limit"
            ),
        ))
        # Wheeled-on-stairs: the fixed stair-slope threshold is now inferred by
        # SPARQL as the "Steep" category (classification.py), so the ONTOLOGY owns
        # that numeric inference.  Equivalent to the old cfg.wheeled_on_stairs:
        # Steep ⟺ slope > cfg.wheeled_stair_slope_limit (same injected constant);
        # the mobility check stays Python (a string equality, not a numeric rule).
        self.block_rules.append(AffordanceRule(
            name="block_climbable_steep_zone_wheeled",
            affordance=AffordanceType.CLIMBABLE,
            requires_category="Steep",
            condition=lambda e, a: (
                a.get("mobility_type", MobilityType.WHEELED.value)
                == MobilityType.WHEELED.value
            ),
            explanation_template=(
                "{entity} is a Steep zone (slope > stair limit) — "
                "wheeled robot cannot climb"
            ),
        ))
        # Class-conditioned rule (the ontology is load-bearing here): a
        # Staircase — by its CLASS, not merely its numeric slope — cannot be
        # ascended by a wheeled robot.  Fires only for Staircase instances.
        self.block_rules.append(AffordanceRule(
            name="block_climbable_wheeled_on_staircase_class",
            affordance=AffordanceType.CLIMBABLE,
            requires_class="Staircase",
            condition=lambda e, a: (
                a.get("mobility_type", MobilityType.WHEELED.value)
                == MobilityType.WHEELED.value
            ),
            explanation_template=(
                "{entity} is a Staircase — a wheeled robot cannot ascend steps"
            ),
        ))

        # ── OPENABLE ───────────────────────────────────────────────────────
        # OPENABLE is ONLY relevant for spaces that explicitly carry a
        # 'door_state' of CLOSED or LOCKED.  Doorless corridors are unaffected.
        # It drives the door-opening soft cost (not feasibility — feasibility is
        # handled by the TRAVERSABLE door blocks above).

        def _has_closed_or_locked_door(e: dict) -> bool:
            """True only when the space has an explicit, non-open door."""
            return e.get("door_state") in (
                DoorState.CLOSED.value, DoorState.LOCKED.value
            )

        self.grant_rules.append(AffordanceRule(
            name="openable_if_can_open_and_not_locked",
            affordance=AffordanceType.OPENABLE,
            # cfg.can_open_closed_door encodes "door is CLOSED (not locked) and
            # robot can_open_doors" — the same predicate the TRAVERSABLE door
            # block negates, guaranteeing the two stay consistent.
            condition=lambda e, a: cfg.can_open_closed_door(e, a),
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
        # Low illumination degrades sensor reliability.  Threshold lives in
        # cfg.observable_threshold; below it the robot cannot localise reliably.
        # OBSERVABLE does not hard-block traversal — it adds a cost penalty.

        self.grant_rules.append(AffordanceRule(
            name="observable_if_lit",
            affordance=AffordanceType.OBSERVABLE,
            condition=lambda e, _a: cfg.lit_enough(e),
            explanation_template=(
                "{entity} illumination above threshold — sufficient for reliable sensing"
            ),
        ))
        self.block_rules.append(AffordanceRule(
            name="block_observable_dark",
            affordance=AffordanceType.OBSERVABLE,
            condition=lambda e, _a: not cfg.lit_enough(e),
            explanation_template=(
                "{entity} too dark — localisation uncertainty"
            ),
        ))

        # ── AVOIDABLE ──────────────────────────────────────────────────────
        # Any space can in principle be excluded from the plan by routing
        # around it.  Always granted, never blocked — it exists so the
        # explanation layer can describe avoidance decisions.

        self.grant_rules.append(AffordanceRule(
            name="always_avoidable",
            affordance=AffordanceType.AVOIDABLE,
            condition=lambda _e, _a: True,
            explanation_template="{entity} can be excluded from the navigation plan",
        ))

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _entity_view(self, entity: OntologyIndividual) -> Dict[str, Any]:
        """
        The property dict a rule actually sees: explicit instance properties
        layered on top of inherited class-level defaults (instance wins).

        This is where the ontology class hierarchy becomes load-bearing for
        inference — a space that omits a property picks up its class's default,
        so two individuals with identical *explicit* properties but different
        *classes* can yield different affordances.
        """
        merged: Dict[str, Any] = {}
        # Walk root → leaf so a subclass default overrides its ancestor's.
        chain: List = []
        node = entity.ont_class
        while node is not None:
            chain.append(node)
            node = node.parent
        for node in reversed(chain):
            merged.update(node.defaults)
        merged.update(entity.properties)   # explicit instance values win
        return merged

    def compute(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
    ) -> AffordanceResult:
        """
        Evaluate all grant-rules and block-rules for this (space, agent) pair.

        Algorithm
        ---------
        1. Build the class-aware entity view (instance props over class defaults).
        2. Evaluate every grant-rule; collect (affordance → explanation) for fires.
        3. Evaluate every block-rule; collect (affordance → reason) for fires.
        4. Net = granted.keys() − blocked.keys()

        A rule with ``requires_class`` set is skipped unless the entity's
        ontology class is, or descends from, that class.  A rule with
        ``requires_category`` set is skipped unless the entity belongs to that
        SPARQL-inferred category.  So rule applicability consults both the
        ontology hierarchy and the SPARQL numeric classification.

        Edge cases
        ----------
        - If a rule's format-template has keys absent from the property dict,
          we catch the exception and fall back to a partial format.  This keeps
          the engine robust against incomplete ontology individuals.
        """
        e = self._entity_view(entity)
        a = agent.properties

        # SPARQL-inferred numeric categories for this space (memoized).  Computed
        # from the class-aware property view, so counterfactual clones with
        # mutated properties reclassify correctly.
        cats = self.classifier.derived_categories(e, a)

        granted: Dict[AffordanceType, str] = {}
        blocked: Dict[AffordanceType, str] = {}

        def _applies(rule: AffordanceRule) -> bool:
            if rule.requires_class is not None and not entity.is_instance_of_name(rule.requires_class):
                return False
            if rule.requires_category is not None and rule.requires_category not in cats:
                return False
            return True

        # Evaluate grant rules
        for rule in self.grant_rules:
            if not _applies(rule):
                continue
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
            if not _applies(rule):
                continue
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

    # Affordances every passable edge must have.  Door and slope hard-barriers
    # are folded into these (see _register_default_rules), so feasibility is a
    # single membership test — the cost function never re-reads raw properties
    # to make a feasibility decision.
    _HARD_REQUIRED = (
        AffordanceType.TRAVERSABLE,
        AffordanceType.PASSABLE,
        AffordanceType.CLIMBABLE,
    )

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

        Passability contract (single source of truth)
        ----------------------------------------------
        An edge is impassable **iff** the net affordance set is missing one of
        ``{TRAVERSABLE, PASSABLE, CLIMBABLE}``.  Every former raw-property
        hard-block (locked / unopenable door, over-steep slope, wheeled-on-
        stairs) is now a block rule that removes one of these affordances, so
        this function makes no independent feasibility decision — it only adds
        soft penalties below the gate.  ``door_state``/``slope`` are read here
        purely as *cost inputs*, never to decide feasibility.

        Soft cost formula (all weights from NavCostConfig)
        --------------------------------------------------
        cost = base_distance
             + door_opening_penalty   (if door CLOSED and OPENABLE present)
             + surface_friction
             + crowd_density   × crowd_cost_weight
             + obstacle_density × obstacle_cost_weight
             + visibility_penalty     (if OBSERVABLE missing)
             + slope_angle × slope_cost_coeff
             × emergency_discount     (if emergency_route)
        """
        result = self.compute(entity, agent)
        af  = set(result.affordances)
        cfg = self.config

        # ── SINGLE hard-block gate: feasibility read ONLY from the affordances ──
        if any(req not in af for req in self._HARD_REQUIRED):
            return math.inf, result

        # ── Soft costs only below this line (class-aware property view) ─────────
        e    = self._entity_view(entity)
        cost = base_distance

        # Door-opening adds time cost (robot must stop, engage manipulator)
        door_state = e.get("door_state", DoorState.OPEN.value)
        if door_state == DoorState.CLOSED.value and AffordanceType.OPENABLE in af:
            cost += cfg.door_opening_penalty

        # Surface friction (material resistance / slip precaution)
        surface = e.get("surface_type", SurfaceType.SMOOTH.value)
        cost += cfg.surface_cost.get(surface, 0.0)

        # Crowd — high weight because crowds slow the robot significantly
        cost += e.get("crowd_density",    cfg.default_crowd_density)    * cfg.crowd_cost_weight
        # Obstacles — moderate weight (robot can weave around scattered obstacles)
        cost += e.get("obstacle_density", cfg.default_obstacle_density) * cfg.obstacle_cost_weight

        # Low visibility adds a navigation-uncertainty penalty
        if AffordanceType.OBSERVABLE not in af:
            cost += cfg.visibility_penalty

        # Gradual slope adds mild extra cost (motor load)
        slope = e.get("slope_angle", cfg.default_slope_angle)
        if slope > 0.0:
            cost += slope * cfg.slope_cost_coeff

        # Emergency-designated routes are preferred
        if e.get("emergency_route", False):
            cost *= cfg.emergency_discount

        return max(cost, cfg.cost_floor), result   # guard against zero-cost edges
