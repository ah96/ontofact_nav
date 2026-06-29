"""
ontofact_nav/counterfactual.py
==============================
Counterfactual affordance reasoning engine.

What is a counterfactual explanation?
--------------------------------------
A counterfactual answers: "What would need to be different for outcome Y
to have occurred instead of X?"

Applied to navigation: given that the robot chose path A, we answer:
  "Why didn't it take path B?"
  → "Because property P on corridor C has value V, making C impassable.
     If P were changed to V', the robot would prefer path B."

Minimal-change semantics (Lewis 1973; Pearl 2000)
-------------------------------------------------
The "closest world" in which the alternative is preferred is the one differing
least from the actual world.  We operationalise "least" as **minimum
cardinality**: among all subsets of candidate single-property changes that make
the alternative feasible *and* at least as cheap as the chosen path, return one
of smallest size (ties broken by resulting cost, then total operator effort).
A minimum-cardinality satisfying set is automatically subset-minimal — no proper
subset also satisfies, because every smaller cardinality was exhaustively tried
and rejected.  This replaces the earlier heuristic that simply applied *every*
applicable fix at once (which was the maximal, not the minimal, change set).

Honest attribution
-------------------
We run the search twice: once over the full candidate pool (the reported
``changes`` / ``cf_cost`` / ``cost_delta``) and once restricted to
operator-*actionable* candidates only.  ``is_achievable`` is true iff the
actionable-only search finds a set that makes the alternative preferred **on its
own** — so "achievable, saves X" is always backed by a world an operator can
actually reach.  ``actionable_only_cost`` / ``actionable_only_delta`` report that
world explicitly, so a saving that secretly depends on structural construction is
never advertised as operator-doable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .affordance import AffordanceReasoner
from .config import NavCostConfig
from .domain import AffordanceType, DoorState, MobilityType, SurfaceType
from .navigation import NavEdge, NavPath, AStarPlanner
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Property taxonomy — two orthogonal axes, one source of truth
# ---------------------------------------------------------------------------
# is_actionable() — CAN an operator effect this at runtime *at all*, without
#                   physical construction/renovation?  (binary)
# effort()        — GIVEN it is doable, HOW MUCH effort?  (low / medium / high)
#
# The axes are independent: a change can be actionable yet medium-effort (e.g.
# a restricted-area access policy change is doable but takes administrative
# work).  Both methods read this single table so the two can never disagree.
PROPERTY_TAXONOMY: Dict[str, Tuple[bool, str]] = {
    # property            (actionable, effort)
    "door_state":         (True,  "low"),
    "crowd_density":      (True,  "low"),
    "illumination":       (True,  "low"),
    "obstacle_density":   (True,  "low"),
    "is_hazardous":       (True,  "low"),
    "is_accessible":      (True,  "medium"),   # doable, but an admin/policy action
    "restricted":         (True,  "medium"),
    "surface_type":       (False, "medium"),
    "width":              (False, "high"),     # structural construction
    "slope_angle":        (False, "high"),
}
_DEFAULT_TAXONOMY: Tuple[bool, str] = (False, "low")

# Ordinal ranking of effort for tie-breaking minimal sets (lower = cheaper).
_EFFORT_RANK: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PropertyChange:
    """
    A single property mutation in a counterfactual scenario.

    Represents the statement:
      "IF {individual}.{property_name} were {counterfactual_value}
       (instead of {original_value}) THEN …"

    Attributes
    ----------
    individual_name      : name of the space individual to be modified
    property_name        : the property key to change
    original_value       : current value in the actual world
    counterfactual_value : hypothetical value in the counterfactual world
    change_type          : "enable" | "disable" | "improve" | "relax"
                           purely informational — used by the explanation layer
    rationale            : one-sentence human explanation of why this change helps
    """
    individual_name:      str
    property_name:        str
    original_value:       Any
    counterfactual_value: Any
    change_type:          str = "modify"
    rationale:            str = ""

    def description(self) -> str:
        """Compact one-line description for report output."""
        return (
            f"'{self.individual_name}'.{self.property_name}: "
            f"{self.original_value!r} → {self.counterfactual_value!r}"
        )

    def is_actionable(self) -> bool:
        """
        True if a human operator could realistically make this change at runtime
        (without physical construction or renovation).

        This is the *can-they-do-it-at-all* axis; the *how-hard* axis is
        ``effort()``.  Both read PROPERTY_TAXONOMY (see module top) so they
        cannot drift apart — e.g. ``restricted`` is actionable yet medium effort.
        """
        return PROPERTY_TAXONOMY.get(self.property_name, _DEFAULT_TAXONOMY)[0]

    def effort(self) -> str:
        """
        Qualitative effort estimate, orthogonal to ``is_actionable()``.

        HIGH   — requires physical construction (months, high cost)
        MEDIUM — requires policy / admin change (days, moderate cost)
        LOW    — immediate operator action (minutes, no cost)
        """
        return PROPERTY_TAXONOMY.get(self.property_name, _DEFAULT_TAXONOMY)[1]


@dataclass
class Counterfactual:
    """
    The result of a "why not path X?" query.

    Attributes
    ----------
    query            : natural language question that prompted this analysis
    actual_path      : the path the robot actually chose
    alternative_path : the path being questioned
    changes          : the MINIMAL set of PropertyChange objects that make the
                       alternative feasible and preferred (smallest such set;
                       may include structural, non-actionable changes)
    cf_cost          : cost of the alternative under ``changes``
    cost_delta       : actual_cost − cf_cost (positive ⇒ alternative now cheaper)
    is_achievable    : True iff the *actionable-only* changes ALONE make the
                       alternative feasible and preferred (see attribution note)
    explanation      : auto-generated narrative sentence

    Attribution (actionable-only world)
    -----------------------------------
    actionable_only_changes : minimal subset of operator-actionable changes that
                              alone make the alternative preferred (empty if none)
    actionable_only_cost    : alternative's cost under those actionable changes
                              (``inf`` if actionable changes cannot make it so)
    actionable_only_delta   : actual_cost − actionable_only_cost (0.0 if infinite)
                              — the saving genuinely attributable to operator action
    """
    query:            str
    actual_path:      NavPath
    alternative_path: NavPath
    changes:          List[PropertyChange]
    cf_cost:          float
    cost_delta:       float
    is_achievable:    bool
    explanation:      str = ""

    actionable_only_changes: List[PropertyChange] = field(default_factory=list)
    actionable_only_cost:    float = math.inf
    actionable_only_delta:   float = 0.0

    def num_changes(self) -> int:
        return len(self.changes)

    def actionable_changes(self) -> List[PropertyChange]:
        """Return the operator-actionable members of the minimal change set."""
        return [c for c in self.changes if c.is_actionable()]

    def non_actionable_changes(self) -> List[PropertyChange]:
        """Return the structural/construction-level members of the minimal set."""
        return [c for c in self.changes if not c.is_actionable()]


# ---------------------------------------------------------------------------
# Counterfactual engine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """
    Generates counterfactual explanations for navigation decisions.

    Constructor parameters
    ----------------------
    reasoner         : AffordanceReasoner — reused for re-evaluation
    planner          : AStarPlanner       — used to evaluate node sequences
    onto_individuals : dict               — live world name → individual mapping
                       (used as the source for cloning)
    config           : NavCostConfig      — numeric policy; defaults to the
                       reasoner's own config so the two cannot drift apart
    """

    def __init__(
        self,
        reasoner:         AffordanceReasoner,
        planner:          AStarPlanner,
        onto_individuals: Dict[str, OntologyIndividual],
        config:           Optional[NavCostConfig] = None,
    ) -> None:
        self.reasoner  = reasoner
        self.planner   = planner
        self.onto_inds = onto_individuals
        # Default to the reasoner's config: a single shared policy means the
        # candidate generator and the affordance rules use the same thresholds.
        self.config    = config or reasoner.config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_why_not(
        self,
        actual_path: NavPath,
        alt_nodes:   List[str],
        agent:       OntologyIndividual,
    ) -> Counterfactual:
        """
        Answer: "Why didn't the robot take the route through *alt_nodes*?"

        Steps
        -----
        1. Evaluate the alternative's cost in the current world.
        2. If already preferred → trivial counterfactual (no changes needed).
        3. Collect the *pool* of candidate single-property fixes from every
           problematic edge (infinite cost, or disproportionately expensive).
        4. Search for the MINIMAL subset of the pool that makes the alternative
           feasible and preferred (Issue-1 minimality).
        5. Independently search the actionable-only sub-pool to decide
           ``is_achievable`` and the attributed saving (Issue-2 attribution).
        """
        alt_path = self.planner.evaluate_sequence(alt_nodes, agent)
        query    = "Why not: " + " → ".join(alt_nodes) + "?"
        cfg      = self.config

        # ── Trivial case: alternative is already preferred ──────────────────
        if alt_path.is_feasible and alt_path.total_cost <= actual_path.total_cost:
            delta = actual_path.total_cost - alt_path.total_cost
            return Counterfactual(
                query            = query,
                actual_path      = actual_path,
                alternative_path = alt_path,
                changes          = [],
                cf_cost          = alt_path.total_cost,
                cost_delta       = delta,
                is_achievable    = True,
                explanation      = (
                    "The alternative path is already feasible and equally or "
                    "more cost-effective — the planner should have chosen it. "
                    "(Possible tie-breaking or graph asymmetry.)"
                ),
                actionable_only_changes = [],
                actionable_only_cost    = alt_path.total_cost,
                actionable_only_delta   = delta,
            )

        # ── Collect the candidate pool from problematic edges ───────────────
        # "Problematic" = infinite cost (hard-blocked) OR cost disproportionate
        # to its distance (e.g. a very crowded corridor).
        # Heuristic limitation: a merely *mildly* costly edge (below the
        # cf_problematic_factor threshold) contributes no candidates, so if such
        # an edge is the deciding margin the reported change set can understate
        # what would flip the decision.  This deliberately targets the edges that
        # dominate the gap rather than every possible micro-optimisation.
        pool: List[PropertyChange] = []
        for edge in alt_path.edges:
            cost, _ = self.reasoner.navigation_cost(
                edge.space_individual, agent, base_distance=edge.distance
            )
            baseline = edge.distance + cfg.cf_problematic_buffer
            if cost == math.inf or cost > cfg.cf_problematic_factor * baseline:
                pool.extend(self._candidate_changes(edge.space_individual, agent))

        target = actual_path.total_cost

        # ── Issue 1: minimal full subset (the reported change set) ──────────
        changes, cf_cost, _preferred = self._minimal_subset(
            alt_path, agent, pool, target, restrict_actionable=False
        )
        cost_delta = (actual_path.total_cost - cf_cost
                      if math.isfinite(cf_cost) else 0.0)

        # ── Issue 2: independent actionable-only search → attribution ───────
        act_changes, act_cost, act_preferred = self._minimal_subset(
            alt_path, agent, pool, target, restrict_actionable=True
        )
        act_delta = (actual_path.total_cost - act_cost
                     if math.isfinite(act_cost) else 0.0)

        explanation = self._narrative(
            actual_path, alt_path, changes, cf_cost,
            act_changes, act_cost, act_preferred,
        )

        return Counterfactual(
            query            = query,
            actual_path      = actual_path,
            alternative_path = alt_path,
            changes          = changes,
            cf_cost          = cf_cost,
            cost_delta       = cost_delta,
            is_achievable    = act_preferred,
            explanation      = explanation,
            actionable_only_changes = act_changes,
            actionable_only_cost    = act_cost,
            actionable_only_delta   = act_delta,
        )

    def batch_why_not(
        self,
        actual_path: NavPath,
        candidates:  List[NavPath],
        agent:       OntologyIndividual,
    ) -> List[Counterfactual]:
        """
        Explain all candidate alternative paths in one call.

        Deduplicates by node sequence so the same alternative is not
        explained twice (Yen's can occasionally produce near-duplicates
        due to floating-point cost ties).
        """
        result:    List[Counterfactual] = []
        seen_seqs: set                  = {tuple(actual_path.nodes)}

        for cand in candidates:
            seq = tuple(cand.nodes)
            if seq not in seen_seqs:
                seen_seqs.add(seq)
                cf = self.explain_why_not(actual_path, list(cand.nodes), agent)
                result.append(cf)

        return result

    # ------------------------------------------------------------------
    # Candidate pool generation
    # ------------------------------------------------------------------

    def _candidate_changes(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
    ) -> List[PropertyChange]:
        """
        Enumerate every single-property fix that *could* help this edge — the
        candidate pool, NOT an applied set.  The minimal-subset search decides
        which of these are actually needed.

        Every condition delegates to a ``NavCostConfig`` predicate (the same one
        the affordance rule fires on), so the pool cannot drift from the rules.
        The entity is read through the reasoner's class-aware view, so inherited
        class defaults (e.g. a Staircase's slope) are visible here too.
        """
        changes: List[PropertyChange] = []
        cfg = self.config
        e   = self.reasoner._entity_view(entity)
        a   = agent.properties

        # ── Hazard (actionable) ─────────────────────────────────────────────
        if e.get("is_hazardous", False):
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="is_hazardous",
                original_value=True, counterfactual_value=False,
                change_type="disable",
                rationale=f"Clearing the hazard makes {entity.name} traversable.",
            ))

        # ── Restricted area (actionable, policy) ────────────────────────────
        if e.get("restricted", False):
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="restricted",
                original_value=True, counterfactual_value=False,
                change_type="enable",
                rationale=f"Granting robot access to the restricted {entity.name}.",
            ))

        # ── Inaccessible (actionable, policy) ───────────────────────────────
        if not e.get("is_accessible", True):
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="is_accessible",
                original_value=False, counterfactual_value=True,
                change_type="enable",
                rationale=f"Marking {entity.name} as accessible.",
            ))

        # ── Door (actionable) ───────────────────────────────────────────────
        door_state = e.get("door_state")   # None if the space has no door
        if door_state in (DoorState.CLOSED.value, DoorState.LOCKED.value):
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="door_state",
                original_value=door_state, counterfactual_value=DoorState.OPEN.value,
                change_type="enable",
                rationale=f"Opening (or unlocking) the {entity.name} door.",
            ))

        # ── Width (structural) ──────────────────────────────────────────────
        if not cfg.fits_width(e, a):
            actual_w = e.get("width", cfg.default_space_width)
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="width",
                original_value=round(actual_w, 2),
                counterfactual_value=round(cfg.required_width(a)
                                           + cfg.cf_width_safety_margin, 2),
                change_type="improve",
                rationale=f"Widening {entity.name} to accommodate robot + clearance.",
            ))

        # ── Slope (structural) ──────────────────────────────────────────────
        if (not cfg.slope_within_limit(e, a)) or cfg.wheeled_on_stairs(e, a):
            max_slope = a.get("max_slope_angle", cfg.default_max_slope_angle)
            cap = max_slope
            if a.get("mobility_type", MobilityType.WHEELED.value) == MobilityType.WHEELED.value:
                # also clear the wheeled-on-stairs hard limit
                cap = min(cap, cfg.wheeled_stair_slope_limit)
            slope = e.get("slope_angle", cfg.default_slope_angle)
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="slope_angle",
                original_value=round(slope, 1),
                counterfactual_value=round(cap - cfg.cf_slope_margin, 1),
                change_type="improve",
                rationale=(f"Installing a ramp that reduces {entity.name} slope "
                           f"to within robot limits."),
            ))

        # ── Crowd density (actionable, soft cost) ───────────────────────────
        crowd = e.get("crowd_density", cfg.default_crowd_density)
        if crowd > cfg.crowd_block_threshold:
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="crowd_density",
                original_value=round(crowd, 2), counterfactual_value=cfg.cf_crowd_target,
                change_type="improve",
                rationale=f"Reducing crowd density in {entity.name} to enable safe transit.",
            ))

        # ── Illumination (actionable, soft cost) ────────────────────────────
        if not cfg.lit_enough(e):
            illum = e.get("illumination", cfg.default_illumination)
            changes.append(PropertyChange(
                individual_name=entity.name, property_name="illumination",
                original_value=round(illum, 2), counterfactual_value=cfg.cf_illumination_target,
                change_type="improve",
                rationale=f"Increasing lighting in {entity.name} for reliable perception.",
            ))

        return changes

    # ------------------------------------------------------------------
    # Minimal-subset search
    # ------------------------------------------------------------------

    def _minimal_subset(
        self,
        alt_path:            NavPath,
        agent:               OntologyIndividual,
        pool:                List[PropertyChange],
        target_cost:         float,
        restrict_actionable: bool = False,
    ) -> Tuple[List[PropertyChange], float, bool]:
        """
        Find the smallest-cardinality subset of *pool* that makes *alt_path*
        feasible AND no more expensive than *target_cost*.

        Returns ``(changes, cf_cost, preferred)``:
          - the minimal satisfying subset and its cost, ``preferred=True``; or
          - if no subset is preferred, the cheapest FEASIBLE subset found,
            ``preferred=False`` (may be the empty set); or
          - ``([], inf, False)`` if even the full pool cannot make it feasible.

        Minimum cardinality ⇒ subset-minimal: a smaller cardinality was tried
        exhaustively first, so no proper subset of the result also satisfies.
        Ties are broken by (resulting cost, total effort) — order-independent.
        """
        if restrict_actionable:
            pool = [c for c in pool if c.is_actionable()]
        pool = self._dedupe(pool)
        n = len(pool)

        empty_cost = self._evaluate_subset(alt_path, agent, [])
        if n == 0:
            preferred = math.isfinite(empty_cost) and empty_cost <= target_cost
            return ([], empty_cost, preferred)

        # Guard against pathological pools: exhaustive search is 2^n.
        if n > self.config.cf_max_pool:
            return self._reduce_from_full(alt_path, agent, pool, target_cost)

        # Seed "best feasible" with the do-nothing world if it is already
        # feasible, so we never report changes when none are needed.
        best_feasible: Optional[Tuple[List[PropertyChange], float]] = (
            ([], empty_cost) if math.isfinite(empty_cost) else None
        )

        for k in range(1, n + 1):                       # ascending cardinality
            winners: List[Tuple[List[PropertyChange], float]] = []
            for combo in itertools.combinations(pool, k):
                subset = list(combo)
                cf = self._evaluate_subset(alt_path, agent, subset)
                if math.isfinite(cf):
                    if best_feasible is None or cf < best_feasible[1]:
                        best_feasible = (subset, cf)
                    if cf <= target_cost:               # feasible AND preferred
                        winners.append((subset, cf))
            if winners:
                winners.sort(key=lambda w: (
                    w[1], sum(_EFFORT_RANK[c.effort()] for c in w[0])
                ))
                return (winners[0][0], winners[0][1], True)

        if best_feasible is not None:
            return (best_feasible[0], best_feasible[1], False)
        return ([], math.inf, False)

    def _reduce_from_full(
        self,
        alt_path:    NavPath,
        agent:       OntologyIndividual,
        pool:        List[PropertyChange],
        target_cost: float,
    ) -> Tuple[List[PropertyChange], float, bool]:
        """
        Subset-minimal fallback for pools too large for exhaustive search.

        Every candidate change is **monotone**: it is feasibility-non-decreasing
        and cost-non-increasing (clearing a blocker can only help feasibility;
        widening, regrading, opening a door, de-crowding, or lighting a space can
        only lower or keep the cost — width is not even in the cost formula).
        Therefore applying the *whole* pool yields the maximum feasibility and the
        minimum achievable cost: if the full pool is not feasible-and-preferred,
        no subset is either.

        Given monotonicity, a single backward-elimination pass from the full set
        yields a subset-minimal result: try to drop each change (highest-effort /
        structural first, so the kept set stays as actionable as possible); keep
        the drop whenever the goal still holds.  This is robust to
        jointly-required blockers that a forward-greedy search cannot assemble
        (no single addition improves an all-infinite path).
        """
        full_cost = self._evaluate_subset(alt_path, agent, pool)
        if not math.isfinite(full_cost):
            return ([], math.inf, False)   # even everything cannot make it feasible

        preferred = full_cost <= target_cost
        # Goal to preserve while eliminating: feasibility, plus preference iff the
        # full set was preferred (the best we can do, by monotonicity).
        selected = list(pool)
        current = full_cost
        order = sorted(pool, key=lambda c: -_EFFORT_RANK[c.effort()])
        for c in order:
            trial = [x for x in selected if x is not c]
            cf = self._evaluate_subset(alt_path, agent, trial)
            ok = math.isfinite(cf) and (cf <= target_cost if preferred else True)
            if ok:
                selected = trial
                current = cf
        return (selected, current, preferred)

    # ------------------------------------------------------------------
    # World simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedupe(pool: List[PropertyChange]) -> List[PropertyChange]:
        """Drop duplicate atoms (same individual + property + target value)."""
        seen: set = set()
        uniq: List[PropertyChange] = []
        for c in pool:
            key = (c.individual_name, c.property_name, repr(c.counterfactual_value))
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def _build_overrides(
        self,
        changes: List[PropertyChange],
    ) -> Dict[str, OntologyIndividual]:
        """
        Group *changes* by individual; clone each affected live individual once
        (clones are graph-less, so the live world and RDF graph are untouched)
        and apply the property mutations to the clone.
        """
        overrides: Dict[str, OntologyIndividual] = {}
        for change in changes:
            if change.individual_name not in overrides:
                orig = self.onto_inds.get(change.individual_name)
                if orig is None:
                    continue
                overrides[change.individual_name] = orig.clone()
            if change.individual_name in overrides:
                overrides[change.individual_name].set(
                    change.property_name, change.counterfactual_value
                )
        return overrides

    def _evaluate_subset(
        self,
        alt_path: NavPath,
        agent:    OntologyIndividual,
        subset:   List[PropertyChange],
    ) -> float:
        """Total cost of *alt_path* with *subset* applied to cloned individuals."""
        return self._evaluate_with_overrides(
            alt_path, agent, self._build_overrides(subset)
        )

    def _evaluate_with_overrides(
        self,
        path:      NavPath,
        agent:     OntologyIndividual,
        overrides: Dict[str, OntologyIndividual],
    ) -> float:
        """
        Evaluate path cost using counterfactually modified individuals.

        For each edge, if the edge's space individual has a counterfactual
        clone in *overrides*, use the clone; otherwise use the original.  Any
        single impassable edge makes the whole path cost infinite.
        """
        total = 0.0
        for edge in path.edges:
            entity = overrides.get(edge.space_individual.name, edge.space_individual)
            cost, _ = self.reasoner.navigation_cost(entity, agent, edge.distance)
            if cost == math.inf:
                return math.inf
            total += cost
        return total

    # ------------------------------------------------------------------
    # Narrative
    # ------------------------------------------------------------------

    def _narrative(
        self,
        actual:        NavPath,
        alt:           NavPath,
        changes:       List[PropertyChange],
        cf_cost:       float,
        act_changes:   List[PropertyChange],
        act_cost:      float,
        act_preferred: bool,
    ) -> str:
        """
        Generate a one-paragraph English explanation.

        Crucially, the advertised saving is attributed honestly: if the
        alternative only becomes preferred after structural (non-actionable)
        changes, the narrative says so rather than implying an operator can
        unlock the saving alone.
        """
        actual_label = (
            " → ".join(actual.nodes) if actual.nodes else "⚠ infeasible actual path"
        )
        alt_label = " → ".join(alt.nodes)
        actual_cost_label = (
            f"{actual.total_cost:.2f}"
            if actual.total_cost < math.inf
            else "∞ (infeasible)"
        )

        # No changes needed / found.
        if not changes:
            if not alt.is_feasible:
                return (
                    f"Robot chose {actual_label} over {alt_label} because the "
                    f"alternative is infeasible with no identifiable property fix."
                )
            return (
                f"Robot chose {actual_label} (cost {actual_cost_label}) over "
                f"{alt_label} (cost {alt.total_cost:.2f}) because the chosen route "
                f"is already cheaper.  No world changes would make the alternative "
                f"preferred — this is the optimal decision."
            )

        actual_feasible = math.isfinite(actual.total_cost)
        reason = ", ".join(c.description() for c in changes[:3])
        head = (
            f"Robot chose {actual_label} over {alt_label} because the following "
            f"would have to change for the alternative to be preferred: {reason}."
        )

        if act_preferred and math.isfinite(act_cost):
            # Operator-actionable changes alone suffice.
            act_reason = ", ".join(c.description() for c in act_changes)
            if actual_feasible:
                saving = actual.total_cost - act_cost
                tail = (f"applying [{act_reason}] alone makes the alternative cost "
                        f"{act_cost:.2f} vs {actual_cost_label} "
                        f"(saving {saving:.2f} units).")
            else:
                tail = (f"applying [{act_reason}] alone makes the otherwise-"
                        f"infeasible alternative feasible at cost {act_cost:.2f}.")
            return f"{head}  These are operator-actionable: {tail}"

        if math.isfinite(cf_cost) and cf_cost <= actual.total_cost:
            # Preferred (or feasible) only with structural help — say so explicitly.
            if actual_feasible:
                saving = actual.total_cost - cf_cost
                detail = (f"The {saving:.2f}-unit saving (to cost {cf_cost:.2f}) "
                          f"requires structural changes")
            else:
                detail = (f"The alternative only becomes feasible (cost "
                          f"{cf_cost:.2f}) after structural changes")
            return (
                f"{head}  {detail} an operator cannot make alone (e.g. widening "
                f"or regrading); it is NOT achievable by runtime action."
            )

        if math.isfinite(cf_cost):
            return (
                f"{head}  Even with these changes the alternative would cost "
                f"{cf_cost:.2f}, still above the chosen route — so it remains "
                f"the worse option."
            )

        return (
            f"{head}  Even after these changes the alternative remains infeasible."
        )
