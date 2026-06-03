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

This implements the *minimal-change principle* from counterfactual logic
(Lewis, 1973; Pearl, 2000): among all possible world states in which the
alternative path is preferred, find the one that differs least from the
actual world.

Algorithm
---------
For each edge on the alternative path:
  1. If cost == inf (hard-blocked) or cost is disproportionately high:
       → call _minimal_changes() to find the smallest set of property
         mutations that would make the edge traversable / cheaper.
  2. Apply those mutations to deep copies of the relevant individuals
     (counterfactual world — does not mutate the live world model).
  3. Re-evaluate the alternative path cost in the counterfactual world.
  4. Compute cost delta = actual_cost − cf_cost.
  5. Filter changes by actionability (can an operator do this today?).

PropertyChange.is_actionable() separates:
  - Actionable (low effort): open a door, clear a crowd, fix lighting
  - Structural (high effort): widen a corridor, reduce a slope

This distinction is critical for generating *useful* recommendations —
telling a user to "widen the corridor" is less useful than "open the door."
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .affordance import AffordanceReasoner, AffordanceResult
from .domain import AffordanceType, DoorState, MobilityType, SurfaceType
from .navigation import NavEdge, NavPath, AStarPlanner
from .ontology import OntologyIndividual


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

        Actionable properties: door state, access flags, crowd density,
        illumination, obstacle density — all mutable without structural work.

        Non-actionable: corridor width, slope angle — require construction.
        """
        ACTIONABLE = {
            "door_state",
            "is_accessible",
            "is_hazardous",
            "restricted",
            "obstacle_density",
            "crowd_density",
            "illumination",
        }
        return self.property_name in ACTIONABLE

    def effort(self) -> str:
        """
        Qualitative effort estimate for human operators.

        HIGH   — requires physical construction (months, high cost)
        MEDIUM — requires policy / admin change (days, moderate cost)
        LOW    — immediate operator action (minutes, no cost)
        """
        HIGH   = {"width", "slope_angle"}
        MEDIUM = {"surface_type", "is_accessible", "restricted"}
        if self.property_name in HIGH:
            return "high"
        if self.property_name in MEDIUM:
            return "medium"
        return "low"


@dataclass
class Counterfactual:
    """
    The result of a "why not path X?" query.

    Encapsulates the actual path taken, the alternative being questioned,
    the minimal property changes that would have altered the decision,
    and derived metrics (cost delta, achievability).

    Attributes
    ----------
    query            : natural language question that prompted this analysis
    actual_path      : the path the robot actually chose
    alternative_path : the path being questioned
    changes          : minimal set of PropertyChange objects
    cf_cost          : cost of alternative in the counterfactual world
    cost_delta       : actual_cost − cf_cost (positive = alternative now cheaper)
    is_achievable    : True if at least one change is actionable
    explanation      : auto-generated narrative sentence
    """
    query:            str
    actual_path:      NavPath
    alternative_path: NavPath
    changes:          List[PropertyChange]
    cf_cost:          float
    cost_delta:       float
    is_achievable:    bool
    explanation:      str = ""

    def num_changes(self) -> int:
        return len(self.changes)

    def actionable_changes(self) -> List[PropertyChange]:
        """Return only the changes an operator can make at runtime."""
        return [c for c in self.changes if c.is_actionable()]

    def non_actionable_changes(self) -> List[PropertyChange]:
        """Return only the structural/construction-level changes."""
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
    """

    def __init__(
        self,
        reasoner:         AffordanceReasoner,
        planner:          AStarPlanner,
        onto_individuals: Dict[str, OntologyIndividual],
    ) -> None:
        self.reasoner  = reasoner
        self.planner   = planner
        self.onto_inds = onto_individuals

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
        3. Identify problematic edges (inf cost or > 1.5× baseline).
        4. Generate minimal changes for each problematic edge.
        5. Apply changes to cloned individuals → re-evaluate → compute delta.
        6. Return Counterfactual with full explanation.

        Parameters
        ----------
        actual_path : the path the robot actually took
        alt_nodes   : the node sequence of the alternative to explain
        agent       : the robot agent individual
        """
        alt_path = self.planner.evaluate_sequence(alt_nodes, agent)
        query    = "Why not: " + " → ".join(alt_nodes) + "?"

        # ── Trivial case: alternative is already preferred ──────────────────
        # This can happen when the planner's k-shortest diverges from the user's
        # expectation, or when the world changed between planning and querying.
        if alt_path.is_feasible and alt_path.total_cost <= actual_path.total_cost:
            return Counterfactual(
                query            = query,
                actual_path      = actual_path,
                alternative_path = alt_path,
                changes          = [],
                cf_cost          = alt_path.total_cost,
                cost_delta       = actual_path.total_cost - alt_path.total_cost,
                is_achievable    = True,
                explanation      = (
                    "The alternative path is already feasible and equally or "
                    "more cost-effective — the planner should have chosen it. "
                    "(Possible tie-breaking or graph asymmetry.)"
                ),
            )

        # ── Identify problematic edges ──────────────────────────────────────
        # "Problematic" = infinite cost (hard-blocked) OR cost > 1.5× the
        # sum of edge distance + a 2-unit door-opening buffer.  The 1.5× factor
        # catches edges that are technically passable but disproportionately
        # expensive (e.g. extremely crowded corridors).
        changes: List[PropertyChange] = []
        for edge in alt_path.edges:
            cost, af = self.reasoner.navigation_cost(
                edge.space_individual, agent, base_distance=edge.distance
            )
            baseline = edge.distance + 2.0   # 2.0 = door-opening buffer
            if cost == math.inf or cost > 1.5 * baseline:
                edge_changes = self._minimal_changes(edge.space_individual, agent, af)
                changes.extend(edge_changes)

        # ── Simulate counterfactual world ───────────────────────────────────
        # Clone each affected individual (so the live world is NOT mutated)
        # and apply the proposed property changes to the clones.
        cf_overrides: Dict[str, OntologyIndividual] = {}
        for change in changes:
            if change.individual_name not in cf_overrides:
                orig = self.onto_inds.get(change.individual_name)
                if orig:
                    # Deep clone isolates this counterfactual world from others
                    cf_overrides[change.individual_name] = orig.clone()
            if change.individual_name in cf_overrides:
                cf_overrides[change.individual_name].set(
                    change.property_name, change.counterfactual_value
                )

        # ── Re-evaluate alternative path cost in counterfactual world ───────
        cf_cost    = self._evaluate_with_overrides(alt_path, agent, cf_overrides)
        cost_delta = actual_path.total_cost - cf_cost
        is_achievable = any(c.is_actionable() for c in changes)
        explanation   = self._narrative(actual_path, alt_path, changes, cf_cost, cost_delta)

        return Counterfactual(
            query            = query,
            actual_path      = actual_path,
            alternative_path = alt_path,
            changes          = changes,
            cf_cost          = cf_cost,
            cost_delta       = cost_delta,
            is_achievable    = is_achievable,
            explanation      = explanation,
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _minimal_changes(
        self,
        entity: OntologyIndividual,
        agent:  OntologyIndividual,
        af:     AffordanceResult,
    ) -> List[PropertyChange]:
        """
        Generate the smallest set of property mutations that would make
        *entity* traversable and reasonably cheap for *agent*.

        Strategy: check each blocking condition in priority order (cheapest
        actionable fix first).  We do NOT solve the full optimization problem —
        this is a greedy heuristic that works well for the typical cases.

        Priority order:
          1. Hazard removal (enable traversal)
          2. Access grant   (enable traversal)
          3. Door opening   (enable traversal)
          4. Geometry fix   (enable traversal — structural)
          5. Slope fix      (enable traversal — structural)
          6. Crowd reduction (reduce cost)
          7. Lighting fix   (reduce cost)
        """
        changes: List[PropertyChange] = []
        e = entity.properties
        a = agent.properties

        # ── Priority 1: Hazard ──────────────────────────────────────────────
        if e.get("is_hazardous", False):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "is_hazardous",
                original_value       = True,
                counterfactual_value = False,
                change_type          = "disable",
                rationale            = f"Clearing the hazard makes {entity.name} traversable.",
            ))

        # ── Priority 2a: Restricted area ───────────────────────────────────
        if e.get("restricted", False):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "restricted",
                original_value       = True,
                counterfactual_value = False,
                change_type          = "enable",
                rationale            = f"Granting robot access to the restricted area.",
            ))

        # ── Priority 2b: Inaccessible ──────────────────────────────────────
        if not e.get("is_accessible", True):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "is_accessible",
                original_value       = False,
                counterfactual_value = True,
                change_type          = "enable",
                rationale            = f"Marking {entity.name} as accessible.",
            ))

        # ── Priority 3: Door state ─────────────────────────────────────────
        door_state = e.get("door_state")   # None if space has no door
        if door_state in (DoorState.CLOSED.value, DoorState.LOCKED.value):
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "door_state",
                original_value       = door_state,
                counterfactual_value = DoorState.OPEN.value,
                change_type          = "enable",
                rationale            = f"Opening (or unlocking) the {entity.name} door.",
            ))

        # ── Priority 4: Width (structural) ─────────────────────────────────
        # Required width = robot_width + 2 × min_clearance (one buffer per side)
        robot_w   = a.get("robot_width",   0.5)
        clearance = a.get("min_clearance", 0.15)
        required  = robot_w + 2 * clearance
        actual_w  = e.get("width", 10.0)
        if actual_w < required:
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "width",
                original_value       = round(actual_w, 2),
                # Add 0.1 m safety margin above the bare minimum
                counterfactual_value = round(required + 0.1, 2),
                change_type          = "improve",
                rationale            = (
                    f"Widening {entity.name} to accommodate robot + clearance."
                ),
            ))

        # ── Priority 5: Slope (structural) ─────────────────────────────────
        slope     = e.get("slope_angle", 0.0)
        max_slope = a.get("max_slope_angle", 8.0)
        if slope > max_slope:
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "slope_angle",
                original_value       = round(slope, 1),
                # Reduce to 1° below robot's limit to ensure rule fires
                counterfactual_value = round(max_slope - 1.0, 1),
                change_type          = "improve",
                rationale            = (
                    f"Installing a ramp that reduces {entity.name} slope "
                    f"to within robot limits."
                ),
            ))

        # ── Priority 6: Crowd density (actionable, soft cost) ───────────────
        # Only flag crowd as a blocker if it is very high (> 50%).
        # Lower densities are already captured in the cost formula without
        # making the path infeasible.
        crowd = e.get("crowd_density", 0.0)
        if crowd > 0.5:
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "crowd_density",
                original_value       = round(crowd, 2),
                counterfactual_value = 0.2,   # reduce to low-crowd level
                change_type          = "improve",
                rationale            = (
                    f"Reducing crowd density in {entity.name} to enable safe transit."
                ),
            ))

        # ── Priority 7: Illumination (actionable, soft cost) ────────────────
        illum = e.get("illumination", 1.0)
        if illum <= 0.25:   # matches the OBSERVABLE block threshold in rules
            changes.append(PropertyChange(
                individual_name      = entity.name,
                property_name        = "illumination",
                original_value       = round(illum, 2),
                counterfactual_value = 0.6,   # well-lit but not maximum
                change_type          = "improve",
                rationale            = (
                    f"Increasing lighting in {entity.name} for reliable perception."
                ),
            ))

        return changes

    def _evaluate_with_overrides(
        self,
        path:      NavPath,
        agent:     OntologyIndividual,
        overrides: Dict[str, OntologyIndividual],
    ) -> float:
        """
        Evaluate path cost using counterfactually modified individuals.

        For each edge, if the edge's space individual has a counterfactual
        clone in *overrides*, use the clone; otherwise use the original.
        This allows evaluating a mix of changed and unchanged spaces.
        """
        total = 0.0
        for edge in path.edges:
            # Use the counterfactual clone if available, else the live individual
            entity = overrides.get(edge.space_individual.name, edge.space_individual)
            cost, _ = self.reasoner.navigation_cost(entity, agent, edge.distance)
            total += cost
        return total

    def _narrative(
        self,
        actual:     NavPath,
        alt:        NavPath,
        changes:    List[PropertyChange],
        cf_cost:    float,
        cost_delta: float,
    ) -> str:
        """
        Generate a one-paragraph English explanation for the counterfactual.

        Four cases:
          1. No changes, alternative feasible but costlier → "optimal decision"
          2. No changes, alternative infeasible → "no fixable property"
          3. Changes found, counterfactual improves cost → include savings
          4. Changes found, still infeasible after changes → note residual issue
        """
        # Format actual path label, handling the case where it is infeasible
        actual_label = (
            " → ".join(actual.nodes)
            if actual.nodes
            else "⚠ infeasible actual path"
        )
        actual_cost_label = (
            f"{actual.total_cost:.2f}"
            if actual.total_cost < math.inf
            else "∞ (infeasible)"
        )

        if not changes and not alt.is_feasible:
            # Alternative is hard-blocked and we found no fixable property
            return (
                f"Robot chose {actual_label} over {' → '.join(alt.nodes)} "
                f"because the alternative is infeasible with no identifiable "
                f"single-property fix."
            )

        if not changes:
            # Alternative is feasible but more expensive → correct decision
            return (
                f"Robot chose {actual_label} (cost {actual_cost_label}) over "
                f"{' → '.join(alt.nodes)} (cost {alt.total_cost:.2f}) because "
                f"the chosen route is already {abs(cost_delta):.2f} cost units "
                f"cheaper.  No world changes are needed — this is the optimal decision."
            )

        # Summarise the top-3 changes in the narrative
        reason_parts  = [c.description() for c in changes[:3]]
        reason        = ", ".join(reason_parts)

        if cost_delta > 0 and cf_cost < math.inf:
            # After changes, alternative becomes cheaper
            suffix = (
                f"  If corrected, the alternative would cost {cf_cost:.2f} vs "
                f"actual {actual_cost_label} (saving {cost_delta:.2f} units)."
            )
        elif cf_cost == math.inf:
            # Even with changes the alternative remains blocked
            suffix = (
                "  Even after these minimal changes the alternative remains infeasible."
            )
        else:
            suffix = ""

        return (
            f"Robot chose {actual_label} over {' → '.join(alt.nodes)} because "
            f"the following properties block or penalise the route: {reason}.{suffix}"
        )
