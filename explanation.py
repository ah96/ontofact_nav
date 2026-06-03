"""
ontofact_nav/explanation.py
===========================
Natural language explanation generator.

Three-layer explanation model
------------------------------
Layer 1 — PATH RATIONALE
  One English sentence per traversed segment.  Template-selected based on the
  most salient affordance feature of that segment (emergency route, door state,
  crowd level, surface type, …).

Layer 2 — COUNTERFACTUAL ANALYSIS
  For each alternative path analysed: the query, the alternative's cost, the
  required world changes, and the cost delta in the counterfactual world.

Layer 3 — OPERATOR RECOMMENDATIONS
  De-duplicated, effort-labelled actionable changes an operator can make to
  unlock alternative routes.  Only actionable changes are included (door state,
  crowd density, illumination, …) — structural changes (corridor width) appear
  in the counterfactual section but not in recommendations.

Design notes
------------
- Templates are stored as module-level dicts so subclasses / plugins can
  extend them without touching the class implementation.
- The format_report() method is a pure function of a NavigationExplanation
  object — it performs no inference itself.
- Property template selection in _recommendations() uses exact key matching
  on PropertyChange.property_name.  Add entries to _CF_PROPERTY_TEMPLATES to
  support new properties.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .affordance import AffordanceResult
from .counterfactual import Counterfactual, PropertyChange
from .domain import AffordanceType, DoorState, SurfaceType
from .navigation import NavPath
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Structured explanation container
# ---------------------------------------------------------------------------

@dataclass
class NavigationExplanation:
    """
    All explanation artefacts for a single navigation episode.

    Produced by ExplanationGenerator.generate() and consumed by
    ExplanationGenerator.format_report() for human display.

    Attributes
    ----------
    robot_name          : name of the navigating agent
    start / goal        : start and goal node IDs
    chosen_path         : the NavPath selected by the planner
    path_rationale      : one sentence per traversed segment (Layer 1)
    affordance_warnings : blocked-affordance alerts (informational)
    counterfactuals     : per-alternative Counterfactual objects (Layer 2)
    recommendations     : actionable operator suggestions (Layer 3)
    executive_summary   : one-paragraph overview
    """
    robot_name:          str
    start:               str
    goal:                str
    chosen_path:         NavPath
    path_rationale:      List[str]
    affordance_warnings: List[str]
    counterfactuals:     List[Counterfactual]
    recommendations:     List[str]
    executive_summary:   str


# ---------------------------------------------------------------------------
# Segment rationale templates
# ---------------------------------------------------------------------------

# Each entry is selected by matching the most salient property of the segment.
# Entries are checked in the order listed in _segment_rationale(), so more
# specific conditions should appear higher in the match chain.
_SEGMENT_TEMPLATES: Dict[str, str] = {
    # Designated emergency route — always preferred if on-path
    "emergency":  "Chose emergency-designated route through {name}.",
    # Door was already open — no action required
    "door_open":  "Passed through {name} — door was open.",
    # Door was closed; robot opened it with its arm
    "door_opened":"Opened {name} door with manipulation arm (+2.5 cost units).",
    # Crowd is heavy but traversable — note the extra caution
    "crowded":    "Navigated {name} despite high crowd density ({crowd:.0%}).",
    # Wet floor requires reduced speed
    "wet":        "Cautiously traversed wet surface in {name}.",
    # Low illumination forces conservative navigation
    "dark":       "Navigated {name} with low illumination — localisation uncertainty added.",
    # Width is close to robot limit — squeeze through
    "narrow":     "Squeezed through {name} (width {width:.1f} m, clearance OK).",
    # Generic: list the granted affordances
    "default":    "Traversed {name} [{affordances}].",
}

# ---------------------------------------------------------------------------
# Counterfactual recommendation templates (per property)
# ---------------------------------------------------------------------------

# Maps a PropertyChange.property_name to an English sentence template.
# Placeholders: {entity} = individual name, {cf_value} = new value,
#               {orig_value} = original value.
# Format strings use both .0% (percentage) and :.2f (decimal) depending on
# the property's natural representation.
_CF_PROPERTY_TEMPLATES: Dict[str, str] = {
    "door_state": (
        "If the door at {entity} were {cf_value} (currently {orig_value}), "
        "the robot could use that route."
    ),
    "is_hazardous": (
        "If hazards in {entity} were cleared, it would become traversable."
    ),
    "restricted": (
        "If robot access to {entity} were granted (currently restricted), "
        "that route would be available."
    ),
    "width": (
        "If {entity} were widened to {cf_value} m (currently {orig_value} m), "
        "the robot would fit through."
    ),
    "crowd_density": (
        # Use .0% format for percentages — nicer than raw decimal
        "If crowd density in {entity} dropped to {cf_value:.0%} "
        "(currently {orig_value:.0%}), that route would be preferred."
    ),
    "is_accessible": (
        "If {entity} were made accessible (currently closed to robot), "
        "the route becomes viable."
    ),
    "obstacle_density": (
        "Clearing obstacles in {entity} to {cf_value:.0%} density "
        "(currently {orig_value:.0%}) would open that path."
    ),
    "illumination": (
        "Improving lighting in {entity} to {cf_value:.0%} "
        "(currently {orig_value:.0%}) would reduce navigation uncertainty."
    ),
    "slope_angle": (
        "If {entity} slope were reduced to {cf_value}° "
        "(currently {orig_value}°) — e.g. by installing a ramp — "
        "the wheeled robot could traverse it."
    ),
}


# ---------------------------------------------------------------------------
# Explanation generator
# ---------------------------------------------------------------------------

class ExplanationGenerator:
    """
    Translates planning artefacts (NavPath, Counterfactual list) into a
    structured, human-readable NavigationExplanation.

    All inference is done by the planner and counterfactual engine.
    This class handles formatting and (optionally) Claude-powered NLG.

    Parameters
    ----------
    use_claude : bool
        When True, ``generate_narrative()`` and ``format_report(use_claude=True)``
        call the Claude API to produce a richer prose explanation alongside the
        structured template output.
    model : str
        Claude model ID.  Defaults to Haiku for low latency; swap to Sonnet/Opus
        for higher quality.
    """

    _SYSTEM_PROMPT = (
        "You are a robot navigation explainability assistant for facility operators. "
        "Given structured JSON data about a robot's navigation decision, write a "
        "concise 2–3 paragraph prose explanation (under 160 words). "
        "Cover: why the chosen route was selected, what the key constraints were, "
        "and the most important actionable recommendation (if any). "
        "Be specific — mention corridor names, cost values, and door states. "
        "No bullet points, no headers, plain prose only."
    )

    def __init__(
        self,
        use_claude: bool = False,
        model:      str  = "claude-haiku-4-5-20251001",
    ) -> None:
        self._use_claude = use_claude
        self._model      = model
        self._client     = None   # lazily initialised on first Claude call

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        robot:           OntologyIndividual,
        start:           str,
        goal:            str,
        chosen_path:     NavPath,
        candidates:      List[NavPath],
        counterfactuals: List[Counterfactual],
    ) -> NavigationExplanation:
        """
        Produce a NavigationExplanation from planning artefacts.

        Parameters
        ----------
        robot           : the navigating robot individual (for its property dict)
        start / goal    : source and destination node IDs
        chosen_path     : the path the planner selected
        candidates      : alternative paths considered (from Yen's algorithm)
        counterfactuals : pre-computed Counterfactual objects for each candidate
        """
        rationale = self._segment_rationale(chosen_path, robot.properties)
        warnings  = self._affordance_warnings(chosen_path)
        recs      = self._recommendations(counterfactuals)
        summary   = self._executive_summary(robot, start, goal, chosen_path, counterfactuals)

        return NavigationExplanation(
            robot_name          = robot.name,
            start               = start,
            goal                = goal,
            chosen_path         = chosen_path,
            path_rationale      = rationale,
            affordance_warnings = warnings,
            counterfactuals     = counterfactuals,
            recommendations     = recs,
            executive_summary   = summary,
        )

    # ------------------------------------------------------------------
    # Layer 1: per-segment rationale
    # ------------------------------------------------------------------

    def _segment_rationale(
        self,
        path:        NavPath,
        agent_props: Dict[str, Any],
    ) -> List[str]:
        """
        Generate one English sentence per traversed segment.

        Template selection priority (highest first):
          emergency route → door state → crowd → wet → dark → narrow → default

        The *fmt* dict is built by merging agent properties, entity properties,
        and computed aliases (crowd = crowd_density) to fill template placeholders.
        Entity properties take precedence so {width} refers to the space's width.
        """
        sentences: List[str] = []

        for edge, af in zip(path.edges, path.affordance_results):
            entity = edge.space_individual
            e      = entity.properties
            afford = [a.value for a in af.affordances]
            name   = entity.name

            # Template selection — checked in priority order
            if e.get("emergency_route", False):
                tmpl = _SEGMENT_TEMPLATES["emergency"]
            elif e.get("door_state") == DoorState.OPEN.value:
                tmpl = _SEGMENT_TEMPLATES["door_open"]
            elif e.get("door_state") == DoorState.CLOSED.value:
                tmpl = _SEGMENT_TEMPLATES["door_opened"]
            elif e.get("crowd_density", 0.0) > 0.6:
                tmpl = _SEGMENT_TEMPLATES["crowded"]
            elif e.get("surface_type") == SurfaceType.WET.value:
                tmpl = _SEGMENT_TEMPLATES["wet"]
            elif e.get("illumination", 1.0) <= 0.25:
                tmpl = _SEGMENT_TEMPLATES["dark"]
            elif (
                e.get("width", 10.0)
                <= agent_props.get("robot_width", 0.5)
                   + 2 * agent_props.get("min_clearance", 0.15) + 0.3
            ):
                # The "+ 0.3" buffer catches widths that are technically passable
                # but still notably snug (within 30 cm of the minimum requirement).
                tmpl = _SEGMENT_TEMPLATES["narrow"]
            else:
                tmpl = _SEGMENT_TEMPLATES["default"]

            # Build the format dict: merge agent → entity → computed aliases.
            # Entity properties come last so they win on key conflicts.
            fmt: Dict[str, Any] = {
                "name":       name,
                "affordances": ", ".join(afford) if afford else "standard",
                "crowd":      e.get("crowd_density", 0.0),  # alias for template
            }
            fmt.update(e)   # adds width, illumination, door_state, etc.

            try:
                sentence = tmpl.format(**fmt)
            except (KeyError, ValueError):
                # Graceful fallback if any template key is missing
                sentence = f"Traversed {name}."

            sentences.append(sentence)

        return sentences

    # ------------------------------------------------------------------
    # Affordance warnings
    # ------------------------------------------------------------------

    def _affordance_warnings(self, path: NavPath) -> List[str]:
        """
        Report blocked affordances on the chosen path.

        These are informational: a blocked affordance does not necessarily
        prevent traversal (e.g. OPENABLE blocked on a doorless corridor is a
        non-event; OBSERVABLE blocked adds a cost penalty but doesn't stop the
        robot).  The warning is included for human situational awareness.
        """
        warnings: List[str] = []
        for af in path.affordance_results:
            for blocked, reason in af.blocking_reasons.items():
                warnings.append(
                    f"[⚠ {af.entity_name}] {blocked.value} blocked — {reason}"
                )
        return warnings

    # ------------------------------------------------------------------
    # Layer 3: operator recommendations
    # ------------------------------------------------------------------

    def _recommendations(self, counterfactuals: List[Counterfactual]) -> List[str]:
        """
        Collect actionable property changes across all counterfactuals,
        de-duplicate them (same individual + property = same recommendation),
        format using per-property English templates, and label with effort level.

        Only actionable changes are included.  Structural changes (corridor width,
        slope angle) require construction and are reported only in the
        counterfactual section, not here, to avoid overwhelming the operator.
        """
        recs: List[str] = []
        seen: Set[str]  = set()   # tracks (individual_name, property_name) pairs

        for cf in counterfactuals:
            for change in cf.actionable_changes():
                # De-duplicate: same property on same individual = same action
                key = f"{change.individual_name}:{change.property_name}"
                if key in seen:
                    continue
                seen.add(key)

                tmpl   = _CF_PROPERTY_TEMPLATES.get(change.property_name)
                effort = change.effort()

                if tmpl:
                    try:
                        text = tmpl.format(
                            entity    = change.individual_name,
                            cf_value  = change.counterfactual_value,
                            orig_value = change.original_value,
                        )
                        recs.append(f"[{effort.upper()} effort] {text}")
                    except (KeyError, ValueError):
                        # Fallback if template substitution fails
                        recs.append(
                            f"[{effort.upper()} effort] Modify "
                            f"{change.individual_name}.{change.property_name}: "
                            f"{change.original_value!r} → {change.counterfactual_value!r}."
                        )

        return recs

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _executive_summary(
        self,
        robot:           OntologyIndividual,
        start:           str,
        goal:            str,
        path:            NavPath,
        counterfactuals: List[Counterfactual],
    ) -> str:
        """
        One-paragraph summary suitable for a dashboard or log entry.

        Reports: robot name, route, number of segments, total cost,
        number of alternatives analysed, and actionable change count.
        """
        n_segs    = len(path.edges)
        n_cf      = len(counterfactuals)
        actionable = sum(len(cf.actionable_changes()) for cf in counterfactuals)

        cf_note = ""
        if n_cf > 0:
            cf_note = (
                f"  {n_cf} alternative route(s) were analysed; "
                f"{actionable} operator action(s) could unlock shorter paths."
            )

        return (
            f"Robot '{robot.name}' successfully navigated from '{start}' to '{goal}' "
            f"via {n_segs} segment(s) with a total affordance-weighted cost of "
            f"{path.total_cost:.2f}.{cf_note}"
        )

    # ------------------------------------------------------------------
    # Claude-powered narrative generation
    # ------------------------------------------------------------------

    def generate_narrative(self, exp: NavigationExplanation) -> str:
        """
        Produce a rich prose narrative for *exp* using the Claude API.

        Returns the template-based executive summary when ``use_claude=False``
        (the default), so callers can always call this method regardless of
        whether Claude is configured.

        The system prompt is sent with ``cache_control: ephemeral`` so that
        repeated calls within a session benefit from prompt-cache savings.
        """
        if not self._use_claude:
            return exp.executive_summary

        context = {
            "robot":      exp.robot_name,
            "route":      f"{exp.start} → {exp.goal}",
            "path":       exp.chosen_path.summary(),
            "segments":   exp.path_rationale,
            "alternatives_analysed": len(exp.counterfactuals),
            "counterfactuals": [
                {
                    "query":      cf.query,
                    "cost_delta": round(cf.cost_delta, 2),
                    "changes": [
                        c.description()
                        + (" [actionable]" if c.is_actionable() else " [structural]")
                        for c in cf.changes
                    ],
                }
                for cf in exp.counterfactuals
            ],
            "recommendations": exp.recommendations,
        }

        client   = self._get_client()
        response = client.messages.create(
            model      = self._model,
            max_tokens = 400,
            system = [{
                "type": "text",
                "text": self._SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages = [{
                "role":    "user",
                "content": "Navigation event:\n" + json.dumps(context, indent=2),
            }],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Formatted report printer
    # ------------------------------------------------------------------

    def format_report(self, exp: NavigationExplanation, use_claude: bool = False) -> str:
        """
        Render a complete human-readable text report from a NavigationExplanation.

        Parameters
        ----------
        exp        : the explanation to render
        use_claude : when True (and ``ExplanationGenerator`` was constructed with
                     ``use_claude=True``), appends a Claude-generated prose narrative
                     section after the structured template output.

        Returns a multi-line string; caller is responsible for printing or logging.
        The report is structured as:
          Header → Path rationale → Affordance warnings →
          Counterfactual analysis → Recommendations → Executive summary
          [→ Claude Narrative (optional)]
        """
        WIDE = 72
        SEP  = "=" * WIDE
        THIN = "-" * WIDE

        def section(title: str) -> str:
            return f"\n{title}\n{THIN}"

        lines: List[str] = [
            SEP,
            "  NAVIGATION EXPLANATION REPORT",
            SEP,
            f"  Robot  : {exp.robot_name}",
            f"  Route  : {exp.start}  →  {exp.goal}",
            f"  Path   : {exp.chosen_path.summary()}",
            f"  Segs   : {exp.chosen_path.segment_names()}",
        ]

        # ── Layer 1: path rationale ─────────────────────────────────────────
        lines.append(section("PATH RATIONALE"))
        for i, sentence in enumerate(exp.path_rationale, 1):
            lines.append(f"  [{i}] {sentence}")

        # ── Affordance warnings (informational) ──────────────────────────────
        if exp.affordance_warnings:
            lines.append(section("AFFORDANCE WARNINGS"))
            for w in exp.affordance_warnings:
                lines.append(f"  {w}")

        # ── Layer 2: counterfactual analysis ────────────────────────────────
        if exp.counterfactuals:
            lines.append(section("COUNTERFACTUAL ANALYSIS  (Why Not Other Paths?)"))
            for i, cf in enumerate(exp.counterfactuals, 1):
                lines.append(f"\n  [{i}] {cf.query}")
                lines.append(f"       Alternative : {cf.alternative_path.summary()}")

                if cf.changes:
                    lines.append("       Changes needed:")
                    for ch in cf.changes:
                        # Tag each change as actionable or structural
                        tag = "✓ actionable" if ch.is_actionable() else "✗ structural"
                        lines.append(
                            f"         • [{tag}] {ch.description()}"
                            f"  — {ch.rationale}"
                        )
                    lines.append(
                        f"       CF cost : {cf.cf_cost:.2f}  |  "
                        f"Delta : {cf.cost_delta:+.2f}"
                    )
                else:
                    # No changes needed: either alternative is cheaper or has
                    # no fixable blocker
                    if cf.cost_delta >= 0:
                        lines.append(
                            "       No blocking properties — "
                            "alternative is already preferred or equal."
                        )
                    else:
                        lines.append(
                            f"       No single-property fix found — actual path is already "
                            f"cheaper (Δcost = {cf.cost_delta:+.2f})."
                        )

        # ── Layer 3: operator recommendations ──────────────────────────────
        if exp.recommendations:
            lines.append(section("OPERATOR RECOMMENDATIONS"))
            for rec in exp.recommendations:
                lines.append(f"  {rec}")

        # ── Executive summary ───────────────────────────────────────────────
        lines.append(section("EXECUTIVE SUMMARY"))
        lines.append(f"  {exp.executive_summary}")

        # ── Claude narrative (optional) ─────────────────────────────────────
        if use_claude and self._use_claude:
            narrative = self.generate_narrative(exp)
            lines.append(section("NARRATIVE (Claude)"))
            for para in narrative.split("\n"):
                if para.strip():
                    lines.append(f"  {para}")

        lines.append(f"\n{SEP}")

        return "\n".join(lines)
