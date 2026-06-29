"""
ontofact_nav/classification.py
==============================
SPARQL-driven classification layer.

Infers affordance-relevant *space categories* from NUMERIC property conditions
that OWL-RL deductive closure cannot express but a SPARQL ``FILTER`` can — e.g.
``nav:Steep`` ⟸ ``slope_angle > threshold``.  The inferred categories gate
affordance rules (see ``AffordanceRule.requires_category``), so the RDF/SPARQL
layer genuinely affects planning: change a rule's ``FILTER`` and routes change;
clear the rules and the category-gated blocks stop firing.

Why memoize?
------------
The counterfactual subset search calls ``navigation_cost`` (→ ``compute`` →
classify) thousands of times per ``navigate()``.  A SPARQL ``ASK`` over a fresh
micro-graph costs ~5 ms, so running it uncached there would cost tens of seconds.
Classification is a **pure function** of a handful of source properties, so we
cache the result keyed on those values.  Most spaces are flat/default → a single
cached "no categories" result.  The SPARQL engine remains the *sole evaluator* of
the numeric condition; the cache merely memoizes its output — it is not a Python
reimplementation of the rule.

Thresholds are injected from ``NavCostConfig`` at rule-build time, so the SPARQL
``FILTER`` and any Python predicate that reads the same constant cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from rdflib import Graph, Literal
from rdflib.plugins.sparql import prepareQuery

from .config import NavCostConfig
from .ontology import _NAV, _PY_TO_XSD

# SPARQL prefix header (mirrors Ontology.sparql_select).
_PREFIXES = (
    "PREFIX nav:  <http://ontofact.nav/>\n"
    "PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>\n"
)


@lru_cache(maxsize=256)
def _prepare(query_str: str):
    """Parse a SPARQL query once and reuse the compiled form across all
    classifiers and graphs (prepared queries are graph-independent)."""
    return prepareQuery(query_str)


@dataclass(frozen=True)
class DerivationRule:
    """A numeric classification rule evaluated by a SPARQL ``ASK``.

    Attributes
    ----------
    category     : the ``nav:`` local name of the inferred category (e.g. "Steep").
    source_props : space property keys the rule reads — used both to build the
                   micro-graph and to form the memo cache key.
    sparql_ask   : a SPARQL ``ASK`` body (no prefixes) over the space subject
                   ``?s`` (and, for ``scope="joint"``, the agent subject ``?a``),
                   with thresholds already inlined.
    scope        : "space" (reads only the space) | "joint" (also reads the agent).
    agent_props  : agent property keys read by a joint rule (cache-key inputs).
    description  : human-readable note.
    """
    category:     str
    source_props: Tuple[str, ...]
    sparql_ask:   str
    scope:        str = "space"
    agent_props:  Tuple[str, ...] = ()
    description:  str = ""


def build_default_rules(cfg: NavCostConfig) -> Tuple[DerivationRule, ...]:
    """Default derivation rules with ``NavCostConfig`` thresholds inlined into the
    SPARQL ``FILTER`` bodies (single source of truth for the numbers)."""
    return (
        DerivationRule(
            category="Steep",
            source_props=("slope_angle",),
            scope="space",
            sparql_ask=(
                "ASK { ?s nav:slope_angle ?sl . "
                f"FILTER(?sl > {cfg.wheeled_stair_slope_limit}) }}"
            ),
            description="slope exceeds the fixed wheeled stair-slope limit",
        ),
        DerivationRule(
            category="HighRiskZone",
            source_props=("crowd_density", "illumination"),
            scope="space",
            sparql_ask=(
                "ASK { ?s nav:crowd_density ?c ; nav:illumination ?i . "
                f"FILTER(?c > {cfg.crowd_block_threshold} && "
                f"?i < {cfg.observable_threshold}) }}"
            ),
            description="crowded AND poorly lit — a combined-condition risk zone",
        ),
    )


class Classifier:
    """Memoized SPARQL classifier: a space's property view → set of categories.

    The classifier is a pure function of the rule-relevant property values, so it
    is safe to cache.  Counterfactual clones reclassify correctly because the
    input is the clone's *current* property view (not a pre-materialized graph).
    """

    # Fixed subjects for the throwaway micro-graph (the ASK uses variables, so
    # the concrete URIs are irrelevant — there is exactly one of each).
    _S = _NAV["_space"]
    _A = _NAV["_agent"]

    def __init__(
        self,
        config: NavCostConfig,
        rules: Optional[Tuple[DerivationRule, ...]] = None,
    ) -> None:
        self.config = config
        self.rules: Tuple[DerivationRule, ...] = (
            tuple(rules) if rules is not None else build_default_rules(config)
        )
        self._cache: Dict[Tuple, FrozenSet[str]] = {}
        self._evals: int = 0   # number of real SPARQL evaluations (cache misses)
        self._validate()
        self._compile()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Guard the cache-key footgun: every property a rule reads in its SPARQL
        (a ``nav:<name> ?var`` predicate) must be declared in source_props/
        agent_props, or the micro-graph and cache key would be incomplete and the
        rule could return a silently wrong result."""
        for rule in self.rules:
            used = set(re.findall(r"nav:(\w+)\s+[?]", rule.sparql_ask))
            declared = set(rule.source_props) | set(rule.agent_props)
            missing = used - declared
            if missing:
                raise ValueError(
                    f"DerivationRule {rule.category!r} reads {sorted(missing)} in "
                    f"its SPARQL but does not declare them in source_props/"
                    f"agent_props — the cache key and micro-graph would be "
                    f"incomplete."
                )

    def _compile(self) -> None:
        """Pre-compile each rule's SPARQL ASK once (rdflib re-parses otherwise —
        ~14x slower per call), so cache misses execute a prepared query."""
        self._prepared = [
            (rule.category, _prepare(_PREFIXES + rule.sparql_ask))
            for rule in self.rules
        ]

    def set_rules(self, rules: Tuple[DerivationRule, ...]) -> None:
        """Replace the rule set (e.g. to disable classification) and reset cache."""
        self.rules = tuple(rules)
        self._validate()
        self._compile()
        self.clear_cache()

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Source-property sets (for cache keys + micro-graph construction)
    # ------------------------------------------------------------------

    def _union(self, attr: str) -> Tuple[str, ...]:
        seen: List[str] = []
        for rule in self.rules:
            for prop in getattr(rule, attr):
                if prop not in seen:
                    seen.append(prop)
        return tuple(seen)

    @staticmethod
    def _keyed(value: Any) -> Tuple[str, Any]:
        # Include the type name so e.g. True (bool) and 1 (int) — which are == and
        # hash-equal in Python but classify differently in SPARQL — never collide.
        return (type(value).__name__, value)

    def _key(self, space_view: Dict[str, Any], agent_view: Dict[str, Any]) -> Tuple:
        sp = tuple((p, self._keyed(space_view.get(p))) for p in self._union("source_props"))
        ap = tuple((p, self._keyed(agent_view.get(p))) for p in self._union("agent_props"))
        return (sp, ap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def derived_categories(
        self,
        space_view: Dict[str, Any],
        agent_view: Dict[str, Any],
    ) -> FrozenSet[str]:
        """Return the set of categories the space belongs to (cache-first)."""
        if not self.rules:
            return frozenset()
        key = self._key(space_view, agent_view)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._evaluate(space_view, agent_view)
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Evaluation — the ONLY place a SPARQL query runs
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        space_view: Dict[str, Any],
        agent_view: Dict[str, Any],
    ) -> FrozenSet[str]:
        self._evals += 1
        g = Graph()
        self._add_triples(g, self._S, space_view, self._union("source_props"))
        self._add_triples(g, self._A, agent_view, self._union("agent_props"))
        out = set()
        for category, prepared in self._prepared:
            if bool(g.query(prepared)):
                out.add(category)
        return frozenset(out)

    @staticmethod
    def _add_triples(g: Graph, subject, view: Dict[str, Any], props: Tuple[str, ...]) -> None:
        for prop in props:
            value = view.get(prop)
            if value is None:
                continue   # absent → no triple → rule's FILTER simply won't match
            g.add((subject, _NAV[prop], Literal(value, datatype=_PY_TO_XSD.get(type(value)))))
