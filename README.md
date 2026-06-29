# Ontology-Guided Counterfactual Affordance Reasoning for Actionable Robot Navigation Explanations

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-115%20passing-brightgreen.svg)](tests/)

An explainability framework for mobile robot navigation that combines:

- **Domain ontology** — RDFLib-backed OWL class/property hierarchy; real SPARQL queries; OWL-RL deductive closure. The ontology is **load-bearing** for planning in two ways: (1) class-conditioned rules + inherited class defaults (a Staircase and a Corridor with identical properties plan differently), and (2) a **SPARQL-driven classification layer** that infers numeric categories (`nav:Steep`, `nav:HighRiskZone`) via real SPARQL `ASK` — something OWL-RL closure cannot express — which gate affordance rules and therefore change routes  
- **Affordance theory** — forward-chaining rules infer what a robot *can* do in each space; a single passability contract (the cost function decides feasibility purely from the affordance set)  
- **A\* path planning** — affordance-weighted costs; NetworkX k-shortest paths for alternatives  
- **Counterfactual reasoning** — genuine **minimum-cardinality subset search** answers "Why not path X?", with honest actionable-vs-structural attribution  
- **Natural language generation** — three-layer reports: rationale → counterfactual → recommendations; optional Claude API narrative  
- **Tunable cost model** — every threshold and weight lives in one injectable, frozen `NavCostConfig`  
- **REST API** — FastAPI service exposes `navigate`, `why-not`, and SPARQL over HTTP  
- **Visualization** — NetworkX + Matplotlib renders the graph with color-coded path overlays  

---

## Installation

The project uses a standard `src/` layout and is installed as an editable package
(recommended, so the `ontofact_nav` package and the `ontofact-nav` console script
resolve from anywhere):

```bash
# From the project root (directory containing pyproject.toml)
pip install -e .            # runtime only
pip install -e ".[dev]"     # runtime + test tooling (pytest, httpx)

# Alternatively, install dependencies directly without packaging
pip install -r requirements.txt
```

**Runtime dependencies:** `networkx`, `rdflib`, `owlrl`, `anthropic`, `pydantic`, `fastapi`, `uvicorn`, `matplotlib`  
**Dev/test:** `pytest`, `httpx`

> The optional Claude API narrative (see [`explanation.py`](#explanationpy--explanation-generator))
> is **off by default** — the framework is fully functional without an `ANTHROPIC_API_KEY`.

---

## Quick start

```bash
# Run demo scenarios (console script installed by `pip install -e .`)
ontofact-nav                # both scenarios
ontofact-nav hospital       # hospital only (6 sub-scenarios)
ontofact-nav warehouse      # warehouse only (4 sub-scenarios)

# Equivalent module form (no console script needed)
python -m ontofact_nav hospital

# Start the REST API
uvicorn ontofact_nav.api:app --reload

# Run the test suite (115 tests)
pytest
```

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)  
2. [Architecture](#architecture)  
3. [File Structure](#file-structure)  
4. [Module Reference](#module-reference)  
5. [REST API](#rest-api)  
6. [Visualization](#visualization)  
7. [Scenarios](#scenarios)  
8. [Extending the System](#extending-the-system)  
9. [Sample Output](#sample-output)  
10. [Design Decisions](#design-decisions)  
11. [Development](#development)  
12. [References](#references)  

---

## Theoretical Background

### Ontology

An **ontology** is a formal vocabulary of concepts and their relationships for a domain.
This framework uses an OWL-inspired structure (classes, properties, individuals) backed by
**RDFLib** for serialisation and real SPARQL querying, and **OWL-RL** for deductive closure
(subclass-chain propagation, property domain/range inferences).

Every navigable space (corridor, elevator, doorway…) and every robot is an *individual* in
the ontology, typed by a class in the hierarchy and characterised by data properties.

**Three reasoning modes feed planning** — and all three are genuinely load-bearing:

1. **OWL-RL subsumption** over *asserted* types (`apply_reasoning()`): subclass-chain propagation,
   property domain/range. Monotonic, qualitative, agent-independent.
2. **Class-conditioned rules + inherited class defaults**: an `AffordanceRule` may require a class
   (`requires_class="Staircase"`), and a class may declare defaults inherited by its individuals,
   so class *membership* changes affordances.
3. **SPARQL numeric classification** (`requires_category`): categories like `nav:Steep`
   (`slope > threshold`) and `nav:HighRiskZone` (`crowded AND dark`) are inferred by real SPARQL
   `ASK` from *property values*. OWL-RL/RDFS closure cannot do arithmetic or compare an
   individual's properties; a SPARQL `FILTER` can. The thresholds are injected from
   [`NavCostConfig`](#configpy--navcostconfig) so the rule and any Python predicate cannot drift.

> **Why memoize the classifier?** The counterfactual subset search calls the cost function
> thousands of times per `navigate()`; a fresh SPARQL `ASK` costs ~5 ms, so running it uncached
> would cost tens of seconds. Classification is a *pure function* of a few source properties, so the
> result is cached on those values — collapsing the thousands of calls to a handful of distinct
> evaluations (most spaces are flat → one cached "no categories" result). The SPARQL engine remains
> the sole evaluator of the numeric condition; the cache is not a Python reimplementation. Queries
> are pre-compiled once (`prepareQuery`) and shared, so each evaluation is ~0.3 ms.

The agent-relative slope limit (`slope ≤ robot.max_slope_angle`) deliberately stays a numeric
`NavCostConfig` predicate, not a SPARQL category: it is a per-agent join with no shared constant to
unify. Only the *fixed* stair-slope threshold (a config constant) is migrated to SPARQL — an honest
split (the classifier *can* do agent-relative joins; see `test_joint_rule_does_agent_relative_classification`).

### Affordance Theory

J.J. Gibson (1979) coined the term *affordance* to describe the action possibilities
that an environment offers to an agent.  For navigation, affordances include:

| Affordance | Meaning |
|------------|---------|
| `traversable` | The robot is allowed to enter / move through the space |
| `passable` | The robot's body fits (width + clearance ≤ space width) |
| `climbable` | The slope angle is within the robot's mobility limit |
| `openable` | The robot can open a door blocking this space |
| `observable` | The space is well-enough lit for reliable sensor readings |
| `avoidable` | The space may be excluded from the plan |

Affordances are not stored directly — they are *inferred* from ontology properties
(width, slope_angle, door_state, illumination…) using a forward-chaining rule engine.
This means a single property change (e.g. propping a door open) automatically propagates
through the affordance layer into updated navigation costs.

### Counterfactual Reasoning

A counterfactual explanation answers the question:  
> *"What would need to be different for outcome Y to have occurred instead of X?"*

Applied to navigation: for each alternative path the planner considered but rejected,
the counterfactual engine searches for the **smallest set of world-property changes** that
would make that path feasible *and* preferred (cost ≤ the chosen path).  This implements the
*minimal-change principle* from counterfactual logic (Lewis 1973; Pearl 2000) as a genuine
**minimum-cardinality subset search** over a pool of candidate single-property fixes — an
ascending-cardinality enumeration that tries every 1-change set, then every 2-change set, and
so on, returning the first cardinality that flips the decision (ties broken by resulting cost,
then operator effort).  A minimum-cardinality solution is automatically subset-minimal, so the
result is never padded with changes that aren't needed.

**Honest attribution.** The search is run twice — over the full candidate pool, and again
restricted to operator-*actionable* candidates only.  `is_achievable` is true **iff** the
actionable-only changes *alone* make the alternative preferred, and `actionable_only_delta`
reports the saving genuinely attributable to runtime operator action.  So a saving that
secretly depends on structural construction (widening a corridor, regrading a ramp) is never
advertised as something an operator can unlock by themselves.

Actionability filtering separates changes a human operator can make today
(open a door, clear a crowd, fix lighting) from structural changes (widen a corridor, reduce a
slope), along two orthogonal axes: *can an operator do it at all* (`is_actionable()`) and *how
much effort* (`effort()`).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OntofactNavigator                               │
│                           (orchestrator)                                │
└──────┬──────────────────┬──────────────────┬─────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌────────────┐   ┌────────────────┐   ┌──────────────────────┐
│  Ontology  │   │  AffordanceR.  │   │  ExplanationGen.     │
│ (RDFLib +  │   │                │   │                      │
│  OWL-RL)  │   │ grant rules    │   │ Layer 1: rationale   │
│ SPARQL     │   │ block rules    │   │ Layer 2: cf-analysis │
│ individuals│   │ cost formula   │   │ Layer 3: recommend.  │
└────┬───────┘   └──────┬─────────┘   │ + Claude narrative   │
     │                  │             └──────────────────────┘
     │         ┌────────┴──────────┐
     │         │                   │
     ▼         ▼                   ▼
┌──────────┐ ┌──────────────┐ ┌────────────────────────┐
│ NavGraph │ │  A* Planner  │ │  CounterfactualEngine  │
│(pydantic)│ │              │ │                        │
│ NavNode  │ │  find_path() │ │  explain_why_not()     │
│ NavEdge  │ │  find_k_path │ │  batch_why_not()       │
└──────────┘ │  (NetworkX)  │ └────────────────────────┘
             └──────────────┘
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
┌─────────────┐           ┌──────────────────┐
│ FastAPI     │           │ Visualization    │
│ api.py      │           │ (Matplotlib +    │
│ REST layer  │           │  NetworkX draw)  │
└─────────────┘           └──────────────────┘
```

**Data flow for a single `navigate()` call:**

```
1. A* finds the optimal path (affordance-weighted costs)
2. NetworkX shortest_simple_paths finds k-1 alternative paths
3. For each alternative, CounterfactualEngine runs:
     a. Evaluate alternative's cost in current world
     b. Identify problematic edges (inf cost or high cost)
     c. Generate minimal property changes to fix each edge
     d. Simulate changes → compute new cost → compute delta
4. ExplanationGenerator formats all artefacts into a report
   (optionally enriched with a Claude API narrative)
```

---

## File Structure

```
.                                  Project root
├── pyproject.toml                 Packaging, dependencies, console script, tool config
├── requirements.txt               Convenience dependency list (mirrors pyproject)
├── LICENSE                        MIT license
├── README.md
├── .gitignore
├── src/
│   └── ontofact_nav/
│       ├── __init__.py            Public API re-exports
│       ├── __main__.py            `python -m ontofact_nav` entry point
│       ├── main.py                CLI scenario runner
│       ├── config.py              NavCostConfig — thresholds, weights + SSOT predicates
│       ├── classification.py      SPARQL-driven numeric category inference (memoized)
│       ├── ontology.py            Ontology engine (RDFLib + OWL-RL); class defaults; classify()
│       ├── domain.py              Navigation domain schema + enums
│       ├── affordance.py          Affordance rule engine (pydantic models)
│       ├── navigation.py          NavGraph, A*, NetworkX k-paths (pydantic models)
│       ├── counterfactual.py      Minimal-subset counterfactual engine
│       ├── explanation.py         Three-layer NL generator (+ Claude API)
│       ├── orchestrator.py        OntofactNavigator integration class
│       ├── visualization.py       Matplotlib/NetworkX graph rendering
│       ├── api.py                 FastAPI REST service
│       └── scenarios/
│           ├── hospital.py        6-scenario hospital floor plan demo
│           └── warehouse.py       4-scenario warehouse demo
└── tests/                         pytest suite — 115 tests
    ├── conftest.py                     Shared fixtures (ontology, agent/space factories)
    ├── test_ontology.py                Ontology, querying, SPARQL, OWL-RL (14 tests)
    ├── test_affordance.py              Affordance rules + cost formula (21 tests)
    ├── test_navigation.py              Validation, A*, k-paths (16 tests)
    ├── test_counterfactual.py          Counterfactual engine + navigator (16 tests)
    ├── test_config.py                  NavCostConfig injection (5 tests)
    ├── test_passability_ssot.py        Single-source-of-truth feasibility (5 tests)
    ├── test_ontology_load_bearing.py   Class-driven planning + RDF sync (9 tests)
    ├── test_counterfactual_minimal.py  Minimum-cardinality subset + large-pool fallback (5 tests)
    ├── test_counterfactual_attribution.py  Actionable-only attribution + hidden-blocker (5 tests)
    ├── test_classification.py          SPARQL inference drives/flips planning; class-node + key-collision guards (14 tests)
    └── test_api.py                     FastAPI endpoints, JSON-safety, hospital scenario (5 tests)
```

> **Why `src/` layout?** It guarantees the test suite and any consumer import the
> *installed* package rather than accidentally picking up the source directory from
> the current working directory — the recommended layout for distributable Python
> packages.

---

## Module Reference

### `ontology.py` — Ontology Engine

Core data structures for the knowledge representation layer, backed by a parallel
**RDFLib** `Graph` for real SPARQL querying and **OWL-RL** deductive closure.

| Class / Method | Description |
|----------------|-------------|
| `OntologyClass` | Node in the class hierarchy. `is_subclass_of()` / `is_subclass_of_name()` walk the parent chain; `defaults` + `resolved_default()` provide inherited class-level property values. |
| `OntologyProperty` | Property slot descriptor (domain class, Python type). |
| `OntologyIndividual` | Instance with a property dict. `get()` is class-aware (explicit value → inherited class default → fallback); `clone()` makes a graph-less counterfactual copy. |
| `Ontology.defclass(..., defaults=…)` | Declare a class (+ optional inherited defaults); populates the Python hierarchy and mirrors class defaults onto the RDF class URI. |
| `Ontology.create()` | Factory: construct + register a live individual; adds RDF triples and links the individual to the graph. |
| `Ontology.query()` | Python dict-based filter query (class + equality constraints). |
| `Ontology.sparql_select()` | Real SPARQL SELECT via RDFLib (`nav:` prefix pre-injected). |
| `Ontology.apply_reasoning()` | OWL-RL deductive closure (RDFS or full OWL-RL). |

**Load-bearing ontology.** The class hierarchy is not decorative — it changes planning outcomes:
- **Class-conditioned rules** (`AffordanceRule.requires_class`) let inference depend on the class
  (a `Staircase` is non-climbable for a wheeled robot regardless of its numeric slope).
- **Inherited class defaults** flow into the affordance rules, so an individual that omits a
  property picks up its class's default. A `Staircase` and a `Corridor` created with *identical
  explicit properties* can therefore plan differently (the Staircase inherits a stair-like slope).
- **Live RDF write-through:** `OntologyIndividual.set()` on a registered individual updates the
  RDF graph, so SPARQL reflects the current world — while clones stay graph-less so counterfactual
  worlds never leak into the store.
- **SPARQL category inference** (`requires_category`): see [`classification.py`](#classificationpy--sparql-classification-layer).

**`Ontology.classify(rules)`** runs a SPARQL `INSERT` per derivation rule to *materialise* the
inferred category types onto live individuals, so they are visible to `sparql_select` (e.g.
`SELECT ?s WHERE { ?s a nav:Steep }`). This is for **inspection only** — the planning path uses the
in-memory, memoized `Classifier` and never writes to the graph; clones (graph-less) are never affected.

```python
onto = build_navigation_ontology()
onto.create("hallway_1", "Corridor", width=2.5, is_accessible=True)

# Python dict query
results = onto.query("Corridor", is_accessible=True)

# Real SPARQL
rows = onto.sparql_select("""
    SELECT ?ind ?width WHERE {
        ?ind a nav:Corridor ;
             nav:width ?width .
        FILTER (?width < 1.2)
    }
""")
for row in rows:
    print(row.ind, float(row.width))

# OWL-RL inference — after this, querying nav:IndoorSpace matches Corridors, Rooms, etc.
added = onto.apply_reasoning("rdfs")
print(f"{added} triples inferred")
```

> **Note:** `sparql_select` reflects the live world for *registered* individuals (mutations via
> `set()` are written through) and class-level defaults. Counterfactual clones from
> `individual.clone()` are graph-less, so hypothetical worlds are never synced to the RDF graph.
> OWL-RL `apply_reasoning()` augments the graph for SPARQL; the planner itself reasons over the
> Python class hierarchy and affordance rules.

---

### `domain.py` — Navigation Domain Schema

Builds the navigation-specific ontology (20 classes, 23 properties) and
defines all symbolic enums.

**Class hierarchy (partial):**
```
Thing → PhysicalEntity → Space → IndoorSpace → Room
                                              → Corridor
                                              → Staircase
                                              → Elevator
                                              → Doorway
                                 OutdoorSpace → OutdoorPath
                                 Ramp
                       Agent → Robot
```

**Key enums:**

| Enum | Values |
|------|--------|
| `AffordanceType` | `TRAVERSABLE PASSABLE CLIMBABLE OPENABLE OBSERVABLE AVOIDABLE` |
| `DoorState` | `OPEN CLOSED LOCKED` |
| `SurfaceType` | `SMOOTH TILED CARPETED ROUGH WET GRAVEL GRASS` |
| `MobilityType` | `WHEELED LEGGED TRACKED AERIAL` |

Some classes declare **inherited defaults** (e.g. `Staircase` defaults to `slope_angle=30°`,
`surface_type=rough`), which is what lets class membership alone change planning outcomes.

---

### `config.py` — NavCostConfig

A single frozen dataclass holding **every** numeric constant (thresholds, soft-cost weights,
geometric fallbacks, counterfactual targets) **and** the derived feasibility predicates that the
rule engine, the cost function, and the counterfactual search all share. Co-locating the constants
and the predicates means each geometry/threshold formula exists in exactly one place.

```python
from ontofact_nav import NavCostConfig, OntofactNavigator

# Defaults reproduce the framework's historical numbers exactly.
cfg = NavCostConfig(crowd_cost_weight=8.0, observable_threshold=0.4)

# Inject anywhere — the reasoner, engine, and orchestrator all accept `config=`.
nav = OntofactNavigator(onto, graph, config=cfg)
```

Predicate helpers (`required_width`, `fits_width`, `slope_within_limit`, `wheeled_on_stairs`,
`lit_enough`, `can_open_closed_door`) are used by **both** the affordance rules and the
counterfactual candidate generator, so the two cannot drift. The engine defaults its config to the
reasoner's, guaranteeing one shared policy across the stack.

---

### `classification.py` — SPARQL classification layer

Infers space *categories* from **numeric** property conditions using real SPARQL `ASK` — the part
of "load-bearing ontology" that the class hierarchy alone cannot provide (OWL-RL/RDFS closure has
no arithmetic and cannot compare an individual's properties).

| Component | Description |
|-----------|-------------|
| `DerivationRule` | `(category, source_props, sparql_ask, scope, agent_props, description)` — a SPARQL `ASK` body over space subject `?s` (and agent `?a` for `scope="joint"`) with thresholds inlined. |
| `build_default_rules(cfg)` | Default rules with `NavCostConfig` thresholds injected into the `FILTER`s: `Steep` (`slope > wheeled_stair_slope_limit`) and `HighRiskZone` (`crowd_density > crowd_block_threshold && illumination < observable_threshold`). |
| `Classifier(cfg, rules=None)` | `derived_categories(space_view, agent_view) -> frozenset[str]`, **memoized** on the source-prop values; `set_rules()` / `clear_cache()`; `_evals` counter for observability. |

```python
from ontofact_nav import Classifier, NavCostConfig, build_default_rules

clf = Classifier(NavCostConfig())
clf.derived_categories({"slope_angle": 18.0}, {})            # frozenset({'Steep'})
clf.derived_categories({"crowd_density": 0.6, "illumination": 0.2}, {})  # {'HighRiskZone'}
clf.derived_categories({"crowd_density": 0.6, "illumination": 1.0}, {})  # frozenset()  (the AND fails)
```

Affordance rules consume categories via `AffordanceRule.requires_category` (e.g. the migrated
`block_climbable_steep_zone_wheeled` gates on `"Steep"`; the new `block_traversable_high_risk_uncertified`
gates on `"HighRiskZone"`). `compute()` classifies the entity once from its **property view**, so
counterfactual clones with mutated properties reclassify correctly (a clone whose slope drops below
the threshold is no longer `Steep`). Thresholds come from `NavCostConfig`, so the SPARQL `FILTER` and
the Python predicates share one source of truth. See the [Ontology theory](#ontology) note on why the
classifier is memoized and how the agent-relative slope limit stays a numeric predicate by design.

---

### `affordance.py` — Affordance Reasoner

Implements a **forward-chaining rule engine** using two lists of `AffordanceRule` objects
(Pydantic dataclasses — invalid rules are rejected at construction):
- **grant_rules** add an affordance when their condition fires  
- **block_rules** remove an affordance when their condition fires  

Net affordances = granted − blocked.

**Single source of truth for passability.** Every door/slope hard-barrier is folded into a
block rule (locked or unopenable doors remove `TRAVERSABLE`; over-steep slopes and stairs
remove `CLIMBABLE`), and every grant/block condition delegates to a `NavCostConfig` predicate
(`fits_width`, `slope_within_limit`, `can_open_closed_door`, …). Feasibility is therefore decided
in exactly one place — `navigation_cost()` returns `math.inf` **iff** the affordance set is
missing one of `{TRAVERSABLE, PASSABLE, CLIMBABLE}` and makes no independent re-check of raw
properties.

**Cost formula** (used as A\* edge weight; all weights come from `NavCostConfig`):

```
if {TRAVERSABLE, PASSABLE, CLIMBABLE} ⊄ affordances → math.inf      (single feasibility gate)

cost = edge.distance
+ door_opening_penalty (cfg.door_opening_penalty = 2.5, if CLOSED and OPENABLE)
+ surface_friction     (cfg.surface_cost[...] = 0.0–1.4)
+ crowd_density   × cfg.crowd_cost_weight    (4.0)
+ obstacle_density × cfg.obstacle_cost_weight (2.5)
+ visibility_penalty   (cfg.visibility_penalty = 1.8, if not OBSERVABLE)
+ slope × cfg.slope_cost_coeff (0.06)
× cfg.emergency_discount (0.75 if designated emergency)
```

**Class- and category-conditioned rules.** An `AffordanceRule` may carry
`requires_class="Staircase"` (fires only for that asserted class / subclasses) and/or
`requires_category="Steep"` (fires only for that SPARQL-inferred category — see
[`classification.py`](#classificationpy--sparql-classification-layer)). `compute()` merges inherited
class-level defaults into the property view and classifies the entity once, making **both** the
class hierarchy and the SPARQL numeric inference load-bearing for planning.

**Adding a custom rule:**

```python
from ontofact_nav.affordance import AffordanceRule
from ontofact_nav.domain import AffordanceType

nav.reasoner.add_block(AffordanceRule(
    name="block_requires_gowning",
    affordance=AffordanceType.TRAVERSABLE,
    condition=lambda e, a: e.get("requires_gowning", False) and not a.get("has_gown", False),
    explanation_template="{entity} requires gowning — robot does not have a gown",
))
```

---

### `navigation.py` — Graph & Planner

`NavNode`, `NavEdge`, and `NavPath` are **Pydantic dataclasses** — invalid construction
(empty node ID, non-positive distance, non-finite coordinates, negative cost) raises
`ValidationError` immediately.

**`NavigationGraph`** — weighted directed graph backed by an adjacency list.

**`AStarPlanner.find_path()`** — custom A\* with:
- Euclidean-distance admissible heuristic  
- Heap entries `(f_score, g_score, counter, node_id)` — g_score enables correct lazy-deletion  
- Guard against unknown start/goal nodes

**`AStarPlanner.find_k_paths()`** — powered by **NetworkX** `shortest_simple_paths`
(Yen's k-shortest loopless paths algorithm):
1. Builds a temporary agent-specific weighted `DiGraph` (infinite-cost edges excluded)  
2. Calls `nx.shortest_simple_paths(wg, start, goal, weight="weight")`  
3. Converts each node sequence to a full `NavPath` via `evaluate_sequence`

---

### `counterfactual.py` — Counterfactual Engine

**`CounterfactualEngine.explain_why_not(actual_path, alt_nodes, agent)`**

Steps:
1. Evaluate the alternative's cost in the current world  
2. If already cheaper → trivial counterfactual (world is already there)  
3. Iterate edges: flag those with `cost == inf` or `cost > cfg.cf_problematic_factor × (distance + cfg.cf_problematic_buffer)` as problematic  
4. For each problematic edge, `_candidate_changes()` enumerates the **pool** of single-property
   fixes that *could* help (hazard, restricted, accessibility, door, width, slope, crowd,
   illumination — each gated on the same `NavCostConfig` predicate the rule engine uses)  
5. **`_minimal_subset()`** searches that pool by ascending cardinality for the *smallest* subset
   that makes the alternative feasible **and** ≤ the actual cost (tie-break: resulting cost, then
   effort; a `cf_max_pool` cap falls back to greedy forward-select + backward-prune). The result
   is the reported `changes` — minimum-cardinality, never padded.  
6. The same search is run a **second** time over only the *actionable* candidates → drives
   `is_achievable` and `actionable_only_{changes,cost,delta}`  
7. Return a `Counterfactual` whose `changes` are tagged `actionable` or structural, with a
   narrative that states explicitly when a saving requires construction an operator cannot do

> The candidate generator is a pure *pool* — it no longer auto-applies every fix. This is what
> makes the result genuinely minimal rather than the union of all applicable changes.

**Taxonomy (two orthogonal axes, one table).** `PROPERTY_TAXONOMY` drives both
`PropertyChange.is_actionable()` (can an operator do it at runtime without construction?) and
`PropertyChange.effort()` (low / medium / high) so they can never disagree — e.g. lifting a
`restricted` flag is *actionable* yet *medium* effort, while widening a corridor is
*non-actionable* and *high* effort.

---

### `explanation.py` — Explanation Generator

Produces `NavigationExplanation` with three layers:

**Layer 1 — Path rationale:** one English sentence per traversed segment  
**Layer 2 — Counterfactual analysis:** per-alternative changes, cost delta  
**Layer 3 — Operator recommendations:** de-duplicated, effort-labelled actionable changes

**Optional Claude API narrative:**

```python
from ontofact_nav.explanation import ExplanationGenerator

# Enable Claude-powered prose narrative (requires ANTHROPIC_API_KEY)
nav.explainer = ExplanationGenerator(
    use_claude=True,
    model="claude-haiku-4-5-20251001",   # or sonnet/opus
)

# Pass use_claude=True to append the narrative section to the report
report = nav.explainer.format_report(exp, use_claude=True)

# Or generate the narrative string directly
narrative = nav.explainer.generate_narrative(exp)
```

The system prompt is sent with `cache_control: ephemeral` so repeated calls
within a session benefit from prompt-cache savings.

---

### `visualization.py` — Graph Renderer

```python
from ontofact_nav.visualization import draw_navigation_graph

fig = draw_navigation_graph(
    graph,
    chosen_path       = path,
    alternative_paths = alternatives,
    reasoner          = nav.reasoner,
    agent             = robot,
    title             = "Hospital — delivery_bot",
    save_path         = "hospital.png",   # optional
    show              = False,            # set True for interactive display
)
```

**Color coding:**

| Element | Color | Meaning |
|---------|-------|---------|
| Edge | green | Chosen path |
| Edge | orange | Alternative path |
| Edge | red dashed | Impassable for this agent (cost = ∞) |
| Edge | light grey | Background / unreferenced |
| Node | dark green | Start |
| Node | dark red | Goal |
| Node | blue | Intermediate on any highlighted path |
| Node | grey | Unreferenced |

Edge labels show affordance-weighted cost when `reasoner` and `agent` are provided.

---

### `api.py` — FastAPI REST Service

```bash
uvicorn ontofact_nav.api:app --reload
# Swagger UI: http://localhost:8000/docs
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check; lists available agents |
| `/agents` | GET | List robot agent names |
| `/graph` | GET | Navigation graph as JSON (nodes + edges) |
| `/navigate` | POST | Plan a path; returns 3-layer explanation fields |
| `/why-not` | POST | Ad-hoc counterfactual query |
| `/ontology/sparql` | GET | Execute a SPARQL SELECT (`?q=…`) |

**Example requests:**

```bash
# Plan a path
curl -X POST http://localhost:8000/navigate \
  -H "Content-Type: application/json" \
  -d '{"start":"entrance","goal":"icu","agent_name":"delivery_bot","k_alternatives":3}'

# Why not an alternative route?
curl -X POST http://localhost:8000/why-not \
  -H "Content-Type: application/json" \
  -d '{"start":"entrance","goal":"icu","agent_name":"delivery_bot","alt_nodes":["entrance","lobby","icu"]}'

# SPARQL query
curl "http://localhost:8000/ontology/sparql?q=SELECT+?ind+?w+WHERE+{+?ind+a+nav:Corridor+;+nav:width+?w+}"
```

Set `ONTOFACT_SCENARIO=hospital` to boot the full hospital scenario at startup.

---

### `orchestrator.py` — OntofactNavigator

Single-class integration point.

```python
from ontofact_nav import OntofactNavigator, build_navigation_ontology
from ontofact_nav import NavigationGraph, NavNode, NavEdge

onto  = build_navigation_ontology()
graph = NavigationGraph()

# Add nodes (validated: non-empty ID, finite 2-D position)
graph.add_node(NavNode("entrance", (0.0, 0.0)))
graph.add_node(NavNode("icu",      (15.0, 0.0)))

# Add edges (validated: non-empty IDs, distance > 0)
seg = onto.create("seg_entrance_icu", "Corridor", width=2.0, ...)
graph.add_edge(NavEdge("entrance", "icu", seg, distance=15.0))

agent = onto.create("my_robot", "Robot",
    robot_width=0.6, robot_height=1.4,
    min_clearance=0.15, max_slope_angle=8.0,
    mobility_type="wheeled", can_open_doors=True,
    has_arm=True, battery_level=0.9, max_speed=1.2,
)

nav = OntofactNavigator(onto, graph)
path, explanation = nav.navigate("entrance", "icu", agent, k_alternatives=3)
nav.print_report(explanation)

# Ad-hoc why-not query
cf = nav.query_why_not("entrance", "icu", agent,
                        alt_nodes=["entrance", "side_room", "icu"])
print(cf.explanation)
```

---

## REST API

The `api.py` module exposes a self-contained FastAPI application.  On startup it
builds a 4-node demo scenario (entrance → lobby → icu corridor → icu) with two
pre-defined agents (`delivery_bot`, `cargo_bot`).

**Request / response schemas** (Pydantic `BaseModel`):

```python
# POST /navigate
NavigateRequest(start, goal, agent_name, k_alternatives=3)
→ NavigateResponse(robot_name, start, goal, path, total_cost,
                   is_feasible, rationale, recommendations, executive_summary)

# POST /why-not
WhyNotRequest(start, goal, agent_name, alt_nodes)
→ WhyNotResponse(query, changes, cf_cost, cost_delta, is_achievable, explanation)
   #  changes        — the MINIMAL change set (each tagged is_actionable + effort)
   #  cf_cost        — cost under those changes; null if still infeasible
   #  cost_delta     — actual − cf_cost; null when unbounded (e.g. the actual path
   #                   is itself infeasible) so the JSON stays standards-compliant
   #  is_achievable  — true iff operator-actionable changes ALONE suffice

# GET /ontology/sparql?q=…
→ SparqlResponse(columns, rows)
```

Interactive Swagger docs available at `/docs`; ReDoc at `/redoc`.

---

## Visualization

```python
from ontofact_nav import OntofactNavigator, build_navigation_ontology
from ontofact_nav import NavigationGraph, NavNode, NavEdge
from ontofact_nav.visualization import draw_navigation_graph

# ... build onto, graph, agent, nav ...

path, exp = nav.navigate("entrance", "icu_main", agent, k_alternatives=3)
alternatives = [cf.alternative_path for cf in exp.counterfactuals]

draw_navigation_graph(
    graph,
    chosen_path       = path,
    alternative_paths = alternatives,
    reasoner          = nav.reasoner,
    agent             = agent,
    title             = f"Hospital — {agent.name}",
    save_path         = "hospital_path.png",
    show              = False,
)
```

The function uses `NavNode.position` (the 2-D coordinates already stored on each node)
for layout, so the rendered diagram matches the physical floor plan geometry.

---

## Scenarios

### Hospital (`scenarios/hospital.py`)

9-node hospital floor plan with three robot types:

```
entrance ── lobby ──── corridor_a ── icu_entrance ── icu_main
              │              │
         elev_lobby     corridor_b ──────────────────┘
              │              │
           floor2      staff_corridor (restricted)
```

| Scenario | Description |
|----------|-------------|
| 1 | `delivery_bot` (has arm) navigates entrance → ICU; opens closed door |
| 2 | `cargo_bot` (1.1 m wide, no arm) — infeasible; counterfactual identifies two independent blockers |
| 3 | `legged_bot` (no arm) — infeasible; door is the single blocker |
| 4 | Ad-hoc query: "Why not use the staff corridor?" → restricted flag |
| 5 | Live world simulation: door propped open → cost drops from 35.04 → 32.54 |
| 6 | Improvement roadmap for cargo_bot: per-route blocker analysis with effort labels |

### Warehouse (`scenarios/warehouse.py`)

9-node warehouse with wet floor, dark inspection zone, and an 18° ramp to the mezzanine.

| Scenario | Description |
|----------|-------------|
| 1 | `picker_bot` (wheeled) → storage_A; straightforward main aisle |
| 2 | `tracked_bot` (max slope 25°) → mezzanine via ramp; only tracked/legged can do this |
| 3 | "Why can't picker_bot use the ramp?" → slope 18° > 8° limit; suggests ramp installation |
| 4 | `forklift_bot` (1.4 m wide) → storage_A; wide-aisle-only route |

---

## Extending the System

### Add a new space type
```python
onto.defclass("CleanRoom", parent_name="IndoorSpace")
space = onto.create("clean_room_1", "CleanRoom",
    width=3.0, height=2.5, is_accessible=True,
    requires_gowning=True,   # custom property
)
```

### Add a new affordance rule
```python
from ontofact_nav.affordance import AffordanceRule
from ontofact_nav.domain import AffordanceType

nav.reasoner.add_block(AffordanceRule(
    name="block_traversable_requires_gowning",
    affordance=AffordanceType.TRAVERSABLE,
    condition=lambda e, a: (
        e.get("requires_gowning", False)
        and not a.get("has_gown", False)
    ),
    explanation_template="{entity} requires gowning — robot does not have a gown",
))
```

### Add a new robot agent
```python
aerial_bot = onto.create("aerial_bot", "Robot",
    robot_width=0.4, robot_height=0.3,
    min_clearance=0.05, max_slope_angle=90.0,
    mobility_type="aerial", can_open_doors=False,
    has_arm=False, battery_level=0.8, max_speed=5.0,
)
```

### Add a new counterfactual change rule
Open `counterfactual.py` → `_candidate_changes()` and append a candidate to the **pool** (the
minimal-subset search decides whether it is actually needed):
```python
if e.get("requires_gowning", False) and not a.get("has_gown", False):
    changes.append(PropertyChange(
        individual_name=entity.name,
        property_name="requires_gowning",
        original_value=True,
        counterfactual_value=False,
        change_type="disable",
        rationale="Removing gowning requirement for this space.",
    ))
```
If the new property should count as operator-actionable (or carry a particular effort level), add
it to `PROPERTY_TAXONOMY` at the top of `counterfactual.py`.

### Query with SPARQL
```python
# Find all spaces with crowd density above 40%
results = onto.sparql_select("""
    SELECT ?ind ?crowd WHERE {
        ?ind nav:crowd_density ?crowd .
        FILTER (?crowd > 0.4)
    }
    ORDER BY DESC(?crowd)
""")

# After apply_reasoning(), subclass queries work
onto.apply_reasoning("rdfs")
indoor_spaces = onto.sparql_select("SELECT ?ind WHERE { ?ind a nav:IndoorSpace }")
```

---

## Sample Output

```
========================================================================
  NAVIGATION EXPLANATION REPORT
========================================================================
  Robot  : delivery_bot
  Route  : entrance  →  icu_main
  Path   : entrance → lobby → corridor_a → icu_entrance → icu_main  (cost: 35.04)
  Segs   : ['seg_entrance_lobby', 'seg_lobby_corrA', 'seg_corrA_ICUent', 'seg_ICUent_ICUmain']

PATH RATIONALE
------------------------------------------------------------------------
  [1] Chose emergency-designated route through seg_entrance_lobby.
  [2] Traversed seg_lobby_corrA [traversable, passable, climbable, observable, avoidable].
  [3] Squeezed through seg_corrA_ICUent (width 0.9 m, clearance OK).
  [4] Opened seg_ICUent_ICUmain door with manipulation arm (+2.5 cost units).

COUNTERFACTUAL ANALYSIS  (Why Not Other Paths?)
------------------------------------------------------------------------
  [1] Why not: entrance → lobby → corridor_a → corridor_b → icu_entrance → icu_main?
       Alternative : ...  (cost: 47.48)
       No single-property fix found — actual path is already cheaper (Δcost = -12.44).

OPERATOR RECOMMENDATIONS
------------------------------------------------------------------------
  [MEDIUM effort] If robot access to seg_staff_corridor were granted
                  (currently restricted), that route would be available.

EXECUTIVE SUMMARY
------------------------------------------------------------------------
  Robot 'delivery_bot' successfully navigated from 'entrance' to 'icu_main'
  via 4 segment(s) with a total affordance-weighted cost of 35.04.
  1 alternative route(s) were analysed; 0 operator action(s) could unlock shorter paths.
========================================================================
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| SPARQL-driven numeric categories | Combined/threshold space categories (`Steep`, `HighRiskZone`) are inferred by real SPARQL `ASK` with `NavCostConfig`-injected thresholds, gating affordance rules. The ontology genuinely *owns* the numeric inference (OWL-RL closure can't do arithmetic) — deleting the rules changes routes |
| Memoized classifier (pre-compiled) | Classification is a pure function of a few source properties; results are cached on those values, collapsing 1000s of hot-path calls to a handful of distinct SPARQL evaluations. Queries are `prepareQuery`-compiled once and shared, so each eval is ~0.3 ms. Clones reuse the same view ⇒ counterfactual property changes correctly flip the category |
| Agent-relative slope stays numeric | Only the fixed `wheeled_stair_slope_limit` (a config constant) is migrated to SPARQL `nav:Steep`; the per-agent `slope ≤ max_slope_angle` join stays a `NavCostConfig` predicate (no constant to unify). Honest split — the classifier *supports* joint rules but we don't need one here |
| Minimum-cardinality counterfactual search | A genuine ascending-cardinality subset search returns the *smallest* decision-flipping change set — minimal by construction, not the old "apply every applicable fix" union. Bounded by `cf_max_pool` with a backward-elimination fallback (monotone goal) for large pools |
| Independent actionable-only search | `is_achievable` and the advertised saving come from a *separate* search over operator-actionable candidates, so a saving that needs construction is never mislabelled as runtime-achievable (was `any(actionable)`, which could over-promise) |
| Single passability source of truth | Door/slope hard-blocks live in the rule engine (block rules); `navigation_cost` decides feasibility solely from the affordance set. Removes the duplicated raw-property checks that could drift |
| `NavCostConfig` (frozen, injected) | One source for every threshold/weight + the shared geometry predicates; the engine defaults to the reasoner's config so they can't diverge. Defaults reproduce historical numbers |
| Load-bearing class hierarchy | Class-conditioned rules (`requires_class`) and inherited class defaults make the ontology affect planning — changing only an individual's class can change the route. RDF write-through keeps SPARQL in step with live mutations (clones stay graph-less) |
| `PROPERTY_TAXONOMY` single table | `is_actionable()` and `effort()` read one table, so the two orthogonal axes (can-do vs how-hard) can never disagree |
| RDFLib parallel store | Adds real SPARQL without changing the Python dict API; the planner reasons over the Python class hierarchy, while SPARQL/OWL-RL serve querying and inspection |
| OWL-RL via `apply_reasoning()` | Explicit call rather than auto-inference — avoids startup overhead for workloads that don't need subclass SPARQL |
| NetworkX for k-paths | Replaces ~120 lines of hand-rolled Yen's; `shortest_simple_paths` is well-tested and handles loopless paths correctly |
| Custom A\* retained | NetworkX's `astar_path` returns only node IDs; the custom planner returns per-edge `AffordanceResult` objects required by the explanation layer |
| Pydantic dataclasses | Validates inputs at construction time with zero API change; `ValidationError` surfaces malformed nodes/edges immediately rather than silently producing wrong paths |
| Grant/block rule separation | Mirrors OWL's open-world assumption; avoidable affordance is always grantable |
| g-score in heap tuple | Standard A\* lazy-deletion bug: comparing f (= g+h) against g always skips valid nodes |
| `clone()` on individuals | Counterfactual worlds are isolated, graph-less copies; avoids mutating live world state or the RDF graph |
| Claude API narrative optional | `use_claude=False` by default; the framework is fully functional without an API key |
| FastAPI `on_event("startup")` | Scenario is built once at server start; all request handlers share a single `OntofactNavigator` instance |

---

## Development

```bash
# Editable install with test tooling
pip install -e ".[dev]"

# Run the full test suite (115 tests)
pytest

# Run a single test module
pytest tests/test_counterfactual.py -v

# Optional: lint with ruff (configured in pyproject.toml)
ruff check src tests
```

**Test layout** — fixtures live in [`tests/conftest.py`](tests/conftest.py) (a fresh
ontology plus `make_agent` / `make_space` factories with clearly-traversable defaults);
each module overrides only the properties relevant to the behaviour under test.

| Module | Covers |
|--------|--------|
| `test_ontology.py` | Class hierarchy, individuals, `query`, SPARQL, OWL-RL closure |
| `test_affordance.py` | Grant/block rules, hard blockers, exact cost arithmetic, rule validation |
| `test_navigation.py` | Pydantic validation, graph construction, A\*, k-shortest paths |
| `test_counterfactual.py` | Actionability classification, why-not analysis, `OntofactNavigator` |
| `test_config.py` | `NavCostConfig` defaults + injection changing cost/affordance/engine |
| `test_passability_ssot.py` | Door/slope hard-blocks live in the affordance set; no independent re-check |
| `test_ontology_load_bearing.py` | Class alone changes planning; inherited defaults; RDF write-through vs clone isolation |
| `test_counterfactual_minimal.py` | Minimum-cardinality subset; soft fixes excluded; subset-minimality |
| `test_counterfactual_attribution.py` | Actionable-only `is_achievable`; mixed-blocker non-misattribution; axis orthogonality |
| `test_classification.py` | SPARQL `Steep`/`HighRiskZone` inference reroutes A\*; counterfactual flips it; load-bearing on rule removal; memoization bound; agent-relative join capability |
| `test_api.py` | FastAPI endpoints incl. JSON-safety for infeasible-actual why-not |

The package follows the `src/` layout, so always run `pytest` against the **installed**
(editable) package — the configured `tool.pytest.ini_options.testpaths` points at `tests/`.

---

## References

- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.  
- Lewis, D. (1973). *Counterfactuals*. Harvard University Press.  
- Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.  
- Yen, J.Y. (1971). Finding the k Shortest Loopless Paths in a Network. *Management Science* 17(11).  
- Hart, P.E., Nilsson, N.J., Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics* 4(2).
