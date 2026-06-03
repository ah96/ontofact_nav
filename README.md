# Ontology-Guided Counterfactual Affordance Reasoning  
# for Actionable Robot Navigation Explanations

An explainability framework for mobile robot navigation that combines:

- **Domain ontology** — RDFLib-backed OWL class/property hierarchy; real SPARQL queries; OWL-RL deductive closure  
- **Affordance theory** — forward-chaining rules infer what a robot *can* do in each space  
- **A\* path planning** — affordance-weighted costs; NetworkX k-shortest paths for alternatives  
- **Counterfactual reasoning** — minimal-change analysis answers "Why not path X?"  
- **Natural language generation** — three-layer reports: rationale → counterfactual → recommendations; optional Claude API narrative  
- **REST API** — FastAPI service exposes `navigate`, `why-not`, and SPARQL over HTTP  
- **Visualization** — NetworkX + Matplotlib renders the graph with color-coded path overlays  

---

## Installation

```bash
# From the project root (directory containing pyproject.toml)
pip install -e .

# Or just install dependencies directly
pip install -r ontofact_nav/requirements.txt
```

**Dependencies:** `networkx`, `rdflib`, `owlrl`, `anthropic`, `pydantic`, `fastapi`, `uvicorn`, `matplotlib`  
**Dev/test:** `pytest`, `httpx`

---

## Quick start

```bash
# Run demo scenarios
python3 -m ontofact_nav.main              # both scenarios
python3 -m ontofact_nav.main hospital     # hospital only (6 sub-scenarios)
python3 -m ontofact_nav.main warehouse    # warehouse only (4 sub-scenarios)

# Start the REST API
uvicorn ontofact_nav.api:app --reload

# Run the test suite
pytest tests/
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
11. [References](#references)  

---

## Theoretical Background

### Ontology

An **ontology** is a formal vocabulary of concepts and their relationships for a domain.
This framework uses an OWL-inspired structure (classes, properties, individuals) backed by
**RDFLib** for serialisation and real SPARQL querying, and **OWL-RL** for deductive closure
(subclass-chain propagation, property domain/range inferences).

Every navigable space (corridor, elevator, doorway…) and every robot is an *individual* in
the ontology, typed by a class in the hierarchy and characterised by data properties.

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
the counterfactual engine identifies the **minimal set of world-property changes** that
would make that path feasible and preferred.  This implements the *minimal-change principle*
from counterfactual logic (Lewis 1973; Pearl 2000).

Actionability filtering then separates changes a human operator can make today
(open a door, clear a crowd, fix lighting) from structural changes (widen a corridor).

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
.                                (project root — contains pyproject.toml)
├── pyproject.toml               Packaging, entry points, pytest config
├── tests/
│   ├── conftest.py              Shared fixtures
│   ├── test_affordance.py       Affordance rules + cost formula (25 tests)
│   ├── test_navigation.py       Pydantic validation, A*, k-paths (18 tests)
│   ├── test_ontology.py         Ontology, SPARQL, OWL-RL (14 tests)
│   └── test_counterfactual.py   Counterfactual engine + navigator (6 tests)
└── ontofact_nav/
    ├── __init__.py              Public API re-exports
    ├── main.py                  CLI entry point
    ├── requirements.txt         Runtime + dev dependencies
    ├── ontology.py              Ontology engine (RDFLib + OWL-RL)
    ├── domain.py                Navigation domain schema + enums
    ├── affordance.py            Affordance rule engine (pydantic models)
    ├── navigation.py            NavGraph, A*, NetworkX k-paths (pydantic models)
    ├── counterfactual.py        Minimal-change counterfactual engine
    ├── explanation.py           Three-layer NL generator (+ Claude API)
    ├── orchestrator.py          OntofactNavigator integration class
    ├── visualization.py         Matplotlib/NetworkX graph rendering
    ├── api.py                   FastAPI REST service
    └── scenarios/
        ├── hospital.py          6-scenario hospital floor plan demo
        └── warehouse.py         4-scenario warehouse demo
```

---

## Module Reference

### `ontology.py` — Ontology Engine

Core data structures for the knowledge representation layer, backed by a parallel
**RDFLib** `Graph` for real SPARQL querying and **OWL-RL** deductive closure.

| Class / Method | Description |
|----------------|-------------|
| `OntologyClass` | Node in the class hierarchy. `is_subclass_of()` walks the parent chain. |
| `OntologyProperty` | Property slot descriptor (domain class, Python type). |
| `OntologyIndividual` | Instance with a property dict. `clone()` for counterfactual copies. |
| `Ontology.defclass()` | Declare a class; populates both Python dict and RDF graph. |
| `Ontology.create()` | Factory: construct + register an individual; adds RDF triples. |
| `Ontology.query()` | Python dict-based filter query (class + equality constraints). |
| `Ontology.sparql_select()` | Real SPARQL SELECT via RDFLib (`nav:` prefix pre-injected). |
| `Ontology.apply_reasoning()` | OWL-RL deductive closure (RDFS or full OWL-RL). |

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

> **Note:** `sparql_select` reflects the *initial* world state.
> Counterfactual clones (`individual.clone()`) are ephemeral Python objects
> and are not synced back to the RDF graph.

---

### `domain.py` — Navigation Domain Schema

Builds the navigation-specific ontology (20 classes, 22 properties) and
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

---

### `affordance.py` — Affordance Reasoner

Implements a **forward-chaining rule engine** using two lists of `AffordanceRule` objects
(Pydantic dataclasses — invalid rules are rejected at construction):
- **grant_rules** add an affordance when their condition fires  
- **block_rules** remove an affordance when their condition fires  

Net affordances = granted − blocked.

**Cost formula** (used as A* edge weight):

```
base_cost = edge.distance
+ door_opening_penalty (2.5 if closed-but-openable)
+ surface_friction     (0.0–1.4 depending on SurfaceType)
+ crowd_density × 4.0
+ obstacle_density × 2.5
+ visibility_penalty   (1.8 if not OBSERVABLE)
+ slope × 0.06
× emergency_route_discount (0.75 if designated emergency)
→ math.inf if TRAVERSABLE or PASSABLE missing, or door impassable
```

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
3. Iterate edges: flag those with `cost == inf` or `cost > 1.5 × (distance + 2.0)` as problematic  
4. For each problematic edge, `_minimal_changes()` checks (in priority order):
   - hazard flag → set `is_hazardous = False`  
   - restricted flag → set `restricted = False`  
   - accessibility → set `is_accessible = True`  
   - door state → set to `open`  
   - width too narrow → widen to `robot_width + 2×clearance + 0.1`  
   - slope too steep → reduce to `max_slope_angle − 1°`  
   - crowd density > 50% → reduce to 20%  
   - illumination ≤ 25% → raise to 60%  
5. Apply changes to cloned individuals → re-evaluate → compute cost delta  
6. Return `Counterfactual` with changes tagged as `actionable` or structural

**`PropertyChange.is_actionable()`** returns `True` for properties that a human
operator could realistically modify at runtime (door state, access flags, crowd density,
illumination) versus structural properties (corridor width, slope angle).

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
Open `counterfactual.py` → `_minimal_changes()` and add a condition block:
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
| RDFLib parallel store | Adds real SPARQL without changing the Python dict API; counterfactual clones stay as ephemeral Python objects |
| OWL-RL via `apply_reasoning()` | Explicit call rather than auto-inference — avoids startup overhead for workloads that don't need subclass SPARQL |
| NetworkX for k-paths | Replaces ~120 lines of hand-rolled Yen's; `shortest_simple_paths` is well-tested and handles loopless paths correctly |
| Custom A\* retained | NetworkX's `astar_path` returns only node IDs; the custom planner returns per-edge `AffordanceResult` objects required by the explanation layer |
| Pydantic dataclasses | Validates inputs at construction time with zero API change; `ValidationError` surfaces malformed nodes/edges immediately rather than silently producing wrong paths |
| Grant/block rule separation | Mirrors OWL's open-world assumption; avoidable affordance is always grantable |
| g-score in heap tuple | Standard A\* lazy-deletion bug: comparing f (= g+h) against g always skips valid nodes |
| `PropertyChange.is_actionable()` | Separates "what could change" from "what an operator can change today" — critical for actionability |
| `clone()` on individuals | Counterfactual worlds are isolated copies; avoids mutating live world state |
| Claude API narrative optional | `use_claude=False` by default; the framework is fully functional without an API key |
| FastAPI `on_event("startup")` | Scenario is built once at server start; all request handlers share a single `OntofactNavigator` instance |

---

## References

- Gibson, J.J. (1979). *The Ecological Approach to Visual Perception*. Houghton Mifflin.  
- Lewis, D. (1973). *Counterfactuals*. Harvard University Press.  
- Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.  
- Yen, J.Y. (1971). Finding the k Shortest Loopless Paths in a Network. *Management Science* 17(11).  
- Hart, P.E., Nilsson, N.J., Raphael, B. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics* 4(2).
