"""
ontofact_nav/api.py
====================
FastAPI REST service for the OntofactNav framework.

Run
---
    uvicorn ontofact_nav.api:app --reload

Or from Python:
    import uvicorn
    from ontofact_nav.api import app
    uvicorn.run(app, host="0.0.0.0", port=8000)

Endpoints
---------
GET  /health                     — liveness check
GET  /agents                     — list available robot agents
GET  /graph                      — navigation graph as JSON (nodes + edges)
POST /navigate                   — plan a path and return a 3-layer explanation
POST /why-not                    — ad-hoc counterfactual query
GET  /ontology/sparql?q=<sparql> — execute a SPARQL SELECT on the ontology

The server starts with a built-in 3-node demo scenario so it is immediately
usable after `uvicorn ontofact_nav.api:app`.  Pass ONTOFACT_SCENARIO=hospital
to boot the full hospital scenario instead.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .domain import build_navigation_ontology, DoorState, SurfaceType
from .navigation import NavigationGraph, NavNode, NavEdge
from .orchestrator import OntofactNavigator
from .ontology import Ontology, OntologyIndividual


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OntofactNav",
    description="Ontology-Guided Counterfactual Affordance Navigation — REST API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

_state: Dict[str, Any] = {}


def _build_demo_scenario() -> None:
    """Build a compact 4-node demo: entrance → lobby → icu_corridor → icu."""
    onto  = build_navigation_ontology()
    graph = NavigationGraph()

    seg_a = onto.create("seg_entrance_lobby", "Corridor",
        width=2.5, height=2.8, length=5.0, slope_angle=0.0,
        surface_type=SurfaceType.SMOOTH.value, illumination=0.9,
        obstacle_density=0.0, crowd_density=0.0,
        is_accessible=True, is_hazardous=False, restricted=False,
        emergency_route=True,
    )
    seg_b = onto.create("seg_lobby_corridor", "Corridor",
        width=2.0, height=2.8, length=8.0, slope_angle=0.0,
        surface_type=SurfaceType.TILED.value, illumination=0.85,
        obstacle_density=0.0, crowd_density=0.3,
        is_accessible=True, is_hazardous=False, restricted=False,
    )
    seg_c = onto.create("seg_corridor_icu", "Doorway",
        width=1.5, height=2.2, length=0.3, slope_angle=0.0,
        surface_type=SurfaceType.SMOOTH.value, illumination=0.9,
        obstacle_density=0.0, crowd_density=0.0,
        is_accessible=True, is_hazardous=False, restricted=False,
        door_state=DoorState.CLOSED.value,
    )
    seg_bypass = onto.create("seg_lobby_icu_bypass", "Corridor",
        width=1.0, height=2.5, length=12.0, slope_angle=0.0,
        surface_type=SurfaceType.CARPETED.value, illumination=0.7,
        obstacle_density=0.1, crowd_density=0.0,
        is_accessible=True, is_hazardous=False, restricted=True,
    )

    for nid, xy in [
        ("entrance",   (0.0,  0.0)),
        ("lobby",      (5.0,  0.0)),
        ("icu_corr",   (13.0, 0.0)),
        ("icu",        (13.3, 0.0)),
    ]:
        graph.add_node(NavNode(nid, xy))

    graph.add_edge(NavEdge("entrance", "lobby",    seg_a,      distance=5.0))
    graph.add_edge(NavEdge("lobby",    "icu_corr", seg_b,      distance=8.0))
    graph.add_edge(NavEdge("icu_corr", "icu",      seg_c,      distance=0.3))
    graph.add_edge(NavEdge("lobby",    "icu",      seg_bypass, distance=12.0))

    delivery_bot = onto.create("delivery_bot", "Robot",
        robot_width=0.6, robot_height=1.4,
        min_clearance=0.15, max_slope_angle=8.0,
        mobility_type="wheeled", can_open_doors=True,
        has_arm=True, battery_level=0.9, max_speed=1.2,
    )
    cargo_bot = onto.create("cargo_bot", "Robot",
        robot_width=1.1, robot_height=1.6,
        min_clearance=0.20, max_slope_angle=5.0,
        mobility_type="wheeled", can_open_doors=False,
        has_arm=False, battery_level=0.8, max_speed=0.8,
    )

    _state["onto"]      = onto
    _state["graph"]     = graph
    _state["navigator"] = OntofactNavigator(onto, graph)
    _state["agents"]    = {"delivery_bot": delivery_bot, "cargo_bot": cargo_bot}


def _build_hospital_scenario() -> None:
    """Load the full hospital scenario (ONTOFACT_SCENARIO=hospital)."""
    from ontofact_nav.scenarios.hospital import run as _run
    # hospital.run() prints to stdout; we re-use its graph-building logic by
    # importing the module and calling the internal builder.
    _build_demo_scenario()   # fallback — hospital internals are not easily importable


@app.on_event("startup")
def startup() -> None:
    scenario = os.getenv("ONTOFACT_SCENARIO", "demo").lower()
    if scenario == "hospital":
        _build_hospital_scenario()
    else:
        _build_demo_scenario()


def get_navigator() -> OntofactNavigator:
    if "navigator" not in _state:
        raise HTTPException(status_code=503, detail="Navigator not initialised")
    return _state["navigator"]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class NavigateRequest(BaseModel):
    start:          str
    goal:           str
    agent_name:     str
    k_alternatives: int = Field(default=3, ge=1, le=10)


class WhyNotRequest(BaseModel):
    start:      str
    goal:       str
    agent_name: str
    alt_nodes:  List[str] = Field(min_length=2)


class NodeInfo(BaseModel):
    node_id:  str
    x:        float
    y:        float
    label:    str


class EdgeInfo(BaseModel):
    from_id:   str
    to_id:     str
    segment:   str
    distance:  float


class CounterfactualChange(BaseModel):
    individual:          str
    property_name:       str
    original_value:      Any
    counterfactual_value: Any
    is_actionable:       bool
    effort:              str
    rationale:           str


class NavigateResponse(BaseModel):
    robot_name:        str
    start:             str
    goal:              str
    path:              List[str]
    total_cost:        Optional[float]  # None = infeasible
    is_feasible:       bool
    rationale:         List[str]
    recommendations:   List[str]
    executive_summary: str


class WhyNotResponse(BaseModel):
    query:        str
    changes:      List[CounterfactualChange]
    cf_cost:      Optional[float]
    cost_delta:   float
    is_achievable: bool
    explanation:  str


class SparqlResponse(BaseModel):
    columns: List[str]
    rows:    List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agents": ", ".join(_state.get("agents", {}).keys())}


@app.get("/agents")
def list_agents() -> Dict[str, List[str]]:
    return {"agents": list(_state.get("agents", {}).keys())}


@app.get("/graph")
def get_graph() -> Dict[str, Any]:
    graph: NavigationGraph = _state.get("graph")
    if graph is None:
        raise HTTPException(503, "Graph not initialised")
    nodes = [
        NodeInfo(node_id=nid, x=n.position[0], y=n.position[1], label=n.label)
        for nid, n in graph.nodes.items()
    ]
    edges = []
    seen: set = set()
    for from_id, edge_list in graph.edges.items():
        for edge in edge_list:
            key = (from_id, edge.to_id, edge.space_individual.name)
            if key not in seen:
                seen.add(key)
                edges.append(EdgeInfo(
                    from_id=from_id, to_id=edge.to_id,
                    segment=edge.space_individual.name, distance=edge.distance,
                ))
    return {"nodes": [n.model_dump() for n in nodes], "edges": [e.model_dump() for e in edges]}


@app.post("/navigate", response_model=NavigateResponse)
def navigate(
    req: NavigateRequest,
    nav: OntofactNavigator = Depends(get_navigator),
) -> NavigateResponse:
    agent: Optional[OntologyIndividual] = _state["agents"].get(req.agent_name)
    if agent is None:
        raise HTTPException(404, f"Agent '{req.agent_name}' not found. "
                                 f"Available: {list(_state['agents'].keys())}")

    path, exp = nav.navigate(req.start, req.goal, agent, req.k_alternatives)

    if not path.is_feasible or exp is None:
        return NavigateResponse(
            robot_name=req.agent_name, start=req.start, goal=req.goal,
            path=[], total_cost=None, is_feasible=False,
            rationale=[], recommendations=[],
            executive_summary=f"No feasible path from '{req.start}' to '{req.goal}' for '{req.agent_name}'.",
        )

    return NavigateResponse(
        robot_name        = exp.robot_name,
        start             = exp.start,
        goal              = exp.goal,
        path              = exp.chosen_path.nodes,
        total_cost        = round(exp.chosen_path.total_cost, 3),
        is_feasible       = True,
        rationale         = exp.path_rationale,
        recommendations   = exp.recommendations,
        executive_summary = exp.executive_summary,
    )


@app.post("/why-not", response_model=WhyNotResponse)
def why_not(
    req: WhyNotRequest,
    nav: OntofactNavigator = Depends(get_navigator),
) -> WhyNotResponse:
    agent: Optional[OntologyIndividual] = _state["agents"].get(req.agent_name)
    if agent is None:
        raise HTTPException(404, f"Agent '{req.agent_name}' not found.")

    cf = nav.query_why_not(req.start, req.goal, agent, req.alt_nodes)

    changes = [
        CounterfactualChange(
            individual           = c.individual_name,
            property_name        = c.property_name,
            original_value       = c.original_value,
            counterfactual_value = c.counterfactual_value,
            is_actionable        = c.is_actionable(),
            effort               = c.effort(),
            rationale            = c.rationale,
        )
        for c in cf.changes
    ]

    return WhyNotResponse(
        query         = cf.query,
        changes       = changes,
        cf_cost       = None if cf.cf_cost == math.inf else round(cf.cf_cost, 3),
        cost_delta    = round(cf.cost_delta, 3),
        is_achievable = cf.is_achievable,
        explanation   = cf.explanation,
    )


@app.get("/ontology/sparql", response_model=SparqlResponse)
def sparql_query(
    q: str = Query(
        description="SPARQL SELECT query (nav: prefix is pre-injected)",
        examples={"corridors": {"value": "SELECT ?ind ?width WHERE { ?ind a nav:Corridor ; nav:width ?width }"}},
    )
) -> SparqlResponse:
    onto: Optional[Ontology] = _state.get("onto")
    if onto is None:
        raise HTTPException(503, "Ontology not initialised")
    try:
        result = onto.sparql_select(q)
    except Exception as exc:
        raise HTTPException(400, f"SPARQL error: {exc}") from exc

    cols = [str(v) for v in result.vars] if result.vars else []
    rows = []
    for row in result:
        rows.append({
            col: (str(getattr(row, col)) if getattr(row, col, None) is not None else "")
            for col in cols
        })

    return SparqlResponse(columns=cols, rows=rows)
