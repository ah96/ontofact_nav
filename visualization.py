"""
ontofact_nav/visualization.py
==============================
Navigation graph visualization using NetworkX and Matplotlib.

Entry point
-----------
    from ontofact_nav.visualization import draw_navigation_graph

    draw_navigation_graph(
        graph,
        chosen_path=path,
        alternative_paths=alternatives,
        reasoner=nav.reasoner,
        agent=robot,
        title="Hospital — delivery_bot",
    )

Color coding
------------
Nodes:
  green  — start of the chosen path
  red    — goal of the chosen path
  blue   — intermediate node on any highlighted path
  grey   — unreferenced node

Edges:
  green solid    — edge on the chosen path
  orange solid   — edge on an alternative path (not chosen)
  red dashed     — impassable edge for this agent (cost = ∞)
  light grey     — default / background edge

Edge labels (when *reasoner* and *agent* are provided):
  Finite cost shown as a decimal; ∞ for impassable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in all envs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from .affordance import AffordanceReasoner
from .navigation import NavigationGraph, NavPath
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_C = {
    "chosen":      "#2ecc71",   # green
    "alternative": "#e67e22",   # orange
    "impassable":  "#e74c3c",   # red
    "default":     "#bdc3c7",   # light grey
    "node_start":  "#27ae60",   # dark green
    "node_goal":   "#c0392b",   # dark red
    "node_active": "#2980b9",   # blue
    "node_base":   "#95a5a6",   # grey
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_navigation_graph(
    graph:             NavigationGraph,
    chosen_path:       Optional[NavPath]       = None,
    alternative_paths: Optional[List[NavPath]] = None,
    reasoner:          Optional[AffordanceReasoner]   = None,
    agent:             Optional[OntologyIndividual]   = None,
    title:             str                     = "Navigation Graph",
    save_path:         Optional[str]           = None,
    show:              bool                    = True,
    figsize:           Tuple[int, int]         = (13, 8),
) -> plt.Figure:
    """
    Render *graph* as a directed 2-D layout, optionally highlighting paths.

    Parameters
    ----------
    graph             : the NavigationGraph to render
    chosen_path       : path to highlight in green
    alternative_paths : list of alternative paths to highlight in orange
    reasoner + agent  : when both supplied, edge labels show affordance costs
    title             : figure title
    save_path         : if given, save the figure to this file path (PNG/SVG/…)
    show              : call plt.show() when True (set False in headless scripts)
    figsize           : (width, height) in inches

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")

    # ── Build a networkx DiGraph for drawing ─────────────────────────────────
    g = nx.DiGraph()
    pos: Dict[str, Tuple[float, float]] = {}

    for nid, node in graph.nodes.items():
        g.add_node(nid)
        pos[nid] = node.position

    for from_id, edge_list in graph.edges.items():
        for edge in edge_list:
            if edge.to_id in graph.nodes:
                g.add_edge(from_id, edge.to_id, edge_obj=edge)

    # ── Categorise edges ──────────────────────────────────────────────────────
    chosen_edges:      Set[Tuple[str, str]] = set()
    alternative_edges: Set[Tuple[str, str]] = set()
    impassable_edges:  Set[Tuple[str, str]] = set()

    if chosen_path and chosen_path.is_feasible:
        for e in chosen_path.edges:
            chosen_edges.add((e.from_id, e.to_id))

    if alternative_paths:
        for alt in alternative_paths:
            if alt.is_feasible:
                for e in alt.edges:
                    key = (e.from_id, e.to_id)
                    if key not in chosen_edges:
                        alternative_edges.add(key)

    if reasoner and agent:
        for from_id, edge_list in graph.edges.items():
            for edge in edge_list:
                cost, _ = reasoner.navigation_cost(
                    edge.space_individual, agent, base_distance=edge.distance
                )
                if cost == math.inf:
                    impassable_edges.add((from_id, edge.to_id))

    default_edges = [
        (u, v) for u, v in g.edges()
        if (u, v) not in chosen_edges
        and (u, v) not in alternative_edges
        and (u, v) not in impassable_edges
    ]

    # ── Draw edges by category ────────────────────────────────────────────────
    _draw_edges(ax, g, pos, list(default_edges),          _C["default"],     False, 1.0, 0.6)
    _draw_edges(ax, g, pos, list(impassable_edges),       _C["impassable"],  True,  1.5, 0.7)
    _draw_edges(ax, g, pos, list(alternative_edges),      _C["alternative"], False, 2.0, 0.8)
    _draw_edges(ax, g, pos, list(chosen_edges),           _C["chosen"],      False, 2.5, 0.9)

    # ── Categorise and draw nodes ─────────────────────────────────────────────
    start_node = chosen_path.nodes[0]  if (chosen_path and chosen_path.nodes) else None
    goal_node  = chosen_path.nodes[-1] if (chosen_path and chosen_path.nodes) else None
    active_nodes = set()
    if chosen_path:
        active_nodes.update(chosen_path.nodes)
    if alternative_paths:
        for alt in alternative_paths:
            active_nodes.update(alt.nodes)

    node_colors = []
    node_sizes  = []
    for nid in g.nodes():
        if nid == start_node:
            node_colors.append(_C["node_start"])
            node_sizes.append(600)
        elif nid == goal_node:
            node_colors.append(_C["node_goal"])
            node_sizes.append(600)
        elif nid in active_nodes:
            node_colors.append(_C["node_active"])
            node_sizes.append(450)
        else:
            node_colors.append(_C["node_base"])
            node_sizes.append(350)

    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, font_color="white", font_weight="bold", ax=ax)

    # ── Edge cost labels ──────────────────────────────────────────────────────
    if reasoner and agent:
        edge_labels: Dict[Tuple[str, str], str] = {}
        for from_id, edge_list in graph.edges.items():
            for edge in edge_list:
                if edge.to_id not in graph.nodes:
                    continue
                cost, _ = reasoner.navigation_cost(
                    edge.space_individual, agent, base_distance=edge.distance
                )
                label = "∞" if cost == math.inf else f"{cost:.1f}"
                edge_labels[(from_id, edge.to_id)] = label
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_labels,
            font_size=7, font_color="#2c3e50", ax=ax,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7, "ec": "none"},
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=_C["chosen"],      label="Chosen path"),
        mpatches.Patch(color=_C["alternative"], label="Alternative path"),
        mpatches.Patch(color=_C["impassable"],  label="Impassable (∞)"),
        mpatches.Patch(color=_C["default"],     label="Background edge"),
        mpatches.Patch(color=_C["node_start"],  label="Start"),
        mpatches.Patch(color=_C["node_goal"],   label="Goal"),
    ]
    ax.legend(
        handles=legend_items, loc="lower left",
        fontsize=8, framealpha=0.85, ncol=2,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _draw_edges(
    ax:     plt.Axes,
    g:      nx.DiGraph,
    pos:    Dict[str, Tuple[float, float]],
    edges:  List[Tuple[str, str]],
    color:  str,
    dashed: bool,
    width:  float,
    alpha:  float,
) -> None:
    if not edges:
        return
    nx.draw_networkx_edges(
        g, pos,
        edgelist    = edges,
        edge_color  = color,
        style       = "dashed" if dashed else "solid",
        width       = width,
        alpha       = alpha,
        arrows      = True,
        arrowsize   = 14,
        arrowstyle  = "-|>",
        ax          = ax,
        connectionstyle = "arc3,rad=0.08",
    )
