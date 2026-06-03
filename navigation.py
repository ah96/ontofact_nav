"""
ontofact_nav/navigation.py
==========================
Navigation graph, A* path planner, and Yen's k-shortest-paths algorithm.

Graph model
-----------
  Nodes  (NavNode)  = waypoints / locations in the environment (e.g. "lobby",
                      "icu_entrance").  Each node has a 2-D position for the
                      Euclidean-distance heuristic.

  Edges  (NavEdge)  = directed passages between locations.  Each edge carries
                      an OntologyIndividual that describes the physical space
                      being traversed (a corridor, doorway, ramp, etc.).
                      Affordance-weighted cost is computed from that individual
                      at planning time — NOT baked into the graph at load time.

Why store the space individual on the edge (not the node)?
  Navigation cost arises from *traversing* a space, not from *being at* a
  location.  A junction node (lobby) has zero traversal cost; crossing the
  corridor segment between lobby and corridor_a has cost proportional to
  that corridor's properties.  Storing the individual on the edge keeps the
  graph semantics clean and makes counterfactual world-mutation straightforward:
  replace one individual → all edges referencing it recompute their cost.

A* notes
--------
  Heap entries are (f_score, g_score, counter, node_id).

  Why store g_score in the heap?
    The standard lazy-deletion pattern removes stale heap entries by checking
    whether the stored value still matches the current best distance.  The
    common mistake is comparing the stored *f*-score against g[node] — this
    check is ALWAYS true (f = g + h ≥ g), so every entry is incorrectly
    discarded as stale.  Storing g_score and comparing g_stored > g[node]
    is the correct staleness test.

  The counter tiebreaker prevents Python from trying to compare node_id strings
  when two heap entries have equal f-scores and g-scores (which would TypeError
  on non-comparable types).

Yen's k-shortest paths
----------------------
  Finds up to k loopless paths in ascending cost order by repeatedly:
    1. Choosing a spur node along the last found path.
    2. Temporarily removing edges/nodes that would produce a duplicate prefix.
    3. Running A* from the spur node to the goal.
    4. Combining the root portion + spur path into a candidate.
  Candidates are stored in a min-heap B; the cheapest candidate becomes the
  next confirmed path.

  Reference: Yen (1971), "Finding the k Shortest Loopless Paths in a Network."
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .affordance import AffordanceReasoner, AffordanceResult
from .ontology import OntologyIndividual


# ---------------------------------------------------------------------------
# Graph data structures
# ---------------------------------------------------------------------------

@pydantic_dataclass(config=ConfigDict(eq=False))
class NavNode:
    """
    A waypoint / location in the navigation environment.

    Attributes
    ----------
    node_id  : str                   — unique identifier (matches ontology or map)
    position : Tuple[float, float]   — (x, y) in metres; used by Euclidean heuristic
    label    : str                   — human-readable name for reports
    """
    node_id:  str
    position: Tuple[float, float]
    label:    str = ""

    @field_validator("node_id")
    @classmethod
    def _node_id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("node_id must not be empty")
        return v

    @field_validator("position")
    @classmethod
    def _position_valid(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2 or not all(math.isfinite(c) for c in v):
            raise ValueError("position must be a 2-tuple of finite floats")
        return v

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NavNode) and self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True, eq=False))
class NavEdge:
    """
    A directed edge representing a navigable passage.

    The *space_individual* carries all physical properties (width, surface_type,
    door_state, …) that the AffordanceReasoner uses to compute traversal cost.

    Attributes
    ----------
    from_id          : str                  — origin node id
    to_id            : str                  — destination node id
    space_individual : OntologyIndividual   — the physical space being traversed
    distance         : float                — Euclidean length (metres)
    """
    from_id:          str
    to_id:            str
    space_individual: OntologyIndividual
    distance:         float = 1.0

    @field_validator("from_id", "to_id")
    @classmethod
    def _ids_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("from_id and to_id must not be empty")
        return v

    @field_validator("distance")
    @classmethod
    def _distance_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"distance must be > 0, got {v}")
        return v

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NavEdge):
            return False
        return (
            self.from_id == other.from_id
            and self.to_id == other.to_id
            and self.space_individual.name == other.space_individual.name
        )

    def __hash__(self) -> int:
        return hash((self.from_id, self.to_id, self.space_individual.name))


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NavPath:
    """
    A complete navigation path, returned by the planner.

    Attributes
    ----------
    nodes              : List[str]            — ordered waypoint IDs
    edges              : List[NavEdge]        — ordered traversed passages
    total_cost         : float                — sum of affordance-weighted edge costs
    affordance_results : List[AffordanceResult] — per-edge inference results
    is_feasible        : bool                 — False if no path exists
    """
    nodes:               List[str]
    edges:               List[NavEdge]
    total_cost:          float
    affordance_results:  List[AffordanceResult]
    is_feasible:         bool = True

    @field_validator("total_cost")
    @classmethod
    def _cost_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"total_cost must be >= 0, got {v}")
        return v

    def segment_names(self) -> List[str]:
        """Return the individual name of each traversed passage."""
        return [e.space_individual.name for e in self.edges]

    def summary(self) -> str:
        """One-line human-readable path summary."""
        if not self.is_feasible or not self.nodes:
            return "⚠  no feasible path"
        return " → ".join(self.nodes) + f"  (cost: {self.total_cost:.2f})"

    def __len__(self) -> int:
        return len(self.nodes)

    @staticmethod
    def infeasible() -> "NavPath":
        """Convenience constructor for the 'no path found' sentinel."""
        return NavPath(
            nodes=[], edges=[], total_cost=math.inf,
            affordance_results=[], is_feasible=False
        )


# ---------------------------------------------------------------------------
# Navigation graph
# ---------------------------------------------------------------------------

class NavigationGraph:
    """
    Weighted directed graph stored as an adjacency list.

    Adjacency list structure: ``edges[from_id] → [NavEdge, …]``

    ``add_edge(edge, bidirectional=True)`` automatically adds the reverse
    edge so callers don't need to manually specify both directions for
    symmetric corridors.

    Graph mutation during Yen's algorithm
    -------------------------------------
    Yen's temporarily removes edges (``remove_edge``) and nodes (by popping
    from the nodes dict) to prevent the spur-path search from using the same
    prefix.  Both operations are reversed (``restore_edge``, nodes.update)
    after each spur-path search.  This in-place mutation is safe here because
    the planner runs on a single thread.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, NavNode]       = {}
        # defaultdict so accessing a missing key returns [] instead of KeyError
        self.edges: Dict[str, List[NavEdge]] = defaultdict(list)

    # ------------------------------------------------------------------

    def add_node(self, node: NavNode) -> NavNode:
        """Register a waypoint.  Returns the node for chaining."""
        self.nodes[node.node_id] = node
        return node

    def add_edge(
        self,
        edge:          NavEdge,
        bidirectional: bool = True,
    ) -> None:
        """
        Add a directed edge.  When *bidirectional* is True (the default),
        also adds the reverse edge sharing the same space_individual —
        traversal cost in the reverse direction uses the same physical space.
        """
        self.edges[edge.from_id].append(edge)
        if bidirectional:
            reverse = NavEdge(
                from_id          = edge.to_id,
                to_id            = edge.from_id,
                space_individual = edge.space_individual,  # same physical space
                distance         = edge.distance,
            )
            self.edges[edge.to_id].append(reverse)

    def neighbors(self, node_id: str) -> List[NavEdge]:
        """
        Return all outgoing edges from *node_id*.

        Uses dict.get() (not defaultdict[]) to avoid creating an empty list
        entry for every node queried during A*.
        """
        return list(self.edges.get(node_id, []))

    def all_space_individuals(self) -> Iterator[OntologyIndividual]:
        """Iterate over unique space individuals referenced by any edge."""
        seen: Set[str] = set()
        for edge_list in self.edges.values():
            for e in edge_list:
                if e.space_individual.name not in seen:
                    seen.add(e.space_individual.name)
                    yield e.space_individual

    def _euclidean(self, a: str, b: str) -> float:
        """Euclidean distance between two node positions (metres)."""
        na, nb = self.nodes[a], self.nodes[b]
        return math.hypot(
            na.position[0] - nb.position[0],
            na.position[1] - nb.position[1],
        )

    # ------------------------------------------------------------------
    # Yen's algorithm support: temporary graph mutation
    # ------------------------------------------------------------------

    def remove_edge(self, from_id: str, edge: NavEdge) -> bool:
        """
        Temporarily remove *edge* from the adjacency list of *from_id*.
        Returns True if found and removed, False if the edge was already absent.
        """
        try:
            self.edges[from_id].remove(edge)
            return True
        except ValueError:
            return False

    def restore_edge(self, from_id: str, edge: NavEdge) -> None:
        """Re-add a previously removed edge (Yen's graph restoration step)."""
        self.edges[from_id].append(edge)


# ---------------------------------------------------------------------------
# A* path planner
# ---------------------------------------------------------------------------

class AStarPlanner:
    """
    A* search using affordance-weighted edge costs.

    The heuristic is Euclidean distance between node positions, which is
    admissible (never overestimates) because all edge costs ≥ edge.distance
    and cost is measured in the same units as Euclidean distance.
    """

    def __init__(
        self,
        graph:    NavigationGraph,
        reasoner: AffordanceReasoner,
    ) -> None:
        self.graph    = graph
        self.reasoner = reasoner

    # ------------------------------------------------------------------

    def _heuristic(self, a: str, b: str) -> float:
        """Euclidean heuristic h(a, b) — admissible for this cost function."""
        return self.graph._euclidean(a, b)

    def find_path(
        self,
        start: str,
        goal:  str,
        agent: OntologyIndividual,
    ) -> NavPath:
        """
        Standard A* search returning the lowest-cost path from *start* to *goal*.

        Returns NavPath.infeasible() if no path exists.

        Heap entry format: (f_score, g_score, counter, node_id)

        The *g_score* stored in the heap entry is compared against the
        current best g[node_id] when the entry is popped.  If g_stored > g[node]
        (a shorter path was found later), the entry is discarded as stale.

        Why not compare f_stored against g[node]?
          f = g + h ≥ g always, so that comparison would mark EVERY entry as
          stale and the algorithm would never expand any node.
        """
        if start not in self.graph.nodes or goal not in self.graph.nodes:
            return NavPath.infeasible()

        open_heap: List[Tuple[float, float, int, str]] = []
        counter   = itertools.count()   # tiebreaker to avoid comparing node_ids
        heapq.heappush(open_heap, (0.0, 0.0, next(counter), start))

        # g[node] = best known cost from start; defaults to inf for unvisited nodes
        g: Dict[str, float] = defaultdict(lambda: math.inf)
        g[start] = 0.0

        # came_from[node] = (predecessor_node, edge, affordance_result)
        # start has no predecessor (None sentinel)
        came_from: Dict[str, Optional[Tuple[str, NavEdge, AffordanceResult]]] = {
            start: None
        }

        while open_heap:
            f_val, g_stored, _, current = heapq.heappop(open_heap)

            # Goal reached — reconstruct and return the path
            if current == goal:
                return self._reconstruct(start, goal, came_from, g[goal])

            # Staleness check: this heap entry was overtaken by a cheaper path
            if g_stored > g[current] + 1e-9:
                continue

            # Expand neighbours
            for edge in self.graph.neighbors(current):
                # Skip edges to nodes that have been temporarily hidden by Yen's
                if edge.to_id not in self.graph.nodes:
                    continue

                cost, af = self.reasoner.navigation_cost(
                    edge.space_individual,
                    agent,
                    base_distance=edge.distance,
                )
                # math.inf cost = impassable edge; skip without relaxation
                if cost == math.inf:
                    continue

                tentative_g = g[current] + cost
                if tentative_g < g[edge.to_id]:
                    # Found a better path to edge.to_id — relax
                    g[edge.to_id]         = tentative_g
                    came_from[edge.to_id] = (current, edge, af)
                    h = self._heuristic(edge.to_id, goal)
                    heapq.heappush(
                        open_heap,
                        (tentative_g + h, tentative_g, next(counter), edge.to_id),
                    )

        # Heap exhausted without reaching goal
        return NavPath.infeasible()

    def _reconstruct(
        self,
        start:      str,
        goal:       str,
        came_from:  Dict,
        total_cost: float,
    ) -> NavPath:
        """
        Back-trace *came_from* from goal to start, building the ordered
        node/edge/affordance lists.
        """
        nodes: List[str]             = []
        edges: List[NavEdge]         = []
        afs:   List[AffordanceResult] = []

        cur = goal
        while cur != start:
            prev, edge, af = came_from[cur]
            # Insert at front to build in source-to-goal order
            nodes.insert(0, cur)
            edges.insert(0, edge)
            afs.insert(0, af)
            cur = prev
        nodes.insert(0, start)   # add the start node (has no edge)

        return NavPath(
            nodes=nodes,
            edges=edges,
            total_cost=total_cost,
            affordance_results=afs,
            is_feasible=True,
        )

    # ------------------------------------------------------------------
    # Yen's k-shortest loopless paths
    # ------------------------------------------------------------------

    def find_k_paths(
        self,
        start: str,
        goal:  str,
        agent: OntologyIndividual,
        k:     int = 4,
    ) -> List[NavPath]:
        """
        Find up to *k* distinct loopless paths from *start* to *goal*,
        ordered by ascending affordance-weighted cost.

        Uses NetworkX ``shortest_simple_paths`` (Yen's k-shortest loopless
        paths algorithm) over a temporary agent-specific weighted DiGraph.
        Infinite-cost edges are excluded so the planner never routes through
        impassable spaces.  Each NetworkX node sequence is then converted to
        a full NavPath (with per-edge AffordanceResult) via evaluate_sequence.

        Returns
        -------
        List of NavPath objects (length ≤ k).  May be shorter if fewer than
        k distinct loopless paths exist in the graph.
        """
        # Build an agent-specific weighted DiGraph (finite edges only).
        # This is a temporary structure: it does not replace self.graph and
        # is discarded after this call.
        wg: nx.DiGraph = nx.DiGraph()
        for nid in self.graph.nodes:
            wg.add_node(nid)

        for from_id, edge_list in self.graph.edges.items():
            for edge in edge_list:
                if edge.to_id not in self.graph.nodes:
                    continue
                cost, _ = self.reasoner.navigation_cost(
                    edge.space_individual, agent, base_distance=edge.distance
                )
                if cost < math.inf:
                    # When multiple edges share (from, to) keep the cheaper one.
                    existing = wg.get_edge_data(from_id, edge.to_id)
                    if existing is None or existing["weight"] > cost:
                        wg.add_edge(from_id, edge.to_id, weight=cost)

        paths: List[NavPath] = []
        try:
            for node_seq in nx.shortest_simple_paths(wg, start, goal, weight="weight"):
                path = self.evaluate_sequence(node_seq, agent)
                if path.is_feasible:
                    paths.append(path)
                if len(paths) >= k:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return paths

    # ------------------------------------------------------------------
    # Evaluate an explicit node sequence (for counterfactual use)
    # ------------------------------------------------------------------

    def evaluate_sequence(
        self,
        node_sequence: List[str],
        agent:         OntologyIndividual,
    ) -> NavPath:
        """
        Compute the cost of a specific node sequence WITHOUT replanning.

        Used by the counterfactual engine to evaluate user-specified
        alternative paths ("Why not go via room_b → room_c?").

        Returns NavPath.infeasible() if any edge in the sequence is missing
        from the graph or has infinite cost in the current world.
        """
        if len(node_sequence) < 2:
            return NavPath(node_sequence, [], 0.0, [], True)

        edges: List[NavEdge]          = []
        afs:   List[AffordanceResult] = []
        total  = 0.0

        for i in range(len(node_sequence) - 1):
            fid, tid = node_sequence[i], node_sequence[i + 1]

            # Find the first matching edge (there should be exactly one)
            matches = [e for e in self.graph.neighbors(fid) if e.to_id == tid]
            if not matches:
                return NavPath.infeasible()   # no edge between these nodes

            edge = matches[0]
            cost, af = self.reasoner.navigation_cost(
                edge.space_individual, agent, base_distance=edge.distance
            )

            if cost == math.inf:
                # Sequence is infeasible at this edge; return partial info
                return NavPath(
                    nodes              = node_sequence,
                    edges              = edges + [edge],
                    total_cost         = math.inf,
                    affordance_results = afs + [af],
                    is_feasible        = False,
                )

            total += cost
            edges.append(edge)
            afs.append(af)

        return NavPath(
            nodes              = node_sequence,
            edges              = edges,
            total_cost         = total,
            affordance_results = afs,
            is_feasible        = True,
        )
