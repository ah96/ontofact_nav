"""
ontofact_nav/orchestrator.py
============================
OntofactNavigator — the single entry-point for the framework.

Wires together all subsystems:
  Ontology + NavigationGraph
    → AffordanceReasoner    (infers affordances from properties)
    → AStarPlanner          (finds optimal + k-alternative paths)
    → CounterfactualEngine  (explains why alternatives were rejected)
    → ExplanationGenerator  (formats results into human-readable reports)

Usage pattern
-------------
1. Build an Ontology via build_navigation_ontology() and populate it with
   space individuals (see scenarios/hospital.py for a worked example).
2. Build a NavigationGraph with NavNode and NavEdge objects.
3. Construct OntofactNavigator(onto, graph).
4. Call navigate(start, goal, agent) to get (NavPath, NavigationExplanation).
5. Call print_report(explanation) to display the three-layer report.
6. Call query_why_not(start, goal, agent, alt_nodes) for ad-hoc queries.

Thread safety
-------------
OntofactNavigator is NOT thread-safe.  The CounterfactualEngine mutates
the NavigationGraph temporarily during Yen's algorithm (edge removal for
loopless path generation) and clones individuals for counterfactual worlds.
Do not share one instance across concurrent requests.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .affordance import AffordanceReasoner
from .counterfactual import Counterfactual, CounterfactualEngine
from .explanation import ExplanationGenerator, NavigationExplanation
from .navigation import AStarPlanner, NavPath, NavigationGraph
from .ontology import Ontology, OntologyIndividual


class OntofactNavigator:
    """
    Top-level façade for the ontofact_nav framework.

    Constructs and owns all subsystem instances.  External callers interact
    only with this class — the internal subsystem APIs are considered private.

    Attributes (public for advanced use)
    ------------------------------------
    reasoner  : AffordanceReasoner   — extend with add_grant() / add_block()
    planner   : AStarPlanner         — access for raw path queries
    cf_engine : CounterfactualEngine — access for custom counterfactuals
    explainer : ExplanationGenerator — access for custom report formatting
    """

    def __init__(
        self,
        onto:  Ontology,
        graph: NavigationGraph,
    ) -> None:
        self.onto  = onto
        self.graph = graph

        # Construct subsystems in dependency order:
        #   1. AffordanceReasoner — no dependencies
        #   2. AStarPlanner       — depends on graph + reasoner
        #   3. CounterfactualEngine — depends on reasoner + planner + onto.individuals
        #   4. ExplanationGenerator — no planner dependency (pure formatting)
        self.reasoner  = AffordanceReasoner()
        self.planner   = AStarPlanner(graph, self.reasoner)
        self.cf_engine = CounterfactualEngine(
            reasoner         = self.reasoner,
            planner          = self.planner,
            onto_individuals = onto.individuals,  # live reference — reflects mutations
        )
        self.explainer = ExplanationGenerator()

    # ------------------------------------------------------------------
    # Primary API: navigate with full explanation
    # ------------------------------------------------------------------

    def navigate(
        self,
        start:          str,
        goal:           str,
        agent:          OntologyIndividual,
        k_alternatives: int  = 3,
        explain:        bool = True,
    ) -> Tuple[NavPath, Optional[NavigationExplanation]]:
        """
        Plan a path from *start* to *goal* for *agent* and (optionally)
        generate a three-layer explanation.

        Parameters
        ----------
        start           : starting node ID
        goal            : goal node ID
        agent           : robot OntologyIndividual
        k_alternatives  : number of alternative paths to analyse (default 3)
        explain         : if False, skip explanation generation (faster)

        Returns
        -------
        (NavPath, NavigationExplanation)  — explanation is None if explain=False
        or if no feasible path was found.
        """
        # Request k_alternatives+1 paths: the best one is chosen, the rest
        # become alternatives for counterfactual analysis.
        paths = self.planner.find_k_paths(
            start, goal, agent, k=k_alternatives + 1
        )

        if not paths:
            print(f"[OntofactNavigator] No feasible path: {start} → {goal}")
            return NavPath.infeasible(), None

        chosen = paths[0]   # lowest-cost path (Yen's returns sorted order)

        if not explain:
            return chosen, None

        candidates = paths[1:]   # remaining paths are the alternatives

        # Generate a counterfactual for each alternative
        cfs = self.cf_engine.batch_why_not(chosen, candidates, agent)

        # Format into a structured explanation
        exp = self.explainer.generate(
            robot           = agent,
            start           = start,
            goal            = goal,
            chosen_path     = chosen,
            candidates      = candidates,
            counterfactuals = cfs,
        )

        return chosen, exp

    # ------------------------------------------------------------------
    # Ad-hoc why-not query
    # ------------------------------------------------------------------

    def query_why_not(
        self,
        start:     str,
        goal:      str,
        agent:     OntologyIndividual,
        alt_nodes: List[str],
    ) -> Counterfactual:
        """
        Answer an operator's ad-hoc question:
          "Why didn't the robot go through *alt_nodes*?"

        Re-plans the actual path first (so the actual-path context is current),
        then generates a counterfactual for the user-specified node sequence.

        This is separate from the explain=True flow because the user may want
        to query an arbitrary path that was not one of the k-shortest candidates.

        Parameters
        ----------
        start     : starting node ID
        goal      : goal node ID
        agent     : robot OntologyIndividual
        alt_nodes : the alternative node sequence to explain

        Returns
        -------
        Counterfactual — includes explanation string, changes, and cost delta.
        """
        # Replan the actual path without generating an explanation (fast)
        actual, _ = self.navigate(start, goal, agent, explain=False)
        return self.cf_engine.explain_why_not(actual, alt_nodes, agent)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def print_report(self, exp: NavigationExplanation) -> None:
        """
        Format *exp* as a human-readable text report and print to stdout.

        The formatted string is also returned implicitly through format_report()
        if the caller needs to redirect output (e.g. to a log file).
        """
        print(self.explainer.format_report(exp))
