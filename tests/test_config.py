"""Issue 5 — magic numbers centralised in an injectable NavCostConfig."""

from __future__ import annotations

import pytest

from ontofact_nav import AffordanceType, NavCostConfig
from ontofact_nav.affordance import AffordanceReasoner
from ontofact_nav.counterfactual import CounterfactualEngine
from ontofact_nav.navigation import AStarPlanner, NavigationGraph


def test_config_defaults_reproduce_legacy_numbers(reasoner, make_space, agent):
    cfg = NavCostConfig()
    assert cfg.crowd_cost_weight == 4.0
    assert cfg.door_opening_penalty == 2.5
    assert cfg.visibility_penalty == 1.8
    assert cfg.emergency_discount == 0.75
    assert cfg.observable_threshold == 0.25
    # The default reasoner reproduces the historical cost exactly.
    cost, _ = reasoner.navigation_cost(make_space(crowd_density=0.5), agent, 10.0)
    assert cost == pytest.approx(12.0)   # 10 + 0.5 * 4.0


def test_injected_config_changes_cost(make_space, make_agent):
    r = AffordanceReasoner(config=NavCostConfig(crowd_cost_weight=10.0))
    cost, _ = r.navigation_cost(make_space(crowd_density=0.5), make_agent(), 10.0)
    assert cost == pytest.approx(15.0)   # 10 + 0.5 * 10.0


def test_injected_threshold_changes_affordance(make_space, make_agent):
    a = make_agent()
    strict = AffordanceReasoner(config=NavCostConfig(observable_threshold=0.5))
    assert strict.compute(make_space(illumination=0.4), a).missing(AffordanceType.OBSERVABLE)
    # The default reasoner (threshold 0.25) would grant OBSERVABLE for 0.4.
    assert AffordanceReasoner().compute(make_space(illumination=0.4), a).has(
        AffordanceType.OBSERVABLE
    )


def test_config_is_frozen():
    cfg = NavCostConfig()
    with pytest.raises(Exception):
        cfg.crowd_cost_weight = 9.0   # frozen dataclass → cannot reassign


def test_engine_defaults_to_reasoner_config():
    r = AffordanceReasoner()
    engine = CounterfactualEngine(r, AStarPlanner(NavigationGraph(), r), {})
    assert engine.config is r.config   # shared policy → cannot drift
