"""Tests for the affordance rule engine and the navigation-cost formula."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from ontofact_nav import AffordanceType
from ontofact_nav.affordance import AffordanceRule


# ---------------------------------------------------------------------------
# Affordance inference
# ---------------------------------------------------------------------------

def test_clear_corridor_grants_all_core_affordances(reasoner, make_space, agent):
    result = reasoner.compute(make_space(), agent)
    for af in (
        AffordanceType.TRAVERSABLE,
        AffordanceType.PASSABLE,
        AffordanceType.CLIMBABLE,
        AffordanceType.OBSERVABLE,
        AffordanceType.AVOIDABLE,
    ):
        assert result.has(af)


def test_hazardous_space_blocks_traversable(reasoner, make_space, agent):
    result = reasoner.compute(make_space(is_hazardous=True), agent)
    assert result.missing(AffordanceType.TRAVERSABLE)
    assert AffordanceType.TRAVERSABLE in result.blocked_affordances


def test_narrow_corridor_blocks_passable(reasoner, make_space, agent):
    # required width = 0.6 + 2*0.15 = 0.9; 0.8 is too narrow
    result = reasoner.compute(make_space(width=0.8), agent)
    assert result.missing(AffordanceType.PASSABLE)


def test_dark_space_blocks_observable_but_not_traversable(reasoner, make_space, agent):
    result = reasoner.compute(make_space(illumination=0.2), agent)
    assert result.missing(AffordanceType.OBSERVABLE)
    assert result.has(AffordanceType.TRAVERSABLE)


def test_closed_door_openable_only_with_capability(reasoner, make_space, make_agent):
    space = make_space(door_state="closed")
    assert reasoner.compute(space, make_agent("opener", can_open_doors=True)).has(
        AffordanceType.OPENABLE
    )
    assert reasoner.compute(space, make_agent("no_arm", can_open_doors=False)).missing(
        AffordanceType.OPENABLE
    )


# ---------------------------------------------------------------------------
# Navigation cost — hard blockers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("props", [
    {"is_hazardous": True},
    {"restricted": True},
    {"is_accessible": False},
    {"width": 0.5},                 # too narrow to fit
    {"door_state": "locked"},       # always impassable
])
def test_hard_blockers_yield_infinite_cost(reasoner, make_space, agent, props):
    cost, _ = reasoner.navigation_cost(make_space(**props), agent, base_distance=10.0)
    assert cost == math.inf


def test_closed_door_without_capability_is_impassable(reasoner, make_space, make_agent):
    space = make_space(door_state="closed")
    cost, _ = reasoner.navigation_cost(
        space, make_agent("no_arm", can_open_doors=False), base_distance=10.0
    )
    assert cost == math.inf


def test_steep_slope_blocks_wheeled_but_allows_tracked(reasoner, make_space, make_agent):
    ramp = make_space(cls="Ramp", slope_angle=18.0)
    wheeled = make_agent("wheely", mobility_type="wheeled", max_slope_angle=8.0)
    tracked = make_agent("tracky", mobility_type="tracked", max_slope_angle=25.0)

    assert reasoner.navigation_cost(ramp, wheeled, base_distance=10.0)[0] == math.inf
    assert math.isfinite(reasoner.navigation_cost(ramp, tracked, base_distance=10.0)[0])


# ---------------------------------------------------------------------------
# Navigation cost — soft penalties (exact arithmetic)
# ---------------------------------------------------------------------------

def test_clear_corridor_cost_equals_distance(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(make_space(), agent, base_distance=10.0)
    assert cost == pytest.approx(10.0)


def test_wet_surface_adds_friction(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(
        make_space(surface_type="wet"), agent, base_distance=10.0
    )
    assert cost == pytest.approx(11.4)   # 10 + 1.4


def test_closed_openable_door_adds_penalty(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(
        make_space(door_state="closed"), agent, base_distance=10.0
    )
    assert cost == pytest.approx(12.5)   # 10 + 2.5


def test_crowd_density_is_weighted_heavily(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(
        make_space(crowd_density=0.5), agent, base_distance=10.0
    )
    assert cost == pytest.approx(12.0)   # 10 + 0.5*4.0


def test_dark_space_adds_visibility_penalty(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(
        make_space(illumination=0.2), agent, base_distance=10.0
    )
    assert cost == pytest.approx(11.8)   # 10 + 1.8


def test_emergency_route_discount(reasoner, make_space, agent):
    cost, _ = reasoner.navigation_cost(
        make_space(emergency_route=True), agent, base_distance=10.0
    )
    assert cost == pytest.approx(7.5)    # 10 * 0.75


# ---------------------------------------------------------------------------
# Rule registration & validation
# ---------------------------------------------------------------------------

def test_custom_block_rule_takes_effect(reasoner, make_space, make_agent):
    reasoner.add_block(AffordanceRule(
        name="block_requires_gowning",
        affordance=AffordanceType.TRAVERSABLE,
        condition=lambda e, a: e.get("requires_gowning", False) and not a.get("has_gown", False),
        explanation_template="{entity} requires gowning",
    ))
    space = make_space(requires_gowning=True)
    cost, _ = reasoner.navigation_cost(space, make_agent("ungowned"), base_distance=10.0)
    assert cost == math.inf


def test_affordance_rule_rejects_empty_name():
    with pytest.raises(ValidationError):
        AffordanceRule(
            name="",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, a: True,
            explanation_template="x",
        )


def test_affordance_rule_rejects_negative_priority():
    with pytest.raises(ValidationError):
        AffordanceRule(
            name="r",
            affordance=AffordanceType.TRAVERSABLE,
            condition=lambda e, a: True,
            explanation_template="x",
            priority=-1,
        )
