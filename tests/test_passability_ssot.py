"""Issue 4 — single source of truth: navigation_cost decides feasibility purely
from the affordance set (door/slope hard-blocks folded into block rules)."""

from __future__ import annotations

import math

from ontofact_nav import AffordanceType, NavCostConfig
from ontofact_nav.affordance import AffordanceReasoner


def test_door_locked_blocks_traversable(reasoner, make_space, agent):
    # The door hard-block is now visible in the affordance set, not only as inf cost.
    res = reasoner.compute(make_space(door_state="locked"), agent)
    assert res.missing(AffordanceType.TRAVERSABLE)
    assert reasoner.navigation_cost(make_space(door_state="locked"), agent, 10.0)[0] == math.inf


def test_closed_uncrossable_blocks_traversable(reasoner, make_space, make_agent):
    res = reasoner.compute(make_space(door_state="closed"),
                           make_agent("noarm", can_open_doors=False))
    assert res.missing(AffordanceType.TRAVERSABLE)


def test_closed_crossable_keeps_traversable(reasoner, make_space, make_agent):
    res = reasoner.compute(make_space(door_state="closed"),
                           make_agent("opener", can_open_doors=True))
    assert res.has(AffordanceType.TRAVERSABLE)
    assert res.has(AffordanceType.OPENABLE)   # drives the +door soft cost


def test_slope_hardblock_lives_in_climbable(make_space, make_agent):
    r = AffordanceReasoner()
    wheeled = make_agent("w", mobility_type="wheeled", max_slope_angle=8.0)
    ramp = make_space(cls="Ramp", slope_angle=18.0)
    assert r.compute(ramp, wheeled).missing(AffordanceType.CLIMBABLE)
    assert r.navigation_cost(ramp, wheeled, 10.0)[0] == math.inf


def test_navigation_cost_has_no_independent_slope_recheck(make_space, make_agent):
    # If CLIMBABLE is granted (permissive config), the SAME steep ramp becomes
    # finite — proving navigation_cost gates only on the affordance set and does
    # NOT re-read slope to make an independent feasibility decision.
    permissive = AffordanceReasoner(config=NavCostConfig(wheeled_stair_slope_limit=99.0))
    wheeled = make_agent("w2", mobility_type="wheeled", max_slope_angle=99.0)
    ramp = make_space(cls="Ramp", slope_angle=18.0)
    cost, res = permissive.navigation_cost(ramp, wheeled, 10.0)
    assert res.has(AffordanceType.CLIMBABLE)
    assert math.isfinite(cost)
