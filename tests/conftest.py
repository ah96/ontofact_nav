"""Shared pytest fixtures for the ontofact_nav test suite.

Provides a fresh ontology per test plus small factory fixtures for building
robot agents and space individuals with sensible, clearly-traversable defaults.
Individual tests override only the properties relevant to what they exercise.
"""

from __future__ import annotations

import pytest

from ontofact_nav import build_navigation_ontology
from ontofact_nav.affordance import AffordanceReasoner

# A standard wheeled delivery robot: fits typical corridors, can open doors.
DEFAULT_AGENT = dict(
    robot_width=0.6,
    robot_height=1.4,
    min_clearance=0.15,
    max_slope_angle=8.0,
    mobility_type="wheeled",
    can_open_doors=True,
    has_arm=True,
    battery_level=0.9,
    max_speed=1.2,
)

# A clearly traversable corridor: wide, tall, smooth, well-lit, accessible.
DEFAULT_SPACE = dict(
    width=2.0,
    height=2.5,
    surface_type="smooth",
    illumination=1.0,
    is_accessible=True,
)


@pytest.fixture
def onto():
    """A fresh navigation ontology (20 classes, 22 properties, no individuals)."""
    return build_navigation_ontology()


@pytest.fixture
def reasoner():
    """A reasoner pre-loaded with the default affordance rule set."""
    return AffordanceReasoner()


@pytest.fixture
def make_agent(onto):
    """Factory: create a Robot individual, overriding any default property."""
    def _make(name="bot", **overrides):
        return onto.create(name, "Robot", **{**DEFAULT_AGENT, **overrides})
    return _make


@pytest.fixture
def agent(make_agent):
    """A single standard robot agent."""
    return make_agent()


@pytest.fixture
def make_space(onto):
    """Factory: create a space individual with auto-incremented default name."""
    counter = {"n": 0}

    def _make(cls="Corridor", name=None, **props):
        if name is None:
            counter["n"] += 1
            name = f"space_{counter['n']}"
        return onto.create(name, cls, **{**DEFAULT_SPACE, **props})

    return _make
