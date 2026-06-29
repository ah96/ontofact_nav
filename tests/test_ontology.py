"""Tests for the ontology engine: hierarchy, individuals, querying, SPARQL, OWL-RL."""

from __future__ import annotations

import operator

from ontofact_nav import build_navigation_ontology


# ---------------------------------------------------------------------------
# Schema / class hierarchy
# ---------------------------------------------------------------------------

def test_build_ontology_class_and_property_counts(onto):
    assert len(onto.classes) == 20
    assert len(onto.properties) == 23   # 13 space + 10 robot (incl. risk_certified)


def test_expected_classes_present(onto):
    for name in ("Thing", "Space", "IndoorSpace", "Corridor", "Robot", "Ramp"):
        assert name in onto.classes


def test_subclass_is_transitive_and_reflexive(onto):
    corridor = onto.classes["Corridor"]
    space = onto.classes["Space"]
    assert corridor.is_subclass_of(space) is True
    assert corridor.is_subclass_of(corridor) is True   # reflexive
    assert space.is_subclass_of(corridor) is False     # not symmetric


def test_ancestors_are_root_first(onto):
    assert onto.classes["Corridor"].ancestors() == [
        "Thing", "PhysicalEntity", "Space", "IndoorSpace", "Corridor",
    ]


# ---------------------------------------------------------------------------
# Individuals
# ---------------------------------------------------------------------------

def test_create_registers_individual(onto):
    ind = onto.create("c1", "Corridor", width=1.5)
    assert onto.individual("c1") is ind
    assert ind.get("width") == 1.5
    assert ind.is_instance_of(onto.classes["Space"])


def test_get_returns_default_for_missing_property(onto):
    ind = onto.create("c1", "Corridor")
    assert ind.get("nonexistent", 42) == 42


def test_set_mutates_only_target_individual(onto):
    a = onto.create("a", "Corridor", width=1.0)
    b = onto.create("b", "Corridor", width=1.0)
    a.set("width", 9.0)
    assert a.get("width") == 9.0
    assert b.get("width") == 1.0


def test_clone_is_independent_deep_copy(onto):
    original = onto.create("c1", "Corridor", width=1.0)
    clone = original.clone()
    clone.set("width", 5.0)
    assert original.get("width") == 1.0   # original untouched
    assert clone.name == "c1_cf"          # default suffix
    assert clone.ont_class is original.ont_class


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def test_query_by_class_uses_subclass_chain(onto):
    onto.create("c1", "Corridor", width=2.0)
    onto.create("r1", "Room", width=4.0)
    # "Space" should match both subtypes
    assert len(onto.query("Space")) == 2
    assert len(onto.query("Corridor")) == 1


def test_query_with_property_filter(onto):
    onto.create("open1", "Corridor", is_accessible=True)
    onto.create("shut1", "Corridor", is_accessible=False)
    results = onto.query("Corridor", is_accessible=True)
    assert [i.name for i in results] == ["open1"]


def test_sparql_comparator_range_query(onto):
    onto.create("narrow", "Corridor", width=0.9)
    onto.create("wide", "Corridor", width=2.5)
    narrow = onto.sparql("Corridor", "width", operator.lt, 1.2)
    assert [i.name for i in narrow] == ["narrow"]


# ---------------------------------------------------------------------------
# RDFLib SPARQL + OWL-RL reasoning
# ---------------------------------------------------------------------------

def test_sparql_select_returns_rows(onto):
    onto.create("c1", "Corridor", width=2.0)
    rows = list(onto.sparql_select("SELECT ?ind WHERE { ?ind a nav:Corridor }"))
    assert any(str(r.ind).endswith("c1") for r in rows)


def test_apply_reasoning_enables_subclass_queries(onto):
    onto.create("c1", "Corridor", width=2.0)
    q = "SELECT ?ind WHERE { ?ind a nav:Space }"

    before = list(onto.sparql_select(q))
    assert before == []   # no subclass inference without reasoning

    added = onto.apply_reasoning("rdfs")
    assert added > 0

    after = list(onto.sparql_select(q))
    assert any(str(r.ind).endswith("c1") for r in after)


def test_build_navigation_ontology_is_independent_each_call():
    a = build_navigation_ontology()
    b = build_navigation_ontology()
    a.create("only_in_a", "Corridor")
    assert a.individual("only_in_a") is not None
    assert b.individual("only_in_a") is None
