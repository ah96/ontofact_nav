"""
ontofact_nav/ontology.py
========================
OWL-inspired ontology engine backed by RDFLib.

Concepts mirrored from OWL (Web Ontology Language):
  OntologyClass      ≈ owl:Class            — node in the is-a hierarchy
  OntologyProperty   ≈ owl:DatatypeProperty — named slot on individuals
  OntologyIndividual ≈ owl:NamedIndividual  — a concrete instance

The Python dict-based API (defclass / defproperty / create / query) is fully
preserved for backward compatibility.  RDFLib provides a parallel RDF graph
that enables real SPARQL queries via ``Ontology.sparql_select()``.

RDF graph notes
---------------
- Namespace: ``http://ontofact.nav/`` (prefix ``nav:``)
- Classes are typed ``owl:Class``; properties ``owl:DatatypeProperty``
- Individuals are typed with their class URI and carry data-property triples
- ``OntologyIndividual.set()`` mutates the Python dict only.  The RDF graph
  captures the initial world state; counterfactual clones are ephemeral Python
  objects and intentionally not synced back to RDF.

Thread-safety: Ontology objects are NOT thread-safe.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef

_NAV = Namespace("http://ontofact.nav/")

# Map Python built-in types to XSD datatypes for RDF Literal tagging.
_PY_TO_XSD: Dict[type, URIRef] = {
    float: XSD.double,
    int:   XSD.integer,
    bool:  XSD.boolean,
    str:   XSD.string,
}


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------

class OntologyClass:
    """
    A class in the ontology hierarchy (analogous to an OWL Class).

    Classes form a single-inheritance tree via the *parent* pointer.
    Multiple-inheritance is intentionally omitted to keep subclass checks O(depth).

    Attributes
    ----------
    name        : str              — unique identifier within the ontology
    parent      : OntologyClass    — immediate superclass (None for root)
    children    : List[OntologyClass] — direct subclasses (populated automatically)
    description : str              — human-readable documentation string
    """

    def __init__(
        self,
        name: str,
        parent: Optional["OntologyClass"] = None,
        description: str = "",
    ) -> None:
        self.name        = name
        self.parent      = parent
        self.children:   List["OntologyClass"] = []
        self.description = description

        # Register this class as a child of its parent so callers can
        # traverse the tree downward (e.g. for printing the schema).
        if parent is not None:
            parent.children.append(self)

    # ------------------------------------------------------------------
    def is_subclass_of(self, other: "OntologyClass") -> bool:
        """
        Return True if *self* is equal to, or a transitive subclass of, *other*.

        Walks the parent chain upward until it either finds *other* or reaches
        the root.  O(depth) — acceptable for shallow hierarchies (<20 levels).

        Examples
        --------
        Corridor.is_subclass_of(Space)    → True
        Space.is_subclass_of(Corridor)    → False
        Corridor.is_subclass_of(Corridor) → True  (reflexive)
        """
        node: Optional[OntologyClass] = self
        while node is not None:
            if node.name == other.name:
                return True
            node = node.parent
        return False

    def ancestors(self) -> List[str]:
        """
        Return class names from the root down to (and including) self.

        Useful for logging and debugging the inference chain.
        Example: Corridor.ancestors() → ["Thing", "PhysicalEntity", "Space",
                                          "IndoorSpace", "Corridor"]
        """
        chain: List[str] = []
        node: Optional[OntologyClass] = self
        while node is not None:
            chain.append(node.name)
            node = node.parent
        return list(reversed(chain))   # root-first order

    def __repr__(self) -> str:
        return f"OntologyClass({self.name!r})"


# ---------------------------------------------------------------------------
# Property descriptor
# ---------------------------------------------------------------------------

class OntologyProperty:
    """
    Describes a property slot that individuals of a given class may carry.

    Analogous to an owl:DatatypeProperty (since all properties here are
    scalar Python values, not references to other individuals).

    The framework does NOT enforce type-checking at assignment time — that
    would add overhead without meaningful benefit in a research prototype.
    Types are stored here for documentation and tooling purposes.
    """

    def __init__(
        self,
        name:        str,
        domain:      str,        # name of the OntologyClass this applies to
        range_type:  type,       # expected Python type (float, bool, str, …)
        description: str  = "",
        functional:  bool = True,  # True = at most one value per individual
    ) -> None:
        self.name        = name
        self.domain      = domain
        self.range_type  = range_type
        self.description = description
        self.functional  = functional

    def __repr__(self) -> str:
        return f"OntologyProperty({self.name!r}, domain={self.domain!r})"


# ---------------------------------------------------------------------------
# Individual (instance)
# ---------------------------------------------------------------------------

class OntologyIndividual:
    """
    A concrete instance of an OntologyClass.

    Properties are stored in a plain Python dict for maximum flexibility —
    callers may set arbitrary keys even if no matching OntologyProperty
    has been declared.  This mirrors OWL's open-world assumption.

    The *clone()* method is central to counterfactual reasoning: it creates
    a deep copy of this individual so the counterfactual engine can mutate
    properties without affecting the live world model.
    """

    def __init__(
        self,
        name:       str,
        ont_class:  OntologyClass,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name        = name
        self.ont_class   = ont_class
        # Use dict() rather than {} literal so *properties* is never shared
        # between instances when the caller passes the same dict object.
        self.properties: Dict[str, Any] = dict(properties or {})

    # ------------------------------------------------------------------
    # Property access
    # ------------------------------------------------------------------

    def get(self, prop: str, default: Any = None) -> Any:
        """Retrieve a property value, returning *default* if absent."""
        return self.properties.get(prop, default)

    def set(self, prop: str, value: Any) -> None:
        """Set a property value in-place (mutates this individual)."""
        self.properties[prop] = value

    # ------------------------------------------------------------------
    # Class membership
    # ------------------------------------------------------------------

    def is_instance_of(self, cls: OntologyClass) -> bool:
        """
        True if this individual's class is equal to or a subclass of *cls*.

        Used by Ontology.query() to filter by class name.
        """
        return self.ont_class.is_subclass_of(cls)

    # ------------------------------------------------------------------
    # Counterfactual copy
    # ------------------------------------------------------------------

    def clone(self, new_name: Optional[str] = None) -> "OntologyIndividual":
        """
        Return a deep copy of this individual with an optional new name.

        Why deep-copy?  The counterfactual engine modifies properties on the
        clone to simulate hypothetical worlds.  A shallow copy would cause
        mutations to bleed back into the original world model.

        Convention: if *new_name* is omitted, the suffix "_cf" is appended
        so clones are distinguishable in log output.
        """
        return OntologyIndividual(
            name      = new_name if new_name is not None else self.name + "_cf",
            ont_class = self.ont_class,
            # copy.deepcopy handles nested dicts/lists within property values
            properties = copy.deepcopy(self.properties),
        )

    def __repr__(self) -> str:
        return f"Individual({self.name!r} : {self.ont_class.name})"


# ---------------------------------------------------------------------------
# Ontology store
# ---------------------------------------------------------------------------

class Ontology:
    """
    Central registry for classes, properties, and individuals.

    Provides a declarative schema-building DSL (defclass/defproperty/create)
    and a simple querying interface (query/sparql).

    Querying uses linear scans over the individuals dict.  For the graph
    sizes typical in robot navigation (tens to hundreds of individuals) this
    is fast enough; a real production system would index by class.
    """

    def __init__(self, name: str) -> None:
        self.name         = name
        self.classes:     Dict[str, OntologyClass]      = {}
        self.properties:  Dict[str, OntologyProperty]   = {}
        self.individuals: Dict[str, OntologyIndividual] = {}

        # RDFLib graph — parallel store for real SPARQL queries.
        self._rdf = Graph()
        self._rdf.bind("nav",  _NAV)
        self._rdf.bind("owl",  OWL)
        self._rdf.bind("rdfs", RDFS)

    # ------------------------------------------------------------------
    # Schema building
    # ------------------------------------------------------------------

    def defclass(
        self,
        name:        str,
        parent_name: Optional[str] = None,
        description: str           = "",
    ) -> OntologyClass:
        """
        Declare a class and register it in the hierarchy.

        *parent_name* must already have been registered via a prior defclass()
        call.  Declaring classes in top-down order is the intended pattern
        (see domain.py for an example).
        """
        parent = self.classes.get(parent_name) if parent_name else None
        cls    = OntologyClass(name=name, parent=parent, description=description)
        self.classes[name] = cls

        cls_uri = _NAV[name]
        self._rdf.add((cls_uri, RDF.type,        OWL.Class))
        self._rdf.add((cls_uri, RDFS.label,      Literal(name)))
        if description:
            self._rdf.add((cls_uri, RDFS.comment, Literal(description)))
        if parent_name:
            self._rdf.add((cls_uri, RDFS.subClassOf, _NAV[parent_name]))

        return cls

    def defproperty(
        self,
        name:        str,
        domain:      str,
        range_type:  type,
        description: str  = "",
    ) -> OntologyProperty:
        """
        Declare a data property and register it.

        Properties are purely descriptive in this implementation — they do not
        enforce constraints at runtime.  They serve as documentation and could
        be used by a validator layer in a more complete system.
        """
        prop = OntologyProperty(
            name=name,
            domain=domain,
            range_type=range_type,
            description=description,
        )
        self.properties[name] = prop

        prop_uri = _NAV[name]
        self._rdf.add((prop_uri, RDF.type,        OWL.DatatypeProperty))
        self._rdf.add((prop_uri, RDFS.domain,     _NAV[domain]))
        xsd_type = _PY_TO_XSD.get(range_type, XSD.string)
        self._rdf.add((prop_uri, RDFS.range,      xsd_type))
        if description:
            self._rdf.add((prop_uri, RDFS.comment, Literal(description)))

        return prop

    # ------------------------------------------------------------------
    # Individual management
    # ------------------------------------------------------------------

    def add_individual(self, individual: OntologyIndividual) -> OntologyIndividual:
        """Register an externally constructed individual."""
        self.individuals[individual.name] = individual
        return individual

    def individual(self, name: str) -> Optional[OntologyIndividual]:
        """Look up a single individual by name (returns None if absent)."""
        return self.individuals.get(name)

    def create(
        self,
        name:       str,
        class_name: str,
        **properties: Any,
    ) -> OntologyIndividual:
        """
        Factory shortcut: look up the class by name, construct an individual,
        register it, and return it.

        Keyword arguments become the individual's property dict.

        Example
        -------
        robot = onto.create("bot_1", "Robot",
                            robot_width=0.6, can_open_doors=True)
        """
        cls = self.classes[class_name]
        ind = OntologyIndividual(name=name, ont_class=cls, properties=properties)
        self.individuals[name] = ind

        ind_uri = _NAV[name]
        self._rdf.add((ind_uri, RDF.type,   _NAV[class_name]))
        self._rdf.add((ind_uri, RDFS.label, Literal(name)))
        for prop_name, value in properties.items():
            xsd_type = _PY_TO_XSD.get(type(value), XSD.string)
            self._rdf.add((ind_uri, _NAV[prop_name], Literal(value, datatype=xsd_type)))

        return ind

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        class_name:   Optional[str] = None,
        **prop_filters: Any,
    ) -> List[OntologyIndividual]:
        """
        Retrieve individuals matching an optional class constraint plus
        zero or more exact-value property filters.

        Analogous to a simple SPARQL SELECT with an rdf:type filter and
        equality constraints.

        Parameters
        ----------
        class_name   : filter to this class and all its subclasses
        **prop_filters : property=value equality checks

        Returns
        -------
        List of matching OntologyIndividual objects (unsorted).
        """
        results: List[OntologyIndividual] = list(self.individuals.values())

        # Class filter — uses the subclass chain so "Space" matches corridors too
        if class_name is not None:
            cls = self.classes.get(class_name)
            if cls:
                results = [i for i in results if i.is_instance_of(cls)]

        # Property equality filters — all must hold (logical AND)
        for prop, expected in prop_filters.items():
            results = [i for i in results if i.get(prop) == expected]

        return results

    def sparql(
        self,
        class_name:  str,
        prop_name:   str,
        comparator:  Callable[[Any, Any], bool],
        value:       Any,
    ) -> List[OntologyIndividual]:
        """
        Range query using an arbitrary comparator function.

        Enables queries like "all Corridors narrower than 1.2 m":

            import operator
            onto.sparql("Corridor", "width", operator.lt, 1.2)

        Parameters
        ----------
        class_name  : class to filter on
        prop_name   : property to compare
        comparator  : binary predicate, e.g. operator.lt / operator.gt
        value       : right-hand side of the comparison
        """
        cls = self.classes.get(class_name)
        if cls is None:
            return []
        return [
            i
            for i in self.individuals.values()
            if i.is_instance_of(cls)
            # Missing property values default to 0 for numeric comparisons
            and comparator(i.get(prop_name, 0), value)
        ]

    def apply_reasoning(self, semantics: str = "rdfs") -> int:
        """
        Run OWL-RL deductive closure expansion on the RDF graph.

        After calling this, ``sparql_select`` queries see inferred triples:
        subclass-chain propagation (so ``?x a nav:Space`` matches Corridors,
        Rooms, etc.), property domain/range inferences, and more.

        Parameters
        ----------
        semantics : ``"rdfs"`` (default) — RDFS entailment rules only.
                    ``"owlrl"``           — full OWL-RL closure (slower).

        Returns
        -------
        int — number of new triples added by inference.

        Example
        -------
        onto.apply_reasoning()
        results = onto.sparql_select(
            "SELECT ?ind WHERE { ?ind a nav:Space }"  # matches ALL space subtypes
        )
        """
        import owlrl
        before = len(self._rdf)
        cls = owlrl.OWLRL_Semantics if semantics == "owlrl" else owlrl.RDFS_Semantics
        owlrl.DeductiveClosure(cls).expand(self._rdf)
        return len(self._rdf) - before

    def sparql_select(self, query: str):
        """
        Execute a real SPARQL SELECT query against the RDFLib graph.

        The ``nav:`` prefix (``http://ontofact.nav/``) is injected
        automatically, so callers can write compact queries:

            results = onto.sparql_select('''
                SELECT ?ind ?width WHERE {
                    ?ind a nav:Corridor ;
                         nav:width ?width .
                    FILTER (?width < 1.2)
                }
            ''')
            for row in results:
                print(row.ind, float(row.width))

        Returns an ``rdflib.query.Result`` (iterable of ``ResultRow`` objects).
        Note: reflects the initial world state only — mutations via
        ``OntologyIndividual.set()`` are not synced back to the RDF graph.
        """
        prefixed = (
            "PREFIX nav:  <http://ontofact.nav/>\n"
            "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>\n"
        ) + query
        return self._rdf.query(prefixed)

    # ------------------------------------------------------------------
    # Debugging utilities
    # ------------------------------------------------------------------

    def print_schema(self) -> None:
        """
        Pretty-print the class hierarchy as an indented tree, then
        show summary counts for classes, properties, and individuals.
        """
        def _print(cls: OntologyClass, depth: int = 0) -> None:
            print("  " * depth + f"├── {cls.name}")
            for child in cls.children:
                _print(child, depth + 1)

        # Only top-level (root) classes have no parent
        roots = [c for c in self.classes.values() if c.parent is None]
        print(f"\nOntology: {self.name}")
        for root in roots:
            _print(root)
        print(
            f"\n{len(self.classes)} classes  |  "
            f"{len(self.properties)} properties  |  "
            f"{len(self.individuals)} individuals\n"
        )
