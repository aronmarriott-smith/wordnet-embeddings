"""The GraphSource interface that build_graph.py depends on.

build_graph.py only ever talks to this interface, never to a specific
corpus library — that's what lets a new lexical resource (another NLTK
corpus, or eventually a non-NLTK one like ConceptNet or Wiktionary) be
added by writing one new module, with no changes to build_graph.py itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EntityRecord:
    """One graph entity: its id, its surface-form lemmas, and its typed
    relations to other entities (by id).

    Plain strings only, deliberately — no source-specific type (e.g. NLTK's
    Synset) ever crosses this boundary, so build_graph.py and any future
    GraphSource implementation stay decoupled from each other's libraries.
    """

    id: str
    lemmas: tuple[str, ...]
    relations: tuple[tuple[str, str], ...]  # (relation_name, tail_entity_id)


class GraphSource(Protocol):
    """A pluggable source of a typed relational graph plus entity surface forms."""

    def ensure_available(self) -> None:
        """Download/prepare the underlying data, if needed."""
        ...

    def iter_entities(self) -> Iterator[EntityRecord]:
        """Yield every entity in the graph exactly once."""
        ...
