"""Tests for build_graph.py: the source-agnostic graph extraction pipeline.

Exercises build_graph() against a tiny in-memory GraphSource, so these tests
say nothing about WordNet specifically — see test_wordnet_source.py for that.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from wordnet_embeddings.build_graph import GraphStats, build_graph
from wordnet_embeddings.sources import SOURCES, get_source
from wordnet_embeddings.sources.base import EntityRecord


class FakeSource:
    """A tiny in-memory GraphSource for exercising build_graph() without nltk."""

    def __init__(self, entities: list[EntityRecord]) -> None:
        self._entities = entities

    def ensure_available(self) -> None:
        pass

    def iter_entities(self) -> Iterator[EntityRecord]:
        yield from self._entities


FIXTURE_ENTITIES = [
    EntityRecord(id="cat.n.01", lemmas=("cat", "true_cat"), relations=(("hypernym", "feline.n.01"),)),
    EntityRecord(id="feline.n.01", lemmas=("feline",), relations=()),
]


def test_build_graph_writes_triples_and_lemma_map(tmp_path: Path) -> None:
    triples_path = tmp_path / "triples.tsv"
    lemma_map_path = tmp_path / "lemma_synsets.tsv"

    stats = build_graph(FakeSource(FIXTURE_ENTITIES), triples_path, lemma_map_path)

    assert stats == GraphStats(entities=2, triples=1, lemma_rows=3)
    assert triples_path.read_text(encoding="utf-8") == "cat.n.01\thypernym\tfeline.n.01\n"
    assert lemma_map_path.read_text(encoding="utf-8") == (
        "cat\tcat.n.01\ntrue_cat\tcat.n.01\nfeline\tfeline.n.01\n"
    )


def test_build_graph_does_not_leave_tmp_files(tmp_path: Path) -> None:
    triples_path = tmp_path / "triples.tsv"
    lemma_map_path = tmp_path / "lemma_synsets.tsv"

    build_graph(FakeSource(FIXTURE_ENTITIES), triples_path, lemma_map_path)

    assert not triples_path.with_name("triples.tsv.tmp").exists()
    assert not lemma_map_path.with_name("lemma_synsets.tsv.tmp").exists()


def test_build_graph_creates_missing_parent_directories(tmp_path: Path) -> None:
    triples_path = tmp_path / "nested" / "triples.tsv"
    lemma_map_path = tmp_path / "nested" / "lemma_synsets.tsv"

    build_graph(FakeSource(FIXTURE_ENTITIES), triples_path, lemma_map_path)

    assert triples_path.exists()
    assert lemma_map_path.exists()


def test_build_graph_empty_source_writes_empty_files(tmp_path: Path) -> None:
    triples_path = tmp_path / "triples.tsv"
    lemma_map_path = tmp_path / "lemma_synsets.tsv"

    stats = build_graph(FakeSource([]), triples_path, lemma_map_path)

    assert stats == GraphStats(entities=0, triples=0, lemma_rows=0)
    assert triples_path.read_text(encoding="utf-8") == ""
    assert lemma_map_path.read_text(encoding="utf-8") == ""


def test_get_source_returns_registered_source() -> None:
    assert type(get_source("wordnet")).__name__ == "WordNetSource"


def test_get_source_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        get_source("conceptnet")


def test_sources_registry_contains_wordnet() -> None:
    assert "wordnet" in SOURCES
