"""Tests for OpenEnglishWordNetSource: the GraphSource adapter over NLTK's
`english_wordnet` (Open English WordNet) download package.

Mirrors test_wordnet_source.py — same Synset API, different underlying data.
Patches the reader's all_synsets() to a fixed, tiny set so these tests don't
pay the cost of a full corpus walk.
"""

from __future__ import annotations

import pytest

from wordnet_embeddings.sources.open_english_wordnet import OpenEnglishWordNetSource


def test_iter_entities_yields_lemmas_and_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    source = OpenEnglishWordNetSource()
    reader = source._reader()
    dog = reader.synset("dog.n.01")
    monkeypatch.setattr(reader, "all_synsets", lambda: iter([dog]))
    monkeypatch.setattr(source, "_reader", lambda: reader)

    [entity] = list(source.iter_entities())

    assert entity.id == "dog.n.01"
    assert "dog" in entity.lemmas
    assert any(rel_name == "hypernym" for rel_name, _ in entity.relations)


def test_iter_entities_respects_custom_relation_set(monkeypatch: pytest.MonkeyPatch) -> None:
    source = OpenEnglishWordNetSource(relations={"hypernym": lambda s: s.hypernyms()})
    reader = source._reader()
    dog = reader.synset("dog.n.01")
    monkeypatch.setattr(reader, "all_synsets", lambda: iter([dog]))
    monkeypatch.setattr(source, "_reader", lambda: reader)

    [entity] = list(source.iter_entities())

    assert {name for name, _ in entity.relations} == {"hypernym"}


def test_ensure_available_delegates_to_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "wordnet_embeddings.sources.open_english_wordnet.ensure_nltk_corpus",
        lambda package_id: calls.append(package_id),
    )

    OpenEnglishWordNetSource().ensure_available()

    assert calls == ["english_wordnet"]
