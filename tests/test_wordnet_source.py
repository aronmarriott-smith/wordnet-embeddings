"""Tests for WordNetSource: the GraphSource adapter over NLTK's wordnet corpus.

Uses real Synset objects (NLTK's wordnet corpus is a project dependency) but
patches all_synsets() to a fixed, tiny set so these tests don't pay the cost
of (or depend on the exact content of) a full ~117k-synset corpus walk.
"""

from __future__ import annotations

import pytest

from wordnet_embeddings.sources.wordnet import WordNetSource


def test_iter_entities_yields_lemmas_and_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    from nltk.corpus import wordnet as wn

    dog = wn.synset("dog.n.01")
    monkeypatch.setattr(wn, "all_synsets", lambda: iter([dog]))

    [entity] = list(WordNetSource().iter_entities())

    assert entity.id == "dog.n.01"
    assert "dog" in entity.lemmas
    assert ("hypernym", "canine.n.02") in entity.relations


def test_iter_entities_respects_custom_relation_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from nltk.corpus import wordnet as wn

    dog = wn.synset("dog.n.01")
    monkeypatch.setattr(wn, "all_synsets", lambda: iter([dog]))

    source = WordNetSource(relations={"hypernym": lambda s: s.hypernyms()})
    [entity] = list(source.iter_entities())

    assert {name for name, _ in entity.relations} == {"hypernym"}


def test_iter_entities_yields_one_record_per_synset(monkeypatch: pytest.MonkeyPatch) -> None:
    from nltk.corpus import wordnet as wn

    dog = wn.synset("dog.n.01")
    cat = wn.synset("cat.n.01")
    monkeypatch.setattr(wn, "all_synsets", lambda: iter([dog, cat]))

    entities = list(WordNetSource().iter_entities())

    assert [e.id for e in entities] == ["dog.n.01", "cat.n.01"]


def test_ensure_available_is_a_noop_when_corpus_present() -> None:
    # The dev/CI environment already has the wordnet corpus downloaded;
    # this should return without attempting any download.
    WordNetSource().ensure_available()


def test_ensure_available_delegates_to_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    # The download/error-handling behavior itself is covered by
    # test_nltk_utils.py; this just confirms WordNetSource asks for the
    # right NLTK package id.
    calls: list[str] = []
    monkeypatch.setattr(
        "wordnet_embeddings.sources.wordnet.ensure_nltk_corpus",
        lambda package_id: calls.append(package_id),
    )

    WordNetSource().ensure_available()

    assert calls == ["wordnet"]
