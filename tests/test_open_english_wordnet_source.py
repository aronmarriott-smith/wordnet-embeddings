"""Tests for OpenEnglishWordNetSource: the GraphSource adapter over the `wn`
package's oewn:2025+ lexicon.

Uses fake Synset/Wordnet objects (not the real ~120k-synset lexicon) matching
the small surface of the `wn` package this source actually depends on:
`.id`, `.lemmas()`, `.relations()`.
"""

from __future__ import annotations

import pytest

from wordnet_embeddings.sources.open_english_wordnet import (
    LEXICON_SPEC,
    OpenEnglishWordNetSource,
)


class FakeSynset:
    def __init__(self, id: str, lemmas: list[str], relations: dict[str, list["FakeSynset"]] | None = None) -> None:
        self.id = id
        self._lemmas = lemmas
        self._relations = relations or {}

    def lemmas(self) -> list[str]:
        return self._lemmas

    def relations(self) -> dict[str, list["FakeSynset"]]:
        return self._relations


class FakeWordnet:
    def __init__(self, synsets: list[FakeSynset]) -> None:
        self._synsets = synsets

    def synsets(self) -> list[FakeSynset]:
        return self._synsets


def test_iter_entities_yields_lemmas_and_relations(monkeypatch: pytest.MonkeyPatch) -> None:
    canine = FakeSynset("oewn-canine", ["canine"])
    dog = FakeSynset("oewn-dog", ["dog", "domestic dog"], {"hypernym": [canine]})
    monkeypatch.setattr("wn.Wordnet", lambda spec: FakeWordnet([dog, canine]))

    entities = list(OpenEnglishWordNetSource().iter_entities())

    assert entities[0].id == "oewn-dog"
    assert entities[0].lemmas == ("dog", "domestic_dog")
    assert entities[0].relations == (("hypernym", "oewn-canine"),)


def test_iter_entities_normalises_multiword_lemmas_to_underscores(monkeypatch: pytest.MonkeyPatch) -> None:
    synset = FakeSynset("oewn-x", ["domestic dog", "Canis familiaris"])
    monkeypatch.setattr("wn.Wordnet", lambda spec: FakeWordnet([synset]))

    [entity] = list(OpenEnglishWordNetSource().iter_entities())

    assert entity.lemmas == ("domestic_dog", "Canis_familiaris")


def test_iter_entities_respects_custom_relation_set(monkeypatch: pytest.MonkeyPatch) -> None:
    canine = FakeSynset("oewn-canine", ["canine"])
    dog = FakeSynset("oewn-dog", ["dog"], {"hypernym": [canine], "similar": [canine]})
    monkeypatch.setattr("wn.Wordnet", lambda spec: FakeWordnet([dog]))

    source = OpenEnglishWordNetSource(relations={"hypernym": "hypernym"})
    [entity] = list(source.iter_entities())

    assert entity.relations == (("hypernym", "oewn-canine"),)


def test_iter_entities_yields_one_record_per_synset(monkeypatch: pytest.MonkeyPatch) -> None:
    dog = FakeSynset("oewn-dog", ["dog"])
    cat = FakeSynset("oewn-cat", ["cat"])
    monkeypatch.setattr("wn.Wordnet", lambda spec: FakeWordnet([dog, cat]))

    entities = list(OpenEnglishWordNetSource().iter_entities())

    assert [e.id for e in entities] == ["oewn-dog", "oewn-cat"]


def test_ensure_available_is_a_noop_when_lexicon_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wn.lexicons", lambda lexicon: [object()])
    monkeypatch.setattr("wn.download", lambda *_a, **_k: pytest.fail("should not download"))

    OpenEnglishWordNetSource().ensure_available()


def test_ensure_available_downloads_when_lexicon_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("wn.lexicons", lambda lexicon: [])
    monkeypatch.setattr("wn.download", lambda spec: calls.append(spec))

    OpenEnglishWordNetSource().ensure_available()

    assert calls == [LEXICON_SPEC]
