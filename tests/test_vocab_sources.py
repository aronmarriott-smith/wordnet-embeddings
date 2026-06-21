"""Tests for VocabSource implementations: flat word lists, no relations."""

from __future__ import annotations

import types

import pytest

from wordnet_embeddings.sources import VOCAB_SOURCES, get_vocab_source
from wordnet_embeddings.sources.vocab import StopwordsCorpusSource, WordsCorpusSource

# NLTK's nltk.corpus.{words,stopwords} are LazyCorpusLoader instances that
# only resolve into a real corpus reader on first attribute access — which
# would otherwise require the corpus to already be downloaded just to
# *resolve* a monkeypatch target. Replacing the loader object itself (one
# level up) sidesteps that, so these tests don't depend on local NLTK data.


def test_words_corpus_source_yields_words(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_corpus = types.SimpleNamespace(words=lambda: ["cat", "dog", "zzyzx"])
    monkeypatch.setattr("nltk.corpus.words", fake_corpus)

    assert list(WordsCorpusSource().iter_words()) == ["cat", "dog", "zzyzx"]


def test_stopwords_corpus_source_yields_english_words(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_words(lang: str) -> list[str]:
        calls.append(lang)
        return ["the", "a", "an"]

    fake_corpus = types.SimpleNamespace(words=fake_words)
    monkeypatch.setattr("nltk.corpus.stopwords", fake_corpus)

    assert list(StopwordsCorpusSource().iter_words()) == ["the", "a", "an"]
    assert calls == ["english"]


def test_get_vocab_source_returns_registered_source() -> None:
    assert type(get_vocab_source("words")).__name__ == "WordsCorpusSource"
    assert type(get_vocab_source("stopwords")).__name__ == "StopwordsCorpusSource"


def test_get_vocab_source_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown vocab source"):
        get_vocab_source("brown")


def test_vocab_sources_registry_contents() -> None:
    assert set(VOCAB_SOURCES) == {"words", "stopwords"}
