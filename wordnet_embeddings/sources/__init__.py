"""Registry of GraphSource and VocabSource implementations, selectable by name
on the build_graph CLI.

Deliberately NOT implemented here (see CUSTOM_EMBEDDINGS_RESEARCH.md Part 6):
- brown, gutenberg, webtext, nps_chat — plain/tagged text corpora with no
  entity relations. These fit a future "Level 0.5" distributional/
  co-occurrence embedding pipeline, not this triples-graph extractor.
- cmudict (CMU Pronouncing Dictionary) — word -> phoneme mapping, not
  entity-to-entity relations. Revisit only if a phonetic-similarity relation
  type is wanted later.
"""

from __future__ import annotations

from wordnet_embeddings.sources.base import EntityRecord, GraphSource
from wordnet_embeddings.sources.open_english_wordnet import OpenEnglishWordNetSource
from wordnet_embeddings.sources.vocab import (
    StopwordsCorpusSource,
    VocabSource,
    WordsCorpusSource,
)
from wordnet_embeddings.sources.wordnet import WordNetSource

SOURCES: dict[str, type[GraphSource]] = {
    "wordnet": WordNetSource,
    "oewn": OpenEnglishWordNetSource,
}

VOCAB_SOURCES: dict[str, type[VocabSource]] = {
    "words": WordsCorpusSource,
    "stopwords": StopwordsCorpusSource,
}


def get_source(name: str) -> GraphSource:
    try:
        return SOURCES[name]()
    except KeyError:
        raise ValueError(f"Unknown source {name!r}; available: {sorted(SOURCES)}") from None


def get_vocab_source(name: str) -> VocabSource:
    try:
        return VOCAB_SOURCES[name]()
    except KeyError:
        raise ValueError(f"Unknown vocab source {name!r}; available: {sorted(VOCAB_SOURCES)}") from None


__all__ = [
    "SOURCES",
    "VOCAB_SOURCES",
    "EntityRecord",
    "GraphSource",
    "VocabSource",
    "get_source",
    "get_vocab_source",
]
