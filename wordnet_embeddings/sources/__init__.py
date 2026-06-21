"""Registry of GraphSource implementations, selectable by name on the build_graph CLI."""

from __future__ import annotations

from wordnet_embeddings.sources.base import EntityRecord, GraphSource
from wordnet_embeddings.sources.open_english_wordnet import OpenEnglishWordNetSource
from wordnet_embeddings.sources.wordnet import WordNetSource

SOURCES: dict[str, type[GraphSource]] = {
    "wordnet": WordNetSource,
    "oewn": OpenEnglishWordNetSource,
}


def get_source(name: str) -> GraphSource:
    try:
        return SOURCES[name]()
    except KeyError:
        raise ValueError(f"Unknown source {name!r}; available: {sorted(SOURCES)}") from None


__all__ = ["EntityRecord", "GraphSource", "SOURCES", "get_source"]
