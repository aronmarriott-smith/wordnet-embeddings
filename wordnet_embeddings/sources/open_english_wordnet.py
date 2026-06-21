"""GraphSource for the Open English WordNet (globalwordnet/english-wordnet).

Ships the same WNDB file format and Synset API as Princeton WordNet, just a
more actively maintained vocabulary — NLTK distributes it as the separate
`english_wordnet` download package. See CUSTOM_EMBEDDINGS_RESEARCH.md Part 6:
"could supplement, or eventually replace, WordNet 3.0 as the base graph".
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from nltk.corpus.reader.wordnet import Synset, WordNetCorpusReader

from wordnet_embeddings.sources._nltk_utils import ensure_nltk_corpus
from wordnet_embeddings.sources.base import EntityRecord
from wordnet_embeddings.sources.wordnet import RELATIONS, entities_from_synsets

log = logging.getLogger(__name__)

NLTK_PACKAGE = "english_wordnet"


class OpenEnglishWordNetSource:
    """Open English WordNet synsets and relations, via NLTK's `english_wordnet` package."""

    def __init__(self, relations: dict[str, Callable[[Synset], list[Synset]]] | None = None) -> None:
        self.relations = RELATIONS if relations is None else relations

    def ensure_available(self) -> None:
        ensure_nltk_corpus(NLTK_PACKAGE)

    def _reader(self) -> WordNetCorpusReader:
        import nltk

        root = nltk.data.find(f"corpora/{NLTK_PACKAGE}")
        return WordNetCorpusReader(root, None)

    def iter_entities(self) -> Iterator[EntityRecord]:
        yield from entities_from_synsets(self._reader().all_synsets(), self.relations)
