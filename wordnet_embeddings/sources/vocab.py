"""Vocabulary-only sources: flat word lists with no entity relations.

These don't fit GraphSource (there's nothing to put in EntityRecord.relations)
so they get their own, smaller protocol: just a list of words to append to
the lemma map. NOTE: under the current pipeline this has no effect on the
final embedding table — export.py only keeps a lemma row if its entity_id
got a trained embedding (i.e. appeared in at least one triple), so words
contributed here fall back to the 'undefined' OOV vector at export time,
exactly like any other out-of-vocabulary word. They're wired in now as
plumbing for later (e.g. once these words are tied into relations, or once
retrofitting/distributional signal is added — see
CUSTOM_EMBEDDINGS_RESEARCH.md Part 6, "Level 0.5").
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from wordnet_embeddings.sources._nltk_utils import ensure_nltk_corpus


class VocabSource(Protocol):
    """A flat word list to append to the lemma map (no relations)."""

    def ensure_available(self) -> None: ...

    def iter_words(self) -> Iterator[str]: ...


class WordsCorpusSource:
    """NLTK's Words Corpus (~236k English words), no relations."""

    def ensure_available(self) -> None:
        ensure_nltk_corpus("words")

    def iter_words(self) -> Iterator[str]:
        from nltk.corpus import words

        yield from words.words()


class StopwordsCorpusSource:
    """NLTK's Stopwords Corpus, English only (project scope), no relations."""

    def ensure_available(self) -> None:
        ensure_nltk_corpus("stopwords")

    def iter_words(self) -> Iterator[str]:
        from nltk.corpus import stopwords

        yield from stopwords.words("english")
