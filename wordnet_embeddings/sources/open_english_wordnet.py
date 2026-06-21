"""GraphSource for the Open English WordNet (globalwordnet/english-wordnet),
via the `wn` package (https://pypi.org/project/wn/) rather than NLTK.

NLTK only bundles older snapshots of OEWN (the `english_wordnet` download
package). `wn` pulls releases directly from en-word.net by version tag,
including "2025+" — chosen for its improved proper-noun coverage over both
Princeton WordNet and NLTK's older OEWN snapshot. See
CUSTOM_EMBEDDINGS_RESEARCH.md Part 6.

Relation-type names differ from Princeton WordNet's (WN-LMF vocabulary, e.g.
"mero_part" instead of "part_meronym"): RELATIONS maps our existing relation
names (shared with wordnet.py, for consistency across every source's
triples.tsv) onto wn's native relation-type strings. A few relation types
exist here with no Princeton WordNet equivalent (e.g. "is_caused_by",
"exemplifies") and keep their native names; conversely "verb_group" and the
"usage_domain" pair don't exist in this data model at all, so they're absent
here (confirmed by a full scan of all 120k+ synsets, not an oversight).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from wordnet_embeddings.sources.base import EntityRecord

log = logging.getLogger(__name__)

LEXICON_SPEC = "oewn:2025+"

# our relation name -> wn's native relation-type string (see module docstring)
RELATIONS: dict[str, str] = {
    "hypernym":          "hypernym",
    "instance_hypernym": "instance_hypernym",
    "hyponym":           "hyponym",
    "instance_hyponym":  "instance_hyponym",
    "part_meronym":      "mero_part",
    "part_holonym":      "holo_part",
    "member_meronym":    "mero_member",
    "member_holonym":    "holo_member",
    "substance_meronym": "mero_substance",
    "substance_holonym": "holo_substance",
    "attribute":         "attribute",
    "also_see":          "also",
    "similar_to":        "similar",
    "entails":           "entails",
    "is_entailed_by":    "is_entailed_by",
    "causes":            "causes",
    "is_caused_by":      "is_caused_by",
    "topic_domain":      "domain_topic",
    "in_topic_domain":   "has_domain_topic",
    "region_domain":     "domain_region",
    "in_region_domain":  "has_domain_region",
    "exemplifies":       "exemplifies",
    "is_exemplified_by": "is_exemplified_by",
}


class OpenEnglishWordNetSource:
    """Open English WordNet 2025+ synsets and relations, via the `wn` package."""

    def __init__(self, relations: dict[str, str] | None = None) -> None:
        self.relations = RELATIONS if relations is None else relations

    def ensure_available(self) -> None:
        import wn

        if not wn.lexicons(lexicon=LEXICON_SPEC):
            log.info("%s not found; downloading (one-time setup)...", LEXICON_SPEC)
            wn.download(LEXICON_SPEC)

    def iter_entities(self) -> Iterator[EntityRecord]:
        import wn

        wordnet = wn.Wordnet(LEXICON_SPEC)
        for synset in wordnet.synsets():
            # One relations() call per synset (a dict keyed by wn's native
            # relation-type name) rather than get_related() once per entry in
            # self.relations — ~18x fewer queries against wn's sqlite backend,
            # which matters at ~120k synsets.
            native_relations = synset.relations()
            relations = tuple(
                (our_name, tail.id)
                for our_name, wn_name in self.relations.items()
                for tail in native_relations.get(wn_name, [])
            )
            # wn's lemmas() are space-joined ("domestic dog"); the project's
            # vocab convention (matching Princeton WordNet/NLTK) is
            # underscore-joined ("domestic_dog") — normalise so multi-word
            # lookups in the C engine's tokeniser stay consistent across sources.
            lemmas = tuple(lemma.replace(" ", "_") for lemma in synset.lemmas())
            yield EntityRecord(id=synset.id, lemmas=lemmas, relations=relations)
