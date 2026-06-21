"""GraphSource implementation backed by NLTK's Princeton WordNet corpus reader.

English only, per the project's scope decision in CUSTOM_EMBEDDINGS_RESEARCH.md.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator

from nltk.corpus.reader.wordnet import Synset

from wordnet_embeddings.sources._nltk_utils import ensure_nltk_corpus
from wordnet_embeddings.sources.base import EntityRecord

log = logging.getLogger(__name__)

# All typed WordNet relations, keyed by the name that becomes the relation
# column in the triples file. Including both directions (e.g. hypernym and
# hyponym) is standard practice in KGE: they get separate relation embeddings
# and carry different semantic signal.
RELATIONS: dict[str, Callable[[Synset], list[Synset]]] = {
    "hypernym":          lambda s: s.hypernyms(),
    "instance_hypernym": lambda s: s.instance_hypernyms(),
    "hyponym":           lambda s: s.hyponyms(),
    "instance_hyponym":  lambda s: s.instance_hyponyms(),
    "part_meronym":      lambda s: s.part_meronyms(),
    "part_holonym":      lambda s: s.part_holonyms(),
    "member_meronym":    lambda s: s.member_meronyms(),
    "member_holonym":    lambda s: s.member_holonyms(),
    "substance_meronym": lambda s: s.substance_meronyms(),
    "substance_holonym": lambda s: s.substance_holonyms(),
    "attribute":         lambda s: s.attributes(),
    "also_see":          lambda s: s.also_sees(),
    "similar_to":        lambda s: s.similar_tos(),
    "entails":           lambda s: s.entailments(),
    "causes":            lambda s: s.causes(),
    "verb_group":        lambda s: s.verb_groups(),
    "in_topic_domain":   lambda s: s.in_topic_domains(),
    "in_region_domain":  lambda s: s.in_region_domains(),
    "in_usage_domain":   lambda s: s.in_usage_domains(),
    "topic_domain":      lambda s: s.topic_domains(),
    "region_domain":     lambda s: s.region_domains(),
    "usage_domain":      lambda s: s.usage_domains(),
}


def entities_from_synsets(
    synsets: Iterable[Synset],
    relations: dict[str, Callable[[Synset], list[Synset]]],
) -> Iterator[EntityRecord]:
    """Convert an iterable of NLTK Synsets into plain EntityRecords.

    Shared by every WordNet-shaped source (Princeton WordNet, Open English
    WordNet, ...) since they all expose the same Synset API — only how the
    synsets are obtained differs between sources.
    """
    for synset in synsets:
        synset_relations = tuple(
            (rel_name, tail.name())
            for rel_name, rel_fn in relations.items()
            for tail in rel_fn(synset)
        )
        lemmas = tuple(lemma.name() for lemma in synset.lemmas())
        yield EntityRecord(id=synset.name(), lemmas=lemmas, relations=synset_relations)


class WordNetSource:
    """English WordNet 3.0 synsets and relations, via nltk.corpus.wordnet."""

    def __init__(self, relations: dict[str, Callable[[Synset], list[Synset]]] | None = None) -> None:
        self.relations = RELATIONS if relations is None else relations

    def ensure_available(self) -> None:
        ensure_nltk_corpus("wordnet")

    def iter_entities(self) -> Iterator[EntityRecord]:
        from nltk.corpus import wordnet as wn

        yield from entities_from_synsets(wn.all_synsets(), self.relations)
