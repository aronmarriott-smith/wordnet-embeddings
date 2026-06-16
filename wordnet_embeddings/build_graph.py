"""Extract WordNet's synset relation graph as PyKEEN-compatible triples.

Each synset becomes a graph entity (identified by NLTK's dotted name,
e.g. "car.n.01"). Each typed relation between synsets becomes an edge.
Output format: two-column-header-free TSV (head \\t relation \\t tail)
readable by PyKEEN's TriplesFactory.from_path().

Usage::

    python -m wordnet_embeddings.build_graph
    python -m wordnet_embeddings.build_graph --output data/triples.tsv

Also writes a lemma -> synset mapping used later by export.py to derive
lemma-level embeddings by averaging per-synset vectors.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

DEFAULT_OUTPUT = Path("data/triples.tsv")
DEFAULT_LEMMA_MAP_OUTPUT = Path("data/lemma_synsets.tsv")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# All typed WordNet relations, keyed by the name that becomes the relation
# column in the triples file. Including both directions (e.g. hypernym and
# hyponym) is standard practice in KGE: they get separate relation embeddings
# and carry different semantic signal.
RELATIONS: dict[str, Callable] = {
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


def ensure_wordnet() -> None:
    """Download WordNet corpus if not already present."""
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        next(wn.all_synsets())
    except LookupError:
        log.info("WordNet corpus not found; downloading (one-time setup)...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def build_triples(
    output_path: Path = DEFAULT_OUTPUT,
    relations: dict[str, Callable] | None = None,
    log_every: int = 20_000,
) -> int:
    """Write synset relation triples to output_path as a tab-separated file.

    Args:
        output_path: destination path (.tsv).
        relations: dict of {relation_name: synset_method}. Defaults to
            the module-level RELATIONS (all 22 WordNet relation types).
        log_every: log progress every N synsets.

    Returns:
        Total number of triples written.
    """
    from nltk.corpus import wordnet as wn

    if relations is None:
        relations = RELATIONS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building triples -> %s", output_path)

    total_synsets = 0
    total_triples = 0

    with output_path.open("w", encoding="utf-8") as f:
        for synset in wn.all_synsets():
            head = synset.name()
            for rel_name, rel_fn in relations.items():
                for tail_synset in rel_fn(synset):
                    f.write(f"{head}\t{rel_name}\t{tail_synset.name()}\n")
                    total_triples += 1
            total_synsets += 1
            if total_synsets % log_every == 0:
                log.info("  %d synsets processed, %d triples so far", total_synsets, total_triples)

    log.info(
        "Done: %d synsets, %d triples written to %s",
        total_synsets,
        total_triples,
        output_path,
    )
    return total_triples


def build_lemma_synset_map(
    output_path: Path = DEFAULT_LEMMA_MAP_OUTPUT,
) -> int:
    """Write a lemma -> synset_id mapping to output_path as TSV.

    One row per (lemma, synset) pair. A polysemous word appears on multiple
    rows (one per sense). Used downstream in export.py to derive per-lemma
    embeddings by averaging the embedding vectors of all synsets that lemma
    belongs to (Part 1 of the research doc, "From synset vectors to chunk
    vectors").

    Args:
        output_path: destination path (.tsv).

    Returns:
        Total number of rows written.
    """
    from nltk.corpus import wordnet as wn

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building lemma-synset map -> %s", output_path)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for synset in wn.all_synsets():
            synset_id = synset.name()
            for lemma in synset.lemmas():
                f.write(f"{lemma.name()}\t{synset_id}\n")
                count += 1

    log.info("Done: %d lemma-synset rows written to %s", count, output_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Triples TSV output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--lemma-map-output", type=Path, default=DEFAULT_LEMMA_MAP_OUTPUT,
        help=f"Lemma-synset map output path (default: {DEFAULT_LEMMA_MAP_OUTPUT})",
    )
    parser.add_argument(
        "--skip-lemma-map", action="store_true",
        help="Skip writing the lemma-synset map",
    )
    args = parser.parse_args()

    ensure_wordnet()
    build_triples(args.output)
    if not args.skip_lemma_map:
        build_lemma_synset_map(args.lemma_map_output)


if __name__ == "__main__":
    main()
