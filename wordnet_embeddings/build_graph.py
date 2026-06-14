"""Extract WordNet's synset relation graph as PyKEEN-compatible triples.

See CUSTOM_EMBEDDINGS_RESEARCH.md (green-ai repo), Part 1, for the design
rationale: synsets are nodes, typed relations (hypernym, hyponym, meronym,
holonym, similar-to, etc.) are edges — this is "Option A" (decided).

TODO:
- Decide the final set of relation types to include (open item in the
  research doc's "Open questions" list) — RELATION_TYPES below is a
  starting point, not final.
- Iterate `nltk.corpus.wordnet.all_synsets()`, emit one row per
  (head_synset, relation, tail_synset) triple for each relation type.
- Write triples to a TSV file (entity1, relation, entity2) suitable for
  `pykeen.triples.TriplesFactory.from_path`.
- Also write a lemma -> [synset_id, ...] mapping, needed later to derive
  lemma-level vectors from trained synset vectors (Part 1, "From synset
  vectors to chunk vectors").
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path("data/triples.tsv")
DEFAULT_LEMMA_MAP_OUTPUT = Path("data/lemma_synsets.tsv")

# TODO: finalise this list (research doc open item).
RELATION_TYPES = [
    "hypernym",
    "hyponym",
    "part_meronym",
    "part_holonym",
    "also_see",
    "similar_to",
]


def build_triples(output_path: Path = DEFAULT_OUTPUT) -> None:
    """Write WordNet synset relation triples to `output_path` as TSV."""
    raise NotImplementedError("build_graph.build_triples is not implemented yet")


def build_lemma_synset_map(output_path: Path = DEFAULT_LEMMA_MAP_OUTPUT) -> None:
    """Write a lemma -> [synset_id, ...] mapping to `output_path` as TSV."""
    raise NotImplementedError(
        "build_graph.build_lemma_synset_map is not implemented yet"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lemma-map-output", type=Path, default=DEFAULT_LEMMA_MAP_OUTPUT)
    args = parser.parse_args()
    build_triples(args.output)
    build_lemma_synset_map(args.lemma_map_output)


if __name__ == "__main__":
    main()
