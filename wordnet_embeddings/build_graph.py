"""Extract a GraphSource's relation triples and lemma map for PyKEEN/export.py.

Each graph entity becomes a row in the lemma map (one row per lemma it has)
and a head in zero or more triples. Output format: header-free TSV
(head \\t relation \\t tail), readable by PyKEEN's TriplesFactory.from_path().

The graph source itself (WordNet today; see wordnet_embeddings/sources/) is
selected via --source, so a new lexical resource can be added by implementing
the GraphSource protocol — no changes needed here.

Usage::

    python -m wordnet_embeddings.build_graph
    python -m wordnet_embeddings.build_graph --source wordnet --output data/triples.tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

from wordnet_embeddings.config import LEMMA_MAP_PATH, TRIPLES_PATH
from wordnet_embeddings.sources import SOURCES, GraphSource, get_source

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class GraphStats(NamedTuple):
    entities: int
    triples: int
    lemma_rows: int


def build_graph(
    source: GraphSource,
    triples_path: Path = TRIPLES_PATH,
    lemma_map_path: Path = LEMMA_MAP_PATH,
    log_every: int = 20_000,
) -> GraphStats:
    """Write source's triples and lemma map in a single pass over its entities.

    Writes go to `<path>.tmp` and are only moved into place (atomic rename)
    once the full pass succeeds, so a failed/interrupted run never leaves a
    corrupt or partial file at the real output path.

    Args:
        source: the GraphSource to extract from.
        triples_path: destination for the (head, relation, tail) TSV.
        lemma_map_path: destination for the (lemma, entity_id) TSV.
        log_every: log progress every N entities.

    Returns:
        GraphStats(entities, triples, lemma_rows) processed/written.
    """
    triples_path.parent.mkdir(parents=True, exist_ok=True)
    lemma_map_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Building graph from %s -> %s, %s", type(source).__name__, triples_path, lemma_map_path)

    triples_tmp = triples_path.with_name(triples_path.name + ".tmp")
    lemma_map_tmp = lemma_map_path.with_name(lemma_map_path.name + ".tmp")

    entities = triples = lemma_rows = 0

    with (
        triples_tmp.open("w", encoding="utf-8") as triples_f,
        lemma_map_tmp.open("w", encoding="utf-8") as lemma_f,
    ):
        for entity in source.iter_entities():
            for rel_name, tail_id in entity.relations:
                triples_f.write(f"{entity.id}\t{rel_name}\t{tail_id}\n")
                triples += 1
            for lemma in entity.lemmas:
                lemma_f.write(f"{lemma}\t{entity.id}\n")
                lemma_rows += 1
            entities += 1
            if entities % log_every == 0:
                log.info("  %d entities processed, %d triples so far", entities, triples)

    triples_tmp.replace(triples_path)
    lemma_map_tmp.replace(lemma_map_path)

    log.info(
        "Done: %d entities, %d triples -> %s, %d lemma rows -> %s",
        entities, triples, triples_path, lemma_rows, lemma_map_path,
    )
    return GraphStats(entities, triples, lemma_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=sorted(SOURCES), default="wordnet",
        help="Graph source to extract from (default: wordnet)",
    )
    parser.add_argument(
        "--output", type=Path, default=TRIPLES_PATH,
        help=f"Triples TSV output path (default: {TRIPLES_PATH})",
    )
    parser.add_argument(
        "--lemma-map-output", type=Path, default=LEMMA_MAP_PATH,
        help=f"Lemma-synset map output path (default: {LEMMA_MAP_PATH})",
    )
    args = parser.parse_args()

    source = get_source(args.source)
    source.ensure_available()
    build_graph(source, args.output, args.lemma_map_output)


if __name__ == "__main__":
    main()
