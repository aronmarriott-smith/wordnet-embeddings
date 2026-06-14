"""Train 128-dim TransE embeddings on the WordNet triples via PyKEEN.

See CUSTOM_EMBEDDINGS_RESEARCH.md, Parts 1-3:
- Model: TransE (or another PyKEEN KGE model) — typed knowledge-graph
  embeddings (decision: "Option A").
- Dimension: 128 (decision, Part 2) — our own embedding space, no
  compatibility target with any other model.
- Hardware: any CPU machine; no GPU required for Level 0 (Part 3).

TODO:
- Load triples produced by `build_graph.py` via
  `pykeen.triples.TriplesFactory.from_path`.
- Run `pykeen.pipeline.pipeline(...)` with model="TransE",
  embedding_dim=EMBED_DIM.
- Persist the trained model/embeddings for `export.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EMBED_DIM = 128
DEFAULT_TRIPLES = Path("data/triples.tsv")
DEFAULT_OUTPUT = Path("data/model")


def train(triples_path: Path = DEFAULT_TRIPLES, output_path: Path = DEFAULT_OUTPUT) -> None:
    """Train a 128-dim TransE model on `triples_path`, saving to `output_path`."""
    raise NotImplementedError("train.train is not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triples", type=Path, default=DEFAULT_TRIPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(args.triples, args.output)


if __name__ == "__main__":
    main()
