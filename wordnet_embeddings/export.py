"""Export trained synset embeddings to the device-ready binary table format.

See CUSTOM_EMBEDDINGS_RESEARCH.md, Parts 1, 2 and 4:
- Derive lemma-level vectors by averaging synset vectors per lemma, using
  the lemma -> synsets map from `build_graph.py` (Part 1, "From synset
  vectors to chunk vectors").
- Add one extra `undefined` row for OOV tokens (Part 1, OOV decision).
- Quantise to int8 with a scale factor (Part 2).
- Write a flat, mmap-friendly binary file: header (vocab size, dim=128,
  scale) + one int8[128] record per vocabulary entry, word2vec-`.bin`-style
  (Part 4).

TODO:
- Implement the synset -> lemma averaging step.
- Implement int8 quantisation (choose scale: per-table or per-vector).
- Define and document the binary header layout precisely.
- Also emit a plain-text word2vec-format export and a
  `sentence-transformers`-compatible directory for MTEB (Part 5) — these
  can be separate functions/scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EMBED_DIM = 128
DEFAULT_MODEL = Path("data/model")
DEFAULT_OUTPUT = Path("data/embeddings.bin")


def export_binary(model_path: Path = DEFAULT_MODEL, output_path: Path = DEFAULT_OUTPUT) -> None:
    """Export `model_path` to the quantised binary table at `output_path`."""
    raise NotImplementedError("export.export_binary is not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_binary(args.model, args.output)


if __name__ == "__main__":
    main()
