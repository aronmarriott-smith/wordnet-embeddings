"""Export trained synset embeddings to a device-ready binary table.

Reads the outputs of train.py (entity_embeddings.npy, entity_to_id.json)
and build_graph.py (lemma_synsets.tsv), and produces:

  data/vocab.txt       — one lemma per line; line N = row N in the table
  data/embeddings.bin  — binary header + int8[vocab_size × EMBED_DIM]

Binary format (mirrored in engine/include/embed.h):
  bytes  0-3:  magic b'WNEB'
  bytes  4-7:  uint32 vocab_size
  bytes  8-11: uint32 embed_dim
  bytes 12-15: float32 scale (dequantise: float_val = int8_val * scale)
  bytes 16+:   int8[vocab_size * embed_dim], row-major
               row 0 is always the 'undefined' OOV entry

Usage::

    python -m wordnet_embeddings.export
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

MAGIC = b"WNEB"
UNDEFINED_LEMMA = "undefined"

DEFAULT_MODEL = Path("data/model")
DEFAULT_LEMMA_MAP = Path("data/lemma_synsets.tsv")
DEFAULT_VOCAB_OUT = Path("data/vocab.txt")
DEFAULT_EMBED_OUT = Path("data/embeddings.bin")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def export(
    model_path: Path = DEFAULT_MODEL,
    lemma_map_path: Path = DEFAULT_LEMMA_MAP,
    vocab_out: Path = DEFAULT_VOCAB_OUT,
    embed_out: Path = DEFAULT_EMBED_OUT,
) -> None:
    # --- Load trained synset embeddings ---
    embeddings = np.load(model_path / "entity_embeddings.npy")  # (num_entities, EMBED_DIM)
    with (model_path / "entity_to_id.json").open() as f:
        entity_to_id: dict[str, int] = json.load(f)
    log.info("Loaded %d synset embeddings (%d-dim)", len(entity_to_id), embeddings.shape[1])

    # --- Build lemma -> [synset row indices] map from build_graph output ---
    lemma_to_rows: dict[str, list[int]] = defaultdict(list)
    skipped = 0
    with lemma_map_path.open(encoding="utf-8") as f:
        for line in f:
            lemma, synset_id = line.rstrip("\n").split("\t")
            if synset_id in entity_to_id:
                lemma_to_rows[lemma].append(entity_to_id[synset_id])
            else:
                skipped += 1
    if skipped:
        log.warning("%d lemma-synset pairs skipped (synset absent from training split)", skipped)
    log.info("Lemma map: %d unique lemmas", len(lemma_to_rows))

    # --- Derive lemma-level vectors by averaging their synset vectors ---
    # A polysemous word (e.g. "bank") gets the mean of all its sense vectors.
    lemmas = sorted(lemma_to_rows.keys())
    lemma_vecs = np.stack([
        embeddings[lemma_to_rows[l]].mean(axis=0) for l in lemmas
    ])  # (num_lemmas, EMBED_DIM)

    # --- Prepend the 'undefined' OOV entry at row 0 ---
    # Corpus mean: non-zero and semantically neutral; avoids zero-vector
    # cosine similarity issues. Row 0 is the C engine's default miss target.
    undefined_vec = embeddings.mean(axis=0, keepdims=True)
    all_vecs = np.vstack([undefined_vec, lemma_vecs])  # (vocab_size, EMBED_DIM)
    vocab = [UNDEFINED_LEMMA] + lemmas

    # --- Quantise to int8 ---
    # Global scale: max abs value -> 127. Simple, sufficient for MVP;
    # per-vector scale would be slightly more accurate if quality is lacking.
    scale = float(np.abs(all_vecs).max() / 127.0)
    quantised = np.clip(np.round(all_vecs / scale), -128, 127).astype(np.int8)
    vocab_size, embed_dim = quantised.shape
    log.info(
        "Quantised: %d vocab × %d dims | scale=%.6f | table size=%.1f KB",
        vocab_size, embed_dim, scale, vocab_size * embed_dim / 1024,
    )

    # --- Write vocab.txt (line N = row N) ---
    vocab_out.parent.mkdir(parents=True, exist_ok=True)
    vocab_out.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    log.info("Wrote %s", vocab_out)

    # --- Write embeddings.bin ---
    # Header: magic(4) + vocab_size(4) + embed_dim(4) + scale(4) = 16 bytes
    embed_out.parent.mkdir(parents=True, exist_ok=True)
    with embed_out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIf", vocab_size, embed_dim, scale))
        f.write(quantised.tobytes())  # row-major int8 array
    log.info("Wrote %s (%.1f KB)", embed_out, embed_out.stat().st_size / 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lemma-map", type=Path, default=DEFAULT_LEMMA_MAP)
    parser.add_argument("--vocab-out", type=Path, default=DEFAULT_VOCAB_OUT)
    parser.add_argument("--embed-out", type=Path, default=DEFAULT_EMBED_OUT)
    args = parser.parse_args()
    export(args.model, args.lemma_map, args.vocab_out, args.embed_out)


if __name__ == "__main__":
    main()
