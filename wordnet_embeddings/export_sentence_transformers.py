"""Export the trained lemma embeddings to a sentence-transformers-loadable model.

Reads the quantised WNEB table produced by export.py (data/vocab.txt,
data/embeddings.bin) and rebuilds it as a `StaticEmbedding` module — an
embedding-bag + mean-pooling model with no transformer layers, mirroring
what the C engine (engine/src/embed.c) does at inference time. The result is
loadable as `SentenceTransformer(path)` by anyone, including MTEB.

Tokenisation here is the Python-equivalent of the C engine's embed_text():
lowercase, split into maximal runs of ASCII letters, anything outside the
vocab maps to the 'undefined' row (row 0). It does NOT lemmatise (e.g.
"running" stays "running", doesn't become "run") — see
docs/CUSTOM_EMBEDDINGS_RESEARCH.md Part 1 step 3 for the intended future
pipeline; the C engine doesn't lemmatise yet either, so this stays faithful
to current real-world behaviour rather than the aspirational one.

Usage::

    python -m wordnet_embeddings.export_sentence_transformers
"""

from __future__ import annotations

import argparse
import logging
import struct
from pathlib import Path

import numpy as np

from wordnet_embeddings.config import EMBED_PATH, ST_EXPORT_DIR, VOCAB_PATH, WNEB_MAGIC

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

HEADER_SIZE = 16  # magic(4) + vocab_size(4) + embed_dim(4) + scale(4)


def load_wneb(vocab_path: Path, embed_path: Path) -> tuple[list[str], np.ndarray]:
    """Read the WNEB binary table + vocab file, dequantised to float32."""
    vocab = vocab_path.read_text(encoding="utf-8").splitlines()

    raw = embed_path.read_bytes()
    magic, vocab_size, embed_dim, scale = struct.unpack_from("<4sIIf", raw, 0)
    if magic != WNEB_MAGIC:
        raise ValueError(f"{embed_path}: bad magic {magic!r}, expected {WNEB_MAGIC!r}")
    if vocab_size != len(vocab):
        raise ValueError(
            f"vocab size mismatch: {embed_path} has {vocab_size}, {vocab_path} has {len(vocab)}"
        )

    table = np.frombuffer(raw, dtype=np.int8, count=vocab_size * embed_dim, offset=HEADER_SIZE)
    vectors = table.reshape(vocab_size, embed_dim).astype(np.float32) * scale
    return vocab, vectors


def build_tokenizer(vocab: list[str]):
    from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers

    # First-occurrence-wins on duplicate lemma strings (e.g. a genuine
    # WordNet lemma can collide with the "undefined" OOV sentinel at row 0)
    # to match the C engine's hash table exactly: ht_insert never overwrites
    # on collision, and ht_get returns the first match it probes to, which
    # is always the earliest-inserted row for an exact string match.
    word_to_id: dict[str, int] = {}
    for i, w in enumerate(vocab):
        word_to_id.setdefault(w, i)
    tok = Tokenizer(models.WordLevel(vocab=word_to_id, unk_token="undefined"))
    tok.normalizer = normalizers.Lowercase()
    # Matches embed_text()'s tokenisation: maximal runs of ASCII letters are
    # tokens, everything else (digits, punctuation, whitespace) is a
    # delimiter and contributes nothing.
    tok.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(r"[^a-zA-Z]+"), behavior="removed")
    return tok


def export(
    vocab_path: Path = VOCAB_PATH,
    embed_path: Path = EMBED_PATH,
    output_dir: Path = ST_EXPORT_DIR,
) -> None:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    vocab, vectors = load_wneb(vocab_path, embed_path)
    log.info("Loaded %d lemma vectors (%d-dim) from %s", *vectors.shape, embed_path)

    tokenizer = build_tokenizer(vocab)
    static_embedding = StaticEmbedding(tokenizer, embedding_weights=vectors)
    model = SentenceTransformer(modules=[static_embedding])

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    log.info("Saved sentence-transformers model -> %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=VOCAB_PATH)
    parser.add_argument("--embeddings", type=Path, default=EMBED_PATH)
    parser.add_argument("--output", type=Path, default=ST_EXPORT_DIR)
    args = parser.parse_args()
    export(args.vocab, args.embeddings, args.output)


if __name__ == "__main__":
    main()
