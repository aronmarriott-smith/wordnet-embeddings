"""Central configuration for all wordnet_embeddings modules."""

from pathlib import Path

# ---- Model hyperparameters ----
EMBED_DIM  = 128  # our own embedding space; also mirrored in engine/include/embed.h
NUM_EPOCHS = 200

# ---- Binary format ----
WNEB_MAGIC     = b"WNEB"
UNDEFINED_LEMMA = "undefined"  # OOV fallback entry; always row 0 of the embedding table

# ---- Data paths ----
TRIPLES_PATH   = Path("data/triples.tsv")
LEMMA_MAP_PATH = Path("data/lemma_synsets.tsv")
MODEL_DIR      = Path("data/model")
VOCAB_PATH     = Path("data/vocab.txt")
EMBED_PATH     = Path("data/embeddings.bin")
ST_EXPORT_DIR  = Path("data/sentence_transformers")  # MTEB/sentence-transformers-loadable export

# ---- Engine build command, by os.name (tests/test_embed_ctypes.py) ----
# Windows has no reliable native toolchain for engine/ (see README's "Building
# the C engine on Windows" — repeated PATH/temp-dir failures in native MinGW).
# The fix is bin/build_windows.ps1 (Docker cross-compile), but it's too slow
# and too heavy a dependency (requires Docker running) to auto-invoke on every
# test session, so Windows has no auto-build command: tests/test_embed_ctypes.py
# just uses whatever engine/libembed.so already exists and tells you to run
# that script yourself if it's missing. Other platforms still auto-rebuild via
# `make lib`, which is cheap and has no extra dependencies.
ENGINE_BUILD_COMMAND: dict[str, list[str] | None] = {
    "nt": None,
    "posix": ["make", "lib"],
}
