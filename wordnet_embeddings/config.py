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
