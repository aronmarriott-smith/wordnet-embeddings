# WordNet Embeddings (Level 0)

Custom, from-scratch knowledge-graph embeddings trained on
[Princeton WordNet](https://wordnet.princeton.edu/), designed to run on
constrained hardware (originally targeting a Raspberry Pi 2B — 1 GB RAM,
armv7, no GPU).

This is the implementation companion to the research/planning doc in the
[green-ai](https://github.com/aronmarriott-smith/green-ai) project:
`CUSTOM_EMBEDDINGS_RESEARCH.md`. Read that first for the full rationale and
design decisions — this repo is the scaffolding for "Level 0" as defined
there, included as a git submodule of green-ai.

## Decisions (see the research doc for rationale)

- **Training approach:** typed knowledge-graph embedding (TransE via
  [PyKEEN](https://github.com/pykeen/pykeen)) over WordNet's synset relation
  graph — "Option A".
- **Dimensionality:** 128 — our own embedding space, no compatibility target
  with any other model.
- **Scope:** English only (+ symbols/punctuation), inherited from WordNet.
- **OOV handling:** a dedicated `undefined` embedding-table entry, used for
  every out-of-vocabulary token.
- **Training hardware:** any CPU machine (an Intel i7 MBP was used for
  development) — no GPU required for Level 0.
- **Inference:** a small custom C engine (`engine/`), exposed to Python via
  `ctypes` (`wordnet_embeddings/`).

## Layout

```
wordnet_embeddings/   # Python: graph extraction, training, export
engine/               # C inference engine + its own test suite
tests/                # Python integration tests (ctypes <-> engine)
data/                 # Generated graphs/models/exports (gitignored)
```

## Status

🚧 **Scaffolding only.** Nothing here trains or runs yet — see the `TODO`
comments in each module, and the checklist below.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python3 -c "import nltk; nltk.download('wordnet')"
```

## Open items

(mirrors the "Open questions / next steps" list in
`CUSTOM_EMBEDDINGS_RESEARCH.md`)

- [ ] Decide which WordNet relation types to include as graph edges
- [ ] Implement `wordnet_embeddings/build_graph.py` — export WordNet synsets
      + relations as PyKEEN-compatible triples
- [ ] Implement `wordnet_embeddings/train.py` — 128-dim TransE training via
      PyKEEN
- [ ] Design the binary embedding-table format (int8, header, `undefined`
      entry, mmap-friendly) and implement `wordnet_embeddings/export.py`
- [ ] Export WordNet's lemmatisation rules for the C engine's tokeniser
- [ ] Implement `engine/src/embed.c` (tokenise, lookup, pool, normalise)
      and its test suite (`engine/tests/`, `tests/`)
- [ ] Cross-compile/smoke-test `engine/` on the Raspberry Pi 2B
- [ ] Reference exports (word2vec format, `sentence-transformers` directory
      for MTEB)

## License

Apache 2.0 — see `LICENSE`.
