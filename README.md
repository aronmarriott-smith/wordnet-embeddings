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
wordnet_embeddings/   # Python: graph extraction, training, export, benchmarking
engine/               # C inference engine + its own test suite
tests/                # Python integration tests (ctypes <-> engine)
data/                 # Generated graphs/models/exports (gitignored)
benchmarks/           # Tracked MTEB STS results (benchmarks/raw/ is gitignored)
```

## Status

The full pipeline works end-to-end: graph extraction → training (CPU or
GPU) → export → C engine inference, plus an MTEB benchmarking path. See
`bin/` for the runnable scripts and Usage below.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python3 -c "import nltk; nltk.download('wordnet')"
```

## Usage

Each step is a `bin/*.sh` script — run from the repo root, in order:

```bash
bin/build_graph.sh   # graph source -> data/triples.tsv + data/lemma_synsets.tsv
bin/train.sh         # TransE training -> data/model/ (--epochs N for a quick smoke-test)
bin/export.sh        # data/model/ -> data/vocab.txt + data/embeddings.bin (the C engine's format)
bin/test.sh          # builds engine/libembed.so, runs the ctypes integration tests
```

`bin/train.sh` auto-detects an NVIDIA GPU (`nvidia-smi` on PATH) and
installs CUDA-enabled torch if present; falls back to CPU otherwise.

### Building the C engine on Windows

Native Windows MinGW (`engine/Makefile`, used by `bin/test.sh`) has repeatedly
hit environment issues on this project (PATH loss, temp-directory resolution
failures deep in gcc's own subprocess spawning). **Recommended:** cross-compile
from a Linux Docker container instead — same toolchain family, no native
Windows gcc involved:

```powershell
.\bin\build_windows.ps1   # or bin/build_windows.sh from Git Bash
```

This builds a small Debian + `gcc-mingw-w64-x86-64` image
(`engine/Dockerfile.windows`) and runs `make CC=x86_64-w64-mingw32-gcc lib`
with `engine/` bind-mounted, so the output (`engine/libembed.so` — a real
Windows PE DLL, named `.so` to match the project's ctypes-loading convention)
lands directly on the host. Builds the library only — the resulting `.exe`
can't run inside the Linux container, so verify it for real afterwards via
`pytest tests/` or `bin/test.sh` on Windows. Requires Docker Desktop.

`bin/build_graph.sh` extracts from a pluggable `GraphSource`
(`wordnet_embeddings/sources/`), selected with `--source` (default
`wordnet`):

| `--source` | Data | Relations |
|---|---|---|
| `wordnet` (default) | NLTK's Princeton WordNet 3.0 | hypernym/hyponym/meronym/etc. (22 types) |
| `oewn` | [Open English WordNet](https://github.com/globalwordnet/english-wordnet), via NLTK's `english_wordnet` package | same relation set — same Synset API as Princeton WordNet |

Adding another English lexical graph means implementing the `GraphSource`
protocol in a new module under `wordnet_embeddings/sources/` and registering
it in `sources/__init__.py` — no changes needed to `build_graph.py` itself.

`--extra-vocab words stopwords` appends NLTK's flat word lists (Words
Corpus, Stopwords Corpus) to the lemma map alongside the main source. These
have no relations, so **they currently have no effect on the trained
embedding table** — `export.py` only keeps a lemma whose entity got a
trained embedding (i.e. appeared in at least one triple), so these words
fall back to the `undefined` OOV vector at export time like any other
out-of-vocabulary word. Wired in as plumbing for when that changes (e.g.
once these are tied into relations, or distributional signal is added —
see `CUSTOM_EMBEDDINGS_RESEARCH.md` Part 6, "Level 0.5").

**Not implemented as sources** (see `CUSTOM_EMBEDDINGS_RESEARCH.md` Part 6):
Brown, Gutenberg, WebText, and NPS Chat are plain/tagged text corpora with
no entity relations — they fit a future distributional/co-occurrence
embedding pipeline ("Level 0.5"), not this triples-graph extractor. The CMU
Pronouncing Dictionary maps words to phoneme sequences, not entity-to-entity
relations — revisit only if a phonetic-similarity relation type is wanted.

### Benchmarking (MTEB STS)

The model can be evaluated against MTEB's English Semantic Textual
Similarity (STS) task family — the closest standard, comparable-across-models
benchmark for a static, mean-pooled lemma embedding like this one (see
`docs/CUSTOM_EMBEDDINGS_RESEARCH.md` Part 5 for why MTEB/`sentence-transformers`
compatibility was the chosen evaluation path).

```bash
bin/export_sentence_transformer.sh   # data/vocab.txt + embeddings.bin -> data/sentence_transformers/
bin/benchmark.sh                     # runs MTEB's English STS tasks against it
```

This writes a timestamped `benchmarks/sts_summary_<UTC timestamp>.json`
(committed — this is the cross-iteration tracking history; `benchmarks/raw/`,
mteb's detailed per-task dump, is regenerated each run and gitignored).

**Reading the scores:** these are Spearman-correlation-based STS scores
(0-1), the standard metric, but this model isn't competing on the usual
leaderboard terms. It's a static, mean-pooled, closed-WordNet-vocabulary
embedding (architecturally closer to word2vec/GloVe than to a transformer
embedding model), and `embed_text()` doesn't lemmatise yet (see
`docs/CUSTOM_EMBEDDINGS_RESEARCH.md` Part 1) — so inflected words miss the
vocab and fall back to the `undefined` entry. Modern transformer embedding
models score 0.80-0.90+ on the same tasks; ~0.35 is this architecture's
current baseline, not a bug. The summary file's value is in tracking *this
model's own* improvement across iterations (e.g. after adding lemmatisation,
gloss-text retrofitting, or a larger `EMBED_DIM`), not chasing that absolute
number.

## Open items

(mirrors the "Open questions / next steps" list in
`CUSTOM_EMBEDDINGS_RESEARCH.md`)

- [x] Decide which WordNet relation types to include as graph edges
- [x] Implement `wordnet_embeddings/build_graph.py` — export WordNet synsets
      + relations as PyKEEN-compatible triples
- [x] Implement `wordnet_embeddings/train.py` — 128-dim TransE training via
      PyKEEN
- [x] Design the binary embedding-table format (int8, header, `undefined`
      entry, mmap-friendly) and implement `wordnet_embeddings/export.py`
- [ ] Export WordNet's lemmatisation rules for the C engine's tokeniser —
      `embed_text()` currently tokenises but doesn't lemmatise; see the
      Benchmarking section above
- [x] Implement `engine/src/embed.c` (tokenise, lookup, pool, normalise)
      and its test suite (`engine/tests/`, `tests/`)
- [ ] Cross-compile/smoke-test `engine/` on the Raspberry Pi 2B (built and
      tested on Linux/Windows dev machines so far)
- [x] Reference exports (`sentence-transformers` directory for MTEB) — see
      `bin/export_sentence_transformer.sh`. word2vec-format export still open.
- [x] Generalise `build_graph.py` behind a `GraphSource` protocol
      (`wordnet_embeddings/sources/`) so other English lexical graphs (Open
      English WordNet, ConceptNet, Wiktionary — see
      `CUSTOM_EMBEDDINGS_RESEARCH.md` Part 6) can be added without touching
      `build_graph.py`. Only `WordNetSource` is implemented so far.

## License

Apache 2.0 — see `LICENSE`.
