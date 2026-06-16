#ifndef WORDNET_EMBED_H
#define WORDNET_EMBED_H

/*
 * Minimal embedding lookup/pooling engine.
 *
 * See CUSTOM_EMBEDDINGS_RESEARCH.md (green-ai repo), Part 4, for the full
 * design:
 * - Embedding table: flat binary file, header + int8[EMBED_DIM] records,
 *   mmap'd (word2vec `.bin`-style).
 * - embed_text: tokenise -> lemmatise -> lookup (falling back to the
 *   `undefined` entry for OOV tokens) -> mean-pool -> L2-normalise.
 *
 * TODO: this header/implementation is a stub. Nothing is implemented yet.
 */

#define EMBED_DIM 128

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded embedding table. */
typedef struct embed_model embed_model;

/* Load the embedding table from `dir` (expects dir/embeddings.bin and dir/vocab.txt).
 * Returns NULL on failure. */
embed_model *embed_load(const char *dir);

/* Free resources associated with `model`. */
void embed_free(embed_model *model);

/*
 * Embed `text` using `model`, writing EMBED_DIM floats to `out_vec`
 * (caller-allocated, L2-normalised on return).
 * Returns 0 on success, non-zero on error.
 */
int embed_text(embed_model *model, const char *text, float *out_vec);

#ifdef __cplusplus
}
#endif

#endif /* WORDNET_EMBED_H */
