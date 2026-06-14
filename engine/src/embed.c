#include "embed.h"

#include <stdlib.h>

/*
 * TODO (see CUSTOM_EMBEDDINGS_RESEARCH.md, Part 4):
 * - struct embed_model: mmap'd table pointer, vocab size, vocab index
 *   (hash map lemma -> row), quantisation scale, `undefined` row index.
 * - embed_load: open + mmap the binary table, build/load the vocab index.
 * - embed_text: tokenise (lowercase, strip punctuation, multi-word lemma
 *   check), lemmatise, look up each token (falling back to `undefined`),
 *   dequantise + accumulate a running mean, L2-normalise into out_vec.
 */

struct embed_model {
    int placeholder;
};

embed_model *embed_load(const char *path) {
    (void)path;
    return NULL;
}

void embed_free(embed_model *model) {
    free(model);
}

int embed_text(embed_model *model, const char *text, float *out_vec) {
    (void)model;
    (void)text;
    (void)out_vec;
    return -1; /* not implemented */
}
