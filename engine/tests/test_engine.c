/*
 * C-level unit tests for the embedding engine primitives.
 *
 * See CUSTOM_EMBEDDINGS_RESEARCH.md (green-ai repo), Part 4, "Testing the
 * inference engine" — these test mechanics (lookup, dequantisation,
 * pooling, normalisation, `undefined` fallback) against a tiny synthetic
 * table, NOT the trained model's quality.
 *
 * TODO: build a small fixture table (a handful of known words + the
 * `undefined` entry) and assert on embed_text() for known inputs:
 * - known single-word input -> expected vector
 * - multi-word chunk -> correct mean-pooling
 * - OOV word -> `undefined` vector contribution
 * - empty/punctuation-only input -> doesn't crash, returns `undefined`
 * - output vector has length EMBED_DIM and is L2-normalised (norm ~= 1.0)
 */

#include <stdio.h>

#include "embed.h"

int main(void) {
    printf("test_engine: no tests implemented yet (scaffolding only)\n");
    return 0;
}
