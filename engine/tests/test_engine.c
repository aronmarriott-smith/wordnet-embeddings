/*
 * C unit tests for the embedding engine.
 * Tests mechanics against a tiny synthetic fixture — not model quality.
 *
 * Fixture (3 entries, 4-dim, scale=1/127):
 *   row 0 "undefined": [  0, 100,   0, 0 ]  — non-zero for OOV verification
 *   row 1 "cat":       [127,   0,   0, 0 ]
 *   row 2 "dog":       [  0,   0, 127, 0 ]
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "embed.h"

static const uint32_t FX_DIM   = 4;
static const float    FX_SCALE = 1.0f / 127.0f;
static const char    *fx_dir   = "embed_test_fixture";

static void write_fixture(void) {
#ifdef _WIN32
    mkdir(fx_dir);
#else
    mkdir(fx_dir, 0700);
#endif

    char p[512];

    snprintf(p, sizeof p, "%s/vocab.txt", fx_dir);
    FILE *f = fopen(p, "w");
    assert(f);
    fprintf(f, "undefined\ncat\ndog\n");
    fclose(f);

    snprintf(p, sizeof p, "%s/embeddings.bin", fx_dir);
    f = fopen(p, "wb");
    assert(f);
    fwrite("WNEB", 1, 4, f);
    uint32_t vs = 3, dim = FX_DIM;
    float scale = FX_SCALE;
    fwrite(&vs,    4, 1, f);
    fwrite(&dim,   4, 1, f);
    fwrite(&scale, 4, 1, f);
    int8_t rows[3][4] = {
        {  0, 100,   0, 0 },
        {127,   0,   0, 0 },
        {  0,   0, 127, 0 },
    };
    fwrite(rows, 1, sizeof rows, f);
    fclose(f);
}

static void cleanup_fixture(void) {
    char p[512];
    snprintf(p, sizeof p, "%s/vocab.txt", fx_dir);
    remove(p);
    snprintf(p, sizeof p, "%s/embeddings.bin", fx_dir);
    remove(p);
    remove(fx_dir);
}

static float l2norm(const float *v, uint32_t n) {
    float s = 0;
    for (uint32_t i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

int main(void) {
    write_fixture();
    embed_model *m = embed_load(fx_dir);
    assert(m && "embed_load failed");

    float out[EMBED_DIM];

    /* 1. known word "cat" -> row 1 -> [1.0, 0, 0, 0] after L2 norm */
    memset(out, 0, sizeof out);
    assert(embed_text(m, "cat", out) == 0);
    assert(fabsf(out[0] - 1.0f) < 1e-5f);
    assert(fabsf(out[1]) < 1e-5f);
    assert(fabsf(l2norm(out, FX_DIM) - 1.0f) < 1e-5f);
    printf("PASS: cat        -> [%.3f, %.3f, %.3f, %.3f]\n",
           out[0], out[1], out[2], out[3]);

    /* 2. known word "dog" -> row 2 -> [0, 0, 1.0, 0] */
    memset(out, 0, sizeof out);
    assert(embed_text(m, "dog", out) == 0);
    assert(fabsf(out[2] - 1.0f) < 1e-5f);
    assert(fabsf(l2norm(out, FX_DIM) - 1.0f) < 1e-5f);
    printf("PASS: dog        -> [%.3f, %.3f, %.3f, %.3f]\n",
           out[0], out[1], out[2], out[3]);

    /* 3. OOV word -> undefined (row 0) -> [0, 1.0, 0, 0] after norm */
    memset(out, 0, sizeof out);
    assert(embed_text(m, "xyz", out) == 0);
    assert(fabsf(out[1] - 1.0f) < 1e-5f);
    assert(fabsf(l2norm(out, FX_DIM) - 1.0f) < 1e-5f);
    printf("PASS: xyz (OOV)  -> [%.3f, %.3f, %.3f, %.3f]\n",
           out[0], out[1], out[2], out[3]);

    /* 4. punctuation-only -> empty tokenisation -> inject undefined */
    memset(out, 0, sizeof out);
    assert(embed_text(m, "!!!", out) == 0);
    assert(fabsf(out[1] - 1.0f) < 1e-5f);
    printf("PASS: !!! (empty)-> [%.3f, %.3f, %.3f, %.3f]\n",
           out[0], out[1], out[2], out[3]);

    /* 5. multi-word "cat dog" -> mean of rows 1+2 -> [1/√2, 0, 1/√2, 0] */
    memset(out, 0, sizeof out);
    assert(embed_text(m, "cat dog", out) == 0);
    float expected = 1.0f / sqrtf(2.0f);
    assert(fabsf(out[0] - expected) < 1e-4f);
    assert(fabsf(out[2] - expected) < 1e-4f);
    assert(fabsf(l2norm(out, FX_DIM) - 1.0f) < 1e-5f);
    printf("PASS: \"cat dog\" -> [%.3f, %.3f, %.3f, %.3f]\n",
           out[0], out[1], out[2], out[3]);

    /* 6. output is always L2-normalised */
    memset(out, 0, sizeof out);
    embed_text(m, "cat", out);
    assert(fabsf(l2norm(out, FX_DIM) - 1.0f) < 1e-5f);
    printf("PASS: output is L2-normalised\n");

    embed_free(m);
    cleanup_fixture();
    printf("All tests passed.\n");
    return 0;
}
