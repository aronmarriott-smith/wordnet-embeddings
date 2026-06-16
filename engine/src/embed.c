/* embed.c — WordNet embedding inference engine (WNEB format) */

#include "embed.h"

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define WNEB_MAGIC  "WNEB"
#define HEADER_SIZE 16  /* magic(4) + vocab_size(4) + embed_dim(4) + scale(4) */

/* ---- open-addressing hash table: lemma -> row index ---- */

typedef struct { char *key; uint32_t row; } ht_slot;
typedef struct { ht_slot *slots; uint32_t cap; } ht;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) h = (h ^ (unsigned char)*s++) * 16777619u;
    return h;
}

static int ht_init(ht *t, uint32_t n) {
    uint32_t cap = 16;
    while (cap < n * 2) cap <<= 1;  /* 0.5 max load factor */
    t->slots = calloc(cap, sizeof(ht_slot));
    t->cap   = cap;
    return t->slots ? 0 : -1;
}

static void ht_free(ht *t) {
    for (uint32_t i = 0; i < t->cap; i++) free(t->slots[i].key);
    free(t->slots);
}

static int ht_insert(ht *t, const char *key, uint32_t row) {
    uint32_t i = fnv1a(key) & (t->cap - 1);
    while (t->slots[i].key) i = (i + 1) & (t->cap - 1);
    t->slots[i].key = strdup(key);
    t->slots[i].row = row;
    return t->slots[i].key ? 0 : -1;
}

/* Returns row index, or 0 (undefined fallback) on miss. */
static uint32_t ht_get(const ht *t, const char *key) {
    uint32_t i = fnv1a(key) & (t->cap - 1);
    while (t->slots[i].key) {
        if (strcmp(t->slots[i].key, key) == 0) return t->slots[i].row;
        i = (i + 1) & (t->cap - 1);
    }
    return 0;
}

/* ---- model ---- */

struct embed_model {
    void         *map;       /* mmap of embeddings.bin */
    size_t        map_size;
    const int8_t *table;     /* int8[vocab_size * embed_dim] within map */
    uint32_t      vocab_size;
    uint32_t      embed_dim;
    float         scale;     /* dequantise: float_val = int8_val * scale */
    ht            vocab;     /* lemma -> row index */
};

/* ---- public API ---- */

embed_model *embed_load(const char *dir) {
    embed_model *m = calloc(1, sizeof *m);
    if (!m) return NULL;

    char path[4096];
    snprintf(path, sizeof path, "%s/embeddings.bin", dir);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { free(m); return NULL; }

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); free(m); return NULL; }

    m->map_size = (size_t)st.st_size;
    m->map = mmap(NULL, m->map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->map == MAP_FAILED) { free(m); return NULL; }

    const unsigned char *p = (const unsigned char *)m->map;
    if (m->map_size < HEADER_SIZE || memcmp(p, WNEB_MAGIC, 4) != 0) {
        munmap(m->map, m->map_size); free(m); return NULL;
    }
    memcpy(&m->vocab_size, p + 4,  4);
    memcpy(&m->embed_dim,  p + 8,  4);
    memcpy(&m->scale,      p + 12, 4);
    m->table = (const int8_t *)(p + HEADER_SIZE);

    snprintf(path, sizeof path, "%s/vocab.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f || ht_init(&m->vocab, m->vocab_size) < 0) {
        if (f) fclose(f);
        munmap(m->map, m->map_size); free(m); return NULL;
    }

    char line[512];
    uint32_t row = 0;
    while (fgets(line, sizeof line, f) && row < m->vocab_size) {
        size_t n = strlen(line);
        if (n && line[n - 1] == '\n') line[--n] = '\0';
        if (n) ht_insert(&m->vocab, line, row);
        row++;
    }
    fclose(f);
    return m;
}

void embed_free(embed_model *m) {
    if (!m) return;
    ht_free(&m->vocab);
    if (m->map && m->map != MAP_FAILED) munmap(m->map, m->map_size);
    free(m);
}

int embed_text(embed_model *m, const char *text, float *out) {
    if (!m || !text || !out) return -1;

    uint32_t dim = m->embed_dim;
    float *acc = calloc(dim, sizeof(float));
    if (!acc) return -1;

    char tok[256];
    int ti = 0, count = 0;

    /* tokenise: split on non-alpha chars, lowercase each token, lookup */
    for (const char *p = text; ; p++) {
        if (*p && isalpha((unsigned char)*p)) {
            if (ti < (int)sizeof(tok) - 1)
                tok[ti++] = (char)tolower((unsigned char)*p);
        } else {
            if (ti > 0) {
                tok[ti] = '\0';
                uint32_t row = ht_get(&m->vocab, tok);
                const int8_t *vec = m->table + (size_t)row * dim;
                for (uint32_t d = 0; d < dim; d++) acc[d] += vec[d] * m->scale;
                count++;
                ti = 0;
            }
            if (!*p) break;
        }
    }

    /* empty / all-punctuation input: inject undefined row (row 0) */
    if (count == 0) {
        for (uint32_t d = 0; d < dim; d++) acc[d] = m->table[d] * m->scale;
        count = 1;
    }

    /* mean pool then L2-normalise into out */
    float inv = 1.0f / (float)count;
    float norm = 0.0f;
    for (uint32_t d = 0; d < dim; d++) {
        out[d] = acc[d] * inv;
        norm += out[d] * out[d];
    }
    free(acc);

    norm = sqrtf(norm);
    if (norm > 1e-8f) {
        inv = 1.0f / norm;
        for (uint32_t d = 0; d < dim; d++) out[d] *= inv;
    }

    return 0;
}
