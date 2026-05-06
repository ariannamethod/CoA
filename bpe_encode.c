/*
 * bpe_encode.c — pre-encode text corpus to binary tokens file
 *
 * Usage:
 *   ./bpe_encode merges.txt corpus.txt out.tokens
 *
 * Output format:
 *   [int32 n_tokens][int32 * n_tokens]   -- little-endian on host arch
 *
 * Reuses notorch's nt_bpe_load + nt_bpe_encode (post-fix two-pointer O(n) per merge).
 *
 * Build: cc -O3 -march=native -DUSE_SIMD -mavx2 -mfma bpe_encode.c notorch.c -lm -lpthread -o bpe_encode
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "notorch.h"

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1.0e6;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s merges.txt corpus.txt out.tokens\n", argv[0]);
        return 1;
    }
    const char* merges_path = argv[1];
    const char* corpus_path = argv[2];
    const char* out_path    = argv[3];

    nt_bpe bpe;
    int nm = nt_bpe_load(&bpe, merges_path);
    if (nm <= 0) {
        fprintf(stderr, "[error] cannot load BPE merges from %s\n", merges_path);
        return 2;
    }
    fprintf(stderr, "[bpe_encode] BPE: %s — %d merges, vocab=%d\n",
            merges_path, nm, bpe.vocab_size);

    FILE* f = fopen(corpus_path, "rb");
    if (!f) { fprintf(stderr, "[error] cannot open corpus %s\n", corpus_path); return 3; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); fprintf(stderr, "[error] OOM\n"); return 4; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f); free(buf);
        fprintf(stderr, "[error] short read\n"); return 5;
    }
    buf[sz] = 0;
    fclose(f);
    fprintf(stderr, "[bpe_encode] corpus: %s (%.1f KB)\n", corpus_path, sz / 1024.0);

    int* tokens = (int*)malloc((size_t)sz * sizeof(int));
    if (!tokens) { free(buf); fprintf(stderr, "[error] OOM tokens\n"); return 6; }

    fprintf(stderr, "[bpe_encode] encoding...\n");
    double t0 = now_s();
    int n_tokens = nt_bpe_encode(&bpe, buf, (int)sz, tokens, (int)sz);
    double t1 = now_s();
    fprintf(stderr, "[bpe_encode] %d tokens in %.2fs (compression %.2fx, %.1f MB/s)\n",
            n_tokens, t1 - t0,
            (double)sz / (double)n_tokens,
            (sz / 1024.0 / 1024.0) / (t1 - t0));

    free(buf);

    FILE* o = fopen(out_path, "wb");
    if (!o) { free(tokens); fprintf(stderr, "[error] cannot open %s for write\n", out_path); return 7; }
    int32_t header = (int32_t)n_tokens;
    fwrite(&header, sizeof(int32_t), 1, o);
    if (fwrite(tokens, sizeof(int), n_tokens, o) != (size_t)n_tokens) {
        fclose(o); free(tokens);
        fprintf(stderr, "[error] short write\n"); return 8;
    }
    fclose(o);
    free(tokens);

    long out_sz = (long)(sizeof(int32_t) + (size_t)n_tokens * sizeof(int));
    fprintf(stderr, "[bpe_encode] saved %s (%.1f KB, header + %d tokens)\n",
            out_path, out_sz / 1024.0, n_tokens);
    return 0;
}
