/*
 * coa_infer.c — load saved CoA-v1 .bin + run generation across multiple temps.
 *
 * Build:
 *   make coa_infer       # CPU, SIMD AVX2+FMA via env CFLAGS
 *
 * Run:
 *   ./coa_infer coa_v1_paired_on.bin
 *   ./coa_infer coa_v1_paired_off.bin
 *
 * Output: 4 prompts × 4 temperatures = 16 generation samples per .bin.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "notorch.h"
#include "loragrad.h"

#define COA_BLOCK_SIZE    512
#define COA_N_LAYER         5
#define COA_N_EMBD        512
#define COA_N_HEAD          8
#define COA_HEAD_DIM       (COA_N_EMBD / COA_N_HEAD)
#define COA_RRPRAM_R       32
#define COA_MLP_DIM      1024
#define COA_GEN_LEN        60
#define COA_VOCAB_BASE   2048

typedef struct {
    int vocab_size, n_layer, n_embd, n_head, head_dim, block_size, rrpram_r;
    nt_tensor* wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo;
        nt_tensor *wr_combined, *wvr, *wj;
        nt_tensor *rms2, *w_gate, *w_up, *w_down;
    } L[8];
    nt_tensor* rms_final;
    nt_tensor* lm_head;
} coa_model;

static int coa_forward(coa_model* m, int* tokens, int* targets) {
    int T = m->block_size, E = m->n_embd, V = m->vocab_size, H = m->n_head, D = m->head_dim;
    int wte_i = nt_tape_param(m->wte); nt_tape_no_decay(wte_i);
    int li[8][12];
    for (int l = 0; l < m->n_layer; ++l) {
        li[l][0]  = nt_tape_param(m->L[l].rms1);  nt_tape_no_decay(li[l][0]);
        li[l][1]  = nt_tape_param(m->L[l].wq);
        li[l][2]  = nt_tape_param(m->L[l].wk);
        li[l][3]  = nt_tape_param(m->L[l].wv);
        li[l][4]  = nt_tape_param(m->L[l].wo);
        li[l][5]  = nt_tape_param(m->L[l].wr_combined);
        li[l][6]  = nt_tape_param(m->L[l].wvr);
        li[l][7]  = nt_tape_param(m->L[l].wj);
        li[l][8]  = nt_tape_param(m->L[l].rms2);  nt_tape_no_decay(li[l][8]);
        li[l][9]  = nt_tape_param(m->L[l].w_gate);
        li[l][10] = nt_tape_param(m->L[l].w_up);
        li[l][11] = nt_tape_param(m->L[l].w_down);
    }
    int rmsf_i = nt_tape_param(m->rms_final); nt_tape_no_decay(rmsf_i);
    int head_i = nt_tape_param(m->lm_head);
    nt_tensor* tok_t = nt_tensor_new(T);
    nt_tensor* tgt_t = nt_tensor_new(T);
    for (int i = 0; i < T; ++i) { tok_t->data[i] = (float)tokens[i]; tgt_t->data[i] = (float)targets[i]; }
    int tok_i = nt_tape_record(tok_t, NT_OP_NONE, -1, -1, 0);
    int tgt_i = nt_tape_record(tgt_t, NT_OP_NONE, -1, -1, 0);
    nt_tensor_free(tok_t); nt_tensor_free(tgt_t);
    int h = nt_seq_embedding(wte_i, -1, tok_i, T, E);
    for (int l = 0; l < m->n_layer; ++l) {
        int xn = nt_seq_rmsnorm(h, li[l][0], T, E);
        int q = nt_seq_linear(li[l][1], xn, T);
        int k = nt_seq_linear(li[l][2], xn, T);
        int v = nt_seq_linear(li[l][3], xn, T);
        q = nt_rope(q, T, D); k = nt_rope(k, T, D);
        int out_c = nt_mh_causal_attention(q, k, v, T, D);
        int v_r   = nt_seq_linear(li[l][6], xn, T);
        int out_r = nt_rrpram_lowrank_attention(li[l][5], xn, v_r, T, E, H, D);
        int out_e = nt_seq_linear(li[l][7], xn, T);
        int sum_cr  = nt_add(out_c, out_r);
        int sum_cre = nt_add(sum_cr, out_e);
        int blended = nt_scale(sum_cre, 1.0f / 3.0f);
        int proj = nt_seq_linear(li[l][4], blended, T);
        h = nt_add(h, proj);
        xn = nt_seq_rmsnorm(h, li[l][8], T, E);
        int gate_pre = nt_seq_linear(li[l][9],  xn, T);
        int up       = nt_seq_linear(li[l][10], xn, T);
        int swi      = nt_swiglu(gate_pre, up);
        int down     = nt_seq_linear(li[l][11], swi, T);
        h = nt_add(h, down);
    }
    int hf     = nt_seq_rmsnorm(h, rmsf_i, T, E);
    int logits = nt_seq_linear(head_i, hf, T);
    return nt_seq_cross_entropy(logits, tgt_i, T, V);
}

static void coa_generate(coa_model* m, nt_bpe* bpe, const char* prompt,
                         int max_tokens, float temp, int top_k)
{
    int T = m->block_size, V = m->vocab_size;
    int ctx[COA_BLOCK_SIZE];
    int gen_len = 0;
    int prompt_len = (int)strlen(prompt);
    int tmp_buf[COA_BLOCK_SIZE];
    int n_pt = nt_bpe_encode(bpe, prompt, prompt_len, tmp_buf, T / 2);
    for (int i = 0; i < n_pt; ++i) ctx[gen_len++] = tmp_buf[i];

    printf("%s", prompt);
    nt_train_mode(0);
    for (int s = 0; s < max_tokens && gen_len < T - 1; ++s) {
        int tokens[COA_BLOCK_SIZE], targets[COA_BLOCK_SIZE];
        for (int i = 0; i < gen_len; ++i) tokens[i] = ctx[i];
        for (int i = gen_len; i < T; ++i) tokens[i] = 0;
        memset(targets, 0, sizeof(targets));
        nt_tape_start();
        int loss_idx = coa_forward(m, tokens, targets);
        nt_tape* tape = nt_tape_get();
        int logits_idx = tape->entries[loss_idx].parent1;
        float* last_logits = tape->entries[logits_idx].output->data + (gen_len - 1) * V;
        for (int i = 0; i < V; ++i) last_logits[i] /= temp;
        /* top-k filter: zero everything below top-k */
        if (top_k > 0 && top_k < V) {
            float thr = last_logits[0];
            float copy[V];
            for (int i = 0; i < V; ++i) copy[i] = last_logits[i];
            /* partial sort — find k-th largest via simple selection */
            for (int k = 0; k < top_k; ++k) {
                float mx = -1e30f; int mi = 0;
                for (int i = 0; i < V; ++i) if (copy[i] > mx) { mx = copy[i]; mi = i; }
                copy[mi] = -1e30f;
                if (k == top_k - 1) thr = mx;
            }
            for (int i = 0; i < V; ++i) if (last_logits[i] < thr) last_logits[i] = -1e9f;
        }
        float mx = last_logits[0];
        for (int i = 1; i < V; ++i) if (last_logits[i] > mx) mx = last_logits[i];
        float sm = 0;
        for (int i = 0; i < V; ++i) { last_logits[i] = expf(last_logits[i] - mx); sm += last_logits[i]; }
        for (int i = 0; i < V; ++i) last_logits[i] /= sm;
        float r = (float)rand() / (float)RAND_MAX, cum = 0;
        int next = 0;
        for (int i = 0; i < V; ++i) { cum += last_logits[i]; if (cum >= r) { next = i; break; } }
        char obuf[NT_BPE_MAX_TOKEN_LEN + 1];
        int olen = nt_bpe_decode(bpe, &next, 1, obuf, sizeof(obuf));
        for (int i = 0; i < olen; ++i) {
            unsigned char c = (unsigned char)obuf[i];
            if (c >= 32 && c < 127) printf("%c", c);
            else if (c == '\n') printf("\n");
            else printf("?");
        }
        fflush(stdout);
        ctx[gen_len++] = next;
        nt_tape_clear();
    }
    nt_train_mode(1);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin\n", argv[0]); return 1; }
    const char* bin_path = argv[1];
    srand(42);

    /* BPE */
    nt_bpe bpe;
    if (nt_bpe_load(&bpe, "bpe_2048_merges.txt") <= 0) {
        fprintf(stderr, "fatal: cannot load bpe_2048_merges.txt\n"); return 2;
    }

    /* Init model — fresh tensors that we'll overwrite from .bin */
    coa_model m;
    memset(&m, 0, sizeof(m));
    m.vocab_size = bpe.vocab_size;
    m.n_layer    = COA_N_LAYER;
    m.n_embd     = COA_N_EMBD;
    m.n_head     = COA_N_HEAD;
    m.head_dim   = COA_HEAD_DIM;
    m.block_size = COA_BLOCK_SIZE;
    m.rrpram_r   = COA_RRPRAM_R;
    int E = COA_N_EMBD, T = COA_BLOCK_SIZE, H = COA_N_HEAD, R = COA_RRPRAM_R, M = COA_MLP_DIM;
    m.wte = nt_tensor_new2d(m.vocab_size, E);
    for (int l = 0; l < m.n_layer; ++l) {
        m.L[l].rms1 = nt_tensor_new(E);
        m.L[l].wq   = nt_tensor_new2d(E, E);
        m.L[l].wk   = nt_tensor_new2d(E, E);
        m.L[l].wv   = nt_tensor_new2d(E, E);
        m.L[l].wo   = nt_tensor_new2d(E, E);
        m.L[l].wr_combined = nt_tensor_new(H * R * (E + T));
        m.L[l].wvr  = nt_tensor_new2d(E, E);
        m.L[l].wj   = nt_tensor_new2d(E, E);
        m.L[l].rms2 = nt_tensor_new(E);
        m.L[l].w_gate = nt_tensor_new2d(M, E);
        m.L[l].w_up   = nt_tensor_new2d(M, E);
        m.L[l].w_down = nt_tensor_new2d(E, M);
    }
    m.rms_final = nt_tensor_new(E);
    m.lm_head   = nt_tensor_new2d(m.vocab_size, E);

    /* Build flat array for direct nt_load order match */
    nt_tensor* refs[64];
    int rn = 0;
    refs[rn++] = m.wte;
    for (int l = 0; l < m.n_layer; ++l) {
        refs[rn++] = m.L[l].rms1;
        refs[rn++] = m.L[l].wq;
        refs[rn++] = m.L[l].wk;
        refs[rn++] = m.L[l].wv;
        refs[rn++] = m.L[l].wo;
        refs[rn++] = m.L[l].wr_combined;
        refs[rn++] = m.L[l].wvr;
        refs[rn++] = m.L[l].wj;
        refs[rn++] = m.L[l].rms2;
        refs[rn++] = m.L[l].w_gate;
        refs[rn++] = m.L[l].w_up;
        refs[rn++] = m.L[l].w_down;
    }
    refs[rn++] = m.rms_final;
    refs[rn++] = m.lm_head;

    /* Load .bin and copy data into existing model tensors */
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(bin_path, &n_loaded);
    if (!loaded) { fprintf(stderr, "fatal: nt_load failed for %s\n", bin_path); return 3; }
    if (n_loaded != rn) {
        fprintf(stderr, "fatal: param count mismatch (loaded %d, expected %d)\n", n_loaded, rn);
        return 4;
    }
    for (int i = 0; i < rn; ++i) {
        if (loaded[i]->len != refs[i]->len) {
            fprintf(stderr, "fatal: param[%d] len mismatch (loaded %d, expected %d)\n",
                    i, loaded[i]->len, refs[i]->len);
            return 5;
        }
        memcpy(refs[i]->data, loaded[i]->data, sizeof(float) * refs[i]->len);
        nt_tensor_free(loaded[i]);
    }
    free(loaded);
    printf("[load] %d params from %s\n\n", rn, bin_path);

    /* Generation grid: 4 prompts × 4 temps × 1 top_k variant */
    const char* prompts[] = {
        "The chain ",
    };
    float temps[] = { 0.3f, 0.5f, 0.8f, 1.0f };
    int top_ks[] = { 40, 40, 40, 0 };  /* last: no top-k = full vocab at temp=1.0 */
    int n_prompts = 1, n_temps = 4;

    for (int p = 0; p < n_prompts; ++p) {
        for (int t = 0; t < n_temps; ++t) {
            printf("══════ prompt=\"%s\" temp=%.1f top_k=%d ──────\n",
                   prompts[p], temps[t], top_ks[t]);
            coa_generate(&m, &bpe, prompts[p], COA_GEN_LEN, temps[t], top_ks[t]);
            printf("\n");
        }
    }
    return 0;
}
