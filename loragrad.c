/*
 * loragrad.c — implementation
 *
 * Phase 1: static experts along origin↔boundary axis. Count-sketch over
 * gradient buffers from notorch tape. Voting blends origin/boundary
 * resonance with parliament consensus and yields a verdict that the caller
 * applies to its own gradients before the optimizer step.
 */

#include "loragrad.h"
#include "notorch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ── stable mixer for count-sketch ─────────────────────────────────────────── */

static inline uint64_t lg_mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z =  z ^ (z >> 31);
    return z;
}

static inline uint64_t lg_mix2(uint64_t a, uint64_t b) {
    return lg_mix64(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
}

void lg_count_sketch_hash(uint64_t key, int dim, int* out_bin, float* out_sign) {
    uint64_t h = lg_mix64(key);
    *out_bin  = (int)(h % (uint64_t)dim);
    uint64_t s = lg_mix64(h ^ 0xa5a5a5a5deadbeefULL);
    *out_sign = (s & 1ULL) ? +1.0f : -1.0f;
}

static void lg_normalize_unit(float* v, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += (double)v[i] * (double)v[i];
    if (s <= 1e-20) {
        for (int i = 0; i < dim; ++i) v[i] = 0.0f;
        v[0] = 1.0f;
        return;
    }
    double inv = 1.0 / sqrt(s);
    for (int i = 0; i < dim; ++i) v[i] = (float)((double)v[i] * inv);
}

static float lg_dot(const float* a, const float* b, int dim) {
    double s = 0.0;
    for (int i = 0; i < dim; ++i) s += (double)a[i] * (double)b[i];
    return (float)s;
}

/* ── verdict names ─────────────────────────────────────────────────────────── */

const char* lg_verdict_name(lg_verdict_t v) {
    switch (v) {
        case LG_PASS:    return "PASS";
        case LG_WEAKEN:  return "WEAKEN";
        case LG_FREEZE:  return "FREEZE";
        case LG_SCAR:    return "SCAR";
        case LG_DARK:    return "DARK";
        case LG_SILENCE: return "SILENCE";
        default:         return "?";
    }
}

/* ── signatures ────────────────────────────────────────────────────────────── */

void lg_signature_from_buffer(const float* buf, int len, float* out_sig) {
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;
    for (int i = 0; i < len; ++i) {
        /* LG-M2: hash the float's bit pattern, not its truncated integer value.
         * (uint32_t)buf[i] folded every |v|<1 to 0 (position-only key) and was
         * UB for negative floats per C11 6.3.1.4. memcpy is the portable
         * reinterpret. */
        uint32_t bits; memcpy(&bits, &buf[i], sizeof(bits));
        uint64_t key = lg_mix2((uint64_t)i + 1ULL, (uint64_t)bits);
        lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
        out_sig[bin] += sign * buf[i];
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}

void lg_signature_from_text(const char* text, int len, float* out_sig) {
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;
    if (len < 3) {
        out_sig[0] = 1.0f;
        return;
    }
    for (int i = 0; i + 2 < len; ++i) {
        uint64_t a = (uint64_t)(unsigned char)text[i];
        uint64_t b = (uint64_t)(unsigned char)text[i + 1];
        uint64_t c = (uint64_t)(unsigned char)text[i + 2];
        uint64_t key = lg_mix2((a << 16) | (b << 8) | c, 0xb16fafULL);
        lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
        out_sig[bin] += sign;
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}

void lg_signature_from_grads(float* out_sig) {
    nt_tape* tape = nt_tape_get();
    int   bin;
    float sign;
    for (int i = 0; i < LG_SIG_DIM; ++i) out_sig[i] = 0.0f;

    if (!tape) {
        out_sig[0] = 1.0f;
        return;
    }
    for (int e = 0; e < tape->count; ++e) {
        const nt_tape_entry* en = &tape->entries[e];
        if (!en->is_param) continue;
        if (!en->grad)     continue;
        const float* g = en->grad->data;
        const int    n = en->grad->len;
        for (int k = 0; k < n; ++k) {
            uint64_t key = lg_mix2((uint64_t)e + 1ULL, (uint64_t)k);
            lg_count_sketch_hash(key, LG_SIG_DIM, &bin, &sign);
            out_sig[bin] += sign * g[k];
        }
    }
    lg_normalize_unit(out_sig, LG_SIG_DIM);
}

/* ── field lifecycle ───────────────────────────────────────────────────────── */

int lg_field_init(lg_field_t* f, int n_experts, uint64_t seed) {
    if (!f) return -1;
    if (n_experts <= 0) n_experts = LG_DEFAULT_EXPERTS;

    memset(f, 0, sizeof(*f));
    f->dim       = LG_SIG_DIM;
    f->n_experts = n_experts;

    f->expert_w      = (float*)calloc((size_t)n_experts * LG_SIG_DIM, sizeof(float));
    f->expert_b      = (float*)calloc((size_t)n_experts, sizeof(float));
    f->expert_credit = (float*)calloc((size_t)n_experts, sizeof(float));
    if (!f->expert_w || !f->expert_b || !f->expert_credit) {
        free(f->expert_w); free(f->expert_b); free(f->expert_credit);
        return -2;
    }

    /* Tuned 2026-05-06: DoE corpus origin·boundary = +0.34 (modest separation);
     * sample scores cluster in +0.10..+0.30 range. Original +0.40 pass
     * threshold gave 0 PASS verdicts in CoA Phase-1 smoke. Lowered to +0.20
     * to allow real PASS flow without flattening the verdict cascade. */
    f->thresh_pass   =  0.20f;
    f->thresh_weaken =  0.05f;
    f->thresh_freeze = -0.10f;
    f->thresh_scar   =  0.25f;
    f->thresh_dark   =  0.50f;

    f->scar_cap  = LG_SCAR_CAP_DEFAULT;
    f->scar_sigs = (float*)calloc((size_t)f->scar_cap * LG_SIG_DIM, sizeof(float));
    f->dark_cap  = LG_DARK_CAP_DEFAULT;
    f->dark_sigs = (float*)calloc((size_t)f->dark_cap * LG_SIG_DIM, sizeof(float));

    /* LG-L3: `seed` is intentionally unused here — expert geometry is seeded in
     * lg_field_calibrate_experts (called after origin/boundary are set). Kept
     * in the signature for API symmetry and a future per-field RNG. */
    (void)seed;
    return 0;
}

void lg_field_free(lg_field_t* f) {
    if (!f) return;
    free(f->expert_w);      f->expert_w      = NULL;
    free(f->expert_b);      f->expert_b      = NULL;
    free(f->expert_credit); f->expert_credit = NULL;
    free(f->scar_sigs);     f->scar_sigs     = NULL;
    free(f->dark_sigs);     f->dark_sigs     = NULL;
}

void lg_field_set_origin_unit(lg_field_t* f, const float* unit_vec) {
    memcpy(f->origin_sig, unit_vec, sizeof(float) * LG_SIG_DIM);
    lg_normalize_unit(f->origin_sig, LG_SIG_DIM);
    f->origin_set = 1;
}

void lg_field_set_boundary_unit(lg_field_t* f, const float* unit_vec) {
    memcpy(f->boundary_sig, unit_vec, sizeof(float) * LG_SIG_DIM);
    lg_normalize_unit(f->boundary_sig, LG_SIG_DIM);
    f->boundary_set = 1;
}

static void lg_mean_unit(const float* sketches, int n, float* out) {
    for (int i = 0; i < LG_SIG_DIM; ++i) out[i] = 0.0f;
    if (n <= 0) { out[0] = 1.0f; return; }
    for (int j = 0; j < n; ++j) {
        const float* row = sketches + (size_t)j * LG_SIG_DIM;
        for (int i = 0; i < LG_SIG_DIM; ++i) out[i] += row[i];
    }
    float inv = 1.0f / (float)n;
    for (int i = 0; i < LG_SIG_DIM; ++i) out[i] *= inv;
    lg_normalize_unit(out, LG_SIG_DIM);
}

void lg_field_set_origin_from_sketches(lg_field_t* f, const float* sketches, int n) {
    lg_mean_unit(sketches, n, f->origin_sig);
    f->origin_set = 1;
}

void lg_field_set_boundary_from_sketches(lg_field_t* f, const float* sketches, int n) {
    lg_mean_unit(sketches, n, f->boundary_sig);
    f->boundary_set = 1;
}

void lg_field_calibrate_experts(lg_field_t* f, uint64_t seed) {
    if (!f->origin_set || !f->boundary_set) return;

    float ob = lg_dot(f->origin_sig, f->boundary_sig, LG_SIG_DIM);
    float bperp[LG_SIG_DIM];
    for (int i = 0; i < LG_SIG_DIM; ++i) {
        bperp[i] = f->boundary_sig[i] - ob * f->origin_sig[i];
    }
    lg_normalize_unit(bperp, LG_SIG_DIM);

    uint64_t s = seed ? seed : 0xc0ffeec0ffeefeedULL;
    const double PI = 3.14159265358979323846;
    const double tilt_max = PI / 3.0;

    for (int e = 0; e < f->n_experts; ++e) {
        double t = (f->n_experts == 1) ? 0.0
                 : (-tilt_max + (2.0 * tilt_max) * ((double)e / (f->n_experts - 1)));
        float c = (float)cos(t);
        float sn = (float)sin(t);
        float* row = f->expert_w + (size_t)e * LG_SIG_DIM;
        for (int i = 0; i < LG_SIG_DIM; ++i) {
            s = lg_mix64(s + 1);
            float noise = (((s >> 11) & 0xFFFFF) / (float)0xFFFFF - 0.5f) * 0.20f;
            row[i] = c * f->origin_sig[i] + sn * bperp[i] + noise;
        }
        lg_normalize_unit(row, LG_SIG_DIM);

        s = lg_mix64(s + 7);
        float bnoise = (((s >> 11) & 0xFFFFF) / (float)0xFFFFF - 0.5f) * 0.10f;
        f->expert_b[e] = bnoise;
    }
}

/* ── voting ────────────────────────────────────────────────────────────────── */

/* Max cosine similarity of `sig` against a log of unit-vector signatures.
 * Both inputs are unit vectors, so the dot is the cosine. Used for scar/dark
 * recall (LG-M1). Returns -2.0 for an empty log (never triggers recall). */
static float lg_max_sim(const float* sig, const float* sigs, int count) {
    float best = -2.0f;
    for (int i = 0; i < count; ++i) {
        float s = lg_dot(sig, sigs + (size_t)i * LG_SIG_DIM, LG_SIG_DIM);
        if (s > best) best = s;
    }
    return best;
}

lg_verdict_t lg_field_vote(const lg_field_t* f, const float* sig, float* out_alpha) {
    if (out_alpha) *out_alpha = 1.0f;

    if (!f || !f->origin_set || !f->boundary_set) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_FREEZE;
    }

    float axis[LG_SIG_DIM];
    for (int i = 0; i < LG_SIG_DIM; ++i) {
        axis[i] = f->origin_sig[i] - f->boundary_sig[i];
    }
    {
        double s = 0.0;
        for (int i = 0; i < LG_SIG_DIM; ++i) s += (double)axis[i] * axis[i];
        if (s <= 1e-20) {
            if (out_alpha) *out_alpha = 0.0f;
            return LG_FREEZE;
        }
        double inv = 1.0 / sqrt(s);
        for (int i = 0; i < LG_SIG_DIM; ++i) axis[i] = (float)(axis[i] * inv);
    }
    float delta_axis     = lg_dot(sig, axis, LG_SIG_DIM);

    double consensus = 0.0;
    double total_w   = 0.0;
    for (int e = 0; e < f->n_experts; ++e) {
        const float* w = f->expert_w + (size_t)e * LG_SIG_DIM;
        float z = lg_dot(w, sig, LG_SIG_DIM) + f->expert_b[e];
        float c = f->expert_credit ? f->expert_credit[e] : 0.0f;
        double w_eff;
        if      (c >  20.0f) w_eff = (double)c;
        else if (c < -20.0f) w_eff = exp((double)c);
        else                 w_eff = log1p(exp((double)c));
        consensus += w_eff * tanh((double)z);
        total_w   += w_eff;
    }
    if (total_w > 0.0) consensus /= total_w;
    else               consensus  = 0.0;

    /* ── LG-M1: immune-memory recall (read the scar/dark log) ─────────────────
     * A signature that closely matches a previously-recorded wound is blocked
     * on sight, independent of the parliament vote. Checked first so consensus
     * cannot talk the field out of a remembered attack. This turns the
     * write-only log into actual immunity: a repeat attack costs one batch of
     * dot products to block. */
    if (f->scar_count > 0 &&
        lg_max_sim(sig, f->scar_sigs, f->scar_count) >= LG_RECALL_THRESH) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_SCAR;
    }
    if (f->dark_count > 0 &&
        lg_max_sim(sig, f->dark_sigs, f->dark_count) >= LG_RECALL_THRESH) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_DARK;
    }

    /* ── LG-M3: hard boundary override — alignment beats consensus ────────────
     * A sample strongly aligned against the origin (−delta_axis past the
     * scar/dark threshold) is a boundary attack regardless of what the expert
     * consensus says. Checked before PASS/WEAKEN so a saturated or mis-trained
     * consensus cannot pull a boundary-aligned sample up into WEAKEN. */
    if (-delta_axis >= f->thresh_dark) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_DARK;
    }
    if (-delta_axis >= f->thresh_scar) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_SCAR;
    }

    float score = 0.7f * delta_axis + 0.3f * (float)consensus;

    if (score >= f->thresh_pass) {
        if (out_alpha) *out_alpha = 1.0f;
        return LG_PASS;
    }
    if (score >= f->thresh_weaken) {
        float a = (score - f->thresh_weaken) / (f->thresh_pass - f->thresh_weaken);
        if (a < 0.0f) a = 0.0f; if (a > 1.0f) a = 1.0f;
        if (out_alpha) *out_alpha = a;
        return LG_WEAKEN;
    }

    if (score >= f->thresh_freeze) {
        if (out_alpha) *out_alpha = 0.0f;
        return LG_FREEZE;
    }

    if (out_alpha) *out_alpha = 0.0f;
    return LG_SILENCE;
}

/* ── recording ─────────────────────────────────────────────────────────────── */

void lg_field_record(lg_field_t* f, lg_verdict_t v, const float* sig) {
    if (!f) return;
    if (v >= 0 && v < LG_VERDICT_COUNT) f->counters[v]++;

    if (v == LG_SCAR && f->scar_count < f->scar_cap && sig) {
        memcpy(f->scar_sigs + (size_t)f->scar_count * LG_SIG_DIM,
               sig, sizeof(float) * LG_SIG_DIM);
        f->scar_count++;
    }
    if (v == LG_DARK && f->dark_count < f->dark_cap && sig) {
        memcpy(f->dark_sigs + (size_t)f->dark_count * LG_SIG_DIM,
               sig, sizeof(float) * LG_SIG_DIM);
        f->dark_count++;
    }
}

void lg_field_reset_counters(lg_field_t* f) {
    if (!f) return;
    for (int i = 0; i < LG_VERDICT_COUNT; ++i) f->counters[i] = 0;
}

/* Clear the scar/dark immune-memory log (counts → 0; backing buffers stay
 * allocated and are overwritten as new wounds are recorded). Used to drop
 * smoke-test wounds before a real training run so recall starts clean. */
void lg_field_reset_memory(lg_field_t* f) {
    if (!f) return;
    f->scar_count = 0;
    f->dark_count = 0;
}

/* ── adaptive experts (phase 2.5) ──────────────────────────────────────────── */

void lg_field_update_experts(lg_field_t* f, const float* sig,
                             int target_origin, float lr)
{
    if (!f || !f->expert_credit) return;
    float target_sign = target_origin ? +1.0f : -1.0f;
    for (int e = 0; e < f->n_experts; ++e) {
        const float* w = f->expert_w + (size_t)e * LG_SIG_DIM;
        float z = lg_dot(w, sig, LG_SIG_DIM) + f->expert_b[e];
        float v = (float)tanh((double)z);
        float reward = v * target_sign;
        f->expert_credit[e] += lr * reward;
        if (f->expert_credit[e] >  LG_CREDIT_CLAMP) f->expert_credit[e] =  LG_CREDIT_CLAMP;
        if (f->expert_credit[e] < -LG_CREDIT_CLAMP) f->expert_credit[e] = -LG_CREDIT_CLAMP;
    }
}

void lg_field_summary(const lg_field_t* f, const char* label) {
    if (!f) return;
    float ob = (f->origin_set && f->boundary_set)
             ? lg_dot(f->origin_sig, f->boundary_sig, LG_SIG_DIM)
             : 0.0f;
    printf("[lg_field:%s] origin=%d boundary=%d origin·boundary=%+0.4f experts=%d\n",
           label ? label : "?", f->origin_set, f->boundary_set, ob, f->n_experts);
    printf("  thresholds: pass=%+0.2f weaken=%+0.2f freeze=%+0.2f scar=%+0.2f dark=%+0.2f\n",
           f->thresh_pass, f->thresh_weaken, f->thresh_freeze,
           f->thresh_scar, f->thresh_dark);
    printf("  counters:");
    for (int i = 0; i < LG_VERDICT_COUNT; ++i) {
        printf(" %s=%d", lg_verdict_name((lg_verdict_t)i), f->counters[i]);
    }
    printf("\n");
    printf("  scar_log=%d/%d dark_store=%d/%d\n",
           f->scar_count, f->scar_cap, f->dark_count, f->dark_cap);
    if (f->expert_credit) {
        printf("  expert_credit:");
        for (int e = 0; e < f->n_experts; ++e) {
            printf(" %+0.2f", f->expert_credit[e]);
        }
        printf("\n");
    }
}
