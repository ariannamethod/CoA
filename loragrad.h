/*
 * loragrad.h — low-rank living gradient
 *
 * lorograd is not an optimizer and not an output filter. It is an immune layer
 * between an incoming impact (sample / batch) and a weight update. A small
 * parliament of experts decides for each impact what kind of plasticity, if
 * any, that impact is allowed to produce inside the system.
 *
 * Hierarchy (top → down):
 *   origin    — primary constitution of the field (small voice corpus)
 *   boundary  — short immune oath (what may not enter the trunk)
 *   experts   — vote on plasticity mode for the current impact
 *   verdict   — PASS / WEAKEN / FREEZE / SCAR / DARK / SILENCE
 *   gradient  — routed through the verdict, then handed to the optimizer
 *
 * Phase 1 (this file): static experts initialized along the origin↔boundary
 * axis, gradient sketches via count-sketch over notorch tape grads, smoke
 * demonstrating COHERENT / VIOLATION / NOISE behavior. No training run.
 */

#ifndef LORAGRAD_H
#define LORAGRAD_H

#include <stdint.h>
#include <stddef.h>

#define LG_SIG_DIM       64       /* signature dimension R^64             */
#define LG_DEFAULT_EXPERTS 8      /* parliament size                      */
#define LG_SCAR_CAP_DEFAULT 256   /* max scar log entries                 */
#define LG_DARK_CAP_DEFAULT 256   /* max dark matter entries              */

typedef enum {
    LG_PASS    = 0,  /* gradient flows unchanged                          */
    LG_WEAKEN  = 1,  /* gradient scaled by alpha                          */
    LG_FREEZE  = 2,  /* gradient = 0 (no learning from this impact)       */
    LG_SCAR    = 3,  /* gradient = 0 + recorded as wound                  */
    LG_DARK    = 4,  /* gradient = 0 + stored as external knowledge       */
    LG_SILENCE = 5,  /* no event — dropped without trace                  */
    LG_VERDICT_COUNT
} lg_verdict_t;

const char* lg_verdict_name(lg_verdict_t v);

typedef struct {
    int   dim;                              /* = LG_SIG_DIM                       */

    float origin_sig[LG_SIG_DIM];           /* unit vector — voice                */
    float boundary_sig[LG_SIG_DIM];         /* unit vector — what we refuse       */
    int   origin_set;
    int   boundary_set;

    int   n_experts;
    float* expert_w;                        /* [n_experts, dim]                   */
    float* expert_b;                        /* [n_experts]                        */
    float* expert_credit;                   /* [n_experts] — phase 2.5 adaptive   */

    /* Verdict thresholds on (origin_score - boundary_score) ∈ [-1, +1].          */
    float thresh_pass;     /* >= → PASS                                           */
    float thresh_weaken;   /* >= and < pass → WEAKEN                              */
    float thresh_freeze;   /* >= and < weaken → FREEZE                            */
    float thresh_scar;     /* < freeze and boundary > origin → SCAR               */
    float thresh_dark;     /* < scar and boundary >> origin → DARK                */

    /* Scar log — wound shapes, no gradient                                       */
    int    scar_count;
    int    scar_cap;
    float* scar_sigs;                       /* [scar_cap, dim]                    */

    /* Dark matter — knowledge stored as external object                          */
    int    dark_count;
    int    dark_cap;
    float* dark_sigs;                       /* [dark_cap, dim]                    */

    /* Counters (for smoke reporting)                                             */
    int counters[LG_VERDICT_COUNT];
} lg_field_t;

/* ── Field lifecycle ──────────────────────────────────────────────────────── */

int  lg_field_init(lg_field_t* f, int n_experts, uint64_t seed);
void lg_field_free(lg_field_t* f);

void lg_field_set_origin_unit  (lg_field_t* f, const float* unit_vec);
void lg_field_set_boundary_unit(lg_field_t* f, const float* unit_vec);

void lg_field_set_origin_from_sketches  (lg_field_t* f, const float* sketches, int n);
void lg_field_set_boundary_from_sketches(lg_field_t* f, const float* sketches, int n);

void lg_field_calibrate_experts(lg_field_t* f, uint64_t seed);

/* ── Voting ───────────────────────────────────────────────────────────────── */

lg_verdict_t lg_field_vote(const lg_field_t* f, const float* sig, float* out_alpha);
void          lg_field_record(lg_field_t* f, lg_verdict_t v, const float* sig);

void lg_field_update_experts(lg_field_t* f, const float* sig,
                             int target_origin, float lr);

#define LG_CREDIT_CLAMP 10.0f

void lg_field_reset_counters(lg_field_t* f);
void lg_field_summary(const lg_field_t* f, const char* label);

/* ── Signatures ───────────────────────────────────────────────────────────── */

void lg_signature_from_grads(float* out_sig);
void lg_signature_from_buffer(const float* buf, int len, float* out_sig);
void lg_signature_from_text(const char* text, int len, float* out_sig);
void lg_count_sketch_hash(uint64_t key, int dim, int* out_bin, float* out_sign);

#endif /* LORAGRAD_H */
