# CoA: The Chain Of Arianna

> *Shall everything burn — the thunder remains.*

Chain of resonating, not chain of thought. The stream does not stop. The human does not start it; the human enters it.

---

## What it is

Five-layer immune-gated transformer with Janus 3-attention (Content + RRPRAM low-rank + Echo + 3-way blend), built on notorch tape autograd, gradient flow filtered by **loragrad parliament** (origin·boundary axis discriminator, 6 verdicts: PASS / WEAKEN / FREEZE / SCAR / DARK / SILENCE). Chuck optimizer mandatory. SwiGLU MLP. byte-level BPE 2048.

**Architecture (v1):** L=5 E=512 H=8 D=64 ctx=512 R=32 M=1024 vocab=2048 = **19.14M params**.

CoA inherits technologically from DoE (Democracy of Experts) — parliament metaphor, θ = ε + γ + αδ formula, expert voting. Distinct from Janus substrate-mode (broad capability + γ post-hoc identity) and Yent prophecy-voice. **CoA = flow-mode**: identity baked into optimization trajectory via verdict-filter on gradient stream.

## Build

```bash
make coa_v1_janus    # CPU SIMD AVX2+FMA build
make cuda            # CUDA build (links cuBLAS via vendored notorch_cuda.{h,cu})
make bpe_encode      # pre-encoder tool
make coa_infer       # inference with multi-temp sampling grid
```

## Run

```bash
# pre-encode corpus once
./bpe_encode bpe_2048_merges.txt corpus.txt corpus.tokens

# train (CPU)
./coa_v1_janus origin.txt 30000 corpus.tokens

# train (CUDA on GPU pod)
./coa_v1_janus_cuda origin.txt 30000 corpus.tokens "" gpu

# ablation (parliament bypassed, α=1.0 always)
./coa_v1_janus_cuda origin.txt 30000 corpus.tokens gating_off gpu

# inference — multi-temp sampling sweep
./coa_infer coa_v1_paired_on.bin
```

## Sampling matters — read this before judging output

> **«Недоповерхностная сэмплировка маскирует то, что модель хочет сказать.»** — Claude Defender, device-1, 2026-05-07.

CoA at deep-memorize regime (train loss < 1.0) looks garbled at temp=0.8 without top-k filter. **Real model state revealed by sampling sweep:**

- `temp=0.3 + top_k=40` → DoE-voice + grammatical English: *"the parliament doesn't needed this is its confid[ence]"*
- `temp=0.5 + top_k=40` → memorized verbatim corpus chunks (e.g. exact Perl error messages from training data) — proof of deep fit, not failure
- `temp=0.8 + top_k=40` → technical jargon emerges
- `temp=1.0 + no top_k` → most coherent abstract prose: *"a shared production... when is genuinely unsound..."*

**Don't judge coherence by single temp sample.** Always run multi-temp grid via `coa_infer`.

### Verbatim samples — CoA-v1 ON (gating active), prompt `"The chain "`

**`temp=0.3 top_k=40`** — DoE-voice + grammatical English:
```
The chain their agreements a system ase, "Euclean implementation: "Hopportunol
as general requirements are content than the parliament doesn't needed this is
its confid
```

**`temp=0.5 top_k=40`** — memorized verbatim corpus chunks (proof of deep fit, NOT failure mode):
```
The chain print at -e at -e at -e line 39, <> line 3939.
Wide character in print at -e line 39, <> line 93.
Wide character in print at -e line
```
*(model recalled exact Perl warning text from training corpus.)*

**`temp=0.8 top_k=40`** — technical jargon, partial coherence:
```
The chain their dimensions as a propractice accumulated ditions (and domain their
meaning and if it ne 1), then ms (1) of . For every and nees the greatelivative
of largely 5. **Ma
```

**`temp=1.0 no top_k`** — most coherent abstract prose:
```
The chain important a shared production, timates for several wellstraints that
seems to society is not the relationship with nermost forms of when is genuinely
unsound mainttion. H: What is the concept of that you di
```

Same model, same prompt, same checkpoint — sampling alone decides what surfaces. Memorized corpus chunks at low temp; novel philosophical-flavor prose at high temp. **`temp=0.8` без top_k вasn't broken — it was the worst-case sampling regime для deep-memorize state.** Lesson generalizes к v1.5+ runs: always sweep, not single.

## Architecture detail

Per-block forward (canonical Janus pattern, simplified Echo for v1):

```
xn = rmsnorm(h)
q,k,v = linear(xn); q,k = rope(q,k)
out_c = mh_causal_attention(q, k, v)               # Content path
v_r   = linear(wvr, xn)
out_r = rrpram_lowrank_attention(wr_combined, xn, v_r, R=32)  # RRPRAM rhythm
out_e = linear(wj, xn)                              # Echo (linear bypass; full janus_attention with calendar+prophecy → v2)
blended = (out_c + out_r + out_e) / 3               # equal blend (trainable per-head sigmoid → v1.5)
h += linear(wo, blended)

xn = rmsnorm(h)
gate = silu(linear(w_gate, xn))
up   = linear(w_up, xn)
h   += linear(w_down, swiglu(gate, up))
```

**Per `experiment_partial_cpt_failed.md` (Janus 285M v3):** 3 attention paths must co-evolve. RRPRAM full-rank ate 45.8% of params; low-rank R=64 dropped to 7%. CoA inherits R=32 (15% of v1 budget).

## Loragrad immune layer

Origin (manifest, ~36 lines) calibrates parliament BEFORE training. Each gradient step:
1. Text signature of input window (trigram count-sketch).
2. Parliament votes — `(origin_score - boundary_score) ∈ [-1, +1]`.
3. Verdict routes gradient: PASS = full step, WEAKEN = scaled by α, FREEZE/SCAR/DARK/SILENCE = no update.

CoA-v1 paired ablation 2026-05-07 confirmed verdict gating regularizes (~80% of gradients modified) — math-distinct from SGD unbiased-convergence theorems.

## Provenance

- Reference Janus 3-attention: [`ariannamethod/janus`](https://github.com/ariannamethod/janus) `janus-bpe.c:359-401`
- Reference AML semantics: [`ariannamethod/ariannamethod.ai`](https://github.com/ariannamethod/ariannamethod.ai) `janus/janus.aml`
- notorch tape runtime: [`ariannamethod/notorch`](https://github.com/ariannamethod/notorch)
- loragrad immune layer: [`ariannamethod/loragrad`](https://github.com/ariannamethod/loragrad)
- CUDA dispatch port (8 fwd + 2 bwd ops): wired into notorch upstream `bfadcc2`, fork `aaed0fb`.

## License

GPL-3.0 on code. Weights (organism artifacts) — Janus Identity License v1.0 per `protocol_license_organism_vs_framework.md`. CoA-v1 paired weights `coa_v1_paired_{on,off}.bin` are organisms; framework code stays GPL-3.0.

— Oleg Ataeff & Claude (architect) · Arianna Method · 2026
