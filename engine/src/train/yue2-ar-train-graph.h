#pragma once
// train/yue2-ar-train-graph.h — trainable twin of the YuE2 AR (causal) half.
//
// HOT-Step file. TRAINING-SIDE: built into ace-train only, never into
// ace-server. Implements docs/plans/yue2/14-ar-lora-contract.md, which is the
// authority for every shape, op and constant below and cites the file:line each
// one was read out of. Where a comment says "contract §N", that is the section
// to re-read before changing anything.
//
// Upstream recipe being ported (the RECIPE, not the code):
// Mothersuperior's `scripts/ar_lora_cursor.py` + `ar_prep.py` from
// `yue2-mothersuperior-realaudio-tokenizer-v4`. Template for the C++/GGML
// posture: train/yue2-nar-train-graph.h (the NAR twin, shipped the same week)
// and train/lm-ckpt.h (per-layer checkpointing + the chunked CE head).
//
// ── What this file is, in one sentence ─────────────────────────────────────
//
// `yue2_at_block` is `yue2_ar_block` (yue2-lm-graph.h:136-199) with three
// changes and nothing else: LoRA sites at the seven projections, NO KV cache
// (K/V go straight into attention), and a swap of `ggml_flash_attn_ext` for
// either the manual soft_max chain or HOT-Step's fused training op. Every other
// op, both norms, the RoPE call and the SwiGLU spelling are copied verbatim,
// because a trainer that computes a subtly different forward trains a model
// that is not the one we ship.
//
// ── Trap 1: SET_ROWS has no backward, so the KV cache goes ──────────────────
//
// `yue2_ar_forward`'s KV cache is a PURE ROUND TRIP INSIDE ONE GRAPH: it is
// allocated at exactly T rows, written by that same call's `ggml_set_rows`, and
// read back through a view two nodes later (yue2-lm-graph.h:61-73 says so in as
// many words). `GGML_OP_SET_ROWS` is absent from `ggml_compute_backward`'s
// switch, so it cannot appear upstream of a loss — and removing it and feeding
// `k_w`/`v_w` straight into attention computes the identical mathematical
// result (contract §1.2).
//
// TWO CONSEQUENCES, both worth saying out loud:
//
//   1. The trainer's forward is NOT bit-identical to `yue2_ar_forward`. That
//      cache is F16 (yue2-lm-graph.h:265-266), so inference rounds K and V to
//      half precision between the projection and attention; we keep them F32.
//      Strictly MORE precise than the thing we validate against, never less —
//      but anyone writing a parity check must expect an F16-sized delta, not
//      zero (contract §6.6).
//   2. Nothing here touches `Yue2ArKvCache` / `yue2_ar_prefill` /
//      `yue2_ar_decode_step`. Those are the generation loop.
//
// ── Trap 2: the V gradient ABORTS the process, and the NAR fix applies ──────
//
// The AR analogue of yue2-nar-train-graph.h's trap 1b, with a different
// symptom. In the EXACT path `yue2_lm_attn_f32` consumes V as
// `ggml_cont(ggml_transpose(v))` (yue2-lm-graph.h:127) and we hand it
// `v_w = ggml_cont(ggml_permute(v, 0,2,1,3))`. Follow the gradient:
//
//   grads[vt]             <- MUL_MAT src0 arm -> out_prod/repeat_back -> FRESH
//   CONT(vt) backward     -> asserts grad contiguous (ggml.c:7202): passes
//   TRANSPOSE backward    -> stores ggml_transpose(grad) VERBATIM (ggml.c:7257)
//   grads[v_w]            = a NON-CONTIGUOUS view
//   CONT(v_w) backward    -> GGML_ASSERT(ggml_is_contiguous(grad))  <-- ABORT
//
// The NAR trainer's version of this was silent wrong numbers (its consumer was
// `ggml_set`, whose backward slices with the DESTINATION's strides). Ours is a
// hard assert inside `ggml_compute_backward` — the better failure of the two,
// and still one a first run hits.
//
// FIX, same as the NAR trainer's: wrap `k_w` and `v_w` each in an identity
// `ggml_reshape_4d`. RESHAPE's backward does
// `grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad)`
// (ggml.c:7209-7212), which normalises the transposed view before CONT ever
// sees it, and costs nothing on the K side where the gradient is already
// contiguous. BOTH sides get the wrapper deliberately: the K/V asymmetry is an
// accident of which ops attention happens to apply, and anyone editing
// `yue2_lm_attn_f32` could flip it. A plain `ggml_cont` after the fact is NOT a
// substitute — CONT's backward asserts the grad is already contiguous.
//
// In the FLASH path the problem does not arise and the right move is
// lm-graph.h:1471-1478 exactly: drop the `ggml_cont` on q/k/v and pass the
// permuted views. The fused op reads them through nb[1..3] on purpose (only
// nb[0] == 4 is required); materialising them hands back part of the saving.
//
// ── Trap 3: the token axis must never reach ne2 of a trainable mul_mat ──────
//
// mm3-dit-train-graph.h:37-44's rule — ggml then emits out_prod PER TOKEN.
// Every projection here consumes `n`, a 2-D [H, S], so ne2 == 1 at all seven
// sites; the `reshape_4d` to [D, Nh, S, 1] happens AFTER the LoRA add.
// DO NOT HOIST THE RESHAPE ABOVE THE LoRA ADD.
//
// ── Trap 4: RMSNorm end to end, which is the easy half ──────────────────────
//
// `GGML_OP_NORM` has no backward; `GGML_OP_RMS_NORM` does (ggml.c:7012). YuE2
// is RMS on both halves, so no norm surgery is needed at all — contrast MM3's
// LayerNorm rebuild-from-primitives (mm3-dit-train-graph.h:106-152).
//
// ── Ops for the record (contract §1.7) ─────────────────────────────────────
//
//   GGML_OP_SET_ROWS         NO   absent from ggml_compute_backward
//   GGML_OP_CONCAT           NO   absent (not on this path anyway)
//   GGML_OP_NORM             NO   absent (not used — YuE2 is RMS)
//   GGML_OP_FLASH_ATTN_EXT   NO   absent
//   GGML_OP_FLASH_ATTN_TRAIN yes  ggml.c:7424, HOT-Step patch
//   GGML_OP_RMS_NORM         yes  ggml.c:7012
//   GGML_OP_ROPE             yes  ggml.c:7296  (positions non-differentiable)
//   GGML_OP_SOFT_MAX         yes  ggml.c:7284, max_bias == 0 only on CUDA
//   GGML_UNARY_OP_SILU       yes  ggml.c:7374
//   GGML_OP_CROSS_ENTROPY_*  yes  ggml.c:6392-6422
//
// `ggml_swiglu_split` would save one [F, S] retained tensor per layer, and is
// DELIBERATELY NOT TAKEN in v1 (contract §1.7): the governing principle is
// mirror `yue2_ar_block:196-198` op for op, and under per-layer checkpointing
// only one layer is live so the saving is ~240 MB, not the 6.7 GB it would be
// monolithically. The seam is one line.
//
// ── Why everything below is F32-weight by default ──────────────────────────
//
// lm-ckpt.h:7-13: `ggml_build_backward_expand` computes the ACTIVATION gradient
// of `mul_mat(W, x)` as `ggml_out_prod(W, transpose(grad))`, and `out_prod` is
// F32-only on CUDA and GGML_ABORTs for BF16 on CPU. So every frozen projection
// on the gradient path must be F32 WHILE ITS GRAPH RUNS. `yue2_at_w` is the one
// place that cast happens, and `Yue2AtOpts::weights_f32` is the one switch —
// which is what makes D13 ("the recomputed forward must use the SAME options as
// the collect pass") enforceable rather than a convention. Skipping the cast in
// the collect pass is faster and silently turns checkpointing from an identity
// into a ~1e-3 approximation.
//
// One F32 layer window is 50.33M * 4 B = 192 MiB. Twenty-eight of them would be
// 5.4 GiB, which is the entire reason the checkpoint driver exists.
//
// ── What is NOT here ───────────────────────────────────────────────────────
//
// - No adapted `token_embd` / `output` (contract §2.3). The chunked-CE head
//   below deliberately bypasses autodiff at the output — it computes dL/dh by
//   hand and copies it into the trunk's gradient buffer — so a trainable head
//   would need its own gradient path back through that hand-built seam. That is
//   a different and considerably more delicate design; upstream does not do it;
//   and merging a delta into `token_embd` on a quantized base means requanting
//   the one tensor with a cross-tensor type invariant.
// - No `proj` tier. `vae2llm` / `llm2vae` / `time_embd.{0,1}` belong to the flow
//   head and have no part in the AR forward. An AR adapter that names them is
//   malformed.
// - No lyric-cursor auxiliary loss (contract §3.4). DEFERRED: its targets come
//   from Demucs vocal separation + torchaudio MMS forced alignment, neither of
//   which exists in this engine, and the manifest carries no lyrics string for
//   the char offsets to point into. The seam is named in the runner
//   (`Yue2ArTrainArgs::cursor_weight`, refused above 0).
// - No export, no CLI. Those are the runner's.
//
// ── Risk noted, not resolved: BF16 base weights in the backward ────────────
//
// Same flag the NAR file raises (yue2-nar-train-graph.h:160-172). With
// `weights_f32` on (the default) the frozen arm never sees BF16, so this is
// about `--weights native` only. Kept as a flag rather than deleted because the
// cast costs a kernel per weight per graph and someone will want to measure it.

#include "train/flash-prec.h"            // dit_flash_probe, dit_flash_prec_label
#include "train/lm-optim.h"              // LmOptim: param_slot / acc, read by yue2_at_fill_gacc
#include "train/yue2-nar-train-graph.h"  // Yue2TrainLora, Yue2NtRng, yue2_nt_init_adapters (contract §2.4)

#include "backend.h"
#include "yue2/yue2-lm-graph.h"   // yue2_lm_attn_f32
#include "yue2/yue2-model.h"      // Yue2Model, Yue2LmConfig, Yue2LmLayer
#include "yue2/yue2-tokenizer.h"  // YUE2_CODEC_OFFSET, YUE2_MUSIC_END, YUE2_LATENT_START

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // getenv/atoi — YUE2_AT_SUP_SHIFT, the supervised-slice control
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// The LoRA factor struct, the splitmix64 RNG and the kaiming/zero init are
// REUSED VERBATIM from the NAR trainer rather than re-declared (contract §2.4:
// "Yue2NtRng is reused verbatim ... Do not swap it"). Two copies of a seeded
// generator is exactly how "--seed 42 reproduces it" stops being true.
using Yue2AtLora = Yue2TrainLora;

// ── The output-head slice (contract §3.3) ──────────────────────────────────
//
// MUSIC_END plus all 32,768 codec tokens, stopping exactly before
// LATENT_START, which is training-time-only and never in a live token stream.
//
//   slice row 0            = MUSIC_END (151852)
//   slice rows             = LATENT_START - MUSIC_END = 184621 - 151852 = 32769
//   token id -> slice row  = id - MUSIC_END          (MUSIC_END -> 0, c -> c+1)
//
// THIS IS MATCHED TO INFERENCE, NOT AN APPROXIMATION FOR SPEED. The semantic
// stage masks logits to [CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE) with
// MUSIC_END whitelisted (yue2-pipeline.h:289-290, 351; yue2-sample.h:150-152),
// so a softmax over this slice IS the distribution the sampler draws from.
//
// It is also a NAMED DIVERGENCE from upstream, which takes the full 184,704-way
// softmax (ar_lora_cursor.py:62). Our CE omits the text-token mass, so our
// numbers are NOT comparable to the README's — which matters, because
// `minted_val` loss is the thing to watch. The runner logs a full-vocab CE at
// every eval for exactly that reason.
static const int64_t YUE2_AT_SLICE_ROW0 = (int64_t) YUE2_MUSIC_END;                        // 151852
static const int64_t YUE2_AT_SLICE_ROWS = (int64_t) YUE2_LATENT_START - (int64_t) YUE2_MUSIC_END;  // 32769

// ── YUE2_AT_SUP_SHIFT: the supervised-slice negative control (contract §6.5) ─
//
// Finite differences verify the BACKWARD against the FORWARD, so a forward that
// is off by one row passes the FD gate happily: both arms move together. The
// off-by-one in contract §3.1 — the hidden state at row r predicts ids[r+1], so
// the supervised span starts at row P-1 and NOT at P — is the mistake in this
// design most likely to be made and least likely to be noticed, because a loss
// that falls from a wrong starting point still falls.
//
// This env var shifts the column the head reads (and writes dL/dh back into) by
// N rows without touching the targets, in the same spirit as YUE2_FD_LOSSGRAD.
// -1 is contract §6.5's control: h[P-2 : S-2] against targets ids[P:]. The loss
// must then jump well clear of ln(32769) = 10.397 and stay there. It is read
// once, banners once, and a real run never sets it.
static int yue2_at_sup_shift_env() {
    static int  shift  = 0;
    static bool loaded = false;
    if (!loaded) {
        loaded          = true;
        const char * ev = std::getenv("YUE2_AT_SUP_SHIFT");
        if (ev && ev[0]) {
            shift = atoi(ev);
            if (shift != 0) {
                fprintf(stderr,
                        "[yue2-at] NEGATIVE CONTROL: YUE2_AT_SUP_SHIFT=%d — the CE head reads hidden rows "
                        "[P-1%+d .. ] against UNSHIFTED targets ids[P..]. The supervision window is "
                        "deliberately wrong; a loss that does not move is the finding.\n",
                        shift, shift);
            }
        }
    }
    return shift;
}

// ── Attention mode (contract §1.3) ─────────────────────────────────────────
//
// `ggml_flash_attn_ext` has no backward, so the inference branch at
// yue2-lm-graph.h:185 cannot appear here. The choice between the two
// replacements is not cosmetic: exact retains an [S, S, Nh] F32 softmax per
// layer, which at S = 10,001 is 6.0 GiB.
enum Yue2AtAttnMode {
    YUE2_AT_FA_EXACT     = 0,  // yue2_lm_attn_f32: mul_mat -> soft_max_ext -> mul_mat
    YUE2_AT_FA_FLASH     = 1,  // GGML_OP_FLASH_ATTN_TRAIN, backend-default precision (TF32 on sm_80+)
    YUE2_AT_FA_FLASH_F32 = 2,  // the same fused op pinned to strict f32
};

static const char * yue2_at_attn_name(Yue2AtAttnMode m) {
    switch (m) {
        case YUE2_AT_FA_FLASH:
            return "flash";
        case YUE2_AT_FA_FLASH_F32:
            return "flash-f32";
        case YUE2_AT_FA_EXACT:
        default:
            return "exact";
    }
}

static bool yue2_at_parse_attn(const std::string & s, Yue2AtAttnMode * out) {
    if (s.empty() || s == "exact") {
        *out = YUE2_AT_FA_EXACT;
    } else if (s == "flash") {
        *out = YUE2_AT_FA_FLASH;
    } else if (s == "flash-f32" || s == "flash_f32") {
        *out = YUE2_AT_FA_FLASH_F32;
    } else {
        return false;
    }
    return true;
}

// Per-graph options. ONE struct, read once per micro-step and passed to every
// pass, so P2 (forward collect), P3 (tail) and P7 (backward segment) cannot
// disagree — which is what lm-ckpt.h's D13 depends on.
struct Yue2AtOpts {
    Yue2AtAttnMode attn        = YUE2_AT_FA_EXACT;
    bool           weights_f32 = true;  // see the header note on out_prod
};

// The one place a frozen weight is widened. A no-op for an F32 tensor, so
// `--weights native` on an F32 base emits the identical graph.
static ggml_tensor * yue2_at_w(ggml_context * ctx, ggml_tensor * w, const Yue2AtOpts & o) {
    if (!o.weights_f32 || w->type == GGML_TYPE_F32) {
        return w;
    }
    return ggml_cast(ctx, w, GGML_TYPE_F32);
}

// yue2_lm_rms with the frozen gain widened under the same switch. The gain is
// broadcast and frozen, so MUL's backward only ever builds its src0 arm.
static ggml_tensor * yue2_at_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps,
                                 const Yue2AtOpts & o) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), yue2_at_w(ctx, w, o));
}

// y = W x  (+ scale * B(A x) when the site is active).
//
// `yue2_nt_linear` with the frozen-weight cast folded in — same body otherwise,
// and the same reason for existing: one helper per site so a site cannot
// silently be left un-adapted.
static ggml_tensor * yue2_at_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                                    const Yue2AtLora & lo, const Yue2AtOpts & o) {
    ggml_tensor * y = ggml_mul_mat(ctx, yue2_at_w(ctx, w, o), x);
    if (!lo.on()) {
        return y;
    }
    ggml_tensor * ax = ggml_mul_mat(ctx, lo.a, x);   // [rank, S]
    ggml_tensor * bx = ggml_mul_mat(ctx, lo.b, ax);  // [out,  S]
    return ggml_add(ctx, y, ggml_scale(ctx, bx, lo.scale));
}

// ── LoRA sites (contract §2.1/§2.2) ────────────────────────────────────────
//
// The AR twins of the NAR preset, with the `nar_` prefix dropped. GQA is why k
// and v are half-width (Nkv = 8 against Nh = 16), and it is the same asymmetry
// yue2-adapter.h's merge preflight already reads off the base shape.
struct Yue2AtLayerAdapters {
    Yue2AtLora q, k, v, o;      // blk.N.attn_{q,k,v,output}
    Yue2AtLora gate, up, down;  // blk.N.ffn_{gate,up,down}
};

struct Yue2AtAdapters {
    std::vector<Yue2AtLayerAdapters> blk;  // block_count entries
};

// Upstream's own group: ar_lora_cursor.py:21-23 wraps
// self_attn.{q,k,v,o}_proj and mlp.{gate,up,down}_proj on every layer and
// nothing else, and its saved `targets` string reads
// "ar self_attn qkvo + mlp gate/up/down". Matching that exactly is what makes
// the README's rank/lr/step numbers mean anything here.
enum Yue2AtTarget {
    YUE2_AT_T_ATTN     = 0,  // attn:     q, k, v, o                  25,690,112 params at r64
    YUE2_AT_T_ATTN_MLP = 1,  // attn_mlp: + gate, up, down (default)  69,730,304 params at r64
};

static const char * yue2_at_target_name(Yue2AtTarget t) {
    return t == YUE2_AT_T_ATTN ? "attn" : "attn_mlp";
}

// `attn_mlp_embed` is named here on purpose: contract §2.3 recommends against
// it and says to state in the CLI help that it is UNIMPLEMENTED rather than
// pretend the preset exists. A silent "unknown target" would read as a typo.
static bool yue2_at_parse_target(const std::string & s, Yue2AtTarget * out, std::string * why) {
    if (s == "attn") {
        *out = YUE2_AT_T_ATTN;
    } else if (s.empty() || s == "attn_mlp") {
        *out = YUE2_AT_T_ATTN_MLP;
    } else if (s == "attn_mlp_embed") {
        if (why) {
            *why = "--target attn_mlp_embed is NOT IMPLEMENTED (contract §2.3). The chunked-CE head "
                   "computes dL/dh by hand and copies it into the trunk's gradient buffer, so a trainable "
                   "`output` needs its own gradient path back through that seam; `token_embd` additionally "
                   "carries a cross-tensor type invariant with latent_pos_embed. Upstream adapts neither.";
        }
        return false;
    } else {
        if (why) {
            *why = "--target must be attn or attn_mlp";
        }
        return false;
    }
    return true;
}

// How many tensors yue2_at_make_adapters will create — for sizing the context.
static size_t yue2_at_adapter_tensor_count(int n_layers, Yue2AtTarget target) {
    const size_t per_layer = (target == YUE2_AT_T_ATTN) ? 4u : 7u;
    return (size_t) n_layers * per_layer * 2u;
}

// Allocate A/B per site into a caller-owned context, name them "<tag>.A" /
// "<tag>.B", ggml_set_param both, push onto `params` in order. Mirrors
// yue2_nt_make_adapters (yue2-nar-train-graph.h:748-803) exactly.
//
// Tags are the GGUF module names without ".weight" — `blk.N.attn_q`,
// `blk.N.ffn_down`, and so on. The exporter maps them by prefixing "yue2." and
// suffixing ".lora_{A,B}.weight" and does nothing else.
//
// CAPITALISATION IS LOAD-BEARING: yue2_nt_init_adapters keys kaiming-vs-zero on
// the tensor name's LAST CHARACTER, and yue2-adapter.h's suffix lists are
// case-sensitive on both sides.
//
// `n_layers` lets the FD gate build adapters for a truncated stack. A real run
// passes block_count.
static bool yue2_at_make_adapters(ggml_context * ctx, const Yue2Model & m, int n_layers, int64_t rank,
                                  float alpha, Yue2AtTarget target, Yue2AtAdapters * ad,
                                  std::vector<ggml_tensor *> * params) {
    const Yue2LmConfig & c = m.lm_cfg;
    const int64_t        H = (int64_t) c.embedding_length;                          // 2048
    const int64_t        Q = (int64_t) c.head_count * (int64_t) c.key_length;       // 2048
    const int64_t        K = (int64_t) c.head_count_kv * (int64_t) c.key_length;    // 1024, GQA
    const int64_t        F = (int64_t) c.feed_forward_length;                       // 6144
    const float          sc = alpha / (float) rank;

    if (rank <= 0 || n_layers <= 0) {
        return false;
    }

    auto mk = [&](Yue2AtLora * lo, int64_t in, int64_t out, const std::string & tag) {
        lo->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rank);
        lo->b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out);
        if (!lo->a || !lo->b) {
            lo->a = lo->b = nullptr;  // ctx arena too small; the caller's count check catches it
            return;
        }
        lo->scale = sc;
        ggml_set_name(lo->a, (tag + ".A").c_str());
        ggml_set_name(lo->b, (tag + ".B").c_str());
        ggml_set_param(lo->a);
        ggml_set_param(lo->b);
        params->push_back(lo->a);
        params->push_back(lo->b);
    };

    const bool want_mlp = (target != YUE2_AT_T_ATTN);
    ad->blk.assign((size_t) n_layers, Yue2AtLayerAdapters{});
    for (int i = 0; i < n_layers; i++) {
        // Decimal and UNPADDED on both sides (yue2_fmt("blk.%d...") in
        // yue2-model.h; "blk." + std::to_string(i) here) — the kind of thing
        // that otherwise bites at block 10.
        const std::string p = "blk." + std::to_string(i) + ".";
        mk(&ad->blk[(size_t) i].q, H, Q, p + "attn_q");
        mk(&ad->blk[(size_t) i].k, H, K, p + "attn_k");
        mk(&ad->blk[(size_t) i].v, H, K, p + "attn_v");
        mk(&ad->blk[(size_t) i].o, Q, H, p + "attn_output");
        if (want_mlp) {
            mk(&ad->blk[(size_t) i].gate, H, F, p + "ffn_gate");
            mk(&ad->blk[(size_t) i].up, H, F, p + "ffn_up");
            mk(&ad->blk[(size_t) i].down, F, H, p + "ffn_down");
        }
    }
    return true;
}

// Trainable parameters at a given rank/preset — for the log line and the VRAM
// model. Counted, not measured.
static int64_t yue2_at_param_count(const Yue2LmConfig & c, int n_layers, int64_t rank, Yue2AtTarget target) {
    const int64_t H = (int64_t) c.embedding_length;
    const int64_t Q = (int64_t) c.head_count * (int64_t) c.key_length;
    const int64_t K = (int64_t) c.head_count_kv * (int64_t) c.key_length;
    const int64_t F = (int64_t) c.feed_forward_length;
    int64_t       per = rank * ((H + Q) + 2 * (H + K) + (Q + H));
    if (target != YUE2_AT_T_ATTN) {
        per += rank * (2 * (H + F) + (F + H));
    }
    return per * (int64_t) n_layers;
}

// ── Sequence assembly (contract §3.1) ──────────────────────────────────────
//
// One training example is ONE WHOLE SONG:
//
//   ids = prefix                                P tokens, ids[P-1] == MUSIC_START
//       ++ [code + YUE2_CODEC_OFFSET ...]       N tokens
//       ++ [MUSIC_END]                          1 token, unless truncated
//
// The prefix comes from `yue2_token_prefixes(tok, style, lyrics, YUE2_COT_OFF)`
// (yue2-tokenizer.h:495-509), which ALREADY ends ABC_START, ABC_END,
// MUSIC_START for cot=off — so MUSIC_START is INSIDE the prefix and must not be
// appended again. That matches ar_prep.py:32 and ar_lora_cursor.py:52-53.
//
// The `+ YUE2_CODEC_OFFSET` happens HERE and nowhere else, the way
// yue2_nar_train_cond_ids does it and the way the pipeline does it
// (yue2-pipeline.h:450). Stored codes stay RAW in [0, 32768). Double-offsetting
// is finite, in-vocabulary and completely wrong, which is why the tokenizer's
// own manifest note says so in as many words.
//
// SUPERVISED ROWS. Next-token CE: the hidden state at row r predicts ids[r+1].
//
//   supervised rows : r = P-1 .. S-2      (n_sup = S - P of them)
//   targets         : ids[P .. S-1]
//   masked          : rows 0 .. P-2 — every position inside the instruction,
//                     [Tags], style, [Lyrics], lyrics and the ABC brackets
//                     contributes NOTHING.
//
// The prefix is excluded BY SLICING, NOT BY WEIGHTING: the head only ever sees
// h[P-1 : S-1], so the prefix rows never reach the loss graph at all. That is
// ar_lora_cursor.py:60 (`h=hn[Lp-1:-1]; tgt=ids[0,Lp:]`) transcribed, and it is
// also what makes the chunked head cheap — the expensive part is proportional
// to n_sup, not to S.
//
// TRUNCATION. Upstream caps at MAXLEN and, when the codec stream does not fit,
// appends NO MUSIC_END (ar_lora_cursor.py:52-53). Mirrored exactly: a truncated
// song's final position is a codec token, not an ending, and supervising a fake
// ending there is precisely the failure MM3 spent a month on
// (project-mm3-adapter-eos-suppression).
struct Yue2AtSeq {
    std::vector<int32_t> ids;      // S token ids
    std::vector<int32_t> rows;     // n_sup slice rows, one per supervised position (CODEC range only)
    int64_t              prefix   = 0;  // P
    int64_t              n_sup    = 0;  // S - P (codec + MUSIC_END rows only)
    bool                 truncated = false;
    bool                 has_end   = false;
    // --abc-dropout / item 3: for a `full` example, the row range that
    // supervises the SHEET (plus ABC_END, MUSIC_START) — the FULL-VOCAB head
    // (yue2_at_head_text_chunked), disjoint from and immediately before the
    // codec range above (text_sup0 + text_n_sup == prefix - 1, the codec
    // head's own col0). text_n_sup == 0 means no text component at all — every
    // `off`-mode sequence, and any `full` prefix built with text_head_len == 0.
    int64_t              text_sup0  = 0;
    int64_t              text_n_sup = 0;
};

// `codec` is RAW. `max_len` is upstream's MAXLEN. Returns false with `err` set
// when the prefix alone does not leave room for a single codec token — that is
// a song that cannot be trained at this length, and the caller SKIPS it with a
// named warning rather than cropping (contract §4.4: a random crop teaches that
// a song may legitimately begin at position c0, which is MM3's crop-regime bug
// reproduced from scratch).
//
// `text_head_len`: 0 for an `off`-mode prefix (yue2_token_prefixes(...,
// YUE2_COT_OFF)) — no text component, seq.text_n_sup stays 0. For a `full`-mode
// prefix (YUE2_COT_FULL, sheet spliced in), the caller's own head_len_full(_
// dropped) — the row of ABC_START — so this can compute the text-supervised
// span [text_head_len, P-2] that sits immediately before the codec range
// (item 3: sheet tokens, ABC_END and MUSIC_START are IN the loss for `full`).
static bool yue2_at_build_sequence(const std::vector<int32_t> & prefix, const std::vector<int32_t> & codec,
                                   int64_t max_len, Yue2AtSeq * out, std::string * err,
                                   int64_t text_head_len = 0) {
    const int64_t P = (int64_t) prefix.size();
    if (P < 1) {
        if (err) {
            *err = "empty prefix";
        }
        return false;
    }
    if (prefix[(size_t) P - 1] != (int32_t) YUE2_MUSIC_START) {
        // Loud rather than silent: every prefix this trainer builds (cot=off
        // OR cot=full) ENDS with MUSIC_START, and one that does not is either
        // a different cot mode or a hand-assembled vector, both of which would
        // train against the wrong supervised boundary and still show a
        // falling loss.
        if (err) {
            *err = "the prefix does not end in MUSIC_START (" + std::to_string((long long) YUE2_MUSIC_START) +
                   ") — yue2_token_prefixes(..., YUE2_COT_OFF|YUE2_COT_FULL) is the only assembly this trainer "
                   "supports";
        }
        return false;
    }
    if (text_head_len < 0 || text_head_len > P - 2) {
        if (err) {
            *err = "text_head_len " + std::to_string((long long) text_head_len) +
                   " does not fall strictly before the codec boundary of a " + std::to_string((long long) P) +
                   "-token prefix";
        }
        return false;
    }
    const int64_t room = max_len - P - 1;
    if (room < 1) {
        if (err) {
            *err = "the prefix is " + std::to_string((long long) P) + " tokens, which leaves no room for a "
                   "codec stream inside --max-len " + std::to_string((long long) max_len);
        }
        return false;
    }
    if (codec.empty()) {
        if (err) {
            *err = "no codec ids — run `ace-train yue2-tokenize` over the manifest first";
        }
        return false;
    }

    const int64_t N = std::min<int64_t>((int64_t) codec.size(), room);
    out->ids.clear();
    out->ids.reserve((size_t) (P + N + 1));
    out->ids.insert(out->ids.end(), prefix.begin(), prefix.end());
    for (int64_t i = 0; i < N; i++) {
        const int32_t raw = codec[(size_t) i];
        if (raw < 0 || raw >= (int32_t) YUE2_CODEC_SIZE) {
            if (err) {
                *err = "codec id " + std::to_string((long long) raw) + " at frame " +
                       std::to_string((long long) i) + " is outside [0, " +
                       std::to_string((long long) YUE2_CODEC_SIZE) +
                       ") — these must be RAW codes, never pre-offset token ids";
            }
            return false;
        }
        out->ids.push_back(raw + (int32_t) YUE2_CODEC_OFFSET);
    }
    out->truncated = (int64_t) codec.size() > room;
    out->has_end   = !out->truncated;
    if (out->has_end) {
        out->ids.push_back((int32_t) YUE2_MUSIC_END);
    }

    const int64_t S = (int64_t) out->ids.size();
    out->prefix     = P;
    out->n_sup      = S - P;
    out->rows.assign((size_t) out->n_sup, 0);
    for (int64_t i = 0; i < out->n_sup; i++) {
        // MUSIC_END -> 0, raw code c -> c + 1. One subtraction covers both,
        // because CODEC_OFFSET == MUSIC_END + 1.
        const int64_t row = (int64_t) out->ids[(size_t) (P + i)] - YUE2_AT_SLICE_ROW0;
        if (row < 0 || row >= YUE2_AT_SLICE_ROWS) {
            if (err) {
                *err = "supervised target " + std::to_string((long long) out->ids[(size_t) (P + i)]) +
                       " falls outside the scored slice [" + std::to_string((long long) YUE2_AT_SLICE_ROW0) +
                       ", " + std::to_string((long long) (YUE2_AT_SLICE_ROW0 + YUE2_AT_SLICE_ROWS)) + ")";
            }
            return false;
        }
        out->rows[(size_t) i] = (int32_t) row;
    }
    // Reset first: `out` is reused across draws, and an `off` draw after a
    // `full` one must not inherit the previous song's sheet range (that ran
    // P5b on a sequence with no sheet and failed the range check).
    out->text_sup0  = 0;
    out->text_n_sup = 0;
    if (text_head_len > 0) {
        // [text_head_len, P-2] predicts ids[text_head_len+1 .. P-1] — the
        // sheet, ABC_END, MUSIC_START — and butts up EXACTLY against the codec
        // range's own col0 (P-1): row P-2 predicts ids[P-1] == MUSIC_START,
        // row P-1 predicts ids[P] == the first codec id, so the two ranges
        // are contiguous and disjoint, never double- or un-supervising a row.
        out->text_sup0  = text_head_len;
        out->text_n_sup = P - 1 - text_head_len;
    }
    return true;
}

// ── One trainable AR block: mirrors yue2_ar_block op for op ────────────────
//
// h [H,S] -> [H,S]. No KV cache (trap 1), no set_rows, no view. `mask` is the
// F16 [S,S] causal mask; `pos` is the I32 [S] position vector (pos[i] == i).
static ggml_tensor * yue2_at_block(ggml_context * ctx, const Yue2LmConfig & c, const Yue2LmLayer & w,
                                   const Yue2AtLayerAdapters & ad, ggml_tensor * h, ggml_tensor * pos,
                                   ggml_tensor * mask, const Yue2AtOpts & o) {
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t S   = h->ne[1];

    ggml_tensor * n = yue2_at_rms(ctx, h, w.attn_norm, c.rms_eps, o);

    // The LoRA add happens BEFORE the reshape_4d, which is what keeps S out of
    // ne2 at every trainable matmul (trap 3). DO NOT HOIST THE RESHAPE.
    ggml_tensor * q = ggml_reshape_4d(ctx, yue2_at_linear(ctx, w.attn_q, n, ad.q, o), D, Nh, S, 1);
    ggml_tensor * k = ggml_reshape_4d(ctx, yue2_at_linear(ctx, w.attn_k, n, ad.k, o), D, Nkv, S, 1);
    ggml_tensor * v = ggml_reshape_4d(ctx, yue2_at_linear(ctx, w.attn_v, n, ad.v, o), D, Nkv, S, 1);

    // Per-head QK RMSNorm (dim = head_dim = 128, eps 1e-6), strictly BEFORE
    // RoPE, never on V (yue2-lm-graph.h:152-155). No ggml_cont: the reshape_4d's
    // source ends in ggml_add, so it is contiguous.
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_eps), yue2_at_w(ctx, w.attn_q_norm, o));
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_eps), yue2_at_w(ctx, w.attn_k_norm, o));

    // NeoX half-split rotation, theta = rope_freq_base (1e6). Identical call to
    // yue2-lm-graph.h:163-166 including every scaling argument. ROPE has a
    // backward and its positions are on ggml's non-differentiable list.
    q = ggml_rope_ext(ctx, q, pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f,
                      0.0f, 0.0f);

    ggml_tensor * attn = nullptr;
    if (o.attn == YUE2_AT_FA_EXACT) {
        // [D,Nkv,S,1] -> [D,S,Nkv,1] contiguous, then the IDENTITY RESHAPE that
        // is trap 2's fix. It is load-bearing, not tidying: without it CONT's
        // backward aborts on V's transposed gradient view. Both sides get it.
        ggml_tensor * k_w = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)), D, S, Nkv, 1);
        ggml_tensor * v_w = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)), D, S, Nkv, 1);
        ggml_tensor * q4  = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,S,Nh,1]
        // The scale stays INSIDE soft_max_ext: its backward reads it back out
        // of op_params (ggml.c:7284-7295), so pre-scaling q would lose it. CUDA
        // only supports SOFT_MAX_BACK at max_bias == 0, which is what this
        // passes.
        attn = yue2_lm_attn_f32(ctx, q4, k_w, v_w, mask, c.softmax_scale);
    } else {
        // FUSED. NO ggml_cont on q/k/v (lm-graph.h:1471-1478): the op reads them
        // through nb[1..3] on purpose, PERMUTE's backward stores a permuted view
        // into grads[q], and `q` is a reshape_4d whose backward conts it.
        ggml_tensor * q4 = ggml_permute(ctx, q, 0, 2, 1, 3);  // [D,S,Nh, 1]
        ggml_tensor * k4 = ggml_permute(ctx, k, 0, 2, 1, 3);  // [D,S,Nkv,1]
        ggml_tensor * v4 = ggml_permute(ctx, v, 0, 2, 1, 3);
        ggml_tensor * pk = ggml_flash_attn_train(ctx, q4, k4, v4, mask, c.softmax_scale);
        ggml_flash_attn_train_set_prec(pk, o.attn == YUE2_AT_FA_FLASH_F32 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
        attn = ggml_flash_attn_train_get_o(ctx, pk);  // contiguous [D,Nh,S]
    }
    attn = ggml_reshape_2d(ctx, attn, H, S);

    h = ggml_add(ctx, h, yue2_at_linear(ctx, w.attn_output, attn, ad.o, o));

    // SwiGLU: down(silu(gate) * up), spelled out rather than ggml_swiglu_split
    // — the inference path spells it out too, and keeping the two identical is
    // the point of this file (see the header).
    ggml_tensor * n2   = yue2_at_rms(ctx, h, w.ffn_norm, c.rms_eps, o);
    ggml_tensor * gate = ggml_silu(ctx, yue2_at_linear(ctx, w.ffn_gate, n2, ad.gate, o));
    ggml_tensor * up   = yue2_at_linear(ctx, w.ffn_up, n2, ad.up, o);
    return ggml_add(ctx, h, yue2_at_linear(ctx, w.ffn_down, ggml_mul(ctx, gate, up), ad.down, o));
}

// ── The fused-attention capability probe (flash-attn adoption contract §2) ──
//
// A `false` from ggml_backend_supports_op is NOT a failure the run would see:
// backend_sched_new registers the CPU backend alongside the GPU one, so the
// scheduler would SPLIT the graph and copy Q/K/V and the F16 mask over PCIe
// every layer of every step. Correct, unusably slow, LOW VRAM, and
// indistinguishable from a pass on every number the run reports. So flash mode
// asks explicitly, at init, at the run's REAL shapes, and refuses to start.
static bool yue2_at_flash_probe(ggml_backend_t backend, const Yue2LmConfig & c, int64_t s_max,
                                std::string * err) {
    bool fwd_ok = false, bwd_ok = false;
    dit_flash_probe(backend, (int) c.key_length, (int) c.head_count, (int) c.head_count_kv, (int) s_max,
                    (int) s_max, 1, c.softmax_scale, &fwd_ok, &bwd_ok);
    if (fwd_ok && bwd_ok) {
        return true;
    }
    if (err) {
        char b[320];
        snprintf(b, sizeof(b),
                 "--attn flash: this backend does not support GGML_OP_FLASH_ATTN_TRAIN at D=%u Nh=%u Nkv=%u "
                 "S=%lld (forward %s, backward %s). Falling back would split the graph onto the CPU and "
                 "look like a pass on every number the run reports, so this is a hard error — use "
                 "--attn exact.",
                 c.key_length, c.head_count, c.head_count_kv, (long long) s_max, fwd_ok ? "ok" : "NO",
                 bwd_ok ? "ok" : "NO");
        *err = b;
    }
    return false;
}

// ── The causal mask (contract §1.6) ────────────────────────────────────────
//
// Taken verbatim from yue2_ar_forward:404-413: F16, 0 below the diagonal and
// -INFINITY above, built on the host and uploaded once per micro-step.
// `ggml_soft_max_ext` asserts contiguity and ne[0] == scores->ne[0];
// `ggml_flash_attn_train` additionally requires F16 — the existing mask already
// satisfies both, which is a small piece of luck.
//
// ONE persistent flat [S_MAX * S_MAX] F16 buffer, viewed as a contiguous [S, S]
// block per step, with the row stride written as S * ggml_element_size(buf)
// rather than S * 2 (lm-graph.h:1500-1510's rule). It is the one quadratic term
// that survives flash mode, at 2 bytes per element instead of Nh * 4 = 64.
static void yue2_at_fill_mask(std::vector<uint16_t> * host, int64_t S) {
    // Row-wise fills rather than S^2 calls to ggml_fp32_to_fp16: at S = 12,288
    // that is 151 M conversions per micro-step, which is measurable host time
    // spent producing two distinct values.
    const uint16_t keep = ggml_fp32_to_fp16(0.0f);
    const uint16_t drop = ggml_fp32_to_fp16(-INFINITY);
    host->assign((size_t) (S * S), drop);
    for (int64_t i = 0; i < S; i++) {  // i = query position, j <= i is visible
        std::fill(host->begin() + (size_t) (i * S), host->begin() + (size_t) (i * S + i + 1), keep);
    }
}

// ── Persistent state: the checkpoint boundaries, the head and the labels ────
//
// One backend buffer PER GROUP, because ggml_backend_buffer_clear() is
// whole-buffer and Gh[0]/Gh[1] are cleared on different cadences. Straight off
// lm-ckpt.h:264-300.
//
//   C[0..L-1]  [H, S_MAX] F32   checkpoint boundaries         never cleared
//   Gh[0]      [H, S_MAX] F32   == t_G, seeded by the head    cleared per micro-step
//   Gh[1]      [H, S_MAX] F32   the ping-pong partner         cleared per segment
//   t_H        [H, S_MAX] F32   post-final-norm hidden states written once per step
//   t_head     [H, SL]   F32    the sliced output head        written once at alloc
//   t_headT    [SL, H]   F32    its transpose, for dL/dh      written once at alloc
//   t_labc     [SL, CH]  F32    one dense chunk of labels     sparse set/clear
//   t_msk      [S_MAX*S_MAX] F16                              per micro-step
//   t_ids/t_pos [S_MAX] I32                                   per micro-step
//   t_gs/t_one  [1] F32                                       per chunk / constant
struct Yue2AtState {
    const Yue2Model * m     = nullptr;
    int64_t           s_max = 0;
    int64_t           chunk = 256;
    int               L     = 0;

    ggml_context *        ctx_ckpt = nullptr;
    ggml_backend_buffer_t buf_ckpt = nullptr;
    ggml_context *        ctx_gh0  = nullptr;
    ggml_backend_buffer_t buf_gh0  = nullptr;
    ggml_context *        ctx_gh1  = nullptr;
    ggml_backend_buffer_t buf_gh1  = nullptr;
    ggml_context *        ctx_misc = nullptr;  // t_H, t_msk, t_ids, t_pos, t_gs, t_one
    ggml_backend_buffer_t buf_misc = nullptr;
    ggml_context *        ctx_head = nullptr;  // t_head + t_headT (+ t_full_headT when want_text_head)
    ggml_backend_buffer_t buf_head = nullptr;
    ggml_context *        ctx_lab  = nullptr;  // t_labc (+ t_labc_full when want_text_head)
    ggml_backend_buffer_t buf_lab  = nullptr;
    ggml_context *        ctx_cur  = nullptr;  // t_curT: the lyric-cursor target block
    ggml_backend_buffer_t buf_cur  = nullptr;

    std::vector<ggml_tensor *> C;
    ggml_tensor *              Gh[2]   = { nullptr, nullptr };
    ggml_tensor *              t_H     = nullptr;
    ggml_tensor *              t_head  = nullptr;
    ggml_tensor *              t_headT = nullptr;
    ggml_tensor *              t_labc  = nullptr;
    ggml_tensor *              t_msk   = nullptr;
    ggml_tensor *              t_ids   = nullptr;
    ggml_tensor *              t_pos   = nullptr;
    ggml_tensor *              t_gs    = nullptr;
    ggml_tensor *              t_one   = nullptr;
    ggml_tensor *              t_curT  = nullptr;  // [L_max, s_max] F32; a song uses a packed [L, nF] prefix of it
    int64_t                    cur_l_max = 0;

    // --abc-dropout (item 3): the FULL-VOCAB head for a `full` example's
    // sheet-token span. Built ONLY when want_text_head was passed to
    // yue2_at_state_alloc (a run that could actually draw a `full` example) —
    // t_full_headT alone is ~V*H*4 bytes (≈1.5 GB at V≈184.7k, H=2048), so a
    // run with `--no-abc` or no sheets in its manifest pays nothing for this.
    bool                       want_text_head = false;
    int64_t                    chunk_text     = 64;  // rows per text-head CE chunk (V is huge; keep chunks small)
    ggml_tensor *              t_full_headT   = nullptr;  // [V, H] F32 — the vocab-fastest transpose, for dL/dh
    ggml_tensor *              t_labc_full    = nullptr;  // [V, chunk_text] F32 — one dense chunk of labels

    std::vector<uint8_t>  arena;  // one reused graph arena
    std::vector<uint16_t> mask_host;
    std::vector<int32_t>  pos_host;
    std::vector<ggml_tensor *> gacc;
    int64_t               last_mask_S = 0;
    const void *          last_cur    = nullptr;  // which song's targets t_curT currently holds

    size_t fixed_bytes() const {
        size_t                      b       = 0;
        const ggml_backend_buffer_t bufs[7] = { buf_ckpt, buf_gh0, buf_gh1, buf_misc, buf_head, buf_lab, buf_cur };
        for (int i = 0; i < 7; i++) {
            if (bufs[i]) {
                b += ggml_backend_buffer_get_size(bufs[i]);
            }
        }
        return b;
    }
};

static void yue2_at_state_free(Yue2AtState * st) {
    ggml_backend_buffer_t bufs[7] = { st->buf_ckpt, st->buf_gh0, st->buf_gh1,
                                      st->buf_misc, st->buf_head, st->buf_lab, st->buf_cur };
    for (int i = 0; i < 7; i++) {
        if (bufs[i]) {
            ggml_backend_buffer_free(bufs[i]);
        }
    }
    ggml_context * ctxs[7] = { st->ctx_ckpt, st->ctx_gh0, st->ctx_gh1, st->ctx_misc,
                               st->ctx_head, st->ctx_lab, st->ctx_cur };
    for (int i = 0; i < 7; i++) {
        if (ctxs[i]) {
            ggml_free(ctxs[i]);
        }
    }
    *st = Yue2AtState{};
}

// Node budgets. Every graph below builds AT MOST ONE LAYER — that is the whole
// point of the checkpointed path — so these are a layer's counts, not a trunk's.
static int yue2_at_forward_nodes() {
    return 1024;
}
static int yue2_at_segment_nodes() {
    return 8192;
}

// Build the sliced output head and its transpose, once, on the device.
//
// `m.lm.output` is ggml [H, V]: rows of H, V of them. The slice is rows
// [ROW0, ROW0 + SL), which is a CONTIGUOUS span of that buffer, so a plain
// view_2d is contiguous and ggml_cpy widens it in one kernel. The transpose is
// then taken off the already-widened copy, which keeps the dependency inside
// one graph.
static bool yue2_at_build_head(Yue2AtState * st, ggml_backend_sched_t sched, std::string * err) {
    const Yue2Model & m   = *st->m;
    ggml_tensor *     out = m.lm.output;
    if (!out) {
        *err = "the LM has no `output` tensor";
        return false;
    }
    const int64_t H = out->ne[0];
    if (YUE2_AT_SLICE_ROW0 + YUE2_AT_SLICE_ROWS > out->ne[1]) {
        *err = "the output head has " + std::to_string((long long) out->ne[1]) +
               " rows, fewer than the scored slice needs (" +
               std::to_string((long long) (YUE2_AT_SLICE_ROW0 + YUE2_AT_SLICE_ROWS)) + ")";
        return false;
    }

    std::vector<uint8_t> arena((size_t) 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false) +
                               1024 * 1024);
    ggml_init_params     ip  = { arena.size(), arena.data(), /*no_alloc*/ true };
    ggml_context *       ctx = ggml_init(ip);
    ggml_cgraph *        gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
    ggml_tensor * view = ggml_view_2d(ctx, out, H, YUE2_AT_SLICE_ROWS, out->nb[1],
                                      (size_t) YUE2_AT_SLICE_ROW0 * out->nb[1]);
    ggml_tensor * hc   = ggml_cpy(ctx, view, st->t_head);
    ggml_build_forward_expand(gf, hc);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, ggml_transpose(ctx, hc)), st->t_headT));
    ggml_backend_sched_reset(sched);
    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    ggml_free(ctx);
    if (!ok) {
        *err = "building the sliced output head failed";
    }
    return ok;
}

// Build the FULL-VOCAB output head's transpose, once, on the device — the
// --abc-dropout / item 3 counterpart of yue2_at_build_head above, for the
// sheet-token span of a `full` example.
//
// UNLIKE the sliced head, this builds ONLY the transpose (`t_full_headT`,
// [V, H] F32), not a forward copy: yue2_at_head_text_chunked's forward matmul
// reads `m.lm.output` directly in its own native dtype, exactly the way the
// eval-only yue2_at_head_fullvocab_ce already does — nothing there needs a
// gradient, so no F32 mirror is worth keeping around for it. Only the
// backward's dh = mul_mat(headT, dl) needs vocab-fastest layout, which native
// storage (H fastest) never has, hence this one-time transpose+upcast. The
// transient intermediate (`ggml_cont(transpose(out))`, native dtype, [V, H])
// is freed by the scheduler once this graph completes; nothing here is kept
// beyond `t_full_headT` itself.
static bool yue2_at_build_text_head(Yue2AtState * st, ggml_backend_sched_t sched, std::string * err) {
    const Yue2Model & m   = *st->m;
    ggml_tensor *     out = m.lm.output;
    if (!out) {
        *err = "the LM has no `output` tensor";
        return false;
    }
    if (!st->t_full_headT) {
        *err = "yue2_at_build_text_head called without a t_full_headT tensor (want_text_head was false at alloc)";
        return false;
    }
    std::vector<uint8_t> arena((size_t) 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false) +
                               1024 * 1024);
    ggml_init_params     ip  = { arena.size(), arena.data(), /*no_alloc*/ true };
    ggml_context *       ctx = ggml_init(ip);
    ggml_cgraph *        gf  = ggml_new_graph_custom(ctx, 16, /*grads=*/false);
    ggml_tensor * tp = ggml_cont(ctx, ggml_transpose(ctx, out));  // [V, H], native dtype
    ggml_build_forward_expand(gf, ggml_cpy(ctx, tp, st->t_full_headT));
    ggml_backend_sched_reset(sched);
    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    ggml_free(ctx);
    if (!ok) {
        *err = "building the full-vocab output head transpose failed";
    }
    return ok;
}

// `cursor_l_max` sizes the lyric-cursor target block: the largest lyric-token
// span any artist song in this run carries. 0 = the cursor term is off and no
// block is allocated, which leaves every graph byte-identical to a run that
// never heard of it.
//
// `want_text_head` (item 3): whether this run could ever draw a `full`
// example — allocates the full-vocab head's persistent buffers
// (t_full_headT, t_labc_full) when true. false leaves every graph
// byte-identical to a run that never heard of --abc-dropout, same as
// cursor_l_max == 0 does for the cursor term.
static bool yue2_at_state_alloc(Yue2AtState * st, const Yue2Model * m, int64_t s_max, int64_t chunk,
                                ggml_backend_sched_t sched, std::string * err, int64_t cursor_l_max = 0,
                                bool want_text_head = false) {
    const Yue2LmConfig & c = m->lm_cfg;
    const int64_t        H = (int64_t) c.embedding_length;
    const int            L = (int) c.block_count;

    st->m              = m;
    st->s_max          = s_max;
    st->chunk          = std::max<int64_t>(1, chunk);
    st->L              = L;
    st->cur_l_max      = std::max<int64_t>(0, cursor_l_max);
    st->want_text_head = want_text_head;
    st->chunk_text     = std::min<int64_t>(st->chunk, 64);

    auto mkctx = [](size_t n) {
        ggml_init_params p = { n * ggml_tensor_overhead() + 4096, nullptr, /*no_alloc*/ true };
        return ggml_init(p);
    };
    st->ctx_ckpt = mkctx((size_t) L + 4);
    st->ctx_gh0  = mkctx(4);
    st->ctx_gh1  = mkctx(4);
    st->ctx_misc = mkctx(16);
    st->ctx_head = mkctx(st->want_text_head ? 8 : 4);
    st->ctx_lab  = mkctx(st->want_text_head ? 8 : 4);
    st->ctx_cur  = st->cur_l_max > 0 ? mkctx(4) : nullptr;
    if (!st->ctx_ckpt || !st->ctx_gh0 || !st->ctx_gh1 || !st->ctx_misc || !st->ctx_head || !st->ctx_lab ||
        (st->cur_l_max > 0 && !st->ctx_cur)) {
        *err = "yue2_at_state_alloc: ggml_init failed";
        return false;
    }

    st->C.assign((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        st->C[(size_t) i] = ggml_new_tensor_2d(st->ctx_ckpt, GGML_TYPE_F32, H, s_max);
        ggml_set_name(st->C[(size_t) i], ("yue2_at_C." + std::to_string(i)).c_str());
        // LOAD-BEARING, and the thing this design does not work without
        // (lm-ckpt.h:454-458). ggml_set_param is legal on a pre-allocated leaf —
        // it only asserts op == NONE — and being a PARAM is what makes the
        // boundary a NODE of the segment graph rather than a leaf, which is the
        // only reason yue2_at_fill_gacc can hand it a persistent dL/dX buffer.
        // Without it the backward has nowhere to deposit dL/dC[l], every
        // segment below the last one sees a zero upstream gradient, and the run
        // trains the top layer only — with a loss that still falls.
        ggml_set_param(st->C[(size_t) i]);
        ggml_set_input(st->C[(size_t) i]);
    }
    st->Gh[0] = ggml_new_tensor_2d(st->ctx_gh0, GGML_TYPE_F32, H, s_max);
    st->Gh[1] = ggml_new_tensor_2d(st->ctx_gh1, GGML_TYPE_F32, H, s_max);
    ggml_set_name(st->Gh[0], "yue2_at_Gh0");
    ggml_set_name(st->Gh[1], "yue2_at_Gh1");
    ggml_set_input(st->Gh[0]);
    ggml_set_input(st->Gh[1]);

    st->t_H   = ggml_new_tensor_2d(st->ctx_misc, GGML_TYPE_F32, H, s_max);
    st->t_msk = ggml_new_tensor_1d(st->ctx_misc, GGML_TYPE_F16, s_max * s_max);
    st->t_ids = ggml_new_tensor_1d(st->ctx_misc, GGML_TYPE_I32, s_max);
    st->t_pos = ggml_new_tensor_1d(st->ctx_misc, GGML_TYPE_I32, s_max);
    st->t_gs  = ggml_new_tensor_1d(st->ctx_misc, GGML_TYPE_F32, 1);
    st->t_one = ggml_new_tensor_1d(st->ctx_misc, GGML_TYPE_F32, 1);
    ggml_set_name(st->t_H, "yue2_at_tH");
    ggml_set_name(st->t_msk, "yue2_at_mask");
    ggml_set_name(st->t_ids, "yue2_at_ids");
    ggml_set_name(st->t_pos, "yue2_at_pos");
    ggml_set_name(st->t_gs, "yue2_at_gs");
    ggml_set_name(st->t_one, "yue2_at_one");
    ggml_set_input(st->t_H);
    ggml_set_input(st->t_msk);
    ggml_set_input(st->t_ids);
    ggml_set_input(st->t_pos);
    ggml_set_input(st->t_gs);
    ggml_set_input(st->t_one);

    st->t_head  = ggml_new_tensor_2d(st->ctx_head, GGML_TYPE_F32, H, YUE2_AT_SLICE_ROWS);
    st->t_headT = ggml_new_tensor_2d(st->ctx_head, GGML_TYPE_F32, YUE2_AT_SLICE_ROWS, H);
    ggml_set_name(st->t_head, "yue2_at_head");
    ggml_set_name(st->t_headT, "yue2_at_headT");
    ggml_set_input(st->t_head);
    ggml_set_input(st->t_headT);
    st->t_labc = ggml_new_tensor_2d(st->ctx_lab, GGML_TYPE_F32, YUE2_AT_SLICE_ROWS, st->chunk);
    ggml_set_name(st->t_labc, "yue2_at_labels");
    ggml_set_input(st->t_labc);
    if (st->want_text_head) {
        const int64_t V     = (int64_t) c.vocab_size;
        st->t_full_headT    = ggml_new_tensor_2d(st->ctx_head, GGML_TYPE_F32, V, H);
        st->t_labc_full     = ggml_new_tensor_2d(st->ctx_lab, GGML_TYPE_F32, V, st->chunk_text);
        ggml_set_name(st->t_full_headT, "yue2_at_full_headT");
        ggml_set_name(st->t_labc_full, "yue2_at_labels_full");
        ggml_set_input(st->t_full_headT);
        ggml_set_input(st->t_labc_full);
    }
    if (st->cur_l_max > 0) {
        // One block for the largest song; a smaller song packs its [L, nF]
        // targets into the front of it and views exactly that much, so the
        // view is contiguous whatever L_max is.
        st->t_curT = ggml_new_tensor_2d(st->ctx_cur, GGML_TYPE_F32, st->cur_l_max, s_max);
        ggml_set_name(st->t_curT, "yue2_at_cursorT");
        ggml_set_input(st->t_curT);
    }

    struct Pair {
        ggml_context **         ctx;
        ggml_backend_buffer_t * buf;
        const char *            tag;
    };
    Pair pairs[7] = { { &st->ctx_ckpt, &st->buf_ckpt, "checkpoint boundaries" },
                      { &st->ctx_gh0, &st->buf_gh0, "gradient ping" },
                      { &st->ctx_gh1, &st->buf_gh1, "gradient pong" },
                      { &st->ctx_misc, &st->buf_misc, "hidden states + mask" },
                      { &st->ctx_head, &st->buf_head, "sliced output head" },
                      { &st->ctx_lab, &st->buf_lab, "label chunk" },
                      { &st->ctx_cur, &st->buf_cur, "lyric-cursor targets" } };
    for (Pair & p : pairs) {
        if (!*p.ctx) {
            continue;  // the cursor block is optional
        }
        *p.buf = ggml_backend_alloc_ctx_tensors(*p.ctx, m->backend);
        if (!*p.buf) {
            *err = std::string("yue2_at_state_alloc: out of VRAM allocating the ") + p.tag;
            return false;
        }
        // NO GGML_BACKEND_BUFFER_USAGE_WEIGHTS here, deliberately, and the
        // contrast with the LoRA buffer in the runner is the point: these are
        // graph DESTINATIONS (every one of them is a ggml_cpy target at some
        // phase), not read-only leaves. lm-ckpt.h:530-535 allocates its
        // equivalent six buffers with no usage hint for the same reason, and it
        // is the proven design.
        ggml_backend_buffer_clear(*p.buf, 0);
    }

    // t_one is a CONSTANT 1.0: the segment surrogate's loss gradient. It is NOT
    // the optimizer's t_lossgrad, and that distinction is lm-ckpt.h's D9 — the
    // 1/grad_accum scaling lives in the head's `gs`, so seeding the surrogate
    // with it as well would square it.
    {
        const float one = 1.0f;
        ggml_backend_tensor_set(st->t_one, &one, 0, sizeof(float));
    }
    if (!yue2_at_build_head(st, sched, err)) {
        return false;
    }
    if (st->want_text_head && !yue2_at_build_text_head(st, sched, err)) {
        return false;
    }

    // Sized together with the node cap, deliberately: oversizing the CAP while
    // undersizing the ARENA is what bit the MM3 DiT trainer — ggml_new_tensor
    // returns NULL, expansion carries on, and the crash lands later in alloc
    // with no message.
    const size_t nodes = (size_t) yue2_at_segment_nodes();
    st->arena.resize(ggml_tensor_overhead() * (nodes * 2) + ggml_graph_overhead_custom(nodes, true) +
                     (size_t) 32 * 1024 * 1024);
    st->pos_host.assign((size_t) s_max, 0);
    for (int64_t i = 0; i < s_max; i++) {
        st->pos_host[(size_t) i] = (int32_t) i;  // pos[i] = i and nothing else (contract §1.6)
    }
    return true;
}

// ── PARAM-flag node classes in a segment graph, IN PRIORITY ORDER ──────────
//
//   1. the segment's checkpoint input  -> g_out  (its dL/dX; NOT in param_slot)
//   2. a LoRA A/B tensor               -> opt->acc[slot]
//   3. the surrogate loss node         -> t_one  (NOT opt->t_lossgrad — D9)
//
// `ckpt_in` carries the PARAM flag AND is absent from param_slot, so the order
// of the first two branches is LOAD-BEARING: swapped, the GGML_ASSERT below
// fires on every segment. Straight off lm_ckpt_fill_gacc.
static void yue2_at_fill_gacc(const LmOptim * o, ggml_cgraph * gf, ggml_tensor * ckpt_in, ggml_tensor * g_out,
                              ggml_tensor * t_one, std::vector<ggml_tensor *> * gacc) {
    const int n = ggml_graph_n_nodes(gf);
    gacc->assign((size_t) n, nullptr);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(gf, i);
        if (ckpt_in && nd == ckpt_in) {
            (*gacc)[(size_t) i] = g_out;
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_PARAM) {
            auto it = o->param_slot.find(nd);
            GGML_ASSERT(it != o->param_slot.end());
            (*gacc)[(size_t) i] = o->acc[(size_t) it->second];
            continue;
        }
        if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
            (*gacc)[(size_t) i] = t_one;
        }
    }
}

// ── One micro-step ─────────────────────────────────────────────────────────

struct Yue2AtRun {
    const Yue2Model *    m     = nullptr;
    LmOptim *            opt   = nullptr;
    ggml_backend_sched_t sched = nullptr;
    Yue2AtState *        st    = nullptr;
    const Yue2AtAdapters * ad  = nullptr;

    Yue2AtOpts opts;
    int64_t    grad_accum = 1;
    int        n_layers   = 0;  // truncated stacks for the FD gate; 0 = block_count

    // dL/dloss — THE SEED OF THE WHOLE BACKWARD, and in this trainer it lives
    // HERE and nowhere else.
    //
    // The NAR trainer autodiffs through its loss node, so its seed is the
    // grad_acc ggml_build_backward_expand is handed for that node
    // (`o->t_lossgrad`, lm-optim.h:534). This trainer does NOT autodiff through
    // the head (P5): it calls ggml_cross_entropy_loss_back itself with `t_gs`
    // as the incoming gradient, and the trunk surrogate below is seeded with a
    // CONSTANT 1.0 because the surrogate is an identity, not the loss (D9).
    // So `gs` IS this trainer's t_lossgrad, and a factor applied anywhere else
    // — including opt->t_lossgrad, which the segment graphs never read — never
    // reaches a single gradient.
    //
    // That is exactly what made the FD gate's YUE2_FD_LOSSGRAD negative control
    // a no-op: it wrote the seed to a tensor nothing consumed, so a deliberately
    // wrong gradient produced byte-identical probe output and still passed. The
    // seed has to multiply `gs`, and it does, below.
    //
    // 1.0 for a real run (the 1/grad_accum scaling is already in `gs`); the
    // gate sets it from YUE2_FD_LOSSGRAD.
    float      lossgrad   = 1.0f;

    // Contract §6.5's forward-side negative control. 0 for every real run; the
    // head reads hidden columns P-1+sup_shift+i and writes dL/dh back into the
    // SAME columns, so the gradient stays self-consistent and the FD gate stays
    // PASS — which is the entire point: this is a fault FD cannot see.
    int sup_shift = 0;

    bool forward_only = false;  // evaluation: stop after the head
    // When non-null the head downloads each chunk's logits and re-aggregates
    // the CE on the HOST IN DOUBLE (contract §6.2). The in-graph
    // ggml_cross_entropy_loss is deterministic to the bit but only ~1e-4
    // accurate, and DIFFERENCING TWO SUCH VALUES is catastrophic cancellation —
    // which is the entire numeric arm of the FD gate. It is also an INDEPENDENT
    // implementation of the loss rather than the same kernel twice.
    double * host_ce = nullptr;

    // ── the lyric-cursor auxiliary term (ar_lora_cursor.py:62-67) ──
    //
    // `cur` is this micro-step's targets (null = none: a minted song, an
    // unbound artist song, or the term switched off). `cur_w` is the trainable
    // [H, H] cursor head, owned by the optimizer like any LoRA factor.
    // `cursor_weight` is upstream's CUR_W. The eval path never sets `cur`, so
    // the reported artist/minted_val numbers stay the LM loss alone, exactly as
    // upstream's evaluate() reports them.
    //
    // `host_cur`, when non-null, is the term re-aggregated ON THE HOST IN
    // DOUBLE for the FD gate's numeric arm — the same reason `host_ce` exists.
    const struct Yue2AtCursor * cur           = nullptr;
    ggml_tensor *               cur_w         = nullptr;
    float                       cursor_weight = 0.0f;
    double *                    host_cur      = nullptr;
};

// ── The lyric-cursor targets (ar_lora_cursor.py:33-49, `cursor_targets`) ───
//
// Per artist song, a distribution over its lyric TOKENS for every codec frame:
// which words is the singer on at frame i. Built once on the host and cached on
// the song. The forced-alignment words come in as upstream's own
// `cursor_words.npy` layout, [n_words, 5] = (start_s, end_s, score, char0,
// char1), with the char offsets in CODEPOINTS into the exact lyrics string the
// prefix was tokenized from.
//
// Layout: T is ggml [L, nF] — element (l, i) at i*L + l — so a frame's
// distribution is one contiguous row, which is what ggml_cross_entropy_loss
// consumes.
struct Yue2AtCursor {
    bool               bound = false;
    int64_t            j0    = 0;  // first lyric-token column of t_H
    int64_t            L     = 0;  // lyric tokens: columns [j0, j0+L)
    int64_t            nF    = 0;  // frames carrying a target: hidden columns [P-1, P-1+nF)
    std::vector<float> T;          // [L * nF]
    std::string        why;        // when !bound: the reason, for the log
};

// GPT-2 byte-level BPE: a token's vocab string is a sequence of "byte-encoder"
// codepoints, one per original byte. Inverting the encoder gives the bytes back;
// counting lead bytes (anything but 10xxxxxx) counts Unicode codepoints in the
// running decode, which is what Python's `len(tok.decode(...))` measures —
// upstream's `offs` are codepoint offsets and the words' char0/char1 are too.
// A token that splits a multi-byte character across itself and the next token
// still counts right cumulatively: the lead byte is counted once, wherever it
// falls, and Python's `errors="replace"` yields one U+FFFD for a dangling valid
// prefix, which is also one.
struct Yue2AtByteDecoder {
    std::unordered_map<std::string, int> str2byte;
    explicit Yue2AtByteDecoder(const BPETokenizer * tok) {
        for (int b = 0; b < 256; b++) {
            str2byte[tok->byte2str[b]] = b;
        }
    }
    // Append the raw bytes of one token; returns the number of codepoints added.
    int64_t append(const BPETokenizer * tok, int id, std::string * bytes) const {
        if (id < 0 || id >= (int) tok->id_to_str.size()) {
            return 0;
        }
        const std::string & s = tok->id_to_str[(size_t) id];
        int64_t             n = 0;
        size_t              p = 0;
        while (p < s.size()) {
            // one UTF-8 codepoint of the encoder string
            const unsigned char c0 = (unsigned char) s[p];
            size_t len = c0 < 0x80 ? 1 : (c0 >> 5) == 0x6 ? 2 : (c0 >> 4) == 0xE ? 3 : 4;
            len        = std::min(len, s.size() - p);
            auto it    = str2byte.find(s.substr(p, len));
            p += len;
            if (it == str2byte.end()) {
                continue;
            }
            const unsigned char b = (unsigned char) it->second;
            bytes->push_back((char) b);
            if ((b & 0xC0) != 0x80) {
                n++;  // a lead byte (or ASCII) starts a codepoint
            }
        }
        return n;
    }
};

// Transcribed from ar_lora_cursor.py:33-49. Every rule that decides a target is
// upstream's, including the two that look like accidents and are not:
//
//   * a lyric token that overlaps no word falls to the token at
//     searchsorted(offs, c0) — upstream's `or [min(len-1, searchsorted)]`;
//   * frames before the first word are attributed to word 0, because
//     `clip(searchsorted(..., side="right") - 1, 0, ...)` clips at 0. The
//     carry-forward rule ("the last word that has started") is the whole
//     mechanism, and it is deliberately blind to word END times.
//
// Returns bound=false, with `why`, when the prefix does not reconstruct from
// head + lyrics — upstream's own silent `return None` at :39, made loud here
// because a run that binds nothing prints "cursor nan" forever and finishes.
// `cot`: which instruction sentence the head text is assembled under. Every
// existing caller trains cot=off and relies on the default; --abc-dropout's
// full-mode cursor twins (yue2-ar-train-run.h) pass YUE2_COT_FULL because
// `full`'s instruction sentence differs from `off`'s (yue2_instruction()), so
// reusing off's would mismatch the actual prefix_full/prefix_full_dropped
// this is asked to bind against. Only the INSTRUCTION differs by cot here —
// the [Tags]/style/[Lyrics]/lyrics shape, and everything this function
// derives from it (j0, L, per-frame word targets), is identical either way.
static Yue2AtCursor yue2_at_cursor_build(const BPETokenizer * tok, const std::string & style,
                                         const std::string & lyrics, const std::vector<int32_t> & prefix,
                                         int64_t n_frames, const std::vector<float> & words5,
                                         int64_t max_frames_hint = 0, Yue2Cot cot = YUE2_COT_OFF) {
    Yue2AtCursor out;
    const int64_t nW = (int64_t) (words5.size() / 5);
    if (nW < 1) {
        out.why = "no words";
        return out;
    }
    // head = INSTRUCTIONS[cot] + "\n[Tags]\n" + style + "\n[Lyrics]\n"   (:35)
    std::string head = yue2_instruction(cot);
    head += "\n[Tags]\n";
    head += style;
    head += "\n[Lyrics]\n";
    std::vector<int> ids_head, ids_full;
    try {
        ids_head = yue2_bpe_encode(tok, head);
        ids_full = yue2_bpe_encode(tok, yue2_assemble_text(style, lyrics, cot));
    } catch (const std::exception & e) {
        out.why = std::string("tokenize: ") + e.what();
        return out;
    }
    const size_t nh = ids_head.size();
    if (ids_full.size() < nh || !std::equal(ids_head.begin(), ids_head.end(), ids_full.begin())) {
        out.why = "the re-tokenised head is not a prefix of head+lyrics (a BPE merge crossed the [Lyrics] "
                  "boundary)";
        return out;
    }
    // prefix[1 : 1+len(ids_full)] == ids_full   (:39) — the +1 is the EOD/BOS.
    if (prefix.size() < 1 + ids_full.size()) {
        out.why = "prefix shorter than head+lyrics";
        return out;
    }
    for (size_t k = 0; k < ids_full.size(); k++) {
        if (prefix[1 + k] != (int32_t) ids_full[k]) {
            out.why = "stored prefix differs from the re-tokenised head+lyrics at token " + std::to_string(k);
            return out;
        }
    }
    const int64_t j0 = 1 + (int64_t) nh;
    const int64_t L  = (int64_t) ids_full.size() - (int64_t) nh;
    if (L < 1) {
        out.why = "no lyric tokens";
        return out;
    }

    // char END offset of each lyric token, in codepoints (:42)
    Yue2AtByteDecoder    dec(tok);
    std::vector<int64_t> offs((size_t) L, 0);
    {
        std::string bytes;
        int64_t     cum = 0;
        for (int64_t k = 0; k < L; k++) {
            cum += dec.append(tok, ids_full[nh + (size_t) k], &bytes);
            offs[(size_t) k] = cum;
        }
    }
    // starts = [0] + offs[:-1]; tok_of_word (:43-44)
    std::vector<std::vector<int32_t>> tok_of_word((size_t) nW);
    for (int64_t w = 0; w < nW; w++) {
        const int64_t c0 = (int64_t) words5[(size_t) w * 5 + 3];
        const int64_t c1 = (int64_t) words5[(size_t) w * 5 + 4];
        std::vector<int32_t> & ks = tok_of_word[(size_t) w];
        for (int64_t k = 0; k < L; k++) {
            const int64_t start = k == 0 ? 0 : offs[(size_t) k - 1];
            if (start < c1 && offs[(size_t) k] > c0) {
                ks.push_back((int32_t) k);
            }
        }
        if (ks.empty()) {
            // np.searchsorted(offs, c0): first k with offs[k] >= c0
            int64_t k = (int64_t) (std::lower_bound(offs.begin(), offs.end(), c0) - offs.begin());
            ks.push_back((int32_t) std::min<int64_t>(L - 1, k));
        }
    }
    // widx[i] = clip(searchsorted(wstart, i/25, side="right") - 1, 0, nW-1)  (:45)
    const int64_t nF = max_frames_hint > 0 ? std::min(n_frames, max_frames_hint) : n_frames;
    if (nF < 1) {
        out.why = "no frames";
        return out;
    }
    std::vector<float> wstart((size_t) nW);
    for (int64_t w = 0; w < nW; w++) {
        wstart[(size_t) w] = words5[(size_t) w * 5 + 0];
    }
    out.T.assign((size_t) (L * nF), 0.0f);
    for (int64_t i = 0; i < nF; i++) {
        const float   t  = (float) i / 25.0f;
        // side="right": number of starts <= t, then -1
        int64_t       w  = (int64_t) (std::upper_bound(wstart.begin(), wstart.end(), t) - wstart.begin()) - 1;
        w                = std::max<int64_t>(0, std::min<int64_t>(nW - 1, w));
        const std::vector<int32_t> & ks = tok_of_word[(size_t) w];
        const float                  v  = 1.0f / (float) ks.size();
        for (int32_t k : ks) {
            out.T[(size_t) (i * L + k)] = v;
        }
    }
    out.bound = true;
    out.j0    = j0;
    out.L     = L;
    out.nF    = nF;
    return out;
}

// One dense [SL, Sc] label block, set sparse and cleared sparse. A dense
// one-hot at the full vocabulary would be 3.3 GB at n_sup = 4501; over the
// slice and one chunk at a time it is 33.6 MB, and it is what
// ggml_cross_entropy_loss's `ggml_are_same_shape(a, b)` assert requires.
// `row_count` defaults to YUE2_AT_SLICE_ROWS for every existing (codec-head)
// call site; yue2_at_head_text_chunked passes V (the full vocabulary) instead,
// since its `rows` are RAW absolute token ids rather than slice-relative ones.
struct Yue2AtLabelGuard {
    ggml_tensor *   t;
    const int32_t * rows;
    int             Sc;
    int64_t         row_count;

    Yue2AtLabelGuard(ggml_tensor * t_, const int32_t * rows_, int Sc_, int64_t row_count_ = YUE2_AT_SLICE_ROWS)
        : t(t_), rows(rows_), Sc(Sc_), row_count(row_count_) {
        for (int i = 0; i < Sc; i++) {
            const size_t off = ((size_t) i * (size_t) row_count + (size_t) rows[i]) * sizeof(float);
            const float  one = 1.0f;
            ggml_backend_tensor_set(t, &one, off, sizeof(float));
        }
    }
    ~Yue2AtLabelGuard() {
        for (int i = 0; i < Sc; i++) {
            const size_t off = ((size_t) i * (size_t) row_count + (size_t) rows[i]) * sizeof(float);
            const float  z   = 0.0f;
            ggml_backend_tensor_set(t, &z, off, sizeof(float));
        }
    }
};

// P5: the chunked, sliced CE head. NO AUTODIFF THROUGH THE HEAD.
//
// Per chunk of CH supervised rows: build [SL, Sc] logits off t_H, call
// ggml_cross_entropy_loss for the reported number and ggml_cross_entropy_loss_back
// for the gradient, project dL/dh = mul_mat(headT, dl) and COPY it into the
// trunk's gradient buffer. lm_ckpt_head_chunked:1082-1165 is the template and
// the loss-scale trick at :1094-1099 is exact and copied verbatim:
//
//     gs = Sc / (n_sup * grad_accum)
//
// because ggml_cross_entropy_loss_back divides by its own nrows == Sc, so the
// product is 1 / (n_sup * grad_accum) — and therefore the trunk surrogate's
// loss gradient is 1.0, NOT 1/grad_accum.
//
// `total_n_sup` (item 3): 0 (the default) means "scale against this head's
// OWN seq.n_sup" — every existing call site, and every `off`-mode sequence,
// so those are BYTE IDENTICAL to before --abc-dropout existed. A `full`
// example's caller passes `seq.n_sup + seq.text_n_sup` here (and to
// yue2_at_head_text_chunked) so the two heads' contributions — this one's
// codec range, the other's sheet range — sum to ONE ordinary mean
// cross-entropy over the whole supervised span, not two independently
// averaged losses.
static bool yue2_at_head_chunked(Yue2AtRun & r, const Yue2AtSeq & seq, bool count_loss, double * ce_out,
                                 int64_t total_n_sup = 0) {
    Yue2AtState & st    = *r.st;
    const int64_t H     = (int64_t) r.m->lm_cfg.embedding_length;
    const int64_t n_sup = seq.n_sup;
    const int64_t TOT   = total_n_sup > 0 ? total_n_sup : n_sup;
    const int64_t CH    = st.chunk;
    const int64_t GA    = std::max<int64_t>(1, r.grad_accum);

    // The supervised span's first hidden column. P-1 for every real run; the
    // control shifts it and leaves the targets alone (contract §6.5).
    const int64_t S    = (int64_t) seq.ids.size();
    const int64_t col0 = seq.prefix - 1 + (int64_t) r.sup_shift;
    if (col0 < 0 || col0 + n_sup > S) {
        fprintf(stderr, "[yue2-at] supervised span [%lld, %lld) with shift %d falls outside the %lld-token "
                        "sequence\n",
                (long long) col0, (long long) (col0 + n_sup), r.sup_shift, (long long) S);
        return false;
    }

    double ce      = 0.0;
    double ce_host = 0.0;
    std::vector<float> lg_host;
    for (int64_t i = 0; i < n_sup; i += CH) {
        const int64_t Sc = std::min(CH, n_sup - i);

        // `r.lossgrad` is dL/dloss and this is the ONLY place it enters the
        // backward — see the field's comment on Yue2AtRun. 1.0 for every real
        // run, so this multiply is a no-op outside the FD gate's negative
        // control. The REPORTED loss (`lc`, read below) is deliberately
        // untouched by it: the control must move the gradient and nothing else,
        // or the probe's numeric arm would move with it and hide the defect.
        const float gs = r.lossgrad * (float) Sc / ((float) TOT * (float) GA);
        ggml_backend_tensor_set(st.t_gs, &gs, 0, sizeof(float));

        Yue2AtLabelGuard guard(st.t_labc, seq.rows.data() + i, (int) Sc);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), /*no_alloc*/ true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, /*grads=*/false);

        // Column of t_H the chunk starts at: the supervised span begins at
        // row P-1 (the hidden state that predicts ids[P]).
        const size_t  col = (size_t) (col0 + i);
        ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, Sc, st.t_H->nb[1], col * st.t_H->nb[1]));
        ggml_tensor * lg  = ggml_mul_mat(ctx, st.t_head, hd);  // [SL, Sc]
        ggml_tensor * lb  = ggml_view_2d(ctx, st.t_labc, YUE2_AT_SLICE_ROWS, Sc, st.t_labc->nb[1], 0);
        ggml_tensor * lc  = ggml_cross_entropy_loss(ctx, lg, lb);  // scalar, mean over Sc rows
        ggml_set_output(lc);
        ggml_build_forward_expand(gf, lc);
        if (r.host_ce) {
            // The logits are an intermediate; without this the arena is free to
            // reuse their memory before the readback.
            ggml_set_output(lg);
            ggml_build_forward_expand(gf, lg);
        }
        if (!r.forward_only) {
            ggml_tensor * dl = ggml_cross_entropy_loss_back(ctx, st.t_gs, lg, lb);  // (grad, logits, labels)
            ggml_tensor * dh = ggml_mul_mat(ctx, st.t_headT, dl);                   // [H, Sc]
            ggml_tensor * gv = ggml_view_2d(ctx, st.Gh[0], H, Sc, st.Gh[0]->nb[1], col * st.Gh[0]->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, dh, gv));
        }

        ggml_backend_sched_reset(r.sched);
        const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && count_loss) {
            float lv = 0.0f;
            ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
            ce += (double) lv * (double) Sc / (double) TOT;
        }
        if (ok && r.host_ce) {
            lg_host.assign((size_t) (YUE2_AT_SLICE_ROWS * Sc), 0.0f);
            ggml_backend_tensor_get(lg, lg_host.data(), 0, lg_host.size() * sizeof(float));
            for (int64_t s = 0; s < Sc; s++) {
                const float * row = lg_host.data() + (size_t) s * (size_t) YUE2_AT_SLICE_ROWS;
                double        mx  = (double) row[0];
                for (int64_t j = 1; j < YUE2_AT_SLICE_ROWS; j++) {
                    mx = std::max(mx, (double) row[j]);
                }
                double sum = 0.0;
                for (int64_t j = 0; j < YUE2_AT_SLICE_ROWS; j++) {
                    sum += std::exp((double) row[j] - mx);
                }
                ce_host += mx + std::log(sum) - (double) row[seq.rows[(size_t) (i + s)]];
            }
        }
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }
    if (count_loss && ce_out) {
        *ce_out = ce;
    }
    if (r.host_ce) {
        *r.host_ce = ce_host / (double) n_sup;
    }
    return true;
}

// P5b: the chunked, FULL-VOCABULARY, TRAINABLE head for a `full` example's
// TEXT span (item 3: sheet tokens, ABC_END, MUSIC_START — everything
// yue2_at_build_sequence recorded as seq.text_sup0/seq.text_n_sup). Same shape
// as yue2_at_head_chunked (P5) above — chunked CE + CE-back, dh projected and
// copied into the SAME Gh[0] gradient buffer P5 writes, at the DISJOINT column
// range [text_sup0, text_sup0+text_n_sup) immediately before P5's own col0 —
// but reading the model's FULL output projection instead of the codec slice,
// because sheet-token targets range over the whole ordinary vocabulary (BPE
// ids of the rendered ABC text) plus two special ids (ABC_END, MUSIC_START)
// that sit just outside it: not a contiguous span the way codec ids are.
//
// Forward reads `m.lm.output` in its own NATIVE dtype (no F32 mirror kept —
// exactly the eval-only yue2_at_head_fullvocab_ce below); only the backward's
// dh = mul_mat(headT, dl) needs a vocab-fastest copy, which is `st.t_full_headT`
// (built once, yue2_at_build_text_head). `total_n_sup` is documented on
// yue2_at_head_chunked above: both heads scale against the SAME total so their
// two contributions sum to one ordinary mean CE.
//
// The FD gate's `sup_shift` control (contract §6.5) is NOT applied here: it
// only ever exercises synthetic off-mode sequences (seq.text_n_sup == 0), so
// this function is never on its path.
static bool yue2_at_head_text_chunked(Yue2AtRun & r, const Yue2AtSeq & seq, bool count_loss, int64_t total_n_sup,
                                      double * ce_out, std::string * why = nullptr) {
    Yue2AtState &     st    = *r.st;
    const Yue2Model & m     = *r.m;
    const int64_t     H     = (int64_t) m.lm_cfg.embedding_length;
    const int64_t     V     = (int64_t) m.lm_cfg.vocab_size;
    const int64_t     n_txt = seq.text_n_sup;
    const int64_t     CH    = st.chunk_text;
    const int64_t     GA    = std::max<int64_t>(1, r.grad_accum);
    const int64_t     TOT   = total_n_sup > 0 ? total_n_sup : n_txt;

    if (!st.t_full_headT || !st.t_labc_full) {
        return false;  // caller names the error: state wasn't allocated with want_text_head
    }
    const int64_t col0 = seq.text_sup0;
    const int64_t S    = (int64_t) seq.ids.size();
    if (n_txt < 1 || col0 < 0 || col0 + n_txt >= S) {
        if (why) *why = "text supervision range out of the sequence: text_sup0=" + std::to_string(col0)
                      + " text_n_sup=" + std::to_string(n_txt) + " S=" + std::to_string(S);
        return false;
    }

    double                ce = 0.0;
    std::vector<int32_t>  tgt;
    for (int64_t i = 0; i < n_txt; i += CH) {
        const int64_t Sc = std::min(CH, n_txt - i);
        tgt.resize((size_t) Sc);
        for (int64_t s = 0; s < Sc; s++) {
            // RAW absolute vocab id — no slice offset: row (col0+i+s) predicts
            // ids[col0+i+s+1], the next sheet/ABC_END/MUSIC_START token.
            tgt[(size_t) s] = seq.ids[(size_t) (col0 + i + s + 1)];
        }

        const float gs = r.lossgrad * (float) Sc / ((float) TOT * (float) GA);
        ggml_backend_tensor_set(st.t_gs, &gs, 0, sizeof(float));

        Yue2AtLabelGuard guard(st.t_labc_full, tgt.data(), (int) Sc, V);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), /*no_alloc*/ true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, /*grads=*/false);

        const size_t  col = (size_t) (col0 + i);
        ggml_tensor * hd  = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, Sc, st.t_H->nb[1], col * st.t_H->nb[1]));
        ggml_tensor * lg  = ggml_mul_mat(ctx, m.lm.output, hd);  // [V, Sc] — native weight x F32 activation
        ggml_tensor * lb  = ggml_view_2d(ctx, st.t_labc_full, V, Sc, st.t_labc_full->nb[1], 0);
        ggml_tensor * lc  = ggml_cross_entropy_loss(ctx, lg, lb);
        ggml_set_output(lc);
        ggml_build_forward_expand(gf, lc);
        if (!r.forward_only) {
            ggml_tensor * dl = ggml_cross_entropy_loss_back(ctx, st.t_gs, lg, lb);
            ggml_tensor * dh = ggml_mul_mat(ctx, st.t_full_headT, dl);  // [H, Sc]
            ggml_tensor * gv = ggml_view_2d(ctx, st.Gh[0], H, Sc, st.Gh[0]->nb[1], col * st.Gh[0]->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, dh, gv));
        }

        ggml_backend_sched_reset(r.sched);
        const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok && count_loss) {
            float lv = 0.0f;
            ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
            ce += (double) lv * (double) Sc / (double) TOT;
        }
        ggml_free(ctx);
        if (!ok) {
            if (why) *why = "ggml_backend_sched_graph_compute failed on text chunk rows " + std::to_string(i)
                          + ".." + std::to_string(i + Sc) + " of " + std::to_string(n_txt) + " (S=" + std::to_string(S)
                          + ", V=" + std::to_string(V) + ") — usually an allocation failure; check VRAM";
            return false;
        }
    }
    if (count_loss && ce_out) {
        *ce_out = ce;
    }
    return true;
}

// The FULL-VOCAB CE, forward only, for evals (contract §3.3).
//
// Runs against the base head in its NATIVE type — no F32 copy, because nothing
// here needs a gradient, and a 184,704 x 2048 F32 mirror would be 1.5 GB. The
// dense label block is the expensive part (~95 MB per 128-row chunk), so this
// is deliberately an EVAL-only path: it is 5.6x the sliced head's cost and the
// only reason it exists is that our sliced CE numbers are not comparable to
// upstream's README, and minted_val loss is the thing to watch.
static bool yue2_at_head_fullvocab_ce(Yue2AtRun & r, const Yue2AtSeq & seq, int64_t chunk, double * ce_out) {
    Yue2AtState &     st    = *r.st;
    const Yue2Model & m     = *r.m;
    const int64_t     H     = (int64_t) m.lm_cfg.embedding_length;
    const int64_t     V     = (int64_t) m.lm_cfg.vocab_size;
    const int64_t     n_sup = seq.n_sup;
    const int64_t     CH    = std::max<int64_t>(1, chunk);

    double             ce = 0.0;
    std::vector<float> lg_host;
    for (int64_t i = 0; i < n_sup; i += CH) {
        const int64_t Sc = std::min(CH, n_sup - i);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), /*no_alloc*/ true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
        const size_t     col = (size_t) (seq.prefix - 1 + (int64_t) r.sup_shift + i);
        ggml_tensor *    hd =
            ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, Sc, st.t_H->nb[1], col * st.t_H->nb[1]));
        ggml_tensor * lg = ggml_mul_mat(ctx, m.lm.output, hd);  // [V, Sc]
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, lg);
        ggml_backend_sched_reset(r.sched);
        const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            lg_host.assign((size_t) (V * Sc), 0.0f);
            ggml_backend_tensor_get(lg, lg_host.data(), 0, lg_host.size() * sizeof(float));
            for (int64_t s = 0; s < Sc; s++) {
                const float * row = lg_host.data() + (size_t) s * (size_t) V;
                const int64_t tgt = YUE2_AT_SLICE_ROW0 + (int64_t) seq.rows[(size_t) (i + s)];
                double        mx  = (double) row[0];
                for (int64_t j = 1; j < V; j++) {
                    mx = std::max(mx, (double) row[j]);
                }
                double sum = 0.0;
                for (int64_t j = 0; j < V; j++) {
                    sum += std::exp((double) row[j] - mx);
                }
                ce += mx + std::log(sum) - (double) row[tgt];
            }
        }
        ggml_free(ctx);
        if (!ok) {
            return false;
        }
    }
    if (ce_out) {
        *ce_out = ce / (double) n_sup;
    }
    return true;
}

// ── P6: the lyric-cursor term (ar_lora_cursor.py:62-67) ────────────────────
//
// Upstream, with hn the post-final-norm hidden states, h = hn[Lp-1 : -1],
// nF = min(frames, len(h)):
//
//     q    = cursor_head(h[:nF])                 [nF, H]   (nn.Linear, no bias)
//     kh   = hn[j0:j1]                           [L, H]    the lyric tokens
//     sc   = (q @ kh.T) * H^-0.5                 [nF, L]
//     loss = -(T * log_softmax(sc)).sum(-1).mean()
//
// and total = lm + CUR_W * loss, over ACC micro-steps.
//
// Both operands live in t_H already: the supervised columns [P-1, P-1+nF) and
// the lyric columns [j0, j0+L) — which the head never touches, and which is why
// this term is not a bolt-on to the head but a second seed into the SAME Gh[0]
// buffer. Gh[0] is [H, S_MAX], the trunk backward runs over all S columns, so
// gradient landing in the lyric-prefix columns flows back through every layer
// exactly as the supervised columns' does. Nothing about the segment backward
// changes.
//
// NO AUTODIFF, same as the head: forward in ggml for the reported number,
// backward by hand. Written out because four transposes is exactly where the
// V-gradient bug hid in the NAR trainer, and the FD gate probes `cursor_head.W`
// for that reason:
//
//     R    = mul_mat(hk, q)                      ggml [L, nF],  R[l,i] = Σ_h hk[h,l] q[h,i]
//     sc   = s·R,  s = H^-0.5
//     dsc  = CE_back(gs, sc, T)                  = gs·(softmax(sc) - T)/nF   (mean's 1/nF is inside)
//     dR   = s·dsc
//     dq   = mul_mat(hkᵀ, dR)                    [H, nF],  dq[h,i]  = Σ_l hk[h,l] dR[l,i]
//     dhk  = mul_mat(qᵀ, dRᵀ)                    [H, L],   dhk[h,l] = Σ_i q[h,i]  dR[l,i]
//     dhq  = mul_mat(Wᵀ, dq)                     [H, nF],  dhq[h,i] = Σ_o W[h,o]  dq[o,i]
//     dW   = mul_mat(hqᵀ, dqᵀ)                   [H, H],   dW[h,o]  = Σ_i hq[h,i] dq[o,i]
//
// with q = mul_mat(W, hq), W ggml [H_in, H_out] (torch's Linear weight [out, in]
// transposed by ggml's ne order, which is what makes identity init the same
// tensor in both). gs = lossgrad · CUR_W / grad_accum: CE_back supplies the
// mean's 1/nF, the head's `gs` convention supplies the 1/grad_accum, and
// lossgrad is the FD gate's negative control, which must reach this term too
// or the gate could not fail on it.
//
// Gh[0] columns [P-1, P-1+nF) already hold the head's dL/dh, so dhq is ADDED;
// [j0, j0+L) are zero (P4 cleared the buffer) and are added for uniformity.
static bool yue2_at_cursor_term(Yue2AtRun & r, const Yue2AtSeq & seq, bool count_loss, double * cur_out) {
    Yue2AtState &        st = *r.st;
    const Yue2AtCursor & cu = *r.cur;
    const int64_t        H  = (int64_t) r.m->lm_cfg.embedding_length;
    const int64_t        S  = (int64_t) seq.ids.size();
    const int64_t        GA = std::max<int64_t>(1, r.grad_accum);
    const int64_t        col0 = seq.prefix - 1 + (int64_t) r.sup_shift;
    const int64_t        L  = cu.L, nF = std::min(cu.nF, seq.n_sup);

    if (!st.t_curT || L > st.cur_l_max || nF < 1 || col0 < 0 || col0 + nF > S || cu.j0 + L > seq.prefix) {
        fprintf(stderr, "[yue2-at] cursor: targets [L %lld, nF %lld] do not fit (L_max %lld, S %lld, P %lld)\n",
                (long long) L, (long long) nF, (long long) st.cur_l_max, (long long) S, (long long) seq.prefix);
        return false;
    }
    if (st.last_cur != (const void *) &cu) {
        ggml_backend_tensor_set(st.t_curT, cu.T.data(), 0, (size_t) (L * nF) * sizeof(float));
        st.last_cur = &cu;
    }
    const float s  = 1.0f / std::sqrt((float) H);
    const float gs = r.lossgrad * r.cursor_weight / (float) GA;
    ggml_backend_tensor_set(st.t_gs, &gs, 0, sizeof(float));

    ggml_init_params ip  = { st.arena.size(), st.arena.data(), /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, /*grads=*/false);

    ggml_tensor * hq = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, nF, st.t_H->nb[1], (size_t) col0 * st.t_H->nb[1]));
    ggml_tensor * hk = ggml_cont(ctx, ggml_view_2d(ctx, st.t_H, H, L, st.t_H->nb[1], (size_t) cu.j0 * st.t_H->nb[1]));
    ggml_tensor * T  = ggml_view_2d(ctx, st.t_curT, L, nF, (size_t) L * sizeof(float), 0);
    ggml_tensor * q  = ggml_mul_mat(ctx, r.cur_w, hq);           // [H, nF]
    ggml_tensor * sc = ggml_scale(ctx, ggml_mul_mat(ctx, hk, q), s);  // [L, nF]
    ggml_tensor * lc = ggml_cross_entropy_loss(ctx, sc, T);       // scalar, mean over nF
    ggml_set_output(lc);
    ggml_build_forward_expand(gf, lc);
    if (r.host_cur) {
        ggml_set_output(hq);
        ggml_set_output(hk);
        ggml_build_forward_expand(gf, hq);
        ggml_build_forward_expand(gf, hk);
    }
    if (!r.forward_only) {
        auto slot = r.opt->param_slot.find(r.cur_w);
        GGML_ASSERT(slot != r.opt->param_slot.end());
        ggml_tensor * accW = r.opt->acc[(size_t) slot->second];

        ggml_tensor * dR  = ggml_scale(ctx, ggml_cross_entropy_loss_back(ctx, st.t_gs, sc, T), s);  // [L, nF]
        ggml_tensor * dq  = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, hk)), dR);          // [H, nF]
        ggml_tensor * dhk = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, q)),
                                         ggml_cont(ctx, ggml_transpose(ctx, dR)));                    // [H, L]
        ggml_tensor * dhq = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, r.cur_w)), dq);     // [H, nF]
        ggml_tensor * dW  = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, hq)),
                                         ggml_cont(ctx, ggml_transpose(ctx, dq)));                    // [H, H]

        ggml_tensor * gq = ggml_view_2d(ctx, st.Gh[0], H, nF, st.Gh[0]->nb[1], (size_t) col0 * st.Gh[0]->nb[1]);
        ggml_tensor * gk = ggml_view_2d(ctx, st.Gh[0], H, L, st.Gh[0]->nb[1], (size_t) cu.j0 * st.Gh[0]->nb[1]);
        // read-then-write on the same view inside one graph: the add depends
        // on the view, the cpy depends on the add, so the order is forced.
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, gq, dhq), gq));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, gk, dhk), gk));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_add(ctx, accW, dW), accW));
    }

    ggml_backend_sched_reset(r.sched);
    const bool ok = ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
    if (ok && count_loss && cur_out) {
        float lv = 0.0f;
        ggml_backend_tensor_get(lc, &lv, 0, sizeof(float));
        *cur_out = (double) lv;
    }
    if (ok && r.host_cur) {
        // Independent, in double: download the two operands and the head, redo
        // the forward on the host. The FD gate differences this, so it must
        // not be the same f32 kernel twice.
        std::vector<float> Hq((size_t) (H * nF)), Hk((size_t) (H * L)), W((size_t) (H * H));
        ggml_backend_tensor_get(hq, Hq.data(), 0, Hq.size() * sizeof(float));
        ggml_backend_tensor_get(hk, Hk.data(), 0, Hk.size() * sizeof(float));
        ggml_backend_tensor_get(r.cur_w, W.data(), 0, W.size() * sizeof(float));
        std::vector<double> qd((size_t) H), scd((size_t) L);
        double              tot = 0.0;
        for (int64_t i = 0; i < nF; i++) {
            const float * x = Hq.data() + (size_t) i * (size_t) H;
            for (int64_t o = 0; o < H; o++) {
                double acc = 0.0;
                const float * wcol = W.data() + (size_t) o * (size_t) H;  // W[h,o] at o*H + h
                for (int64_t h = 0; h < H; h++) {
                    acc += (double) wcol[h] * (double) x[h];
                }
                qd[(size_t) o] = acc;
            }
            double mx = -1e300;
            for (int64_t l = 0; l < L; l++) {
                const float * k = Hk.data() + (size_t) l * (size_t) H;
                double        d = 0.0;
                for (int64_t h = 0; h < H; h++) {
                    d += (double) k[h] * qd[(size_t) h];
                }
                scd[(size_t) l] = d * (double) s;
                mx              = std::max(mx, scd[(size_t) l]);
            }
            double z = 0.0;
            for (int64_t l = 0; l < L; l++) {
                z += std::exp(scd[(size_t) l] - mx);
            }
            const double lz = mx + std::log(z);
            for (int64_t l = 0; l < L; l++) {
                const float t = cu.T[(size_t) (i * L + l)];
                if (t != 0.0f) {
                    tot -= (double) t * (scd[(size_t) l] - lz);
                }
            }
        }
        *r.host_cur = tot / (double) nF;
    }
    ggml_free(ctx);
    return ok;
}

// ── The micro-step (contract §4.3, lm-ckpt.h:15-30's shape) ────────────────
//
//   P0  upload ids, positions and the causal mask
//   P1  embedding                 -> C[0]                      grads = false
//   P2  collect  C[l] -> C[l+1],  l = 0 .. L-2                 grads = false
//   P3  tail forward: layer L-1 + final norm -> t_H            grads = false
//   P4  t_G := 0   (masked columns must stay EXACTLY zero)
//   P5  the chunked CE head writes dL/dh into t_G, no autodiff
//   P6  the lyric-cursor term ADDS its dL/dh into t_G — supervised AND lyric
//       columns — and its dW into the cursor head's accumulator, no autodiff
//   P7  backward segments l = L-1 .. 0: recompute the layer from C[l], build
//       the surrogate SUM(Y (.) dY), backward it, accumulate LoRA grads into
//       opt.acc[] and dL/dC[l] into the other Gh ping-pong buffer.
//
// WHY THE SURROGATE IS EXACT: ggml_sum's backward is repeat(grad); with
// grad == t_one == 1.0 that is an all-ones [H,S], and ggml_mul's backward
// w.r.t. Y is mul(ones, dY) = dY — exact in fp32. So the segment sees precisely
// the upstream gradient and the parameter gradients equal a whole-graph
// backward's by the chain rule.
//
// D13: the recomputed forward MUST use the same `r.opts` as the collect pass.
// It does, structurally: `opts` is read once, here, and passed to every call.
static bool yue2_at_micro_step(Yue2AtRun & r, const Yue2AtSeq & seq, bool count_loss, double * ce_out,
                               std::string * err, double * cur_out = nullptr) {
    const Yue2Model &    m  = *r.m;
    Yue2AtState &        st = *r.st;
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        H  = (int64_t) c.embedding_length;
    const int64_t        S  = (int64_t) seq.ids.size();
    const int            L  = r.n_layers > 0 ? r.n_layers : (int) c.block_count;

    if (S > st.s_max) {
        if (err) {
            *err = "sequence of " + std::to_string((long long) S) + " tokens exceeds the allocated s_max of " +
                   std::to_string((long long) st.s_max);
        }
        return false;
    }
    GGML_ASSERT(seq.prefix >= 1 && seq.n_sup >= 1);

    // ── P0: inputs ───────────────────────────────────────────────────────
    ggml_backend_tensor_set(st.t_ids, seq.ids.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(st.t_pos, st.pos_host.data(), 0, (size_t) S * sizeof(int32_t));
    if (st.last_mask_S != S) {
        yue2_at_fill_mask(&st.mask_host, S);
        ggml_backend_tensor_set(st.t_msk, st.mask_host.data(), 0, st.mask_host.size() * sizeof(uint16_t));
        st.last_mask_S = S;
    }

    const Yue2AtOpts opts = r.opts;
    auto             mask_view = [&](ggml_context * ctx) {
        return ggml_view_2d(ctx, st.t_msk, S, S, (size_t) S * ggml_element_size(st.t_msk), 0);
    };
    auto run_graph = [&](ggml_cgraph * gf) -> bool {
        ggml_backend_sched_reset(r.sched);
        return ggml_backend_sched_graph_compute(r.sched, gf) == GGML_STATUS_SUCCESS;
    };

    // ── P1: embedding -> C[0] ────────────────────────────────────────────
    //
    // get_rows on a BF16 source yields F32 (an exact upcast) and needs no
    // backward: token_embd is frozen and GET_ROWS' row indices are on ggml's
    // non-differentiable list.
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, /*grads=*/false);
        ggml_tensor *    h0 = ggml_get_rows(ctx, m.lm.token_embd, ggml_view_1d(ctx, st.t_ids, S, 0));
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx, h0, ggml_view_2d(ctx, st.C[0], H, S, st.C[0]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            if (err) {
                *err = "P1 (embedding) failed";
            }
            return false;
        }
    }

    // ── P2: forward-collect, l = 0 .. L-2 ────────────────────────────────
    for (int l = 0; l < L - 1; l++) {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, yue2_at_forward_nodes(), /*grads=*/false);
        ggml_tensor *    X   = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        ggml_tensor *    pv  = ggml_view_1d(ctx, st.t_pos, S, 0);
        ggml_tensor *    Y = yue2_at_block(ctx, c, m.lm.blk[(size_t) l], r.ad->blk[(size_t) l], X, pv,
                                           mask_view(ctx), opts);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Y,
                                               ggml_view_2d(ctx, st.C[(size_t) (l + 1)], H, S,
                                                            st.C[(size_t) (l + 1)]->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            if (err) {
                *err = "P2 (forward collect) failed at layer " + std::to_string(l);
            }
            return false;
        }
    }

    // ── P3: tail forward (layer L-1 + final norm) -> t_H ─────────────────
    {
        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, yue2_at_forward_nodes(), /*grads=*/false);
        ggml_tensor *    X =
            ggml_view_2d(ctx, st.C[(size_t) (L - 1)], H, S, st.C[(size_t) (L - 1)]->nb[1], 0);
        ggml_tensor * pv = ggml_view_1d(ctx, st.t_pos, S, 0);
        ggml_tensor * Y  = yue2_at_block(ctx, c, m.lm.blk[(size_t) (L - 1)], r.ad->blk[(size_t) (L - 1)], X,
                                         pv, mask_view(ctx), opts);
        ggml_tensor * hN = yue2_at_rms(ctx, Y, m.lm.output_norm, c.rms_eps, opts);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, hN, ggml_view_2d(ctx, st.t_H, H, S, st.t_H->nb[1], 0)));
        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            if (err) {
                *err = "P3 (tail forward) failed";
            }
            return false;
        }
    }

    // ── P4: t_G := 0 ─────────────────────────────────────────────────────
    if (!r.forward_only) {
        ggml_backend_buffer_clear(st.buf_gh0, 0);
    }

    // ── P5: the CE head(s) ───────────────────────────────────────────────
    //
    // `off` (seq.text_n_sup == 0): P5 alone, scaled against its own n_sup —
    // BYTE IDENTICAL to before --abc-dropout existed. `full`: P5 (codec range)
    // and P5b (text range, item 3's sheet-token loss) each contribute their
    // slice of ONE combined mean CE over total_n_sup = n_sup + text_n_sup, and
    // their dh writes land in disjoint, contiguous columns of the SAME Gh[0]
    // P4 just cleared — see yue2_at_head_chunked's and
    // yue2_at_head_text_chunked's own comments for the full argument.
    {
        const int64_t total_sup  = seq.n_sup + seq.text_n_sup;
        double        codec_ce   = 0.0;
        double        text_ce    = 0.0;
        if (!yue2_at_head_chunked(r, seq, count_loss, &codec_ce, total_sup)) {
            if (err) {
                *err = "P5 (CE head) failed";
            }
            return false;
        }
        if (seq.text_n_sup > 0) {
            if (!st.t_full_headT || !st.t_labc_full) {
                if (err) {
                    *err = "P5b (text CE head) failed: this run's state has no full-vocab head — it was "
                           "allocated without want_text_head, but this example draws \"full\"";
                }
                return false;
            }
            std::string why;
            if (!yue2_at_head_text_chunked(r, seq, count_loss, total_sup, &text_ce, &why)) {
                if (err) {
                    *err = "P5b (text CE head) failed: " + (why.empty() ? std::string("(no detail)") : why);
                }
                return false;
            }
        }
        if (count_loss && ce_out) {
            *ce_out = codec_ce + text_ce;
        }
    }

    // ── P6: the lyric-cursor term ────────────────────────────────────────
    //
    // Skipped for a plain forward-only eval (upstream's evaluate() reports the
    // LM loss alone); run for a training micro-step, and for the FD gate's
    // numeric arm, which is forward-only but asks for the host-double value.
    if (cur_out) {
        *cur_out = std::nan("");
    }
    if (r.cur && r.cur->bound && r.cursor_weight > 0.0f && r.cur_w && (!r.forward_only || r.host_cur)) {
        if (!yue2_at_cursor_term(r, seq, count_loss, cur_out)) {
            if (err) {
                *err = "P6 (lyric-cursor term) failed";
            }
            return false;
        }
    } else if (r.host_cur) {
        *r.host_cur = 0.0;
    }
    if (r.forward_only) {
        return true;  // evaluation stops here: the loss is what it came for
    }

    // ── P7: backward segments, l = L-1 .. 0 ──────────────────────────────
    int cur = 0;  // Gh[0] == t_G, already seeded by the head
    for (int l = L - 1; l >= 0; l--) {
        const int nxt = 1 - cur;
        // MANDATORY: ggml_acc_or_set accumulates INTO a supplied grad_acc, so
        // the destination must start at zero.
        ggml_backend_buffer_clear(nxt == 0 ? st.buf_gh0 : st.buf_gh1, 0);

        ggml_init_params ip  = { st.arena.size(), st.arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, yue2_at_segment_nodes(), /*grads=*/true);

        ggml_tensor * X  = ggml_view_2d(ctx, st.C[(size_t) l], H, S, st.C[(size_t) l]->nb[1], 0);
        ggml_tensor * pv = ggml_view_1d(ctx, st.t_pos, S, 0);
        ggml_tensor * Y =
            yue2_at_block(ctx, c, m.lm.blk[(size_t) l], r.ad->blk[(size_t) l], X, pv, mask_view(ctx), opts);
        if (l == L - 1) {
            Y = yue2_at_rms(ctx, Y, m.lm.output_norm, c.rms_eps, opts);
        }
        ggml_tensor * dY   = ggml_view_2d(ctx, st.Gh[cur], H, S, st.Gh[cur]->nb[1], 0);
        ggml_tensor * Lsur = ggml_sum(ctx, ggml_mul(ctx, Y, dY));  // L' = SUM(Y (.) dY)
        ggml_set_loss(Lsur);
        ggml_set_output(Lsur);
        ggml_build_forward_expand(gf, Lsur);

        // fill_gacc is indexed by FORWARD-node order, so it must run after
        // forward_expand and before backward_expand. Getting that order wrong
        // is silent.
        yue2_at_fill_gacc(r.opt, gf, st.C[(size_t) l], st.Gh[nxt], st.t_one, &st.gacc);
        ggml_build_backward_expand(ctx, gf, st.gacc.data());

        const bool ok = run_graph(gf);
        ggml_free(ctx);
        if (!ok) {
            if (err) {
                *err = "P7 (backward segment) failed at layer " + std::to_string(l);
            }
            return false;
        }
        cur = nxt;
    }
    // Gh[cur] now holds dL/d(embedding output). It is DISCARDED: token_embd is
    // frozen and carries no LoRA site (contract §2.3).
    return true;
}

// ── The VRAM model (contract §4.2) ─────────────────────────────────────────
//
// COUNTED, NOT MEASURED. The term trusted least is the per-token activation
// figure: a hand count of the tensors yue2_at_block retains plus the LoRA
// branch's intermediates, assuming the arena reuses NONE of them inside one
// graph. Real arenas reuse some, so it is an UPPER BOUND, which is the correct
// direction for a VRAM estimate (flash-attn-training skill §5: over-predict,
// never under). A run logs the measured arena total alongside it so this
// estimate can be corrected from evidence rather than argued about.
struct Yue2AtVram {
    double base_gib     = 0.0;  // the resident GGUF
    double lora_gib     = 0.0;  // factors + grads + AdamW m,v
    double window_gib   = 0.0;  // one layer's F32 weight window
    double mask_gib     = 0.0;  // the F16 causal mask
    double ckpt_gib     = 0.0;  // C[0..L-1]
    double act_gib      = 0.0;  // one segment's live activations
    double gh_gib       = 0.0;  // the Gh ping-pong
    double head_gib     = 0.0;  // the sliced head + its transpose
    double labels_gib   = 0.0;  // one chunk of logits + dense labels
    double softmax_gib  = 0.0;  // the retained [S,S,Nh] softmax, exact mode only
    double total_gib    = 0.0;  // the sum, before allocator slack
    double with_slack_gib = 0.0;  // + 25 %
};

static Yue2AtVram yue2_at_vram(const Yue2LmConfig & c, int64_t S, int64_t rank, Yue2AtTarget target,
                               Yue2AtAttnMode attn, int64_t chunk, double base_bytes) {
    const double  G   = 1024.0 * 1024.0 * 1024.0;
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t L   = (int64_t) c.block_count;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Q   = Nh * (int64_t) c.key_length;
    const int64_t K   = (int64_t) c.head_count_kv * (int64_t) c.key_length;
    const int64_t F   = (int64_t) c.feed_forward_length;
    const int64_t lora = yue2_at_param_count(c, (int) L, rank, target);
    // 7 projections + 4 norms, F32, one layer.
    const double  wlayer = (double) (2 * H * Q + 2 * H * K + 3 * H * F + 2 * H + 2 * (int64_t) c.key_length) * 4.0;
    // Hand count of the retained per-token activations in one block, F32.
    const double  per_tok = (double) (12 * H + 2 * Q + 2 * K + 4 * F + 8 * rank) * 4.0;

    Yue2AtVram v;
    v.base_gib   = base_bytes / G;
    v.lora_gib   = (double) lora * 4.0 * 4.0 / G;  // param + grad + m + v
    v.window_gib = wlayer / G;
    v.mask_gib   = (double) (2 * S * S) / G;
    v.ckpt_gib   = (double) (L * S * H * 4) / G;
    v.act_gib    = (double) S * per_tok / G;
    v.gh_gib     = (double) (3 * S * H * 4) / G;  // Gh[0], Gh[1], t_H
    v.head_gib   = (double) (2 * H * YUE2_AT_SLICE_ROWS * 4) / G;
    v.labels_gib = (double) (2 * YUE2_AT_SLICE_ROWS * chunk * 4) / G;
    v.softmax_gib = (attn == YUE2_AT_FA_EXACT) ? (double) (S * S * Nh * 4) / G : 0.0;
    v.total_gib  = v.base_gib + v.lora_gib + v.window_gib + v.mask_gib + v.ckpt_gib + v.act_gib + v.gh_gib +
                  v.head_gib + v.labels_gib + v.softmax_gib;
    v.with_slack_gib = v.total_gib * 1.25;
    return v;
}

static void yue2_at_vram_report(const char * tag, const Yue2AtVram & v, int64_t S, Yue2AtAttnMode attn) {
    fprintf(stderr,
            "[%s] VRAM model at S=%lld, --attn %s (COUNTED, not measured; the activation term is an upper "
            "bound):\n"
            "[%s]   base GGUF %.2f + LoRA/AdamW %.2f + F32 layer window %.2f + mask %.2f + boundaries %.2f\n"
            "[%s]   + segment activations %.2f + Gh/t_H %.2f + head slice %.2f + label chunk %.2f%s\n"
            "[%s]   = %.2f GiB, %.2f GiB with 25%% allocator slack (target: a 32 GB card, ~30 GiB usable)\n",
            tag, (long long) S, yue2_at_attn_name(attn), tag, v.base_gib, v.lora_gib, v.window_gib, v.mask_gib,
            v.ckpt_gib, tag, v.act_gib, v.gh_gib, v.head_gib, v.labels_gib,
            v.softmax_gib > 0.0 ? "" : " (no retained softmax: fused attention)", tag, v.total_gib,
            v.with_slack_gib);
    if (v.softmax_gib > 0.0) {
        fprintf(stderr,
                "[%s]   the retained [S,S,Nh] softmax alone is %.2f GiB — it is the quadratic term, and "
                "--attn flash removes it\n",
                tag, v.softmax_gib);
    }
}
