#pragma once
// train/yue2-nar-train-graph.h — trainable twin of the YuE2 NAR flow network.
//
// HOT-Step file. TRAINING-SIDE: built into ace-train only, never into
// ace-server. Phase 2 of docs/plans/yue2/08-nar-lora-trainer.md, implementing
// docs/plans/yue2/09-nar-train-graph-contract.md, which is the authority for
// every shape and op choice below and cites the file:line each one was read
// out of. Where a comment here says "contract §N", that is the section to
// re-read before changing anything.
//
// Upstream recipe being ported (the RECIPE, not the code):
// ComfyUI-YuE2-Trainer `trainer_core/train.py` (`training_velocity`,
// `sample_t`, the MSE loss) over `trainer_core/yue2_ref/modeling_yue2.py`
// (`nar_velocity`, `_shift_t_value`, `TimestepEmbedder`,
// `AudioPositionEmbedding`). Template for the C++/GGML posture:
// train/mm3-dit-train-graph.h.
//
// ── What this file is, in one sentence ─────────────────────────────────────
//
// `yue2_nt_block` is `yue2_nar_block` (yue2-nar-graph.h:318-376) with three
// changes and nothing else: LoRA sites at the seven projections, `ggml_set`
// instead of `ggml_concat` for the K/V assembly, and no flash-attention
// branch. Every other op, every reshape order and both position schemes are
// copied verbatim, because a trainer that computes a subtly different forward
// trains a model that is not the one we ship.
//
// ── Trap 1: CONCAT has no backward, so the K/V assembly changes shape ──────
//
// The inference block concatenates the frozen AR-prefix K/V cache with this
// call's fresh NAR K/V (yue2-nar-graph.h:353-354). `GGML_OP_CONCAT` is not in
// `ggml_compute_backward`'s switch, so that op cannot appear anywhere upstream
// of the loss. Contract §1 chose `ggml_set` into a persistent F32 canvas whose
// rows [0, ar_len) are pre-filled with the prefill's K/V:
//
//     k_cat = ggml_set(ctx, kv.k[i], k_w, nb1, nb2, nb3, ar_len * nb1)
//
// `GGML_OP_SET`'s backward (engine/ggml/src/ggml.c:7160-7186) hands `src1` —
// our NAR rows — an exact unscaled slice of the incoming gradient, PROVIDED
// that gradient is in the canvas's own layout (trap 1b below). The canvas
// is a constant leaf with no `ggml_set_param`, so `src0_needs_grads` is false
// and the `ggml_neg`/`ggml_acc` arm at 7178-7181 is never built. The CUDA gate
// (ggml-cuda.cu:4925-4931) is F32/I32 only and demands all three types match,
// which is the second reason the F16 cast at yue2-nar-graph.h:347-348 is
// dropped here (the first being contract §3's F32-end-to-end policy).
//
// The offset and strides fit the op's int32 op_params (ggml.c:3460-3461):
// offset = ar_len*D*4 = 102,400 and nb2 = D*S_kv*4 = 231,424 at a 10 s clip.
// A whole-song S_kv would NOT fit — one of several reasons 10 s clips are
// structural here rather than a default someone picked (contract §8).
//
// ── Trap 1b: SET's backward assumes the canvas layout; V's grad breaks it ──
//
// SET's backward slices the incoming gradient with
//
//     tensor_grad_view = ggml_view_4d(ctx, grad, src1->ne[0..3],
//                                     nb1, nb2, nb3, offset)   // ggml.c:7174
//
// where nb1/nb2/nb3/offset are read straight back out of the SET's op_params,
// i.e. they describe the CANVAS. `ggml_view_4d` overrides nb[1..3] only —
// nb[0] stays at the element size from `ggml_new_tensor_impl` — and it asserts
// nothing about contiguity, so if `grad` is not already laid out like the
// canvas, the slice silently reads the wrong elements. Element count is
// unchanged, so it stays inside the buffer: no OOB, no assert, just wrong
// numbers all the way up through `ad.v`/`w.attn_v` and every earlier block.
//
// K and V are ASYMMETRIC here, which is what makes it easy to miss:
//
//   K: `scores = mul_mat(k_cat, q4)`. MUL_MAT's src0 arm is out_prod +
//      view_4d + `ggml_repeat_back` (ggml.c:7036-7052, the GQA broadcast arm),
//      and repeat_back allocates a FRESH tensor (ggml.c:2615-2626), so
//      grads[k_cat] is contiguous [D, S_kv, Nkv, 1] — natural strides that
//      happen to equal the canvas's. Correct by provenance, not by design.
//
//   V: `yue2_lm_attn_f32` uses V only as `cont(transpose(v))`
//      (yue2-lm-graph.h:124). That same fresh contiguous [S_kv, D, Nkv, 1]
//      passes through CONT's backward unchanged, then TRANSPOSE's backward
//      stores `ggml_transpose(grad)` VERBATIM (ggml.c:7256-7259 into the
//      empty-slot arm of `ggml_add_or_set`, ggml.c:6809-6813). grads[v_cat]
//      is therefore a NON-CONTIGUOUS view: ne [D, S_kv, Nkv, 1] over memory
//      still laid out [S_kv, D, Nkv].
//
// Fix, applied at the k_cat/v_cat call site: wrap each SET in an identity
// `ggml_reshape_4d`. RESHAPE's backward does
// `grad_cont = ggml_is_contiguous(grad) ? grad : ggml_cont(ctx, grad)`
// (ggml.c:7209-7213), which normalises the transposed view into the canvas's
// layout before SET ever sees it, and costs nothing on the K side (its grad is
// already contiguous, so the reshape is a pass-through view). Both sides get
// the wrapper deliberately: the asymmetry above is an accident of which ops
// attention happens to apply to K vs V, and anyone touching
// `yue2_lm_attn_f32` could flip it.
//
// A plain `ggml_cont` after the SET does NOT work as a substitute: CONT's
// backward ASSERTS the incoming grad is contiguous (ggml.c:7201) and would
// abort rather than fix anything.
//
// ── Trap 2: no flash attention, ever ───────────────────────────────────────
//
// `GGML_OP_FLASH_ATTN_EXT` has no backward. We always take
// `yue2_lm_attn_f32` (yue2-lm-graph.h:116-127) — the same manual
// mul_mat -> soft_max_ext -> mul_mat path MM3's trainer takes
// (mm3-dit-train-graph.h:167-170). The custom fused training ops
// (GGML_OP_FLASH_ATTN_TRAIN, see the flash-attn-training skill) are NOT wired
// here: they would remove the [S_kv, N_nar, Nh] retained softmax, which is
// only ~14 MB/layer at T=250, so there is nothing to buy yet.
//
// The mask is NULL, and that is correctness rather than a shortcut: every NAR
// query attends over the whole [prefix ++ NAR] span bidirectionally
// (yue2-nar-graph.h:65-73), and `training_velocity` builds the same all-ones
// block for the NAR query rows. `soft_max_ext`'s `scale` argument has a
// correct backward (ggml.c:7284-7295 reads it back out of op_params), so do
// NOT pre-scale q — CUDA only supports SOFT_MAX_BACK at max_bias == 0, which
// is what we pass.
//
// ── Trap 3: the token axis must never reach ne2 of a trainable mul_mat ─────
//
// mm3-dit-train-graph.h:37-44. Every LoRA site here consumes a 2-D activation
// ([H, N_nar] or [F, N_nar]), so ne2 == 1 at all eleven sites; the 4-D
// `reshape_4d(..., D, Nh, T, 1)` happens AFTER the LoRA add, so T only reaches
// ne2 in the attention matmuls, whose operands are activations, not factors.
// Contract §1 has the full table.
//
// YuE2 has GQA (Nh=16, Nkv=8) where MM3 does not, so the broadcast arm of
// MUL_MAT's backward (ggml.c:7036-7052, the `repeat_back` over nr2) is
// exercised here for the first time in this codebase. Its CUDA gate
// (ggml-cuda.cu:5014-5015) needs `src0->ne[2]*ne[3] <= 32768`; ours is 8*2=16.
//
// ── Trap 4: no NORM, and rms_norm's input is already contiguous ────────────
//
// Unlike MM3 (LayerNorm, hence mm3_dt_ln's rebuild-from-primitives at
// mm3-dit-train-graph.h:106-152), YuE2 is RMSNorm end to end and RMS_NORM has
// a backward. No LN surgery is needed at all.
//
// The per-head QK RMSNorm at yue2-nar-graph.h:335-336 normalises over
// ne0 = D = 128 of a `reshape_4d` of a `mul_mat`. In the trainer the inner op
// becomes `yue2_nt_linear`, whose last op is `ggml_add` — also contiguous — so
// `ggml_reshape_4d`'s own contiguity assert still passes and `ggml_rms_norm`
// still sees a contiguous tensor. **No extra `ggml_cont` at the rms_norm
// call** (contract §3). The only `cont`s that are required are the two
// permutes into K/V and the one on q4, all of which the inference path
// already has.
//
// ── What is NOT here (listed, not polished) ────────────────────────────────
//
// - No gradient-checkpointing DRIVER. The three-way split
//   (`yue2_nt_prologue` / `yue2_nt_stack` / `yue2_nt_epilogue`) exists so
//   mm3-dit-train-ckpt.h's segment driver can be adapted later without
//   restating the forward, but contract §8 measured peak VRAM at ~11-15 GB
//   monolithic for a 10 s clip and concluded checkpointing buys nothing at
//   v1. Whoever writes the driver: the prologue holds trained parameters
//   under the `proj` preset, so segment 0 must recompute it inside its own
//   gradient graph (mm3-dit-train-ckpt.h:37-41).
// - No export. Plan §4's safetensors key scheme is written by the runner
//   through train/st-write.h; the tag strings below are the TRAINER's internal
//   names (chosen to match the contract's FD probe labels), not the on-disk
//   keys.
// - No caption dropout / cond-uncond swap. The seam is in place
//   (`Yue2NarTrainCond`, and the runner caches one canvas per distinct
//   conditioning) but nothing draws from it yet.
//
// ── Risk noted, not resolved: BF16 base weights in the backward ────────────
//
// The shipped LM GGUF stores weights in their NATIVE type, which for the
// reference checkpoint is BF16. The frozen-weight arm of MUL_MAT's backward
// under the mm-backward patch is `mul_mat(cont(transpose(W)), grad)`, i.e. it
// conts a transposed BF16 tensor. MM3's trainer only ever exercises that path
// on F16/Q8_0 bases. `--fd-check` sidesteps it entirely (the F32 isolation in
// yue2-nar-train-run.h mirrors the probed layers), so a BF16-specific fault
// there would show up in the first long run, not in the gate. Flagged rather
// than guessed at, because nothing here has been compiled or run yet.

#include "yue2/yue2-lm-graph.h"   // yue2_lm_rms, yue2_lm_attn_f32, Yue2ArKvCache
#include "yue2/yue2-model.h"      // Yue2Model, Yue2LmConfig, Yue2NarLayer
#include "yue2/yue2-nar-graph.h"  // Yue2NarChunk, yue2_nar_shift_t, yue2_nar_time_features
#include "yue2/yue2-tokenizer.h"  // YUE2_CODEC_OFFSET, YUE2_MUSIC_END

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Conditioning seam (contract §9) ────────────────────────────────────────
//
// Training conditions on the cot="off" text prefix through MUSIC_START, then
// optionally this clip's own codec ids, then MUSIC_END — the exact shape
// yue2-pipeline.h:447-452 builds for inference. Upstream's text-only regime
// (train.py:99, "AR conditioning tokens incl. MUSIC_END (no codec tokens)")
// is just `codec_ids` empty; when the community audio -> semantic-token
// encoder lands, filling that field makes training condition exactly as
// inference does, with NO graph change (ar_len is read off the assembled
// vector and the canvas is sized to it).
//
// Codec ids stay RAW (pre-offset) everywhere they are stored; `+
// YUE2_CODEC_OFFSET` happens here and only here, as in the pipeline.
struct Yue2NarTrainCond {
    std::vector<int32_t> prefix_ids;  // yue2_token_prefixes(tok, style, lyrics, YUE2_COT_OFF)
    std::vector<int32_t> codec_ids;   // RAW ids for THIS clip; EMPTY = text-only regime
};

static std::vector<int32_t> yue2_nar_train_cond_ids(const Yue2NarTrainCond & c) {
    std::vector<int32_t> out = c.prefix_ids;
    out.reserve(c.prefix_ids.size() + c.codec_ids.size() + 1);
    for (int32_t id : c.codec_ids) {
        out.push_back(id + (int32_t) YUE2_CODEC_OFFSET);
    }
    out.push_back((int32_t) YUE2_MUSIC_END);
    return out;
}

// ── LoRA sites ─────────────────────────────────────────────────────────────
//
// delta = (alpha/rank) * B @ A added to the base projection's output.
// a: [in, rank]   b: [rank, out]   — ggml order, so both are 2-D, ne2 == 1.
//
// No `out_mask` (contrast MM3TrainLora at mm3-dit-train-graph.h:56-73): YuE2's
// attention projections are SEPARATE q/k/v tensors, not one fused qkv, so
// there is nothing to mask — "V rows only" is simply "adapt v and not q, k",
// which the target preset expresses directly.
struct Yue2TrainLora {
    ggml_tensor * a     = nullptr;
    ggml_tensor * b     = nullptr;
    float         scale = 1.0f;  // alpha / rank
    bool          on() const { return a && b; }
};

struct Yue2TrainLayerAdapters {
    Yue2TrainLora q, k, v, o;      // nar_attn_{q,k,v,output}
    Yue2TrainLora gate, up, down;  // nar_ffn_{gate,up,down}
};

// Contract §6 spells this struct `Yue2TrainBlock`; the deliverable list spells
// it `Yue2TrainLayerAdapters` (and "block" already means "one of the 28
// transformer blocks" in this file). Same type, both names.
using Yue2TrainBlock = Yue2TrainLayerAdapters;

struct Yue2TrainAdapters {
    std::vector<Yue2TrainLayerAdapters> blk;                        // block_count entries
    Yue2TrainLora                       vae2llm, llm2vae;           // "proj" preset only
    Yue2TrainLora                       time0, time1;               // "proj" preset only
};

// y = W x  (+ scale * B(A x) when the site is active).
//
// `mm3_dt_linear` (mm3-dit-train-graph.h:90-104) minus the mask branch. W is
// the frozen base in its GGUF type; x is [in, S]; the LoRA branch runs F32.
// One helper for every site so a site cannot silently be left un-adapted.
static ggml_tensor * yue2_nt_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x,
                                    const Yue2TrainLora & lo) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (!lo.on()) {
        return y;
    }
    ggml_tensor * ax = ggml_mul_mat(ctx, lo.a, x);   // [rank, S]
    ggml_tensor * bx = ggml_mul_mat(ctx, lo.b, ax);  // [out,  S]
    return ggml_add(ctx, y, ggml_scale(ctx, bx, lo.scale));
}

// ── The AR-prefix canvas ───────────────────────────────────────────────────
//
// `Yue2ArKvCache` (yue2-lm-graph.h:494-500) stores post-QK-norm, post-RoPE K/V
// as F16. The training graph needs F32 (see trap 1) and needs room for the
// NAR rows the `ggml_set` writes, so the trainer keeps its OWN persistent
// canvas built once per distinct conditioning sequence:
//
//   k[i], v[i] : [D, S_kv, Nkv, 1] F32,  rows [0, ar_len) = the AR prefill,
//                                        rows [ar_len, S_kv) overwritten every
//                                        micro-step by the SET.
//
// Persistent because 28 x 2 x [128, 452, 8] F32 is 104 MB; re-uploading that
// every micro-step would put a PCIe transfer in the inner loop for no reason.
// Every row is defined before it is read, but the buffer is zero-cleared at
// allocation anyway — yue2-lm-graph.h:58-69's uninitialised-KV lesson.
struct Yue2NarTrainKv {
    ggml_context *              ctx = nullptr;
    ggml_backend_buffer_t       buf = nullptr;
    std::vector<ggml_tensor *> k, v;  // one [D, S_kv, Nkv, 1] F32 tensor per layer
    int64_t                     ar_len = 0;  // rows held by the frozen prefix
    int64_t                     n_nar  = 0;  // NAR query rows (= chunk_len + 2)
    int64_t                     s_kv   = 0;  // ar_len + n_nar
};

static void yue2_nt_kv_free(Yue2NarTrainKv * kv) {
    if (kv->buf) {
        ggml_backend_buffer_free(kv->buf);
        kv->buf = nullptr;
    }
    if (kv->ctx) {
        ggml_free(kv->ctx);
        kv->ctx = nullptr;
    }
    kv->k.clear();
    kv->v.clear();
    kv->ar_len = kv->n_nar = kv->s_kv = 0;
}

// Widen a filled `Yue2NarChunk`'s F16 AR cache into a fresh F32 canvas.
//
// `n_layers` lets the FD gate build a canvas for a TRUNCATED NAR stack (it
// only ever runs the first K blocks); a real run passes block_count.
//
// The per-head copy is a single contiguous span on both sides: the cache is
// [D, ar_len, Nkv, 1] contiguous, so head h's rows live at byte offset
// h*nb[2] with ar_len*D entries, and the destination's head h starts at
// h*dst->nb[2] — the same flatten order, which is exactly what
// yue2_ar_kv_cache_dump_layer relies on (yue2-lm-graph.h:570-600).
static bool yue2_nt_kv_from_chunk(const Yue2Model & m, const Yue2NarChunk & chunk, int n_layers,
                                  Yue2NarTrainKv * out, std::string * err) {
    if (!m.lm_resident) {
        if (err) {
            *err = "YuE2 LM is not resident (yue2_load_parts(want_lm=true) first)";
        }
        return false;
    }
    if (chunk.ar_cache.filled != chunk.ar_length || chunk.ar_length <= 0) {
        if (err) {
            *err = "yue2_nt_kv_from_chunk: the chunk's AR prefill is missing or partial";
        }
        return false;
    }
    const Yue2LmConfig & c   = m.lm_cfg;
    const int64_t        D   = (int64_t) c.key_length;
    const int64_t        Nkv = (int64_t) c.head_count_kv;
    const int            L   = std::min(n_layers, (int) chunk.ar_cache.k.size());
    if (L <= 0) {
        if (err) {
            *err = "yue2_nt_kv_from_chunk: n_layers must be > 0";
        }
        return false;
    }

    out->ar_len = chunk.ar_length;
    out->n_nar  = chunk.nar_length;
    out->s_kv   = chunk.ar_length + chunk.nar_length;

    ggml_init_params ip = { (size_t) (L * 2) * ggml_tensor_overhead() + 1024, nullptr, /*no_alloc*/ true };
    out->ctx            = ggml_init(ip);
    if (!out->ctx) {
        if (err) {
            *err = "yue2_nt_kv_from_chunk: ggml_init failed";
        }
        return false;
    }
    out->k.assign((size_t) L, nullptr);
    out->v.assign((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        out->k[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, out->s_kv, Nkv, 1);
        out->v[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, out->s_kv, Nkv, 1);
        ggml_set_name(out->k[(size_t) i], ("yue2_nt_kvk." + std::to_string(i)).c_str());
        ggml_set_name(out->v[(size_t) i], ("yue2_nt_kvv." + std::to_string(i)).c_str());
    }
    out->buf = ggml_backend_alloc_ctx_tensors(out->ctx, m.backend);
    if (!out->buf) {
        yue2_nt_kv_free(out);
        if (err) {
            *err = "yue2_nt_kv_from_chunk: backend buffer allocation failed (out of VRAM?)";
        }
        return false;
    }
    // Defensive, per the file-header note: every row IS written before it is
    // read, but a masked-but-uninitialised KV row has bitten this codebase.
    ggml_backend_buffer_clear(out->buf, 0);
    // Tell the scheduler these live on the compute backend. Without the hint
    // the graph gets cut at every canvas read — the same 293-split, 47 s/step
    // failure mm3-dit-train-run.h:585-592 documents for the LoRA buffer.
    ggml_backend_buffer_set_usage(out->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t          n_head_elems = (size_t) (chunk.ar_length * D);
    std::vector<uint16_t> raw(n_head_elems);
    std::vector<float>    f32(n_head_elems);
    for (int i = 0; i < L; i++) {
        ggml_tensor * src[2] = { chunk.ar_cache.k[(size_t) i], chunk.ar_cache.v[(size_t) i] };
        ggml_tensor * dst[2] = { out->k[(size_t) i], out->v[(size_t) i] };
        for (int s = 0; s < 2; s++) {
            for (int64_t h = 0; h < Nkv; h++) {
                ggml_backend_tensor_get(src[s], raw.data(), (size_t) h * src[s]->nb[2],
                                        n_head_elems * sizeof(uint16_t));
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) n_head_elems);
                ggml_backend_tensor_set(dst[s], f32.data(), (size_t) h * dst[s]->nb[2],
                                        n_head_elems * sizeof(float));
            }
        }
    }
    return true;
}

// ── Graph inputs ───────────────────────────────────────────────────────────
//
// Layouts are the INFERENCE graph's (yue2_nar_velocity:455-469), not the
// obvious ones, so that a divergence is a compile/assert failure rather than
// a silently wrong model.
struct Yue2NarTrainInputs {
    // [LD, N_nar] F32. Row 0 and row N_nar-1 are the zero boundary rows; rows
    // 1..T carry the NOISED latents x_t. Channel-fastest, exactly what
    // yue2_nar_velocity uploads (x_nar_host, :419-420).
    ggml_tensor * x_nar = nullptr;
    // [256, 1] F32 sinusoid features, cos-block then sin-block, computed on
    // the host by yue2_nar_time_features. NOT the final embedding: the
    // two-layer MLP over it runs IN-GRAPH (contract §4), so it stays one
    // definition shared with inference.
    ggml_tensor * time_feat = nullptr;
    // [N_nar] i32, the LOCAL AudioPositionEmbedding index min(i, max-1),
    // restarting at 0 per chunk.
    ggml_tensor * local_idx = nullptr;
    // [N_nar] i32, the GLOBAL RoPE position ar_len + i. NEVER aliased with
    // local_idx — yue2-nar-graph.h:43-52 is the whole warning.
    ggml_tensor * rope_pos = nullptr;
    // [LD, T] F32 velocity target (noise - z), already channel-major so it
    // matches `pred` with no transpose in the graph (contract §7).
    ggml_tensor * vtarget = nullptr;
};

// Host-side companions to the four constant inputs. Split out so the runner,
// the FD gate and any future loop all build them one way.
struct Yue2NarTrainHostInputs {
    std::vector<float>   x_nar;      // LD * N_nar
    std::vector<float>   time_feat;  // 256
    std::vector<int32_t> local_idx;  // N_nar
    std::vector<int32_t> rope_pos;   // N_nar
};

// `x_t` is [T, LD] channel-fastest (index t*LD + d), the same layout
// yue2_nar_velocity's `state` uses. `raw_t` is the ALREADY logit-transformed
// scalar — contract §7's time contract: hand in `raw_t`, never `t` and never
// the shifted value, because yue2_nar_velocity applies `_shift_t_value` and
// the sinusoid itself (:384-389, 415-417) and the solver passes logit(t)
// at :602-603. Getting this wrong trains against a different time
// parameterisation than inference samples with, and nothing would report it.
//
// A size mismatch is REPORTED, never absorbed: substituting a zero x_t would
// hand the caller a well-formed input block describing a latent nobody asked
// for, and the FD gate downstream would read it as a broken backward. Same
// posture yue2_nar_velocity takes on its own state-size check
// (yue2-nar-graph.h:404-409).
static bool yue2_nt_host_inputs(const Yue2LmConfig & c, int64_t ar_len, int64_t n_nar,
                                const std::vector<float> & x_t, double raw_t,
                                Yue2NarTrainHostInputs * out, std::string * err = nullptr) {
    const int64_t LD = (int64_t) c.latent_dim;
    const int64_t T  = n_nar - 2;

    if (T <= 0) {
        if (err) {
            *err = "yue2_nt_host_inputs: n_nar must be >= 3 (two boundary rows plus at least one frame)";
        }
        return false;
    }
    if ((int64_t) x_t.size() != T * LD) {
        if (err) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "yue2_nt_host_inputs: x_t size mismatch (got %zu, expected (n_nar-2)*latent_dim = %lld)",
                     x_t.size(), (long long) (T * LD));
            *err = buf;
        }
        return false;
    }

    out->x_nar.assign((size_t) (n_nar * LD), 0.0f);   // boundary rows stay zero
    memcpy(out->x_nar.data() + (size_t) LD, x_t.data(), (size_t) (T * LD) * sizeof(float));

    const double shifted = yue2_nar_shift_t(raw_t, (double) c.timestep_shift);
    yue2_nar_time_features(shifted, &out->time_feat);

    out->local_idx.assign((size_t) n_nar, 0);
    out->rope_pos.assign((size_t) n_nar, 0);
    const int64_t max_idx = (int64_t) c.max_latent_frames - 1;
    for (int64_t i = 0; i < n_nar; i++) {
        out->local_idx[(size_t) i] = (int32_t) std::min<int64_t>(i, max_idx);
        out->rope_pos[(size_t) i]  = (int32_t) (ar_len + i);
    }
    return true;
}

// ── One trainable NAR block: mirrors yue2_nar_block op for op ──────────────
//
// `kv_k` / `kv_v` are this layer's persistent F32 canvas, [D, S_kv, Nkv, 1],
// rows [0, ar_len) already holding the frozen prefix. `ar_len` is DERIVED
// (S_kv - N_nar) rather than passed: one fewer argument to get out of sync
// with the canvas it addresses.
static ggml_tensor * yue2_nt_block(ggml_context * ctx, const Yue2LmConfig & c, const Yue2NarLayer & w,
                                   const Yue2TrainLayerAdapters & ad, ggml_tensor * h, ggml_tensor * rope_pos,
                                   ggml_tensor * kv_k, ggml_tensor * kv_v) {
    const int64_t H   = (int64_t) c.embedding_length;
    const int64_t D   = (int64_t) c.key_length;
    const int64_t Nh  = (int64_t) c.head_count;
    const int64_t Nkv = (int64_t) c.head_count_kv;
    const int64_t T   = h->ne[1];              // N_nar, the NAR query length
    const int64_t ar_len = kv_k->ne[1] - T;    // rows the frozen prefix occupies

    ggml_tensor * n = yue2_lm_rms(ctx, h, w.attn_norm, c.rms_eps);

    // The LoRA add happens BEFORE the reshape_4d, which is what keeps T out of
    // ne2 at every trainable matmul (trap 3). Do not hoist the reshape.
    ggml_tensor * q = ggml_reshape_4d(ctx, yue2_nt_linear(ctx, w.attn_q, n, ad.q), D, Nh, T, 1);
    ggml_tensor * k = ggml_reshape_4d(ctx, yue2_nt_linear(ctx, w.attn_k, n, ad.k), D, Nkv, T, 1);
    ggml_tensor * v = ggml_reshape_4d(ctx, yue2_nt_linear(ctx, w.attn_v, n, ad.v), D, Nkv, T, 1);

    // Per-head QK RMSNorm before RoPE, no norm on V. No ggml_cont here: the
    // reshape_4d's source ends in ggml_add, so it is contiguous (trap 4 /
    // contract §3). ggml_mul broadcasts a FROZEN [D] weight, so MUL's backward
    // only ever needs the src0 arm — same situation as MM3's frozen gamma/beta
    // (mm3-dit-train-graph.h:138-140).
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, c.rms_eps), w.attn_q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, c.rms_eps), w.attn_k_norm);

    // GLOBAL RoPE positions (ar_len + local index). Identical call to
    // yue2-nar-graph.h:340-343, including GGML_ROPE_TYPE_NEOX and every
    // scaling argument; ROPE has a backward.
    q = ggml_rope_ext(ctx, q, rope_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, rope_pos, NULL, (int) D, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f,
                      1.0f, 0.0f, 0.0f);

    // [D,Nkv,T,1] -> [D,T,Nkv,1] contiguous, then written into the canvas at
    // row ar_len. NO ggml_cast to F16 (trap 1): ggml_set's CUDA gate is
    // F32/I32 only and requires src0, src1 and dst to share a type.
    ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    const size_t off = (size_t) ar_len * kv_k->nb[1];
    // The prefix rows come through untouched and carry no gradient (the canvas
    // is not a param), which is exactly the frozen-AR contract.
    //
    // The identity reshape_4d on each SET is LOAD-BEARING, not tidying — see
    // trap 1b in the file header. SET's backward slices the incoming gradient
    // with `ggml_view_4d(grad, ..., nb1, nb2, nb3, offset)` taken from the
    // CANVAS's op_params, which is only meaningful if that gradient is already
    // in the canvas's (contiguous) layout. V's is not: attention conts a
    // TRANSPOSE of it, and TRANSPOSE's backward re-transposes verbatim, so
    // `grads[v_cat]` arrives as a non-contiguous view. RESHAPE's backward
    // (ggml.c:7209-7213) conts a non-contiguous grad before reshaping, which
    // normalises exactly that. A plain ggml_cont here would NOT work — CONT's
    // backward asserts the grad is already contiguous (ggml.c:7201).
    ggml_tensor * k_cat = ggml_reshape_4d(ctx, ggml_set(ctx, kv_k, k_w, kv_k->nb[1], kv_k->nb[2], kv_k->nb[3], off),
                                          D, kv_k->ne[1], Nkv, 1);
    ggml_tensor * v_cat = ggml_reshape_4d(ctx, ggml_set(ctx, kv_v, v_w, kv_v->nb[1], kv_v->nb[2], kv_v->nb[3], off),
                                          D, kv_v->ne[1], Nkv, 1);

    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,1]

    // ALWAYS the manual path (trap 2). Mask NULL = full bidirectional
    // attention over [prefix ++ NAR]; scale stays inside soft_max_ext because
    // its backward reads it back out of op_params.
    ggml_tensor * attn = yue2_lm_attn_f32(ctx, q4, k_cat, v_cat, nullptr, c.softmax_scale);
    attn               = ggml_reshape_2d(ctx, attn, H, T);

    h = ggml_add(ctx, h, yue2_nt_linear(ctx, w.attn_output, attn, ad.o));

    ggml_tensor * n2   = yue2_lm_rms(ctx, h, w.ffn_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, yue2_nt_linear(ctx, w.ffn_gate, n2, ad.gate));
    ggml_tensor * up   = yue2_nt_linear(ctx, w.ffn_up, n2, ad.up);
    // SwiGLU spelled out as silu(gate) * up, never ggml_swiglu — the fused GLU
    // op is in ggml_compute_backward's list but the inference path spells it
    // out too, and keeping the two identical is the point of this file.
    return ggml_add(ctx, h, yue2_nt_linear(ctx, w.ffn_down, ggml_mul(ctx, gate, up), ad.down));
}

// ── Forward, split three ways for a future checkpointing driver ────────────
//
// The segment boundary is a single [H, N_nar] tensor: YuE2 has no
// cross-attention, no AdaLN and no per-layer conditioning, so the same "one
// tensor is the whole boundary" argument MM3 makes
// (mm3-dit-train-graph.h:222-239) holds here. Default segments = 1; contract
// §8 says do not write the driver for phase 2.

static ggml_tensor * yue2_nt_prologue(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                      const Yue2NarTrainInputs & in);
static ggml_tensor * yue2_nt_stack(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                   const Yue2NarTrainKv & kv, const Yue2NarTrainInputs & in, ggml_tensor * h,
                                   int lo, int hi);
static ggml_tensor * yue2_nt_epilogue(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                      ggml_tensor * h);

// Input assembly + n_layers NAR blocks + final norm + llm2vae, returning the
// predicted velocity for the CONTENT rows only: [LD, T], T = N_nar - 2.
//
// `n_layers` is explicit rather than read from `c.block_count` so the FD gate
// can truncate the stack without mutating the model's config (which other
// code, including the AR prefill that produced the canvas, reads).
static ggml_tensor * yue2_nt_build_forward(ggml_context * ctx, const Yue2Model & m,
                                           const Yue2TrainAdapters & ad, const Yue2NarTrainKv & kv,
                                           const Yue2NarTrainInputs & in, int n_layers) {
    ggml_tensor * h = yue2_nt_prologue(ctx, m, ad, in);
    h               = yue2_nt_stack(ctx, m, ad, kv, in, h, 0, n_layers);
    return yue2_nt_epilogue(ctx, m, ad, h);
}

static ggml_tensor * yue2_nt_prologue(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                      const Yue2NarTrainInputs & in) {
    // vae2llm(x_nar) + time_embedder(shifted t) + pos_emb[local_idx] —
    // yue2_nar_velocity:471-480, with the ONE difference contract §5 names:
    // the content rows of x_nar are the noised latents x_t, not an ODE state.
    //
    // `token_emb` for LATENT_START/LATENT_END is deliberately absent. Upstream
    // builds the full token embedding and then ASSIGNS over every NAR row
    // (modeling_yue2.py:665, train.py:117-118) — assigned, not added — so our
    // skipping embed_tokens for the NAR span is bit-equivalent, and the
    // trainer inherits that from the inference path rather than re-deriving it.
    ggml_tensor * ve = ggml_add(ctx, yue2_nt_linear(ctx, m.lm.vae2llm_w, in.x_nar, ad.vae2llm), m.lm.vae2llm_b);

    ggml_tensor * t1 =
        ggml_silu(ctx, ggml_add(ctx, yue2_nt_linear(ctx, m.lm.time_embd_w[0], in.time_feat, ad.time0),
                                m.lm.time_embd_b[0]));
    ggml_tensor * t2 =
        ggml_add(ctx, yue2_nt_linear(ctx, m.lm.time_embd_w[1], t1, ad.time1), m.lm.time_embd_b[1]);  // [H,1]

    ggml_tensor * pos_emb = ggml_get_rows(ctx, m.lm.latent_pos_embed, in.local_idx);  // [H, N_nar]

    // KEEP THIS NESTING (contract §4): add(add(ve, t2), pos_emb) puts the
    // [H,1] broadcast on src1, where ADD's backward reduces it with
    // repeat_back (ggml.c:6901-6907) into a shape with ne2*ne3 == 1, inside
    // the CUDA cap. Reordering would move the broadcast to src0 and change
    // which arm has to reduce.
    return ggml_add(ctx, ggml_add(ctx, ve, t2), pos_emb);  // [H, N_nar]
}

// Blocks [lo, hi). The whole point of the split: a checkpoint segment would
// rebuild exactly this range from a saved boundary instead of from block 0.
static ggml_tensor * yue2_nt_stack(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                   const Yue2NarTrainKv & kv, const Yue2NarTrainInputs & in, ggml_tensor * h,
                                   int lo, int hi) {
    const Yue2LmConfig & c = m.lm_cfg;
    for (int i = lo; i < hi; i++) {
        h = yue2_nt_block(ctx, c, m.lm.nar_blk[(size_t) i], ad.blk[(size_t) i], h, in.rope_pos,
                          kv.k[(size_t) i], kv.v[(size_t) i]);
    }
    return h;
}

static ggml_tensor * yue2_nt_epilogue(ggml_context * ctx, const Yue2Model & m, const Yue2TrainAdapters & ad,
                                      ggml_tensor * h) {
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        LD = (int64_t) c.latent_dim;
    const int64_t        T  = h->ne[1] - 2;  // content rows, boundaries dropped

    // Shared AR/NAR final norm ("model.norm"), then llm2vae.
    ggml_tensor * h_final  = yue2_lm_rms(ctx, h, m.lm.output_norm, c.rms_eps);
    ggml_tensor * out_full = ggml_add(ctx, yue2_nt_linear(ctx, m.lm.llm2vae_w, h_final, ad.llm2vae),
                                      m.lm.llm2vae_b);  // [LD, N_nar]

    // VIEW + CONT, not get_rows. The inference path gathers rows 1..T with
    // ggml_get_rows (:497) because it also wants the gather machinery for
    // other shapes; here the content rows are a contiguous span of columns
    // 1..T of a 2-D tensor, so the cheaper op is also the simpler one, and
    // both VIEW and CONT have backward (contract §7).
    return ggml_cont(ctx, ggml_view_2d(ctx, out_full, LD, T, out_full->nb[1], out_full->nb[1]));
}

// ── Loss ───────────────────────────────────────────────────────────────────
//
// Rectified-flow MSE against the caller-supplied velocity target:
//
//     raw  ~ N(0,1)                 logit-normal t (train.py:131-139)
//     t    = sigmoid(raw)
//     x_t  = (1-t)*z + t*noise      z = VAE posterior MEAN
//     v*   = noise - z              (train.py:231-233)
//     loss = mean((v_pred - v*)^2)
//
// THE SIGN IS SETTLED HERE, unlike MM3's (mm3-dit-train-graph.h:344-352 left
// it open and then MEASURED the opposite of what it argued). Both sides agree
// on YuE2: with x_t = (1-t)z + t*noise, dx/dt = noise - z, and the shipped
// midpoint solver integrates t: 1 -> 0 with `state -= v*dt`
// (yue2-nar-graph.h:601-627). MM3's `output_negated` trap does not transfer —
// there is no ComfyUI-vs-diffusers negation question in this port.
//
// `pred` and `target` are both [LD, T] channel-major, so the divisor is
// LD*T = every element, matching mm3_dt_loss (mm3-dit-train-graph.h:358-361).
static ggml_tensor * yue2_nt_loss(ggml_context * ctx, ggml_tensor * pred, ggml_tensor * target) {
    ggml_tensor * d = ggml_sub(ctx, pred, target);
    return ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, d)), 1.0f / (float) ggml_nelements(d));
}

// The same formula on the host in double, from a downloaded `pred`. Used by
// the FD gate's NUMERIC arm: an in-graph f32 sum over LD*T terms is perfectly
// deterministic but only ~1e-7 accurate, and differencing two such values is
// the catastrophic cancellation mm3-lm-train-run.h:1190-1200 measured and
// removed the same way. It is also an INDEPENDENT implementation of the loss
// rather than the same kernel twice.
static double yue2_nt_loss_host(const std::vector<float> & pred, const std::vector<float> & target) {
    const size_t n   = std::min(pred.size(), target.size());
    double       sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double) pred[i] - (double) target[i];
        sum += d * d;
    }
    return n ? sum / (double) n : 0.0;
}

// ── Target presets ─────────────────────────────────────────────────────────
//
// Upstream's own groups (lora.py:28-39). `nar_attn_mlp` is the default from
// plan §7's starting point; `nar_attn_mlp_proj` additionally adapts the four
// flow heads, which is what makes the timestep path and both latent
// projections trainable (and is what the FD gate wants, so every probe the
// contract names has a tensor to address).
enum Yue2NtTarget {
    YUE2_NT_ATTN          = 0,  // nar_attn:          q, k, v, o
    YUE2_NT_ATTN_MLP      = 1,  // nar_attn_mlp:      + gate, up, down     (default)
    YUE2_NT_ATTN_MLP_PROJ = 2,  // nar_attn_mlp_proj: + vae2llm, llm2vae, time_embd.{0,1}
};

static const char * yue2_nt_target_name(Yue2NtTarget t) {
    switch (t) {
        case YUE2_NT_ATTN:
            return "nar_attn";
        case YUE2_NT_ATTN_MLP_PROJ:
            return "nar_attn_mlp_proj";
        case YUE2_NT_ATTN_MLP:
        default:
            return "nar_attn_mlp";
    }
}

static bool yue2_nt_parse_target(const std::string & s, Yue2NtTarget * out) {
    if (s == "nar_attn") {
        *out = YUE2_NT_ATTN;
    } else if (s == "nar_attn_mlp" || s.empty()) {
        *out = YUE2_NT_ATTN_MLP;
    } else if (s == "nar_attn_mlp_proj") {
        *out = YUE2_NT_ATTN_MLP_PROJ;
    } else {
        return false;
    }
    return true;
}

// ── Adapter allocation ─────────────────────────────────────────────────────
//
// Mirrors mm3_train_make_adapters (mm3-dit-train-run.h:232-282): allocate A/B
// per site into a caller-owned context, name them "<tag>.A" / "<tag>.B",
// ggml_set_param both, and push them onto `params` in order. Initialisation is
// a SEPARATE pass (yue2_nt_init_adapters) because it has to run AFTER
// ggml_backend_alloc_ctx_tensors, and because the FD gate needs a different
// distribution than a real run does.
//
// Tag names are the trainer's INTERNAL names and match the FD probe labels in
// contract §10 (`blk.0.nar_attn_q.A`, `llm2vae.B`, `time_embd.1.B`). Plan §4's
// on-disk safetensors keys (`yue2.blk.N.nar_attn_q.lora_a`, `yue2.time_embed.0`)
// are produced by the exporter, which does not exist yet; when it does, it
// maps from these, and this comment is the note that they are NOT the same
// strings.
//
// `n_layers` lets the FD gate build adapters for a truncated stack. A real run
// passes block_count.
static bool yue2_nt_make_adapters(ggml_context * ctx, const Yue2Model & m, int n_layers, int64_t rank,
                                  float alpha, Yue2NtTarget target, Yue2TrainAdapters * ad,
                                  std::vector<ggml_tensor *> * params) {
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        H  = (int64_t) c.embedding_length;
    const int64_t        Q  = (int64_t) c.head_count * (int64_t) c.key_length;      // 2048
    const int64_t        K  = (int64_t) c.head_count_kv * (int64_t) c.key_length;   // 1024
    const int64_t        F  = (int64_t) c.feed_forward_length;                      // 6144
    const int64_t        LD = (int64_t) c.latent_dim;                               // 64
    const float          sc = alpha / (float) rank;

    if (rank <= 0) {
        return false;
    }

    auto mk = [&](Yue2TrainLora * lo, int64_t in, int64_t out, const std::string & tag) {
        lo->a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rank);
        lo->b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out);
        if (!lo->a || !lo->b) {
            lo->a = lo->b = nullptr;  // ctx arena too small; caller's alloc check catches it
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

    const bool want_mlp  = (target != YUE2_NT_ATTN);
    const bool want_proj = (target == YUE2_NT_ATTN_MLP_PROJ);

    ad->blk.assign((size_t) n_layers, Yue2TrainLayerAdapters{});
    for (int i = 0; i < n_layers; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        mk(&ad->blk[(size_t) i].q, H, Q, p + "nar_attn_q");
        mk(&ad->blk[(size_t) i].k, H, K, p + "nar_attn_k");
        mk(&ad->blk[(size_t) i].v, H, K, p + "nar_attn_v");
        mk(&ad->blk[(size_t) i].o, Q, H, p + "nar_attn_output");
        if (want_mlp) {
            mk(&ad->blk[(size_t) i].gate, H, F, p + "nar_ffn_gate");
            mk(&ad->blk[(size_t) i].up, H, F, p + "nar_ffn_up");
            mk(&ad->blk[(size_t) i].down, F, H, p + "nar_ffn_down");
        }
    }
    if (want_proj) {
        mk(&ad->vae2llm, LD, H, "vae2llm");
        mk(&ad->llm2vae, H, LD, "llm2vae");
        mk(&ad->time0, 256, H, "time_embd.0");
        mk(&ad->time1, H, H, "time_embd.1");
    }
    return true;
}

// How many tensors yue2_nt_make_adapters will create — for sizing the context.
static size_t yue2_nt_adapter_tensor_count(int n_layers, Yue2NtTarget target) {
    const size_t per_layer = (target == YUE2_NT_ATTN) ? 4u : 7u;
    const size_t heads     = (target == YUE2_NT_ATTN_MLP_PROJ) ? 4u : 0u;
    return ((size_t) n_layers * per_layer + heads) * 2u;
}

// The one random stream this file has. splitmix64 + Box-Muller, byte-for-byte
// the generator mm3_fill_noise_train uses (mm3-dit-train-run.h:316-334), and
// deliberately NOT <random>: std::normal_distribution and
// std::uniform_real_distribution are specified only by their distribution, not
// their byte stream, so the same seed gives different weights under a
// different stdlib and a "reproduce it with --seed 42" report means nothing
// across builds (trap #8 in the mm3-backend skill).
struct Yue2NtRng {
    uint64_t s;
    bool     has_spare = false;
    double   spare     = 0.0;

    explicit Yue2NtRng(uint64_t seed) : s(seed) {}

    // U(0,1), 53 bits.
    double u01() {
        s += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s;
        z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return ((z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    // N(0,1). Box-Muller draws in pairs; the sine half is cached rather than
    // discarded, so a caller that wants an odd count does not silently skip a
    // draw and shift every value after it.
    double normal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        const double u1 = std::max(1e-12, u01()), u2 = u01();
        const double r = std::sqrt(-2.0 * std::log(u1)), th = 6.283185307179586 * u2;
        spare     = r * std::sin(th);
        has_spare = true;
        return r * std::cos(th);
    }
};

// Init, run AFTER ggml_backend_alloc_ctx_tensors.
//
// MM3's MECHANISM, upstream's DISTRIBUTION (contract §6). The mechanism is
// mm3-dit-train-run.h:600-612: one loop over `params`, keyed on the LAST
// CHARACTER of the tensor name ('A' -> fill, anything else -> zero), from one
// seeded stream — Yue2NtRng here, where MM3 used std::mt19937_64, because a
// stdlib-independent byte stream is what makes "--seed 42" mean the same
// adapter on every build. The distribution is upstream's kaiming-uniform
// (lora.py:56-57 — kaiming_uniform_(a=sqrt(5)) is PyTorch's nn.Linear default,
// which works out to U(-1/sqrt(fan_in), +1/sqrt(fan_in))) rather than MM3's
// N(0, 0.02).
//
// B stays EXACTLY zero, so a fresh adapter is an exact no-op — that is plan
// §6's phase-2 second gate (zero-init LoRA velocity must equal
// yue2_nar_velocity), and it is also why `b_sigma > 0` exists: with B == 0,
// dL/dA is identically zero and every `.A` probe in the FD gate would report
// ||g|| = 0 and pass vacuously (mm3-dit-train-run.h:452-455 hits the same
// thing). The gate passes a non-zero b_sigma; a real run passes 0.
static void yue2_nt_init_adapters(const std::vector<ggml_tensor *> & params, uint64_t seed, float b_sigma) {
    Yue2NtRng          rng(seed);
    std::vector<float> v;
    for (ggml_tensor * t : params) {
        const size_t n    = (size_t) ggml_nelements(t);
        const char * nm   = ggml_get_name(t);
        const size_t len  = nm ? strlen(nm) : 0;
        const bool   is_a = len > 2 && nm[len - 1] == 'A';
        v.assign(n, 0.0f);
        if (is_a) {
            // fan_in is ne0 for an [in, rank] A factor.
            const double bound = 1.0 / std::sqrt((double) t->ne[0]);
            for (size_t i = 0; i < n; i++) {
                v[i] = (float) (bound * (2.0 * rng.u01() - 1.0));  // U(-bound, +bound)
            }
        } else if (b_sigma > 0.0f) {
            for (size_t i = 0; i < n; i++) {
                v[i] = (float) ((double) b_sigma * rng.normal());
            }
        }
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
    }
}

// Deterministic standard normal, through the same Yue2NtRng stream the
// adapter init uses — see that struct for why <random> is not an option here.
static void yue2_nt_fill_normal(std::vector<float> * out, uint64_t seed) {
    Yue2NtRng rng(seed);
    for (size_t i = 0; i < out->size(); i++) {
        (*out)[i] = (float) rng.normal();
    }
}

// x_t = (1-t)*z + t*noise,  v* = noise - z   (contract §7).
// All three vectors are [T, LD] channel-fastest.
static void yue2_nt_make_xt_target(const std::vector<float> & z, const std::vector<float> & noise, float t,
                                   std::vector<float> * xt, std::vector<float> * target) {
    xt->resize(z.size());
    target->resize(z.size());
    for (size_t i = 0; i < z.size(); i++) {
        (*xt)[i]     = (1.0f - t) * z[i] + t * noise[i];
        (*target)[i] = noise[i] - z[i];
    }
}
