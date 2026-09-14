#pragma once
// yue2/yue2-tok-head.h — Mothersuperior's YuE2 real-audio semantic tokenizer:
// the HEAD half, native C++/GGML.
//
// HOT-Step file (does not exist upstream, no acestep.cpp analog). Included
// only by bring-up tooling (engine/tools/yue2-probe.cpp --tok-head-parity) in
// this milestone; hot-step-server.cpp is NOT touched.
//
// ── What this file is, and what it deliberately is not ────────────────────
//
// The full tokenizer chain is
//
//   audio -> mono 24 kHz -> MERT-v2-FullSong -> hidden_states[20]
//         -> concat 30 s chunks -> interpolate to T25 -> fp16 store
//         -> per-track per-channel instance norm
//         -> THIS FILE (8-layer transformer head)
//         -> argmax -> code in [0, 32768) -> YuE2 token = code + 151853
//
// Everything left of "THIS FILE" is MERT and is NOT implemented here. The head
// is separable on purpose: it takes `[T, 1024]` instance-normed features and
// nothing else, and the oracle ships a MERT-FREE fixture for it
// (07_head_unit_*, a seeded `torch.randn` input), so it can be gated on its own
// before a single mel bin exists in C++. That is exactly how it was brought up.
//
// ── Ground truth ──────────────────────────────────────────────────────────
//
// docs/plans/yue2/12-tokenizer-oracle-pin.md §1 is AUTHORITATIVE and wins over
// anything said here. It was produced by RUNNING the reference, not by reading
// it. Transport (tensor names, shapes, F32-forcing policy, KV keys) comes from
// engine/tools/convert-yue2-tok.py, which wrote the file this loads.
//
// ONE CORRECTION, and it is not a disagreement with the pin so much as the pin
// disagreeing with itself: §1.6's prose says the activation is the exact-erf
// GELU, and §5's fixtures — which the pin derives its authority from — say it
// is the tanh approximation, by three orders of magnitude. This file follows
// the fixtures. Everything, including the mechanism that produces the
// contradiction, is in the GELU section below. Nothing else in §1 is touched.
//
// The reference module, verbatim from `ms-tok/scripts/ar_prep.py` (identical in
// train_v2.py, joint.py and nar_lora.py — only the dropout argument differs,
// and `.eval()` neutralises that):
//
//   class Tok(nn.Module):
//       s.inp  = nn.Linear(1024, 512)
//       s.pos  = nn.Parameter(torch.zeros(1, 512, 512))
//       layer  = nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True,
//                                           norm_first=True, activation="gelu")
//       s.enc  = nn.TransformerEncoder(layer, 8)          # norm=None !
//       s.norm = nn.LayerNorm(512)
//       s.head = nn.Linear(512, 32768)
//       def forward(s, x):
//           return s.head(s.norm(s.enc(s.inp(x) + s.pos[:, :x.shape[1]])))
//
// ── The five traps, each of which compiles and runs while being wrong ─────
//
//  1. THERE IS NO `enc.norm.*`. `nn.TransformerEncoder(layer, L)` was built
//     with norm=None, so the encoder stack applies NO final normalisation of
//     its own. The single final LayerNorm is `Tok.norm`, a separate module.
//     A port that adds an encoder-level norm "because transformers have one"
//     is wrong by a whole LayerNorm and will still produce plausible codes.
//     Asserted in transport too: convert-yue2-tok.py:975-977 consumes exactly
//     `norm.{weight,bias}` and its unconsumed-tensor check would have caught
//     an `enc.norm.*` if one existed (pin §1.4).
//
//  2. `in_proj_weight` IS FUSED, rows [0:512]=Wq, [512:1024]=Wk,
//     [1024:1536]=Wv — query FIRST (pin §1.5). Same order for the [1536] bias.
//     The GGUF keeps it fused (`tok.blk.N.attn_qkv.weight`, ne=(512,1536)) on
//     purpose; this file slices three free views out of the projected result,
//     which is the only place the order is spelled out in C++.
//
//  3. `norm_first=True` — both residual branches are PRE-normed and there is
//     no post-block norm. `x = x + attn(LN1(x)); x = x + ffn(LN2(x))`.
//
//  4. THE GELU IS THE TANH APPROXIMATION, not exact erf — and this is the one
//     place this file knowingly contradicts pin §1.6. It is not a judgement
//     call: the pin's own fixtures settle it, 200x either way. Full workings
//     in "GELU" below. We use `ggml_gelu` (the tanh form). `ggml_gelu_erf`
//     also exists in this ggml (ggml.h:1183, CUDA at ggml-cuda.cu:2113 /
//     :4779) and is the WRONG function here.
//
//  5. NO MASK ANYWHERE. Not causal, no key-padding mask. Every frame in the
//     window attends to every other frame INCLUDING the zero padding of a
//     short final window — that padding is part of the reference's answer, not
//     an artifact to be masked away. `ggml_soft_max_ext(..., mask=nullptr,
//     scale=1/sqrt(64), max_bias=0)`.
//
// ── Shapes and the GGML axis order ────────────────────────────────────────
//
// Host side, everything is row-major `[T, C]`: feature/logit row `t` is
// contiguous. That is byte-for-byte what the fixtures hold and what numpy
// writes, so nothing is transposed on the way in or out.
//
// In ggml that same buffer IS `ne = (C, T)` — ne0 is the fastest axis. So:
//
//   input   x        ne = (1024, T)      T <= 512
//   after inp        ne = ( 512, T)
//   tok.pos          ne = ( 512, 512) = (D, WIN); pos[:T] is a free view
//   qkv              ne = (1536, T)
//   per head         ne = (  64, 8, T)   view_3d, no copy
//   logits           ne = (32768, T)
//   codes            ne = (T)            I32, ggml_argmax over ne0
//
// `ggml_mul_mat(W, x)` with W ne=(in, out) and x ne=(in, T) gives ne=(out, T),
// which is precisely PyTorch's `Linear` on `[T, in]`. The converter already
// stores every weight that way, so there is not one transpose in this file.
//
// ── Precision: why every weight is promoted to F32 at load ────────────────
//
// `yue2_tok_head_load` sets `Yue2Loader::force_f32`, so the F16 matmul weights
// in a `--outtype f16` file are widened once, at load, into the weight buffer
// (171 MB for 42.8 M parameters — a rounding error next to the LM). Two
// reasons, and neither is "F32 is more accurate than F16":
//
//   * It pins ONE ggml_mul_mat code path. With an F32 src0 and ne12==1 the
//     CUDA backend runs a plain cublasSgemm (ggml-cuda.cu:1568). With an F16
//     src0 it picks by heuristic between mul_mat_vec, MMQ and a cuBLAS call
//     that narrows src1 to F16 as well — i.e. the ACTIVATIONS get rounded,
//     which the reference never does outside its bf16 autocast region.
//   * The oracle's stage fixtures are FP32 with autocast explicitly off, so
//     an F32 graph is the thing they can actually bisect.
//
// It does NOT recover the F16 STORAGE rounding already baked into the file.
// That is a property of the GGUF, not of this port, which is why the gate bar
// in yue2-probe.cpp is chosen from the file's own tensor type.
//
// TF32 IS ALSO A NARROWING, and ggml turns it on for you. ggml creates its
// cuBLAS handle with CUBLAS_TF32_TENSOR_OP_MATH (ggml-cuda/common.cuh), so
// every F32 cublasSgemm above runs with an 11-bit mantissa unless the process
// set NVIDIA_TF32_OVERRIDE=0 BEFORE its first CUDA context. Call
// `yue2_tok_disable_tf32()` as the first statement of main(); setting it later
// is a silent no-op, and `yue2_tok_head_load` warns loudly if nobody did.
//
// MEASURED on the head-unit fixture, f32 GGUF, RTX 5090: logits rel-L2 against
// the oracle is 8.62e-07 with TF32 off and 5.35e-04 with it on — 620x. The
// argmax is 512/512 EITHER WAY, which is the part worth remembering: TF32 is
// invisible to the code-agreement gate and shows up only in the logits. Don't
// conclude from a clean argmax that the precision regime was right.
//
// ── GELU: the pin's prose says erf; its fixtures say tanh ─────────────────
//
// Pin §1.6 states that `activation="gelu"` is `F.gelu` with
// `approximate='none'`, "the exact erf GELU, not the tanh approximation". That
// is true of `F.gelu` called directly. It is NOT true of what the reference
// actually executes, and the pin's own `07_head_unit_*` fixtures say so:
//
//   this port, ggml_gelu (tanh)      vs fixture logits: rel-L2 8.62e-07, argmax 512/512
//   this port, ggml_gelu_erf         vs fixture logits: rel-L2 1.60e-03, argmax 510/512
//
// Measured a second time in float64 on the CPU, in numpy, from the .pt
// weights — a completely independent implementation, so this is not a ggml
// quirk:
//
//   numpy float64, tanh              vs fixture logits: rel-L2 5.52e-06, argmax 512/512
//   numpy float64, erf               vs fixture logits: rel-L2 1.60e-03, argmax 510/512
//
// Three digits of agreement one way, three orders of magnitude the other. The
// two frames that disagree under erf (55 and 214) are the SAME two frames in
// both implementations, which is what a systematic activation difference looks
// like and what arithmetic noise does not.
//
// THE MECHANISM, because a measurement without one invites someone to "fix" it
// back. `ar_prep.py` builds `nn.TransformerEncoderLayer(..., batch_first=True,
// norm_first=True, activation="gelu")`, wraps 8 of them in
// `nn.TransformerEncoder`, calls `.eval()` and runs under `@torch.no_grad()` on
// CUDA. Those are exactly the conditions for PyTorch's fused encoder-layer fast
// path, and that kernel's FFN uses the TANH form. Reproduced standalone on this
// box (torch 2.10.0+cu128, RTX 5090), 8 layers of randomly initialised weights,
// no HOT-Step code involved:
//
//   nn.TransformerEncoder output vs manual erf-GELU replay : rel 1.92e-04
//   nn.TransformerEncoder output vs manual tanh-GELU replay: rel 9.53e-07
//
// And the trap inside the trap: run the SAME module on the CPU with a small
// input and the fast path is not taken, so it matches the erf replay EXACTLY
// (rel 0.0) and the tanh replay at 3.97e-05. The reference's activation
// therefore depends on the device it runs on. CUDA is the regime the shipped
// tokenizer, the published 16.1% match rate and every fixture here were
// produced in, so CUDA/tanh is what parity means and what this file
// implements. A CPU-captured oracle would disagree with all of them.
//
// If someone later re-runs the oracle on a torch build where the fast path is
// gone, this will flip back and the gate will catch it in one run — which is
// the argument for gating on the fixture rather than on the prose.
//
// ── Windowed inference over a whole track (pin §1.7) ──────────────────────
//
// `ar_prep.py::predict()`, ported line for line in yue2_tok_head_predict:
//
//   starts = range(0, max(1, T - 512 + 1), 256)            # stride WIN//2
//   if starts[-1] + 512 < T: starts += [max(0, T - 512)]   # flush-right tail
//   for s0 in starts:
//       xw = x[s0:s0+512]; n = len(xw)
//       if n < 512: xw = zero-pad to 512                   # and it IS attended
//       pred = argmax(head(xw))[:n]
//       lo = s0     + (0 if s0 == 0      else 128)         # trim WIN//4
//       hi = s0 + n - (0 if s0 + n >= T  else 128)
//       out[lo:hi] = pred[lo-s0 : hi-s0]
//
// Each 512-frame window contributes only its middle 256 frames; the first
// window keeps its left edge and the last keeps its right. Windows are written
// in ASCENDING order, so where the trims do not abut, a later window wins —
// that overwrite is load-bearing for the flush-right tail, which overlaps the
// previous window by up to 126 frames.
//
// Because predict() always pads to exactly 512, the graph is built once for
// T=512 and reused for every window of every track. `yue2_tok_head_forward`
// still accepts any T <= 512 (rebuilding on change) because the gate calls it
// directly and because `pos` really is sliced, not interpolated.

#include "yue2-model.h"  // GGUFModel + WeightCtx + Yue2Loader + yue2_find_variant

#include "backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Node budget, measured shape per encoder layer:
//   LayerNorm           3 (norm, mul, add)          x2  =  6
//   qkv proj            2 (mul_mat, add)                 =  2
//   q/k/v views         3                                =  3
//   permute + cont      6                                =  6
//   kq / softmax / kqv  3                                =  3
//   permute + cont_2d   2                                =  2
//   out proj + residual 3                                =  3
//   ffn                 5 (up, +b, gelu, down, +b)       =  5
//   ffn residual        1                                =  1
//                                                   = 31 per layer
// 8 layers = 248, plus inp (4) + final norm (3) + head (2) + argmax (1) = 258.
// Cap at 1024 for the same >2x margin yue2-vae-encode.h keeps.
#define YUE2_TOK_HEAD_MAX_NODES 1024

// ── TF32 opt-out ───────────────────────────────────────────────────────────

// Identical twin of yue2_vae_enc_disable_tf32() (yue2-vae-encode.h). Repeated
// rather than included because that header drags the entire Oobleck VAE
// encoder graph in with it, and a TU that only wants the tokenizer head should
// not pay for that. Both set the same process-wide variable; calling both is
// harmless. MUST run before the first ggml_backend_cuda_init — the CUDA driver
// reads NVIDIA_TF32_OVERRIDE when the context is created, so a later call is a
// silent no-op.
static void yue2_tok_disable_tf32() {
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", /*overwrite*/ 1);
#endif
}

static bool yue2_tok_tf32_disabled() {
    const char * v = getenv("NVIDIA_TF32_OVERRIDE");
    return v && v[0] == '0' && v[1] == '\0';
}

// ── Config ─────────────────────────────────────────────────────────────────

// Every field is read from the GGUF's yue2tok.* KV, then checked against the
// pin's constants by yue2_tok_head_validate_config. The checks are not
// paranoia about our own converter: they are what stops someone pointing this
// at a differently-shaped head and getting plausible garbage, exactly the
// failure mode pin §6 item 1 warns about for H.
struct Yue2TokHeadConfig {
    uint32_t win        = 0;  // yue2tok.head.window                 (512)
    uint32_t d          = 0;  // yue2tok.head.embedding_length       (512)
    uint32_t layers     = 0;  // yue2tok.head.block_count            (8)
    uint32_t heads      = 0;  // yue2tok.head.attention.head_count   (8)  <- NOT weight-derived
    uint32_t head_dim   = 0;  // yue2tok.head.attention.key_length   (64)
    uint32_t ff         = 0;  // yue2tok.head.feed_forward_length    (2048)
    uint32_t din        = 0;  // yue2tok.head.input_length           (1024)
    uint32_t vocab      = 0;  // yue2tok.head.vocab_size             (32768)
    float    ln_eps     = 0;  // yue2tok.head.layer_norm_epsilon     (1e-5)
    float    attn_scale = 0;  // yue2tok.head.attention.softmax_scale(1/8)
    uint32_t stride     = 0;  // yue2tok.head.window_stride          (256)
    uint32_t trim       = 0;  // yue2tok.head.window_trim            (128)
    uint32_t codec_off  = 0;  // yue2tok.codec_offset                (151853)
    uint32_t codec_size = 0;  // yue2tok.codec_size                  (32768)
    bool     norm_first = false;
    bool     has_enc_norm = true;  // must be FALSE — trap 1 above
    bool     causal       = true;  // must be FALSE — trap 5 above
    bool     qkv_fused    = false;
    // yue2tok.head.activation. Parsed for reporting, NOT validated: the
    // converter writes "gelu_erf" from pin §1.6 and the reference actually
    // runs the tanh form. See the GELU section in this file's header.
    std::string activation;
};

struct Yue2TokHeadLayer {
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;
    ggml_tensor * qkv_w   = nullptr;  // ne (D, 3D) — fused, q/k/v row order per trap 2
    ggml_tensor * qkv_b   = nullptr;  // ne (3D)
    ggml_tensor * out_w   = nullptr;  // ne (D, D)
    ggml_tensor * out_b   = nullptr;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;
    ggml_tensor * ffn_up_w   = nullptr;  // ne (D, FF)
    ggml_tensor * ffn_up_b   = nullptr;
    ggml_tensor * ffn_down_w = nullptr;  // ne (FF, D)
    ggml_tensor * ffn_down_b = nullptr;
};

struct Yue2TokHead {
    Yue2TokHeadConfig cfg;
    std::string       path;      // the GGUF this came from
    std::string       quant;     // "f16" / "f32" — the filename's type token
    ggml_type         store_type = GGML_TYPE_F32;  // tok.head.weight's type IN THE FILE
    bool              loaded     = false;

    // Weights (all promoted to F32 at load — see the precision note above).
    WeightCtx     wctx   = {};
    ggml_tensor * pos    = nullptr;  // ne (D, WIN)
    ggml_tensor * inp_w  = nullptr;  // ne (din, D)
    ggml_tensor * inp_b  = nullptr;
    std::vector<Yue2TokHeadLayer> blk;
    ggml_tensor * norm_w = nullptr;  // Tok.norm — NOT an encoder norm
    ggml_tensor * norm_b = nullptr;
    ggml_tensor * head_w = nullptr;  // ne (D, vocab)
    ggml_tensor * head_b = nullptr;

    // Backend + graph state.
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    ggml_context *       gctx        = nullptr;
    uint8_t *            gbuf        = nullptr;
    ggml_cgraph *        graph       = nullptr;
    ggml_tensor *        g_in        = nullptr;  // ne (din, T)
    ggml_tensor *        g_logits    = nullptr;  // ne (vocab, T)
    ggml_tensor *        g_codes     = nullptr;  // ne (T) I32
    int64_t              graph_T     = 0;
};

// ── Loading ────────────────────────────────────────────────────────────────

static void yue2_tok_head_free_graph(Yue2TokHead * h) {
    if (h->gctx) {
        if (h->sched) {
            ggml_backend_sched_reset(h->sched);
        }
        ggml_free(h->gctx);
        free(h->gbuf);
    }
    h->gctx     = nullptr;
    h->gbuf     = nullptr;
    h->graph    = nullptr;
    h->g_in     = nullptr;
    h->g_logits = nullptr;
    h->g_codes  = nullptr;
    h->graph_T  = 0;
}

static void yue2_tok_head_free(Yue2TokHead * h) {
    yue2_tok_head_free_graph(h);
    if (h->sched) {
        ggml_backend_sched_free(h->sched);
        h->sched = nullptr;
    }
    wctx_free(&h->wctx);
    h->blk.clear();
    h->pos = h->inp_w = h->inp_b = h->norm_w = h->norm_b = h->head_w = h->head_b = nullptr;
    if (h->backend_ref) {
        backend_release(h->backend, h->cpu_backend);
        h->backend     = nullptr;
        h->cpu_backend = nullptr;
        h->backend_ref = false;
    }
    h->loaded = false;
}

static void yue2_tok_parse_head_config(const GGUFModel & gf, Yue2TokHeadConfig * c) {
    c->win        = gf_get_u32(gf, "yue2tok.head.window");
    c->d          = gf_get_u32(gf, "yue2tok.head.embedding_length");
    c->layers     = gf_get_u32(gf, "yue2tok.head.block_count");
    c->heads      = gf_get_u32(gf, "yue2tok.head.attention.head_count");
    c->head_dim   = gf_get_u32(gf, "yue2tok.head.attention.key_length");
    c->ff         = gf_get_u32(gf, "yue2tok.head.feed_forward_length");
    c->din        = gf_get_u32(gf, "yue2tok.head.input_length");
    c->vocab      = gf_get_u32(gf, "yue2tok.head.vocab_size");
    c->ln_eps     = gf_get_f32(gf, "yue2tok.head.layer_norm_epsilon");
    c->attn_scale = gf_get_f32(gf, "yue2tok.head.attention.softmax_scale");
    c->stride     = gf_get_u32(gf, "yue2tok.head.window_stride");
    c->trim       = gf_get_u32(gf, "yue2tok.head.window_trim");
    c->codec_off  = gf_get_u32(gf, "yue2tok.codec_offset");
    c->codec_size = gf_get_u32(gf, "yue2tok.codec_size");
    c->norm_first = gf_get_bool(gf, "yue2tok.head.norm_first");
    c->has_enc_norm = gf_get_bool(gf, "yue2tok.head.has_encoder_norm");
    c->causal       = gf_get_bool(gf, "yue2tok.head.attention.causal");
    c->qkv_fused    = gf_get_bool(gf, "yue2tok.head.attention.qkv_fused");
    const char * act = gf_get_str(gf, "yue2tok.head.activation");
    c->activation    = act ? act : "";
}

// Refuse a structurally different head before any VRAM is touched. Same
// posture as yue2_validate_lm_config: accumulate, never exit.
static void yue2_tok_head_validate_config(const Yue2TokHeadConfig & c, std::vector<std::string> * errs) {
    auto need = [&](bool ok, const char * what) {
        if (!ok && errs && errs->size() < 24) {
            errs->push_back(std::string("tok head config: ") + what);
        }
    };
    need(c.win == 512, "yue2tok.head.window != 512 (pos is (1,512,512) in the checkpoint)");
    need(c.d == 512, "yue2tok.head.embedding_length != 512");
    need(c.layers == 8, "yue2tok.head.block_count != 8");
    need(c.heads == 8, "yue2tok.head.attention.head_count != 8 (pin §1.3: source-derived, not in the weights)");
    need(c.head_dim > 0 && c.heads * c.head_dim == c.d, "head_count * key_length != embedding_length");
    need(c.ff == 4 * c.d, "yue2tok.head.feed_forward_length != 4*D");
    need(c.din == 1024, "yue2tok.head.input_length != 1024 (MERT hidden width)");
    need(c.vocab == 32768, "yue2tok.head.vocab_size != 32768");
    need(c.ln_eps > 0.0f && c.ln_eps < 1e-3f, "yue2tok.head.layer_norm_epsilon out of range (want 1e-5)");
    need(c.stride == c.win / 2, "yue2tok.head.window_stride != WIN/2");
    need(c.trim == c.win / 4, "yue2tok.head.window_trim != WIN/4");
    need(c.codec_size == c.vocab, "yue2tok.codec_size != vocab_size");
    need(c.codec_off > 0, "yue2tok.codec_offset is 0 or missing (want 151853)");
    // The three that are traps, not shapes.
    need(c.norm_first, "yue2tok.head.norm_first is false — this head is pre-norm");
    need(!c.has_enc_norm, "yue2tok.head.has_encoder_norm is true — there is no enc.norm.* in this checkpoint");
    need(!c.causal, "yue2tok.head.attention.causal is true — this head is bidirectional and unmasked");
    need(c.qkv_fused, "yue2tok.head.attention.qkv_fused is false — the GGUF keeps in_proj_weight fused");
    // NOT CHECKED: yue2tok.head.activation. The converter writes "gelu_erf"
    // because pin §1.6 says so, and the pin's own fixtures say otherwise (see
    // the GELU section in this file's header — measured 200x either way). This
    // port follows the fixtures. Validating against the KV here would reject
    // every file the converter has ever written, so the KV is deliberately
    // ignored rather than trusted or "fixed" in a shipped GGUF nobody would
    // re-download. The activation is gated by --tok-head-parity instead, which
    // compares against the reference's actual output and not its label.
}

// Load the tok.* half of a yue2-tok GGUF. The mert.* tensors in the same file
// are deliberately left alone: this milestone is head-only, and the presence
// of 783 unread tensors is not an error (unlike the converter's own
// unconsumed check, which runs over a checkpoint it claims to fully
// understand).
static bool yue2_tok_head_load(const std::string & path, Yue2TokHead * h, std::string * err) {
    yue2_tok_head_free(h);

    GGUFModel gf = {};
    if (!gf_load(&gf, path.c_str())) {
        if (err) {
            *err = "cannot open GGUF: " + path;
        }
        return false;
    }

    const char *      arch_c = gf_get_str(gf, "general.architecture");
    const std::string arch   = arch_c ? arch_c : "";
    if (arch != "yue2-tok") {
        if (err) {
            *err = path + ": general.architecture is '" + arch + "', expected 'yue2-tok'";
        }
        gf_close(&gf);
        return false;
    }

    std::vector<std::string> errs;
    yue2_tok_parse_head_config(gf, &h->cfg);
    yue2_tok_head_validate_config(h->cfg, &errs);
    if (!errs.empty()) {
        if (err) {
            *err = path + ": " + errs[0] + (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        return false;
    }

    // Remember what the file STORES the matmul weights as, before we promote
    // them. The probe picks its rel-L2 bar from this: F16 storage is a ~2e-4
    // rel-L2 perturbation of every weight and the port cannot undo it.
    if (ggml_tensor * ht = ggml_get_tensor(gf.meta, "tok.head.weight")) {
        h->store_type = ht->type;
    }

    BackendPair bp = backend_init("YuE2-TokHead");
    h->backend     = bp.backend;
    h->cpu_backend = bp.cpu_backend;
    h->backend_ref = true;

    const int64_t D = h->cfg.d, FF = h->cfg.ff, DIN = h->cfg.din, V = h->cfg.vocab, WIN = h->cfg.win;
    const int     L = (int) h->cfg.layers;

    wctx_init(&h->wctx, 12 * L + 8);

    // Same loader the LM and VAE use, with force_f32 on (see Yue2Loader's own
    // comment in yue2-model.h for why). Shapes below are the CONFIG's, not the
    // file's, so a mis-sized tensor is an error and not a reinterpretation.
    Yue2Loader ld;
    ld.wctx      = &h->wctx;
    ld.gf        = &gf;
    ld.errors    = &errs;
    ld.force_f32 = true;

    h->pos   = ld.req("tok.pos", D, WIN);  // ne (D, WIN): pos[:T] is a view on ne1
    h->inp_w = ld.req("tok.inp.weight", DIN, D);
    h->inp_b = ld.req("tok.inp.bias", D);

    h->blk.assign((size_t) L, Yue2TokHeadLayer{});
    for (int i = 0; i < L; i++) {
        Yue2TokHeadLayer & b = h->blk[(size_t) i];
        const std::string  p = "tok.blk." + std::to_string(i) + ".";
        b.norm1_w    = ld.req(p + "norm1.weight", D);
        b.norm1_b    = ld.req(p + "norm1.bias", D);
        b.qkv_w      = ld.req(p + "attn_qkv.weight", D, 3 * D);
        b.qkv_b      = ld.req(p + "attn_qkv.bias", 3 * D);
        b.out_w      = ld.req(p + "attn_output.weight", D, D);
        b.out_b      = ld.req(p + "attn_output.bias", D);
        b.norm2_w    = ld.req(p + "norm2.weight", D);
        b.norm2_b    = ld.req(p + "norm2.bias", D);
        b.ffn_up_w   = ld.req(p + "ffn_up.weight", D, FF);
        b.ffn_up_b   = ld.req(p + "ffn_up.bias", FF);
        b.ffn_down_w = ld.req(p + "ffn_down.weight", FF, D);
        b.ffn_down_b = ld.req(p + "ffn_down.bias", D);
    }

    h->norm_w = ld.req("tok.norm.weight", D);
    h->norm_b = ld.req("tok.norm.bias", D);
    h->head_w = ld.req("tok.head.weight", D, V);
    h->head_b = ld.req("tok.head.bias", V);

    if (!errs.empty()) {
        if (err) {
            *err = path + ": " + errs[0] + (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        yue2_tok_head_free(h);
        return false;
    }

    if (!wctx_alloc(&h->wctx, h->backend)) {
        if (err) {
            *err = path + ": backend buffer allocation failed for the tokenizer head";
        }
        gf_close(&gf);
        yue2_tok_head_free(h);
        return false;
    }
    gf_close(&gf);  // safe: wctx_alloc has copied every byte to the backend

    h->sched = backend_sched_new(bp, YUE2_TOK_HEAD_MAX_NODES * 2);
    h->path  = path;
    {
        // "yue2-tok-<type>.gguf" -> "<type>", for logging only.
        const std::string base   = yue2_basename(path);
        const std::string prefix = "yue2-tok-";
        if (base.size() > prefix.size() + 5 && base.compare(0, prefix.size(), prefix) == 0) {
            h->quant = base.substr(prefix.size(), base.size() - prefix.size() - 5);
        }
    }
    h->loaded = true;

    const size_t bytes = h->wctx.buffer ? ggml_backend_buffer_get_size(h->wctx.buffer) : 0;
    fprintf(stderr,
            "[YuE2-TokHead] %s: L=%u D=%u H=%u FF=%u din=%u vocab=%u WIN=%u, stored %s -> F32, %.1f MB\n",
            yue2_basename(path).c_str(), h->cfg.layers, h->cfg.d, h->cfg.heads, h->cfg.ff, h->cfg.din, h->cfg.vocab,
            h->cfg.win, ggml_type_name(h->store_type), (double) bytes / (1024.0 * 1024.0));
    if (!yue2_tok_tf32_disabled()) {
        fprintf(stderr,
                "[YuE2-TokHead] WARNING: NVIDIA_TF32_OVERRIDE is not \"0\". Every matmul in this head is\n"
                "               an F32 ggml_mul_mat and ggml's cuBLAS handle runs those on TF32 tensor\n"
                "               cores (11-bit mantissa). Call yue2_tok_disable_tf32() at the top of main().\n");
    }
    return true;
}

// Find the best yue2-tok-*.gguf under <models_dir>/yue2 or <models_dir>,
// best-first by the same quant ladder the LM and VAE use (yue2_quant_rank).
//
// NOTE the ladder ranks f16 AHEAD of f32, which reads backwards here. It is
// the right call for the LM — bf16/f16 are the converter's native outputs and
// f32 is a debugging artifact that doubles the file for nothing — and it is
// not worth a second ladder just for this model. The consequence is that when
// both yue2-tok-f16.gguf and yue2-tok-f32.gguf are on disk, discovery picks
// the f16 one. For a bisect against the FP32 stage fixtures, pass the path
// explicitly (yue2-probe's --tok-gguf) instead of relying on discovery.
static bool yue2_tok_head_find(const std::string & models_dir, std::string * out_path) {
    const std::vector<std::string> dirs = { models_dir + YUE2_SEP "yue2", models_dir };
    return yue2_find_variant(dirs, "tok", out_path);
}

// ── Graph ──────────────────────────────────────────────────────────────────

// LayerNorm with a trained weight AND bias — every LayerNorm in this head has
// both (pin §1.4: "there is no bias-free variant anywhere"). ggml_norm
// normalises over ne0, which is D here.
static ggml_tensor * yue2_tok_ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// y = W x + b, with W stored [out, in] in torch order == ne (in, out) in ggml.
static ggml_tensor * yue2_tok_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * t = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, t, b) : t;
}

// One pre-norm encoder layer. `x` is ne (D, T).
static ggml_tensor * yue2_tok_head_layer(ggml_context * ctx, const Yue2TokHeadConfig & c,
                                         const Yue2TokHeadLayer & w, ggml_tensor * x) {
    const int64_t D  = c.d;
    const int64_t HD = c.head_dim;
    const int64_t NH = c.heads;
    const int64_t T  = x->ne[1];

    // ── attention branch: x = x + attn(norm1(x)) ──
    ggml_tensor * h = yue2_tok_ln(ctx, x, w.norm1_w, w.norm1_b, c.ln_eps);

    // FUSED projection, then three views. Row order is q, k, v (trap 2) and
    // this is the only place it is written down in C++ — offsets are in BYTES
    // into a contiguous ne (3D, T) F32 tensor.
    ggml_tensor * qkv = yue2_tok_linear(ctx, w.qkv_w, w.qkv_b, h);  // ne (3D, T)

    // view_3d per head: ne0 = head_dim (contiguous), ne1 = head (stride
    // head_dim floats), ne2 = frame (stride 3D floats). No copy.
    const size_t es  = ggml_element_size(qkv);
    const size_t nb1 = (size_t) HD * es;
    const size_t nb2 = qkv->nb[1];
    ggml_tensor * q  = ggml_view_3d(ctx, qkv, HD, NH, T, nb1, nb2, (size_t) (0 * D) * es);
    ggml_tensor * k  = ggml_view_3d(ctx, qkv, HD, NH, T, nb1, nb2, (size_t) (1 * D) * es);
    ggml_tensor * v  = ggml_view_3d(ctx, qkv, HD, NH, T, nb1, nb2, (size_t) (2 * D) * es);

    // (hd, head, t) -> (hd, t, head) for q and k; -> (t, hd, head) for v.
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    // NO MASK, NOT CAUSAL (trap 5). scale comes from the GGUF (1/sqrt(64)).
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // ne (T, T, NH)
    kq               = ggml_soft_max_ext(ctx, kq, /*mask*/ nullptr, c.attn_scale, /*max_bias*/ 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                     // ne (hd, t, head)
    kqv               = ggml_permute(ctx, kqv, 0, 2, 1, 3);           // ne (hd, head, t)
    ggml_tensor * att = ggml_cont_2d(ctx, kqv, D, T);                 // ne (D, T)

    att = yue2_tok_linear(ctx, w.out_w, w.out_b, att);
    x   = ggml_add(ctx, x, att);

    // ── FFN branch: x = x + linear2(gelu_tanh(linear1(norm2(x)))) ──
    ggml_tensor * f = yue2_tok_ln(ctx, x, w.norm2_w, w.norm2_b, c.ln_eps);
    f               = yue2_tok_linear(ctx, w.ffn_up_w, w.ffn_up_b, f);
    // TANH GELU, NOT exact erf. This is the one place this port knowingly
    // departs from the pin's prose, and it is the pin's own fixtures that
    // forced it — see "GELU: the pin's prose says erf; its fixtures say tanh"
    // in this file's header for the measurements and the mechanism.
    f = ggml_gelu(ctx, f);
    f               = yue2_tok_linear(ctx, w.ffn_down_w, w.ffn_down_b, f);
    return ggml_add(ctx, x, f);
}

// Full head. `x` is ne (din, T) F32 == row-major [T, din] on the host.
// Returns the logits, ne (vocab, T).
static ggml_tensor * yue2_tok_head_build(ggml_context * ctx, const Yue2TokHead & h, ggml_tensor * x) {
    const Yue2TokHeadConfig & c = h.cfg;
    const int64_t             T = x->ne[1];

    ggml_tensor * s = yue2_tok_linear(ctx, h.inp_w, h.inp_b, x);  // ne (D, T)

    // pos is SLICED, never interpolated (pin §1.6): positions 0..T-1. The view
    // is contiguous because ne0 is the full row width, so the add is a plain
    // elementwise op.
    ggml_tensor * pos = ggml_view_2d(ctx, h.pos, (int64_t) c.d, T, h.pos->nb[1], 0);
    s                 = ggml_add(ctx, s, pos);

    for (size_t i = 0; i < h.blk.size(); i++) {
        s = yue2_tok_head_layer(ctx, c, h.blk[i], s);
    }

    // Tok.norm — the ONLY final LayerNorm (trap 1). There is no encoder norm.
    s = yue2_tok_ln(ctx, s, h.norm_w, h.norm_b, c.ln_eps);
    return yue2_tok_linear(ctx, h.head_w, h.head_b, s);  // ne (vocab, T)
}

static bool yue2_tok_head_ensure_graph(Yue2TokHead * h, int64_t T, std::string * err) {
    if (h->graph && h->graph_T == T) {
        return true;
    }
    yue2_tok_head_free_graph(h);

    const size_t ctx_bytes = ggml_tensor_overhead() * (YUE2_TOK_HEAD_MAX_NODES + 64) +
                             ggml_graph_overhead_custom(YUE2_TOK_HEAD_MAX_NODES, false);
    h->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!h->gbuf) {
        if (err) {
            *err = "out of host memory allocating the tokenizer head graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, h->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(h->gbuf);
        h->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the tokenizer head graph context";
        }
        return false;
    }

    h->g_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) h->cfg.din, T);
    ggml_set_name(h->g_in, "yue2_tok_head_in");
    ggml_set_input(h->g_in);

    h->g_logits = yue2_tok_head_build(ctx, *h, h->g_in);
    ggml_set_name(h->g_logits, "yue2_tok_head_logits");
    ggml_set_output(h->g_logits);

    // argmax on the device, so predict() reads back T int32s per window
    // instead of T*32768 floats (64 MB at T=512). Ties: the CUDA kernel's
    // reduction uses a strict `>` across a warp shuffle, so an exact tie is
    // resolved by reduction order rather than by lowest index, unlike torch's
    // argmax. In a 32768-wide float logit row an exact tie is a measure-zero
    // event and the parity gate would show it immediately; it is recorded here
    // rather than worked around.
    h->g_codes = ggml_argmax(ctx, h->g_logits);
    ggml_set_name(h->g_codes, "yue2_tok_head_codes");
    ggml_set_output(h->g_codes);

    h->graph = ggml_new_graph_custom(ctx, YUE2_TOK_HEAD_MAX_NODES, false);
    ggml_build_forward_expand(h->graph, h->g_logits);
    ggml_build_forward_expand(h->graph, h->g_codes);

    ggml_backend_sched_reset(h->sched);
    if (!ggml_backend_sched_alloc_graph(h->sched, h->graph)) {
        ggml_free(ctx);
        free(h->gbuf);
        h->gbuf  = nullptr;
        h->graph = nullptr;
        if (err) {
            *err = "tokenizer head graph allocation failed (out of VRAM?) for T=" + std::to_string(T);
        }
        return false;
    }

    h->gctx    = ctx;
    h->graph_T = T;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(h->sched, h->backend);
    fprintf(stderr, "[YuE2-TokHead] Graph: T=%lld, %d nodes, %d splits, compute buffer %.0f MB\n", (long long) T,
            ggml_graph_n_nodes(h->graph), ggml_backend_sched_get_n_splits(h->sched),
            (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// ── Public API ─────────────────────────────────────────────────────────────

// One head forward over a single window.
//
//   x       : din*T contiguous floats, ROW-MAJOR [T, din] — frame t at
//             x + t*din. This is exactly the reference's `[T, 1024]`
//             instance-normed feature block and exactly what the fixtures
//             hold; nothing is transposed.
//   T       : 1 .. WIN (512). Larger is refused, because `pos` only has 512
//             trained rows and slicing past them is undefined, not clamped.
//   logits  : optional. Resized to vocab*T, ROW-MAJOR [T, vocab].
//   codes   : optional. Resized to T; argmax per frame, in [0, vocab).
//
// Not thread-safe (one cached graph per head), same contract as the VAE's.
static bool yue2_tok_head_forward(Yue2TokHead * h, const float * x, int64_t T, std::vector<float> * logits,
                                  std::vector<int32_t> * codes, std::string * err) {
    if (!h->loaded) {
        if (err) {
            *err = "tokenizer head is not loaded";
        }
        return false;
    }
    if (T <= 0 || T > (int64_t) h->cfg.win) {
        if (err) {
            *err = "T=" + std::to_string(T) + " outside 1.." + std::to_string(h->cfg.win) +
                   " (pos has exactly WIN trained rows)";
        }
        return false;
    }
    if (!yue2_tok_head_ensure_graph(h, T, err)) {
        return false;
    }

    ggml_backend_tensor_set(h->g_in, x, 0, (size_t) ((int64_t) h->cfg.din * T) * sizeof(float));
    if (ggml_backend_sched_graph_compute(h->sched, h->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "tokenizer head graph compute failed";
        }
        return false;
    }

    if (logits) {
        logits->resize((size_t) ((int64_t) h->cfg.vocab * T));
        ggml_backend_tensor_get(h->g_logits, logits->data(), 0, logits->size() * sizeof(float));
    }
    if (codes) {
        codes->resize((size_t) T);
        ggml_backend_tensor_get(h->g_codes, codes->data(), 0, codes->size() * sizeof(int32_t));
    }
    return true;
}

// The window start list from ar_prep.py::predict(), and nothing else — split
// out so the geometry can be gated without running 11 forwards, and so the
// one line that is easy to get subtly wrong (`max(1, T - WIN + 1)`, which is
// what makes a short track produce exactly one window instead of none) lives
// in a single place.
//
//   starts = list(range(0, max(1, T - WIN + 1), STRIDE))
//   if starts[-1] + WIN < T: starts.append(max(0, T - WIN))
//
// Worked example from pin §1.7, checked by the probe: T=2946 gives the ten
// strided starts 0,256,...,2304 plus a flush-right 2434, and the resulting
// write ranges cover [0, 2946) with no hole.
static void yue2_tok_head_window_starts(int64_t win, int64_t stride, int64_t T, std::vector<int64_t> * starts) {
    starts->clear();
    const int64_t limit = std::max<int64_t>(1, T - win + 1);
    for (int64_t s0 = 0; s0 < limit; s0 += stride) {
        starts->push_back(s0);
    }
    if (starts->back() + win < T) {
        starts->push_back(std::max<int64_t>(0, T - win));
    }
}

// Whole-track windowed inference — the exact port of ar_prep.py::predict()
// described at the top of this file. `x` is din*T_total contiguous floats,
// row-major [T_total, din], already instance-normed. `codes` is resized to
// T_total and filled with argmax codes in [0, vocab).
//
// Every window is zero-padded to WIN before the forward, because the
// reference does and because that padding is ATTENDED (no key-padding mask),
// so it changes the answer for the real frames next to it. Only the first `n`
// outputs of a padded window are ever used.
static bool yue2_tok_head_predict(Yue2TokHead * h, const float * x, int64_t T, std::vector<int32_t> * codes,
                                  std::string * err) {
    if (!h->loaded) {
        if (err) {
            *err = "tokenizer head is not loaded";
        }
        return false;
    }
    if (T <= 0) {
        if (err) {
            *err = "predict needs at least one frame";
        }
        return false;
    }
    const int64_t WIN    = (int64_t) h->cfg.win;
    const int64_t STRIDE = (int64_t) h->cfg.stride;  // WIN/2 = 256
    const int64_t TRIM   = (int64_t) h->cfg.trim;    // WIN/4 = 128
    const int64_t DIN    = (int64_t) h->cfg.din;

    codes->assign((size_t) T, 0);

    std::vector<int64_t> starts;
    yue2_tok_head_window_starts(WIN, STRIDE, T, &starts);

    std::vector<float>   win((size_t) (DIN * WIN));
    std::vector<int32_t> pred;
    for (int64_t s0 : starts) {
        const int64_t n = std::min<int64_t>(WIN, T - s0);
        // Zero-pad to exactly WIN. The pad is attended, not masked.
        std::memcpy(win.data(), x + s0 * DIN, (size_t) (n * DIN) * sizeof(float));
        if (n < WIN) {
            std::memset(win.data() + n * DIN, 0, (size_t) ((WIN - n) * DIN) * sizeof(float));
        }
        if (!yue2_tok_head_forward(h, win.data(), WIN, /*logits*/ nullptr, &pred, err)) {
            return false;
        }
        // Trim WIN/4 from each side, except at the true start / true end.
        const int64_t lo = s0 + (s0 == 0 ? 0 : TRIM);
        const int64_t hi = s0 + n - ((s0 + n >= T) ? 0 : TRIM);
        for (int64_t i = lo; i < hi; i++) {
            (*codes)[(size_t) i] = pred[(size_t) (i - s0)];  // ascending order: later windows win
        }
    }
    return true;
}

// code -> YuE2 semantic token id. Trivial, but it is the one place the
// CODEC_OFFSET lives on this side of the port, and it comes from the file's
// own KV rather than from a literal.
static inline int32_t yue2_tok_code_to_token(const Yue2TokHead & h, int32_t code) {
    return code + (int32_t) h.cfg.codec_off;
}
