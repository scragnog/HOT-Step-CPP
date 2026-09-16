#pragma once
// yue2/sheetsage-decoder.h — SheetSage2's BART-style decoder in GGML: a
// self-attention KV cache that grows one row per decode step, a
// cross-attention K/V pair computed ONCE per window and reused verbatim
// forever after, prefill() for the whole prompt prefix in one forward and
// step() for one token at a time — the exact two-call shape doc 20 §3.6
// describes production inference using.
//
// HOT-Step file (no acestep.cpp / upstream analog).
//
// AUTHORITY: docs/plans/yue2/20-sheetsage2-model-pin.md §3 (decoder math) and
// §6 (precision policy); docs/plans/yue2/22-sheetsage2-fixtures.md (G2: exact
// files, teacher-forced comparison rule, the two-fixture synthetic-<eos>
// exception explained below). Where this file and those documents disagree,
// the documents win. `K:/yue2-bakeoff/.../sheetsage_decoder.py` (ai-toolkit's
// plain-torch rewrite) is the literal reference ported here, not
// transformers.BartDecoder — see doc 20 §3's own header note on why.
// `SheetSageModel`/`SheetSageDecoderWeights`/`SheetSageDecoderConfig` are
// sheetsage-model.h's (a DIFFERENT lane loads them); this file only writes
// the compute graph that reads them.
//
// ── Self-attention KV cache: same one-shot-prefill / incremental-decode
// shape as yue2-lm-graph.h's Yue2ArKvCache ──
//
// `[D, capacity, Nh, 1]` F32 per layer (D=64, Nh=8 — doc 20 §3.2). Capacity is
// fixed at `max_output_seq_len` (5120) regardless of how long any particular
// window's own prefix is — doc 20 §3.6 point 5 proves the generation loop
// itself never needs more than that many total rows (prefix + generated) in
// one window, and a fresh cache is allocated per window: positions restart at
// 0 for every window's own decode loop (there is no cross-window self-cache
// carry-over — a later window's "prefix" is the overlap-prefix TEXT
// `build_overlap_prefix_tokens()` re-embeds from scratch, not a KV state
// handoff). F32, not F16 like the YuE2 AR cache: G2's gate is 100% masked-
// argmax agreement on every captured step of every window, a much tighter
// bar than the AR gate's rel-L2/agreement-rate framing, and this cache is a
// few hundred MB at most (§ below) — there is no VRAM pressure pushing
// toward F16 here the way there is for a many-thousand-token LM cache.
//
// ── Cross-attention K/V: doc 20 §3.4, computed ONCE, never touched again ──
//
// `[D, T_mem, Nh, 1]` F32 per layer, computed from the window's `memory`
// (encoder output, doc 20 §2.5) the FIRST time `sheetsage_dec_prefill()` is
// called against a freshly-allocated cache and reused VERBATIM by every
// later `sheetsage_dec_step()` call — this is `cache_static=True` in
// `sheetsage_decoder.py`'s own `_Attention.forward`, read literally: the
// merged file's `k_proj(memory)`/`v_proj(memory)` never run again after the
// first call. Never masked (doc 20 §3.4: `encoder_attention_mask` is `None`
// unconditionally and `_Attention.forward` has no mask parameter for this
// call site at all) — every decode step, at every window, cross-attends
// every one of the `T_mem` memory frames, real audio or zero-padded silence,
// unconditionally.
//
// ── Positions: +2 offset (doc 20 §3.1) ──
//
// The position id fed to `embed_positions` for the token written at
// self-cache row `t` (0-indexed from the start of THIS window's own decoder
// input sequence, prefix included) is `t + position_offset` (2).
// `sheetsage_dec_prefill()` feeds positions `0+2 .. P-1+2` for a P-token
// prefix; `sheetsage_dec_step()` feeds `cache.filled + 2`.
//
// ── Precision policy (doc 20 §6) ──
//
// Weights: NATIVE (BF16/F16, whatever the GGUF stores — sheetsage-model.h's
// loader policy, unchanged here). Activations: F32 throughout, via manual
// `ggml_soft_max_ext` attention (no flash-attn path in this file — the
// reference's own production loop runs under bf16 autocast, so plain F32
// `ggml_mul_mat`/`ggml_soft_max_ext` is already MORE accurate than what this
// is validated against, and this bring-up lane's KV tensors are small enough
// (§ above) that flash-attn's throughput would buy nothing worth the extra
// precision-debugging surface — a later performance pass can add it
// following `yue2_ar_block`'s exact precedent if the production decode loop
// ever needs it). No embedding scale (`embed_scale` is pinned to 1.0 and the
// reference's own rewrite never multiplies by it either — `sheetsage_dec_*`
// below refuse to run against a GGUF that claims otherwise, rather than
// silently doing the wrong thing). Logits come back as plain F32 `ggml_tensor`
// output with no extra upcast op needed: every activation feeding the final
// `ggml_mul_mat` against the tied embedding is already F32.
//
// GATED BY: engine/tools/yue2-probe.cpp --sheetsage-decoder-parity.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sheetsage-model.h"
#include "yue2-mert.h"  // yue2_mert_detail::ln -- LayerNorm-over-ne0, reused verbatim (same math, same shape [C,T])

#define SS2_DEC_MAX_NODES 4096

// ── KV cache: self-attention (grows) + cross-attention (fixed, set once) ───

struct SheetSageDecKvCache {
    int64_t capacity_self = 0;  // total self-attn rows this cache can ever hold (== max_output_seq_len)
    int64_t filled        = 0;  // self-attn rows [0,filled) are valid/written
    int64_t T_mem         = 0;  // this window's memory length (encoder output frame count)
    bool    cross_ready   = false;  // true once sheetsage_dec_prefill() has computed cross_k/cross_v

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::vector<ggml_tensor *> self_k, self_v;    // per layer: [D, capacity_self, Nh, 1] F32
    std::vector<ggml_tensor *> cross_k, cross_v;  // per layer: [D, T_mem, Nh, 1]        F32, written once
};

// Allocates a fresh cache for ONE window: self-attn capacity fixed at
// `capacity_self` (pass `model.cfg.max_output_seq_len`, doc 20 §3.6 point 5),
// cross-attn sized to that window's own `T_mem` (the encoder memory this
// window produced — 7500 for the pinned 300 s/24 kHz config, doc 20 §1.2,
// but this file takes it as a parameter rather than hard-coding it).
static bool sheetsage_dec_kv_cache_alloc(const SheetSageModel & m, int64_t capacity_self, int64_t T_mem,
                                         SheetSageDecKvCache * out, std::string * err) {
    *out = SheetSageDecKvCache{};
    if (capacity_self <= 0 || T_mem <= 0) {
        if (err) {
            *err = "sheetsage_dec_kv_cache_alloc: capacity_self and T_mem must both be > 0";
        }
        return false;
    }
    const int64_t D  = (int64_t) m.cfg.dec.key_length;
    const int64_t Nh = (int64_t) m.cfg.dec.head_count;
    const int     L  = (int) m.cfg.dec.block_count;

    ggml_init_params ip = { (size_t) (L * 4) * ggml_tensor_overhead() + 1024, NULL, /*no_alloc*/ true };
    out->ctx             = ggml_init(ip);
    if (!out->ctx) {
        if (err) {
            *err = "ggml_init failed for the SheetSage2 decoder KV cache context";
        }
        return false;
    }
    out->self_k.assign((size_t) L, nullptr);
    out->self_v.assign((size_t) L, nullptr);
    out->cross_k.assign((size_t) L, nullptr);
    out->cross_v.assign((size_t) L, nullptr);
    for (int i = 0; i < L; i++) {
        out->self_k[(size_t) i]  = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, capacity_self, Nh, 1);
        out->self_v[(size_t) i]  = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, capacity_self, Nh, 1);
        out->cross_k[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, T_mem, Nh, 1);
        out->cross_v[(size_t) i] = ggml_new_tensor_4d(out->ctx, GGML_TYPE_F32, D, T_mem, Nh, 1);
    }
    out->buf = ggml_backend_alloc_ctx_tensors(out->ctx, m.backend);
    if (!out->buf) {
        ggml_free(out->ctx);
        out->ctx = nullptr;
        if (err) {
            *err = "backend buffer allocation failed for the SheetSage2 decoder KV cache (out of VRAM?)";
        }
        return false;
    }
    // Defensive, not load-bearing: cross_k/cross_v are wholesale-written by
    // sheetsage_dec_prefill()'s ggml_cpy before cross_ready ever reads true,
    // and every self_k/self_v row is written by ggml_set_rows before its own
    // column is ever read (same posture as yue2_ar_kv_cache_alloc).
    ggml_backend_buffer_clear(out->buf, 0);
    out->capacity_self = capacity_self;
    out->T_mem          = T_mem;
    out->filled          = 0;
    out->cross_ready     = false;
    return true;
}

static void sheetsage_dec_kv_cache_free(SheetSageDecKvCache * c) {
    if (c->buf) {
        ggml_backend_buffer_free(c->buf);
        c->buf = nullptr;
    }
    if (c->ctx) {
        ggml_free(c->ctx);
        c->ctx = nullptr;
    }
    c->self_k.clear();
    c->self_v.clear();
    c->cross_k.clear();
    c->cross_v.clear();
    c->capacity_self = 0;
    c->filled        = 0;
    c->T_mem         = 0;
    c->cross_ready   = false;
}

// ── Attention math ──────────────────────────────────────────────────────────

// Manual F32 attention. q [D,Tq,Nh,1], k/v [D,Tk,Nh,1] -> [D,Nh,Tq,1]. Copy of
// yue2-lm-graph.h's yue2_lm_attn_f32 (which is itself a copy of MM3's
// mm3_lm_attn_f32) — same math, duplicated rather than cross-included per
// this codebase's own convention of each lane owning its tiny attention
// kernel (see yue2-lm-graph.h's file header for the precedent). `mask` may
// be null (cross-attention, doc 20 §3.4: never masked).
static ggml_tensor * ss2_dec_attn_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                      ggml_tensor * mask, float scale) {
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);                     // [Tk, Tq, Nh, 1]
    scores               = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));       // [Tk, D, Nh, 1]
    ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);                // [D, Tq, Nh, 1]
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));           // [D, Nh, Tq, 1]
}

// Self-attention with an incremental KV cache (doc 20 §3.3). `x` [H,T] is the
// RAW layer input (post-LN — there is no pre-norm here, unlike the Conformer/
// AR blocks elsewhere in this codebase: q/k/v project `x` directly). Writes
// this call's T new rows into `kcache`/`vcache` at `rows`, then attends over
// the window `[0, n_kv_win)` of the cache (which already includes the rows
// just written — same scatter-before-attend convention as
// yue2_ar_decode_step, doc 20 §3.3's own "a single new query position is
// free to attend every cached key, which is exactly correct for an
// already-causal KV cache").
static ggml_tensor * ss2_dec_self_attn(ggml_context * ctx, ggml_cgraph * gf, const SheetSageDecoderConfig & c,
                                       const SheetSageDecoderBlock & w, ggml_tensor * x, ggml_tensor * mask,
                                       ggml_tensor * rows, ggml_tensor * kcache, ggml_tensor * vcache,
                                       int64_t n_kv_win) {
    const int64_t D  = (int64_t) c.key_length;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t T  = x->ne[1];

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.self_attn_q_w, x), w.self_attn_q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.self_attn_k_w, x), w.self_attn_k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.self_attn_v_w, x), w.self_attn_v_b);

    q = ggml_reshape_4d(ctx, q, D, Nh, T, 1);
    k = ggml_reshape_4d(ctx, k, D, Nh, T, 1);
    v = ggml_reshape_4d(ctx, v, D, Nh, T, 1);

    ggml_tensor * k_w = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [D,T,Nh,1]
    ggml_tensor * v_w = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kcache, k_w, rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vcache, v_w, rows));

    ggml_tensor * k_win =
        ggml_view_4d(ctx, kcache, D, n_kv_win, Nh, 1, kcache->nb[1], kcache->nb[2], kcache->nb[3], 0);
    ggml_tensor * v_win =
        ggml_view_4d(ctx, vcache, D, n_kv_win, Nh, 1, vcache->nb[1], vcache->nb[2], vcache->nb[3], 0);

    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,1]

    const float   scale = 1.0f / std::sqrt((float) D);  // SDPA default, doc 20 §3.2
    ggml_tensor * attn  = ss2_dec_attn_f32(ctx, q4, k_win, v_win, mask, scale);
    attn                = ggml_reshape_2d(ctx, attn, H, T);
    return ggml_add(ctx, ggml_mul_mat(ctx, w.self_attn_o_w, attn), w.self_attn_o_b);
}

// Cross-attention against the STATIC memory K/V (doc 20 §3.4): Q is
// recomputed from `x` every call (it depends on the decoder's own evolving
// state), K/V are read from `cross_k`/`cross_v` VERBATIM — this function
// never writes them, only sheetsage_dec_prefill() does, once. No mask, ever.
static ggml_tensor * ss2_dec_cross_attn(ggml_context * ctx, const SheetSageDecoderConfig & c,
                                        const SheetSageDecoderBlock & w, ggml_tensor * x, ggml_tensor * cross_k,
                                        ggml_tensor * cross_v) {
    const int64_t D  = (int64_t) c.key_length;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t T  = x->ne[1];

    ggml_tensor * q  = ggml_add(ctx, ggml_mul_mat(ctx, w.cross_attn_q_w, x), w.cross_attn_q_b);
    q                = ggml_reshape_4d(ctx, q, D, Nh, T, 1);
    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D,T,Nh,1]

    const float   scale = 1.0f / std::sqrt((float) D);
    ggml_tensor * attn  = ss2_dec_attn_f32(ctx, q4, cross_k, cross_v, /*mask=*/nullptr, scale);
    attn                = ggml_reshape_2d(ctx, attn, H, T);
    return ggml_add(ctx, ggml_mul_mat(ctx, w.cross_attn_o_w, attn), w.cross_attn_o_b);
}

// One decoder layer, POST-LN, doc 20 §3.2's residual order exactly:
//
//   h, self_kv  = self_attn(x, causal_mask, past=self_cache)
//   x           = self_attn_norm(x + h)
//   h, cross_kv = cross_attn(x, memory, past=cross_cache, cache_static=True)
//   x           = cross_attn_norm(x + h)
//   x           = final_norm(x + fc2(gelu_erf(fc1(x))))
//
// `yue2_mert_detail::ln` (yue2-mert.h) is LayerNorm-over-ne0 then affine,
// exactly PyTorch's `nn.LayerNorm(d_model)` applied to a `[C,T]` tensor —
// reused verbatim rather than reimplemented, same as sheetsage-encoder.h
// reuses yue2-mert.h's block code.
static ggml_tensor * ss2_dec_layer(ggml_context * ctx, ggml_cgraph * gf, const SheetSageDecoderConfig & c,
                                   const SheetSageDecoderBlock & w, ggml_tensor * x, ggml_tensor * self_mask,
                                   ggml_tensor * rows, ggml_tensor * self_k, ggml_tensor * self_v, int64_t n_kv_win,
                                   ggml_tensor * cross_k, ggml_tensor * cross_v) {
    ggml_tensor * h = ss2_dec_self_attn(ctx, gf, c, w, x, self_mask, rows, self_k, self_v, n_kv_win);
    x               = yue2_mert_detail::ln(ctx, ggml_add(ctx, x, h), w.self_attn_norm_w, w.self_attn_norm_b,
                                           c.layer_norm_eps);

    h = ss2_dec_cross_attn(ctx, c, w, x, cross_k, cross_v);
    x = yue2_mert_detail::ln(ctx, ggml_add(ctx, x, h), w.cross_attn_norm_w, w.cross_attn_norm_b, c.layer_norm_eps);

    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_up_w, x), w.ffn_up_b);
    ff               = ggml_gelu_erf(ctx, ff);  // exact erf, doc 20 §3.2 -- "gelu_erf" activation, not tanh
    ff               = ggml_add(ctx, ggml_mul_mat(ctx, w.ffn_down_w, ff), w.ffn_down_b);
    return yue2_mert_detail::ln(ctx, ggml_add(ctx, x, ff), w.final_norm_w, w.final_norm_b, c.layer_norm_eps);
}

// token_embedding(ids) + position_embedding(positions), then
// layernorm_embedding (doc 20 §3.1) -- `positions` must already carry the +2
// offset (the caller's job: sheetsage_dec_prefill/step below build it).
static ggml_tensor * ss2_dec_embed(ggml_context * ctx, const SheetSageModel & m, ggml_tensor * ids,
                                   ggml_tensor * positions) {
    ggml_tensor * tok = ggml_get_rows(ctx, m.dec.tok_embd, ids);        // [512,T]
    ggml_tensor * pos = ggml_get_rows(ctx, m.dec.pos_embd, positions);  // [512,T]
    ggml_tensor * x   = ggml_add(ctx, tok, pos);
    return yue2_mert_detail::ln(ctx, x, m.dec.norm_embd_w, m.dec.norm_embd_b, m.cfg.dec.layer_norm_eps);
}

// ── Public API ───────────────────────────────────────────────────────────────

struct SheetSageDecodeResult {
    int64_t             T = 0;  // number of positions logits were requested at
    int64_t             V = 0;  // vocab_size
    std::vector<float> logits;  // [T, V] row-major (position-major), F32, unmasked, un-argmax'd
};

// The first forward of a window: embeds+runs the WHOLE prompt prefix
// (`ids`, length P) through the full causal self-attention stack in one
// shot, and — because this is the FIRST call against a freshly-allocated
// cache (`cache.filled` must be 0, `cache.cross_ready` must be false) —
// ALSO computes this window's cross-attention K/V from `memory` for every
// layer and writes them into `cache.cross_k`/`cross_v` (doc 20 §3.4: this
// happens exactly once per window, never again). `memory` is `[T_mem, 512]`
// row-major F32 (the SAME layout as `06_memory.f32` / `SheetSageEncodeResult
// ::memory`, and `cache.T_mem` must already equal `T_mem`, checked below) —
// ne0=channel is the fastest-varying axis in both, so it is fed to the graph
// with no transpose. `logit_positions` are row indices into `ids` to compute
// full-vocab logits at; production reads only the last one (doc 20 §3.6
// point 2, "Only the last position's logits are read"), a teacher-forced
// parity replay may want more.
static bool sheetsage_dec_prefill(const SheetSageModel & m, SheetSageDecKvCache & cache, const float * memory,
                                  int64_t T_mem, const std::vector<int32_t> & ids,
                                  const std::vector<int64_t> & logit_positions, SheetSageDecodeResult * out,
                                  std::string * err) {
    if (cache.filled != 0 || cache.cross_ready) {
        if (err) {
            *err = "sheetsage_dec_prefill: cache is not fresh (filled != 0 or cross_ready) -- prefill must be "
                   "the first call against a freshly-allocated cache";
        }
        return false;
    }
    if (cache.T_mem != T_mem) {
        if (err) {
            *err = "sheetsage_dec_prefill: T_mem does not match the cache's own allocation";
        }
        return false;
    }
    const SheetSageDecoderConfig & c = m.cfg.dec;
    if (c.embed_scale != 1.0f) {
        // doc 20 §3.1: BartConfig's scale_embedding is False for this pin and
        // sheetsage_decoder.py's own rewrite never multiplies by it either --
        // this file has no scale-embedding op to apply one correctly.
        if (err) {
            *err = "sheetsage2.decoder.embed_scale != 1.0 -- this port has no scale-embedding path (doc 20 §3.1)";
        }
        return false;
    }
    const int64_t T = (int64_t) ids.size();
    if (T <= 0 || T > cache.capacity_self) {
        if (err) {
            *err = "sheetsage_dec_prefill: ids.size() must be in (0, cache.capacity_self]";
        }
        return false;
    }
    const int64_t D  = (int64_t) c.key_length;
    const int64_t Nh = (int64_t) c.head_count;
    const int64_t H  = (int64_t) c.embedding_length;
    const int64_t V  = (int64_t) m.cfg.vocab_size;
    const int     L  = (int) c.block_count;
    const int64_t Kl = (int64_t) logit_positions.size();

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (SS2_DEC_MAX_NODES + 256) + ggml_graph_overhead_custom(SS2_DEC_MAX_NODES, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        if (err) {
            *err = "out of host memory allocating the SheetSage2 decoder prefill graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        if (err) {
            *err = "ggml_init failed for the SheetSage2 decoder prefill graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, SS2_DEC_MAX_NODES, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(in_pos);
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
    ggml_set_input(in_rows);
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);  // [n_kv=T, T], causal (past_len==0)
    ggml_set_input(in_mask);
    ggml_tensor * in_mem = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T_mem);  // encoder memory, this window
    ggml_set_input(in_mem);
    ggml_tensor * in_lidx = nullptr;
    if (Kl > 0) {
        in_lidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Kl);
        ggml_set_input(in_lidx);
    }

    // ── cross-attention K/V for every layer, computed ONCE, written into the
    // cache's persistent tensors via ggml_cpy (a wholesale copy, not
    // ggml_set_rows -- there is no incremental growth here, doc 20 §3.4) ──
    for (int l = 0; l < L; l++) {
        const SheetSageDecoderBlock & w = m.dec.blk[(size_t) l];
        ggml_tensor * ck = ggml_add(ctx, ggml_mul_mat(ctx, w.cross_attn_k_w, in_mem), w.cross_attn_k_b);
        ggml_tensor * cv = ggml_add(ctx, ggml_mul_mat(ctx, w.cross_attn_v_w, in_mem), w.cross_attn_v_b);
        ck               = ggml_reshape_4d(ctx, ck, D, Nh, T_mem, 1);
        cv               = ggml_reshape_4d(ctx, cv, D, Nh, T_mem, 1);
        ggml_tensor * ck_w = ggml_cont(ctx, ggml_permute(ctx, ck, 0, 2, 1, 3));  // [D,T_mem,Nh,1]
        ggml_tensor * cv_w = ggml_cont(ctx, ggml_permute(ctx, cv, 0, 2, 1, 3));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ck_w, cache.cross_k[(size_t) l]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, cv_w, cache.cross_v[(size_t) l]));
    }

    ggml_tensor * x = ss2_dec_embed(ctx, m, in_ids, in_pos);
    for (int l = 0; l < L; l++) {
        x = ss2_dec_layer(ctx, gf, c, m.dec.blk[(size_t) l], x, in_mask, in_rows, cache.self_k[(size_t) l],
                          cache.self_v[(size_t) l], /*n_kv_win=*/T, cache.cross_k[(size_t) l],
                          cache.cross_v[(size_t) l]);
    }

    ggml_tensor * out_logits = nullptr;
    if (Kl > 0) {
        ggml_tensor * gathered = ggml_get_rows(ctx, x, in_lidx);        // [H,Kl]
        out_logits             = ggml_mul_mat(ctx, m.dec.tok_embd, gathered);  // [V,Kl], tied embedding (doc 20 §3.5)
        ggml_set_output(out_logits);
        ggml_build_forward_expand(gf, out_logits);
    }

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, SS2_DEC_MAX_NODES);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        if (err) {
            *err = "SheetSage2 decoder prefill graph allocation failed (out of VRAM?) at T=" + std::to_string(T);
        }
        return false;
    }

    ggml_backend_tensor_set(in_ids, ids.data(), 0, (size_t) T * sizeof(int32_t));
    std::vector<int32_t> pos_host((size_t) T);
    std::vector<int64_t> rows_host((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        pos_host[(size_t) i]  = (int32_t) (i + (int64_t) c.position_offset);  // doc 20 §3.1: t + 2
        rows_host[(size_t) i] = i;
    }
    ggml_backend_tensor_set(in_pos, pos_host.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(in_rows, rows_host.data(), 0, (size_t) T * sizeof(int64_t));

    std::vector<uint16_t> mask_host((size_t) (T * T));
    for (int64_t i = 0; i < T; i++) {      // i = query position
        for (int64_t j = 0; j < T; j++) {  // j = key position
            mask_host[(size_t) (i * T + j)] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) (T * T) * sizeof(uint16_t));
    ggml_backend_tensor_set(in_mem, memory, 0, (size_t) (H * T_mem) * sizeof(float));

    if (in_lidx) {
        std::vector<int32_t> lidx((size_t) Kl);
        for (int64_t i = 0; i < Kl; i++) {
            lidx[(size_t) i] = (int32_t) logit_positions[(size_t) i];
        }
        ggml_backend_tensor_set(in_lidx, lidx.data(), 0, (size_t) Kl * sizeof(int32_t));
    }

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (!ok) {
        if (err) {
            *err = "SheetSage2 decoder prefill graph compute failed";
        }
    } else {
        cache.filled      = T;
        cache.cross_ready = true;
        out->T             = Kl;
        out->V             = V;
        if (Kl > 0) {
            out->logits.assign((size_t) (Kl * V), 0.0f);
            ggml_backend_tensor_get(out_logits, out->logits.data(), 0, (size_t) (Kl * V) * sizeof(float));
        }
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}

// One decode step: feeds `token_id` at absolute self-cache position
// `cache.filled` (the next unfilled row), attends over self-cache rows
// `[0, cache.filled]` inclusive and cross-attends the SAME cross_k/cross_v
// `sheetsage_dec_prefill()` computed for this window, and returns the
// full-vocab logits this call's own position predicts (`out` sized to
// exactly `V` floats). Advances `cache.filled` by 1 on success. Requires
// `cache.cross_ready` (call `sheetsage_dec_prefill()` first).
static bool sheetsage_dec_step(const SheetSageModel & m, SheetSageDecKvCache & cache, int32_t token_id,
                               std::vector<float> * logits_out, std::string * err) {
    if (!cache.cross_ready) {
        if (err) {
            *err = "sheetsage_dec_step: cross-attention K/V not computed yet -- call sheetsage_dec_prefill() first";
        }
        return false;
    }
    if (cache.filled >= cache.capacity_self) {
        if (err) {
            *err = "sheetsage_dec_step: self-attention cache is full (filled == capacity_self)";
        }
        return false;
    }
    const SheetSageDecoderConfig & c        = m.cfg.dec;
    const int64_t                  pos      = cache.filled;
    const int64_t                  n_kv_win = pos + 1;
    const int64_t                  V        = (int64_t) m.cfg.vocab_size;
    const int                      L        = (int) c.block_count;

    const size_t ctx_bytes =
        ggml_tensor_overhead() * (SS2_DEC_MAX_NODES + 256) + ggml_graph_overhead_custom(SS2_DEC_MAX_NODES, false);
    uint8_t * gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!gbuf) {
        if (err) {
            *err = "out of host memory allocating the SheetSage2 decoder step graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(gbuf);
        if (err) {
            *err = "ggml_init failed for the SheetSage2 decoder step graph context";
        }
        return false;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, SS2_DEC_MAX_NODES, false);

    ggml_tensor * in_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_ids);
    ggml_tensor * in_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(in_pos);
    ggml_tensor * in_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_input(in_rows);
    // [n_kv_win, 1] -- one query row, fully visible over [0,pos] (n_kv_win ==
    // pos+1 and the view always starts at cache row 0), so an all-zero mask
    // is already correct -- doc 20 §3.3: "from step 2 onward there is no mask
    // at all... a single new query position is free to attend every cached
    // key, which is exactly correct for an already-causal KV cache."
    ggml_tensor * in_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_win, 1);
    ggml_set_input(in_mask);

    ggml_tensor * x = ss2_dec_embed(ctx, m, in_ids, in_pos);  // [H,1]
    for (int l = 0; l < L; l++) {
        x = ss2_dec_layer(ctx, gf, c, m.dec.blk[(size_t) l], x, in_mask, in_rows, cache.self_k[(size_t) l],
                          cache.self_v[(size_t) l], n_kv_win, cache.cross_k[(size_t) l], cache.cross_v[(size_t) l]);
    }
    ggml_tensor * out_logits = ggml_mul_mat(ctx, m.dec.tok_embd, x);  // [V,1]
    ggml_set_output(out_logits);
    ggml_build_forward_expand(gf, out_logits);

    BackendPair           bp    = { m.backend, m.cpu_backend, strcmp(ggml_backend_name(m.backend), "CPU") != 0 };
    ggml_backend_sched_t sched = backend_sched_new(bp, SS2_DEC_MAX_NODES);
    bool                 ok    = sched && ggml_backend_sched_alloc_graph(sched, gf);
    if (!ok) {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        free(gbuf);
        if (err) {
            *err = "SheetSage2 decoder step graph allocation failed (out of VRAM?) at pos=" + std::to_string(pos);
        }
        return false;
    }

    const int32_t id32 = token_id;
    ggml_backend_tensor_set(in_ids, &id32, 0, sizeof(int32_t));
    const int32_t pos32 = (int32_t) (pos + (int64_t) c.position_offset);
    ggml_backend_tensor_set(in_pos, &pos32, 0, sizeof(int32_t));
    const int64_t row64 = pos;
    ggml_backend_tensor_set(in_rows, &row64, 0, sizeof(int64_t));
    std::vector<uint16_t> mask_host((size_t) n_kv_win, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(in_mask, mask_host.data(), 0, (size_t) n_kv_win * sizeof(uint16_t));

    ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        cache.filled = pos + 1;
        logits_out->assign((size_t) V, 0.0f);
        ggml_backend_tensor_get(out_logits, logits_out->data(), 0, (size_t) V * sizeof(float));
    } else if (err) {
        *err = "SheetSage2 decoder step graph compute failed";
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    free(gbuf);
    return ok;
}
