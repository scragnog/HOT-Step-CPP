#pragma once
// heartmula/lm-graph.h: the HeartMuLa global (backbone) LM as a KV-cached AR graph.
//
// HOT-Step file. Port of heartlib HeartMuLa.generate_frame's backbone half
// (modeling_heartmula.py) on torchtune llama3_2; the local half is local-graph.h.
//
// Design, following minimax/mm3-lm-graph.h where the models agree:
//
// A. CFG IS A BATCH OF TWO IN ne2/ne3. Row 0 is conditioned, row 1 unconditioned
//    (heartlib's uncond_mask). One graph, one weight read per step.
//
// B. THE INPUT EMBEDDING IS BUILT IN-GRAPH FROM NINE COLUMNS. heartlib sums
//    masked embeddings over 8 audio columns (audio_embeddings[code + k * 8197])
//    and one text column (text_embeddings[id]); the unconditional row replaces the
//    text embedding by unconditional_text_embedding BEFORE the mask is applied, so
//    masked text stays zero in both rows. The MuQ slot is then overwritten:
//    muq_linear(muq_embed) for the conditioned row, the unconditional embedding for
//    the other. heartlib only supports a zero MuQ embedding (ref_audio raises), so
//    muq_linear(0) is exactly muq_linear.bias and the matrix is never read. Every
//    term is a per-(position, row) gate supplied as data, so one cached graph serves
//    prompts, audio frames and the MuQ slot.
//
// C. ONE F16 KV CACHE [D, n_ctx, Nkv, B] PER LAYER, WRITTEN AT ROW = POSITION.
//    The mask is built on the host from the set of written rows, so the sparse
//    long-position fixture (a write at 4096 after 26 contiguous rows) is the same
//    code path as normal decoding. Sized from the request, zeroed once.
//
// D. CHUNKED PREFILL. Prompts run in chunks of at most HEARTMULA_PREFILL_CHUNK
//    positions, bounding activations (a long lyric is ~2k tokens x 2 rows).
//
// E. F32 ACTIVATIONS, FLASH ATTENTION ON GPU (head dim 128). The local decoder's
//    head dim 384 has no flash kernel, so it always takes the manual path.
//    HEARTMULA_NO_FLASH=1 forces manual attention here too, for parity debugging.
//
// F. OPTIONAL RUNTIME LORA (adapter.h) ON THE BACKBONE q/k/v/output PROJECTIONS ONLY.
//    The graphs are built with the adapter's factors in them, so graphs and KV rows
//    are keyed on (adapter id, scale): a switch, a reload or a new scale drops both
//    (heartmula_lm_sync_adapter) and the caller must prefill again. Scale 0 or no
//    adapter builds the base graph node for node.
//
// G. MULACOVER (upstream MuLaCover.generate_frame), only while a cover is bound by
//    heartmula_cover_bind (mulacover.h); unbound, the graph is the plain one node for
//    node. Per adapted layer N = 20..27: tag MLP h = h * (1 + scale) + shift BEFORE the
//    backbone block, from silu(style) per CFG row (Qwen vector, -1 on the unconditioned
//    row); then GACA after it (heartmula_gaca). The prompt's style slot (g->qwen_pos)
//    gets qwen_linear(style) on the conditioned row and the unconditional embedding on
//    the other, as the MuQ slot does. The context K/V never change during a request, so
//    bind writes them once; a step only adds the queries.

#include "heartmula/adapter.h"
#include "heartmula/model.h"

#include "ggml.h"
#include "hot-step-build-flags.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HEARTMULA_MAX_NODES     4096
#define HEARTMULA_KV_BUCKET     256
#define HEARTMULA_PREFILL_CHUNK 512

// ── Shared transformer pieces (also used by local-graph.h) ──────────────────

static ggml_tensor * heartmula_rms(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// torchtune's interleaved Llama-3 scaled RoPE: ggml NORMAL mode plus freq factors.
static ggml_tensor * heartmula_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, const HeartMuLaStack & s) {
    return ggml_rope_ext(ctx, x, pos, s.freq_factors, s.cfg.head_dim, GGML_ROPE_TYPE_NORMAL, 0, s.cfg.rope_base, 1.0f,
                         0.0f, 1.0f, 0.0f, 0.0f);
}

// q [D, T, Nh, B], k/v [D, n_kv, Nkv, B] -> [D, Nh, T, B]. GQA through mul_mat
// broadcast: q head h reads kv head h / (Nh / Nkv), torchtune's expand order.
static ggml_tensor * heartmula_attn_manual(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                           ggml_tensor * mask, float scale) {
    ggml_tensor * scores = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), mask, scale, 0.0f);  // [n_kv, T, Nh, B]
    ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));                              // [n_kv, D, Nkv, B]
    return ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, scores), 0, 2, 1, 3));
}

// One projection with the adapter. With home weights (adapter.h, basin re-base) the
// base term is (1 - beta) W x + beta S x; HeartMuLa adapters are plain LoRA, so the
// factor term is qwen3_linear_lora's.
static ggml_tensor * heartmula_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * home, float beta,
                                      const QwLoraPair * p, ggml_tensor * x) {
    if (!home || beta == 0.0f) {
        return qwen3_linear_lora(ctx, w, p, x);
    }
    ggml_tensor * y = ggml_mul_mat(ctx, home, x);
    if (beta != 1.0f) {
        y = ggml_add(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, w, x), 1.0f - beta), ggml_scale(ctx, y, beta));
    }
    if (p && p->A && p->B && p->scale != 0.0f) {
        y = ggml_add(ctx, y, ggml_mul_mat(ctx, p->B, ggml_scale(ctx, ggml_mul_mat(ctx, p->A, x), p->scale)));
    }
    return y;
}

// One llama block. h [H, T, B]. KV writes are expanded first so they are ordered
// before this layer's cache read (a view carries no dependency on the write).
static ggml_tensor * heartmula_block(ggml_context * ctx, ggml_cgraph * gf, const HeartMuLaStack & s,
                                     const HeartMuLaLayer & w, ggml_tensor * h, ggml_tensor * pos, ggml_tensor * mask,
                                     ggml_tensor * rows, ggml_tensor * kc, ggml_tensor * vc, int64_t n_kv, bool flash,
                                     const QwLoraLayer * lora = nullptr,
                                     const std::array<ggml_tensor *, 7> * home = nullptr, float beta = 0.0f) {
    // Site order is adapter.h's proj_name: q, k, v, o, w1, w2, w3.
    auto lin = [&](ggml_tensor * W, int site, QwLoraSlot slot, ggml_tensor * x) {
        return heartmula_linear(ctx, W, home ? (*home)[(size_t) site] : nullptr, beta, qwen3_lora_slot(lora, slot), x);
    };
    const HeartMuLaLmConfig & c   = s.cfg;
    const int64_t             D   = c.head_dim;
    const int64_t             T   = h->ne[1];
    const int64_t             B   = h->ne[2];
    ggml_tensor *             n   = heartmula_rms(ctx, h, w.sa_norm, c.rms_eps);
    ggml_tensor *             q   = ggml_reshape_4d(ctx, lin(w.q, 0, QW_LORA_Q, n), D, c.n_head, T, B);
    ggml_tensor *             k   = ggml_reshape_4d(ctx, lin(w.k, 1, QW_LORA_K, n), D, c.n_head_kv, T, B);
    ggml_tensor *             v   = ggml_reshape_4d(ctx, lin(w.v, 2, QW_LORA_V, n), D, c.n_head_kv, T, B);
    q = heartmula_rope(ctx, q, pos, s);
    k = heartmula_rope(ctx, k, pos, s);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, kc, ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)), rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, vc, ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)), rows));
    ggml_tensor * kw = ggml_view_4d(ctx, kc, D, n_kv, c.n_head_kv, B, kc->nb[1], kc->nb[2], kc->nb[3], 0);
    ggml_tensor * vw = ggml_view_4d(ctx, vc, D, n_kv, c.n_head_kv, B, vc->nb[1], vc->nb[2], vc->nb[3], 0);
    ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, T, Nh, B]

    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * attn;
    if (flash) {
        attn = ggml_flash_attn_ext(ctx, q4, kw, vw, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = heartmula_attn_manual(ctx, q4, kw, vw, mask, scale);
    }
    attn = ggml_reshape_3d(ctx, attn, D * c.n_head, T, B);
    h    = ggml_add(ctx, h, lin(w.o, 3, QW_LORA_O, attn));

    ggml_tensor * n2   = heartmula_rms(ctx, h, w.mlp_norm, c.rms_eps);
    ggml_tensor * gate = ggml_silu(ctx, lin(w.w1, 4, QW_LORA_GATE, n2));
    ggml_tensor * up   = lin(w.w3, 6, QW_LORA_UP, n2);
    return ggml_add(ctx, h, lin(w.w2, 5, QW_LORA_DOWN, ggml_mul(ctx, gate, up)));
}

// One cached graph with its own scheduler (sched_alloc_graph owns the allocation).
struct HeartMuLaSlot {
    ggml_backend_sched_t sched = nullptr;
    ggml_context *       gctx  = nullptr;
    std::vector<uint8_t> gbuf;
    ggml_cgraph *        graph = nullptr;
    int64_t              T     = 0;
    int64_t              n_kv  = 0;

    // inputs: per (position, row) flattened position-major within a row, row-major outside
    ggml_tensor * in_text  = nullptr;  // I32 [T*B]
    ggml_tensor * in_audio = nullptr;  // I32 [8*T*B] flat audio_embeddings rows
    ggml_tensor * in_wt    = nullptr;  // F32 [1, T*B] text gate
    ggml_tensor * in_wu    = nullptr;  // F32 [1, T*B] unconditional-text gate
    ggml_tensor * in_wm    = nullptr;  // F32 [1, T*B] MuQ-bias gate
    ggml_tensor * in_wa    = nullptr;  // F32 [1, 8, T*B] audio gates
    ggml_tensor * in_pos   = nullptr;  // I32 [T]
    ggml_tensor * in_rows  = nullptr;  // I64 [T]
    ggml_tensor * in_mask  = nullptr;  // F16 [n_kv, T]

    // Note G, cover graphs only
    ggml_tensor * in_style = nullptr;  // F32 [1024, 1, B] tag-MLP input per row
    ggml_tensor * in_qwen  = nullptr;  // F32 [1024] style vector for the prompt slot
    ggml_tensor * in_wq    = nullptr;  // F32 [1, T*B] style-slot gate
    ggml_tensor * in_cmask = nullptr;  // F16 [steps, T] context padding mask

    ggml_tensor * out_hidden = nullptr;  // F32 [H, 1, B] last position, normed
    ggml_tensor * out_logits = nullptr;  // F32 [8197, 1, B] codebook 0

    // Named intermediates kept as outputs for parity (HeartMuLaLmGraph::taps), [H, T, B]
    std::vector<std::pair<std::string, ggml_tensor *>> taps;
};

static void heartmula_slot_free(HeartMuLaSlot * s) {
    if (s->gctx) {
        ggml_free(s->gctx);
    }
    if (s->sched) {
        ggml_backend_sched_free(s->sched);
    }
    *s = HeartMuLaSlot{};
}

// Create the slot's ggml context and graph; the builder fills it, then alloc.
static ggml_context * heartmula_slot_begin(HeartMuLaSlot * s, const BackendPair & bp, ggml_cgraph ** gf) {
    if (!s->sched) {
        s->sched = backend_sched_new(bp, HEARTMULA_MAX_NODES);
    }
    const size_t bytes = ggml_tensor_overhead() * (HEARTMULA_MAX_NODES + 64) +
                         ggml_graph_overhead_custom(HEARTMULA_MAX_NODES, false);
    s->gbuf.assign(bytes, 0);
    ggml_init_params ip = { bytes, s->gbuf.data(), true };
    s->gctx             = ggml_init(ip);
    *gf                 = ggml_new_graph_custom(s->gctx, HEARTMULA_MAX_NODES, false);
    return s->gctx;
}

static bool heartmula_slot_alloc(HeartMuLaSlot * s, ggml_cgraph * gf, std::string * err, const char * what) {
    s->graph = gf;
    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, gf)) {
        ggml_free(s->gctx);
        s->gctx  = nullptr;
        s->graph = nullptr;
        if (err) {
            *err = std::string("HeartMuLa ") + what + " graph allocation failed (out of VRAM?)";
        }
        return false;
    }
    return true;
}

static ggml_tensor * heartmula_input(ggml_context * ctx, ggml_type t, int64_t n0, int64_t n1, int64_t n2,
                                     const char * name) {
    const int64_t ne[3] = { n0, n1, n2 };
    ggml_tensor * x     = ggml_new_tensor(ctx, t, n2 > 1 ? 3 : n1 > 1 ? 2 : 1, ne);
    ggml_set_name(x, name);
    ggml_set_input(x);
    return x;
}

// ── Global graph ────────────────────────────────────────────────────────────

struct HeartMuLaLmGraph {
    bool flash = false;
    int  B     = 2;  // CFG rows: 2 when cfg_scale != 1 (heartlib bs_size), else 1

    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
    int64_t                    n_ctx = 0;
    std::vector<uint8_t>       valid;  // KV rows written so far (mask source, note C)

    // Note F. The binding is set by the caller; the key is what the graphs and KV rows were made with.
    const HeartMuLaAdapter * adapter       = nullptr;
    HeartMuLaAdapterScales   adapter_scale;
    uint64_t                 key_id        = 0;  // 0 = base graph
    HeartMuLaAdapterScales   key_scale{ 0.0f };  // global 0 = the base key

    HeartMuLaSlot prefill;
    HeartMuLaSlot decode;

    // Note G. Bound by heartmula_cover_bind after heartmula_lm_reserve (a reserve that
    // reallocates drops it). Per cover layer the roped cross-attention and gate K/V of
    // that layer's context, F16 [384, steps, 8, B] and [12, steps, 1, B].
    const HeartMuLaCover *     cover     = nullptr;
    int64_t                    cover_len = 0;   // context steps
    int64_t                    qwen_pos  = -1;  // prompt position of the style slot
    ggml_context *             cover_ctx = nullptr;
    ggml_backend_buffer_t      cover_buf = nullptr;
    std::vector<ggml_tensor *> ca_k, ca_v, gate_k, gate_v;
    std::vector<float>         style;      // [B][1024] tag-MLP input
    std::vector<float>         qwen;       // [1024] style-slot input
    std::vector<uint8_t>       cover_pad;  // [steps] 1 = padding row
    bool                       taps = false;  // keep assembled/preN/postN for heartmula_lm_tap

    // host staging
    std::vector<int32_t>  text, audio, pos;
    std::vector<float>    wt, wu, wm, wa, wq;
    std::vector<int64_t>  rows;
    std::vector<uint16_t> mask, cmask;
};

static void heartmula_cover_unbind(HeartMuLaLmGraph * g) {
    if (g->cover_buf) {
        ggml_backend_buffer_free(g->cover_buf);
    }
    if (g->cover_ctx) {
        ggml_free(g->cover_ctx);
    }
    g->cover     = nullptr;
    g->cover_len = 0;
    g->qwen_pos  = -1;
    g->cover_ctx = nullptr;
    g->cover_buf = nullptr;
    g->ca_k.clear();
    g->ca_v.clear();
    g->gate_k.clear();
    g->gate_v.clear();
    heartmula_slot_free(&g->prefill);
    heartmula_slot_free(&g->decode);
}

static void heartmula_lm_free(HeartMuLaLmGraph * g) {
    heartmula_cover_unbind(g);
    heartmula_slot_free(&g->prefill);
    heartmula_slot_free(&g->decode);
    if (g->kv_buf) {
        ggml_backend_buffer_free(g->kv_buf);
    }
    if (g->kv_ctx) {
        ggml_free(g->kv_ctx);
    }
    *g = HeartMuLaLmGraph{};
}

static int64_t heartmula_bucket(int64_t n) {
    return (n + HEARTMULA_KV_BUCKET - 1) / HEARTMULA_KV_BUCKET * HEARTMULA_KV_BUCKET;
}

// Allocate (or grow) the KV cache for `n_ctx_needed` positions and reset it.
static bool heartmula_lm_reserve(const HeartMuLaModel & m, HeartMuLaLmGraph * g, int cfg_rows, int64_t n_ctx_needed,
                                 std::string * err) {
    const HeartMuLaLmConfig & c    = m.global.cfg;
    const int64_t             want = std::min<int64_t>(heartmula_bucket(n_ctx_needed), c.n_ctx);
    if (n_ctx_needed > c.n_ctx) {
        if (err) {
            *err = "HeartMuLa context " + std::to_string((long long) n_ctx_needed) + " exceeds " +
                   std::to_string(c.n_ctx);
        }
        return false;
    }
    if (g->n_ctx < want || g->B != cfg_rows) {
        const HeartMuLaAdapter * adapter = g->adapter;
        const HeartMuLaAdapterScales scale = g->adapter_scale;
        heartmula_lm_free(g);
        g->adapter       = adapter;
        g->adapter_scale = scale;
        g->B             = cfg_rows;
        const char * nf = std::getenv("HEARTMULA_NO_FLASH");
        g->flash        = m.bp.has_gpu && !HOT_STEP_FA_DISABLED && !(nf && nf[0] && nf[0] != '0');
        ggml_init_params ip = { (size_t) (2 * c.n_layer) * ggml_tensor_overhead() + 1024, nullptr, true };
        g->kv_ctx           = ggml_init(ip);
        for (int i = 0; i < c.n_layer; i++) {
            g->kv_k.push_back(ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F16, c.head_dim, want, c.n_head_kv, g->B));
            g->kv_v.push_back(ggml_new_tensor_4d(g->kv_ctx, GGML_TYPE_F16, c.head_dim, want, c.n_head_kv, g->B));
        }
        g->kv_buf = ggml_backend_alloc_ctx_tensors(g->kv_ctx, m.bp.backend);
        if (!g->kv_buf) {
            heartmula_lm_free(g);
            if (err) {
                *err = "HeartMuLa KV cache allocation failed for " + std::to_string((long long) want) + " positions";
            }
            return false;
        }
        // Masked window columns must hold finite values, never stray NaN bit patterns.
        ggml_backend_buffer_clear(g->kv_buf, 0);
        g->n_ctx = want;
        fprintf(stderr, "[HeartMuLa] KV cache: %lld positions x %d rows = %.2f GiB, flash=%s\n", (long long) want,
                g->B, (double) ggml_backend_buffer_get_size(g->kv_buf) / (1024.0 * 1024.0 * 1024.0),
                g->flash ? "yes" : "no");
    }
    g->valid.assign((size_t) g->n_ctx, 0);
    return true;
}

// Bind an adapter (nullptr = base) and its dials. The graphs follow on the next step.
static void heartmula_lm_set_adapter(HeartMuLaLmGraph * g, const HeartMuLaAdapter * adapter,
                                     const HeartMuLaAdapterScales & scales) {
    g->adapter       = adapter;
    g->adapter_scale = scales;
}

static void heartmula_lm_set_adapter(HeartMuLaLmGraph * g, const HeartMuLaAdapter * adapter, float scale) {
    HeartMuLaAdapterScales s;
    s.global = scale;
    heartmula_lm_set_adapter(g, adapter, s);
}

// Make the graphs and KV rows match the live binding (note F). Reads the adapter's id
// every time, so a reload of the same struct counts as a new adapter.
static void heartmula_lm_sync_adapter(HeartMuLaLmGraph * g) {
    const bool             on = g->adapter && g->adapter->loaded() && g->adapter_scale.global != 0.0f;
    const uint64_t         id = on ? g->adapter->id : 0;
    HeartMuLaAdapterScales sc = on ? g->adapter_scale : HeartMuLaAdapterScales{};
    if (!on) sc.global = 0.0f;
    if (id == g->key_id && sc == g->key_scale) {
        return;
    }
    heartmula_slot_free(&g->prefill);
    heartmula_slot_free(&g->decode);
    std::fill(g->valid.begin(), g->valid.end(), (uint8_t) 0);
    if (g->kv_buf) {
        ggml_backend_buffer_clear(g->kv_buf, 0);
    }
    g->key_id    = id;
    g->key_scale = sc;
}

// Llama-3 scaled RoPE at any head width (x [D, heads, T, B], pos [T]).
static ggml_tensor * heartmula_rope_dim(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, ggml_tensor * ff) {
    return ggml_rope_ext(ctx, x, pos, ff, (int) x->ne[0], GGML_ROPE_TYPE_NORMAL, 0, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f,
                         0.0f);
}

// Note G: upstream CrossAttentionBlock, h + tanh(gate) * attn. attn is 8 x 384 cross
// attention of norm(h) over the layer's context; gate is one-head x 12 attention of
// x_gate_proj(norm(h)) over context_gate_proj(context), then gate_proj. Queries are
// roped at the backbone positions, keys (cached) at the context row index; the mask
// drops padding rows. Head dims 384 and 12 have no flash kernel: manual attention.
static ggml_tensor * heartmula_gaca(ggml_context * ctx, const HeartMuLaCoverLayer & w, const HeartMuLaCover & c,
                                    ggml_tensor * h, ggml_tensor * pos, ggml_tensor * cmask, ggml_tensor * ck,
                                    ggml_tensor * cv, ggml_tensor * gk, ggml_tensor * gv) {
    const int64_t T = h->ne[1], B = h->ne[2];
    const int64_t D = HEARTMULA_COVER_CA_HD, G = HEARTMULA_COVER_GATE_DIM;
    ggml_tensor * n = heartmula_rms(ctx, h, w.ca_norm, 1e-5f);

    ggml_tensor * q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.ca_q, n), D, HEARTMULA_COVER_HEADS, T, B);
    q               = ggml_cont(ctx, ggml_permute(ctx, heartmula_rope_dim(ctx, q, pos, c.ff_ca), 0, 2, 1, 3));
    ggml_tensor * a = heartmula_attn_manual(ctx, q, ck, cv, cmask, 1.0f / sqrtf((float) D));
    a               = ggml_mul_mat(ctx, w.ca_o, ggml_reshape_3d(ctx, a, D * HEARTMULA_COVER_HEADS, T, B));

    ggml_tensor * x  = ggml_add(ctx, ggml_mul_mat(ctx, w.x_gate_w, n), w.x_gate_b);  // [12, T, B]
    ggml_tensor * gq = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.gate_q, x), G, 1, T, B);
    gq               = ggml_cont(ctx, ggml_permute(ctx, heartmula_rope_dim(ctx, gq, pos, c.ff_gate), 0, 2, 1, 3));
    ggml_tensor * ga = heartmula_attn_manual(ctx, gq, gk, gv, cmask, 1.0f / sqrtf((float) G));
    ga               = ggml_mul_mat(ctx, w.gate_o, ggml_reshape_3d(ctx, ga, G, T, B));
    ggml_tensor * gate = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w.gate_w, ga), w.gate_b));  // [1, T, B]
    return ggml_add(ctx, h, ggml_mul(ctx, a, gate));
}

static bool heartmula_lm_build(const HeartMuLaModel & m, HeartMuLaLmGraph * g, HeartMuLaSlot * s, int64_t T,
                               int64_t n_kv, std::string * err) {
    heartmula_slot_free(s);
    const HeartMuLaStack & st = m.global;
    const int64_t          H  = st.cfg.n_embd;
    const int64_t          B  = g->B;
    const int64_t          N  = T * B;
    ggml_cgraph *          gf = nullptr;

    // Note F: the factors, scaled by alpha / rank x the site's dials, for the live key only.
    std::vector<QwLoraLayer> lora;
    if (g->key_id) {
        const HeartMuLaAdapter * ad = g->adapter;
        if (!ad || ad->id != g->key_id || ad->attached != m.variant + "|" + m.base_fingerprint ||
            ad->layers.size() != (size_t) st.cfg.n_layer) {
            if (err) {
                *err = "HeartMuLa adapter does not belong to the loaded model";
            }
            return false;
        }
        lora = ad->layers;
        for (int i = 0; i < st.cfg.n_layer; i++) {
            for (int slot = 0; slot < QW_LORA_NSLOTS; slot++) {
                const bool mlp = slot == QW_LORA_GATE || slot == QW_LORA_UP || slot == QW_LORA_DOWN;
                lora[(size_t) i].p[slot].scale *= g->key_scale.site(i, st.cfg.n_layer, mlp);
            }
        }
    }
    ggml_context * ctx = heartmula_slot_begin(s, m.bp, &gf);

    s->in_text  = heartmula_input(ctx, GGML_TYPE_I32, N, 1, 1, "hm_text");
    s->in_audio = heartmula_input(ctx, GGML_TYPE_I32, HEARTMULA_CODEBOOKS * N, 1, 1, "hm_audio");
    s->in_wt    = heartmula_input(ctx, GGML_TYPE_F32, 1, N, 1, "hm_wt");
    s->in_wu    = heartmula_input(ctx, GGML_TYPE_F32, 1, N, 1, "hm_wu");
    s->in_wm    = heartmula_input(ctx, GGML_TYPE_F32, 1, N, 1, "hm_wm");
    s->in_wa    = heartmula_input(ctx, GGML_TYPE_F32, 1, HEARTMULA_CODEBOOKS, N, "hm_wa");
    s->in_pos   = heartmula_input(ctx, GGML_TYPE_I32, T, 1, 1, "hm_pos");
    s->in_rows  = heartmula_input(ctx, GGML_TYPE_I64, T, 1, 1, "hm_rows");
    s->in_mask  = heartmula_input(ctx, GGML_TYPE_F16, n_kv, T, 1, "hm_mask");

    // Note B: h = sum_k wa_k audio_k + wt text + wu uncond + wm muq_bias.
    ggml_tensor * a = ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.audio_embd, s->in_audio), H, HEARTMULA_CODEBOOKS, N);
    a               = ggml_mul(ctx, a, s->in_wa);
    ggml_tensor * h = ggml_view_2d(ctx, a, H, N, a->nb[2], 0);
    for (int k = 1; k < HEARTMULA_CODEBOOKS; k++) {
        h = ggml_add(ctx, h, ggml_view_2d(ctx, a, H, N, a->nb[2], (size_t) k * a->nb[1]));
    }
    h = ggml_add(ctx, h, ggml_mul(ctx, ggml_get_rows(ctx, m.text_embd, s->in_text), s->in_wt));
    h = ggml_add(ctx, h, ggml_mul(ctx, ggml_repeat(ctx, m.uncond_embd, h), s->in_wu));
    h = ggml_add(ctx, h, ggml_mul(ctx, ggml_repeat(ctx, m.muq_bias, h), s->in_wm));

    // Note G: the style slot, the tag-MLP input and the context mask.
    const HeartMuLaCover * cv    = g->cover;
    ggml_tensor *          style = nullptr;
    if (cv) {
        s->in_style = heartmula_input(ctx, GGML_TYPE_F32, HEARTMULA_COVER_STYLE_DIM, 1, B, "hm_style");
        s->in_qwen  = heartmula_input(ctx, GGML_TYPE_F32, HEARTMULA_COVER_STYLE_DIM, 1, 1, "hm_qwen");
        s->in_wq    = heartmula_input(ctx, GGML_TYPE_F32, 1, N, 1, "hm_wq");
        s->in_cmask = heartmula_input(ctx, GGML_TYPE_F16, g->cover_len, T, 1, "hm_cmask");
        ggml_tensor * qp = ggml_add(ctx, ggml_mul_mat(ctx, cv->qwen_w, s->in_qwen), cv->qwen_b);  // [H]
        h     = ggml_add(ctx, h, ggml_mul(ctx, ggml_repeat(ctx, qp, h), s->in_wq));
        style = ggml_silu(ctx, s->in_style);
    }
    h = ggml_reshape_3d(ctx, h, H, T, B);
    auto tap = [&](const std::string & name, ggml_tensor * t) {
        if (!g->taps) return t;
        t = ggml_cont(ctx, t);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        s->taps.push_back({ name, t });
        return t;
    };
    h = tap("assembled", h);

    for (int i = 0; i < st.cfg.n_layer; i++) {
        const int                   ci = i - HEARTMULA_COVER_FIRST;
        const HeartMuLaCoverLayer * cl = cv && ci >= 0 && ci < HEARTMULA_COVER_LAYERS ? &cv->layers[(size_t) ci] : nullptr;
        if (cl) {
            // upstream StyleMLP: (scale, shift) = mlp(silu(style)).chunk(2)
            ggml_tensor * ss    = ggml_add(ctx, ggml_mul_mat(ctx, cl->tag_w, style), cl->tag_b);  // [2H, 1, B]
            ggml_tensor * scale = ggml_cont(ctx, ggml_view_3d(ctx, ss, H, 1, B, ss->nb[1], ss->nb[2], 0));
            ggml_tensor * shift =
                ggml_cont(ctx, ggml_view_3d(ctx, ss, H, 1, B, ss->nb[1], ss->nb[2], (size_t) H * ss->nb[0]));
            h = ggml_add(ctx, ggml_add(ctx, h, ggml_mul(ctx, h, scale)), shift);
        }
        h = heartmula_block(ctx, gf, st, st.layers[(size_t) i], h, s->in_pos, s->in_mask, s->in_rows, g->kv_k[i],
                            g->kv_v[i], n_kv, g->flash, lora.empty() ? nullptr : &lora[(size_t) i],
                            lora.empty() || g->adapter->home.empty() ? nullptr : &g->adapter->home[(size_t) i],
                            g->key_scale.rebase);
        if (cl) {
            h = tap("pre" + std::to_string(i), h);
            h = heartmula_gaca(ctx, *cl, *cv, h, s->in_pos, s->in_cmask, g->ca_k[(size_t) ci], g->ca_v[(size_t) ci],
                               g->gate_k[(size_t) ci], g->gate_v[(size_t) ci]);
            h = tap("post" + std::to_string(i), h);
        }
    }
    ggml_tensor * last = ggml_view_3d(ctx, h, H, 1, B, h->nb[1], h->nb[2], (size_t) (T - 1) * h->nb[1]);
    s->out_hidden      = heartmula_rms(ctx, ggml_cont(ctx, last), st.norm, st.cfg.rms_eps);
    s->out_logits      = ggml_mul_mat(ctx, m.c0_head, s->out_hidden);
    ggml_set_output(s->out_hidden);
    ggml_set_output(s->out_logits);
    ggml_build_forward_expand(gf, s->out_hidden);
    ggml_build_forward_expand(gf, s->out_logits);
    s->T    = T;
    s->n_kv = n_kv;
    return heartmula_slot_alloc(s, gf, err, T > 1 ? "prefill" : "decode");
}

// One input position: 9 columns of ids and masks, heartlib layout (text last).
struct HeartMuLaPosition {
    int32_t id[HEARTMULA_COLUMNS]   = {};
    bool    mask[HEARTMULA_COLUMNS] = {};
};

// Run T positions (shared by every CFG row) at absolute `positions`. `muq_at` is
// the index within this call of the MuQ slot, or -1. Writes the last position's
// normed hidden [B][H] and codebook-0 logits [B][8197].
static bool heartmula_lm_step(const HeartMuLaModel & m, HeartMuLaLmGraph * g, const HeartMuLaPosition * in,
                              const int64_t * positions, int64_t T, int64_t muq_at, float * out_hidden,
                              float * out_logits, std::string * err) {
    heartmula_lm_sync_adapter(g);
    const int64_t B = g->B;
    int64_t       hi = 0;
    for (int64_t i = 0; i < T; i++) {
        if (positions[i] < 0 || positions[i] >= g->n_ctx) {
            if (err) {
                *err = "HeartMuLa position " + std::to_string((long long) positions[i]) + " is outside the KV cache";
            }
            return false;
        }
        hi = std::max(hi, positions[i] + 1);
    }
    const int64_t  n_kv = std::min(heartmula_bucket(hi), g->n_ctx);
    HeartMuLaSlot * s   = T > 1 ? &g->prefill : &g->decode;
    if (!s->graph || s->T != T || s->n_kv != n_kv) {
        if (!heartmula_lm_build(m, g, s, T, n_kv, err)) {
            return false;
        }
    }

    const int64_t N = T * B;
    g->text.assign((size_t) N, 0);
    g->audio.assign((size_t) N * HEARTMULA_CODEBOOKS, 0);
    g->wt.assign((size_t) N, 0.0f);
    g->wu.assign((size_t) N, 0.0f);
    g->wm.assign((size_t) N, 0.0f);
    g->wq.assign((size_t) N, 0.0f);
    g->wa.assign((size_t) N * HEARTMULA_CODEBOOKS, 0.0f);
    for (int64_t r = 0; r < B; r++) {
        const bool uncond = r == 1;
        for (int64_t i = 0; i < T; i++) {
            const HeartMuLaPosition & p = in[i];
            const size_t              n = (size_t) (r * T + i);
            for (int k = 0; k < HEARTMULA_CODEBOOKS; k++) {
                g->audio[n * HEARTMULA_CODEBOOKS + k] = p.id[k] + k * HEARTMULA_AUDIO_VOCAB;
                g->wa[n * HEARTMULA_CODEBOOKS + k]    = p.mask[k] ? 1.0f : 0.0f;
            }
            const bool text = p.mask[HEARTMULA_COLUMNS - 1];
            g->text[n]      = p.id[HEARTMULA_COLUMNS - 1];
            g->wt[n]        = text && !uncond ? 1.0f : 0.0f;
            g->wu[n]        = text && uncond ? 1.0f : 0.0f;
            if (i == muq_at) {
                // heartlib overwrites the summed embedding at the MuQ index.
                std::fill(g->wa.begin() + (ptrdiff_t) (n * HEARTMULA_CODEBOOKS),
                          g->wa.begin() + (ptrdiff_t) ((n + 1) * HEARTMULA_CODEBOOKS), 0.0f);
                g->wt[n] = 0.0f;
                g->wu[n] = uncond ? 1.0f : 0.0f;
                g->wm[n] = uncond ? 0.0f : 1.0f;
            }
            if (g->cover && positions[i] == g->qwen_pos) {
                // Note G: the style slot, overwritten like the MuQ slot.
                std::fill(g->wa.begin() + (ptrdiff_t) (n * HEARTMULA_CODEBOOKS),
                          g->wa.begin() + (ptrdiff_t) ((n + 1) * HEARTMULA_CODEBOOKS), 0.0f);
                g->wt[n] = 0.0f;
                g->wu[n] = uncond ? 1.0f : 0.0f;
                g->wq[n] = uncond ? 0.0f : 1.0f;
            }
        }
    }
    g->pos.resize((size_t) T);
    g->rows.resize((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        g->pos[(size_t) i]                  = (int32_t) positions[i];
        g->rows[(size_t) i]                 = positions[i];
        g->valid[(size_t) positions[i]] = 1;
    }
    g->mask.resize((size_t) (n_kv * T));
    for (int64_t i = 0; i < T; i++) {
        for (int64_t j = 0; j < n_kv; j++) {
            const bool see                       = g->valid[(size_t) j] && j <= positions[i];
            g->mask[(size_t) (i * n_kv + j)] = ggml_fp32_to_fp16(see ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(s->in_text, g->text.data(), 0, g->text.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s->in_audio, g->audio.data(), 0, g->audio.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s->in_wt, g->wt.data(), 0, g->wt.size() * sizeof(float));
    ggml_backend_tensor_set(s->in_wu, g->wu.data(), 0, g->wu.size() * sizeof(float));
    ggml_backend_tensor_set(s->in_wm, g->wm.data(), 0, g->wm.size() * sizeof(float));
    ggml_backend_tensor_set(s->in_wa, g->wa.data(), 0, g->wa.size() * sizeof(float));
    ggml_backend_tensor_set(s->in_pos, g->pos.data(), 0, g->pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(s->in_rows, g->rows.data(), 0, g->rows.size() * sizeof(int64_t));
    ggml_backend_tensor_set(s->in_mask, g->mask.data(), 0, g->mask.size() * sizeof(uint16_t));
    if (g->cover) {
        g->cmask.resize((size_t) (g->cover_len * T));
        for (int64_t i = 0; i < T; i++) {
            for (int64_t j = 0; j < g->cover_len; j++) {
                g->cmask[(size_t) (i * g->cover_len + j)] =
                    ggml_fp32_to_fp16(g->cover_pad[(size_t) j] ? -INFINITY : 0.0f);
            }
        }
        ggml_backend_tensor_set(s->in_style, g->style.data(), 0, g->style.size() * sizeof(float));
        ggml_backend_tensor_set(s->in_qwen, g->qwen.data(), 0, g->qwen.size() * sizeof(float));
        ggml_backend_tensor_set(s->in_wq, g->wq.data(), 0, g->wq.size() * sizeof(float));
        ggml_backend_tensor_set(s->in_cmask, g->cmask.data(), 0, g->cmask.size() * sizeof(uint16_t));
    }
    if (ggml_backend_sched_graph_compute(s->sched, s->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "HeartMuLa global graph compute failed";
        }
        return false;
    }
    if (out_hidden) {
        ggml_backend_tensor_get(s->out_hidden, out_hidden, 0, ggml_nbytes(s->out_hidden));
    }
    if (out_logits) {
        ggml_backend_tensor_get(s->out_logits, out_logits, 0, ggml_nbytes(s->out_logits));
    }
    return true;
}

// A tap of the last prefill (T > 1) or decode step, [B][T][H] as torch lays it out.
static bool heartmula_lm_tap(const HeartMuLaLmGraph & g, bool prefill, const std::string & name,
                             std::vector<float> * out) {
    for (const auto & t : (prefill ? g.prefill : g.decode).taps) {
        if (t.first == name) {
            out->resize((size_t) ggml_nelements(t.second));
            ggml_backend_tensor_get(t.second, out->data(), 0, ggml_nbytes(t.second));
            return true;
        }
    }
    return false;
}

// Prefill a prompt in chunks (note D). Positions are 0..T-1 as heartlib's `pos`.
static bool heartmula_lm_prefill(const HeartMuLaModel & m, HeartMuLaLmGraph * g, const HeartMuLaPosition * in,
                                 int64_t T, int64_t muq_idx, float * out_hidden, float * out_logits, std::string * err,
                                 int64_t chunk = HEARTMULA_PREFILL_CHUNK) {
    std::vector<int64_t> pos;
    for (int64_t start = 0; start < T; start += chunk) {
        const int64_t n = std::min(chunk, T - start);
        pos.resize((size_t) n);
        for (int64_t i = 0; i < n; i++) {
            pos[(size_t) i] = start + i;
        }
        const bool    last = start + n == T;
        const int64_t muq  = muq_idx >= start && muq_idx < start + n ? muq_idx - start : -1;
        if (!heartmula_lm_step(m, g, in + start, pos.data(), n, muq, last ? out_hidden : nullptr,
                               last ? out_logits : nullptr, err)) {
            return false;
        }
    }
    return true;
}
