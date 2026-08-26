#pragma once
// train/lm-kvprefix.h — a FROZEN, NO-GRAD history prefix for the LM trainer.
//
// docs/plans/2026-08-26-lm-frozen-kv-prefix.md
//
// WHY THIS EXISTS
//
// A training crop taken at frame c0 is presented at its true RoPE position but
// with an EMPTY context: ~750 keys of evidence labelled as position 3000. Early
// layers work in local windows and do not care; late layers read out and do not
// care. The middle third of the stack is the part doing long-range aggregation,
// and it is being taught to produce position-3000 behaviour from
// position-750 evidence. That is the band Rob has been switching off at render
// time with `scaleMid 0`, and it is the band this file exists to repair.
//
// The reason it was never fixed by simply cropping longer is memory, not
// principle: the backward retains [S, S, heads] attention scores and ggml's
// flash-attn has no backward, so peak grows as S^2 and the crop ceilings at
// 4272 frames. But a prefix needs NO backward. Running it forward-only and
// keeping just K and V costs Nkv*D floats per position per layer — 288 KB per
// position in F32 across MM3's 36 layers, so a whole 5099-frame track is about
// 1.4 GB against the 27.7 GB the quadratic path spends to reach 84 % coverage.
//
// WHAT THE ADAPTER LOSES
//
// No gradient with respect to the adapter at prefix positions. It keeps
// query-side gradient at every supervised position, which is precisely "learn
// to attend back correctly" — the defect being fixed. And because the prefill
// runs WITH the adapter live, the window conditions on the adapter's own
// accumulated drift rather than on a clean teacher context.
//
// LAYOUT
//
//   prefill   [ prompt 0..P-1 ][ frames c0-Npfx .. c0-1 ]   K/V stored, Q cols
//   window    [ prompt 0..P-1 ][ frames c0 .. c0+Fin-1  ]   trained as before
//
// The window keeps its prompt, so every existing supervision index — n_masked,
// s_tr, targets, the chunked CE head, the depth loss — is untouched. The
// prefill stores the prompt's K/V too (it is a plain causal stream, and
// special-casing it would be a second thing to get wrong), and the window's
// mask blanks those columns so no key is ever softmaxed over twice.
//
// THE ORDERING HAZARD, AND WHY THERE ARE TWO GRAPHS PER CHUNK
//
// A prefill chunk both READS the store (as the base of its attention splice,
// whose tail columns must still be zero) and WRITES it (its own K/V). ggml
// guarantees ordering only through data dependencies, and those two touch the
// same tensor without one. So a chunk captures into a small staging buffer in
// graph A and a separate graph B moves staging into the store. Cheap, and it
// cannot race.

#include "train/lm-common.h"
#include "train/lm-data.h"
#include "train/lm-graph.h"

#include <algorithm>
#include <string>
#include <vector>

struct LmKvPrefix {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // Indexed by ABSOLUTE layer; null outside [layer_lo, layer_hi).
    std::vector<ggml_tensor *> k, v;    // [Nkv*D, cap]
    std::vector<ggml_tensor *> sk, sv;  // [Nkv*D, chunk] staging

    ggml_tensor * t_pos = nullptr;  // [chunk]      I32, absolute RoPE positions
    ggml_tensor * t_msk = nullptr;  // [msk_cap]    F32, rectangular causal mask

    int64_t row     = 0;  // Nkv * D
    int64_t cap     = 0;  // q_max + s_max: the window accs its own K/V at column n
    int64_t q_max   = 0;
    int64_t s_max   = 0;
    int64_t msk_cap = 0;
    int     chunk   = 0;
    int     layer_lo = 0, layer_hi = 0;

    int n = 0;  // columns currently filled — the window's kv_pfx

    std::vector<float>   msk_scratch;
    std::vector<int32_t> pos_scratch;

    size_t bytes() const { return buf ? ggml_backend_buffer_get_size(buf) : 0; }
};

static void lm_kvprefix_free(LmKvPrefix * st) {
    if (st->buf) {
        ggml_backend_buffer_free(st->buf);
    }
    if (st->ctx) {
        ggml_free(st->ctx);
    }
    *st = LmKvPrefix{};
}

// VRAM the store will hold, before allocating it — the VRAM planner needs this
// to decide a crop, and a planner that guesses is a planner that OOMs mid-run.
static size_t lm_kvprefix_bytes(const Qwen3LMConfig & c, int layer_lo, int layer_hi, int64_t q_max, int64_t s_max,
                                int chunk) {
    const int64_t row = (int64_t) c.n_kv_heads * (int64_t) c.head_dim;
    const int64_t nl  = (int64_t) (layer_hi - layer_lo);
    const int64_t cap = q_max + s_max;
    size_t        b   = 0;
    b += (size_t) nl * 2u * (size_t) row * (size_t) cap * sizeof(float);    // store
    b += (size_t) nl * 2u * (size_t) row * (size_t) chunk * sizeof(float);  // staging
    b += (size_t) chunk * sizeof(int32_t);
    b += (size_t) (q_max + std::max<int64_t>(s_max, chunk)) * (size_t) std::max<int64_t>(s_max, chunk) *
         sizeof(float);
    return b;
}

static bool lm_kvprefix_alloc(LmKvPrefix * st, Qwen3LM * lm, int layer_lo, int layer_hi, int64_t q_max,
                              int64_t s_max, int chunk, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    if (q_max <= 0 || s_max <= 0 || chunk <= 0) {
        *err = "invalid KV-prefix configuration (q_max / s_max / chunk)";
        return false;
    }
    if (layer_hi <= layer_lo) {
        *err = "invalid KV-prefix layer range";
        return false;
    }

    st->row      = (int64_t) c.n_kv_heads * (int64_t) c.head_dim;
    st->q_max    = q_max;
    st->s_max    = s_max;
    st->cap      = q_max + s_max;
    st->chunk    = chunk;
    st->layer_lo = layer_lo;
    st->layer_hi = layer_hi;
    // One buffer serves both the window ([q_max + S, S]) and a prefill chunk
    // ([q_max + chunk, chunk]); size it for whichever is larger.
    const int64_t w = std::max<int64_t>(s_max, chunk);
    st->msk_cap     = (q_max + w) * w;

    const int nl = layer_hi - layer_lo;
    {
        ggml_init_params p = { (size_t) (4 * nl + 8) * ggml_tensor_overhead(), nullptr, true };
        st->ctx            = ggml_init(p);
    }
    if (!st->ctx) {
        *err = "cannot create the KV-prefix context";
        return false;
    }

    st->k.assign((size_t) c.n_layers, nullptr);
    st->v.assign((size_t) c.n_layers, nullptr);
    st->sk.assign((size_t) c.n_layers, nullptr);
    st->sv.assign((size_t) c.n_layers, nullptr);
    for (int l = layer_lo; l < layer_hi; l++) {
        char nm[40];
        st->k[(size_t) l]  = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, st->row, st->cap);
        st->v[(size_t) l]  = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, st->row, st->cap);
        st->sk[(size_t) l] = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, st->row, chunk);
        st->sv[(size_t) l] = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, st->row, chunk);
        snprintf(nm, sizeof(nm), "kvpfx.k.%d", l);
        ggml_set_name(st->k[(size_t) l], nm);
        snprintf(nm, sizeof(nm), "kvpfx.v.%d", l);
        ggml_set_name(st->v[(size_t) l], nm);
        for (ggml_tensor * t : { st->k[(size_t) l], st->v[(size_t) l], st->sk[(size_t) l], st->sv[(size_t) l] }) {
            ggml_set_input(t);
        }
    }
    st->t_pos = ggml_new_tensor_1d(st->ctx, GGML_TYPE_I32, chunk);
    st->t_msk = ggml_new_tensor_1d(st->ctx, GGML_TYPE_F32, st->msk_cap);
    ggml_set_name(st->t_pos, "kvpfx.pos");
    ggml_set_name(st->t_msk, "kvpfx.mask");
    ggml_set_input(st->t_pos);
    ggml_set_input(st->t_msk);

    st->buf = ggml_backend_alloc_ctx_tensors(st->ctx, lm->backend);
    if (!st->buf) {
        *err = "KV-prefix buffer allocation failed";
        return false;
    }
    ggml_backend_buffer_clear(st->buf, 0);
    st->pos_scratch.resize((size_t) chunk);
    st->n = 0;

    fprintf(stderr, "[train-lm] kv prefix: layers [%d,%d) x %lld cols x %lld f32 x2 = %.1f MB (chunk %d)\n", layer_lo,
            layer_hi, (long long) st->cap, (long long) st->row, st->bytes() / 1048576.0, chunk);
    return true;
}

/** Build the store for a Q-position stream.
 *
 *  `embed(ctx, user, i0, n)` returns the [H, n] hidden input for stream
 *  positions i0 .. i0+n-1; `pos` holds Q absolute RoPE positions. `opts` must
 *  be the SAME LmLayerOpts the trained window will use (rank mask included) —
 *  a prefix computed under a different subnetwork is a different model, and the
 *  window would be conditioning on a context its own weights never produced.
 *
 *  Forward only. Nothing here is a parameter and no graph carries gradients. */
static bool lm_kvprefix_run(LmKvPrefix * st, Qwen3LM * lm, ggml_backend_sched_t sched, std::vector<uint8_t> & arena,
                            const LmLayerOpts & opts,
                            ggml_tensor * (*embed) (ggml_context *, void *, int64_t, int64_t), void * user,
                            const int32_t * pos, int64_t Q, std::string * err) {
    const Qwen3LMConfig & c = lm->cfg;
    if (Q > st->q_max) {
        *err = "KV-prefix stream longer than the allocated capacity";
        return false;
    }
    GGML_ASSERT(opts.attn_head_block == 0 && "the blocked attention path cannot fill a KV prefix");
    GGML_ASSERT(opts.wt == nullptr && "the prefill is forward-only; a Lever A collector would be dead nodes");

    // The window views [0, n + S) of the store and accs its own K/V at column
    // n, so columns n .. n+S MUST be zero. A previous micro-step with a longer
    // prefix leaves stale values there, so clear the whole store — a memset of
    // a couple of gigabytes is microseconds against a multi-second step, and
    // the alternative is a correctness bug that only shows on short crops.
    ggml_backend_buffer_clear(st->buf, 0);
    st->n = 0;
    if (Q == 0) {
        return true;
    }

    for (int64_t i0 = 0; i0 < Q; i0 += st->chunk) {
        const int64_t n = std::min<int64_t>(st->chunk, Q - i0);

        for (int64_t j = 0; j < n; j++) {
            st->pos_scratch[(size_t) j] = pos[i0 + j];
        }
        ggml_backend_tensor_set(st->t_pos, st->pos_scratch.data(), 0, (size_t) n * sizeof(int32_t));
        // A plain causal stream: this chunk sees everything already stored plus
        // its own past. pfx_lo 0 / n_prompt 0 — the prompt is part of the
        // stream here, and only the WINDOW has to skip it.
        lm_causal_mask_prefix((int) i0, 0, (int) n, 0, &st->msk_scratch);
        ggml_backend_tensor_set(st->t_msk, st->msk_scratch.data(), 0, st->msk_scratch.size() * sizeof(float));

        // ── graph A: embedding + every layer, capturing K/V into staging ──
        {
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 8192, /*grads=*/false);

            ggml_tensor * pv = ggml_view_1d(ctx, st->t_pos, n, 0);
            ggml_tensor * mv = ggml_view_2d(ctx, st->t_msk, i0 + n, n, (size_t) (i0 + n) * sizeof(float), 0);
            ggml_tensor * h  = embed(ctx, user, i0, n);
            if (!h) {
                ggml_free(ctx);
                *err = "KV-prefix embedding builder failed";
                return false;
            }

            std::vector<LmKvCapture> caps((size_t) (st->layer_hi - st->layer_lo));
            for (int l = st->layer_lo; l < st->layer_hi; l++) {
                LmKvCapture & cap = caps[(size_t) (l - st->layer_lo)];
                cap.k_dst         = ggml_view_2d(ctx, st->sk[(size_t) l], st->row, n, st->sk[(size_t) l]->nb[1], 0);
                cap.v_dst         = ggml_view_2d(ctx, st->sv[(size_t) l], st->row, n, st->sv[(size_t) l]->nb[1], 0);

                LmLayerOpts lo = opts;
                lo.kv_k        = st->k[(size_t) l];
                lo.kv_v        = st->v[(size_t) l];
                lo.kv_pfx      = (int) i0;
                lo.kv_cap      = &cap;
                h              = lm_train_layer(ctx, c, &lm->layers[l], h, pv, mv, (int) n, lo);
            }
            // The hidden output of the last layer is the thing that carries the
            // stream forward WITHIN this graph; nothing outside needs it, so the
            // graph's roots are the capture copies. Expanding h too would keep a
            // final norm alive for no reason.
            for (const LmKvCapture & cap : caps) {
                for (ggml_tensor * t : cap.nodes) {
                    ggml_build_forward_expand(gf, t);
                }
            }
            ggml_backend_sched_reset(sched);
            const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
            if (!ok) {
                *err = "KV-prefix forward failed (lower --prefix-chunk?)";
                return false;
            }
        }

        // ── graph B: staging -> store at column i0 (see the hazard note) ──
        {
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 512, /*grads=*/false);
            for (int l = st->layer_lo; l < st->layer_hi; l++) {
                ggml_tensor * ks = ggml_view_2d(ctx, st->sk[(size_t) l], st->row, n, st->sk[(size_t) l]->nb[1], 0);
                ggml_tensor * vs = ggml_view_2d(ctx, st->sv[(size_t) l], st->row, n, st->sv[(size_t) l]->nb[1], 0);
                ggml_tensor * kd = ggml_view_2d(ctx, st->k[(size_t) l], st->row, n, st->k[(size_t) l]->nb[1],
                                                (size_t) i0 * st->k[(size_t) l]->nb[1]);
                ggml_tensor * vd = ggml_view_2d(ctx, st->v[(size_t) l], st->row, n, st->v[(size_t) l]->nb[1],
                                                (size_t) i0 * st->v[(size_t) l]->nb[1]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ks, kd));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, vs, vd));
            }
            ggml_backend_sched_reset(sched);
            const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            ggml_free(ctx);
            if (!ok) {
                *err = "KV-prefix store write failed";
                return false;
            }
        }

        st->n = (int) (i0 + n);
    }
    return true;
}
