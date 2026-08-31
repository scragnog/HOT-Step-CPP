#pragma once
// minimax/mm3-rec7.h — the rec7 STATE encoder: DAV latents -> LM frame hiddens.
//
// HOT-Step file (does not exist upstream).
//
// rec7 (PurpleOrc/m3-rec7-encoder) predicts, per 25 Hz frame, the 4096-d
// hidden state the MM3 language model would have produced for that frame —
// not codes. Its trunk is the SAME V4Encoder architecture as the community
// RVQ code encoders, so everything below rides mm3-rvq-encode.h's loader and
// window graph unchanged; this file adds only what a STATE encoder needs:
//
//   • the HHead (1088 -> 2048 -> gelu -> 4096) on top of the trunk features
//   • read_states windowing: 128-frame windows on hop 72, overlap-AVERAGED —
//     deliberately different from the code path's first-wins hop-128, and
//     part of the model's contract (rec7_model.py::read_states)
//   • the two 16k-row LM table slices (lm_head / embed_tokens over the
//     semantic rows) the cond builder needs, converted INTO the GGUF so the
//     8B never loads here
//   • the renderer-condition builder: h -> c0 (semantic argmax) -> the
//     released depth decoder's greedy chain (mm3-depth-graph.h, the same
//     code generation runs) -> [F, 8, 4096] frame hiddens, byte-compatible
//     with MM3ArResult::frame_hiddens and therefore with everything stage 2
//     of the pipeline consumes.
//
// Convert with engine/tools/convert-rvq-encoder.py --head --m3; the file
// carries `mm3rvq.has_state_head` and loads here, NOT via mm3_rvq_load alone
// (a state gguf handed to the code path would run but use rec7's own aux
// depth head, whose codes are the encoder's documented weak half).
//
// PRECISION: the python reference runs the trunk under bf16 autocast and the
// head in f32. This port is f32 end-to-end, so parity against a reference
// fixture is bounded by the reference's own bf16 noise — the self-test bar is
// per-frame cosine, not byte equality.

#include "mm3-rvq-encode.h"

struct MM3Rec7 {
    MM3Rvq        rvq;                 // trunk + windows (loader/graph reused)
    bool          loaded = false;
    uint32_t      state_dim = 0, states_hop = 0, sem_offset = 0;
    ggml_tensor * hh0_w = nullptr, * hh0_b = nullptr;   // Linear 1088 -> 2048
    ggml_tensor * hh2_w = nullptr, * hh2_b = nullptr;   // Linear 2048 -> 4096
    ggml_tensor * lm_head = nullptr;                    // [4096, 16384] sem lm_head rows
    ggml_tensor * lm_embd = nullptr;                    // [4096, 16384] sem embed rows
    WeightCtx     xctx = {};           // the extra tensors' own buffer
    size_t        vram = 0;

    // cached head graph: feats [D, F] -> states [4096, F]; and the c0 graph:
    // states chunk [4096, N] -> argmax input logits [16384, N]
    ggml_context * h_ctx = nullptr;
    ggml_cgraph *  h_graph = nullptr;
    ggml_gallocr_t h_alloc = nullptr;
    ggml_tensor *  h_in = nullptr, * h_out = nullptr, * h_logits = nullptr;
};

static void mm3_rec7_free(MM3Rec7 * r) {
    if (r->h_alloc) { ggml_gallocr_free(r->h_alloc); r->h_alloc = nullptr; }
    if (r->h_ctx)   { ggml_free(r->h_ctx); r->h_ctx = nullptr; }
    r->h_graph = nullptr; r->h_in = nullptr; r->h_out = nullptr; r->h_logits = nullptr;
    if (r->xctx.buffer) { ggml_backend_buffer_free(r->xctx.buffer); r->xctx.buffer = nullptr; }
    if (r->xctx.ctx)    { ggml_free(r->xctx.ctx); r->xctx.ctx = nullptr; }
    r->xctx = {};
    mm3_rvq_free(&r->rvq);
    r->loaded = false;
    r->vram = 0;
}

static bool mm3_rec7_load(MM3Rec7 * r, const char * path, std::string * err) {
    if (r->loaded) {
        return true;
    }
    if (!mm3_rvq_load(&r->rvq, path, err)) {
        return false;
    }
    // Second pass over the same file for the state-encoder extras. Wasteful by
    // one header parse, and worth it: mm3_rvq_load stays byte-identical for the
    // code path, which is the "gate off = today exactly" guarantee.
    GGUFModel gf = {};
    if (!gf_load(&gf, path)) {
        mm3_rec7_free(r);
        if (err) *err = std::string("cannot reopen ") + path;
        return false;
    }
    if (!gf_get_bool(gf, "mm3rvq.has_state_head")) {
        gf_close(&gf);
        mm3_rec7_free(r);
        if (err) *err = std::string(path) + " is a CODE encoder gguf (no state head); "
                        "convert rec7 with --head/--m3";
        return false;
    }
    r->state_dim  = gf_get_u32(gf, "mm3rvq.state_dim");
    r->states_hop = gf_get_u32(gf, "mm3rvq.states_hop");
    r->sem_offset = gf_get_u32(gf, "mm3rvq.sem_offset");
    const int64_t D = (int64_t) r->rvq.cfg.d_model, H = (int64_t) r->state_dim;
    const int64_t V = (int64_t) r->rvq.cfg.sem_vocab;
    if (!H || !r->states_hop) {
        gf_close(&gf);
        mm3_rec7_free(r);
        if (err) *err = "rec7 gguf is missing state-head metadata";
        return false;
    }

    std::vector<std::string> errs;
    wctx_init(&r->xctx, 8);
    MM3Loader ld{ &r->xctx, &gf, nullptr, &errs };
    r->hh0_w   = ld.req("hhead.net.0.weight", D, 2048);
    r->hh0_b   = ld.req("hhead.net.0.bias", 2048);
    r->hh2_w   = ld.req("hhead.net.2.weight", 2048, H);
    r->hh2_b   = ld.req("hhead.net.2.bias", H);
    r->lm_head = ld.req("lm.sem_head", H, V);
    r->lm_embd = ld.req("lm.sem_embd", H, V);
    if (!errs.empty()) {
        gf_close(&gf);
        mm3_rec7_free(r);
        if (err) *err = errs[0];
        return false;
    }
    for (const auto & pc : r->xctx.pending) {
        r->vram += pc.nbytes;
    }
    if (!wctx_alloc(&r->xctx, r->rvq.backend)) {
        gf_close(&gf);
        mm3_rec7_free(r);
        if (err) *err = "backend buffer allocation failed for the rec7 state head";
        return false;
    }
    gf_close(&gf);
    r->loaded = true;
    fprintf(stderr, "[MM3Rec7] state head + LM tables loaded (%.1f MB extra, hop %u)\n",
            (double) r->vram / (1024.0 * 1024.0), r->states_hop);
    return true;
}

// feats [D, F] -> states [H, F] and semantic logits [V, F], one cached graph.
static bool mm3_rec7_ensure_head_graph(MM3Rec7 * r, std::string * err) {
    if (r->h_graph) {
        return true;
    }
    const int64_t D = (int64_t) r->rvq.cfg.d_model, F = (int64_t) r->rvq.cfg.frames;
    const size_t     meta = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false);
    ggml_init_params ip   = { meta, nullptr, true };
    r->h_ctx = ggml_init(ip);
    if (!r->h_ctx) {
        if (err) *err = "MM3Rec7: head ggml_init failed";
        return false;
    }
    ggml_context * ctx = r->h_ctx;
    r->h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, F);
    ggml_set_input(r->h_in);
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, r->hh0_w, r->h_in), r->hh0_b);
    x = ggml_gelu_erf(ctx, x);
    r->h_out = ggml_add(ctx, ggml_mul_mat(ctx, r->hh2_w, x), r->hh2_b);       // [H, F]
    ggml_set_output(r->h_out);
    // c0 logits ride the same graph: they are only ever wanted per frame that
    // already has a state, and the matmul shares the states' lifetime.
    r->h_logits = ggml_mul_mat(ctx, r->lm_head, r->h_out);                     // [V, F]
    ggml_set_output(r->h_logits);
    r->h_graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(r->h_graph, r->h_out);
    ggml_build_forward_expand(r->h_graph, r->h_logits);
    r->h_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r->rvq.backend));
    if (!r->h_alloc || !ggml_gallocr_reserve(r->h_alloc, r->h_graph)) {
        if (err) *err = "MM3Rec7: head graph allocation failed";
        return false;
    }
    return true;
}

// One window's states (+ optional c0 logits argmax). feats [F][D] host-major
// exactly as mm3_rvq_encode_window returned them.
static bool mm3_rec7_head_window(MM3Rec7 * r, const float * feats, float * states,
                                 int32_t * c0, std::string * err) {
    if (!mm3_rec7_ensure_head_graph(r, err)) {
        return false;
    }
    if (!ggml_gallocr_alloc_graph(r->h_alloc, r->h_graph)) {
        if (err) *err = "MM3Rec7: head gallocr_alloc_graph failed";
        return false;
    }
    ggml_backend_tensor_set(r->h_in, feats, 0, ggml_nbytes(r->h_in));
    if (ggml_backend_graph_compute(r->rvq.backend, r->h_graph) != GGML_STATUS_SUCCESS) {
        if (err) *err = "MM3Rec7: head graph compute failed";
        return false;
    }
    if (states) {
        ggml_backend_tensor_get(r->h_out, states, 0, ggml_nbytes(r->h_out));
    }
    if (c0) {
        const int64_t      V = (int64_t) r->rvq.cfg.sem_vocab, F = (int64_t) r->rvq.cfg.frames;
        std::vector<float> logits((size_t) (V * F));
        ggml_backend_tensor_get(r->h_logits, logits.data(), 0, (size_t) (V * F) * sizeof(float));
        for (int64_t f = 0; f < F; f++) {
            const float * row = logits.data() + (size_t) (f * V);
            int64_t       best = 0;
            float         bv   = row[0];
            for (int64_t j = 1; j < V; j++) {
                if (row[j] > bv) { bv = row[j]; best = j; }
            }
            c0[f] = (int32_t) best;
        }
    }
    return true;
}

// c0 = argmax(lm_head @ h) over [T] states, in 128-frame chunks (the last one
// padded). This is states_to_streams' semantic argmax — computed from the
// FINAL averaged states, exactly like the reference.
static bool mm3_rec7_c0_from_states(MM3Rec7 * r, const float * h, int64_t T, int32_t * out_c0,
                                    std::string * err) {
    const int64_t H = (int64_t) r->state_dim, V = (int64_t) r->rvq.cfg.sem_vocab;
    const int64_t CHUNK = 128;

    const size_t     meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false);
    ggml_init_params ip   = { meta, nullptr, true };
    ggml_context *   ctx  = ggml_init(ip);
    if (!ctx) {
        if (err) *err = "MM3Rec7: c0 ggml_init failed";
        return false;
    }
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, CHUNK);
    ggml_set_input(in);
    ggml_tensor * lg = ggml_mul_mat(ctx, r->lm_head, in);                      // [V, CHUNK]
    ggml_set_output(lg);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, lg);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(r->rvq.backend));
    bool ok = alloc && ggml_gallocr_reserve(alloc, graph);

    std::vector<float> buf((size_t) (H * CHUNK), 0.0f);
    std::vector<float> logits((size_t) (V * CHUNK));
    for (int64_t s = 0; ok && s < T; s += CHUNK) {
        const int64_t n = std::min<int64_t>(CHUNK, T - s);
        memcpy(buf.data(), h + (size_t) (s * H), (size_t) (n * H) * sizeof(float));
        if (n < CHUNK) {
            memset(buf.data() + (size_t) (n * H), 0, (size_t) ((CHUNK - n) * H) * sizeof(float));
        }
        ok = ggml_gallocr_alloc_graph(alloc, graph);
        if (!ok) break;
        ggml_backend_tensor_set(in, buf.data(), 0, ggml_nbytes(in));
        ok = ggml_backend_graph_compute(r->rvq.backend, graph) == GGML_STATUS_SUCCESS;
        if (!ok) break;
        ggml_backend_tensor_get(lg, logits.data(), 0, (size_t) (V * CHUNK) * sizeof(float));
        for (int64_t i = 0; i < n; i++) {
            const float * row = logits.data() + (size_t) (i * V);
            int64_t       best = 0;
            float         bv   = row[0];
            for (int64_t j = 1; j < V; j++) {
                if (row[j] > bv) { bv = row[j]; best = j; }
            }
            out_c0[(size_t) (s + i)] = (int32_t) best;
        }
    }
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (!ok && err && err->empty()) *err = "MM3Rec7: c0 pass failed";
    return ok;
}

/** rec7_model.py::read_states, verbatim in behaviour.
 *
 *  z [CH][L] channel-major DAV latents -> out_h [T][state_dim] states and
 *  (optionally) out_c0 [T] semantic argmax codes. Returns T via *out_T;
 *  out_h/out_c0 must hold mm3_rec7_n_frames(L) entries. Windows overlap on
 *  `states_hop` and overlapping frames are AVERAGED — first-wins here would be
 *  a silent porting bug, not a simplification. */
static int64_t mm3_rec7_n_frames(int64_t L) {
    // python: T_tot = int(L / 3.45) - 2; T = T_tot - 1. The 3.45 is rec7's own
    // constant (not 441/128 = 3.4453125) — reproduce it, do not "fix" it.
    const int64_t t_tot = (int64_t) ((double) L / 3.45) - 2;
    return t_tot - 1;
}

static bool mm3_rec7_read_states(MM3Rec7 * r, const float * z, int64_t L, float * out_h,
                                 int32_t * out_c0, int64_t * out_T, std::string * err) {
    const MM3RvqConfig & c = r->rvq.cfg;
    const int64_t F = (int64_t) c.frames, CH = (int64_t) c.latent_channels;
    const int64_t Lmax = (int64_t) c.latent_window_max, H = (int64_t) r->state_dim;
    const int64_t t_tot = (int64_t) ((double) L / 3.45) - 2;
    const int64_t T     = t_tot - 1;
    if (T < F + 2) {
        if (err) *err = "audio too short for rec7 (need > ~5.5 s)";
        return false;
    }
    const std::vector<int64_t> st = mm3_rvq_frame_starts(c, t_tot);

    // window offsets: range(0, max(1, T-F+1), hop) + a final flush to T-F.
    std::vector<int64_t> offs;
    for (int64_t o = 0; o < std::max<int64_t>(1, T - F + 1); o += (int64_t) r->states_hop) {
        offs.push_back(o);
    }
    if (offs.empty() || offs.back() != T - F) {
        offs.push_back(std::max<int64_t>(0, T - F));
    }

    std::vector<float> acc((size_t) (T * H), 0.0f);
    std::vector<float> cnt((size_t) T, 0.0f);

    std::vector<float> lat((size_t) (Lmax * CH));
    std::vector<float> pool((size_t) (F * Lmax));
    std::vector<float> feats((size_t) (F * (int64_t) c.d_model));
    std::vector<float> hw((size_t) (F * H));
    std::vector<int64_t> bounds((size_t) F + 1);

    for (const int64_t o : offs) {
        for (int64_t j = 0; j <= F; j++) {
            bounds[(size_t) j] = st[(size_t) (o + j)] - st[(size_t) o];
        }
        const int64_t n = bounds[(size_t) F];
        if (n > Lmax || st[(size_t) o] + n > L) {
            continue;                    // the reference's guard, verbatim
        }
        memset(lat.data(), 0, lat.size() * sizeof(float));
        for (int64_t ch = 0; ch < CH; ch++) {
            memcpy(lat.data() + (size_t) (ch * Lmax), z + (size_t) (ch * L + st[(size_t) o]),
                   (size_t) n * sizeof(float));
        }
        mm3_rvq_pool_matrix(c, bounds.data(), pool.data());
        if (!mm3_rvq_encode_window(&r->rvq, lat.data(), pool.data(), feats.data(), nullptr, err)) {
            return false;
        }
        if (!mm3_rec7_head_window(r, feats.data(), hw.data(), nullptr, err)) {
            return false;
        }
        const int64_t e = std::min<int64_t>(o + F, T);
        for (int64_t t = o; t < e; t++) {
            float *       dst = acc.data() + (size_t) (t * H);
            const float * src = hw.data() + (size_t) ((t - o) * H);
            for (int64_t k = 0; k < H; k++) {
                dst[k] += src[k];
            }
            cnt[(size_t) t] += 1.0f;
        }
    }

    for (int64_t t = 0; t < T; t++) {
        const float d = cnt[(size_t) t] > 0.0f ? cnt[(size_t) t] : 1.0f;
        float *     row = acc.data() + (size_t) (t * H);
        for (int64_t k = 0; k < H; k++) {
            row[k] /= d;
        }
    }
    memcpy(out_h, acc.data(), acc.size() * sizeof(float));

    if (out_c0) {
        if (!mm3_rec7_c0_from_states(r, acc.data(), T, out_c0, err)) {
            return false;
        }
    }
    if (out_T) {
        *out_T = T;
    }
    return true;
}
