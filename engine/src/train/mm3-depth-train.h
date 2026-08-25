#pragma once
// train/mm3-depth-train.h — the acoustic loss: teacher-forced CE through the
// FROZEN depth decoder, gradients into the LM adapter via last_hidden.
//
// WHY THIS EXISTS
//
// At generation, the depth decoder produces every acoustic codebook — the
// timbre — conditioned on the LM's last_hidden_state per frame
// (mm3-ar-loop.h: `depth_decode(last_hidden, sampled)`). Training an LM
// adapter on semantic CE alone leaves that hidden state completely
// unconstrained: the adapter is free to drift it anywhere that helps semantic
// prediction, and the frozen depth decoder then decodes states it was never
// trained on. The audible result is formant-shifted vocals — "chipmunk" on
// one run, "goblin" on another, direction unconstrained — while the adapter
// itself measures perfectly healthy. Diagnosed 2026-08-25 on the ADTR and
// Fightstar runs; the ear-validated "MLP 0.5" render dial was this fault
// being managed empirically.
//
// THE FIX
//
// Supervise the acoustic books too. Ground truth is already in every
// `.codes` file (books 1..7), and the depth decoder is already on the
// trainer's command line. Per sampled frame, one teacher-forced causal pass
// replaces inference's seven sequential graphs:
//
//   seq = proj([h_frame, embed_sem(gt), audio_embd(gt ac1..6)]) + pos[0..7]
//   4 x block (16 heads x 256, causal, NO RoPE)  ->  output_norm
//   position c (1..7) -> head_{c-1} -> CE vs gt book c
//
// The depth weights are FROZEN — only h_frame is a param, and its gradient
// is added into the same t_G buffer the semantic head fills, at the same
// column (frame j lives at column n_masked-1+j in both t_H and t_G). The
// existing segment backward then carries it into the adapter unchanged.
//
// Frames are SUBSAMPLED per micro-step (default 128 of the crop): the depth
// stack is small but 7 heads x every frame would rival the main backward's
// cost. Selection is stateless — derived from (seed, opt step) — so a
// paused-and-resumed run replays the identical selection.
//
// VALIDATED, NOT TRUSTED: mm3_depth_train_fdcheck() compares the autodiff
// gradient at h against central finite differences. The lesson of the runtime
// LoKR audit is that "the math reads right" is not evidence; --fd-check runs
// this probe before training starts.

#include "train/lm-ckpt.h"
#include "qwen3-enc.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// ── weights ─────────────────────────────────────────────────────────────────

struct MM3DepthTrainBlock {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq        = nullptr;
    ggml_tensor * wk        = nullptr;
    ggml_tensor * wv        = nullptr;
    ggml_tensor * wo        = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * gate      = nullptr;
    ggml_tensor * up        = nullptr;
    ggml_tensor * down      = nullptr;
};

struct MM3DepthTrain {
    // Frozen decoder weights, file dtype, own buffer on the training backend.
    ggml_context *        wctx    = nullptr;
    ggml_backend_buffer_t wbuf    = nullptr;
    ggml_tensor *         proj    = nullptr;   // [4096, 4096]
    ggml_tensor *         pos     = nullptr;   // [4096, 16] learned absolute
    ggml_tensor *         onorm   = nullptr;   // [4096]
    ggml_tensor *         heads[7] = {};       // [4096, 1024] each
    MM3DepthTrainBlock    blk[4];

    // The LM's tables, borrowed (semantic embeds come from the LM by contract
    // — mm3-depth-graph.h convention 1; acoustic from the flat 7168-row table
    // the trainer already loads for its own inputs).
    ggml_tensor * lm_embed    = nullptr;
    ggml_tensor * audio_embd  = nullptr;
    int64_t       sem_offset  = 0;       // +151675
    int64_t       audio_vocab = 1024;

    // Config.
    int    n_heads   = 16;
    int    head_dim  = 256;
    int    K         = 128;    // frames sampled per micro-step
    float  lambda    = 1.0f;   // loss weight; 0 disables the term entirely
    uint64_t seed    = 42;

    ggml_backend_t backend = nullptr;

    // Scratch reused across steps.
    std::vector<uint8_t> arena;

    // Set per micro-step by the trainer before lm_ckpt_micro_step runs: the
    // crop's ground-truth codes (8 int32 per frame, book 0 semantic) and its
    // geometry. `codes` points at the SAMPLE's storage — valid for the step.
    const int32_t * codes    = nullptr;  // [n_frames_total * 8], crop at c0
    int64_t         c0       = 0;
    int64_t         n_sup    = 0;        // supervised positions incl. EOS
    bool            at_end   = false;    // last supervised position is EOS
    int64_t         opt_step = 0;

    double last_loss = 0.0;              // mean CE of the term, for the log
};

// ── loading ─────────────────────────────────────────────────────────────────

// Reads the full decoder out of the depth GGUF. Kept separate from the
// trainer's existing audio_embd load on purpose: that one predates this file
// and other code paths depend on its exact behaviour.
static bool mm3_depth_train_load(MM3DepthTrain * d, const char * depth_path,
                                 ggml_backend_t backend, std::string * err) {
    gguf_init_params ip = { /*no_alloc=*/false, /*ctx=*/nullptr };
    ggml_context *   meta = nullptr;
    ip.ctx                = &meta;
    gguf_context * gf     = gguf_init_from_file(depth_path, ip);
    if (!gf) {
        *err = std::string("cannot open ") + depth_path;
        return false;
    }

    // Count the tensors we keep so the dest context is sized exactly.
    const int keep = 3 + 7 + 4 * 9;
    ggml_init_params wip = {
        /*mem_size  =*/(size_t) (keep + 2) * ggml_tensor_overhead(),
        /*mem_buffer=*/nullptr,
        /*no_alloc  =*/true,
    };
    d->wctx = ggml_init(wip);

    auto want = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * src = ggml_get_tensor(meta, name);
        if (!src) {
            return nullptr;
        }
        ggml_tensor * t = ggml_dup_tensor(d->wctx, src);
        ggml_set_name(t, name);
        return t;
    };

    char nm[128];
    d->proj  = want("depth.proj.weight");
    d->pos   = want("depth.pos_embd.weight");
    d->onorm = want("depth.output_norm.weight");
    bool ok  = d->proj && d->pos && d->onorm;
    for (int h = 0; h < 7 && ok; h++) {
        snprintf(nm, sizeof(nm), "depth.head.%d.weight", h);
        ok = (d->heads[h] = want(nm)) != nullptr;
    }
    for (int b = 0; b < 4 && ok; b++) {
        MM3DepthTrainBlock & B = d->blk[b];
        const char * sfx[9] = { "attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                                "ffn_norm",  "ffn_gate", "ffn_up", "ffn_down" };
        ggml_tensor ** dst[9] = { &B.attn_norm, &B.wq, &B.wk, &B.wv, &B.wo,
                                  &B.ffn_norm, &B.gate, &B.up, &B.down };
        for (int i = 0; i < 9 && ok; i++) {
            snprintf(nm, sizeof(nm), "depth.blk.%d.%s.weight", b, sfx[i]);
            ok = (*dst[i] = want(nm)) != nullptr;
        }
    }
    if (!ok) {
        *err = std::string(depth_path) + " is missing depth decoder tensors — "
               "the acoustic loss needs the full decoder, not just audio_embd";
        gguf_free(gf);
        ggml_free(meta);
        return false;
    }

    d->wbuf = ggml_backend_alloc_ctx_tensors(d->wctx, backend);
    if (!d->wbuf) {
        *err = "depth decoder VRAM allocation failed (~1.2 GB at f16)";
        gguf_free(gf);
        ggml_free(meta);
        return false;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(d->wctx); t; t = ggml_get_next_tensor(d->wctx, t)) {
        ggml_tensor * src = ggml_get_tensor(meta, t->name);
        ggml_backend_tensor_set(t, src->data, 0, ggml_nbytes(src));
    }
    gguf_free(gf);
    ggml_free(meta);

    d->backend = backend;
    d->arena.resize(64u * 1024 * 1024);
    return true;
}

static void mm3_depth_train_free(MM3DepthTrain * d) {
    if (d->wbuf) {
        ggml_backend_buffer_free(d->wbuf);
    }
    if (d->wctx) {
        ggml_free(d->wctx);
    }
    *d = MM3DepthTrain{};
}

// ── the graph ───────────────────────────────────────────────────────────────

// Stateless per-(seed, step) frame selection: resume replays it exactly.
static void mm3_depth_pick_frames(const MM3DepthTrain & d, std::vector<int64_t> * out) {
    out->clear();
    // Frames with a full acoustic ground truth: supervised position j predicts
    // frame c0+j, and the EOS position (when present) has no acoustic books.
    const int64_t usable = d.at_end ? d.n_sup - 1 : d.n_sup;
    if (usable <= 0) {
        return;
    }
    const int64_t K = std::min<int64_t>(d.K, usable);
    // splitmix64 stream keyed by (seed, step) — no state to carry.
    uint64_t z = d.seed * 0x9E3779B97F4A7C15ull + (uint64_t) d.opt_step;
    auto nxt = [&z]() {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    // Partial Fisher–Yates over an index vector: exact, unbiased, cheap at
    // crop sizes (a few thousand entries).
    std::vector<int64_t> idx((size_t) usable);
    for (int64_t i = 0; i < usable; i++) {
        idx[(size_t) i] = i;
    }
    for (int64_t i = 0; i < K; i++) {
        const int64_t j = i + (int64_t) (nxt() % (uint64_t) (usable - i));
        std::swap(idx[(size_t) i], idx[(size_t) j]);
        out->push_back(idx[(size_t) i]);
    }
}

// Runs the teacher-forced pass and adds lambda-scaled dL/dh into t_G.
// Shaped as the lm-ckpt aux-head hook: called after the semantic head, before
// the backward segments.
static bool mm3_depth_train_head(LmCkptRun & r, const LmSample & s, void * user) {
    MM3DepthTrain & d = *(MM3DepthTrain *) user;
    if (d.lambda <= 0.0f || !d.codes) {
        return true;
    }
    std::vector<int64_t> frames;
    mm3_depth_pick_frames(d, &frames);
    if (frames.empty()) {
        return true;
    }
    const int64_t K  = (int64_t) frames.size();
    const int64_t H  = d.proj->ne[0];
    const int64_t AV = d.audio_vocab;
    LmCkptState & st = *r.st;

    // Host-side gathers: h columns out of t_H, ids and one-hot labels from the
    // ground-truth codes. Column j of the supervised span lives at full-
    // sequence column n_masked-1+j — the same offset the semantic head uses.
    std::vector<float>   h_host((size_t) (H * K));
    std::vector<int32_t> sem_ids((size_t) K);
    std::vector<int32_t> ac_ids((size_t) (6 * K));   // feedback books 1..6
    std::vector<float>   labels((size_t) (7 * AV * K), 0.0f);
    for (int64_t i = 0; i < K; i++) {
        const int64_t j   = frames[(size_t) i];
        const size_t  col = (size_t) (s.n_masked - 1 + j);
        ggml_backend_tensor_get(st.t_H, h_host.data() + (size_t) (i * H),
                                col * st.t_H->nb[1], (size_t) H * sizeof(float));
        const int32_t * f = d.codes + (size_t) ((d.c0 + j) * 8);
        sem_ids[(size_t) i] = f[0] + (int32_t) d.sem_offset;
        for (int64_t b = 0; b < 6; b++) {
            // Feedback embedding for book b+1 reads the flat table at
            // code + b*audio_vocab — mm3-depth-graph.h convention 2.
            ac_ids[(size_t) (b * K + i)] = f[1 + b] + (int32_t) (b * AV);
        }
        for (int64_t c = 0; c < 7; c++) {
            labels[(size_t) (c * AV * K + i * AV + f[1 + c])] = 1.0f;
        }
    }

    ggml_init_params ip  = { d.arena.size(), d.arena.data(), true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 2048, /*grads=*/true);

    // Inputs.
    ggml_tensor * t_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, K);
    ggml_set_name(t_h, "depth.h_in");
    // Param + input, exactly like lm-ckpt's checkpoint tensors: param so the
    // backward targets it, input so the scheduler allocates it as a leaf and
    // the accumulator pattern below can hand it a persistent grad buffer.
    ggml_set_param(t_h);
    ggml_set_input(t_h);
    ggml_tensor * t_gacc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, K);
    ggml_set_name(t_gacc, "depth.h_grad");
    ggml_set_input(t_gacc);
    ggml_set_output(t_gacc);
    ggml_tensor * t_one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t_one, "depth.one");
    ggml_set_input(t_one);
    ggml_tensor * t_sem = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, K);
    ggml_set_name(t_sem, "depth.sem_ids");
    ggml_set_input(t_sem);
    ggml_tensor * t_ac = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6 * K);
    ggml_set_name(t_ac, "depth.ac_ids");
    ggml_set_input(t_ac);
    ggml_tensor * t_msk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
    ggml_set_name(t_msk, "depth.mask");
    ggml_set_input(t_msk);
    ggml_tensor * t_lab[7];
    for (int c = 0; c < 7; c++) {
        t_lab[c] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, AV, K);
        ggml_format_name(t_lab[c], "depth.lab.%d", c);
        ggml_set_input(t_lab[c]);
    }

    // Sequence assembly: [H, 8, K], token 0 = h, 1 = semantic embed, 2..7 =
    // acoustic feedback embeds. ggml_concat is avoided on the param path —
    // acc's backward is a plain view, proven machinery.
    ggml_tensor * sem_e = ggml_get_rows(ctx, d.lm_embed, t_sem);            // [H, K]
    ggml_tensor * ac_e  = ggml_get_rows(ctx, d.audio_embd, t_ac);          // [H, 6K]

    // Sequence assembly: every slot is repeat-to-[H,8,K] * one-hot-column
    // mask, summed. Deliberately NO ggml_acc and NO views on this path: the
    // acc-chain variant mis-executed ORDER-DEPENDENTLY (the loss changed when
    // debug probes changed node order — the gallocr in-place aliasing trap the
    // Lever A notes warn about), and repeat/mul/add are immune to it. The 24
    // extra nodes are noise at S=8.
    ggml_tensor * base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 8, K);
    ggml_set_name(base, "depth.seq0");
    ggml_set_input(base);                       // uploaded as zeros; shape ref + add identity
    ggml_tensor * sel[8];
    for (int t = 0; t < 8; t++) {
        sel[t] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, 8, 1);
        ggml_format_name(sel[t], "depth.sel.%d", t);
        ggml_set_input(sel[t]);
    }
    ggml_tensor * h3  = ggml_reshape_3d(ctx, t_h, H, 1, K);
    ggml_tensor * seq = ggml_add(ctx, base, ggml_mul(ctx, ggml_repeat(ctx, h3, base), sel[0]));
    seq = ggml_add(ctx, seq,
                   ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, sem_e, H, 1, K), base), sel[1]));
    for (int64_t b = 0; b < 6; b++) {
        ggml_tensor * eb = ggml_cont(ctx, ggml_view_2d(ctx, ac_e, H, K, ac_e->nb[1],
                                                       (size_t) (b * K) * ac_e->nb[1]));
        seq = ggml_add(ctx, seq,
                       ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_3d(ctx, eb, H, 1, K), base),
                                sel[(size_t) (2 + b)]));
    }
    // proj + learned absolute positions (rows 0..7 of a 16-row table).
    ggml_tensor * x = ggml_mul_mat(ctx, qwen3_f32(ctx, d.proj), seq);       // [H, 8, K]
    ggml_tensor * pos8 = ggml_view_2d(ctx, d.pos, H, 8, d.pos->nb[1], 0);
    x = ggml_add(ctx, x, qwen3_f32(ctx, ggml_cont(ctx, pos8)));

    const int   nh = d.n_heads;
    const int   hd = d.head_dim;
    const float qs = 1.0f / sqrtf((float) hd);
    for (int b = 0; b < 4; b++) {
        const MM3DepthTrainBlock & B = d.blk[b];
        ggml_tensor * a = lm_rms(ctx, x, B.attn_norm, 1e-6f);
        ggml_tensor * q = ggml_mul_mat(ctx, qwen3_f32(ctx, B.wq), a);
        ggml_tensor * k = ggml_mul_mat(ctx, qwen3_f32(ctx, B.wk), a);
        ggml_tensor * v = ggml_mul_mat(ctx, qwen3_f32(ctx, B.wv), a);
        q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, nh, 8, K), 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, nh, 8, K), 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, nh, 8, K), 1, 2, 0, 3));
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                          // [8, 8, nh, K]
        kq = ggml_scale(ctx, kq, qs);
        kq = ggml_add(ctx, kq, t_msk);                                       // causal, broadcast
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor * o = ggml_mul_mat(ctx, v, kq);                          // [hd, 8, nh, K]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                // [hd, nh, 8, K]
        o = ggml_reshape_3d(ctx, o, (int64_t) hd * nh, 8, K);
        o = ggml_mul_mat(ctx, qwen3_f32(ctx, B.wo), o);
        x = ggml_add(ctx, x, o);
        ggml_tensor * m = lm_rms(ctx, x, B.ffn_norm, 1e-6f);
        ggml_tensor * g = ggml_silu(ctx, ggml_mul_mat(ctx, qwen3_f32(ctx, B.gate), m));
        ggml_tensor * u = ggml_mul_mat(ctx, qwen3_f32(ctx, B.up), m);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, qwen3_f32(ctx, B.down), ggml_mul(ctx, g, u)));
    }
    ggml_tensor * y = lm_rms(ctx, x, d.onorm, 1e-6f);                        // [H, 8, K]

    // Heads: position c (1..7) through head c-1, CE against book c. The
    // combined loss is the MEAN over the 7*K predictions, scaled by
    // lambda/grad_accum to sit beside the semantic term.
    ggml_tensor * L = nullptr;
    for (int c = 1; c <= 7; c++) {
        ggml_tensor * hc = ggml_cont(ctx, ggml_view_3d(ctx, y, H, 1, K,
                                     y->nb[1], y->nb[2], (size_t) c * y->nb[1]));
        hc = ggml_reshape_2d(ctx, hc, H, K);
        ggml_tensor * lg = ggml_mul_mat(ctx, qwen3_f32(ctx, d.heads[c - 1]), hc);  // [1024, K]
        ggml_tensor * lc = ggml_cross_entropy_loss(ctx, lg, t_lab[c - 1]);         // mean over K
        L = L ? ggml_add(ctx, L, lc) : lc;
    }
    L = ggml_scale(ctx, L, d.lambda / (7.0f * (float) std::max(1, r.grad_accum)));
    ggml_set_loss(L);
    ggml_build_forward_expand(gf, L);
    // The accumulator array is NOT optional in this tree: the patched backward
    // takes its dL/dL seed from the LOSS node's accumulator (t_one). Passing
    // nullptr leaves the seed unset and every gradient computes as exactly
    // zero — measured, and the reason the first three tripwire attempts died.
    // Same convention as lm_ckpt_fill_gacc: indexed by forward node position.
    {
        std::vector<ggml_tensor *> gacc((size_t) ggml_graph_n_nodes(gf), nullptr);
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * nd = ggml_graph_node(gf, i);
            if (nd == t_h) {
                gacc[(size_t) i] = t_gacc;
            } else if (nd->flags & GGML_TENSOR_FLAG_LOSS) {
                gacc[(size_t) i] = t_one;
            }
        }
        ggml_build_backward_expand(ctx, gf, gacc.data());
    }

    ggml_backend_sched_reset(r.sched);
    if (!ggml_backend_sched_alloc_graph(r.sched, gf)) {
        ggml_free(ctx);
        return false;
    }
    // Uploads. The accumulator must START at zero — it is accumulated into,
    // not assigned (the lm-ckpt P6 comment makes the same point about Gh).
    {
        const float one = 1.0f;
        ggml_backend_tensor_set(t_one, &one, 0, sizeof(float));
        std::vector<float> gz((size_t) (H * K), 0.0f);
        ggml_backend_tensor_set(t_gacc, gz.data(), 0, gz.size() * sizeof(float));
    }
    ggml_backend_tensor_set(t_h, h_host.data(), 0, h_host.size() * sizeof(float));
    ggml_backend_tensor_set(t_sem, sem_ids.data(), 0, sem_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_ac, ac_ids.data(), 0, ac_ids.size() * sizeof(int32_t));
    {
        float mk[64];
        for (int r2 = 0; r2 < 8; r2++) {
            for (int c2 = 0; c2 < 8; c2++) {
                mk[r2 * 8 + c2] = c2 <= r2 ? 0.0f : -INFINITY;
            }
        }
        // ggml [8,8]: ne0 indexes the K-position, ne1 the Q-position; row-major
        // upload therefore writes Q-rows, matching soft_max's mask contract.
        ggml_backend_tensor_set(t_msk, mk, 0, sizeof(mk));
    }
    {
        std::vector<float> zeros((size_t) (H * 8 * K), 0.0f);
        ggml_backend_tensor_set(base, zeros.data(), 0, zeros.size() * sizeof(float));
        std::vector<float> selh((size_t) (H * 8));
        for (int t = 0; t < 8; t++) {
            std::fill(selh.begin(), selh.end(), 0.0f);
            std::fill(selh.begin() + (size_t) (t * H), selh.begin() + (size_t) ((t + 1) * H), 1.0f);
            ggml_backend_tensor_set(sel[t], selh.data(), 0, selh.size() * sizeof(float));
        }
    }
    for (int c = 0; c < 7; c++) {
        ggml_backend_tensor_set(t_lab[c], labels.data() + (size_t) (c * AV * K),
                                0, (size_t) (AV * K) * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(r.sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_reset(r.sched);
        ggml_free(ctx);
        return false;
    }

    // Record the loss (undo the lambda/GA scale so the log reads plain CE).
    float lval = 0.0f;
    ggml_backend_tensor_get(L, &lval, 0, sizeof(float));

    d.last_loss = d.lambda > 0.0f
                    ? (double) lval * (double) (std::max(1, r.grad_accum)) / (double) d.lambda
                    : 0.0;

    // Scatter-add dL/dh into t_G at the same columns the h values came from.
    std::vector<float> ghost((size_t) (H * K));
    ggml_backend_tensor_get(t_gacc, ghost.data(), 0, ghost.size() * sizeof(float));
    ggml_tensor * t_G = st.Gh[0];
    std::vector<float> col((size_t) H);
    for (int64_t i = 0; i < K; i++) {
        const int64_t j   = frames[(size_t) i];
        const size_t  off = (size_t) (s.n_masked - 1 + j) * t_G->nb[1];
        ggml_backend_tensor_get(t_G, col.data(), off, (size_t) H * sizeof(float));
        const float * g = ghost.data() + (size_t) (i * H);
        for (int64_t e = 0; e < H; e++) {
            col[(size_t) e] += g[e];
        }
        ggml_backend_tensor_set(t_G, col.data(), off, (size_t) H * sizeof(float));
    }
    ggml_backend_sched_reset(r.sched);
    ggml_free(ctx);
    return true;
}

// ── validation ──────────────────────────────────────────────────────────────

// The gradient tripwire: a DIRECTIONAL finite difference along the analytic
// gradient. A per-element probe cannot work here — with the loss averaged
// over 7*K predictions, per-element dL/dh is ~1e-5, so an eps-wiggle moves
// the loss by less than float32's ulp at CE~7 and the difference reads as
// exactly zero (measured). Perturbing the whole column along g/|g| makes the
// expected response eps*|g|, chosen large enough to resolve, and the check
// becomes fd ~= |g| — one number against one number.
static bool mm3_depth_train_fdcheck(MM3DepthTrain & d, LmCkptRun & r, const LmSample & s,
                                    std::string * err) {
    LmCkptState & st     = *r.st;
    const int64_t H      = d.proj->ne[0];
    const int     save_K = d.K;
    d.K = 2;

    auto run_loss = [&]() -> double {
        ggml_backend_buffer_clear(st.buf_gh0, 0);
        if (!mm3_depth_train_head(r, s, &d)) {
            return NAN;
        }
        return d.last_loss;
    };

    std::vector<int64_t> frames;
    mm3_depth_pick_frames(d, &frames);
    if (frames.empty()) {
        d.K = save_K;
        *err = "fd-check: no usable frames";
        return false;
    }

    // Analytic gradient of the graph loss at the first probed column.
    ggml_backend_buffer_clear(st.buf_gh0, 0);
    if (!mm3_depth_train_head(r, s, &d)) {
        d.K = save_K;
        *err = "fd-check: hook failed";
        return false;
    }
    const int64_t j0   = frames[0];
    const size_t  gcol = (size_t) (s.n_masked - 1 + j0) * st.Gh[0]->nb[1];
    const size_t  hcol = (size_t) (s.n_masked - 1 + j0) * st.t_H->nb[1];
    std::vector<float> g((size_t) H), h0((size_t) H), hp((size_t) H);
    ggml_backend_tensor_get(st.Gh[0], g.data(), gcol, (size_t) H * sizeof(float));
    ggml_backend_tensor_get(st.t_H, h0.data(), hcol, (size_t) H * sizeof(float));
    double gnorm = 0.0;
    for (int64_t e = 0; e < H; e++) {
        gnorm += (double) g[(size_t) e] * (double) g[(size_t) e];
    }
    gnorm = sqrt(gnorm);
    const float lam_ga = d.lambda / (float) std::max(1, r.grad_accum);
    if (!(gnorm > 0.0)) {
        d.K = save_K;
        *err = "fd-check: analytic gradient is exactly zero — the backward is not reaching h";
        return false;
    }

    // eps sized so the GRAPH loss moves ~1e-2 — far above f32 readback ulp.
    const double eps = std::min(10.0, std::max(1e-2, 1e-2 / gnorm));
    auto step_to = [&](double sgn) {
        for (int64_t e = 0; e < H; e++) {
            hp[(size_t) e] = h0[(size_t) e] + (float) (sgn * eps * (double) g[(size_t) e] / gnorm);
        }
        ggml_backend_tensor_set(st.t_H, hp.data(), hcol, (size_t) H * sizeof(float));
    };
    step_to(+1.0);
    const double lp = run_loss();
    step_to(-1.0);
    const double lm = run_loss();
    ggml_backend_tensor_set(st.t_H, h0.data(), hcol, (size_t) H * sizeof(float));
    ggml_backend_buffer_clear(st.buf_gh0, 0);
    d.K = save_K;

    // last_loss is plain CE; the analytic gradient carries lambda/GA.
    const double fd  = (lp - lm) / (2.0 * eps) * (double) lam_ga;
    const double rel = fabs(fd - gnorm) / std::max(1e-12, fabs(fd) + gnorm);
    fprintf(stderr,
            "[mm3-depth-train] fd directional: analytic |g| %.6e fd %.6e rel %.3f (eps %.3g)\n",
            gnorm, fd, rel, eps);
    if (!(rel < 0.05)) {
        *err = "fd-check FAILED on the depth loss gradient — do not train with it";
        return false;
    }
    fprintf(stderr, "[mm3-depth-train] fd-check PASS\n");
    return true;
}
