#pragma once
// yue2/sheetsage-encoder.h — SheetSage2 encoder: one 300 s window in,
// softmax-weighted memory [T, 512] out.
//
// HOT-Step file (no acestep.cpp / upstream analog).
//
// AUTHORITY: docs/plans/yue2/20-sheetsage2-model-pin.md §1 (the window:
// padding rule, side, stride rounding) and §2 (the encoder: subsampling
// reused verbatim, all 24 Conformer blocks, the layer_weight-softmax-weighted
// sum, the final projection). docs/plans/yue2/22-sheetsage2-fixtures.md (G1:
// exactly which files the oracle dumped and the comparison rule). Where this
// file and those documents disagree, the documents win.
//
// ── ONE GRAPH, MEL TO MEMORY ─────────────────────────────────────────────────
//
// Rather than calling yue2-mert.h's yue2_mert_encode_chunk() (which returns
// only the LAST block's hidden state, plus whichever taps were asked for,
// each needing its own device->host copy) this file builds ONE ggml graph
// that runs doc 20 §2.5's whole formula on-device:
//
//   mixed = hidden_0 * w[0]              hidden_0 = post-subsampling embedding,
//                                         i.e. yue2_mert_sub_build()'s own
//                                         output, BEFORE any Conformer block
//   for n in 0..23:
//       hidden = block_n(hidden)
//       mixed  = mixed + hidden * w[n+1]
//   memory_pre_proj = mixed              -- 05_memory_pre_proj fixture
//   memory          = proj(mixed)        -- 06_memory fixture, the decoder's
//                                            cross-attention memory
//
// w[] = softmax(layer_weight): a FIXED 25-element model PARAMETER, not
// input-dependent — 04_layer_weight_softmax.f32 is identical on every window
// of every fixture (doc 22). It is therefore computed ONCE per loaded model,
// on the CPU, at graph-prepare time (softmax of 25 floats does not deserve a
// ggml op) and fed into the graph as 25 compile-time ggml_scale() factors.
// This is exact, not an approximation: softmax(layer_weight) never changes
// between windows, because layer_weight never changes between windows.
//
// ── REUSE, NOT A FORK ────────────────────────────────────────────────────────
//
// yue2_mert_sub_build() and yue2_mert_detail::mert_block() — mel-to-subsample
// and one Conformer block respectively — come from yue2-mert.h COMPLETELY
// UNCHANGED. This file adds no new mel/subsampling/Conformer math, only: (a)
// the weighted-sum-and-project head doc 20 §2.5 describes, and (b) the window
// pad/trim rule MERT's own 30 s chunk-and-stitch design (yue2-mert.h's
// yue2_mert_chunk_plan/yue2_mert_encode) never needed. SheetSage2 has NO
// cross-window stitching at the encoder level at all (doc 20 §1.4: each
// window's memory is used as-is by the decoder; whole-song stitching, if any,
// happens at the decoded-EVENT level, entirely outside this file).
//
// ── WINDOW PADDING (doc 20 §1.2) ─────────────────────────────────────────────
//
// Every window the encoder ever sees is EXACTLY round(window_seconds *
// sample_rate) samples wide — 7,200,000 for the pinned 300 s/24 kHz config —
// always, zero-padded on the RIGHT when the real audio is shorter, and
// additionally padded up to a multiple of `samples_per_frame` (a no-op for
// the pinned config: 7,200,000 / 960 = 7500 exactly, but implemented anyway
// per doc 20 §1.2 point 3). There is NO masking capability anywhere in this
// stack (doc 20 §2.3): padding is encoded through GRN, RoPE and attention
// exactly like real audio, on purpose — the model was trained this way (doc
// 20 §2.1's GRN note).
//
// GATED BY: engine/tools/yue2-probe.cpp --sheetsage-encoder-parity.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "sheetsage-model.h"
#include "yue2-mert.h"  // yue2_mert_mel, yue2_mert_sub_build, yue2_mert_detail::mert_block, yue2_mert_sub_out_len

#define SS2_ENC_MAX_NODES 16384

// One cached ggml graph: mel [n_mel, T_mel] in, `memory`/`mixed` (and,
// optionally, every hidden state) out. Rebuilt only when T_mel or
// want_hidden changes — for SheetSage2, T_mel is always 30000 (the mel frame
// count of a full padded 300 s window, doc 20 §1.2/§1.3), so in steady state
// this graph is built exactly ONCE per process and reused for every window
// of every song.
struct SheetSageEncoderGraph {
    ggml_backend_t       backend       = nullptr;
    ggml_backend_t       cpu_backend   = nullptr;
    bool                 backend_ref   = false;
    ggml_backend_sched_t sched         = nullptr;
    WeightCtx            prep          = {};  // grn_eps constant
    const void *         weights_token = nullptr;

    ggml_tensor * grn_eps = nullptr;  // [1,1] F32, shared by every GRN in the sub stack

    // softmax(layer_weight), cached at prepare time (see file header) —
    // recomputed only when the model's weight-buffer identity changes.
    std::vector<float> softmax_w;  // [hidden_state_count]

    ggml_context * gctx  = nullptr;
    uint8_t *      gbuf  = nullptr;
    ggml_cgraph *  graph = nullptr;

    ggml_tensor * input  = nullptr;  // [n_mel, T_mel]  F32, input
    ggml_tensor * pos    = nullptr;  // [T_sub]         I32, input, always 0..T_sub-1
    ggml_tensor * mixed  = nullptr;  // [1024, T_sub]   == 05_memory_pre_proj
    ggml_tensor * memory = nullptr;  // [512, T_sub]    == 06_memory

    // hidden[0] = post-subsampling embedding (state 0); hidden[1..24] = the
    // output of Conformer blocks 0..23 (states 1..24). Populated only when
    // ensure_graph was called with want_hidden=true (the probe's use case —
    // production inference never needs these, only `memory`).
    std::vector<ggml_tensor *> hidden;

    int64_t graph_T_mel       = 0;
    bool    graph_want_hidden = false;
};

static void sheetsage_enc_free_graph(SheetSageEncoderGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx   = nullptr;
    g->gbuf   = nullptr;
    g->graph  = nullptr;
    g->input  = nullptr;
    g->pos    = nullptr;
    g->mixed  = nullptr;
    g->memory = nullptr;
    g->hidden.clear();
    g->graph_T_mel     = 0;
    g->graph_want_hidden = false;
}

static void sheetsage_enc_graph_free(SheetSageEncoderGraph * g) {
    sheetsage_enc_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->grn_eps       = nullptr;
    g->weights_token = nullptr;
    g->softmax_w.clear();
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// Allocates the backend, the GRN epsilon constant, and computes+caches
// softmax(layer_weight) — everything that depends on the MODEL but not on a
// particular window. Re-runs only when the model's weight buffer identity
// (m.wctx.buffer, via m.mert.wctx.buffer) changes.
static bool sheetsage_enc_prepare(const SheetSageModel & m, SheetSageEncoderGraph * g, std::string * err) {
    const void * token = m.mert.wctx.buffer;
    if (g->sched && g->weights_token == token) {
        return true;
    }
    sheetsage_enc_graph_free(g);

    BackendPair bp = backend_init("SheetSage2-enc");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    wctx_init(&g->prep, 2);
    g->grn_eps = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(g->grn_eps, "sheetsage.enc.grn_eps");
    auto eps = std::make_unique<float[]>(1);
    eps[0]   = m.mert.cfg.sub_grn_eps;
    g->prep.pending.push_back({ g->grn_eps, eps.get(), sizeof(float), 0 });
    g->prep.staging.push_back(std::move(eps));
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the SheetSage2 encoder constants";
        }
        sheetsage_enc_graph_free(g);
        return false;
    }

    // softmax(layer_weight), doc 20 §2.5: computed in double, cast to float —
    // more accurate than the reference's own float32 softmax, not merely
    // matching it (same precision policy yue2-mert.h's mel frontend uses).
    const int64_t NH = (int64_t) m.cfg.hidden_state_count;
    if (NH <= 0 || !m.head.layer_w) {
        if (err) {
            *err = "SheetSage2 encoder head (layer_weight) is not loaded";
        }
        sheetsage_enc_graph_free(g);
        return false;
    }
    std::vector<float> raw((size_t) NH);
    ggml_backend_tensor_get(m.head.layer_w, raw.data(), 0, raw.size() * sizeof(float));
    double mx = -1e300;
    for (float v : raw) {
        mx = std::max(mx, (double) v);
    }
    std::vector<double> e((size_t) NH);
    double               sum = 0.0;
    for (int64_t i = 0; i < NH; i++) {
        e[(size_t) i] = std::exp((double) raw[(size_t) i] - mx);
        sum += e[(size_t) i];
    }
    g->softmax_w.resize((size_t) NH);
    for (int64_t i = 0; i < NH; i++) {
        g->softmax_w[(size_t) i] = (float) (e[(size_t) i] / sum);
    }

    g->sched         = backend_sched_new(bp, SS2_ENC_MAX_NODES * 2);
    g->weights_token = token;
    return true;
}

// Builds (or reuses) the graph for a given T_mel. `want_hidden` additionally
// taps every one of the hidden_state_count states as a live graph output —
// only the probe needs this; production inference should leave it false so
// the scheduler is free to reuse block-output buffers instead of keeping all
// 25 of them (each ~30 MB at T_sub=7500) alive as named outputs.
static bool sheetsage_enc_ensure_graph(const SheetSageModel & m, SheetSageEncoderGraph * g, int64_t T_mel,
                                       bool want_hidden, std::string * err) {
    if (g->gctx && g->graph_T_mel == T_mel && g->graph_want_hidden == want_hidden) {
        return true;
    }
    sheetsage_enc_free_graph(g);

    if (!m.mert.blocks_loaded) {
        if (err) {
            *err = "the SheetSage2 encoder blocks are not loaded — call sheetsage_model_load() first";
        }
        return false;
    }
    const int64_t T_sub = yue2_mert_sub_out_len(m.mert.cfg, T_mel);
    if (T_sub <= 0) {
        if (err) {
            *err = "T_mel=" + std::to_string(T_mel) + " is too short for the subsampling stack";
        }
        return false;
    }

    const size_t ctx_bytes = ggml_tensor_overhead() * (SS2_ENC_MAX_NODES + 256) +
                             ggml_graph_overhead_custom(SS2_ENC_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the SheetSage2 encoder graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the SheetSage2 encoder graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) m.mert.cfg.num_mel_bins, T_mel);
    ggml_set_name(g->input, "ss2_enc_mel");
    ggml_set_input(g->input);

    g->pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_sub);
    ggml_set_name(g->pos, "ss2_enc_pos");
    ggml_set_input(g->pos);

    // state 0: post-subsampling embedding, BEFORE any Conformer block runs
    // (doc 20 §2.5 — "hidden at that point is the subsampling module's
    // output"). Reused verbatim from yue2-mert.h.
    ggml_tensor * h0 = yue2_mert_sub_build(ctx, m.mert, g->grn_eps, g->input);

    g->hidden.clear();
    if (want_hidden) {
        g->hidden.push_back(h0);
    }

    ggml_tensor * mixed = ggml_scale(ctx, h0, g->softmax_w[0]);

    ggml_tensor * x = h0;
    for (size_t n = 0; n < m.mert.w.blk.size(); n++) {
        // gelu_tanh=false: exact erf, doc 20 §2.3 ("F.gelu / nn.GELU with
        // approximate='none' at every site, no fused kernel to override it") —
        // same determination yue2-mert.h's own header already made for this
        // identical Conformer block code.
        x = yue2_mert_detail::mert_block(ctx, m.mert.cfg, m.mert.w.blk[n], x, g->pos, /*tanh_gelu=*/false);
        if (want_hidden) {
            g->hidden.push_back(x);
        }
        ggml_tensor * term = ggml_scale(ctx, x, g->softmax_w[n + 1]);
        mixed              = ggml_add(ctx, mixed, term);
    }

    g->mixed = mixed;
    ggml_set_name(g->mixed, "ss2_enc_mixed");
    ggml_set_output(g->mixed);

    g->memory = ggml_add(ctx, ggml_mul_mat(ctx, m.head.proj_w, mixed), m.head.proj_b);
    ggml_set_name(g->memory, "ss2_enc_memory");
    ggml_set_output(g->memory);

    if (want_hidden) {
        for (size_t i = 0; i < g->hidden.size(); i++) {
            char nm[48];
            snprintf(nm, sizeof(nm), "ss2_enc_hidden%02d", (int) i);
            ggml_set_name(g->hidden[i], nm);
            ggml_set_output(g->hidden[i]);
        }
    }

    g->graph = ggml_new_graph_custom(ctx, SS2_ENC_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->memory);
    ggml_build_forward_expand(g->graph, g->mixed);
    for (ggml_tensor * t : g->hidden) {
        ggml_build_forward_expand(g->graph, t);
    }

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf   = nullptr;
        g->graph  = nullptr;
        g->memory = nullptr;
        g->mixed  = nullptr;
        if (err) {
            *err = "SheetSage2 encoder graph allocation failed (out of VRAM?) for T_mel=" + std::to_string(T_mel);
        }
        return false;
    }

    g->gctx              = ctx;
    g->graph_T_mel       = T_mel;
    g->graph_want_hidden = want_hidden;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr,
            "[SheetSage2] Encoder graph: T_mel=%lld -> T_sub=%lld, %zu blocks, %d nodes, %d splits, "
            "compute %.0f MB%s\n",
            (long long) T_mel, (long long) T_sub, m.mert.w.blk.size(), ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0),
            want_hidden ? " (+hidden-state taps)" : "");
    return true;
}

// ── Window pad/trim (doc 20 §1.2) ───────────────────────────────────────────

// Pads (right side, zero-fill) or trims `pcm` to exactly `window_samples`.
// Trimming is a defensive no-op path, never exercised by a correct caller:
// the pipeline lane's window plan always hands exactly one window's worth of
// samples (already <= window_samples) per call, the same way
// slice_audio()/`_prepare_audio` do in the reference (doc 20 §1.2 points 1-2).
static void sheetsage_pad_window(const float * pcm, int64_t n_samples, int64_t window_samples,
                                 std::vector<float> * out) {
    out->assign((size_t) std::max<int64_t>(window_samples, 0), 0.0f);
    const int64_t n = std::min(n_samples, window_samples);
    if (n > 0 && pcm) {
        std::memcpy(out->data(), pcm, (size_t) n * sizeof(float));
    }
}

// Doc 20 §1.2 point 3: pad the window length itself up to a multiple of
// `samples_per_frame` (a no-op for the pinned 300 s/24 kHz/960-stride config
// — 7,200,000 / 960 = 7500 exactly — but implemented so a future config that
// doesn't divide evenly is still correct).
static int64_t sheetsage_window_samples(const SheetSageModel & m) {
    const double  sr      = (double) m.mert.cfg.sample_rate;
    int64_t       samples = (int64_t) std::llround((double) m.cfg.window_seconds * sr);
    const int64_t stride  = (int64_t) m.mert.cfg.samples_per_frame;
    if (stride > 0 && samples % stride != 0) {
        samples += stride - (samples % stride);
    }
    return samples;
}

// ── Public API ───────────────────────────────────────────────────────────────

struct SheetSageEncodeOptions {
    // Keep every hidden_states[k] (k = 0..hidden_state_count-1) as a returned
    // buffer, for --sheetsage-encoder-parity. Off (default) for production
    // inference, which only needs `memory`.
    bool want_hidden_states = false;

    // NOTE on "exact" precision (doc 23, 2026-09-16 investigation): there is
    // no per-encode "exact" switch here, deliberately. The precision lever
    // that closes the gap to the F32 oracle is which TYPE the encoder's
    // matmul weights were loaded as — SheetSageModelLoadOptions::exact
    // (sheetsage-model.h) — which is a LOAD-TIME decision, since the weights
    // are already resident in the backend buffer by the time any window is
    // encoded. There is also no flash-vs-exact attention choice to make at
    // this level: yue2_mert_detail::mert_attn() (yue2-mert.h) never uses
    // ggml_flash_attn_ext for this encoder in the first place — QK^T/AV are
    // plain ggml_mul_mat between F32 activation tensors and the softmax is
    // already the manual, always-exact ggml_soft_max_ext(..., mask=nullptr,
    // ...) path. So this struct has nothing left to toggle for attention.
};

struct SheetSageEncodeResult {
    std::vector<float> memory;           // [T_sub, 512]  row-major -- == 06_memory
    std::vector<float> memory_pre_proj;  // [T_sub, 1024] row-major -- == 05_memory_pre_proj
    std::vector<float> softmax_weights;  // [hidden_state_count]    -- == 04_layer_weight_softmax
    int64_t             T_sub = 0;

    // [hidden_state_count][T_sub * 1024], row-major [T_sub,1024] per entry --
    // == 03_hidden_00..NN.f32. Only populated when opt.want_hidden_states.
    std::vector<std::vector<float>> hidden_states;
};

// One window, start to finish: pad/trim -> mel (CPU) -> subsample + 24
// Conformer blocks + weighted sum + projection (one GPU/CPU ggml graph) ->
// memory. `pcm`/`n_samples` is mono 24 kHz audio for JUST this window (the
// pipeline lane owns the sliding-window plan — hop/overlap/lookahead — and
// hands this function one window at a time); this function does the final
// pad/trim to the exact window length itself, per doc 20 §1.2, so a window
// shorter than 300 s (the common case: the last window of a short track) is
// handled correctly without the caller having to know the padded length.
//
// Not thread-safe: one cached graph per SheetSageEncoderGraph, same
// contract as yue2-mert.h's Yue2MertEncGraph.
static bool sheetsage_encode_window(const SheetSageModel & m, SheetSageEncoderGraph * g, const float * pcm,
                                    int64_t n_samples, const SheetSageEncodeOptions & opt,
                                    SheetSageEncodeResult * out, std::string * err) {
    *out = SheetSageEncodeResult{};
    if (!sheetsage_enc_prepare(m, g, err)) {
        return false;
    }

    const int64_t window_samples = sheetsage_window_samples(m);
    std::vector<float> win;
    sheetsage_pad_window(pcm, n_samples, window_samples, &win);

    std::vector<float> mel;
    int64_t             T_mel = 0;
    if (!yue2_mert_mel(m.mert, win.data(), (int64_t) win.size(), &mel, &T_mel, err)) {
        return false;
    }

    if (!sheetsage_enc_ensure_graph(m, g, T_mel, opt.want_hidden_states, err)) {
        return false;
    }

    const int64_t T_sub = g->memory->ne[1];
    ggml_backend_tensor_set(g->input, mel.data(), 0, ggml_nbytes(g->input));

    // Positions restart at 0 for every window (doc 20 §2.2: "RoPE covers
    // positions 0..7499 for every window, padding included, every time").
    std::vector<int32_t> pos((size_t) T_sub);
    for (int64_t t = 0; t < T_sub; t++) {
        pos[(size_t) t] = (int32_t) t;
    }
    ggml_backend_tensor_set(g->pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "SheetSage2 encoder graph compute failed";
        }
        return false;
    }

    out->T_sub           = T_sub;
    out->softmax_weights = g->softmax_w;

    const int64_t PC = g->memory->ne[0];  // 512
    out->memory.resize((size_t) (PC * T_sub));
    ggml_backend_tensor_get(g->memory, out->memory.data(), 0, out->memory.size() * sizeof(float));

    const int64_t HC = g->mixed->ne[0];  // 1024
    out->memory_pre_proj.resize((size_t) (HC * T_sub));
    ggml_backend_tensor_get(g->mixed, out->memory_pre_proj.data(), 0, out->memory_pre_proj.size() * sizeof(float));

    if (opt.want_hidden_states) {
        out->hidden_states.assign(g->hidden.size(), {});
        for (size_t i = 0; i < g->hidden.size(); i++) {
            out->hidden_states[i].resize((size_t) (g->hidden[i]->ne[0] * g->hidden[i]->ne[1]));
            ggml_backend_tensor_get(g->hidden[i], out->hidden_states[i].data(), 0,
                                    out->hidden_states[i].size() * sizeof(float));
        }
    }
    return true;
}
