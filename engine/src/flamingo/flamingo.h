// flamingo.h: Music Flamingo (nvidia/music-flamingo-2601-hf) for ace-caption.
//
// Weights come from engine/tools/convert-flamingo.py:
//   flamingo-aud-f16.gguf   arch "mfl-aud": AF-Whisper encoder + learned positions
//                           + the 2-layer GELU projector into the LM
//   flamingo-lm-q8_0.gguf   arch "qwen2":   the 7B Qwen2 LM + tokenizer
//
// Pipeline (transformers modeling_musicflamingo.py is the source of truth):
//
//   16 kHz mono -> 30 s windows, each zero-padded to 480000 samples
//     -> Whisper log-mel per window [128, 3000]   (moss-mel.h; max is PER WINDOW)
//     -> conv1 k3 p1 + GELU, conv2 k3 s2 p1 + GELU -> [1280, 1500] + learned positions
//     -> trimmed to the window's valid length, 32 pre-norm Whisper layers
//     -> AvgPool(2) over time, LayerNorm         -> [1280, <=750]
//     -> rotary time embedding (host, float64, exactly as the reference does it)
//     -> projector: linear 1280->3584, GELU, linear 3584->3584
//     -> spliced over the <sound> rows of the prompt
//     -> Qwen2-7B: RMSNorm, q/k/v WITH bias, NeoX RoPE 1e6, GQA 28/4, SwiGLU
//
// THINGS THAT FAIL QUIETLY
// ------------------------
// 1. The mel max/floor is per 30 s window, over the zero-padded window -- that is what
//    WhisperFeatureExtractor does per batch item. Not per track (MOSS is per track).
// 2. Partial last window: the reference runs the full 3000 frames with a key mask. Here
//    the conv stem runs on all 3000 frames (so the boundary frame sees the same padded
//    neighbours) and THEN the sequence is trimmed to the valid length. Every encoder op
//    after the convs is per-token or attention, so trimming == masking, exactly.
// 3. GELU is the erf form everywhere (ACT2FN["gelu"]), never the tanh approximation.
// 4. Whisper k_proj has no bias; q is pre-scaled by hd^-0.5 and attention runs unscaled.
// 5. The RoTE window axis uses round(window_start / 30 s) / 1200, the time axis uses
//    frame / 1200 * 2pi, and both are multiplied by -timestamp * 2pi. Rotate-half is the
//    INTERLEAVED form (pairs 2i, 2i+1), not the split-halves form.

#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "moss/moss-mel.h"
#include "weight-ctx.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace flamingo {

// ---------------------------------------------------------------------------
// Audio tower
// ---------------------------------------------------------------------------

struct AudioHParams {
    uint32_t n_layer = 0, d_model = 0, n_head = 0, n_ffn = 0, n_mels = 0, n_pos = 0;
    uint32_t lm_embd = 0, sample_rate = 0, n_fft = 0, hop = 0, chunk_s = 0, max_audio_s = 0;
    uint32_t rote_head_dim = 0, rote_max_pos = 0;
    uint32_t audio_token_id = 0, audio_bos_id = 0, audio_eos_id = 0;
    float    eps = 1e-5f, rote_partial = 0.0f, rote_theta = 0.0f, frame_step = 0.0f;
};

struct AudioBlock {
    ggml_tensor *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *ln2_w, *ln2_b, *up_w, *up_b, *down_w, *down_b;
};

struct AudioTower {
    AudioHParams hp;
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr, *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *pos = nullptr, *norm_w = nullptr, *norm_b = nullptr;
    ggml_tensor *proj1_w = nullptr, *proj1_b = nullptr, *proj2_w = nullptr, *proj2_b = nullptr;
    std::vector<AudioBlock> blocks;
    WeightCtx wctx = {};
};

inline void free_audio(AudioTower * m) {
    wctx_free(&m->wctx);
    *m = {};
}

inline bool load_audio(AudioTower * m, const char * path, ggml_backend_t backend) {
    *m = {};
    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        return false;
    }
    const char * arch = gf_get_str(gf, "general.architecture");
    if (!arch || std::string(arch) != "mfl-aud") {
        fprintf(stderr, "[Flamingo] %s: architecture '%s', expected 'mfl-aud'\n", path,
                arch ? arch : "(none)");
        gf_close(&gf);
        return false;
    }
    AudioHParams & h = m->hp;
    auto u = [&](const char * k) { return gf_get_u32(gf, (std::string("mfl.audio.") + k).c_str()); };
    auto f = [&](const char * k) { return gf_get_f32(gf, (std::string("mfl.audio.") + k).c_str()); };
    h.n_layer = u("block_count");          h.d_model = u("embedding_length");
    h.n_head = u("head_count");            h.n_ffn = u("feed_forward_length");
    h.n_mels = u("n_mels");                h.n_pos = u("max_source_positions");
    h.lm_embd = u("lm_embedding_length");  h.sample_rate = u("sample_rate");
    h.n_fft = u("n_fft");                  h.hop = u("hop_length");
    h.chunk_s = u("chunk_seconds");        h.max_audio_s = u("max_audio_seconds");
    h.rote_head_dim = u("rote_head_dim");  h.rote_max_pos = u("rote_max_positions");
    h.audio_token_id = u("audio_token_id"); h.audio_bos_id = u("audio_bos_id");
    h.audio_eos_id = u("audio_eos_id");
    h.eps = f("layer_norm_eps");           h.rote_partial = f("rote_partial");
    h.rote_theta = f("rote_theta");        h.frame_step = f("frame_step");
    if (!h.n_layer || !h.d_model || !h.n_head || !h.lm_embd) {
        fprintf(stderr, "[Flamingo] %s: missing audio hparams\n", path);
        gf_close(&gf);
        return false;
    }

    wctx_init(&m->wctx, (int) (32 + 16 * h.n_layer));
    auto t = [&](const std::string & n) { return gf_load_tensor(&m->wctx, gf, n); };
    m->conv1_w = t("aud.conv1.weight"); m->conv1_b = t("aud.conv1.bias");
    m->conv2_w = t("aud.conv2.weight"); m->conv2_b = t("aud.conv2.bias");
    m->pos = t("aud.pos.weight");
    m->norm_w = t("aud.norm.weight");   m->norm_b = t("aud.norm.bias");
    m->proj1_w = t("aud.proj.linear_1.weight"); m->proj1_b = t("aud.proj.linear_1.bias");
    m->proj2_w = t("aud.proj.linear_2.weight"); m->proj2_b = t("aud.proj.linear_2.bias");
    m->blocks.resize(h.n_layer);
    for (uint32_t i = 0; i < h.n_layer; ++i) {
        const std::string p = "aud.blk." + std::to_string(i) + ".";
        AudioBlock & b = m->blocks[i];
        b.ln1_w = t(p + "attn_norm.weight"); b.ln1_b = t(p + "attn_norm.bias");
        b.q_w = t(p + "attn_q.weight");      b.q_b = t(p + "attn_q.bias");
        b.k_w = t(p + "attn_k.weight");
        b.v_w = t(p + "attn_v.weight");      b.v_b = t(p + "attn_v.bias");
        b.o_w = t(p + "attn_out.weight");    b.o_b = t(p + "attn_out.bias");
        b.ln2_w = t(p + "ffn_norm.weight");  b.ln2_b = t(p + "ffn_norm.bias");
        b.up_w = t(p + "ffn_up.weight");     b.up_b = t(p + "ffn_up.bias");
        b.down_w = t(p + "ffn_down.weight"); b.down_b = t(p + "ffn_down.bias");
    }
    const bool ok = wctx_alloc(&m->wctx, backend);
    gf_close(&gf);
    if (!ok) {
        free_audio(m);
    }
    return ok;
}

namespace detail {

inline ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                                ggml_tensor * b, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

// Conv1d (+bias) with an explicit F32 im2col: w [K, IC, OC], x [T, IC, N] -> [OC, OT, N].
// Output is channels-first, which is what the transformer stack wants.
inline ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int pad) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]),
                                   ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]));
    y = ggml_reshape_3d(ctx, y, w->ne[2], col->ne[1], col->ne[2]);
    return ggml_add(ctx, y, b);
}

struct Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    void free() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
        *this = {};
    }
    bool begin(size_t n_nodes) {
        ggml_init_params ip = { ggml_tensor_overhead() * (n_nodes + 64) +
                                    ggml_graph_overhead_custom(n_nodes, false),
                                nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        gf = ggml_new_graph_custom(ctx, n_nodes, false);
        return true;
    }
    bool alloc_on(ggml_backend_t backend) {
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(alloc, gf);
    }
};

}  // namespace detail

// Encodes N windows that all keep the same number of valid post-conv frames `Lv`
// (so full windows batch together; a short last window runs on its own).
// mel: N x row-major [n_mels, 3000]. Writes N x [P, d_model] pooled rows, P = (Lv-2)/2+1.
inline bool encode_windows(const AudioTower & m, const float * mel, int N, int Lv,
                           ggml_backend_t backend, std::vector<float> & out, int * out_P) {
    using namespace detail;
    const AudioHParams & h = m.hp;
    const int64_t D = h.d_model, H = h.n_head, hd = D / H;
    const int64_t T = (int64_t) h.n_pos * 2;   // 3000 mel frames
    // Python's floor division: a window with one valid frame pools to ZERO rows.
    // C++ truncates (1-2)/2 to 0 and would give one row, reading past the window
    // (a track a few ms over a multiple of 30 s crashed the whole batch).
    const int64_t P = Lv >= 2 ? (Lv - 2) / 2 + 1 : 0;
    if (P <= 0) {
        *out_P = 0;
        return true;
    }

    Graph g;
    if (!g.begin((size_t) (256 + 64 * h.n_layer))) return false;
    ggml_context * ctx = g.ctx;

    ggml_tensor * in_mel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, h.n_mels, N);
    ggml_set_input(in_mel);

    ggml_tensor * x = ggml_gelu_erf(ctx, conv1d(ctx, m.conv1_w, m.conv1_b, in_mel, 1, 1));   // [D, T, N]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));                                     // [T, D, N]
    x = ggml_gelu_erf(ctx, conv1d(ctx, m.conv2_w, m.conv2_b, x, 2, 1));                       // [D, 1500, N]
    x = ggml_add(ctx, x, m.pos);
    if (Lv < (int) h.n_pos) {
        x = ggml_cont(ctx, ggml_view_3d(ctx, x, D, Lv, N, x->nb[1], x->nb[2], 0));
    }

    for (uint32_t il = 0; il < h.n_layer; ++il) {
        const AudioBlock & b = m.blocks[il];
        ggml_tensor * cur = layer_norm(ctx, x, b.ln1_w, b.ln1_b, h.eps);
        ggml_tensor * q = ggml_scale(ctx, ggml_add(ctx, ggml_mul_mat(ctx, b.q_w, cur), b.q_b),
                                     1.0f / sqrtf((float) hd));
        ggml_tensor * k = ggml_mul_mat(ctx, b.k_w, cur);
        ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, b.v_w, cur), b.v_b);
        // [D, L, N] -> [hd, H, L, N] -> [hd, L, H, N]
        auto heads = [&](ggml_tensor * t) {
            return ggml_permute(ctx, ggml_reshape_4d(ctx, t, hd, H, Lv, N), 0, 2, 1, 3);
        };
        q = heads(q);
        ggml_tensor * kf = ggml_cast(ctx, heads(k), GGML_TYPE_F16);
        ggml_tensor * vf = ggml_cast(ctx, heads(v), GGML_TYPE_F16);
        ggml_tensor * a = ggml_flash_attn_ext(ctx, q, kf, vf, nullptr, 1.0f, 0.0f, 0.0f);   // [hd, H, L, N]
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_3d(ctx, a, D, Lv, N);
        x = ggml_add(ctx, x, ggml_add(ctx, ggml_mul_mat(ctx, b.o_w, a), b.o_b));

        cur = layer_norm(ctx, x, b.ln2_w, b.ln2_b, h.eps);
        cur = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, b.up_w, cur), b.up_b));
        x = ggml_add(ctx, x, ggml_add(ctx, ggml_mul_mat(ctx, b.down_w, cur), b.down_b));
    }

    // AvgPool1d(2, 2): mean of frame pairs, dropping an odd trailing frame.
    ggml_tensor * ev = ggml_view_3d(ctx, x, D, P, N, x->nb[1] * 2, x->nb[2], 0);
    ggml_tensor * od = ggml_view_3d(ctx, x, D, P, N, x->nb[1] * 2, x->nb[2], x->nb[1]);
    x = ggml_scale(ctx, ggml_add(ctx, ggml_cont(ctx, ev), ggml_cont(ctx, od)), 0.5f);
    x = layer_norm(ctx, x, m.norm_w, m.norm_b, h.eps);
    ggml_set_output(x);
    ggml_build_forward_expand(g.gf, x);

    if (!g.alloc_on(backend)) {
        fprintf(stderr, "[Flamingo] encoder graph allocation failed (N=%d)\n", N);
        g.free();
        return false;
    }
    ggml_backend_tensor_set(in_mel, mel, 0, (size_t) N * h.n_mels * T * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize((size_t) (D * P * N));
        ggml_backend_tensor_get(x, out.data(), 0, out.size() * sizeof(float));
        *out_P = (int) P;
    }
    g.free();
    return ok;
}

// Rotary time embedding, in place on rows [n, D] that came from window `w`
// (frames 0..n-1). Float64 like the reference.
inline void apply_rote(const AudioHParams & h, float * rows, int n, int w) {
    const int dim = (int) ((float) h.rote_head_dim * h.rote_partial);    // 256
    const int half = dim / 2;                                            // 128 inv_freqs
    const double kTwoPi = 6.283185307179586;
    const double max_pos = (double) h.rote_max_pos;                      // 1200
    const double step = (double) h.frame_step * 4.0;                     // 0.04 s per row
    const double win_len = step * (double) h.n_pos / 2.0;                // 750 rows = 30 s
    std::vector<double> inv(half);
    for (int i = 0; i < half; ++i) {
        inv[i] = 1.0 / std::pow((double) h.rote_theta, (2.0 * i) / (double) dim);
    }
    const double win_start = (double) w * win_len;
    const double win_pos = std::round(win_start / win_len) / max_pos;
    std::vector<double> fr(2 * dim);
    const int D = (int) h.d_model;
    for (int f = 0; f < n; ++f) {
        const double ts = win_start + (double) f * step;
        const double ang = -ts * kTwoPi;
        for (int i = 0; i < half; ++i) {
            const double wf = win_pos * inv[i];
            const double tf = ((double) f / max_pos * kTwoPi) * inv[i];
            fr[2 * i] = fr[2 * i + 1] = wf * ang;
            fr[dim + 2 * i] = fr[dim + 2 * i + 1] = tf * ang;
        }
        float * r = rows + (size_t) f * D;
        for (int j = 0; j < 2 * dim; j += 2) {
            const double x0 = r[j], x1 = r[j + 1];
            const double c0 = std::cos(fr[j]), s0 = std::sin(fr[j]);
            const double c1 = std::cos(fr[j + 1]), s1 = std::sin(fr[j + 1]);
            r[j] = (float) (x0 * c0 - x1 * s0);
            r[j + 1] = (float) (x1 * c1 + x0 * s1);
        }
    }
}

// Projector over rows [n, D] -> [n, lm_embd].
inline bool project(const AudioTower & m, const std::vector<float> & rows, int n,
                    ggml_backend_t backend, std::vector<float> & out) {
    detail::Graph g;
    if (!g.begin(64)) return false;
    ggml_tensor * in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, m.hp.d_model, n);
    ggml_set_input(in);
    ggml_tensor * y = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.proj1_w, in), m.proj1_b);
    y = ggml_gelu_erf(g.ctx, y);
    y = ggml_add(g.ctx, ggml_mul_mat(g.ctx, m.proj2_w, y), m.proj2_b);
    ggml_set_output(y);
    ggml_build_forward_expand(g.gf, y);
    if (!g.alloc_on(backend)) {
        g.free();
        return false;
    }
    ggml_backend_tensor_set(in, rows.data(), 0, rows.size() * sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        out.resize((size_t) m.hp.lm_embd * n);
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    }
    g.free();
    return ok;
}

// Whole track: pcm 16 kHz mono -> audio embeddings [n_tokens, lm_embd].
// `max_batch` caps how many full windows share one encoder graph (VRAM vs speed).
inline bool encode_audio(const AudioTower & m, const std::vector<float> & pcm,
                         ggml_backend_t backend, std::vector<float> & embeds, int * n_tokens,
                         int max_batch = 8) {
    const AudioHParams & h = m.hp;
    const size_t win = (size_t) h.sample_rate * h.chunk_s;                    // 480000
    const size_t max_win = h.max_audio_s / h.chunk_s;                          // 40
    size_t n_win = std::max<size_t>(1, (pcm.size() + win - 1) / win);
    if (n_win > max_win) {
        fprintf(stderr, "[Flamingo] audio is %.0f s; using the first %u s\n",
                (double) pcm.size() / h.sample_rate, h.max_audio_s);
        n_win = max_win;
    }
    moss::MelParams mp;
    mp.sample_rate = (int) h.sample_rate;
    mp.n_fft = (int) h.n_fft;
    mp.hop_length = (int) h.hop;
    mp.n_mels = (int) h.n_mels;
    const int T = (int) h.n_pos * 2;

    // Per-window mel + valid post-conv length. Windows are independent (the log floor
    // is per window), so each gets its own thread.
    const auto t_mel0 = std::chrono::steady_clock::now();
    std::vector<float> mel((size_t) n_win * h.n_mels * T);
    std::vector<int> lv(n_win), frames(n_win, 0);
    {
        std::vector<std::thread> pool;
        const size_t n_thr = std::max<size_t>(1, std::min<size_t>(n_win, std::thread::hardware_concurrency()));
        for (size_t ti = 0; ti < n_thr; ++ti) {
            pool.emplace_back([&, ti]() {
                std::vector<float> chunk(win);
                for (size_t w = ti; w < n_win; w += n_thr) {
                    const size_t s0 = w * win;
                    const size_t n = std::min(win, pcm.size() - s0);
                    std::fill(chunk.begin(), chunk.end(), 0.0f);
                    std::copy(pcm.begin() + (long) s0, pcm.begin() + (long) (s0 + n), chunk.begin());
                    std::vector<float> wm = moss::log_mel(chunk.data(), win, mp, &frames[w]);
                    if (frames[w] == T) {
                        std::copy(wm.begin(), wm.end(), mel.begin() + (long) (w * h.n_mels * T));
                    }
                    const int valid = (int) ((n + h.hop - 1) / h.hop);
                    lv[w] = (valid - 1) / 2 + 1;
                }
            });
        }
        for (std::thread & t : pool) t.join();
    }
    for (size_t w = 0; w < n_win; ++w) {
        if (frames[w] != T) {
            fprintf(stderr, "[Flamingo] mel produced %d frames, expected %d\n", frames[w], T);
            return false;
        }
    }
    fprintf(stderr, "[Flamingo] mel: %zu windows in %.2fs\n", n_win,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_mel0).count());

    // Encode: consecutive windows with the same valid length go through one batched graph.
    std::vector<float> rows;   // [n_tok, D], RoTE applied
    const int D = (int) h.d_model;
    for (size_t w0 = 0; w0 < n_win;) {
        size_t w1 = w0 + 1;
        while (w1 < n_win && lv[w1] == lv[w0] && (int) (w1 - w0) < max_batch) ++w1;
        std::vector<float> out;
        int P = 0;
        if (!encode_windows(m, mel.data() + w0 * h.n_mels * T, (int) (w1 - w0), lv[w0], backend,
                            out, &P)) {
            return false;
        }
        for (size_t w = w0; w < w1; ++w) {
            float * r = out.data() + (w - w0) * (size_t) P * D;
            apply_rote(h, r, P, (int) w);
            rows.insert(rows.end(), r, r + (size_t) P * D);
        }
        w0 = w1;
    }
    *n_tokens = (int) (rows.size() / D);
    return project(m, rows, *n_tokens, backend, embeds);
}

// ---------------------------------------------------------------------------
// LM: Qwen2
// ---------------------------------------------------------------------------

struct LmLayer {
    ggml_tensor *attn_norm, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w;
    ggml_tensor *ffn_norm, *gate, *up, *down;
};

struct LmModel {
    uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_vocab = 0;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    ggml_tensor *tok = nullptr, *out_norm = nullptr, *output = nullptr;
    std::vector<LmLayer> layers;
    WeightCtx wctx = {};
};

inline void free_lm(LmModel * m) {
    wctx_free(&m->wctx);
    *m = {};
}

inline bool load_lm(LmModel * m, const char * path, ggml_backend_t backend) {
    *m = {};
    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        return false;
    }
    const char * arch = gf_get_str(gf, "general.architecture");
    if (!arch || std::string(arch) != "qwen2") {
        fprintf(stderr, "[Flamingo] %s: architecture '%s', expected 'qwen2'\n", path,
                arch ? arch : "(none)");
        gf_close(&gf);
        return false;
    }
    m->n_layer = gf_get_u32(gf, "qwen2.block_count");
    m->n_embd = gf_get_u32(gf, "qwen2.embedding_length");
    m->n_head = gf_get_u32(gf, "qwen2.attention.head_count");
    m->n_head_kv = gf_get_u32(gf, "qwen2.attention.head_count_kv");
    m->head_dim = gf_get_u32(gf, "qwen2.attention.key_length");
    m->n_vocab = gf_get_u32(gf, "qwen2.vocab_size");
    m->rms_eps = gf_get_f32(gf, "qwen2.attention.layer_norm_rms_epsilon");
    m->rope_theta = gf_get_f32(gf, "qwen2.rope.freq_base");
    wctx_init(&m->wctx, (int) (16 + 14 * m->n_layer));
    auto t = [&](const std::string & n) { return gf_load_tensor(&m->wctx, gf, n); };
    m->tok = t("token_embd.weight");
    m->out_norm = t("output_norm.weight");
    m->output = t("output.weight");
    m->layers.resize(m->n_layer);
    for (uint32_t i = 0; i < m->n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LmLayer & l = m->layers[i];
        l.attn_norm = t(p + "attn_norm.weight");
        l.q_w = t(p + "attn_q.weight"); l.q_b = t(p + "attn_q.bias");
        l.k_w = t(p + "attn_k.weight"); l.k_b = t(p + "attn_k.bias");
        l.v_w = t(p + "attn_v.weight"); l.v_b = t(p + "attn_v.bias");
        l.o_w = t(p + "attn_output.weight");
        l.ffn_norm = t(p + "ffn_norm.weight");
        l.gate = t(p + "ffn_gate.weight");
        l.up = t(p + "ffn_up.weight");
        l.down = t(p + "ffn_down.weight");
    }
    const bool ok = wctx_alloc(&m->wctx, backend);
    gf_close(&gf);
    if (!ok) {
        free_lm(m);
    }
    return ok;
}

// F16 KV cache, all layers in one buffer, zeroed so a padded window reads finite values.
struct Kv {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t n_ctx = 0, n_past = 0;
};

inline void kv_free(Kv * kv) {
    if (kv->buf) ggml_backend_buffer_free(kv->buf);
    if (kv->ctx) ggml_free(kv->ctx);
    *kv = {};
}

inline bool kv_init(Kv * kv, const LmModel & m, int64_t n_ctx, ggml_backend_t backend) {
    *kv = {};
    n_ctx = GGML_PAD(n_ctx, 256);
    ggml_init_params ip = { ggml_tensor_overhead() * (2 * m.n_layer + 8), nullptr, true };
    kv->ctx = ggml_init(ip);
    if (!kv->ctx) return false;
    for (uint32_t i = 0; i < m.n_layer; ++i) {
        kv->k.push_back(ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F16, m.head_dim, n_ctx, m.n_head_kv));
        kv->v.push_back(ggml_new_tensor_3d(kv->ctx, GGML_TYPE_F16, m.head_dim, n_ctx, m.n_head_kv));
    }
    kv->buf = ggml_backend_alloc_ctx_tensors(kv->ctx, backend);
    if (!kv->buf) {
        kv_free(kv);
        return false;
    }
    ggml_backend_buffer_clear(kv->buf, 0);
    kv->n_ctx = n_ctx;
    return true;
}

// One evaluation of T tokens at positions [n_past, n_past+T). The graph is kept
// between calls while (T, padded window) stay the same, so the decode loop rebuilds
// it once per 256 tokens instead of once per token.
struct Eval {
    detail::Graph g;
    int64_t key_T = -1, key_pad = -1;
    bool key_audio = false;
    ggml_tensor *ids = nullptr, *pos = nullptr, *rows = nullptr, *mask = nullptr;
    ggml_tensor *amask = nullptr, *avals = nullptr, *logits = nullptr;
    std::vector<uint16_t> mask_data;
    void free() { g.free(); key_T = key_pad = -1; }
};

// ids [T]; audio_vals [H*T] (zeros at text rows) + text_mask [T] for the prefill splice,
// both empty on decode. Returns last-position logits.
inline bool eval(const LmModel & m, Kv & kv, Eval & ev, const std::vector<int32_t> & ids,
                 const std::vector<float> & audio_vals, const std::vector<float> & text_mask,
                 ggml_backend_t backend, std::vector<float> * logits) {
    const int64_t T = (int64_t) ids.size(), H = m.n_embd, D = m.head_dim;
    const int64_t Nh = m.n_head, Nkv = m.n_head_kv;
    const int64_t n_past = kv.n_past, n_kv = n_past + T;
    const int64_t pad = std::min<int64_t>(GGML_PAD(n_kv, 256), kv.n_ctx);
    const bool audio = !audio_vals.empty();
    if (n_kv > kv.n_ctx) {
        fprintf(stderr, "[Flamingo] context full (%lld > %lld)\n", (long long) n_kv,
                (long long) kv.n_ctx);
        return false;
    }

    if (ev.key_T != T || ev.key_pad != pad || ev.key_audio != audio) {
        ev.free();
        if (!ev.g.begin((size_t) (256 + 48 * m.n_layer))) return false;
        ggml_context * ctx = ev.g.ctx;
        ev.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);   ggml_set_input(ev.ids);
        ev.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);   ggml_set_input(ev.pos);
        ev.rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);  ggml_set_input(ev.rows);
        ev.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, pad, T); ggml_set_input(ev.mask);
        ggml_tensor * h = ggml_get_rows(ctx, m.tok, ev.ids);   // [H, T]
        if (audio) {
            ev.amask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, T); ggml_set_input(ev.amask);
            ev.avals = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T); ggml_set_input(ev.avals);
            h = ggml_add(ctx, ggml_mul(ctx, h, ev.amask), ev.avals);
        }
        const float scale = 1.0f / sqrtf((float) D);
        for (uint32_t il = 0; il < m.n_layer; ++il) {
            const LmLayer & w = m.layers[il];
            ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, h, m.rms_eps), w.attn_norm);
            ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, n), w.q_b);
            ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, n), w.k_b);
            ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, n), w.v_b);
            q = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, q, D, Nh, T), ev.pos, nullptr, (int) D,
                              GGML_ROPE_TYPE_NEOX, 0, m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, ggml_reshape_3d(ctx, k, D, Nkv, T), ev.pos, nullptr, (int) D,
                              GGML_ROPE_TYPE_NEOX, 0, m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            v = ggml_reshape_3d(ctx, v, D, Nkv, T);
            ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));   // [D, T, Nkv]
            ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
            ggml_build_forward_expand(ev.g.gf, ggml_set_rows(ctx, kv.k[il], kp, ev.rows));
            ggml_build_forward_expand(ev.g.gf, ggml_set_rows(ctx, kv.v[il], vp, ev.rows));
            ggml_tensor * kw = ggml_view_3d(ctx, kv.k[il], D, pad, Nkv, kv.k[il]->nb[1], kv.k[il]->nb[2], 0);
            ggml_tensor * vw = ggml_view_3d(ctx, kv.v[il], D, pad, Nkv, kv.v[il]->nb[1], kv.v[il]->nb[2], 0);
            ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);                   // [D, T, Nh]
            ggml_tensor * a = ggml_flash_attn_ext(ctx, qp, kw, vw, ev.mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(a, GGML_PREC_F32);                        // [D, Nh, T]
            a = ggml_reshape_2d(ctx, a, H, T);
            h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.o_w, a));
            ggml_tensor * n2 = ggml_mul(ctx, ggml_rms_norm(ctx, h, m.rms_eps), w.ffn_norm);
            ggml_tensor * ff = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, w.gate, n2)),
                                        ggml_mul_mat(ctx, w.up, n2));
            h = ggml_add(ctx, h, ggml_mul_mat(ctx, w.down, ff));
        }
        h = ggml_mul(ctx, ggml_rms_norm(ctx, h, m.rms_eps), m.out_norm);
        ggml_tensor * last = ggml_view_2d(ctx, h, H, 1, h->nb[1], (size_t) (T - 1) * h->nb[1]);
        ev.logits = ggml_mul_mat(ctx, m.output, ggml_cont(ctx, last));
        ggml_set_output(ev.logits);
        ggml_build_forward_expand(ev.g.gf, ev.logits);
        if (!ev.g.alloc_on(backend)) {
            fprintf(stderr, "[Flamingo] LM graph allocation failed (T=%lld)\n", (long long) T);
            ev.free();
            return false;
        }
        ev.key_T = T;
        ev.key_pad = pad;
        ev.key_audio = audio;
    }

    std::vector<int32_t> pos((size_t) T);
    std::vector<int64_t> rows((size_t) T);
    for (int64_t i = 0; i < T; ++i) {
        pos[(size_t) i] = (int32_t) (n_past + i);
        rows[(size_t) i] = n_past + i;
    }
    ev.mask_data.assign((size_t) (pad * T), 0);
    const uint16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t i = 0; i < T; ++i) {
        for (int64_t j = n_past + i + 1; j < pad; ++j) {
            ev.mask_data[(size_t) (i * pad + j)] = neg_inf;
        }
    }
    ggml_backend_tensor_set(ev.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ev.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(ev.rows, rows.data(), 0, rows.size() * sizeof(int64_t));
    ggml_backend_tensor_set(ev.mask, ev.mask_data.data(), 0, ev.mask_data.size() * sizeof(uint16_t));
    if (audio) {
        ggml_backend_tensor_set(ev.amask, text_mask.data(), 0, text_mask.size() * sizeof(float));
        ggml_backend_tensor_set(ev.avals, audio_vals.data(), 0, audio_vals.size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend, ev.g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Flamingo] LM compute failed\n");
        return false;
    }
    logits->resize(m.n_vocab);
    ggml_backend_tensor_get(ev.logits, logits->data(), 0, logits->size() * sizeof(float));
    kv.n_past = n_kv;
    return true;
}

}  // namespace flamingo
