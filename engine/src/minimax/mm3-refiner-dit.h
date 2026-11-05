#pragma once
// minimax/mm3-refiner-dit.h — MM3 latent refiner DiT and its deterministic bridge.
//
// HOT-Step file. The 137M flow DiT of terminusresearch
// minimax-music3-latent-refiner v0.10 (model.py MiniMaxMusic3LatentRefiner) and
// bridge_sample, the loop the package's _refine_chunk runs per window:
//
//   x0 = (DAV latents - latent_mean) / latent_std, [L, 128]
//   schedule = linspace(1, 0, steps + 1) in F32, Euler from x0, no noise, no seed
//   x <- x - (s[i] - s[i+1]) * v(x, s[i])
//   out = x * latent_std + latent_mean, back to [128, L]
//
// One forward: tokens = [context (x0), target (x)], 2L of them. Both pass
// proj_in on [latent, MERT layer 7]; the target adds degraded_in_proj(x0), the
// context adds context_embed. Conditioning = time_embed(t) + task_embed[3] +
// style_proj(CLAP vector). Block i adds layer_cond_proj(MERT layer
// min(1+i, 11)) to both halves, then adaLN attention and MLP with RoPE whose
// positions restart at 0 for the target half (the duplicated cos/sin in
// model.py). The velocity is proj_out(out_norm(target half)).
//
// Weights: mm3-refiner-refiner-f32.gguf (arch "mm3-refiner-refiner") from
// tools/mm3-refiner/convert.py, torch names and shapes unchanged, with
// normalization.latent_{mean,std}. Conditioning comes from mm3-refiner-mert.h
// (all 13 normalised states) and a 512-d CLAP style vector.
//
// Gated by mm3-refiner-dit-probe / tools/mm3-refiner/check-dit.ps1.

// backend.h needs <string> first.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"
#include "yyjson.h"

static constexpr int     MM3_DIT_LATENT    = 128;
static constexpr int64_t MM3_DIT_DIM       = 768;
static constexpr int64_t MM3_DIT_COND      = 768;
static constexpr int64_t MM3_DIT_STYLE     = 512;
static constexpr int     MM3_DIT_DEPTH     = 12;
static constexpr int     MM3_DIT_HEADS     = 12;
static constexpr int     MM3_DIT_STATES    = 13;
static constexpr int     MM3_DIT_COND_LAYER = 7;
static constexpr int     MM3_DIT_TASK      = 3;  // bridge passes no task: the last of task_count + 1 rows
static constexpr int     MM3_DIT_TFREQ     = 128;
static constexpr float   MM3_DIT_EPS       = 1e-5f;
static constexpr int     MM3_DIT_MAX_NODES = 2048;

struct Mm3DitBlock {
    ggml_tensor *qkv_w, *out_w, *up_w, *up_b, *down_w, *down_b, *ada_w, *ada_b, *lc_w;
};

struct Mm3RefinerDit {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            wctx        = {};

    ggml_tensor *proj_in_w = nullptr, *proj_in_b = nullptr, *deg_w = nullptr, *ctx_embed = nullptr;
    ggml_tensor *t0_w = nullptr, *t0_b = nullptr, *t2_w = nullptr, *t2_b = nullptr;
    ggml_tensor *s0_w = nullptr, *s0_b = nullptr, *s2_w = nullptr, *s2_b = nullptr;
    ggml_tensor *task_w = nullptr, *out_w = nullptr, *out_b = nullptr;
    Mm3DitBlock  block[MM3_DIT_DEPTH] = {};

    std::vector<float> latent_mean;  // [128]
    std::vector<float> latent_std;   // [128]
};

static void mm3_dit_free(Mm3RefinerDit * m) {
    if (m->sched) {
        ggml_backend_sched_free(m->sched);
    }
    wctx_free(&m->wctx);
    if (m->backend) {
        backend_release(m->backend, m->cpu_backend);
    }
    *m = Mm3RefinerDit{};
}

static bool mm3_dit_check_config(const char * json, std::string * err) {
    yyjson_doc * doc = json ? yyjson_read(json, strlen(json), 0) : nullptr;
    if (!doc) {
        *err = "mm3.refiner.config missing or not JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    auto         num  = [&](const char * k) { return yyjson_get_num(yyjson_obj_get(root, k)); };
    const bool   ok   = num("latent_dim") == MM3_DIT_LATENT && num("cond_dim") == MM3_DIT_COND &&
                    num("d_model") == MM3_DIT_DIM && num("depth") == MM3_DIT_DEPTH && num("heads") == MM3_DIT_HEADS &&
                    num("mert_layer_count") == MM3_DIT_STATES && num("mert_input_layer") == MM3_DIT_COND_LAYER &&
                    num("style_dim") == MM3_DIT_STYLE && num("task_count") == MM3_DIT_TASK;
    yyjson_doc_free(doc);
    if (!ok) {
        *err = "refiner config differs from the v0.10 layout this port implements";
    }
    return ok;
}

static bool mm3_dit_load(Mm3RefinerDit * m, const std::string & path, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    mm3_dit_free(m);
    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    struct Close {
        GGUFModel * g;
        ~Close() { gf_close(g); }
    } close_gf{ &gf };
    auto str = [&](const char * k) {
        int64_t i = gguf_find_key(gf.gguf, k);
        return i >= 0 && gguf_get_kv_type(gf.gguf, i) == GGUF_TYPE_STRING ? gguf_get_val_str(gf.gguf, i) : nullptr;
    };
    const char * arch = str("general.architecture");
    if (!arch || std::string(arch) != "mm3-refiner-refiner") {
        *err = path + ": architecture is not mm3-refiner-refiner";
        return false;
    }
    if (!mm3_dit_check_config(str("mm3.refiner.config"), err)) {
        return false;
    }
    err->clear();
    // ne is the reversed torch shape; every tensor must be F32 with exactly it.
    auto find = [&](const std::string & name, std::initializer_list<int64_t> ne) -> ggml_tensor * {
        if (!err->empty()) {
            return nullptr;
        }
        ggml_tensor * src = gguf_find_tensor(gf.gguf, name.c_str()) < 0 ? nullptr : ggml_get_tensor(gf.meta, name.c_str());
        bool          ok  = src && src->type == GGML_TYPE_F32 && ggml_n_dims(src) <= (int) ne.size();
        int           d   = 0;
        for (int64_t n : ne) {
            ok = ok && src->ne[d++] == n;
        }
        if (!ok) {
            *err = "tensor " + name + " missing or wrong shape/type in " + path;
            return nullptr;
        }
        return src;
    };
    for (auto [name, out] : { std::pair{ "normalization.latent_mean", &m->latent_mean },
                              std::pair{ "normalization.latent_std", &m->latent_std } }) {
        if (find(name, { MM3_DIT_LATENT })) {
            out->resize(MM3_DIT_LATENT);
            std::memcpy(out->data(), gf.mapping + gf.data_offset + gguf_get_tensor_offset(gf.gguf, gguf_find_tensor(gf.gguf, name)),
                        MM3_DIT_LATENT * sizeof(float));
        }
    }
    for (float s : m->latent_std) {
        if (err->empty() && !(s > 0.0f && std::isfinite(s))) {
            *err = "normalization.latent_std has a non-positive entry";
        }
    }
    if (!err->empty()) {
        return false;
    }

    BackendPair bp = backend_init("MM3-DiT");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    wctx_init(&m->wctx, 16 + 9 * MM3_DIT_DEPTH);
    auto req = [&](const std::string & name, std::initializer_list<int64_t> ne) -> ggml_tensor * {
        return find(name, ne) ? gf_load_tensor(&m->wctx, gf, name) : nullptr;
    };
    const int64_t D = MM3_DIT_DIM, C = MM3_DIT_COND, F = 4 * D;
    m->proj_in_w = req("proj_in.weight", { MM3_DIT_LATENT + C, D });
    m->proj_in_b = req("proj_in.bias", { D });
    m->deg_w     = req("degraded_in_proj.weight", { MM3_DIT_LATENT, D });
    m->ctx_embed = req("context_embed", { D });
    m->t0_w      = req("time_embed.0.weight", { 2 * MM3_DIT_TFREQ, D });
    m->t0_b      = req("time_embed.0.bias", { D });
    m->t2_w      = req("time_embed.2.weight", { D, D });
    m->t2_b      = req("time_embed.2.bias", { D });
    m->s0_w      = req("style_proj.0.weight", { MM3_DIT_STYLE, D });
    m->s0_b      = req("style_proj.0.bias", { D });
    m->s2_w      = req("style_proj.2.weight", { D, D });
    m->s2_b      = req("style_proj.2.bias", { D });
    m->task_w    = req("task_embed.weight", { D, MM3_DIT_TASK + 1 });
    m->out_w     = req("proj_out.weight", { D, MM3_DIT_LATENT });
    m->out_b     = req("proj_out.bias", { MM3_DIT_LATENT });
    for (int i = 0; i < MM3_DIT_DEPTH; i++) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Mm3DitBlock &     B = m->block[i];
        B.qkv_w  = req(p + "qkv.weight", { D, 3 * D });
        B.out_w  = req(p + "attn_out.weight", { D, D });
        B.up_w   = req(p + "mlp.0.weight", { D, F });
        B.up_b   = req(p + "mlp.0.bias", { F });
        B.down_w = req(p + "mlp.2.weight", { F, D });
        B.down_b = req(p + "mlp.2.bias", { D });
        B.ada_w  = req(p + "adaln.weight", { D, 6 * D });
        B.ada_b  = req(p + "adaln.bias", { 6 * D });
        B.lc_w   = req(p + "layer_cond_proj.weight", { C, D });
    }
    if (!err->empty()) {
        mm3_dit_free(m);
        return false;
    }
    if (!wctx_alloc(&m->wctx, m->backend)) {
        *err = "backend buffer allocation failed for the refiner weights";
        mm3_dit_free(m);
        return false;
    }
    m->sched = backend_sched_new(bp, MM3_DIT_MAX_NODES);
    fprintf(stderr, "[MM3-DiT] Loaded %s: %.1f MB, %.0f ms\n", path.c_str(),
            (double) ggml_backend_buffer_get_size(m->wctx.buffer) / (1024.0 * 1024.0),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

// torch.linspace(1, 0, steps + 1) in F32, its two-sided formula included.
static std::vector<float> mm3_dit_schedule(int steps) {
    const int          n    = steps + 1;
    const float        step = (0.0f - 1.0f) / (float) steps;
    std::vector<float> s((size_t) n);
    for (int i = 0; i < n; i++) {
        s[(size_t) i] = i < n / 2 ? 1.0f + step * (float) i : 0.0f - step * (float) (n - 1 - i);
    }
    return s;
}

// timestep_features: [sin(t f 1000), cos(t f 1000)], f = exp(-ln(1e4) k / 128), F32 like torch.
static void mm3_dit_time_features(float t, float * out) {
    const float c = (float) -std::log(10000.0);
    for (int k = 0; k < MM3_DIT_TFREQ; k++) {
        const float a = t * std::exp(c * (float) k / (float) MM3_DIT_TFREQ) * 1000.0f;
        out[k]                 = std::sin(a);
        out[k + MM3_DIT_TFREQ] = std::cos(a);
    }
}

// DAV latents [128][L] -> normalised bridge input [L][128].
static std::vector<float> mm3_dit_normalize(const Mm3RefinerDit * m, const float * dav, int64_t L) {
    std::vector<float> x((size_t) (L * MM3_DIT_LATENT));
    for (int64_t j = 0; j < L; j++) {
        for (int c = 0; c < MM3_DIT_LATENT; c++) {
            x[(size_t) (j * MM3_DIT_LATENT + c)] = (dav[c * L + j] - m->latent_mean[c]) / m->latent_std[c];
        }
    }
    return x;
}

// Bridge output [L][128] -> DAV latents [128][L].
static std::vector<float> mm3_dit_denormalize(const Mm3RefinerDit * m, const float * x, int64_t L) {
    std::vector<float> dav((size_t) (L * MM3_DIT_LATENT));
    for (int64_t j = 0; j < L; j++) {
        for (int c = 0; c < MM3_DIT_LATENT; c++) {
            dav[(size_t) (c * L + j)] = x[j * MM3_DIT_LATENT + c] * m->latent_std[c] + m->latent_mean[c];
        }
    }
    return dav;
}

namespace mm3_dit_detail {

static ggml_tensor * lin(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
}

// LayerNorm without affine, then * (1 + scale) + shift.
static ggml_tensor * modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, MM3_DIT_EPS), ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
}

// One RefinerBlock on h [D, 2L]; lc [D, L] is layer_cond_proj of this block's MERT layer.
static ggml_tensor * block(ggml_context * ctx, const Mm3DitBlock & B, ggml_tensor * h, ggml_tensor * tc,
                           ggml_tensor * lc, ggml_tensor * pos) {
    const int64_t D = MM3_DIT_DIM, HD = D / MM3_DIT_HEADS, NH = MM3_DIT_HEADS, T = h->ne[1];
    h                 = ggml_add(ctx, h, ggml_concat(ctx, lc, lc, 1));
    ggml_tensor * ada = lin(ctx, B.ada_w, B.ada_b, tc);  // [6D]: shift_a scale_a gate_a shift_m scale_m gate_m
    auto          chunk = [&](int i) { return ggml_view_1d(ctx, ada, D, (size_t) i * D * sizeof(float)); };

    ggml_tensor * qkv = lin(ctx, B.qkv_w, nullptr, modulate(ctx, h, chunk(0), chunk(1)));  // [3D, T]: [3][head][hd]
    auto          part = [&](int i) {
        return ggml_cont(ctx, ggml_view_3d(ctx, qkv, HD, NH, T, HD * sizeof(float), qkv->nb[1], (size_t) i * D * sizeof(float)));
    };
    ggml_tensor * q = ggml_rope_ext(ctx, part(0), pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k = ggml_rope_ext(ctx, part(1), pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q               = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));        // [hd, T, head]
    k               = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v = ggml_cont(ctx, ggml_permute(ctx, part(2), 1, 2, 0, 3));  // [T, hd, head]
    // ponytail: dense [2L, 2L, head] F32 scores (1.3 GB for a 30 s window, quadratic);
    // query-blocked or flash attention once direct mode feeds whole songs.
    ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, q), nullptr, 1.0f / std::sqrt((float) HD), 0.0f);
    ggml_tensor * a  = ggml_cont_2d(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, v, kq), 0, 2, 1, 3), D, T);
    h                = ggml_add(ctx, h, ggml_mul(ctx, lin(ctx, B.out_w, nullptr, a), chunk(2)));

    ggml_tensor * f = ggml_silu(ctx, lin(ctx, B.up_w, B.up_b, modulate(ctx, h, chunk(3), chunk(4))));
    return ggml_add(ctx, h, ggml_mul(ctx, lin(ctx, B.down_w, B.down_b, f), chunk(5)));
}

}  // namespace mm3_dit_detail

// Called after Euler step k (0-based) with the new state and the velocity that
// produced it, both [L][128]. Return false to cancel.
using Mm3DitStepFn = std::function<bool(int step, const float * state, const float * velocity)>;

// The deterministic bridge for one window.
//   x0     normalised DAV latents [L][128] (mm3_dit_normalize); also the context
//   layers all 13 normalised MERT states [13][L][768] (Mm3MertResult::normalized)
//   style  CLAP vector [512]
//   out    receives the final normalised state [L][128] (mm3_dit_denormalize next)
static bool mm3_dit_bridge(Mm3RefinerDit * m, const float * x0, const float * layers, const float * style, int64_t L,
                           int steps, std::vector<float> * out, const Mm3DitStepFn & on_step, std::string * err) {
    using namespace mm3_dit_detail;
    if (L < 1 || steps < 1) {
        *err = "refiner bridge needs at least one frame and one step";
        return false;
    }
    const int64_t        D = MM3_DIT_DIM, T = 2 * L;
    const size_t         bytes = ggml_tensor_overhead() * (MM3_DIT_MAX_NODES + 64) +
                         ggml_graph_overhead_custom(MM3_DIT_MAX_NODES, false);
    std::vector<uint8_t> buf(bytes);
    ggml_init_params     ip  = { bytes, buf.data(), true };
    ggml_context *       ctx = ggml_init(ip);
    struct Free {
        ggml_context * c;
        ~Free() { ggml_free(c); }
    } free_ctx{ ctx };

    // Constants stay resident across steps: inputs flagged as outputs are never
    // freed by the allocator, so they are uploaded once.
    auto input = [&](ggml_tensor * t, bool constant) {
        ggml_set_input(t);
        if (constant) {
            ggml_set_output(t);
        }
        return t;
    };
    ggml_tensor * x     = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MM3_DIT_LATENT, L), false);
    ggml_tensor * tf    = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * MM3_DIT_TFREQ), false);
    ggml_tensor * deg   = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, MM3_DIT_LATENT, L), true);
    ggml_tensor * lay   = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, MM3_DIT_COND, L, MM3_DIT_STATES), true);
    ggml_tensor * sty   = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MM3_DIT_STYLE), true);
    ggml_tensor * pos   = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T), true);
    auto          layer = [&](int s) { return ggml_view_2d(ctx, lay, MM3_DIT_COND, L, lay->nb[1], (size_t) s * lay->nb[2]); };

    ggml_tensor * cond = layer(MM3_DIT_COND_LAYER);
    ggml_tensor * h    = ggml_add(ctx, lin(ctx, m->proj_in_w, m->proj_in_b, ggml_concat(ctx, x, cond, 0)),
                                  lin(ctx, m->deg_w, nullptr, deg));
    ggml_tensor * c    = ggml_add(ctx, lin(ctx, m->proj_in_w, m->proj_in_b, ggml_concat(ctx, deg, cond, 0)), m->ctx_embed);
    h                  = ggml_concat(ctx, c, h, 1);  // [D, 2L], context first

    ggml_tensor * tc = lin(ctx, m->t2_w, m->t2_b, ggml_silu(ctx, lin(ctx, m->t0_w, m->t0_b, tf)));
    tc               = ggml_add(ctx, tc, ggml_view_1d(ctx, m->task_w, D, (size_t) MM3_DIT_TASK * m->task_w->nb[1]));
    tc               = ggml_add(ctx, tc, lin(ctx, m->s2_w, m->s2_b, ggml_silu(ctx, lin(ctx, m->s0_w, m->s0_b, sty))));
    for (int i = 0; i < MM3_DIT_DEPTH; i++) {
        ggml_tensor * lc = lin(ctx, m->block[i].lc_w, nullptr, layer(std::min(1 + i, MM3_DIT_STATES - 2)));
        h                = block(ctx, m->block[i], h, tc, lc, pos);
    }
    ggml_tensor * tgt = ggml_view_2d(ctx, h, D, L, h->nb[1], (size_t) L * h->nb[1]);
    ggml_tensor * vel = lin(ctx, m->out_w, m->out_b, ggml_norm(ctx, tgt, MM3_DIT_EPS));  // [128, L]
    ggml_set_output(vel);

    ggml_cgraph * gr = ggml_new_graph_custom(ctx, MM3_DIT_MAX_NODES, false);
    ggml_build_forward_expand(gr, vel);
    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gr)) {
        *err = "refiner graph allocation failed (out of memory?) for " + std::to_string(L) + " frames";
        return false;
    }
    std::vector<int32_t> positions((size_t) T);
    for (int64_t j = 0; j < T; j++) {
        positions[(size_t) j] = (int32_t) (j % L);
    }
    ggml_backend_tensor_set(deg, x0, 0, ggml_nbytes(deg));
    ggml_backend_tensor_set(lay, layers, 0, ggml_nbytes(lay));
    ggml_backend_tensor_set(sty, style, 0, ggml_nbytes(sty));
    ggml_backend_tensor_set(pos, positions.data(), 0, ggml_nbytes(pos));

    const std::vector<float> sched = mm3_dit_schedule(steps);
    const size_t             n     = (size_t) (L * MM3_DIT_LATENT);
    std::vector<float>       v(n), feat(2 * MM3_DIT_TFREQ);
    out->assign(x0, x0 + n);
    bool ok = true;
    for (int k = 0; k < steps && ok; k++) {
        mm3_dit_time_features(sched[(size_t) k], feat.data());
        ggml_backend_tensor_set(x, out->data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(tf, feat.data(), 0, ggml_nbytes(tf));
        if (ggml_backend_sched_graph_compute(m->sched, gr) != GGML_STATUS_SUCCESS) {
            *err = "refiner graph compute failed";
            ok   = false;
            break;
        }
        if (k == 0) {
            fprintf(stderr, "[MM3-DiT] %lld frames, %d steps, %d nodes, %d splits, compute %.0f MB\n", (long long) L,
                    steps, ggml_graph_n_nodes(gr), ggml_backend_sched_get_n_splits(m->sched),
                    (double) ggml_backend_sched_get_buffer_size(m->sched, m->backend) / (1024.0 * 1024.0));
        }
        ggml_backend_tensor_get(vel, v.data(), 0, n * sizeof(float));
        const float dt = sched[(size_t) k] - sched[(size_t) k + 1];
        for (size_t i = 0; i < n; i++) {
            (*out)[i] = (*out)[i] - dt * v[i];
        }
        if (on_step && !on_step(k, out->data(), v.data())) {
            *err = "cancelled";
            ok   = false;
        }
    }
    ggml_backend_sched_reset(m->sched);
    return ok;
}
