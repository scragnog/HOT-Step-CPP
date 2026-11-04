#pragma once
#include "rvq.h"

namespace heartcodec {

// Flow tensors use [channels,time,batch]; convolution tensors use [time,channels,batch].
inline ggml_tensor * conv(ggml_context * c, const Weights & w, const std::string & p,
                          ggml_tensor * x, int pad, int dilation = 1) {
    auto kernel = w.get(p + ".weight");
    require(kernel->ne[1] == x->ne[1], p + ": convolution width mismatch");
    auto col = ggml_im2col(c, kernel, x, 1, 0, pad, 0, dilation, 0, false, GGML_TYPE_F32);
    const int64_t time = col->ne[1], batch = col->ne[2], channels = kernel->ne[2];
    // Metal multiplies F32 by an F16 src1 only in its matrix kernel (more than 8
    // rows, 64+ wide); smaller kernels go up to F32 first (exact, GH #329).
    if (kernel->type != GGML_TYPE_F32 && (channels <= 8 || kernel->ne[0] * kernel->ne[1] < 64))
        kernel = ggml_cast(c, kernel, GGML_TYPE_F32);
    auto y = ggml_mul_mat(c, ggml_reshape_2d(c, col, col->ne[0], time * batch),
                         ggml_reshape_2d(c, kernel, kernel->ne[0] * kernel->ne[1], channels));
    y = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, y, time, batch, channels), 0, 2, 1, 3));
    if (auto b = w.optional(p + ".bias")) y = ggml_add(c, y, ggml_reshape_3d(c, b, 1, channels, 1));
    return y;
}

inline ggml_tensor * project(ggml_context * c, const Weights & w, const std::string & p, ggml_tensor * x) {
    x = ggml_cont(c, ggml_transpose(c, x));
    x = conv(c, w, p + ".ffn_1", x, 1);
    x = ggml_cont(c, ggml_transpose(c, x));
    x = ggml_scale(c, x, float(1.0 / std::sqrt(3.0)));
    return linear(c, w, p + ".ffn_2", x);
}

inline ggml_tensor * timestep_embedding(Graph & g, const Weights & w, const std::string & prefix,
                                       const std::vector<float> & timesteps) {
    std::vector<float> emb(timesteps.size() * 512);
    for (size_t b = 0; b < timesteps.size(); ++b) {
        for (int k = 0; k < 256; ++k) {
            float freq = std::exp(-std::log(10000.0f) * float(k) / 256.0f);
            float angle = timesteps[b] * freq * 1000.0f;
            emb[b * 512 + k] = std::cos(angle);
            emb[b * 512 + k + 256] = std::sin(angle);
        }
    }
    auto x = g.input(GGML_TYPE_F32, {512, int64_t(timesteps.size())}, emb.data());
    x = linear(g.ctx, w, prefix + ".emb.timestep_embedder.linear_1", x);
    x = ggml_silu(g.ctx, x);
    return linear(g.ctx, w, prefix + ".emb.timestep_embedder.linear_2", x);
}

inline ggml_tensor * attention(Graph & g, const Weights & w, const std::string & p, ggml_tensor * x,
                               ggml_tensor * positions) {
    auto c = g.ctx;
    const int64_t dim = x->ne[0], time = x->ne[1], batch = x->ne[2], heads = 24, hd = dim / heads;
    require(dim == 1536 || dim == 3072, p + ": unsupported attention width");
    auto q = ggml_reshape_4d(c, linear(c, w, p + ".q_proj", x), hd, heads, time, batch);
    auto k = ggml_reshape_4d(c, linear(c, w, p + ".k_proj", x), hd, heads, time, batch);
    auto v = ggml_reshape_4d(c, linear(c, w, p + ".v_proj", x), hd, heads, time, batch);
    q = ggml_rope(c, q, positions, int(hd), 0);
    k = ggml_rope(c, k, positions, int(hd), 0);
    q = ggml_permute(c, q, 0, 2, 1, 3);
    k = ggml_permute(c, k, 0, 2, 1, 3);
    auto scores = ggml_mul_mat(c, k, q);
    scores = ggml_soft_max(c, ggml_scale(c, scores, 1.0f / std::sqrt(float(hd))));
    v = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
    auto y = ggml_mul_mat(c, v, scores);
    y = ggml_cont(c, ggml_permute(c, y, 0, 2, 1, 3));
    y = ggml_reshape_3d(c, y, dim, time, batch);
    return linear(c, w, p + ".o_proj", y);
}

inline ggml_tensor * mod_row(ggml_context * c, ggml_tensor * mod, int row, int64_t dim, int64_t batch) {
    return ggml_view_3d(c, mod, dim, 1, batch, size_t(dim) * sizeof(float),
                        size_t(dim * 6) * sizeof(float), size_t(row * dim) * sizeof(float));
}

inline ggml_tensor * block(Graph & g, const Weights & w, const std::string & p,
                           ggml_tensor * x, ggml_tensor * mod, ggml_tensor * positions) {
    auto c = g.ctx;
    const auto dim = x->ne[0], batch = x->ne[2];
    auto table = w.get(p + ".scale_shift_table");
    require(table->ne[0] == dim && table->ne[1] == 6, p + ": invalid modulation table");
    auto m = ggml_add(c, ggml_reshape_3d(c, mod, dim, 6, batch), table);
    auto row = [&](int i) { return mod_row(c, m, i, dim, batch); };
    auto norm = [&](ggml_tensor * in, const std::string & np, int shift, int scale) {
        auto y = ggml_mul(c, ggml_rms_norm(c, in, 1e-6f), w.get(p + np + ".weight"));
        return ggml_add(c, ggml_add(c, y, ggml_mul(c, y, row(scale))), row(shift));
    };
    auto h = attention(g, w, p + ".attn", norm(x, ".attn_norm", 0, 1), positions);
    x = ggml_add(c, x, ggml_mul(c, h, row(2)));
    auto n = norm(x, ".mlp_norm", 3, 4);
    h = ggml_mul(c, ggml_silu(c, linear(c, w, p + ".mlp.gate", n)), linear(c, w, p + ".mlp.up", n));
    h = linear(c, w, p + ".mlp.down", h);
    return ggml_add(c, x, ggml_mul(c, h, row(5)));
}

inline ggml_tensor * flow_stage(Graph & g, const Weights & w, const std::string & p, ggml_tensor * x,
                                const std::vector<float> & ts, ggml_tensor * positions, bool second) {
    auto c = g.ctx;
    const std::string suffix = second ? "_2" : "";
    auto emb = timestep_embedding(g, w, p + ".adaln_single" + suffix, ts);
    auto mod = linear(c, w, p + ".adaln_single" + suffix + ".linear", ggml_silu(c, emb));
    for (int i = 0; i < (second ? 6 : 24); ++i)
        x = block(g, w, p + ".transformer_blocks" + suffix + "." + std::to_string(i), x, mod, positions);
    const int64_t dim = x->ne[0], batch = x->ne[2];
    auto table = w.get(p + ".scale_shift_table" + suffix);
    auto shift = ggml_view_1d(c, table, dim, 0);
    auto scale = ggml_view_1d(c, table, dim, size_t(dim) * sizeof(float));
    auto embedding = ggml_reshape_3d(c, emb, dim, 1, batch);
    auto normed = ggml_norm(c, x, 1e-6f);
    auto scaling = ggml_add(c, embedding, scale);
    return ggml_add(c, ggml_add(c, normed, ggml_mul(c, normed, scaling)), ggml_add(c, embedding, shift));
}

inline std::vector<float> flow_velocity(const Weights & w, ggml_backend_t backend,
                                       const std::vector<float> & input, int64_t frames,
                                       const std::vector<float> & timesteps) {
    const int64_t batch = int64_t(timesteps.size());
    require(frames > 0 && batch > 0 && input.size() == size_t(batch * frames * 1024), "invalid flow input shape");
    Graph g(backend, 8192);  // 30 blocks of ~60 nodes plus projections
    std::vector<int32_t> pos(size_t(frames), 0);
    for (int64_t i = 0; i < frames; ++i) pos[size_t(i)] = int32_t(i);
    auto positions = g.input(GGML_TYPE_I32, {frames}, pos.data());
    auto in = g.input(GGML_TYPE_F32, {1024, frames, batch}, input.data());
    const std::string p = "flow_matching.estimator";
    auto x = project(g.ctx, w, p + ".proj_in", in);
    x = flow_stage(g, w, p, x, timesteps, positions, false);
    x = ggml_concat(g.ctx, in, x, 0);
    x = project(g.ctx, w, p + ".connection_proj", x);
    x = flow_stage(g, w, p, x, timesteps, positions, true);
    return g.run(project(g.ctx, w, p + ".proj_out", x));
}

using Cancel = std::function<bool()>;
using Velocity = std::function<std::vector<float>(const std::vector<float> &, int64_t, const std::vector<float> &)>;

// Supplied noise isolates solver and chunk parity from random generator differences.
inline std::vector<float> solve_euler(const Velocity & velocity, const std::vector<float> & noise,
                                      const std::vector<float> & context, const std::vector<float> & mu,
                                      int64_t frames, int64_t overlap, int steps, float guidance,
                                      const Cancel & cancelled = {}) {
    require(frames > 0 && steps > 0 && overlap >= 0 && overlap <= frames && std::isfinite(guidance), "invalid solver parameters");
    require(noise.size() == size_t(frames * 256) && context.size() == noise.size() &&
            mu.size() == size_t(frames * 512), "invalid solver input shape");
    auto x = noise;
    float t = 0, dt = 1.0f / float(steps);
    const int batch = guidance > 1 ? 2 : 1;
    for (int s = 1; s <= steps; ++s) {
        check_cancel(cancelled);
        for (int64_t i = 0; i < overlap * 256; ++i)
            x[size_t(i)] = (1.0f - (1.0f - 1e-6f) * t) * noise[size_t(i)] + t * context[size_t(i)];
        std::vector<float> input(size_t(batch * frames * 1024), 0);
        for (int b = 0; b < batch; ++b) for (int64_t f = 0; f < frames; ++f) {
            auto dst = input.data() + (b * frames + f) * 1024;
            std::copy_n(x.data() + f * 256, 256, dst);
            if (f < overlap) std::copy_n(context.data() + f * 256, 256, dst + 256);
            if (batch == 1 || b == 1) std::copy_n(mu.data() + f * 512, 512, dst + 512);
        }
        auto v = velocity(input, frames, std::vector<float>(size_t(batch), t));
        require(v.size() == size_t(batch) * x.size(), "invalid velocity output shape");
        for (size_t i = 0; i < x.size(); ++i) {
            float d = batch == 2 ? v[i] + guidance * (v[x.size() + i] - v[i]) : v[i];
            x[i] += dt * d;
        }
        t += dt;
        // Match torch.linspace and the reference's correction after each step.
        if (s < steps) {
            float next = s + 1 <= steps / 2 ? float(s + 1) / float(steps) :
                         1.0f - float(steps - s - 1) / float(steps);
            dt = next - t;
        }
    }
    std::copy_n(context.begin(), size_t(overlap * 256), x.begin());
    return x;
}

inline std::vector<float> condition(const Weights & flow, ggml_backend_t backend,
                                    const std::vector<float> & rvq, int64_t frames) {
    require(frames > 0 && rvq.size() == size_t(frames * 512), "invalid RVQ feature shape");
    Graph g(backend, 64);
    auto x = g.input(GGML_TYPE_F32, {512, frames}, rvq.data());
    auto projected = g.run(linear(g.ctx, flow, "flow_matching.cond_feature_emb", x));
    std::vector<float> out(size_t(frames * 2 * 512));
    for (int64_t t = 0; t < frames * 2; ++t)
        std::copy_n(projected.data() + (t / 2) * 512, 512, out.data() + t * 512);
    return out;
}

} // namespace heartcodec
