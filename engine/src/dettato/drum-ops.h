#pragma once
// CPU graph and tensor operations for the Dettato drum models.

#include "../cpu-gru.h"
#include "../safetensors.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <thread>

namespace dettato::drums::detail {

using Data = std::vector<float>;

struct WeightFile {
    STFile file{};
    explicit WeightFile(const char * path) {
        if (!st_open(&file, path)) throw std::runtime_error("cannot open drum weights");
    }
    ~WeightFile() { st_close(&file); }
    WeightFile(const WeightFile &) = delete;
    WeightFile & operator=(const WeightFile &) = delete;

    Data load(const std::string & name, std::initializer_list<int> shape) const {
        const auto * e = st_find(file, name.c_str());
        size_t count = 1;
        bool ok = e && e->dtype == "F32" && e->n_dims == int(shape.size());
        int d = 0;
        for (int dim : shape) {
            ok = ok && e->shape[d++] == dim;
            count *= size_t(dim);
        }
        if (!ok || e->data_end > file.file_size - file.data_offset || e->data_end < e->data_start ||
            e->data_end - e->data_start != count * sizeof(float)) {
            throw std::runtime_error("invalid drum tensor " + name);
        }
        Data data(count);
        std::memcpy(data.data(), st_data(file, *e), count * sizeof(float));
        return data;
    }
};

struct Cpu {
    int threads;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{nullptr, ggml_backend_free};

    explicit Cpu(int n_threads) : threads(std::max(1, n_threads)) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) throw std::runtime_error("no GGML CPU backend");
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!backend) throw std::runtime_error("cannot initialize GGML CPU backend");
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads"));
        if (set_threads) set_threads(backend.get(), threads);
    }

    template <class Fn>
    void parallel(int count, Fn fn) const {
        const int workers = std::min(count, threads);
        std::vector<std::thread> pool;
        for (int i = 1; i < workers; ++i) {
            pool.emplace_back([&, i] { fn(i * count / workers, (i + 1) * count / workers); });
        }
        if (workers) fn(0, count / workers);
        for (auto & worker : pool) worker.join();
    }
};

struct Graph {
    static constexpr int nodes = 32;
    const Cpu & cpu;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};

    explicit Graph(const Cpu & c) :
        cpu(c), context(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false),
                                   nullptr, true}), ggml_free) {
        if (!context) throw std::runtime_error("cannot create drum graph");
    }

    void run(ggml_tensor * out, std::initializer_list<std::pair<ggml_tensor *, const float *>> inputs, float * dst) {
        auto * graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, out);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), cpu.backend.get()));
        if (!buffer) throw std::runtime_error("cannot allocate drum graph tensors");
        for (auto [tensor, src] : inputs) ggml_backend_tensor_set(tensor, src, 0, ggml_nbytes(tensor));
        if (ggml_backend_graph_compute(cpu.backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("drum graph compute failed");
        }
        ggml_backend_tensor_get(out, dst, 0, ggml_nbytes(out));
    }
};

inline float silu(float x) { return x * cpu_gru::sigmoid(x); }

struct Linear {
    int in, out;
    Data w, b;
    Linear(const WeightFile & file, const std::string & name, int ni, int no) :
        in(ni), out(no), w(file.load(name + ".weight", {no, ni})), b(file.load(name + ".bias", {no})) {}
    Linear(const WeightFile & file, const std::string & name, const std::string & suffix, int ni, int no) :
        in(ni), out(no), w(file.load(name + ".weight" + suffix, {no, ni})),
        b(file.load(name + ".bias" + suffix, {no})) {}

    Linear(const WeightFile & file, const std::string & name, int ni, int no, std::initializer_list<int> shape) :
        in(ni), out(no), w(file.load(name + ".weight", shape)), b(file.load(name + ".bias", {no})) {}

    Data operator()(const Cpu & cpu, const float * x, int rows) const {
        Graph graph(cpu);
        auto * ctx = graph.context.get();
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, rows);
        auto * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
        auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
        auto * y = ggml_add(ctx, ggml_mul_mat(ctx, weight, input), bias);
        Data data(size_t(rows) * out);
        graph.run(y, {{input, x}, {weight, w.data()}, {bias, b.data()}}, data.data());
        return data;
    }
};

struct BiGru {
    int hidden;
    Linear ih_f, ih_b, hh_f, hh_b;

    BiGru(const WeightFile & file, const std::string & name, int in, int h, int layer = 0) :
        hidden(h),
        ih_f(file, name, "_ih_l" + std::to_string(layer), in, 3 * h),
        ih_b(file, name, "_ih_l" + std::to_string(layer) + "_reverse", in, 3 * h),
        hh_f(file, name, "_hh_l" + std::to_string(layer), h, 3 * h),
        hh_b(file, name, "_hh_l" + std::to_string(layer) + "_reverse", h, 3 * h) {}

    // Each batch is an independent sequence. The temporal GRUs use batch 1.
    Data operator()(const Cpu & cpu, const Data & x, int length, int batch = 1) const {
        const auto gx_f = ih_f(cpu, x.data(), length * batch);
        const auto gx_b = ih_b(cpu, x.data(), length * batch);
        Data out(size_t(length) * batch * 2 * hidden);
        cpu.parallel(batch * 2, [&](int first, int end) {
            for (int i = first; i < end; ++i) {
                const size_t start = size_t(i / 2) * length;
                const bool reverse = i % 2 != 0;
                const auto & hh = reverse ? hh_b : hh_f;
                const auto & gx = reverse ? gx_b : gx_f;
                cpu_gru::dir(hidden, hh.w.data(), hh.b.data(), gx.data() + start * 3 * hidden, length, reverse,
                             out.data() + start * 2 * hidden, 2 * hidden, reverse ? hidden : 0);
            }
        });
        return out;
    }
};

// [channels, frames, bands], bands contiguous.
struct Image {
    int c, t, f;
    Data data;
    Image(int nc, int nt, int nf) : c(nc), t(nt), f(nf), data(size_t(nc) * nt * nf) {}
    float * row(int ch, int frame) { return data.data() + (size_t(ch) * t + frame) * f; }
    const float * row(int ch, int frame) const { return data.data() + (size_t(ch) * t + frame) * f; }
};

struct Conv {
    int in, out, kt, kf, stride;
    bool depthwise;
    Data w, b;

    Conv(const WeightFile & file, const std::string & name, int ni, int no, int ny, int nx,
         int step = 1, bool dw = false) :
        in(ni), out(no), kt(ny), kf(nx), stride(step), depthwise(dw),
        w(file.load(name + ".weight", {no, dw ? 1 : ni, ny, nx})), b(file.load(name + ".bias", {no})) {}

    void fold_norm(const WeightFile & file, const std::string & name) {
        const auto gain = file.load(name + ".weight", {out});
        const auto bias = file.load(name + ".bias", {out});
        const auto mean = file.load(name + ".running_mean", {out});
        const auto var = file.load(name + ".running_var", {out});
        const size_t per_channel = w.size() / out;
        for (int c = 0; c < out; ++c) {
            const float scale = gain[c] / std::sqrt(var[c] + 1e-5f);
            b[c] = (b[c] - mean[c]) * scale + bias[c];
            for (size_t i = 0; i < per_channel; ++i) w[size_t(c) * per_channel + i] *= scale;
        }
    }

    Image operator()(const Cpu & cpu, const Image & x, bool activate = false) const {
        const int pt = kt / 2, pf = kf / 2;
        Image y(out, (x.t - 1) / stride + 1, x.f);
        if (depthwise) {
            cpu.parallel(out, [&](int lo, int hi) {
                for (int c = lo; c < hi; ++c) {
                    const float * kernel = w.data() + size_t(c) * kt * kf;
                    for (int t = 0; t < y.t; ++t) {
                        float * dst = y.row(c, t);
                        std::fill_n(dst, y.f, b[c]);
                        for (int dy = -pt; dy <= pt; ++dy) {
                            const int at = t * stride + dy;
                            if (at < 0 || at >= x.t) continue;
                            const float * src = x.row(c, at);
                            for (int dx = -pf; dx <= pf; ++dx) {
                                const float k = kernel[(dy + pt) * kf + dx + pf];
                                for (int f = std::max(0, -dx); f < std::min(x.f, x.f - dx); ++f) {
                                    dst[f] += src[f + dx] * k;
                                }
                            }
                        }
                        if (activate) for (int f = 0; f < y.f; ++f) dst[f] = silu(dst[f]);
                    }
                }
            });
            return y;
        }
        // Bound im2col memory. The overlapping input rows keep padding at song
        // edges only, so every tile sees the same receptive field.
        constexpr int tile = 64;
        const int cols = in * kt * kf;
        for (int start = 0; start < y.t; start += tile) {
            const int count = std::min(tile, y.t - start);
            const int raw_start = start * stride - pt;
            const int span = (count - 1) * stride + kt;
            const int copy_start = std::max(0, raw_start);
            const int copy_end = std::min(raw_start + span, x.t);
            Image part(in, span, x.f);
            for (int c = 0; c < in; ++c) {
                std::copy_n(x.row(c, copy_start), size_t(copy_end - copy_start) * x.f,
                            part.row(c, copy_start - raw_start));
            }
            Graph graph(cpu);
            auto * ctx = graph.context.get();
            auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x.f, span, in);
            auto * weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kf, kt, in, out);
            auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out);
            auto * windows = ggml_im2col(ctx, weight, input, 1, stride, pf, 0, 1, 1, true, GGML_TYPE_F32);
            auto * res = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, weight, cols, out),
                                     ggml_reshape_2d(ctx, windows, cols, count * x.f));
            res = ggml_add(ctx, res, bias);
            Data block(size_t(count) * x.f * out);
            graph.run(res, {{input, part.data.data()}, {weight, w.data()}, {bias, b.data()}}, block.data());
            for (int t = 0; t < count; ++t) {
                for (int c = 0; c < out; ++c) {
                    float * dst = y.row(c, start + t);
                    for (int f = 0; f < x.f; ++f) {
                        const float value = block[(size_t(t) * x.f + f) * out + c];
                        dst[f] = activate ? silu(value) : value;
                    }
                }
            }
        }
        return y;
    }
};

inline Image pool(const Image & x, int time, int band) {
    Image y(x.c, x.t / time, x.f / band);
    for (int c = 0; c < x.c; ++c) {
        for (int t = 0; t < y.t; ++t) {
            for (int f = 0; f < y.f; ++f) {
                float v = x.row(c, t * time)[f * band];
                for (int j = 0; j < time; ++j) {
                    for (int i = 0; i < band; ++i) v = std::max(v, x.row(c, t * time + j)[f * band + i]);
                }
                y.row(c, t)[f] = v;
            }
        }
    }
    return y;
}

struct Norm {
    Data weight, bias;
    Norm(const WeightFile & file, const std::string & name, int channels) :
        weight(file.load(name + ".weight", {channels})), bias(file.load(name + ".bias", {channels})) {}

    static std::pair<float, float> stats(const float * x, size_t n) {
        double sum = 0, squared = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += x[i];
            squared += double(x[i]) * x[i];
        }
        const double mean = sum / double(n);
        return {float(mean), float(1.0 / std::sqrt(std::max(0.0, squared / double(n) - mean * mean) + 1e-5))};
    }

    void layer(Data & x, int rows) const {
        const size_t width = weight.size();
        for (int t = 0; t < rows; ++t) {
            float * row = x.data() + size_t(t) * width;
            const auto [mean, scale] = stats(row, width);
            for (size_t i = 0; i < width; ++i) row[i] = (row[i] - mean) * scale * weight[i] + bias[i];
        }
    }

    void group_silu(const Cpu & cpu, Image & x, int groups) const {
        const int channels = x.c / groups;
        const size_t area = size_t(x.t) * x.f;
        cpu.parallel(groups, [&](int lo, int hi) {
            for (int g = lo; g < hi; ++g) {
                const auto [mean, scale] = stats(x.row(g * channels, 0), channels * area);
                for (int c = g * channels; c < (g + 1) * channels; ++c) {
                    float * plane = x.row(c, 0);
                    for (size_t i = 0; i < area; ++i) plane[i] = silu((plane[i] - mean) * scale * weight[c] + bias[c]);
                }
            }
        });
    }
};

inline Data attention(const Cpu & cpu, const Data & x, int frames, const Linear & qkv, const Linear & proj,
                      int heads) {
    const auto packed = qkv(cpu, x.data(), frames);
    const int width = qkv.out / 3;
    const int lane = width / heads;
    const size_t elements = size_t(frames) * width;
    Data q(elements), k(elements), v(elements);
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < frames; ++t) {
            const size_t offset = (size_t(h) * frames + t) * lane;
            const float * row = packed.data() + size_t(t) * 3 * width + h * lane;
            std::copy_n(row, lane, q.data() + offset);
            std::copy_n(row + width, lane, k.data() + offset);
            std::copy_n(row + 2 * width, lane, v.data() + offset);
        }
    }
    Graph graph(cpu);
    auto * ctx = graph.context.get();
    auto * qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, lane, frames, heads);
    auto * kt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, lane, frames, heads);
    auto * vt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, lane, frames, heads);
    auto * y = ggml_flash_attn_ext(ctx, qt, kt, vt, nullptr, 1.0f / std::sqrt(float(lane)), 0, 0);
    // Flash attention yields [head_dim, heads, frames], already packed per frame.
    Data values(elements);
    graph.run(y, {{qt, q.data()}, {kt, k.data()}, {vt, v.data()}}, values.data());
    return proj(cpu, values.data(), frames);
}

inline void add(Data & x, const Data & y) {
    for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}

}  // namespace dettato::drums::detail
