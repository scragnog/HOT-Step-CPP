#include "../../engine/ggml/include/ggml.h"
#include "../../engine/ggml/include/ggml-alloc.h"
#include "../../engine/ggml/include/ggml-backend.h"
#include "../../engine/ggml/include/ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "../../engine/ggml/include/ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMagic = 0x314B5441u;
constexpr size_t kHeaderBytes = 6 * sizeof(uint32_t);

struct ContextGuard {
    ggml_context * p = nullptr;
    ~ContextGuard() { if (p) ggml_free(p); }
};
struct BackendGuard {
    ggml_backend_t p = nullptr;
    ~BackendGuard() { if (p) ggml_backend_free(p); }
};
struct BufferGuard {
    ggml_backend_buffer_t p = nullptr;
    ~BufferGuard() { if (p) ggml_backend_buffer_free(p); }
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

size_t checked_mul(size_t a, size_t b, const char * what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        fail(std::string("size overflow in ") + what);
    }
    return a * b;
}

void read_exact(std::ifstream & in, void * dst, size_t bytes, const char * what) {
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail(std::string("input is too large for ") + what);
    }
    in.read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) {
        fail(std::string("truncated input while reading ") + what);
    }
}

struct Fixture {
    uint32_t rows = 0;
    uint32_t width = 0;
    uint32_t outputs = 0;
    uint32_t rotation = 0;
    bool bf16 = false;
    std::vector<float> x;
    std::vector<int8_t> weight;
    std::vector<float> scales;
    std::vector<float> dy;
    std::vector<float> bias;
};

Fixture load_fixture(const std::filesystem::path & path) {
    std::error_code ec;
    const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
    if (ec) fail("cannot stat input fixture: " + path.string());
    if (file_bytes < kHeaderBytes) fail("input fixture is shorter than its header");

    std::ifstream in(path, std::ios::binary);
    if (!in) fail("cannot open input fixture: " + path.string());

    uint32_t h[6] = {};
    read_exact(in, h, sizeof(h), "header");
    if (h[0] != kMagic) fail("input fixture has an invalid magic");
    if (h[1] == 0 || h[2] == 0 || h[3] == 0) fail("input fixture has an empty dimension");
    if (h[5] > 1) fail("input fixture has an invalid bf16 flag");
    if (h[1] > 24576 || h[2] > 8192 || h[3] > 184704 ||
        h[2] % 16 || h[3] % 8 || (h[2] < 128 && h[3] % 16)) fail("unsupported projection dimensions");
    uint32_t rotation = h[4];
    if (!rotation || h[2] % rotation) fail("invalid rotation size");
    while (rotation > 1 && rotation % 4 == 0) rotation /= 4;
    if (rotation != 1) fail("rotation size must be a power of four");

    const size_t rows = h[1], width = h[2], outputs = h[3];
    const size_t x_count = checked_mul(rows, width, "x");
    const size_t w_count = checked_mul(outputs, width, "weight");
    const size_t dy_count = checked_mul(rows, outputs, "dy");
    size_t expected = kHeaderBytes;
    expected += checked_mul(x_count, sizeof(float), "x bytes");
    expected += checked_mul(w_count, sizeof(int8_t), "weight bytes");
    expected += checked_mul(outputs, sizeof(float), "scale bytes");
    expected += checked_mul(dy_count, sizeof(float), "dy bytes");
    expected += checked_mul(outputs, sizeof(float), "bias bytes");
    if (file_bytes != expected) fail("input fixture byte size does not match its header");

    Fixture f;
    f.rows = h[1]; f.width = h[2]; f.outputs = h[3];
    f.rotation = h[4]; f.bf16 = h[5] != 0;
    f.x.resize(x_count); f.weight.resize(w_count); f.scales.resize(outputs);
    f.dy.resize(dy_count); f.bias.resize(outputs);
    read_exact(in, f.x.data(), x_count * sizeof(float), "x");
    read_exact(in, f.weight.data(), w_count * sizeof(int8_t), "weight");
    read_exact(in, f.scales.data(), outputs * sizeof(float), "scales");
    read_exact(in, f.dy.data(), dy_count * sizeof(float), "dy");
    read_exact(in, f.bias.data(), outputs * sizeof(float), "bias");
    return f;
}

bool same_path(const std::filesystem::path & a, const std::filesystem::path & b) {
    std::error_code ea, eb;
    const auto ca = std::filesystem::weakly_canonical(a, ea);
    const auto cb = std::filesystem::weakly_canonical(b, eb);
    return !ea && !eb && ca == cb;
}

void write_output(const std::filesystem::path & path, const std::vector<float> & y,
                  const std::vector<float> & dx) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot create output fixture: " + path.string());
    out.write(reinterpret_cast<const char *>(y.data()),
              static_cast<std::streamsize>(y.size() * sizeof(float)));
    out.write(reinterpret_cast<const char *>(dx.data()),
              static_cast<std::streamsize>(dx.size() * sizeof(float)));
    if (!out) fail("failed writing output fixture");
}

void require_support(ggml_backend_t backend, ggml_tensor * op, const char * name,
                     bool expected) {
    if (!backend || ggml_backend_supports_op(backend, op) != expected) {
        fail(std::string("unexpected backend supports_op result for ") + name);
    }
}

int run(const std::filesystem::path & input, const std::filesystem::path & output,
        bool no_bias) {
    if (same_path(input, output)) fail("refusing to overwrite input fixture");
    if (std::filesystem::exists(output)) fail("refusing to overwrite output fixture");
    const Fixture f = load_fixture(input);

    BackendGuard backend;
    backend.p = ggml_backend_cpu_init();
    if (!backend.p) fail("failed to initialize CPU backend");
#ifndef GGML_USE_CUDA
    fail("probe must be compiled with GGML_USE_CUDA");
#else
    BackendGuard cuda;
    cuda.p = ggml_backend_cuda_init(0);
    if (!cuda.p) fail("failed to initialize CUDA backend");
#endif

    ggml_init_params params{};
    params.mem_size = 64u * 1024u * 1024u;
    params.no_alloc = true;
    ContextGuard ctx{ggml_init(params)};
    if (!ctx.p) fail("failed to initialize GGML context");

    ggml_tensor * weight = ggml_new_tensor_2d(ctx.p, GGML_TYPE_I8, f.width, f.outputs);
    ggml_tensor * x = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, f.width, f.rows);
    ggml_tensor * scales = ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, f.outputs);
    ggml_tensor * bias = no_bias ? nullptr : ggml_new_tensor_1d(ctx.p, GGML_TYPE_F32, f.outputs);
    ggml_tensor * dy = ggml_new_tensor_2d(ctx.p, GGML_TYPE_F32, f.outputs, f.rows);
    if (!weight || !x || !scales || !dy || (!no_bias && !bias)) fail("failed creating GGML inputs");
    ggml_set_param(x);

    ggml_tensor * y = ggml_convrot8(ctx.p, weight, x, scales, bias,
                                    static_cast<int32_t>(f.rotation), f.bf16);
    ggml_tensor * dx = ggml_convrot8_back(ctx.p, dy, y);
    if (!y || !dx) fail("ConvRot constructors returned null");
    require_support(backend.p, y, "convrot8 on CPU", false);
    require_support(backend.p, dx, "convrot8_back on CPU", false);
#ifdef GGML_USE_CUDA
    require_support(cuda.p, y, "convrot8 on CUDA", true);
    require_support(cuda.p, dx, "convrot8_back on CUDA", true);
#endif

    ggml_tensor * loss = ggml_sum(ctx.p, ggml_mul(ctx.p, y, dy));
    if (!loss) fail("failed constructing loss");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.p, 4096, true);
    if (!graph) fail("failed creating graph");
    ggml_build_forward_expand(graph, loss);
    ggml_build_forward_expand(graph, dx);
    ggml_set_loss(loss);
    ggml_build_backward_expand(ctx.p, graph, nullptr);

#ifdef GGML_USE_CUDA
    BufferGuard buffer{ggml_backend_alloc_ctx_tensors(ctx.p, cuda.p)};
#else
    BufferGuard buffer{ggml_backend_alloc_ctx_tensors(ctx.p, backend.p)};
#endif
    if (!buffer.p) fail("failed allocating GGML tensors");

    ggml_backend_t compute =
#ifdef GGML_USE_CUDA
        cuda.p;
#else
        backend.p;
#endif
    ggml_backend_tensor_set(weight, f.weight.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(x, f.x.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(scales, f.scales.data(), 0, ggml_nbytes(scales));
    ggml_backend_tensor_set(dy, f.dy.data(), 0, ggml_nbytes(dy));
    if (bias) ggml_backend_tensor_set(bias, f.bias.data(), 0, ggml_nbytes(bias));
    ggml_graph_reset(graph);
    if (ggml_backend_graph_compute(compute, graph) != GGML_STATUS_SUCCESS) {
        fail("GGML backend graph compute failed");
    }

    std::vector<float> y_out(f.rows * f.outputs), dx_out(f.rows * f.width), grad_x(dx_out.size());
    ggml_backend_tensor_get(y, y_out.data(), 0, ggml_nbytes(y));
    ggml_backend_tensor_get(dx, dx_out.data(), 0, ggml_nbytes(dx));
    ggml_tensor * x_grad = ggml_graph_get_grad(graph, x);
    if (!x_grad) fail("autograd did not produce an x gradient");
    ggml_backend_tensor_get(x_grad, grad_x.data(), 0, ggml_nbytes(x_grad));
    for (size_t i = 0; i < grad_x.size(); ++i) {
        if (!std::isfinite(grad_x[i]) || !std::isfinite(dx_out[i]) || grad_x[i] != dx_out[i])
            fail("explicit dx differs from autograd x gradient");
    }
    if (ggml_graph_get_grad(graph, weight) || ggml_graph_get_grad(graph, scales) ||
        (bias && ggml_graph_get_grad(graph, bias))) {
        fail("frozen ConvRot inputs unexpectedly received gradients");
    }
    write_output(output, y_out, dx_out);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3 && argc != 4) {
            std::cerr << "usage: convrot_ggml_probe INPUT OUTPUT [--no-bias]\n";
            return 2;
        }
        const bool no_bias = argc == 4 && std::string(argv[3]) == "--no-bias";
        if (argc == 4 && !no_bias) fail("unknown option");
        return run(argv[1], argv[2], no_bias);
    } catch (const std::exception & e) {
        std::cerr << "convrot_ggml_probe: " << e.what() << "\n";
        return 1;
    }
}
