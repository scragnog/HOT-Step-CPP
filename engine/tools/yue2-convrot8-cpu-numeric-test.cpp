// yue2-convrot8-cpu-numeric-test.cpp — numeric correctness gate for
// engine/patches/zzzz-yue2-convrot8-cpu.patch (the new CPU
// ggml_compute_forward_convrot8{,_back} kernel).
//
// This is the test that actually matters for the CPU port: yue2-aitk-metal-gap
// only checks that ggml_backend_supports_op() now says yes for CPU; it never
// runs the kernel. This tool runs the REAL ggml GGML_OP_CONVROT8 /
// GGML_OP_CONVROT8_BACK ops on the CPU backend and compares their output,
// bit for bit, against tools/yue2-aitk-reference/convrot_cpu.h's
// aitk_reference::convrot8 -- this project's own scalar oracle, which
// test_convrot_ggml_contract.cpp and the Windows/CUDA probes already treat
// as validated against the real CUDA kernel (tools/yue2-aitk-reference/README.md:
// "11 cases pass against Torch 2.9.1+cu128 ... on the 5090"). Matching that
// oracle exactly is therefore a real correctness argument, not just an
// internal consistency check -- it transitively checks the CPU port against
// the CUDA numerics without needing CUDA hardware.
//
// Exact (bit-for-bit) equality is expected, not a tolerance: both sides run
// the identical scalar algorithm, the identical bf16 round-to-nearest-even
// cast, and accumulate the int32 dot product in the identical (ascending-k)
// order, so nothing here should introduce floating-point reordering.
//
// Usage: yue2-convrot8-cpu-numeric-test  (no arguments; exits 0 on PASS)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../tools/yue2-aitk-reference/convrot_cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Case {
    const char * name;
    int64_t rows, in, out, rotation;
    bool use_bf16;
    bool with_bias;
};

// Mirrors block()'s actual shapes (yue2-aitk-graph.h) at reduced size, plus
// the rotation=1 (no-op) edge and a case matching test_convrot_ggml_contract's
// own fixture (in=16, out=8, rows=3, rotation=4) for cross-reference.
const Case kCases[] = {
    {"contract-fixture-shape", 3, 16, 8, 4, false, true},
    {"rotation-1-noop",        5, 16, 8, 1, true,  true},
    {"no-bias",                4, 64, 32, 64, true, false},
    {"bf16-off",               6, 64, 128, 64, false, true},
    {"bf16-on-larger",         5, 256, 256, 256, true, true},
    {"single-row",             1, 64, 64, 64, true, true},
};

std::mt19937 rng(0xC0FFEEu);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}
std::vector<int8_t> random_i8(size_t n) {
    std::uniform_int_distribution<int> dist(-127, 127);
    std::vector<int8_t> v(n);
    for (auto & x : v) x = (int8_t) dist(rng);
    return v;
}

bool nearly_equal_bits(const std::vector<float> & a, const std::vector<float> & b,
                        const char * what, const char * case_name) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] %s: size mismatch (%zu vs %zu)\n", case_name, what, a.size(), b.size());
        return false;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ba, bb;
        std::memcpy(&ba, &a[i], sizeof(ba));
        std::memcpy(&bb, &b[i], sizeof(bb));
        if (ba != bb) {
            if (mismatches < 5) {
                std::fprintf(stderr, "[%s] %s[%zu]: oracle=%.9g ggml-cpu=%.9g (bits %08x vs %08x)\n",
                             case_name, what, i, (double) a[i], (double) b[i], ba, bb);
            }
            ++mismatches;
        }
    }
    if (mismatches) {
        std::fprintf(stderr, "[%s] %s: %zu / %zu elements differ\n", case_name, what, mismatches, a.size());
        return false;
    }
    return true;
}

bool run_case(const Case & c) {
    const auto x       = random_floats((size_t) c.rows * c.in, -3.0f, 3.0f);
    const auto weights = random_i8((size_t) c.out * c.in);
    const auto scales  = random_floats((size_t) c.out, 0.001f, 2.0f);
    const auto grad    = random_floats((size_t) c.rows * c.out, -1.5f, 1.5f);
    const auto bias    = c.with_bias ? random_floats((size_t) c.out, -1.0f, 1.0f)
                                      : std::vector<float>((size_t) c.out, 0.0f);

    aitk_reference::ConvRotResult oracle;
    try {
        oracle = aitk_reference::convrot8(x, weights, scales, grad, bias,
                                          (int) c.rows, (int) c.in, (int) c.out,
                                          (int) c.rotation, c.use_bf16);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[%s] oracle threw: %s\n", c.name, e.what());
        return false;
    }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { std::fprintf(stderr, "[%s] ggml_init failed\n", c.name); return false; }

    ggml_tensor * weight_i8  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, c.in, c.out);
    ggml_tensor * x_f32      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.in, c.rows);
    ggml_tensor * scales_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out);
    ggml_tensor * bias_f32   = c.with_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out) : nullptr;
    ggml_tensor * dy_f32     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.out, c.rows);

    ggml_tensor * forward = ggml_convrot8(ctx, weight_i8, x_f32, scales_f32, bias_f32, (int) c.rotation, c.use_bf16);
    ggml_tensor * back    = ggml_convrot8_back(ctx, dy_f32, forward);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, forward);
    ggml_build_forward_expand(graph, back);

    // By type, not ggml_backend_cpu_init(): with GGML_BACKEND_DL (the Windows
    // build) the CPU backend is a loaded module, so that symbol is not linkable.
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { std::fprintf(stderr, "[%s] CPU backend init failed\n", c.name); ggml_free(ctx); return false; }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        std::fprintf(stderr, "[%s] tensor allocation failed\n", c.name);
        ggml_backend_free(cpu); ggml_free(ctx); return false;
    }

    ggml_backend_tensor_set(weight_i8, weights.data(), 0, weights.size() * sizeof(int8_t));
    ggml_backend_tensor_set(x_f32, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(scales_f32, scales.data(), 0, scales.size() * sizeof(float));
    if (bias_f32) ggml_backend_tensor_set(bias_f32, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_tensor_set(dy_f32, grad.data(), 0, grad.size() * sizeof(float));

    const bool supported = ggml_backend_supports_op(cpu, forward) && ggml_backend_supports_op(cpu, back);
    if (!supported) {
        std::fprintf(stderr, "[%s] CPU backend does not report ConvRot8 support "
                              "(is engine/patches/zzzz-yue2-convrot8-cpu.patch applied?)\n", c.name);
        ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx); return false;
    }

    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[%s] graph compute failed\n", c.name);
        ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx); return false;
    }

    std::vector<float> ggml_output((size_t) c.rows * c.out);
    std::vector<float> ggml_input_grad((size_t) c.rows * c.in);
    ggml_backend_tensor_get(forward, ggml_output.data(), 0, ggml_output.size() * sizeof(float));
    ggml_backend_tensor_get(back, ggml_input_grad.data(), 0, ggml_input_grad.size() * sizeof(float));

    bool ok = true;
    ok &= nearly_equal_bits(oracle.output, ggml_output, "forward output", c.name);
    ok &= nearly_equal_bits(oracle.input_gradient, ggml_input_grad, "input gradient", c.name);

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(cpu);
    ggml_free(ctx);

    if (ok) std::printf("[%s] PASS (rows=%lld in=%lld out=%lld rotation=%lld bf16=%d bias=%d)\n",
                         c.name, (long long) c.rows, (long long) c.in, (long long) c.out,
                         (long long) c.rotation, c.use_bf16, c.with_bias);
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();
    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(c)) all_ok = false;
    }
    if (all_ok) {
        std::puts("yue2-convrot8-cpu-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-convrot8-cpu-numeric-test: FAIL");
    return 1;
}
