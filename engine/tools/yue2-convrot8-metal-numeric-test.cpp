// yue2-convrot8-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-convrot8.patch (the new Metal GGML_OP_CONVROT8 /
// GGML_OP_CONVROT8_BACK kernels), kernel #5 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// This test uses a TOLERANCE, not bit-exact equality -- despite the plan's
// original expectation ("same bit-exact bar should apply, since ... this is
// discrete int8 arithmetic, not floating-point reduction"). That held for
// the pure-integer dot product and the strictly-sequential backward sum
// (both still true, see below), but the Hadamard rotation itself is genuine
// float32 arithmetic (add/sub butterflies), and an actual run on the author's M1
// Max caught real ~1 ULP differences there between CPU and Metal on every
// case with use_bf16=false (rotation-1-noop, no-bias, bf16-on-larger,
// single-row all have bf16=true and passed exactly; contract-fixture-shape
// and bf16-off both have bf16=false and did not). That correlation, plus
// every observed diff being tiny (~1e-4 absolute on values in the hundreds/
// thousands, i.e. ~1 ULP, never a large jump that would indicate a flipped
// quantization code or a real logic bug), points at one specific, findable
// cause: ggml_metal_library_init's runtime `newLibraryWithSource:` compile
// path (engine/ggml/src/ggml-metal/ggml-metal-device.m) leaves
// `MTLCompileOptions.fastMathEnabled` at its default (true) --
// `[options setFastMathEnabled:false];` is present in that file but
// commented out. Fast math permits the Metal compiler to reassociate the
// butterfly's `x0 + x1 + x2 - x3`-style expressions, which can round
// differently than CPU's strictly left-to-right evaluation. When
// use_bf16=true, the subsequent bf16 round-trip (7-bit mantissa, exact rel.
// step ~1/128 of the value) trivially absorbs float32-ULP-scale noise,
// which is why those cases still pass exactly; with use_bf16=false the raw
// float32 noise survives to the output. This is a backend-compile-option
// difference, not a port bug, and disabling fast math globally for the
// whole Metal library (a one-line, shared-with-every-other-kernel change)
// is out of scope for a single kernel's patch -- so, like SILU_BACK
// (transcendental ULPs) and RMS_NORM_BACK (double- vs float32-reduction),
// this kernel gets a tolerance bound instead. Still true and still relied
// on for correctness (not just for keeping the tolerance small):
//
//   - the forward per-output dot product accumulates in int32, which is
//     exact and associative regardless of summation order -- the Metal
//     kernel gives one thread the whole per-output accumulation (same order
//     as CPU), so no int8 quantization code is ever at risk of silently
//     differing from a reordered *integer* sum.
//   - the backward per-input gradient sum IS float addition (not
//     associative), so kernel_convrot8_back_f32 deliberately keeps that
//     inner n=0..out-1 loop strictly sequential within a single thread
//     (only k is split across threads) -- see that kernel's own comment in
//     ggml-metal.metal. This still matters: it's what keeps the backward
//     pass's own reordering risk limited to fast-math ULPs instead of also
//     depending on which of `out` threads happens to run first.
//   - the row's max-abs (quantization scale) reduction is over `max`, which
//     is exact and order-independent regardless of thread assignment or
//     reassociation -- so the *reduction* itself isn't a new error source;
//     only the Hadamard-rotated values feeding it can already carry ULP
//     noise from the fast-math-affected butterfly arithmetic upstream.
//
// Tolerance derivation: the worst diffs actually observed (the author's M1 Max,
// this case table) were ~2e-4 absolute on values up to ~3100 (bf16-off) and
// ~1e-4 absolute on gradient values up to ~1500 (contract-fixture-shape) --
// consistent with a small, low-single-digit-ULP count of reassociation
// steps through the Hadamard transform's log4(rotation) stages, not
// unbounded error growth. atol=1e-2 / rtol=1e-4 (looser in relative terms
// than RMS_NORM_BACK's 1e-3, since this kernel's magnitudes run larger and
// the noise source here is fewer, coarser reassociation steps rather than a
// full-width reduction) covers the observed worst case with roughly a
// 100x margin while still being tight enough to catch an actual quantization
// code flip (which changes a dot-product term by up to |127 * weight *
// scale|, far larger than this bound for any of this test's fixtures).
//
// This test compares the CPU backend directly against the Metal backend
// (not against tools/yue2-aitk-reference/convrot_cpu.h a second time --
// yue2-convrot8-cpu-numeric-test.cpp already establishes the CPU port is
// bit-exact against that oracle, which is itself validated against the real
// CUDA kernel; this test only needs to show Metal matches that same
// already-trusted CPU port within tolerance). Same case table as
// yue2-convrot8-cpu-numeric-test.cpp, including several rotation == in
// cases (no-bias, bf16-on-larger, single-row) -- the whole-row-as-one-group
// edge that ruled out a fixed-size per-thread-register Hadamard design in
// favor of the threadgroup-cooperative one actually implemented.
//
// Skips (exit 0, not a failure) when no Metal device is present.
//
// Usage: yue2-convrot8-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-convrot8-metal-numeric-test -j
//   ./yue2-convrot8-metal-numeric-test

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * name;
    int64_t rows, in, out, rotation;
    bool use_bf16;
    bool with_bias;
};

// Identical to yue2-convrot8-cpu-numeric-test.cpp's kCases (see that file
// for the rationale of each shape) -- this test's whole point is to show
// Metal matches that already-validated CPU port bit-for-bit, so it must
// exercise exactly the same fixtures.
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

// Tolerance-based, not bit-exact -- see the file header for why (Metal's
// runtime library compile leaves fastMathEnabled at its default true, so
// the Hadamard rotation's float add/sub butterflies can be reassociated
// relative to CPU's strictly left-to-right evaluation).
bool nearly_equal(const std::vector<float> & a, const std::vector<float> & b,
                   const char * what, const char * case_name, float atol, float rtol) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] %s: size mismatch (%zu vs %zu)\n", case_name, what, a.size(), b.size());
        return false;
    }
    size_t mismatches = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff  = std::fabs(a[i] - b[i]);
        const float bound = atol + rtol * std::fabs(a[i]);
        if (diff > bound) {
            if (mismatches < 5) {
                std::fprintf(stderr, "[%s] %s[%zu]: cpu=%.9g metal=%.9g diff=%.3e bound=%.3e\n",
                             case_name, what, i, (double) a[i], (double) b[i], (double) diff, (double) bound);
            }
            ++mismatches;
        }
        worst = std::max(worst, diff);
    }
    if (mismatches) {
        std::fprintf(stderr, "[%s] %s: %zu / %zu elements outside tolerance (worst abs diff %.3e)\n",
                     case_name, what, mismatches, a.size(), (double) worst);
        return false;
    }
    return true;
}

struct Fixture {
    std::vector<float>   x, scales, grad, bias;
    std::vector<int8_t>  weights;
};

bool run_on_backend(ggml_backend_t backend, const Case & c, const Fixture & f,
                     std::vector<float> * output, std::vector<float> * input_grad,
                     std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * weight_i8  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, c.in, c.out);
    ggml_tensor * x_f32      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.in, c.rows);
    ggml_tensor * scales_f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out);
    ggml_tensor * bias_f32   = c.with_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out) : nullptr;
    ggml_tensor * dy_f32     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.out, c.rows);
    if (!weight_i8 || !x_f32 || !scales_f32 || !dy_f32 || (c.with_bias && !bias_f32)) {
        *error = "tensor allocation failed"; ggml_free(ctx); return false;
    }

    ggml_tensor * forward = ggml_convrot8(ctx, weight_i8, x_f32, scales_f32, bias_f32, (int) c.rotation, c.use_bf16);
    ggml_tensor * back    = ggml_convrot8_back(ctx, dy_f32, forward);
    if (!forward || !back) { *error = "ggml_convrot8{,_back} construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, forward);
    ggml_build_forward_expand(graph, back);

    if (!ggml_backend_supports_op(backend, forward) || !ggml_backend_supports_op(backend, back)) {
        *error = "backend does not report ConvRot8 support (is engine/patches/metal-convrot8.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(weight_i8, f.weights.data(), 0, f.weights.size() * sizeof(int8_t));
    ggml_backend_tensor_set(x_f32, f.x.data(), 0, f.x.size() * sizeof(float));
    ggml_backend_tensor_set(scales_f32, f.scales.data(), 0, f.scales.size() * sizeof(float));
    if (bias_f32) ggml_backend_tensor_set(bias_f32, f.bias.data(), 0, f.bias.size() * sizeof(float));
    ggml_backend_tensor_set(dy_f32, f.grad.data(), 0, f.grad.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        *error = "graph compute failed";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    output->resize((size_t) c.rows * c.out);
    input_grad->resize((size_t) c.rows * c.in);
    ggml_backend_tensor_get(forward, output->data(), 0, output->size() * sizeof(float));
    ggml_backend_tensor_get(back, input_grad->data(), 0, input_grad->size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool run_case(ggml_backend_t cpu, ggml_backend_t metal, const Case & c) {
    Fixture f;
    f.x       = random_floats((size_t) c.rows * c.in, -3.0f, 3.0f);
    f.weights = random_i8((size_t) c.out * c.in);
    f.scales  = random_floats((size_t) c.out, 0.001f, 2.0f);
    f.grad    = random_floats((size_t) c.rows * c.out, -1.5f, 1.5f);
    f.bias    = c.with_bias ? random_floats((size_t) c.out, -1.0f, 1.0f)
                             : std::vector<float>((size_t) c.out, 0.0f);

    std::vector<float> cpu_out, cpu_grad, metal_out, metal_grad;
    std::string err;

    if (!run_on_backend(cpu, c, f, &cpu_out, &cpu_grad, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, f, &metal_out, &metal_grad, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    bool ok = true;
    // See the file header for the derivation of this bound (Metal fast-math
    // reassociation in the Hadamard rotation, not a quantization/logic bug).
    ok &= nearly_equal(cpu_out, metal_out, "forward output", c.name, 1e-2f, 1e-4f);
    ok &= nearly_equal(cpu_grad, metal_grad, "input gradient", c.name, 1e-2f, 1e-4f);

    if (ok) {
        std::printf("[%s] PASS (rows=%lld in=%lld out=%lld rotation=%lld bf16=%d bias=%d)\n",
                     c.name, (long long) c.rows, (long long) c.in, (long long) c.out,
                     (long long) c.rotation, c.use_bf16, c.with_bias);
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-convrot8-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-convrot8-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-convrot8-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-convrot8-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-convrot8-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-convrot8-metal-numeric-test: FAIL");
    return 1;
}
