// yue2-rms-norm-back-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-rms-norm-back.patch (the new Metal
// GGML_OP_RMS_NORM_BACK kernel), kernel #4 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// Like kernel #3 (REPEAT_BACK), this is not a port: upstream
// ggml-org/llama.cpp's Metal backend has no rms_norm_back kernel either
// (checked directly against current master's kernels/norm.metal,
// ggml-metal-ops.cpp and ggml-metal-device.m -- no matches).
//
// Unlike every prior kernel in this phase, the CPU reference
// (ggml-cpu/ops.cpp's ggml_compute_forward_rms_norm_back_f32) accumulates
// its two per-row reduction sums (sum_xx = sum(x*x), sum_xdz = sum(x*dz))
// in ggml_float, which is `double` (see ggml-cpu/vec.h). Apple GPUs have no
// double-precision support at all, so the new Metal kernel
// (kernel_rms_norm_back_f32) accumulates in float32 -- the same choice
// ggml-cuda/norm.cu's own rms_norm_back_f32 already makes (its sum_xx/sum_xg
// are plain `float`, reduced via warp_reduce_sum, never double). So a
// bit-exact bar is neither achievable nor the established standard for this
// op; this test uses a tolerance, like SILU_BACK's test, but for a
// different underlying reason (there: transcendental ULPs; here: reduction
// precision).
//
// Tolerance derivation: float32 addition has ~1.2e-7 relative rounding
// error per step; a sequential accumulation of N terms has a relative error
// bound that grows with N, but random-sign rounding in practice tracks
// closer to sqrt(N)*eps than N*eps. For this project's realistic row
// lengths (head_dim ~16-128, hidden dim up to ~1024), sqrt(N)*1.2e-7 stays
// under 1e-5, well inside the atol=1e-4 / rtol=1e-3 bound used below even
// after the extra amplification from squaring (sum_xx) and division
// (scale_x, rrms). The "large-hidden-dim" case intentionally pushes N well
// past this project's real usage (2048) as a stress margin, not because
// anything in this project's graphs runs RMS_NORM_BACK that wide.
//
// ggml_rms_norm_back(ctx, grad, x, eps) is directly forward-computable (see
// ggml.c: result = ggml_dup_tensor(ctx, a), src[0] = grad, src[1] = x), so
// -- like SILU_BACK's and REPEAT_BACK's tests -- this needs no
// backward-through-autodiff graph, just one call per case.
//
// Skips (exit 0, not a failure) when no Metal device is present.
//
// Usage: yue2-rms-norm-back-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-rms-norm-back-metal-numeric-test -j
//   ./yue2-rms-norm-back-metal-numeric-test

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
    int64_t ne00, ne01, ne02, ne03; // ne00 = row length (the reduced dim)
    float   eps;
};

const Case kCases[] = {
    // Typical per-head q_norm/k_norm width in this project's LM graphs
    // (yue2-aitk-graph.h / lm-graph.h call ggml_rms_norm with
    // c.rms_norm_eps, default 1e-6f -- see engine/src/train/spike-lm.h).
    {"head-dim-tiny",       16,   4, 1, 1, 1e-6f},
    {"head-dim-typical",    64,   8, 1, 1, 1e-6f},
    {"one-threadgroup-max", 1024, 2, 1, 1, 1e-6f},
    // Awkward (prime) row length -- doesn't divide any SIMD/threadgroup
    // width evenly, exercises the tpitg.x += ntg.x remainder loop.
    {"prime-awkward",       997,  16, 1, 1, 1e-6f},
    // 4D grid dispatch: ne02/ne03 > 1 exercises the tgpig.y / tgpig.z ->
    // nb02/nb03 (grad) and nb12/nb13 (x) offset arithmetic, not just
    // tgpig.x -> nb01/nb11.
    {"multi-dim-grid",      64,   3, 2, 2, 1e-6f},
    // Larger eps -- shifts mean_eps/sum_eps well off of sum_xx, checked
    // separately from the near-zero-x case below (which relies on the
    // *default* small eps to still produce a huge rrms).
    {"large-eps",           128,  4, 1, 1, 1e-2f},
    // Stress margin: pushes N well past anything this project's graphs
    // actually reduce over (see file header), to confirm the tolerance
    // bound holds with room to spare, not because RMS_NORM_BACK runs this
    // wide here.
    {"large-hidden-dim",    2048, 2, 1, 1, 1e-6f},
};

std::mt19937 rng(0xB5A55EEDu);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// x-values plus a few deliberately-tiny magnitudes clustered at the start of
// each row, so sum_xx is dominated by eps rather than by x -- the case CPU's
// mean_eps = sum_xx/N + eps and rrms = 1/sqrt(mean_eps) are guarding against.
std::vector<float> make_x(int64_t ne00, int64_t nrows) {
    std::vector<float> v = random_floats((size_t) (ne00 * nrows), -4.0f, 4.0f);
    for (int64_t r = 0; r < nrows; ++r) {
        float * row = v.data() + r * ne00;
        const int64_t ntiny = std::min<int64_t>(3, ne00);
        for (int64_t i = 0; i < ntiny; ++i) {
            row[i] = (i % 2 == 0 ? 1.0f : -1.0f) * 1e-4f;
        }
    }
    return v;
}

bool nearly_equal(const std::vector<float> & a, const std::vector<float> & b,
                   const char * what, const char * case_name, float atol, float rtol) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] %s: size mismatch (%zu vs %zu)\n", case_name, what, a.size(), b.size());
        return false;
    }
    size_t mismatches = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
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

bool run_on_backend(ggml_backend_t backend, const Case & c,
                     const std::vector<float> & x_vals, const std::vector<float> & grad_vals,
                     std::vector<float> * output, std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * x    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne00, c.ne01, c.ne02, c.ne03);
    ggml_tensor * grad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne00, c.ne01, c.ne02, c.ne03);
    if (!x || !grad) { *error = "tensor allocation failed"; ggml_free(ctx); return false; }

    ggml_tensor * dx = ggml_rms_norm_back(ctx, grad, x, c.eps);
    if (!dx) { *error = "ggml_rms_norm_back construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, dx);

    if (!ggml_backend_supports_op(backend, dx)) {
        *error = "backend does not report RMS_NORM_BACK support (is engine/patches/metal-rms-norm-back.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,    x_vals.data(),    0, x_vals.size()    * sizeof(float));
    ggml_backend_tensor_set(grad, grad_vals.data(), 0, grad_vals.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        *error = "graph compute failed";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    output->resize((size_t) ggml_nelements(dx));
    ggml_backend_tensor_get(dx, output->data(), 0, ggml_nbytes(dx));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool run_case(ggml_backend_t cpu, ggml_backend_t metal, const Case & c) {
    const int64_t nrows = c.ne01 * c.ne02 * c.ne03;
    const auto x_vals    = make_x(c.ne00, nrows);
    const auto grad_vals = random_floats((size_t) (c.ne00 * nrows), -2.0f, 2.0f);

    std::vector<float> cpu_out, metal_out;
    std::string err;

    if (!run_on_backend(cpu, c, x_vals, grad_vals, &cpu_out, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, x_vals, grad_vals, &metal_out, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    // See the file header for the derivation of this bound (float32
    // vs. CPU's double-accumulating reduction, not a transcendental-ULP
    // tolerance like SILU_BACK's).
    const bool ok = nearly_equal(cpu_out, metal_out, "output", c.name, 1e-4f, 1e-3f);
    if (ok) {
        std::printf("[%s] PASS (ne00=%lld rows=%lld eps=%.3g, %lld elems)\n",
                     c.name, (long long) c.ne00, (long long) nrows, (double) c.eps,
                     (long long) (c.ne00 * nrows));
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-rms-norm-back-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-rms-norm-back-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-rms-norm-back-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-rms-norm-back-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-rms-norm-back-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-rms-norm-back-metal-numeric-test: FAIL");
    return 1;
}
