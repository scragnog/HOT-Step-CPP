// yue2-silu-back-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-silu-back.patch (the new Metal GGML_OP_SILU_BACK
// kernel, ported from upstream ggml-org/llama.cpp), kernel #2 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// Unlike kernel #1 (BF16_ROUND, a bit-exact cast), SILU_BACK is a real
// floating-point computation -- dx = dy * s * (1 + x*(1 - s)), s =
// sigmoid(x) -- that both sides evaluate through a transcendental exp().
// ggml-cpu/vec.h's ggml_silu_backward_f32 uses libm's expf(); the ported
// Metal kernel (ggml-metal.metal's kernel_silu_back_f32) uses MSL's exp() on
// the GPU, which is not guaranteed bit-identical to libm even though both
// are correctly-rounded-ish IEEE-754 float. So this test uses a tolerance,
// not the bit-exact bar Phase 1's ConvRot8 test and this Phase's own
// BF16_ROUND test use -- matching how Phase 1's joint-loss test (the other
// precedent involving exp()/log()) was scoped.
//
// Also unlike BF16_ROUND, SILU_BACK is directly forward-computable (it's a
// first-class ggml op, not something reached only through autodiff), so
// this test just builds and runs the op once per case -- no backward graph
// needed.
//
// Skips (exit 0, not a failure) when no Metal device is present.
//
// Usage: yue2-silu-back-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-silu-back-metal-numeric-test -j
//   ./yue2-silu-back-metal-numeric-test

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
    int64_t ne0, ne1; // 2D shape: ne0 = row length, ne1 = row count
};

const Case kCases[] = {
    {"tiny",           7,   3},   // 21 elems
    {"one-threadgroup",256, 1},   // exactly a plausible max threadgroup width
    {"medium",         300, 200}, // 60000 elems, multiple threadgroups
    {"large",          997, 500}, // 498500 elems, awkward (prime) row length
};

std::mt19937 rng(0x51105ACu);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// x-values plus deliberate saturation edge cases (sigmoid(x) -> 0 or 1,
// where the (1 - s) or s factor can lose most of its precision).
std::vector<float> make_x(int64_t n) {
    std::vector<float> v = random_floats((size_t) n, -8.0f, 8.0f);
    size_t i = 0;
    auto set = [&](float x) { if (i < v.size()) v[i++] = x; };
    set(0.0f);
    set(-0.0f);
    set(20.0f);   // sigmoid saturates to ~1
    set(-20.0f);  // sigmoid saturates to ~0
    set(1.0f);
    set(-1.0f);
    set(0.0001f); // near the inflection point
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
                     const std::vector<float> & x_vals, const std::vector<float> & dy_vals,
                     std::vector<float> * output, std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
    ggml_tensor * dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
    if (!x || !dy) { *error = "tensor allocation failed"; ggml_free(ctx); return false; }

    ggml_tensor * dx = ggml_silu_back(ctx, dy, x);
    if (!dx) { *error = "ggml_silu_back construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, dx);

    if (!ggml_backend_supports_op(backend, dx)) {
        *error = "backend does not report SILU_BACK support (is engine/patches/metal-silu-back.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,  x_vals.data(),  0, x_vals.size()  * sizeof(float));
    ggml_backend_tensor_set(dy, dy_vals.data(), 0, dy_vals.size() * sizeof(float));

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
    const int64_t n = c.ne0 * c.ne1;
    const auto x_vals  = make_x(n);
    const auto dy_vals = random_floats((size_t) n, -2.0f, 2.0f);

    std::vector<float> cpu_out, metal_out;
    std::string err;

    if (!run_on_backend(cpu, c, x_vals, dy_vals, &cpu_out, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, x_vals, dy_vals, &metal_out, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    // Tolerance: this is a single exp() evaluation feeding a low-order
    // polynomial in x, s -- not a reduction/summation, so error shouldn't
    // accumulate. 1e-4 absolute + 1e-4 relative is generous against libm vs
    // GPU exp() ULP differences while still catching a wrong formula or a
    // src0/src1 (grad/x) argument-order swap, which would show up as
    // differences of order 1 almost everywhere, not a handful of ULPs.
    const bool ok = nearly_equal(cpu_out, metal_out, "output", c.name, 1e-4f, 1e-4f);
    if (ok) {
        std::printf("[%s] PASS (ne0=%lld ne1=%lld, %lld elems)\n",
                     c.name, (long long) c.ne0, (long long) c.ne1, (long long) n);
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-silu-back-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-silu-back-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-silu-back-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-silu-back-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-silu-back-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-silu-back-metal-numeric-test: FAIL");
    return 1;
}
