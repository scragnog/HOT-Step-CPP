// yue2-repeat-back-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-repeat-back.patch (the new Metal
// GGML_OP_REPEAT_BACK kernel), kernel #3 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// Unlike kernel #2 (SILU_BACK), this is not a port: neither this project's
// pinned ggml (c044c6f0) nor current upstream ggml-org/llama.cpp implements
// REPEAT_BACK on Metal at all (checked directly against upstream's
// ggml-metal-device.m/ops.cpp/kernels/*.metal -- no matches). The new
// kernel's summation order was deliberately written to match
// ggml-cpu/ops.cpp's ggml_compute_forward_repeat_back_f32 nesting
// (dim3 outermost, dim0 innermost) -- the same order ggml-cuda's own
// k_repeat_back uses too -- so this is pure summation with a well-defined
// order and bit-exact equality (not a tolerance) is the right bar, same as
// Phase 1's ConvRot8 test and this Phase's BF16_ROUND test.
//
// REPEAT_BACK is directly forward-computable (ggml_repeat_back(ctx, a, b)
// sums `a` down to `b`'s shape), so -- like SILU_BACK's test -- this needs
// no backward-through-autodiff graph, just one call per case.
//
// Skips (exit 0, not a failure) when no Metal device is present.
//
// Usage: yue2-repeat-back-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-repeat-back-metal-numeric-test -j
//   ./yue2-repeat-back-metal-numeric-test

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * name;
    int64_t ne0, ne1, ne2, ne3; // destination (small, post-reduction) shape
    int64_t nr0, nr1, nr2, nr3; // repeat factors -- source shape is ne*nr per dim
};

const Case kCases[] = {
    // Matches the real caller exactly: join_halves() in
    // engine/src/train/yue2-aitk-graph.h repeats a [D,heads,S,1] tensor to
    // [2D,heads,S,1] -- dim0 doubled, everything else untouched.
    {"real-usage-dim0x2", 37, 5, 3, 2,  2, 1, 1, 1},
    {"dim0-only-x3",      64, 4, 2, 1,  3, 1, 1, 1},
    {"dim1-only-x4",      16, 6, 2, 1,  1, 4, 1, 1},
    {"all-dims",          11, 4, 3, 2,  2, 3, 2, 2},
    {"large-dim0",        500, 64, 1, 1, 3, 1, 1, 1},
    {"no-op-nr-all-1",    20, 5, 3, 2,  1, 1, 1, 1}, // degenerate: dst == src
};

std::mt19937 rng(0x8EA7BAC1u);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

bool bit_exact(const std::vector<float> & a, const std::vector<float> & b,
                const char * case_name) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "[%s] size mismatch (%zu vs %zu)\n", case_name, a.size(), b.size());
        return false;
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ba, bb;
        std::memcpy(&ba, &a[i], sizeof(ba));
        std::memcpy(&bb, &b[i], sizeof(bb));
        if (ba != bb) {
            if (mismatches < 5) {
                std::fprintf(stderr, "[%s] [%zu]: cpu=%.9g metal=%.9g (bits %08x vs %08x)\n",
                             case_name, i, (double) a[i], (double) b[i], ba, bb);
            }
            ++mismatches;
        }
    }
    if (mismatches) {
        std::fprintf(stderr, "[%s] %zu / %zu elements differ\n", case_name, mismatches, a.size());
        return false;
    }
    return true;
}

bool run_on_backend(ggml_backend_t backend, const Case & c,
                     const std::vector<float> & src_vals,
                     std::vector<float> * output, std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                            c.ne0 * c.nr0, c.ne1 * c.nr1, c.ne2 * c.nr2, c.ne3 * c.nr3);
    ggml_tensor * shape = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne0, c.ne1, c.ne2, c.ne3);
    if (!src || !shape) { *error = "tensor allocation failed"; ggml_free(ctx); return false; }

    ggml_tensor * dst = ggml_repeat_back(ctx, src, shape);
    if (!dst) { *error = "ggml_repeat_back construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, dst);

    if (!ggml_backend_supports_op(backend, dst)) {
        *error = "backend does not report REPEAT_BACK support (is engine/patches/metal-repeat-back.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    // `shape` is only ever read for its ne[]; ggml_backend_alloc_ctx_tensors
    // still allocates real backing storage for it since it has no other
    // producer, but nothing ever reads that storage back -- ggml_repeat_back
    // does not take b as a graph src (see ggml.c: result->src[0] = a only).
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(src, src_vals.data(), 0, src_vals.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        *error = "graph compute failed";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    output->resize((size_t) ggml_nelements(dst));
    ggml_backend_tensor_get(dst, output->data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool run_case(ggml_backend_t cpu, ggml_backend_t metal, const Case & c) {
    const int64_t n_src = c.ne0 * c.nr0 * c.ne1 * c.nr1 * c.ne2 * c.nr2 * c.ne3 * c.nr3;
    const auto src_vals = random_floats((size_t) n_src, -10.0f, 10.0f);

    std::vector<float> cpu_out, metal_out;
    std::string err;

    if (!run_on_backend(cpu, c, src_vals, &cpu_out, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, src_vals, &metal_out, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    const bool ok = bit_exact(cpu_out, metal_out, c.name);
    if (ok) {
        std::printf("[%s] PASS (dst=%lldx%lldx%lldx%lld, nr=%lldx%lldx%lldx%lld, %lld src elems)\n",
                     c.name, (long long) c.ne0, (long long) c.ne1, (long long) c.ne2, (long long) c.ne3,
                     (long long) c.nr0, (long long) c.nr1, (long long) c.nr2, (long long) c.nr3,
                     (long long) n_src);
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-repeat-back-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-repeat-back-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-repeat-back-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-repeat-back-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-repeat-back-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-repeat-back-metal-numeric-test: FAIL");
    return 1;
}
