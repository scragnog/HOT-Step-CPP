// yue2-out-prod-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-out-prod.patch (the new Metal GGML_OP_OUT_PROD
// kernel, F32-only), kernel #6 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// OUT_PROD is a standard ggml op (not one of this project's own, unlike
// CONVROT8/RMS_NORM_BACK) -- the 16 nodes yue2-aitk-metal-gap reports come
// from ggml_build_backward_expand's own mul_mat activation-gradient formula
// hitting the AR block's expert-adapter (LoRA) matmuls. Per the plan,
// yue2_aitk_graph::linear()'s own comment confirms "FP32 trainables and
// FP32 LoRA math", so this port only needs the F32xF32->F32 path -- the CPU
// reference (ggml-cpu/ops.cpp) also has quantized-src0 and F16-src0
// variants this project's graph never produces, and neither the Metal
// kernel nor this test attempt those.
//
// This test expects EXACT (bit-for-bit) equality, not a tolerance. Unlike
// kernel #5 (CONVROT8), whose Hadamard rotation is a fixed multi-term
// expression (x0+x1+x2-x3) that Metal's fast-math-enabled runtime compile
// could reassociate, this kernel's only floating-point operation is a
// single accumulator loop per output element (sum += src0[...] *
// src1[...], strictly ascending k=0..ne01-1, one thread owns the whole
// reduction for its own output element -- see kernel_out_prod_f32's own
// comment in ggml-metal.metal for why this matches
// ggml_compute_forward_out_prod_f32's own accumulation order exactly,
// despite that CPU function's cache-blocking making the loop nest look
// different). This is the same shape of computation as kernel #3
// (REPEAT_BACK), whose own accumulation loop stayed bit-exact under this
// project's fast-math-enabled Metal compile (see metal-repeat-back.patch's
// own test) -- so a bit-exact bar is attempted here first, consistent with
// that precedent, rather than assuming a tolerance is needed the way
// kernel #5 turned out to require one. If a real run on Metal hardware
// shows otherwise, this test should be revisited the same empirical way
// kernel #5's was (see yue2-convrot8-metal-numeric-test.cpp's header).
//
// Compares the CPU backend directly against the Metal backend, since
// OUT_PROD is upstream ggml functionality with no project-specific oracle
// to validate the CPU side against separately.
//
// Skips (exit 0, not a failure) when no Metal device is present.
//
// Usage: yue2-out-prod-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-out-prod-metal-numeric-test -j
//   ./yue2-out-prod-metal-numeric-test

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
    // src0: [ne00, ne01, ne02, ne03]; src1: [ne10, ne01, ne2, ne3]
    // (src1's own ne[1] must equal src0's ne01 -- ggml_can_out_prod's
    // contract); dst: [ne00, ne10, ne2, ne3].
    int64_t ne00, ne01, ne02, ne03;
    int64_t ne10, ne2, ne3;
};

const Case kCases[] = {
    // Small, no broadcast (ne02==ne2, ne03==ne3): the plain LoRA-matmul-
    // gradient shape (out_prod(weight, transpose(grad))).
    {"small-no-broadcast",   16, 8, 1, 1,   12, 1, 1},
    // Awkward (prime) reduction dim -- doesn't divide any thread/simdgroup
    // width evenly, exercises the k-loop's own bound directly (this kernel
    // has no tpitg-style remainder concern since k is a plain scalar loop,
    // but ne00 not dividing threads-per-threadgroup evenly is still worth
    // covering).
    {"prime-reduction-dim",  23, 37, 1, 1,   19, 1, 1},
    // GQA-style broadcast on dim2: src0 has 2 "heads", dst/src1 have 6 ->
    // dps2 = 3, exercises the i02 = i2/dps2 mapping.
    {"broadcast-dim2",       16, 8, 2, 1,   12, 6, 1},
    // Broadcast on both dim2 and dim3.
    {"broadcast-dim2-dim3",  16, 8, 2, 2,   12, 6, 4},
    // Larger, closer to a real hidden-dim x LoRA-rank shape (rank=8,
    // hidden=256) -- stress margin, not a claim this project runs OUT_PROD
    // this wide in the AR block specifically.
    {"lora-rank-shape",     256, 8, 1, 1,  256, 1, 1},
    // Single reduction step (ne01=1) -- degenerate case, no real
    // "accumulation order" to get wrong, useful as a sanity floor.
    {"single-k",              8, 1, 1, 1,    6, 1, 1},
};

std::mt19937 rng(0xACED5EEDu);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

bool exactly_equal(const std::vector<float> & a, const std::vector<float> & b,
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
                std::fprintf(stderr, "[%s] %s[%zu]: cpu=%.9g metal=%.9g (bits %08x vs %08x)\n",
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

bool run_on_backend(ggml_backend_t backend, const Case & c,
                     const std::vector<float> & src0_vals, const std::vector<float> & src1_vals,
                     std::vector<float> * output, std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * src0 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne00, c.ne01, c.ne02, c.ne03);
    ggml_tensor * src1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.ne10, c.ne01, c.ne2,  c.ne3);
    if (!src0 || !src1) { *error = "tensor allocation failed"; ggml_free(ctx); return false; }

    ggml_tensor * dst = ggml_out_prod(ctx, src0, src1);
    if (!dst) { *error = "ggml_out_prod construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, dst);

    if (!ggml_backend_supports_op(backend, dst)) {
        *error = "backend does not report OUT_PROD support (is engine/patches/metal-out-prod.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(src0, src0_vals.data(), 0, src0_vals.size() * sizeof(float));
    ggml_backend_tensor_set(src1, src1_vals.data(), 0, src1_vals.size() * sizeof(float));

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
    const auto src0_vals = random_floats((size_t) c.ne00 * c.ne01 * c.ne02 * c.ne03, -3.0f, 3.0f);
    const auto src1_vals = random_floats((size_t) c.ne10 * c.ne01 * c.ne2  * c.ne3,  -2.0f, 2.0f);

    std::vector<float> cpu_out, metal_out;
    std::string err;

    if (!run_on_backend(cpu, c, src0_vals, src1_vals, &cpu_out, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, src0_vals, src1_vals, &metal_out, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    const bool ok = exactly_equal(cpu_out, metal_out, "output", c.name);
    if (ok) {
        std::printf("[%s] PASS (src0=[%lld,%lld,%lld,%lld] src1_ne10=%lld ne2=%lld ne3=%lld, %lld elems)\n",
                     c.name, (long long) c.ne00, (long long) c.ne01, (long long) c.ne02, (long long) c.ne03,
                     (long long) c.ne10, (long long) c.ne2, (long long) c.ne3,
                     (long long) (c.ne00 * c.ne10 * c.ne2 * c.ne3));
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-out-prod-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-out-prod-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-out-prod-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-out-prod-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-out-prod-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-out-prod-metal-numeric-test: FAIL");
    return 1;
}
