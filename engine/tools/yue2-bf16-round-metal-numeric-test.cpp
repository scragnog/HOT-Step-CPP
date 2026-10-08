// yue2-bf16-round-metal-numeric-test.cpp — numeric correctness gate for
// engine/patches/metal-bf16-round.patch (the new Metal
// GGML_OP_UNARY / GGML_UNARY_OP_BF16_ROUND kernel), kernel #1 of
// docs/plans/yue2-joint-training-metal-port.md's Phase 3.
//
// Per that plan's "Methodology" section, the CPU backend is already a
// complete, validated oracle for the whole AR block graph (Phase 1/2 got
// CPU to 0 unsupported nodes, and BF16_ROUND's own CPU kernel
// (zzz-yue2-bf16-round.patch) has been exercised throughout Phase 1/2's
// CPU-only Joint Training path). So this test needs no separate reference
// implementation: it builds the exact same GGML_UNARY_OP_BF16_ROUND
// forward+backward graph against both the CPU and Metal backends and diffs
// the two directly.
//
// Bit-exact is the right bar here (matching Phase 1's ConvRot8 test, not
// Phase 1's joint-loss test): the op is a pure per-element fp32->bf16->fp32
// round-trip cast with no reduction/summation, so there is no legitimate
// source of floating-point reordering between a GPU thread and a scalar CPU
// loop for this kernel specifically -- unlike e.g. RMS_NORM_BACK, which
// *will* need a tolerance-based bar once it's ported (see the plan's #4).
//
// Forward: y = BF16_ROUND(x). Backward: since BF16_ROUND is registered in
// ggml_compute_backward (zzz-yue2-bf16-round.patch, ggml.c) as
// "dx = BF16_ROUND(dy)", this test differentiates loss = sum(y * upstream)
// w.r.t. x, which makes the incoming gradient at the BF16_ROUND node exactly
// `upstream` -- so the fetched dx is a second, independent BF16_ROUND
// application, exercising the same kernel a second time with different
// input data in the same run.
//
// Skips (exit 0, not a failure) when no Metal device is present -- e.g. run
// by mistake on a non-Mac machine -- since this specific test only makes
// sense on the author's Mac (see AGENTS.md: Metal/Xcode code never builds or runs
// in the Linux dev environment used to draft these plans).
//
// Usage: yue2-bf16-round-metal-numeric-test   (no arguments)
//
//   cmake .
//   cmake --build . --target yue2-bf16-round-metal-numeric-test -j
//   ./yue2-bf16-round-metal-numeric-test

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
    int64_t ne0, ne1; // 2D shape: ne0 = row length, ne1 = row count
};

// Chosen to exercise every Metal dispatch path BF16_ROUND can take
// (ggml_metal_op_unary / ggml_metal_library_get_pipeline_unary):
//  - the "cnt" fast path (ggml_is_contiguous && nelements < 32768) vs the
//    generic strided dispatch (>= 32768 elements);
//  - the vectorized "_4" float4/bfloat4 kernel (ne0 % 4 == 0) vs the scalar
//    float/bfloat kernel (ne0 % 4 != 0).
// BF16_ROUND's own constructor (ggml_bf16_round in ggml.c) ggml_cont()s a
// non-contiguous input first, so every case here is already contiguous by
// construction -- there is no non-contiguous variant to test.
const Case kCases[] = {
    {"tiny-scalar",      7,     3},   // ne0%4!=0,  21 elems     -> cnt=true,  c4=false
    {"tiny-vec4",        16,    4},   // ne0%4==0,  64 elems     -> cnt=true,  c4=true
    {"large-scalar",     257,   200}, // ne0%4!=0,  51400 elems  -> cnt=false, c4=false
    {"large-vec4",       300,   200}, // ne0%4==0,  60000 elems  -> cnt=false, c4=true
    {"single-row-large", 40000, 1},   // 1 row, 40000 elems      -> cnt=false, c4=true
};

std::mt19937 rng(0xBF16u);

std::vector<float> random_floats(size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = dist(rng);
    return v;
}

// Builds a float exactly halfway between two adjacent bf16 representable
// values (bit 15 of the mantissa set, all lower bits clear), with the bit
// that survives truncation (bit 16, the retained bf16 LSB) forced to
// `low_bit_set`. Round-half-to-even and round-half-up agree everywhere
// EXCEPT exactly here, so these cases are the ones that actually pin down
// which rounding mode a kernel implements.
float make_tie_case(uint32_t mantissa_hi7, bool low_bit_set, bool negative) {
    const uint32_t exp_bits = 127u << 23; // magnitude in [1, 2)
    uint32_t mant = ((mantissa_hi7 & 0x7Fu) << 17) | (low_bit_set ? (1u << 16) : 0u) | 0x8000u;
    uint32_t bits = exp_bits | (mant & 0x7FFFFFu);
    if (negative) bits |= 0x80000000u;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Random data plus a fixed block of deterministic edge cases prepended to
// every tensor (every case in kCases has at least 21 elements, so all of
// these always land inside the tensor).
std::vector<float> make_values(int64_t n, float lo, float hi) {
    std::vector<float> v = random_floats((size_t) n, lo, hi);
    size_t i = 0;
    auto set = [&](float x) { if (i < v.size()) v[i++] = x; };
    set(0.0f);
    set(-0.0f);
    set(1.0f);
    set(-1.0f);
    set(make_tie_case(0x15, true,  false)); // exact tie, retained bit already 1 -> stays (round to even)
    set(make_tie_case(0x15, false, false)); // exact tie, retained bit 0 -> rounds away from zero to even
    set(make_tie_case(0x2A, true,  true));
    set(make_tie_case(0x2A, false, true));
    set(3.4028235e38f);   // near FLT_MAX
    set(-3.4028235e38f);
    set(1.1754944e-38f);  // near the smallest normal float
    set(-1.1754944e-38f);
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

// Builds y = BF16_ROUND(x), loss = sum(y * upstream), differentiates, and
// runs the whole forward+backward graph on `backend`. The gradient at x
// equals BF16_ROUND(upstream) by the op's own backward rule, so this
// exercises the kernel twice (forward on `input`, "backward" on `upstream`)
// per call.
bool run_on_backend(ggml_backend_t backend, const Case & c,
                     const std::vector<float> & input, const std::vector<float> & upstream,
                     std::vector<float> * output, std::vector<float> * input_grad,
                     std::string * error) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, true) + 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { *error = "ggml_init failed"; return false; }

    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
    ggml_tensor * up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
    if (!x || !up) { *error = "tensor allocation failed"; ggml_free(ctx); return false; }
    ggml_set_param(x);

    ggml_tensor * y    = ggml_bf16_round(ctx, x);
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, y, up));
    if (!y || !loss) { *error = "op construction failed"; ggml_free(ctx); return false; }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, true);
    if (!graph) { *error = "graph allocation failed"; ggml_free(ctx); return false; }
    ggml_build_forward_expand(graph, loss);
    ggml_set_loss(loss);
    ggml_build_backward_expand(ctx, graph, nullptr);

    if (!ggml_backend_supports_op(backend, y)) {
        *error = "backend does not report BF16_ROUND support (is engine/patches/metal-bf16-round.patch applied?)";
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { *error = "tensor allocation on backend failed"; ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,  input.data(),    0, input.size()    * sizeof(float));
    ggml_backend_tensor_set(up, upstream.data(), 0, upstream.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        *error = "graph compute failed";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_tensor * gx = ggml_graph_get_grad(graph, x);
    if (!gx) {
        *error = "missing input gradient";
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    output->resize((size_t) ggml_nelements(y));
    input_grad->resize((size_t) ggml_nelements(gx));
    ggml_backend_tensor_get(y,  output->data(),     0, ggml_nbytes(y));
    ggml_backend_tensor_get(gx, input_grad->data(), 0, ggml_nbytes(gx));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool run_case(ggml_backend_t cpu, ggml_backend_t metal, const Case & c) {
    const int64_t n = c.ne0 * c.ne1;
    const auto input    = make_values(n, -1000.0f, 1000.0f);
    const auto upstream = make_values(n, -1.0f, 1.0f);

    std::vector<float> cpu_out, cpu_grad, metal_out, metal_grad;
    std::string err;

    if (!run_on_backend(cpu, c, input, upstream, &cpu_out, &cpu_grad, &err)) {
        std::fprintf(stderr, "[%s] CPU run failed: %s\n", c.name, err.c_str());
        return false;
    }
    if (!run_on_backend(metal, c, input, upstream, &metal_out, &metal_grad, &err)) {
        std::fprintf(stderr, "[%s] Metal run failed: %s\n", c.name, err.c_str());
        return false;
    }

    bool ok = true;
    ok &= nearly_equal_bits(cpu_out,  metal_out,  "forward output", c.name);
    ok &= nearly_equal_bits(cpu_grad, metal_grad, "input gradient", c.name);

    if (ok) {
        std::printf("[%s] PASS (ne0=%lld ne1=%lld, %lld elems)\n",
                     c.name, (long long) c.ne0, (long long) c.ne1, (long long) (c.ne0 * c.ne1));
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr, "yue2-bf16-round-metal-numeric-test: no CPU backend device found\n");
        return 1;
    }
    ggml_backend_dev_t metal_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!metal_dev) {
        std::printf("yue2-bf16-round-metal-numeric-test: SKIP (no Metal/GPU device on this machine -- "
                     "this test only makes sense on the author's Mac)\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t metal = ggml_backend_dev_init(metal_dev, nullptr);
    if (!cpu || !metal) {
        std::fprintf(stderr, "yue2-bf16-round-metal-numeric-test: backend init failed (cpu=%p metal=%p)\n",
                     (void *) cpu, (void *) metal);
        if (cpu) ggml_backend_free(cpu);
        if (metal) ggml_backend_free(metal);
        return 1;
    }

    std::printf("yue2-bf16-round-metal-numeric-test: CPU=%s Metal=%s\n",
                ggml_backend_dev_description(cpu_dev), ggml_backend_dev_description(metal_dev));

    bool all_ok = true;
    for (const auto & c : kCases) {
        if (!run_case(cpu, metal, c)) all_ok = false;
    }

    ggml_backend_free(cpu);
    ggml_backend_free(metal);

    if (all_ok) {
        std::puts("yue2-bf16-round-metal-numeric-test: ALL PASS");
        return 0;
    }
    std::puts("yue2-bf16-round-metal-numeric-test: FAIL");
    return 1;
}
