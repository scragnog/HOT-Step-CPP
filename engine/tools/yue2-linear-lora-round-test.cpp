// yue2-linear-lora-round-test.cpp -- checks whether removing the seemingly
// redundant round(ctx, x) in yue2-aitk-graph.h's linear() (LoRA branch) is
// actually a bit-exact no-op, the way the RMS double-round fix was.
//
// linear() currently does (LoRA branch only):
//   ax = mul_mat(adapter->a, round(ctx, x))
// At every real call site (block()'s qkv/output/gate_up/down), x is ALREADY
// the direct output of a ggml_bf16_round() node (e.g. rms()'s mul_bf16
// result), so round(ctx, x) looks like the same round(round(y)) == round(y)
// pattern already fixed in rms(). Forward-wise that's true bit-for-bit,
// since x's value already equals round(x)'s value.
//
// BUT unlike the rms() case, x here is NOT a single-consumer chain: it also
// feeds linear()'s "base" convrot8 branch directly (unrounded). So x has (at
// least) two consumers whenever an adapter is present. BF16_ROUND's own
// backward is "ggml_bf16_round(grad)" (rounds the incoming gradient) -- so
// the LoRA branch's contribution to x's total gradient is currently rounded
// BEFORE being summed with the base branch's contribution, and that sum
// then gets rounded AGAIN by x's own producing round() node when the
// gradient propagates further upstream. round(a + round(b)) is NOT in
// general equal to round(a + b) even when b is already bf16-representable,
// so removing the "redundant" round() could change backward numerics even
// though it changes nothing forward. This test builds both forms as real
// ggml graphs (forward AND backward, mirroring the two-consumer topology)
// and checks bit-for-bit equality of both, rather than trusting the
// forward-only idempotence argument.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(0x600DF00Du);
std::vector<float> random_vec(size_t n, float lo = -3.0f, float hi = 3.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (float & x : v) x = dist(rng);
    return v;
}

struct Built {
    ggml_tensor * p;      // leaf param, pre-round (stand-in for whatever produced "n")
    ggml_tensor * w_base; // fixed weight, stand-in for convrot8's base branch
    ggml_tensor * a;      // LoRA A, param
    ggml_tensor * b;      // LoRA B, param
    ggml_tensor * target; // fixed, for a scalar loss
    ggml_tensor * y;      // forward output
    ggml_tensor * loss;
};

// old_extra_round: mirrors linear()'s current round(ctx, x) in the LoRA
// branch; false mirrors the proposed fix (mul_mat(a, x) directly).
Built build(ggml_context * ctx, int64_t input, int64_t rank, int64_t output, int64_t seq,
            float scale, bool old_extra_round) {
    Built r{};
    r.p     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input, seq);
    r.w_base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input, output);
    r.a      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input, rank);
    r.b      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, output);
    r.target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output, seq);

    ggml_set_param(r.p);
    ggml_set_param(r.a);
    ggml_set_param(r.b);
    // w_base and target are fixed inputs, not trained -- mirrors the real
    // graph's convrot8 weights (not adapter params) and an external target.

    // x mirrors "n" in the real graph: literally the output of a
    // ggml_bf16_round() node, exactly like rms()'s mul_bf16 result.
    ggml_tensor * x = ggml_bf16_round(ctx, r.p);

    // base branch: consumes x directly, unrounded -- same as linear()'s
    // ggml_convrot8(w.weight_i8, x, ...) call.
    ggml_tensor * base = ggml_mul_mat(ctx, r.w_base, x);
    ggml_mul_mat_set_prec(base, GGML_PREC_F32);

    // LoRA branch: old code rounds x again here; the fix removes that.
    ggml_tensor * lora_in = old_extra_round ? ggml_bf16_round(ctx, x) : x;
    ggml_tensor * ax = ggml_mul_mat(ctx, r.a, lora_in);
    ggml_mul_mat_set_prec(ax, GGML_PREC_F32);
    ggml_tensor * delta = ggml_mul_mat(ctx, r.b, ax);
    ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
    delta = ggml_bf16_round(ctx, ggml_scale(ctx, delta, scale));

    r.y = ggml_bf16_round(ctx, ggml_add(ctx, base, delta));
    r.loss = ggml_sum(ctx, ggml_mul(ctx, r.y, r.target));
    return r;
}

struct Result {
    std::vector<float> y, grad_p, grad_a, grad_b;
};

bool run(ggml_backend_t backend, int64_t input, int64_t rank, int64_t output, int64_t seq,
         float scale, bool old_extra_round,
         const std::vector<float> & p_data, const std::vector<float> & wbase_data,
         const std::vector<float> & a_data, const std::vector<float> & b_data,
         const std::vector<float> & target_data, Result * out) {
    ggml_init_params ip{}; ip.mem_size = 4u*1024u*1024u; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    Built built = build(ctx, input, rank, output, seq, scale, old_extra_round);

    ggml_cgraph * g = ggml_new_graph_custom(ctx, 256, true);
    bool ok = g != nullptr;
    if (ok) {
        ggml_build_forward_expand(g, built.loss);
        ggml_set_loss(built.loss);
        ggml_build_backward_expand(ctx, g, nullptr);
    }

    ggml_gallocr_t alloc = ok ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)) : nullptr;
    ok = ok && alloc && ggml_gallocr_alloc_graph(alloc, g);

    if (ok) {
        ggml_backend_tensor_set(built.p,      p_data.data(),      0, ggml_nbytes(built.p));
        ggml_backend_tensor_set(built.w_base, wbase_data.data(),  0, ggml_nbytes(built.w_base));
        ggml_backend_tensor_set(built.a,      a_data.data(),      0, ggml_nbytes(built.a));
        ggml_backend_tensor_set(built.b,      b_data.data(),      0, ggml_nbytes(built.b));
        ggml_backend_tensor_set(built.target, target_data.data(), 0, ggml_nbytes(built.target));
        ok = ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS;
    }

    if (ok) {
        out->y.resize(ggml_nelements(built.y));
        ggml_backend_tensor_get(built.y, out->y.data(), 0, ggml_nbytes(built.y));

        ggml_tensor * gp = ggml_graph_get_grad(g, built.p);
        ggml_tensor * ga = ggml_graph_get_grad(g, built.a);
        ggml_tensor * gb = ggml_graph_get_grad(g, built.b);
        ok = gp && ga && gb;
        if (ok) {
            out->grad_p.resize(ggml_nelements(gp));
            out->grad_a.resize(ggml_nelements(ga));
            out->grad_b.resize(ggml_nelements(gb));
            ggml_backend_tensor_get(gp, out->grad_p.data(), 0, ggml_nbytes(gp));
            ggml_backend_tensor_get(ga, out->grad_a.data(), 0, ggml_nbytes(ga));
            ggml_backend_tensor_get(gb, out->grad_b.data(), 0, ggml_nbytes(gb));
        }
    }

    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

bool bit_exact(const std::vector<float> & x, const std::vector<float> & y, const char * name) {
    bool ok = x.size() == y.size() &&
              std::memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0;
    if (!ok) {
        size_t n = std::min(x.size(), y.size());
        size_t first_diff = n;
        float worst = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            if (x[i] != y[i]) {
                if (first_diff == n) first_diff = i;
                worst = std::max(worst, std::fabs(x[i]-y[i]));
            }
        }
        std::fprintf(stderr, "  %s MISMATCH: sizes %zu/%zu, first_diff=%zu, worst_abs=%.3e\n",
                     name, x.size(), y.size(), first_diff, (double) worst);
    }
    return ok;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-linear-lora-round-test: SKIP (no Metal device)"); return 0; }
    ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!backend) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    // Deliberately not a "nice" bf16-friendly size; also deliberately uses
    // values that are NOT already bf16-representable (random_vec's full F32
    // precision), so p's own round() actually does something, matching the
    // real graph where "n" is a genuine rms_norm/mul output, not a constant.
    const int64_t input = 64, rank = 8, output = 96, seq = 11;
    const float scale = 0.7f;

    auto p_data      = random_vec(size_t(input * seq));
    auto wbase_data  = random_vec(size_t(input * output));
    auto a_data      = random_vec(size_t(input * rank));
    auto b_data      = random_vec(size_t(rank * output));
    auto target_data = random_vec(size_t(output * seq));

    Result old_r, new_r;
    bool ok = run(backend, input, rank, output, seq, scale, true,
                  p_data, wbase_data, a_data, b_data, target_data, &old_r);
    ok = ok && run(backend, input, rank, output, seq, scale, false,
                    p_data, wbase_data, a_data, b_data, target_data, &new_r);

    bool y_ok = ok && bit_exact(old_r.y, new_r.y, "forward y");
    bool gp_ok = ok && bit_exact(old_r.grad_p, new_r.grad_p, "grad_p");
    bool ga_ok = ok && bit_exact(old_r.grad_a, new_r.grad_a, "grad_a");
    bool gb_ok = ok && bit_exact(old_r.grad_b, new_r.grad_b, "grad_b");

    std::printf("[forward]      %s\n", y_ok  ? "PASS" : "FAIL");
    std::printf("[grad_p]       %s\n", gp_ok ? "PASS" : "FAIL");
    std::printf("[grad_a]       %s\n", ga_ok ? "PASS" : "FAIL");
    std::printf("[grad_b]       %s\n", gb_ok ? "PASS" : "FAIL");

    bool all_ok = ok && y_ok && gp_ok && ga_ok && gb_ok;
    std::puts(all_ok ? "yue2-linear-lora-round-test: ALL PASS"
                      : "yue2-linear-lora-round-test: FAIL (see mismatches above -- "
                        "if only grad_p differs, the round() is NOT a safe no-op removal)");
    return all_ok ? 0 : 1;
}
