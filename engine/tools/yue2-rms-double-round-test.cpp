// yue2-rms-double-round-test.cpp -- confirms removing the redundant outer
// round() in yue2-aitk-graph.h's rms() is a no-op change.
//
// rms() used to compute:
//   mul_bf16(round(rms_norm(round(x))), weight)
// where mul_bf16(a,b) := round(mul(round(a), round(b))) -- so the "a" side
// was round(round(rms_norm(round(x)))), a DOUBLE application of
// ggml_bf16_round on the rms_norm output (mul_bf16's own round(a) is the
// second one). bf16_round_cast is a pure fp32->bf16->fp32 round-trip: once
// a value is exactly representable in bf16 (as it is right after the first
// round), rounding it again is a no-op -- round(round(v)) == round(v) for
// every finite v, forward AND for BF16_ROUND's backward (which itself just
// rounds the gradient, so the same idempotence applies there too).
//
// This test builds both the old (double-round) and new (single-round)
// forms as real ggml graphs on the Metal backend and checks the outputs
// are bit-for-bit identical, rather than trusting the argument alone.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(0xA5A5C0DEu);
std::vector<float> random_vec(size_t n) {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> v(n);
    for (float & x : v) x = dist(rng);
    return v;
}

// old: mul(round(round(rms_norm(round(x)))), round(weight)), then round.
// new: mul(round(rms_norm(round(x))),        round(weight)), then round.
ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * weight, float eps, bool old_double_round) {
    ggml_tensor * n = ggml_rms_norm(ctx, ggml_bf16_round(ctx, x), eps);
    n = ggml_bf16_round(ctx, n);
    if (old_double_round) {
        n = ggml_bf16_round(ctx, n); // the redundant second application
    }
    ggml_tensor * a = ggml_bf16_round(ctx, n);              // mul_bf16's own round(a)
    ggml_tensor * b = ggml_bf16_round(ctx, weight);          // mul_bf16's own round(b)
    return ggml_bf16_round(ctx, ggml_mul(ctx, a, b));
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-rms-double-round-test: SKIP (no Metal device)"); return 0; }
    ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!backend) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    const int64_t hidden = 2048, seq = 23;
    const float eps = 1e-6f;
    auto x_data = random_vec(size_t(hidden * seq));
    auto w_data = random_vec(size_t(hidden));

    auto run = [&](bool old_double_round, std::vector<float> * out) -> bool {
        ggml_init_params p{}; p.mem_size = 1u*1024u*1024u; p.no_alloc = true;
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        ggml_tensor * y = build(ctx, x, w, eps, old_double_round);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_set_input(x); ggml_set_input(w); ggml_set_output(y);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        bool ok = alloc && ggml_gallocr_alloc_graph(alloc, graph);
        if (ok) {
            ggml_backend_tensor_set(x, x_data.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(w, w_data.data(), 0, ggml_nbytes(w));
            ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        if (ok) {
            out->resize(ggml_nelements(y));
            ggml_backend_tensor_get(y, out->data(), 0, ggml_nbytes(y));
        }
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return ok;
    };

    std::vector<float> old_out, new_out;
    bool ok = run(true, &old_out) && run(false, &new_out);
    ok = ok && old_out.size() == new_out.size() &&
         std::memcmp(old_out.data(), new_out.data(), old_out.size()*sizeof(float)) == 0;

    std::printf("[double-round-vs-single-round] %s\n", ok ? "PASS" : "FAIL");
    std::puts(ok ? "yue2-rms-double-round-test: ALL PASS" : "yue2-rms-double-round-test: FAIL");
    return ok ? 0 : 1;
}
