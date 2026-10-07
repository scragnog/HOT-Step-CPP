// yue2-mul-bf16-dispatch-profile.cpp -- measures the real wall-clock cost of
// mul_bf16()'s per-call dispatch count (yue2-aitk-graph.h), to decide
// whether a BF16_ROUND "prologue fusion" (rounding mul_bf16's two inputs
// inline in the same dispatch as the multiply, instead of as two separate
// kernel_unary_impl passes) is worth the added complexity described in the
// discussion around this tool -- rather than guessing.
//
// mul_bf16(a,b) := round(mul(round(a), round(b))). The trailing round is
// already epilogue-fused into the mul's own dispatch (see the
// "BF16_ROUND epilogue fusion" comment in ggml-metal.metal /
// ggml-metal-ops.cpp), so the CURRENT cost is 3 dispatches: round(a),
// round(b), mul-with-round-epilogue. A full prologue+epilogue fusion would
// collapse this to 1 dispatch: mul-with-round-prologue-and-epilogue.
//
// This tool builds the exact 9 mul_bf16() call shapes one transformer block
// makes per forward pass (yue2_aitk_graph::rope()'s 4 calls each for the Q
// and K rotations, plus the gate*up activation mul), with forward+backward,
// at production-realistic sizes, and times two variants:
//   "current"       -- real mul_bf16: round(a), round(b), mul+round  (3 dispatches/call)
//   "fused (proxy)" -- skips BOTH input rounds: just mul+round        (1 dispatch/call)
// The "fused" variant is numerically wrong (a proxy for measurement only,
// not a correctness claim) but has the EXACT dispatch count a real prologue
// fusion would achieve, so the wall-clock delta is a trustworthy upper
// bound on what that fusion could save -- measured, not estimated.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(0xB16D15EDu);
std::vector<float> random_vec(size_t n) {
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> v(n);
    for (float & x : v) x = dist(rng);
    return v;
}

ggml_tensor * prof_round(ggml_context * ctx, ggml_tensor * x) {
    return ggml_bf16_round(ctx, x);
}

// fused_proxy=true skips both input round() calls -- a stand-in for a
// prologue-fused kernel's dispatch count (numerically wrong, but the point
// is the dispatch count / wall time, not correctness).
ggml_tensor * prof_mul_bf16(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, bool fused_proxy) {
    if (fused_proxy) {
        return prof_round(ctx, ggml_mul(ctx, a, b));
    }
    return prof_round(ctx, ggml_mul(ctx, prof_round(ctx, a), prof_round(ctx, b)));
}

struct CallShape {
    const char * label;
    int64_t ne0, ne1, ne2, ne3; // "a" shape (the large, per-head operand)
    int64_t bne0, bne1, bne2, bne3; // "b" shape (the small broadcast operand, e.g. cos/sin, or same as a)
};

// One block's worth of mul_bf16 calls, production-realistic sizes:
// hidden=2048, intermediate=6144, heads=16, kv_heads=8, head_dim=128, S=1500
// (60s at 25fps -- T69's actual clip length; see rope()'s [D/2,heads,S,1]
// shape and the gate/up activation's [intermediate,S] shape).
std::vector<CallShape> block_call_shapes() {
    const int64_t half_d = 64; // head_dim/2
    const int64_t S = 1500;
    const int64_t heads = 16, kv_heads = 8, intermediate = 6144;
    return {
        // q-rope: ac, bs, bc, as -- "a"/"b" here are x's two halves (both
        // [half_d,heads,S,1]), multiplied against cosine/sine
        // ([half_d,1,S,1], broadcasts over heads).
        {"q-rope-ac", half_d, heads, S, 1,  half_d, 1, S, 1},
        {"q-rope-bs", half_d, heads, S, 1,  half_d, 1, S, 1},
        {"q-rope-bc", half_d, heads, S, 1,  half_d, 1, S, 1},
        {"q-rope-as", half_d, heads, S, 1,  half_d, 1, S, 1},
        // k-rope: same 4, but kv_heads instead of heads.
        {"k-rope-ac", half_d, kv_heads, S, 1,  half_d, 1, S, 1},
        {"k-rope-bs", half_d, kv_heads, S, 1,  half_d, 1, S, 1},
        {"k-rope-bc", half_d, kv_heads, S, 1,  half_d, 1, S, 1},
        {"k-rope-as", half_d, kv_heads, S, 1,  half_d, 1, S, 1},
        // activation: gate * up, both [intermediate, S].
        {"gate*up",   intermediate, S, 1, 1,  intermediate, S, 1, 1},
    };
}

struct Built {
    std::vector<ggml_tensor *> a, b;
    ggml_tensor * loss;
};

Built build(ggml_context * ctx, const std::vector<CallShape> & shapes, bool fused_proxy) {
    Built built;
    ggml_tensor * acc = nullptr;
    for (const auto & s : shapes) {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.ne0, s.ne1, s.ne2, s.ne3);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.bne0, s.bne1, s.bne2, s.bne3);
        ggml_set_param(a);
        ggml_set_param(b);
        built.a.push_back(a);
        built.b.push_back(b);
        ggml_tensor * y = prof_mul_bf16(ctx, a, b, fused_proxy);
        ggml_tensor * s_ = ggml_sum(ctx, y);
        acc = acc ? ggml_add(ctx, acc, s_) : s_;
    }
    built.loss = acc;
    return built;
}

// Returns average ms per iteration (forward+backward), after warm-up.
double time_variant(ggml_backend_t backend, const std::vector<CallShape> & shapes,
                    bool fused_proxy, int iters, std::string * err) {
    ggml_init_params ip{}; ip.mem_size = 16u*1024u*1024u; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);

    Built built = build(ctx, shapes, fused_proxy);

    ggml_cgraph * g = ggml_new_graph_custom(ctx, 512, true);
    ggml_build_forward_expand(g, built.loss);
    ggml_set_loss(built.loss);
    ggml_build_backward_expand(ctx, g, nullptr);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, g)) {
        *err = "alloc failed";
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return -1.0;
    }

    for (size_t i = 0; i < built.a.size(); ++i) {
        auto ad = random_vec(size_t(ggml_nelements(built.a[i])));
        auto bd = random_vec(size_t(ggml_nelements(built.b[i])));
        ggml_backend_tensor_set(built.a[i], ad.data(), 0, ggml_nbytes(built.a[i]));
        ggml_backend_tensor_set(built.b[i], bd.data(), 0, ggml_nbytes(built.b[i]));
    }

    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(backend, g);
        auto t1 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            *err = "compute failed";
            ggml_gallocr_free(alloc);
            ggml_free(ctx);
            return -1.0;
        }
        if (it > 0) { // first iter is warm-up (pipeline compiles etc.)
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return total_ms / double(iters - 1);
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-mul-bf16-dispatch-profile: SKIP (no Metal device)"); return 0; }
    ggml_backend_t backend = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!backend) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    auto shapes = block_call_shapes();
    const int iters = 20;

    std::string err;
    double t_current = time_variant(backend, shapes, false, iters, &err);
    if (t_current < 0) { std::fprintf(stderr, "current variant: %s\n", err.c_str()); return 1; }
    double t_fused = time_variant(backend, shapes, true, iters, &err);
    if (t_fused < 0) { std::fprintf(stderr, "fused-proxy variant: %s\n", err.c_str()); return 1; }

    const double per_layer_saving_ms = t_current - t_fused;
    const int n_layers = 28; // this project's YuE2 layer count

    std::printf("mul_bf16 dispatch profile -- 9 calls/layer (4 q-rope + 4 k-rope + 1 gate*up), "
                "forward+backward, %d iterations:\n", iters - 1);
    std::printf("  current (3 dispatches/call): %.3f ms/layer\n", t_current);
    std::printf("  fused proxy (1 dispatch/call, numerically wrong -- timing only): %.3f ms/layer\n", t_fused);
    std::printf("  measured saving: %.3f ms/layer (%.1f%%)\n",
                 per_layer_saving_ms, t_current > 0 ? 100.0 * per_layer_saving_ms / t_current : 0.0);
    std::printf("  extrapolated to %d layers: %.2f ms/step upper bound "
                "(assumes full prologue+epilogue fusion, no other overhead)\n",
                n_layers, per_layer_saving_ms * n_layers);

    std::puts("yue2-mul-bf16-dispatch-profile: DONE (diagnostic tool, no PASS/FAIL)");
    return 0;
}
