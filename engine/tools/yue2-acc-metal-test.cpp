// yue2-acc-metal-test.cpp -- isolates the op the NaN bisect blamed: the
// GGML_OP_ACC that autodiff emits for the backward of a view (the two halves
// of the gate_up gradient, [12288, N] made from two [6144, N] gradients).
//
// Shape of the real graph (node 186 of the layer-27 block backward at N=17082):
//   z  = scale(x, 0)                       -- x = forward gate_up output, [12288, N]
//   y0 = acc(z,  g_gate, nb1,nb2,nb3, 0)    -- [6144, N] into the first half
//   y1 = acc(y0, g_up,   nb1,nb2,nb3, 6144*4)
// With finite g the result must be finite and equal on Metal and CPU.
// The bisect saw a finite zero src0 and a finite src1 (max 6.4e-6) produce
// 8405 non-finite values with max|x| = 3.3e38 on Metal.
//
// Usage: yue2-acc-metal-test [--quick]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Out { std::vector<float> y0, y1; };

bool run(ggml_backend_t backend, int64_t n, const std::vector<float> & x, const std::vector<float> & g0,
         const std::vector<float> & g1, Out * out) {
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false) + 4096;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;
    const int64_t W = 12288, H = 6144;
    ggml_tensor * tx  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, n);
    ggml_tensor * tg0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n);
    ggml_tensor * tg1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n);
    const size_t nb1 = tx->nb[1], nb2 = tx->nb[2], nb3 = tx->nb[3];
    ggml_tensor * z  = ggml_scale(ctx, tx, 0.0f);
    ggml_tensor * y0 = ggml_acc(ctx, z,  tg0, nb1, nb2, nb3, 0);
    ggml_tensor * y1 = ggml_acc(ctx, y0, tg1, nb1, nb2, nb3, (size_t) H * sizeof(float));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y1);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(tx,  x.data(),  0, x.size()*sizeof(float));
    ggml_backend_tensor_set(tg0, g0.data(), 0, g0.size()*sizeof(float));
    ggml_backend_tensor_set(tg1, g1.data(), 0, g1.size()*sizeof(float));
    const bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_synchronize(backend);
    out->y0.resize((size_t) W*n); out->y1.resize((size_t) W*n);
    ggml_backend_tensor_get(y0, out->y0.data(), 0, out->y0.size()*sizeof(float));
    ggml_backend_tensor_get(y1, out->y1.data(), 0, out->y1.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct Diff { size_t nonfinite = 0, mismatch = 0, first_bad = (size_t) -1; float maxabs = 0; };

Diff compare(const std::vector<float> & ref, const std::vector<float> & got) {
    Diff d;
    for (size_t i = 0; i < ref.size(); ++i) {
        const bool fin = std::isfinite(got[i]);
        if (!fin) ++d.nonfinite; else d.maxabs = std::fmax(d.maxabs, std::fabs(got[i]));
        if (!fin || got[i] != ref[i]) { ++d.mismatch; if (d.first_bad == (size_t) -1) d.first_bad = i; }
    }
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev || !cpu_dev) { std::puts("yue2-acc-metal-test: SKIP (need CPU and GPU device)"); return 0; }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    const int64_t sizes[] = {1024, 4096, 8069, 12455, 13500, 17082, 20000};
    bool all_ok = true;
    for (int64_t n : sizes) {
        if (quick && n > 8069) continue;
        std::mt19937 rng(1234 + (unsigned) n);
        std::uniform_real_distribution<float> ux(-4.0f, 4.0f), ug(-6.4e-6f, 6.4e-6f);
        std::vector<float> x((size_t) 12288*n), g0((size_t) 6144*n), g1((size_t) 6144*n);
        for (auto & v : x) v = ux(rng);
        for (auto & v : g0) v = ug(rng);
        for (auto & v : g1) v = ug(rng);
        Out ref;
        if (!run(cpu, n, x, g0, g1, &ref)) { std::fprintf(stderr, "[n=%lld] cpu run failed\n", (long long) n); all_ok = false; continue; }
        for (int rep = 0; rep < 3; ++rep) {
            Out got;
            if (!run(gpu, n, x, g0, g1, &got)) { std::fprintf(stderr, "[n=%lld] gpu run failed\n", (long long) n); all_ok = false; break; }
            const Diff d0 = compare(ref.y0, got.y0), d1 = compare(ref.y1, got.y1);
            const bool ok = d0.mismatch == 0 && d1.mismatch == 0;
            all_ok &= ok;
            std::printf("[n=%6lld rep %d] %s | y0: nonfinite=%zu mismatch=%zu first=%lld max=%.3e | y1: nonfinite=%zu mismatch=%zu first=%lld max=%.3e\n",
                        (long long) n, rep, ok ? "PASS" : "FAIL",
                        d0.nonfinite, d0.mismatch, (long long) d0.first_bad, (double) d0.maxabs,
                        d1.nonfinite, d1.mismatch, (long long) d1.first_bad, (double) d1.maxabs);
            if (!ok && d1.first_bad != (size_t) -1)
                std::printf("           first bad y1 element -> row(token)=%zu col=%zu\n", d1.first_bad / 12288, d1.first_bad % 12288);
        }
    }
    std::puts(all_ok ? "yue2-acc-metal-test: ALL PASS" : "yue2-acc-metal-test: FAIL");
    return all_ok ? 0 : 1;
}
