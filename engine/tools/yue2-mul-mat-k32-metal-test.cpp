// yue2-mul-mat-k32-metal-test.cpp -- equivalence + speed gate for the f32 K = 32
// (LoRA rank) mul_mat kernel kernel_mul_mm_k32_f32 (GGML_METAL_MM_K32).
//
// Not a bit-exactness test: the matrix-vector kernel reduces on 32 threads, the new
// kernel via simdgroup_float8x8 MMA, so both Metal paths and the CPU reference agree
// only to ordinary float32 rounding. Shapes are the real YuE2 adapter matmuls
// (delta = adapter->b @ ax: src0 [32, M], src1 [32, S], M = 4096/2048/12288,
// S = 9503/9504/15496 tokens). GGML_METAL_MM_K32=0 selects the old path.
//
// Usage: yue2-mul-mat-k32-metal-test [--quick]  (--quick skips the S > 1024 cases)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Case {
    const char * name;
    int64_t m, n; // dst = [m, n] = src0[32, m]^T-contracted with src1[32, n], K fixed at 32
    bool vs_cpu, timed;
};

const Case kCases[] = {
    // edge shapes around the 64-wide N tile (M must be a multiple of 64 for the new kernel;
    // N <= 8 stays on the matrix-vector path and must simply agree)
    {"n8-mv-path",            128,     8, true,  false},
    {"n9",                    128,     9, true,  false},
    {"n63",                   128,    63, true,  false},
    {"n64",                   128,    64, true,  false},
    {"n65",                   128,    65, true,  false},
    {"n1000",                 192,  1000, true,  false},
    // m not a multiple of 64 -> must fall back to the old path and still agree
    {"m-not-mult64",           96,   300, true,  false},
    // real YuE2 LoRA "B" matmuls (delta = adapter->b @ ax), rank 32
    {"qkv-s1024",            4096,  1024, true,  true},
    {"out-s1024",            2048,  1024, true,  true},
    {"gate-up-s1024",       12288,  1024, true,  true},
    {"qkv-nar-s9503",        4096,  9503, false, true},
    {"out-nar-s9504",        2048,  9504, false, true},
    {"gate-up-nar-s9503",   12288,  9503, false, true},
    {"qkv-ar-s15496",        4096, 15496, false, true},
    {"out-ar-s15496",        2048, 15496, true,  true},
    {"gate-up-ar-s15496",   12288, 15496, false, true},
};

std::mt19937 rng(0xC0FFEEu);

struct Fixture {
    std::vector<float> src0, src1; // src0 [32, m], src1 [32, n], both contiguous
};

Fixture make_fixture(const Case & c) {
    Fixture f;
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    f.src0.resize((size_t) 32 * c.m); for (auto & v : f.src0) v = u(rng);
    f.src1.resize((size_t) 32 * c.n); for (auto & v : f.src1) v = u(rng);
    return f;
}

struct Result {
    std::vector<float> out; // [m, n]
    double ms = 0;
};

using clk = std::chrono::steady_clock;

bool run(ggml_backend_t backend, const Case & c, const Fixture & f, const char * mode,
         int iters, Result * r, std::string * err) {
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(8, false) + 4096;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) { *err = "ggml_init"; return false; }
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, c.m); // src0 [K=32, m]
    ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, c.n); // src1 [K=32, n]
    ggml_tensor * y = ggml_mul_mat(ctx, a, g);                          // dst [m, n], like delta = b @ ax
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { *err = "alloc"; ggml_free(ctx); return false; }
    ggml_backend_tensor_set(a, f.src0.data(), 0, f.src0.size()*sizeof(float));
    ggml_backend_tensor_set(g, f.src1.data(), 0, f.src1.size()*sizeof(float));

    if (mode) setenv("GGML_METAL_MM_K32", mode, 1);
    bool ok = true;
    for (int it = 0; it < iters + 1 && ok; ++it) { // first iteration warms up pipeline compile
        const auto t0 = clk::now();
        ok &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        const auto t1 = clk::now();
        if (it > 0) r->ms += std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }
    if (!ok) *err = "compute";
    r->out.resize((size_t) c.m*c.n);
    ggml_backend_tensor_get(y, r->out.data(), 0, r->out.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct Diff { size_t outside = 0; float worst_abs = 0, worst_rel = 0; };

Diff compare(const std::vector<float> & ref, const std::vector<float> & got, float atol, float rtol) {
    Diff d;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(ref[i] - got[i]);
        const float bound = atol + rtol*std::fabs(ref[i]);
        if (!(diff <= bound) || std::isnan(diff)) ++d.outside;
        if (diff > d.worst_abs) d.worst_abs = diff;
        const float rel = diff / (std::fabs(ref[i]) + 1e-12f);
        if (rel > d.worst_rel) d.worst_rel = rel;
    }
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-mul-mat-k32-metal-test: SKIP (no Metal device)"); return 0; }
    ggml_backend_t cpu = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!gpu) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    // Tolerance: plain float32 GEMM with a different (blocked) reduction
    // order than the strictly-sequential reference -- ordinary rounding
    // noise, no quantization step to amplify it into a bucket flip (unlike
    // CONVROT8's bf16 rounding), so a tight bound is appropriate.
    const float atol = 1e-4f, rtol = 1e-4f;

    bool all_ok = true;
    for (const Case & c : kCases) {
        if (quick && (c.n > 1024)) continue;
        const Fixture f = make_fixture(c);
        // 20 iters (1 warm-up + 19 timed, see run()'s "if (it > 0)") --
        // bumped from 3 after a noisy run showed even the *unchanged*
        // a-qkv/a-down cases swing between runs; more samples averages
        // out system/thermal noise for a trustworthy before/after read.
        const int iters = c.timed ? 20 : 1;
        Result legacy, tiled, ref;
        std::string err;
        if (!run(gpu, c, f, "0", iters, &legacy, &err)) { std::fprintf(stderr, "[%s] legacy: %s\n", c.name, err.c_str()); all_ok = false; continue; }
        if (!run(gpu, c, f, "1", iters, &tiled, &err))  { std::fprintf(stderr, "[%s] tiled: %s\n",  c.name, err.c_str()); all_ok = false; continue; }

        const Diff dt = compare(legacy.out, tiled.out, atol, rtol);
        bool ok = dt.outside == 0;

        std::string cpu_note;
        if (c.vs_cpu && cpu) {
            unsetenv("GGML_METAL_MM_K32");
            if (run(cpu, c, f, nullptr, 1, &ref, &err)) {
                const Diff dc = compare(ref.out, tiled.out, atol, rtol);
                ok &= dc.outside == 0;
                char tmp[128];
                std::snprintf(tmp, sizeof tmp, " | vs CPU: outside=%zu worst_abs=%.2e worst_rel=%.2e",
                              dc.outside, (double) dc.worst_abs, (double) dc.worst_rel);
                cpu_note = tmp;
            } else { ok = false; cpu_note = " | CPU run failed"; }
        }

        std::printf("[%s] %s m=%lld n=%lld | tiled vs legacy: outside=%zu/%zu worst_abs=%.2e worst_rel=%.2e%s\n",
                    c.name, ok ? "PASS" : "FAIL", (long long) c.m, (long long) c.n,
                    dt.outside, legacy.out.size(), (double) dt.worst_abs, (double) dt.worst_rel, cpu_note.c_str());
        if (c.timed) {
            std::printf("    time  legacy %.2f ms -> tiled %.2f ms (%.1fx)\n",
                        legacy.ms, tiled.ms, tiled.ms > 0 ? legacy.ms/tiled.ms : 0.0);
        }
        all_ok &= ok;
    }
    if (cpu) ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::puts(all_ok ? "yue2-mul-mat-k32-metal-test: ALL PASS" : "yue2-mul-mat-k32-metal-test: FAIL");
    return all_ok ? 0 : 1;
}
