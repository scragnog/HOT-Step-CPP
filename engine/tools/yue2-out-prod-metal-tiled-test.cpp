// yue2-out-prod-metal-tiled-test.cpp -- equivalence + speed gate for the
// tiled Metal OUT_PROD kernel (GGML_METAL_OUT_PROD_TILED).
//
// Unlike yue2-convrot8-metal-tiled-test.cpp, this is NOT a bit-exactness
// test: kernel_out_prod_mm_f32 reduces via simdgroup_float8x8 MMA (blocked/
// pairwise order), while both the per-row Metal kernel and the CPU
// reference sum strictly ascending k=0..ne01-1 (see kernel_out_prod_f32's
// own comment) -- so tiled-vs-untiled and tiled-vs-CPU are both ordinary
// float32 rounding-tolerance comparisons here, not bit-for-bit.
//
// Builds src0/src1 exactly the way yue2-aitk-graph.h's LoRA backward does
// (ggml_out_prod(ctx, a, ggml_transpose(ctx, grad))) so the tiled kernel's
// general-stride path is exercised against the real (non-contiguous src1)
// shape, not just a convenient contiguous fixture.
//
// Usage: yue2-out-prod-metal-tiled-test [--quick]  (--quick skips the
// largest (gate_up "B"-site, K=12288) timed case)

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
#ifdef _WIN32
// MSVC has no setenv/unsetenv; every call here overwrites, as _putenv_s does.
static int setenv(const char * name, const char * value, int) { return _putenv_s(name, value); }
static int unsetenv(const char * name) { return _putenv_s(name, ""); }
#endif

namespace {

struct Case {
    const char * name;
    int64_t m, k, n; // dst = [m, n], reduction over k
    bool vs_cpu, timed;
};

const Case kCases[] = {
    // small / edge shapes: partial M and N tiles, K not a multiple of 32
    {"edge-small",           5,    7,    3, true,  false},
    {"edge-partial-m",      50,   40,   20, true,  false},
    {"edge-partial-n",      20,   40,   50, true,  false},
    {"edge-k-not-mult32",   64,   50,   64, true,  false},
    {"edge-tiny-m",          1,  200,   64, true,  false},
    // real YuE2 LoRA sites (rank 32, S 1024): "A"-adapter gradient (small K,
    // large M) and "B"-adapter gradient (small M, large K -- the expensive
    // one this kernel targets) at all four fused linear widths
    {"a-qkv",             2048,   32, 1024, true,  true},
    {"a-down",             6144,  32, 1024, true,  true},
    {"b-qkv",                32, 4096, 1024, false, true},
    {"b-output",             32, 2048, 1024, false, true},
    {"b-gate-up",            32,12288, 1024, false, true},
    // 2026-09-29: rank actually used by T69's crashing "base-matched"
    // recipe run (derived from the non-finite-gradient element counts in
    // the yue2-joint-train crash report: 131072 = rank(64) * hidden(2048)
    // for qkv/output/gate_up's lora_A, and 786432 = rank(64) *
    // 2*intermediate(12288) for gate_up's lora_B) -- NEVER exercised by
    // this test before (only rank 32 above, and the kernel's own "measured
    // ~31% faster" note is from a rank-256 run). vs_cpu is on for all of
    // these, unlike the original rank-32 "b-*" cases, specifically to
    // catch a silently-wrong-but-finite tiled result, not just a tiled-
    // vs-legacy mismatch (both could be wrong the same way).
    {"a-qkv-r64",           2048,   64, 1024, true,  true},
    {"a-output-r64",        2048,   64, 1024, true,  true},
    {"a-gate-up-r64",       2048,   64, 1024, true,  true},
    {"a-down-r64",          6144,   64, 1024, true,  true},
    {"b-qkv-r64",             64, 4096, 1024, true,  true},
    {"b-output-r64",          64, 2048, 1024, true,  true},
    {"b-gate-up-r64",         64,12288, 1024, true,  true},
    // Real training sequences are whole songs: 8069..12455 semantic frames plus
    // the prompt prefix, i.e. up to ~13.5k tokens, not 1024. n is the token
    // count here; every case above used only n = 1024, so anything that depends
    // on the number of column tiles or on large offsets was never exercised.
    {"a-qkv-s8069-r64",     2048,   64,  8069, true,  true},
    {"a-down-s13500-r64",   6144,   64, 13500, true,  true},
    {"a-gate-up-s13500-r64",2048,   64, 13500, true,  true},
    {"b-qkv-s8069-r64",       64, 4096,  8069, true,  true},
    {"b-output-s13500-r64",   64, 2048, 13500, true,  true},
    {"b-gate-up-s13500-r64",  64,12288, 13500, true,  true},
    {"b-gate-up-s13500-r32",  32,12288, 13500, true,  true},
};

std::mt19937 rng(0xC0FFEEu);

struct Fixture {
    std::vector<float> src0, grad; // src0 [m,k], grad [k,n] (transposed in-graph to feed out_prod)
};

Fixture make_fixture(const Case & c) {
    Fixture f;
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    f.src0.resize((size_t) c.m * c.k); for (auto & v : f.src0) v = u(rng);
    f.grad.resize((size_t) c.k * c.n); for (auto & v : f.grad) v = u(rng);
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
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.m, c.k); // src0 [m,k]
    ggml_tensor * g = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.n); // grad [k,n], contiguous
    ggml_tensor * y = ggml_out_prod(ctx, a, ggml_transpose(ctx, g));    // dst [m,n], real usage shape
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { *err = "alloc"; ggml_free(ctx); return false; }
    ggml_backend_tensor_set(a, f.src0.data(), 0, f.src0.size()*sizeof(float));
    ggml_backend_tensor_set(g, f.grad.data(), 0, f.grad.size()*sizeof(float));

    if (mode) setenv("GGML_METAL_OUT_PROD_TILED", mode, 1);
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
    if (!gpu_dev) { std::puts("yue2-out-prod-metal-tiled-test: SKIP (no Metal device)"); return 0; }
    ggml_backend_t cpu = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!gpu) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    // Tolerance: plain float32 GEMM with a different (blocked) reduction
    // order than the strictly-sequential reference -- ordinary rounding
    // noise, no quantization step to amplify it into a bucket flip (unlike
    // CONVROT8's bf16 rounding), so a tight bound is appropriate.
    const float atol = 2e-3f, rtol = 2e-3f;

    bool all_ok = true;
    for (const Case & c : kCases) {
        if (quick && (c.k >= 12288 || c.n > 1024)) continue;
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
            unsetenv("GGML_METAL_OUT_PROD_TILED");
            if (run(cpu, c, f, nullptr, 1, &ref, &err)) {
                const Diff dc = compare(ref.out, tiled.out, atol, rtol);
                ok &= dc.outside == 0;
                char tmp[128];
                std::snprintf(tmp, sizeof tmp, " | vs CPU: outside=%zu worst_abs=%.2e worst_rel=%.2e",
                              dc.outside, (double) dc.worst_abs, (double) dc.worst_rel);
                cpu_note = tmp;
            } else { ok = false; cpu_note = " | CPU run failed"; }
        }

        std::printf("[%s] %s m=%lld k=%lld n=%lld | tiled vs legacy: outside=%zu/%zu worst_abs=%.2e worst_rel=%.2e%s\n",
                    c.name, ok ? "PASS" : "FAIL", (long long) c.m, (long long) c.k, (long long) c.n,
                    dt.outside, legacy.out.size(), (double) dt.worst_abs, (double) dt.worst_rel, cpu_note.c_str());
        if (c.timed) {
            std::printf("    time  legacy %.2f ms -> tiled %.2f ms (%.1fx)\n",
                        legacy.ms, tiled.ms, tiled.ms > 0 ? legacy.ms/tiled.ms : 0.0);
        }
        all_ok &= ok;
    }
    if (cpu) ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::puts(all_ok ? "yue2-out-prod-metal-tiled-test: ALL PASS" : "yue2-out-prod-metal-tiled-test: FAIL");
    return all_ok ? 0 : 1;
}
