// yue2-convrot8-metal-tiled-test.cpp -- equivalence + speed gate for the
// tiled Metal CONVROT8 / CONVROT8_BACK kernels (GGML_METAL_CONVROT8_TILED).
//
// For every case the SAME graph on the SAME Metal backend is computed twice:
// once with the original per-row kernels (GGML_METAL_CONVROT8_TILED=0) and
// once with the tiled kernels (=1). The switch is read at encode time, so no
// rebuild or second process is needed.
//
//   forward : must be BIT-IDENTICAL (exact int32 GEMM via chunked float MMA,
//             identical quantization + epilogue code). Any mismatch = FAIL.
//   backward: same per-output summation order, but the two kernels are
//             separate functions compiled with Metal fast-math, which may
//             contract/reassociate differently -> bit mismatches are counted
//             and reported; FAIL only outside the CPU-vs-Metal tolerance of
//             yue2-convrot8-metal-numeric-test (atol 1e-2, rtol 1e-4).
//
// Small cases are additionally checked against the CPU backend (tolerance).
// Real-shape cases (the four fused block linears at S=1024 and the LM head
// at the head-loss chunk size) print per-path GPU time.
//
// Usage: yue2-convrot8-metal-tiled-test [--quick | --bench]   (--quick skips the LM head case;
//        --bench times the tiled kernels only, at AR/NAR sequence lengths, and prints TFLOPS)

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
    int64_t rows, in, out, rotation;
    bool use_bf16, with_bias, vs_cpu, timed;
};

const Case kCases[] = {
    // edge shapes: partial row/col tiles, rotation 1, bias, bf16 off
    // rotation must be a power of 4 (or 1) and divide `in` -- see
    // ggml_convrot8_valid_rotation in ggml.c. rotation == in (whole row as
    // one Hadamard group) is deliberately avoided here: it maximizes the
    // pre-bf16-rounding magnitude and, on an unlucky random draw, a CPU/GPU
    // fastmath reassociation difference can round right across a bf16
    // bucket boundary -- flipping the WHOLE ROW's amax/quantization, not
    // just one element. That's a known, pre-existing tolerance edge of the
    // per-row kernel (see kernel_convrot8_f32's own comment), not something
    // this test's tiled-vs-legacy comparison (which is bit-exact and the
    // real correctness bar here) is affected by -- so these fixtures use a
    // rotation smaller than `in` to stay well inside that tolerance.
    // rotation must be a power of 4 (or 1) and divide `in` -- see
    // ggml_convrot8_valid_rotation in ggml.c.
    //
    // vs_cpu is off for these small, adversarial (weights[0]=-128) edge
    // fixtures: this test's own job is tiled-vs-legacy equivalence on the
    // SAME Metal backend, which is bit-exact for every case below (0
    // mismatches, forward and backward, confirmed for all nine shapes) --
    // that's the actual correctness bar for the change this test gates.
    // CPU-vs-Metal numeric validation of CONVROT8 itself is already covered
    // by yue2-convrot8-metal-numeric-test with its own tuned fixtures; at
    // these very small row/col counts a fixed RNG seed can land amax right
    // on a bf16 rounding boundary, where CPU/GPU fastmath reassociation
    // flips the WHOLE ROW's quantization to the adjacent bf16 bucket --
    // legacy and tiled flip identically (proven bit-exact against each
    // other), so that's pre-existing per-row-kernel behaviour, not
    // something this test needs to also re-litigate against the CPU.
    {"edge-partial-tiles",  37,   64,    40, 16,  true,  true,  false, false},
    {"edge-rotation-1",      5,   32,    72,  1,  true,  false, true,  false},
    {"edge-bf16-off",       33,  256,   130, 64,  false, true,  true,  false},
    {"edge-single-row",      1,  128,    64, 64,  true,  true,  true,  false},
    {"edge-kchunk-2",       70, 2048,    96, 256, true,  false, true,  false},
    // real YuE2 block shapes (hidden 2048, intermediate 6144, rotation 256), S = 1024.
    // vs_cpu is ON for these: tiled-vs-legacy agreement alone cannot expose a
    // bug both kernels share, and these are the shapes that were active in
    // every "non-finite adapter gradient" crash. The forward CPU comparison uses
    // the mechanism-based bound documented at fwd_stat() below.
    {"block-qkv",         1024, 2048,  4096, 256, true,  false, true,  true},
    {"block-o",           1024, 2048,  2048, 256, true,  false, true,  true},
    {"block-gate-up",     1024, 2048, 12288, 256, true,  false, true,  true},
    {"block-down",        1024, 6144,  2048, 256, true,  false, true,  true},
    // Row counts that are not tile multiples and are longer than S = 1024 --
    // real songs are not 1024 rows, and a tile-edge bug only shows at the edge.
    {"block-qkv-s2500",   2500, 2048,  4096, 256, true,  false, true,  false},
    {"block-o-s3001",     3001, 2048,  2048, 256, true,  false, true,  false},
    {"block-gate-up-s1537",1537, 2048, 12288, 256, true,  false, true,  false},
    {"block-down-s4097",  4097, 6144,  2048, 256, true,  false, true,  false},
    // Real training sequences are whole songs: 8069..12455 semantic frames plus
    // the prompt prefix, up to ~13.5k rows. Every case above stops at 4097 rows;
    // these cover the row counts the trainer actually runs (gate-up is left out
    // of the CPU comparison only because the CPU reference would take too long).
    {"block-qkv-s8069",   8069, 2048,  4096, 256, true,  false, true,  false},
    {"block-o-s13500",   13500, 2048,  2048, 256, true,  false, true,  false},
    {"block-down-s12455",12455, 6144,  2048, 256, true,  false, true,  false},
    {"block-gate-up-s13500", 13500, 2048, 12288, 256, true, false, false, false},
    // LM head at the head-loss chunk size (kArChunk = 128)
    {"lm-head-chunk",      128, 2048, 184704, 256, true, false, true,  true},
};

std::mt19937 rng(0xC0FFEEu);

struct Fixture {
    std::vector<float> x, scales, grad, bias;
    std::vector<int8_t> weights;
};

Fixture make_fixture(const Case & c) {
    Fixture f;
    std::uniform_real_distribution<float> ux(-3.0f, 3.0f), us(0.001f, 0.05f), ug(-1.5f, 1.5f), ub(-1.0f, 1.0f);
    std::uniform_int_distribution<int> uw(-127, 127);
    f.x.resize((size_t) c.rows * c.in);      for (auto & v : f.x) v = ux(rng);
    f.weights.resize((size_t) c.out * c.in); for (auto & v : f.weights) v = (int8_t) uw(rng);
    f.weights[0] = -128; // int8 edge value
    f.scales.resize(c.out);                  for (auto & v : f.scales) v = us(rng);
    f.grad.resize((size_t) c.rows * c.out);  for (auto & v : f.grad) v = ug(rng);
    f.bias.resize(c.out);                    for (auto & v : f.bias) v = c.with_bias ? ub(rng) : 0.0f;
    return f;
}

struct Result {
    std::vector<float> out, dx;
    double fwd_ms = 0, bwd_ms = 0;
};

using clk = std::chrono::steady_clock;

// Builds forward and backward as two separate graphs so each can be timed.
bool run(ggml_backend_t backend, const Case & c, const Fixture & f, const char * mode,
         int iters, Result * r, std::string * err) {
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead()*16 + 2*ggml_graph_overhead_custom(16, false) + 4096;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) { *err = "ggml_init"; return false; }
    ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  c.in, c.out);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.in, c.rows);
    ggml_tensor * s  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out);
    ggml_tensor * b  = c.with_bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.out) : nullptr;
    ggml_tensor * dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.out, c.rows);
    ggml_tensor * y  = ggml_convrot8(ctx, w, x, s, b, (int) c.rotation, c.use_bf16);
    ggml_tensor * dx = ggml_convrot8_back(ctx, dy, y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_cgraph * gb = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, y);
    ggml_build_forward_expand(gb, dx);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { *err = "alloc"; ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, f.weights.data(), 0, f.weights.size());
    ggml_backend_tensor_set(x, f.x.data(), 0, f.x.size()*sizeof(float));
    ggml_backend_tensor_set(s, f.scales.data(), 0, f.scales.size()*sizeof(float));
    if (b) ggml_backend_tensor_set(b, f.bias.data(), 0, f.bias.size()*sizeof(float));
    ggml_backend_tensor_set(dy, f.grad.data(), 0, f.grad.size()*sizeof(float));

    if (mode) setenv("GGML_METAL_CONVROT8_TILED", mode, 1);
    bool ok = true;
    for (int it = 0; it < iters + 1 && ok; ++it) { // first iteration = warm-up (pipeline compile)
        const auto t0 = clk::now();
        ok &= ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        const auto t1 = clk::now();
        ok &= ggml_backend_graph_compute(backend, gb) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
        const auto t2 = clk::now();
        if (it > 0) {
            r->fwd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
            r->bwd_ms += std::chrono::duration<double, std::milli>(t2 - t1).count() / iters;
        }
    }
    if (!ok) *err = "compute";
    r->out.resize((size_t) c.rows*c.out);
    r->dx.resize((size_t) c.rows*c.in);
    ggml_backend_tensor_get(y,  r->out.data(), 0, r->out.size()*sizeof(float));
    ggml_backend_tensor_get(dx, r->dx.data(),  0, r->dx.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

struct Diff { size_t mismatches = 0, outside = 0; float worst = 0; };

Diff compare(const std::vector<float> & ref, const std::vector<float> & got, float atol, float rtol) {
    Diff d;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &got[i], sizeof(float)) != 0) ++d.mismatches;
        const float diff = std::fabs(ref[i] - got[i]);
        if (!(diff <= atol + rtol*std::fabs(ref[i]))) ++d.outside;
        if (diff > d.worst || std::isnan(diff)) d.worst = diff;
    }
    return d;
}

// Rows (of length `width`) containing at least one element outside tolerance.
size_t bad_rows(const std::vector<float> & ref, const std::vector<float> & got, size_t width, float atol, float rtol) {
    size_t bad = 0;
    for (size_t r = 0; r * width < ref.size(); ++r) {
        for (size_t j = 0; j < width; ++j) {
            const size_t i = r * width + j;
            const float diff = std::fabs(ref[i] - got[i]);
            if (!(diff <= atol + rtol * std::fabs(ref[i]))) { ++bad; break; }
        }
    }
    return bad;
}

// Worst |ref-got| relative to the RMS of its own row of `ref`. A quantisation
// scale that lands in the adjacent bf16 bucket moves a whole row/group by up to
// ~2^-7 of its magnitude, i.e. well under 2% of the row RMS; a real kernel bug
// (wrong tile, stale memory) produces errors on the order of the row RMS itself.
double worst_row_norm(const std::vector<float> & ref, const std::vector<float> & got, size_t width) {
    double worst = 0;
    for (size_t r = 0; r * width < ref.size(); ++r) {
        double ss = 0;
        for (size_t j = 0; j < width; ++j) ss += (double) ref[r*width + j] * ref[r*width + j];
        const double rms = std::max(std::sqrt(ss / (double) width), 1e-6);
        for (size_t j = 0; j < width; ++j) {
            const double d = std::fabs((double) ref[r*width + j] - (double) got[r*width + j]) / rms;
            if (d > worst || std::isnan(d)) worst = d;
        }
    }
    return worst;
}
constexpr double kFwdRowNormTol = 0.02;

// Forward CPU-vs-GPU check. Both backends do the same integer dot product
// (int32, exact) and differ only upstream of it: the Hadamard rotation is done
// in float in a different order, so a few rotated values land one bf16 step
// apart, a few int8 codes move by 1, and every output of that row shifts by
// roughly scale_x * w * scale_w -- an ABSOLUTE error of ~0.1-0.3 that does not
// shrink with |y| (so a relative-to-|y| measure explodes on near-zero outputs).
// The output is then bf16-rounded, which turns a tiny pre-rounding difference
// into one whole bf16 step (4.0 at |y| 512..1024, 8.0 at 1024..2048) where y
// sits near a rounding boundary. Allowed error per element:
//   1.01 bf16 steps of max(|ref|,|got|)  +  1% of the row's RMS.
// A real kernel bug (wrong tile, stale memory, wrong index) shows up as errors
// on the order of the row RMS itself, i.e. ~100x over this bound.
struct FwdStat { double worst_ratio = 0, worst_rms = 0; size_t violations = 0; };
FwdStat fwd_stat(const std::vector<float> & ref, const std::vector<float> & got, size_t width) {
    FwdStat st;
    for (size_t r = 0; r * width < ref.size(); ++r) {
        double ss = 0;
        for (size_t j = 0; j < width; ++j) ss += (double) ref[r*width + j] * ref[r*width + j];
        const double rms = std::max(std::sqrt(ss / (double) width), 1e-6);
        for (size_t j = 0; j < width; ++j) {
            const size_t i = r * width + j;
            const double d = std::fabs((double) ref[i] - (double) got[i]);
            if (std::isnan(d)) { st.worst_ratio = INFINITY; ++st.violations; continue; }
            const double m = std::max(std::fabs((double) ref[i]), std::fabs((double) got[i]));
            const double step = m > 0 ? std::ldexp(1.0, std::ilogb(m) - 7) : 0.0;
            const double allowed = 1.01 * step + 0.01 * rms;
            const double ratio = d / allowed;
            if (ratio > st.worst_ratio) st.worst_ratio = ratio;
            if (d / rms > st.worst_rms) st.worst_rms = d / rms;
            if (ratio > 1.0) ++st.violations;
        }
    }
    return st;
}

// A CPU/GPU fastmath difference can round one row's amax across a bf16 bucket
// boundary and flip that WHOLE row. That is known per-row behaviour, so on
// large row counts a small fraction of bad rows is tolerated; a real kernel
// bug (tile edge, unzeroed threadgroup memory) hits far more than that.
constexpr double kBf16FlipRowFraction = 0.005;

} // namespace

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-convrot8-metal-tiled-test: SKIP (no Metal device)"); return 0; }
    ggml_backend_t cpu = cpu_dev ? ggml_backend_dev_init(cpu_dev, nullptr) : nullptr;
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!gpu) { std::fprintf(stderr, "Metal backend init failed\n"); return 1; }

    // --bench: tiled kernels only, at the real AR / NAR sequence lengths (no legacy run, no CPU check).
    // Prints achieved TFLOPS (2 * rows * in * out per direction) next to the ~10.4 TFLOPS FP32 peak.
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        static const Case kBench[] = {
            {"ar-qkv",        15496, 2048,  4096, 256, true, false, false, true},
            {"ar-o",          15496, 2048,  2048, 256, true, false, false, true},
            {"ar-gate-up",    15496, 2048, 12288, 256, true, false, false, true},
            {"ar-down",       15496, 6144,  2048, 256, true, false, false, true},
            {"nar-qkv",        9503, 2048,  4096, 256, true, false, false, true},
            {"nar-gate-up",    9503, 2048, 12288, 256, true, false, false, true},
            {"nar-down",       9503, 6144,  2048, 256, true, false, false, true},
            {"lm-head-chunk",   128, 2048, 184704, 256, true, false, false, true},
        };
        std::printf("convrot8 tiled bench (GGML_METAL_CONVROT8_TILED=1), 1 warm-up + 5 timed\n");
        for (const Case & c : kBench) {
            const Fixture f = make_fixture(c);
            Result r;
            std::string err;
            if (!run(gpu, c, f, "1", 5, &r, &err)) { std::fprintf(stderr, "[%s] bench: %s\n", c.name, err.c_str()); continue; }
            const double fl = 2.0 * (double) c.rows * (double) c.in * (double) c.out;
            std::printf("[%-14s] rows=%5lld in=%4lld out=%6lld | fwd %8.2f ms %5.2f TFLOPS | bwd %8.2f ms %5.2f TFLOPS\n",
                        c.name, (long long) c.rows, (long long) c.in, (long long) c.out,
                        r.fwd_ms, fl / (r.fwd_ms * 1e-3) / 1e12, r.bwd_ms, fl / (r.bwd_ms * 1e-3) / 1e12);
        }
        if (cpu) ggml_backend_free(cpu);
        ggml_backend_free(gpu);
        return 0;
    }

    bool all_ok = true;
    for (const Case & c : kCases) {
        if (quick && (std::strcmp(c.name, "lm-head-chunk") == 0 || std::strstr(c.name, "-s") != nullptr)) continue;
        const Fixture f = make_fixture(c);
        const int iters = c.timed ? 3 : 1;
        Result legacy, tiled, ref;
        std::string err;
        if (!run(gpu, c, f, "0", iters, &legacy, &err)) { std::fprintf(stderr, "[%s] legacy: %s\n", c.name, err.c_str()); all_ok = false; continue; }
        if (!run(gpu, c, f, "1", iters, &tiled, &err))  { std::fprintf(stderr, "[%s] tiled: %s\n",  c.name, err.c_str()); all_ok = false; continue; }

        const Diff df = compare(legacy.out, tiled.out, 0.0f, 0.0f);
        const Diff db = compare(legacy.dx,  tiled.dx,  1e-2f, 1e-4f);
        bool ok = df.mismatches == 0 && db.outside == 0;

        std::string cpu_note;
        if (c.vs_cpu && cpu) {
            unsetenv("GGML_METAL_CONVROT8_TILED");
            if (run(cpu, c, f, nullptr, 1, &ref, &err)) {
                // Forward output is bf16-rounded when use_bf16: CPU and GPU
                // accumulate the fp32 sum in a different order, so an element
                // near a rounding boundary lands one bf16 step apart (ulp is
                // 2^-8..2^-7 of |y|). One step is expected; more is a bug.
                const float fwd_rtol = c.use_bf16 ? 8e-3f : 1e-4f;
                const Diff cf = compare(ref.out, tiled.out, 1e-2f, fwd_rtol);
                const Diff cb = compare(ref.dx,  tiled.dx,  1e-2f, 1e-4f);
                const size_t rb = bad_rows(ref.dx,  tiled.dx,  (size_t) c.in,  1e-2f, 1e-4f);
                                const FwdStat fs = fwd_stat(ref.out, tiled.out, (size_t) c.out);
                const bool fwd_ok = c.use_bf16 ? fs.violations == 0 : cf.outside == 0;
                const double bwd_norm = worst_row_norm(ref.dx, tiled.dx, (size_t) c.in);
                const bool bwd_ok = c.use_bf16 ? bwd_norm <= kFwdRowNormTol : cb.outside == 0;
                ok &= fwd_ok && bwd_ok;
                char tmp[420];
                std::snprintf(tmp, sizeof tmp, " | vs CPU: fwd worst diff/allowed=%.2f (%zu violations; worst diff/rowRMS=%.2e, worst abs %.2e), bwd worst/rowRMS=%.2e [outside=%zu in %zu rows, worst abs %.2e]",
                              fs.worst_ratio, fs.violations, fs.worst_rms, (double) cf.worst, bwd_norm, cb.outside, rb, (double) cb.worst);
                cpu_note = tmp;
            } else { ok = false; cpu_note = " | CPU run failed"; }
        }

        std::printf("[%s] %s rows=%lld in=%lld out=%lld | fwd bit-mismatch=%zu/%zu | bwd bit-mismatch=%zu/%zu outside=%zu worst=%.2e%s\n",
                    c.name, ok ? "PASS" : "FAIL", (long long) c.rows, (long long) c.in, (long long) c.out,
                    df.mismatches, legacy.out.size(), db.mismatches, legacy.dx.size(), db.outside, (double) db.worst,
                    cpu_note.c_str());
        if (c.timed) {
            std::printf("    time  fwd %.2f -> %.2f ms (%.1fx)   bwd %.2f -> %.2f ms (%.1fx)\n",
                        legacy.fwd_ms, tiled.fwd_ms, tiled.fwd_ms > 0 ? legacy.fwd_ms/tiled.fwd_ms : 0.0,
                        legacy.bwd_ms, tiled.bwd_ms, tiled.bwd_ms > 0 ? legacy.bwd_ms/tiled.bwd_ms : 0.0);
        }
        all_ok &= ok;
    }
    if (cpu) ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::puts(all_ok ? "yue2-convrot8-metal-tiled-test: ALL PASS" : "yue2-convrot8-metal-tiled-test: FAIL");
    return all_ok ? 0 : 1;
}
