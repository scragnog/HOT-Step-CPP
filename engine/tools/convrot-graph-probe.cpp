// convrot-graph-probe.cpp — ConvRot8 under CUDA graph capture and replay.
//
// Qualification probe for the HOT-ggml port of ConvRot8 (the ConvRot/workspace
// gate in the ggml upgrade plan). The port dropped the old StreamGuard, which
// called cublasSetStream around every ConvRot GEMM; cublasSetStream also resets
// the handle's workspace, which upstream ggml now attaches per stream and which
// a CUDA graph capture depends on. This probe checks, on the real CUDA backend:
//
//   * YuE2-shaped ConvRot8 forward (I8 weight, rotation 256, BF16 compute) and
//     its gradient through ggml autodiff (GGML_OP_CONVROT8_BACK), with CUDA
//     graphs enabled: every phase runs one direct compute, one capture and then
//     replays; each cycle builds a fresh graph (a new capture); each phase after
//     the first flips the input's device pointer, which forces a re-capture and
//     a cudaGraphExecUpdate of the same graph.
//   * Two graph kinds per cycle: "convrot" (ConvRot forward + backward only) and
//     "mixed" (ConvRot forward, an F32 GEMM, its autodiff, ConvRot backward and
//     a trailing F32 GEMM on the gradient), so ordinary cuBLAS GEMMs run on the
//     same stream's handle before and after ConvRot inside one capture.
//   * Every output and gradient is finite, non-zero and bitwise identical to the
//     first compute (direct, captured, replayed, re-captured, every cycle).
//   * No stream reset: cuBLAS's own API log (cublasSetLoggerCallback) is counted
//     per compute. A cublasSetStream that is not part of creating a handle is a
//     reset. A positive control on a probe-owned handle proves the log sees both
//     calls before the count is trusted.
//   * Replays really replay: a replayed compute makes no cuBLAS API call at all,
//     while the direct and capture computes must. Graphs silently disabled (old
//     GPU, GGML_CUDA_DISABLE_GRAPHS, a build without GGML_CUDA_GRAPHS) fail here.
//   * A capture that loses its workspace aborts inside ggml-cuda (CUDA_CHECK /
//     CUBLAS_CHECK / the ConvRot status assert). Any exit without the final
//     "RESULT" line is a FAIL.
//
// Usage: convrot-graph-probe [--device CUDA0] [--cycles 3] [--phases 3]
//                            [--computes 6] [--rows 256] [--width 2048]
//                            [--hidden 1024] [--fp32]
//   --computes per phase, at least 3 (direct, capture, then replays).
//   --fp32 runs ConvRot with compute_bf16 = false instead of YuE2's BF16.
//   GGML_CUDA_GRAPH_LOG=1 additionally prints ggml-cuda's per-compute graph
//   decision line (the variable exists in both the baseline and candidate).
//
// Exit: 0 all checks pass, 1 a check failed, 2 setup error (no CUDA device,
// the cuBLAS log cannot see the calls it must count, bad arguments).
//
// Expected at baseline (old ggml + StreamGuard): the stream-reset check fails,
// because both ConvRot's StreamGuard and old ggml's cuBLAS mul_mat/out_prod
// call cublasSetStream on every op; the per-kind counts tell the two apart.
// Every other check is expected to pass on both revisions.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── cuBLAS API log counters ────────────────────────────────────────────────
static std::atomic<long> g_msgs{0}, g_create{0}, g_setstream{0}, g_gemm{0};

static void cublas_log_cb(const char * msg) {
    if (!msg) return;
    g_msgs++;
    if (strstr(msg, "cublasCreate"))    g_create++;
    if (strstr(msg, "cublasSetStream")) g_setstream++;
    if (strstr(msg, "emm"))             g_gemm++;   // cublasSgemm, cublasGemmEx, ...
}

struct LogSnap {
    long msgs, create, setstream, gemm;
    static LogSnap now() { return { g_msgs.load(), g_create.load(), g_setstream.load(), g_gemm.load() }; }
    LogSnap operator-(const LogSnap & o) const {
        return { msgs - o.msgs, create - o.create, setstream - o.setstream, gemm - o.gemm };
    }
};

// ── deterministic inputs ────────────────────────────────────────────────────
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t) (s >> 33); }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (float) next() / 2147483648.0f; }
};

static void fill_f32(ggml_tensor * t, Rng & r, float lo, float hi) {
    std::vector<float> v((size_t) ggml_nelements(t));
    for (auto & x : v) x = r.uniform(lo, hi);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

static void fill_i8(ggml_tensor * t, Rng & r) {
    std::vector<int8_t> v((size_t) ggml_nelements(t));
    for (auto & x : v) x = (int8_t) ((int) (r.next() % 255) - 127);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// ── one graph kind ──────────────────────────────────────────────────────────
struct Opts {
    std::string device;
    int cycles = 3, phases = 3, computes = 6, rows = 256, width = 2048, hidden = 1024;
    bool bf16 = true;
};

struct Probe {
    const char * kind = "";
    bool mixed = false;
    ggml_context * pctx = nullptr;        // inputs and parameters (allocated)
    ggml_backend_buffer_t pbuf = nullptr;
    ggml_context * gctx = nullptr;        // graph (gallocr)
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x = nullptr, * x_alt = nullptr;
    void * x_data = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> outs;

    void free_all() {
        if (galloc) ggml_gallocr_free(galloc);
        if (gctx) ggml_free(gctx);
        if (pbuf) ggml_backend_buffer_free(pbuf);
        if (pctx) ggml_free(pctx);
        *this = Probe{};
    }
};

static bool build(Probe & p, const char * kind, bool mixed, const Opts & o, ggml_backend_t be) {
    p.kind = kind;
    p.mixed = mixed;
    ggml_init_params pip = { ggml_tensor_overhead() * 16, nullptr, true };
    p.pctx = ggml_init(pip);
    ggml_tensor * w  = ggml_new_tensor_2d(p.pctx, GGML_TYPE_I8,  o.width, o.width);
    ggml_tensor * s  = ggml_new_tensor_1d(p.pctx, GGML_TYPE_F32, o.width);
    ggml_tensor * b  = ggml_new_tensor_1d(p.pctx, GGML_TYPE_F32, o.width);
    p.x              = ggml_new_tensor_2d(p.pctx, GGML_TYPE_F32, o.width, o.rows);
    p.x_alt          = ggml_new_tensor_2d(p.pctx, GGML_TYPE_F32, o.width, o.rows);
    ggml_tensor * m  = ggml_new_tensor_2d(p.pctx, GGML_TYPE_F32, o.width, o.hidden);
    ggml_tensor * n  = ggml_new_tensor_2d(p.pctx, GGML_TYPE_F32, o.width, o.hidden);
    ggml_tensor * c  = ggml_new_tensor_2d(p.pctx, GGML_TYPE_F32, mixed ? o.hidden : o.width, o.rows);
    p.pbuf = ggml_backend_alloc_ctx_tensors(p.pctx, be);
    if (!p.pbuf) return false;

    Rng r(0x5EEDC0DEULL);  // same seed every cycle: every compute must match the first
    fill_i8(w, r);
    fill_f32(s, r, 0.5f / (127.0f * std::sqrt((float) o.width)), 1.5f / (127.0f * std::sqrt((float) o.width)));
    fill_f32(b, r, -0.1f, 0.1f);
    fill_f32(p.x, r, -1.0f, 1.0f);
    {
        std::vector<uint8_t> tmp(ggml_nbytes(p.x));
        ggml_backend_tensor_get(p.x, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(p.x_alt, tmp.data(), 0, tmp.size());
    }
    fill_f32(m, r, -1.0f / std::sqrt((float) o.width), 1.0f / std::sqrt((float) o.width));
    fill_f32(n, r, -1.0f / std::sqrt((float) o.width), 1.0f / std::sqrt((float) o.width));
    fill_f32(c, r, -1.0f, 1.0f);
    p.x_data = p.x->data;

    ggml_set_param(p.x);
    if (mixed) ggml_set_param(m);

    ggml_init_params gip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, true), nullptr, true };
    p.gctx = ggml_init(gip);
    ggml_context * ctx = p.gctx;

    ggml_tensor * y = ggml_convrot8(ctx, w, p.x, s, b, 256, o.bf16);
    ggml_tensor * h = mixed ? ggml_mul_mat(ctx, m, y) : y;
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, h, c));
    ggml_set_loss(loss);

    p.gf = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(p.gf, loss);
    ggml_build_backward_expand(ctx, p.gf, nullptr);

    ggml_tensor * dx = ggml_graph_get_grad(p.gf, p.x);
    if (!dx) { fprintf(stderr, "[probe] %s: no gradient for x\n", kind); return false; }
    p.outs = { { "y", y }, { "loss", loss }, { "dx", dx } };
    if (mixed) {
        ggml_tensor * dm = ggml_graph_get_grad(p.gf, m);
        if (!dm) { fprintf(stderr, "[probe] %s: no gradient for m\n", kind); return false; }
        // A GEMM after ConvRot's backward, inside the same capture.
        ggml_tensor * z = ggml_mul_mat(ctx, n, dx);
        ggml_build_forward_expand(p.gf, z);
        p.outs.push_back({ "h", h });
        p.outs.push_back({ "dm", dm });
        p.outs.push_back({ "z", z });
    }
    for (auto & kv : p.outs) ggml_set_output(kv.second);
    ggml_tensor * loss_acc = ggml_graph_get_grad_acc(p.gf, loss);
    if (loss_acc) ggml_set_input(loss_acc);

    // Refuse to run anything the backend would not take (no silent fallback:
    // this probe computes directly on the CUDA backend, without a scheduler).
    for (int i = 0; i < ggml_graph_n_nodes(p.gf); ++i) {
        ggml_tensor * t = ggml_graph_node(p.gf, i);
        if (!ggml_backend_supports_op(be, t)) {
            fprintf(stderr, "[probe] %s: backend does not support %s (%s)\n", kind, ggml_op_desc(t), t->name);
            return false;
        }
    }

    p.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(p.galloc, p.gf)) return false;
    ggml_graph_reset(p.gf);  // loss gradient = 1
    return true;
}

// ── checks ─────────────────────────────────────────────────────────────────
struct Ref { std::vector<std::vector<uint8_t>> bytes; };

static int g_fail = 0;
static void fail(const char * fmt, const std::string & a, const std::string & b = "") {
    fprintf(stdout, "FAIL ");
    fprintf(stdout, fmt, a.c_str(), b.c_str());
    fprintf(stdout, "\n");
    g_fail++;
}

static void check_outputs(Probe & p, Ref & ref, const std::string & where) {
    const bool first = ref.bytes.empty();
    for (size_t k = 0; k < p.outs.size(); ++k) {
        ggml_tensor * t = p.outs[k].second;
        std::vector<uint8_t> buf(ggml_nbytes(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        const float * f = (const float *) buf.data();
        const size_t nf = buf.size() / sizeof(float);
        size_t nonfinite = 0, nonzero = 0;
        for (size_t i = 0; i < nf; ++i) {
            if (!std::isfinite(f[i])) nonfinite++;
            else if (f[i] != 0.0f) nonzero++;
        }
        const std::string name = std::string(p.kind) + "." + p.outs[k].first;
        if (nonfinite) fail("%s: non-finite values at %s", name, where);
        if (!nonzero)  fail("%s: all zero at %s", name, where);
        if (first) {
            ref.bytes.push_back(std::move(buf));
        } else if (ref.bytes[k] != buf) {
            size_t diff = 0;
            for (size_t i = 0; i < nf; ++i) diff += memcmp(&f[i], ref.bytes[k].data() + i * sizeof(float), sizeof(float)) != 0;
            fail("%s: not bitwise identical to the first compute at %s", name, where + " (" + std::to_string(diff) + " elements differ)");
        }
    }
}

int main(int argc, char ** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&](int & dst) {
            if (i + 1 >= argc) { fprintf(stderr, "[probe] %s needs a value\n", a.c_str()); exit(2); }
            dst = atoi(argv[++i]);
        };
        if      (a == "--device")   { if (i + 1 >= argc) { fprintf(stderr, "[probe] --device needs a value\n"); return 2; } o.device = argv[++i]; }
        else if (a == "--cycles")   num(o.cycles);
        else if (a == "--phases")   num(o.phases);
        else if (a == "--computes") num(o.computes);
        else if (a == "--rows")     num(o.rows);
        else if (a == "--width")    num(o.width);
        else if (a == "--hidden")   num(o.hidden);
        else if (a == "--fp32")     o.bf16 = false;
        else { fprintf(stderr, "[probe] unknown argument %s (see the header of convrot-graph-probe.cpp)\n", a.c_str()); return 2; }
    }
    if (o.cycles < 1 || o.phases < 1 || o.computes < 3 || o.rows < 1 || o.width < 256 || o.width % 256 || o.hidden < 1) {
        fprintf(stderr, "[probe] need cycles>=1 phases>=1 computes>=3 rows>=1 hidden>=1, width a multiple of 256\n");
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    if (!o.device.empty()) {
        dev = ggml_backend_dev_by_name(o.device.c_str());
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (strncmp(ggml_backend_dev_name(d), "CUDA", 4) == 0) { dev = d; break; }
        }
    }
    if (!dev || strncmp(ggml_backend_dev_name(dev), "CUDA", 4) != 0) {
        fprintf(stderr, "[probe] no CUDA device%s%s (is ggml-cuda.dll beside the exe?)\n",
                o.device.empty() ? "" : " named ", o.device.c_str());
        return 2;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "[probe] cannot init %s\n", ggml_backend_dev_name(dev)); return 2; }
    printf("backend %s | device %s (%s) | rows %d width %d hidden %d | %s | cycles %d phases %d computes %d\n",
           ggml_backend_name(be), ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
           o.rows, o.width, o.hidden, o.bf16 ? "bf16" : "fp32", o.cycles, o.phases, o.computes);

    // cuBLAS API log, then a positive control on a probe-owned handle: it must
    // see one cublasCreate and two cublasSetStream calls, or the counts below
    // mean nothing.
    cublasLoggerConfigure(1, 0, 0, nullptr);
    cublasSetLoggerCallback(cublas_log_cb);
    {
        const LogSnap before = LogSnap::now();
        cudaStream_t st = nullptr;
        cublasHandle_t h = nullptr;
        if (cudaStreamCreate(&st) != cudaSuccess || cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[probe] control: cannot create a stream or cuBLAS handle\n");
            return 2;
        }
        cublasSetStream(h, st);
        cublasSetStream(h, st);
        cublasDestroy(h);
        cudaStreamDestroy(st);
        const LogSnap d = LogSnap::now() - before;
        printf("control: cuBLAS log saw create=%ld setstream=%ld (want >=1 and >=2)\n", d.create, d.setstream);
        if (d.create < 1 || d.setstream < 2) {
            fprintf(stderr, "[probe] the cuBLAS API log does not report cublasCreate/cublasSetStream; "
                            "the stream-reset check cannot run\n");
            return 2;
        }
    }

    const struct { const char * name; bool mixed; } kinds[] = { { "convrot", false }, { "mixed", true } };
    Ref refs[2];
    long resets[2] = { 0, 0 }, gemm_seen[2] = { 0, 0 }, replays[2] = { 0, 0 }, captures[2] = { 0, 0 };
    long computes_done = 0;
    // Every graph stays alive until exit: ggml-cuda keys its CUDA graphs on
    // nodes[0]'s address, and a freed graph's address reused by the next cycle
    // would make that cycle's first compute look like a replay of the old one.
    std::vector<Probe> probes((size_t) o.cycles * 2);

    for (int cyc = 0; cyc < o.cycles; ++cyc) {
        for (int k = 0; k < 2; ++k) {
            Probe & p = probes[(size_t) cyc * 2 + k];
            if (!build(p, kinds[k].name, kinds[k].mixed, o, be)) {
                fprintf(stderr, "[probe] %s: graph setup failed\n", kinds[k].name);
                return 2;
            }
            for (int ph = 0; ph < o.phases; ++ph) {
                // Phase 0 uses x's own memory, odd phases the identical copy:
                // a changed source pointer forces ggml-cuda to re-capture.
                p.x->data = (ph % 2) ? p.x_alt->data : p.x_data;
                for (int c = 0; c < o.computes; ++c) {
                    const std::string where = "cycle " + std::to_string(cyc) + " phase " + std::to_string(ph) +
                                              " compute " + std::to_string(c);
                    const LogSnap before = LogSnap::now();
                    if (ggml_backend_graph_compute(be, p.gf) != GGML_STATUS_SUCCESS) {
                        fail("%s: graph compute failed at %s", kinds[k].name, where);
                        continue;
                    }
                    ggml_backend_synchronize(be);
                    const LogSnap d = LogSnap::now() - before;
                    computes_done++;
                    resets[k] += d.setstream - d.create > 0 ? d.setstream - d.create : 0;
                    if (c < 2) {
                        // c == 0 runs directly (new graph or changed pointer),
                        // c == 1 is the capture; both call cuBLAS.
                        if (d.msgs == 0) fail("%s: no cuBLAS call in a direct/capture compute at %s", kinds[k].name, where);
                        if (d.gemm == 0) fail("%s: no GEMM in the cuBLAS log at %s", kinds[k].name, where);
                        gemm_seen[k] += d.gemm;
                        if (c == 1) captures[k]++;
                    } else if (d.msgs != 0) {
                        fail("%s: cuBLAS was called during a compute that should have been a graph replay, at %s"
                             " (CUDA graphs disabled or re-captured every time?)", kinds[k].name, where);
                    } else {
                        replays[k]++;
                    }
                    check_outputs(p, refs[k], where);
                }
            }
            p.x->data = p.x_data;
        }
    }

    for (int k = 0; k < 2; ++k) {
        printf("%-7s captures %ld replays %ld | GEMM log entries in direct+capture computes %ld | stream resets %ld\n",
               kinds[k].name, captures[k], replays[k], gemm_seen[k], resets[k]);
        if (resets[k]) fail("%s: %s cublasSetStream calls outside handle creation (stream reset)", kinds[k].name,
                            std::to_string(resets[k]));
    }
    // Same number of direct+capture computes per kind, so the mixed graph's
    // own F32 GEMMs must add log entries on top of ConvRot's.
    if (gemm_seen[1] <= gemm_seen[0]) fail("%s: no ordinary cuBLAS GEMM beyond ConvRot's own (%s)", "mixed",
                                           std::to_string(gemm_seen[1]) + " vs " + std::to_string(gemm_seen[0]));

    for (auto & p : probes) p.free_all();
    ggml_backend_free(be);
    printf("RESULT %s: %ld computes, %d failed checks\n", g_fail ? "FAIL" : "PASS", computes_done, g_fail);
    return g_fail ? 1 : 0;
}
