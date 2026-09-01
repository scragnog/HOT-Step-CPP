// fattn-train-test.cpp — gate 1 for GGML_OP_FLASH_ATTN_TRAIN{,_BACK}.
//
// Runs, ON THE CPU BACKEND, two independent arms over identical inputs:
//
//   arm A (reference)  q,k,v[,mask] -> dit_attn_f32's exact op chain
//                      (mul_mat -> soft_max_ext(mask, scale) -> mul_mat ->
//                       permute -> cont), differentiated by ggml autodiff
//   arm B (fused)      q,k,v[,mask] -> ggml_flash_attn_train, O taken through
//                      ggml_flash_attn_train_get_o, same loss, autodiff
//
// Loss is sum(O * W) with a fixed random cotangent W, so grad(O) == W exactly
// and every gradient is exercised (a plain sum(O) lets sign errors cancel).
// Each arm gets its OWN ggml_context and cgraph — one shared context would
// either sum the two arms' gradients or leave one arm ungraded, and the
// comparison would be meaningless either way.
//
// PASS per case:
//   * forward O           max rel err <= 1e-4 vs the reference
//   * dQ, dK, dV          max rel err <= 1e-4 each
//   * every -INF-masked position contributes EXACTLY 0.0f to dK/dV
//   * reference gradients are non-zero (guards a vacuously green run — see
//     docs/plans/fattn-train-spec.md §7.1 on the loss-gradient accumulator)
//   * nothing is NaN or inf
//
// Grid: docs/plans/2026-09-01-flash-attn-backward.md gate 1 / spec §7.2.
// Masks come from dit_sa_mask / dit_ca_mask in train/dit-data.h — the very
// functions the trainer ships — never hand-rolled here.
//
// Usage: fattn-train-test [--backend cpu|cuda] [--extra] [--large] [--cheap]
//                         [--quick] [--fwd-only] [--threads N] [--bench]
//   --backend cuda  run arm B on the GPU backend; arm A (the reference chain)
//                   stays on the CPU backend, so the comparison is genuinely
//                   CPU-vs-CUDA. Whether the gradients are compared too is NOT
//                   hardcoded: the tool probes ggml_backend_supports_op for
//                   GGML_OP_FLASH_ATTN_TRAIN_BACK per case and drops to a
//                   forward-only comparison when the backward has no kernel
//                   there. In this mode every case also runs the flash arm
//                   TWICE and requires O, LSE, dQ, dK and dV to be bitwise
//                   identical.
//   --extra     append the cases the plan doc\'s gate-1 grid does not carry:
//               S_kv != S (the cross-attention shape, spec 7.2) and hand-built
//               fully-masked query rows (spec 4.4 / 7.3.9). Off by default so
//               the gate-1 case count stays what the plan doc quotes.
//   --large     append the one case this whole project exists for: S = S_kv =
//               3000, B = 1, Nh = 32, Nkv = 8, D = 128, window mask. The
//               REFERENCE CHAIN IS NOT RUN THERE — a retained [3000,3000,32]
//               softmax plus its backward is precisely the thing flash removes,
//               so arm A runs the FUSED op on the CPU backend instead and the
//               comparison is CUDA-fused vs CPU-fused. Combine with --quick to
//               shrink the rest of the grid around it.
//   --cheap     drop the S = 1000 null-mask cases (the two slowest cells)
//   --quick     stop at S = 198
//   --fwd-only  debug: skip the backward expansion entirely
//   --bench     skip the parity grid entirely and instead time FORWARD+BACKWARD
//               on the CUDA backend for the manual dit_attn_f32 chain (autodiff)
//               vs the fused ops, at S = S_kv in {625, 1250, 3000}, B = 1,
//               Nh = 32, Nkv = 8, D = 128, window mask, 10 warm-up + 50 timed
//               iterations. Graph build + allocation happen once, outside the
//               timed loop. If the manual arm cannot allocate at a given S (the
//               point of this project), that cell is reported SKIP rather than
//               crashing the rest of the grid. --backend is ignored: both arms
//               run on CUDA. Ignores --extra/--large/--cheap/--quick/--fwd-only.
// Exit code 0 only if every case passes (n/a to --bench, which always exits 0
// unless CUDA itself is unavailable or a compute call fails outright).
//
// The `flashMB` column is the flash arm's total ggml backend buffer — on CUDA,
// its VRAM. Every tensor in the arm gets its own storage (ggml-alloc's reuse
// planner is deliberately off, see arm_run), so it is a strict over-estimate of
// what the same graph costs inside the trainer, and an honest high-water for
// the tool itself.

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "ggml.h"

#include "train/dit-data.h"  // dit_sa_mask, dit_ca_mask

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// train/lm-common.h (pulled in by dit-data.h) forward-declares the trainer's
// JSONL sink and leaves it to the TU to define; ace-train.cpp is where the real
// one lives. This tool has no JSONL stream, so it swallows the line.
static void jl(const char * fmt, ...) {
    (void) fmt;
}

static const float PASS_REL = 1e-4f;
static bool g_fwd_only = false;   // debug: skip the backward expansion entirely

// ─── the reference attention chain ──────────────────────────────────────────
//
// Copied verbatim from engine/src/dit-graph.h:318-332 (dit_attn_f32). It is
// copied rather than included so this tool cannot perturb the inference path,
// and it must stay byte-identical to the original — if that helper ever
// changes, change this with it.
//
// Q: [D, S, Nh, B], K/V: [D, S_kv, Nkv, B], mask: [S_kv, S] F16 or NULL.
// Returns: [D, Nh, S, B].
static struct ggml_tensor * ref_attn_f32(struct ggml_context * ctx,
                                         struct ggml_tensor *  q,
                                         struct ggml_tensor *  k,
                                         struct ggml_tensor *  v,
                                         struct ggml_tensor *  mask,
                                         float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// ─── deterministic fill (no time-based seeds anywhere) ──────────────────────

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    float next() {  // uniform in [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t x = (uint32_t) (s >> 33);
        return (float) (x & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
};

static void fill(std::vector<float> & v, uint64_t seed) {
    Rng r(seed);
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = r.next();
    }
}

// ─── cases ──────────────────────────────────────────────────────────────────

enum MaskKind {
    MASK_NONE = 0,   // NULL mask
    MASK_WIN,        // sliding window 128, [S_kv, S] broadcast over heads+batch
    MASK_WIN_PAD,    // window + per-element padding, [S_kv, S, 1, B]
    MASK_CA_COL,     // dit_ca_mask with whole key columns masked for every query
    MASK_DEAD_ROW,   // hand-built: a few query rows with EVERY key masked
};

static const char * mask_name(MaskKind m) {
    switch (m) {
        case MASK_NONE:    return "null";
        case MASK_WIN:     return "window";
        case MASK_WIN_PAD: return "window+pad";
        case MASK_CA_COL:  return "ca-deadcol";
        case MASK_DEAD_ROW: return "dead-row";
    }
    return "?";
}

struct Case {
    std::string name;
    int64_t     S    = 0;
    int64_t     S_kv = 0;
    int64_t     B    = 0;
    int64_t     Nh   = 0;
    int64_t     Nkv  = 0;
    int64_t     D    = 0;
    MaskKind    mask = MASK_NONE;
    // Arm A runs the FUSED op on the CPU backend instead of the autodiff'd
    // dit_attn_f32 chain. Set for cases where the reference chain is the very
    // thing being avoided: at S = 3000 it retains a [3000,3000,32] softmax plus
    // its backward, which is ~4.6 GB of graph before a single gradient exists.
    bool        ref_fused = false;
};

struct Result {
    bool   ok        = false;
    bool   grads     = false;   // were the three gradients compared at all
    float  fwd_rel   = 0.0f;
    float  dq_rel    = 0.0f;
    float  dk_rel    = 0.0f;
    float  dv_rel    = 0.0f;
    double ms        = 0.0;
    size_t flash_buf = 0;       // flash arm's backend buffer bytes (VRAM on CUDA)
    size_t ref_buf   = 0;       // reference arm's backend buffer bytes (host)
    size_t flash_dev_free = 0;  // device free bytes at the flash arm's peak
    std::string maskzero = "n/a";
    std::string det      = "n/a";
    std::string note;
};

// ─── comparison ─────────────────────────────────────────────────────────────

static float rel_err(const std::vector<float> & a, const std::vector<float> & b, bool * finite) {
    float maxb = 0.0f;
    float maxd = 0.0f;
    *finite    = true;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            *finite = false;
            return INFINITY;
        }
        const float ab = std::fabs(b[i]);
        if (ab > maxb) {
            maxb = ab;
        }
        const float d = std::fabs(a[i] - b[i]);
        if (d > maxd) {
            maxd = d;
        }
    }
    return maxd / (maxb + 1e-6f);
}

static float max_abs(const std::vector<float> & a) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const float x = std::fabs(a[i]);
        if (x > m) {
            m = x;
        }
    }
    return m;
}

// ─── plain host attention, for triage only ──────────────────────────────────
//
// Not part of the PASS criteria — printed when a case fails, so the table says
// WHICH arm drifted rather than just that the two disagree.
static void host_attn(const Case & c,
                      const std::vector<float> &    hq,
                      const std::vector<float> &    hk,
                      const std::vector<float> &    hv,
                      const std::vector<uint16_t> & hm,
                      int64_t                       mask_ne1,
                      int64_t                       mask_ne3,
                      const std::vector<float> &    hw,
                      std::vector<float> &          o,
                      std::vector<float> &          dv) {
    const float   scale = 1.0f / std::sqrt((float) c.D);
    const int64_t G     = c.Nh / c.Nkv;
    o.assign((size_t) (c.D * c.Nh * c.S * c.B), 0.0f);
    dv.assign((size_t) (c.D * c.S_kv * c.Nkv * c.B), 0.0f);
    std::vector<float> p((size_t) c.S_kv);
    for (int64_t b = 0; b < c.B; b++) {
        for (int64_t h = 0; h < c.Nh; h++) {
            const int64_t hkv = h / G;
            for (int64_t i = 0; i < c.S; i++) {
                const float * qi = hq.data() + (size_t) (c.D * (i + c.S * (h + c.Nh * b)));
                float m = -INFINITY;
                for (int64_t j = 0; j < c.S_kv; j++) {
                    float mv = 0.0f;
                    if (!hm.empty()) {
                        const int64_t mi = j + c.S_kv * (i + mask_ne1 * (b % mask_ne3));
                        mv = ggml_fp16_to_fp32(hm[(size_t) mi]);
                    }
                    float s = 0.0f;
                    if (mv != -INFINITY) {
                        const float * kj = hk.data() + (size_t) (c.D * (j + c.S_kv * (hkv + c.Nkv * b)));
                        for (int64_t d = 0; d < c.D; d++) {
                            s += qi[d] * kj[d];
                        }
                        s = scale * s + mv;
                    } else {
                        s = -INFINITY;
                    }
                    p[(size_t) j] = s;
                    if (s > m) {
                        m = s;
                    }
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < c.S_kv; j++) {
                    p[(size_t) j] = (p[(size_t) j] == -INFINITY) ? 0.0f : std::exp(p[(size_t) j] - m);
                    sum += p[(size_t) j];
                }
                float * orow = o.data() + (size_t) (c.D * (h + c.Nh * (i + c.S * b)));
                if (sum <= 0.0f) {
                    continue;
                }
                const float * doi = hw.empty() ? nullptr
                                  : hw.data() + (size_t) (c.D * (h + c.Nh * (i + c.S * b)));
                for (int64_t j = 0; j < c.S_kv; j++) {
                    const float w = p[(size_t) j] / sum;
                    if (w == 0.0f) {
                        continue;
                    }
                    const float * vj = hv.data() + (size_t) (c.D * (j + c.S_kv * (hkv + c.Nkv * b)));
                    for (int64_t d = 0; d < c.D; d++) {
                        orow[d] += w * vj[d];
                    }
                    if (doi) {
                        float * dvj = dv.data() + (size_t) (c.D * (j + c.S_kv * (hkv + c.Nkv * b)));
                        for (int64_t d = 0; d < c.D; d++) {
                            dvj[d] += w * doi[d];
                        }
                    }
                }
            }
        }
    }
}

// ─── one arm ────────────────────────────────────────────────────────────────

struct Arm {
    ggml_context * ctx  = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *  q    = nullptr;
    ggml_tensor *  k    = nullptr;
    ggml_tensor *  v    = nullptr;
    ggml_tensor *  w    = nullptr;
    ggml_tensor *  m    = nullptr;
    ggml_tensor *  o    = nullptr;
    ggml_tensor *  pkd  = nullptr;   // flash arm only: the packed O|LSE tensor
    ggml_tensor *  lsec = nullptr;   // flash arm only: cont() of the LSE view
    ggml_tensor *  loss = nullptr;
    // Device free bytes sampled with this arm's buffers STILL LIVE, right after
    // its graph has run — so it counts the CUDA pool scratch the backward asks
    // for as well as the tensors. Sampling after the arm is destroyed reports
    // the buffers back, which is how a VRAM figure ends up looking like 4 MB.
    size_t dev_free_at_peak = 0;

    ~Arm() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static bool arm_run(Arm &                       a,
                    bool                        flash,
                    bool                        with_grads,
                    const Case &                c,
                    int64_t                     mask_ne1,
                    int64_t                     mask_ne3,
                    const std::vector<float> &  hq,
                    const std::vector<float> &  hk,
                    const std::vector<float> &  hv,
                    const std::vector<float> &  hw,
                    const std::vector<uint16_t> & hm,
                    ggml_backend_t              be,
                    std::vector<float> &        out_o,
                    std::vector<float> &        out_dq,
                    std::vector<float> &        out_dk,
                    std::vector<float> &        out_dv,
                    std::vector<float> *        out_lse = nullptr) {
    const float scale = 1.0f / std::sqrt((float) c.D);

    const size_t mem = ggml_tensor_overhead() * 8192 +
                       ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true) + (1u << 20);

    ggml_init_params ip = {};
    ip.mem_size         = mem;
    ip.mem_buffer       = nullptr;
    ip.no_alloc         = true;
    a.ctx               = ggml_init(ip);
    if (!a.ctx) {
        fprintf(stderr, "[fattn-train-test] ggml_init failed\n");
        return false;
    }

    a.q = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, c.D, c.S,    c.Nh,  c.B);
    a.k = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, c.D, c.S_kv, c.Nkv, c.B);
    a.v = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, c.D, c.S_kv, c.Nkv, c.B);
    a.w = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, c.D, c.Nh,   c.S,   c.B);
    ggml_set_name(a.q, "q");
    ggml_set_name(a.k, "k");
    ggml_set_name(a.v, "v");
    ggml_set_name(a.w, "w");
    ggml_set_param(a.q);
    ggml_set_param(a.k);
    ggml_set_param(a.v);
    ggml_set_input(a.w);

    if (c.mask != MASK_NONE) {
        a.m = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F16, c.S_kv, mask_ne1, 1, mask_ne3);
        ggml_set_name(a.m, "mask");
        ggml_set_input(a.m);
    }

    if (flash) {
        a.pkd = ggml_flash_attn_train(a.ctx, a.q, a.k, a.v, a.m, scale);
        a.o   = ggml_flash_attn_train_get_o(a.ctx, a.pkd);
    } else {
        a.o = ref_attn_f32(a.ctx, a.q, a.k, a.v, a.m, scale);
    }
    ggml_set_name(a.o, "o");

    a.loss = ggml_sum(a.ctx, ggml_mul(a.ctx, a.o, a.w));
    ggml_set_name(a.loss, "loss");
    ggml_set_loss(a.loss);

    a.gf = ggml_new_graph_custom(a.ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(a.gf, a.loss);
    if (with_grads) {
        ggml_build_backward_expand(a.ctx, a.gf, nullptr);
    }

    // The LSE region is read back for the finiteness check and for the CUDA
    // determinism check. It is appended AFTER the backward expansion on
    // purpose: a differentiated consumer of the LSE view would fire
    // ggml_acc_or_set on `packed` a second time (spec 9.7) and change the very
    // gradient path under test.
    if (flash && a.pkd) {
        a.lsec = ggml_cont(a.ctx, ggml_flash_attn_train_get_lse(a.ctx, a.pkd));
        ggml_build_forward_expand(a.gf, a.lsec);
    }

    // A gradient is not necessarily a dense tensor in its source's logical
    // order. The reference arm's dV in particular arrives as ggml's TRANSPOSE
    // backward hands it over: the right `ne`, but vt's strides — so reading its
    // bytes linearly reads the transpose. Materialise each gradient through
    // ggml_cont before it is read back, or the comparison is layout noise.
    ggml_tensor * keep[5] = { a.o, nullptr, nullptr, nullptr, a.lsec };
    if (with_grads) {
        ggml_tensor * src[3] = { a.q, a.k, a.v };
        for (int i = 0; i < 3; i++) {
            ggml_tensor * g = ggml_graph_get_grad(a.gf, src[i]);
            if (!g) {
                fprintf(stderr, "[fattn-train-test] missing gradient (arm %s)\n", flash ? "flash" : "ref");
                return false;
            }
            keep[i + 1] = ggml_cont(a.ctx, g);
            ggml_build_forward_expand(a.gf, keep[i + 1]);
        }
    }
    for (int i = 0; i < 5; i++) {
        if (!keep[i]) {
            continue;
        }
        ggml_set_output(keep[i]);
        if (keep[i]->view_src) {
            ggml_set_output(keep[i]->view_src);
        }
    }

    // Every tensor in the context gets its OWN storage. ggml-alloc's reuse
    // planner is deliberately NOT used here: it hands one address to several
    // tensors, and a correctness oracle must not rest on the liveness analysis
    // it is meant to be checking.
    a.buf = ggml_backend_alloc_ctx_tensors(a.ctx, be);
    if (!a.buf) {
        fprintf(stderr, "[fattn-train-test] tensor allocation failed (arm %s)\n", flash ? "flash" : "ref");
        return false;
    }

    ggml_backend_tensor_set(a.q, hq.data(), 0, ggml_nbytes(a.q));
    ggml_backend_tensor_set(a.k, hk.data(), 0, ggml_nbytes(a.k));
    ggml_backend_tensor_set(a.v, hv.data(), 0, ggml_nbytes(a.v));
    ggml_backend_tensor_set(a.w, hw.data(), 0, ggml_nbytes(a.w));
    if (a.m) {
        ggml_backend_tensor_set(a.m, hm.data(), 0, ggml_nbytes(a.m));
    }

    // ggml_set_loss only ALLOCATES grad_accs[loss]; it never initialises it.
    // Without this write every gradient in both arms is scaled by whatever the
    // buffer held — and if that is zero, the whole gate passes vacuously.
    if (with_grads) {
        ggml_tensor * lacc = ggml_graph_get_grad_acc(a.gf, a.loss);
        if (!lacc) {
            fprintf(stderr, "[fattn-train-test] no loss grad accumulator\n");
            return false;
        }
        const float one = 1.0f;
        ggml_backend_tensor_set(lacc, &one, 0, sizeof(float));
    }

    if (ggml_backend_graph_compute(be, a.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[fattn-train-test] graph compute failed (arm %s)\n", flash ? "flash" : "ref");
        return false;
    }

    {
        ggml_backend_dev_t dv = ggml_backend_get_device(be);
        if (dv) {
            size_t total = 0;
            ggml_backend_dev_memory(dv, &a.dev_free_at_peak, &total);
        }
    }

    out_o.resize((size_t) ggml_nelements(a.o));
    ggml_backend_tensor_get(a.o, out_o.data(), 0, ggml_nbytes(a.o));
    if (with_grads) {
        out_dq.resize((size_t) ggml_nelements(keep[1]));
        out_dk.resize((size_t) ggml_nelements(keep[2]));
        out_dv.resize((size_t) ggml_nelements(keep[3]));
        ggml_backend_tensor_get(keep[1], out_dq.data(), 0, ggml_nbytes(keep[1]));
        ggml_backend_tensor_get(keep[2], out_dk.data(), 0, ggml_nbytes(keep[2]));
        ggml_backend_tensor_get(keep[3], out_dv.data(), 0, ggml_nbytes(keep[3]));
    } else {
        out_dq.clear();
        out_dk.clear();
        out_dv.clear();
    }
    if (out_lse && a.lsec) {
        out_lse->resize((size_t) ggml_nelements(a.lsec));
        ggml_backend_tensor_get(a.lsec, out_lse->data(), 0, ggml_nbytes(a.lsec));
    }
    return true;
}

// ─── capability probe ───────────────────────────────────────────────────────
//
// spec §9.8: whether the gradients get compared on a given backend is a fact
// about that backend, not a constant in this file. ggml_backend_supports_op
// needs a real node, so build one in a no_alloc scratch context (supports_op
// only ever inspects shapes and types) and ask.
static bool probe_supports(ggml_backend_t be, const Case & c, bool back) {
    ggml_init_params ip = {};
    ip.mem_size         = ggml_tensor_overhead() * 64;
    ip.mem_buffer       = nullptr;
    ip.no_alloc         = true;

    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.S,    c.Nh,  c.B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.S_kv, c.Nkv, c.B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.D, c.S_kv, c.Nkv, c.B);
    ggml_tensor * m = (c.mask == MASK_NONE)
                    ? nullptr
                    : ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.S_kv, c.S);

    const float   scale = 1.0f / std::sqrt((float) c.D);
    ggml_tensor * node  = ggml_flash_attn_train(ctx, q, k, v, m, scale);
    if (back) {
        ggml_tensor * dfwd = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(node));
        node = ggml_flash_attn_train_back(ctx, q, k, v, m, node, dfwd, scale);
    }

    const bool ok = ggml_backend_supports_op(be, node);
    ggml_free(ctx);
    return ok;
}

// ─── --bench mode ───────────────────────────────────────────────────────────
//
// HOT-Step patch: flash-attn-train (bench mode is test-tool-only, no ggml
// source touched). Times FORWARD+BACKWARD on the CUDA backend for two arms at
// production geometry: the autodiff'd dit_attn_f32 chain (arm "manual") vs
// the fused ggml_flash_attn_train{,_back} ops (arm "fused"). Graph build and
// allocation happen ONCE, outside the timed loop — only
// ggml_backend_graph_compute is timed, 10 warm-up + 50 timed iterations.
//
// docs/plans/2026-09-01-flash-attn-backward.md bench task.

struct BenchArm {
    ggml_context *         ctx = nullptr;
    ggml_cgraph *          gf  = nullptr;
    ggml_backend_buffer_t  buf = nullptr;
    ~BenchArm() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// Builds + allocates one arm's graph. Returns false (arm left half-built,
// destructor cleans up whatever was created) on allocation failure — EXPECTED
// for the manual arm at large S (that is the point of this whole project), so
// the caller reports it as a result cell rather than treating it as a crash.
static bool bench_build(BenchArm & a, bool flash, int64_t S, int64_t S_kv, int64_t B,
                        int64_t Nh, int64_t Nkv, int64_t D, ggml_backend_t be,
                        std::string & err) {
    const float scale = 1.0f / std::sqrt((float) D);

    const size_t mem = ggml_tensor_overhead() * 8192 +
                       ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true) + (1u << 20);
    ggml_init_params ip = {};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    a.ctx         = ggml_init(ip);
    if (!a.ctx) {
        err = "ggml_init failed";
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, D, S,    Nh,  B);
    ggml_tensor * k = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * v = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, D, S_kv, Nkv, B);
    ggml_tensor * w = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, D, Nh,   S,   B);
    ggml_set_name(q, "q");
    ggml_set_name(k, "k");
    ggml_set_name(v, "v");
    ggml_set_name(w, "w");
    ggml_set_param(q);
    ggml_set_param(k);
    ggml_set_param(v);
    ggml_set_input(w);

    ggml_tensor * m = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F16, S_kv, S, 1, 1);
    ggml_set_name(m, "mask");
    ggml_set_input(m);

    ggml_tensor * o;
    if (flash) {
        ggml_tensor * pkd = ggml_flash_attn_train(a.ctx, q, k, v, m, scale);
        o                 = ggml_flash_attn_train_get_o(a.ctx, pkd);
    } else {
        o = ref_attn_f32(a.ctx, q, k, v, m, scale);
    }
    ggml_set_name(o, "o");

    ggml_tensor * loss = ggml_sum(a.ctx, ggml_mul(a.ctx, o, w));
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);

    a.gf = ggml_new_graph_custom(a.ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(a.gf, loss);
    ggml_build_backward_expand(a.ctx, a.gf, nullptr);

    a.buf = ggml_backend_alloc_ctx_tensors(a.ctx, be);
    if (!a.buf) {
        err = "allocation failed (OOM at this S)";
        return false;
    }

    // Values don't matter for a timing run (NaN could theoretically hit a
    // slow denormal path in some kernels, so fill with real numbers anyway,
    // same generator gate 1 uses — never time-seeded).
    std::vector<float> hq((size_t) (D * S * Nh * B)), hk((size_t) (D * S_kv * Nkv * B)),
        hv((size_t) (D * S_kv * Nkv * B)), hw((size_t) (D * Nh * S * B));
    fill(hq, 0xB0001001u + (uint64_t) S);
    fill(hk, 0xB0001002u + (uint64_t) S);
    fill(hv, 0xB0001003u + (uint64_t) S);
    fill(hw, 0xB0001004u + (uint64_t) S);
    std::vector<uint16_t> hm;
    dit_sa_mask((int) S, 128, &hm);   // bench grid keeps S_kv == S (self-attn shape)

    ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, hk.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, hv.data(), 0, ggml_nbytes(v));
    ggml_backend_tensor_set(w, hw.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(m, hm.data(), 0, ggml_nbytes(m));

    // ggml_set_loss only allocates grad_accs[loss]; it never initialises it
    // (same trap arm_run guards against above).
    ggml_tensor * lacc = ggml_graph_get_grad_acc(a.gf, loss);
    if (lacc) {
        const float one = 1.0f;
        ggml_backend_tensor_set(lacc, &one, 0, sizeof(float));
    }
    return true;
}

struct BenchResult {
    bool        built  = false;   // graph construction + allocation succeeded
    bool        ok     = false;   // built AND ran clean through the timed loop
    double      ms      = 0.0;    // ms/iter over the 50 timed iterations
    size_t      buf     = 0;      // backend buffer bytes (VRAM high-water)
    std::string note;
};

static BenchResult bench_run(bool flash, int64_t S, int64_t S_kv, int64_t B, int64_t Nh,
                             int64_t Nkv, int64_t D, ggml_backend_t be) {
    BenchResult r;
    BenchArm    a;
    std::string err;
    if (!bench_build(a, flash, S, S_kv, B, Nh, Nkv, D, be, err)) {
        r.note = err;
        return r;
    }
    r.built = true;
    r.buf   = a.buf ? ggml_backend_buffer_get_size(a.buf) : 0;

    for (int i = 0; i < 10; i++) {
        if (ggml_backend_graph_compute(be, a.gf) != GGML_STATUS_SUCCESS) {
            r.note = "graph compute failed during warm-up";
            return r;
        }
    }
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < 50; i++) {
        if (ggml_backend_graph_compute(be, a.gf) != GGML_STATUS_SUCCESS) {
            r.note = "graph compute failed during timed run";
            return r;
        }
    }
    const int64_t t1 = ggml_time_us();
    r.ms            = (double) (t1 - t0) / 1000.0 / 50.0;
    r.ok            = true;
    return r;
}

static int run_bench(ggml_backend_t be_cuda) {
    const int64_t S_list[] = { 625, 1250, 3000 };
    const int64_t B = 1, Nh = 32, Nkv = 8, D = 128;

    printf("fattn-train-test --bench — FORWARD+BACKWARD timing on %s\n",
           ggml_backend_name(be_cuda));
    printf("B=%lld Nh=%lld Nkv=%lld D=%lld  window mask  10 warm-up + 50 timed iters\n\n",
           (long long) B, (long long) Nh, (long long) Nkv, (long long) D);
    printf("%-6s %-8s %12s %10s   %12s %10s\n",
           "S", "arm", "ms/iter", "bufMB", "status", "");

    for (size_t si = 0; si < sizeof(S_list) / sizeof(S_list[0]); si++) {
        const int64_t S = S_list[si];
        for (int arm = 0; arm < 2; arm++) {
            const bool  flash = (arm == 1);
            const char * name = flash ? "fused" : "manual";
            const BenchResult r = bench_run(flash, S, S, B, Nh, Nkv, D, be_cuda);
            if (!r.built) {
                printf("%-6lld %-8s %12s %10s   %12s %s\n",
                       (long long) S, name, "n/a", "n/a", "SKIP", r.note.c_str());
            } else if (!r.ok) {
                printf("%-6lld %-8s %12s %10.1f   %12s %s\n",
                       (long long) S, name, "n/a", (double) r.buf / (1024.0 * 1024.0),
                       "FAIL", r.note.c_str());
            } else {
                printf("%-6lld %-8s %12.3f %10.1f   %12s\n",
                       (long long) S, name, r.ms, (double) r.buf / (1024.0 * 1024.0), "ok");
            }
            fflush(stdout);
        }
    }
    printf("\nbufMB is each arm's ggml backend buffer (VRAM) high-water — one allocation\n"
           "per tensor (ggml-alloc reuse is off), so it over-states what the same graph\n"
           "costs inside the trainer, same convention as the flashMB column above.\n");
    return 0;
}

// ─── one case ───────────────────────────────────────────────────────────────

static Result run_case(const Case & c, ggml_backend_t be_ref, ggml_backend_t be_flash, bool det) {
    Result r;

    if (!probe_supports(be_flash, c, /*back =*/ false)) {
        r.note = "flash backend does not support GGML_OP_FLASH_ATTN_TRAIN";
        return r;
    }
    // Forward-only when the backward has no kernel on this backend — probed,
    // never assumed. Fully-masked query rows are forward-only whatever the
    // backend: the reference chain emits NaN there (spec 4.4), the loss is a
    // sum over O, and one NaN in the loss makes every reference gradient NaN.
    const bool with_grads = !g_fwd_only &&
                            c.mask != MASK_DEAD_ROW &&
                            probe_supports(be_flash, c, /*back =*/ true);
    r.grads = with_grads;

    const int64_t nq = c.D * c.S    * c.Nh  * c.B;
    const int64_t nk = c.D * c.S_kv * c.Nkv * c.B;
    const int64_t no = c.D * c.Nh   * c.S   * c.B;

    std::vector<float> hq((size_t) nq), hk((size_t) nk), hv((size_t) nk), hw((size_t) no);
    fill(hq, 0x51ED0001u + (uint64_t) c.S * 131 + (uint64_t) c.B);
    fill(hk, 0x51ED0002u + (uint64_t) c.S * 131 + (uint64_t) c.B);
    fill(hv, 0x51ED0003u + (uint64_t) c.S * 131 + (uint64_t) c.B);
    fill(hw, 0x51ED0004u + (uint64_t) c.S * 131 + (uint64_t) c.B);

    // ── mask, straight out of the trainer's own builders ──
    std::vector<uint16_t> hm;
    std::vector<int>      valid_S((size_t) c.B, (int) c.S);
    std::vector<float>    enc_keep;
    int64_t               mask_ne1 = c.S;
    int64_t               mask_ne3 = 1;

    std::vector<int64_t>  dead_rows;
    if (c.mask == MASK_DEAD_ROW) {
        dead_rows.push_back(0);
        dead_rows.push_back(c.S / 2);
        dead_rows.push_back(c.S - 1);
    }

    if (c.mask == MASK_WIN) {
        dit_sa_mask((int) c.S, 128, &hm);
    } else if (c.mask == MASK_WIN_PAD) {
        // per-element pad: element 0 is padded to 3/4, the rest are full.
        mask_ne3 = c.B;
        hm.resize((size_t) (c.S_kv * c.S * c.B));
        for (int64_t b = 0; b < c.B; b++) {
            const int vS = (b == 0) ? (int) (c.S * 3 / 4) : (int) c.S;
            valid_S[(size_t) b] = vS;
            std::vector<uint16_t> one;
            dit_sa_mask((int) c.S, 128, &one, vS);
            memcpy(hm.data() + (size_t) b * (size_t) (c.S_kv * c.S), one.data(),
                   one.size() * sizeof(uint16_t));
        }
        // §7.3.3a: the pad columns are only reachable from padded QUERY rows,
        // whose loss weight the trainer drives to zero. Reproduce that, so
        // "masked positions contribute exactly zero" is actually assertable on
        // dK/dV pad columns.
        for (int64_t b = 0; b < c.B; b++) {
            for (int64_t s = valid_S[(size_t) b]; s < c.S; s++) {
                for (int64_t h = 0; h < c.Nh; h++) {
                    memset(hw.data() + (size_t) (c.D * (h + c.Nh * (s + c.S * b))), 0,
                           (size_t) c.D * sizeof(float));
                }
            }
        }
    } else if (c.mask == MASK_CA_COL) {
        // dit_ca_mask masks a key column for EVERY query — the only mask shape
        // that produces a genuinely dead column (dit_sa_mask deliberately can
        // not, spec §7.3.3).
        enc_keep.assign((size_t) c.S_kv, 1.0f);
        for (int64_t j = 3; j < c.S_kv; j += 7) {
            enc_keep[(size_t) j] = 0.0f;
        }
        dit_ca_mask((int) c.S_kv, (int) c.S, enc_keep, &hm);
    } else if (c.mask == MASK_DEAD_ROW) {
        // Hand-built, because neither dit_sa_mask nor dit_ca_mask can produce
        // one: both are written precisely so soft_max never sees an all--INF
        // row (dit-data.h:516-524). Our ops define the row instead of
        // inheriting the NaN, and this is the only case that checks it.
        hm.assign((size_t) (c.S_kv * c.S), ggml_fp32_to_fp16(0.0f));
        const uint16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (size_t di = 0; di < dead_rows.size(); di++) {
            const int64_t qi = dead_rows[di];
            for (int64_t j = 0; j < c.S_kv; j++) {
                hm[(size_t) (qi * c.S_kv + j)] = ninf;
            }
        }
    }

    const int64_t t0 = ggml_time_us();

    std::vector<float> ro, rdq, rdk, rdv;
    std::vector<float> fo, fdq, fdk, fdv, flse;
    {
        // The reference arm ALWAYS runs on the CPU backend: it is the oracle,
        // and running it beside the thing it is checking would be circular.
        // c.ref_fused swaps WHICH oracle — the autodiff'd dit_attn_f32 chain,
        // or the CPU fused impl that ops.cpp calls the behavioural reference —
        // never which backend it runs on.
        Arm ref;
        if (!arm_run(ref, c.ref_fused, with_grads, c, mask_ne1, mask_ne3, hq, hk, hv, hw, hm,
                     be_ref, ro, rdq, rdk, rdv)) {
            r.note = "reference arm failed";
            return r;
        }
        r.ref_buf = ref.buf ? ggml_backend_buffer_get_size(ref.buf) : 0;
    }
    {
        Arm fl;
        if (!arm_run(fl, true, with_grads, c, mask_ne1, mask_ne3, hq, hk, hv, hw, hm,
                     be_flash, fo, fdq, fdk, fdv, &flse)) {
            r.note = "flash arm failed";
            return r;
        }
        r.flash_buf      = fl.buf ? ggml_backend_buffer_get_size(fl.buf) : 0;
        r.flash_dev_free = fl.dev_free_at_peak;
    }

    r.ms = (double) (ggml_time_us() - t0) / 1000.0;

    // ── LSE must be finite everywhere (spec §4.4, §7.3.4) ──
    for (size_t i = 0; i < flse.size(); i++) {
        if (!std::isfinite(flse[i])) {
            char buf[128];
            snprintf(buf, sizeof(buf), "LSE[%zu] is not finite (%g)", i, (double) flse[i]);
            r.note = buf;
            return r;
        }
    }

    // ── determinism: the same inputs on the same device, twice, bitwise ──
    //
    // The backward is where this actually bites. An atomicAdd-based dQ would
    // still pass every tolerance above and still make an A/B against
    // --attn exact uninterpretable, so the second run compares the WHOLE packed
    // output — O, LSE, dQ, dK, dV — byte for byte, not just the forward.
    if (det) {
        std::vector<float> fo2, fdq2, fdk2, fdv2, flse2;
        Arm fl2;
        if (!arm_run(fl2, true, with_grads, c, mask_ne1, mask_ne3, hq, hk, hv, hw, hm,
                     be_flash, fo2, fdq2, fdk2, fdv2, &flse2)) {
            r.note = "flash arm (second run) failed";
            return r;
        }
        auto same = [](const std::vector<float> & a, const std::vector<float> & b) {
            return a.size() == b.size() &&
                   (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        };
        const char * which = nullptr;
        if      (!same(fo2,   fo))   { which = "O";   }
        else if (!same(flse2, flse)) { which = "LSE"; }
        else if (!same(fdq2,  fdq))  { which = "dQ";  }
        else if (!same(fdk2,  fdk))  { which = "dK";  }
        else if (!same(fdv2,  fdv))  { which = "dV";  }
        if (which) {
            char buf[64];
            snprintf(buf, sizeof(buf), "FAIL(%s)", which);
            r.det  = buf;
            r.note = "not deterministic across two runs";
            return r;
        }
        r.det = with_grads ? "bitwise" : "bitwise(fwd)";
    }

    // ── fully-masked query rows: checked DIRECTLY, never against the
    //    reference (spec 4.4 / 7.3.9). dit_attn_f32 is specified to produce
    //    NaN on such a row; ours is specified to produce O = 0 and LSE = 0.
    //    A NaN-vs-0 mismatch there would be the correct behaviour failing.
    if (c.mask == MASK_DEAD_ROW) {
        for (size_t di = 0; di < dead_rows.size(); di++) {
            const int64_t sdx = dead_rows[di];
            for (int64_t b = 0; b < c.B; b++) {
                for (int64_t hh = 0; hh < c.Nh; hh++) {
                    const size_t obase = (size_t) (c.D * (hh + c.Nh * (sdx + c.S * b)));
                    for (int64_t d = 0; d < c.D; d++) {
                        if (fo[obase + (size_t) d] != 0.0f) {
                            char buf[160];
                            snprintf(buf, sizeof(buf),
                                     "fully-masked row %lld: O[h=%lld,d=%lld,b=%lld] = %g, want 0",
                                     (long long) sdx, (long long) hh, (long long) d,
                                     (long long) b, (double) fo[obase + (size_t) d]);
                            r.note     = buf;
                            r.maskzero = "FAIL(dead-row)";
                            return r;
                        }
                        // the reference is NaN here by construction; neutralise
                        // both sides so the REST of the tensor still compares
                        ro[obase + (size_t) d] = 0.0f;
                    }
                    const size_t lidx = (size_t) (hh + c.Nh * (sdx + c.S * b));
                    if (lidx < flse.size() && flse[lidx] != 0.0f) {
                        char buf[160];
                        snprintf(buf, sizeof(buf),
                                 "fully-masked row %lld: LSE[h=%lld,b=%lld] = %g, want 0",
                                 (long long) sdx, (long long) hh, (long long) b,
                                 (double) flse[lidx]);
                        r.note     = buf;
                        r.maskzero = "FAIL(dead-row)";
                        return r;
                    }
                }
            }
        }
        r.maskzero = "ok(dead-row)";
    }

    // 7.3.1a — a green comparison against all-zero references proves nothing.
    if (with_grads &&
        (max_abs(rdq) == 0.0f || max_abs(rdk) == 0.0f || max_abs(rdv) == 0.0f)) {
        r.note = "reference gradients are all zero (vacuous)";
        return r;
    }

    bool fin = true, f2 = true, f3 = true, f4 = true;
    // Triage is single-threaded and O(S**2 * Nh * D): ~74 GFLOP at S = 3000,
    // i.e. minutes for a diagnostic nobody has asked for yet. Above the grid's
    // own sizes the two fused arms are the diagnosis.
    if (c.S <= 1500) {
        // Printed only when something is off: a third, dependency-free host
        // implementation says WHICH arm drifted rather than just that the two
        // disagree. It earned its keep — the first two red runs were both
        // harness faults (ggml-alloc buffer reuse, then a transposed-view
        // read-back of dV), not op faults.
        std::vector<float> ho, hdv;
        host_attn(c, hq, hk, hv, hm, mask_ne1, mask_ne3, hw, ho, hdv);
        bool hf1 = true, hf2 = true, hf3 = true, hf4 = true;
        const float ref_o    = rel_err(ro,  ho,  &hf1);
        const float flash_o  = rel_err(fo,  ho,  &hf2);
        const float ref_dv   = rel_err(rdv, hdv, &hf3);
        const float flash_dv = rel_err(fdv, hdv, &hf4);
        if (ref_o > PASS_REL || flash_o > PASS_REL || ref_dv > PASS_REL || flash_dv > PASS_REL) {
            fprintf(stderr, "[triage] %s vs host:  O ref %.3e flash %.3e   dV ref %.3e flash %.3e\n",
                    c.name.c_str(), (double) ref_o, (double) flash_o,
                    (double) ref_dv, (double) flash_dv);
        }
    }
    r.fwd_rel = rel_err(fo, ro, &fin);
    if (with_grads) {
        r.dq_rel = rel_err(fdq, rdq, &f2);
        r.dk_rel = rel_err(fdk, rdk, &f3);
        r.dv_rel = rel_err(fdv, rdv, &f4);
    }
    if (!fin || !f2 || !f3 || !f4) {
        r.note = "non-finite value";
        return r;
    }

    // ── masked positions contribute EXACTLY 0.0f ──
    auto col_is_zero = [&](int64_t j, int64_t b) -> bool {
        for (int64_t hk2 = 0; hk2 < c.Nkv; hk2++) {
            const size_t off = (size_t) (c.D * (j + c.S_kv * (hk2 + c.Nkv * b)));
            for (int64_t d = 0; d < c.D; d++) {
                const float a = fdk[off + (size_t) d];
                const float bb = fdv[off + (size_t) d];
                if (a != 0.0f || bb != 0.0f || std::isnan(a) || std::isnan(bb)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (c.mask == MASK_DEAD_ROW) {
        // already asserted above, directly
    } else if (!with_grads) {
        r.maskzero = "n/a(fwd)";
    } else if (c.mask == MASK_WIN_PAD) {
        bool ok = true;
        int64_t bad_j = -1, bad_b = -1;
        for (int64_t b = 0; b < c.B && ok; b++) {
            for (int64_t j = valid_S[(size_t) b]; j < c.S_kv; j++) {
                if (!col_is_zero(j, b)) {
                    ok = false; bad_j = j; bad_b = b;
                    break;
                }
            }
        }
        if (ok) {
            r.maskzero = "ok(pad-col)";
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "FAIL(pad-col j=%lld b=%lld)", (long long) bad_j, (long long) bad_b);
            r.maskzero = buf;
            r.note     = "masked pad column has non-zero gradient";
            return r;
        }
    } else if (c.mask == MASK_CA_COL) {
        bool ok = true;
        int64_t bad_j = -1;
        for (int64_t j = 0; j < c.S_kv && ok; j++) {
            if (enc_keep[(size_t) j] > 0.5f) {
                continue;
            }
            for (int64_t b = 0; b < c.B; b++) {
                if (!col_is_zero(j, b)) {
                    ok = false; bad_j = j;
                    break;
                }
            }
        }
        if (ok) {
            r.maskzero = "ok(dead-col)";
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "FAIL(dead-col j=%lld)", (long long) bad_j);
            r.maskzero = buf;
            r.note     = "masked key column has non-zero gradient";
            return r;
        }
    }

    r.ok = (r.fwd_rel <= PASS_REL) &&
           (!with_grads || ((r.dq_rel <= PASS_REL) &&
                            (r.dk_rel <= PASS_REL) &&
                            (r.dv_rel <= PASS_REL)));
    if (!r.ok && r.note.empty()) {
        r.note = "tolerance";
    }
    // Say so in the table rather than in a footnote: a row whose oracle is the
    // CPU fused impl is a weaker claim than one measured against the autodiff'd
    // chain, and it should read that way.
    if (c.ref_fused) {
        r.note = r.note.empty() ? "ref = CPU fused impl"
                                : (r.note + " (ref = CPU fused impl)");
    }
    return r;
}

// ─── grid ───────────────────────────────────────────────────────────────────

static const char * yn(bool b) {
    return b ? "yes" : "no";
}

// %10s-wide cell that says "n/a" rather than printing a meaningless 0.000e+00
// when a quantity was never compared.
static std::string cell(bool have, float x) {
    char b[32];
    if (have) {
        snprintf(b, sizeof(b), "%10.3e", (double) x);
    } else {
        snprintf(b, sizeof(b), "%10s", "n/a");
    }
    return std::string(b);
}

int main(int argc, char ** argv) {
    bool        cheap = false;
    bool        quick = false;
    bool        extra = false;
    bool        large = false;
    bool        bench = false;
    int         nth   = 0;
    std::string want_backend = "cpu";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--cheap") {
            cheap = true;
        } else if (a == "--fwd-only") {
            g_fwd_only = true;
        } else if (a == "--quick") {
            quick = true;
        } else if (a == "--extra") {
            extra = true;
        } else if (a == "--large") {
            large = true;
        } else if (a == "--bench") {
            bench = true;
        } else if (a == "--threads" && i + 1 < argc) {
            nth = atoi(argv[++i]);
        } else if (a == "--backend" && i + 1 < argc) {
            want_backend = argv[++i];
            if (want_backend != "cpu" && want_backend != "cuda") {
                fprintf(stderr, "[fattn-train-test] --backend takes cpu or cuda\n");
                return 2;
            }
        } else {
            fprintf(stderr, "usage: fattn-train-test [--backend cpu|cuda] [--extra] [--large]"
                            " [--cheap] [--quick] [--fwd-only] [--threads N] [--bench]\n");
            return 2;
        }
    }

    ggml_time_init();

    if (bench) {
        // --bench times the CUDA backend only, both arms — the question is
        // "fused vs manual on the GPU", not CPU-vs-CUDA (that is what
        // --backend cuda without --bench already answers). --backend is
        // ignored in this mode.
        ggml_backend_load_all();
        ggml_backend_t be_cuda = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!be_cuda) {
            fprintf(stderr, "[fattn-train-test] --bench requires a CUDA backend "
                            "(is ggml-cuda.dll beside the exe?)\n");
            return 2;
        }
        const int rc = run_bench(be_cuda);
        ggml_backend_free(be_cuda);
        return rc;
    }

    // CPU backend via the registry API (the CPU backend ships as a loadable
    // module, so ggml_backend_cpu_init is not linkable here) — same shape as
    // engine/src/backend.h's cpu_backend_new.
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     be  = dev ? ggml_backend_dev_init(dev, nullptr)
                                 : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) {
        fprintf(stderr, "[fattn-train-test] no CPU backend\n");
        return 2;
    }

    // Arm B's backend. Arm A stays on `be` (CPU) whatever this is.
    ggml_backend_t be_flash = be;
    if (want_backend == "cuda") {
        be_flash = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!be_flash) {
            fprintf(stderr, "[fattn-train-test] no GPU backend available "
                            "(is ggml-cuda.dll beside the exe?)\n");
            ggml_backend_free(be);
            return 2;
        }
    }
    const bool cuda_mode = (be_flash != be);
    if (nth <= 0) {
        nth = (int) std::thread::hardware_concurrency();
        if (nth <= 0) {
            nth = 4;
        }
    }
    {
        ggml_backend_dev_t d   = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = d ? ggml_backend_dev_backend_reg(d) : nullptr;
        if (reg) {
            auto set_fn = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_fn) {
                set_fn(be, nth);
            }
        }
    }

    const int64_t Nh = 32;
    const int64_t D  = 128;

    // GQA at B > 1 is NOT testable through the reference: ggml's mul_mat
    // backward asserts tmp->ne[3] == 1 when it folds a broadcast src0
    // (ggml.c, GGML_OP_MUL_MAT case), i.e. it aborts for B > 1 with Nkv < Nh.
    // That is exactly why the trainer pre-expands K/V with dit_expand_heads at
    // B > 1 (spec §1.1), so B = 2 is tested at the shape the trainer really
    // sends (Nkv == Nh) and GQA 32/8 is tested at B = 1, where it is real.
    const int64_t S_list[] = { 64, 129, 198, 384, 1000 };

    std::vector<Case> cases;
    for (size_t si = 0; si < sizeof(S_list) / sizeof(S_list[0]); si++) {
        const int64_t S = S_list[si];
        if (quick && S > 198) {
            continue;
        }
        for (int64_t B = 1; B <= 2; B++) {
            const int64_t Nkv = (B == 1) ? 8 : Nh;
            for (int mk = MASK_NONE; mk <= MASK_WIN_PAD; mk++) {
                // The plan doc allows dropping S = 1000 to its two cheapest
                // mask cases if the grid runs long. It does not — the whole
                // grid is ~8 s on 32 threads, the S = 1000 null mask being the
                // slowest cell at 1.7 s — so nothing is dropped. `--cheap`
                // drops it anyway, for a slower box.
                if (cheap && S >= 1000 && mk == MASK_NONE) {
                    continue;
                }
                Case c;
                c.S = S; c.S_kv = S; c.B = B; c.Nh = Nh; c.Nkv = Nkv; c.D = D;
                c.mask = (MaskKind) mk;
                char nm[96];
                snprintf(nm, sizeof(nm), "S%lld/B%lld/%s", (long long) S, (long long) B, mask_name(c.mask));
                c.name = nm;
                cases.push_back(c);
            }
            // dead-column case: the only shape that makes "masked positions
            // contribute exactly zero" directly assertable (spec §7.3.3).
            if (S <= 198) {
                Case c;
                c.S = S; c.S_kv = S; c.B = B; c.Nh = Nh; c.Nkv = Nkv; c.D = D;
                c.mask = MASK_CA_COL;
                char nm[96];
                snprintf(nm, sizeof(nm), "S%lld/B%lld/%s", (long long) S, (long long) B, mask_name(c.mask));
                c.name = nm;
                cases.push_back(c);
            }
        }
    }

    if (extra) {
        // S_kv != S — the cross-attention shape (spec 0.1 / 7.2). The trainer
        // does not call the op there yet, but an untested S_kv != S turns that
        // flip into a rewrite rather than a flag.
        const int64_t xs[2][2] = { { 384, 769 }, { 65, 200 } };
        for (int xi = 0; xi < 2; xi++) {
            for (int64_t B = 1; B <= 2; B++) {
                for (int mk = 0; mk < 2; mk++) {
                    Case c;
                    c.S    = xs[xi][0];
                    c.S_kv = xs[xi][1];
                    c.B    = B;
                    c.Nh   = Nh;
                    c.Nkv  = (B == 1) ? 8 : Nh;
                    c.D    = D;
                    c.mask = (mk == 0) ? MASK_NONE : MASK_CA_COL;
                    char nm[96];
                    snprintf(nm, sizeof(nm), "S%lld:%lld/B%lld/%s", (long long) c.S,
                             (long long) c.S_kv, (long long) B, mask_name(c.mask));
                    c.name = nm;
                    cases.push_back(c);
                }
            }
        }
        // fully-masked query rows (spec 4.4 / 7.3.9)
        const int64_t ds[2] = { 64, 198 };
        for (int di = 0; di < 2; di++) {
            for (int64_t B = 1; B <= 2; B++) {
                Case c;
                c.S = ds[di]; c.S_kv = ds[di]; c.B = B; c.Nh = Nh;
                c.Nkv = (B == 1) ? 8 : Nh; c.D = D; c.mask = MASK_DEAD_ROW;
                char nm[96];
                snprintf(nm, sizeof(nm), "S%lld/B%lld/%s", (long long) c.S,
                         (long long) B, mask_name(c.mask));
                c.name = nm;
                cases.push_back(c);
            }
        }
    }

    if (large) {
        // The case the project exists for. Production geometry at a crop the
        // exact path cannot reach: at S = 3000 the retained self-attention
        // softmax alone is 1.15 GB per layer (spec §0.1), which is why the
        // reference chain is NOT the oracle here — see Case::ref_fused.
        Case c;
        c.S    = 3000;
        c.S_kv = 3000;
        c.B    = 1;
        c.Nh   = Nh;
        c.Nkv  = 8;         // GQA 4:1, the real trainer shape at B = 1
        c.D    = D;
        c.mask = MASK_WIN;  // sliding_window = 128, i.e. the tile skip matters
        c.ref_fused = true;
        c.name = "S3000/B1/window(large)";
        cases.push_back(c);
    }

    printf("fattn-train-test — GGML_OP_FLASH_ATTN_TRAIN{,_BACK} vs autodiff\'d dit_attn_f32\n");
    printf("reference arm: %s (%d threads)   flash arm: %s\n",
           ggml_backend_name(be), nth, ggml_backend_name(be_flash));
    if (!cases.empty()) {
        printf("flash-arm supports_op:  FLASH_ATTN_TRAIN %s   FLASH_ATTN_TRAIN_BACK %s"
               "   (probed, not assumed)\n",
               yn(probe_supports(be_flash, cases[0], false)),
               yn(probe_supports(be_flash, cases[0], true)));
    }
    printf("Nh=%lld D=%lld   tol=%.0e   cases=%zu   determinism check: %s\n\n",
           (long long) Nh, (long long) D, (double) PASS_REL, cases.size(), yn(cuda_mode));
    printf("%-26s %5s %5s %3s %4s %11s  %10s %10s %10s %10s  %-14s %-12s %8s %8s  %s\n",
           "case", "S", "S_kv", "B", "Nkv", "mask", "fwd_rel", "dQ_rel", "dK_rel", "dV_rel",
           "masked-zero", "det", "ms", "flashMB", "verdict");

    // Device-level free memory, sampled once before anything is allocated, so
    // the --large line can quote a real high-water and not just what ggml asked
    // for. Zero on backends that do not report it.
    size_t dev_free0 = 0, dev_total = 0;
    {
        ggml_backend_dev_t fd = ggml_backend_get_device(be_flash);
        if (fd) {
            ggml_backend_dev_memory(fd, &dev_free0, &dev_total);
        }
    }

    int    failed    = 0;
    size_t peak_buf  = 0;
    Result large_res;
    for (size_t i = 0; i < cases.size(); i++) {
        const Case & c = cases[i];
        const Result r = run_case(c, be, be_flash, cuda_mode);
        if (!r.ok) {
            failed++;
        }
        if (r.flash_buf > peak_buf) {
            peak_buf = r.flash_buf;
        }
        if (c.ref_fused && c.S >= 3000) {
            large_res = r;
        }
        printf("%-26s %5lld %5lld %3lld %4lld %11s  %10.3e %s %s %s  %-14s %-12s %8.1f %8.1f  %s%s%s\n",
               c.name.c_str(), (long long) c.S, (long long) c.S_kv, (long long) c.B,
               (long long) c.Nkv, mask_name(c.mask),
               (double) r.fwd_rel,
               cell(r.grads, r.dq_rel).c_str(),
               cell(r.grads, r.dk_rel).c_str(),
               cell(r.grads, r.dv_rel).c_str(),
               r.maskzero.c_str(), r.det.c_str(), r.ms,
               (double) r.flash_buf / (1024.0 * 1024.0),
               r.ok ? "PASS" : "FAIL",
               r.note.empty() ? "" : " — ", r.note.c_str());
        fflush(stdout);
    }

    printf("\npeak flash-arm backend buffer: %.1f MB  (one allocation per tensor;"
           " ggml-alloc reuse is off, so this over-states a real graph)\n",
           (double) peak_buf / (1024.0 * 1024.0));

    if (large && large_res.flash_buf) {
        ggml_backend_dev_t fd = ggml_backend_get_device(be_flash);
        printf("--large  S=3000 S_kv=3000 B=1 Nh=32 Nkv=8 D=128 window mask:\n");
        printf("         flash arm (%s) backend buffers  %.1f MB\n",
               ggml_backend_name(be_flash), (double) large_res.flash_buf / (1024.0 * 1024.0));
        printf("         CPU fused reference arm          %.1f MB (host)\n",
               (double) large_res.ref_buf / (1024.0 * 1024.0));
        if (dev_total && large_res.flash_dev_free) {
            // Sampled with the arm's buffers still live, so it includes the
            // CUDA pool scratch the backward's delta kernel allocates.
            printf("         device %s: %.0f MB total\n",
                   fd ? ggml_backend_dev_description(fd) : "?",
                   (double) dev_total / (1024.0 * 1024.0));
            printf("         device free at tool start        %.0f MB\n",
                   (double) dev_free0 / (1024.0 * 1024.0));
            printf("         device free at the large peak    %.0f MB\n",
                   (double) large_res.flash_dev_free / (1024.0 * 1024.0));
            printf("         => device high-water for the case %.0f MB"
                   " (buffers + backend scratch + context)\n",
                   ((double) dev_free0 - (double) large_res.flash_dev_free) / (1024.0 * 1024.0));
        }
    }

    printf("\n%s — %d/%zu cases passed\n", failed == 0 ? "PASS" : "FAIL",
           (int) cases.size() - failed, cases.size());

    if (be_flash != be) {
        ggml_backend_free(be_flash);
    }
    ggml_backend_free(be);
    return failed == 0 ? 0 : 1;
}
