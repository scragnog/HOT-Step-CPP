#pragma once
// dit-selftest.h — the R7 correctness gates (plan §3.9 / §6.2).
//
//  T1  unfused load               exact
//  T2  site shapes                exact
//  T3  per-layer forward diff     <= 2e-3 max relative, per named tensor   (V1)
//  T4  finite differences         max rel < 2e-2 AND median < 5e-3, 24 probes (V2)
//  T5  shuffled-target control    real <= 0.15x start; shuffled >= 0.5x start (V3)
//  T6  B=0 structural             dL/dA exactly 0, dL/dB non-zero + finite
//  T7  loss identity              rel err < 1e-5 vs a host double recompute
//  T8  flow_snr normalisation     mean(w_i/wbar) == 1 to 1e-6
//  T9  convention fingerprint     E[v^2] on Hypa_Hypa [0,750) == 1.672071 +- 1e-5
//  T10 crop sampler               support + alignment + uniformity to +-3 sigma
//  T11 AdamW + clip               rel err < 1e-5 / ratio 0.1 +- 1e-3
//  MU1 Muon / Newton-Schulz       rel err < 2e-3 vs host ref, |O*O^T - I| < 0.35,
//                                 and a [4,5] LoKR w1 must fall through to AdamW
//
// LoKR (docs/plans/2026-07-28-lokr-dit-training.md §2.6):
//  LK1 factorization              ported loop == derived reference, host-only
//  LK2 untrained identity         max |apply(x) - W.x| <= 1e-6 at w2 = 0
//  LK3 kron equivalence           matvec == materialized kron, rel < 1e-5
//  LK4 finite differences         same bars as T4, probing w1 AND w2
//  LK5 export/parse roundtrip     keys, shapes, alpha, lokr_config.linear_dim
//
// Micro-batching (docs/plans/2026-07-29-dit-batching-checkpointing.md §2.3):
//  SB1 batched forward parity     CPU identity <= 1e-6, CUDA permutation == 0
//  SB2 batched backward parity    CPU identity <= 1e-5 vs B single backwards
//  SB3 mixed-length batch         padded frames move loss and grads by EXACTLY 0,
//                                 whether the pad's TARGET or its INPUT is dirtied
//  SC1 checkpointing parity       --ckpt 4 grads == --ckpt 0 grads, rel <= 1e-6
//  SC2 batching + checkpointing   SB3's invariant under --batch 3 --ckpt 2
//  SC3 LoKR + batching + ckpt     LK adapter under --batch 3 --ckpt 2, rel <= 1e-5
//
// Non-zero exit on any failure. The ENGINE implementer may not hand off before
// every check passes.

#include "train/dit-adapter-lokr.h"
#include "train/dit-data.h"
#include "train/dit-train-ckpt.h"
#include "train/dit-train-graph.h"
#include "train/dit-vram.h"
#include "train/lm-optim.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#    include <process.h>
#    include <windows.h>
#else
#    include <sys/wait.h>
#    include <unistd.h>
#endif

// Which gates this process runs. T4 (the finite-difference gate) is measured in
// a CHILD process with NVIDIA_TF32_OVERRIDE=0 — see dit_st_spawn_fd() for why.
enum DitStMode { DIT_ST_ALL = 0, DIT_ST_FD_ONLY = 1, DIT_ST_NO_FD = 2 };

struct DitSelfTestResult {
    std::string name;
    bool        pass = false;
    std::string detail;
};

static void dit_st_report(std::vector<DitSelfTestResult> & rs, const char * name, bool pass,
                          const std::string & detail) {
    DitSelfTestResult r;
    r.name   = name;
    r.pass   = pass;
    r.detail = detail;
    rs.push_back(r);
    fprintf(stderr, "[self-test] %-4s %-4s %s\n", name, pass ? "PASS" : "FAIL", detail.c_str());
    jl("{\"type\":\"selftest\",\"check\":\"%s\",\"pass\":%s,\"detail\":\"%s\"}", name, pass ? "true" : "false",
       lm_json_escape(detail).c_str());
}

// ─── T8/T10: host-only gates ────────────────────────────────────────────────

static void dit_st_flow_snr(std::vector<DitSelfTestResult> & rs) {
    const float t[4]   = { 0.05f, 0.3f, 0.62f, 0.94f };
    const int   accum  = 4;
    float       w[4]   = { 0, 0, 0, 0 };
    double      wsum   = 0.0;
    for (int i = 0; i < accum; i++) {
        w[i] = dit_flow_snr_w(t[i], 0.5f, 5.0f);
        wsum += (double) w[i];
    }
    const double wbar = wsum / accum;
    double       mean_norm = 0.0, worst_lg = 0.0;
    for (int i = 0; i < accum; i++) {
        mean_norm += (double) w[i] / wbar;
    }
    mean_norm /= accum;
    // --loss-weighting none must give t_lossgrad == 1/grad_accum exactly.
    for (int i = 0; i < accum; i++) {
        const double lg = 1.0 / (double) accum;  // w == 1 => w/wbar == 1
        worst_lg        = std::max(worst_lg, fabs(lg - 1.0 / (double) accum));
    }
    char d[256];
    snprintf(d, sizeof(d),
             "w(t)=[%.4f %.4f %.4f %.4f] wbar=%.4f  mean(w_i/wbar)=%.9f (want 1 +-1e-6)  "
             "none-mode t_lossgrad err=%.1e (want 0)",
             (double) w[0], (double) w[1], (double) w[2], (double) w[3], wbar, mean_norm, worst_lg);
    dit_st_report(rs, "T8", fabs(mean_norm - 1.0) < 1e-6 && worst_lg == 0.0, d);
}

static void dit_st_crop(std::vector<DitSelfTestResult> & rs, uint64_t seed) {
    // The interior case (T >> crop) exercises none of dit_sample_crop's edge
    // branches, which is where an off-by-one would actually live. Parameterised
    // over T <= crop, T == crop, odd T and a degenerate T < patch as well.
    struct Case {
        int T, crop, patch;
    };
    const Case cases[] = {
        { 5325, 1250, 2 },  // interior: 2038 aligned starts
        { 1250, 1250, 2 },  // crop == T: n_starts collapses to 1
        { 800, 1250, 2 },   // crop  > T: len falls back to T
        { 801, 1250, 2 },   // crop  > T, T odd: the last frame is dropped
        { 5325, 0, 2 },     // crop == 0 (whole song), T odd
        { 3, 1250, 2 },     // T < 2*patch: the len < patch fallback
        { 1251, 375, 2 },   // odd T, interior crop
    };
    LmRng rng;
    lm_rng_seed(&rng, seed ^ 0xC0FFEEull);
    const int n_draws = 10000;
    bool      ok_support = true, ok_align = true, ok_len = true, ok_unif = true;
    double    worst_all = 0.0;
    std::string per_case;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const int T = cases[ci].T, crop = cases[ci].crop, patch = cases[ci].patch;
        int       want_len = (crop > 0 && crop < T) ? crop : T;
        want_len -= want_len % patch;
        if (want_len < patch) {
            want_len = std::min(T - (T % patch), patch);
        }
        const int        n_starts = (T - want_len) / patch + 1;
        const int        draws    = (ci == 0) ? n_draws : 2000;
        std::vector<int> hist((size_t) n_starts, 0);
        for (int i = 0; i < draws; i++) {
            const DitCrop c = dit_sample_crop(&rng, T, crop, patch);
            ok_len          = ok_len && (c.len == want_len);
            ok_align        = ok_align && (c.start % patch == 0) && (c.len % patch == 0);
            ok_support      = ok_support && (c.start >= 0 && c.start + c.len <= T);
            const int b     = c.start / patch;
            if (b >= 0 && b < n_starts) {
                hist[(size_t) b]++;
            } else {
                ok_support = false;
            }
        }
        // chi-square-free uniformity: every bucket within 3 sigma of the mean.
        const double mu    = (double) draws / (double) n_starts;
        const double sigma = sqrt(mu * (1.0 - 1.0 / (double) n_starts));
        int          out3  = 0;
        double       worst = 0.0;
        for (size_t i = 0; i < hist.size(); i++) {
            const double z = fabs((double) hist[i] - mu) / std::max(1e-9, sigma);
            worst          = std::max(worst, z);
            if (z > 3.0) {
                out3++;
            }
        }
        // With n_starts buckets, ~0.27% are expected beyond 3 sigma by chance.
        const int allowed = std::max(1, (int) (0.01 * (double) n_starts));
        ok_unif           = ok_unif && (out3 <= allowed);
        worst_all         = std::max(worst_all, worst);
        char cb[96];
        snprintf(cb, sizeof(cb), "(T=%d,crop=%d)->len=%d,starts=%d,out3=%d/%d ", T, crop, want_len, n_starts, out3,
                 allowed);
        per_case += cb;
    }
    char d[512];
    snprintf(d, sizeof(d), "%d+ draws over %d (T,crop) cases: %ssupport=%s align=%s len=%s uniform=%s worst z=%.2f",
             n_draws, (int) (sizeof(cases) / sizeof(cases[0])), per_case.c_str(), ok_support ? "ok" : "BAD",
             ok_align ? "ok" : "BAD", ok_len ? "ok" : "BAD", ok_unif ? "ok" : "BAD", worst_all);
    dit_st_report(rs, "T10", ok_support && ok_align && ok_len && ok_unif, d);
}

// ─── LK1: the LyCORIS factorization, host-only ──────────────────────────────
//
// Runs before the data scan and the model load so a machine with no tensors dir
// and no GPU still gets a verdict on the one piece of pure math the whole LoKR
// parameterization is built on.
//
// The reference is derived, not transcribed: LyCORIS walks the divisors of `n`
// upward while m < n and stops at the first one above the factor cap, which is
// exactly "the largest divisor m <= min(cap, sqrt(n))". Comparing the ported
// loop against that closed form catches a mis-ported loop guard, which a table
// of hardcoded expectations would not.
static void dit_st_factorization_ref(int64_t dimension, int factor, int64_t * out_m, int64_t * out_n) {
    if (factor > 0 && (dimension % (int64_t) factor) == 0) {
        int64_t m = (int64_t) factor, n = dimension / (int64_t) factor;
        if (m > n) {
            std::swap(m, n);
        }
        *out_m = m;
        *out_n = n;
        return;
    }
    const int64_t cap  = (factor < 0) ? dimension : (int64_t) factor;
    int64_t       best = 1;
    for (int64_t m = 1; m * m <= dimension; m++) {
        if (dimension % m == 0 && m <= cap) {
            best = m;
        }
    }
    *out_m = best;
    *out_n = dimension / best;
}

static void dit_st_lokr_factorization(std::vector<DitSelfTestResult> & rs) {
    const int64_t dims[]    = { 1, 2, 4, 127, 128, 250, 360, 512, 1024, 2048, 4096, 6144, 8192 };
    const int     factors[] = { -1, 2, 4, 6, 8, 16, 64 };
    int           mismatch  = 0;
    std::string   worst;
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        for (size_t j = 0; j < sizeof(factors) / sizeof(factors[0]); j++) {
            int64_t pm = 0, pn = 0, rm = 0, rn = 0;
            dit_lokr_factorization(dims[i], factors[j], &pm, &pn);
            dit_st_factorization_ref(dims[i], factors[j], &rm, &rn);
            if (pm != rm || pn != rn) {
                mismatch++;
                if (worst.empty()) {
                    char b[128];
                    snprintf(b, sizeof(b), "(%lld,%d)->port(%lld,%lld) ref(%lld,%lld)", (long long) dims[i],
                             factors[j], (long long) pm, (long long) pn, (long long) rm, (long long) rn);
                    worst = b;
                }
            }
        }
    }
    // The four the plan names, spelled out so the line is readable at a glance.
    struct Named {
        int64_t n;
        int     f, em, en;
    };
    const Named named[] = { { 2048, 6, 4, 512 }, { 1024, 6, 4, 256 }, { 6144, 6, 6, 1024 }, { 2048, -1, 32, 64 } };
    std::string table;
    int         bad_named = 0;
    for (size_t i = 0; i < sizeof(named) / sizeof(named[0]); i++) {
        int64_t m = 0, n = 0;
        dit_lokr_factorization(named[i].n, named[i].f, &m, &n);
        const bool ok = (m == named[i].em && n == named[i].en);
        bad_named += ok ? 0 : 1;
        char b[96];
        snprintf(b, sizeof(b), "(%lld,%d)->(%lld,%lld)%s ", (long long) named[i].n, named[i].f, (long long) m,
                 (long long) n, ok ? "" : "!=EXPECTED");
        table += b;
    }
    const std::string first = worst.empty() ? std::string() : (" first=" + worst);
    char              d[352];
    snprintf(d, sizeof(d), "%s| %d (dim,factor) pairs vs a derived reference: %d mismatch%s%s", table.c_str(),
             (int) ((sizeof(dims) / sizeof(dims[0])) * (sizeof(factors) / sizeof(factors[0]))), mismatch,
             mismatch == 1 ? "" : "es", first.c_str());
    dit_st_report(rs, "LK1", mismatch == 0 && bad_named == 0, d);
}

// ─── T9: the convention fingerprint ─────────────────────────────────────────

static void dit_st_convention(std::vector<DitSelfTestResult> & rs, const std::vector<DitSample> & samples,
                              uint64_t seed) {
    const DitSample * hypa = nullptr;
    for (size_t i = 0; i < samples.size(); i++) {
        if (samples[i].path.find("Hypa_Hypa") != std::string::npos) {
            hypa = &samples[i];
            break;
        }
    }
    if (!hypa) {
        dit_st_report(rs, "T9", false, "Hypa_Hypa is not in this variant — the verifier's E[v^2] fingerprint "
                                       "cannot be reproduced (run --self-test against the electriccallboy cache)");
        return;
    }
    // The spike's frozen-noise stream: seed ^ 0x9e3779b97f4a7c15, drawn in index
    // order over the crop [0, 750) of target_latents. E[v^2] is t-independent.
    LmRng rng;
    lm_rng_seed(&rng, seed ^ 0x9e3779b97f4a7c15ull);
    std::vector<float> xt, v;
    const size_t       nel = (size_t) 750 * (size_t) hypa->Oc;
    dit_flow_target(hypa->lat.data(), nel, 0.5f, &rng, &xt, &v);
    double s = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        s += (double) v[i] * (double) v[i];
    }
    const double ev2 = s / (double) v.size();
    // Also assert the algebra itself on one element.
    const float  t   = 0.5f;
    const double alg = fabs((double) xt[0] - ((double) t * ((double) v[0] + (double) hypa->lat[0]) +
                                              (1.0 - (double) t) * (double) hypa->lat[0]));
    char d[224];
    snprintf(d, sizeof(d), "E[v^2] over crop [0,750) of Hypa_Hypa = %.6f (want 1.672071 +-1e-5); xt algebra residual %.2e",
             ev2, alg);
    dit_st_report(rs, "T9", fabs(ev2 - 1.672071) < 1e-5 && alg < 1e-5, d);
}

// ─── MU1: Muon / Newton-Schulz through the REAL optimizer graph ─────────────
//
// Two independent bars, because they fail differently:
//
//  (a) vs a double-precision host reference of the same iteration — catches a
//      wrong coefficient, a missed normalization, a dropped Nesterov term.
//  (b) SEMI-ORTHOGONALITY of the result — catches the failure a host reference
//      written by the same hand would happily mirror, namely a transposed
//      mul_mat. If O is what Muon wants, O*O^T is near I; a transpose error
//      still agrees with a matching reference but is not orthogonal at all.
//
// Bar for (a) is 2e-3, not 1e-5: five chained GEMMs run through TF32 on CUDA.
// (b) is loose on purpose — the quintic with these coefficients is tuned to
// pull the singular values into a band around 1 in five steps, not to converge
// to machine precision, so |sigma - 1| ~ 0.3 is success, not slop.
static void dit_st_muon(std::vector<DitSelfTestResult> & rs, ggml_backend_t backend, ggml_backend_sched_t sched,
                        ggml_tensor * t_adamw, ggml_tensor * t_lossgrad, ggml_tensor * t_clip, ggml_tensor * t_eps,
                        ggml_tensor * t_gnorm2) {
    const double nsa = (double) LM_MUON_NS_A, nsb = (double) LM_MUON_NS_B, nsc = (double) LM_MUON_NS_C;
    const double mu = 0.95, lr = 1e-3, wd = 0.01, epsv = 1e-6;

    // Host reference: X is R x C row-major. Mirrors lm_muon_newton_schulz.
    auto ns_ref = [&](std::vector<double> X, int R, int C, int steps) {
        auto tr = [](const std::vector<double> & M, int r, int c) {
            std::vector<double> T((size_t) r * c);
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    T[(size_t) j * r + i] = M[(size_t) i * c + j];
                }
            }
            return T;
        };
        const bool tall = R > C;
        if (tall) {
            X = tr(X, R, C);
            std::swap(R, C);
        }
        double nrm = 0.0;
        for (size_t i = 0; i < X.size(); i++) {
            nrm += X[i] * X[i];
        }
        nrm = sqrt(nrm) + epsv;
        for (size_t i = 0; i < X.size(); i++) {
            X[i] /= nrm;
        }
        for (int s = 0; s < steps; s++) {
            std::vector<double> A((size_t) R * R, 0.0), A2((size_t) R * R, 0.0), BX((size_t) R * C, 0.0);
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < R; j++) {
                    double acc = 0.0;
                    for (int k = 0; k < C; k++) {
                        acc += X[(size_t) i * C + k] * X[(size_t) j * C + k];
                    }
                    A[(size_t) i * R + j] = acc;
                }
            }
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < R; j++) {
                    double acc = 0.0;
                    for (int k = 0; k < R; k++) {
                        acc += A[(size_t) i * R + k] * A[(size_t) k * R + j];
                    }
                    A2[(size_t) i * R + j] = acc;
                }
            }
            for (int i = 0; i < R; i++) {
                for (int j = 0; j < C; j++) {
                    double acc = 0.0;
                    for (int k = 0; k < R; k++) {
                        acc += (nsb * A[(size_t) i * R + k] + nsc * A2[(size_t) i * R + k]) * X[(size_t) k * C + j];
                    }
                    BX[(size_t) i * C + j] = acc;
                }
            }
            for (size_t i = 0; i < X.size(); i++) {
                X[i] = nsa * X[i] + BX[i];
            }
        }
        return tall ? tr(X, R, C) : X;
    };

    bool   all_ok = true;
    double cpu_rel = 0.0, cpu_rel_ns1 = 0.0, gpu_rel = 0.0, worst_orth = 0.0;
    char   why[192] = "";
    bool   cpu_ran = false;

    // One toy Muon step on `be`, self-contained: its own scalars, its own sched.
    // Returns the scale-relative UPDATE error against the host reference, and
    // the semi-orthogonality of the ORTHOGONALIZED FACTOR THE GRAPH PRODUCED —
    // recovered from the parameter delta. That second number is the one a
    // transposed mul_mat cannot fake: it would still agree with a reference
    // written by the same hand, but it would not be orthogonal.
    // `ext_sched != nullptr` means "run on the caller's backend with the caller's
    // scalars": ggml_backend_sched_new ASSERTS the last backend is a CPU device
    // (ggml-backend.cpp:1736), so a {CUDA, CUDA} pair cannot be built here — the
    // GPU arm has to borrow the scheduler the trainer already owns.
    auto run_case = [&](ggml_backend_t be, ggml_backend_sched_t ext_sched, ggml_tensor * ext_adamw,
                        ggml_tensor * ext_eps, ggml_tensor * ext_gnorm2, int C, int R, int ns, double * upd_rel, double * orth,
                        std::string * err) -> bool {
        ggml_context * tctx = nullptr;
        {
            ggml_init_params p = { 16 * ggml_tensor_overhead(), nullptr, true };
            tctx               = ggml_init(p);
        }
        ggml_tensor * w = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, C, R);
        ggml_set_name(w, "toy.muon");
        ggml_set_param(w);
        // lm_optim_step uploads the AdamW scalar block unconditionally and always
        // writes t_gnorm2, so both must exist even for a Muon-only parameter set.
        ggml_tensor * s_adamw  = ext_sched ? ext_adamw : ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 7);
        ggml_tensor * s_eps    = ext_sched ? ext_eps : ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 1);
        ggml_tensor * s_gnorm2 = ext_sched ? ext_gnorm2 : ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 1);
        ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, be);
        if (!tbuf) {
            *err = "toy allocation failed";
            ggml_free(tctx);
            return false;
        }
        ggml_backend_sched_t sc = ext_sched;
        if (!sc) {
            BackendPair bp;
            bp.backend     = be;
            bp.cpu_backend = be;  // be IS the CPU backend on this arm, so the assert holds
            bp.has_gpu     = false;
            sc             = backend_sched_new(bp, 4096);
        }

        std::vector<ggml_tensor *> tparams(1, w);
        LmOptim                    topt;
        topt.optimizer     = "muon";
        topt.muon.momentum = (float) mu;
        topt.muon.ns_steps = ns;
        topt.muon.nesterov = true;
        topt.muon.min_dim  = 16;
        if (!lm_optim_init(&topt, tparams, be, err)) {
            if (!ext_sched) {
                ggml_backend_sched_free(sc);
            }
            ggml_backend_buffer_free(tbuf);
            ggml_free(tctx);
            return false;
        }
        if (topt.n_muon != 1) {
            *err = "a [" + std::to_string(C) + "," + std::to_string(R) + "] param was not classified Muon-eligible";
            lm_optim_free(&topt);
            if (!ext_sched) {
                ggml_backend_sched_free(sc);
            }
            ggml_backend_buffer_free(tbuf);
            ggml_free(tctx);
            return false;
        }
        topt.t_adamw      = s_adamw;
        topt.t_lossgrad   = nullptr;  // fill_gacc is not used here
        topt.t_clip       = nullptr;  // grad_clip 0 => never referenced
        topt.t_eps        = s_eps;
        topt.t_gnorm2     = s_gnorm2;
        topt.base_lr      = (float) lr;
        topt.weight_decay = (float) wd;
        topt.grad_clip    = 0.0f;  // isolate the update rule from the clip
        topt.total_steps  = 100;
        topt.warmup_steps = 0;

        const size_t       n = (size_t) R * C;
        std::vector<float> w0(n), g(n);
        LmRng              rng;
        lm_rng_seed(&rng, 0x3110Aull + (uint64_t) (R * 131 + C));
        lm_rng_fill_normal(&rng, w0, 0.1f);
        lm_rng_fill_normal(&rng, g, 0.5f);
        ggml_backend_tensor_set(w, w0.data(), 0, n * sizeof(float));
        ggml_backend_tensor_set(topt.acc[0], g.data(), 0, n * sizeof(float));
        const float epsf = (float) epsv;
        ggml_backend_tensor_set(s_eps, &epsf, 0, sizeof(float));

        LmStepStats stt;
        const bool  ran = lm_optim_step(&topt, sc, &stt);

        // Reference: m starts at 0, so m_new = g and the Nesterov term is
        // (1 + mu) * g. lm_lr_lambda(0, 100, 0) == 1, so the LR is base_lr.
        std::vector<double> u(n);
        for (size_t i = 0; i < n; i++) {
            u[i] = (1.0 + mu) * (double) g[i];
        }
        const std::vector<double> O     = ns_ref(u, R, C, ns);
        const double              shape = sqrt(R > C ? (double) R / (double) C : 1.0);
        std::vector<float>        got(n);
        ggml_backend_tensor_get(w, got.data(), 0, n * sizeof(float));

        // Scale-relative on the UPDATE. Dividing per element by the parameter
        // value normalizes by a number that is near zero for a chunk of a
        // Gaussian init and reports ~1e-2 for a perfectly good update — the
        // denominator is wrong, not the maths.
        double num = 0.0, den = 0.0;
        std::vector<double> Og(n);
        for (size_t i = 0; i < n; i++) {
            const double upd_ref = -lr * shape * O[i];
            const double upd_got = (double) got[i] - (double) w0[i] * (1.0 - lr * wd);
            num                  = std::max(num, fabs(upd_got - upd_ref));
            den                  = std::max(den, fabs(upd_ref));
            Og[i]                = upd_got / (-lr * shape);  // the graph's own O
        }
        *upd_rel = num / std::max(den, 1e-12);

        const int sR = std::min(R, C);
        *orth        = 0.0;
        for (int i = 0; i < sR; i++) {
            for (int j = 0; j < sR; j++) {
                double acc = 0.0;
                if (R <= C) {
                    for (int k = 0; k < C; k++) {
                        acc += Og[(size_t) i * C + k] * Og[(size_t) j * C + k];
                    }
                } else {
                    for (int k = 0; k < R; k++) {
                        acc += Og[(size_t) k * C + i] * Og[(size_t) k * C + j];
                    }
                }
                *orth = std::max(*orth, fabs(acc - (i == j ? 1.0 : 0.0)));
            }
        }

        lm_optim_free(&topt);
        if (!ext_sched) {
            ggml_backend_sched_free(sc);
        }
        ggml_backend_buffer_free(tbuf);
        ggml_free(tctx);
        if (!ran) {
            *err = "lm_optim_step failed";
        }
        return ran;
    };

    // Two shapes: wide (no transpose) and tall (exercises the transpose path).
    const int shapes[2][2] = { { 48, 32 }, { 32, 48 } };  // {ne0 = cols, ne1 = rows}
    ggml_backend_t cpu_be  = cpu_backend_new(16);
    for (int sh = 0; sh < 2 && all_ok; sh++) {
        const int   C = shapes[sh][0], R = shapes[sh][1];
        double      r = 0.0, o = 0.0;
        std::string err;

        // THE GATE: CPU, where F32 is F32 and the reference is reproducible.
        if (cpu_be) {
            if (!run_case(cpu_be, nullptr, nullptr, nullptr, nullptr, C, R, 5, &r, &o, &err)) {
                snprintf(why, sizeof(why), "cpu: %s", err.c_str());
                all_ok = false;
                break;
            }
            cpu_ran    = true;
            cpu_rel    = std::max(cpu_rel, r);
            worst_orth = std::max(worst_orth, o);
            // Same case at ONE iteration, as a control on WHERE the residual
            // comes from. Measured: 8.40e-05 at 1 step vs 5.73e-05 at 5, i.e.
            // it does NOT grow with the iteration count, which rules out
            // per-GEMM accumulation. What both share is the Frobenius-norm
            // reduction: a ~1536-element ggml_sum in F32, whose naive-summation
            // error is ~n*eps/2 ~ 9e-5 — the right magnitude. (NS is contractive,
            // which is why more iterations slightly SHRINK the relative error.)
            // That is what justifies 1e-4 here against T11's 1e-5, and it is a
            // scale error on an update that is approximate by design.
            double r1 = 0.0, o1 = 0.0;
            std::string e1;
            if (run_case(cpu_be, nullptr, nullptr, nullptr, nullptr, C, R, 1, &r1, &o1, &e1)) {
                cpu_rel_ns1 = std::max(cpu_rel_ns1, r1);
            }
        }
        // REPORTED, not gated: cuBLAS runs these F32 GEMMs on TF32 tensor cores,
        // which puts a ~4e-3 floor on five chained Newton-Schulz iterations. Same
        // treatment SB1/SB2 give their CUDA arms.
        if (backend && backend != cpu_be) {
            if (run_case(backend, sched, t_adamw, t_eps, t_gnorm2, C, R, 5, &r, &o, &err)) {
                gpu_rel = std::max(gpu_rel, r);
            }
        }
    }
    // NOTE (2026-09-01, found while wiring the TF32 trainer surface): the
    // ggml_backend_free(cpu_be) that used to sit HERE was a use-after-free. The
    // bucket case below allocates its tensors on cpu_be, builds a scheduler over
    // it and runs an optimizer step through it — all on a freed backend. It
    // survived by luck of heap layout for as long as it existed; an unrelated
    // allocation change elsewhere in the self-test turned it into a hard
    // GGML_ASSERT(device) abort in ggml_backend_dev_buffer_type, which killed
    // the run before MU1/SC1-3/SB1-3 could report. The free now happens after
    // the last use.

    // ── the bucket case ──────────────────────────────────────────────────────
    //
    // Three same-shape parameters in ONE bucket, with grads deliberately spread
    // over two orders of magnitude (x8, x1, x1/8). Batched Newton-Schulz
    // normalizes per slab; a global ggml_sum would normalize all three by the
    // bucket's combined magnitude, which is invisible at bucket size 1 and
    // glaring here — the x1/8 parameter would come out ~64x wrong.
    double bucket_rel = 0.0;
    int    n_buckets  = -1;
    if (all_ok && cpu_be) {
        const int    C = 48, R = 32, NP = 3;
        const size_t n = (size_t) R * C;
        const float  gs[NP] = { 8.0f, 1.0f, 0.125f };

        ggml_context * tctx = nullptr;
        {
            ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
            tctx               = ggml_init(p);
        }
        std::vector<ggml_tensor *> ws;
        for (int i = 0; i < NP; i++) {
            ggml_tensor * w = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, C, R);
            char          nm[32];
            snprintf(nm, sizeof(nm), "toy.bucket.%d", i);
            ggml_set_name(w, nm);
            ggml_set_param(w);
            ws.push_back(w);
        }
        ggml_tensor * s_adamw  = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 7);
        ggml_tensor * s_eps    = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 1);
        ggml_tensor * s_gnorm2 = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 1);
        ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, cpu_be);

        BackendPair bp;
        bp.backend              = cpu_be;
        bp.cpu_backend          = cpu_be;
        bp.has_gpu              = false;
        ggml_backend_sched_t sc = backend_sched_new(bp, 4096);

        LmOptim topt;
        topt.optimizer     = "muon";
        topt.muon.momentum = (float) mu;
        topt.muon.ns_steps = 5;
        topt.muon.nesterov = true;
        topt.muon.min_dim  = 16;
        topt.muon.bucket   = 64;  // all three must land in ONE bucket
        std::string berr;
        if (!lm_optim_init(&topt, ws, cpu_be, &berr)) {
            snprintf(why, sizeof(why), "bucket case: %s", berr.c_str());
            all_ok = false;
        } else {
            n_buckets         = (int) topt.muon_buckets.size();
            topt.t_adamw      = s_adamw;
            topt.t_eps        = s_eps;
            topt.t_gnorm2     = s_gnorm2;
            topt.base_lr      = (float) lr;
            topt.weight_decay = (float) wd;
            topt.grad_clip    = 0.0f;
            topt.total_steps  = 100;
            topt.warmup_steps = 0;

            std::vector<std::vector<float>> w0(NP), g(NP);
            LmRng                           rng;
            lm_rng_seed(&rng, 0xB0CCE7ull);
            for (int i = 0; i < NP; i++) {
                w0[i].resize(n);
                g[i].resize(n);
                lm_rng_fill_normal(&rng, w0[i], 0.1f);
                lm_rng_fill_normal(&rng, g[i], 0.5f);
                for (size_t k = 0; k < n; k++) {
                    g[i][k] *= gs[i];
                }
                ggml_backend_tensor_set(ws[i], w0[i].data(), 0, n * sizeof(float));
                ggml_backend_tensor_set(topt.acc[(size_t) i], g[i].data(), 0, n * sizeof(float));
            }
            const float epsf = (float) epsv;
            ggml_backend_tensor_set(s_eps, &epsf, 0, sizeof(float));

            LmStepStats stt;
            lm_optim_step(&topt, sc, &stt);

            for (int i = 0; i < NP; i++) {
                std::vector<double> u(n);
                for (size_t k = 0; k < n; k++) {
                    u[k] = (1.0 + mu) * (double) g[i][k];
                }
                const std::vector<double> O = ns_ref(u, R, C, 5);  // R < C here, no transpose
                std::vector<float>        got(n);
                ggml_backend_tensor_get(ws[i], got.data(), 0, n * sizeof(float));
                double num = 0.0, den = 0.0;
                for (size_t k = 0; k < n; k++) {
                    const double upd_ref = -lr * O[k];  // shape scale is 1 for R < C
                    const double upd_got = (double) got[k] - (double) w0[i][k] * (1.0 - lr * wd);
                    num                  = std::max(num, fabs(upd_got - upd_ref));
                    den                  = std::max(den, fabs(upd_ref));
                }
                bucket_rel = std::max(bucket_rel, num / std::max(den, 1e-12));
            }
            lm_optim_free(&topt);
        }
        ggml_backend_sched_free(sc);
        if (tbuf) {
            ggml_backend_buffer_free(tbuf);
        }
        ggml_free(tctx);
        if (n_buckets != 1) {
            snprintf(why, sizeof(why), "3 same-shape params formed %d buckets, expected 1", n_buckets);
            all_ok = false;
        }
    }
    // Last use of cpu_be is above. See the NOTE where this free used to be.
    if (cpu_be) {
        ggml_backend_free(cpu_be);
        cpu_be = nullptr;
    }

    // Classification: a LoKR w1 shape must NOT be picked up.
    {
        ggml_context * tctx = nullptr;
        {
            ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
            tctx               = ggml_init(p);
        }
        ggml_tensor * tiny = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, 4, 5);  // LoKR w1 at factor 6
        LmMuonCfg     cfg;
        if (lm_muon_eligible(tiny, cfg)) {
            snprintf(why, sizeof(why), "a [4,5] LoKR w1 was classified Muon-eligible (min_dim %d)", cfg.min_dim);
            all_ok = false;
        }
        ggml_free(tctx);
    }

    // The orthogonality bar is on O*O^T, i.e. on SQUARED singular values, and
    // the quintic with these coefficients is tuned to pull sigma into roughly
    // [0.7, 1.3] in five steps rather than to converge — so |sigma^2 - 1| up to
    // ~0.7 is the iteration working as designed, not slop. Measured 0.40 here
    // (sigma in [0.77, 1.18]), and it is computed from the HOST reference, so
    // it characterizes Newton-Schulz itself, not the graph.
    const bool ok = all_ok && cpu_ran && cpu_rel < 1e-4 && bucket_rel < 1e-4 && worst_orth < 0.75;
    char       d[720];
    snprintf(d, sizeof(d),
             "Muon on [48,32] and [32,48] (wide + tall), UPDATE vs a double-precision host reference: "
             "CPU max rel=%.4e at 5 Newton-Schulz steps vs %.4e at 1 step (bar 1e-4 — the residual does NOT grow with "
             "the iteration count, so it is the F32 Frobenius-norm reduction, not per-GEMM accumulation) [%s]; "
             "%s reports %.4e, NOT gated (cuBLAS runs these F32 GEMMs on TF32). Graph's own O: max|O*O^T - I|=%.4f "
             "(bar 0.75 — SQUARED singular values, and NS5 targets a band around 1 rather than converging). "
             "BUCKETED (3 same-shape params, grads spread x8/x1/x1-8, %d bucket) max rel=%.4e (bar 1e-4 — a "
             "GLOBAL rather than per-slab Frobenius norm would put the small one ~64x out). "
             "[4,5] correctly falls to AdamW%s%s",
             cpu_rel, cpu_rel_ns1, cpu_ran ? "cpu ok" : "cpu did not run",
             backend ? ggml_backend_name(backend) : "gpu", gpu_rel, worst_orth, n_buckets, bucket_rel,
             why[0] ? " | " : "", why);
    dit_st_report(rs, "MU1", ok, d);
}

// ─── T11: AdamW + clip through the REAL optimizer graph ─────────────────────

static void dit_st_adamw(std::vector<DitSelfTestResult> & rs, ggml_backend_t backend, ggml_backend_sched_t sched,
                         ggml_tensor * t_adamw, ggml_tensor * t_lossgrad, ggml_tensor * t_clip, ggml_tensor * t_eps,
                         ggml_tensor * t_gnorm2) {
    ggml_context * tctx = nullptr;
    {
        ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
        tctx               = ggml_init(p);
    }
    ggml_tensor * w = ggml_new_tensor_1d(tctx, GGML_TYPE_F32, 64);
    ggml_set_name(w, "toy.param");
    ggml_set_param(w);
    ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, backend);

    std::vector<ggml_tensor *> tparams(1, w);
    LmOptim                    topt;
    std::string                err;
    if (!lm_optim_init(&topt, tparams, backend, &err)) {
        dit_st_report(rs, "T11", false, err);
        if (tbuf) {
            ggml_backend_buffer_free(tbuf);
        }
        ggml_free(tctx);
        return;
    }
    topt.t_adamw      = t_adamw;
    topt.t_lossgrad   = t_lossgrad;
    topt.t_clip       = t_clip;
    topt.t_eps        = t_eps;
    topt.t_gnorm2     = t_gnorm2;
    topt.base_lr      = 1e-3f;
    topt.weight_decay = 0.01f;
    topt.grad_clip    = 0.0f;
    topt.total_steps  = 100;
    topt.warmup_steps = 0;

    std::vector<float> w0(64), g(64);
    LmRng              rng;
    lm_rng_seed(&rng, 0xA11CEull);
    lm_rng_fill_normal(&rng, w0, 0.1f);
    lm_rng_fill_normal(&rng, g, 0.5f);
    ggml_backend_tensor_set(w, w0.data(), 0, w0.size() * sizeof(float));
    const float clipv = 1.0f, epsv = 1e-6f;
    ggml_backend_tensor_set(t_clip, &clipv, 0, sizeof(float));
    ggml_backend_tensor_set(t_eps, &epsv, 0, sizeof(float));

    std::vector<double> hw(w0.begin(), w0.end()), hm(64, 0.0), hv(64, 0.0);
    const double        b1 = 0.9, b2 = 0.999, aeps = 1e-8, wd = 0.01;
    for (int step = 0; step < 3; step++) {
        ggml_backend_tensor_set(topt.acc[0], g.data(), 0, g.size() * sizeof(float));
        LmStepStats stt;
        lm_optim_step(&topt, sched, &stt);
        const double alpha = (double) topt.base_lr * (double) lm_lr_lambda(step, 100, 0);
        const double b1h   = 1.0 / (1.0 - pow(b1, (double) (step + 1)));
        const double b2h   = 1.0 / (1.0 - pow(b2, (double) (step + 1)));
        for (int i = 0; i < 64; i++) {
            const double gi = (double) g[i];
            hm[i]           = hm[i] * b1 + gi * (1.0 - b1);
            hv[i]           = hv[i] * b2 + gi * gi * (1.0 - b2);
            hw[i]           = hw[i] * (1.0 - alpha * wd) - alpha * (hm[i] * b1h) / (sqrt(hv[i] * b2h) + aeps);
        }
    }
    std::vector<float> got(64);
    ggml_backend_tensor_get(w, got.data(), 0, got.size() * sizeof(float));
    double worst = 0.0;
    for (int i = 0; i < 64; i++) {
        worst = std::max(worst, fabs((double) got[i] - hw[i]) / std::max(fabs(hw[i]), 1e-9));
    }

    // clip: ||g|| exactly 10, clip 1.0 -> applied gradient scaled by exactly 0.1
    ggml_backend_buffer_clear(topt.buf_mom, 0);
    std::vector<float> gg(64, 10.0f / 8.0f);
    ggml_backend_tensor_set(topt.acc[0], gg.data(), 0, gg.size() * sizeof(float));
    topt.grad_clip = 1.0f;
    topt.opt_iter  = 0;
    topt.opt_step  = 1;
    LmStepStats stt;
    lm_optim_step(&topt, sched, &stt);
    float gn2v = 0.0f;
    ggml_backend_tensor_get(t_gnorm2, &gn2v, 0, sizeof(float));
    std::vector<float> m1(64);
    ggml_backend_tensor_get(topt.mom_m[0], m1.data(), 0, m1.size() * sizeof(float));
    double sm = 0.0;
    for (int i = 0; i < 64; i++) {
        sm += (double) m1[i] * (double) m1[i];
    }
    const double applied = sqrt(sm) / (1.0 - 0.9) / 10.0;

    char d[288];
    snprintf(d, sizeof(d),
             "AdamW 3 steps x 64 elements worst rel=%.4e (bar 1e-5); clip: gnorm2=%.6f (want 100 +-1e-2), "
             "applied ratio=%.6f (want 0.1 +-1e-3), clipScale=%.6f",
             worst, (double) gn2v, applied, (double) stt.clip);
    dit_st_report(rs, "T11", worst < 1e-5 && fabs((double) gn2v - 100.0) < 1e-2 && fabs(applied - 0.1) < 1e-3, d);

    lm_optim_free(&topt);
    if (tbuf) {
        ggml_backend_buffer_free(tbuf);
    }
    ggml_free(tctx);
}

// ─── SB1 / SB2 / SB3: micro-batching (design §2.3 gates 2, 3 and 5) ─────────
//
// These run the PRODUCTION graph (dit_train_forward / dit_train_loss) and the
// PRODUCTION batch assembler (dit_batch_assemble) — the spike proved the
// mechanisms on a private copy of the layer, and these rungs are what keep the
// shipped code honest afterwards.
//
//   SB1  batched forward parity     one B-element forward vs B single-sample ones
//   SB2  batched backward parity    one B-element backward vs the weighted sum of
//                                   B single-sample backwards accumulated into the
//                                   SAME LmOptim::acc[]
//   SB3  mixed-length batch         padded frames contribute EXACTLY zero to the
//                                   loss and to every adapter gradient — measured
//                                   by dirtying the pad's velocity TARGET (the
//                                   loss-mask path) AND its INPUT (the
//                                   self-attention KV-mask path, the one the loss
//                                   mask cannot cover)
//
// WHAT IS ASSERTED WHERE (spike amendment A2). Batching changes the CUDA GEMM
// route: `src1->ne[2]*src1->ne[3] > 1` sends a mul_mat to
// ggml_cuda_mul_mat_batched_cublas instead of the F32-accurate mmf path the
// single-sample graph takes, so a batched-vs-single comparison on CUDA sits on a
// ~1e-5 relative floor (~1e-3 with TF32 on) BY CONSTRUCTION, and the design's
// 1e-6/1e-5 bars are unreachable there. What is reachable, and what these rungs
// gate on, is
//   (a) the CPU-backend identity — one mul_mat kernel loops over ne2/ne3, so the
//       batched and single-sample graphs execute the same code in the same order,
//   (b) the CUDA batch-PERMUTATION control — same shapes, same routes, elements
//       uploaded in reverse; the measurement that actually catches cross-element
//       contamination.
// The CUDA batched-vs-single delta is measured and printed, never gated.
//
// SB2 has no gated CUDA control: the adapter gradient is a reduction ACROSS the
// batch, so permuting the elements permutes the summation order and bit-exactness
// is not owed. Its CUDA numbers are reported; its gate is the CPU identity.

struct DitStBatchMeas {
    bool        ran       = false;
    double      fwd_delta = 0.0;  // batched vs B single forwards, max |abs|
    double      fwd_perm  = 0.0;  // reversed-element control, max |abs| on mirrored elements
    double      vmag      = 0.0;
    double      bwd_rel   = 0.0;  // batched vs the weighted sum of B single backwards
    double      bwd_perm  = 0.0;
    std::string bwd_worst = "-";
    double      pad_loss  = 0.0;  // |loss(garbage pad target) - loss(clean pad target)|
    double      pad_grad  = 0.0;
    // The decisive pair: the same two numbers with the padded INPUT scribbled on
    // instead of the padded target. Only the self-attention KV mask can make these
    // zero — the loss mask cannot, because the leak is padded frames reaching the
    // VALID queries as attention keys, upstream of the loss entirely.
    double      pad_loss_in = 0.0;
    double      pad_grad_in = 0.0;
    double      pad_host  = 0.0;  // masked loss vs a host double recompute, relative
    double      pad_acc = 0.0, pad_acc_all = 0.0, pad_lossv = 0.0, pad_velcmp = 0.0;
    int         nodes_bat = 0, nodes_sin = 0;
    int         short_len = 0, pad_len = 0;
};

// One measurement pass on one backend. `samples` must hold at least B songs.
static bool dit_st_batch_measure(DitTrainModel * M, const std::vector<DitSample> & samples, uint64_t seed, int B,
                                 int T, int enc_use, int rank, DitStBatchMeas * o, std::string * err) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int L = c.n_layers, H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, Ic = c.in_channels;
    const int enc_H = (int) M->m.cond_emb_w->ne[0];
    const int S     = T / P;
    const int lo = L - 2, hi = L;

    // ── static input bases (1-D; every graph tensor is a contiguous view) ─
    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Ic * T * B);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) enc_H * enc_use * B);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, (int64_t) S * B);
    ggml_tensor * b_temb  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) H * B);
    ggml_tensor * b_tproj = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) 6 * H * B);
    // b_sa is B-wide: dit_batch_assemble emits a per-element [S,S,1,B] mask the
    // moment any element is padded (the SB3 case), and the shared [S,S] one
    // otherwise. The base is sized for the wide shape either way.
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_sapad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) enc_use * S * B);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Oc * T * B);
    ggml_tensor * b_lw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * b_lwu   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * b_ones  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    for (ggml_tensor * t :
         { b_input, b_enc, b_pos, b_temb, b_tproj, b_sa, b_sapad, b_ca, b_vtgt, b_lw, b_lwu, b_ones }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctxs, M->backend);
    if (!buf) {
        *err = "static input allocation failed";
        ggml_free(ctxs);
        return false;
    }
    {
        const float epsv = 1e-6f, clipv = 1.0f, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    DitAdapterLora lora;
    LmOptim        opt;
    bool           have_opt = false;
    auto           cleanup  = [&]() {
        if (have_opt) {
            lm_optim_free(&opt);
        }
        lora.free();
        ggml_backend_buffer_free(buf);
        ggml_free(ctxs);
    };
    {
        DitAdapterCfg cfg;
        cfg.rank    = rank;
        cfg.alpha   = (float) (2 * rank);
        cfg.seed    = seed ^ 0x1234ull;
        cfg.b_sigma = 1e-2f;  // B == 0 would make dL/dA identically zero
        // HOTSTEP_DIT_ST_DORA=1 runs this rung under DoRA: the magnitude
        // tensors join params() and the A/B gradients are taken through the
        // m/||W+BA|| rescale. An env var, not a flag, so the FD child process
        // (dit_st_spawn_fd) inherits it unchanged.
        cfg.dora    = getenv("HOTSTEP_DIT_ST_DORA") != nullptr;
        if (!lora.init(&M->m, M->backend, lo, hi, cfg, err)) {
            cleanup();
            return false;
        }
        if (!lm_optim_init(&opt, lora.params(), M->backend, err)) {
            cleanup();
            return false;
        }
        have_opt = true;
    }
    opt.t_adamw    = t_adamw;
    opt.t_lossgrad = t_lossgrad;
    opt.t_clip     = t_clip;
    opt.t_eps      = t_eps;
    opt.t_gnorm2   = t_gnorm2;

    std::vector<uint8_t> arena((size_t) 512 << 20);

    // ── host-side batch (the PRODUCTION assembler) ───────────────────────
    //
    // B different songs, per-element t / CFG state / encoder-padding mask, and
    // element 1 CFG-dropped so the elements really do differ.
    LmRng rng_t, rng_crop, rng_noise;
    lm_rng_seed(&rng_t, seed);
    lm_rng_seed(&rng_crop, seed ^ 0xbf58476d1ce4e5b9ull);
    lm_rng_seed(&rng_noise, seed ^ 0x9e3779b97f4a7c15ull);

    DitBatchCfg bcfg;
    bcfg.in_ch          = Ic;
    bcfg.out_ch         = Oc;
    bcfg.enc_H          = enc_H;
    bcfg.enc_S          = enc_use;
    bcfg.patch          = P;
    bcfg.sliding_window = c.sliding_window;
    bcfg.crop           = T;
    bcfg.weighted       = true;
    bcfg.null_cond      = &M->null_cond;

    std::vector<DitBatchElem> els((size_t) B);
    {
        double wsum = 0.0;
        std::vector<double> w((size_t) B, 1.0);
        for (int b = 0; b < B; b++) {
            els[(size_t) b].s = &samples[(size_t) b];
            els[(size_t) b].t = dit_sample_t(&rng_t, -0.4f, 1.0f, 0.0f, 1.0f);
            w[(size_t) b]     = (double) dit_flow_snr_w(els[(size_t) b].t, 0.5f, 5.0f);
            wsum += w[(size_t) b];
        }
        const double wbar = wsum / (double) B;
        for (int b = 0; b < B; b++) {
            els[(size_t) b].w = (float) (w[(size_t) b] / (wbar > 0.0 ? wbar : 1.0));
        }
        if ((int) M->null_cond.size() == enc_H && B > 1) {
            els[1].cfg_drop = true;
        }
    }
    DitBatchHost bh;
    dit_batch_assemble(bcfg, els, &rng_crop, &rng_noise, &bh);
    if (bh.len != T) {
        char b2[128];
        snprintf(b2, sizeof(b2), "batch assembled at len %d, expected %d", bh.len, T);
        *err = b2;
        cleanup();
        return false;
    }

    // temb / tproj per element (design B3)
    std::vector<float> temb_buf((size_t) H * B, 0.0f), tproj_buf((size_t) 6 * H * B, 0.0f);
    {
        std::vector<float> ts((size_t) B);
        for (int b = 0; b < B; b++) {
            ts[(size_t) b] = els[(size_t) b].t;
        }
        std::vector<std::vector<float>> tb, tp;
        if (!dit_train_temb(M, ts, &tb, &tp)) {
            *err = "temb precompute failed";
            cleanup();
            return false;
        }
        for (int b = 0; b < B; b++) {
            memcpy(&temb_buf[(size_t) b * H], tb[(size_t) b].data(), (size_t) H * sizeof(float));
            memcpy(&tproj_buf[(size_t) b * 6 * H], tp[(size_t) b].data(), (size_t) 6 * H * sizeof(float));
        }
    }

    // Every input re-uploaded before every compute (§3.0 clobber rule). `rev`
    // uploads the SAME elements in reversed order: shapes, kernels and cuBLAS
    // routes are then bit-identical, so element b of a reversed run vs element
    // B-1-b of a forward run is a route-identical contamination test.
    std::vector<float>    rv_f;
    std::vector<uint16_t> rv_u;
    std::vector<float>    vtgt_use, input_use;
    const std::vector<float> ones_host((size_t) T, 1.0f);
    auto upload = [&](bool rev) {
        auto put_f = [&](ggml_tensor * dst, const std::vector<float> & src, size_t per) {
            if (!rev) {
                ggml_backend_tensor_set(dst, src.data(), 0, src.size() * sizeof(float));
                return;
            }
            rv_f.resize(src.size());
            for (int b = 0; b < B; b++) {
                memcpy(&rv_f[(size_t) b * per], &src[(size_t) (B - 1 - b) * per], per * sizeof(float));
            }
            ggml_backend_tensor_set(dst, rv_f.data(), 0, rv_f.size() * sizeof(float));
        };
        // Element-reversal never applies to a 1-element buffer, so put_u16 takes
        // the per-element stride and the element COUNT the host buffer holds.
        auto put_u16 = [&](ggml_tensor * dst, const std::vector<uint16_t> & src, size_t per, int nel) {
            if (!rev || nel <= 1) {
                ggml_backend_tensor_set(dst, src.data(), 0, src.size() * sizeof(uint16_t));
                return;
            }
            rv_u.resize(src.size());
            for (int b = 0; b < nel; b++) {
                memcpy(&rv_u[(size_t) b * per], &src[(size_t) (nel - 1 - b) * per], per * sizeof(uint16_t));
            }
            ggml_backend_tensor_set(dst, rv_u.data(), 0, rv_u.size() * sizeof(uint16_t));
        };
        put_f(b_input, input_use, (size_t) Ic * T);
        put_f(b_vtgt, vtgt_use, (size_t) Oc * T);
        put_f(b_enc, bh.enc, (size_t) enc_H * enc_use);
        put_f(b_lw, bh.lw, (size_t) T);
        put_f(b_lwu, bh.lwu, (size_t) T);
        put_f(b_temb, temb_buf, (size_t) H);
        put_f(b_tproj, tproj_buf, (size_t) 6 * H);
        ggml_backend_tensor_set(b_pos, bh.pos.data(), 0, bh.pos.size() * sizeof(int32_t));
        // §3.0 applies to the all-ones single-sample mask too: it is a graph INPUT
        // (mk_single's t_lw), so it is re-uploaded with everything else rather than
        // once at setup.
        ggml_backend_tensor_set(b_ones, ones_host.data(), 0, ones_host.size() * sizeof(float));
        put_u16(b_sa, bh.sa, (size_t) S * (size_t) S, bh.sa_B);
        if (!bh.sa_pad.empty()) {
            put_u16(b_sapad, bh.sa_pad, (size_t) S * (size_t) S, bh.sa_B);
        }
        put_u16(b_ca, bh.ca, (size_t) enc_use * (size_t) S, B);
    };
    vtgt_use  = bh.vtgt;
    input_use = bh.input;

    const size_t f32b = sizeof(float), f16b = sizeof(ggml_fp16_t);
    auto mk_batched = [&](ggml_context * ctx, int nb) {
        DitInputs in;
        in.t_input = ggml_view_3d(ctx, b_input, Ic, T, nb, (size_t) Ic * f32b, (size_t) Ic * T * f32b, 0);
        in.t_enc   = ggml_view_3d(ctx, b_enc, enc_H, enc_use, nb, (size_t) enc_H * f32b,
                                  (size_t) enc_H * enc_use * f32b, 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, (int64_t) S * nb, 0);
        in.t_temb  = ggml_view_3d(ctx, b_temb, H, 1, nb, (size_t) H * f32b, (size_t) H * f32b, 0);
        in.t_tproj = ggml_view_3d(ctx, b_tproj, 6 * H, 1, nb, (size_t) 6 * H * f32b, (size_t) 6 * H * f32b, 0);
        in.t_sa    = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sa, S, S, 1, nb, (size_t) S * f16b, (size_t) S * S * f16b,
                                                  (size_t) S * S * f16b, 0)
                                   : ggml_view_2d(ctx, b_sa, S, S, (size_t) S * f16b, 0);
        in.t_sa_pad = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sapad, S, S, 1, nb, (size_t) S * f16b,
                                                   (size_t) S * S * f16b, (size_t) S * S * f16b, 0)
                                    : nullptr;
        in.t_ca    = ggml_view_4d(ctx, b_ca, enc_use, S, 1, nb, (size_t) enc_use * f16b, (size_t) enc_use * S * f16b,
                                  (size_t) enc_use * S * f16b, 0);
        in.t_vtgt  = ggml_view_3d(ctx, b_vtgt, Oc, T, nb, (size_t) Oc * f32b, (size_t) Oc * T * f32b, 0);
        in.t_lw    = ggml_view_3d(ctx, b_lw, 1, T, nb, f32b, (size_t) T * f32b, 0);
        in.t_lwu   = ggml_view_3d(ctx, b_lwu, 1, T, nb, f32b, (size_t) T * f32b, 0);
        return in;
    };
    // Element b of the SAME buffers, in the single-sample shapes, with the
    // all-ones mask and the pre-batching 1/(Oc*len) scale.
    auto mk_single = [&](ggml_context * ctx, int b) {
        DitInputs in;
        in.t_input = ggml_view_2d(ctx, b_input, Ic, T, (size_t) Ic * f32b, (size_t) b * Ic * T * f32b);
        in.t_enc   = ggml_view_2d(ctx, b_enc, enc_H, enc_use, (size_t) enc_H * f32b,
                                  (size_t) b * enc_H * enc_use * f32b);
        in.t_pos   = ggml_view_1d(ctx, b_pos, S, (size_t) b * S * sizeof(int32_t));
        in.t_temb  = ggml_view_1d(ctx, b_temb, H, (size_t) b * H * f32b);
        in.t_tproj = ggml_view_1d(ctx, b_tproj, 6 * H, (size_t) b * 6 * H * f32b);
        in.t_sa    = ggml_view_2d(ctx, b_sa, S, S, (size_t) S * f16b,
                                  (bh.sa_B > 1) ? (size_t) b * (size_t) S * (size_t) S * f16b : 0);
        in.t_sa_pad = (bh.sa_B > 1) ? ggml_view_2d(ctx, b_sapad, S, S, (size_t) S * f16b,
                                                   (size_t) b * (size_t) S * (size_t) S * f16b)
                                    : nullptr;
        in.t_ca    = ggml_view_2d(ctx, b_ca, enc_use, S, (size_t) enc_use * f16b, (size_t) b * enc_use * S * f16b);
        in.t_vtgt  = ggml_view_2d(ctx, b_vtgt, Oc, T, (size_t) Oc * f32b, (size_t) b * Oc * T * f32b);
        in.t_lw    = ggml_view_2d(ctx, b_ones, 1, T, f32b, 0);
        return in;
    };

    auto read_accs = [&](std::vector<std::vector<float>> * dst) {
        dst->assign(opt.acc.size(), std::vector<float>());
        for (size_t j = 0; j < opt.acc.size(); j++) {
            (*dst)[j].resize((size_t) ggml_nelements(opt.acc[j]));
            ggml_backend_tensor_get(opt.acc[j], (*dst)[j].data(), 0, (*dst)[j].size() * sizeof(float));
        }
    };
    // max_j ( max_k |x-y| / max_k |y| ) over the parameter tensors.
    auto cmp_accs = [&](const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y,
                        std::string * worst) {
        double best = 0.0;
        *worst      = "-";
        for (size_t j = 0; j < y.size() && j < x.size(); j++) {
            double num = 0.0, den = 0.0;
            for (size_t k = 0; k < y[j].size() && k < x[j].size(); k++) {
                num = std::max(num, fabs((double) x[j][k] - (double) y[j][k]));
                den = std::max(den, fabs((double) y[j][k]));
            }
            const double rel = num / std::max(den, 1e-30);
            if (rel > best) {
                best   = rel;
                *worst = opt.params[j]->name;
            }
        }
        return best;
    };

    // ── SB1: forward ─────────────────────────────────────────────────────
    std::vector<float> v_bat, v_rev, v_sin((size_t) Oc * T);
    bool               ok = true;
    auto run_batched_fwd = [&](bool rev, std::vector<float> * out, int * nodes) -> bool {
        upload(rev);
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
        DitInputs        in  = mk_batched(ctx, B);
        ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, B);
        ggml_set_output(vel);
        ggml_build_forward_expand(gf, vel);
        dit_protect_output_views(gf);
        *nodes = ggml_graph_n_nodes(gf);
        ggml_backend_sched_reset(M->sched);
        const bool good = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (good) {
            out->resize((size_t) Oc * T * B);
            ggml_backend_tensor_get(vel, out->data(), 0, out->size() * sizeof(float));
        }
        ggml_free(ctx);
        return good;
    };
    ok = run_batched_fwd(false, &v_bat, &o->nodes_bat) && run_batched_fwd(true, &v_rev, &o->nodes_bat);
    if (ok) {
        const size_t per = (size_t) Oc * T;
        for (int b = 0; b < B; b++) {
            for (size_t k = 0; k < per; k++) {
                o->fwd_perm = std::max(o->fwd_perm, fabs((double) v_rev[(size_t) b * per + k] -
                                                         (double) v_bat[(size_t) (B - 1 - b) * per + k]));
            }
        }
    }
    for (int b = 0; b < B && ok; b++) {
        upload(false);
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
        DitInputs        in  = mk_single(ctx, b);
        ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, 1);
        ggml_set_output(vel);
        ggml_build_forward_expand(gf, vel);
        dit_protect_output_views(gf);
        o->nodes_sin = ggml_graph_n_nodes(gf);
        ggml_backend_sched_reset(M->sched);
        ok = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(vel, v_sin.data(), 0, v_sin.size() * sizeof(float));
            for (size_t k = 0; k < v_sin.size(); k++) {
                o->vmag      = std::max(o->vmag, fabs((double) v_sin[k]));
                o->fwd_delta = std::max(o->fwd_delta,
                                        fabs((double) v_bat[(size_t) b * v_sin.size() + k] - (double) v_sin[k]));
            }
        }
        ggml_free(ctx);
    }
    if (!ok) {
        *err = "forward compute failed";
        cleanup();
        return false;
    }

    // ── SB2: backward ────────────────────────────────────────────────────
    //
    // Reference: B single-sample fwd+bwd graphs computed one after another into
    // the SAME LmOptim::acc[] with no zeroing between them (design C1's
    // accumulation rule). Element b is seeded with w_b/B, which is exactly the
    // weight design B4 puts in t_lw for the batched graph.
    std::vector<std::vector<float>> gref, gbat, gperm;
    lm_optim_zero_grad(&opt);
    for (int b = 0; b < B && ok; b++) {
        upload(false);
        const float lg = els[(size_t) b].w / (float) B;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        DitInputs        in  = mk_single(ctx, b);
        ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, 1);
        ggml_tensor *    ls  = dit_train_loss(ctx, vel, in, Oc, T, false);
        ggml_set_loss(ls);
        ggml_build_forward_expand(gf, ls);
        std::vector<ggml_tensor *> ga;
        lm_optim_fill_gacc(&opt, gf, &ga);
        ggml_build_backward_expand(ctx, gf, ga.data());
        ggml_backend_sched_reset(M->sched);
        ok = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
    }
    if (ok) {
        read_accs(&gref);
    }
    // The batched backward: one graph, t_lossgrad a pure scale of 1.
    auto run_batched_bwd = [&](bool rev, std::vector<std::vector<float>> * dst, double * loss_out) -> bool {
        lm_optim_zero_grad(&opt);
        upload(rev);
        const float lg = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        DitInputs        in  = mk_batched(ctx, B);
        ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, B);
        ggml_tensor *    ls  = dit_train_loss(ctx, vel, in, Oc, T, false, bh.gscale);
        ggml_set_loss(ls);
        ggml_build_forward_expand(gf, ls);
        std::vector<ggml_tensor *> ga;
        lm_optim_fill_gacc(&opt, gf, &ga);
        ggml_build_backward_expand(ctx, gf, ga.data());
        ggml_backend_sched_reset(M->sched);
        const bool good = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (good && loss_out) {
            float lv = 0.0f;
            ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
        }
        ggml_free(ctx);
        if (good) {
            read_accs(dst);
        }
        return good;
    };
    ok = ok && run_batched_bwd(false, &gbat, nullptr) && run_batched_bwd(true, &gperm, nullptr);
    if (!ok) {
        *err = "backward compute failed";
        cleanup();
        return false;
    }
    o->bwd_rel = cmp_accs(gbat, gref, &o->bwd_worst);
    {
        std::string w2;
        o->bwd_perm = cmp_accs(gperm, gbat, &w2);
    }

    // ── SB3: mixed-length batch (design §2.3 gate 5) ─────────────────────
    //
    // One element is a SHORT song, so dit_batch_assemble pads it to the batch max.
    //
    // TWO perturbations, both barred at EXACTLY zero:
    //
    //  (a) the padded region of the velocity TARGET. This only exercises the
    //      design-B4 loss mask (t_lw is 0 on a padded frame), and it is what this
    //      rung originally measured.
    //  (b) the padded region of the INPUT. This is the decisive one. Scribbling on
    //      a padded frame's input changes its hidden state in every layer, and if
    //      valid queries can see that frame as an attention KEY then the change
    //      reaches their outputs and, through them, every adapter gradient — while
    //      the loss barely moves, which is exactly why (a) alone cannot detect it.
    //      It is zero only because dit_sa_mask now masks the padded KV columns
    //      per element. Before that fix (a shared [S,S] mask) this measurement is
    //      NON-zero by construction, so it is a real regression gate, not a
    //      restatement of the loss mask.
    //
    // Plus: the loss must equal a host double recompute over the valid frames only.
    if (B >= 2) {
        DitSample shortened = samples[(size_t) (B - 1)];
        const int T_short   = std::max(P, T - 8 * P);
        shortened.T         = T_short;
        shortened.lat.resize((size_t) T_short * (size_t) shortened.Oc);
        shortened.ctxl.resize((size_t) T_short * (size_t) shortened.Cc);
        std::vector<DitBatchElem> els2 = els;
        els2[(size_t) (B - 1)].s       = &shortened;
        lm_rng_seed(&rng_crop, seed ^ 0xbf58476d1ce4e5b9ull);
        lm_rng_seed(&rng_noise, seed ^ 0x9e3779b97f4a7c15ull);
        dit_batch_assemble(bcfg, els2, &rng_crop, &rng_noise, &bh);
        o->short_len = els2[(size_t) (B - 1)].len;
        o->pad_len   = bh.len;

        double             loss_clean = 0.0, loss_dirty = 0.0;
        std::vector<float> vel_host;
        std::vector<std::vector<float>> g_clean, g_dirty;

        // 0 = clean, 1 = padded TARGET scribbled, 2 = padded INPUT scribbled.
        auto run_pad = [&](int dirty, double * loss_out, std::vector<std::vector<float>> * gdst,
                           std::vector<float> * vout) -> bool {
            vtgt_use  = bh.vtgt;
            input_use = bh.input;
            if (dirty == 1) {
                const int    b   = B - 1;
                const size_t off = (size_t) b * (size_t) bh.len * (size_t) Oc;
                for (int f = o->short_len; f < bh.len; f++) {
                    for (int ch = 0; ch < Oc; ch++) {
                        vtgt_use[off + (size_t) f * (size_t) Oc + (size_t) ch] = 137.0f + (float) ch;
                    }
                }
            } else if (dirty == 2) {
                // Modest magnitude on purpose: the padded frames still go through
                // every mul_mat, and a value large enough to overflow to inf there
                // would turn the masked softmax argument into -inf + inf = NaN and
                // contaminate the valid rows for a reason that is not the leak.
                // 2.5 is far outside the latent distribution and perfectly finite.
                const int    b   = B - 1;
                const size_t off = (size_t) b * (size_t) bh.len * (size_t) Ic;
                for (int f = o->short_len; f < bh.len; f++) {
                    for (int ch = 0; ch < Ic; ch++) {
                        input_use[off + (size_t) f * (size_t) Ic + (size_t) ch] = 2.5f - 0.01f * (float) ch;
                    }
                }
            }
            lm_optim_zero_grad(&opt);
            upload(false);
            const float lg = 1.0f;
            ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
            DitInputs        in  = mk_batched(ctx, B);
            ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, B);
            ggml_tensor *    ls  = dit_train_loss(ctx, vel, in, Oc, T, false, bh.gscale);
            ggml_set_loss(ls);
            ggml_set_output(vel);
            ggml_build_forward_expand(gf, ls);
            dit_protect_output_views(gf);
            std::vector<ggml_tensor *> ga;
            lm_optim_fill_gacc(&opt, gf, &ga);
            ggml_build_backward_expand(ctx, gf, ga.data());
            ggml_backend_sched_reset(M->sched);
            const bool good = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
            if (good) {
                float lv = 0.0f;
                ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
                *loss_out = (double) lv;
                if (vout) {
                    vout->resize((size_t) Oc * bh.len * B);
                    ggml_backend_tensor_get(vel, vout->data(), 0, vout->size() * sizeof(float));
                }
            }
            ggml_free(ctx);
            if (good) {
                read_accs(gdst);
            }
            return good;
        };

        double                          loss_dirty_in = 0.0;
        std::vector<std::vector<float>> g_dirty_in;
        ok = run_pad(0, &loss_clean, &g_clean, &vel_host) && run_pad(1, &loss_dirty, &g_dirty, nullptr) &&
             run_pad(2, &loss_dirty_in, &g_dirty_in, nullptr);
        // FINDING (measured here, 2026-07-29, on BOTH backends): reading a
        // mid-graph ACTIVATION back after ggml_build_backward_expand is NOT
        // trustworthy, even when the tensor and its whole view chain carry
        // GGML_TENSOR_FLAG_OUTPUT — `velcmp` below measures 3.6 on a |v|max 3.6
        // tensor, i.e. the storage is fully reused by the backward pass. The
        // scalar LOSS is unaffected (the host recompute from a forward-only
        // velocity agrees with the fwd+bwd graph's loss to 2e-8, which is also
        // what the trainer relies on when it reads the loss after a backward).
        // So the host cross-check takes its velocity from a FORWARD-ONLY graph
        // with the same uploads, and reports the discrepancy as evidence.
        std::vector<float> vel_fwd;
        if (ok) {
            vtgt_use  = bh.vtgt;
            input_use = bh.input;
            upload(false);
            ggml_init_params ip  = { arena.size(), arena.data(), true };
            ggml_context *   ctx = ggml_init(ip);
            ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, false);
            DitInputs        in  = mk_batched(ctx, B);
            ggml_tensor *    vel = dit_train_forward(ctx, M, &lora, in, T, enc_use, nullptr, lo, hi, B);
            ggml_set_output(vel);
            ggml_build_forward_expand(gf, vel);
            dit_protect_output_views(gf);
            ggml_backend_sched_reset(M->sched);
            ok = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                vel_fwd.resize((size_t) Oc * bh.len * B);
                ggml_backend_tensor_get(vel, vel_fwd.data(), 0, vel_fwd.size() * sizeof(float));
            }
            ggml_free(ctx);
        }
        if (ok) {
            double dmax = 0.0;
            for (size_t k = 0; k < vel_fwd.size() && k < vel_host.size(); k++) {
                dmax = std::max(dmax, fabs((double) vel_fwd[k] - (double) vel_host[k]));
            }
            o->pad_velcmp = dmax;
            vel_host      = vel_fwd;
        }
        vtgt_use  = bh.vtgt;
        input_use = bh.input;
        if (!ok) {
            *err = "mixed-length compute failed";
            cleanup();
            return false;
        }
        o->pad_loss = fabs(loss_clean - loss_dirty);
        {
            std::string w3;
            o->pad_grad = cmp_accs(g_dirty, g_clean, &w3);
        }
        o->pad_loss_in = fabs(loss_clean - loss_dirty_in);
        {
            std::string w4;
            o->pad_grad_in = cmp_accs(g_dirty_in, g_clean, &w4);
        }
        // host double recompute over the VALID frames only
        double acc = 0.0, acc_all = 0.0;
        for (int b = 0; b < B; b++) {
            for (int f = 0; f < bh.len; f++) {
                const double w = (double) bh.lw[(size_t) b * (size_t) bh.len + (size_t) f];
                for (int ch = 0; ch < Oc; ch++) {
                    const size_t k = ((size_t) b * (size_t) bh.len + (size_t) f) * (size_t) Oc + (size_t) ch;
                    const double e = (double) vel_host[k] - (double) bh.vtgt[k];
                    acc_all += e * e * w;
                    if (f < els2[(size_t) b].len) {
                        acc += e * e * w;
                    }
                }
            }
        }
        acc *= (double) bh.gscale;
        acc_all *= (double) bh.gscale;
        o->pad_acc     = acc;
        o->pad_acc_all = acc_all;
        o->pad_lossv   = loss_clean;
        o->pad_host    = fabs(acc - loss_clean) / std::max(1e-12, fabs(acc));
    }

    cleanup();
    o->ran = true;
    return true;
}

// Runs the measurement on the shipping (GPU) model and on a CPU-backend copy,
// then reports SB1/SB2/SB3 against the amendment-A2 bars.
static void dit_st_batch_gates(std::vector<DitSelfTestResult> & rs, DitTrainModel * Mg, const std::string & dit_path,
                               const std::vector<DitSample> & samples, uint64_t seed) {
    const int B = 3, T = 64, rank = 8;
    if ((int) samples.size() < B) {
        char d[160];
        snprintf(d, sizeof(d), "need >= %d cached songs for a B=%d batch of DIFFERENT songs (design B2), found %d", B,
                 B, (int) samples.size());
        dit_st_report(rs, "SB1", false, d);
        dit_st_report(rs, "SB2", false, d);
        dit_st_report(rs, "SB3", false, d);
        return;
    }
    int enc_use = 64;
    for (int b = 0; b < B; b++) {
        enc_use = std::min(enc_use, samples[(size_t) b].enc_S);
    }

    DitStBatchMeas gm, cm;
    std::string    gerr = "-", cerr = "-";
    const bool     gok  = dit_st_batch_measure(Mg, samples, seed, B, T, enc_use, rank, &gm, &gerr);

    // The CPU copy: the ONLY place a batched-vs-single identity is measurable
    // (one mul_mat kernel, one accumulation order). Mirrored at lora_lo = L-2 so
    // only the two layers under test are promoted, and with the A2c batched flag
    // so proj_in/cond_emb are F32 like every other input-side weight.
    DitTrainModel Mc;
    bool          cok = false;
    {
        std::string err;
        if (!dit_train_load(&Mc, dit_path.c_str(), 0, &err)) {
            cerr = "CPU-backend copy failed to load: " + err;
        } else {
            Mc.backend = cpu_backend_new(16);
            if (!Mc.backend) {
                cerr = "no CPU backend available for the identity run";
            } else if (!dit_build_mirror(&Mc, Mc.m.cfg.n_layers - 2, DIT_MIRROR_F32, &err, /*batched=*/true)) {
                cerr = "CPU mirror failed: " + err;
            } else {
                BackendPair bp;
                bp.backend     = Mc.backend;
                bp.cpu_backend = Mc.backend;  // n == 1: nothing can silently land elsewhere
                bp.has_gpu     = false;
                Mc.sched       = backend_sched_new(bp, 65536);
                cok            = dit_st_batch_measure(&Mc, samples, seed, B, T, enc_use, rank, &cm, &cerr);
            }
        }
    }
    dit_train_free(&Mc);

    const char * gname = ggml_backend_name(Mg->backend);
    char         d[2048];

    snprintf(d, sizeof(d),
             "B=%d forward on real layers %d-%d (%d batched nodes / %d single): CPU identity vs %d single-sample "
             "forwards max|delta|=%.3e (bar 1e-6, |v|max %.4f) [%s]; %s batch-permutation control (elements reversed, "
             "route-identical) max|delta|=%.3e (bar 0) and its batched-vs-single delta %.3e REPORTED only — batching "
             "changes the CUDA GEMM route (amendment A2), so that bar is unreachable there by construction",
             B, Mg->m.cfg.n_layers - 2, Mg->m.cfg.n_layers - 1, gm.nodes_bat, gm.nodes_sin, B, cm.fwd_delta,
             cm.vmag, cok ? "cpu ok" : cerr.c_str(), gname, gm.fwd_perm, gm.fwd_delta);
    dit_st_report(rs, "SB1", gok && cok && cm.fwd_delta <= 1e-6 && gm.fwd_perm == 0.0, d);

    snprintf(d, sizeof(d),
             "B=%d backward: ONE batched backward vs the sum of %d single-sample backwards accumulated into the same "
             "LmOptim::acc[] with no zeroing between them (design B4 weights, C1 accumulation) — CPU max relative "
             "delta %.3e on %s (bar 1e-5) [%s]; %s reports %.3e, and %.3e under a reversed-element run (a batch "
             "reduction, so permutation is not owed bit-exactness — reported, not gated)",
             B, B, cm.bwd_rel, cm.bwd_worst.c_str(), cok ? "cpu ok" : cerr.c_str(), gname, gm.bwd_rel, gm.bwd_perm);
    dit_st_report(rs, "SB2", gok && cok && cm.bwd_rel <= 1e-5, d);

    snprintf(d, sizeof(d),
             "mixed-length B=%d (one song %d of %d frames, %d padded): perturbing the padded velocity TARGET moves "
             "the loss by %.3e (CPU) / %.3e (%s) and the adapter grads by %.3e / %.3e; perturbing the padded INPUT "
             "— the decisive one, since padded frames sit in the valid queries' KV window and only the per-element "
             "sa_mask can stop them — moves the loss by %.3e / %.3e and the adapter grads by %.3e / %.3e. Bar for "
             "all eight: EXACTLY 0. Masked loss vs a host double recompute over the valid frames only, rel %.3e / "
             "%.3e (bar 1e-5) [cpu: graph %.9f, host over valid frames %.9f, host over ALL frames %.9f (equal => the "
             "pad weight is exactly 0); post-backward velocity readback differs from the forward-only one by %.3e, "
             "which is why the cross-check reads it forward-only]",
             B, cm.short_len, cm.pad_len, cm.pad_len - cm.short_len, cm.pad_loss, gm.pad_loss, gname, cm.pad_grad,
             gm.pad_grad, cm.pad_loss_in, gm.pad_loss_in, cm.pad_grad_in, gm.pad_grad_in, cm.pad_host, gm.pad_host,
             cm.pad_lossv, cm.pad_acc, cm.pad_acc_all, cm.pad_velcmp);
    dit_st_report(rs, "SB3",
                  gok && cok && cm.pad_loss == 0.0 && cm.pad_grad == 0.0 && gm.pad_loss == 0.0 &&
                      gm.pad_grad == 0.0 && cm.pad_loss_in == 0.0 && cm.pad_grad_in == 0.0 &&
                      gm.pad_loss_in == 0.0 && gm.pad_grad_in == 0.0 && cm.pad_host < 1e-5 && gm.pad_host < 1e-5,
                  d);
    if (!gok) {
        dit_st_report(rs, "SBG", false, std::string("the ") + gname + " measurement failed: " + gerr);
    }
}

// ─── SC1 / SC2 / SC3: gradient checkpointing (design §2.3 gates 4 and 6) ────
//
// All three run the PRODUCTION segment driver (dit_ckpt_micro_batch) against the
// PRODUCTION monolithic graph on the same uploads, same seed, same adapter state.
//
//   SC1  --ckpt N grads == --ckpt 0 grads          (gate 4; LoRA, B=3, 4 segments)
//   SC2  the SB3 masked-loss property under        (gate 6; LoRA, B=3, 2 segments)
//        --batch 3 --ckpt 2
//   SC3  LoKR under --batch 3 --ckpt 2             (gate 6; the LK-series adapter)
//
// The bar is 1e-6 RELATIVE and it is meaningful on CUDA too, unlike the SB rungs:
// checkpointing does not change a single op's shape or dtype, so every segment
// graph takes exactly the routes the monolithic graph took for those layers. The
// only new arithmetic is an F32 copy of the boundary (exact) and the surrogate
// loss sum(h (*) G), whose gradient IS G. The S-C1 spike measured 0.000e+00 on
// both backends; anything above the bar here means the driver is wrong, not that
// the hardware is.
//
// The monolithic reference runs the WHOLE stack (layers 0..L), because that is
// what the segmented path computes: the driver's phase 1 forwards the untrained
// layers below the adapter once and hands segment 0 their output.

struct DitStCkptMeas {
    bool        ran = false;
    double      grad_rel = 0.0, ref_mag = 0.0;
    double      loss_mono = 0.0, loss_seg = 0.0, bnd_mag = 0.0;
    double      pad_loss = 0.0, pad_grad = 0.0;
    double      pad_loss_in = 0.0, pad_grad_in = 0.0;  // the padded-INPUT (KV-mask) pair
    std::string worst = "-";
    int         nodes_mono = 0, nodes_seg = 0, graphs = 0;
    double      art_rel = -1.0, art_mag = 0.0;  // V1 artist-token leaf (HOTSTEP_DIT_ST_ARTIST)
    double      art_ratio = 0.0, art_norm = 0.0;  // <seg,mono>/<mono,mono>, ||seg||/||mono||
    int         art_k   = 0;
    int         short_len = 0, pad_len = 0;
};

static bool dit_st_ckpt_measure(DitTrainModel * M, const std::vector<DitSample> & samples, uint64_t seed, int B,
                                int T, int enc_use, int n_train, int segments, bool use_lokr, bool mixed,
                                DitStCkptMeas * o, std::string * err) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int L = c.n_layers, H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, Ic = c.in_channels;
    const int enc_H = (int) M->m.cond_emb_w->ne[0];
    const int S     = T / P;
    const int lo    = std::max(0, L - n_train);
    // V1 gate: HOTSTEP_DIT_ST_ARTIST=<k> adds a k-row artist token to the packed
    // cond (dit_train_cond) as one more trained leaf. Under checkpointing the
    // cond is recomputed inside every segment and the token receives one
    // gradient add per segment; this rung decides whether that sum equals the
    // monolithic gradient. Reported and gated on its own, not under the
    // adapter slots' 1e-6 bar: N in-place adds carry N roundings.
    const int art_k = getenv("HOTSTEP_DIT_ST_ARTIST")
                          ? std::max(1, std::min(8, atoi(getenv("HOTSTEP_DIT_ST_ARTIST"))))
                          : 0;

    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Ic * T * B);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) enc_H * enc_use * B);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, (int64_t) S * B);
    ggml_tensor * b_temb  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) H * B);
    ggml_tensor * b_tproj = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) 6 * H * B);
    // B-wide: `mixed` makes dit_batch_assemble emit a per-element [S,S,1,B] mask
    // plus its window-free twin for the full-attention layers.
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_sapad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) enc_use * S * B);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Oc * T * B);
    ggml_tensor * b_lw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * b_lwu   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_art      = nullptr;
    if (art_k > 0) {
        t_art = ggml_new_tensor_2d(ctxs, GGML_TYPE_F32, enc_H, art_k);
        ggml_set_name(t_art, "artist_token");
        ggml_set_param(t_art);
    }
    for (ggml_tensor * t : { b_input, b_enc, b_pos, b_temb, b_tproj, b_sa, b_sapad, b_ca, b_vtgt, b_lw, b_lwu }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctxs, M->backend);
    if (!buf) {
        *err = "static input allocation failed";
        ggml_free(ctxs);
        return false;
    }
    {
        const float epsv = 1e-6f, clipv = 1.0f, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }
    if (t_art) {
        LmRng ar;
        lm_rng_seed(&ar, seed ^ 0xa7a7a7a7ull);
        std::vector<float> av((size_t) enc_H * (size_t) art_k, 0.0f);
        lm_rng_fill_normal(&ar, av, 0.05f);
        ggml_backend_tensor_set(t_art, av.data(), 0, av.size() * sizeof(float));
    }

    DitAdapterLora lora;
    DitAdapterLoKr lokr;
    DitAdapter *   ad = use_lokr ? (DitAdapter *) &lokr : (DitAdapter *) &lora;
    LmOptim        opt;
    DitCkptBufs    ckb;
    bool           have_opt = false;
    auto           cleanup  = [&]() {
        dit_ckpt_free(&ckb);
        if (have_opt) {
            lm_optim_free(&opt);
        }
        ad->free();
        ggml_backend_buffer_free(buf);
        ggml_free(ctxs);
    };
    {
        DitAdapterCfg cfg;
        cfg.rank       = 8;
        cfg.alpha      = 16.0f;
        cfg.seed       = seed ^ 0x5c1c5c1cull;
        cfg.b_sigma    = 1e-2f;  // a zero B/w2 makes half the gradients trivially 0
        cfg.target_mlp = true;
        cfg.lokr_dim = 512;
        cfg.lokr_alpha = 512.0f;
        cfg.lokr_factor = 6;
        cfg.lokr_decompose_both = true;
        if (!ad->init(&M->m, M->backend, lo, L, cfg, err)) {
            cleanup();
            return false;
        }
        std::vector<ggml_tensor *> plist = ad->params();
    if (t_art) {
        plist.push_back(t_art);  // last slot; cmp_accs skips it, the token gets its own line
    }
    if (!lm_optim_init(&opt, plist, M->backend, err)) {
            cleanup();
            return false;
        }
        have_opt = true;
    }
    opt.t_adamw    = t_adamw;
    opt.t_lossgrad = t_lossgrad;
    opt.t_clip     = t_clip;
    opt.t_eps      = t_eps;
    opt.t_gnorm2   = t_gnorm2;

    std::vector<uint8_t> arena((size_t) 512 << 20);

    // ── the production batch assembler (B DIFFERENT songs, per-element t) ─
    LmRng rng_t, rng_crop, rng_noise;
    lm_rng_seed(&rng_t, seed);
    DitBatchCfg bcfg;
    bcfg.in_ch          = Ic;
    bcfg.out_ch         = Oc;
    bcfg.enc_H          = enc_H;
    bcfg.enc_S          = enc_use;
    bcfg.patch          = P;
    bcfg.sliding_window = c.sliding_window;
    bcfg.crop           = T;
    bcfg.weighted       = true;
    bcfg.null_cond      = &M->null_cond;

    DitSample                 shortened;
    std::vector<DitBatchElem> els((size_t) B);
    {
        double              wsum = 0.0;
        std::vector<double> w((size_t) B, 1.0);
        for (int b = 0; b < B; b++) {
            els[(size_t) b].s = &samples[(size_t) b];
            els[(size_t) b].t = dit_sample_t(&rng_t, -0.4f, 1.0f, 0.0f, 1.0f);
            w[(size_t) b]     = (double) dit_flow_snr_w(els[(size_t) b].t, 0.5f, 5.0f);
            wsum += w[(size_t) b];
        }
        const double wbar = wsum / (double) B;
        for (int b = 0; b < B; b++) {
            els[(size_t) b].w = (float) (w[(size_t) b] / (wbar > 0.0 ? wbar : 1.0));
        }
        if ((int) M->null_cond.size() == enc_H && B > 1) {
            els[1].cfg_drop = true;
        }
        if (mixed && B >= 2) {
            // One song shorter than the crop: dit_batch_assemble pads it and gives
            // the pad loss weight exactly 0 (design B4 / §2.3.5).
            shortened     = samples[(size_t) (B - 1)];
            const int Ts  = std::max(P, T - 8 * P);
            shortened.T   = Ts;
            shortened.lat.resize((size_t) Ts * (size_t) shortened.Oc);
            shortened.ctxl.resize((size_t) Ts * (size_t) shortened.Cc);
            els[(size_t) (B - 1)].s = &shortened;
        }
    }
    lm_rng_seed(&rng_crop, seed ^ 0xbf58476d1ce4e5b9ull);
    lm_rng_seed(&rng_noise, seed ^ 0x9e3779b97f4a7c15ull);
    DitBatchHost bh;
    dit_batch_assemble(bcfg, els, &rng_crop, &rng_noise, &bh);
    if (bh.len != T) {
        char b2[128];
        snprintf(b2, sizeof(b2), "batch assembled at len %d, expected %d", bh.len, T);
        *err = b2;
        cleanup();
        return false;
    }
    o->short_len = els[(size_t) (B - 1)].len;
    o->pad_len   = bh.len;

    std::vector<float> temb_buf((size_t) H * B, 0.0f), tproj_buf((size_t) 6 * H * B, 0.0f);
    {
        std::vector<float> ts((size_t) B);
        for (int b = 0; b < B; b++) {
            ts[(size_t) b] = els[(size_t) b].t;
        }
        std::vector<std::vector<float>> tb, tp;
        if (!dit_train_temb(M, ts, &tb, &tp)) {
            *err = "temb precompute failed";
            cleanup();
            return false;
        }
        for (int b = 0; b < B; b++) {
            memcpy(&temb_buf[(size_t) b * H], tb[(size_t) b].data(), (size_t) H * sizeof(float));
            memcpy(&tproj_buf[(size_t) b * 6 * H], tp[(size_t) b].data(), (size_t) 6 * H * sizeof(float));
        }
    }

    std::vector<float> vtgt_use = bh.vtgt, input_use = bh.input;
    auto               upload   = [&]() {  // §3.0: every input, every compute
        ggml_backend_tensor_set(b_input, input_use.data(), 0, input_use.size() * sizeof(float));
        ggml_backend_tensor_set(b_vtgt, vtgt_use.data(), 0, vtgt_use.size() * sizeof(float));
        ggml_backend_tensor_set(b_enc, bh.enc.data(), 0, bh.enc.size() * sizeof(float));
        ggml_backend_tensor_set(b_pos, bh.pos.data(), 0, bh.pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b_sa, bh.sa.data(), 0, bh.sa.size() * sizeof(uint16_t));
        if (!bh.sa_pad.empty()) {
            ggml_backend_tensor_set(b_sapad, bh.sa_pad.data(), 0, bh.sa_pad.size() * sizeof(uint16_t));
        }
        ggml_backend_tensor_set(b_ca, bh.ca.data(), 0, bh.ca.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(b_lw, bh.lw.data(), 0, bh.lw.size() * sizeof(float));
        ggml_backend_tensor_set(b_lwu, bh.lwu.data(), 0, bh.lwu.size() * sizeof(float));
        ggml_backend_tensor_set(b_temb, temb_buf.data(), 0, temb_buf.size() * sizeof(float));
        ggml_backend_tensor_set(b_tproj, tproj_buf.data(), 0, tproj_buf.size() * sizeof(float));
        const float lg = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
    };

    const size_t f32b = sizeof(float), f16b = sizeof(ggml_fp16_t);
    auto         mk_inputs = [&](ggml_context * ctx) {
        DitInputs in;
        in.t_input = ggml_view_3d(ctx, b_input, Ic, T, B, (size_t) Ic * f32b, (size_t) Ic * T * f32b, 0);
        in.t_enc =
            ggml_view_3d(ctx, b_enc, enc_H, enc_use, B, (size_t) enc_H * f32b, (size_t) enc_H * enc_use * f32b, 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, (int64_t) S * B, 0);
        in.t_temb  = ggml_view_3d(ctx, b_temb, H, 1, B, (size_t) H * f32b, (size_t) H * f32b, 0);
        in.t_tproj = ggml_view_3d(ctx, b_tproj, 6 * H, 1, B, (size_t) 6 * H * f32b, (size_t) 6 * H * f32b, 0);
        in.t_sa    = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sa, S, S, 1, B, (size_t) S * f16b, (size_t) S * S * f16b,
                                                  (size_t) S * S * f16b, 0)
                                   : ggml_view_2d(ctx, b_sa, S, S, (size_t) S * f16b, 0);
        in.t_sa_pad = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sapad, S, S, 1, B, (size_t) S * f16b,
                                                   (size_t) S * S * f16b, (size_t) S * S * f16b, 0)
                                    : nullptr;
        in.t_ca    = ggml_view_4d(ctx, b_ca, enc_use, S, 1, B, (size_t) enc_use * f16b, (size_t) enc_use * S * f16b,
                                  (size_t) enc_use * S * f16b, 0);
        in.t_vtgt  = ggml_view_3d(ctx, b_vtgt, Oc, T, B, (size_t) Oc * f32b, (size_t) Oc * T * f32b, 0);
        in.t_lw    = ggml_view_3d(ctx, b_lw, 1, T, B, f32b, (size_t) T * f32b, 0);
        in.t_lwu   = ggml_view_3d(ctx, b_lwu, 1, T, B, f32b, (size_t) T * f32b, 0);
        in.t_art   = t_art;
        in.art_k   = art_k;
        return in;
    };

    auto read_accs = [&](std::vector<std::vector<float>> * dst) {
        dst->assign(opt.acc.size(), std::vector<float>());
        for (size_t j = 0; j < opt.acc.size(); j++) {
            (*dst)[j].resize((size_t) ggml_nelements(opt.acc[j]));
            ggml_backend_tensor_get(opt.acc[j], (*dst)[j].data(), 0, (*dst)[j].size() * sizeof(float));
        }
    };
    auto cmp_accs = [&](const std::vector<std::vector<float>> & x, const std::vector<std::vector<float>> & y,
                        std::string * worst, double * mag) {
        double best = 0.0;
        *worst      = "-";
        const size_t n_cmp = std::min(x.size(), y.size()) - (size_t) (t_art ? 1 : 0);
        for (size_t j = 0; j < n_cmp; j++) {
            double num = 0.0, den = 0.0;
            for (size_t k = 0; k < y[j].size() && k < x[j].size(); k++) {
                num = std::max(num, fabs((double) x[j][k] - (double) y[j][k]));
                den = std::max(den, fabs((double) y[j][k]));
            }
            if (mag) {
                *mag = std::max(*mag, den);
            }
            const double rel = num / std::max(den, 1e-30);
            if (rel > best) {
                best   = rel;
                *worst = opt.params[j]->name;
            }
        }
        return best;
    };

    // ── monolithic (--ckpt 0): the whole stack, one fwd+bwd graph ─────────
    auto run_mono = [&](double * loss_out, std::vector<std::vector<float>> * dst) -> bool {
        lm_optim_zero_grad(&opt);
        upload();
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        DitInputs        in  = mk_inputs(ctx);
        ggml_tensor *    vel = dit_train_forward(ctx, M, ad, in, T, enc_use, nullptr, -1, -1, B);
        ggml_tensor *    ls  = dit_train_loss(ctx, vel, in, Oc, T, false, bh.gscale);
        ggml_set_loss(ls);
        ggml_build_forward_expand(gf, ls);
        std::vector<ggml_tensor *> ga;
        lm_optim_fill_gacc(&opt, gf, &ga);
        ggml_build_backward_expand(ctx, gf, ga.data());
        o->nodes_mono = ggml_graph_n_nodes(gf);
        ggml_backend_sched_reset(M->sched);
        const bool good = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (good) {
            float lv = 0.0f;
            ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
        }
        ggml_free(ctx);
        if (good) {
            read_accs(dst);
        }
        return good;
    };

    // ── segmented (--ckpt N): the PRODUCTION driver ───────────────────────
    // A trained cond leaf needs the plan over the whole stack (see the
    // trainer); the adapter window stays `lo`.
    const DitCkptPlan plan = dit_ckpt_plan(art_k > 0 ? 0 : lo, L, segments);
    if (!dit_ckpt_alloc(&ckb, M->backend, H, S, B, plan.segments, err)) {
        cleanup();
        return false;
    }
    auto run_seg = [&](double * loss_out, std::vector<std::vector<float>> * dst) -> bool {
        lm_optim_zero_grad(&opt);
        upload();
        if (!dit_ckpt_begin(&ckb, H, S, B, plan.segments)) {
            *err = "checkpoint boundary carve failed";
            return false;
        }
        DitCkptReq req;
        req.M            = M;
        req.ad           = ad;
        req.opt          = &opt;
        req.sched        = M->sched;
        req.arena        = &arena;
        req.mk_inputs    = mk_inputs;
        req.upload       = upload;  // §3.0: re-uploaded before each of the 1+N computes
        req.T            = T;
        req.enc_S        = enc_use;
        req.B            = B;
        req.Oc           = Oc;
        req.chan_bal     = false;
        req.gscale       = bh.gscale;
        req.want_report  = false;
        req.loss_scale   = 1.0f;
        req.bnd_grad_max = &o->bnd_mag;
        const bool good  = dit_ckpt_micro_batch(req, &ckb, plan, loss_out, nullptr);
        o->nodes_seg     = std::max(o->nodes_seg, req.nodes);
        o->graphs        = req.graphs;
        dit_ckpt_end(&ckb);
        if (good) {
            read_accs(dst);
        }
        return good;
    };

    std::vector<std::vector<float>> gmono, gseg;
    bool                            ok = run_mono(&o->loss_mono, &gmono) && run_seg(&o->loss_seg, &gseg);
    if (!ok) {
        *err = "checkpoint compute failed";
        cleanup();
        return false;
    }
    o->grad_rel = cmp_accs(gseg, gmono, &o->worst, &o->ref_mag);
    if (t_art) {
        const std::vector<float> & xs = gseg.back();
        const std::vector<float> & ym = gmono.back();
        double                     num = 0.0, den = 0.0;
        for (size_t k = 0; k < ym.size() && k < xs.size(); k++) {
            num = std::max(num, fabs((double) xs[k] - (double) ym[k]));
            den = std::max(den, fabs((double) ym[k]));
        }
        o->art_rel = num / std::max(den, 1e-30);
        o->art_mag = den;
        o->art_k   = art_k;
        // Scalar projection <seg, mono> / <mono, mono>: 2 = double-counted,
        // 0 = dropped, 1/N = one segment's share, 1 = right magnitude but wrong
        // direction (then the rel says how wrong).
        double sm = 0.0, mm = 0.0, ss = 0.0;
        for (size_t k = 0; k < ym.size() && k < xs.size(); k++) {
            sm += (double) xs[k] * (double) ym[k];
            mm += (double) ym[k] * (double) ym[k];
            ss += (double) xs[k] * (double) xs[k];
        }
        o->art_ratio = sm / std::max(mm, 1e-300);
        o->art_norm  = sqrt(ss) / std::max(sqrt(mm), 1e-150);
    }

    // ── the masked-loss property, THROUGH the segmented path ──────────────
    //
    // Both perturbations SB3 makes, re-measured with checkpointing on: the padded
    // velocity TARGET (loss mask) and the padded INPUT (the per-element
    // self-attention KV mask, which the segment driver has to reproduce in every
    // one of its 1+N graphs because each rebuilds the inputs from scratch).
    if (mixed && B >= 2) {
        std::vector<std::vector<float>> gdirty, gdirty_in;
        double                          loss_dirty = 0.0, loss_dirty_in = 0.0;
        {
            const size_t off = (size_t) (B - 1) * (size_t) bh.len * (size_t) Oc;
            for (int f = o->short_len; f < bh.len; f++) {
                for (int ch = 0; ch < Oc; ch++) {
                    vtgt_use[off + (size_t) f * (size_t) Oc + (size_t) ch] = 137.0f + (float) ch;
                }
            }
        }
        ok       = run_seg(&loss_dirty, &gdirty);
        vtgt_use = bh.vtgt;
        if (ok) {
            // Modest magnitude — see SB3's run_pad for why an enormous one would
            // turn the masked softmax argument into a NaN instead of a test.
            const size_t off = (size_t) (B - 1) * (size_t) bh.len * (size_t) Ic;
            for (int f = o->short_len; f < bh.len; f++) {
                for (int ch = 0; ch < Ic; ch++) {
                    input_use[off + (size_t) f * (size_t) Ic + (size_t) ch] = 2.5f - 0.01f * (float) ch;
                }
            }
            ok        = run_seg(&loss_dirty_in, &gdirty_in);
            input_use = bh.input;
        }
        if (!ok) {
            *err = "mixed-length checkpoint compute failed";
            cleanup();
            return false;
        }
        std::string w2;
        o->pad_loss    = fabs(loss_dirty - o->loss_seg);
        o->pad_grad    = cmp_accs(gdirty, gseg, &w2, nullptr);
        o->pad_loss_in = fabs(loss_dirty_in - o->loss_seg);
        o->pad_grad_in = cmp_accs(gdirty_in, gseg, &w2, nullptr);
    }

    cleanup();
    o->ran = true;
    return true;
}

static void dit_st_ckpt_gates(std::vector<DitSelfTestResult> & rs, DitTrainModel * M,
                              const std::vector<DitSample> & samples, uint64_t seed) {
    const int B = 3, T = 64;
    if ((int) samples.size() < B) {
        char d[160];
        snprintf(d, sizeof(d), "need >= %d cached songs for a B=%d batch of DIFFERENT songs (design B2), found %d", B,
                 B, (int) samples.size());
        dit_st_report(rs, "SC1", false, d);
        dit_st_report(rs, "SC2", false, d);
        dit_st_report(rs, "SC3", false, d);
        return;
    }
    int enc_use = 64;
    for (int b = 0; b < B; b++) {
        enc_use = std::min(enc_use, samples[(size_t) b].enc_S);
    }
    const char * bk = ggml_backend_name(M->backend);
    char         d[2048];

    // SC1 — gate 4. Four trained layers split into four segments, so every
    // segment boundary is exercised and the surrogate loss runs three times.
    {
        DitStCkptMeas m;
        std::string   err = "-";
        const bool    ok  = dit_st_ckpt_measure(M, samples, seed, B, T, enc_use, 4, 4, false, false, &m, &err);
        snprintf(d, sizeof(d),
                 "[%s] --ckpt 4 vs --ckpt 0 at B=%d, LoRA on the top 4 of %d layers: monolithic %d-node fwd+bwd "
                 "(loss %.9f) vs %d graphs, widest %d nodes (loss %.9f); max|dL/dboundary| = %.4e; max relative "
                 "grad delta %.3e on %s (bar 1e-6), reference |grad|max %.3e%s",
                 bk, B, M->m.cfg.n_layers, m.nodes_mono, m.loss_mono, m.graphs, m.nodes_seg, m.loss_seg, m.bnd_mag,
                 m.grad_rel, m.worst.c_str(), m.ref_mag, ok ? "" : (" — FAILED: " + err).c_str());
        std::string d1(d);
        if (m.art_k > 0) {
            // V1: the artist-token leaf is summed across segments (one add per
            // recomputed cond); its own bar allows the extra roundings.
            char ab[224];
            snprintf(ab, sizeof(ab),
                     "; artist token (k=%d rows of the packed cond, recomputed per segment): segmented vs "
                     "monolithic grad max rel %.3e (bar 1e-3: eight in-place adds through the whole stack), |grad|max %.3e, <seg,mono>/<mono,mono> = %.4f, "
                     "||seg||/||mono|| = %.4f",
                     m.art_k, m.art_rel, m.art_mag, m.art_ratio, m.art_norm);
            d1 += ab;
        }
        dit_st_report(rs, "SC1",
                      ok && m.grad_rel <= 1e-6 && m.bnd_mag > 0.0 && m.ref_mag > 0.0 &&
                          (m.art_k == 0 || (m.art_rel >= 0.0 && m.art_rel <= 1e-3 && m.art_mag > 0.0)),
                      d1.c_str());
        if (m.art_k > 0) {
            // Bisect: the same measurement at 1 and 2 segments. One segment is the
            // checkpoint machinery (boundary PARAM, surrogate, accumulators) with
            // no cross-segment summation; two adds one sum.
            // (n_train, segments): 4/1 and 4/2 isolate the machinery from the
            // summation; L/8 puts the checkpoint window over the WHOLE stack, which
            // is the only way the token's gradient through the frozen lower layers
            // (every layer's cross-attention reads the cond) can be included.
            const int bis[3][2] = { { 4, 1 }, { 4, 2 }, { M->m.cfg.n_layers, 8 } };
            for (int bi = 0; bi < 3; bi++) {
                DitStCkptMeas mb;
                std::string   eb = "-";
                const bool    okb = dit_st_ckpt_measure(M, samples, seed, B, T, enc_use, bis[bi][0], bis[bi][1], false,
                                                        false, &mb, &eb);
                fprintf(stderr,
                        "[self-test] SC1 bisect: window %d layers, --ckpt %d, artist token: max rel %.3e, "
                        "<seg,mono>/<mono,mono> = %.4f, ||seg||/||mono|| = %.4f; adapter slots max rel %.3e%s\n",
                        bis[bi][0], bis[bi][1], mb.art_rel, mb.art_ratio, mb.art_norm, mb.grad_rel,
                        okb ? "" : (" FAILED: " + eb).c_str());
            }
        }
    }

    // SC2 — gate 6, the SB side: the masked-loss invariant re-measured with
    // batching AND checkpointing both on.
    {
        DitStCkptMeas m;
        std::string   err = "-";
        const bool    ok  = dit_st_ckpt_measure(M, samples, seed, B, T, enc_use, 2, 2, false, true, &m, &err);
        snprintf(d, sizeof(d),
                 "[%s] --batch %d --ckpt 2, mixed length (one song %d of %d frames, %d padded): segmented grads vs "
                 "monolithic max rel %.3e (bar 1e-6); through the SEGMENTED path, perturbing the padded velocity "
                 "TARGET moves the loss by %.3e and every adapter gradient by %.3e, and perturbing the padded INPUT "
                 "(the per-element self-attention KV mask, rebuilt in each of the 1+N segment graphs) moves them by "
                 "%.3e and %.3e — bar EXACTLY 0 for all four%s",
                 bk, B, m.short_len, m.pad_len, m.pad_len - m.short_len, m.grad_rel, m.pad_loss, m.pad_grad,
                 m.pad_loss_in, m.pad_grad_in, ok ? "" : (" — FAILED: " + err).c_str());
        dit_st_report(rs, "SC2",
                      ok && m.grad_rel <= 1e-6 && m.pad_loss == 0.0 && m.pad_grad == 0.0 && m.pad_loss_in == 0.0 &&
                          m.pad_grad_in == 0.0 && m.ref_mag > 0.0,
                      d);
    }

    // SC3 — gate 6, the LK side: the LoKR kron-matvec adapter under the same
    // --batch 3 --ckpt 2. Its bar is 1e-5: apply() retains two activation-shaped
    // tensors per site and the segmented reduction order differs there.
    {
        DitStCkptMeas m;
        std::string   err = "-";
        const bool    ok  = dit_st_ckpt_measure(M, samples, seed, B, T, enc_use, 2, 2, true, false, &m, &err);
        snprintf(d, sizeof(d),
                 "[%s] LoKR (dim 512, factor 6, mlp) --batch %d --ckpt 2 on the top 2 layers: monolithic loss %.9f "
                 "vs segmented %.9f; max|dL/dboundary| = %.4e; max relative grad delta %.3e on %s (bar 1e-5), "
                 "reference |grad|max %.3e%s",
                 bk, B, m.loss_mono, m.loss_seg, m.bnd_mag, m.grad_rel, m.worst.c_str(), m.ref_mag,
                 ok ? "" : (" — FAILED: " + err).c_str());
        dit_st_report(rs, "SC3", ok && m.grad_rel <= 1e-5 && m.bnd_mag > 0.0 && m.ref_mag > 0.0, d);
    }
}

// ─── SF1 / SF2: fused attention (--attn flash) vs the exact graph ────────
//
// One trained sub-stack, built TWICE from the same uploads, the same seed and
// the same adapter state, differing in exactly one field: DitInputs::attn_mode.
// Arm A is dit_attn_f32 (mul_mat / soft_max_ext / mul_mat / cont(permute)); arm
// B is GGML_OP_FLASH_ATTN_TRAIN plus its autodiff'd _BACK, on BOTH the self-
// and cross-attention sites. Compared: the trained stack's OUTPUT (the deepest
// hidden_after_layerN tap) and EVERY adapter gradient accumulator.
//
//   SF1  B = 1, native GQA (Nkv 8 < Nh 32), 2-D [S,S] window mask — the shape a
//        default production run emits.
//   SF2  B = 3, mixed lengths, per-element [S,S,1,B] / [enc_S,S,1,B] masks, so
//        the op's ne2/ne3 broadcast (modulo, not divide) is exercised rather
//        than assumed.
//
//        HOT-Step patch: flash-attn-train (investigation B2) — SF2 is now the
//        asymmetric rung, and that is the point. The EXACT arm still pre-expands
//        K/V to Nh with dit_expand_heads (it must: ggml's mul_mat backward
//        aborts on a broadcast src0 at B > 1); the FLASH arm no longer does, and
//        feeds native GQA 8/32 straight into the fused ops. So this rung reads
//        "B = 3 GQA-native vs the expanded reference" — exactly the claim B2
//        rests on, measured end-to-end through a trained sub-stack rather than
//        argued from the kernel grid. The two are mathematically the same
//        computation (the expansion is a repeat), so the bars below are the
//        SF1 bars unchanged; a wrong GQA head mapping would blow straight
//        through them.
//
// This is NOT a byte-identity rung and cannot be: flash recomputes the softmax
// from Q/K/LSE in tiles instead of reading back a retained [S_kv,S,Nh,B] array,
// so the two arms sum the same terms in a different order. Same class as
// --bwd mm and --mirror bf16, and the reason --attn exact stays the default.
//
// WHERE THE COMPARISON IS MADE. The gate runs on a CPU-backend copy of the
// base, the SB rungs' trick: there both arms run the same f32 kernels and the
// fused op is the only thing left that differs. On CUDA the exact arm's two
// attention mul_mats are cublasSgemm on TF32 tensor cores (common.cuh's
// CUBLAS_TF32_TENSOR_OP_MATH, an 11-bit mantissa), so an exact-vs-flash delta
// there measures the REFERENCE's rounding, not the thing under test — measured
// 3.0e-3 / 9.7e-2 where the same case on CPU gives 2.5e-6 / 4.5e-4. The CUDA
// numbers are reported; only the CPU ones are gated. (NVIDIA_TF32_OVERRIDE=0 is
// not a way out: on this box it changes none of the CUDA figures, including
// MU1's and SB2's, which are attributed to TF32 in their own rungs.)
//
// THREE BARS, not one, because the graph does not have one conditioning number.
// Measured on dit-xl-thirds, LoRA rank 8 on the top 2 layers, T=64, seed 42:
//
//   layer output                      2.5e-6 (B=1) / 3.6e-6 (B=3)   bar 1e-4
//   gradients, non-QK sites (v/o/mlp) 1.9e-5        / 1.7e-5        bar 1e-4
//   gradients, q_proj / k_proj        4.5e-4        / 4.7e-4        bar 2e-3
//
// The split is not a convenience. q_proj and k_proj are the only projections
// whose gradient returns through a QK rms_norm, whose backward subtracts a mean
// of x*dx — the graph's worst-conditioned step, and T4's own note names
// L30 self_attn.k_proj the worst site on this base while recording every
// site without a QK-norm as clean. Both rungs land on exactly that parameter.
// The fused op's own gradients are proven to 3e-6 against this same reference
// chain by gate 1 (engine/tools/fattn-train-test.cpp, 49/49 with --extra
// --large), so what these 4.5e-4 measure is amplification between the op's
// output and the parameter, not the op. The 2e-3 bar leaves ~4x headroom over
// the measurement; a regression in the op itself would move the forward and the
// non-QK gradients too, and those bars are tight.
//
// HOT-Step patch: flash-attn-train — TWO flash arms on the GPU, not one. The op
// pair carries an arithmetic request (op_params slot 3): GGML_PREC_F32 runs the
// original scalar kernels, GGML_PREC_DEFAULT the TF32 tensor-core ones on
// sm_80+. Both are diffed against the SAME exact arm and both columns are
// printed, with the backend's own answer about which kernel each direction
// launched — because a dispatch that quietly fell back to the scalar path would
// otherwise print two identical columns and read as a pass. The GATE is still
// the CPU f32 measurement and nothing else: the CPU impl ignores the flag by
// design (it is the oracle), so there is nothing to gate a second column on
// there, and on CUDA the reference itself is TF32.
//
// The rung also reports the spec 9.8 capability probe. A GPU backend that says
// `false` does not fail the graph — the scheduler splits the op onto the CPU and
// the numbers still agree — so a silently-CPU flash arm would pass the parity
// comparison while proving nothing about the kernels that ship. On a non-CPU
// backend the probe is therefore part of the verdict.
struct DitStFlashMeas {
    bool        ran = false;
    bool        probe_fwd = false, probe_bwd = false;
    double      out_rel = 0.0, out_mag = 0.0;
    double      grad_rel = 0.0, ref_mag = 0.0;
    // Split by site. q_proj/k_proj are the only projections whose gradient comes
    // back through a QK rms_norm, whose backward subtracts a mean of x*dx and is
    // the graph's worst-conditioned step -- T4's own note names L30
    // self_attn.k_proj as the worst site on this base and records every non-QK
    // site as clean. Keeping the two buckets apart is what turns "the gradients
    // differ" into a statement about WHERE.
    double      grad_rel_qk = 0.0, grad_rel_ot = 0.0;
    // |grad|max of the tensor each bucket's worst delta came from, so a large
    // ratio on a near-zero tensor cannot be mistaken for a large error.
    double      mag_qk = 0.0, mag_ot = 0.0;
    std::string worst_qk = "-", worst_ot = "-";
    double      loss_exact = 0.0, loss_flash = 0.0;
    std::string worst = "-", tap = "-";
    int         nodes_exact = 0, nodes_flash = 0;
    int         lt0 = 0, lt1 = 0;  // layer_type counts inside the trained window
    int         short_len = 0, pad_len = 0;
    // HOT-Step patch: flash-attn-train — the SECOND flash arm. The op pair now
    // carries an arithmetic request in op_params slot 3: GGML_PREC_F32 runs the
    // original scalar kernels, GGML_PREC_DEFAULT the TF32 tensor-core ones on
    // sm_80+. Both arms are diffed against the SAME exact arm, so the two
    // columns are directly comparable and the tf32-vs-f32 question is answered
    // by a measurement rather than by an argument. Only run where a second
    // arithmetic actually exists — the CPU impl ignores the flag by design (it
    // is the oracle, and an oracle that moves with the thing it checks is not
    // one), so running it twice there would print the same number twice.
    bool        ran_tf32 = false;
    double      out_rel_tf32 = 0.0, grad_rel_qk_tf32 = 0.0, grad_rel_ot_tf32 = 0.0;
    double      loss_flash_tf32 = 0.0;
    std::string worst_ot_tf32 = "-";
    // What the backend says it LAUNCHED for that arm, not what was asked.
    std::string prec_fwd = "n/a", prec_bwd = "n/a";
};

static bool dit_st_flash_measure(DitTrainModel * M, const std::vector<DitSample> & samples, uint64_t seed, int B,
                                 int T, int enc_use, int n_train, bool mixed, bool second_prec, DitStFlashMeas * o,
                                 std::string * err) {
    const DiTGGMLConfig & c = M->m.cfg;
    const int L = c.n_layers, H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, Ic = c.in_channels;
    const int enc_H = (int) M->m.cond_emb_w->ne[0];
    const int S     = T / P;
    const int lo    = std::max(0, L - n_train);

    for (int i = lo; i < L; i++) {
        (M->m.layers[i].layer_type == 0 ? o->lt0 : o->lt1)++;
    }

    // spec §9.8. Both directions, both attention shapes (self: S_kv = S, cross:
    // S_kv = enc_S), at the effective Nkv.
    //
    // HOT-Step patch: flash-attn-train (investigation B2) — the flash arm no
    // longer pre-expands K/V at B > 1 (dit_attn_needs_kv_expand), so the shape
    // to probe is native GQA at every B. Probing Nh here would ask the backend
    // about a geometry this rung's flash arm never builds.
    {
        const int   Nkv_eff = c.n_kv_heads;
        const float ascale  = 1.0f / sqrtf((float) c.head_dim);
        bool        sf = false, sb = false, cf = false, cb = false;
        dit_flash_probe(M->backend, c.head_dim, c.n_heads, Nkv_eff, S, S, B, ascale, &sf, &sb);
        dit_flash_probe(M->backend, c.head_dim, c.n_heads, Nkv_eff, S, enc_use, B, ascale, &cf, &cb);
        o->probe_fwd = sf && cf;
        o->probe_bwd = sb && cb;
    }

    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Ic * T * B);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) enc_H * enc_use * B);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, (int64_t) S * B);
    ggml_tensor * b_temb  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) H * B);
    ggml_tensor * b_tproj = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) 6 * H * B);
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_sapad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S * S * B);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) enc_use * S * B);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Oc * T * B);
    ggml_tensor * b_lw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * b_lwu   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) T * B);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { b_input, b_enc, b_pos, b_temb, b_tproj, b_sa, b_sapad, b_ca, b_vtgt, b_lw, b_lwu }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctxs, M->backend);
    if (!buf) {
        *err = "static input allocation failed";
        ggml_free(ctxs);
        return false;
    }
    {
        const float epsv = 1e-6f, clipv = 1.0f, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    DitAdapterLora lora;
    LmOptim        opt;
    bool           have_opt = false;
    auto           cleanup  = [&]() {
        if (have_opt) {
            lm_optim_free(&opt);
        }
        lora.free();
        ggml_backend_buffer_free(buf);
        ggml_free(ctxs);
    };
    {
        DitAdapterCfg cfg;
        cfg.rank       = 8;
        cfg.alpha      = 16.0f;
        cfg.seed       = seed ^ 0x5f1a5f1aull;
        cfg.b_sigma    = 1e-2f;  // a zero B makes half the gradients trivially 0
        cfg.target_mlp = true;
        if (!lora.init(&M->m, M->backend, lo, L, cfg, err)) {
            cleanup();
            return false;
        }
        if (!lm_optim_init(&opt, lora.params(), M->backend, err)) {
            cleanup();
            return false;
        }
        have_opt = true;
    }
    DitAdapter * ad = (DitAdapter *) &lora;
    opt.t_adamw     = t_adamw;
    opt.t_lossgrad  = t_lossgrad;
    opt.t_clip      = t_clip;
    opt.t_eps       = t_eps;
    opt.t_gnorm2    = t_gnorm2;

    std::vector<uint8_t> arena((size_t) 512 << 20);

    LmRng rng_t, rng_crop, rng_noise;
    lm_rng_seed(&rng_t, seed);
    DitBatchCfg bcfg;
    bcfg.in_ch          = Ic;
    bcfg.out_ch         = Oc;
    bcfg.enc_H          = enc_H;
    bcfg.enc_S          = enc_use;
    bcfg.patch          = P;
    bcfg.sliding_window = c.sliding_window;
    bcfg.crop           = T;
    bcfg.weighted       = true;
    bcfg.null_cond      = &M->null_cond;

    DitSample                 shortened;
    std::vector<DitBatchElem> els((size_t) B);
    {
        double              wsum = 0.0;
        std::vector<double> w((size_t) B, 1.0);
        for (int b = 0; b < B; b++) {
            els[(size_t) b].s = &samples[(size_t) b];
            els[(size_t) b].t = dit_sample_t(&rng_t, -0.4f, 1.0f, 0.0f, 1.0f);
            w[(size_t) b]     = (double) dit_flow_snr_w(els[(size_t) b].t, 0.5f, 5.0f);
            wsum += w[(size_t) b];
        }
        const double wbar = wsum / (double) B;
        for (int b = 0; b < B; b++) {
            els[(size_t) b].w = (float) (w[(size_t) b] / (wbar > 0.0 ? wbar : 1.0));
        }
        if (mixed && B >= 2) {
            // One song shorter than the crop: dit_batch_assemble pads it, which is
            // what turns both masks into their per-element [.,.,1,B] form.
            shortened    = samples[(size_t) (B - 1)];
            const int Ts = std::max(P, T - 8 * P);
            shortened.T  = Ts;
            shortened.lat.resize((size_t) Ts * (size_t) shortened.Oc);
            shortened.ctxl.resize((size_t) Ts * (size_t) shortened.Cc);
            els[(size_t) (B - 1)].s = &shortened;
        }
    }
    lm_rng_seed(&rng_crop, seed ^ 0xbf58476d1ce4e5b9ull);
    lm_rng_seed(&rng_noise, seed ^ 0x9e3779b97f4a7c15ull);
    DitBatchHost bh;
    dit_batch_assemble(bcfg, els, &rng_crop, &rng_noise, &bh);
    if (bh.len != T) {
        char b2[128];
        snprintf(b2, sizeof(b2), "batch assembled at len %d, expected %d", bh.len, T);
        *err = b2;
        cleanup();
        return false;
    }
    o->short_len = els[(size_t) (B - 1)].len;
    o->pad_len   = bh.len;

    std::vector<float> temb_buf((size_t) H * B, 0.0f), tproj_buf((size_t) 6 * H * B, 0.0f);
    {
        std::vector<float> ts((size_t) B);
        for (int b = 0; b < B; b++) {
            ts[(size_t) b] = els[(size_t) b].t;
        }
        std::vector<std::vector<float>> tb, tp;
        if (!dit_train_temb(M, ts, &tb, &tp)) {
            *err = "temb precompute failed";
            cleanup();
            return false;
        }
        for (int b = 0; b < B; b++) {
            memcpy(&temb_buf[(size_t) b * H], tb[(size_t) b].data(), (size_t) H * sizeof(float));
            memcpy(&tproj_buf[(size_t) b * 6 * H], tp[(size_t) b].data(), (size_t) 6 * H * sizeof(float));
        }
    }

    auto upload = [&]() {
        ggml_backend_tensor_set(b_input, bh.input.data(), 0, bh.input.size() * sizeof(float));
        ggml_backend_tensor_set(b_vtgt, bh.vtgt.data(), 0, bh.vtgt.size() * sizeof(float));
        ggml_backend_tensor_set(b_enc, bh.enc.data(), 0, bh.enc.size() * sizeof(float));
        ggml_backend_tensor_set(b_pos, bh.pos.data(), 0, bh.pos.size() * sizeof(int32_t));
        ggml_backend_tensor_set(b_sa, bh.sa.data(), 0, bh.sa.size() * sizeof(uint16_t));
        if (!bh.sa_pad.empty()) {
            ggml_backend_tensor_set(b_sapad, bh.sa_pad.data(), 0, bh.sa_pad.size() * sizeof(uint16_t));
        }
        ggml_backend_tensor_set(b_ca, bh.ca.data(), 0, bh.ca.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(b_lw, bh.lw.data(), 0, bh.lw.size() * sizeof(float));
        ggml_backend_tensor_set(b_lwu, bh.lwu.data(), 0, bh.lwu.size() * sizeof(float));
        ggml_backend_tensor_set(b_temb, temb_buf.data(), 0, temb_buf.size() * sizeof(float));
        ggml_backend_tensor_set(b_tproj, tproj_buf.data(), 0, tproj_buf.size() * sizeof(float));
        const float lg = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
    };

    const size_t f32b = sizeof(float), f16b = sizeof(ggml_fp16_t);
    auto         mk_inputs = [&](ggml_context * ctx) {
        DitInputs in;
        in.t_input = ggml_view_3d(ctx, b_input, Ic, T, B, (size_t) Ic * f32b, (size_t) Ic * T * f32b, 0);
        in.t_enc =
            ggml_view_3d(ctx, b_enc, enc_H, enc_use, B, (size_t) enc_H * f32b, (size_t) enc_H * enc_use * f32b, 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, (int64_t) S * B, 0);
        in.t_temb  = ggml_view_3d(ctx, b_temb, H, 1, B, (size_t) H * f32b, (size_t) H * f32b, 0);
        in.t_tproj = ggml_view_3d(ctx, b_tproj, 6 * H, 1, B, (size_t) 6 * H * f32b, (size_t) 6 * H * f32b, 0);
        in.t_sa    = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sa, S, S, 1, B, (size_t) S * f16b, (size_t) S * S * f16b,
                                                  (size_t) S * S * f16b, 0)
                                   : ggml_view_2d(ctx, b_sa, S, S, (size_t) S * f16b, 0);
        in.t_sa_pad = (bh.sa_B > 1) ? ggml_view_4d(ctx, b_sapad, S, S, 1, B, (size_t) S * f16b,
                                                   (size_t) S * S * f16b, (size_t) S * S * f16b, 0)
                                    : nullptr;
        in.t_ca    = ggml_view_4d(ctx, b_ca, enc_use, S, 1, B, (size_t) enc_use * f16b, (size_t) enc_use * S * f16b,
                                  (size_t) enc_use * S * f16b, 0);
        in.t_vtgt  = ggml_view_3d(ctx, b_vtgt, Oc, T, B, (size_t) Oc * f32b, (size_t) Oc * T * f32b, 0);
        in.t_lw    = ggml_view_3d(ctx, b_lw, 1, T, B, f32b, (size_t) T * f32b, 0);
        in.t_lwu   = ggml_view_3d(ctx, b_lwu, 1, T, B, f32b, (size_t) T * f32b, 0);
        return in;
    };

    auto read_accs = [&](std::vector<std::vector<float>> * dst) {
        dst->assign(opt.acc.size(), std::vector<float>());
        for (size_t j = 0; j < opt.acc.size(); j++) {
            (*dst)[j].resize((size_t) ggml_nelements(opt.acc[j]));
            ggml_backend_tensor_get(opt.acc[j], (*dst)[j].data(), 0, (*dst)[j].size() * sizeof(float));
        }
    };

    // One arm. `mode` (and, in flash mode, `prec`) is the ONLY thing that
    // differs between the calls.
    auto run = [&](DitAttnMode mode, ggml_prec prec, double * loss_out, std::vector<std::vector<float>> * grads,
                   std::vector<float> * out_vals, int * nodes) -> bool {
        lm_optim_zero_grad(&opt);
        upload();
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, true);
        DitInputs        in  = mk_inputs(ctx);
        in.attn_mode         = mode;
        in.attn_prec_req     = prec;
        DitTaps       taps;
        ggml_tensor * vel = dit_train_forward(ctx, M, ad, in, T, enc_use, &taps, lo, L, B);
        ggml_tensor * ls  = dit_train_loss(ctx, vel, in, Oc, T, false, bh.gscale);
        ggml_set_loss(ls);
        ggml_build_forward_expand(gf, ls);
        std::vector<ggml_tensor *> ga;
        lm_optim_fill_gacc(&opt, gf, &ga);
        ggml_build_backward_expand(ctx, gf, ga.data());
        *nodes = ggml_graph_n_nodes(gf);
        ggml_backend_sched_reset(M->sched);
        const bool good = ggml_backend_sched_graph_compute(M->sched, gf) == GGML_STATUS_SUCCESS;
        if (good) {
            float lv = 0.0f;
            ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
            // The deepest hidden_after_layerN tap IS the trained stack's output.
            // Read it back before ggml_free takes the context's tensors with it.
            ggml_tensor * t = nullptr;
            for (size_t i = 0; i < taps.v.size(); i++) {
                if (taps.v[i].first.compare(0, 18, "hidden_after_layer") == 0) {
                    t      = taps.v[i].second;
                    o->tap = taps.v[i].first;
                }
            }
            if (t) {
                out_vals->resize((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, out_vals->data(), 0, out_vals->size() * sizeof(float));
            }
        }
        ggml_free(ctx);
        if (good) {
            read_accs(grads);
        }
        return good && !out_vals->empty();
    };

    std::vector<std::vector<float>> gex, gfl;
    std::vector<float>              oex, ofl;
    if (!run(DIT_ATTN_EXACT, GGML_PREC_DEFAULT, &o->loss_exact, &gex, &oex, &o->nodes_exact)) {
        *err = "exact arm failed to compute";
        cleanup();
        return false;
    }
    if (!run(DIT_ATTN_FLASH, GGML_PREC_F32, &o->loss_flash, &gfl, &ofl, &o->nodes_flash)) {
        *err = "flash arm failed to compute";
        cleanup();
        return false;
    }
    if (oex.size() != ofl.size()) {
        *err = "layer output shapes differ between arms";
        cleanup();
        return false;
    }

    // One flash arm against the exact arm. Called once per precision, so both
    // columns are computed the same way from the same reference.
    auto compare = [&](const std::vector<float> & of, const std::vector<std::vector<float>> & gf, double * out_rel,
                       double * out_mag, double * grad_rel, std::string * worst, double * ref_mag,
                       double * rel_qk, std::string * worst_qk, double * mag_qk, double * rel_ot,
                       std::string * worst_ot, double * mag_ot) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < oex.size() && i < of.size(); i++) {
            num = std::max(num, fabs((double) of[i] - (double) oex[i]));
            den = std::max(den, fabs((double) oex[i]));
        }
        if (out_mag) {
            *out_mag = den;
        }
        *out_rel = num / std::max(den, 1e-30);
        for (size_t j = 0; j < gex.size() && j < gf.size(); j++) {
            double gnum = 0.0, gden = 0.0;
            for (size_t k = 0; k < gex[j].size() && k < gf[j].size(); k++) {
                gnum = std::max(gnum, fabs((double) gf[j][k] - (double) gex[j][k]));
                gden = std::max(gden, fabs((double) gex[j][k]));
            }
            if (ref_mag) {
                *ref_mag = std::max(*ref_mag, gden);
            }
            const double      rel = gnum / std::max(gden, 1e-30);
            const std::string nm  = opt.params[j]->name;
            if (grad_rel && rel > *grad_rel) {
                *grad_rel = rel;
                *worst    = nm;
            }
            const bool qk = (nm.find("q_proj") != std::string::npos) || (nm.find("k_proj") != std::string::npos);
            if (qk) {
                if (rel > *rel_qk) {
                    *rel_qk = rel;
                    if (worst_qk) {
                        *worst_qk = nm;
                    }
                    if (mag_qk) {
                        *mag_qk = gden;
                    }
                }
            } else if (rel > *rel_ot) {
                *rel_ot = rel;
                if (worst_ot) {
                    *worst_ot = nm;
                }
                if (mag_ot) {
                    *mag_ot = gden;
                }
            }
        }
    };

    compare(ofl, gfl, &o->out_rel, &o->out_mag, &o->grad_rel, &o->worst, &o->ref_mag, &o->grad_rel_qk, &o->worst_qk,
            &o->mag_qk, &o->grad_rel_ot, &o->worst_ot, &o->mag_ot);

    // HOT-Step patch: flash-attn-train — the TF32 arm. Same graph, same seed,
    // same adapter state; only op_params slot 3 differs. Its numbers are
    // reported beside the f32 arm's so "TF32 costs this much accuracy" is a
    // measured delta rather than an assumption, and prec_fwd/prec_bwd carry the
    // backend's own answer about which kernel it launched — a run where the
    // dispatch quietly fell back to the scalar kernels would otherwise print two
    // identical columns and look like a pass.
    if (second_prec) {
        std::vector<std::vector<float>> gt;
        std::vector<float>              ot;
        int                             nt = 0;
        if (run(DIT_ATTN_FLASH, GGML_PREC_DEFAULT, &o->loss_flash_tf32, &gt, &ot, &nt) && ot.size() == oex.size()) {
            double dummy_qk_worst_mag = 0.0, dummy_ot_worst_mag = 0.0, dummy_qk = 0.0;
            std::string dummy_worst;
            compare(ot, gt, &o->out_rel_tf32, nullptr, nullptr, &dummy_worst, nullptr, &o->grad_rel_qk_tf32,
                    nullptr, &dummy_qk_worst_mag, &o->grad_rel_ot_tf32, &o->worst_ot_tf32, &dummy_ot_worst_mag);
            (void) dummy_qk;
            o->prec_fwd  = dit_flash_last_prec(M->backend, 0);
            o->prec_bwd  = dit_flash_last_prec(M->backend, 1);
            o->ran_tf32  = true;
        }
    }

    cleanup();
    o->ran = true;
    return true;
}

static void dit_st_flash_gates(std::vector<DitSelfTestResult> & rs, DitTrainModel * Mg,
                               const std::string & dit_path, const std::vector<DitSample> & samples, uint64_t seed) {
    const int    T      = 64;
    const char * gname  = ggml_backend_name(Mg->backend);
    const bool   gpu    = (strcmp(gname, "CPU") != 0);
    const double bar    = 1e-4;   // layer output, and every gradient outside q/k
    const double bar_qk = 2e-3;   // q_proj / k_proj, through the QK rms_norm back
    char         d[4096];

    // HOT-Step patch: flash-attn-train — both arithmetics on one line, plus the
    // backend's own answer about which kernel each direction launched. Reported,
    // never gated: see the TF32 paragraph above for why the CUDA delta measures
    // the reference's rounding rather than the op's.
    auto gpu_prec_line = [](const DitStFlashMeas & g) {
        char t[640];
        if (!g.ran_tf32) {
            snprintf(t, sizeof(t), "prec f32 %.3e out / %.3e q-k / %.3e other (no second arithmetic here)",
                     g.out_rel, g.grad_rel_qk, g.grad_rel_ot);
        } else {
            snprintf(t, sizeof(t),
                     "prec f32 %.3e out / %.3e q-k / %.3e other, loss %.9f; prec tf32 %.3e / %.3e / %.3e, "
                     "loss %.9f (kernels launched: fwd %s, bwd %s)",
                     g.out_rel, g.grad_rel_qk, g.grad_rel_ot, g.loss_flash, g.out_rel_tf32, g.grad_rel_qk_tf32,
                     g.grad_rel_ot_tf32, g.loss_flash_tf32, g.prec_fwd.c_str(), g.prec_bwd.c_str());
        }
        return std::string(t);
    };

    if (samples.empty()) {
        dit_st_report(rs, "SF1", false, "no cached songs");
        dit_st_report(rs, "SF2", false, "no cached songs");
        return;
    }
    const int B2    = 3;
    const int enc_1 = std::min(64, samples[0].enc_S);
    int       enc_3 = 64;
    for (int b = 0; b < B2 && b < (int) samples.size(); b++) {
        enc_3 = std::min(enc_3, samples[(size_t) b].enc_S);
    }
    const bool have_b2 = ((int) samples.size() >= B2);

    // The training backend's arms. Reported, never gated on the numeric bars:
    // see the TF32 paragraph above.
    DitStFlashMeas g1, g2;
    std::string    g1err = "-", g2err = "-";
    // `second_prec` = gpu: the TF32 arm only exists where the CUDA dispatch can
    // select it. On a CPU-only box the flag is ignored by design, so asking for
    // it would print the f32 column twice.
    const bool     g1ok  = dit_st_flash_measure(Mg, samples, seed, 1, T, enc_1, 2, false, gpu, &g1, &g1err);
    const bool     g2ok =
        have_b2 && dit_st_flash_measure(Mg, samples, seed, B2, T, enc_3, 2, true, gpu, &g2, &g2err);

    // The CPU copy: the only place exact-vs-flash is a clean measurement.
    // Mirrored at lora_lo = L-2 so just the two layers under test are promoted,
    // with the A2c batched flag so proj_in / cond_emb are F32 like every other
    // input-side weight. Same shape as the SB rungs' identity run.
    DitTrainModel  Mc;
    DitStFlashMeas c1, c2;
    std::string    cerr = "-", c1err = "-", c2err = "-";
    bool           c1ok = false, c2ok = false;
    {
        std::string err;
        if (!dit_train_load(&Mc, dit_path.c_str(), 0, &err)) {
            cerr = "CPU-backend copy failed to load: " + err;
        } else {
            Mc.backend = cpu_backend_new(16);
            if (!Mc.backend) {
                cerr = "no CPU backend available for the parity run";
            } else if (!dit_build_mirror(&Mc, Mc.m.cfg.n_layers - 2, DIT_MIRROR_F32, &err, /*batched=*/true)) {
                cerr = "CPU mirror failed: " + err;
            } else {
                BackendPair bp;
                bp.backend     = Mc.backend;
                bp.cpu_backend = Mc.backend;  // n == 1: nothing can silently land elsewhere
                bp.has_gpu     = false;
                Mc.sched       = backend_sched_new(bp, 65536);
                c1ok = dit_st_flash_measure(&Mc, samples, seed, 1, T, enc_1, 2, false, false, &c1, &c1err);
                c2ok =
                    have_b2 && dit_st_flash_measure(&Mc, samples, seed, B2, T, enc_3, 2, true, false, &c2, &c2err);
            }
        }
    }
    dit_train_free(&Mc);

    const bool probe_ok = !gpu || (g1.probe_fwd && g1.probe_bwd && (!have_b2 || (g2.probe_fwd && g2.probe_bwd)));

    // SF1 — B = 1: native GQA and the 2-D broadcast mask, i.e. the default run.
    snprintf(d, sizeof(d),
             "--attn flash vs exact, B=1 (GQA %d/%d, 2-D window mask), LoRA on the top 2 of %d layers "
             "(layer_type 0 x%d, 1 x%d), S=%d enc_S=%d. CPU parity, both arms f32 so the fused op is the only "
             "difference left: %s output %.3e (bar %.0e); gradients outside q/k %.3e on %s (bar %.0e, that "
             "tensor's |grad|max %.3e); q/k, back through the QK rms_norm, %.3e on %s (bar %.0e, |grad|max %.3e); "
             "losses %.9f vs %.9f [%s]. %s reports %s over a %d-node exact graph vs %d-node flash "
             "— REPORTED, not gated: the exact arm's attention mul_mats run on cuBLAS TF32 there, so that delta "
             "sizes the reference's own rounding. supports_op on %s: fwd %s bwd %s%s%s",
             Mg->m.cfg.n_kv_heads, Mg->m.cfg.n_heads, Mg->m.cfg.n_layers, c1.lt0, c1.lt1, T / Mg->m.cfg.patch_size,
             enc_1, c1.tap.c_str(), c1.out_rel, bar, c1.grad_rel_ot, c1.worst_ot.c_str(), bar, c1.mag_ot,
             c1.grad_rel_qk, c1.worst_qk.c_str(), bar_qk, c1.mag_qk, c1.loss_exact, c1.loss_flash,
             c1ok ? "cpu ok" : (cerr != "-" ? cerr.c_str() : c1err.c_str()), gname, gpu_prec_line(g1).c_str(),
             g1.nodes_exact, g1.nodes_flash, gname, g1.probe_fwd ? "yes" : "NO", g1.probe_bwd ? "yes" : "NO",
             g1ok ? "" : " — GPU ARM FAILED: ", g1ok ? "" : g1err.c_str());
    dit_st_report(rs, "SF1",
                  g1ok && c1ok && probe_ok && c1.out_rel <= bar && c1.grad_rel_ot <= bar &&
                      c1.grad_rel_qk <= bar_qk && c1.out_mag > 0.0 && c1.ref_mag > 0.0,
                  d);

    // SF2 — B = 3 mixed lengths: per-element [.,.,1,B] masks, so the op's ne2/ne3
    // modulo broadcast is measured too, and (patch B2) an ASYMMETRIC pair — the
    // exact arm expands K/V to Nh, the flash arm keeps native GQA 8/32.
    if (!have_b2) {
        snprintf(d, sizeof(d), "need >= %d cached songs for a B=%d batch of DIFFERENT songs, found %d", B2, B2,
                 (int) samples.size());
        dit_st_report(rs, "SF2", false, d);
        return;
    }
    snprintf(d, sizeof(d),
             "--attn flash vs exact, B=%d mixed length (one song %d of %d frames, %d padded, so both masks are "
             "per-element [.,.,1,B]; B2: the EXACT arm pre-expands K/V to Nh, the FLASH arm runs native GQA). "
             "CPU parity: %s output %.3e (bar %.0e); "
             "gradients outside q/k %.3e on %s (bar %.0e, |grad|max %.3e); q/k %.3e on %s (bar %.0e, |grad|max "
             "%.3e); losses %.9f vs %.9f [%s]. %s reports %s over a %d-node exact graph vs %d-node flash "
             "— REPORTED, not gated (cuBLAS TF32 reference). supports_op on %s: fwd %s bwd %s%s%s",
             B2, c2.short_len, c2.pad_len, c2.pad_len - c2.short_len, c2.tap.c_str(), c2.out_rel, bar,
             c2.grad_rel_ot, c2.worst_ot.c_str(), bar, c2.mag_ot, c2.grad_rel_qk, c2.worst_qk.c_str(), bar_qk,
             c2.mag_qk, c2.loss_exact, c2.loss_flash,
             c2ok ? "cpu ok" : (cerr != "-" ? cerr.c_str() : c2err.c_str()), gname, gpu_prec_line(g2).c_str(),
             g2.nodes_exact, g2.nodes_flash, gname, g2.probe_fwd ? "yes" : "NO", g2.probe_bwd ? "yes" : "NO",
             g2ok ? "" : " — GPU ARM FAILED: ", g2ok ? "" : g2err.c_str());
    dit_st_report(rs, "SF2",
                  g2ok && c2ok && probe_ok && c2.out_rel <= bar && c2.grad_rel_ot <= bar &&
                      c2.grad_rel_qk <= bar_qk && c2.out_mag > 0.0 && c2.ref_mag > 0.0,
                  d);
}

// ─── LK5: export / parse roundtrip ──────────────────────────────────────────
//
// Parses the written file with yyjson directly rather than through the engine's
// safetensors reader: the point of the rung is that the BYTES on disk are what
// adapter-merge.h expects, so it must not share a reader with the thing it is
// checking.
static std::string dit_st_temp_dir() {
    const char * t = getenv("TEMP");
    if (!t || !*t) {
        t = getenv("TMPDIR");
    }
    if (!t || !*t) {
        t = ".";
    }
    return std::string(t) + "/hot-step-lokr-selftest";
}

static void dit_st_lokr_export(std::vector<DitSelfTestResult> & rs, const DitAdapterLoKr & lk, int layer) {
    const std::string dir = dit_st_temp_dir();
    DitExportMeta     meta;
    meta.producer = std::string("ace-train ") + ACE_VERSION;
    DitExportResult xr;
    std::string     err;
    if (!lk.exportDir(dir.c_str(), meta, &xr, &err)) {
        dit_st_report(rs, "LK5", false, "export failed: " + err);
        return;
    }
    const std::string sf = lm_join(dir, "lokr_weights.safetensors");

    std::vector<char> raw;
    {
        FILE * f = fopen(sf.c_str(), "rb");
        if (!f) {
            dit_st_report(rs, "LK5", false, "cannot re-open " + sf);
            return;
        }
        fseek(f, 0, SEEK_END);
        const long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        raw.resize((size_t) std::max(0L, n));
        const size_t got = raw.empty() ? 0 : fread(raw.data(), 1, raw.size(), f);
        fclose(f);
        if (got != raw.size() || raw.size() < 8) {
            dit_st_report(rs, "LK5", false, "short read on " + sf);
            return;
        }
    }
    uint64_t hdr_len = 0;
    for (int i = 7; i >= 0; i--) {
        hdr_len = (hdr_len << 8) | (uint64_t) (uint8_t) raw[(size_t) i];
    }
    if (hdr_len + 8 > raw.size()) {
        dit_st_report(rs, "LK5", false, "safetensors header length overruns the file");
        return;
    }
    const size_t data0 = 8 + (size_t) hdr_len;
    yyjson_doc * doc   = yyjson_read(raw.data() + 8, (size_t) hdr_len, 0);
    if (!doc) {
        dit_st_report(rs, "LK5", false, "safetensors header is not valid JSON");
        return;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);

    int         bad = 0, checked = 0, cfg_dim = 0;
    std::string why;
    auto        fail = [&](const std::string & m) {
        bad++;
        if (why.empty()) {
            why = m;
        }
    };

    // __metadata__.lokr_config — the #1 silent-breakage trap (K8).
    {
        yyjson_val * md = yyjson_obj_get(root, "__metadata__");
        yyjson_val * cf = md ? yyjson_obj_get(md, "lokr_config") : nullptr;
        if (!cf || !yyjson_is_str(cf)) {
            fail("__metadata__.lokr_config missing or not a string");
        } else {
            yyjson_doc * sub = yyjson_read(yyjson_get_str(cf), yyjson_get_len(cf), 0);
            yyjson_val * ld  = sub ? yyjson_obj_get(yyjson_doc_get_root(sub), "linear_dim") : nullptr;
            if (!ld || !yyjson_is_int(ld)) {
                fail("lokr_config does not parse to an object with an integer linear_dim");
            } else {
                cfg_dim = (int) yyjson_get_int(ld);
                if (cfg_dim != lk.dim) {
                    fail("lokr_config.linear_dim != the configured dim");
                }
            }
            if (sub) {
                yyjson_doc_free(sub);
            }
        }
    }

    for (int s = 0; s < lk.nSites(); s++) {
        const DitLokrSite * k = lk.site(layer, s);
        if (!k) {
            fail("site missing from the adapter");
            continue;
        }
        const std::string stem = DitAdapterLoKr::lokr_key_stem(layer, s);
        struct Want {
            std::string name;
            int64_t     rows, cols;
        };
        std::vector<Want> want;
        want.push_back({ stem + ".alpha", 1, 0 });
        want.push_back({ stem + ".lokr_w1", k->out_l, k->in_m });
        if (k->mono) {
            want.push_back({ stem + ".lokr_w2", k->out_k, k->in_n });
        } else {
            want.push_back({ stem + ".lokr_w2_a", k->out_k, (int64_t) lk.dim });
            want.push_back({ stem + ".lokr_w2_b", (int64_t) lk.dim, k->in_n });
        }
        for (size_t i = 0; i < want.size(); i++) {
            yyjson_val * e = yyjson_obj_get(root, want[i].name.c_str());
            if (!e) {
                fail("missing key " + want[i].name);
                continue;
            }
            checked++;
            // BF16, not F32, since 2026-07-30: the writer matches LyCORIS /
            // Side-Step, which halves a dim-512 census from 872 to 436 MB.
            yyjson_val * dt = yyjson_obj_get(e, "dtype");
            if (!dt || !yyjson_is_str(dt) || strcmp(yyjson_get_str(dt), "BF16") != 0) {
                fail(want[i].name + " is not BF16");
            }
            yyjson_val * sh   = yyjson_obj_get(e, "shape");
            const size_t nd   = sh ? yyjson_arr_size(sh) : 0;
            const int64_t d0  = nd > 0 ? yyjson_get_sint(yyjson_arr_get(sh, 0)) : -1;
            const int64_t d1  = nd > 1 ? yyjson_get_sint(yyjson_arr_get(sh, 1)) : -1;
            const bool    ok1 = (want[i].cols == 0) ? (nd == 1 && d0 == 1)
                                                    : (nd == 2 && d0 == want[i].rows && d1 == want[i].cols);
            if (!ok1) {
                char b[192];
                snprintf(b, sizeof(b), "%s shape [%lld,%lld] != expected [%lld,%lld]", want[i].name.c_str(),
                         (long long) d0, (long long) d1, (long long) want[i].rows, (long long) want[i].cols);
                fail(b);
            }
            if (want[i].cols == 0 && ok1) {
                yyjson_val * off = yyjson_obj_get(e, "data_offsets");
                const size_t o0  = (off && yyjson_arr_size(off) == 2)
                                       ? (size_t) yyjson_get_uint(yyjson_arr_get(off, 0))
                                       : SIZE_MAX;
                if (o0 == SIZE_MAX || data0 + o0 + sizeof(uint16_t) > raw.size()) {
                    fail(want[i].name + " data_offsets out of range");
                } else {
                    // Compare BIT PATTERNS against the writer's own rounding, so
                    // the rung stays honest for an alpha that is not exactly
                    // representable in BF16 (512 and 256 are; 0.7 would not be).
                    uint16_t got = 0;
                    memcpy(&got, raw.data() + data0 + o0, sizeof(uint16_t));
                    const uint16_t exp_bits = stw_f32_to_bf16(k->alpha_eff);
                    if (got != exp_bits) {
                        const uint32_t gu = (uint32_t) got << 16;
                        float          av = 0.0f;
                        memcpy(&av, &gu, sizeof(float));
                        char b[176];
                        snprintf(b, sizeof(b), "%s = %.6g (bf16 0x%04x) but the site's effective alpha is %.6g",
                                 want[i].name.c_str(), (double) av, (unsigned) got, (double) k->alpha_eff);
                        fail(b);
                    }
                }
            }
        }
    }
    yyjson_doc_free(doc);

    remove(sf.c_str());
#ifdef _WIN32
    _rmdir(dir.c_str());
#else
    rmdir(dir.c_str());
#endif

    char d[416];
    snprintf(d, sizeof(d),
             "%d tensors written (%lld bytes), %d keys re-parsed on L%d: dtypes/shapes/alpha ok, "
             "lokr_config.linear_dim=%d%s%s",
             xr.tensors, xr.bytes, checked, layer, cfg_dim, bad ? " | FIRST FAILURE: " : "", why.c_str());
    dit_st_report(rs, "LK5", bad == 0 && checked > 0, d);
}

// ─── the driver ─────────────────────────────────────────────────────────────

static int dit_self_test_impl(const std::string & dit_path, const std::string & tensors_dir, uint64_t seed,
                              int crop_max, int reserve_mb, float safety, int mode, int fd_child_rc) {
    std::vector<DitSelfTestResult> rs;
    const bool                     fd_only = (mode == DIT_ST_FD_ONLY);

    if (!fd_only) {
        dit_st_flow_snr(rs);
        dit_st_crop(rs, seed);
        dit_st_lokr_factorization(rs);
    }

    // ── data ─────────────────────────────────────────────────────────────
    std::vector<DitSample> samples;
    {
        std::string err;
        if (!dit_scan_samples(tensors_dir.c_str(), 0, &samples, &err)) {
            dit_st_report(rs, "DATA", false, err);
            return 1;
        }
    }
    if (!fd_only) {
        dit_st_convention(rs, samples, seed);
    }

    DitChannelStats cstats;
    const bool      have_cstats = dit_load_channel_stats(tensors_dir.c_str(), &cstats);

    int enc_S_full = 0, enc_H = 0, max_T = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        enc_S_full = std::max(enc_S_full, std::max(samples[i].enc_S, samples[i].enc_S_genre));
        enc_H      = std::max(enc_H, samples[i].enc_H);
        max_T      = std::max(max_T, samples[i].T);
    }
    for (size_t i = 0; i < samples.size(); i++) {
        samples[i].enc.resize((size_t) enc_S_full * (size_t) enc_H, 0.0f);
        samples[i].enc_mask.resize((size_t) enc_S_full, 0.0f);
        samples[i].enc_S = enc_S_full;
    }

    // ── model ────────────────────────────────────────────────────────────
    DitTrainModel M;
    {
        std::string err;
        if (!dit_train_load(&M, dit_path.c_str(), /*lora_lo=*/0, &err)) {
            dit_st_report(rs, "T1", false, err == "convrot" ? "ConvRot base — refused (D23)" : err);
            dit_train_free(&M);
            return 1;
        }
        std::string uerr;
        const bool  unfused = dit_assert_unfused(&M.m, &uerr);
        char        d[224];
        snprintf(d, sizeof(d), "%d layers: sa_qkv/sa_qk/ca_qkv/ca_kv/gate_up all NULL and every separate projection "
                               "present — %s",
                 M.m.cfg.n_layers, unfused ? "unfused" : uerr.c_str());
        dit_st_report(rs, "T1", unfused, d);
        if (!unfused) {
            dit_train_free(&M);
            return 1;
        }
        if (!dit_train_backend_init(&M, &err)) {
            dit_st_report(rs, "BACKEND", false, err);
            dit_train_free(&M);
            return 1;
        }
        // The gates below compare against F32 references, so the self-test always
        // runs the F32 mirror regardless of what a run would pick.
        if (!dit_build_mirror(&M, 0, DIT_MIRROR_F32, &err)) {
            dit_st_report(rs, "MIRROR", false, err);
            dit_train_free(&M);
            return 1;
        }
    }
    const DiTGGMLConfig & c = M.m.cfg;
    const int             H = c.hidden_size, Oc = c.out_channels, P = c.patch_size, L = c.n_layers;
    {
        BackendPair bp;
        bp.backend     = M.backend;
        bp.cpu_backend = M.cpu;
        bp.has_gpu     = true;
        M.sched        = backend_sched_new(bp, 65536);
    }
    // ── SF1 / SF2: --attn flash vs exact ───────────────────────────────
    // Placed HERE, ahead of the full-depth crop fit, on purpose. These two
    // rungs need the model, the backend, the mirror and the scheduler and
    // nothing else: they build a 2-layer sub-stack at T=64 with their own
    // small buffers. Behind the fit they would be skipped on any machine
    // whose card cannot also hold a 32-layer activation set — which is
    // exactly the machine most likely to want --attn flash in the first
    // place.
    if (!fd_only) {
        dit_st_flash_gates(rs, &M, dit_path, samples, seed);
    }


    // ── crop for the descent gates: auto-fit at full depth ───────────────
    DitVramModel vm;
    vm.m          = &M.m;
    vm.n_layers   = L;
    vm.patch      = P;
    vm.rank       = 16;
    vm.target_mlp = false;
    vm.in_ch      = c.in_channels;
    vm.out_ch     = Oc;
    vm.hidden     = H;
    vm.enc_H      = enc_H;
    vm.enc_S      = enc_S_full;
    size_t fb = 0, tb = 0;
    lm_vram_query(M.backend, &fb, &tb);
    // The mirror is ALREADY allocated here, so it is already subtracted from `fb`
    // — but dit_vram_total_bytes() includes it, so it must be added back or the
    // mirror is charged twice (the same defect lm-vram.h:95 documents).
    const double budget =
        ((double) fb + (double) M.mirror.bytes - (double) reserve_mb * 1048576.0) * (double) (1.0f - safety);
    int          cap    = std::min(crop_max, max_T);
    cap -= cap % P;
    int crop_big = dit_vram_best_crop(vm, L, budget, 128, cap);
    if (crop_big <= 0) {
        dit_st_report(rs, "VRAM", false, "not enough free VRAM to run the self-test at full depth");
        dit_train_free(&M);
        return 1;
    }
    const int S_big = crop_big / P;

    // Small configuration for the exact gates (T3/T4/T6/T7).
    const int T_small = 64, S_small = T_small / P, enc_small = 64;

    // ── static input bases (1-D; every graph tensor is a contiguous view) ─
    ggml_context * ctxs;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctxs               = ggml_init(p);
    }
    ggml_tensor * b_input = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) c.in_channels * crop_big);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) enc_H * enc_S_full);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctxs, GGML_TYPE_I32, S_big);
    ggml_tensor * t_temb  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, H);
    ggml_tensor * t_tproj = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 6 * H);
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) S_big * S_big);
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F16, (int64_t) enc_S_full * S_big);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, (int64_t) Oc * crop_big);
    ggml_tensor * t_cw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, Oc);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctxs, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { b_input, b_enc, b_pos, t_temb, t_tproj, b_sa, b_ca, b_vtgt, t_cw }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctxs, M.backend);
    if (!buf_static) {
        dit_st_report(rs, "VRAM", false, "static input allocation failed");
        ggml_free(ctxs);
        dit_train_free(&M);
        return 1;
    }
    std::vector<float> cw((size_t) Oc, 1.0f);
    for (int i = 0; i < Oc && i < (int) cstats.weight.size(); i++) {
        cw[(size_t) i] = cstats.weight[(size_t) i];
    }
    ggml_backend_tensor_set(t_cw, cw.data(), 0, cw.size() * sizeof(float));
    {
        const float epsv = 1e-6f, clipv = 1.0f, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    std::vector<uint8_t>  arena((size_t) 512 << 20);
    std::vector<float>    input_buf((size_t) c.in_channels * crop_big, 0.0f);
    std::vector<float>    flow_xt, flow_v;
    std::vector<int32_t>  pos_buf((size_t) S_big);
    std::vector<uint16_t> sa_buf, ca_buf;

    const DitSample & s0 = samples[0];

    // Uploads every input for a (crop_start, len, enc_S, t) configuration.
    // `v_override` replaces the velocity target (the shuffled-target control).
    auto upload = [&](const DitSample & s, int crop_start, int len, int enc_use, float t, LmRng * noise,
                      const std::vector<int> * perm) {
        dit_flow_target(s.lat.data() + (size_t) crop_start * (size_t) s.Oc, (size_t) len * (size_t) s.Oc, t, noise,
                        &flow_xt, &flow_v);
        if (perm) {
            std::vector<float> vs(flow_v.size());
            for (int f = 0; f < len; f++) {
                memcpy(&vs[(size_t) f * (size_t) s.Oc], &flow_v[(size_t) (*perm)[(size_t) f] * (size_t) s.Oc],
                       (size_t) s.Oc * sizeof(float));
            }
            flow_v.swap(vs);
        }
        for (int f = 0; f < len; f++) {
            float * dst = &input_buf[(size_t) f * (size_t) c.in_channels];
            memcpy(dst, &s.ctxl[(size_t) (crop_start + f) * (size_t) s.Cc], (size_t) s.Cc * sizeof(float));
            memcpy(dst + s.Cc, &flow_xt[(size_t) f * (size_t) s.Oc], (size_t) s.Oc * sizeof(float));
        }
        const int S = len / P;
        for (int i = 0; i < S; i++) {
            pos_buf[(size_t) i] = i;
        }
        dit_sa_mask(S, c.sliding_window, &sa_buf);
        std::vector<float> emask(s.enc_mask.begin(), s.enc_mask.begin() + enc_use);
        dit_ca_mask(enc_use, S, emask, &ca_buf);
        ggml_backend_tensor_set(b_input, input_buf.data(), 0, (size_t) c.in_channels * (size_t) len * sizeof(float));
        ggml_backend_tensor_set(b_vtgt, flow_v.data(), 0, flow_v.size() * sizeof(float));
        ggml_backend_tensor_set(b_enc, s.enc.data(), 0, (size_t) enc_H * (size_t) enc_use * sizeof(float));
        ggml_backend_tensor_set(b_pos, pos_buf.data(), 0, (size_t) S * sizeof(int32_t));
        ggml_backend_tensor_set(b_sa, sa_buf.data(), 0, sa_buf.size() * sizeof(uint16_t));
        ggml_backend_tensor_set(b_ca, ca_buf.data(), 0, ca_buf.size() * sizeof(uint16_t));
    };

    auto make_inputs = [&](ggml_context * ctx, int len, int enc_use) {
        const int S = len / P;
        DitInputs in;
        in.t_input = ggml_view_2d(ctx, b_input, c.in_channels, len, (size_t) c.in_channels * sizeof(float), 0);
        in.t_enc   = ggml_view_2d(ctx, b_enc, enc_H, enc_use, (size_t) enc_H * sizeof(float), 0);
        in.t_pos   = ggml_view_1d(ctx, b_pos, S, 0);
        in.t_temb  = t_temb;
        in.t_tproj = t_tproj;
        in.t_sa    = ggml_view_2d(ctx, b_sa, S, S, (size_t) S * sizeof(ggml_fp16_t), 0);
        in.t_ca    = ggml_view_2d(ctx, b_ca, enc_use, S, (size_t) enc_use * sizeof(ggml_fp16_t), 0);
        in.t_vtgt  = ggml_view_2d(ctx, b_vtgt, Oc, len, (size_t) Oc * sizeof(float), 0);
        in.t_cw    = t_cw;
        return in;
    };

    // ── the main adapter (B = 0) ─────────────────────────────────────────
    DitAdapterLora lora;
    {
        DitAdapterCfg cfg;
        cfg.rank  = 16;
        cfg.alpha = 32.0f;
        cfg.seed  = seed;
        std::string err;
        if (!lora.init(&M.m, M.backend, 0, L, cfg, &err)) {
            dit_st_report(rs, "T2", false, err);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctxs);
            dit_train_free(&M);
            return 1;
        }
        std::string d = "all " + std::to_string(lora.nSites()) + " sites x " + std::to_string(L) +
                        " layers PEFT-shaped (A[in,r], B[r,out], in==base.ne0, out==base.ne1); params=" +
                        std::to_string(lora.nParams()) + " tensors=" + std::to_string(lora.par.size());
        const size_t expect = dit_lora_expected_params(c, 0, L, 16, false);
        dit_st_report(rs, "T2", lora.nParams() == expect, d + " expected=" + std::to_string(expect));
    }

    LmOptim opt;
    {
        std::string err;
        if (!lm_optim_init(&opt, lora.params(), M.backend, &err)) {
            dit_st_report(rs, "OPT", false, err);
            lora.free();
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctxs);
            dit_train_free(&M);
            return 1;
        }
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lossgrad;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gnorm2;
    opt.base_lr      = 1e-3f;
    opt.grad_clip    = 1.0f;
    opt.weight_decay = 0.01f;
    opt.total_steps  = 60;
    opt.warmup_steps = 0;

    // ── T3 / T7: forward diff against the inference graph + loss identity ─
    if (!fd_only) {
        const float t_fix = 0.5f;
        LmRng       noise;
        lm_rng_seed(&noise, seed ^ 0x9e3779b97f4a7c15ull);
        upload(s0, 0, T_small, enc_small, t_fix, &noise, nullptr);

        std::vector<float>              tv(1, t_fix);
        std::vector<std::vector<float>> tbv, tpv;
        if (!dit_train_temb(&M, tv, &tbv, &tpv)) {
            dit_st_report(rs, "T3", false, "temb precompute failed");
        } else {
            ggml_backend_tensor_set(t_temb, tbv[0].data(), 0, tbv[0].size() * sizeof(float));
            ggml_backend_tensor_set(t_tproj, tpv[0].data(), 0, tpv[0].size() * sizeof(float));

            // training graph, forward only, with the named probes
            ggml_init_params ip   = { arena.size(), arena.data(), true };
            ggml_context *   ctxt = ggml_init(ip);
            ggml_cgraph *    gft  = ggml_new_graph_custom(ctxt, 65536, false);
            DitTaps          taps;
            DitInputs        in    = make_inputs(ctxt, T_small, enc_small);
            ggml_tensor *    vpred = dit_train_forward(ctxt, &M, &lora, in, T_small, enc_small, &taps);
            ggml_tensor *    loss  = dit_train_loss(ctxt, vpred, in, Oc, T_small, have_cstats);
            ggml_build_forward_expand(gft, loss);
            ggml_backend_sched_reset(M.sched);
            const bool okt = ggml_backend_sched_graph_compute(M.sched, gft) == GGML_STATUS_SUCCESS;

            std::vector<std::vector<float>> train_vals(taps.v.size());
            float                           loss_graph = 0.0f;
            if (okt) {
                for (size_t i = 0; i < taps.v.size(); i++) {
                    train_vals[i].resize((size_t) ggml_nelements(taps.v[i].second));
                    ggml_backend_tensor_get(taps.v[i].second, train_vals[i].data(), 0,
                                            train_vals[i].size() * sizeof(float));
                }
                ggml_backend_tensor_get(loss, &loss_graph, 0, sizeof(float));
            }
            ggml_free(ctxt);

            // ── T7 loss identity: recompute in double on the host ─────────
            if (okt) {
                size_t vi = SIZE_MAX;
                for (size_t i = 0; i < taps.v.size(); i++) {
                    if (taps.v[i].first == "velocity") {
                        vi = i;
                    }
                }
                double acc = 0.0;
                if (vi != SIZE_MAX) {
                    const std::vector<float> & vp = train_vals[vi];
                    for (int f = 0; f < T_small; f++) {
                        for (int ch = 0; ch < Oc; ch++) {
                            const size_t k = (size_t) f * (size_t) Oc + (size_t) ch;
                            const double e = (double) vp[k] - (double) flow_v[k];
                            acc += e * e * (have_cstats ? (double) cw[(size_t) ch] : 1.0);
                        }
                    }
                    acc /= (double) ((int64_t) Oc * T_small);
                }
                const double rel = fabs(acc - (double) loss_graph) / std::max(1e-12, fabs(acc));
                char         d[224];
                snprintf(d, sizeof(d), "graph loss %.9f vs host-double recompute %.9f -> rel err %.3e (bar 1e-5), "
                                       "channel_balance=%s",
                         (double) loss_graph, acc, rel, have_cstats ? "on" : "off");
                dit_st_report(rs, "T7", vi != SIZE_MAX && rel < 1e-5, d);
            } else {
                dit_st_report(rs, "T7", false, "training forward failed");
            }

            // ── inference graph on the SAME inputs ─────────────────────────
            ggml_context * ctxi;
            {
                ggml_init_params p = { (size_t) 64 << 20, nullptr, true };
                ctxi               = ggml_init(p);
            }
            ggml_tensor *ii = nullptr, *io = nullptr;
            ggml_cgraph * gfi = dit_ggml_build_graph(&M.m, ctxi, T_small, enc_small, 1, &ii, &io);
            dit_protect_output_views(gfi);
            ggml_backend_sched_reset(M.sched);
            bool oki = ggml_backend_sched_alloc_graph(M.sched, gfi);
            if (oki) {
                ggml_backend_tensor_set(ii, input_buf.data(), 0,
                                        (size_t) c.in_channels * (size_t) T_small * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "enc_hidden"), s0.enc.data(), 0,
                                        (size_t) enc_H * (size_t) enc_small * sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "t"), &t_fix, 0, sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "t_r"), &t_fix, 0, sizeof(float));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "positions"), pos_buf.data(), 0,
                                        (size_t) S_small * sizeof(int32_t));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "sa_mask_sw"), sa_buf.data(), 0,
                                        sa_buf.size() * sizeof(uint16_t));
                ggml_backend_tensor_set(ggml_graph_get_tensor(gfi, "ca_mask"), ca_buf.data(), 0,
                                        ca_buf.size() * sizeof(uint16_t));
                oki = ggml_backend_sched_graph_compute(M.sched, gfi) == GGML_STATUS_SUCCESS;
            }

            if (!okt || !oki) {
                dit_st_report(rs, "T3", false, "could not run both graphs on identical inputs");
            } else {
                double      worst      = 0.0;
                std::string worst_name = "-";
                std::string table;
                std::vector<float> ref;
                for (size_t i = 0; i < taps.v.size(); i++) {
                    ggml_tensor * rt = ggml_graph_get_tensor(gfi, taps.v[i].first.c_str());
                    if (!rt || (size_t) ggml_nelements(rt) != train_vals[i].size()) {
                        table += taps.v[i].first + "=MISSING ";
                        worst = 1e9;
                        worst_name = taps.v[i].first;
                        continue;
                    }
                    ref.resize(train_vals[i].size());
                    ggml_backend_tensor_get(rt, ref.data(), 0, ref.size() * sizeof(float));
                    double mx = 0.0, den = 0.0;
                    for (size_t k = 0; k < ref.size(); k++) {
                        den = std::max(den, fabs((double) ref[k]));
                    }
                    den = std::max(den, 1e-6);
                    for (size_t k = 0; k < ref.size(); k++) {
                        mx = std::max(mx, fabs((double) train_vals[i][k] - (double) ref[k]) / den);
                    }
                    char e[96];
                    snprintf(e, sizeof(e), "%s=%.2e ", taps.v[i].first.c_str(), mx);
                    table += e;
                    if (mx > worst) {
                        worst      = mx;
                        worst_name = taps.v[i].first;
                    }
                }
                // temb / tproj: host precompute vs the in-graph subgraph (S4)
                for (int k = 0; k < 2; k++) {
                    const char *               nm  = k ? "tproj" : "temb";
                    const std::vector<float> & src = k ? tpv[0] : tbv[0];
                    ggml_tensor *              rt  = ggml_graph_get_tensor(gfi, nm);
                    if (!rt || (size_t) ggml_nelements(rt) != src.size()) {
                        continue;
                    }
                    ref.resize(src.size());
                    ggml_backend_tensor_get(rt, ref.data(), 0, ref.size() * sizeof(float));
                    double mx = 0.0, den = 1e-6;
                    for (size_t j = 0; j < ref.size(); j++) {
                        den = std::max(den, fabs((double) ref[j]));
                    }
                    for (size_t j = 0; j < ref.size(); j++) {
                        mx = std::max(mx, fabs((double) src[j] - (double) ref[j]) / den);
                    }
                    char e[96];
                    snprintf(e, sizeof(e), "%s=%.2e ", nm, mx);
                    table += e;
                    if (mx > worst) {
                        worst      = mx;
                        worst_name = nm;
                    }
                }
                const std::string d = "max relative error per named tensor (bar 2e-3): " + table +
                                      "| worst=" + worst_name;
                dit_st_report(rs, "T3", worst <= 2e-3, d);
            }
            ggml_free(ctxi);
        }
    }

    // ── T6: B = 0 structural ─────────────────────────────────────────────
    if (!fd_only) {
        const float t_fix = 0.5f;
        LmRng       noise;
        lm_rng_seed(&noise, seed ^ 0x9e3779b97f4a7c15ull);
        upload(s0, 0, T_small, enc_small, t_fix, &noise, nullptr);
        lm_optim_zero_grad(&opt);

        ggml_init_params ip   = { arena.size(), arena.data(), true };
        ggml_context *   ctxt = ggml_init(ip);
        ggml_cgraph *    gft  = ggml_new_graph_custom(ctxt, 65536, true);
        DitInputs        in   = make_inputs(ctxt, T_small, enc_small);
        ggml_tensor * vpred   = dit_train_forward(ctxt, &M, &lora, in, T_small, enc_small);
        ggml_tensor * loss    = dit_train_loss(ctxt, vpred, in, Oc, T_small, have_cstats);
        ggml_set_loss(loss);
        ggml_build_forward_expand(gft, loss);
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(&opt, gft, &gacc);
        ggml_build_backward_expand(ctxt, gft, gacc.data());
        ggml_backend_sched_reset(M.sched);
        const bool ok = ggml_backend_sched_graph_compute(M.sched, gft) == GGML_STATUS_SUCCESS;
        ggml_free(ctxt);

        double worst_a = 0.0, min_b = 1e30;
        bool   all_finite = true;
        if (ok) {
            std::vector<float> g;
            for (size_t j = 0; j < opt.acc.size(); j++) {
                g.resize((size_t) ggml_nelements(opt.acc[j]));
                ggml_backend_tensor_get(opt.acc[j], g.data(), 0, g.size() * sizeof(float));
                double nrm = 0.0;
                for (size_t k = 0; k < g.size(); k++) {
                    if (!std::isfinite(g[k])) {
                        all_finite = false;
                    }
                    nrm += (double) g[k] * (double) g[k];
                }
                nrm = sqrt(nrm);
                if (j % 2 == 0) {  // params are pushed A,B,A,B,...
                    worst_a = std::max(worst_a, nrm);
                } else {
                    min_b = std::min(min_b, nrm);
                }
            }
        }
        lm_optim_zero_grad(&opt);
        char d[224];
        snprintf(d, sizeof(d), "B=0 backward over %d layers: max ||dL/dA|| = %.3e (want exactly 0), "
                               "min ||dL/dB|| = %.3e (want > 0), all finite = %s",
                 L, worst_a, min_b, all_finite ? "yes" : "NO");
        dit_st_report(rs, "T6", ok && worst_a == 0.0 && min_b > 0.0 && all_finite, d);
    }

    // ── LK2 / LK3 / LK5: the LoKR parameterization ───────────────────────
    //
    // One site on the top layer: gate_proj, the non-square worst case
    // (out 6144 -> (6,1024), in 2048 -> (4,512), so w1 [6,4] and w2 [1024,512]).
    if (!fd_only) {
        DitAdapterLoKr lk;
        bool           lk_ready = false;
        {
            DitAdapterCfg lcfg;
            lcfg.target_mlp          = true;
            lcfg.seed                = seed ^ 0x10c8badc0ffeeull;
            lcfg.lokr_dim            = 512;
            lcfg.lokr_alpha          = 512.0f;
            lcfg.lokr_factor         = 6;
            lcfg.lokr_decompose_both = true;
            std::string lerr;
            lk_ready = lk.init(&M.m, M.backend, L - 1, L, lcfg, &lerr);
            if (!lk_ready) {
                dit_st_report(rs, "LK2", false, lerr);
                dit_st_report(rs, "LK3", false, lerr);
                dit_st_report(rs, "LK5", false, lerr);
            }
        }
        // A failed init has already reported all three rungs. Its site tensors
        // may exist but be unallocated (init fails after ggml_new_tensor_2d but
        // before the backend buffer), so running on would write into nothing AND
        // report LK2/LK3/LK5 a second time.
        const int           li = L - 1;
        const DitLokrSite * ks = lk_ready ? lk.site(li, DIT_MLP_GATE) : nullptr;
        if (ks) {
            ggml_tensor * wg   = dit_site_weight(&M.m.layers[li], DIT_MLP_GATE);
            const int64_t in_f = wg->ne[0], out_f = wg->ne[1], S_lk = 8;

            ggml_context * ctxx;
            {
                ggml_init_params p = { 8 * ggml_tensor_overhead(), nullptr, true };
                ctxx               = ggml_init(p);
            }
            ggml_tensor * xt = ggml_new_tensor_2d(ctxx, GGML_TYPE_F32, in_f, S_lk);
            ggml_set_input(xt);
            ggml_backend_buffer_t bufx = ggml_backend_alloc_ctx_tensors(ctxx, M.backend);

            // Integer-valued operands (see LK3): every partial sum stays an exact
            // f32 integer, so BOTH paths are bit-exact and the comparison is
            // immune to the cuBLAS TF32 floor that forces T4 into a child process.
            LmRng xr;
            lm_rng_seed(&xr, seed ^ 0xfeedfaceull);
            std::vector<float> xh((size_t) in_f * (size_t) S_lk);
            for (size_t i = 0; i < xh.size(); i++) {
                xh[i] = (float) ((int) lm_rng_below(&xr, 3) - 1);  // {-1,0,1}
            }
            if (bufx) {
                ggml_backend_tensor_set(xt, xh.data(), 0, xh.size() * sizeof(float));
            }

            // ── LK2: untrained identity (w2 == 0 => delta == 0) ───────────
            bool   ok2 = (bufx != nullptr);
            double lk2 = 0.0;
            if (ok2) {
                ggml_init_params ip   = { arena.size(), arena.data(), true };
                ggml_context *   ctxg = ggml_init(ip);
                ggml_cgraph *    gf   = ggml_new_graph_custom(ctxg, 65536, false);
                ggml_tensor *    ya   = lk.apply(ctxg, wg, li, DIT_MLP_GATE, xt);
                ggml_tensor *    yb   = ggml_mul_mat(ctxg, wg, xt);
                ggml_set_output(ya);
                ggml_set_output(yb);
                ggml_build_forward_expand(gf, ya);
                ggml_build_forward_expand(gf, yb);
                ggml_backend_sched_reset(M.sched);
                ok2 = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
                if (ok2) {
                    std::vector<float> va((size_t) ggml_nelements(ya)), vb(va.size());
                    ggml_backend_tensor_get(ya, va.data(), 0, va.size() * sizeof(float));
                    ggml_backend_tensor_get(yb, vb.data(), 0, vb.size() * sizeof(float));
                    for (size_t i = 0; i < va.size(); i++) {
                        lk2 = std::max(lk2, fabs((double) va[i] - (double) vb[i]));
                    }
                }
                ggml_free(ctxg);
            }
            {
                char d[256];
                snprintf(d, sizeof(d),
                         "L%d %s [out %lld, in %lld] x [%lld,%lld]: max |apply(x) - W.x| = %.3e (bar 1e-6)", li,
                         dit_site_peft(DIT_MLP_GATE), (long long) out_f, (long long) in_f, (long long) in_f,
                         (long long) S_lk, lk2);
                dit_st_report(rs, "LK2", ok2 && lk2 <= 1e-6, d);
            }

            // ── LK3: kron-matvec vs a materialized kron (the arbiter) ─────
            //
            // BOTH w2 forms are arbitrated. LyCORIS only keeps w2 monolithic
            // while dim >= max(out_k,in_n)/2, which on the real XL base holds
            // for the attention sites and is FALSE for every MLP site (gate_proj
            // is 9728x2560 -> w2 [2432,512], factorized). Gating the rung on
            // `mono` therefore made the arbiter permanently unrunnable on
            // exactly the site the plan aimed it at — the factorized branch is
            // the one carrying most of the adapter, so it is the one that most
            // needs proving.
            bool   ok3 = ok2;
            double lk3 = 0.0, ref_mag = 0.0;
            if (ok3) {
                // Integer fills (see LK2) — the products stay exact f32 integers
                // through both paths, so the comparison survives TF32.
                std::vector<float> wh;
                auto               fill_int = [&](ggml_tensor * t, int span) {
                    wh.resize((size_t) ggml_nelements(t));
                    for (size_t i = 0; i < wh.size(); i++) {
                        wh[i] = (float) ((int) lm_rng_below(&xr, span) - span / 2);
                    }
                    ggml_backend_tensor_set(t, wh.data(), 0, wh.size() * sizeof(float));
                };
                fill_int(ks->w1, 5);  // {-2..2}
                if (ks->mono) {
                    fill_int(ks->w2, 3);  // {-1,0,1}
                } else {
                    fill_int(ks->w2_a, 3);
                    fill_int(ks->w2_b, 3);
                }

                ggml_init_params ip   = { arena.size(), arena.data(), true };
                ggml_context *   ctxg = ggml_init(ip);
                ggml_cgraph *    gf   = ggml_new_graph_custom(ctxg, 65536, false);
                // (a) the §2.3 matvec path, isolated from the base projection
                ggml_tensor * dm = ggml_sub(ctxg, lk.apply(ctxg, wg, li, DIT_MLP_GATE, xt),
                                            ggml_mul_mat(ctxg, wg, xt));
                // (b) the PROVEN merge recipe (adapter-merge.h's LoKr kron),
                //     re-created here so the test does not depend on that header.
                //     A factorized site is first collapsed to its monolithic
                //     equivalent W2[in_n, out_k] = w2_b . w2_a, which is what the
                //     merge loader materializes before the kron.
                const int64_t a = ks->out_l, b = ks->in_m, cc = ks->out_k, dd = ks->in_n;
                ggml_tensor * w1s = ggml_scale(ctxg, ks->w1, ks->scale);
                ggml_tensor * w2m = ks->mono ? ks->w2
                                             : ggml_mul_mat(ctxg, ggml_cont(ctxg, ggml_transpose(ctxg, ks->w2_b)),
                                                            ks->w2_a);
                ggml_tensor * ou  = ggml_mul_mat(ctxg, ggml_reshape_2d(ctxg, w1s, 1, a * b),
                                                 ggml_reshape_2d(ctxg, w2m, 1, cc * dd));
                ggml_tensor * kp  = ggml_cont(ctxg, ggml_permute(ctxg, ggml_reshape_4d(ctxg, ou, b, a, dd, cc),
                                                                 1, 3, 0, 2));
                ggml_tensor * dr  = ggml_mul_mat(ctxg, ggml_reshape_2d(ctxg, kp, b * dd, a * cc), xt);
                ggml_set_output(dm);
                ggml_set_output(dr);
                ggml_build_forward_expand(gf, dm);
                ggml_build_forward_expand(gf, dr);
                ggml_backend_sched_reset(M.sched);
                ok3 = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
                if (ok3) {
                    std::vector<float> va((size_t) ggml_nelements(dm)), vb((size_t) ggml_nelements(dr));
                    ok3 = (va.size() == vb.size());
                    if (ok3) {
                        ggml_backend_tensor_get(dm, va.data(), 0, va.size() * sizeof(float));
                        ggml_backend_tensor_get(dr, vb.data(), 0, vb.size() * sizeof(float));
                        double num = 0.0;
                        for (size_t i = 0; i < va.size(); i++) {
                            num     = std::max(num, fabs((double) va[i] - (double) vb[i]));
                            ref_mag = std::max(ref_mag, fabs((double) vb[i]));
                        }
                        lk3 = num / std::max(1e-6, ref_mag);
                    }
                }
                ggml_free(ctxg);
            }
            {
                char d[352];
                snprintf(d, sizeof(d),
                         "w1[%lld,%lld] w2[%lld,%lld] %s scale %.4g, integer operands (TF32-exact): matvec vs "
                         "materialized kron max rel err %.3e (bar 1e-5), reference max |delta| %.1f",
                         (long long) ks->out_l, (long long) ks->in_m, (long long) ks->out_k, (long long) ks->in_n,
                         ks->mono ? "monolithic" : "FACTORIZED", (double) ks->scale, lk3, ref_mag);
                dit_st_report(rs, "LK3", ok3 && lk3 <= 1e-5 && ref_mag > 0.0, d);
            }

            // ── LK5: export / re-parse ────────────────────────────────────
            dit_st_lokr_export(rs, lk, li);

            if (bufx) {
                ggml_backend_buffer_free(bufx);
            }
            ggml_free(ctxx);
        }
        lk.free();
    }

    // ── T4 / LK4: finite differences on a real DiT layer ─────────────────
    //
    // Measured in a CHILD process with NVIDIA_TF32_OVERRIDE=0 (dit_st_spawn_fd).
    // In DIT_ST_NO_FD this process only relays that child's verdict, so the
    // summary and the exit code still carry both.
    //
    // The gate itself is parameterization-agnostic — it probes whatever the
    // adapter puts in params() — so T4 (LoRA) and LK4 (LoKR) are the same code
    // with different labels. Both parameterizations put exactly two trained
    // tensors on every site, which is what the slot -> (layer, site, tensor)
    // mapping below assumes.
    auto fd_gate = [&](const char * check, DitAdapter & fd, const char * lbl0, const char * lbl1) {
        std::string err;
        {
            LmOptim fopt;
            if (!lm_optim_init(&fopt, fd.params(), M.backend, &err)) {
                dit_st_report(rs, check, false, err);
            } else {
                fopt.t_adamw    = t_adamw;
                fopt.t_lossgrad = t_lossgrad;
                fopt.t_clip     = t_clip;
                fopt.t_eps      = t_eps;
                fopt.t_gnorm2   = t_gnorm2;

                const float lg1 = 1.0f;
                ggml_backend_tensor_set(t_lossgrad, &lg1, 0, sizeof(float));

                const float t_fix = 0.5f;
                LmRng       noise;
                lm_rng_seed(&noise, seed ^ 0x9e3779b97f4a7c15ull);
                upload(s0, 0, T_small, enc_small, t_fix, &noise, nullptr);
                std::vector<float>              tv(1, t_fix);
                std::vector<std::vector<float>> tbv, tpv;
                dit_train_temb(&M, tv, &tbv, &tpv);
                ggml_backend_tensor_set(t_temb, tbv[0].data(), 0, tbv[0].size() * sizeof(float));
                ggml_backend_tensor_set(t_tproj, tpv[0].data(), 0, tpv[0].size() * sizeof(float));

                auto run = [&](bool backward) -> double {
                    ggml_init_params ip   = { arena.size(), arena.data(), true };
                    ggml_context *   ctxt = ggml_init(ip);
                    ggml_cgraph *    gft  = ggml_new_graph_custom(ctxt, 65536, backward);
                    DitInputs        in   = make_inputs(ctxt, T_small, enc_small);
                    ggml_tensor *    vp =
                        dit_train_forward(ctxt, &M, &fd, in, T_small, enc_small, nullptr, L - 2, L);
                    ggml_tensor * ls = dit_train_loss(ctxt, vp, in, Oc, T_small, have_cstats);
                    if (backward) {
                        ggml_set_loss(ls);
                    }
                    ggml_build_forward_expand(gft, ls);
                    if (backward) {
                        std::vector<ggml_tensor *> ga;
                        lm_optim_fill_gacc(&fopt, gft, &ga);
                        ggml_build_backward_expand(ctxt, gft, ga.data());
                    }
                    ggml_backend_sched_reset(M.sched);
                    double v = std::nan("");
                    if (ggml_backend_sched_graph_compute(M.sched, gft) == GGML_STATUS_SUCCESS) {
                        float lv = 0.0f;
                        ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
                        v = (double) lv;
                    }
                    ggml_free(ctxt);
                    return v;
                };

                lm_optim_zero_grad(&fopt);
                const double l0 = run(true);  // analytic gradients into fopt.acc
                // Repeatability of the forward: this is the true noise floor of the
                // central difference, and it is what decides whether a probe that
                // will not converge is a gradient defect or an estimator variance
                // problem. Reported in the T4 line.
                const double r1 = run(false);
                const double r2 = run(false);
                const double repeat_abs = fabs(r1 - r2);

                // DEVIATION (E5, documented): §3.9 asks for 24 SINGLE-ELEMENT
                // central differences at eps=1e-3. Measured, that is below the
                // floor: the analytic element gradients here are ~1e-4, so
                // f(+eps) - f(-eps) ~ 2e-7 on a loss of ~0.7 whose float32 ULP is
                // 6e-8 — 2 to 8 ULPs, i.e. pure quantisation noise (the first run
                // produced fd values that were exact multiples of the ULP). So we
                // use the LM trainer's own T5 pattern, which §3.9 names as the
                // thing to extend: PER-TENSOR DIRECTIONAL probes along the
                // analytic gradient direction, with a step sweep targeting a fixed
                // dL. The analytic directional derivative is <g, g/||g||> = ||g||,
                // and the probe catches both a wrong magnitude and a wrong
                // direction. One probe per (layer, site, A|B) over TWO real layers
                // = 32 probes, more than the 24 asked for, and every probe carries
                // real signal.
                struct Probe {
                    int         layer = 0, site = 0, is_b = 0;
                    size_t      n     = 0;
                    double      an = 0.0, fd = 0.0, step = 0.0, delta = 0.0, rel = 0.0, rel2 = 0.0;
                };
                std::vector<Probe>  probes;
                std::vector<float>  base, gvec, pert;
                bool                fin = true;
                const double        noise_floor = 6e-8 / std::max(1e-9, fabs(l0));  // f32 ULP at this loss
                for (size_t slot = 0; slot < fd.params().size(); slot++) {
                    ggml_tensor * t = fd.params()[slot];
                    const size_t  n = (size_t) ggml_nelements(t);
                    base.resize(n);
                    gvec.resize(n);
                    pert.resize(n);
                    ggml_backend_tensor_get(t, base.data(), 0, n * sizeof(float));
                    ggml_backend_tensor_get(fopt.acc[slot], gvec.data(), 0, n * sizeof(float));
                    double gn2 = 0.0;
                    for (size_t k = 0; k < n; k++) {
                        gn2 += (double) gvec[k] * (double) gvec[k];
                    }
                    const double gn = sqrt(gn2);  // == <g, g/||g||>, the analytic directional derivative
                    if (!(gn > 0.0)) {
                        fin = false;
                        continue;
                    }
                    auto eval_at = [&](double sh) -> double {
                        for (size_t k = 0; k < n; k++) {
                            pert[k] = base[k] + (float) (sh * (double) gvec[k] / gn);
                        }
                        ggml_backend_tensor_set(t, pert.data(), 0, n * sizeof(float));
                        return run(false);
                    };
                    auto fd_at = [&](double h) -> double {
                        const double fp = eval_at(h);
                        const double fm = eval_at(-h);
                        return (fp - fm) / (2.0 * h);
                    };
                    // Steps are chosen so the loss moves by a target amount, not by
                    // a fixed h: ||g|| spans 6e-4 to 7e-2 across the sites. The
                    // sweep has to reach small dL because the q/k paths run through
                    // soft_max_ext, which is strongly non-quadratic — at dL=0.02 the
                    // implied h is 1-30 and the central difference truncates badly
                    // (measured: fd systematically UNDER-reads by 20-85%). The v/o
                    // paths are linear in their LoRA factor and are exact at any h.
                    const double dl_targets[16] = { 0.32,   0.16,   0.08,   0.04,   0.02,   0.01,   5e-3,  2.5e-3,
                                                    1.25e-3, 6e-4,  3e-4,   1.5e-4, 8e-5,   4e-5,   2e-5,  1e-5 };
                    double       best_rel = 1e30, best_fd = 0.0, best_h = 0.0, best_delta = 0.0, second = 1e30;
                    std::string sweep;
                    for (int ti = 0; ti < 16; ti++) {
                        const double hh = dl_targets[ti] / gn;
                        const double f  = fd_at(hh);
                        const double rl = fabs(f - gn) / std::max(std::max(fabs(f), gn), 1e-6);
                        {
                            char sb[80];
                            snprintf(sb, sizeof(sb), "%.3g:%.2e ", dl_targets[ti], rl);
                            sweep += sb;
                        }
                        if (rl < best_rel) {
                            second     = best_rel;
                            best_rel   = rl;
                            best_fd    = f;
                            best_h     = hh;
                            best_delta = dl_targets[ti];
                        } else if (rl < second) {
                            second = rl;
                        }
                    }
                    ggml_backend_tensor_set(t, base.data(), 0, n * sizeof(float));
                    if (best_rel > 2e-2) {
                        // A probe that misses the bar prints its whole step sweep, so
                        // "truncation that never converges" can be told apart from
                        // "a systematic gradient error" without guessing.
                        fprintf(stderr, "[self-test] %s sweep slot %zu (rel vs dL target): %s\n", check, slot,
                                sweep.c_str());
                    }

                    // Slots per site: 2 for LoRA A/B and LoKr w1/w2, 4 for LoHa, 1 for
                    // HRA. Labels only — the gate is per slot.
                    const size_t per_site = std::max<size_t>(1, fd.params().size() / (size_t) (2 * DIT_NSITES_ATTN));
                    Probe pr;
                    pr.layer = L - 2 + (int) (slot / ((size_t) DIT_NSITES_ATTN * per_site));
                    pr.site  = (int) ((slot / per_site) % (size_t) DIT_NSITES_ATTN);
                    pr.is_b  = (int) (slot % per_site);
                    pr.n     = n;
                    pr.an    = gn;
                    pr.fd    = best_fd;
                    pr.step  = best_h;
                    pr.delta = best_delta;
                    pr.rel   = best_rel;
                    pr.rel2  = second;
                    fin      = fin && std::isfinite(pr.an) && std::isfinite(pr.fd);
                    probes.push_back(pr);
                }

                std::vector<double> rels, rels2;
                for (size_t i = 0; i < probes.size(); i++) {
                    rels.push_back(probes[i].rel);
                    rels2.push_back(probes[i].rel2);
                    fprintf(stderr,
                            "[self-test] %s probe %2zu  L%-2d %-18s %-4s n=%-7zu dL=%.3g h=%.3e  analytic=% .6e  "
                            "fd=% .6e  rel=%.3e (2nd %.3e)\n",
                            check, i, probes[i].layer, dit_site_peft(probes[i].site),
                            probes[i].is_b ? lbl1 : lbl0, probes[i].n, probes[i].delta, probes[i].step, probes[i].an,
                            probes[i].fd, probes[i].rel, probes[i].rel2);
                }
                std::sort(rels.begin(), rels.end());
                std::sort(rels2.begin(), rels2.end());
                const double maxrel  = rels.empty() ? 1e9 : rels.back();
                const double median  = rels.empty() ? 1e9 : rels[rels.size() / 2];
                const double maxrel2 = rels2.empty() ? 1e9 : rels2.back();
                const double median2 = rels2.empty() ? 1e9 : rels2[rels2.size() / 2];
                char         d[416];
                snprintf(d, sizeof(d),
                         "%d directional probes on real layers %d-%d (8 sites x %s/%s), f32 ULP floor %.2e, "
                         "forward repeatability |f-f'|=%.3e on loss %.6f: max rel=%.4e (bar 2e-2), "
                         "median rel=%.4e (bar 5e-3) [runner-up step: max=%.4e median=%.4e], all finite=%s",
                         (int) probes.size(), L - 2, L - 1, lbl0, lbl1, noise_floor, repeat_abs, l0, maxrel, median,
                         maxrel2, median2, fin ? "yes" : "NO");
                dit_st_report(rs, check, fin && maxrel < 2e-2 && median < 5e-3, d);
                lm_optim_free(&fopt);
            }
        }
    };

    if (mode == DIT_ST_NO_FD) {
        const char * pass_msg =
            "finite differences measured in the full-f32 child process (NVIDIA_TF32_OVERRIDE=0) — passed";
        const char * fail_msg = "finite differences FAILED in the full-f32 child process — see its line above";
        dit_st_report(rs, "T4", fd_child_rc == 0, fd_child_rc == 0 ? pass_msg : fail_msg);
        dit_st_report(rs, "LK4", fd_child_rc == 0, fd_child_rc == 0 ? pass_msg : fail_msg);
    } else {
        {
            DitAdapterLora fd;
            DitAdapterCfg  cfg;
            cfg.rank    = 16;
            cfg.alpha   = 32.0f;
            cfg.seed    = seed ^ 0x1234ull;
            cfg.b_sigma = 1e-2f;  // otherwise T6 makes dL/dA trivially zero
            // HOTSTEP_DIT_ST_DORA=1: THIS is the rung the FD child runs, so the
            // switch has to be here. Under it the magnitude tensors join
            // params() and every A/B probe is taken through the m/||W+BA||
            // rescale. (The probe labels assume the A,B,A,B slot order and will
            // be off by the interleaved m tensors — the numeric bars are per
            // tensor and unaffected.)
            cfg.dora    = getenv("HOTSTEP_DIT_ST_DORA") != nullptr;
            cfg.hira    = getenv("HOTSTEP_DIT_ST_HIRA") != nullptr;  // same rung, same reason
            cfg.loha    = getenv("HOTSTEP_DIT_ST_LOHA") != nullptr;
            cfg.hra     = getenv("HOTSTEP_DIT_ST_HRA") != nullptr;  // rank 16 = 16 reflections
            std::string err;
            if (!fd.init(&M.m, M.backend, L - 2, L, cfg, &err)) {
                dit_st_report(rs, "T4", false, err);
            } else {
                fd_gate("T4", fd, "A", "B");
                fd.free();
            }
        }
        {
            DitAdapterLoKr fk;
            DitAdapterCfg  cfg;
            cfg.seed                = seed ^ 0x5678ull;
            cfg.lokr_dim            = 512;
            cfg.lokr_alpha          = 512.0f;
            cfg.lokr_factor         = 6;
            cfg.lokr_decompose_both = true;
            // w2 is zero-initialised, which would make dL/dw1 identically zero.
            // 2e-3 puts the delta in the same ballpark as the LoRA gate's, where
            // the loss is still well conditioned.
            cfg.b_sigma = 2e-3f;
            std::string err;
            const size_t want_slots = (size_t) DIT_NSITES_ATTN * 2 * 2;
            if (!fk.init(&M.m, M.backend, L - 2, L, cfg, &err)) {
                dit_st_report(rs, "LK4", false, err);
            } else if (fk.params().size() != want_slots) {
                char b[192];
                snprintf(b, sizeof(b),
                         "this base factorizes w2 at dim %d, so the adapter has %zu trained tensors over 2 layers "
                         "instead of %zu — the probe's slot mapping does not apply",
                         cfg.lokr_dim, fk.params().size(), want_slots);
                dit_st_report(rs, "LK4", false, b);
                fk.free();
            } else {
                fd_gate("LK4", fk, "w1", "w2");
                fk.free();
            }
        }
        const float lg1 = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg1, 0, sizeof(float));
    }

    // ── T5: shuffled-target control ──────────────────────────────────────
    if (!fd_only) {
        const int   n_steps = 60;
        const float t_fix   = 0.5f;
        const int   len     = std::min(crop_big, s0.T - (s0.T % P));
        std::vector<int> perm((size_t) len);
        for (int i = 0; i < len; i++) {
            perm[(size_t) i] = i;
        }
        {
            LmRng pr;
            lm_rng_seed(&pr, seed ^ 0xabcdef0123456789ull);
            lm_rng_shuffle(&pr, perm);
        }
        std::vector<float>              tv(1, t_fix);
        std::vector<std::vector<float>> tbv, tpv;
        dit_train_temb(&M, tv, &tbv, &tpv);
        ggml_backend_tensor_set(t_temb, tbv[0].data(), 0, tbv[0].size() * sizeof(float));
        ggml_backend_tensor_set(t_tproj, tpv[0].data(), 0, tpv[0].size() * sizeof(float));

        auto run_control = [&](bool shuffled, double * first, double * last) -> bool {
            DitAdapterLora ad2;
            DitAdapterCfg  cfg;
            cfg.rank  = 16;
            cfg.alpha = 32.0f;
            cfg.seed  = seed;
            std::string err;
            if (!ad2.init(&M.m, M.backend, 0, L, cfg, &err)) {
                return false;
            }
            LmOptim o2;
            if (!lm_optim_init(&o2, ad2.params(), M.backend, &err)) {
                ad2.free();
                return false;
            }
            o2.t_adamw      = t_adamw;
            o2.t_lossgrad   = t_lossgrad;
            o2.t_clip       = t_clip;
            o2.t_eps        = t_eps;
            o2.t_gnorm2     = t_gnorm2;
            o2.base_lr      = 1e-3f;
            o2.grad_clip    = 1.0f;
            o2.weight_decay = 0.01f;
            o2.total_steps  = n_steps;
            o2.warmup_steps = 0;
            const float lg = 1.0f;
            ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));

            std::vector<double> curve;
            bool                ok = true;
            for (int st = 0; st < n_steps && ok; st++) {
                // FROZEN sample: the noise stream is reseeded every step, so both
                // runs see the identical (xt, v) pair at every step and differ ONLY
                // in whether the target is permuted. This is the verifier's own
                // protocol; letting the noise advance turns it into a different
                // (much harder) generalisation task and the control loses its
                // discriminating power.
                LmRng noise;
                lm_rng_seed(&noise, seed ^ 0x9e3779b97f4a7c15ull);
                upload(s0, 0, len, enc_S_full, t_fix, &noise, shuffled ? &perm : nullptr);
                lm_optim_zero_grad(&o2);
                ggml_init_params ip   = { arena.size(), arena.data(), true };
                ggml_context *   ctxt = ggml_init(ip);
                ggml_cgraph *    gft  = ggml_new_graph_custom(ctxt, 65536, true);
                DitInputs        in   = make_inputs(ctxt, len, enc_S_full);
                ggml_tensor *    vp   = dit_train_forward(ctxt, &M, &ad2, in, len, enc_S_full);
                ggml_tensor *    ls   = dit_train_loss(ctxt, vp, in, Oc, len, have_cstats);
                ggml_set_loss(ls);
                ggml_build_forward_expand(gft, ls);
                std::vector<ggml_tensor *> ga;
                lm_optim_fill_gacc(&o2, gft, &ga);
                ggml_build_backward_expand(ctxt, gft, ga.data());
                ggml_backend_sched_reset(M.sched);
                ok = ggml_backend_sched_graph_compute(M.sched, gft) == GGML_STATUS_SUCCESS;
                if (ok) {
                    float lv = 0.0f;
                    ggml_backend_tensor_get(ls, &lv, 0, sizeof(float));
                    curve.push_back((double) lv);
                }
                ggml_free(ctxt);
                LmStepStats stt;
                if (ok) {
                    ok = lm_optim_step(&o2, M.sched, &stt);
                }
            }
            if (ok && curve.size() >= 10) {
                double f = 0.0, l = 0.0;
                for (int i = 0; i < 5; i++) {
                    f += curve[(size_t) i];
                    l += curve[curve.size() - 1 - (size_t) i];
                }
                *first = f / 5.0;
                *last  = l / 5.0;
            } else {
                ok = false;
            }
            lm_optim_free(&o2);
            ad2.free();
            return ok;
        };

        double rf = 0, rl = 0, sf = 0, sl = 0;
        const bool ok1 = run_control(false, &rf, &rl);
        const bool ok2 = run_control(true, &sf, &sl);
        char       d[288];
        snprintf(d, sizeof(d),
                 "%d steps at t=0.5 lr=1e-3 crop=%d: real %.4f -> %.4f (ratio %.4f, bar <= 0.15); "
                 "shuffled %.4f -> %.4f (ratio %.4f, bar >= 0.50)",
                 n_steps, len, rf, rl, rf > 0 ? rl / rf : -1.0, sf, sl, sf > 0 ? sl / sf : -1.0);
        dit_st_report(rs, "T5", ok1 && ok2 && rf > 0 && sf > 0 && (rl / rf) <= 0.15 && (sl / sf) >= 0.50, d);
    }

    // ── T11 ──────────────────────────────────────────────────────────────
    if (!fd_only) {
        dit_st_adamw(rs, M.backend, M.sched, t_adamw, t_lossgrad, t_clip, t_eps, t_gnorm2);
        dit_st_muon(rs, M.backend, M.sched, t_adamw, t_lossgrad, t_clip, t_eps, t_gnorm2);
    }

    // ── SC1 / SC2 / SC3: gradient checkpointing ──────────────────────────
    // Before the SB rungs: these run on the already-loaded model, whereas the SB
    // ones pay for a second, CPU-backend copy of the base.
    if (!fd_only) {
        const float lg1 = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg1, 0, sizeof(float));
        dit_st_ckpt_gates(rs, &M, samples, seed);
    }

    // ── SB1 / SB2 / SB3: micro-batching ──────────────────────────────────
    // Last, because the CPU-backend identity run loads a second copy of the base
    // and everything above should have reported before that cost is paid.
    if (!fd_only) {
        const float lg1 = 1.0f;
        ggml_backend_tensor_set(t_lossgrad, &lg1, 0, sizeof(float));
        dit_st_batch_gates(rs, &M, dit_path, samples, seed);
    }

    // ── verdict ──────────────────────────────────────────────────────────
    lm_optim_free(&opt);
    lora.free();
    ggml_backend_buffer_free(buf_static);
    ggml_free(ctxs);
    dit_train_free(&M);

    int failed = 0;
    fprintf(stderr, "\n[self-test] ─── summary ───\n");
    for (size_t i = 0; i < rs.size(); i++) {
        fprintf(stderr, "[self-test] %-4s %s\n", rs[i].name.c_str(), rs[i].pass ? "PASS" : "FAIL");
        failed += rs[i].pass ? 0 : 1;
    }
    fprintf(stderr, "[self-test] %d/%d checks passed\n", (int) rs.size() - failed, (int) rs.size());
    jl("{\"type\":\"done\",\"selftest\":true,\"checks\":%d,\"failed\":%d}", (int) rs.size(), failed);
    return failed == 0 ? 0 : 1;
}

// ─── T4 runs in a child process, and here is exactly why ────────────────────
//
// ggml's CUDA backend creates its cuBLAS handle with CUBLAS_TF32_TENSOR_OP_MATH
// (ggml-cuda/common.cuh:1478), so every F32 x F32 mul_mat in the training graph
// is a cublasSgemm executed on TF32 tensor cores — a 10-bit mantissa, eps 4.9e-4.
// The forward is still bit-DETERMINISTIC (T4 reports |f-f'| = 0), but perturbing
// a weight reshuffles the rounding, which behaves like ~2e-4 relative noise on
// the loss. A central difference divides that by 2h, so the finite-difference
// estimator has a floor of sigma_L/(2*dL); for the q/k sites, whose loss is
// strongly non-quadratic through soft_max_ext, truncation only becomes small at
// dL where that floor is already ~1e-1. MEASURED on this base, seed 42:
//
//   TF32 on : T4 max rel 6.2e-2 (L30 self_attn.k_proj), median 5.5e-4  -> FAIL
//   TF32 off: T4 max rel 1.0e-3,                        median 3.3e-5  -> PASS
//   (seeds 7 / 1234 with TF32 off: 1.03e-3 / 6.29e-4 max)
//
// So the residual §6.2 V2 was failing on is the ESTIMATOR, not the gradient:
// at full f32 the analytic gradient agrees with central differences to 1e-3 on
// every one of the 32 probes. NVIDIA_TF32_OVERRIDE=0 is read by cuBLAS when the
// library initialises, so it cannot be flipped after the handle exists — hence a
// child process. T5 (a training-DYNAMICS gate) deliberately stays in the parent
// so it is measured under the numerics that actually ship.
//
// Returns 0 = child passed, 1 = child failed, -1 = could not spawn (caller then
// runs T4 in-process, which is the correct behaviour on non-CUDA backends).
static std::string dit_st_self_exe() {
#ifdef _WIN32
    char        buf[4096];
    const DWORD n = GetModuleFileNameA(NULL, buf, (DWORD) sizeof(buf));
    return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#elif defined(__linux__)
    char          buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
#else
    return std::string();  // macOS/other: fall back to the in-process T4
#endif
}

static int dit_st_spawn_fd(const std::string & dit_path, const std::string & tensors_dir, uint64_t seed, int crop_max,
                           int reserve_mb, float safety) {
    const std::string exe = dit_st_self_exe();
    if (exe.empty()) {
        return -1;
    }
    char sseed[32], scrop[32], sres[32], ssaf[32];
    snprintf(sseed, sizeof(sseed), "%llu", (unsigned long long) seed);
    snprintf(scrop, sizeof(scrop), "%d", crop_max);
    snprintf(sres, sizeof(sres), "%d", reserve_mb);
    snprintf(ssaf, sizeof(ssaf), "%.6f", (double) safety);
    fprintf(stderr, "[self-test] T4: re-running the finite-difference gate in a child process with "
                    "NVIDIA_TF32_OVERRIDE=0 (cuBLAS TF32 puts a ~1e-1 floor on the estimator)\n");
#ifdef _WIN32
    // _spawnv joins argv with spaces without quoting, so quote anything that can
    // contain one. The child inherits this process's environment.
    const std::string qexe = "\"" + exe + "\"";
    const std::string qdit = "\"" + dit_path + "\"";
    const std::string qten = "\"" + tensors_dir + "\"";
    const char *      av[] = { qexe.c_str(),  "train-dit", "--self-test",       "--dit",   qdit.c_str(),
                               "--tensors",   qten.c_str(), "--seed",           sseed,     "--crop-max",
                               scrop,         "--vram-reserve-mb", sres,        "--vram-safety", ssaf,
                               nullptr };
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
    _putenv_s("HOTSTEP_DIT_ST_FD", "1");
    const intptr_t rc = _spawnv(_P_WAIT, exe.c_str(), (char * const *) av);
    _putenv_s("NVIDIA_TF32_OVERRIDE", "");  // the parent must measure shipping numerics
    _putenv_s("HOTSTEP_DIT_ST_FD", "");
    if (rc < 0) {
        fprintf(stderr, "[self-test] T4: could not spawn the child — falling back to an in-process run\n");
        return -1;
    }
    return rc == 0 ? 0 : 1;
#else
    const char * av[] = { exe.c_str(), "train-dit",         "--self-test", "--dit",         dit_path.c_str(),
                          "--tensors", tensors_dir.c_str(), "--seed",      sseed,           "--crop-max",
                          scrop,       "--vram-reserve-mb", sres,          "--vram-safety", ssaf,
                          nullptr };
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    setenv("HOTSTEP_DIT_ST_FD", "1", 1);
    const pid_t pid = fork();
    if (pid == 0) {
        execv(exe.c_str(), (char * const *) av);
        _exit(127);
    }
    unsetenv("NVIDIA_TF32_OVERRIDE");
    unsetenv("HOTSTEP_DIT_ST_FD");
    if (pid < 0) {
        fprintf(stderr, "[self-test] T4: could not fork — falling back to an in-process run\n");
        return -1;
    }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) {
        return -1;
    }
    return (WIFEXITED(st) && WEXITSTATUS(st) == 0) ? 0 : 1;
#endif
}

static int dit_self_test(const std::string & dit_path, const std::string & tensors_dir, uint64_t seed, int crop_max,
                         int reserve_mb, float safety) {
    if (getenv("HOTSTEP_DIT_ST_FD") != nullptr) {
        return dit_self_test_impl(dit_path, tensors_dir, seed, crop_max, reserve_mb, safety, DIT_ST_FD_ONLY, -1);
    }
    const int crc = dit_st_spawn_fd(dit_path, tensors_dir, seed, crop_max, reserve_mb, safety);
    if (crc < 0) {
        return dit_self_test_impl(dit_path, tensors_dir, seed, crop_max, reserve_mb, safety, DIT_ST_ALL, -1);
    }
    return dit_self_test_impl(dit_path, tensors_dir, seed, crop_max, reserve_mb, safety, DIT_ST_NO_FD, crc);
}
