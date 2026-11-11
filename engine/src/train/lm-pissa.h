#pragma once
// lm-pissa.h — PiSSA initialisation for the shared LM LoRA layer (Meng 2024).
// HOT-Step file, 2026-09-05. Ported from train/dit-pissa.h; the maths is the
// same, one design decision is not.
//
// A plain LoRA starts at A ~ N(0, 1/sqrt(in)), B = 0 and has to find the
// directions that matter. PiSSA starts the pair on the base weight's own top-r
// singular directions (A = V_r sqrt(S_r), B = sqrt(S_r) U_r^T) so the optimizer
// fine-tunes the principal subspace directly, which is where most of a weight's
// capacity lives. For that to leave the first forward unchanged, the same
// energy has to come OUT of the frozen path.
//
// THE ONE DEVIATION FROM THE DiT. dit_pissa_init writes W_res = W - s B0 A0
// back into the F32 mirror. The LM cannot: mm3-lm-train's shipped recipe trains
// against a q8_0 base (the f16 one leaves no room for the optimizer), and there
// is no F32 mirror to receive a residual — requantizing W_res would perturb
// every weight in the model to fix r directions. So the residual is folded into
// the ADAPTER instead. lm_lora_init allocates a frozen A0/B0 per site and
// lm_linear emits
//
//     y = W x + s (B A) x - s (B0 A0) x,
//
// which is exactly (W - s B0 A0) + s B A as a function, starts at exactly W,
// and never touches a base weight. The cost is two extra rank-r mul_mats per
// site per forward and 2x the adapter's parameter memory (A0/B0 are the same
// shape as A/B) — paid only under --pissa.
//
// The SVD is randomized, with the two big products on the GPU:
//   stage T (GPU)  W^T into a scratch, plus ||W||_F^2   — one cast per site,
//                  which is also how a q8_0 site gets an F32 view of itself
//   stage A (GPU)  Y = (W W^T)^iters W Omega            q = r + oversample
//   stage B (host) Q = qr(Y)                            modified Gram-Schmidt
//   stage C (GPU)  C = Q^T W                            [q, in]
//   stage D (host) C C^T = Uc diag(lambda) Uc^T (Jacobi), S = sqrt(lambda),
//                  U_r = Q Uc_r, V_r = C^T Uc_r / S_r
//
// The captured-energy line (sum lambda_k / ||W||_F^2) is the correctness
// diagnostic: a working SVD captures far more than a random rank-r subspace's
// r/min(in,out).
//
// Export (lm-hra.h's neighbour, in lm-export.h) is a plain PEFT LoRA of rank 2r
// on the ORIGINAL base — s(BA - B0A0) = s [B, -B0] [A; A0] — so the merge and
// runtime paths need no PiSSA knowledge whatsoever.

#include "backend.h"
#include "pissa-residual.h"
#include "qwen3-enc.h"
#include "train/lm-export.h"
#include "train/lm-graph.h"
#include "train/svd-host.h"
#include "hot-step-fsutf8.h"
#include "train/preprocess-io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct LmPissaStats {
    double energy_min = 1.0, energy_mean = 0.0;  // captured / ||W||_F^2 over sites
    int    sites      = 0;
    double seconds    = 0.0;                     // wall time of the init, SVD or cache
    bool   from_cache = false;
};

// F16-aware upload/download of a frozen factor: the tensor may be F16 under
// --pissa-frozen-f16, the host side is always F32.
static void lm_pissa_tensor_set_f32(ggml_tensor * t, const std::vector<float> & v) {
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t) v.size());
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    }
}
static void lm_pissa_tensor_get_f32(const ggml_tensor * t, std::vector<float> & v) {
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(v.size());
        ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(h.data(), v.data(), (int64_t) v.size());
    } else {
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    }
}

// ── SVD init cache (2026-09-06) ─────────────────────────────────────────────
//
// The randomized SVD is deterministic (fixed RNG seed, fixed base, fixed
// rank/oversample/iters), so its output is a pure function of the base file
// and four integers. One MM3 run at r128 spends its whole init recomputing
// 252 sites that every previous run at the same rank already computed. The
// cache stores the F32 host factors and each site's captured-energy fraction;
// a hit skips the GPU stages entirely and uploads the same bytes the SVD
// would have produced. Keyed on the base's file name + size + mtime, so a
// re-quantized base of the same name misses rather than poisons.
//
// Layout (little-endian): magic "HPIS", u32 version=1, i32 rank, i32 q,
// i32 iters, i32 layer_lo, i32 layer_hi, i32 n_sites, then per site:
// i32 layer, i32 slot, i32 in, i32 out, f64 frac, f32 A0[in*r], f32 B0[r*out].
struct LmPissaCacheSite {
    int                l = 0, s = 0, in = 0, out = 0;
    double             frac = 0.0;
    std::vector<float> A0, B0;
};
static std::string lm_pissa_cache_path(const std::string & dir, const std::string & lm_path, int rank, int q, int iters,
                                       int layer_lo, int layer_hi) {
    if (dir.empty() || lm_path.empty()) {
        return "";
    }
    HS_STAT_T sb {};
    if (hs_stat(lm_path, &sb) != 0) {
        return "";
    }
    std::string base = lm_path;
    const size_t cut = base.find_last_of("/\\");
    if (cut != std::string::npos) {
        base = base.substr(cut + 1);
    }
    char tail[160];
    snprintf(tail, sizeof(tail), ".%lld.%lld.r%d.q%d.i%d.L%d-%d.v1.pissa", (long long) sb.st_size, (long long) sb.st_mtime,
             rank, q, iters, layer_lo, layer_hi);
    return dir + "/" + base + tail;
}
static bool lm_pissa_cache_read(const std::string & path, int rank, int q, int iters, int layer_lo, int layer_hi,
                                std::vector<LmPissaCacheSite> * out) {
    FILE * f = hs_fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    auto rd = [&](void * p, size_t n) { return fread(p, 1, n, f) == n; };
    char     magic[4] = { 0, 0, 0, 0 };
    uint32_t ver      = 0;
    int32_t  hdr[6]   = { 0, 0, 0, 0, 0, 0 };
    bool ok = rd(magic, 4) && rd(&ver, 4) && rd(hdr, sizeof(hdr)) && memcmp(magic, "HPIS", 4) == 0 && ver == 1 &&
              hdr[0] == rank && hdr[1] == q && hdr[2] == iters && hdr[3] == layer_lo && hdr[4] == layer_hi && hdr[5] > 0;
    if (ok) {
        out->resize((size_t) hdr[5]);
        for (LmPissaCacheSite & c : *out) {
            int32_t h[4];
            ok = rd(h, sizeof(h)) && rd(&c.frac, sizeof(double));
            if (!ok) {
                break;
            }
            c.l = h[0]; c.s = h[1]; c.in = h[2]; c.out = h[3];
            if (c.in <= 0 || c.out <= 0 || c.in > (1 << 20) || c.out > (1 << 20)) {
                ok = false;
                break;
            }
            c.A0.resize((size_t) c.in * (size_t) rank);
            c.B0.resize((size_t) rank * (size_t) c.out);
            ok = rd(c.A0.data(), c.A0.size() * sizeof(float)) && rd(c.B0.data(), c.B0.size() * sizeof(float));
            if (!ok) {
                break;
            }
        }
    }
    fclose(f);
    if (!ok) {
        out->clear();
    }
    return ok;
}
static bool lm_pissa_cache_write(const std::string & path, int rank, int q, int iters, int layer_lo, int layer_hi,
                                 const std::vector<LmPissaCacheSite> & sites) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp.c_str(), "wb");
    if (!f) {
        return false;
    }
    auto wr = [&](const void * p, size_t n) { return fwrite(p, 1, n, f) == n; };
    const uint32_t ver = 1;
    const int32_t  hdr[6] = { rank, q, iters, layer_lo, layer_hi, (int32_t) sites.size() };
    bool ok = wr("HPIS", 4) && wr(&ver, 4) && wr(hdr, sizeof(hdr));
    for (const LmPissaCacheSite & c : sites) {
        if (!ok) {
            break;
        }
        const int32_t h[4] = { c.l, c.s, c.in, c.out };
        ok = wr(h, sizeof(h)) && wr(&c.frac, sizeof(double)) && wr(c.A0.data(), c.A0.size() * sizeof(float)) &&
             wr(c.B0.data(), c.B0.size() * sizeof(float));
    }
    ok = (fclose(f) == 0) && ok;
    if (ok) {
        hs_remove(path.c_str());
        ok = hs_rename(tmp.c_str(), path.c_str()) == 0;
    }
    if (!ok) {
        hs_remove(tmp.c_str());
    }
    return ok;
}

// ── the shipped residual (2026-09-09) ───────────────────────────────────────
//
// pissa-residual.h describes the file. This is its writer: canonical (scale 1)
// factors, F16, one file per base and rank, beside the base GGUF. Once it
// exists every adapter trained against it exports in the DELTA form (rank r,
// adapter-only, ~0.7 GB at r128 instead of 2.8 GB) and the runtime loader
// rebuilds the full pair from the two files.
static bool lm_pissa_residual_write(const std::string & path, const PissaResidualMeta & m,
                                    const std::vector<PissaResidualSite> & sites, std::string * err) {
    std::vector<STWTensor> tensors;
    tensors.reserve(sites.size() * 2 + 1);
    float hi = 0.0f, lo = 0.0f;
    pissa_size_split(m.base_size, &hi, &lo);
    std::vector<float> meta = { (float) PISSA_RESIDUAL_VERSION, (float) m.rank,     (float) m.q,
                                (float) m.iters,                (float) m.layer_lo, (float) m.layer_hi,
                                hi,                             lo,
                                (float) m.energy_mean,          (float) m.energy_min };
    {
        STWTensor t;
        t.name           = "pissa.meta";
        t.shape          = { (int64_t) meta.size() };
        t.data           = meta.data();
        t.dtype_override = STW_F32;  // base size hi/lo and the version must not round
        tensors.push_back(t);
    }
    std::vector<std::string> names;
    names.reserve(sites.size() * 2);
    for (const PissaResidualSite & s : sites) {
        names.push_back("pissa.L" + std::to_string(s.l) + "." + s.site + ".A0");
        names.push_back("pissa.L" + std::to_string(s.l) + "." + s.site + ".B0");
    }
    for (size_t i = 0; i < sites.size(); i++) {
        const PissaResidualSite & s = sites[i];
        STWTensor a;
        a.name  = names[2 * i];
        a.shape = { (int64_t) m.rank, (int64_t) s.in };   // torch [rank, in] == ggml [in, rank]
        a.data  = s.A0.data();
        tensors.push_back(a);
        STWTensor b;
        b.name  = names[2 * i + 1];
        b.shape = { (int64_t) s.out, (int64_t) m.rank };  // torch [out, rank] == ggml [rank, out]
        b.data  = s.B0.data();
        tensors.push_back(b);
    }
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "hot_step_pissa_residual", "v1" });
    md.push_back({ "producer", std::string("ace-train ") + ACE_VERSION });
    if (!st_write_file(path.c_str(), tensors, md, STW_F16)) {
        *err = "cannot write " + path;
        return false;
    }
    return true;
}

// Fills A/B with the SVD factors and A0/B0 with a frozen copy of them. The base
// is not written. `sched` runs the GPU stages; it is reset around every graph,
// so call this before the training graphs exist.
//
// Sources, in order: the residual file beside the base (residual_path, must
// match rank / q / iters / layers / base size), the SVD cache (cache_path),
// the SVD itself. Whatever the source, the factors the trainer holds are the
// F16-ROUNDED canonical values divided by sqrt(s): the same bytes a loader will
// read back from the residual file, so an adapter exported in the delta form
// reproduces the trained function exactly and not to within f16 of it. When the
// residual file did not exist and residual_path is writable, it is written.
static bool lm_pissa_init(LmLora * L, ggml_backend_sched_t sched, int oversample, int iters, LmPissaStats * stats,
                          std::string * err, const std::string & cache_path = "",
                          const std::string & residual_path = "", long long base_size = 0) {
    Qwen3LM * lm = L->model;
    if (!lm || !L->pissa || L->layer_lo >= L->layer_hi) {
        *err = "PiSSA: adapter not initialised for it";
        return false;
    }
    const int r = L->rank;
    const int q = r + std::max(0, oversample);
    iters       = std::max(0, std::min(4, iters));
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    };
    // The factors are stored so that s * B0 A0 reproduces W's rank-r truncation
    // EXACTLY, whatever the scale is. The DiT bakes sqrt(S) into each factor and
    // then subtracts s*B0A0, which only lands on the true PiSSA residual when
    // alpha == rank; dividing both factors by sqrt(s) here makes it right for
    // any alpha and for rsLoRA, at zero cost when s == 1.
    const double sqrt_s     = (L->scale > 0.0f) ? sqrt((double) L->scale) : 1.0;
    const double inv_sqrt_s = 1.0 / sqrt_s;
    L->pissa_residual.clear();
    L->pissa_fingerprint.clear();
    L->pissa_base_size = 0;

    std::string residual_base;
    if (!residual_path.empty()) {
        const size_t cut = residual_path.find_last_of("/\\");
        residual_base    = cut == std::string::npos ? residual_path : residual_path.substr(cut + 1);
    }

    // ── residual file: the shipped frozen half ───────────────────────────
    if (!residual_path.empty() && hs_file_exists(residual_path)) {
        PissaResidualMeta              rm;
        std::vector<PissaResidualSite> rsites;
        std::string                    rerr;
        if (!pissa_residual_read_meta(residual_path, &rm, &rerr)) {
            fprintf(stderr, "[pissa] residual %s unreadable (%s); falling back\n", residual_path.c_str(), rerr.c_str());
        } else if (rm.rank != r || rm.q != q || rm.iters != iters || rm.layer_lo != L->layer_lo ||
                   rm.layer_hi != L->layer_hi || (base_size > 0 && rm.base_size != base_size)) {
            fprintf(stderr,
                    "[pissa] residual %s was made for rank %d q %d iters %d layers %d-%d base %lld B; this run is "
                    "rank %d q %d iters %d layers %d-%d base %lld B. Not used; the export will be standalone.\n",
                    residual_path.c_str(), rm.rank, rm.q, rm.iters, rm.layer_lo, rm.layer_hi, rm.base_size, r, q,
                    iters, L->layer_lo, L->layer_hi, base_size);
        } else if (!pissa_residual_read(residual_path, &rm, &rsites, &rerr, true)) {
            fprintf(stderr, "[pissa] residual %s unreadable (%s); falling back\n", residual_path.c_str(), rerr.c_str());
        } else {
            bool ok = true;
            int  n  = 0;
            for (int l = L->layer_lo; l < L->layer_hi && ok; l++) {
                for (int s = 0; s < QW_LORA_NSLOTS; s++) {
                    QwLoraPair & pr = L->layers[l].p[s];
                    if (!pr.A || !pr.B || !pr.A0 || !pr.B0) {
                        continue;
                    }
                    PissaResidualSite * c = nullptr;
                    for (PissaResidualSite & cand : rsites) {
                        if (cand.l == l && cand.site == lm_slot_peft_name(s)) {
                            c = &cand;
                            break;
                        }
                    }
                    if (!c || c->in != (int) pr.A->ne[0] || c->out != (int) pr.B->ne[1]) {
                        fprintf(stderr, "[pissa] residual %s has no usable site for layer %d %s; falling back\n",
                                residual_path.c_str(), l, lm_slot_peft_name(s));
                        ok = false;
                        break;
                    }
                    if (inv_sqrt_s != 1.0) {
                        for (float & x : c->A0) x = (float) (x * inv_sqrt_s);
                        for (float & x : c->B0) x = (float) (x * inv_sqrt_s);
                    }
                    ggml_backend_tensor_set(pr.A, c->A0.data(), 0, c->A0.size() * sizeof(float));
                    ggml_backend_tensor_set(pr.B, c->B0.data(), 0, c->B0.size() * sizeof(float));
                    lm_pissa_tensor_set_f32(pr.A0, c->A0);
                    lm_pissa_tensor_set_f32(pr.B0, c->B0);
                    // Done with this site's 5+ MB; the file holds 252 of them.
                    std::vector<float>().swap(c->A0);
                    std::vector<float>().swap(c->B0);
                    n++;
                }
            }
            if (ok) {
                L->pissa_residual  = residual_base;
                L->pissa_fingerprint = rm.fingerprint;
                L->pissa_base_size = rm.base_size;
                if (stats) {
                    stats->sites       = n;
                    stats->energy_mean = rm.energy_mean;
                    stats->energy_min  = rm.energy_min;
                    stats->seconds     = elapsed();
                    stats->from_cache  = true;
                }
                fprintf(stderr, "[pissa] init: %d sites from residual %s in %.1f s (delta export)\n", n,
                        residual_path.c_str(), elapsed());
                return true;
            }
        }
    }

    // Canonical, f16-rounded copies of every site's frozen pair, for the
    // residual write. Only collected when there is somewhere to write them.
    // A fallback must not overwrite a residual that existing adapters still need.
    const bool                     want_write = !residual_path.empty() && !hs_file_exists(residual_path);
    std::vector<PissaResidualSite> res_out;
    std::vector<ggml_fp16_t>       h16;
    // Takes trainer-space host factors, rounds them through the canonical F16
    // representation the residual file uses, uploads, and keeps the canonical
    // copy for the write. This is what makes "what the trainer subtracted" and
    // "what the file says" the same numbers.
    auto place = [&](QwLoraPair & pr, int l, int s, int in, int out, std::vector<float> & A0t,
                     std::vector<float> & B0t) {
        auto canon_round = [&](std::vector<float> & v) {
            if (sqrt_s != 1.0) {
                for (float & x : v) x = (float) (x * sqrt_s);
            }
            h16.resize(v.size());
            ggml_fp32_to_fp16_row(v.data(), h16.data(), (int64_t) v.size());
            ggml_fp16_to_fp32_row(h16.data(), v.data(), (int64_t) v.size());
        };
        canon_round(A0t);
        canon_round(B0t);
        if (want_write) {
            PissaResidualSite rs;
            rs.l    = l;
            rs.site = lm_slot_peft_name(s);
            rs.in   = in;
            rs.out  = out;
            rs.A0   = A0t;
            rs.B0   = B0t;
            res_out.push_back(std::move(rs));
        }
        if (inv_sqrt_s != 1.0) {
            for (float & x : A0t) x = (float) (x * inv_sqrt_s);
            for (float & x : B0t) x = (float) (x * inv_sqrt_s);
        }
        ggml_backend_tensor_set(pr.A, A0t.data(), 0, A0t.size() * sizeof(float));
        ggml_backend_tensor_set(pr.B, B0t.data(), 0, B0t.size() * sizeof(float));
        lm_pissa_tensor_set_f32(pr.A0, A0t);
        lm_pissa_tensor_set_f32(pr.B0, B0t);
    };
    auto write_residual = [&](double e_mean, double e_min) {
        if (!want_write || res_out.empty()) {
            return;
        }
        PissaResidualMeta wm;
        wm.version     = PISSA_RESIDUAL_VERSION;
        wm.rank        = r;
        wm.q           = q;
        wm.iters       = iters;
        wm.layer_lo    = L->layer_lo;
        wm.layer_hi    = L->layer_hi;
        wm.base_size   = base_size;
        wm.energy_mean = e_mean;
        wm.energy_min  = e_min;
        std::string werr;
        if (lm_pissa_residual_write(residual_path, wm, res_out, &werr)) {
            STFile written;
            if (st_open(&written, residual_path.c_str())) {
                L->pissa_fingerprint = pissa_residual_fingerprint(written);
                st_close(&written);
                L->pissa_residual  = residual_base;
                L->pissa_base_size = base_size;
                fprintf(stderr, "[pissa] residual generated: %s (%zu sites, F16). Adapters from this run fit "
                        "ONLY this exact file. Back it up with the adapters; another residual with the same "
                        "name, rank and base is not interchangeable.\n", residual_path.c_str(), res_out.size());
            } else {
                fprintf(stderr, "[pissa] cannot fingerprint generated residual %s; exporting standalone rank-2r files\n",
                        residual_path.c_str());
            }
        } else {
            fprintf(stderr, "[pissa] residual NOT written (%s); this run exports standalone rank-2r files\n",
                    werr.c_str());
        }
        std::vector<PissaResidualSite>().swap(res_out);
    };

    // ── cache hit: upload the stored factors and skip the SVD ─────────────
    std::vector<LmPissaCacheSite> cached;
    if (!cache_path.empty() && lm_pissa_cache_read(cache_path, r, q, iters, L->layer_lo, L->layer_hi, &cached)) {
        size_t idx = 0;
        double e_sum = 0.0, e_min = 1.0;
        int    n_e   = 0;
        bool   ok    = true;
        for (int l = L->layer_lo; l < L->layer_hi && ok; l++) {
            for (int s = 0; s < QW_LORA_NSLOTS; s++) {
                QwLoraPair & pr = L->layers[l].p[s];
                if (!pr.A || !pr.B || !pr.A0 || !pr.B0) {
                    continue;
                }
                if (idx >= cached.size()) {
                    ok = false;
                    break;
                }
                LmPissaCacheSite & c = cached[idx++];
                if (c.l != l || c.s != s || c.in != (int) pr.A->ne[0] || c.out != (int) pr.B->ne[1]) {
                    ok = false;
                    break;
                }
                place(pr, l, s, c.in, c.out, c.A0, c.B0);
                std::vector<float>().swap(c.A0);
                std::vector<float>().swap(c.B0);
                if (stats) {
                    stats->energy_min = std::min(stats->energy_min, c.frac);
                    stats->sites++;
                }
                e_min = std::min(e_min, c.frac);
                e_sum += c.frac;
                n_e++;
            }
        }
        if (ok && idx == cached.size()) {
            const double e_mean = n_e > 0 ? e_sum / (double) n_e : 0.0;
            if (stats) {
                stats->energy_mean = e_mean;
                stats->seconds     = elapsed();
                stats->from_cache  = true;
            }
            fprintf(stderr, "[pissa] init: %zu sites from cache in %.1f s (%s)\n", cached.size(), elapsed(),
                    cache_path.c_str());
            write_residual(e_mean, e_min);
            return true;
        }
        // A stale or mismatched file: fall through to the SVD and overwrite it.
        fprintf(stderr, "[pissa] cache %s does not match this run; recomputing\n", cache_path.c_str());
        std::vector<PissaResidualSite>().swap(res_out);
        if (stats) {
            stats->energy_min = 1.0;
            stats->sites      = 0;
        }
    }
    std::vector<LmPissaCacheSite> to_cache;

    // Scratch sized by the largest site in the trained window.
    int64_t in_max = 0, out_max = 0;
    for (int s = 0; s < QW_LORA_NSLOTS; s++) {
        ggml_tensor * w = lm_slot_weight(&lm->layers[L->layer_lo], s);
        if (!w) {
            *err = "PiSSA: a trained slot has no base weight (fused projections?)";
            return false;
        }
        in_max  = std::max(in_max, w->ne[0]);
        out_max = std::max(out_max, w->ne[1]);
    }
    if ((int64_t) q > std::min(in_max, out_max)) {
        *err = "PiSSA: rank + oversample exceeds the smallest weight dimension";
        return false;
    }

    ggml_context * sctx = nullptr;
    {
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        sctx                = ggml_init(ip);
    }
    // t_wt is the site's weight TRANSPOSED and in F32: [out, in]. Both stages
    // that contract over the output axis need it, and building it with
    // qwen3_f32 is what gives a q8_0 site an F32 view of itself without
    // dequantizing the whole model.
    ggml_tensor * t_wt    = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, in_max);
    ggml_tensor * t_omega = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, in_max, q);
    ggml_tensor * t_y     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_q     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_c     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, q, in_max);
    ggml_tensor * t_en    = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { t_wt, t_omega, t_y, t_q, t_c, t_en }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, lm->backend);
    if (!sbuf) {
        *err = "PiSSA: scratch allocation failed";
        ggml_free(sctx);
        return false;
    }
    fprintf(stderr, "[pissa] scratch %.1f MB (largest site %lld x %lld), rank %d + oversample %d, %d power iteration(s)\n",
            (double) ggml_backend_buffer_get_size(sbuf) / 1048576.0, (long long) in_max, (long long) out_max, r,
            std::max(0, oversample), iters);
    auto done = [&](bool ok) {
        ggml_backend_buffer_free(sbuf);
        ggml_free(sctx);
        return ok;
    };

    std::vector<uint8_t> arena(ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false));
    auto                 compute = [&](ggml_cgraph * gf, ggml_context * ctx) -> bool {
        ggml_backend_sched_reset(sched);
        const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
        ggml_free(ctx);
        return ok;
    };
    auto view2 = [&](ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1) {
        return ggml_view_2d(ctx, t, ne0, ne1, (size_t) ne0 * sizeof(float), 0);
    };

    // sqrt_s / inv_sqrt_s are defined at the top of this function.
    LmRng rng;
    lm_rng_seed(&rng, 0x9155a000ull ^ (uint64_t) L->rank);
    std::vector<float> host;
    double             e_sum = 0.0;
    int                n_e   = 0;

    for (int l = L->layer_lo; l < L->layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair &  pr = L->layers[l].p[s];
            if (!pr.A || !pr.B || !pr.A0 || !pr.B0) {
                continue;
            }
            ggml_tensor * w   = lm_slot_weight(&lm->layers[l], s);
            if (!w) {
                *err = "PiSSA: a trained slot has no base weight";
                return done(false);
            }
            const int in  = (int) w->ne[0];
            const int out = (int) w->ne[1];

            // ── T: W^T (F32) into the scratch, and ||W||_F^2 ───────────────
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    wf  = qwen3_f32(ctx, w);                          // [in, out]
                ggml_tensor *    wt  = ggml_cont(ctx, ggml_transpose(ctx, wf));    // [out, in]
                ggml_build_forward_expand(gf, ggml_cpy(ctx, wt, view2(ctx, t_wt, out, in)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_sum(ctx, ggml_sqr(ctx, wf)), t_en));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: transpose/norm graph failed";
                    return done(false);
                }
            }
            float w_fro2 = 0.0f;
            ggml_backend_tensor_get(t_en, &w_fro2, 0, sizeof(float));

            // ── A: Y = (W W^T)^iters W Omega ──────────────────────────────
            host.assign((size_t) in * (size_t) q, 0.0f);
            lm_rng_fill_normal(&rng, host, 1.0f);
            ggml_backend_tensor_set(t_omega, host.data(), 0, host.size() * sizeof(float));
            // One graph per pass, with Y's columns brought back to norm [0.5, 1)
            // on the host in between. The power iteration grows Y by about
            // sigma_max^2 a pass (2.5e6 on layer 0's q_proj), and a GPU matmul
            // that stages F32 operands through f16 turns that into inf: cuBLAS's
            // F16 path did (hence both operands F32 below), and Metal's
            // kernel_mul_mm_f32_f32 stages BOTH operands as half whatever the
            // dtype (GH #335, mlp.up_proj on layer 0). The range finder only uses
            // Y's column space, and a power-of-two factor is exact in F32, so a
            // backend that multiplies in F32 gives the same bits as the old
            // single graph (the CPU cache is byte-identical). That does not hold
            // through half: the rescaled entries round differently there, and
            // small ones go subnormal or to zero, with no overflow involved.
            for (int it = 0; it <= iters; it++) {
                if (it > 0) {
                    host.resize((size_t) out * (size_t) q);
                    ggml_backend_tensor_get(t_y, host.data(), 0, host.size() * sizeof(float));
                    for (int k = 0; k < q; k++) {
                        float * col = host.data() + (size_t) k * (size_t) out;
                        double  n2  = 0.0;
                        for (int j = 0; j < out; j++) {
                            n2 += (double) col[j] * (double) col[j];
                        }
                        int e = 0;
                        std::frexp(std::sqrt(n2), &e);
                        if (std::isfinite(n2) && n2 > 0.0 && e != 0) {
                            for (int j = 0; j < out; j++) {
                                col[j] = std::ldexp(col[j], -e);
                            }
                        }
                    }
                    ggml_backend_tensor_set(t_y, host.data(), 0, host.size() * sizeof(float));
                }
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                // BOTH operands are F32 here, and that is not tidiness: the base
                // in its own F16 dtype sends src1 through cuBLAS's F16 conversion.
                ggml_tensor * wf = qwen3_f32(ctx, w);
                ggml_tensor * y;
                if (it == 0) {
                    y = ggml_mul_mat(ctx, wf, view2(ctx, t_omega, in, q));                    // [out, q]
                } else {
                    ggml_tensor * z = ggml_mul_mat(ctx, view2(ctx, t_wt, out, in), view2(ctx, t_y, out, q));  // [in, q]
                    y               = ggml_mul_mat(ctx, wf, z);                                // [out, q]
                }
                ggml_build_forward_expand(gf, ggml_cpy(ctx, y, view2(ctx, t_y, out, q)));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: stage A graph failed";
                    return done(false);
                }
            }
            // ── B: Q = qr(Y) on the host ──────────────────────────────────
            std::vector<double> Yd((size_t) out * (size_t) q);
            host.resize(Yd.size());
            ggml_backend_tensor_get(t_y, host.data(), 0, host.size() * sizeof(float));
            double ymax = 0.0;
            for (size_t i = 0; i < Yd.size(); i++) {
                Yd[i] = (double) host[i];
                ymax  = std::max(ymax, std::fabs(Yd[i]));
            }
            if (!std::isfinite(ymax)) {
                char b[224];
                snprintf(b, sizeof(b), "PiSSA: layer %d slot %s: the range-finder product is non-finite", l,
                         lm_slot_peft_name(s));
                *err = b;
                return done(false);
            }
            hs_qr_mgs(Yd, out, q);
            for (size_t i = 0; i < Yd.size(); i++) {
                host[i] = (float) Yd[i];
            }
            ggml_backend_tensor_set(t_q, host.data(), 0, host.size() * sizeof(float));
            // ── C: C = Q^T W  [q, in] ─────────────────────────────────────
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    c   = ggml_mul_mat(ctx, view2(ctx, t_q, out, q), view2(ctx, t_wt, out, in));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, c, view2(ctx, t_c, q, in)));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: stage C graph failed";
                    return done(false);
                }
            }
            // ── D: small SVD on the host, then A0/B0 ──────────────────────
            std::vector<double> Cd((size_t) q * (size_t) in);  // element (k, i) at i*q + k
            host.resize(Cd.size());
            ggml_backend_tensor_get(t_c, host.data(), 0, host.size() * sizeof(float));
            double cmax = 0.0;
            for (size_t i = 0; i < Cd.size(); i++) {
                Cd[i] = (double) host[i];
                cmax  = std::max(cmax, std::fabs(Cd[i]));
            }
            if (!std::isfinite(cmax)) {
                char b[224];
                snprintf(b, sizeof(b), "PiSSA: layer %d slot %s: Q^T W is non-finite (|Y| max %.4g)", l,
                         lm_slot_peft_name(s), ymax);
                *err = b;
                return done(false);
            }
            std::vector<double> G((size_t) q * (size_t) q, 0.0);
            for (int i = 0; i < in; i++) {
                const double * ci = Cd.data() + (size_t) i * (size_t) q;
                for (int k = 0; k < q; k++) {
                    for (int l2 = k; l2 < q; l2++) {
                        G[(size_t) k * (size_t) q + (size_t) l2] += ci[k] * ci[l2];
                    }
                }
            }
            for (int k = 0; k < q; k++) {
                for (int l2 = 0; l2 < k; l2++) {
                    G[(size_t) k * (size_t) q + (size_t) l2] = G[(size_t) l2 * (size_t) q + (size_t) k];
                }
            }
            std::vector<double> evals, evecs;
            hs_jacobi_sym(G, q, evals, evecs);
            std::vector<int> order((size_t) q);
            for (int k = 0; k < q; k++) {
                order[(size_t) k] = k;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return evals[(size_t) a] > evals[(size_t) b]; });

            std::vector<float> A0((size_t) in * (size_t) r), B0((size_t) r * (size_t) out);
            double             captured = 0.0;
            for (int k = 0; k < r; k++) {
                const int      idx = order[(size_t) k];
                const double   lam = std::max(evals[(size_t) idx], 0.0);
                const double   S   = sqrt(lam);
                // Both factors carry sqrt(S_k)/sqrt(s), so B0 A0 = W_topr / s
                // and the graph's s * B0 A0 is W's rank-r truncation exactly.
                const double   sS  = sqrt(S) * inv_sqrt_s;
                const double * uc  = evecs.data() + (size_t) idx * (size_t) q;  // Uc[:, idx]
                captured += lam;
                // V_r[i] = C^T uc / S ; A0[i, k] = V_r[i] * sqrt(S)  (ggml A: [in, r], (i,k) at k*in+i)
                for (int i = 0; i < in; i++) {
                    const double * ci = Cd.data() + (size_t) i * (size_t) q;
                    double         v  = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        v += ci[kk] * uc[kk];
                    }
                    A0[(size_t) k * (size_t) in + (size_t) i] = (S > 0.0) ? (float) (v / S * sS) : 0.0f;
                }
                // U_r[j] = Q uc ; B0[k, j] = sqrt(S) * U_r[j]        (ggml B: [r, out], (k,j) at j*r+k)
                for (int j = 0; j < out; j++) {
                    double u = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        u += Yd[(size_t) kk * (size_t) out + (size_t) j] * uc[kk];
                    }
                    B0[(size_t) j * (size_t) r + (size_t) k] = (float) (sS * u);
                }
            }
            // Rounded through the residual's F16 before upload, so the frozen
            // pair the trainer subtracts is byte-for-byte what the file keeps.
            place(pr, l, s, in, out, A0, B0);

            // A NaN here would silently poison the adapter — A0/B0 are frozen
            // inputs, so the very first forward would be non-finite and the run
            // would die somewhere far from the cause. Refuse at the site.
            if (!std::isfinite(captured) || !std::isfinite((double) w_fro2)) {
                char b[288];
                snprintf(b, sizeof(b),
                         "PiSSA: layer %d slot %s produced a non-finite SVD (||W||_F^2 %.6g, captured %.6g, "
                         "top eigenvalue %.6g, base dtype %s, %dx%d)",
                         l, lm_slot_peft_name(s), (double) w_fro2, captured, evals[(size_t) order[0]],
                         ggml_type_name(w->type), in, out);
                *err = b;
                return done(false);
            }
            const double frac = (w_fro2 > 0.0f) ? captured / (double) w_fro2 : 0.0;
            if (stats) {
                stats->energy_min = std::min(stats->energy_min, frac);
                stats->sites++;
            }
            e_sum += frac;
            n_e++;
            if (!cache_path.empty()) {
                LmPissaCacheSite c;
                c.l = l; c.s = s; c.in = in; c.out = out; c.frac = frac;
                c.A0 = A0;
                c.B0 = B0;
                to_cache.push_back(std::move(c));
            }
        }
    }
    const double e_mean = n_e > 0 ? e_sum / (double) n_e : 0.0;
    if (stats && n_e > 0) {
        stats->energy_mean = e_mean;
    }
    if (stats) {
        stats->seconds = elapsed();
    }
    if (!cache_path.empty()) {
        const bool wrote = lm_pissa_cache_write(cache_path, r, q, iters, L->layer_lo, L->layer_hi, to_cache);
        fprintf(stderr, "[pissa] init: %d sites by SVD in %.1f s (%s %s)\n", n_e, elapsed(),
                wrote ? "cached to" : "cache NOT written:", cache_path.c_str());
    } else {
        fprintf(stderr, "[pissa] init: %d sites by SVD in %.1f s\n", n_e, elapsed());
    }
    write_residual(e_mean, stats ? stats->energy_min : e_mean);
    return done(true);
}

// Both LM trainers size their training scheduler from the training graph, which
// is built long after the adapter bank exists — so the init gets one of its
// own, and the ~200 MB of scratch it holds is returned before anything else is
// allocated.
static bool lm_pissa_init_standalone(LmLora * L, int oversample, int iters, LmPissaStats * stats, std::string * err,
                                     const std::string & cache_dir = "", const std::string & lm_path = "",
                                     bool use_residual = true) {
    if (!L->model) {
        *err = "PiSSA: the adapter bank is not attached to a model";
        return false;
    }
    std::string cache_path;
    if (!cache_dir.empty()) {
        pm_mkdir_p(cache_dir);
        cache_path = lm_pissa_cache_path(cache_dir, lm_path, L->rank, L->rank + std::max(0, oversample),
                                         std::max(0, std::min(4, iters)), L->layer_lo, L->layer_hi);
    }
    // The residual lives beside the base GGUF (pissa-residual.h). Without a base
    // path there is nowhere to look, and the run exports standalone files.
    const std::string residual_path = use_residual ? pissa_residual_path(lm_path, L->rank) : std::string();
    const long long   base_size     = use_residual ? pissa_file_size(lm_path) : 0;
    BackendPair bp;
    bp.backend     = L->model->backend;
    bp.cpu_backend = L->model->cpu_backend;
    bp.has_gpu     = bp.backend != bp.cpu_backend;
    ggml_backend_sched_t sc = backend_sched_new(bp, 2048);
    const bool           ok = lm_pissa_init(L, sc, oversample, iters, stats, err, cache_path, residual_path, base_size);
    ggml_backend_sched_free(sc);
    return ok;
}

// The PiSSA export: a plain rank-2r LoRA on the ORIGINAL base.
//
// TWO CHOICES HERE, both about numbers rather than algebra.
//
// 1. THE FACTORIZATION. The obvious one is the DiT's,
//        s(BA - B0A0) = s [B, -B0] [A; A0],
//    and it is a catastrophic-cancellation trap: |B| and |B0| are both the full
//    SVD magnitude while their difference is the whole adapter, so two nearly
//    equal halves have to cancel to something ~1% of themselves. The MM3 runtime
//    loader stores adapter tensors as f16 (~5e-4 per element), which is applied
//    to the HALVES — the surviving error is then tens of percent OF THE DELTA.
//    Measured at 2.1e-3 against a delta that was itself ~0, i.e. unbounded.
//    The identity
//        BA - B0A0 = (B - B0) A + B0 (A - A0)
//    is exactly as rank-2r, but every term is now the size of the DRIFT rather
//    than the size of the base's singular values, and the two ADD instead of
//    cancelling. Rounding is then 1e-3 of the delta, which is what it should be.
//
// 2. THE SCALE. Baked into the first factor rather than carried as a doubled
//    alpha (the DiT's convention), because alpha is an integer in the config and
//    s = alpha/sqrt(r) under rsLoRA is not. Baking keeps one code path exact for
//    every scale and reduces to a scale of 1 at the default alpha == rank.
static bool lm_export_pissa(const LmLora & L, const Qwen3LMConfig & cfg, const LmExportMeta & meta,
                            const std::string & out_dir, LmExportResult * res, std::string * err,
                            const LmExtraExport * extra = nullptr) {
    const int          r = L.rank;
    std::vector<float> a, b, a0, b0;
    LmPeftOverride     ovr;
    ovr.rank  = 2 * r;
    ovr.alpha = 2 * r;  // loader scale alpha/rank = 1
    ovr.dtype = STW_F16;  // 2026-09-09: half the file; every term is drift-sized (see above)
    ovr.site  = [&](int l, int s, LmPeftFactors * f, std::string * e) -> bool {
        const QwLoraPair & pr = L.layers[l].p[s];
        if (!pr.A || !pr.B || !pr.has_pissa()) {
            e->clear();
            return false;
        }
        const int64_t in  = pr.A->ne[0];
        const int64_t out = pr.B->ne[1];
        a.assign((size_t) in * (size_t) r, 0.0f);
        b.assign((size_t) r * (size_t) out, 0.0f);
        a0.assign(a.size(), 0.0f);
        b0.assign(b.size(), 0.0f);
        ggml_backend_tensor_get(pr.A, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(pr.B, b.data(), 0, b.size() * sizeof(float));
        lm_pissa_tensor_get_f32(pr.A0, a0);
        lm_pissa_tensor_get_f32(pr.B0, b0);

        f->in  = in;
        f->out = out;
        f->rr  = 2 * r;
        // A_cat is ggml [in, 2r]: columns 0..r-1 are A, r..2r-1 are (A - A0).
        f->A.resize(a.size() + a0.size());
        std::copy(a.begin(), a.end(), f->A.begin());
        for (size_t i = 0; i < a0.size(); i++) {
            f->A[a.size() + i] = a[i] - a0[i];
        }
        // B_cat is ggml [2r, out]: per output column j, s * [(B - B0)(j), B0(j)].
        f->B.assign((size_t) 2 * (size_t) r * (size_t) out, 0.0f);
        for (int64_t j = 0; j < out; j++) {
            for (int k = 0; k < r; k++) {
                const size_t src = (size_t) j * (size_t) r + (size_t) k;
                f->B[(size_t) j * (size_t) (2 * r) + (size_t) k] = L.scale * (b[src] - b0[src]);
                f->B[(size_t) j * (size_t) (2 * r) + (size_t) r + (size_t) k] = L.scale * b0[src];
            }
        }
        return true;
    };
    return lm_export_peft(L, cfg, meta, out_dir, res, err, extra, &ovr);
}

// ─── the DELTA export (2026-09-09) ───────────────────────────────────────────
//
// When the init read (or wrote) the residual file, the adapter no longer needs
// to carry the frozen half. The file holds, at rank r:
//
//     lora_A = A - A0                 [r, in]
//     lora_B = s (B - B0)             [out, r]
//
// plus hot_step.param_method = 4, hot_step.pissa.meta and the residual's name
// in adapter_config.json. minimax/mm3-lm-adapter.h rebuilds
//
//     [s(B - B0), sqrt(s) B0_canon] [A0_canon/sqrt(s) + (A - A0); A - A0]
//
// which is the rank-2r pair lm_export_pissa writes, from the two files. Same
// numbers, same runtime cost; 0.7 GB on disk at r128 F16 instead of 2.8 GB.
//
// Not a plain LoRA: an external PEFT loader would apply (B - B0)(A - A0), which
// is nothing. That is what the marker and the metadata key are for.
static bool lm_export_pissa_delta(const LmLora & L, const Qwen3LMConfig & cfg, const LmExportMeta & meta,
                                  const std::string & out_dir, LmExportResult * res, std::string * err,
                                  const LmExtraExport * extra = nullptr) {
    if (L.pissa_residual.empty() || L.pissa_fingerprint.size() != 5) {
        *err = "PiSSA delta export: no residual fingerprint was captured at initialization";
        return false;
    }
    const int          r = L.rank;
    std::vector<float> a, b, a0, b0;
    LmPeftOverride     ovr;
    ovr.rank   = r;
    ovr.alpha  = r;  // loader scale 1; s is baked into lora_B and recorded in the meta
    ovr.marker = 4;
    ovr.dtype  = STW_F16;
    float hi = 0.0f, lo = 0.0f;
    pissa_size_split(L.pissa_base_size, &hi, &lo);
    ovr.pissa_meta = { (float) r, L.scale, (float) L.layer_lo, (float) L.layer_hi, hi, lo };
    ovr.pissa_fingerprint = L.pissa_fingerprint;
    ovr.cfg_extra  = "  \"hot_step_pissa_residual\": \"" + lm_json_escape(L.pissa_residual) + "\",\n";
    ovr.site       = [&](int l, int s, LmPeftFactors * f, std::string * e) -> bool {
        const QwLoraPair & pr = L.layers[l].p[s];
        if (!pr.A || !pr.B || !pr.has_pissa()) {
            e->clear();
            return false;
        }
        const int64_t in  = pr.A->ne[0];
        const int64_t out = pr.B->ne[1];
        a.assign((size_t) in * (size_t) r, 0.0f);
        b.assign((size_t) r * (size_t) out, 0.0f);
        a0.assign(a.size(), 0.0f);
        b0.assign(b.size(), 0.0f);
        ggml_backend_tensor_get(pr.A, a.data(), 0, a.size() * sizeof(float));
        ggml_backend_tensor_get(pr.B, b.data(), 0, b.size() * sizeof(float));
        lm_pissa_tensor_get_f32(pr.A0, a0);
        lm_pissa_tensor_get_f32(pr.B0, b0);
        f->in  = in;
        f->out = out;
        f->rr  = r;
        // Same ggml layouts as the trainer's own tensors, so both are elementwise.
        f->A.resize(a.size());
        for (size_t i = 0; i < a.size(); i++) {
            f->A[i] = a[i] - a0[i];
        }
        f->B.resize(b.size());
        for (size_t i = 0; i < b.size(); i++) {
            f->B[i] = L.scale * (b[i] - b0[i]);
        }
        return true;
    };
    return lm_export_peft(L, cfg, meta, out_dir, res, err, extra, &ovr);
}
