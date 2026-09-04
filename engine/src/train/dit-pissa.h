// dit-pissa.h — PiSSA initialisation for the DiT LoRA (Meng et al. 2024).
//
// A plain LoRA starts from A ~ N(0, 1/sqrt(in)), B = 0 and has to discover the
// directions that matter from scratch. PiSSA starts the pair on the base
// weight's own top-r singular directions (A = V_r sqrt(S_r), B = sqrt(S_r) U_r^T)
// and moves that energy OUT of the base (W_res = W - s B A) so the first forward
// is still exactly W. The optimizer then fine-tunes the principal subspace
// directly, which is where most of a weight's capacity lives.
//
// The SVD is a randomized one with the two big products on the GPU (a host SVD
// of 352 matrices at this size would take hours):
//
//   stage A (GPU)  Y = W (W^T (W Omega))...   q = r + oversample columns, `iters`
//                  power iterations, Omega ~ N(0,1)
//   stage B (host) Q = qr(Y)                  modified Gram-Schmidt in double
//   stage C (GPU)  C = Q^T W                  [q, in]
//   stage D (host) C C^T = Uc diag(lambda) Uc^T (Jacobi), S = sqrt(lambda),
//                  U_r = Q Uc_r, V_r = C^T Uc_r / S_r
//   stage E (GPU)  W_res = W - s (B A)        written back into the mirror
//
// Host stages run one thread per site. The per-layer captured-energy line
// (||B A||_F^2 / ||W||_F^2) is the correctness diagnostic: a working SVD
// captures far more than a random rank-r subspace's r/min(in,out).
//
// Export is a standard PEFT LoRA of rank 2r on the ORIGINAL base:
//   s (B A - B0 A0)  =  s [B, -B0] [A; A0]      (dit-adapter.h exportDir)
// so the merge and runtime paths need no PiSSA knowledge. A0/B0 are kept on
// disk (pissa_init.L<l>.safetensors in the run dir), not in RAM or VRAM.
#pragma once

#include "safetensors.h"
#include "train/dit-adapter.h"
#include "train/st-write.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

// Modified Gram-Schmidt on a column-major out x q matrix (column j at j*out).
// Two passes: the second removes the loss of orthogonality the first leaves
// when the power iteration has made the columns nearly parallel.
static void dit_pissa_qr(std::vector<double> & Y, int out, int q) {
    for (int pass = 0; pass < 2; pass++) {
        for (int j = 0; j < q; j++) {
            double * cj = Y.data() + (size_t) j * (size_t) out;
            for (int i = 0; i < j; i++) {
                const double * ci  = Y.data() + (size_t) i * (size_t) out;
                double         dot = 0.0;
                for (int k = 0; k < out; k++) {
                    dot += ci[k] * cj[k];
                }
                for (int k = 0; k < out; k++) {
                    cj[k] -= dot * ci[k];
                }
            }
            double nrm = 0.0;
            for (int k = 0; k < out; k++) {
                nrm += cj[k] * cj[k];
            }
            nrm = sqrt(nrm);
            if (nrm < 1e-300) {
                // Degenerate column: replace with a unit vector on an axis that
                // no earlier column occupies (only reachable for rank-deficient W).
                for (int k = 0; k < out; k++) {
                    cj[k] = 0.0;
                }
                cj[j % out] = 1.0;
                continue;
            }
            for (int k = 0; k < out; k++) {
                cj[k] /= nrm;
            }
        }
    }
}

// Cyclic Jacobi on a symmetric q x q matrix G (row-major, overwritten).
// evecs is column-major (eigenvector k at k*q); evals[k] pairs with it.
static void dit_pissa_jacobi(std::vector<double> & G, int q, std::vector<double> & evals,
                             std::vector<double> & evecs) {
    evecs.assign((size_t) q * (size_t) q, 0.0);
    for (int i = 0; i < q; i++) {
        evecs[(size_t) i * (size_t) q + (size_t) i] = 1.0;
    }
    auto at = [&](int r, int c) -> double & { return G[(size_t) r * (size_t) q + (size_t) c]; };
    for (int sweep = 0; sweep < 60; sweep++) {
        double off = 0.0;
        for (int r = 0; r < q; r++) {
            for (int c = r + 1; c < q; c++) {
                off += at(r, c) * at(r, c);
            }
        }
        if (off < 1e-22) {
            break;
        }
        for (int p = 0; p < q; p++) {
            for (int r = p + 1; r < q; r++) {
                const double apq = at(p, r);
                if (fabs(apq) < 1e-300) {
                    continue;
                }
                const double theta = (at(r, r) - at(p, p)) / (2.0 * apq);
                const double t     = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                const double cs    = 1.0 / sqrt(t * t + 1.0);
                const double sn    = t * cs;
                for (int k = 0; k < q; k++) {
                    const double gkp = at(k, p), gkr = at(k, r);
                    at(k, p) = cs * gkp - sn * gkr;
                    at(k, r) = sn * gkp + cs * gkr;
                }
                for (int k = 0; k < q; k++) {
                    const double gpk = at(p, k), grk = at(r, k);
                    at(p, k) = cs * gpk - sn * grk;
                    at(r, k) = sn * gpk + cs * grk;
                }
                for (int k = 0; k < q; k++) {
                    double & vkp = evecs[(size_t) p * (size_t) q + (size_t) k];
                    double & vkr = evecs[(size_t) r * (size_t) q + (size_t) k];
                    const double a = vkp, b = vkr;
                    vkp = cs * a - sn * b;
                    vkr = sn * a + cs * b;
                }
            }
        }
    }
    evals.resize((size_t) q);
    for (int i = 0; i < q; i++) {
        evals[(size_t) i] = at(i, i);
    }
}

struct DitPissaStats {
    double energy_min = 1.0, energy_mean = 0.0;  // captured ||BA||^2/||W||^2 over sites
    int    sites      = 0;
};

// Overwrites ad->A/B with the PiSSA factors, subtracts s*BA from the mirror's
// base weights, writes <out_dir>/pissa_init.L<l>.safetensors with A0/B0.
static bool dit_pissa_init(DitAdapterLora * ad, ggml_backend_t backend, ggml_backend_sched_t sched,
                           const std::string & out_dir, int oversample, int iters, DitPissaStats * stats,
                           std::string * err) {
    DiTGGML * m = ad->model;
    if (!m || ad->lo >= ad->hi) {
        *err = "PiSSA: adapter not initialised";
        return false;
    }
    const int r = ad->rank;
    const int q = r + std::max(0, oversample);
    iters       = std::max(0, std::min(4, iters));

    // Scratch sized by the largest site.
    int64_t in_max = 0, out_max = 0;
    for (int s = 0; s < ad->n_sites; s++) {
        ggml_tensor * w = dit_site_weight(&m->layers[ad->lo], s);
        if (!w) {
            *err = "PiSSA: site without a weight";
            return false;
        }
        in_max  = std::max(in_max, w->ne[0]);
        out_max = std::max(out_max, w->ne[1]);
    }
    if (q > std::min(in_max, out_max)) {
        *err = "PiSSA: rank + oversample exceeds the smallest weight dimension";
        return false;
    }

    ggml_context * sctx = nullptr;
    {
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        sctx                = ggml_init(ip);
    }
    ggml_tensor * t_omega = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, in_max, q);
    ggml_tensor * t_y     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_q     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, q);
    ggml_tensor * t_c     = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, q, in_max);
    ggml_tensor * t_wres  = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, in_max, out_max);
    ggml_tensor * t_en    = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 2);
    for (ggml_tensor * t : { t_omega, t_y, t_q, t_c, t_wres, t_en }) {
        ggml_set_input(t);
    }
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
    if (!sbuf) {
        *err = "PiSSA: scratch allocation failed";
        ggml_free(sctx);
        return false;
    }
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
        return ggml_view_2d(ctx, t, ne0, ne1, ne0 * sizeof(float), 0);
    };

    LmRng rng;
    lm_rng_seed(&rng, 0x9155a000ull ^ (uint64_t) ad->rank);
    std::vector<float> host;
    double             e_sum = 0.0;
    int                n_e   = 0;

    for (int l = ad->lo; l < ad->hi; l++) {
        std::vector<STWTensor>          tensors;
        std::vector<std::vector<float>> store;
        for (int s = 0; s < ad->n_sites; s++) {
            DitLoraPair & pr  = ad->layers[(size_t) (l - ad->lo)][(size_t) s];
            ggml_tensor * w   = dit_site_weight(&m->layers[l], s);
            const int     in  = (int) w->ne[0];
            const int     out = (int) w->ne[1];

            // ── A: Y = (W W^T)^iters W Omega ──────────────────────────────
            host.assign((size_t) in * (size_t) q, 0.0f);
            lm_rng_fill_normal(&rng, host, 1.0f);
            ggml_backend_tensor_set(t_omega, host.data(), 0, host.size() * sizeof(float));
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    wf  = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
                ggml_tensor *    wt  = ggml_cont(ctx, ggml_transpose(ctx, wf));  // [out, in]
                ggml_tensor *    y   = ggml_mul_mat(ctx, wf, view2(ctx, t_omega, in, q));  // [out, q]
                for (int it = 0; it < iters; it++) {
                    ggml_tensor * z = ggml_mul_mat(ctx, wt, y);  // [in, q]
                    y               = ggml_mul_mat(ctx, wf, z);  // [out, q]
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
            for (size_t i = 0; i < Yd.size(); i++) {
                Yd[i] = (double) host[i];
            }
            dit_pissa_qr(Yd, out, q);
            for (size_t i = 0; i < Yd.size(); i++) {
                host[i] = (float) Yd[i];
            }
            ggml_backend_tensor_set(t_q, host.data(), 0, host.size() * sizeof(float));
            // ── C: C = Q^T W  [q, in] ─────────────────────────────────────
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    wf  = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
                ggml_tensor *    wt  = ggml_cont(ctx, ggml_transpose(ctx, wf));      // [out, in]
                ggml_tensor *    c   = ggml_mul_mat(ctx, view2(ctx, t_q, out, q), wt);  // [q, in]
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
            for (size_t i = 0; i < Cd.size(); i++) {
                Cd[i] = (double) host[i];
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
            dit_pissa_jacobi(G, q, evals, evecs);
            std::vector<int> order((size_t) q);
            for (int k = 0; k < q; k++) {
                order[(size_t) k] = k;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return evals[(size_t) a] > evals[(size_t) b]; });

            std::vector<float> A0((size_t) in * (size_t) r), B0((size_t) r * (size_t) out);
            double             captured = 0.0;
            for (int k = 0; k < r; k++) {
                const int      idx  = order[(size_t) k];
                const double   lam  = std::max(evals[(size_t) idx], 0.0);
                const double   S    = sqrt(lam);
                const double   sS   = sqrt(S);
                const double * uc   = evecs.data() + (size_t) idx * (size_t) q;  // Uc[:, idx]
                captured += lam;
                // V_r[i] = C^T uc / S ; A0[i, k] = V_r[i] * sqrt(S)   (ggml A: (in, r), (i, k) at k*in + i)
                for (int i = 0; i < in; i++) {
                    const double * ci = Cd.data() + (size_t) i * (size_t) q;
                    double         v  = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        v += ci[kk] * uc[kk];
                    }
                    A0[(size_t) k * (size_t) in + (size_t) i] = (S > 0.0) ? (float) (v / S * sS) : 0.0f;
                }
                // U_r[j] = Q uc ; B0[k, j] = sqrt(S) * U_r[j]         (ggml B: (r, out), (k, j) at j*r + k)
                for (int j = 0; j < out; j++) {
                    double u = 0.0;
                    for (int kk = 0; kk < q; kk++) {
                        u += Yd[(size_t) kk * (size_t) out + (size_t) j] * uc[kk];
                    }
                    B0[(size_t) j * (size_t) r + (size_t) k] = (float) (sS * u);
                }
            }
            ggml_backend_tensor_set(pr.A, A0.data(), 0, A0.size() * sizeof(float));
            ggml_backend_tensor_set(pr.B, B0.data(), 0, B0.size() * sizeof(float));

            // ── E: W_res = W - s (B0 A0), energy diagnostic, write back ───
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 256, false);
                ggml_tensor *    wf  = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
                ggml_tensor * delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, pr.A)), pr.B);  // [in, out]
                ggml_tensor * wres  = ggml_sub(ctx, wf, ggml_scale(ctx, delta, ad->scale));
                ggml_tensor * e_d   = ggml_sum(ctx, ggml_sqr(ctx, ggml_scale(ctx, delta, ad->scale)));
                ggml_tensor * e_w   = ggml_sum(ctx, ggml_sqr(ctx, wf));
                ggml_tensor * e2    = ggml_concat(ctx, e_d, e_w, 0);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, wres, view2(ctx, t_wres, in, out)));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, e2, t_en));
                if (!compute(gf, ctx)) {
                    *err = "PiSSA: stage E graph failed";
                    return done(false);
                }
            }
            float en[2] = { 0.0f, 0.0f };
            ggml_backend_tensor_get(t_en, en, 0, sizeof(en));
            const double frac = (en[1] > 0.0f) ? (double) en[0] / (double) en[1] : 0.0;
            (void) captured;
            if (stats) {
                stats->energy_min = std::min(stats->energy_min, frac);
                stats->sites++;
            }
            e_sum += frac;
            n_e++;

            host.resize((size_t) in * (size_t) out);
            ggml_backend_tensor_get(t_wres, host.data(), 0, host.size() * sizeof(float));
            if (w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(w, host.data(), 0, host.size() * sizeof(float));
            } else if (w->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> hb(host.size());
                ggml_fp32_to_bf16_row(host.data(), hb.data(), (int64_t) host.size());
                ggml_backend_tensor_set(w, hb.data(), 0, hb.size() * sizeof(ggml_bf16_t));
            } else if (w->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> hh(host.size());
                ggml_fp32_to_fp16_row(host.data(), hh.data(), (int64_t) host.size());
                ggml_backend_tensor_set(w, hh.data(), 0, hh.size() * sizeof(ggml_fp16_t));
            } else {
                *err = std::string("PiSSA: mirror weight type ") + ggml_type_name(w->type) +
                       " cannot take the residual (use an F32/BF16/F16 mirror)";
                return done(false);
            }

            // A0/B0 to disk for the export (row-major shapes like the PEFT layout).
            char nm[160];
            store.push_back(std::move(A0));
            snprintf(nm, sizeof(nm), "L%d.%s.pissa_A0", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) r, (int64_t) in }, nullptr });
            store.push_back(std::move(B0));
            snprintf(nm, sizeof(nm), "L%d.%s.pissa_B0", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) out, (int64_t) r }, nullptr });
        }
        for (size_t i = 0; i < tensors.size(); i++) {
            tensors[i].data = store[i].data();
        }
        std::vector<std::pair<std::string, std::string>> md;
        md.push_back({ "format", "pt" });
        md.push_back({ "hot_step_pissa", "v1" });
        char fn[64];
        snprintf(fn, sizeof(fn), "pissa_init.L%d.safetensors", l);
        const std::string path = out_dir + "/" + fn;
        if (!st_write_file(path.c_str(), tensors, md, STW_F32)) {
            *err = "PiSSA: cannot write " + path;
            return done(false);
        }
    }
    if (stats && n_e > 0) {
        stats->energy_mean = e_sum / (double) n_e;
    }
    return done(true);
}

// Export a PiSSA run as a plain PEFT LoRA of rank 2r on the ORIGINAL base:
//   s (B A - B0 A0) = s [B, -B0] [A; A0]
// alpha is doubled with the rank so the loader's alpha/r reproduces s. A0/B0
// come from the init files dit_pissa_init wrote next to the run.
static bool dit_pissa_export(const DitAdapterLora & ad, const char * dir, const DitExportMeta & meta,
                             DitExportResult * res, std::string * err) {
    const std::string d(dir);
    if (!pm_mkdir_p(d)) {
        *err = "cannot create " + d;
        return false;
    }
    if (ad.pissa_dir.empty()) {
        *err = "PiSSA export: no init directory recorded";
        return false;
    }
    if (!DitAdapterLora::dit_write_adapter_config(d, 2 * ad.rank, (int) (2.0f * ad.alpha + 0.5f), ad.n_sites == DIT_NSITES,
                                                  meta.base_model_path, false, ad.rslora, "LORA")) {
        *err = "cannot write adapter_config.json in " + d;
        return false;
    }
    std::vector<STWTensor>          tensors;
    std::vector<std::vector<float>> store;
    const int                       r = ad.rank;
    std::vector<float>              cur;
    for (int l = ad.lo; l < ad.hi; l++) {
        char fn[64];
        snprintf(fn, sizeof(fn), "pissa_init.L%d.safetensors", l);
        const std::string ipath = lm_join(ad.pissa_dir, fn);
        STFile            st;
        if (!st_open(&st, ipath.c_str())) {
            *err = "PiSSA export: cannot open " + ipath;
            return false;
        }
        for (int s = 0; s < ad.n_sites; s++) {
            const DitLoraPair & pr = ad.layers[(size_t) (l - ad.lo)][(size_t) s];
            if (!pr.A || !pr.B) {
                continue;
            }
            const int in  = (int) pr.A->ne[0];
            const int out = (int) pr.B->ne[1];
            char      nm[208];
            snprintf(nm, sizeof(nm), "L%d.%s.pissa_A0", l, dit_site_peft(s));
            const STEntry * ea = st_find(st, nm);
            snprintf(nm, sizeof(nm), "L%d.%s.pissa_B0", l, dit_site_peft(s));
            const STEntry * eb = st_find(st, nm);
            if (!ea || !eb || ea->dtype != "F32" || eb->dtype != "F32") {
                *err = std::string("PiSSA export: init factors missing for layer ") + std::to_string(l) + " site " +
                       dit_site_peft(s);
                st_close(&st);
                return false;
            }
            const float * a0 = (const float *) st_data(st, *ea);  // row-major (r, in) == ggml (in, r)
            const float * b0 = (const float *) st_data(st, *eb);  // row-major (out, r) == ggml (r, out)

            // A_cat: rows 0..r-1 trained A, rows r..2r-1 A0. Row-major (2r, in).
            cur.assign((size_t) in * (size_t) r, 0.0f);
            ggml_backend_tensor_get(pr.A, cur.data(), 0, cur.size() * sizeof(float));
            std::vector<float> acat((size_t) 2 * (size_t) r * (size_t) in);
            std::copy(cur.begin(), cur.end(), acat.begin());
            std::copy(a0, a0 + (size_t) r * (size_t) in, acat.begin() + (ptrdiff_t) ((size_t) r * (size_t) in));
            store.push_back(std::move(acat));
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_A.weight", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) 2 * r, (int64_t) in }, nullptr });

            // B_cat: per output row j, [B[j, 0..r), -B0[j, 0..r)]. Row-major (out, 2r).
            cur.assign((size_t) r * (size_t) out, 0.0f);
            ggml_backend_tensor_get(pr.B, cur.data(), 0, cur.size() * sizeof(float));
            std::vector<float> bcat((size_t) out * (size_t) 2 * (size_t) r);
            for (int j = 0; j < out; j++) {
                for (int k = 0; k < r; k++) {
                    bcat[(size_t) j * (size_t) (2 * r) + (size_t) k]            = cur[(size_t) j * (size_t) r + (size_t) k];
                    bcat[(size_t) j * (size_t) (2 * r) + (size_t) r + (size_t) k] = -b0[(size_t) j * (size_t) r + (size_t) k];
                }
            }
            store.push_back(std::move(bcat));
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_B.weight", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) out, (int64_t) 2 * r }, nullptr });
        }
        st_close(&st);
    }
    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].data = store[i].data();
    }
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "producer", meta.producer });
    md.push_back({ "hot_step_dit_trainer", "v1" });
    md.push_back({ "hot_step_pissa", "rank-2r LoRA on the original base: s(BA - B0A0)" });
    if (!meta.trigger.empty()) {
        md.push_back({ "hot_step_trigger", meta.trigger });
        md.push_back({ "hot_step_trigger_position",
                       meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
        md.push_back({ "modelspec.trigger_phrase", meta.trigger });
    }
    // F32, not BF16: the two halves [B, -B0][A; A0] are large and nearly
    // cancel (each ~0.7 |W| at rank 128 on this DiT), so BF16's 3.9e-3 relative
    // precision leaves a noise residual of the same order as the adapter's own
    // delta. The blind test's "strangely louder" PiSSA render (2026-09-04) is
    // the suspected symptom. Twice the file, exact merge.
    const std::string sf = lm_join(d, "adapter_model.safetensors");
    if (!st_write_file(sf.c_str(), tensors, md, STW_F32)) {
        *err = "cannot write " + sf;
        return false;
    }
    if (res) {
        long long bytes = 0;
        pm_stat_file(sf, &bytes, NULL);
        res->tensors = (int) tensors.size();
        res->bytes   = bytes;
    }
    return true;
}
