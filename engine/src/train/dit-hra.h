// dit-hra.h — Householder Reflection Adaptation (Yuan et al. 2024) for the DiT.
//
// Orthogonal fine-tuning without a Cayley transform: y = W (R x) with
// R = H_{r-1} ... H_0, H_k = I - 2 v_k v_k^T / ||v_k||^2. Each reflection is a
// rank-1 update on the activations, so the whole thing is matmuls — the
// reason it is buildable in ggml where OFT/BOFT (matrix inverse per step) are
// not. The vectors live in DitLoraPair::A as [in, r] (dit-adapter.h, `hra`);
// column pairs start equal, so R = I and step 0 is exactly the base.
//
// Export needs no new loader: R - I has rank <= r (each reflection only moves
// the span of its vector), so with Q an orthonormal basis of span(v_0..v_{r-1})
//   W (R - I) = W (R - I) Q Q^T = (W U) Q^T,  U = (R - I) Q
// is an exact rank-r LoRA: A = Q^T, B = W U, alpha = r (scale 1). W U runs on
// the GPU through the adapter's scheduler handle. The raw vectors go to
// hra_vectors.safetensors beside it, which is what a resume reads.
#pragma once

#include "safetensors.h"
#include "train/dit-adapter.h"
#include "train/dit-pissa.h"  // dit_pissa_qr
#include "train/st-write.h"

#include <string>
#include <vector>

// Trained parameters: one [in, r] block per site.
static size_t dit_hra_expected_params(const DiTGGMLConfig & c, int lo, int hi, int r, bool target_mlp) {
    const int64_t H = c.hidden_size, Q = (int64_t) c.n_heads * c.head_dim, F = c.intermediate_size;
    int64_t       per_layer = 0;
    per_layer += 3 * H * r + Q * r;  // sa q,k,v (in=H), o (in=Q)
    per_layer += 3 * H * r + Q * r;  // ca
    if (target_mlp) {
        per_layer += 2 * H * r + F * r;  // gate, up (in=H), down (in=F)
    }
    return (size_t) per_layer * (size_t) std::max(0, hi - lo);
}

// Apply the site's reflections, in graph order, to a column-major set of
// `n` vectors of length `in` (vector j at j*in). V is column-major [in, r].
static void dit_hra_reflect(const std::vector<float> & V, int in, int r, std::vector<double> & X, int n) {
    for (int k = 0; k < r; k++) {
        const float * v  = V.data() + (size_t) k * (size_t) in;
        double        n2 = 0.0;
        for (int i = 0; i < in; i++) {
            n2 += (double) v[i] * (double) v[i];
        }
        if (n2 <= 0.0) {
            continue;
        }
        for (int j = 0; j < n; j++) {
            double * x   = X.data() + (size_t) j * (size_t) in;
            double   dot = 0.0;
            for (int i = 0; i < in; i++) {
                dot += (double) v[i] * x[i];
            }
            const double f = 2.0 * dot / n2;
            for (int i = 0; i < in; i++) {
                x[i] -= f * (double) v[i];
            }
        }
    }
}

static bool dit_hra_export(const DitAdapterLora & ad, const char * dir, const DitExportMeta & meta,
                           DitExportResult * res, std::string * err) {
    const std::string d(dir);
    if (!pm_mkdir_p(d)) {
        *err = "cannot create " + d;
        return false;
    }
    if (!ad.sched || !ad.model) {
        *err = "HRA export: no scheduler handle (set lora.sched after init)";
        return false;
    }
    const int r = ad.rank;
    if (!DitAdapterLora::dit_write_adapter_config(d, r, r, ad.n_sites == DIT_NSITES, meta.base_model_path, false, false,
                                                  "LORA")) {
        *err = "cannot write adapter_config.json in " + d;
        return false;
    }

    // GPU scratch for W U: U [in_max, r] in, P [out_max, r] out.
    int64_t in_max = 0, out_max = 0;
    for (int s = 0; s < ad.n_sites; s++) {
        ggml_tensor * w = dit_site_weight(&ad.model->layers[ad.lo], s);
        in_max          = std::max(in_max, w->ne[0]);
        out_max         = std::max(out_max, w->ne[1]);
    }
    ggml_context * sctx = nullptr;
    {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
        sctx                = ggml_init(ip);
    }
    ggml_tensor * t_u = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, in_max, r);
    ggml_tensor * t_p = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, out_max, r);
    ggml_set_input(t_u);
    ggml_set_input(t_p);
    ggml_backend_buffer_t sbuf = ggml_backend_alloc_ctx_tensors(sctx, ad.model ? ggml_backend_sched_get_backend(ad.sched, 0) : nullptr);
    if (!sbuf) {
        *err = "HRA export: scratch allocation failed";
        ggml_free(sctx);
        return false;
    }
    std::vector<uint8_t> arena(ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false));

    std::vector<STWTensor>          tensors, vtensors;
    std::vector<std::vector<float>> store, vstore;
    bool                            ok = true;
    for (int l = ad.lo; l < ad.hi && ok; l++) {
        for (int s = 0; s < ad.n_sites && ok; s++) {
            const DitLoraPair & pr = ad.layers[(size_t) (l - ad.lo)][(size_t) s];
            if (!pr.A) {
                continue;
            }
            ggml_tensor * w   = dit_site_weight(&ad.model->layers[l], s);
            const int     in  = (int) pr.A->ne[0];
            const int     out = (int) w->ne[1];

            std::vector<float> V((size_t) in * (size_t) r);
            ggml_backend_tensor_get(pr.A, V.data(), 0, V.size() * sizeof(float));

            // Q = orthonormal basis of span(v), U = (R - I) Q, both column-major.
            std::vector<double> Qd(V.begin(), V.end());
            dit_pissa_qr(Qd, in, r);
            std::vector<double> Ud(Qd);
            dit_hra_reflect(V, in, r, Ud, r);
            for (size_t i = 0; i < Ud.size(); i++) {
                Ud[i] -= Qd[i];
            }

            // B = W U on the GPU: mul_mat(W [in,out], U [in,r]) -> [out, r].
            std::vector<float> Uf(Ud.begin(), Ud.end());
            ggml_backend_tensor_set(t_u, Uf.data(), 0, Uf.size() * sizeof(float));
            {
                ggml_init_params ip  = { arena.size(), arena.data(), true };
                ggml_context *   ctx = ggml_init(ip);
                ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 64, false);
                ggml_tensor *    wf  = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
                ggml_tensor *    uv  = ggml_view_2d(ctx, t_u, in, r, (size_t) in * sizeof(float), 0);
                ggml_tensor *    p   = ggml_mul_mat(ctx, wf, uv);
                ggml_build_forward_expand(gf, ggml_cpy(ctx, p, ggml_view_2d(ctx, t_p, out, r, (size_t) out * sizeof(float), 0)));
                ggml_backend_sched_reset(ad.sched);
                ok = ggml_backend_sched_graph_compute(ad.sched, gf) == GGML_STATUS_SUCCESS;
                ggml_free(ctx);
                if (!ok) {
                    *err = "HRA export: W U graph failed";
                    break;
                }
            }
            std::vector<float> P((size_t) out * (size_t) r);  // element (j, k) at k*out + j
            ggml_backend_tensor_get(t_p, P.data(), 0, P.size() * sizeof(float));

            char nm[208];
            // A row-major (r, in): row k = Q column k.
            std::vector<float> A((size_t) r * (size_t) in);
            for (size_t i = 0; i < A.size(); i++) {
                A[i] = (float) Qd[i];
            }
            store.push_back(std::move(A));
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_A.weight", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) r, (int64_t) in }, nullptr });
            // B row-major (out, r): element (j, k).
            std::vector<float> B((size_t) out * (size_t) r);
            for (int j = 0; j < out; j++) {
                for (int k = 0; k < r; k++) {
                    B[(size_t) j * (size_t) r + (size_t) k] = P[(size_t) k * (size_t) out + (size_t) j];
                }
            }
            store.push_back(std::move(B));
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_B.weight", l, dit_site_peft(s));
            tensors.push_back({ nm, { (int64_t) out, (int64_t) r }, nullptr });

            // Raw vectors for the resume: row-major (r, in) == ggml (in, r).
            vstore.push_back(std::move(V));
            snprintf(nm, sizeof(nm), "L%d.%s.hra_v", l, dit_site_peft(s));
            vtensors.push_back({ nm, { (int64_t) r, (int64_t) in }, nullptr });
        }
    }
    ggml_backend_buffer_free(sbuf);
    ggml_free(sctx);
    if (!ok) {
        return false;
    }
    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].data = store[i].data();
    }
    for (size_t i = 0; i < vtensors.size(); i++) {
        vtensors[i].data = vstore[i].data();
    }
    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "producer", meta.producer });
    md.push_back({ "hot_step_dit_trainer", "v1" });
    md.push_back({ "hot_step_hra", "W(R-I) of r Householder reflections as an exact rank-r LoRA" });
    if (!meta.trigger.empty()) {
        md.push_back({ "hot_step_trigger", meta.trigger });
        md.push_back({ "hot_step_trigger_position",
                       meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
        md.push_back({ "modelspec.trigger_phrase", meta.trigger });
    }
    const std::string sf = lm_join(d, "adapter_model.safetensors");
    if (!st_write_file(sf.c_str(), tensors, md, STW_BF16)) {
        *err = "cannot write " + sf;
        return false;
    }
    std::vector<std::pair<std::string, std::string>> vmd;
    vmd.push_back({ "format", "pt" });
    vmd.push_back({ "hot_step_hra_vectors", "v1" });
    const std::string vf = lm_join(d, "hra_vectors.safetensors");
    if (!st_write_file(vf.c_str(), vtensors, vmd, STW_F32)) {
        *err = "cannot write " + vf;
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
