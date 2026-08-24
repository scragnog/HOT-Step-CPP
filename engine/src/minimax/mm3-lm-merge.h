#pragma once
// minimax/mm3-lm-merge.h — merge-mode LM adapters: fold scale·B·A into the
// resident LM weights.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-job.h.
//
// The runtime mode (mm3-lm-adapter.h + mm3_lm_mm) keeps the base weights
// untouched and applies low-rank deltas in-graph — live per-generation dials,
// at a measured +28 % on the LM decode step at r256 (252 adapted modules add
// ~1000 nodes to a 2545-node decode graph; launch overhead on top of the +8 %
// extra weight streaming). Merge mode trades the dials for the step cost:
// W' = W + s·B·A is computed ONCE into the resident weights and the AR loop
// runs the plain base graph at full speed. Changing adapter or any scale dial
// re-merges (in staged mode the LM reloads from disk every generation anyway,
// so "re-merge" is simply "merge after warm"; under keep-loaded a pristine
// reload is forced first via MM3Model::lm_merge_tag).
//
// Mechanics, per adapted module:
//   delta_f32 = mul_mat(cont(transpose(A)), cast(B, F32))   // [in, out] on GPU
//   merged    = cast(W, F32) + s * delta                    // ggml_cast handles
//                                                           // f16 AND quantized
//                                                           // bases (the
//                                                           // quant-cpy-kquant
//                                                           // machinery)
//   f16 base:        cast back to F16 in-graph, read back, write into W
//   quantized base:  read back f32, ggml_quantize_chunk on the host, write raw
//                    quantized bytes into W — i.e. exactly "quantize the merged
//                    weights", the same noise a merged-then-quantized GGUF
//                    would carry. One extra requantization vs the runtime
//                    path; ear-validate per adapter.
//
// Failure contract: a merge that dies part-way leaves the weights in a mixed
// state, so the caller MUST drop LM residency on failure (mm3-job.h does) —
// the tag stays clear and the next generation reloads a pristine base.

#include "mm3-lm-adapter.h"
#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Identity of one merge: adapter path + mtime + every scale dial. Compared
// against MM3Model::lm_merge_tag to decide "already merged" vs "reload first".
static std::string mm3_lm_merge_make_tag(const std::string & path, int64_t mtime, const MM3LmAdapterScales & s) {
    char buf[128];
    snprintf(buf, sizeof(buf), "|%lld|%.6g|%.6g|%.6g|%.6g|%.6g|%.6g", (long long) mtime, (double) s.global,
             (double) s.attn, (double) s.mlp, (double) s.early, (double) s.mid, (double) s.late);
    return path + buf;
}

// The base LM tensor an adapter module targets.
static ggml_tensor * mm3_lm_merge_target(const MM3Model & m, int layer, int module) {
    const MM3LmLayer & b = m.lm.blk[(size_t) layer];
    switch (module) {
        case MM3_LM_ADAPTER_Q:    return b.attn_q;
        case MM3_LM_ADAPTER_K:    return b.attn_k;
        case MM3_LM_ADAPTER_V:    return b.attn_v;
        case MM3_LM_ADAPTER_O:    return b.attn_output;
        case MM3_LM_ADAPTER_GATE: return b.ffn_gate;
        case MM3_LM_ADAPTER_UP:   return b.ffn_up;
        case MM3_LM_ADAPTER_DOWN: return b.ffn_down;
        default:                  return nullptr;
    }
}

// Fold scale·B·A into every adapted module of the resident LM. On success the
// caller records mm3_lm_merge_make_tag(...) in m->lm_merge_tag; on failure it
// must drop LM residency (see the failure contract above).
static bool mm3_lm_merge_apply(MM3Model * m, const MM3LmAdapter * ad, const MM3LmAdapterScales & sc,
                               std::string * err) {
    if (!m->lm_resident) {
        if (err) {
            *err = "the LM is not resident; merge must run after warm";
        }
        return false;
    }
    // A LoKr delta is a Kronecker product, not a low-rank pair, and the loop
    // below skips every module that has no lora_A/lora_B. Without this guard a
    // LoKr adapter merges ZERO modules, reports success, and renders as the
    // base model — which reads as "the adapter does nothing" and sends you
    // debugging the training instead of the mode. Runtime mode applies LoKr
    // properly (mm3_lm_mm -> qwen3_lokr_delta), so say so.
    if (ad && ad->is_lokr) {
        if (err) {
            *err = "this is a LoKr adapter and merge mode only understands LoRA; "
                   "use lm_adapter_mode=runtime (the default), which applies LoKr correctly";
        }
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();

    BackendPair bp = backend_init("MM3-LM-Merge");
    ggml_backend_sched_t sched = backend_sched_new(bp, 64);

    std::vector<float>   f32;
    std::vector<uint8_t> qbuf;
    int                  n_merged  = 0;
    size_t               moved     = 0;
    bool                 ok        = true;

    for (int l = 0; l < MM3_LM_ADAPTER_LAYERS && ok; l++) {
        for (int mod = 0; mod < MM3_LM_ADAPTER_MODULES && ok; mod++) {
            const MM3LmAdapterPair & p = ad->mods[l][mod];
            if (!p.a || !p.b) {
                continue;
            }
            const float s = ad->effective(l, mod, sc);
            if (s == 0.0f) {
                continue;
            }
            ggml_tensor * w = mm3_lm_merge_target(*m, l, mod);
            if (!w) {
                if (err) {
                    *err = "merge target missing at layer " + std::to_string(l);
                }
                ok = false;
                break;
            }
            // A [in, r], B [r, out] must match W [in, out].
            if (p.a->ne[0] != w->ne[0] || p.b->ne[1] != w->ne[1] || p.a->ne[1] != p.b->ne[0]) {
                if (err) {
                    *err = "adapter/base shape mismatch at layer " + std::to_string(l) + " module " +
                           std::to_string(mod);
                }
                ok = false;
                break;
            }
            const bool quantized = ggml_is_quantized(w->type);
            if (quantized && ggml_quantize_requires_imatrix(w->type)) {
                if (err) {
                    *err = std::string("base type ") + ggml_type_name(w->type) +
                           " needs an importance matrix to requantize — merge mode cannot target it; use the "
                           "runtime adapter mode instead";
                }
                ok = false;
                break;
            }

            // Small per-module graph: merged_f32 = cast(W) + s * (Aᵀ · B).
            const size_t         need = ggml_tensor_overhead() * 48 + ggml_graph_overhead_custom(64, false);
            std::vector<uint8_t> gvec(need);
            ggml_init_params     ip  = { need, gvec.data(), /*no_alloc*/ true };
            ggml_context *       ctx = ggml_init(ip);
            if (!ctx) {
                if (err) {
                    *err = "ggml_init failed for the merge graph";
                }
                ok = false;
                break;
            }
            ggml_tensor * a_t    = ggml_cont(ctx, ggml_transpose(ctx, p.a));            // [r, in] f16
            ggml_tensor * b32    = ggml_cast(ctx, p.b, GGML_TYPE_F32);                  // [r, out]
            ggml_tensor * delta  = ggml_mul_mat(ctx, a_t, b32);                         // [in, out] f32
            ggml_tensor * basef  = ggml_cast(ctx, w, GGML_TYPE_F32);                    // [in, out]
            ggml_tensor * merged = ggml_add(ctx, basef, ggml_scale(ctx, delta, s));
            ggml_tensor * outt   = w->type == GGML_TYPE_F16 ? ggml_cast(ctx, merged, GGML_TYPE_F16) : merged;
            ggml_set_output(outt);

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(gf, outt);
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gf) ||
                ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
                if (err) {
                    *err = "merge graph compute failed at layer " + std::to_string(l) + " (out of VRAM?)";
                }
                ggml_free(ctx);
                ok = false;
                break;
            }

            const int64_t ne0 = w->ne[0], ne1 = w->ne[1];
            if (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32) {
                // Same type as the graph output: read back and write straight in.
                const size_t bytes = ggml_nbytes(w);
                qbuf.resize(bytes);
                ggml_backend_tensor_get(outt, qbuf.data(), 0, bytes);
                ggml_backend_tensor_set(w, qbuf.data(), 0, bytes);
                moved += bytes * 2;
            } else {
                // Quantized base: f32 down, requantize on the host, raw blocks up.
                f32.resize((size_t) (ne0 * ne1));
                ggml_backend_tensor_get(outt, f32.data(), 0, f32.size() * sizeof(float));
                qbuf.resize(ggml_nbytes(w));
                const size_t written = ggml_quantize_chunk(w->type, f32.data(), qbuf.data(), 0, ne1, ne0, nullptr);
                if (written != ggml_nbytes(w)) {
                    if (err) {
                        *err = "requantize size mismatch at layer " + std::to_string(l) + " (" +
                               std::to_string((long long) written) + " vs " +
                               std::to_string((long long) ggml_nbytes(w)) + ")";
                    }
                    ggml_free(ctx);
                    ok = false;
                    break;
                }
                ggml_backend_tensor_set(w, qbuf.data(), 0, qbuf.size());
                moved += f32.size() * sizeof(float) + qbuf.size();
            }
            ggml_free(ctx);
            n_merged++;
        }
    }

    ggml_backend_sched_free(sched);
    backend_release(bp.backend, bp.cpu_backend);

    if (ok) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[MM3] LM adapter MERGED: %d modules into %s base, %.1f GB moved, %.0f ms\n", n_merged,
                ggml_type_name(m->lm.blk[0].attn_q ? m->lm.blk[0].attn_q->type : GGML_TYPE_F16),
                (double) moved / 1073741824.0, ms);
    }
    return ok;
}
