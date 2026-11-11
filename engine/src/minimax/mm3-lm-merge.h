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
//
// ── Parameterizations beyond plain LoRA (2026-09-05) ────────────────────────
//
// Three variations ride the same per-module graph, chosen off what the adapter
// carries rather than off a flag:
//
//   LoHa  delta = (A1 B1) (.) (A2 B2)          hada_w* keys present
//   HiRA  W' = W + W (.) (s*BA)                peft_type HIRA
//   DoRA  W' = m * (W + s*BA) / ||.||_col      lora_magnitude_vector present
//
// LoHa and HiRA are MERGE-ONLY: neither delta is low-rank, so a runtime apply
// would materialise a full [in, out] tensor per module per token. The runtime
// path refuses them by name (mm3-job.h / mm3-server.h), which is the same split
// the DiT already makes between adapter-runtime.h and adapter-merge.h. DoRA
// works in both — the runtime computes its denominator once at load
// (mm3-lm-dora.h) because W, A and B are frozen there.

#include "mm3-lm-adapter.h"
#include "mm3-model.h"

#include "backend.h"
#include "ggml.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <vector>

// Identity of one merge: adapter path + mtime + every scale dial. Compared
// against MM3Model::lm_merge_tag to decide "already merged" vs "reload first".
static bool mm3_lm_merge_device_enabled(bool requested_gpu = true) {
    const char * value = std::getenv("MM3_MERGE_DEVICE");
    // The request defaults to GPU. Retain process-level CPU/verification
    // overrides for diagnostics; an explicit CPU request always stays CPU.
    return requested_gpu && (!value || (std::strcmp(value, "0") != 0 && std::strcmp(value, "verify") != 0));
}

// GPU preference can resolve to host requantization on another backend.
// Resolve the device even before model loading so cold saved-plan lookup and
// the later save use the same identity. This temporary reference owns no
// model allocations and is released before returning.
static std::string mm3_lm_merge_backend_identity() {
    BackendPair bp = backend_init("MM3-Merge-Key");
    const ggml_backend_dev_t dev = ggml_backend_get_device(bp.backend);
    std::string identity = ggml_backend_name(bp.backend);
    if (dev) {
        identity += "|";
        identity += ggml_backend_dev_name(dev);
        identity += "|";
        identity += ggml_backend_dev_description(dev);
    }
    backend_release(bp.backend, bp.cpu_backend);
    return identity;
}

static std::string mm3_lm_merge_make_tag(const std::string & path, int64_t mtime, const MM3LmAdapterScales & s,
                                       bool requested_gpu) {
    char buf[128];
    snprintf(buf, sizeof(buf), "|%lld|%.6g|%.6g|%.6g|%.6g|%.6g|%.6g", (long long) mtime, (double) s.global,
             (double) s.attn, (double) s.mlp, (double) s.early, (double) s.mid, (double) s.late);
    return path + buf + (mm3_lm_merge_device_enabled(requested_gpu) ? "|device-experimental" : "|host-reference");
}

// Whether a merged adapter can still fade (MM3ArOptions::lm_adapter_merged):
// the graph takes the delta back out as (mult - 1)*s*BA, which is exact only
// for a plain low-rank pair (PiSSA delta files load as one). DoRA rescales by
// a norm of the merged weight, LoHa and HiRA have no runtime form, and LoKr
// does not merge at all.
static bool mm3_lm_merge_fade_ok(const MM3LmAdapter * ad) {
    return ad && !ad->is_lokr && !ad->is_loha && !ad->is_hira && ad->dora_n == 0;
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
static size_t mm3_lm_merge_quantize(ggml_type type, const float * src, void * dst, int64_t rows,
                                    int64_t cols, int threads) {
    // Each worker owns complete rows and uses the same reference quantizer.
    // No shared reductions or altered rounding: bytes must match serial use.
    const int workers = (int) std::min<int64_t>(threads, std::max<int64_t>(1, rows / 64));
    if (workers <= 1) {
        return ggml_quantize_chunk(type, src, dst, 0, rows, cols, nullptr);
    }
    ggml_quantize_init(type);
    std::vector<std::future<size_t>> jobs;
    try {
        for (int i = 0; i < workers; i++) {
            const int64_t first = rows * i / workers, last = rows * (i + 1) / workers;
            jobs.emplace_back(std::async(std::launch::async, [=]() {
                return ggml_quantize_chunk(type, src, dst, first * cols, last - first, cols, nullptr);
            }));
        }
        size_t written = 0;
        for (auto & job : jobs) {
            written += job.get();
        }
        return written;
    } catch (const std::exception & e) {
        // Futures join before returning; the caller drops partially merged
        // residency on a size mismatch, just as for other merge failures.
        fprintf(stderr, "[MM3] LM merge quantization worker failed: %s\n", e.what());
        return 0;
    }
}

static bool mm3_lm_merge_apply(MM3Model * m, const MM3LmAdapter * ad, const MM3LmAdapterScales & sc,
                               bool requested_gpu, std::string * err) {
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
    // GPU quantization has different rounding from the CPU reference path.
    // 'verify' computes both paths but writes the CPU reference weights.
    const char * device_env = std::getenv("MM3_MERGE_DEVICE");
    const bool verify_device = requested_gpu && device_env && std::strcmp(device_env, "verify") == 0;
    const bool want_device = verify_device || mm3_lm_merge_device_enabled(requested_gpu);
    fprintf(stderr, "[MM3] LM merge policy: %s\n", verify_device ? "GPU verification (CPU weights written)" :
            want_device ? "GPU preferred (CPU-assisted fallback)" : "CPU-assisted");
    int n_device = 0, n_verified = 0;
    size_t mismatched_bytes = 0;
    std::vector<uint8_t> device_bytes;
    const char * threads_env = std::getenv("MM3_MERGE_THREADS");
    const int quant_threads = std::max(1, std::min(32, threads_env ? std::atoi(threads_env) : backend_cpu_n_threads()));
    const bool verify_cpu = std::getenv("MM3_MERGE_VERIFY_CPU") != nullptr;
    int n_cpu_verified = 0;
    std::vector<uint8_t> serial_bytes;
    // Host-clock phase totals. graph_compute synchronizes the scheduler;
    // downloads/uploads include their staging-vector allocation costs.
    double phases[7] = {};
    auto tick = std::chrono::steady_clock::now();
    auto mark = [&](int phase) {
        const auto now = std::chrono::steady_clock::now();
        phases[phase] += std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
    };

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

            // Small per-module graph: merged_f32 = cast(W) + s * (Aáµ€ · B),
            // with three variations that all keep that shape.
            tick = std::chrono::steady_clock::now();
            const size_t         need = ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(96, false);
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
            ggml_mul_mat_set_prec(delta, GGML_PREC_F32);
            ggml_tensor * basef  = ggml_cast(ctx, w, GGML_TYPE_F32);                    // [in, out]
            ggml_tensor * merged = nullptr;
            if (p.has_loha()) {
                // LoHa: delta = (A1 B1) (.) (A2 B2). Both products are full
                // [in, out] tensors; this is the reason it is merge-only.
                ggml_tensor * a2t = ggml_cont(ctx, ggml_transpose(ctx, p.a2));
                ggml_tensor * b2f = ggml_cast(ctx, p.b2, GGML_TYPE_F32);
                ggml_tensor * d2  = ggml_mul_mat(ctx, a2t, b2f);
                ggml_mul_mat_set_prec(d2, GGML_PREC_F32);
                merged = ggml_add(ctx, basef, ggml_scale(ctx, ggml_mul(ctx, delta, d2), s));
            } else if (ad->is_hira) {
                // HiRA: W' = W + W (.) (s*BA). Modulating the update by the base
                // elementwise is what makes it high-rank, and what makes it
                // impossible to express as a runtime low-rank pair.
                merged = ggml_add(ctx, basef, ggml_mul(ctx, basef, ggml_scale(ctx, delta, s)));
            } else {
                merged = ggml_add(ctx, basef, ggml_scale(ctx, delta, s));
            }
            if (p.m) {
                // DoRA: W' = m * (W + s*BA) / ||W + s*BA||_col. The norm is
                // recomputed here from the weights being written, so it is
                // exact for this merge — not the value the trainer last held.
                // Shapes: merged is [in, out], sum_rows gives [1, out], and
                // m reshaped to [1, out] broadcasts back over `in`.
                ggml_tensor * nr = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, merged)));   // [1, out]
                ggml_tensor * mv = ggml_reshape_2d(ctx, p.m, 1, p.m->ne[0]);                    // [1, out]
                merged           = ggml_mul(ctx, merged, ggml_div(ctx, mv, nr));
            }
            ggml_tensor * outt   = w->type == GGML_TYPE_F16 ? ggml_cast(ctx, merged, GGML_TYPE_F16) : merged;
            bool device_write = false;
            if (want_device && bp.has_gpu && w->buffer && !ggml_backend_buffer_is_host(w->buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(w->buffer)) == ggml_backend_get_device(bp.backend) &&
                (w->type == GGML_TYPE_Q8_0 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32)) {
                ggml_tensor * candidate = w->type == GGML_TYPE_F32 ? merged : ggml_cast(ctx, merged, w->type);
                if (ggml_backend_supports_op(bp.backend, candidate)) {
                    outt = candidate;
                    device_write = true;
                    if (verify_device && quantized) {
                        ggml_set_output(merged);  // preserve the exact pre-quantization input for comparison
                    }
                }
            }
            ggml_set_output(outt);

            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(gf, outt);
            mark(0);
            ggml_backend_sched_reset(sched);
            if (device_write) {
                ggml_backend_sched_set_tensor_backend(sched, outt, bp.backend);
            }
            const bool allocated = ggml_backend_sched_alloc_graph(sched, gf);
            mark(1);
            const bool computed = allocated && ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
            mark(2);
            if (!computed) {
                if (err) {
                    *err = "merge graph compute failed at layer " + std::to_string(l) + " (out of VRAM?)";
                }
                ggml_free(ctx);
                ok = false;
                break;
            }

            const int64_t ne0 = w->ne[0], ne1 = w->ne[1];
            if (device_write && !verify_device) {
                // Separate output storage keeps the base intact until the
                // entire graph has consumed it. This copy is synchronous.
                ggml_backend_tensor_copy(outt, w);
                mark(5);
                n_device++;
            } else if (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32) {
                // Same type as the graph output: read back and write straight in.
                const size_t bytes = ggml_nbytes(w);
                qbuf.resize(bytes);
                ggml_backend_tensor_get(outt, qbuf.data(), 0, bytes);
                mark(3);
                ggml_backend_tensor_set(w, qbuf.data(), 0, bytes);
                mark(5);
                moved += bytes * 2;
            } else {
                // Quantized base: f32 down, requantize on the host, raw blocks up.
                f32.resize((size_t) (ne0 * ne1));
                ggml_backend_tensor_get(merged, f32.data(), 0, f32.size() * sizeof(float));
                mark(3);
                qbuf.resize(ggml_nbytes(w));
                const size_t written = mm3_lm_merge_quantize(w->type, f32.data(), qbuf.data(), ne1, ne0, quant_threads);
                mark(4);
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
                if (verify_cpu) {
                    serial_bytes.resize(qbuf.size());
                    ggml_quantize_chunk(w->type, f32.data(), serial_bytes.data(), 0, ne1, ne0, nullptr);
                    if (std::memcmp(qbuf.data(), serial_bytes.data(), qbuf.size()) != 0) {
                        if (err) *err = "parallel merge quantization differs from the serial reference";
                        ggml_free(ctx);
                        ok = false;
                        break;
                    }
                    n_cpu_verified++;
                    mark(4);
                }
                if (device_write && verify_device) {
                    device_bytes.resize(qbuf.size());
                    ggml_backend_tensor_get(outt, device_bytes.data(), 0, device_bytes.size());
                    if (std::memcmp(qbuf.data(), device_bytes.data(), qbuf.size()) == 0) {
                        n_verified++;
                    } else {
                        for (size_t i = 0; i < qbuf.size(); i++) {
                            mismatched_bytes += qbuf[i] != device_bytes[i];
                        }
                    }
                    n_device++;
                    mark(4);
                }
                ggml_backend_tensor_set(w, qbuf.data(), 0, qbuf.size());
                mark(5);
                moved += f32.size() * sizeof(float) + qbuf.size();
            }
            ggml_free(ctx);
            mark(6);
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
        fprintf(stderr, "[MM3] LM merge phases: graph %.0f, allocate %.0f, compute %.0f, download %.0f, "
                        "CPU quantize %.0f, upload %.0f, cleanup %.0f ms\n",
                phases[0], phases[1], phases[2], phases[3], phases[4], phases[5], phases[6]);
        if (verify_device) {
            fprintf(stderr, "[MM3] LM merge device: %d modules, verify=%s, %d byte-identical, %llu differing bytes\n",
                    n_device, verify_device ? "yes (CPU weights written)" : "no", n_verified,
                    (unsigned long long) mismatched_bytes);
        } else if (want_device) {
            fprintf(stderr, "[MM3] LM merge device: %d modules written; byte verification disabled\n", n_device);
        }
        fprintf(stderr, "[MM3] LM merge CPU quantizer: %d threads, %d modules verified against serial\n",
                quant_threads, n_cpu_verified);
    }
    return ok;
}
