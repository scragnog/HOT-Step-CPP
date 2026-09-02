#pragma once
// dit-mirror.h — CPU-backend base load + streamed F32 weight mirror (plan §3.3).
//
// THE ORDERING IS THE WHOLE POINT (D3). The spike built the mirror from weights
// that were already resident on the GPU, so base + mirror were both in VRAM at
// once: +5.0 GB transient at K=32, +8.4 GB at K=8. A card that fits the steady
// state OOMs on that transient. Here the base GGUF is loaded onto the CPU
// backend (GGML_BACKEND=CPU around dit_ggml_load), the mirror is built straight
// into VRAM from the host copy, and the host weight buffer is released. Peak
// VRAM is the mirror alone — gate V9 measures it.
//
// Why a mirror at all (D2): ggml.c:6599-6612 computes the gradient w.r.t. the
// ACTIVATION input of mul_mat as ggml_out_prod(src0, transpose(grad)), pushing
// the frozen weight through out_prod, and ggml-cuda/out-prod.cu:11-13 asserts
// F32 on all three tensors. So every weight in a gradient-carrying layer must be
// F32, LoRA site or not.
//
// ...unless out_prod itself takes BF16. engine/patches/bf16-out-prod.patch adds
// exactly that to the CUDA kernel, which is what DIT_MIRROR_BF16 below trades on:
// the per-layer MATMUL weights — the ~99 % of the mirror by bytes — then stay in
// their native BF16 and the mirror roughly halves. Everything else (norms,
// scale_shift_table, the output head, temb) is tiny and reaches its gradient
// through non-out_prod ops, so it stays F32-promoted in both modes.
//
// docs/plans/2026-07-28-dit-trainer-implementation.md §3.3

#include "dit.h"
#include "train/lm-common.h"

#include <cstdlib>
#include <string>
#include <vector>

struct DitMirror {
    ggml_context *        ctx       = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    size_t                bytes     = 0;
    size_t                bytes_f32 = 0;
    int                   n_f32     = 0;
    int                   n_keep    = 0;
};

struct DitTrainModel {
    DiTGGML              m;                  // structure + rewired tensor pointers
    DitMirror            mirror;
    ggml_backend_t       backend = nullptr;  // CUDA (or best), owned by the trainer
    ggml_backend_t       cpu     = nullptr;  // the loader's CPU backend (shared cache)
    ggml_backend_sched_t sched   = nullptr;
    std::vector<float>   null_cond;          // host copy for CFG dropout
    int                  lora_lo    = 0;
    bool                 loaded     = false;
    bool                 mirrored   = false;
    long long            load_ms    = 0;
};

// ─── mirror precision policy ────────────────────────────────────────────────

// DIT_MIRROR_BF16_F32 (Rob, 2026-09-02) exists because the two shipped modes
// bundle two independent decisions that turned out to pull opposite ways:
//
//   STORAGE  — what precision the frozen weight sits in for the whole run.
//   COMPUTE  — what precision its GEMM runs at, forward and backward.
//
// `bf16` picks BF16 for both. The GGUF is BF16 anyway, so the storage half is
// lossless — but the compute half rounds the ACTIVATIONS on the way in and the
// GRADIENTS on the way back to bf16's 8-bit mantissa at every trainable-layer
// matmul, and Rob hears that: adapters trained under `--mirror bf16` render
// coarse/"bitty" audio where the same recipe at `--mirror f32` does not.
//
// `f32` picks F32 for both and sounds right, but the storage half costs ~8 GB
// more on a 32-layer XL base, which on a 32 GB card collapses the flash-mode
// auto-fit crop from ~1500 to ~552 — a worse adapter for a different reason.
//
// `bf16-f32` takes the storage of the first and the compute of the second: the
// mirror is byte-for-byte the bf16 mirror, and the graph inserts a per-site
// ggml_cast(w, F32) IN FRONT of each trainable layer's mul_mat (see
// DitAdapter::applyP in dit-adapter.h). The cast is transient — one weight's
// worth, freed by the allocator after its forward consumer — so residency
// tracks bf16 while the arithmetic tracks f32.
enum DitMirrorMode {
    DIT_MIRROR_F32 = 0,  // shipped behaviour: every trainable-layer weight -> F32
    DIT_MIRROR_BF16,     // trainable-layer matmul weights keep their native BF16
    DIT_MIRROR_BF16_F32, // BF16 storage, F32 compute (in-graph cast at each site)
};

static inline const char * dit_mirror_mode_name(DitMirrorMode mode) {
    switch (mode) {
        case DIT_MIRROR_BF16:     return "bf16";
        case DIT_MIRROR_BF16_F32: return "bf16-f32";
        default:                  return "f32";
    }
}

// Does this mode leave a trainable layer's native-BF16 matmul weight in BF16?
// True for both bf16 modes — they differ only in what the GRAPH does with it,
// never in what the mirror BUILDER stores, so dit_mirror_bytes_for() and the
// auto-fit's fixed term are identical between them by construction.
static inline bool dit_mirror_stores_bf16(DitMirrorMode mode) {
    return mode == DIT_MIRROR_BF16 || mode == DIT_MIRROR_BF16_F32;
}

// Does the graph promote each trainable-layer weight to F32 at its mul_mat site?
static inline bool dit_mirror_casts_in_graph(DitMirrorMode mode) {
    return mode == DIT_MIRROR_BF16_F32;
}

// ─── slot enumeration (shared by the sizing model and the builder) ──────────

struct DitSlot {
    ggml_tensor ** field;
    bool           promote;
};

// Collect every weight the training graph touches, with the §3.3 promotion
// policy applied for `lora_lo`. Pure bookkeeping — allocates nothing.
//
// `batched` (spike amendment A2c) is the micro-batch flag. proj_in / cond_emb
// normally stay native because they consume graph INPUTS and never an activation
// gradient — but a batch axis gives their mul_mat a src1 with ne2 > 1, and
// ggml_cuda_mul_mat() dispatches on exactly that (`src1->ne[2]*src1->ne[3] > 1`
// routes to the batched-cuBLAS path, ggml-cuda.cu:2618-2621), which on a BF16
// base is a BF16 GEMM instead of the F32-accurate mmf path the single-sample
// route takes. Measured on cond_emb: 3.5e-3 relative, i.e. BF16 eps. Promoting
// the four tensors costs a few MB. STRICTLY B > 1 — at B == 1 the dispatch is
// unchanged and promoting would break the §2.3.1 byte-identity anchor.
static void dit_mirror_slots(DiTGGML * m, int lora_lo, DitMirrorMode mode, std::vector<DitSlot> * slots,
                             bool batched = false) {
    slots->clear();
    auto add = [&](ggml_tensor ** f, bool promote) {
        if (f && *f) {
            slots->push_back({ f, promote });
        }
    };
    // A trainable layer's MATMUL weight: the only slot class DIT_MIRROR_BF16
    // touches, and only when the base actually stores it as BF16 — the patched
    // out_prod takes F32 or BF16 src0 and nothing else, so an F16 base still gets
    // the F32 promotion (and a quantized one is refused outright, below).
    //
    // DIT_MIRROR_BF16_F32 stores exactly what DIT_MIRROR_BF16 stores — the
    // difference lives in the graph, not here (dit_mirror_stores_bf16).
    auto add_mm = [&](ggml_tensor ** f, bool trainable) {
        const bool native_bf16 = (f && *f && (*f)->type == GGML_TYPE_BF16);
        add(f, trainable && !(dit_mirror_stores_bf16(mode) && native_bf16));
    };

    // Globals: proj_in / cond_emb consume graph INPUTS (never an activation
    // gradient), so they stay native; the output head descends from every
    // trainable layer and must be F32.
    add(&m->proj_in_w, batched);
    add(&m->proj_in_b, batched);
    add(&m->cond_emb_w, batched);
    add(&m->cond_emb_b, batched);
    add(&m->norm_out, true);
    add(&m->out_scale_shift, true);
    add(&m->proj_out_w, true);
    add(&m->proj_out_b, true);
    add(&m->scalar_one, false);

    // temb: tiny, and §3.4 runs the temb subgraph on the training backend.
    DiTGGMLTembWeights * tw[2] = { &m->time_embed, &m->time_embed_r };
    for (int i = 0; i < 2; i++) {
        add(&tw[i]->linear_1_w, true);
        add(&tw[i]->linear_1_b, true);
        add(&tw[i]->linear_2_w, true);
        add(&tw[i]->linear_2_b, true);
        add(&tw[i]->time_proj_w, true);
        add(&tw[i]->time_proj_b, true);
    }

    for (int i = 0; i < m->cfg.n_layers; i++) {
        DiTGGMLLayer & ly = m->layers[i];
        const bool     tr = (i >= lora_lo);  // this layer carries gradients
        add(&ly.self_attn_norm, tr);
        add_mm(&ly.sa_q_proj, tr);
        add_mm(&ly.sa_k_proj, tr);
        add_mm(&ly.sa_v_proj, tr);
        add(&ly.sa_q_norm, tr);
        add(&ly.sa_k_norm, tr);
        add_mm(&ly.sa_o_proj, tr);
        add(&ly.cross_attn_norm, tr);
        add_mm(&ly.ca_q_proj, tr);
        add_mm(&ly.ca_k_proj, tr);  // D5: cross-attn K/V now host adapters
        add_mm(&ly.ca_v_proj, tr);
        add(&ly.ca_q_norm, tr);
        add(&ly.ca_k_norm, tr);
        add_mm(&ly.ca_o_proj, tr);
        add(&ly.mlp_norm, tr);
        add_mm(&ly.gate_proj, tr);
        add_mm(&ly.up_proj, tr);
        add_mm(&ly.down_proj, tr);
        add(&ly.scale_shift_table, tr);
    }
}

// Exact mirror byte count for a given lora_lo, without allocating anything.
// This is what the auto-fit's fixed term uses (see dit-vram.h). It derives from
// the slot promote flags, so the precision mode is followed automatically — the
// auto-fit can never size a run against a mirror the builder would not build.
static size_t dit_mirror_bytes_for(DiTGGML * m, int lora_lo, DitMirrorMode mode, bool batched = false) {
    std::vector<DitSlot> slots;
    dit_mirror_slots(m, lora_lo, mode, &slots, batched);
    size_t total = 0;
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s   = *slots[i].field;
        const size_t  ne  = (size_t) ggml_nelements(s);
        const bool    f32 = slots[i].promote || s->type == GGML_TYPE_F32;
        total += f32 ? ne * 4 : ggml_nbytes(s);
    }
    return total;
}

// The transient F32 window DIT_MIRROR_BF16_F32 adds on top of the bf16 mirror,
// in bytes: the largest single trainable layer's projection set, promoted to F32.
//
// Why one LAYER and not one WEIGHT: the casts are per-site nodes with the layer's
// own lifetime pattern, and ggml_gallocr is free to keep several of them live at
// once (sa q/k/v are built back to back before the first is consumed, and the
// LoKR apply branch lengthens every site's live range). Charging the whole layer
// is the cheap conservative answer — ~0.5 GB on a 32-layer XL base against the
// ~8 GB the mode saves against a full F32 mirror.
//
// Only the eleven weights that reach DitAdapter::applyP are counted, because
// those are the only mul_mat src0s the graph casts. Norms, tables and the head
// are already F32 in the mirror and are never cast.
static size_t dit_mirror_cast_window_bytes(DiTGGML * m, int lora_lo, DitMirrorMode mode) {
    if (!dit_mirror_casts_in_graph(mode) || !m) {
        return 0;
    }
    size_t worst = 0;
    for (int i = (lora_lo < 0 ? 0 : lora_lo); i < m->cfg.n_layers; i++) {
        DiTGGMLLayer & ly = m->layers[i];
        ggml_tensor *  w[11] = { ly.sa_q_proj, ly.sa_k_proj, ly.sa_v_proj, ly.sa_o_proj,
                                 ly.ca_q_proj, ly.ca_k_proj, ly.ca_v_proj, ly.ca_o_proj,
                                 ly.gate_proj, ly.up_proj,   ly.down_proj };
        size_t         sum = 0;
        for (int j = 0; j < 11; j++) {
            if (w[j] && w[j]->type == GGML_TYPE_BF16) {
                sum += (size_t) ggml_nelements(w[j]) * 4;
            }
        }
        if (sum > worst) {
            worst = sum;
        }
    }
    return worst;
}

// ─── the §3.2 post-load assertion ───────────────────────────────────────────

static bool dit_assert_unfused(DiTGGML * m, std::string * err) {
    for (int i = 0; i < m->cfg.n_layers; i++) {
        DiTGGMLLayer & ly = m->layers[i];
        if (ly.sa_qkv || ly.sa_qk || ly.ca_qkv || ly.ca_kv || ly.gate_up) {
            char b[192];
            snprintf(b, sizeof(b),
                     "layer %d loaded FUSED (sa_qkv=%d sa_qk=%d ca_qkv=%d ca_kv=%d gate_up=%d) — "
                     "g_dit_load_no_fuse did not take effect",
                     i, ly.sa_qkv != nullptr, ly.sa_qk != nullptr, ly.ca_qkv != nullptr, ly.ca_kv != nullptr,
                     ly.gate_up != nullptr);
            *err = b;
            return false;
        }
        if (!ly.sa_q_proj || !ly.sa_k_proj || !ly.sa_v_proj || !ly.sa_o_proj || !ly.ca_q_proj || !ly.ca_k_proj ||
            !ly.ca_v_proj || !ly.ca_o_proj || !ly.gate_proj || !ly.up_proj || !ly.down_proj) {
            char b[128];
            snprintf(b, sizeof(b), "layer %d is missing an unfused projection", i);
            *err = b;
            return false;
        }
    }
    return true;
}

// ─── env guard ──────────────────────────────────────────────────────────────

struct DitEnvGuard {
    std::string prev;
    bool        had = false;

    void force_cpu() {
        const char * p = getenv("GGML_BACKEND");
        had            = (p != nullptr);
        if (had) {
            prev = p;
        }
#ifdef _WIN32
        _putenv_s("GGML_BACKEND", "CPU");
#else
        setenv("GGML_BACKEND", "CPU", 1);
#endif
    }

    ~DitEnvGuard() {
#ifdef _WIN32
        _putenv_s("GGML_BACKEND", had ? prev.c_str() : "");
#else
        if (had) {
            setenv("GGML_BACKEND", prev.c_str(), 1);
        } else {
            unsetenv("GGML_BACKEND");
        }
#endif
    }
};

// ─── ConvRot refusal (D23) ──────────────────────────────────────────────────

// True when the GGUF carries acestep.convrot_map. Checked BEFORE any allocation.
static bool dit_gguf_has_convrot(const char * path) {
    if (!dit_ends_with_gguf(path)) {
        return false;  // safetensors dirs carry no convrot map
    }
    GGUFModel gf;
    if (!gf_load(&gf, path)) {
        return false;  // the real load reports the failure
    }
    const char * s   = gf_get_str(gf, "acestep.convrot_map");
    const bool   has = (s != nullptr && s[0] != '\0');
    gf_close(&gf);
    return has;
}

// ─── the load ───────────────────────────────────────────────────────────────

static void dit_train_free(DitTrainModel * M) {
    if (M->sched) {
        ggml_backend_sched_free(M->sched);
        M->sched = nullptr;
    }
    if (M->mirror.buf) {
        ggml_backend_buffer_free(M->mirror.buf);
        M->mirror.buf = nullptr;
    }
    if (M->mirror.ctx) {
        ggml_free(M->mirror.ctx);
        M->mirror.ctx = nullptr;
    }
    if (M->backend) {
        ggml_backend_free(M->backend);
        M->backend = nullptr;
    }
    if (M->loaded) {
        // The mirror already released wctx.buffer and rewired every pointer; the
        // remaining work is the weight context + the shared backend refcount.
        M->m.sched = nullptr;  // freed above / never ours
        dit_ggml_free(&M->m);
        M->loaded = false;
    }
}

// Loads the base on the CPU backend, then streams the F32 mirror into VRAM.
// `lora_lo` is the lowest trainable layer index.
//
// Failure modes, all before or during the mirror and all reported by the caller
// as a `fatal`:
//   convrot-unsupported  the GGUF carries acestep.convrot_map (D23)
//   model-load           quantized base (D20), fused load, CPU force failed
static bool dit_train_load(DitTrainModel * M, const char * gguf_path, int lora_lo, std::string * err) {
    const int64_t t0 = ggml_time_ms();
    M->lora_lo       = lora_lo;

    // 1) config + ConvRot gate, before any allocation.
    if (!dit_ggml_load_config(&M->m.cfg, gguf_path)) {
        *err = std::string("cannot read the DiT config from ") + gguf_path;
        return false;
    }
    if (dit_gguf_has_convrot(gguf_path)) {
        *err = "convrot";  // sentinel: the caller maps this to reason=convrot-unsupported
        return false;
    }

    // 2/3) force the CPU backend and take the unfused branch. Both globals are
    // restored on EVERY exit path.
    {
        DitEnvGuard guard;
        guard.force_cpu();
        g_dit_load_no_fuse = true;
        // adapter_path stays nullptr, so dit.h's "adapter contributed nothing"
        // hard-fail is never reached — the whole reason the flag exists (D4).
        const bool ok      = dit_ggml_load(&M->m, gguf_path, nullptr, 1.0f);
        g_dit_load_no_fuse = false;
        if (!ok) {
            *err = std::string("cannot load the DiT base ") + gguf_path;
            return false;
        }
    }
    M->loaded = true;
    M->cpu    = M->m.cpu_backend;

    // 4) the CPU force must actually have taken.
    if (strcmp(ggml_backend_name(M->m.backend), "CPU") != 0) {
        *err = "could not load the DiT base on the CPU backend — streamed mirror unavailable";
        return false;
    }
    // The loader's inference-sized scheduler is useless to us and points at CPU.
    if (M->m.sched) {
        ggml_backend_sched_free(M->m.sched);
        M->m.sched = nullptr;
    }

    // 5) §3.2 post-load assertion.
    if (!dit_assert_unfused(&M->m, err)) {
        return false;
    }

    // null_condition_emb -> host (before the weight buffer goes away).
    if (M->m.null_condition_emb) {
        const size_t ne = (size_t) ggml_nelements(M->m.null_condition_emb);
        M->null_cond.assign(ne, 0.0f);
        if (M->m.null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> b16(ne);
            ggml_backend_tensor_get(M->m.null_condition_emb, b16.data(), 0, ne * sizeof(uint16_t));
            for (size_t i = 0; i < ne; i++) {
                const uint32_t w = (uint32_t) b16[i] << 16;
                memcpy(&M->null_cond[i], &w, 4);
            }
        } else if (M->m.null_condition_emb->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(M->m.null_condition_emb, M->null_cond.data(), 0, ne * sizeof(float));
        } else {
            M->null_cond.clear();
        }
    }

    M->load_ms = (long long) (ggml_time_ms() - t0);
    return true;
}

// The trainer's own compute backend. Separate from dit_train_load() so the
// caller can run the VRAM auto-fit against the real device before allocating.
static bool dit_train_backend_init(DitTrainModel * M, std::string * err) {
    M->backend = ggml_backend_init_best();
    if (!M->backend) {
        *err = "no compute backend available for training";
        return false;
    }
    if (strcmp(ggml_backend_name(M->backend), "CPU") == 0) {
        *err = "DiT training needs a GPU backend (CUDA/Vulkan); only CPU is available";
        ggml_backend_free(M->backend);
        M->backend = nullptr;
        return false;
    }
    return true;
}

// Build the mirror straight into VRAM at `lora_lo` and release the host copy.
static bool dit_build_mirror(DitTrainModel * M, int lora_lo, DitMirrorMode mode, std::string * err,
                             bool batched = false) {
    M->lora_lo = lora_lo;
    std::vector<DitSlot> slots;
    dit_mirror_slots(&M->m, lora_lo, mode, &slots, batched);

    {
        ggml_init_params p = { (slots.size() + 16) * ggml_tensor_overhead(), nullptr, true };
        M->mirror.ctx      = ggml_init(p);
    }
    if (!M->mirror.ctx) {
        *err = "cannot create the mirror context";
        return false;
    }

    // Refuse a quantized base BEFORE allocating a byte of VRAM (D20): a
    // quantized layer cannot carry an activation gradient at all.
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s = *slots[i].field;
        if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_BF16 && s->type != GGML_TYPE_F16) {
            char b[256];
            snprintf(b, sizeof(b),
                     "base weight '%s' is %s — DiT training needs a BF16/F16/F32 base "
                     "(quantized bases cannot be trained: ggml_out_prod takes F32 or BF16 only)",
                     s->name, ggml_type_name(s->type));
            *err = b;
            return false;
        }
    }

    std::vector<ggml_tensor *> dsts(slots.size(), nullptr);
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor *   s  = *slots[i].field;
        const ggml_type ty = slots[i].promote ? GGML_TYPE_F32 : s->type;
        ggml_tensor *   d  = ggml_new_tensor_4d(M->mirror.ctx, ty, s->ne[0], s->ne[1], s->ne[2], s->ne[3]);
        ggml_set_name(d, s->name);
        dsts[i] = d;
        M->mirror.bytes += ggml_nbytes(d);
        if (ty == GGML_TYPE_F32 && s->type != GGML_TYPE_F32) {
            M->mirror.bytes_f32 += ggml_nbytes(d);
            M->mirror.n_f32++;
        } else {
            M->mirror.n_keep++;
        }
    }

    M->mirror.buf = ggml_backend_alloc_ctx_tensors(M->mirror.ctx, M->backend);
    if (!M->mirror.buf) {
        char b[192];
        snprintf(b, sizeof(b), "F32 weight mirror allocation failed (%.1f MB)", M->mirror.bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->mirror.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Stream: host tensor -> host convert -> device tensor -> rewire.
    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (size_t i = 0; i < slots.size(); i++) {
        ggml_tensor * s  = *slots[i].field;
        ggml_tensor * d  = dsts[i];
        const size_t  ne = (size_t) ggml_nelements(s);
        raw.resize(ggml_nbytes(s));
        ggml_backend_tensor_get(s, raw.data(), 0, raw.size());

        if (d->type == s->type) {
            ggml_backend_tensor_set(d, raw.data(), 0, raw.size());
        } else if (s->type == GGML_TYPE_BF16) {
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t b = (uint32_t) u[j] << 16;
                float          v;
                memcpy(&v, &b, 4);
                f32[j] = v;
            }
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        } else {  // F16 (quantized already refused above)
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(d, f32.data(), 0, ne * sizeof(float));
        }
        *slots[i].field = d;
    }

    // Release the host weight buffer — the mirror fully replaces it.
    if (M->m.wctx.buffer) {
        ggml_backend_buffer_free(M->m.wctx.buffer);
        M->m.wctx.buffer = nullptr;
    }
    // NB `M->m.backend` deliberately stays the loader's CPU backend: dit_ggml_free
    // routes it through backend_release(), which would free the trainer's own CUDA
    // backend a second time if we retargeted it here. No graph builder reads it.
    M->m.use_flash_attn = false;  // GGML_OP_FLASH_ATTN_EXT has no backward
    M->mirrored         = true;
    return true;
}
