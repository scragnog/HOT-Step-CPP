#pragma once
// train/yue2-nar-train-run.h — the runner around yue2-nar-train-graph.h.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Phases 2 and 4 of
// docs/plans/yue2/08-nar-lora-trainer.md, implementing the gate half of
// docs/plans/yue2/09-nar-train-graph-contract.md §10 and the loop, the
// checkpoint/resume path and the exporter on top of it.
//
// ── What is in here ────────────────────────────────────────────────────────
//
//   `--fd-check N`  the gradient gate (phase 2). Loads the LM, builds one
//                   conditioning prefix, prefills the AR K/V once, draws a
//                   STAND-IN latent `z ~ N(0,1)`, runs one forward + backward
//                   and finite-differences the analytic gradient against the
//                   measured loss change. The stand-in is correct for a
//                   gradient check and wrong for anything else: FD tests that
//                   the backward agrees with the forward, which is a property
//                   of the GRAPH, not of the data.
//   (no --fd-check) the training loop (phase 4): manifest -> clip latents ->
//                   rectified-flow MSE, AdamW with warmup + cosine, grad
//                   accumulation, global-norm clipping, `--save-every`
//                   checkpoints, `--resume`, and a safetensors export in the
//                   key scheme 08 §4 (AS CORRECTED) and 10 §3 specify.
//
// `--vae-dir` is still accepted and ignored: latents reach this file through
// the preprocess manifest, never by encoding audio here.
//
// ── Determinism, and what it costs ─────────────────────────────────────────
//
// Every random decision in the loop — which clip, whether the caption is
// dropped, t, the noise — is drawn from a stream seeded by
// (seed, micro-step index, purpose), NOT from one long-lived generator. That
// is deliberate and it is what makes `--resume` exact: the data stream at
// micro-step k is a pure function of k, so a run that pauses at step 300 and
// resumes sees exactly the draws the uninterrupted run would have seen. A
// single rolling generator would have to be serialised, and any change to how
// many draws a step consumes would silently re-shuffle the rest of the run.
//
// ── Why the gate is mandatory and why it needs --nar-layers ────────────────
//
// A falling loss is not evidence of a correct backward — the DiT-trainer delta
// fiasco is this codebase's own proof (08-nar-lora-trainer.md §3). MM3's LM
// trainer gets to run TWO complementary checks (FD against the forward, and
// checkpointed-vs-naive backward against each other); with segments = 1 there
// is no second backward route here, so FD is the only gate and it has to be
// able to return a VERDICT rather than a note.
//
// That is what `--nar-layers K` buys. Against the shipped BF16 base, the
// difference between two forward evaluations is dominated by BF16 rounding
// rather than by the defect being looked for — mm3-lm-train-run.h:1040-1060
// measured exactly this failure mode (relative error GROWING as eps shrinks,
// the signature of cancellation rather than truncation) and fixed it by
// mirroring a truncated stack to F32. Here it is far cheaper than it was
// there: MM3 needed a pre-sliced F32 lm_head over an 8.6 B trunk (~1.7 GB);
// our output head is llm2vae [2048, 64], a rounding error. Two NAR layers plus
// the four flow heads is ~420 MB.
//
// The AR prefix needs no mirror at all. It enters the training graph as an F32
// CONSTANT canvas either way (contract §1), and a constant's precision cannot
// create a gradient-vs-forward discrepancy.
//
// Isolation is REFUSED on a quantized base, exactly as mm3_f32_isolate refuses
// it: dequantizing to measure would measure the quantizer.
//
// ── The silent no-op the gate has to dodge ─────────────────────────────────
//
// LoRA's B factor is zero-initialised, so dL/dA is identically zero and every
// `.A` probe would report ||g|| = 0 and "pass" while measuring nothing
// (mm3-dit-train-run.h:452-455 documents the same trap on the accumulator
// scan). `yue2_nt_init_adapters` therefore takes a `b_sigma`, and the gate
// passes 1e-2 where a real run passes 0.
//
// ── Negative control ───────────────────────────────────────────────────────
//
// Contract §10: a gate that has only ever passed is a green light, not a gate.
// `YUE2_FD_LOSSGRAD=<x>` seeds dL/dloss at x instead of 1.0 — set it to 2 and
// every probe must report rel ~= 0.5 (|1-2|/2), which is the signature of an
// analytic gradient off by a constant factor. Run it once after any change to
// this file; a PASS that survives a deliberately wrong seed gradient is not
// measuring the backward.
//
// The contract's second control (shift the ggml_set offset by one row) is a
// FORWARD-side fault and FD is expected to stay PASS on it — it is caught by
// the zero-init velocity check (plan §6 phase-2 gate 2), which is not
// implemented here yet. Noted so nobody reads a PASS as covering it.

#include "train/gpu-mem.h"
#include "train/lm-optim.h"
#include "train/preprocess-io.h"  // pm_mkdir_p, pm_js_*, pm_file_exists
#include "train/st-write.h"       // the exporter
#include "train/yue2-nar-train-graph.h"
#include "train/yue2-sidecar.h"  // yue2_style_string: the style template shared with yue2-ar-train

#include "hot-step-fsutf8.h"
#include "yue2/yue2-lm-graph.h"
#include "yue2/yue2-model.h"
#include "yue2/yue2-nar-graph.h"
#include "yue2/yue2-tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ── Arguments ──────────────────────────────────────────────────────────────

struct Yue2NarTrainArgs {
    // --lm points at a yue2-lm-<type>.gguf. Discovery is directory-based
    // (yue2_discover scans <dir>/yue2 then <dir>), so the path is split into a
    // search directory and a type token and handed to the normal loader rather
    // than opening the file behind its back.
    std::string lm_path;
    std::string models_dir;  // --models, used when --lm is absent
    // Accepted and IGNORED until phase 1 (yue2_vae_encode) lands. Declared now
    // so the CLI shape does not change under scripts once it does.
    std::string vae_dir;

    // The manifest `ace-train yue2-preprocess` writes (yue2_preprocess.json,
    // schema in 08 §5) and the directory that takes the checkpoints, the
    // snapshots and the final adapter.
    std::string manifest;
    std::string out_dir;
    std::string name = "yue2_nar_lora";  // export stem: <out>/<name>.safetensors

    // Conditioning for the gate. Real runs take these per clip from the
    // manifest; --style/--lyrics exist so the gate can build a realistic
    // prefix without one.
    std::string style;
    std::string lyrics;

    std::string target = "nar_attn_mlp";  // nar_attn | nar_attn_mlp | nar_attn_mlp_proj
    // Set by the CLI when --target was actually typed. The gate promotes an
    // UNSET target to nar_attn_mlp_proj (see yue2_nar_fdcheck_main); without
    // this flag it could not tell "the default" from "the user asked for
    // exactly this" and would silently override the second.
    bool        target_set = false;
    int64_t     rank   = 16;
    float       alpha  = 16.0f;
    int64_t     frames = 250;  // 10 s at 25 fps — contract §8 says this is structural
    // Set by the CLI when --frames was actually typed. Without it the loop
    // could not tell "the default 250" from "the user asked for exactly this"
    // and would have to choose between silently overriding a typed value and
    // silently ignoring the manifest's own clip length.
    bool        frames_set  = false;
    double      clip_seconds = 10.0;  // --frames wins when both are given
    uint64_t    seed   = 42;

    // ── the loop (08 §7's starting point, upstream's numbers) ──
    float       lr              = 1e-4f;
    int64_t     steps           = 800;
    int64_t     warmup          = 50;
    std::string lr_scheduler    = "cosine";  // cosine | constant
    int64_t     grad_accum      = 1;
    float       max_grad_norm   = 1.0f;
    float       weight_decay    = 0.01f;
    float       caption_dropout = 0.1f;
    std::string t_sampling      = "logit-normal";  // logit-normal | uniform
    // Prepended to every caption, upstream's `trigger_word`
    // (train.py:192 — `style = trigger if not caption else f"{trigger}, {caption}"`).
    // It is what the trained style is addressed by at generation time, so it
    // is written into the exported adapter's metadata as well.
    std::string trigger;
    // "upstream" (default) or "bare" — see Yue2ArTrainArgs::style_template.
    std::string style_template = "upstream";

    int64_t     save_every = 0;   // 0 = only the final export
    int64_t     log_every  = 10;
    // AR-prefix K/V canvases held at once. One canvas is ~104 MB at the
    // default clip length, so this is a real VRAM knob — see Yue2NtKvCache.
    int64_t     kv_cache   = 8;
    bool        resume     = false;  // continue from <out>/yue2_nar_ckpt.bin

    // --fd-check N: run the gradient gate over N probes instead of training.
    int    fd_check  = 0;
    double fd_eps    = 1e-2;  // a FLOOR on the step, not the step (contract §10)
    int    nar_layers = 2;    // F32 isolation depth; 0 = no isolation (report only)
};

// ── F32 isolation of the NAR stack ─────────────────────────────────────────
//
// mm3-f32-isolate.h's surgery, scaled down to what this gate reasons about.
// MIRRORED: the first `n_layers` NAR blocks (11 tensors each), the shared
// final norm, and the four flow heads with their biases. LEFT ALONE:
// latent_pos_embed (consumed only by ggml_get_rows, an exact widening, so
// mirroring it would cost memory to change nothing) and the entire AR half
// (it only ever produced the constant canvas).
//
// The original weight buffer is NOT freed: the untouched layers still live in
// it, so freeing it would dangle 26 layers' worth of pointers.
struct Yue2NtF32Slice {
    ggml_context *        ctx       = nullptr;
    ggml_backend_buffer_t buf       = nullptr;
    size_t                bytes     = 0;
    int                   n_tensors = 0;
    int                   n_layers  = 0;
};

static void yue2_nt_f32_free(Yue2NtF32Slice * M) {
    if (M->buf) {
        ggml_backend_buffer_free(M->buf);
    }
    if (M->ctx) {
        ggml_free(M->ctx);
    }
    *M = Yue2NtF32Slice{};
}

static bool yue2_nt_f32_isolate(Yue2Model * m, int n_layers, Yue2NtF32Slice * M, std::string * err) {
    if (n_layers < 1 || n_layers > (int) m->lm.nar_blk.size()) {
        *err = "f32-isolate: --nar-layers out of range";
        return false;
    }

    struct Slot {
        ggml_tensor ** field;
        ggml_tensor *  src;
        ggml_tensor *  dst;
    };
    std::vector<Slot> slots;

    const int budget = n_layers * 11 + 16;
    ggml_init_params p = { (size_t) budget * ggml_tensor_overhead() + 4096, nullptr, /*no_alloc*/ true };
    M->ctx             = ggml_init(p);
    if (!M->ctx) {
        *err = "f32-isolate: context alloc failed";
        return false;
    }

    bool bad_type = false;
    std::string bad_name, bad_type_name;
    auto add = [&](ggml_tensor ** field) {
        if (!field || !*field) {
            return;
        }
        ggml_tensor * s = *field;
        if (s->type != GGML_TYPE_F32 && s->type != GGML_TYPE_F16 && s->type != GGML_TYPE_BF16) {
            // A quantized base is a legitimate thing to TRAIN (the in-graph
            // cast handles it) but not a thing to ISOLATE.
            if (!bad_type) {
                bad_type      = true;
                bad_name      = ggml_get_name(s);
                bad_type_name = ggml_type_name(s->type);
            }
            return;
        }
        // Mirrors preserve ne0/ne1 only; every tensor here is 1-D or 2-D.
        ggml_tensor * d = ggml_new_tensor_2d(M->ctx, GGML_TYPE_F32, s->ne[0], s->ne[1]);
        ggml_set_name(d, ggml_get_name(s));
        M->bytes += ggml_nbytes(d);
        slots.push_back({ field, s, d });
    };

    for (int i = 0; i < n_layers; i++) {
        Yue2NarLayer & ly = m->lm.nar_blk[(size_t) i];
        add(&ly.attn_norm);
        add(&ly.attn_q);
        add(&ly.attn_k);
        add(&ly.attn_v);
        add(&ly.attn_output);
        add(&ly.attn_q_norm);
        add(&ly.attn_k_norm);
        add(&ly.ffn_norm);
        add(&ly.ffn_gate);
        add(&ly.ffn_up);
        add(&ly.ffn_down);
    }
    add(&m->lm.output_norm);
    add(&m->lm.vae2llm_w);
    add(&m->lm.vae2llm_b);
    add(&m->lm.llm2vae_w);
    add(&m->lm.llm2vae_b);
    add(&m->lm.time_embd_w[0]);
    add(&m->lm.time_embd_b[0]);
    add(&m->lm.time_embd_w[1]);
    add(&m->lm.time_embd_b[1]);

    if (bad_type) {
        *err = "f32-isolate: '" + bad_name + "' is " + bad_type_name +
               " — the gate needs an F16/BF16/F32 base (run it on bf16 even when training on a quant); "
               "pass --nar-layers 0 to skip isolation and get a report instead of a verdict";
        return false;
    }

    M->buf = ggml_backend_alloc_ctx_tensors(M->ctx, m->backend);
    if (!M->buf) {
        char b[160];
        snprintf(b, sizeof(b), "f32-isolate: allocation failed (%.1f MB)", M->bytes / 1048576.0);
        *err = b;
        return false;
    }
    ggml_backend_buffer_set_usage(M->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<uint8_t> raw;
    std::vector<float>   f32;
    for (Slot & sl : slots) {
        ggml_tensor * s      = sl.src;
        const size_t  nbytes = ggml_nbytes(s);
        const size_t  ne     = (size_t) ggml_nelements(s);
        raw.resize(nbytes);
        ggml_backend_tensor_get(s, raw.data(), 0, nbytes);
        if (s->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(sl.dst, raw.data(), 0, nbytes);
        } else if (s->type == GGML_TYPE_F16) {
            f32.resize(ne);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f32.data(), (int64_t) ne);
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        } else {  // BF16 — widen by hand, same as mm3-f32-isolate.h:174-183
            f32.resize(ne);
            const uint16_t * u = (const uint16_t *) raw.data();
            for (size_t j = 0; j < ne; j++) {
                const uint32_t bits = (uint32_t) u[j] << 16;
                float          val;
                memcpy(&val, &bits, 4);
                f32[j] = val;
            }
            ggml_backend_tensor_set(sl.dst, f32.data(), 0, ne * sizeof(float));
        }
        *sl.field = sl.dst;
    }

    M->n_tensors = (int) slots.size();
    M->n_layers  = n_layers;
    fprintf(stderr,
            "[yue2-fd] F32 isolation: %d NAR layers + final norm + flow heads, %d tensors, %.1f MB "
            "(latent_pos_embed left native — get_rows is an exact widening; the AR prefix enters as an "
            "F32 constant canvas either way)\n",
            n_layers, M->n_tensors, M->bytes / 1048576.0);
    return true;
}

// ── Model discovery from --lm <path> ───────────────────────────────────────
//
// yue2_discover is directory-based and picks the best quant unless pinned, so
// "--lm D:/models/yue2/yue2-lm-bf16.gguf" becomes
// yue2_discover(m, "D:/models/yue2", "bf16"). A path whose basename is not
// yue2-lm-<type>.gguf is rejected rather than silently loading a different
// file than the one named.
static bool yue2_nt_discover(Yue2Model * m, const Yue2NarTrainArgs & a, std::string * err) {
    if (!a.lm_path.empty()) {
        std::string path = a.lm_path;
        for (char & ch : path) {
            if (ch == '\\') {
                ch = '/';
            }
        }
        const size_t slash = path.find_last_of('/');
        const std::string dir  = (slash == std::string::npos) ? std::string(".") : path.substr(0, slash);
        const std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
        if (base.size() <= 13 || base.compare(0, 8, "yue2-lm-") != 0 ||
            base.compare(base.size() - 5, 5, ".gguf") != 0) {
            *err = "--lm must name a yue2-lm-<type>.gguf (got '" + base + "')";
            return false;
        }
        const std::string type = base.substr(8, base.size() - 13);
        yue2_discover(m, dir.c_str(), type.c_str());
    } else if (!a.models_dir.empty()) {
        yue2_discover(m, a.models_dir.c_str());
    } else {
        *err = "yue2-nar-train: one of --lm <yue2-lm-*.gguf> or --models <dir> is required";
        return false;
    }
    if (!yue2_available(*m)) {
        *err = m->meta_errors.empty() ? "YuE2 LM GGUF not found or its metadata probe failed"
                                      : m->meta_errors[0];
        return false;
    }
    return true;
}

// Discover + load the LM half, and say the one thing about this checkpoint
// that changes what training MEANS.
//
// Contract §7: upstream noises with the UNSHIFTED t while telling the network
// shift(raw). At this checkpoint's shift = 1.0 the two coincide exactly
// (yue2-nar-graph.h:188-195); at any other shift the upstream recipe is
// internally inconsistent and porting it silently would port the bug. Warn
// loudly rather than refuse — the recipe still runs, it is just a recipe
// nobody has reconciled.
static bool yue2_nt_open_model(Yue2Model * m, const Yue2NarTrainArgs & a, const char * tag, std::string * err) {
    if (!yue2_nt_discover(m, a, err)) {
        return false;
    }
    if (!a.vae_dir.empty()) {
        fprintf(stderr, "[%s] --vae-dir is accepted and IGNORED: latents reach this trainer through the "
                        "preprocess manifest, never by encoding audio here\n", tag);
    }
    if (!yue2_load_parts(m, /*want_lm=*/true, /*want_vae=*/false, YUE2_VAE_STANDARD, /*want_encoder=*/false,
                         err)) {
        *err = "load: " + *err;
        return false;
    }
    if (std::fabs((double) m->lm_cfg.timestep_shift - 1.0) > 1e-6) {
        fprintf(stderr,
                "[%s] WARNING: yue2.timestep_shift = %.6f, not 1.0. The upstream recipe noises with the "
                "unshifted t while feeding the network shift(raw); those only coincide at shift 1.0. "
                "Reconcile before training anything on this checkpoint.\n",
                tag, (double) m->lm_cfg.timestep_shift);
    }
    return true;
}

// ── Trainable state: adapters, optimizer scalars, optimizer, schedulers ────
//
// Shared by the gate and the loop so there is exactly one place that knows
// which five host-owned scalars LmOptim needs and which buffer usage hint the
// adapters have to carry. Both of those are things this codebase has already
// paid for once (mm3-dit-train-run.h:553-583 for the scalars, :590-598 for the
// hint) and neither fails loudly.
struct Yue2NtTrainCtx {
    ggml_context *             actx = nullptr;
    ggml_backend_buffer_t      abuf = nullptr;
    Yue2TrainAdapters          ad;
    std::vector<ggml_tensor *> params;
    ggml_tensor *              t_lossgrad = nullptr;
    ggml_tensor *              t_adamw    = nullptr;
    ggml_tensor *              t_clip     = nullptr;
    ggml_tensor *              t_eps      = nullptr;
    ggml_tensor *              t_gnorm2   = nullptr;
    LmOptim                    opt;
    ggml_backend_sched_t       sched  = nullptr;  // forward (+ backward)
    ggml_backend_sched_t       osched = nullptr;  // optimizer only
};

static void yue2_nt_train_ctx_free(Yue2NtTrainCtx * C) {
    if (C->sched) {
        ggml_backend_sched_free(C->sched);
        C->sched = nullptr;
    }
    if (C->osched) {
        ggml_backend_sched_free(C->osched);
        C->osched = nullptr;
    }
    lm_optim_free(&C->opt);
    if (C->abuf) {
        ggml_backend_buffer_free(C->abuf);
        C->abuf = nullptr;
    }
    if (C->actx) {
        ggml_free(C->actx);
        C->actx = nullptr;
    }
    C->params.clear();
    C->ad = Yue2TrainAdapters{};
}

// `b_sigma` is the gate's escape from LoRA's zero-init trap (see the file
// header); a real run passes 0 so a fresh adapter is an exact no-op.
// `lossgrad` is dL/dloss: 1/grad_accum for a real run, 1.0 (or the negative
// control's value) for the gate.
static bool yue2_nt_train_ctx_init(const Yue2Model & m, int n_layers, int64_t rank, float alpha,
                                   Yue2NtTarget target, uint64_t seed, float b_sigma, float lossgrad,
                                   float grad_clip, const char * tag, Yue2NtTrainCtx * C, std::string * err) {
    const size_t     n_tensors = yue2_nt_adapter_tensor_count(n_layers, target);
    ggml_init_params aip       = { (n_tensors + 16) * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    C->actx                    = ggml_init(aip);
    if (!C->actx || !yue2_nt_make_adapters(C->actx, m, n_layers, rank, alpha, target, &C->ad, &C->params) ||
        C->params.size() != n_tensors) {
        char b[160];
        snprintf(b, sizeof(b), "adapter allocation failed (%zu of %zu tensors)", C->params.size(), n_tensors);
        *err = b;
        return false;
    }
    // The five host-owned optimizer scalars. LmOptim declares every one of
    // them nullptr and lm_optim_init creates NONE — each is written or read
    // unconditionally inside lm_optim_step, so a missing one is a null
    // dereference AFTER the graph has computed, which reads as a fault in the
    // optimizer maths. mm3-dit-train-run.h:553-583 is the full post-mortem.
    C->t_lossgrad = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_adamw    = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 7);
    C->t_clip     = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_eps      = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    C->t_gnorm2   = ggml_new_tensor_1d(C->actx, GGML_TYPE_F32, 1);
    ggml_set_name(C->t_lossgrad, "lossgrad");
    ggml_set_name(C->t_adamw, "adamw_params");
    ggml_set_name(C->t_clip, "grad_clip");
    ggml_set_name(C->t_eps, "eps");
    ggml_set_name(C->t_gnorm2, "gnorm2");

    C->abuf = ggml_backend_alloc_ctx_tensors(C->actx, m.backend);
    if (!C->abuf) {
        *err = "adapter buffer allocation failed";
        return false;
    }
    // Without the WEIGHTS usage hint the scheduler cuts the graph once per
    // LoRA parameter — 293 splits and 47.6 s/step on MM3, GPU at 9 %.
    ggml_backend_buffer_set_usage(C->abuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    yue2_nt_init_adapters(C->params, seed, b_sigma);
    {
        const float epsv = 1e-6f;
        ggml_backend_tensor_set(C->t_lossgrad, &lossgrad, 0, sizeof(float));
        ggml_backend_tensor_set(C->t_clip, &grad_clip, 0, sizeof(float));
        ggml_backend_tensor_set(C->t_eps, &epsv, 0, sizeof(float));
    }

    C->opt.optimizer  = "adamw";  // 08 §7: Muon is MM3's r256 answer, not needed at r16
    C->opt.t_lossgrad = C->t_lossgrad;
    C->opt.t_adamw    = C->t_adamw;
    C->opt.t_clip     = C->t_clip;
    C->opt.t_eps      = C->t_eps;
    C->opt.t_gnorm2   = C->t_gnorm2;
    C->opt.grad_clip  = grad_clip;
    if (!lm_optim_init(&C->opt, C->params, m.backend, err)) {
        *err = "optimizer init: " + *err;
        return false;
    }
    fprintf(stderr, "[%s] %zu LoRA tensors over %d NAR layers, rank %lld, alpha %.1f, target %s\n", tag,
            C->params.size(), n_layers, (long long) rank, (double) alpha, yue2_nt_target_name(target));

    BackendPair bp{};
    bp.backend     = m.backend;
    bp.cpu_backend = m.cpu_backend;
    bp.has_gpu     = m.backend != m.cpu_backend;
    C->sched       = backend_sched_new(bp, 65536);
    C->osched      = backend_sched_new(bp, 16384);  // optimizer graph only
    if (!C->sched || !C->osched) {
        *err = "scheduler alloc failed";
        return false;
    }
    return true;
}

// ── One micro-step ─────────────────────────────────────────────────────────
//
// Builds forward -> loss, optionally the backward, computes, and returns both
// the in-graph loss and the predicted velocity (the numeric arm re-aggregates
// the loss from `pred` in double — see yue2_nt_loss_host on why).
//
// Graph/backward order is dit-train-run.h's, copied because getting it wrong
// is silent: forward_expand FIRST, then fill_gacc (it is indexed by
// forward-node order and is wrong if built early), then backward_expand, then
// sched reset -> alloc -> upload -> compute.
static bool yue2_nt_micro(const Yue2Model & m, ggml_backend_sched_t sched, const Yue2TrainAdapters & ad,
                          const Yue2NarTrainKv & kv, LmOptim * opt, int n_layers,
                          const Yue2NarTrainHostInputs & hin, const std::vector<float> & vtarget_h,
                          bool backward, float * loss_out, std::vector<float> * pred_out, std::string * err) {
    const Yue2LmConfig & c     = m.lm_cfg;
    const int64_t        LD    = (int64_t) c.latent_dim;
    const int64_t        N_nar = kv.n_nar;
    const int64_t        T     = N_nar - 2;

    // 28 layers x ~30 forward nodes + a ~3x backward is ~2.6k; 65,536 is
    // generous rather than "safe at any size". Oversizing the CAP while
    // undersizing the ARENA is what bit the MM3 DiT trainer
    // (mm3-dit-train-run.h:336-343): ggml_new_tensor returns NULL, expansion
    // carries on, and the crash lands later in alloc with no message. Both are
    // sized together here.
    const size_t MAX_NODES = 65536;
    const size_t meta      = ggml_tensor_overhead() * (MAX_NODES * 2) +
                        ggml_graph_overhead_custom(MAX_NODES, true) + (size_t) 64 * 1024 * 1024;
    ggml_init_params ip  = { meta, nullptr, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        if (err) {
            *err = "ggml_init failed (yue2_nt_micro)";
        }
        return false;
    }

    Yue2NarTrainInputs in;
    in.x_nar     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, LD, N_nar);
    in.time_feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    in.local_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_nar);
    in.rope_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_nar);
    in.vtarget   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, LD, T);
    if (!in.x_nar || !in.time_feat || !in.local_idx || !in.rope_pos || !in.vtarget) {
        ggml_free(ctx);
        if (err) {
            *err = "input tensor alloc failed (ctx arena too small)";
        }
        return false;
    }
    ggml_set_name(in.x_nar, "yue2_nt_x_nar");
    ggml_set_name(in.time_feat, "yue2_nt_time_feat");
    ggml_set_name(in.local_idx, "yue2_nt_local_idx");
    ggml_set_name(in.rope_pos, "yue2_nt_rope_pos");
    ggml_set_name(in.vtarget, "yue2_nt_vtarget");
    ggml_set_input(in.x_nar);
    ggml_set_input(in.time_feat);
    ggml_set_input(in.local_idx);
    ggml_set_input(in.rope_pos);
    ggml_set_input(in.vtarget);

    ggml_tensor * pred = yue2_nt_build_forward(ctx, m, ad, kv, in, n_layers);
    ggml_tensor * loss = yue2_nt_loss(ctx, pred, in.vtarget);
    ggml_set_output(pred);
    if (backward) {
        ggml_set_loss(loss);
    }
    ggml_set_output(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, backward);
    if (!gf) {
        ggml_free(ctx);
        if (err) {
            *err = "ggml_new_graph_custom failed";
        }
        return false;
    }
    ggml_build_forward_expand(gf, loss);
    ggml_build_forward_expand(gf, pred);
    if (backward) {
        std::vector<ggml_tensor *> gacc;
        lm_optim_fill_gacc(opt, gf, &gacc);
        ggml_build_backward_expand(ctx, gf, gacc.data());
    }

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_free(ctx);
        if (err) {
            *err = "sched_alloc_graph failed (out of VRAM?)";
        }
        return false;
    }

    ggml_backend_tensor_set(in.x_nar, hin.x_nar.data(), 0, hin.x_nar.size() * sizeof(float));
    ggml_backend_tensor_set(in.time_feat, hin.time_feat.data(), 0, hin.time_feat.size() * sizeof(float));
    ggml_backend_tensor_set(in.local_idx, hin.local_idx.data(), 0, hin.local_idx.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in.rope_pos, hin.rope_pos.data(), 0, hin.rope_pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in.vtarget, vtarget_h.data(), 0, vtarget_h.size() * sizeof(float));

    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        if (loss_out) {
            ggml_backend_tensor_get(loss, loss_out, 0, sizeof(float));
        }
        if (pred_out) {
            pred_out->assign((size_t) (LD * T), 0.0f);
            ggml_backend_tensor_get(pred, pred_out->data(), 0, pred_out->size() * sizeof(float));
        }
    } else if (err) {
        *err = "graph compute failed";
    }
    ggml_free(ctx);
    return ok;
}

// ── The gate ───────────────────────────────────────────────────────────────

static int yue2_nar_fdcheck_main(const Yue2NarTrainArgs & a) {
    // TF32 turns an F32 matmul into a ~1e-3-accurate one, which is the same
    // order as the defect the gate looks for (mm3-lm-train-run.h:1124-1128).
#ifdef _WIN32
    _putenv_s("NVIDIA_TF32_OVERRIDE", "0");
#else
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
#endif

    std::string err;
    static Yue2Model m;
    if (!yue2_nt_open_model(&m, a, "yue2-fd", &err)) {
        fprintf(stderr, "[yue2-fd] %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-fd] the gate uses z ~ N(0,1) as a stand-in latent; it tests the GRAPH, not the "
                    "data, so no manifest is read here\n");
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        LD = (int64_t) c.latent_dim;
    const int64_t        T  = a.frames;
    if (T <= 0) {
        fprintf(stderr, "[yue2-fd] --frames must be > 0\n");
        return 1;
    }

    Yue2NtTarget target = YUE2_NT_ATTN_MLP;
    if (!yue2_nt_parse_target(a.target, &target)) {
        fprintf(stderr, "[yue2-fd] --target must be nar_attn, nar_attn_mlp or nar_attn_mlp_proj\n");
        return 1;
    }
    // Contract §10 names one HEAD probe (llm2vae.B, or time_embd.1.B under
    // proj), but §6 makes the four flow heads sites only under the `proj`
    // preset — so under the default preset that probe has no tensor to
    // address. The gate therefore promotes the preset to nar_attn_mlp_proj
    // unless the caller asked for something specific, which is also the
    // stricter check: every site the graph can ever have is probed.
    if (!a.target_set && target != YUE2_NT_ATTN_MLP_PROJ) {
        target = YUE2_NT_ATTN_MLP_PROJ;
        fprintf(stderr, "[yue2-fd] default --target promoted to nar_attn_mlp_proj so the flow-head probes "
                        "have tensors to address (contract §10); pass --target explicitly to keep a "
                        "narrower preset\n");
    }

    // ── conditioning: the cot="off" prefix, exactly as inference builds it ──
    BPETokenizer tok;
    if (!yue2_tokenizer_load_from_gguf(&tok, m.lm_file.path)) {
        fprintf(stderr, "[yue2-fd] tokenizer: cannot read tokenizer.ggml.* from %s\n", m.lm_file.path.c_str());
        return 1;
    }
    Yue2NarTrainCond cond;
    try {
        // The protocol lives in yue2-tokenizer.h and is NOT reimplemented here;
        // yue2_nar_train_cond_ids then appends codec ids (none, in the
        // text-only regime) and MUSIC_END, which is yue2-pipeline.h:447-452.
        const std::vector<int> pre = yue2_token_prefixes(&tok, a.style, a.lyrics, YUE2_COT_OFF, nullptr);
        cond.prefix_ids.assign(pre.begin(), pre.end());
    } catch (const std::exception & e) {
        fprintf(stderr, "[yue2-fd] prefix assembly failed: %s\n", e.what());
        return 1;
    }
    const std::vector<int32_t> ar_ids = yue2_nar_train_cond_ids(cond);

    // Prefill the AR K/V once. This runs the FULL 28-layer AR stack — it must
    // happen BEFORE any truncation, and it is deliberately outside the
    // isolation: the AR half is frozen, carries no LoRA site, and reaches the
    // training graph only as a constant.
    Yue2NarChunk chunk;
    if (!yue2_nar_chunk_init(m, ar_ids, T, &chunk, &err)) {
        fprintf(stderr, "[yue2-fd] AR prefill: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-fd] conditioning: %zu prefix ids (cot=off, %zu codec ids) -> ar_len %lld; "
                    "T %lld frames, N_nar %lld, S_kv %lld\n",
            cond.prefix_ids.size(), cond.codec_ids.size(), (long long) chunk.ar_length, (long long) T,
            (long long) chunk.nar_length, (long long) (chunk.ar_length + chunk.nar_length));

    // ── F32 isolation ──
    Yue2NtF32Slice iso;
    bool           isolated = false;
    const int      n_layers = a.nar_layers > 0 ? std::min(a.nar_layers, (int) c.block_count) : (int) c.block_count;
    if (a.nar_layers > 0) {
        if (!yue2_nt_f32_isolate(&m, n_layers, &iso, &err)) {
            fprintf(stderr, "[yue2-fd] %s\n", err.c_str());
            yue2_nt_f32_free(&iso);
            yue2_nar_chunk_free(&chunk);
            return 1;
        }
        isolated = true;
    } else {
        fprintf(stderr, "[yue2-fd] --nar-layers 0: no F32 isolation, so this run REPORTS, it does not "
                        "gate (BF16 rounding is larger than the defect being looked for)\n");
    }

    // ── the persistent F32 AR-prefix canvas ──
    Yue2NarTrainKv kv;
    if (!yue2_nt_kv_from_chunk(m, chunk, n_layers, &kv, &err)) {
        fprintf(stderr, "[yue2-fd] KV canvas: %s\n", err.c_str());
        yue2_nt_f32_free(&iso);
        yue2_nar_chunk_free(&chunk);
        return 1;
    }
    // The AR cache's own F16 copy is dead weight once widened.
    yue2_nar_chunk_free(&chunk);

    // ── adapters, optimizer, scheduler ──
    //
    // B NON-ZERO for the gate (b_sigma 1e-2). With B == 0 (the production
    // init) dL/dA is identically zero and every .A probe passes while
    // measuring nothing.
    //
    // YUE2_FD_LOSSGRAD is the negative control (file header): seed dL/dloss at
    // 2.0 and every probe must land at rel ~= 0.5.
    float        lg = 1.0f;
    const char * ev = std::getenv("YUE2_FD_LOSSGRAD");
    if (ev && ev[0]) {
        lg = (float) atof(ev);
        fprintf(stderr, "[yue2-fd] NEGATIVE CONTROL: dL/dloss seeded at %.3f — expect rel ~= %.3f on "
                        "every probe, and a FAIL\n",
                (double) lg, (double) std::fabs(1.0f - lg) / (double) std::max(1e-6f, std::fabs(lg)));
    }
    Yue2NtTrainCtx C;
    if (!yue2_nt_train_ctx_init(m, n_layers, a.rank, a.alpha, target, a.seed, /*b_sigma=*/1e-2f,
                                /*lossgrad=*/lg, /*grad_clip=*/1.0f, "yue2-fd", &C, &err)) {
        fprintf(stderr, "[yue2-fd] %s\n", err.c_str());
        return 1;
    }
    Yue2TrainAdapters &          ad     = C.ad;
    std::vector<ggml_tensor *> & params = C.params;
    LmOptim &                    opt    = C.opt;
    ggml_backend_sched_t         sched  = C.sched;

    // ── the fixed sample: z, noise, t ──
    //
    // z is a STAND-IN, not a real clip (file header). Separate seed streams so
    // adding a knob later cannot shift an unrelated draw.
    std::vector<float> z((size_t) (T * LD)), noise((size_t) (T * LD)), x_t, vtarget;
    yue2_nt_fill_normal(&z, a.seed ^ 0x11ull);
    yue2_nt_fill_normal(&noise, a.seed ^ 0x22ull);

    double raw_t = 0.0;
    if (a.t_sampling == "uniform") {
        // Contract §7: a uniform t must be pushed through logit+clamp before it
        // becomes the time scalar, because yue2_nar_velocity's argument IS the
        // logit-transformed value (yue2-nar-graph.h:384-389).
        std::vector<float> u(2, 0.0f);
        yue2_nt_fill_normal(&u, a.seed ^ 0x33ull);
        const double t01 = 0.5 * (1.0 + std::erf((double) u[0] / std::sqrt(2.0)));  // N(0,1) -> U(0,1)
        raw_t            = yue2_nar_logit_clamped(std::min(0.999, std::max(0.001, t01)));
    } else {
        // Logit-normal: raw IS the sample, no clamp needed (train.py:131-139).
        std::vector<float> r(2, 0.0f);
        yue2_nt_fill_normal(&r, a.seed ^ 0x33ull);
        raw_t = (double) r[0];
    }
    const float t_val = (float) (1.0 / (1.0 + std::exp(-raw_t)));
    yue2_nt_make_xt_target(z, noise, t_val, &x_t, &vtarget);

    Yue2NarTrainHostInputs hin;
    if (!yue2_nt_host_inputs(c, kv.ar_len, kv.n_nar, x_t, raw_t, &hin, &err)) {
        fprintf(stderr, "[yue2-fd] %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-fd] t sampling %s: raw_t %.6f -> t %.6f (shift %.3f)\n", a.t_sampling.c_str(),
            raw_t, (double) t_val, (double) c.timestep_shift);

    // ── numeric arm: forward only, loss re-aggregated on the host in double ──
    std::vector<float> pred_host;
    auto forward_loss = [&]() -> double {
        float f = 0.0f;
        if (!yue2_nt_micro(m, sched, ad, kv, &opt, n_layers, hin, vtarget, /*backward=*/false, &f, &pred_host,
                           &err)) {
            fprintf(stderr, "[yue2-fd] forward failed: %s\n", err.c_str());
            return std::nan("");
        }
        return yue2_nt_loss_host(pred_host, vtarget);
    };

    const double l0 = forward_loss();
    const double l1 = forward_loss();
    if (std::isnan(l0) || std::isnan(l1)) {
        return 1;  // forward_loss already printed why
    }
    // Measured, not assumed. Everything the probes claim has to be large
    // compared to this; printing it is what turned a mystifying per-entry
    // result into an obvious one on the MM3 gate.
    fprintf(stderr, "[yue2-fd] base loss %.6f (repeat %.6f, |delta| %.2e)\n", l0, l1, std::fabs(l1 - l0));

    // ── analytic arm: one forward + backward ──
    ggml_backend_buffer_clear(opt.buf_grad, 0);
    float loss_graph = 0.0f;
    if (!yue2_nt_micro(m, sched, ad, kv, &opt, n_layers, hin, vtarget, /*backward=*/true, &loss_graph, nullptr,
                       &err)) {
        fprintf(stderr, "[yue2-fd] backward failed: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-fd] in-graph loss %.6f vs host-double %.6f (|delta| %.2e — the in-graph f32 sum "
                    "is the reason the numeric arm re-aggregates)\n",
            (double) loss_graph, l0, std::fabs((double) loss_graph - l0));

    auto find_param = [&](const char * name) -> ggml_tensor * {
        for (ggml_tensor * t : params) {
            const char * nm = ggml_get_name(t);
            if (nm && strcmp(nm, name) == 0) {
                return t;
            }
        }
        return nullptr;
    };
    auto grad_vec = [&](ggml_tensor * par) -> std::vector<float> {
        auto it = opt.param_slot.find(par);
        GGML_ASSERT(it != opt.param_slot.end());
        ggml_tensor *      acc = opt.acc[(size_t) it->second];
        std::vector<float> g((size_t) ggml_nelements(acc));
        ggml_backend_tensor_get(acc, g.data(), 0, g.size() * sizeof(float));
        return g;
    };

    // Contract §10's six, in order. Blocks 0 and 1 rather than 0/13/27 because
    // --nar-layers 2 is what makes the check a verdict. Any probe whose tensor
    // does not exist under the chosen preset is skipped with a note rather
    // than silently dropped.
    const char * probe_names[] = {
        "blk.0.nar_attn_q.A", "blk.0.nar_attn_q.B", "blk.0.nar_ffn_down.B",
        "blk.1.nar_attn_v.B", "blk.1.nar_ffn_gate.A", "llm2vae.B",
        "time_embd.1.B",
    };
    const int    n_candidates = (int) (sizeof(probe_names) / sizeof(probe_names[0]));
    const int    n_want       = a.fd_check > 0 ? a.fd_check : 6;
    struct Probe { const char * name; ggml_tensor * par; };
    std::vector<Probe> probes;
    for (int i = 0; i < n_candidates && (int) probes.size() < n_want; i++) {
        ggml_tensor * p = find_param(probe_names[i]);
        if (!p) {
            fprintf(stderr, "[yue2-fd] probe '%s' skipped: no such site under --target %s / --nar-layers %d\n",
                    probe_names[i], yue2_nt_target_name(target), n_layers);
            continue;
        }
        probes.push_back({ probe_names[i], p });
    }
    if (probes.empty()) {
        fprintf(stderr, "[yue2-fd] no probe has a tensor to address — nothing was checked\n");
        return 1;
    }

    // THE STEP IS PER PROBE and --fd-eps is only its FLOOR. What must clear the
    // forward's own resolution is the LOSS CHANGE, which along the unit
    // gradient direction is exactly 2*h*||g||; a fixed eps measures every probe
    // at a different signal-to-noise ratio, in direct proportion to ||g||.
    // mm3-lm-train-run.h:1459-1500 has the measured sweep behind this rule.
    const double dl_min = std::max(1e-4, 1e-3 * std::fabs(l0));
    fprintf(stderr,
            "[yue2-fd] step floor: dL >= %.2e; a probe whose 2*eps*||g|| clears it keeps eps %.3g, and a "
            "raised step is capped at 0.05*||w||\n",
            dl_min, a.fd_eps);

    fprintf(stderr, "\n[yue2-fd] %-24s %10s %13s %9s %9s %13s %8s\n", "probe (whole tensor)", "n", "||g||",
            "step", "h/||w||", "numeric", "rel");
    double              worst    = 0.0;
    int                 n_raised = 0;
    std::vector<bool>   fd_clamped;
    std::vector<double> fd_rel;
    for (const Probe & pr : probes) {
        const std::vector<float> g = grad_vec(pr.par);
        double                   norm2 = 0.0;
        for (float x : g) {
            norm2 += (double) x * (double) x;
        }
        const double gnorm = std::sqrt(norm2);

        // PERTURB A DIRECTION, NOT A SINGLE ENTRY. Along v = g/||g|| the
        // directional derivative is exactly ||g|| and the loss change is
        // ~2*h*||g||, thousands of times the f32 floor; a per-entry difference
        // measures rounding only (mm3-lm-train-run.h:1104-1117).
        std::vector<float> w0((size_t) ggml_nelements(pr.par)), wtmp(w0.size());
        ggml_backend_tensor_get(pr.par, w0.data(), 0, w0.size() * sizeof(float));
        double wnorm2 = 0.0;
        for (float x : w0) {
            wnorm2 += (double) x * (double) x;
        }
        const double wnorm = std::sqrt(wnorm2);

        // ...AND THE RAISE NEEDS A CEILING. A probe whose gradient is
        // suppressed by the zero-ish B init lands at ||g|| ~ 1e-3, and
        // dl_min/(2*||g||) is then a step of order 1 — far outside the linear
        // regime the central difference assumes, so the O(h^2 f''') truncation
        // term swamps the measurement and the probe FAILS in a way that looks
        // exactly like a broken backward. Cap the step at 5% of the probed
        // tensor's own norm. "Could not reach the signal floor" is a different
        // statement from "the gradient is wrong", so it gets reported as a
        // different one: INCONCLUSIVE below, and h/||w|| in the table — any
        // row sitting at 0.05 is capped.
        double     h       = a.fd_eps;
        bool       clamped = false;
        if (gnorm > 0.0 && 2.0 * a.fd_eps * gnorm < dl_min) {
            h = dl_min / (2.0 * gnorm);
            n_raised++;
            const double h_max = 0.05 * wnorm;
            if (wnorm > 0.0 && h > h_max) {
                h       = h_max;
                clamped = true;
            }
        }
        const double h_over_w = wnorm > 0.0 ? h / wnorm : std::nan("");

        double num = std::nan("");
        if (gnorm > 0.0) {
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] + h * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lp = forward_loss();
            for (size_t k = 0; k < w0.size(); k++) {
                wtmp[k] = (float) ((double) w0[k] - h * (double) g[k] / gnorm);
            }
            ggml_backend_tensor_set(pr.par, wtmp.data(), 0, wtmp.size() * sizeof(float));
            const double lm_ = forward_loss();
            num              = (lp - lm_) / (2.0 * h);
        }
        ggml_backend_tensor_set(pr.par, w0.data(), 0, w0.size() * sizeof(float));

        const double rel = std::fabs(num - gnorm) / std::max(1e-12, gnorm);
        fprintf(stderr, "[yue2-fd] %-24s %10zu %13.6e %9.3g %9.3g %13.6e %8.3f%s\n", pr.name, g.size(), gnorm,
                h, h_over_w, num, rel,
                clamped ? "  <- step CAPPED at 0.05*||w||: could not reach the signal floor" : "");
        fd_rel.push_back(rel);
        fd_clamped.push_back(clamped);
        // `worst` is the verdict's headline number, so a capped probe must not
        // feed it — see the INCONCLUSIVE note below.
        if (!clamped) {
            worst = std::max(worst, rel);
        }
    }

    // The bar is MM3's, unchanged, and it is a VERDICT only under isolation.
    // Central differencing carries a genuine O(h^2 * f''') truncation error
    // that no amount of precision removes; 2e-2 leaves room for probe-to-probe
    // variation without admitting a real scale error — a wrong gradient scale
    // misses by a FACTOR, not by a percent.
    const double bar = isolated ? 2e-2 : 0.15;
    int          n_bad          = 0;
    int          n_inconclusive = 0;
    for (size_t i = 0; i < fd_rel.size(); i++) {
        // A CAPPED PROBE IS INCONCLUSIVE, NOT A FAILURE. Its step never
        // reached the loss change the forward can resolve, so its `rel` is
        // measuring the forward's noise floor and truncation error, not the
        // backward. Counting it as bad would print "do NOT train past it" over
        // a measurement that says nothing — the exact spurious verdict the cap
        // exists to prevent. It is subtracted from the verdict instead, and
        // said out loud below.
        if (fd_clamped[i]) {
            n_inconclusive++;
            continue;
        }
        // NaN fails: `!(r < bar)` rather than `r >= bar`, so a probe whose
        // numeric side came back NaN (a failed forward, or ||g|| == 0 with a
        // measurable loss change) counts against the gate instead of sliding
        // through a comparison that is false for NaN either way.
        if (!(fd_rel[i] < bar)) {
            n_bad++;
        }
    }
    const int n_checked = (int) fd_rel.size() - n_inconclusive;

    if (n_inconclusive) {
        fprintf(stderr,
                "\n[yue2-fd] %d/%zu probes INCONCLUSIVE: the step needed to move the loss by %.2e would\n"
                "[yue2-fd]   have exceeded 5%% of the tensor's own norm, which is outside the linear\n"
                "[yue2-fd]   regime a central difference assumes. Their ||g|| is too small to measure\n"
                "[yue2-fd]   this way (the 1e-2 B init at :610 is what suppresses it); raise that init or\n"
                "[yue2-fd]   probe another site — do NOT read a capped probe as a wrong gradient.\n",
                n_inconclusive, probes.size(), dl_min);
    }

    if (isolated) {
        const bool verdict_pass = (n_bad == 0 && n_checked > 0);
        fprintf(stderr,
                "\n[yue2-fd] GATE %s: finite differences, F32-isolated to %d NAR layers, bar %.0e, "
                "%d/%d measurable probes, worst %.4f (step raised on %d, capped on %d)\n",
                n_checked == 0 ? "NO VERDICT" : (verdict_pass ? "PASS" : "FAIL"), n_layers, bar,
                n_checked - n_bad, n_checked, worst, n_raised, n_inconclusive);
        if (n_checked == 0) {
            fprintf(stderr,
                    "[yue2-fd]   Every probe was capped, so NOTHING was measured. This is not a pass.\n");
        } else if (n_bad) {
            fprintf(stderr,
                    "[yue2-fd]   The analytic gradient disagrees with the measured loss change. With\n"
                    "[yue2-fd]   segments = 1 there is no second backward route to cross-check against,\n"
                    "[yue2-fd]   so this is the only gradient gate — do NOT train past it.\n");
        }
    } else {
        fprintf(stderr,
                "\n[yue2-fd] %d/%d measurable probes within %.0f%% (worst %.4f) — INDICATIVE ONLY (no F32 "
                "isolation).\n"
                "[yue2-fd]   Add --nar-layers 2 to turn this into a verdict.\n",
                n_checked - n_bad, n_checked, bar * 100.0, worst);
    }

    yue2_nt_train_ctx_free(&C);
    yue2_nt_kv_free(&kv);
    yue2_nt_f32_free(&iso);
    // No measurable probe is not a pass: exit non-zero so a script cannot read
    // "every probe was capped" as a green gate.
    return (isolated && (n_bad || n_checked == 0)) ? 1 : 0;
}

// ── The preprocess manifest ────────────────────────────────────────────────
//
// `ace-train yue2-preprocess` writes <dir>/yue2_preprocess.json: one record per
// CLIP, each naming the cached latent file it lives in, the frame offset into
// that file and how many frames it covers. The writer is
// train/yue2-preprocess-run.h:903-985 and this reader is checked against it,
// field for field. The obvious aliases are accepted anyway so a hand-written
// or third-party manifest loads, but nothing here guesses: where a field is
// absent, the assumption is stated at the point it is made — and where a wrong
// assumption could not be caught downstream (`latent_layout`), the field is
// required outright rather than defaulted.
//
// ── THE LAYOUT TRAP, because "channel-major" means opposite things ──────────
//
// The cache holds `latent_dim * source_frames` raw f32 at
//
//     index = t * latent_dim + c        (preprocess-run.h:90-97, and the
//                                        manifest says so: latent_layout =
//                                        "frame_major", latent_index =
//                                        "t * latent_dim + c")
//
// i.e. one contiguous row of LD per FRAME. That is ALREADY what the training
// graph wants: `x_t` and `vtarget` are ggml [LD, T], ne0 = LD, so their memory
// is `index = t * LD + c` (yue2-nar-train-graph.h:426-428 and :412-415,
// "[T, LD] channel-fastest"). So a clip is ONE CONTIGUOUS SPAN of the cache
// starting at `offset * LD`, and this reader does exactly one seek and one
// read — no per-channel loop, no transpose.
//
// The transpose exists, but it is the PREPROCESSOR's: yue2_vae_encode_tiled
// emits channel-major (`assembled[c * T + f]`, yue2-vae-encode.h:694-697) and
// preprocess-run.h:846-857 transposes once, at encode time, off the training
// hot path. Upstream caches the same way (data.py:118, `z.T` -> [frames, 64]).
//
// The word "channel-major" means opposite things in the two files, which is
// why the manifest STATES the layout and this reader REFUSES a manifest that
// does not, or that states one it cannot read, rather than inferring. Nothing
// downstream could catch the wrong guess: both orderings occupy exactly the
// same bytes, so the size checks below pass either way and the run would train
// on interleaved garbage at full speed. That confusion has already cost this
// project a project's worth of debugging once (the RVQ-encoder latent-layout
// trap); it is written down here rather than discovered again.
//
// OTHER ASSUMPTIONS:
//
//   * The clip array is `clips` (`samples`/`items` accepted).
//   * `latent_dtype` is "f32" in every v1 manifest; "f16" is read too, because
//     08 §0's recipe describes an fp16 cache and a future producer may switch.
//     An unknown spelling is refused rather than assumed.
//   * Paths are relative to the MANIFEST's directory unless absolute.
//   * `codec_ids` is optional (08 §5's seam, reserved and never written by v1).
//     A string is a manifest-relative path to little-endian i32 ids; an array
//     is the ids inline. RAW ids either way — the +YUE2_CODEC_OFFSET happens
//     in yue2_nar_train_cond_ids and nowhere else.
//   * A clip with no caption trains on the trigger word alone, which is
//     upstream's `caption_mode="none"` (nodes.py:88), not an error.

struct Yue2TrainClip {
    std::string          id;
    std::string          source;   // informational: the file the clip was cut from
    std::string          caption;
    std::string          lyrics;   // upstream always trains lyrics=""; honoured if present
    std::string          genre, bpm, key;  // sidecar fields for the style template
    std::string          latents;  // resolved path to the cached latent file
    int64_t              offset = 0;  // frame offset INTO that file
    int64_t              frames = 0;  // frames this clip covers
    // The SOURCE file's frame count. With the frame-major cache this is a
    // BOUNDS CHECK, not a stride — the row stride is `latent_dim` — which is
    // what preprocess-run.h:1028-1029 already calls it. It still has to be
    // right: a stale value is how a clip ends up reading past its own file.
    int64_t              source_frames = 0;
    int                  bits = 32;  // 32 = f32 cache (v1), 16 = f16
    std::vector<int32_t> codec_ids;  // RAW ids; empty = the text-only regime
};

struct Yue2TrainSet {
    std::vector<Yue2TrainClip> clips;
    int64_t                    latent_dim  = 0;    // 0 = the manifest did not say
    int64_t                    clip_frames = 0;    // 0 = the manifest did not say
    double                     frame_rate  = 0.0;  // 0 = the manifest did not say
    bool                       codec_ids_present = false;
    std::string                format;
    std::string                dir;
};

static yyjson_val * yue2_nt_jsv(yyjson_val * o, std::initializer_list<const char *> keys) {
    if (!o) {
        return nullptr;
    }
    for (const char * k : keys) {
        yyjson_val * v = yyjson_obj_get(o, k);
        if (v && !yyjson_is_null(v)) {
            return v;
        }
    }
    return nullptr;
}

static std::string yue2_nt_jstr(yyjson_val * o, std::initializer_list<const char *> keys) {
    yyjson_val * v = yue2_nt_jsv(o, keys);
    return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
}

static int64_t yue2_nt_jint(yyjson_val * o, std::initializer_list<const char *> keys, int64_t dflt,
                            bool * found = nullptr) {
    yyjson_val * v = yue2_nt_jsv(o, keys);
    if (found) {
        *found = (v != nullptr);
    }
    if (!v) {
        return dflt;
    }
    if (yyjson_is_int(v)) {
        return (int64_t) yyjson_get_sint(v);
    }
    if (yyjson_is_num(v)) {
        return (int64_t) yyjson_get_num(v);
    }
    if (yyjson_is_str(v)) {
        return (int64_t) atoll(yyjson_get_str(v));
    }
    if (found) {
        *found = false;
    }
    return dflt;
}

static double yue2_nt_jnum(yyjson_val * o, std::initializer_list<const char *> keys, double dflt) {
    yyjson_val * v = yue2_nt_jsv(o, keys);
    if (!v) {
        return dflt;
    }
    if (yyjson_is_num(v)) {
        return yyjson_get_num(v);
    }
    if (yyjson_is_str(v)) {
        return atof(yyjson_get_str(v));
    }
    return dflt;
}

static bool yue2_nt_is_abs(const std::string & p) {
    return (p.size() > 1 && p[1] == ':') || (!p.empty() && (p[0] == '/' || p[0] == '\\'));
}

static std::string yue2_nt_join(const std::string & dir, const std::string & p) {
    if (p.empty() || dir.empty() || yue2_nt_is_abs(p)) {
        return p;
    }
    return dir + "/" + p;
}

static std::string yue2_nt_dirname(const std::string & p) {
    const size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? std::string(".") : p.substr(0, s);
}

// 64-bit seek. A whole-album latent cache passes 2 GB easily, and `fseek`'s
// `long` is 32-bit on Windows — a silent wrap there would read the WRONG CLIP
// and train perfectly happily on it.
static int yue2_nt_seek64(FILE * f, int64_t off) {
#ifdef _WIN32
    return _fseeki64(f, off, SEEK_SET);
#else
    return fseek(f, (long) off, SEEK_SET);  // long is 64-bit on the POSIX targets
#endif
}

static int yue2_nt_dtype_bits(const std::string & s, int dflt) {
    if (s.empty()) {
        return dflt;
    }
    if (s == "f32" || s == "F32" || s == "float32" || s == "fp32") {
        return 32;
    }
    if (s == "f16" || s == "F16" || s == "float16" || s == "fp16" || s == "half") {
        return 16;
    }
    return -1;  // unknown: the caller refuses rather than guessing 16 and reading noise
}

// A raw little-endian i32 file of codec ids.
static bool yue2_nt_read_i32_file(const std::string & path, std::vector<int32_t> * out, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open codec_ids file " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || (n % 4) != 0) {
        fclose(f);
        *err = path + " is not a whole number of i32 codec ids (" + std::to_string((long long) n) + " bytes)";
        return false;
    }
    out->assign((size_t) (n / 4), 0);
    const bool ok = out->empty() || fread(out->data(), 4, out->size(), f) == out->size();
    fclose(f);
    if (!ok) {
        *err = "short read on " + path;
        return false;
    }
    return true;
}

static bool yue2_nt_load_manifest(const std::string & path, Yue2TrainSet * out, std::string * err) {
    yyjson_read_err rerr;
    yyjson_doc *    doc = yyjson_read_file(path.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        *err = "cannot parse " + path + ": " + (rerr.msg ? rerr.msg : "unknown error");
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        *err = path + ": root is not an object";
        return false;
    }
    out->dir         = yue2_nt_dirname(path);
    out->format      = yue2_nt_jstr(root, { "format" });
    out->latent_dim  = yue2_nt_jint(root, { "latent_dim", "latentDim", "channels" }, 0);
    out->clip_frames = yue2_nt_jint(root, { "clip_frames", "clipFrames", "frames_per_clip" }, 0);
    out->frame_rate =
        yue2_nt_jnum(root, { "frames_per_second", "frame_rate", "frameRate", "fps", "latent_fps" }, 0.0);
    {
        yyjson_val * cp        = yue2_nt_jsv(root, { "codec_ids_present" });
        out->codec_ids_present = cp && yyjson_is_bool(cp) && yyjson_get_bool(cp);
    }
    if (!out->format.empty() && out->format.compare(0, 16, "yue2-preprocess-") != 0) {
        fprintf(stderr, "[yue2-train] NOTE: %s says format=\"%s\"; this reader expects a yue2-preprocess-v1 "
                        "manifest and is going on field names alone\n",
                path.c_str(), out->format.c_str());
    }
    // The layout is STATED by the writer (preprocess-run.h:923) precisely so a
    // reader does not have to infer it, and it is MANDATORY here. An absent
    // field is refused rather than defaulted, because a default is a guess and
    // no check downstream can catch a wrong one: the two orderings are exactly
    // the same bytes, so the file-size check below passes either way, the
    // short-read check can never fire, and the run trains on interleaved
    // garbage at full speed with a plausible-looking loss curve. Every v1
    // manifest carries the field, so nothing legitimate is turned away.
    const std::string layout = yue2_nt_jstr(root, { "latent_layout" });
    if (layout.empty()) {
        yyjson_doc_free(doc);
        *err = path + ": no \"latent_layout\" — this reader will not guess. A v1 manifest always states "
                      "it (latent_layout=\"frame_major\", latent_index=\"t * latent_dim + c\"); re-run "
                      "`ace-train yue2-preprocess`, or add the field if the cache really is frame-major";
        return false;
    }
    if (layout != "frame_major") {
        yyjson_doc_free(doc);
        *err = path + ": latent_layout \"" + layout +
               "\" — this reader only knows \"frame_major\" (index = t * latent_dim + c), which is what "
               "yue2-preprocess writes and what the training graph's [LD, T] tensors already are";
        return false;
    }
    const std::string root_dtype = yue2_nt_jstr(root, { "latent_dtype", "dtype" });
    const int         root_bits  = yue2_nt_dtype_bits(root_dtype, 32);
    if (root_bits < 0) {
        yyjson_doc_free(doc);
        *err = path + ": latent_dtype \"" + root_dtype + "\" is not one this reader knows (f32 or f16)";
        return false;
    }
    // A latents directory, when a producer keeps the caches somewhere other
    // than beside the manifest. v1 does not write one: its per-clip `latents`
    // is already manifest-relative.
    const std::string lat_dir = yue2_nt_join(out->dir, yue2_nt_jstr(root, { "latents_dir", "latentsDir",
                                                                            "cache_dir" }));

    yyjson_val * arr = yue2_nt_jsv(root, { "clips", "samples", "items" });
    if (!arr || !yyjson_is_arr(arr)) {
        yyjson_doc_free(doc);
        *err = path + ": no clips[] array (also looked for samples[] and items[])";
        return false;
    }

    size_t          idx = 0, max = 0;
    yyjson_val *    it  = nullptr;
    std::string     ferr;
    yyjson_arr_foreach(arr, idx, max, it) {
        if (!yyjson_is_obj(it)) {
            continue;
        }
        Yue2TrainClip cl;
        cl.id      = yue2_nt_jstr(it, { "id", "clip_id", "name" });
        cl.source  = yue2_nt_jstr(it, { "source", "file", "audio", "filename", "path" });
        cl.caption = yue2_nt_jstr(it, { "caption", "text", "prompt", "style" });
        cl.lyrics  = yue2_nt_jstr(it, { "lyrics" });
        cl.genre   = yue2_nt_jstr(it, { "genre" });
        cl.bpm     = yue2_nt_jstr(it, { "bpm" });
        cl.key     = yue2_nt_jstr(it, { "key" });
        const std::string lat = yue2_nt_jstr(it, { "latents", "latent", "latent_file", "latents_file",
                                                   "latent_path", "cache" });
        if (lat.empty()) {
            ferr = "clip \"" + cl.id + "\" names no latent file (looked for latents/latent/latent_file)";
            break;
        }
        cl.latents = yue2_nt_is_abs(lat) ? lat : yue2_nt_join(lat_dir.empty() ? out->dir : lat_dir, lat);
        bool got_frames = false;
        cl.offset = yue2_nt_jint(it, { "offset_frames", "offset", "start", "start_frame", "frame_offset" }, 0);
        cl.frames = yue2_nt_jint(it, { "frames", "n_frames", "num_frames", "length", "clip_frames" }, 0,
                                 &got_frames);
        cl.source_frames = yue2_nt_jint(it, { "source_frames", "sourceFrames", "file_frames" }, 0);
        if (!got_frames || cl.frames <= 0) {
            // NOT defaulted to the run's clip length: a clip whose length the
            // manifest does not state is a clip whose end this reader cannot
            // find, and reading past it is a silent train-on-zeros.
            ferr = "clip \"" + cl.id + "\" does not state its frame count";
            break;
        }
        const std::string cd = yue2_nt_jstr(it, { "latent_dtype", "dtype" });
        cl.bits              = yue2_nt_dtype_bits(cd, root_bits);
        if (cl.bits < 0) {
            ferr = "clip \"" + cl.id + "\": dtype \"" + cd + "\" is not f16 or f32";
            break;
        }
        yyjson_val * ci = yue2_nt_jsv(it, { "codec_ids", "codecIds", "codec" });
        if (ci && yyjson_is_str(ci)) {
            const std::string cp = yue2_nt_join(out->dir, yyjson_get_str(ci));
            if (!yue2_nt_read_i32_file(cp, &cl.codec_ids, &ferr)) {
                break;
            }
        } else if (ci && yyjson_is_arr(ci)) {
            size_t       ci_i = 0, ci_n = 0;
            yyjson_val * cv = nullptr;
            yyjson_arr_foreach(ci, ci_i, ci_n, cv) {
                cl.codec_ids.push_back((int32_t) yyjson_get_sint(cv));
            }
        }
        if (cl.id.empty()) {
            cl.id = "clip" + std::to_string((long long) out->clips.size());
        }
        out->clips.push_back(std::move(cl));
    }
    yyjson_doc_free(doc);
    if (!ferr.empty()) {
        *err = path + ": " + ferr;
        return false;
    }
    if (out->clips.empty()) {
        *err = path + " holds no clips";
        return false;
    }

    // ── every cache file, once, BEFORE step 1 ──
    //
    // `source_frames` bounds every clip's read, so getting it wrong reads real
    // floats from the wrong place and trains without complaint. The file's
    // own size settles it: bytes / (latent_dim * elem) IS the source frame
    // count. Checking it here means a missing, truncated or stale cache is a
    // startup error instead of a failure 137 steps in.
    if (out->latent_dim > 0) {
        std::map<std::string, int64_t> file_frames;
        for (Yue2TrainClip & cl : out->clips) {
            auto it2 = file_frames.find(cl.latents);
            if (it2 == file_frames.end()) {
                long long bytes = 0, mtime = 0;
                if (!pm_stat_file(cl.latents, &bytes, &mtime)) {
                    *err = "cannot stat the latent cache " + cl.latents + " (clip \"" + cl.id + "\")";
                    return false;
                }
                const int64_t esz     = (cl.bits == 32) ? 4 : 2;
                const int64_t per_row = out->latent_dim * esz;
                if (per_row <= 0 || bytes % per_row != 0) {
                    *err = cl.latents + " is " + std::to_string(bytes) + " bytes, not a whole number of " +
                           std::to_string((long long) out->latent_dim) + "-channel frames — a truncated or "
                           "mismatched cache";
                    return false;
                }
                it2 = file_frames.emplace(cl.latents, (int64_t) (bytes / per_row)).first;
            }
            const int64_t have = it2->second;
            if (cl.source_frames <= 0) {
                cl.source_frames = have;  // manifest did not say; the file does
            } else if (cl.source_frames != have) {
                *err = "clip \"" + cl.id + "\": the manifest says its source holds " +
                       std::to_string((long long) cl.source_frames) + " frames, " + cl.latents + " holds " +
                       std::to_string((long long) have) +
                       " — re-run yue2-preprocess, the cache and the manifest are out of step";
                return false;
            }
            if (cl.offset < 0 || cl.offset + cl.frames > cl.source_frames) {
                *err = "clip \"" + cl.id + "\" spans frames " + std::to_string((long long) cl.offset) + ".." +
                       std::to_string((long long) (cl.offset + cl.frames)) + " of a " +
                       std::to_string((long long) cl.source_frames) + "-frame source";
                return false;
            }
        }
    }
    return true;
}

// One clip's latents, read straight out of the frame-major cache into the
// frame-major `[T, LD]` (index t*LD + c) that yue2_nt_make_xt_target and
// yue2_nt_host_inputs require. The on-disk order IS the wanted order, so this
// is ONE seek and ONE read, with no transpose — see the layout note above
// Yue2TrainClip for why the transpose lives in the preprocessor instead.
//
// `source_frames` is a bounds check here, not a stride: the row stride is LD.
static bool yue2_nt_read_clip(const Yue2TrainClip & cl, int64_t LD, int64_t T, std::vector<float> * z,
                              std::string * err) {
    z->assign((size_t) (T * LD), 0.0f);
    const int64_t have = cl.source_frames > 0 ? cl.source_frames : cl.frames;
    if (cl.offset < 0 || cl.offset + T > have) {
        *err = "clip \"" + cl.id + "\" has only " + std::to_string((long long) (have - cl.offset)) +
               " frames from its offset, " + std::to_string((long long) T) + " wanted";
        return false;
    }
    FILE * f = hs_fopen(cl.latents, "rb");
    if (!f) {
        *err = "cannot open " + cl.latents;
        return false;
    }
    const int64_t esz = (cl.bits == 32) ? 4 : 2;
    const size_t  n   = (size_t) (T * LD);
    bool          ok  = (yue2_nt_seek64(f, cl.offset * LD * esz) == 0);
    if (ok) {
        if (cl.bits == 32) {
            ok = fread(z->data(), sizeof(float), n, f) == n;
        } else {
            std::vector<ggml_fp16_t> half(n);
            ok = fread(half.data(), sizeof(ggml_fp16_t), n, f) == n;
            if (ok) {
                for (size_t i = 0; i < n; i++) {
                    (*z)[i] = ggml_fp16_to_fp32(half[i]);
                }
            }
        }
    }
    fclose(f);
    if (!ok) {
        *err = "short read on " + cl.latents + " (clip \"" + cl.id + "\", offset " +
               std::to_string((long long) cl.offset) + ", source " + std::to_string((long long) have) +
               " frames, " + std::to_string((long long) T) +
               " wanted) — the manifest and the cache disagree about this file";
        return false;
    }
    return true;
}

// ── The AR-prefix K/V cache ────────────────────────────────────────────────
//
// OUR ADVANTAGE OVER UPSTREAM, and the contract says so (09 §1): upstream
// re-runs the whole conditioned sequence every step and caches only the token
// IDS (train.py:186-190, `cond_cache`); we prefill the AR K/V once per
// distinct conditioning and keep the widened F32 canvas, so a step costs the
// NAR half alone.
//
// The canvas is ~104 MB at the default clip length (28 layers x 2 x
// [128, S_kv, 8] F32), so this is a VRAM knob, not a lookup table. BOUNDED at
// `--kv-cache` entries; over that the LEAST RECENTLY USED canvas is freed and
// rebuilt on demand. A rebuild is one AR prefill — real work, but bounded and
// loud (the eviction counter is printed at the end), never a wrong answer.
//
// With the text-only regime there are as many entries as distinct captions
// plus one for the dropout prefix, which is what "there will be few" means.
// With per-clip `codec_ids` every clip is its own conditioning, the cache
// thrashes by construction, and the loop says so once rather than quietly
// spending an AR prefill per step.
struct Yue2NtKvEntry {
    std::string    key;
    Yue2NarTrainKv kv;
    uint64_t       used = 0;
};

struct Yue2NtKvCache {
    std::vector<Yue2NtKvEntry *> e;
    size_t                       cap       = 8;
    uint64_t                     tick      = 0;
    int64_t                      builds    = 0;
    int64_t                      hits      = 0;
    int64_t                      evictions = 0;
    size_t                       bytes     = 0;  // per entry, measured on the first build
};

static void yue2_nt_kvcache_free(Yue2NtKvCache * C) {
    for (Yue2NtKvEntry * p : C->e) {
        yue2_nt_kv_free(&p->kv);
        delete p;
    }
    C->e.clear();
}

static Yue2NarTrainKv * yue2_nt_kvcache_get(Yue2NtKvCache * C, const Yue2Model & m, const std::string & key,
                                            const std::vector<int32_t> & ar_ids, int64_t T, int n_layers,
                                            std::string * err) {
    C->tick++;
    for (Yue2NtKvEntry * p : C->e) {
        if (p->key == key) {
            p->used = C->tick;
            C->hits++;
            return &p->kv;
        }
    }
    // Room first, so the eviction happens BEFORE the new canvas is allocated
    // rather than on top of it — the peak is cap entries, not cap + 1.
    while (C->e.size() >= C->cap && !C->e.empty()) {
        size_t oldest = 0;
        for (size_t i = 1; i < C->e.size(); i++) {
            if (C->e[i]->used < C->e[oldest]->used) {
                oldest = i;
            }
        }
        yue2_nt_kv_free(&C->e[oldest]->kv);
        delete C->e[oldest];
        C->e.erase(C->e.begin() + (long) oldest);
        C->evictions++;
    }

    const int64_t ctx_len = (int64_t) m.lm_cfg.context_length;
    if (ctx_len > 0 && (int64_t) ar_ids.size() + T + 2 > ctx_len) {
        char b[224];
        snprintf(b, sizeof(b),
                 "conditioning (%zu ids) + %lld latent rows exceeds the LM context (%lld). Shorten the "
                 "caption/lyrics or the clip.",
                 ar_ids.size(), (long long) (T + 2), (long long) ctx_len);
        *err = b;
        return nullptr;
    }

    Yue2NarChunk chunk;
    if (!yue2_nar_chunk_init(m, ar_ids, T, &chunk, err)) {
        return nullptr;
    }
    Yue2NtKvEntry * ne = new Yue2NtKvEntry();
    ne->key            = key;
    ne->used           = C->tick;
    if (!yue2_nt_kv_from_chunk(m, chunk, n_layers, &ne->kv, err)) {
        yue2_nar_chunk_free(&chunk);
        delete ne;
        return nullptr;
    }
    yue2_nar_chunk_free(&chunk);  // the F16 AR copy is dead weight once widened
    if (!C->bytes) {
        const int64_t D   = (int64_t) m.lm_cfg.key_length;
        const int64_t Nkv = (int64_t) m.lm_cfg.head_count_kv;
        C->bytes = (size_t) ((int64_t) n_layers * 2 * D * ne->kv.s_kv * Nkv * 4);
    }
    C->e.push_back(ne);
    C->builds++;
    return &ne->kv;
}

// ── Conditioning ───────────────────────────────────────────────────────────

// upstream train.py:192 — the trigger word IS the style when there is no
// caption, and prefixes it when there is.
static std::string yue2_nt_style(const std::string & trigger, const std::string & caption) {
    if (caption.empty()) {
        return trigger;
    }
    if (trigger.empty()) {
        return caption;
    }
    return trigger + ", " + caption;
}

// ── Deterministic per-micro-step draws ─────────────────────────────────────
//
// See the file header: every draw is a pure function of (seed, micro-step,
// purpose), which is what makes resume exact. splitmix64's own mixing
// function, so two nearby (k, tag) pairs decorrelate.
static inline uint64_t yue2_nt_seed_mix(uint64_t seed, uint64_t k, uint64_t tag) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ull * (k + 1) + 0xD1B54A32D192ED03ull * (tag + 1);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

enum {
    YUE2_NT_TAG_CLIP    = 1,
    YUE2_NT_TAG_DROPOUT = 2,
    YUE2_NT_TAG_T       = 3,
    YUE2_NT_TAG_NOISE   = 4,
};

// ── Learning rate ──────────────────────────────────────────────────────────
//
// upstream's `lr_at` (train.py:181-188), to the step: linear warmup over
// `warmup` steps then a cosine to ZERO, or a constant after warmup.
//
// LmOptim's own schedule is NEUTRALISED rather than approximated. lm_optim_step
// computes `base_lr * lm_lr_lambda(opt_step, total_steps, warmup_steps,
// lr_floor)`, and lm_lr_lambda's warmup arm is `step / warmup` where upstream's
// is `(step + 1) / warmup` — so step 0 would run at lr 0, and the cosine would
// bottom at 10% of base rather than 0. Setting {lr_floor 1, total 1, warmup 0}
// makes that lambda identically 1.0 (floor + (1-floor)*... with floor == 1), and
// `base_lr`, set per step from here, then carries the whole schedule exactly.
static double yue2_nt_lr_at(const Yue2NarTrainArgs & a, int64_t step0) {
    const double lr = (double) a.lr;
    if (a.warmup > 0 && step0 < a.warmup) {
        return lr * (double) (step0 + 1) / (double) a.warmup;
    }
    if (a.lr_scheduler != "cosine") {
        return lr;
    }
    const double denom = (double) std::max<int64_t>(1, a.steps - a.warmup);
    const double ratio = std::min(1.0, (double) (step0 - a.warmup) / denom);
    return lr * 0.5 * (1.0 + std::cos(3.14159265358979 * ratio));
}

// ── Checkpoint / resume ────────────────────────────────────────────────────
//
// IS RESUME BIT-EXACT? Yes, within one machine + build, and the qualifier is
// the whole point of writing it down:
//
//   * What is persisted: every LoRA factor, every AdamW moment (m and v), the
//     optimizer's step and iteration counters (opt_iter drives the bias
//     correction, so dropping it would silently re-warm Adam), and the step
//     counter itself. Tensors travel BY NAME and a name in the file that is
//     not in the live run — or the reverse — is an error, not a shrug.
//   * What does NOT need persisting: the RNG. Every draw is a pure function of
//     (seed, micro-step index, purpose) (file header), so the data stream after
//     a resume is exactly the stream the uninterrupted run would have seen.
//     This is the one place this trainer is simpler than mm3-lm-resume.h,
//     which has to carry an RNG, an epoch order and a cursor.
//   * What BREAKS it: a different GPU, driver, ggml build or backend
//     (reductions reassociate), a different base GGUF or quant, or a changed
//     manifest (clip order is part of the data stream). None of those is
//     visible from inside this file, so they are listed here and nowhere else.
//   * What the FINGERPRINT catches, because it is everything else the CLI can
//     change: rank, param count, layer count, target, grad-accum, frames,
//     alpha, seed — and `cond_hash`, which folds in --trigger, --lyrics,
//     --t-sampling and --caption-dropout. Those four looked like "just
//     conditioning" and are not: the first two ARE the prefix the model is
//     trained to answer to (and --trigger is written into the exported
//     metadata, so half a run under one trigger would export as if it were all
//     the other), and the last two re-parameterise draws off the same seeded
//     stream, which moves the data a resume sees while leaving the loss curve
//     looking untouched. --style is deliberately out: it reaches the FD gate
//     only, never this loop.
//   * What is a NOTE rather than a refusal: --steps, --lr, --warmup,
//     --lr-scheduler, --weight-decay and --max-grad-norm. Each only reshapes
//     the schedule or the update rule from here on, which is a legitimate
//     thing to want mid-run; the resumed run is then not bit-identical to the
//     uninterrupted one and the note says so out loud.
//   * The state file is single-machine, host-endian and NOT a distribution
//     format — it is written and read by the same binary minutes apart.

static const char     YUE2_NT_CKPT_MAGIC[8] = { 'Y', '2', 'N', 'A', 'R', 'C', 'K', '1' };
static const uint32_t YUE2_NT_CKPT_VERSION  = 2;  // v2 added cond_hash + the schedule knobs

struct Yue2NtCkptState {
    // Fingerprint. A resume into a different run shape is refused, not fudged.
    int32_t  rank = 0, n_params = 0, n_layers = 0, target = 0;
    int32_t  grad_accum = 0, frames = 0, total_steps = 0;
    float    alpha = 0.0f, lr = 0.0f;
    uint64_t seed = 0;
    // Everything about the CONDITIONING and the draw stream that argv can
    // change, folded into one value so adding a knob is a one-line change to
    // yue2_nt_cond_hash rather than another field and another comparison.
    uint64_t cond_hash = 0;
    // Schedule / update-rule knobs. NOTE-on-mismatch, like total_steps and lr:
    // they change the run from here on without invalidating the weights.
    int32_t  warmup = 0, lr_sched = 0;  // lr_sched: 0 = cosine, 1 = constant
    float    weight_decay = 0.0f, max_grad_norm = 0.0f;
    // Position in the run.
    int32_t  steps_done = 0;
    double   loss_sum   = 0.0;  // summed micro-step loss, for the running mean
    int64_t  n_micro    = 0;
    int32_t  opt_step = 0, opt_iter = 0;
};

// FNV-1a over the knobs that decide WHAT is trained rather than how fast.
// Floats go in by their exact bits, not a formatted string, so 0.1f and
// 0.100001f are different runs and say so.
static uint64_t yue2_nt_cond_hash(const Yue2NarTrainArgs & a) {
    uint64_t h = 1469598103934665603ull;
    auto     mix_bytes = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t) b[i];
            h *= 1099511628211ull;
        }
    };
    // A length prefix and a separator, so ("ab","c") and ("a","bc") differ.
    auto mix_str = [&](const std::string & s) {
        const uint32_t n = (uint32_t) s.size();
        mix_bytes(&n, sizeof(n));
        mix_bytes(s.data(), s.size());
        const uint8_t sep = 0xFF;
        mix_bytes(&sep, 1);
    };
    mix_str(a.trigger);       // prefixes every caption (yue2_style_string)
    mix_str(a.style_template);
    mix_str(a.lyrics);        // the fallback when a clip carries none
    mix_str(a.t_sampling);    // re-parameterises t off the same seeded stream
    mix_bytes(&a.caption_dropout, sizeof(a.caption_dropout));  // ditto, the dropout draw
    // NOT --style. It reaches the FD gate only (yue2_nar_fdcheck_main builds
    // its prefix from it); the loop builds style from --trigger + the clip's
    // own caption and never reads a.style. Fingerprinting it would refuse a
    // resume over a knob that provably changed nothing.
    return h;
}

template <typename T>
static bool yue2_nt_w(FILE * f, const T & v) {
    return fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
static bool yue2_nt_r(FILE * f, T * v) {
    return fread(v, sizeof(T), 1, f) == 1;
}

static bool yue2_nt_w_str(FILE * f, const std::string & s) {
    const uint32_t n = (uint32_t) s.size();
    return yue2_nt_w(f, n) && (n == 0 || fwrite(s.data(), 1, n, f) == n);
}

static bool yue2_nt_r_str(FILE * f, std::string * s) {
    uint32_t n = 0;
    if (!yue2_nt_r(f, &n) || n > 4096) {
        return false;
    }
    s->assign(n, '\0');
    return n == 0 || fread(&(*s)[0], 1, n, f) == n;
}

static bool yue2_nt_w_tensor(FILE * f, ggml_tensor * t, std::vector<uint8_t> * scratch) {
    const size_t bytes = ggml_nbytes(t);
    scratch->resize(bytes);
    ggml_backend_tensor_get(t, scratch->data(), 0, bytes);
    const uint64_t n64 = (uint64_t) bytes;
    return yue2_nt_w_str(f, std::string(ggml_get_name(t))) && yue2_nt_w(f, n64) &&
           fwrite(scratch->data(), 1, bytes, f) == bytes;
}

static std::string yue2_nt_ckpt_path(const std::string & out_dir) {
    return out_dir + "/yue2_nar_ckpt.bin";
}

// .tmp + rename, so a crash mid-write cannot leave a truncated file that a
// resume would half-believe.
static bool yue2_nt_ckpt_save(const std::string & path, const Yue2NtCkptState & st, const Yue2NtTrainCtx & C,
                              std::string * err) {
    const std::string tmp = path + ".tmp";
    FILE *            f   = hs_fopen(tmp, "wb");
    if (!f) {
        *err = "cannot open " + tmp + " for writing";
        return false;
    }
    std::vector<uint8_t> scratch;
    bool                 ok = fwrite(YUE2_NT_CKPT_MAGIC, 1, sizeof(YUE2_NT_CKPT_MAGIC), f) ==
              sizeof(YUE2_NT_CKPT_MAGIC);
    ok = ok && yue2_nt_w(f, YUE2_NT_CKPT_VERSION);
    ok = ok && yue2_nt_w(f, st.rank) && yue2_nt_w(f, st.n_params) && yue2_nt_w(f, st.n_layers);
    ok = ok && yue2_nt_w(f, st.target) && yue2_nt_w(f, st.grad_accum) && yue2_nt_w(f, st.frames);
    ok = ok && yue2_nt_w(f, st.total_steps) && yue2_nt_w(f, st.alpha) && yue2_nt_w(f, st.lr);
    ok = ok && yue2_nt_w(f, st.seed) && yue2_nt_w(f, st.cond_hash);
    ok = ok && yue2_nt_w(f, st.warmup) && yue2_nt_w(f, st.lr_sched);
    ok = ok && yue2_nt_w(f, st.weight_decay) && yue2_nt_w(f, st.max_grad_norm);
    ok = ok && yue2_nt_w(f, st.steps_done) && yue2_nt_w(f, st.loss_sum) && yue2_nt_w(f, st.n_micro);
    ok = ok && yue2_nt_w(f, st.opt_step) && yue2_nt_w(f, st.opt_iter);
    {
        const uint32_t n = (uint32_t) C.params.size();
        ok               = ok && yue2_nt_w(f, n);
        for (size_t j = 0; ok && j < C.params.size(); j++) {
            ok = yue2_nt_w_tensor(f, C.params[j], &scratch);
        }
    }
    {
        uint32_t n = 0;
        for (size_t j = 0; j < C.opt.mom_m.size(); j++) {
            if (C.opt.mom_m[j]) {
                n++;
            }
            if (j < C.opt.mom_v.size() && C.opt.mom_v[j]) {
                n++;
            }
        }
        ok = ok && yue2_nt_w(f, n);
        for (size_t j = 0; ok && j < C.opt.mom_m.size(); j++) {
            if (C.opt.mom_m[j]) {
                ok = ok && yue2_nt_w_tensor(f, C.opt.mom_m[j], &scratch);
            }
            if (ok && j < C.opt.mom_v.size() && C.opt.mom_v[j]) {
                ok = yue2_nt_w_tensor(f, C.opt.mom_v[j], &scratch);
            }
        }
    }
    const uint32_t eof_marker = 0xD09EF00Du;  // "done", so a truncated tail is loud
    ok                        = ok && yue2_nt_w(f, eof_marker);
    if (fclose(f) != 0) {
        ok = false;
    }
    if (!ok) {
        *err = "write failed (disk full?) — " + tmp;
        hs_remove(tmp);
        return false;
    }
    hs_remove(path);
    if (hs_rename(tmp, path) != 0) {
        *err = "cannot rename " + tmp + " into place";
        return false;
    }
    return true;
}

static bool yue2_nt_ckpt_load(const std::string & path, const Yue2NtCkptState & want, Yue2NtCkptState * got,
                              Yue2NtTrainCtx * C, std::string * err) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        *err = "cannot open " + path + " — --resume needs the checkpoint --save-every wrote";
        return false;
    }
    auto fail = [&](const std::string & why) {
        *err = path + ": " + why;
        fclose(f);
        return false;
    };
    char magic[sizeof(YUE2_NT_CKPT_MAGIC)];
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, YUE2_NT_CKPT_MAGIC, sizeof(magic)) != 0) {
        return fail("not a yue2-nar-train checkpoint");
    }
    uint32_t ver = 0;
    if (!yue2_nt_r(f, &ver) || ver != YUE2_NT_CKPT_VERSION) {
        return fail("checkpoint version " + std::to_string(ver) + ", this build writes " +
                    std::to_string(YUE2_NT_CKPT_VERSION));
    }
    Yue2NtCkptState st;
    bool            ok = yue2_nt_r(f, &st.rank) && yue2_nt_r(f, &st.n_params) && yue2_nt_r(f, &st.n_layers) &&
              yue2_nt_r(f, &st.target) && yue2_nt_r(f, &st.grad_accum) && yue2_nt_r(f, &st.frames) &&
              yue2_nt_r(f, &st.total_steps) && yue2_nt_r(f, &st.alpha) && yue2_nt_r(f, &st.lr) &&
              yue2_nt_r(f, &st.seed) && yue2_nt_r(f, &st.cond_hash) && yue2_nt_r(f, &st.warmup) &&
              yue2_nt_r(f, &st.lr_sched) && yue2_nt_r(f, &st.weight_decay) &&
              yue2_nt_r(f, &st.max_grad_norm) && yue2_nt_r(f, &st.steps_done) &&
              yue2_nt_r(f, &st.loss_sum) && yue2_nt_r(f, &st.n_micro) && yue2_nt_r(f, &st.opt_step) &&
              yue2_nt_r(f, &st.opt_iter);
    if (!ok) {
        return fail("truncated header");
    }
    // Every one of these changes what the run IS. Resuming across one would
    // look like training and be something else.
    if (st.rank != want.rank || st.n_params != want.n_params || st.n_layers != want.n_layers ||
        st.target != want.target || st.grad_accum != want.grad_accum || st.frames != want.frames ||
        st.seed != want.seed || st.alpha != want.alpha) {
        char b[320];
        snprintf(b, sizeof(b),
                 "was written by a different run (rank %d/%d, params %d/%d, layers %d/%d, target %d/%d, "
                 "grad-accum %d/%d, frames %d/%d, seed %llu/%llu, alpha %.3f/%.3f) — start a new --out "
                 "rather than resuming into it",
                 st.rank, want.rank, st.n_params, want.n_params, st.n_layers, want.n_layers, st.target,
                 want.target, st.grad_accum, want.grad_accum, st.frames, want.frames,
                 (unsigned long long) st.seed, (unsigned long long) want.seed, (double) st.alpha,
                 (double) want.alpha);
        return fail(b);
    }
    if (st.cond_hash != want.cond_hash) {
        // Separate message because the fix is different: nothing about the run
        // SHAPE is wrong, the conditioning or the draw stream moved. Half a run
        // under one trigger word and half under another is not the adapter
        // either half describes, and the export would claim the second.
        return fail("was written with different conditioning or sampling — one of --trigger, --lyrics, "
                    "--t-sampling or --caption-dropout has changed since. Put them back, or start a new "
                    "--out; resuming across them trains an adapter neither set describes");
    }
    if (st.total_steps != want.total_steps || st.lr != want.lr || st.warmup != want.warmup ||
        st.lr_sched != want.lr_sched || st.weight_decay != want.weight_decay ||
        st.max_grad_norm != want.max_grad_norm) {
        // These only reshape the SCHEDULE and the update rule, both recomputed
        // from the args each step. Loud, because the resumed run is then not
        // the run the checkpoint came from — but not fatal, because extending
        // or re-tuning a run is a legitimate thing to want.
        fprintf(stderr,
                "[yue2-train] NOTE: resuming with --steps %d / --lr %.3g / --warmup %d / --lr-scheduler %s "
                "/ --weight-decay %.3g / --max-grad-norm %.3g into a checkpoint written at %d / %.3g / %d / "
                "%s / %.3g / %.3g. The schedule and the update rule change from here; the weights do not.\n",
                want.total_steps, (double) want.lr, want.warmup, want.lr_sched ? "constant" : "cosine",
                (double) want.weight_decay, (double) want.max_grad_norm, st.total_steps, (double) st.lr,
                st.warmup, st.lr_sched ? "constant" : "cosine", (double) st.weight_decay,
                (double) st.max_grad_norm);
    }

    std::unordered_map<std::string, ggml_tensor *> live;
    for (ggml_tensor * t : C->params) {
        live[ggml_get_name(t)] = t;
    }
    for (size_t j = 0; j < C->opt.mom_m.size(); j++) {
        if (C->opt.mom_m[j]) {
            live[ggml_get_name(C->opt.mom_m[j])] = C->opt.mom_m[j];
        }
        if (j < C->opt.mom_v.size() && C->opt.mom_v[j]) {
            live[ggml_get_name(C->opt.mom_v[j])] = C->opt.mom_v[j];
        }
    }
    std::vector<uint8_t> scratch;
    int                  restored = 0;
    for (int block = 0; block < 2; block++) {
        uint32_t n = 0;
        if (!yue2_nt_r(f, &n)) {
            return fail("truncated tensor block header");
        }
        for (uint32_t i = 0; i < n; i++) {
            std::string name;
            uint64_t    bytes = 0;
            if (!yue2_nt_r_str(f, &name) || !yue2_nt_r(f, &bytes)) {
                return fail("truncated tensor record");
            }
            auto lit = live.find(name);
            if (lit == live.end()) {
                return fail("holds tensor \"" + name + "\", which this run does not have");
            }
            if (ggml_nbytes(lit->second) != (size_t) bytes) {
                return fail("tensor \"" + name + "\" is the wrong size");
            }
            scratch.resize((size_t) bytes);
            if (fread(scratch.data(), 1, (size_t) bytes, f) != (size_t) bytes) {
                return fail("truncated data for \"" + name + "\"");
            }
            ggml_backend_tensor_set(lit->second, scratch.data(), 0, (size_t) bytes);
            restored++;
        }
    }
    uint32_t eof_marker = 0;
    if (!yue2_nt_r(f, &eof_marker) || eof_marker != 0xD09EF00Du) {
        return fail("no end marker — the file is truncated");
    }
    fclose(f);
    if (restored != (int) live.size()) {
        *err = path + ": restored " + std::to_string(restored) + " of " + std::to_string(live.size()) +
               " live tensors — a partial restore is indistinguishable from training on garbage";
        return false;
    }
    C->opt.opt_step = st.opt_step;
    C->opt.opt_iter = st.opt_iter;
    *got            = st;
    return true;
}

// ── Export ─────────────────────────────────────────────────────────────────
//
// THE HALF THAT MUST NOT BE WRONG. The key spellings here are the ones
// `yue2-adapter.h` parses, checked against that file rather than against the
// plan (10-adapter-merge-notes.md §3 lists where the plan's original spellings
// disagreed with the code on both sides, and 08 §4 carries the correction):
//
//   * `yue2.` prefix        -> stripped by yue2_lora_target (yue2-adapter.h:261)
//   * `.lora_A.weight` /
//     `.lora_B.weight`      -> the FIRST entry of each list in
//                              yue2_lora_suffix (yue2-adapter.h:183-186). The
//                              lowercase `.lora_a` form the plan originally
//                              specified is accepted by that parser only as a
//                              tolerated legacy spelling; we write the
//                              canonical one.
//   * CAPITALISATION IS LOAD-BEARING on both sides: yue2_nt_init_adapters keys
//     kaiming-vs-zero init on the tensor name's last character (graph.h:880),
//     and the parser's A/B lists are case-sensitive.
//   * `time_embd.{0,1}`     -> NOT `time_embed.{0,2}`. That is the GGUF's own
//                              spelling (yue2-model.h:874/:876) and the
//                              trainer's tag (graph.h:798-799); the parser
//                              remaps the old one WITH A WARNING
//                              (yue2-adapter.h:294-303), which we must not
//                              trip.
//   * module names          -> `blk.N.nar_attn_{q,k,v,output}`,
//                              `blk.N.nar_ffn_{gate,up,down}`, `vae2llm`,
//                              `llm2vae`, `time_embd.{0,1}` — exactly
//                              yue2_lora_site_family's NAR list, and exactly
//                              the trainer's own tags, so the mapping is
//                              "prefix with yue2., suffix with .lora_X.weight"
//                              and nothing else. The AR twins (`attn_q` without
//                              the `nar_`) are legal keys there too now, so the
//                              `format: "yue2-nar-lora-v1"` this exporter
//                              writes is what makes a mis-spelled key an error
//                              rather than a silent merge into the other half.
//
// SHAPES. safetensors shapes are PyTorch order, outermost first, and ggml's ne
// is the reverse — so a ggml A of [in, rank] IS torch [rank, in], and a ggml B
// of [rank, out] IS torch [out, rank]. The bytes need no transposition, only
// the shape array reversed. That is what the parser's preflight checks
// (yue2-adapter.h:670-676: A.shape[1] == base ne0, B.shape[0] == base ne1,
// B.shape[1] == A.shape[0]).
//
// ALPHA. adapter-merge.h's convention is a baked per-module `.alpha` scalar,
// else an adapter_config.json sidecar. We write neither fiction: `alpha` goes
// in `__metadata__`, which yue2_adapter_read_meta parses itself
// (yue2-adapter.h:370-397) and which feeds the same alpha/rank scaling
// (:587). st_write_file writes every metadata value as a STRING and that
// reader accepts both spellings, so a quoted number is correct here.
static bool yue2_nt_export(const Yue2TrainAdapters & ad, const Yue2NarTrainArgs & a, Yue2NtTarget target,
                           int64_t steps_done, int64_t clip_frames, const std::string & base_id,
                           const std::string & path, std::string * err) {
    struct Ent {
        std::string         name;
        const ggml_tensor * t;
    };
    std::vector<Ent> ents;
    auto             add_site = [&](const Yue2TrainLora & lo, const std::string & mod) {
        if (!lo.on()) {
            return;
        }
        ents.push_back({ "yue2." + mod + ".lora_A.weight", lo.a });
        ents.push_back({ "yue2." + mod + ".lora_B.weight", lo.b });
    };
    for (size_t i = 0; i < ad.blk.size(); i++) {
        const std::string p = "blk." + std::to_string((long long) i) + ".";
        add_site(ad.blk[i].q, p + "nar_attn_q");
        add_site(ad.blk[i].k, p + "nar_attn_k");
        add_site(ad.blk[i].v, p + "nar_attn_v");
        add_site(ad.blk[i].o, p + "nar_attn_output");
        add_site(ad.blk[i].gate, p + "nar_ffn_gate");
        add_site(ad.blk[i].up, p + "nar_ffn_up");
        add_site(ad.blk[i].down, p + "nar_ffn_down");
    }
    add_site(ad.vae2llm, "vae2llm");
    add_site(ad.llm2vae, "llm2vae");
    add_site(ad.time0, "time_embd.0");
    add_site(ad.time1, "time_embd.1");
    if (ents.empty()) {
        *err = "nothing to export: no LoRA site is active";
        return false;
    }

    // The blobs must outlive st_write_file, and STWTensor holds a bare
    // pointer into them — reserve so no push_back can reallocate the storage
    // out from under an already-recorded pointer.
    std::vector<std::vector<float>> blobs;
    blobs.reserve(ents.size());
    std::vector<STWTensor> tens;
    tens.reserve(ents.size());
    for (const Ent & e : ents) {
        const int64_t n0 = e.t->ne[0], n1 = e.t->ne[1];
        blobs.emplace_back((size_t) (n0 * n1), 0.0f);
        ggml_backend_tensor_get(const_cast<ggml_tensor *>(e.t), blobs.back().data(), 0,
                                (size_t) (n0 * n1) * sizeof(float));
        STWTensor t;
        t.name  = e.name;
        t.shape = { n1, n0 };  // torch order = ggml ne reversed
        t.data  = blobs.back().data();
        tens.push_back(std::move(t));
    }

    char buf[64];
    std::vector<std::pair<std::string, std::string>> md;
    md.emplace_back("format", "yue2-nar-lora-v1");
    snprintf(buf, sizeof(buf), "%lld", (long long) a.rank);
    md.emplace_back("rank", buf);
    snprintf(buf, sizeof(buf), "%.6f", (double) a.alpha);
    md.emplace_back("alpha", buf);
    md.emplace_back("targets", yue2_nt_target_name(target));
    snprintf(buf, sizeof(buf), "%lld", (long long) steps_done);
    md.emplace_back("steps", buf);
    snprintf(buf, sizeof(buf), "%lld", (long long) clip_frames);
    md.emplace_back("clip_frames", buf);
    md.emplace_back("trigger", a.trigger);
    md.emplace_back("style_template", a.style_template);
    // No sha is computed anywhere in this tree, so `base_sha` carries the
    // identity we actually have: the base LM file it trained against. The
    // loader only prints it (yue2-adapter.h:504-508), so a breadcrumb is
    // exactly what it is for — and an invented hash would be worse than none.
    md.emplace_back("base_sha", base_id);
    md.emplace_back("cot", "off");

    if (!st_write_file(path.c_str(), tens, md, STW_F32)) {
        *err = "safetensors write failed for " + path;
        return false;
    }
    fprintf(stderr, "[yue2-train] exported %zu tensors -> %s\n", tens.size(), path.c_str());
    return true;
}

// ── The loop ───────────────────────────────────────────────────────────────

static int yue2_nar_train_loop(const Yue2NarTrainArgs & a) {
    std::string err;

    // ── arguments ──
    if (a.manifest.empty()) {
        fprintf(stderr,
                "ace-train yue2-nar-train: --manifest <yue2_preprocess.json> is required for training.\n"
                "  Build one with `ace-train yue2-preprocess`, or run the gradient gate instead:\n"
                "      ace-train yue2-nar-train --lm <yue2-lm-*.gguf> --fd-check 6 --nar-layers 2\n");
        return 2;
    }
    if (a.out_dir.empty()) {
        fprintf(stderr, "ace-train yue2-nar-train: --out <dir> is required (checkpoints + the adapter)\n");
        return 2;
    }
    if (a.steps <= 0 || a.grad_accum <= 0 || a.rank <= 0) {
        fprintf(stderr, "ace-train yue2-nar-train: --steps, --grad-accum and --rank must all be > 0\n");
        return 2;
    }
    if (a.caption_dropout < 0.0f || a.caption_dropout > 1.0f) {
        fprintf(stderr, "ace-train yue2-nar-train: --caption-dropout must be in [0, 1]\n");
        return 2;
    }
    if (a.lr_scheduler != "cosine" && a.lr_scheduler != "constant") {
        fprintf(stderr, "ace-train yue2-nar-train: --lr-scheduler must be cosine or constant\n");
        return 2;
    }
    Yue2NtTarget target = YUE2_NT_ATTN_MLP;
    if (!yue2_nt_parse_target(a.target, &target)) {
        fprintf(stderr, "ace-train yue2-nar-train: --target must be nar_attn, nar_attn_mlp or "
                        "nar_attn_mlp_proj\n");
        return 2;
    }

    // ── dataset ──
    Yue2TrainSet ds;
    if (!yue2_nt_load_manifest(a.manifest, &ds, &err)) {
        fprintf(stderr, "[yue2-train] %s\n", err.c_str());
        return 1;
    }

    static Yue2Model m;
    if (!yue2_nt_open_model(&m, a, "yue2-train", &err)) {
        fprintf(stderr, "[yue2-train] %s\n", err.c_str());
        return 1;
    }
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        LD = (int64_t) c.latent_dim;
    if (ds.latent_dim > 0 && ds.latent_dim != LD) {
        fprintf(stderr,
                "[yue2-train] the manifest says latent_dim %lld, this LM wants %lld — the cache was written "
                "against a different VAE\n",
                (long long) ds.latent_dim, (long long) LD);
        return 1;
    }

    // ── clip length ──
    //
    // 25 latent frames/s = sample_rate 48000 / downsampling_ratio 1920
    // (03-reference-numerics.md §0). It is NOT read off the model here because
    // the trainer loads the LM only (want_vae=false), so yue2vae.* is not
    // populated; the manifest's own frame_rate wins when it states one.
    const double  fps = ds.frame_rate > 0.0 ? ds.frame_rate : 25.0;
    int64_t       T   = a.frames;
    const char *  T_from = "--frames";
    if (!a.frames_set) {
        if (ds.clip_frames > 0) {
            T      = ds.clip_frames;
            T_from = "the manifest's clip_frames";
        } else if (a.clip_seconds > 0.0) {
            T      = (int64_t) llround(a.clip_seconds * fps);
            T_from = "--clip-seconds";
        }
    } else if (ds.clip_frames > 0 && ds.clip_frames != T) {
        fprintf(stderr,
                "[yue2-train] NOTE: --frames %lld overrides the manifest's clip_frames %lld; clips are read "
                "from their own offset and truncated to %lld frames\n",
                (long long) T, (long long) ds.clip_frames, (long long) T);
    }
    if (T <= 0) {
        fprintf(stderr, "[yue2-train] clip length resolved to %lld frames\n", (long long) T);
        return 1;
    }

    // Clips shorter than the training window are dropped, not padded: a padded
    // tail is a stretch of silence the model would be taught to render.
    std::vector<const Yue2TrainClip *> clips;
    int64_t                            n_short = 0, n_codec = 0;
    for (const Yue2TrainClip & cl : ds.clips) {
        if (cl.frames < T) {
            n_short++;
            continue;
        }
        if (!cl.codec_ids.empty()) {
            n_codec++;
        }
        clips.push_back(&cl);
    }
    if (clips.empty()) {
        fprintf(stderr, "[yue2-train] every clip in the manifest is shorter than %lld frames (%.1f s)\n",
                (long long) T, (double) T / fps);
        return 1;
    }
    fprintf(stderr, "[yue2-train] %zu clip(s) of %lld frames (%.1f s at %.3g fps), %lld dropped as short\n",
            clips.size(), (long long) T, (double) T / fps, fps, (long long) n_short);
    if (n_codec) {
        fprintf(stderr,
                "[yue2-train] %lld clip(s) carry codec_ids — conditioning is then PER CLIP (08 §5's seam), "
                "so the AR-prefix cache holds one canvas per clip and will evict at --kv-cache %lld unless "
                "that covers the set\n",
                (long long) n_codec, (long long) a.kv_cache);
    } else if (ds.codec_ids_present) {
        // The writer's own instruction for this case: treat the clip as
        // text-only rather than guess (yue2-preprocess-run.h:126-127).
        fprintf(stderr, "[yue2-train] NOTE: the manifest sets codec_ids_present but no clip carries ids — "
                        "training in the text-only regime\n");
    }

    // ── tokenizer ──
    BPETokenizer tok;
    if (!yue2_tokenizer_load_from_gguf(&tok, m.lm_file.path)) {
        fprintf(stderr, "[yue2-train] tokenizer: cannot read tokenizer.ggml.* from %s\n",
                m.lm_file.path.c_str());
        return 1;
    }

    // ── trainable state ──
    const int n_layers = (int) c.block_count;
    // b_sigma 0: B stays EXACTLY zero, so the adapter starts as an exact no-op
    // (08 §6's phase-2 second gate). The FD gate is the only caller that wants
    // otherwise, and it says why.
    const float    lossgrad = 1.0f / (float) std::max<int64_t>(1, a.grad_accum);
    Yue2NtTrainCtx C;
    if (!yue2_nt_train_ctx_init(m, n_layers, a.rank, a.alpha, target, a.seed, /*b_sigma=*/0.0f, lossgrad,
                                a.max_grad_norm, "yue2-train", &C, &err)) {
        fprintf(stderr, "[yue2-train] %s\n", err.c_str());
        return 1;
    }
    C.opt.weight_decay = a.weight_decay;
    // Neutralise LmOptim's own cosine — yue2_nt_lr_at carries the schedule.
    C.opt.lr_floor     = 1.0f;
    C.opt.total_steps  = 1;
    C.opt.warmup_steps = 0;

    // ── resume ──
    Yue2NtCkptState want;
    want.rank        = (int32_t) a.rank;
    want.n_params    = (int32_t) C.params.size();
    want.n_layers    = (int32_t) n_layers;
    want.target      = (int32_t) target;
    want.grad_accum  = (int32_t) a.grad_accum;
    want.frames      = (int32_t) T;
    want.total_steps = (int32_t) a.steps;
    want.alpha       = a.alpha;
    want.lr          = a.lr;
    want.seed        = a.seed;
    want.cond_hash   = yue2_nt_cond_hash(a);
    want.warmup      = (int32_t) a.warmup;
    want.lr_sched    = (a.lr_scheduler == "constant") ? 1 : 0;
    want.weight_decay  = a.weight_decay;
    want.max_grad_norm = a.max_grad_norm;

    int64_t step0     = 0;    // completed optimizer steps
    double  loss_sum  = 0.0;  // whole-run running mean
    int64_t n_micro   = 0;
    const std::string ckpt_path = yue2_nt_ckpt_path(a.out_dir);
    if (a.resume) {
        Yue2NtCkptState got;
        if (!yue2_nt_ckpt_load(ckpt_path, want, &got, &C, &err)) {
            fprintf(stderr, "[yue2-train] resume: %s\n", err.c_str());
            return 1;
        }
        step0    = got.steps_done;
        loss_sum = got.loss_sum;
        n_micro  = got.n_micro;
        fprintf(stderr, "[yue2-train] resumed from %s at step %lld/%lld (AdamW iter %d)\n", ckpt_path.c_str(),
                (long long) step0, (long long) a.steps, C.opt.opt_iter);
        if (step0 >= a.steps) {
            fprintf(stderr, "[yue2-train] that run is already finished; nothing to do\n");
            yue2_nt_train_ctx_free(&C);
            return 0;
        }
    } else if (pm_file_exists(ckpt_path)) {
        // Refuse rather than overwrite: someone who meant to resume and forgot
        // the flag would otherwise silently restart from zero and lose the run.
        fprintf(stderr,
                "[yue2-train] %s already exists. Pass --resume to continue that run, or point --out "
                "somewhere else to start a new one.\n",
                ckpt_path.c_str());
        return 1;
    }
    if (!pm_mkdir_p(a.out_dir)) {
        fprintf(stderr, "[yue2-train] cannot create %s\n", a.out_dir.c_str());
        return 1;
    }

    // ── conditioning cache ──
    //
    // Two caches: token ids per style string (cheap, unbounded — a few hundred
    // ints each) and the AR-prefix K/V canvases (expensive, bounded).
    std::map<std::string, std::vector<int32_t>> cond_ids_cache;
    Yue2NtKvCache                               kvc;
    kvc.cap = (size_t) std::max<int64_t>(1, a.kv_cache);

    // ONE key for both caches, and it is the WHOLE conditioning — style,
    // lyrics and the raw codec ids. Keying the canvas on anything smaller (the
    // clip id, say) would hand two clips that share an id but not their codec
    // ids each other's AR prefix, which renders and trains and is wrong.
    auto cond_key = [](const std::string & style, const std::string & lyrics,
                       const std::vector<int32_t> & codec) {
        std::string key = style;
        key += '\x01';
        key += lyrics;
        if (!codec.empty()) {
            key += '\x02';
            key.append((const char *) codec.data(), codec.size() * sizeof(int32_t));
        }
        return key;
    };

    auto cond_ids_for = [&](const std::string & key, const std::string & style, const std::string & lyrics,
                            const std::vector<int32_t> & codec) -> const std::vector<int32_t> * {
        auto it = cond_ids_cache.find(key);
        if (it != cond_ids_cache.end()) {
            return &it->second;
        }
        Yue2NarTrainCond cond;
        try {
            // The protocol lives in yue2-tokenizer.h and is NOT reimplemented
            // here; yue2_nar_train_cond_ids then appends the codec ids (with
            // the +YUE2_CODEC_OFFSET that happens there and only there) and
            // MUSIC_END, which is yue2-pipeline.h:447-452.
            const std::vector<int> pre = yue2_token_prefixes(&tok, style, lyrics, YUE2_COT_OFF, nullptr);
            cond.prefix_ids.assign(pre.begin(), pre.end());
        } catch (const std::exception & e) {
            fprintf(stderr, "[yue2-train] prefix assembly failed: %s\n", e.what());
            return nullptr;
        }
        cond.codec_ids = codec;
        return &(cond_ids_cache[key] = yue2_nar_train_cond_ids(cond));
    };

    // Caption dropout swaps in the EMPTY-STYLE prefix — style "", not "the
    // trigger without the caption" (train.py:199-201). That is what keeps the
    // base style reachable after training.
    fprintf(stderr,
            "[yue2-train] lr %.3g (%s, warmup %lld), %lld steps x %lld micro, clip-norm %.2f, wd %.3g, "
            "t %s, caption-dropout %.2f, trigger \"%s\"\n",
            (double) a.lr, a.lr_scheduler.c_str(), (long long) a.warmup, (long long) a.steps,
            (long long) a.grad_accum, (double) a.max_grad_norm, (double) a.weight_decay,
            a.t_sampling.c_str(), (double) a.caption_dropout, a.trigger.c_str());
    if (a.trigger.empty()) {
        fprintf(stderr, "[yue2-train] WARNING: no --trigger. The adapter will have no word to address it by "
                        "at generation time, which is upstream's whole mechanism for reaching the style.\n");
    }

    // ── the loop ──
    std::vector<float>     z, noise, x_t, vtarget;
    Yue2NarTrainHostInputs hin;
    double                 window_sum = 0.0;
    int64_t                window_n   = 0;
    const int64_t          t0_ms      = ggml_time_ms();
    bool                   said_ctx   = false;

    for (int64_t step = step0 + 1; step <= a.steps; step++) {
        const double lr_now = yue2_nt_lr_at(a, step - 1);
        C.opt.base_lr       = (float) lr_now;
        lm_optim_zero_grad(&C.opt);

        double      step_loss = 0.0;
        double      last_t    = 0.0;
        const char * last_id  = "";
        for (int64_t g = 0; g < a.grad_accum; g++) {
            const uint64_t k = (uint64_t) ((step - 1) * a.grad_accum + g);

            Yue2NtRng    r_clip(yue2_nt_seed_mix(a.seed, k, YUE2_NT_TAG_CLIP));
            const size_t pick = (size_t) (r_clip.u01() * (double) clips.size());
            const Yue2TrainClip & cl = *clips[std::min(pick, clips.size() - 1)];
            last_id                  = cl.id.c_str();

            bool dropped = false;
            if (a.caption_dropout > 0.0f) {
                Yue2NtRng r_drop(yue2_nt_seed_mix(a.seed, k, YUE2_NT_TAG_DROPOUT));
                dropped = r_drop.u01() < (double) a.caption_dropout;
            }
            const std::string style  = dropped ? std::string()
                                               : yue2_style_string(a.trigger, cl.caption, cl.genre, cl.bpm, cl.key,
                                                                   a.style_template != "bare");
            const std::string lyrics = dropped ? std::string() : (cl.lyrics.empty() ? a.lyrics : cl.lyrics);
            const std::string            key    = cond_key(style, lyrics, cl.codec_ids);
            const std::vector<int32_t> * ar_ids = cond_ids_for(key, style, lyrics, cl.codec_ids);
            if (!ar_ids) {
                return 1;  // cond_ids_for already said why
            }
            Yue2NarTrainKv * kv = yue2_nt_kvcache_get(&kvc, m, key, *ar_ids, T, n_layers, &err);
            if (!kv) {
                fprintf(stderr, "[yue2-train] AR prefill: %s\n", err.c_str());
                return 1;
            }
            if (!said_ctx) {
                said_ctx = true;
                fprintf(stderr,
                        "[yue2-train] conditioning: %zu ids -> ar_len %lld; N_nar %lld, S_kv %lld, "
                        "%.1f MB per cached canvas, cap %zu\n",
                        ar_ids->size(), (long long) kv->ar_len, (long long) kv->n_nar,
                        (long long) kv->s_kv, kvc.bytes / 1048576.0, kvc.cap);
            }

            if (!yue2_nt_read_clip(cl, LD, T, &z, &err)) {
                fprintf(stderr, "[yue2-train] %s\n", err.c_str());
                return 1;
            }

            // t, exactly upstream's sample_t (train.py:131-139). The LOGIT
            // value is what the network is told (contract §7: yue2_nar_velocity
            // applies the shift and the sinusoid itself); the PROBABILITY is
            // what noises the latents.
            double raw_t = 0.0, t_val = 0.0;
            {
                Yue2NtRng r_t(yue2_nt_seed_mix(a.seed, k, YUE2_NT_TAG_T));
                if (a.t_sampling == "uniform") {
                    t_val = 0.001 + 0.998 * r_t.u01();
                    raw_t = yue2_nar_logit_clamped(t_val);
                } else {
                    raw_t = r_t.normal();
                    t_val = 1.0 / (1.0 + std::exp(-raw_t));
                }
            }
            last_t = t_val;

            noise.assign(z.size(), 0.0f);
            yue2_nt_fill_normal(&noise, yue2_nt_seed_mix(a.seed, k, YUE2_NT_TAG_NOISE));
            yue2_nt_make_xt_target(z, noise, (float) t_val, &x_t, &vtarget);
            if (!yue2_nt_host_inputs(c, kv->ar_len, kv->n_nar, x_t, raw_t, &hin, &err)) {
                fprintf(stderr, "[yue2-train] %s\n", err.c_str());
                return 1;
            }

            float lv = 0.0f;
            if (!yue2_nt_micro(m, C.sched, C.ad, *kv, &C.opt, n_layers, hin, vtarget, /*backward=*/true, &lv,
                               nullptr, &err)) {
                fprintf(stderr, "[yue2-train] step %lld: %s\n", (long long) step, err.c_str());
                return 1;
            }
            step_loss += (double) lv;
            loss_sum += (double) lv;
            n_micro++;
        }

        LmStepStats st{};
        if (!lm_optim_step(&C.opt, C.osched, &st)) {
            fprintf(stderr, "[yue2-train] optimizer step failed at step %lld\n", (long long) step);
            return 1;
        }

        const double mean = step_loss / (double) a.grad_accum;
        window_sum += mean;
        window_n++;
        if (a.log_every > 0 && (step % a.log_every == 0 || step == step0 + 1 || step == a.steps)) {
            const double  elapsed = (double) (ggml_time_ms() - t0_ms) / 1000.0;
            const DitGpuMem gm    = dit_gpu_mem_query(m.backend);
            fprintf(stderr,
                    "[yue2-train] step %5lld/%lld  loss %.5f (win %.5f, run %.5f)  |g| %.4f  lr %.2e  "
                    "t %.3f  %.2fs/it  vram %zu/%zu MB (%s)  [%s]\n",
                    (long long) step, (long long) a.steps, mean, window_sum / (double) window_n,
                    n_micro ? loss_sum / (double) n_micro : 0.0, (double) st.grad_norm, (double) st.lr,
                    last_t, elapsed / (double) std::max<int64_t>(1, step - step0), gm.used_mb(),
                    gm.total_mb(), gm.source(), last_id);
            window_sum = 0.0;
            window_n   = 0;
        }

        if (a.save_every > 0 && step % a.save_every == 0 && step < a.steps) {
            Yue2NtCkptState st_save = want;
            st_save.steps_done      = (int32_t) step;
            st_save.loss_sum        = loss_sum;
            st_save.n_micro         = n_micro;
            st_save.opt_step        = C.opt.opt_step;
            st_save.opt_iter        = C.opt.opt_iter;
            if (!yue2_nt_ckpt_save(ckpt_path, st_save, C, &err)) {
                fprintf(stderr, "[yue2-train] checkpoint: %s\n", err.c_str());
                return 1;
            }
            // std::string, not a fixed buffer: a long --name would silently
            // truncate into a path that collides with the previous snapshot.
            const std::string snap =
                a.out_dir + "/" + a.name + "_step" + std::to_string((long long) step) + ".safetensors";
            if (!yue2_nt_export(C.ad, a, target, step, T, m.lm_file.path, snap, &err)) {
                fprintf(stderr, "[yue2-train] snapshot export: %s\n", err.c_str());
                return 1;
            }
            fprintf(stderr, "[yue2-train] checkpoint at step %lld -> %s (+ snapshot)\n", (long long) step,
                    ckpt_path.c_str());
        }
    }

    // ── export ──
    const std::string out = a.out_dir + "/" + a.name + ".safetensors";
    if (!yue2_nt_export(C.ad, a, target, a.steps, T, m.lm_file.path, out, &err)) {
        fprintf(stderr, "[yue2-train] export: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr,
            "[yue2-train] done: %lld steps, mean loss %.5f, AR-prefix cache %lld build(s) / %lld hit(s) / "
            "%lld eviction(s)\n",
            (long long) a.steps, n_micro ? loss_sum / (double) n_micro : 0.0, (long long) kvc.builds,
            (long long) kvc.hits, (long long) kvc.evictions);
    fprintf(stderr, "[yue2-train] load it with /yue2/select-model's adapter field (yue2-adapter.h)%s%s\n",
            a.trigger.empty() ? "" : "; put the trigger word at the start of the style prompt: ",
            a.trigger.empty() ? "" : a.trigger.c_str());

    // The resume state is the run's MIDDLE, not its result. Leaving it behind
    // would let a later `--resume` in the same directory come back from a run
    // that has already finished and exported.
    hs_remove(ckpt_path);
    yue2_nt_kvcache_free(&kvc);
    yue2_nt_train_ctx_free(&C);
    return 0;
}

// ── Entry point ────────────────────────────────────────────────────────────

static int yue2_nar_train_run(const Yue2NarTrainArgs & a) {
    if (a.fd_check > 0) {
        return yue2_nar_fdcheck_main(a);
    }
    return yue2_nar_train_loop(a);
}
