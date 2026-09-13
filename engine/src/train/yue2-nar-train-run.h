#pragma once
// train/yue2-nar-train-run.h — the runner around yue2-nar-train-graph.h.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Phase 2 of
// docs/plans/yue2/08-nar-lora-trainer.md, implementing the gate half of
// docs/plans/yue2/09-nar-train-graph-contract.md §10.
//
// ── Scope of THIS increment: the gate, and nothing else ────────────────────
//
// `ace-train yue2-nar-train --fd-check N` is the ONLY thing that works here.
// It loads the LM, builds one conditioning prefix, prefills the AR K/V once,
// draws a stand-in latent, runs one forward + backward, and finite-differences
// the analytic gradient against the measured loss change. Everything else —
// the dataset manifest, the training loop, checkpoint/resume, safetensors
// export — is a TODO that returns an error, deliberately: a silent no-op that
// prints "done" is how a trainer ships that never trained anything.
//
// The stand-in latent is `z ~ N(0,1)`, not a real VAE-encoded clip. That is
// correct for a gradient check and wrong for anything else: FD tests that the
// backward agrees with the forward, which is a property of the GRAPH and not
// of the data. Phase 1 (`yue2_vae_encode`) supplies real latents; until it
// lands, `--vae-dir` is accepted and ignored so the eventual CLI shape does
// not change under anyone.
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

#include "train/lm-optim.h"
#include "train/yue2-nar-train-graph.h"

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
#include <stdexcept>
#include <string>
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

    std::string manifest;  // TODO: preprocess manifest (clip latents + captions)
    std::string out_dir;   // TODO: checkpoints + exported adapter

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
    uint64_t    seed   = 42;

    // Real-loop knobs, parsed and stored so the CLI is stable; the loop that
    // reads them is a TODO below.
    float       lr         = 1e-4f;
    int64_t     steps      = 800;
    int64_t     grad_accum = 1;
    std::string t_sampling = "logit-normal";  // logit-normal | uniform

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
    if (!yue2_nt_discover(&m, a, &err)) {
        fprintf(stderr, "[yue2-fd] %s\n", err.c_str());
        return 1;
    }
    if (!a.vae_dir.empty()) {
        fprintf(stderr, "[yue2-fd] --vae-dir is accepted and IGNORED: the gate uses z ~ N(0,1) as a "
                        "stand-in latent (real clips arrive with phase 1, yue2_vae_encode)\n");
    }
    if (!yue2_load_parts(&m, /*want_lm=*/true, /*want_vae=*/false, YUE2_VAE_STANDARD, /*want_encoder=*/false,
                         &err)) {
        fprintf(stderr, "[yue2-fd] load: %s\n", err.c_str());
        return 1;
    }
    const Yue2LmConfig & c  = m.lm_cfg;
    const int64_t        LD = (int64_t) c.latent_dim;
    const int64_t        T  = a.frames;
    if (T <= 0) {
        fprintf(stderr, "[yue2-fd] --frames must be > 0\n");
        return 1;
    }

    // Contract §7: upstream noises with the UNSHIFTED t while telling the
    // network shift(raw). At this checkpoint's shift = 1.0 the two coincide
    // exactly (yue2-nar-graph.h:188-195); at any other shift the upstream
    // recipe is internally inconsistent and porting it silently would port the
    // bug. Warn loudly rather than refuse — the gate itself is still valid,
    // it just gates a recipe nobody has reconciled.
    if (std::fabs((double) c.timestep_shift - 1.0) > 1e-6) {
        fprintf(stderr,
                "[yue2-fd] WARNING: yue2.timestep_shift = %.6f, not 1.0. The upstream recipe noises with "
                "the unshifted t while feeding the network shift(raw); those only coincide at shift 1.0. "
                "Reconcile before training anything on this checkpoint.\n",
                (double) c.timestep_shift);
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

    // ── adapters ──
    const size_t     n_tensors = yue2_nt_adapter_tensor_count(n_layers, target);
    ggml_init_params aip       = { (n_tensors + 16) * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    ggml_context *   actx      = ggml_init(aip);
    Yue2TrainAdapters          ad;
    std::vector<ggml_tensor *> params;
    if (!actx || !yue2_nt_make_adapters(actx, m, n_layers, a.rank, a.alpha, target, &ad, &params) ||
        params.size() != n_tensors) {
        fprintf(stderr, "[yue2-fd] adapter allocation failed (%zu of %zu tensors)\n", params.size(),
                n_tensors);
        return 1;
    }
    // The five host-owned optimizer scalars. LmOptim declares every one of
    // them nullptr and lm_optim_init creates NONE — each is written or read
    // unconditionally inside lm_optim_step, so a missing one is a null
    // dereference AFTER the graph has computed, which reads as a fault in the
    // optimizer maths. mm3-dit-train-run.h:553-583 is the full post-mortem.
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 7);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(actx, GGML_TYPE_F32, 1);
    ggml_set_name(t_lossgrad, "lossgrad");
    ggml_set_name(t_adamw, "adamw_params");
    ggml_set_name(t_clip, "grad_clip");
    ggml_set_name(t_eps, "eps");
    ggml_set_name(t_gnorm2, "gnorm2");

    ggml_backend_buffer_t abuf = ggml_backend_alloc_ctx_tensors(actx, m.backend);
    if (!abuf) {
        fprintf(stderr, "[yue2-fd] adapter buffer allocation failed\n");
        return 1;
    }
    // Without the WEIGHTS usage hint the scheduler cuts the graph once per
    // LoRA parameter — 293 splits and 47.6 s/step on MM3, GPU at 9 %.
    ggml_backend_buffer_set_usage(abuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // B NON-ZERO for the gate. With B == 0 (the production init) dL/dA is
    // identically zero and every .A probe passes while measuring nothing.
    yue2_nt_init_adapters(params, a.seed, /*b_sigma=*/1e-2f);

    {
        // YUE2_FD_LOSSGRAD is the negative control (file header): seed
        // dL/dloss at 2.0 and every probe must land at rel ~= 0.5.
        float       lg  = 1.0f;
        const char * ev = std::getenv("YUE2_FD_LOSSGRAD");
        if (ev && ev[0]) {
            lg = (float) atof(ev);
            fprintf(stderr, "[yue2-fd] NEGATIVE CONTROL: dL/dloss seeded at %.3f — expect rel ~= %.3f on "
                            "every probe, and a FAIL\n",
                    (double) lg, (double) std::fabs(1.0f - lg) / (double) std::max(1e-6f, std::fabs(lg)));
        }
        const float clip = 1.0f, epsv = 1e-6f;
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, sizeof(float));
        ggml_backend_tensor_set(t_clip, &clip, 0, sizeof(float));
        ggml_backend_tensor_set(t_eps, &epsv, 0, sizeof(float));
    }

    LmOptim opt;
    opt.optimizer  = "adamw";
    opt.t_lossgrad = t_lossgrad;
    opt.t_adamw    = t_adamw;
    opt.t_clip     = t_clip;
    opt.t_eps      = t_eps;
    opt.t_gnorm2   = t_gnorm2;
    if (!lm_optim_init(&opt, params, m.backend, &err)) {
        fprintf(stderr, "[yue2-fd] optimizer init: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "[yue2-fd] %zu LoRA tensors over %d NAR layers, rank %lld, alpha %.1f, target %s\n",
            params.size(), n_layers, (long long) a.rank, (double) a.alpha, yue2_nt_target_name(target));

    BackendPair bp{};
    bp.backend     = m.backend;
    bp.cpu_backend = m.cpu_backend;
    bp.has_gpu     = m.backend != m.cpu_backend;
    ggml_backend_sched_t sched = backend_sched_new(bp, 65536);
    if (!sched) {
        fprintf(stderr, "[yue2-fd] scheduler alloc failed\n");
        return 1;
    }

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
    yue2_nt_host_inputs(c, kv.ar_len, kv.n_nar, x_t, raw_t, &hin);
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
    fprintf(stderr, "[yue2-fd] step floor: dL >= %.2e; a probe whose 2*eps*||g|| clears it keeps eps %.3g\n",
            dl_min, a.fd_eps);

    fprintf(stderr, "\n[yue2-fd] %-24s %10s %13s %9s %13s %8s\n", "probe (whole tensor)", "n", "||g||",
            "step", "numeric", "rel");
    double              worst    = 0.0;
    int                 n_raised = 0;
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
        double h = a.fd_eps;
        if (gnorm > 0.0 && 2.0 * a.fd_eps * gnorm < dl_min) {
            h = dl_min / (2.0 * gnorm);
            n_raised++;
        }

        std::vector<float> w0((size_t) ggml_nelements(pr.par)), wtmp(w0.size());
        ggml_backend_tensor_get(pr.par, w0.data(), 0, w0.size() * sizeof(float));

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
        fprintf(stderr, "[yue2-fd] %-24s %10zu %13.6e %9.3g %13.6e %8.3f\n", pr.name, g.size(), gnorm, h, num,
                rel);
        fd_rel.push_back(rel);
        worst = std::max(worst, rel);
    }

    // The bar is MM3's, unchanged, and it is a VERDICT only under isolation.
    // Central differencing carries a genuine O(h^2 * f''') truncation error
    // that no amount of precision removes; 2e-2 leaves room for probe-to-probe
    // variation without admitting a real scale error — a wrong gradient scale
    // misses by a FACTOR, not by a percent.
    const double bar = isolated ? 2e-2 : 0.15;
    int          n_bad = 0;
    for (double r : fd_rel) {
        // NaN fails: `!(r < bar)` rather than `r >= bar`, so a probe whose
        // numeric side came back NaN (a failed forward, or ||g|| == 0 with a
        // measurable loss change) counts against the gate instead of sliding
        // through a comparison that is false for NaN either way.
        if (!(r < bar)) {
            n_bad++;
        }
    }

    if (isolated) {
        fprintf(stderr,
                "\n[yue2-fd] GATE %s: finite differences, F32-isolated to %d NAR layers, bar %.0e, "
                "%d/%zu probes, worst %.4f (step raised on %d)\n",
                n_bad == 0 ? "PASS" : "FAIL", n_layers, bar, (int) probes.size() - n_bad, probes.size(),
                worst, n_raised);
        if (n_bad) {
            fprintf(stderr,
                    "[yue2-fd]   The analytic gradient disagrees with the measured loss change. With\n"
                    "[yue2-fd]   segments = 1 there is no second backward route to cross-check against,\n"
                    "[yue2-fd]   so this is the only gradient gate — do NOT train past it.\n");
        }
    } else {
        fprintf(stderr,
                "\n[yue2-fd] %d/%zu probes within %.0f%% (worst %.4f) — INDICATIVE ONLY (no F32 "
                "isolation).\n"
                "[yue2-fd]   Add --nar-layers 2 to turn this into a verdict.\n",
                (int) probes.size() - n_bad, probes.size(), bar * 100.0, worst);
    }

    ggml_backend_sched_free(sched);
    yue2_nt_kv_free(&kv);
    yue2_nt_f32_free(&iso);
    return (isolated && n_bad) ? 1 : 0;
}

// ── TODO stubs — errors, never silent no-ops ───────────────────────────────
//
// Each of these is a real phase-4 deliverable (08-nar-lora-trainer.md §6).
// They return 1 and say what is missing rather than doing nothing quietly,
// because a trainer that prints "done" without training is the exact failure
// this codebase has shipped before.

static int yue2_nar_train_loop(const Yue2NarTrainArgs & a) {
    (void) a;
    fprintf(stderr,
            "ace-train yue2-nar-train: the training loop is NOT IMPLEMENTED yet.\n"
            "  Phase 2 ships the graph and the gradient gate only. Run:\n"
            "      ace-train yue2-nar-train --lm <yue2-lm-*.gguf> --fd-check 6 --nar-layers 2\n"
            "  Still missing (08-nar-lora-trainer.md §6): phase 1's yue2_vae_encode (real clip\n"
            "  latents), ace-train yue2-preprocess (the manifest this loop would read), the loop\n"
            "  itself with AdamW + cosine + clip-norm, checkpoint/resume, and safetensors export\n"
            "  through train/st-write.h in plan §4's key scheme.\n");
    return 1;
}

// ── Entry point ────────────────────────────────────────────────────────────

static int yue2_nar_train_run(const Yue2NarTrainArgs & a) {
    if (a.fd_check > 0) {
        return yue2_nar_fdcheck_main(a);
    }
    return yue2_nar_train_loop(a);
}
