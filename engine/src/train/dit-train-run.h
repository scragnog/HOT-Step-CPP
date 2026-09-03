#pragma once
// dit-train-run.h — `ace-train train-dit` stage driver (plan §3.5.1-§3.5.3).
//
// train -> export. The loop is lm-optim.h, unchanged and unmodified (D1): our own
// persistent grad accumulators driven by ggml_build_backward_expand, and our own
// merged norm+clip+AdamW graph. The DiT trainer is the LM trainer with a
// different graph and a different loss.
//
// Emission contract (§2.2): `start` is the first JSONL line, `done` is the last;
// a `fatal` line replaces `done` and precedes a non-zero exit.
//
// JSONL FIELDS WHOSE MEANING CHANGED WITH MICRO-BATCHING (2026-07-29):
//   step.micro   was "samples in this optimizer step". It is now MICRO-BATCHES in
//                this optimizer step; the sample count is micro x batch, except in
//                a short final window where the last micro-batch is narrower.
//                (`vram.batch` carries B, so a consumer can still compute it.)
//   start.batch  gone: `start` fires before the model is loaded and cannot know
//   start.ckpt   what the VRAM fit / dataset size / A1 cap leave of the request.
//                They are now `start.batchRequested` / `start.ckptRequested`, and
//                the RESOLVED pair is `vram.batch` / `vram.ckptSegments`.
//   data.batch   the resolved B (unchanged in name; it never carried the request).

#include "train/artist-token-io.h"
#include "train/dit-data.h"
#include "train/dit-export.h"
#include "train/dit-resume.h"
#include "train/dit-selftest.h"
#include "train/dit-train-ckpt.h"

#include <algorithm>
#include <map>
#include "train/dit-node-profile.h"
#include "train/dit-train-graph.h"
#include "train/dit-vram.h"
#include "train/lm-optim.h"
#include "version.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// ─── args (§2.1) ────────────────────────────────────────────────────────────

struct DitTrainArgs {
    std::vector<std::string> stages{ "train", "export" };

    std::string tensors_dir, out_dir, models_dir, dit_path, dit_name;

    std::string adapter_type = "lora";
    int         rank = 128, alpha = 256;
    // LoKR (--adapter-type lokr); Uber-LoKR-4 defaults, K2.
    int   lokr_dim            = 512;
    float lokr_alpha          = 512.0f;
    int   lokr_factor         = 6;
    bool  lokr_decompose_both = true;
    // ── Artist token (textual inversion), V1: conditioning soft prompt ──────
    //
    // k learned rows appended to the padded encoder sequence, so they reach the
    // DiT through cross-attention having passed the same frozen cond_emb a real
    // caption row does. Touches no weight. Off unless --artist-token names one.
    std::string artist_token = "";
    int         artist_k     = 8;
    bool        artist_only  = false;  // freeze the adapter, train only the rows
    // DoRA (weight-decomposed LoRA): a learned per-output-row magnitude on top
    // of the LoRA direction. The merge path already reads it
    // (adapter-merge.h lora_magnitude_vector); this is the trainer half.
    bool        dora         = false;
    // ON by default: an attention-only LoRA leaves the MLP projections — where
    // most of the timbre lives — frozen. --no-target-mlp turns it back off.
    bool        target_mlp = true;
    int         layers     = 0;  // 0 = auto (top-K depth)

    std::string loss_weighting = "flow_snr";
    float       snr_gamma = 5.0f, t_bias = 0.5f;
    bool        channel_balance = true;
    float       timestep_mu = -0.4f, timestep_sigma = 1.0f, t_min = 0.0f, t_max = 1.0f, cfg_ratio = 0.15f;
    int         genre_ratio = 30;

    float       lr = 5e-4f;
    int         epochs = 400, grad_accum = 4;
    // Micro-batching (design C5): crops per micro-batch, from B DIFFERENT songs.
    // --grad-accum counts MICRO-BATCHES, so effective samples per optimizer step
    // is batch x grad_accum.
    //
    // DEFAULT 1, i.e. OFF (2026-07-29, measured — was 5). The full-depth bench on
    // a 32 GB RTX 5090 (LoKR dim512+MLP, 32 layers, bf16 mirror) came out ~2.5x
    // SLOWER at batch 5 + auto-checkpointing than at batch 1: a full-depth graph
    // is already compute-bound, so a wider batch buys nothing while paying
    // checkpointing's extra no-grad forward AND a smaller auto-fit crop. Shallow /
    // partial-depth runs are the opposite (~2.4x faster per crop at --layers 8),
    // which is exactly who should raise this. See docs/TRAINING.md.
    int         batch = 1;
    // Gradient checkpointing (design C3): 0 = off (the monolithic graph, which
    // stays the code path taken whenever this is 0), 1 = auto (the VRAM fit
    // picks), N >= 2 = exactly N segments.
    int         ckpt = 1;
    float       warmup_ratio = 0.05f, grad_clip = 1.0f, weight_decay = 0.01f;
    int         seed        = 42;
    float       target_loss = 0.4f;
    std::string order       = "shuffle";

    // crop_max 800 (2026-09-03): three clean ear A/Bs (nirvana 438 vs 1544,
    // mika 552 vs 1516 on its own track and on a foreign track) all preferred
    // the short-crop adapter. Long crops reach low loss 2-3x faster and export
    // deep memorisers of 60 s windows (coarse/bitty); short crops memorise
    // quiet windows too faithfully (fuzz in quiet passages). 800 is the middle
    // Rob's best-liked adapter came from (fightstar, 824). Explicit --crop-max
    // still wins in both directions.
    int   crop = 0, crop_min = 375, crop_max = 800;
    // Set by the CLI parser when --crop-max was given explicitly. Only the
    // DEFAULT cap is lifted in flash mode; a user's number is never moved.
    bool  crop_max_user = false;
    // Crop regime (2026-08-29): `song`/`structured` are the fixed defaults;
    // `zero`/`random` reproduce the legacy run (positions lied, endpoints
    // starved) for A/B archaeology. Runs across the divide are NOT comparable.
    std::string crop_anchor = "song";        // song|zero
    std::string crop_mode   = "structured";  // structured|random
    // Endpoint shares are a CEILING, not a quota: dit_crop_endpoint_frac()
    // scales them down by how much of a track one crop actually covers, so a
    // short crop relaxes toward the uniform sampler instead of spiking two
    // positions. See the 2026-08-30 note on dit_sample_crop_structured.
    float crop_start_frac = 0.2f;
    // Restored to 0.2 (2026-09-01) after a day at 0. The 2026-08-30 zeroing
    // rested on a positional theory that pure-RoPE attention makes impossible
    // (see the correction note on dit_sample_crop_structured), and on "the
    // base DiT already ends well, nothing to rescue" — refuted by the
    // 2026-08-31 overnight batch: every adapter trained with the end share at
    // 0 still cut off mid-stream, because endings then get ~1.4 supervised
    // draws per 500-epoch run and the LoKr freely damages the base's
    // ending behaviour. The share now splits flush-jitter / closing-region
    // draws, which keeps the real decay-to-silence in every flush draw while
    // varying the window enough to avoid the 08-29 memorisation.
    float crop_end_frac   = 0.2f;
    // Opening window the start share spreads over, in crop lengths.
    int   crop_start_window = 1;
    // Endpoint share ceiling as a multiple of the crop's natural coverage.
    float crop_endpoint_k  = 2.0f;
    // --crop-jitter (2026-09-03): each draw's length is uniform over the
    // patch-aligned lengths in [crop_min, crop], so a long-cap run still gets
    // whole optimizer steps that are ONLY a quiet intro or breakdown (the
    // thing short crops gave for free) while keeping long-context draws.
    // Off = byte-identical to the two-draw sampler.
    bool  crop_jitter = false;
    int   vram_reserve_mb = 2048;
    float vram_safety     = 0.05f;
    // Frozen-weight mirror precision:
    //   "f32"      shipped — every trainable-layer weight promoted to F32.
    //   "bf16"     halves the mirror; needs the patched CUDA out_prod
    //              (engine/patches/bf16-out-prod.patch). Stores AND computes bf16.
    //   "bf16-f32" bf16 storage, f32 compute (2026-09-02): same mirror bytes as
    //              "bf16", but each trainable-layer weight is cast to F32 in the
    //              graph at its mul_mat site, so activations and gradients are
    //              never rounded to bf16. Rob: bf16 compute audibly coarsened
    //              renders; this is the mode that fixes it without paying the
    //              f32 mirror's ~8 GB (and the crop it costs).
    std::string mirror = "f32";

    // MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch):
    //   "outprod" = ggml upstream, out_prod(src0, transpose(grad)) — F32-only on CUDA
    //   "mm"      = mul_mat(cont(transpose(src0)), grad) — dtype-agnostic, BF16
    //               tensor cores. ~1.7-1.8x per layer per step on an RTX 5090.
    // ENGINE DEFAULT IS "outprod" so a bare ace-train invocation is unchanged; the
    // server passes --bwd mm. Selected by setting GGML_BACKWARD_MM before any
    // backward graph is built (ace-train.cpp).
    std::string bwd = "outprod";

    // Attention formulation (--attn, docs/plans/2026-09-01-flash-attn-backward.md):
    //   "exact" = dit_attn_f32, the shipped graph, byte-identical to pre-flag runs
    //   "flash" = the fused GGML_OP_FLASH_ATTN_TRAIN{,_BACK} pair on BOTH the
    //             self- and cross-attention sites. EXPERIMENTAL. Lower VRAM per
    //             token (nothing S_kv x S is ever materialised), currently SLOWER
    //             per step, and its gradients differ from exact within a measured
    //             drift band rather than matching bit for bit.
    //
    // HOT-Step patch: flash-attn-train — a third value, "flash-f32". Same fused
    // ops, but GGML_PREC_F32 instead of the default, which pins the CUDA dispatch
    // to v1's strict scalar kernels instead of the TF32 tensor-core ones. It is
    // the A/B partner that separates "did fusion change the training" from "did
    // TF32 change it", and the escape hatch if TF32 ever turns out to matter. A
    // CLI/API surface: the Training Studio checkbox emits "flash" or "exact"
    // (tf32 design §3.5).
    std::string attn = "exact";

    // Optimizer (2026-07-30). "adamw" is the shipped path and the default;
    // "muon" puts every 2-D parameter whose short side is >= muon_min_dim on
    // orthogonalized-momentum updates and leaves the rest (LoKR w1 at [4,5], and
    // anything 1-D) on AdamW — the hybrid every published Muon uses. Muon's LR
    // does NOT mean AdamW's, hence a multiplier on the shared schedule rather
    // than a second absolute rate.
    std::string optimizer       = "adamw";
    // Prodigy (2026-09-03, same contract as train-lm / mm3-lm-train): lr is a
    // schedule multiplier forced to 1.0; the step size is estimated from d0.
    double      prodigy_d0      = 1e-6;
    float       muon_lr_scale   = 1.0f;
    float       muon_momentum   = 0.95f;
    int         muon_ns_steps   = 5;
    bool        muon_nesterov   = true;
    int         muon_min_dim    = 16;
    int         muon_bucket     = 16;

    // Step-time profiling (docs/plans/2026-07-30-dit-trainer-step-profile.md §2).
    // 0 = OFF and nothing about the run changes. N > 0 = time every micro-step
    // into the DitStepProf buckets and print a breakdown every N micro-steps.
    // Exists because the trainer's own fitted model predicts ~144 ms/step at
    // S=342 and the measured runs sit at ~1753 ms, and nobody could say of what.
    int profile_step = 0;
    // Op-level histogram for ONE warmed-up micro-step (plan §2.3). Implies the
    // bucket profiler, since it hangs off the same step counter.
    bool profile_ops = false;

    // Trigger word embedded in the exported adapter. Empty = fall back to the
    // variant's preprocess_meta.json custom_tag/tag_position; still empty after
    // that = no trigger keys written and the adapter is byte-identical to a
    // pre-trigger build. docs/plans/2026-07-28-adapter-trigger-embedding.md T5
    std::string trigger, trigger_position;

    float milestone_step = 0.1f;
    int   milestone_keep = 6;

    // Resume (--init-adapter, 2026-08-10): dir of an exported run whose factors
    // seed this one. Identity hyperparams adopted from its dit_train_log.json
    // by dit_resume_prepare (cmd_train_dit); explicit contradictions exit 2.
    // init_from_ma5 carries the source's saved_ma5 for provenance/logging.
    std::string init_adapter;
    double      init_from_ma5 = -1.0;

    bool overwrite = false;
    int  limit     = 0;
    bool self_test = false;
};

// Resolve the trigger to embed: explicit CLI flags win, else the variant's
// preprocess_meta.json, else nothing. `replace` positions are dropped with a
// warn (the tag was never applied to those captions — T4).
static void dit_resolve_trigger(const std::string & tensors_dir, std::string * trigger, std::string * position) {
    if (trigger->empty() && !tensors_dir.empty()) {
        lm_read_variant_tag(tensors_dir, trigger, position);
    }
    std::string why;
    if (!lm_trigger_normalize(trigger, position, &why) && !why.empty()) {
        jl("{\"type\":\"log\",\"level\":\"warn\",\"message\":\"%s\"}", lm_json_escape(why).c_str());
        fprintf(stderr, "[train-dit] %s\n", why.c_str());
    }
}

static inline bool dit_has_stage(const DitTrainArgs & a, const char * s) {
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (a.stages[i] == s) {
            return true;
        }
    }
    return false;
}

// ─── step-time profile (plan §2) ─────────────────────────────────────────────
//
// Wall-clock buckets around ONE micro-step, so "1753 ms per crop" becomes a
// breakdown instead of a mystery. Costs nothing when off: every timer sits
// behind `on`, and the un-profiled path keeps the original single
// `ggml_backend_sched_graph_compute` call rather than the
// alloc / compute_async / synchronize decomposition it is literally defined as
// (ggml-backend.cpp:1883-1887). Same three calls either way — the split only
// exists so the allocator/split pass can be told apart from the GPU work — but
// keeping the default path on the original call means a normal run cannot have
// been perturbed by this at all.
enum DitProfBucket {
    DPB_ASSEMBLE = 0,  // crop draw, flow target, masks, host buffers
    DPB_UPLOAD,        // the 11 unconditional tensor_set uploads + t_lossgrad
    DPB_BUILD,         // ggml_init + new_graph + forward + loss + forward_expand
    DPB_BACKWARD,      // fill_gacc + build_backward_expand
    DPB_ALLOC,         // sched_reset + sched_alloc_graph (split + gallocr)
    DPB_COMPUTE,       // compute_async + synchronize (the only GPU bucket)
    DPB_READBACK,      // loss / report tensor_get
    DPB_FREE,          // ggml_free(ctx)
    DPB_N
};

static const char * const DIT_PROF_NAMES[DPB_N] = { "assemble", "upload",   "build",    "backward",
                                                    "alloc",    "compute",  "readback", "free" };

struct DitStepProf {
    bool      on    = false;
    int       every = 0;  // report cadence, micro-steps
    long long win[DPB_N] = {};  // us since the last report
    long long tot[DPB_N] = {};  // us over the whole run
    long long win_steps  = 0;
    long long tot_steps  = 0;
    // The optimizer step, timed separately — it is per OPTIMIZER step, not per
    // micro-step, so it is not one of the buckets above.
    long long optim_us    = 0;
    long long optim_steps = 0;
    // Graph shape, captured every timed micro-step (constant in practice — if it
    // is not, that is itself the finding).
    // No leaf count: this ggml exposes only ggml_graph_n_nodes publicly, and
    // n_leafs is not worth reaching into ggml-impl.h for.
    int nodes = 0, splits = 0, copies = 0;

    inline long long t() const { return on ? ggml_time_us() : 0; }
    inline void add(DitProfBucket b, long long t0, long long t1) {
        if (!on) {
            return;
        }
        win[b] += t1 - t0;
        tot[b] += t1 - t0;
    }
    inline long long win_total() const {
        long long s = 0;
        for (int i = 0; i < DPB_N; i++) {
            s += win[i];
        }
        return s;
    }
    inline long long tot_total() const {
        long long s = 0;
        for (int i = 0; i < DPB_N; i++) {
            s += tot[i];
        }
        return s;
    }
};

// One breakdown per `--profile-step N` micro-steps: window mean per bucket in ms
// and its share, then the window resets. `splits` is the one to read first — any
// value above 1 means the scheduler put part of the graph on another backend and
// is copying activations across the bus every step.
static void dit_prof_report(DitStepProf & p) {
    if (p.win_steps <= 0) {
        return;
    }
    const double    n   = (double) p.win_steps;
    const long long tot = p.win_total();

    std::string line = "[train-dit] profile ";
    char        b[160];
    snprintf(b, sizeof(b), "(%lld steps, %.1f ms/step):", p.win_steps, (double) tot / n / 1000.0);
    line += b;

    std::string js = "{\"type\":\"profile\",\"buckets\":{";
    for (int i = 0; i < DPB_N; i++) {
        const double ms  = (double) p.win[i] / n / 1000.0;
        const double pct = tot > 0 ? 100.0 * (double) p.win[i] / (double) tot : 0.0;
        snprintf(b, sizeof(b), " %s %.1f (%.0f%%)", DIT_PROF_NAMES[i], ms, pct);
        line += b;
        snprintf(b, sizeof(b), "%s\"%s\":%.3f", i ? "," : "", DIT_PROF_NAMES[i], ms);
        js += b;
    }
    fprintf(stderr, "%s; nodes %d splits %d copies %d; optimizer %.1f ms x%lld\n", line.c_str(), p.nodes, p.splits,
            p.copies, p.optim_steps ? (double) p.optim_us / (double) p.optim_steps / 1000.0 : 0.0,
            (long long) p.optim_steps);

    snprintf(b, sizeof(b), "},\"steps\":%lld,\"msPerStep\":%.3f,", p.win_steps, (double) tot / n / 1000.0);
    js += b;
    snprintf(b, sizeof(b), "\"nodes\":%d,\"splits\":%d,\"copies\":%d}", p.nodes, p.splits, p.copies);
    js += b;
    jl("%s", js.c_str());

    for (int i = 0; i < DPB_N; i++) {
        p.win[i] = 0;
    }
    p.win_steps = 0;
}

// ─── op-level profile (plan §2.3) ────────────────────────────────────────────
//
// Times ONE micro-step node by node through the scheduler's eval callback and
// aggregates by (op, output shape) — the shape is what identifies which op in
// which chain, without having to name every intermediate. Armed for a single
// step because the callback serialises the whole graph: absolute times are
// inflated, so ONLY the relative shares mean anything, and the printout says so.
struct DitOpProf {
    bool                                          armed = false;
    long long                                     t0    = 0;
    std::map<std::string, std::pair<long long, int>> agg;  // key -> (us, count)
};

static bool dit_op_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    DitOpProf * p = (DitOpProf *) ud;
    if (ask) {
        p->t0 = ggml_time_us();
        return true;  // yes, call us back once it has been computed
    }
    const long long dt = ggml_time_us() - p->t0;
    char            key[160];
    snprintf(key, sizeof(key), "%-14s [%lld,%lld,%lld,%lld]", ggml_op_name(t->op), (long long) t->ne[0],
             (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    std::pair<long long, int> & e = p->agg[key];
    e.first += dt;
    e.second++;
    return true;
}

static void dit_op_prof_print(DitOpProf & p) {
    std::vector<std::pair<std::string, std::pair<long long, int>>> v(p.agg.begin(), p.agg.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string, std::pair<long long, int>> & a,
                 const std::pair<std::string, std::pair<long long, int>> & b) {
                  return a.second.first > b.second.first;
              });
    long long tot = 0;
    for (size_t i = 0; i < v.size(); i++) {
        tot += v[i].second.first;
    }
    fprintf(stderr,
            "[train-dit] op profile: %.0f ms over %zu op/shape keys — eval-callback SERIALISED, so absolute times are "
            "inflated and only the shares are meaningful\n",
            (double) tot / 1000.0, v.size());
    for (size_t i = 0; i < v.size() && i < 24; i++) {
        fprintf(stderr, "[train-dit]   %8.1f ms  %5.1f%%  x%-6d %s\n", (double) v[i].second.first / 1000.0,
                tot > 0 ? 100.0 * (double) v[i].second.first / (double) tot : 0.0, v[i].second.second,
                v[i].first.c_str());
    }
    // Per-OP roll-up. The shape-keyed table above scatters one op across dozens
    // of rows (a LoKR step has 235 keys), so "how much does CONT cost" is not
    // answerable from it. This collapses the key at the first space.
    std::map<std::string, std::pair<long long, int>> byop;
    for (size_t i = 0; i < v.size(); i++) {
        const size_t sp = v[i].first.find(' ');
        std::pair<long long, int> & e = byop[v[i].first.substr(0, sp == std::string::npos ? v[i].first.size() : sp)];
        e.first += v[i].second.first;
        e.second += v[i].second.second;
    }
    std::vector<std::pair<std::string, std::pair<long long, int>>> o(byop.begin(), byop.end());
    std::sort(o.begin(), o.end(),
              [](const std::pair<std::string, std::pair<long long, int>> & a,
                 const std::pair<std::string, std::pair<long long, int>> & b) {
                  return a.second.first > b.second.first;
              });
    fprintf(stderr, "[train-dit] op profile by OP:");
    for (size_t i = 0; i < o.size(); i++) {
        fprintf(stderr, " %s %.1f (%.1f%%, x%d)", o[i].first.c_str(), (double) o[i].second.first / 1000.0,
                tot > 0 ? 100.0 * (double) o[i].second.first / (double) tot : 0.0, o[i].second.second);
    }
    fprintf(stderr, "\n");
}

struct DitTrainOutcome {
    int       epochs_run        = 0;
    double    final_loss        = -1.0;
    double    best_loss         = -1.0;
    int       best_epoch        = 0;
    // What is ACTUALLY sitting in out_dir when the run ends (2026-07-30).
    // The adapter is no longer whatever the last epoch produced: it is the
    // best-scoring one, or the epoch that tripped the target if one did.
    // saved_ma5 is the ma5 at that epoch, so it is directly comparable to
    // target_loss; saved_reason is "target" or "best".
    double      saved_ma5    = -1.0;
    int         saved_epoch  = 0;
    std::string saved_reason;
    bool      stopped_on_target = false;
    int       samples           = 0;
    int       crop              = 0;
    int       layers            = 0;
    int       batch             = 1;
    int       ckpt              = 1;  // checkpoint segments that actually ran (1 = off)
    bool      exported          = false;
    int       export_tensors    = 0;
    long long ms                = 0;
};

// Milestone directory label. %.1f is right for the 0.1 default and wrong for
// anything finer, so the number of decimals tracks --milestone-step.
static std::string dit_milestone_label(double v, float step) {
    int dec = 1;
    for (int d = 1; d <= 4; d++) {
        dec                 = d;
        const double scaled = (double) step * pow(10.0, d);
        if (fabs(scaled - floor(scaled + 0.5)) < 1e-6) {
            break;  // `step` is representable with d decimals
        }
    }
    char b[48];
    snprintf(b, sizeof(b), "%.*f", dec, v);
    return std::string(b);
}

// ─── the training stage ─────────────────────────────────────────────────────

static int dit_train_stage(const DitTrainArgs & a, DitTrainLog * log, DitTrainOutcome * out) {
    const int64_t       t_stage0 = ggml_time_ms();
    const bool          is_lokr  = (a.adapter_type == "lokr");
    // Not const: the CUDA-only guard below downgrades this to DIT_MIRROR_F32
    // on a non-CUDA backend instead of refusing the run outright.
    DitMirrorMode mirror_mode = (a.mirror == "bf16")       ? DIT_MIRROR_BF16
                                : (a.mirror == "bf16-f32") ? DIT_MIRROR_BF16_F32
                                                           : DIT_MIRROR_F32;
    jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"begin\",\"total\":%d}", a.epochs);

    // ── samples (host-resident for the whole run) ────────────────────────
    std::vector<DitSample> samples;
    {
        std::string err;
        if (!dit_scan_samples(a.tensors_dir.c_str(), a.limit, &samples, &err)) {
            lm_fatal("no-samples", err);
            return 1;
        }
    }
    DitChannelStats cstats;
    const bool      have_cstats = dit_load_channel_stats(a.tensors_dir.c_str(), &cstats);
    const bool      chan_bal    = a.channel_balance && have_cstats;
    if (a.channel_balance && !have_cstats) {
        lm_log("info", "no channel_stats.json in the variant — channel balancing disabled (Side-Step does the same)");
    }

    // Pad every encoder state to one common enc_S so the graph shape is constant.
    //
    // DEVIATION (E4, documented): §3.1 leaves enc_S per-song. A varying enc_S
    // changes the graph shape every micro-step, which makes ggml_gallocr grow
    // mid-epoch (the exact "40 songs in, OOM" the high-water probe exists to
    // prevent) and makes the footprint model unfalsifiable. Padding to enc_S_max
    // and zeroing the padded positions in the encoder-padding mask is NUMERICALLY
    // EXACT — soft_max_ext gives masked positions exactly zero weight and exactly
    // zero gradient (§3.4 item 1) — and costs only cross-attention FLOPs on the
    // shorter prompts.
    int enc_S = 0, enc_H = 0, max_T = 0, min_T = 0;
    int genre_samples = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        enc_S = std::max(enc_S, std::max(samples[i].enc_S, samples[i].enc_S_genre));
        enc_H = std::max(enc_H, samples[i].enc_H);
        max_T = std::max(max_T, samples[i].T);
        min_T = (i == 0) ? samples[i].T : std::min(min_T, samples[i].T);
        if (!samples[i].enc_genre.empty()) {
            genre_samples++;
        }
    }
    // enc_S is the WHOLE DATASET's padded length, so ONE long-lyrics song sets it
    // for every sample. That matters because attention K/V scale with (S + enc_S)
    // and the VRAM auto-fit reduces CROP before depth — a long encoder sequence
    // is therefore paid for in audio window, silently, unless it is said out loud.
    // Threshold is the reference pipeline's own buffer (512 lyric + 1 + 256 text).
    if (enc_S > 769) {
        char eb[224];
        snprintf(eb, sizeof(eb),
                 "encoder sequence enc_S=%d exceeds the reference 769 (one song's lyrics/caption set this for the "
                 "WHOLE dataset). Cross-attention and the crop the auto-fit can afford both pay for it — lower "
                 "--max-lyric-tokens if the fitted crop came out shorter than you wanted",
                 enc_S);
        lm_log("warn", eb);
        fprintf(stderr, "[train-dit] %s\n", eb);
    }
    // Artist token: k EXTRA encoder positions on the end of every sample. They
    // are zero in enc (the parameter is accumulated into them in-graph) but
    // UNMASKED in enc_mask — the opposite of ordinary padding. dit_ca_mask
    // writes -INFINITY wherever the mask says padding, so leaving these at 0
    // would mask the token out of every cross-attention and hand it exactly
    // zero gradient, which reads as "the learning rate is too low" rather than
    // as a wiring bug.
    const int art_k      = a.artist_token.empty() ? 0 : a.artist_k;
    const int enc_S_base = enc_S;
    enc_S += art_k;
    for (size_t i = 0; i < samples.size(); i++) {
        DitSample & s = samples[i];
        s.enc.resize((size_t) enc_S * (size_t) enc_H, 0.0f);
        s.enc_mask.resize((size_t) enc_S, 0.0f);
        s.enc_S = enc_S;
        for (int j = 0; j < art_k; j++) {
            s.enc_mask[(size_t) (enc_S_base + j)] = 1.0f;
        }
        if (!s.enc_genre.empty()) {
            s.enc_genre.resize((size_t) enc_S * (size_t) enc_H, 0.0f);
            s.enc_mask_genre.resize((size_t) enc_S, 0.0f);
            s.enc_S_genre = enc_S;
            for (int j = 0; j < art_k; j++) {
                s.enc_mask_genre[(size_t) (enc_S_base + j)] = 1.0f;
            }
        }
    }
    const int n = (int) samples.size();

    // ── load the base on the CPU backend (D3) ────────────────────────────
    DitTrainModel M;
    {
        std::string err;
        if (!dit_train_load(&M, a.dit_path.c_str(), /*lora_lo=*/0, &err)) {
            if (err == "convrot") {
                lm_fatal("convrot-unsupported",
                         "this DiT base carries acestep.convrot_map — ConvRot bases are not trainable in v1 "
                         "(adapter-merge.h already refuses merge mode on them)");
            } else {
                lm_fatal("model-load", err);
            }
            dit_train_free(&M);
            return 1;
        }
    }
    const DiTGGMLConfig & c   = M.m.cfg;
    const int             H   = c.hidden_size, Oc = c.out_channels, P = c.patch_size, L = c.n_layers;
    jl("{\"type\":\"model\",\"stage\":\"dit\",\"path\":\"%s\",\"ms\":%lld,\"layers\":%d,\"hidden\":%d,\"heads\":%d,"
       "\"kvHeads\":%d,\"headDim\":%d,\"inCh\":%d,\"outCh\":%d,\"patch\":%d,\"slidingWindow\":%d,\"convrot\":false}",
       lm_json_escape(a.dit_path).c_str(), M.load_ms, L, H, c.n_heads, c.n_kv_heads, c.head_dim, c.in_channels, Oc, P,
       c.sliding_window);

    if (enc_H != (int) M.m.cond_emb_w->ne[0]) {
        char b[192];
        snprintf(b, sizeof(b), "cached encoder states are [.., %d] but this DiT's condition_embedder wants %d — the "
                               "variant was preprocessed against a different base",
                 enc_H, (int) M.m.cond_emb_w->ne[0]);
        lm_fatal("model-load", b);
        dit_train_free(&M);
        return 1;
    }
    if (samples[0].Oc != Oc || samples[0].Cc + Oc != c.in_channels) {
        char b[192];
        snprintf(b, sizeof(b), "cached latents are [%d target / %d context] but this DiT wants %d/%d", samples[0].Oc,
                 samples[0].Cc, Oc, c.in_channels - Oc);
        lm_fatal("model-load", b);
        dit_train_free(&M);
        return 1;
    }

    // ── the trainer's own compute backend ────────────────────────────────
    {
        std::string err;
        if (!dit_train_backend_init(&M, &err)) {
            lm_fatal("vram", err);
            dit_train_free(&M);
            return 1;
        }
    }

    // ── --mirror bf16 is CUDA-only ───────────────────────────────────────
    // Only ggml-cuda's out_prod carries the BF16 patch; the CPU and Vulkan
    // implementations still assert F32 src0, so a BF16 mirror there would abort
    // deep inside the first backward pass. bf16 is now the server's default
    // mirror choice (BF16 training defaults, 2026-07-29), and CPU/Vulkan
    // release builds exist, so this is no longer a rare CLI misuse — soften
    // from a fatal refusal to a graceful fallback: warn and continue the run
    // with the f32 mirror. Downstream reporting (vram JSONL, mirror log line,
    // dit_train_log.json) must reflect the ACTUAL mode used, never the
    // requested one — hence reassigning mirror_mode/log->mirror here rather
    // than just skipping the refusal.
    //
    // `bf16-f32` falls back for a different reason and is held to the same rule.
    // It never reaches out_prod with a BF16 tensor (the graph hands the matmul an
    // F32 cast), but it does need the backend to run a BF16 -> F32 CPY at weight
    // shapes. ggml-cuda does (cpy.cu / supports_op); Vulkan's coverage is not
    // established, and a `false` there does NOT abort — backend_sched would split
    // those casts onto the CPU and the run would look fine while crawling. Rather
    // than ship that trap, non-CUDA takes the same graceful f32 fallback.
    if (dit_mirror_stores_bf16(mirror_mode) && strncmp(ggml_backend_name(M.backend), "CUDA", 4) != 0) {
        char b[320];
        snprintf(b, sizeof(b),
                 "%s mirror requires CUDA — falling back to f32 mirror (this run picked '%s'; %s).",
                 dit_mirror_mode_name(mirror_mode), ggml_backend_name(M.backend),
                 mirror_mode == DIT_MIRROR_BF16
                     ? "only ggml-cuda's out_prod carries the BF16 patch — see engine/patches/bf16-out-prod.patch"
                     : "only ggml-cuda is known to run the in-graph BF16 -> F32 weight cast on the device");
        lm_log("warn", b);
        mirror_mode  = DIT_MIRROR_F32;
        log->mirror  = "f32";
    }

    // ── honest device VRAM (gpu-mem.h) ───────────────────────────────────
    // cudaMemGetInfo counts another process's EVICTABLE memory as free under
    // WDDM — measured 31 GB "free" against nvidia-smi's 22.4 GB on this card.
    // Everything that sizes or polices the run reads NVML when it is available.
    const DitGpuMem gmem = dit_gpu_mem_query(M.backend);

    // ── card support gate: NOT HERE ANY MORE (D9 retired 2026-09-02) ─────
    //
    // A flat `total_mb >= 16384` check used to sit at this line and refuse the
    // card before the footprint model had priced anything. It is gone: the gate
    // is now the auto-fit's own floor (dit_vram_fit's floor_bytes vs
    // budget_bytes) and it is applied at the `!fit.ok` refusal below, where the
    // run's mirror precision, attention formulation, checkpoint segments, depth
    // and adapter size are all known and all count. See the retirement note at
    // the bottom of dit-vram.h for why the constant stopped being true.

    // ── VRAM auto-fit (D10/D11) ──────────────────────────────────────────
    DitVramModel vm;
    vm.m           = &M.m;
    vm.n_layers    = L;
    vm.patch       = P;
    vm.rank        = a.rank;
    vm.target_mlp  = a.target_mlp;
    vm.is_lokr     = is_lokr;
    vm.lokr_dim    = a.lokr_dim;
    vm.lokr_factor = a.lokr_factor;
    vm.mirror      = mirror_mode;
    vm.in_ch       = c.in_channels;
    vm.out_ch      = Oc;
    vm.hidden      = H;
    vm.enc_H       = enc_H;
    vm.enc_S       = enc_S;
    vm.n_heads     = c.n_heads;
    vm.n_kv_heads  = c.n_kv_heads;
    vm.head_dim    = c.head_dim;
    // A batch wider than the dataset cannot be assembled from B DIFFERENT songs
    // (design B2), so clamp before the fit sizes anything against it.
    vm.batch       = std::max(1, std::min(a.batch, n));
    if (vm.batch < a.batch) {
        char bb[192];
        snprintf(bb, sizeof(bb), "--batch %d reduced to %d: a micro-batch takes B DIFFERENT songs and this variant has "
                                 "%d",
                 a.batch, vm.batch, n);
        lm_log("warn", bb);
    }

    // Which arena model prices this run (--attn). The mode is resolved HERE, not
    // at the probe below, because the footprint model and the crop walk are the
    // whole point of the flag: flash removes both retained softmaxes, and pricing
    // a flash run with the exact polynomial yields the exact run's crop.
    // "flash" and "flash-f32" are the same GRAPH and the same VRAM footprint —
    // they differ only in the arithmetic op_params slot 3 asks the kernels for —
    // so both take the flash arena model and the lifted crop cap.
    const bool flash_attn = (a.attn == "flash" || a.attn == "flash-f32");
    vm.flash_attn         = flash_attn;
    const ggml_prec attn_prec_req = (a.attn == "flash-f32") ? GGML_PREC_F32 : GGML_PREC_DEFAULT;

    // ── flash mode lifts the crop_max cap (plan doc §"Flash-aware VRAM model" 2)
    //
    // crop_max defaults to 1250 and CLAMPS the walk (dit_vram_fit's `cap`), a
    // number chosen when a 1250-frame crop was the most a 32 GB card could hold
    // at full depth. With the S^2 term gone that cap, not the VRAM, is what
    // decides the crop — the model would be moot. So in flash mode the DEFAULT
    // cap lifts to the dataset's longest track; an explicit --crop-max always
    // wins, in both directions.
    // 2026-09-03: the lift is GONE. Long auto-fit crops (~1500) lost three
    // clean ear A/Bs to short ones (see crop_max above); the cap is a quality
    // guard now, not a VRAM one. An explicit --crop-max still wins both ways.
    int crop_max_eff = a.crop_max;
    if (flash_attn && !a.crop_max_user && max_T > a.crop_max) {
        char cb[224];
        snprintf(cb, sizeof(cb),
                 "--attn %s: crop-max stays at the %d default (longest track %d frames) — long crops under-train "
                 "quiet passages; pass --crop-max to raise it, ideally with --crop-jitter",
                 a.attn.c_str(), a.crop_max, max_T);
        lm_log("info", cb);
        fprintf(stderr, "[train-dit] %s\n", cb);
    }

    // The C4 auto-fit ORDER at a fixed B: full depth, shrinking the crop; then
    // raising the segment count; then reducing B with a warn; and only when all
    // of that has failed, the depth ladder — depth is the quality axis (Rob's
    // rule), so it is the last thing given up. `depth_ladder=false` is what holds
    // the earlier steps ahead of it.
    auto run_fit = [&](bool allow_depth) {
        return dit_vram_fit(vm, M.backend, a.vram_reserve_mb, a.vram_safety, a.crop, a.layers, a.crop_min, crop_max_eff,
                            max_T, &gmem, a.ckpt, allow_depth);
    };
    auto ordered_fit = [&]() {
        DitVramFit f = run_fit(false);
        while (!f.ok && vm.batch > 1) {
            char bb[224];
            snprintf(bb, sizeof(bb),
                     "--batch %d reduced to %d: the widest checkpoint split at the shortest crop still does not fit "
                     "the VRAM budget at full depth (C4 order: batch gives before depth)",
                     vm.batch, vm.batch - 1);
            lm_log("warn", bb);
            vm.batch -= 1;
            f = run_fit(false);
        }
        if (!f.ok) {
            f = run_fit(true);
        }
        return f;
    };
    DitVramFit fit = ordered_fit();
    // A1's CUDA repeat_back cap is a CORRECTNESS constraint on
    // Nkv * max(S, enc_S) * B (exceeding it aborts mid-backward), so it is applied
    // after the crop is known and the fit re-run at the reduced batch — a smaller
    // B frees budget, which the crop walk should be allowed to spend, and a longer
    // crop can re-trip the cap. BOTH attentions expand K/V: self-attention over S
    // tokens, cross-attention over enc_S, and enc_S is dataset geometry the crop
    // walk never touches. Terminates: vm.batch strictly decreases and
    // dit_vram_max_batch never returns below 1, at which point it is a no-op.
    //
    // HOT-Step patch: flash-attn-train (investigation B2) — `flash_attn` disarms
    // the whole loop. There is no expansion in flash mode, so there is no
    // repeat_back node and no kernel cap to respect; the only thing left holding
    // B down is VRAM, which ordered_fit() already prices.
    for (int it = 0; it < 8 && fit.ok && fit.crop > 0; it++) {
        const int S_fit = fit.crop / P;
        const int cap   = dit_vram_max_batch(c.n_heads, c.n_kv_heads, S_fit, enc_S, vm.batch, flash_attn);
        if (cap >= vm.batch) {
            break;
        }
        char bb[288];
        snprintf(bb, sizeof(bb),
                 "--batch %d reduced to %d: ggml's CUDA repeat_back caps n_kv_heads*max(S,enc_S)*B at %d and this run "
                 "is %d*max(%d,%d)*%d (A1 expanded-KV backward)",
                 vm.batch, cap, DIT_REPEAT_BACK_MAX, c.n_kv_heads, S_fit, enc_S, vm.batch);
        lm_log("warn", bb);
        vm.batch = cap;
        fit      = ordered_fit();
    }
    // The loop is bounded, so it must never be able to EXIT still violating the
    // cap. A hard clamp with no re-fit closes that: shrinking B only ever lowers
    // every term of the footprint model, so the fit stays valid (just
    // conservative) at the smaller batch.
    if (fit.ok && fit.crop > 0) {
        const int S_fit = fit.crop / P;
        const int cap   = dit_vram_max_batch(c.n_heads, c.n_kv_heads, S_fit, enc_S, vm.batch, flash_attn);
        if (cap < vm.batch) {
            char bb[288];
            snprintf(bb, sizeof(bb),
                     "--batch %d hard-clamped to %d after the repeat_back re-fit loop ran out of iterations; the fit "
                     "is kept as-is (a smaller batch only frees VRAM)",
                     vm.batch, cap);
            lm_log("warn", bb);
            vm.batch = cap;
        }
    }
    // ── the refusal, and the ONLY VRAM gate (D9's flat 16 GB check retired) ─
    //
    // `!fit.ok` means the walk exhausted every axis it is allowed to move, which
    // is the same statement as "floor_bytes does not fit budget_bytes" — so this
    // is the fit-derived gate, not merely its error path. Three things it must
    // say, because a refusal that only prints a number leaves the user guessing
    // which of a dozen flags to reach for:
    //   1. the floor and the CONFIGURATION it is (depth/crop/segments/batch, plus
    //      the choices the fit is not allowed to move: adapter, mirror, attn);
    //   2. the measured free VRAM and the subtraction that turned it into a
    //      budget (reserve, safety) — the reserve in particular is a knob;
    //   3. the two or three settings that would lower the floor most, PRICED.
    if (!fit.ok) {
        const bool          bf16_ok = strncmp(ggml_backend_name(M.backend), "CUDA", 4) == 0;
        const DitVramAdvice adv     = dit_vram_floor_advice(vm, fit, a.ckpt, bf16_ok);
        // How much of the floor is the frozen base itself. When this is most of
        // it — and on a 32-layer XL base with a bf16 mirror it is ~7.9 GB of an
        // ~8.3 GB floor — no run setting can help and the message must not
        // pretend otherwise; the only routes left are a bigger card or a
        // smaller/quantised base.
        const long long mirror_mb =
            (long long) (dit_mirror_bytes_for(&M.m, L - fit.floor_layers, mirror_mode, false) / 1048576);
        const long long floor_mb  = (long long) (fit.floor_bytes / 1048576.0);
        const long long budget_mb = (long long) (fit.budget_bytes / 1048576.0);
        const long long short_mb  = floor_mb - budget_mb;
        const long long all_mb    = (long long) (adv.all_bytes / 1048576.0);
        char            extra[352];
        snprintf(extra, sizeof(extra),
                 ",\"needMb\":%lld,\"freeMb\":%lld,\"floorMb\":%lld,\"budgetMb\":%lld,\"totalMb\":%lld,"
                 "\"mirrorMb\":%lld,\"allLeversMb\":%lld,\"floorCrop\":%d,\"floorLayers\":%d,\"floorSegments\":%d",
                 floor_mb, (long long) fit.free_mb, floor_mb, budget_mb, (long long) fit.total_mb, mirror_mb, all_mb,
                 fit.floor_crop, fit.floor_layers, fit.floor_segments);
        char adapter[96];
        if (is_lokr) {
            snprintf(adapter, sizeof(adapter), "LoKR dim %d/factor %d%s", a.lokr_dim, a.lokr_factor,
                     a.target_mlp ? "+MLP" : "");
        } else {
            snprintf(adapter, sizeof(adapter), "LoRA rank %d%s", a.rank, a.target_mlp ? "+MLP" : "");
        }
        // The closing sentence: either the levers can close the gap, or they
        // cannot and the base is the wall.
        char tail[352];
        if (adv.n == 0) {
            snprintf(tail, sizeof(tail), "Nothing left to trade: every VRAM lever this trainer has is already pulled, "
                                         "and %lld MB of the floor is the frozen base's mirror. A bigger card or a "
                                         "smaller (or quantised) base is the only way from here.",
                     mirror_mb);
        } else if (all_mb > budget_mb) {
            snprintf(tail, sizeof(tail),
                     "Biggest levers left: %s — but all of them together only reach %lld MB, still %lld MB over, and "
                     "%lld MB of the floor is the frozen base's mirror. That is the wall here, not the run's settings.",
                     adv.text.c_str(), all_mb, all_mb - budget_mb, mirror_mb);
        } else {
            snprintf(tail, sizeof(tail), "Biggest levers left: %s (%lld MB short; all of them together reach %lld MB, "
                                         "which fits).",
                     adv.text.c_str(), short_mb, all_mb);
        }
        char b[900];
        snprintf(b, sizeof(b),
                 "not enough free VRAM: the smallest configuration this run can express needs ~%lld MB and the budget "
                 "is %lld MB (%lld MB short). Floor = %d layers, crop %d, %d checkpoint segment%s, batch 1, %s, %s "
                 "mirror, --attn %s — of which the frozen-weight mirror is %lld MB. Card %lld MB total, %lld MB free "
                 "(%s), minus %d MB reserve, minus %.0f%% safety. %s",
                 floor_mb, budget_mb, short_mb, fit.floor_layers, fit.floor_crop, fit.floor_segments,
                 fit.floor_segments == 1 ? "" : "s", adapter, dit_mirror_mode_name(mirror_mode),
                 a.attn.c_str(), mirror_mb, (long long) fit.total_mb, (long long) fit.free_mb, fit.free_source,
                 a.vram_reserve_mb, (double) a.vram_safety * 100.0, tail);
        lm_fatal("vram", b, extra);
        dit_train_free(&M);
        return 1;
    }
    const int K        = fit.layers;
    const int crop_len = fit.crop;
    const int lora_lo  = L - K;
    const int S_max    = crop_len / P;
    const int B        = std::max(1, vm.batch);
    out->crop          = crop_len;
    out->layers        = K;
    out->batch         = B;
    // Only the TRAINED range is segmented (dit-train-ckpt.h): the layers below it
    // hold no adapter parameter, so no gradient flows through them and a segment
    // that spanned only those would trip build_backward_expand's any_params
    // assert. dit_ckpt_plan clamps the count to that range.
    const DitCkptPlan ckplan = dit_ckpt_plan(lora_lo, L, (a.ckpt == 0) ? 1 : std::max(1, fit.segments));
    const int         SEG    = ckplan.segments;
    out->ckpt                = SEG;

    // ── A1 pre-flight (hard) ─────────────────────────────────────────────
    // The fit above is supposed to have made this impossible. It is checked again
    // here, on the FINAL numbers, because the failure mode it guards is a GGML
    // abort() inside the CUDA backward — the process dies with no JSONL, no
    // `fatal`, and no way for the server to say why. A clean fatal beats that
    // even if the branch is never taken. Only B > 1 expands K/V at all.
    //
    // HOT-Step patch: flash-attn-train (investigation B2) — and only EXACT mode.
    // This is the last of three places the cap is applied (the other two are the
    // re-fit loop and its hard clamp above); leaving it armed in flash mode would
    // refuse exactly the batches the fused ops make legal. On this dataset that
    // is not hypothetical: enc_S is 1877, so 8*1877*4 = 60064 and B = 4 was
    // refused outright, at any crop, because cross-attention's expanded K/V — not
    // the crop, not the VRAM — blew the REPEAT_BACK kernel's limit.
    if (B > 1 && !flash_attn && c.n_heads != c.n_kv_heads) {
        const long long tok  = (long long) std::max(S_max, enc_S);
        const long long prod = (long long) c.n_kv_heads * tok * (long long) B;
        if (prod > DIT_REPEAT_BACK_MAX) {
            char extra[128];
            snprintf(extra, sizeof(extra), ",\"batch\":%d,\"S\":%d,\"encS\":%d", B, S_max, enc_S);
            char b[320];
            snprintf(b, sizeof(b),
                     "batch %d cannot run: ggml's CUDA repeat_back caps n_kv_heads*max(S,enc_S)*B at %d and this "
                     "configuration is %d*%lld*%d = %lld (S %d, enc_S %d) — lower --batch or --crop-max",
                     B, DIT_REPEAT_BACK_MAX, c.n_kv_heads, tok, B, prod, S_max, enc_S);
            lm_fatal("vram", b, extra);
            dit_train_free(&M);
            return 1;
        }
    }

    // ── the streamed mirror, built ONCE at the chosen depth ──────────────
    // `B > 1` is A2c: proj_in/cond_emb are promoted to F32 only when a batch axis
    // exists, so the §2.3.1 anchor at --batch 1 keeps the exact mirror it had.
    {
        std::string err;
        if (!dit_build_mirror(&M, lora_lo, mirror_mode, &err, B > 1)) {
            lm_fatal(err.find("quantized") != std::string::npos ? "model-load" : "vram", err);
            dit_train_free(&M);
            return 1;
        }
    }
    fprintf(stderr,
            "[train-dit] mirror (%s): %d promoted to F32 (%.1f MB) + %d kept native, %.1f MB total; device-wide %zu "
            "MB\n",
            dit_mirror_mode_name(mirror_mode), M.mirror.n_f32, M.mirror.bytes_f32 / 1048576.0, M.mirror.n_keep,
            M.mirror.bytes / 1048576.0, dit_gpu_mem_query(M.backend).used_mb());

    // ── scheduler ────────────────────────────────────────────────────────
    {
        BackendPair bp;
        bp.backend     = M.backend;
        bp.cpu_backend = M.cpu;
        bp.has_gpu     = true;
        M.sched        = backend_sched_new(bp, 65536);
        if (!M.sched) {
            lm_fatal("vram", "cannot create the training scheduler");
            dit_train_free(&M);
            return 1;
        }
    }

    // ── --attn flash: capability probe (spec §9.8) ─────────────────────
    //
    // Asked HERE, right after the scheduler exists and before a single graph is
    // built. The scheduler above registers the CPU backend as a second backend,
    // so an unsupported op does not fail: it silently splits onto the CPU, and a
    // CPU-split run has low VRAM, a quiet NVML tripwire and correct-looking
    // losses. Every measurement downstream would be poisoned while reading as a
    // pass. Both directions are probed, at BOTH attention shapes the run will
    // actually emit (self-attention S_kv = S, cross-attention S_kv = enc_S), with
    // the effective Nkv.
    //
    // HOT-Step patch: flash-attn-train (investigation B2) — that Nkv is now
    // c.n_kv_heads at EVERY B. The expansion to Nh only ever existed to keep
    // ggml's mul_mat backward alive at B > 1, and dit_attn_needs_kv_expand()
    // no longer performs it in flash mode, so the shape the probe asks about has
    // to be the native GQA one or it is probing a geometry the run never emits.
    const DitAttnMode attn_mode = flash_attn ? DIT_ATTN_FLASH : DIT_ATTN_EXACT;
    if (attn_mode == DIT_ATTN_FLASH) {
        const int   Nkv_eff = c.n_kv_heads;
        const float ascale  = 1.0f / sqrtf((float) c.head_dim);
        bool        sf = false, sb = false, cf = false, cb = false;
        dit_flash_probe(M.backend, c.head_dim, c.n_heads, Nkv_eff, S_max, S_max, B, ascale, &sf, &sb);
        dit_flash_probe(M.backend, c.head_dim, c.n_heads, Nkv_eff, S_max, enc_S, B, ascale, &cf, &cb);
        if (!(sf && sb && cf && cb)) {
            char extra[256];
            snprintf(extra, sizeof(extra),
                     ",\"attn\":\"%s\",\"selfFwd\":%s,\"selfBwd\":%s,\"crossFwd\":%s,\"crossBwd\":%s",
                     a.attn.c_str(), sf ? "true" : "false", sb ? "true" : "false", cf ? "true" : "false",
                     cb ? "true" : "false");
            char b[512];
            snprintf(b, sizeof(b),
                     "--attn %s: backend %s does not support the fused attention ops at this geometry "
                     "(D %d, Nh %d, Nkv %d, S %d, enc_S %d, B %d) — self fwd %s / bwd %s, cross fwd %s / bwd %s. "
                     "Refusing to start: the scheduler would silently run them on the CPU instead, which is "
                     "correct, unusably slow, and looks like a pass on every number this run reports. Use "
                     "--attn exact.",
                     a.attn.c_str(), ggml_backend_name(M.backend), c.head_dim, c.n_heads, Nkv_eff, S_max, enc_S,
                     B, sf ? "yes" : "NO", sb ? "yes" : "NO", cf ? "yes" : "NO", cb ? "yes" : "NO");
            lm_fatal("attn-unsupported", b, extra);
            dit_train_free(&M);
            return 1;
        }
        char b[448];
        snprintf(b, sizeof(b),
                 "--attn %s: %s supports FLASH_ATTN_TRAIN and FLASH_ATTN_TRAIN_BACK at D %d, Nh %d, Nkv %d, "
                 "S %d, enc_S %d, B %d — no CPU split. Requested arithmetic: %s (the backend resolves it per "
                 "launch; dit_train_log.json records what actually ran)",
                 a.attn.c_str(), ggml_backend_name(M.backend), c.head_dim, c.n_heads, Nkv_eff, S_max, enc_S, B,
                 attn_prec_req == GGML_PREC_F32 ? "strict f32" : "tf32 where available");
        lm_log("info", b);
        fprintf(stderr, "[train-dit] %s\n", b);
    }

    // ── static input buffers (1-D bases; every graph tensor is a CONTIGUOUS
    //    view at offset 0, because soft_max_ext will not take a strided mask) ─
    ggml_context * ctx_static;
    {
        ggml_init_params p = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx_static         = ggml_init(p);
    }
    // Every one is B-wide (design B1). b_sa is B-wide only for its PADDED shape:
    // with nothing padded the sliding-window mask is [S,S] with ne2 == ne3 == 1
    // and broadcasts over heads AND batch; a padded micro-batch needs one mask
    // per element (dit-data.h), so the base is allocated for the wide case.
    ggml_tensor * b_input = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) c.in_channels * crop_len * B);
    ggml_tensor * b_enc   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) enc_H * enc_S * B);
    ggml_tensor * b_pos   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_I32, (int64_t) S_max * B);
    ggml_tensor * b_temb  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) H * B);
    ggml_tensor * b_tproj = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) 6 * H * B);
    ggml_tensor * b_sa    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F16, (int64_t) S_max * S_max * B);
    // The window-free pad mask for the full-attention layers. Only B > 1 can ever
    // produce a padded element (a 1-element batch is padded to its own length), so
    // at B == 1 this is never allocated and the graph is the pre-batching one.
    ggml_tensor * b_sa_pad =
        (B > 1) ? ggml_new_tensor_1d(ctx_static, GGML_TYPE_F16, (int64_t) S_max * S_max * B) : nullptr;
    ggml_tensor * b_ca    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F16, (int64_t) enc_S * S_max * B);
    ggml_tensor * b_vtgt  = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) Oc * crop_len * B);
    ggml_tensor * b_lw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) crop_len * B);
    ggml_tensor * b_lwu   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, (int64_t) crop_len * B);
    ggml_tensor * t_cw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, Oc);
    ggml_tensor * t_adamw    = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 7);
    ggml_tensor * t_lossgrad = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_clip     = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_eps      = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_tensor * t_gnorm2   = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    for (ggml_tensor * t : { b_input, b_enc, b_pos, b_temb, b_tproj, b_sa, b_ca, b_vtgt, b_lw, b_lwu, t_cw }) {
        ggml_set_input(t);
    }
    if (b_sa_pad) {
        ggml_set_input(b_sa_pad);
    }
    ggml_backend_buffer_t buf_static = ggml_backend_alloc_ctx_tensors(ctx_static, M.backend);
    if (!buf_static) {
        lm_fatal("vram", "static input allocation failed");
        ggml_free(ctx_static);
        dit_train_free(&M);
        return 1;
    }
    {
        std::vector<float> cw((size_t) Oc, 1.0f);
        for (int i = 0; i < Oc && i < (int) cstats.weight.size(); i++) {
            cw[(size_t) i] = cstats.weight[(size_t) i];
        }
        ggml_backend_tensor_set(t_cw, cw.data(), 0, cw.size() * sizeof(float));
        const float epsv = 1e-6f, clipv = a.grad_clip, lg = 1.0f;
        ggml_backend_tensor_set(t_eps, &epsv, 0, 4);
        ggml_backend_tensor_set(t_clip, &clipv, 0, 4);
        ggml_backend_tensor_set(t_lossgrad, &lg, 0, 4);
    }

    // ── adapter + optimizer ──────────────────────────────────────────────
    DitAdapterLora lora;
    DitAdapterLoKr lokr;
    DitAdapter *   adapter = is_lokr ? (DitAdapter *) &lokr : (DitAdapter *) &lora;
    {
        DitAdapterCfg cfg;
        cfg.rank                = a.rank;
        cfg.alpha               = (float) a.alpha;
        cfg.target_mlp          = a.target_mlp;
        cfg.dora                = a.dora;
        if (a.dora && a.adapter_type != "lora") {
            lm_fatal("args", "--dora applies to the LoRA parameterization only (LoKr has no per-row direction to rescale)");
            return 1;
        }
        if (a.dora && !a.init_adapter.empty()) {
            // dit-resume.h restores A/B only. m is re-derived as ||W||_col here, so
            // at resume the ratio m/||W+BA|| is not the 1.0 a fresh run starts from.
            lm_log("warn", "DoRA: lora_magnitude_vector is NOT resumed from --init-adapter; m restarts at ||W||_col");
        }
        cfg.seed                = (uint64_t) a.seed;
        cfg.lokr_dim            = a.lokr_dim;
        cfg.lokr_alpha          = a.lokr_alpha;
        cfg.lokr_factor         = a.lokr_factor;
        cfg.lokr_decompose_both = a.lokr_decompose_both;
        std::string err;
        if (!adapter->init(&M.m, M.backend, lora_lo, L, cfg, &err)) {
            lm_fatal("unsupported-adapter", err);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
        // --mirror bf16-f32: tell the parameterization seam which layers get the
        // in-graph BF16 -> F32 weight cast. Set here, once, from the RESOLVED
        // mirror mode (the CUDA fallback above may already have downgraded it),
        // and left at INT_MAX in every other mode so the graph is unchanged.
        adapter->f32_cast_lo = dit_mirror_casts_in_graph(mirror_mode) ? lora_lo : INT_MAX;
        size_t expect =
            is_lokr ? dit_lokr_expected_params(c, lora_lo, L, a.lokr_dim, a.lokr_factor, a.target_mlp)
                    : dit_lora_expected_params(c, lora_lo, L, a.rank, a.target_mlp);
        if (a.dora && !is_lokr) {
            // DoRA adds one magnitude per output row at every trained site. The
            // VRAM model is left unaware: ~1.3M floats on the 32-layer DiT, and
            // its optimizer state, is noise against the LoRA itself.
            const int n_sites = a.target_mlp ? DIT_NSITES : DIT_NSITES_ATTN;
            for (int l = lora_lo; l < L; l++) {
                for (int s2 = 0; s2 < n_sites; s2++) {
                    ggml_tensor * w = dit_site_weight(&M.m.layers[l], s2);
                    expect += w ? (size_t) w->ne[1] : 0;
                }
            }
        }
        if (adapter->nParams() != expect) {
            char b[192];
            snprintf(b, sizeof(b), "%s parameter count %zu != expected %zu", adapter->typeName(), adapter->nParams(),
                     expect);
            lm_fatal("unsupported-adapter", b);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
    }
    // Resume: overwrite the fresh init with the source run's factors. After
    // adapter->init so the tensors exist; coverage-tolerant (see dit-resume.h —
    // the trained layer window may differ between runs).
    if (!a.init_adapter.empty()) {
        DitResumeStats rs;
        std::string    err;
        const bool     rok = is_lokr ? dit_resume_load_lokr(&lokr, a.init_adapter, &rs, &err)
                                     : dit_resume_load_lora(&lora, a.init_adapter, &rs, &err);
        if (!rok) {
            lm_fatal("resume", err);
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
        if (rs.loaded == 0) {
            lm_fatal("resume", "no tensors from " + a.init_adapter +
                                   " overlap this run's trained layer window — nothing was resumed");
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
        char b[320];
        snprintf(b, sizeof(b),
                 "resumed %d tensors from %s (source ma5 %.4f); %d site(s) start fresh, %d file layer(s) outside "
                 "this window",
                 rs.loaded, a.init_adapter.c_str(), a.init_from_ma5, rs.fresh_sites, rs.skipped_file);
        lm_log("info", b);
        jl("{\"type\":\"resume\",\"initAdapter\":\"%s\",\"tensors\":%d,\"freshSites\":%d,\"sourceMa5\":%.6f}",
           lm_json_escape(a.init_adapter).c_str(), rs.loaded, rs.fresh_sites, a.init_from_ma5);
    }
    const DitAdapter * ad = adapter;

    // ── Artist token parameter ────────────────────────────────────────────
    //
    // [enc_H, k] F32, zero-init. Zero means the appended encoder rows are all
    // zeros, which after the frozen cond_emb is a constant bias row — so step 0
    // is a well-defined starting point rather than noise, and a run that never
    // moved is visibly distinct from one that moved wrongly.
    ggml_tensor *         t_art   = nullptr;
    ggml_context *        art_ctx = nullptr;
    ggml_backend_buffer_t art_buf = nullptr;
    if (art_k > 0) {
        ggml_init_params ap = { 2 * ggml_tensor_overhead(), nullptr, true };
        art_ctx             = ggml_init(ap);
        if (!art_ctx) {
            lm_fatal("vram", "cannot create the artist-token context");
            adapter->free();
            dit_train_free(&M);
            return 1;
        }
        t_art = ggml_new_tensor_2d(art_ctx, GGML_TYPE_F32, enc_H, art_k);
        ggml_set_name(t_art, "artist_token");
        ggml_set_param(t_art);
        art_buf = ggml_backend_alloc_ctx_tensors(art_ctx, M.backend);
        if (!art_buf) {
            lm_fatal("vram", "artist-token parameter buffer allocation failed");
            ggml_free(art_ctx);
            adapter->free();
            dit_train_free(&M);
            return 1;
        }
        std::vector<float> z((size_t) ggml_nelements(t_art), 0.0f);
        ggml_backend_tensor_set(t_art, z.data(), 0, z.size() * sizeof(float));
        char ab[192];
        snprintf(ab, sizeof(ab), "artist token \"%s\": k=%d at encoder rows [%d,%d)%s", a.artist_token.c_str(), art_k,
                 enc_S_base, enc_S_base + art_k, a.artist_only ? ", adapter frozen" : ", trained with the adapter");
        lm_log("info", ab);
    }

    // Optimizer parameter set. Freezing the adapter means CLEARING
    // GGML_TENSOR_FLAG_PARAM, not just omitting it here: lm_optim_fill_gacc
    // asserts every PARAM-flagged graph node has an optimizer slot
    // (lm-optim.h:470), so flagged-but-unoptimized aborts the run.
    std::vector<ggml_tensor *> train_params;
    if (art_k > 0 && a.artist_only) {
        const std::vector<ggml_tensor *> & pv0 = adapter->params();
        for (size_t i = 0; i < pv0.size(); i++) {
            pv0[i]->flags &= ~(int32_t) GGML_TENSOR_FLAG_PARAM;
        }
    } else {
        train_params = adapter->params();
    }
    if (t_art) {
        train_params.push_back(t_art);
    }

    // Artist token export. Shares artist-token-io.h with the LM trainer so the
    // two sites cannot drift into two formats.
    auto art_export = [&](const std::string & dir, std::string * e) -> bool {
        if (!t_art) {
            return true;
        }
        ArtistTokenMeta m;
        m.name       = a.artist_token;
        m.site       = "as15_dit";
        m.base_model = a.dit_path;
        m.k          = art_k;
        return artist_token_write(t_art, m, dir, e);
    };

    LmOptim opt;
    // BEFORE init: the per-parameter rule split and the optimizer-state
    // allocation are both decided there (a Muon parameter gets no v buffer).
    opt.optimizer      = a.optimizer;
    opt.muon.lr_scale  = a.muon_lr_scale;
    opt.muon.momentum  = a.muon_momentum;
    opt.muon.ns_steps  = a.muon_ns_steps;
    opt.muon.nesterov  = a.muon_nesterov;
    opt.muon.min_dim   = a.muon_min_dim;
    opt.muon.bucket    = a.muon_bucket;
    // Consumed INSIDE lm_optim_init (o->prodigy_d = d0): set it before, not
    // with base_lr below.
    opt.prodigy_d0     = (float) a.prodigy_d0;
    {
        std::string err;
        if (!lm_optim_init(&opt, train_params, M.backend, &err)) {
            lm_fatal("vram", err);
            adapter->free();
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
    }
    opt.t_adamw      = t_adamw;
    opt.t_lossgrad   = t_lossgrad;
    opt.t_clip       = t_clip;
    opt.t_eps        = t_eps;
    opt.t_gnorm2     = t_gnorm2;
    opt.base_lr      = a.lr;
    if (a.optimizer == "prodigy") {
        // Under Prodigy lr is GAMMA, a schedule multiplier; d carries the
        // magnitude. A hand-tuned 5e-4 left here would scale every step by
        // 5e-4 and read as "Prodigy does not converge".
        if (a.lr != 1.0f) {
            char pb[192];
            snprintf(pb, sizeof(pb),
                     "prodigy: --lr %.3g is IGNORED. Prodigy sets its own step size; lr is only a schedule "
                     "multiplier and is forced to 1.0.", (double) a.lr);
            lm_log("warn", pb);
        }
        opt.base_lr = 1.0f;
    }
    opt.weight_decay = a.weight_decay;
    opt.grad_clip    = a.grad_clip;

    // ── checkpoint boundary buffers (C1) ─────────────────────────────────
    // One device allocation for the whole run, sized for the widest micro-batch
    // (S_max, B). Nothing is allocated when --ckpt 0 resolved to a single
    // segment, so the monolithic path costs exactly what it did before.
    DitCkptBufs ckbufs;
    if (SEG > 1) {
        std::string err;
        if (!dit_ckpt_alloc(&ckbufs, M.backend, H, S_max, B, SEG, &err)) {
            lm_fatal("vram", err);
            lm_optim_free(&opt);
            adapter->free();
            ggml_backend_buffer_free(buf_static);
            ggml_free(ctx_static);
            dit_train_free(&M);
            return 1;
        }
    }

    // C5: --grad-accum counts MICRO-BATCHES now, so an optimizer step consumes
    // B x grad_accum songs and stepsPerEpoch = ceil(n / (B*GA)). At B == 1 this
    // is the pre-batching formula unchanged.
    const int win_songs    = B * a.grad_accum;
    const int steps_per_ep = (n + win_songs - 1) / win_songs;
    const int total_steps  = std::max(1, steps_per_ep * a.epochs);
    // Percentage-only warmup is a hyperparameter-porting hazard. The Uber-LoKR-4
    // preset's lr (1e-2) is Side-Step's, and Side-Step warms up over a FIXED ~50
    // of its ~400 optimizer steps. Our defaults give a much smaller effective
    // batch and far fewer optimizer steps, so 5 % compressed to ~7 steps on a
    // 150-epoch run: the LR hit full scale before the adapter had settled and
    // three DiT LoKR runs blew up right at the end of the ramp (2026-07-29).
    // Floor the ramp at 50 steps, and cap it at HALF the run so warmup can never
    // dominate training. ratio == 0 still means "no warmup" (explicit opt-out),
    // and a 1-step run gets 0 because lm_lr_lambda(0, ...) == 0 by design — a
    // warmup covering every step would train nothing at all.
    int warmup_steps = 0;
    if (a.warmup_ratio > 0.0f) {
        warmup_steps = std::max(50, (int) ((double) total_steps * (double) a.warmup_ratio));
        warmup_steps = std::min(warmup_steps, total_steps / 2);
        if (warmup_steps < 1) {
            warmup_steps = 0;  // only reachable at total_steps == 1
        }
    }
    opt.total_steps  = total_steps;
    opt.warmup_steps = warmup_steps;

    // ── events (§2.2 order: model, vram, data, then steps) ───────────────
    // `deviceWideMb` is (total - free) for the WHOLE card, unlike every other
    // number here: it is the only figure that can see Windows quietly backing an
    // over-subscribed allocation with shared system memory. Sampled here, after
    // the mirror / adapter / optimizer are resident but before the probe.
    jl("{\"type\":\"vram\",\"freeMb\":%lld,\"totalMb\":%lld,\"freeSource\":\"%s\",\"reserveMb\":%d,\"mirrorMb\":%lld,"
       "\"mirror\":\"%s\",\"crop\":%d,\"layers\":%d,\"batch\":%d,\"ckptSegments\":%d,\"ckptSource\":\"%s\","
       "\"boundaryMb\":%lld,\"loraParams\":%lld,\"estMb\":%lld,"
       "\"deviceWideMb\":%lld,\"source\":\"%s\",\"cropSource\":\"%s\",\"layersSource\":\"%s\"}",
       (long long) fit.free_mb, (long long) fit.total_mb, fit.free_source, a.vram_reserve_mb,
       (long long) (M.mirror.bytes / 1048576), dit_mirror_mode_name(mirror_mode), crop_len, K, B, SEG,
       (a.ckpt == 1) ? "auto" : "user", (long long) (ckbufs.bytes / 1048576),
       (long long) adapter->nParams(), (long long) (fit.est_bytes / 1048576.0),
       (long long) dit_gpu_mem_query(M.backend).used_mb(), (fit.crop_user && fit.layers_user) ? "user" : "auto",
       fit.crop_user ? "user" : "auto", fit.layers_user ? "user" : "auto");
    if (K < L) {
        // §2.8 gates the UI banner on layersSource === 'auto', so this line must
        // say which it was; and the number that matters is the fitted BUDGET, not
        // the card's total VRAM (which the auto-fit never sees).
        const double budget_mb =
            ((double) fit.free_mb - (double) a.vram_reserve_mb) * (double) (1.0f - a.vram_safety);
        char b[288];
        snprintf(b, sizeof(b),
                 "%s trained depth is %d of %d layers (fitted against a %lld MB budget: %lld MB free - %d MB reserve, "
                 "%.0f%% safety) — adapter quality at partial depth is unvalidated",
                 fit.layers_user ? "Requested" : "Auto-fit reduced:", K, L, (long long) budget_mb,
                 (long long) fit.free_mb, a.vram_reserve_mb, (double) a.vram_safety * 100.0);
        lm_log("warn", b);
    }

    // D13: "If null_condition_emb is absent from the base, CFG dropout is
    // disabled with a `warn` and the run continues." Disabling it by short-circuit
    // alone left dit_train_log.json claiming cfg_ratio 0.15 on a base that never
    // applied it. Decide once, warn once, and use the same flag everywhere so a
    // `step` event can never report cfgDrop:1 without a replacement having happened.
    const bool cfg_ok = !M.null_cond.empty() && (int) M.null_cond.size() == enc_H;
    if (a.cfg_ratio > 0.0f && !cfg_ok) {
        char cb[256];
        snprintf(cb, sizeof(cb),
                 "CFG dropout disabled: this base has %s null_condition_emb, so the requested --cfg-ratio %.3g is not "
                 "applied (D13)",
                 M.null_cond.empty() ? "no" : "a wrongly-sized", (double) a.cfg_ratio);
        lm_log("warn", cb);
    }

    LmRng rng_order, rng_t, rng_noise, rng_crop, rng_cond;
    lm_rng_seed(&rng_order, (uint64_t) a.seed);
    lm_rng_seed(&rng_t, (uint64_t) a.seed ^ 0x517cc1b727220a95ull);
    lm_rng_seed(&rng_noise, (uint64_t) a.seed ^ 0x9e3779b97f4a7c15ull);
    lm_rng_seed(&rng_crop, (uint64_t) a.seed ^ 0xbf58476d1ce4e5b9ull);
    lm_rng_seed(&rng_cond, (uint64_t) a.seed ^ 0xd1b54a32d192ed03ull);

    // ── the micro-batch ──────────────────────────────────────────────────
    DitStepProf prf;
    // --profile-ops implies the step timer, since it fires off its step counter;
    // without an explicit cadence it just never reports a window.
    prf.on    = (a.profile_step > 0 || a.profile_ops);
    prf.every = (a.profile_step > 0) ? a.profile_step : 1000000;
    if (prf.on && SEG > 1) {
        // The segmented path builds 1+SEG graphs inside dit_ckpt_micro_batch and
        // does not expose the seams; profiling it would silently attribute the
        // whole thing to one bucket, which is worse than refusing.
        lm_log("warn", "--profile-step is only instrumented for the unsegmented graph; run it with --ckpt 0");
        prf.on = false;
    }
    DitOpProf            opprof;
    // Per-NODE profile with site attribution (dit-node-profile.h). Env-gated:
    // DIT_PROFILE_NODES=1 profiles micro-step 15, =N profiles micro-step N.
    // Nothing below runs, and no callback is installed, when the var is unset.
    DitNodeProfile nodeprof;
    dit_node_prof_init(&nodeprof, a.out_dir.c_str(), a.attn.c_str());
    if (nodeprof.enabled) {
        nodeprof.attn_prec = (attn_prec_req == GGML_PREC_F32) ? "f32" : "tf32-where-available";
        nodeprof.adapter   = ad->typeName();
        nodeprof.enc_S     = enc_S;
        nodeprof.D         = c.head_dim;
        nodeprof.Nh        = c.n_heads;
        nodeprof.Nkv       = c.n_kv_heads;
        nodeprof.layers    = L - lora_lo;
        nodeprof.crop      = crop_len;
        const std::vector<ggml_tensor *> & pv = ad->params();
        for (size_t i = 0; i < pv.size(); i++) {
            nodeprof.params.insert(pv[i]);
        }
        if (SEG > 1) {
            lm_log("warn", "DIT_PROFILE_NODES is only instrumented for the unsegmented graph; run it with --ckpt 0");
            nodeprof.enabled = false;
        }
    }
    std::vector<uint8_t> arena((size_t) 512 << 20);
    DitBatchHost         bh;
    DitBatchCfg          bcfg;
    bcfg.in_ch          = c.in_channels;
    bcfg.out_ch         = Oc;
    bcfg.enc_H          = enc_H;
    bcfg.enc_S          = enc_S;
    bcfg.patch          = P;
    bcfg.sliding_window = c.sliding_window;
    bcfg.crop           = crop_len;
    bcfg.anchor_song    = (a.crop_anchor != "zero");
    bcfg.structured     = (a.crop_mode != "random");
    bcfg.jitter         = a.crop_jitter;
    bcfg.jitter_min     = a.crop_min;
    // Median track length drives the coverage scaling — mean would let one
    // 6-minute outlier shrink the share for every other song in the set.
    int T_median = 0;
    {
        std::vector<int> Ts;
        Ts.reserve(samples.size());
        for (size_t i = 0; i < samples.size(); i++) {
            Ts.push_back(samples[i].T);
        }
        if (!Ts.empty()) {
            std::nth_element(Ts.begin(), Ts.begin() + (long) (Ts.size() / 2), Ts.end());
            T_median = Ts[Ts.size() / 2];
        }
    }
    bcfg.start_window   = a.crop_start_window;
    bcfg.start_frac     = dit_crop_endpoint_frac(a.crop_start_frac, crop_len, T_median, a.crop_endpoint_k);
    bcfg.end_frac       = dit_crop_endpoint_frac(a.crop_end_frac, crop_len, T_median, a.crop_endpoint_k);
    log->crop_start_frac_eff = bcfg.start_frac;
    log->crop_end_frac_eff   = bcfg.end_frac;
    if (bcfg.structured) {
        char cb[256];
        snprintf(cb, sizeof(cb),
                 "crop policy: structured - start %.1f%% over a %d-crop opening window, "
                 "end %.1f%% (half flush-jitter, half closing region), random %.1f%% "
                 "(requested %.0f/%.0f%%, scaled by crop coverage %d/%d = %.1f%%)",
                 (double) bcfg.start_frac * 100.0, bcfg.start_window, (double) bcfg.end_frac * 100.0,
                 (1.0 - (double) bcfg.start_frac - (double) bcfg.end_frac) * 100.0,
                 (double) a.crop_start_frac * 100.0, (double) a.crop_end_frac * 100.0, crop_len, T_median,
                 T_median > 0 ? 100.0 * (double) crop_len / (double) T_median : 0.0);
        lm_log("info", cb);
    }
    if (bcfg.jitter) {
        char cb[160];
        snprintf(cb, sizeof(cb), "crop jitter: each draw's length uniform over [%d, %d] frames", bcfg.jitter_min,
                 crop_len);
        lm_log("info", cb);
        fprintf(stderr, "[train-dit] %s\n", cb);
    }
    bcfg.weighted       = (a.loss_weighting == "flow_snr");
    bcfg.null_cond      = &M.null_cond;
    int graph_nodes = 0, sched_splits = 0, sched_copies = 0;
    int last_cfg_drop = 0, last_crop_start = 0;

    // One batched forward (+ backward). `els` carries the per-ELEMENT t / CFG /
    // genre decisions (design B3); the crop and noise draws happen inside
    // dit_batch_assemble, in the order the pre-batching trainer used.
    //
    // `lossgrad` is what t_lossgrad carries for THIS micro-batch (design B4: the
    // grad-accum scale, times the flow_snr weight at B == 1). It is uploaded here
    // rather than by the caller because the segment driver has to re-seed it per
    // segment and must know the head segment's value.
    auto micro_batch = [&](std::vector<DitBatchElem> & els, const std::vector<std::vector<float>> & temb_h,
                           const std::vector<std::vector<float>> & tproj_h, const std::vector<int> & tidx,
                           bool backward, float lossgrad, double * loss_out, double * raw_out) -> bool {
        const long long p_t0 = prf.t();
        const int nb = (int) els.size();
        dit_batch_assemble(bcfg, els, &rng_crop, &rng_noise, &bh);
        const int len = bh.len;
        const int S   = bh.S;
        last_crop_start = els[(size_t) (nb - 1)].crop_start;
        last_cfg_drop   = els[(size_t) (nb - 1)].cfg_drop ? 1 : 0;

        std::vector<float> temb_buf((size_t) H * (size_t) nb, 0.0f);
        std::vector<float> tproj_buf((size_t) 6 * (size_t) H * (size_t) nb, 0.0f);
        for (int b = 0; b < nb; b++) {
            memcpy(&temb_buf[(size_t) b * (size_t) H], temb_h[(size_t) tidx[(size_t) b]].data(),
                   (size_t) H * sizeof(float));
            memcpy(&tproj_buf[(size_t) b * 6 * (size_t) H], tproj_h[(size_t) tidx[(size_t) b]].data(),
                   (size_t) 6 * (size_t) H * sizeof(float));
        }

        // §3.0: EVERY input is re-uploaded, unconditionally, before EVERY graph
        // compute — including each of the segmented path's 1+N computes, which is
        // why this is a lambda the checkpoint driver also calls. GGML clobbers
        // input buffers; "only upload if changed" is the bug class the per-section
        // adapter masking work hit. That rule applies to the new B-wide buffers
        // exactly as it did to the old ones. (t_lossgrad is NOT here: the segment
        // driver re-seeds it per segment and must stay in charge of it.)
        auto upload_inputs = [&]() {
            ggml_backend_tensor_set(b_input, bh.input.data(), 0, bh.input.size() * sizeof(float));
            ggml_backend_tensor_set(b_vtgt, bh.vtgt.data(), 0, bh.vtgt.size() * sizeof(float));
            ggml_backend_tensor_set(b_enc, bh.enc.data(), 0, bh.enc.size() * sizeof(float));
            ggml_backend_tensor_set(b_pos, bh.pos.data(), 0, bh.pos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(b_sa, bh.sa.data(), 0, bh.sa.size() * sizeof(uint16_t));
            if (b_sa_pad && !bh.sa_pad.empty()) {
                ggml_backend_tensor_set(b_sa_pad, bh.sa_pad.data(), 0, bh.sa_pad.size() * sizeof(uint16_t));
            }
            ggml_backend_tensor_set(b_ca, bh.ca.data(), 0, bh.ca.size() * sizeof(uint16_t));
            ggml_backend_tensor_set(b_lw, bh.lw.data(), 0, bh.lw.size() * sizeof(float));
            ggml_backend_tensor_set(b_lwu, bh.lwu.data(), 0, bh.lwu.size() * sizeof(float));
            ggml_backend_tensor_set(b_temb, temb_buf.data(), 0, temb_buf.size() * sizeof(float));
            ggml_backend_tensor_set(b_tproj, tproj_buf.data(), 0, tproj_buf.size() * sizeof(float));
        };
        const long long p_t1 = prf.t();  // end of assemble + host-buffer prep
        upload_inputs();
        ggml_backend_tensor_set(t_lossgrad, &lossgrad, 0, sizeof(float));
        const long long p_t2 = prf.t();  // end of upload
        prf.add(DPB_ASSEMBLE, p_t0, p_t1);
        prf.add(DPB_UPLOAD, p_t1, p_t2);

        // Every view is CONTIGUOUS at offset 0 — soft_max_ext will not take a
        // strided mask, and ggml_reshape refuses a non-contiguous source. Built
        // per GRAPH, because the segmented path builds one per segment.
        auto mk_inputs = [&](ggml_context * gctx) {
            const size_t f32 = sizeof(float), f16 = sizeof(ggml_fp16_t);
            DitInputs    in;
            in.t_input = ggml_view_3d(gctx, b_input, c.in_channels, len, nb, (size_t) c.in_channels * f32,
                                      (size_t) c.in_channels * (size_t) len * f32, 0);
            in.t_enc   = ggml_view_3d(gctx, b_enc, enc_H, enc_S, nb, (size_t) enc_H * f32,
                                      (size_t) enc_H * (size_t) enc_S * f32, 0);
            in.t_pos   = ggml_view_1d(gctx, b_pos, (int64_t) S * nb, 0);
            in.t_temb  = ggml_view_3d(gctx, b_temb, H, 1, nb, (size_t) H * f32, (size_t) H * f32, 0);
            in.t_tproj = ggml_view_3d(gctx, b_tproj, 6 * H, 1, nb, (size_t) 6 * H * f32, (size_t) 6 * H * f32, 0);
            // [S,S] broadcast when nothing is padded; [S,S,1,B] per element when
            // something is (soft_max_ext: a.ne2 % 1 == 0, a.ne3 % B == 0).
            in.t_sa    = (bh.sa_B > 1) ? ggml_view_4d(gctx, b_sa, S, S, 1, nb, (size_t) S * f16,
                                                      (size_t) S * (size_t) S * f16, (size_t) S * (size_t) S * f16, 0)
                                       : ggml_view_2d(gctx, b_sa, S, S, (size_t) S * f16, 0);
            in.t_sa_pad = (bh.sa_B > 1 && b_sa_pad)
                              ? ggml_view_4d(gctx, b_sa_pad, S, S, 1, nb, (size_t) S * f16,
                                             (size_t) S * (size_t) S * f16, (size_t) S * (size_t) S * f16, 0)
                              : nullptr;
            in.t_ca    = ggml_view_4d(gctx, b_ca, enc_S, S, 1, nb, (size_t) enc_S * f16,
                                      (size_t) enc_S * (size_t) S * f16, (size_t) enc_S * (size_t) S * f16, 0);
            in.t_vtgt =
                ggml_view_3d(gctx, b_vtgt, Oc, len, nb, (size_t) Oc * f32, (size_t) Oc * (size_t) len * f32, 0);
            in.t_lw  = ggml_view_3d(gctx, b_lw, 1, len, nb, f32, (size_t) len * f32, 0);
            in.t_lwu = ggml_view_3d(gctx, b_lwu, 1, len, nb, f32, (size_t) len * f32, 0);
            in.t_cw  = t_cw;
            // The ONLY place the mode enters the graph build; mk_inputs is shared
            // by the monolithic path below and by the segment driver, so both get
            // it from one line.
            in.attn_mode     = attn_mode;
            in.attn_prec_req = attn_prec_req;
            // Same reasoning as attn_mode: one line, and both the monolithic
            // path and the segment driver pick it up.
            in.t_art = t_art;
            in.art_k = art_k;
            return in;
        };

        // ── the segmented path (C1) ──────────────────────────────────────
        // Only ever taken when the fit resolved more than one segment; --ckpt 0
        // keeps the monolithic graph below, byte for byte.
        if (SEG > 1 && backward) {
            if (!dit_ckpt_begin(&ckbufs, H, S, nb, SEG)) {
                return false;
            }
            DitCkptReq req;
            req.M           = &M;
            req.ad          = ad;
            req.opt         = &opt;
            req.sched       = M.sched;
            req.arena       = &arena;
            req.mk_inputs   = mk_inputs;
            req.upload      = upload_inputs;  // §3.0: re-uploaded before every segment compute
            req.T           = len;
            req.enc_S       = enc_S;
            req.B           = nb;
            req.Oc          = Oc;
            req.chan_bal    = chan_bal;
            req.gscale      = bh.gscale;
            req.want_report = (nb > 1 && bcfg.weighted);
            req.loss_scale  = lossgrad;
            const bool cok  = dit_ckpt_micro_batch(req, &ckbufs, ckplan, loss_out, raw_out);
            graph_nodes     = req.nodes;
            dit_ckpt_end(&ckbufs);
            return cok;
        }

        // Node profiler: armed BEFORE the build, because the site tags are laid
        // down by construction scopes inside the graph builder (dit-node-profile.h).
        const bool node_now = nodeprof.enabled && backward && prf.tot_steps >= (long long) nodeprof.at_step &&
                              prf.tot_steps < (long long) (nodeprof.at_step + nodeprof.n_steps);
        if (node_now) {
            nodeprof.reset();
            nodeprof.S       = S;
            nodeprof.B       = nb;
            nodeprof.tagging = true;
            g_dnp            = &nodeprof;
        }

        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, 65536, /*grads=*/backward);
        DitInputs        in  = mk_inputs(ctx);

        // The unweighted report node only exists at B > 1, where the flow_snr
        // weight has moved into t_lw and the loss node is no longer the raw mean.
        ggml_tensor * report = nullptr;
        ggml_tensor * vpred  = dit_train_forward(ctx, &M, ad, in, len, enc_S, nullptr, -1, -1, nb);
        ggml_tensor * loss   = dit_train_loss(ctx, vpred, in, Oc, len, chan_bal, bh.gscale,
                                              (nb > 1 && bcfg.weighted) ? &report : nullptr);
        if (backward) {
            ggml_set_loss(loss);
        }
        ggml_build_forward_expand(gf, loss);
        if (report) {
            ggml_build_forward_expand(gf, report);
        }
        const long long p_t3 = prf.t();  // end of forward build
        if (node_now) {
            // Everything in the graph right now is FORWARD; anything appended
            // below belongs to the backward pass. That split is what tells a
            // node's fwd cost from its bwd cost.
            dit_node_prof_mark_forward(&nodeprof, gf);
        }
        if (backward) {
            // AFTER both forward expansions: gacc is indexed by forward-node index.
            std::vector<ggml_tensor *> gacc;
            lm_optim_fill_gacc(&opt, gf, &gacc);
            ggml_build_backward_expand(ctx, gf, gacc.data());
        }
        const long long p_t4 = prf.t();  // end of backward expand
        graph_nodes = ggml_graph_n_nodes(gf);
        GGML_ASSERT(graph_nodes < 65536);

        if (node_now) {
            nodeprof.tagging     = false;  // the graph is built; no more tagging
            g_dnp                = nullptr;
            nodeprof.graph_nodes = graph_nodes;
        }

        // Arm the op profiler on one warmed-up micro-step, then disarm. Only one
        // eval callback can be installed, and the node profiler is the superset
        // (it carries the op roll-up too), so it wins when both are asked for.
        const bool op_now = (a.profile_ops && !node_now && prf.tot_steps == 15);
        if (op_now) {
            ggml_backend_sched_set_eval_callback(M.sched, dit_op_eval_cb, &opprof);
        }
        if (node_now) {
            ggml_backend_sched_set_eval_callback(M.sched, dnp_eval_cb, &nodeprof);
        }
        ggml_backend_sched_reset(M.sched);
        const long long p_t5 = prf.t();
        if (prf.on) {
            // Pull the split + gallocr pass out of the GPU bucket. compute_async
            // skips its own alloc once is_alloc is set, so this is the same work
            // in the same order (ggml-backend.cpp:1889-1895) — only the seam is
            // new. Off this path, the original single call stands untouched.
            if (!ggml_backend_sched_alloc_graph(M.sched, gf)) {
                ggml_free(ctx);
                return false;
            }
        }
        const long long p_t6 = prf.t();
        const bool ok = ggml_backend_sched_graph_compute(M.sched, gf) == GGML_STATUS_SUCCESS;
        const long long p_t7 = prf.t();
        if (op_now) {
            ggml_backend_sched_set_eval_callback(M.sched, nullptr, nullptr);
            dit_op_prof_print(opprof);
            opprof.agg.clear();
        }
        if (node_now) {
            ggml_backend_sched_set_eval_callback(M.sched, nullptr, nullptr);
            nodeprof.step_compute_us += p_t7 - p_t6;  // this step's whole compute bucket, serialised
            nodeprof.captured = true;
            nodeprof.done++;
            if (nodeprof.done >= nodeprof.n_steps) {
                dit_node_prof_print(nodeprof);
            }
        }
        if (ok && loss_out) {
            float lv = 0.0f;
            ggml_backend_tensor_get(loss, &lv, 0, sizeof(float));
            *loss_out = (double) lv;
            if (raw_out) {
                if (report) {
                    float rv = 0.0f;
                    ggml_backend_tensor_get(report, &rv, 0, sizeof(float));
                    *raw_out = (double) rv;
                } else {
                    *raw_out = (double) lv;
                }
            }
        }
        const long long p_t8 = prf.t();
        // Graph shape and, more to the point, the SPLIT COUNT: anything above 1
        // means the scheduler could not keep the whole graph on one backend and
        // is copying activations across the bus every micro-step. Three getters,
        // recorded whether or not profiling is on, so every run carries them.
        sched_splits = ggml_backend_sched_get_n_splits(M.sched);
        sched_copies = ggml_backend_sched_get_n_copies(M.sched);
        if (node_now) {
            nodeprof.sched_splits = sched_splits;
        }
        ggml_free(ctx);
        const long long p_t9 = prf.t();

        if (prf.on) {
            prf.add(DPB_BUILD, p_t2, p_t3);
            prf.add(DPB_BACKWARD, p_t3, p_t4);
            prf.add(DPB_ALLOC, p_t5, p_t6);
            prf.add(DPB_COMPUTE, p_t6, p_t7);
            prf.add(DPB_READBACK, p_t7, p_t8);
            prf.add(DPB_FREE, p_t8, p_t9);
            prf.nodes  = graph_nodes;
            prf.splits = sched_splits;
            prf.copies = sched_copies;
            prf.win_steps++;
            prf.tot_steps++;
            if (prf.win_steps >= (long long) prf.every) {
                dit_prof_report(prf);
            }
        }
        return ok;
    };

    // ── teardown helper (used by every failure path from here on) ────────
    auto teardown = [&]() {
        dit_ckpt_free(&ckbufs);
        lm_optim_free(&opt);
        adapter->free();
        if (art_buf) {
            ggml_backend_buffer_free(art_buf);
        }
        if (art_ctx) {
            ggml_free(art_ctx);
        }
        ggml_backend_buffer_free(buf_static);
        ggml_free(ctx_static);
        dit_train_free(&M);
    };

    // ── high-water probe (mandatory, §3.7) ───────────────────────────────
    //
    // One throwaway micro-BATCH at the chosen (crop, K, B) on the LONGEST sample,
    // loss discarded. This forces ggml_gallocr to its peak up front, so a
    // 40-songs-in OOM becomes an immediate honest failure, and it establishes
    // the leak counter's baseline. B copies of the longest song is the widest
    // graph the run can build, which is exactly what the probe is for.
    //
    // THE PROBE SHARES rng_crop / rng_noise WITH TRAINING, ON PURPOSE. The
    // pre-batching trainer probed through the same micro_step lambda, so its
    // probe burned exactly one rng_crop draw (and one crop's worth of rng_noise)
    // before step 1. At B == 1 this probe burns exactly the same, which is what
    // makes the §2.3.1 byte-identity anchor hold. Giving the probe a private RNG
    // would look tidier and would SHIFT every training crop by one draw — it
    // breaks the anchor rather than protecting it. Do not "harden" this.
    //
    // Diagnosing an anchor mismatch: --order also decides the per-epoch song
    // permutation, and a permutation change moves cropStart and loss while
    // leaving t (rng_t) and cfgDrop (rng_cond) untouched — the exact fingerprint
    // an rng_crop stream shift produces. Check `start.order` matches on BOTH
    // sides before suspecting the RNG.
    LmVramTracker tracker;
    {
        size_t longest = 0;
        for (size_t i = 1; i < samples.size(); i++) {
            if (samples[i].T > samples[longest].T) {
                longest = i;
            }
        }
        std::vector<float>              tv((size_t) B, 0.5f);
        std::vector<std::vector<float>> tb, tp;
        double                          lv  = 0.0;
        const int64_t                   tp0 = ggml_time_ms();
        if (!dit_train_temb(&M, tv, &tb, &tp)) {
            lm_fatal("vram", "the timestep-embedding precompute graph failed");
            teardown();
            return 1;
        }
        std::vector<DitBatchElem> els((size_t) B);
        std::vector<int>          tidx((size_t) B);
        for (int b = 0; b < B; b++) {
            els[(size_t) b].s = &samples[longest];
            els[(size_t) b].t = 0.5f;
            tidx[(size_t) b]  = b;
        }
        if (!micro_batch(els, tb, tp, tidx, true, 1.0f, &lv, nullptr)) {
            lm_fatal("vram", "the high-water probe failed — not enough VRAM for this crop/depth/batch");
            teardown();
            return 1;
        }
        lm_optim_zero_grad(&opt);
        size_t fixed = M.mirror.buf ? ggml_backend_buffer_get_size(M.mirror.buf) : M.mirror.bytes;
        fixed += ggml_backend_buffer_get_size(buf_static);
        if (lora.buf) {
            fixed += ggml_backend_buffer_get_size(lora.buf);
        }
        if (opt.buf_grad) {
            fixed += ggml_backend_buffer_get_size(opt.buf_grad);
        }
        if (opt.buf_mom) {
            fixed += ggml_backend_buffer_get_size(opt.buf_mom);
        }
        tracker.probe_baseline(M.backend, M.sched, fixed);
        // NVML, not cudaMemGetInfo: the whole point of the tripwire below is to
        // catch a device that is fuller than CUDA admits.
        const DitGpuMem gm_probe      = dit_gpu_mem_query(M.backend);
        const size_t    device_wide_mb = gm_probe.used_mb();
        const size_t    device_total_mb = gm_probe.total_mb();
        char            seg_note[128]  = "";
        if (SEG > 1) {
            snprintf(seg_note, sizeof(seg_note), ", widest of %d graphs over %d checkpoint segments (%lld MB of "
                                                 "boundary buffers)",
                     1 + SEG, SEG, (long long) (ckbufs.bytes / 1048576));
        }
        fprintf(stderr,
                "[train-dit] graph %d nodes (fwd+bwd)%s; high-water probe loss=%.6f in %lld ms; trainer-owned %zu MB "
                "(est %lld MB), device-wide %zu MB\n",
                graph_nodes, seg_note, lv, (long long) (ggml_time_ms() - tp0), tracker.base_mb,
                (long long) (fit.est_bytes / 1048576.0), device_wide_mb);
        // The arena on its own, measured against the arena term of whichever
        // model priced the run. `trainer-owned` above bundles the mirror, the
        // adapter, the optimizer state and the static inputs with it, and those
        // are computed exactly rather than fitted — so the bundled figure cannot
        // calibrate a polynomial. This line is what the flash-mode fit is read
        // off (docs/plans/2026-09-01-flash-attn-backward.md, MEASUREMENT 2).
        {
            DitVramModel vmp = vm;
            vmp.segments     = SEG;
            const int    Kr  = dit_vram_seg_layers(vmp, K);
            double       aest = flash_attn ? dit_vram_arena_bytes_flash(S_max, Kr, B, enc_S, c.n_heads, c.n_kv_heads,
                                                                        c.head_dim)
                                           : dit_vram_arena_bytes(S_max, Kr, B);
            // The LoKR kron intermediates live in the SAME scheduler buffer the
            // measured figure counts, so leaving them out of the estimate made a
            // LoKR run's line unreadable — that is how the 2026-09-02 refit's
            // defect report ended up quoting a 73 % over-prediction as a 45 %
            // under-prediction (dit-vram.h, flash LoKR apply arena). Same term
            // dit_vram_total_bytes charges, same arguments.
            if (vm.is_lokr) {
                const double Bf = (double) std::max(1, B);
                aest += flash_attn ? dit_vram_lokr_apply_bytes_flash(c, L - Kr, L, vm.lokr_dim, vm.lokr_factor,
                                                                    vm.target_mlp, (double) S_max * Bf,
                                                                    (double) std::max(1, enc_S) * Bf)
                                   : dit_lokr_apply_arena_bytes(c, L - Kr, L, vm.lokr_dim, vm.lokr_factor,
                                                                vm.target_mlp, S_max * std::max(1, B));
            }
            const size_t ameas = M.sched ? ggml_backend_sched_get_buffer_size(M.sched, M.backend) : 0;
            fprintf(stderr,
                    "[train-dit] arena %lld MB measured vs %lld MB est (%s model, S %d, K %d, Kr %d, seg %d, B %d, "
                    "enc_S %d, fixed %lld MB, boundary %lld MB)\n",
                    (long long) (ameas / 1048576), (long long) (aest / 1048576.0),
                    flash_attn ? "flash" : "exact", S_max, K, Kr, SEG, B, enc_S,
                    (long long) (fixed / 1048576), (long long) (ckbufs.bytes / 1048576));
        }
        // The probe succeeding does NOT mean it fitted: on Windows an
        // over-subscribed CUDA allocation is silently backed by shared system
        // memory instead of failing, and the run then trains at a crawl. Only the
        // device-wide figure can see it — the trainer-owned one counts the same
        // bytes whether they landed in VRAM or in host RAM. Warn, never fail: the
        // spill is a performance cliff, not an error, and the reserve/safety
        // margins already own the refusal path. (Written as `+ 512 >` so a card
        // reporting under 512 MB total cannot underflow the size_t.)
        if (device_wide_mb + 512 > device_total_mb) {
            char b[320];
            snprintf(b, sizeof(b),
                     "VRAM overcommitted after probe (%lld of %lld MB device-wide, %s) — Windows is likely spilling "
                     "into shared GPU memory; reduce crop, layers, or lokr dim, or close other GPU apps",
                     (long long) device_wide_mb, (long long) device_total_mb, gm_probe.source());
            lm_log("warn", b);
        }
    }

    jl("{\"type\":\"data\",\"samples\":%d,\"skipped\":0,\"minFrames\":%d,\"maxFrames\":%d,\"encSMax\":%d,"
       "\"channelStats\":%s,\"genreSamples\":%d,\"batch\":%d,\"stepsPerEpoch\":%d,\"totalSteps\":%d,"
       "\"warmupSteps\":%d,\"graphNodes\":%d}",
       n, min_T, max_T, enc_S, have_cstats ? "true" : "false", genre_samples, B, steps_per_ep, total_steps,
       warmup_steps, graph_nodes);

    dit_milestone_reset(a.out_dir);

    // ── log config ───────────────────────────────────────────────────────
    log->layers         = K;
    log->n_layers_total = L;
    log->layers_source  = fit.layers_user ? "user" : "auto";
    log->crop           = crop_len;
    log->crop_source    = fit.crop_user ? "user" : "auto";
    // Which footprint model priced the crop/depth/segments above, and the enc_S
    // it was priced at — both required to re-interpret this run's VRAM figures
    // (fattn-train-spec.md §11.3). crop_max is the EFFECTIVE cap, so a flash run
    // whose default cap was lifted records the number the walk actually used.
    log->arena_model    = flash_attn ? "flash" : "exact";
    log->enc_S          = enc_S;
    log->crop_max       = crop_max_eff;
    log->batch          = B;
    log->ckpt           = SEG;
    log->samples        = n;
    // Runtime facts, known once the high-water probe has built and run one graph.
    // The backend in particular: it used to reach only a stderr tail the server
    // discards, which left "was this run even on CUDA?" unanswerable after the
    // fact for a whole day of runs.
    log->backend      = ggml_backend_name(M.backend);
    log->graph_nodes  = graph_nodes;
    log->sched_splits = sched_splits;
    log->sched_copies = sched_copies;
    log->partial_depth  = K < L;
    log->channel_balance = chan_bal;
    log->vram_free_mb   = fit.free_mb;
    log->vram_total_mb  = fit.total_mb;
    log->vram_mirror_mb = M.mirror.bytes / 1048576;
    log->vram_est_mb    = (size_t) (fit.est_bytes / 1048576.0);

    // ── epoch loop (§3.5.3) ──────────────────────────────────────────────
    {
        DitExportMeta xmeta;
        xmeta.base_model_path = a.dit_path;
        xmeta.producer        = std::string("ace-train ") + ACE_VERSION;
        xmeta.trigger          = log->trigger;
        xmeta.trigger_position = log->trigger_position;

        std::vector<double> ep_hist;
        double              ladder      = 0.0;
        bool                ladder_seed = false;
        long long           ep_ms_sum   = 0;
        int                 global_step = 0;
        int                 rc          = 0;

        for (int epoch = 0; epoch < a.epochs && rc == 0; epoch++) {
            const int64_t    t_ep0 = ggml_time_ms();
            std::vector<int> order((size_t) n);
            for (int i = 0; i < n; i++) {
                order[(size_t) i] = i;
            }
            if (a.order != "fixed") {
                LmRng er;
                lm_rng_seed(&er, (uint64_t) a.seed + (uint64_t) epoch * 7919ull);
                lm_rng_shuffle(&er, order);
            }

            double      running = 0.0, running_raw = 0.0;
            int         n_micro = 0;
            LmStepStats last_stats;
            lm_optim_zero_grad(&opt);

            // A WINDOW is one optimizer step: grad_accum micro-batches of B songs.
            // DoRA: refresh the per-site weight norms ONCE per epoch here and once
            // per window below — A/B only change at optimizer steps, so the norm
            // is exact for every micro-batch in between (PEFT computes it from a
            // detached lora_weight per forward; same quantity, fewer passes).
            for (int w0 = 0; w0 < n && rc == 0; w0 += win_songs) {
                {
                    std::string perr;
                    if (!adapter->preWindow(M.backend, M.sched, &perr)) {
                        lm_fatal("train", "adapter preWindow failed: " + perr);
                        rc = 1;
                        break;
                    }
                }
                const int wlen = std::min(win_songs, n - w0);      // songs in this window
                const int n_mb = (wlen + B - 1) / B;               // micro-batches (the last may be short)

                // Sample the WHOLE window's timesteps up front: the flow_snr mean
                // needs them, and so does the temb precompute (§3.5's batch-of-1 trap).
                // One t PER ELEMENT (design B3) — at B == 1 that is one per
                // micro-step, i.e. exactly the pre-batching draw sequence.
                std::vector<float> ts((size_t) wlen), wgt((size_t) wlen);
                double             wsum = 0.0;
                for (int i = 0; i < wlen; i++) {
                    ts[(size_t) i]  = dit_sample_t(&rng_t, a.timestep_mu, a.timestep_sigma, a.t_min, a.t_max);
                    wgt[(size_t) i] = (a.loss_weighting == "flow_snr")
                                          ? dit_flow_snr_w(ts[(size_t) i], a.t_bias, a.snr_gamma)
                                          : 1.0f;
                    wsum += (double) wgt[(size_t) i];
                }
                const double wbar = wsum / std::max(1, wlen);

                std::vector<std::vector<float>> temb_h, tproj_h;
                if (!dit_train_temb(&M, ts, &temb_h, &tproj_h)) {
                    lm_fatal("vram", "the timestep-embedding precompute graph failed mid-epoch");
                    rc = 1;
                    break;
                }

                double        win_loss = 0.0, win_raw = 0.0;
                const int64_t t_win0 = ggml_time_ms();
                for (int mb = 0; mb < n_mb; mb++) {
                    const int i0 = mb * B;
                    const int nb = std::min(B, wlen - i0);

                    // Per-ELEMENT genre / CFG draws, in element order — the same
                    // rng_cond sequence the pre-batching trainer consumed, including
                    // its short-circuits (a draw only happens when the preceding
                    // conditions hold).
                    std::vector<DitBatchElem> els((size_t) nb);
                    std::vector<int>          tidx((size_t) nb);
                    for (int b = 0; b < nb; b++) {
                        const DitSample & s = samples[(size_t) order[(size_t) (w0 + i0 + b)]];
                        DitBatchElem &    e = els[(size_t) b];
                        e.s         = &s;
                        e.t         = ts[(size_t) (i0 + b)];
                        e.w         = (float) ((double) wgt[(size_t) (i0 + b)] / (wbar > 0.0 ? wbar : 1.0));
                        e.use_genre = a.genre_ratio > 0 && !s.enc_genre.empty() &&
                                      (lm_rng_uniform(&rng_cond) * 100.0f < (float) a.genre_ratio);
                        e.cfg_drop = a.cfg_ratio > 0.0f && cfg_ok && (lm_rng_uniform(&rng_cond) < a.cfg_ratio);
                        tidx[(size_t) b] = i0 + b;
                    }

                    // B4: at B > 1 the per-element weight has moved into t_lw and
                    // the scalar is the pure grad-accum scale. At B == 1 it keeps
                    // carrying wnorm, which is what makes §2.3.1 hold.
                    //
                    // THE SCALE IS AN ELEMENT SHARE, NOT 1/n_mb. Write the window
                    // out. The in-graph t_lw already normalises WITHIN a
                    // micro-batch: for flow_snr each element carries
                    // w_b/(Oc*len_b*nb), so loss_mb = (1/nb) * sum_b w_b*m_b where
                    // m_b is element b's own mean. The accumulated gradient is
                    // sum_mb lg_mb * loss_mb. With lg_mb = nb/wlen that telescopes
                    // to (1/wlen) * sum_over_every_element w_b*m_b — Side-Step's
                    // global mean-of-per-sample-means over the whole optimizer
                    // window. With the old lg_mb = 1/n_mb it was a mean of
                    // MICRO-BATCH means instead, which is the same thing only while
                    // every micro-batch is full: a tail of nb < B elements got
                    // 1/n_mb of the window instead of nb/wlen, over-weighting each
                    // of its elements by (B/nb).
                    //
                    // At B == 1, n_mb == wlen so nb/wlen == 1/n_mb exactly and the
                    // §2.3.1 byte-identity anchor is untouched, short final window
                    // included.
                    //
                    // For --loss-weighting none the in-graph weight is 1/(sum of
                    // Oc*len_b over the MICRO-batch), so the combined reduction is
                    // the exact global element mean when the crops are equal-length
                    // (they are, except for songs shorter than the crop) and an
                    // element-weighted approximation of it otherwise. Making that
                    // exact would need the window's total frame count before any
                    // crop has been drawn, which the streaming assembler cannot
                    // give.
                    const float share = (float) nb / (float) wlen;
                    const float lg    = (nb == 1) ? (els[0].w * share) : share;

                    double lv = 0.0, raw = 0.0;
                    if (!micro_batch(els, temb_h, tproj_h, tidx, true, lg, &lv, &raw)) {
                        lm_fatal("vram", "graph compute failed mid-epoch");
                        rc = 1;
                        break;
                    }
                    // `lv` already carries the per-element weights at B > 1 and is
                    // the raw mean at B == 1, so the host multiply only applies to
                    // the single-element case.
                    const double wl = (nb == 1) ? ((double) els[0].w * lv) : (lv * (double) nb);
                    const double wr = (nb == 1) ? lv : (raw * (double) nb);
                    win_loss += wl;
                    win_raw += wr;
                    running += wl;
                    running_raw += wr;
                    n_micro += nb;
                }
                if (rc != 0) {
                    break;
                }
                // The optimizer step sits OUTSIDE micro_batch, so it is in none
                // of the DPB buckets — and with Muon it is the thing under
                // suspicion. Timed separately, reported with the step profile.
                const long long p_o0 = prf.on ? ggml_time_us() : 0;
                if (!lm_optim_step(&opt, M.sched, &last_stats)) {
                    lm_fatal("vram", "optimizer step failed");
                    rc = 1;
                    break;
                }
                if (prf.on) {
                    prf.optim_us += ggml_time_us() - p_o0;
                    prf.optim_steps++;
                }
                global_step++;
                const size_t vram_mb = tracker.sample();
                jl("{\"type\":\"step\",\"epoch\":%d,\"step\":%d,\"totalSteps\":%d,\"micro\":%d,\"loss\":%.6f,"
                   "\"rawLoss\":%.6f,\"lr\":%.9g,\"gradNorm\":%.6f,\"clipScale\":%.6f,\"t\":%.4f,\"crop\":%d,"
                   "\"cropStart\":%d,\"cfgDrop\":%d,\"ms\":%lld,\"vramMb\":%lld}",
                   epoch + 1, global_step, total_steps, n_mb, win_loss / (double) wlen, win_raw / (double) wlen,
                   (double) last_stats.lr, (double) last_stats.grad_norm, (double) last_stats.clip,
                   (double) ts[(size_t) (wlen - 1)], crop_len, last_crop_start, last_cfg_drop,
                   (long long) (ggml_time_ms() - t_win0), (long long) vram_mb);
            }
            if (rc != 0) {
                break;
            }

            const double    avg     = running / std::max(1, n_micro);
            const double    avg_raw = running_raw / std::max(1, n_micro);
            const long long ems     = (long long) (ggml_time_ms() - t_ep0);
            ep_ms_sum += ems;
            ep_hist.push_back(avg);
            double ma5 = 0.0;
            {
                const int kk = (int) std::min<size_t>(5, ep_hist.size());
                for (int i = 0; i < kk; i++) {
                    ma5 += ep_hist[ep_hist.size() - 1 - (size_t) i];
                }
                ma5 /= (double) kk;
            }

            const bool best = (out->best_loss < 0.0) || (avg < out->best_loss);
            if (best) {
                out->best_loss  = avg;
                out->best_epoch = epoch + 1;
            }
            out->final_loss = avg;
            out->epochs_run = epoch + 1;

            DitEpochRec rec;
            rec.epoch     = epoch + 1;
            rec.loss      = avg;
            rec.ma5       = ma5;
            rec.raw_loss  = avg_raw;
            rec.lr        = last_stats.lr;
            rec.grad_norm = last_stats.grad_norm;
            rec.ms        = ems;
            log->epochs_log.push_back(rec);
            // HOT-Step patch: flash-attn-train — the RESOLVED arithmetic, asked
            // of the backend that just ran an epoch's worth of graphs rather
            // than restated from the flag. Refreshed each epoch so the field is
            // populated even if the run is cancelled after the first one.
            if (attn_mode == DIT_ATTN_FLASH) {
                log->attn_prec = dit_flash_prec_label(M.backend);
            }
            log->epochs_run = out->epochs_run;
            log->final_loss = out->final_loss;
            // Latest Prodigy step-size estimate, exported so a resumed run can
            // seed its --prodigy-d0 from it (dit-resume.h). 0 for other optimizers.
            log->prodigy_d  = (a.optimizer == "prodigy") ? opt.prodigy_d : 0.0;
            log->best_loss  = out->best_loss;
            log->best_epoch = out->best_epoch;

            // Export the BEST epoch, not the last one (2026-07-30).
            //
            // This used to export unconditionally every epoch, which met D16
            // ("a cancel or crash always leaves a usable adapter") but meant a
            // run that missed its target shipped whatever the FINAL epoch
            // happened to produce — frequently worse than something it passed
            // through 50 epochs earlier. Exporting only on improvement keeps
            // D16 fully (there is still an adapter on disk from epoch 1 onward,
            // and now it is the best one) and does strictly LESS I/O.
            //
            // Selection is on MA5, not the single-epoch mean, for the same
            // reason the target test is: one epoch over 11 songs is a handful
            // of random crop/timestep draws, so the single-epoch minimum tends
            // to be the luckiest draw rather than the best adapter. MA5 cuts
            // that variance ~2.2x. best_loss/best_epoch below stay single-epoch
            // — they are what the UI reports, and changing them would silently
            // redefine a number people have been reading.
            //
            // hit_target forces an export even if it is not an improvement:
            // when the run stops on target, the adapter must be the epoch that
            // tripped it.
            const bool hit_target = (a.target_loss > 0.0f && ma5 <= (double) a.target_loss);
            const bool best_ma5   = (out->saved_ma5 < 0.0) || (ma5 < out->saved_ma5);
            if (best_ma5 || hit_target) {
                DitExportResult xr;
                std::string     xerr;
                if (!dit_export_peft(*ad, xmeta, a.out_dir.c_str(), &xr, &xerr)) {
                    lm_fatal("export", xerr);
                    rc = 1;
                    break;
                }
                if (!art_export(a.out_dir, &xerr)) {
                    lm_fatal("export", xerr);
                    rc = 1;
                    break;
                }
                out->exported       = true;
                out->export_tensors = xr.tensors;
                out->saved_ma5      = best_ma5 ? ma5 : out->saved_ma5;
                out->saved_epoch    = epoch + 1;
                out->saved_reason   = hit_target ? "target" : "best";
                log->saved_ma5      = out->saved_ma5;
                log->saved_epoch    = out->saved_epoch;
                log->saved_reason   = out->saved_reason;
            }

            if (a.milestone_step > 0.0f) {
                if (!ladder_seed) {
                    ladder      = floor(ma5 / (double) a.milestone_step) * (double) a.milestone_step;
                    ladder_seed = true;
                }
                while (ma5 <= ladder + 1e-12) {
                    // %.1f collides for any --milestone-step below 0.1 (1.15 and
                    // 1.10 both format to "1.1"), which makes two ladder rungs
                    // share one directory and lets dit_milestone_prune delete a
                    // path a surviving log entry still references. Widen the label
                    // just enough for the configured step.
                    const std::string label = dit_milestone_label(ladder, a.milestone_step);
                    const double      lval  = atof(label.c_str());
                    const std::string rel   = "milestones/loss_" + label;
                    const std::string mdir  = lm_join(a.out_dir, rel);
                    DitExportResult   mr;
                    std::string       merr;
                    if (dit_export_peft(*ad, xmeta, mdir.c_str(), &mr, &merr)) {
                        // A milestone without its token would be an adapter that
                        // silently expects conditioning rows it does not ship.
                        if (!art_export(mdir, &merr)) {
                            lm_log("warn", "milestone artist token not written: " + merr);
                        }
                        DitMilestoneRec ms;
                        ms.loss  = lval;
                        ms.epoch = epoch + 1;
                        ms.path  = rel;
                        log->milestones.push_back(ms);
                        jl("{\"type\":\"milestone\",\"loss\":%.4g,\"epoch\":%d,\"path\":\"%s\",\"bytes\":%lld}", lval,
                           epoch + 1, lm_json_escape(mdir).c_str(), mr.bytes);
                        dit_milestone_prune(a.out_dir, &log->milestones, a.milestone_keep);
                    } else {
                        lm_log("warn", "milestone snapshot failed: " + merr);
                    }
                    ladder -= (double) a.milestone_step;
                }
            }

            if (prf.tot_steps > 0) {
                // Run-mean per bucket, refreshed every epoch so a killed run still
                // leaves its profile behind.
                log->profile_steps = prf.tot_steps;
                log->profile_ms.clear();
                for (int i = 0; i < DPB_N; i++) {
                    log->profile_ms.push_back(
                        { DIT_PROF_NAMES[i], (double) prf.tot[i] / (double) prf.tot_steps / 1000.0 });
                }
            }
            log->vram_peak_mb = tracker.peak_mb;
            log->total_ms     = (long long) (ggml_time_ms() - t_stage0);
            if (!dit_write_train_log(a.out_dir, *log)) {
                lm_log("warn", "cannot write dit_train_log.json in " + a.out_dir);
            }

            const long long eta =
                (long long) ((double) ep_ms_sum / (double) (epoch + 1) * (double) (a.epochs - epoch - 1));
            jl("{\"type\":\"epoch\",\"epoch\":%d,\"epochs\":%d,\"loss\":%.6f,\"ma5\":%.6f,\"rawLoss\":%.6f,"
               "\"lr\":%.9g,\"gradNorm\":%.6f,\"ms\":%lld,\"etaMs\":%lld,\"best\":%s}",
               epoch + 1, a.epochs, avg, ma5, avg_raw, (double) last_stats.lr, (double) last_stats.grad_norm, ems, eta,
               best ? "true" : "false");
            jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"train\"}", epoch + 1, a.epochs);
            fprintf(stderr, "[train-dit] epoch %d/%d loss=%.6f ma5=%.6f raw=%.6f lr=%.3e gnorm=%.3f (%lld ms)\n",
                    epoch + 1, a.epochs, avg, ma5, avg_raw, (double) last_stats.lr, (double) last_stats.grad_norm, ems);

            if (a.target_loss > 0.0f && ma5 <= (double) a.target_loss) {
                out->stopped_on_target  = true;
                log->target_stop        = true;
                log->target_stop_epoch  = epoch + 1;
                log->target_stop_loss   = avg;
                log->target_stop_ma5    = ma5;
                jl("{\"type\":\"target_stop\",\"epoch\":%d,\"loss\":%.6f,\"ma5\":%.6f,\"targetLoss\":%.6g}", epoch + 1,
                   avg, ma5, (double) a.target_loss);
                break;
            }
        }

        dit_prof_report(prf);  // flush a partial window, if any
        if (nodeprof.captured) {
            // The reference the serialised numbers are read against: mean compute
            // bucket over every micro-step EXCEPT the profiled one, whose own
            // cost is subtracted rather than averaged away.
            const long long other_steps = prf.tot_steps - nodeprof.done;
            nodeprof.real_steps         = other_steps;
            nodeprof.real_compute_ms =
                other_steps > 0 ? (double) (prf.tot[DPB_COMPUTE] - nodeprof.step_compute_us) / other_steps / 1000.0
                                : 0.0;
            dit_node_prof_write(nodeprof);
        }
        log->vram_peak_mb = tracker.peak_mb;
        out->samples      = n;
        out->ms           = (long long) (ggml_time_ms() - t_stage0);

        char b[224];
        snprintf(b, sizeof(b), "leak counter: baseline %zu MB, peak %zu MB, max delta %lld MB over %d optimizer steps",
                 tracker.base_mb, tracker.peak_mb, tracker.max_delta, global_step);
        fprintf(stderr, "[train-dit] %s\n", b);
        jl("{\"type\":\"leak\",\"baselineMb\":%lld,\"peakMb\":%lld,\"deltaMb\":%lld,\"steps\":%d}",
           (long long) tracker.base_mb, (long long) tracker.peak_mb, tracker.max_delta, global_step);

        if (rc == 0) {
            // savedEpoch/savedMa5/savedReason say which epoch's adapter is in
            // the run dir. finalLoss is the LAST epoch and is usually NOT it.
            fprintf(stderr, "[train-dit] adapter saved from epoch %d (ma5 %.4f, %s) of %d run\n",
                    out->saved_epoch, out->saved_ma5, out->saved_reason.c_str(), out->epochs_run);
            jl("{\"type\":\"stage\",\"stage\":\"train\",\"state\":\"end\",\"epochsRun\":%d,\"finalLoss\":%.6f,"
               "\"bestLoss\":%.6f,\"savedEpoch\":%d,\"savedMa5\":%.6f,\"savedReason\":\"%s\","
               "\"stoppedOnTarget\":%s,\"ms\":%lld}",
               out->epochs_run, out->final_loss, out->best_loss, out->saved_epoch, out->saved_ma5,
               out->saved_reason.c_str(), out->stopped_on_target ? "true" : "false", out->ms);
        }

        teardown();
        return rc;
    }
}

// ─── main entry ─────────────────────────────────────────────────────────────

static int dit_train_main(const DitTrainArgs & a) {
    const bool   is_lokr      = (a.adapter_type == "lokr");
    const char * weights_leaf = is_lokr ? "lokr_weights.safetensors" : "adapter_model.safetensors";
    if (a.self_test) {
        return dit_self_test(a.dit_path, a.tensors_dir, (uint64_t) a.seed, std::min(a.crop_max, 750),
                             a.vram_reserve_mb, a.vram_safety);
    }

    const int64_t t_run0 = ggml_time_ms();

    std::string stage_csv;
    for (size_t i = 0; i < a.stages.size(); i++) {
        if (i) {
            stage_csv += ",";
        }
        stage_csv += "\"" + a.stages[i] + "\"";
    }

    // LoKR carries its own geometry fields; `rank`/`alpha` stay at 0 so a relay
    // cannot mistake the LoRA defaults for the configuration that ran.
    char adapter_fields[160];
    if (is_lokr) {
        snprintf(adapter_fields, sizeof(adapter_fields),
                 "\"rank\":0,\"alpha\":0,\"lokrDim\":%d,\"lokrAlpha\":%.9g,\"lokrFactor\":%d,\"lokrDecomposeBoth\":%s",
                 a.lokr_dim, (double) a.lokr_alpha, a.lokr_factor, a.lokr_decompose_both ? "true" : "false");
    } else {
        snprintf(adapter_fields, sizeof(adapter_fields), "\"rank\":%d,\"alpha\":%d", a.rank, a.alpha);
    }
    // `start` is emitted BEFORE the model is loaded, so it cannot know the
    // RESOLVED batch or segment count — those are what the VRAM fit, the dataset's
    // song count and the A1 repeat_back cap leave of the request, and they are
    // decided inside dit_train_stage. The two fields here are therefore named for
    // what they are, REQUESTED, and the resolved pair is on the `vram` event
    // (`batch` / `ckptSegments`, plus `ckptSource`), which is emitted after the
    // fit. Design §2.1.
    jl("{\"type\":\"start\",\"stages\":[%s],\"tensors\":\"%s\",\"out\":\"%s\",\"dit\":\"%s\",\"adapterType\":\"%s\","
       "%s,\"targetMlp\":%s,\"lr\":%.9g,\"epochs\":%d,\"gradAccum\":%d,\"batchRequested\":%d,\"ckptRequested\":%d,"
       "\"targetLoss\":%.9g,"
       "\"gradClip\":%.9g,\"seed\":%d,\"order\":\"%s\",\"lossWeighting\":\"%s\",\"channelBalance\":%s,"
       "\"cfgRatio\":%.9g,"
       "\"genreRatio\":%d,\"mirror\":\"%s\",\"bwd\":\"%s\"}",
       stage_csv.c_str(), lm_json_escape(a.tensors_dir).c_str(), lm_json_escape(a.out_dir).c_str(),
       lm_json_escape(a.dit_name.empty() ? a.dit_path : a.dit_name).c_str(), a.adapter_type.c_str(), adapter_fields,
       a.target_mlp ? "true" : "false", (double) a.lr, a.epochs, a.grad_accum, a.batch, a.ckpt,
       (double) a.target_loss,
       (double) a.grad_clip, a.seed, a.order.c_str(), a.loss_weighting.c_str(),
       a.channel_balance ? "true" : "false", (double) a.cfg_ratio, a.genre_ratio, a.mirror.c_str(), a.bwd.c_str());

    if (a.adapter_type != "lora" && !is_lokr) {
        lm_fatal("unsupported-adapter", "--adapter-type must be lora|lokr");
        return 1;
    }

    DitTrainLog     log;
    DitTrainOutcome out;
    log.producer        = std::string("ace-train ") + ACE_VERSION;
    log.created_at      = pm_iso8601_utc_now();
    log.adapter_type    = a.adapter_type;
    log.rank            = a.rank;
    log.alpha           = a.alpha;
    log.is_lokr             = is_lokr;
    log.lokr_dim            = a.lokr_dim;
    log.lokr_alpha          = a.lokr_alpha;
    log.lokr_factor         = a.lokr_factor;
    log.lokr_decompose_both = a.lokr_decompose_both;
    log.target_mlp      = a.target_mlp;
    log.dit_path        = a.dit_path;
    log.dit_name        = a.dit_name;
    log.tensors         = a.tensors_dir;
    // Trigger word (T5): CLI flags win, else the variant's preprocess_meta.json.
    log.trigger          = a.trigger;
    log.trigger_position = a.trigger_position;
    dit_resolve_trigger(a.tensors_dir, &log.trigger, &log.trigger_position);
    if (!log.trigger.empty()) {
        fprintf(stderr, "[train-dit] trigger \"%s\" (%s) will be embedded in the adapter\n", log.trigger.c_str(),
                log.trigger_position.c_str());
    }
    log.crop_min        = a.crop_min;
    log.crop_max        = a.crop_max;
    log.crop_anchor     = a.crop_anchor;
    log.crop_mode       = a.crop_mode;
    log.crop_start_frac = a.crop_start_frac;
    log.crop_end_frac   = a.crop_end_frac;
    log.crop_start_window    = a.crop_start_window;
    log.crop_endpoint_k      = a.crop_endpoint_k;
    log.crop_jitter          = a.crop_jitter;
    log.mirror          = a.mirror;
    log.bwd             = a.bwd;
    log.attn_mode       = a.attn;
    log.optimizer       = a.optimizer;
    // log.prodigy_d is filled by dit_train_stage, where the optimizer lives.
    log.init_adapter    = a.init_adapter;
    log.init_from_ma5   = a.init_from_ma5;
    log.lr              = a.lr;
    log.epochs          = a.epochs;
    log.grad_accum      = a.grad_accum;
    log.batch           = a.batch;  // dit_train_stage overwrites with what actually ran
    log.ckpt            = a.ckpt;   // ditto: the RESOLVED segment count replaces it
    log.grad_clip       = a.grad_clip;
    log.weight_decay    = a.weight_decay;
    log.warmup_ratio    = a.warmup_ratio;
    log.seed            = a.seed;
    log.order           = a.order;
    log.loss_weighting  = a.loss_weighting;
    log.snr_gamma       = a.snr_gamma;
    log.t_bias          = a.t_bias;
    log.channel_balance = a.channel_balance;
    log.timestep_mu     = a.timestep_mu;
    log.timestep_sigma  = a.timestep_sigma;
    log.t_min           = a.t_min;
    log.t_max           = a.t_max;
    log.cfg_ratio       = a.cfg_ratio;
    log.genre_ratio     = a.genre_ratio;
    log.target_loss     = a.target_loss;

    if (!pm_mkdir_p(a.out_dir)) {
        lm_fatal("export", "cannot create the adapter output directory " + a.out_dir);
        return 1;
    }
    if (a.overwrite) {
        // BOTH layouts, not just this run's: a leftover lokr_weights.safetensors
        // next to a fresh LoRA export (or vice versa) wins some loader probes and
        // loses others, so the directory would serve the stale adapter.
        remove(lm_join(a.out_dir, "adapter_model.safetensors").c_str());
        remove(lm_join(a.out_dir, "adapter_config.json").c_str());
        remove(lm_join(a.out_dir, "lokr_weights.safetensors").c_str());
    }

    if (dit_has_stage(a, "train")) {
        const int rc = dit_train_stage(a, &log, &out);
        if (rc != 0) {
            return rc;
        }
    }

    if (dit_has_stage(a, "export")) {
        jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"begin\"}");
        const int64_t t_x0 = ggml_time_ms();
        if (!out.exported) {
            // §2.2: a `done` line means the run completed. Exiting 0 having written
            // nothing left the server's post-check as the only backstop and gave a
            // CLI user no signal at all. The adapter only exists in memory during
            // dit_train_stage, so `export` without `train` can never produce one.
            lm_fatal("export", dit_has_stage(a, "train") ?
                                   "the train stage produced no epoch, so there is no adapter to export" :
                                   "--stages export needs the train stage: the adapter only exists in memory during "
                                   "training, so there is nothing to write");
            return 1;
        } else {
            log.total_ms = (long long) (ggml_time_ms() - t_run0);
            if (!dit_write_train_log(a.out_dir, log)) {
                lm_fatal("export", "cannot write dit_train_log.json in " + a.out_dir);
                return 1;
            }
            long long bytes = 0;
            pm_stat_file(lm_join(a.out_dir, weights_leaf), &bytes, NULL);
            jl("{\"type\":\"export\",\"path\":\"%s\",\"tensors\":%d,\"bytes\":%lld,\"ms\":%lld}",
               lm_json_escape(a.out_dir).c_str(), out.export_tensors, bytes, (long long) (ggml_time_ms() - t_x0));
            jl("{\"type\":\"stage\",\"stage\":\"export\",\"state\":\"end\",\"ms\":%lld}",
               (long long) (ggml_time_ms() - t_x0));
        }
    }

    const long long run_ms = (long long) (ggml_time_ms() - t_run0);
    jl("{\"type\":\"done\",\"stages\":[%s],\"epochsRun\":%d,\"finalLoss\":%.6f,\"stoppedOnTarget\":%s,"
       "\"adapter\":\"%s\",\"samples\":%d,\"crop\":%d,\"layers\":%d,\"batch\":%d,\"ckpt\":%d,\"ms\":%lld}",
       stage_csv.c_str(), out.epochs_run, out.final_loss, out.stopped_on_target ? "true" : "false",
       lm_json_escape(a.out_dir).c_str(), out.samples, out.crop, out.layers, out.batch, out.ckpt, run_ms);

    fprintf(stderr, "[train-dit] done in %.1f s\n", (double) run_ms / 1000.0);
    return 0;
}
