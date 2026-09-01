#pragma once
// dit-export.h — PEFT writer, dit_train_log.json and milestone snapshots
// (plan §2.3, §2.4, §3.8).
//
// <out>/adapter_config.json          written FIRST (D19)
// <out>/adapter_model.safetensors    2 * sites * K tensors, F32, format "pt"
//                                    (LoKR writes <out>/lokr_weights.safetensors
//                                    instead — dit-adapter-lokr.h owns the layout)
// <out>/dit_train_log.json           rewritten after every epoch
// <out>/milestones/loss_<v>/         optional snapshots, pruned to --milestone-keep

#include "model-registry.h"  // registry_list_subdirs
#include "train/dit-adapter.h"
#include "train/lm-common.h"

#include <string>
#include <utility>  // std::pair (profile_ms) — MSVC gets it via <vector>, gcc/clang do not
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#else
#    include <unistd.h>
#endif

struct DitEpochRec {
    int       epoch = 0;
    double    loss = 0.0, ma5 = 0.0, raw_loss = 0.0, lr = 0.0, grad_norm = 0.0;
    long long ms = 0;
};

struct DitMilestoneRec {
    double      loss  = 0.0;
    int         epoch = 0;
    std::string path;  // relative to <out>
};

struct DitTrainLog {
    std::string producer, created_at;
    // config (§2.4, snake_case to mirror Side-Step's logs)
    std::string adapter_type = "lora";
    int         rank = 16, alpha = 32;
    // LoKR geometry, written only when adapter_type == "lokr".
    bool  is_lokr             = false;
    int   lokr_dim            = 512;
    float lokr_alpha          = 512.0f;
    int   lokr_factor         = 6;
    bool  lokr_decompose_both = true;
    bool        target_mlp = false;
    int         layers = 0, n_layers_total = 0;
    std::string layers_source = "auto";
    std::string dit_path, dit_name, tensors;
    int         samples = 0;
    int         crop = 0, crop_min = 375, crop_max = 1250;
    std::string crop_source = "auto";
    std::string crop_anchor = "song", crop_mode = "structured";
    float       crop_start_frac = 0.2f, crop_end_frac = 0.2f;
    // What the endpoint shares actually came out as after the crop-coverage
    // scaling — the requested fracs alone no longer describe the run.
    float       crop_start_frac_eff = 0.0f, crop_end_frac_eff = 0.0f;
    int         crop_start_window = 1;
    float       crop_endpoint_k  = 2.0f;
    std::string mirror      = "f32";  // frozen-weight mirror precision (f32|bf16)
    std::string bwd         = "outprod";  // MUL_MAT activation-gradient form (outprod|mm)
    double      lr = 5e-4;
    int         epochs = 100, grad_accum = 4;
    // Effective micro-batch after the dataset-size and A1 repeat_back clamps —
    // what actually ran, never what was asked for.
    int         batch = 1;
    // Gradient-checkpoint segments that actually ran (1 = off).
    int         ckpt = 1;
    double      grad_clip = 1.0, weight_decay = 0.01, warmup_ratio = 0.05;
    int         seed  = 42;
    std::string order = "shuffle";
    std::string loss_weighting = "flow_snr";
    double      snr_gamma = 5.0, t_bias = 0.5;
    // Resume provenance (--init-adapter, 2026-08-10). Additive; readers default
    // to "trained from scratch". init_from_ma5 is the SOURCE run's saved_ma5.
    std::string init_adapter;
    double      init_from_ma5 = -1.0;
    bool        channel_balance = true;
    double      timestep_mu = -0.4, timestep_sigma = 1.0, t_min = 0.0, t_max = 1.0, cfg_ratio = 0.15;
    int         genre_ratio = 0;
    double      target_loss = 0.4;
    // Trigger word embedded in the adapter's __metadata__ (empty = none).
    // docs/plans/2026-07-28-adapter-trigger-embedding.md §2.2
    std::string trigger, trigger_position;

    std::vector<DitEpochRec>     epochs_log;
    std::vector<DitMilestoneRec> milestones;
    bool                         target_stop = false;
    int                          target_stop_epoch = 0;
    double                       target_stop_loss = 0.0, target_stop_ma5 = 0.0;
    double                       final_loss = -1.0, best_loss = -1.0;
    int                          best_epoch = 0, epochs_run = 0;
    // Which epoch's adapter is actually IN the run dir (2026-07-30). Since the
    // export became best-only, "final_loss" is no longer what shipped — read
    // these instead. saved_reason: "target" (the epoch that tripped the auto-
    // stop) or "best" (lowest ma5 seen).
    double                       saved_ma5 = -1.0;
    int                          saved_epoch = 0;
    std::string                  saved_reason;
    bool                         partial_depth = false;

    size_t    vram_free_mb = 0, vram_total_mb = 0, vram_mirror_mb = 0, vram_est_mb = 0, vram_peak_mb = 0;
    long long total_ms = 0;

    // ─── runtime facts (2026-07-30) ──────────────────────────────────────────
    //
    // Every one of these was needed to interpret a run this week and had been
    // thrown away: the backend went only to a stderr tail the server drops, and
    // the graph/split counts went nowhere at all. `ckpt` above is already the
    // RESOLVED segment count; these join it so a finished run explains itself.
    std::string backend;                                  // "CUDA0" / "CPU" / ...
    int         graph_nodes = 0;                          // the fwd+bwd graph
    int         sched_splits = 0, sched_copies = 0;       // > 1 split = backend fallback
    // Mean us per micro-step bucket, only when --profile-step ran (steps == 0
    // means it did not, and the whole block is omitted from the JSON).
    long long        profile_steps = 0;
    std::vector<std::pair<std::string, double>> profile_ms;
};

static bool dit_write_train_log(const std::string & dir, const DitTrainLog & m) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    // v3 (2026-09-01): the end share is back on (split flush-jitter / closing
    // region), so a v2 run's crop stream no longer replays at the same seed.
    yyjson_mut_obj_add_strcpy(doc, root, "format", "hot-step-dit-train-v3");
    yyjson_mut_obj_add_strcpy(doc, root, "producer", m.producer.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "created_at", m.created_at.c_str());

    yyjson_mut_val * cfg = yyjson_mut_obj_add_obj(doc, root, "config");
    yyjson_mut_obj_add_strcpy(doc, cfg, "adapter_type", m.adapter_type.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "rank", m.rank);
    yyjson_mut_obj_add_int(doc, cfg, "alpha", m.alpha);
    if (m.is_lokr) {
        yyjson_mut_obj_add_int(doc, cfg, "lokr_dim", m.lokr_dim);
        yyjson_mut_obj_add_real(doc, cfg, "lokr_alpha", (double) m.lokr_alpha);
        yyjson_mut_obj_add_int(doc, cfg, "lokr_factor", m.lokr_factor);
        yyjson_mut_obj_add_bool(doc, cfg, "lokr_decompose_both", m.lokr_decompose_both);
    }
    yyjson_mut_obj_add_real(doc, cfg, "dropout", 0.0);
    yyjson_mut_obj_add_bool(doc, cfg, "target_mlp", m.target_mlp);
    yyjson_mut_obj_add_int(doc, cfg, "layers", m.layers);
    yyjson_mut_obj_add_strcpy(doc, cfg, "layers_source", m.layers_source.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "n_layers_total", m.n_layers_total);
    yyjson_mut_obj_add_strcpy(doc, cfg, "dit_path", m.dit_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "dit_name", m.dit_name.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "tensors", m.tensors.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "samples", m.samples);
    yyjson_mut_obj_add_int(doc, cfg, "crop", m.crop);
    yyjson_mut_obj_add_strcpy(doc, cfg, "crop_source", m.crop_source.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "crop_min", m.crop_min);
    yyjson_mut_obj_add_int(doc, cfg, "crop_max", m.crop_max);
    yyjson_mut_obj_add_strcpy(doc, cfg, "crop_anchor", m.crop_anchor.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "crop_mode", m.crop_mode.c_str());
    yyjson_mut_obj_add_real(doc, cfg, "crop_start_frac", (double) m.crop_start_frac);
    yyjson_mut_obj_add_real(doc, cfg, "crop_end_frac", (double) m.crop_end_frac);
    yyjson_mut_obj_add_real(doc, cfg, "crop_start_frac_eff", (double) m.crop_start_frac_eff);
    yyjson_mut_obj_add_real(doc, cfg, "crop_end_frac_eff", (double) m.crop_end_frac_eff);
    yyjson_mut_obj_add_int(doc, cfg, "crop_start_window", m.crop_start_window);
    yyjson_mut_obj_add_real(doc, cfg, "crop_endpoint_k", (double) m.crop_endpoint_k);
    yyjson_mut_obj_add_strcpy(doc, cfg, "mirror", m.mirror.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "bwd", m.bwd.c_str());
    yyjson_mut_obj_add_real(doc, cfg, "lr", m.lr);
    yyjson_mut_obj_add_int(doc, cfg, "epochs", m.epochs);
    yyjson_mut_obj_add_int(doc, cfg, "grad_accum", m.grad_accum);
    yyjson_mut_obj_add_int(doc, cfg, "batch", m.batch);
    yyjson_mut_obj_add_int(doc, cfg, "ckpt", m.ckpt);
    yyjson_mut_obj_add_real(doc, cfg, "grad_clip", m.grad_clip);
    yyjson_mut_obj_add_real(doc, cfg, "weight_decay", m.weight_decay);
    yyjson_mut_obj_add_real(doc, cfg, "warmup_ratio", m.warmup_ratio);
    yyjson_mut_obj_add_int(doc, cfg, "seed", m.seed);
    yyjson_mut_obj_add_strcpy(doc, cfg, "order", m.order.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "loss_weighting", m.loss_weighting.c_str());
    yyjson_mut_obj_add_real(doc, cfg, "snr_gamma", m.snr_gamma);
    yyjson_mut_obj_add_real(doc, cfg, "t_bias", m.t_bias);
    yyjson_mut_obj_add_bool(doc, cfg, "channel_balance", m.channel_balance);
    yyjson_mut_obj_add_real(doc, cfg, "timestep_mu", m.timestep_mu);
    yyjson_mut_obj_add_real(doc, cfg, "timestep_sigma", m.timestep_sigma);
    yyjson_mut_obj_add_real(doc, cfg, "t_min", m.t_min);
    yyjson_mut_obj_add_real(doc, cfg, "t_max", m.t_max);
    yyjson_mut_obj_add_real(doc, cfg, "cfg_ratio", m.cfg_ratio);
    yyjson_mut_obj_add_int(doc, cfg, "genre_ratio", m.genre_ratio);
    yyjson_mut_obj_add_real(doc, cfg, "target_loss", m.target_loss);
    yyjson_mut_obj_add_strcpy(doc, cfg, "trigger", m.trigger.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "trigger_position", m.trigger_position.c_str());
    if (!m.init_adapter.empty()) {
        yyjson_mut_obj_add_strcpy(doc, cfg, "init_adapter", m.init_adapter.c_str());
        yyjson_mut_obj_add_real(doc, cfg, "init_from_ma5", m.init_from_ma5);
    }

    yyjson_mut_val * eps = yyjson_mut_obj_add_arr(doc, root, "epochs");
    for (size_t i = 0; i < m.epochs_log.size(); i++) {
        const DitEpochRec & e  = m.epochs_log[i];
        yyjson_mut_val *    eo = yyjson_mut_arr_add_obj(doc, eps);
        yyjson_mut_obj_add_int(doc, eo, "epoch", e.epoch);
        yyjson_mut_obj_add_real(doc, eo, "loss", e.loss);
        yyjson_mut_obj_add_real(doc, eo, "ma5", e.ma5);
        yyjson_mut_obj_add_real(doc, eo, "raw_loss", e.raw_loss);
        yyjson_mut_obj_add_real(doc, eo, "lr", e.lr);
        yyjson_mut_obj_add_real(doc, eo, "grad_norm", e.grad_norm);
        yyjson_mut_obj_add_int(doc, eo, "ms", (int64_t) e.ms);
    }

    if (m.target_stop) {
        yyjson_mut_val * ts = yyjson_mut_obj_add_obj(doc, root, "target_loss_stop");
        yyjson_mut_obj_add_int(doc, ts, "epoch", m.target_stop_epoch);
        yyjson_mut_obj_add_real(doc, ts, "loss", m.target_stop_loss);
        yyjson_mut_obj_add_real(doc, ts, "ma5", m.target_stop_ma5);
    }

    yyjson_mut_obj_add_real(doc, root, "final_loss", m.final_loss);
    yyjson_mut_obj_add_real(doc, root, "best_loss", m.best_loss);
    yyjson_mut_obj_add_int(doc, root, "best_epoch", m.best_epoch);
    yyjson_mut_obj_add_int(doc, root, "epochs_run", m.epochs_run);
    // Which epoch's weights this file actually holds.
    yyjson_mut_obj_add_real(doc, root, "saved_ma5", m.saved_ma5);
    yyjson_mut_obj_add_int(doc, root, "saved_epoch", m.saved_epoch);
    yyjson_mut_obj_add_strcpy(doc, root, "saved_reason",
                              m.saved_reason.empty() ? "best" : m.saved_reason.c_str());
    yyjson_mut_obj_add_bool(doc, root, "partial_depth", m.partial_depth);

    yyjson_mut_val * ms = yyjson_mut_obj_add_arr(doc, root, "milestones");
    for (size_t i = 0; i < m.milestones.size(); i++) {
        yyjson_mut_val * mo = yyjson_mut_arr_add_obj(doc, ms);
        yyjson_mut_obj_add_real(doc, mo, "loss", m.milestones[i].loss);
        yyjson_mut_obj_add_int(doc, mo, "epoch", m.milestones[i].epoch);
        yyjson_mut_obj_add_strcpy(doc, mo, "path", m.milestones[i].path.c_str());
    }

    yyjson_mut_val * vr = yyjson_mut_obj_add_obj(doc, root, "vram");
    yyjson_mut_obj_add_int(doc, vr, "free_mb", (int64_t) m.vram_free_mb);
    yyjson_mut_obj_add_int(doc, vr, "total_mb", (int64_t) m.vram_total_mb);
    yyjson_mut_obj_add_int(doc, vr, "mirror_mb", (int64_t) m.vram_mirror_mb);
    yyjson_mut_obj_add_int(doc, vr, "est_mb", (int64_t) m.vram_est_mb);
    yyjson_mut_obj_add_int(doc, vr, "peak_mb", (int64_t) m.vram_peak_mb);

    yyjson_mut_val * rt = yyjson_mut_obj_add_obj(doc, root, "runtime");
    yyjson_mut_obj_add_strcpy(doc, rt, "backend", m.backend.c_str());
    yyjson_mut_obj_add_int(doc, rt, "graph_nodes", m.graph_nodes);
    yyjson_mut_obj_add_int(doc, rt, "sched_splits", m.sched_splits);
    yyjson_mut_obj_add_int(doc, rt, "sched_copies", m.sched_copies);
    if (m.profile_steps > 0) {
        yyjson_mut_val * pf = yyjson_mut_obj_add_obj(doc, rt, "profile_ms");
        yyjson_mut_obj_add_int(doc, pf, "steps", (int64_t) m.profile_steps);
        for (size_t i = 0; i < m.profile_ms.size(); i++) {
            // strcpy the KEY: the add_* helpers take the key by pointer without
            // copying, and these come from a vector rather than string literals.
            yyjson_mut_obj_add(pf, yyjson_mut_strcpy(doc, m.profile_ms[i].first.c_str()),
                               yyjson_mut_real(doc, m.profile_ms[i].second));
        }
    }

    yyjson_mut_obj_add_int(doc, root, "total_ms", (int64_t) m.total_ms);

    size_t len  = 0;
    char * text = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_ALLOW_INVALID_UNICODE, &len);
    yyjson_mut_doc_free(doc);
    if (!text) {
        return false;
    }
    const std::string s(text, len);
    ::free(text);
    return pm_write_atomic(lm_join(dir, "dit_train_log.json"), s);
}

// ─── PEFT export (delegates to the parameterization seam, §3.6/§3.8) ────────

static bool dit_export_peft(const DitAdapter & ad, const DitExportMeta & meta, const char * out_dir,
                            DitExportResult * res, std::string * err) {
    return ad.exportDir(out_dir, meta, res, err);
}

// ─── milestones ─────────────────────────────────────────────────────────────

// Every file exportDir() can have written, whatever the parameterization: PEFT
// (adapter_model.safetensors + adapter_config.json) or LyCORIS LoKR
// (lokr_weights.safetensors). Naming only the PEFT pair here would leave a LoKR
// snapshot behind and _rmdir would then refuse the directory.
static void dit_milestone_remove_weights(const std::string & dir) {
    remove(lm_join(dir, "adapter_model.safetensors").c_str());
    remove(lm_join(dir, "adapter_config.json").c_str());
    remove(lm_join(dir, "lokr_weights.safetensors").c_str());
}

// Remove milestone dirs left by a PREVIOUS run into the same <out>: this run's
// log would not list them, yet the UI's adapter picker would still offer them.
static void dit_milestone_reset(const std::string & out_dir) {
    const std::string        root = lm_join(out_dir, "milestones");
    std::vector<std::string> subs;
    registry_list_subdirs(root.c_str(), &subs);
    int removed = 0;
    for (size_t i = 0; i < subs.size(); i++) {
        if (subs[i].rfind("loss_", 0) != 0) {
            continue;  // only our own naming — never touch anything else
        }
        const std::string dir = lm_join(root, subs[i]);
        dit_milestone_remove_weights(dir);
        remove(lm_join(dir, "dit_train_log.json").c_str());
#ifdef _WIN32
        _rmdir(dir.c_str());
#else
        rmdir(dir.c_str());
#endif
        removed++;
    }
    if (removed > 0) {
        char b[128];
        snprintf(b, sizeof(b), "cleared %d milestone dir(s) from a previous run", removed);
        lm_log("info", b);
    }
}

static void dit_milestone_prune(const std::string & out_dir, std::vector<DitMilestoneRec> * list, int keep) {
    if (keep <= 0 || (int) list->size() <= keep) {
        return;
    }
    // The ladder descends, so "newest" == smallest loss. Drop the largest.
    while ((int) list->size() > keep) {
        size_t worst = 0;
        for (size_t i = 1; i < list->size(); i++) {
            if ((*list)[i].loss > (*list)[worst].loss) {
                worst = i;
            }
        }
        const std::string dir = lm_join(out_dir, (*list)[worst].path);
        dit_milestone_remove_weights(dir);
#ifdef _WIN32
        _rmdir(dir.c_str());
#else
        rmdir(dir.c_str());
#endif
        list->erase(list->begin() + (long) worst);
    }
}
