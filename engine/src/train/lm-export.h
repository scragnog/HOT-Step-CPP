#pragma once
// lm-export.h — PEFT writer (§2.4, §2.5, §3.6).
//
// <out>/adapter_model.safetensors   2*7*L tensors, F32, format "pt"
// <out>/adapter_config.json
// <out>/lm_train_log.json
// <out>/milestones/loss_<v>/        optional snapshots (L20)
//
// Layout proof (do NOT transpose anything):
//   our ggml A is ne0=in, ne1=r  -> contiguous read is row-major [r][in]  -> shape {r, in}
//   our ggml B is ne0=r, ne1=out -> contiguous read is row-major [out][r] -> shape {out, r}
// lm-adapter.h:180 does ggml_new_tensor_2d(ctx, ty, shape[1], shape[0]), which
// puts them straight back into A[in,r] / B[r,out].
//
// adapter_config.json is written BEFORE adapter_model.safetensors so a crash
// between the two never leaves a model file whose alpha silently falls back to
// 1.0 (lm-adapter.h:202-206).

#include "model-registry.h"  // registry_list_subdirs() — stale-milestone sweep
#include "train/lm-common.h"
#include "train/lm-graph.h"
#include "train/st-write.h"
#include "version.h"

#include <string>
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#else
#    include <unistd.h>
#endif

struct LmEpochRec {
    int       epoch = 0;
    double    loss = 0.0, lr = 0.0, grad_norm = 0.0;
    long long ms = 0;
};

struct LmMilestoneRec {
    double      loss  = 0.0;
    int         epoch = 0;
    std::string path;  // relative to <out>
};

struct LmExportMeta {
    std::string producer, created_at;
    std::string lm_path, lm_size, codes_path, tensors_dir, order;
    int         rank = 16, alpha = 32;
    double      lr = 1e-4, grad_clip = 1.0, weight_decay = 0.01, warmup_ratio = 0.05, target_loss = 0.4;
    int         epochs = 16, grad_accum = 4, seed = 42;
    bool        loss_on_cot = true;
    int         samples = 0, skipped_long = 0;
    int         max_len = 0;
    std::string max_len_source = "auto";

    // Trigger word embedded in the exported adapter's __metadata__ (empty = none).
    // docs/plans/2026-07-28-adapter-trigger-embedding.md §2.1
    std::string trigger, trigger_position;

    std::vector<LmEpochRec>     epoch_log;
    std::vector<LmMilestoneRec> milestones;
    bool                        target_stop = false;
    int                         target_stop_epoch = 0;
    double                      target_stop_loss  = 0.0;
    double                      final_loss = -1.0, best_loss = -1.0;
    int                         best_epoch = 0, epochs_run = 0;
    // Which epoch's adapter this file actually holds (2026-07-30). The export
    // is best-only now, so final_loss is not what shipped. saved_reason is
    // "target" (the epoch that tripped the auto-stop) or "best".
    double                      saved_loss = -1.0;
    int                         saved_epoch = 0;
    std::string                 saved_reason;

    size_t    vram_free_mb = 0, vram_total_mb = 0, vram_mirror_mb = 0, vram_est_mb = 0, vram_peak_mb = 0;
    long long total_ms = 0;

    // 4B low-VRAM path (2026-07-28 plan §2.3). Purely additive; every reader
    // defaults each field individually.
    bool        low_vram        = false;
    int         attn_head_block = 0;
    int         chunk           = 0;
    std::string vram_mode       = "naive";
    size_t      vram_base_mb = 0, vram_ckpt_mb = 0, vram_seg_peak_mb = 0;

    // Speed levers (2026-07-28 plan §2.3). Additive; every reader defaults each
    // field individually. RESUME RULE: a resume path must REFUSE to continue a
    // run whose recorded `weights` differs from the requested one — under
    // --weights bf16 the gradients are not the same quantity (S6).
    std::string weights = "f32-window";
    int         batch   = 1;

    // MUL_MAT activation-gradient formulation (engine/patches/mm-backward.patch).
    // Additive; readers default it to "outprod". Unlike `weights` this does NOT
    // change the quantity being computed — both arms compute the same gradient,
    // one via out_prod and one via mul_mat — so it is NOT a resume barrier.
    std::string bwd = "outprod";

    // Attention formulation (--attn, docs/plans/2026-09-02-lm-flash-attn.md D5).
    // Additive; readers default `attn_mode` to "exact" and `attn_prec` to "n/a".
    //
    // NOT a resume barrier, the same call as `bwd`: flash computes the same
    // gradient by a different summation order, within a measured drift band, so
    // a resume records it and refuses nothing (D7).
    //
    // TWO FIELDS, not one, and this is the point of the whole logging rule.
    // `attn_mode` is what was REQUESTED. `attn_prec` is what the backend
    // actually launched, read back through the registry after an epoch has run
    // (dit_flash_prec_label). The CUDA dispatch drops to the scalar f32 kernels
    // on pre-Ampere devices, at D != 128 and on unaligned views, so two runs
    // whose logs both say "flash" can differ in arithmetic. And because
    // op_params zero-init IS GGML_PREC_DEFAULT, every "flash" run on Ampere+ was
    // already TF32 before the knob existed and said nothing about it — the exact
    // blind spot this field closes (skill §2 point 6).
    std::string attn_mode = "exact";
    std::string attn_prec = "n/a";

    // Adapter parameterization and optimizer (2026-07-30). Recorded because a
    // finished run could not otherwise tell you whether it was a LoRA or a
    // LoKr, or AdamW or Muon — the same blind spot the DiT's `runtime` block
    // closed after a day of runs whose backend was unknowable after the fact.
    std::string adapter_type = "lora";
    int         lokr_dim     = 0;
    float       lokr_alpha   = 0.0f;
    int         lokr_factor  = 0;
    std::string optimizer     = "adamw";
    float       muon_lr_scale = 1.0f;
    int         muon_ns_steps = 0;
    int         muon_params   = 0;   // parameters actually on Muon
    int         muon_buckets  = 0;

    // Resume provenance (--init-adapter, 2026-08-09). Additive; readers
    // default to "trained from scratch". init_from_loss is the SOURCE run's
    // saved_loss — the loss this run's factors started from.
    std::string init_adapter;
    double      init_from_loss = -1.0;

    // Levers C and D (2026-09-02). Additive; readers default each field.
    // NEITHER IS A RESUME BARRIER — like `bwd` and unlike `weights`, they do
    // not change what the gradient IS, only which rows it is taken over, so a
    // resume adopts nothing and refuses nothing on their account. Recorded
    // anyway: a finished run could not otherwise tell you whether its trigger
    // was ever trained alone, or what it was regularised against.
    float                    caption_dropout = 0.0f;
    int                      reg_every       = 0;
    int                      reg_songs       = 0;
    int                      reg_topk        = 0;
    std::vector<std::string> reg_codes;
    std::string              reg_prior_dir;
};

// ─── adapter_config.json (frozen literal, §2.4) ─────────────────────────────
//
// Written by hand rather than via yyjson_mut so the emitted bytes match §2.4
// key-for-key and order-for-order (deviation noted in the handoff report).
static bool lm_write_adapter_config(const std::string & dir, int rank, int alpha, const std::string & base_model) {
    std::string j;
    j += "{\n";
    j += "  \"alpha_pattern\": {},\n";
    j += "  \"auto_mapping\": null,\n";
    j += "  \"base_model_name_or_path\": \"" + lm_json_escape(base_model) + "\",\n";
    j += "  \"bias\": \"none\",\n";
    j += "  \"fan_in_fan_out\": false,\n";
    j += "  \"inference_mode\": true,\n";
    j += "  \"init_lora_weights\": true,\n";
    j += "  \"layer_replication\": null,\n";
    j += "  \"layers_pattern\": null,\n";
    j += "  \"layers_to_transform\": null,\n";
    j += "  \"loftq_config\": {},\n";
    char b[64];
    snprintf(b, sizeof(b), "  \"lora_alpha\": %d,\n", alpha);
    j += b;
    j += "  \"lora_dropout\": 0.0,\n";
    j += "  \"megatron_config\": null,\n";
    j += "  \"megatron_core\": \"megatron.core\",\n";
    j += "  \"modules_to_save\": null,\n";
    j += "  \"peft_type\": \"LORA\",\n";
    snprintf(b, sizeof(b), "  \"r\": %d,\n", rank);
    j += b;
    j += "  \"rank_pattern\": {},\n";
    j += "  \"revision\": null,\n";
    j += "  \"target_modules\": [\n";
    j += "    \"k_proj\",\n    \"gate_proj\",\n    \"up_proj\",\n    \"v_proj\",\n    \"down_proj\",\n";
    j += "    \"q_proj\",\n    \"o_proj\"\n  ],\n";
    j += "  \"task_type\": \"CAUSAL_LM\",\n";
    j += "  \"use_dora\": false,\n";
    j += "  \"use_rslora\": false\n";
    j += "}\n";
    return pm_write_atomic(lm_join(dir, "adapter_config.json"), j);
}

// ─── lm_train_log.json (§2.5) ───────────────────────────────────────────────

static bool lm_write_train_log(const std::string & dir, const LmExportMeta & m) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "format", "hot-step-lm-train-v1");
    yyjson_mut_obj_add_strcpy(doc, root, "producer", m.producer.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "created_at", m.created_at.c_str());

    yyjson_mut_val * cfg = yyjson_mut_obj_add_obj(doc, root, "config");
    yyjson_mut_obj_add_strcpy(doc, cfg, "lm_size", m.lm_size.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "lm_path", m.lm_path.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "rank", m.rank);
    yyjson_mut_obj_add_int(doc, cfg, "alpha", m.alpha);
    yyjson_mut_obj_add_real(doc, cfg, "dropout", 0.0);
    yyjson_mut_obj_add_real(doc, cfg, "lr", m.lr);
    yyjson_mut_obj_add_int(doc, cfg, "epochs", m.epochs);
    yyjson_mut_obj_add_int(doc, cfg, "grad_accum", m.grad_accum);
    yyjson_mut_obj_add_real(doc, cfg, "grad_clip", m.grad_clip);
    yyjson_mut_obj_add_real(doc, cfg, "weight_decay", m.weight_decay);
    yyjson_mut_obj_add_real(doc, cfg, "warmup_ratio", m.warmup_ratio);
    yyjson_mut_obj_add_bool(doc, cfg, "loss_on_cot", m.loss_on_cot);
    yyjson_mut_obj_add_int(doc, cfg, "seed", m.seed);
    yyjson_mut_obj_add_strcpy(doc, cfg, "order", m.order.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "samples", m.samples);
    yyjson_mut_obj_add_int(doc, cfg, "skipped_long", m.skipped_long);
    yyjson_mut_obj_add_real(doc, cfg, "target_loss", m.target_loss);
    yyjson_mut_obj_add_int(doc, cfg, "max_len", m.max_len);
    yyjson_mut_obj_add_strcpy(doc, cfg, "max_len_source", m.max_len_source.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "codes", m.codes_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "tensors", m.tensors_dir.c_str());
    yyjson_mut_obj_add_bool(doc, cfg, "low_vram", m.low_vram);
    yyjson_mut_obj_add_int(doc, cfg, "attn_head_block", m.attn_head_block);
    yyjson_mut_obj_add_int(doc, cfg, "chunk", m.chunk);
    yyjson_mut_obj_add_strcpy(doc, cfg, "weights", m.weights.c_str());
    yyjson_mut_obj_add_int(doc, cfg, "batch", m.batch);
    yyjson_mut_obj_add_strcpy(doc, cfg, "bwd", m.bwd.c_str());
    // Always written, in both modes: "which attention did this adapter train
    // under, and in what arithmetic" has to be answerable from the log alone.
    yyjson_mut_obj_add_strcpy(doc, cfg, "attn_mode", m.attn_mode.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "attn_prec", m.attn_prec.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "adapter_type", m.adapter_type.c_str());
    if (m.adapter_type == "lokr") {
        yyjson_mut_obj_add_int(doc, cfg, "lokr_dim", m.lokr_dim);
        yyjson_mut_obj_add_real(doc, cfg, "lokr_alpha", (double) m.lokr_alpha);
        yyjson_mut_obj_add_int(doc, cfg, "lokr_factor", m.lokr_factor);
    }
    yyjson_mut_obj_add_strcpy(doc, cfg, "optimizer", m.optimizer.c_str());
    if (m.optimizer == "muon") {
        yyjson_mut_obj_add_real(doc, cfg, "muon_lr_scale", (double) m.muon_lr_scale);
        yyjson_mut_obj_add_int(doc, cfg, "muon_ns_steps", m.muon_ns_steps);
        // What the rule split ACTUALLY came out as — the number that says
        // whether Muon did anything. A rank-8 LoRA puts zero parameters on it.
        yyjson_mut_obj_add_int(doc, cfg, "muon_params", m.muon_params);
        yyjson_mut_obj_add_int(doc, cfg, "muon_buckets", m.muon_buckets);
    }
    yyjson_mut_obj_add_strcpy(doc, cfg, "trigger", m.trigger.c_str());
    yyjson_mut_obj_add_strcpy(doc, cfg, "trigger_position", m.trigger_position.c_str());
    // Always written (0.0 = off) — the question "was this adapter's trigger
    // ever trained on its own?" has to be answerable from the log alone.
    yyjson_mut_obj_add_real(doc, cfg, "caption_dropout", (double) m.caption_dropout);
    if (m.reg_every > 0) {
        yyjson_mut_obj_add_int(doc, cfg, "reg_every", m.reg_every);
        yyjson_mut_obj_add_int(doc, cfg, "reg_songs", m.reg_songs);
        yyjson_mut_obj_add_int(doc, cfg, "reg_topk", m.reg_topk);
        yyjson_mut_obj_add_strcpy(doc, cfg, "reg_prior_dir", m.reg_prior_dir.c_str());
        yyjson_mut_val * rc = yyjson_mut_obj_add_arr(doc, cfg, "reg_codes");
        for (size_t i = 0; i < m.reg_codes.size(); i++) {
            yyjson_mut_arr_add_strcpy(doc, rc, m.reg_codes[i].c_str());
        }
    }
    if (!m.init_adapter.empty()) {
        yyjson_mut_obj_add_strcpy(doc, cfg, "init_adapter", m.init_adapter.c_str());
        yyjson_mut_obj_add_real(doc, cfg, "init_from_loss", m.init_from_loss);
    }

    yyjson_mut_val * eps = yyjson_mut_obj_add_arr(doc, root, "epochs");
    for (size_t i = 0; i < m.epoch_log.size(); i++) {
        const LmEpochRec & e  = m.epoch_log[i];
        yyjson_mut_val *   eo = yyjson_mut_arr_add_obj(doc, eps);
        yyjson_mut_obj_add_int(doc, eo, "epoch", e.epoch);
        yyjson_mut_obj_add_real(doc, eo, "loss", e.loss);
        yyjson_mut_obj_add_real(doc, eo, "lr", e.lr);
        yyjson_mut_obj_add_real(doc, eo, "grad_norm", e.grad_norm);
        yyjson_mut_obj_add_int(doc, eo, "ms", (int64_t) e.ms);
    }

    if (m.target_stop) {
        yyjson_mut_val * ts = yyjson_mut_obj_add_obj(doc, root, "target_loss_stop");
        yyjson_mut_obj_add_int(doc, ts, "epoch", m.target_stop_epoch);
        yyjson_mut_obj_add_real(doc, ts, "loss", m.target_stop_loss);
    }

    yyjson_mut_obj_add_real(doc, root, "final_loss", m.final_loss);
    yyjson_mut_obj_add_real(doc, root, "best_loss", m.best_loss);
    yyjson_mut_obj_add_int(doc, root, "best_epoch", m.best_epoch);
    yyjson_mut_obj_add_real(doc, root, "saved_loss", m.saved_loss);
    yyjson_mut_obj_add_int(doc, root, "saved_epoch", m.saved_epoch);
    yyjson_mut_obj_add_strcpy(doc, root, "saved_reason",
                              m.saved_reason.empty() ? "best" : m.saved_reason.c_str());
    yyjson_mut_obj_add_int(doc, root, "epochs_run", m.epochs_run);

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
    yyjson_mut_obj_add_strcpy(doc, vr, "mode", m.vram_mode.c_str());
    yyjson_mut_obj_add_int(doc, vr, "base_mb", (int64_t) m.vram_base_mb);
    yyjson_mut_obj_add_int(doc, vr, "ckpt_mb", (int64_t) m.vram_ckpt_mb);
    yyjson_mut_obj_add_int(doc, vr, "seg_peak_mb", (int64_t) m.vram_seg_peak_mb);
    yyjson_mut_obj_add_strcpy(doc, vr, "weights", m.weights.c_str());
    yyjson_mut_obj_add_int(doc, vr, "batch", m.batch);

    yyjson_mut_obj_add_int(doc, root, "total_ms", (int64_t) m.total_ms);

    size_t len  = 0;
    char * text = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_ALLOW_INVALID_UNICODE, &len);
    yyjson_mut_doc_free(doc);
    if (!text) {
        return false;
    }
    const std::string s(text, len);
    free(text);
    return pm_write_atomic(lm_join(dir, "lm_train_log.json"), s);
}

// ─── adapter_model.safetensors ──────────────────────────────────────────────

struct LmExportResult {
    int       tensors = 0;
    long long bytes   = 0;
};

// Writes adapter_config.json then adapter_model.safetensors into `out_dir`.
static bool lm_export_peft(const LmLora & L, const Qwen3LMConfig & cfg, const LmExportMeta & meta,
                           const std::string & out_dir, LmExportResult * res, std::string * err) {
    (void) cfg;
    if (!pm_mkdir_p(out_dir)) {
        *err = "cannot create " + out_dir;
        return false;
    }
    if (!lm_write_adapter_config(out_dir, L.rank, (int) (L.alpha + 0.5f), meta.lm_path)) {
        *err = "cannot write adapter_config.json in " + out_dir;
        return false;
    }

    std::vector<STWTensor>          tensors;
    std::vector<std::vector<float>> store;
    tensors.reserve((size_t) (L.layer_hi - L.layer_lo) * QW_LORA_NSLOTS * 2);
    store.reserve(tensors.capacity());

    // layer-major, slot order q,k,v,o,gate,up,down, lora_A before lora_B
    for (int l = L.layer_lo; l < L.layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            const QwLoraPair & pr = L.layers[l].p[s];
            if (!pr.A || !pr.B) {
                continue;
            }
            ggml_tensor * both[2] = { pr.A, pr.B };
            const char *  suffix[2] = { "lora_A", "lora_B" };
            for (int k = 0; k < 2; k++) {
                ggml_tensor * t = both[k];
                store.push_back(std::vector<float>((size_t) ggml_nelements(t)));
                std::vector<float> & buf = store.back();
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                char nm[192];
                snprintf(nm, sizeof(nm), "base_model.model.model.layers.%d.%s.%s.weight", l, lm_slot_peft_name(s),
                         suffix[k]);
                STWTensor st;
                st.name  = nm;
                st.shape = { t->ne[1], t->ne[0] };  // torch [rows, cols]
                st.data  = buf.data();
                tensors.push_back(st);
            }
        }
    }

    // `store` may reallocate as it grows; it is 1:1 with `tensors`, so re-point
    // every data pointer once the vector has stopped moving.
    GGML_ASSERT(store.size() == tensors.size());
    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].data = store[i].data();
    }

    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "pt" });
    md.push_back({ "producer", std::string("ace-train ") + ACE_VERSION });
    md.push_back({ "hot_step_lm_trainer", "v1" });
    // Trigger word (all three keys together or none — §2.1). Unknown metadata
    // keys are ignored by every other safetensors consumer, so the adapter stays
    // loadable by PEFT / ComfyUI / Side-Step exactly as before.
    if (!meta.trigger.empty()) {
        md.push_back({ "hot_step_trigger", meta.trigger });
        md.push_back({ "hot_step_trigger_position", meta.trigger_position.empty() ? std::string("prepend")
                                                                                  : meta.trigger_position });
        md.push_back({ "modelspec.trigger_phrase", meta.trigger });
    }

    const std::string sf = lm_join(out_dir, "adapter_model.safetensors");
    if (!st_write_file(sf.c_str(), tensors, md, STW_F32)) {
        *err = "cannot write " + sf;
        return false;
    }
    long long bytes = 0;
    pm_stat_file(sf, &bytes, NULL);
    if (res) {
        res->tensors = (int) tensors.size();
        res->bytes   = bytes;
    }
    return true;
}

// ─── LoKr export (2026-07-30) ───────────────────────────────────────────────
//
// LyCORIS layout, IDENTICAL to what the DiT trainer writes, so adapter-merge.h's
// existing kron reader applies and there is not a second format to support:
//
//   <stem>.alpha        [1]
//   <stem>.lokr_w1      [out_l, in_m]                    (row-major, so the ggml
//   <stem>.lokr_w2      [out_k, in_n]   monolithic        tensors are written
//   <stem>.lokr_w2_a    [out_k, dim]    factorized        transposed — ne1 first)
//   <stem>.lokr_w2_b    [dim,   in_n]
//
// stem = "lycoris_layers_<L>_<site with dots as underscores>", matching the DiT.
// BF16 like the DiT's (the LM's PEFT LoRA export stays F32 for Side-Step
// parity, but nothing external produces LM LoKr files to be compatible with).
static bool lm_export_lokr(const LmLora & L, const LmExportMeta & meta, const std::string & out_dir,
                           LmExportResult * res, std::string * err) {
    if (!pm_mkdir_p(out_dir)) {
        *err = "cannot create " + out_dir;
        return false;
    }
    std::vector<STWTensor>          tensors;
    std::vector<std::vector<float>> store;

    for (int l = L.layer_lo; l < L.layer_hi; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            const QwLoraPair & pr = L.layers[l].p[s];
            if (!pr.has_lokr()) {
                continue;
            }
            std::string site(lm_slot_peft_name(s));
            for (size_t i = 0; i < site.size(); i++) {
                if (site[i] == '.') {
                    site[i] = '_';
                }
            }
            char stemb[128];
            snprintf(stemb, sizeof(stemb), "lycoris_layers_%d_%s", l, site.c_str());
            const std::string stem(stemb);

            // alpha, reconstructed from the in-graph scale so the file always
            // agrees with what training actually applied.
            store.push_back(std::vector<float>(1, pr.lokr_scale * (float) L.lokr_dim));
            STWTensor sa;
            sa.name  = stem + ".alpha";
            sa.shape = { 1 };
            sa.data  = nullptr;
            tensors.push_back(sa);

            ggml_tensor * ts[3]  = { pr.w1, pr.w2 ? pr.w2 : pr.w2_a, pr.w2 ? nullptr : pr.w2_b };
            const char *  sfx[3] = { "lokr_w1", pr.w2 ? "lokr_w2" : "lokr_w2_a", "lokr_w2_b" };
            for (int i = 0; i < 3 && ts[i]; i++) {
                store.push_back(std::vector<float>((size_t) ggml_nelements(ts[i])));
                std::vector<float> & v = store.back();
                ggml_backend_tensor_get(ts[i], v.data(), 0, v.size() * sizeof(float));
                STWTensor st;
                st.name  = stem + "." + sfx[i];
                st.shape = { ts[i]->ne[1], ts[i]->ne[0] };  // row-major (rows, cols)
                st.data  = nullptr;
                tensors.push_back(st);
            }
        }
    }
    // `store` reallocates as it grows; re-point once it has stopped moving.
    if (store.size() != tensors.size()) {
        *err = "lokr export bookkeeping mismatch";
        return false;
    }
    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].data = store[i].data();
    }

    char cfgb[256];
    snprintf(cfgb, sizeof(cfgb),
             "{\"linear_dim\":%d,\"linear_alpha\":%.6g,\"factor\":%d,\"decompose_both\":%s,\"target_mlp\":true}",
             L.lokr_dim, (double) L.lokr_alpha, L.lokr_factor, L.lokr_decompose ? "true" : "false");

    std::vector<std::pair<std::string, std::string>> md;
    md.push_back({ "format", "lycoris" });
    md.push_back({ "algo", "lokr" });
    md.push_back({ "producer", meta.producer });
    // Distinguishes an LM LoKr from a DiT one: the stems collide by design (both
    // are lycoris_layers_<L>_<site>) and only the site set differs, so a loader
    // that guesses wrong would silently apply an LM adapter to a DiT.
    md.push_back({ "hot_step_lm_lokr", "v1" });
    md.push_back({ "lokr_config", std::string(cfgb) });
    if (!meta.trigger.empty()) {
        md.push_back({ "hot_step_trigger", meta.trigger });
        md.push_back({ "hot_step_trigger_position",
                       meta.trigger_position.empty() ? std::string("prepend") : meta.trigger_position });
        md.push_back({ "modelspec.trigger_phrase", meta.trigger });
    }

    const std::string sf = lm_join(out_dir, "lokr_weights.safetensors");
    if (!st_write_file(sf.c_str(), tensors, md, STW_BF16)) {
        *err = "cannot write " + sf;
        return false;
    }
    long long bytes = 0;
    pm_stat_file(sf, &bytes, NULL);
    if (res) {
        res->tensors = (int) tensors.size();
        res->bytes   = bytes;
    }
    return true;
}

// ─── milestone snapshots (L20) ──────────────────────────────────────────────

// Remove milestone directories left behind by a PREVIOUS run into the same <out>.
//
// meta->milestones starts empty on every run, so lm_milestone_prune() only ever
// considers snapshots this run created — a loss_8.6/ from an unrelated earlier
// run (different rank, different base, possibly a different dataset) survives on
// disk and is offered by the UI's adapter picker while lm_train_log.json lists
// `milestones: []`. Runs once, before the first epoch export, whether or not
// milestones are enabled this time.
static void lm_milestone_reset(const std::string & out_dir) {
    const std::string        root = lm_join(out_dir, "milestones");
    std::vector<std::string> subs;
    registry_list_subdirs(root.c_str(), &subs);
    int removed = 0;
    for (size_t i = 0; i < subs.size(); i++) {
        if (subs[i].rfind("loss_", 0) != 0) {
            continue;  // only our own naming — never touch anything else
        }
        const std::string dir = lm_join(root, subs[i]);
        remove(lm_join(dir, "adapter_model.safetensors").c_str());
        remove(lm_join(dir, "adapter_config.json").c_str());
        remove(lm_join(dir, "lm_train_log.json").c_str());
#ifdef _WIN32
        _rmdir(dir.c_str());  // fails harmlessly if the user put something else there
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

static void lm_milestone_prune(const std::string & out_dir, std::vector<LmMilestoneRec> * list, int keep) {
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
        remove(lm_join(dir, "adapter_model.safetensors").c_str());
        remove(lm_join(dir, "adapter_config.json").c_str());
#ifdef _WIN32
        _rmdir(dir.c_str());
#else
        rmdir(dir.c_str());
#endif
        list->erase(list->begin() + (long) worst);
    }
}
