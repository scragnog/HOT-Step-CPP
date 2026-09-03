#pragma once
// lm-resume.h — continue LM adapter training from an exported run
// (`ace-train train-lm --init-adapter <dir>`), 2026-08-09.
//
// Why: training stops on a uniform target loss, but the eval tool
// (server/scripts/lm-adapter-eval.ts) showed the *distribution shift toward
// the artist* peaks at a per-artist point that loss cannot see. The strategy
// is train-past-then-select: resume an existing adapter with a lower target
// and milestone snapshots on, then pick the best snapshot by eval. Resume
// pays only for the NEW epochs — the existing 183-adapter corpus keeps its
// runs as starting points instead of retraining from scratch.
//
// What resuming means here: the exported factors are the complete trainable
// state; optimizer momentum is not saved, so a resumed run is a fine-tune
// continuation (fresh warmup + cosine), not a bit-exact "never stopped" run.
// Exports are BF16 for LoKr (F32 for PEFT LoRA), so LoKr resume also eats one
// BF16 rounding of the factors. Both are deliberate v1 trade-offs.
//
// REFUSE RULES (lm-export.h §2.3 "RESUME RULE" made real). Identity
// hyperparameters come FROM THE SOURCE RUN's lm_train_log.json; a CLI value
// that contradicts the source is a hard exit 2, never a coercion:
//   - adapter_type, and its shape params (rank/alpha | lokr dim/alpha/factor):
//     different values would allocate tensors the file cannot fill.
//   - weights (f32-window|bf16): under bf16 the gradients are not the same
//     quantity (S6) — continuing across modes silently changes what "loss
//     4.0 → 1.5" means.
//   - lm_size: a 4B adapter on a 0.6B base dies late and confusingly
//     ("36 layers but model has 28"); refuse up front. A different file of
//     the SAME size only warns — quant/bf16 variants of one base are fine.
//
// NOT identity, deliberately: `bwd` and `attn` (--attn exact|flash|flash-f32,
// D7). Both change the SUMMATION ORDER of a gradient, not the gradient — flash
// recomputes the softmax from Q/K/LSE in tiles instead of reading back a
// retained array, and over 200 same-seed DiT epochs it drifted LESS than
// `--bwd mm` does. So neither is read from the source log, neither is adopted,
// and a CLI value that differs from the source run's is not a contradiction to
// refuse. They are RECORDED in lm_train_log.json (lm-export.h attn_mode /
// attn_prec) so a finished run can still say which it used; `weights` is the
// only lever in this family that IS a barrier, because bf16 rounds the
// activation gradient at every layer and changes what the loss curve means.
//
// The honest gate: a resumed run's FIRST epoch loss should land near the
// source's saved_loss (teacher-forced, same data). lm-train-run.h logs the
// comparison; a large gap means the init mapping or the dataset changed.

#include "safetensors.h"
#include "train/lm-graph.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ─── source-run identity (from lm_train_log.json) ───────────────────────────

struct LmResumeSource {
    std::string dir;                    // the --init-adapter dir
    std::string adapter_type = "lora";
    int         rank = 0, alpha = 0;
    int         lokr_dim = 0, lokr_factor = 0;
    float       lokr_alpha = 0.0f;
    bool        lokr_decompose = true;
    std::string weights;                // f32-window|bf16
    std::string lm_size, lm_path;
    std::string trigger, trigger_position;
    double      saved_loss  = -1.0;
    int         saved_epoch = 0;
    // Prodigy's final step-size estimate from the source run (0 = not prodigy
    // or an older log). Adopted as this leg's --prodigy-d0 unless the user set
    // one, so a chained leg does not restart its warm-up from 1e-6.
    double      prodigy_d   = 0.0;
};

/** Which identity flags the user typed explicitly (tracked by cmd_train_lm's
 *  parser). Explicit-and-different from the source → refuse; omitted → adopt. */
struct LmResumeExplicit {
    bool rank = false, alpha = false, adapter_type = false;
    bool lokr_dim = false, lokr_alpha = false, lokr_factor = false;
    bool weights = false;
    bool prodigy_d0 = false;   // not identity: only decides whether the source's d is adopted
};

static bool lm_resume_read_log(const std::string & dir, LmResumeSource * src, std::string * err) {
    const std::string log_path = dir + "/lm_train_log.json";
    yyjson_doc * doc = yyjson_read_file(log_path.c_str(), 0, NULL, NULL);
    if (!doc) {
        *err = "cannot read " + log_path + " — resume needs the source run's recorded config";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * cfg  = yyjson_obj_get(root, "config");
    if (!cfg || !yyjson_is_obj(cfg)) {
        yyjson_doc_free(doc);
        *err = log_path + " has no config object";
        return false;
    }
    auto s = [&](const char * k) -> std::string {
        yyjson_val * v = yyjson_obj_get(cfg, k);
        return yyjson_is_str(v) ? yyjson_get_str(v) : "";
    };
    auto i = [&](const char * k, int dflt) -> int {
        yyjson_val * v = yyjson_obj_get(cfg, k);
        return yyjson_is_num(v) ? (int) yyjson_get_num(v) : dflt;
    };
    src->dir              = dir;
    src->adapter_type     = s("adapter_type").empty() ? "lora" : s("adapter_type");
    src->rank             = i("rank", 0);
    src->alpha            = i("alpha", 0);
    src->lokr_dim         = i("lokr_dim", 0);
    src->lokr_factor      = i("lokr_factor", 0);
    {
        yyjson_val * v  = yyjson_obj_get(cfg, "lokr_alpha");
        src->lokr_alpha = yyjson_is_num(v) ? (float) yyjson_get_num(v) : 0.0f;
    }
    src->weights          = s("weights").empty() ? "f32-window" : s("weights");
    {
        yyjson_val * v = yyjson_obj_get(cfg, "prodigy_d");
        src->prodigy_d = yyjson_is_num(v) ? yyjson_get_num(v) : 0.0;
    }
    src->lm_size          = s("lm_size");
    src->lm_path          = s("lm_path");
    src->trigger          = s("trigger");
    src->trigger_position = s("trigger_position");
    {
        yyjson_val * v   = yyjson_obj_get(root, "saved_loss");
        src->saved_loss  = yyjson_is_num(v) ? yyjson_get_num(v) : -1.0;
        yyjson_val * e   = yyjson_obj_get(root, "saved_epoch");
        src->saved_epoch = yyjson_is_num(e) ? (int) yyjson_get_num(e) : 0;
    }
    yyjson_doc_free(doc);
    return true;
}

// ─── weight loading (the exporters inverted) ────────────────────────────────

/** BF16 → F32 widening (bf16 is the top half of the f32 bit pattern). */
static void lm_resume_bf16_to_f32(const uint16_t * in, float * out, size_t n) {
    for (size_t j = 0; j < n; j++) {
        const uint32_t u = ((uint32_t) in[j]) << 16;
        memcpy(&out[j], &u, sizeof(float));
    }
}

/** Fetch one tensor from `st` into `dst` (an F32 trainer param). The exporters
 *  wrote raw ggml memory with the shape recorded as {ne1, ne0}, so byte order
 *  matches exactly — only the dtype may need widening. */
static bool lm_resume_fill(const STFile & st, const char * name, ggml_tensor * dst, std::string * err) {
    const STEntry * e = st_find(st, name);
    if (!e) {
        *err = std::string("tensor missing from adapter file: ") + name;
        return false;
    }
    int64_t n = 1;
    for (int d = 0; d < e->n_dims; d++) {
        n *= e->shape[d];
    }
    if (n != ggml_nelements(dst)) {
        char b[224];
        snprintf(b, sizeof(b), "%s holds %lld values but the trainer tensor wants %lld — source run shape mismatch",
                 name, (long long) n, (long long) ggml_nelements(dst));
        *err = b;
        return false;
    }
    const void * data = st_data(st, *e);
    if (e->dtype == "F32") {
        ggml_backend_tensor_set(dst, data, 0, (size_t) n * sizeof(float));
    } else if (e->dtype == "BF16") {
        std::vector<float> f((size_t) n);
        lm_resume_bf16_to_f32((const uint16_t *) data, f.data(), (size_t) n);
        ggml_backend_tensor_set(dst, f.data(), 0, f.size() * sizeof(float));
    } else {
        *err = std::string(name) + " has unsupported dtype " + e->dtype;
        return false;
    }
    return true;
}

/** Load an exported adapter's factors into the freshly-initialized trainer
 *  params. Every expected tensor must be present — a partial adapter would
 *  train from a silently-wrong start. Returns the number of tensors loaded. */
static bool lm_resume_load(LmLora * L, const std::string & dir, int * n_loaded, std::string * err) {
    const std::string path =
        dir + (L->is_lokr ? "/lokr_weights.safetensors" : "/adapter_model.safetensors");
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    int  count = 0;
    bool ok    = true;
    for (int l = L->layer_lo; l < L->layer_hi && ok; l++) {
        for (int s = 0; s < QW_LORA_NSLOTS && ok; s++) {
            QwLoraPair & pr = L->layers[l].p[s];
            if (L->is_lokr) {
                if (!pr.has_lokr()) {
                    continue;
                }
                std::string site(lm_slot_peft_name(s));
                for (size_t j = 0; j < site.size(); j++) {
                    if (site[j] == '.') {
                        site[j] = '_';
                    }
                }
                char stem[128];
                snprintf(stem, sizeof(stem), "lycoris_layers_%d_%s", l, site.c_str());

                // The stored alpha must reproduce the scale this run will
                // apply — a mismatch means the file came from a different
                // dim/alpha config than the log claimed.
                const STEntry * ae = st_find(st, (std::string(stem) + ".alpha").c_str());
                if (ae) {
                    float av = 0.0f;
                    if (ae->dtype == "F32") {
                        memcpy(&av, st_data(st, *ae), sizeof(float));
                    } else if (ae->dtype == "BF16") {
                        uint16_t u16;
                        memcpy(&u16, st_data(st, *ae), sizeof(uint16_t));
                        lm_resume_bf16_to_f32(&u16, &av, 1);
                    }
                    const float want = pr.lokr_scale * (float) L->lokr_dim;
                    if (fabsf(av - want) > 0.01f * fmaxf(1.0f, fabsf(want))) {
                        char b[192];
                        snprintf(b, sizeof(b), "%s.alpha is %.4g but this config implies %.4g — wrong source file?",
                                 stem, (double) av, (double) want);
                        *err = b;
                        ok   = false;
                        break;
                    }
                }
                ok = ok && lm_resume_fill(st, (std::string(stem) + ".lokr_w1").c_str(), pr.w1, err);
                count++;
                if (ok && pr.w2) {
                    ok = lm_resume_fill(st, (std::string(stem) + ".lokr_w2").c_str(), pr.w2, err);
                    count++;
                } else if (ok) {
                    ok = lm_resume_fill(st, (std::string(stem) + ".lokr_w2_a").c_str(), pr.w2_a, err) &&
                         lm_resume_fill(st, (std::string(stem) + ".lokr_w2_b").c_str(), pr.w2_b, err);
                    count += 2;
                }
            } else {
                if (!pr.A || !pr.B) {
                    continue;
                }
                char nm[192];
                snprintf(nm, sizeof(nm), "base_model.model.model.layers.%d.%s.lora_A.weight", l,
                         lm_slot_peft_name(s));
                ok = lm_resume_fill(st, nm, pr.A, err);
                if (ok) {
                    snprintf(nm, sizeof(nm), "base_model.model.model.layers.%d.%s.lora_B.weight", l,
                             lm_slot_peft_name(s));
                    ok = lm_resume_fill(st, nm, pr.B, err);
                }
                count += 2;
            }
        }
    }
    st_close(&st);
    if (ok && n_loaded) {
        *n_loaded = count;
    }
    return ok;
}

// ─── CLI-side prepare: adopt-or-refuse ──────────────────────────────────────

/** Called by cmd_train_lm after flag parsing. Reads the source run's config,
 *  refuses explicit contradictions, adopts everything else. `errbuf` gets a
 *  ready-to-print message on false. */
template <typename ArgsT>
static bool lm_resume_prepare(ArgsT * a, const LmResumeExplicit & saw, LmResumeSource * src, std::string * errbuf) {
    if (!lm_resume_read_log(a->init_adapter, src, errbuf)) {
        return false;
    }

    struct Conflict {
        const char * flag;
        std::string  cli, source;
    };
    std::vector<Conflict> bad;
    auto check_i = [&](bool explicit_set, const char * flag, int cli, int source) {
        if (explicit_set && source > 0 && cli != source) {
            bad.push_back({ flag, std::to_string(cli), std::to_string(source) });
        }
    };
    if (saw.adapter_type && a->adapter_type != src->adapter_type) {
        bad.push_back({ "--adapter-type", a->adapter_type, src->adapter_type });
    }
    if (saw.weights && a->weights != src->weights) {
        bad.push_back({ "--weights", a->weights, src->weights });
    }
    check_i(saw.rank, "--rank", a->rank, src->rank);
    check_i(saw.alpha, "--alpha", a->alpha, src->alpha);
    check_i(saw.lokr_dim, "--lokr-dim", a->lokr_dim, src->lokr_dim);
    check_i(saw.lokr_factor, "--lokr-factor", a->lokr_factor, src->lokr_factor);
    if (saw.lokr_alpha && src->lokr_alpha > 0.0f && fabsf(a->lokr_alpha - src->lokr_alpha) > 1e-3f) {
        bad.push_back({ "--lokr-alpha", std::to_string(a->lokr_alpha), std::to_string(src->lokr_alpha) });
    }
    if (!bad.empty()) {
        std::string m = "--init-adapter refuses to continue " + a->init_adapter +
                        " with a different identity than it was trained with:\n";
        for (size_t j = 0; j < bad.size(); j++) {
            m += "  " + std::string(bad[j].flag) + " " + bad[j].cli + " but the source run used " + bad[j].source +
                 "\n";
        }
        m += "Drop the conflicting flag(s) — resume adopts the source run's identity.";
        *errbuf = m;
        return false;
    }

    // Adopt. Shape identity always comes from the source; sampling knobs
    // (lr, epochs, target, schedule) stay whatever this run asked for.
    a->adapter_type = src->adapter_type;
    if (src->rank > 0) {
        a->rank = src->rank;
    }
    if (src->alpha > 0) {
        a->alpha = src->alpha;
    }
    if (src->lokr_dim > 0) {
        a->lokr_dim = src->lokr_dim;
    }
    if (src->lokr_factor > 0) {
        a->lokr_factor = src->lokr_factor;
    }
    if (src->lokr_alpha > 0.0f) {
        a->lokr_alpha = src->lokr_alpha;
    }
    if (!src->weights.empty()) {
        a->weights = src->weights;
    }
    // Prodigy: carry the learned step size into this leg. Not an identity
    // check — a run that switched optimizer simply ignores it.
    if (a->optimizer == "prodigy" && !saw.prodigy_d0 && src->prodigy_d > 0.0) {
        a->prodigy_d0 = src->prodigy_d;
        fprintf(stderr, "[train-lm] resume: adopting the source run's Prodigy step size d = %.3g as --prodigy-d0\n",
                src->prodigy_d);
    }
    if (a->trigger.empty() && !src->trigger.empty()) {
        a->trigger          = src->trigger;
        a->trigger_position = src->trigger_position;
    }
    return true;
}
