#pragma once
// dit-resume.h — continue DiT adapter training from an exported run
// (`ace-train train-dit --init-adapter <dir>`), 2026-08-10.
//
// The DiT twin of lm-resume.h (read that header's rationale — same
// train-past-then-select strategy, same adopt-or-refuse identity rules, made
// real from the same "RESUME RULE" the log format documented). Differences
// that matter here:
//
//   - LAYER COVERAGE IS TOLERANT, not exact. A DiT run trains a top-K depth
//     window ([lora_lo, L)) that the VRAM auto-fit resolves at runtime, so a
//     resumed run's window can legitimately differ from the source's (other
//     card, other --layers). Sites covered by the file load; sites the file
//     lacks keep their fresh zero-delta init (they were never trained); file
//     tensors outside the new window are skipped. Both cases are counted and
//     logged. A resume that loads ZERO tensors is still refused.
//   - The epoch-1 honesty check is INFO-ONLY: a DiT epoch draws random
//     timesteps and crop windows, so its loss is stochastic and will not
//     reproduce the source's saved ma5 the way the LM's teacher-forced pass
//     reproduces saved_loss.
//   - `mirror` (f32|bf16|bf16-f32) and `bwd` are NOT identity: both are measured at
//     ~7e-5 loss drift (docs/TRAINING.md), so a resume may switch them.
//     target_mlp and the adapter shape params ARE identity — they change
//     which tensors exist.

#include "safetensors.h"
#include "train/dit-adapter-lokr.h"
#include "train/dit-adapter.h"
#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ─── source-run identity (from dit_train_log.json) ──────────────────────────

struct DitResumeSource {
    std::string dir;
    std::string adapter_type = "lora";
    int         rank = 0, alpha = 0;
    int         lokr_dim = 0, lokr_factor = 0;
    float       lokr_alpha = 0.0f;
    bool        target_mlp = true;
    int         layers = 0;                 // the --layers ARG the source ran with
    std::string dit_name;                   // base model filename
    double      saved_ma5  = -1.0;
    int         saved_epoch = 0;
    std::string trigger, trigger_position;
    // Prodigy's final step-size estimate from the source run (0 = not prodigy
    // or an older log); adopted as this run's --prodigy-d0 unless given.
    double      prodigy_d  = 0.0;
};

struct DitResumeExplicit {
    bool adapter_type = false, rank = false, alpha = false;
    bool lokr_dim = false, lokr_alpha = false, lokr_factor = false;
    bool target_mlp = false, layers = false;
    bool prodigy_d0 = false;   // not identity: only decides whether the source's d is adopted
};

static bool dit_resume_read_log(const std::string & dir, DitResumeSource * src, std::string * err) {
    const std::string log_path = dir + "/dit_train_log.json";
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
    src->dir          = dir;
    src->adapter_type = s("adapter_type").empty() ? "lora" : s("adapter_type");
    src->rank         = i("rank", 0);
    src->alpha        = i("alpha", 0);
    src->lokr_dim     = i("lokr_dim", 0);
    src->lokr_factor  = i("lokr_factor", 0);
    {
        yyjson_val * v  = yyjson_obj_get(cfg, "lokr_alpha");
        src->lokr_alpha = yyjson_is_num(v) ? (float) yyjson_get_num(v) : 0.0f;
    }
    {
        yyjson_val * v  = yyjson_obj_get(cfg, "target_mlp");
        src->target_mlp = yyjson_is_bool(v) ? yyjson_get_bool(v) : true;
    }
    src->layers           = i("layers", 0);
    src->dit_name         = s("dit_name");
    {
        yyjson_val * v = yyjson_obj_get(cfg, "prodigy_d");
        src->prodigy_d = yyjson_is_num(v) ? yyjson_get_num(v) : 0.0;
    }
    src->trigger          = s("trigger");
    src->trigger_position = s("trigger_position");
    {
        yyjson_val * v   = yyjson_obj_get(root, "saved_ma5");
        src->saved_ma5   = yyjson_is_num(v) ? yyjson_get_num(v) : -1.0;
        yyjson_val * e   = yyjson_obj_get(root, "saved_epoch");
        src->saved_epoch = yyjson_is_num(e) ? (int) yyjson_get_num(e) : 0;
    }
    yyjson_doc_free(doc);
    return true;
}

// ─── weight loading ─────────────────────────────────────────────────────────

static void dit_resume_bf16_to_f32(const uint16_t * in, float * out, size_t n) {
    for (size_t j = 0; j < n; j++) {
        const uint32_t u = ((uint32_t) in[j]) << 16;
        memcpy(&out[j], &u, sizeof(float));
    }
}

/** Fill one trainer tensor from the file if the entry exists.
 *  Returns 1 loaded, 0 absent, -1 shape/dtype error (err set). */
static int dit_resume_fill(const STFile & st, const std::string & name, ggml_tensor * dst, std::string * err) {
    const STEntry * e = st_find(st, name.c_str());
    if (!e) {
        return 0;
    }
    int64_t n = 1;
    for (int d = 0; d < e->n_dims; d++) {
        n *= e->shape[d];
    }
    if (n != ggml_nelements(dst)) {
        char b[224];
        snprintf(b, sizeof(b), "%s holds %lld values but the trainer tensor wants %lld — source shape mismatch",
                 name.c_str(), (long long) n, (long long) ggml_nelements(dst));
        *err = b;
        return -1;
    }
    const void * data = st_data(st, *e);
    if (e->dtype == "F32") {
        ggml_backend_tensor_set(dst, data, 0, (size_t) n * sizeof(float));
    } else if (e->dtype == "BF16") {
        std::vector<float> f((size_t) n);
        dit_resume_bf16_to_f32((const uint16_t *) data, f.data(), (size_t) n);
        ggml_backend_tensor_set(dst, f.data(), 0, f.size() * sizeof(float));
    } else {
        *err = name + " has unsupported dtype " + e->dtype;
        return -1;
    }
    return 1;
}

struct DitResumeStats {
    int loaded = 0;        // tensors filled from the file
    int fresh_sites = 0;   // trainer sites the file does not cover (stay zero-delta)
    int skipped_file = 0;  // file layer indices outside the trained window
};

static bool dit_resume_load_lora(DitAdapterLora * ad, const std::string & dir, DitResumeStats * out,
                                 std::string * err) {
    const std::string path = dir + "/adapter_model.safetensors";
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    bool ok = true;
    for (int l = ad->lo; l < ad->hi && ok; l++) {
        for (int s = 0; s < ad->n_sites && ok; s++) {
            DitLoraPair & pr = ad->layers[(size_t) (l - ad->lo)][(size_t) s];
            if (!pr.A || !pr.B) {
                continue;
            }
            char nm[208];
            if (ad->loha && pr.A2 && pr.B2) {
                // LoHa exports in the LyCORIS layout (dit-adapter.h exportDir):
                // w?_b is our A, w?_a our B. All four or none per site.
                const std::string stem = dit_lycoris_key_stem(l, s);
                ggml_tensor *     dst[4] = { pr.A, pr.B, pr.A2, pr.B2 };
                const char *      sfx[4] = { "hada_w1_b", "hada_w1_a", "hada_w2_b", "hada_w2_a" };
                int               got    = 0;
                for (int k = 0; k < 4 && ok; k++) {
                    const int r = dit_resume_fill(st, stem + "." + sfx[k], dst[k], err);
                    if (r < 0) {
                        ok = false;
                    } else {
                        got += r;
                    }
                }
                if (!ok) {
                    break;
                }
                if (got == 4) {
                    out->loaded += 4;
                } else if (got == 0) {
                    out->fresh_sites++;
                } else {
                    char b[192];
                    snprintf(b, sizeof(b), "layer %d site %s has %d of 4 LoHa factors in %s — corrupt export", l,
                             dit_site_peft(s), got, path.c_str());
                    *err = b;
                    ok   = false;
                }
                continue;
            }
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_A.weight", l, dit_site_peft(s));
            const int ra = dit_resume_fill(st, nm, pr.A, err);
            snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_B.weight", l, dit_site_peft(s));
            const int rb = ra < 0 ? -1 : dit_resume_fill(st, nm, pr.B, err);
            if (ra < 0 || rb < 0) {
                ok = false;
            } else if (ra == 1 && rb == 1) {
                out->loaded += 2;
                if (ad->dora && pr.m) {
                    // DoRA's magnitude. Absent from the file (a plain-LoRA source
                    // resumed as DoRA) it keeps its ||W||_col init, which makes the
                    // first step the plain-LoRA forward — the same start a fresh
                    // DoRA run has, so silence is correct there.
                    snprintf(nm, sizeof(nm), "base_model.model.layers.%d.%s.lora_magnitude_vector.weight", l,
                             dit_site_peft(s));
                    const int rm = dit_resume_fill(st, nm, pr.m, err);
                    if (rm < 0) {
                        ok = false;
                    } else {
                        out->loaded += rm;
                    }
                }
            } else if (ra == 0 && rb == 0) {
                out->fresh_sites++;
            } else {
                char b[192];
                snprintf(b, sizeof(b), "layer %d site %s has only half a LoRA pair in %s — corrupt export", l,
                         dit_site_peft(s), path.c_str());
                *err = b;
                ok   = false;
            }
        }
    }
    // File layers outside the trained window (counted for the log line).
    for (size_t i = 0; i < st.entries.size() && ok; i++) {
        int fl = -1;
        if (sscanf(st.entries[i].name.c_str(), "base_model.model.layers.%d.", &fl) == 1 ||
            sscanf(st.entries[i].name.c_str(), "lycoris_layers_%d_", &fl) == 1) {
            if (fl < ad->lo || fl >= ad->hi) {
                out->skipped_file++;
            }
        }
    }
    st_close(&st);
    return ok;
}

static bool dit_resume_load_lokr(DitAdapterLoKr * ad, const std::string & dir, DitResumeStats * out,
                                 std::string * err) {
    const std::string path = dir + "/lokr_weights.safetensors";
    STFile st;
    if (!st_open(&st, path.c_str())) {
        *err = "cannot open " + path;
        return false;
    }
    bool ok = true;
    for (int l = ad->lo; l < ad->hi && ok; l++) {
        for (int s = 0; s < ad->n_sites && ok; s++) {
            DitLokrSite & k = ad->layers[(size_t) (l - ad->lo)][(size_t) s];
            if (!k.w1) {
                continue;
            }
            const std::string stem = DitAdapterLoKr::lokr_key_stem(l, s);
            const int r1 = dit_resume_fill(st, stem + ".lokr_w1", k.w1, err);
            if (r1 < 0) { ok = false; break; }
            if (r1 == 0) { out->fresh_sites++; continue; }

            // The stored alpha must reproduce what this run will apply.
            const STEntry * ae = st_find(st, (stem + ".alpha").c_str());
            if (ae) {
                float av = 0.0f;
                if (ae->dtype == "F32") {
                    memcpy(&av, st_data(st, *ae), sizeof(float));
                } else if (ae->dtype == "BF16") {
                    uint16_t u16;
                    memcpy(&u16, st_data(st, *ae), sizeof(uint16_t));
                    dit_resume_bf16_to_f32(&u16, &av, 1);
                }
                if (fabsf(av - k.alpha_eff) > 0.01f * fmaxf(1.0f, fabsf(k.alpha_eff))) {
                    char b[192];
                    snprintf(b, sizeof(b), "%s.alpha is %.4g but this config implies %.4g — wrong source file?",
                             stem.c_str(), (double) av, (double) k.alpha_eff);
                    *err = b;
                    ok   = false;
                    break;
                }
            }
            out->loaded++;
            if (k.mono) {
                const int r2 = dit_resume_fill(st, stem + ".lokr_w2", k.w2, err);
                if (r2 <= 0) { *err = r2 == 0 ? stem + ".lokr_w2 missing (mono/factorized mismatch)" : *err; ok = false; }
                else out->loaded++;
            } else {
                const int r2a = dit_resume_fill(st, stem + ".lokr_w2_a", k.w2_a, err);
                const int r2b = r2a <= 0 ? 0 : dit_resume_fill(st, stem + ".lokr_w2_b", k.w2_b, err);
                if (r2a <= 0 || r2b <= 0) {
                    if (r2a == 0 || r2b == 0) *err = stem + ".lokr_w2_a/b missing (mono/factorized mismatch)";
                    ok = false;
                } else out->loaded += 2;
            }
        }
    }
    for (size_t i = 0; i < st.entries.size() && ok; i++) {
        int fl = -1;
        if (sscanf(st.entries[i].name.c_str(), "lycoris_layers_%d_", &fl) == 1) {
            if (fl < ad->lo || fl >= ad->hi) {
                out->skipped_file++;
            }
        }
    }
    st_close(&st);
    return ok;
}

// ─── CLI-side prepare: adopt-or-refuse ──────────────────────────────────────

template <typename ArgsT>
static bool dit_resume_prepare(ArgsT * a, const DitResumeExplicit & saw, DitResumeSource * src,
                               std::string * errbuf) {
    if (!dit_resume_read_log(a->init_adapter, src, errbuf)) {
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
    check_i(saw.rank, "--rank", a->rank, src->rank);
    check_i(saw.alpha, "--alpha", a->alpha, src->alpha);
    check_i(saw.lokr_dim, "--lokr-dim", a->lokr_dim, src->lokr_dim);
    check_i(saw.lokr_factor, "--lokr-factor", a->lokr_factor, src->lokr_factor);
    if (saw.lokr_alpha && src->lokr_alpha > 0.0f && fabsf(a->lokr_alpha - src->lokr_alpha) > 1e-3f) {
        bad.push_back({ "--lokr-alpha", std::to_string(a->lokr_alpha), std::to_string(src->lokr_alpha) });
    }
    if (saw.target_mlp && a->target_mlp != src->target_mlp) {
        bad.push_back({ "--no-target-mlp", a->target_mlp ? "mlp on" : "mlp off",
                        src->target_mlp ? "mlp on" : "mlp off" });
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

    a->adapter_type = src->adapter_type;
    if (src->rank > 0) a->rank = src->rank;
    if (src->alpha > 0) a->alpha = src->alpha;
    if (src->lokr_dim > 0) a->lokr_dim = src->lokr_dim;
    if (src->lokr_factor > 0) a->lokr_factor = src->lokr_factor;
    if (src->lokr_alpha > 0.0f) a->lokr_alpha = src->lokr_alpha;
    a->target_mlp = src->target_mlp;
    // Layer window: adopt the source's --layers ARG unless the user pinned one.
    // The loader is coverage-tolerant either way (see header).
    if (!saw.layers && src->layers > 0) a->layers = src->layers;
    // Prodigy: carry the learned step size into the resumed run. Not an
    // identity check — a run that switched optimizer simply ignores it.
    if (a->optimizer == "prodigy" && !saw.prodigy_d0 && src->prodigy_d > 0.0) {
        a->prodigy_d0 = src->prodigy_d;
        fprintf(stderr, "[train-dit] resume: adopting the source run's Prodigy step size d = %.3g as --prodigy-d0\n",
                src->prodigy_d);
    }
    if (a->trigger.empty() && !src->trigger.empty()) {
        a->trigger          = src->trigger;
        a->trigger_position = src->trigger_position;
    }
    return true;
}
