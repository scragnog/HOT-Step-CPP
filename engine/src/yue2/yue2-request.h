#pragma once
// yue2/yue2-request.h — YuE2 synth request parsing + default resolution.
// Milestone M7, docs/plans/yue2/06-engine-port-plan.md §7.
//
// HOT-Step file (no acestep.cpp analog). Scope: JSON in, typed Yue2Request
// out. No GGML, no VRAM, no tokenizer call — cheap enough to run on a cold
// server (mirrors mm3-request.h's own "text work only" posture).
//
// NOT reused from mm3-request.h: its ~500 lines of MM3-Structured-Caption
// text hygiene (_clean_caption/_normalize_lyrics-equivalents) do not apply —
// YuE2's own prompt assembly (yue2-tokenizer.h's yue2_assemble_text) is a
// literal "[Tags]\n{style}\n[Lyrics]\n{lyrics}\n" block with no cleaning step
// at all (06-engine-port-plan.md §7). REUSED: the typed-JSON-field-reader
// shape (mm3_req_num/mm3_req_str's "fail loudly on a type mismatch, silently
// default when absent" contract) — copied here as yue2_req_num/yue2_req_str
// rather than shared, since mm3-request.h is minimax/-scoped and not meant
// to be included from engine/src/yue2/.
//
// Request text (style/lyrics/abc) is assumed ALREADY NFC-normalized by the
// caller (Node), per yue2-tokenizer.h's own file-header convention — this
// file does not re-normalize anything.

#include "yue2-model.h"     // Yue2VaeVariant, Yue2LmConfig
#include "yue2-tokenizer.h" // Yue2Cot, yue2_cot_from_name

#include "yyjson.h"

#include <cstdint>
#include <cmath>
#include <random>
#include <string>

enum Yue2NoiseSource { YUE2_NOISE_NATIVE = 0, YUE2_NOISE_FIXTURE = 1 };

struct Yue2Request {
    std::string id = "yue2";
    std::string style;
    std::string lyrics;
    Yue2Cot     cot = YUE2_COT_OFF;

    std::string abc;               // externally-supplied ABC text; skips the plan stage's model call
    bool        abc_provided = false;
    // Stop after the plan stage and return only the ABC score (no semantic/
    // NAR/VAE work, no audio). The score-preview flow: plan once, let the
    // user look at it, then re-submit the same request with `abc` set.
    bool        plan_only = false;

    uint64_t seed         = 0;
    bool     seed_present = false;
    // Training previews only: 0 retains the model's normal semantic limit.
    // This bounds work rather than trimming a full-song render afterward.
    int preview_max_frames = 0;

    float cfg_scale         = 0.0f;  // resolved by yue2_request_resolve_defaults if not present
    bool  cfg_scale_present = false;

    int         ode_steps = 0;  // 0 = use the checkpoint's own GGUF default (resolved below)
    std::string ode_method;     // "" = checkpoint default ("midpoint"); only "midpoint" is implemented

    // ── Ending controls, semantic stage only (2026-09-15) ────────────────
    // Measured on the adapter ladders (_LISTENING/2026-09-14/RESULTS.md, 77/83):
    // a trained AR reaches its ending and MUSIC_END then loses the draw to a
    // codec token, after which the stream drones to the frame cap. Two knobs,
    // both off by default so the stock model's behaviour is untouched:
    //   end_threshold  in (0,1]: stop when the sampler's own distribution
    //                  (after mask, penalty, temperature, top-k/top-p) puts at
    //                  least this much mass on MUSIC_END. 0 = off.
    //   end_bias       additive logit bias on MUSIC_END, applied from
    //                  end_bias_from_sec of generated audio and ramping
    //                  linearly to full strength over end_bias_ramp_sec. 0 = off.
    // Both respect min_tokens (END stays blocked before it).
    float end_threshold    = 0.0f;
    float end_bias         = 0.0f;
    float end_bias_from_sec = 0.0f;
    float end_bias_ramp_sec = 0.0f;

    Yue2VaeVariant vae_variant = YUE2_VAE_STANDARD;

    // Validator-only (docs/plans/yue2/06-engine-port-plan.md §7): absent from
    // any user-facing UI. "fixture" requires noise_fixture_path to hold a raw
    // little-endian FP32 buffer of exactly total_frames*latent_dim values,
    // whole-song, no per-chunk reseed — there is deliberately no RNG
    // fallback on a mismatch (fails the request outright).
    Yue2NoiseSource noise_source = YUE2_NOISE_NATIVE;
    std::string     noise_fixture_path;
};

static bool yue2_req_num(yyjson_val * root, const char * key, double * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_num(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a number";
        }
        return false;
    }
    *out     = yyjson_get_num(v);  // NOT yyjson_get_real -- a JSON int literal reads as 0 through that call
    *present = true;
    return true;
}

static bool yue2_req_str(yyjson_val * root, const char * key, std::string * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_str(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a string";
        }
        return false;
    }
    out->assign(yyjson_get_str(v), yyjson_get_len(v));
    *present = true;
    return true;
}

// Parses `body` (a JSON object) into `out`. Does NOT resolve mode-dependent
// defaults (cfg_scale/ode_steps/seed) — call yue2_request_resolve_defaults
// next, once the LM config is available (needs the GGUF's own ode_steps/
// ode_method defaults).
static bool yue2_parse_request(const std::string & body, Yue2Request * out, std::string * err) {
    yyjson_doc * doc = yyjson_read(body.data(), body.size(), 0);
    if (!doc) {
        if (err) {
            *err = "invalid JSON body";
        }
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        if (err) {
            *err = "request body must be a JSON object";
        }
        yyjson_doc_free(doc);
        return false;
    }

    bool        present = false;
    std::string s;

    if (!yue2_req_str(root, "id", &out->id, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }

    if (!yue2_req_str(root, "style", &out->style, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (!present || out->style.empty()) {
        if (err) {
            *err = "\"style\" is required and must be non-empty";
        }
        yyjson_doc_free(doc);
        return false;
    }

    if (!yue2_req_str(root, "lyrics", &out->lyrics, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    // lyrics may legitimately be empty/absent -- unlike MM3's own reference,
    // YuE2's protocol.py has no hard non-empty-lyrics requirement.

    if (!yue2_req_str(root, "cot", &s, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        if (!yue2_cot_from_name(s, &out->cot)) {
            if (err) {
                *err = "\"cot\" must be one of \"off\"|\"melody\"|\"full\"";
            }
            yyjson_doc_free(doc);
            return false;
        }
    }

    if (!yue2_req_str(root, "abc", &out->abc, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    out->abc_provided = present && !out->abc.empty();

    if (yyjson_val * v = yyjson_obj_get(root, "plan_only")) {
        if (!yyjson_is_bool(v) && !yyjson_is_null(v)) {
            if (err) {
                *err = "\"plan_only\" must be a boolean";
            }
            yyjson_doc_free(doc);
            return false;
        }
        out->plan_only = yyjson_is_bool(v) && yyjson_get_bool(v);
    }
    if (out->plan_only && out->cot == YUE2_COT_OFF) {
        if (err) {
            *err = "\"plan_only\" needs \"cot\" melody or full -- cot=off has no plan stage";
        }
        yyjson_doc_free(doc);
        return false;
    }
    if (out->plan_only && out->abc_provided) {
        if (err) {
            *err = "\"plan_only\" and \"abc\" are exclusive -- the score is already known";
        }
        yyjson_doc_free(doc);
        return false;
    }

    double num = 0.0;
    if (!yue2_req_num(root, "seed", &num, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        out->seed         = (uint64_t) num;
        out->seed_present = true;
    }

    if (!yue2_req_num(root, "cfg_scale", &num, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        out->cfg_scale         = (float) num;
        out->cfg_scale_present = true;
    }

    if (!yue2_req_num(root, "ode_steps", &num, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        out->ode_steps = (int) num;
    }

    if (!yue2_req_num(root, "preview_max_frames", &num, &present, err)) {
        yyjson_doc_free(doc); return false;
    }
    if (present) {
        if (!std::isfinite(num) || num < 0 || num > 9000 || std::floor(num) != num) {
            if (err) *err = "preview_max_frames must be an integer in [0,9000]";
            yyjson_doc_free(doc); return false;
        }
        out->preview_max_frames = static_cast<int>(num);
    }

    // Ending controls (see the struct). Range-checked here so a typo is a
    // 400, not a silent no-op or a stream that ends at frame 200.
    struct EndField {
        const char * key;
        float *      dst;
        double       lo, hi;
    };
    const EndField end_fields[] = {
        { "end_threshold", &out->end_threshold, 0.0, 1.0 },
        { "end_bias", &out->end_bias, -50.0, 50.0 },
        { "end_bias_from_sec", &out->end_bias_from_sec, 0.0, 3600.0 },
        { "end_bias_ramp_sec", &out->end_bias_ramp_sec, 0.0, 3600.0 },
    };
    for (const EndField & f : end_fields) {
        if (!yue2_req_num(root, f.key, &num, &present, err)) {
            yyjson_doc_free(doc);
            return false;
        }
        if (present) {
            if (!(num >= f.lo && num <= f.hi)) {
                if (err) {
                    *err = std::string(f.key) + " must be within [" + std::to_string(f.lo) + ", " +
                           std::to_string(f.hi) + "]";
                }
                yyjson_doc_free(doc);
                return false;
            }
            *f.dst = (float) num;
        }
    }

    if (!yue2_req_str(root, "ode_method", &out->ode_method, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present && out->ode_method != "midpoint") {
        if (err) {
            *err = "\"ode_method\" must be \"midpoint\" (the only method implemented)";
        }
        yyjson_doc_free(doc);
        return false;
    }

    if (!yue2_req_str(root, "vae_variant", &s, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        if (s == "standard") {
            out->vae_variant = YUE2_VAE_STANDARD;
        } else if (s == "legacy") {
            out->vae_variant = YUE2_VAE_LEGACY;
        } else {
            if (err) {
                *err = "\"vae_variant\" must be \"standard\"|\"legacy\"";
            }
            yyjson_doc_free(doc);
            return false;
        }
    }

    if (!yue2_req_str(root, "noise_source", &s, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (present) {
        if (s == "native") {
            out->noise_source = YUE2_NOISE_NATIVE;
        } else if (s == "fixture") {
            out->noise_source = YUE2_NOISE_FIXTURE;
        } else {
            if (err) {
                *err = "\"noise_source\" must be \"native\"|\"fixture\"";
            }
            yyjson_doc_free(doc);
            return false;
        }
    }
    if (!yue2_req_str(root, "noise_fixture_path", &out->noise_fixture_path, &present, err)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (out->noise_source == YUE2_NOISE_FIXTURE && out->noise_fixture_path.empty()) {
        if (err) {
            *err = "noise_source=\"fixture\" requires \"noise_fixture_path\"";
        }
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_doc_free(doc);
    return true;
}

// Mode-dependent defaults (protocol.py's own SongRequest.guidance property,
// docs/plans/yue2/06-engine-port-plan.md §7): 1.01 for cot=="off", 1.0
// otherwise, unless the request set cfg_scale explicitly. ode_steps/method
// fall back to the checkpoint's own GGUF-declared values. A missing seed
// gets a fresh random one so every request is reproducible after the fact
// (the resolved seed is not currently echoed back in the result — rough
// edge, see the milestone report).
static void yue2_request_resolve_defaults(Yue2Request * req, const Yue2LmConfig & lm_cfg) {
    if (!req->cfg_scale_present) {
        req->cfg_scale = (req->cot == YUE2_COT_OFF) ? 1.01f : 1.0f;
    }
    if (req->ode_steps <= 0) {
        req->ode_steps = (int) lm_cfg.ode_steps;
    }
    if (req->ode_method.empty()) {
        req->ode_method = lm_cfg.ode_method;
    }
    if (!req->seed_present) {
        std::random_device rd;
        req->seed         = (uint64_t) rd() | ((uint64_t) rd() << 32);
        req->seed_present = true;
    }
}
