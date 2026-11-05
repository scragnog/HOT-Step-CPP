#pragma once
// heartmula/pipeline.h: HeartMuLa text to music, the component catalogue and VRAM admission.
//
// HOT-Step file. Implements engine/docs/HEARTMULA_CONTRACT.md for one process, with no
// HTTP: heartmula/server.h wraps it for hot-step-server, tools/heartmula-generate.cpp
// for the CLI and the smoke test.
//
// A. CATALOGUE. Discovery reads GGUF headers only. Identity comes from the metadata;
//    a file whose name disagrees is listed with an error. Load-time checks 1 to 4 run
//    here (architecture/version, pair, tokenizer and RVQ dependencies); 6 runs in
//    the loaders (model.h shapes, rvq.h config).
//
// B. TWO PHASES ON ONE DEVICE. The LM phase holds tokenizer, global, local and the KV
//    cache; the codec phase holds RVQ, flow and scalar decoder. Without keep-loaded
//    each phase frees the other before it loads, and everything is freed at the end,
//    after a failure and after a cancel too. The KV cache never outlives a request.
//
// C. ADMISSION BEFORE EACH PHASE. need = bytes about to load + KV + reserve; free =
//    device free now + what the caller can reclaim from idle backends. need > free
//    refuses before anything moves; need > device free alone reclaims first.
//
// D. CANCEL AND PROGRESS AT EVERY PHASE: between components, between prefill chunks
//    (at most HEARTMULA_PREFILL_CHUNK positions), after every frame, between flow
//    steps and scalar chunks (heartcodec::Cancelled).
//
// E. SEEDS. The LM draws come from HeartMuLaDraws(seed); the codec noise is Box-Muller
//    on raw mt19937_64 words, so a noise seed gives the same noise on every platform.

#include "heartmula/codec-decode.h"
#include "heartmula/local-graph.h"
#include "heartmula/tokenizer.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <random>

#define HEARTMULA_FRAME_RATE        12.5
#define HEARTMULA_MAX_DURATION_S    240
#define HEARTMULA_AUDIO_EOS         8193
#define HEARTMULA_DEFAULT_RESERVE_MB 1536

// ── Catalogue (note A) ──────────────────────────────────────────────────────

enum HeartMuLaRole {
    HM_GLOBAL, HM_LOCAL, HM_FLOW, HM_SCALAR, HM_RVQ, HM_TOKENIZER,
    HM_SCALAR_ENC, HM_MUENC, HM_WAVLM, HM_WHISPER, HM_QUERY, HM_ROLE_COUNT
};
static const char * const k_heartmula_roles[HM_ROLE_COUNT] = {
    "global", "local", "flow", "scalar-decoder", "rvq", "tokenizer",
    "scalar-encoder", "muencoder", "wavlm", "whisper", "query"
};

struct HeartMuLaFile {
    int         role = -1;
    std::string type;      // quant token from the name; "" for the tokenizer
    std::string filename, path, variant, fingerprint, error;
    uint64_t    bytes     = 0;  // on disk
    uint64_t    gpu_bytes = 0;  // tensor bytes once loaded on a GPU
    uint32_t    file_type = 0;
    std::string tokenizer_json;  // LM: dependencies.tokenizer; tokenizer: its source_sha256
    std::string rvq_sha256;      // dependency, or the RVQ file's own hash
    bool        compatible = false;
};

struct HeartMuLaCatalogue {
    std::vector<HeartMuLaFile> files;
    std::vector<std::string>   errors;
};

namespace heartmula_pipeline_detail {

inline std::string lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

inline std::string kv_str(gguf_context * g, const char * key) {
    const int64_t i = gguf_find_key(g, key);
    return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_STRING ? gguf_get_val_str(g, i) : "";
}

inline int64_t kv_u32(gguf_context * g, const char * key) {
    const int64_t i = gguf_find_key(g, key);
    return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_UINT32 ? (int64_t) gguf_get_val_u32(g, i) : -1;
}

inline bool json_equal(const std::string & a, const std::string & b) {
    yyjson_doc * da = yyjson_read(a.data(), a.size(), 0);
    yyjson_doc * db = yyjson_read(b.data(), b.size(), 0);
    const bool   eq = da && db && yyjson_equals(yyjson_doc_get_root(da), yyjson_doc_get_root(db));
    yyjson_doc_free(da);
    yyjson_doc_free(db);
    return eq;
}

// One header. Fills identity, dependency and size fields, or `error`.
inline HeartMuLaFile read_header(const std::filesystem::path & p) {
    HeartMuLaFile f;
    f.path     = p.string();
    f.filename = p.filename().string();
    std::error_code ec;
    f.bytes = (uint64_t) std::filesystem::file_size(p, ec);
    ggml_context *   meta = nullptr;
    gguf_context *   g    = gguf_init_from_file(f.path.c_str(), { true, &meta });
    if (!g) {
        f.error = f.filename + ": not a readable GGUF";
        return f;
    }
    const std::string component = kv_str(g, "heartmula.component");
    for (int r = 0; r < HM_ROLE_COUNT; r++) {
        if (component == k_heartmula_roles[r]) f.role = r;
    }
    const std::string arch = kv_str(g, "general.architecture");
    f.variant              = kv_str(g, "heartmula.variant");
    f.fingerprint          = kv_str(g, "heartmula.base_fingerprint");
    f.file_type            = (uint32_t) std::max<int64_t>(0, kv_u32(g, "general.file_type"));
    const std::string stem = f.filename.size() > 5 ? f.filename.substr(0, f.filename.size() - 5) : f.filename;
    f.type                 = f.role == HM_TOKENIZER ? "" : stem.substr(stem.rfind('-') + 1);
    std::string want_arch, want_name;
    if (f.role == HM_GLOBAL || f.role == HM_LOCAL) {
        want_arch = std::string("heartmula-") + component;
        want_name = "heartmula-" + lower(f.variant) + "-" + component + "-" + f.type + ".gguf";
    } else if (f.role == HM_TOKENIZER) {
        want_arch = want_name = "heartmula-tokenizer";
        want_name += ".gguf";
    } else if (f.role >= 0) {
        want_arch = std::string("heartcodec-") + component;
        want_name = want_arch + "-" + f.type + ".gguf";
    }
    if (component == "lora") {
        f.role = -2;
    } else if (f.role < 0) {
        f.error = f.filename + ": heartmula.component '" + component + "' is not a known role";
    } else if (arch != want_arch) {
        f.error = f.filename + ": general.architecture is '" + arch + "', expected '" + want_arch + "'";
    } else if (kv_u32(g, "heartmula.format_version") != HEARTMULA_FORMAT_VERSION) {
        f.error = f.filename + ": unsupported heartmula.format_version";
    } else if (f.filename != want_name) {
        f.error = f.filename + ": name disagrees with its metadata (expected " + want_name + ")";
    }
    if (f.role == HM_TOKENIZER) {
        f.tokenizer_json = kv_str(g, "heartmula.tokenizer.source_sha256");
    } else if (f.role >= 0) {
        const std::string deps = kv_str(g, "heartmula.dependencies");
        if (yyjson_doc * d = yyjson_read(deps.data(), deps.size(), 0)) {
            yyjson_val * root = yyjson_doc_get_root(d);
            yyjson_val * tok  = yyjson_obj_get(root, "tokenizer");
            yyjson_val * rvq  = yyjson_obj_get(root, "rvq_sha256");
            if (tok) {
                char * s = yyjson_val_write(tok, 0, nullptr);
                f.tokenizer_json = s ? s : "";
                free(s);
            }
            if (yyjson_is_str(rvq)) f.rvq_sha256 = yyjson_get_str(rvq);
            yyjson_doc_free(d);
        }
    }
    const std::string role = f.role >= 0 ? k_heartmula_roles[f.role] : "";
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        const bool codec = f.role == HM_FLOW || f.role == HM_SCALAR || f.role == HM_RVQ;
        // rvq.h widens F16 to F32 except linear matrices on a GPU.
        f.gpu_bytes += codec && t->type == GGML_TYPE_F16 && !heartcodec::keep_native(role, t, true)
                           ? (uint64_t) ggml_nelements(t) * 4
                           : (uint64_t) ggml_nbytes(t);
    }
    if (f.role == HM_RVQ && f.error.empty()) {
        yue2_aitk::sha256::digest digest;
        if (yue2_aitk::sha256::file(f.path, digest)) f.rvq_sha256 = digest.hex();
        else f.error = f.filename + ": cannot hash the codebook";
    }
    gguf_free(g);
    ggml_free(meta);
    return f;
}

}  // namespace heartmula_pipeline_detail

// Scan <models>/heartmula/ then <models>/ (first filename wins) and run checks 1, 3, 4.
static HeartMuLaCatalogue heartmula_discover(const std::string & models_dir) {
    using namespace heartmula_pipeline_detail;
    namespace fs = std::filesystem;
    HeartMuLaCatalogue cat;
    std::vector<std::string> seen;
    for (const fs::path dir : { fs::path(models_dir) / "heartmula", fs::path(models_dir) }) {
        std::error_code ec;
        std::vector<fs::path> paths;
        for (const auto & e : fs::directory_iterator(dir, ec)) {
            const std::string n = e.path().filename().string();
            const bool ours = n.rfind("heartmula-", 0) == 0 || n.rfind("heartcodec-", 0) == 0;
            if (e.is_regular_file() && ours && e.path().extension() == ".gguf") paths.push_back(e.path());
        }
        std::sort(paths.begin(), paths.end());
        for (const auto & p : paths) {
            if (std::find(seen.begin(), seen.end(), p.filename().string()) != seen.end()) continue;
            seen.push_back(p.filename().string());
            HeartMuLaFile f = read_header(p);
            if (f.role != -2) cat.files.push_back(std::move(f));  // -2: an adapter, listed by the Node server
        }
    }
    const HeartMuLaFile * rvq = nullptr, * tok = nullptr;
    for (const auto & f : cat.files) {
        if (f.error.empty() && f.role == HM_RVQ && !rvq) rvq = &f;
        if (f.error.empty() && f.role == HM_TOKENIZER && !tok) tok = &f;
    }
    for (auto & f : cat.files) {
        if (!f.error.empty()) {
            cat.errors.push_back(f.error);
            continue;
        }
        const bool lm = f.role == HM_GLOBAL || f.role == HM_LOCAL;
        if (f.role != HM_RVQ && f.role != HM_TOKENIZER) {
            if (!rvq) {
                f.error = f.filename + ": no compatible heartcodec-rvq file is present";
            } else if (f.rvq_sha256 != rvq->rvq_sha256) {
                f.error = f.filename + ": rvq_sha256 differs from " + rvq->filename;
            }
        }
        if (f.error.empty() && lm) {
            if (!tok) {
                f.error = f.filename + ": no compatible heartmula-tokenizer.gguf is present";
            } else if (!json_equal(f.tokenizer_json, tok->tokenizer_json)) {
                f.error = f.filename + ": dependencies.tokenizer differs from " + tok->filename;
            }
        }
        f.compatible = f.error.empty();
        if (!f.compatible) cat.errors.push_back(f.error);
    }
    return cat;
}

// ── Pick ────────────────────────────────────────────────────────────────────

// What select-model or a job asks for. "" = the default for that field.
struct HeartMuLaPick {
    std::string variant, global, local, flow, scalar, adapter;
    float       adapter_scale = 1.0f;
};

struct HeartMuLaResolved {
    std::string   variant;
    HeartMuLaFile file[HM_TOKENIZER + 1];  // global, local, flow, scalar, rvq, tokenizer
    std::string   adapter;                  // "" = none
    std::string   adapter_home;             // re-base: the adapter's home global, "" = trained on this base
    float         adapter_scale = 1.0f;
    uint64_t      adapter_bytes = 0;        // factors, plus the home weights when re-basing
};

namespace heartmula_pipeline_detail {

// Load-time check 5 from the header; shapes against the base are adapter.h's.
// An adapter trained on another variant attaches through basin re-base (adapter.h) when
// its home global is installed: same fingerprint and variant as the adapter. Prefers the
// loaded global's quant, then F16, then any; `home` gets its path and `bytes` also
// counts the home weights of the targeted projections.
inline bool check_adapter(const std::string & path, const HeartMuLaFile & global, const HeartMuLaCatalogue & cat,
                          uint64_t * bytes, std::string * home, int * status, std::string * err) {
    std::error_code ec;
    *bytes = (uint64_t) std::filesystem::file_size(path, ec);
    ggml_context * meta = nullptr;
    gguf_context * g    = ec ? nullptr : gguf_init_from_file(path.c_str(), { true, &meta });
    if (!g) {
        *status = 404;
        *err    = path + ": no readable adapter GGUF";
        return false;
    }
    const std::string name = std::filesystem::path(path).filename().string();
    *status                = 409;
    if (kv_str(g, "heartmula.component") != "lora" || kv_str(g, "heartmula.lora.target_component") != "global") {
        *err = name + ": not a HeartMuLa LoRA for the global LM";
    } else if (kv_str(g, "heartmula.base_fingerprint") != global.fingerprint ||
               lower(kv_str(g, "heartmula.variant")) != lower(global.variant)) {
        const std::string fp = kv_str(g, "heartmula.base_fingerprint"), variant = kv_str(g, "heartmula.variant");
        const auto home_of = [&](const HeartMuLaFile & f) {
            return f.role == HM_GLOBAL && f.compatible && f.fingerprint == fp && lower(f.variant) == lower(variant);
        };
        const HeartMuLaFile * hit = nullptr;
        for (const std::string & pref : { global.type, std::string("F16"), std::string() }) {
            for (const auto & f : cat.files) {
                if (!hit && home_of(f) && (pref.empty() || f.type == pref)) hit = &f;
            }
        }
        std::vector<std::string> targets;
        if (!hit) {
            *err = name + ": trained on the " + variant + " model; install a " + variant +
                   " global model to re-base it onto " + global.variant;
        } else if (!heartmula_adapter_detail::parse_targets(kv_str(g, "heartmula.lora.targets"), &targets)) {
            *err = name + ": heartmula.lora.targets is missing or not a JSON array of names";
        } else {
            ggml_context * hmeta = nullptr;
            gguf_context * hg    = gguf_init_from_file(hit->path.c_str(), { true, &hmeta });
            if (!hg) {
                *err = hit->filename + ": cannot read the home model";
            } else {
                for (const auto & t : targets) {
                    if (ggml_tensor * w = ggml_get_tensor(hmeta, t.c_str())) *bytes += ggml_nbytes(w);
                }
                *home = hit->path;
                gguf_free(hg);
                ggml_free(hmeta);
            }
        }
    }
    gguf_free(g);
    ggml_free(meta);
    return err->empty();
}

}  // namespace heartmula_pipeline_detail

// Resolve a pick against the catalogue. On failure *status is 404 (no such variant,
// type or adapter file) or 409 (exists but fails a check).
static bool heartmula_resolve(const HeartMuLaCatalogue & cat, const HeartMuLaPick & pick, HeartMuLaResolved * out,
                              int * status, std::string * err) {
    using heartmula_pipeline_detail::lower;
    auto fail = [&](int s, const std::string & m) {
        *status = s;
        *err    = m;
        return false;
    };
    out->variant = pick.variant.empty() ? "3B" : pick.variant;
    bool variant_found = false;
    for (const auto & f : cat.files) {
        if ((f.role == HM_GLOBAL || f.role == HM_LOCAL) && lower(f.variant) == lower(out->variant)) {
            variant_found = true;
            out->variant  = f.variant;
        }
    }
    if (!variant_found) return fail(404, "no HeartMuLa global/local files for variant '" + out->variant + "'");
    const std::string * want[HM_TOKENIZER + 1] = { &pick.global, &pick.local, &pick.flow, &pick.scalar, nullptr, nullptr };
    for (int r = 0; r <= HM_TOKENIZER; r++) {
        std::vector<const HeartMuLaFile *> c;
        for (const auto & f : cat.files) {
            const bool lm = r == HM_GLOBAL || r == HM_LOCAL;
            if (f.role == r && (!lm || lower(f.variant) == lower(out->variant))) c.push_back(&f);
        }
        const HeartMuLaFile * hit = nullptr;
        if (want[r] && !want[r]->empty()) {
            for (auto * f : c) {
                if (lower(f->type) == lower(*want[r])) hit = f;
            }
            if (!hit) return fail(404, std::string("no ") + k_heartmula_roles[r] + " file of type '" + *want[r] + "'");
            if (!hit->compatible) return fail(409, hit->error);
        } else {
            for (const char * pref : { "Q8_0", "F16" }) {
                for (auto * f : c) {
                    if (!hit && f->compatible && f->type == pref) hit = f;
                }
            }
            if (!hit) {
                for (auto * f : c) {
                    if (f->compatible && (!hit || f->bytes > hit->bytes)) hit = f;  // the most precise
                }
            }
            if (!hit) {
                for (auto * f : c) {
                    if (!f->error.empty()) return fail(409, f->error);
                }
                return fail(404, std::string("no ") + k_heartmula_roles[r] + " file is present");
            }
        }
        out->file[r] = *hit;
    }
    const HeartMuLaFile & g = out->file[HM_GLOBAL], & l = out->file[HM_LOCAL];
    if (g.fingerprint != l.fingerprint) return fail(409, l.filename + ": base_fingerprint differs from " + g.filename);
    const HeartMuLaFile & fl = out->file[HM_FLOW], & sc = out->file[HM_SCALAR];
    if (fl.fingerprint != sc.fingerprint) return fail(409, sc.filename + ": base_fingerprint differs from " + fl.filename);
    out->adapter       = pick.adapter;
    out->adapter_scale = pick.adapter_scale;
    out->adapter_bytes = 0;
    out->adapter_home.clear();
    err->clear();
    return pick.adapter.empty() || heartmula_pipeline_detail::check_adapter(pick.adapter, g, cat, &out->adapter_bytes,
                                                                            &out->adapter_home, status, err);
}

// ── Request ─────────────────────────────────────────────────────────────────

struct HeartMuLaRequest {
    std::string tags, lyrics, id = "heartmula";
    double      duration_s     = 120.0;
    uint64_t    seed           = 0, noise_seed = 0;
    bool        seed_given     = false, noise_seed_given = false;
    float       temperature    = 1.0f;
    int         topk           = 50;
    float       cfg_scale      = 1.5f;
    int         codec_steps    = 10;
    float       codec_guidance = 1.25f;
    std::vector<std::array<int32_t, HEARTMULA_CODEBOOKS>> codes;  // given: skip the LM
    bool        codes_only = false;
    // The job's own pick: absent keeps the current one, "" or null resets to default.
    std::optional<std::string> variant, global, local, flow, scalar, adapter;
    std::optional<float>       adapter_scale;
    HeartMuLaAdapterScales     adapter_groups;  // attn/mlp/early/mid/late; global comes from the pick
    // Sampler plugins for the HeartCodec flow decode (codec-plugins.h), ACE's field names.
    // All empty = the native Euler loop.
    std::string                        infer_method, scheduler, guidance_mode;
    float                              flow_shift = 1.0f, apg_norm_threshold = 2.5f;
    std::map<std::string, std::string> plugin_params;  // "<plugin>:<key>" -> value
};

namespace heartmula_pipeline_detail {

inline bool get_u64(yyjson_val * v, uint64_t * out) {
    if (yyjson_is_uint(v)) { *out = yyjson_get_uint(v); return true; }
    if (yyjson_is_sint(v)) { if (yyjson_get_sint(v) < 0) return false; *out = (uint64_t) yyjson_get_sint(v); return true; }
    if (yyjson_is_real(v)) {
        const double d = yyjson_get_real(v);
        if (d < 0 || d > 9007199254740992.0 || d != std::floor(d)) return false;
        *out = (uint64_t) d;
        return true;
    }
    if (yyjson_is_str(v)) {
        const std::string s = yyjson_get_str(v);
        if (s.empty() || s.size() > 20 || s.find_first_not_of("0123456789") != std::string::npos) return false;
        errno       = 0;
        char * end  = nullptr;
        *out        = std::strtoull(s.c_str(), &end, 10);
        return errno == 0 && *end == 0;
    }
    return false;
}

inline bool get_num(yyjson_val * v, double lo, double hi, bool integer, double * out) {
    if (!yyjson_is_num(v)) return false;
    *out = yyjson_get_num(v);
    return std::isfinite(*out) && *out >= lo && *out <= hi && (!integer || *out == std::floor(*out));
}

}  // namespace heartmula_pipeline_detail

// Parse a /heartmula/synth body (also the CLI's --request). On failure *err names the field.
static bool heartmula_parse_request(const std::string & body, HeartMuLaRequest * r, std::string * err) {
    using namespace heartmula_pipeline_detail;
    yyjson_doc * doc = yyjson_read(body.data(), body.size(), 0);
    struct Free { yyjson_doc * d; ~Free() { yyjson_doc_free(d); } } guard{ doc };
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!yyjson_is_obj(root)) { *err = "body must be a JSON object"; return false; }
    auto bad = [&](const std::string & m) { *err = m; return false; };
    size_t idx, max;
    yyjson_val * key, * v;
    bool has_tags = false;
    yyjson_obj_foreach(root, idx, max, key, v) {
        const std::string k = yyjson_get_str(key);
        double d = 0;
        auto pick_str = [&](std::optional<std::string> & o) {
            if (yyjson_is_null(v)) { o = std::string(); return true; }
            if (!yyjson_is_str(v)) return false;
            o = std::string(yyjson_get_str(v));
            return true;
        };
        if (k == "tags") {
            if (!yyjson_is_str(v) || !yyjson_get_len(v) || yyjson_get_len(v) > 4096) return bad("tags must be a non-empty string of at most 4096 bytes");
            r->tags  = std::string(yyjson_get_str(v), yyjson_get_len(v));
            has_tags = true;
        } else if (k == "lyrics") {
            if (!yyjson_is_str(v) || yyjson_get_len(v) > 32768) return bad("lyrics must be a string of at most 32768 bytes");
            r->lyrics = std::string(yyjson_get_str(v), yyjson_get_len(v));
        } else if (k == "duration_s") {
            if (!get_num(v, 1, HEARTMULA_MAX_DURATION_S, false, &d)) return bad("duration_s must be a number from 1 to 240");
            r->duration_s = d;
        } else if (k == "seed") {
            if (!get_u64(v, &r->seed)) return bad("seed must be a non-negative integer or a decimal string");
            r->seed_given = true;
        } else if (k == "noise_seed") {
            if (!get_u64(v, &r->noise_seed)) return bad("noise_seed must be a non-negative integer or a decimal string");
            r->noise_seed_given = true;
        } else if (k == "temperature") {
            if (!get_num(v, 0, 4, false, &d) || d <= 0) return bad("temperature must be greater than 0 and at most 4");
            r->temperature = (float) d;
        } else if (k == "topk") {
            if (!get_num(v, 1, HEARTMULA_AUDIO_VOCAB, true, &d)) return bad("topk must be an integer from 1 to 8197");
            r->topk = (int) d;
        } else if (k == "cfg_scale") {
            if (!get_num(v, 1, 10, false, &d)) return bad("cfg_scale must be a number from 1 to 10");
            r->cfg_scale = (float) d;
        } else if (k == "codec_steps") {
            if (!get_num(v, 1, 100, true, &d)) return bad("codec_steps must be an integer from 1 to 100");
            r->codec_steps = (int) d;
        } else if (k == "codec_guidance") {
            if (!get_num(v, 1, 10, false, &d)) return bad("codec_guidance must be a number from 1 to 10");
            r->codec_guidance = (float) d;
        } else if (k == "variant") {
            if (!pick_str(r->variant)) return bad("variant must be a string");
        } else if (k == "global_type") {
            if (!pick_str(r->global)) return bad("global_type must be a string");
        } else if (k == "local_type") {
            if (!pick_str(r->local)) return bad("local_type must be a string");
        } else if (k == "flow_type") {
            if (!pick_str(r->flow)) return bad("flow_type must be a string");
        } else if (k == "scalar_decoder_type") {
            if (!pick_str(r->scalar)) return bad("scalar_decoder_type must be a string");
        } else if (k == "adapter") {
            if (!pick_str(r->adapter)) return bad("adapter must be a string or null");
        } else if (k == "adapter_scale") {
            if (!get_num(v, 0, 2, false, &d)) return bad("adapter_scale must be a number from 0 to 2");
            r->adapter_scale = (float) d;
        } else if (k == "adapter_rebase") {
            if (!get_num(v, 0, 1, false, &d)) return bad("adapter_rebase must be a number from 0 to 1");
            r->adapter_groups.rebase = (float) d;
        } else if (k == "adapter_scale_attn" || k == "adapter_scale_mlp" || k == "adapter_scale_early" ||
                   k == "adapter_scale_mid" || k == "adapter_scale_late") {
            if (!get_num(v, 0, 2, false, &d)) return bad(k + " must be a number from 0 to 2");
            HeartMuLaAdapterScales & g = r->adapter_groups;
            (k == "adapter_scale_attn" ? g.attn : k == "adapter_scale_mlp" ? g.mlp : k == "adapter_scale_early" ? g.early
             : k == "adapter_scale_mid" ? g.mid : g.late) = (float) d;
        } else if (k == "infer_method" || k == "scheduler" || k == "guidance_mode") {
            if (!yyjson_is_str(v) && !yyjson_is_null(v)) return bad(k + " must be a string");
            (k == "infer_method" ? r->infer_method : k == "scheduler" ? r->scheduler : r->guidance_mode) =
                yyjson_is_str(v) ? yyjson_get_str(v) : "";
        } else if (k == "flow_shift") {
            if (!get_num(v, 0.01, 20, false, &d)) return bad("flow_shift must be a number from 0.01 to 20");
            r->flow_shift = (float) d;
        } else if (k == "apg_norm_threshold") {
            if (!get_num(v, 0, 100, false, &d)) return bad("apg_norm_threshold must be a number from 0 to 100");
            r->apg_norm_threshold = (float) d;
        } else if (k == "plugin_params") {
            if (yyjson_is_null(v)) continue;
            if (!yyjson_is_obj(v)) return bad("plugin_params must be an object");
            // The coercion mm3-request.h and hot-step-server.cpp use, so one control means one thing.
            size_t       pi, pmax;
            yyjson_val * pk, * pv;
            yyjson_obj_foreach(v, pi, pmax, pk, pv) {
                std::string vs;
                if (yyjson_is_str(pv)) vs = yyjson_get_str(pv);
                else if (yyjson_is_real(pv)) vs = std::to_string(yyjson_get_real(pv));
                else if (yyjson_is_int(pv)) vs = std::to_string(yyjson_get_int(pv));
                else if (yyjson_is_bool(pv)) vs = yyjson_get_bool(pv) ? "true" : "false";
                r->plugin_params[yyjson_get_str(pk)] = vs;
            }
        } else if (k == "codes") {
            if (!yyjson_is_arr(v) || yyjson_arr_size(v) < 1 || yyjson_arr_size(v) > 3000) return bad("codes must be an array of 1 to 3000 frames");
            size_t fi, fmax;
            yyjson_val * frame;
            yyjson_arr_foreach(v, fi, fmax, frame) {
                std::array<int32_t, HEARTMULA_CODEBOOKS> c{};
                if (!yyjson_is_arr(frame) || yyjson_arr_size(frame) != HEARTMULA_CODEBOOKS) return bad("codes: each frame must hold 8 integers");
                for (int q = 0; q < HEARTMULA_CODEBOOKS; q++) {
                    if (!get_num(yyjson_arr_get(frame, (size_t) q), 0, 8191, true, &d)) return bad("codes: each code must be an integer from 0 to 8191");
                    c[(size_t) q] = (int32_t) d;
                }
                r->codes.push_back(c);
            }
        } else if (k == "codes_only") {
            if (!yyjson_is_bool(v)) return bad("codes_only must be a bool");
            r->codes_only = yyjson_get_bool(v);
        } else if (k == "id") {
            if (!yyjson_is_str(v)) return bad("id must be a string");
            r->id = yyjson_get_str(v);
        } else {
            return bad("unknown field '" + k + "'");
        }
    }
    if (!has_tags) return bad("tags is required");
    if (!r->codes.empty() && r->codes_only) return bad("codes and codes_only cannot be combined");
    if (!r->seed_given) r->seed = ((uint64_t) std::random_device{}() << 32) ^ std::random_device{}();
    if (!r->noise_seed_given) r->noise_seed = r->seed;
    return true;
}

// The request's pick laid over the current one.
static HeartMuLaPick heartmula_request_pick(const HeartMuLaPick & base, const HeartMuLaRequest & r) {
    HeartMuLaPick p = base;
    if (r.variant) p.variant = *r.variant;
    if (r.global) p.global = *r.global;
    if (r.local) p.local = *r.local;
    if (r.flow) p.flow = *r.flow;
    if (r.scalar) p.scalar = *r.scalar;
    if (r.adapter) p.adapter = *r.adapter;
    if (r.adapter_scale) p.adapter_scale = *r.adapter_scale;
    return p;
}

static int64_t heartmula_max_frames(double duration_s) {
    return (int64_t) std::floor(duration_s * HEARTMULA_FRAME_RATE);
}

// heartlib preprocess: tags, one zero MuQ slot, lyrics. Returns the text column.
static std::vector<int32_t> heartmula_prompt_ids(const HeartMuLaTokenizer & t, const std::string & tags,
                                                 const std::string & lyrics, int64_t * muq_idx) {
    auto ids = heartmula_tokenizer_prompt(t, tags, true);
    *muq_idx = (int64_t) ids.size();
    ids.push_back(0);
    const auto lyr = heartmula_tokenizer_prompt(t, lyrics, false);
    ids.insert(ids.end(), lyr.begin(), lyr.end());
    return ids;
}

// ── Runtime and admission (notes B, C) ──────────────────────────────────────

struct HeartMuLaRuntime {
    BackendPair     bp     = {};
    bool            bp_ref = false;
    HeartMuLaModel  lm;
    HeartMuLaCover  cover;     // MuLaCover only, loaded with its LM
    HeartMuLaAdapter adapter;  // rides the LM phase
    std::string     lm_key;  // global path | local path
    uint64_t        lm_bytes = 0;
    std::unique_ptr<heartcodec::Weights> rvq, flow, scalar;
    std::string     codec_key;
    uint64_t        codec_bytes = 0;
};

static void heartmula_free_lm(HeartMuLaRuntime * rt) {
    heartmula_adapter_free(&rt->adapter);
    heartmula_cover_free(&rt->cover);
    heartmula_model_free(&rt->lm);
    rt->lm_key.clear();
    rt->lm_bytes = 0;
}

static void heartmula_free_codec(HeartMuLaRuntime * rt) {
    rt->rvq.reset();
    rt->flow.reset();
    rt->scalar.reset();
    rt->codec_key.clear();
    rt->codec_bytes = 0;
}

static void heartmula_free_all(HeartMuLaRuntime * rt) {
    heartmula_free_lm(rt);
    heartmula_free_codec(rt);
    if (rt->bp_ref) {
        backend_release(rt->bp.backend, rt->bp.cpu_backend);
        rt->bp     = {};
        rt->bp_ref = false;
    }
}

static void heartmula_ensure_backend(HeartMuLaRuntime * rt) {
    if (!rt->bp_ref) {
        rt->bp     = backend_init("HeartMuLa");
        rt->bp_ref = true;
    }
}

static uint64_t heartmula_reserve_bytes() {
    const char * e = std::getenv("HEARTMULA_VRAM_RESERVE_MB");
    return (uint64_t) (e && *e ? std::atof(e) : HEARTMULA_DEFAULT_RESERVE_MB) * 1024 * 1024;
}

// Global KV bytes as heartmula_lm_reserve allocates them (F16, bucketed).
static uint64_t heartmula_kv_bytes(int64_t positions, int cfg_rows) {
    const HeartMuLaLmConfig c = heartmula_global_config();
    const int64_t n = std::min<int64_t>(heartmula_bucket(positions), c.n_ctx);
    return (uint64_t) c.n_layer * c.n_head_kv * c.head_dim * 2 * 2 * (uint64_t) n * (uint64_t) cfg_rows;
}

struct HeartMuLaAdmission {
    std::string phase;
    uint64_t    need = 0, free = 0, reclaimable = 0, reserve = 0, weights = 0, kv = 0;
    int64_t     positions = 0;
    bool        cfg = false, ok = false, reclaimed = false;

    std::string message() const {
        char buf[200];
        snprintf(buf, sizeof(buf), "HeartMuLa needs %.1f GB for this request; %.1f GB is free and %.1f GB can be reclaimed",
                 need / 1073741824.0, free / 1073741824.0, reclaimable / 1073741824.0);
        return buf;
    }
};

struct HeartMuLaHooks {
    std::function<bool()>                                    cancelled;
    std::function<void(const char * stage, int step, int total)> progress;
    std::function<uint64_t()>                                reclaimable;  // idle VRAM of other backends
    std::function<void()>                                    reclaim;      // evict it
    heartcodec::Solve                                        solve;        // sampler plugins, empty = native
};

// Physical free VRAM. Under Windows WDDM, cudaMemGetInfo counts other processes'
// idle VRAM as free (WDDM pages it out to system memory on demand): measured 24 GB
// "free" with 31.6 of 32 GB in use. NVML reports what nvidia-smi does, so an
// admitted load never relies on that spill; ggml's figure is the fallback.
// NVML ABI mirror and loader as in train/gpu-mem.h (which needs the trainer's
// lm-common.h, so it is not included here). The device is found by PCI bus id,
// because NVML's index order need not match CUDA's. False when NVML is absent or
// has no device at that address.
static bool heartmula_nvml_free(const char * pci_bus_id, uint64_t * out) {
#ifdef _WIN32
    struct Mem { unsigned long long total, free, used; };
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!lib) return false;
    auto f_init = (int (*)(void)) (void *) GetProcAddress(lib, "nvmlInit_v2");
    auto f_down = (int (*)(void)) (void *) GetProcAddress(lib, "nvmlShutdown");
    auto f_hand = (int (*)(const char *, void **)) (void *) GetProcAddress(lib, "nvmlDeviceGetHandleByPciBusId_v2");
    auto f_mem  = (int (*)(void *, Mem *)) (void *) GetProcAddress(lib, "nvmlDeviceGetMemoryInfo");
    bool ok = false;
    if (f_init && f_hand && f_mem && f_init() == 0) {
        void * dev = nullptr;
        Mem    m   = { 0, 0, 0 };
        if (f_hand(pci_bus_id, &dev) == 0 && dev && f_mem(dev, &m) == 0 && m.total > 0) {
            *out = m.free;
            ok   = true;
        }
        if (f_down) f_down();
    }
    FreeLibrary(lib);
    return ok;
#else
    (void) pci_bus_id;
    (void) out;
    return false;
#endif
}

// Free memory on the device a backend runs on. A GPU with a PCI address NVML
// knows gets NVML's figure; CPU and every other device get ggml's. The device
// handle belongs to the ggml registry and outlives any backend on it, so the
// HTTP threads may query a handle the worker published.
static uint64_t heartmula_device_free(ggml_backend_dev_t dev) {
    if (!dev) return 0;
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    uint64_t nvml = 0;
    if (props.type != GGML_BACKEND_DEVICE_TYPE_CPU && props.device_id && heartmula_nvml_free(props.device_id, &nvml)) {
        return nvml;
    }
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    return f;
}

static uint64_t heartmula_device_free(const HeartMuLaRuntime & rt) {
    return heartmula_device_free(rt.bp.backend ? ggml_backend_get_device(rt.bp.backend) : nullptr);
}

// Note C. Reclaims from other backends only when that is what makes it fit.
static HeartMuLaAdmission heartmula_admit(const HeartMuLaRuntime & rt, const char * phase, uint64_t weights,
                                          uint64_t kv, int64_t positions, bool cfg, const HeartMuLaHooks & hooks) {
    HeartMuLaAdmission a;
    a.phase       = phase;
    a.weights     = weights;
    a.kv          = kv;
    a.positions   = positions;
    a.cfg         = cfg;
    a.reserve     = heartmula_reserve_bytes();
    a.need        = weights + kv + a.reserve;
    a.free        = heartmula_device_free(rt);
    a.reclaimable = hooks.reclaimable ? hooks.reclaimable() : 0;
    a.ok          = a.need <= a.free + a.reclaimable;
    if (a.ok && a.need > a.free && hooks.reclaim) {
        hooks.reclaim();
        a.reclaimed = true;
    }
    fprintf(stderr, "[HeartMuLa] admission %s: need %.2f GiB (weights %.2f, kv %.2f, reserve %.2f), free %.2f GiB, reclaimable %.2f GiB -> %s\n",
            phase, a.need / 1073741824.0, weights / 1073741824.0, kv / 1073741824.0, a.reserve / 1073741824.0, a.free / 1073741824.0, a.reclaimable / 1073741824.0,
            !a.ok ? "refused" : a.reclaimed ? "admitted after reclaiming" : "admitted");
    return a;
}

// LM weights for the pick. Loads only when the resident pair differs.
static bool heartmula_load_lm(HeartMuLaRuntime * rt, const HeartMuLaResolved & p, std::string * err) {
    const std::string key = p.file[HM_GLOBAL].path + "|" + p.file[HM_LOCAL].path;
    if (rt->lm_key == key && rt->lm.loaded) return true;
    heartmula_free_lm(rt);
    if (!heartmula_model_load(&rt->lm, p.file[HM_GLOBAL].path.c_str(), p.file[HM_LOCAL].path.c_str(), err)) return false;
    if (heartmula_pipeline_detail::lower(rt->lm.variant) == "mulacover" &&
        !heartmula_cover_load(&rt->cover, rt->lm, p.file[HM_GLOBAL].path.c_str(), err)) {
        heartmula_free_lm(rt);
        return false;
    }
    rt->lm_key   = key;
    rt->lm_bytes = ggml_backend_buffer_get_size(rt->lm.wctx_global.buffer) + ggml_backend_buffer_get_size(rt->lm.wctx_local.buffer) +
                   (rt->cover.loaded ? ggml_backend_buffer_get_size(rt->cover.wctx.buffer) : 0);
    return true;
}

// The pick's adapter, with the same home weights, is already loaded.
static bool heartmula_adapter_resident(const HeartMuLaRuntime & rt, const HeartMuLaResolved & p) {
    return rt.adapter.loaded() && rt.adapter.path == p.adapter && rt.adapter.home_path == p.adapter_home;
}

// The pick's adapter over the resident LM (adapter.h repeats check 5 with the shapes).
static bool heartmula_load_adapter(HeartMuLaRuntime * rt, const HeartMuLaResolved & p, std::string * err) {
    if (p.adapter.empty() || heartmula_adapter_resident(*rt, p)) return true;
    return heartmula_adapter_load(&rt->adapter, rt->lm, p.adapter.c_str(), err, p.adapter_home.c_str());
}

static uint64_t heartmula_lm_need(const HeartMuLaRuntime & rt, const HeartMuLaResolved & p) {
    const bool resident = rt.lm.loaded && rt.lm_key == p.file[HM_GLOBAL].path + "|" + p.file[HM_LOCAL].path;
    const bool adapter  = p.adapter.empty() || (resident && heartmula_adapter_resident(rt, p));
    return (resident ? 0 : p.file[HM_GLOBAL].gpu_bytes + p.file[HM_LOCAL].gpu_bytes) + (adapter ? 0 : p.adapter_bytes);
}

static uint64_t heartmula_codec_need(const HeartMuLaRuntime & rt, const HeartMuLaResolved & p) {
    const std::string key = p.file[HM_RVQ].path + "|" + p.file[HM_FLOW].path + "|" + p.file[HM_SCALAR].path;
    return rt.codec_key == key ? 0 : p.file[HM_RVQ].gpu_bytes + p.file[HM_FLOW].gpu_bytes + p.file[HM_SCALAR].gpu_bytes;
}

// Free whatever the pick no longer names (LM pair, adapter, codec set), so admission
// measures their VRAM as free instead of charging the replacement on top of them.
static void heartmula_drop_stale(HeartMuLaRuntime * rt, const HeartMuLaResolved * p) {
    if (rt->lm.loaded && (!p || rt->lm_key != p->file[HM_GLOBAL].path + "|" + p->file[HM_LOCAL].path)) {
        heartmula_free_lm(rt);
    }
    if (rt->adapter.loaded() && (!p || !heartmula_adapter_resident(*rt, *p))) heartmula_adapter_free(&rt->adapter);
    const std::string codec = p ? p->file[HM_RVQ].path + "|" + p->file[HM_FLOW].path + "|" + p->file[HM_SCALAR].path : "";
    if (!rt->codec_key.empty() && rt->codec_key != codec) heartmula_free_codec(rt);
}

static uint64_t heartmula_resident_bytes(const HeartMuLaRuntime & rt) {
    return rt.lm_bytes + rt.codec_bytes + (rt.adapter.wctx.buffer ? ggml_backend_buffer_get_size(rt.adapter.wctx.buffer) : 0);
}

// ── Generate ────────────────────────────────────────────────────────────────

struct HeartMuLaResult {
    std::vector<int32_t> codes;  // [frames][8]
    int64_t              frames = 0;
    std::vector<float>   audio;  // planar stereo [2][samples]
    int64_t              samples = 0;
    int64_t              prompt_tokens = 0, max_frames = 0;
    std::string          end_reason;  // completed | limit_hit | cancelled
    std::string          error_code, error;
    std::optional<HeartMuLaAdmission> admission;
    double               t_admission = 0, t_load = 0, t_prefill = 0, t_frames = 0, t_codec = 0;
};

namespace heartmula_pipeline_detail {

inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Note E: N(0, 1) from raw 64-bit words, identical on every standard library.
struct Gaussian {
    std::mt19937_64 rng;
    explicit Gaussian(uint64_t seed) : rng(seed) {}
    std::vector<float> operator()(size_t n) {
        std::vector<float> out(n);
        for (size_t i = 0; i < n; i += 2) {
            const double u1 = ((double) (rng() >> 11) + 0.5) * 0x1.0p-53, u2 = (double) (rng() >> 11) * 0x1.0p-53;
            const double r  = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
            out[i] = (float) (r * std::cos(a));
            if (i + 1 < n) out[i + 1] = (float) (r * std::sin(a));
        }
        return out;
    }
};

}  // namespace heartmula_pipeline_detail

// One request, both phases. Returns false with error_code set on failure or cancel
// ("cancelled"). Residency on every exit follows note B.
static bool heartmula_generate(HeartMuLaRuntime * rt, const HeartMuLaResolved & p, const HeartMuLaTokenizer & tok,
                               const HeartMuLaRequest & req, bool keep_loaded, const HeartMuLaHooks & hooks,
                               HeartMuLaResult * res) {
    using namespace heartmula_pipeline_detail;
    HeartMuLaLmGraph    g;
    HeartMuLaLocalGraph lg;
    auto progress = [&](const char * s, int a, int b) { if (hooks.progress) hooks.progress(s, a, b); };
    auto cancelled = [&]() { return hooks.cancelled && hooks.cancelled(); };
    auto finish = [&](const char * code, const std::string & msg) {
        heartmula_local_free(&lg);
        heartmula_lm_free(&g);
        res->frames = (int64_t) res->codes.size() / HEARTMULA_CODEBOOKS;
        if (!keep_loaded) {
            heartmula_free_lm(rt);
            heartmula_free_codec(rt);
        }
        if (!code) return true;
        res->error_code = code;
        res->error      = msg;
        if (!strcmp(code, "cancelled")) res->end_reason = "cancelled";
        fprintf(stderr, "[HeartMuLa] %s: %s\n", code, msg.c_str());
        return false;
    };
    auto load_error = [&](const std::string & e) {
        return finish(e.find("alloc") != std::string::npos ? "vram_alloc_failed" : "load_failed", e);
    };
    heartmula_ensure_backend(rt);
    heartmula_drop_stale(rt, &p);
    std::string err;
    const int   B = req.cfg_scale != 1.0f ? 2 : 1;

    if (req.codes.empty()) {
        // ── LM phase ──
        int64_t    muq = 0;
        const auto ids = heartmula_prompt_ids(tok, req.tags, req.lyrics, &muq);
        const int64_t T = (int64_t) ids.size();
        res->prompt_tokens = T;
        res->max_frames    = heartmula_max_frames(req.duration_s);
        const int64_t positions = T + res->max_frames + 1;
        if (positions > heartmula_global_config().n_ctx) {
            return finish("context_overflow", "prompt " + std::to_string(T) + " + frames " + std::to_string(res->max_frames) + " exceeds 8192");
        }
        auto t0 = std::chrono::steady_clock::now();
        progress("admission", 0, 0);
        if (!keep_loaded) heartmula_free_codec(rt);
        uint64_t weights = heartmula_lm_need(*rt, p);
        const uint64_t kv = heartmula_kv_bytes(positions, B);
        if (keep_loaded && rt->codec_bytes && weights + kv + heartmula_reserve_bytes() > heartmula_device_free(*rt)) {
            heartmula_free_codec(rt);  // co-residency only if both phases fit
        }
        res->admission = heartmula_admit(*rt, "lm", weights, kv, positions, B == 2, hooks);
        res->t_admission = ms_since(t0);
        if (!res->admission->ok) return finish("vram_insufficient", res->admission->message());
        if (cancelled()) return finish("cancelled", "cancelled before loading");

        t0 = std::chrono::steady_clock::now();
        progress("loading_lm", 0, 2);
        if (!heartmula_load_lm(rt, p, &err)) return load_error(err);
        progress("loading_lm", 2, 2);
        if (!p.adapter.empty()) {
            progress("loading_adapter", 0, 0);
            if (cancelled()) return finish("cancelled", "cancelled before loading the adapter");
            if (!heartmula_load_adapter(rt, p, &err)) return finish("incompatible", err);
        }
        if (!heartmula_lm_reserve(rt->lm, &g, B, positions, &err) || !heartmula_local_prepare(rt->lm, &lg, B, &err)) {
            return finish("vram_alloc_failed", err);
        }
        // lm-graph.h note F: scale 0 or no adapter builds the base graph.
        HeartMuLaAdapterScales scales = req.adapter_groups;
        scales.global                 = p.adapter_scale;
        heartmula_lm_set_adapter(&g, p.adapter.empty() ? nullptr : &rt->adapter, scales);
        res->t_load += ms_since(t0);
        if (cancelled()) return finish("cancelled", "cancelled after loading");

        // Prefill in bounded chunks (lm-graph.h note D), cancel between them.
        t0 = std::chrono::steady_clock::now();
        std::vector<HeartMuLaPosition> prompt((size_t) T);
        for (int64_t i = 0; i < T; i++) {
            prompt[(size_t) i].id[HEARTMULA_COLUMNS - 1]   = ids[(size_t) i];
            prompt[(size_t) i].mask[HEARTMULA_COLUMNS - 1] = true;
        }
        const int64_t H = rt->lm.global.cfg.n_embd;
        std::vector<float>   hidden((size_t) (B * H)), c0((size_t) (B * HEARTMULA_AUDIO_VOCAB));
        std::vector<int64_t> pos;
        for (int64_t start = 0; start < T; start += HEARTMULA_PREFILL_CHUNK) {
            progress("prefill", (int) start, (int) T);
            if (cancelled()) return finish("cancelled", "cancelled during prefill");
            const int64_t n = std::min<int64_t>(HEARTMULA_PREFILL_CHUNK, T - start);
            pos.resize((size_t) n);
            for (int64_t i = 0; i < n; i++) pos[(size_t) i] = start + i;
            const bool last = start + n == T;
            if (!heartmula_lm_step(rt->lm, &g, prompt.data() + start, pos.data(), n,
                                   muq >= start && muq < start + n ? muq - start : -1, last ? hidden.data() : nullptr,
                                   last ? c0.data() : nullptr, &err)) {
                return finish("vram_alloc_failed", err);
            }
        }
        progress("prefill", (int) T, (int) T);
        res->t_prefill = ms_since(t0);

        // Frames: frame 0 from the prompt, then one global step per frame (heartlib _forward).
        t0 = std::chrono::steady_clock::now();
        HeartMuLaDraws  draws(req.seed);
        const auto      choose = heartmula_sampling_chooser(req.cfg_scale, req.topk, req.temperature, &draws);
        res->end_reason        = "limit_hit";
        for (int64_t f = 0; f < res->max_frames; f++) {
            progress("frames", (int) f, (int) res->max_frames);
            if (cancelled()) return finish("cancelled", "cancelled during frames");
            if (f > 0) {
                HeartMuLaPosition in;
                for (int q = 0; q < HEARTMULA_CODEBOOKS; q++) {
                    in.id[q]   = res->codes[(size_t) ((f - 1) * HEARTMULA_CODEBOOKS + q)];
                    in.mask[q] = true;
                }
                const int64_t position = T + f - 1;
                if (!heartmula_lm_step(rt->lm, &g, &in, &position, 1, -1, hidden.data(), c0.data(), &err)) {
                    return finish("vram_alloc_failed", err);
                }
            }
            int32_t codes[HEARTMULA_CODEBOOKS];
            if (!heartmula_local_frame(rt->lm, &lg, hidden.data(), c0.data(), choose, codes, nullptr, &err)) {
                return finish("vram_alloc_failed", err);
            }
            if (*std::max_element(codes, codes + HEARTMULA_CODEBOOKS) >= HEARTMULA_AUDIO_EOS) {
                // The frame that ended the song, as sampled: which codebooks carried the end token.
                fprintf(stderr, "[HeartMuLa] end of song at frame %lld: codes %d %d %d %d %d %d %d %d\n",
                        (long long) (res->codes.size() / HEARTMULA_CODEBOOKS), codes[0], codes[1], codes[2], codes[3],
                        codes[4], codes[5], codes[6], codes[7]);
                res->end_reason = "completed";
                break;
            }
            // Reject reserved code 8192 before another LM step or a codes-only return.
            if (std::any_of(codes, codes + HEARTMULA_CODEBOOKS, [](int32_t c) { return c < 0 || c >= 8192; })) {
                return finish("bad_codes", "a sampled code is outside the codec's 8192-entry codebook");
            }
            res->codes.insert(res->codes.end(), codes, codes + HEARTMULA_CODEBOOKS);
        }
        res->frames   = (int64_t) res->codes.size() / HEARTMULA_CODEBOOKS;
        res->t_frames = ms_since(t0);
        progress("frames", (int) res->frames, (int) res->max_frames);
        fprintf(stderr, "[HeartMuLa] LM: %lld prompt tokens, %lld frames (%s), prefill %.0f ms, frames %.0f ms (%.1f ms/frame)\n",
                (long long) T, (long long) res->frames, res->end_reason.c_str(), res->t_prefill, res->t_frames,
                res->frames ? res->t_frames / (double) res->frames : 0.0);
        heartmula_local_free(&lg);
        heartmula_lm_free(&g);  // the KV cache never outlives the request
        if (res->frames == 0) return finish("no_audio", "the model ended the song before its first frame");
        if (req.codes_only) return finish(nullptr, "");
    } else {
        for (const auto & c : req.codes) res->codes.insert(res->codes.end(), c.begin(), c.end());
        res->frames     = (int64_t) req.codes.size();
        res->max_frames = res->frames;
        res->end_reason = "completed";
    }
    for (int32_t c : res->codes) {
        if (c < 0 || c >= 8192) {
            return finish("bad_codes", "a sampled code is outside the codec's 8192-entry codebook");
        }
    }

    // ── Codec phase ──
    auto t0 = std::chrono::steady_clock::now();
    progress("admission", 0, 0);
    if (!keep_loaded) heartmula_free_lm(rt);
    const uint64_t weights = heartmula_codec_need(*rt, p);
    if (keep_loaded && rt->lm_bytes && weights + heartmula_reserve_bytes() > heartmula_device_free(*rt)) {
        heartmula_free_lm(rt);
    }
    HeartMuLaAdmission a = heartmula_admit(*rt, "codec", weights, 0, 0, req.codec_guidance > 1.0f, hooks);
    res->t_admission += ms_since(t0);
    if (!a.ok) {
        res->admission = a;
        return finish("vram_insufficient", a.message());
    }
    if (cancelled()) return finish("cancelled", "cancelled before loading the codec");

    t0 = std::chrono::steady_clock::now();
    const std::string key = p.file[HM_RVQ].path + "|" + p.file[HM_FLOW].path + "|" + p.file[HM_SCALAR].path;
    try {
        if (rt->codec_key != key) {
            heartmula_free_codec(rt);
            const int roles[3] = { HM_RVQ, HM_FLOW, HM_SCALAR };
            std::unique_ptr<heartcodec::Weights> * slots[3] = { &rt->rvq, &rt->flow, &rt->scalar };
            for (int i = 0; i < 3; i++) {
                progress("loading_codec", i, 3);
                if (cancelled()) return finish("cancelled", "cancelled while loading the codec");
                *slots[i] = std::make_unique<heartcodec::Weights>();
                (*slots[i])->load(p.file[roles[i]].path, k_heartmula_roles[roles[i]], rt->bp.backend);
                rt->codec_bytes += ggml_backend_buffer_get_size((*slots[i])->storage.buffer);
            }
            rt->codec_key = key;
        }
        progress("loading_codec", 3, 3);
        res->t_load += ms_since(t0);

        t0 = std::chrono::steady_clock::now();
        std::vector<int64_t> planar((size_t) (HEARTMULA_CODEBOOKS * res->frames));
        for (int64_t t = 0; t < res->frames; t++) {
            for (int q = 0; q < HEARTMULA_CODEBOOKS; q++) {
                planar[(size_t) (q * res->frames + t)] = res->codes[(size_t) (t * HEARTMULA_CODEBOOKS + q)];
            }
        }
        Gaussian noise(req.noise_seed);
        res->audio = heartcodec::decode(
            *rt->rvq, *rt->flow, *rt->scalar, rt->bp.backend, planar, res->frames, std::ref(noise), req.codec_steps,
            req.codec_guidance, cancelled,
            [&](int n, int chunks) { progress("codec_decode", n * req.codec_steps, chunks * req.codec_steps); }, {},
            hooks.solve);
        res->samples = (int64_t) res->audio.size() / 2;
        res->t_codec = ms_since(t0);
    } catch (const heartcodec::Cancelled &) {
        return finish("cancelled", "cancelled during codec decode");
    } catch (const std::exception & e) {
        return load_error(e.what());
    }
    fprintf(stderr, "[HeartMuLa] codec: %lld frames -> %.2f s audio in %.0f ms\n", (long long) res->frames,
            res->samples / 48000.0, res->t_codec);
    return finish(nullptr, "");
}
