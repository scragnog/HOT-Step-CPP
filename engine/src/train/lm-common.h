#pragma once
// lm-common.h — shared small utilities for `ace-train train-lm`.
//
// Scope (plan §1.1 / §3.2 / §3.7):
//   * stmd_read()      — read ONLY the __metadata__ object of a safetensors
//                        file (engine/src/safetensors.h deliberately skips it)
//   * sha1_16()        — 16 hex chars of SHA-1, used for the `_tok` identity
//   * LmRng            — deterministic xoshiro256** + Fisher-Yates shuffle
//   * lm_json_escape() — JSON string escaping for the JSONL stream
//   * lm_vram_*_mb()   — CUDA device memory probes
//   * lm_size_label_from_config() — "0.6B" / "1.7B" / "4B" / "custom"
//
// docs/plans/2026-07-27-lm-trainer-implementation.md
//
// NOTE: this header (and every other lm-*.h) is only ever included from
// engine/tools/ace-train.cpp. `jl()` / `json_escape()` are defined there, but
// AFTER the include, so they are forward-declared below rather than duplicated
// (plan §3.1: "Reuse the existing g_jsonl / jl() / json_escape()").

#include "config-json.h"           // Qwen3LMConfig
#include "train/preprocess-io.h"   // pm_* path/json helpers, stw_replace_file
#include "yyjson.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>   // tolower — lm_normalize_language
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>  // std::pair — lm_normalize_language's name table
#include <vector>

// ─── JSONL plumbing (defined in ace-train.cpp) ──────────────────────────────

static void jl(const char * fmt, ...);

// ─── JSON escaping ──────────────────────────────────────────────────────────

static std::string lm_json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char) s[i];
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", (unsigned) c);
                    o += b;
                } else {
                    o.push_back((char) c);
                }
        }
    }
    return o;
}

// One `log` event (§2.2). level ∈ info|warn|error.
static void lm_log(const char * level, const std::string & msg) {
    jl("{\"type\":\"log\",\"level\":\"%s\",\"message\":\"%s\"}", level, lm_json_escape(msg).c_str());
    fprintf(stderr, "[train-lm] %s: %s\n", level, msg.c_str());
}

// One `fatal` event (§2.2). Always the last line before a non-zero exit.
static void lm_fatal(const char * reason, const std::string & msg, const std::string & extra = std::string()) {
    jl("{\"type\":\"fatal\",\"reason\":\"%s\",\"message\":\"%s\"%s}", reason, lm_json_escape(msg).c_str(),
       extra.c_str());
    fprintf(stderr, "[train-lm] FATAL (%s): %s\n", reason, msg.c_str());
}

// ─── deterministic RNG (xoshiro256** seeded through splitmix64) ─────────────

struct LmRng {
    uint64_t s[4];
};

static inline uint64_t lm_splitmix64(uint64_t * x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void lm_rng_seed(LmRng * r, uint64_t seed) {
    uint64_t x = seed ? seed : 0x123456789ABCDEFull;
    for (int i = 0; i < 4; i++) {
        r->s[i] = lm_splitmix64(&x);
    }
}

static inline uint64_t lm_rng_next(LmRng * r) {
    const uint64_t s1     = r->s[1];
    const uint64_t result = ((s1 * 5ull) << 7 | (s1 * 5ull) >> 57) * 9ull;
    const uint64_t t      = s1 << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= s1;
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = (r->s[3] << 45) | (r->s[3] >> 19);
    return result;
}

// Uniform in [0, n) without modulo bias.
static inline uint64_t lm_rng_below(LmRng * r, uint64_t n) {
    if (n <= 1) {
        return 0;
    }
    const uint64_t limit = UINT64_MAX - (UINT64_MAX % n);
    uint64_t       v;
    do {
        v = lm_rng_next(r);
    } while (v >= limit);
    return v % n;
}

static inline float lm_rng_uniform(LmRng * r) {  // (0,1)
    return (float) ((lm_rng_next(r) >> 11) + 1) / (float) (1ull << 53);
}

static inline float lm_rng_normal(LmRng * r) {
    const float u1 = lm_rng_uniform(r);
    const float u2 = lm_rng_uniform(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

static inline void lm_rng_fill_normal(LmRng * r, std::vector<float> & v, float sigma) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = lm_rng_normal(r) * sigma;
    }
}

// In-place Fisher-Yates.
static inline void lm_rng_shuffle(LmRng * r, std::vector<int> & v) {
    for (size_t i = v.size(); i > 1; i--) {
        const size_t j = (size_t) lm_rng_below(r, (uint64_t) i);
        std::swap(v[i - 1], v[j]);
    }
}

// ─── SHA-1 (only used for the FSQ-tokenizer identity string) ────────────────

struct LmSha1 {
    uint32_t h[5];
    uint64_t len;
    uint8_t  buf[64];
    size_t   buf_len;
};

static inline uint32_t lm_sha1_rol(uint32_t v, int b) {
    return (v << b) | (v >> (32 - b));
}

static void lm_sha1_block(LmSha1 * c, const uint8_t * p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t) p[i * 4] << 24) | ((uint32_t) p[i * 4 + 1] << 16) | ((uint32_t) p[i * 4 + 2] << 8) |
               (uint32_t) p[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++) {
        w[i] = lm_sha1_rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = c->h[0], b = c->h[1], d = c->h[2], e = c->h[3], f = c->h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t k, t;
        if (i < 20) {
            t = (b & d) | ((~b) & e);
            k = 0x5A827999u;
        } else if (i < 40) {
            t = b ^ d ^ e;
            k = 0x6ED9EBA1u;
        } else if (i < 60) {
            t = (b & d) | (b & e) | (d & e);
            k = 0x8F1BBCDCu;
        } else {
            t = b ^ d ^ e;
            k = 0xCA62C1D6u;
        }
        const uint32_t tmp = lm_sha1_rol(a, 5) + t + f + k + w[i];
        f                  = e;
        e                  = d;
        d                  = lm_sha1_rol(b, 30);
        b                  = a;
        a                  = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += d;
    c->h[3] += e;
    c->h[4] += f;
}

static void lm_sha1_init(LmSha1 * c) {
    c->h[0]   = 0x67452301u;
    c->h[1]   = 0xEFCDAB89u;
    c->h[2]   = 0x98BADCFEu;
    c->h[3]   = 0x10325476u;
    c->h[4]   = 0xC3D2E1F0u;
    c->len    = 0;
    c->buf_len = 0;
}

static void lm_sha1_update(LmSha1 * c, const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    c->len += (uint64_t) n * 8ull;
    while (n > 0) {
        const size_t take = std::min(n, (size_t) 64 - c->buf_len);
        memcpy(c->buf + c->buf_len, p, take);
        c->buf_len += take;
        p += take;
        n -= take;
        if (c->buf_len == 64) {
            lm_sha1_block(c, c->buf);
            c->buf_len = 0;
        }
    }
}

static std::string lm_sha1_hex(LmSha1 * c) {
    const uint64_t bits = c->len;
    const uint8_t  pad  = 0x80;
    lm_sha1_update(c, &pad, 1);
    const uint8_t z = 0;
    while (c->buf_len != 56) {
        lm_sha1_update(c, &z, 1);
    }
    uint8_t lenbe[8];
    for (int i = 0; i < 8; i++) {
        lenbe[i] = (uint8_t) ((bits >> (56 - 8 * i)) & 0xFFu);
    }
    c->len = 0;  // the length bytes themselves are not counted
    memcpy(c->buf + 56, lenbe, 8);
    lm_sha1_block(c, c->buf);
    c->buf_len = 0;

    char out[41];
    for (int i = 0; i < 5; i++) {
        snprintf(out + i * 8, 9, "%08x", c->h[i]);
    }
    return std::string(out, 40);
}

// First 16 hex chars of SHA-1(s).
static std::string sha1_16(const std::string & s) {
    LmSha1 c;
    lm_sha1_init(&c);
    lm_sha1_update(&c, s.data(), s.size());
    return lm_sha1_hex(&c).substr(0, 16);
}

// Bump whenever the extract stage's float -> code mapping changes: it is folded
// into `_tok`, so every cached lm_codes.jsonl row is invalidated and re-encoded.
//   v1 — fsq-tok.h::fsq_encode_index (old vqp symmetry-preserving bound)
//   v2 — reference-conformant encoder (vqp >= 1.27.21 ResidualFSQ: soft clamp +
//        hard clamp), 100% match vs the PyTorch reference. Originally lived in
//        train/lm-fsq.h::fsq_encode_index_ref; that file is gone — the same
//        formula is now the ENGINE-WIDE encoder in src/fsq-quant.h, so extract
//        calls the shared tok_ggml_encode and codes stay bit-identical to v2.
#define LM_FSQ_ENCODER_VERSION "fsq-v2"

// _tok = sha1(dit_path + "\0" + size + "\0" + mtime_ms + "\0" + encoder)[0:16]  (§2.3)
static std::string lm_tok_identity(const std::string & dit_path) {
    long long bytes = 0, mtime = 0;
    pm_stat_file(dit_path, &bytes, &mtime);
    char b[64];
    snprintf(b, sizeof(b), "%lld", bytes);
    std::string s = dit_path;
    s.push_back('\0');
    s += b;
    s.push_back('\0');
    snprintf(b, sizeof(b), "%lld", mtime);
    s += b;
    s.push_back('\0');
    s += LM_FSQ_ENCODER_VERSION;
    return sha1_16(s);
}

// ─── safetensors __metadata__ reader (L14) ──────────────────────────────────
//
// engine/src/safetensors.h skips __metadata__ on purpose (safetensors.h:141),
// and we do not edit it. This reads just the header object.
static bool stmd_read(const char * path, std::map<std::string, std::string> * out) {
    out->clear();
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    uint8_t hl[8];
    if (fread(hl, 1, 8, f) != 8) {
        fclose(f);
        return false;
    }
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; i++) {
        hdr_len |= (uint64_t) hl[i] << (8 * i);
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    const long fsz = ftell(f);
    if (hdr_len == 0 || hdr_len > (uint64_t) 64 * 1024 * 1024 || fsz <= 8 || hdr_len > (uint64_t) (fsz - 8)) {
        fclose(f);
        return false;
    }
    std::vector<char> hdr((size_t) hdr_len);
    if (fseek(f, 8, SEEK_SET) != 0 || fread(hdr.data(), 1, hdr.size(), f) != hdr.size()) {
        fclose(f);
        return false;
    }
    fclose(f);

    yyjson_doc * doc = yyjson_read(hdr.data(), hdr.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md   = (root && yyjson_is_obj(root)) ? yyjson_obj_get(root, "__metadata__") : NULL;
    if (md && yyjson_is_obj(md)) {
        yyjson_obj_iter it;
        yyjson_obj_iter_init(md, &it);
        yyjson_val * k;
        while ((k = yyjson_obj_iter_next(&it)) != NULL) {
            yyjson_val * v = yyjson_obj_iter_get_val(k);
            if (yyjson_is_str(k) && v && yyjson_is_str(v)) {
                (*out)[yyjson_get_str(k)] = yyjson_get_str(v);
            }
        }
    }
    yyjson_doc_free(doc);
    return true;  // true with an empty map when there is no __metadata__
}

static inline std::string lm_md_get(const std::map<std::string, std::string> & md, const char * key) {
    auto it = md.find(key);
    return it == md.end() ? std::string() : it->second;
}

// Normalize a dataset language to an ISO code the INFERENCE FSM can emit
// (2026-08-11). metadata-fsm.h:203-208 constrains the CoT's `language:` field
// to 51 ISO codes; a full name like "english" is not among them, so an adapter
// trained on it learns a token the sampler is forbidden to produce — with the
// language unpinned the model's mass sits on a blocked word and whichever
// allowed code survives the mask wins (measured: sr/pt/fr on 3 of 6 seeds).
// The whole 183-dataset corpus carried "english"; the tensor __metadata__ this
// reads STILL DOES, so normalizing here is what stops a re-extract from
// reintroducing it. Unrecognised values become "unknown" — a value the FSM
// does accept — rather than passing through.
static std::string lm_normalize_language(const std::string & raw_in) {
    std::string raw;
    for (char c : raw_in) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            raw += (char) tolower((unsigned char) c);
        }
    }
    if (raw.empty()) {
        return "";  // absent stays absent — build_cot_yaml omits the line
    }
    static const char * kValid[] = {
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",  "es", "fa", "fi",
        "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja",  "ko", "la", "lt",
        "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa",  "sk", "sr", "sv",
        "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
    };
    for (const char * v : kValid) {
        if (raw == v) {
            return raw;
        }
    }
    static const std::pair<const char *, const char *> kNames[] = {
        { "arabic", "ar" }, { "azerbaijani", "az" }, { "bulgarian", "bg" }, { "bengali", "bn" },
        { "catalan", "ca" }, { "czech", "cs" }, { "danish", "da" }, { "german", "de" },
        { "greek", "el" }, { "english", "en" }, { "spanish", "es" }, { "persian", "fa" },
        { "farsi", "fa" }, { "finnish", "fi" }, { "french", "fr" }, { "hebrew", "he" },
        { "hindi", "hi" }, { "croatian", "hr" }, { "haitian", "ht" }, { "hungarian", "hu" },
        { "indonesian", "id" }, { "icelandic", "is" }, { "italian", "it" }, { "japanese", "ja" },
        { "korean", "ko" }, { "latin", "la" }, { "lithuanian", "lt" }, { "malay", "ms" },
        { "nepali", "ne" }, { "dutch", "nl" }, { "norwegian", "no" }, { "punjabi", "pa" },
        { "polish", "pl" }, { "portuguese", "pt" }, { "romanian", "ro" }, { "russian", "ru" },
        { "sanskrit", "sa" }, { "slovak", "sk" }, { "serbian", "sr" }, { "swedish", "sv" },
        { "swahili", "sw" }, { "tamil", "ta" }, { "telugu", "te" }, { "thai", "th" },
        { "tagalog", "tl" }, { "turkish", "tr" }, { "ukrainian", "uk" }, { "urdu", "ur" },
        { "vietnamese", "vi" }, { "cantonese", "yue" }, { "chinese", "zh" }, { "mandarin", "zh" },
    };
    for (const auto & kv : kNames) {
        if (raw == kv.first) {
            return kv.second;
        }
    }
    // "en-gb" / "en_us": take the primary subtag when it is itself valid.
    const size_t dash = raw.find_first_of("-_");
    if (dash != std::string::npos) {
        const std::string primary = raw.substr(0, dash);
        for (const char * v : kValid) {
            if (primary == v) {
                return primary;
            }
        }
    }
    return "unknown";
}

// Apply the dataset's custom tag exactly the way preprocess-run.h:192-204 does.
// The training caption MUST carry the trigger word — §2.3's own example
// ("electriccallboy, This track is …") and Side-Step's abba lm_codes.jsonl
// ("abba, This track is …") both do.
static std::string lm_apply_tag(const std::string & text, const std::string & tag, const std::string & position) {
    if (tag.empty()) {
        return text;
    }
    const std::string pos = position.empty() ? std::string("prepend") : position;
    if (pos == "prepend") {
        return text.empty() ? tag : tag + ", " + text;
    }
    if (pos == "append") {
        return text.empty() ? tag : text + ", " + tag;
    }
    if (pos == "replace") {
        return tag;  // the caption IS the trigger — see preprocess-run.h
    }
    return text;
}

// Normalise a (trigger, position) pair for embedding. All three positions are
// embeddable: each one puts the trigger into the training caption, so claiming
// it is truthful. "replace" means the caption IS the trigger — the adapter saw
// nothing but that token, and inference has to reproduce exactly that or it is
// off-distribution, which is why the position travels with the trigger instead
// of being assumed. (Before 2026-08-12 "replace" applied no tag at all during
// preprocessing, so it was correctly refused here; preprocess-run.h now honours
// it.) `reason` receives a human-readable note when a non-empty tag is dropped.
static bool lm_trigger_normalize(std::string * tag, std::string * position, std::string * reason) {
    if (!tag || tag->empty()) {
        if (position) {
            position->clear();
        }
        return false;
    }
    std::string pos = position ? *position : std::string();
    if (pos.empty()) {
        pos = "prepend";
    }
    if (pos != "prepend" && pos != "append" && pos != "replace") {
        if (reason) {
            *reason = "trigger \"" + *tag + "\" not embedded: unknown tag_position \"" + pos +
                      "\" (expected prepend | append | replace)";
        }
        tag->clear();
        if (position) {
            position->clear();
        }
        return false;
    }
    if (position) {
        *position = pos;
    }
    return true;
}

// ─── VRAM probes ────────────────────────────────────────────────────────────

static inline void lm_vram_query(ggml_backend_t backend, size_t * free_b, size_t * total_b) {
    *free_b  = 0;
    *total_b = 0;
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    if (!dev) {
        return;
    }
    ggml_backend_dev_memory(dev, free_b, total_b);
}

static inline size_t lm_vram_used_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return t == 0 ? 0 : (t - f) / (1024 * 1024);
}

static inline size_t lm_vram_free_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return f / (1024 * 1024);
}

static inline size_t lm_vram_total_mb(ggml_backend_t backend) {
    size_t f = 0, t = 0;
    lm_vram_query(backend, &f, &t);
    return t / (1024 * 1024);
}

// ─── model-size label ───────────────────────────────────────────────────────

// Label written into adapter_config / lm_train_log and used for the 4B refusal.
static std::string lm_size_label_from_config(const Qwen3LMConfig & c) {
    if (c.hidden_size <= 1024 && c.n_layers <= 28) {
        return "0.6B";
    }
    if (c.hidden_size <= 2048 && c.n_layers <= 28) {
        return "1.7B";
    }
    if (c.hidden_size >= 2560 || c.n_layers >= 36) {
        return "4B";
    }
    return "custom";
}

// ─── low-VRAM defaults (4B plan §2.1 / D6) ──────────────────────────────────

// `g` is legal iff it divides n_heads AND every block still owns >= 1 kv head
// (g * Nkv must be a whole multiple of Nh, so the GQA broadcast factor Nh/Nkv
// survives the split). g == 0 means "off".
static inline bool lm_ckpt_head_block_ok(const Qwen3LMConfig & c, int g) {
    if (g == 0) {
        return true;
    }
    if (g < 0 || g >= c.n_heads || c.n_heads % g != 0) {
        return false;
    }
    return ((int64_t) g * (int64_t) c.n_kv_heads) % (int64_t) c.n_heads == 0;
}

// Default heads-per-attention-block when --low-vram is active:
//   n_heads <= 16 -> 0 (off), else 8.
//
// `attn_flash` (D3) forces 0. Head blocking exists to cap the `3*hb*S^2*4`
// score/softmax transient, and the fused op has no S^2 term at all — so under
// --attn flash the blocking would buy nothing and cost four ggml_acc copies per
// layer, and lm_attn_head_blocked has no fused arm (it asserts). The default
// argument is false, so every existing caller resolves exactly as it shipped.
static inline int lm_ckpt_default_head_block(const Qwen3LMConfig & c, bool attn_flash = false) {
    if (attn_flash) {
        return 0;
    }
    if (c.n_heads <= 16) {
        return 0;
    }
    return lm_ckpt_head_block_ok(c, 8) ? 8 : 0;
}

// ─── attention mask allocation / upload (--attn flash, D4) ──────────────────
//
// The exact path's mask is an F32 [S_kv*S] flat buffer that ggml_soft_max_ext
// reads through a contiguous [S_kv, S] view. The fused op will not take that:
// ggml_flash_attn_train asserts `mask->type == GGML_TYPE_F16` (engine/ggml/src/
// ggml.c). So flash mode allocates the SAME flat layout in F16 and converts on
// upload; exact mode allocates F32 and uploads byte-identical bytes.
//
// WHY NOT AN IN-GRAPH CAST. A ggml_cast of the mask would be S^2 elements per
// layer per graph — at 4B/S 3500 that is 24 MB of extra traffic 36 times a
// forward, to reproduce a buffer that never changes within a micro-step. The
// conversion belongs at the one host-side upload.
//
// EVERY view of the buffer must therefore size its row stride with
// ggml_element_size(t) instead of sizeof(float). That expression is IDENTICAL
// for an F32 tensor, which is what keeps exact mode byte-identical, and is the
// only reason it is safe to change the shipped view sites.
static inline ggml_tensor * lm_mask_alloc(ggml_context * ctx, int64_t n, bool flash) {
    return ggml_new_tensor_1d(ctx, flash ? GGML_TYPE_F16 : GGML_TYPE_F32, n);
}

// Upload a host F32 mask into `t`, converting per element when `t` is F16.
// -INFINITY survives the conversion as F16 -inf (0xFC00), which is what the
// fused op's tile-skip test looks for — the same convention dit-data.h uses.
static inline void lm_mask_set(ggml_tensor * t, const std::vector<float> & m) {
    GGML_ASSERT(t != nullptr);
    GGML_ASSERT((int64_t) m.size() <= ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, m.data(), 0, m.size() * sizeof(float));
        return;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F16);
    std::vector<ggml_fp16_t> h(m.size());
    for (size_t i = 0; i < m.size(); i++) {
        h[i] = ggml_fp32_to_fp16(m[i]);
    }
    ggml_backend_tensor_set(t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
}

// ─── graph signature (self-test T22, --attn flag-off structural identity) ───
//
// A 64-bit FNV-1a over the graph's OP SEQUENCE and every node's shape. This is
// the D2 tripwire in a form that survives without a reverted-tree baseline
// binary: with --attn exact the emitted graph must be the pre-flash graph to the
// byte, and "same op sequence, same shapes, same node count" is a check the
// trainer can make against a recorded constant on any later build.
//
// Deliberately NOT hashing tensor DATA (nothing has run) or pointers/names
// (allocation-order dependent). What it catches is the failure this flag makes
// possible: an accidentally-emitted extra node, a reordered builder, or a cast
// that only appears in one arm.
static inline uint64_t lm_graph_op_hash(ggml_cgraph * gf) {
    uint64_t   h = 1469598103934665603ull;  // FNV-1a offset basis
    auto       mix = [&h](uint64_t v) {
        for (int b = 0; b < 8; b++) {
            h ^= (v >> (b * 8)) & 0xffull;
            h *= 1099511628211ull;  // FNV-1a prime
        }
    };
    const int n = ggml_graph_n_nodes(gf);
    mix((uint64_t) n);
    for (int i = 0; i < n; i++) {
        const ggml_tensor * t = ggml_graph_node(gf, i);
        mix((uint64_t) t->op);
        mix((uint64_t) t->type);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            mix((uint64_t) t->ne[d]);
        }
    }
    return h;
}

// ─── misc ───────────────────────────────────────────────────────────────────

// Write `text` to `path` via tmp + stw_replace_file (never leaves a partial file).
static bool lm_write_atomic(const std::string & path, const std::string & text) {
    return pm_write_atomic(path, text);
}

static inline std::string lm_join(const std::string & dir, const std::string & leaf) {
    if (dir.empty()) {
        return leaf;
    }
    const char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + leaf : dir + "/" + leaf;
}

// Parent directory of `path`, or "" when it has no separator. Counterpart to
// preprocess-io.h's pm_basename(); trailing separators are not trimmed because
// every caller passes a file path.
static inline std::string lm_dirname(const std::string & path) {
    const size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? std::string() : path.substr(0, p);
}

// Read the variant's dataset-level trigger word out of <dir>/preprocess_meta.json
// (written by preprocess-io.h:515-516). Both trainers use it as the default for
// the trigger they embed in the exported adapter, so an already-preprocessed
// variant and a hand-run CLI training both get the tag for free.
//
// Returns false (and leaves both outputs untouched) when the file is missing or
// unparseable — an absent trigger is not an error, it just means no keys get
// written and the adapter stays byte-identical to a pre-trigger build.
//
// docs/plans/2026-07-28-adapter-trigger-embedding.md T5
static bool lm_read_variant_tag(const std::string & tensors_dir, std::string * tag, std::string * position) {
    const std::string p   = lm_join(tensors_dir, "preprocess_meta.json");
    yyjson_doc *      doc = yyjson_read_file(p.c_str(), 0, NULL, NULL);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }
    if (tag) {
        *tag = pm_js_str(root, "custom_tag");
    }
    if (position) {
        *position = pm_js_str(root, "tag_position");
    }
    yyjson_doc_free(doc);
    return true;
}

// "%.1f" with a '.' decimal separator regardless of locale (milestone dir names).
static inline std::string lm_fmt1(double v) {
    char b[64];
    snprintf(b, sizeof(b), "%.1f", v);
    for (size_t i = 0; b[i]; i++) {
        if (b[i] == ',') {
            b[i] = '.';
        }
    }
    return std::string(b);
}
