#pragma once
// preprocess-io.h: manifest / dataset.json parsing, per-song metadata records,
// preprocess_meta.json + channel_stats.json writers, path helpers and the
// atomic-write primitive used by `ace-train preprocess`.
//
// docs/plans/2026-07-27-preprocess-implementation.md §2.3, §2.5, §2.6, §3.3

#include "hot-step-fsutf8.h"  // hs_fopen/hs_stat/… — UTF-8 paths, not ANSI
#include "safetensors.h"
#include "yyjson.h"

#include "train/st-write.h"  // stw_replace_file — shared crash-safe rename

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <direct.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#else
#    include <sys/stat.h>
#    include <sys/types.h>
#endif

// ─── structs ────────────────────────────────────────────────────────────────

struct PSample {
    std::string id, audio_path, filename, rel_path, out_stem;
    std::string caption, genre, lyrics;
    std::string keyscale, timesignature, language;
    std::string custom_tag, tag_position, prompt_override;  // prompt_override: "" | "caption" | "genre"
    int         bpm             = 0;                        // 0 = N/A
    int         duration        = 0;                        // seconds; 0 = derive from T_latent
    int         repeat          = 1;
    bool        is_instrumental = false;
    long long   src_bytes = 0, src_mtime_ms = 0;
};

struct PManifest {
    std::string          dataset_id, slug, name, source_dir, dataset_json_path;
    std::string          custom_tag, tag_position;  // dataset-level defaults
    int                  genre_ratio = 0;
    std::vector<PSample> samples;
};

// Per-song record for preprocess_meta.json `samples[]`.
struct PSampleMeta {
    std::string id, file, rel_path, audio_path, error;
    long long   src_bytes = 0, src_mtime_ms = 0, bytes = 0;
    int         t_latent = 0, s_total = 0, s_text = 0, s_lyric = 0, s_total_genre = 0;
    bool        has_genre = false, ok = false, skipped = false;
};

struct PreprocessMeta {  // everything §2.5 needs, filled by ace-train.cpp
    std::string producer, created_at, manifest, dataset_json, audio_dir;
    std::string model_variant, dit_path, vae_path, text_encoder_path;
    std::string compat, dtype, timbre, normalize;
    double      target_db    = -1.0;
    int         max_duration = 240, max_caption_tokens = 256, max_lyric_tokens = 512;
    int         vae_chunk = 1024, vae_overlap = 64;
    std::string custom_tag, tag_position;
    int         genre_ratio = 0;
    int         total = 0, processed = 0, skipped = 0, failed = 0;
    std::vector<PSampleMeta> samples;
};

// ─── small filesystem helpers ───────────────────────────────────────────────
//
// All of these go through hot-step-fsutf8.h rather than the narrow CRT calls:
// dataset paths are UTF-8 and Windows' narrow API would decode them in the ANSI
// codepage. See that header for the album this silently ate.

static bool pm_path_exists(const std::string & path) {
    return hs_path_exists(path);
}

static bool pm_file_exists(const std::string & path) {
    return hs_file_exists(path);
}

// dataset.json stores absolute audio paths, so moving or renaming the dataset
// folder strands every one of them. Look beside dataset.json first, for the
// longest tail of the stored path that exists (sub/x.flac before x.flac). A
// pushed manifest keeps the controller's absolute paths, and on a worker those
// may resolve to a network mount, so a local copy must win over the stored
// path. Falls back to the stored path when nothing matches.
static std::string pm_resolve_audio(const std::string & audio, const std::string & ds_path) {
    if (audio.empty()) {
        return audio;
    }
    const size_t      ds  = ds_path.find_last_of("/\\");
    const std::string dir = ds == std::string::npos ? std::string() : ds_path.substr(0, ds + 1);
    for (size_t p = audio.find_first_of("/\\"); p != std::string::npos; p = audio.find_first_of("/\\", p + 1)) {
        const std::string cand = dir + audio.substr(p + 1);
        if (pm_file_exists(cand)) {
            return cand;
        }
    }
    return audio;
}

// stat() a file -> byte size + mtime in epoch ms. Returns false when absent.
static bool pm_stat_file(const std::string & path, long long * bytes, long long * mtime_ms) {
    HS_STAT_T st;
    if (hs_stat(path, &st) != 0) {
        return false;
    }
    if (bytes) {
        *bytes = (long long) st.st_size;
    }
    if (mtime_ms) {
        *mtime_ms = (long long) st.st_mtime * 1000LL;
    }
    return true;
}

// mkdir -p (single-level parents included). Returns true when the dir exists after the call.
static bool pm_mkdir_p(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }
    if (pm_path_exists(dir)) {
        return true;
    }
    std::string acc;
    for (size_t i = 0; i <= dir.size(); i++) {
        if (i == dir.size() || dir[i] == '/' || dir[i] == '\\') {
            if (!acc.empty() && acc != "." && !(acc.size() == 2 && acc[1] == ':')) {
                if (!pm_path_exists(acc)) {
                    hs_mkdir(acc);
                }
            }
        }
        if (i < dir.size()) {
            acc.push_back(dir[i]);
        }
    }
    return pm_path_exists(dir);
}

// ─── string helpers ─────────────────────────────────────────────────────────

static std::string pm_trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (unsigned char) s[a] <= ' ') {
        a++;
    }
    while (b > a && (unsigned char) s[b - 1] <= ' ') {
        b--;
    }
    return s.substr(a, b - a);
}

static std::string pm_lower(const std::string & s) {
    std::string o = s;
    for (size_t i = 0; i < o.size(); i++) {
        if (o[i] >= 'A' && o[i] <= 'Z') {
            o[i] = (char) (o[i] - 'A' + 'a');
        }
    }
    return o;
}

// Lowercase extension of a path, including the dot ("" when none).
static std::string pm_ext_of(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    size_t dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return "";
    }
    return pm_lower(path.substr(dot));
}

static std::string pm_basename(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// P6 sanitizer: strip the extension, '/'|'\\' -> "__", every character outside
// [A-Za-z0-9._-] -> a single '_', truncate to 96 chars.
//
// Non-ASCII is collapsed PER UTF-8 CODEPOINT (one '_' per character), which is
// what the §6.1 table requires and what the server's JS regex produces.
static std::string pm_safe_stem(const std::string & rel_path) {
    // 1. strip the last extension (of the basename only)
    std::string s     = rel_path;
    size_t      slash = s.find_last_of("/\\");
    size_t      dot   = s.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        s = s.substr(0, dot);
    }

    // 2. separators -> "__", then sanitize
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char) s[i];
        if (c == '/' || c == '\\') {
            out += "__";
            i++;
            continue;
        }
        if (c < 0x80) {
            bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-';
            out.push_back(keep ? (char) c : '_');
            i++;
            continue;
        }
        // multi-byte UTF-8 codepoint -> exactly one '_'
        size_t len = 1;
        if ((c & 0xE0u) == 0xC0u) {
            len = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            len = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            len = 4;
        }
        out.push_back('_');
        i += len;
    }

    // 3. truncate
    if (out.size() > 96) {
        out.resize(96);
    }
    return out;
}

static std::string pm_iso8601_utc_now() {
    time_t    now = time(NULL);
    struct tm tmv;
#ifdef _WIN32
    gmtime_s(&tmv, &now);
#else
    gmtime_r(&now, &tmv);
#endif
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ", tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return std::string(buf);
}

// Time signatures are labeled '4/4', but the ACE-Step convention everywhere at
// inference is the NUMERATOR only ('4'): translateParams sends it that way for
// every generation and the audition pins match it (2026-08-12 parity test).
// Training must condition on the same token, so every trainer-facing consumer
// (LM extract/prompt build, DiT preprocess metas) routes through this.
static inline std::string pm_timesig_numerator(const std::string & ts) {
    const size_t slash = ts.find('/');
    return slash == std::string::npos ? ts : ts.substr(0, slash);
}

// ─── yyjson accessors (tolerant: wrong type == missing) ─────────────────────

static std::string pm_js_str(yyjson_val * obj, const char * key, const char * dflt = "") {
    yyjson_val * v = obj ? yyjson_obj_get(obj, key) : NULL;
    if (v && yyjson_is_str(v)) {
        return std::string(yyjson_get_str(v));
    }
    return std::string(dflt);
}

static int pm_js_int(yyjson_val * obj, const char * key, int dflt = 0) {
    yyjson_val * v = obj ? yyjson_obj_get(obj, key) : NULL;
    if (!v) {
        return dflt;
    }
    if (yyjson_is_int(v)) {
        return (int) yyjson_get_sint(v);
    }
    if (yyjson_is_num(v)) {
        return (int) yyjson_get_num(v);
    }
    return dflt;
}

static long long pm_js_i64(yyjson_val * obj, const char * key, long long dflt = 0) {
    yyjson_val * v = obj ? yyjson_obj_get(obj, key) : NULL;
    if (!v) {
        return dflt;
    }
    if (yyjson_is_int(v)) {
        return (long long) yyjson_get_sint(v);
    }
    if (yyjson_is_num(v)) {
        return (long long) yyjson_get_num(v);
    }
    return dflt;
}

static bool pm_js_bool(yyjson_val * obj, const char * key, bool dflt = false) {
    yyjson_val * v = obj ? yyjson_obj_get(obj, key) : NULL;
    if (v && yyjson_is_bool(v)) {
        return yyjson_get_bool(v);
    }
    return dflt;
}

// ─── manifest / dataset.json parsers ────────────────────────────────────────

// preprocess_manifest.json (§2.3) — written by the Node server.
static bool pm_load_manifest(const char * path, PManifest & out, std::string & err) {
    yyjson_read_err rerr;
    yyjson_doc *    doc = yyjson_read_file(path, 0, NULL, &rerr);
    if (!doc) {
        err = std::string("cannot parse manifest ") + path + ": " + (rerr.msg ? rerr.msg : "unknown error");
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        err = "manifest root is not an object";
        yyjson_doc_free(doc);
        return false;
    }

    out                    = PManifest();
    out.dataset_id         = pm_js_str(root, "datasetId");
    out.slug               = pm_js_str(root, "slug");
    out.name               = pm_js_str(root, "name");
    out.source_dir         = pm_js_str(root, "sourceDir");
    out.dataset_json_path  = pm_js_str(root, "datasetJsonPath");
    out.custom_tag         = pm_js_str(root, "customTag");
    out.tag_position       = pm_js_str(root, "tagPosition");
    out.genre_ratio        = pm_js_int(root, "genreRatio", 0);

    yyjson_val * arr = yyjson_obj_get(root, "samples");
    if (!arr || !yyjson_is_arr(arr)) {
        err = "manifest has no samples[] array";
        yyjson_doc_free(doc);
        return false;
    }

    size_t       idx = 0, max = 0;
    yyjson_val * it;
    yyjson_arr_foreach(arr, idx, max, it) {
        if (!yyjson_is_obj(it)) {
            continue;
        }
        PSample s;
        s.id              = pm_js_str(it, "id");
        s.audio_path      = pm_js_str(it, "audioPath");
        s.filename        = pm_js_str(it, "filename");
        s.rel_path        = pm_js_str(it, "relPath");
        s.out_stem        = pm_js_str(it, "outStem");
        s.caption         = pm_js_str(it, "caption");
        s.genre           = pm_js_str(it, "genre");
        s.lyrics          = pm_js_str(it, "lyrics");
        s.keyscale        = pm_js_str(it, "keyscale");
        s.timesignature   = pm_js_str(it, "timesignature");
        s.language        = pm_js_str(it, "language");
        s.custom_tag      = pm_js_str(it, "customTag");
        s.tag_position    = pm_js_str(it, "tagPosition");
        s.prompt_override = pm_js_str(it, "promptOverride");
        s.bpm             = pm_js_int(it, "bpm", 0);
        s.duration        = pm_js_int(it, "duration", 0);
        s.repeat          = pm_js_int(it, "repeat", 1);
        s.is_instrumental = pm_js_bool(it, "isInstrumental", false);
        s.src_bytes       = pm_js_i64(it, "srcBytes", 0);
        s.src_mtime_ms    = pm_js_i64(it, "srcMtimeMs", 0);

        if (s.rel_path.empty()) {
            s.rel_path = s.filename.empty() ? pm_basename(s.audio_path) : s.filename;
        }
        if (s.filename.empty()) {
            s.filename = pm_basename(s.audio_path);
        }
        if (s.out_stem.empty()) {
            s.out_stem = pm_safe_stem(s.rel_path) + (s.id.empty() ? "" : "-" + s.id);
        }
        out.samples.push_back(s);
    }

    yyjson_doc_free(doc);
    if (out.samples.empty()) {
        err = "manifest samples[] is empty";
        return false;
    }
    return true;
}

// Side-Step dataset.json (standalone path, §2.3 "--dataset mode mapping").
static bool pm_load_dataset_json(const char * path, PManifest & out, std::string & err) {
    yyjson_read_err rerr;
    yyjson_doc *    doc = yyjson_read_file(path, 0, NULL, &rerr);
    if (!doc) {
        err = std::string("cannot parse dataset ") + path + ": " + (rerr.msg ? rerr.msg : "unknown error");
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        err = "dataset.json root is not an object";
        yyjson_doc_free(doc);
        return false;
    }

    out                   = PManifest();
    out.dataset_json_path = path;

    yyjson_val * md = yyjson_obj_get(root, "metadata");
    if (md && yyjson_is_obj(md)) {
        out.custom_tag   = pm_js_str(md, "custom_tag");
        out.tag_position = pm_js_str(md, "tag_position");
        out.genre_ratio  = pm_js_int(md, "genre_ratio", 0);
        out.name         = pm_js_str(md, "name");
        out.source_dir   = pm_js_str(md, "audio_dir");
        if (out.source_dir.empty()) {
            out.source_dir = pm_js_str(md, "source_audio_dir");
        }
    }
    if (out.source_dir.empty()) {
        std::string p     = path;
        size_t      slash = p.find_last_of("/\\");
        out.source_dir    = slash == std::string::npos ? std::string(".") : p.substr(0, slash);
    }

    yyjson_val * arr = yyjson_obj_get(root, "samples");
    if (!arr || !yyjson_is_arr(arr)) {
        err = "dataset.json has no samples[] array";
        yyjson_doc_free(doc);
        return false;
    }

    size_t       idx = 0, max = 0;
    yyjson_val * it;
    int          n = 0;
    yyjson_arr_foreach(arr, idx, max, it) {
        if (!yyjson_is_obj(it)) {
            continue;
        }
        PSample s;
        s.id              = pm_js_str(it, "id");
        s.audio_path      = pm_js_str(it, "audio_path");
        s.filename        = pm_js_str(it, "filename");
        s.caption         = pm_js_str(it, "caption");
        s.genre           = pm_js_str(it, "genre");
        s.lyrics          = pm_js_str(it, "lyrics");
        s.keyscale        = pm_js_str(it, "keyscale");
        s.timesignature   = pm_js_str(it, "timesignature");
        s.language        = pm_js_str(it, "language");
        s.custom_tag      = pm_js_str(it, "custom_tag");
        s.tag_position    = "";  // per-sample tag_position is dataset-level only in this mode
        s.prompt_override = pm_js_str(it, "prompt_override");
        s.bpm             = pm_js_int(it, "bpm", 0);
        s.duration        = pm_js_int(it, "duration", 0);
        s.repeat          = pm_js_int(it, "repeat", 1);
        s.is_instrumental = pm_js_bool(it, "is_instrumental", false);

        if (s.filename.empty()) {
            s.filename = pm_basename(s.audio_path);
        }
        if (s.audio_path.empty() && !s.filename.empty()) {
            s.audio_path = out.source_dir + "/" + s.filename;
        }
        s.rel_path = s.filename;

        char idx_b[16];
        snprintf(idx_b, sizeof(idx_b), "%04d", n);
        s.out_stem = pm_safe_stem(s.rel_path) + "-" + (s.id.empty() ? std::string(idx_b) : s.id);

        pm_stat_file(s.audio_path, &s.src_bytes, &s.src_mtime_ms);
        out.samples.push_back(s);
        n++;
    }

    yyjson_doc_free(doc);
    if (out.samples.empty()) {
        err = "dataset.json samples[] is empty";
        return false;
    }
    return true;
}

// ─── atomic write ───────────────────────────────────────────────────────────

// Write bytes to `path + ".__writing__"`, then rename over `path`.
static bool pm_write_atomic(const std::string & path, const std::string & bytes) {
    std::string tmp = path + ".__writing__";
    FILE *      f   = hs_fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "[Preprocess] cannot open %s for writing\n", tmp.c_str());
        return false;
    }
    bool ok = bytes.empty() || (fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size());
    ok      = ok && (fflush(f) == 0) && (ferror(f) == 0);
    fclose(f);
    if (!ok) {
        hs_remove(tmp);
        return false;
    }
    if (!stw_replace_file(tmp.c_str(), path.c_str())) {
        fprintf(stderr, "[Preprocess] rename %s -> %s failed (previous file left intact)\n", tmp.c_str(), path.c_str());
        hs_remove(tmp);
        return false;
    }
    return true;
}

// ─── preprocess_meta.json (§2.5) ────────────────────────────────────────────

static bool pm_write_meta_json(const char * path, const PreprocessMeta & m) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "format", "hot-step-preprocess-v1");
    yyjson_mut_obj_add_strcpy(doc, root, "producer", m.producer.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "created_at", m.created_at.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "manifest", m.manifest.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "dataset_json", m.dataset_json.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "audio_dir", m.audio_dir.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "source_audio_dir", m.audio_dir.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "model_variant", m.model_variant.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "dit_path", m.dit_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "vae_path", m.vae_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "text_encoder_path", m.text_encoder_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "compat", m.compat.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "dtype", m.dtype.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "timbre", m.timbre.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "normalize", m.normalize.c_str());
    yyjson_mut_obj_add_real(doc, root, "target_db", m.target_db);
    yyjson_mut_obj_add_real(doc, root, "target_lufs", -14.0);
    yyjson_mut_obj_add_int(doc, root, "max_duration", m.max_duration);
    yyjson_mut_obj_add_int(doc, root, "max_caption_tokens", m.max_caption_tokens);
    yyjson_mut_obj_add_int(doc, root, "max_lyric_tokens", m.max_lyric_tokens);
    yyjson_mut_obj_add_int(doc, root, "vae_chunk", m.vae_chunk);
    yyjson_mut_obj_add_int(doc, root, "vae_overlap", m.vae_overlap);
    yyjson_mut_obj_add_real(doc, root, "latent_fps", 25.0);
    yyjson_mut_obj_add_strcpy(doc, root, "custom_tag", m.custom_tag.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "tag_position", m.tag_position.c_str());
    yyjson_mut_obj_add_int(doc, root, "genre_ratio", m.genre_ratio);
    yyjson_mut_obj_add_int(doc, root, "total", m.total);
    yyjson_mut_obj_add_int(doc, root, "processed", m.processed);
    yyjson_mut_obj_add_int(doc, root, "skipped", m.skipped);
    yyjson_mut_obj_add_int(doc, root, "failed", m.failed);

    yyjson_mut_val * arr = yyjson_mut_obj_add_arr(doc, root, "samples");
    for (size_t i = 0; i < m.samples.size(); i++) {
        const PSampleMeta & s = m.samples[i];
        yyjson_mut_val *    e = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, e, "id", s.id.c_str());
        yyjson_mut_obj_add_strcpy(doc, e, "file", s.file.c_str());
        yyjson_mut_obj_add_strcpy(doc, e, "relPath", s.rel_path.c_str());
        yyjson_mut_obj_add_strcpy(doc, e, "audioPath", s.audio_path.c_str());
        yyjson_mut_obj_add_sint(doc, e, "srcBytes", s.src_bytes);
        yyjson_mut_obj_add_sint(doc, e, "srcMtimeMs", s.src_mtime_ms);
        yyjson_mut_obj_add_int(doc, e, "tLatent", s.t_latent);
        yyjson_mut_obj_add_int(doc, e, "sTotal", s.s_total);
        yyjson_mut_obj_add_int(doc, e, "sText", s.s_text);
        yyjson_mut_obj_add_int(doc, e, "sLyric", s.s_lyric);
        yyjson_mut_obj_add_int(doc, e, "sTotalGenre", s.s_total_genre);
        yyjson_mut_obj_add_bool(doc, e, "hasGenre", s.has_genre);
        yyjson_mut_obj_add_sint(doc, e, "bytes", s.bytes);
        yyjson_mut_obj_add_bool(doc, e, "ok", s.ok);
        yyjson_mut_obj_add_bool(doc, e, "skipped", s.skipped);
        yyjson_mut_obj_add_strcpy(doc, e, "error", s.error.c_str());
        yyjson_mut_arr_add_val(arr, e);
    }

    size_t len  = 0;
    char * text = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY | YYJSON_WRITE_ALLOW_INVALID_UNICODE, &len);
    yyjson_mut_doc_free(doc);
    if (!text) {
        fprintf(stderr, "[Preprocess] failed to serialize %s\n", path);
        return false;
    }
    std::string bytes(text, len);
    free(text);
    return pm_write_atomic(path, bytes);
}

// ─── channel_stats.json (§2.6) ──────────────────────────────────────────────

static bool pm_write_channel_stats(const char * path, const double * sum, const double * sqsum, long long n_frames,
                                   int n_samples) {
    if (n_frames <= 0) {
        return false;
    }
    std::string out = "{\n  \"channel_mean\": [";
    char        buf[64];
    double      mean[64], stdv[64];
    for (int c = 0; c < 64; c++) {
        mean[c]    = sum[c] / (double) n_frames;
        double var = sqsum[c] / (double) n_frames - mean[c] * mean[c];
        if (var < 1e-10) {
            var = 1e-10;  // biased variance, NOT Bessel-corrected
        }
        stdv[c] = sqrt(var);
    }
    for (int c = 0; c < 64; c++) {
        snprintf(buf, sizeof(buf), "%.17g", mean[c]);
        out += (c ? ", " : "");
        out += buf;
    }
    out += "],\n  \"channel_std\": [";
    for (int c = 0; c < 64; c++) {
        snprintf(buf, sizeof(buf), "%.17g", stdv[c]);
        out += (c ? ", " : "");
        out += buf;
    }
    snprintf(buf, sizeof(buf), "],\n  \"n_samples\": %d,\n", n_samples);
    out += buf;
    snprintf(buf, sizeof(buf), "  \"n_frames\": %lld\n}\n", n_frames);
    out += buf;
    return pm_write_atomic(path, out);
}

// ─── cached-latent readback (skipped songs still feed channel stats) ────────

static float pm_bf16_to_f32(uint16_t h) {
    uint32_t b = (uint32_t) h << 16;
    float    f;
    memcpy(&f, &b, 4);
    return f;
}

// Read `target_latents` back out of an existing cache file, plus the
// __metadata__ ints that preprocess_meta.json needs for a skipped entry.
static bool pm_read_cached_latents(const char * path, std::vector<float> & out, int * T_out, PSampleMeta * meta_out) {
    STFile st;
    if (!st_open(&st, path)) {
        return false;
    }
    const STEntry * e = st_find(st, "target_latents");
    if (!e || e->n_dims != 2 || e->shape[1] != 64) {
        st_close(&st);
        return false;
    }
    const int   T = (int) e->shape[0];
    const void * d = st_data(st, *e);
    if (!d || T <= 0) {
        st_close(&st);
        return false;
    }
    out.resize((size_t) T * 64);
    if (e->dtype == "F32") {
        memcpy(out.data(), d, (size_t) T * 64 * sizeof(float));
    } else if (e->dtype == "BF16") {
        const uint16_t * src = (const uint16_t *) d;
        for (size_t i = 0; i < (size_t) T * 64; i++) {
            out[i] = pm_bf16_to_f32(src[i]);
        }
    } else if (e->dtype == "F16") {
        const uint16_t * src = (const uint16_t *) d;
        for (size_t i = 0; i < (size_t) T * 64; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        st_close(&st);
        return false;
    }

    if (T_out) {
        *T_out = T;
    }

    // Sequence lengths + genre presence come from the header's __metadata__,
    // which safetensors.h deliberately skips. Re-read the header with yyjson.
    if (meta_out) {
        meta_out->t_latent  = T;
        meta_out->has_genre = (st_find(st, "encoder_hidden_states_genre") != NULL);
        const STEntry * eh  = st_find(st, "encoder_hidden_states");
        if (eh && eh->n_dims == 2) {
            meta_out->s_total = (int) eh->shape[0];
        }
        const STEntry * ehg = st_find(st, "encoder_hidden_states_genre");
        if (ehg && ehg->n_dims == 2) {
            meta_out->s_total_genre = (int) ehg->shape[0];
        }

        FILE * f = hs_fopen(path, "rb");
        if (f) {
            uint8_t hl_le[8];
            if (fread(hl_le, 1, 8, f) == 8) {
                uint64_t hl = 0;
                for (int i = 7; i >= 0; i--) {
                    hl = (hl << 8) | hl_le[i];
                }
                if (hl > 0 && hl < (64u << 20)) {
                    std::string hdr((size_t) hl, '\0');
                    if (fread(&hdr[0], 1, (size_t) hl, f) == (size_t) hl) {
                        yyjson_doc * doc = yyjson_read(hdr.data(), hdr.size(), 0);
                        if (doc) {
                            yyjson_val * root = yyjson_doc_get_root(doc);
                            yyjson_val * md   = root ? yyjson_obj_get(root, "__metadata__") : NULL;
                            if (md && yyjson_is_obj(md)) {
                                std::string v;
                                v = pm_js_str(md, "s_text");
                                if (!v.empty()) {
                                    meta_out->s_text = atoi(v.c_str());
                                }
                                v = pm_js_str(md, "s_lyric");
                                if (!v.empty()) {
                                    meta_out->s_lyric = atoi(v.c_str());
                                }
                            }
                            yyjson_doc_free(doc);
                        }
                    }
                }
            }
            fclose(f);
        }
    }

    st_close(&st);
    return true;
}

// ─── cached-settings readback (resume must not lie about provenance) ────────

// The settings a cache file was actually produced with, from its __metadata__.
// Skipping purely on filename existence lets preprocess_meta.json attribute the
// CURRENT run's compat/dtype/timbre/max_duration to files built with different
// ones, and the server reports that meta to the UI as provenance. The resume
// path compares these and re-encodes on a mismatch.
struct PCacheSettings {
    std::string compat, dtype, timbre, normalize, model_variant, target_db;
    // The trigger pair is part of the CAPTION the text encoder baked into this
    // file (preprocess-run.h pp_caption_text), so a change to either has to
    // re-encode. Without them, flipping tag_position prepend→replace left every
    // cached file "matching" and the run silently trained on the old captions
    // while preprocess_meta.json advertised the new setting (2026-08-12).
    std::string custom_tag, tag_position;
    int         max_duration = -1, max_caption_tokens = -1, max_lyric_tokens = -1;
    int         vae_chunk = -1, vae_overlap = -1;
    bool        valid = false;  // false = no readable __metadata__ (legacy/corrupt file)
};

static bool pm_read_cached_settings(const char * path, PCacheSettings & s) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    uint8_t hl_le[8];
    if (fread(hl_le, 1, 8, f) != 8) {
        fclose(f);
        return false;
    }
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) {
        hl = (hl << 8) | hl_le[i];
    }
    if (hl == 0 || hl >= (64u << 20)) {
        fclose(f);
        return false;
    }
    std::string hdr((size_t) hl, '\0');
    const bool  got = fread(&hdr[0], 1, (size_t) hl, f) == (size_t) hl;
    fclose(f);
    if (!got) {
        return false;
    }
    yyjson_doc * doc = yyjson_read(hdr.data(), hdr.size(), 0);
    if (!doc) {
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md   = root ? yyjson_obj_get(root, "__metadata__") : NULL;
    if (md && yyjson_is_obj(md)) {
        s.compat        = pm_js_str(md, "compat");
        s.dtype         = pm_js_str(md, "dtype");
        s.timbre        = pm_js_str(md, "timbre");
        s.normalize     = pm_js_str(md, "normalize");
        s.model_variant = pm_js_str(md, "model_variant");
        s.target_db     = pm_js_str(md, "target_db");
        s.custom_tag    = pm_js_str(md, "custom_tag");
        s.tag_position  = pm_js_str(md, "tag_position");

        std::string v;
        v = pm_js_str(md, "max_duration");
        if (!v.empty()) s.max_duration = atoi(v.c_str());
        v = pm_js_str(md, "max_caption_tokens");
        if (!v.empty()) s.max_caption_tokens = atoi(v.c_str());
        v = pm_js_str(md, "max_lyric_tokens");
        if (!v.empty()) s.max_lyric_tokens = atoi(v.c_str());
        v = pm_js_str(md, "vae_chunk");
        if (!v.empty()) s.vae_chunk = atoi(v.c_str());
        v = pm_js_str(md, "vae_overlap");
        if (!v.empty()) s.vae_overlap = atoi(v.c_str());

        s.valid = !s.compat.empty() || !s.dtype.empty() || s.max_duration >= 0;
    }
    yyjson_doc_free(doc);
    return s.valid;
}
