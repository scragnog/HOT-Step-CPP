#pragma once
// lm-extract.h — stage 1 of `ace-train train-lm`.
//
// preprocess variant dir (per-song safetensors with `target_latents [T,64] F32`
// @25 Hz + a full __metadata__ block)  ->  5 Hz FSQ codes  ->  lm_codes.jsonl.
//
// Zero audio decode, zero VAE work: `tok_ggml_encode` consumes exactly the
// tensor preprocess already wrote (L13), and every prompt field the trainer
// needs is already in that file's __metadata__ (L14).
//
// Cache policy (L15): a song is REUSED iff its `_id`, `_st_bytes`,
// `_st_mtime_ms` and `_tok` all match the tensor file on disk. The whole
// jsonl is then rewritten atomically in preprocess_meta.json sample order.
//
// docs/plans/2026-07-27-lm-trainer-implementation.md §2.3, §3.2

#include "fsq-tok.h"
#include "gguf-weights.h"
#include "silence-latent.h"
#include "train/lm-common.h"
#include "version.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

// ─── one lm_codes.jsonl row (§2.3) ──────────────────────────────────────────

struct LmCodeRow {
    std::string      file, caption, lyrics, keyscale, timesignature, language;
    int              bpm = 0, duration = 0;
    // The preprocess hard-cut ate this song's real ending (duration reached
    // --max-duration). The trainer then must NOT teach im_end at the cut —
    // that is a lie about where songs end (2026-08-29; a third of the corpus
    // carried it at the old 240 s cap).
    bool             truncated = false;
    std::vector<int> codes;
    std::string      id, tok;  // _id, _tok
    long long        st_bytes = 0, st_mtime_ms = 0;
};

struct LmExtractOpts {
    std::string tensors_dir;  // preprocess variant dir
    std::string codes_path;   // <tensors>/lm_codes.jsonl unless overridden
    std::string dit_path;     // resolved; defaults to preprocess_meta.json's dit_path
    bool        overwrite = false;
    int         limit     = 0;
};

// ─── jsonl serialization ────────────────────────────────────────────────────

static std::string lm_codes_row_to_json(const LmCodeRow & r) {
    std::string o;
    o.reserve(r.codes.size() * 6 + r.caption.size() + r.lyrics.size() + 256);
    o += "{\"file\":\"" + lm_json_escape(r.file) + "\"";
    o += ",\"caption\":\"" + lm_json_escape(r.caption) + "\"";
    o += ",\"lyrics\":\"" + lm_json_escape(r.lyrics) + "\"";
    char b[64];
    snprintf(b, sizeof(b), ",\"bpm\":%d", r.bpm);
    o += b;
    o += ",\"keyscale\":\"" + lm_json_escape(r.keyscale) + "\"";
    o += ",\"timesignature\":\"" + lm_json_escape(r.timesignature) + "\"";
    o += ",\"language\":\"" + lm_json_escape(r.language) + "\"";
    snprintf(b, sizeof(b), ",\"duration\":%d", r.duration);
    o += b;
    if (r.truncated) {
        o += ",\"truncated\":true";
    }
    o += ",\"codes\":[";
    for (size_t i = 0; i < r.codes.size(); i++) {
        if (i) {
            o.push_back(',');
        }
        snprintf(b, sizeof(b), "%d", r.codes[i]);
        o += b;
    }
    o += "]";
    o += ",\"_id\":\"" + lm_json_escape(r.id) + "\"";
    snprintf(b, sizeof(b), ",\"_st_bytes\":%lld", r.st_bytes);
    o += b;
    snprintf(b, sizeof(b), ",\"_st_mtime_ms\":%lld", r.st_mtime_ms);
    o += b;
    o += ",\"_tok\":\"" + lm_json_escape(r.tok) + "\"}";
    return o;
}

static bool lm_codes_parse_line(const char * data, size_t len, LmCodeRow * out) {
    yyjson_doc * doc = yyjson_read(data, len, 0);
    if (!doc) {
        return false;
    }
    yyjson_val * r = yyjson_doc_get_root(doc);
    if (!r || !yyjson_is_obj(r)) {
        yyjson_doc_free(doc);
        return false;
    }
    out->file          = pm_js_str(r, "file");
    out->caption       = pm_js_str(r, "caption");
    out->lyrics        = pm_js_str(r, "lyrics");
    out->keyscale      = pm_js_str(r, "keyscale");
    out->timesignature = pm_js_str(r, "timesignature");
    out->language      = pm_js_str(r, "language");
    out->bpm           = pm_js_int(r, "bpm", 0);
    out->duration      = pm_js_int(r, "duration", 0);
    out->truncated     = pm_js_bool(r, "truncated", false);
    out->id            = pm_js_str(r, "_id");
    out->tok           = pm_js_str(r, "_tok");
    out->st_bytes      = pm_js_i64(r, "_st_bytes", 0);
    out->st_mtime_ms   = pm_js_i64(r, "_st_mtime_ms", 0);

    out->codes.clear();
    yyjson_val * c = yyjson_obj_get(r, "codes");
    if (c && yyjson_is_arr(c)) {
        out->codes.reserve(yyjson_arr_size(c));
        size_t       i, m;
        yyjson_val * v;
        yyjson_arr_foreach(c, i, m, v) {
            out->codes.push_back((int) yyjson_get_int(v));
        }
    }
    yyjson_doc_free(doc);
    return !out->codes.empty();
}

// Read a whole lm_codes.jsonl. Missing file -> true with an empty vector.
static bool lm_codes_read_file(const char * path, std::vector<LmCodeRow> * out, std::string * err) {
    out->clear();
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return true;
    }
    std::string line;
    char        buf[65536];
    int         bad = 0, idx = 0;
    while (fgets(buf, sizeof(buf), f)) {
        line += buf;
        if (line.empty() || line[line.size() - 1] != '\n') {
            continue;  // long line, keep reading
        }
        while (!line.empty() && (line[line.size() - 1] == '\n' || line[line.size() - 1] == '\r')) {
            line.resize(line.size() - 1);
        }
        if (!line.empty()) {
            LmCodeRow r;
            if (lm_codes_parse_line(line.data(), line.size(), &r)) {
                out->push_back(r);
            } else {
                bad++;
            }
        }
        idx++;
        line.clear();
    }
    if (!line.empty()) {  // last line with no trailing newline
        LmCodeRow r;
        if (lm_codes_parse_line(line.data(), line.size(), &r)) {
            out->push_back(r);
        } else {
            bad++;
        }
    }
    fclose(f);
    if (bad > 0 && err) {
        char b[160];
        snprintf(b, sizeof(b), "%d unparseable row(s) in %s (ignored)", bad, path);
        *err = b;
    }
    (void) idx;
    return true;
}

static bool lm_codes_write_file(const std::string & path, const std::vector<LmCodeRow> & rows) {
    std::string all;
    size_t      est = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        est += rows[i].codes.size() * 6 + rows[i].caption.size() + rows[i].lyrics.size() + 512;
    }
    all.reserve(est);
    for (size_t i = 0; i < rows.size(); i++) {
        all += lm_codes_row_to_json(rows[i]);
        all.push_back('\n');
    }
    return pm_write_atomic(path, all);
}

// ─── preprocess_meta.json reader (order + file/id list) ─────────────────────

struct LmMetaSample {
    std::string id, file;
    bool        ok = false;
};

struct LmPreprocessMeta {
    std::string               dit_path, model_variant;
    std::vector<LmMetaSample> samples;
};

static bool lm_read_preprocess_meta(const std::string & tensors_dir, LmPreprocessMeta * out, std::string * err) {
    const std::string p   = lm_join(tensors_dir, "preprocess_meta.json");
    yyjson_doc *      doc = yyjson_read_file(p.c_str(), 0, NULL, NULL);
    if (!doc) {
        *err = "cannot read " + p;
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        *err = p + " is not a JSON object";
        return false;
    }
    out->dit_path      = pm_js_str(root, "dit_path");
    out->model_variant = pm_js_str(root, "model_variant");

    yyjson_val * arr = yyjson_obj_get(root, "samples");
    if (arr && yyjson_is_arr(arr)) {
        size_t       i, m;
        yyjson_val * v;
        yyjson_arr_foreach(arr, i, m, v) {
            if (!yyjson_is_obj(v)) {
                continue;
            }
            LmMetaSample s;
            s.id   = pm_js_str(v, "id");
            s.file = pm_js_str(v, "file");
            s.ok   = pm_js_bool(v, "ok", false);
            out->samples.push_back(s);
        }
    }
    yyjson_doc_free(doc);
    if (out->samples.empty()) {
        *err = p + " lists no samples";
        return false;
    }
    return true;
}

// ─── stage driver ───────────────────────────────────────────────────────────

static bool lm_extract_run(const LmExtractOpts & o, std::string * err) {
    const int64_t t_stage0 = ggml_time_ms();

    LmPreprocessMeta meta;
    if (!lm_read_preprocess_meta(o.tensors_dir, &meta, err)) {
        return false;
    }

    std::string dit_path = o.dit_path.empty() ? meta.dit_path : o.dit_path;
    if (dit_path.empty() || !pm_path_exists(dit_path)) {
        *err = "cannot resolve the DiT (FSQ tokenizer source): '" + dit_path + "'";
        return false;
    }
    const std::string tok_id = lm_tok_identity(dit_path);

    int total = (int) meta.samples.size();
    if (o.limit > 0 && o.limit < total) {
        total = o.limit;
    }

    // existing codes, keyed by _id
    std::vector<LmCodeRow> cached;
    std::string            cache_warn;
    lm_codes_read_file(o.codes_path.c_str(), &cached, &cache_warn);
    if (!cache_warn.empty()) {
        lm_log("warn", cache_warn);
    }
    std::map<std::string, size_t> by_id;
    for (size_t i = 0; i < cached.size(); i++) {
        if (!cached[i].id.empty()) {
            by_id[cached[i].id] = i;
        }
    }

    jl("{\"type\":\"stage\",\"stage\":\"extract\",\"state\":\"begin\",\"total\":%d}", total);
    fprintf(stderr, "[train-lm] extract: %d songs from %s\n", total, o.tensors_dir.c_str());

    // ── FSQ tokenizer + silence_latent ──────────────────────────────────
    const int64_t t_tok0 = ggml_time_ms();
    TokGGML       tok    = {};
    if (!tok_ggml_load(&tok, dit_path.c_str())) {
        *err = "cannot load the FSQ tokenizer from " + dit_path;
        return false;
    }
    std::vector<float> silence;
    if (ends_with_gguf(dit_path.c_str())) {
        GGUFModel gf = {};
        if (!gf_load(&gf, dit_path.c_str())) {
            tok_ggml_free(&tok);
            *err = "cannot open DiT " + dit_path;
            return false;
        }
        const void * sl = gf_get_data(gf, "silence_latent");
        if (!sl) {
            gf_close(&gf);
            tok_ggml_free(&tok);
            *err = "silence_latent missing from " + dit_path;
            return false;
        }
        silence.assign((const float *) sl, (const float *) sl + (size_t) 15000 * 64);
        gf_close(&gf);
    } else {
        const std::string pt = dit_path + WS_SEP + "silence_latent.pt";
        if (!sl_read_silence_latent(pt.c_str(), silence)) {
            tok_ggml_free(&tok);
            *err = "cannot read " + pt;
            return false;
        }
    }
    if (silence.size() < 64 * 5) {
        tok_ggml_free(&tok);
        *err = "silence_latent is too small in " + dit_path;
        return false;
    }
    jl("{\"type\":\"model\",\"stage\":\"fsq-tok\",\"path\":\"%s\",\"ms\":%lld}", lm_json_escape(dit_path).c_str(),
       (long long) (ggml_time_ms() - t_tok0));

    // ── per-song loop ───────────────────────────────────────────────────
    std::vector<LmCodeRow> rows;
    std::vector<int>       row_idx;  // meta.samples index each produced row came from
    rows.reserve((size_t) total);
    row_idx.reserve((size_t) total);
    int processed = 0, skipped = 0, failed = 0;

    auto emit_progress = [&](void) {
        jl("{\"type\":\"progress\",\"completed\":%d,\"total\":%d,\"phase\":\"extract\",\"processed\":%d,"
           "\"skipped\":%d,\"failed\":%d}",
           processed + skipped + failed, total, processed, skipped, failed);
    };

    for (int i = 0; i < total; i++) {
        const LmMetaSample & s = meta.samples[(size_t) i];
        if (!s.ok || s.file.empty()) {
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "no tensor cache for this song");
            emit_progress();
            continue;
        }

        const std::string path = lm_join(o.tensors_dir, s.file);
        long long         bytes = 0, mtime = 0;
        if (!pm_stat_file(path, &bytes, &mtime)) {
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "tensor file missing");
            emit_progress();
            continue;
        }

        if (!o.overwrite) {
            auto it = by_id.find(s.id);
            if (it != by_id.end()) {
                const LmCodeRow & c = cached[it->second];
                if (c.st_bytes == bytes && c.st_mtime_ms == mtime && c.tok == tok_id && !c.codes.empty()) {
                    rows.push_back(c);
                    row_idx.push_back(i);
                    skipped++;
                    jl("{\"type\":\"extract_skip\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"reason\":\"cached\"}",
                       i, lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str());
                    emit_progress();
                    continue;
                }
            }
        }

        const int64_t t0 = ggml_time_ms();

        STFile st = {};
        if (!st_open(&st, path.c_str())) {
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "cannot open safetensors");
            emit_progress();
            continue;
        }
        const STEntry * e = st_find(st, "target_latents");
        if (!e || e->n_dims != 2 || e->shape[1] != 64) {
            st_close(&st);
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "target_latents missing or wrong shape");
            emit_progress();
            continue;
        }

        const int    T = (int) e->shape[0];
        const void * d = st_data(st, *e);
        if (!d || T <= 0) {
            st_close(&st);
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "target_latents has no data");
            emit_progress();
            continue;
        }

        std::vector<float> lat((size_t) T * 64);
        if (e->dtype == "F32") {
            memcpy(lat.data(), d, lat.size() * sizeof(float));
        } else if (e->dtype == "BF16") {
            const uint16_t * u = (const uint16_t *) d;
            for (size_t k = 0; k < lat.size(); k++) {
                lat[k] = pm_bf16_to_f32(u[k]);
            }
        } else {
            st_close(&st);
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"unsupported "
               "target_latents dtype %s\"}",
               i, lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), lm_json_escape(e->dtype).c_str());
            emit_progress();
            continue;
        }
        st_close(&st);

        LmCodeRow row;
        row.codes.assign((size_t) ((T + 4) / 5), 0);
        // tok_ggml_encode's float -> level step is the reference-conformant
        // ResidualFSQ(preserve_symmetry, bound_hard_clamp) path (fsq-quant.h),
        // shared with the inference tokenizer since the FSQ conformance fix.
        const int n = tok_ggml_encode(&tok, lat.data(), T, row.codes.data(), silence.data());
        if (n <= 0 || n != (int) row.codes.size()) {
            failed++;
            jl("{\"type\":\"extract_fail\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"error\":\"%s\"}", i,
               lm_json_escape(s.id).c_str(), lm_json_escape(s.file).c_str(), "FSQ encode failed");
            emit_progress();
            continue;
        }

        std::map<std::string, std::string> md;
        stmd_read(path.c_str(), &md);
        row.file          = s.file;
        row.caption       = lm_apply_tag(lm_md_get(md, "caption"), lm_md_get(md, "custom_tag"),
                                         lm_md_get(md, "tag_position"));
        row.lyrics        = lm_md_get(md, "lyrics");
        row.keyscale      = lm_md_get(md, "keyscale");
        // Numerator only ('4/4' → '4') — matches inference conditioning
        // (2026-08-12 parity test); same class of fix as the language line.
        row.timesignature = pm_timesig_numerator(lm_md_get(md, "timesignature"));
        // Normalized: the preprocess __metadata__ this reads carries the
        // dataset's raw default_language, which was 'english' corpus-wide.
        row.language      = lm_normalize_language(lm_md_get(md, "language"));
        row.bpm           = atoi(lm_md_get(md, "bpm").c_str());
        row.duration      = atoi(lm_md_get(md, "duration").c_str());
        // duration is llround(T/25) of the STORED latent, so a capped song
        // lands exactly on max_duration. A song naturally that long is a
        // harmless false positive (its ending is at the cut anyway).
        {
            const int cap = atoi(lm_md_get(md, "max_duration").c_str());
            row.truncated = cap > 0 && row.duration >= cap;
        }
        row.id            = s.id.empty() ? lm_md_get(md, "sample_id") : s.id;
        row.tok           = tok_id;
        row.st_bytes      = bytes;
        row.st_mtime_ms   = mtime;
        rows.push_back(row);
        row_idx.push_back(i);
        processed++;

        jl("{\"type\":\"extract_song\",\"index\":%d,\"id\":\"%s\",\"file\":\"%s\",\"tLatent\":%d,\"nCodes\":%d,"
           "\"cached\":false,\"ms\":%lld}",
           i, lm_json_escape(row.id).c_str(), lm_json_escape(row.file).c_str(), T, n,
           (long long) (ggml_time_ms() - t0));
        emit_progress();
    }

    tok_ggml_free(&tok);
    silence.clear();
    silence.shrink_to_fit();

    // --limit is documented as a debug knob ("first n songs only", §2.1), NOT as
    // a cache-truncating operation: rewriting the file from `rows` alone would
    // silently delete every previously extracted row past the limit (§2.3 says
    // the file holds all rows in preprocess_meta.json order). Merge what this run
    // produced back over the cache, keeping that order.
    std::vector<LmCodeRow> final_rows;
    if (o.limit > 0 && total < (int) meta.samples.size()) {
        size_t k = 0;
        for (size_t i = 0; i < meta.samples.size(); i++) {
            if (k < row_idx.size() && row_idx[k] == (int) i) {
                final_rows.push_back(rows[k]);
                k++;
                continue;
            }
            if ((int) i < total) {
                continue;  // inside the limit and produced nothing — a real failure
            }
            auto it = by_id.find(meta.samples[i].id);
            if (it != by_id.end() && !cached[it->second].codes.empty()) {
                final_rows.push_back(cached[it->second]);
            }
        }
    } else {
        final_rows.swap(rows);
    }

    if (!lm_codes_write_file(o.codes_path, final_rows)) {
        *err = "cannot write " + o.codes_path;
        return false;
    }

    jl("{\"type\":\"stage\",\"stage\":\"extract\",\"state\":\"end\",\"total\":%d,\"processed\":%d,\"cached\":%d,"
       "\"failed\":%d,\"codes\":\"%s\",\"ms\":%lld}",
       total, processed, skipped, failed, lm_json_escape(o.codes_path).c_str(),
       (long long) (ggml_time_ms() - t_stage0));
    fprintf(stderr, "[train-lm] extract done: %d encoded, %d cached, %d failed -> %s\n", processed, skipped, failed,
            o.codes_path.c_str());
    return true;
}
