#pragma once
// train/yue2-sheet-run.h — `ace-train yue2-sheet`: fill the `abc` / `abc_error`
// slot with SheetSage2 lead-sheet transcriptions of each source's own audio.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Structural precedent:
// train/yue2-tokenize-run.h and train/yue2-align-run.h — same shape, walk the
// manifest's sources[], produce one artefact per source, rewrite the manifest
// in place. Read those files' headers first if this one's is not enough.
//
// AUTHORITY: docs/plans/yue2/23-sheetsage2-progress.md (state, hard rules) and
// docs/plans/yue2/19-sheetsage2-cot-training.md ("Engine plan", the
// `ace-train yue2-sheet` bullet, and the soft-failure decision). The actual
// transcription work — encode, grammar-masked decode, stitching, notation —
// is `yue2/sheetsage-pipeline.h`'s `sheetsage_transcribe()`; this file is
// nothing but the manifest walk, the decode-to-24kHz-mono step (mirroring
// yue2-tokenize-run.h's own decode block byte for byte), and the per-source
// atomic manifest rewrite.
//
// ── What it does ────────────────────────────────────────────────────────────
//
//   read <out>/yue2_preprocess.json
//   load the SheetSage2 GGUF ONCE (exact=true unless --fast)
//   for every SOURCE without "abc"/"abc_error" (or all with --force):
//       decode                (the same two routes yue2-tokenize uses)
//       mean the two channels -> mono, resample 48 kHz -> 24 kHz
//       sheetsage_transcribe  -> {ok, abc, error, windows, ms_*}
//       write into that source's row: "abc" on success, else "abc_error",
//       plus "abc_producer"/"abc_windows"/"abc_ms"
//       REWRITE THE WHOLE MANIFEST, atomically, before touching the next
//       source — see "Crash safety" below.
//
// ── Soft failure is expected and is not this tool failing ──────────────────
//
// Doc 19's decisions section: the reference can decode a track and still fail
// to render ABC (a chord/melody interval too short for the notation grid, no
// key decoded, etc). `sheetsage_transcribe()` reports that as
// `{ok=false, error="..."}`, never a thrown exception, and this tool's job is
// simply to file that error text into "abc_error" instead of "abc" — a
// source with "abc_error" trains as `off` in the 50/50 draw (phase 4, not
// this file's concern) and the run's own exit code stays 0. Doc 23's ABC
// failure survey measured this at roughly 4% of real tracks; a run that hits
// exactly that rate has done nothing wrong.
//
// A source that could not even be DECODED (missing file, unreadable codec,
// resample failure) is a different class of problem — infrastructure, not
// content — and is deliberately NOT written as "abc_error": that field is
// reserved for the reference's own typed failure text (doc 19's "the engine
// reproduces the reference's error TEXT ... internally use typed error
// codes, not string matching"), and a decode failure has no such text to
// reproduce. It is logged loudly and left for a re-run instead, exactly as
// yue2-tokenize-run.h leaves an undecodable source without codec_ids.
//
// ── Crash safety ─────────────────────────────────────────────────────────────
//
// Unlike yue2-tokenize (one manifest rewrite at the very end, because every
// source's cost is a couple of seconds), a single SheetSage2 transcription
// can run for MINUTES (doc 23's G6 table: 5-13 minutes per long track on the
// CPU backend). Losing that work to a crash three sources later, because the
// manifest was only ever going to be rewritten once at the end, would be the
// wrong trade here. So this tool re-reads the manifest fresh and rewrites it
// atomically (temp + rename, `pm_write_atomic`) after EVERY source, success
// or soft failure alike. A `--force` re-run or a crash mid-run always leaves
// the manifest in a valid, fully-parseable state that names exactly which
// sources are done.
//
// ── Fields this tool owns ────────────────────────────────────────────────────
//
// Per source: "abc" (the ABC text) XOR "abc_error" (the reference-shaped
// error text) — never both at once, "abc_producer" (e.g.
// "sheetsage2-f16.gguf exact cuda" — model file, precision mode, backend),
// "abc_windows" (the window count `sheetsage_transcribe` used),
// "abc_ms" (total wall time for that one source — decode + resample +
// transcribe, i.e. what a re-run would cost again on a cache miss).
// Root: "abc_present" (true once any source has been processed),
// "abc_producer" (this TOOL's own identity, "ace-train yue2-sheet <version>",
// distinct from the per-source model/precision/backend tag above — same
// two-producer-fields convention yue2-tokenize-run.h uses for
// "codec_ids_producer" vs its per-tensor tokenizer name), "abc_created_at",
// "abc_sources_ok" / "abc_sources_failed" (recomputed by scanning every
// source's CURRENT "abc"/"abc_error" state on every write, so these two
// counts are always right even after a partial run, a --force re-run of one
// song, or an --only slice).

#include "audio-resample.h"       // audio_resample (48 kHz -> 24 kHz)
#include "hot-step-fsutf8.h"      // hs_fopen / hs_remove — UTF-8 paths, not ANSI
#include "version.h"              // ACE_VERSION, stamped into the rewritten manifest
#include "yyjson.h"

#include "train/preprocess-io.h"        // pm_* path/string/atomic-write helpers
#include "train/yue2-preprocess-run.h"  // yp_read_native_48k / yp_read_ffmpeg / yp_contains_ci
#include "yue2/sheetsage-pipeline.h"    // sheetsage_transcribe() + everything it composes
#include "yue2/yue2-model.h"            // yue2_list_dir / yue2_quant_rank / YUE2_SEP / yue2_basename

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Arguments ────────────────────────────────────────────────────────────────

struct Yue2SheetArgs {
    std::string manifest;    // --manifest <yue2_preprocess.json>   (required)
    std::string model_path;  // --model <sheetsage2-*.gguf>
    std::string models_dir;  // --models <dir>, searched for sheetsage2-*.gguf
    std::string only;        // --only <substr>, case-insensitive, on the source name
    int         threads = 0; // --threads N, forwarded to SheetSageTranscribeOptions::threads
    bool        force   = false;  // --force, re-transcribe sources that already have abc/abc_error
    bool        fast    = false;  // --fast, load_opt.exact=false (default is exact=true)
};

// ── Small helpers ────────────────────────────────────────────────────────────

namespace yue2_sheet_detail {

static std::string dirname_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

static std::string join_path(const std::string & dir, const std::string & rel) {
    if (dir.empty() || dir == ".") {
        return rel;
    }
    const char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + rel : dir + "/" + rel;
}

static std::string lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    return s;
}

// Best-first sheetsage2-<quant>.gguf under <models_dir>/yue2 or <models_dir>.
// Same search-dir precedence and quant ladder as yue2_find_variant
// (yue2-model.h), but that helper's prefix is hard-coded "yue2-<stem>-" and
// the converter names this file "sheetsage2-<type>.gguf" with no "yue2-"
// segment (engine/tools/convert-sheetsage2.py:1048) — so it needs its own
// four-line copy rather than a call into that function.
static bool find_model(const std::string & models_dir, std::string * out_path) {
    const std::vector<std::string> dirs = { models_dir + YUE2_SEP "yue2", models_dir };
    const std::string              prefix = "sheetsage2-";
    std::string                    best_path;
    int                             best_rank = 1 << 30;
    for (const auto & dir : dirs) {
        std::vector<std::string> names;
        yue2_list_dir(dir, &names);
        for (const auto & n : names) {
            if (n.size() <= prefix.size() + 5 || n.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            if (n.compare(n.size() - 5, 5, ".gguf") != 0) {
                continue;
            }
            const std::string quant = n.substr(prefix.size(), n.size() - prefix.size() - 5);
            const int         rank  = yue2_quant_rank(quant);
            if (rank < best_rank) {
                best_rank = rank;
                best_path = dir + YUE2_SEP + n;
            }
        }
        if (!best_path.empty()) {
            break;  // this dir supplied a match; mirrors yue2_find_variant's "subdir wins" ordering
        }
    }
    if (best_path.empty()) {
        return false;
    }
    *out_path = best_path;
    return true;
}

struct SourceRow {
    std::string name;         // manifest sources[].name
    std::string path;         // manifest sources[].source — the audio file
    std::string latent_rel;   // manifest sources[].latents, may be empty (the match key when present)
    bool        has_abc       = false;
    bool        has_abc_error = false;
};

// Re-reads the manifest fresh, sets/replaces exactly one source row's
// abc/abc_error/abc_producer/abc_windows/abc_ms, recomputes the root-level
// abc_sources_ok/abc_sources_failed by scanning every source's CURRENT state,
// and writes the whole thing back atomically. Matches the source by
// "latents" when the caller has one (unique cache key, same precedent as
// yue2-align-run.h), falling back to "name" otherwise.
static bool write_source_result(const std::string & manifest_path, const std::string & latent_key,
                                 const std::string & name_key, bool ok, const std::string & abc_text,
                                 const std::string & error_text, const std::string & producer_tag, int windows,
                                 double ms, int64_t * out_ok, int64_t * out_failed) {
    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * doc = yyjson_read_file(manifest_path.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-sheet] cannot re-read %s for the rewrite: %s\n", manifest_path.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return false;
    }
    yyjson_mut_doc * mdoc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * mroot = yyjson_val_mut_copy(mdoc, yyjson_doc_get_root(doc));
    yyjson_doc_free(doc);
    if (!mroot) {
        fprintf(stderr, "[yue2-sheet] cannot copy the manifest for rewriting\n");
        yyjson_mut_doc_free(mdoc);
        return false;
    }
    yyjson_mut_doc_set_root(mdoc, mroot);

    yyjson_mut_val * sarr = yyjson_mut_obj_get(mroot, "sources");
    if (!sarr || !yyjson_mut_is_arr(sarr)) {
        fprintf(stderr, "[yue2-sheet] %s has no sources[] array on rewrite\n", manifest_path.c_str());
        yyjson_mut_doc_free(mdoc);
        return false;
    }

    yyjson_mut_val * hit = nullptr;
    {
        size_t           i = 0, n = 0;
        yyjson_mut_val * it = nullptr;
        yyjson_mut_arr_foreach(sarr, i, n, it) {
            if (!yyjson_mut_is_obj(it)) {
                continue;
            }
            yyjson_mut_val *   lv = yyjson_mut_obj_get(it, "latents");
            yyjson_mut_val *   nv = yyjson_mut_obj_get(it, "name");
            const std::string  lat = (lv && yyjson_mut_is_str(lv)) ? yyjson_mut_get_str(lv) : "";
            const std::string  nm  = (nv && yyjson_mut_is_str(nv)) ? yyjson_mut_get_str(nv) : "";
            const bool         by_latent = !latent_key.empty() && !lat.empty();
            if (by_latent ? (lat == latent_key) : (nm == name_key)) {
                hit = it;
                break;
            }
        }
    }
    if (!hit) {
        fprintf(stderr, "[yue2-sheet] cannot find source \"%s\" in %s on rewrite (was it edited concurrently?)\n",
                name_key.c_str(), manifest_path.c_str());
        yyjson_mut_doc_free(mdoc);
        return false;
    }

    yyjson_mut_obj_remove_str(hit, "abc");
    yyjson_mut_obj_remove_str(hit, "abc_error");
    if (ok) {
        yyjson_mut_obj_add_strcpy(mdoc, hit, "abc", abc_text.c_str());
    } else {
        yyjson_mut_obj_add_strcpy(mdoc, hit, "abc_error", error_text.c_str());
    }
    yyjson_mut_obj_remove_str(hit, "abc_producer");
    yyjson_mut_obj_add_strcpy(mdoc, hit, "abc_producer", producer_tag.c_str());
    yyjson_mut_obj_remove_str(hit, "abc_windows");
    yyjson_mut_obj_add_int(mdoc, hit, "abc_windows", windows);
    yyjson_mut_obj_remove_str(hit, "abc_ms");
    yyjson_mut_obj_add_real(mdoc, hit, "abc_ms", ms);

    // Recompute from the manifest's OWN current state, not from an in-memory
    // running tally: a --force re-run of one --only source, or a crash that
    // left an earlier run partial, must not corrupt these two counts.
    int64_t ok_count = 0, failed_count = 0;
    {
        size_t           i = 0, n = 0;
        yyjson_mut_val * it = nullptr;
        yyjson_mut_arr_foreach(sarr, i, n, it) {
            if (!yyjson_mut_is_obj(it)) {
                continue;
            }
            yyjson_mut_val * av = yyjson_mut_obj_get(it, "abc");
            yyjson_mut_val * ev = yyjson_mut_obj_get(it, "abc_error");
            if (av && yyjson_mut_is_str(av) && yyjson_mut_get_len(av) > 0) {
                ok_count++;
            } else if (ev && yyjson_mut_is_str(ev) && yyjson_mut_get_len(ev) > 0) {
                failed_count++;
            }
        }
    }

    yyjson_mut_obj_remove_str(mroot, "abc_present");
    yyjson_mut_obj_add_bool(mdoc, mroot, "abc_present", true);
    yyjson_mut_obj_remove_str(mroot, "abc_producer");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "abc_producer", (std::string("ace-train yue2-sheet ") + ACE_VERSION).c_str());
    yyjson_mut_obj_remove_str(mroot, "abc_created_at");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "abc_created_at", pm_iso8601_utc_now().c_str());
    yyjson_mut_obj_remove_str(mroot, "abc_sources_ok");
    yyjson_mut_obj_add_int(mdoc, mroot, "abc_sources_ok", ok_count);
    yyjson_mut_obj_remove_str(mroot, "abc_sources_failed");
    yyjson_mut_obj_add_int(mdoc, mroot, "abc_sources_failed", failed_count);

    size_t mlen  = 0;
    char * mjson = yyjson_mut_write(mdoc, YYJSON_WRITE_PRETTY, &mlen);
    yyjson_mut_doc_free(mdoc);
    if (!mjson) {
        fprintf(stderr, "[yue2-sheet] cannot serialize the rewritten manifest\n");
        return false;
    }
    const bool wrote = pm_write_atomic(manifest_path, std::string(mjson, mlen));
    free(mjson);
    if (wrote) {
        if (out_ok) {
            *out_ok = ok_count;
        }
        if (out_failed) {
            *out_failed = failed_count;
        }
    }
    return wrote;
}

}  // namespace yue2_sheet_detail

// ── The run ──────────────────────────────────────────────────────────────────

static int yue2_sheet_run(const Yue2SheetArgs & a) {
    using namespace yue2_sheet_detail;
    const auto t_run_start = std::chrono::steady_clock::now();

    if (a.manifest.empty()) {
        fprintf(stderr, "[yue2-sheet] --manifest <yue2_preprocess.json> is required\n");
        return 2;
    }
    if (!pm_file_exists(a.manifest)) {
        fprintf(stderr, "[yue2-sheet] manifest not found: %s\n", a.manifest.c_str());
        return 1;
    }
    if (a.model_path.empty() && a.models_dir.empty()) {
        fprintf(stderr, "[yue2-sheet] one of --model <sheetsage2-*.gguf> or --models <dir> is required\n");
        return 2;
    }

    // ── read the manifest ──────────────────────────────────────────────────
    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * doc = yyjson_read_file(a.manifest.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-sheet] cannot parse %s: %s\n", a.manifest.c_str(), rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        fprintf(stderr, "[yue2-sheet] %s has no top-level object\n", a.manifest.c_str());
        yyjson_doc_free(doc);
        return 1;
    }

    std::vector<SourceRow> sources;
    {
        yyjson_val * sarr = yyjson_obj_get(root, "sources");
        if (!sarr || !yyjson_is_arr(sarr)) {
            fprintf(stderr, "[yue2-sheet] %s has no sources[] array — a lead sheet is a per-SOURCE artefact "
                            "(the whole song's own audio), never a clip's\n",
                    a.manifest.c_str());
            yyjson_doc_free(doc);
            return 1;
        }
        size_t       i = 0, n = 0;
        yyjson_val * it = nullptr;
        yyjson_arr_foreach(sarr, i, n, it) {
            if (!yyjson_is_obj(it)) {
                continue;
            }
            SourceRow s;
            s.name       = pm_js_str(it, "name");
            s.path       = pm_js_str(it, "source");
            s.latent_rel = pm_js_str(it, "latents");
            const std::string abc_v   = pm_js_str(it, "abc");
            const std::string err_v   = pm_js_str(it, "abc_error");
            s.has_abc       = !abc_v.empty();
            s.has_abc_error = !err_v.empty();
            if (s.name.empty() || s.path.empty()) {
                fprintf(stderr, "[yue2-sheet] sources[%zu] is missing name/source — refusing rather than "
                                "writing a lead sheet nothing can be matched to\n",
                        i);
                yyjson_doc_free(doc);
                return 1;
            }
            sources.push_back(std::move(s));
        }
    }
    yyjson_doc_free(doc);
    doc  = nullptr;
    root = nullptr;

    // ── pick the sources this run touches ───────────────────────────────────
    std::vector<size_t> picked;
    size_t              n_matching_only = 0, n_already_done = 0;
    for (size_t i = 0; i < sources.size(); i++) {
        if (!yp_contains_ci(sources[i].name, a.only)) {
            continue;
        }
        n_matching_only++;
        const bool done = sources[i].has_abc || sources[i].has_abc_error;
        if (done) {
            n_already_done++;
        }
        if (!a.force && done) {
            continue;  // already done — see the "cached" tally below
        }
        picked.push_back(i);
    }
    // Under --force, an already-done source is still selected (it will be
    // re-transcribed), so it is not "skipped" the way it is without --force —
    // report it as "already had abc/abc_error, redoing" instead of "cached".
    const size_t n_cached = a.force ? 0 : n_already_done;
    if (picked.empty()) {
        fprintf(stderr,
                "[yue2-sheet] nothing to do: no source in %s matches --only '%s' and lacks abc/abc_error "
                "(pass --force to re-transcribe)\n",
                a.manifest.c_str(), a.only.c_str());
        return 0;  // not an error: a fully-cached manifest is a finished stage
    }

    // ── resolve + load the model ONCE ───────────────────────────────────────
    std::string model_path = a.model_path;
    if (model_path.empty()) {
        if (!find_model(a.models_dir, &model_path)) {
            fprintf(stderr, "[yue2-sheet] no sheetsage2-*.gguf under %s/yue2 or %s\n", a.models_dir.c_str(),
                    a.models_dir.c_str());
            return 1;
        }
        fprintf(stderr, "[yue2-sheet] --model not given; discovery picked %s\n", model_path.c_str());
    }

    SheetSageModelLoadOptions load_opt;
    load_opt.exact = !a.fast;  // default: the CPU-and-CUDA-alike precision fix (doc 23); --fast opts out

    SheetSageModel m;
    std::string    err;
    if (!sheetsage_model_load(&m, model_path, &err, load_opt)) {
        fprintf(stderr, "[yue2-sheet] SheetSage2 load failed (%s): %s\n", model_path.c_str(), err.c_str());
        return 1;
    }

    const std::string backend_tag = lower(ggml_backend_name(m.backend));
    const std::string producer_tag =
        yue2_basename(model_path) + " " + (load_opt.exact ? "exact" : "fast") + " " + backend_tag;
    const std::string threads_tag =
        a.threads > 0 ? (" | --threads " + std::to_string(a.threads)) : std::string();

    fprintf(stderr,
            "[yue2-sheet] %zu of %zu source(s) match --only '%s' (%zu already have abc/abc_error) | %zu selected "
            "for this run%s\n"
            "[yue2-sheet] model %s | %s%s | backend %s%s\n",
            n_matching_only, sources.size(), a.only.c_str(), n_already_done, picked.size(),
            a.force && n_already_done ? " (--force: redoing already-done sources too)" : "",
            yue2_basename(model_path).c_str(), load_opt.exact ? "exact" : "fast", a.force ? " | --force" : "",
            backend_tag.c_str(), threads_tag.c_str());

    SheetSageTranscribeOptions topt;
    topt.exact   = load_opt.exact;
    topt.threads = a.threads;

    // ── decode plumbing (the same two routes yue2-tokenize-run.h uses) ──────
    const std::string out_dir    = dirname_of(a.manifest);
    const std::string tmp_dir    = join_path(out_dir, ".tmp");
    const std::string tmp_wav    = join_path(tmp_dir, "yue2_sheet_decode.wav");
    bool              tmp_dir_made = false;
    const std::string ffmpeg     = "ffmpeg";
    const int64_t     MERT_SR    = (int64_t) m.mert.cfg.sample_rate;  // 24000

    size_t n_ok = 0, n_soft_failed = 0, n_infra_failed = 0;
    double total_audio_sec = 0.0, total_wall_ms = 0.0;
    int64_t last_ok_count = 0, last_failed_count = 0;

    for (size_t pi = 0; pi < picked.size(); pi++) {
        SourceRow & s  = sources[picked[pi]];
        const auto  t0 = std::chrono::steady_clock::now();

        // ── decode -> 48 kHz planar stereo -> mono -> resample to 24 kHz ──
        const std::string ext        = pm_ext_of(s.path);
        const bool        use_ffmpeg = !(ext == ".wav" || ext == ".mp3");
        int                T48       = 0;
        float *            planar    = nullptr;
        std::string        derr;
        if (use_ffmpeg) {
            if (!tmp_dir_made) {
                pm_mkdir_p(tmp_dir);
                tmp_dir_made = true;
            }
            planar = yp_read_ffmpeg(ffmpeg, s.path, tmp_wav, 48000, &T48, &derr);
        } else {
            planar = yp_read_native_48k(s.path, &T48);
            if (!planar || T48 <= 0) {
                derr = "decode failed";
            }
        }
        if (!planar || T48 <= 0) {
            free(planar);
            fprintf(stderr, "[yue2-sheet] %zu/%zu FAIL %-46s decode: %s\n", pi + 1, picked.size(), s.name.c_str(),
                    derr.c_str());
            n_infra_failed++;
            continue;
        }

        std::vector<float> mono((size_t) T48);
        const float *      L = planar;
        const float *      R = planar + T48;
        for (int i = 0; i < T48; i++) {
            mono[(size_t) i] = 0.5f * (L[i] + R[i]);
        }
        free(planar);

        int     n24 = 0;
        float * r   = audio_resample(mono.data(), T48, 48000, (int) MERT_SR, 1, &n24);
        mono.clear();
        mono.shrink_to_fit();
        if (!r || n24 <= 0) {
            free(r);
            fprintf(stderr, "[yue2-sheet] %zu/%zu FAIL %-46s resample 48000 -> %lld Hz failed\n", pi + 1,
                    picked.size(), s.name.c_str(), (long long) MERT_SR);
            n_infra_failed++;
            continue;
        }
        std::vector<float> pcm24(r, r + n24);
        free(r);
        const double audio_sec = (double) pcm24.size() / (double) MERT_SR;

        // ── transcribe ──
        SheetSageTranscribeResult res;
        const bool                 call_ok = sheetsage_transcribe(m, pcm24.data(), (int64_t) pcm24.size(), topt,
                                                                   &res, &err);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!call_ok) {
            // A setup/precondition failure (doc 19's sheetsage-pipeline.h header:
            // "these are bugs in the CALLER"), not per-source content — every
            // source shares the same model and options, so this cannot be fixed
            // by moving on to the next one.
            fprintf(stderr, "[yue2-sheet] FATAL sheetsage_transcribe setup error on %s: %s\n", s.name.c_str(),
                    err.c_str());
            sheetsage_model_free(&m);
            if (tmp_dir_made) {
                hs_remove(tmp_wav);
            }
            return 1;
        }

        const bool wrote = write_source_result(a.manifest, s.latent_rel, s.name, res.ok, res.abc, res.error,
                                                producer_tag, res.windows, ms, &last_ok_count, &last_failed_count);
        if (!wrote) {
            fprintf(stderr, "[yue2-sheet] FATAL cannot write %s after transcribing %s\n", a.manifest.c_str(),
                    s.name.c_str());
            sheetsage_model_free(&m);
            if (tmp_dir_made) {
                hs_remove(tmp_wav);
            }
            return 1;
        }

        total_audio_sec += audio_sec;
        total_wall_ms += ms;
        if (res.ok) {
            n_ok++;
            fprintf(stderr,
                    "[yue2-sheet] %zu/%zu ok    %-46s %7.1f s audio | windows=%d abc=%zuB | %.1f s (%.2fx rt)\n",
                    pi + 1, picked.size(), s.name.c_str(), audio_sec, res.windows, res.abc.size(), ms / 1000.0,
                    audio_sec / (ms / 1000.0));
        } else {
            n_soft_failed++;
            fprintf(stderr,
                    "[yue2-sheet] %zu/%zu error %-46s %7.1f s audio | windows=%d abc_error=\"%s\" | %.1f s\n",
                    pi + 1, picked.size(), s.name.c_str(), audio_sec, res.windows, res.error.c_str(), ms / 1000.0);
        }
    }

    if (tmp_dir_made) {
        hs_remove(tmp_wav);
    }
    sheetsage_model_free(&m);

    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run_start).count();
    fprintf(stderr,
            "\n[yue2-sheet] done: %zu ok, %zu soft-failed (abc_error), %zu infra-failed (decode), of %zu "
            "selected (%zu already cached); manifest now shows abc_sources_ok=%lld abc_sources_failed=%lld\n"
            "[yue2-sheet] %.2f min of audio in %.1f s wall\n",
            n_ok, n_soft_failed, n_infra_failed, picked.size(), n_cached, (long long) last_ok_count,
            (long long) last_failed_count, total_audio_sec / 60.0, wall_s);
    if (n_infra_failed) {
        fprintf(stderr,
                "[yue2-sheet] NOTE: %zu source(s) could not be decoded at all and were left untouched in the "
                "manifest (not written as abc_error, which is reserved for the reference's own typed failures) — "
                "re-run to retry them.\n",
                n_infra_failed);
    }
    // Per doc 19/23: soft failure (abc_error) is an expected, non-error
    // outcome. Only a load/manifest-level problem returns non-zero, and every
    // such path above already returned by the time we get here.
    return 0;
}
