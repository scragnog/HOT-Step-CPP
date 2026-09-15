#pragma once
// train/yue2-align-run.h — `ace-train yue2-align`: fill the `cursor_words` slot
// the AR trainer's lyric-cursor loss reads, natively.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). It replaces
// engine/tools/yue2-cursor-bridge.py — and, behind it, upstream's
// cursor_prep.py plus a torchaudio install — with the C++ MMS_FA port. The
// CONSUMER (`yue2_at_read_words5` / `yue2_at_cursor_build`,
// train/yue2-ar-train-run.h) is not touched: this file writes exactly the bytes
// the bridge wrote, into exactly the same field, so a manifest cannot tell
// which producer filled it.
//
// Structural precedent: train/yue2-tokenize-run.h. Same shape — walk the
// manifest's sources[], produce one artefact per source, rewrite the manifest
// in place with a per-source relative path. Read that file's header first if
// this one's is not enough.
//
// ── What it does ───────────────────────────────────────────────────────────
//
//   read <cache>/yue2_preprocess.json
//   for every SOURCE it lists:
//       find <stems>/<source stem>/vocals.wav     (the Demucs htdemucs layout)
//       decode, mean the two channels -> mono, resample to 16 kHz
//       yue2_mmsfa_emissions over the WHOLE track   -> [T, 28] log-probs
//       yue2_cursor_align(..., source.lyrics)       -> (start, end, score,
//                                                       char0, char1) per word
//       write <cache>/cursor/<source stem>.f32
//   rewrite the manifest (a .bak beside it): sources[].cursor_words = that path
//
// ── THE WHOLE TRACK, IN ONE FORWARD. NOT NEGOTIABLE. ───────────────────────
//
// docs/plans/yue2/17-mms-fa-port.md §0.1 item 1: MMS_FA layer-normalises the
// waveform per utterance, one mean and one variance over the entire input. A
// track aligned in chunks is a different model input in every chunk, and the
// emissions would not be the ones the reference produced. yue2_mmsfa_emissions
// does that normalisation itself over exactly the samples it is handed, so the
// rule here is simply: hand it the whole song. It costs ~5 GB of compute buffer
// and ~100 s per 3.7-minute track on 16 CPU threads, which is the price.
//
// The CTC Viterbi behind yue2_cursor_align is the other big allocation:
// T x (2N+1) doubles plus a byte backpointer, so a 4-minute track with a
// 1300-character sheet is about 11148 x 2600 x 9 = 260 MB. It runs after the
// graph is freed of nothing in particular — both fit comfortably, but that is
// why this tool is one source at a time and never a batch.
//
// ── THE CHAR OFFSETS ARE THE FRAGILE PART ──────────────────────────────────
//
// Columns 4 and 5 of every row are CODEPOINT offsets into the manifest's own
// `lyrics` string, and the cursor loss indexes the tokenised prefix with them.
// The bridge had to compare the aligner's lyrics file to the manifest field
// byte for byte, because the aligner was a separate process reading a separate
// file and a single collapsed space would silently misplace every word after
// it. This tool has no such seam: it aligns the manifest's `lyrics` field
// itself, which is the string the trainer will measure the offsets against
// (`yue2_at_utf8_len(s.lyrics)`). That entire class of failure is gone, so
// there is nothing here that corresponds to the bridge's REFUSED-on-mismatch
// path — and, for the same reason, a source with an empty `lyrics` is skipped
// by name rather than aligned against something else on disk. Do NOT add a
// .lyrics.txt fallback: that would put the seam back.
//
// ── No cache-skip ──────────────────────────────────────────────────────────
//
// Unlike yue2-tokenize, a source whose cursor file already exists is realigned
// anyway. The file carries no shape a cache check could verify (its length
// depends on the lyrics, not on the audio), and a stale one — written before a
// lyric edit — is exactly the input that trains a plausible, wrong cursor.
// `--only` and `--limit` are how you do less work.

#include "audio-io.h"          // audio_read_buf — WAV/MP3 from a buffer, planar stereo
#include "audio-resample.h"    // audio_resample — the engine's Kaiser polyphase
#include "hot-step-fsutf8.h"   // hs_fopen — UTF-8 paths
#include "version.h"           // ACE_VERSION, stamped into the rewritten manifest
#include "yyjson.h"

#include "train/preprocess-io.h"    // pm_* path/string/atomic-write helpers
#include "yue2/yue2-ctc-align.h"    // yue2_cursor_align — the CPU half, no model
#include "yue2/yue2-mmsfa.h"        // the acoustic model

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Arguments ──────────────────────────────────────────────────────────────

struct Yue2AlignArgs {
    std::string manifest;  // --manifest <yue2_preprocess.json>   (required)
    std::string mmsfa;     // --mmsfa <mms-fa-f32.gguf>           (required)
    std::string stems;     // --stems <dir>  holding <name>/vocals.wav
    std::string only;      // --only <substr>, case-insensitive, on the source name
    int         limit = 0; // --limit N, debug: first N matching sources
    bool        cpu   = false;  // --cpu, recorded for the log; the env var is set by the caller
};

// ── Small helpers ──────────────────────────────────────────────────────────

namespace yue2_align_detail {

// Local copies of yue2-tokenize-run.h's two path helpers. Eight lines, kept
// local so this header does not have to include the whole tokenizer (MERT
// blocks, the 8-layer head) to join two strings.
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

// "01-green_day-warning.flac" -> "01-green_day-warning". The manifest's
// sources[].name carries the extension; the stem directory, the cursor file
// and the bridge's own naming all drop it (yue2-cursor-bridge.py:60).
static std::string source_stem(const std::string & name) {
    std::string s     = name;
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }
    const size_t dot = s.find_last_of('.');
    if (dot != std::string::npos && dot > 0) {
        s.erase(dot);
    }
    return s;
}

static bool contains_ci(const std::string & hay, const std::string & needle) {
    if (needle.empty()) {
        return true;
    }
    auto lower = [](std::string s) {
        for (char & c : s) {
            if (c >= 'A' && c <= 'Z') {
                c = (char) (c - 'A' + 'a');
            }
        }
        return s;
    };
    return lower(hay).find(lower(needle)) != std::string::npos;
}

static bool read_file_bytes(const std::string & path, std::vector<uint8_t> * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return false;
    }
    out->resize((size_t) n);
    const bool ok = fread(out->data(), 1, (size_t) n, f) == (size_t) n;
    fclose(f);
    return ok;
}

// One vocal stem -> mono 16 kHz, the model's input rate.
//
// audio_read_buf gives PLANAR stereo at the file's own rate (a mono file is
// duplicated into both channels by wav.h, so the mean below is still that
// file's samples). Demucs htdemucs writes 44.1 kHz stereo PCM16, so the
// resample is the normal path, not the exception.
//
// The resampler is the engine's own Kaiser polyphase, not torchaudio's — the
// same substitution yue2-tokenize already makes and books into its tolerance.
// Its output length is floor(n * ratio), one sample shorter than torchaudio's
// round() on this exact file (3567640 vs 3567641), which is below the
// resolution of a 320-sample hop and changes T by nothing.
static bool read_stem_16k(const std::string & path, int target_sr, std::vector<float> * pcm, int * sr_in,
                          std::string * err) {
    std::vector<uint8_t> bytes;
    if (!read_file_bytes(path, &bytes)) {
        *err = "cannot read " + path;
        return false;
    }
    int     T = 0, sr = 0;
    float * planar = audio_read_buf(bytes.data(), bytes.size(), &T, &sr);
    if (!planar || T <= 0 || sr <= 0) {
        free(planar);
        *err = "cannot decode " + path + " (WAV and MP3 are the formats this reader knows)";
        return false;
    }
    *sr_in = sr;
    std::vector<float> mono((size_t) T);
    const float *      L = planar;
    const float *      R = planar + T;
    for (int i = 0; i < T; i++) {
        mono[(size_t) i] = 0.5f * (L[i] + R[i]);
    }
    free(planar);

    if (sr == target_sr) {
        *pcm = std::move(mono);
        return true;
    }
    int     n16 = 0;
    float * r   = audio_resample(mono.data(), T, sr, target_sr, 1, &n16);
    if (!r || n16 <= 0) {
        free(r);
        *err = "resample " + std::to_string(sr) + " -> " + std::to_string(target_sr) + " Hz failed for " + path;
        return false;
    }
    pcm->assign(r, r + n16);
    free(r);
    return true;
}

// Codepoints in a UTF-8 string — the unit the cursor spans' char offsets are
// in, and what the trainer measures them against (yue2_at_utf8_len).
static int64_t utf8_len(const std::string & s) {
    int64_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            n++;
        }
    }
    return n;
}

// The same acceptance the trainer's reader applies (yue2_at_read_words5) and
// the bridge applied before writing. Checked HERE so a bad alignment never
// reaches the manifest, rather than 137 steps into a training run.
static bool words5_ok(const std::vector<float> & rows5, int64_t n_chars, std::string * why) {
    if (rows5.empty() || rows5.size() % 5 != 0) {
        *why = "not a whole number of 5-column rows";
        return false;
    }
    const size_t nW = rows5.size() / 5;
    float        prev = -1.0f;
    for (size_t w = 0; w < nW; w++) {
        const float * r = rows5.data() + w * 5;
        if (!(r[0] >= 0.0f) || !(r[1] >= r[0]) || r[0] < prev) {
            *why = "word " + std::to_string(w) + ": spans are not monotonic";
            return false;
        }
        if (!(r[3] >= 0.0f) || !(r[4] >= r[3]) || r[4] > (float) n_chars) {
            *why = "word " + std::to_string(w) + ": char span runs past the lyrics";
            return false;
        }
        prev = r[0];
    }
    return true;
}

struct SourceRow {
    std::string name;        // manifest sources[].name, with extension
    std::string stem;        // source_stem(name) — the stems dir and cursor file key
    std::string latent_rel;  // manifest sources[].latents — the unique match key
    std::string lyrics;
    /** manifest sources[].instrumental. A track with no singing is EXCLUDED
     *  from this stage rather than skipped by it: there is nothing to align and
     *  nothing wrong, so it must not read as a fault and must not be what makes
     *  the stage fail. Absent in a cache written before the field existed,
     *  which reads as false and behaves exactly as it used to. */
    bool        instrumental = false;

    // Filled by the run.
    bool        done = false;
    std::string cursor_rel;
    int64_t     words   = 0;
    int64_t     frames  = 0;
    double      first_s = 0.0, last_s = 0.0;
    double      audio_s = 0.0;
    double      model_ms = 0.0, align_ms = 0.0, wall_ms = 0.0;
};

}  // namespace yue2_align_detail

// ── The run ────────────────────────────────────────────────────────────────

static int yue2_align_run(const Yue2AlignArgs & a) {
    using namespace yue2_align_detail;
    const auto t_run_start = std::chrono::steady_clock::now();

    if (a.manifest.empty()) {
        fprintf(stderr, "[yue2-align] --manifest <yue2_preprocess.json> is required\n");
        return 2;
    }
    if (a.mmsfa.empty()) {
        fprintf(stderr, "[yue2-align] --mmsfa <mms-fa-f32.gguf> is required\n");
        return 2;
    }
    if (a.stems.empty()) {
        fprintf(stderr, "[yue2-align] --stems <dir> is required: this tool needs a VOCAL stem per source "
                        "(<dir>/<name>/vocals.wav), not the mixed song\n");
        return 2;
    }
    if (!pm_file_exists(a.manifest)) {
        fprintf(stderr, "[yue2-align] manifest not found: %s\n", a.manifest.c_str());
        return 1;
    }

    const std::string cache_dir  = dirname_of(a.manifest);
    const std::string cursor_dir = join_path(cache_dir, "cursor");

    // ── read the manifest ──────────────────────────────────────────────
    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * doc = yyjson_read_file(a.manifest.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-align] cannot parse %s: %s\n", a.manifest.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        fprintf(stderr, "[yue2-align] %s has no top-level object\n", a.manifest.c_str());
        yyjson_doc_free(doc);
        return 1;
    }

    std::vector<SourceRow> sources;
    {
        yyjson_val * sarr = yyjson_obj_get(root, "sources");
        if (!sarr || !yyjson_is_arr(sarr)) {
            fprintf(stderr, "[yue2-align] %s has no sources[] array — cursor spans are a per-SOURCE artefact "
                            "(a clip inherits them through its source, see yue2-ar-train-run.h)\n",
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
            s.latent_rel = pm_js_str(it, "latents");
            s.lyrics     = pm_js_str(it, "lyrics");
            {
                yyjson_val * iv = yyjson_obj_get(it, "instrumental");
                s.instrumental  = iv && yyjson_is_bool(iv) && yyjson_get_bool(iv);
            }
            s.stem       = source_stem(s.name);
            if (s.name.empty() || s.stem.empty()) {
                fprintf(stderr, "[yue2-align] sources[%zu] has no name — refusing rather than writing a cursor "
                                "file nothing can be matched to\n",
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

    std::vector<size_t> picked;
    for (size_t i = 0; i < sources.size(); i++) {
        if (!contains_ci(sources[i].name, a.only)) {
            continue;
        }
        picked.push_back(i);
        if (a.limit > 0 && (int) picked.size() >= a.limit) {
            break;
        }
    }
    if (picked.empty()) {
        fprintf(stderr, "[yue2-align] no source in %s matches --only '%s'\n", a.manifest.c_str(), a.only.c_str());
        return 1;
    }

    // ── load MMS_FA ────────────────────────────────────────────────────
    static Yue2MmsfaModel m;  // static: keeps backend handles off the stack, as the other one-shot commands do
    std::string           err;
    if (!yue2_mmsfa_load(&m, a.mmsfa, &err)) {
        fprintf(stderr, "[yue2-align] MMS_FA load failed: %s\n", err.c_str());
        return 1;
    }
    const int64_t SR = (int64_t) m.cfg.sample_rate;  // 16000
    const int     C  = (int) m.cfg.n_labels;         // 28
    if (SR <= 0 || C <= 1 || m.cfg.labels.size() != (size_t) C) {
        fprintf(stderr, "[yue2-align] the GGUF states sample_rate=%lld, %d labels (%zu listed)\n", (long long) SR, C,
                m.cfg.labels.size());
        yue2_mmsfa_free(&m);
        return 1;
    }

    if (!pm_mkdir_p(cursor_dir)) {
        fprintf(stderr, "[yue2-align] cannot create %s\n", cursor_dir.c_str());
        yue2_mmsfa_free(&m);
        return 1;
    }

    fprintf(stderr,
            "[yue2-align] %zu of %zu source(s) from %s\n"
            "[yue2-align] model %s on %s | %lld Hz in, %d CTC labels, blank '%s' | stems %s/<name>/vocals.wav\n",
            picked.size(), sources.size(), a.manifest.c_str(), yue2_basename(a.mmsfa).c_str(),
            ggml_backend_name(m.backend), (long long) SR, C, m.cfg.labels.empty() ? "?" : m.cfg.labels[0].c_str(),
            a.stems.c_str());
    if (a.cpu) {
        fprintf(stderr, "[yue2-align] --cpu: the backend was pinned to CPU before any context was created\n");
    }

    // ── per source ─────────────────────────────────────────────────────
    Yue2MmsfaGraph g;
    size_t         n_done = 0, n_skipped = 0, n_failed = 0, n_instrumental = 0;
    double         total_audio_s = 0.0, total_model_ms = 0.0, total_align_ms = 0.0;

    for (size_t pi = 0; pi < picked.size(); pi++) {
        SourceRow & s  = sources[picked[pi]];
        const auto  t0 = std::chrono::steady_clock::now();

        // Both skips are loud and by name: a silently unaligned source trains
        // with no cursor loss at all, and nothing downstream says so.
        // An instrumental is not a problem to report. It is counted apart from
        // the skips so it cannot make the run look like it went wrong, and so
        // an all-instrumental corpus does not fail the stage.
        if (s.instrumental) {
            fprintf(stderr, "[yue2-align] %zu/%zu INSTRUMENTAL %-38s excluded, nothing sung to align\n",
                    pi + 1, picked.size(), s.name.c_str());
            n_instrumental++;
            continue;
        }
        if (pm_trim(s.lyrics).empty()) {
            fprintf(stderr, "[yue2-align] %zu/%zu SKIP %-46s no lyrics in the manifest\n", pi + 1, picked.size(),
                    s.name.c_str());
            n_skipped++;
            continue;
        }
        const std::string stem_path = join_path(join_path(a.stems, s.stem), "vocals.wav");
        if (!pm_file_exists(stem_path)) {
            fprintf(stderr, "[yue2-align] %zu/%zu SKIP %-46s no vocal stem at %s\n", pi + 1, picked.size(),
                    s.name.c_str(), stem_path.c_str());
            n_skipped++;
            continue;
        }

        std::vector<float> pcm;
        int                sr_in = 0;
        if (!read_stem_16k(stem_path, (int) SR, &pcm, &sr_in, &err)) {
            fprintf(stderr, "[yue2-align] %zu/%zu FAIL %-46s %s\n", pi + 1, picked.size(), s.name.c_str(),
                    err.c_str());
            n_failed++;
            continue;
        }
        s.audio_s = (double) pcm.size() / (double) SR;

        // The WHOLE track, one forward — see the header.
        const auto         t1 = std::chrono::steady_clock::now();
        std::vector<float> em;
        if (!yue2_mmsfa_emissions(m, &g, pcm.data(), (int64_t) pcm.size(), &em, Yue2MmsfaOptions{}, nullptr, &err)) {
            fprintf(stderr, "[yue2-align] %zu/%zu FAIL %-46s emissions: %s\n", pi + 1, picked.size(), s.name.c_str(),
                    err.c_str());
            n_failed++;
            continue;
        }
        const auto t2 = std::chrono::steady_clock::now();
        s.model_ms    = std::chrono::duration<double, std::milli>(t2 - t1).count();
        s.frames      = (int64_t) em.size() / C;

        std::vector<float> rows5;
        // audio_seconds is the STEM's duration, not the manifest's: cursor_prep
        // derives seconds-per-frame as audio_seconds / T, and T came from these
        // very samples. Using the latent cache's duration instead would shift
        // every span by the difference between the two decodes.
        if (!yue2_cursor_align(em.data(), s.frames, C, s.audio_s, m.cfg.labels, s.lyrics, &rows5, &err)) {
            fprintf(stderr, "[yue2-align] %zu/%zu FAIL %-46s align: %s\n", pi + 1, picked.size(), s.name.c_str(),
                    err.c_str());
            n_failed++;
            continue;
        }
        s.align_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();

        const int64_t n_chars = utf8_len(s.lyrics);
        std::string   why;
        if (!words5_ok(rows5, n_chars, &why)) {
            fprintf(stderr, "[yue2-align] %zu/%zu FAIL %-46s %s\n", pi + 1, picked.size(), s.name.c_str(),
                    why.c_str());
            n_failed++;
            continue;
        }

        s.cursor_rel            = "cursor/" + s.stem + ".f32";
        const std::string cpath = join_path(cache_dir, s.cursor_rel);
        if (!pm_write_atomic(cpath, std::string((const char *) rows5.data(), rows5.size() * sizeof(float)))) {
            fprintf(stderr, "[yue2-align] %zu/%zu FAIL %-46s cannot write %s\n", pi + 1, picked.size(),
                    s.name.c_str(), cpath.c_str());
            n_failed++;
            continue;
        }

        s.words   = (int64_t) (rows5.size() / 5);
        s.first_s = rows5[0];
        s.last_s  = rows5[rows5.size() - 4];
        s.done    = true;
        n_done++;
        total_audio_s += s.audio_s;
        total_model_ms += s.model_ms;
        total_align_ms += s.align_ms;
        s.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr,
                "[yue2-align] %zu/%zu ok   %-46s %6.1f s @%d Hz -> %lld frames | %lld words, %.1f..%.1f s, "
                "%lld lyric chars | model %.1f s, ctc %.1f s (%.2fx rt)\n",
                pi + 1, picked.size(), s.name.c_str(), s.audio_s, sr_in, (long long) s.frames, (long long) s.words,
                s.first_s, s.last_s, (long long) n_chars, s.model_ms / 1000.0, s.align_ms / 1000.0,
                s.audio_s / (s.wall_ms / 1000.0));
    }

    yue2_mmsfa_graph_free(&g);
    yue2_mmsfa_free(&m);

    if (n_done == 0 && n_instrumental > 0 && n_skipped == 0 && n_failed == 0) {
        // Every source is an instrumental. There was never anything to align,
        // so this is a finished stage and not a failed one; failing here would
        // stop a pipeline over a correctly labelled corpus.
        fprintf(stderr, "[yue2-align] every source is an instrumental (%zu) - nothing to align, and nothing "
                        "wrong. The manifest is unchanged.\n", n_instrumental);
        return 0;
    }
    if (n_done == 0) {
        fprintf(stderr, "[yue2-align] nothing aligned (%zu skipped, %zu failed, %zu instrumental). The manifest "
                        "is unchanged.\n",
                n_skipped, n_failed, n_instrumental);
        return 1;
    }

    // ── rewrite the manifest ───────────────────────────────────────────
    //
    // Re-read and mutable-copy rather than re-serialising from the structs
    // above: this tool parses four of the manifest's forty-odd fields and has
    // no business rewriting the rest (yue2-tokenize-run.h makes the same call).
    // A .bak is left beside it, as yue2-cursor-bridge.py did — and, as there,
    // only when one does not already exist, so a second run cannot overwrite
    // the pre-alignment copy with a post-alignment one.
    {
        std::vector<uint8_t> orig;
        const std::string    bak = a.manifest + ".bak";
        if (!pm_file_exists(bak) && read_file_bytes(a.manifest, &orig)) {
            pm_write_atomic(bak, std::string((const char *) orig.data(), orig.size()));
        }
    }
    memset(&rerr, 0, sizeof(rerr));
    doc = yyjson_read_file(a.manifest.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-align] cannot re-read %s for the rewrite: %s\n", a.manifest.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_mut_doc * mdoc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * mroot = yyjson_val_mut_copy(mdoc, yyjson_doc_get_root(doc));
    yyjson_doc_free(doc);
    doc = nullptr;
    if (!mroot) {
        fprintf(stderr, "[yue2-align] cannot copy the manifest for rewriting\n");
        yyjson_mut_doc_free(mdoc);
        return 1;
    }
    yyjson_mut_doc_set_root(mdoc, mroot);

    yyjson_mut_obj_remove_str(mroot, "cursor_words_note");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "cursor_words_note",
                              "Per-source \"cursor_words\" is a manifest-relative path to little-endian f32 rows of "
                              "(start_s, end_s, score, char0, char1), one per aligned word, in time order. char0/"
                              // Deliberately ASCII, unlike this file's comments: it is DATA, written from a C++
                              // literal into a file other tools parse, and every other note in this manifest is
                              // ASCII too. A build without /utf-8 would re-encode a fancy dash silently.
                              "char1 are CODEPOINT offsets into THIS source's \"lyrics\" string -- not bytes, and "
                              "not into any other copy of the lyrics. Produced by CTC forced alignment of that "
                              "lyric sheet against the source's vocal stem.");
    yyjson_mut_obj_remove_str(mroot, "cursor_words_producer");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "cursor_words_producer",
                              (std::string("ace-train yue2-align ") + ACE_VERSION).c_str());
    yyjson_mut_obj_remove_str(mroot, "cursor_words_model");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "cursor_words_model", yue2_basename(a.mmsfa).c_str());
    yyjson_mut_obj_remove_str(mroot, "cursor_words_created_at");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "cursor_words_created_at", pm_iso8601_utc_now().c_str());

    // Matched on the source's own identifying field, never on array position:
    // the parse pass skipped non-object entries, so an index would put one
    // source's spans on another's row if a manifest ever carried a stray null.
    // `latents` is unique by construction (its name is a content hash); `name`
    // is the fallback for a manifest that predates it.
    int64_t written = 0;
    {
        yyjson_mut_val * sarr = yyjson_mut_obj_get(mroot, "sources");
        size_t           i = 0, n = 0;
        yyjson_mut_val * it = nullptr;
        if (sarr && yyjson_mut_is_arr(sarr)) {
            yyjson_mut_arr_foreach(sarr, i, n, it) {
                if (!yyjson_mut_is_obj(it)) {
                    continue;
                }
                yyjson_mut_val * lv = yyjson_mut_obj_get(it, "latents");
                yyjson_mut_val * nv = yyjson_mut_obj_get(it, "name");
                const std::string lat = (lv && yyjson_mut_is_str(lv)) ? yyjson_mut_get_str(lv) : "";
                const std::string nm  = (nv && yyjson_mut_is_str(nv)) ? yyjson_mut_get_str(nv) : "";
                const SourceRow * hit = nullptr;
                for (const SourceRow & s : sources) {
                    if (!s.done) {
                        continue;
                    }
                    const bool by_latent = !lat.empty() && !s.latent_rel.empty();
                    if (by_latent ? (s.latent_rel == lat) : (s.name == nm)) {
                        hit = &s;
                        break;
                    }
                }
                if (!hit) {
                    continue;
                }
                yyjson_mut_obj_remove_str(it, "cursor_words");
                yyjson_mut_obj_add_strcpy(mdoc, it, "cursor_words", hit->cursor_rel.c_str());
                written++;
            }
        }
    }

    size_t mlen  = 0;
    char * mjson = yyjson_mut_write(mdoc, YYJSON_WRITE_PRETTY, &mlen);
    yyjson_mut_doc_free(mdoc);
    if (!mjson) {
        fprintf(stderr, "[yue2-align] cannot serialize the rewritten manifest\n");
        return 1;
    }
    const bool wrote = pm_write_atomic(a.manifest, std::string(mjson, mlen));
    free(mjson);
    if (!wrote) {
        fprintf(stderr, "[yue2-align] cannot write %s\n", a.manifest.c_str());
        return 1;
    }

    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run_start).count();
    fprintf(stderr,
            "\n[yue2-align] done: %zu aligned, %zu skipped, %zu failed, of %zu source(s); %lld now carry "
            "cursor_words\n"
            "[yue2-align] %.2f min of vocal stem in %.1f s wall (model %.1f s, CTC %.1f s) = %.2fx realtime\n",
            n_done, n_skipped, n_failed, sources.size(), (long long) written, total_audio_s / 60.0, wall_s,
            total_model_ms / 1000.0, total_align_ms / 1000.0,
            total_audio_s / (wall_s > 0.0 ? wall_s : 1.0));
    if (n_skipped || n_failed) {
        fprintf(stderr, "[yue2-align] NOTE: a source without cursor_words trains with the cursor loss OFF. That is "
                        "the AR trainer's documented behaviour for a mixed manifest, not an error — but check the "
                        "SKIP lines above before assuming it was meant.\n");
    }
    return n_failed ? 1 : 0;
}
