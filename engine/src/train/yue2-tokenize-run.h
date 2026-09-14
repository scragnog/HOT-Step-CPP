#pragma once
// train/yue2-tokenize-run.h — `ace-train yue2-tokenize`: fill the `codec_ids`
// slot that `yue2-preprocess` reserved and left empty.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). It is the producer half of
// the seam documented in train/yue2-preprocess-run.h around line 124 and in
// docs/plans/yue2/08-nar-lora-trainer.md §5; the consumer half
// (`yue2_nt_load_manifest` / `yue2_nar_train_cond_ids`) already shipped and is
// NOT touched by this file. Where the two could drift, this one moves.
//
// ── What it does ───────────────────────────────────────────────────────────
//
//   read <out>/yue2_preprocess.json
//   for every SOURCE file it lists:
//       decode                (the same two routes yue2-preprocess uses)
//       mean the two channels -> mono                     (tok_oracle.py:143)
//       resample 48 kHz -> the MERT sample rate (24 kHz)  (tok_oracle.py:145)
//       yue2_mert_encode      -> [T25, 1024] instance-normed features
//       yue2_tok_head_predict -> [T25] RAW codes in [0, 32768)
//       write <out>/codes/<latent stem>.i32          (the SOURCE-level array)
//       write <out>/codes/<clip id>.i32   per clip   (sliced by offset_frames)
//   rewrite the manifest: codec_ids_present = true, and every clip that got a
//   slice gains "codec_ids": "codes/<clip id>.i32".
//
// ── Why per-CLIP files and not one per source ──────────────────────────────
//
// Because that is what the reader already does, and the reader is the
// contract. `yue2_nt_load_manifest` resolves a clip's `codec_ids` string
// against the MANIFEST'S directory and reads THE WHOLE FILE into
// `Yue2TrainClip::codec_ids` (yue2-nar-train-run.h:1363-1371). There is no
// offset and no length field on that path, so a clip pointing at the
// source-level array would condition on the whole song. The source-level file
// is still written — it is the cache that makes a re-run cheap and the thing
// you diff when a slice looks wrong — but it is never what a clip names.
//
// ── Alignment: the whole point of the exercise ─────────────────────────────
//
// A clip's codes only mean anything if code[i] describes the same 40 ms as
// latent row i. Two independent chains produce those two streams:
//
//   latents: 48 kHz stereo -> YuE2 VAE, downsampling_ratio 1920 -> 25 fps
//   codes:   24 kHz mono   -> MERT (hop 240, /4 subsample) -> 25 fps,
//            then interpolated to T25 = round(n24 / 24000 * 25)
//
// Both land on 25 fps from the same decoded samples, so they agree to within a
// frame — but "within a frame" is not "equal", and the two formulas genuinely
// disagree by one on some lengths (the VAE's tiling floor vs T25's round). A
// one-frame shift would be invisible: training would simply learn a slightly
// lagged conditioning and nobody would ever see an error.
//
// So this tool does NOT trust the agreement. It computes the source's frame
// count from the manifest (which the trainer's own bounds check derives from
// the latent file's size, yue2-nar-train-run.h's "every cache file, once"
// pass), and RESIZES the code array to exactly that:
//
//   T25 > frames : truncate the tail
//   T25 < frames : repeat the last code to fill
//
// Both directions are reported per source, with the exact delta, and the
// per-clip slice is then a plain contiguous span — `codes[offset ..
// offset+frames)` — for the same reason a clip's latents are (the frame-major
// cache note in yue2-preprocess-run.h). A delta bigger than one frame is
// treated as a failure, not a resize: that means the audio the manifest names
// is no longer the audio the latents came from.
//
// ── RAW codes, and where the +151853 lives ─────────────────────────────────
//
// The files this writes hold RAW codes in [0, 32768). NOT token ids. The
// `+ YUE2_CODEC_OFFSET` happens in `yue2_nar_train_cond_ids`
// (yue2-nar-train-graph.h:204-212) and nowhere else, exactly as the inference
// pipeline does it. The manifest note this tool writes says so in as many
// words, so the two sides cannot drift into double-offsetting — which would
// be finite, in-vocabulary, and completely wrong.
//
// ── TF32 must be off before the first CUDA context ─────────────────────────
//
// Same rule, same reason as yue2-preprocess: the env var is read when the CUDA
// context is CREATED. cmd_yue2_tokenize sets it as its first statement and
// this function REFUSES to run otherwise. The tokenizer's own parity run
// (yue2-probe --mert-block-parity) was gated with TF32 off; a TF32 cuBLAS
// handle puts the subsampler ~270x outside the port's true error and every
// code written here would carry it invisibly.
//
// ── Precision ──────────────────────────────────────────────────────────────
//
// `fp16_feature_store` is left OFF. The reference's production path
// (prep_real.py's `.half()`) rounds the interpolated features to fp16 before
// the instance norm, and the stages-30s fixtures keep it; the FP32 oracle
// drops it. Measured cost of the choice is 0.03-0.13% of frames changing
// argmax (docs/plans/yue2/13-tokenizer-fp32-fixtures.md), which is well inside
// the 2.34% that fp32-vs-bf16 costs on its own. Off is the more accurate half
// of a difference too small to matter, and it is one fewer knob.
//
// The resampler is the engine's own Kaiser polyphase, not scipy's
// `resample_poly`. Pin §6 item 4 already books that difference into the same
// tolerance bucket as bf16. It is named here so nobody hunts it later.

#include "audio-resample.h"       // audio_resample (48 kHz -> 24 kHz)
#include "hot-step-fsutf8.h"      // hs_fopen — UTF-8 paths, not ANSI
#include "version.h"              // ACE_VERSION, stamped into the rewritten manifest
#include "yyjson.h"

#include "train/preprocess-io.h"      // pm_* path/string/atomic-write helpers
#include "train/yue2-preprocess-run.h"  // yp_read_native_48k / yp_read_ffmpeg / yp_* helpers
#include "yue2/yue2-mert.h"          // the 21 Conformer blocks + mel frontend
#include "yue2/yue2-tok-head.h"      // the 8-layer head + yue2_tok_disable_tf32

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// ── Arguments ──────────────────────────────────────────────────────────────

struct Yue2TokenizeArgs {
    std::string manifest;    // --manifest <yue2_preprocess.json>   (required)
    std::string tok_path;    // --tok <yue2-tok-*.gguf>             (or --models)
    std::string models_dir;  // --models <dir>, searched for yue2-tok-*.gguf
    std::string only;        // --only <substr>, case-insensitive, on the source name
    std::string ffmpeg = "ffmpeg";  // --ffmpeg, "" disables the non-WAV/MP3 route
    std::string decode = "auto";    // auto | ffmpeg
    int         limit  = 0;         // --limit N, debug: first N matching sources
    bool        force  = false;     // --force, re-encode sources that already have codes
};

// ── Small helpers ──────────────────────────────────────────────────────────

namespace yue2_tokenize_detail {

// dirname, with both separators, no trailing slash. "a/b.json" -> "a";
// a bare filename -> ".".
static std::string dirname_of(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return ".";
    }
    return path.substr(0, slash);
}

static std::string join_path(const std::string & dir, const std::string & rel) {
    if (dir.empty() || dir == ".") {
        return rel;
    }
    const char last = dir[dir.size() - 1];
    return (last == '/' || last == '\\') ? dir + rel : dir + "/" + rel;
}

// "latents/foo_deadbeef.f32" -> "foo_deadbeef". Falls back to the whole string
// when it does not look like that, which only costs an odd filename.
static std::string latent_stem(const std::string & rel) {
    std::string s     = rel;
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
        s = s.substr(slash + 1);
    }
    if (s.size() > 4 && s.compare(s.size() - 4, 4, ".f32") == 0) {
        s.erase(s.size() - 4);
    }
    return s;
}

// A raw little-endian i32 file, the format yue2_nt_read_i32_file expects.
// Written through the crash-safe write-then-rename: at 4 bytes per frame a
// 10-minute track is 60 kB, so there is nothing to stream.
static bool write_i32(const std::string & path, const int32_t * data, size_t n) {
    const std::string bytes((const char *) data, n * sizeof(int32_t));
    return pm_write_atomic(path, bytes);
}

// True when `path` already holds exactly `frames` i32 values. The cache check:
// a source whose codes are already the right length is skipped without --force.
static bool i32_file_has(const std::string & path, int64_t frames) {
    long long bytes = 0;
    if (!pm_stat_file(path, &bytes, nullptr)) {
        return false;
    }
    return bytes == (long long) frames * 4;
}

static bool read_i32(const std::string & path, std::vector<int32_t> * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || (n % 4) != 0) {
        fclose(f);
        return false;
    }
    out->assign((size_t) (n / 4), 0);
    const bool ok = fread(out->data(), 4, out->size(), f) == out->size();
    fclose(f);
    return ok;
}

// One source row as this tool needs it: what the manifest said, plus what we
// produced. `clips` is filled from the manifest's clips[] array, matched on
// the latent path (unique per source by construction — the cache key is a hash
// of the file's name, size, mtime and the VAE it was encoded with).
struct SourceRow {
    std::string name;        // manifest sources[].name
    std::string path;        // manifest sources[].source — the audio file
    std::string latent_rel;  // manifest sources[].latents
    std::string stem;        // latent_stem(latent_rel)
    int64_t     frames = 0;  // manifest sources[].frames — the LATENT frame count
    int64_t     n_clips = 0;

    // Filled by the run.
    bool        done       = false;
    bool        cache_hit  = false;
    int64_t     T25        = 0;   // what the tokenizer produced, before the resize
    int64_t     resized    = 0;   // T25 - frames, signed, 0 when they agreed
    std::string codes_rel;        // "codes/<stem>.i32"
    double      seconds    = 0.0;
    double      wall_ms    = 0.0;
};

struct ClipRow {
    std::string id;
    std::string latent_rel;
    int64_t     offset = 0;
    int64_t     frames = 0;
    // Filled by the run.
    std::string codec_rel;  // "codes/<id>.i32", empty when this clip got none
};

}  // namespace yue2_tokenize_detail

// ── The run ────────────────────────────────────────────────────────────────

static int yue2_tokenize_run(const Yue2TokenizeArgs & a) {
    using namespace yue2_tokenize_detail;
    const auto t_run_start = std::chrono::steady_clock::now();

    if (a.manifest.empty()) {
        fprintf(stderr, "[yue2-tokenize] --manifest <yue2_preprocess.json> is required\n");
        return 2;
    }
    if (!pm_file_exists(a.manifest)) {
        fprintf(stderr, "[yue2-tokenize] manifest not found: %s\n", a.manifest.c_str());
        return 1;
    }
    // See the header: the env var is read when the CUDA context is created, so
    // a late call is a silent no-op and every code written would carry TF32
    // error with no visible symptom.
    if (!yue2_tok_tf32_disabled()) {
        fprintf(stderr,
                "[yue2-tokenize] REFUSING: NVIDIA_TF32_OVERRIDE is not \"0\". The tokenizer was gated against the "
                "oracle with TF32 OFF; on a TF32 cuBLAS handle the subsampler alone lands two orders of magnitude "
                "outside the port's own error and every code written here would carry it silently. The variable is "
                "read when the CUDA context is CREATED, so it must be set before this process touches a backend.\n");
        return 1;
    }

    const std::string out_dir   = dirname_of(a.manifest);
    const std::string codes_dir = join_path(out_dir, "codes");

    // ── read the manifest ──────────────────────────────────────────────
    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * doc = yyjson_read_file(a.manifest.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-tokenize] cannot parse %s: %s\n", a.manifest.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_obj(root)) {
        fprintf(stderr, "[yue2-tokenize] %s has no top-level object\n", a.manifest.c_str());
        yyjson_doc_free(doc);
        return 1;
    }

    const std::string format = pm_js_str(root, "format");
    if (format != "yue2-preprocess-v1") {
        // Not fatal: the codec_ids slot was specified as an optional field that
        // does NOT bump `format` (yue2-preprocess-run.h:137-141), so a later v2
        // should still work here. Say so rather than guessing silently.
        fprintf(stderr, "[yue2-tokenize] NOTE: manifest format is \"%s\", not \"yue2-preprocess-v1\". Continuing — "
                        "codec_ids is a format-compatible addition — but check the field names if this fails.\n",
                format.c_str());
    }
    // Prefer the derived rate over the stated `frame_rate` field: sample_rate /
    // downsampling_ratio is what the latent cache's fps actually IS, and a
    // stale `frame_rate` would let a mismatch through the gate below.
    const int64_t man_sr    = pm_js_i64(root, "sample_rate", 0);
    const int64_t man_ratio = pm_js_i64(root, "downsampling_ratio", 0);
    const double  man_fps   = (man_sr > 0 && man_ratio > 0) ? (double) man_sr / (double) man_ratio : 0.0;
    const int64_t latent_dim  = pm_js_i64(root, "latent_dim", 0);
    const int64_t clip_frames = pm_js_i64(root, "clip_frames", 0);
    if (clip_frames <= 0) {
        fprintf(stderr, "[yue2-tokenize] %s does not state clip_frames\n", a.manifest.c_str());
        yyjson_doc_free(doc);
        return 1;
    }

    // sources[]
    std::vector<SourceRow> sources;
    {
        yyjson_val * sarr = yyjson_obj_get(root, "sources");
        if (!sarr || !yyjson_is_arr(sarr)) {
            fprintf(stderr, "[yue2-tokenize] %s has no sources[] array — this tool works per SOURCE file, not per "
                            "clip (a clip is a slice of a source's codes, never its own encode)\n",
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
            s.frames     = pm_js_i64(it, "frames", 0);
            s.n_clips    = pm_js_i64(it, "clips", 0);
            s.seconds    = (double) s.frames / (man_fps > 0.0 ? man_fps : 25.0);
            s.stem       = latent_stem(s.latent_rel);
            if (s.path.empty() || s.frames <= 0 || s.stem.empty()) {
                fprintf(stderr, "[yue2-tokenize] sources[%zu] is missing source/frames/latents — refusing rather "
                                "than writing codes nothing can be matched to\n",
                        i);
                yyjson_doc_free(doc);
                return 1;
            }
            sources.push_back(std::move(s));
        }
    }

    // clips[], indexed by latent path so a clip finds its source without
    // re-deriving the cache key.
    std::vector<ClipRow>                clips;
    std::map<std::string, std::vector<size_t>> clips_by_latent;
    {
        yyjson_val * carr = yyjson_obj_get(root, "clips");
        if (!carr || !yyjson_is_arr(carr)) {
            fprintf(stderr, "[yue2-tokenize] %s has no clips[] array\n", a.manifest.c_str());
            yyjson_doc_free(doc);
            return 1;
        }
        size_t       i = 0, n = 0;
        yyjson_val * it = nullptr;
        yyjson_arr_foreach(carr, i, n, it) {
            if (!yyjson_is_obj(it)) {
                continue;
            }
            ClipRow c;
            c.id         = pm_js_str(it, "id");
            c.latent_rel = pm_js_str(it, "latents");
            c.offset     = pm_js_i64(it, "offset_frames", 0);
            c.frames     = pm_js_i64(it, "frames", 0);
            if (c.id.empty() || c.frames <= 0) {
                fprintf(stderr, "[yue2-tokenize] clips[%zu] has no id or no frame count\n", i);
                yyjson_doc_free(doc);
                return 1;
            }
            clips_by_latent[c.latent_rel].push_back(clips.size());
            clips.push_back(std::move(c));
        }
    }
    yyjson_doc_free(doc);
    doc  = nullptr;
    root = nullptr;

    // ── pick the sources this run touches ──────────────────────────────
    std::vector<size_t> picked;
    for (size_t i = 0; i < sources.size(); i++) {
        if (!yp_contains_ci(sources[i].name, a.only)) {
            continue;
        }
        picked.push_back(i);
        if (a.limit > 0 && (int) picked.size() >= a.limit) {
            break;
        }
    }
    if (picked.empty()) {
        fprintf(stderr, "[yue2-tokenize] no source in %s matches --only '%s'\n", a.manifest.c_str(), a.only.c_str());
        return 1;
    }

    // ── load the tokenizer (MERT blocks + head, one GGUF) ──────────────
    std::string tok_path = a.tok_path;
    if (tok_path.empty()) {
        if (a.models_dir.empty()) {
            fprintf(stderr, "[yue2-tokenize] one of --tok <yue2-tok-*.gguf> or --models <dir> is required\n");
            return 2;
        }
        if (!yue2_tok_head_find(a.models_dir, &tok_path)) {
            fprintf(stderr, "[yue2-tokenize] no yue2-tok-*.gguf under %s/yue2 or %s\n", a.models_dir.c_str(),
                    a.models_dir.c_str());
            return 1;
        }
        // Discovery's quant ladder ranks f16 ahead of f32 (yue2-tok-head.h's
        // note on yue2_tok_head_find). That is the right default here — f16
        // storage costs a fraction of the bf16 gap the reference itself runs
        // at — but say which file was picked so it is never a surprise.
        fprintf(stderr, "[yue2-tokenize] --tok not given; discovery picked %s\n", tok_path.c_str());
    }

    static Yue2MertModel m;  // static: keeps backend handles off the stack, as the other one-shot commands do
    std::string          err;
    if (!yue2_mert_load(&m, tok_path, /*want_blocks=*/true, &err)) {
        fprintf(stderr, "[yue2-tokenize] MERT load failed: %s\n", err.c_str());
        return 1;
    }
    static Yue2TokHead head;
    if (!yue2_tok_head_load(tok_path, &head, &err)) {
        fprintf(stderr, "[yue2-tokenize] tokenizer head load failed: %s\n", err.c_str());
        yue2_mert_free(&m);
        return 1;
    }

    const int64_t MERT_SR = (int64_t) m.cfg.sample_rate;   // 24000
    const double  MERT_FPS = (double) m.cfg.frame_rate;    // 25.0
    if (MERT_SR <= 0 || !(MERT_FPS > 0.0)) {
        fprintf(stderr, "[yue2-tokenize] the tokenizer GGUF states sample_rate=%lld frame_rate=%.3f\n",
                (long long) MERT_SR, MERT_FPS);
        yue2_tok_head_free(&head);
        yue2_mert_free(&m);
        return 1;
    }
    if (man_fps > 0.0 && std::fabs(man_fps - MERT_FPS) > 1e-6) {
        // The whole alignment argument rests on both streams being 25 fps.
        // A mismatch is not something to paper over with a resize.
        fprintf(stderr,
                "[yue2-tokenize] REFUSING: the latent cache is %.4f fps (sample_rate / downsampling_ratio) but the "
                "tokenizer produces %.4f fps. Codes and latents would not describe the same frames and no resize "
                "can fix that.\n",
                man_fps, MERT_FPS);
        yue2_tok_head_free(&head);
        yue2_mert_free(&m);
        return 1;
    }

    if (!pm_mkdir_p(codes_dir)) {
        fprintf(stderr, "[yue2-tokenize] cannot create %s\n", codes_dir.c_str());
        yue2_tok_head_free(&head);
        yue2_mert_free(&m);
        return 1;
    }

    fprintf(stderr,
            "[yue2-tokenize] %zu of %zu source(s) from %s | %zu clip(s) of %lld frames at %.2f fps\n"
            "[yue2-tokenize] tokenizer %s | MERT %u blocks, %lld Hz -> %.2f fps | head %u layers, vocab %u, "
            "codec_offset %u (added at conditioning time, NOT here)\n",
            picked.size(), sources.size(), a.manifest.c_str(), clips.size(), (long long) clip_frames, man_fps,
            yue2_basename(tok_path).c_str(), (unsigned) m.w.blk.size(), (long long) MERT_SR, MERT_FPS,
            head.cfg.layers, head.cfg.vocab, head.cfg.codec_off);

    // ── per source ─────────────────────────────────────────────────────
    Yue2MertEncGraph  g;
    const std::string tmp_dir = join_path(out_dir, ".tmp");
    const std::string tmp_wav = join_path(tmp_dir, "yue2_tokenize_decode.wav");
    bool              tmp_dir_made = false;

    size_t n_done = 0, n_cached = 0, n_failed = 0;
    double total_audio_sec = 0.0, total_encode_ms = 0.0;
    int64_t n_clip_files = 0;
    std::vector<int32_t> first_codes;  // the first source's leading codes, printed at the end

    for (size_t pi = 0; pi < picked.size(); pi++) {
        SourceRow &       s   = sources[picked[pi]];
        const auto        t0  = std::chrono::steady_clock::now();
        s.codes_rel           = "codes/" + s.stem + ".i32";
        const std::string cp  = join_path(out_dir, s.codes_rel);

        std::vector<int32_t> codes;

        if (!a.force && i32_file_has(cp, s.frames) && read_i32(cp, &codes) &&
            (int64_t) codes.size() == s.frames) {
            s.cache_hit = true;
            s.T25       = s.frames;
            n_cached++;
            fprintf(stderr, "[yue2-tokenize] %zu/%zu cached  %-46s %lld frames\n", pi + 1, picked.size(),
                    s.name.c_str(), (long long) s.frames);
        } else {
            // ── decode, the same two routes yue2-preprocess uses ──
            const std::string ext        = pm_ext_of(s.path);
            const bool        use_ffmpeg = (a.decode == "ffmpeg") || !(ext == ".wav" || ext == ".mp3");
            if (use_ffmpeg && a.ffmpeg.empty()) {
                fprintf(stderr, "[yue2-tokenize] %zu/%zu SKIP %s: %s needs ffmpeg and --ffmpeg is empty\n", pi + 1,
                        picked.size(), s.name.c_str(), ext.c_str());
                n_failed++;
                continue;
            }
            int     T48    = 0;
            float * planar = nullptr;
            if (use_ffmpeg) {
                if (!tmp_dir_made) {
                    pm_mkdir_p(tmp_dir);
                    tmp_dir_made = true;
                }
                std::string derr;
                planar = yp_read_ffmpeg(a.ffmpeg, s.path, tmp_wav, 48000, &T48, &derr);
                if (!planar) {
                    fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: %s\n", pi + 1, picked.size(), s.name.c_str(),
                            derr.c_str());
                    n_failed++;
                    continue;
                }
            } else {
                planar = yp_read_native_48k(s.path, &T48);
                if (!planar || T48 <= 0) {
                    free(planar);
                    fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: decode failed\n", pi + 1, picked.size(),
                            s.name.c_str());
                    n_failed++;
                    continue;
                }
            }

            // Decoding at 48 kHz and resampling here — rather than asking
            // ffmpeg for 24 kHz directly — is deliberate: the latents in this
            // same cache came from the 48 kHz decode of this same file, so
            // both streams start from identical samples. A second decode at a
            // second rate would put a different resampler between them.
            //
            // mono = mean of the two channels (tok_oracle.py:143). The buffer
            // is PLANAR, so channel 1 starts at T48.
            std::vector<float> mono((size_t) T48);
            const float *      L = planar;
            const float *      R = planar + T48;
            for (int i = 0; i < T48; i++) {
                mono[(size_t) i] = 0.5f * (L[i] + R[i]);
            }
            free(planar);

            int     n24 = 0;
            float * r   = audio_resample(mono.data(), T48, 48000, (int) MERT_SR, 1, &n24);
            if (!r || n24 <= 0) {
                free(r);
                fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: resample 48000 -> %lld Hz failed\n", pi + 1,
                        picked.size(), s.name.c_str(), (long long) MERT_SR);
                n_failed++;
                continue;
            }
            std::vector<float> pcm(r, r + n24);
            free(r);
            mono.clear();
            mono.shrink_to_fit();

            // ── encode ──
            Yue2MertEncodeOptions opt;  // gelu_tanh=false (erf, per the block fixtures), fp16 store off
            Yue2MertEncodeResult  res;
            if (!yue2_mert_encode(m, &g, pcm.data(), (int64_t) pcm.size(), opt, &res, &err)) {
                fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: MERT encode: %s\n", pi + 1, picked.size(),
                        s.name.c_str(), err.c_str());
                n_failed++;
                continue;
            }
            if (!yue2_tok_head_predict(&head, res.feat.data(), res.T25, &codes, &err)) {
                fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: head predict: %s\n", pi + 1, picked.size(),
                        s.name.c_str(), err.c_str());
                n_failed++;
                continue;
            }
            s.T25 = res.T25;
            total_encode_ms += res.total_ms;

            // ── the alignment resize (see the header) ──
            const int64_t delta = s.T25 - s.frames;
            if (delta > 1 || delta < -1) {
                fprintf(stderr,
                        "[yue2-tokenize] %zu/%zu FAIL %s: the tokenizer produced %lld frames but the latent cache "
                        "holds %lld. More than one frame apart means the audio this manifest names is not the audio "
                        "those latents came from — re-run yue2-preprocess rather than letting this resize hide it.\n",
                        pi + 1, picked.size(), s.name.c_str(), (long long) s.T25, (long long) s.frames);
                n_failed++;
                continue;
            }
            s.resized = delta;
            if (delta > 0) {
                codes.resize((size_t) s.frames);
            } else if (delta < 0) {
                const int32_t last = codes.empty() ? 0 : codes.back();
                codes.resize((size_t) s.frames, last);
            }

            if (!write_i32(cp, codes.data(), codes.size())) {
                fprintf(stderr, "[yue2-tokenize] %zu/%zu FAIL %s: cannot write %s\n", pi + 1, picked.size(),
                        s.name.c_str(), cp.c_str());
                n_failed++;
                continue;
            }
            fprintf(stderr,
                    "[yue2-tokenize] %zu/%zu encoded %-46s %7.1f s | latents %lld frames, codes %lld frames%s | "
                    "%.1f s (%.0fx rt)\n",
                    pi + 1, picked.size(), s.name.c_str(), s.seconds, (long long) s.frames, (long long) s.T25,
                    delta == 0 ? " (exact)" : (delta > 0 ? " -> trimmed 1" : " -> padded 1"), res.total_ms / 1000.0,
                    s.seconds / (res.total_ms / 1000.0));
        }

        // ── per-clip slices ──
        //
        // A plain contiguous span, exactly as yue2_nt_read_clip takes a clip's
        // latents out of the frame-major cache. Bounds are checked against the
        // array we just sized to `frames`, so an out-of-range clip is a loud
        // failure rather than a short file the reader would happily accept.
        bool clip_ok = true;
        const auto & idxs = clips_by_latent[s.latent_rel];
        for (size_t ci : idxs) {
            ClipRow & c = clips[ci];
            if (c.offset < 0 || c.frames <= 0 || c.offset + c.frames > (int64_t) codes.size()) {
                fprintf(stderr,
                        "[yue2-tokenize] FAIL clip \"%s\": wants frames [%lld, %lld) of a %zu-frame code array\n",
                        c.id.c_str(), (long long) c.offset, (long long) (c.offset + c.frames), codes.size());
                clip_ok = false;
                break;
            }
            const std::string rel = "codes/" + c.id + ".i32";
            const std::string p   = join_path(out_dir, rel);
            if (a.force || !i32_file_has(p, c.frames)) {
                if (!write_i32(p, codes.data() + c.offset, (size_t) c.frames)) {
                    fprintf(stderr, "[yue2-tokenize] FAIL clip \"%s\": cannot write %s\n", c.id.c_str(), p.c_str());
                    clip_ok = false;
                    break;
                }
            }
            c.codec_rel = rel;
            n_clip_files++;
        }
        if (!clip_ok) {
            n_failed++;
            continue;
        }

        if (first_codes.empty() && !codes.empty()) {
            first_codes.assign(codes.begin(), codes.begin() + (std::min<size_t>(16, codes.size())));
        }

        s.done = true;
        n_done++;
        total_audio_sec += s.seconds;
        s.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    if (tmp_dir_made) {
        hs_remove(tmp_wav);
    }
    yue2_mert_enc_graph_free(&g);
    yue2_tok_head_free(&head);
    yue2_mert_free(&m);

    if (n_done == 0) {
        fprintf(stderr, "[yue2-tokenize] nothing tokenized (%zu failed). The manifest is unchanged.\n", n_failed);
        return 1;
    }

    // ── the explicit alignment check, printed ──────────────────────────
    //
    // Not a formality. It re-derives, from the files on disk, that every clip
    // that now carries codec_ids has exactly as many codes as latent rows, and
    // that its span sits inside its source. A silent one-frame lag is the
    // failure this whole tool is built to avoid, so it is proved, not assumed.
    fprintf(stderr, "\n[yue2-tokenize] ── alignment ──\n");
    int64_t checked = 0, bad = 0;
    for (const SourceRow & s : sources) {
        if (!s.done) {
            continue;
        }
        const std::string lp = join_path(out_dir, s.latent_rel);
        long long         lbytes = 0;
        pm_stat_file(lp, &lbytes, nullptr);
        // latent_dim from the manifest, f32 per the cache's latent_dtype. This
        // re-derives the source frame count the way the trainer's own startup
        // pass does (yue2-nar-train-run.h's "every cache file, once"), so a
        // stale `frames` field shows up here rather than 137 steps in.
        const int64_t lat_frames = latent_dim > 0 ? lbytes / (latent_dim * 4) : 0;
        long long     cbytes     = 0;
        pm_stat_file(join_path(out_dir, s.codes_rel), &cbytes, nullptr);
        const int64_t code_frames = cbytes / 4;
        const bool    ok          = (code_frames == s.frames);
        fprintf(stderr, "[yue2-tokenize]   %-46s latents %lld, codes %lld  %s%s\n", s.name.c_str(),
                (long long) s.frames, (long long) code_frames, ok ? "ALIGNED" : "MISMATCH",
                s.resized == 0 ? "" : (s.resized > 0 ? "  (tokenizer ran 1 long, trimmed)"
                                                     : "  (tokenizer ran 1 short, padded)"));
        if (lat_frames > 0 && lat_frames != s.frames) {
            fprintf(stderr, "[yue2-tokenize]     WARNING: the latent file holds %lld frames, the manifest says %lld\n",
                    (long long) lat_frames, (long long) s.frames);
        }
        checked++;
        bad += ok ? 0 : 1;
    }
    for (const ClipRow & c : clips) {
        if (c.codec_rel.empty()) {
            continue;
        }
        long long cb = 0;
        pm_stat_file(join_path(out_dir, c.codec_rel), &cb, nullptr);
        if (cb != (long long) c.frames * 4) {
            fprintf(stderr, "[yue2-tokenize]   clip \"%s\": %lld bytes for %lld frames — MISMATCH\n", c.id.c_str(),
                    cb, (long long) c.frames);
            bad++;
        }
    }
    fprintf(stderr, "[yue2-tokenize]   %lld source(s) checked, %lld clip code file(s), %lld mismatch(es)\n",
            (long long) checked, (long long) n_clip_files, (long long) bad);
    if (bad) {
        fprintf(stderr, "[yue2-tokenize] REFUSING to rewrite the manifest with a misaligned corpus\n");
        return 1;
    }

    // ── rewrite the manifest ───────────────────────────────────────────
    //
    // Read again and mutable-copy, rather than rebuilding from the structs
    // above: this tool parses five of the manifest's thirty-odd fields and has
    // no business re-serialising the rest. Everything it did not touch comes
    // through byte-identical in value.
    memset(&rerr, 0, sizeof(rerr));
    doc = yyjson_read_file(a.manifest.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-tokenize] cannot re-read %s for the rewrite: %s\n", a.manifest.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_mut_doc * mdoc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * mroot = yyjson_val_mut_copy(mdoc, yyjson_doc_get_root(doc));
    yyjson_doc_free(doc);
    doc = nullptr;
    if (!mroot) {
        fprintf(stderr, "[yue2-tokenize] cannot copy the manifest for rewriting\n");
        yyjson_mut_doc_free(mdoc);
        return 1;
    }
    yyjson_mut_doc_set_root(mdoc, mroot);

    yyjson_mut_obj_remove_str(mroot, "codec_ids_present");
    yyjson_mut_obj_add_bool(mdoc, mroot, "codec_ids_present", true);
    yyjson_mut_obj_remove_str(mroot, "codec_ids_note");
    // The one sentence that stops the two sides double-offsetting. It says RAW,
    // it names the constant, and it names the function that adds it.
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "codec_ids_note",
                              "Per-clip \"codec_ids\" is a manifest-relative path to little-endian int32 values "
                              "covering exactly that clip's frame range, one per latent frame. They are RAW CODES "
                              "in [0, 32768), NOT pre-offset YuE2 token ids: the + codec_offset (151853) is added "
                              "by yue2_nar_train_cond_ids at conditioning time and nowhere else. A clip without "
                              "the field trains in the text-only regime.");
    yyjson_mut_obj_remove_str(mroot, "codec_ids_producer");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "codec_ids_producer",
                              (std::string("ace-train yue2-tokenize ") + ACE_VERSION).c_str());
    yyjson_mut_obj_remove_str(mroot, "codec_ids_tokenizer");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "codec_ids_tokenizer", yue2_basename(tok_path).c_str());
    yyjson_mut_obj_remove_str(mroot, "codec_ids_created_at");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "codec_ids_created_at", pm_iso8601_utc_now().c_str());

    // Both loops below match on the object's OWN identifying field, not on its
    // position in the array. The parse pass skipped any non-object entry, so
    // indexing by position would silently put one source's codes on the next
    // source's row if a manifest ever carried a stray null — a mislabelling
    // nothing downstream could detect.
    //
    // sources[]: the whole-source array, informational. The trainer reads
    // clips[] only (yue2-preprocess-run.h's closing note), so this is for logs,
    // re-runs and diffing a slice against the song it came from.
    {
        std::map<std::string, const SourceRow *> by_latent;
        for (const SourceRow & s : sources) {
            if (s.done) {
                by_latent[s.latent_rel] = &s;
            }
        }
        yyjson_mut_val * sarr = yyjson_mut_obj_get(mroot, "sources");
        size_t           i = 0, n = 0;
        yyjson_mut_val * it = nullptr;
        if (sarr && yyjson_mut_is_arr(sarr)) {
            yyjson_mut_arr_foreach(sarr, i, n, it) {
                if (!yyjson_mut_is_obj(it)) {
                    continue;
                }
                yyjson_mut_val * lv = yyjson_mut_obj_get(it, "latents");
                if (!lv || !yyjson_mut_is_str(lv)) {
                    continue;
                }
                auto f = by_latent.find(yyjson_mut_get_str(lv));
                if (f == by_latent.end()) {
                    continue;
                }
                yyjson_mut_obj_remove_str(it, "codec_ids");
                yyjson_mut_obj_add_strcpy(mdoc, it, "codec_ids", f->second->codes_rel.c_str());
            }
        }
    }

    // clips[]: the field the reader actually consumes.
    int64_t written = 0;
    {
        std::map<std::string, const ClipRow *> by_id;
        for (const ClipRow & c : clips) {
            if (!c.codec_rel.empty()) {
                by_id[c.id] = &c;
            }
        }
        yyjson_mut_val * carr = yyjson_mut_obj_get(mroot, "clips");
        size_t           i = 0, n = 0;
        yyjson_mut_val * it = nullptr;
        if (carr && yyjson_mut_is_arr(carr)) {
            yyjson_mut_arr_foreach(carr, i, n, it) {
                if (!yyjson_mut_is_obj(it)) {
                    continue;
                }
                yyjson_mut_val * iv = yyjson_mut_obj_get(it, "id");
                if (!iv || !yyjson_mut_is_str(iv)) {
                    continue;
                }
                auto f = by_id.find(yyjson_mut_get_str(iv));
                if (f == by_id.end()) {
                    continue;
                }
                yyjson_mut_obj_remove_str(it, "codec_ids");
                yyjson_mut_obj_add_strcpy(mdoc, it, "codec_ids", f->second->codec_rel.c_str());
                written++;
            }
        }
    }

    size_t mlen  = 0;
    char * mjson = yyjson_mut_write(mdoc, YYJSON_WRITE_PRETTY, &mlen);
    yyjson_mut_doc_free(mdoc);
    if (!mjson) {
        fprintf(stderr, "[yue2-tokenize] cannot serialize the rewritten manifest\n");
        return 1;
    }
    const bool wrote = pm_write_atomic(a.manifest, std::string(mjson, mlen));
    free(mjson);
    if (!wrote) {
        fprintf(stderr, "[yue2-tokenize] cannot write %s\n", a.manifest.c_str());
        return 1;
    }

    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run_start).count();
    char rt[64];
    if (total_encode_ms > 0.0) {
        snprintf(rt, sizeof(rt), ", %.0fx realtime on the encode", total_audio_sec / (total_encode_ms / 1000.0));
    } else {
        snprintf(rt, sizeof(rt), ", nothing re-encoded");
    }
    fprintf(stderr,
            "\n[yue2-tokenize] done: %zu source(s) tokenized (%zu from cache, %zu failed), %lld of %zu clip(s) now "
            "carry codec_ids, %.2f min of audio in %.1f s wall%s\n",
            n_done, n_cached, n_failed, (long long) written, clips.size(), total_audio_sec / 60.0, wall_s, rt);
    if (written < (int64_t) clips.size()) {
        fprintf(stderr,
                "[yue2-tokenize] NOTE: %lld clip(s) have no codec_ids and will train text-only. That is the "
                "reader's documented behaviour for a mixed manifest, not an error — but a partial run is usually "
                "--only or --limit, so re-run without them if you meant the whole corpus.\n",
                (long long) ((int64_t) clips.size() - written));
    }
    if (!first_codes.empty()) {
        std::string preview;
        for (size_t i = 0; i < first_codes.size(); i++) {
            char b[24];
            snprintf(b, sizeof(b), "%s%d", i ? ", " : "", (int) first_codes[i]);
            preview += b;
        }
        fprintf(stderr, "[yue2-tokenize] first 16 codes of the first source: [%s]\n", preview.c_str());
    }
    return n_failed ? 1 : 0;
}
