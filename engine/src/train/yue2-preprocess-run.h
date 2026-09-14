#pragma once
// train/yue2-preprocess-run.h — `ace-train yue2-preprocess`: a folder of audio
// -> cached YuE2 VAE latents + the manifest `yue2-nar-train` reads.
//
// HOT-Step file, TRAINING-SIDE (ace-train only). Phase 4 of
// docs/plans/yue2/08-nar-lora-trainer.md, the data half. Native port of the
// upstream trainer's trainer_core/data.py (scan -> caption pairing -> encode
// once -> cache -> cut fixed clips), with the encode running through
// yue2/yue2-vae-encode.h instead of torch.
//
// ── What one run does ──────────────────────────────────────────────────────
//
//   ace-train yue2-preprocess --audio <folder> --out <cache dir>
//                             --models <dir> [--vae standard|legacy|<path>]
//                             [--clip-seconds 10] [--caption-mode ace|txt|default|none]
//                             [--default-caption "..."] [--only <substr>]
//                             [--ffmpeg <path>] [--force]
//                             [--captions-only]   refill caption/lyrics in an
//                                                 existing manifest; no VAE, no
//                                                 decode, nothing else touched
//
//   1. Flat scan of --audio for .wav/.flac/.mp3/.ogg/.m4a (no recursion —
//      data.py iterates one directory and so do we).
//   2. Caption (and, in `ace` mode, lyrics) from the same-named .txt, or a
//      fixed --default-caption, or none.
//   3. Decode to 48 kHz stereo f32 PLANAR (see "Decoding" below).
//   4. yue2_vae_encode_tiled -> posterior MEAN, transposed to [frames, 64] and
//      cached per SOURCE FILE as raw f32 under <out>/latents/.
//   5. Cut into fixed clips of --clip-seconds; the short tail is DROPPED; a
//      file shorter than one clip is skipped with a warning.
//   6. Write <out>/yue2_preprocess.json.
//
// A rerun over an unchanged folder re-encodes nothing: step 4 is keyed on the
// file's identity (below) and step 5 is pure arithmetic over the cache.
//
// ── Decoding: the repo's own path, no new decoder, no new resampler ────────
//
// Two routes, picked per file, exactly the split `ace-train preprocess` already
// ships (train/preprocess-run.h:272-345):
//
//   .wav / .mp3  -> read the bytes through hs_fopen (UTF-8 paths; the CRT's
//                   narrow fopen cannot see a non-ASCII filename on Windows)
//                   and hand the buffer to audio_read_48k_buf(), which sniffs
//                   RIFF-vs-MP3 from the content, upmixes mono, and resamples
//                   to 48 kHz with the repo's Kaiser-windowed polyphase
//                   resampler (audio-resample.h).
//   everything   -> ffmpeg `-ac 2 -ar 48000 -c:a pcm_f32le` into a temp WAV,
//   else            read back with audio_io_read_wav_buf(). Same call shape as
//                   cmd_mm3_preprocess. Without --ffmpeg on PATH these files
//                   are SKIPPED with a named reason, never silently dropped.
//
// `--decode ffmpeg` forces every file down the second route, so a corpus that
// mixes WAV with FLAC can be given one single resampler instead of two.
//
// Both routes produce PLANAR stereo (index = ch*S + s), which is the exact
// layout yue2_vae_encode_tiled consumes — no shuffling here. Getting that wrong
// is not a subtle bug: cmd_mm3_preprocess's own comment records five LoRA runs
// trained on audio that was silently an octave down because an interleaved
// buffer was sliced as if it were planar.
//
// audio_io_read_wav_buf ALSO reports the rate it found, and the ffmpeg route
// checks it against the VAE's own sample_rate before encoding. There is no
// path here that resamples to something other than 48 kHz, so a VAE config
// asking for a different rate is refused up front rather than fed wrong audio.
//
// ── Cache identity (and where it deliberately differs from data.py) ────────
//
// data.py hashes `path|size|mtime_ns|clip_seconds|v1` with sha256. Ours is a
// 64-bit FNV-1a over
//
//     <basename>|<size>|<mtime_ms>|<vae variant>|<vae file>|v1
//
// rendered as 16 hex characters. Three deliberate differences:
//
//   * FNV-1a, not sha256. There is no sha256 in this tree and this is a cache
//     key, not a security boundary — the only property that matters is that
//     equal inputs give equal keys.
//   * The VAE VARIANT AND FILE are in the key; data.py's are not. A latent
//     encoded by yue2-vae-legacy, or by a differently-quantized standard VAE,
//     is a DIFFERENT latent, and reusing it across a variant switch would
//     poison a whole run with no symptom but worse loss.
//   * clip_seconds is NOT in the key, and data.py's is. Cached latents do not
//     depend on the clip length at all — the clip length only decides how the
//     cache is SLICED — so including it means a corpus is fully re-encoded
//     just to try 15 s clips. The clip length is instead guarded at the
//     MANIFEST level: a rerun with a different --clip-seconds over an existing
//     <out> is refused unless --force, and with --force it re-cuts from the
//     cache in seconds without touching the VAE.
//
// Paths are not canonicalised and the scan does not recurse, so a basename is
// unique within one --audio folder; the full path adds fragility (a relative
// vs absolute --audio would invalidate every entry) and no discrimination.
//
// ── Latent layout on disk: FRAME-MAJOR, and say so out loud ───────────────
//
// <out>/latents/<stem>.f32 holds frames * latent_dim raw f32 with
//
//     index = t * latent_dim + c      (t in [0,frames), c in [0,64))
//
// so a clip is ONE CONTIGUOUS SPAN starting at `offset_frames * latent_dim`.
// The manifest states it as `latent_layout: "frame_major"` and
// `latent_index: "t * latent_dim + c"` rather than leaving it to be inferred.
//
// This costs a host-side transpose here, because `yue2_vae_encode_tiled`
// emits the OTHER layout (channel-major, index = c*frames + t). The transpose
// is on this side on purpose, and the reason is worth stating because the
// opposite choice reads just as defensible from the graph contract:
//
//   * `x_t` — the tensor a clip's latents become — is `[T, LD]` CHANNEL-
//     FASTEST, i.e. frame-major (`yue2-nar-train-graph.h`'s Yue2NarTrainHostInputs
//     note, and `yue2_nar_velocity`'s own `state`). That is the consumer.
//   * upstream data.py caches `z.T` -> `[frames, 64]`, the same way.
//   * 09-nar-train-graph-contract.md §7's "vtarget is uploaded already
//     transposed to [64, 250] channel-major" is about the UPLOAD of the loss
//     target, a step further on. Reading it as the cache layout is the easy
//     mistake, and it is the one this file deliberately does not make.
//
// The two orderings have identical size and both produce finite, plausible
// targets when swapped — which is exactly why the layout is written into the
// manifest and asserted by the reader rather than agreed in conversation.
// This project has already paid for that lesson once (the MM3 RVQ encoder's
// frame-major/channel-major mismatch).
//
// `train/yue2-nar-train-run.h`'s `yue2_nt_load_manifest` / `yue2_nt_read_clip`
// are THE consumer of everything written here; that reader is the contract
// this writer is matched to, field name by field name.
//
// ── `codec_ids`: the reserved slot for the community token encoder ─────────
//
// 08-nar-lora-trainer.md §5: another contributor is building an audio ->
// semantic-token encoder. When it lands, ground-truth codec ids per clip make
// the trainer condition exactly as inference does, and the manifest must not
// need a schema bump to carry them.
//
// The slot is therefore SPECIFIED NOW and left EMPTY:
//
//   * top level: `"codec_ids_present": false` — a reader branches on this once
//     instead of probing every clip.
//   * per clip (RESERVED, never written by this version): `"codec_ids"`, a
//     path relative to the manifest, pointing at little-endian int32 ids for
//     exactly this clip's frame range. Absent = the text-only regime of the
//     upstream recipe, which is what phase 4 trains.
//
// A future producer fills both; `format` stays `yue2-preprocess-v1`, because
// an optional field that consumers already know to look for is not a schema
// change. A consumer that sees `codec_ids_present: true` but no per-clip field
// on some clip should treat that clip as text-only rather than guess.
//
// ── `caption` and `lyrics`: the AR prefix's two halves ─────────────────────
//
// Same posture as `codec_ids`, and for the same reason: optional fields whose
// contract lives in the file, not in a plan document. Added after the v1
// manifest shipped, so `format` does not move.
//
//   * per SOURCE and per CLIP: `caption` — the STYLE string, the descriptive
//     prose that goes in the AR prefix's `[Tags]` block — and `lyrics`, the
//     tagged sheet that goes in `[Lyrics]`. The AR trainer reads sources[]
//     (whole songs, docs/plans/yue2/14-ar-lora-contract.md §5.1) and the NAR
//     trainer reads clips[], so both rows carry both strings.
//   * top level: `caption_format`, one of
//       "ace-sidecar"  parsed out of an ACE Option-A `<stem>.txt`: genre, bpm,
//                      key and the rest are stripped, and `lyrics` is filled
//                      where the sidecar had a lyric sheet.
//       "plain"        a free-form caption, no lyrics (--caption-mode txt or
//                      default). A consumer must REFUSE such a caption if it
//                      contains a `lyrics:` line, because that is a whole
//                      sidecar fed in as a style — see below.
//       "none"         the empty-style text-only prefix.
//   * top level: `caption_note`, the same contract in prose, plus
//     `caption_producer` / `caption_updated_at` when --captions-only wrote them.
//
// BACKWARDS COMPATIBLE BY CONSTRUCTION. A manifest written before these fields
// existed simply has no `lyrics` and no `caption_format`; every reader in this
// tree resolves a missing string to "" and treats an absent format as "plain".
// Nothing is refused for being old.
//
// The reason `caption_format` is WRITTEN rather than inferred is contract §5.2
// item 3: a caption that happens to contain a whole ACE sidecar trains at full
// speed on a mangled prefix — `bpm: 121` and the entire lyric sheet inside
// `[Tags]`, `[Lyrics]` empty — and nothing downstream can catch it. Declaring
// the shape is the same move `latent_layout` makes, for the same reason.
//
// `--captions-only` refills exactly these fields in an existing manifest and
// carries every other key through untouched, so a corpus that already cost a
// VAE pass and a tokenizer pass does not have to be rebuilt to gain a style
// and a lyric sheet. See yue2_preprocess_captions_only().
//
// ── TF32 must be off before the first CUDA context ─────────────────────────
//
// yue2-vae-encode.h's header measures it: with ggml's default TF32 cuBLAS
// handle the encoder lands at rel-L2 5.0e-2 against the oracle, 50x the parity
// gate, because the error snowballs through the six blocks. Every latent this
// tool writes would carry it, silently, and the only visible symptom would be
// a trainer that learns slightly the wrong thing.
//
// The env var is read by the CUDA driver when the context is CREATED, so
// setting it here would be too late if anything had already initialised a
// backend. The caller (cmd_yue2_preprocess in ace-train.cpp) sets it as its
// first statement; this file asserts it and REFUSES to run otherwise rather
// than emitting a corpus it cannot stand behind.

#include "audio-io.h"          // audio_read_48k_buf / audio_io_read_wav_buf (+ audio-resample.h)
#include "hot-step-fsutf8.h"   // hs_fopen/hs_system/hs_remove — UTF-8 paths, not ANSI
#include "version.h"           // ACE_VERSION, stamped into the manifest's producer
#include "yyjson.h"

#include "train/preprocess-io.h"  // pm_* path/string/atomic-write helpers
#include "train/yue2-sidecar.h"      // the ACE Option-A sidecar parser, shared with yue2-ar-train
#include "yue2/yue2-model.h"
#include "yue2/yue2-vae-encode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef _WIN32
#    include <dirent.h>
#endif

// ── Arguments ──────────────────────────────────────────────────────────────

struct Yue2PreprocessArgs {
    std::string audio_dir;   // --audio          (required)
    std::string out_dir;     // --out            (required)
    std::string models_dir;  // --models         search dir, same as yue2-probe
    // --vae: "standard" | "legacy" | a path to a yue2-vae-<variant>-<quant>.gguf.
    // A path also supplies --models when that was not given.
    std::string vae_arg = "standard";

    std::string caption_mode    = "txt";  // ace | txt | default | none
    std::string default_caption;          // --default-caption, used by caption-mode=default
    std::string only;                     // --only <substr>, case-insensitive, on the basename
    std::string ffmpeg = "ffmpeg";        // --ffmpeg, "" disables the non-WAV/MP3 route
    std::string decode = "auto";          // auto | ffmpeg

    double  clip_seconds = 10.0;  // --clip-seconds (250 frames at 25 fps)
    // yue2-vae-encode.h's own shipped defaults, in frames: a 30 s core with a
    // 20-frame halo (its required receptive field is 10 frames).
    int64_t tile_frames  = 750;   // --tile-frames
    int64_t halo_frames  = 20;    // --halo-frames
    int     limit        = 0;     // --limit N, debug: first N matching files
    bool    force        = false; // --force, re-cut over a manifest with a different clip length
    // --captions-only: touch NOTHING but the caption/lyrics fields of an
    // existing <out>/yue2_preprocess.json. No VAE, no decode, no CUDA context,
    // and every other field — codec_ids included — is carried through byte for
    // byte. See yue2_preprocess_captions_only().
    bool    captions_only = false;
};

// ── Small helpers ──────────────────────────────────────────────────────────

// 64-bit FNV-1a, rendered as 16 hex characters. See the header note on why
// this is not sha256.
static std::string yp_key_hex(const std::string & s) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < s.size(); i++) {
        h ^= (uint64_t) (unsigned char) s[i];
        h *= 1099511628211ULL;
    }
    char b[24];
    snprintf(b, sizeof(b), "%016llx", (unsigned long long) h);
    return std::string(b);
}

// Directory listing that survives a non-ASCII filename on Windows.
//
// NOT registry_list_dir(): that one is FindFirstFileA, i.e. the ANSI codepage,
// which is the exact failure hot-step-fsutf8.h was written for (a 14-track
// album that preprocessed to zero songs). Names come back UTF-8 on both
// platforms. Files only; subdirectories are ignored, matching data.py's
// `folder.iterdir()` + `is_file()`.
static void yp_list_dir(const std::string & dir, std::vector<std::string> * names) {
#ifdef _WIN32
    const std::wstring  pattern = hs_widen(dir) + L"\\*";
    WIN32_FIND_DATAW    fd;
    HANDLE              h = FindFirstFileW(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue;
        }
        const int len = WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, nullptr, 0, nullptr, nullptr);
        if (len <= 1) {
            continue;
        }
        std::vector<char> u8((size_t) len);
        WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, u8.data(), len, nullptr, nullptr);
        names->push_back(std::string(u8.data()));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return;
    }
    struct dirent * e;
    while ((e = readdir(d)) != nullptr) {
        const std::string full = dir + "/" + e->d_name;
        HS_STAT_T         st;
        if (hs_stat(full, &st) == 0 && HS_ISREG(st.st_mode)) {
            names->push_back(e->d_name);
        }
    }
    closedir(d);
#endif
}

static bool yp_is_audio_ext(const std::string & ext) {
    // data.py's AUDIO_EXTENSIONS, unchanged.
    return ext == ".wav" || ext == ".flac" || ext == ".mp3" || ext == ".ogg" || ext == ".m4a";
}

// Case-insensitive substring test (the --only filter).
static bool yp_contains_ci(const std::string & hay, const std::string & needle) {
    if (needle.empty()) {
        return true;
    }
    return pm_lower(hay).find(pm_lower(needle)) != std::string::npos;
}

// Read the whole of `path` as text, trimmed. Returns false when absent.
static bool yp_read_text(const std::string & path, std::string * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    std::string acc;
    char        buf[4096];
    size_t      n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        acc.append(buf, n);
    }
    fclose(f);
    // Strip a UTF-8 BOM: it would otherwise ride into the style string and
    // tokenize as a stray glyph in every caption from a Notepad-saved .txt.
    if (acc.size() >= 3 && (unsigned char) acc[0] == 0xEF && (unsigned char) acc[1] == 0xBB &&
        (unsigned char) acc[2] == 0xBF) {
        acc.erase(0, 3);
    }
    *out = pm_trim(acc);
    return true;
}

// Replace the last extension with ".txt" (data.py's `file.with_suffix(".txt")`).
static std::string yp_caption_path(const std::string & audio_path) {
    const size_t slash = audio_path.find_last_of("/\\");
    const size_t dot   = audio_path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return audio_path + ".txt";
    }
    return audio_path.substr(0, dot) + ".txt";
}

// ── caption / lyrics, per source (contract §5.2) ───────────────────────────
//
// Four modes, and the difference between the first two is the whole point:
//
//   ace      parse `<stem>.txt` as an ACE Option-A sidecar (train/yue2-sidecar.h)
//            and take its `caption:` prose as the STYLE and its `lyrics:` sheet
//            as the LYRICS. `genre`, `bpm`, `key` and the rest are parsed and
//            then DISCARDED — they are separate fields, and a YuE2 style prompt
//            carrying them is the field-noise problem, not a richer prompt.
//   txt      the whole `<stem>.txt` becomes the caption, verbatim. That is
//            data.py's behaviour and it is right for a corpus whose .txt files
//            hold nothing but a description. Against a HOT-Step dataset sidecar
//            it is wrong in the expensive way: `bpm: 121` and the entire lyric
//            sheet land inside [Tags] and [Lyrics] comes out empty, which
//            trains at full speed on a mangled prefix.
//   default  --default-caption for every source; no lyrics.
//   none     empty style, empty lyrics — the upstream text-only prefix.
//
// Why `<stem>.mm3.txt` is NOT read here: it is a MiniMax Structured Caption,
// and its first line of substance is "Basic Attributes: bpm is 121. key is D,
// and scale is major." — exactly the shape a YuE2 style must not carry. The ACE
// sidecar's `caption:` line is already the descriptive prose.
// The three sidecar fields the style TEMPLATE consumes (yue2_style_string).
// Carried through the manifest as their own keys so a trainer on a machine
// without the dataset (the living-room box has no M:) can still build the
// template; they never enter `caption` itself.
struct YpStyleMeta {
    std::string genre, bpm, key;
};

static void yp_resolve_caption(const Yue2PreprocessArgs & a, const std::string & audio_path,
                               std::string * caption, std::string * lyrics, bool * sidecar_hit,
                               YpStyleMeta * sm = nullptr) {
    caption->clear();
    lyrics->clear();
    *sidecar_hit = false;
    if (sm) {
        *sm = YpStyleMeta{};
    }
    if (a.caption_mode == "ace") {
        std::string raw;
        if (yp_read_text(yp_caption_path(audio_path), &raw)) {
            std::map<std::string, std::string> meta;
            *sidecar_hit = yue2_sidecar_parse_meta(raw, caption, lyrics, &meta);
            if (sm) {
                auto get = [&](const char * k) {
                    auto it = meta.find(k);
                    return it == meta.end() ? std::string() : it->second;
                };
                sm->genre = get("genre");
                sm->bpm   = get("bpm");
                sm->key   = get("key");
            }
        }
    } else if (a.caption_mode == "txt") {
        std::string cap;
        if (yp_read_text(yp_caption_path(audio_path), &cap)) {
            *caption = cap;
        }
    } else if (a.caption_mode == "default") {
        *caption = pm_trim(a.default_caption);
    }
    // "none" -> both empty, which is the empty-style prefix at training time
}

// What the manifest DECLARES about the shape of `caption`. A consumer branches
// on this rather than sniffing the string: yue2-ar-train refuses a caption
// containing a `lyrics:` line unless the format says "ace-sidecar", because a
// whole sidecar fed in as a style is the one mistake nothing downstream can
// catch (yue2-ar-train-run.h's yue2_at_load_manifest).
static const char * yp_caption_format(const std::string & caption_mode) {
    if (caption_mode == "ace") {
        return "ace-sidecar";
    }
    if (caption_mode == "none") {
        return "none";
    }
    return "plain";
}

// The manifest's own note for the caption/lyrics pair, written beside
// codec_ids_note for the same reason that one exists: the fields are optional,
// so a consumer needs the contract in the file rather than in a plan document.
#define YP_CAPTION_NOTE                                                                                          \
    "\"caption\" is the STYLE string that goes in the AR prefix's [Tags] block and \"lyrics\" is the tagged "     \
    "sheet that goes in [Lyrics]; both are written per source AND per clip. \"caption_format\" says what shape "  \
    "\"caption\" is in: \"ace-sidecar\" = parsed out of an ACE Option-A <stem>.txt, so it is descriptive prose "  \
    "with genre/bpm/key stripped and \"lyrics\" is populated where the sidecar had a lyric sheet; \"plain\" = a " \
    "free-form caption with no lyrics field (--caption-mode txt or default), and a consumer must REFUSE one "     \
    "that contains a `lyrics:` line, because that is a whole sidecar fed in as a style; \"none\" = the "          \
    "empty-style text-only prefix. Both fields and caption_format are OPTIONAL: a manifest written before they "  \
    "existed simply has no lyrics and no caption_format, and a reader treats that as \"plain\"."

// ── Decode to 48 kHz planar stereo ─────────────────────────────────────────

// Native route: WAV or MP3, decoded from a buffer we read ourselves so the
// path survives as UTF-8. Mirrors pp_audio_read_48k.
static float * yp_read_native_48k(const std::string & path, int * T_out) {
    *T_out   = 0;
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return nullptr;
    }
    fseek(f, 0, SEEK_END);
    const long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) {
        fclose(f);
        return nullptr;
    }
    uint8_t * buf = (uint8_t *) malloc((size_t) fsize);
    if (!buf) {
        fclose(f);
        return nullptr;
    }
    const size_t nr = fread(buf, 1, (size_t) fsize, f);
    fclose(f);
    if (nr != (size_t) fsize) {
        free(buf);
        return nullptr;
    }
    float * planar = audio_read_48k_buf(buf, (size_t) fsize, T_out);
    free(buf);
    return planar;
}

// ffmpeg route: anything ffmpeg reads -> `sr` Hz stereo f32 WAV -> planar.
// The produced rate is CHECKED, not assumed (cmd_mm3_preprocess does the same).
static float * yp_read_ffmpeg(const std::string & ffmpeg, const std::string & src, const std::string & tmp_wav,
                              int sr, int * T_out, std::string * err) {
    *T_out = 0;
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "\"%s\" -y -v error -i \"%s\" -ac 2 -ar %d -c:a pcm_f32le -f wav \"%s\"",
             ffmpeg.c_str(), src.c_str(), sr, tmp_wav.c_str());
#ifdef _WIN32
    const std::string wrapped = "\"" + std::string(cmd) + "\"";  // cmd.exe strips the outer pair
    const int         rc      = hs_system(wrapped);
#else
    const int rc = hs_system(cmd);
#endif
    if (rc != 0 || !pm_file_exists(tmp_wav)) {
        *err = "ffmpeg exit " + std::to_string(rc);
        return nullptr;
    }
    FILE * wf = hs_fopen(tmp_wav, "rb");
    if (!wf) {
        *err = "cannot reopen the ffmpeg temp wav";
        return nullptr;
    }
    fseek(wf, 0, SEEK_END);
    const long wsz = ftell(wf);
    fseek(wf, 0, SEEK_SET);
    std::vector<uint8_t> wbuf((size_t) (wsz > 0 ? wsz : 0));
    const bool           wread = wsz > 0 && fread(wbuf.data(), 1, (size_t) wsz, wf) == (size_t) wsz;
    fclose(wf);
    if (!wread) {
        *err = "cannot read the ffmpeg temp wav back";
        return nullptr;
    }
    int     got_sr = 0;
    float * planar = audio_io_read_wav_buf(wbuf.data(), wbuf.size(), T_out, &got_sr);
    if (!planar || *T_out <= 0) {
        free(planar);
        *err   = "cannot decode the ffmpeg transcode";
        *T_out = 0;
        return nullptr;
    }
    if (got_sr != sr) {
        free(planar);
        *err   = "ffmpeg produced " + std::to_string(got_sr) + " Hz, wanted " + std::to_string(sr);
        *T_out = 0;
        return nullptr;
    }
    return planar;
}

// ── VAE resolution (yue2-probe's rules, plus an exact-file check) ──────────

struct Yue2PpVae {
    Yue2VaeVariant variant     = YUE2_VAE_STANDARD;
    std::string    models_dir;
    std::string    want_path;  // non-empty when --vae named a file
};

// --vae accepts a variant token or a yue2-vae-<variant>-<quant>.gguf path.
// A path fixes both the variant and (when --models is absent) the search dir.
static bool yp_parse_vae_arg(const Yue2PreprocessArgs & a, Yue2PpVae * out, std::string * err) {
    out->models_dir = a.models_dir;
    std::string v   = a.vae_arg.empty() ? std::string("standard") : a.vae_arg;
    if (v == "standard" || v == "legacy") {
        out->variant = (v == "legacy") ? YUE2_VAE_LEGACY : YUE2_VAE_STANDARD;
        if (out->models_dir.empty()) {
            *err = "--models <dir> is required (or pass --vae <yue2-vae-*.gguf>)";
            return false;
        }
        return true;
    }

    // A path. Split it and read the variant out of the filename rather than
    // guessing — loading a legacy VAE while the manifest claims standard would
    // silently cross-contaminate every cache entry.
    for (char & ch : v) {
        if (ch == '\\') {
            ch = '/';
        }
    }
    const size_t      slash = v.find_last_of('/');
    const std::string dir   = (slash == std::string::npos) ? std::string(".") : v.substr(0, slash);
    const std::string base  = (slash == std::string::npos) ? v : v.substr(slash + 1);
    const std::string pfx   = "yue2-vae-";
    if (base.size() <= pfx.size() + 5 || base.compare(0, pfx.size(), pfx) != 0 ||
        base.compare(base.size() - 5, 5, ".gguf") != 0) {
        *err = "--vae must be standard, legacy, or a path to a yue2-vae-<variant>-<quant>.gguf (got '" + base + "')";
        return false;
    }
    const std::string rest = base.substr(pfx.size(), base.size() - pfx.size() - 5);  // "<variant>-<quant>"
    if (rest.compare(0, 9, "standard-") == 0) {
        out->variant = YUE2_VAE_STANDARD;
    } else if (rest.compare(0, 7, "legacy-") == 0) {
        out->variant = YUE2_VAE_LEGACY;
    } else {
        *err = "cannot read a VAE variant out of '" + base + "' (expected yue2-vae-standard-* or yue2-vae-legacy-*)";
        return false;
    }
    out->want_path = v;
    if (out->models_dir.empty()) {
        out->models_dir = dir;
    }
    return true;
}

// ── Cached-latent I/O ──────────────────────────────────────────────────────

// A cache file is frames * latent_dim raw f32, frame-major. Its byte size is
// therefore the whole of its metadata: a short or misaligned file is a torn
// write from an interrupted run and is re-encoded rather than trusted.
static bool yp_cache_frames(const std::string & path, int64_t latent_dim, int64_t * frames_out) {
    long long bytes = 0;
    if (!pm_stat_file(path, &bytes, nullptr) || bytes <= 0) {
        return false;
    }
    const int64_t per_frame = latent_dim * (int64_t) sizeof(float);
    if (per_frame <= 0 || (int64_t) bytes % per_frame != 0) {
        return false;
    }
    *frames_out = (int64_t) bytes / per_frame;
    return *frames_out > 0;
}

static bool yp_write_cache(const std::string & path, const std::vector<float> & latents) {
    // Small by construction — 25 fps * 64 channels * 4 B = 6.4 kB per second,
    // so a 10-minute track is under 4 MB — which is why this can go through the
    // crash-safe write-then-rename rather than streaming.
    const std::string bytes((const char *) latents.data(), latents.size() * sizeof(float));
    return pm_write_atomic(path, bytes);
}

// ── Manifest guard ─────────────────────────────────────────────────────────

// An existing yue2_preprocess.json whose clip length differs from this run's is
// refused without --force. The cost of --force is only a re-CUT (cached latents
// are clip-length independent, see the header), which is why the message says
// so: a user who meant it should not be scared into deleting the cache.
static bool yp_check_existing(const std::string & manifest_path, int64_t clip_frames, const char * variant_name,
                              bool force, std::string * err) {
    if (!pm_file_exists(manifest_path)) {
        return true;
    }
    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * doc = yyjson_read_file(manifest_path.c_str(), 0, nullptr, &rerr);
    if (!doc) {
        fprintf(stderr, "[yue2-preprocess] WARN: %s exists but does not parse (%s) — it will be replaced\n",
                manifest_path.c_str(), rerr.msg ? rerr.msg : "unknown");
        return true;
    }
    yyjson_val *  root     = yyjson_doc_get_root(doc);
    const int64_t prev_cf  = (int64_t) pm_js_i64(root, "clip_frames", 0);
    const std::string prev_var = pm_js_str(root, "vae_variant", "");
    yyjson_doc_free(doc);

    if (!prev_var.empty() && prev_var != variant_name) {
        // Not fatal: the VAE variant is part of the cache key, so the old
        // files stay valid for whoever wants them and this run writes its own.
        fprintf(stderr,
                "[yue2-preprocess] WARN: %s was built with the '%s' VAE, this run uses '%s'. The cache key "
                "includes the variant, so nothing is reused across the switch — the old latents are left in "
                "place and cost disk until you delete them.\n",
                manifest_path.c_str(), prev_var.c_str(), variant_name);
    }
    if (prev_cf > 0 && prev_cf != clip_frames && !force) {
        *err = "existing " + manifest_path + " was built with clip_frames=" + std::to_string(prev_cf) +
               ", this run wants " + std::to_string(clip_frames) +
               ". Pass --force to re-cut it (the cached latents are reused; nothing is re-encoded), or point "
               "--out somewhere else to keep both.";
        return false;
    }
    return true;
}

// ── One source file ────────────────────────────────────────────────────────

struct Yue2PpSource {
    std::string name;         // basename
    std::string path;         // full path as scanned
    std::string caption;      // the STYLE string: descriptive prose, no field noise
    std::string lyrics;       // the tagged lyric sheet, or empty for an instrumental
    YpStyleMeta sm;           // genre / bpm / key for the style template, kept OUT of caption
    // pm_safe_stem(name) + "_" + cache key. The KEY is part of it on purpose:
    // the sanitized stem alone collides (a.wav and a.flac both sanitize to
    // "a"), which would give two different songs the same clip ids and the
    // same cache file.
    std::string stem;
    std::string latent_rel;   // "latents/<stem>.f32"
    int64_t     frames = 0;
    int64_t     clips  = 0;
    bool        cache_hit = false;
    double      seconds   = 0.0;
};

// ── `--captions-only`: refill the text fields of an existing manifest ──────
//
// The cache behind `<out>/yue2_preprocess.json` costs a VAE pass and, once
// `ace-train yue2-tokenize` has run, a MERT + head pass on top. Re-running the
// full preprocess to pick up a caption mode that did not exist when the corpus
// was built would rewrite the manifest from scratch and take `codec_ids`,
// `codec_ids_present` and the whole tokenizer provenance block with it — the
// latents on disk would survive, the codes on disk would survive, and the
// manifest that names them would not.
//
// So this path does what `yue2-tokenize` does for its own field: read the
// manifest, MUTATE the fields it owns, write it back. It touches
//
//     root      caption_mode, default_caption, caption_format, caption_note,
//               caption_producer, caption_updated_at
//     sources[] caption, lyrics
//     clips[]   caption, lyrics
//
// and nothing else. Every other key is carried through by value. No model is
// loaded, no audio is decoded, and no CUDA context is created, so the TF32 rule
// that governs the encode path does not apply here and is deliberately not
// re-asserted (asserting it would be theatre — there is nothing to corrupt).
//
// Sources are matched to clips on `latents`, which is unique per source by
// construction (the cache key is in the stem). A clip whose `latents` names no
// source is left exactly as it was rather than guessed at.
static int yue2_preprocess_captions_only(const Yue2PreprocessArgs & a) {
    if (a.out_dir.empty()) {
        fprintf(stderr, "[yue2-preprocess] --captions-only needs --out <dir> (the cache holding the manifest)\n");
        return 1;
    }
    if (a.caption_mode != "ace" && a.caption_mode != "txt" && a.caption_mode != "default" &&
        a.caption_mode != "none") {
        fprintf(stderr, "[yue2-preprocess] --caption-mode must be ace, txt, default or none\n");
        return 1;
    }
    if (a.caption_mode == "default" && pm_trim(a.default_caption).empty()) {
        fprintf(stderr, "[yue2-preprocess] --caption-mode default needs a non-empty --default-caption\n");
        return 1;
    }
    const std::string manifest_path = a.out_dir + "/yue2_preprocess.json";
    if (!pm_file_exists(manifest_path)) {
        fprintf(stderr,
                "[yue2-preprocess] --captions-only needs an existing manifest; %s does not exist. Run a normal "
                "preprocess first.\n",
                manifest_path.c_str());
        return 1;
    }

    yyjson_read_err rerr;
    memset(&rerr, 0, sizeof(rerr));
    yyjson_doc * rdoc = yyjson_read_file(manifest_path.c_str(), 0, nullptr, &rerr);
    if (!rdoc) {
        fprintf(stderr, "[yue2-preprocess] cannot parse %s: %s\n", manifest_path.c_str(),
                rerr.msg ? rerr.msg : "unknown error");
        return 1;
    }
    yyjson_mut_doc * mdoc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * mroot = yyjson_val_mut_copy(mdoc, yyjson_doc_get_root(rdoc));
    yyjson_doc_free(rdoc);
    if (!mroot || !yyjson_mut_is_obj(mroot)) {
        fprintf(stderr, "[yue2-preprocess] %s has no top-level object\n", manifest_path.c_str());
        yyjson_mut_doc_free(mdoc);
        return 1;
    }
    yyjson_mut_doc_set_root(mdoc, mroot);

    // sources[]: resolve from each row's own `source` audio path.
    struct YpCapRow {
        std::string caption;
        std::string lyrics;
        YpStyleMeta sm;
    };
    auto put_sm = [&](yyjson_mut_val * obj, const YpStyleMeta & sm) {
        for (const auto & kv : { std::pair<const char *, const std::string *>{ "genre", &sm.genre },
                                 std::pair<const char *, const std::string *>{ "bpm", &sm.bpm },
                                 std::pair<const char *, const std::string *>{ "key", &sm.key } }) {
            yyjson_mut_obj_remove_str(obj, kv.first);
            yyjson_mut_obj_add_strcpy(mdoc, obj, kv.first, kv.second->c_str());
        }
    };
    std::map<std::string, YpCapRow> by_latent;
    int64_t n_src = 0, n_cap = 0, n_lyr = 0, n_missing = 0;
    {
        yyjson_mut_val * sarr = yyjson_mut_obj_get(mroot, "sources");
        if (!sarr || !yyjson_mut_is_arr(sarr)) {
            fprintf(stderr, "[yue2-preprocess] %s has no sources[] array\n", manifest_path.c_str());
            yyjson_mut_doc_free(mdoc);
            return 1;
        }
        size_t           i, n;
        yyjson_mut_val * it = nullptr;
        yyjson_mut_arr_foreach(sarr, i, n, it) {
            if (!yyjson_mut_is_obj(it)) {
                continue;
            }
            yyjson_mut_val * pv = yyjson_mut_obj_get(it, "source");
            yyjson_mut_val * lv = yyjson_mut_obj_get(it, "latents");
            if (!pv || !yyjson_mut_is_str(pv)) {
                continue;
            }
            const std::string apath = yyjson_mut_get_str(pv);
            n_src++;
            YpCapRow row;
            bool     hit = false;
            yp_resolve_caption(a, apath, &row.caption, &row.lyrics, &hit, &row.sm);
            if (a.caption_mode == "ace" && !hit) {
                fprintf(stderr, "[yue2-preprocess]   no readable ACE sidecar beside %s\n", apath.c_str());
                n_missing++;
            }
            if (!row.caption.empty()) {
                n_cap++;
            }
            if (!row.lyrics.empty()) {
                n_lyr++;
            }
            yyjson_mut_obj_remove_str(it, "caption");
            yyjson_mut_obj_add_strcpy(mdoc, it, "caption", row.caption.c_str());
            yyjson_mut_obj_remove_str(it, "lyrics");
            yyjson_mut_obj_add_strcpy(mdoc, it, "lyrics", row.lyrics.c_str());
            put_sm(it, row.sm);
            if (lv && yyjson_mut_is_str(lv)) {
                by_latent[yyjson_mut_get_str(lv)] = row;
            }
        }
    }

    int64_t n_clip = 0, n_clip_unmatched = 0;
    {
        yyjson_mut_val * carr = yyjson_mut_obj_get(mroot, "clips");
        size_t           i, n;
        yyjson_mut_val * it = nullptr;
        if (carr && yyjson_mut_is_arr(carr)) {
            yyjson_mut_arr_foreach(carr, i, n, it) {
                if (!yyjson_mut_is_obj(it)) {
                    continue;
                }
                yyjson_mut_val * lv = yyjson_mut_obj_get(it, "latents");
                if (!lv || !yyjson_mut_is_str(lv)) {
                    n_clip_unmatched++;
                    continue;
                }
                auto f = by_latent.find(yyjson_mut_get_str(lv));
                if (f == by_latent.end()) {
                    n_clip_unmatched++;
                    continue;
                }
                n_clip++;
                yyjson_mut_obj_remove_str(it, "caption");
                yyjson_mut_obj_add_strcpy(mdoc, it, "caption", f->second.caption.c_str());
                yyjson_mut_obj_remove_str(it, "lyrics");
                yyjson_mut_obj_add_strcpy(mdoc, it, "lyrics", f->second.lyrics.c_str());
                put_sm(it, f->second.sm);
            }
        }
    }

    yyjson_mut_obj_remove_str(mroot, "caption_mode");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "caption_mode", a.caption_mode.c_str());
    yyjson_mut_obj_remove_str(mroot, "default_caption");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "default_caption", a.default_caption.c_str());
    yyjson_mut_obj_remove_str(mroot, "caption_format");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "caption_format", yp_caption_format(a.caption_mode));
    yyjson_mut_obj_remove_str(mroot, "caption_note");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "caption_note", YP_CAPTION_NOTE);
    yyjson_mut_obj_remove_str(mroot, "caption_producer");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "caption_producer",
                              (std::string("ace-train yue2-preprocess --captions-only ") + ACE_VERSION).c_str());
    yyjson_mut_obj_remove_str(mroot, "caption_updated_at");
    yyjson_mut_obj_add_strcpy(mdoc, mroot, "caption_updated_at", pm_iso8601_utc_now().c_str());

    size_t mlen  = 0;
    char * mjson = yyjson_mut_write(mdoc, YYJSON_WRITE_PRETTY, &mlen);
    yyjson_mut_doc_free(mdoc);
    if (!mjson) {
        fprintf(stderr, "[yue2-preprocess] cannot serialize the rewritten manifest\n");
        return 1;
    }
    const bool wrote = pm_write_atomic(manifest_path, std::string(mjson, mlen));
    free(mjson);
    if (!wrote) {
        fprintf(stderr, "[yue2-preprocess] cannot write %s\n", manifest_path.c_str());
        return 1;
    }
    fprintf(stderr,
            "[yue2-preprocess] captions-only: %lld source(s) updated (%lld with a caption, %lld with lyrics, "
            "%lld with no readable sidecar), %lld clip(s) updated, %lld clip(s) unmatched. caption_format=%s. "
            "Latents, codes and codec_ids untouched -> %s\n",
            (long long) n_src, (long long) n_cap, (long long) n_lyr, (long long) n_missing, (long long) n_clip,
            (long long) n_clip_unmatched, yp_caption_format(a.caption_mode), manifest_path.c_str());
    return 0;
}

// ── The run ────────────────────────────────────────────────────────────────

static int yue2_preprocess_run(const Yue2PreprocessArgs & a) {
    if (a.captions_only) {
        return yue2_preprocess_captions_only(a);
    }
    // ── argument validation ────────────────────────────────────────────
    if (a.audio_dir.empty()) {
        fprintf(stderr, "[yue2-preprocess] --audio <folder> is required\n");
        return 1;
    }
    if (a.out_dir.empty()) {
        fprintf(stderr, "[yue2-preprocess] --out <dir> is required\n");
        return 1;
    }
    if (a.caption_mode != "ace" && a.caption_mode != "txt" && a.caption_mode != "default" &&
        a.caption_mode != "none") {
        fprintf(stderr, "[yue2-preprocess] --caption-mode must be ace, txt, default or none\n");
        return 1;
    }
    if (a.decode != "auto" && a.decode != "ffmpeg") {
        fprintf(stderr, "[yue2-preprocess] --decode must be auto or ffmpeg\n");
        return 1;
    }
    if (a.caption_mode == "default" && pm_trim(a.default_caption).empty()) {
        fprintf(stderr, "[yue2-preprocess] --caption-mode default needs a non-empty --default-caption\n");
        return 1;
    }
    // data.py's own bounds. Below 1 s there is not enough context for the NAR
    // to render anything; above 60 s the clip stops being a clip.
    if (!(a.clip_seconds >= 1.0 && a.clip_seconds <= 60.0)) {
        fprintf(stderr, "[yue2-preprocess] --clip-seconds must be within 1..60 (got %.3f)\n", a.clip_seconds);
        return 1;
    }
    if (a.decode == "ffmpeg" && a.ffmpeg.empty()) {
        fprintf(stderr, "[yue2-preprocess] --decode ffmpeg needs --ffmpeg <path>\n");
        return 1;
    }
    if (!pm_path_exists(a.audio_dir)) {
        fprintf(stderr, "[yue2-preprocess] --audio folder not found: %s\n", a.audio_dir.c_str());
        return 1;
    }
    // See the header: a TF32 cuBLAS handle puts this encoder 50x outside its
    // own parity gate, and the corpus would carry it invisibly.
    if (!yue2_vae_enc_tf32_disabled()) {
        fprintf(stderr,
                "[yue2-preprocess] REFUSING: NVIDIA_TF32_OVERRIDE is not 0. The YuE2 encoder is 50x outside its "
                "parity gate on a TF32 cuBLAS handle, and every cached latent would carry that error with no "
                "visible symptom. The variable is read when the CUDA context is created, so it must be set "
                "before this process touches a backend.\n");
        return 1;
    }

    // ── resolve + load the VAE (encoder half) ──────────────────────────
    Yue2PpVae   vs;
    std::string err;
    if (!yp_parse_vae_arg(a, &vs, &err)) {
        fprintf(stderr, "[yue2-preprocess] %s\n", err.c_str());
        return 1;
    }

    // static, like yue2_nar_fdcheck_main's: one-shot command, and it keeps a
    // model carrying backend handles off the stack.
    static Yue2Model m;
    yue2_discover(&m, vs.models_dir.c_str());
    const char * variant_name = YUE2_VAE_VARIANT_NAME[vs.variant];
    if (!m.vae_file[vs.variant].found) {
        fprintf(stderr, "[yue2-preprocess] no yue2-vae-%s-*.gguf under %s/yue2 or %s\n", variant_name,
                vs.models_dir.c_str(), vs.models_dir.c_str());
        return 1;
    }
    if (!vs.want_path.empty()) {
        // Discovery is best-quant-first, so it can legitimately pick a file
        // other than the one named. Refuse rather than encode a whole corpus
        // with a VAE the operator did not ask for.
        std::string got = m.vae_file[vs.variant].path;
        for (char & ch : got) {
            if (ch == '\\') {
                ch = '/';
            }
        }
        if (pm_lower(got) != pm_lower(vs.want_path)) {
            fprintf(stderr,
                    "[yue2-preprocess] --vae named %s but discovery picked %s (it takes the best quant in the "
                    "directory). Move the unwanted file out of the search path, or pass --models <dir> --vae %s "
                    "and accept the pick.\n",
                    vs.want_path.c_str(), got.c_str(), variant_name);
            return 1;
        }
    }

    if (!yue2_load_parts(&m, /*want_lm=*/false, /*want_vae=*/true, vs.variant, /*want_encoder=*/true, &err)) {
        fprintf(stderr, "[yue2-preprocess] VAE load failed: %s\n", err.c_str());
        return 1;
    }

    const Yue2VaeConfig & vc = m.vae_cfg;
    const int64_t         SR = (int64_t) vc.sample_rate;
    const int64_t         RATIO = (int64_t) vc.downsampling_ratio;
    const int64_t         LD    = (int64_t) vc.latent_dim;
    if (SR <= 0 || RATIO <= 0 || LD <= 0) {
        fprintf(stderr, "[yue2-preprocess] VAE config is incomplete (sr=%lld ratio=%lld latent_dim=%lld)\n",
                (long long) SR, (long long) RATIO, (long long) LD);
        yue2_unload(&m);
        return 1;
    }
    if ((int) vc.audio_channels != 2) {
        // Both decode routes produce exactly two planar channels and the clamp
        // loop below walks 2*T floats. A different channel count would make
        // every read past channel 0 land in the wrong place — finite,
        // plausible, and wrong.
        fprintf(stderr, "[yue2-preprocess] this VAE wants %u audio channels; this tool only decodes stereo\n",
                vc.audio_channels);
        yue2_unload(&m);
        return 1;
    }
    if (SR != 48000) {
        // Both decode routes here land on 48 kHz and nothing in this tool can
        // target another rate. Refuse instead of feeding the encoder audio at
        // the wrong rate, which would be silent and pitch-shifted.
        fprintf(stderr,
                "[yue2-preprocess] this VAE wants %lld Hz, but both decode routes here produce 48000 Hz. "
                "Refusing rather than encoding resampled-to-the-wrong-rate audio.\n",
                (long long) SR);
        yue2_unload(&m);
        return 1;
    }

    const double  fps         = (double) SR / (double) RATIO;               // 25.0
    const int64_t clip_frames = (int64_t) llround(a.clip_seconds * fps);    // 250 at the defaults
    if (clip_frames < 1) {
        fprintf(stderr, "[yue2-preprocess] --clip-seconds %.3f rounds to 0 frames at %.3f fps\n", a.clip_seconds,
                fps);
        yue2_unload(&m);
        return 1;
    }
    const int64_t core_samples = a.tile_frames * RATIO;
    const int64_t halo_samples = a.halo_frames * RATIO;
    if (a.tile_frames < 1 || halo_samples < (int64_t) YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES) {
        fprintf(stderr,
                "[yue2-preprocess] --tile-frames must be >= 1 and --halo-frames >= %lld (the encoder's receptive "
                "field); got tile=%lld halo=%lld\n",
                (long long) ((int64_t) YUE2_VAE_ENC_REQUIRED_HALO_SAMPLES / 1920), (long long) a.tile_frames,
                (long long) a.halo_frames);
        yue2_unload(&m);
        return 1;
    }

    // ── output layout + the clip-length guard ──────────────────────────
    const std::string latents_dir  = a.out_dir + "/latents";
    const std::string tmp_dir      = a.out_dir + "/.tmp";
    const std::string manifest_path = a.out_dir + "/yue2_preprocess.json";
    if (!pm_mkdir_p(latents_dir)) {
        fprintf(stderr, "[yue2-preprocess] cannot create %s\n", latents_dir.c_str());
        yue2_unload(&m);
        return 1;
    }
    if (!yp_check_existing(manifest_path, clip_frames, variant_name, a.force, &err)) {
        fprintf(stderr, "[yue2-preprocess] %s\n", err.c_str());
        yue2_unload(&m);
        return 1;
    }

    // ── scan ───────────────────────────────────────────────────────────
    std::vector<std::string> names;
    yp_list_dir(a.audio_dir, &names);
    std::sort(names.begin(), names.end());  // data.py's sorted(folder.iterdir())

    std::vector<std::string> picked;
    for (const auto & n : names) {
        if (!yp_is_audio_ext(pm_ext_of(n))) {
            continue;
        }
        if (!yp_contains_ci(n, a.only)) {
            continue;
        }
        picked.push_back(n);
        if (a.limit > 0 && (int) picked.size() >= a.limit) {
            break;
        }
    }
    if (picked.empty()) {
        const std::string filt = a.only.empty() ? std::string() : (" matching --only '" + a.only + "'");
        fprintf(stderr, "[yue2-preprocess] no .wav/.flac/.mp3/.ogg/.m4a files in %s%s\n", a.audio_dir.c_str(),
                filt.c_str());
        yue2_unload(&m);
        return 1;
    }
    fprintf(stderr, "[yue2-preprocess] %zu source file(s) in %s; VAE %s (%s), clips %.2f s = %lld frames @ %.3f fps\n",
            picked.size(), a.audio_dir.c_str(), m.vae_file[vs.variant].name.c_str(), variant_name, a.clip_seconds,
            (long long) clip_frames, fps);

    // ── per-file encode (cached) ───────────────────────────────────────
    // Captured BEFORE the loop because the manifest is written after
    // yue2_unload(), and nothing promises the model's file records survive it.
    const std::string vae_base = m.vae_file[vs.variant].name;
    const std::string vae_path = m.vae_file[vs.variant].path;
    std::vector<Yue2PpSource> sources;
    size_t  n_cached = 0, n_encoded = 0, n_skipped = 0, n_failed = 0;
    int64_t total_clips = 0;
    double  total_sec   = 0.0;
    const std::string tmp_wav = tmp_dir + "/yue2_decode.wav";
    bool    tmp_dir_made = false;

    for (size_t i = 0; i < picked.size(); i++) {
        const std::string name = picked[i];
        const std::string path = a.audio_dir + "/" + name;
        const std::string ext  = pm_ext_of(name);

        long long fbytes = 0, fmtime = 0;
        if (!pm_stat_file(path, &fbytes, &fmtime)) {
            fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: cannot stat\n", i + 1, picked.size(), name.c_str());
            n_failed++;
            continue;
        }

        Yue2PpSource s;
        s.name = name;
        s.path = path;
        bool sidecar_hit = false;
        yp_resolve_caption(a, path, &s.caption, &s.lyrics, &sidecar_hit, &s.sm);
        (void) sidecar_hit;

        const std::string key = yp_key_hex(name + "|" + std::to_string(fbytes) + "|" + std::to_string(fmtime) + "|" +
                                           variant_name + "|" + vae_base + "|v1");
        s.stem                        = pm_safe_stem(name) + "_" + key;
        const std::string latent_rel  = "latents/" + s.stem + ".f32";
        const std::string latent_path = a.out_dir + "/" + latent_rel;
        s.latent_rel = latent_rel;

        int64_t frames = 0;
        if (yp_cache_frames(latent_path, LD, &frames)) {
            s.cache_hit = true;
            n_cached++;
        } else {
            // ── decode ──
            const bool use_ffmpeg = (a.decode == "ffmpeg") || !(ext == ".wav" || ext == ".mp3");
            if (use_ffmpeg && a.ffmpeg.empty()) {
                fprintf(stderr, "[yue2-preprocess] %zu/%zu SKIP %s: %s needs ffmpeg and --ffmpeg is empty\n", i + 1,
                        picked.size(), name.c_str(), ext.c_str());
                n_skipped++;
                continue;
            }
            int     T      = 0;
            float * planar = nullptr;
            if (use_ffmpeg) {
                if (!tmp_dir_made) {
                    pm_mkdir_p(tmp_dir);
                    tmp_dir_made = true;
                }
                std::string derr;
                planar = yp_read_ffmpeg(a.ffmpeg, path, tmp_wav, (int) SR, &T, &derr);
                if (!planar) {
                    fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: %s\n", i + 1, picked.size(), name.c_str(),
                            derr.c_str());
                    n_failed++;
                    continue;
                }
            } else {
                planar = yp_read_native_48k(path, &T);
                if (!planar || T <= 0) {
                    free(planar);
                    fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: decode failed\n", i + 1, picked.size(),
                            name.c_str());
                    n_failed++;
                    continue;
                }
            }

            // data.py clamps to [-1, 1] and does not normalise. Matched here:
            // loudness is part of what the NAR renders, and rescaling every
            // clip would teach it a level the base model does not use.
            const size_t n_samp = (size_t) T * 2;
            for (size_t k = 0; k < n_samp; k++) {
                if (planar[k] > 1.0f) {
                    planar[k] = 1.0f;
                } else if (planar[k] < -1.0f) {
                    planar[k] = -1.0f;
                } else if (!(planar[k] == planar[k])) {  // NaN
                    planar[k] = 0.0f;
                }
            }

            // ── encode ──
            std::vector<float> latents;
            int64_t            got = 0;
            std::string        eerr;
            const bool ok = yue2_vae_encode_tiled(m, planar, (int64_t) T, core_samples, halo_samples, &latents, &got,
                                                  /*boundaries=*/nullptr, &eerr);
            free(planar);
            if (!ok || got <= 0) {
                fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: VAE encode: %s\n", i + 1, picked.size(),
                        name.c_str(), eerr.empty() ? "no frames" : eerr.c_str());
                n_failed++;
                continue;
            }
            bool finite = true;
            for (size_t k = 0; k < latents.size(); k++) {
                const float v = latents[k];
                if (!(v == v) || v > 3.4e38f || v < -3.4e38f) {
                    finite = false;
                    break;
                }
            }
            if (!finite) {
                fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: VAE encode produced NaN/Inf\n", i + 1,
                        picked.size(), name.c_str());
                n_failed++;
                continue;
            }
            // Channel-major [LD, got] out of the encoder -> frame-major
            // [got, LD] on disk, the layout yue2_nt_read_clip reads and
            // x_t uses. See the header: this is the one transpose on the
            // path and it belongs here, not in the trainer's inner loop.
            std::vector<float> framed((size_t) (got * LD));
            for (int64_t c = 0; c < LD; c++) {
                const float * srcc = latents.data() + (size_t) (c * got);
                for (int64_t t = 0; t < got; t++) {
                    framed[(size_t) (t * LD + c)] = srcc[t];
                }
            }
            latents.swap(framed);
            if (!yp_write_cache(latent_path, latents)) {
                fprintf(stderr, "[yue2-preprocess] %zu/%zu FAIL %s: cannot write %s\n", i + 1, picked.size(),
                        name.c_str(), latent_path.c_str());
                n_failed++;
                continue;
            }
            frames = got;
            n_encoded++;
        }

        s.frames  = frames;
        s.seconds = (double) frames / fps;
        s.clips   = frames / clip_frames;  // the short tail is dropped, as upstream
        if (s.clips < 1) {
            fprintf(stderr,
                    "[yue2-preprocess] %zu/%zu SKIP %s: %.1f s is shorter than one %.1f s clip\n", i + 1,
                    picked.size(), name.c_str(), s.seconds, a.clip_seconds);
            n_skipped++;
            continue;
        }

        total_clips += s.clips;
        total_sec   += s.seconds;
        sources.push_back(s);
        fprintf(stderr, "[yue2-preprocess] %zu/%zu %s %-44s %7.1f s -> %lld frames, %lld clip(s)%s%s\n", i + 1,
                picked.size(), s.cache_hit ? "cached " : "encoded", name.c_str(), s.seconds, (long long) frames,
                (long long) s.clips, s.caption.empty() ? " [no caption]" : "",
                (a.caption_mode == "ace" && s.lyrics.empty()) ? " [no lyrics]" : "");
    }

    if (tmp_dir_made) {
        hs_remove(tmp_wav);
    }
    yue2_vae_enc_graph_free(&g_yue2_vae_enc);
    yue2_unload(&m);

    if (sources.empty() || total_clips == 0) {
        fprintf(stderr,
                "[yue2-preprocess] nothing usable: %zu encoded, %zu cached, %zu skipped, %zu failed. Every source "
                "is shorter than one clip, or none decoded — lower --clip-seconds or check --ffmpeg.\n",
                n_encoded, n_cached, n_skipped, n_failed);
        return 1;
    }

    // ── manifest ───────────────────────────────────────────────────────
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "format", "yue2-preprocess-v1");
    yyjson_mut_obj_add_strcpy(doc, root, "producer", (std::string("ace-train ") + ACE_VERSION).c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "created_at", pm_iso8601_utc_now().c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "audio_dir", a.audio_dir.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "vae_file", vae_path.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "vae_name", vae_base.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "vae_variant", variant_name);
    yyjson_mut_obj_add_int(doc, root, "sample_rate", SR);
    yyjson_mut_obj_add_int(doc, root, "downsampling_ratio", RATIO);
    // Both spellings: `frame_rate` is what yue2_nt_load_manifest looks for,
    // `frames_per_second` is what the rest of this tree calls it.
    yyjson_mut_obj_add_real(doc, root, "frame_rate", fps);
    yyjson_mut_obj_add_real(doc, root, "frames_per_second", fps);
    yyjson_mut_obj_add_int(doc, root, "latent_dim", LD);
    // Stated, not implied — see the header. index = t * latent_dim + c.
    yyjson_mut_obj_add_strcpy(doc, root, "latent_layout", "frame_major");
    yyjson_mut_obj_add_strcpy(doc, root, "latent_index", "t * latent_dim + c");
    // Read by yue2_nt_load_manifest's root_bits. Its DEFAULT is f16 (upstream
    // caches half), so omitting this would have every latent read as noise.
    yyjson_mut_obj_add_strcpy(doc, root, "latent_dtype", "f32");
    yyjson_mut_obj_add_strcpy(doc, root, "dtype", "f32");
    yyjson_mut_obj_add_strcpy(doc, root, "latent_kind", "posterior_mean");
    yyjson_mut_obj_add_real(doc, root, "clip_seconds", a.clip_seconds);
    yyjson_mut_obj_add_int(doc, root, "clip_frames", clip_frames);
    yyjson_mut_obj_add_strcpy(doc, root, "caption_mode", a.caption_mode.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "default_caption", a.default_caption.c_str());
    // The SHAPE of `caption`, stated rather than left to be sniffed. See the
    // header's "caption / lyrics" note and yp_caption_format().
    yyjson_mut_obj_add_strcpy(doc, root, "caption_format", yp_caption_format(a.caption_mode));
    yyjson_mut_obj_add_strcpy(doc, root, "caption_note", YP_CAPTION_NOTE);
    // RESERVED slot, 08-nar-lora-trainer.md §5. False = the text-only regime;
    // a later producer sets it true and adds a per-clip "codec_ids" path
    // WITHOUT bumping `format`.
    yyjson_mut_obj_add_bool(doc, root, "codec_ids_present", false);
    yyjson_mut_obj_add_strcpy(doc, root, "codec_ids_note",
                              "RESERVED: per-clip \"codec_ids\" is a manifest-relative path to little-endian "
                              "int32 semantic-token ids covering exactly that clip's frame range. Absent (and "
                              "codec_ids_present=false) means the text-only conditioning regime.");
    yyjson_mut_obj_add_int(doc, root, "n_sources", (int64_t) sources.size());
    yyjson_mut_obj_add_int(doc, root, "n_clips", total_clips);
    yyjson_mut_obj_add_real(doc, root, "total_audio_sec", total_sec);
    yyjson_mut_obj_add_int(doc, root, "n_encoded", (int64_t) n_encoded);
    yyjson_mut_obj_add_int(doc, root, "n_cache_hits", (int64_t) n_cached);
    yyjson_mut_obj_add_int(doc, root, "n_skipped", (int64_t) n_skipped);
    yyjson_mut_obj_add_int(doc, root, "n_failed", (int64_t) n_failed);

    yyjson_mut_val * sarr = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "sources", sarr);
    yyjson_mut_val * carr = yyjson_mut_arr(doc);
    yyjson_mut_obj_add_val(doc, root, "clips", carr);

    int64_t captioned = 0, lyriced = 0;
    for (const auto & s : sources) {
        yyjson_mut_val * so = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_strcpy(doc, so, "name", s.name.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "source", s.path.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "caption", s.caption.c_str());
        // Per SOURCE as well as per clip: the AR trainer reads sources[] (whole
        // songs), the NAR trainer reads clips[]. Both need the same two strings.
        yyjson_mut_obj_add_strcpy(doc, so, "lyrics", s.lyrics.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "genre", s.sm.genre.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "bpm", s.sm.bpm.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "key", s.sm.key.c_str());
        yyjson_mut_obj_add_strcpy(doc, so, "latents", s.latent_rel.c_str());
        yyjson_mut_obj_add_int(doc, so, "frames", s.frames);
        yyjson_mut_obj_add_int(doc, so, "clips", s.clips);
        yyjson_mut_obj_add_real(doc, so, "duration_sec", s.seconds);
        yyjson_mut_obj_add_bool(doc, so, "cache_hit", s.cache_hit);
        yyjson_mut_arr_append(sarr, so);

        for (int64_t k = 0; k < s.clips; k++) {
            char idbuf[192];
            snprintf(idbuf, sizeof(idbuf), "%s_c%04lld", s.stem.c_str(), (long long) k);
            yyjson_mut_val * co = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_strcpy(doc, co, "id", idbuf);
            yyjson_mut_obj_add_strcpy(doc, co, "source", s.path.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "caption", s.caption.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "lyrics", s.lyrics.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "genre", s.sm.genre.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "bpm", s.sm.bpm.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "key", s.sm.key.c_str());
            yyjson_mut_obj_add_strcpy(doc, co, "latents", s.latent_rel.c_str());
            // The reader needs source_frames as well as offset_frames: it
            // BOUNDS the clip's contiguous read against the file it came from
            // (the row stride is latent_dim, not this — see the header).
            yyjson_mut_obj_add_int(doc, co, "source_frames", s.frames);
            yyjson_mut_obj_add_int(doc, co, "offset_frames", k * clip_frames);
            yyjson_mut_obj_add_int(doc, co, "frames", clip_frames);
            // "codec_ids": RESERVED, see codec_ids_note. Not written by v1.
            yyjson_mut_arr_append(carr, co);
        }
        if (!s.caption.empty()) {
            captioned += s.clips;
        }
        if (!s.lyrics.empty()) {
            lyriced++;
        }
    }

    size_t mlen  = 0;
    char * mjson = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, &mlen);
    yyjson_mut_doc_free(doc);
    if (!mjson) {
        fprintf(stderr, "[yue2-preprocess] cannot serialize the manifest\n");
        return 1;
    }
    const bool wrote = pm_write_atomic(manifest_path, std::string(mjson, mlen));
    free(mjson);
    if (!wrote) {
        fprintf(stderr, "[yue2-preprocess] cannot write %s\n", manifest_path.c_str());
        return 1;
    }

    fprintf(stderr,
            "[yue2-preprocess] done: %zu sources (%zu encoded, %zu cached, %zu skipped, %zu failed), %lld clips "
            "x %.1f s, %lld with a caption, %lld/%zu source(s) with lyrics, %.2f min of audio -> %s\n",
            sources.size(), n_encoded, n_cached, n_skipped, n_failed, (long long) total_clips, a.clip_seconds,
            (long long) captioned, (long long) lyriced, sources.size(), total_sec / 60.0, manifest_path.c_str());
    // Partial success is still success: the failures are named above and the
    // manifest only lists what actually cached. A run where NOTHING worked
    // already returned 1.
    return 0;
}

// ── The consumer lives in the trainer, not here ───────────────────────
//
// There is deliberately NO reader in this file. `train/yue2-nar-train-run.h`
// already owns one — `yue2_nt_load_manifest` + `yue2_nt_read_clip` — and a
// second implementation of the same schema is the drift this whole header is
// written to avoid. This writer is matched to THAT reader, field by field:
//
//   root   format="yue2-preprocess-v1", latent_dim, clip_frames, frame_rate
//          (+ frames_per_second), latent_dtype="f32" (+ dtype: its default is
//          f16, so omitting this reads every latent as noise), latent_layout=
//          "frame_major", latent_index, codec_ids_present=false, plus the
//          provenance and counts nothing parses.
//   clips[] id, source, caption, lyrics, latents (relative to the manifest's
//          dir), offset_frames, frames, source_frames (a bounds check for the
//          reader; the row stride is latent_dim, not this).
//   sources[] one row per input file, carrying the same caption/lyrics pair.
//          Informational for the NAR trainer, which reads clips[] only.
//
// `sources[]` was listed last on purpose: a consumer that read it instead of
// `clips[]` would train on whole songs, and its `frames` is the SOURCE's, not
// a clip's. That is no longer a hazard but a second reader — the AR trainer
// (`train/yue2-ar-train-run.h`) reads sources[] DELIBERATELY, because whole
// songs are exactly what it trains on (contract §4.4: no crop policy) and the
// source-level `codec_ids` is the whole-song code array. The two readers want
// different arrays of the same manifest; both get the same two text fields.
