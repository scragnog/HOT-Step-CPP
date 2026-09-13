#pragma once
// mm3-retarget-run.h — `ace-train mm3-retarget`: rescue dataset tracks that are longer than MM3 can hold.
//
// MM3's checkpoint caps audio at 9000 frames = 6:00 (mm3-model.h, max_audio_frames), and MM3 LM training drops every
// longer track (--drop-over-frames, mm3-lm-train-run.h). Measured on this library that is 269 tracks, and they are
// the long ones an artist adapter would most like to hear finish.
//
// Rather than cropping, this removes ONE interior span so the intro and the real ending both survive. The cut is a
// jump between two downbeats the song treats as interchangeable — see retarget-search.h for the criterion, which is
// the whole idea. Output is an edited WAV per track plus a DERIVED dataset.json pointing at it; `mm3-codes` resolves
// audio through each sample's audio_path and `mm3-lm-train` reads lyrics from the same manifest, so both stages pick
// the edits up with no further change. `filename` is deliberately left alone: the trainer resolves the MM3 caption as
// <captions>/<stem>.mm3.txt off that field, and the caption still describes the song.
//
// ── The vocal rule, and why this stage is conservative ──────────────────────────────────────────────────────────
//
// A dataset lyric sheet is untimed plain text, and the trainer builds its prompt from caption + those lyrics. Excise
// sung material without editing the sheet and the model learns a lyric/audio mismatch — worse than losing the track.
// Editing the sheet needs a forced alignment, which is a separate piece of work (the MM3 LM's own decode attention
// over the lyric columns, the mm3-align.h mechanism, run teacher-forced over the track's codes).
//
// So this stage REFUSES any cut whose removed span contains singing, and leaves the lyrics untouched. A refusal
// leaves the track exactly where it is today — still dropped by --drop-over-frames — so this can only ever help.
//
// ── Where the vocal activity comes from ─────────────────────────────────────────────────────────────────────────
//
// NOT from SuperSep in this process. ace-train is deliberately standalone — header-only, no acestep-core link, no
// ONNX Runtime (see the comment on its target in engine/CMakeLists.txt) — and supersep.h is explicit that its VRAM
// policy is sequential with the GGML model store, which a second process cannot honour while ace-server is alive.
//
// So the server separates through the engine's existing /supersep/separate endpoint, the way tools/vocal-end already
// does, and drops one sidecar per track next to the run:
//
//     <vocal-dir>/<id>.json   { "hop": 0.1, "runs": [[12.4, 31.8], [44.0, 61.2], ...] }
//
// `runs` are vocal-ACTIVE intervals in seconds. A missing sidecar is a hard error, never a silent "assume
// instrumental" — that assumption would cut through singing and quietly poison the lyrics.
//
// ── Sample rate ─────────────────────────────────────────────────────────────────────────────────────────────────
//
// Everything runs at 44.1 kHz stereo: it is what the sidecar's timings were measured against, and one decode serves
// both the analysis and the splice. The sources are often 96 kHz FLAC, so this resamples them — which costs nothing
// here, because the edited WAV's only consumer is mm3-codes, and that resamples to the DAV encoder's rate anyway.

#include "retarget-beats.h"
#include "retarget-dsp.h"
#include "retarget-search.h"

#include "../audio-io.h"
#include "../hot-step-fsutf8.h"
#include "preprocess-io.h"

#include "yyjson.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct MM3RetargetArgs {
    std::string dataset;                 // input dataset.json
    std::string out_dir;                 // edits + derived dataset.json land here
    std::string ffmpeg   = "ffmpeg";
    std::string essentia;                // essentia_streaming_extractor_music
    std::string vocal_dir;               // <id>.json vocal-activity sidecars, written by the server
    double      target       = 360.0;    // the 9000-frame cap in seconds
    double      margin       = 4.0;      // remove this much beyond the overrun
    double      protect_head = 8.0;      // an early cut still leaves the song opening the way it opens
    double      protect_tail = 45.0;     // a late cut deletes the approach to the ending, which is the point
    double      max_cost     = 0.45;     // refuse above this lead-in mismatch
    double      guard        = 1.0;      // seconds either side of each end that must be vocal-free
    double      xfade_ms     = 30.0;
    double      snippet      = 10.0;     // seconds either side of the seam in the audition clip
    int         bpb          = 4;
    int         phrase       = 4;        // removal must be a whole number of THIS many bars
    int         lookback     = 8;
    int         alts         = 2;        // extra runner-up seams rendered for audition
    bool        dry          = false;
};

// One track's outcome, for the console table and the manifest patch.
struct MM3RetargetOne {
    std::string id, filename, audio_path, edited_path;
    double      duration = 0, edited = 0, need = 0;
    double      t_a = 0, t_b = 0, removal = 0, context_cost = 0, seam = 0;
    int         bars = 0, n_candidates = 0;
    bool        ok = false;
    std::string why;                     // why not, when !ok
};

// ── small helpers ───────────────────────────────────────────────────────────────────────────────────────────────

static std::string rt_json_str(yyjson_val * o, const char * k) {
    yyjson_val * v = yyjson_obj_get(o, k);
    return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
}

static double rt_json_num(yyjson_val * o, const char * k) {
    yyjson_val * v = yyjson_obj_get(o, k);
    return (v && yyjson_is_num(v)) ? yyjson_get_num(v) : 0.0;
}

static bool rt_read_file(const std::string & path, std::string * out) {
    FILE * f = hs_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return false;
    }
    out->resize((size_t) sz);
    const bool got = fread(&(*out)[0], 1, (size_t) sz, f) == (size_t) sz;
    fclose(f);
    return got;
}

// Expand the server's vocal-activity sidecar into a per-hop mask.
static bool rt_load_vocal(const std::string & path, double total_s, double * hop_out,
                          std::vector<uint8_t> * mask, std::string * err) {
    std::string buf;
    if (!rt_read_file(path, &buf)) {
        *err = "no vocal-activity sidecar at " + path
             + " (the server writes one per track from SuperSep before this runs)";
        return false;
    }
    yyjson_doc * d = yyjson_read(buf.c_str(), buf.size(), 0);
    yyjson_val * r = d ? yyjson_doc_get_root(d) : nullptr;
    yyjson_val * hv = r ? yyjson_obj_get(r, "hop") : nullptr;
    yyjson_val * rv = r ? yyjson_obj_get(r, "runs") : nullptr;
    if (!rv || !yyjson_is_arr(rv)) {
        if (d) {
            yyjson_doc_free(d);
        }
        *err = path + " has no `runs` array";
        return false;
    }
    const double hop = (hv && yyjson_is_num(hv) && yyjson_get_num(hv) > 0) ? yyjson_get_num(hv) : 0.1;
    const size_t n   = (size_t) (total_s / hop) + 2;
    mask->assign(n, 0);
    yyjson_val *    run;
    yyjson_arr_iter it = yyjson_arr_iter_with(rv);
    while ((run = yyjson_arr_iter_next(&it))) {
        if (!yyjson_is_arr(run) || yyjson_arr_size(run) < 2) {
            continue;
        }
        const double t0 = yyjson_get_num(yyjson_arr_get(run, 0));
        const double t1 = yyjson_get_num(yyjson_arr_get(run, 1));
        if (!(t1 > t0)) {
            continue;
        }
        size_t i0 = (size_t) (t0 / hop);
        size_t i1 = (size_t) (t1 / hop) + 1;
        if (i1 > n) {
            i1 = n;
        }
        for (size_t i = i0; i < i1; i++) {
            (*mask)[i] = 1;
        }
    }
    yyjson_doc_free(d);
    *hop_out = hop;
    return true;
}

// Decode anything ffmpeg reads to 44.1 kHz stereo float WAV, then load it PLANAR.
// The planar/interleaved distinction is not cosmetic here: audio_io_read_wav_buf returns channel-major
// (L at [0,T), R at [T,2T)), while SuperSep and the splice both want interleaved frames.
static bool rt_decode_44k(const std::string & ffmpeg, const std::string & audio, const std::string & tmp_wav,
                          std::vector<float> * planar, int64_t * n_frames, std::string * err) {
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "\"%s\" -y -v error -i \"%s\" -ac 2 -ar 44100 -c:a pcm_f32le -f wav \"%s\"",
             ffmpeg.c_str(), audio.c_str(), tmp_wav.c_str());
#ifdef _WIN32
    // cmd.exe strips the outer pair of quotes off a command that starts with one, so the whole thing needs
    // wrapping again. Same idiom as cmd_mm3_codes.
    const std::string wrapped = "\"" + std::string(cmd) + "\"";
    const int         rcode   = hs_system(wrapped);
#else
    const int rcode = hs_system(cmd);
#endif
    if (rcode != 0 || !pm_file_exists(tmp_wav)) {
        *err = "ffmpeg exit " + std::to_string(rcode);
        return false;
    }
    std::string buf;
    if (!rt_read_file(tmp_wav, &buf)) {
        *err = "cannot reopen the decoded wav";
        return false;
    }
    int     T = 0, sr = 0;
    float * raw = audio_io_read_wav_buf((const uint8_t *) buf.data(), buf.size(), &T, &sr);
    if (!raw || T <= 0 || sr != 44100) {
        free(raw);
        *err = "decoded wav is not 44100 Hz stereo";
        return false;
    }
    planar->assign(raw, raw + (size_t) T * 2);
    free(raw);
    *n_frames = T;
    return true;
}

static void rt_planar_to_interleaved(const std::vector<float> & planar, int64_t n, std::vector<float> * out) {
    out->resize((size_t) n * 2);
    const float * L = planar.data();
    const float * R = planar.data() + n;
    for (int64_t i = 0; i < n; i++) {
        (*out)[(size_t) (i * 2)]     = L[i];
        (*out)[(size_t) (i * 2 + 1)] = R[i];
    }
}

static void rt_interleaved_to_planar(const float * inter, int64_t n, std::vector<float> * out) {
    out->resize((size_t) n * 2);
    for (int64_t i = 0; i < n; i++) {
        (*out)[(size_t) i]       = inter[i * 2];
        (*out)[(size_t) (n + i)] = inter[i * 2 + 1];
    }
}

static bool rt_write_wav24(const std::string & path, const float * interleaved, int64_t n_frames, int sr) {
    std::vector<float> planar;
    rt_interleaved_to_planar(interleaved, n_frames, &planar);
    // 24-bit, not 16: this file is an intermediate that mm3-codes re-reads, and there is no reason to spend a
    // quantisation on it.
    const std::string wav = audio_encode_wav_s24(planar.data(), (int) n_frames, sr);
    FILE *            f   = hs_fopen(path, "wb");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(wav.data(), 1, wav.size(), f) == wav.size();
    fclose(f);
    return ok;
}

// Mono sum for analysis. Chroma and MFCC both want one channel, and a mid-sum keeps anything panned hard.
static void rt_mono(const std::vector<float> & planar, int64_t n, std::vector<float> * out) {
    out->resize((size_t) n);
    const float * L = planar.data();
    const float * R = planar.data() + n;
    for (int64_t i = 0; i < n; i++) {
        (*out)[(size_t) i] = 0.5f * (L[i] + R[i]);
    }
}

// ── one track ───────────────────────────────────────────────────────────────────────────────────────────────────

static bool mm3_retarget_track(const MM3RetargetArgs & a, MM3RetargetOne * r) {
    const std::string tmp_wav  = a.out_dir + "/_retarget_tmp.wav";
    const std::string tmp_json = a.out_dir + "/_retarget_essentia.json";

    std::vector<float> planar;
    int64_t            n_frames = 0;
    std::string        err;
    if (!rt_decode_44k(a.ffmpeg, r->audio_path, tmp_wav, &planar, &n_frames, &err)) {
        r->why = err;
        return false;
    }
    const int    SR  = 44100;
    const double dur = (double) n_frames / SR;
    r->duration      = dur;
    r->need          = dur - a.target + a.margin;
    if (r->need <= 0) {
        r->why = "already under the cap";
        return false;
    }

    // 1. beats, from the Essentia binary the release already ships
    RetargetBeats beats;
    if (!rt_beats_from_essentia(a.essentia, r->audio_path, tmp_json, &beats, &err)) {
        r->why = "beats: " + err;
        return false;
    }
    const int nb = (int) beats.t.size();

    // 2. per-beat harmony and timbre
    std::vector<float> mono;
    rt_mono(planar, n_frames, &mono);
    const int          N_FFT = 4096, HOP = 1024;   // 23.2 ms hop at 44.1 kHz — the reference's 512/22050
    std::vector<float> mag;
    int                n_sp = 0, n_bins = 0;
    if (!rt_stft_mag(mono.data(), (int64_t) mono.size(), N_FFT, HOP, &mag, &n_sp, &n_bins)) {
        r->why = "stft failed";
        return false;
    }
    std::vector<float> chroma, mfcc, chroma_b, mfcc_b, feat, dist;
    rt_chroma(mag.data(), n_sp, n_bins, SR, N_FFT, &chroma);
    rt_mfcc(mag.data(), n_sp, n_bins, SR, N_FFT, 40, 20, &mfcc);
    const int Dm = 19;   // rt_mfcc drops c0
    const double frame_hop_s = (double) HOP / SR;
    rt_beat_sync(chroma.data(), 12, n_sp, frame_hop_s, beats.t.data(), nb, true, &chroma_b);
    rt_beat_sync(mfcc.data(), Dm, n_sp, frame_hop_s, beats.t.data(), nb, false, &mfcc_b);
    int D = 0;
    rt_feature_stack(chroma_b.data(), 12, mfcc_b.data(), Dm, nb, &feat, &D);
    rt_cosine_dist(feat.data(), D, nb, &dist);

    double    conf  = 0.0;
    const int phase = rt_pick_phase(chroma_b.data(), nb, beats, a.bpb, &conf);
    fprintf(stderr, "[mm3-retarget]   %.1f bpm, %d beats, phase %d/%d (confidence %.2f)\n", beats.bpm, nb, phase,
            a.bpb, conf);

    // 3. where the singing is, from the server's sidecar. This stage refuses to cut through any of it, so a missing
    //    sidecar fails the track rather than defaulting to "assume instrumental".
    std::vector<float> inter;
    rt_planar_to_interleaved(planar, n_frames, &inter);
    double               VHOP = 0.1;
    std::vector<uint8_t> vocal;
    if (!rt_load_vocal(a.vocal_dir + "/" + r->id + ".json", dur, &VHOP, &vocal, &err)) {
        r->why = err;
        return false;
    }

    // 4. the search. The guard keeps each END out of a vocal phrase; this extra pass is the stronger rule that
    //    nothing SUNG may be inside the removed span at all, which is what lets the lyric sheet stay untouched.
    RetargetSearchCfg cfg;
    cfg.bpb          = a.bpb;
    cfg.phrase       = a.phrase;
    cfg.lookback     = a.lookback;
    cfg.need         = r->need;
    cfg.protect_head = a.protect_head;
    cfg.protect_tail = a.protect_tail;
    cfg.guard        = a.guard;
    cfg.n_keep       = 16;
    std::vector<RetargetCut> cuts;
    r->n_candidates = rt_search(dist.data(), nb, beats.t.data(), phase, vocal.data(), (int) vocal.size(), VHOP, cfg,
                                &cuts);

    std::vector<RetargetCut> silent;
    for (const RetargetCut & c : cuts) {
        const size_t i0 = (size_t) (c.t_a / VHOP);
        const size_t i1 = (size_t) (c.t_b / VHOP) < vocal.size() ? (size_t) (c.t_b / VHOP) : vocal.size();
        bool         sung = false;
        for (size_t i = i0; i < i1; i++) {
            if (vocal[i]) {
                sung = true;
                break;
            }
        }
        if (!sung) {
            silent.push_back(c);
        }
    }
    if (silent.empty()) {
        r->why = cuts.empty() ? "no downbeat pair clears the length and vocal-gap constraints"
                              : "every candidate removal contains singing (needs the lyric-alignment stage)";
        return false;
    }
    const RetargetCut & best = silent[0];
    r->t_a = best.t_a;
    r->t_b = best.t_b;
    r->removal      = best.removal;
    r->context_cost = best.context_cost;
    r->bars         = best.bars;
    if (best.context_cost > a.max_cost) {
        char buf[160];
        snprintf(buf, sizeof(buf), "best lead-in cost %.3f exceeds --max-cost %.2f", best.context_cost, a.max_cost);
        r->why = buf;
        return false;
    }

    // 5. render
    std::vector<float> out;
    int64_t            out_n = 0;
    if (!rt_splice(inter.data(), n_frames, 2, SR, best.t_a, best.t_b, a.xfade_ms, &out, &out_n, &r->seam)) {
        r->why = "splice failed";
        return false;
    }
    r->edited      = (double) out_n / SR;
    r->edited_path = a.out_dir + "/" + r->id + ".edit.wav";
    if (!rt_write_wav24(r->edited_path, out.data(), out_n, SR)) {
        r->why = "cannot write " + r->edited_path;
        return false;
    }
    // Audition clips. Nothing about this is validated until someone has heard the joins, so make them easy to find.
    const int64_t s0 = (int64_t) ((r->seam - a.snippet) * SR) > 0 ? (int64_t) ((r->seam - a.snippet) * SR) : 0;
    const int64_t s1 = (int64_t) ((r->seam + a.snippet) * SR) < out_n ? (int64_t) ((r->seam + a.snippet) * SR) : out_n;
    if (s1 > s0) {
        rt_write_wav24(a.out_dir + "/" + r->id + ".seam.wav", out.data() + s0 * 2, s1 - s0, SR);
    }
    for (int k = 1; k <= a.alts && k < (int) silent.size(); k++) {
        std::vector<float> ay;
        int64_t            an = 0;
        double             at = 0;
        if (!rt_splice(inter.data(), n_frames, 2, SR, silent[(size_t) k].t_a, silent[(size_t) k].t_b, a.xfade_ms, &ay,
                       &an, &at)) {
            continue;
        }
        const int64_t a0 = (int64_t) ((at - a.snippet) * SR) > 0 ? (int64_t) ((at - a.snippet) * SR) : 0;
        const int64_t a1 = (int64_t) ((at + a.snippet) * SR) < an ? (int64_t) ((at + a.snippet) * SR) : an;
        if (a1 > a0) {
            rt_write_wav24(a.out_dir + "/" + r->id + ".alt" + std::to_string(k) + ".seam.wav", ay.data() + a0 * 2,
                           a1 - a0, SR);
        }
    }
    r->ok = true;
    return true;
}

// ── driver ──────────────────────────────────────────────────────────────────────────────────────────────────────

static int mm3_retarget_run(const MM3RetargetArgs & a) {
    if (a.dataset.empty() || a.out_dir.empty()) {
        fprintf(stderr, "ace-train mm3-retarget: --dataset and --out are required\n");
        return 1;
    }
    if (!a.dry && a.essentia.empty()) {
        fprintf(stderr, "ace-train mm3-retarget: --essentia <essentia_streaming_extractor_music> is required\n");
        return 1;
    }
    if (!a.dry && a.vocal_dir.empty()) {
        fprintf(stderr, "ace-train mm3-retarget: --vocal-dir is required. This stage will not cut a track without "
                        "knowing where the singing is; the server writes <id>.json sidecars from SuperSep first.\n");
        return 1;
    }
    std::string buf;
    if (!rt_read_file(a.dataset, &buf)) {
        fprintf(stderr, "cannot read %s\n", a.dataset.c_str());
        return 1;
    }
    yyjson_doc * doc  = yyjson_read(buf.c_str(), buf.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    yyjson_val * arr  = root ? yyjson_obj_get(root, "samples") : nullptr;
    if (!arr || !yyjson_is_arr(arr)) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        fprintf(stderr, "%s has no `samples` array\n", a.dataset.c_str());
        return 1;
    }

    std::vector<MM3RetargetOne> over;
    yyjson_val *                s;
    yyjson_arr_iter             it = yyjson_arr_iter_with(arr);
    size_t                      n_total = 0;
    while ((s = yyjson_arr_iter_next(&it))) {
        n_total++;
        const double d = rt_json_num(s, "duration");
        if (d <= a.target) {
            continue;
        }
        MM3RetargetOne r;
        r.id         = rt_json_str(s, "id");
        r.filename   = rt_json_str(s, "filename");
        r.audio_path = rt_json_str(s, "audio_path");
        r.duration   = d;
        over.push_back(r);
    }
    fprintf(stderr, "[mm3-retarget] %zu samples, %zu over %.0fs\n", n_total, over.size(), a.target);
    for (const MM3RetargetOne & r : over) {
        fprintf(stderr, "[mm3-retarget]   %5.2f min  %s\n", r.duration / 60.0, r.filename.c_str());
    }
    if (a.dry || over.empty()) {
        yyjson_doc_free(doc);
        return 0;
    }
    if (!pm_mkdir_p(a.out_dir)) {
        yyjson_doc_free(doc);
        fprintf(stderr, "cannot create %s\n", a.out_dir.c_str());
        return 1;
    }

    int edited = 0;
    for (MM3RetargetOne & r : over) {
        fprintf(stderr, "[mm3-retarget] %s (%.2f min)\n", r.filename.c_str(), r.duration / 60.0);
        if (mm3_retarget_track(a, &r)) {
            edited++;
            fprintf(stderr,
                    "[mm3-retarget]   CUT %.1fs (%d bars) at %.1f -> %.1f, cost %.3f, seam at %.1fs; %.2f min out\n",
                    r.removal, r.bars, r.t_a, r.t_b, r.context_cost, r.seam, r.edited / 60.0);
        } else {
            fprintf(stderr, "[mm3-retarget]   KEPT AS-IS: %s\n", r.why.c_str());
        }
    }

    // Derived manifest: a copy of the original with audio_path and duration repointed on the tracks that were
    // edited. Lyrics are untouched by construction — nothing sung was removed.
    yyjson_mut_doc * mdoc = yyjson_doc_mut_copy(doc, nullptr);
    yyjson_doc_free(doc);
    if (!mdoc) {
        fprintf(stderr, "cannot copy the manifest\n");
        return 1;
    }
    yyjson_mut_val * mroot = yyjson_mut_doc_get_root(mdoc);
    yyjson_mut_val * marr  = mroot ? yyjson_mut_obj_get(mroot, "samples") : nullptr;
    if (marr) {
        yyjson_mut_val *    ms;
        yyjson_mut_arr_iter mit = yyjson_mut_arr_iter_with(marr);
        while ((ms = yyjson_mut_arr_iter_next(&mit))) {
            yyjson_mut_val *  idv = yyjson_mut_obj_get(ms, "id");
            const std::string id  = (idv && yyjson_mut_is_str(idv)) ? yyjson_mut_get_str(idv) : "";
            for (const MM3RetargetOne & r : over) {
                if (!r.ok || r.id != id) {
                    continue;
                }
                yyjson_mut_obj_remove_key(ms, "audio_path");
                // _strcpy, not _str: add_str stores the pointer, and r.edited_path is a std::string that does not
                // outlive this loop.
                yyjson_mut_obj_add_strcpy(mdoc, ms, "audio_path", r.edited_path.c_str());
                yyjson_mut_obj_remove_key(ms, "duration");
                yyjson_mut_obj_add_real(mdoc, ms, "duration", r.edited);
            }
        }
    }
    const std::string dst = a.out_dir + "/dataset.json";
    const bool wrote = yyjson_mut_write_file(dst.c_str(), mdoc, YYJSON_WRITE_PRETTY, nullptr, nullptr);
    yyjson_mut_doc_free(mdoc);
    if (!wrote) {
        fprintf(stderr, "cannot write %s\n", dst.c_str());
        return 1;
    }
    fprintf(stderr, "[mm3-retarget] %d of %zu edited; the rest stay as they are and are still dropped by "
                    "--drop-over-frames\n",
            edited, over.size());
    fprintf(stderr, "[mm3-retarget] derived manifest: %s\n", dst.c_str());
    return 0;
}
