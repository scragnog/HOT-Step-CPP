#pragma once
// train/retarget-beats.h — beat grid + metric-phase pick for `ace-train mm3-retarget`.
//
// WHY: the retarget cut (see retarget-search.h) has to land on a downbeat, not just any beat, or the bar
// grid goes out of phase across the join even though the removal is a whole number of bars. Essentia's
// streaming extractor gives us beat TIMES for free (already shipped for Dataset Studio analysis — see
// server/src/services/training/essentiaClient.ts) but not which of those beats is beat 1. This file adds
// the phase pick on top, ported from tools/mm3-retarget/retarget.py analyse().
//
// TRAP THIS AVOIDS: onset strength alone is close to a coin flip on rock, where the backbeat (2 and 4) is
// often louder than the downbeat. The Python oracle scores three independent cues — attack loudness,
// low-band/kick weight, and harmonic change — and sums their z-scores. We only have two of those cues here
// (loudness, low-band ratio) because Essentia's JSON does not carry a full chroma-per-beat matrix the way
// librosa's `analyse()` does; chroma comes from wherever the caller already computed chroma-per-beat for
// the cost search (retarget-search.h), and rt_pick_phase() takes it as a parameter so this file does not
// need to know how it was made. The third cue (chroma-change) is computed here from that input.
//
// DIVERGENCE FROM THE ORACLE: retarget.py's onset-strength cue is a librosa per-FRAME onset envelope
// resampled to beat times. We do not have a frame-rate onset envelope here (Essentia's beat tracker does
// not expose one in this JSON), so we substitute `beats_loudness` — Essentia's own per-beat attack-strength
// estimate — as the "attack weight" cue. Both cues answer the same question (how hard does each beat hit)
// so the z-score-and-sum scheme is unaffected; this is a substitution of data source, not of method.
//
// hs_system()/hs_fopen() (hot-step-fsutf8.h) are used instead of system()/fopen() throughout: dataset audio
// paths are UTF-8 and the narrow CRT calls mis-decode anything outside 7-bit ASCII on Windows.

#include "hot-step-fsutf8.h"
#include "yyjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// One track's beat grid, as read back from Essentia's JSON.
struct RetargetBeats {
    std::vector<double> t;          // beat times, seconds, strictly increasing
    std::vector<double> loudness;   // per-beat attack strength; 0.0-filled where Essentia omitted it
    std::vector<double> low_ratio;  // per-beat band-0 (kick/bass) loudness ratio; 0.0-filled likewise
    double bpm          = 0.0;
    bool   has_loudness = false;  // false => rt_pick_phase must fall back to chroma alone
};

namespace retarget_beats_detail {

// Reads a numeric array member into `out`, leaving `out` empty if the key is absent, not an array, or
// empty. Never fails the caller — every field this file reads from Essentia is optional except beat times.
static inline void hs_read_num_array(yyjson_val * obj, const char * key, std::vector<double> * out) {
    out->clear();
    if (!obj) {
        return;
    }
    yyjson_val * arr = yyjson_obj_get(obj, key);
    if (!arr || !yyjson_is_arr(arr)) {
        return;
    }
    out->reserve(yyjson_arr_size(arr));
    yyjson_val *    v;
    yyjson_arr_iter it = yyjson_arr_iter_with(arr);
    while ((v = yyjson_arr_iter_next(&it))) {
        out->push_back(yyjson_is_num(v) ? yyjson_get_num(v) : 0.0);
    }
}

// beats_loudness_band_ratio is an array of arrays (one row per beat); band 0 is the lowest band, i.e. the
// kick/bass weight the phase picker wants. Rows shorter than one element, or the wrong type, read as 0.0
// rather than failing the whole track — a single malformed row must not sink an otherwise-usable grid.
static inline void hs_read_band0_array(yyjson_val * obj, const char * key, std::vector<double> * out) {
    out->clear();
    if (!obj) {
        return;
    }
    yyjson_val * arr = yyjson_obj_get(obj, key);
    if (!arr || !yyjson_is_arr(arr)) {
        return;
    }
    out->reserve(yyjson_arr_size(arr));
    yyjson_val *    row;
    yyjson_arr_iter it = yyjson_arr_iter_with(arr);
    while ((row = yyjson_arr_iter_next(&it))) {
        double       band0 = 0.0;
        yyjson_val * first = (row && yyjson_is_arr(row)) ? yyjson_arr_get_first(row) : nullptr;
        if (first && yyjson_is_num(first)) {
            band0 = yyjson_get_num(first);
        }
        out->push_back(band0);
    }
}

}  // namespace retarget_beats_detail

// Runs the Essentia streaming extractor on `audio_path`, parses its beat grid, and deletes the temp JSON
// it wrote. Returns false with a user-facing *err on any failure; RetargetBeats fields are only meaningful
// when this returns true. Caller owns `tmp_json_path` (a unique per-track scratch path) and picks where it
// lives; this function always removes it before returning, success or failure, so it never litters the
// disk even across a batch of hundreds of tracks.
static inline bool rt_beats_from_essentia(const std::string & essentia_bin, const std::string & audio_path,
                                           const std::string & tmp_json_path, RetargetBeats * out,
                                           std::string * err) {
    auto fail = [&](const std::string & msg) -> bool {
        if (err) {
            *err = msg;
        }
        hs_remove(tmp_json_path);  // best-effort; harmless if it was never written
        return false;
    };

    if (!hs_file_exists(essentia_bin)) {
        return fail("Essentia binary not found at \"" + essentia_bin +
                    "\" — install/point Dataset Studio at essentia_streaming_extractor_music before "
                    "retargeting tracks.");
    }
    if (!hs_file_exists(audio_path)) {
        return fail("audio file not found: \"" + audio_path + "\"");
    }

    // Same quoting idiom as cmd_mm3_codes in ace-train.cpp: each path gets its own quotes for the paths
    // themselves, and on _WIN32 the whole command line is wrapped in one more pair of quotes because
    // cmd.exe's /C strips exactly one outer layer before argv splitting happens.
    char cmd[4096];
    snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\" \"%s\"", essentia_bin.c_str(), audio_path.c_str(),
             tmp_json_path.c_str());
#ifdef _WIN32
    const std::string wrapped = "\"" + std::string(cmd) + "\"";
    const int         rc      = hs_system(wrapped);
#else
    const int rc = hs_system(cmd);
#endif
    // Essentia writes progress to stderr even on a clean run, and some builds return a nonzero exit code
    // on success anyway — same lesson essentiaClient.ts already learned. Success is judged by the output
    // file existing, not by rc; rc is only reported in the error text when that file never appears.
    if (!hs_file_exists(tmp_json_path)) {
        return fail("Essentia produced no output for \"" + audio_path + "\" (exit " + std::to_string(rc) +
                    "). Check the file decodes (try re-exporting as WAV) and that the Essentia binary runs "
                    "standalone from a terminal.");
    }

    FILE * jf = hs_fopen(tmp_json_path, "rb");
    if (!jf) {
        return fail("could not reopen Essentia's own output at \"" + tmp_json_path + "\"");
    }
    fseek(jf, 0, SEEK_END);
    const long jsz = ftell(jf);
    fseek(jf, 0, SEEK_SET);
    if (jsz <= 0) {
        fclose(jf);
        return fail("Essentia's output for \"" + audio_path + "\" is empty");
    }
    std::string jbuf((size_t) jsz, '\0');
    const bool  jread = fread(&jbuf[0], 1, (size_t) jsz, jf) == (size_t) jsz;
    fclose(jf);
    if (!jread) {
        return fail("short read on Essentia's output for \"" + audio_path + "\"");
    }

    yyjson_doc * doc = yyjson_read(jbuf.c_str(), jbuf.size(), 0);
    if (!doc) {
        return fail("Essentia's output for \"" + audio_path + "\" is not valid JSON");
    }
    yyjson_val * root   = yyjson_doc_get_root(doc);
    yyjson_val * rhythm = yyjson_obj_get(root, "rhythm");
    if (!rhythm) {
        yyjson_doc_free(doc);
        return fail("Essentia's output for \"" + audio_path +
                    "\" has no rhythm data — the extractor may have failed on this file's audio content.");
    }

    RetargetBeats rb;
    retarget_beats_detail::hs_read_num_array(rhythm, "beats_position", &rb.t);
    if (rb.t.size() < 16) {
        const size_t got = rb.t.size();
        yyjson_doc_free(doc);
        return fail("\"" + audio_path + "\" has too few detected beats (" + std::to_string(got) +
                    ", need >= 16) to search for a retarget cut — the beat tracker likely failed on this "
                    "track's tempo or mix.");
    }

    yyjson_val * bpm_v = yyjson_obj_get(rhythm, "bpm");
    rb.bpm             = (bpm_v && yyjson_is_num(bpm_v)) ? yyjson_get_num(bpm_v) : 0.0;

    std::vector<double> loud_raw, low_raw;
    retarget_beats_detail::hs_read_num_array(rhythm, "beats_loudness", &loud_raw);
    retarget_beats_detail::hs_read_band0_array(rhythm, "beats_loudness_band_ratio", &low_raw);
    yyjson_doc_free(doc);

    // Both arrays are allowed to be shorter than beats_position (or absent -> size 0). Zero-fill the tail
    // rather than truncating the grid: a beat with no loudness datum still has a valid TIME, and the phase
    // picker needs every beat's time to build its p::bpb strides correctly.
    rb.has_loudness = !loud_raw.empty() || !low_raw.empty();
    rb.loudness.assign(rb.t.size(), 0.0);
    rb.low_ratio.assign(rb.t.size(), 0.0);
    for (size_t i = 0; i < loud_raw.size() && i < rb.loudness.size(); ++i) {
        rb.loudness[i] = loud_raw[i];
    }
    for (size_t i = 0; i < low_raw.size() && i < rb.low_ratio.size(); ++i) {
        rb.low_ratio[i] = low_raw[i];
    }

    *out = std::move(rb);
    hs_remove(tmp_json_path);
    return true;
}

// z-scores `v` in place: (x - mean) / max(stddev, eps). Matches retarget.py's z() including its epsilon
// floor, so a near-constant cue (e.g. every beat the same loudness) scores as ~0 everywhere instead of
// blowing up, rather than being excluded.
static inline std::vector<double> rt_zscore(const std::vector<double> & v) {
    if (v.empty()) {
        return v;
    }
    double mean = 0.0;
    for (double x : v) {
        mean += x;
    }
    mean /= (double) v.size();
    double var = 0.0;
    for (double x : v) {
        var += (x - mean) * (x - mean);
    }
    var /= (double) v.size();
    const double sd = std::max(std::sqrt(var), 1e-9);
    std::vector<double> z(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        z[i] = (v[i] - mean) / sd;
    }
    return z;
}

// Picks the metric phase (which of the `bpb` beats in a bar is beat 1) by scoring each phase on the mean,
// over every beat at that phase, of up to three cues: beat loudness, low-band/kick ratio, and chroma
// change from the previous beat (chords turn at the bar line far more reliably than the loudest hit lands
// on it — see retarget.py's comment, same reasoning here). Each cue is z-scored ACROSS THE bpb PHASE
// SCORES, not across beats, before the three are summed — that is what makes attack-weight, kick-weight
// and harmony-change commensurable regardless of their native units. A cue whose backing data is absent
// (chroma_beats == nullptr, or beats.has_loudness == false) is skipped entirely rather than contributing
// zeros, which would otherwise silently drag every phase's score toward the missing cue's mean.
//
// A wrong phase does not break the meter: both cut ends shift together (retarget-search.h always advances
// by whole bars from the picked phase), so the removal stays a whole number of bars either way. It only
// risks landing the join one, two or three beats off the "true" bar line, which can still sound fine or can
// cost musical plausibility — never correctness. Treat a low confidence as a warning to check by ear, not
// as a reason to refuse the track.
static inline int rt_pick_phase(const float * chroma_beats, int n_beats, const RetargetBeats & beats,
                                 int bpb, double * confidence_out) {
    if (confidence_out) {
        *confidence_out = 0.0;
    }
    if (bpb <= 0) {
        return 0;
    }
    const size_t nb = (size_t) std::max(n_beats, 0);

    // Cue 1 & 2: per-beat loudness / low-band ratio, phase-averaged.
    std::vector<double> phase_loud(bpb, 0.0), phase_low(bpb, 0.0);
    if (beats.has_loudness) {
        std::vector<int> n_loud(bpb, 0), n_low(bpb, 0);
        const size_t     bn = beats.t.size();
        for (size_t i = 0; i < bn; ++i) {
            const int p = (int) (i % (size_t) bpb);
            phase_loud[(size_t) p] += beats.loudness[i];
            n_loud[(size_t) p]++;
            phase_low[(size_t) p] += beats.low_ratio[i];
            n_low[(size_t) p]++;
        }
        for (int p = 0; p < bpb; ++p) {
            if (n_loud[(size_t) p] > 0) {
                phase_loud[(size_t) p] /= (double) n_loud[(size_t) p];
            }
            if (n_low[(size_t) p] > 0) {
                phase_low[(size_t) p] /= (double) n_low[(size_t) p];
            }
        }
    }

    // Cue 3: chroma change |chroma[i] - chroma[i-1]| (L2 over the 12 pitch classes), 0 at i = 0, then
    // phase-averaged the same way. chroma_beats is column-major, 12 rows x n_beats columns.
    std::vector<double> phase_chg(bpb, 0.0);
    const bool          have_chroma = (chroma_beats != nullptr && nb > 0);
    if (have_chroma) {
        std::vector<int> n_chg(bpb, 0);
        for (size_t i = 0; i < nb; ++i) {
            double chg = 0.0;
            if (i > 0) {
                double sumsq = 0.0;
                for (int c = 0; c < 12; ++c) {
                    const double a = (double) chroma_beats[i * 12 + (size_t) c];
                    const double b = (double) chroma_beats[(i - 1) * 12 + (size_t) c];
                    const double d = a - b;
                    sumsq += d * d;
                }
                chg = std::sqrt(sumsq);
            }
            const int p = (int) (i % (size_t) bpb);
            phase_chg[(size_t) p] += chg;
            n_chg[(size_t) p]++;
        }
        for (int p = 0; p < bpb; ++p) {
            if (n_chg[(size_t) p] > 0) {
                phase_chg[(size_t) p] /= (double) n_chg[(size_t) p];
            }
        }
    }

    std::vector<double> total(bpb, 0.0);
    if (beats.has_loudness) {
        const std::vector<double> zl = rt_zscore(phase_loud);
        const std::vector<double> zk = rt_zscore(phase_low);
        for (int p = 0; p < bpb; ++p) {
            total[(size_t) p] += zl[(size_t) p] + zk[(size_t) p];
        }
    }
    if (have_chroma) {
        const std::vector<double> zc = rt_zscore(phase_chg);
        for (int p = 0; p < bpb; ++p) {
            total[(size_t) p] += zc[(size_t) p];
        }
    }

    int best = 0;
    for (int p = 1; p < bpb; ++p) {
        if (total[(size_t) p] > total[(size_t) best]) {
            best = p;
        }
    }
    if (confidence_out && bpb > 1) {
        std::vector<double> sorted_desc = total;
        std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<double>());
        const double top    = sorted_desc[0];
        const double second = sorted_desc[1];
        // Guard the DIVISION, not the answer. Substituting 1.0 when the denominator collapses reports full
        // confidence in an arbitrary phase-0 pick — which is what happens when Essentia gives neither
        // beats_loudness nor beats_loudness_band_ratio and the chroma cue is flat, i.e. exactly the case the
        // caller needs warning about. Degenerate input must read as 0.
        *confidence_out = (top - second) / std::max(std::fabs(top) + std::fabs(second), 1e-9);
    } else if (confidence_out) {
        *confidence_out = 1.0;
    }
    return best;
}
