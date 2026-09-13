#pragma once
// retarget-search.h — candidate cut search + audio splice for `ace-train mm3-retarget`.
//
// Reference oracle: tools/mm3-retarget/retarget.py (find_cuts / splice / vocal_mask). This header ports the
// three numeric cores of that script; the caller (elsewhere in the mm3-retarget subcommand) owns beat
// tracking, chroma/MFCC feature extraction and the distance matrix — none of that is std-only, so it stays
// in Python/librosa territory and this side only consumes the resulting `dist` matrix and beat grid.
//
// THE ONE THING THAT MAKES THIS WORK: context_cost matches the LEAD-IN to each cut end (the `lookback`
// beats immediately before beat_t[a] and before beat_t[b]), not the two beats sitting at the seam. Scoring
// dist[a][b] directly is the obvious first attempt and produces cuts that are harmonically fine at the
// exact join sample but arrive from a completely different musical run-up, which reads as a splice even
// when the note under the crossfade matches. See rt_search()'s inner loop for where this is implemented.
//
// THE OTHER TRAP: rt_splice()'s crossfade half-width `h` is a FRAME count (one value per sample position
// across all channels), not a raw sample count. Using half the interleaved-buffer sample count instead of
// half the frame count silently halves the audible crossfade on stereo (or divides it by N on N-channel
// audio) while every index still looks "in bounds" — nothing crashes, the seam is just twice as abrupt as
// requested. Kept in frames throughout, matching the reference's frame-indexed (soundfile always_2d) arrays.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// MSVC does not define M_PI unless _USE_MATH_DEFINES is set before <cmath>, and this header may be pulled
// in after some other TU already included <cmath> without it. Carry our own constant (see moss-mel.h).
// A namespace-scope constexpr has internal linkage by default, so this is safe even if some future TU
// includes both files.
constexpr double kRetargetPi = 3.14159265358979323846;

// One candidate cut: jump from downbeat `a` to downbeat `b`, removing [t_a, t_b).
struct RetargetCut {
    int    a = 0, b = 0;          // beat indices
    double t_a = 0, t_b = 0;      // seconds
    double removal = 0;           // t_b - t_a
    double context_cost = 0;      // the lead-in match, the number that gets compared to max_cost
    double cost = 0;              // context_cost + the overshoot penalty, the ranking key
    int    bars = 0;              // (b - a) / bpb
};

struct RetargetSearchCfg {
    int    bpb = 4, phrase = 4, lookback = 8;
    double need = 0;              // seconds that MUST come out
    double protect_head = 8.0, protect_tail = 45.0, guard = 1.0;
    int    n_keep = 8;
};

namespace retarget_detail {

// Rounds to 4 decimal places, matching the reference's round(x, 4) on context_cost/cost. Ranking sorts on
// the ROUNDED cost (as the Python does, since it sorts the dict it already rounded into), so this has to
// happen before the sort below, not just at the end for display.
inline double rt_round4(double x) {
    return std::round(x * 10000.0) / 10000.0;
}

// "vocal inactive over [t0, t1]" at `hop_s` resolution. A null mask (no SuperSep stem available) always
// reads as inactive, matching the reference's `act is None -> True` no-constraint path.
inline bool rt_vocal_quiet(const uint8_t* vocal_active, int n_vocal, double hop_s, double t0, double t1) {
    if (vocal_active == nullptr || n_vocal <= 0 || hop_s <= 0.0) {
        return true;
    }
    int64_t i0 = static_cast<int64_t>(std::floor(t0 / hop_s));
    if (i0 < 0) {
        i0 = 0;
    }
    int64_t i1 = static_cast<int64_t>(std::ceil(t1 / hop_s)) + 1;
    const int64_t n64 = static_cast<int64_t>(n_vocal);
    if (i1 > n64) {
        i1 = n64;
    }
    if (i1 <= i0) {
        return true;
    }
    for (int64_t i = i0; i < i1; ++i) {
        if (vocal_active[static_cast<size_t>(i)] != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace retarget_detail

// Ranked downbeat-pair candidates for one track. `dist` is the n_beats x n_beats cosine-distance matrix,
// row-major (dist[i * n_beats + j]). `vocal_active` may be null (no vocal constraint at all, matching
// --no-vocal in the reference); when non-null it is a boolean-ish mask at `vocal_hop_s` resolution.
// Returns the TOTAL number of pairs that passed every constraint before sparsification (the reference's
// n_all, which it prints as "N candidate pairs"); the sparse top cfg.n_keep list lands in *out.
inline int rt_search(const float* dist, int n_beats, const double* beat_t, int phase,
                      const uint8_t* vocal_active, int n_vocal, double vocal_hop_s,
                      const RetargetSearchCfg& cfg, std::vector<RetargetCut>* out) {
    if (out == nullptr) {
        return 0;
    }
    out->clear();
    if (dist == nullptr || beat_t == nullptr || n_beats <= 0 || cfg.bpb <= 0) {
        return 0;
    }

    const int bpb = cfg.bpb;
    const int lookback = std::max(0, cfg.lookback);
    const double end_t = beat_t[n_beats - 1];

    // Candidate downbeats: same grid serves as both the `a` pool and the `b` pool, exactly as the
    // reference builds one `dbs` list and nests it against itself.
    std::vector<int> dbs;
    for (int i = phase; i < n_beats; i += bpb) {
        if (i - lookback < 0) {
            continue;
        }
        if (beat_t[i] < cfg.protect_head || beat_t[i] > end_t - cfg.protect_tail) {
            continue;
        }
        dbs.push_back(i);
    }

    std::vector<double> w;
    if (lookback > 0) {
        w.resize(static_cast<size_t>(lookback));
        double wsum = 0.0;
        for (int k = 0; k < lookback; ++k) {
            const double wk = std::pow(0.9, static_cast<double>(k));
            w[static_cast<size_t>(k)] = wk;
            wsum += wk;
        }
        if (wsum > 0.0) {
            for (int k = 0; k < lookback; ++k) {
                w[static_cast<size_t>(k)] /= wsum;
            }
        }
    }

    std::vector<RetargetCut> all;
    const int64_t n_beats64 = static_cast<int64_t>(n_beats);

    for (size_t ia = 0; ia < dbs.size(); ++ia) {
        const int a = dbs[ia];
        const double t_a = beat_t[a];
        if (!retarget_detail::rt_vocal_quiet(vocal_active, n_vocal, vocal_hop_s, t_a - cfg.guard, t_a)) {
            continue;  // whole `a` is disqualified, mirroring the reference's outer-loop `continue`
        }
        // dbs is built in strictly ascending beat-index order, so b <= a is exactly ib <= ia — start
        // past it instead of re-testing b <= a for every (a, b) pair as the reference does line-for-line.
        for (size_t ib = ia + 1; ib < dbs.size(); ++ib) {
            const int b = dbs[ib];
            if (cfg.phrase > 1 && (((b - a) / bpb) % cfg.phrase) != 0) {
                continue;
            }
            const double t_b = beat_t[b];
            const double removal = t_b - t_a;
            if (removal < cfg.need) {
                continue;
            }
            if (b + 1 >= n_beats) {
                continue;
            }
            if (!retarget_detail::rt_vocal_quiet(vocal_active, n_vocal, vocal_hop_s, t_b, t_b + cfg.guard)) {
                continue;
            }

            double context_cost = 0.0;
            for (int k = 0; k < lookback; ++k) {
                const int64_t idx_a = static_cast<int64_t>(a - 1 - k);
                const int64_t idx_b = static_cast<int64_t>(b - 1 - k);
                context_cost += w[static_cast<size_t>(k)] *
                    static_cast<double>(dist[idx_a * n_beats64 + idx_b]);
            }
            const double cost = context_cost + 0.10 * (removal - cfg.need) / std::max(cfg.need, 1.0);

            RetargetCut c;
            c.a = a;
            c.b = b;
            c.t_a = t_a;
            c.t_b = t_b;
            c.removal = removal;
            c.context_cost = retarget_detail::rt_round4(context_cost);
            c.cost = retarget_detail::rt_round4(cost);
            c.bars = (b - a) / bpb;
            all.push_back(c);
        }
    }

    // Stable sort on the already-rounded cost: ties keep the (a, b) discovery order, same as Python's
    // stable list.sort() over dicts whose 'cost' field was rounded before sorting.
    std::stable_sort(all.begin(), all.end(),
                      [](const RetargetCut& x, const RetargetCut& y) { return x.cost < y.cost; });

    for (size_t i = 0; i < all.size() && static_cast<int>(out->size()) < cfg.n_keep; ++i) {
        const RetargetCut& c = all[i];
        bool far_enough = true;
        for (size_t j = 0; j < out->size(); ++j) {
            const RetargetCut& kept = (*out)[j];
            if (!(std::fabs(c.t_a - kept.t_a) > 5.0 || std::fabs(c.t_b - kept.t_b) > 5.0)) {
                far_enough = false;
                break;
            }
        }
        if (far_enough) {
            out->push_back(c);
        }
    }
    return static_cast<int>(all.size());
}

// Equal-power crossfade splice of interleaved audio, removing [t_a, t_b) with an xfade_ms crossfade
// centred on each cut boundary. Fails (returns false) rather than producing a truncated/garbage buffer
// when the audio is too short around either boundary for the requested crossfade.
//
// DIVERGENCE from the reference: t_a/t_b -> sample index uses std::llround (round-half-away-from-zero).
// Python's round() is round-half-to-even. The two differ only when t*sr lands on an exact .5 boundary,
// which does not happen for real beat-tracker timestamps against an integer sample rate; noted rather
// than reproduced, since C++ has no built-in banker's rounding and it is not worth hand-rolling for a
// case that cannot occur here.
inline bool rt_splice(const float* in, int64_t n_frames, int channels, int sr,
                       double t_a, double t_b, double xfade_ms,
                       std::vector<float>* out, int64_t* out_frames, double* seam_time_s) {
    if (out == nullptr || in == nullptr || channels <= 0 || sr <= 0 || n_frames <= 0) {
        return false;
    }
    const int64_t ch = static_cast<int64_t>(channels);

    // h is a FRAME count (see file header comment) — half the crossfade, not half the interleaved
    // sample count.
    int64_t h = static_cast<int64_t>(static_cast<double>(sr) * xfade_ms / 2000.0);
    if (h < 1) {
        h = 1;
    }
    if (n_frames - h < h) {
        return false;  // shorter than one crossfade half; the reference's clamp would invert the range
    }

    int64_t sa = static_cast<int64_t>(std::llround(t_a * static_cast<double>(sr)));
    int64_t sb = static_cast<int64_t>(std::llround(t_b * static_cast<double>(sr)));
    const int64_t lo = h;
    const int64_t hi = n_frames - h;
    sa = std::max(lo, std::min(sa, hi));
    sb = std::max(lo, std::min(sb, hi));
    if (sb - sa < 4 * h) {
        return false;
    }

    const int64_t pre = sa - h;
    const int64_t mid = 2 * h;
    const int64_t post = n_frames - (sb + h);
    const int64_t total = pre + mid + post;

    out->assign(static_cast<size_t>(total * ch), 0.0f);
    float* dst = out->data();
    int64_t w_off = 0;

    for (int64_t i = 0; i < pre; ++i) {
        const float* src = in + (i * ch);
        for (int64_t c = 0; c < ch; ++c) {
            dst[(w_off + i) * ch + c] = src[c];
        }
    }
    w_off += pre;

    const double denom = static_cast<double>(mid - 1);  // mid >= 2 always (h >= 1), so denom >= 1
    for (int64_t i = 0; i < mid; ++i) {
        const double r = static_cast<double>(i) / denom;
        const double gain_out = std::cos(r * kRetargetPi / 2.0);
        const double gain_in  = std::sin(r * kRetargetPi / 2.0);
        const float* src_out = in + ((sa - h + i) * ch);
        const float* src_in  = in + ((sb - h + i) * ch);
        for (int64_t c = 0; c < ch; ++c) {
            dst[(w_off + i) * ch + c] = static_cast<float>(
                static_cast<double>(src_out[c]) * gain_out + static_cast<double>(src_in[c]) * gain_in);
        }
    }
    w_off += mid;

    for (int64_t i = 0; i < post; ++i) {
        const float* src = in + ((sb + h + i) * ch);
        for (int64_t c = 0; c < ch; ++c) {
            dst[(w_off + i) * ch + c] = src[c];
        }
    }

    if (out_frames != nullptr) {
        *out_frames = total;
    }
    if (seam_time_s != nullptr) {
        *seam_time_s = static_cast<double>(sa - h) / static_cast<double>(sr);
    }
    return true;
}

// Vocal-activity mask from a mono stem: RMS per hop_s window, dB relative to the loudest window, active
// where dB > thresh_db, then any active run shorter than min_len_s is cleared. Reference defaults:
// hop_s=0.1, thresh_db=-30.0, min_len_s=0.6 (not defaulted here — pass them explicitly).
inline void rt_vocal_mask(const float* stem_mono, int64_t n, int sr, double hop_s, double thresh_db,
                           double min_len_s, std::vector<uint8_t>* out) {
    if (out == nullptr) {
        return;
    }
    out->clear();
    if (stem_mono == nullptr || n <= 0 || sr <= 0 || hop_s <= 0.0) {
        return;
    }

    int64_t hop_n = static_cast<int64_t>(static_cast<double>(sr) * hop_s);
    if (hop_n < 1) {
        hop_n = 1;
    }
    const int64_t m = n / hop_n;
    if (m <= 0) {
        return;
    }

    std::vector<double> rms(static_cast<size_t>(m));
    double rms_max = 1e-12;
    for (int64_t i = 0; i < m; ++i) {
        const int64_t start = i * hop_n;
        double acc = 0.0;
        for (int64_t j = 0; j < hop_n; ++j) {
            const double v = static_cast<double>(stem_mono[start + j]);
            acc += v * v;
        }
        const double r = std::sqrt(acc / static_cast<double>(hop_n)) + 1e-12;
        rms[static_cast<size_t>(i)] = r;
        rms_max = std::max(rms_max, r);
    }

    out->assign(static_cast<size_t>(m), static_cast<uint8_t>(0));
    for (int64_t i = 0; i < m; ++i) {
        const double db = 20.0 * std::log10(rms[static_cast<size_t>(i)] / rms_max);
        (*out)[static_cast<size_t>(i)] = (db > thresh_db) ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }

    // Debounce: clear any active run shorter than min_len_s. Two-pointer run walk, same structure as the
    // reference's while-loop (not a vectorised min-filter), so a run's boundary lands identically.
    int64_t i = 0;
    while (i < m) {
        if ((*out)[static_cast<size_t>(i)] != 0) {
            int64_t j = i;
            while (j < m && (*out)[static_cast<size_t>(j)] != 0) {
                ++j;
            }
            if (static_cast<double>(j - i) * hop_s < min_len_s) {
                for (int64_t k = i; k < j; ++k) {
                    (*out)[static_cast<size_t>(k)] = 0;
                }
            }
            i = j;
        } else {
            ++i;
        }
    }
}
