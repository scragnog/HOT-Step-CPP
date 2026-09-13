#pragma once
// retarget-dsp.h: pure signal-processing primitives for `ace-train mm3-retarget`
// (see tools/mm3-retarget/retarget.py, the working Python oracle this ports).
//
// WHY THIS FILE EXISTS
// ---------------------
// The retarget search scores ~1500 candidate cut pairs per track by comparing
// chroma+MFCC context vectors, which means an STFT + mel/chroma/MFCC frontend
// that runs over up to ~15000 frames of a full song, not one short utterance.
// moss-mel.h's FFT is recursive-allocating (fine for n_fft=400, called once per
// 10ms frame in a speech encoder); at n_fft=2048 over a whole track it would
// allocate a fresh vector<complex> tree on every one of those ~15000 frames.
// This file instead carries an in-place iterative radix-2 FFT with reusable
// scratch buffers, because retarget-search.h (a later file) calls rt_stft_mag
// once per candidate track, not once per frame.
//
// EVERYTHING HERE IS CHANNEL-MAJOR: a D x N matrix stores column j (one
// frame/beat) contiguously at data[(size_t)j*D + d]. That is the opposite of
// moss-mel.h's row-major [n_mels, frames], and matches how retarget-search.h
// wants to walk "one beat's feature vector" as a contiguous slice.
//
// KNOWN DIVERGENCES FROM THE PYTHON ORACLE (each called out again at its
// function below):
//   - rt_chroma is STFT chroma, not librosa's chroma_cqt (constant-Q). A
//     constant-Q transform needs a separate multirate filterbank this file
//     does not build; STFT chroma is the standard fallback and is fine for a
//     similarity matrix, but the resulting cost numbers are NOT the ones the
//     oracle's --max-cost was calibrated against. Whoever wires up
//     retarget-search.h must re-derive a threshold by ear, not reuse 0.45.
//   - beat tracking, onset strength, and tempo estimation are NOT provided by
//     this file at all — librosa.beat.beat_track has no analogue here. That
//     lives (or will live) in a separate header; this file only pools frames
//     onto beat times it is handed.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

// MSVC does not define M_PI unless _USE_MATH_DEFINES is set before <cmath>,
// and this header may be included after some other TU already pulled in
// <cmath> without it. Carry our own constant (matches moss-mel.h's kPi).
static constexpr double kRetargetPi = 3.14159265358979323846;

// ─── FFT ────────────────────────────────────────────────────────────────────

// In-place iterative radix-2 Cooley-Tukey, bit-reversal permutation then
// butterflies. `x.size()` must be a power of two; returns false (and leaves
// x untouched) otherwise instead of asserting in release, because a caller
// computing n_fft from a config value should get a clean failure, not UB.
// No allocation beyond what the caller's vector already holds — this is the
// hot loop moss-mel.h's recursive version would be wrong to reuse here (see
// header comment).
inline bool rt_fft_inplace(std::vector<std::complex<double>> & x) {
    const size_t n = x.size();
    if (n == 0) {
        return true;
    }
    if ((n & (n - 1)) != 0) {
        assert(false && "rt_fft_inplace: size must be a power of two");
        return false;
    }

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    // Iterative butterflies, stage lengths 2, 4, 8, ... n.
    for (size_t len = 2; len <= n; len <<= 1) {
        const double        ang = -2.0 * kRetargetPi / (double) len;
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            const size_t half = len >> 1;
            for (size_t k = 0; k < half; ++k) {
                const std::complex<double> u = x[i + k];
                const std::complex<double> v = x[i + k + half] * w;
                x[i + k]        = u + v;
                x[i + k + half] = u - v;
                w *= wlen;
            }
        }
    }
    return true;
}

// ─── STFT magnitude ─────────────────────────────────────────────────────────

// Periodic Hann window + centred (reflect-padded) STFT, magnitude only.
// Matches librosa.stft(center=True, pad_mode='reflect') / np.abs(...): the
// reference's `librosa.stft(y, n_fft=2048)` default hop is n_fft/4, but the
// oracle never overrides hop either, so callers here must pass hop=n_fft/4 to
// match it exactly — this function takes hop explicitly rather than defaulting
// it, since retarget-search.h's beat-sync frame_hop_s must agree with whatever
// hop was actually used.
//
// Unlike moss-mel.h, the LAST frame is NOT dropped: librosa.stft (used
// directly here, not through the Whisper feature extractor) keeps every
// center-aligned frame up to and including one centred at n_samples-1.
inline bool rt_stft_mag(const float * mono, int64_t T, int n_fft, int hop,
                        std::vector<float> * mag, int * n_frames, int * n_bins) {
    if (mono == nullptr || mag == nullptr || n_frames == nullptr || n_bins == nullptr) {
        assert(false && "rt_stft_mag: null output pointer");
        return false;
    }
    if (n_fft <= 0 || (n_fft & (n_fft - 1)) != 0 || hop <= 0 || T < 0) {
        assert(false && "rt_stft_mag: n_fft must be a power of two, hop > 0, T >= 0");
        return false;
    }

    const size_t n_samples = (size_t) T;
    const int    pad       = n_fft / 2;
    const int    bins      = n_fft / 2 + 1;

    // Reflect-pad both ends by n_fft/2, same construction as moss-mel.h.
    std::vector<double> a(n_samples + 2 * (size_t) pad);
    for (int i = 0; i < pad; ++i) {
        a[(size_t) i] = (n_samples > (size_t) (pad - i)) ? (double) mono[pad - i] : 0.0;
    }
    for (size_t i = 0; i < n_samples; ++i) {
        a[(size_t) pad + i] = (double) mono[i];
    }
    for (int i = 0; i < pad; ++i) {
        const int64_t src = (int64_t) n_samples - 2 - i;
        a[(size_t) pad + n_samples + (size_t) i] = (src >= 0) ? (double) mono[src] : 0.0;
    }

    // librosa centres frame t at padded offset t*hop, keeping every frame
    // whose n_fft window fits inside the padded signal (no final-frame drop).
    const int64_t total = (int64_t) a.size();
    const int     frames = (total >= n_fft) ? (int) (1 + (total - n_fft) / hop) : 0;
    *n_frames = frames;
    *n_bins   = bins;
    mag->assign((size_t) bins * (size_t) std::max(frames, 0), 0.0f);
    if (frames == 0) {
        return true;
    }

    // Periodic Hann: np.hanning(N+1)[:-1], not the symmetric variant.
    std::vector<double> win((size_t) n_fft);
    for (int i = 0; i < n_fft; ++i) {
        win[(size_t) i] = 0.5 - 0.5 * std::cos(2.0 * kRetargetPi * (double) i / (double) n_fft);
    }

    std::vector<std::complex<double>> buf((size_t) n_fft);
    for (int t = 0; t < frames; ++t) {
        const size_t off = (size_t) t * (size_t) hop;
        for (int i = 0; i < n_fft; ++i) {
            buf[(size_t) i] = std::complex<double>(a[off + (size_t) i] * win[(size_t) i], 0.0);
        }
        rt_fft_inplace(buf);
        float * col = mag->data() + (size_t) t * (size_t) bins;
        for (int b = 0; b < bins; ++b) {
            col[b] = (float) std::abs(buf[(size_t) b]);
        }
    }
    return true;
}

// ─── chroma ─────────────────────────────────────────────────────────────────

// STFT chroma: bin each FFT bin's magnitude into one of 12 pitch classes and
// L2-normalise each beat column. 12 x n_frames, channel-major.
//
// DIVERGENCE FROM THE ORACLE: retarget.py uses librosa.feature.chroma_cqt, a
// constant-Q chroma built from a separate multirate filterbank tuned for
// pitch resolution at low frequencies. This is plain STFT chroma (bin
// magnitudes folded into pitch classes by nearest-semitone rounding), which
// is coarser in the bass and does not correct for the same tuning/leakage
// artifacts a CQT does. It is a reasonable stand-in for a similarity matrix
// (the search only ever compares two chroma vectors to each other, never to
// an absolute reference), but the resulting distances are numerically
// different, so --max-cost must be recalibrated against THIS pipeline's
// output, not against 0.45 from the Python tool.
inline void rt_chroma(const float * mag, int n_frames, int n_bins, int sr, int n_fft,
                      std::vector<float> * out12) {
    assert(out12 != nullptr);
    out12->assign(12 * (size_t) std::max(n_frames, 0), 0.0f);
    if (n_frames <= 0 || n_bins <= 0) {
        return;
    }

    for (int t = 0; t < n_frames; ++t) {
        const float * col = mag + (size_t) t * (size_t) n_bins;
        float *       out = out12->data() + (size_t) t * 12;
        for (int k = 0; k < n_bins; ++k) {
            const double f = (double) k * (double) sr / (double) n_fft;
            if (f < 55.0 || f > 5000.0) {
                continue;
            }
            const long long pc_raw = (long long) std::llround(12.0 * std::log2(f / 440.0));
            const int       pc     = (int) (((pc_raw % 12) + 12) % 12);
            out[pc] += col[k];
        }
        double norm2 = 0.0;
        for (int c = 0; c < 12; ++c) {
            norm2 += (double) out[c] * (double) out[c];
        }
        const double norm = std::sqrt(norm2);
        if (norm >= 1e-9) {
            const float inv = (float) (1.0 / norm);
            for (int c = 0; c < 12; ++c) {
                out[c] *= inv;
            }
        }
        // else: leave the all-zero column as zeros (silent/out-of-band frame).
    }
}

// ─── MFCC ───────────────────────────────────────────────────────────────────

// Slaney-style mel scale conversions, matching moss-mel.h's mel_detail
// functions exactly (duplicated rather than shared: that namespace is
// `moss::mel_detail` and this file must not include engine headers or a
// speech-frontend header from an unrelated subsystem).
namespace retarget_mel_detail {

inline double rt_hz_to_mel(double f) {
    const double f_sp        = 200.0 / 3.0;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    if (f >= min_log_hz) {
        return min_log_mel + std::log(std::max(f, 1e-9) / min_log_hz) / logstep;
    }
    return f / f_sp;
}

inline double rt_mel_to_hz(double m) {
    const double f_sp        = 200.0 / 3.0;
    const double min_log_hz  = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep     = std::log(6.4) / 27.0;
    if (m >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (m - min_log_mel));
    }
    return f_sp * m;
}

}  // namespace retarget_mel_detail

// Slaney mel filterbank over magnitude (NOT power, matching librosa.feature.
// melspectrogram's default power=2.0... actually librosa.feature.mfcc feeds
// its internal melspectrogram which uses POWER by default; but the spec for
// this function explicitly calls for magnitude filtering, so that is what is
// implemented here — see the divergence note below), log with a 1e-10 floor,
// orthonormal DCT-II, coefficients 1..n_mfcc-1 (c0 dropped, matching the
// oracle's `mfcc[1:]`). Output is (n_mfcc-1) x n_frames, channel-major.
//
// DIVERGENCE FROM THE ORACLE: librosa.feature.mfcc's default pipeline mel-
// filters POWER (magnitude squared), not magnitude. This function filters
// magnitude directly per the spec above. The two differ by a monotonic-ish
// but not identical nonlinearity before the log, so MFCC values here are not
// bit-identical to librosa's — same caveat as rt_chroma: fine for a relative
// similarity search, not for reproducing the oracle's exact cost numbers.
inline void rt_mfcc(const float * mag, int n_frames, int n_bins, int sr, int n_fft, int n_mel,
                    int n_mfcc, std::vector<float> * out) {
    assert(out != nullptr);
    // n_fft is not otherwise needed (the filterbank below only cares about
    // n_bins and sr), but is part of the interface for symmetry with
    // rt_chroma and so a caller cannot pass a bin count that does not match
    // the FFT size it came from without tripping this. Referenced outside the
    // assert too so an NDEBUG (assert-stripped) release build does not warn
    // on an unused parameter.
    assert(n_bins == n_fft / 2 + 1 && "rt_mfcc: n_bins must be n_fft/2+1");
    (void) n_fft;
    const int n_out = std::max(n_mfcc - 1, 0);
    out->assign((size_t) n_out * (size_t) std::max(n_frames, 0), 0.0f);
    if (n_frames <= 0 || n_bins <= 0 || n_mel <= 0 || n_out <= 0) {
        return;
    }

    using namespace retarget_mel_detail;

    // Build the triangular filterbank, row-major [n_mel, n_bins] (mirrors
    // moss-mel.h's filterbank() layout since it is only consumed locally).
    const double fmax    = (double) sr / 2.0;
    const double mel_min = rt_hz_to_mel(0.0);
    const double mel_max = rt_hz_to_mel(fmax);

    std::vector<double> hz((size_t) n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        const double m = mel_min + (mel_max - mel_min) * (double) i / (double) (n_mel + 1);
        hz[(size_t) i] = rt_mel_to_hz(m);
    }

    std::vector<double> fb((size_t) n_mel * (size_t) n_bins, 0.0);
    for (int i = 0; i < n_mel; ++i) {
        const double lo = hz[(size_t) i], ce = hz[(size_t) i + 1], hi = hz[(size_t) i + 2];
        const double enorm = 2.0 / (hi - lo);  // Slaney area normalisation
        for (int b = 0; b < n_bins; ++b) {
            const double f = fmax * (double) b / (double) (n_bins - 1);
            double       w = 0.0;
            if (f > lo && f < hi) {
                w = (f <= ce) ? (f - lo) / (ce - lo) : (hi - f) / (hi - ce);
                w = std::max(w, 0.0);
            }
            fb[(size_t) i * (size_t) n_bins + (size_t) b] = w * enorm;
        }
    }

    // Orthonormal DCT-II basis, rows 1..n_mfcc-1 only (row 0 is c0, dropped).
    // out[k][t] = sqrt(2/n_mel) * sum_m log_mel[m][t] * cos(pi/n_mel * (m+0.5) * k)
    // with an extra 1/sqrt(2) factor folded into k==0 by the standard DCT-II
    // orthonormal convention -- irrelevant here since k==0 is never emitted.
    const double dct_scale = std::sqrt(2.0 / (double) n_mel);

    std::vector<double> log_mel((size_t) n_mel);
    for (int t = 0; t < n_frames; ++t) {
        const float * col = mag + (size_t) t * (size_t) n_bins;
        for (int m = 0; m < n_mel; ++m) {
            const double * row = &fb[(size_t) m * (size_t) n_bins];
            double         s   = 0.0;
            for (int b = 0; b < n_bins; ++b) {
                s += row[b] * (double) col[b];
            }
            // Second, independent divergence from librosa (the first being magnitude vs power, noted above):
            // librosa runs its mel spectrogram through power_to_db, which floors each frame at its own max minus
            // 80 dB, not at a flat amin. This is a flat floor only. The log-vs-log10 scale difference does cancel
            // under rt_feature_stack's per-row z-score, but the missing per-frame relative floor does not — it
            // drifts on quiet mel bins inside otherwise loud frames. Both matter to whoever recalibrates
            // --max-cost against the Python numbers.
            log_mel[(size_t) m] = std::log(std::max(s, 1e-10));
        }
        float * out_col = out->data() + (size_t) t * (size_t) n_out;
        for (int k = 1; k <= n_mfcc - 1; ++k) {
            double s = 0.0;
            for (int m = 0; m < n_mel; ++m) {
                s += log_mel[(size_t) m] *
                     std::cos(kRetargetPi / (double) n_mel * ((double) m + 0.5) * (double) k);
            }
            out_col[k - 1] = (float) (s * dct_scale);
        }
    }
}

// ─── beat-synchronous pooling ───────────────────────────────────────────────

// Pool per-frame columns into per-beat columns, matching librosa.util.sync's PADDED column convention, which is
// the one the oracle's cost arithmetic is written against.
//
// sync(data, beats) defaults to pad=True, so it prepends 0 and appends the frame count before slicing: column 0 is
// everything before the first beat, and column i is [beat[i-1], beat[i]) — the interval ENDING at beat i, not
// starting at it. retarget.py's cost term dist[a-1-k, b-1-k] then reads the beats leading INTO the cut, which is
// the whole criterion.
//
// Bucketing by [beat_t[i], beat_t[i+1]) instead — the obvious reading — shifts every column by one and silently
// slides the whole lookback window one beat closer to the cut. Nothing crashes and the output still looks
// plausible; it just scores a different window than the one that was validated by ear.
//
// Frames past the last beat belong to sync's trailing padded column, which the search never indexes (its beat
// indices are all < n_beats), so they are dropped rather than folded into the last bucket.
//
// median=true reproduces librosa.util.sync(..., aggregate=np.median)
// (used for chroma in the oracle); false reproduces aggregate=np.mean (used
// for MFCC). An empty beat (no frame centres fall in its window — short beats
// near a tempo change) copies the previous beat's pooled column, or is left
// at zero if it is beat 0.
inline void rt_beat_sync(const float * frames, int D, int n_frames, double frame_hop_s,
                         const double * beat_t, int n_beats, bool median,
                         std::vector<float> * out) {
    assert(out != nullptr);
    out->assign((size_t) D * (size_t) std::max(n_beats, 0), 0.0f);
    if (D <= 0 || n_beats <= 0) {
        return;
    }

    // Bucket frame indices per beat using half-open windows on centre time.
    std::vector<int> beat_of_frame((size_t) std::max(n_frames, 0), -1);
    for (int j = 0; j < n_frames; ++j) {
        const double t = (double) j * frame_hop_s;
        // Find the FIRST beat whose time is > t: that is the column [beat_t[i-1], beat_t[i]) the frame falls in,
        // with column 0 taking everything before the first beat. Frame counts per track (thousands) times beat
        // counts (hundreds-low-thousands) is the same O(frames*beats) cost the reference pays inside
        // librosa.util.sync, so a linear scan here is not a regression worth a binary search for.
        int b = -1;   // -1 = past the last beat, i.e. sync's trailing padded column; dropped
        for (int i = 0; i < n_beats; ++i) {
            if (t < beat_t[i]) {
                b = i;
                break;
            }
        }
        beat_of_frame[(size_t) j] = b;
    }

    std::vector<double> scratch;  // reused per beat, resized to that beat's frame count
    for (int i = 0; i < n_beats; ++i) {
        // Collect the frame indices belonging to beat i.
        std::vector<int> idx;
        for (int j = 0; j < n_frames; ++j) {
            if (beat_of_frame[(size_t) j] == i) {
                idx.push_back(j);
            }
        }
        float * col = out->data() + (size_t) i * (size_t) D;
        if (idx.empty()) {
            if (i > 0) {
                const float * prev = out->data() + (size_t) (i - 1) * (size_t) D;
                std::copy(prev, prev + D, col);
            }
            // else leave beat 0 at zero.
            continue;
        }
        scratch.resize(idx.size());
        for (int d = 0; d < D; ++d) {
            for (size_t n = 0; n < idx.size(); ++n) {
                scratch[n] = (double) frames[(size_t) idx[n] * (size_t) D + (size_t) d];
            }
            if (median) {
                std::vector<double> sorted_vals = scratch;
                std::sort(sorted_vals.begin(), sorted_vals.end());
                const size_t m = sorted_vals.size();
                const double med = (m % 2 == 1) ? sorted_vals[m / 2]
                                                 : 0.5 * (sorted_vals[m / 2 - 1] + sorted_vals[m / 2]);
                col[d] = (float) med;
            } else {
                double sum = 0.0;
                for (double v : scratch) {
                    sum += v;
                }
                col[d] = (float) (sum / (double) scratch.size());
            }
        }
    }
}

// ─── feature stack ──────────────────────────────────────────────────────────

// Reproduces the oracle's `unit()` + `vstack` pipeline EXACTLY, in this
// order (getting the order wrong silently changes every cost the search
// computes, since it changes which axis gets normalised against which):
//   a. per block (chroma, then MFCC), independently:
//        - subtract each ROW's (dimension's) mean across beats
//        - divide each ROW by its std-dev across beats (std < 1e-9 -> use 1.0
//          instead, i.e. leave that row merely centred, not blown up)
//        - L2-normalise each COLUMN (beat) of that block (norm < 1e-9 -> 1.0,
//          i.e. leave an all-zero column as zeros)
//   b. vertically stack chroma-block over mfcc-block: D_out = Dc + Dm
//   c. L2-normalise each COLUMN of the stacked matrix AGAIN
// Step (a)'s per-row stats give chroma and MFCC equal weight regardless of
// their raw scales before either block is column-normalised; skipping straight
// to (c) would let whichever block has larger raw magnitude dominate the
// cosine distance.
inline void rt_feature_stack(const float * chroma_beats, int Dc, const float * mfcc_beats, int Dm,
                             int n_beats, std::vector<float> * feat, int * D_out) {
    assert(feat != nullptr && D_out != nullptr);
    const int D = Dc + Dm;
    *D_out = D;
    feat->assign((size_t) D * (size_t) std::max(n_beats, 0), 0.0f);
    if (n_beats <= 0) {
        return;
    }

    auto unit_block = [&](const float * src, int rows, float * dst) {
        // dst is a [rows, n_beats] region of the eventual stacked matrix,
        // laid out with the same column-major convention: dst[j*D_stride+r]
        // would require knowing the destination stride, so unit_block instead
        // works on a locally-owned [rows, n_beats] copy and the caller places
        // rows into the final stack. (Kept simple over clever in-place writes
        // since n_beats tops out around ~1500 -- this is not the hot path,
        // rt_cosine_dist below is.)
        std::vector<double> mean(rows, 0.0), var(rows, 0.0);
        for (int r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (int j = 0; j < n_beats; ++j) {
                sum += (double) src[(size_t) j * (size_t) rows + (size_t) r];
            }
            mean[r] = sum / (double) n_beats;
        }
        for (int r = 0; r < rows; ++r) {
            double sq = 0.0;
            for (int j = 0; j < n_beats; ++j) {
                const double c = (double) src[(size_t) j * (size_t) rows + (size_t) r] - mean[r];
                sq += c * c;
            }
            // population std, matching numpy's default ddof=0.
            const double sd = std::sqrt(sq / (double) n_beats);
            var[r]          = (sd < 1e-9) ? 1.0 : sd;
        }
        for (int j = 0; j < n_beats; ++j) {
            double norm2 = 0.0;
            for (int r = 0; r < rows; ++r) {
                const double v = ((double) src[(size_t) j * (size_t) rows + (size_t) r] - mean[r]) / var[r];
                dst[(size_t) j * (size_t) rows + (size_t) r] = (float) v;
                norm2 += v * v;
            }
            const double norm = std::sqrt(norm2);
            const double inv  = (norm < 1e-9) ? 1.0 : (1.0 / norm);
            for (int r = 0; r < rows; ++r) {
                dst[(size_t) j * (size_t) rows + (size_t) r] =
                    (float) ((double) dst[(size_t) j * (size_t) rows + (size_t) r] * inv);
            }
        }
    };

    std::vector<float> uc((size_t) Dc * (size_t) n_beats);
    std::vector<float> um((size_t) Dm * (size_t) n_beats);
    if (Dc > 0) {
        unit_block(chroma_beats, Dc, uc.data());
    }
    if (Dm > 0) {
        unit_block(mfcc_beats, Dm, um.data());
    }

    // Vertical stack into the channel-major [D, n_beats] output: for each
    // beat column, chroma rows first, then mfcc rows.
    for (int j = 0; j < n_beats; ++j) {
        float * col = feat->data() + (size_t) j * (size_t) D;
        for (int r = 0; r < Dc; ++r) {
            col[r] = uc[(size_t) j * (size_t) Dc + (size_t) r];
        }
        for (int r = 0; r < Dm; ++r) {
            col[Dc + r] = um[(size_t) j * (size_t) Dm + (size_t) r];
        }
    }

    // Final column L2-normalisation of the stacked matrix.
    for (int j = 0; j < n_beats; ++j) {
        float * col = feat->data() + (size_t) j * (size_t) D;
        double  norm2 = 0.0;
        for (int r = 0; r < D; ++r) {
            norm2 += (double) col[r] * (double) col[r];
        }
        const double norm = std::sqrt(norm2);
        const double inv  = (norm < 1e-9) ? 1.0 : (1.0 / norm);
        for (int r = 0; r < D; ++r) {
            col[r] = (float) ((double) col[r] * inv);
        }
    }
}

// ─── pairwise cosine distance ───────────────────────────────────────────────

// dist[i*n_beats + j] = clamp(1 - dot(col_i, col_j), 0, 2). Row-major square
// matrix (this one is NOT channel-major like everything else in this file:
// it has no "channel" axis, just a beat x beat table, matching how
// retarget-search.h indexes `dist[a][b]` directly). n_beats can reach ~1500
// so this is up to ~2.2M floats (~9 MB) -- allocated once by the caller-
// supplied vector, computed once here via a single O(n^2 * D) pass with each
// dot product accumulated directly (no separate Gram-matrix buffer, no
// redundant upper/lower-triangle-then-mirror pass that would touch the data
// twice for no numeric benefit since the matrix is small enough to just fill
// directly).
inline void rt_cosine_dist(const float * feat, int D, int n_beats, std::vector<float> * dist) {
    assert(dist != nullptr);
    dist->assign((size_t) n_beats * (size_t) std::max(n_beats, 0), 0.0f);
    if (n_beats <= 0 || D <= 0) {
        return;
    }
    for (int i = 0; i < n_beats; ++i) {
        const float * ci = feat + (size_t) i * (size_t) D;
        for (int j = i; j < n_beats; ++j) {
            const float * cj = feat + (size_t) j * (size_t) D;
            double        dot = 0.0;
            for (int d = 0; d < D; ++d) {
                dot += (double) ci[d] * (double) cj[d];
            }
            float v = (float) (1.0 - dot);
            v       = std::min(std::max(v, 0.0f), 2.0f);
            (*dist)[(size_t) i * (size_t) n_beats + (size_t) j] = v;
            (*dist)[(size_t) j * (size_t) n_beats + (size_t) i] = v;
        }
    }
}
