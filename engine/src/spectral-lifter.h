#pragma once
// spectral-lifter.h: native C++ port of the Spectral Lifter pipeline
//
// Removes AI shimmer artifacts, reduces spectral noise, optionally extends
// high-frequency content and shapes transients. Operates on planar stereo
// float32 audio [L0..LN, R0..RN] at 48kHz.
//
// Pipeline stages (all optional via params):
//   1. Cutoff Analysis  — detect 12-16kHz rolloff boundary
//   2. Spectral Denoise — sigmoid soft-gated spectral subtraction
//   3. HF Extension     — spectral mirroring above cutoff (optional)
//   4. Transient Shaping — HPSS-based percussive boost (optional)
//   5. Multiband Dynamics — de-essing + shimmer + HF artifact suppression
//
// Self-contained: includes a minimal radix-2 FFT (same pattern as denoiser.h).
// Zero external dependencies, zero upstream file changes.
//
// Usage:
//   SpectralLifterParams p;
//   spectral_lifter_default(&p);
//   spectral_lifter_process(audio, n_samples, 48000, &p);

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Parameters ───────────────────────────────────────────────────────────────

struct SpectralLifterParams {
    float denoise_strength;    // 0.0–1.0, 0=skip. Higher = more aggressive
    float noise_floor;         // 0.01–0.5, minimum mask value (residual leakage)
    float hf_mix;              // 0.0–0.5, HF extension blend. 0=off
    float transient_boost;     // 0.0–1.0, percussive enhancement. 0=off
    float shimmer_reduction;   // 0.0–12.0 dB, shimmer band attenuation
};

static inline void spectral_lifter_default(SpectralLifterParams * p) {
    p->denoise_strength   = 0.3f;
    p->noise_floor        = 0.1f;
    p->hf_mix             = 0.0f;
    p->transient_boost    = 0.0f;
    p->shimmer_reduction  = 6.0f;
}

// ── Internal Implementation ──────────────────────────────────────────────────

namespace sl_detail {

// ── Minimal Radix-2 FFT (same as denoiser_detail) ───────────────────────────

struct Cpx {
    float re, im;
    Cpx() : re(0), im(0) {}
    Cpx(float r, float i) : re(r), im(i) {}
    Cpx operator+(const Cpx & o) const { return {re + o.re, im + o.im}; }
    Cpx operator-(const Cpx & o) const { return {re - o.re, im - o.im}; }
    Cpx operator*(const Cpx & o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
};

static void bit_rev(Cpx * x, int N) {
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { Cpx t = x[i]; x[i] = x[j]; x[j] = t; }
    }
}

static void fft_ip(Cpx * x, int N, bool inv) {
    bit_rev(x, N);
    for (int len = 2; len <= N; len <<= 1) {
        float ang = 2.0f * 3.14159265358979f / (float)len * (inv ? -1.0f : 1.0f);
        Cpx wl(cosf(ang), sinf(ang));
        for (int i = 0; i < N; i += len) {
            Cpx w(1, 0);
            for (int j = 0; j < len / 2; j++) {
                Cpx u = x[i + j], v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w = w * wl;
            }
        }
    }
    if (inv) {
        float s = 1.0f / (float)N;
        for (int i = 0; i < N; i++) { x[i].re *= s; x[i].im *= s; }
    }
}

static void rfft(const float * in, Cpx * out, int N) {
    std::vector<Cpx> buf(N);
    for (int i = 0; i < N; i++) buf[i] = Cpx(in[i], 0);
    fft_ip(buf.data(), N, false);
    for (int i = 0; i <= N / 2; i++) out[i] = buf[i];
}

static void irfft(const Cpx * in, float * out, int N) {
    std::vector<Cpx> buf(N);
    for (int i = 0; i <= N / 2; i++) buf[i] = in[i];
    for (int i = 1; i < N / 2; i++) buf[N - i] = Cpx(in[i].re, -in[i].im);
    fft_ip(buf.data(), N, true);
    for (int i = 0; i < N; i++) out[i] = buf[i].re;
}

// ── STFT Constants ──────────────────────────────────────────────────────────

static const int SL_FFT   = 2048;
static const int SL_HOP   = 512;
static const int SL_BINS  = SL_FFT / 2 + 1;

static void build_hann(float * w, int N) {
    for (int i = 0; i < N; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979f * (float)i / (float)N));
}

// ── STFT Frame Storage ──────────────────────────────────────────────────────

struct StftData {
    int n_frames;
    std::vector<float> mag;     // [n_frames * SL_BINS]
    std::vector<Cpx>   spec;    // [n_frames * SL_BINS]
    std::vector<float> window;

    void compute(const float * samples, int T) {
        window.resize(SL_FFT);
        build_hann(window.data(), SL_FFT);

        n_frames = 0;
        for (int s = 0; s + SL_FFT <= T; s += SL_HOP) n_frames++;
        if (n_frames < 2) return;

        mag.resize((size_t)n_frames * SL_BINS);
        spec.resize((size_t)n_frames * SL_BINS);
        std::vector<float> fb(SL_FFT);

        for (int f = 0; f < n_frames; f++) {
            int off = f * SL_HOP;
            for (int i = 0; i < SL_FFT; i++)
                fb[i] = (off + i < T) ? samples[off + i] * window[i] : 0.0f;
            rfft(fb.data(), &spec[(size_t)f * SL_BINS], SL_FFT);
            for (int b = 0; b < SL_BINS; b++) {
                auto & c = spec[(size_t)f * SL_BINS + b];
                mag[(size_t)f * SL_BINS + b] = sqrtf(c.re * c.re + c.im * c.im);
            }
        }
    }

    void synthesize(float * out, int T) {
        std::vector<float> output(T, 0.0f);
        std::vector<float> wsum(T, 0.0f);
        std::vector<float> fb(SL_FFT);

        for (int f = 0; f < n_frames; f++) {
            int off = f * SL_HOP;
            irfft(&spec[(size_t)f * SL_BINS], fb.data(), SL_FFT);
            for (int i = 0; i < SL_FFT && off + i < T; i++) {
                output[off + i] += fb[i] * window[i];
                wsum[off + i]   += window[i] * window[i];
            }
        }
        for (int i = 0; i < T; i++) {
            if (wsum[i] > 1e-8f) out[i] = output[i] / wsum[i];
            else out[i] = 0.0f;
        }
    }
};

// ── Stage 1: Cutoff Analysis ────────────────────────────────────────────────

static float detect_cutoff(const float * mag, int n_frames, int sr) {
    // Mean spectrum across all frames
    std::vector<float> mean_spec(SL_BINS, 0.0f);
    for (int f = 0; f < n_frames; f++)
        for (int b = 0; b < SL_BINS; b++)
            mean_spec[b] += mag[(size_t)f * SL_BINS + b];
    float inv_f = 1.0f / (float)n_frames;
    for (int b = 0; b < SL_BINS; b++) mean_spec[b] *= inv_f;

    // Convert to dB
    float ref = *std::max_element(mean_spec.begin(), mean_spec.end());
    if (ref < 1e-10f) return 16000.0f;
    std::vector<float> db(SL_BINS);
    for (int b = 0; b < SL_BINS; b++)
        db[b] = 20.0f * log10f(fmaxf(mean_spec[b], 1e-10f) / ref);

    // Find steepest drop between 12kHz and 16kHz
    float bin_hz = (float)sr / (float)SL_FFT;
    int lo = (int)(12000.0f / bin_hz);
    int hi = (int)(16000.0f / bin_hz);
    lo = std::max(0, std::min(lo, SL_BINS - 2));
    hi = std::min(hi, SL_BINS - 1);

    float min_diff = 0.0f;
    int   min_idx  = lo;
    for (int b = lo; b < hi; b++) {
        float d = db[b + 1] - db[b];
        if (d < min_diff) { min_diff = d; min_idx = b; }
    }
    return (float)min_idx * bin_hz;
}

// ── Stage 2: Spectral Denoising ─────────────────────────────────────────────
// Sigmoid soft-gated spectral subtraction with temporal+frequency smoothing.
// Much better than the Python version's primitive hard-clip mask.

static void spectral_denoise(StftData & st, float strength, float floor) {
    if (strength <= 0.0f || st.n_frames < 2) return;

    const int nf = st.n_frames;

    // Noise profile: 5th percentile per bin (same as Python)
    std::vector<float> noise(SL_BINS);
    std::vector<float> col(nf);
    for (int b = 0; b < SL_BINS; b++) {
        for (int f = 0; f < nf; f++)
            col[f] = st.mag[(size_t)f * SL_BINS + b];
        std::sort(col.begin(), col.end());
        int p5 = std::max(0, (int)(nf * 0.05f));
        noise[b] = col[p5];
    }

    // Threshold = noise * over_sub (strength maps to over-subtraction)
    float over_sub = 0.5f + 3.5f * strength;

    // Build gain masks with sigmoid soft gating
    std::vector<float> masks((size_t)nf * SL_BINS);
    for (int f = 0; f < nf; f++) {
        for (int b = 0; b < SL_BINS; b++) {
            float m   = st.mag[(size_t)f * SL_BINS + b];
            float thr = noise[b] * over_sub;
            float softness = fmaxf(thr * 0.3f, 1e-8f);
            float x = (m - thr) / softness;
            float s = 1.0f / (1.0f + expf(-x));
            masks[(size_t)f * SL_BINS + b] = s;
        }
    }

    // Temporal smoothing (bidirectional IIR)
    float release = 0.5f + 0.49f * 0.7f;  // moderate smoothing
    float attack  = 0.3f;
    for (int b = 0; b < SL_BINS; b++) {
        float prev = masks[b];
        for (int f = 1; f < nf; f++) {
            float curr = masks[(size_t)f * SL_BINS + b];
            float c = (curr > prev) ? attack : release;
            float sm = c * prev + (1.0f - c) * curr;
            masks[(size_t)f * SL_BINS + b] = sm;
            prev = sm;
        }
    }
    for (int b = 0; b < SL_BINS; b++) {
        float prev = masks[(size_t)(nf - 1) * SL_BINS + b];
        for (int f = nf - 2; f >= 0; f--) {
            float curr = masks[(size_t)f * SL_BINS + b];
            float sm   = 0.5f * curr + 0.5f * (release * prev + (1.0f - release) * curr);
            masks[(size_t)f * SL_BINS + b] = sm;
            prev = sm;
        }
    }

    // Frequency smoothing (Gaussian blur, sigma=4 bins)
    {
        const int sigma = 4;
        std::vector<float> tmp(SL_BINS);
        for (int f = 0; f < nf; f++) {
            float * row = &masks[(size_t)f * SL_BINS];
            memcpy(tmp.data(), row, SL_BINS * sizeof(float));
            for (int b = 0; b < SL_BINS; b++) {
                float sum = 0, wt = 0;
                for (int k = -sigma; k <= sigma; k++) {
                    int idx = b + k;
                    if (idx < 0 || idx >= SL_BINS) continue;
                    float g = expf(-0.5f * (float)(k * k) / (float)(sigma * sigma));
                    sum += tmp[idx] * g;
                    wt += g;
                }
                row[b] = sum / wt;
            }
        }
    }

    // Clamp to floor
    for (size_t i = 0; i < (size_t)nf * SL_BINS; i++)
        if (masks[i] < floor) masks[i] = floor;

    // Apply masks to spectra
    for (int f = 0; f < nf; f++) {
        for (int b = 0; b < SL_BINS; b++) {
            float g = masks[(size_t)f * SL_BINS + b];
            st.spec[(size_t)f * SL_BINS + b].re *= g;
            st.spec[(size_t)f * SL_BINS + b].im *= g;
        }
    }
}

// ── Stage 3: HF Extension (Spectral Mirroring) ─────────────────────────────

static void hf_extend(float * samples, int T, int sr, float cutoff, float mix) {
    if (mix <= 0.0f || T < SL_FFT) return;

    StftData st;
    st.compute(samples, T);
    if (st.n_frames < 2) return;

    float bin_hz = (float)sr / (float)SL_FFT;
    int src_lo   = (int)(8000.0f / bin_hz);      // source: 8-16kHz
    int src_hi   = (int)(16000.0f / bin_hz);
    int dst_lo   = (int)(fmaxf(cutoff - 1000.0f, 12000.0f) / bin_hz);
    src_lo = std::max(1, std::min(src_lo, SL_BINS - 1));
    src_hi = std::min(src_hi, SL_BINS - 1);
    dst_lo = std::max(1, std::min(dst_lo, SL_BINS - 1));

    int src_span = src_hi - src_lo;
    if (src_span <= 0) return;

    // Mirror source bins into the HF region with amplitude tapering
    for (int f = 0; f < st.n_frames; f++) {
        for (int i = 0; i < src_span; i++) {
            int dst = dst_lo + i;
            if (dst >= SL_BINS) break;
            int src = src_lo + i;
            // Taper: fade out as we go higher
            float taper = 1.0f - (float)i / (float)src_span;
            taper *= taper;  // quadratic falloff
            float g = mix * taper;
            st.spec[(size_t)f * SL_BINS + dst].re += st.spec[(size_t)f * SL_BINS + src].re * g;
            st.spec[(size_t)f * SL_BINS + dst].im += st.spec[(size_t)f * SL_BINS + src].im * g;
        }
    }

    st.synthesize(samples, T);
}

// ── Stage 4: Transient Shaping (simplified HPSS) ────────────────────────────

static void transient_shape(float * samples, int T, int sr, float boost) {
    if (boost <= 0.0f || T < SL_FFT) return;

    // Compute spectrogram
    StftData full;
    full.compute(samples, T);
    if (full.n_frames < 2) return;

    const int nf = full.n_frames;

    // Percussive mask via frequency-axis median filtering
    // Percussive = content that changes rapidly across frequency (drums, clicks)
    const int med_k = 15;  // median filter kernel size
    std::vector<float> perc_mask((size_t)nf * SL_BINS);
    std::vector<float> kern(med_k);

    for (int f = 0; f < nf; f++) {
        for (int b = 0; b < SL_BINS; b++) {
            // Gather frequency-axis neighbors
            int count = 0;
            for (int k = -med_k / 2; k <= med_k / 2; k++) {
                int idx = b + k;
                if (idx < 0) idx = 0;
                if (idx >= SL_BINS) idx = SL_BINS - 1;
                kern[count++] = full.mag[(size_t)f * SL_BINS + idx];
            }
            std::sort(kern.begin(), kern.begin() + count);
            float freq_med = kern[count / 2];

            // Percussive ratio: higher when signal exceeds frequency median
            float sig = full.mag[(size_t)f * SL_BINS + b];
            float ratio = (sig > 1e-10f) ? fminf(sig / (freq_med + 1e-10f), 4.0f) : 0.0f;
            perc_mask[(size_t)f * SL_BINS + b] = fmaxf(0.0f, fminf(1.0f, (ratio - 1.0f)));
        }
    }

    // Extract percussive component
    StftData perc;
    perc.n_frames = nf;
    perc.spec.resize((size_t)nf * SL_BINS);
    perc.window = full.window;
    for (int f = 0; f < nf; f++) {
        for (int b = 0; b < SL_BINS; b++) {
            float m = perc_mask[(size_t)f * SL_BINS + b];
            perc.spec[(size_t)f * SL_BINS + b].re = full.spec[(size_t)f * SL_BINS + b].re * m;
            perc.spec[(size_t)f * SL_BINS + b].im = full.spec[(size_t)f * SL_BINS + b].im * m;
        }
    }

    // Synthesize percussive and add to original
    std::vector<float> perc_audio(T, 0.0f);
    perc.synthesize(perc_audio.data(), T);

    for (int i = 0; i < T; i++)
        samples[i] += perc_audio[i] * boost;
}

// ── Stage 5: Multiband Dynamics ─────────────────────────────────────────────

struct BandConfig {
    float lo_hz, hi_hz;
    float reduction_db;
    float threshold_pct;  // percentile for threshold
};

static void multiband_dynamics(float * samples, int T, int sr,
                                float shimmer_db) {
    if (T < SL_FFT) return;

    StftData st;
    st.compute(samples, T);
    if (st.n_frames < 2) return;

    const int nf = st.n_frames;
    float bin_hz = (float)sr / (float)SL_FFT;

    BandConfig bands[] = {
        { 5000,  8000, 3.0f,        80.0f },   // sibilance
        {10000, 14000, shimmer_db,   60.0f },   // shimmer
        {18000, 24000, 12.0f,        50.0f },   // HF artifacts
    };

    for (auto & band : bands) {
        if (band.reduction_db <= 0.0f) continue;

        int b_lo = std::max(0, (int)(band.lo_hz / bin_hz));
        int b_hi = std::min(SL_BINS, (int)(band.hi_hz / bin_hz));
        if (b_lo >= b_hi || b_lo >= SL_BINS) continue;
        int bw = b_hi - b_lo;

        // Compute per-frame band energy
        std::vector<float> energy(nf);
        for (int f = 0; f < nf; f++) {
            float sum = 0;
            for (int b = b_lo; b < b_hi; b++)
                sum += st.mag[(size_t)f * SL_BINS + b];
            energy[f] = sum / (float)bw;
        }

        // Threshold from percentile
        std::vector<float> sorted_e(energy);
        std::sort(sorted_e.begin(), sorted_e.end());
        int p_idx = std::min((int)(nf * band.threshold_pct / 100.0f), nf - 1);
        float thresh = sorted_e[p_idx];

        float red_lin = powf(10.0f, -band.reduction_db / 20.0f);

        // Build per-frame gain mask
        std::vector<float> gain(nf, 1.0f);
        for (int f = 0; f < nf; f++) {
            if (energy[f] > thresh) {
                gain[f] = red_lin + (1.0f - red_lin) * (thresh / (energy[f] + 1e-10f));
            }
        }

        // Smooth gain (5-sample moving average)
        std::vector<float> sg(nf);
        for (int f = 0; f < nf; f++) {
            float sum = 0; int cnt = 0;
            for (int k = -2; k <= 2; k++) {
                int idx = f + k;
                if (idx >= 0 && idx < nf) { sum += gain[idx]; cnt++; }
            }
            sg[f] = std::clamp(sum / (float)cnt, red_lin, 1.0f);
        }

        // Apply to spectra
        for (int f = 0; f < nf; f++) {
            for (int b = b_lo; b < b_hi; b++) {
                st.spec[(size_t)f * SL_BINS + b].re *= sg[f];
                st.spec[(size_t)f * SL_BINS + b].im *= sg[f];
            }
        }
    }

    st.synthesize(samples, T);
}

// ── Per-Channel Pipeline ────────────────────────────────────────────────────

static void process_mono(float * samples, int T, int sr,
                          const SpectralLifterParams * p) {
    if (T < SL_FFT) return;

    // 1. Cutoff analysis
    StftData an;
    an.compute(samples, T);
    if (an.n_frames < 2) return;
    float cutoff = detect_cutoff(an.mag.data(), an.n_frames, sr);
    fprintf(stderr, "[Spectral Lifter]   Cutoff: %.0f Hz\n", cutoff);

    // 2. Spectral denoise (in-place on STFT, then synthesize)
    if (p->denoise_strength > 0.0f) {
        spectral_denoise(an, p->denoise_strength, p->noise_floor);
        an.synthesize(samples, T);
    }

    // 3. HF extension (operates on time-domain, does its own STFT)
    if (p->hf_mix > 0.0f) {
        hf_extend(samples, T, sr, cutoff, p->hf_mix);
    }

    // 4. Transient shaping
    if (p->transient_boost > 0.0f) {
        transient_shape(samples, T, sr, p->transient_boost);
    }

    // 5. Multiband dynamics
    if (p->shimmer_reduction > 0.0f) {
        multiband_dynamics(samples, T, sr, p->shimmer_reduction);
    }
}

}  // namespace sl_detail

// ── Public API ───────────────────────────────────────────────────────────────

// Process planar stereo audio [L0..LN, R0..RN] in-place.
// T = samples per channel, sr = sample rate (48000).
// Returns 0 on success.
static int spectral_lifter_process(float * audio, int T, int sr,
                                    const SpectralLifterParams * params) {
    if (!audio || T <= 0 || !params) return -1;

    // Check if anything is enabled
    bool any = params->denoise_strength > 0.0f ||
               params->hf_mix > 0.0f ||
               params->transient_boost > 0.0f ||
               params->shimmer_reduction > 0.0f;
    if (!any) return 0;

    fprintf(stderr, "[Spectral Lifter] Processing %d samples @ %dHz "
            "(denoise=%.2f, floor=%.2f, hf=%.2f, transient=%.2f, shimmer=%.1fdB)\n",
            T, sr, params->denoise_strength, params->noise_floor,
            params->hf_mix, params->transient_boost, params->shimmer_reduction);

    // Left channel: audio[0..T-1]
    sl_detail::process_mono(audio, T, sr, params);

    // Right channel: audio[T..2T-1]
    sl_detail::process_mono(audio + T, T, sr, params);

    fprintf(stderr, "[Spectral Lifter] Done\n");
    return 0;
}
