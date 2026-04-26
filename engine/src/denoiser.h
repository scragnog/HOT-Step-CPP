#pragma once
// denoiser.h: post-VAE spectral gating denoiser (HOT-Step)
//
// Removes stationary noise (fuzz/fizz) from VAE-decoded audio using
// STFT-based spectral gating with a pre-computed noise profile.
//
// Two modes:
//   1. Profile-based (recommended): uses a reference noise sample to build
//      an exact spectral fingerprint of the VAE noise. Call
//      audio_denoise_compute_profile() once at startup, then pass the
//      profile to audio_denoise().
//   2. Self-estimation (fallback): if no profile is provided, estimates
//      the noise floor from the audio itself (less accurate).
//
// Fully self-contained: includes a minimal radix-2 Cooley-Tukey FFT
// so there are zero external dependencies and zero upstream file changes.
//
// Usage:
//   // One-time: compute profile from noise sample
//   NoiseProfile profile;
//   audio_denoise_compute_profile(noise_samples, n_samples, 48000, &profile);
//
//   // Per-generation: denoise with profile
//   audio_denoise(planar, n_samples, 48000, strength, smoothing, mix, &profile);
//
// Audio layout: planar stereo [L0..LN, R0..RN], float32.
// Each channel is processed independently.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── Minimal Radix-2 FFT ─────────────────────────────────────────────────────
// Complex-valued in-place Cooley-Tukey, power-of-2 only.
// Only used internally by the denoiser — not exposed as public API.

namespace denoiser_detail {

struct Complex {
    float re, im;
    Complex() : re(0), im(0) {}
    Complex(float r, float i) : re(r), im(i) {}
    Complex operator+(const Complex & o) const { return {re + o.re, im + o.im}; }
    Complex operator-(const Complex & o) const { return {re - o.re, im - o.im}; }
    Complex operator*(const Complex & o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
};

// Bit-reversal permutation
static void bit_reverse(Complex * x, int N) {
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            Complex tmp = x[i];
            x[i]        = x[j];
            x[j]        = tmp;
        }
    }
}

// In-place complex FFT. inverse=false → forward, inverse=true → inverse.
// Forward: no scaling. Inverse: divides by N.
static void fft_inplace(Complex * x, int N, bool inverse) {
    bit_reverse(x, N);

    for (int len = 2; len <= N; len <<= 1) {
        float ang = 2.0f * 3.14159265358979323846f / (float) len * (inverse ? -1.0f : 1.0f);
        Complex wlen(cosf(ang), sinf(ang));

        for (int i = 0; i < N; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; j++) {
                Complex u = x[i + j];
                Complex v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w = w * wlen;
            }
        }
    }

    if (inverse) {
        float inv_n = 1.0f / (float) N;
        for (int i = 0; i < N; i++) {
            x[i].re *= inv_n;
            x[i].im *= inv_n;
        }
    }
}

// Real-valued forward FFT: N real samples → N/2+1 complex bins.
static void rfft(const float * in, Complex * out, int N) {
    std::vector<Complex> buf(N);
    for (int i = 0; i < N; i++) {
        buf[i] = Complex(in[i], 0.0f);
    }
    fft_inplace(buf.data(), N, false);
    for (int i = 0; i <= N / 2; i++) {
        out[i] = buf[i];
    }
}

// Real-valued inverse FFT: N/2+1 complex bins → N real samples.
static void irfft(const Complex * in, float * out, int N) {
    std::vector<Complex> buf(N);
    for (int i = 0; i <= N / 2; i++) {
        buf[i] = in[i];
    }
    for (int i = 1; i < N / 2; i++) {
        buf[N - i] = Complex(in[i].re, -in[i].im);
    }
    fft_inplace(buf.data(), N, true);
    for (int i = 0; i < N; i++) {
        out[i] = buf[i].re;
    }
}

// ── STFT parameters (shared by profile computation and denoising) ────────────
static const int DENOISE_FFT_SIZE = 8192;
static const int DENOISE_HOP      = DENOISE_FFT_SIZE / 2;
static const int DENOISE_N_BINS   = DENOISE_FFT_SIZE / 2 + 1;

// Build Hann window
static void build_hann(float * window, int N) {
    for (int i = 0; i < N; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * (float) i / (float) N));
    }
}

// ── Noise Profile ────────────────────────────────────────────────────────────

// Per-bin median magnitude from a reference noise sample.
// Computed once, used for all subsequent denoise calls.
struct NoiseProfile {
    std::vector<float> median_mag;   // median magnitude per FFT bin (DENOISE_N_BINS values)
    int                sample_rate;
    int                n_frames;     // how many STFT frames contributed
    bool               valid;

    NoiseProfile() : sample_rate(0), n_frames(0), valid(false) {}
};

// Compute median of a float vector (modifies input — sorts in place)
static float median_inplace(std::vector<float> & v) {
    if (v.empty()) return 0.0f;
    size_t n = v.size();
    size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (n % 2 == 0) {
        float a = v[mid];
        std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
        return (a + v[mid - 1]) * 0.5f;
    }
    return v[mid];
}

// ── Spectral Gating Core ─────────────────────────────────────────────────────

// Denoise a single mono channel in-place.
// If profile is non-null and valid, uses it for thresholding (recommended).
// Otherwise falls back to self-estimation.
static void denoise_mono(float * samples, int T, int sr,
                         float strength, float smoothing, float mix,
                         const NoiseProfile * profile) {
    if (T <= 0 || strength <= 0.0f) return;

    const int N      = DENOISE_FFT_SIZE;
    const int hop    = DENOISE_HOP;
    const int n_bins = DENOISE_N_BINS;

    // ── Hann window ──
    std::vector<float> window(N);
    build_hann(window.data(), N);

    // ── Compute all STFT frames ──
    int n_frames = 0;
    for (int start = 0; start + N <= T; start += hop) {
        n_frames++;
    }
    if (n_frames < 2) return;

    // Allocate magnitude + spectra storage
    std::vector<float>   magnitudes((size_t) n_frames * n_bins);
    std::vector<Complex> spectra((size_t) n_frames * n_bins);
    std::vector<float>   frame_buf(N);

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        for (int i = 0; i < N; i++) {
            frame_buf[i] = (offset + i < T) ? samples[offset + i] * window[i] : 0.0f;
        }
        rfft(frame_buf.data(), &spectra[(size_t) f * n_bins], N);
        for (int b = 0; b < n_bins; b++) {
            const Complex & c = spectra[(size_t) f * n_bins + b];
            magnitudes[(size_t) f * n_bins + b] = sqrtf(c.re * c.re + c.im * c.im);
        }
    }

    // ── Derive internal parameters from user controls ──
    // strength 0→1 maps to over-subtraction factor
    float over_sub = 0.5f + 3.5f * strength;

    // smoothing 0→1 maps to temporal + frequency smoothing
    float release_coeff = 0.5f + 0.49f * smoothing;
    float attack_coeff  = 0.3f;
    int   freq_sigma    = 1 + (int) (15.0f * smoothing);

    // ── Build per-bin threshold ──
    std::vector<float> threshold(n_bins);

    if (profile && profile->valid && (int) profile->median_mag.size() == n_bins) {
        // ── Profile-based threshold (recommended) ──
        // The profile gives us the exact spectral shape of the VAE noise.
        // Scale it by over_sub to control aggressiveness.
        for (int b = 0; b < n_bins; b++) {
            threshold[b] = profile->median_mag[b] * over_sub;
        }
        // fprintf(stderr, "[Denoiser] Using noise profile (%d frames, over_sub=%.2f)\n",
        //         profile->n_frames, over_sub);
    } else {
        // ── Self-estimation fallback ──
        // Per-bin mean + std across all frames.
        std::vector<float> noise_mean(n_bins, 0.0f);
        std::vector<float> noise_std(n_bins, 0.0f);

        for (int b = 0; b < n_bins; b++) {
            float sum = 0.0f;
            for (int f = 0; f < n_frames; f++) {
                sum += magnitudes[(size_t) f * n_bins + b];
            }
            noise_mean[b] = sum / (float) n_frames;
        }
        for (int b = 0; b < n_bins; b++) {
            float sum_sq = 0.0f;
            for (int f = 0; f < n_frames; f++) {
                float diff = magnitudes[(size_t) f * n_bins + b] - noise_mean[b];
                sum_sq += diff * diff;
            }
            noise_std[b] = sqrtf(sum_sq / (float) n_frames);
        }

        float n_std_factor = 6.0f - 5.5f * strength;
        for (int b = 0; b < n_bins; b++) {
            threshold[b] = noise_mean[b] * over_sub + n_std_factor * noise_std[b];
        }
    }

    // ── Build soft masks ──
    std::vector<float> masks((size_t) n_frames * n_bins);
    const float softness_scale = 0.3f;

    for (int f = 0; f < n_frames; f++) {
        for (int b = 0; b < n_bins; b++) {
            float mag = magnitudes[(size_t) f * n_bins + b];
            float thr = threshold[b];
            float softness = fmaxf(thr * softness_scale, 1e-8f);
            float x = (mag - thr) / softness;
            float s = 1.0f / (1.0f + expf(-x));
            masks[(size_t) f * n_bins + b] = s;
        }
    }

    // ── Temporal smoothing (IIR, bidirectional) ──
    for (int b = 0; b < n_bins; b++) {
        float prev = masks[b];
        for (int f = 1; f < n_frames; f++) {
            float curr  = masks[(size_t) f * n_bins + b];
            float coeff = (curr > prev) ? attack_coeff : release_coeff;
            float smoothed = coeff * prev + (1.0f - coeff) * curr;
            masks[(size_t) f * n_bins + b] = smoothed;
            prev = smoothed;
        }
    }
    for (int b = 0; b < n_bins; b++) {
        float prev = masks[(size_t) (n_frames - 1) * n_bins + b];
        for (int f = n_frames - 2; f >= 0; f--) {
            float curr     = masks[(size_t) f * n_bins + b];
            float smoothed = 0.5f * curr + 0.5f * (release_coeff * prev + (1.0f - release_coeff) * curr);
            masks[(size_t) f * n_bins + b] = smoothed;
            prev = smoothed;
        }
    }

    // ── Frequency smoothing (1D Gaussian blur across bins) ──
    if (freq_sigma > 1) {
        std::vector<float> temp(n_bins);
        for (int f = 0; f < n_frames; f++) {
            float * row = &masks[(size_t) f * n_bins];
            memcpy(temp.data(), row, (size_t) n_bins * sizeof(float));
            for (int b = 0; b < n_bins; b++) {
                float sum = 0.0f;
                float wt  = 0.0f;
                for (int k = -freq_sigma; k <= freq_sigma; k++) {
                    int idx = b + k;
                    if (idx < 0 || idx >= n_bins) continue;
                    float g = expf(-0.5f * (float) (k * k) / (float) (freq_sigma * freq_sigma));
                    sum += temp[idx] * g;
                    wt += g;
                }
                row[b] = sum / wt;
            }
        }
    }

    // ── Clamp mask minimum (spectral floor, prevents musical noise) ──
    const float mask_floor = 0.01f;
    for (size_t i = 0; i < (size_t) n_frames * n_bins; i++) {
        if (masks[i] < mask_floor) masks[i] = mask_floor;
    }

    // ── Apply masks to spectra ──
    for (int f = 0; f < n_frames; f++) {
        for (int b = 0; b < n_bins; b++) {
            float m = masks[(size_t) f * n_bins + b];
            spectra[(size_t) f * n_bins + b].re *= m;
            spectra[(size_t) f * n_bins + b].im *= m;
        }
    }

    // ── ISTFT with overlap-add ──
    std::vector<float> output(T, 0.0f);
    std::vector<float> win_sum(T, 0.0f);

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        irfft(&spectra[(size_t) f * n_bins], frame_buf.data(), N);
        for (int i = 0; i < N && offset + i < T; i++) {
            output[offset + i]  += frame_buf[i] * window[i];
            win_sum[offset + i] += window[i] * window[i];
        }
    }

    for (int i = 0; i < T; i++) {
        if (win_sum[i] > 1e-8f) {
            output[i] /= win_sum[i];
        }
    }

    // ── Dry/wet mix ──
    for (int i = 0; i < T; i++) {
        samples[i] = (1.0f - mix) * samples[i] + mix * output[i];
    }
}

}  // namespace denoiser_detail

// ── Public API ───────────────────────────────────────────────────────────────

// Pre-computed noise profile for spectral gating.
using NoiseProfile = denoiser_detail::NoiseProfile;

// Compute a noise profile from a mono audio sample.
// The input should be the "noise only" signal (e.g. inverse output from a
// professional denoiser, or a quiet section with only VAE fuzz).
// T = samples per channel, sr = sample rate.
// Returns 0 on success, -1 on error.
static int audio_denoise_compute_profile(const float * mono_samples, int T, int sr,
                                          NoiseProfile * out) {
    if (!mono_samples || T <= 0 || !out) return -1;

    const int N      = denoiser_detail::DENOISE_FFT_SIZE;
    const int hop    = denoiser_detail::DENOISE_HOP;
    const int n_bins = denoiser_detail::DENOISE_N_BINS;

    // Build Hann window
    std::vector<float> window(N);
    denoiser_detail::build_hann(window.data(), N);

    // Count frames
    int n_frames = 0;
    for (int start = 0; start + N <= T; start += hop) {
        n_frames++;
    }
    if (n_frames < 2) {
        fprintf(stderr, "[Denoiser] Profile: sample too short (%d samples, need >= %d)\n", T, N + hop);
        return -1;
    }

    // Compute magnitudes for all frames
    // Storage: per_bin_mags[bin][frame] for efficient median computation
    std::vector<std::vector<float>> per_bin_mags(n_bins, std::vector<float>(n_frames));
    std::vector<float> frame_buf(N);
    std::vector<denoiser_detail::Complex> spectrum(n_bins);

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        for (int i = 0; i < N; i++) {
            frame_buf[i] = (offset + i < T) ? mono_samples[offset + i] * window[i] : 0.0f;
        }
        denoiser_detail::rfft(frame_buf.data(), spectrum.data(), N);
        for (int b = 0; b < n_bins; b++) {
            const auto & c = spectrum[b];
            per_bin_mags[b][f] = sqrtf(c.re * c.re + c.im * c.im);
        }
    }

    // Compute median per bin
    out->median_mag.resize(n_bins);
    for (int b = 0; b < n_bins; b++) {
        out->median_mag[b] = denoiser_detail::median_inplace(per_bin_mags[b]);
    }

    out->sample_rate = sr;
    out->n_frames    = n_frames;
    out->valid       = true;

    // Log some stats
    float total_energy = 0.0f;
    for (int b = 0; b < n_bins; b++) {
        total_energy += out->median_mag[b];
    }
    float peak_mag = *std::max_element(out->median_mag.begin(), out->median_mag.end());
    int   peak_bin = (int) (std::max_element(out->median_mag.begin(), out->median_mag.end()) - out->median_mag.begin());
    float peak_freq = (float) peak_bin * (float) sr / (float) N;

    fprintf(stderr, "[Denoiser] Noise profile computed: %d frames, %d bins\n", n_frames, n_bins);
    fprintf(stderr, "[Denoiser]   Peak noise: %.1f Hz (bin %d, mag=%.4f)\n", peak_freq, peak_bin, peak_mag);
    fprintf(stderr, "[Denoiser]   Total energy: %.2f, avg per bin: %.6f\n",
            total_energy, total_energy / (float) n_bins);

    return 0;
}

// Denoise planar stereo audio [L0..LN, R0..RN] in-place.
// T = samples per channel, sr = sample rate (48000).
// strength: 0.0 = off, 1.0 = maximum suppression.
// smoothing: 0.0 = sharp gate, 1.0 = very smooth transition.
// mix: 0.0 = all dry (original), 1.0 = all denoised.
// profile: pre-computed noise profile (recommended), or NULL for self-estimation.
// Returns 0 on success.
static int audio_denoise(float * audio, int T, int sr,
                         float strength, float smoothing, float mix,
                         const NoiseProfile * profile = nullptr) {
    if (!audio || T <= 0 || strength <= 0.0f) return 0;

    // Clamp parameters
    strength  = fminf(fmaxf(strength, 0.0f), 1.0f);
    smoothing = fminf(fmaxf(smoothing, 0.0f), 1.0f);
    mix       = fminf(fmaxf(mix, 0.0f), 1.0f);

    const char * mode = (profile && profile->valid) ? "profile" : "self-estimate";
    fprintf(stderr, "[Denoiser] Processing %d samples @ %dHz (strength=%.2f, smoothing=%.2f, mix=%.2f, mode=%s)\n",
            T, sr, strength, smoothing, mix, mode);

    // Process left channel: audio[0..T-1]
    denoiser_detail::denoise_mono(audio, T, sr, strength, smoothing, mix, profile);

    // Process right channel: audio[T..2T-1]
    denoiser_detail::denoise_mono(audio + T, T, sr, strength, smoothing, mix, profile);

    fprintf(stderr, "[Denoiser] Done\n");
    return 0;
}
