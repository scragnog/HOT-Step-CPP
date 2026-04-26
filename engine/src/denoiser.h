#pragma once
// denoiser.h: post-VAE spectral gating denoiser (HOT-Step)
//
// Removes stationary noise (fuzz/fizz) from VAE-decoded audio using
// STFT-based spectral gating. Designed to replicate the behaviour of
// Samplitude's Denoiser with Buzz_50Hz_Bright_1 profile.
//
// Fully self-contained: includes a minimal radix-2 Cooley-Tukey FFT
// so there are zero external dependencies and zero upstream file changes.
//
// Usage:
//   audio_denoise(planar, n_samples, 48000, strength, smoothing, mix);
//
// Audio layout: planar stereo [L0..LN, R0..RN], float32.
// Each channel is processed independently.

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
// Uses the standard real-FFT packing trick: interleave even/odd into a
// half-size complex FFT, then unpack with twiddle factors.
// For simplicity, we just do a full-size complex FFT with zero imaginary parts.
static void rfft(const float * in, Complex * out, int N) {
    std::vector<Complex> buf(N);
    for (int i = 0; i < N; i++) {
        buf[i] = Complex(in[i], 0.0f);
    }
    fft_inplace(buf.data(), N, false);
    // Copy first N/2+1 bins (Hermitian symmetry: we only need the positive half)
    for (int i = 0; i <= N / 2; i++) {
        out[i] = buf[i];
    }
}

// Real-valued inverse FFT: N/2+1 complex bins → N real samples.
static void irfft(const Complex * in, float * out, int N) {
    std::vector<Complex> buf(N);
    // Fill positive frequencies
    for (int i = 0; i <= N / 2; i++) {
        buf[i] = in[i];
    }
    // Hermitian mirror for negative frequencies
    for (int i = 1; i < N / 2; i++) {
        buf[N - i] = Complex(in[i].re, -in[i].im);
    }
    fft_inplace(buf.data(), N, true);
    for (int i = 0; i < N; i++) {
        out[i] = buf[i].re;
    }
}

// ── Spectral Gating Core ─────────────────────────────────────────────────────

// Denoise a single mono channel in-place.
// samples: float array of length T.
// sr: sample rate.
// strength: 0.0 (off) to 1.0 (aggressive). Controls noise threshold + over-subtraction.
// smoothing: 0.0 (sharp gate) to 1.0 (very smooth). Controls temporal + frequency mask smoothing.
// mix: 0.0 (all dry) to 1.0 (all denoised). Dry/wet blend.
static void denoise_mono(float * samples, int T, int sr,
                         float strength, float smoothing, float mix) {
    if (T <= 0 || strength <= 0.0f) return;

    // ── STFT parameters (matching Samplitude screenshot) ──
    const int N   = 8192;   // FFT size (Resolution setting)
    const int hop = N / 2;  // 50% overlap
    const int n_bins = N / 2 + 1;

    // ── Hann window ──
    std::vector<float> window(N);
    for (int i = 0; i < N; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * (float) i / (float) N));
    }

    // ── Compute all STFT frames ──
    int n_frames = 0;
    for (int start = 0; start + N <= T; start += hop) {
        n_frames++;
    }
    if (n_frames < 2) return;  // Too short to denoise

    // Allocate magnitude + phase storage
    std::vector<float>   magnitudes((size_t) n_frames * n_bins);
    std::vector<Complex> spectra((size_t) n_frames * n_bins);
    std::vector<float>   frame_buf(N);

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        // Apply window
        for (int i = 0; i < N; i++) {
            frame_buf[i] = (offset + i < T) ? samples[offset + i] * window[i] : 0.0f;
        }
        // Forward FFT
        rfft(frame_buf.data(), &spectra[(size_t) f * n_bins], N);
        // Compute magnitude
        for (int b = 0; b < n_bins; b++) {
            const Complex & c = spectra[(size_t) f * n_bins + b];
            magnitudes[(size_t) f * n_bins + b] = sqrtf(c.re * c.re + c.im * c.im);
        }
    }

    // ── Noise floor estimation ──
    // Per-bin statistics: mean and standard deviation across all frames.
    // Stationary noise (VAE fuzz) has consistent per-bin energy, so
    // mean + n_std * std gives a robust threshold.
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

    // ── Derive internal parameters from user controls ──
    // strength 0→1 maps to:
    //   n_std_factor: 6→0.5 (inverse — higher strength = lower threshold = more aggressive)
    //   over_sub: 1.0→6.0 (more over-subtraction at higher strength)
    float n_std_factor = 6.0f - 5.5f * strength;
    float over_sub     = 1.0f + 5.0f * strength;

    // smoothing 0→1 maps to:
    //   release_coeff: 0.5→0.99 (temporal IIR release smoothing)
    //   attack_coeff: always fast (0.3) — matches Samplitude Attack=1.0
    //   freq_sigma: 1→16 bins (frequency-domain Gaussian smoothing)
    float release_coeff = 0.5f + 0.49f * smoothing;
    float attack_coeff  = 0.3f;
    int   freq_sigma    = 1 + (int) (15.0f * smoothing);

    // ── Compute noise threshold per bin ──
    std::vector<float> threshold(n_bins);
    for (int b = 0; b < n_bins; b++) {
        threshold[b] = noise_mean[b] * over_sub + n_std_factor * noise_std[b];
    }

    // ── Build soft masks ──
    // For each frame and bin: mask = sigmoid((mag - threshold) / softness)
    // softness scales with the threshold to be relative
    std::vector<float> masks((size_t) n_frames * n_bins);
    const float softness_scale = 0.3f;  // Controls sigmoid steepness

    for (int f = 0; f < n_frames; f++) {
        for (int b = 0; b < n_bins; b++) {
            float mag = magnitudes[(size_t) f * n_bins + b];
            float thr = threshold[b];
            float softness = fmaxf(thr * softness_scale, 1e-8f);
            float x = (mag - thr) / softness;
            // Sigmoid: 1 / (1 + exp(-x))
            float s = 1.0f / (1.0f + expf(-x));
            masks[(size_t) f * n_bins + b] = s;
        }
    }

    // ── Temporal smoothing (IIR) ──
    // Forward pass: asymmetric attack/release (Samplitude: Attack=1.0, Release=10.0)
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
    // Backward pass (bidirectional smoothing for symmetry)
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

    // ── Clamp mask minimum (spectral floor) ──
    // Prevents "musical noise" (bubbly artifacts from hard gating)
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
    std::vector<float> win_sum(T, 0.0f);  // For COLA normalization

    for (int f = 0; f < n_frames; f++) {
        int offset = f * hop;
        // Inverse FFT
        irfft(&spectra[(size_t) f * n_bins], frame_buf.data(), N);
        // Apply window and accumulate
        for (int i = 0; i < N && offset + i < T; i++) {
            output[offset + i]  += frame_buf[i] * window[i];
            win_sum[offset + i] += window[i] * window[i];
        }
    }

    // Normalize by window sum (COLA condition for Hann + 50% overlap ≈ 1.0)
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

// Denoise planar stereo audio [L0..LN, R0..RN] in-place.
// T = samples per channel, sr = sample rate (48000).
// strength: 0.0 = off, 1.0 = maximum suppression.
// smoothing: 0.0 = sharp gate, 1.0 = very smooth transition.
// mix: 0.0 = all dry (original), 1.0 = all denoised.
// Returns 0 on success.
static int audio_denoise(float * audio, int T, int sr,
                         float strength, float smoothing, float mix) {
    if (!audio || T <= 0 || strength <= 0.0f) return 0;

    // Clamp parameters
    strength  = fminf(fmaxf(strength, 0.0f), 1.0f);
    smoothing = fminf(fmaxf(smoothing, 0.0f), 1.0f);
    mix       = fminf(fmaxf(mix, 0.0f), 1.0f);

    fprintf(stderr, "[Denoiser] Processing %d samples @ %dHz (strength=%.2f, smoothing=%.2f, mix=%.2f)\n",
            T, sr, strength, smoothing, mix);

    // Process left channel: audio[0..T-1]
    denoiser_detail::denoise_mono(audio, T, sr, strength, smoothing, mix);

    // Process right channel: audio[T..2T-1]
    denoiser_detail::denoise_mono(audio + T, T, sr, strength, smoothing, mix);

    fprintf(stderr, "[Denoiser] Done\n");
    return 0;
}
