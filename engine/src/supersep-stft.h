#pragma once
// supersep-stft.h: STFT/iSTFT for SuperSep stem separation.
//
// Uses pocketfft (header-only, already vendored for mastering.cpp).
// Operates on interleaved stereo float audio at 44100 Hz.
//
// STFT produces complex spectrogram: [channels][freq_bins][time_frames][2]
//   where [2] is (real, imag).
//
// iSTFT reconstructs audio from masked spectrogram via overlap-add.
//
// Part of HOT-Step CPP. MIT license.

#ifndef SUPERSEP_STFT_H
#define SUPERSEP_STFT_H

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// pocketfft: header-only FFT (already vendored at vendor/pocketfft/)
#ifdef _MSC_VER
#    pragma warning(push, 0)
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#    pragma GCC diagnostic ignored "-Wshadow"
#endif

#include "pocketfft_hdronly.h"

#ifdef _MSC_VER
#    pragma warning(pop)
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// ── STFT Parameters ─────────────────────────────────────────────────────

struct StftParams {
    int n_fft;       // FFT size (default: 2048)
    int hop_length;  // Hop size (default: 441 for 44100 Hz, ~10ms)
    int n_channels;  // 1 = mono, 2 = stereo
};

static inline StftParams stft_default_params() {
    return { 2048, 441, 2 };
}

// ── Complex Spectrogram ─────────────────────────────────────────────────

// Layout: [channel][freq_bin][time_frame][2]  (2 = real, imag)
// freq_bins = n_fft/2 + 1
struct ComplexSpec {
    float * data;        // Owned, malloc'd
    int     n_channels;
    int     n_freqs;     // n_fft / 2 + 1
    int     n_frames;    // time frames
    int     n_fft;
    int     hop_length;

    // Access element: spec[ch][f][t] → (real, imag)
    inline float * at(int ch, int f, int t) {
        return data + ((size_t)ch * n_freqs * n_frames + (size_t)f * n_frames + t) * 2;
    }
    inline const float * at(int ch, int f, int t) const {
        return data + ((size_t)ch * n_freqs * n_frames + (size_t)f * n_frames + t) * 2;
    }
};

// ── Hann Window ─────────────────────────────────────────────────────────

static inline std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n));
    }
    return w;
}

// ── Forward STFT ────────────────────────────────────────────────────────

// Compute STFT on interleaved stereo (or mono) audio.
// audio: interleaved samples [L0,R0,L1,R1,...] for stereo
// n_frames_audio: per-channel frame count
// Returns ComplexSpec with data owned by caller (free with stft_free).
static ComplexSpec stft_forward(
    const float *    audio,
    int              n_frames_audio,
    const StftParams & params
) {
    const int n_fft     = params.n_fft;
    const int hop       = params.hop_length;
    const int n_ch      = params.n_channels;
    const int n_freqs   = n_fft / 2 + 1;
    const int pad       = n_fft / 2;  // center padding
    const int padded_len = n_frames_audio + n_fft;  // with padding on both sides
    const int n_time    = (padded_len - n_fft) / hop + 1;

    auto window = hann_window(n_fft);

    // Allocate output: [n_ch][n_freqs][n_time][2]
    ComplexSpec spec;
    spec.n_channels = n_ch;
    spec.n_freqs    = n_freqs;
    spec.n_frames   = n_time;
    spec.n_fft      = n_fft;
    spec.hop_length = hop;
    spec.data       = (float *)calloc((size_t)n_ch * n_freqs * n_time * 2, sizeof(float));

    // pocketfft setup
    pocketfft::shape_t shape_fft = { (size_t)n_fft };
    pocketfft::stride_t stride_in  = { sizeof(double) };
    pocketfft::stride_t stride_out = { sizeof(std::complex<double>) };
    pocketfft::shape_t axes = { 0 };

    std::vector<double> fft_in(n_fft);
    std::vector<std::complex<double>> fft_out(n_freqs);

    for (int ch = 0; ch < n_ch; ch++) {
        for (int t = 0; t < n_time; t++) {
            int start = t * hop - pad;

            // Fill FFT input with windowed samples
            for (int i = 0; i < n_fft; i++) {
                int idx = start + i;
                float sample = 0.0f;
                if (idx >= 0 && idx < n_frames_audio) {
                    if (n_ch == 1) {
                        sample = audio[idx];
                    } else {
                        sample = audio[idx * n_ch + ch];
                    }
                }
                fft_in[i] = (double)(sample * window[i]);
            }

            // Forward FFT (real-to-complex)
            pocketfft::r2c(shape_fft, stride_in, stride_out, axes, pocketfft::FORWARD,
                           fft_in.data(), fft_out.data(), 1.0);

            // Store complex coefficients
            for (int f = 0; f < n_freqs; f++) {
                float * dst = spec.at(ch, f, t);
                dst[0] = (float)fft_out[f].real();
                dst[1] = (float)fft_out[f].imag();
            }
        }
    }

    return spec;
}

// ── Inverse STFT ────────────────────────────────────────────────────────

// Reconstruct audio from complex spectrogram via overlap-add.
// Returns interleaved stereo (or mono) audio. Caller must free().
// out_frames: receives per-channel frame count.
static float * stft_inverse(
    const ComplexSpec & spec,
    int                 target_length,  // desired output length (per channel)
    int *               out_frames
) {
    const int n_fft     = spec.n_fft;
    const int hop       = spec.hop_length;
    const int n_ch      = spec.n_channels;
    const int n_freqs   = spec.n_freqs;
    const int n_time    = spec.n_frames;
    const int pad       = n_fft / 2;

    auto window = hann_window(n_fft);

    // Compute window normalization (for overlap-add)
    int recon_len = (n_time - 1) * hop + n_fft;
    std::vector<double> win_sum(recon_len, 0.0);
    for (int t = 0; t < n_time; t++) {
        int offset = t * hop;
        for (int i = 0; i < n_fft; i++) {
            if (offset + i < recon_len) {
                win_sum[offset + i] += (double)(window[i] * window[i]);
            }
        }
    }

    // Output buffer: interleaved
    int out_len = target_length > 0 ? target_length : (recon_len - n_fft);
    *out_frames = out_len;
    float * output = (float *)calloc((size_t)out_len * n_ch, sizeof(float));

    // pocketfft setup for inverse
    pocketfft::shape_t shape_fft = { (size_t)n_fft };
    pocketfft::stride_t stride_in  = { sizeof(std::complex<double>) };
    pocketfft::stride_t stride_out = { sizeof(double) };
    pocketfft::shape_t axes = { 0 };

    std::vector<std::complex<double>> fft_in(n_freqs);
    std::vector<double> fft_out(n_fft);

    for (int ch = 0; ch < n_ch; ch++) {
        // Overlap-add buffer for this channel
        std::vector<double> recon(recon_len, 0.0);

        for (int t = 0; t < n_time; t++) {
            // Load complex coefficients
            for (int f = 0; f < n_freqs; f++) {
                const float * src = spec.at(ch, f, t);
                fft_in[f] = std::complex<double>(src[0], src[1]);
            }

            // Inverse FFT (complex-to-real, using c2r with hermitian input)
            pocketfft::c2r(shape_fft, stride_in, stride_out, axes, pocketfft::BACKWARD,
                           fft_in.data(), fft_out.data(), 1.0 / n_fft);

            // Windowed overlap-add
            int offset = t * hop;
            for (int i = 0; i < n_fft; i++) {
                if (offset + i < recon_len) {
                    recon[offset + i] += fft_out[i] * (double)window[i];
                }
            }
        }

        // Normalize by window sum and extract unpadded region
        for (int i = 0; i < out_len; i++) {
            int src_idx = i + pad;
            double val = 0.0;
            if (src_idx < recon_len && win_sum[src_idx] > 1e-8) {
                val = recon[src_idx] / win_sum[src_idx];
            }
            if (n_ch == 1) {
                output[i] = (float)val;
            } else {
                output[i * n_ch + ch] = (float)val;
            }
        }
    }

    return output;
}

// ── Utility ─────────────────────────────────────────────────────────────

static void stft_free(ComplexSpec * spec) {
    if (spec && spec->data) {
        free(spec->data);
        spec->data = nullptr;
    }
}

// Apply a soft mask to a complex spectrogram.
// mask: [n_channels][n_freqs][n_frames] float values in [0, 1]
// Multiplies both real and imaginary parts by the mask value.
static void stft_apply_mask(ComplexSpec * spec, const float * mask) {
    const int n_ch   = spec->n_channels;
    const int n_freq = spec->n_freqs;
    const int n_time = spec->n_frames;

    for (int ch = 0; ch < n_ch; ch++) {
        for (int f = 0; f < n_freq; f++) {
            for (int t = 0; t < n_time; t++) {
                size_t mask_idx = (size_t)ch * n_freq * n_time + (size_t)f * n_time + t;
                float m = mask[mask_idx];
                float * c = spec->at(ch, f, t);
                c[0] *= m;
                c[1] *= m;
            }
        }
    }
}

#endif // SUPERSEP_STFT_H
