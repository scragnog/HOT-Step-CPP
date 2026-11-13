#pragma once
// yourmt3/features.h: YourMT3+ input features, as MuLaCover's melody
// transcriber prepares them. Mono float32 audio (channel mean) resampled to
// 16 kHz with torchaudio.functional.resample, zero-padded to whole 32,767-sample
// segments, then torchaudio's magnitude spectrogram (n_fft 2048, hop 300, Hann,
// centred with reflect padding) without the DC bin, log(x + 1e-5) and a min-max
// normalisation per segment. Output [segment][110 frames][1024 bins].
//
// The resampler is torchaudio's Hann-windowed sinc (lowpass width 6, rolloff
// 0.99) on the gcd-reduced ratio, with the kernel computed in float32 as
// torchaudio does for a float32 waveform. It is deterministic and has no
// rate-dependent stages, so every integer source rate takes the exact path.
// The FFT runs in double; torch's float32 STFT is the only error source left.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "chordnet/features.h"  // chordnet::Fft (double radix-2)

namespace yourmt3 {

constexpr int SR = 16000, SEG = 32767, NFFT = 2048, HOP = 300, NFRAME = 1 + SEG / HOP, NBIN = NFFT / 2;
constexpr int BATCH = 8;  // segments per prediction batch upstream

// torchaudio.functional.resample(waveform, sr_in, sr_out), default arguments.
static std::vector<float> resample(const std::vector<float> & x, int sr_in, int sr_out) {
    if (sr_in == sr_out) return x;
    const int g = std::gcd(sr_in, sr_out), orig = sr_in / g, nw = sr_out / g;
    const double base = std::min(orig, nw) * 0.99;
    const int width = (int) std::ceil(6.0 * orig / base), K = 2 * width + orig;
    const float fbase = (float) base, fpi = (float) M_PI, fscale = (float) (base / orig);
    std::vector<float> k((size_t) nw * K);
    for (int j = 0; j < nw; j++)
        for (int i = 0; i < K; i++) {
            float t = (float) (-j) / (float) nw + (float) (i - width) / (float) orig;
            t *= fbase;
            t = std::min(6.0f, std::max(-6.0f, t));
            float w = std::cos(t * fpi / 6.0f / 2.0f);
            w *= w;
            t *= fpi;
            const float s = t == 0.0f ? 1.0f : std::sin(t) / t;
            k[(size_t) j * K + i] = s * (w * fscale);
        }
    const long long n = (long long) x.size();
    const long long n_out = (long long) std::ceil((double) nw * (double) n / (double) orig);
    std::vector<float> y((size_t) n_out);
    // conv1d over [width zeros, x, width + orig zeros] with stride orig
    for (long long o = 0; o < n_out; o++) {
        const long long b = o / nw, j = o % nw, s0 = b * orig - width;
        const float * kj = &k[(size_t) j * K];
        double acc = 0.0;
        const long long i0 = std::max(0LL, -s0), i1 = std::min((long long) K, n - s0);
        for (long long i = i0; i < i1; i++) acc += (double) x[(size_t) (s0 + i)] * kj[i];
        y[(size_t) o] = (float) acc;
    }
    return y;
}

// Zero-padded to whole segments: [n_seg][SEG].
static std::vector<float> pad_segments(const std::vector<float> & wav16k, int * n_seg) {
    const size_t n = wav16k.size(), pad = (SEG - n % SEG) % SEG;
    std::vector<float> out(wav16k);
    out.resize(n + pad, 0.0f);
    *n_seg = (int) (out.size() / SEG);
    return out;
}

// Spectrogram of n_seg segments laid out as pad_segments returns them, with
// torch's periodic Hann window as stored in the checkpoint. [n_seg][NFRAME][NBIN].
static std::vector<float> spectrogram(const float * seg, int n_seg, const float * window) {
    static const chordnet::Fft fft(NFFT);
    std::vector<float> out((size_t) n_seg * NFRAME * NBIN);
    std::vector<chordnet::cd> a(NFFT);
    std::vector<float> padded(SEG + NFFT);
    for (int b = 0; b < n_seg; b++) {
        const float * x = seg + (size_t) b * SEG;
        for (int i = 0; i < SEG + NFFT; i++) {
            int s = i - NFFT / 2;
            if (s < 0) s = -s;
            if (s >= SEG) s = 2 * (SEG - 1) - s;
            padded[i] = x[s];
        }
        float * o = &out[(size_t) b * NFRAME * NBIN];
        for (int f = 0; f < NFRAME; f++) {
            for (int i = 0; i < NFFT; i++) a[i] = chordnet::cd((double) padded[f * HOP + i] * window[i], 0.0);
            fft.run(a.data());
            for (int k = 1; k <= NBIN; k++) o[(size_t) f * NBIN + k - 1] = std::log((float) std::abs(a[k]) + 1e-5f);
        }
        // minmax_normalize: (x - min) / (max - min + 0.008), then nan_to_num
        const auto mm = std::minmax_element(o, o + (size_t) NFRAME * NBIN);
        const float lo = *mm.first, den = *mm.second - lo + 0.008f;
        for (size_t i = 0; i < (size_t) NFRAME * NBIN; i++) {
            const float v = (o[i] - lo) / den;
            o[i] = std::isfinite(v) ? v : 0.0f;
        }
    }
    return out;
}

}  // namespace yourmt3
