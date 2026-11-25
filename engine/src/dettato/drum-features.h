#pragma once
// Dettato 2.0 drum features, ported from KobiMusic's Apache-2.0 implementation.

#include <cstdio>
#include "../audio-resample.h"
#include "ggml.h"
#include "../../vendor/pocketfft/pocketfft_hdronly.h"

#include <algorithm>
#include <complex>
#include <vector>

namespace dettato::drums {

inline constexpr int sample_rate = 44100;
inline constexpr int min_source_rate = 40000;
inline constexpr int feature_bands = 280;
inline constexpr int mel_bands = 160;
inline constexpr int cq_bands = 120;

// Linear-phase HQ response fitted to SoXR impulse measurements in units of the
// lower rate's samples. Zero extension and double-precision positions matter
// at the song edges and on long inputs respectively.
inline std::vector<float> resample(const float * input, size_t n, int from, int to) {
    if (from == to) return {input, input + n};
    constexpr double cutoff = 0.956841;
    constexpr double radius = 110.0;
    constexpr double beta = 15.97;
    constexpr int phases = 512;
    const double ratio = double(to) / from;
    const double scale = std::min(1.0, ratio);
    const int half = int(std::ceil(radius / scale));
    const int taps = 2 * half + 1;
    const double gain = scale / audio_resample_bessel_i0(beta);
    std::vector<float> table(size_t(phases + 1) * taps);
    for (int p = 0; p <= phases; ++p) {
        for (int j = -half; j <= half; ++j) {
            const double d = (j - double(p) / phases) * scale;
            if (std::abs(d) >= radius) continue;
            const double sinc = d == 0 ? cutoff : std::sin(M_PI * cutoff * d) / (M_PI * d);
            const double t = d / radius;
            table[size_t(p) * taps + j + half] =
                float(gain * sinc * audio_resample_bessel_i0(beta * std::sqrt(1 - t * t)));
        }
    }
    std::vector<float> out(size_t(std::llround(n * ratio)));
    for (size_t i = 0; i < out.size(); ++i) {
        const double center = double(i) * from / to;
        const auto base = int64_t(center);
        const double phase = (center - base) * phases;
        const int p = std::min(int(phase), phases - 1);
        const float mix = float(phase - p);
        const float * a = table.data() + size_t(p) * taps + half;
        const float * b = a + taps;
        const int first = int(std::max<int64_t>(-half, -base));
        const int last = int(std::min<int64_t>(half, int64_t(n) - base - 1));
        float sum = 0;
        for (int j = first; j <= last; ++j) {
            sum += input[base + j] * (a[j] + mix * (b[j] - a[j]));
        }
        out[i] = sum;
    }
    return out;
}

struct Filter {
    int first = 0;
    std::vector<float> weights;
};

inline std::vector<Filter> filterbank(int rate, int fft, int bands, double low, double high, bool cq) {
    std::vector<double> centers(bands + 2);
    const int bins = fft / 2 + 1;
    const double lo_mel = std::log1p(low / 700.0);
    const double hi_mel = std::log1p(high / 700.0);
    for (int i = 0; i < bands + 2; ++i) {
        centers[i] = cq ? low * std::exp2((i - 1) / 24.0)
                        : 700.0 * std::expm1(lo_mel + i * (hi_mel - lo_mel) / (bands + 1));
    }
    std::vector<Filter> bank(bands);
    // Mel uses linspace up to Nyquist even for upstream's odd-sized high FFT.
    const double step = cq ? double(rate) / fft : double(rate) / (2 * (bins - 1));
    for (int k = 0; k < bands; ++k) {
        const double lo = centers[k], ce = centers[k + 1], hi = centers[k + 2];
        auto & filter = bank[k];
        filter.first = int(std::floor(lo / step));
        const int end = std::min(bins - 1, int(std::ceil(hi / step)));
        double sum = 0.0;
        for (int j = filter.first; j <= end; ++j) {
            const double f = j * step;
            const double w = std::max(0.0, std::min((f - lo) / (ce - lo), (hi - f) / (hi - ce)));
            sum += w;
            filter.weights.push_back(float(w));
        }
        if (cq && sum < 1.0) {
            const double j = ce / step;
            filter.first = int(j);
            filter.weights = {float(1 - (j - filter.first)), float(j - filter.first)};
        } else {
            const float scale = float(cq ? 1.0 / sum : 2.0 / (hi - lo));
            for (float & w : filter.weights) w *= scale;
        }
    }
    return bank;
}

inline void logspec(const std::vector<float> & x, int fft, int hop, const std::vector<Filter> & bank,
                    int frames, int band_offset, int threads, float * out) {
    constexpr int block = 96;
    const int bins = fft / 2 + 1;
    std::vector<float> hann(fft), input(size_t(block) * fft), power(bins);
    std::vector<std::complex<float>> spectrum(size_t(block) * bins);
    for (int i = 0; i < fft; ++i) hann[i] = 0.5f - 0.5f * std::cos(float(2 * M_PI) * i / fft);
    const pocketfft::stride_t src{ptrdiff_t(fft * sizeof(float)), sizeof(float)};
    const pocketfft::stride_t dst{ptrdiff_t(bins * sizeof(std::complex<float>)), sizeof(std::complex<float>)};
    for (int start = 0; start < frames; start += block) {
        const int count = std::min(block, frames - start);
        for (int t = 0; t < count; ++t) {
            const int64_t first = int64_t(start + t) * hop - fft / 2;
            for (int j = 0; j < fft; ++j) {
                const int64_t k = first + j;
                input[size_t(t) * fft + j] = k >= 0 && size_t(k) < x.size() ? x[k] * hann[j] : 0.0f;
            }
        }
        pocketfft::r2c({size_t(count), size_t(fft)}, src, dst, size_t(1), pocketfft::FORWARD,
                       input.data(), spectrum.data(), 1.0f, size_t(threads));
        for (int t = 0; t < count; ++t) {
            const auto * s = spectrum.data() + size_t(t) * bins;
            for (int j = 0; j < bins; ++j) power[j] = std::norm(s[j]);
            float * row = out + size_t(start + t) * feature_bands + band_offset;
            for (size_t k = 0; k < bank.size(); ++k) {
                const auto & f = bank[k];
                float energy = 0.0f;
                for (size_t j = 0; j < f.weights.size(); ++j) energy += f.weights[j] * power[f.first + j];
                row[k] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(std::log(energy + 1e-5f)));
            }
        }
    }
}

// Mono source samples -> [100 fps frames, 280], rounded through FP16.
inline std::vector<float> features(const float * mono, size_t n, int source_rate, int threads = 4) {
    if (!mono || n == 0 || source_rate < min_source_rate) return {};
    auto x44 = resample(mono, n, source_rate, sample_rate);
    if (x44.size() < sample_rate / 10) return {};
    double squared = 0.0;
    for (float x : x44) squared += double(x) * x;
    const float gain = float(std::pow(10.0, -18.0 / 20) / (std::sqrt(squared / x44.size()) + 1e-9));
    for (float & x : x44) x *= gain;
    const auto x16 = resample(x44.data(), x44.size(), sample_rate, 16000);
    const int frames = int(std::min(x16.size() / 160, (x44.size() - 1) / 441 + 1));
    std::vector<float> out(size_t(frames) * feature_bands);
    static const auto lo = filterbank(16000, 512, 128, 20.0, 7800.0, false);
    static const auto hi = filterbank(sample_rate, 1411, 32, 7800.0, 20000.0, false);
    static const auto cq = filterbank(16000, 6144, cq_bands, 36.708, 0.0, true);
    threads = std::max(1, threads);
    logspec(x16, 512, 160, lo, frames, 0, threads, out.data());
    logspec(x44, 1411, 441, hi, frames, 128, threads, out.data());
    logspec(x16, 6144, 160, cq, frames, mel_bands, threads, out.data());
    return out;
}

}  // namespace dettato::drums
