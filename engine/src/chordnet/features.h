#pragma once
// chordnet/features.h: ChordNet input features. Mono audio at 22,050 Hz, the
// pitch-tuning estimate, the hybrid constant-Q transform (288 bins, 36 per
// octave from F#0, hop 512) and the 18:270 crop the network reads.
//
// The tuning estimate and the hybrid CQT are written from the algorithms of
// librosa 0.11 (ISC): estimate_tuning/piptrack, hybrid_cqt, pseudo_cqt and
// vqt with early downsampling. Where librosa computes in float32 (the tuning
// histogram), the same float32 steps are kept, because a bin edge decides it.
//
// librosa resamples with soxr's "HQ" recipe. Its 2:1 stage was measured from
// outside (impulses through soxr 1.1.0) and is a linear-phase Kaiser-windowed
// sinc: cutoff 0.47843592 of the lower rate, beta 13.04003601, support 194.5
// samples at the lower rate (fit residual 3e-8). The CQT's 2:1 and 4:1 steps
// and a 44.1 kHz source are therefore reproduced; soxr's start-up transient
// on 4:1 is not modelled and moves the CQT by under 1e-5. Other ratios use
// the same sinc as an approximation; see to_model_rate in inference.h.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <vector>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

namespace chordnet {

constexpr int SR = 22050, HOP = 512, BPO = 36, NBINS = 288, CROP0 = 18, CROP1 = 270, NFEAT = CROP1 - CROP0;

using cd = std::complex<double>;

// ---------------------------------------------------------------- FFT
struct Fft {
    int n = 0;
    std::vector<cd> tw;
    std::vector<int> rev;
    explicit Fft(int n_) : n(n_), tw(n_ / 2), rev(n_) {
        for (int i = 0; i < n / 2; i++) tw[i] = std::polar(1.0, -2.0 * M_PI * i / n);
        int lg = 0;
        while ((1 << lg) < n) lg++;
        for (int i = 0; i < n; i++) {
            int r = 0;
            for (int b = 0; b < lg; b++) r |= ((i >> b) & 1) << (lg - 1 - b);
            rev[i] = r;
        }
    }
    void run(cd * a) const {
        for (int i = 0; i < n; i++)
            if (i < rev[i]) std::swap(a[i], a[rev[i]]);
        for (int len = 2; len <= n; len <<= 1) {
            const int step = n / len;
            for (int i = 0; i < n; i += len)
                for (int j = 0; j < len / 2; j++) {
                    const cd u = a[i + j], v = a[i + j + len / 2] * tw[j * step];
                    a[i + j]           = u + v;
                    a[i + j + len / 2] = u - v;
                }
        }
    }
};

// ---------------------------------------------------------------- resampling
static double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * x * 0.25;
    for (int k = 1; k < 200; k++) {
        term *= y / ((double) k * k);
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

// soxr_hq response in units of the lower rate's samples.
static double soxr_hq_kernel(double t) {
    constexpr double FC = 0.47843592, BETA = 13.04003601, HALF = 194.5 / 2;
    if (std::fabs(t) > HALF) return 0.0;
    const double u = t / HALF;
    const double w = bessel_i0(BETA * std::sqrt(std::max(0.0, 1.0 - u * u))) / bessel_i0(BETA);
    const double x = 2.0 * FC * t;
    return 2.0 * FC * (x == 0.0 ? 1.0 : std::sin(M_PI * x) / (M_PI * x)) * w;
}

// librosa.resample(res_type="soxr_hq", fix=True): soxr returns round(n * r)
// samples, librosa pads or trims to ceil(n * r). scale divides by sqrt(r).
static std::vector<float> resample(const std::vector<float> & x, long long sr_in, long long sr_out, bool scale) {
    const long long g = std::gcd(sr_in, sr_out), L = sr_out / g, M = sr_in / g;
    const long long n = (long long) x.size();
    const double r = (double) sr_out / (double) sr_in;
    const long long n_out  = (long long) std::ceil((double) n * r);
    const long long n_soxr = (2 * n * L + M) / (2 * M);
    const bool   down  = L < M;
    const double unit  = (double) std::max(L, M);       // phase units per lower-rate sample
    const double gain  = down ? (double) L / M : 1.0;
    const int    K     = (int) std::ceil(194.5 / 2 * unit / L) + 1;  // input taps either side
    std::vector<double> tab((size_t) L * (2 * K + 1));
    for (long long p = 0; p < L; p++)
        for (int d = -K; d <= K; d++) tab[(size_t) p * (2 * K + 1) + d + K] = gain * soxr_hq_kernel((p + (double) d * L) / unit);
    std::vector<float> y((size_t) n_out, 0.0f);
    const double s = scale ? std::sqrt((double) M / L) : 1.0;
    for (long long m = 0; m < std::min(n_out, n_soxr); m++) {
        const long long pos = m * M, k0 = pos / L, p = pos % L;
        const double * t = &tab[(size_t) p * (2 * K + 1)];
        double acc = 0.0;
        for (int d = -K; d <= K; d++) {
            const long long k = k0 - d;
            if (k >= 0 && k < n) acc += t[d + K] * x[(size_t) k];
        }
        y[(size_t) m] = (float) (acc * s);
    }
    return y;
}

// ---------------------------------------------------------------- STFT
// Centred frames, zero padding, periodic Hann or rectangular window.
// fn(frame, spectrum[0..n_fft/2]) is called once per frame.
template <typename F>
static int stft_frames(const std::vector<float> & y, int n_fft, int hop, bool hann, F && fn) {
    const int n = (int) y.size(), frames = 1 + n / hop, pad = n_fft / 2;
    Fft fft(n_fft);
    std::vector<double> win(n_fft, 1.0);
    if (hann)
        for (int i = 0; i < n_fft; i++) win[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n_fft);
    std::vector<cd> buf(n_fft);
    for (int t = 0; t < frames; t++) {
        const int start = t * hop - pad;
        for (int i = 0; i < n_fft; i++) {
            const int j = start + i;
            buf[i] = cd(j >= 0 && j < n ? win[i] * y[j] : 0.0, 0.0);
        }
        fft.run(buf.data());
        fn(t, buf.data());
    }
    return frames;
}

// ---------------------------------------------------------------- tuning
// librosa.estimate_tuning(y, sr=22050, bins_per_octave=36): piptrack on a
// 2048-point magnitude STFT, keep peaks at or above the median magnitude,
// histogram their residuals in 0.01-bin steps. float32 as librosa runs it.
static double estimate_tuning(const std::vector<float> & y) {
    constexpr int NF = 2048, NB = NF / 2 + 1;
    std::vector<float> pitches, mags;
    std::vector<float> S(NB), avg(NB), shift(NB), z(NB);
    const double bin_hz = (double) SR / NF;
    stft_frames(y, NF, NF / 4, true, [&](int, const cd * X) {
        float mx = 0.0f;
        for (int k = 0; k < NB; k++) {
            S[k] = (float) std::abs(X[k]);
            mx   = std::max(mx, S[k]);
        }
        avg[0] = S[1] - S[0];
        avg[NB - 1] = S[NB - 1] - S[NB - 2];
        for (int k = 1; k < NB - 1; k++) avg[k] = (S[k + 1] - S[k - 1]) / 2.0f;
        shift[0] = shift[NB - 1] = 0.0f;
        for (int k = 1; k < NB - 1; k++) {
            // numba's stencil: float32 sums, but 2 * x and x / 2 promote to float64
            const double a = (double) (S[k + 1] + S[k - 1]) - 2.0 * (double) S[k];
            const double b = (double) (S[k + 1] - S[k - 1]) / 2.0;
            shift[k] = std::fabs(b) >= std::fabs(a) ? 0.0f : (float) (-b / a);
        }
        const float ref = 0.1f * mx;
        for (int k = 0; k < NB; k++) z[k] = S[k] > ref ? S[k] : 0.0f;
        for (int k = 1; k < NB; k++) {
            const bool lmax = k == NB - 1 ? z[k] > z[k - 1] : (z[k] > z[k - 1] && z[k] >= z[k + 1]);
            const double f = k * (1.0 / (NF * (1.0 / SR)));
            if (!lmax || !(150.0 <= f && f < 4000.0)) continue;
            const float pitch = (float) (((double) k + (double) shift[k]) * bin_hz);
            const float dskew = 0.5f * avg[k] * shift[k];
            if (pitch > 0.0f) {
                pitches.push_back(pitch);
                mags.push_back(S[k] + dskew);
            }
        }
    });
    if (pitches.empty()) return 0.0;
    std::vector<float> sorted = mags;
    const size_t h = sorted.size() / 2;
    std::nth_element(sorted.begin(), sorted.begin() + h, sorted.end());
    float thr = sorted[h];
    if (sorted.size() % 2 == 0) {
        const float lo = *std::max_element(sorted.begin(), sorted.begin() + h);
        thr = (lo + thr) / 2.0f;
    }
    int counts[100] = {};
    bool any = false;
    for (size_t i = 0; i < pitches.size(); i++) {
        if (!(mags[i] >= thr)) continue;
        const float octs = (float) std::log2((double) (float) ((double) pitches[i] / 27.5));
        float res = std::fmod((float) (36.0 * (double) octs), 1.0f);
        if (res >= 0.5f) res -= 1.0f;
        // edges are i * 0.01 - 0.5 in double; the last bin is closed
        int b = 0;
        while (b < 99 && (double) res >= (b + 1) * 0.01 + -0.5) b++;
        counts[b]++;
        any = true;
    }
    if (!any) return 0.0;
    const int best = (int) (std::max_element(counts, counts + 100) - counts);
    return best * 0.01 + -0.5;
}

// ---------------------------------------------------------------- CQT kernels
struct SparseRow {
    std::vector<int> k;
    std::vector<cd>  v;
};

// librosa filters._relative_bandwidth
static std::vector<double> relative_bandwidth(const std::vector<double> & f) {
    const size_t n = f.size();
    std::vector<double> lf(n), alpha(n);
    for (size_t i = 0; i < n; i++) lf[i] = std::log2(f[i]);
    for (size_t i = 0; i < n; i++) {
        const double bpo = i == 0 ? 1.0 / (lf[1] - lf[0]) : i == n - 1 ? 1.0 / (lf[n - 1] - lf[n - 2]) : 2.0 / (lf[i + 1] - lf[i - 1]);
        const double e = std::pow(2.0, 2.0 / bpo);
        alpha[i] = (e - 1.0) / (e + 1.0);
    }
    return alpha;
}

static std::vector<double> wavelet_lengths(const std::vector<double> & f, const std::vector<double> & alpha, double sr, double * cutoff) {
    constexpr double HANN_BW = 1.50018310546875;
    std::vector<double> len(f.size());
    double fc = 0.0;
    for (size_t i = 0; i < f.size(); i++) {
        const double Q = 1.0 / alpha[i];
        len[i] = Q * sr / f[i];
        fc = std::max(fc, f[i] * (1.0 + 0.5 * HANN_BW / Q));
    }
    if (cutoff) *cutoff = fc;
    return len;
}

static int pow2_ceil(double x) { return (int) std::pow(2.0, std::ceil(std::log2(x))); }

// librosa constantq.__vqt_filter_fft: Hann-windowed complex wavelets, L1
// normalised, centred in a power-of-two frame, FFT, then sparsify_rows(0.01).
static std::vector<SparseRow> vqt_filter_fft(const std::vector<double> & f, const std::vector<double> & alpha, double sr, int min_fft, int * n_fft_out) {
    const std::vector<double> len = wavelet_lengths(f, alpha, sr, nullptr);
    const int max_len = pow2_ceil(*std::max_element(len.begin(), len.end()));
    const int n_fft   = std::max(max_len, min_fft);
    Fft fft(n_fft);
    std::vector<SparseRow> rows(f.size());
    std::vector<cd> buf(n_fft);
    std::vector<double> mag(n_fft / 2 + 1), srt;
    for (size_t b = 0; b < f.size(); b++) {
        const double ilen = len[b];
        const long long a0 = (long long) std::floor(-ilen / 2), a1 = (long long) std::floor(ilen / 2);
        const int N = (int) (a1 - a0), lpad = (max_len - N) / 2;
        std::fill(buf.begin(), buf.end(), cd(0, 0));
        double l1 = 0.0;
        for (int j = 0; j < N; j++) {
            const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * j / N);
            const cd s = std::polar(1.0, (double) (a0 + j) * 2 * M_PI * f[b] / sr) * w;
            buf[lpad + j] = s;
            l1 += std::abs(s);
        }
        const double g = ilen / n_fft / l1;
        for (int j = 0; j < N; j++) buf[lpad + j] *= g;
        fft.run(buf.data());
        const int nb = n_fft / 2 + 1;
        double norm = 0.0;
        for (int k = 0; k < nb; k++) norm += (mag[k] = std::abs(buf[k]));
        srt.assign(mag.begin(), mag.begin() + nb);
        std::sort(srt.begin(), srt.end());
        double cum = 0.0;
        int idx = nb - 1;
        for (int k = 0; k < nb; k++) {
            cum += srt[k] / norm;
            if (!(cum < 0.01)) { idx = k; break; }
        }
        const double cut = srt[idx];
        for (int k = 0; k < nb; k++)
            if (mag[k] >= cut) { rows[b].k.push_back(k); rows[b].v.push_back(buf[k]); }
    }
    *n_fft_out = n_fft;
    return rows;
}

// ---------------------------------------------------------------- hybrid CQT
// librosa.hybrid_cqt(y, sr=22050, hop_length=512, fmin=F#0, n_bins=288,
// bins_per_octave=36, tuning). Returns |CQT| as [frames][288].
static std::vector<float> hybrid_cqt(const std::vector<float> & y, double tuning, int * frames_out) {
    const double fmin = 440.0 * std::pow(2.0, (18 - 69.0) / 12.0) * std::pow(2.0, tuning / BPO);
    std::vector<double> freqs(NBINS);
    for (int i = 0; i < NBINS; i++) freqs[i] = fmin * std::pow(2.0, (double) i / BPO);
    const std::vector<double> alpha = relative_bandwidth(freqs);
    const std::vector<double> lens  = wavelet_lengths(freqs, alpha, SR, nullptr);
    int n_pseudo = 0;
    for (double l : lens) n_pseudo += std::pow(2.0, std::ceil(std::log2(l))) < 2 * HOP;
    const int n_full = NBINS - n_pseudo;

    std::vector<std::vector<float>> parts;  // [bins][frames] blocks, top first
    std::vector<int> part_bins;

    if (n_pseudo > 0) {  // pseudo_cqt: |basis| . |STFT|, hop 512, Hann
        const double fmin_p = freqs[n_full];
        std::vector<double> fp(n_pseudo);
        for (int i = 0; i < n_pseudo; i++) fp[i] = fmin_p * std::pow(2.0, (double) i / BPO);
        int n_fft = 0;
        const std::vector<SparseRow> basis = vqt_filter_fft(fp, relative_bandwidth(fp), SR, 2 * pow2_ceil(HOP), &n_fft);
        const int nf = 1 + (int) y.size() / HOP;
        std::vector<float> C((size_t) n_pseudo * nf);
        const double sc = 1.0 / std::sqrt((double) n_fft);
        stft_frames(y, n_fft, HOP, true, [&](int t, const cd * X) {
            for (int b = 0; b < n_pseudo; b++) {
                double acc = 0.0;
                for (size_t j = 0; j < basis[b].k.size(); j++) acc += std::abs(basis[b].v[j]) * std::abs(X[basis[b].k[j]]);
                C[(size_t) b * nf + t] = (float) (acc * sc);
            }
        });
        parts.push_back(std::move(C));
        part_bins.push_back(n_pseudo);
    }

    if (n_full > 0) {  // vqt with early downsampling and per-octave halving
        const int n_oct = (n_full + BPO - 1) / BPO, n_filt = std::min(BPO, n_full);
        std::vector<double> fr(n_full);
        {
            std::vector<double> all;
            for (int o = 0; o < n_oct; o++)
                for (int k = 0; k < BPO; k++) all.push_back(std::pow(2.0, o) * std::pow(2.0, (double) k / BPO));
            all.resize(n_full);
            std::sort(all.begin(), all.end());
            for (int i = 0; i < n_full; i++) fr[i] = all[i] * fmin;
        }
        const std::vector<double> al = relative_bandwidth(fr);
        double cutoff = 0.0;
        wavelet_lengths(fr, al, SR, &cutoff);
        const double nyq = SR / 2.0;
        int two = 0;
        for (int h = HOP; h % 2 == 0; h /= 2) two++;
        const int c1 = std::max(0, (int) std::ceil(std::log2(nyq / cutoff)) - 1 - 1);
        const int c2 = std::max(0, two - n_oct + 1);
        const int ds = std::min(c1, c2);
        std::vector<float> my_y = y;
        double sr = SR;
        int hop = HOP;
        if (ds > 0) {
            const int fac = 1 << ds;
            hop /= fac;
            my_y = resample(my_y, fac, 1, true);
            sr /= fac;
        }
        double my_sr = sr;
        int my_hop = hop;
        std::vector<std::vector<cd>> oct_resp;
        std::vector<int> oct_bins, oct_frames;
        for (int i = 0; i < n_oct; i++) {
            const int hi = n_full - n_filt * i, lo = std::max(0, n_full - n_filt * (i + 1));
            std::vector<double> fo(fr.begin() + lo, fr.begin() + hi), ao(al.begin() + lo, al.begin() + hi);
            int n_fft = 0;
            std::vector<SparseRow> basis = vqt_filter_fft(fo, ao, my_sr, 0, &n_fft);
            const double g = std::sqrt(sr / my_sr);
            const int nb = hi - lo, nf = 1 + (int) my_y.size() / my_hop;
            std::vector<cd> R((size_t) nb * nf);
            stft_frames(my_y, n_fft, my_hop, false, [&](int t, const cd * X) {
                for (int b = 0; b < nb; b++) {
                    cd acc(0, 0);
                    for (size_t j = 0; j < basis[b].k.size(); j++) acc += basis[b].v[j] * X[basis[b].k[j]];
                    R[(size_t) b * nf + t] = acc * g;
                }
            });
            oct_resp.push_back(std::move(R));
            oct_bins.push_back(nb);
            oct_frames.push_back(nf);
            if (my_hop % 2 == 0 && i + 1 < n_oct) {  // the last halving feeds nothing
                my_hop /= 2;
                my_sr /= 2.0;
                my_y = resample(my_y, 2, 1, true);
            }
        }
        const std::vector<double> lsc = wavelet_lengths(fr, al, sr, nullptr);
        const int nf = *std::min_element(oct_frames.begin(), oct_frames.end());
        std::vector<float> V((size_t) n_full * nf);
        int end = n_full;
        for (size_t i = 0; i < oct_resp.size(); i++) {
            const int nb = oct_bins[i];
            for (int b = 0; b < nb; b++) {
                const int bin = end - nb + b;
                for (int t = 0; t < nf; t++)
                    V[(size_t) bin * nf + t] = (float) std::abs(oct_resp[i][(size_t) b * oct_frames[i] + t] / std::sqrt(lsc[bin]));
            }
            end -= nb;
        }
        parts.push_back(std::move(V));
        part_bins.push_back(n_full);
    }

    // __trim_stack: common frame count, pseudo bins on top
    std::vector<int> pf;
    for (size_t i = 0; i < parts.size(); i++) pf.push_back((int) (parts[i].size() / part_bins[i]));
    const int T = *std::min_element(pf.begin(), pf.end());
    std::vector<float> out((size_t) T * NBINS);
    int end = NBINS;
    for (size_t i = 0; i < parts.size(); i++) {
        for (int b = 0; b < part_bins[i]; b++)
            for (int t = 0; t < T; t++) out[(size_t) t * NBINS + end - part_bins[i] + b] = parts[i][(size_t) b * pf[i] + t];
        end -= part_bins[i];
    }
    *frames_out = T;
    return out;
}

// [T][288] magnitudes to the network input [T][252]
static std::vector<float> crop(const std::vector<float> & mag, int T) {
    std::vector<float> x((size_t) T * NFEAT);
    for (int t = 0; t < T; t++)
        std::copy(&mag[(size_t) t * NBINS + CROP0], &mag[(size_t) t * NBINS + CROP1], &x[(size_t) t * NFEAT]);
    return x;
}

}  // namespace chordnet
