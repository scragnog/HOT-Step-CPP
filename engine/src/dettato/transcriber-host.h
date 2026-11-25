#pragma once
// dettato/transcriber-host.h: the host-side parts of dettato's transcriber
// (transcriber/frontend.py, transcriber/core.py, transcriber/decode.py and
// grouper/retune.py of dettato 2.0.0; see LICENSE here):
//   - the harmonic log-frequency front end (HCQT) of one 8 s window;
//   - the window layout and the Hann-weighted overlap-add of the six maps and
//     of the float16 taps, in upstream's order and precision;
//   - the peak-picking note decode, with numpy's float32 steps kept where a
//     threshold reads them;
//   - the tuning estimate.
// Audio enters at 16 kHz; the caller resamples it, and applies the retune, with libsoxr
// (vendor/soxr), as upstream does through python-soxr.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "chordnet/features.h"  // chordnet::Fft (double radix-2)
#include "ggml.h"               // ggml_fp16_t conversions

namespace dettato::transcriber {

constexpr int    SR = 16000, HOP = 160, N_KEYS = 88, PITCH_LO = 21, BPS = 3, N_BINS = 288;
constexpr int    N_FFT = 2048, N_FFT_X = 512, N_HARM = 8, N_CH = 9;
constexpr int    WIN = 128000, T10 = WIN / HOP, FW = 400, STEP = 32000, SUB_STEP = 64000, PAD = WIN / 2, G_PAD = 200;
constexpr int    N_HEADS_OUT = 6;
constexpr double FRAME_S = 0.01 * 2;  // model.hop_out = 0.01 * t_stride
constexpr double LOG_EPS = 1e-5;
static const char * const HEAD_NAMES[N_HEADS_OUT] = { "onset", "frame", "offset", "onset2", "frame2", "offset2" };

// ---------------------------------------------------------------- front end
struct Band {
    int                lo = 0;
    std::vector<float> w;
};

// logfreq_filterbank: triangular filters, bps per semitone from fmin, each summing to 1 (float64, stored as float32).
static std::vector<Band> logfreq_filterbank(int n_bins, int n_fft, double fmin) {
    const int           nf = n_fft / 2 + 1;
    std::vector<double> f(nf);
    const double        step = (SR / 2.0) / (nf - 1);
    for (int i = 0; i < nf; i++) f[i] = i * step;
    f[nf - 1] = SR / 2.0;
    std::vector<Band> fb(n_bins);
    for (int i = 0; i < n_bins; i++) {
        const double ctr = fmin * std::pow(2.0, i / (12.0 * BPS));
        double       lo = ctr * std::pow(2.0, -1.0 / (12 * BPS)), hi = ctr * std::pow(2.0, 1.0 / (12 * BPS));
        const double width = std::max(hi - lo, 2.0 * SR / n_fft);
        lo = ctr - width / 2;
        hi = ctr + width / 2;
        std::vector<double> v(nf);
        double              sum = 0.0;
        int                 a = -1, b = -1;
        for (int k = 0; k < nf; k++) {
            const double up = (f[k] - lo) / std::max(ctr - lo, 1e-9), down = (hi - f[k]) / std::max(hi - ctr, 1e-9);
            v[k] = std::max(0.0, std::min(up, down));
            if (v[k] > 0) {
                if (a < 0) a = k;
                b = k;
            }
            sum += v[k];
        }
        if (sum > 0) {
            fb[i].lo = a;
            for (int k = a; k <= b; k++) fb[i].w.push_back((float) (v[k] / sum));
        }
    }
    return fb;
}

struct Front {
    std::vector<Band>  fb, fb_x;  // [N_BINS + hi - lo] on the long window, [N_BINS] on the short one
    std::vector<float> win, win_x;
    int                shifts[N_HARM], lo = 0;
};

static Front make_front() {
    static const double harm[N_HARM] = { 0.5, 1, 2, 3, 4, 5, 6, 7 };
    Front               F;
    int                 lo = 0, hi = 0;
    for (int i = 0; i < N_HARM; i++) {
        F.shifts[i] = (int) std::nearbyint(12 * BPS * std::log2(harm[i]));
        lo          = std::min(lo, F.shifts[i]);
        hi          = std::max(hi, F.shifts[i]);
    }
    F.lo   = lo;
    F.fb   = logfreq_filterbank(N_BINS + hi - lo, N_FFT, 27.5 * std::pow(2.0, lo / (12.0 * BPS)));
    F.fb_x = logfreq_filterbank(N_BINS, N_FFT_X, 27.5);
    for (int n : { N_FFT, N_FFT_X }) {  // torch.hann_window (periodic), float32
        std::vector<float> & w = n == N_FFT ? F.win : F.win_x;
        w.resize(n);
        for (int i = 0; i < n; i++) w[i] = (float) (0.5 - 0.5 * std::cos(2.0 * M_PI * i / n));
    }
    return F;
}

// Log band energies of one window of n samples, centred frames with zero padding (torch.stft center, pad_mode
// constant), frames 0..T-1. out[b * T + t] for the n_out bands; lin (optional) gets the band energies themselves.
static void log_bands(const float * x, int n_fft, const std::vector<float> & win, const std::vector<Band> & fb, float * out,
                      int n = WIN, int T = T10, float * lin = nullptr) {
    chordnet::Fft                     fft(n_fft);
    std::vector<std::complex<double>> z(n_fft);
    std::vector<double>               p0(n_fft / 2 + 1), p1(n_fft / 2 + 1);
    const int                         half = n_fft / 2, nb = (int) fb.size();
    for (int t = 0; t < T; t += 2) {  // two real frames per complex FFT
        for (int j = 0; j < n_fft; j++) {
            const int s0 = t * HOP - half + j, s1 = s0 + HOP;
            const double a = (s0 >= 0 && s0 < n) ? (double) (x[s0] * win[j]) : 0.0;
            const double b = (s1 >= 0 && s1 < n && t + 1 < T) ? (double) (x[s1] * win[j]) : 0.0;
            z[j] = { a, b };
        }
        fft.run(z.data());
        for (int k = 0; k <= half; k++) {
            const std::complex<double> zk = z[k], zn = std::conj(z[(n_fft - k) % n_fft]);
            const std::complex<double> X0 = 0.5 * (zk + zn), X1 = std::complex<double>(0, -0.5) * (zk - zn);
            p0[k] = std::norm(X0);
            p1[k] = std::norm(X1);
        }
        for (int i = 0; i < nb; i++) {
            const Band & B = fb[i];
            double       e0 = 0, e1 = 0;
            for (size_t k = 0; k < B.w.size(); k++) {
                e0 += B.w[k] * p0[B.lo + k];
                e1 += B.w[k] * p1[B.lo + k];
            }
            out[(size_t) i * T + t] = (float) std::log(e0 + LOG_EPS);
            if (t + 1 < T) out[(size_t) i * T + t + 1] = (float) std::log(e1 + LOG_EPS);
            if (lin) {
                lin[(size_t) i * T + t] = (float) e0;
                if (t + 1 < T) lin[(size_t) i * T + t + 1] = (float) e1;
            }
        }
    }
}

// HCQT of one window (8 s for the transcriber) -> [N_CH][T][N_BINS] (WHCN, bins fastest), as torch's (C, T, F).
// lin (optional): the long window's band energies [F.fb.size()][T], the grouper's linear power P.
static void hcqt(const Front & F, const float * x, float * out, int n = WIN, int T = T10, float * lin = nullptr) {
    const int          n_long = (int) F.fb.size();
    std::vector<float> lf((size_t) n_long * T), lx((size_t) N_BINS * T);
    log_bands(x, N_FFT, F.win, F.fb, lf.data(), n, T, lin);
    log_bands(x, N_FFT_X, F.win_x, F.fb_x, lx.data(), n, T);
    for (int c = 0; c < N_CH; c++) {
        float * o = out + (size_t) c * T * N_BINS;
        for (int b = 0; b < N_BINS; b++) {
            const int a   = c < N_HARM ? F.shifts[c] - F.lo + b : b;
            const bool in = c == N_HARM || a < n_long;  // the pad branch is never taken with this config
            const float * src = c < N_HARM ? &lf[(size_t) a * T] : &lx[(size_t) b * T];
            for (int t = 0; t < T; t++) o[(size_t) t * N_BINS + b] = in ? src[t] : (float) std::log(LOG_EPS);
        }
    }
}

// ---------------------------------------------------------------- window layout (Transcriber._layout)
struct Layout {
    int              T = 0;      // song frames of 20 ms
    std::vector<int> starts;     // padded-sample start of every window
};

static Layout layout(long long n) {
    Layout L;
    L.T = (int) std::ceil((double) n / SR / FRAME_S);
    for (long long s = 0; s < n + PAD; s += STEP) L.starts.push_back((int) s);
    return L;
}

static int frame_offset(int s0) { return (int) std::nearbyint((double) s0 / SR / FRAME_S) - G_PAD; }

// np.hanning(FW + 2)[1:-1] as float32
static std::vector<float> taper() {
    std::vector<float> w(FW);
    for (int i = 0; i < FW; i++) w[i] = (float) (0.5 - 0.5 * std::cos(2.0 * M_PI * (i + 1) / (FW + 1)));
    return w;
}

// Hann-weighted sums of the maps (float32) and of float16 taps (_Average, device path: float16 sums,
// float16 division by the float16 weight sum).
struct Average {
    int                       T = 0;
    std::vector<float>        P[N_HEADS_OUT], wsum;
    std::vector<ggml_fp16_t>  hidden, stem;  // [T][88][C], only when requested
    bool                      want_hidden = false, want_stem = false;
    int                       C = 0;

    void init(int T_, int C_, bool h, bool s) {
        T = T_;
        C = C_;
        want_hidden = h;
        want_stem   = s;
        for (auto & p : P) p.assign((size_t) T * N_KEYS, 0.0f);
        wsum.assign(T, 0.0f);
        if (h) hidden.assign((size_t) T * N_KEYS * C, ggml_fp32_to_fp16(0.0f));
        if (s) stem.assign((size_t) T * N_KEYS * C, ggml_fp32_to_fp16(0.0f));
    }

    // fn(t0, t1) over [a, b) in parallel slices of frames
    template <typename F> static void frames(int a, int b, F fn) {
        const int n = b - a, nt = std::max(1, std::min({ 16, (int) std::thread::hardware_concurrency(), n / 8 }));
        std::vector<std::thread> pool;
        for (int i = 0; i < nt; i++) pool.emplace_back(fn, a + (int) ((long long) n * i / nt), a + (int) ((long long) n * (i + 1) / nt));
        for (auto & th : pool) th.join();
    }

    static void add_tap(std::vector<ggml_fp16_t> & buf, const float * v, int a, int b, int g0, const float * wt, int C) {
        // v: one window's tap [FW][88][C]
        frames(a, b, [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            const float   w   = wt[t - a];
            const float * src = v + (size_t) (t - g0) * N_KEYS * C;
            ggml_fp16_t * dst = &buf[(size_t) t * N_KEYS * C];
            for (int i = 0; i < N_KEYS * C; i++) {
                const ggml_fp16_t x = ggml_fp32_to_fp16(src[i] * w);
                dst[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(dst[i]) + ggml_fp16_to_fp32(x));
            }
        }
        });
    }

    // probs: one window's maps [6][FW][88]
    void add(int a, int b, int g0, const float * wt, const float * probs, const float * hid, const float * stm) {
        for (int h = 0; h < N_HEADS_OUT; h++)
            for (int t = a; t < b; t++) {
                const float * src = probs + ((size_t) h * FW + (t - g0)) * N_KEYS;
                float *       dst = &P[h][(size_t) t * N_KEYS];
                const float   w   = wt[t - a];
                for (int k = 0; k < N_KEYS; k++) dst[k] += src[k] * w;
            }
        if (want_hidden) add_tap(hidden, hid, a, b, g0, wt, C);
        if (want_stem) add_tap(stem, stm, a, b, g0, wt, C);
        for (int t = a; t < b; t++) wsum[t] += wt[t - a];
    }

    static void finish_tap(std::vector<ggml_fp16_t> & buf, const std::vector<float> & ws, int C) {
        frames(0, (int) ws.size(), [&](int t0, int t1) {
            for (int t = t0; t < t1; t++) {
                const float d = ggml_fp16_to_fp32(ggml_fp32_to_fp16(std::max(ws[t], 1e-6f)));
                ggml_fp16_t * p = &buf[(size_t) t * N_KEYS * C];
                for (int i = 0; i < N_KEYS * C; i++) p[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(p[i]) / d);
            }
        });
    }

    void finish() {
        for (int t = 0; t < T; t++) {
            const float d = std::max(wsum[t], 1e-6f);
            for (auto & p : P)
                for (int k = 0; k < N_KEYS; k++) p[(size_t) t * N_KEYS + k] /= d;
        }
        if (want_hidden) finish_tap(hidden, wsum, C);
        if (want_stem) finish_tap(stem, wsum, C);
    }
};

// ---------------------------------------------------------------- note decode (decode.py, Transcriber.notes)
struct Note {
    double onset, offset;
    int    pitch;
    float  conf;
    int    slot;
};

// Sub-frame position of a peak from its neighbours, numpy float32 steps.
static double refine_peak(const float * p, int T, int t, int l) {
    const float C  = p[(size_t) t * N_KEYS + l];
    const float A  = t > 0 ? p[(size_t) (t - 1) * N_KEYS + l] : C;
    const float Bv = t + 1 < T ? p[(size_t) (t + 1) * N_KEYS + l] : C;
    const float den = 2.0f * (C - std::min(A, Bv));
    if (den <= 1e-6f) return 0.0;  // numpy 2: the Python float compares as float32
    float r = (Bv - A) / den;
    r       = std::min(0.5f, std::max(-0.5f, r));
    return (double) r;
}

struct DecodeTh {
    double on_th = 0.25, off_th = 0.3, fr_th = 0.4, v2_th = 0.3;
    int    fr_low_frames = 2, on_nms = 3, off_min_frames = 2;
    double min_dur = 0.02;
};

// One voice: maps [T][88] -> (onset s, offset s, pitch), sorted.
static std::vector<Note> decode_voice(const float * on, const float * fr, const float * off, int T, double on_th, const DecodeTh & th) {
    auto peaks = [&](const float * p, double thr, int l, std::vector<int> & out) {
        out.clear();
        for (int t = 0; t < T; t++) {
            const float c = p[(size_t) t * N_KEYS + l];
            const float left = t > 0 ? p[(size_t) (t - 1) * N_KEYS + l] : -1.0f;
            const float right = t + 1 < T ? p[(size_t) (t + 1) * N_KEYS + l] : -1.0f;
            if (c >= (float) thr && c > left && c >= right) out.push_back(t);
        }
    };
    std::vector<Note> notes;
    std::vector<int>  ons, offs, lows;
    std::vector<char> low(T), run(T);
    for (int l = 0; l < N_KEYS; l++) {
        peaks(on, on_th, l, ons);
        if (ons.empty()) continue;
        if (th.on_nms > 0 && ons.size() > 1) {
            std::vector<int> order(ons.size());
            for (size_t i = 0; i < ons.size(); i++) order[i] = (int) i;
            std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
                return on[(size_t) ons[x] * N_KEYS + l] > on[(size_t) ons[y] * N_KEYS + l];
            });
            std::vector<char> keep(ons.size(), 1);
            for (int oi : order) {
                if (!keep[oi]) continue;
                for (size_t j = 0; j < ons.size(); j++)
                    if ((int) j != oi && std::abs(ons[j] - ons[oi]) <= th.on_nms) keep[j] = 0;
            }
            std::vector<int> kept;
            for (size_t j = 0; j < ons.size(); j++)
                if (keep[j]) kept.push_back(ons[j]);
            ons.swap(kept);
        }
        peaks(off, th.off_th, l, offs);
        for (int t = 0; t < T; t++) low[t] = fr[(size_t) t * N_KEYS + l] < (float) th.fr_th;
        for (int t = 0; t < T; t++) run[t] = low[t];
        for (int k = 1; k < th.fr_low_frames; k++)
            for (int t = 0; t < T - k; t++) run[t] = run[t] && low[t + k];
        lows.clear();
        for (int t = 0; t < T; t++)
            if (run[t]) lows.push_back(t);
        for (size_t i = 0; i < ons.size(); i++) {
            const int    t0  = ons[i];
            const double u0  = t0 + refine_peak(on, T, t0, l);
            const int    nxt = i + 1 < ons.size() ? ons[i + 1] : T;
            auto         j1  = std::lower_bound(lows.begin(), lows.end(), t0 + 1);
            const int    e_fr = j1 != lows.end() ? *j1 : T;
            auto         j2  = std::lower_bound(offs.begin(), offs.end(), t0 + th.off_min_frames);
            const int    e_off = j2 != offs.end() ? *j2 : T;
            double       u1;
            if (e_off < T && e_off <= std::min(e_fr, nxt)) u1 = e_off + refine_peak(off, T, e_off, l);
            else if (e_fr <= nxt) u1 = e_fr - 0.5;
            else u1 = nxt + (nxt < T ? refine_peak(on, T, nxt, l) : 0.0);
            const double on_s = u0 * FRAME_S, off_s = std::max(u1 * FRAME_S, u0 * FRAME_S + th.min_dur);
            notes.push_back({ on_s, off_s, PITCH_LO + l, 0.0f, 0 });
        }
    }
    std::sort(notes.begin(), notes.end(), [](const Note & a, const Note & b) {
        if (a.onset != b.onset) return a.onset < b.onset;
        if (a.offset != b.offset) return a.offset < b.offset;
        return a.pitch < b.pitch;
    });
    return notes;
}

// Both voices -> sorted by (onset, offset, pitch), slot 1 first among ties; conf = the voice's onset probability.
static std::vector<Note> decode_notes(const std::vector<float> * maps, int T, const DecodeTh & th = DecodeTh()) {
    std::vector<Note> rows;
    for (int slot = 1; slot <= 2; slot++) {
        const int    o     = slot == 1 ? 0 : 3;
        const double on_th = slot == 1 ? th.on_th : th.v2_th;
        const float * on   = maps[o].data();
        for (Note n : decode_voice(on, maps[o + 1].data(), maps[o + 2].data(), T, on_th, th)) {
            const int t = (int) std::min(std::max(std::nearbyint(n.onset / FRAME_S), 0.0), (double) (T - 1));
            n.conf      = on[(size_t) t * N_KEYS + (n.pitch - PITCH_LO)];
            n.slot      = slot;
            rows.push_back(n);
        }
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Note & a, const Note & b) {
        if (a.onset != b.onset) return a.onset < b.onset;
        if (a.offset != b.offset) return a.offset < b.offset;
        return a.pitch < b.pitch;
    });
    return rows;
}

// Pitched velocity as the grouper writes it: int(clip(20 + 107 * conf, 1, 127)), float32 arithmetic.
static int velocity(float conf) {
    const float v = 20.0f + 107.0f * conf;
    return (int) std::min(127.0f, std::max(1.0f, v));
}

// ---------------------------------------------------------------- tuning (grouper/retune.py)
struct Tuning {
    double estimate_cents = 0, strength = 0, shift_cents = 0, ratio = 1;
};

static Tuning estimate_tuning(const std::vector<float> & y) {
    constexpr int    NF = 8192, H = 6, HOPT = 1024;
    constexpr double FMIN = 65.0, FMAX = 1000.0, STEPC = 5.0, REL = 0.5;
    Tuning           out;
    if ((long long) y.size() < NF) return out;
    std::vector<float> win(NF);
    for (int i = 0; i < NF; i++) win[i] = (float) (0.5 - 0.5 * std::cos(2.0 * M_PI * i / (NF - 1)));  // np.hanning
    const long long nfr = 1 + ((long long) y.size() - NF) / HOPT;
    const double    c0 = 1200 * std::log2(FMIN / 440), c1 = 1200 * std::log2(FMAX / 440);
    const int       nc = (int) std::ceil((c1 - c0) / STEPC);
    std::vector<double> cents(nc), f0(nc);
    for (int i = 0; i < nc; i++) {
        cents[i] = c0 + i * STEPC;
        f0[i]    = 440 * std::pow(2.0, cents[i] / 1200);
    }
    const int nb = NF / 2 + 1;
    struct Tap { int i0; float w; };
    std::vector<std::vector<Tap>> taps(H);  // per harmonic, per cent bin (i0 < 0: not ok)
    for (int h = 1; h <= H; h++) {
        taps[h - 1].resize(nc);
        for (int i = 0; i < nc; i++) {
            const double fb = h * f0[i] * NF / SR;
            if (fb < nb - 2) taps[h - 1][i] = { (int) std::floor(fb), (float) (fb - std::floor(fb)) };
            else taps[h - 1][i] = { -1, 0.0f };
        }
    }
    std::vector<double> cs, ws;
    // per frame (frames are independent; upstream batches 512 at a time, which does not change the values)
    const int nthreads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    std::vector<std::vector<std::pair<double, float>>> per(nthreads);
    std::vector<std::thread> pool;
    for (int ti = 0; ti < nthreads; ti++)
        pool.emplace_back([&, ti]() {
            chordnet::Fft                     fft(NF);
            std::vector<std::complex<double>> z(NF);
            std::vector<float>                S(nb), sal(nc);
            for (long long f = ti; f < nfr; f += nthreads) {
                const float * x = y.data() + f * HOPT;
                for (int j = 0; j < NF; j++) z[j] = { (double) (x[j] * win[j]), 0.0 };
                fft.run(z.data());
                for (int k = 0; k < nb; k++) S[k] = std::sqrt((float) std::abs(z[k]));
                std::fill(sal.begin(), sal.end(), 0.0f);
                for (int h = 1; h <= H; h++) {
                    const float g = (float) std::pow(0.85, h - 1);
                    for (int i = 0; i < nc; i++) {
                        const Tap & tp = taps[h - 1][i];
                        if (tp.i0 < 0) continue;
                        sal[i] += g * ((1 - tp.w) * S[tp.i0] + tp.w * S[tp.i0 + 1]);
                    }
                }
                float mx = sal[0];
                for (int i = 1; i < nc; i++) mx = std::max(mx, sal[i]);
                const float thr = (float) REL * mx;
                for (int i = 0; i + 2 < nc; i++) {
                    const float a = sal[i], b = sal[i + 1], c = sal[i + 2];
                    if (b > a && b >= c && b > thr && mx > 0) {
                        float d = 0.5f * (a - c) / (a - 2 * b + c - 1e-12f);
                        d       = std::min(1.0f, std::max(-1.0f, d));
                        double v = cents[i + 1] + (double) (d * (float) STEPC) + 50;  // float32 product, float64 sum
                        v        = std::fmod(v, 100.0);
                        if (v < 0) v += 100.0;
                        per[ti].push_back({ v - 50, b });
                    }
                }
            }
        });
    for (auto & t : pool) t.join();
    double wsum = 0, re = 0, im = 0;
    for (auto & v : per)
        for (auto & p : v) {
            wsum += p.second;
            re += p.second * std::cos(2 * M_PI * p.first / 100);
            im += p.second * std::sin(2 * M_PI * p.first / 100);
        }
    if (wsum <= 0) return out;
    re /= wsum;
    im /= wsum;
    out.estimate_cents = std::atan2(im, re) * 100 / (2 * M_PI);
    out.strength       = std::hypot(re, im);
    const bool shift   = std::fabs(out.estimate_cents) >= 10.0 && out.strength >= 0.12;
    out.shift_cents    = shift ? -out.estimate_cents : 0.0;
    out.ratio          = std::pow(2.0, out.shift_cents / 1200.0);
    return out;
}

}  // namespace dettato::transcriber
