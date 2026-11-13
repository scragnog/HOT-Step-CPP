#pragma once
// chordnet/tempo.h: the tempo librosa.beat.beat_track(y=y, sr=sr) returns,
// which MuLaCover takes as the song's BPM at the source sample rate. Written
// from librosa 0.11 (ISC, see LICENSE): onset_strength with aggregate=median,
// then feature.tempo. The beat positions are not needed: beat_track estimates
// the tempo before it tracks beats and returns that estimate unchanged.
//   onset   power STFT (n_fft 2048, hop 512, centred, zero padding, periodic
//           Hann), 128 Slaney mel bands to sr/2, power_to_db (amin 1e-10,
//           top_db 80), positive first difference, median over bands, three
//           leading zeros (lag 1 + n_fft / (2 hop)), cut to the frame count.
//           An envelope with no positive value gives tempo 0, as librosa does.
//   tempo   autocorrelation tempogram over 8 s windows (periodic Hann,
//           linear-ramp padding, max normalised per frame), averaged over
//           time, scored log1p(1e6 tg) plus a log-normal prior around 120 BPM
//           (one octave deviation) with lags faster than 320 BPM excluded;
//           the best lag gives 60 sr / (512 lag).
// librosa runs the STFT and mel stages in float32; this runs them in double.
// On the reference clips the envelopes differ by up to 1.1e-5 while the best
// lag leads the runner-up by 0.066 to 0.29 in score, so the lag is the same.

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "chordnet/features.h"  // Fft, stft_frames

namespace chordnet {

// librosa slaney mel scale (hz_to_mel / mel_to_hz, htk=False)
static double slaney_hz_to_mel(double f) {
    const double f_sp = 200.0 / 3, logstep = std::log(6.4) / 27.0;
    return f >= 1000.0 ? 1000.0 / f_sp + std::log(f / 1000.0) / logstep : f / f_sp;
}
static double slaney_mel_to_hz(double m) {
    const double f_sp = 200.0 / 3, logstep = std::log(6.4) / 27.0, min_log_mel = 1000.0 / f_sp;
    return m >= min_log_mel ? 1000.0 * std::exp(logstep * (m - min_log_mel)) : f_sp * m;
}

// librosa.onset.onset_strength(y, sr, hop_length=512, aggregate=np.median)
static std::vector<double> onset_strength(const std::vector<float> & y, int sr) {
    constexpr int NFFT = 2048, NB = NFFT / 2 + 1, NMEL = 128;
    // librosa.filters.mel(sr, n_fft=2048, n_mels=128, fmin=0, fmax=sr/2, norm="slaney"),
    // kept as each band's nonzero bin range
    std::vector<double> mel_f(NMEL + 2);
    const double m1 = slaney_hz_to_mel(0.5 * sr);
    for (int i = 0; i < NMEL + 2; i++) mel_f[i] = slaney_mel_to_hz(m1 * i / (NMEL + 1));  // linspace(0, m1, 130)
    std::vector<std::vector<double>> wt(NMEL);
    std::vector<int>                 lo(NMEL, NB);
    for (int m = 0; m < NMEL; m++) {
        const double enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        std::vector<double> w(NB);
        int hi = 0;
        for (int k = 0; k < NB; k++) {
            const double f     = (double) k * sr / NFFT;
            const double lower = -(mel_f[m] - f) / (mel_f[m + 1] - mel_f[m]);
            const double upper = (mel_f[m + 2] - f) / (mel_f[m + 2] - mel_f[m + 1]);
            w[k] = std::max(0.0, std::min(lower, upper)) * enorm;
            if (w[k] > 0) { lo[m] = std::min(lo[m], k); hi = k + 1; }
        }
        if (hi > lo[m]) wt[m].assign(w.begin() + lo[m], w.begin() + hi);
    }
    std::vector<double> db;  // [frame][NMEL]
    std::vector<double> pw(NB);
    const int T = stft_frames(y, NFFT, 512, true, [&](int, const cd * X) {
        for (int k = 0; k < NB; k++) pw[k] = std::norm(X[k]);
        for (int m = 0; m < NMEL; m++) {
            double s = 0;
            for (size_t j = 0; j < wt[m].size(); j++) s += wt[m][j] * pw[lo[m] + j];
            db.push_back(10.0 * std::log10(std::max(1e-10, s)));
        }
    });
    const double top = *std::max_element(db.begin(), db.end()) - 80.0;
    for (double & v : db) v = std::max(v, top);
    std::vector<double> env(T, 0.0), d(NMEL);
    for (int t = 0; t + 1 < T && t + 3 < T; t++) {
        for (int m = 0; m < NMEL; m++) d[m] = std::max(0.0, db[(size_t) (t + 1) * NMEL + m] - db[(size_t) t * NMEL + m]);
        std::nth_element(d.begin(), d.begin() + NMEL / 2, d.end());
        const double hi = d[NMEL / 2], lo2 = *std::max_element(d.begin(), d.begin() + NMEL / 2);
        env[t + 3] = 0.5 * (lo2 + hi);  // even count: mean of the two middle values
    }
    return env;
}

// librosa.feature.tempo(onset_envelope, sr, hop_length=512) with the defaults
// beat_track uses; returns 0 for an envelope that is zero everywhere.
static double librosa_tempo(const std::vector<double> & env, int sr) {
    if (std::none_of(env.begin(), env.end(), [](double v) { return v != 0; })) return 0.0;
    const int win = (int) ((long long) (8.0 * sr) / 512);  // time_to_frames(8.0)
    const int n = (int) env.size(), half = win / 2;
    // np.pad(mode="linear_ramp", end_values=0): ramps from 0 up to each edge value
    std::vector<double> p((size_t) n + 2 * half);
    for (int i = 0; i < half; i++) p[i] = env[0] * i / half;
    for (int i = 0; i < n; i++) p[half + i] = env[i];
    for (int j = 0; j < half; j++) p[(size_t) half + n + j] = env[n - 1] * (half - 1 - j) / half;
    int nf = 1;
    while (nf < 2 * win - 1) nf <<= 1;
    const Fft fft(nf);
    std::vector<cd>     a(nf);
    std::vector<double> win_w(win), tg(win, 0.0);
    for (int i = 0; i < win; i++) win_w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / win);
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < nf; i++) a[i] = i < win ? cd(p[(size_t) t + i] * win_w[i], 0.0) : cd(0.0, 0.0);
        fft.run(a.data());
        for (int i = 0; i < nf; i++) a[i] = cd(std::norm(a[i]), 0.0);
        fft.run(a.data());  // real, even spectrum: the forward transform is the inverse times nf
        double mx = 0;
        for (int l = 0; l < win; l++) mx = std::max(mx, std::fabs(a[l].real()));
        const double scale = mx < std::numeric_limits<double>::min() ? 1.0 : mx;
        for (int l = 0; l < win; l++) tg[l] += a[l].real() / scale;
    }
    int    best  = -1;
    double score = -INFINITY;
    for (int l = 1; l < win; l++) {
        const double bpm = 60.0 * sr / (512.0 * l);
        if (bpm >= 320.0) continue;
        const double pr = std::log2(bpm) - std::log2(120.0);
        const double s  = std::log1p(1e6 * (tg[l] / n)) - 0.5 * pr * pr;
        if (s > score) { score = s; best = l; }
    }
    return best > 0 ? 60.0 * sr / (512.0 * best) : 0.0;
}

static double beat_track_tempo(const std::vector<float> & y, int sr) { return librosa_tempo(onset_strength(y, sr), sr); }

}  // namespace chordnet
