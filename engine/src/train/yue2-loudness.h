#pragma once
// yue2-loudness.h — one loudness for every YuE2 training track (2026-09-23)
//
// Datasets arrive at whatever level they were mastered at: a brickwalled album
// sits near -8 LUFS with overs past 0 dBFS, an older one near -16. The adapter
// learns that level along with everything else, and on the loud ones it then
// overshoots it: plus44's step-100 preview rendered 4 dB hotter than its own
// training audio and clipped 2% of its samples. So every track is brought to
// one integrated loudness before anything encodes it.
//
// The gain is measured ONCE, by yue2-preprocess, and recorded per source in
// the manifest (`loudness_gain_db`). The later stages that decode the same
// file (yue2-tokenize, yue2-sheet) apply that recorded gain rather than
// measuring again, so every stream built from one track starts from the same
// samples.
//
// The measurement is a port of integratedLufs() in
// server/src/services/generation/audioLevel.ts (ITU-R BS.1770-4: K-weighting,
// 400 ms blocks at 75% overlap, -70 LUFS absolute gate, -10 LU relative gate),
// so training and the generation-side normalizer agree on what a LUFS is.
// Gain is a plain scalar, boost or cut. A boost stops short of pushing the
// sample peak past the ceiling: a very dynamic track then lands a little under
// the target, which beats limiting it and teaching the model the limiter.

#include <algorithm>
#include <cmath>
#include <vector>

// BS.1770 K-weighting filter coefficients at 48 kHz (the rate every YuE2
// training decode produces): a high-shelf pre-filter, then the RLB high-pass.
static void yue2_kweight_48k(const float * x, int n, std::vector<double> * out) {
    static const double b1[3] = { 1.53512485958697, -2.69169618940638, 1.19839281085285 };
    static const double a1[3] = { 1.0, -1.69065929318241, 0.73248077421585 };
    static const double b2[3] = { 1.0, -2.0, 1.0 };
    static const double a2[3] = { 1.0, -1.99004745483398, 0.99007225036621 };
    out->resize((size_t) n);
    double x1 = 0, x2 = 0, y1 = 0, y2 = 0, z1 = 0, z2 = 0, w1 = 0, w2 = 0;
    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const double y  = b1[0] * xi + b1[1] * x1 + b1[2] * x2 - a1[1] * y1 - a1[2] * y2;
        x2 = x1; x1 = xi; y2 = y1; y1 = y;
        const double w = b2[0] * y + b2[1] * z1 + b2[2] * z2 - a2[1] * w1 - a2[2] * w2;
        z2 = z1; z1 = y; w2 = w1; w1 = w;
        (*out)[(size_t) i] = w;
    }
}

// Integrated loudness of 48 kHz PLANAR stereo (channel 1 starts at T).
// -inf for silence or anything shorter than one 400 ms block.
static double yue2_integrated_lufs_48k(const float * planar, int T) {
    const int block = 19200, hop = 4800;  // 400 ms, 100 ms at 48 kHz
    if (T < block) return -INFINITY;
    std::vector<double> kl, kr;
    yue2_kweight_48k(planar, T, &kl);
    yue2_kweight_48k(planar + T, T, &kr);
    const int nb = (T - block) / hop + 1;
    std::vector<double> loud((size_t) nb);
    for (int b = 0; b < nb; b++) {
        double pl = 0, pr = 0;
        for (int i = b * hop, e = i + block; i < e; i++) { pl += kl[(size_t) i] * kl[(size_t) i]; pr += kr[(size_t) i] * kr[(size_t) i]; }
        const double p = pl / block + pr / block;
        loud[(size_t) b] = p > 0 ? -0.691 + 10.0 * std::log10(p) : -INFINITY;
    }
    auto mean_of = [&](double gate) {
        double sum = 0; int n = 0;
        for (double l : loud) if (l > gate) { sum += std::pow(10.0, (l + 0.691) / 10.0); n++; }
        return n ? -0.691 + 10.0 * std::log10(sum / n) : -INFINITY;
    };
    const double ungated = mean_of(-70.0);
    if (!std::isfinite(ungated)) return -INFINITY;
    return mean_of(std::max(-70.0, ungated - 10.0));
}

// The gain (dB) that brings `planar` to `target_lufs`, a boost capped so the
// sample peak stays at or under `ceiling_dbfs`. 0 for silence. Reports what it
// measured so the caller can log it.
static double yue2_loudness_gain_db(const float * planar, int T, double target_lufs, double ceiling_dbfs,
                                    double * lufs_out, bool * capped_out) {
    const double lufs = yue2_integrated_lufs_48k(planar, T);
    float peak = 0.0f;
    for (size_t k = 0, n = (size_t) T * 2; k < n; k++) peak = std::max(peak, std::fabs(planar[k]));
    if (lufs_out) *lufs_out = lufs;
    if (capped_out) *capped_out = false;
    if (!std::isfinite(lufs) || peak <= 0.0f) return 0.0;
    double gain = target_lufs - lufs;
    const double headroom = ceiling_dbfs - 20.0 * std::log10((double) peak);
    if (gain > 0.0 && gain > headroom) {
        gain = std::max(0.0, headroom);
        if (capped_out) *capped_out = true;
    }
    return gain;
}

static void yue2_apply_gain_db(float * planar, int T, double gain_db) {
    if (gain_db == 0.0) return;
    const float g = (float) std::pow(10.0, gain_db / 20.0);
    for (size_t k = 0, n = (size_t) T * 2; k < n; k++) planar[k] *= g;
}
