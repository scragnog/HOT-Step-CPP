#pragma once
// midi-grid.h: beat grid, tempo map and bar alignment for MuScriptor MIDI
//
// Port of upstream MuScriptor's utils/beats.py (grid fit, meter, onset delay)
// and the timing half of utils/midi.py (export options, grid resolution,
// tempo-map timeline), v0.4.0. The beats come from beat-this.h. Pure maths on
// times in seconds and beats; the note-level steps (quantize, gap fill) and
// the SMF writer live in ace-midi.cpp.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#define GRID_MAX_TEMPO_RESIDUAL 0.05   // of a beat: below it the song is "steady"
#define GRID_MIN_METER_AGREEMENT 0.9   // bars that must agree to write a meter
#define GRID_MIN_BEATS          8
#define GRID_EXTRA_BEAT_SPACING 0.7    // closer than this * median IBI = extra beat
#define GRID_MIN_CONCENTRATION  0.5    // onset-delay |R| floor
#define GRID_MIN_ONSETS         40
#define GRID_MAX_ONSET_DELAY    0.04   // seconds
#define GRID_TEMPO_MAP_TOL      0.025  // seconds a tempo map may put a beat off
#define GRID_SMOOTHING_WINDOW   9

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const int GRID_SUBDIVISIONS[] = { 1, 2, 3, 4, 6, 8, 12, 16, 24 };

// Python's float %, never negative for a positive divisor
static double grid_pymod(double a, double b) { return a - floor(a / b) * b; }

struct BeatGrid {
    double              bpm            = 120.0;
    int                 beats_per_bar  = 0;      // 0: meter unknown, write no time signature
    double              first_downbeat = 0.0;    // seconds
    std::vector<double> beats;                   // tracked beats; empty for a hand-made grid
    bool                delay_known    = false;
    double              onset_delay    = 0.0;    // seconds the onsets sit late
    int                 subdivision    = 0;      // grid steps per beat to quantize to; 0 none
    bool                steady         = true;   // fits one tempo; else write a tempo map

    double bar_seconds() const { return beats_per_bar ? beats_per_bar * 60.0 / bpm : 0.0; }

    // Seconds to delay every note so bar lines land on downbeats (MIDI has no
    // pickup bar), grown by whole bars until it covers min_shift.
    double bar_offset(double min_shift) const {
        double step = bar_seconds(), offset;
        if (step == 0.0) {
            step   = 60.0 / bpm;
            offset = 0.0;
        } else {
            offset = grid_pymod(step - grid_pymod(first_downbeat, step), step);
        }
        if (offset < min_shift) offset += step * ceil((min_shift - offset) / step);
        return offset;
    }
};

struct GridExport {                 // upstream ExportOptions
    double bpm            = 0.0;    // 0: keep detected
    int    beats_per_bar  = 0;      // 0: keep detected
    bool   has_downbeat   = false;
    double first_downbeat = 0.0;
    int    tempo_map      = -1;     // -1: follow the beats when unsteady, 0 fixed, 1 follow
    double beat_scale     = 1.0;    // power of two
    bool   quantize       = false;
    int    subdivision    = 0;      // 0: the measured one
    double strength       = 1.0;
    double fill_gaps      = 0.0;    // beats
    bool   trim_leading   = false;
};

static size_t grid_nearest(const std::vector<double> & v, double t) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); i++) {
        if (fabs(v[i] - t) < fabs(v[best] - t)) best = i;
    }
    return best;
}

// np.interp: clamped linear interpolation, xp increasing
static double grid_interp(double x, const std::vector<double> & xp, const std::vector<double> & fp) {
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back()) return fp.back();
    size_t i = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();
    double t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
    return fp[i - 1] + t * (fp[i] - fp[i - 1]);
}

// Least-squares tempo: (bpm, residual std in seconds) of beat time on index
static std::pair<double, double> grid_fit_tempo(const std::vector<double> & beats) {
    const double n = (double) beats.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < beats.size(); i++) {
        sx += i; sy += beats[i]; sxx += (double) i * i; sxy += i * beats[i];
    }
    double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    double icept = (sy - slope * sx) / n;
    double mean = 0, var = 0;
    std::vector<double> r(beats.size());
    for (size_t i = 0; i < beats.size(); i++) mean += (r[i] = beats[i] - (icept + slope * i));
    mean /= n;
    for (double e : r) var += (e - mean) * (e - mean);
    return { 60.0 / slope, sqrt(var / n) };
}

// Put the tracker's missed beats back and drop its extra ones, measured
// against the median spacing.
static std::vector<double> grid_regularized(const std::vector<double> & beats) {
    if (beats.size() < 3) return beats;
    std::vector<double> d;
    for (size_t i = 1; i < beats.size(); i++) d.push_back(beats[i] - beats[i - 1]);
    std::sort(d.begin(), d.end());
    const double period = d.size() % 2 ? d[d.size() / 2] : 0.5 * (d[d.size() / 2 - 1] + d[d.size() / 2]);
    std::vector<double> kept = { beats[0] };
    for (size_t i = 1; i < beats.size(); i++) {
        const double prev = kept.back(), apart = (beats[i] - prev) / period;
        if (apart < GRID_EXTRA_BEAT_SPACING) continue;
        int missing = std::max(1, (int) nearbyint(apart));
        for (int k = 1; k < missing; k++) kept.push_back(prev + (beats[i] - prev) * k / missing);
        kept.push_back(beats[i]);
    }
    return kept;
}

// Beats per bar from downbeat spacing in tracked beats, 0 if bars disagree
static int grid_infer_meter(const std::vector<double> & beats, const std::vector<double> & downbeats) {
    if (downbeats.size() < 3 || beats.size() < 2) return 0;
    std::vector<int> counts;
    size_t prev = grid_nearest(beats, downbeats[0]);
    for (size_t i = 1; i < downbeats.size(); i++) {
        size_t idx = grid_nearest(beats, downbeats[i]);
        int    c   = (int) idx - (int) prev;
        if (c >= 2) counts.push_back(c);
        prev = idx;
    }
    if (counts.empty()) return 0;
    std::sort(counts.begin(), counts.end());
    int best = 0, best_n = 0;
    for (size_t i = 0; i < counts.size();) {   // np.unique + argmax: smallest value wins a tie
        size_t j = i;
        while (j < counts.size() && counts[j] == counts[i]) j++;
        if ((int) (j - i) > best_n) { best = counts[i]; best_n = (int) (j - i); }
        i = j;
    }
    return (double) best_n / counts.size() >= GRID_MIN_METER_AGREEMENT ? best : 0;
}

// beats.py detect_grid, after the tracker. False when there are too few beats.
static bool grid_detect(const std::vector<double> & tracked, const std::vector<double> & downbeats, BeatGrid * g) {
    std::vector<double> beats = grid_regularized(tracked);
    if (beats.size() < GRID_MIN_BEATS) return false;
    auto fit = grid_fit_tempo(beats);
    *g                = BeatGrid();
    g->bpm            = fit.first;
    g->steady         = fit.second <= GRID_MAX_TEMPO_RESIDUAL * 60.0 / fit.first;
    const int meter   = grid_infer_meter(beats, downbeats);
    g->beats_per_bar  = meter ? meter : 4;
    g->first_downbeat = downbeats.empty() ? beats[0] : downbeats[0];
    g->beats          = beats;
    fprintf(stderr, "[ace-midi] grid: %.3f BPM%s, %s, first downbeat %.3f s (residual %.1f ms)\n", g->bpm,
            g->steady ? "" : " on average, not steady",
            meter ? (std::to_string(meter) + "/4").c_str() : "assumed 4/4",
            g->first_downbeat, fit.second * 1000.0);
    return true;
}

// How late the onsets sit against the beat subdivision they are on: each onset
// is a unit vector at its phase within the subdivision, and the mean vector's
// angle is the delay, its length how tightly they sit. False when unsure.
struct OnsetDelay { double seconds, concentration; int subdivision, n; };

static bool grid_onset_delay(const std::vector<double> & onsets, const BeatGrid & g, OnsetDelay * out) {
    if (g.beats.size() < 2) return false;
    const double period = 60.0 / g.bpm;
    std::vector<double> times;
    for (double t : onsets) times.push_back(nearbyint(t * 1000.0) / 1000.0);
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    std::vector<double> idx(g.beats.size()), phase;
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (double) i;
    for (double t : times) {
        if (t >= g.beats.front() && t <= g.beats.back()) phase.push_back(grid_interp(t, g.beats, idx));
    }
    if (phase.size() < GRID_MIN_ONSETS) return false;

    int best = 0;
    double best_r = -1.0, best_delay = 0.0;
    for (int s : GRID_SUBDIVISIONS) {
        if (period / (2 * s) < GRID_MAX_ONSET_DELAY) continue;
        double re = 0, im = 0;
        for (double p : phase) {
            double a = 2 * M_PI * grid_pymod(p * s, 1.0);
            re += cos(a);
            im += sin(a);
        }
        re /= phase.size();
        im /= phase.size();
        double r = sqrt(re * re + im * im);
        if (r > best_r) { best_r = r; best = s; best_delay = atan2(im, re) / (2 * M_PI * s); }
    }
    if (!best || best_r < GRID_MIN_CONCENTRATION) return false;
    const double delay = best_delay * period;
    if (fabs(delay) > GRID_MAX_ONSET_DELAY) return false;
    *out = { delay, best_r, best, (int) phase.size() };
    return true;
}

static void grid_measure_onset_delay(BeatGrid * g, const std::vector<double> & onsets) {
    OnsetDelay d;
    g->delay_known = true;
    if (grid_onset_delay(onsets, *g, &d)) {
        fprintf(stderr, "[ace-midi] onsets sit %+.1f ms off a 1/%d-beat grid (|R| = %.2f over %d onsets)\n",
                1000 * d.seconds, d.subdivision, d.concentration, d.n);
        g->onset_delay = d.seconds;
        g->subdivision = d.subdivision;
    } else {
        g->onset_delay = 0.0;
        g->subdivision = 0;
    }
}

// Each beat moved onto a parabola through its neighbours (Savitzky-Golay):
// divides the tracker's 20 ms frame jitter down, keeps steady accelerations.
static std::vector<double> grid_smoothed(const std::vector<double> & beats) {
    const int n = (int) beats.size();
    if (n < 4) return beats;
    const int width = std::min(GRID_SMOOTHING_WINDOW, n);
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) {
        int lo = std::min(std::max(0, i - width / 2), n - width);
        // normal equations for y = a + b x + c x^2, x = index - i; out = a
        double s[5] = { 0 }, t[3] = { 0 };
        for (int j = lo; j < lo + width; j++) {
            double x = j - i, p = 1;
            for (int k = 0; k < 5; k++) { s[k] += p; if (k < 3) t[k] += p * beats[j]; p *= x; }
        }
        double m[3][4] = { { s[0], s[1], s[2], t[0] }, { s[1], s[2], s[3], t[1] }, { s[2], s[3], s[4], t[2] } };
        for (int c = 0; c < 3; c++) {
            int piv = c;
            for (int r = c + 1; r < 3; r++) if (fabs(m[r][c]) > fabs(m[piv][c])) piv = r;
            for (int k = 0; k < 4; k++) std::swap(m[c][k], m[piv][k]);
            for (int r = 0; r < 3; r++) {
                if (r == c) continue;
                double f = m[r][c] / m[c][c];
                for (int k = c; k < 4; k++) m[r][k] -= f * m[c][k];
            }
        }
        out[i] = m[0][3] / m[0][0];
    }
    return out;
}

// scale times as many beats per second: doubling adds midpoints, halving keeps
// the half holding the beat nearest the downbeat
static std::vector<double> grid_rescaled(std::vector<double> beats, double scale, double downbeat) {
    while (scale > 1 && beats.size() >= 2) {
        std::vector<double> d;
        for (size_t i = 0; i + 1 < beats.size(); i++) { d.push_back(beats[i]); d.push_back(0.5 * (beats[i] + beats[i + 1])); }
        d.push_back(beats.back());
        beats = d;
        scale /= 2;
    }
    while (scale < 1 && beats.size() >= 4) {
        std::vector<double> h;
        for (size_t i = grid_nearest(beats, downbeat) % 2; i < beats.size(); i += 2) h.push_back(beats[i]);
        beats = h;
        scale *= 2;
    }
    return beats;
}

// The subdivision a steady grid's beats (laid out over the onsets) hold them on
static int grid_measure_subdivision(const std::vector<double> & onsets, BeatGrid g) {
    if (onsets.empty()) return 0;
    if (g.steady) {
        const double period = 60.0 / g.bpm;
        double mn = *std::min_element(onsets.begin(), onsets.end());
        double mx = *std::max_element(onsets.begin(), onsets.end());
        int first = (int) floor((mn - g.first_downbeat) / period), last = (int) ceil((mx - g.first_downbeat) / period);
        g.beats.clear();
        for (int k = first; k <= last; k++) g.beats.push_back(g.first_downbeat + period * k);
    }
    OnsetDelay d;
    return grid_onset_delay(onsets, g, &d) ? d.subdivision : 0;
}

// midi.py resolve_grid: the detected grid (or none) with the export options'
// corrections applied. steady=false means g.beats are the grid (tempo map).
static BeatGrid grid_resolve(const BeatGrid * detected, const GridExport & o, const std::vector<double> & onsets) {
    const bool tempo_known = detected || o.bpm > 0;
    BeatGrid   g           = detected ? *detected : BeatGrid();
    if (!g.delay_known) grid_measure_onset_delay(&g, onsets);

    const bool follow = o.bpm <= 0 && g.beats.size() >= 2 && (o.tempo_map >= 0 ? o.tempo_map == 1 : !g.steady);
    const double scale = o.beat_scale;
    int bpb = o.beats_per_bar;
    if (!bpb && g.beats_per_bar) {
        double s = g.beats_per_bar * scale;
        bpb = (s >= 1 && s == floor(s)) ? (int) s : g.beats_per_bar;
    }
    const double fd = o.has_downbeat ? o.first_downbeat : g.first_downbeat;

    if (follow) {
        std::vector<double> beats = grid_rescaled(grid_smoothed(g.beats), scale, fd);
        g.bpm            = grid_fit_tempo(beats).first;
        g.first_downbeat = beats[grid_nearest(beats, fd)];  // bar lines go on beats
        g.beats          = beats;
        if (scale != 1) g.subdivision = 0;
        g.steady = false;
    } else {
        const double bpm     = o.bpm > 0 ? o.bpm : g.bpm * scale;
        const bool   retimed = fabs(bpm - g.bpm) > 1e-9 * std::max(fabs(bpm), fabs(g.bpm));
        g.bpm            = bpm;
        g.first_downbeat = fd;
        if (retimed) { g.beats.clear(); g.subdivision = 0; }
        g.steady = true;
    }
    g.beats_per_bar = bpb;

    if (!tempo_known) g.subdivision = 0;
    else if (o.subdivision > 0) g.subdivision = o.subdivision;
    else if (o.quantize && !g.subdivision) {
        std::vector<double> shifted;
        for (double t : onsets) shifted.push_back(t - g.onset_delay);
        g.subdivision = grid_measure_subdivision(shifted, g);
    }
    return g;
}

// Where moments of the recording (seconds) fall in the score (beats from tick
// 0): linear between the points, the edge tempo continued beyond them.
struct Timeline {
    std::vector<double> times, positions;

    static double extrap(double x, const std::vector<double> & xp, const std::vector<double> & fp) {
        const size_t n = xp.size();
        if (x < xp[0]) return fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]);
        if (x > xp[n - 1]) return fp[n - 1] + (x - xp[n - 1]) * (fp[n - 1] - fp[n - 2]) / (xp[n - 1] - xp[n - 2]);
        return grid_interp(x, xp, fp);
    }
    double beats_at(double s) const { return extrap(s, times, positions); }
    double seconds_at(double b) const { return extrap(b, positions, times); }
    void   shift(double beats) { for (double & p : positions) p += beats; }

    // (tick, microseconds per beat) wherever the tempo changes, from tick 0
    std::vector<std::pair<int, int>> tempo_changes(int tpb) const {
        std::vector<std::pair<int, int>> ch;
        for (size_t i = 0; i + 1 < times.size(); i++) {
            int tempo = (int) nearbyint((times[i + 1] - times[i]) / (positions[i + 1] - positions[i]) * 1e6);
            int tick  = std::max(0, (int) nearbyint(positions[i] * tpb));
            if (ch.empty()) { ch.push_back({ 0, tempo }); continue; }
            if (tempo == ch.back().second) continue;
            if (tick == ch.back().first) ch.back().second = tempo;
            else ch.push_back({ tick, tempo });
        }
        return ch;
    }

    // Keep only the bar lines where the tempo must change for no beat between
    // them to sit more than tol seconds off the line through them.
    Timeline simplified(double tol, int bar) const {
        const size_t last = times.size() - 1;
        std::vector<size_t> cand;
        for (size_t i = 1; i < last; i++) if (grid_pymod(positions[i], bar) == 0) cand.push_back(i);
        cand.push_back(last);
        auto fits = [&](size_t a, size_t b) {
            for (size_t i = a; i <= b; i++) {
                double line = times[a] + (positions[i] - positions[a]) * (times[b] - times[a]) / (positions[b] - positions[a]);
                if (fabs(times[i] - line) > tol) return false;
            }
            return true;
        };
        std::vector<size_t> kept = { 0 };
        for (size_t i = 0; i < cand.size(); i++) {
            if (cand[i] == last || !fits(kept.back(), cand[i + 1])) kept.push_back(cand[i]);
        }
        Timeline t;
        for (size_t k : kept) { t.times.push_back(times[k]); t.positions.push_back(positions[k]); }
        return t;
    }
};

// The timeline the grid is written with, bar 1 at tick 0, delayed by whole
// bars until `earliest` (seconds) falls at or after tick 0.
static Timeline grid_timeline(const BeatGrid & g, double earliest) {
    Timeline t;
    if (g.steady) {
        const double period = 60.0 / g.bpm, start = -g.bar_offset(-earliest);
        t.times     = { start, start + period };
        t.positions = { 0.0, 1.0 };
        return t;
    }
    const double down = (double) grid_nearest(g.beats, g.first_downbeat);
    t.times = g.beats;
    for (size_t i = 0; i < g.beats.size(); i++) t.positions.push_back((double) i - down);
    const int bar = g.beats_per_bar ? g.beats_per_bar : 1;
    t = t.simplified(GRID_TEMPO_MAP_TOL, bar);
    t.shift(bar * ceil(-t.beats_at(earliest) / bar));
    return t;
}

// ---------------------------------------------------------------------------
// Self-test: synthetic grids with known answers (ace-midi --self-test-grid)
// ---------------------------------------------------------------------------

static int grid_self_test() {
    int fails = 0;
    auto check = [&](bool ok, const char * what) {
        printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) fails++;
    };
    // 3/4 at 100 BPM with one missed and one extra beat, downbeat on beat 2
    std::vector<double> beats, downs;
    for (int i = 0; i < 60; i++) {
        double t = 0.37 + i * 0.6;
        if (i == 20) continue;               // missed
        beats.push_back(t);
        if (i == 30) beats.push_back(t + 0.1);  // extra
        if (i % 3 == 1) downs.push_back(t);
    }
    BeatGrid g;
    check(grid_detect(beats, downs, &g), "detect");
    check(fabs(g.bpm - 100.0) < 1e-6 && g.steady, "100 BPM, steady after regularizing");
    check(g.beats_per_bar == 3, "meter 3");
    check(g.beats.size() == 60, "missed beat restored, extra dropped");

    BeatGrid unsure;
    check(grid_detect(beats, { beats[0], beats[3], beats[7] }, &unsure) && unsure.beats_per_bar == 4,
          "conflicting bar lengths default to 4/4");
    check(grid_detect(beats, {}, &unsure) && unsure.beats_per_bar == 4, "missing downbeats default to 4/4");
    check(!grid_detect({}, {}, &unsure), "no grid without tracked beats");

    // onsets 15 ms late on eighths
    std::vector<double> on;
    for (int i = 0; i < 100; i++) on.push_back(0.37 + i * 0.3 + 0.015);
    OnsetDelay d;
    // (1/4 and 1/6 fit an eighth grid equally well; only the delay is pinned)
    check(grid_onset_delay(on, g, &d) && fabs(d.seconds - 0.015) < 1e-3, "onset delay 15 ms");

    // steady timeline: the first downbeat lands on a bar line
    GridExport o;
    BeatGrid r = grid_resolve(&g, o, on);
    Timeline t = grid_timeline(r, 0.0);
    double p = t.beats_at(r.first_downbeat);
    check(fabs(p / 3 - nearbyint(p / 3)) < 1e-9, "downbeat on a bar line");
    for (int direction : { -1, 1 }) {
        GridExport shift;
        shift.has_downbeat = true;
        shift.first_downbeat = g.first_downbeat + direction * 60.0 / g.bpm;
        const auto moved = grid_timeline(grid_resolve(&g, shift, on), 0.0);
        const double bar_phase = grid_pymod(moved.beats_at(g.first_downbeat), 3.0);
        check(fabs(bar_phase - (direction < 0 ? 1.0 : 2.0)) < 1e-9, "downbeat shift moves bars one beat");
    }
    auto tc = t.tempo_changes(480);
    check(tc.size() == 1 && tc[0].second == 600000, "one tempo, 600000 us");

    // accelerating beats: tempo map, beats land on whole positions
    std::vector<double> acc, adown;
    double tt = 1.0;
    for (int i = 0; i < 64; i++) {
        acc.push_back(tt);
        if (i % 4 == 0) adown.push_back(tt);
        tt += 0.6 - i * 0.004;
    }
    BeatGrid a;
    check(grid_detect(acc, adown, &a) && !a.steady && a.beats_per_bar == 4, "accelerando: unsteady, 4/4");
    BeatGrid ar = grid_resolve(&a, o, {});
    Timeline at = grid_timeline(ar, 0.0);
    double worst = 0;
    for (size_t i = 0; i < acc.size(); i++) {
        double b = at.beats_at(ar.beats[i]);
        worst = std::max(worst, fabs(at.seconds_at(nearbyint(b)) - ar.beats[i]));
    }
    check(worst <= GRID_TEMPO_MAP_TOL + 1e-9, "tempo map within 25 ms of every beat");
    check(at.tempo_changes(480).size() > 1, "tempo map has changes");
    check(fabs(grid_pymod(at.beats_at(ar.first_downbeat), 4)) < 1e-6, "accelerando downbeat on a bar line");

    // halving keeps the downbeat's parity
    std::vector<double> half = grid_rescaled({ 0, 1, 2, 3, 4, 5, 6, 7 }, 0.5, 3.0);
    check(half.size() == 4 && half[0] == 1, "halving keeps the downbeat half");

    printf("%s (%d failed)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
