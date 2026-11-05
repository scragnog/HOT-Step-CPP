#pragma once
// minimax/mm3-refiner.h — the MM3 latent refiner as an audio -> audio stage.
//
// HOT-Step file. terminusresearch minimax-music3-latent-refiner v0.10 end to end,
// the chain MiniMaxMusic3RefinerPipeline.refine_latents + decode_latents run, plus
// the loudness convention of the runner we validated it with (run_refine.py):
//
//   audio (any rate, mono or stereo) -> 44.1 kHz stereo, zero-padded to the
//   512-sample DAV hop -> windows (30 s, 2 s overlap by default, or one dense
//   sequence) -> per window: DAV encode, MERT conditioning, CLAP style vector,
//   deterministic bridge -> linear-fade overlap stitch -> DAV decode -> trim ->
//   match the input's integrated loudness (BS.1770, pyloudnorm's K-weighting)
//   -> clip to [-1, 1] -> back to the input rate -> mix with the input.
//
// Residency. The four modules (DAV encoder, MERT, refiner DiT, DAV decoder) go
// through ModelStore. Under EVICT_STRICT only one GPU module is resident at a
// time, so the run is phased: every window is encoded, then every window is
// conditioned, then every window is bridged, then the stitched latents are
// decoded; each module is required once and released before the next. Between
// phases the per-window state lives on the host (the MERT states are the bulk:
// 13 x frames x 768 floats, about 100 MB per 30 s window). Under EVICT_NEVER the
// modules simply stay resident for the next job. The store handles are RAII, so
// success, error and cancel all release.
//
// Style. The refiner takes a 512-d CLAP vector per window. Nothing here makes it:
// the caller passes a Mm3StyleFn, and a run without one fails up front, never with
// zero or constant vectors. The probe feeds the fixture vectors; the production
// producer (the CLAP port, or the frozen corpus mean) is wired in separately.
//
// Windowing, overlap weights, hop rounding and trim follow pipeline.py exactly
// (python round() is half-to-even, so std::nearbyint; the fades are torch.linspace
// in float32). mix == 0 returns the input untouched and loads nothing.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../audio-resample.h"
#include "../model-store.h"
#include "mm3-dav-encode.h"
#include "mm3-refiner-clap.h"
// mm3-dit-graph.h (the MM3 backend's flow DiT, in the server's include set) #defines
// MM3_DIT_MAX_NODES; the refiner DiT header declares a constant of that name. Hide the macro
// while that header is read. (Renaming the constant there would end this.)
#pragma push_macro("MM3_DIT_MAX_NODES")
#undef MM3_DIT_MAX_NODES
#include "mm3-refiner-dit.h"
#pragma pop_macro("MM3_DIT_MAX_NODES")
#include "mm3-refiner-mert.h"
#include "mm3-vocoder-graph.h"

static constexpr int     MM3_REF_SR         = 44100;
static constexpr int     MM3_REF_HOP        = 512;
static constexpr int64_t MM3_REF_MIN_SAMPLES = 22050;  // 0.5 s: the shortest clip the parity fixtures cover
// Direct mode runs one dense attention over 2L tokens: 12 heads x (2L)^2 F32 scores
// (about 190 L^2 bytes, plus the softmax output), so it is capped. 5168 frames is 60 s.
static constexpr int64_t MM3_REF_DIRECT_MAX_FRAMES = 5168;

struct Mm3RefinerParams {
    int    steps           = 32;    // 1..64
    double window_seconds  = 30.0;  // 0 = direct: the whole clip as one sequence
    double overlap_seconds = 2.0;   // >= 0, shorter than the window
    float  mix             = 1.0f;  // 0 = input unchanged (loads nothing), 1 = refined
    bool   capture         = false; // keep intermediates in the result (parity probe)
};

struct Mm3RefinerPaths {
    std::string enc;      // mm3-enc-*.gguf
    std::string mert;     // mm3-refiner-mert-*.gguf
    std::string refiner;  // mm3-refiner-refiner-*.gguf
    std::string voc;      // mm3-voc-*.gguf
    std::string clap;     // mm3-refiner-clap-*.gguf; read only when no Mm3StyleFn is passed
};

// Optional override of the CLAP style producer (the parity probe feeds fixture vectors,
// the job's test hook a fixed one). Called once per window with the window's 44.1 kHz
// stereo audio ([2][n] planar) so a producer can crop what it needs; writes the 512-d vector.
using Mm3StyleFn = std::function<bool(int window, const float * planar, int64_t n, float * out512, std::string * err)>;

enum class Mm3RefinerPhase { LOAD, CONDITION, BRIDGE, DECODE, OUTPUT };

struct Mm3RefinerHooks {
    // step/total: CONDITION counts windows (once for the DAV encode pass, once for MERT),
    // BRIDGE counts DiT forwards over windows x steps. Other phases report 0/0.
    std::function<void(Mm3RefinerPhase phase, int step, int total)> progress;
    std::function<bool()>                                           cancelled;  // true = stop
};

struct Mm3RefinerResult {
    std::vector<float> audio;  // planar stereo [2][n] at sample_rate; mono input comes back as dual mono
    int                sample_rate = 0;
    int64_t            n           = 0;
    int                windows     = 0;
    int64_t            frames      = 0;
    double             lufs_in     = -INFINITY;  // the input and the raw refined audio, at 44.1 kHz
    double             lufs_out    = -INFINITY;
    double             gain_db     = 0.0;        // applied loudness match; 0 when either is unmeasurable
    // capture only:
    std::vector<std::vector<float>> window_dav;      // per window [128][L], DAV encoder output
    std::vector<float>              latents;         // stitched [128][frames]
    std::vector<float>              raw_audio;       // decoded, trimmed, before the loudness match; planar [2][n44]
};

// ── Windowing (pipeline.py window_starts and refine_latents) ────────────────

struct Mm3RefinerPlan {
    int64_t              window_samples = 0;
    int64_t              overlap_samples = 0;
    std::vector<int64_t> starts;       // sample starts, all multiples of the hop
    int64_t              frames = 0;   // total latent frames
    bool                 single() const { return starts.size() == 1; }
};

// total_samples is the hop-padded length. Returns 0 when valid, else the HTTP status
// the route should answer with: 400 for a bad parameter, 413 for direct mode past the limit.
static int mm3_refiner_plan(int64_t total_samples, const Mm3RefinerParams & p, Mm3RefinerPlan * plan,
                            std::string * err) {
    if (p.steps < 1 || p.steps > 64) {
        *err = "steps must be 1 to 64";
        return 400;
    }
    if (!(p.window_seconds >= 0.0) || !(p.overlap_seconds >= 0.0) || !std::isfinite(p.window_seconds) ||
        !std::isfinite(p.overlap_seconds)) {
        *err = "window and overlap must be finite and not negative";
        return 400;
    }
    if (!(p.mix >= 0.0f && p.mix <= 1.0f)) {
        *err = "mix must be 0 to 1";
        return 400;
    }
    if (total_samples < MM3_REF_MIN_SAMPLES) {
        *err = "audio is shorter than 0.5 s";
        return 400;
    }
    plan->starts.clear();
    plan->frames = total_samples / MM3_REF_HOP;
    if (p.window_seconds == 0.0) {
        if (plan->frames > MM3_REF_DIRECT_MAX_FRAMES) {
            *err = "direct mode is limited to " + std::to_string(MM3_REF_DIRECT_MAX_FRAMES) + " frames (60 s); this clip has " +
                   std::to_string(plan->frames);
            return 413;
        }
        plan->window_samples  = total_samples;
        plan->overlap_samples = 0;
        plan->starts          = { 0 };
        return 0;
    }
    // python round() is half-to-even; nearbyint is under the default rounding mode. The hop
    // counts are clamped as doubles before the int64 conversion (a window of 1e300 s must not
    // reach an out-of-range cast); 1e12 hops is far past any real clip and any limit below.
    auto hops = [](double seconds) { return (int64_t) std::min(std::nearbyint(seconds * MM3_REF_SR / MM3_REF_HOP), 1e12); };
    const int64_t w = std::max<int64_t>(1, hops(p.window_seconds)) * MM3_REF_HOP;
    const int64_t o = hops(p.overlap_seconds) * MM3_REF_HOP;
    if (o >= w) {
        *err = "overlap must be shorter than the window";
        return 400;
    }
    // The DiT attends densely over 2L tokens whichever mode picked L, so L is bounded here too:
    // the whole clip when it fits one window, else the window itself.
    const int64_t seq_frames = total_samples <= w ? plan->frames : w / MM3_REF_HOP;
    if (seq_frames > MM3_REF_DIRECT_MAX_FRAMES) {
        *err = "each window is limited to " + std::to_string(MM3_REF_DIRECT_MAX_FRAMES) + " frames (60 s) because attention grows with its square; this request would use " +
               std::to_string(seq_frames) + ". Use a shorter window";
        return 413;
    }
    plan->window_samples  = w;
    plan->overlap_samples = o;
    if (total_samples <= w) {
        plan->starts = { 0 };
        return 0;
    }
    const int64_t stride = w - o;
    for (int64_t s = 0; s <= total_samples - w; s += stride) {
        plan->starts.push_back(s);
    }
    int64_t final_start = total_samples - w;
    final_start -= final_start % MM3_REF_HOP;
    if (final_start > plan->starts.back()) {
        plan->starts.push_back(final_start);
    }
    return 0;
}

// torch.linspace(a, b, n) in float32 (two-sided formula, as in mm3_dit_schedule).
static std::vector<float> mm3_refiner_linspace(float a, float b, int n) {
    std::vector<float> s((size_t) n);
    const float        step = (b - a) / (float) (n - 1);
    for (int i = 0; i < n; i++) {
        s[(size_t) i] = i < n / 2 ? a + step * (float) i : b - step * (float) (n - 1 - i);
    }
    return s;
}

// The overlap blend of one window (pipeline.py refine_latents). `available` frames of
// window `index` start at start_frame; prev_end/next_start are the neighbours' edges in frames.
static std::vector<float> mm3_refiner_blend(const Mm3RefinerPlan & plan, size_t index, int64_t available) {
    const int64_t      start_frame  = plan.starts[index] / MM3_REF_HOP;
    const int64_t      chunk_frames = (plan.window_samples + MM3_REF_HOP - 1) / MM3_REF_HOP;
    std::vector<float> blend((size_t) available, 1.0f);
    if (index > 0) {
        const int64_t previous_end = plan.starts[index - 1] / MM3_REF_HOP + chunk_frames;
        const int64_t left         = std::min<int64_t>(available, std::max<int64_t>(0, previous_end - start_frame));
        if (left > 0) {
            const std::vector<float> ramp = mm3_refiner_linspace(0.0f, 1.0f, (int) left + 2);
            for (int64_t i = 0; i < left; i++) {
                blend[(size_t) i] = ramp[(size_t) i + 1];
            }
        }
    }
    if (index + 1 < plan.starts.size()) {
        const int64_t next_start = plan.starts[index + 1] / MM3_REF_HOP;
        const int64_t right      = std::min<int64_t>(available, std::max<int64_t>(0, start_frame + available - next_start));
        if (right > 0) {
            const std::vector<float> fade = mm3_refiner_linspace(1.0f, 0.0f, (int) right + 2);
            for (int64_t i = 0; i < right; i++) {
                float & b = blend[(size_t) (available - right + i)];
                b         = std::min(b, fade[(size_t) i + 1]);
            }
        }
    }
    return blend;
}

// Overlap-add of the windows' latents ([128][L] each) into [128][frames], then the
// division by the summed weights (refine_latents' accumulated / weights).
struct Mm3RefinerStitch {
    const Mm3RefinerPlan & plan;
    std::vector<float>     latents, weights;

    explicit Mm3RefinerStitch(const Mm3RefinerPlan & p)
        : plan(p), latents((size_t) (128 * p.frames), 0.0f), weights((size_t) p.frames, 0.0f) {}

    void add(size_t w, const float * gen, int64_t L) {
        const int64_t            start_frame = plan.starts[w] / MM3_REF_HOP;
        const int64_t            available   = std::min<int64_t>(L, plan.frames - start_frame);
        const std::vector<float> blend       = mm3_refiner_blend(plan, w, available);
        for (int c = 0; c < 128; c++) {
            float * acc = latents.data() + (int64_t) c * plan.frames + start_frame;
            for (int64_t i = 0; i < available; i++) {
                acc[i] += gen[(size_t) (c * L + i)] * blend[(size_t) i];
            }
        }
        for (int64_t i = 0; i < available; i++) {
            weights[(size_t) (start_frame + i)] += blend[(size_t) i];
        }
    }

    bool finish(std::string * err) {
        for (float wt : weights) {
            if (wt == 0.0f) {
                *err = "window overlap left uncovered latent frames";
                return false;
            }
        }
        for (int c = 0; c < 128; c++) {
            float * acc = latents.data() + (int64_t) c * plan.frames;
            for (int64_t i = 0; i < plan.frames; i++) {
                acc[i] /= weights[(size_t) i];
            }
        }
        return true;
    }
};

// ── Loudness (pyloudnorm.Meter(rate).integrated_loudness, K-weighting) ───────

struct Mm3Biquad {
    double b[3], a[3];
};

// pyloudnorm IIRfilter: the RBJ cookbook high shelf and high pass, normalised by a0.
static Mm3Biquad mm3_refiner_biquad(bool shelf, double G, double Q, double fc, double rate) {
    const double A = std::pow(10.0, G / 40.0), w0 = 2.0 * 3.14159265358979323846 * (fc / rate), cw = std::cos(w0), alpha = std::sin(w0) / (2.0 * Q);
    double       b0, b1, b2, a0, a1, a2;
    if (shelf) {
        const double sA = 2.0 * std::sqrt(A) * alpha;
        b0 = A * ((A + 1) + (A - 1) * cw + sA);
        b1 = -2 * A * ((A - 1) + (A + 1) * cw);
        b2 = A * ((A + 1) + (A - 1) * cw - sA);
        a0 = (A + 1) - (A - 1) * cw + sA;
        a1 = 2 * ((A - 1) - (A + 1) * cw);
        a2 = (A + 1) - (A - 1) * cw - sA;
    } else {
        b0 = (1 + cw) / 2;
        b1 = -(1 + cw);
        b2 = (1 + cw) / 2;
        a0 = 1 + alpha;
        a1 = -2 * cw;
        a2 = 1 - alpha;
    }
    return { { b0 / a0, b1 / a0, b2 / a0 }, { 1.0, a1 / a0, a2 / a0 } };
}

// scipy lfilter (direct form II transposed); pyloudnorm stores each stage back
// into a float32 array, so the result is rounded to float like it.
static void mm3_refiner_filter(const Mm3Biquad & f, std::vector<float> * x) {
    double z1 = 0, z2 = 0;
    for (float & v : *x) {
        const double in  = v;
        const double out = f.b[0] * in + z1;
        z1               = f.b[1] * in - f.a[1] * out + z2;
        z2               = f.b[2] * in - f.a[2] * out;
        v                = (float) out;
    }
}

// Integrated loudness of planar stereo [2][n] at `rate`; -inf when unmeasurable
// (silence, or shorter than one 400 ms block).
static double mm3_refiner_lufs(const float * planar, int64_t n, double rate) {
    const double T_g = 0.400, step = 0.25;
    if ((double) n < T_g * rate) {
        return -INFINITY;
    }
    const Mm3Biquad shelf = mm3_refiner_biquad(true, 4.0, 1.0 / std::sqrt(2.0), 1500.0, rate);
    const Mm3Biquad hp    = mm3_refiner_biquad(false, 0.0, 0.5, 38.0, rate);
    const double    T      = (double) n / rate;
    const int64_t   blocks = (int64_t) (std::nearbyint((T - T_g) / (T_g * step)) + 1);
    std::vector<double> z[2];
    for (int ch = 0; ch < 2; ch++) {
        std::vector<float> x(planar + ch * n, planar + (ch + 1) * n);
        mm3_refiner_filter(shelf, &x);
        mm3_refiner_filter(hp, &x);
        z[ch].resize((size_t) blocks);
        for (int64_t j = 0; j < blocks; j++) {
            const int64_t l = (int64_t) (T_g * ((double) j * step) * rate);
            const int64_t u = std::min<int64_t>(n, (int64_t) (T_g * ((double) j * step + 1.0) * rate));
            double        s = 0.0;
            for (int64_t i = l; i < u; i++) {
                s += (double) x[(size_t) i] * (double) x[(size_t) i];
            }
            z[ch][(size_t) j] = s / (T_g * rate);
        }
    }
    std::vector<double> loud((size_t) blocks);
    for (int64_t j = 0; j < blocks; j++) {
        loud[(size_t) j] = -0.691 + 10.0 * std::log10(z[0][(size_t) j] + z[1][(size_t) j]);
    }
    auto mean_z = [&](double abs_gate, double rel_gate, bool rel, double * out) {
        double  sum[2] = { 0, 0 };
        int64_t cnt    = 0;
        for (int64_t j = 0; j < blocks; j++) {
            const double l = loud[(size_t) j];
            if (rel ? (l > rel_gate && l > abs_gate) : (l >= abs_gate)) {
                sum[0] += z[0][(size_t) j];
                sum[1] += z[1][(size_t) j];
                cnt++;
            }
        }
        if (cnt == 0) {
            return false;
        }
        out[0] = sum[0] / (double) cnt;
        out[1] = sum[1] / (double) cnt;
        return true;
    };
    double avg[2];
    if (!mean_z(-70.0, 0.0, false, avg)) {
        return -INFINITY;
    }
    const double gamma_r = -0.691 + 10.0 * std::log10(avg[0] + avg[1]) - 10.0;
    if (!mean_z(-70.0, gamma_r, true, avg)) {
        return -INFINITY;
    }
    return -0.691 + 10.0 * std::log10(avg[0] + avg[1]);
}

// ── The run ─────────────────────────────────────────────────────────────────

namespace mm3_refiner_detail {

struct Cancelled {};  // thrown across the phase lambdas, caught once in mm3_refiner_run

static std::vector<float> copy_window(const std::vector<float> & planar, int64_t total, int64_t start, int64_t n) {
    std::vector<float> w((size_t) (2 * n));
    for (int ch = 0; ch < 2; ch++) {
        std::memcpy(w.data() + ch * n, planar.data() + ch * total + start, (size_t) n * sizeof(float));
    }
    return w;
}

static ModelKey key(ModelKind kind, const std::string & path) {
    ModelKey k{};
    k.kind          = kind;
    k.path          = path;
    k.adapter_scale = 1.0f;
    return k;
}

}  // namespace mm3_refiner_detail

// planar_in: stereo [2][n_in] at sr_in (the caller duplicates mono, as the package does).
static bool mm3_refiner_run(ModelStore * store, const Mm3RefinerPaths & paths, const Mm3RefinerParams & p,
                            const float * planar_in, int64_t n_in, int sr_in, const Mm3StyleFn & style,
                            const Mm3RefinerHooks & hooks, Mm3RefinerResult * out, std::string * err) {
    using namespace mm3_refiner_detail;
    *out         = Mm3RefinerResult{};
    out->sample_rate = sr_in;
    out->n           = n_in;

    // 44.1 kHz stereo, hop padded. n44 is the unpadded length the output is trimmed back to.
    std::vector<float> wav;
    int64_t            n44 = n_in;
    if (sr_in == MM3_REF_SR) {
        wav.assign(planar_in, planar_in + 2 * n_in);
    } else {
        int         n_rs = 0;
        float *     rs   = audio_resample(planar_in, (int) n_in, sr_in, MM3_REF_SR, 2, &n_rs);
        if (!rs) {
            *err = "resample to 44.1 kHz failed";
            return false;
        }
        wav.assign(rs, rs + 2 * (int64_t) n_rs);
        free(rs);
        n44 = n_rs;
    }
    const int64_t padded = (n44 + MM3_REF_HOP - 1) / MM3_REF_HOP * MM3_REF_HOP;
    {
        std::vector<float> pad((size_t) (2 * padded), 0.0f);
        for (int ch = 0; ch < 2; ch++) {
            std::memcpy(pad.data() + ch * padded, wav.data() + ch * n44, (size_t) n44 * sizeof(float));
        }
        wav.swap(pad);
    }

    Mm3RefinerPlan plan;
    if (mm3_refiner_plan(padded, p, &plan, err) != 0) {
        return false;
    }
    if (p.mix == 0.0f) {  // exact bypass: nothing loads, the input comes back as it is
        out->audio.assign(planar_in, planar_in + 2 * n_in);
        return true;
    }
    if (!style && paths.clap.empty()) {
        *err = "no style source: the refiner needs a CLAP style vector per window and no CLAP file is given";
        return false;
    }
    // Fail on a missing or unreadable file before anything loads, naming its role.
    {
        const struct {
            const char *        role;
            const std::string * path;
        } files[] = { { "DAV encoder", &paths.enc }, { "MERT", &paths.mert }, { "refiner", &paths.refiner }, { "decoder", &paths.voc }, { "CLAP", &paths.clap } };
        for (const auto & f : files) {
            if (f.path == &paths.clap && style) {
                continue;  // an override supplies the vectors, the file is not read
            }
            GGUFModel gf = {};
            if (!gf_load(&gf, f.path->c_str())) {
                *err = std::string(f.role) + ": cannot read " + *f.path;
                return false;
            }
            gf_close(&gf);
        }
    }
    // A mono source (identical channels) stays mono: the refiner is not channel-symmetric, so its
    // two outputs differ slightly and would add a faint stereo image the recording never had.
    const bool mono_in = std::memcmp(wav.data(), wav.data() + padded, (size_t) n44 * sizeof(float)) == 0;
    const int  W       = (int) plan.starts.size();
    out->windows    = W;
    out->frames     = plan.frames;
    auto progress   = [&](Mm3RefinerPhase ph, int s, int t) {
        if (hooks.progress) {
            hooks.progress(ph, s, t);
        }
    };
    auto stop = [&]() {
        if (hooks.cancelled && hooks.cancelled()) {
            throw Cancelled{};
        }
    };

    std::vector<std::vector<float>> dav((size_t) W);     // [128][L] per window
    std::vector<std::vector<float>> layers((size_t) W);  // [13][L][768] per window, freed after its bridge
    std::vector<float>              latents;             // stitched [128][frames]

    try {
        // 1. DAV encode every window.
        {
            progress(Mm3RefinerPhase::LOAD, 0, 0);
            stop();
            MM3Enc * enc = store_require_mm3_enc(store, key(MODEL_MM3_ENC, paths.enc));
            if (!enc) {
                *err = "DAV encoder: " + store_mm3_last_error();
                return false;
            }
            ModelHandle h(store, enc);
            for (int w = 0; w < W; w++) {
                stop();
                progress(Mm3RefinerPhase::CONDITION, w, W);
                const int64_t start = plan.single() ? 0 : plan.starts[(size_t) w];
                const int64_t len   = plan.single() ? padded : plan.window_samples;
                const float * ch[2] = { wav.data() + start, wav.data() + padded + start };
                int64_t       L     = 0;
                if (!mm3_enc_encode(enc, ch, len, &dav[(size_t) w], &L, err)) {
                    return false;
                }
            }
            progress(Mm3RefinerPhase::CONDITION, W, W);
        }
        // 2. MERT conditioning for every window.
        {
            progress(Mm3RefinerPhase::LOAD, 0, 0);
            stop();
            ModelKey k = key(MODEL_MM3_REFINER_MERT, paths.mert);
            k.adapter_path = paths.refiner;
            Mm3Mert * mert = store_require_mm3_refiner_mert(store, k);
            if (!mert) {
                *err = "MERT: " + store_mm3_last_error();
                return false;
            }
            ModelHandle h(store, mert);
            for (int w = 0; w < W; w++) {
                stop();
                progress(Mm3RefinerPhase::CONDITION, w, W);
                const int64_t start = plan.single() ? 0 : plan.starts[(size_t) w];
                const int64_t len   = plan.single() ? padded : plan.window_samples;
                const std::vector<float> chunk = copy_window(wav, padded, start, len);
                Mm3MertResult            r;
                if (!mm3_mert_encode(mert, chunk.data(), len, len / MM3_REF_HOP, &r, err)) {
                    return false;
                }
                layers[(size_t) w] = std::move(r.normalized);
            }
            progress(Mm3RefinerPhase::CONDITION, W, W);
        }
        // 2b. One style vector per window: the CLAP tower, or the override. It loads and
        // releases on its own (the store holds one GPU module at a time), so the vectors
        // wait on the host like the MERT states.
        std::vector<float> styles((size_t) W * MM3_DIT_STYLE);
        {
            Mm3Clap *               clap = nullptr;
            std::unique_ptr<ModelHandle> h;
            if (!style) {
                progress(Mm3RefinerPhase::LOAD, 0, 0);
                stop();
                clap = store_require_mm3_refiner_clap(store, key(MODEL_MM3_REFINER_CLAP, paths.clap));
                if (!clap) {
                    *err = "CLAP: " + store_mm3_last_error();
                    return false;
                }
                h.reset(new ModelHandle(store, clap));
            }
            for (int w = 0; w < W; w++) {
                stop();
                progress(Mm3RefinerPhase::CONDITION, w, W);
                const int64_t start = plan.single() ? 0 : plan.starts[(size_t) w];
                const int64_t len   = plan.single() ? padded : plan.window_samples;
                const std::vector<float> chunk = copy_window(wav, padded, start, len);
                float *                  sv    = styles.data() + (size_t) w * MM3_DIT_STYLE;
                if (style) {
                    if (!style(w, chunk.data(), len, sv, err)) {
                        if (err->empty()) {
                            *err = "style source failed";
                        }
                        return false;
                    }
                } else {
                    Mm3ClapResult r;
                    if (!mm3_clap_encode(clap, chunk.data(), len, &r, err)) {
                        return false;
                    }
                    std::memcpy(sv, r.vector.data(), MM3_DIT_STYLE * sizeof(float));
                }
            }
            progress(Mm3RefinerPhase::CONDITION, W, W);
        }
        // 3. Bridge every window, stitching as each finishes.
        {
            progress(Mm3RefinerPhase::LOAD, 0, 0);
            stop();
            Mm3RefinerDit * dit = store_require_mm3_refiner_dit(store, key(MODEL_MM3_REFINER_DIT, paths.refiner));
            if (!dit) {
                *err = "refiner: " + store_mm3_last_error();
                return false;
            }
            ModelHandle h(store, dit);
            Mm3RefinerStitch stitch(plan);
            const int total_steps = W * p.steps;
            for (int w = 0; w < W; w++) {
                stop();
                const int64_t len   = plan.single() ? padded : plan.window_samples;
                const int64_t L     = len / MM3_REF_HOP;
                const float * sv    = styles.data() + (size_t) w * MM3_DIT_STYLE;
                const std::vector<float> x0 = mm3_dit_normalize(dit, dav[(size_t) w].data(), L);
                std::vector<float>       x;
                auto on_step = [&](int k, const float *, const float *) {
                    progress(Mm3RefinerPhase::BRIDGE, w * p.steps + k + 1, total_steps);
                    return !(hooks.cancelled && hooks.cancelled());
                };
                progress(Mm3RefinerPhase::BRIDGE, w * p.steps, total_steps);
                if (!mm3_dit_bridge(dit, x0.data(), layers[(size_t) w].data(), sv, L, p.steps, &x, on_step, err)) {
                    if (*err == "cancelled") {
                        throw Cancelled{};
                    }
                    return false;
                }
                std::vector<float>().swap(layers[(size_t) w]);  // host state is the bulk: free it now
                std::vector<float> gen = mm3_dit_denormalize(dit, x.data(), L);  // [128][L]
                if (plan.single()) {
                    latents = std::move(gen);
                    continue;
                }
                stitch.add((size_t) w, gen.data(), L);
            }
            if (!plan.single()) {
                if (!stitch.finish(err)) {
                    return false;
                }
                latents = std::move(stitch.latents);
            }
        }
        // 4. Decode the stitched latents.
        std::vector<float> decoded;
        {
            progress(Mm3RefinerPhase::LOAD, 0, 0);
            stop();
            Mm3RefVoc * voc = store_require_mm3_refiner_voc(store, key(MODEL_MM3_REFINER_VOC, paths.voc));
            if (!voc) {
                *err = "decoder: " + store_mm3_last_error();
                return false;
            }
            ModelHandle h(store, voc);
            progress(Mm3RefinerPhase::DECODE, 0, 0);
            if (!mm3_vocoder_decode_with(voc->model, &voc->graph, latents.data(), plan.frames, decoded, err)) {
                return false;
            }
        }
        stop();
        progress(Mm3RefinerPhase::OUTPUT, 0, 0);

        // 5. Trim to the original length, match loudness, clip.
        const int64_t dec_n = (int64_t) decoded.size() / 2;
        std::vector<float> refined((size_t) (2 * n44), 0.0f);
        for (int ch = 0; ch < 2; ch++) {
            std::memcpy(refined.data() + ch * n44, decoded.data() + ch * dec_n, (size_t) std::min(n44, dec_n) * sizeof(float));
        }
        if (mono_in) {
            for (int64_t i = 0; i < n44; i++) {
                refined[(size_t) i] = refined[(size_t) (n44 + i)] = 0.5f * (refined[(size_t) i] + refined[(size_t) (n44 + i)]);
            }
        }
        if (p.capture) {
            out->window_dav = dav;
            out->latents    = latents;
            out->raw_audio  = refined;
        }
        {
            // the meter needs the unpadded [2][n44] layout (wav is strided by `padded`)
            std::vector<float> dry((size_t) (2 * n44));
            for (int ch = 0; ch < 2; ch++) {
                std::memcpy(dry.data() + ch * n44, wav.data() + ch * padded, (size_t) n44 * sizeof(float));
            }
            out->lufs_in = mm3_refiner_lufs(dry.data(), n44, MM3_REF_SR);
        }
        out->lufs_out = mm3_refiner_lufs(refined.data(), n44, MM3_REF_SR);
        if (std::isfinite(out->lufs_in) && std::isfinite(out->lufs_out)) {
            out->gain_db = out->lufs_in - out->lufs_out;
            const float g = (float) std::pow(10.0, out->gain_db / 20.0);
            for (float & v : refined) {
                v *= g;
            }
        }
        for (float & v : refined) {
            v = std::min(1.0f, std::max(-1.0f, v));
        }

        // 6. Back to the input rate and length, then the mix.
        std::vector<float> wet;
        if (sr_in == MM3_REF_SR) {
            wet = std::move(refined);
        } else {
            int     n_rs = 0;
            float * rs   = audio_resample(refined.data(), (int) n44, MM3_REF_SR, sr_in, 2, &n_rs);
            if (!rs) {
                *err = "resample to the input rate failed";
                return false;
            }
            wet.assign((size_t) (2 * n_in), 0.0f);
            for (int ch = 0; ch < 2; ch++) {
                std::memcpy(wet.data() + ch * n_in, rs + (int64_t) ch * n_rs, (size_t) std::min<int64_t>(n_in, n_rs) * sizeof(float));
            }
            free(rs);
        }
        if (p.mix < 1.0f) {
            for (int64_t i = 0; i < 2 * n_in; i++) {
                wet[(size_t) i] = p.mix * wet[(size_t) i] + (1.0f - p.mix) * planar_in[i];
            }
        }
        out->audio = std::move(wet);
        return true;
    } catch (const Cancelled &) {
        *err = "cancelled";
        return false;
    }
}
