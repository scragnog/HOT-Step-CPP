#pragma once
// chordnet/inference.h: audio to contiguous chord segments in seconds, as
// MuLaCover's ChordTranscriber does before its beat rounding (which stays with
// the caller): mono 22,050 Hz, tuning, hybrid CQT, crop, five folds, mean of
// the six head probabilities, Viterbi, segments. CPU only.

#include <chrono>
#include <string>
#include <vector>

#include "backend.h"
#include "chordnet/decode.h"
#include "chordnet/features.h"
#include "chordnet/model.h"
#include "chordnet/weights.h"

namespace chordnet {

struct Result {
    std::vector<float>   wav;          // mono 22,050 Hz
    double               tuning = 0.0; // bins (1/36 octave)
    int                  T      = 0;   // frames
    std::vector<float>   mag;          // [T][288] |hybrid CQT|
    std::vector<float>   x;            // [T][252] network input
    FoldOut              fold[NFOLD];
    std::vector<float>   mean[NHEAD];  // [T][n] fold means
    Decoded              dec;
    std::vector<Segment> segs;
    double ms_features = 0, ms_model = 0, ms_decode = 0;
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// librosa.load(sr=22050, mono=True) after the caller's channel mean.
static std::vector<float> to_model_rate(const std::vector<float> & mono, int sr) {
    return sr == SR ? mono : resample(mono, sr, SR, false);
}

// The build loads ggml backends as modules, so the CPU backend comes from the
// registry; the modules are loaded here only if nothing else has loaded them.
static ggml_backend_t cpu_backend(int n_threads) {
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) ggml_backend_load_all();
    return cpu_backend_new(n_threads);
}

static void run_fold(ggml_backend_t be, const Fold & f, const float * x, int T, FoldOut & out) {
    std::vector<float> proj[2];
    run_graph(be, f, x, T, false, out, proj);
    run_lstm(f, proj, T, out.lstm);
    run_heads(f, out.lstm, T, out);
}

// np.mean over the folds of float32 arrays: a float32 running sum, then / 5.
static void fold_mean(const FoldOut f[NFOLD], int T, std::vector<float> mean[NHEAD]) {
    for (int h = 0; h < NHEAD; h++) {
        const size_t n = (size_t) T * HEAD_DIM[h];
        mean[h].assign(n, 0.0f);
        for (size_t i = 0; i < n; i++) {
            float s = f[0].head[h][i];
            for (int k = 1; k < NFOLD; k++) s += f[k].head[h][i];
            mean[h][i] = s / (float) NFOLD;
        }
    }
}

static std::vector<std::string> frame_labels(const Decoded & d) {
    std::vector<std::string> out;
    for (int i : d.label) out.push_back(d.names[i]);
    return out;
}

static void transcribe(const Weights & w, const std::vector<Template> & tpl, const std::vector<float> & mono, int sr,
                       int n_threads, Result & r) {
    double t0 = now_ms();
    r.wav    = to_model_rate(mono, sr);
    r.tuning = estimate_tuning(r.wav);
    r.mag    = hybrid_cqt(r.wav, r.tuning, &r.T);
    r.x      = crop(r.mag, r.T);
    double t1 = now_ms();
    ggml_backend_t be = cpu_backend(n_threads);
    for (int k = 0; k < NFOLD; k++) run_fold(be, w.fold[k], r.x.data(), r.T, r.fold[k]);
    ggml_backend_free(be);
    double t2 = now_ms();
    fold_mean(r.fold, r.T, r.mean);
    r.dec  = viterbi(tpl, r.mean, r.T);
    r.segs = segments(frame_labels(r.dec));
    double t3 = now_ms();
    r.ms_features = t1 - t0;
    r.ms_model    = t2 - t1;
    r.ms_decode   = t3 - t2;
}

}  // namespace chordnet
