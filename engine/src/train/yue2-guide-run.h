#pragma once
// train/yue2-guide-run.h — `ace-train yue2-guide`: one audio file in, one
// source-guide file out (format and use: yue2/yue2-guide.h).
//
// Same chain as yue2-tokenize, minus the manifest: decode at 48 kHz, bring the
// track to the loudness yue2-preprocess gives training audio (so the
// tokenizer sees the level its codes were built from), mean to mono, resample
// to the MERT rate, MERT encode, then the tokenizer head's top-K per frame
// instead of its argmax. It lives in ace-train rather than ace-server because
// the tokenizer must run with TF32 off from the moment the CUDA context is
// created (yue2-tokenize-run.h explains why); ace-server's context is shared
// with everything else and never makes that promise.

#include "audio-resample.h"
#include "hot-step-fsutf8.h"
#include "train/preprocess-io.h"
#include "train/yue2-loudness.h"
#include "train/yue2-preprocess-run.h"  // yp_read_native_48k / yp_read_ffmpeg
#include "yue2/yue2-guide.h"
#include "yue2/yue2-mert.h"
#include "yue2/yue2-tok-head.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

struct Yue2GuideArgs {
    std::string audio;              // --audio <file>        (required)
    std::string out;                // --out <file.y2gd>     (required)
    std::string tok_path;           // --tok <yue2-tok-*.gguf> (or --models)
    std::string models_dir;         // --models <dir>
    std::string ffmpeg = "ffmpeg";  // --ffmpeg, used for anything but WAV/MP3
    int         topk   = 64;        // --topk K
    double      lufs   = -14.0;     // --lufs, 0 = keep the file's own level
};

static int yue2_guide_run(const Yue2GuideArgs & a) {
    const auto t0 = std::chrono::steady_clock::now();
    if (a.audio.empty() || a.out.empty()) {
        fprintf(stderr, "[yue2-guide] --audio <file> and --out <file> are required\n");
        return 2;
    }
    if (a.topk < 1 || a.topk > 1024) {
        fprintf(stderr, "[yue2-guide] --topk must be in 1..1024\n");
        return 2;
    }
    if (!yue2_tok_tf32_disabled()) {
        fprintf(stderr, "[yue2-guide] REFUSING: NVIDIA_TF32_OVERRIDE is not \"0\" (see yue2-tokenize)\n");
        return 1;
    }
    std::string tok_path = a.tok_path;
    if (tok_path.empty()) {
        if (a.models_dir.empty() || !yue2_tok_head_find(a.models_dir, &tok_path)) {
            fprintf(stderr, "[yue2-guide] no yue2-tok-*.gguf found (pass --tok or --models)\n");
            return 1;
        }
    }

    // ── decode ──
    const std::string ext = pm_ext_of(a.audio);
    int               T48 = 0;
    float *           planar = nullptr;
    std::string       err;
    if (ext == ".wav" || ext == ".mp3") {
        planar = yp_read_native_48k(a.audio, &T48);
    }
    if (!planar) {
        if (a.ffmpeg.empty()) {
            fprintf(stderr, "[yue2-guide] cannot decode %s natively and --ffmpeg is empty\n", a.audio.c_str());
            return 1;
        }
        const std::string tmp = a.out + ".decode.wav";
        planar                = yp_read_ffmpeg(a.ffmpeg, a.audio, tmp, 48000, &T48, &err);
        std::remove(tmp.c_str());
        if (!planar) {
            fprintf(stderr, "[yue2-guide] decode failed: %s\n", err.c_str());
            return 1;
        }
    }
    if (T48 <= 0) {
        free(planar);
        fprintf(stderr, "[yue2-guide] %s decoded to no samples\n", a.audio.c_str());
        return 1;
    }
    if (a.lufs != 0.0) {
        double lufs = 0.0;
        bool   capped = false;
        const double gain = yue2_loudness_gain_db(planar, T48, a.lufs, -1.0, &lufs, &capped);
        yue2_apply_gain_db(planar, T48, gain);
        fprintf(stderr, "[yue2-guide] loudness %.1f LUFS -> gain %+.2f dB%s\n", lufs, gain, capped ? " (peak-capped)" : "");
    }
    std::vector<float> mono((size_t) T48);
    for (int i = 0; i < T48; i++) mono[(size_t) i] = 0.5f * (planar[i] + planar[T48 + i]);
    free(planar);

    // ── models ──
    static Yue2MertModel m;
    if (!yue2_mert_load(&m, tok_path, /*want_blocks=*/true, &err)) {
        fprintf(stderr, "[yue2-guide] MERT load failed: %s\n", err.c_str());
        return 1;
    }
    static Yue2TokHead head;
    if (!yue2_tok_head_load(tok_path, &head, &err)) {
        fprintf(stderr, "[yue2-guide] tokenizer head load failed: %s\n", err.c_str());
        yue2_mert_free(&m);
        return 1;
    }
    const int MERT_SR = (int) m.cfg.sample_rate;
    int       n24     = 0;
    float *   r       = audio_resample(mono.data(), T48, 48000, MERT_SR, 1, &n24);
    if (!r || n24 <= 0) {
        free(r);
        fprintf(stderr, "[yue2-guide] resample 48000 -> %d Hz failed\n", MERT_SR);
        yue2_tok_head_free(&head);
        yue2_mert_free(&m);
        return 1;
    }
    std::vector<float> pcm(r, r + n24);
    free(r);

    // ── encode ──
    Yue2MertEncGraph      g;
    Yue2MertEncodeOptions opt;
    Yue2MertEncodeResult  res;
    Yue2Guide             guide;
    bool ok = yue2_mert_encode(m, &g, pcm.data(), (int64_t) pcm.size(), opt, &res, &err);
    if (ok && m.cfg.level_band) {
        fprintf(stderr, "[yue2-guide] tokenizer level band [%.1f, %.1f] dBFS -> gain %+.2f dB\n", m.cfg.level_low,
                m.cfg.level_high, res.level_gain_db);
    }
    if (ok) {
        guide.T = res.T25;
        guide.K = a.topk;
        ok      = yue2_tok_head_predict_topk(&head, res.feat.data(), res.T25, a.topk, &guide.ids, &guide.lp, &err);
    }
    yue2_tok_head_free(&head);
    yue2_mert_free(&m);
    if (!ok) {
        fprintf(stderr, "[yue2-guide] encode failed: %s\n", err.c_str());
        return 1;
    }
    if (!yue2_guide_write(a.out, guide, &err)) {
        fprintf(stderr, "[yue2-guide] %s\n", err.c_str());
        return 1;
    }
    double top1 = 0.0;
    for (int64_t f = 0; f < guide.T; f++) top1 += std::exp((double) guide.lp[(size_t) (f * guide.K)]);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[yue2-guide] wrote %s: %lld frames (%.1f s) x top-%d, mean top-1 prob %.3f, %.1f s\n",
            a.out.c_str(), (long long) guide.T, guide.T / 25.0, a.topk, guide.T ? top1 / (double) guide.T : 0.0, sec);
    return 0;
}
