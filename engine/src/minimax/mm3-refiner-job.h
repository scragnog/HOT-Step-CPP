#pragma once
// minimax/mm3-refiner-job.h — POST /mm3/refine, the MM3 latent refiner as a job.
//
// HOT-Step file. Included MID-FILE by tools/hot-step-server.cpp (after minimax/mm3-job.h):
// it needs Job, job_create(), job_set_phase(), work_push(), json_error() and g_store,
// which that file defines. Registered by mm3_refiner_register_routes(svr, models_dir).
// It rides the generic job API, so progress, cancel and the result are GET/POST /job.
//
// Wire contract (frozen with slice 4, docs/dev/engine.md carries the long form):
//
//   POST /mm3/refine?steps=32&window=30&overlap=2&mix=1       body: WAV (or MP3) audio
//     steps    int 1..64                       window   seconds, 0 = direct (one dense sequence)
//     overlap  seconds >= 0, shorter than window     mix  0..1, 0 = input back unchanged
//     200 {"id"}            queued on the GPU worker FIFO (mix=0 completes at once, no queue)
//     400 {"error"}         empty or undecodable body, a parameter out of range or malformed
//     413 {"error"}         direct mode past MM3_REF_DIRECT_MAX_FRAMES
//     501 {"error"}         a pack file is missing (the message names it)
//   GET  /job?id=           status JSON; phase is queued, loading_refiner, refine_condition
//                           (window/windows, reported once each for the DAV encode pass, MERT and CLAP),
//                           refine_bridge (DiT forwards done / windows x steps), vae_decode,
//                           encoding_output; failed adds "error"
//   POST /job?id=&cancel=1  honoured between bridge steps and between phases
//   GET  /job?id=&result=1  audio/wav, float32 stereo at the input rate and frame count; 404 until done
//
// Files are read from <models>/mm3/: mm3-refiner-refiner-*.gguf, mm3-refiner-mert-*.gguf,
// mm3-refiner-clap-*.gguf, mm3-enc-*.gguf, mm3-voc-*.gguf (f32 / f16 first, else the first by name).
//
// Style. The CLAP audio tower (mm3-refiner-clap.h, through ModelStore) makes the 512-d style
// vector of every window. A missing CLAP file answers 501 like any other pack file; nothing
// falls back to made-up conditioning. HOTSTEP_MM3_REFINE_TEST_STYLE overrides the tower.

#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "mm3-refiner.h"

static std::string g_mm3_refiner_dir;  // <models>/mm3

// TEST ONLY override of the CLAP tower: HOTSTEP_MM3_REFINE_TEST_STYLE names a raw
// little-endian float32 file of 512 values used for every window (tools/mm3-refiner/check-integration.ps1
// writes one from a fixture). Unset, or anything else wrong with it, returns an empty function and the
// CLAP tower runs: a bad override never invents a vector.
static Mm3StyleFn mm3_refiner_style_override() {
    static const std::vector<float> test_style = [] {
        std::vector<float> v;
        if (const char * path = std::getenv("HOTSTEP_MM3_REFINE_TEST_STYLE")) {
            if (FILE * f = std::fopen(path, "rb")) {
                v.resize(MM3_DIT_STYLE);
                if (std::fread(v.data(), sizeof(float), v.size(), f) != v.size() || std::fgetc(f) != EOF) {
                    v.clear();
                }
                std::fclose(f);
            }
            fprintf(stderr, "[MM3-Refine] test style from %s: %s\n", path, v.empty() ? "unusable, ignored" : "loaded");
        }
        return v;
    }();
    if (test_style.empty()) {
        return nullptr;
    }
    return [](int, const float *, int64_t, float * out, std::string *) {
        std::memcpy(out, test_style.data(), test_style.size() * sizeof(float));
        return true;
    };
}

// The file in `dir` starting with `prefix` and ending .gguf: the preferred quant if present, else the first by name.
static std::string mm3_refiner_find(const std::string & dir, const std::string & prefix, const std::string & preferred) {
    namespace fs = std::filesystem;
    std::vector<std::string> names;
    std::error_code          ec;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.size() > prefix.size() + 5 && n.compare(0, prefix.size(), prefix) == 0 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
            names.push_back(n);
        }
    }
    std::sort(names.begin(), names.end());
    for (const auto & n : names) {
        if (n == prefix + preferred + ".gguf") {
            return dir + "/" + n;
        }
    }
    return names.empty() ? std::string() : dir + "/" + names.front();
}

// Resolves the pack files; on a miss says which one is absent. The CLAP file is not needed
// when the test override supplies the style.
static bool mm3_refiner_resolve(Mm3RefinerPaths * out, bool need_clap, std::string * err) {
    const std::string & d = g_mm3_refiner_dir;
    out->refiner          = mm3_refiner_find(d, "mm3-refiner-refiner-", "f32");
    out->mert             = mm3_refiner_find(d, "mm3-refiner-mert-", "f32");
    out->clap             = mm3_refiner_find(d, "mm3-refiner-clap-", "f32");
    out->enc              = mm3_refiner_find(d, "mm3-enc-", "f16");
    out->voc              = mm3_refiner_find(d, "mm3-voc-", "f16");
    const struct {
        const std::string * path;
        const char *        what;
    } need[] = { { &out->refiner, "mm3-refiner-refiner-*.gguf" },
                 { &out->mert, "mm3-refiner-mert-*.gguf" },
                 { &out->clap, "mm3-refiner-clap-*.gguf" },
                 { &out->enc, "mm3-enc-*.gguf" },
                 { &out->voc, "mm3-voc-*.gguf" } };
    for (const auto & n : need) {
        if (n.path->empty() && (n.path != &out->clap || need_clap)) {
            *err = std::string("latent refiner pack incomplete: no ") + n.what + " in " + d;
            return false;
        }
    }
    return true;
}

// Strict number parsing for query parameters: the whole string must be the number.
static bool mm3_refiner_num(const httplib::Request & req, const char * key, double * v) {
    if (!req.has_param(key)) {
        return true;
    }
    const std::string s = req.get_param_value(key);
    char *            end = nullptr;
    const double      x   = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(x)) {
        return false;
    }
    *v = x;
    return true;
}

static void mm3_refiner_handle(const httplib::Request & req, httplib::Response & res) {
    auto bad = [&](int status, const std::string & msg) {  // json_error() is defined after this header's include point
        yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
        yyjson_mut_val * root = yyjson_mut_obj(doc);
        yyjson_mut_doc_set_root(doc, root);
        yyjson_mut_obj_add_strn(doc, root, "error", msg.c_str(), msg.size());
        char * json = yyjson_mut_write(doc, 0, NULL);
        yyjson_mut_doc_free(doc);
        res.status = status;
        res.set_content(json ? json : "{}", "application/json");
        free(json);
    };

    double steps = 32, window = 30, overlap = 2, mix = 1;
    if (!mm3_refiner_num(req, "steps", &steps) || !mm3_refiner_num(req, "window", &window) ||
        !mm3_refiner_num(req, "overlap", &overlap) || !mm3_refiner_num(req, "mix", &mix)) {
        bad(400, "steps, window, overlap and mix must be numbers");
        return;
    }
    if (steps != std::floor(steps) || steps < 1 || steps > 64) {
        bad(400, "steps must be a whole number from 1 to 64");
        return;
    }
    Mm3RefinerParams p;
    p.steps           = (int) steps;
    p.window_seconds  = window;
    p.overlap_seconds = overlap;
    p.mix             = (float) mix;
    if (req.body.empty()) {
        bad(400, "Empty body (expected WAV audio)");
        return;
    }

    int     T = 0, sr = 0;
    float * planar = audio_read_buf((const uint8_t *) req.body.data(), req.body.size(), &T, &sr);
    if (!planar || T <= 0 || sr <= 0) {
        free(planar);
        bad(400, "Failed to decode audio");
        return;
    }
    std::vector<float> audio(planar, planar + 2 * (size_t) T);
    free(planar);

    // Validate against the length the pipeline will see (44.1 kHz, hop padded).
    const int64_t n44    = sr == MM3_REF_SR ? T : (int64_t) ((double) T * ((double) MM3_REF_SR / (double) sr));
    const int64_t padded = (n44 + MM3_REF_HOP - 1) / MM3_REF_HOP * MM3_REF_HOP;
    Mm3RefinerPlan plan;
    std::string    err;
    if (const int code = mm3_refiner_plan(padded, p, &plan, &err)) {
        bad(code, err);
        return;
    }

    if (p.mix == 0.0f) {  // the input back unchanged: nothing loads, so nothing waits for the GPU worker
        auto job         = job_create();
        job->result_body = audio_encode_wav(audio.data(), T, sr, WAV_F32);
        job->result_mime = "audio/wav";
        job_set_phase(*job, JobPhase::DONE);
        job->status.store(1);
        res.set_content("{\"id\":\"" + job->id + "\"}", "application/json");
        return;
    }
    Mm3RefinerPaths paths;
    const Mm3StyleFn style = mm3_refiner_style_override();
    if (!mm3_refiner_resolve(&paths, !style, &err)) {
        bad(501, err);
        return;
    }

    auto job = job_create();
    fprintf(stderr, "[MM3-Refine] job %s: %.1f s at %d Hz, steps=%d window=%g overlap=%g mix=%g, %zu window(s)\n", job->id.c_str(),
            (double) T / sr, sr, p.steps, p.window_seconds, p.overlap_seconds, (double) p.mix, plan.starts.size());
    work_push([job, paths, p, audio = std::move(audio), T, sr, style]() {
        auto finish = [&](int status, JobPhase phase, const std::string & why) {
            job->result_error = why;
            job_set_phase(*job, phase);
            job->status.store(status);
        };
        if (job->cancel.load()) {
            finish(3, JobPhase::CANCELLED, "");
            return;
        }
        Mm3RefinerHooks hooks;
        hooks.cancelled = [&]() { return job->cancel.load(); };
        hooks.progress  = [&](Mm3RefinerPhase ph, int step, int total) {
            switch (ph) {
                case Mm3RefinerPhase::LOAD:      job_set_phase(*job, JobPhase::LOADING_REFINER); break;
                case Mm3RefinerPhase::CONDITION: job_set_phase(*job, JobPhase::REFINE_CONDITION, step, total); break;
                case Mm3RefinerPhase::BRIDGE:    job_set_phase(*job, JobPhase::REFINE_BRIDGE, step, total); break;
                case Mm3RefinerPhase::DECODE:    job_set_phase(*job, JobPhase::VAE_DECODE); break;
                case Mm3RefinerPhase::OUTPUT:    job_set_phase(*job, JobPhase::ENCODING_OUTPUT); break;
            }
        };
        Mm3RefinerResult r;
        std::string      err;
        const auto       t0 = std::chrono::steady_clock::now();
        if (!mm3_refiner_run(g_store, paths, p, audio.data(), T, sr, style, hooks, &r, &err)) {
            if (err == "cancelled") {
                fprintf(stderr, "[MM3-Refine] job %s cancelled\n", job->id.c_str());
                finish(3, JobPhase::CANCELLED, "");
            } else {
                fprintf(stderr, "[MM3-Refine] job %s failed: %s\n", job->id.c_str(), err.c_str());
                finish(2, JobPhase::FAILED, err);
            }
            return;
        }
        job_set_phase(*job, JobPhase::ENCODING_OUTPUT);
        job->result_body = audio_encode_wav(r.audio.data(), (int) r.n, r.sample_rate, WAV_F32);
        job->result_mime = "audio/wav";
        fprintf(stderr, "[MM3-Refine] job %s done: %d window(s), %lld frames, %.0f ms, gain %.3f dB\n", job->id.c_str(), r.windows,
                (long long) r.frames, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), r.gain_db);
        job_set_phase(*job, JobPhase::DONE);
        job->status.store(1);
    });
    res.set_content("{\"id\":\"" + job->id + "\"}", "application/json");
}

static void mm3_refiner_register_routes(httplib::Server & svr, const char * models_dir) {
    g_mm3_refiner_dir = std::string(models_dir) + "/mm3";
    svr.Post("/mm3/refine", mm3_refiner_handle);
}
