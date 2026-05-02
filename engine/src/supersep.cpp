// supersep.cpp: ONNX Runtime-based stem separation pipeline.
// Implements the 4-stage SuperSep pipeline using ONNX models.
// Part of HOT-Step CPP. MIT license.

#include "supersep.h"
#include "supersep-stft.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef HOT_STEP_SUPERSEP

#include <onnxruntime_cxx_api.h>

// ── Internal types ──────────────────────────────────────────────────────

struct SuperSep {
    std::string   model_dir;
    int           device_id;
    Ort::Env      env;
    Ort::Session *s1_bs_roformer;      // Stage 1
    Ort::Session *s2_mel_band;         // Stage 2
    Ort::Session *s3_mdx23c;           // Stage 3
    Ort::Session *s4_htdemucs;         // Stage 4
    Ort::SessionOptions session_opts;

    SuperSep() : env(ORT_LOGGING_LEVEL_WARNING, "supersep"),
                 s1_bs_roformer(nullptr), s2_mel_band(nullptr),
                 s3_mdx23c(nullptr), s4_htdemucs(nullptr) {}
    ~SuperSep() {
        delete s1_bs_roformer;
        delete s2_mel_band;
        delete s3_mdx23c;
        delete s4_htdemucs;
    }
};

// ── Constants ───────────────────────────────────────────────────────────

static const int SUPERSEP_SR = 44100;
static const float SILENCE_THRESHOLD_DB = -60.0f;

// Stem definitions for each stage
struct StemDef {
    const char *key;
    const char *name;
    const char *category;
    int stage;
};

static const StemDef STAGE1_STEMS[] = {
    {"01_Bass",   "Bass",   "instruments", 1},
    {"02_Guitar", "Guitar", "instruments", 1},
    {"03_Piano",  "Piano",  "instruments", 1},
    {"04_Vocals", "Vocals", "vocals",      1},
    {"05_Drums",  "Drums",  "drums",       1},
    {"06_Other",  "Other",  "other",       1},
};
static const int N_STAGE1_STEMS = 6;

static const StemDef STAGE2_STEMS[] = {
    {"04_Lead_Vocals",    "Lead Vocals",    "vocals", 2},
    {"05_Backing_Vocals", "Backing Vocals", "vocals", 2},
};

static const StemDef STAGE3_STEMS[] = {
    {"06_Kick",  "Kick",   "drums", 3},
    {"07_Snare", "Snare",  "drums", 3},
    {"08_Toms",  "Toms",   "drums", 3},
    {"09_HiHat", "Hi-Hat", "drums", 3},
    {"10_Ride",  "Ride",   "drums", 3},
    {"11_Crash", "Crash",  "drums", 3},
};
static const int N_STAGE3_STEMS = 6;

static const StemDef STAGE4_STEMS[] = {
    {"12_Other_Vocal_Bleed", "Vocal Bleed",         "other",       4},
    {"13_Other_Guitar",      "Other Guitar",        "instruments", 4},
    {"14_Other_Piano_Keys",  "Other Piano/Keys",    "instruments", 4},
    {"15_Other_Bass",        "Other Bass",          "instruments", 4},
    {"16_Other_Percussion",  "Other Percussion",    "drums",       4},
    {"17_Residual",          "Residual (Synths/FX)","other",       4},
};
static const int N_STAGE4_STEMS = 6;

// ── Helpers ─────────────────────────────────────────────────────────────

static bool is_silent(const float *audio, int n_frames, int n_ch) {
    float peak = 0.0f;
    int total = n_frames * n_ch;
    for (int i = 0; i < total; i++) {
        float a = fabsf(audio[i]);
        if (a > peak) peak = a;
    }
    if (peak < 1e-10f) return true;
    float db = 20.0f * log10f(peak);
    return db < SILENCE_THRESHOLD_DB;
}

static Ort::Session * load_onnx_model(SuperSep *ctx, const char *filename) {
    std::string path = ctx->model_dir + "/" + filename;
    fprintf(stderr, "[SuperSep] Loading ONNX model: %s\n", path.c_str());

#ifdef _WIN32
    // Convert to wide string for Windows
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    return new Ort::Session(ctx->env, wpath.data(), ctx->session_opts);
#else
    return new Ort::Session(ctx->env, path.c_str(), ctx->session_opts);
#endif
}

// Run ONNX inference on a spectrogram-based model (stages 1-3).
// Input: complex spectrogram [1, n_ch, n_freqs, n_time, 2]
// Output: masks [1, n_stems, n_freqs, n_time] or similar
static std::vector<float> run_spec_model(
    Ort::Session *session,
    const ComplexSpec &spec,
    std::vector<int64_t> &out_shape
) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Prepare input tensor: [1, channels, freqs, time, 2]
    std::vector<int64_t> input_shape = {1, spec.n_channels, spec.n_freqs, spec.n_frames, 2};
    size_t input_size = 1;
    for (auto d : input_shape) input_size *= (size_t)d;

    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(spec.data), input_size, input_shape.data(), input_shape.size());

    // Get input/output names
    auto in_name = session->GetInputNameAllocated(0, alloc);
    auto out_name = session->GetOutputNameAllocated(0, alloc);
    const char *in_names[] = { in_name.get() };
    const char *out_names[] = { out_name.get() };

    // Run inference
    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    // Extract output
    auto &out_tensor = outputs[0];
    auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
    out_shape = type_info.GetShape();
    size_t out_size = type_info.GetElementCount();
    const float *out_data = out_tensor.GetTensorData<float>();

    return std::vector<float>(out_data, out_data + out_size);
}

// Run ONNX inference on a waveform-based model (stage 4: HTDemucs).
// Input: waveform [1, 2, n_samples]
// Output: sources [1, n_sources, 2, n_samples]
static std::vector<float> run_wave_model(
    Ort::Session *session,
    const float *interleaved,
    int n_frames,
    std::vector<int64_t> &out_shape
) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Deinterleave: [1, 2, n_samples]
    std::vector<float> input_data(2 * n_frames);
    for (int i = 0; i < n_frames; i++) {
        input_data[i]            = interleaved[i * 2 + 0]; // L
        input_data[n_frames + i] = interleaved[i * 2 + 1]; // R
    }

    std::vector<int64_t> input_shape = {1, 2, (int64_t)n_frames};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    auto in_name = session->GetInputNameAllocated(0, alloc);
    auto out_name = session->GetOutputNameAllocated(0, alloc);
    const char *in_names[] = { in_name.get() };
    const char *out_names[] = { out_name.get() };

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    auto &out_tensor = outputs[0];
    auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
    out_shape = type_info.GetShape();
    size_t out_size = type_info.GetElementCount();
    const float *out_data = out_tensor.GetTensorData<float>();

    return std::vector<float>(out_data, out_data + out_size);
}

// Extract a single stem from model output (interleaved stereo).
// Returns malloc'd buffer. Caller must free.
static float * extract_stem_interleaved(
    const std::vector<float> &model_output,
    const std::vector<int64_t> &shape,
    int stem_idx,
    int n_frames
) {
    // Assuming shape is [1, n_sources, 2, n_samples]
    int n_sources = (int)shape[1];
    int n_ch = (int)shape[2];
    int n_samp = (int)shape[3];
    if (stem_idx >= n_sources) return nullptr;

    int actual_frames = std::min(n_samp, n_frames);
    float *out = (float *)malloc((size_t)actual_frames * 2 * sizeof(float));
    if (!out) return nullptr;

    size_t base = (size_t)stem_idx * n_ch * n_samp;
    for (int i = 0; i < actual_frames; i++) {
        out[i * 2 + 0] = model_output[base + i];               // L
        out[i * 2 + 1] = model_output[base + (size_t)n_samp + i]; // R
    }
    return out;
}

// Apply spectrogram mask and iSTFT to extract a stem.
static float * extract_stem_from_mask(
    const ComplexSpec &input_spec,
    const float *mask,
    int mask_offset,   // offset into mask buffer for this stem
    int n_frames_audio,
    int *out_frames
) {
    // Clone the input spectrogram
    size_t spec_size = (size_t)input_spec.n_channels * input_spec.n_freqs * input_spec.n_frames * 2;
    ComplexSpec masked;
    masked.data = (float *)malloc(spec_size * sizeof(float));
    masked.n_channels = input_spec.n_channels;
    masked.n_freqs = input_spec.n_freqs;
    masked.n_frames = input_spec.n_frames;
    masked.n_fft = input_spec.n_fft;
    masked.hop_length = input_spec.hop_length;
    memcpy(masked.data, input_spec.data, spec_size * sizeof(float));

    // Apply mask
    int mask_stride = input_spec.n_channels * input_spec.n_freqs * input_spec.n_frames;
    stft_apply_mask(&masked, mask + mask_offset);

    // Inverse STFT
    float *audio = stft_inverse(masked, n_frames_audio, out_frames);
    stft_free(&masked);
    return audio;
}

static void add_stem(std::vector<SuperSepStem> &stems, const StemDef &def,
                     float *samples, int n_frames) {
    if (!samples) return;
    if (is_silent(samples, n_frames, 2)) {
        fprintf(stderr, "[SuperSep] Skipping silent stem: %s\n", def.name);
        free(samples);
        return;
    }
    SuperSepStem s;
    snprintf(s.name, sizeof(s.name), "%s", def.name);
    snprintf(s.category, sizeof(s.category), "%s", def.category);
    snprintf(s.stem_type, sizeof(s.stem_type), "%s", def.key);
    s.samples = samples;
    s.n_samples = n_frames * 2;
    s.n_frames = n_frames;
    s.stage = def.stage;
    stems.push_back(s);
}

// ── Public API ──────────────────────────────────────────────────────────

SuperSep * supersep_init(const char * model_dir, int device_id) {
    auto *ctx = new SuperSep();
    ctx->model_dir = model_dir;
    ctx->device_id = device_id;

    // Configure session options
    ctx->session_opts.SetIntraOpNumThreads(4);
    ctx->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef GGML_USE_CUDA
    if (device_id >= 0) {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id = device_id;
        ctx->session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        fprintf(stderr, "[SuperSep] CUDA EP enabled (device %d)\n", device_id);
    }
#else
    (void)device_id;
    fprintf(stderr, "[SuperSep] CPU mode (no CUDA)\n");
#endif

    fprintf(stderr, "[SuperSep] Initialized (models: %s)\n", model_dir);
    return ctx;
}

SuperSepResult * supersep_run(
    SuperSep *          ctx,
    const float *       audio,
    int                 n_frames,
    SuperSepLevel       level,
    supersep_progress_fn progress,
    supersep_cancel_fn   cancel,
    void *              user_data
) {
    if (!ctx || !audio || n_frames <= 0) return nullptr;

    auto cb = [&](int stage, const char *msg, float pct) {
        if (progress) progress(stage, msg, pct, user_data);
    };
    auto cancelled = [&]() -> bool {
        return cancel && cancel(user_data);
    };

    std::vector<SuperSepStem> stems;
    int stages[] = {1, 0, 0, 0};
    if (level >= SUPERSEP_VOCAL_SPLIT) stages[1] = 1;
    if (level >= SUPERSEP_FULL)        stages[2] = 1;
    if (level >= SUPERSEP_MAXIMUM)     stages[3] = 1;

    // Stems from stage 1 that feed into later stages
    float *s1_vocals = nullptr, *s1_drums = nullptr, *s1_other = nullptr;
    int s1_vocal_frames = 0, s1_drum_frames = 0, s1_other_frames = 0;

    // ── STAGE 1: Primary 6-stem split ────────────────────────────────
    cb(1, "Loading BS-RoFormer model...", 0.05f);
    if (cancelled()) return nullptr;

    try {
        if (!ctx->s1_bs_roformer) {
            ctx->s1_bs_roformer = load_onnx_model(ctx, "bs_roformer_sw.onnx");
        }

        cb(1, "Computing STFT...", 0.10f);
        StftParams sp = stft_default_params();
        ComplexSpec spec = stft_forward(audio, n_frames, sp);

        cb(1, "Running BS-RoFormer inference...", 0.15f);
        std::vector<int64_t> out_shape;
        auto masks = run_spec_model(ctx->s1_bs_roformer, spec, out_shape);

        cb(1, "Extracting stems...", 0.20f);
        int mask_stride = spec.n_channels * spec.n_freqs * spec.n_frames;

        // Extract each stem via mask + iSTFT
        for (int i = 0; i < N_STAGE1_STEMS && i < (int)out_shape[1]; i++) {
            int stem_frames = 0;
            float *stem_audio = extract_stem_from_mask(spec, masks.data(),
                i * mask_stride, n_frames, &stem_frames);

            // Keep vocals/drums/other for later stages
            if (i == 3 && stages[1]) { // Vocals → Stage 2
                s1_vocals = stem_audio;
                s1_vocal_frames = stem_frames;
            } else if (i == 4 && stages[2]) { // Drums → Stage 3
                s1_drums = stem_audio;
                s1_drum_frames = stem_frames;
            } else if (i == 5 && stages[3]) { // Other → Stage 4
                s1_other = stem_audio;
                s1_other_frames = stem_frames;
            } else {
                add_stem(stems, STAGE1_STEMS[i], stem_audio, stem_frames);
            }
        }
        stft_free(&spec);

        // If not going deeper, add the kept stems
        if (!stages[1] && s1_vocals) {
            add_stem(stems, STAGE1_STEMS[3], s1_vocals, s1_vocal_frames);
            s1_vocals = nullptr;
        }
        if (!stages[2] && s1_drums) {
            add_stem(stems, STAGE1_STEMS[4], s1_drums, s1_drum_frames);
            s1_drums = nullptr;
        }
        if (!stages[3] && s1_other) {
            add_stem(stems, STAGE1_STEMS[5], s1_other, s1_other_frames);
            s1_other = nullptr;
        }

        cb(1, "Stage 1 complete", 0.25f);
    } catch (const std::exception &e) {
        fprintf(stderr, "[SuperSep] Stage 1 failed: %s\n", e.what());
        cb(1, "Stage 1 failed", 0.25f);
        // Can't continue without stage 1
        return nullptr;
    }

    // ── STAGE 2: Vocal sub-separation ────────────────────────────────
    if (stages[1] && s1_vocals) {
        cb(2, "Loading Mel-Band RoFormer model...", 0.30f);
        if (cancelled()) { free(s1_vocals); free(s1_drums); free(s1_other); return nullptr; }

        try {
            if (!ctx->s2_mel_band) {
                ctx->s2_mel_band = load_onnx_model(ctx, "mel_band_roformer_karaoke.onnx");
            }

            cb(2, "Splitting lead/backing vocals...", 0.35f);
            StftParams sp = stft_default_params();
            ComplexSpec spec = stft_forward(s1_vocals, s1_vocal_frames, sp);

            std::vector<int64_t> out_shape;
            auto masks = run_spec_model(ctx->s2_mel_band, spec, out_shape);

            int mask_stride = spec.n_channels * spec.n_freqs * spec.n_frames;
            int n_out = std::min((int)out_shape[1], 2);

            for (int i = 0; i < n_out; i++) {
                int sf = 0;
                float *sa = extract_stem_from_mask(spec, masks.data(),
                    i * mask_stride, s1_vocal_frames, &sf);
                add_stem(stems, STAGE2_STEMS[i], sa, sf);
            }
            stft_free(&spec);
            cb(2, "Vocal split complete", 0.45f);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] Stage 2 failed: %s\n", e.what());
            // Fallback: keep original vocals
            StemDef fallback = {"04_Vocals", "Vocals", "vocals", 1};
            add_stem(stems, fallback, s1_vocals, s1_vocal_frames);
            s1_vocals = nullptr;
            cb(2, "Vocal split failed, keeping original", 0.45f);
        }
        free(s1_vocals);
        s1_vocals = nullptr;
    }

    // ── STAGE 3: Drum sub-separation ─────────────────────────────────
    if (stages[2] && s1_drums) {
        cb(3, "Loading MDX23C DrumSep model...", 0.50f);
        if (cancelled()) { free(s1_drums); free(s1_other); return nullptr; }

        try {
            if (!ctx->s3_mdx23c) {
                ctx->s3_mdx23c = load_onnx_model(ctx, "mdx23c_drumsep.onnx");
            }

            cb(3, "Splitting drums...", 0.55f);
            StftParams sp = stft_default_params();
            ComplexSpec spec = stft_forward(s1_drums, s1_drum_frames, sp);

            std::vector<int64_t> out_shape;
            auto masks = run_spec_model(ctx->s3_mdx23c, spec, out_shape);

            int mask_stride = spec.n_channels * spec.n_freqs * spec.n_frames;
            int n_out = std::min((int)out_shape[1], N_STAGE3_STEMS);

            for (int i = 0; i < n_out; i++) {
                int sf = 0;
                float *sa = extract_stem_from_mask(spec, masks.data(),
                    i * mask_stride, s1_drum_frames, &sf);
                add_stem(stems, STAGE3_STEMS[i], sa, sf);
            }
            stft_free(&spec);
            cb(3, "Drum split complete", 0.70f);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] Stage 3 failed: %s\n", e.what());
            StemDef fallback = {"05_Drums", "Drums", "drums", 1};
            add_stem(stems, fallback, s1_drums, s1_drum_frames);
            s1_drums = nullptr;
            cb(3, "Drum split failed, keeping original", 0.70f);
        }
        free(s1_drums);
        s1_drums = nullptr;
    }

    // ── STAGE 4: "Other" refinement (HTDemucs waveform model) ────────
    if (stages[3] && s1_other) {
        cb(4, "Loading HTDemucs model...", 0.75f);
        if (cancelled()) { free(s1_other); return nullptr; }

        try {
            if (!ctx->s4_htdemucs) {
                ctx->s4_htdemucs = load_onnx_model(ctx, "htdemucs_6s.onnx");
            }

            cb(4, "Refining 'other' stem...", 0.80f);
            std::vector<int64_t> out_shape;
            auto sources = run_wave_model(ctx->s4_htdemucs, s1_other,
                                          s1_other_frames, out_shape);

            int n_out = std::min((int)out_shape[1], N_STAGE4_STEMS);
            for (int i = 0; i < n_out; i++) {
                float *sa = extract_stem_interleaved(sources, out_shape, i, s1_other_frames);
                add_stem(stems, STAGE4_STEMS[i], sa, s1_other_frames);
            }
            cb(4, "Other refinement complete", 0.90f);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] Stage 4 failed: %s\n", e.what());
            StemDef fallback = {"06_Other", "Other", "other", 1};
            add_stem(stems, fallback, s1_other, s1_other_frames);
            s1_other = nullptr;
            cb(4, "Other refinement failed, keeping original", 0.90f);
        }
        free(s1_other);
        s1_other = nullptr;
    }

    // ── Collect results ──────────────────────────────────────────────
    cb(0, "Finalizing...", 0.95f);

    auto *result = (SuperSepResult *)malloc(sizeof(SuperSepResult));
    result->n_stems = (int)stems.size();
    result->stems = (SuperSepStem *)malloc(sizeof(SuperSepStem) * stems.size());
    memcpy(result->stems, stems.data(), sizeof(SuperSepStem) * stems.size());

    fprintf(stderr, "[SuperSep] Done — %d stems extracted\n", result->n_stems);
    cb(0, "Complete", 1.0f);
    return result;
}

void supersep_result_free(SuperSepResult * result) {
    if (!result) return;
    for (int i = 0; i < result->n_stems; i++) {
        free(result->stems[i].samples);
    }
    free(result->stems);
    free(result);
}

void supersep_free(SuperSep * ctx) {
    delete ctx;
}

float * supersep_recombine(
    const SuperSepStem * stems,
    const float *        volumes,
    const bool *         muted,
    int                  n_stems,
    int *                out_frames
) {
    // Find max length
    int max_frames = 0;
    for (int i = 0; i < n_stems; i++) {
        if (!muted[i] && volumes[i] > 0.0f && stems[i].n_frames > max_frames) {
            max_frames = stems[i].n_frames;
        }
    }
    if (max_frames <= 0) { *out_frames = 0; return nullptr; }

    // Mix
    std::vector<double> mixed(max_frames * 2, 0.0);
    for (int i = 0; i < n_stems; i++) {
        if (muted[i] || volumes[i] <= 0.0f) continue;
        float vol = volumes[i];
        int nf = stems[i].n_frames;
        const float *s = stems[i].samples;
        for (int f = 0; f < nf && f < max_frames; f++) {
            mixed[f * 2 + 0] += (double)(s[f * 2 + 0] * vol);
            mixed[f * 2 + 1] += (double)(s[f * 2 + 1] * vol);
        }
    }

    // Normalize (-1dB headroom)
    double peak = 0.0;
    for (auto v : mixed) { double a = fabs(v); if (a > peak) peak = a; }
    if (peak > 0.0) {
        double target = pow(10.0, -1.0 / 20.0);
        double gain = target / peak;
        for (auto &v : mixed) v *= gain;
    }

    // Convert to float
    float *out = (float *)malloc(sizeof(float) * max_frames * 2);
    for (int i = 0; i < max_frames * 2; i++) {
        out[i] = (float)mixed[i];
    }
    *out_frames = max_frames;
    return out;
}

#else // !HOT_STEP_SUPERSEP — stub implementations

SuperSep * supersep_init(const char *, int) {
    fprintf(stderr, "[SuperSep] Not compiled (HOT_STEP_SUPERSEP not defined)\n");
    return nullptr;
}

SuperSepResult * supersep_run(SuperSep *, const float *, int, SuperSepLevel,
                              supersep_progress_fn, supersep_cancel_fn, void *) {
    return nullptr;
}

void supersep_result_free(SuperSepResult *) {}
void supersep_free(SuperSep *) {}

float * supersep_recombine(const SuperSepStem *, const float *, const bool *,
                           int, int *) {
    return nullptr;
}

#endif // HOT_STEP_SUPERSEP
