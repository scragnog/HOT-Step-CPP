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

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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

// BS-RoFormer STFT params (from BS-Roformer-SW.yaml)
static const int BS_N_FFT       = 2048;
static const int BS_HOP_LENGTH  = 512;
static const int BS_WIN_LENGTH  = 2048;
static const int BS_CHUNK_SIZE  = 588800;  // ~13.4s at 44100Hz
static const int BS_N_FREQS     = BS_N_FFT / 2 + 1;  // 1025
static const int BS_NUM_STEMS   = 6;

// Mel-Band RoFormer lookup tables (auto-generated from mel filterbank)
#include "mel_band_tables.inc"

// Stem definitions for each stage
struct StemDef {
    const char *key;
    const char *name;
    const char *category;
    int stage;
};

static const StemDef STAGE1_STEMS[] = {
    {"01_Bass",   "Bass",   "instruments", 1},
    {"02_Drums",  "Drums",  "drums",       1},
    {"03_Other",  "Other",  "other",       1},
    {"04_Vocals", "Vocals", "vocals",      1},
    {"05_Guitar", "Guitar", "instruments", 1},
    {"06_Piano",  "Piano",  "instruments", 1},
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

// ── BS-RoFormer STFT-based processing ───────────────────────────────────
//
// The ONNX model expects post-STFT input and produces mask output.
// Preprocessing:  waveform → STFT → rearrange → model input [1, T, F*C*2]
// Postprocessing: model output (mask) [1, stems, F*C, T, 2] → apply mask → iSTFT
//
// Reference: ZFTurbo/MSS_ONNX_TensorRT/models/preprocess.py

// Process one chunk through BS-RoFormer.
// chunk: interleaved stereo [L0,R0,L1,R1,...], chunk_frames per channel.
// Returns n_stems interleaved stereo buffers (malloc'd). Caller frees each.
static bool bs_roformer_process_chunk(
    Ort::Session *session,
    const float *chunk,          // interleaved stereo
    int chunk_frames,            // per-channel frame count
    int n_stems,
    std::vector<float *> &stem_outputs,  // [n_stems] malloc'd interleaved stereo
    std::vector<int> &stem_frame_counts
) {
    const int n_fft     = BS_N_FFT;
    const int hop       = BS_HOP_LENGTH;
    const int n_freqs   = BS_N_FREQS;  // 1025
    const int n_ch      = 2;  // stereo

    // ── STFT ─────────────────────────────────────────────────────────
    StftParams sp;
    sp.n_fft = n_fft;
    sp.hop_length = hop;
    sp.n_channels = n_ch;
    ComplexSpec spec = stft_forward(chunk, chunk_frames, sp);
    const int T = spec.n_frames;

    fprintf(stderr, "[SuperSep] BS-RoFormer STFT: %d freqs x %d time frames\n", n_freqs, T);

    // ── Rearrange STFT output to model input format ──────────────────
    // Python: stft_repr has shape [batch, stereo, freq, time, 2] (real/imag)
    //   rearrange 'b s f t c -> b (f s) t c' → [1, freq*stereo, time, 2]
    //   rearrange 'b f t c -> b t (f c)'     → [1, time, freq*stereo*2]
    //
    // Our ComplexSpec layout: [ch][freq][time][2]
    // Target: [1, T, n_freqs*n_ch*2]  where dimension ordering is
    //   for each time step: [f0_ch0_re, f0_ch0_im, f0_ch1_re, f0_ch1_im, f1_ch0_re, ...]
    //
    // Actually from the Python code:
    //   'b s f t c -> b (f s) t c' merges freq and stereo with freq leading
    //   Then 'b f t c -> b t (f c)' flattens the (f_s) and c dimensions
    //   So the order is: [f0_s0_re, f0_s0_im, f0_s1_re, f0_s1_im, f1_s0_re, ...]

    int input_dim = n_freqs * n_ch * 2;  // 1025 * 2 * 2 = 4100
    std::vector<float> model_input((size_t)T * input_dim);

    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            for (int ch = 0; ch < n_ch; ch++) {
                const float *c = spec.at(ch, f, t);
                // Index into flattened: (f * n_ch + ch) * 2 + {0,1}
                int base = (f * n_ch + ch) * 2;
                model_input[(size_t)t * input_dim + base + 0] = c[0]; // real
                model_input[(size_t)t * input_dim + base + 1] = c[1]; // imag
            }
        }
    }

    // Also save the stft_repr in the rearranged format [n_freqs*n_ch, T, 2]
    // for mask application later
    int fs = n_freqs * n_ch;  // 2050
    std::vector<float> stft_repr((size_t)fs * T * 2);
    for (int f = 0; f < n_freqs; f++) {
        for (int ch = 0; ch < n_ch; ch++) {
            int fs_idx = f * n_ch + ch;
            for (int t = 0; t < T; t++) {
                const float *c = spec.at(ch, f, t);
                stft_repr[((size_t)fs_idx * T + t) * 2 + 0] = c[0];
                stft_repr[((size_t)fs_idx * T + t) * 2 + 1] = c[1];
            }
        }
    }

    // ── Run ONNX inference ───────────────────────────────────────────
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, (int64_t)T, (int64_t)input_dim};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, model_input.data(), model_input.size(), input_shape.data(), input_shape.size());

    auto in_name = session->GetInputNameAllocated(0, alloc);
    auto out_name = session->GetOutputNameAllocated(0, alloc);
    const char *in_names[] = { in_name.get() };
    const char *out_names[] = { out_name.get() };

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    auto &out_tensor = outputs[0];
    auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
    auto out_shape = type_info.GetShape();
    const float *mask_data = out_tensor.GetTensorData<float>();

    fprintf(stderr, "[SuperSep] BS-RoFormer output shape: [");
    for (size_t i = 0; i < out_shape.size(); i++)
        fprintf(stderr, "%s%lld", i ? "," : "", (long long)out_shape[i]);
    fprintf(stderr, "]\n");

    // Expected output: [1, n_stems, fs(=2050), T, 2]
    int out_stems = (out_shape.size() >= 2) ? (int)out_shape[1] : 0;
    int actual_stems = std::min(out_stems, n_stems);

    // ── Apply mask and iSTFT for each stem ───────────────────────────
    // mask shape: [1, stems, fs, T, 2] where fs = n_freqs * n_ch = 2050
    // stft_repr: [fs, T, 2]
    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // Then iSTFT each stem

    for (int s = 0; s < actual_stems; s++) {
        // Build per-stem complex spectrogram: stft_repr * mask[s]
        ComplexSpec stem_spec;
        stem_spec.n_channels = n_ch;
        stem_spec.n_freqs = n_freqs;
        stem_spec.n_frames = T;
        stem_spec.n_fft = n_fft;
        stem_spec.hop_length = hop;
        stem_spec.data = (float *)calloc((size_t)n_ch * n_freqs * T * 2, sizeof(float));

        for (int f = 0; f < n_freqs; f++) {
            for (int ch = 0; ch < n_ch; ch++) {
                int fs_idx = f * n_ch + ch;
                for (int t = 0; t < T; t++) {
                    // stft_repr[fs_idx, t] complex
                    float sr = stft_repr[((size_t)fs_idx * T + t) * 2 + 0];
                    float si = stft_repr[((size_t)fs_idx * T + t) * 2 + 1];
                    // mask[1, s, fs_idx, t, 2]
                    size_t mask_off = ((size_t)s * fs * T + (size_t)fs_idx * T + t) * 2;
                    float mr = mask_data[mask_off + 0];
                    float mi = mask_data[mask_off + 1];
                    // Complex multiply
                    float *dst = stem_spec.at(ch, f, t);
                    dst[0] = sr * mr - si * mi;
                    dst[1] = sr * mi + si * mr;
                }
            }
        }

        int out_len = 0;
        float *stem_audio = stft_inverse(stem_spec, chunk_frames, &out_len);
        stft_free(&stem_spec);

        stem_outputs.push_back(stem_audio);
        stem_frame_counts.push_back(out_len);
    }

    stft_free(&spec);
    return true;
}

// Full BS-RoFormer separation with chunking + overlap-add.
// audio: interleaved stereo, n_frames per channel.
// Returns per-stem interleaved stereo buffers.
static bool bs_roformer_separate(
    Ort::Session *session,
    const float *audio,
    int n_frames,
    int n_stems,
    std::vector<float *> &stem_outputs,   // [n_stems], each is interleaved stereo
    std::vector<int> &stem_frame_counts,
    std::function<void(int, const char *, float)> cb,
    std::function<bool()> cancelled
) {
    const int chunk_size = BS_CHUNK_SIZE;

    if (n_frames <= chunk_size) {
        // Single chunk — pad to full chunk_size if needed
        cb(1, "Running BS-RoFormer inference...", 0.10f);

        if (n_frames < chunk_size) {
            // Pad with zeros to full chunk_size
            std::vector<float> padded(chunk_size * 2, 0.0f);
            memcpy(padded.data(), audio, (size_t)n_frames * 2 * sizeof(float));
            bool ok = bs_roformer_process_chunk(session, padded.data(), chunk_size, n_stems,
                                                  stem_outputs, stem_frame_counts);
            // Trim outputs back to original length
            if (ok) {
                for (int s = 0; s < (int)stem_outputs.size(); s++) {
                    stem_frame_counts[s] = n_frames;
                }
            }
            return ok;
        }
        return bs_roformer_process_chunk(session, audio, n_frames, n_stems,
                                          stem_outputs, stem_frame_counts);
    }

    // Multiple chunks with overlap-add.
    // Use a small crossfade overlap (~1 second) for seamless joins.
    // The model's chunk_size is ~13.4s, so overlap of ~1s is ~7%.
    const int crossfade = 44100;  // 1 second crossfade region
    int step = chunk_size - crossfade;
    int n_chunks = (n_frames - crossfade + step - 1) / step;
    if (n_chunks < 1) n_chunks = 1;

    fprintf(stderr, "[SuperSep] Chunking: %d frames into %d chunks "
            "(chunk=%d, crossfade=%d, step=%d)\n",
            n_frames, n_chunks, chunk_size, crossfade, step);

    // Allocate output accumulators (per stem)
    std::vector<std::vector<float>> accum(n_stems, std::vector<float>(n_frames * 2, 0.0f));
    std::vector<std::vector<float>> weight(n_stems, std::vector<float>(n_frames * 2, 0.0f));

    // Crossfade window: ramp up over first crossfade/2, constant 1, ramp down over last crossfade/2
    std::vector<float> fade_window(chunk_size, 1.0f);
    int half_fade = crossfade / 2;
    for (int i = 0; i < half_fade; i++) {
        float t = (float)i / (float)half_fade;
        fade_window[i] = t;  // fade in
        fade_window[chunk_size - 1 - i] = t;  // fade out
    }

    for (int c = 0; c < n_chunks; c++) {
        if (cancelled()) return false;

        int start = c * step;
        int end = std::min(start + chunk_size, n_frames);
        int this_chunk = end - start;

        float pct = 0.10f + 0.15f * (float)c / (float)n_chunks;
        char msg[64];
        snprintf(msg, sizeof(msg), "Processing chunk %d/%d...", c + 1, n_chunks);
        cb(1, msg, pct);

        // Extract chunk — ALWAYS pad to full chunk_size for fixed ONNX input shape
        std::vector<float> chunk_buf(chunk_size * 2, 0.0f);
        memcpy(chunk_buf.data(), audio + start * 2, (size_t)this_chunk * 2 * sizeof(float));

        std::vector<float *> chunk_stems;
        std::vector<int> chunk_counts;

        if (!bs_roformer_process_chunk(session, chunk_buf.data(), chunk_size,
                                        n_stems, chunk_stems, chunk_counts)) {
            for (auto p : chunk_stems) free(p);
            return false;
        }

        // Overlap-add with crossfade window
        // For first chunk: no fade-in. For last chunk: no fade-out.
        for (int s = 0; s < (int)chunk_stems.size() && s < n_stems; s++) {
            for (int i = 0; i < this_chunk; i++) {
                float w = fade_window[i];
                // First chunk: don't fade in
                if (c == 0 && i < half_fade) w = 1.0f;
                // Last chunk: don't fade out
                if (c == n_chunks - 1 && i >= this_chunk - half_fade) w = 1.0f;

                int dst = (start + i) * 2;
                if (dst + 1 < n_frames * 2) {
                    accum[s][dst + 0] += chunk_stems[s][i * 2 + 0] * w;
                    accum[s][dst + 1] += chunk_stems[s][i * 2 + 1] * w;
                    weight[s][dst + 0] += w;
                    weight[s][dst + 1] += w;
                }
            }
            free(chunk_stems[s]);
        }
    }

    // Normalize and output
    for (int s = 0; s < n_stems; s++) {
        float *out = (float *)malloc((size_t)n_frames * 2 * sizeof(float));
        for (int i = 0; i < n_frames * 2; i++) {
            out[i] = (weight[s][i] > 1e-8f) ? accum[s][i] / weight[s][i] : 0.0f;
        }
        stem_outputs.push_back(out);
        stem_frame_counts.push_back(n_frames);
    }

    return true;
}

// ── Mel-Band RoFormer STFT-based processing ─────────────────────────────
//
// Similar to BS-RoFormer but with mel-band frequency gathering.
// Input:  waveform → STFT → rearrange → gather mel indices → [1, 3958, T, 2]
// Output: mask [1, 1, 3958, T, 2] → scatter-add → average → complex multiply → iSTFT
//
// The model outputs 1 source; the complement is computed by subtraction.
// Reference: ZFTurbo/MSS_ONNX_TensorRT/models/preprocess.py Mel_band_roformer_processor

static bool mel_band_process_chunk(
    Ort::Session *session,
    const float *chunk,           // interleaved stereo vocals
    int chunk_frames,             // per-channel frame count (must be MB_CHUNK_SAMPLES)
    float *&out_lead,             // output: lead vocals (malloc'd interleaved stereo)
    float *&out_backing,          // output: backing vocals (malloc'd interleaved stereo)
    int &out_frames
) {
    const int n_fft   = MB_STFT_N_FFT;
    const int hop     = MB_STFT_HOP;
    const int n_freqs = MB_N_FREQS;  // 1025
    const int n_ch    = 2;
    const int fs      = n_freqs * n_ch;  // 2050 (stereo freq dimension)

    // ── STFT ─────────────────────────────────────────────────────────
    StftParams sp;
    sp.n_fft = n_fft;
    sp.hop_length = hop;
    sp.n_channels = n_ch;
    ComplexSpec spec = stft_forward(chunk, chunk_frames, sp);
    const int T = spec.n_frames;

    fprintf(stderr, "[SuperSep] Mel-Band STFT: %d freqs x %d time frames\n", n_freqs, T);

    // DEBUG: Check input signal level
    { float pk = 0; for (int i = 0; i < chunk_frames*2; i++) { float a = fabsf(chunk[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG input chunk peak: %.6f\n", pk); }

    // ── Build stft_repr [fs, T, 2] ──────────────────────────────────
    // Rearrange: 'b s f t c -> b (f s) t c' with freq leading
    std::vector<float> stft_repr((size_t)fs * T * 2);
    for (int f = 0; f < n_freqs; f++) {
        for (int ch = 0; ch < n_ch; ch++) {
            int fs_idx = f * n_ch + ch;
            for (int t = 0; t < T; t++) {
                const float *c = spec.at(ch, f, t);
                stft_repr[((size_t)fs_idx * T + t) * 2 + 0] = c[0];
                stft_repr[((size_t)fs_idx * T + t) * 2 + 1] = c[1];
            }
        }
    }

    // ── Gather mel-band frequency indices → model input [3958, T, 2] ─
    const int n_gathered = MB_N_FREQ_INDICES_STEREO;
    std::vector<float> model_input((size_t)n_gathered * T * 2);

    for (int i = 0; i < MB_N_FREQ_INDICES_MONO; i++) {
        int mono_f = MB_FREQ_INDICES_MONO[i];
        for (int ch = 0; ch < n_ch; ch++) {
            int stereo_gather_idx = i * n_ch + ch;  // output index
            int stereo_src_idx = mono_f * n_ch + ch; // source in stft_repr
            for (int t = 0; t < T; t++) {
                model_input[((size_t)stereo_gather_idx * T + t) * 2 + 0] =
                    stft_repr[((size_t)stereo_src_idx * T + t) * 2 + 0];
                model_input[((size_t)stereo_gather_idx * T + t) * 2 + 1] =
                    stft_repr[((size_t)stereo_src_idx * T + t) * 2 + 1];
            }
        }
    }

    // ── Run ONNX inference ───────────────────────────────────────────
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, (int64_t)n_gathered, (int64_t)T, 2};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem, model_input.data(), model_input.size(), input_shape.data(), input_shape.size());

    auto in_name = session->GetInputNameAllocated(0, alloc);
    auto out_name = session->GetOutputNameAllocated(0, alloc);
    const char *in_names[] = { in_name.get() };
    const char *out_names[] = { out_name.get() };

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input_tensor, 1, out_names, 1);

    auto &out_tensor = outputs[0];
    auto type_info = out_tensor.GetTensorTypeAndShapeInfo();
    auto out_shape = type_info.GetShape();
    const float *mask_data = out_tensor.GetTensorData<float>();

    fprintf(stderr, "[SuperSep] Mel-Band output shape: [");
    for (size_t i = 0; i < out_shape.size(); i++)
        fprintf(stderr, "%s%lld", i ? "," : "", (long long)out_shape[i]);
    fprintf(stderr, "]\n");

    // DEBUG: Check model input and output levels
    { float pk = 0; for (size_t i = 0; i < model_input.size(); i++) { float a = fabsf(model_input[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG model_input peak: %.6f\n", pk); }
    { float pk = 0; size_t n = (size_t)n_gathered * T * 2; for (size_t i = 0; i < n; i++) { float a = fabsf(mask_data[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG mask_data peak: %.6f\n", pk); }

    // ── Scatter-add mask back to full spectrogram space ──────────────
    // mask_data: [1, 1, n_gathered, T, 2]
    // Scatter into masks_summed: [fs, T, 2]
    std::vector<float> masks_summed((size_t)fs * T * 2, 0.0f);

    for (int i = 0; i < MB_N_FREQ_INDICES_MONO; i++) {
        int mono_f = MB_FREQ_INDICES_MONO[i];
        for (int ch = 0; ch < n_ch; ch++) {
            int stereo_gather_idx = i * n_ch + ch;
            int stereo_dst_idx = mono_f * n_ch + ch;
            for (int t = 0; t < T; t++) {
                masks_summed[((size_t)stereo_dst_idx * T + t) * 2 + 0] +=
                    mask_data[((size_t)stereo_gather_idx * T + t) * 2 + 0];
                masks_summed[((size_t)stereo_dst_idx * T + t) * 2 + 1] +=
                    mask_data[((size_t)stereo_gather_idx * T + t) * 2 + 1];
            }
        }
    }

    // ── Average by band overlap count ────────────────────────────────
    for (int f = 0; f < n_freqs; f++) {
        float denom = (float)MB_NUM_BANDS_PER_FREQ[f];
        if (denom < 1e-8f) denom = 1.0f;
        for (int ch = 0; ch < n_ch; ch++) {
            int fs_idx = f * n_ch + ch;
            for (int t = 0; t < T; t++) {
                masks_summed[((size_t)fs_idx * T + t) * 2 + 0] /= denom;
                masks_summed[((size_t)fs_idx * T + t) * 2 + 1] /= denom;
            }
        }
    }

    // ── Complex multiply: result = stft_repr * averaged_mask ─────────
    // Then un-rearrange back to ComplexSpec for iSTFT
    ComplexSpec result_spec;
    result_spec.n_channels = n_ch;
    result_spec.n_freqs = n_freqs;
    result_spec.n_frames = T;
    result_spec.n_fft = n_fft;
    result_spec.hop_length = hop;
    result_spec.data = (float *)calloc((size_t)n_ch * n_freqs * T * 2, sizeof(float));

    for (int f = 0; f < n_freqs; f++) {
        for (int ch = 0; ch < n_ch; ch++) {
            int fs_idx = f * n_ch + ch;
            for (int t = 0; t < T; t++) {
                float sr = stft_repr[((size_t)fs_idx * T + t) * 2 + 0];
                float si = stft_repr[((size_t)fs_idx * T + t) * 2 + 1];
                float mr = masks_summed[((size_t)fs_idx * T + t) * 2 + 0];
                float mi = masks_summed[((size_t)fs_idx * T + t) * 2 + 1];
                float *dst = result_spec.at(ch, f, t);
                dst[0] = sr * mr - si * mi;  // real
                dst[1] = sr * mi + si * mr;  // imag
            }
        }
    }

    // DEBUG: Check mask and result spectrogram levels
    { float pk = 0; for (size_t i = 0; i < (size_t)fs*T*2; i++) { float a = fabsf(masks_summed[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG masks_summed peak (after avg): %.6f\n", pk); }
    { float pk = 0; size_t n = (size_t)n_ch*n_freqs*T*2; for (size_t i = 0; i < n; i++) { float a = fabsf(result_spec.data[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG result_spec peak: %.6f\n", pk); }

    // ── iSTFT → lead vocals ─────────────────────────────────────────
    int lead_frames = 0;
    out_lead = stft_inverse(result_spec, chunk_frames, &lead_frames);
    stft_free(&result_spec);

    // ── Backing vocals = original - lead ─────────────────────────────
    out_backing = (float *)malloc((size_t)chunk_frames * 2 * sizeof(float));
    for (int i = 0; i < chunk_frames * 2; i++) {
        out_backing[i] = chunk[i] - out_lead[i];
    }

    // DEBUG: Check output levels
    { float pk = 0; for (int i = 0; i < chunk_frames*2; i++) { float a = fabsf(out_lead[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG lead output peak: %.6f\n", pk); }
    { float pk = 0; for (int i = 0; i < chunk_frames*2; i++) { float a = fabsf(out_backing[i]); if (a>pk) pk=a; }
      fprintf(stderr, "[SuperSep] DBG backing output peak: %.6f\n", pk);
    }

    out_frames = chunk_frames;
    stft_free(&spec);
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────

SuperSep * supersep_init(const char * model_dir, int device_id) {
    auto *ctx = new SuperSep();
    ctx->model_dir = model_dir;
    ctx->device_id = device_id;

    // Configure session options
    ctx->session_opts.SetIntraOpNumThreads(4);
    ctx->session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CUDA EP if requested. The ORT GPU SDK bundles its own CUDA EP
    // (onnxruntime_providers_cuda.dll) so this doesn't depend on the system
    // CUDA Toolkit that CMake's CUDAToolkit_FOUND checks for.
    if (device_id >= 0) {
        try {
            OrtCUDAProviderOptions cuda_opts;
            memset(&cuda_opts, 0, sizeof(cuda_opts));
            cuda_opts.device_id = device_id;
            // Prevent exponential arena growth — allocate only what's needed
            cuda_opts.arena_extend_strategy = 1;  // kSameAsRequested (not kNextPowerOfTwo)
            // No hard memory cap — the model's attention layers need ~3GB per MatMul.
            // VRAM is reclaimed after job completion via supersep_release_models().
            ctx->session_opts.AppendExecutionProvider_CUDA(cuda_opts);
            fprintf(stderr, "[SuperSep] CUDA EP enabled (device %d, arena=exact)\n", device_id);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] CUDA EP failed: %s — falling back to CPU\n", e.what());
        }
    } else {
        fprintf(stderr, "[SuperSep] CPU mode (device_id=%d)\n", device_id);
    }

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

        std::vector<float *> s1_stems;
        std::vector<int> s1_counts;

        bool ok = bs_roformer_separate(
            ctx->s1_bs_roformer, audio, n_frames, BS_NUM_STEMS,
            s1_stems, s1_counts, cb, cancelled);

        if (!ok) {
            for (auto p : s1_stems) free(p);
            return nullptr;
        }

        cb(1, "Extracting stems...", 0.20f);
        fprintf(stderr, "[SuperSep] Stage 1: got %d stems\n", (int)s1_stems.size());

        // DEBUG: Log per-stem peak amplitudes to verify stem ordering
        for (int i = 0; i < (int)s1_stems.size(); i++) {
            float pk = 0;
            for (int j = 0; j < s1_counts[i] * 2; j++) {
                float a = fabsf(s1_stems[i][j]);
                if (a > pk) pk = a;
            }
            fprintf(stderr, "[SuperSep] Stage 1 stem[%d] (%s): peak=%.6f, frames=%d\n",
                    i, (i < N_STAGE1_STEMS ? STAGE1_STEMS[i].name : "?"), pk, s1_counts[i]);
        }

        // Assign stems — always output all Stage 1 stems.
        // Stems needed by later stages get duplicated: one copy goes to output,
        // the original pointer is kept for the downstream stage.
        for (int i = 0; i < N_STAGE1_STEMS && i < (int)s1_stems.size(); i++) {
            // Keep vocals/drums/other for later stages (hold the original pointer)
            if (i == 3 && stages[1]) { // Vocals → Stage 2
                s1_vocals = s1_stems[i];
                s1_vocal_frames = s1_counts[i];
            } else if (i == 4 && stages[2]) { // Drums → Stage 3
                s1_drums = s1_stems[i];
                s1_drum_frames = s1_counts[i];
            } else if (i == 5 && stages[3]) { // Other → Stage 4
                s1_other = s1_stems[i];
                s1_other_frames = s1_counts[i];
            }

            // Always add to output (duplicate buffer if held for later stage)
            float *buf = s1_stems[i];
            if (i == 3 && stages[1] || i == 4 && stages[2] || i == 5 && stages[3]) {
                size_t nbytes = (size_t)s1_counts[i] * 2 * sizeof(float);
                buf = (float *)malloc(nbytes);
                memcpy(buf, s1_stems[i], nbytes);
            }
            add_stem(stems, STAGE1_STEMS[i], buf, s1_counts[i]);
        }

        cb(1, "Stage 1 complete", 0.25f);
    } catch (const std::exception &e) {
        fprintf(stderr, "[SuperSep] Stage 1 failed: %s\n", e.what());
        cb(1, "Stage 1 failed", 0.25f);
        // Can't continue without stage 1
        return nullptr;
    }

    // ── STAGE 2: Vocal sub-separation (Mel-Band RoFormer) ─────────────
    if (stages[1] && s1_vocals) {
        cb(2, "Loading Mel-Band RoFormer model...", 0.30f);
        if (cancelled()) { free(s1_vocals); free(s1_drums); free(s1_other); return nullptr; }

        try {
            if (!ctx->s2_mel_band) {
                ctx->s2_mel_band = load_onnx_model(ctx, "mel_band_roformer_karaoke.onnx");
            }

            cb(2, "Splitting lead/backing vocals...", 0.35f);

            // Mel-Band RoFormer: chunking + overlap-add across full vocal track.
            // Chunk size is 352800 samples (~8s). Use 1s crossfade overlap.
            const int mb_chunk = MB_CHUNK_SAMPLES;
            const int mb_crossfade = 44100;  // 1 second
            const int mb_step = mb_chunk - mb_crossfade;
            const int nf = s1_vocal_frames;

            // Allocate accumulators for lead + backing
            std::vector<double> lead_accum(nf * 2, 0.0);
            std::vector<double> back_accum(nf * 2, 0.0);
            std::vector<double> weight_accum(nf * 2, 0.0);

            int n_chunks = (nf <= mb_chunk) ? 1 : (nf - mb_crossfade + mb_step - 1) / mb_step;
            if (n_chunks < 1) n_chunks = 1;

            // Crossfade window
            std::vector<float> fade_win(mb_chunk, 1.0f);
            int half_fade = mb_crossfade / 2;
            for (int i2 = 0; i2 < half_fade; i2++) {
                float t = (float)i2 / (float)half_fade;
                fade_win[i2] = t;
                fade_win[mb_chunk - 1 - i2] = t;
            }

            fprintf(stderr, "[SuperSep] Mel-Band: %d frames -> %d chunks (chunk=%d, step=%d)\n",
                    nf, n_chunks, mb_chunk, mb_step);

            bool any_ok = false;
            for (int c = 0; c < n_chunks; c++) {
                if (cancelled()) { free(s1_vocals); free(s1_drums); free(s1_other); return nullptr; }

                int start = c * mb_step;
                int end = std::min(start + mb_chunk, nf);
                int this_chunk = end - start;

                float pct = 0.35f + 0.10f * (float)c / (float)n_chunks;
                char msg[64];
                snprintf(msg, sizeof(msg), "Vocal chunk %d/%d...", c + 1, n_chunks);
                cb(2, msg, pct);

                // Pad to full chunk size (model expects fixed input)
                std::vector<float> chunk_buf(mb_chunk * 2, 0.0f);
                memcpy(chunk_buf.data(), s1_vocals + start * 2,
                       (size_t)this_chunk * 2 * sizeof(float));

                float *lead_chunk = nullptr, *back_chunk = nullptr;
                int chunk_out = 0;
                bool ok = mel_band_process_chunk(
                    ctx->s2_mel_band, chunk_buf.data(), mb_chunk,
                    lead_chunk, back_chunk, chunk_out);

                if (ok && lead_chunk && back_chunk) {
                    any_ok = true;
                    // Overlap-add with crossfade
                    for (int i2 = 0; i2 < this_chunk; i2++) {
                        float w = fade_win[i2];
                        if (c == 0 && i2 < half_fade) w = 1.0f;
                        if (c == n_chunks - 1 && i2 >= this_chunk - half_fade) w = 1.0f;

                        int dst = (start + i2) * 2;
                        if (dst + 1 < nf * 2) {
                            lead_accum[dst + 0] += lead_chunk[i2 * 2 + 0] * w;
                            lead_accum[dst + 1] += lead_chunk[i2 * 2 + 1] * w;
                            back_accum[dst + 0] += back_chunk[i2 * 2 + 0] * w;
                            back_accum[dst + 1] += back_chunk[i2 * 2 + 1] * w;
                            weight_accum[dst + 0] += w;
                            weight_accum[dst + 1] += w;
                        }
                    }
                }
                free(lead_chunk);
                free(back_chunk);
            }

            if (any_ok) {
                // Normalize and output
                float *lead_out = (float *)malloc((size_t)nf * 2 * sizeof(float));
                float *back_out = (float *)malloc((size_t)nf * 2 * sizeof(float));
                for (int i2 = 0; i2 < nf * 2; i2++) {
                    lead_out[i2] = (weight_accum[i2] > 1e-8) ? (float)(lead_accum[i2] / weight_accum[i2]) : 0.0f;
                    back_out[i2] = (weight_accum[i2] > 1e-8) ? (float)(back_accum[i2] / weight_accum[i2]) : 0.0f;
                }
                add_stem(stems, STAGE2_STEMS[0], lead_out, nf);
                add_stem(stems, STAGE2_STEMS[1], back_out, nf);
            } else {
                throw std::runtime_error("All mel-band chunks failed");
            }

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
            std::vector<int64_t> out_shape;
            auto output = run_wave_model(ctx->s3_mdx23c, s1_drums, s1_drum_frames, out_shape);

            int n_out = (out_shape.size() >= 2) ? std::min((int)out_shape[1], N_STAGE3_STEMS) : 0;
            for (int i = 0; i < n_out; i++) {
                float *sa = extract_stem_interleaved(output, out_shape, i, s1_drum_frames);
                add_stem(stems, STAGE3_STEMS[i], sa, s1_drum_frames);
            }
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

void supersep_release_models(SuperSep * ctx) {
    if (!ctx) return;
    if (ctx->s1_bs_roformer) { delete ctx->s1_bs_roformer; ctx->s1_bs_roformer = nullptr; }
    if (ctx->s2_mel_band)    { delete ctx->s2_mel_band;    ctx->s2_mel_band    = nullptr; }
    if (ctx->s3_mdx23c)      { delete ctx->s3_mdx23c;      ctx->s3_mdx23c      = nullptr; }
    if (ctx->s4_htdemucs)    { delete ctx->s4_htdemucs;    ctx->s4_htdemucs    = nullptr; }
    fprintf(stderr, "[SuperSep] Released all ONNX sessions (VRAM freed)\n");
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
void supersep_release_models(SuperSep *) {}

float * supersep_recombine(const SuperSepStem *, const float *, const bool *,
                           int, int *) {
    return nullptr;
}

#endif // HOT_STEP_SUPERSEP
