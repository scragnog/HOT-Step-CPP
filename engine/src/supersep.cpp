// supersep.cpp: GGML stem separation pipeline.
// Implements the SuperSep stages on GGML models.
// Part of HOT-Step CPP. MIT license.

#include "supersep.h"
#include "supersep-stft.h"
#include "bs-roformer-ggml.h"
#include "mdx23c-ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HOT_STEP_SUPERSEP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif


// ── Internal types ──────────────────────────────────────────────────────

struct SuperSep {
    std::string   model_dir;
    int           device_id;
    BsRoformer *  s1_bs_roformer;      // Stage 1 — GGML
    BsRoformer *  s2_mel_band;         // Stage 2 - GGML
    Mdx23c *      s3_mdx23c;           // Stage 3 - GGML
    // Leap Xe pair (SUPERSEP_STABLESTEP) — GGML, not ONNX Runtime.
    BsRoformer * leap_voc;             // vocal-target
    BsRoformer * leap_inst;            // instrumental-target

    SuperSep() : s1_bs_roformer(nullptr), s2_mel_band(nullptr),
                 s3_mdx23c(nullptr),
                 leap_voc(nullptr), leap_inst(nullptr) {}
    ~SuperSep() {
        if (s1_bs_roformer) { bsr_free(s1_bs_roformer); delete s1_bs_roformer; }
        if (s2_mel_band) { bsr_free(s2_mel_band); delete s2_mel_band; }
        if (s3_mdx23c) { mdx_free(s3_mdx23c); delete s3_mdx23c; }
        if (leap_voc)  { bsr_free(leap_voc);  delete leap_voc;  }
        if (leap_inst) { bsr_free(leap_inst); delete leap_inst; }
    }
};

// ── Constants ───────────────────────────────────────────────────────────

static const int SUPERSEP_SR = 44100;
static const float SILENCE_THRESHOLD_DB = -60.0f;

// BS-RoFormer STFT params. These are shared by every BS-RoFormer variant we
// carry — BS-Roformer-SW.yaml and pcunwa's BS-Roformer-Leap Xe configs agree
// on n_fft/hop/win/dim_freqs_in exactly, so the C++ STFT front-end is common.
static const int BS_N_FFT       = 2048;
static const int BS_HOP_LENGTH  = 512;
static const int BS_WIN_LENGTH  = 2048;
static const int BS_CHUNK_SIZE  = 588800;  // ~13.4s at 44100Hz (SW default)
static const int BS_N_FREQS     = BS_N_FFT / 2 + 1;  // 1025
static const int BS_NUM_STEMS   = 6;

// Per-checkpoint config. Only the stem count, the trained chunk length and the
// file name vary between variants; the band split and mask estimator are baked
// into the ONNX graph, so the engine never sees them.
struct BsRoformerModel {
    const char *filename;
    int         chunk_size;  // trained chunk length in frames @44.1kHz
    int         n_stems;     // model output stem count (num_stems in the yaml)
};

// 6-stem general-purpose model — drives every Cover Studio level. GGML.
static const BsRoformerModel BS_MODEL_SW = {
    "bs_roformer_sw-F32.gguf", BS_CHUNK_SIZE, BS_NUM_STEMS
};

// pcunwa/BS-Roformer-Leap "Xe" pair. Both are num_stems=1, dim=256, depth=16
// and differ only in target_instrument (vocals vs other). chunk_size 881559
// (~20.0 s) is the trained value from leap_xe_config_*.yaml — lowering it
// trades separation quality for attention VRAM if that becomes a problem.
//
// These run through GGML (bs-roformer-ggml.h), not ONNX Runtime, so they are
// .gguf. n_stems here is informational; the real value comes from the GGUF.
static const BsRoformerModel BS_MODEL_LEAP_XE_VOC  = {"bs_leap_xe_voc-F32.gguf",  881559, 1};
static const BsRoformerModel BS_MODEL_LEAP_XE_INST = {"bs_leap_xe_inst-F32.gguf", 881559, 1};

// Mel-Band RoFormer lookup tables (auto-generated from mel filterbank)
#include "mel_band_tables.inc"

// Stage 2 karaoke split. Mel-Band RoFormer shares the BS-RoFormer graph — its
// bands overlap and are gathered before the network (mel_band_tables.inc), so
// the only differences the engine sees are hop 441 and the gathered in_dim.
static const BsRoformerModel MEL_MODEL_KARAOKE = {
    "mel_band_karaoke-F32.gguf", MB_CHUNK_SAMPLES, 1
};


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

// SUPERSEP_VOCALS_ONLY: the karaoke model run on the full mix yields the
// vocal bed (output A) and everything else (output B).
static const StemDef VOCALS_ONLY_STEMS[] = {
    {"01_Vocals",       "Vocals",       "vocals",      1},
    {"02_Instrumental", "Instrumental", "instruments", 1},
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
static void add_stem(std::vector<SuperSepStem> &stems, const StemDef &def,
                     float *samples, int n_frames, bool hidden = false) {
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
    s.hidden = hidden;
    stems.push_back(s);
}

// ── BS-RoFormer STFT-based processing ───────────────────────────────────
//
// STFT and iSTFT are native (supersep-stft.h); the network only predicts the
// complex mask, which is applied here.
//   Preprocessing:  waveform -> STFT -> rearrange -> [in_dim, T]
//   Postprocessing: mask [2, T, F*C] -> complex multiply -> iSTFT
//
// Tensor layout follows ZFTurbo/MSS_ONNX_TensorRT/models/preprocess.py.

// Bitmask selecting every stem (default for callers that want all of them).
static const uint32_t BS_ALL_STEMS = 0xFFFFFFFFu;

// ── GGML BS-RoFormer (Leap Xe) ──────────────────────────────────────────
//
// Same STFT front-end, mask semantics and overlap-add as the ONNX path above;
// only the mask predictor differs. Single-stem by construction.

static bool bsr_ggml_process_chunk(
    BsRoformer *          m,
    const float *         chunk,        // interleaved stereo
    int                   chunk_frames,
    uint32_t              stem_mask,    // bit i set = stem i wanted
    std::vector<float *> &stem_outputs,
    std::vector<int> &    stem_frame_counts
) {
    const int n_fft   = BS_N_FFT;
    const int hop     = BS_HOP_LENGTH;
    const int n_freqs = BS_N_FREQS;
    const int n_ch    = 2;

    StftParams sp;
    sp.n_fft      = n_fft;
    sp.hop_length = hop;
    sp.n_channels = n_ch;
    ComplexSpec spec = stft_forward(chunk, chunk_frames, sp);
    const int T = spec.n_frames;

    const int in_dim = n_freqs * n_ch * 2;  // 4100
    const int fs     = n_freqs * n_ch;      // 2050

    // Model input: per time step, (f * n_ch + ch) * 2 + {re,im}
    std::vector<float> model_in((size_t) T * in_dim);
    std::vector<float> stft_repr((size_t) fs * T * 2);
    for (int t = 0; t < T; t++) {
        for (int f = 0; f < n_freqs; f++) {
            for (int ch = 0; ch < n_ch; ch++) {
                const float *c = spec.at(ch, f, t);
                int fs_idx = f * n_ch + ch;
                model_in[(size_t) t * in_dim + fs_idx * 2 + 0] = c[0];
                model_in[(size_t) t * in_dim + fs_idx * 2 + 1] = c[1];
                stft_repr[((size_t) fs_idx * T + t) * 2 + 0]   = c[0];
                stft_repr[((size_t) fs_idx * T + t) * 2 + 1]   = c[1];
            }
        }
    }

    const int n_stems = m->cfg.n_stems;
    std::vector<float> mask((size_t) n_stems * in_dim * T);
    bsr_forward(m, model_in.data(), T, mask.data());

    // mask layout per stem: [fs][t][{re,im}] — identical to the ONNX graph's
    // [1, S, fs, T, 2], so the apply loop is unchanged.
    for (int s = 0; s < n_stems; s++) {
        // Unwanted stems skip the mask multiply and the iSTFT. The network
        // predicts every stem in one forward regardless, but the per-stem
        // inverse transform is pure CPU work — 5/6 of it is thrown away by a
        // 1-of-6 caller like VOCALS_ONLY.
        if (!(stem_mask & (1u << s))) {
            stem_outputs.push_back(nullptr);
            stem_frame_counts.push_back(0);
            continue;
        }

        ComplexSpec stem_spec;
        stem_spec.n_channels = n_ch;
        stem_spec.n_freqs    = n_freqs;
        stem_spec.n_frames   = T;
        stem_spec.n_fft      = n_fft;
        stem_spec.hop_length = hop;
        stem_spec.data = (float *) calloc((size_t) n_ch * n_freqs * T * 2, sizeof(float));

        for (int f = 0; f < n_freqs; f++) {
            for (int ch = 0; ch < n_ch; ch++) {
                int fs_idx = f * n_ch + ch;
                for (int t = 0; t < T; t++) {
                    float sr = stft_repr[((size_t) fs_idx * T + t) * 2 + 0];
                    float si = stft_repr[((size_t) fs_idx * T + t) * 2 + 1];
                    size_t off = ((size_t) s * fs * T + (size_t) fs_idx * T + t) * 2;
                    float mr = mask[off + 0];
                    float mi = mask[off + 1];
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

// Chunked overlap-add, mirroring bs_roformer_separate (1 s crossfade).
static bool bs_roformer_separate_ggml(
    BsRoformer *          m,
    const float *         audio,
    int                   n_frames,
    int                   chunk_size,
    uint32_t              stem_mask,
    std::vector<float *> &stem_outputs,
    std::vector<int> &    stem_frame_counts,
    std::function<void(int, const char *, float)> cb,
    std::function<bool()> cancelled
) {
    const int n_stems = m->cfg.n_stems;

    if (n_frames <= chunk_size) {
        cb(1, "Running BS-RoFormer (GGML) inference...", 0.10f);
        if (n_frames < chunk_size) {
            std::vector<float> padded((size_t) chunk_size * 2, 0.0f);
            memcpy(padded.data(), audio, (size_t) n_frames * 2 * sizeof(float));
            bool ok = bsr_ggml_process_chunk(m, padded.data(), chunk_size, stem_mask,
                                             stem_outputs, stem_frame_counts);
            if (ok) {
                for (size_t s = 0; s < stem_outputs.size(); s++) {
                    if (stem_outputs[s]) stem_frame_counts[s] = n_frames;
                }
            }
            return ok;
        }
        return bsr_ggml_process_chunk(m, audio, n_frames, stem_mask,
                                      stem_outputs, stem_frame_counts);
    }

    const int crossfade = 44100;
    int step     = chunk_size - crossfade;
    int n_chunks = (n_frames - crossfade + step - 1) / step;
    if (n_chunks < 1) n_chunks = 1;

    fprintf(stderr, "[SuperSep] GGML chunking: %d frames into %d chunks "
            "(chunk=%d, crossfade=%d)\n", n_frames, n_chunks, chunk_size, crossfade);

    // Only allocate accumulators for requested stems — each pair is ~32 MB per
    // stem for a 3-minute song at 44.1 kHz.
    std::vector<std::vector<float>> accum(n_stems);
    std::vector<std::vector<float>> weight(n_stems);
    for (int s = 0; s < n_stems; s++) {
        if (!(stem_mask & (1u << s))) continue;
        accum[s].assign((size_t) n_frames * 2, 0.0f);
        weight[s].assign((size_t) n_frames * 2, 0.0f);
    }

    std::vector<float> fade_window((size_t) chunk_size, 1.0f);
    int half_fade = crossfade / 2;
    for (int i = 0; i < half_fade; i++) {
        float t = (float) i / (float) half_fade;
        fade_window[(size_t) i] = t;
        fade_window[(size_t) (chunk_size - 1 - i)] = t;
    }

    for (int c = 0; c < n_chunks; c++) {
        if (cancelled()) return false;

        int start      = c * step;
        int end        = std::min(start + chunk_size, n_frames);
        int this_chunk = end - start;

        char msg[64];
        snprintf(msg, sizeof(msg), "Processing chunk %d/%d...", c + 1, n_chunks);
        cb(1, msg, 0.10f + 0.15f * (float) c / (float) n_chunks);

        std::vector<float> chunk_buf((size_t) chunk_size * 2, 0.0f);
        memcpy(chunk_buf.data(), audio + (size_t) start * 2,
               (size_t) this_chunk * 2 * sizeof(float));

        std::vector<float *> chunk_stems;
        std::vector<int>     chunk_counts;
        if (!bsr_ggml_process_chunk(m, chunk_buf.data(), chunk_size, stem_mask,
                                    chunk_stems, chunk_counts)) {
            for (auto p : chunk_stems) free(p);
            return false;
        }

        for (int s = 0; s < (int) chunk_stems.size() && s < n_stems; s++) {
            if (!chunk_stems[s]) continue;  // stem not requested
            for (int i = 0; i < this_chunk; i++) {
                float w = fade_window[(size_t) i];
                if (c == 0 && i < half_fade) w = 1.0f;
                if (c == n_chunks - 1 && i >= this_chunk - half_fade) w = 1.0f;
                int dst = (start + i) * 2;
                if (dst + 1 < n_frames * 2) {
                    accum[s][dst + 0]  += chunk_stems[s][i * 2 + 0] * w;
                    accum[s][dst + 1]  += chunk_stems[s][i * 2 + 1] * w;
                    weight[s][dst + 0] += w;
                    weight[s][dst + 1] += w;
                }
            }
            free(chunk_stems[s]);
        }
    }

    for (int s = 0; s < n_stems; s++) {
        if (!(stem_mask & (1u << s))) {
            stem_outputs.push_back(nullptr);
            stem_frame_counts.push_back(0);
            continue;
        }
        float *out = (float *) malloc((size_t) n_frames * 2 * sizeof(float));
        for (int i = 0; i < n_frames * 2; i++) {
            out[i] = (weight[s][i] > 1e-8f) ? accum[s][i] / weight[s][i] : 0.0f;
        }
        stem_outputs.push_back(out);
        stem_frame_counts.push_back(n_frames);
    }
    return true;
}

// Load a BS-RoFormer GGUF into *slot on first use. Returns false on failure
// (missing/corrupt file); the slot is left null so a later attempt can retry.
// True if the model file is present. Lets optional models (the Leap pair) be
// probed without bsr_load printing FATAL for what is a normal fallback.
static bool bsr_model_present(SuperSep * ctx, const BsRoformerModel & cfg) {
    std::string path = ctx->model_dir + "/" + cfg.filename;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static bool bsr_load_slot(SuperSep * ctx, BsRoformer ** slot, const BsRoformerModel & cfg) {
    if (*slot) return true;
    std::string path = ctx->model_dir + "/" + cfg.filename;
    BsRoformer *m = new BsRoformer();
    if (!bsr_load(m, path.c_str())) {
        delete m;
        return false;
    }
    *slot = m;
    return true;
}

// Run one single-stem BS-RoFormer variant (a Leap Xe checkpoint) over the full
// input and return its neural output. Loads the GGUF into *slot on first use.
// Returns a malloc'd interleaved stereo buffer, or nullptr on failure /
// cancellation. Caller owns the buffer.
//
// This path is GGML, not ONNX Runtime — see bs-roformer-ggml.h for why. The
// short version: the time transformer's materialised attention at the trained
// 20 s chunk is 8.5 GB, and ggml_flash_attn_ext never builds it.
static float * leap_run_single(
    SuperSep *                ctx,
    BsRoformer **             slot,
    const BsRoformerModel &   cfg,
    const float *             audio,
    int                       n_frames,
    int *                     out_frames,
    std::function<void(int, const char *, float)> cb,
    std::function<bool()>     cancelled
) {
    if (!bsr_load_slot(ctx, slot, cfg)) return nullptr;

    std::vector<float *> outs;
    std::vector<int>     counts;
    if (!bs_roformer_separate_ggml(*slot, audio, n_frames, cfg.chunk_size,
                                   BS_ALL_STEMS, outs, counts, cb, cancelled)) {
        for (auto p : outs) free(p);
        return nullptr;
    }

    // num_stems=1: index 0 is the target. Free anything else defensively in
    // case a future checkpoint ever carries more than one stem.
    float *stem = nullptr;
    for (size_t i = 0; i < outs.size(); i++) {
        if (i == 0) { stem = outs[i]; *out_frames = counts[i]; }
        else        { free(outs[i]); }
    }
    return stem;
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
    BsRoformer *model,            // GGML Mel-Band RoFormer
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

    // ── Gather mel-band frequency indices → model input [in_dim, T] ──
    // Mel bands OVERLAP, so this is a gather (with repeats), not a slice. The
    // GGML graph wants time-major with (f,c) innermost — 'b t (f c)' — whereas
    // the old ONNX graph took 'b f t c'. Everything downstream of the network
    // is unchanged: the mask still comes back as [stem][f][t][re,im].
    const int n_gathered = MB_N_FREQ_INDICES_STEREO;
    const int in_dim     = n_gathered * 2;
    std::vector<float> model_input((size_t)T * in_dim);

    for (int i = 0; i < MB_N_FREQ_INDICES_MONO; i++) {
        int mono_f = MB_FREQ_INDICES_MONO[i];
        for (int ch = 0; ch < n_ch; ch++) {
            int stereo_gather_idx = i * n_ch + ch;   // destination band slot
            int stereo_src_idx    = mono_f * n_ch + ch;  // source in stft_repr
            for (int t = 0; t < T; t++) {
                model_input[(size_t)t * in_dim + stereo_gather_idx * 2 + 0] =
                    stft_repr[((size_t)stereo_src_idx * T + t) * 2 + 0];
                model_input[(size_t)t * in_dim + stereo_gather_idx * 2 + 1] =
                    stft_repr[((size_t)stereo_src_idx * T + t) * 2 + 1];
            }
        }
    }

    // ── Run GGML inference ───────────────────────────────────────────
    const int n_stems_out = model->cfg.n_stems;
    std::vector<float> mask_buf((size_t)n_stems_out * in_dim * T);
    bsr_forward(model, model_input.data(), T, mask_buf.data());
    const float *mask_data = mask_buf.data();

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

// ── MDX-style STFT processing (MDX23C / HTDemucs) ───────────────────────
//
// Both MDX23C and HTDemucs models expect STFT input with stripped STFT layers.
// Input layout:  [1, 4, dim_f, T]  where 4 = stereo(2) × complex(2)
//   Channel order: [left_real, left_imag, right_real, right_imag]
//
// MDX23C output:  [1, n_stems, 4, dim_f, T]  (5D)
// HTDemucs output: [1, n_stems*4, dim_f, T]  (4D, channels flattened)
//
// STFT params:  n_fft=4096, hop=1024, freq truncated to dim_f.
// Chunk size:   (dim_t - 1) * hop samples (produces exactly dim_t time frames).
//

static bool mdx_process_chunk(
    Mdx23c *model,
    const float *chunk,       // interleaved stereo PCM
    int chunk_frames,         // per-channel frame count
    int n_fft,
    int hop,
    int dim_f,                // freq truncation (1024 for MDX23C)
    int n_stems,              // max stems to extract
    std::vector<float *> &stem_outputs,
    std::vector<int> &stem_counts) {

    const int n_ch = 2;
    const int full_freqs = n_fft / 2 + 1;  // 2049

    // 1. STFT
    StftParams sp;
    sp.n_fft = n_fft;
    sp.hop_length = hop;
    sp.n_channels = n_ch;
    ComplexSpec spec = stft_forward(chunk, chunk_frames, sp);
    int T = spec.n_frames;

    fprintf(stderr, "[SuperSep] MDX STFT: %d freqs x %d frames (dim_f=%d)\n",
            full_freqs, T, dim_f);

    // 2. Build model input [1, 4, dim_f, T]
    //    Layout: (c ri) f t  →  ch0_real, ch0_imag, ch1_real, ch1_imag
    size_t input_size = (size_t)4 * dim_f * T;
    std::vector<float> model_input(input_size, 0.0f);

    for (int ch = 0; ch < n_ch; ch++) {
        for (int f = 0; f < dim_f && f < full_freqs; f++) {
            for (int t = 0; t < T; t++) {
                const float *c = spec.at(ch, f, t);
                // Real → channel ch*2, Imag → channel ch*2+1
                model_input[((size_t)(ch*2) * dim_f + f) * T + t] = c[0];
                model_input[((size_t)(ch*2+1) * dim_f + f) * T + t] = c[1];
            }
        }
    }

    // 3. Run GGML inference. Input packing above is already [c][f][t] with t
    //    fastest, which is exactly the layout mdx_forward expects.
    const int n_inst = model->cfg.n_instruments;
    const int out_T = T, out_df = dim_f;
    std::vector<float> out_buf((size_t) n_inst * 4 * dim_f * T);
    mdx_forward(model, model_input.data(), T, out_buf.data());
    const float *out_data = out_buf.data();
    int actual_stems = n_inst;
    if (actual_stems > n_stems) actual_stems = n_stems;

    fprintf(stderr, "[SuperSep] MDX (GGML): %d stems, dim_f=%d, T=%d\n",
            actual_stems, out_df, out_T);

    // 5. iSTFT each stem
    for (int s = 0; s < actual_stems; s++) {
        ComplexSpec stem_spec;
        stem_spec.n_channels = n_ch;
        stem_spec.n_freqs = full_freqs;
        stem_spec.n_frames = out_T;
        stem_spec.n_fft = n_fft;
        stem_spec.hop_length = hop;
        stem_spec.data = (float *)calloc((size_t)n_ch * full_freqs * out_T * 2, sizeof(float));

        for (int ch = 0; ch < n_ch; ch++) {
            for (int f = 0; f < out_df && f < full_freqs; f++) {
                for (int t = 0; t < out_T; t++) {
                    // out_data is [stem][ch*2+{re,im}][f][t], t fastest.
                    const size_t base = (size_t)s * 4 * out_df * out_T;
                    float re = out_data[base + (size_t)(ch*2)   * out_df * out_T + (size_t)f * out_T + t];
                    float im = out_data[base + (size_t)(ch*2+1) * out_df * out_T + (size_t)f * out_T + t];
                    float *dst = stem_spec.at(ch, f, t);
                    dst[0] = re;
                    dst[1] = im;
                }
            }
        }

        int out_frames = 0;
        float *audio = stft_inverse(stem_spec, chunk_frames, &out_frames);
        stft_free(&stem_spec);


        stem_outputs.push_back(audio);
        stem_counts.push_back(out_frames);
    }

    stft_free(&spec);
    return true;
}

// ── Public API ──────────────────────────────────────────────────────────

SuperSep * supersep_init(const char * model_dir, int device_id) {
    auto *ctx = new SuperSep();
    ctx->model_dir = model_dir;
    ctx->device_id = device_id;

    // Backend selection (CUDA / Vulkan / Metal / CPU) is handled by GGML via
    // backend_init(); device_id is retained for API compatibility.

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

    // ── STABLESTEP: dual Leap Xe pass, both stems are neural ─────────────
    // Runs the vocal-target checkpoint and the instrumental-target checkpoint
    // over the same input and keeps each one's own prediction, discarding both
    // arithmetic complements. Neither output is an error bucket: the vocal that
    // gets re-applied after the SA3 refine is the vocal model's optimised
    // output, and the instrumental that feeds SA3 is the instrumental model's.
    //
    // The two sessions are loaded and released strictly in sequence so peak
    // VRAM stays at one model, matching the stage-1 → stage-2 handoff below.
    if (level == SUPERSEP_STABLESTEP) {
        // bs_roformer_separate reports progress in the 0.10..0.25 band; remap
        // each pass into its own slice so the UI shows one monotonic sweep.
        auto sub_cb = [&](float lo, float hi) {
            return [=](int st, const char *m, float p) {
                float t = (p - 0.10f) / 0.15f;
                t = (t < 0.0f) ? 0.0f : (t > 1.0f ? 1.0f : t);
                cb(st, m, lo + (hi - lo) * t);
            };
        };

        float *vocals = nullptr, *inst = nullptr;
        int voc_frames = 0, inst_frames = 0;

        try {
            // ── Pass 1: vocal-target ─────────────────────────────────────
            cb(1, "Loading Leap Xe vocal model...", 0.03f);
            if (cancelled()) return nullptr;
            vocals = leap_run_single(ctx, &ctx->leap_voc, BS_MODEL_LEAP_XE_VOC,
                                     audio, n_frames, &voc_frames,
                                     sub_cb(0.05f, 0.48f), cancelled);
            if (!vocals) throw std::runtime_error("Leap Xe vocal pass produced no stem");

            // Release before the second model loads — peak VRAM = one model.
            if (ctx->leap_voc) {
                bsr_free(ctx->leap_voc);
                delete ctx->leap_voc;
                ctx->leap_voc = nullptr;
                fprintf(stderr, "[SuperSep] Released Leap Xe vocal model (VRAM freed for instrumental pass)\n");
            }

            // ── Pass 2: instrumental-target ──────────────────────────────
            cb(1, "Loading Leap Xe instrumental model...", 0.50f);
            if (cancelled()) { free(vocals); return nullptr; }
            inst = leap_run_single(ctx, &ctx->leap_inst, BS_MODEL_LEAP_XE_INST,
                                   audio, n_frames, &inst_frames,
                                   sub_cb(0.52f, 0.92f), cancelled);
            if (!inst) throw std::runtime_error("Leap Xe instrumental pass produced no stem");

            if (ctx->leap_inst) {
                bsr_free(ctx->leap_inst);
                delete ctx->leap_inst;
                ctx->leap_inst = nullptr;
                fprintf(stderr, "[SuperSep] Released Leap Xe instrumental model\n");
            }

            // Same stem identities as VOCALS_ONLY so the server/StableStep side
            // needs no changes — only how the two buffers were produced differs.
            // add_stem takes ownership (and frees outright if the stem is
            // silent) — drop our handles so the catch path can't double-free.
            add_stem(stems, VOCALS_ONLY_STEMS[0], vocals, voc_frames);
            vocals = nullptr;
            add_stem(stems, VOCALS_ONLY_STEMS[1], inst, inst_frames);
            inst = nullptr;
            cb(1, "Dual-model vocal split complete", 0.95f);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] StableStep dual-Leap split failed: %s\n", e.what());
            free(vocals);
            free(inst);
            for (auto &s : stems) free(s.samples);
            return nullptr;
        }

        SuperSepResult *result = (SuperSepResult *)malloc(sizeof(SuperSepResult));
        result->n_stems = (int)stems.size();
        result->stems = (SuperSepStem *)malloc(sizeof(SuperSepStem) * stems.size());
        memcpy(result->stems, stems.data(), sizeof(SuperSepStem) * stems.size());
        fprintf(stderr, "[SuperSep] Done — %d stems extracted (dual Leap Xe)\n", result->n_stems);
        return result;
    }

    // ── VOCALS_ONLY: BS-RoFormer pass, 2-stem output ─────────────────────
    // The 6-stem model computes all stems in one forward anyway; we keep only
    // the Vocals stem (lead AND backing vocals — unlike the karaoke model,
    // whose "vocals" is lead-only and leaves backing vocals in the
    // instrumental, where downstream SA3 refinement would scramble them) and
    // derive Instrumental as mix − vocals. The complement guarantees
    // vocals + instrumental reconstructs the input exactly — no residual is
    // lost to the split. Used by the StableStep post-processing stage.
    if (level == SUPERSEP_VOCALS_ONLY) {
        cb(1, "Loading BS-RoFormer model...", 0.05f);
        if (cancelled()) return nullptr;
        try {
            if (!bsr_load_slot(ctx, &ctx->s1_bs_roformer, BS_MODEL_SW)) {
                throw std::runtime_error(std::string("cannot load ") + BS_MODEL_SW.filename);
            }

            std::vector<float *> s1_stems;
            std::vector<int> s1_counts;
            // Only stem 3 (Vocals) is kept — the mask skips the mask-multiply,
            // iSTFT and overlap-add accumulators for the other five.
            bool ok = bs_roformer_separate_ggml(
                ctx->s1_bs_roformer, audio, n_frames, BS_MODEL_SW.chunk_size,
                1u << 3, s1_stems, s1_counts, cb, cancelled);
            if (!ok) {
                for (auto p : s1_stems) free(p);
                return nullptr;
            }

            // Vocals = stage-1 stem index 3; free the rest immediately.
            float *vocals = nullptr;
            int voc_frames = 0;
            for (int i = 0; i < (int)s1_stems.size(); i++) {
                if (i == 3) { vocals = s1_stems[i]; voc_frames = s1_counts[i]; }
                else free(s1_stems[i]);
            }
            if (!vocals) throw std::runtime_error("BS-RoFormer produced no vocals stem");

            cb(1, "Deriving instrumental (mix - vocals)...", 0.90f);
            int nf = std::min(voc_frames, n_frames);
            float *inst = (float *)malloc((size_t)n_frames * 2 * sizeof(float));
            for (int i2 = 0; i2 < n_frames * 2; i2++) {
                float v = (i2 < nf * 2) ? vocals[i2] : 0.0f;
                inst[i2] = audio[i2] - v;
            }
            add_stem(stems, VOCALS_ONLY_STEMS[0], vocals, voc_frames);
            add_stem(stems, VOCALS_ONLY_STEMS[1], inst, n_frames);
            cb(1, "Vocal split complete", 0.95f);
        } catch (const std::exception &e) {
            fprintf(stderr, "[SuperSep] Vocals-only failed: %s\n", e.what());
            for (auto &s : stems) free(s.samples);
            return nullptr;
        }

        SuperSepResult *result = (SuperSepResult *)malloc(sizeof(SuperSepResult));
        result->n_stems = (int)stems.size();
        result->stems = (SuperSepStem *)malloc(sizeof(SuperSepStem) * stems.size());
        memcpy(result->stems, stems.data(), sizeof(SuperSepStem) * stems.size());
        fprintf(stderr, "[SuperSep] Done — %d stems extracted (vocals-only)\n", result->n_stems);
        return result;
    }

    int stages[] = {1, 0, 0, 0};
    if (level >= SUPERSEP_VOCAL_SPLIT && level <= SUPERSEP_MAXIMUM) stages[1] = 1;
    if (level >= SUPERSEP_FULL        && level <= SUPERSEP_MAXIMUM) stages[2] = 1;

    // Stems from stage 1 that feed into later stages
    float *s1_vocals = nullptr, *s1_drums = nullptr, *s1_other = nullptr;
    int s1_vocal_frames = 0, s1_drum_frames = 0, s1_other_frames = 0;

    // ── STAGE 0: Leap Xe vocal/instrumental front door ───────────────
    // BS-Roformer-Leap Xe is the strongest vocal/instrumental separator we
    // carry, so when its pair is installed it does that split FIRST and the
    // rest of the pipeline works on its outputs:
    //
    //   Leap vocals       -> the Vocals stem, and the Stage 2 lead/backing input
    //   Leap instrumental -> the Stage 1 input, so the 6-stem model never has
    //                        to fight vocal energy for the instrument stems
    //
    // Both stems are each model's own prediction; neither is a mix-minus
    // residual. Costs two extra full passes over the audio — the quality of
    // every downstream stem depends on this split, so it is worth paying.
    //
    // Falls back to running Stage 1 on the raw mix when the pair is absent.
    const float * stage1_input = audio;
    std::vector<float> leap_inst_buf;
    float * leap_vocals = nullptr;
    int     leap_vocal_frames = 0;
    bool    use_leap = false;

    if (bsr_model_present(ctx, BS_MODEL_LEAP_XE_VOC) &&
        bsr_model_present(ctx, BS_MODEL_LEAP_XE_INST) &&
        bsr_load_slot(ctx, &ctx->leap_voc, BS_MODEL_LEAP_XE_VOC)) {
        cb(1, "Separating vocals (Leap Xe)...", 0.02f);
        if (cancelled()) return nullptr;
        int vf = 0;
        float *v = leap_run_single(ctx, &ctx->leap_voc, BS_MODEL_LEAP_XE_VOC,
                                   audio, n_frames, &vf,
                                   [&](int st, const char *m, float) { cb(st, m, 0.03f); },
                                   cancelled);
        if (ctx->leap_voc) {
            bsr_free(ctx->leap_voc); delete ctx->leap_voc; ctx->leap_voc = nullptr;
        }
        if (v) {
            cb(1, "Separating instruments (Leap Xe)...", 0.05f);
            int inf = 0;
            float *inst = nullptr;
            if (bsr_load_slot(ctx, &ctx->leap_inst, BS_MODEL_LEAP_XE_INST)) {
                inst = leap_run_single(ctx, &ctx->leap_inst, BS_MODEL_LEAP_XE_INST,
                                       audio, n_frames, &inf,
                                       [&](int st, const char *m, float) { cb(st, m, 0.07f); },
                                       cancelled);
            }
            if (ctx->leap_inst) {
                bsr_free(ctx->leap_inst); delete ctx->leap_inst; ctx->leap_inst = nullptr;
            }
            if (inst) {
                leap_vocals       = v;
                leap_vocal_frames = vf;
                leap_inst_buf.assign(inst, inst + (size_t)inf * 2);
                free(inst);
                stage1_input = leap_inst_buf.data();
                use_leap     = true;
                fprintf(stderr, "[SuperSep] Leap Xe front door: vocals %d frames, "
                                "instrumental %d frames -> Stage 1\n", vf, inf);
            } else {
                free(v);
                fprintf(stderr, "[SuperSep] Leap instrumental pass failed — "
                                "falling back to Stage 1 on the raw mix\n");
            }
        }
    }
    if (!use_leap) {
        fprintf(stderr, "[SuperSep] Leap Xe pair unavailable — Stage 1 runs on the mix\n");
    }

    // ── STAGE 1: Primary 6-stem split ────────────────────────────────
    cb(1, "Loading BS-RoFormer model...", 0.08f);
    if (cancelled()) { free(leap_vocals); return nullptr; }

    try {
        if (!bsr_load_slot(ctx, &ctx->s1_bs_roformer, BS_MODEL_SW)) {
            throw std::runtime_error(std::string("cannot load ") + BS_MODEL_SW.filename);
        }

        std::vector<float *> s1_stems;
        std::vector<int> s1_counts;

        // Vocals (index 3) come from Leap when the front door ran, so the
        // 6-stem model's own vocals output is not needed.
        const uint32_t s1_mask = use_leap ? (BS_ALL_STEMS & ~(1u << 3)) : BS_ALL_STEMS;

        bool ok = bs_roformer_separate_ggml(
            ctx->s1_bs_roformer, stage1_input, n_frames, BS_MODEL_SW.chunk_size,
            s1_mask, s1_stems, s1_counts, cb, cancelled);

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
            // Vocals come from the Leap front door when it ran; the 6-stem
            // model's index 3 was masked off and is null.
            if (i == 3 && use_leap) continue;
            if (!s1_stems[i]) continue;

            // Keep drums/vocals for later stages (hold the original pointer)
            if (i == 1 && stages[2]) { // Drums (index 1) → Stage 3
                s1_drums = s1_stems[i];
                s1_drum_frames = s1_counts[i];
            } else if (i == 3 && stages[1]) { // Vocals (index 3) → Stage 2
                s1_vocals = s1_stems[i];
                s1_vocal_frames = s1_counts[i];
            }

            // Always add to output (duplicate buffer if held for later stage)
            float *buf = s1_stems[i];
            bool is_held = (i == 1 && stages[2]) || (i == 3 && stages[1]);
            if (is_held) {
                size_t nbytes = (size_t)s1_counts[i] * 2 * sizeof(float);
                buf = (float *)malloc(nbytes);
                memcpy(buf, s1_stems[i], nbytes);
            }
            add_stem(stems, STAGE1_STEMS[i], buf, s1_counts[i], /*hidden=*/is_held);
        }

        // Leap vocals take the Vocals slot and feed Stage 2.
        if (use_leap && leap_vocals) {
            if (stages[1]) {
                s1_vocals       = leap_vocals;   // ownership moves to the stage-2 path
                s1_vocal_frames = leap_vocal_frames;
                size_t nbytes = (size_t)leap_vocal_frames * 2 * sizeof(float);
                float *copy = (float *)malloc(nbytes);
                memcpy(copy, leap_vocals, nbytes);
                add_stem(stems, STAGE1_STEMS[3], copy, leap_vocal_frames, /*hidden=*/true);
            } else {
                add_stem(stems, STAGE1_STEMS[3], leap_vocals, leap_vocal_frames);
            }
            leap_vocals = nullptr;  // owned by stems / s1_vocals from here
        }

        // The instrumental buffer has served its purpose as the Stage 1 input.
        leap_inst_buf.clear();
        leap_inst_buf.shrink_to_fit();

        cb(1, "Stage 1 complete", 0.25f);
    } catch (const std::exception &e) {
        fprintf(stderr, "[SuperSep] Stage 1 failed: %s\n", e.what());
        cb(1, "Stage 1 failed", 0.25f);
        free(leap_vocals);
        for (auto &s : stems) free(s.samples);
        // Can't continue without stage 1
        return nullptr;
    }

    // ── STAGE 2: Vocal sub-separation (Mel-Band RoFormer) ─────────────
    if (stages[1] && s1_vocals) {
        // Release Stage 1 model to free VRAM before loading Stage 2
        if (ctx->s1_bs_roformer) {
            bsr_free(ctx->s1_bs_roformer);
            delete ctx->s1_bs_roformer;
            ctx->s1_bs_roformer = nullptr;
            fprintf(stderr, "[SuperSep] Released BS-RoFormer (VRAM freed for Stage 2)\n");
        }

        cb(2, "Loading Mel-Band RoFormer model...", 0.30f);
        if (cancelled()) { free(s1_vocals); free(s1_drums); free(s1_other); return nullptr; }

        try {
            if (!ctx->s2_mel_band) {
                if (!bsr_load_slot(ctx, &ctx->s2_mel_band, MEL_MODEL_KARAOKE)) {
                    throw std::runtime_error("cannot load mel-band model");
                }
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
        // Release previous stage models to free VRAM
        if (ctx->s2_mel_band) { bsr_free(ctx->s2_mel_band); delete ctx->s2_mel_band; ctx->s2_mel_band = nullptr; }
        if (ctx->s1_bs_roformer) { bsr_free(ctx->s1_bs_roformer); delete ctx->s1_bs_roformer; ctx->s1_bs_roformer = nullptr; }
        fprintf(stderr, "[SuperSep] Released previous models (VRAM freed for Stage 3)\n");

        cb(3, "Loading MDX23C DrumSep model...", 0.50f);
        if (cancelled()) { free(s1_drums); free(s1_other); return nullptr; }

        try {
            if (!ctx->s3_mdx23c) {
                Mdx23c *mm = new Mdx23c();
                if (!mdx_load(mm, (ctx->model_dir + "/mdx23c_drumsep-F32.gguf").c_str())) {
                    delete mm;
                    throw std::runtime_error("cannot load mdx23c_drumsep-F32.gguf");
                }
                ctx->s3_mdx23c = mm;
            }

            cb(3, "Splitting drums...", 0.55f);

            // MDX23C: n_fft=4096, hop=1024, dim_f=1024, dim_t=256
            // chunk_frames = (256-1) * 1024 = 261120 (~5.9s)
            const int mdx_nfft = 4096, mdx_hop = 1024, mdx_dimf = 1024;
            const int mdx_chunk = 255 * mdx_hop;  // 261120
            const int mdx_xfade = 44100;
            const int mdx_step = mdx_chunk - mdx_xfade;
            const int nf = s1_drum_frames;

            int n_chunks = (nf <= mdx_chunk) ? 1 : (nf - mdx_xfade + mdx_step - 1) / mdx_step;
            if (n_chunks < 1) n_chunks = 1;

            fprintf(stderr, "[SuperSep] MDX23C: %d frames -> %d chunks\n", nf, n_chunks);

            // Accumulators for up to 6 drum stems
            const int max_drum_stems = N_STAGE3_STEMS;
            std::vector<std::vector<double>> drum_accum(max_drum_stems, std::vector<double>(nf * 2, 0.0));
            std::vector<double> drum_weight(nf * 2, 0.0);
            int found_stems = 0;

            // Crossfade window
            std::vector<float> fade_win(mdx_chunk, 1.0f);
            int half_fade = mdx_xfade / 2;
            for (int i2 = 0; i2 < half_fade; i2++) {
                float t = (float)i2 / (float)half_fade;
                fade_win[i2] = t;
                fade_win[mdx_chunk - 1 - i2] = t;
            }

            for (int c = 0; c < n_chunks; c++) {
                if (cancelled()) { free(s1_drums); free(s1_other); return nullptr; }

                int start = c * mdx_step;
                int end = std::min(start + mdx_chunk, nf);
                int this_chunk = end - start;

                float pct = 0.55f + 0.15f * (float)c / (float)n_chunks;
                char msg[64]; snprintf(msg, sizeof(msg), "Drum chunk %d/%d...", c + 1, n_chunks);
                cb(3, msg, pct);

                std::vector<float> chunk_buf(mdx_chunk * 2, 0.0f);
                memcpy(chunk_buf.data(), s1_drums + start * 2, (size_t)this_chunk * 2 * sizeof(float));

                std::vector<float *> chunk_stems;
                std::vector<int> chunk_counts;
                if (!mdx_process_chunk(ctx->s3_mdx23c, chunk_buf.data(), mdx_chunk,
                                       mdx_nfft, mdx_hop, mdx_dimf, max_drum_stems,
                                       chunk_stems, chunk_counts)) {
                    for (auto p : chunk_stems) free(p);
                    continue;
                }

                found_stems = std::max(found_stems, (int)chunk_stems.size());

                for (int s = 0; s < (int)chunk_stems.size(); s++) {
                    for (int i2 = 0; i2 < this_chunk; i2++) {
                        float w = fade_win[i2];
                        if (c == 0 && i2 < half_fade) w = 1.0f;
                        if (c == n_chunks - 1 && i2 >= this_chunk - half_fade) w = 1.0f;
                        int dst = (start + i2) * 2;
                        if (dst + 1 < nf * 2) {
                            drum_accum[s][dst + 0] += chunk_stems[s][i2 * 2 + 0] * w;
                            drum_accum[s][dst + 1] += chunk_stems[s][i2 * 2 + 1] * w;
                            if (s == 0) { drum_weight[dst + 0] += w; drum_weight[dst + 1] += w; }
                        }
                    }
                    free(chunk_stems[s]);
                }
            }

            // Normalize and add stems
            for (int s = 0; s < found_stems && s < max_drum_stems; s++) {
                float *out = (float *)malloc((size_t)nf * 2 * sizeof(float));
                for (int i2 = 0; i2 < nf * 2; i2++)
                    out[i2] = (drum_weight[i2] > 1e-8) ? (float)(drum_accum[s][i2] / drum_weight[i2]) : 0.0f;
                add_stem(stems, STAGE3_STEMS[s], out, nf);
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
    if (ctx->s1_bs_roformer) { bsr_free(ctx->s1_bs_roformer); delete ctx->s1_bs_roformer; ctx->s1_bs_roformer = nullptr; }
    if (ctx->s2_mel_band)    { bsr_free(ctx->s2_mel_band); delete ctx->s2_mel_band; ctx->s2_mel_band = nullptr; }
    if (ctx->s3_mdx23c)      { mdx_free(ctx->s3_mdx23c); delete ctx->s3_mdx23c; ctx->s3_mdx23c = nullptr; }
    if (ctx->leap_voc)  { bsr_free(ctx->leap_voc);  delete ctx->leap_voc;  ctx->leap_voc  = nullptr; }
    if (ctx->leap_inst) { bsr_free(ctx->leap_inst); delete ctx->leap_inst; ctx->leap_inst = nullptr; }
    fprintf(stderr, "[SuperSep] Released all models (VRAM freed)\n");
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
