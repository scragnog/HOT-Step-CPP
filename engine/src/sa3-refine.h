#pragma once
// sa3-refine.h: Stable Audio 3 SDEdit-style refiner (ONNX Runtime, TRT/CUDA EP)
//
// Post-processing module: encodes 44.1kHz stereo audio into SAME-L latent
// space, partially re-noises it (strength = init noise level), denoises with
// the SA3 medium DiT (8-step distilled rectified flow, cfg=1 — single forward
// per step), and decodes back to audio. Used to re-render instrumental stems
// with real treble detail (de-fizz).
//
// Five ONNX graphs in one directory (see tools/onnx-export/export_sa3_*.py):
//   sa3-text_encoder.onnx     [1,256]i64 + [1,256]bool -> [1,256,768]   (fp32!)
//   sa3-seconds_embedder.onnx [1]f32                   -> [1,768]
//   sa3-same_encoder.onnx     [1,2,524288]             -> [1,256,128]   (static chunk)
//   sa3-same_decoder.onnx     [1,256,128]              -> [1,2,524288]  (static chunk)
//   sa3-dit.onnx              x[1,256,T] t[1] cross[1,257,768] glob[1,768]
//                             local[1,257,T] pad[1,T]bool -> v[1,256,T] (dynamic T)
//
// Numerical reference: tools/onnx-export/e2e_sa3_ort.py (cosine 0.999999 vs
// PyTorch). Schedule quirk preserved on purpose: t[0] is forced to `strength`
// AFTER the LogSNR warp, so the first step's t appears to jump upward — the
// reference pipeline does exactly this; do not "fix" it.
//
// Tokenization happens in Node (T5Gemma SentencePiece — bpe.h can't parse
// tokenizer.json); the endpoint receives 256 padded token ids + valid count.
//
// Thread safety: none. Caller serialises access (single GPU worker thread).
// Guarded by HOT_STEP_SUPERSEP (shared ORT dependency).
//
// Part of HOT-Step CPP. MIT license.

#ifndef SA3_REFINE_H
#define SA3_REFINE_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ── Constants (match the exported graphs / SA3 medium config) ───────────
// Outside the SUPERSEP guard: the server endpoint references these even in
// stub builds.

#define SA3_SR             44100
#define SA3_DS             4096      // audio samples per latent frame
#define SA3_LAT_CH         256
#define SA3_COND_DIM       768
#define SA3_TOK_LEN        256       // text encoder token slots (padded)
#define SA3_CHUNK_LAT      128       // SAME graphs traced at this chunk size
#define SA3_CHUNK_SAMPLES  (SA3_CHUNK_LAT * SA3_DS)  // 524288
#define SA3_OVERLAP_LAT    32        // tiling overlap in latent frames
#define SA3_HEADROOM_SEC   6.0f      // schedule/padding headroom (generate() default)
#define SA3_ALIGN_SAMPLES  (SA3_DS * 2)  // encoder chunk_size(32)/stride(16) alignment
#define SA3_T_BUCKET       256   // latent-length bucket (~23.8s). The ORT TRT EP
                                 // builds one engine per DiT input shape; bucketing
                                 // caps that at ~16 engines total. padding_mask
                                 // makes the extra padding semantically inert.

#ifdef HOT_STEP_SUPERSEP

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

// ── Context ────────────────────────────────────────────────────────────

struct Sa3Refine {
    Ort::Env      env;
    Ort::Session *text_enc;
    Ort::Session *seconds_emb;
    Ort::Session *same_enc;
    Ort::Session *same_dec;
    Ort::Session *dit;
    std::string   dir;
    bool          using_trt;

    Sa3Refine() : env(ORT_LOGGING_LEVEL_WARNING, "sa3-refine"),
                  text_enc(nullptr), seconds_emb(nullptr), same_enc(nullptr),
                  same_dec(nullptr), dit(nullptr), using_trt(false) {}
    ~Sa3Refine() {
        delete text_enc; delete seconds_emb; delete same_enc;
        delete same_dec; delete dit;
    }
};

// ── Session factory (same EP ladder as vae-ort.h: TRT -> CUDA -> CPU) ───

static inline Ort::Session * sa3_make_session(Sa3Refine * ctx, const char * onnx_path,
                                              bool trt_fp16, int device_id) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(GGML_USE_CUDA)
    std::string trt_cache_dir;
    {
        std::string p = onnx_path;
        auto slash = p.find_last_of("/\\");
        trt_cache_dir = (slash != std::string::npos) ? p.substr(0, slash) : ".";
    }
    {
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.device_id                    = device_id;
        trt_opts.trt_max_partition_iterations = 1000;
        trt_opts.trt_min_subgraph_size        = 1;
        trt_opts.trt_max_workspace_size       = (size_t)2 << 30;
        trt_opts.trt_fp16_enable              = trt_fp16 ? 1 : 0;
        trt_opts.trt_engine_cache_enable      = 1;
        trt_opts.trt_engine_cache_path        = trt_cache_dir.c_str();

        const OrtApi & api = Ort::GetApi();
        OrtStatus * status = api.SessionOptionsAppendExecutionProvider_TensorRT(opts, &trt_opts);
        if (status) {
            std::string msg = api.GetErrorMessage(status);
            api.ReleaseStatus(status);
            fprintf(stderr, "[SA3] TensorRT EP unavailable: %s — trying CUDA EP\n", msg.c_str());
        } else {
            ctx->using_trt = true;
        }
    }
    try {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        cuda_opts.device_id             = device_id;
        cuda_opts.arena_extend_strategy = 1;  // kSameAsRequested
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    } catch (const std::exception & e) {
        fprintf(stderr, "[SA3] CUDA EP failed: %s — falling back to CPU\n", e.what());
    }
#endif

    try {
#ifdef _WIN32
        int wlen = MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, nullptr, 0);
        std::vector<wchar_t> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, onnx_path, -1, wpath.data(), wlen);
        return new Ort::Session(ctx->env, wpath.data(), opts);
#else
        return new Ort::Session(ctx->env, onnx_path, opts);
#endif
    } catch (const std::exception & e) {
        fprintf(stderr, "[SA3] FATAL: session creation failed for %s: %s\n", onnx_path, e.what());
        return nullptr;
    }
}

// Load all five graphs from `dir`. Text encoder + seconds embedder stay fp32
// in TRT (text-enc precedent: layernorm overflows in fp16); the three big
// transformer graphs run fp16.
static inline bool sa3_load(Sa3Refine * ctx, const char * dir, int device_id = 0) {
    if (!ctx || !dir) return false;
    ctx->dir = dir;
    std::string d = dir;
    ctx->text_enc    = sa3_make_session(ctx, (d + "/sa3-text_encoder.onnx").c_str(),     false, device_id);
    ctx->seconds_emb = sa3_make_session(ctx, (d + "/sa3-seconds_embedder.onnx").c_str(), false, device_id);
    // All transformer graphs fp32 under TRT. They carry deliberate fp32 casts
    // (norms, timestep path — logsnr transform amplifies low-precision error
    // ~380x) that trt_fp16_enable would flatten. Measured: blanket TRT fp16 =
    // cosine 0.966, DiT-only fp32 = 0.9953, vs 0.9996 for properly-scoped
    // torch fp16. Scoped fp16 can return via natively-converted fp16 graphs.
    ctx->same_enc    = sa3_make_session(ctx, (d + "/sa3-same_encoder.onnx").c_str(),     false, device_id);
    ctx->same_dec    = sa3_make_session(ctx, (d + "/sa3-same_decoder.onnx").c_str(),     false, device_id);
    ctx->dit         = sa3_make_session(ctx, (d + "/sa3-dit.onnx").c_str(),              false, device_id);
    bool ok = ctx->text_enc && ctx->seconds_emb && ctx->same_enc && ctx->same_dec && ctx->dit;
    fprintf(stderr, "[SA3] Loaded 5 graphs from %s (TRT=%s): %s\n",
            dir, ctx->using_trt ? "yes" : "no", ok ? "OK" : "FAILED");
    return ok;
}

static inline void sa3_free_sessions(Sa3Refine * ctx) {
    if (!ctx) return;
    delete ctx->text_enc;    ctx->text_enc = nullptr;
    delete ctx->seconds_emb; ctx->seconds_emb = nullptr;
    delete ctx->same_enc;    ctx->same_enc = nullptr;
    delete ctx->same_dec;    ctx->same_dec = nullptr;
    delete ctx->dit;         ctx->dit = nullptr;
}

// ── Small run helpers ──────────────────────────────────────────────────

struct Sa3Feed {
    const char * name;
    Ort::Value   value;
};

// Run a session feeding by NAME, tolerating graph-pruned inputs (the exporter
// drops inputs the graph never uses — e.g. the DiT's cross-attention mask).
// Returns the first output.
static inline Ort::Value sa3_run(Ort::Session * sess, std::vector<Sa3Feed> & feeds) {
    Ort::AllocatorWithDefaultOptions alloc;
    size_t n_in = sess->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> name_holders;
    std::vector<const char *> in_names;
    std::vector<Ort::Value>   in_values;
    for (size_t i = 0; i < n_in; i++) {
        auto nm = sess->GetInputNameAllocated(i, alloc);
        for (auto & f : feeds) {
            if (strcmp(nm.get(), f.name) == 0) {
                in_names.push_back(f.name);
                in_values.push_back(std::move(f.value));
                break;
            }
        }
        name_holders.push_back(std::move(nm));
    }
    auto out_name = sess->GetOutputNameAllocated(0, alloc);
    const char * out_names[] = { out_name.get() };
    auto outs = sess->Run(Ort::RunOptions{nullptr},
                          in_names.data(), in_values.data(), in_values.size(),
                          out_names, 1);
    return std::move(outs[0]);
}

static inline Ort::Value sa3_tensor_f32(const float * data, std::vector<int64_t> shape) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return Ort::Value::CreateTensor<float>(mem, const_cast<float *>(data), n,
                                           shape.data(), shape.size());
}

// ── Schedule: LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0) ───
// Mirrors build_schedule + LogSNRShift.shift for the model's default
// sampling dist shift (seq-len invariant since rate=0).

static inline void sa3_build_schedule(int steps, float strength, std::vector<float> & out) {
    out.resize(steps + 1);
    for (int i = 0; i <= steps; i++) {
        float t = strength * (1.0f - (float)i / (float)steps);  // linspace(strength, 0)
        float t_out;
        if (t <= 0.0f)      t_out = 0.0f;
        else if (t >= 1.0f) t_out = 1.0f;
        else {
            float logsnr = 2.0f - t * (2.0f - (-6.2f));
            t_out = 1.0f / (1.0f + expf(logsnr));  // sigmoid(-logsnr)
        }
        out[i] = t_out;
    }
    out[0] = strength;  // reference forces t[0]=sigma_max AFTER the warp
}

// ── Tiled SAME encode/decode (overlap-trim, mirrors encode_audio/decode_audio) ──

static inline void sa3_chunk_starts(int total, int size, int hop, std::vector<int> & starts) {
    starts.clear();
    for (int s = 0; s + size <= total; s += hop) starts.push_back(s);
    if (starts.empty() || starts.back() != total - size) starts.push_back(total - size);
}

// audio: planar stereo [2][S] (S multiple of SA3_DS, >= one chunk after padding
// handled by caller). latents_out: [SA3_LAT_CH][L] channel-first, L = S/SA3_DS.
static inline bool sa3_encode_tiled(Sa3Refine * ctx, const float * audio, int S,
                                    std::vector<float> & latents_out) {
    int L = S / SA3_DS;
    latents_out.assign((size_t)SA3_LAT_CH * L, 0.0f);
    int hop = (SA3_CHUNK_LAT - SA3_OVERLAP_LAT) * SA3_DS;
    std::vector<int> starts;
    sa3_chunk_starts(S, SA3_CHUNK_SAMPLES, hop, starts);
    int half = SA3_OVERLAP_LAT / 2;
    std::vector<float> chunk_buf((size_t)2 * SA3_CHUNK_SAMPLES);
    for (size_t i = 0; i < starts.size(); i++) {
        int s = starts[i];
        // planar [2][S] -> chunk planar [2][CHUNK]
        memcpy(chunk_buf.data(),                     audio + s,     sizeof(float) * SA3_CHUNK_SAMPLES);
        memcpy(chunk_buf.data() + SA3_CHUNK_SAMPLES, audio + S + s, sizeof(float) * SA3_CHUNK_SAMPLES);
        std::vector<Sa3Feed> feeds;
        feeds.push_back({"audio", sa3_tensor_f32(chunk_buf.data(), {1, 2, SA3_CHUNK_SAMPLES})});
        Ort::Value out = sa3_run(ctx->same_enc, feeds);
        const float * lat = out.GetTensorData<float>();  // [1,256,128]
        bool first = (i == 0), last = (i + 1 == starts.size());
        int os = last ? (L - SA3_CHUNK_LAT) : (s / SA3_DS);
        int left  = first ? 0 : half;
        int right = last ? SA3_CHUNK_LAT : SA3_CHUNK_LAT - half;
        for (int c = 0; c < SA3_LAT_CH; c++) {
            memcpy(latents_out.data() + (size_t)c * L + os + left,
                   lat + (size_t)c * SA3_CHUNK_LAT + left,
                   sizeof(float) * (right - left));
        }
    }
    return true;
}

// latents: [SA3_LAT_CH][L] -> audio_out planar [2][L*SA3_DS]
static inline bool sa3_decode_tiled(Sa3Refine * ctx, const float * latents, int L,
                                    std::vector<float> & audio_out) {
    int S = L * SA3_DS;
    audio_out.assign((size_t)2 * S, 0.0f);
    int hop = SA3_CHUNK_LAT - SA3_OVERLAP_LAT;
    std::vector<int> starts;
    sa3_chunk_starts(L, SA3_CHUNK_LAT, hop, starts);
    int half_s = (SA3_OVERLAP_LAT / 2) * SA3_DS;
    std::vector<float> chunk_buf((size_t)SA3_LAT_CH * SA3_CHUNK_LAT);
    for (size_t i = 0; i < starts.size(); i++) {
        int s = starts[i];
        for (int c = 0; c < SA3_LAT_CH; c++) {
            memcpy(chunk_buf.data() + (size_t)c * SA3_CHUNK_LAT,
                   latents + (size_t)c * L + s, sizeof(float) * SA3_CHUNK_LAT);
        }
        std::vector<Sa3Feed> feeds;
        feeds.push_back({"latents", sa3_tensor_f32(chunk_buf.data(), {1, SA3_LAT_CH, SA3_CHUNK_LAT})});
        Ort::Value out = sa3_run(ctx->same_dec, feeds);
        const float * aud = out.GetTensorData<float>();  // [1,2,524288]
        bool first = (i == 0), last = (i + 1 == starts.size());
        int os = (last ? (L - SA3_CHUNK_LAT) : s) * SA3_DS;
        int left  = first ? 0 : half_s;
        int right = last ? SA3_CHUNK_SAMPLES : SA3_CHUNK_SAMPLES - half_s;
        memcpy(audio_out.data() + os + left,     aud + left,                     sizeof(float) * (right - left));
        memcpy(audio_out.data() + S + os + left, aud + SA3_CHUNK_SAMPLES + left, sizeof(float) * (right - left));
    }
    return true;
}

// ── Main refine ────────────────────────────────────────────────────────
//
// audio: planar stereo [2][T44] at 44.1kHz. token_ids: SA3_TOK_LEN padded ids.
// strength: init noise level (0.3 default recipe). steps: 8. pingpong: true
// for production (per-step renoise), false = deterministic Euler (validation).
// zero_noise: replaces all noise draws with zeros (numerical validation vs
// the Python harness — C++ RNG can't reproduce torch's stream).
// out: planar stereo [2][T44] at 44.1kHz, clamped to [-1,1].

static inline bool sa3_refine_run(Sa3Refine * ctx,
                                  const float * audio, int T44,
                                  const int64_t * token_ids, int n_tokens,
                                  float strength, int steps,
                                  bool pingpong, uint64_t seed, bool zero_noise,
                                  std::vector<float> & out) {
    if (!ctx || !ctx->dit || !audio || T44 <= 0 || steps < 1) return false;
    float seconds_total = (float)T44 / SA3_SR;

    // Adapted sample size (mirrors _adapt_sample_size)
    int64_t target = (int64_t)((seconds_total + SA3_HEADROOM_SEC) * SA3_SR);
    target = ((target + SA3_DS - 1) / SA3_DS) * SA3_DS;
    target = ((target + SA3_ALIGN_SAMPLES - 1) / SA3_ALIGN_SAMPLES) * SA3_ALIGN_SAMPLES;
    int S = (int)target;
    if (S < SA3_CHUNK_SAMPLES) S = SA3_CHUNK_SAMPLES;  // static chunk graph minimum
    int L = S / SA3_DS;
    L = ((L + SA3_T_BUCKET - 1) / SA3_T_BUCKET) * SA3_T_BUCKET;  // TRT shape bucketing
    S = L * SA3_DS;

    // Conditioning
    std::vector<int64_t> ids(token_ids, token_ids + SA3_TOK_LEN);
    std::vector<char> tok_mask(SA3_TOK_LEN);
    for (int i = 0; i < SA3_TOK_LEN; i++) tok_mask[i] = (i < n_tokens) ? 1 : 0;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> ids_shape = {1, SA3_TOK_LEN};
    std::vector<Sa3Feed> tfeeds;
    tfeeds.push_back({"input_ids", Ort::Value::CreateTensor<int64_t>(
        mem, ids.data(), ids.size(), ids_shape.data(), ids_shape.size())});
    tfeeds.push_back({"attention_mask", Ort::Value::CreateTensor<bool>(
        mem, reinterpret_cast<bool *>(tok_mask.data()), tok_mask.size(),
        ids_shape.data(), ids_shape.size())});
    Ort::Value text_out = sa3_run(ctx->text_enc, tfeeds);
    const float * text_emb = text_out.GetTensorData<float>();  // [1,256,768]

    float sec_in = seconds_total;
    std::vector<Sa3Feed> sfeeds;
    sfeeds.push_back({"seconds", sa3_tensor_f32(&sec_in, {1})});
    Ort::Value sec_out = sa3_run(ctx->seconds_emb, sfeeds);
    const float * sec_emb = sec_out.GetTensorData<float>();    // [1,768]

    // cross = concat(text[256], seconds[1]) -> [1,257,768]
    std::vector<float> cross((size_t)(SA3_TOK_LEN + 1) * SA3_COND_DIM);
    memcpy(cross.data(), text_emb, sizeof(float) * SA3_TOK_LEN * SA3_COND_DIM);
    memcpy(cross.data() + (size_t)SA3_TOK_LEN * SA3_COND_DIM, sec_emb, sizeof(float) * SA3_COND_DIM);

    // Pad audio to S, encode
    std::vector<float> padded((size_t)2 * S, 0.0f);
    memcpy(padded.data(),     audio,       sizeof(float) * T44);
    memcpy(padded.data() + S, audio + T44, sizeof(float) * T44);
    std::vector<float> x;
    if (!sa3_encode_tiled(ctx, padded.data(), S, x)) return false;

    // Mix with noise: x = init*(1-s) + noise*s
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); i++) {
        float n = zero_noise ? 0.0f : gauss(rng);
        x[i] = x[i] * (1.0f - strength) + n * strength;
    }

    // Schedule + padding mask
    std::vector<float> sigmas;
    sa3_build_schedule(steps, strength, sigmas);
    int eff = (int)ceil((double)(int64_t)(seconds_total * SA3_SR) / SA3_DS);
    int headroom_tokens = (int)(SA3_HEADROOM_SEC * SA3_SR / SA3_DS);
    int valid = eff + headroom_tokens; if (valid > L) valid = L;
    std::vector<char> pad_mask(L);
    for (int i = 0; i < L; i++) pad_mask[i] = (i < valid) ? 1 : 0;
    std::vector<float> local_add((size_t)(SA3_LAT_CH + 1) * L, 0.0f);  // no inpaint

    // Sampler loop (default pingpong, matching rf_denoiser production path)
    std::vector<int64_t> x_shape    = {1, SA3_LAT_CH, L};
    std::vector<int64_t> pad_shape  = {1, L};
    for (int i = 0; i < steps; i++) {
        float t_curr = sigmas[i], t_next = sigmas[i + 1];
        std::vector<Sa3Feed> feeds;
        feeds.push_back({"x", sa3_tensor_f32(x.data(), x_shape)});
        feeds.push_back({"t", sa3_tensor_f32(&t_curr, {1})});
        feeds.push_back({"cross_attn_cond", sa3_tensor_f32(cross.data(), {1, SA3_TOK_LEN + 1, SA3_COND_DIM})});
        feeds.push_back({"global_embed", sa3_tensor_f32(sec_emb, {1, SA3_COND_DIM})});
        feeds.push_back({"local_add_cond", sa3_tensor_f32(local_add.data(), {1, SA3_LAT_CH + 1, L})});
        feeds.push_back({"padding_mask", Ort::Value::CreateTensor<bool>(
            mem, reinterpret_cast<bool *>(pad_mask.data()), pad_mask.size(),
            pad_shape.data(), pad_shape.size())});
        Ort::Value v_out = sa3_run(ctx->dit, feeds);
        const float * v = v_out.GetTensorData<float>();
        if (pingpong) {
            // denoised = x - t*v; x = (1-t_next)*denoised + t_next*randn
            for (size_t k = 0; k < x.size(); k++) {
                float denoised = x[k] - t_curr * v[k];
                float n = zero_noise ? 0.0f : gauss(rng);
                x[k] = (1.0f - t_next) * denoised + t_next * n;
            }
        } else {
            float dt = t_next - t_curr;
            for (size_t k = 0; k < x.size(); k++) x[k] += dt * v[k];
        }
        fprintf(stderr, "[SA3] step %d/%d t=%.4f->%.4f\n", i + 1, steps, t_curr, t_next);
    }

    // Decode, zero padded region, trim to input length, clamp
    std::vector<float> decoded;
    if (!sa3_decode_tiled(ctx, x.data(), L, decoded)) return false;
    int S_dec = L * SA3_DS;
    int valid_samples = valid * SA3_DS;
    out.assign((size_t)2 * T44, 0.0f);
    for (int chn = 0; chn < 2; chn++) {
        const float * src = decoded.data() + (size_t)chn * S_dec;
        float * dst = out.data() + (size_t)chn * T44;
        int n = T44 < valid_samples ? T44 : valid_samples;
        for (int k = 0; k < n; k++) {
            float vsm = src[k];
            dst[k] = vsm < -1.0f ? -1.0f : (vsm > 1.0f ? 1.0f : vsm);
        }
    }
    return true;
}

#else  // !HOT_STEP_SUPERSEP — stubs

struct Sa3Refine {};
static inline bool sa3_load(Sa3Refine *, const char *, int = 0) { return false; }
static inline void sa3_free_sessions(Sa3Refine *) {}
static inline bool sa3_refine_run(Sa3Refine *, const float *, int, const int64_t *, int,
                                  float, int, bool, uint64_t, bool,
                                  std::vector<float> &) { return false; }

#endif // HOT_STEP_SUPERSEP
#endif // SA3_REFINE_H
