#pragma once
// sa3-refine.h: Stable Audio 3 SDEdit-style refiner (GGML)
//
// Post-processing module: encodes 44.1kHz stereo audio into SAME-L latent
// space, partially re-noises it (strength = init noise level), denoises with
// the SA3 medium DiT (8-step distilled rectified flow, cfg=1 — single forward
// per step), and decodes back to audio. Used to re-render instrumental stems
// with real treble detail (de-fizz).
//
// The orchestration (schedule, tiled encode/decode, sampler, conditioning
// assembly) is backend-agnostic: it drives an Sa3Backend interface with five
// operations, implemented on GGML:
//
//   GGML (Sa3GgmlRefine) — four GGUFs in one directory (parity-gated vs the
//   ONNX-export goldens, see tools/sa3-ggml-test.cpp):
//     sa3-text-enc-BF16.gguf  (sa3-t5gemma-enc.h; also seconds embedder from
//     sa3-dit-BF16.gguf)      the DiT GGUF's conditioner tensors
//     sa3-same-enc-F16.gguf   (sa3-same-ggml.h, one 128-latent chunk)
//     sa3-same-dec-F16.gguf
//     sa3-dit-BF16.gguf       (sa3-dit-ggml.h, graph rebuilt per T)
//
// All backend buffers are torch-contiguous / channel-first, matching the
// exported ONNX graphs: text [256,768] token-major, latents [256,L], audio
// planar [2,S], cross [257,768] token-major, local [257,T].
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
//
// Part of HOT-Step CPP. MIT license.

#ifndef SA3_REFINE_H
#define SA3_REFINE_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// HOT-Step Lua plugin system — the SAME registry, plugins and dispatch entry
// points the ACE-Step sampler and the MM3 flow DiT use. Nothing in
// engine/plugins/ knows SA3 exists; the plugin layer never had a backend
// dependency (every lua_call_* takes raw float* + counts + a param map).
#include "guidance/apg-core.h"
#include "guidance/guidance-interface.h"
#include "lua-plugin-registry.h"
#include "sampler-schedule.h"
#include "synth-finite.h"
#include "solvers/solver-interface.h"

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
#define SA3_T_BUCKET       256   // latent-length bucket (~23.8s). Introduced to cap
                                 // TensorRT engine builds per input shape; kept so
                                 // the padded length (and so the output) is
                                 // unchanged. padding_mask makes the padding inert.

// ── Backend interface ──────────────────────────────────────────────────
//
// Five compute ops behind which the GGML modules sit.
// Layouts (all f32, channel-first / torch-contiguous, batch 1 implied):
//   text_encode        ids[SA3_TOK_LEN] i64 + valid count -> out[256*768]
//                      token-major (out[s*768 + h]), learned-padding rows
//                      already substituted.
//   seconds_embed      seconds -> out[768]
//   same_encode_chunk  audio planar [2][524288] -> latents [256][128]
//   same_decode_chunk  latents [256][128] -> audio planar [2][524288]
//   dit_forward        x[256][T], t, cross[257][768] token-major,
//                      glob[768], local[257][T], pad_mask[T] (0/1 bytes)
//                      -> v[256][T]
// Each op returns false on failure (aborts the refine).

struct Sa3Backend {
    virtual ~Sa3Backend() = default;
    virtual const char * name() const = 0;
    virtual bool text_encode(const int64_t * ids, int n_tokens, float * out) = 0;
    virtual bool seconds_embed(float seconds, float * out) = 0;
    virtual bool same_encode_chunk(const float * audio, float * latents) = 0;
    virtual bool same_decode_chunk(const float * latents, float * audio) = 0;
    virtual bool dit_forward(const float * x, float t, const float * cross,
                             const float * glob, const float * local,
                             const char * pad_mask, int T, float * v) = 0;
};

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
static inline bool sa3_encode_tiled(Sa3Backend * be, const float * audio, int S,
                                    std::vector<float> & latents_out) {
    int L = S / SA3_DS;
    latents_out.assign((size_t)SA3_LAT_CH * L, 0.0f);
    int hop = (SA3_CHUNK_LAT - SA3_OVERLAP_LAT) * SA3_DS;
    std::vector<int> starts;
    sa3_chunk_starts(S, SA3_CHUNK_SAMPLES, hop, starts);
    int half = SA3_OVERLAP_LAT / 2;
    std::vector<float> chunk_buf((size_t)2 * SA3_CHUNK_SAMPLES);
    std::vector<float> lat_buf((size_t)SA3_LAT_CH * SA3_CHUNK_LAT);
    for (size_t i = 0; i < starts.size(); i++) {
        int s = starts[i];
        // planar [2][S] -> chunk planar [2][CHUNK]
        memcpy(chunk_buf.data(),                     audio + s,     sizeof(float) * SA3_CHUNK_SAMPLES);
        memcpy(chunk_buf.data() + SA3_CHUNK_SAMPLES, audio + S + s, sizeof(float) * SA3_CHUNK_SAMPLES);
        if (!be->same_encode_chunk(chunk_buf.data(), lat_buf.data())) return false;
        const float * lat = lat_buf.data();  // [256,128]
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
// n_valid_samples >= 0: decoder chunks whose kept output range starts at or beyond n_valid_samples
// are not computed (the input is padded to a latent bucket plus headroom, and the caller trims that
// tail anyway). The result for [0, n_valid_samples) is bit-identical. SA3_DEC_SKIP=0 turns it off.
static inline bool sa3_decode_tiled(Sa3Backend * be, const float * latents, int L,
                                    std::vector<float> & audio_out, int n_valid_samples = -1,
                                    std::string * finite_error = nullptr) {
    static const bool skip_pad = getenv("SA3_DEC_SKIP") == NULL || atoi(getenv("SA3_DEC_SKIP")) != 0;
    if (L <= 0) return false;
    int S = L * SA3_DS;
    audio_out.assign((size_t)2 * S, 0.0f);
    // The postprocess decode callback (decode_fn lambda feeding lua_call_postprocess,
    // ~line 904) lets a plugin (e.g. md_audio_tiled) pick a tile size T_lat smaller
    // than SA3_CHUNK_LAT. The SAME decoder graph (both ONNX and GGML backends) is
    // traced/called at a fixed SA3_CHUNK_LAT input length, so short latents can't go
    // through the overlap-tiled loop below (it would index negative into `latents`
    // and corrupt the heap). Zero-pad up to one full chunk, decode once, and keep
    // only the first L*SA3_DS samples.
    if (L < SA3_CHUNK_LAT) {
        std::vector<float> pad_lat((size_t)SA3_LAT_CH * SA3_CHUNK_LAT, 0.0f);
        for (int c = 0; c < SA3_LAT_CH; c++) {
            memcpy(pad_lat.data() + (size_t)c * SA3_CHUNK_LAT,
                   latents + (size_t)c * L, sizeof(float) * L);
        }
        std::vector<float> aud_buf((size_t)2 * SA3_CHUNK_SAMPLES);
        if (!be->same_decode_chunk(pad_lat.data(), aud_buf.data())) return false;
        memcpy(audio_out.data(),     aud_buf.data(),                     sizeof(float) * S);
        memcpy(audio_out.data() + S, aud_buf.data() + SA3_CHUNK_SAMPLES, sizeof(float) * S);
        return synth_finite(audio_out.data(), audio_out.size(), "SA3 SAME decode", finite_error);
    }
    int hop = SA3_CHUNK_LAT - SA3_OVERLAP_LAT;
    std::vector<int> starts;
    sa3_chunk_starts(L, SA3_CHUNK_LAT, hop, starts);
    int half_s = (SA3_OVERLAP_LAT / 2) * SA3_DS;
    std::vector<float> chunk_buf((size_t)SA3_LAT_CH * SA3_CHUNK_LAT);
    std::vector<float> aud_buf((size_t)2 * SA3_CHUNK_SAMPLES);
    for (size_t i = 0; i < starts.size(); i++) {
        int s = starts[i];
        bool first = (i == 0), last = (i + 1 == starts.size());
        int os = (last ? (L - SA3_CHUNK_LAT) : s) * SA3_DS;
        int left  = first ? 0 : half_s;
        int right = last ? SA3_CHUNK_SAMPLES : SA3_CHUNK_SAMPLES - half_s;
        if (skip_pad && n_valid_samples >= 0 && os + left >= n_valid_samples) continue;
        for (int c = 0; c < SA3_LAT_CH; c++) {
            memcpy(chunk_buf.data() + (size_t)c * SA3_CHUNK_LAT,
                   latents + (size_t)c * L + s, sizeof(float) * SA3_CHUNK_LAT);
        }
        if (!be->same_decode_chunk(chunk_buf.data(), aud_buf.data())) return false;
        const float * aud = aud_buf.data();  // [2,524288]
        memcpy(audio_out.data() + os + left,     aud + left,                     sizeof(float) * (right - left));
        memcpy(audio_out.data() + S + os + left, aud + SA3_CHUNK_SAMPLES + left, sizeof(float) * (right - left));
    }
    return synth_finite(audio_out.data(), audio_out.size(), "SA3 SAME decode", finite_error);
}

// ── SA3 ⇄ Lua plugin bridge ────────────────────────────────────────────
//
// Routes the HOT-Step solver/scheduler/guidance plugins through the SA3
// refine loop. All-empty (the default, and `sp == nullptr`) runs the original
// pingpong/euler dispatch expression for expression — the refine stays
// bit-identical to before this hookup existed.
//
// ── CONVENTION 1: TIME ALREADY RUNS THE ACE WAY ────────────────────────
//
// Unlike MM3 (whose sigma ASCENDS and needed a sign flip — see
// minimax/mm3-plugins.h), SA3's schedule already matches ACE's:
//
//   ACE  t descends 1 -> 0. Solver contract: `xt -= vt * (t_curr - t_prev)`.
//   SA3  sigmas descend strength -> 0, and the Euler branch below is
//        `dt = t_next - t_curr` (negative), `x += dt * v`
//        == `x -= (t_curr - t_next) * v`.
//
// Character for character the same update. So no sign flip, and no schedule
// re-parameterisation: SA3's `sigmas` array IS an ACE schedule that happens to
// start at `strength` instead of 1.0. That one difference is what the
// scheduler-override rescale below exists to reconcile.
//
// ── CONVENTION 2: THE LATENTS ARE TRANSPOSED ───────────────────────────
//
//   ACE  memory is TIME-MAJOR [T][Oc]  (index t*Oc + c) — see hot-step-sampler.h.
//   SA3  latents are CHANNEL-MAJOR [SA3_LAT_CH][L] (index c*L + j) — that is
//        what sa3_encode_tiled writes and what dit_forward consumes.
//
// This matters for GUIDANCE and only for guidance. apg_project() normalises and
// projects PER CHANNEL OVER TIME and indexes `[t*Oc + c]` to do it
// (guidance/apg-core.h). Hand it a channel-major buffer and it reinterprets the
// block as [L][256]: the "channels" it normalises are scrambled mixtures, so
// every cond-only enhancement plugin operates on nonsense. The bridge therefore
// transposes into the ACE view before any guidance plugin sees a buffer, and
// back afterwards.
//
// The SOLVER path needs no transpose: lua_call_solver_step takes only a flat
// element count `n` (solver-interface.h), and lua_call_solver_loop uses T/Oc
// solely to publish `n_per = T*Oc` — neither hands a plugin the shape it would
// need to index across channels. Solvers are elementwise by contract.
//
// ── GUIDANCE ON A MODEL WITH NO UNCOND ─────────────────────────────────
//
// SA3 was trained at cfg=1: there is no unconditional branch, and a zero-cond
// forward yields noise rather than a valid uncond signal. So the cond velocity
// is passed as BOTH pred_cond and pred_uncond. APG then sees diff = 0 and is a
// pass-through, while cond-only plugins still apply their enhancements. This is
// deliberate and it means APG-family guidance is a NO-OP on SA3 — the dropdown
// is only meaningful for plugins that do cond-side work.

struct Sa3PluginParams {
    std::string solver;         // "" = the original pingpong/euler dispatch
    std::string scheduler;      // "" = sa3_build_schedule
    std::string guidance_mode;  // "" = no guidance
    float       guidance_scale = 1.0f;
    float       apg_norm_threshold = 2.5f;
    std::unordered_map<std::string, std::string> plugin_params;
    // Postprocess plugin: name of the Lua postprocess plugin to use for the
    // latent->audio decode. Mirrors ACE's request.h postprocess_plugin field
    // and ops_vae_decode_postprocess (pipeline-synth-ops.cpp). Orthogonal to
    // solver/scheduler/guidance -- it only replaces the SAME decoder call,
    // not the denoise loop -- so it does NOT gate on those via any().
    std::string postprocess_plugin;  // "" = built-in sa3_decode_tiled
    // SA3 is a distilled model sampled with a pingpong renoise scheme; a Lua
    // solver's ODE-style update skips that renoise, which on this model comes
    // out ~2x hot and crackly regardless of plugin (final rms ~1.80 vs native
    // ~0.89). true (default) wraps every Lua solver step in the same
    // pingpong renoise the native path uses, driven by the solver's effective
    // velocity over the step. false = old behaviour (solver update only, no
    // renoise) for A/B. No effect when no solver is selected, or on the
    // owns_loop path (see sa3-refine.h NOTE at the owns_loop branch).
    bool sa3_lua_renoise = true;
    // Seed breeding (/sa3-synth only): slerp the initial noise of `seed`
    // toward the noise of seed_b. seed_b < 0 or seed_blend == 0 = off.
    int64_t seed_b     = -1;
    float   seed_blend = 0.0f;

    bool any() const {
        return !solver.empty() || !scheduler.empty() || !guidance_mode.empty()
            || !postprocess_plugin.empty()
            || (seed_b >= 0 && seed_blend > 0.0f);  // breeding needs sp on the native path too
    }
};

// cm: channel-major [Oc][T], index c*T + t   (SA3)
// tm: time-major    [T][Oc], index t*Oc + c  (ACE)
static inline void sa3_to_ace_view(const float * cm, float * tm, int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        const float * src = cm + (size_t) c * T;
        for (int t = 0; t < T; t++) tm[(size_t) t * Oc + c] = src[t];
    }
}

static inline void sa3_from_ace_view(const float * tm, float * cm, int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        float * dst = cm + (size_t) c * T;
        for (int t = 0; t < T; t++) dst[t] = tm[(size_t) t * Oc + c];
    }
}

// ── Main refine (backend-agnostic orchestration) ───────────────────────
//
// audio: planar stereo [2][T44] at 44.1kHz. token_ids: SA3_TOK_LEN padded ids.
// strength: init noise level (0.3 default recipe). steps: 8. pingpong: true
// for production (per-step renoise), false = deterministic Euler (validation).
// zero_noise: replaces all noise draws with zeros (numerical validation vs
// the Python harness — C++ RNG can't reproduce torch's stream).
// sp: optional Lua solver/scheduler/guidance selection (nullptr = legacy path).
//     NOTE: SA3 is a distilled model tuned for the pingpong renoise scheme, not
//     bare ODE integration -- a Lua solver step with no renoise runs ~2x hot on
//     this model regardless of plugin. So (per-step, non-owns_loop) Lua solvers
//     are, by default, wrapped in the same pingpong renoise the native path
//     uses: `sa3_lua_renoise` (default true) computes the solver's effective
//     velocity over the step and pingpongs from it. Set it false to fall back
//     to the solver's own update with no renoise. owns_loop solvers always run
//     without renoise (they drive the whole loop themselves).
// out: planar stereo [2][T44] at 44.1kHz, NOT clamped.
//
// The decode used to saturate to [-1,1] here. A separated instrumental stem at
// any ordinary crest factor peaks above 1.0, so that flat-topped the refined
// bed at the earliest possible point, before the caller's envelope/RMS/blend
// maths and before the mix — where a vocal summed on top hides the damage from
// anything measuring the finished track. The caller decides: hot-step-server's
// sa3_sat saturates for integer responses, audio_encode_wav clamps s16/s24
// regardless, and an f32 response carries the peaks through to whichever stage
// owns the ceiling. See docs/plans/2026-08-28-post-processing-gain-staging.md.

static inline bool sa3_refine_run_backend(Sa3Backend * be,
                                          const float * audio, int T44,
                                          const int64_t * token_ids, int n_tokens,
                                          float strength, int steps,
                                          bool pingpong, uint64_t seed, bool zero_noise,
                                          std::vector<float> & out,
                                          const Sa3PluginParams * sp = nullptr,
                                          bool noise_only = false,
                                          const char * dump_latent_path = nullptr,
                                          const std::function<bool()> & cancelled = nullptr,
                                          std::string * finite_error = nullptr) {
    if (finite_error) finite_error->clear();
    // noise_only (HOT-Step /sa3-synth): text-to-audio from pure gaussian noise.
    // `audio` is ignored (may be nullptr); T44 gives the requested output
    // length; strength is forced to 1.0 so the schedule runs the full
    // sigma_max -> 0 range. Everything after the latent init -- conditioning,
    // plugin routing, schedule, denoise loop, decode, trim -- is the refine
    // path unchanged. SDEdit (the default) is the secondary use of this model;
    // this is what it was trained for.
    if (noise_only) {
        strength = 1.0f;
    }
    if (!be || (!noise_only && !audio) || T44 <= 0 || steps < 1) return false;
    float seconds_total = (float)T44 / SA3_SR;

    // Adapted sample size (mirrors _adapt_sample_size)
    int64_t target = (int64_t)((seconds_total + SA3_HEADROOM_SEC) * SA3_SR);
    target = ((target + SA3_DS - 1) / SA3_DS) * SA3_DS;
    target = ((target + SA3_ALIGN_SAMPLES - 1) / SA3_ALIGN_SAMPLES) * SA3_ALIGN_SAMPLES;
    int S = (int)target;
    if (S < SA3_CHUNK_SAMPLES) S = SA3_CHUNK_SAMPLES;  // static chunk graph minimum
    int L = S / SA3_DS;
    L = ((L + SA3_T_BUCKET - 1) / SA3_T_BUCKET) * SA3_T_BUCKET;  // shape bucketing, see SA3_T_BUCKET
    S = L * SA3_DS;

    // Conditioning
    std::vector<float> text_emb((size_t)SA3_TOK_LEN * SA3_COND_DIM);  // [256,768]
    if (!be->text_encode(token_ids, n_tokens, text_emb.data())) return false;

    std::vector<float> sec_emb(SA3_COND_DIM);  // [768]
    if (!be->seconds_embed(seconds_total, sec_emb.data())) return false;

    // cross = concat(text[256], seconds[1]) -> [257,768]
    std::vector<float> cross((size_t)(SA3_TOK_LEN + 1) * SA3_COND_DIM);
    memcpy(cross.data(), text_emb.data(), sizeof(float) * SA3_TOK_LEN * SA3_COND_DIM);
    memcpy(cross.data() + (size_t)SA3_TOK_LEN * SA3_COND_DIM, sec_emb.data(), sizeof(float) * SA3_COND_DIM);

    std::vector<float> x;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    // Seed breeding (noise_only): seed B gets its own stream, advanced in
    // lockstep with seed A. The initial latent is slerped below, and every
    // pingpong re-noise draw is mixed too (renoise_draw), otherwise the
    // re-noise from seed A dominates and the render stays at A.
    const float breed_t = sp ? sp->seed_blend : 0.0f;
    const bool  breeding = noise_only && !zero_noise && sp && sp->seed_b >= 0 && breed_t > 0.0f;
    std::mt19937_64 rng_b((uint64_t) (sp && sp->seed_b >= 0 ? sp->seed_b : 0));
    std::normal_distribution<float> gauss_b(0.0f, 1.0f);
    // Independent gaussians are ~orthogonal, so slerp reduces to cos/sin weights.
    const float rn_wa = breed_t >= 1.0f ? 0.0f : cosf(breed_t * 1.5707963f);
    const float rn_wb = breed_t >= 1.0f ? 1.0f : sinf(breed_t * 1.5707963f);
    auto renoise_draw = [&]() -> float {
        if (zero_noise) return 0.0f;
        if (!breeding) return gauss(rng);
        const float a = gauss(rng);
        const float b = gauss_b(rng_b);
        return breed_t >= 1.0f ? b : rn_wa * a + rn_wb * b;
    };
    if (noise_only) {
        // Pure noise init. Latent is SA3 channel-major [Oc][L] -- same layout
        // sa3_encode_tiled produces, so the loop below is oblivious to which
        // path filled it.
        x.resize((size_t) SA3_LAT_CH * L);
        for (auto & xi : x) xi = zero_noise ? 0.0f : gauss(rng);
        const float bt = breed_t;
        if (breeding) {
            // Same RNG, seed B; whole latent treated as one vector.
            const size_t n = x.size();
            std::vector<float> nb(n);
            for (auto & v : nb) v = gauss_b(rng_b);
            fprintf(stderr, "[SA3] seed breeding: A=%llu B=%lld blend=%.3f\n",
                    (unsigned long long) seed, (long long) sp->seed_b, bt);
            if (bt >= 1.0f) {
                x = nb;
            } else {
                double dot = 0.0, na = 0.0, nbb = 0.0;
                for (size_t i = 0; i < n; i++) {
                    dot += (double) x[i] * nb[i];
                    na  += (double) x[i] * x[i];
                    nbb += (double) nb[i] * nb[i];
                }
                const double la = sqrt(na), lb = sqrt(nbb);
                double c = (la > 0.0 && lb > 0.0) ? dot / (la * lb) : 1.0;
                c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
                const double om     = acos(c);
                const bool   nearly = sin(om) < 1e-4;
                const double wa = nearly ? 1.0 - bt : sin((1.0 - bt) * om) / sin(om);
                const double wb = nearly ? (double) bt : sin(bt * om) / sin(om);
                double nl = 0.0;
                for (size_t i = 0; i < n; i++) {
                    x[i] = (float) (wa * x[i] + wb * nb[i]);
                    nl += (double) x[i] * x[i];
                }
                if (nearly && nl > 0.0) {
                    const float k = (float) (((1.0 - bt) * la + bt * lb) / sqrt(nl));
                    for (auto & v : x) v *= k;
                }
            }
        }
    } else {
        // Pad audio to S, encode
        std::vector<float> padded((size_t)2 * S, 0.0f);
        memcpy(padded.data(),     audio,       sizeof(float) * T44);
        memcpy(padded.data() + S, audio + T44, sizeof(float) * T44);
        if (!sa3_encode_tiled(be, padded.data(), S, x)) return false;

        // Mix with noise: x = init*(1-s) + noise*s
        for (size_t i = 0; i < x.size(); i++) {
            float n = zero_noise ? 0.0f : gauss(rng);
            x[i] = x[i] * (1.0f - strength) + n * strength;
        }
    }

    // ── Plugin resolution ──────────────────────────────────────────────
    // Resolved once, before the loop. An unknown name is a warning plus a fall
    // back to native for that slot — never a hard failure. A user who typo'd a
    // solver should still get audio back.
    const int   Oc    = SA3_LAT_CH;   // ACE-view channel count
    const int   T     = L;            // ACE-view time frames
    const int   n_lat = Oc * T;

    // Model context for the Lua plugins: SA3 is 44.1 kHz with a 256-channel
    // latent at SA3_SR/SA3_DS frames per second, none of which match ACE's
    // defaults. Plugins that don't read model_context are unaffected.
    LuaModelContext sa3_lua_ctx;
    sa3_lua_ctx.native_sr       = SA3_SR;
    sa3_lua_ctx.latent_fps      = SA3_SR / SA3_DS;
    sa3_lua_ctx.latent_channels = Oc;
    sa3_lua_ctx.model_id        = "sa3";
    // Measured 2026-09-03: native pingpong, 24 steps, 180 s, seed 35602196465681.
    // final latent rms=0.6854, per-channel [0.4645 .. 1.1730]. ACE's is ~2.0.
    sa3_lua_ctx.latent_rms      = 0.685f;

    LuaPlugin * solver_plugin    = nullptr;
    LuaPlugin * guidance_plugin  = nullptr;
    bool        use_lua_solver    = false;
    bool        use_lua_guidance  = false;
    bool        use_lua_scheduler = false;
    bool        guidance_native_apg = false;

    if (sp && sp->any()) {
        auto & reg = PluginRegistry::instance();

        if (!sp->solver.empty()) {
            solver_plugin = reg.solver_lookup(sp->solver.c_str());
            if (!solver_plugin) {
                fprintf(stderr, "[SA3] WARNING: unknown solver '%s', using pingpong/euler\n", sp->solver.c_str());
            } else if (solver_plugin->owns_loop && pingpong) {
                // Mirrors mm3-plugins.h ~285-293: a full-loop solver drives its
                // own iteration and would bypass the per-step re-noise the
                // distilled pingpong model needs. Native pingpong takes over
                // for this slot instead of a hard failure.
                fprintf(stderr,
                        "[SA3] NOTE: owns_loop solver '%s' not supported on SA3 (distilled pingpong model needs "
                        "per-step re-noise) -- using native pingpong\n",
                        solver_plugin->name.c_str());
                solver_plugin = nullptr;
            } else {
                use_lua_solver = true;
            }
        }

        if (!sp->scheduler.empty()) {
            use_lua_scheduler = (reg.scheduler_lookup(sp->scheduler.c_str()) != nullptr);
            if (!use_lua_scheduler) {
                fprintf(stderr, "[SA3] WARNING: unknown scheduler '%s', using sa3_build_schedule\n",
                        sp->scheduler.c_str());
            }
        }

        if (!sp->guidance_mode.empty() && sp->guidance_scale > 1.0f) {
            // Guidance named exactly "apg" takes the native C++ path, mirroring
            // hot-step-sampler.h's `use_apg_native` and mm3-plugins.h. Editing
            // apg.lua's body has never done anything on ACE and must not start
            // doing something here.
            if (sp->guidance_mode == "apg") {
                guidance_native_apg = true;
                use_lua_guidance    = true;
            } else {
                guidance_plugin = reg.guidance_lookup(sp->guidance_mode.c_str());
                if (!guidance_plugin) {
                    fprintf(stderr, "[SA3] WARNING: unknown guidance '%s', skipping guidance\n",
                            sp->guidance_mode.c_str());
                } else {
                    use_lua_guidance = true;
                }
            }
        }

        if (use_lua_solver || use_lua_scheduler || use_lua_guidance) {
            fprintf(stderr, "[SA3] Plugins: solver=%s scheduler=%s guidance=%s\n",
                    solver_plugin ? solver_plugin->name.c_str() : "(pingpong/euler)",
                    use_lua_scheduler ? sp->scheduler.c_str() : "(sa3 logsnr)",
                    guidance_native_apg ? "apg (native)"
                                        : (guidance_plugin ? guidance_plugin->name.c_str() : "(none)"));
            if (use_lua_solver && pingpong) {
                // owns_loop is already filtered out above when pingpong is on
                // (native pingpong takes over instead). The wrapper applies to
                // per-step Lua solvers that do NOT declare themselves
                // stochastic -- those handle their own noise injection and
                // must not also be re-noised by this wrapper (stacking the
                // two diverged md_pingpong_simple to rms 7000+, observed
                // 2026-09-13).
                if (solver_plugin->stochastic) {
                    fprintf(stderr, "[SA3] Lua solver re-noise wrapper: OFF (solver '%s' declares "
                                    "stochastic; it handles its own noise. Observed 2026-09-13: wrapper + "
                                    "md_pingpong_simple diverged to rms 7000+)\n", solver_plugin->name.c_str());
                } else if (sp->sa3_lua_renoise) {
                    fprintf(stderr, "[SA3] Lua solver re-noise wrapper: ON (pingpong after each step)\n");
                } else {
                    fprintf(stderr, "[SA3] Lua solver re-noise wrapper: OFF — solver '%s' update only, "
                                    "no renoise.\n", solver_plugin->name.c_str());
                }
            }
            if (use_lua_guidance) {
                fprintf(stderr, "[SA3] NOTE: SA3 has no CFG uncond (trained at cfg=1); the cond velocity is "
                                "passed as both preds, so APG-family guidance is a pass-through.\n");
            }
            if (use_lua_guidance && guidance_plugin && guidance_plugin->has_post_step) {
                fprintf(stderr, "[SA3] Guidance '%s' post_step: enabled (cond-only evals)\n",
                        guidance_plugin->name.c_str());
            }
            if (solver_plugin && solver_plugin->needs_model) {
                fprintf(stderr, "[SA3] NOTE: '%s' is a multi-evaluation solver (%d NFE) — expect the refine "
                                "to scale roughly with NFE.\n", solver_plugin->name.c_str(), solver_plugin->nfe);
            }
        }
    }

    // ── Schedule ───────────────────────────────────────────────────────
    std::vector<float> sigmas;
    sa3_build_schedule(steps, strength, sigmas);
    if (use_lua_scheduler) {
        // Lua schedulers are written for full denoising and emit sigma_max = 1.0.
        // SA3 refine is SDEdit: it starts at `strength` (0.3 by default), not 1.0.
        // Rescale the returned curve linearly onto [strength -> 0] so the
        // scheduler's SPACING (the thing the user picked it for) is preserved
        // while SA3's partial-noise starting point is respected.
        //
        // sampler_build_scheduler_override reads g_hotstep_params.scheduler —
        // the caller sets and restores it around this call (hot-step-server.cpp).
        std::vector<float> ov;
        sampler_build_scheduler_override(ov, steps, sigmas.data(), sa3_lua_ctx);
        if (!ov.empty()) {
            const float sched_start = ov[0];
            if (sched_start > 1e-6f) {
                const float k = strength / sched_start;
                for (auto & s : ov) s *= k;
            }
            ov[0] = strength;      // pin the start exactly, no float drift
            ov.push_back(0.0f);    // the override drops the x0 endpoint; the loop indexes sigmas[i+1]
            sigmas = ov;
            if (noise_only) {
                fprintf(stderr, "[SA3] Scheduler '%s': full range (synth, sigma_max 1.0) "
                                "model_context model_id=%s native_sr=%d latent_rms=%.3f\n",
                        sp->scheduler.c_str(), sa3_lua_ctx.model_id, sa3_lua_ctx.native_sr,
                        (double) sa3_lua_ctx.latent_rms);
            } else {
                fprintf(stderr, "[SA3] Scheduler '%s': rescaled sigma_max %.4f -> strength %.4f "
                                "model_context model_id=%s native_sr=%d latent_rms=%.3f\n",
                        sp->scheduler.c_str(), (double) sched_start, (double) strength,
                        sa3_lua_ctx.model_id, sa3_lua_ctx.native_sr, (double) sa3_lua_ctx.latent_rms);
            }
        }
    }

    int eff = (int)ceil((double)(int64_t)(seconds_total * SA3_SR) / SA3_DS);
    int headroom_tokens = (int)(SA3_HEADROOM_SEC * SA3_SR / SA3_DS);
    int valid = eff + headroom_tokens; if (valid > L) valid = L;
    std::vector<char> pad_mask(L);
    for (int i = 0; i < L; i++) pad_mask[i] = (i < valid) ? 1 : 0;
    std::vector<float> local_add((size_t)(SA3_LAT_CH + 1) * L, 0.0f);  // no inpaint

    // Sampler loop (default pingpong, matching rf_denoiser production path)
    std::vector<float> v((size_t)n_lat);   // velocity, SA3 channel-major [Oc][T]

    // Guidance/solver scratch, ACE time-major view. tm_v is shared by both
    // roles -- guidance leaves the guided velocity there (see
    // sa3_apply_guidance below), which is exactly the value a Lua solver's
    // vt_readonly/vt_buf need to start from -- they are never live at once
    // within a step. Only allocated when the corresponding lua path runs.
    std::vector<float> tm_c, tm_x, tm_v;
    APGMomentumBuffer  apg_mbuf(-0.75f);
    APGWorkspace       apg_ws;
    GuidanceCtx        g_ctx = { 0, steps, 0.0f, 0.0f };
    if (use_lua_guidance) {
        tm_c.assign((size_t)n_lat, 0.0f);
        tm_v.assign((size_t)n_lat, 0.0f);
        apg_ws.resize(n_lat);
        if (guidance_plugin && guidance_plugin->has_post_step) {
            // post_step needs tm_x too (xt in ACE view) even when there is no
            // Lua solver -- the block below only allocates it for use_lua_solver.
            tm_x.assign((size_t)n_lat, 0.0f);
        }
    }
    // Persistent scratch for the Lua solver path: SA3 keeps latents/velocity
    // channel-major [Oc][T] natively, but solver plugins (step() and
    // owns_loop's sample()) are written against ACE's time-major [T][Oc]
    // view. Mirrors mm3-plugins.h's tm_x/tm_v. lua_model_native_x is the
    // model callback's own scratch -- see sa3_model_fn_ace below. Also needed
    // by post_step (sa3_post_step_eval calls sa3_model_fn_ace too) even when
    // there is no Lua solver, so it is sized off use_lua_solver OR post_step.
    const bool sa3_wants_post_step = use_lua_guidance && guidance_plugin && guidance_plugin->has_post_step;
    std::vector<float> lua_model_native_x;
    if (use_lua_solver || sa3_wants_post_step) {
        lua_model_native_x.assign((size_t)n_lat, 0.0f);
    }
    if (use_lua_solver) {
        tm_x.assign((size_t)n_lat, 0.0f);
        tm_v.assign((size_t)n_lat, 0.0f);
        fprintf(stderr, "[SA3] Lua solver '%s': latents presented time-major (ACE view) Oc=%d L=%d\n",
                solver_plugin->name.c_str(), Oc, T);
    }

    // Re-noise wrapper for the per-step Lua solver path (see Sa3PluginParams::
    // sa3_lua_renoise above). x_prev is a native-layout (channel-major, same
    // shape as `x`) snapshot taken right before the solver's step() call;
    // after the solver updates x, x_prev/x/the interval give the solver's
    // effective velocity over the step, which is then re-noised exactly like
    // the native pingpong branch. Does not apply to owns_loop (filtered out
    // above when pingpong is on) or to solvers that declare stochastic=true --
    // those handle their own noise injection and must not also be re-noised
    // by this wrapper (stacking the two diverged md_pingpong_simple to rms
    // 7000+, observed 2026-09-13). sa3_lua_ctx.engine_renoise mirrors this so
    // stochastic solvers can see it is false and inject their own noise.
    const bool lua_renoise_active = use_lua_solver && pingpong
        && !solver_plugin->owns_loop && !solver_plugin->stochastic && sp->sa3_lua_renoise;
    sa3_lua_ctx.engine_renoise = lua_renoise_active;
    std::vector<float> x_prev;
    if (lua_renoise_active) {
        x_prev.assign((size_t)n_lat, 0.0f);
    }

    // Apply guidance in-place on `v`. SA3 has no uncond, so the cond velocity is
    // both preds — see the header block. Transposes into the ACE view first:
    // apg_project indexes [t*Oc + c] and would otherwise scramble the channels.
    auto sa3_apply_guidance = [&]() {
        if (!use_lua_guidance) return;
        sa3_to_ace_view(v.data(), tm_c.data(), Oc, T);
        if (guidance_native_apg) {
            apg_forward(tm_c.data(), tm_c.data(), sp->guidance_scale, apg_mbuf,
                        tm_v.data(), Oc, T, sp->apg_norm_threshold, apg_ws);
        } else if (guidance_plugin->has_guide) {
            lua_call_guidance(*guidance_plugin, tm_c.data(), tm_c.data(), sp->guidance_scale,
                              apg_mbuf, tm_v.data(), Oc, T, g_ctx, sp->apg_norm_threshold,
                              sp->plugin_params, sa3_lua_ctx);
        } else {
            // post_step-only guidance plugin (no guide()): nothing to blend,
            // pass the cond velocity through unchanged. Avoids the "no guide()
            // function" error firing every step for a plugin that was never
            // meant to have one.
            memcpy(tm_v.data(), tm_c.data(), sizeof(float) * tm_c.size());
        }
        sa3_from_ace_view(tm_v.data(), v.data(), Oc, T);
    };

    // Evaluate the DiT at (xt_in, t_val) and leave the guided velocity in `v`.
    // Matches SolverModelFn / LoopModelFn: multi-eval solvers call this and then
    // read the result back out of the vt buffer they were handed.
    SolverModelFn sa3_model_fn = [&](const float * xt_in, float t_val) {
        if (!be->dit_forward(xt_in, t_val, cross.data(), sec_emb.data(),
                             local_add.data(), pad_mask.data(), L, v.data())) {
            fprintf(stderr, "[SA3] dit_forward failed at t=%.4f\n", t_val);
            return;
        }
        sa3_apply_guidance();
    };

    // Wrapped model callback for Lua solver paths only (step()'s multi-eval
    // model_fn and owns_loop's sample()). Those plugins call back with an
    // intermediate xt in ACE view, but dit_forward -- and sa3_model_fn's own
    // guidance transpose -- require SA3's native channel-major layout.
    // Convert in, evaluate via sa3_model_fn (which leaves the result, guided
    // if applicable, in native `v`), then convert that into tm_v -- the same
    // buffer registered with Lua as vt_buf. Mirrors mm3-plugins.h's
    // multi_eval model_fn (mm3-pipeline.h ~376-399). The native pingpong path
    // never calls this -- it uses sa3_model_fn directly on native x/v.
    SolverModelFn sa3_model_fn_ace = [&](const float * xt_in_ace, float t_val) {
        sa3_from_ace_view(xt_in_ace, lua_model_native_x.data(), Oc, T);
        sa3_model_fn(lua_model_native_x.data(), t_val);
        sa3_to_ace_view(v.data(), tm_v.data(), Oc, T);
    };

    // post_step hook for guidance plugins that need extra model evals (e.g.
    // manifold-projection corrections), mirrors mm3-plugins.h's
    // mm3_plugins_post_step (~478-499). SA3 has no uncond forward -- both
    // eval closures below run the cond model via sa3_model_fn_ace, and
    // vt_cond_buf/vt_uncond_buf both point at tm_v since there is nothing to
    // tell the two branches apart on this model. A post_step plugin written
    // to depend on cond and uncond diverging sees identical preds here, same
    // limitation guide() already has on SA3 (see the NOTE above). Skipped
    // entirely for guidance plugins with no post_step() (has_post_step
    // false), and for the final step (its output feeds nothing downstream).
    PostStepModelFn sa3_post_step_eval = [&](const float * xt_ace, float t_val) {
        sa3_model_fn_ace(xt_ace, t_val);
    };
    auto sa3_call_post_step = [&](int step_idx, float t_curr, float t_next) {
        if (!sa3_wants_post_step) return;
        if (step_idx >= steps - 1) return;
        sa3_to_ace_view(x.data(), tm_x.data(), Oc, T);
        GuidanceCtx ps_ctx = { step_idx, steps, t_curr - t_next, t_next };
        lua_call_post_step(*guidance_plugin, tm_x.data(), t_next, n_lat,
                           sa3_post_step_eval, sa3_post_step_eval, tm_v.data(), tm_v.data(),
                           ps_ctx, sp->plugin_params, sa3_lua_ctx);
        sa3_from_ace_view(tm_x.data(), x.data(), Oc, T);
    };

    if (use_lua_solver && solver_plugin->owns_loop) {
        // Full-loop solver: it takes the whole schedule and drives its own
        // iteration. Safe here in a way it is not on MM3 — SA3 refines the clip
        // in one pass, so there is no per-step window-overlap blend to bypass.
        // Full-loop solvers get the whole buffer up front (owns_loop hands the
        // plugin the entire xt/vt for the run, not just one step) -- same ACE
        // view conversion, just bracketing the single lua_call_solver_loop
        // call instead of a per-step call.
        //
        // NOTE: unlike the per-step path below, there is no re-noise wrapper
        // here — the plugin owns the whole loop, so there is no per-step
        // boundary to pingpong at. owns_loop solvers run on SA3 without
        // pingpong re-noise, period; sa3_lua_renoise does not apply to them.
        sa3_to_ace_view(x.data(), tm_x.data(), Oc, T);
        sa3_to_ace_view(v.data(), tm_v.data(), Oc, T);

        LoopModelFn loop_model_fn = [&](const float * xt_in, float t_val) { sa3_model_fn_ace(xt_in, t_val); };
        LoopOnStepFn loop_on_step = [&](int step_idx, float t_curr, float t_next) -> bool {
            g_ctx.step_idx = step_idx;
            g_ctx.t_curr   = t_curr;
            g_ctx.dt       = t_curr - t_next;
            fprintf(stderr, "[SA3] step %d/%d t=%.4f->%.4f [lua-loop]\n",
                    step_idx + 1, steps, t_curr, t_next);
            return cancelled && cancelled();
        };

        if (!lua_call_solver_loop(*solver_plugin, tm_x.data(), tm_v.data(), sigmas.data(), steps,
                                  n_lat, /*N=*/1, T, Oc, loop_model_fn, loop_on_step,
                                  sp->plugin_params, sa3_lua_ctx)) {
            fprintf(stderr, "[SA3] FATAL: full-loop solver '%s' failed (Lua error or missing sample())\n",
                    solver_plugin->name.c_str());
            return false;
        }
        // A cancel via on_step returns normally from sample(); stop here.
        if (cancelled && cancelled()) {
            fprintf(stderr, "[SA3] cancelled [lua-loop]\n");
            return false;
        }

        sa3_from_ace_view(tm_x.data(), x.data(), Oc, T);
    } else {
        int64_t seed64 = (int64_t) seed;  // lives across the whole sampling loop below
        SolverState solver_state;
        std::vector<float> tm_v_snap;  // ACE-view velocity snapshot for multi-eval solvers
        if (use_lua_solver) {
            solver_state.seeds   = &seed64;   // request seed so Lua stochastic solvers honour it (was nullptr -> seed 0)
            solver_state.batch_n = 1;
            solver_state.n_per   = n_lat;
            if (breeding) {  // Lua stochastic solvers mix their own lockstep B stream
                solver_state.seed_b     = sp->seed_b;
                solver_state.seed_blend = breed_t;
            }
            solver_state.xt_scratch.assign((size_t)n_lat, 0.0f);  // multi-eval solvers expect this pre-sized
            if (solver_plugin->needs_model) tm_v_snap.assign((size_t)n_lat, 0.0f);
        }

        for (int i = 0; i < steps; i++) {
            if (cancelled && cancelled()) {
                fprintf(stderr, "[SA3] cancelled before step %d/%d\n", i + 1, steps);
                return false;
            }
            float t_curr = sigmas[i], t_next = sigmas[i + 1];
            g_ctx.step_idx = i;
            g_ctx.t_curr   = t_curr;
            g_ctx.dt       = t_curr - t_next;

            if (!be->dit_forward(x.data(), t_curr, cross.data(), sec_emb.data(),
                                 local_add.data(), pad_mask.data(), L, v.data())) return false;
            sa3_apply_guidance();

            if (use_lua_solver) {
                // Snapshot x BEFORE the solver's step() call, native (channel-
                // major) layout — the re-noise wrapper below needs the pre-step
                // and post-step x to recover the solver's effective velocity
                // over [t_curr, t_next].
                if (lua_renoise_active) {
                    memcpy(x_prev.data(), x.data(), sizeof(float) * n_lat);
                }

                // Present the latents/velocity in ACE's time-major view --
                // plugins are written against that layout, not SA3's native
                // channel-major one (see sa3_to_ace_view/sa3_from_ace_view
                // above). tm_v already holds the guided velocity as of
                // sa3_apply_guidance() above.
                sa3_to_ace_view(x.data(), tm_x.data(), Oc, T);
                sa3_to_ace_view(v.data(), tm_v.data(), Oc, T);

                // Buffer separation, as hot-step-sampler.h does: a multi-eval
                // solver's model_fn overwrites tm_v (via sa3_model_fn_ace), so
                // the "original velocity" argument must be a snapshot or
                // solvers that read it after calling model_fn (Heun's
                // corrector, UniPC's history) silently degrade to Euler.
                const float * vt_readonly = tm_v.data();
                if (solver_plugin->needs_model) {
                    tm_v_snap.assign(tm_v.begin(), tm_v.end());
                    vt_readonly = tm_v_snap.data();
                }
                solver_state.step_index = i;
                lua_call_solver_step(*solver_plugin, tm_x.data(), vt_readonly, t_curr, t_next,
                                     n_lat, solver_state, sa3_model_fn_ace, tm_v.data(),
                                     sp->plugin_params, sa3_lua_ctx);

                sa3_from_ace_view(tm_x.data(), x.data(), Oc, T);

                // Re-noise wrapper (sa3_lua_renoise, default true): SA3 is a
                // distilled model tuned for pingpong renoise, not bare ODE
                // integration — without this, every Lua solver comes out ~2x
                // hot/crackly on this model regardless of plugin. `v_eff` is
                // the solver's effective velocity over the step (for a pure
                // Euler plugin this equals the model velocity; for corrected
                // solvers it folds in their correction) -- computed per-element
                // as a local scalar, no extra n_lat scratch buffer needed
                // (x_prev is the only new allocation this wrapper requires;
                // `v`'s next use is next iteration's dit_forward, which
                // overwrites it regardless, so there was nothing to protect by
                // reusing it here). `denoised`/pingpong math below mirrors the
                // native branch exactly, including drawing from the SAME
                // `rng`/`gauss` stream at the SAME point in the per-step
                // sequence a native pingpong step would, so per-step draw
                // count and statistics match native step-for-step.
                if (lua_renoise_active && t_curr > t_next + 1e-6f) {
                    const float inv_dt = 1.0f / (t_curr - t_next);
                    for (size_t k = 0; k < x.size(); k++) {
                        float v_eff    = (x_prev[k] - x[k]) * inv_dt;
                        float denoised = x_prev[k] - t_curr * v_eff;
                        float n = renoise_draw();
                        x[k] = (t_next > 0.0f) ? ((1.0f - t_next) * denoised + t_next * n) : denoised;
                    }
                }
                sa3_call_post_step(i, t_curr, t_next);
            } else if (pingpong) {
                // denoised = x - t*v; x = (1-t_next)*denoised + t_next*randn
                for (size_t k = 0; k < x.size(); k++) {
                    float denoised = x[k] - t_curr * v[k];
                    float n = renoise_draw();
                    x[k] = (1.0f - t_next) * denoised + t_next * n;
                }
                sa3_call_post_step(i, t_curr, t_next);
            } else {
                float dt = t_next - t_curr;
                for (size_t k = 0; k < x.size(); k++) x[k] += dt * v[k];
                // NOTE: plain-ODE branch also needs post_step, mirroring the
                // pingpong branch above -- without this call, guidance plugins
                // relying on post_step() silently never fire when pingpong is off.
                sa3_call_post_step(i, t_curr, t_next);
            }
            fprintf(stderr, "[SA3] step %d/%d t=%.4f->%.4f%s%s\n", i + 1, steps, t_curr, t_next,
                    use_lua_solver ? " [lua-solver]" : "", use_lua_guidance ? " [lua-guidance]" : "");
        }
    }

    // HOT-Step diagnostic: latent statistics + optional raw dump. The Lua
    // plugins were tuned against ACE's 64-channel latent (RMS ~2.0); SA3's
    // 256-channel latent sits wherever it sits, and every rms_target slider
    // needs that number. Always logs; only writes a file when asked.
    {
        const int Lv = valid > L ? L : valid;
        double sum_sq = 0.0, sum_abs = 0.0;
        float  vmin = x.empty() ? 0.0f : x[0], vmax = vmin;
        std::vector<double> ch_sq((size_t) Oc, 0.0);
        for (int cidx = 0; cidx < Oc; cidx++) {
            for (int t = 0; t < Lv; t++) {
                float v = x[(size_t) cidx * L + t];
                double d = (double) v;
                sum_sq  += d * d;
                sum_abs += std::fabs(d);
                ch_sq[(size_t) cidx] += d * d;
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        const double n_valid = (double) Oc * Lv;
        const double rms  = std::sqrt(sum_sq / n_valid);
        const double mabs = sum_abs / n_valid;
        double ch_rms_min = 1e30, ch_rms_max = 0.0;
        for (int cidx = 0; cidx < Oc; cidx++) {
            double r = std::sqrt(ch_sq[(size_t) cidx] / (double) Lv);
            if (r < ch_rms_min) ch_rms_min = r;
            if (r > ch_rms_max) ch_rms_max = r;
        }
        fprintf(stderr, "[SA3] final latent: rms=%.4f mean|x|=%.4f min=%.3f max=%.3f "
                        "per-ch rms [%.4f .. %.4f]  (Oc=%d L=%d valid=%d, %s)\n",
                rms, mabs, vmin, vmax, ch_rms_min, ch_rms_max, Oc, L, Lv,
                noise_only ? "synth" : "refine");
        if (dump_latent_path && dump_latent_path[0]) {
            // Flat f32, channel-major [Oc][L] exactly as held in memory.
            // Header: int32 ndims=2, int64 shape[2]={Oc, L}, then data.
            FILE * fp = fopen(dump_latent_path, "wb");
            if (fp) {
                const int32_t ndims = 2;
                const int64_t shape[2] = { (int64_t) Oc, (int64_t) L };
                fwrite(&ndims, sizeof(int32_t), 1, fp);
                fwrite(shape, sizeof(int64_t), 2, fp);
                fwrite(x.data(), sizeof(float), x.size(), fp);
                fclose(fp);
                fprintf(stderr, "[SA3] latent dumped -> %s (%zu floats)\n", dump_latent_path, x.size());
            } else {
                fprintf(stderr, "[SA3] WARNING: could not open %s for latent dump\n", dump_latent_path);
            }
        }
    }

    // Decode, zero the padded region, trim to input length. No saturation —
    // see the note on `out` above.
    //
    // HOT-Step: optional Lua postprocess plugin, mirrors ACE's
    // ops_vae_decode_postprocess (pipeline-synth-ops.cpp) -- routes the
    // latent->audio decode through a Lua plugin's process() instead of the
    // built-in tiled decoder. ACE's hook sits per-batch-item, once per output
    // waveform (its own decode_fn is internally tiled, same as SA3's
    // sa3_decode_tiled); SA3 only ever has one item per request, so this is
    // a single call over the WHOLE latent `x` (not just the valid/trimmed
    // region) -- exactly the same span the built-in path below decodes, so
    // `S_dec = L * SA3_DS` below stays correct either way.
    //
    // Buffer contract, verified against lua-plugin.h's lua_call_postprocess:
    //   - latents in:  time-major [T][C_lat], same as ACE's dit_out. SA3's
    //     latent `x` is channel-major [Oc][L] (see CONVENTION 2 above), so it
    //     is transposed with the existing sa3_to_ace_view() helper.
    //   - audio out:   planar per-channel [C_aud][T_audio] (channel-major
    //     blocks, NOT interleaved samples) -- identical to SA3's `out`/
    //     `decoded` layout already used by sa3_decode_tiled(), so no
    //     transpose is needed on the way back.
    //   - the vae_decode_fn callback the plugin can invoke follows the same
    //     time-major-in convention, so it transposes back to channel-major
    //     before calling sa3_decode_tiled() and hands back planar audio.
    LuaPlugin * pp_plugin = nullptr;
    if (sp && !sp->postprocess_plugin.empty()) {
        pp_plugin = PluginRegistry::instance().postprocess_lookup(sp->postprocess_plugin.c_str());
        if (!pp_plugin) {
            fprintf(stderr, "[SA3] Postprocess WARNING: plugin '%s' not found, falling back to built-in decode\n",
                    sp->postprocess_plugin.c_str());
        }
    }

    std::vector<float> decoded;
    int valid_samples = valid * SA3_DS;
    bool decoded_ok = false;
    if (pp_plugin) {
        std::string plugin_finite_error;
        std::vector<float> tm_lat((size_t) L * Oc);
        sa3_to_ace_view(x.data(), tm_lat.data(), Oc, L);

        PostprocessVaeDecodeFn decode_fn =
            [&](const float * latent_tm, int T_lat, float * aud_out, int max_T) -> int {
                // Decoder graphs (ONNX + GGML) are fixed at SA3_CHUNK_LAT, so the only
                // short-tile path sa3_decode_tiled takes is zero-pad ("padded"); logged
                // here because the tile size is chosen by the plugin (e.g. md_audio_tiled),
                // not by this callback.
                if (T_lat < SA3_CHUNK_LAT) {
                    fprintf(stderr, "[SA3] Postprocess tile decode: T_lat=%d (%s)\n",
                            T_lat, "padded");
                }
                std::vector<float> cm_lat((size_t) Oc * T_lat);
                sa3_from_ace_view(latent_tm, cm_lat.data(), Oc, T_lat);
                std::vector<float> aud;
                if (!sa3_decode_tiled(be, cm_lat.data(), T_lat, aud, -1, &plugin_finite_error)) return -1;
                int t_audio = T_lat * SA3_DS;
                if (t_audio > max_T) t_audio = max_T;
                memcpy(aud_out,          aud.data(),                  sizeof(float) * (size_t) t_audio);
                memcpy(aud_out + max_T,  aud.data() + T_lat * SA3_DS, sizeof(float) * (size_t) t_audio);
                return t_audio;
            };

        int max_T_audio = L * SA3_DS;
        std::vector<float> pp_audio((size_t) 2 * max_T_audio);
        int T_audio = lua_call_postprocess(*pp_plugin, tm_lat.data(), L, Oc, 2,
                                           pp_audio.data(), max_T_audio, decode_fn,
                                           sp->plugin_params, sa3_lua_ctx, SA3_DS);
        // A plugin may swallow a failed decode; never fall back after invalid PCM.
        if (!plugin_finite_error.empty()) {
            if (finite_error) *finite_error = plugin_finite_error;
            return false;
        }
        if (T_audio > 0) {
            decoded.assign((size_t) 2 * max_T_audio, 0.0f);
            int n = T_audio < max_T_audio ? T_audio : max_T_audio;
            memcpy(decoded.data(),               pp_audio.data(),               sizeof(float) * (size_t) n);
            memcpy(decoded.data() + max_T_audio, pp_audio.data() + max_T_audio, sizeof(float) * (size_t) n);
            decoded_ok = true;
            fprintf(stderr, "[SA3] Postprocess '%s' decode: T_audio=%d (T_latent=%d)\n",
                    pp_plugin->name.c_str(), T_audio, L);
        } else {
            fprintf(stderr, "[SA3] Postprocess '%s' ERROR: plugin decode failed, falling back to built-in decode\n",
                    pp_plugin->name.c_str());
        }
    }
    if (!decoded_ok) {
        if (!sa3_decode_tiled(be, x.data(), L, decoded, T44 < valid_samples ? T44 : valid_samples,
                              finite_error)) return false;
    }
    int S_dec = L * SA3_DS;
    out.assign((size_t)2 * T44, 0.0f);
    int n_valid = 0;
    for (int chn = 0; chn < 2; chn++) {
        const float * src = decoded.data() + (size_t)chn * S_dec;
        float * dst = out.data() + (size_t)chn * T44;
        int n = T44 < valid_samples ? T44 : valid_samples;
        n_valid = n;
        for (int k = 0; k < n; k++) {
            dst[k] = src[k];
        }
    }

    if (!synth_finite(out.data(), out.size(), "SA3 final PCM", finite_error)) return false;

    // Audio-domain rms/peak in dBFS over the final trimmed planar-stereo
    // output, for quick clip/level sanity checks in the engine log.
    {
        double sum_sq = 0.0;
        float  peak   = 0.0f;
        size_t count  = (size_t) 2 * n_valid;
        for (int chn = 0; chn < 2; chn++) {
            const float * dst = out.data() + (size_t)chn * T44;
            for (int k = 0; k < n_valid; k++) {
                float v = dst[k];
                sum_sq += (double) v * (double) v;
                float av = fabsf(v);
                if (av > peak) peak = av;
            }
        }
        float rms = count > 0 ? (float) sqrt(sum_sq / (double) count) : 0.0f;
        float rms_db  = rms  > 0.0f ? 20.0f * log10f(rms)  : -120.0f;
        float peak_db = peak > 0.0f ? 20.0f * log10f(peak) : -120.0f;
        fprintf(stderr, "[SA3] audio: rms=%.1f dBFS peak=%.1f dBFS (%d ch, %d samples @ %d Hz)\n",
                rms_db, peak_db, 2, n_valid, SA3_SR);
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// GGML backend
// ═══════════════════════════════════════════════════════════════════════

#include "sa3-dit-ggml.h"
#include "sa3-same-ggml.h"
#include "sa3-t5gemma-enc.h"

struct Sa3GgmlRefine final : Sa3Backend {
    SA3T5GemmaEnc      text_enc_m{};
    SA3SecondsEmbedder seconds_m{};
    SA3Same            same_enc_m{};
    SA3Same            same_dec_m{};
    SA3DiT             dit_m{};
    bool               loaded = false;
    std::string        dir;

    const char * name() const override { return "gguf"; }

    bool text_encode(const int64_t * ids, int n_tokens, float * out) override {
        int32_t ids32[SA3_TOK_LEN];
        uint8_t mask[SA3_TOK_LEN];
        for (int i = 0; i < SA3_TOK_LEN; i++) {
            ids32[i] = (int32_t)ids[i];
            mask[i]  = (i < n_tokens) ? 1 : 0;
        }
        // output token-major [256,768] == ONNX [1,256,768]
        sa3_t5gemma_forward(&text_enc_m, ids32, mask, SA3_TOK_LEN, out);
        return true;
    }
    bool seconds_embed(float seconds, float * out) override {
        sa3_seconds_embed(seconds_m, seconds, out);
        return true;
    }
    bool same_encode_chunk(const float * audio, float * latents) override {
        // module I/O is torch-contiguous: audio planar [2,S], latents [256,128]
        sa3_same_forward(&same_enc_m, audio, latents, SA3_CHUNK_LAT);
        return true;
    }
    bool same_decode_chunk(const float * latents, float * audio) override {
        sa3_same_forward(&same_dec_m, latents, audio, SA3_CHUNK_LAT);
        return true;
    }
    bool dit_forward(const float * x, float t, const float * cross, const float * glob,
                     const float * local, const char * pad_mask, int T, float * v) override {
        // module I/O is torch-contiguous: x [256,T], cross [257,768] token-major,
        // local [257,T], out [256,T] — identical to the orchestration buffers.
        sa3_dit_forward(&dit_m, x, t, cross, SA3_TOK_LEN + 1, glob, local,
                        reinterpret_cast<const uint8_t *>(pad_mask), T, v);
        return true;
    }
};

// Load all four GGUFs from `dir` (the models root, same place the pp-vae
// GGUFs live). `dit_file` = the generation DiT filename in `dir` (BF16 or a
// quant). The seconds embedder reads its two tiny tensors from that DiT
// GGUF (conditioner.* extras). `adapters` (optional) = StableStep DoRA
// adapter GGUFs merged into the DiT weights at load (see sa3-dit-ggml.h).
static inline bool sa3_ggml_load(Sa3GgmlRefine * m, const char * dir, const char * dit_file,
                                 const std::vector<Sa3AdapterSpec> * adapters = nullptr) {
    if (!m || !dir || !dit_file) return false;
    m->dir = dir;
    std::string d   = dir;
    std::string dit = d + "/" + dit_file;
    bool ok = sa3_t5gemma_load(&m->text_enc_m, (d + "/sa3-text-enc-BF16.gguf").c_str())
           && sa3_seconds_embedder_load(&m->seconds_m, dit.c_str())
           && sa3_same_load(&m->same_enc_m, (d + "/sa3-same-enc-F16.gguf").c_str(), true)
           && sa3_same_load(&m->same_dec_m, (d + "/sa3-same-dec-F16.gguf").c_str(), false)
           && sa3_dit_load(&m->dit_m, dit.c_str(), adapters);
    m->loaded = ok;
    fprintf(stderr, "[SA3] GGML backend loaded from %s, DiT %s (%zu adapters): %s\n", dir, dit_file,
            adapters ? adapters->size() : 0, ok ? "OK" : "FAILED");
    return ok;
}

static inline void sa3_ggml_free(Sa3GgmlRefine * m) {
    if (!m) return;
    // Module free functions tolerate never-loaded (zero-initialised) members.
    if (m->text_enc_m.sched) sa3_t5gemma_free(&m->text_enc_m);
    if (m->same_enc_m.sched) sa3_same_free(&m->same_enc_m);
    if (m->same_dec_m.sched) sa3_same_free(&m->same_dec_m);
    if (m->dit_m.sched)      sa3_dit_free(&m->dit_m);
    m->loaded = false;
}

static inline bool sa3_refine_run_ggml(Sa3GgmlRefine * m,
                                       const float * audio, int T44,
                                       const int64_t * token_ids, int n_tokens,
                                       float strength, int steps,
                                       bool pingpong, uint64_t seed, bool zero_noise,
                                       std::vector<float> & out,
                                       const Sa3PluginParams * sp = nullptr,
                                       const std::function<bool()> & cancelled = nullptr,
                                       std::string * finite_error = nullptr) {
    if (!m || !m->loaded) return false;
    return sa3_refine_run_backend(m, audio, T44, token_ids, n_tokens,
                                  strength, steps, pingpong, seed, zero_noise, out, sp,
                                  /*noise_only=*/false, /*dump_latent_path=*/nullptr, cancelled, finite_error);
}

// HOT-Step /sa3-synth: text-to-audio from noise on the GGML backend.
// T44 = requested output length in 44.1k samples.
static inline bool sa3_synth_run_ggml(Sa3GgmlRefine * m,
                                      int T44,
                                      const int64_t * token_ids, int n_tokens,
                                      int steps, bool pingpong, uint64_t seed, bool zero_noise,
                                      std::vector<float> & out,
                                      const Sa3PluginParams * sp = nullptr,
                                      const char * dump_latent_path = nullptr,
                                      const std::function<bool()> & cancelled = nullptr,
                                      std::string * finite_error = nullptr) {
    if (!m || !m->loaded) return false;
    return sa3_refine_run_backend(m, nullptr, T44, token_ids, n_tokens,
                                  1.0f, steps, pingpong, seed, zero_noise, out, sp,
                                  /*noise_only=*/true, dump_latent_path, cancelled, finite_error);
}
#endif // SA3_REFINE_H
