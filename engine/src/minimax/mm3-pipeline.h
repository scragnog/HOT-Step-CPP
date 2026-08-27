#pragma once
// minimax/mm3-pipeline.h — THE ASSEMBLY: the end-to-end MiniMax-Music3 pipeline.
//
// HOT-Step file (does not exist upstream). Every module below this file has
// already been ported and proven against its own fixture in isolation:
//
//     mm3-ar-loop.h       stage 1: Qwen3 AR planner + RVQ depth decoder
//     mm3-cond-graph.h    condition encoder (AR hiddens -> latent-rate conditioning)
//     mm3-dit-graph.h     flow DiT + the single-window Euler loop
//     mm3-vocoder-graph.h DAC-style vocoder (latents -> stereo PCM)
//
// This header is the ORCHESTRATION that joins them: the chunk-window loop, the
// overlap arithmetic, and the stitch. It adds exactly one piece of new math —
// the per-step overlap blend — and otherwise only moves buffers between the
// proven modules.
//
// ── The chunk contract (reference: diffusers @ dafe3733,
//    modular_pipelines/minimax_music3/{before_denoise,denoise,decoders}.py) ──
//
// 1. WINDOWS. The AR stage emits F frames at 25 fps. The flow stage consumes
//    them in 200-frame windows with a 100-frame hop:
//
//        chunk_starts = [0]                         if F <= 200
//                     = range(0, F - 100, 100)      otherwise
//
//    Note the `F - 100` (not `F`): the last window is allowed to run off the
//    end of the frame array and is simply shorter. 300 frames -> starts
//    [0, 100] -> 2 windows, both a full 200 frames. This is the 12 s fixture.
//
// 2. LATENT LENGTH. Each window's condition encoder output is
//    L = int(win_frames * 3.4453125) latent frames (86.1328125 Hz against
//    25 fps). 200 frames -> 689 latents. mm3_cond_latent_length() owns the
//    exact truncation order.
//
// 3. THE OVERLAP. Consecutive windows share 100 AR frames == ~344 latents, but
//    only the first OVERLAP_LATENTS = 172 of them are spliced. Window k > 0
//    receives, from window k-1:
//
//        prev_latent    = latents_{k-1}[..., L-344 : L-172]     [128, 172]
//        prev_condition = condition_{k-1}[L-344 : L-172, ...]   [172, 2048]
//
//    and then:
//      (a) BEFORE the flow loop, overwrites its own conditioning over
//          [0, overlap) with prev_condition — the previous window's view of
//          those frames wins;
//      (b) at the TOP OF EVERY EULER STEP, re-blends its latents over
//          [0, overlap):
//
//              x[:, :ov] = (1 - (1 - 1e-6) * t) * noise_prompt + t * prev_latent
//
//          where `noise_prompt` is this window's ORIGINAL noise over the
//          overlap (snapshot before step 0, never re-drawn) and `t` is the
//          scheduler timestep (0 = pure noise, ->1 = clean). This is the one
//          genuinely new piece of arithmetic in this increment;
//      (c) AFTER the last step, overwrites [0, overlap) with prev_latent
//          outright — the blend was a soft boundary condition for the DiT, the
//          hard splice is what gets vocoded.
//
//    overlap itself is min(prev_latent_len, L), so a short final window
//    degrades gracefully.
//
// 4. THE STITCH. Every window is vocoded WHOLE (L * 512 samples/channel), then
//    cropped in the sample domain:
//
//        left  = 0 if k == 0        else CROP_LEFT_LATENTS  (86)  * 512
//        right = 0 if k == last     else CROP_RIGHT_LATENTS (258) * 512
//
//    86 + 258 = 344 = the full latent overlap, so the kept spans tile the song
//    exactly once. Two 689-latent windows -> 431*512 + 603*512 = 529,408
//    samples/channel = 12.005 s at 44.1 kHz, which is exactly the fixture's
//    `final_audio.bin`. Finally clamp to [-1, 1].
//
//    Note the ordering: vocode THEN crop. Cropping the latents first would be
//    cheaper and wrong — the vocoder's receptive field bleeds across the seam.
//
// ── Noise ───────────────────────────────────────────────────────────────────
//
// Each window draws its own [128, L] Gaussian noise. In parity mode the caller
// supplies it (the fixtures' `flow_w{0,1}_noise_latents.bin`). Otherwise it is
// derived deterministically from `seed` and the window index via a splitmix64
// stream + Box-Muller — deliberately NOT std::normal_distribution, whose bit
// pattern is libstdc++/MSVC-dependent. Same seed + same window index => same
// latents on every platform and every build.
//
// This will NOT match torch's noise for the same seed number (torch's Mersenne
// normal stream is its own thing), so a free-run generation here is not
// bit-comparable to a diffusers run. Parity is proven by REPLAYING the
// reference's own noise, not by re-deriving it.
//
// ── Concurrency ─────────────────────────────────────────────────────────────
//
// Not thread-safe. Every module it drives keeps a process-global graph cache
// (g_mm3_lm / g_mm3_cond / g_mm3_dit / g_mm3_voc); the caller serialises
// (mm3-server.h holds g_mm3_mutex for the whole generate).

#include "mm3-ar-loop.h"
#include "mm3-cond-graph.h"
#include "mm3-dit-graph.h"
#include "mm3-model.h"
#include "mm3-plugins.h"
#include "mm3-tokenizer.h"
#include "mm3-vocoder-graph.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// Reference constants. `_OVERLAP_LATENT_LENGTH` and `_CROP_LEFT_LATENT` are
// literals in diffusers too (denoise.py / decoders.py) — they are NOT derived
// from the window geometry there, so they are not derived here either. The
// relations 344 = 2*172 and 258 = 344 - 86 hold, and are documented rather
// than exploited.
#define MM3_OVERLAP_LATENTS    172
#define MM3_CARRY_SPAN_LATENTS 344  // = 2 * MM3_OVERLAP_LATENTS
#define MM3_CROP_LEFT_LATENTS  86
#define MM3_CROP_RIGHT_LATENTS 258  // = MM3_CARRY_SPAN_LATENTS - MM3_CROP_LEFT_LATENTS

// ── Progress ────────────────────────────────────────────────────────────────

// One callback for every stage transition and every unit of inner work. The
// job-queue integration point: a future /mm3/synth on the shared GPU worker
// forwards these straight into the existing job progress channel.
struct MM3GenProgress {
    const char * stage     = "";  // "ar" | "cond" | "flow" | "vocode" | "stitch" | "done"
    int64_t      window    = -1;  // 0-based; -1 for song-wide stages
    int64_t      n_windows = 0;
    int64_t      step      = 0;   // AR frame, or Euler step, within this stage
    int64_t      n_steps   = 0;
};

using MM3ProgressCb = std::function<void(const MM3GenProgress &)>;

// ── Streaming chunk sink ────────────────────────────────────────────────────
//
// Called once per window, on the generating thread, the moment that window's
// PCM is FINAL — after it has been vocoded and cropped. Nothing downstream
// ever touches those samples again (fact 4: vocode-then-crop, and the kept
// spans tile the song exactly once), so a chunk is safe to play the instant it
// arrives.
//
//   seq            0-based window index; chunks arrive in order, no gaps
//   sample_offset  where this chunk starts in the finished song, per channel
//   n_samples      length of this chunk, per channel. 0 is legal for a
//                  degenerate final window and must be tolerated
//   planar         2 * n_samples floats, [L: n_samples][R: n_samples], already
//                  clamped to [-1, 1] with NaN/Inf zeroed — i.e. bit-for-bit
//                  the values the finished WAV will carry. BORROWED: valid
//                  only for the duration of the call.
//
// Installing a sink also changes WHEN the vocoder runs: inline, per window,
// instead of in one pass after every window has been denoised. That is a
// scheduling change only — the vocoder is a pure function of one window's
// latents, and the latents are never revisited once mm3_flow_sample_chunk
// returns — but it is why the sink is opt-in rather than always on. A render
// with no sink keeps today's exact stage order, VRAM profile and graph-cache
// behaviour, byte for byte.
using MM3ChunkCb = std::function<void(int take, int64_t seq, int64_t sample_offset, int64_t n_samples,
                                      const float * planar,
                                      int sample_rate)>;

// Which span of a vocoded window survives the stitch (fact 4). THE one place
// this is computed: the streaming sink and the final stitch must agree sample
// for sample, or the streamed concatenation stops being the saved file — and
// that is exactly the bug an ear cannot localise.
static inline void mm3_window_crop_lr(bool is_first, bool is_last, int64_t L, int64_t UP, int64_t * left,
                                      int64_t * len) {
    const int64_t T = L * UP;
    const int64_t l = is_first ? 0 : (int64_t) MM3_CROP_LEFT_LATENTS * UP;
    // The LAST window has no right crop — getting this wrong truncates the
    // ending, and on a streamed render it would truncate it silently.
    const int64_t r = is_last ? 0 : (int64_t) MM3_CROP_RIGHT_LATENTS * UP;
    int64_t       n = T - l - r;
    if (n < 0) {
        n = 0;
    }
    *left = l;
    *len  = n;
}

/** The same thing, addressed by index once the window count is settled.
 *
 *  Two spellings, because the streaming dispatcher knows "is there another
 *  window after this one?" before it knows how many there are in total — but
 *  they resolve to ONE piece of arithmetic, which is what keeps the streamed
 *  bytes and the stitched file identical. */
static inline void mm3_window_crop(int64_t k, int64_t NW, int64_t L, int64_t UP, int64_t * left, int64_t * len) {
    mm3_window_crop_lr(k == 0, k == NW - 1, L, UP, left, len);
}

/** The finished-audio value of one vocoder sample. Shared by the streaming
 *  sink and the final clamp pass so the two cannot drift. */
static inline float mm3_clamp_sample(float v) {
    if (std::isnan(v) || std::isinf(v)) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    if (v < -1.0f) {
        return -1.0f;
    }
    return v;
}

// ── The overlap-aware Euler loop ────────────────────────────────────────────
//
// A strict superset of mm3_flow_sample() (mm3-dit-graph.h), which stays exactly
// as committed and remains the path behind POST /mm3/flow-sample. With
// overlap == 0 the two are the same arithmetic in the same order; validation
// ladder step A re-proves that by replaying window 0 through THIS function and
// comparing against the same fixture the committed one is proven against.
//
//   noise        128*L floats, channel-major. Also serves as the overlap
//                noise prompt: `noise[:, :overlap]` is read at every step, so
//                it must stay valid for the whole call (it is never mutated —
//                the loop mutates `out_latents`, its own copy).
//   cond         2048*L floats, torch [1, L, 2048] order. Uploaded once.
//   prev_latent  128*prev_stride floats, channel-major, the previous window's
//                carry. Only [0, overlap) of each channel is read.
//   plugins      optional Lua solver/scheduler/guidance override (mm3-plugins.h).
//                nullptr, or a run with `active == false`, leaves EVERY
//                expression below exactly as the fixtures proved it — the
//                plugin path is a branch, never a rewrite of the default.
//   uncond_interval
//                CFG guidance-delta cache. 1 (the default) = the exact
//                reference: both branches every step. N >= 2 evaluates the
//                UNCONDITIONAL branch only on the warmup steps (0, 1), the
//                final step, and every Nth step, reconstructing it in between
//                from the cached guidance delta. See the design note below.
static bool mm3_flow_sample_chunk(const MM3Model & m, const float * noise, const float * cond, int64_t L, int steps,
                                  float cfg_scale, int64_t overlap, const float * prev_latent, int64_t prev_stride,
                                  std::vector<float> & out_latents, MM3FlowStats * stats,
                                  const std::function<void(int, int)> & on_step,
                                  const std::function<bool()> & should_cancel, std::string * err,
                                  MM3PluginRun * plugins = nullptr, int uncond_interval = 1) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (steps <= 0 || steps > 1000) {
        if (err) {
            *err = "steps must be in 1..1000";
        }
        return false;
    }
    if (overlap < 0 || overlap > L) {
        if (err) {
            *err = "overlap must be in 0..L";
        }
        return false;
    }
    if (overlap > 0 && (!prev_latent || prev_stride < overlap)) {
        if (err) {
            *err = "a positive overlap needs a previous-window carry at least that long";
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }

    const int64_t C = (int64_t) m.synth_cfg.dit.in_channels;
    const int64_t N = C * L;

    // Hand the Lua plugins MM3's real geometry. The latent rate is the flow
    // latent rate, derived the same way mm3_cond_latent_length() derives L —
    // one second of AR frames run through the conditioning resampler.
    if (plugins) {
        const MM3CondConfig & lc  = m.synth_cfg.cond;
        int                   fps = 0;
        if (lc.input_sampling_rate && lc.output_hop_length && m.lm_cfg.frame_rate) {
            fps = (int) mm3_cond_latent_length(lc, (int64_t) m.lm_cfg.frame_rate);
        }
        mm3_plugins_set_geometry(*plugins, C, fps);
    }
    out_latents.assign((size_t) N, 0.0f);
    memcpy(out_latents.data(), noise, (size_t) N * sizeof(float));

    // ── Plugin path (opt-in) ────────────────────────────────────────────────
    // `use_plugins` false => the original arithmetic, untouched. See
    // mm3-plugins.h for the sigma<->t and channel-major<->time-major mappings.
    const bool use_plugins = plugins && plugins->active;

    // Batched CFG (mm3-dit-graph.h design note B): both branches in one graph
    // compute. Opt-in via MM3_DIT_CFG_BATCH=1 — measured SLOWER than two-pass
    // (see the note), kept for re-testing on future ggml versions. Disabled
    // for post_step guidance plugins regardless — those evaluate ONE branch at
    // a time through the B=1 graph, and alternating B=2/B=1 computes would
    // rebuild the graph twice per step.
    const bool wants_post_step = use_plugins && plugins->guidance && plugins->guidance->has_post_step;
    const bool batched_cfg     = mm3_dit_cfg_batched() && !wants_post_step;

    // -- CFG guidance-delta cache --------------------------------------------
    //
    // CFG is `v = u + s*(c - u)`, which is `v = c + (s-1)*delta` for the
    // guidance delta `delta = c - u`. The conditional prediction `c` moves fast
    // along the trajectory; `delta` moves slowly. So `delta` can be held for a
    // step: evaluate the unconditional branch every Nth step, and in between
    // rebuild `u = c - delta` from the FRESH `c` and the cached delta.
    //
    // That rebuild -- rather than a special-cased CFG combine -- is deliberate:
    // every consumer downstream (the native combine, mm3_plugins_guide, APG)
    // keeps receiving a coherent (cond, uncond) pair and needs no knowledge of
    // this at all. The approximation it makes is exactly "the delta is one step
    // stale", which is the whole claim being traded on.
    //
    // Steps 0 and 1 (warmup, where the delta is still moving) and the final
    // step (which lands the sample) always evaluate both branches.
    //
    // Suppressed, not honoured-partially, in two cases:
    //   - batched_cfg: both branches come out of ONE compute there, so skipping
    //     the uncond half would mean swapping to a B=1 graph mid-loop. No win,
    //     and a graph swap per step is its own hazard.
    //   - post_step guidance plugins: those drive their own extra forwards of
    //     each branch and are entitled to real ones.
    const int  uncond_every = uncond_interval < 1 ? 1 : (uncond_interval > steps ? steps : uncond_interval);
    const bool delta_cache  = uncond_every >= 2 && !batched_cfg && !wants_post_step;
    if (uncond_every >= 2 && !delta_cache) {
        fprintf(stderr, "[MM3-Flow] flow_uncond_interval=%d ignored (%s)\n", uncond_every,
                batched_cfg ? "batched CFG" : "post_step guidance plugin");
    }
    std::vector<float> cfg_delta;
    if (delta_cache) {
        cfg_delta.assign((size_t) N, 0.0f);
    }
    int64_t n_uncond = 0;

    std::vector<float> sigmas, timesteps;
    if (plugins && plugins->scheduler) {
        mm3_plugins_schedule(*plugins, steps, &sigmas, &timesteps);
    } else {
        mm3_flow_sigmas(steps, &sigmas, &timesteps);
    }

    std::vector<float> pred_c((size_t) N);
    std::vector<float> pred_u((size_t) N);

    // Plugin-only scratch. Left empty (zero allocation) on the native path.
    std::vector<float> v_guided, pred_c2, pred_u2, v_scratch;
    bool               sub_eval_failed = false;
    std::string        sub_eval_err;
    int                cur_step   = 0;
    float              cur_dt_ace = 0.0f;

    SolverModelFn   model_fn;
    PostStepModelFn eval_cond_fn, eval_uncond_fn;

    if (use_plugins) {
        // One window == one ACE "generation" for state purposes: momentum and
        // solver history reset here, never across the seam.
        plugins->begin_window(N);
        v_guided.assign((size_t) N, 0.0f);

        const bool multi_eval    = plugins->solver && plugins->solver->needs_model;
        const bool wants_post    = wants_post_step;
        if (multi_eval || wants_post) {
            pred_c2.assign((size_t) N, 0.0f);
            pred_u2.assign((size_t) N, 0.0f);
            v_scratch.assign((size_t) N, 0.0f);
        }

        // Multi-eval solvers (Heun, RK4, UniPC, DOPRI5…) re-evaluate the model
        // at an intermediate (xt, t). That is two more full 2.4B forwards per
        // call — budget accordingly; they are charged to stats->forwards.
        //
        // `cond` is deliberately null in every sub-evaluation: the conditioning
        // is already resident, uploaded by the step-0 conditional pass which
        // always precedes any solver call. Re-uploading would cost a host->device
        // copy per sub-evaluation for a value that cannot have changed.
        if (multi_eval) {
            model_fn = [&](const float * xt_ace, float t_val) {
                const float sig = 1.0f - t_val;
                mm3_plug_from_ace_view(xt_ace, plugins->cm_tmp.data(), C, L, /*negate=*/false);
                // A sub-evaluation needs both branches — the batched graph's
                // exact shape — so it follows the main loop's batching choice
                // and never forces a B-switch rebuild mid-window.
                const bool ok =
                    batched_cfg ? mm3_dit_run_cfg2(m, &g_mm3_dit, plugins->cm_tmp.data(), nullptr, sig, L,
                                                   pred_c2.data(), pred_u2.data(), &sub_eval_err)
                                : mm3_dit_run(m, &g_mm3_dit, plugins->cm_tmp.data(), nullptr, 1.0f, sig, L,
                                              pred_c2.data(), &sub_eval_err) &&
                                      mm3_dit_run(m, &g_mm3_dit, plugins->cm_tmp.data(), nullptr, 0.0f, sig, L,
                                                  pred_u2.data(), &sub_eval_err);
                if (!ok) {
                    sub_eval_failed = true;
                    return;
                }
                plugins->extra_forwards += 2;
                // Leaves the guided velocity in plugins->tm_v — which IS the
                // vt_buf the solver was handed. Same contract as ACE's
                // evaluate_velocity, momentum update included.
                mm3_plugins_guide(*plugins, pred_c2.data(), pred_u2.data(), cfg_scale, C, L, cur_step, steps, t_val,
                                  cur_dt_ace, v_scratch.data());
            };
        }

        // post_step() evaluates ONE branch at a time and writes it into the
        // buffers the plugin reads as vt_cond / vt_uncond (tm_c / tm_u).
        if (wants_post) {
            eval_cond_fn = [&](const float * xt_ace, float t_val) {
                const float sig = 1.0f - t_val;
                mm3_plug_from_ace_view(xt_ace, plugins->cm_tmp.data(), C, L, /*negate=*/false);
                if (!mm3_dit_run(m, &g_mm3_dit, plugins->cm_tmp.data(), nullptr, 1.0f, sig, L, pred_c2.data(),
                                 &sub_eval_err)) {
                    sub_eval_failed = true;
                    return;
                }
                plugins->extra_forwards += 1;
                mm3_plug_to_ace_view(pred_c2.data(), plugins->tm_c.data(), C, L, /*negate=*/true);
            };
            eval_uncond_fn = [&](const float * xt_ace, float t_val) {
                const float sig = 1.0f - t_val;
                mm3_plug_from_ace_view(xt_ace, plugins->cm_tmp.data(), C, L, /*negate=*/false);
                if (!mm3_dit_run(m, &g_mm3_dit, plugins->cm_tmp.data(), nullptr, 0.0f, sig, L, pred_u2.data(),
                                 &sub_eval_err)) {
                    sub_eval_failed = true;
                    return;
                }
                plugins->extra_forwards += 1;
                mm3_plug_to_ace_view(pred_u2.data(), plugins->tm_u.data(), C, L, /*negate=*/true);
            };
        }
    }

    const auto t_all  = std::chrono::steady_clock::now();
    double     fwd_ms = 0.0, first_ms = 0.0, last_ms = 0.0;

    for (int i = 0; i < steps; i++) {
        const float t = timesteps[(size_t) i];

        // ── the overlap blend (the new logic) ──
        // x[:, :ov] = (1 - (1-1e-6)*t) * noise_prompt + t * prev_latent.
        // The (1 - 1e-6) is the reference's, verbatim: at t = 1 it leaves a
        // 1e-6 sliver of noise rather than a pure copy.
        if (overlap > 0) {
            const float a = 1.0f - (1.0f - 1e-6f) * t;
            for (int64_t c = 0; c < C; c++) {
                const float * np = noise + c * L;
                const float * pl = prev_latent + c * prev_stride;
                float *       x  = out_latents.data() + c * L;
                for (int64_t j = 0; j < overlap; j++) {
                    x[j] = a * np[j] + t * pl[j];
                }
            }
        }

        const auto t0 = std::chrono::steady_clock::now();

        // The condition is uploaded on the first pass only; every later pass
        // reuses the resident copy and toggles the gate. Batched: one compute
        // returns both branches (batch 0 gated 1.0 = conditional, batch 1
        // gated 0.0 == zeros_like(condition)).
        if (batched_cfg) {
            if (!mm3_dit_run_cfg2(m, &g_mm3_dit, out_latents.data(), i == 0 ? cond : nullptr, t, L, pred_c.data(),
                                  pred_u.data(), err)) {
                return false;
            }
            n_uncond++;
        } else {
            if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), i == 0 ? cond : nullptr, 1.0f, t, L, pred_c.data(),
                             err)) {
                return false;
            }
            // Warmup, final step, and every Nth step evaluate the real
            // unconditional branch and refresh the delta; the rest rebuild it.
            const bool eval_uncond = !delta_cache || i < 2 || i == steps - 1 || (i % uncond_every) == 0;
            if (eval_uncond) {
                if (!mm3_dit_run(m, &g_mm3_dit, out_latents.data(), nullptr, 0.0f, t, L, pred_u.data(), err)) {
                    return false;
                }
                n_uncond++;
                if (delta_cache) {
                    for (int64_t j = 0; j < N; j++) {
                        cfg_delta[(size_t) j] = pred_c[(size_t) j] - pred_u[(size_t) j];
                    }
                }
            } else {
                for (int64_t j = 0; j < N; j++) {
                    pred_u[(size_t) j] = pred_c[(size_t) j] - cfg_delta[(size_t) j];
                }
            }
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwd_ms += ms;
        if (i == 0) {
            first_ms = ms;
        }
        last_ms = ms;

        const float dsigma = sigmas[(size_t) i + 1] - sigmas[(size_t) i];
        if (!use_plugins) {
            for (int64_t j = 0; j < N; j++) {
                const float u = pred_u[(size_t) j];
                const float v = u + cfg_scale * (pred_c[(size_t) j] - u);
                out_latents[(size_t) j] += dsigma * v;
            }
        } else {
            // Everything a plugin sees is in the ACE convention: t descending,
            // velocity sign-flipped, latents time-major. mm3-plugins.h owns the
            // mapping and its proof.
            const float t_ace      = 1.0f - sigmas[(size_t) i];
            const float t_next_ace = 1.0f - sigmas[(size_t) i + 1];
            cur_step               = i;
            cur_dt_ace             = t_ace - t_next_ace;  // == dsigma, exactly

            mm3_plugins_guide(*plugins, pred_c.data(), pred_u.data(), cfg_scale, C, L, i, steps, t_ace, cur_dt_ace,
                              v_guided.data());

            if (plugins->solver && i < steps - 1) {
                // The solver contract reserves the final step for the engine.
                mm3_plugins_solver_step(*plugins, out_latents.data(), C, L, t_ace, t_next_ace, i, model_fn);
            } else {
                // Native Euler, and — at i == steps-1 — the engine-owned final
                // step. Those are the SAME expression here: dsigma is
                // 1 - sigma[steps-1] == t_ace, so `x += dsigma * v_mm3` is
                // ACE's `x0 = x - t_curr * v_ace` with the signs cancelled.
                for (int64_t j = 0; j < N; j++) {
                    out_latents[(size_t) j] += dsigma * v_guided[(size_t) j];
                }
            }

            // MM3 always runs both CFG branches, so the "CFG must be live" half
            // of ACE's gate is unconditionally true; only the final-step
            // exclusion carries over.
            if (plugins->guidance && plugins->guidance->has_post_step && i < steps - 1) {
                mm3_plugins_post_step(*plugins, out_latents.data(), C, L, t_next_ace, i, steps, cur_dt_ace,
                                      eval_cond_fn, eval_uncond_fn);
            }

            // A sub-evaluation cannot report failure through the void-returning
            // solver/post_step callbacks, so it latches and is collected here.
            if (sub_eval_failed) {
                if (err) {
                    *err = sub_eval_err.empty() ? "plugin sub-evaluation failed" : sub_eval_err;
                }
                return false;
            }
        }

        if (on_step) {
            on_step(i + 1, steps);
        }
        // Between Euler steps: the coarsest useful cancel grain in this stage
        // (one step is two DiT forwards, ~1 s at 200 frames). Bailing here
        // leaves out_latents half-denoised, which is fine — the caller
        // discards everything on a cancel.
        if (should_cancel && should_cancel()) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }
    }

    // The hard splice: the blend was a boundary condition for the DiT, this is
    // what actually leaves the window.
    if (overlap > 0) {
        for (int64_t c = 0; c < C; c++) {
            memcpy(out_latents.data() + c * L, prev_latent + c * prev_stride, (size_t) overlap * sizeof(float));
        }
    }

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();
    // Multi-eval solvers and post_step() add forwards that a flat 2/step count
    // would hide, making a Heun run look as cheap as an Euler one.
    const int64_t n_forwards = (int64_t) steps + n_uncond + (use_plugins ? plugins->extra_forwards : 0);
    if (stats) {
        stats->steps         = steps;
        stats->forwards      = (int) n_forwards;
        stats->total_ms      = total_ms;
        stats->forward_ms    = fwd_ms;
        stats->first_ms      = first_ms;
        stats->last_ms       = last_ms;
        stats->compute_bytes = g_mm3_dit.compute_bytes;
    }
    const std::string dc_tag = delta_cache ? (" dc" + std::to_string(uncond_every)) : std::string();
    fprintf(stderr,
            "[MM3-Flow] Window: L=%lld, ov=%lld, %d steps, cfg %.2f%s%s, %lld forwards -> %.0f ms (%.0f ms/step, "
            "%.0f ms/forward)\n",
            (long long) L, (long long) overlap, steps, (double) cfg_scale, batched_cfg ? " (batched)" : "",
            dc_tag.c_str(), (long long) n_forwards, total_ms, total_ms / (double) steps,
            fwd_ms / (double) (n_forwards > 0 ? n_forwards : 1));
    return true;
}

// ── Deterministic per-window noise ──────────────────────────────────────────

static inline uint64_t mm3_splitmix64(uint64_t & s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// N(0,1) via Box-Muller over splitmix64 doubles. Explicitly hand-rolled so the
// bytes are identical on every toolchain (see the header note on noise).
static void mm3_fill_noise(uint64_t seed, int64_t window, std::vector<float> & out, int64_t n) {
    out.assign((size_t) n, 0.0f);
    uint64_t s = seed ^ (0xA24BAED4963EE407ULL * (uint64_t) (window + 1));
    mm3_splitmix64(s);  // discard one, so window 0 does not start on the raw seed
    for (int64_t i = 0; i < n; i += 2) {
        double u1, u2;
        do {
            u1 = (double) (mm3_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 <= 1e-300);
        u2                = (double) (mm3_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
        const double r    = std::sqrt(-2.0 * std::log(u1));
        const double th   = 6.283185307179586476925286766559 * u2;
        out[(size_t) i]   = (float) (r * std::cos(th));
        if (i + 1 < n) {
            out[(size_t) (i + 1)] = (float) (r * std::sin(th));
        }
    }
}

// ── Request / result ────────────────────────────────────────────────────────

struct MM3GenRequest {
    // Either an assembled prompt template (tokenised here) or explicit ids.
    // `ids_cond` wins when both are given; `ids_uncond` may be left empty and
    // is then derived by the 3-token CFG mask rule.
    std::string          prompt;
    std::vector<int32_t> ids_cond;
    std::vector<int32_t> ids_uncond;

    int64_t  max_frames = 300;

    // Semantic-stream sampling knobs (mm3-sample.h); defaults = reference.
    float       lm_temperature = 1.0f;
    int         lm_top_k       = 0;
    float       lm_top_p       = 0.0f;
    float       lm_rep_penalty = 1.0f;
    int         lm_rep_window  = 320;
    std::string lm_rep_mode    = "dry";
    float       lm_dry_base    = 1.75f;
    int         lm_dry_min_len = 15;
    uint64_t seed       = 42;
    int      steps      = 30;
    float    cfg_flow   = 1.7f;

    /** CFG guidance-delta cache (mm3_flow_sample_chunk). 1 = the exact
     *  reference, both branches every step. N >= 2 evaluates the unconditional
     *  branch on the warmup steps, the final step, and every Nth step, holding
     *  the guidance delta in between. N=2 removes ~22% of the flow forwards. */
    int      flow_uncond_interval = 1;

    // Seed for the AR stage ONLY. Unset (the default) ties it to `seed`, which
    // is what MM3 natively does — one seed drives both the code sampling and
    // the per-window flow noise. Splitting them exists for the same reason ACE
    // has `lm_seed`: it lets the flow noise be rerolled without changing the
    // plan, which is what keeps the AR cache (mm3-job.h) alive across a reroll.
    // Two fields rather than a -1 sentinel because a drawn seed uses the full
    // 64-bit range and would go negative through any signed spelling.
    bool     ar_seed_set = false;
    uint64_t ar_seed     = 0;

    // ── AR cache hit (mm3-job.h owns the storage) ──────────────────────────
    // When set, stage 1 is skipped ENTIRELY — no LM, no depth decoder, no
    // per-frame forward passes — and this block is fed straight to the
    // condition encoder in its place. Layout is MM3ArResult::frame_hiddens:
    // [cached_frames, num_codebooks, embedding_length] floats, layer-major.
    // BORROWED and read-only for the whole call; the cache outlives it.
    const float * cached_hiddens = nullptr;
    int64_t       cached_frames  = 0;

    // Lua solver/scheduler/guidance overrides shared with the ACE sampler
    // (mm3-plugins.h). All-empty — the default — runs the parity-proven native
    // path, so a caller that knows nothing about plugins is unaffected.
    MM3PluginSel plugins;

    // Parity mode: replay the reference's own AR codes instead of sampling.
    // Iteration-indexed, entry 0 is the un-emitted iteration (see mm3-ar-loop.h).
    std::vector<int32_t> forced_semantic;  // [I]
    std::vector<int32_t> forced_acoustic;  // [I * 7], flat, iteration-major

    // Runtime LM LoRA. The pointer is BORROWED (the job layer owns the loaded
    // adapter and outlives the generation); nullptr = base model.
    const MM3LmAdapter * lm_adapter = nullptr;
    MM3LmAdapterScales   lm_adapter_scales = {};

    // Parity mode: per-window initial noise, 128*L floats channel-major. Entry
    // k applies to window k; an empty entry (or a short vector) falls back to
    // the seeded derivation for that window, so supplying window 0 alone is
    // legal.
    std::vector<std::vector<float>> forced_noise;

    bool keep_window_latents = false;  // populate MM3GenResult::window_latents

    // Returns true to abort. Polled once per AR frame, once per Euler step, and
    // at every stage boundary. On abort mm3_generate() returns false with
    // *err == MM3_ERR_CANCELLED, which the job worker turns into job status 3
    // (cancelled) rather than 2 (failed).
    /** Emit LRC lyric timestamps from the LM's alignment heads. Costs the
     *  manual attention path on 3 of 36 layers; see mm3-align.h. */
    bool want_lrc = false;
    std::function<bool()> should_cancel;
    /** Called once, after the AR stage and before the condition encoder — the
     *  only point where the LM is finished but the flow stack is not yet
     *  needed. Lets the caller swap residency (drop the LM, load cond/dit/voc)
     *  without this file owning a VRAM policy. Return false (setting *err) to
     *  abort the run. Optional: unset means "everything stays resident". */
    std::function<bool(std::string * err)> after_ar;
    /** Streaming sink (MM3ChunkCb above). Unset — the default — runs the
     *  original serial pipeline: every window denoised, then every window
     *  vocoded, then stitched. Set, the vocoder moves inside the window loop
     *  and each window's final PCM is handed over as soon as it exists. The
     *  callback runs on this thread and must not block for long; mm3-job.h's
     *  implementation is a memcpy into a queue. */
    MM3ChunkCb on_chunk;
    /** Dispatch windows WHILE the AR stage is still planning, instead of after
     *  it finishes. This is what makes streaming worth having on a fresh plan:
     *  serially the first window cannot exist until the planner is done, which
     *  is most of the render.
     *
     *  It requires the LM and the flow stack to be CO-RESIDENT - there is no
     *  mid-run handover to swap them - so the decision belongs to the caller,
     *  the only layer that knows the VRAM situation. mm3-job.h sets it after a
     *  measured check and falls back to the serial path otherwise. Ignored
     *  without `on_chunk`, on an AR cache hit, or alongside `after_ar`. */
    bool       stream_interleave = false;
};

struct MM3GenResult;

// Everything stage 2 carries from one window to the next, for ONE take.
//
// Ensemble renders run K of these side by side: the takes decode in lockstep in
// stage 1, so window k becomes dispatchable for all of them at the same moment,
// and they then advance together instead of one song finishing before the next
// starts. That is the difference between "three streams growing" and "three
// renders queued".
//
// Splitting this out is also what keeps `process_window` the ONLY copy of the
// per-window body — the alternative was a second, take-aware version of it,
// and two pieces of crop arithmetic that merely look alike is exactly the bug
// this file is written to avoid.
struct MM3TakeRun {
    MM3GenResult * out = nullptr;
    // Re-read per window, never cached: while interleaving, the planner is
    // still appending to this take's frame_hiddens between dispatches.
    std::function<const float *()> hiddens;

    std::vector<std::vector<float>> chunk_latents;
    std::vector<std::vector<float>> wavs;
    std::vector<float>              prev_latent;  // [128, 172] channel-major
    int64_t                         prev_len = 0;
    std::vector<float>              prev_cond;    // [172, 2048] row-major
    int64_t                         stream_offset = 0;  // samples/ch already emitted
    int64_t                         windows_done  = 0;  // next window index to dispatch
    MM3PluginRun                    plugin_run;
    // This take's flow-noise seed: req.seed + take index, so take 0 is exactly
    // the noise a single-track render at that seed has always drawn.
    uint64_t                        seed = 0;

    // Settled once the planner stops (per take — an early EOS makes them differ).
    std::vector<int64_t> starts;
    int64_t              F  = 0;
    int64_t              NW = 0;
};

struct MM3GenResult {
    std::vector<float> audio;        // planar stereo: [ch0 T][ch1 T]
    int64_t            n_samples = 0;  // T, per channel
    int                sample_rate = 0;

    int64_t              frames    = 0;
    int64_t              n_windows = 0;
    std::vector<int64_t> chunk_starts;
    std::vector<int64_t> chunk_frames;
    std::vector<int64_t> window_L;
    std::vector<int64_t> window_overlap;
    std::vector<int64_t> forced_noise_used;  // 1 = replayed, 0 = seed-derived

    std::vector<std::vector<float>> window_latents;  // only when keep_window_latents

    // AR stage detail (codes + timings). frame_hiddens are moved in and are
    // ~39 MB at 300 frames — the caller owns them. EMPTY on an AR cache hit:
    // the block was never produced by this run, it was borrowed (see
    // MM3GenRequest::cached_hiddens), so nothing here may be moved out of it.
    MM3ArResult ar;
    /** True when stage 1 was served from MM3GenRequest::cached_hiddens. */
    bool        ar_cached = false;

    double  ar_ms    = 0.0;
    double  cond_ms  = 0.0;
    double  flow_ms  = 0.0;
    double  voc_ms   = 0.0;
    double  total_ms = 0.0;
    int64_t flow_forwards = 0;

    // The vocoder graph does not expose its compute size (mm3-vocoder-graph.h
    // logs it instead), so only the DiT's is reported here.
    size_t dit_compute_bytes = 0;

    bool  has_nan = false;
    float peak    = 0.0f;
    double rms    = 0.0;
};

// ── The assembly ────────────────────────────────────────────────────────────

// Plan, condition, denoise, vocode and stitch K songs from one prompt.
//
// `outs` is an array of `takes` results. Stage 1 decodes all K in a single
// batched AR pass (mm3-ar-loop.h); stage 2 then runs per take, window by
// window, so the takes advance TOGETHER rather than one song completing before
// the next begins. With a streaming sink installed that is K streams growing
// side by side.
//
// takes == 1 is the single-song path, unchanged in every observable way: one
// MM3TakeRun, take 0's seed is `req.seed`, and the AR reduces to the loop it
// always ran. mm3_generate() below is the one-take wrapper every existing
// caller still uses.
//
// `tok` is loaded on demand and only when the request carries a prompt rather
// than explicit ids.
static bool mm3_generate_takes(const MM3Model & m, const MM3GenRequest & req, MM3Tokenizer * tok,
                               const MM3ProgressCb & progress, MM3GenResult * outs, int takes, std::string * err) {
    const auto t_all = std::chrono::steady_clock::now();

    const MM3LmConfig &   lc = m.lm_cfg;
    const MM3CondConfig & cc = m.synth_cfg.cond;
    const MM3VocConfig &  vc = m.synth_cfg.voc;

    const int64_t NCB = (int64_t) lc.num_codebooks - 1;         // 7 acoustic
    const int64_t H   = (int64_t) lc.embedding_length;          // 4096
    const int64_t LAY = (int64_t) lc.num_codebooks;             // 8 mixed layers
    const int64_t CH  = (int64_t) m.synth_cfg.dit.in_channels;  // 128
    const int64_t UP  = (int64_t) vc.total_upsample;            // 512

    // Ensemble takes, clamped to the row budget the AR kernels can serve
    // (mm3-model.h). Clamped rather than rejected, and mm3_ar_plan_takes logs
    // when it bites, so a caller asking for more than the checkpoint allows
    // gets fewer songs rather than an error.
    const int K = mm3_clamp_takes(m.lm_cfg, takes);

    const int64_t win_frames = m.synth_cfg.dit.window_frames ? (int64_t) m.synth_cfg.dit.window_frames : 200;
    const int64_t hop_frames = m.synth_cfg.dit.hop_frames ? (int64_t) m.synth_cfg.dit.hop_frames : 100;

    for (int t = 0; t < K; t++) {
        outs[t]             = MM3GenResult{};
        outs[t].sample_rate = (int) vc.sampling_rate;
    }

    // ── token ids ──
    std::vector<int32_t> ids_cond = req.ids_cond, ids_uncond = req.ids_uncond;
    if (ids_cond.empty()) {
        if (req.prompt.empty()) {
            if (err) {
                *err = "need either a prompt or ids_cond";
            }
            return false;
        }
        if (!tok || !mm3_tokenizer_load(m, tok, err)) {
            return false;
        }
        mm3_tokenizer_encode(*tok, req.prompt, &ids_cond);
    }
    if (ids_cond.size() < 3) {
        if (err) {
            *err = "the prompt must tokenise to at least 3 tokens";
        }
        return false;
    }
    if (ids_uncond.empty()) {
        mm3_tokenizer_uncond(lc, ids_cond, &ids_uncond);
    }
    if (ids_uncond.size() != ids_cond.size()) {
        if (err) {
            *err = "ids_uncond must be the same length as ids_cond";
        }
        return false;
    }

    // ── stage 1: AR plan ──
    MM3ArOptions aopt;
    aopt.max_frames        = req.max_frames;
    aopt.knobs.temperature = req.lm_temperature;
    aopt.knobs.top_k       = req.lm_top_k;
    aopt.knobs.top_p       = req.lm_top_p;
    aopt.knobs.rep_penalty = req.lm_rep_penalty;
    aopt.knobs.rep_window  = req.lm_rep_window;
    aopt.knobs.rep_mode    = mm3_rep_mode_from_name(req.lm_rep_mode);
    aopt.knobs.dry_base    = req.lm_dry_base;
    aopt.knobs.dry_min_len = req.lm_dry_min_len;
    if (aopt.knobs.any_active()) {
        fprintf(stderr,
                "[MM3-AR] sampler knobs: temp %.2f top_k %d top_p %.2f rep %.3f (%s, window %d)" "\n",
                aopt.knobs.temperature, aopt.knobs.top_k, aopt.knobs.top_p, aopt.knobs.rep_penalty,
                req.lm_rep_mode.c_str(), aopt.knobs.rep_window);
    }
    aopt.seed              = req.ar_seed_set ? req.ar_seed : req.seed;
    aopt.collect_hiddens   = true;  // the whole point: the condition encoder eats these
    aopt.lm_adapter        = req.lm_adapter;
    aopt.lm_adapter_scales = req.lm_adapter_scales;
    if (!req.forced_semantic.empty()) {
        if ((int64_t) req.forced_acoustic.size() != (int64_t) req.forced_semantic.size() * NCB) {
            if (err) {
                *err = "forced_acoustic must be forced_semantic.size() * 7 entries";
            }
            return false;
        }
        aopt.forced_semantic = req.forced_semantic.data();
        aopt.forced_acoustic = req.forced_acoustic.data();
        aopt.forced_len      = (int64_t) req.forced_semantic.size();
    }
    aopt.should_cancel = req.should_cancel;

    // Stage-boundary cancel check. Returns true when the caller has bailed out,
    // having already written MM3_ERR_CANCELLED into *err.
    auto bail = [&]() -> bool {
        if (req.should_cancel && req.should_cancel()) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return true;
        }
        return false;
    };

    aopt.want_lrc = req.want_lrc;
    aopt.tok      = tok;


    // ------------------------------------------------------------------------
    // Stage 2, hoisted ABOVE the AR call so the planner can drive it.
    //
    // The per-window body lives in `process_window` and is the ONLY copy of it.
    // Two callers reach it - the AR loop's on_frame_ready hook (interleaved)
    // and the sweep after the AR returns (serial, and the tail of an
    // interleaved run) - and they must produce identical numbers, so they must
    // not be two pieces of code that merely look alike.
    // ------------------------------------------------------------------------

    // With a streaming sink installed the vocoder runs INSIDE the window body
    // (see MM3ChunkCb): window k is denoised, vocoded, cropped and handed over
    // before window k+1 is conditioned. Without one, stage 3 vocodes everything
    // in a second pass, exactly as it always has.
    const bool streaming = (bool) req.on_chunk;

    // INTERLEAVING: dispatch a window while the planner is still planning.
    //
    // This is the difference between "first audio after the whole plan" and
    // "first audio a few seconds in". It needs the LM and the flow stack
    // CO-RESIDENT, which is a VRAM decision the caller owns - mm3-job.h sets
    // `stream_interleave` only when it has checked. Three things disqualify it
    // here regardless:
    //   - no sink: nothing to dispatch to;
    //   - an AR cache hit: there is no AR loop to interleave WITH, and the
    //     whole block is already in hand, so the serial sweep starts emitting
    //     immediately anyway (this is the plan's trap 1);
    //   - an `after_ar` hook: that hook means "the LM is finished, swap
    //     residency", and interleaving has already run stage 2 against a live
    //     LM by the time the AR returns. The two are mutually exclusive by
    //     construction; this is the belt, and it fails SAFE (serial).
    const bool interleave = req.stream_interleave && streaming && !req.cached_hiddens && !req.after_ar;

    // SHARED scratch. Safe to share across takes because process_window is
    // called serially — never two windows in flight at once.
    std::vector<float> cond;
    std::vector<float> noise;
    std::vector<float> stream_buf;  // scratch for the cropped chunk

    // Resolve the plugin selection ONCE per song, not once per window: the
    // lookup logs what will run, and repeating it per window would spam the
    // engine log with the same three lines for every 200-frame chunk. The
    // per-window state reset lives in MM3PluginRun::begin_window instead.
    // Resolved once and copied per take, so K takes do not produce K copies of
    // the same three log lines.
    const MM3PluginRun plugin_proto = mm3_plugins_resolve(req.plugins);

    // THE PLANNER'S OUTPUT LIVES HERE FOR THE WHOLE CALL, and is moved onto
    // `outs` only at the very end.
    //
    // It cannot be written straight into outs[t].ar and it cannot be moved
    // across early: while interleaving, `st.hiddens()` is re-read on every
    // dispatched window WHILE the planner is still appending. Moving the block
    // mid-call leaves the streaming path reading a hollowed-out vector — which
    // is exactly the crash this shape prevents.
    std::vector<MM3ArResult> ars((size_t) K);

    std::vector<MM3TakeRun> tk((size_t) K);
    for (int t = 0; t < K; t++) {
        MM3TakeRun & st = tk[(size_t) t];
        st.out          = &outs[t];
        st.plugin_run   = plugin_proto;
        // Take 0 draws exactly the noise a single-track render at this seed has
        // always drawn; the others are offset so their flow noise is their own.
        st.seed = req.seed + (uint64_t) t;
        // The condition encoder's input, from whichever source. Borrowed on a
        // cache hit; owned by outs[t].ar otherwise. Neither is written to —
        // mm3_cond_encode takes it as `const float *`.
        //
        // RE-READ PER WINDOW, never cached in a local. While interleaving, the
        // AR loop is still appending to frame_hiddens between dispatches.
        // mm3-ar-loop.h reserves the full max_frames block up front so it does
        // not in fact reallocate — but a `const float * const` captured before
        // the planner ran would be a dangling pointer the day that reserve
        // changes, and the failure would be silent garbage audio.
        MM3ArResult * arp    = &ars[(size_t) t];
        const float * cached = (t == 0) ? req.cached_hiddens : nullptr;
        st.hiddens           = [arp, cached]() -> const float * {
            return cached ? cached : arp->frame_hiddens.data();
        };
    }

    // How many windows a full-length plan would produce. Only ever used for
    // progress reporting while the true count is still unknown.
    const int64_t nw_estimate =
        req.max_frames <= win_frames ? 1 : (req.max_frames - hop_frames + hop_frames - 1) / hop_frames;
    int64_t ar_frames_done = 0;
    // Interleaving nests stage 2 INSIDE the AR call, so the wall time of that
    // call is no longer the AR stage's cost - it is AR plus every window
    // dispatched along the way, and ar_ms + flow_ms would then sum to more than
    // the whole run. Measured here and subtracted, so the reported split still
    // means what it has always meant. (Observed before the fix: ar 50.3 s +
    // flow 27.4 s on a 60.2 s render.)
    double dispatch_ms = 0.0;

    // In interleaved mode every stage collapses into ONE line. The planner and
    // the flow stage are now taking turns several times a second, so reporting
    // their own stages would make a caller's progress bar jump between the AR
    // band and the flow band continuously. `window` (audio actually produced)
    // and `step` (frames planned) both travel forward; mm3-job.h maps them.
    MM3ProgressCb window_report = progress;
    if (interleave && progress) {
        window_report = [&](const MM3GenProgress &) {
            // The leading take's window count: they advance in lockstep, so this
            // is how far the render as a whole has got.
            int64_t done = 0;
            for (const MM3TakeRun & s : tk) {
                done = s.windows_done > done ? s.windows_done : done;
            }
            progress(MM3GenProgress{ "stream", done, nw_estimate, ar_frames_done, req.max_frames });
        };
    }

    // Condition, denoise, carry, and (when streaming) vocode + emit ONE window.
    //
    //   k        window index, dispatched strictly in order
    //   cs       first AR frame of the window
    //   wf       frames in the window (200, or shorter for the final one)
    //   is_last  no window follows - decides the right crop, and getting it
    //            wrong truncates the ending
    auto process_window = [&](int t, int64_t k, int64_t cs, int64_t wf, bool is_last, int64_t nw_hint,
                              const MM3ProgressCb & rep) -> bool {
        MM3TakeRun &   st  = tk[(size_t) t];
        MM3GenResult * out = st.out;
        st.chunk_latents.resize((size_t) k + 1);
        st.wavs.resize((size_t) k + 1);
        out->chunk_frames.push_back(wf);

        if (bail()) {
            return false;
        }
        if (rep) {
            rep(MM3GenProgress{ "cond", k, nw_hint, 0, 1 });
        }
        const auto t_c = std::chrono::steady_clock::now();
        int64_t    L   = 0;
        if (!mm3_cond_encode(m, st.hiddens() + (size_t) (cs * LAY * H), wf, cond, &L, err)) {
            return false;
        }
        out->cond_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c).count();
        out->window_L.push_back(L);

        // (3a) the previous window's conditioning wins over the overlap.
        const int64_t overlap = st.prev_len > 0 ? (st.prev_len < L ? st.prev_len : L) : 0;
        if (overlap > 0) {
            memcpy(cond.data(), st.prev_cond.data(), (size_t) (overlap * (int64_t) cc.out_dim) * sizeof(float));
        }
        out->window_overlap.push_back(overlap);

        // noise: replayed or seed-derived.
        const int64_t NN = CH * L;
        bool          forced = false;
        if ((int64_t) req.forced_noise.size() > k && (int64_t) req.forced_noise[(size_t) k].size() == NN) {
            noise  = req.forced_noise[(size_t) k];
            forced = true;
        } else {
            mm3_fill_noise(st.seed, k, noise, NN);
        }
        out->forced_noise_used.push_back(forced ? 1 : 0);

        if (rep) {
            rep(MM3GenProgress{ "flow", k, nw_hint, 0, req.steps });
        }
        MM3FlowStats       fstats;
        std::vector<float> lat;
        auto               on_step = [&](int i, int n) {
            if (rep) {
                rep(MM3GenProgress{ "flow", k, nw_hint, i, n });
            }
        };
        if (!mm3_flow_sample_chunk(m, noise.data(), cond.data(), L, req.steps, req.cfg_flow, overlap,
                                   overlap > 0 ? st.prev_latent.data() : nullptr, st.prev_len, lat, &fstats, on_step,
                                   req.should_cancel, err, &st.plugin_run, req.flow_uncond_interval)) {
            return false;
        }
        out->flow_ms += fstats.total_ms;
        out->flow_forwards += fstats.forwards;
        out->dit_compute_bytes = fstats.compute_bytes;

        // (3) carry: latents[L-344 : L-172] and the matching condition rows.
        const int64_t ostart = L - MM3_CARRY_SPAN_LATENTS > 0 ? L - MM3_CARRY_SPAN_LATENTS : 0;
        const int64_t oend   = L - MM3_OVERLAP_LATENTS > ostart ? L - MM3_OVERLAP_LATENTS : ostart;
        st.prev_len          = oend - ostart;
        st.prev_latent.assign((size_t) (CH * st.prev_len), 0.0f);
        for (int64_t c = 0; c < CH; c++) {
            memcpy(st.prev_latent.data() + c * st.prev_len, lat.data() + c * L + ostart,
                   (size_t) st.prev_len * sizeof(float));
        }
        st.prev_cond.assign((size_t) (st.prev_len * (int64_t) cc.out_dim), 0.0f);
        memcpy(st.prev_cond.data(), cond.data() + ostart * (int64_t) cc.out_dim,
               (size_t) (st.prev_len * (int64_t) cc.out_dim) * sizeof(float));

        st.chunk_latents[(size_t) k] = std::move(lat);

        // -- streaming: vocode + emit this window now --
        // The carry above is everything window k+1 needs; window k's own
        // latents are finished, so vocoding them here costs nothing extra and
        // buys one window of audio per flow pass instead of the whole song at
        // the end.
        if (streaming) {
            if (bail()) {
                return false;
            }
            if (rep) {
                rep(MM3GenProgress{ "vocode", k, nw_hint, 0, 1 });
            }
            const auto t_v1 = std::chrono::steady_clock::now();
            if (!mm3_vocoder_decode(m, st.chunk_latents[(size_t) k].data(), L, st.wavs[(size_t) k], err)) {
                return false;
            }
            out->voc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_v1).count();

            int64_t left = 0, len = 0;
            mm3_window_crop_lr(k == 0, is_last, L, UP, &left, &len);
            stream_buf.assign((size_t) (2 * len), 0.0f);
            for (int ch = 0; ch < 2; ch++) {
                const float * src = st.wavs[(size_t) k].data() + (int64_t) ch * (L * UP) + left;
                float *       dst = stream_buf.data() + (int64_t) ch * len;
                for (int64_t i = 0; i < len; i++) {
                    dst[i] = mm3_clamp_sample(src[i]);
                }
            }
            req.on_chunk(t, k, st.stream_offset, len, stream_buf.data(), out->sample_rate);
            st.stream_offset += len;
        }

        st.windows_done = k + 1;
        return true;
    };

    // The planner's per-frame hook: dispatch every window whose geometry is
    // settled, then hand control straight back so the next frame can be planned.
    //
    // WHY THE `+ 1`. While the planner is running we do not know F, and window k
    // needs a RIGHT CROP iff a window k+1 exists - iff F > cs + win. The planner
    // has not stopped, so F is at least the frames emitted; requiring
    // `frames >= cs + win + 1` makes F > cs + win outright. Dispatching at
    // `cs + win` instead would be wrong in exactly one case, and it is a case
    // that happens: the planner hits EOS on that very frame, F == cs + win,
    // window k turns out to be the LAST one, and it has just been emitted with
    // 258 latents (3.0 s) cropped off its end. One extra frame of latency -
    // 40 ms of audio - buys a crop that can never be wrong.
    // Assigned here rather than beside the other aopt fields because it needs
    // `interleave` and `windows_done`, and because the frame count has to be
    // tracked whether or not anyone is listening to progress.
    aopt.n_takes  = K;
    aopt.on_frame = [&](int64_t f, int64_t total) {
        ar_frames_done = f;
        if (!progress) {
            return;
        }
        if (interleave) {
            progress(MM3GenProgress{ "stream", tk[0].windows_done, nw_estimate, f, total });
        } else {
            progress(MM3GenProgress{ "ar", -1, 0, f, total });
        }
    };

    if (interleave) {
        // Dispatch every window whose geometry has settled, FOR EVERY TAKE.
        //
        // The takes decode in lockstep, so their counts move together and a
        // window becomes dispatchable for all of them in the same iteration —
        // which is what makes K streams grow side by side rather than in
        // series. A take that hit EOS early simply stops qualifying; its
        // `frames[t]` stops rising and the loop below stops finding work for it.
        aopt.on_takes_frame_ready = [&](const int64_t * frames, int n, std::string * e) -> bool {
            for (int t = 0; t < n; t++) {
                MM3TakeRun & st = tk[(size_t) t];
                while (st.windows_done * hop_frames + win_frames + 1 <= frames[t]) {
                    const int64_t k  = st.windows_done;
                    const auto    t0 = std::chrono::steady_clock::now();
                    const bool    ok = process_window(t, k, k * hop_frames, win_frames, /*is_last=*/false,
                                                      nw_estimate, window_report);
                    dispatch_ms +=
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (!ok) {
                        if (e && err && e != err) {
                            *e = *err;
                        }
                        return false;
                    }
                }
            }
            return true;
        };
        fprintf(stderr, "[MM3-Pipe] interleaved streaming: %d take(s) dispatch while the planner is still running\n",
                K);
    }

    if (req.cached_hiddens) {
        // AR cache hit: the caller has last run's frame-hidden block and the
        // AR-affecting inputs have not changed, so there is nothing to compute.
        // The codes and the LRC are NOT reproduced here — the cache holds those
        // too and the job layer copies them back onto the result.
        //
        // The cache holds ONE plan, so a hit is a one-take render by
        // construction; mm3-job.h never offers a cached block for K > 1.
        outs[0].ar_cached = true;
        outs[0].ar_ms     = 0.0;
        ars[0].n_frames   = req.cached_frames;
    } else {
        const auto t_ar = std::chrono::steady_clock::now();
        // One batched pass for all K takes. Every take's result lands in its own
        // MM3ArResult, and take 0 carries the shared stage timings.
        if (!mm3_ar_plan_takes(m, ids_cond.data(), ids_uncond.data(), (int64_t) ids_cond.size(), aopt, ars.data(),
                               err)) {
            return false;
        }
        double ar_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ar).count();
        // See `dispatch_ms`: on an interleaved run the windows rendered inside
        // this call are not planning time.
        ar_ms -= dispatch_ms;
        if (ar_ms < 0.0) {
            ar_ms = 0.0;
        }
        // The planning cost is ONE batched pass, so it is reported whole on
        // every take rather than divided — dividing would imply each take could
        // have been had for that price, which is the opposite of the point.
        for (int t = 0; t < K; t++) {
            outs[t].ar_ms = ar_ms;
        }
    }
    for (int t = 0; t < K; t++) {
        outs[t].frames = ars[(size_t) t].n_frames;
    }

    // -- window plan (fact 1), now that each take's F is known --
    //
    // PER TAKE, because an early EOS makes the frame counts differ: take 1 can
    // be a full-length song while take 2 stopped at 40 s, and they then have
    // different window counts and different final-window geometry.
    for (int t = 0; t < K; t++) {
        MM3TakeRun &   st  = tk[(size_t) t];
        MM3GenResult * out = st.out;
        const int64_t  F   = ars[(size_t) t].n_frames;
        if (F <= 0) {
            if (err) {
                *err = req.cached_hiddens ? "the cached AR block claims zero frames"
                                          : "the AR stage emitted zero frames";
                if (K > 1) {
                    *err += " (take " + std::to_string(t) + ")";
                }
            }
            return false;
        }
        if (!(t == 0 && req.cached_hiddens) && (int64_t) ars[(size_t) t].frame_hiddens.size() != F * LAY * H) {
            if (err) {
                *err = "the AR stage returned a frame-hidden block of the wrong size";
                if (K > 1) {
                    *err += " (take " + std::to_string(t) + ")";
                }
            }
            return false;
        }

        if (F <= win_frames) {
            st.starts.push_back(0);
        } else {
            for (int64_t s = 0; s < F - hop_frames; s += hop_frames) {
                st.starts.push_back(s);
            }
        }
        st.F              = F;
        st.NW             = (int64_t) st.starts.size();
        out->n_windows    = st.NW;
        out->chunk_starts = st.starts;

        // Anything already dispatched must agree with the plan that just
        // settled. It does by construction (see the `+ 1` note above), but a
        // mismatch here is a mis-cropped or mis-placed window, and the only
        // instrument that would notice is an ear — so it is checked rather than
        // trusted.
        if (st.windows_done > st.NW) {
            if (err) {
                *err = "streaming dispatched more windows (" + std::to_string(st.windows_done) +
                       ") than the plan holds (" + std::to_string(st.NW) + ")";
            }
            return false;
        }
        for (int64_t k = 0; k < st.windows_done; k++) {
            if (st.starts[(size_t) k] != k * hop_frames || out->chunk_frames[(size_t) k] != win_frames ||
                k == st.NW - 1) {
                if (err) {
                    *err = "a streamed window disagrees with the settled window plan at index " + std::to_string(k);
                }
                return false;
            }
        }
    }

    // -- stage-1 -> stage-2 residency handover --
    //
    // The AR stage is finished with the LM and the depth decoder: everything
    // stage 2 needs from them is `out->ar.frame_hiddens`, which is a host-side
    // block. This is the one point where the caller can drop the 8.59B LM and
    // bring in the flow stack, so the two never have to be resident together.
    //
    // Policy lives in the caller (mm3-job.h), not here - this file stays a pure
    // compute path. An absent hook means "keep everything resident", which is
    // exactly what the bring-up endpoints want.
    //
    // On an AR cache hit there is nothing to hand over - the LM was never
    // loaded - so mm3-job.h simply does not install the hook and brings the flow
    // stack in up front. The call still happens if a caller set one anyway:
    // "prepare for stage 2" is the honest reading of the contract. An
    // interleaved run never has one (it would contradict co-residency), and
    // `interleave` above is disabled if one is somehow set.
    if (req.after_ar && !req.after_ar(err)) {
        return false;
    }

    // -- stage 2: the windows the planner did not already dispatch --
    // For a serial run that is all of them; for an interleaved run it is the
    // tail - the short final window, and whichever full windows the planner
    // outran. Same function either way, which is the whole point of it being a
    // function.
    // Round-robin over takes rather than take-by-take: on a serial (non
    // interleaved) run this is what keeps K streams growing together instead of
    // the first song finishing before the second starts. Takes that have run
    // out of windows are simply skipped.
    {
        int64_t max_nw = 0;
        for (int t = 0; t < K; t++) {
            max_nw = tk[(size_t) t].NW > max_nw ? tk[(size_t) t].NW : max_nw;
        }
        for (int64_t k = 0; k < max_nw; k++) {
            for (int t = 0; t < K; t++) {
                MM3TakeRun & st = tk[(size_t) t];
                if (k < st.windows_done || k >= st.NW) {
                    continue;
                }
                const int64_t cs = st.starts[(size_t) k];
                const int64_t ce = (cs + win_frames < st.F) ? cs + win_frames : st.F;
                if (!process_window(t, k, cs, ce - cs, /*is_last=*/k == st.NW - 1, st.NW, progress)) {
                    return false;
                }
            }
        }
    }

    // ── stage 3: vocode + stitch (fact 4), per take ──
    // Already done, window by window, when a streaming sink is installed.
    for (int t = 0; t < K; t++) {
        MM3TakeRun &   st  = tk[(size_t) t];
        MM3GenResult * out = st.out;
        const int64_t  NW  = st.NW;

        if (!streaming) {
            const auto t_v = std::chrono::steady_clock::now();
            for (int64_t k = 0; k < NW; k++) {
                if (bail()) {
                    return false;
                }
                if (progress) {
                    progress(MM3GenProgress{ "vocode", k, NW, 0, 1 });
                }
                if (!mm3_vocoder_decode(m, st.chunk_latents[(size_t) k].data(), out->window_L[(size_t) k],
                                        st.wavs[(size_t) k], err)) {
                    return false;
                }
            }
            out->voc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_v).count();
        }

        if (progress) {
            progress(MM3GenProgress{ "stitch", -1, NW, 0, 1 });
        }

        std::vector<int64_t> keep_left((size_t) NW), keep_len((size_t) NW);
        int64_t              total = 0;
        for (int64_t k = 0; k < NW; k++) {
            mm3_window_crop(k, NW, out->window_L[(size_t) k], UP, &keep_left[(size_t) k], &keep_len[(size_t) k]);
            total += keep_len[(size_t) k];
        }

        out->audio.assign((size_t) (2 * total), 0.0f);
        out->n_samples = total;
        int64_t at = 0;
        for (int64_t k = 0; k < NW; k++) {
            const int64_t T = out->window_L[(size_t) k] * UP;
            for (int ch = 0; ch < 2; ch++) {
                memcpy(out->audio.data() + (int64_t) ch * total + at,
                       st.wavs[(size_t) k].data() + (int64_t) ch * T + keep_left[(size_t) k],
                       (size_t) keep_len[(size_t) k] * sizeof(float));
            }
            at += keep_len[(size_t) k];
        }

        double sum_sq = 0.0;
        for (float & v : out->audio) {
            if (std::isnan(v) || std::isinf(v)) {
                out->has_nan = true;
            }
            // Same expression the streaming sink applied to its copy — that
            // identity is what makes the streamed concatenation bit-identical.
            v = mm3_clamp_sample(v);
            const float a = std::fabs(v);
            if (a > out->peak) {
                out->peak = a;
            }
            sum_sq += (double) v * (double) v;
        }
        out->rms = out->audio.empty() ? 0.0 : std::sqrt(sum_sq / (double) out->audio.size());

        if (req.keep_window_latents) {
            out->window_latents = std::move(st.chunk_latents);
        }

        out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();

        fprintf(stderr,
                "[MM3-Pipe] %s%lld frames -> %lld window(s) -> %lld samples/ch (%.2fs @ %d Hz) | "
                "AR %.0f ms%s, cond %.0f ms, flow %.0f ms, voc %.0f ms, total %.0f ms\n",
                K > 1 ? ("take " + std::to_string(t) + ": ").c_str() : "", (long long) st.F, (long long) NW,
                (long long) total, (double) total / (double) (out->sample_rate > 0 ? out->sample_rate : 1),
                out->sample_rate, out->ar_ms, out->ar_cached ? " (cached)" : "", out->cond_ms, out->flow_ms,
                out->voc_ms, out->total_ms);
    }

    // Safe now: every window is rendered, so nothing reads `ars` again.
    for (int t = 0; t < K; t++) {
        outs[t].ar = std::move(ars[(size_t) t]);
    }

    if (progress) {
        progress(MM3GenProgress{ "done", -1, tk[0].NW, tk[0].NW, tk[0].NW });
    }
    if (K > 1) {
        fprintf(stderr, "[MM3-Pipe] %d takes from one prompt in %.0f ms (one batched plan + %d flow passes)\n", K,
                outs[0].total_ms, K);
    }
    return true;
}

// Render ONE song — the shape every existing caller was written against.
static bool mm3_generate(const MM3Model & m, const MM3GenRequest & req, MM3Tokenizer * tok,
                         const MM3ProgressCb & progress, MM3GenResult * out, std::string * err) {
    return mm3_generate_takes(m, req, tok, progress, out, 1, err);
}
