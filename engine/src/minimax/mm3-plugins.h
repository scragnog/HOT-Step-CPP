#pragma once
// minimax/mm3-plugins.h — the ACE ⇄ MM3 sampler-plugin bridge.
//
// HOT-Step file (does not exist upstream). Lets the SAME Lua solver, scheduler
// and guidance plugins that drive the ACE-Step 1.5 DiT drive MiniMax-Music3's
// flow DiT, unmodified. Nothing in engine/plugins/ or plugins/ knows this file
// exists, and nothing in it needs to.
//
// This is a pure adapter: no ggml, no model, no GPU. It converts between the
// two pipelines' conventions and dispatches into lua-plugin.h. The forward
// passes and the window/overlap machinery stay in mm3-pipeline.h, which builds
// the model callback and calls the helpers here.
//
// ── WHY THIS IS POSSIBLE AT ALL ─────────────────────────────────────────────
//
// The plugin layer (lua-plugin.h / lua-plugin-registry.h) never had an ACE
// dependency. Every dispatch entry point takes raw `float *`, element counts
// and a param map. What was ACE-specific was the SAMPLER (hot-step-sampler.h),
// not the plugins. So no plugin needed changing and no plugin API was widened;
// MM3 simply grew its own call sites.
//
// ── CONVENTION 1: TIME RUNS THE OTHER WAY ───────────────────────────────────
//
//   ACE  t descends 1 -> 0.  Solver contract: `xt -= vt * (t_curr - t_prev)`,
//        with dt = t_curr - t_prev > 0. Final step is the engine's:
//        `x0 = xt - vt * t_curr`.
//   MM3  sigma ASCENDS 0 -> 1 (FlowMatchEulerDiscreteScheduler with
//        invert_sigmas=true). Update: `x += (sigma[i+1] - sigma[i]) * v`.
//
// Substitute  sigma = 1 - t  and  v_ace = -v_mm3:
//
//     t_curr = 1 - sigma[i],  t_prev = 1 - sigma[i+1]
//     dt     = t_curr - t_prev = sigma[i+1] - sigma[i] = dsigma   (positive)
//     xt -= v_ace * dt  =  xt - (-v_mm3) * dsigma  =  xt + v_mm3 * dsigma   ✓
//
// which is MM3's update exactly. The terminal step agrees too, and that is the
// part that makes this a real mapping rather than a coincidence: MM3's last
// increment is `(sigma[steps] - sigma[steps-1]) * v` with sigma[steps] == 1, so
//
//     x += (1 - sigma[steps-1]) * v_mm3  =  x - t_curr * v_ace
//
// which is ACE's engine-owned x0 formula, character for character. So MM3's
// `sigmas` array of steps+1 entries with a trailing 1.0 IS ACE's "N scheduler
// timesteps plus an engine-owned final step to t=0". The two samplers have the
// same shape; only the parameterisation was flipped.
//
// Consequence: plugins are handed t and v in the ACE convention they were
// written against, ALWAYS. The flip happens here, at the boundary, once.
//
// ── CONVENTION 2: THE LATENTS ARE TRANSPOSED ────────────────────────────────
//
//   ACE  memory is TIME-MAJOR [T][Oc]  (ggml ne = [Oc, T]) — see the indexing
//        `xt[b*n_per + t*Oc + c]` in hot-step-sampler.h.
//   MM3  latents are CHANNEL-MAJOR [C=128][L].
//
// This matters and is not cosmetic. apg_forward() normalises and projects PER
// CHANNEL OVER TIME and indexes `[t*Oc + c]` to do it (apg-core.h:55-77). Hand
// it a channel-major buffer with Oc=128, T=L and it groups along neither axis —
// it reinterprets the block as [L][128] and the "channels" it normalises are
// scrambled mixtures. So the bridge transposes into the ACE view before any
// plugin sees a buffer, and back afterwards.
//
// Cost is 4 transposes of C*L floats per step (88k at 689 latents) against two
// 2.4B-parameter forwards. It does not register.
//
// ── WHAT IS OPT-IN, AND WHY THAT MATTERS ────────────────────────────────────
//
// MM3's validation bar is forced-replay parity against the fixtures (per-module
// >= 0.999 corr, full-clip 0.9988). A request that names NO plugin runs the
// existing native arithmetic, expression for expression — mm3-pipeline.h keeps
// the original inner loop and only branches when `MM3PluginRun::active` is set.
// Naming any plugin is a deliberate departure from the parity-proven path.
//
// Note that this holds even for `infer_method=euler`, whose Lua body is the
// same maths: Lua computes in double and rounds to float on store, the native
// loop computes in float throughout, so the two can differ in the last ulp.
// That is expected. Parity is a property of the DEFAULT path.
//
// ── NOT SUPPORTED (yet) ─────────────────────────────────────────────────────
//
//   owns_loop solvers   A full-loop solver drives its own iteration, so the
//                       per-step overlap blend (mm3-pipeline.h) would have to
//                       move into the on_step callback to survive. Detected and
//                       refused here with a warning; the run falls back to the
//                       native Euler update rather than producing a song with
//                       broken window seams.
//   postprocess         Those plugins replace ACE's tiled VAE decode. MM3's
//                       vocoder is a DAC-style x512 stack at 44.1 kHz with its
//                       own layout — a different contract, not a port.
//   legacy sideband     stork_substeps / beat_stability / frequency_damping and
//                       friends are read from g_hotstep_params by the ACE
//                       sampler. MM3 leaves them at their SolverState defaults;
//                       plugins written against declared `params` (the
//                       supported channel) are unaffected.
//
// ── STATE AND SCOPE ─────────────────────────────────────────────────────────
//
// MM3 denoises a song as a sequence of INDEPENDENT 200-frame windows, each a
// fresh N-step flow loop. `MM3PluginRun::begin_window()` is therefore the
// analogue of "one ACE generation": it resets the APG momentum buffer and the
// solver state, and drives step_index back to 0 so stateful plugins self-reset
// on their documented trigger. Momentum leaking across a window boundary would
// smear one window's guidance history onto the next one's first step, right at
// the seam the overlap machinery exists to hide.
//
// Lua states are process-global and shared with the ACE sampler. That is safe
// because both backends serialise on the same FIFO GPU worker — no ACE
// generation can be inside step() while MM3 is.

#include "guidance/apg-core.h"
#include "guidance/guidance-interface.h"
#include "lua-plugin-registry.h"
#include "solvers/solver-interface.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// ── What the request asked for ──────────────────────────────────────────────

// All empty == the native path. Names are looked up in the same PluginRegistry
// the ACE sampler uses, so "the dropdown lists it" and "MM3 can run it" are the
// same statement.
struct MM3PluginSel {
    std::string solver;     // request "infer_method" — the ACE field name, kept
    std::string scheduler;  // request "scheduler"
    std::string guidance;   // request "guidance_mode"

    // MM3's own scheduler is hardcoded shift=1 upstream, so 1.0 reproduces it.
    // Only consulted when a scheduler plugin is named — the native sigma build
    // does not take a shift at all.
    float shift = 1.0f;

    float apg_norm_threshold = 2.5f;

    std::unordered_map<std::string, std::string> params;  // "<plugin>:<key>" -> value

    bool any() const { return !solver.empty() || !scheduler.empty() || !guidance.empty(); }
};

// ── Resolved plugins + per-run scratch ──────────────────────────────────────

struct MM3PluginRun {
    MM3PluginSel sel;

    LuaPlugin * solver    = nullptr;
    LuaPlugin * scheduler = nullptr;
    LuaPlugin * guidance  = nullptr;

    // Guidance named exactly "apg" takes the native C++ path, mirroring
    // hot-step-sampler.h's `use_apg_native`. Editing apg.lua's body has never
    // done anything on ACE and must not start doing something here.
    bool guidance_native_apg = false;

    // True when anything at all is overridden. mm3-pipeline.h branches on this
    // and ONLY on this: false means the original code runs untouched.
    bool active = false;

    // Model context handed to every Lua plugin call so a plugin can branch on
    // the real sample rate / latent geometry instead of assuming ACE's. The
    // defaults here are MM3's fixed ones; mm3_plugins_set_geometry() fills in
    // the two that are read off the loaded model.
    LuaModelContext lua_ctx = [] {
        LuaModelContext c;
        c.native_sr = 44100;   // MM3 renders at 44.1 kHz, not ACE's 48 kHz
        c.model_id  = "mm3";
        return c;
    }();

    // Extra DiT forwards charged by multi-eval solvers and post_step(), counted
    // PER WINDOW (reset by begin_window) and folded into MM3FlowStats::forwards
    // so the log stays honest about NFE instead of reporting a flat 2/step.
    int64_t extra_forwards = 0;

    // ── scratch, all [C*L] ──
    std::vector<float> tm_c;     // pred_cond,   ACE view
    std::vector<float> tm_u;     // pred_uncond, ACE view
    std::vector<float> tm_v;     // guided velocity, ACE view
    std::vector<float> tm_x;     // latents, ACE view
    std::vector<float> tm_vpre;  // velocity snapshot for multi-eval solvers
    std::vector<float> cm_tmp;   // channel-major staging for model_fn

    APGMomentumBuffer mbuf;
    APGWorkspace      apg_ws;
    SolverState       solver_state;

    // Start of one window's flow loop. See the state note in the header block.
    void begin_window(int64_t n) {
        const size_t sn = (size_t) n;
        if (!active) {
            return;
        }
        tm_c.assign(sn, 0.0f);
        tm_u.assign(sn, 0.0f);
        tm_v.assign(sn, 0.0f);
        tm_x.assign(sn, 0.0f);
        tm_vpre.assign(sn, 0.0f);
        cm_tmp.assign(sn, 0.0f);

        extra_forwards = 0;

        mbuf = APGMomentumBuffer();  // fresh: `initialized` back to false
        apg_ws.resize((int) n);

        solver_state            = SolverState{};
        solver_state.step_index = 0;
        solver_state.batch_n    = 1;
        solver_state.n_per      = (int) n;
        // Multi-eval solvers are documented to receive this pre-allocated.
        solver_state.xt_scratch.assign(sn, 0.0f);
    }
};

// ── Layout conversion ───────────────────────────────────────────────────────
//
// cm: channel-major [C][L], index c*L + j      (MM3)
// tm: time-major    [L][C], index j*C + c      (ACE)
//
// `negate` folds the velocity sign flip into the copy that was happening
// anyway, so the convention change costs nothing beyond the transpose.

static inline void mm3_plug_to_ace_view(const float * cm, float * tm, int64_t C, int64_t L, bool negate) {
    if (negate) {
        for (int64_t c = 0; c < C; c++) {
            const float * src = cm + c * L;
            for (int64_t j = 0; j < L; j++) {
                tm[j * C + c] = -src[j];
            }
        }
    } else {
        for (int64_t c = 0; c < C; c++) {
            const float * src = cm + c * L;
            for (int64_t j = 0; j < L; j++) {
                tm[j * C + c] = src[j];
            }
        }
    }
}

static inline void mm3_plug_from_ace_view(const float * tm, float * cm, int64_t C, int64_t L, bool negate) {
    if (negate) {
        for (int64_t c = 0; c < C; c++) {
            float * dst = cm + c * L;
            for (int64_t j = 0; j < L; j++) {
                dst[j] = -tm[j * C + c];
            }
        }
    } else {
        for (int64_t c = 0; c < C; c++) {
            float * dst = cm + c * L;
            for (int64_t j = 0; j < L; j++) {
                dst[j] = tm[j * C + c];
            }
        }
    }
}

// ── Resolution ──────────────────────────────────────────────────────────────

// Look the three names up and report, once per generation, exactly what will
// run. An unknown name is a WARNING plus a fall back to native for that slot —
// never a hard failure. A user who typo'd a solver should still get a song.
static MM3PluginRun mm3_plugins_resolve(const MM3PluginSel & sel) {
    MM3PluginRun run;
    run.sel = sel;
    if (!sel.any()) {
        return run;  // active == false: the native path, untouched
    }

    auto & reg = PluginRegistry::instance();

    if (!sel.solver.empty()) {
        run.solver = reg.solver_lookup(sel.solver.c_str());
        if (!run.solver) {
            fprintf(stderr, "[MM3-Plugins] WARNING: unknown solver '%s', using native Euler\n", sel.solver.c_str());
        } else if (run.solver->owns_loop) {
            // See "NOT SUPPORTED" in the header block: a full-loop solver would
            // bypass the per-step overlap blend and break every window seam.
            fprintf(stderr,
                    "[MM3-Plugins] WARNING: solver '%s' is a full-loop (owns_loop) solver, which MM3 does not "
                    "support yet — using native Euler\n",
                    sel.solver.c_str());
            run.solver = nullptr;
        }
    }

    if (!sel.scheduler.empty()) {
        run.scheduler = reg.scheduler_lookup(sel.scheduler.c_str());
        if (!run.scheduler) {
            fprintf(stderr, "[MM3-Plugins] WARNING: unknown scheduler '%s', using the native sigma schedule\n",
                    sel.scheduler.c_str());
        }
    }

    if (!sel.guidance.empty()) {
        if (sel.guidance == "apg") {
            run.guidance_native_apg = true;
        } else {
            run.guidance = reg.guidance_lookup(sel.guidance.c_str());
            if (!run.guidance) {
                fprintf(stderr, "[MM3-Plugins] WARNING: unknown guidance '%s', using native CFG\n",
                        sel.guidance.c_str());
            }
        }
    }

    run.active = run.solver || run.scheduler || run.guidance || run.guidance_native_apg;
    if (run.active) {
        fprintf(stderr, "[MM3-Plugins] solver=%s scheduler=%s guidance=%s\n",
                run.solver ? run.solver->name.c_str() : "(native euler)",
                run.scheduler ? run.scheduler->name.c_str() : "(native sigmas)",
                run.guidance_native_apg ? "apg (native)" : (run.guidance ? run.guidance->name.c_str() : "(native cfg)"));

        // Cost warnings, once per generation rather than once per window. The
        // flow stage already dominates a long render; these multiply it.
        if (run.solver && run.solver->needs_model) {
            fprintf(stderr,
                    "[MM3-Plugins] NOTE: '%s' is a multi-evaluation solver (%d NFE) — each sub-evaluation is two "
                    "more full DiT forwards. Expect the flow stage to scale roughly with NFE.\n",
                    run.solver->name.c_str(), run.solver->nfe);
        }
        if (run.guidance && run.guidance->has_post_step) {
            fprintf(stderr,
                    "[MM3-Plugins] NOTE: guidance '%s' declares post_step() — extra DiT forwards per step, on top "
                    "of the two CFG passes.\n",
                    run.guidance->name.c_str());
        }
    }
    return run;
}

// ── Schedule ────────────────────────────────────────────────────────────────

// Build MM3's sigma/timestep arrays from an ACE scheduler plugin.
//
// The plugin writes `steps` DESCENDING t values in the ACE convention. We map
// sigma[i] = 1 - t[i] (so sigma ascends) and append the terminal sigma = 1.0,
// which is the engine-owned final step in both parameterisations.
//
// `timesteps` stays equal to `sigmas` because in MM3 they ARE the same array —
// the value fed to the DiT and to the overlap blend is sigma itself.
//
// The caller has already sized both vectors; this only fills them.
// Fill in the parts of the Lua model context that are only known once the
// model is loaded: the flow DiT's channel count, and the flow LATENT rate.
//
// The latent rate is deliberately not the AR frame rate (25). The conditioning
// resampler turns F AR frames into mm3_cond_latent_length(F) latents — 200 ->
// 689, so ~86/s — and it is those latents the plugins actually step over.
// Reporting 25 here would hand plugins a number three and a half times off.
static void mm3_plugins_set_geometry(MM3PluginRun & run, int64_t latent_channels, int latent_fps) {
    run.lua_ctx.latent_channels = (int) latent_channels;
    if (latent_fps > 0) {
        run.lua_ctx.latent_fps = latent_fps;
    }
}

static void mm3_plugins_schedule(MM3PluginRun & run, int steps, std::vector<float> * sigmas,
                                 std::vector<float> * timesteps) {
    std::vector<float> t_ace((size_t) steps, 0.0f);
    lua_call_scheduler(*run.scheduler, t_ace.data(), steps, run.sel.shift, run.sel.params, run.lua_ctx);

    sigmas->assign((size_t) steps + 1, 0.0f);
    timesteps->assign((size_t) steps, 0.0f);
    for (int i = 0; i < steps; i++) {
        // A scheduler that emits out-of-range t would drive the overlap blend
        // outside [0,1] and the DiT outside its trained conditioning. Clamp
        // rather than trust: this is third-party Lua.
        float t = t_ace[(size_t) i];
        if (!(t == t)) {  // NaN
            t = 0.0f;
        }
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        const float s              = 1.0f - t;
        (*sigmas)[(size_t) i]      = s;
        (*timesteps)[(size_t) i]   = s;
    }
    (*sigmas)[(size_t) steps] = 1.0f;

    fprintf(stderr, "[MM3-Plugins] Schedule: %s (%s), shift=%.2f, sigma %.6f..%.6f\n",
            run.scheduler->display_name.c_str(), run.scheduler->name.c_str(), (double) run.sel.shift,
            (double) (*timesteps).front(), (double) (*timesteps).back());
}

// ── Guidance ────────────────────────────────────────────────────────────────

// Combine the conditional and unconditional predictions for one step.
//
//   pred_c / pred_u   MM3 convention, channel-major [C][L]
//   out_v             MM3 convention, channel-major [C][L] — what the Euler
//                     update consumes
//
// When a solver plugin is also active this leaves `run.tm_v` holding the same
// velocity in the ACE view, so the solver dispatch below does not transpose it
// a second time.
static void mm3_plugins_guide(MM3PluginRun & run, const float * pred_c, const float * pred_u, float cfg_scale,
                              int64_t C, int64_t L, int step_idx, int total_steps, float t_ace, float dt_ace,
                              float * out_v) {
    const int64_t N = C * L;

    if (run.guidance || run.guidance_native_apg) {
        mm3_plug_to_ace_view(pred_c, run.tm_c.data(), C, L, /*negate=*/true);
        mm3_plug_to_ace_view(pred_u, run.tm_u.data(), C, L, /*negate=*/true);

        GuidanceCtx ctx;
        ctx.step_idx    = step_idx;
        ctx.total_steps = total_steps;
        ctx.dt          = dt_ace;
        ctx.t_curr      = t_ace;

        if (run.guidance_native_apg) {
            apg_forward(run.tm_c.data(), run.tm_u.data(), cfg_scale, run.mbuf, run.tm_v.data(), (int) C, (int) L,
                        run.sel.apg_norm_threshold, run.apg_ws);
        } else {
            lua_call_guidance(*run.guidance, run.tm_c.data(), run.tm_u.data(), cfg_scale, run.mbuf, run.tm_v.data(),
                              (int) C, (int) L, ctx, run.sel.apg_norm_threshold, run.sel.params, run.lua_ctx);
        }
        mm3_plug_from_ace_view(run.tm_v.data(), out_v, C, L, /*negate=*/true);
    } else {
        // Native CFG, the same expression mm3-pipeline.h has always used.
        for (int64_t j = 0; j < N; j++) {
            const float u = pred_u[(size_t) j];
            out_v[(size_t) j] = u + cfg_scale * (pred_c[(size_t) j] - u);
        }
        if (run.solver) {
            mm3_plug_to_ace_view(out_v, run.tm_v.data(), C, L, /*negate=*/true);
        }
    }
}

// ── Solver ──────────────────────────────────────────────────────────────────

// Advance the latents by one step through a Lua solver.
//
//   xt         MM3 convention, channel-major [C][L] — modified in place
//   model_fn   evaluates the guided velocity at (xt_ace_view, t_ace) and writes
//              it into the buffer handed as arg 7. Built by mm3-pipeline.h,
//              which owns the DiT. May be empty for single-eval solvers.
//
// Only called for steps [0, steps-1). The FINAL step is the engine's in both
// pipelines and stays native — see the header block: MM3's last increment and
// ACE's x0 formula are the same expression.
static void mm3_plugins_solver_step(MM3PluginRun & run, float * xt, int64_t C, int64_t L, float t_curr, float t_next,
                                    int step_idx, SolverModelFn model_fn) {
    const int64_t N = C * L;

    mm3_plug_to_ace_view(xt, run.tm_x.data(), C, L, /*negate=*/false);

    // Buffer separation, exactly as hot-step-sampler.h:1389-1405: a multi-eval
    // solver's model_fn overwrites the vt buffer, so the "original velocity"
    // argument must be a snapshot or solvers that read it after calling
    // model_fn (Heun's corrector, UniPC's history) silently degrade to Euler.
    const float * vt_readonly = run.tm_v.data();
    if (run.solver->needs_model) {
        run.tm_vpre.assign(run.tm_v.begin(), run.tm_v.end());
        vt_readonly = run.tm_vpre.data();
    }

    run.solver_state.step_index = step_idx;
    lua_call_solver_step(*run.solver, run.tm_x.data(), vt_readonly, t_curr, t_next, (int) N, run.solver_state,
                         model_fn, run.tm_v.data(), run.sel.params, run.lua_ctx);

    mm3_plug_from_ace_view(run.tm_x.data(), xt, C, L, /*negate=*/false);
}

// ── post_step ───────────────────────────────────────────────────────────────

// Guidance plugins may define post_step() for extra corrective forwards
// (CFG-MP's manifold projection). MM3 always runs both CFG branches, so ACE's
// "CFG must be live" half of the gate is unconditionally true here; only the
// final-step exclusion carries over. Each eval is a full 2.4B forward, so on
// MM3 — where a window step is already ~1 s at 689 latents — this is expensive
// by construction. mm3_plugins_resolve() warns about it once per generation,
// and the real count lands in the per-window "N forwards" log line.
static void mm3_plugins_post_step(MM3PluginRun & run, float * xt, int64_t C, int64_t L, float t_next, int step_idx,
                                  int total_steps, float dt_ace, PostStepModelFn eval_cond, PostStepModelFn eval_uncond) {
    mm3_plug_to_ace_view(xt, run.tm_x.data(), C, L, /*negate=*/false);

    GuidanceCtx ctx;
    ctx.step_idx    = step_idx;
    ctx.total_steps = total_steps;
    ctx.dt          = dt_ace;
    ctx.t_curr      = t_next;

    lua_call_post_step(*run.guidance, run.tm_x.data(), t_next, (int) (C * L), eval_cond, eval_uncond, run.tm_c.data(),
                       run.tm_u.data(), ctx, run.sel.params, run.lua_ctx);

    mm3_plug_from_ace_view(run.tm_x.data(), xt, C, L, /*negate=*/false);
}
