#pragma once
// dit-sampler.h: DiT sampling loop with modular solver/guidance dispatch
//
// Flow matching sampler with modular solver and guidance mode dispatch.
// Solvers are resolved by name via solvers/solver-registry.h.
// Guidance modes are resolved by name via guidance/guidance-registry.h.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include "dcw.h"
#include "debug.h"
#include "dit-graph.h"
#include "dit.h"
#include "guidance/guidance-registry.h"
#include "hot-step-params.h"
#include "philox.h"
#include "schedulers/scheduler-registry.h"
#include "solvers/solver-registry.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// APG core primitives are now in guidance/apg-core.h
// (included transitively via guidance/guidance-registry.h)

// Flow matching generation loop (batched)
// Runs num_steps euler steps to denoise N latent samples in parallel.
//
// noise:            [N * T * Oc]  N contiguous [T, Oc] noise blocks
// context_latents:  [N * T * ctx_ch]  N contiguous context blocks
// enc_hidden:       [enc_S * H_enc * N]  per-batch encoder outputs (caller-stacked)
// schedule:         array of num_steps timestep values
// output:           [N * T * Oc]  generated latents (caller-allocated)
static int dit_ggml_generate(DiTGGML *           model,
                             const float *       noise,
                             const float *       context_latents,
                             const float *       enc_hidden_data,
                             int                 enc_S,
                             int                 T,
                             int                 N,
                             int                 num_steps,
                             const float *       schedule,
                             float *             output,
                             float               guidance_scale       = 1.0f,
                             const DebugDumper * dbg                  = nullptr,
                             const float *       context_switch       = nullptr,
                             int                 cover_steps          = -1,
                             bool (*cancel)(void *)                   = nullptr,
                             void *          cancel_data              = nullptr,
                             const int *     real_S                   = nullptr,
                             const int *     real_enc_S               = nullptr,
                             const float *   enc_switch               = nullptr,
                             const int *     real_enc_S_switch        = nullptr,
                             // --- upstream params (must match pipeline-synth-ops.cpp call) ---
                             bool            use_sde                  = false,
                             const int64_t * seeds                    = nullptr,
                             bool            use_batch_cfg            = true,
                             // --- HOT-Step extensions (defaulted, not passed by upstream) ---
                             const float *   repaint_src              = nullptr,
                             int             repaint_t0               = 0,
                             int             repaint_t1               = 0,
                             float           repaint_injection_ratio  = 0.5f,
                             int             repaint_crossfade_frames = 0,
                             const char *    solver_name              = "euler",
                             const char *    guidance_mode            = "apg",
                              float           apg_momentum             = 0.75f,
                              float           apg_norm_threshold       = 2.5f,
                              int             stork_substeps           = 10,
                              float           beat_stability           = 0.25f,
                              float           frequency_damping        = 0.4f,
                              float           temporal_smoothing       = 0.13f) {
    // ── HOT-Step sideband override ─────────────────────────────────────
    // The upstream pipeline-synth-ops.cpp calls us with default params.
    // Read the real values from the global set by hot-step-server.cpp.
    solver_name        = g_hotstep_params.solver_name.c_str();
    guidance_mode      = g_hotstep_params.guidance_mode.c_str();
    apg_momentum       = g_hotstep_params.apg_momentum;
    apg_norm_threshold = g_hotstep_params.apg_norm_threshold;
    stork_substeps     = g_hotstep_params.stork_substeps;
    beat_stability     = g_hotstep_params.beat_stability;
    frequency_damping  = g_hotstep_params.frequency_damping;
    temporal_smoothing = g_hotstep_params.temporal_smoothing;

    // ── Custom timesteps override (highest priority) ────────────────────
    // If custom_timesteps CSV is provided, it completely replaces the schedule.
    // This takes priority over BOTH the upstream shifted-linear schedule AND
    // the sideband scheduler override below.
    std::vector<float> custom_ts_schedule;
    bool custom_ts_active = false;
    if (!g_hotstep_params.custom_timesteps.empty()) {
        // Parse CSV floats: "0.97,0.76,...,0.085,0" → vector
        const char * p = g_hotstep_params.custom_timesteps.c_str();
        while (*p) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            char * end = nullptr;
            float v = strtof(p, &end);
            if (end == p) break;  // no parse progress
            custom_ts_schedule.push_back(v);
            p = end;
        }
        // Drop trailing 0 (x0 endpoint) — sampler handles final step separately
        if (!custom_ts_schedule.empty() && custom_ts_schedule.back() == 0.0f) {
            custom_ts_schedule.pop_back();
        }
        if (!custom_ts_schedule.empty()) {
            num_steps = (int) custom_ts_schedule.size();
            schedule = custom_ts_schedule.data();
            custom_ts_active = true;
            fprintf(stderr, "[DiT] Custom timesteps: %d steps (overrides scheduler)\n", num_steps);
        }
    }

    // ── Scheduler override ────────────────────────────────────────────────
    // The upstream ops_build_schedule() creates a shifted-linear schedule.
    // If a custom scheduler is configured (and no custom_timesteps), rebuild here.
    std::vector<float> custom_schedule;
    if (!custom_ts_active && !g_hotstep_params.scheduler.empty()) {
        custom_schedule.resize(num_steps);
        // Extract shift from the existing schedule (shift warp was already applied
        // by upstream, but we need the raw shift for our scheduler functions).
        // We get it from the SynthState via the global — but the sampler doesn't
        // have access to SynthState. Instead, just recompute from schedule[0]:
        //   schedule[0] = shift*1/(1+(shift-1)*1) = 1.0 regardless of shift.
        // So we can't extract shift from the schedule. Use the upstream value.
        // The shift is embedded in the schedule already; our schedulers apply
        // their own shift, so we need the raw shift value. We'll read it from
        // the first schedule value: if schedule[0] != 1.0, it was shifted.
        // Actually, for shifted-linear: t_0 = shift*1/(1+(shift-1)) = 1.0 always.
        // We need to infer shift or just use a reasonable default.
        // PRAGMATIC: pass shift=1.0 and let our scheduler handle it,
        // since the upstream schedule was already shift-warped and we're replacing it entirely.
        // The shift value from ops_resolve_params is in SynthState.shift,
        // but we can back-calculate if schedule was built with shifted-linear:
        //   t_1 = shift * u / (1 + (shift-1)*u)  where u = 1 - 1/N
        //   solving: shift = t1*(1-u) / (u*(1-t1))
        float shift_val = 1.0f; // default
        if (num_steps >= 2 && schedule[0] > 0.0f) {
            float u = 1.0f - 1.0f / (float) num_steps;
            float t1 = schedule[1];
            if (t1 > 0.0f && t1 < 1.0f && u > 0.0f) {
                float denom = u * (1.0f - t1);
                if (denom > 1e-8f) {
                    shift_val = t1 * (1.0f - u) / denom;
                    if (shift_val < 0.5f) shift_val = 1.0f;
                    if (shift_val > 10.0f) shift_val = 3.0f;
                }
            }
        }

        const std::string & ss = g_hotstep_params.scheduler;

        // Composite scheduler dispatch (ported from old ops_build_schedule)
        if (ss.rfind("composite:", 0) == 0) {
            const char * body = ss.c_str() + 10;
            const char * plus = strchr(body, '+');
            if (plus) {
                std::string name_a(body, plus - body);
                const char * after_plus = plus + 1;
                const char * colon1 = strchr(after_plus, ':');
                std::string name_b;
                float crossover = 0.0f, split = 0.5f;
                if (colon1) {
                    name_b = std::string(after_plus, colon1 - after_plus);
                    crossover = (float) atof(colon1 + 1);
                    const char * colon2 = strchr(colon1 + 1, ':');
                    if (colon2) split = (float) atof(colon2 + 1);
                } else {
                    name_b = std::string(after_plus);
                }
                if (crossover < 0.0f) crossover = 0.0f;
                if (crossover > 1.0f) crossover = 1.0f;
                if (split < 0.0f) split = 0.0f;
                if (split > 1.0f) split = 1.0f;

                const SchedulerInfo * sa = scheduler_lookup(name_a.c_str());
                const SchedulerInfo * sb = scheduler_lookup(name_b.c_str());
                if (!sa) sa = scheduler_lookup("linear");
                if (!sb) sb = scheduler_lookup("linear");

                std::vector<float> va(num_steps), vb(num_steps);
                sa->fn(va.data(), num_steps, shift_val);
                sb->fn(vb.data(), num_steps, shift_val);

                float zone_lo = split - crossover * 0.5f;
                float zone_hi = split + crossover * 0.5f;
                for (int i = 0; i < num_steps; i++) {
                    float frac = (float) i / (float) num_steps;
                    float w;
                    if (crossover < 1e-6f || frac <= zone_lo) {
                        w = (frac < split) ? 0.0f : 1.0f;
                    } else if (frac >= zone_hi) {
                        w = 1.0f;
                    } else {
                        w = (frac - zone_lo) / (zone_hi - zone_lo);
                    }
                    custom_schedule[i] = (1.0f - w) * va[i] + w * vb[i];
                }
                for (int i = 1; i < num_steps; i++) {
                    if (custom_schedule[i] > custom_schedule[i-1])
                        custom_schedule[i] = custom_schedule[i-1];
                }
                scheduler_clamp(custom_schedule.data(), num_steps);
                fprintf(stderr, "[DiT] Custom schedule: composite %s+%s (cross=%.2f, split=%.2f), shift=%.2f\n",
                        sa->display_name, sb->display_name, crossover, split, shift_val);
            }
        } else {
            // Standard scheduler lookup
            const SchedulerInfo * sched = scheduler_lookup(ss.c_str());
            if (!sched) {
                fprintf(stderr, "[DiT] WARNING: unknown scheduler '%s', using linear\n", ss.c_str());
                sched = scheduler_lookup("linear");
            }
            sched->fn(custom_schedule.data(), num_steps, shift_val);
            fprintf(stderr, "[DiT] Custom schedule: %s (%s), shift=%.2f\n",
                    sched->display_name, sched->name, shift_val);
        }
        schedule = custom_schedule.data();
    }

    DiTGGMLConfig & c       = model->cfg;
    int             Oc      = c.out_channels;      // 64
    int             ctx_ch  = c.in_channels - Oc;  // 128
    int             in_ch   = c.in_channels;       // 192
    int             S       = T / c.patch_size;
    int             n_per   = T * Oc;              // elements per sample
    int             n_total = N * n_per;           // total output elements

    // CFG batching: pack conditional + unconditional into one graph of size 2*N.
    // Slots [0, N): conditional (real encoder states).
    // Slots [N, 2*N): unconditional (null encoder states).
    // Single forward produces both predictions, halving DiT compute per step.
    // When batch_cfg is false, two separate forwards per step (saves activation memory).
    bool do_cfg    = (guidance_scale > 1.0f) && model->null_condition_emb;
    bool batch_cfg = do_cfg && use_batch_cfg;
    int  N_graph   = batch_cfg ? 2 * N : N;

    if (guidance_scale > 1.0f && !model->null_condition_emb) {
        fprintf(stderr, "[DiT] WARNING: guidance_scale=%.1f but null_condition_emb not found. Disabling CFG.\n",
                guidance_scale);
    }

    fprintf(stderr, "[DiT] Batch N=%d, T=%d, S=%d, enc_S=%d%s\n", N, T, S, enc_S,
            batch_cfg ? ", CFG batched 2N" : (do_cfg ? ", CFG 2-pass" : ""));

    // Graph context (generous fixed allocation, shapes are constant across steps)
    size_t               ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/ctx_buf.data(),
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_tensor * t_input  = NULL;
    struct ggml_tensor * t_output = NULL;
    struct ggml_cgraph * gf       = dit_ggml_build_graph(model, ctx, T, enc_S, N_graph, &t_input, &t_output);

    fprintf(stderr, "[DiT] Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    struct ggml_tensor * t_enc = ggml_graph_get_tensor(gf, "enc_hidden");
    int                  H_enc = (int) t_enc->ne[0];  // encoder hidden size (from condition_embedder)

    // Allocate compute buffers.
    // Critical: reset FIRST (clears old state), THEN force inputs to GPU, THEN alloc.
    // Without GPU forcing, inputs default to CPU where the scheduler aliases their
    // buffers with intermediates. enc_hidden is read at every cross-attn layer (24x),
    // so CPU aliasing corrupts it mid-graph. With N>1 the larger buffers trigger
    // more aggressive aliasing, causing batch sample 1+ to produce noise.
    ggml_backend_sched_reset(model->sched);
    if (model->backend != model->cpu_backend) {
        const char * input_names[] = { "enc_hidden", "input_latents", "t",           "t_r",
                                       "positions",  "sa_mask_sw",    "sa_mask_pad", "ca_mask" };
        for (const char * iname : input_names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, iname);
            if (t) {
                ggml_backend_sched_set_tensor_backend(model->sched, t, model->backend);
            }
        }
    }
    if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
        fprintf(stderr, "[DiT] FATAL: failed to allocate graph\n");
        ggml_free(ctx);
        return -1;
    }

    // Encoder hidden states: re-uploaded per step (scheduler clobbers input buffers).
    // When CFG batched, slots [0,N) hold real encoder states, [N,2N) hold null.
    // t_enc was declared above for backend forcing

    // t_r is set per-step in the loop (= t_curr, same as Python reference)
    struct ggml_tensor * t_tr = ggml_graph_get_tensor(gf, "t_r");

    // Positions: [0, 1, ..., S-1] repeated N_graph times for batch rope indexing
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(S * N_graph);
    for (int b = 0; b < N_graph; b++) {
        for (int i = 0; i < S; i++) {
            pos_data[b * S + i] = i;
        }
    }
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N_graph * sizeof(int32_t));

    // Self-attention masks: per-batch, combines sliding window and padding.
    // GGML flash_attn_ext mask layout: [ne0=KV_len, ne1=Q_len, 1, N_graph]
    // Linear element offset: ki + qi*ne0 + b*ne0*ne1
    //   sa_mask_sw  [S, S, 1, N_graph]: layer_type=0 (sliding window + padding)
    //   sa_mask_pad [S, S, 1, N_graph]: layer_type=1 (full attention, padding only)
    // When real_S is NULL, all positions are real (mask is all 0.0).
    struct ggml_tensor * t_sa_mask_sw  = ggml_graph_get_tensor(gf, "sa_mask_sw");
    struct ggml_tensor * t_sa_mask_pad = ggml_graph_get_tensor(gf, "sa_mask_pad");

    int                   win = c.sliding_window;
    std::vector<uint16_t> sa_sw_data(S * S * N_graph);
    std::vector<uint16_t> sa_pad_data(S * S * N_graph);

    // Fill masks for real samples, then duplicate for uncond slots
    for (int b = 0; b < N; b++) {
        int rs = real_S ? real_S[b] : S;  // real sequence length for this batch element
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < S; ki++) {
                bool real_pos = (qi < rs) && (ki < rs);
                int  dist     = (qi > ki) ? (qi - ki) : (ki - qi);
                bool in_win   = (win <= 0) || (S <= win) || (dist <= win);

                // offset = ki + qi*S + b*S*S  (ne0=S indexed by ki, ne1=S indexed by qi)
                int off = b * S * S + qi * S + ki;

                float sw_val    = (real_pos && in_win) ? 0.0f : -INFINITY;
                sa_sw_data[off] = ggml_fp32_to_fp16(sw_val);

                float pad_val    = real_pos ? 0.0f : -INFINITY;
                sa_pad_data[off] = ggml_fp32_to_fp16(pad_val);
            }
        }
        if (batch_cfg) {
            memcpy(&sa_sw_data[(N + b) * S * S], &sa_sw_data[b * S * S], S * S * sizeof(uint16_t));
            memcpy(&sa_pad_data[(N + b) * S * S], &sa_pad_data[b * S * S], S * S * sizeof(uint16_t));
        }
    }
    ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
    ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N_graph * sizeof(uint16_t));

    // Cross-attention mask: per-batch encoder padding.
    // [ne0=enc_S (KV), ne1=S (Q), 1, N_graph] blocks padding in enc_hidden.
    // Value depends only on ki (encoder position), independent of qi.
    // Linear offset for element (ki, qi, 0, b) = ki + qi*enc_S + b*enc_S*S
    struct ggml_tensor *  t_ca_mask = ggml_graph_get_tensor(gf, "ca_mask");
    std::vector<uint16_t> ca_data(enc_S * S * N_graph);

    for (int b = 0; b < N; b++) {
        int re = real_enc_S ? real_enc_S[b] : enc_S;
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < enc_S; ki++) {
                // offset = ki + qi*enc_S + b*enc_S*S  (ne0=enc_S indexed by ki)
                float v                                  = (ki < re) ? 0.0f : -INFINITY;
                ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
            }
        }
        if (batch_cfg) {
            memcpy(&ca_data[(N + b) * enc_S * S], &ca_data[b * enc_S * S], enc_S * S * sizeof(uint16_t));
        }
    }
    ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N_graph * sizeof(uint16_t));

    std::vector<APGMomentumBuffer> apg_mbufs;
    std::vector<float>             null_enc_buf;

    if (do_cfg) {
        apg_mbufs.reserve(N);
        for (int i = 0; i < N; i++) {
            apg_mbufs.emplace_back((double) -apg_momentum);  // configurable momentum
        }
        fprintf(stderr, "[DiT] CFG enabled: guidance_scale=%.1f, %s, N_graph=%d",
                guidance_scale, batch_cfg ? "batched" : "2-pass", N_graph);
        if (apg_momentum != 0.75f || apg_norm_threshold != 2.5f) {
            fprintf(stderr, ", apg_momentum=%.2f, norm_threshold=%.1f", apg_momentum, apg_norm_threshold);
        }
        fprintf(stderr, "\n");
    }

    // Prepare host buffers (all N real samples contiguous)
    std::vector<float> xt(noise, noise + n_total);
    std::vector<float> vt(n_total);

    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
    }

    // input_buf: [in_ch, T, N_graph]
    // Pre-fill context_latents for slots [0, N). Uncond slots [N, 2N) are duplicated.
    // The xt portion (noisy latent) is updated per step in the loop.
    std::vector<float> input_buf(in_ch * T * N_graph);
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[b * T * in_ch + t * in_ch], &context_latents[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));
        }
        if (batch_cfg) {
            memcpy(&input_buf[(N + b) * T * in_ch], &input_buf[b * T * in_ch], T * in_ch * sizeof(float));
        }
    }

    // enc_buf: [H_enc, enc_S, N_graph]
    // Slots [0, N): real encoder hidden states from caller.
    // Batched CFG: slots [N, 2N) hold null_condition_emb broadcast to [H_enc, enc_S].
    // 2-pass CFG: null_enc_buf is a separate buffer uploaded before the uncond forward.
    std::vector<float> enc_buf(H_enc * enc_S * N_graph);
    memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));
    if (do_cfg) {
        int                emb_n = (int) ggml_nelements(model->null_condition_emb);
        std::vector<float> null_emb(emb_n);

        if (model->null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> bf16_buf(emb_n);
            ggml_backend_tensor_get(model->null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
            for (int i = 0; i < emb_n; i++) {
                uint32_t w = (uint32_t) bf16_buf[i] << 16;
                memcpy(&null_emb[i], &w, 4);
            }
        } else {
            ggml_backend_tensor_get(model->null_condition_emb, null_emb.data(), 0, emb_n * sizeof(float));
        }

        if (dbg && dbg->enabled) {
            debug_dump_1d(dbg, "null_condition_emb", null_emb.data(), emb_n);
        }

        // Broadcast [H_enc] to [H_enc, enc_S] then fill uncond destination
        std::vector<float> null_enc_single(H_enc * enc_S);
        for (int s = 0; s < enc_S; s++) {
            memcpy(&null_enc_single[s * H_enc], null_emb.data(), H_enc * sizeof(float));
        }
        if (dbg && dbg->enabled) {
            debug_dump_2d(dbg, "null_enc_hidden", null_enc_single.data(), enc_S, H_enc);
        }
        if (batch_cfg) {
            // Pack null into graph slots [N, 2N)
            for (int b = 0; b < N; b++) {
                memcpy(enc_buf.data() + (N + b) * enc_S * H_enc, null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        } else {
            // Separate buffer for 2-pass re-upload
            null_enc_buf.resize(H_enc * enc_S * N);
            for (int b = 0; b < N; b++) {
                memcpy(null_enc_buf.data() + b * enc_S * H_enc, null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        }
    }
    ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));

    struct ggml_tensor * t_t = ggml_graph_get_tensor(gf, "t");

    // ── Solver dispatch setup ─────────────────────────────────────────────
    const SolverInfo * solver_info = solver_lookup(solver_name);
    if (!solver_info) {
        fprintf(stderr, "[DiT] ERROR: unknown solver '%s', falling back to euler\n", solver_name);
        solver_info = solver_lookup("euler");
    }
    fprintf(stderr, "[DiT] Solver: %s (%s, %d NFE/step, order %d)\n",
            solver_info->display_name, solver_info->name,
            solver_info->nfe, solver_info->order);

    // Initialize solver state
    SolverState solver_state;
    solver_state.seeds            = seeds;
    solver_state.batch_n          = N;
    solver_state.n_per            = n_per;
    solver_state.xt_scratch.resize(n_total);
    solver_state.stork_substeps   = stork_substeps;
    solver_state.beat_stability   = beat_stability;
    solver_state.frequency_damping = frequency_damping;
    solver_state.temporal_smoothing = temporal_smoothing;

    // ── Guidance mode dispatch setup ─────────────────────────────────────
    const GuidanceInfo * guidance_info = guidance_lookup(guidance_mode);
    if (!guidance_info) {
        fprintf(stderr, "[DiT] ERROR: unknown guidance mode '%s', falling back to apg\n", guidance_mode);
        guidance_info = guidance_lookup("apg");
    }
    fprintf(stderr, "[DiT] Guidance: %s (%s)\n", guidance_info->display_name, guidance_info->name);

    // Per-step guidance context (updated in the main loop, captured by lambda)
    GuidanceCtx g_ctx = {0, num_steps, 0.0f, 0.0f};

    // ── evaluate_velocity lambda ─────────────────────────────────────────────
    // Evaluates the DiT model at an arbitrary (xt_in, t_val) point and writes
    // the CFG-processed velocity into `vt`. This is the "model_fn" that
    // multi-evaluation solvers (RK4, Heun, etc.) need to call at intermediate
    // timesteps.
    //
    // IMPORTANT: captures all mutable state by reference. Modifies:
    //   - t_t, t_tr (GPU timestep tensors)
    //   - input_buf (host staging buffer for xt portion)
    //   - vt, vt_cond, vt_uncond (velocity outputs)
    //   - GPU constant buffers are re-uploaded (scheduler clobbers them)
    auto evaluate_velocity = [&](const float * xt_in, float t_val) {
        // Set timestep
        if (t_t) {
            ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float));
        }
        if (t_tr) {
            ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float));
        }

        // Re-upload constants (scheduler may reuse input buffers as scratch)
        ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N_graph * sizeof(int32_t));
        ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N_graph * sizeof(uint16_t));

        // Pack xt into input tensor (cond + uncond slots)
        for (int b = 0; b < N; b++) {
            for (int t = 0; t < T; t++) {
                memcpy(&input_buf[b * T * in_ch + t * in_ch + ctx_ch], &xt_in[b * n_per + t * Oc], Oc * sizeof(float));
            }
            if (batch_cfg) {
                for (int t = 0; t < T; t++) {
                    memcpy(&input_buf[(N + b) * T * in_ch + t * in_ch + ctx_ch], &xt_in[b * n_per + t * Oc],
                           Oc * sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N_graph * sizeof(float));

        // Forward pass
        ggml_backend_sched_graph_compute(model->sched, gf);

        // Read output and apply CFG/APG
        if (batch_cfg) {
            std::vector<float> full_output(n_per * N_graph);
            ggml_backend_tensor_get(t_output, full_output.data(), 0, n_per * N_graph * sizeof(float));
            memcpy(vt_cond.data(), full_output.data(), n_total * sizeof(float));
            memcpy(vt_uncond.data(), full_output.data() + n_total, n_total * sizeof(float));
            for (int b = 0; b < N; b++) {
                guidance_info->fn(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale,
                                  apg_mbufs[b], vt.data() + b * n_per, Oc, T, g_ctx,
                                  apg_norm_threshold);
            }
        } else if (do_cfg) {
            ggml_backend_tensor_get(t_output, vt_cond.data(), 0, n_total * sizeof(float));
            // Unconditional pass
            ggml_backend_tensor_set(t_enc, null_enc_buf.data(), 0, H_enc * enc_S * N * sizeof(float));
            ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));
            if (t_t) {
                ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float));
            }
            if (t_tr) {
                ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float));
            }
            ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
            ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N * sizeof(uint16_t));
            ggml_backend_sched_graph_compute(model->sched, gf);
            ggml_backend_tensor_get(t_output, vt_uncond.data(), 0, n_total * sizeof(float));
            for (int b = 0; b < N; b++) {
                guidance_info->fn(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale,
                                  apg_mbufs[b], vt.data() + b * n_per, Oc, T, g_ctx,
                                  apg_norm_threshold);
            }
        } else {
            ggml_backend_tensor_get(t_output, vt.data(), 0, n_total * sizeof(float));
        }
    };

    // Flow matching loop
    bool switched_cover = false;
    for (int step = 0; step < num_steps; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[DiT] Cancelled at step %d/%d\n", step, num_steps);
            ggml_free(ctx);
            return -1;
        }
        float t_curr = schedule[step];

        // Cover mode: at cover_steps, swap context to silence and enc_hidden to text2music
        if (context_switch && cover_steps >= 0 && step >= cover_steps && !switched_cover) {
            switched_cover = true;
            for (int b = 0; b < N; b++) {
                for (int t = 0; t < T; t++) {
                    memcpy(&input_buf[b * T * in_ch + t * in_ch], &context_switch[b * T * ctx_ch + t * ctx_ch],
                           ctx_ch * sizeof(float));
                }
                if (batch_cfg) {
                    memcpy(&input_buf[(N + b) * T * in_ch], &input_buf[b * T * in_ch], T * in_ch * sizeof(float));
                }
            }
            if (enc_switch) {
                memcpy(enc_buf.data(), enc_switch, H_enc * enc_S * N * sizeof(float));
                if (real_enc_S_switch) {
                    for (int b = 0; b < N; b++) {
                        int re = real_enc_S_switch[b];
                        for (int qi = 0; qi < S; qi++) {
                            for (int ki = 0; ki < enc_S; ki++) {
                                float v                                  = (ki < re) ? 0.0f : -INFINITY;
                                ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
                            }
                        }
                    }
                }
            }
            fprintf(stderr, "[DiT] Cover: switched to non-cover context at step %d/%d\n", step, num_steps);
        }

        // Update guidance context for this step
        float t_next_for_ctx = (step + 1 < num_steps) ? schedule[step + 1] : 0.0f;
        g_ctx.step_idx   = step;
        g_ctx.t_curr     = t_curr;
        g_ctx.dt         = t_curr - t_next_for_ctx;

        // Evaluate velocity at (xt, t_curr) — first evaluation (k1 for RK4, only eval for Euler)
        evaluate_velocity(xt.data(), t_curr);

        // dump intermediate tensors on step 0 (sample 0 only for batch)
        if (step == 0 && dbg && dbg->enabled) {
            auto dump_named = [&](const char * name) {
                struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                if (t) {
                    int64_t            n0           = t->ne[0];
                    int64_t            n1           = t->ne[1];
                    int64_t            sample_elems = n0 * n1;
                    std::vector<float> buf(sample_elems);
                    ggml_backend_tensor_get(t, buf.data(), 0, sample_elems * sizeof(float));
                    if (n1 <= 1) {
                        debug_dump_1d(dbg, name, buf.data(), (int) n0);
                    } else {
                        debug_dump_2d(dbg, name, buf.data(), (int) n0, (int) n1);
                    }
                }
            };
            dump_named("tproj");
            dump_named("temb");
            dump_named("temb_t");
            dump_named("temb_r");
            dump_named("sinusoidal_t");
            dump_named("sinusoidal_r");
            dump_named("temb_lin1_t");
            dump_named("temb_lin1_r");
            dump_named("hidden_after_proj_in");
            dump_named("proj_in_input");
            dump_named("enc_after_cond_emb");
            dump_named("layer0_sa_input");
            dump_named("layer0_q_after_rope");
            dump_named("layer0_k_after_rope");
            dump_named("layer0_sa_output");
            dump_named("layer0_attn_out");
            dump_named("layer0_after_self_attn");
            dump_named("layer0_after_cross_attn");
            dump_named("hidden_after_layer0");
            dump_named("hidden_after_layer6");
            dump_named("hidden_after_layer12");
            dump_named("hidden_after_layer18");
            char last_layer_name[64];
            snprintf(last_layer_name, sizeof(last_layer_name), "hidden_after_layer%d", c.n_layers - 1);
            dump_named(last_layer_name);
        }

        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_2d(dbg, name, vt.data(), T, Oc);
        }

        // step update (all N samples)
        if (step == num_steps - 1) {
            // final step: predict x0 (same for all solvers)
            for (int i = 0; i < n_total; i++) {
                output[i] = xt[i] - vt[i] * t_curr;
            }
        } else {
            float t_next = schedule[step + 1];

            // ── Cache pre-step state for DCW ─────────────────────────────
            // The predicted clean sample denoised = x - v*t must use the
            // PRE-step latent AND the raw velocity from the first model
            // evaluation (matching upstream Python's xt_before_step and
            // vt_for_denoise).  Multi-eval solvers (RF-Solver, Heun, RK4)
            // overwrite both xt[] and vt[] during their step, so we must
            // snapshot both before the solver runs.
            std::vector<float> xt_pre;
            std::vector<float> vt_pre;
            if (g_hotstep_params.dcw_enabled) {
                xt_pre.assign(xt.begin(), xt.end());
                vt_pre.assign(vt.begin(), vt.end());
            }

            // ── Modular solver dispatch ──────────────────────────────────
            // The solver step function modifies xt[] in-place.
            // Multi-eval solvers call evaluate_velocity() via the model_fn
            // callback; single-eval solvers ignore it.
            solver_state.step_index = step;
            solver_info->step_fn(
                xt.data(), vt.data(), t_curr, t_next, n_total,
                solver_state, evaluate_velocity, vt.data()
            );

            // ── DCW correction (Differential Correction in Wavelet domain) ──
            // Apply frequency-domain correction after the solver step.
            // Scaler follows upstream Eq. 20/21: low = t * scaler,
            // high = (1-t) * scaler. No step-count normalization.
            if (g_hotstep_params.dcw_enabled) {
                // Compute denoised (predicted x0) from PRE-step state:
                //   denoised = xt_before_step - vt_for_denoise * t_curr
                // Both xt and vt may have been overwritten by the solver
                // (e.g. RF-Solver writes v_mid into vt_buf).
                std::vector<float> denoised(n_total);
                for (int i = 0; i < n_total; i++) {
                    denoised[i] = xt_pre[i] - vt_pre[i] * t_curr;
                }

                const std::string & dcw_mode = g_hotstep_params.dcw_mode;

                // Effective scaler: matches upstream Python exactly.
                // low  = t_curr * scaler
                // high = (1 - t_curr) * high_scaler
                float eff_scaler = t_curr * g_hotstep_params.dcw_scaler;

                if (dcw_mode == "pix") {
                    dcw_correct_pix(xt.data(), denoised.data(), eff_scaler, n_total);
                } else if (dcw_mode == "low") {
                    dcw_correct_low(xt.data(), denoised.data(), eff_scaler, T, Oc, N);
                } else if (dcw_mode == "high") {
                    dcw_correct_high(xt.data(), denoised.data(), eff_scaler, T, Oc, N);
                } else if (dcw_mode == "double") {
                    float eff_high = (1.0f - t_curr) * g_hotstep_params.dcw_high_scaler;
                    dcw_correct_double(xt.data(), denoised.data(), eff_scaler, eff_high, T, Oc, N);
                }

                if (step == 0) {
                    fprintf(stderr, "[DCW] mode=%s scaler=%.4f high_scaler=%.4f (eff_low=%.4f eff_high=%.4f at t=%.3f)\n",
                            dcw_mode.c_str(), g_hotstep_params.dcw_scaler, g_hotstep_params.dcw_high_scaler,
                            eff_scaler, (1.0f - t_curr) * g_hotstep_params.dcw_high_scaler, t_curr);
                }
            }

            // repaint injection: replace preserved regions with noised source.
            if (repaint_src && repaint_t1 > repaint_t0) {
                int injection_cutoff = (int) (repaint_injection_ratio * (float) num_steps + 0.5f);
                if (step < injection_cutoff) {
                    for (int b = 0; b < N; b++) {
                        for (int t = 0; t < T; t++) {
                            if (t < repaint_t0 || t >= repaint_t1) {
                                for (int ch = 0; ch < Oc; ch++) {
                                    int idx = b * n_per + t * Oc + ch;
                                    xt[idx] = t_next * noise[idx] + (1.0f - t_next) * repaint_src[t * Oc + ch];
                                }
                            }
                        }
                    }
                }
            }
        }

        // debug dump (sample 0 only)
        if (dbg && dbg->enabled) {
            char name[64];
            if (step == num_steps - 1) {
                snprintf(name, sizeof(name), "dit_x0");
                debug_dump_2d(dbg, name, output, T, Oc);
            } else {
                snprintf(name, sizeof(name), "dit_step%d_xt", step);
                debug_dump_2d(dbg, name, xt.data(), T, Oc);
            }
        }

        fprintf(stderr, "[DiT] Step %d/%d t=%.3f [%s]\n", step + 1, num_steps, t_curr,
                solver_info->display_name);
    }

    // Boundary blend: smooth repaint zone edges in latent space.
    // Python: apply_repaint_boundary_blend(x_gen, clean_src, repaint_mask, crossfade_frames)
    // soft_mask[t] = 1.0 inside [t0,t1), ramp at edges, 0.0 outside.
    // x_gen[t] = soft_mask[t]*x_gen[t] + (1-soft_mask[t])*repaint_src[t]
    if (repaint_src && repaint_t1 > repaint_t0 && repaint_crossfade_frames > 0) {
        int cf         = repaint_crossfade_frames;
        int fade_start = repaint_t0 - cf > 0 ? repaint_t0 - cf : 0;
        int fade_end   = repaint_t1 + cf < T ? repaint_t1 + cf : T;
        for (int t = fade_start; t < fade_end; t++) {
            if (t >= repaint_t0 && t < repaint_t1) {
                continue;  // inside zone: keep generated output unchanged
            }
            float m;
            if (t < repaint_t0) {
                // left ramp: [fade_start, t0) -> 0..1 excluding endpoints
                int rl = repaint_t0 - fade_start;
                m      = (float) (t - fade_start + 1) / (float) (rl + 1);
            } else {
                // right ramp: [t1, fade_end) -> 1..0 excluding endpoints
                int rl = fade_end - repaint_t1;
                m      = (float) (fade_end - t) / (float) (rl + 1);
            }
            for (int b = 0; b < N; b++) {
                for (int ch = 0; ch < Oc; ch++) {
                    int idx     = b * n_per + t * Oc + ch;
                    output[idx] = m * output[idx] + (1.0f - m) * repaint_src[t * Oc + ch];
                }
            }
        }
    }

    // Batch diagnostic: report per-sample stats to catch corruption
    if (N >= 2) {
        for (int b = 0; b < N; b++) {
            const float * s  = output + b * n_per;
            float         mn = s[0], mx = s[0], sum = 0.0f;
            int           n_nan = 0;
            for (int i = 0; i < n_per; i++) {
                float v = s[i];
                if (v != v) {
                    n_nan++;
                    continue;
                }
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
                sum += v;
            }
            fprintf(stderr, "[DiT] Batch%d output: min=%.4f max=%.4f mean=%.6f nan=%d\n", b, mn, mx,
                    sum / (float) n_per, n_nan);
        }
    }

    // ── Latent post-processing (sideband) ─────────────────────────────────
    // Apply rescale + shift to output latents before VAE decode.
    // Formula: output[i] = output[i] * latent_rescale + latent_shift
    // No-op at defaults (rescale=1.0, shift=0.0).
    {
        float ls = g_hotstep_params.latent_shift;
        float lr = g_hotstep_params.latent_rescale;
        if (lr != 1.0f || ls != 0.0f) {
            for (int i = 0; i < n_total; i++) {
                output[i] = output[i] * lr + ls;
            }
            fprintf(stderr, "[DiT] Latent post-processing: rescale=%.3f shift=%.3f\n", lr, ls);
        }
    }

    ggml_free(ctx);
    return 0;
}
