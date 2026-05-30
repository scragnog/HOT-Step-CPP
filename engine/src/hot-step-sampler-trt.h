#pragma once
// hot-step-sampler-trt.h: TRT-accelerated DiT sampling loop
//
// Mirrors the GGML sampler (hot-step-sampler.h) but uses dit_trt_forward()
// for the neural network evaluation. All solver/guidance/DCW/repaint logic
// is identical — only the model forward pass changes.
//
// Data layout: host FP32 [B, T, C] <-> GPU BF16 [B, T, C] for TRT
// Masks, RoPE, position IDs are computed inside the ONNX graph (not here).

#ifdef HOT_STEP_TRT

#include "dcw.h"
#include "debug.h"
#include "dit-trt.h"
#include "guidance/apg-core.h"
#include "hot-step-params.h"
#include "lua-plugin-registry.h"
#include "philox.h"
#include "sampler-dcw.h"
#include "sampler-repaint.h"
#include "sampler-schedule.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ── BF16 <-> FP32 conversion (host) ──────────────────────────────────────
// BF16 is the top 16 bits of FP32: same 8-bit exponent, 7-bit mantissa
// (vs FP32's 23-bit mantissa). Conversion is a simple truncation/zero-fill.

static inline uint16_t fp32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    // Round to nearest even (add 0x7FFF + bit 16 for tie-breaking)
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_fp32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float result;
    memcpy(&result, &u, 4);
    return result;
}

static void convert_fp32_to_bf16(const float* src, uint16_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = fp32_to_bf16(src[i]);
    }
}

static void convert_bf16_to_fp32(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = bf16_to_fp32(src[i]);
    }
}

// TRT-accelerated DiT sampling loop
// Same interface as dit_ggml_generate() but uses DitTrt instead of DiTGGML
static int dit_trt_generate(DitTrt *              trt,
                             const float *         noise,              // [N * T * 64] FP32
                             const float *         context_latents,    // [N * T * 128] FP32
                             const float *         enc_hidden_data,    // [N * enc_S * 2048] FP32
                             int                   enc_S,
                             int                   T,
                             int                   N,
                             int                   num_steps,
                             const float *         schedule,
                             float *               output,             // [N * T * 64] FP32
                             float                 guidance_scale      = 1.0f,
                             const DebugDumper *    dbg                = nullptr,
                             const float *         context_switch      = nullptr,
                             int                   cover_steps         = -1,
                             bool (*cancel)(void *)                    = nullptr,
                             void *                cancel_data         = nullptr,
                             const int *           real_S              = nullptr,
                             const int *           real_enc_S          = nullptr,
                             const float *         enc_switch          = nullptr,
                             const int *           real_enc_S_switch   = nullptr,
                             bool                  use_sde             = false,
                             const int64_t *       seeds               = nullptr,
                             bool                  use_batch_cfg       = true,
                             const float *         neg_enc_data        = nullptr,
                             const char *          solver_name         = "euler",
                             const char *          guidance_mode       = "apg",
                             float                 apg_momentum        = 0.75f,
                             float                 apg_norm_threshold  = 2.5f,
                             int                   stork_substeps      = 10,
                             float                 beat_stability      = 0.25f,
                             float                 frequency_damping   = 0.4f,
                             float                 temporal_smoothing  = 0.13f) {
    // ── HOT-Step sideband override ──────────────────────────────────────
    solver_name        = g_hotstep_params.solver_name.c_str();
    guidance_mode      = g_hotstep_params.guidance_mode.c_str();
    apg_momentum       = g_hotstep_params.apg_momentum;
    apg_norm_threshold = g_hotstep_params.apg_norm_threshold;
    stork_substeps     = g_hotstep_params.stork_substeps;
    beat_stability     = g_hotstep_params.beat_stability;
    frequency_damping  = g_hotstep_params.frequency_damping;
    temporal_smoothing = g_hotstep_params.temporal_smoothing;

    // ── Custom timesteps / scheduler override ───────────────────────────
    std::vector<float> custom_ts_schedule;
    bool custom_ts_active = sampler_parse_custom_timesteps(custom_ts_schedule, num_steps);
    if (custom_ts_active) {
        schedule = custom_ts_schedule.data();
    }
    std::vector<float> custom_schedule;
    if (!custom_ts_active && !g_hotstep_params.scheduler.empty()) {
        sampler_build_scheduler_override(custom_schedule, num_steps, schedule);
        schedule = custom_schedule.data();
    }

    // DiT dimensions
    const int Oc     = 64;    // output channels
    const int ctx_ch = 128;   // context channels
    const int in_ch  = 192;   // input channels (ctx_ch + Oc)
    const int H_enc  = 2048;  // encoder hidden size
    const int n_per  = T * Oc;
    const int n_total = N * n_per;

    // CFG batching: pack cond + uncond into N_graph = 2*N
    // For TRT, we don't have a null_condition_emb tensor — we need it provided
    // For now: CFG requires neg_enc_data (or we disable it)
    // TODO: extract null_condition_emb from the model store
    bool do_cfg    = (guidance_scale > 1.0f);
    bool batch_cfg = do_cfg && use_batch_cfg;
    int  N_graph   = batch_cfg ? 2 * N : N;

    fprintf(stderr, "[DiT-TRT] Batch N=%d, T=%d, enc_S=%d, steps=%d%s\n",
            N, T, enc_S, num_steps,
            batch_cfg ? ", CFG batched 2N" : (do_cfg ? ", CFG 2-pass" : ""));

    // ── Solver & guidance dispatch ──────────────────────────────────────
    auto & plugin_reg = PluginRegistry::instance();
    LuaPlugin * solver_plugin = plugin_reg.solver_lookup(solver_name);
    if (!solver_plugin) {
        fprintf(stderr, "[DiT-TRT] ERROR: unknown solver '%s', falling back to euler\n", solver_name);
        solver_plugin = plugin_reg.solver_lookup("euler");
    }
    fprintf(stderr, "[DiT-TRT] Solver: %s (%s)\n",
            solver_plugin->display_name.c_str(), solver_plugin->name.c_str());

    SolverState solver_state;
    solver_state.seeds            = seeds;
    solver_state.batch_n          = N;
    solver_state.n_per            = n_per;
    solver_state.xt_scratch.resize(n_total);
    solver_state.stork_substeps   = stork_substeps;
    solver_state.beat_stability   = beat_stability;
    solver_state.frequency_damping = frequency_damping;
    solver_state.temporal_smoothing = temporal_smoothing;

    LuaPlugin * guidance_plugin = plugin_reg.guidance_lookup(guidance_mode);
    if (!guidance_plugin) {
        fprintf(stderr, "[DiT-TRT] ERROR: unknown guidance '%s', falling back to apg\n", guidance_mode);
        guidance_plugin = plugin_reg.guidance_lookup("apg");
    }
    bool use_apg_native = (guidance_plugin && guidance_plugin->name == "apg");
    fprintf(stderr, "[DiT-TRT] Guidance: %s (%s)%s\n",
            guidance_plugin->display_name.c_str(), guidance_plugin->name.c_str(),
            use_apg_native ? " [native APG]" : "");

    // ── Host buffers ────────────────────────────────────────────────────
    std::vector<float> xt(noise, noise + n_total);
    std::vector<float> vt(n_total);
    std::vector<float> vt_cond, vt_uncond;
    APGWorkspace apg_ws;
    std::vector<APGMomentumBuffer> apg_mbufs;

    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
        apg_ws.resize(n_per);
        apg_mbufs.reserve(N);
        for (int i = 0; i < N; i++) {
            apg_mbufs.emplace_back(-apg_momentum);
        }
    }

    // ── Prepare input_latents host buffer [N_graph, T, 192] ─────────────
    // Pre-fill context_latents in the first 128 channels
    std::vector<float> input_buf(in_ch * T * N_graph);
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[b * T * in_ch + t * in_ch],
                   &context_latents[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));
        }
        if (batch_cfg) {
            memcpy(&input_buf[(N + b) * T * in_ch],
                   &input_buf[b * T * in_ch],
                   T * in_ch * sizeof(float));
        }
    }

    // ── Prepare enc_hidden host buffer [N_graph, enc_S, 2048] ───────────
    std::vector<float> enc_buf(H_enc * enc_S * N_graph);
    memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));

    // Null encoding for uncond slots (CFG)
    std::vector<float> null_enc_buf;
    if (do_cfg && neg_enc_data) {
        // Broadcast neg_enc_data [H_enc] to [enc_S, H_enc]
        std::vector<float> null_enc_single(H_enc * enc_S);
        for (int s = 0; s < enc_S; s++) {
            memcpy(&null_enc_single[s * H_enc], neg_enc_data, H_enc * sizeof(float));
        }
        if (batch_cfg) {
            for (int b = 0; b < N; b++) {
                memcpy(enc_buf.data() + (N + b) * enc_S * H_enc,
                       null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        } else {
            null_enc_buf.resize(H_enc * enc_S * N);
            for (int b = 0; b < N; b++) {
                memcpy(null_enc_buf.data() + b * enc_S * H_enc,
                       null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        }
    }

    // ── GPU buffers (BF16 for TRT I/O) ──────────────────────────────────
    // STRONGLY_TYPED bf16_mixed engine: input_latents, enc_hidden, velocity
    // are bf16. Timesteps (t, t_r) are fp32. Host staging buffers convert
    // fp32 <-> bf16 for GPU transfer.
    size_t input_bf16_elems = (size_t)N_graph * T * in_ch;
    size_t enc_bf16_elems   = (size_t)N_graph * enc_S * H_enc;
    size_t vel_bf16_elems   = (size_t)N_graph * T * Oc;

    // Host BF16 staging
    std::vector<uint16_t> h_input_bf16(input_bf16_elems);
    std::vector<uint16_t> h_enc_bf16(enc_bf16_elems);
    std::vector<uint16_t> h_vel_bf16(vel_bf16_elems);

    // GPU device memory
    void *d_input = nullptr, *d_enc = nullptr, *d_vel = nullptr;
    float *d_t = nullptr, *d_t_r = nullptr;

    cudaMalloc(&d_input, input_bf16_elems * sizeof(uint16_t));
    cudaMalloc(&d_enc,   enc_bf16_elems * sizeof(uint16_t));
    cudaMalloc(&d_vel,   vel_bf16_elems * sizeof(uint16_t));
    cudaMalloc((void**)&d_t,     N_graph * sizeof(float));
    cudaMalloc((void**)&d_t_r,   N_graph * sizeof(float));

    if (!d_input || !d_enc || !d_vel || !d_t || !d_t_r) {
        fprintf(stderr, "[DiT-TRT] FATAL: cudaMalloc failed\n");
        if (d_input) cudaFree(d_input);
        if (d_enc)   cudaFree(d_enc);
        if (d_vel)   cudaFree(d_vel);
        if (d_t)     cudaFree(d_t);
        if (d_t_r)   cudaFree(d_t_r);
        return -1;
    }

    // Pre-upload encoder hidden states (fp32 → bf16)
    convert_fp32_to_bf16(enc_buf.data(), h_enc_bf16.data(), enc_bf16_elems);
    cudaMemcpy(d_enc, h_enc_bf16.data(), enc_bf16_elems * sizeof(uint16_t),
               cudaMemcpyHostToDevice);

    GuidanceCtx g_ctx = {0, num_steps, 0.0f, 0.0f};

    // ── Forward pass helper ─────────────────────────────────────────────
    // Packs xt into input_buf, converts to BF16, uploads, runs TRT, reads back
    auto trt_forward = [&](const float * xt_in, float t_val, int n_batch,
                           float * vel_out) {
        // Pack xt noise channels into input_buf slots [0..n_batch)
        for (int b = 0; b < N; b++) {
            for (int ti = 0; ti < T; ti++) {
                memcpy(&input_buf[b * T * in_ch + ti * in_ch + ctx_ch],
                       &xt_in[b * n_per + ti * Oc],
                       Oc * sizeof(float));
            }
            if (batch_cfg && n_batch > N) {
                for (int ti = 0; ti < T; ti++) {
                    memcpy(&input_buf[(N + b) * T * in_ch + ti * in_ch + ctx_ch],
                           &xt_in[b * n_per + ti * Oc],
                           Oc * sizeof(float));
                }
            }
        }

        // Convert input to BF16 and upload
        size_t input_n = (size_t)n_batch * T * in_ch;
        convert_fp32_to_bf16(input_buf.data(), h_input_bf16.data(), input_n);
        cudaMemcpy(d_input, h_input_bf16.data(), input_n * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);

        // Set timestep (FP32, broadcast to all batch slots)
        std::vector<float> t_host(n_batch, t_val);
        cudaMemcpy(d_t, t_host.data(), n_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_r, t_host.data(), n_batch * sizeof(float), cudaMemcpyHostToDevice);

        // Re-upload encoder states (fp32 → bf16)
        size_t enc_n = (size_t)n_batch * enc_S * H_enc;
        convert_fp32_to_bf16(enc_buf.data(), h_enc_bf16.data(), enc_n);
        cudaMemcpy(d_enc, h_enc_bf16.data(), enc_n * sizeof(uint16_t),
                   cudaMemcpyHostToDevice);

        // Run TRT forward
        bool ok = dit_trt_forward(trt, d_input, d_enc, d_t, d_t_r,
                                   n_batch, T, enc_S, d_vel, nullptr);
        if (!ok) {
            fprintf(stderr, "[DiT-TRT] Forward pass failed!\n");
            return false;
        }

        // Read back velocity (BF16 → FP32)
        size_t vel_n = (size_t)n_batch * T * Oc;
        cudaMemcpy(h_vel_bf16.data(), d_vel, vel_n * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        convert_bf16_to_fp32(h_vel_bf16.data(), vel_out, vel_n);
        return true;
    };

    // ── evaluate_velocity lambda (same interface as GGML sampler) ────────
    std::vector<float> full_output;
    if (batch_cfg) {
        full_output.resize(n_per * N_graph);
    }

    auto evaluate_velocity = [&](const float * xt_in, float t_val) {
        if (batch_cfg) {
            // Single forward with 2N batch
            trt_forward(xt_in, t_val, N_graph, full_output.data());
            memcpy(vt_cond.data(), full_output.data(), n_total * sizeof(float));
            memcpy(vt_uncond.data(), full_output.data() + n_total, n_total * sizeof(float));
            for (int b = 0; b < N; b++) {
                if (use_apg_native) {
                    apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                Oc, T, apg_norm_threshold, apg_ws);
                } else {
                    lua_call_guidance(*guidance_plugin,
                                     vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                     guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                     Oc, T, g_ctx, apg_norm_threshold,
                                     g_hotstep_params.plugin_params);
                }
            }
        } else if (do_cfg) {
            // 2-pass: conditional
            trt_forward(xt_in, t_val, N, vt_cond.data());
            // Swap encoder to null for unconditional pass
            if (!null_enc_buf.empty()) {
                memcpy(enc_buf.data(), null_enc_buf.data(), H_enc * enc_S * N * sizeof(float));
            }
            trt_forward(xt_in, t_val, N, vt_uncond.data());
            // Restore conditional encoder
            memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));

            for (int b = 0; b < N; b++) {
                if (use_apg_native) {
                    apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                Oc, T, apg_norm_threshold, apg_ws);
                } else {
                    lua_call_guidance(*guidance_plugin,
                                     vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                     guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                     Oc, T, g_ctx, apg_norm_threshold,
                                     g_hotstep_params.plugin_params);
                }
            }
        } else {
            // No CFG — single forward
            trt_forward(xt_in, t_val, N, vt.data());
        }
    };

    // ── Diffusion loop ──────────────────────────────────────────────────
    if (solver_plugin->owns_loop) {
        // Full-loop solver path
        bool switched_cover_fl = false;

        LoopModelFn loop_model_fn = [&](const float * xt_in, float t_val) {
            evaluate_velocity(xt_in, t_val);
        };

        LoopOnStepFn loop_on_step = [&](int step_idx, float t_curr, float t_next) -> bool {
            if (cancel && cancel(cancel_data)) {
                fprintf(stderr, "[DiT-TRT] Cancelled at step %d/%d\n", step_idx, num_steps);
                return true;
            }
            if (context_switch && cover_steps >= 0 &&
                step_idx >= cover_steps && !switched_cover_fl) {
                switched_cover_fl = true;
                for (int b = 0; b < N; b++) {
                    for (int t = 0; t < T; t++) {
                        memcpy(&input_buf[b * T * in_ch + t * in_ch],
                               &context_switch[b * T * ctx_ch + t * ctx_ch],
                               ctx_ch * sizeof(float));
                    }
                    if (batch_cfg) {
                        memcpy(&input_buf[(N + b) * T * in_ch],
                               &input_buf[b * T * in_ch],
                               T * in_ch * sizeof(float));
                    }
                }
                if (enc_switch) {
                    memcpy(enc_buf.data(), enc_switch, H_enc * enc_S * N * sizeof(float));
                }
                fprintf(stderr, "[DiT-TRT] Cover: switched at step %d/%d\n", step_idx, num_steps);
            }
            g_ctx.step_idx = step_idx + 1;
            g_ctx.t_curr   = t_curr;
            g_ctx.dt       = t_curr - t_next;
            sampler_apply_dcw(xt.data(), vt.data(), N, T, Oc, t_curr, t_next, step_idx, num_steps);
            sampler_repaint_inject(xt.data(), noise, nullptr, N, T, Oc,
                                   0, 0, 0.5f, step_idx, num_steps, t_next);
            fprintf(stderr, "[DiT-TRT] Step %d/%d t=%.3f [%s]\n",
                    step_idx + 1, num_steps, t_curr, solver_plugin->display_name.c_str());
            return false;
        };

        lua_call_solver_loop(*solver_plugin, xt.data(), vt.data(), schedule, num_steps,
            n_total, N, T, Oc, loop_model_fn, loop_on_step,
            g_hotstep_params.plugin_params);

        memcpy(output, xt.data(), n_total * sizeof(float));

    } else {
        // Per-step loop
        bool switched_cover = false;

        // Velocity caching
        float cache_ratio = g_hotstep_params.cache_ratio;
        std::vector<bool> step_computes(num_steps, true);
        std::vector<float> vt_cached;
        bool has_cached_vt = false;
        int  cached_count  = 0;

        if (cache_ratio > 0.0f && num_steps > 4) {
            int protect = 2;
            int middle_start = protect;
            int middle_end   = num_steps - protect;
            int middle_len   = middle_end - middle_start;
            if (middle_len > 1) {
                int target_cached = std::min(middle_len - 1,
                                             (int)roundf(cache_ratio * middle_len));
                int target_compute = middle_len - target_cached;
                if (target_compute > 0 && target_cached > 0) {
                    for (int s = middle_start; s < middle_end; s++) {
                        step_computes[s] = false;
                    }
                    for (int ci = 0; ci < target_compute; ci++) {
                        int idx = middle_start + (int)roundf((float)ci * middle_len / target_compute);
                        if (idx < middle_end) step_computes[idx] = true;
                    }
                }
            }
            vt_cached.resize(n_total);
            for (int s = 0; s < num_steps; s++) {
                if (!step_computes[s]) cached_count++;
            }
            fprintf(stderr, "[DiT-TRT] Velocity cache: ratio=%.2f, %d/%d cached\n",
                    cache_ratio, cached_count, num_steps);
        }

        for (int step = 0; step < num_steps; step++) {
            if (cancel && cancel(cancel_data)) {
                fprintf(stderr, "[DiT-TRT] Cancelled at step %d/%d\n", step, num_steps);
                cudaFree(d_input); cudaFree(d_enc); cudaFree(d_vel);
                cudaFree(d_t); cudaFree(d_t_r);
                return -1;
            }
            float t_curr = schedule[step];
            float t_next = (step + 1 < num_steps) ? schedule[step + 1] : 0.0f;

            // Cover mode switch
            if (context_switch && cover_steps >= 0 && step >= cover_steps && !switched_cover) {
                switched_cover = true;
                for (int b = 0; b < N; b++) {
                    for (int t = 0; t < T; t++) {
                        memcpy(&input_buf[b * T * in_ch + t * in_ch],
                               &context_switch[b * T * ctx_ch + t * ctx_ch],
                               ctx_ch * sizeof(float));
                    }
                    if (batch_cfg) {
                        memcpy(&input_buf[(N + b) * T * in_ch],
                               &input_buf[b * T * in_ch],
                               T * in_ch * sizeof(float));
                    }
                }
                if (enc_switch) {
                    memcpy(enc_buf.data(), enc_switch, H_enc * enc_S * N * sizeof(float));
                }
                fprintf(stderr, "[DiT-TRT] Cover: switched at step %d/%d\n", step, num_steps);
            }

            g_ctx.step_idx = step;
            g_ctx.t_curr   = t_curr;
            g_ctx.dt       = t_curr - t_next;

            // Evaluate or cache
            if (!step_computes[step] && has_cached_vt) {
                memcpy(vt.data(), vt_cached.data(), n_total * sizeof(float));
            } else {
                evaluate_velocity(xt.data(), t_curr);
                if (cached_count > 0) {
                    memcpy(vt_cached.data(), vt.data(), n_total * sizeof(float));
                    has_cached_vt = true;
                }
            }

            // Apply solver step
            const float * vt_readonly = vt.data();
            solver_state.step_index = step;
            lua_call_solver_step(*solver_plugin, xt.data(), vt_readonly,
                t_curr, t_next, n_total,
                solver_state, evaluate_velocity, vt.data(),
                g_hotstep_params.plugin_params);

            // DCW
            sampler_apply_dcw(xt.data(), vt.data(), N, T, Oc, t_curr, t_next, step, num_steps);

            // Repaint
            sampler_repaint_inject(xt.data(), noise, nullptr, N, T, Oc,
                                   0, 0, 0.5f, step, num_steps, t_next);

            fprintf(stderr, "[DiT-TRT] Step %d/%d t=%.3f [%s]\n",
                    step + 1, num_steps, t_curr, solver_plugin->display_name.c_str());
        }

        memcpy(output, xt.data(), n_total * sizeof(float));
    }

    // Cleanup GPU buffers
    cudaFree(d_input);
    cudaFree(d_enc);
    cudaFree(d_vel);
    cudaFree(d_t);
    cudaFree(d_t_r);

    fprintf(stderr, "[DiT-TRT] Generation complete (%d steps)\n", num_steps);
    return 0;
}

#endif // HOT_STEP_TRT
