#pragma once
// dit-sampler.h: DiT sampling loop with modular solver/guidance dispatch
//
// Flow matching sampler with modular solver and guidance mode dispatch.
// Solvers are resolved by name via solvers/solver-registry.h.
// Guidance modes are resolved by name via guidance/guidance-registry.h.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include "alignment-config.h"
#include "dcw.h"
#include "debug.h"
#include "dit-alignment-graph.h"
#include "dit-graph.h"
#include "dit.h"
#include "guidance/apg-core.h"
#include "hot-step-params.h"
#include "lua-plugin-registry.h"
#include "philox.h"
#include "sampler-dcw.h"
#include "sampler-repaint.h"
#include "sampler-schedule.h"

#include <algorithm>
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
                             const float *   neg_enc_data             = nullptr,
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
    std::vector<float> custom_ts_schedule;
    bool custom_ts_active = sampler_parse_custom_timesteps(custom_ts_schedule, num_steps);
    if (custom_ts_active) {
        schedule = custom_ts_schedule.data();
    }

    // ── Scheduler override ────────────────────────────────────────────────
    std::vector<float> custom_schedule;
    if (!custom_ts_active && !g_hotstep_params.scheduler.empty()) {
        sampler_build_scheduler_override(custom_schedule, num_steps, schedule);
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

    // Graph context (generous fixed allocation, shapes are constant across steps).
    // Per-section masking multiplies the per-adapter LoRA nodes, so scale the node
    // budget with the separate-adapter count (m->loras) to avoid graph overflow.
    size_t               graph_cap = 8192 + model->loras.size() * 4096;
    size_t               ctx_size = ggml_tensor_overhead() * graph_cap + ggml_graph_overhead_custom(graph_cap, false);
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
        // Per-section adapter masks (dynamically named) — force to backend too.
        for (struct ggml_tensor * mk : model->lora_masks) {
            if (mk) ggml_backend_sched_set_tensor_backend(model->sched, mk, model->backend);
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

    // ── Per-section adapter masks (regional LoRA) ─────────────────────────────
    // P1: proportional frame→section map. Partition the S latent frames across the
    // sections by their relative `size`, then each adapter's mask is that section's
    // weight, with a short linear crossfade at boundaries. Computed once into
    // lora_mask_host and (re-)uploaded whenever the graph is (re)built — the CFG
    // cutoff rebuilds the graph mid-sampling, so a one-time upload would leave the
    // new mask tensors uninitialised for the final steps. P2 will recompute the
    // host buffers mid-sampling from cross-attention alignment.
    std::vector<std::vector<float>> lora_mask_host;  // [adapter][S]
    const bool section_active = (!model->lora_masks.empty() && !g_hotstep_params.adapter_sections.empty());

    // Build the per-adapter masks from a frame→section assignment (hard weights,
    // triangular-smoothed by ±xf for a ~0.5 s crossfade). Reused for both the P1
    // proportional map and the P2 alignment-derived map. Uploads to the GPU mask
    // tensors (the per-step re-upload keeps them alive against the scheduler).
    auto rebuild_section_masks = [&](const std::vector<int> & frame_sec, const char * tag) {
        if (!section_active) return;
        const auto & secs = g_hotstep_params.adapter_sections;
        const size_t Nad  = model->lora_masks.size();
        auto secWeight = [&](int k, size_t i) -> float {
            return (k >= 0 && k < (int) secs.size() && i < secs[k].weights.size()) ? secs[k].weights[i] : 0.0f;
        };
        const int xf = std::max(3, S / 300);
        lora_mask_host.assign(Nad, std::vector<float>(S, 0.0f));
        std::vector<float> hard(S);
        for (size_t i = 0; i < Nad; i++) {
            for (int s = 0; s < S; s++) hard[s] = secWeight(s < (int) frame_sec.size() ? frame_sec[s] : 0, i);
            float mn = 1e9f, mx = -1e9f, sum = 0.0f;
            for (int s = 0; s < S; s++) {
                double acc = 0.0, wsum = 0.0;
                int lo = s - xf < 0 ? 0 : s - xf;
                int hi = s + xf >= S ? S - 1 : s + xf;
                for (int sp = lo; sp <= hi; sp++) {
                    double wd = (double) (xf + 1 - std::abs(sp - s));
                    acc += hard[sp] * wd;
                    wsum += wd;
                }
                float v              = (float) (wsum > 0 ? acc / wsum : hard[s]);
                lora_mask_host[i][s] = v;
                mn = v < mn ? v : mn; mx = v > mx ? v : mx; sum += v;
            }
            ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
            fprintf(stderr, "[Adapter-RT]   mask[%zu] min=%.3f max=%.3f mean=%.3f (%s)\n",
                    i, mn, mx, sum / (float) S, tag);
        }
    };

    // Initial P1 proportional frame→section map (also the fallback if P2 alignment
    // is disabled or fails). Sections partition the S frames by their `size`.
    if (section_active) {
        const auto & secs = g_hotstep_params.adapter_sections;
        double total = 0.0;
        for (const auto & sc : secs) total += (sc.size > 0.0f ? sc.size : 1.0f);
        if (total <= 0.0) total = 1.0;
        std::vector<int> bnd(secs.size() + 1, 0);
        double cum = 0.0;
        for (size_t k = 0; k < secs.size(); k++) {
            cum += (secs[k].size > 0.0f ? secs[k].size : 1.0f);
            bnd[k + 1] = (int) llround((double) S * cum / total);
        }
        bnd[secs.size()] = S;
        std::vector<int> frame_sec(S, 0);
        for (size_t k = 0; k < secs.size(); k++)
            for (int s = bnd[k]; s < bnd[k + 1] && s < S; s++) frame_sec[s] = (int) k;
        rebuild_section_masks(frame_sec, "proportional");
        fprintf(stderr, "[Adapter-RT] Per-section masks: %zu adapters × %d frames, %zu sections\n",
                model->lora_masks.size(), S, secs.size());
    }

    // ── P2: alignment-driven section boundaries ───────────────────────────────
    // Run the cross-attention alignment once at ~align_at, derive frame→section
    // from the model's real lyric→audio alignment, and rebuild the masks. Keeps
    // the P1 proportional map if disabled/failed. Called from the loop with the
    // current batch-0 latent (xt) and velocity (vt).
    bool      p2_done       = false;
    const int p2_align_step = (section_active && g_hotstep_params.adapter_section_align_at > 0.0f &&
                               !g_hotstep_params.adapter_section_token_map.empty())
                                  ? std::max(1, std::min(num_steps - 1, (int) (g_hotstep_params.adapter_section_align_at * num_steps)))
                                  : -1;
    auto run_p2_alignment = [&](const float * xt_data, const float * vt_data, float t_curr) {
        if (p2_done || p2_align_step < 0) return;
        p2_done = true;
        std::vector<float> x0((size_t) T * Oc);
        for (int i = 0; i < T * Oc; i++) x0[i] = xt_data[i] - t_curr * vt_data[i];  // flow-matching x0 estimate

        AlignmentConfig acfg = alignment_config_resolve("", model->cfg.n_layers, model->cfg.n_heads);
        if (!acfg.valid || acfg.total_heads <= 0) {
            fprintf(stderr, "[Adapter-RT] P2: no alignment config — keeping proportional map\n");
            return;
        }
        bool saved_fa       = model->use_flash_attn;
        model->use_flash_attn = false;  // scored attention needs manual (non-flash) path
        std::vector<float> scores((size_t) acfg.total_heads * enc_S * S);
        int rc = dit_alignment_extract(model, acfg, x0.data(), context_latents, enc_hidden_data,
                                       T, S, enc_S, 8, scores.data());
        model->use_flash_attn = saved_fa;
        if (rc != 0) {
            fprintf(stderr, "[Adapter-RT] P2: alignment extract failed — keeping proportional map\n");
            return;
        }
        // frame → dominant lyric token → section (attention summed over heads).
        const auto &     tmap = g_hotstep_params.adapter_section_token_map;
        std::vector<int> frame_sec(S, 0);
        int              last_sec = 0;
        for (int f = 0; f < S; f++) {
            float best = -1e30f; int best_tok = -1;
            for (int tok = 0; tok < enc_S && tok < (int) tmap.size(); tok++) {
                if (tmap[tok] < 0) continue;  // non-lyric token
                float acc = 0.0f;
                for (int h = 0; h < acfg.total_heads; h++)
                    acc += scores[(size_t) h * enc_S * S + (size_t) f * enc_S + tok];
                if (acc > best) { best = acc; best_tok = tok; }
            }
            int sec      = (best_tok >= 0) ? tmap[best_tok] : last_sec;
            frame_sec[f] = sec;
            last_sec     = sec;
        }
        // Median-smooth (±med) to remove single-frame jitter before the crossfade.
        const int med = std::max(1, S / 200);
        std::vector<int> fs2(frame_sec);
        for (int f = 0; f < S; f++) {
            std::vector<int> win;
            for (int d = -med; d <= med; d++) { int p = f + d; if (p >= 0 && p < S) win.push_back(frame_sec[p]); }
            std::sort(win.begin(), win.end());
            fs2[f] = win[win.size() / 2];
        }
        rebuild_section_masks(fs2, "alignment");
        fprintf(stderr, "[Adapter-RT] P2: masks rebuilt from cross-attention alignment (step %d, t=%.3f)\n",
                p2_align_step, t_curr);
    };

    std::vector<APGMomentumBuffer> apg_mbufs;
    std::vector<float>             null_enc_buf;

    // Post-step model eval buffers (only for guidance plugins with has_post_step)
    std::vector<float>             enc_cond_full;    // [H_enc, enc_S, N_graph] all-conditional
    std::vector<float>             enc_uncond_full;  // [H_enc, enc_S, N_graph] all-unconditional

    if (do_cfg) {
        apg_mbufs.reserve(N);
        for (int i = 0; i < N; i++) {
            apg_mbufs.emplace_back(-apg_momentum);  // configurable momentum
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
    std::vector<float> vt_pre_solver;  // snapshot of vt before solver step (for DCW)
                                       // Multi-eval solvers overwrite vt via model_fn,
                                       // but DCW needs the original velocity.

    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    std::vector<float> full_output;   // pre-allocated readback buffer for batched CFG
    APGWorkspace apg_ws;               // pre-allocated APG workspace (~1.5MB reused per step)
    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
        if (batch_cfg) {
            full_output.resize(n_per * N_graph);
        }
        apg_ws.resize(n_per);  // one batch element at a time
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
        // Use neg_enc_data if provided (negative_prompt encoded upstream)
        std::vector<float> null_enc_single(H_enc * enc_S);
        const float * uncond_src = (neg_enc_data != nullptr) ? neg_enc_data : null_emb.data();
        for (int s = 0; s < enc_S; s++) {
            memcpy(&null_enc_single[s * H_enc], uncond_src, H_enc * sizeof(float));
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

    // ── Solver dispatch setup (Lua plugin system) ────────────────────────
    auto & plugin_reg = PluginRegistry::instance();
    LuaPlugin * solver_plugin = plugin_reg.solver_lookup(solver_name);
    if (!solver_plugin) {
        fprintf(stderr, "[DiT] ERROR: unknown solver '%s', falling back to euler\n", solver_name);
        solver_plugin = plugin_reg.solver_lookup("euler");
    }
    fprintf(stderr, "[DiT] Solver: %s (%s, %d NFE/step, order %d)\n",
            solver_plugin->display_name.c_str(), solver_plugin->name.c_str(),
            solver_plugin->nfe, solver_plugin->order);

    // Initialize solver state (still used by SDE Philox path)
    SolverState solver_state;
    solver_state.seeds            = seeds;
    solver_state.batch_n          = N;
    solver_state.n_per            = n_per;
    solver_state.xt_scratch.resize(n_total);
    solver_state.stork_substeps   = stork_substeps;
    solver_state.beat_stability   = beat_stability;
    solver_state.frequency_damping = frequency_damping;
    solver_state.temporal_smoothing = temporal_smoothing;

    // ── Guidance mode dispatch setup (Lua plugin system) ─────────────────
    LuaPlugin * guidance_plugin = plugin_reg.guidance_lookup(guidance_mode);
    if (!guidance_plugin) {
        fprintf(stderr, "[DiT] ERROR: unknown guidance mode '%s', falling back to apg\n", guidance_mode);
        guidance_plugin = plugin_reg.guidance_lookup("apg");
    }
    bool use_apg_native = (guidance_plugin && guidance_plugin->name == "apg");
    fprintf(stderr, "[DiT] Guidance: %s (%s)%s%s\n",
            guidance_plugin->display_name.c_str(), guidance_plugin->name.c_str(),
            use_apg_native ? " [native APG]" : "",
            guidance_plugin->has_post_step ? " [post_step]" : "");

    // Build post-step encoding buffers if the guidance plugin needs model callbacks
    if (guidance_plugin->has_post_step && do_cfg) {
        enc_cond_full.resize(H_enc * enc_S * N_graph);
        enc_uncond_full.resize(H_enc * enc_S * N_graph);
        // null_enc_single is already computed above (inside if(do_cfg))
        // We need to recompute it here since it was local to that block
        std::vector<float> null_enc_ps(H_enc * enc_S);
        {
            int emb_n = (int) ggml_nelements(model->null_condition_emb);
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
            const float * uncond_src_ps = (neg_enc_data != nullptr) ? neg_enc_data : null_emb.data();
            for (int s = 0; s < enc_S; s++) {
                memcpy(&null_enc_ps[s * H_enc], uncond_src_ps, H_enc * sizeof(float));
            }
        }
        for (int b = 0; b < N_graph; b++) {
            int src_b = b % N;
            memcpy(enc_cond_full.data() + b * enc_S * H_enc,
                   enc_hidden_data + src_b * enc_S * H_enc,
                   enc_S * H_enc * sizeof(float));
            memcpy(enc_uncond_full.data() + b * enc_S * H_enc,
                   null_enc_ps.data(),
                   enc_S * H_enc * sizeof(float));
        }
        fprintf(stderr, "[DiT] Post-step model eval buffers ready for '%s'\n",
                guidance_plugin->name.c_str());
    }

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

        // Re-upload constants every step — the GGML scheduler aliases input tensor
        // buffers as scratch space for intermediate computations, clobbering them
        // after each graph compute. Confirmed: skipping these produces blank output.
        ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N_graph * sizeof(int32_t));
        ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N_graph * sizeof(uint16_t));
        // Per-section adapter masks are constant GPU inputs read at every layer —
        // same as enc_hidden, the scheduler clobbers them after each compute, so
        // they MUST be re-uploaded every step. Uploading once left them garbage for
        // steps 2..N, zeroing the masked adapter delta (the "nil adapter" bug).
        for (size_t i = 0; i < model->lora_masks.size() && i < lora_mask_host.size(); i++) {
            if (model->lora_masks[i])
                ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
        }

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
            ggml_backend_tensor_get(t_output, full_output.data(), 0, n_per * N_graph * sizeof(float));
            memcpy(vt_cond.data(), full_output.data(), n_total * sizeof(float));
            memcpy(vt_uncond.data(), full_output.data() + n_total, n_total * sizeof(float));
            for (int b = 0; b < N; b++) {
                if (use_apg_native) {
                    apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale,
                                apg_mbufs[b], vt.data() + b * n_per, Oc, T, apg_norm_threshold, apg_ws);
                } else {
                    lua_call_guidance(*guidance_plugin,
                                     vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                     guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                     Oc, T, g_ctx, apg_norm_threshold,
                                     g_hotstep_params.plugin_params);
                }
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
            for (size_t i = 0; i < model->lora_masks.size() && i < lora_mask_host.size(); i++) {
                if (model->lora_masks[i])
                    ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
            }
            ggml_backend_sched_graph_compute(model->sched, gf);
            ggml_backend_tensor_get(t_output, vt_uncond.data(), 0, n_total * sizeof(float));
            for (int b = 0; b < N; b++) {
                if (use_apg_native) {
                    apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale,
                                apg_mbufs[b], vt.data() + b * n_per, Oc, T, apg_norm_threshold, apg_ws);
                } else {
                    lua_call_guidance(*guidance_plugin,
                                     vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per,
                                     guidance_scale, apg_mbufs[b], vt.data() + b * n_per,
                                     Oc, T, g_ctx, apg_norm_threshold,
                                     g_hotstep_params.plugin_params);
                }
            }
        } else {
            ggml_backend_tensor_get(t_output, vt.data(), 0, n_total * sizeof(float));
        }
    };

    // ── eval_single_pass: run ONE model forward with specified encoding ──────
    // Used by post_step hooks (CFG-MP etc.) to evaluate at arbitrary positions
    // with either conditional or unconditional encoding.
    auto eval_single_pass = [&](const float * xt_in, float t_val,
                                const std::vector<float> & enc_full,
                                float * out_buf) {
        if (t_t) { ggml_backend_tensor_set(t_t, &t_val, 0, sizeof(float)); }
        if (t_tr) { ggml_backend_tensor_set(t_tr, &t_val, 0, sizeof(float)); }

        // Upload encoding (all N_graph slots, all same type)
        ggml_backend_tensor_set(t_enc, enc_full.data(), 0,
                                H_enc * enc_S * N_graph * sizeof(float));

        // Pack xt into all N_graph slots
        for (int b = 0; b < N_graph; b++) {
            int src_b = b % N;
            for (int ti = 0; ti < T; ti++) {
                memcpy(&input_buf[b * T * in_ch + ti * in_ch + ctx_ch],
                       &xt_in[src_b * n_per + ti * Oc],
                       Oc * sizeof(float));
            }
        }
        ggml_backend_tensor_set(t_input, input_buf.data(), 0,
                                in_ch * T * N_graph * sizeof(float));

        // Re-upload masks/positions (full N_graph)
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0,
                                S * N_graph * sizeof(int32_t));
        ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0,
                                S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0,
                                S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0,
                                enc_S * S * N_graph * sizeof(uint16_t));
        for (size_t i = 0; i < model->lora_masks.size() && i < lora_mask_host.size(); i++) {
            if (model->lora_masks[i])
                ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
        }

        // Forward pass
        ggml_backend_sched_graph_compute(model->sched, gf);

        // Read first N samples only
        ggml_backend_tensor_get(t_output, out_buf, 0, n_total * sizeof(float));
    };

    // ── Step-level velocity caching (shared by both solver paths) ────────
    // Build a static schedule: which steps COMPUTE vs reuse cached velocity.
    // Pattern: always compute first 2 and last 2 steps (boundaries matter most).
    // Middle steps are computed every Nth step based on cache_ratio.
    float cache_ratio = g_hotstep_params.cache_ratio;
    std::vector<bool> step_computes(num_steps, true);
    std::vector<float> vt_cached;
    bool has_cached_vt = false;
    int  cached_count  = 0;

    if (cache_ratio > 0.0f && num_steps > 4) {
        int protect = 2;  // protect first/last N steps
        int middle_start = protect;
        int middle_end   = num_steps - protect;
        int middle_len   = middle_end - middle_start;

        if (middle_len > 1) {
            int target_cached = std::min(middle_len - 1,
                                         (int) roundf(cache_ratio * middle_len));
            int target_compute = middle_len - target_cached;

            if (target_compute > 0 && target_cached > 0) {
                for (int s = middle_start; s < middle_end; s++) {
                    step_computes[s] = false;
                }
                for (int ci = 0; ci < target_compute; ci++) {
                    int idx = middle_start + (int) roundf((float) ci * middle_len / target_compute);
                    if (idx < middle_end) step_computes[idx] = true;
                }
            }
        }

        vt_cached.resize(n_total);

        for (int s = 0; s < num_steps; s++) {
            if (!step_computes[s]) cached_count++;
        }
        fprintf(stderr, "[DiT] Velocity cache: ratio=%.2f, %d/%d steps cached, %d computed\n",
                cache_ratio, cached_count, num_steps, num_steps - cached_count);
    }

    // Flow matching loop
    if (solver_plugin->owns_loop) {
        // ── Full-loop solver path ──────────────────────────────────────
        bool switched_cover_fl = false;
        int  current_model_step = 0;  // tracks which step the next model_fn call serves

        LoopModelFn loop_model_fn = [&](const float * xt_in, float t_val) {
            // Velocity caching: skip forward pass on cached steps
            if (cached_count > 0 && !step_computes[current_model_step] && has_cached_vt) {
                memcpy(vt.data(), vt_cached.data(), n_total * sizeof(float));
            } else {
                evaluate_velocity(xt_in, t_val);
                if (cached_count > 0) {
                    memcpy(vt_cached.data(), vt.data(), n_total * sizeof(float));
                    has_cached_vt = true;
                }
            }
            if (current_model_step == p2_align_step) run_p2_alignment(xt_in, vt.data(), t_val);
        };

        LoopOnStepFn loop_on_step = [&](int step_idx, float t_curr, float t_next) -> bool {
            if (cancel && cancel(cancel_data)) {
                fprintf(stderr, "[DiT] Cancelled at step %d/%d\n", step_idx, num_steps);
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
                    if (real_enc_S_switch) {
                        for (int b = 0; b < N; b++) {
                            int re = real_enc_S_switch[b];
                            for (int qi = 0; qi < S; qi++) {
                                for (int ki = 0; ki < enc_S; ki++) {
                                    float v = (ki < re) ? 0.0f : -INFINITY;
                                    ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
                                }
                            }
                        }
                    }
                }
                fprintf(stderr, "[DiT] Cover: switched at step %d/%d\n", step_idx, num_steps);
            }
            // Advance step_idx by +1: full-loop solvers call model_fn (which
            // reads g_ctx for guidance) BEFORE on_step fires, so we set the
            // context for the NEXT iteration's eval. First eval reads 0 from
            // the initial g_ctx, so the +1 here makes step 1's eval see 1, etc.
            g_ctx.step_idx = step_idx + 1;
            g_ctx.t_curr   = t_curr;
            g_ctx.dt       = t_curr - t_next;
            // Full-loop solvers control their own model_fn calls, so vt may
            // have been overwritten. For safety, use vt as-is (full-loop
            // solvers are responsible for their own vt state).
            // TODO: snapshot vt in loop_on_step if full-loop multi-eval solvers need DCW
            sampler_apply_dcw(xt.data(), vt.data(), N, T, Oc, t_curr, t_next, step_idx, num_steps);
            if (guidance_plugin->has_post_step && do_cfg && step_idx < num_steps - 1) {
                PostStepModelFn eval_cond_fn = [&](const float * xt_in, float t_v) {
                    eval_single_pass(xt_in, t_v, enc_cond_full, vt_cond.data());
                };
                PostStepModelFn eval_uncond_fn = [&](const float * xt_in, float t_v) {
                    eval_single_pass(xt_in, t_v, enc_uncond_full, vt_uncond.data());
                };
                lua_call_post_step(*guidance_plugin, xt.data(), t_next, n_total,
                    eval_cond_fn, eval_uncond_fn,
                    vt_cond.data(), vt_uncond.data(),
                    g_ctx, g_hotstep_params.plugin_params);
            }
            sampler_repaint_inject(xt.data(), noise, repaint_src, N, T, Oc,
                                   repaint_t0, repaint_t1, repaint_injection_ratio,
                                   step_idx, num_steps, t_next);

            // Advance step counter for next model_fn call
            current_model_step = step_idx + 1;

            // ── CFG cutoff: switch from 2N batched graph to N cond-only graph ──
            int cfg_cutoff_step = (int) ceilf(g_hotstep_params.cfg_cutoff_ratio * num_steps);
            if (do_cfg && current_model_step == cfg_cutoff_step && cfg_cutoff_step < num_steps) {
                fprintf(stderr, "[DiT] CFG cutoff at step %d/%d — switching to cond-only (N_graph %d -> %d)\n",
                        current_model_step, num_steps, N_graph, N);

                do_cfg    = false;
                batch_cfg = false;
                N_graph   = N;

                ggml_free(ctx);
                ctx_buf.assign(ctx_size, 0);
                struct ggml_init_params gp2 = { ctx_size, ctx_buf.data(), true };
                ctx = ggml_init(gp2);
                gf  = dit_ggml_build_graph(model, ctx, T, enc_S, N_graph, &t_input, &t_output);

                ggml_backend_sched_reset(model->sched);
                if (model->backend != model->cpu_backend) {
                    const char * input_names[] = { "enc_hidden", "input_latents", "t", "t_r",
                                                   "positions", "sa_mask_sw", "sa_mask_pad", "ca_mask" };
                    for (const char * iname : input_names) {
                        struct ggml_tensor * ti = ggml_graph_get_tensor(gf, iname);
                        if (ti) ggml_backend_sched_set_tensor_backend(model->sched, ti, model->backend);
                    }
                    for (struct ggml_tensor * mk : model->lora_masks) {
                        if (mk) ggml_backend_sched_set_tensor_backend(model->sched, mk, model->backend);
                    }
                }
                if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
                    fprintf(stderr, "[DiT] FATAL: failed to re-allocate graph after CFG cutoff\n");
                    // Can't return from lambda with int, just log and continue
                }

                t_enc        = ggml_graph_get_tensor(gf, "enc_hidden");
                t_t          = ggml_graph_get_tensor(gf, "t");
                t_tr         = ggml_graph_get_tensor(gf, "t_r");
                t_pos        = ggml_graph_get_tensor(gf, "positions");
                t_sa_mask_sw = ggml_graph_get_tensor(gf, "sa_mask_sw");
                t_sa_mask_pad= ggml_graph_get_tensor(gf, "sa_mask_pad");
                t_ca_mask    = ggml_graph_get_tensor(gf, "ca_mask");

                input_buf.resize(in_ch * T * N);
                for (int b = 0; b < N; b++) {
                    for (int t = 0; t < T; t++) {
                        memcpy(&input_buf[b * T * in_ch + t * in_ch],
                               &context_latents[b * T * ctx_ch + t * ctx_ch],
                               ctx_ch * sizeof(float));
                    }
                }

                enc_buf.resize(H_enc * enc_S * N);
                memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));

                pos_data.resize(S * N);
                for (int b = 0; b < N; b++)
                    for (int i = 0; i < S; i++)
                        pos_data[b * S + i] = i;

                sa_sw_data.resize(S * S * N);
                sa_pad_data.resize(S * S * N);
                ca_data.resize(enc_S * S * N);

                ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
                ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
                ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N * sizeof(uint16_t));
                ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N * sizeof(uint16_t));
                ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N * sizeof(uint16_t));

                // Re-upload per-section adapter masks into the rebuilt graph's fresh
                // mask tensors (else the post-cutoff steps run with garbage masks).
                for (size_t i = 0; i < model->lora_masks.size() && i < lora_mask_host.size(); i++) {
                    if (model->lora_masks[i])
                        ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
                }

                fprintf(stderr, "[DiT] Graph rebuilt: %d nodes (was 2N, now N=%d)\n",
                        ggml_graph_n_nodes(gf), N);
            }

            fprintf(stderr, "[DiT] Step %d/%d t=%.3f [%s]%s\n",
                    step_idx + 1, num_steps, t_curr, solver_plugin->display_name.c_str(),
                    (cached_count > 0 && !step_computes[step_idx]) ? " (cached)" : "");
            return false;
        };

        lua_call_solver_loop(*solver_plugin, xt.data(), vt.data(), schedule, num_steps,
            n_total, N, T, Oc, loop_model_fn, loop_on_step,
            g_hotstep_params.plugin_params);

        memcpy(output, xt.data(), n_total * sizeof(float));


    } else {
    // ── Existing per-step loop ─────────────────────────────────────
    // (step_computes, vt_cached, cached_count are declared above the if/else split)
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

        // ── CFG cutoff: switch from 2N batched graph to N cond-only graph ──
        // Rebuilds the graph at half size, eliminating the unconditional forward
        // pass for remaining steps. One-time cost (~1ms) saves ~50% compute/step.
        int cfg_cutoff_step = (int) ceilf(g_hotstep_params.cfg_cutoff_ratio * num_steps);
        if (do_cfg && step == cfg_cutoff_step && cfg_cutoff_step < num_steps) {
            fprintf(stderr, "[DiT] CFG cutoff at step %d/%d — switching to cond-only (N_graph %d -> %d)\n",
                    step, num_steps, N_graph, N);

            // Disable CFG for remaining steps
            do_cfg    = false;
            batch_cfg = false;
            N_graph   = N;

            // Rebuild graph with smaller N_graph
            ggml_free(ctx);
            ctx_buf.assign(ctx_size, 0);
            struct ggml_init_params gp2 = { ctx_size, ctx_buf.data(), true };
            ctx = ggml_init(gp2);
            gf  = dit_ggml_build_graph(model, ctx, T, enc_S, N_graph, &t_input, &t_output);

            // Re-allocate scheduler for new graph
            ggml_backend_sched_reset(model->sched);
            if (model->backend != model->cpu_backend) {
                const char * input_names[] = { "enc_hidden", "input_latents", "t", "t_r",
                                               "positions", "sa_mask_sw", "sa_mask_pad", "ca_mask" };
                for (const char * iname : input_names) {
                    struct ggml_tensor * ti = ggml_graph_get_tensor(gf, iname);
                    if (ti) ggml_backend_sched_set_tensor_backend(model->sched, ti, model->backend);
                }
                for (struct ggml_tensor * mk : model->lora_masks) {
                    if (mk) ggml_backend_sched_set_tensor_backend(model->sched, mk, model->backend);
                }
            }
            if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
                fprintf(stderr, "[DiT] FATAL: failed to re-allocate graph after CFG cutoff\n");
                ggml_free(ctx);
                return -1;
            }

            // Update tensor references (lambda captures by [&])
            t_enc        = ggml_graph_get_tensor(gf, "enc_hidden");
            t_t          = ggml_graph_get_tensor(gf, "t");
            t_tr         = ggml_graph_get_tensor(gf, "t_r");
            t_pos        = ggml_graph_get_tensor(gf, "positions");
            t_sa_mask_sw = ggml_graph_get_tensor(gf, "sa_mask_sw");
            t_sa_mask_pad= ggml_graph_get_tensor(gf, "sa_mask_pad");
            t_ca_mask    = ggml_graph_get_tensor(gf, "ca_mask");

            // Resize host buffers for N_graph=N (drop uncond slots)
            input_buf.resize(in_ch * T * N);
            for (int b = 0; b < N; b++) {
                for (int t = 0; t < T; t++) {
                    memcpy(&input_buf[b * T * in_ch + t * in_ch],
                           &context_latents[b * T * ctx_ch + t * ctx_ch],
                           ctx_ch * sizeof(float));
                }
            }

            enc_buf.resize(H_enc * enc_S * N);
            memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));

            pos_data.resize(S * N);
            for (int b = 0; b < N; b++)
                for (int i = 0; i < S; i++)
                    pos_data[b * S + i] = i;

            sa_sw_data.resize(S * S * N);
            sa_pad_data.resize(S * S * N);
            ca_data.resize(enc_S * S * N);
            // Masks for N samples are already in the first N slots — just truncate

            // Upload constants to new tensors
            ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
            ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
            ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N * sizeof(uint16_t));

            // Re-upload per-section adapter masks — the rebuild created fresh mask
            // input tensors ([1,S,1], batch-independent) that would otherwise be
            // uninitialised, silently corrupting the adapter effect for the final
            // (post-cutoff) steps.
            for (size_t i = 0; i < model->lora_masks.size() && i < lora_mask_host.size(); i++) {
                if (model->lora_masks[i])
                    ggml_backend_tensor_set(model->lora_masks[i], lora_mask_host[i].data(), 0, (size_t) S * sizeof(float));
            }

            fprintf(stderr, "[DiT] Graph rebuilt: %d nodes (was 2N, now N=%d)\n",
                    ggml_graph_n_nodes(gf), N);
        }

        // Evaluate velocity at (xt, t_curr) — or reuse cached velocity
        if (!step_computes[step] && has_cached_vt) {
            // CACHE HIT: skip forward pass, reuse previous velocity
            memcpy(vt.data(), vt_cached.data(), n_total * sizeof(float));
        } else {
            // COMPUTE: full forward pass
            evaluate_velocity(xt.data(), t_curr);

            // Cache this velocity for potential reuse by next step(s)
            if (cached_count > 0) {
                memcpy(vt_cached.data(), vt.data(), n_total * sizeof(float));
                has_cached_vt = true;
            }
        }
        if (step == p2_align_step) run_p2_alignment(xt.data(), vt.data(), t_curr);

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

            // ── Buffer separation for multi-eval solvers ──────────────
            // Multi-eval solvers (Heun, UniPC, RK4, etc.) call model_fn
            // during their step, which writes new velocity into the vt
            // buffer. If both the "read-only original velocity" (arg 2)
            // and the "model_fn output buffer" (arg 7) point to the same
            // memory, solvers that reference the original velocity after
            // model_fn get corrupted values (e.g. Heun's correction term
            // becomes zero, silently degrading it to Euler).
            //
            // Fix: snapshot vt into vt_pre_solver and pass it as the
            // read-only arg 2. The live vt buffer (overwritten by
            // model_fn) is passed as the mutable arg 7 (vt_buf).
            const float * vt_readonly = vt.data();
            if (solver_plugin->needs_model) {
                vt_pre_solver.assign(vt.begin(), vt.end());
                vt_readonly = vt_pre_solver.data();
            }

            // ── Lua solver dispatch ────────────────────────────────────
            // The solver step function modifies xt[] in-place.
            // vt_readonly  = original velocity (preserved for read-only access)
            // vt.data()    = mutable buffer where model_fn writes results
            solver_state.step_index = step;
            lua_call_solver_step(
                *solver_plugin,
                xt.data(), vt_readonly, t_curr, t_next, n_total,
                solver_state, evaluate_velocity, vt.data(),
                g_hotstep_params.plugin_params
            );

            // ── DCW correction ──
            // Use the pre-solver velocity for multi-eval solvers (vt was
            // clobbered by intermediate model_fn calls during the step).
            sampler_apply_dcw(xt.data(), vt_readonly, N, T, Oc, t_curr, t_next, step, num_steps);

            // ── Post-step guidance hook (CFG-MP manifold projection etc.) ──
            if (guidance_plugin->has_post_step && do_cfg && step < num_steps - 1) {
                // Build eval lambdas that write to vt_cond / vt_uncond
                PostStepModelFn eval_cond_fn = [&](const float * xt_in, float t_v) {
                    eval_single_pass(xt_in, t_v, enc_cond_full, vt_cond.data());
                };
                PostStepModelFn eval_uncond_fn = [&](const float * xt_in, float t_v) {
                    eval_single_pass(xt_in, t_v, enc_uncond_full, vt_uncond.data());
                };
                lua_call_post_step(
                    *guidance_plugin,
                    xt.data(), t_next, n_total,
                    eval_cond_fn, eval_uncond_fn,
                    vt_cond.data(), vt_uncond.data(),
                    g_ctx, g_hotstep_params.plugin_params
                );
            }

            // ── Repaint injection ──
            sampler_repaint_inject(xt.data(), noise, repaint_src, N, T, Oc,
                                   repaint_t0, repaint_t1, repaint_injection_ratio,
                                   step, num_steps, t_next);
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

        fprintf(stderr, "[DiT] Step %d/%d t=%.3f [%s]%s\n", step + 1, num_steps, t_curr,
                solver_plugin->display_name.c_str(),
                (!step_computes[step] && has_cached_vt) ? " (cached)" : "");
    }
    } // else (per-step loop)

    // Boundary blend: smooth repaint zone edges in latent space.
    sampler_repaint_blend(output, repaint_src, N, T, Oc,
                          repaint_t0, repaint_t1, repaint_crossfade_frames);

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

// ── Linker sentinel for upstream-sync clobber detection ──────────────
// This symbol has external linkage and is compiled into whatever TU
// includes this header (pipeline-synth-ops.cpp).  hot-step-server.cpp
// references it via extern — if an upstream sync replaces the include
// with dit-sampler.h, this symbol vanishes and the linker fails with:
//     "unresolved external symbol hotstep_sampler_linked_"
// That build error is the signal to re-hook hot-step-sampler.h.
#if defined(_MSC_VER)
__declspec(selectany) int hotstep_sampler_linked_ = 1;
#else
__attribute__((weak)) int hotstep_sampler_linked_ = 1;
#endif
