// pipeline-synth-ops.cpp: primitive operations of the synthesis pipeline
//
// Each op takes AceSynth (the pipeline context) and SynthState (the transient
// job state). See pipeline-synth-ops.h for the per-op contract and
// pipeline-synth-impl.h for the struct layouts.

#include "pipeline-synth-ops.h"

#include "dit-sampler.h"
#include "philox.h"
#include "pipeline-synth-impl.h"
#include "schedulers/scheduler-registry.h"
#include "vae-enc.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int FRAMES_PER_SECOND = 25;

static std::vector<int> parse_codes_string(const std::string & s) {
    std::vector<int> codes;
    if (s.empty()) {
        return codes;
    }
    const char * p = s.c_str();
    while (*p) {
        while (*p == ',' || *p == ' ') {
            p++;
        }
        if (!*p) {
            break;
        }
        codes.push_back(atoi(p));
        while (*p && *p != ',') {
            p++;
        }
    }
    return codes;
}

// ops_encode_src
int ops_encode_src(const AceSynth * ctx, const float * src_audio, int src_len, SynthState & s) {
    // Cover mode: load VAE encoder and encode source audio
    s.have_cover = false;
    s.T_cover    = 0;
    if (src_audio && src_len > 0) {
        s.timer.reset();
        int T_audio = src_len;

        VAEEncoder vae_enc = {};
        vae_enc_load(&vae_enc, ctx->params.vae_path);
        int max_T_lat = (T_audio / 1920) + 64;
        s.cover_latents.resize(max_T_lat * 64);

        s.T_cover = vae_enc_encode_tiled(&vae_enc, src_audio, T_audio, s.cover_latents.data(), max_T_lat,
                                         ctx->params.vae_chunk, ctx->params.vae_overlap);
        vae_enc_free(&vae_enc);
        if (s.T_cover < 0) {
            fprintf(stderr, "[Encode-Src] FATAL: encode failed\n");
            return -1;
        }
        s.cover_latents.resize(s.T_cover * 64);
        fprintf(stderr, "[Encode-Src] Encoded: T_cover=%d (%.2fs), %.1f ms\n", s.T_cover,
                (float) s.T_cover * 1920.0f / 48000.0f, s.timer.ms());

        s.have_cover = true;
    }

    return 0;
}

// ops_fsq_roundtrip
void ops_fsq_roundtrip(AceSynth * ctx, SynthState & s) {
    // FSQ roundtrip for cover: tokenize (25Hz->5Hz) + detokenize (5Hz->25Hz).
    // The lossy 5:1 temporal compression destroys micro-timings, ornaments and
    // transients. The DiT receives degraded latents and diverges from the source,
    // producing a free reinterpretation rather than a close remix.
    // cover-nofsq skips this call and feeds clean 25Hz VAE latents directly,
    // producing remixes that stay close to the source.
    // Other tasks (lego, extract, repaint, complete) also use clean latents.
    if (s.have_cover && ctx->have_tok && ctx->have_detok) {
        s.timer.reset();
        int              T_5Hz = (s.T_cover + 4) / 5;
        std::vector<int> codes(T_5Hz);
        int              T_5Hz_actual =
            tok_ggml_encode(&ctx->tok, s.cover_latents.data(), s.T_cover, codes.data(), ctx->silence_full.data());
        if (T_5Hz_actual > 0) {
            int                T_25Hz_rt = T_5Hz_actual * 5;
            std::vector<float> rt_latents(T_25Hz_rt * 64);
            int                ret = detok_ggml_decode(&ctx->detok, codes.data(), T_5Hz_actual, rt_latents.data());
            if (ret >= 0) {
                int copy_T = T_25Hz_rt < s.T_cover ? T_25Hz_rt : s.T_cover;
                memcpy(s.cover_latents.data(), rt_latents.data(), (size_t) copy_T * 64 * sizeof(float));
                fprintf(stderr, "[FSQ-Roundtrip] %d->%d->%d frames, %.1f ms\n", s.T_cover, T_5Hz_actual, copy_T,
                        s.timer.ms());
            }
        }
    }
}

// ops_resolve_params
int ops_resolve_params(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Extract shared params from first request
    s.duration = s.rr.duration > 0 ? s.rr.duration : 30.0f;

    // Resolve DiT sampling params: 0 = auto-detect from model type.
    // Turbo: 8 steps, guidance=1.0, s.shift=3.0
    // Base/SFT: 50 steps, guidance=1.0, s.shift=1.0
    s.num_steps      = s.rr.inference_steps;
    s.guidance_scale = s.rr.guidance_scale;
    s.shift          = s.rr.shift;

    if (s.num_steps <= 0) {
        s.num_steps = ctx->is_turbo ? 8 : 50;
    }
    if (s.num_steps > 100) {
        fprintf(stderr, "[Resolve-Params] WARNING: inference_steps %d clamped to 100\n", s.num_steps);
        s.num_steps = 100;
    }

    if (s.guidance_scale <= 0.0f) {
        s.guidance_scale = 1.0f;
    } else if (ctx->is_turbo && s.guidance_scale > 1.0f) {
        // NOTE: bare turbo models ignore CFG (trained with guidance=1.0), but SFT/turbo merges
        // can benefit from higher guidance. Log a warning but respect the user's setting.
        fprintf(stderr, "[Resolve-Params] NOTE: turbo-flagged model with guidance_scale=%.1f "
                        "(bare turbo ignores CFG; merge models may use it)\n",
                s.guidance_scale);
    }

    if (s.shift == -1.0f) {
        // Auto shift — adaptive based on duration + step count.
        // Python: compute_dynamic_shift(base_shift=3.0, audio_duration, num_steps)
        float dur    = s.duration > 0.0f ? s.duration : 60.0f;
        float dur_f  = 1.0f + 0.15f * ((dur - 60.0f) / 60.0f);
        dur_f        = fmaxf(0.8f, fminf(1.4f, dur_f));
        float step_f = 1.0f + 0.1f * ((30.0f - (float) s.num_steps) / 30.0f);
        step_f       = fmaxf(0.8f, fminf(1.4f, step_f));
        s.shift      = fmaxf(1.0f, fminf(6.0f, 3.0f * dur_f * step_f));
        fprintf(stderr, "[Resolve-Params] Auto shift: duration=%.0fs, steps=%d → shift=%.3f\n",
                dur, s.num_steps, s.shift);
    } else if (s.shift <= 0.0f) {
        s.shift = ctx->is_turbo ? 3.0f : 1.0f;
    }

    // Audio codes: scan all requests to determine s.T from the longest code set.
    // Per-batch codes are decoded in the s.context building loop below.
    // Shorter code sets are padded with silence, longer ones are never truncated.
    s.max_codes_len = 0;
    s.have_codes    = false;
    for (int b = 0; b < batch_n; b++) {
        std::vector<int> cb = parse_codes_string(reqs[b].audio_codes);
        if ((int) cb.size() > s.max_codes_len) {
            s.max_codes_len = (int) cb.size();
        }
        if (!cb.empty()) {
            s.have_codes = true;
        }
    }
    if (s.have_codes) {
        fprintf(stderr, "[Resolve-Params] max audio codes across batch: %d (%.1fs @ 5Hz)\n", s.max_codes_len,
                (float) s.max_codes_len / 5.0f);
    }
    if (s.have_codes && !ctx->have_detok) {
        fprintf(stderr, "[Resolve-Params] FATAL: detokenizer not found\n");
        return -1;
    }

    // ── Solver sub-parameters ────────────────────────────────────────────
    s.stork_substeps    = s.rr.stork_substeps > 0    ? s.rr.stork_substeps    : 10;
    s.beat_stability    = s.rr.beat_stability >= 0.0f ? s.rr.beat_stability    : 0.25f;
    s.frequency_damping = s.rr.frequency_damping >= 0.0f ? s.rr.frequency_damping : 0.4f;
    s.temporal_smoothing = s.rr.temporal_smoothing >= 0.0f ? s.rr.temporal_smoothing : 0.13f;

    // ── Guidance sub-parameters ──────────────────────────────────────────
    s.apg_momentum       = s.rr.apg_momentum > 0.0f       ? s.rr.apg_momentum       : 0.75f;
    s.apg_norm_threshold = s.rr.apg_norm_threshold > 0.0f ? s.rr.apg_norm_threshold : 2.5f;

    if (s.stork_substeps != 10) {
        fprintf(stderr, "[Resolve-Params] stork_substeps: %d\n", s.stork_substeps);
    }
    if (s.beat_stability != 0.25f || s.frequency_damping != 0.4f || s.temporal_smoothing != 0.13f) {
        fprintf(stderr, "[Resolve-Params] jkass: beat=%.2f freq_damp=%.1f temporal=%.2f\n",
                s.beat_stability, s.frequency_damping, s.temporal_smoothing);
    }
    if (s.apg_momentum != 0.75f || s.apg_norm_threshold != 2.5f) {
        fprintf(stderr, "[Resolve-Params] apg: momentum=%.2f norm_threshold=%.1f\n",
                s.apg_momentum, s.apg_norm_threshold);
    }

    return 0;
}

// ops_build_schedule — dispatches via scheduler registry
void ops_build_schedule(SynthState & s) {
    const std::string & ss = s.scheduler;
    s.schedule.resize(s.num_steps);

    // ── Composite scheduler: "composite:<A>+<B>:<crossover>:<split>" ─────
    // Generates both sub-schedules independently, then blends them.
    //   split:     fraction of steps where we transition from A to B (0.0–1.0)
    //   crossover: width of the blend zone as a fraction of total steps (0.0–1.0)
    // crossover=0 is a hard switch; crossover=1 blends across all steps.
    if (ss.rfind("composite:", 0) == 0) {
        // Parse: "composite:bong_tangent+linear:0.50:0.50"
        const char * body = ss.c_str() + 10;  // skip "composite:"
        const char * plus = strchr(body, '+');
        if (!plus) {
            fprintf(stderr, "[Build-Schedule] WARNING: malformed composite '%s' (no '+'), falling back to linear\n",
                    ss.c_str());
            scheduler_linear(s.schedule.data(), s.num_steps, s.shift);
            return;
        }

        std::string name_a(body, plus - body);
        const char * after_plus = plus + 1;

        // Find first ':' after the B name to separate B from crossover param
        const char * colon1 = strchr(after_plus, ':');
        std::string  name_b;
        float        crossover = 0.0f;
        float        split     = 0.5f;

        if (colon1) {
            name_b = std::string(after_plus, colon1 - after_plus);
            crossover = (float) atof(colon1 + 1);
            const char * colon2 = strchr(colon1 + 1, ':');
            if (colon2) {
                split = (float) atof(colon2 + 1);
            }
        } else {
            name_b = std::string(after_plus);
        }

        // Clamp params
        if (crossover < 0.0f) crossover = 0.0f;
        if (crossover > 1.0f) crossover = 1.0f;
        if (split < 0.0f) split = 0.0f;
        if (split > 1.0f) split = 1.0f;

        // Look up sub-schedulers
        const SchedulerInfo * sched_a = scheduler_lookup(name_a.c_str());
        const SchedulerInfo * sched_b = scheduler_lookup(name_b.c_str());
        if (!sched_a) {
            fprintf(stderr, "[Build-Schedule] WARNING: composite sub-scheduler '%s' unknown, using linear\n",
                    name_a.c_str());
            sched_a = scheduler_lookup("linear");
        }
        if (!sched_b) {
            fprintf(stderr, "[Build-Schedule] WARNING: composite sub-scheduler '%s' unknown, using linear\n",
                    name_b.c_str());
            sched_b = scheduler_lookup("linear");
        }

        // Generate both full schedules
        std::vector<float> sched_a_vals(s.num_steps);
        std::vector<float> sched_b_vals(s.num_steps);
        sched_a->fn(sched_a_vals.data(), s.num_steps, s.shift);
        sched_b->fn(sched_b_vals.data(), s.num_steps, s.shift);

        // Composite blend:
        //   pure A for steps before (split - crossover/2)
        //   pure B for steps after  (split + crossover/2)
        //   linear interpolation in the crossover zone
        float zone_lo = split - crossover * 0.5f;
        float zone_hi = split + crossover * 0.5f;

        for (int i = 0; i < s.num_steps; i++) {
            float frac = (float) i / (float) s.num_steps;
            float w;  // weight for schedule B (0.0 = pure A, 1.0 = pure B)

            if (crossover < 1e-6f || frac <= zone_lo) {
                w = (frac < split) ? 0.0f : 1.0f;
            } else if (frac >= zone_hi) {
                w = 1.0f;
            } else {
                w = (frac - zone_lo) / (zone_hi - zone_lo);
            }

            s.schedule[i] = (1.0f - w) * sched_a_vals[i] + w * sched_b_vals[i];
        }

        // Enforce monotonicity: schedule must be non-increasing
        for (int i = 1; i < s.num_steps; i++) {
            if (s.schedule[i] > s.schedule[i - 1]) {
                s.schedule[i] = s.schedule[i - 1];
            }
        }
        scheduler_clamp(s.schedule.data(), s.num_steps);

        fprintf(stderr, "[Build-Schedule] Composite: %s + %s (crossover=%.2f, split=%.2f), %d steps, shift=%.2f\n",
                sched_a->display_name, sched_b->display_name, crossover, split, s.num_steps, s.shift);
        fprintf(stderr, "[Build-Schedule]   t[0]=%.4f  t[%d]=%.4f  t[%d]=%.4f\n",
                s.schedule[0], s.num_steps / 2, s.schedule[s.num_steps / 2],
                s.num_steps - 1, s.schedule[s.num_steps - 1]);
        return;
    }

    // ── Standard lookup with fallback ────────────────────────────────────
    const SchedulerInfo * sched = scheduler_lookup(ss.c_str());
    if (!sched) {
        fprintf(stderr, "[Build-Schedule] WARNING: unknown scheduler '%s', falling back to linear\n",
                ss.c_str());
        sched = scheduler_lookup("linear");
    }

    // ── Handle parameterized scheduler strings ───────────────────────────
    // "power:<exp>"  → power-law with custom exponent
    // "beta:<a>:<b>" → beta distribution with custom alpha/beta
    // Otherwise: use the base scheduler function as-is.
    if (ss.rfind("power:", 0) == 0 && ss.size() > 6) {
        // Custom power exponent
        float p = (float) atof(ss.c_str() + 6);
        if (p < 0.1f) p = 2.0f;
        for (int i = 0; i < s.num_steps; i++) {
            float frac = (float) i / (float) s.num_steps;
            s.schedule[i] = powf(1.0f - frac, p);
        }
        scheduler_clamp(s.schedule.data(), s.num_steps);
        scheduler_apply_shift(s.schedule.data(), s.num_steps, s.shift);
        fprintf(stderr, "[Build-Schedule] Power (p=%.2f), %d steps, shift=%.2f\n", p, s.num_steps, s.shift);
    } else if (ss.rfind("beta:", 0) == 0 && ss.size() > 5) {
        // Custom beta distribution: "beta:<alpha>:<beta>"
        double alpha = 0.5, beta = 0.7;
        const char * p1 = ss.c_str() + 5;
        alpha = atof(p1);
        const char * colon = strchr(p1, ':');
        if (colon) beta = atof(colon + 1);
        if (alpha < 0.01) alpha = 0.5;
        if (beta < 0.01) beta = 0.7;
        scheduler_beta_custom(s.schedule.data(), s.num_steps, s.shift, alpha, beta);
        fprintf(stderr, "[Build-Schedule] Beta (α=%.2f, β=%.2f), %d steps, shift=%.2f\n",
                alpha, beta, s.num_steps, s.shift);
    } else {
        sched->fn(s.schedule.data(), s.num_steps, s.shift);
        fprintf(stderr, "[Build-Schedule] %s (%s), %d steps, shift=%.2f\n",
                sched->display_name, sched->name, s.num_steps, s.shift);
    }
}

// ops_resolve_T
int ops_resolve_T(const AceSynth * ctx, SynthState & s) {
    // s.T = number of 25Hz latent frames for DiT
    // Source tasks: from source audio. Codes: from code count. Else: from s.duration.
    if (s.use_source_context && s.have_cover) {
        s.T        = s.T_cover;
        // s.duration in metas must match actual source length, not JSON default
        s.duration = (float) s.T_cover / (float) FRAMES_PER_SECOND;
    } else if (s.have_codes) {
        s.T = s.max_codes_len * 5;
    } else if (s.use_source_context) {
        // source context requested but neither cover_latents nor codes available.
        // duration fallthrough would produce a meaningless T for source tasks.
        fprintf(stderr, "[Resolve-T] FATAL: use_source_context but no cover_latents and no audio_codes\n");
        return -1;
    } else {
        s.T = (int) (s.duration * FRAMES_PER_SECOND);
    }
    s.T     = ((s.T + ctx->dit_cfg.patch_size - 1) / ctx->dit_cfg.patch_size) * ctx->dit_cfg.patch_size;
    s.S     = s.T / ctx->dit_cfg.patch_size;
    s.enc_S = 0;

    fprintf(stderr, "[Resolve-T] T=%d, S=%d\n", s.T, s.S);
    fprintf(stderr, "[Resolve-T] seed=%lld, steps=%d, guidance=%.1f, shift=%.1f, duration=%.1fs\n",
            (long long) s.rr.seed, s.num_steps, s.guidance_scale, s.shift, s.duration);

    if (s.T > 15000) {
        fprintf(stderr, "[Resolve-T] ERROR: T=%d exceeds silence_latent max 15000, skipping\n", s.T);
        return -1;
    }

    return 0;
}

// ops_encode_timbre
void ops_encode_timbre(const AceSynth * ctx, const float * ref_audio, int ref_len, SynthState & s) {
    // Timbre features from ref_audio (independent of src_audio).
    // VAE-encode ref_audio and pass all frames to the timbre encoder.
    // NULL ref_audio = single silence frame (no timbre conditioning).
    if (ref_audio && ref_len > 0) {
        s.timer.reset();
        VAEEncoder ref_vae = {};
        vae_enc_load(&ref_vae, ctx->params.vae_path);
        int                max_T_ref = (ref_len / 1920) + 64;
        std::vector<float> ref_latents(max_T_ref * 64);
        int                T_ref = vae_enc_encode_tiled(&ref_vae, ref_audio, ref_len, ref_latents.data(), max_T_ref,
                                                        ctx->params.vae_chunk, ctx->params.vae_overlap);
        vae_enc_free(&ref_vae);
        if (T_ref < 0) {
            fprintf(stderr, "[Encode-Timbre] WARNING: ref_audio encode failed, using silence\n");
            s.S_ref_timbre = 1;
            s.timbre_feats.assign(ctx->silence_full.data(), ctx->silence_full.data() + 64);
        } else {
            s.S_ref_timbre = T_ref;
            s.timbre_feats.assign(ref_latents.data(), ref_latents.data() + (size_t) T_ref * 64);
            fprintf(stderr, "[Encode-Timbre] ref_audio: %d frames (%.1fs), %.1f ms\n", T_ref, (float) T_ref / 25.0f,
                    s.timer.ms());
        }
    } else {
        s.S_ref_timbre = 1;
        s.timbre_feats.assign(ctx->silence_full.data(), ctx->silence_full.data() + 64);
    }
}

// ops_encode_text
int ops_encode_text(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // 3. Per-batch text encoding.
    // Each batch element gets its own caption, lyrics, and metadata encoded independently.
    // TextEncoder + CondEncoder run in series (cheap: ~13ms per element).
    // Results are padded to s.max_enc_S with null_cond and stacked for a single DiT batch pass.
    int H_text = ctx->text_enc.cfg.hidden_size;        // 1024
    int H_cond = ctx->cond_enc.lyric_cfg.hidden_size;  // encoder hidden size (2048)

    // null_condition_emb cached on CPU at ace_synth_load. Empty when the model has none.
    s.null_cond_vec.resize(H_cond);
    if (!ctx->null_cond_cpu.empty()) {
        memcpy(s.null_cond_vec.data(), ctx->null_cond_cpu.data(), H_cond * sizeof(float));
    }

    // instruction_str must be set by the orchestrator. Empty means unknown task or bug.
    if (s.instruction_str.empty()) {
        fprintf(stderr, "[Encode-Text] FATAL: instruction_str is empty (unknown task or orchestrator bug)\n");
        return -1;
    }

    // encode each batch element independently
    s.per_enc.resize(batch_n);
    s.per_enc_S.resize(batch_n);

    for (int b = 0; b < batch_n; b++) {
        const AceRequest & rb = reqs[b];

        // per-batch metadata
        char bpm_b[16] = "N/A";
        if (rb.bpm > 0) {
            snprintf(bpm_b, sizeof(bpm_b), "%d", rb.bpm);
        }
        const char * keyscale_b = rb.keyscale.empty() ? "N/A" : rb.keyscale.c_str();
        const char * timesig_b  = rb.timesignature.empty() ? "N/A" : rb.timesignature.c_str();
        const char * language_b = rb.vocal_language.empty() ? "unknown" : rb.vocal_language.c_str();

        char metas_b[512];
        snprintf(metas_b, sizeof(metas_b), "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
                 bpm_b, timesig_b, keyscale_b, (int) s.duration);
        std::string text_str = std::string("# Instruction\n") + s.instruction_str + "\n\n" + "# Caption\n" +
                               rb.caption + "\n\n" + "# Metas\n" + metas_b + "<|endoftext|>\n";
        std::string lyric_str =
            std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + rb.lyrics + "<|endoftext|>";

        // tokenize
        auto text_ids  = bpe_encode(&ctx->bpe, text_str.c_str(), true);
        auto lyric_ids = bpe_encode(&ctx->bpe, lyric_str.c_str(), true);
        int  S_text    = (int) text_ids.size();
        int  S_lyric   = (int) lyric_ids.size();

        // TextEncoder forward
        std::vector<float> text_hidden(H_text * S_text);
        qwen3_forward(&ctx->text_enc, text_ids.data(), S_text, text_hidden.data());

        // lyric embedding (vocab lookup)
        std::vector<float> lyric_embed(H_text * S_lyric);
        qwen3_embed_lookup(&ctx->text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());

        // CondEncoder forward
        s.timer.reset();
        cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                          s.timbre_feats.data(), s.S_ref_timbre, s.per_enc[b], &s.per_enc_S[b]);
        fprintf(stderr, "[Encode-Text Batch%d] %d+%d tokens -> enc_S=%d, %.1f ms\n", b, S_text, S_lyric, s.per_enc_S[b],
                s.timer.ms());

        if (b == 0) {
            debug_dump_2d(&s.dbg, "text_hidden", text_hidden.data(), S_text, H_text);
            debug_dump_2d(&s.dbg, "lyric_embed", lyric_embed.data(), S_lyric, H_text);
            debug_dump_2d(&s.dbg, "enc_hidden", s.per_enc[b].data(), s.per_enc_S[b], H_cond);
        }
    }

    // second encoding pass using s.nc_instruction_str (set by orchestrator).
    // used after s.cover_steps when audio_cover_strength < 1.0 (s.context switches to silence).
    s.need_enc_switch = s.use_source_context && !s.is_repaint && !s.is_lego_region && s.rr.audio_cover_strength < 1.0f;
    s.per_enc_nc.resize(batch_n);
    s.per_enc_S_nc.assign(batch_n, 0);

    if (s.need_enc_switch) {
        for (int b = 0; b < batch_n; b++) {
            const AceRequest & rb = reqs[b];

            char bpm_b[16] = "N/A";
            if (rb.bpm > 0) {
                snprintf(bpm_b, sizeof(bpm_b), "%d", rb.bpm);
            }
            const char * keyscale_b = rb.keyscale.empty() ? "N/A" : rb.keyscale.c_str();
            const char * timesig_b  = rb.timesignature.empty() ? "N/A" : rb.timesignature.c_str();
            const char * language_b = rb.vocal_language.empty() ? "unknown" : rb.vocal_language.c_str();

            char metas_b[512];
            snprintf(metas_b, sizeof(metas_b),
                     "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n", bpm_b, timesig_b,
                     keyscale_b, (int) s.duration);
            std::string text_str = std::string("# Instruction\n") + s.nc_instruction_str + "\n\n" + "# Caption\n" +
                                   rb.caption + "\n\n" + "# Metas\n" + metas_b + "<|endoftext|>\n";
            std::string lyric_str =
                std::string("# Languages\n") + language_b + "\n\n# Lyric\n" + rb.lyrics + "<|endoftext|>";

            auto text_ids  = bpe_encode(&ctx->bpe, text_str.c_str(), true);
            auto lyric_ids = bpe_encode(&ctx->bpe, lyric_str.c_str(), true);
            int  S_text    = (int) text_ids.size();
            int  S_lyric   = (int) lyric_ids.size();

            std::vector<float> text_hidden(H_text * S_text);
            qwen3_forward(&ctx->text_enc, text_ids.data(), S_text, text_hidden.data());

            std::vector<float> lyric_embed(H_text * S_lyric);
            qwen3_embed_lookup(&ctx->text_enc, lyric_ids.data(), S_lyric, lyric_embed.data());

            cond_ggml_forward(&ctx->cond_enc, text_hidden.data(), S_text, lyric_embed.data(), S_lyric,
                              s.timbre_feats.data(), s.S_ref_timbre, s.per_enc_nc[b], &s.per_enc_S_nc[b]);
            fprintf(stderr, "[Encode-Text Batch%d] non-cover: %d+%d tokens -> enc_S=%d\n", b, S_text, S_lyric,
                    s.per_enc_S_nc[b]);
        }
    }

    // find max s.enc_S across both encodings (cover + text2music),
    // pad shorter encodings with null_cond, stack into [H, s.max_enc_S, N]
    s.max_enc_S = 0;
    for (int b = 0; b < batch_n; b++) {
        if (s.per_enc_S[b] > s.max_enc_S) {
            s.max_enc_S = s.per_enc_S[b];
        }
        if (s.need_enc_switch && s.per_enc_S_nc[b] > s.max_enc_S) {
            s.max_enc_S = s.per_enc_S_nc[b];
        }
    }
    s.enc_S = s.max_enc_S;

    s.enc_hidden.resize(H_cond * s.max_enc_S * batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.enc_hidden.data() + b * s.max_enc_S * H_cond;
        memcpy(dst, s.per_enc[b].data(), (size_t) s.per_enc_S[b] * H_cond * sizeof(float));
        for (int si = s.per_enc_S[b]; si < s.max_enc_S; si++) {
            memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
        }
    }

    // pad and stack text2music encoding (same s.max_enc_S for graph compatibility)
    if (s.need_enc_switch) {
        s.enc_hidden_nc.resize(H_cond * s.max_enc_S * batch_n);
        s.per_enc_S_nc_final.resize(batch_n);
        for (int b = 0; b < batch_n; b++) {
            float * dst = s.enc_hidden_nc.data() + b * s.max_enc_S * H_cond;
            memcpy(dst, s.per_enc_nc[b].data(), (size_t) s.per_enc_S_nc[b] * H_cond * sizeof(float));
            for (int si = s.per_enc_S_nc[b]; si < s.max_enc_S; si++) {
                memcpy(dst + si * H_cond, s.null_cond_vec.data(), H_cond * sizeof(float));
            }
            s.per_enc_S_nc_final[b] = s.per_enc_S_nc[b];
        }
    }

    if (batch_n > 1) {
        fprintf(stderr, "[Encode-Text] Per-batch encoding done: max_enc_S=%d\n", s.max_enc_S);
    }

    return 0;
}

// ops_build_context
int ops_build_context(AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Build s.context: [batch_n, s.T, s.ctx_ch] = src_latents[64] + chunk_mask[64]
    // Cover/Lego/Repaint: shared s.context replicated (s.cover_latents from src_audio).
    // Passthrough: per-batch detokenized FSQ codes + silence padding, mask = 1.0.
    // Text2music: silence only, mask = 1.0.
    s.repaint_t0 = 0, s.repaint_t1 = 0;
    if (s.is_repaint) {
        s.repaint_t0 = (int) (s.rs * 48000.0f / 1920.0f);
        s.repaint_t1 = (int) (s.re * 48000.0f / 1920.0f);
        if (s.repaint_t0 < 0) {
            s.repaint_t0 = 0;
        }
        if (s.repaint_t1 > s.T) {
            s.repaint_t1 = s.T;
        }
        if (s.repaint_t0 > s.T) {
            s.repaint_t0 = s.T;
        }
        fprintf(stderr, "[Build-Context] Latent frames: [%d, %d) / %d\n", s.repaint_t0, s.repaint_t1, s.T);
    }

    s.context.resize(batch_n * s.T * s.ctx_ch);

    if (s.use_source_context && s.have_cover) {
        // Cover/Lego/Repaint: build once, replicate (s.cover_latents are shared)
        std::vector<float> context_single(s.T * s.ctx_ch);
        for (int t = 0; t < s.T; t++) {
            bool          in_region = (s.is_repaint || s.is_lego_region) && t >= s.repaint_t0 && t < s.repaint_t1;
            // repaint silences the zone (DiT generates fresh there).
            // lego keeps full cover everywhere (DiT hears the whole backing track).
            const float * src;
            if (s.is_repaint && in_region) {
                src = ctx->silence_full.data() + t * s.Oc;
            } else {
                src = (t < s.T_cover) ? s.cover_latents.data() + t * s.Oc : ctx->silence_full.data() + t * s.Oc;
            }
            // region tasks: explicit 0/1 mask. all others: 1.0 (training distribution).
            float mask_val;
            if (s.is_repaint || s.is_lego_region) {
                mask_val = in_region ? 1.0f : 0.0f;
            } else {
                mask_val = 1.0f;  // training distribution: only 0/1 seen during training
            }
            for (int c = 0; c < s.Oc; c++) {
                context_single[t * s.ctx_ch + c] = src[c];
            }
            for (int c = 0; c < s.Oc; c++) {
                context_single[t * s.ctx_ch + s.Oc + c] = mask_val;
            }
        }
        for (int b = 0; b < batch_n; b++) {
            memcpy(s.context.data() + b * s.T * s.ctx_ch, context_single.data(), s.T * s.ctx_ch * sizeof(float));
        }
    } else {
        // Per-batch context from audio_codes or silence (text2music).
        // use_source_context with neither cover nor codes is an invalid state:
        // the orchestrator promised source context but provided nothing to condition on.
        if (s.use_source_context && !s.have_codes) {
            fprintf(stderr, "[Build-Context] FATAL: use_source_context but no cover_latents and no audio_codes\n");
            return -1;
        }

        // Text2music / codes passthrough: per-batch context with per-batch audio_codes
        for (int b = 0; b < batch_n; b++) {
            float * ctx_dst = s.context.data() + b * s.T * s.ctx_ch;

            // decode this batch item's audio codes (if any)
            int                decoded_T = 0;
            std::vector<float> decoded_latents;
            std::vector<int>   codes_b = parse_codes_string(reqs[b].audio_codes);
            if (!codes_b.empty()) {
                s.timer.reset();
                int T_5Hz        = (int) codes_b.size();
                int T_25Hz_codes = T_5Hz * 5;
                decoded_latents.resize(T_25Hz_codes * s.Oc);

                int ret = detok_ggml_decode(&ctx->detok, codes_b.data(), T_5Hz, decoded_latents.data());
                if (ret < 0) {
                    fprintf(stderr, "[Build-Context Batch%d] FATAL: detokenizer decode failed\n", b);
                    return -1;
                }
                fprintf(stderr, "[Build-Context Batch%d] Detokenizer: %.1f ms, %d codes\n", b, s.timer.ms(), T_5Hz);

                decoded_T = T_25Hz_codes < s.T ? T_25Hz_codes : s.T;
                if (b == 0) {
                    debug_dump_2d(&s.dbg, "detok_output", decoded_latents.data(), T_25Hz_codes, s.Oc);
                }
            }

            // fill s.context: decoded latents then silence, mask = 1.0 (training distribution)
            for (int t = 0; t < s.T; t++) {
                const float * src = (t < decoded_T) ? decoded_latents.data() + t * s.Oc :
                                                      ctx->silence_full.data() + (t - decoded_T) * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    ctx_dst[t * s.ctx_ch + c] = src[c];
                }
                for (int c = 0; c < s.Oc; c++) {
                    ctx_dst[t * s.ctx_ch + s.Oc + c] = 1.0f;
                }
            }
        }
    }

    return 0;
}

// ops_build_context_silence
void ops_build_context_silence(const AceSynth * ctx, int batch_n, SynthState & s) {
    // Cover mode: build silence s.context for audio_cover_strength switching
    // When step >= s.cover_steps, DiT switches from cover s.context to silence s.context
    // Repaint/lego_region: mask handles region; s.context switch never applies
    s.cover_steps = -1;
    if (s.use_source_context && !s.is_repaint && !s.is_lego_region) {
        float cover_strength = s.rr.audio_cover_strength;
        if (cover_strength < 1.0f) {
            // Build silence s.context: all frames use silence_latent
            std::vector<float> silence_single(s.T * s.ctx_ch);
            for (int t = 0; t < s.T; t++) {
                const float * src = ctx->silence_full.data() + t * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    silence_single[t * s.ctx_ch + c] = src[c];
                }
                for (int c = 0; c < s.Oc; c++) {
                    silence_single[t * s.ctx_ch + s.Oc + c] = 1.0f;
                }
            }
            s.context_silence.resize(batch_n * s.T * s.ctx_ch);
            for (int b = 0; b < batch_n; b++) {
                memcpy(s.context_silence.data() + b * s.T * s.ctx_ch, silence_single.data(),
                       s.T * s.ctx_ch * sizeof(float));
            }
            s.cover_steps = (int) ((float) s.num_steps * cover_strength);
            fprintf(stderr, "[Context-Silence] audio_cover_strength=%.2f -> switch at step %d/%d\n", cover_strength,
                    s.cover_steps, s.num_steps);
        }
    }
}

// ops_init_noise_and_repaint
void ops_init_noise_and_repaint(const AceSynth * ctx, const AceRequest * reqs, int batch_n, SynthState & s) {
    // Generate N s.noise samples (Philox4x32-10, matches torch.randn on CUDA with bf16).
    // Each batch item uses its own seed from the request.
    s.noise.resize(batch_n * s.Oc * s.T);
    s.seeds.resize(batch_n);
    for (int b = 0; b < batch_n; b++) {
        float * dst = s.noise.data() + b * s.Oc * s.T;
        s.seeds[b]  = reqs[b].seed;
        philox_randn(reqs[b].seed, dst, s.Oc * s.T, /*bf16_round=*/true);
        fprintf(stderr, "[Init-Noise Batch%d] Philox noise seed=%lld, [%d, %d] solver=%s\n", b, (long long) reqs[b].seed, s.T,
                s.Oc, s.solver.c_str());
    }

    // cover_noise_strength: blend initial noise with clean source latents.
    // xt = nearest_t * noise + (1 - nearest_t) * clean_latents, then truncate schedule.
    // the FSQ roundtrip degrades cover_latents for context conditioning, but noise
    // blending needs the original clean VAE latents. noise_blend_latents holds the
    // clean copy when FSQ was applied; otherwise fall back to cover_latents (already clean).
    if (s.use_source_context && s.have_cover && s.rr.cover_noise_strength > 0.0f) {
        const std::vector<float> & blend_src = s.noise_blend_latents.empty() ? s.cover_latents : s.noise_blend_latents;
        float                      effective_noise_level = 1.0f - s.rr.cover_noise_strength;
        // find nearest timestep in s.schedule
        int                        start_idx             = 0;
        float                      best_dist             = fabsf(s.schedule[0] - effective_noise_level);
        for (int i = 1; i < s.num_steps; i++) {
            float dist = fabsf(s.schedule[i] - effective_noise_level);
            if (dist < best_dist) {
                best_dist = dist;
                start_idx = i;
            }
        }
        float nearest_t = s.schedule[start_idx];
        // blend: xt = nearest_t * s.noise + (1 - nearest_t) * clean_latents
        for (int b = 0; b < batch_n; b++) {
            float * n = s.noise.data() + b * s.Oc * s.T;
            for (int t = 0; t < s.T; t++) {
                int           t_src = t < s.T_cover ? t : s.T_cover - 1;
                const float * src   = blend_src.data() + t_src * s.Oc;
                for (int c = 0; c < s.Oc; c++) {
                    int idx = t * s.Oc + c;
                    n[idx]  = nearest_t * n[idx] + (1.0f - nearest_t) * src[c];
                }
            }
        }
        // truncate s.schedule
        s.schedule.erase(s.schedule.begin(), s.schedule.begin() + start_idx);
        s.num_steps = (int) s.schedule.size();
        // recalculate s.cover_steps with remaining steps
        if (s.cover_steps >= 0) {
            s.cover_steps = (int) ((float) s.num_steps * s.rr.audio_cover_strength);
        }
        fprintf(stderr,
                "[Init-Noise] cover_noise_strength=%.2f -> noise_level=%.4f, nearest_t=%.4f, remaining_steps=%d\n",
                s.rr.cover_noise_strength, effective_noise_level, nearest_t, s.num_steps);
    }

    // DiT Generate
    s.output.resize(batch_n * s.Oc * s.T);

    // Per-batch sequence lengths for attention padding masks.
    // Within a synth_batch_size group, all elements share the same s.T (same codes),
    // so s.per_S[b] = s.S for all b. The s.per_enc_S[] array has real encoder lengths
    // from per-batch text encoding above.
    // These become meaningful when the server/CLI batches requests with different s.T.
    s.per_S.assign(batch_n, s.S);

    // Debug dumps (sample 0)
    debug_dump_2d(&s.dbg, "noise", s.noise.data(), s.T, s.Oc);
    debug_dump_2d(&s.dbg, "context", s.context.data(), s.T, s.ctx_ch);

    fprintf(stderr, "[Init-Noise] Starting: T=%d, S=%d, enc_S=%d, steps=%d, batch=%d%s\n", s.T, s.S, s.enc_S,
            s.num_steps, batch_n, s.use_source_context ? " (cover)" : "");

    // repaint/lego-region injection buffer: full cover latents padded with silence.
    // used for step injection and boundary blend in both repaint and lego-region modes.
    if (s.is_repaint || s.is_lego_region) {
        s.repaint_src.resize(s.T * s.Oc);
        for (int t = 0; t < s.T; t++) {
            const float * src =
                (t < s.T_cover) ? s.cover_latents.data() + t * s.Oc : ctx->silence_full.data() + t * s.Oc;
            memcpy(s.repaint_src.data() + t * s.Oc, src, s.Oc * sizeof(float));
        }
    }
}

// ops_dit_generate
int ops_dit_generate(AceSynth * ctx, int batch_n, SynthState & s, bool (*cancel)(void *), void * cancel_data) {
    s.timer.reset();
    int dit_rc = dit_ggml_generate(
        &ctx->dit, s.noise.data(), s.context.data(), s.enc_hidden.data(), s.enc_S, s.T, batch_n, s.num_steps,
        s.schedule.data(), s.output.data(), s.guidance_scale, &s.dbg,
        s.context_silence.empty() ? nullptr : s.context_silence.data(), s.cover_steps, cancel, cancel_data,
        s.per_S.data(), s.per_enc_S.data(), s.enc_hidden_nc.empty() ? nullptr : s.enc_hidden_nc.data(),
        s.per_enc_S_nc_final.empty() ? nullptr : s.per_enc_S_nc_final.data(),
        s.repaint_src.empty() ? nullptr : s.repaint_src.data(), s.repaint_t0, s.repaint_t1, s.repaint_injection_ratio,
        s.repaint_crossfade_frames, s.solver.c_str(), s.seeds.data(), ctx->params.use_batch_cfg,
        s.guidance_mode.c_str(), s.apg_momentum, s.apg_norm_threshold,
        s.stork_substeps, s.beat_stability, s.frequency_damping, s.temporal_smoothing);
    if (dit_rc != 0) {
        return -1;
    }
    fprintf(stderr, "[DiT-Generate] Total: %.1f ms (%.1f ms/sample)\n", s.timer.ms(), s.timer.ms() / batch_n);

    debug_dump_2d(&s.dbg, "dit_output", s.output.data(), s.T, s.Oc);
    return 0;
}

// ops_vae_decode_and_splice
int ops_vae_decode_and_splice(AceSynth *    ctx,
                              int           batch_n,
                              AceAudio *    out,
                              SynthState &  s,
                              const float * src_audio,
                              int           src_len,
                              bool (*cancel)(void *),
                              void * cancel_data) {
    int                T_latent    = s.T;
    int                T_audio_max = T_latent * 1920;
    std::vector<float> audio(2 * T_audio_max);

    for (int b = 0; b < batch_n; b++) {
        float * dit_out = s.output.data() + b * s.Oc * s.T;

        s.timer.reset();
        int T_audio = vae_ggml_decode_tiled(&ctx->vae, dit_out, T_latent, audio.data(), T_audio_max,
                                            ctx->params.vae_chunk, ctx->params.vae_overlap, cancel, cancel_data);
        if (T_audio < 0) {
            // check if this was a cancellation or a real error
            if (cancel && cancel(cancel_data)) {
                fprintf(stderr, "[VAE-Decode Batch%d] Cancelled\n", b);
                return -1;
            }
            fprintf(stderr, "[VAE-Decode Batch%d] ERROR: decode failed\n", b);
            out[b].samples     = NULL;
            out[b].n_samples   = 0;
            out[b].sample_rate = 48000;
            continue;
        }
        fprintf(stderr, "[VAE-Decode Batch%d] Decode: %.1f ms\n", b, s.timer.ms());

        if (b == 0) {
            debug_dump_2d(&s.dbg, "vae_audio", audio.data(), 2, T_audio);
        }

        // Copy to s.output buffer
        int n_total    = 2 * T_audio;
        out[b].samples = (float *) malloc((size_t) n_total * sizeof(float));
        memcpy(out[b].samples, audio.data(), (size_t) n_total * sizeof(float));
        out[b].n_samples   = T_audio;
        out[b].sample_rate = 48000;

        // Waveform splice: replace non-repaint regions with original source audio.
        // Python: apply_repaint_waveform_splice (when mode != aggressive)
        // mask[s] = 1.0 inside repaint region, 0.0 outside, linear ramp at edges.
        // result = mask * pred + (1-mask) * src  [planar stereo: L:s.T, R:s.T]
        bool have_repaint_region = s.is_repaint || s.is_lego_region;
        if (have_repaint_region && src_audio) {  // always splice (non-aggressive)
            int T_splice = out[b].n_samples < src_len ? out[b].n_samples : src_len;
            int start_s  = (int) (s.rs * 48000.0f);
            int end_s    = (int) (s.re * 48000.0f);
            start_s      = start_s < 0 ? 0 : (start_s > T_splice ? T_splice : start_s);
            end_s        = end_s < start_s ? start_s : (end_s > T_splice ? T_splice : end_s);
            // skip splice if region covers everything
            if (start_s > 0 || end_s < T_splice) {
                int cf_s       = (int) (s.repaint_wav_cf_sec * 48000.0f);
                int fade_start = start_s - cf_s > 0 ? start_s - cf_s : 0;
                int fade_end   = end_s + cf_s < T_splice ? end_s + cf_s : T_splice;
                for (int ch = 0; ch < 2; ch++) {
                    float * pred = out[b].samples + (size_t) ch * out[b].n_samples;
                    // src_audio is interleaved [L0,R0,L1,R1,...]: access via s*2+ch
                    for (int si = 0; si < fade_start; si++) {
                        pred[si] = src_audio[(size_t) si * 2 + ch];
                    }
                    for (int si = fade_start; si < start_s; si++) {
                        // left ramp: 0->1 toward repaint zone (excl endpoints)
                        int   rl  = start_s - fade_start;
                        float m   = (float) (si - fade_start + 1) / (float) (rl + 1);
                        float src = src_audio[(size_t) si * 2 + ch];
                        pred[si]  = m * pred[si] + (1.0f - m) * src;
                    }
                    // [start_s, end_s): keep generated s.output as-is (mask=1)
                    for (int si = end_s; si < fade_end; si++) {
                        // right ramp: 1->0 away from repaint zone (excl endpoints)
                        int   rl  = fade_end - end_s;
                        float m   = (float) (fade_end - si) / (float) (rl + 1);
                        float src = src_audio[(size_t) si * 2 + ch];
                        pred[si]  = m * pred[si] + (1.0f - m) * src;
                    }
                    for (int si = fade_end; si < T_splice; si++) {
                        pred[si] = src_audio[(size_t) si * 2 + ch];
                    }
                }
                fprintf(stderr, "[WAV-Splice Batch%d] wav splice %.1fs-%.1fs cf=%.0fms\n", b, s.rs, s.re,
                        s.repaint_wav_cf_sec * 1000.0f);
            }
        }
    }
    return 0;
}
