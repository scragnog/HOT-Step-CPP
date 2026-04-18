// pipeline-synth.cpp: ACE-Step synthesis pipeline implementation
//
// Resident modules loaded at init: TextEnc, CondEnc, FSQ tok, FSQ detok, BPE.
// DiT and VAE decoder are loaded on demand via explicit phase calls.
// Phase 1 (DiT) and phase 2 (VAE) run on independent jobs so the caller can
// batch many jobs through the DiT while it is resident, unload it, then feed
// the latents to the VAE decoder with the largest tiles the GPU can hold.

#include "pipeline-synth.h"

#include "gguf-weights.h"
#include "pipeline-synth-impl.h"
#include "pipeline-synth-ops.h"
#include "task-types.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

void ace_synth_default_params(AceSynthParams * p) {
    p->text_encoder_path = NULL;
    p->dit_path          = NULL;
    p->vae_path          = NULL;
    p->adapter_path      = NULL;
    p->adapter_scale     = 1.0f;
    p->adapter_group_self_attn   = 1.0f;
    p->adapter_group_cross_attn  = 1.0f;
    p->adapter_group_mlp         = 1.0f;
    p->adapter_group_cond_embed  = 1.0f;
    p->adapter_mode      = NULL;  // NULL or "merge" = classic merge, "runtime" = runtime LoRA
    p->use_fa            = true;
    p->clamp_fp16        = false;
    p->use_batch_cfg     = true;
    p->vae_chunk         = 256;
    p->vae_overlap       = 64;
    p->dump_dir          = NULL;
}

// Read silence_latent, null_condition_emb and is_turbo from the DiT GGUF
// without touching the big tensors. Called once at ace_synth_load.
// Returns false on any missing metadata or tensor read error.
static bool load_dit_cpu_side(AceSynth * ctx, const char * dit_path) {
    GGUFModel gf = {};
    if (!gf_load(&gf, dit_path)) {
        fprintf(stderr, "[Synth-Load] FATAL: cannot reopen %s for metadata\n", dit_path);
        return false;
    }

    ctx->is_turbo = gf_get_bool(gf, "acestep.is_turbo");

    // silence_latent: [15000, 64] f32 needed by FSQ tokenizer padding and by
    // repaint src buffer construction when T extends past the source coverage.
    const void * sl_data = gf_get_data(gf, "silence_latent");
    if (!sl_data) {
        fprintf(stderr, "[Synth-Load] FATAL: silence_latent not found in %s\n", dit_path);
        gf_close(&gf);
        return false;
    }
    ctx->silence_full.resize(15000 * 64);
    memcpy(ctx->silence_full.data(), sl_data, 15000 * 64 * sizeof(float));
    fprintf(stderr, "[Synth-Load] silence_latent: [15000, 64] from GGUF\n");

    // null_condition_emb: [H_cond] BF16 or F32. Absent on some trained variants.
    // Used by ops_encode_text to pad shorter encodings up to max_enc_S.
    struct ggml_tensor * nce_meta = ggml_get_tensor(gf.meta, "null_condition_emb");
    if (nce_meta) {
        int          emb_n = (int) ggml_nelements(nce_meta);
        const void * raw   = gf_get_data(gf, "null_condition_emb");
        ctx->null_cond_cpu.resize(emb_n);
        if (nce_meta->type == GGML_TYPE_BF16) {
            const uint16_t * s = (const uint16_t *) raw;
            for (int i = 0; i < emb_n; i++) {
                uint32_t w = (uint32_t) s[i] << 16;
                memcpy(&ctx->null_cond_cpu[i], &w, 4);
            }
        } else if (nce_meta->type == GGML_TYPE_F32) {
            memcpy(ctx->null_cond_cpu.data(), raw, emb_n * sizeof(float));
        } else {
            fprintf(stderr, "[Synth-Load] FATAL: null_condition_emb unexpected type %d\n", nce_meta->type);
            gf_close(&gf);
            return false;
        }
        fprintf(stderr, "[Synth-Load] null_condition_emb: [%d] cached (CFG available)\n", emb_n);
    } else {
        ctx->null_cond_cpu.clear();
    }

    gf_close(&gf);
    return true;
}

AceSynth * ace_synth_load(const AceSynthParams * params) {
    if (!params->dit_path) {
        fprintf(stderr, "[Synth-Load] ERROR: dit_path is NULL\n");
        return NULL;
    }
    if (!params->text_encoder_path) {
        fprintf(stderr, "[Synth-Load] ERROR: text_encoder_path is NULL\n");
        return NULL;
    }
    if (!params->vae_path) {
        fprintf(stderr, "[Synth-Load] ERROR: vae_path is NULL\n");
        return NULL;
    }

    AceSynth * ctx  = new AceSynth();
    ctx->params     = *params;
    ctx->have_dit   = false;
    ctx->have_vae   = false;
    ctx->have_detok = false;
    ctx->have_tok   = false;

    Timer timer;

    // Cache DiT metadata (config + silence + null_cond + is_turbo) from the GGUF.
    // No tensor weights touched, no GPU buffer allocated for the DiT yet.
    if (!dit_ggml_load_config(&ctx->dit_cfg, params->dit_path)) {
        delete ctx;
        return NULL;
    }
    if (!load_dit_cpu_side(ctx, params->dit_path)) {
        delete ctx;
        return NULL;
    }

    ctx->Oc     = ctx->dit_cfg.out_channels;           // 64
    ctx->ctx_ch = ctx->dit_cfg.in_channels - ctx->Oc;  // 128

    // BPE tokenizer (CPU only)
    timer.reset();
    if (!load_bpe_from_gguf(&ctx->bpe, params->text_encoder_path)) {
        fprintf(stderr, "[Synth-Load] FATAL: BPE load failed from %s\n", params->text_encoder_path);
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Synth-Load] BPE tokenizer: %.1f ms\n", timer.ms());

    // Text encoder (Qwen3 embedding branch)
    timer.reset();
    ctx->text_enc = {};
    qwen3_init_backend(&ctx->text_enc);
    if (!params->use_fa) {
        ctx->text_enc.use_flash_attn = false;
    }
    if (!qwen3_load_text_encoder(&ctx->text_enc, params->text_encoder_path)) {
        fprintf(stderr, "[Synth-Load] FATAL: TextEncoder load failed\n");
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Synth-Load] TextEncoder: %.1f ms\n", timer.ms());

    // Condition encoder (lyric + timbre + text projector)
    timer.reset();
    ctx->cond_enc = {};
    cond_ggml_init_backend(&ctx->cond_enc);
    if (!params->use_fa) {
        ctx->cond_enc.use_flash_attn = false;
    }
    ctx->cond_enc.clamp_fp16 = params->clamp_fp16;
    if (!cond_ggml_load(&ctx->cond_enc, params->dit_path)) {
        fprintf(stderr, "[Synth-Load] FATAL: CondEncoder load failed\n");
        qwen3_free(&ctx->text_enc);
        delete ctx;
        return NULL;
    }
    fprintf(stderr, "[Synth-Load] ConditionEncoder: %.1f ms\n", timer.ms());

    // Detokenizer: weights embedded in the DiT GGUF. Required for audio_codes input.
    timer.reset();
    ctx->detok = {};
    if (detok_ggml_load(&ctx->detok, params->dit_path)) {
        if (!params->use_fa) {
            ctx->detok.use_flash_attn = false;
        }
        ctx->have_detok = true;
        fprintf(stderr, "[Synth-Load] Detokenizer: %.1f ms\n", timer.ms());
    }

    // Tokenizer: weights embedded in the DiT GGUF. Used by cover-mode FSQ roundtrip.
    timer.reset();
    ctx->tok = {};
    if (tok_ggml_load(&ctx->tok, params->dit_path)) {
        if (!params->use_fa) {
            ctx->tok.use_flash_attn = false;
        }
        ctx->have_tok = true;
        fprintf(stderr, "[Synth-Load] Tokenizer: %.1f ms\n", timer.ms());
    }

    fprintf(stderr, "[Synth-Load] Resident modules loaded, turbo=%s, DiT+VAE loaded on demand\n",
            ctx->is_turbo ? "yes" : "no");
    if (params->clamp_fp16) {
        fprintf(stderr, "[Synth-Load] FP16 clamp enabled\n");
    }
    if (!params->use_batch_cfg) {
        fprintf(stderr, "[Synth-Load] Batched DiT CFG disabled (split 2-pass forwards)\n");
    }

    return ctx;
}

bool ace_synth_dit_load(AceSynth * ctx) {
    if (!ctx) {
        return false;
    }
    if (ctx->have_dit) {
        return true;
    }

    Timer timer;
    ctx->dit = {};
    dit_ggml_init_backend(&ctx->dit);
    if (!ctx->params.use_fa) {
        ctx->dit.use_flash_attn = false;
    }
    fprintf(stderr, "[Synth-Phase] DiT backend init: %.1f ms\n", timer.ms());

    AdapterGroupScales adapter_gs;
    adapter_gs.self_attn  = ctx->params.adapter_group_self_attn;
    adapter_gs.cross_attn = ctx->params.adapter_group_cross_attn;
    adapter_gs.mlp        = ctx->params.adapter_group_mlp;
    adapter_gs.cond_embed = ctx->params.adapter_group_cond_embed;

    timer.reset();
    if (!dit_ggml_load(&ctx->dit, ctx->params.dit_path, ctx->params.adapter_path, ctx->params.adapter_scale,
                       adapter_gs, ctx->params.adapter_mode)) {
        fprintf(stderr, "[Synth-Phase] FATAL: DiT load failed\n");
        dit_ggml_free(&ctx->dit);
        return false;
    }
    fprintf(stderr, "[Synth-Phase] DiT weight load: %.1f ms (fa=%s)\n", timer.ms(),
            ctx->dit.use_flash_attn ? "yes" : "no");

    ctx->have_dit = true;
    return true;
}

void ace_synth_dit_unload(AceSynth * ctx) {
    if (!ctx || !ctx->have_dit) {
        return;
    }
    dit_ggml_free(&ctx->dit);
    ctx->have_dit = false;
    fprintf(stderr, "[Synth-Phase] DiT unloaded\n");
}

bool ace_synth_vae_load(AceSynth * ctx) {
    if (!ctx) {
        return false;
    }
    if (ctx->have_vae) {
        return true;
    }
    Timer timer;
    ctx->vae = {};
    vae_ggml_load(&ctx->vae, ctx->params.vae_path);
    ctx->have_vae = true;
    fprintf(stderr, "[Synth-Phase] VAE weight load: %.1f ms\n", timer.ms());
    return true;
}

void ace_synth_vae_unload(AceSynth * ctx) {
    if (!ctx || !ctx->have_vae) {
        return;
    }
    vae_ggml_free(&ctx->vae);
    ctx->have_vae = false;
    fprintf(stderr, "[Synth-Phase] VAE unloaded\n");
}

// Phase 1: everything up to and including DiT generate. The produced job
// carries the latent output, the resolved state, and any silence-padded source
// buffer needed later by the wave splice in phase 2.
AceSynthJob * ace_synth_job_run_dit(AceSynth *         ctx,
                                    const AceRequest * reqs,
                                    const float *      src_audio,
                                    int                src_len,
                                    const float *      ref_audio,
                                    int                ref_len,
                                    int                batch_n,
                                    bool (*cancel)(void *),
                                    void * cancel_data) {
    if (!ctx || !reqs || batch_n < 1 || batch_n > 9) {
        return NULL;
    }
    if (!ctx->have_dit) {
        fprintf(stderr, "[Synth-Phase] FATAL: run_dit called without DiT loaded\n");
        return NULL;
    }

    AceSynthJob * job = new AceSynthJob();
    job->batch_n      = batch_n;

    SynthState & s = job->state;
    s.Oc           = ctx->Oc;
    s.ctx_ch       = ctx->ctx_ch;
    s.left_pad_sec = 0.0f;
    debug_init(&s.dbg, ctx->params.dump_dir);

    // Outpainting: pad src_audio with silence when region coordinates extend
    // beyond source bounds. Padding happens before VAE encode so T_cover
    // reflects the padded duration.
    const float * enc_audio = src_audio;
    int           enc_len   = src_len;
    {
        const std::string & tp = reqs[0].task_type;
        bool                is_region_task =
            (tp == TASK_REPAINT) || (tp == TASK_LEGO && reqs[0].repainting_end > reqs[0].repainting_start);
        if (is_region_task && src_audio && src_len > 0) {
            float src_dur  = (float) src_len / 48000.0f;
            float rs_raw   = reqs[0].repainting_start;
            float re_raw   = reqs[0].repainting_end;
            float end_time = (re_raw < 0.0f) ? src_dur : re_raw;
            float lpad     = (rs_raw < 0.0f) ? -rs_raw : 0.0f;
            float rpad     = (end_time > src_dur) ? end_time - src_dur : 0.0f;

            if (lpad > 0.0f || rpad > 0.0f) {
                int lpad_s       = (int) (lpad * 48000.0f);
                int rpad_s       = (int) (rpad * 48000.0f);
                int padded_total = src_len + lpad_s + rpad_s;

                s.padded_src.resize((size_t) padded_total * 2);
                memset(s.padded_src.data(), 0, s.padded_src.size() * sizeof(float));
                memcpy(s.padded_src.data() + (size_t) lpad_s * 2, src_audio, (size_t) src_len * 2 * sizeof(float));

                s.left_pad_sec = lpad;
                enc_audio      = s.padded_src.data();
                enc_len        = padded_total;

                fprintf(stderr, "[Outpaint] pad left=%.1fs right=%.1fs total=%.1fs\n", lpad, rpad,
                        (float) padded_total / 48000.0f);
            }
        }
    }

    // VAE encode source audio (possibly padded for outpainting).
    // ops_encode_src loads a local VAE encoder, encodes, and frees it:
    // no VAE encoder stays resident while the DiT runs.
    if (ops_encode_src(ctx, enc_audio, enc_len, s) != 0) {
        delete job;
        return NULL;
    }

    // Shared request, mode flags, use_source_context.
    // Per-batch: caption, lyrics, metadata, audio_codes, and seed come from reqs[b].
    // seed must be resolved (non-negative) before calling this function.
    s.rr                 = reqs[0];
    // task_type is the master. Empty = text2music.
    s.task               = s.rr.task_type.empty() ? std::string(TASK_TEXT2MUSIC) : s.rr.task_type;
    // only repaint uses region masking. complete uses full src context (Python behavior).
    s.is_repaint         = (s.task == TASK_REPAINT);
    // lego with valid rs/re = region-constrained: generate only in zone, full audio context
    s.is_lego_region     = (s.task == TASK_LEGO && s.rr.repainting_end > s.rr.repainting_start);
    s.rs                 = s.rr.repainting_start;
    s.re                 = s.rr.repainting_end;
    // use_source_context must be set before ops_resolve_T which uses it for T.
    // Mode routing refines it per task; this first pass covers all source-context tasks.
    s.use_source_context = (s.task == TASK_COVER || s.task == TASK_COVER_NOFSQ || s.task == TASK_REPAINT ||
                            s.task == TASK_LEGO || s.task == TASK_EXTRACT || s.task == TASK_COMPLETE);
    // non-cover encoding pass (for audio_cover_strength < 1.0 switching) always uses text2music.
    s.nc_instruction_str = DIT_INSTR_TEXT2MUSIC;

    // Resolve DiT params (steps, guidance, shift) and scan audio codes
    if (ops_resolve_params(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Promote text2music to cover when codes present (Python _resolve_generate_music_task).
    // DiT was trained with the cover instruction for codes guided generation.
    if (s.task == TASK_TEXT2MUSIC && s.have_codes) {
        s.task               = std::string(TASK_COVER);
        s.use_source_context = true;
    }

    // Timestep schedule
    ops_build_schedule(s);

    // Solver selection: resolve from infer_method, default to "euler"
    s.solver = s.rr.infer_method.empty() ? "euler" : s.rr.infer_method;
    if (s.solver == "ode") s.solver = "euler";  // legacy alias

    // Resolve latent frame count T
    if (ops_resolve_T(ctx, s) != 0) {
        delete job;
        return NULL;
    }

    // Resolve repaint quality params from repaint_strength.
    // Python: _resolve_repaint_config("balanced", strength).
    // 0.0 = aggressive, 0.5 = balanced (default), 1.0 = conservative.
    {
        float rs = s.rr.repaint_strength;
        if (rs < 0.0f) {
            rs = 0.0f;
        }
        if (rs > 1.0f) {
            rs = 1.0f;
        }
        float inv                  = 1.0f - rs;
        s.repaint_injection_ratio  = inv;
        s.repaint_crossfade_frames = (int) (25.0f * inv + 0.5f);
        s.repaint_wav_cf_sec       = 0.05f * inv;
        if (s.is_repaint || s.is_lego_region) {
            fprintf(stderr, "[Synth-Run] repaint_strength=%.2f -> injection=%.2f, crossfade=%d frames, wav_cf=%.0fms\n",
                    rs, s.repaint_injection_ratio, s.repaint_crossfade_frames, s.repaint_wav_cf_sec * 1000.0f);
        }
    }

    // Mode routing: per-task instruction, use_source_context, validation.
    //    All task knowledge lives here. Adding a mode = adding one branch.
    //    Track name is UPPERCASE in instructions (matches Python task_utils.py).
    {
        std::string track_upper = s.rr.track;
        for (char & ch : track_upper) {
            ch = (char) toupper((unsigned char) ch);
        }
        if (s.task == TASK_TEXT2MUSIC) {
            // silence context, no pre-encoding work needed
            s.use_source_context = false;
            s.instruction_str    = DIT_INSTR_TEXT2MUSIC;
        } else if (s.task == TASK_COVER) {
            s.use_source_context  = true;
            s.instruction_str     = DIT_INSTR_COVER;
            // save clean VAE latents before FSQ degrades them.
            // the FSQ roundtrip only affects context conditioning;
            // cover_noise_strength blending needs the original clean latents.
            s.noise_blend_latents = s.cover_latents;
            ops_fsq_roundtrip(ctx, s);  // FSQ degrades source latents, DiT diverges from original
        } else if (s.task == TASK_COVER_NOFSQ) {
            // cover without FSQ roundtrip: DiT works on clean VAE latents at 25Hz.
            // produces remixes that stay close to the source structure and timbre.
            // pass ref_audio = src_audio for best results.
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_COVER;
        } else if (s.task == TASK_REPAINT) {
            s.use_source_context = true;
            s.instruction_str    = DIT_INSTR_REPAINT;
        } else if (s.task == TASK_LEGO) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // all DiT steps hear the backing track
            s.instruction_str         = dit_instr_lego(track_upper);
            validate_track_names(s.rr.track, "Lego");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: lego requires base model, turbo output incoherent\n");
            }
        } else if (s.task == TASK_EXTRACT) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full mix
            s.instruction_str         = dit_instr_extract(track_upper);
            validate_track_names(s.rr.track, "Extract");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: extract requires base model, turbo output incoherent\n");
            }
        } else if (s.task == TASK_COMPLETE) {
            s.use_source_context      = true;
            s.rr.audio_cover_strength = 1.0f;  // DiT sees the full isolated stem
            s.instruction_str         = dit_instr_complete(track_upper);
            validate_track_names(s.rr.track, "Complete");
            fprintf(stderr, "[Synth-Run] task=%s\n", s.task.c_str());
            if (ctx->is_turbo) {
                fprintf(stderr, "[Synth-Run] WARNING: complete requires base model, turbo output incoherent\n");
            }
        }
        // validation: tasks that need source audio or codes
        if (s.use_source_context && !s.have_cover && !s.have_codes) {
            fprintf(stderr, "[Synth-Run] ERROR: task '%s' requires source audio or audio codes\n", s.task.c_str());
            delete job;
            return NULL;
        }
    }

    // Region coordinate adjustment (repaint and lego_region share the same path).
    // Shift rs/re into the padded reference frame. When no padding was applied,
    // left_pad_sec is 0 and the arithmetic is a no-op.
    // adjusted_start = repainting_start + left_padding_duration
    if (s.is_repaint || s.is_lego_region) {
        s.rs += s.left_pad_sec;
        if (s.re < 0.0f) {
            // sentinel: outpaint (start < 0) ends at source start, inpaint ends at source end
            s.re = (s.rr.repainting_start < 0.0f) ? s.left_pad_sec : (float) src_len / 48000.0f + s.left_pad_sec;
        } else {
            s.re += s.left_pad_sec;
        }
        if (s.re <= s.rs) {
            fprintf(stderr, "[Region] ERROR: end (%.1f) <= start (%.1f)\n", s.re, s.rs);
            delete job;
            return NULL;
        }
        fprintf(stderr, "[Region] %.1fs..%.1fs (canvas=%.1fs)\n", s.rs, s.re, (float) s.T_cover * 1920.0f / 48000.0f);
    }

    // Encode timbre from ref_audio (independent of task).
    // ops_encode_timbre loads a local VAE encoder, encodes, and frees it.
    ops_encode_timbre(ctx, ref_audio, ref_len, s);

    // Per-batch text + lyric encoding (main + optional non-cover pass)
    if (ops_encode_text(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Build DiT context [batch_n, T, ctx_ch] = src(64) | mask(64)
    if (ops_build_context(ctx, reqs, batch_n, s) != 0) {
        delete job;
        return NULL;
    }

    // Silence context for audio_cover_strength switching (cover only)
    ops_build_context_silence(ctx, batch_n, s);

    // Noise tensor (Philox), cover noise blend, per_S, repaint_src buffer
    ops_init_noise_and_repaint(ctx, reqs, batch_n, s);

    // DiT denoising loop. Latents land in s.output and stay in job memory
    // until phase 2 feeds them to the VAE decoder.
    if (ops_dit_generate(ctx, batch_n, s, cancel, cancel_data) != 0) {
        delete job;
        return NULL;
    }

    return job;
}

// Phase 2: VAE decode all batch items and apply waveform splice for
// repaint/lego regions. Picks the padded source when the job carried one.
int ace_synth_job_run_vae(AceSynth *    ctx,
                          AceSynthJob * job,
                          const float * splice_src,
                          int           splice_len,
                          AceAudio *    out,
                          bool (*cancel)(void *),
                          void * cancel_data) {
    if (!ctx || !job || !out) {
        return -1;
    }
    if (!ctx->have_vae) {
        fprintf(stderr, "[Synth-Phase] FATAL: run_vae called without VAE loaded\n");
        return -1;
    }

    // Outpainting: splice uses the padded source (silence at extended boundaries).
    const float * sp_audio = job->state.padded_src.empty() ? splice_src : job->state.padded_src.data();
    int           sp_len   = job->state.padded_src.empty() ? splice_len : (int) (job->state.padded_src.size() / 2);

    return ops_vae_decode_and_splice(ctx, job->batch_n, out, job->state, sp_audio, sp_len, cancel, cancel_data);
}

int ace_synth_job_batch_n(const AceSynthJob * job) {
    return job ? job->batch_n : 0;
}

void ace_synth_job_free(AceSynthJob * job) {
    delete job;
}

void ace_audio_free(AceAudio * audio) {
    if (audio && audio->samples) {
        free(audio->samples);
        audio->samples   = NULL;
        audio->n_samples = 0;
    }
}

void ace_synth_free(AceSynth * ctx) {
    if (!ctx) {
        return;
    }
    ace_synth_dit_unload(ctx);
    ace_synth_vae_unload(ctx);
    if (ctx->have_detok) {
        detok_ggml_free(&ctx->detok);
    }
    if (ctx->have_tok) {
        tok_ggml_free(&ctx->tok);
    }
    cond_ggml_free(&ctx->cond_enc);
    qwen3_free(&ctx->text_enc);
    delete ctx;
}
