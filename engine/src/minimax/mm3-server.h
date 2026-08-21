#pragma once
// minimax/mm3-server.h — HTTP surface for the MiniMax-Music3 backend.
//
// HOT-Step file (does not exist upstream). This is the ONE hook include that
// wires the whole MiniMax subsystem into an upstream-derived file:
//
//     engine/tools/hot-step-server.cpp:   #include "minimax/mm3-server.h"
//                                         mm3_register_routes(svr, models_dir);
//
// Nothing else in engine/src/ or engine/tools/ refers to minimax/, so an
// upstream sync can only break this in a way `engine/verify-hooks.ps1` catches
// (and it fails loudly at compile time, not silently).
//
// Endpoints (all family-scoped under /mm3/ so they can never collide with the
// ACE-Step surface):
//
//   GET  /mm3/props       discovery + per-component config summary + residency + VRAM
//                         (+ the quant catalogue under "variants")
//   POST /mm3/warm        load both GGUFs into VRAM (idempotent)
//   POST /mm3/unload      free them (idempotent)
//   POST /mm3/select-model  choose the lm/synth quant to run; unloads on change
//   POST /mm3/voc-decode  DEBUG: vocoder-only decode, latents in -> WAV out
//   POST /mm3/dit-forward DEBUG: one flow-DiT forward, latents+cond in -> velocity out
//   POST /mm3/flow-sample DEBUG: the Euler loop, noise+cond in -> final latents out
//   POST /mm3/depth-frame DEBUG: one frame of RVQ depth decoding, LM hidden in ->
//                         7 codes + 7 hiddens + both pre-CFG logit rows out
//   POST /mm3/cond-encode DEBUG: the condition encoder, fused AR hiddens in ->
//                         latent-rate DiT conditioning out
//   POST /mm3/lm-plan     the AR planning stage end to end: prompt (or token ids)
//                         in -> RVQ codes + the [F, 8, 4096] conditioning block
//   POST /mm3/tokenize-check  caption + lyrics in -> assembled-prompt token
//                         count vs the 5,000 limit. Cold-capable, no VRAM.
//   POST /mm3/synth-e2e   DEPRECATED BRING-UP: the WHOLE pipeline on an httplib
//                         thread. Kept for parity replay + latent dumps only.
//
// The PRODUCTION generation endpoint, POST /mm3/synth, and its progress poller
// GET /mm3/job live in minimax/mm3-job.h — they need hot-step-server.cpp's job
// system (Job / job_create / work_push), which is defined below this include,
// so that file is a second, mid-file hook. Both are checked by verify-hooks.ps1.
//
// Concurrency: the ACE pipeline runs GPU work on a single worker thread while
// these handlers run on httplib threads. g_mm3_mutex serialises MM3 load,
// unload and the bring-up compute endpoints against each other AND against the
// worker-thread /mm3/synth job, which takes the same mutex for its whole run.
// The bring-up endpoints (/mm3/voc-decode, /mm3/dit-forward, /mm3/flow-sample,
// /mm3/depth-frame, /mm3/cond-encode, /mm3/lm-plan, /mm3/synth-e2e) still do
// real GPU work on an httplib thread and can therefore contend with the ACE
// worker — that is exactly what /mm3/synth exists to avoid.

#include "mm3-ar-cache.h"
#include "mm3-ar-loop.h"
#include "mm3-cond-graph.h"
#include "mm3-depth-graph.h"
#include "mm3-dit-graph.h"
#include "mm3-lm-graph.h"
#include "mm3-model.h"
#include "mm3-pipeline.h"
#include "mm3-request.h"
#include "mm3-tokenizer.h"
#include "mm3-vocoder-graph.h"

#include "audio-io.h"
#include "yyjson.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "httplib.h"
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

#include <mutex>
#include <string>

static MM3Model  g_mm3;
static std::mutex g_mm3_mutex;

// Local copy of the server's error-response shape (hot-step-server.cpp's
// json_error is a static defined after this include, so it cannot be reused).
static void mm3_json_error(httplib::Response & res, int status, const std::string & msg) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "error", msg.c_str());
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.status = status;
    res.set_content(json ? json : "{\"error\":\"unknown\"}", "application/json");
    if (json) {
        free(json);
    }
}

static void mm3_json_add_file(yyjson_mut_doc * doc, yyjson_mut_val * parent, const char * key,
                              const MM3FileInfo & fi) {
    yyjson_mut_val * o = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, parent, key, o);
    yyjson_mut_obj_add_bool(doc, o, "found", fi.found);
    yyjson_mut_obj_add_bool(doc, o, "probe_ok", fi.probe_ok);
    yyjson_mut_obj_add_strcpy(doc, o, "name", fi.name.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "path", fi.path.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "arch", fi.arch.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "general_name", fi.general_name.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "license", fi.license.c_str());
    yyjson_mut_obj_add_strcpy(doc, o, "source_layout", fi.source_layout.c_str());
    yyjson_mut_obj_add_uint(doc, o, "converter_version", fi.converter_version);
    yyjson_mut_obj_add_uint(doc, o, "file_type", fi.file_type);
    yyjson_mut_obj_add_uint(doc, o, "file_bytes", fi.file_bytes);
    yyjson_mut_obj_add_uint(doc, o, "tensor_bytes", fi.tensor_bytes);
    yyjson_mut_obj_add_int(doc, o, "n_tensors", fi.n_tensors);
    if (!fi.probe_error.empty()) {
        yyjson_mut_obj_add_strcpy(doc, o, "error", fi.probe_error.c_str());
    }
}

static void mm3_json_add_config(yyjson_mut_doc * doc, yyjson_mut_val * root) {
    yyjson_mut_val * cfg = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "config", cfg);

    // ── lm ──
    {
        const MM3LmConfig & c = g_mm3.lm_cfg;
        yyjson_mut_val *    o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "lm", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "context_length", c.context_length);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "feed_forward_length", c.feed_forward_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_count_kv", c.head_count_kv);
        yyjson_mut_obj_add_uint(doc, o, "key_length", c.key_length);
        yyjson_mut_obj_add_uint(doc, o, "value_length", c.value_length);
        yyjson_mut_obj_add_real(doc, o, "rms_eps", c.rms_eps);
        yyjson_mut_obj_add_real(doc, o, "rope_freq_base", c.rope_freq_base);
        yyjson_mut_obj_add_uint(doc, o, "vocab_size", c.vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "semantic_vocab_offset", c.semantic_vocab_offset);
        yyjson_mut_obj_add_uint(doc, o, "semantic_vocab_size", c.semantic_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "acoustic_vocab_size", c.acoustic_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "num_codebooks", c.num_codebooks);
        yyjson_mut_obj_add_uint(doc, o, "eos_audio", c.eos_audio);
        yyjson_mut_obj_add_uint(doc, o, "frame_rate", c.frame_rate);
        yyjson_mut_obj_add_uint(doc, o, "max_audio_frames", c.max_audio_frames);
        yyjson_mut_obj_add_uint(doc, o, "max_prompt_tokens", c.max_prompt_tokens);
        yyjson_mut_obj_add_real(doc, o, "ar_cfg_scale", c.ar_cfg_scale);
        yyjson_mut_obj_add_uint(doc, o, "ar_top_k", c.ar_top_k);
        yyjson_mut_obj_add_real(doc, o, "ar_embedding_scale", c.ar_embedding_scale);

        yyjson_mut_val * tok = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, o, "tokens", tok);
        yyjson_mut_obj_add_uint(doc, tok, "im_start", c.tok_im_start);
        yyjson_mut_obj_add_uint(doc, tok, "im_end", c.tok_im_end);
        yyjson_mut_obj_add_uint(doc, tok, "audio_cfg", c.tok_audio_cfg);
        yyjson_mut_obj_add_uint(doc, tok, "audio_start", c.tok_audio_start);
        yyjson_mut_obj_add_uint(doc, tok, "audio_end", c.tok_audio_end);
        yyjson_mut_obj_add_uint(doc, tok, "caption_start", c.tok_caption_start);
        yyjson_mut_obj_add_uint(doc, tok, "caption_end", c.tok_caption_end);
        yyjson_mut_obj_add_uint(doc, tok, "lyrics_start", c.tok_lyrics_start);
        yyjson_mut_obj_add_uint(doc, tok, "lyrics_end", c.tok_lyrics_end);
    }

    const MM3SynthConfig & s = g_mm3.synth_cfg;

    yyjson_mut_val * comps = yyjson_mut_arr(doc);
    for (const auto & c : s.components) {
        yyjson_mut_arr_add_strcpy(doc, comps, c.c_str());
    }
    yyjson_mut_obj_add_val(doc, cfg, "synth_components", comps);

    // ── depth ──
    {
        const MM3DepthConfig & c = s.depth;
        yyjson_mut_val *       o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "depth", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "feed_forward_length", c.feed_forward_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_dim", c.head_dim);
        yyjson_mut_obj_add_uint(doc, o, "max_position", c.max_position);
        yyjson_mut_obj_add_real(doc, o, "rms_eps", c.rms_eps);
        yyjson_mut_obj_add_uint(doc, o, "num_codebooks", c.num_codebooks);
        yyjson_mut_obj_add_uint(doc, o, "audio_vocab_size", c.audio_vocab_size);
        yyjson_mut_obj_add_uint(doc, o, "audio_embd_rows", c.audio_embd_rows);
        yyjson_mut_obj_add_bool(doc, o, "causal", c.causal);
        yyjson_mut_obj_add_bool(doc, o, "rope", c.rope);
    }

    // ── cond ──
    {
        const MM3CondConfig & c = s.cond;
        yyjson_mut_val *      o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "cond", o);
        yyjson_mut_obj_add_uint(doc, o, "num_layers", c.num_layers);
        yyjson_mut_obj_add_uint(doc, o, "hidden_dim", c.hidden_dim);
        yyjson_mut_obj_add_uint(doc, o, "out_dim", c.out_dim);
        yyjson_mut_obj_add_uint(doc, o, "kernel_size", c.kernel_size);
        yyjson_mut_obj_add_uint(doc, o, "padding", c.padding);
        yyjson_mut_obj_add_uint(doc, o, "input_sampling_rate", c.input_sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "input_hop_length", c.input_hop_length);
        yyjson_mut_obj_add_uint(doc, o, "output_sampling_rate", c.output_sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "output_hop_length", c.output_hop_length);
        yyjson_mut_obj_add_strcpy(doc, o, "interpolation", c.interpolation.c_str());
        yyjson_mut_obj_add_strcpy(doc, o, "layer_mix", c.layer_mix.c_str());
    }

    // ── dit ──
    {
        const MM3DitConfig & c = s.dit;
        yyjson_mut_val *     o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "dit", o);
        yyjson_mut_obj_add_uint(doc, o, "block_count", c.block_count);
        yyjson_mut_obj_add_uint(doc, o, "embedding_length", c.embedding_length);
        yyjson_mut_obj_add_uint(doc, o, "head_count", c.head_count);
        yyjson_mut_obj_add_uint(doc, o, "head_dim", c.head_dim);
        yyjson_mut_obj_add_uint(doc, o, "ff_inner", c.ff_inner);
        yyjson_mut_obj_add_uint(doc, o, "in_channels", c.in_channels);
        yyjson_mut_obj_add_uint(doc, o, "condition_dim", c.condition_dim);
        yyjson_mut_obj_add_uint(doc, o, "concat_channels", c.concat_channels);
        yyjson_mut_obj_add_real(doc, o, "layer_norm_eps", c.layer_norm_eps);
        yyjson_mut_obj_add_uint(doc, o, "rope_dim", c.rope_dim);
        yyjson_mut_obj_add_real(doc, o, "rope_theta", c.rope_theta);
        yyjson_mut_obj_add_strcpy(doc, o, "rope_type", c.rope_type.c_str());
        yyjson_mut_obj_add_uint(doc, o, "fourier_dim", c.fourier_dim);
        yyjson_mut_obj_add_strcpy(doc, o, "glu_order", c.glu_order.c_str());
        yyjson_mut_obj_add_bool(doc, o, "output_negated", c.output_negated);
        yyjson_mut_obj_add_bool(doc, o, "timestep_token_prepended", c.timestep_token_prepended);
        yyjson_mut_obj_add_bool(doc, o, "pre_post_conv_residual", c.pre_post_conv_residual);
        yyjson_mut_obj_add_bool(doc, o, "attn_bias", c.attn_bias);
        yyjson_mut_obj_add_uint(doc, o, "window_frames", c.window_frames);
        yyjson_mut_obj_add_uint(doc, o, "hop_frames", c.hop_frames);
        yyjson_mut_obj_add_uint(doc, o, "window_latents", c.window_latents);
        yyjson_mut_obj_add_uint(doc, o, "hop_latents", c.hop_latents);
    }

    // ── flow ──
    {
        const MM3FlowConfig & c = s.flow;
        yyjson_mut_val *      o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "flow", o);
        yyjson_mut_obj_add_strcpy(doc, o, "scheduler", c.scheduler.c_str());
        yyjson_mut_obj_add_uint(doc, o, "steps", c.steps);
        yyjson_mut_obj_add_real(doc, o, "cfg_scale", c.cfg_scale);
        yyjson_mut_obj_add_bool(doc, o, "invert_sigmas", c.invert_sigmas);
        yyjson_mut_obj_add_real(doc, o, "shift", c.shift);
        yyjson_mut_obj_add_uint(doc, o, "num_train_timesteps", c.num_train_timesteps);
    }

    // ── voc ──
    {
        const MM3VocConfig & c = s.voc;
        yyjson_mut_val *     o = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, cfg, "voc", o);
        yyjson_mut_obj_add_uint(doc, o, "latent_channels", c.latent_channels);
        yyjson_mut_obj_add_uint(doc, o, "fold_channels", c.fold_channels);
        yyjson_mut_obj_add_uint(doc, o, "dec_in_dim", c.dec_in_dim);
        yyjson_mut_obj_add_uint(doc, o, "hidden_dim", c.hidden_dim);
        yyjson_mut_val * ups = yyjson_mut_arr(doc);
        for (int32_t r : c.upsample_rates) {
            yyjson_mut_arr_add_int(doc, ups, r);
        }
        yyjson_mut_obj_add_val(doc, o, "upsample_rates", ups);
        yyjson_mut_val * dil = yyjson_mut_arr(doc);
        for (int32_t r : c.res_dilations) {
            yyjson_mut_arr_add_int(doc, dil, r);
        }
        yyjson_mut_obj_add_val(doc, o, "res_dilations", dil);
        yyjson_mut_obj_add_uint(doc, o, "total_upsample", c.total_upsample);
        yyjson_mut_obj_add_uint(doc, o, "sampling_rate", c.sampling_rate);
        yyjson_mut_obj_add_uint(doc, o, "channels", c.channels);
        yyjson_mut_obj_add_real(doc, o, "snake_eps", c.snake_eps);
        yyjson_mut_obj_add_bool(doc, o, "final_tanh", c.final_tanh);
        yyjson_mut_obj_add_bool(doc, o, "weight_norm_folded", c.weight_norm_folded);
        yyjson_mut_obj_add_strcpy(doc, o, "snake", c.snake.c_str());
    }
}

// GET /mm3/props
static void mm3_handle_props(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_strcpy(doc, root, "backend", "minimax-m3");
    yyjson_mut_obj_add_strcpy(doc, root, "model", "MiniMax-Music3");
    yyjson_mut_obj_add_bool(doc, root, "available", mm3_available(g_mm3));
    yyjson_mut_obj_add_bool(doc, root, "loaded", g_mm3.loaded);
    yyjson_mut_obj_add_strcpy(doc, root, "models_dir", g_mm3.models_dir.c_str());

    // synth_ready: every role's GGUF found AND its header parsed clean, i.e.
    // POST /mm3/synth will get as far as loading weights. Deliberately NOT a
    // VRAM or residency claim — `loaded` above is residency, and the job's own
    // arbitration decides whether the weights fit.
    const bool synth_ready = mm3_available(g_mm3);
    yyjson_mut_obj_add_bool(doc, root, "synth_ready", synth_ready);
    yyjson_mut_obj_add_uint(doc, root, "prompt_token_limit", MM3_MAX_PROMPT_TOKENS);
    // AR cache (mm3-ar-cache.h). Reported so a caller can tell "reuse_ar is on
    // but the slot is empty" from "reuse_ar did nothing", and can see what the
    // held block is costing in host RAM.
    {
        yyjson_mut_val * ac = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, root, "ar_cache", ac);
        yyjson_mut_obj_add_bool(doc, ac, "present", !g_mm3_ar_cache.key.empty());
        yyjson_mut_obj_add_int(doc, ac, "frames", g_mm3_ar_cache.frames);
        yyjson_mut_obj_add_real(doc, ac, "mb", (double) mm3_ar_cache_bytes() / 1048576.0);
        yyjson_mut_obj_add_int(doc, ac, "hits", g_mm3_ar_cache.hits);
    }
    yyjson_mut_obj_add_uint(doc, root, "max_audio_frames_limit", MM3_MAX_AUDIO_FRAMES);

    yyjson_mut_val * dirs = yyjson_mut_arr(doc);
    for (const auto & d : g_mm3.search_dirs) {
        yyjson_mut_arr_add_strcpy(doc, dirs, d.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "search_dirs", dirs);

    yyjson_mut_val * files = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "files", files);
    mm3_json_add_file(doc, files, "lm", g_mm3.lm_file);
    for (int r = 0; r < MM3_R_COUNT; r++) {
        mm3_json_add_file(doc, files, MM3_SYNTH_ROLE[r], g_mm3.role_file[r]);
    }
    // Legacy key: pre-split clients gate on files.synth.found; the DiT file is
    // the honest stand-in (largest non-LM component, always required).
    mm3_json_add_file(doc, files, "synth", g_mm3.role_file[MM3_R_DIT]);

    // Quant catalogue — what the UI's model dropdowns are built from. "selected"
    // is the quant actually in force (the resolved pick, not the request), so a
    // fallback from a deleted file is visible rather than silent.
    {
        yyjson_mut_val * variants = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_val(doc, root, "variants", variants);
        auto add_role = [&](const char * key, const std::vector<MM3Variant> & vars, const MM3FileInfo & fi,
                            const std::string & want) {
            yyjson_mut_val * o = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_val(doc, variants, key, o);
            yyjson_mut_val * arr = yyjson_mut_arr(doc);
            yyjson_mut_obj_add_val(doc, o, "available", arr);
            std::string selected;
            for (const auto & v : vars) {
                yyjson_mut_val * e = yyjson_mut_obj(doc);
                yyjson_mut_obj_add_strcpy(doc, e, "quant", v.quant.c_str());
                yyjson_mut_obj_add_strcpy(doc, e, "filename", v.name.c_str());
                yyjson_mut_obj_add_uint(doc, e, "bytes", v.bytes);
                yyjson_mut_arr_add_val(arr, e);
                if (v.name == fi.name) {
                    selected = v.quant;
                }
            }
            yyjson_mut_obj_add_strcpy(doc, o, "selected", selected.c_str());
            yyjson_mut_obj_add_strcpy(doc, o, "requested", want.c_str());
        };
        add_role("lm", g_mm3.lm_variants, g_mm3.lm_file, g_mm3.want_lm_quant);
        for (int r = 0; r < MM3_R_COUNT; r++) {
            add_role(MM3_SYNTH_ROLE[r], g_mm3.role_variants[r], g_mm3.role_file[r], g_mm3.want_role_quant[r]);
        }
        // Legacy key: pre-split clients drive one "synth" dropdown; alias the
        // DiT role so they keep working (a bundle appears in its list too).
        add_role("synth", g_mm3.role_variants[MM3_R_DIT], g_mm3.role_file[MM3_R_DIT],
                 g_mm3.want_role_quant[MM3_R_DIT]);
    }

    mm3_json_add_config(doc, root);

    yyjson_mut_val * vram = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_val(doc, root, "vram", vram);
    yyjson_mut_obj_add_uint(doc, vram, "lm_bytes", g_mm3.vram_lm);
    yyjson_mut_obj_add_uint(doc, vram, "synth_bytes", g_mm3.vram_synth);
    yyjson_mut_obj_add_uint(doc, vram, "total_bytes", mm3_vram_bytes(g_mm3));
    yyjson_mut_obj_add_real(doc, vram, "total_mb", (double) mm3_vram_bytes(g_mm3) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, vram, "load_ms", g_mm3.load_ms);
    yyjson_mut_obj_add_uint(doc, vram, "lm_tensors", g_mm3.tmap_lm.size());
    yyjson_mut_obj_add_uint(doc, vram, "synth_tensors", g_mm3.tmap_synth.size());

    yyjson_mut_val * errs = yyjson_mut_arr(doc);
    for (const auto & e : g_mm3.meta_errors) {
        yyjson_mut_arr_add_strcpy(doc, errs, e.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "errors", errs);

    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// POST /mm3/warm — load both files. Idempotent.
static void mm3_handle_warm(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    const bool  was_loaded = g_mm3.loaded;
    std::string err;
    if (!mm3_load(&g_mm3, &err)) {
        mm3_json_error(res, 500, err.empty() ? "MM3 load failed" : err);
        return;
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "loaded", true);
    yyjson_mut_obj_add_bool(doc, root, "already_loaded", was_loaded);
    yyjson_mut_obj_add_uint(doc, root, "lm_bytes", g_mm3.vram_lm);
    yyjson_mut_obj_add_uint(doc, root, "synth_bytes", g_mm3.vram_synth);
    yyjson_mut_obj_add_uint(doc, root, "total_bytes", mm3_vram_bytes(g_mm3));
    yyjson_mut_obj_add_real(doc, root, "total_mb", (double) mm3_vram_bytes(g_mm3) / (1024.0 * 1024.0));
    yyjson_mut_obj_add_real(doc, root, "load_ms", g_mm3.load_ms);
    yyjson_mut_obj_add_uint(doc, root, "lm_tensors", g_mm3.tmap_lm.size());
    yyjson_mut_obj_add_uint(doc, root, "synth_tensors", g_mm3.tmap_synth.size());
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// POST /mm3/unload — free VRAM. Idempotent.
static void mm3_handle_unload(const httplib::Request &, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    const bool   was_loaded = g_mm3.loaded;
    // The AR stage's KV cache is a separate allocation from the weights and can
    // be gigabytes on its own — count it, or /mm3/unload under-reports badly.
    const size_t freed      = mm3_vram_bytes(g_mm3) + g_mm3_lm.kv_bytes;
    // Graph state holds derived weights + schedulers that point into the model's
    // buffers and hold backend references — tear them down first. The runtime
    // LM adapter goes before the graph frees so its borrowed pointer is cleared
    // while the graph still exists.
    mm3_lm_adapter_drop();
    mm3_vocoder_free(&g_mm3_voc);
    mm3_dit_free(&g_mm3_dit);
    mm3_depth_free(&g_mm3_depth);
    mm3_cond_free(&g_mm3_cond);
    mm3_lm_free(&g_mm3_lm);
    mm3_unload(&g_mm3);
    // Host RAM, not VRAM, and hundreds of MB of it — an explicit unload is the
    // user asking for memory back, so the AR slot goes too. (Deliberately NOT
    // hooked into mm3_unload() itself: that fires after every generation when
    // keep-loaded is off, which would drop the slot before it could ever hit.)
    mm3_ar_cache_clear("engine unload");

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "unloaded", was_loaded);
    yyjson_mut_obj_add_bool(doc, root, "loaded", false);
    yyjson_mut_obj_add_uint(doc, root, "freed_bytes", was_loaded ? freed : 0);
    yyjson_mut_obj_add_real(doc, root, "freed_mb", was_loaded ? (double) freed / (1024.0 * 1024.0) : 0.0);
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// FNV-1a over the raw output bytes. Used only by the selftest response, as a
// crisp "byte-identical across runs" signal that rms/peak alone cannot give.
static uint64_t mm3_fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t        h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// POST /mm3/voc-decode — vocoder-only bring-up + parity endpoint.
//
//   ?frames=L        body is raw little-endian F32, exactly 128*L*4 bytes,
//                    channel-major (channel c at offset c*L) — i.e. the memory
//                    order of a torch [1, 128, L] contiguous tensor.
//                    Returns 16-bit PCM stereo WAV at mm3.voc.sampling_rate.
//
//   ?selftest=1      ignores the body. Generates L=256 deterministic
//                    pseudo-random latents (std::mt19937 seeded 20260813,
//                    N(0, 0.5)) and returns JSON statistics instead of audio.
//                    Re-running must reproduce the same hash.
//
// 503 unless MM3 is warm.
static void mm3_handle_voc_decode(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }

    const MM3VocConfig & vc = g_mm3.synth_cfg.voc;
    const int64_t        LC = (int64_t) vc.latent_channels;
    const int            sr = (int) vc.sampling_rate;

    const bool selftest = req.has_param("selftest") && req.get_param_value("selftest") != "0";

    int64_t L = 0;
    if (selftest) {
        L = 256;
    } else {
        if (!req.has_param("frames")) {
            mm3_json_error(res, 400, "missing ?frames=<latent frame count>");
            return;
        }
        L = strtoll(req.get_param_value("frames").c_str(), nullptr, 10);
    }
    if (L <= 0 || L > 8192) {
        mm3_json_error(res, 400, "frames must be in 1..8192");
        return;
    }

    std::vector<float> latents((size_t) (LC * L));
    if (selftest) {
        std::mt19937                          rng(20260813u);
        std::normal_distribution<float>       dist(0.0f, 0.5f);
        for (size_t i = 0; i < latents.size(); i++) {
            latents[i] = dist(rng);
        }
    } else {
        const size_t want = (size_t) (LC * L) * sizeof(float);
        if (req.body.size() != want) {
            char buf[192];
            snprintf(buf, sizeof(buf), "body is %zu bytes, expected %zu (= %lld channels * %lld frames * 4)",
                     req.body.size(), want, (long long) LC, (long long) L);
            mm3_json_error(res, 400, buf);
            return;
        }
        memcpy(latents.data(), req.body.data(), want);
    }

    std::vector<float> audio;
    std::string        err;
    const auto         t0 = std::chrono::steady_clock::now();
    const bool         ok = mm3_vocoder_decode(g_mm3, latents.data(), L, audio, &err);
    const double       ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        mm3_json_error(res, 500, err.empty() ? "vocoder decode failed" : err);
        return;
    }

    const int T = (int) (audio.size() / 2);
    fprintf(stderr, "[MM3-Voc] Decoded %lld latent frames -> %d samples/ch (%.2fs @ %d Hz) in %.0f ms\n",
            (long long) L, T, (double) T / (double) (sr > 0 ? sr : 1), sr, ms);

    if (!selftest) {
        std::string wav = audio_encode_wav_s16(audio.data(), T, sr);
        res.set_content(wav, "audio/wav");
        return;
    }

    double sum_sq  = 0.0;
    float  peak    = 0.0f;
    bool   has_nan = false;
    for (float v : audio) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan = true;
            continue;
        }
        sum_sq += (double) v * (double) v;
        float a = std::fabs(v);
        if (a > peak) {
            peak = a;
        }
    }
    const double rms = audio.empty() ? 0.0 : std::sqrt(sum_sq / (double) audio.size());

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "selftest", true);
    yyjson_mut_obj_add_int(doc, root, "frames", L);
    yyjson_mut_obj_add_int(doc, root, "n_samples", T);
    yyjson_mut_obj_add_int(doc, root, "channels", 2);
    yyjson_mut_obj_add_int(doc, root, "sample_rate", sr);
    yyjson_mut_obj_add_real(doc, root, "rms", rms);
    yyjson_mut_obj_add_real(doc, root, "peak", (double) peak);
    yyjson_mut_obj_add_bool(doc, root, "has_nan", has_nan);
    yyjson_mut_obj_add_real(doc, root, "ms", ms);
    yyjson_mut_obj_add_uint(doc, root, "hash",
                            mm3_fnv1a(audio.data(), audio.size() * sizeof(float)));
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// Shared body parser for the two flow endpoints. Both take the same payload:
// a channel-major [128, L] latent block immediately followed by a [L, 2048]
// condition block, raw little-endian F32 — i.e. exactly
// `np.concatenate([latents.ravel(), condition.ravel()]).tobytes()` from the
// reference dumps, with no header and no framing.
//
// Returns false and writes the response itself on any error.
static bool mm3_parse_flow_body(const httplib::Request & req, httplib::Response & res, int64_t * out_L,
                                const float ** out_latents, const float ** out_cond) {
    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return false;
    }
    if (!req.has_param("frames")) {
        mm3_json_error(res, 400, "missing ?frames=<latent frame count>");
        return false;
    }
    const int64_t L = strtoll(req.get_param_value("frames").c_str(), nullptr, 10);
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        char buf[128];
        snprintf(buf, sizeof(buf), "frames must be in 1..%d", MM3_DIT_MAX_FRAMES);
        mm3_json_error(res, 400, buf);
        return false;
    }

    const int64_t IC   = (int64_t) g_mm3.synth_cfg.dit.in_channels;
    const int64_t CD   = (int64_t) g_mm3.synth_cfg.dit.condition_dim;
    const size_t  want = (size_t) ((IC + CD) * L) * sizeof(float);
    if (req.body.size() != want) {
        char buf[224];
        snprintf(buf, sizeof(buf),
                 "body is %zu bytes, expected %zu (= (%lld latent + %lld condition channels) * %lld frames * 4)",
                 req.body.size(), want, (long long) IC, (long long) CD, (long long) L);
        mm3_json_error(res, 400, buf);
        return false;
    }

    *out_L       = L;
    *out_latents = (const float *) req.body.data();
    *out_cond    = (const float *) req.body.data() + (size_t) (IC * L);
    return true;
}

// POST /mm3/dit-forward?frames=L&t=<float>
//
// One flow-DiT forward pass. Body = latents ‖ condition (see above). Returns the
// raw velocity prediction as 128*L little-endian F32, no framing.
//
// This is the parity workhorse: feed it a reference `flow_w0_sN_latents_in` plus
// `cond_out_w0` at `t = timesteps[N]` and the output should match
// `flow_w0_sN_pred_cond`. Pass a zeroed condition block to reproduce
// `flow_w0_sN_pred_uncond` — CFG's unconditional branch is literally
// zeros_like(condition).
//
// 503 unless MM3 is warm.
static void mm3_handle_dit_forward(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    int64_t       L    = 0;
    const float * lat  = nullptr;
    const float * cond = nullptr;
    if (!mm3_parse_flow_body(req, res, &L, &lat, &cond)) {
        return;
    }

    const float t = req.has_param("t") ? strtof(req.get_param_value("t").c_str(), nullptr) : 0.0f;

    std::vector<float> velocity;
    std::string        err;
    const auto         t0 = std::chrono::steady_clock::now();
    const bool         ok = mm3_dit_forward(g_mm3, lat, cond, L, t, velocity, &err);
    const double       ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        mm3_json_error(res, 500, err.empty() ? "DiT forward failed" : err);
        return;
    }

    fprintf(stderr, "[MM3-DiT] Forward: L=%lld t=%.6f -> %zu floats in %.0f ms\n", (long long) L, (double) t,
            velocity.size(), ms);

    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%.1f", ms);
    res.set_header("X-MM3-Ms", hdr);
    res.set_content((const char *) velocity.data(), velocity.size() * sizeof(float), "application/octet-stream");
}

// POST /mm3/flow-sample?frames=L&steps=30&cfg=1.7
//
// The full Euler loop for one window. Body = noise ‖ condition (same layout as
// /mm3/dit-forward). Returns the final latents as 128*L little-endian F32.
//
// Parity target: `flow_w0_noise_latents` + `cond_out_w0` at the defaults should
// reproduce `flow_w0_latents_final`. Note the reference accumulates its Euler
// updates in bf16 while this runs F32, so loop-level agreement is necessarily
// looser than the per-step agreement /mm3/dit-forward gives.
//
// Window 0 has overlap 0; no chunk blending is implemented here.
//
// 503 unless MM3 is warm.
static void mm3_handle_flow_sample(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    int64_t       L     = 0;
    const float * noise = nullptr;
    const float * cond  = nullptr;
    if (!mm3_parse_flow_body(req, res, &L, &noise, &cond)) {
        return;
    }

    int steps = (int) g_mm3.synth_cfg.flow.steps;
    if (steps <= 0) {
        steps = 30;
    }
    if (req.has_param("steps")) {
        steps = atoi(req.get_param_value("steps").c_str());
    }
    float cfg = g_mm3.synth_cfg.flow.cfg_scale > 0.0f ? g_mm3.synth_cfg.flow.cfg_scale : 1.7f;
    if (req.has_param("cfg")) {
        cfg = strtof(req.get_param_value("cfg").c_str(), nullptr);
    }

    std::vector<float> latents;
    MM3FlowStats       stats;
    std::string        err;
    if (!mm3_flow_sample(g_mm3, noise, cond, L, steps, cfg, latents, &stats, &err)) {
        mm3_json_error(res, 500, err.empty() ? "flow sample failed" : err);
        return;
    }

    char hdr[96];
    snprintf(hdr, sizeof(hdr), "%.1f", stats.total_ms);
    res.set_header("X-MM3-Ms", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", stats.forwards ? stats.forward_ms / (double) stats.forwards : 0.0);
    res.set_header("X-MM3-Ms-Per-Forward", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", (double) stats.compute_bytes / (1024.0 * 1024.0));
    res.set_header("X-MM3-Compute-MB", hdr);
    res.set_content((const char *) latents.data(), latents.size() * sizeof(float), "application/octet-stream");
}

// Parse a comma-separated int list. Returns false on any malformed entry so a
// typo in `?forced=` fails loudly instead of silently decoding codebook 0.
static bool mm3_parse_int_list(const std::string & s, std::vector<int32_t> * out) {
    out->clear();
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) {
            j = s.size();
        }
        const std::string tok = s.substr(i, j - i);
        if (tok.empty()) {
            return false;
        }
        char *          end = nullptr;
        const long long v   = strtoll(tok.c_str(), &end, 10);
        if (!end || *end != '\0') {
            return false;
        }
        out->push_back((int32_t) v);
        if (j == s.size()) {
            break;
        }
        i = j + 1;
    }
    return !out->empty();
}

// POST /mm3/depth-frame?semantic=<code>[&forced=c1,...,c7]
//
// One frame of RVQ depth decoding — the AR loop's inner stage, minus the AR loop.
//
//   body     2 * 4096 little-endian F32, no framing: the global LM's
//            last_hidden_state for this iteration, CONDITIONAL row first then
//            unconditional (the manifest's `cfg_row_order`).
//   semantic the frame's codebook-0 code in [0, 16384) — the code, NOT the token
//            id; the vocab offset is applied inside.
//   forced   optional 7 comma-separated codes. With it the sampler is bypassed
//            and the graph sees exactly the reference's token sequence, which is
//            what makes logit parity meaningful (the reference's multinomial draw
//            is not reproducible outside torch). Without it, greedy argmax.
//
// RESPONSE: raw little-endian F32, no framing, three blocks back to back —
//
//     [        0 ..  7*1024 )   logits_cond    (7, 1024)  pre-CFG
//     [   7*1024 .. 14*1024 )   logits_uncond  (7, 1024)  pre-CFG
//     [  14*1024 .. 14*1024 + 7*4096 )   hidden_states (7, 4096)  cond row
//
// = 43008 floats = 172032 bytes at the reference geometry. numpy:
//
//     a = np.frombuffer(body, "<f4")
//     lc, lu, hs = a[:7168].reshape(7,1024), a[7168:14336].reshape(7,1024), a[14336:].reshape(7,4096)
//
// The sampled codes come back in the `X-MM3-Codes` header (comma-separated) —
// a header rather than a JSON envelope so the body stays a bare float array that
// np.frombuffer can take with no parsing at all.
//
// Parity targets: `depth_i0_logits_cond` / `_uncond` / `_hidden_states`, fed with
// `lm_i0_last_hidden`, semantic = `depth_i0_codes[0]`, forced =
// `depth_i0_acoustic_codes`. NOTE the AR off-by-one: depth iteration i pairs with
// `lm_i<i>_last_hidden`, but EMITTED frame 0 is iteration 1 — iteration 0's codes
// are generated and fed back, yet its hiddens are discarded.
//
// 503 unless MM3 is warm.
static void mm3_handle_depth_frame(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }

    const int64_t H  = (int64_t) g_mm3.synth_cfg.depth.embedding_length;
    const int     NC = (int) g_mm3.synth_cfg.depth.num_codebooks - 1;

    if (!req.has_param("semantic")) {
        mm3_json_error(res, 400, "missing ?semantic=<codebook-0 code>");
        return;
    }
    const int32_t semantic = (int32_t) strtol(req.get_param_value("semantic").c_str(), nullptr, 10);

    std::vector<int32_t> forced;
    if (req.has_param("forced")) {
        if (!mm3_parse_int_list(req.get_param_value("forced"), &forced)) {
            mm3_json_error(res, 400, "?forced= must be a comma-separated list of integers");
            return;
        }
        if ((int) forced.size() != NC) {
            char buf[128];
            snprintf(buf, sizeof(buf), "?forced= has %zu codes, expected %d", forced.size(), NC);
            mm3_json_error(res, 400, buf);
            return;
        }
    }

    const size_t want = (size_t) (2 * H) * sizeof(float);
    if (req.body.size() != want) {
        char buf[176];
        snprintf(buf, sizeof(buf), "body is %zu bytes, expected %zu (= 2 CFG rows * %lld hidden * 4)",
                 req.body.size(), want, (long long) H);
        mm3_json_error(res, 400, buf);
        return;
    }
    const float * hid_cond   = (const float *) req.body.data();
    const float * hid_uncond = hid_cond + H;

    MM3DepthFrame frame;
    std::string   err;
    if (!mm3_depth_decode_frame(g_mm3, hid_cond, hid_uncond, semantic, forced.empty() ? nullptr : forced.data(),
                                &frame, &err)) {
        mm3_json_error(res, 500, err.empty() ? "depth decode failed" : err);
        return;
    }

    std::string codes;
    for (int i = 0; i < frame.n_codes; i++) {
        if (i) {
            codes += ",";
        }
        codes += std::to_string(frame.codes[i]);
    }
    fprintf(stderr, "[MM3-Depth] Frame: semantic=%d %s -> codes [%s] in %.1f ms\n", semantic,
            forced.empty() ? "greedy" : "forced", codes.c_str(), frame.ms);

    std::string body;
    body.reserve((frame.logits_cond.size() + frame.logits_uncond.size() + frame.hiddens.size()) * sizeof(float));
    body.append((const char *) frame.logits_cond.data(), frame.logits_cond.size() * sizeof(float));
    body.append((const char *) frame.logits_uncond.data(), frame.logits_uncond.size() * sizeof(float));
    body.append((const char *) frame.hiddens.data(), frame.hiddens.size() * sizeof(float));

    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%.2f", frame.ms);
    res.set_header("X-MM3-Ms", hdr);
    res.set_header("X-MM3-Codes", codes);
    snprintf(hdr, sizeof(hdr), "%d", frame.n_codes);
    res.set_header("X-MM3-Codebooks", hdr);
    res.set_content(body, "application/octet-stream");
}

// POST /mm3/cond-encode?frames=F
//
// The condition encoder for one window.
//
//   body  F * 8 * 4096 little-endian F32 in torch [1, F, 8*4096] order: frame
//         slowest, then LAYER, then feature. Layer 0 is the LM's last_hidden,
//         layers 1..7 the depth decoder's — exactly `cond_in_w0.bin`.
//
// RESPONSE: raw little-endian F32, 2048 * L values in torch [1, L, 2048] order —
// exactly `cond_out_w0.bin`, and exactly what /mm3/dit-forward and
// /mm3/flow-sample take as their condition block. `X-MM3-Latents` carries L
// (= int(F * 3.4453125)); the body length determines it too, so the header is a
// convenience, not the protocol.
//
// 503 unless MM3 is warm.
static void mm3_handle_cond_encode(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }
    if (!req.has_param("frames")) {
        mm3_json_error(res, 400, "missing ?frames=<AR frame count>");
        return;
    }
    const int64_t F = strtoll(req.get_param_value("frames").c_str(), nullptr, 10);
    if (F <= 0 || F > MM3_COND_MAX_FRAMES) {
        char buf[128];
        snprintf(buf, sizeof(buf), "frames must be in 1..%d", MM3_COND_MAX_FRAMES);
        mm3_json_error(res, 400, buf);
        return;
    }

    const MM3CondConfig & cc   = g_mm3.synth_cfg.cond;
    const size_t          want = (size_t) ((int64_t) cc.num_layers * (int64_t) cc.hidden_dim * F) * sizeof(float);
    if (req.body.size() != want) {
        char buf[208];
        snprintf(buf, sizeof(buf), "body is %zu bytes, expected %zu (= %lld frames * %u layers * %u hidden * 4)",
                 req.body.size(), want, (long long) F, cc.num_layers, cc.hidden_dim);
        mm3_json_error(res, 400, buf);
        return;
    }

    std::vector<float> out;
    int64_t            L = 0;
    std::string        err;
    const auto         t0 = std::chrono::steady_clock::now();
    const bool         ok = mm3_cond_encode(g_mm3, (const float *) req.body.data(), F, out, &L, &err);
    const double       ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        mm3_json_error(res, 500, err.empty() ? "condition encode failed" : err);
        return;
    }

    fprintf(stderr, "[MM3-Cond] Encoded %lld frames -> %lld latent positions in %.1f ms\n", (long long) F,
            (long long) L, ms);

    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%lld", (long long) L);
    res.set_header("X-MM3-Latents", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", ms);
    res.set_header("X-MM3-Ms", hdr);
    res.set_content((const char *) out.data(), out.size() * sizeof(float), "application/octet-stream");
}

// ── /mm3/lm-plan ────────────────────────────────────────────────────────────

static MM3Tokenizer g_mm3_tokenizer;

// Read a JSON array of ints into `out`. Returns false if the key exists but is
// not an array of numbers — a typo must fail loudly, not silently plan nothing.
static bool mm3_json_int_array(yyjson_val * root, const char * key, std::vector<int32_t> * out, bool * present) {
    *present         = false;
    yyjson_val * arr = root ? yyjson_obj_get(root, key) : nullptr;
    if (!arr || yyjson_is_null(arr)) {
        return true;
    }
    if (!yyjson_is_arr(arr)) {
        return false;
    }
    *present = true;
    out->clear();
    out->reserve(yyjson_arr_size(arr));
    yyjson_val *    v;
    yyjson_arr_iter it;
    yyjson_arr_iter_init(arr, &it);
    while ((v = yyjson_arr_iter_next(&it))) {
        if (!yyjson_is_num(v)) {
            return false;
        }
        out->push_back((int32_t) yyjson_get_sint(v));
    }
    return true;
}

static int64_t mm3_json_i64(yyjson_val * root, const char * key, int64_t dflt) {
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    return (v && yyjson_is_num(v)) ? (int64_t) yyjson_get_sint(v) : dflt;
}

static bool mm3_json_bool(yyjson_val * root, const char * key, bool dflt) {
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    return (v && yyjson_is_bool(v)) ? yyjson_get_bool(v) : dflt;
}

// POST /mm3/lm-plan[?dump=N]
//
// The autoregressive planning stage: prompt in, RVQ codes plus the per-frame
// hidden-state block the condition encoder consumes out.
//
// BODY (JSON):
//   prompt          the FULLY ASSEMBLED prompt template, e.g. the contents of
//                   the fixture `tok_prompt_template.txt`. Caption/lyrics
//                   assembly and the reference's text hygiene are a later
//                   increment; this endpoint tokenises what it is given.
//   ids_cond,       explicit token id rows, bypassing the tokenizer. When given
//   ids_uncond      they win over `prompt`; `ids_uncond` alone may be omitted and
//                   is then derived by the 3-token CFG mask rule.
//   max_frames      emitted frames (default 300); capped at mm3.max_audio_frames.
//   seed            std::mt19937_64 seed (default 42).
//   forced_semantic [I]      replay mode: bypass sampling entirely
//   forced_acoustic [I * 7]  flat, iteration-major
//   tokenize_only   true -> return the two id rows and stop. Works COLD (no warm,
//                   no VRAM): the vocab is a GGUF header read.
//   hiddens         true -> append the [F, 8, 4096] conditioning block to the
//                   binary body (39 MB at 300 frames, so opt-in).
//
// QUERY:
//   dump=N          capture the first N iterations' parity tensors and return a
//                   BINARY body instead of JSON.
//
// BINARY BODY (little-endian, no framing; H, SV, NC, F, I, D from the headers):
//   if dump > 0:
//     f32 prefill_hidden [2, H]
//     D x { f32 last_hidden [2, H]   f32 sem_logits [2, SV]   f32 guided [SV]
//           f32 feedback    [2, H]   f32 depth_hidden [NC, H] }
//   if hiddens:
//     f32 frame_hiddens [F, NC+1, H]
//   i32 semantic_all [I]
//   i32 acoustic_all [I, NC]
//
// Parity targets, with forced_semantic = `codes_semantic_all` and
// forced_acoustic = `codes_acoustic_all`:
//   prefill_hidden -> lm_prefill_last_hidden
//   iteration k    -> lm_i{k}_{last_hidden, semantic_logits, guided_logits,
//                             feedback_embed, depth_hidden}
// `guided` carries -inf at masked candidates BY DESIGN (~16333 of 16384); compare
// the finite positions and assert the -inf masks match.
//
// 503 unless MM3 is warm (except tokenize_only).
static void mm3_handle_lm_plan(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    yyjson_doc * doc = req.body.empty() ? nullptr : yyjson_read(req.body.data(), req.body.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!req.body.empty() && (!root || !yyjson_is_obj(root))) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        mm3_json_error(res, 400, "body must be a JSON object");
        return;
    }
    // Everything below must free `doc` on the way out.
    struct DocGuard {
        yyjson_doc * d;
        ~DocGuard() {
            if (d) {
                yyjson_doc_free(d);
            }
        }
    } guard{ doc };

    const MM3LmConfig & lc = g_mm3.lm_cfg;
    const int64_t       NC = (int64_t) lc.num_codebooks - 1;

    // ── token ids: explicit, or tokenise the assembled template ──
    std::vector<int32_t> ids_cond, ids_uncond;
    bool                 have_cond = false, have_uncond = false;
    if (!mm3_json_int_array(root, "ids_cond", &ids_cond, &have_cond) ||
        !mm3_json_int_array(root, "ids_uncond", &ids_uncond, &have_uncond)) {
        mm3_json_error(res, 400, "ids_cond / ids_uncond must be arrays of integers");
        return;
    }
    if (!have_cond) {
        yyjson_val * pv = root ? yyjson_obj_get(root, "prompt") : nullptr;
        if (!pv || !yyjson_is_str(pv)) {
            mm3_json_error(res, 400, "need either \"prompt\" (the assembled template) or \"ids_cond\"");
            return;
        }
        std::string terr;
        if (!mm3_tokenizer_load(g_mm3, &g_mm3_tokenizer, &terr)) {
            mm3_json_error(res, 503, terr);
            return;
        }
        mm3_tokenizer_encode(g_mm3_tokenizer, std::string(yyjson_get_str(pv), yyjson_get_len(pv)), &ids_cond);
        have_cond = true;
    }
    if (ids_cond.size() < 3) {
        mm3_json_error(res, 400, "the prompt must tokenise to at least 3 tokens");
        return;
    }
    if (!have_uncond) {
        mm3_tokenizer_uncond(lc, ids_cond, &ids_uncond);
    } else if (ids_uncond.size() != ids_cond.size()) {
        mm3_json_error(res, 400, "ids_uncond must be the same length as ids_cond");
        return;
    }

    if (mm3_json_bool(root, "tokenize_only", false)) {
        yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
        yyjson_mut_val * orot = yyjson_mut_obj(o);
        yyjson_mut_doc_set_root(o, orot);
        yyjson_mut_obj_add_uint(o, orot, "n_tokens", ids_cond.size());
        yyjson_mut_val * a = yyjson_mut_arr(o);
        for (int32_t id : ids_cond) {
            yyjson_mut_arr_add_int(o, a, id);
        }
        yyjson_mut_obj_add_val(o, orot, "ids_cond", a);
        yyjson_mut_val * b = yyjson_mut_arr(o);
        for (int32_t id : ids_uncond) {
            yyjson_mut_arr_add_int(o, b, id);
        }
        yyjson_mut_obj_add_val(o, orot, "ids_uncond", b);
        char * json = yyjson_mut_write(o, 0, NULL);
        yyjson_mut_doc_free(o);
        res.set_content(json ? json : "{}", "application/json");
        if (json) {
            free(json);
        }
        return;
    }

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }

    // ── options ──
    MM3ArOptions opt;
    opt.max_frames      = mm3_json_i64(root, "max_frames", 300);
    opt.seed            = (uint64_t) mm3_json_i64(root, "seed", 42);
    opt.collect_hiddens = mm3_json_bool(root, "hiddens", false);
    opt.dump_iters      = req.has_param("dump") ? strtoll(req.get_param_value("dump").c_str(), nullptr, 10) : 0;

    std::vector<int32_t> f_sem, f_ac;
    bool                 has_sem = false, has_ac = false;
    if (!mm3_json_int_array(root, "forced_semantic", &f_sem, &has_sem) ||
        !mm3_json_int_array(root, "forced_acoustic", &f_ac, &has_ac)) {
        mm3_json_error(res, 400, "forced_semantic / forced_acoustic must be arrays of integers");
        return;
    }
    if (has_sem != has_ac) {
        mm3_json_error(res, 400, "forced replay needs BOTH forced_semantic and forced_acoustic");
        return;
    }
    if (has_sem) {
        if ((int64_t) f_ac.size() != (int64_t) f_sem.size() * NC) {
            char buf[176];
            snprintf(buf, sizeof(buf), "forced_acoustic has %zu entries, expected %lld (= %zu iterations * %lld)",
                     f_ac.size(), (long long) ((int64_t) f_sem.size() * NC), f_sem.size(), (long long) NC);
            mm3_json_error(res, 400, buf);
            return;
        }
        opt.forced_semantic = f_sem.data();
        opt.forced_acoustic = f_ac.data();
        opt.forced_len      = (int64_t) f_sem.size();
    }

    const bool binary = opt.dump_iters > 0 || opt.collect_hiddens;

    MM3ArResult r;
    std::string err;
    if (!mm3_ar_plan(g_mm3, ids_cond.data(), ids_uncond.data(), (int64_t) ids_cond.size(), opt, &r, &err)) {
        mm3_json_error(res, 500, err.empty() ? "AR planning failed" : err);
        return;
    }

    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.n_frames);
    res.set_header("X-MM3-Frames", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.n_iterations);
    res.set_header("X-MM3-Iterations", hdr);
    snprintf(hdr, sizeof(hdr), "%zu", r.dumps.size());
    res.set_header("X-MM3-Dump-Iters", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.hidden_dim);
    res.set_header("X-MM3-Hidden", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.sem_vocab);
    res.set_header("X-MM3-Sem-Vocab", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) (r.n_codebooks - 1));
    res.set_header("X-MM3-Codebooks", hdr);
    snprintf(hdr, sizeof(hdr), "%zu", ids_cond.size());
    res.set_header("X-MM3-Prompt-Tokens", hdr);
    res.set_header("X-MM3-Eos", r.eos_hit ? "1" : "0");
    snprintf(hdr, sizeof(hdr), "%.1f", r.total_ms);
    res.set_header("X-MM3-Ms", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.prefill_ms);
    res.set_header("X-MM3-Ms-Prefill", hdr);
    snprintf(hdr, sizeof(hdr), "%.3f", r.lm_steps ? r.lm_ms / (double) r.lm_steps : 0.0);
    res.set_header("X-MM3-Ms-Lm-Step", hdr);
    snprintf(hdr, sizeof(hdr), "%.3f", r.n_iterations ? r.depth_ms / (double) r.n_iterations : 0.0);
    res.set_header("X-MM3-Ms-Depth-Frame", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.host_ms);
    res.set_header("X-MM3-Ms-Host", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", (double) g_mm3_lm.kv_bytes / (1024.0 * 1024.0));
    res.set_header("X-MM3-KV-MB", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.nonfinite_logits);
    res.set_header("X-MM3-Nonfinite", hdr);

    if (binary) {
        std::string body;
        size_t      want = 0;
        if (opt.dump_iters > 0) {
            want += (size_t) (2 * r.hidden_dim) * sizeof(float);
            want += r.dumps.size() * (size_t) (2 * r.hidden_dim + 3 * r.sem_vocab + (r.n_codebooks - 1) * r.hidden_dim) *
                    sizeof(float);
        }
        want += r.frame_hiddens.size() * sizeof(float);
        want += (r.semantic_all.size() + r.acoustic_all.size()) * sizeof(int32_t);
        body.reserve(want);

        auto put_f = [&](const std::vector<float> & v) {
            body.append((const char *) v.data(), v.size() * sizeof(float));
        };
        if (opt.dump_iters > 0) {
            put_f(r.prefill_hidden);
            for (const auto & d : r.dumps) {
                put_f(d.last_hidden);
                put_f(d.sem_logits);
                put_f(d.guided);
                put_f(d.feedback);
                put_f(d.depth_hidden);
            }
        }
        put_f(r.frame_hiddens);
        body.append((const char *) r.semantic_all.data(), r.semantic_all.size() * sizeof(int32_t));
        body.append((const char *) r.acoustic_all.data(), r.acoustic_all.size() * sizeof(int32_t));
        res.set_content(body, "application/octet-stream");
        return;
    }

    yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * orot = yyjson_mut_obj(o);
    yyjson_mut_doc_set_root(o, orot);
    yyjson_mut_obj_add_uint(o, orot, "frames", r.n_frames);
    yyjson_mut_obj_add_uint(o, orot, "iterations", r.n_iterations);
    yyjson_mut_obj_add_bool(o, orot, "eos", r.eos_hit);
    yyjson_mut_obj_add_uint(o, orot, "prompt_tokens", ids_cond.size());
    yyjson_mut_obj_add_uint(o, orot, "hidden_dim", r.hidden_dim);
    yyjson_mut_obj_add_uint(o, orot, "num_codebooks", r.n_codebooks);
    yyjson_mut_obj_add_uint(o, orot, "kv_bytes", g_mm3_lm.kv_bytes);
    yyjson_mut_obj_add_uint(o, orot, "kv_positions", g_mm3_lm.n_ctx);
    yyjson_mut_obj_add_uint(o, orot, "nonfinite_logits", r.nonfinite_logits);

    yyjson_mut_val * tm = yyjson_mut_obj(o);
    yyjson_mut_obj_add_val(o, orot, "ms", tm);
    yyjson_mut_obj_add_real(o, tm, "total", r.total_ms);
    yyjson_mut_obj_add_real(o, tm, "prefill", r.prefill_ms);
    yyjson_mut_obj_add_real(o, tm, "lm", r.lm_ms);
    yyjson_mut_obj_add_real(o, tm, "depth", r.depth_ms);
    yyjson_mut_obj_add_real(o, tm, "host", r.host_ms);
    yyjson_mut_obj_add_real(o, tm, "per_lm_step", r.lm_steps ? r.lm_ms / (double) r.lm_steps : 0.0);
    yyjson_mut_obj_add_real(o, tm, "per_frame",
                            r.n_frames ? (r.total_ms - r.prefill_ms) / (double) r.n_frames : 0.0);

    // Codes for EVERY iteration, including the un-emitted iteration 0: emitted
    // frame j is iteration j+1, and a consumer that wants only the emitted codes
    // drops the first entry.
    yyjson_mut_val * sem = yyjson_mut_arr(o);
    for (int32_t v : r.semantic_all) {
        yyjson_mut_arr_add_int(o, sem, v);
    }
    yyjson_mut_obj_add_val(o, orot, "semantic", sem);

    yyjson_mut_val * ac = yyjson_mut_arr(o);
    for (int64_t i = 0; i < r.n_iterations; i++) {
        yyjson_mut_val * row = yyjson_mut_arr(o);
        for (int64_t j = 0; j < r.n_codebooks - 1; j++) {
            yyjson_mut_arr_add_int(o, row, r.acoustic_all[(size_t) (i * (r.n_codebooks - 1) + j)]);
        }
        yyjson_mut_arr_add_val(ac, row);
    }
    yyjson_mut_obj_add_val(o, orot, "acoustic", ac);

    char * json = yyjson_mut_write(o, 0, NULL);
    yyjson_mut_doc_free(o);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

static std::string mm3_json_str(yyjson_val * root, const char * key) {
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v), yyjson_get_len(v)) : std::string();
}

// yyjson_get_num, NOT yyjson_get_real: a JSON integer literal is stored as
// uint/sint and yyjson_get_real returns 0.0 for it, so `"cfg_flow": 2` used to
// read as 0. Same trap as mm3_req_num in mm3-request.h.
static double mm3_json_f64(yyjson_val * root, const char * key, double dflt) {
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    return (v && yyjson_is_num(v)) ? yyjson_get_num(v) : dflt;
}

// Slurp a raw little-endian binary blob. Used only by the parity-replay path of
// /mm3/synth-e2e: the fixture noise blocks are 352 kB each, which is 470 kB of
// base64 per window in a JSON body — a path is the sane wire format for a
// bring-up endpoint that only ever runs against local files.
static bool mm3_read_blob(const std::string & path, std::vector<char> * out, std::string * err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        if (err) {
            *err = "cannot size " + path;
        }
        return false;
    }
    out->resize((size_t) n);
    const size_t got = n > 0 ? fread(out->data(), 1, (size_t) n, f) : 0;
    fclose(f);
    if (got != (size_t) n) {
        if (err) {
            *err = "short read on " + path;
        }
        return false;
    }
    return true;
}

// POST /mm3/tokenize-check
//
// Cheap, COLD-CAPABLE pre-flight for the UI: assemble the prompt template from
// a caption + lyrics exactly as POST /mm3/synth would, tokenise it, and report
// whether it fits the checkpoint's 5,000-token budget. Header-only GGUF read —
// no VRAM, no warm, no GPU.
//
// BODY (JSON):
//   caption  string, required, non-blank
//   lyrics   string, optional; empty -> the instrumental substitution
//   prompt   bool, optional (default false) — echo the assembled template back
//            so a caller can see exactly what the model will be given
//
// RESPONSE: {tokens, limit, ok, caption_clean, lyrics_normalized, instrumental,
//            [prompt]}. `ok` false is a 200, not an error: this endpoint exists
// to answer the question, and only POST /mm3/synth turns a "no" into a 400.
static void mm3_handle_tokenize_check(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    yyjson_doc * doc  = req.body.empty() ? nullptr : yyjson_read(req.body.data(), req.body.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!root || !yyjson_is_obj(root)) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        mm3_json_error(res, 400, "body must be a JSON object");
        return;
    }
    struct DocGuard {
        yyjson_doc * d;
        ~DocGuard() {
            if (d) {
                yyjson_doc_free(d);
            }
        }
    } guard{ doc };

    std::string caption, lyrics, err;
    bool        present = false;
    if (!mm3_req_str(root, "caption", &caption, &present, &err)) {
        mm3_json_error(res, 400, err);
        return;
    }
    if (!present || mm3_str_blank(caption)) {
        mm3_json_error(res, 400, "\"caption\" is required and must be a non-empty string");
        return;
    }
    if (!mm3_req_str(root, "lyrics", &lyrics, &present, &err)) {
        mm3_json_error(res, 400, err);
        return;
    }
    const bool echo = mm3_json_bool(root, "prompt", false);

    bool              instrumental = false;
    const std::string prompt       = mm3_assemble_prompt(caption, lyrics, &instrumental);

    if (!mm3_tokenizer_load(g_mm3, &g_mm3_tokenizer, &err)) {
        mm3_json_error(res, 503, err);
        return;
    }
    std::vector<int32_t> ids;
    mm3_tokenizer_encode(g_mm3_tokenizer, prompt, &ids);

    yyjson_mut_doc * o    = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * orot = yyjson_mut_obj(o);
    yyjson_mut_doc_set_root(o, orot);
    yyjson_mut_obj_add_uint(o, orot, "tokens", ids.size());
    yyjson_mut_obj_add_uint(o, orot, "limit", MM3_MAX_PROMPT_TOKENS);
    yyjson_mut_obj_add_bool(o, orot, "ok", (int64_t) ids.size() <= MM3_MAX_PROMPT_TOKENS);
    yyjson_mut_obj_add_bool(o, orot, "instrumental", instrumental);
    {
        const std::string cc = mm3_clean_caption(caption);
        const std::string ln = mm3_normalize_lyrics(instrumental ? std::string(MM3_INSTRUMENTAL_LYRIC) : lyrics);
        yyjson_mut_obj_add_strcpy(o, orot, "caption_clean", cc.c_str());
        yyjson_mut_obj_add_strcpy(o, orot, "lyrics_normalized", ln.c_str());
    }
    if (echo) {
        yyjson_mut_obj_add_strcpy(o, orot, "prompt", prompt.c_str());
    }
    char * json = yyjson_mut_write(o, 0, NULL);
    yyjson_mut_doc_free(o);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

// POST /mm3/synth-e2e[?dump_latents=1]
//
// DEPRECATED — kept for debugging and parity replay only. Use POST /mm3/synth
// (minimax/mm3-job.h) for anything real: this endpoint does minutes of GPU work
// ON AN HTTPLIB THREAD, so it races the ACE worker for the device, it cannot be
// cancelled, and it reports no progress anywhere but stderr. What it still has
// that /mm3/synth does not: raw `prompt`/`ids_cond` input (no caption/lyrics
// assembly), forced-code and forced-noise parity replay, and ?dump_latents=1.
//
// THE ASSEMBLY, end to end: prompt -> AR plan -> per-window (condition encode ->
// overlap-blended flow sampling) -> vocode -> stitch -> 16-bit stereo WAV.
//
// BODY (JSON):
//   prompt          the FULLY ASSEMBLED prompt template (as /mm3/lm-plan).
//   ids_cond,       explicit token id rows; win over `prompt`. `ids_uncond` may
//   ids_uncond      be omitted and is derived by the 3-token CFG mask rule.
//   max_frames      emitted AR frames (default 300 = 12 s at 25 fps).
//   seed            noise + sampling seed (default 42).
//   steps           Euler steps per window (default: the checkpoint's 30).
//   cfg_flow        flow CFG scale (default: the checkpoint's 1.7).
//
//   PARITY REPLAY (all optional; any subset):
//   forced_semantic / forced_acoustic          inline int arrays, as /mm3/lm-plan
//   forced_semantic_file / forced_acoustic_file  raw i32 blobs instead
//                                              (fixtures codes_{semantic,acoustic}_all.bin)
//   forced_noise_files  array of paths, one per window, each a raw f32
//                       [128, L] channel-major block. An empty string (or a
//                       missing/short entry) falls back to the seeded
//                       derivation for that window, so window 0 alone is legal.
//   dump_dir            where ?dump_latents=1 writes (default %TEMP%/mm3-e2e).
//
// QUERY:
//   dump_latents=1  ALSO write, into dump_dir: `w{k}_latents_final.bin`
//                   ([128, L] f32 channel-major, per window, post overlap
//                   restore — directly comparable to the fixtures'
//                   flow_w{k}_latents_final.bin) and `audio_f32.bin`
//                   ([2, T] planar f32 — comparable to final_audio.bin without
//                   the WAV's 16-bit quantisation).
//
// RESPONSE: audio/wav (16-bit stereo at mm3.voc.sampling_rate), with the run's
// shape and timings in X-MM3-* headers.
//
// 503 unless MM3 is warm.
static void mm3_handle_synth_e2e(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    if (!g_mm3.loaded) {
        mm3_json_error(res, 503, "MiniMax-Music3 is not warm — POST /mm3/warm first");
        return;
    }

    yyjson_doc * doc  = req.body.empty() ? nullptr : yyjson_read(req.body.data(), req.body.size(), 0);
    yyjson_val * root = doc ? yyjson_doc_get_root(doc) : nullptr;
    if (!req.body.empty() && (!root || !yyjson_is_obj(root))) {
        if (doc) {
            yyjson_doc_free(doc);
        }
        mm3_json_error(res, 400, "body must be a JSON object");
        return;
    }
    struct DocGuard {
        yyjson_doc * d;
        ~DocGuard() {
            if (d) {
                yyjson_doc_free(d);
            }
        }
    } guard{ doc };

    const MM3LmConfig & lc  = g_mm3.lm_cfg;
    const int64_t       NCB = (int64_t) lc.num_codebooks - 1;

    // Sampler plugins are deliberately NOT wired here. This is a bring-up /
    // parity endpoint: its whole job is to reproduce the reference arithmetic
    // against the fixtures, and `gr.plugins` left empty is exactly that. Adding
    // them would give the parity harness a way to silently stop measuring
    // parity. Production selection lives on POST /mm3/synth (mm3-request.h).
    MM3GenRequest gr;
    gr.prompt     = mm3_json_str(root, "prompt");
    gr.max_frames = mm3_json_i64(root, "max_frames", 300);
    gr.seed       = (uint64_t) mm3_json_i64(root, "seed", 42);
    gr.steps      = (int) mm3_json_i64(root, "steps", g_mm3.synth_cfg.flow.steps ? g_mm3.synth_cfg.flow.steps : 30);
    gr.cfg_flow   = (float) mm3_json_f64(root, "cfg_flow", g_mm3.synth_cfg.flow.cfg_scale > 0.0f
                                                               ? (double) g_mm3.synth_cfg.flow.cfg_scale
                                                               : 1.7);
    gr.keep_window_latents = req.has_param("dump_latents") && req.get_param_value("dump_latents") != "0";

    bool have_cond = false, have_uncond = false;
    if (!mm3_json_int_array(root, "ids_cond", &gr.ids_cond, &have_cond) ||
        !mm3_json_int_array(root, "ids_uncond", &gr.ids_uncond, &have_uncond)) {
        mm3_json_error(res, 400, "ids_cond / ids_uncond must be arrays of integers");
        return;
    }

    // ── forced codes: inline arrays or raw i32 blobs ──
    bool has_sem = false, has_ac = false;
    if (!mm3_json_int_array(root, "forced_semantic", &gr.forced_semantic, &has_sem) ||
        !mm3_json_int_array(root, "forced_acoustic", &gr.forced_acoustic, &has_ac)) {
        mm3_json_error(res, 400, "forced_semantic / forced_acoustic must be arrays of integers");
        return;
    }
    const std::string sem_file = mm3_json_str(root, "forced_semantic_file");
    const std::string ac_file  = mm3_json_str(root, "forced_acoustic_file");
    std::string       ferr;
    if (!sem_file.empty()) {
        std::vector<char> blob;
        if (!mm3_read_blob(sem_file, &blob, &ferr)) {
            mm3_json_error(res, 400, ferr);
            return;
        }
        gr.forced_semantic.assign((const int32_t *) blob.data(),
                                  (const int32_t *) (blob.data() + (blob.size() / 4) * 4));
        has_sem = true;
    }
    if (!ac_file.empty()) {
        std::vector<char> blob;
        if (!mm3_read_blob(ac_file, &blob, &ferr)) {
            mm3_json_error(res, 400, ferr);
            return;
        }
        gr.forced_acoustic.assign((const int32_t *) blob.data(),
                                  (const int32_t *) (blob.data() + (blob.size() / 4) * 4));
        has_ac = true;
    }
    if (has_sem != has_ac) {
        mm3_json_error(res, 400, "forced replay needs BOTH the semantic and the acoustic codes");
        return;
    }
    if (has_sem && (int64_t) gr.forced_acoustic.size() != (int64_t) gr.forced_semantic.size() * NCB) {
        char buf[192];
        snprintf(buf, sizeof(buf), "forced_acoustic has %zu entries, expected %lld (= %zu iterations * %lld)",
                 gr.forced_acoustic.size(), (long long) ((int64_t) gr.forced_semantic.size() * NCB),
                 gr.forced_semantic.size(), (long long) NCB);
        mm3_json_error(res, 400, buf);
        return;
    }

    // ── forced noise: one raw f32 blob per window ──
    yyjson_val * nf = root ? yyjson_obj_get(root, "forced_noise_files") : nullptr;
    if (nf) {
        if (!yyjson_is_arr(nf)) {
            mm3_json_error(res, 400, "forced_noise_files must be an array of paths");
            return;
        }
        yyjson_val *    v;
        yyjson_arr_iter it;
        yyjson_arr_iter_init(nf, &it);
        while ((v = yyjson_arr_iter_next(&it))) {
            if (!yyjson_is_str(v)) {
                mm3_json_error(res, 400, "forced_noise_files must be an array of paths");
                return;
            }
            const std::string p(yyjson_get_str(v), yyjson_get_len(v));
            if (p.empty()) {
                gr.forced_noise.emplace_back();  // this window derives its own
                continue;
            }
            std::vector<char> blob;
            if (!mm3_read_blob(p, &blob, &ferr)) {
                mm3_json_error(res, 400, ferr);
                return;
            }
            gr.forced_noise.emplace_back((const float *) blob.data(),
                                         (const float *) (blob.data() + (blob.size() / 4) * 4));
        }
    }

    // The job-queue integration point, exercised here as a stderr trace. A
    // production /mm3/synth forwards these into the job progress channel
    // instead; this endpoint has nowhere to send them but the engine log, and
    // an unexercised callback is an unproven callback.
    int64_t       last_pct = -1;
    std::string   last_key;
    MM3ProgressCb progress = [&](const MM3GenProgress & p) {
        const std::string key = std::string(p.stage) + "/" + std::to_string((long long) p.window);
        const int64_t     pct = p.n_steps > 0 ? (p.step * 100 / p.n_steps) : 0;
        if (key != last_key) {
            last_key = key;
            last_pct = -1;
        } else if (pct / 10 == last_pct / 10) {
            return;
        }
        last_pct = pct;
        char where[48] = "";
        if (p.window >= 0) {
            snprintf(where, sizeof(where), " window %lld/%lld", (long long) (p.window + 1), (long long) p.n_windows);
        }
        fprintf(stderr, "[MM3-Pipe] %s%s %lld/%lld\n", p.stage, where, (long long) p.step, (long long) p.n_steps);
    };

    MM3GenResult r;
    std::string  err;
    if (!mm3_generate(g_mm3, gr, &g_mm3_tokenizer, progress, &r, &err)) {
        mm3_json_error(res, 500, err.empty() ? "MM3 generation failed" : err);
        return;
    }

    // ── optional debugging dump ──
    std::string dump_dir;
    if (gr.keep_window_latents) {
        dump_dir = mm3_json_str(root, "dump_dir");
        if (dump_dir.empty()) {
            const char * tmp = std::getenv("TEMP");
            dump_dir         = std::string(tmp ? tmp : ".") + "/mm3-e2e";
        }
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        auto write_f32 = [&](const std::string & name, const std::vector<float> & v) {
            const std::string path = dump_dir + "/" + name;
            FILE *            f    = fopen(path.c_str(), "wb");
            if (!f) {
                fprintf(stderr, "[MM3-Pipe] dump: cannot write %s\n", path.c_str());
                return;
            }
            fwrite(v.data(), sizeof(float), v.size(), f);
            fclose(f);
        };
        for (size_t k = 0; k < r.window_latents.size(); k++) {
            char name[64];
            snprintf(name, sizeof(name), "w%zu_latents_final.bin", k);
            write_f32(name, r.window_latents[k]);
        }
        write_f32("audio_f32.bin", r.audio);
        fprintf(stderr, "[MM3-Pipe] dump: %zu window latent blocks + audio_f32.bin -> %s\n", r.window_latents.size(),
                dump_dir.c_str());
    }

    char hdr[128];
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.frames);
    res.set_header("X-MM3-Frames", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.n_windows);
    res.set_header("X-MM3-Windows", hdr);
    snprintf(hdr, sizeof(hdr), "%lld", (long long) r.n_samples);
    res.set_header("X-MM3-Samples", hdr);
    snprintf(hdr, sizeof(hdr), "%d", r.sample_rate);
    res.set_header("X-MM3-Sample-Rate", hdr);
    snprintf(hdr, sizeof(hdr), "%.4f", r.rms);
    res.set_header("X-MM3-Rms", hdr);
    snprintf(hdr, sizeof(hdr), "%.4f", (double) r.peak);
    res.set_header("X-MM3-Peak", hdr);
    res.set_header("X-MM3-Nan", r.has_nan ? "1" : "0");
    res.set_header("X-MM3-Eos", r.ar.eos_hit ? "1" : "0");
    snprintf(hdr, sizeof(hdr), "%.1f", r.total_ms);
    res.set_header("X-MM3-Ms", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.ar_ms);
    res.set_header("X-MM3-Ms-Ar", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.cond_ms);
    res.set_header("X-MM3-Ms-Cond", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.flow_ms);
    res.set_header("X-MM3-Ms-Flow", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", r.voc_ms);
    res.set_header("X-MM3-Ms-Voc", hdr);
    snprintf(hdr, sizeof(hdr), "%.3f", r.frames ? (r.ar_ms / (double) r.frames) : 0.0);
    res.set_header("X-MM3-Ms-Ar-Frame", hdr);
    snprintf(hdr, sizeof(hdr), "%.1f", (double) r.dit_compute_bytes / (1024.0 * 1024.0));
    res.set_header("X-MM3-Dit-Compute-MB", hdr);
    if (!dump_dir.empty()) {
        res.set_header("X-MM3-Dump-Dir", dump_dir.c_str());
    }
    {
        std::string ov;
        for (size_t k = 0; k < r.window_overlap.size(); k++) {
            ov += (k ? "," : "") + std::to_string((long long) r.window_overlap[k]);
        }
        res.set_header("X-MM3-Overlaps", ov.c_str());
        std::string ls;
        for (size_t k = 0; k < r.window_L.size(); k++) {
            ls += (k ? "," : "") + std::to_string((long long) r.window_L[k]);
        }
        res.set_header("X-MM3-Window-Latents", ls.c_str());
        std::string fn;
        for (size_t k = 0; k < r.forced_noise_used.size(); k++) {
            fn += (k ? "," : "") + std::to_string((long long) r.forced_noise_used[k]);
        }
        res.set_header("X-MM3-Forced-Noise", fn.c_str());
    }

    std::string wav = audio_encode_wav_s16(r.audio.data(), (int) r.n_samples, r.sample_rate);
    res.set_content(wav, "audio/wav");
}

// The single entry point the server calls. Discovers + probes the GGUFs (cheap:
// mmap + header parse, no weight reads) and registers the /mm3/* routes.
// POST /mm3/select-model — body {"lm":"<q>","depth":"<q>","cond":"<q>","dit":"<q>","voc":"<q>"}.
// Legacy alias: {"synth":"<q>"} sets all four non-LM roles at once (the
// pre-split contract; a bundle at that quant satisfies it).
//
// "" (or a missing key) means auto/best-first. Switching a quant drops ONLY the
// residency it feeds (a DiT swap keeps the LM warm); the graph caches key on
// buffer identity and rebuild lazily, but they are torn down here anyway so no
// stale scheduler ever holds a dead buffer. Idempotent — posting the current
// selection is a no-op, so the UI can send the whole selection on every change.
static void mm3_handle_select_model(const httplib::Request & req, httplib::Response & res) {
    std::lock_guard<std::mutex> lock(g_mm3_mutex);

    yyjson_doc * in = req.body.empty() ? nullptr : yyjson_read(req.body.c_str(), req.body.size(), 0);
    yyjson_val * root_in = in ? yyjson_doc_get_root(in) : nullptr;
    auto         has_key = [&](const char * key) {
        yyjson_val * v = root_in ? yyjson_obj_get(root_in, key) : nullptr;
        return v && yyjson_is_str(v);
    };
    auto get_str = [&](const char * key) -> std::string {
        yyjson_val * v = root_in ? yyjson_obj_get(root_in, key) : nullptr;
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : std::string();
    };
    const std::string want_lm = get_str("lm");
    std::string       want_roles[MM3_R_COUNT];
    if (has_key("synth") && !has_key("dit")) {
        // Legacy caller: one quant token for the whole non-LM stack.
        const std::string synth = get_str("synth");
        for (int r = 0; r < MM3_R_COUNT; r++) {
            want_roles[r] = synth;
        }
        // cond/voc exist only near-native; an auto pick keeps a legacy Q-token
        // request loadable on a split install (the bundle covers it otherwise).
        if (!synth.empty()) {
            auto offered = [&](MM3SynthRole r) {
                for (const auto & v : g_mm3.role_variants[r]) {
                    if (v.quant == synth) {
                        return true;
                    }
                }
                return false;
            };
            if (!offered(MM3_R_COND)) {
                want_roles[MM3_R_COND] = "";
            }
            if (!offered(MM3_R_VOC)) {
                want_roles[MM3_R_VOC] = "";
            }
            if (!offered(MM3_R_DEPTH)) {
                want_roles[MM3_R_DEPTH] = "";
            }
        }
    } else {
        for (int r = 0; r < MM3_R_COUNT; r++) {
            want_roles[r] = get_str(MM3_SYNTH_ROLE[r]);
        }
    }
    if (in) {
        yyjson_doc_free(in);
    }

    const bool was_loaded = g_mm3.loaded;
    // Graph state points into the model's buffers; mm3_select_variant drops
    // (some of) those, so tear the graphs down first (same order as
    // /mm3/unload). They rebuild lazily against whatever is resident.
    bool changes = want_lm != g_mm3.want_lm_quant;
    for (int r = 0; r < MM3_R_COUNT; r++) {
        changes = changes || want_roles[r] != g_mm3.want_role_quant[r];
    }
    if (changes) {
        mm3_vocoder_free(&g_mm3_voc);
        mm3_dit_free(&g_mm3_dit);
        mm3_depth_free(&g_mm3_depth);
        mm3_cond_free(&g_mm3_cond);
        mm3_lm_free(&g_mm3_lm);
        // The LM/depth files are part of the AR cache key, so a role change
        // could never hit it again anyway — this just stops a dead slot from
        // holding hundreds of MB of host RAM for the rest of the session.
        mm3_ar_cache_clear("model selection changed");
    }

    std::string err;
    if (!mm3_select_variant(&g_mm3, want_lm, want_roles, &err)) {
        mm3_json_error(res, 400, err);
        return;
    }

    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_bool(doc, root, "changed", changes);
    yyjson_mut_obj_add_bool(doc, root, "unloaded", changes && was_loaded);
    yyjson_mut_obj_add_strcpy(doc, root, "lm", g_mm3.lm_file.found ? g_mm3.lm_file.name.c_str() : "");
    for (int r = 0; r < MM3_R_COUNT; r++) {
        yyjson_mut_obj_add_strcpy(doc, root, MM3_SYNTH_ROLE[r],
                                  g_mm3.role_file[r].found ? g_mm3.role_file[r].name.c_str() : "");
    }
    // Legacy key: pre-split clients read back "synth"; the DiT stands in.
    yyjson_mut_obj_add_strcpy(doc, root, "synth",
                              g_mm3.role_file[MM3_R_DIT].found ? g_mm3.role_file[MM3_R_DIT].name.c_str() : "");
    yyjson_mut_obj_add_bool(doc, root, "available", mm3_available(g_mm3));
    for (const auto & e : g_mm3.meta_errors) {
        yyjson_mut_obj_add_strcpy(doc, root, "error", e.c_str());
        break;
    }
    char * json = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
}

static void mm3_register_routes(httplib::Server & svr, const char * models_dir) {
    mm3_discover(&g_mm3, models_dir);

    svr.Get("/mm3/props", mm3_handle_props);
    svr.Post("/mm3/warm", mm3_handle_warm);
    svr.Post("/mm3/unload", mm3_handle_unload);
    svr.Post("/mm3/select-model", mm3_handle_select_model);
    svr.Post("/mm3/voc-decode", mm3_handle_voc_decode);
    svr.Post("/mm3/dit-forward", mm3_handle_dit_forward);
    svr.Post("/mm3/flow-sample", mm3_handle_flow_sample);
    svr.Post("/mm3/depth-frame", mm3_handle_depth_frame);
    svr.Post("/mm3/cond-encode", mm3_handle_cond_encode);
    svr.Post("/mm3/lm-plan", mm3_handle_lm_plan);
    svr.Post("/mm3/tokenize-check", mm3_handle_tokenize_check);
    svr.Post("/mm3/synth-e2e", mm3_handle_synth_e2e);
    // POST /mm3/synth and GET /mm3/job are registered separately by
    // mm3_register_job_routes() (minimax/mm3-job.h) — they need the job system,
    // which is defined further down hot-step-server.cpp than this include.
}
