#pragma once
// yue2/yue2-pipeline.h — YuE2 end-to-end orchestration: plan -> semantic ->
// NAR -> VAE decode -> stitch/clamp. Milestones M7/M8,
// docs/plans/yue2/06-engine-port-plan.md §7.
//
// HOT-Step file (no acestep.cpp analog). v1 scope, per the plan doc: a
// single-take, non-streaming render. Mirrors mm3-pipeline.h's staged-call
// shape (plan/semantic/synthesize/decode, a progress callback, cancellation
// checks) but none of MM3's ensemble-take/streaming/AR-cache machinery,
// which YuE2 v1 has no equivalent of.
//
// Sampler knobs (temperature/top_p/top_k/repetition_penalty/min_tokens/
// max_tokens) are LOCKED to the checkpoint's own GGUF-declared per-stage
// defaults (open question §8, resolved) -- not exposed on Yue2Request.
//
// ── Rough edges (listed, not polished — task rule) ──────────────────────────
// - Cancellation is checked between AR-decode tokens (plan/semantic) and
//   between NAR chunks, but NOT between VAE tiles -- yue2_vae_decode_tiled
//   (yue2-vae-graph.h) takes no cancel callback. A cancel during a long VAE
//   decode only takes effect once decode finishes.
// - The resolved seed (when the caller didn't supply one) is not echoed back
//   anywhere in the result -- a caller cannot reproduce an unseeded render.
// - noise_source="fixture" is implemented (reads a raw FP32 file) but only
//   exercised here for the parity-bonus check, never by the creation UI, per
//   the plan's own "validator-only" framing.
// - CFG's two branches are two fully independent AR decode loops (KV cache,
//   prefill, per-step decode each), per yue2-lm-graph.h's own design
//   decision -- roughly 2x semantic-stage AR bandwidth whenever cfg_scale!=1.

#include "yue2-lm-graph.h"
#include "yue2-model.h"
#include "yue2-nar-graph.h"
#include "yue2-request.h"
#include "yue2-sample.h"
#include "yue2-tokenizer.h"
#include "yue2-vae-graph.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

// ── Progress reporting ──────────────────────────────────────────────────────

enum Yue2Stage { YUE2_STAGE_PLAN = 0, YUE2_STAGE_SEMANTIC = 1, YUE2_STAGE_NAR = 2, YUE2_STAGE_VAE = 3 };

struct Yue2Progress {
    Yue2Stage stage;
    int64_t   step  = 0;
    int64_t   total = 0;  // <= 0 means "not knowable yet" (caller should treat as unknown, not zero-of-zero)
};

using Yue2ProgressFn = std::function<void(const Yue2Progress &)>;

// ── Result ───────────────────────────────────────────────────────────────────

struct Yue2PipelineResult {
    std::vector<float>   audio_planar;  // [2, samples] -- L then R, matches audio_encode_wav_s16's own contract
    int64_t               samples     = 0;
    int                    sample_rate = 48000;

    std::string score_abc;              // decoded plan-stage ABC text; empty if the plan stage was skipped
    std::vector<int32_t> semantic_ids;  // raw codec ids [0, CODEC_SIZE), CODEC_OFFSET already subtracted

    // Per-stage: "skipped" | "eos" | "limit_hit" | "" (absent -- only plan/
    // semantic carry a terminator/limit concept at all; nar/vae stay "").
    std::string stage_end_reason[4];
    double       stage_ms[4] = { 0.0, 0.0, 0.0, 0.0 };

    int64_t total_frames = 0;  // NAR latent frame count (== semantic codec token count)

    // Aggregate: "completed" | "limit_hit" | "cancelled" | "failed" -- set by
    // the caller from this struct + the pipeline's own return value, per
    // docs/plans/yue2/06-engine-port-plan.md §7's aggregation rule (this
    // function only fills stage_end_reason; yue2-job.h derives end_reason).
    std::string end_reason;
};

// ── Whole-song noise: splitmix64 + Box-Muller, drawn ONCE, sliced per chunk ──
//
// Deliberately NOT mm3_fill_noise()'s per-window-reseed convention (03 §3.10/
// open question §6): YuE2 draws one buffer for the entire song and slices it,
// matching the reference's own single torch.randn((frames,64)) draw. This is
// the engine's own declared deterministic algorithm for PRODUCTION generation
// -- it does not attempt to bit-match PyTorch's CPU RNG (infeasible, per
// 03-reference-numerics.md §5/§0). Fixture/parity mode instead replays stored
// bytes -- see yue2_read_f32_file below.

static inline uint64_t yue2_splitmix64(uint64_t & s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void yue2_fill_noise(uint64_t seed, std::vector<float> * out, int64_t n) {
    out->assign((size_t) n, 0.0f);
    uint64_t s = seed;
    yue2_splitmix64(s);  // discard one, so seed 0 does not start on a raw zero state
    for (int64_t i = 0; i < n; i += 2) {
        double u1, u2;
        do {
            u1 = (double) (yue2_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 <= 1e-300);
        u2                        = (double) (yue2_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
        const double r            = std::sqrt(-2.0 * std::log(u1));
        const double th           = 6.283185307179586476925286766559 * u2;
        (*out)[(size_t) i]        = (float) (r * std::cos(th));
        if (i + 1 < n) {
            (*out)[(size_t) (i + 1)] = (float) (r * std::sin(th));
        }
    }
}

static bool yue2_read_f32_file(const std::string & path, std::vector<float> * out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    const int64_t n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    const int64_t n = ftello(f);
    fseeko(f, 0, SEEK_SET);
#endif
    if (n <= 0 || (n % 4) != 0) {
        fclose(f);
        return false;
    }
    out->assign((size_t) (n / 4), 0.0f);
    const size_t got = fread(out->data(), 1, (size_t) n, f);
    fclose(f);
    return got == (size_t) n;
}

// `chunk_ranges(frames, prefix_tokens, context)` — 03-reference-numerics.md
// §3.7. size is a function of PROMPT length, recomputed per request, never
// hardcoded. Empty return means "size < 1" (prompt too long for the context
// window) -- the reference itself raises in this case.
static std::vector<std::pair<int64_t, int64_t>> yue2_chunk_ranges(int64_t frames, int64_t prefix_tokens,
                                                                   int64_t context) {
    std::vector<std::pair<int64_t, int64_t>> out;
    const int64_t                             size = (context - prefix_tokens - 3) / 2;
    if (size < 1 || frames <= 0) {
        return out;
    }
    for (int64_t a = 0; a < frames; a += size) {
        out.push_back({ a, std::min(a + size, frames) });
    }
    return out;
}

// ── Stage 1: plan (ABC) ──────────────────────────────────────────────────────
static uint64_t yue2_token_hash(const std::vector<int32_t> & ids) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int32_t id : ids) {
        hash ^= (uint32_t) id;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
//
// Skipped entirely for cot=="off" (no ABC span exists at all) and whenever
// the request supplies `abc` externally (the model call is skipped; the
// supplied text is tokenised and used as-is). No CFG here -- the ABC stage
// is never guided in the reference (only 02_semantic's fixtures carry a
// cond/uncond split).
static bool yue2_run_plan_stage(Yue2Model & m, const BPETokenizer & tok, const Yue2Request & req,
                                 std::mt19937_64 & rng, std::atomic<bool> * cancel, const Yue2ProgressFn & progress,
                                 std::vector<int32_t> * abc_ids_out, std::string * score_abc_out,
                                 std::string * stage_end_reason, double * stage_ms, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_ar_step_profile_reset();
    abc_ids_out->clear();
    score_abc_out->clear();

    if (req.cot == YUE2_COT_OFF) {
        *stage_end_reason = "skipped";
        return true;
    }
    if (req.abc_provided) {
        try {
            *abc_ids_out = yue2_bpe_encode(&tok, req.abc);
            yue2_validate_abc_ids(*abc_ids_out, "yue2_run_plan_stage");
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("plan stage: ") + e.what();
            }
            return false;
        }
        *score_abc_out     = req.abc;
        *stage_end_reason = "skipped";
        return true;
    }

    const std::vector<int32_t> prefix = yue2_token_prefixes(&tok, req.style, req.lyrics, req.cot, nullptr);

    Yue2SamplingParams sp;
    sp.temperature        = m.lm_cfg.abc.temperature;
    sp.top_p               = m.lm_cfg.abc.top_p;
    sp.top_k               = (int) m.lm_cfg.abc.top_k;
    sp.repetition_penalty = m.lm_cfg.abc.repetition_penalty;
    sp.penalty_window     = (int) m.lm_cfg.abc.penalty_window;
    sp.min_tokens          = (int) m.lm_cfg.abc.min_tokens;
    sp.max_tokens          = (int) m.lm_cfg.abc.max_tokens;

    Yue2ArKvCache cache;
    const int64_t capacity = (int64_t) prefix.size() + sp.max_tokens + 4;
    if (!yue2_ar_kv_cache_alloc(m, capacity, &cache, err)) {
        return false;
    }

    Yue2ArForwardResult pre;
    if (!yue2_ar_prefill(m, cache, prefix, { (int64_t) prefix.size() - 1 }, {}, &pre, err)) {
        yue2_ar_kv_cache_free(&cache);
        return false;
    }

    std::vector<int32_t> history;
    std::vector<float>   logits = pre.logits;
    bool                  hit_eos = false;
    int64_t                step    = 0;
    for (; step < sp.max_tokens; step++) {
        if (cancel && cancel->load()) {
            if (err) {
                *err = "cancelled";
            }
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        std::vector<float> scores = logits;
        yue2_distribution(scores, sp, YUE2_ABC_END, 0, YUE2_EOD, history, step, /*legacy_off=*/false);
        const int64_t tok_id = sp.temperature == 0.0f ? yue2_sample_argmax(scores) : yue2_sample_draw(scores, rng);
        if (progress) {
            progress({ YUE2_STAGE_PLAN, step + 1, sp.max_tokens });
        }
        if (tok_id == YUE2_ABC_END) {
            hit_eos = true;
            break;
        }
        history.push_back((int32_t) tok_id);
        if (!yue2_ar_decode_step(m, cache, (int32_t) tok_id, &logits, err)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
    }
    yue2_ar_kv_cache_free(&cache);

    *abc_ids_out       = history;
    *score_abc_out     = yue2_bpe_decode(&tok, history);
    fprintf(stderr, "[YuE2-AR-Tokens] plan n=%zu hash=%016llx\n", history.size(),
            (unsigned long long) yue2_token_hash(history));
    *stage_end_reason = hit_eos ? "eos" : "limit_hit";
    *stage_ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    yue2_ar_step_profile_log("plan");
    return true;
}

// ── Stage 2: semantic (codec AR, possibly CFG'd) ────────────────────────────
static bool yue2_run_semantic_stage(Yue2Model & m, const BPETokenizer & tok, const Yue2Request & req,
                                     const std::vector<int32_t> & abc_ids, bool have_abc, std::mt19937_64 & rng,
                                     std::atomic<bool> * cancel, const Yue2ProgressFn & progress,
                                     std::vector<int32_t> * prefix_ids_out, std::vector<int32_t> * codec_ids_out,
                                     std::string * stage_end_reason, double * stage_ms, std::string * err) {
    const auto                  t0      = std::chrono::steady_clock::now();
    yue2_ar_step_profile_reset();
    const std::vector<int32_t> * abc_ptr = have_abc ? &abc_ids : nullptr;

    std::vector<int32_t> pos_prefix;
    try {
        pos_prefix = yue2_token_prefixes(&tok, req.style, req.lyrics, req.cot, abc_ptr);
    } catch (const std::exception & e) {
        if (err) {
            *err = std::string("semantic stage: ") + e.what();
        }
        return false;
    }
    *prefix_ids_out = pos_prefix;

    const bool use_cfg = req.cfg_scale != 1.0f;

    Yue2SamplingParams sp;
    sp.temperature        = m.lm_cfg.semantic.temperature;
    sp.top_p               = m.lm_cfg.semantic.top_p;
    sp.top_k               = (int) m.lm_cfg.semantic.top_k;
    sp.repetition_penalty = m.lm_cfg.semantic.repetition_penalty;
    sp.penalty_window     = (int) m.lm_cfg.semantic.penalty_window;
    sp.min_tokens          = (int) m.lm_cfg.semantic.min_tokens;
    sp.max_tokens          = (int) m.lm_cfg.semantic.max_tokens;
    const bool preview_capped = req.preview_max_frames > 0 && req.preview_max_frames < sp.max_tokens;
    if (preview_capped) sp.max_tokens = req.preview_max_frames;
    const bool legacy_off  = (req.cot == YUE2_COT_OFF);
    const int64_t legal_lo = YUE2_CODEC_OFFSET;
    const int64_t legal_hi = YUE2_CODEC_OFFSET + YUE2_CODEC_SIZE;

    Yue2ArKvCache cond_cache;
    if (!yue2_ar_kv_cache_alloc(m, (int64_t) pos_prefix.size() + sp.max_tokens + 4, &cond_cache, err)) {
        return false;
    }
    Yue2ArForwardResult cond_pre;
    if (!yue2_ar_prefill(m, cond_cache, pos_prefix, { (int64_t) pos_prefix.size() - 1 }, {}, &cond_pre, err)) {
        yue2_ar_kv_cache_free(&cond_cache);
        return false;
    }
    std::vector<float> cond_logits = cond_pre.logits;

    Yue2ArKvCache       uncond_cache;
    bool                 have_uncond = false;
    std::vector<float>   uncond_logits;
    if (use_cfg) {
        std::vector<int32_t> neg_prefix;
        try {
            neg_prefix = yue2_negative_prefix(&tok, req.cot, abc_ptr);
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("semantic stage (negative prefix): ") + e.what();
            }
            yue2_ar_kv_cache_free(&cond_cache);
            return false;
        }
        if (!yue2_ar_kv_cache_alloc(m, (int64_t) neg_prefix.size() + sp.max_tokens + 4, &uncond_cache, err)) {
            yue2_ar_kv_cache_free(&cond_cache);
            return false;
        }
        Yue2ArForwardResult neg_pre;
        if (!yue2_ar_prefill(m, uncond_cache, neg_prefix, { (int64_t) neg_prefix.size() - 1 }, {}, &neg_pre, err)) {
            yue2_ar_kv_cache_free(&cond_cache);
            yue2_ar_kv_cache_free(&uncond_cache);
            return false;
        }
        uncond_logits = neg_pre.logits;
        have_uncond    = true;
    }

    std::vector<int32_t> history;
    bool                   hit_eos          = false;
    bool                   eos_by_threshold = false;
    int64_t                 step    = 0;
    for (; step < sp.max_tokens; step++) {
        if (cancel && cancel->load()) {
            if (err) {
                *err = "cancelled";
            }
            yue2_ar_kv_cache_free(&cond_cache);
            if (have_uncond) {
                yue2_ar_kv_cache_free(&uncond_cache);
            }
            return false;
        }
        std::vector<float> blended;
        if (have_uncond) {
            yue2_cfg_blend(cond_logits, uncond_logits, req.cfg_scale, &blended);
        } else {
            blended = cond_logits;
        }
        // Ending controls (yue2-request.h). The bias lands on the raw logit
        // BEFORE the reference's mask/penalty/temperature/top-k/top-p chain,
        // so the chain itself is untouched; the threshold reads the chain's
        // OUTPUT, the very distribution the draw below samples from.
        if (req.end_bias != 0.0f && step >= (int64_t) sp.min_tokens) {
            const float t_sec = (float) step * 0.04f;  // 25 Hz codec frames
            float       w     = 0.0f;
            if (t_sec >= req.end_bias_from_sec) {
                w = req.end_bias_ramp_sec > 0.0f
                        ? std::min(1.0f, (t_sec - req.end_bias_from_sec) / req.end_bias_ramp_sec)
                        : 1.0f;
            }
            if (w > 0.0f && std::isfinite(blended[(size_t) YUE2_MUSIC_END])) {
                blended[(size_t) YUE2_MUSIC_END] += w * req.end_bias;
            }
        }
        // YUE2_END_TRACE=<path>: uncensored per-step record of MUSIC_END's
        // probability and rank BEFORE the sampler chain (softmax over the
        // legal ids [legal_lo,legal_hi) + END, raw logits, no penalty/temp/
        // top-k/p) and AFTER it, plus the sampled token — the evidence for
        // "the model reached its ending and END lost the draw". Read-only:
        // it touches neither the scores nor the RNG.
        static const char * end_trace_path = std::getenv("YUE2_END_TRACE");
        double              tr_p_pre = -1.0, tr_p_post = -1.0;
        int64_t             tr_r_pre = -1, tr_r_post = -1;
        if (end_trace_path && *end_trace_path) {
            auto stat = [&](const std::vector<float> & s, bool legal_only, double * p, int64_t * rank) {
                float mx = -INFINITY;
                for (int64_t v = 0; v < (int64_t) s.size(); v++) {
                    const bool in = legal_only ? ((v >= legal_lo && v < legal_hi) || v == YUE2_MUSIC_END) : true;
                    if (in && std::isfinite(s[(size_t) v]) && s[(size_t) v] > mx) {
                        mx = s[(size_t) v];
                    }
                }
                double  z = 0.0;
                int64_t r = 0;
                const float se = s[(size_t) YUE2_MUSIC_END];
                for (int64_t v = 0; v < (int64_t) s.size(); v++) {
                    const bool in = legal_only ? ((v >= legal_lo && v < legal_hi) || v == YUE2_MUSIC_END) : true;
                    if (!in || !std::isfinite(s[(size_t) v])) {
                        continue;
                    }
                    z += std::exp((double) (s[(size_t) v] - mx));
                    if (s[(size_t) v] > se) {
                        r++;
                    }
                }
                *p    = std::isfinite(se) ? std::exp((double) (se - mx)) / z : 0.0;
                *rank = std::isfinite(se) ? r : -1;
            };
            stat(blended, true, &tr_p_pre, &tr_r_pre);
        }
        yue2_distribution(blended, sp, YUE2_MUSIC_END, legal_lo, legal_hi, history, step, legacy_off);
        if (end_trace_path && *end_trace_path) {
            float mx = -INFINITY;
            for (float v : blended) {
                if (std::isfinite(v) && v > mx) {
                    mx = v;
                }
            }
            double  z = 0.0;
            int64_t r = 0;
            const float se = blended[(size_t) YUE2_MUSIC_END];
            for (float v : blended) {
                if (!std::isfinite(v)) {
                    continue;
                }
                z += std::exp((double) (v - mx));
                if (v > se) {
                    r++;
                }
            }
            tr_p_post = std::isfinite(se) ? std::exp((double) (se - mx)) / z : 0.0;
            tr_r_post = std::isfinite(se) ? r : -1;
        }
        if (req.end_threshold > 0.0f && step >= (int64_t) sp.min_tokens &&
            std::isfinite(blended[(size_t) YUE2_MUSIC_END])) {
            // P(END) under the post-chain distribution: logsumexp over the
            // finite entries (masked ones are -inf and contribute nothing).
            float mx = -INFINITY;
            for (float v : blended) {
                if (std::isfinite(v) && v > mx) {
                    mx = v;
                }
            }
            double z = 0.0;
            for (float v : blended) {
                if (std::isfinite(v)) {
                    z += std::exp((double) (v - mx));
                }
            }
            const double p_end = std::exp((double) (blended[(size_t) YUE2_MUSIC_END] - mx)) / z;
            if (p_end >= (double) req.end_threshold) {
                if (end_trace_path && *end_trace_path) {
                    if (FILE * tf = fopen(end_trace_path, "a")) {
                        fprintf(tf, "%lld\t%.2f\t%.6g\t%lld\t%.6g\t%lld\t%d\tforced_threshold\n", (long long) step,
                                (double) step * 0.04, tr_p_pre, (long long) tr_r_pre, tr_p_post, (long long) tr_r_post,
                                (int) YUE2_MUSIC_END);
                        fclose(tf);
                    }
                }
                hit_eos           = true;
                eos_by_threshold  = true;
                break;
            }
        }
        const int64_t tok_id = sp.temperature == 0.0f ? yue2_sample_argmax(blended) : yue2_sample_draw(blended, rng);
        if (end_trace_path && *end_trace_path) {
            if (FILE * tf = fopen(end_trace_path, "a")) {
                if (step == 0) {
                    fprintf(tf, "# semantic stage: prefix %lld ids, seed %llu, threshold %.3f, bias %.3f\n",
                            (long long) pos_prefix.size(), (unsigned long long) req.seed, (double) req.end_threshold,
                            (double) req.end_bias);
                    fprintf(tf, "# step\tsec\tp_end_pre\trank_pre\tp_end_post\trank_post\tsampled\tnote\n");
                }
                fprintf(tf, "%lld\t%.2f\t%.6g\t%lld\t%.6g\t%lld\t%lld\t%s\n", (long long) step, (double) step * 0.04,
                        tr_p_pre, (long long) tr_r_pre, tr_p_post, (long long) tr_r_post, (long long) tok_id,
                        tok_id == YUE2_MUSIC_END ? "sampled_end" : "");
                fclose(tf);
            }
        }
        if (progress) {
            progress({ YUE2_STAGE_SEMANTIC, step + 1, sp.max_tokens });
        }
        if (tok_id == YUE2_MUSIC_END) {
            hit_eos = true;
            break;
        }
        history.push_back((int32_t) tok_id);
        codec_ids_out->push_back((int32_t) (tok_id - YUE2_CODEC_OFFSET));

        if (!yue2_ar_decode_step(m, cond_cache, (int32_t) tok_id, &cond_logits, err)) {
            yue2_ar_kv_cache_free(&cond_cache);
            if (have_uncond) {
                yue2_ar_kv_cache_free(&uncond_cache);
            }
            return false;
        }
        if (have_uncond) {
            if (!yue2_ar_decode_step(m, uncond_cache, (int32_t) tok_id, &uncond_logits, err)) {
                yue2_ar_kv_cache_free(&cond_cache);
                yue2_ar_kv_cache_free(&uncond_cache);
                return false;
            }
        }
    }

    yue2_ar_kv_cache_free(&cond_cache);
    if (have_uncond) {
        yue2_ar_kv_cache_free(&uncond_cache);
    }

    *stage_end_reason = hit_eos ? (eos_by_threshold ? "eos_threshold" : "eos") :
        (preview_capped ? "preview_limit" : "limit_hit");
    fprintf(stderr, "[YuE2-AR-Tokens] semantic n=%zu hash=%016llx\n", codec_ids_out->size(),
            (unsigned long long) yue2_token_hash(*codec_ids_out));
    *stage_ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    yue2_ar_step_profile_log("semantic");
    return true;
}

// ── Stage 3: NAR (per-chunk AR-prefill + 32-step midpoint solve) ───────────
static bool yue2_run_nar_stage(Yue2Model & m, const Yue2Request & req, const std::vector<int32_t> & prefix_ids,
                                const std::vector<int32_t> & codec_ids, std::atomic<bool> * cancel,
                                const Yue2ProgressFn & progress, std::vector<float> * song_latents_out,
                                int64_t * total_frames_out, double * stage_ms, std::string * err) {
    const auto    t0     = std::chrono::steady_clock::now();
    const int64_t frames = (int64_t) codec_ids.size();
    if (frames <= 0) {
        if (err) {
            *err = "NAR stage: semantic stage produced zero codec frames";
        }
        return false;
    }

    const auto ranges =
        yue2_chunk_ranges(frames, (int64_t) prefix_ids.size(), (int64_t) m.lm_cfg.context_length);
    if (ranges.empty()) {
        if (err) {
            *err = "NAR stage: chunk_ranges computed size < 1 (prompt too long for the context window)";
        }
        return false;
    }

    const int64_t LD = (int64_t) m.lm_cfg.latent_dim;

    std::vector<float> noise;
    if (req.noise_source == YUE2_NOISE_FIXTURE) {
        if (!yue2_read_f32_file(req.noise_fixture_path, &noise)) {
            if (err) {
                *err = "NAR stage: noise_source=\"fixture\" -- cannot read " + req.noise_fixture_path;
            }
            return false;
        }
        if ((int64_t) noise.size() != frames * LD) {
            if (err) {
                *err = "NAR stage: noise_source=\"fixture\" geometry mismatch (expected " +
                       std::to_string(frames * LD) + " floats, got " + std::to_string(noise.size()) + ")";
            }
            return false;
        }
    } else {
        yue2_fill_noise(req.seed, &noise, frames * LD);
    }

    song_latents_out->clear();
    song_latents_out->reserve((size_t) (frames * LD));

    for (size_t ci = 0; ci < ranges.size(); ci++) {
        if (cancel && cancel->load()) {
            if (err) {
                *err = "cancelled";
            }
            return false;
        }
        const int64_t a         = ranges[ci].first;
        const int64_t b         = ranges[ci].second;
        const int64_t chunk_len = b - a;

        std::vector<int32_t> ar_prefix_ids = prefix_ids;
        ar_prefix_ids.reserve(prefix_ids.size() + (size_t) chunk_len + 1);
        for (int64_t i = a; i < b; i++) {
            ar_prefix_ids.push_back(codec_ids[(size_t) i] + YUE2_CODEC_OFFSET);
        }
        ar_prefix_ids.push_back(YUE2_MUSIC_END);

        Yue2NarChunk chunk;
        const auto nar_init_start = std::chrono::steady_clock::now();
        if (!yue2_nar_chunk_init(m, ar_prefix_ids, chunk_len, &chunk, err)) {
            return false;
        }
        const double nar_init_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - nar_init_start).count();

        std::vector<float> noise_slice(noise.begin() + (size_t) (a * LD), noise.begin() + (size_t) (b * LD));
        Yue2NarSolveResult solve;
        const auto nar_solve_start = std::chrono::steady_clock::now();
        const bool ok = yue2_nar_solve_midpoint(m, chunk, noise_slice, req.ode_steps, {}, false, &solve, err);
        const double nar_solve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - nar_solve_start).count();
        yue2_nar_chunk_free(&chunk);
        if (!ok) {
            return false;
        }

        song_latents_out->insert(song_latents_out->end(), solve.final_latents.begin(), solve.final_latents.end());
        fprintf(stderr, "[YuE2-NAR-Chunk] index=%zu frames=%lld prefix_tokens=%zu init_ms=%.1f solve_ms=%.1f velocity_ms=%.1f calls=%d\n",
                ci, (long long) chunk_len, ar_prefix_ids.size(), nar_init_ms, nar_solve_ms,
                solve.total_velocity_ms, solve.velocity_calls);
        if (progress) {
            progress({ YUE2_STAGE_NAR, (int64_t) ci + 1, (int64_t) ranges.size() });
        }
    }

    *total_frames_out = frames;
    *stage_ms          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ── Stage 4: VAE decode (tiled — degrades to one "tile" == the whole song
//    when total_frames <= decode_core_frames, so this is the general path) ──
static bool yue2_run_vae_stage(Yue2Model & m, const Yue2Request & req, const std::vector<float> & song_latents,
                                int64_t total_frames, const Yue2ProgressFn & progress,
                                std::vector<float> * audio_planar_out, int64_t * samples_out, double * stage_ms,
                                std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    // want_lm=false: leave LM residency exactly as it is, only ensure the
    // requested VAE variant is resident (idempotent if already loaded).
    if (!yue2_load_parts(&m, /*want_lm=*/false, /*want_vae=*/true, req.vae_variant, /*want_encoder=*/false, err)) {
        return false;
    }

    const int64_t LD = (int64_t) m.vae_cfg.latent_dim;
    if ((int64_t) song_latents.size() != total_frames * LD) {
        if (err) {
            *err = "VAE stage: song_latents size mismatch";
        }
        return false;
    }

    // Transpose position-major [T,64] -> channel-major [64,T], the layout
    // yue2_vae_decode_tiled/yue2_vae_decode expect (yue2-vae-graph.h header).
    std::vector<float> latent_ct((size_t) (LD * total_frames));
    for (int64_t t = 0; t < total_frames; t++) {
        for (int64_t c = 0; c < LD; c++) {
            latent_ct[(size_t) (c * total_frames + t)] = song_latents[(size_t) (t * LD + c)];
        }
    }

    const int64_t core = (int64_t) m.vae_cfg.decode_core_frames;
    const int64_t halo = (int64_t) m.vae_cfg.decode_halo_frames;
    const int64_t n_tiles = core > 0 ? (total_frames + core - 1) / core : 1;
    if (progress) {
        progress({ YUE2_STAGE_VAE, 0, n_tiles });
    }
    if (!yue2_vae_decode_tiled(m, latent_ct.data(), total_frames, core, halo, audio_planar_out, samples_out, nullptr,
                               err)) {
        return false;
    }
    if (progress) {
        progress({ YUE2_STAGE_VAE, n_tiles, n_tiles });
    }
    *stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ── Top-level orchestration ─────────────────────────────────────────────────
//
// `m.lm_resident` must already be true (caller loads via yue2_load_parts
// before calling this — the job layer does that). The VAE is loaded lazily
// by yue2_run_vae_stage itself (want_vae=true there), so a caller need only
// guarantee the LM before calling.
static bool yue2_pipeline_run(Yue2Model & m, const BPETokenizer & tok, Yue2Request & req,
                               const Yue2ProgressFn & progress, std::atomic<bool> * cancel,
                               Yue2PipelineResult * out, std::string * err) {
    yue2_request_resolve_defaults(&req, m.lm_cfg);

    // One /yue2/synth job is one imatrix "chunk" — counted here (not inside
    // the per-scheduler hook) so a job that runs plan+semantic+NAR still adds
    // exactly one to imatrix.chunk_count, matching mm3-server.h's own
    // one-call-one-chunk convention for /mm3/lm-plan.
    if (g_yue2_imatrix.armed) {
        g_yue2_imatrix.runs++;
        if (g_yue2_imatrix.sources.size() < 256) {
            std::string label = req.style.empty() ? req.id : req.style;
            label += std::string(" [cot=") + yue2_cot_name(req.cot) + "]";
            g_yue2_imatrix.sources.push_back(label);
        }
    }

    std::mt19937_64 rng(req.seed);

    std::vector<int32_t> abc_ids;
    if (!yue2_run_plan_stage(m, tok, req, rng, cancel, progress, &abc_ids, &out->score_abc,
                             &out->stage_end_reason[YUE2_STAGE_PLAN], &out->stage_ms[YUE2_STAGE_PLAN], err)) {
        return false;
    }
    if (req.plan_only) {
        // Score preview: the plan is the whole result. No audio, no
        // semantic ids; the stage terminator is the job's end reason so a
        // caller can tell a runaway plan (limit_hit) from a finished one.
        // The VAE may not be resident on this path, so sample_rate keeps its
        // default; nothing reads it without audio.
        out->end_reason = out->stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ? "limit_hit" : "completed";
        return true;
    }
    const bool have_abc = (req.cot != YUE2_COT_OFF);  // off never has an ABC span; melody/full always do (sampled or supplied)

    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> codec_ids;
    if (!yue2_run_semantic_stage(m, tok, req, abc_ids, have_abc, rng, cancel, progress, &prefix_ids, &codec_ids,
                                 &out->stage_end_reason[YUE2_STAGE_SEMANTIC], &out->stage_ms[YUE2_STAGE_SEMANTIC],
                                 err)) {
        return false;
    }

    std::vector<float> song_latents;
    if (!yue2_run_nar_stage(m, req, prefix_ids, codec_ids, cancel, progress, &song_latents, &out->total_frames,
                            &out->stage_ms[YUE2_STAGE_NAR], err)) {
        return false;
    }

    if (!yue2_run_vae_stage(m, req, song_latents, out->total_frames, progress, &out->audio_planar, &out->samples,
                            &out->stage_ms[YUE2_STAGE_VAE], err)) {
        return false;
    }

    // Final [-1,1] clamp is pipeline-level, not decoder-level (03-reference-
    // numerics.md §4.5 — the decoder's own last module is Identity).
    for (float & v : out->audio_planar) {
        if (v < -1.0f) {
            v = -1.0f;
        } else if (v > 1.0f) {
            v = 1.0f;
        }
    }

    out->semantic_ids = codec_ids;
    out->sample_rate  = (int) m.vae_cfg.sample_rate;

    const bool any_limit = out->stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ||
                            out->stage_end_reason[YUE2_STAGE_SEMANTIC] == "limit_hit";
    out->end_reason = any_limit ? "limit_hit" :
        (out->stage_end_reason[YUE2_STAGE_SEMANTIC] == "preview_limit" ? "preview_limit" : "completed");
    return true;
}
