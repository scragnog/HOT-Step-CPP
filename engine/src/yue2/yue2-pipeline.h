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
// - CFG's two branches are two sets of one KV cache decoded in one graph
//   (docs/plans/yue2/30-upstream-backports.md), as are the B songs of a
//   batch; each set carries its own absolute positions.

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
#include <iterator>
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

// One rendered track: song `song` (its own plan + semantic stream, LM seed
// `seed`) x noise variation `variation` (NAR noise seed `noise_seed`).
struct Yue2TrackResult {
    int      song      = 0;
    int      variation = 0;
    uint64_t seed       = 0;
    uint64_t noise_seed = 0;

    std::vector<float>   audio_planar;  // [2, samples] -- L then R, matches audio_encode_wav_s16's own contract
    int64_t               samples     = 0;

    std::string score_abc;              // decoded plan-stage ABC text; empty if the plan stage was skipped
    std::vector<int32_t> semantic_ids;  // raw codec ids [0, CODEC_SIZE), CODEC_OFFSET already subtracted

    // Per-stage: "skipped" | "eos" | "limit_hit" | "" (absent -- only plan/
    // semantic carry a terminator/limit concept at all; nar/vae stay "").
    std::string stage_end_reason[4];
    int64_t     total_frames = 0;  // NAR latent frame count (== semantic codec token count)

    // "completed" | "limit_hit" | "preview_limit" -- derived per track by
    // yue2_pipeline_run from the stage reasons (06-engine-port-plan.md §7).
    std::string end_reason;
};

struct Yue2PipelineResult {
    std::vector<Yue2TrackResult> tracks;  // song-major: (0,0) (0,1) .. (1,0) ..
    int                           sample_rate = 48000;
    double                        stage_ms[4] = { 0.0, 0.0, 0.0, 0.0 };
    // Aggregate over tracks: "limit_hit" if any track hit a cap, else
    // "preview_limit" if any was preview-capped, else "completed". The job
    // layer overlays "cancelled"/"failed" from the return value.
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

static uint64_t yue2_token_hash(const std::vector<int32_t> & ids) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int32_t id : ids) {
        hash ^= (uint32_t) id;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

// ── Per-song state carried across the AR stages ────────────────────────────
//
// Song i of a batch owns its RNG (seeded seed + i, and shared by its plan and
// semantic draws exactly as the single-song pipeline shared one RNG), its
// plan, its prompt and its codec stream. B == 1 is bit-for-bit the old path.
struct Yue2SongState {
    std::mt19937_64      rng;
    uint64_t             seed = 0;
    std::vector<int32_t> abc_ids;
    std::string          score_abc;
    std::vector<int32_t> prefix_ids;   // the positive semantic prompt (what NAR prefixes with)
    std::vector<int32_t> codec_ids;    // CODEC_OFFSET already subtracted
    std::string          stage_end_reason[4];
    int64_t              cond_set = 0;  // set of the semantic cache holding this song's positive stream
};

// Vocab windows the two AR stages sample from (doc 30 #1). The legal range
// and the end token of a stage are adjacent in the vocabulary, so the lm_head
// is sliced to just those rows; the sampler is told the same base.
static constexpr int64_t YUE2_PLAN_HEAD_LO = 0;
static constexpr int64_t YUE2_PLAN_HEAD_N  = std::max<int64_t>(YUE2_EOD, YUE2_ABC_END + 1);      // 151849
static constexpr int64_t YUE2_SEM_HEAD_LO  = std::min<int64_t>(YUE2_MUSIC_END, YUE2_CODEC_OFFSET);
static constexpr int64_t YUE2_SEM_HEAD_N   = std::max<int64_t>(YUE2_MUSIC_END + 1, YUE2_CODEC_OFFSET + YUE2_CODEC_SIZE) -
                                             YUE2_SEM_HEAD_LO;                                       // 32769

// ── Stage 1: plan (ABC) ──────────────────────────────────────────────────────
//
// Skipped entirely for cot=="off" (no ABC span exists at all) and whenever
// the request supplies `abc` externally (the model call is skipped; the
// supplied text is tokenised and used as-is). No CFG here -- the ABC stage
// is never guided in the reference (only 02_semantic's fixtures carry a
// cond/uncond split).
//
// B songs decode in lockstep over B cache sets: the shared prompt is
// prefilled once and copied into the other sets, and a song that has hit
// ABC_END keeps feeding ABC_END as a passive row so the graph shape holds
// until every song is done (upstream 2d21090f's design).
static bool yue2_run_plan_stage(Yue2Model & m, const BPETokenizer & tok, const Yue2Request & req,
                                 std::vector<Yue2SongState> & songs, std::atomic<bool> * cancel,
                                 const Yue2ProgressFn & progress, double * stage_ms, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_ar_step_profile_reset();
    const int B = (int) songs.size();
    for (auto & sg : songs) {
        sg.abc_ids.clear();
        sg.score_abc.clear();
    }

    if (req.cot == YUE2_COT_OFF) {
        for (auto & sg : songs) sg.stage_end_reason[YUE2_STAGE_PLAN] = "skipped";
        return true;
    }
    if (req.abc_provided) {
        std::vector<int32_t> ids;
        try {
            ids = yue2_bpe_encode(&tok, req.abc);
            yue2_validate_abc_ids(ids, "yue2_run_plan_stage");
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("plan stage: ") + e.what();
            }
            return false;
        }
        for (auto & sg : songs) {
            sg.abc_ids   = ids;
            sg.score_abc = req.abc;
            sg.stage_end_reason[YUE2_STAGE_PLAN] = "skipped";
        }
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
    if (!yue2_ar_kv_cache_alloc(m, capacity, &cache, err, B)) {
        return false;
    }
    cache.head_lo = YUE2_PLAN_HEAD_LO;
    cache.head_n  = YUE2_PLAN_HEAD_N;
    const int64_t base = cache.head_lo;

    Yue2ArForwardResult pre;
    if (!yue2_ar_prefill(m, cache, prefix, { (int64_t) prefix.size() - 1 }, {}, &pre, err, 0)) {
        yue2_ar_kv_cache_free(&cache);
        return false;
    }
    for (int b = 1; b < B; b++) {
        if (!yue2_ar_kv_cache_copy_set(m, cache, 0, b, cache.filled[0], err)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
    }
    const int64_t W = pre.V;  // logits width (the head window)
    std::vector<float> logits((size_t) B * (size_t) W);
    for (int b = 0; b < B; b++) {
        std::copy(pre.logits.begin(), pre.logits.end(), logits.begin() + (size_t) b * (size_t) W);
    }

    std::vector<std::vector<int32_t>> history((size_t) B);
    std::vector<bool>                 done((size_t) B, false);
    std::vector<int32_t>              next_ids((size_t) B, YUE2_ABC_END);
    int64_t                            step = 0;
    for (; step < sp.max_tokens; step++) {
        if (cancel && cancel->load()) {
            if (err) {
                *err = "cancelled";
            }
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        int n_active = 0;
        for (int b = 0; b < B; b++) {
            if (done[(size_t) b]) {
                next_ids[(size_t) b] = YUE2_ABC_END;  // passive row
                continue;
            }
            std::vector<float> scores(logits.begin() + (size_t) b * (size_t) W,
                                      logits.begin() + (size_t) (b + 1) * (size_t) W);
            yue2_distribution(scores, sp, YUE2_ABC_END, 0, YUE2_EOD, history[(size_t) b], step,
                              /*legacy_off=*/false, base);
            const int64_t tok_id =
                base + (sp.temperature == 0.0f ? yue2_sample_argmax(scores) : yue2_sample_draw(scores, songs[(size_t) b].rng));
            if (tok_id == YUE2_ABC_END) {
                done[(size_t) b] = true;
                songs[(size_t) b].stage_end_reason[YUE2_STAGE_PLAN] = "eos";
                next_ids[(size_t) b] = YUE2_ABC_END;
                continue;
            }
            history[(size_t) b].push_back((int32_t) tok_id);
            next_ids[(size_t) b] = (int32_t) tok_id;
            n_active++;
        }
        if (progress) {
            progress({ YUE2_STAGE_PLAN, step + 1, sp.max_tokens });
        }
        if (n_active == 0 || step + 1 >= sp.max_tokens) {
            break;
        }
        if (!yue2_ar_decode_batch(m, cache, next_ids.data(), &logits, err)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
    }
    yue2_ar_kv_cache_free(&cache);

    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        sg.abc_ids   = history[(size_t) b];
        sg.score_abc = yue2_bpe_decode(&tok, sg.abc_ids);
        if (!done[(size_t) b]) sg.stage_end_reason[YUE2_STAGE_PLAN] = "limit_hit";
        fprintf(stderr, "[YuE2-AR-Tokens] plan song=%d n=%zu hash=%016llx\n", b, sg.abc_ids.size(),
                (unsigned long long) yue2_token_hash(sg.abc_ids));
    }
    *stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    yue2_ar_step_profile_log("plan");
    return true;
}

// ── Stage 2: semantic (codec AR, possibly CFG'd) ────────────────────────────
//
// B songs, each with its own prompt (its own plan), on S = B or 2B sets of
// ONE cache: set cond_set(b) carries the positive stream, cond_set(b)+1 the
// negative one under guidance. The cache is handed to the caller on success
// (`cache_out`): the NAR stage reuses its rows instead of re-prefilling the
// prompt per chunk (doc 30 #2).
static bool yue2_run_semantic_stage(Yue2Model & m, const BPETokenizer & tok, const Yue2Request & req,
                                     bool have_abc, std::vector<Yue2SongState> & songs,
                                     std::atomic<bool> * cancel, const Yue2ProgressFn & progress,
                                     Yue2ArKvCache * cache_out, double * stage_ms, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_ar_step_profile_reset();
    const int  B       = (int) songs.size();
    const bool use_cfg = req.cfg_scale != 1.0f;
    const int  per     = use_cfg ? 2 : 1;
    const int  S       = B * per;

    std::vector<std::vector<int32_t>> neg_prefix((size_t) B);
    int64_t                            max_prefix = 0;
    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        const std::vector<int32_t> * abc_ptr = have_abc ? &sg.abc_ids : nullptr;
        try {
            sg.prefix_ids = yue2_token_prefixes(&tok, req.style, req.lyrics, req.cot, abc_ptr);
            if (use_cfg) neg_prefix[(size_t) b] = yue2_negative_prefix(&tok, req.cot, abc_ptr);
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("semantic stage: ") + e.what();
            }
            return false;
        }
        sg.cond_set = b * per;
        sg.codec_ids.clear();
        max_prefix = std::max<int64_t>(max_prefix, (int64_t) sg.prefix_ids.size());
        if (use_cfg) max_prefix = std::max<int64_t>(max_prefix, (int64_t) neg_prefix[(size_t) b].size());
    }

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

    Yue2ArKvCache & cache = *cache_out;
    if (!yue2_ar_kv_cache_alloc(m, max_prefix + sp.max_tokens + 4, &cache, err, S)) {
        return false;
    }
    cache.head_lo = YUE2_SEM_HEAD_LO;
    cache.head_n  = YUE2_SEM_HEAD_N;
    const int64_t base = cache.head_lo;
    const int64_t end_rel = YUE2_MUSIC_END - base;

    // Prefill every set. A prompt identical to an earlier set's is copied
    // rather than forwarded again (cot=off: every song shares one prompt).
    int64_t            W = 0;
    std::vector<float> logits;  // [S, W]
    for (int s = 0; s < S; s++) {
        const int  b   = s / per;
        const bool neg = use_cfg && (s % per == 1);
        const std::vector<int32_t> & ids = neg ? neg_prefix[(size_t) b] : songs[(size_t) b].prefix_ids;
        int twin = -1;
        for (int t = 0; t < s; t++) {
            const int  tb   = t / per;
            const bool tneg = use_cfg && (t % per == 1);
            const std::vector<int32_t> & tids = tneg ? neg_prefix[(size_t) tb] : songs[(size_t) tb].prefix_ids;
            if (tids == ids) {
                twin = t;
                break;
            }
        }
        Yue2ArForwardResult pre;
        if (twin >= 0) {
            if (!yue2_ar_kv_cache_copy_set(m, cache, twin, s, cache.filled[(size_t) twin], err)) {
                yue2_ar_kv_cache_free(&cache);
                return false;
            }
            const std::vector<float> twin_row(logits.begin() + (size_t) twin * (size_t) W,
                                              logits.begin() + (size_t) (twin + 1) * (size_t) W);
            logits.insert(logits.end(), twin_row.begin(), twin_row.end());
            continue;
        }
        if (!yue2_ar_prefill(m, cache, ids, { (int64_t) ids.size() - 1 }, {}, &pre, err, s)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        W = pre.V;
        logits.insert(logits.end(), pre.logits.begin(), pre.logits.end());
    }

    std::vector<std::vector<int32_t>> history((size_t) B);
    std::vector<bool>                 done((size_t) B, false);
    std::vector<bool>                 by_threshold((size_t) B, false);
    std::vector<int32_t>              next_ids((size_t) S, YUE2_MUSIC_END);
    static const char * end_trace_path = std::getenv("YUE2_END_TRACE");
    const bool trace = end_trace_path && *end_trace_path && B == 1;
    int64_t step = 0;
    for (; step < sp.max_tokens; step++) {
        if (cancel && cancel->load()) {
            if (err) {
                *err = "cancelled";
            }
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        int n_active = 0;
        for (int b = 0; b < B; b++) {
            Yue2SongState & sg = songs[(size_t) b];
            if (done[(size_t) b]) {
                for (int k = 0; k < per; k++) next_ids[(size_t) (sg.cond_set + k)] = YUE2_MUSIC_END;
                continue;
            }
            const float * cond_row = logits.data() + (size_t) sg.cond_set * (size_t) W;
            std::vector<float> blended;
            if (use_cfg) {
                const std::vector<float> cond(cond_row, cond_row + W);
                const std::vector<float> uncond(cond_row + W, cond_row + 2 * W);
                yue2_cfg_blend(cond, uncond, req.cfg_scale, &blended);
            } else {
                blended.assign(cond_row, cond_row + W);
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
                if (w > 0.0f && std::isfinite(blended[(size_t) end_rel])) {
                    blended[(size_t) end_rel] += w * req.end_bias;
                }
            }
            // YUE2_END_TRACE=<path>: uncensored per-step record of MUSIC_END's
            // probability and rank BEFORE the sampler chain (softmax over the
            // legal ids + END, raw logits, no penalty/temp/top-k/p) and AFTER
            // it, plus the sampled token — the evidence for "the model reached
            // its ending and END lost the draw". Read-only; single-song only.
            double  tr_p_pre = -1.0, tr_p_post = -1.0;
            int64_t tr_r_pre = -1, tr_r_post = -1;
            auto stat = [&](const std::vector<float> & s, bool legal_only, double * p, int64_t * rank) {
                float mx = -INFINITY;
                for (int64_t v = 0; v < (int64_t) s.size(); v++) {
                    const int64_t va = v + base;
                    const bool in = legal_only ? ((va >= legal_lo && va < legal_hi) || va == YUE2_MUSIC_END) : true;
                    if (in && std::isfinite(s[(size_t) v]) && s[(size_t) v] > mx) mx = s[(size_t) v];
                }
                double  z = 0.0;
                int64_t r = 0;
                const float se = s[(size_t) end_rel];
                for (int64_t v = 0; v < (int64_t) s.size(); v++) {
                    const int64_t va = v + base;
                    const bool in = legal_only ? ((va >= legal_lo && va < legal_hi) || va == YUE2_MUSIC_END) : true;
                    if (!in || !std::isfinite(s[(size_t) v])) continue;
                    z += std::exp((double) (s[(size_t) v] - mx));
                    if (s[(size_t) v] > se) r++;
                }
                *p    = std::isfinite(se) ? std::exp((double) (se - mx)) / z : 0.0;
                *rank = std::isfinite(se) ? r : -1;
            };
            if (trace) stat(blended, true, &tr_p_pre, &tr_r_pre);
            yue2_distribution(blended, sp, YUE2_MUSIC_END, legal_lo, legal_hi, history[(size_t) b], step, legacy_off,
                              base);
            if (trace) stat(blended, false, &tr_p_post, &tr_r_post);
            if (req.end_threshold > 0.0f && step >= (int64_t) sp.min_tokens &&
                std::isfinite(blended[(size_t) end_rel])) {
                // P(END) under the post-chain distribution: logsumexp over the
                // finite entries (masked ones are -inf and contribute nothing).
                float mx = -INFINITY;
                for (float v : blended) {
                    if (std::isfinite(v) && v > mx) mx = v;
                }
                double z = 0.0;
                for (float v : blended) {
                    if (std::isfinite(v)) z += std::exp((double) (v - mx));
                }
                const double p_end = std::exp((double) (blended[(size_t) end_rel] - mx)) / z;
                if (p_end >= (double) req.end_threshold) {
                    if (trace) {
                        if (FILE * tf = fopen(end_trace_path, "a")) {
                            fprintf(tf, "%lld\t%.2f\t%.6g\t%lld\t%.6g\t%lld\t%d\tforced_threshold\n",
                                    (long long) step, (double) step * 0.04, tr_p_pre, (long long) tr_r_pre,
                                    tr_p_post, (long long) tr_r_post, (int) YUE2_MUSIC_END);
                            fclose(tf);
                        }
                    }
                    done[(size_t) b]         = true;
                    by_threshold[(size_t) b] = true;
                    for (int k = 0; k < per; k++) next_ids[(size_t) (sg.cond_set + k)] = YUE2_MUSIC_END;
                    continue;
                }
            }
            const int64_t tok_id =
                base + (sp.temperature == 0.0f ? yue2_sample_argmax(blended) : yue2_sample_draw(blended, sg.rng));
            if (trace) {
                if (FILE * tf = fopen(end_trace_path, "a")) {
                    if (step == 0) {
                        fprintf(tf, "# semantic stage: prefix %lld ids, seed %llu, threshold %.3f, bias %.3f\n",
                                (long long) sg.prefix_ids.size(), (unsigned long long) sg.seed,
                                (double) req.end_threshold, (double) req.end_bias);
                        fprintf(tf, "# step\tsec\tp_end_pre\trank_pre\tp_end_post\trank_post\tsampled\tnote\n");
                    }
                    fprintf(tf, "%lld\t%.2f\t%.6g\t%lld\t%.6g\t%lld\t%lld\t%s\n", (long long) step,
                            (double) step * 0.04, tr_p_pre, (long long) tr_r_pre, tr_p_post, (long long) tr_r_post,
                            (long long) tok_id, tok_id == YUE2_MUSIC_END ? "sampled_end" : "");
                    fclose(tf);
                }
            }
            if (tok_id == YUE2_MUSIC_END) {
                done[(size_t) b] = true;
                for (int k = 0; k < per; k++) next_ids[(size_t) (sg.cond_set + k)] = YUE2_MUSIC_END;
                continue;
            }
            history[(size_t) b].push_back((int32_t) tok_id);
            sg.codec_ids.push_back((int32_t) (tok_id - YUE2_CODEC_OFFSET));
            for (int k = 0; k < per; k++) next_ids[(size_t) (sg.cond_set + k)] = (int32_t) tok_id;
            n_active++;
        }
        if (progress) {
            progress({ YUE2_STAGE_SEMANTIC, step + 1, sp.max_tokens });
        }
        if (n_active == 0) {
            break;
        }
        // The last content token is forwarded even on the final step so the
        // cache holds every codec row the NAR reads (upstream 2d21090f found
        // the budget-capped case reading a stale row without this).
        if (!yue2_ar_decode_batch(m, cache, next_ids.data(), &logits, err)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
    }

    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        sg.stage_end_reason[YUE2_STAGE_SEMANTIC] =
            done[(size_t) b] ? (by_threshold[(size_t) b] ? "eos_threshold" : "eos")
                             : (preview_capped ? "preview_limit" : "limit_hit");
        fprintf(stderr, "[YuE2-AR-Tokens] semantic song=%d n=%zu hash=%016llx\n", b, sg.codec_ids.size(),
                (unsigned long long) yue2_token_hash(sg.codec_ids));
    }
    *stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    yue2_ar_step_profile_log("semantic");
    return true;
}

// ── Stage 2.5: seal the acoustic chunks while the AR half holds the GPU ────
//
// Every NAR chunk of song b attends over prompt + codec[a:b) + MUSIC_END.
// Those rows are built here, once, into a dedicated cache with one set per
// chunk: the prompt rows (and, for chunk 0, the codec rows too) are copied
// from the semantic cache instead of forwarded again, and only the missing
// tail goes through the AR blocks (doc 30 #2, upstream 72e59ed1). Doing all
// of it before the NAR stage is what lets the AR half leave VRAM before the
// NAR half arrives (doc 30 #5).
struct Yue2ChunkPlan {
    int     song      = 0;
    int64_t a         = 0;
    int64_t b         = 0;
    int64_t set       = 0;  // set of the NAR cache holding this chunk's prefix
};

static bool yue2_seal_chunks(Yue2Model & m, std::vector<Yue2SongState> & songs, Yue2ArKvCache & sem_cache,
                             Yue2ArKvCache * nar_cache, std::vector<Yue2ChunkPlan> * plan, std::string * err) {
    plan->clear();
    int64_t capacity = 0;
    for (int b = 0; b < (int) songs.size(); b++) {
        const Yue2SongState & sg = songs[(size_t) b];
        const int64_t frames     = (int64_t) sg.codec_ids.size();
        const int64_t prefix_len = (int64_t) sg.prefix_ids.size();
        if (frames <= 0) {
            if (err) *err = "NAR stage: semantic stage produced zero codec frames";
            return false;
        }
        const auto ranges = yue2_chunk_ranges(frames, prefix_len, (int64_t) m.lm_cfg.context_length);
        if (ranges.empty()) {
            if (err) *err = "NAR stage: chunk_ranges computed size < 1 (prompt too long for the context window)";
            return false;
        }
        if (sem_cache.filled[(size_t) sg.cond_set] < prefix_len + frames) {
            if (err) *err = "NAR stage: the semantic cache holds fewer rows than prompt + frames (internal inconsistency)";
            return false;
        }
        for (const auto & r : ranges) {
            plan->push_back({ b, r.first, r.second, (int64_t) plan->size() });
            capacity = std::max(capacity, prefix_len + (r.second - r.first) + 1);
        }
    }
    if (!yue2_ar_kv_cache_alloc(m, capacity, nar_cache, err, (int64_t) plan->size())) {
        return false;
    }
    for (const Yue2ChunkPlan & cp : *plan) {
        const Yue2SongState & sg = songs[(size_t) cp.song];
        const int64_t prefix_len = (int64_t) sg.prefix_ids.size();
        const int64_t chunk_len  = cp.b - cp.a;
        std::vector<int32_t> tail;
        const int64_t shared = cp.a == 0 ? prefix_len + chunk_len : prefix_len;
        if (!yue2_ar_kv_cache_copy_rows(m, sem_cache, sg.cond_set, *nar_cache, cp.set, shared, err)) {
            yue2_ar_kv_cache_free(nar_cache);
            return false;
        }
        if (cp.a != 0) {
            tail.reserve((size_t) chunk_len + 1);
            for (int64_t i = cp.a; i < cp.b; i++) tail.push_back(sg.codec_ids[(size_t) i] + YUE2_CODEC_OFFSET);
        }
        tail.push_back(YUE2_MUSIC_END);
        Yue2ArForwardResult dummy;
        const auto t0 = std::chrono::steady_clock::now();
        if (!yue2_ar_prefill(m, *nar_cache, tail, {}, {}, &dummy, err, cp.set)) {
            yue2_ar_kv_cache_free(nar_cache);
            return false;
        }
        fprintf(stderr, "[YuE2-NAR-Seal] song=%d chunk=[%lld,%lld) shared_rows=%lld forwarded=%zu ms=%.1f\n", cp.song,
                (long long) cp.a, (long long) cp.b, (long long) shared, tail.size(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return true;
}

// ── Stage 3: NAR (32-step midpoint solve per sealed chunk) ──────────────────
//
// The M noise variations of a song solve in one graph (doc 30 #6) unless the
// ConvRot LM or a Lua solver is in use, which take one variation at a time.
// Latents come back [song][variation][frames*64].
static bool yue2_run_nar_stage(Yue2Model & m, const Yue2Request & req, std::vector<Yue2SongState> & songs,
                                Yue2ArKvCache & nar_cache, const std::vector<Yue2ChunkPlan> & plan, int n_var,
                                std::atomic<bool> * cancel, const Yue2ProgressFn & progress,
                                std::vector<std::vector<std::vector<float>>> * latents_out, std::string * err) {
    const int64_t LD      = (int64_t) m.lm_cfg.latent_dim;
    const bool    plugins = !(req.nar_solver.empty() && req.nar_scheduler.empty());
    const int     group   = (m.convrot || plugins) ? 1 : n_var;  // variations per graph
    const int     B       = (int) songs.size();

    // Whole-song noise per (song, variation), drawn once and sliced per chunk.
    std::vector<std::vector<std::vector<float>>> noise((size_t) B);
    latents_out->assign((size_t) B, {});
    for (int b = 0; b < B; b++) {
        const int64_t frames = (int64_t) songs[(size_t) b].codec_ids.size();
        noise[(size_t) b].assign((size_t) n_var, {});
        (*latents_out)[(size_t) b].assign((size_t) n_var, {});
        for (int j = 0; j < n_var; j++) {
            std::vector<float> & nz = noise[(size_t) b][(size_t) j];
            if (req.noise_source == YUE2_NOISE_FIXTURE) {
                if (!yue2_read_f32_file(req.noise_fixture_path, &nz)) {
                    if (err) *err = "NAR stage: noise_source=\"fixture\" -- cannot read " + req.noise_fixture_path;
                    return false;
                }
                if ((int64_t) nz.size() != frames * LD) {
                    if (err) {
                        *err = "NAR stage: noise_source=\"fixture\" geometry mismatch (expected " +
                               std::to_string(frames * LD) + " floats, got " + std::to_string(nz.size()) + ")";
                    }
                    return false;
                }
            } else {
                yue2_fill_noise(req.noise_seed + (uint64_t) b + (uint64_t) j, &nz, frames * LD);
            }
            (*latents_out)[(size_t) b][(size_t) j].reserve((size_t) (frames * LD));
        }
    }

    for (size_t ci = 0; ci < plan.size(); ci++) {
        if (cancel && cancel->load()) {
            if (err) *err = "cancelled";
            return false;
        }
        const Yue2ChunkPlan & cp = plan[ci];
        const int64_t chunk_len  = cp.b - cp.a;
        double nar_solve_ms = 0.0, velocity_ms = 0.0;
        int    velocity_calls = 0;
        for (int j0 = 0; j0 < n_var; j0 += group) {
            const int M = std::min(group, n_var - j0);
            Yue2NarChunk chunk;
            if (!yue2_nar_chunk_bind(nar_cache, cp.set, chunk_len, M, &chunk, err)) {
                return false;
            }
            std::vector<float> noise_slice;
            noise_slice.reserve((size_t) (M * chunk_len * LD));
            for (int j = j0; j < j0 + M; j++) {
                const auto & nz = noise[(size_t) cp.song][(size_t) j];
                noise_slice.insert(noise_slice.end(), nz.begin() + (size_t) (cp.a * LD), nz.begin() + (size_t) (cp.b * LD));
            }
            Yue2NarSolveResult solve;
            const auto nar_solve_start = std::chrono::steady_clock::now();
            const bool ok = !plugins
                ? yue2_nar_solve_midpoint(m, chunk, noise_slice, req.ode_steps, {}, false, &solve, err)
                : yue2_nar_solve_plugins(m, chunk, noise_slice, req.ode_steps,
                                         req.nar_solver, req.nar_scheduler, req.plugin_params, &solve, err);
            nar_solve_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - nar_solve_start).count();
            velocity_ms += solve.total_velocity_ms;
            velocity_calls += solve.velocity_calls;
            yue2_nar_chunk_free(&chunk);
            if (!ok) {
                return false;
            }
            for (int j = j0; j < j0 + M; j++) {
                auto & dst = (*latents_out)[(size_t) cp.song][(size_t) j];
                const size_t off = (size_t) ((j - j0) * chunk_len * LD);
                dst.insert(dst.end(), solve.final_latents.begin() + off,
                           solve.final_latents.begin() + off + (size_t) (chunk_len * LD));
            }
        }
        fprintf(stderr, "[YuE2-NAR-Chunk] song=%d index=%zu frames=%lld prefix_tokens=%lld solve_ms=%.1f velocity_ms=%.1f calls=%d variations=%d\n",
                cp.song, ci, (long long) chunk_len, (long long) nar_cache.filled[(size_t) cp.set], nar_solve_ms,
                velocity_ms, velocity_calls, n_var);
        if (progress) {
            progress({ YUE2_STAGE_NAR, (int64_t) ci + 1, (int64_t) plan.size() });
        }
    }
    return true;
}

// ── Stage 4: VAE decode (tiled — degrades to one "tile" == the whole song
//    when total_frames <= the tile core, so this is the general path) ──
static bool yue2_run_vae_stage(Yue2Model & m, const Yue2Request & req, const std::vector<float> & song_latents,
                                int64_t total_frames, const Yue2ProgressFn & progress,
                                std::vector<float> * audio_planar_out, int64_t * samples_out, std::string * err) {
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

    // Tile core 512, not the GGUF's 1024 (doc 30 #4, upstream 689d71e8): the
    // decode takes the same time at either size and the F32 activation peak
    // drops by ~1.5 GB. Halo unchanged. Tiled-vs-untiled was already
    // "mathematically equivalent, bitwise unpromised" (yue2-vae-graph.h).
    const int64_t core = std::min<int64_t>((int64_t) m.vae_cfg.decode_core_frames, 512);
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
    return true;
}

// ── Top-level orchestration ─────────────────────────────────────────────────
//
// Residency (doc 30 #5): `evict_strict` (the job layer passes !g_keep_loaded)
// walks the three parts through VRAM one at a time — the AR half for the
// plan, semantic and chunk-sealing stages, then the NAR half alone, then the
// VAE alone — with the sealed NAR cache surviving the swaps. Otherwise
// whatever is resident stays and missing parts are loaded on demand, which is
// the old shape. Either way the caller need not preload anything.
static bool yue2_pipeline_run(Yue2Model & m, const BPETokenizer & tok, Yue2Request & req,
                               const Yue2ProgressFn & progress, std::atomic<bool> * cancel,
                               Yue2PipelineResult * out, std::string * err, bool evict_strict = false) {
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

    const int B = std::max(1, req.lm_batch_size);
    const int M = std::max(1, req.synth_batch_size);
    std::vector<Yue2SongState> songs((size_t) B);
    for (int b = 0; b < B; b++) {
        songs[(size_t) b].seed = req.seed + (uint64_t) b;
        songs[(size_t) b].rng.seed(songs[(size_t) b].seed);
    }
    auto t_stage = std::chrono::steady_clock::now();
    auto lap = [&](int stage) {
        const auto now = std::chrono::steady_clock::now();
        out->stage_ms[stage] = std::chrono::duration<double, std::milli>(now - t_stage).count();
        t_stage = now;
    };

    // ── AR half ──
    if (!yue2_load_parts(&m, /*want_ar=*/true, /*want_nar=*/!evict_strict, /*want_vae=*/false, req.vae_variant,
                         false, err)) {
        return false;
    }
    if (!yue2_run_plan_stage(m, tok, req, songs, cancel, progress, &out->stage_ms[YUE2_STAGE_PLAN], err)) {
        return false;
    }
    if (req.plan_only) {
        // Score preview: the plan is the whole result. No audio, no
        // semantic ids; the stage terminator is the job's end reason so a
        // caller can tell a runaway plan (limit_hit) from a finished one.
        // The VAE may not be resident on this path, so sample_rate keeps its
        // default; nothing reads it without audio.
        bool any_limit = false;
        for (int b = 0; b < B; b++) {
            Yue2TrackResult tr;
            tr.song       = b;
            tr.seed       = songs[(size_t) b].seed;
            tr.noise_seed = req.noise_seed + (uint64_t) b;
            tr.score_abc  = songs[(size_t) b].score_abc;
            for (int s = 0; s < 4; s++) tr.stage_end_reason[s] = songs[(size_t) b].stage_end_reason[s];
            tr.end_reason = tr.stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ? "limit_hit" : "completed";
            any_limit |= tr.end_reason == "limit_hit";
            out->tracks.push_back(std::move(tr));
        }
        out->end_reason = any_limit ? "limit_hit" : "completed";
        return true;
    }
    const bool have_abc = (req.cot != YUE2_COT_OFF);  // off never has an ABC span; melody/full always do (sampled or supplied)

    Yue2ArKvCache sem_cache;
    if (!yue2_run_semantic_stage(m, tok, req, have_abc, songs, cancel, progress, &sem_cache,
                                 &out->stage_ms[YUE2_STAGE_SEMANTIC], err)) {
        return false;
    }
    t_stage = std::chrono::steady_clock::now();

    Yue2ArKvCache              nar_cache;
    std::vector<Yue2ChunkPlan> plan;
    if (!yue2_seal_chunks(m, songs, sem_cache, &nar_cache, &plan, err)) {
        yue2_ar_kv_cache_free(&sem_cache);
        return false;
    }
    yue2_ar_kv_cache_free(&sem_cache);

    // ── NAR half ──
    if (evict_strict) {
        yue2_evict_half(&m, /*ar=*/true, /*nar=*/false);
    }
    if (!yue2_load_parts(&m, /*want_ar=*/false, /*want_nar=*/true, /*want_vae=*/false, req.vae_variant, false, err)) {
        yue2_ar_kv_cache_free(&nar_cache);
        return false;
    }
    std::vector<std::vector<std::vector<float>>> latents;  // [song][variation][frames*64]
    if (!yue2_run_nar_stage(m, req, songs, nar_cache, plan, M, cancel, progress, &latents, err)) {
        yue2_ar_kv_cache_free(&nar_cache);
        return false;
    }
    yue2_ar_kv_cache_free(&nar_cache);
    lap(YUE2_STAGE_NAR);

    // ── VAE ──
    if (evict_strict) {
        yue2_evict_half(&m, /*ar=*/false, /*nar=*/true);
    }
    bool any_limit = false, any_preview = false;
    for (int b = 0; b < B; b++) {
        for (int j = 0; j < M; j++) {
            if (cancel && cancel->load()) {
                if (err) *err = "cancelled";
                return false;
            }
            Yue2TrackResult tr;
            tr.song         = b;
            tr.variation    = j;
            tr.seed         = songs[(size_t) b].seed;
            tr.noise_seed   = req.noise_seed + (uint64_t) b + (uint64_t) j;
            tr.score_abc    = songs[(size_t) b].score_abc;
            tr.semantic_ids = songs[(size_t) b].codec_ids;
            tr.total_frames = (int64_t) songs[(size_t) b].codec_ids.size();
            for (int s = 0; s < 4; s++) tr.stage_end_reason[s] = songs[(size_t) b].stage_end_reason[s];
            if (!yue2_run_vae_stage(m, req, latents[(size_t) b][(size_t) j], tr.total_frames, progress,
                                    &tr.audio_planar, &tr.samples, err)) {
                return false;
            }
            // Final [-1,1] clamp is pipeline-level, not decoder-level (03-reference-
            // numerics.md §4.5 — the decoder's own last module is Identity).
            for (float & v : tr.audio_planar) {
                if (v < -1.0f) {
                    v = -1.0f;
                } else if (v > 1.0f) {
                    v = 1.0f;
                }
            }
            const bool limit = tr.stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ||
                               tr.stage_end_reason[YUE2_STAGE_SEMANTIC] == "limit_hit";
            const bool preview = tr.stage_end_reason[YUE2_STAGE_SEMANTIC] == "preview_limit";
            tr.end_reason = limit ? "limit_hit" : preview ? "preview_limit" : "completed";
            any_limit |= limit;
            any_preview |= preview;
            out->tracks.push_back(std::move(tr));
            latents[(size_t) b][(size_t) j] = {};  // done with this one
        }
    }
    lap(YUE2_STAGE_VAE);
    out->sample_rate = (int) m.vae_cfg.sample_rate;
    out->end_reason  = any_limit ? "limit_hit" : any_preview ? "preview_limit" : "completed";
    return true;
}
