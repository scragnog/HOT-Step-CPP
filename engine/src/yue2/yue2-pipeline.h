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
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
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
    uint32_t  songs_done = 0;  // bit b: song b of the batch has finished this stage (semantic only)
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
                                                                   int64_t context, int64_t cap = 0) {
    std::vector<std::pair<int64_t, int64_t>> out;
    int64_t                                   size = (context - prefix_tokens - 3) / 2;
    if (size < 1 || frames <= 0) {
        return out;
    }
    // nar_chunk_frames: never above the reference size, balanced chunks.
    if (cap > 0 && cap < size && frames > cap) {
        const int64_t n = (frames + cap - 1) / cap;
        size            = (frames + n - 1) / n;
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
    int                  src_song = 0;    // index in the request's batch; tracks report it, so songs dropped before the render leave no gap to guess at
    std::mt19937_64      rng;
    uint64_t             seed       = 0;
    uint64_t             noise_seed = 0;  // NAR noise for variation j is noise_seed + j
    std::string          style;           // this song's prompt (the request's, or its "songs" entry's)
    std::string          lyrics;
    std::string          abc_text;        // supplied score, when this song brought one
    bool                 abc_given = false;
    std::vector<int32_t> abc_ids;
    std::string          score_abc;
    std::vector<int32_t> prefix_ids;   // the positive semantic prompt (what NAR prefixes with)
    std::vector<int32_t> codec_ids;    // CODEC_OFFSET already subtracted
    std::string          stage_end_reason[4];
    int64_t              cond_set  = 0;   // set of the semantic cache holding this song's positive stream
    int64_t              neg_set   = -1;  // its negative (guidance) stream, -1 without guidance
    int64_t              pair_base = 0;   // first set of this song's pair (cond/neg in either order)
    int64_t              sem_end_step = 0;  // semantic step this stream ended at (a retry's winner is the earliest)
};

// Vocab windows the two AR stages sample from (doc 30 #1). The legal range
// and the end token of a stage are adjacent in the vocabulary, so the lm_head
// is sliced to just those rows; the sampler is told the same base.
static constexpr int64_t YUE2_PLAN_HEAD_LO = 0;
static constexpr int64_t YUE2_PLAN_HEAD_N  = std::max<int64_t>(YUE2_EOD, YUE2_ABC_END + 1);      // 151849
static constexpr int64_t YUE2_SEM_HEAD_LO  = std::min<int64_t>(YUE2_MUSIC_END, YUE2_CODEC_OFFSET);
static constexpr int64_t YUE2_SEM_HEAD_N   = std::max<int64_t>(YUE2_MUSIC_END + 1, YUE2_CODEC_OFFSET + YUE2_CODEC_SIZE) -
                                             YUE2_SEM_HEAD_LO;                                       // 32769

// The checkpoint's per-stage sampler defaults with the request's overrides
// (the LM tab) laid over them. Fields the request left at -1 keep the GGUF
// value, so an untouched tab reproduces the locked defaults exactly.
static Yue2SamplingParams yue2_stage_params(const Yue2LmConfig::Stage & st, const Yue2StageOverride & o) {
    Yue2SamplingParams sp;
    sp.temperature        = o.temperature >= 0.0f ? o.temperature : st.temperature;
    sp.top_p               = o.top_p >= 0.0f ? o.top_p : st.top_p;
    sp.top_k               = o.top_k >= 0 ? o.top_k : (int) st.top_k;
    sp.repetition_penalty = o.repetition_penalty >= 0.0f ? o.repetition_penalty : st.repetition_penalty;
    sp.penalty_window     = o.penalty_window >= 0 ? o.penalty_window : (int) st.penalty_window;
    sp.min_tokens          = o.min_tokens >= 0 ? o.min_tokens : (int) st.min_tokens;
    sp.max_tokens          = o.max_tokens > 0 ? o.max_tokens : (int) st.max_tokens;
    return sp;
}

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
static bool yue2_song_dropped(const Yue2Request & req, int b) {
    return req.songs_dropped && b < 32 && ((req.songs_dropped->load(std::memory_order_relaxed) >> b) & 1u);
}

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
    // Songs that brought a score skip the model; the rest plan together as one
    // batch, set p of the cache holding song plan_idx[p]. A coalesced batch
    // mixes both freely (an approved preview beside a fresh prompt).
    std::vector<int> plan_idx;
    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        if (!sg.abc_given) {
            plan_idx.push_back(b);
            continue;
        }
        try {
            sg.abc_ids = yue2_bpe_encode(&tok, sg.abc_text);
            yue2_validate_abc_ids(sg.abc_ids, "yue2_run_plan_stage");
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("plan stage: ") + e.what();
            }
            return false;
        }
        sg.score_abc = sg.abc_text;
        sg.stage_end_reason[YUE2_STAGE_PLAN] = "skipped";
    }
    const int P = (int) plan_idx.size();
    if (P == 0) {
        return true;
    }

    std::vector<std::vector<int32_t>> prefixes((size_t) P);
    int64_t                            max_prefix = 0;
    for (int p = 0; p < P; p++) {
        const Yue2SongState & sg = songs[(size_t) plan_idx[(size_t) p]];
        prefixes[(size_t) p] = yue2_token_prefixes(&tok, sg.style, sg.lyrics, req.cot, nullptr);
        max_prefix = std::max<int64_t>(max_prefix, (int64_t) prefixes[(size_t) p].size());
    }

    Yue2SamplingParams sp = yue2_stage_params(m.lm_cfg.abc, req.plan);

    Yue2ArKvCache cache;
    const int64_t capacity = max_prefix + sp.max_tokens + 4;
    if (!yue2_ar_kv_cache_alloc(m, capacity, &cache, err, P)) {
        return false;
    }
    cache.head_lo = YUE2_PLAN_HEAD_LO;
    cache.head_n  = YUE2_PLAN_HEAD_N;
    const int64_t base = cache.head_lo;

    // Prefill every set; a prompt identical to an earlier set's is copied
    // rather than forwarded again (lm_batch_size > 1: every song shares one).
    int64_t            W = 0;
    std::vector<float> logits;  // [P, W]
    for (int p = 0; p < P; p++) {
        int twin = -1;
        for (int t = 0; t < p; t++) {
            if (prefixes[(size_t) t] == prefixes[(size_t) p]) {
                twin = t;
                break;
            }
        }
        if (twin >= 0) {
            if (!yue2_ar_kv_cache_copy_set(m, cache, twin, p, cache.filled[(size_t) twin], err)) {
                yue2_ar_kv_cache_free(&cache);
                return false;
            }
            const std::vector<float> twin_row(logits.begin() + (size_t) twin * (size_t) W,
                                              logits.begin() + (size_t) (twin + 1) * (size_t) W);
            logits.insert(logits.end(), twin_row.begin(), twin_row.end());
            continue;
        }
        Yue2ArForwardResult pre;
        const std::vector<int32_t> & prefix = prefixes[(size_t) p];
        if (!yue2_ar_prefill(m, cache, prefix, { (int64_t) prefix.size() - 1 }, {}, &pre, err, p)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        W = pre.V;  // logits width (the head window)
        logits.insert(logits.end(), pre.logits.begin(), pre.logits.end());
    }

    std::vector<std::vector<int32_t>> history((size_t) P);
    std::vector<bool>                 done((size_t) P, false);
    std::vector<int32_t>              next_ids((size_t) P, YUE2_ABC_END);
    std::vector<int32_t>              surv_ids;
    std::vector<float>                surv_vals;
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
        for (int p = 0; p < P; p++) {
            Yue2SongState & sg = songs[(size_t) plan_idx[(size_t) p]];
            if (!done[(size_t) p] && yue2_song_dropped(req, plan_idx[(size_t) p])) {
                done[(size_t) p] = true;
                sg.stage_end_reason[YUE2_STAGE_PLAN] = "dropped";
            }
            if (done[(size_t) p]) {
                next_ids[(size_t) p] = YUE2_ABC_END;  // passive row
                continue;
            }
            // The row is this step's scratch: the next decode overwrites it.
            const int64_t tok_id = base + yue2_sample_row(logits.data() + (size_t) p * (size_t) W, W, sp, YUE2_ABC_END, 0,
                                                          YUE2_EOD, history[(size_t) p], step, /*legacy_off=*/false,
                                                          base, sg.rng, &surv_ids, &surv_vals);
            if (tok_id == YUE2_ABC_END) {
                done[(size_t) p] = true;
                sg.stage_end_reason[YUE2_STAGE_PLAN] = "eos";
                next_ids[(size_t) p] = YUE2_ABC_END;
                continue;
            }
            history[(size_t) p].push_back((int32_t) tok_id);
            next_ids[(size_t) p] = (int32_t) tok_id;
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

    for (int p = 0; p < P; p++) {
        const int       b  = plan_idx[(size_t) p];
        Yue2SongState & sg = songs[(size_t) b];
        sg.abc_ids   = history[(size_t) p];
        sg.score_abc = yue2_bpe_decode(&tok, sg.abc_ids);
        if (!done[(size_t) p]) sg.stage_end_reason[YUE2_STAGE_PLAN] = "limit_hit";
        fprintf(stderr, "[YuE2-AR-Tokens] plan song=%d n=%zu hash=%016llx\n", b, sg.abc_ids.size(),
                (unsigned long long) yue2_token_hash(sg.abc_ids));
    }
    *stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    yue2_ar_step_profile_log("plan");
    return true;
}

// ── Stage 2: semantic (codec AR, possibly CFG'd) ────────────────────────────
//
// B streams, each with its own prompt (its own plan), on S = B or 2B sets of
// ONE cache: slot i holds sets [i*per, i*per+per), the positive stream in
// cond_set and, under guidance, the negative one in neg_set. The cache is
// handed to the caller on success (`cache_out`): the NAR stage reuses its rows
// instead of re-prefilling the prompt per chunk (doc 30 #2).
//
// Three things keep the stage from paying for work nobody will use
// (2026-09-28):
//  - A stream that has ended leaves the decode. Its slot is swapped behind
//    the live ones through one spare slot, and each step decodes the live
//    sets only; the ended stream's KV stays where the NAR will read it. Before
//    this a finished song was fed MUSIC_END every step and kept attending
//    over its whole KV until the last song of the batch ended.
//  - A stream stops at plan_cutoff x its plan's length + 20 s
//    (yue2_plan_cap_frames), not at the stage's 9000-frame cap, and counts as
//    a runaway there.
//  - A retry pass (`first_wins`) holds several draws of each runaway song,
//    grouped by src_song. A group is over the moment one draw ends cleanly;
//    its other draws are abandoned and leave the decode.

// The plan's own length in seconds: vocal bars x beats per bar / tempo, the
// estimate classifyYue2Score (scoreHealth.ts) makes. 0 without a tempo or bars.
static int64_t yue2_plan_seconds(const std::string & abc) {
    double tempo = 0.0;
    int    beats = 4;
    bool   in_vocal = false;
    int64_t bars = 0;
    auto lower = [](std::string s) {
        for (char & ch : s) ch = (char) std::tolower((unsigned char) ch);
        return s;
    };
    size_t pos = 0;
    while (pos <= abc.size()) {
        size_t nl = abc.find('\n', pos);
        if (nl == std::string::npos) nl = abc.size();
        std::string line = abc.substr(pos, nl - pos);
        pos = nl + 1;
        const size_t a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos) continue;
        line = line.substr(a, line.find_last_not_of(" \t\r") - a + 1);
        if (line[0] == '%') continue;
        if (line.size() >= 2 && line[1] == ':' && std::isalpha((unsigned char) line[0])) {
            if (line[0] == 'Q') {
                const size_t eq = line.find('=');
                if (eq != std::string::npos) {
                    const char * p = line.c_str() + eq + 1;
                    while (*p == ' ' || *p == '\t') p++;
                    if (std::isdigit((unsigned char) *p)) tempo = std::strtod(p, nullptr);
                }
            } else if (line[0] == 'M') {
                const int n = std::atoi(line.c_str() + 2);
                beats = n > 0 ? n : 4;
            } else if (line[0] == 'V') {
                const std::string rest  = line.substr(2);
                const size_t      s0    = rest.find_first_not_of(" \t");
                const std::string first = s0 == std::string::npos ? "" : rest.substr(s0, rest.find_first_of(" \t", s0) - s0);
                in_vocal = lower(first).find("vocal") != std::string::npos ||
                           lower(line).find("name=\"vocal") != std::string::npos;
            }
            continue;
        }
        if (!in_vocal) continue;
        size_t s = 0;
        while (s <= line.size()) {
            size_t e = line.find('|', s);
            if (e == std::string::npos) e = line.size();
            std::string seg;  // the segment with chord quotes stripped
            bool        quoted = false;
            for (size_t i = s; i < e; i++) {
                if (line[i] == '"') quoted = !quoted;
                else if (!quoted) seg += line[i];
            }
            const size_t t0 = seg.find_first_not_of(" \t");
            if (line.find_first_not_of(" \t", s) < e) {  // a non-blank segment (chords count)
                int64_t n = 1;
                if (t0 != std::string::npos) {
                    const std::string t = seg.substr(t0, seg.find_last_not_of(" \t") - t0 + 1);
                    if (t[0] == 'Z' && t.find_first_not_of("0123456789", 1) == std::string::npos && t.size() > 1) {
                        n = std::max<int64_t>(1, std::atoll(t.c_str() + 1));
                    }
                }
                bars += n;
            }
            s = e + 1;
        }
    }
    if (bars <= 0 || !(tempo > 0.0)) return 0;
    return (int64_t) std::llround((double) bars * beats * 60.0 / tempo);
}

// Where a stream counts as a runaway: plan_cutoff x the plan's length + 20 s,
// never past the stage cap and never before min_tokens + 10 s.
static int64_t yue2_plan_cap_frames(const Yue2Request & req, const Yue2SamplingParams & sp, const std::string & abc) {
    const int64_t cap = sp.max_tokens;
    if (!(req.plan_cutoff > 0.0f)) return cap;
    const int64_t sec = yue2_plan_seconds(abc);
    if (sec <= 0) return cap;
    const int64_t frames = (int64_t) std::ceil(((double) sec * req.plan_cutoff + 20.0) * 25.0);
    return std::min(cap, std::max(frames, (int64_t) sp.min_tokens + 250));
}

// Swap the KV of slots i and j (per sets each) through the spare slot. Three
// separate copies, each one graph the scheduler finishes before the next, so
// no read can overtake the write it depends on.
static bool yue2_swap_slots(const Yue2Model & m, Yue2ArKvCache & c, int i, int j, int spare, int per, std::string * err) {
    for (int k = 0; k < per; k++) {
        const int64_t a = (int64_t) i * per + k, b = (int64_t) j * per + k, t = (int64_t) spare * per + k;
        const int64_t na = c.filled[(size_t) a], nb = c.filled[(size_t) b];
        if (na > 0 && !yue2_ar_kv_cache_copy_set(m, c, a, t, na, err)) return false;
        if (nb > 0 && !yue2_ar_kv_cache_copy_set(m, c, b, a, nb, err)) return false;
        if (na > 0 && !yue2_ar_kv_cache_copy_set(m, c, t, b, na, err)) return false;
        c.filled[(size_t) a] = nb;
        c.filled[(size_t) b] = na;
    }
    return true;
}

struct Yue2SemanticPass {
    bool     first_wins      = false;  // groups (src_song) end at their first clean draw
    uint32_t extra_done_bits = 0;      // songs outside this pass that have ended (progress only)
};

static bool yue2_stream_clean(const std::string & reason) {
    return reason == "eos" || reason == "eos_threshold";
}

static bool yue2_run_semantic_stage(Yue2Model & m, const BPETokenizer & tok, const Yue2Request & req,
                                     bool have_abc, std::vector<Yue2SongState> & songs,
                                     std::atomic<bool> * cancel, const Yue2ProgressFn & progress,
                                     Yue2ArKvCache * cache_out, double * stage_ms, std::string * err,
                                     const Yue2SemanticPass & pass = {}) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_ar_step_profile_reset();
    const int  B       = (int) songs.size();
    // YUE2_CFG_FORCE_SETS=1 keeps the negative set alive at guidance 1.0 and
    // YUE2_CFG_SWAP=1 puts the positive stream in the pair's SECOND set:
    // together they check that a stream decoded next to another set reproduces
    // the single-set tokens (the blend at guidance 1.0 returns cond untouched).
    static const bool force_sets = std::getenv("YUE2_CFG_FORCE_SETS") != nullptr;
    static const bool swap_sets  = std::getenv("YUE2_CFG_SWAP") != nullptr;
    const bool use_cfg = req.cfg_scale != 1.0f || force_sets;
    const int  per     = use_cfg ? 2 : 1;
    const int  S       = B * per;
    auto seat = [&](Yue2SongState & sg, int slot) {
        sg.pair_base = slot * per;
        sg.cond_set  = sg.pair_base + ((use_cfg && swap_sets) ? 1 : 0);
        sg.neg_set   = use_cfg ? sg.pair_base + ((swap_sets) ? 0 : 1) : -1;
    };

    std::vector<std::vector<int32_t>> neg_prefix((size_t) B);
    int64_t                            max_prefix = 0;
    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        const std::vector<int32_t> * abc_ptr = have_abc ? &sg.abc_ids : nullptr;
        try {
            sg.prefix_ids = yue2_token_prefixes(&tok, sg.style, sg.lyrics, req.cot, abc_ptr);
            if (use_cfg) neg_prefix[(size_t) b] = yue2_negative_prefix(&tok, req.cot, abc_ptr);
        } catch (const std::exception & e) {
            if (err) {
                *err = std::string("semantic stage: ") + e.what();
            }
            return false;
        }
        seat(sg, b);
        sg.codec_ids.clear();
        max_prefix = std::max<int64_t>(max_prefix, (int64_t) sg.prefix_ids.size());
        if (use_cfg) max_prefix = std::max<int64_t>(max_prefix, (int64_t) neg_prefix[(size_t) b].size());
    }

    Yue2SamplingParams sp = yue2_stage_params(m.lm_cfg.semantic, req.semantic);
    const bool preview_capped = req.preview_max_frames > 0 && req.preview_max_frames < sp.max_tokens;
    if (preview_capped) sp.max_tokens = req.preview_max_frames;
    const bool legacy_off  = (req.cot == YUE2_COT_OFF);
    const int64_t legal_lo = YUE2_CODEC_OFFSET;
    const int64_t legal_hi = YUE2_CODEC_OFFSET + YUE2_CODEC_SIZE;

    Yue2ArKvCache & cache = *cache_out;
    const int64_t supplied_n = (int64_t) req.codec_ids.size();
    if (supplied_n > 0 && (B != 1 || use_cfg)) {
        if (err) *err = "codec_ids needs a single song and cfg_scale 1 (the stream is not sampled)";
        return false;
    }
    // The supplied stream is prefilled in one graph at positions after the
    // prefix; past the trained context that is RoPE positions the model never
    // saw, and a huge array is a VRAM exhaustion, not a render.
    if (supplied_n > 0 && max_prefix + supplied_n + 4 > (int64_t) m.lm_cfg.context_length) {
        if (err) *err = "codec_ids too long: " + std::to_string(supplied_n) + " codes plus the prefix exceed the model context of " +
                        std::to_string(m.lm_cfg.context_length);
        return false;
    }
    std::vector<int64_t> cap((size_t) B);
    int64_t              max_cap = 0;
    for (int b = 0; b < B; b++) {
        cap[(size_t) b] = yue2_plan_cap_frames(req, sp, songs[(size_t) b].score_abc);
        max_cap         = std::max(max_cap, cap[(size_t) b]);
    }
    // Rows per set: the prompt plus the longest stream any set may reach.
    const int64_t rows = max_prefix + std::max<int64_t>(max_cap, supplied_n) + 4;
    // The spare slot for compaction. Without room for it the stage still
    // runs; ended streams just stay in the decode.
    bool compact = false;
    if (B > 1 && supplied_n == 0) {
        std::string spare_err;
        compact = yue2_ar_kv_cache_alloc(m, rows, &cache, &spare_err, S + per);
        if (!compact) fprintf(stderr, "[YuE2] no room for the spare set (%s): ended songs stay in the decode\n", spare_err.c_str());
    }
    if (!compact && !yue2_ar_kv_cache_alloc(m, rows, &cache, err, S)) {
        return false;
    }
    cache.head_lo = YUE2_SEM_HEAD_LO;
    cache.head_n  = YUE2_SEM_HEAD_N;
    const int64_t base = cache.head_lo;
    const int64_t end_rel = YUE2_MUSIC_END - base;

    // Prefill every set. A prompt identical to an earlier set's is copied
    // rather than forwarded again (cot=off: every song shares one prompt; a
    // retry pass: every draw of a song shares its prompt).
    int64_t            W = 0;
    std::vector<float> logits;  // [S, W]
    for (int s = 0; s < S; s++) {
        const int  b   = s / per;
        const bool neg = use_cfg && (s == songs[(size_t) b].neg_set);
        const std::vector<int32_t> & ids = neg ? neg_prefix[(size_t) b] : songs[(size_t) b].prefix_ids;
        int twin = -1;
        for (int t = 0; t < s; t++) {
            const int  tb   = t / per;
            const bool tneg = use_cfg && (t == songs[(size_t) tb].neg_set);
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

    if (supplied_n > 0) {
        // Round trip: forward the supplied stream teacher-forced, exactly the
        // rows sampling would have left in the cache, and skip the draw.
        Yue2SongState & sg = songs[0];
        std::vector<int32_t> ids;
        ids.reserve((size_t) supplied_n);
        for (int32_t c : req.codec_ids) ids.push_back(c + YUE2_CODEC_OFFSET);
        Yue2ArForwardResult dummy;
        if (!yue2_ar_prefill(m, cache, ids, {}, {}, &dummy, err, sg.cond_set)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
        sg.codec_ids = req.codec_ids;
        sg.stage_end_reason[YUE2_STAGE_SEMANTIC] = "supplied";
        fprintf(stderr, "[YuE2-AR-Tokens] semantic song=0 n=%zu hash=%016llx (supplied)\n", sg.codec_ids.size(),
                (unsigned long long) yue2_token_hash(sg.codec_ids));
        *stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }

    std::vector<std::vector<int32_t>> history((size_t) B);
    std::vector<bool>    done((size_t) B, false), by_threshold((size_t) B, false), dropped((size_t) B, false),
                         capped((size_t) B, false), abandoned((size_t) B, false);
    std::vector<int64_t> end_step((size_t) B, -1);
    std::vector<int32_t> next_tok((size_t) B, YUE2_MUSIC_END);
    std::vector<int32_t> next_ids((size_t) S, YUE2_MUSIC_END);
    std::vector<int>     slot_song((size_t) B);  // slot i holds song slot_song[i]; [0,n_live) decode
    std::iota(slot_song.begin(), slot_song.end(), 0);
    int                  n_live = B;
    std::vector<float>   scratch((size_t) W);
    std::vector<int32_t> surv_ids;
    std::vector<float>   surv_vals;
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
        for (int b = 0; b < B; b++) {
            Yue2SongState & sg = songs[(size_t) b];
            if (done[(size_t) b]) continue;
            if (yue2_song_dropped(req, sg.src_song)) {
                done[(size_t) b] = dropped[(size_t) b] = true;
                end_step[(size_t) b] = step;
                continue;
            }
            if ((int64_t) sg.codec_ids.size() >= cap[(size_t) b]) {
                done[(size_t) b] = capped[(size_t) b] = true;
                end_step[(size_t) b] = step;
                continue;
            }
            float * row = logits.data() + (size_t) sg.cond_set * (size_t) W;
            if (use_cfg) {
                yue2_cfg_blend_into(row, logits.data() + (size_t) sg.neg_set * (size_t) W, (size_t) W, req.cfg_scale,
                                    scratch.data());
                row = scratch.data();
            }
            // Ending controls (yue2-request.h). The bias lands on the raw logit
            // BEFORE the reference's mask/penalty/temperature/top-k/top-p chain,
            // so the chain itself is untouched; the threshold reads the chain's
            // OUTPUT, the very distribution the draw samples from.
            if (req.end_bias != 0.0f && step >= (int64_t) sp.min_tokens) {
                const float t_sec = (float) step * 0.04f;  // 25 Hz codec frames
                float       w     = 0.0f;
                if (t_sec >= req.end_bias_from_sec) {
                    w = req.end_bias_ramp_sec > 0.0f
                            ? std::min(1.0f, (t_sec - req.end_bias_from_sec) / req.end_bias_ramp_sec)
                            : 1.0f;
                }
                if (w > 0.0f && std::isfinite(row[end_rel])) {
                    row[end_rel] += w * req.end_bias;
                }
            }
            // YUE2_END_TRACE=<path>: per-step record of MUSIC_END's probability
            // and rank BEFORE the sampler chain (softmax over the legal ids +
            // END, raw logits) and AFTER it, plus the sampled token — the
            // evidence for "the model reached its ending and END lost the
            // draw". Read-only; single-song only.
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
            std::vector<float> traced;
            if (trace) {
                traced.assign(row, row + W);
                stat(traced, true, &tr_p_pre, &tr_r_pre);
                yue2_distribution(traced, sp, YUE2_MUSIC_END, legal_lo, legal_hi, history[(size_t) b], step, legacy_off,
                                  base);
                stat(traced, false, &tr_p_post, &tr_r_post);
            }
            const float   threshold = step >= (int64_t) sp.min_tokens ? req.end_threshold : 0.0f;
            const int64_t rel = yue2_sample_row(row, W, sp, YUE2_MUSIC_END, legal_lo, legal_hi, history[(size_t) b], step,
                                                legacy_off, base, sg.rng, &surv_ids, &surv_vals, threshold);
            const int64_t tok_id = rel < 0 ? YUE2_MUSIC_END : base + rel;
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
                            (long long) tok_id,
                            rel < 0 ? "forced_threshold" : (tok_id == YUE2_MUSIC_END ? "sampled_end" : ""));
                    fclose(tf);
                }
            }
            if (rel < 0 || tok_id == YUE2_MUSIC_END) {
                done[(size_t) b]         = true;
                by_threshold[(size_t) b] = rel < 0;
                end_step[(size_t) b]     = step;
                continue;
            }
            history[(size_t) b].push_back((int32_t) tok_id);
            sg.codec_ids.push_back((int32_t) (tok_id - YUE2_CODEC_OFFSET));
            next_tok[(size_t) b] = (int32_t) tok_id;
        }

        // A retry pass: a song whose draw has ended cleanly needs no other.
        uint32_t won = 0;
        if (pass.first_wins) {
            for (int b = 0; b < B; b++) {
                if (done[(size_t) b] && !dropped[(size_t) b] && !capped[(size_t) b] && !abandoned[(size_t) b] &&
                    songs[(size_t) b].src_song < 32) {
                    won |= 1u << songs[(size_t) b].src_song;
                }
            }
            for (int b = 0; b < B; b++) {
                const int g = songs[(size_t) b].src_song;
                if (!done[(size_t) b] && g < 32 && ((won >> g) & 1u)) {
                    done[(size_t) b] = abandoned[(size_t) b] = true;
                    end_step[(size_t) b] = step;
                }
            }
        }
        if (progress) {
            // Bit g: song g is over (every stream of it ended, or it has won).
            uint32_t open = 0;
            for (int b = 0; b < B; b++) {
                if (!done[(size_t) b] && songs[(size_t) b].src_song < 32) open |= 1u << songs[(size_t) b].src_song;
            }
            uint32_t songs_done = pass.extra_done_bits | won;
            for (int b = 0; b < B; b++) {
                const int g = songs[(size_t) b].src_song;
                if (g < 32 && !((open >> g) & 1u)) songs_done |= 1u << g;
            }
            progress({ YUE2_STAGE_SEMANTIC, step + 1, sp.max_tokens, songs_done });
        }

        if (compact) {
            for (int i = 0; i < n_live;) {
                const int b = slot_song[(size_t) i];
                if (!done[(size_t) b]) {
                    i++;
                    continue;
                }
                const int j = n_live - 1;
                if (i != j) {
                    if (!yue2_swap_slots(m, cache, i, j, B, per, err)) {
                        yue2_ar_kv_cache_free(&cache);
                        return false;
                    }
                    std::swap(slot_song[(size_t) i], slot_song[(size_t) j]);
                    seat(songs[(size_t) slot_song[(size_t) i]], i);
                    seat(songs[(size_t) slot_song[(size_t) j]], j);
                }
                n_live--;
            }
        }
        int n_active = 0;
        for (int b = 0; b < B; b++) n_active += done[(size_t) b] ? 0 : 1;
        if (n_active == 0) {
            break;
        }
        const int n_slots = compact ? n_live : B;
        for (int i = 0; i < n_slots; i++) {
            const int b = slot_song[(size_t) i];
            for (int k = 0; k < per; k++) {
                next_ids[(size_t) (i * per + k)] = done[(size_t) b] ? YUE2_MUSIC_END : next_tok[(size_t) b];
            }
        }
        // The last content token is forwarded even on the final step so the
        // cache holds every codec row the NAR reads (upstream 2d21090f found
        // the budget-capped case reading a stale row without this).
        if (!yue2_ar_decode_batch(m, cache, next_ids.data(), &logits, err, (int64_t) n_slots * per)) {
            yue2_ar_kv_cache_free(&cache);
            return false;
        }
    }

    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        sg.stage_end_reason[YUE2_STAGE_SEMANTIC] =
            dropped[(size_t) b]     ? "dropped"
            : abandoned[(size_t) b] ? "abandoned"
            : capped[(size_t) b]    ? "limit_hit"
            : done[(size_t) b]      ? (by_threshold[(size_t) b] ? "eos_threshold" : "eos")
                                    : (preview_capped ? "preview_limit" : "limit_hit");
        sg.sem_end_step = end_step[(size_t) b] >= 0 ? end_step[(size_t) b] : step;
        fprintf(stderr, "[YuE2-AR-Tokens] semantic song=%d n=%zu hash=%016llx%s\n", sg.src_song, sg.codec_ids.size(),
                (unsigned long long) yue2_token_hash(sg.codec_ids),
                capped[(size_t) b] ? " (past its plan's length)" : "");
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
                             Yue2ArKvCache * nar_cache, std::vector<Yue2ChunkPlan> * plan, std::string * err,
                             int64_t chunk_cap = 0) {
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
        const auto ranges = yue2_chunk_ranges(frames, prefix_len, (int64_t) m.lm_cfg.context_length, chunk_cap);
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
                yue2_fill_noise(songs[(size_t) b].noise_seed + (uint64_t) j, &nz, frames * LD);
            }
            (*latents_out)[(size_t) b][(size_t) j].reserve((size_t) (frames * LD));
        }
    }

    // Progress is counted in ODE STEPS, not chunks. A chunk was the unit until
    // #158: one chunk is minutes of solving that reported nothing, so the last
    // thing anything downstream saw was the SEMANTIC stage's final step, and
    // the Node-side watchdog — which cannot tell a stage that ended from a
    // stage that died — cancelled an M4 Pro render 120 s into an 11-minute
    // chunk that was working perfectly.
    //
    // The budget is exact for the native solver (every chunk x variation group
    // runs req.ode_steps). A Lua scheduler may return a different count, so
    // the total is a ceiling there and the step number is clamped to it rather
    // than allowed to overshoot: a bar that stops at 99% is a smaller lie than
    // one that reads 110%.
    int64_t nar_steps_done = 0;
    int64_t nar_steps_total = 0;
    for (const auto & cp : plan) {
        (void) cp;
        nar_steps_total += (int64_t) ((n_var + group - 1) / group) * (int64_t) req.ode_steps;
    }
    if (nar_steps_total <= 0) nar_steps_total = (int64_t) plan.size();
    // Say the NAR stage has STARTED before solving anything. Without this the
    // reported phase stays `semantic` until the first chunk finishes, which is
    // exactly the window #158 died in.
    if (progress) {
        progress({ YUE2_STAGE_NAR, 0, nar_steps_total });
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
            // One tick per ODE step, and one cancel check with it: a cancel
            // used to be read only between chunks, so a request to stop sat
            // unanswered for as long as the chunk had left to run (#158).
            const int64_t steps_before = nar_steps_done;
            const Yue2NarStepFn on_step = [&](int step, int) -> bool {
                if (cancel && cancel->load()) return false;
                nar_steps_done = steps_before + (int64_t) step;
                if (progress) {
                    progress({ YUE2_STAGE_NAR, std::min(nar_steps_done, nar_steps_total), nar_steps_total });
                }
                return true;
            };
            const bool ok = !plugins
                ? yue2_nar_solve_midpoint(m, chunk, noise_slice, req.ode_steps, {}, false, &solve, err,
                                          req.nar_cache_ratio, on_step)
                : yue2_nar_solve_plugins(m, chunk, noise_slice, req.ode_steps,
                                         req.nar_solver, req.nar_scheduler, req.plugin_params, &solve, err,
                                         on_step);
            nar_solve_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - nar_solve_start).count();
            velocity_ms += solve.total_velocity_ms;
            velocity_calls += solve.velocity_calls;
            yue2_nar_chunk_free(&chunk);
            if (!ok) {
                return false;
            }
            // Bank this group's budget rather than whatever the solver
            // actually stepped, so a Lua scheduler's different step count
            // cannot drift the running total against the ceiling above.
            nar_steps_done = steps_before + (int64_t) req.ode_steps;
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
// ── The two halves of a run ─────────────────────────────────────────────────
//
// The pipeline splits where the AR half stops touching the GPU: after the
// chunks are sealed. Everything the NAR half needs from the AR half travels
// in this handoff, so the two can run on different threads and backends
// (yue2-job.h's NAR lane): song N renders while song N+1 composes.
struct Yue2ArHandoff {
    std::vector<Yue2SongState> songs;
    Yue2ArKvCache              nar_cache;
    std::vector<Yue2ChunkPlan> plan;
    int                        M    = 1;
    bool                       done = false;  // plan_only / semantic_only: `out` is already complete
};

static void yue2_handoff_free(Yue2ArHandoff * ho) {
    yue2_ar_kv_cache_free(&ho->nar_cache);
    ho->plan.clear();
    ho->songs.clear();
}

// Plan, compose and seal. On success with ho->done == false the caller owns
// ho->nar_cache and must run yue2_pipeline_run_nar (or yue2_handoff_free).
static bool yue2_pipeline_run_ar(Yue2Model & m, const BPETokenizer & tok, Yue2Request & req,
                                  const Yue2ProgressFn & progress, std::atomic<bool> * cancel,
                                  Yue2PipelineResult * out, Yue2ArHandoff * ho, std::string * err,
                                  bool evict_strict = false) {
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

    const int B = req.songs.empty() ? std::max(1, req.lm_batch_size) : (int) req.songs.size();
    ho->M       = std::max(1, req.synth_batch_size);
    ho->done    = false;
    std::vector<Yue2SongState> & songs = ho->songs;
    songs.assign((size_t) B, Yue2SongState{});
    for (int b = 0; b < B; b++) {
        Yue2SongState & sg = songs[(size_t) b];
        sg.src_song = b;
        // Song b of a plain batch is the request's prompt at seed + b; a
        // "songs" entry brings its own prompt, and its own seeds when it says so.
        sg.seed       = req.seed + (uint64_t) b;
        sg.noise_seed = req.noise_seed + (uint64_t) b;
        sg.style      = req.style;
        sg.lyrics     = req.lyrics;
        sg.abc_text   = req.abc;
        sg.abc_given  = req.abc_provided;
        if (!req.songs.empty()) {
            const Yue2SongSpec & spec = req.songs[(size_t) b];
            sg.style     = spec.style;
            sg.lyrics    = spec.lyrics;
            sg.abc_text  = spec.abc;
            sg.abc_given = spec.abc_provided;
            if (spec.seed_present) sg.seed = spec.seed;
            sg.noise_seed = spec.noise_seed_present ? spec.noise_seed : (spec.seed_present ? spec.seed : sg.noise_seed);
        }
        sg.rng.seed(sg.seed);
    }

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
            tr.noise_seed = songs[(size_t) b].noise_seed;
            tr.score_abc  = songs[(size_t) b].score_abc;
            for (int s = 0; s < 4; s++) tr.stage_end_reason[s] = songs[(size_t) b].stage_end_reason[s];
            tr.end_reason = tr.stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ? "limit_hit" : "completed";
            any_limit |= tr.end_reason == "limit_hit";
            out->tracks.push_back(std::move(tr));
        }
        out->end_reason = any_limit ? "limit_hit" : "completed";
        ho->done        = true;
        return true;
    }
    const bool have_abc = (req.cot != YUE2_COT_OFF);  // off never has an ABC span; melody/full always do (sampled or supplied)

    Yue2ArKvCache sem_cache;
    {
        double stage_ms = 0.0;
        if (!yue2_run_semantic_stage(m, tok, req, have_abc, songs, cancel, progress, &sem_cache, &stage_ms, err)) {
            return false;
        }
        out->stage_ms[YUE2_STAGE_SEMANTIC] += stage_ms;
    }
    // Recompose on a runaway: a song whose composer ran past its plan (or to
    // the stage cap) is drawn again with new seeds, up to req.semantic_retries
    // draws per song. The plan is kept; only the codec stream is redrawn.
    //
    // Draws run side by side in their own cache (2026-09-28), up to
    // YUE2_RETRY_DRAWS at once shared between the runaway songs, and a song is
    // settled by the first of its draws to end cleanly. A decode step reads
    // the weights once whatever the batch width, so four draws cost about what
    // 1.25 serial tries did, and nothing waits for a hopeless draw to reach
    // its cap. The winner's rows go into the song's own set of the first
    // pass's cache, so songs that ended there are never replayed.
    static constexpr int YUE2_RETRY_DRAWS = 4;
    std::vector<int> tries((size_t) B, 0);
    int              draws_cap = YUE2_RETRY_DRAWS;
    for (int round = 1;; round++) {
        std::vector<int> runaway;
        uint32_t         done_bits = 0;
        int              budget    = 0;
        for (int b = 0; b < B; b++) {
            const Yue2SongState & sg = songs[(size_t) b];
            // A song dropped from the batch is never worth another try.
            if (sg.stage_end_reason[YUE2_STAGE_SEMANTIC] == "limit_hit" && !yue2_song_dropped(req, sg.src_song) &&
                tries[(size_t) b] < req.semantic_retries) {
                runaway.push_back(b);
                budget += req.semantic_retries - tries[(size_t) b];
            } else if (sg.src_song < 32) {
                done_bits |= 1u << sg.src_song;
            }
        }
        if (runaway.empty() || (cancel && cancel->load())) break;

        std::vector<int> want(runaway.size(), 0);
        for (int left = std::min(draws_cap, budget); left > 0;) {
            for (size_t r = 0; r < runaway.size() && left > 0; r++) {
                if (tries[(size_t) runaway[r]] + want[r] < req.semantic_retries) {
                    want[r]++;
                    left--;
                }
            }
        }
        std::vector<Yue2SongState> draws;
        std::vector<int>           draw_song;
        for (size_t r = 0; r < runaway.size(); r++) {
            const Yue2SongState & sg = songs[(size_t) runaway[r]];
            for (int d = 0; d < want[r]; d++) {
                Yue2SongState ds;
                ds.src_song   = sg.src_song;
                ds.style      = sg.style;
                ds.lyrics     = sg.lyrics;
                ds.abc_ids    = sg.abc_ids;
                ds.score_abc  = sg.score_abc;
                ds.noise_seed = sg.noise_seed;
                // Try n of a song draws with its seed + n x 1000003, the step
                // the server reads its recompose count back from.
                ds.seed = sg.seed + 1000003ull * (uint64_t) (tries[(size_t) runaway[r]] + d + 1);
                ds.rng.seed(ds.seed);
                draws.push_back(std::move(ds));
                draw_song.push_back(runaway[r]);
            }
        }
        fprintf(stderr, "[YuE2] composer ran past its plan on %zu song(s): drawing %zu recompositions side by side (round %d)\n",
                runaway.size(), draws.size(), round);

        Yue2ArKvCache    retry_cache;
        Yue2SemanticPass pass;
        pass.first_wins      = true;
        pass.extra_done_bits = done_bits;
        double      stage_ms = 0.0;
        std::string rerr;
        if (!yue2_run_semantic_stage(m, tok, req, have_abc, draws, cancel, progress, &retry_cache, &stage_ms, &rerr,
                                     pass)) {
            yue2_ar_kv_cache_free(&retry_cache);
            if (rerr.find("VRAM") != std::string::npos && draws_cap > 1) {
                draws_cap = std::max(1, draws_cap / 2);
                fprintf(stderr, "[YuE2] no VRAM for %zu draws at once; trying %d\n", draws.size(), draws_cap);
                continue;
            }
            yue2_ar_kv_cache_free(&sem_cache);
            if (err) *err = rerr;
            return false;
        }
        out->stage_ms[YUE2_STAGE_SEMANTIC] += stage_ms;

        for (size_t r = 0; r < runaway.size(); r++) {
            const int       b  = runaway[r];
            Yue2SongState & sg = songs[(size_t) b];
            int             w  = -1;
            for (size_t d = 0; d < draws.size(); d++) {
                if (draw_song[d] != b || !yue2_stream_clean(draws[d].stage_end_reason[YUE2_STAGE_SEMANTIC])) continue;
                if (w < 0 || draws[d].sem_end_step < draws[(size_t) w].sem_end_step) w = (int) d;
            }
            tries[(size_t) b] += want[r];
            if (w < 0) {
                fprintf(stderr, "[YuE2] song %d: none of %d draws ended cleanly (%d of %d tries used)\n", sg.src_song,
                        want[r], tries[(size_t) b], req.semantic_retries);
                continue;
            }
            const Yue2SongState & win = draws[(size_t) w];
            const int64_t n = (int64_t) (win.prefix_ids.size() + win.codec_ids.size());
            if (!yue2_ar_kv_cache_copy_rows(m, retry_cache, win.cond_set, sem_cache, sg.cond_set, n, err)) {
                yue2_ar_kv_cache_free(&retry_cache);
                yue2_ar_kv_cache_free(&sem_cache);
                return false;
            }
            const int try_n = (int) ((win.seed - sg.seed) / 1000003ull);
            sg.codec_ids                             = win.codec_ids;
            sg.seed                                  = win.seed;
            sg.stage_end_reason[YUE2_STAGE_SEMANTIC] = win.stage_end_reason[YUE2_STAGE_SEMANTIC];
            fprintf(stderr, "[YuE2] song %d: recomposed by try %d of %d (seed %llu, %zu frames)\n", sg.src_song, try_n,
                    req.semantic_retries, (unsigned long long) win.seed, win.codec_ids.size());
        }
        yue2_ar_kv_cache_free(&retry_cache);
    }
    if (req.semantic_only) {
        // Planner probe: the codec stream is the result. No NAR, no VAE.
        yue2_ar_kv_cache_free(&sem_cache);
        bool any_limit = false;
        for (int b = 0; b < B; b++) {
            Yue2TrackResult tr;
            tr.song         = b;
            tr.seed         = songs[(size_t) b].seed;
            tr.noise_seed   = songs[(size_t) b].noise_seed;
            tr.score_abc    = songs[(size_t) b].score_abc;
            tr.semantic_ids = songs[(size_t) b].codec_ids;
            tr.total_frames = (int64_t) songs[(size_t) b].codec_ids.size();
            for (int s = 0; s < 4; s++) tr.stage_end_reason[s] = songs[(size_t) b].stage_end_reason[s];
            tr.end_reason = (tr.stage_end_reason[YUE2_STAGE_PLAN] == "limit_hit" ||
                             tr.stage_end_reason[YUE2_STAGE_SEMANTIC] == "limit_hit") ? "limit_hit" : "completed";
            any_limit |= tr.end_reason == "limit_hit";
            out->tracks.push_back(std::move(tr));
        }
        out->end_reason = any_limit ? "limit_hit" : "completed";
        ho->done        = true;
        return true;
    }

    // Songs dropped from the batch are not rendered: the kept songs render
    // in order and each track reports its batch index (src_song). Each kept song still
    // reads its own rows of the semantic cache through cond_set.
    for (int b = B - 1; b >= 0; b--) {
        if (!yue2_song_dropped(req, b)) continue;
        fprintf(stderr, "[YuE2] song %d was dropped from the batch; not rendering it\n", b);
        songs.erase(songs.begin() + b);
    }
    if (songs.empty()) {
        yue2_ar_kv_cache_free(&sem_cache);
        if (err) *err = "cancelled";
        return false;
    }
    if (!yue2_seal_chunks(m, songs, sem_cache, &ho->nar_cache, &ho->plan, err, (int64_t) req.nar_chunk_frames)) {
        yue2_ar_kv_cache_free(&sem_cache);
        return false;
    }
    yue2_ar_kv_cache_free(&sem_cache);
    return true;
}

// Render and decode a sealed handoff. Frees the handoff's cache either way.
// `load_parts` false means the caller guarantees the NAR half and the VAE are
// resident and nothing may touch the model's residency here (the overlap
// lane, where the AR half is composing the next song at the same time).
static bool yue2_pipeline_run_nar(Yue2Model & m, const Yue2Request & req, Yue2ArHandoff & ho,
                                   const Yue2ProgressFn & progress, std::atomic<bool> * cancel,
                                   Yue2PipelineResult * out, std::string * err, bool evict_strict = false,
                                   bool load_parts = true) {
    std::vector<Yue2SongState> & songs = ho.songs;
    const int B = (int) songs.size();
    const int M = ho.M;
    auto t_stage = std::chrono::steady_clock::now();
    auto lap = [&](int stage) {
        const auto now = std::chrono::steady_clock::now();
        out->stage_ms[stage] = std::chrono::duration<double, std::milli>(now - t_stage).count();
        t_stage = now;
    };

    // ── NAR half ──
    if (load_parts) {
        if (evict_strict) {
            yue2_evict_half(&m, /*ar=*/true, /*nar=*/false);
        }
        if (!yue2_load_parts(&m, /*want_ar=*/false, /*want_nar=*/true, /*want_vae=*/false, req.vae_variant, false, err)) {
            yue2_handoff_free(&ho);
            return false;
        }
    }
    std::vector<std::vector<std::vector<float>>> latents;  // [song][variation][frames*64]
    if (!yue2_run_nar_stage(m, req, songs, ho.nar_cache, ho.plan, M, cancel, progress, &latents, err)) {
        yue2_handoff_free(&ho);
        return false;
    }
    yue2_ar_kv_cache_free(&ho.nar_cache);
    lap(YUE2_STAGE_NAR);

    // ── VAE ──
    if (load_parts && evict_strict) {
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
            tr.song         = songs[(size_t) b].src_song;
            tr.variation    = j;
            tr.seed         = songs[(size_t) b].seed;
            tr.noise_seed   = songs[(size_t) b].noise_seed + (uint64_t) j;
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

static bool yue2_pipeline_run(Yue2Model & m, const BPETokenizer & tok, Yue2Request & req,
                               const Yue2ProgressFn & progress, std::atomic<bool> * cancel,
                               Yue2PipelineResult * out, std::string * err, bool evict_strict = false) {
    Yue2ArHandoff ho;
    if (!yue2_pipeline_run_ar(m, tok, req, progress, cancel, out, &ho, err, evict_strict)) {
        return false;
    }
    if (ho.done) {
        return true;
    }
    return yue2_pipeline_run_nar(m, req, ho, progress, cancel, out, err, evict_strict);
}
