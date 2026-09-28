#pragma once
// yue2/yue2-sample.h — YuE2 sampler: the exact mask -> penalty -> temperature
// -> top-k -> top-p pipeline, per docs/plans/yue2/03-reference-numerics.md
// §2.3/§2.4 and docs/plans/yue2/06-engine-port-plan.md §4. Milestone M3.
//
// SCOPE: one function (`yue2_distribution`) implementing `sampling.
// distribution()` bit-for-rule (not bit-for-bit — see the tolerance policy
// in docs/plans/yue2/02-fixture-schema.md §9), plus the CFG blend that feeds
// it (`yue2_cfg_blend`) and the parts of the AR decode loop this milestone
// needs to actually draw a token (`yue2_sample_argmax`/`yue2_sample_draw`).
// Consumed by engine/tools/yue2-probe.cpp's --sampler-parity (replays stored
// fixture rows) and --generate (free-running, no parity expected).
//
// ── Why this is NOT a copy of minimax/mm3-sample.h, despite the shared genre ──
//
// MM3's sampler (`mm3_sample_top_k`) is a DIFFERENT reference: no legal-token
// mask, no repetition-penalty-before-temperature order, and critically, its
// very first line does `nan_to_num(..., neginf=-1e9)` — every `-inf` becomes
// a FINITE floor before anything else runs. YuE2's own reference is the
// opposite on this exact point (03-reference-numerics.md §2.3's gap note,
// restated because it is the single easiest thing to get wrong by pattern-
// matching MM3): "a port that... substitutes a finite floor... for -inf
// anywhere in this chain will not reproduce the reference: a finite floor
// gets multiplied by `alpha` and can shift rank instead of staying pinned at
// the bottom." So `yue2_distribution` below uses true IEEE -inf throughout,
// never `mm3_sample_top_k`'s `-1e9`/`+1e9` clamp — do not "fix" this by
// importing that clamp, it would silently break every near--inf comparison
// window_penalty and top-k depend on.
//
// What IS reused, because the underlying algorithm genuinely is the same
// shape (docs/plans/yue2/06-engine-port-plan.md §4 calls this out
// explicitly): the `nth_element`-based O(n) top-k threshold selection
// (`mm3_sample_top_k`'s technique, not its nan-clamped setup), the sorted-
// prefix top-p shape (`mm3_sample_knobbed`'s top-p branch), the frequency-
// count-then-`pow`-then-divide-or-multiply repetition penalty (`mm3_apply_
// rep_penalty`'s `MM3_REP_FREQUENCY` branch, not its default `MM3_REP_DRY`
// mode, which YuE2's reference has no equivalent of at all), and the
// inverse-CDF-walk multinomial draw tail (`mm3_sample_top_k`'s own RNG code,
// verbatim in spirit) for the one part of this pipeline that is provably
// never fixture-diffable regardless of implementation (03-reference-
// numerics.md §5 — PyTorch's own `multinomial` kernel is not portable).
//
// ── legacy_off: a real dtype fork, not an approximation either way skips ──
//
// `cot="off"` runs the ENTIRE mask->penalty->temperature->top-k->top-p chain
// in the model's native BF16 (03-reference-numerics.md §2.4: "scores =
// logits.clone() if legacy_off else logits.float().clone()") — every
// intermediate write is itself a BF16 value, not merely the final output.
// `yue2_maybe_bf16()` round-trips through `ggml_fp32_to_bf16`/`_to_fp32`
// after every op this file's `legacy_off` branch touches (window_penalty's
// scaled value, the temperature division) so a legacy_off row accumulates
// the SAME coarse rounding the reference's own BF16 tensor ops do at each
// step, not just at the end. melody/full (`legacy_off=false`) never rounds
// mid-pipeline at all — the input row is already an exact widened-BF16 value
// (§1 of the fixture schema: BF16 stored as a lossless widened FP32 view),
// and everything downstream of that first read runs in genuine FP32,
// matching `distribution()`'s own `.float()` upcast.
//
// ── CFG blend: always BF16 arithmetic, independent of legacy_off ─────────────
//
// `sampling.py:109`'s `unconditional + cfg_scale * (conditional -
// unconditional)` runs on two BF16 tensors and produces a BF16 result in
// EVERY mode (03-reference-numerics.md §2.5: "the blend arithmetic is
// always BF16 in every mode") — `yue2_cfg_blend` below always rounds each of
// its three ops (subtract, scale, add) through BF16, unconditionally on
// `legacy_off`. The `distribution()` call that follows it is what forks on
// `legacy_off`, not the blend itself.
//
// ── What this file does NOT implement (out of scope for M3) ────────────────
//
// No `-inf`-vs-tied-group multiset accounting (docs/plans/yue2/
// 02-fixture-schema.md §9's four-part tied-group gate) — the comparison
// harness in yue2-probe.cpp's --sampler-parity reports raw per-id agreement
// plus a near-cutoff band, which is looser than §9's exact tied-group rule
// but catches every real bug this milestone is gating on (see that file's
// own comment for the honest tradeoff). No DRY-style penalty (YuE2's
// reference doesn't have one). No batched/multi-branch sampling — the CFG
// blend and the single-branch distribution() call are separate steps by
// design (docs/plans/yue2/06-engine-port-plan.md §3), matching yue2-lm-
// graph.h's own two-independent-forward-calls CFG design.

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

// Round-trips through BF16 (round-to-nearest-even, ggml's own conversion) —
// a no-op when legacy_off is false. See file header for why this matters and
// exactly which ops it must wrap.
static inline float yue2_maybe_bf16(float x, bool legacy_off) {
    if (!legacy_off || !std::isfinite(x)) {
        return x;
    }
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
}

static inline float yue2_round_bf16(float x) {
    return std::isfinite(x) ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(x)) : x;
}

// Per-stage sampling knobs, plain-old-data mirror of Yue2LmConfig::Stage
// (yue2-model.h) — kept as its own type so this file has no hard dependency
// on the model header, only on whatever the caller reads out of it.
struct Yue2SamplingParams {
    float temperature        = 1.0f;
    float top_p              = 1.0f;
    int   top_k              = 0;  // <=0 means "no top-k filtering" (k == vocab_size)
    float repetition_penalty = 1.0f;
    int   penalty_window     = 0;
    int   min_tokens         = 0;
    int   max_tokens         = 0;
};

enum Yue2Phase { YUE2_PHASE_ABC = 0, YUE2_PHASE_SEMANTIC = 1 };

// CFG blend (03-reference-numerics.md §2.5): `guidance == 1.0` returns `cond`
// UNCHANGED (the reference never even computes the negative branch in that
// case, §2.5's own "skipped negative-prefill fast path" note) — every other
// guidance value blends through three BF16-rounded ops, unconditionally.
static inline void yue2_cfg_blend(const std::vector<float> & cond, const std::vector<float> & uncond, float guidance,
                                  std::vector<float> * out) {
    const size_t V = cond.size();
    out->resize(V);
    if (guidance == 1.0f || uncond.size() != V) {
        *out = cond;
        return;
    }
    for (size_t i = 0; i < V; i++) {
        const float diff   = yue2_round_bf16(cond[i] - uncond[i]);
        const float scaled = yue2_round_bf16(diff * guidance);
        (*out)[i]           = yue2_round_bf16(uncond[i] + scaled);
    }
}

// yue2_cfg_blend into a caller buffer (the decode loop's scratch row).
static inline void yue2_cfg_blend_into(const float * cond, const float * uncond, size_t V, float guidance, float * out) {
    if (guidance == 1.0f) {
        std::copy(cond, cond + V, out);
        return;
    }
    for (size_t i = 0; i < V; i++) {
        const float diff   = yue2_round_bf16(cond[i] - uncond[i]);
        const float scaled = yue2_round_bf16(diff * guidance);
        out[i]             = yue2_round_bf16(uncond[i] + scaled);
    }
}

// The exact mask -> penalty -> [greedy short-circuit] -> temperature ->
// top-k -> top-p pipeline (03-reference-numerics.md §2.3/§2.4), REORDERING
// NONE OF IT — see the numerics doc's own "this is explicitly called out as
// a correctness contract, not a style choice" line.
//
//   scores      in/out, size == vocab_size. On entry: the CFG-blended (or
//               plain, if guidance==1) logits row, already at whatever
//               precision its source file/tensor holds (an exact widened
//               BF16 view per the fixture schema's own storage convention —
//               callers must NOT independently round this before calling).
//   eos_id      ABC_END (abc phase) or MUSIC_END (semantic phase).
//   legal_lo/hi additive-mask legal range [lo,hi) — [0,EOD) for abc,
//               [CODEC_OFFSET,CODEC_OFFSET+CODEC_SIZE) for semantic.
//   history     every raw-vocab id generated by THIS branch so far, oldest
//               first (this function slices its own trailing
//               penalty_window — callers pass the FULL history, never a
//               pre-sliced window).
//   step        the AR loop's own counter; step == history.size() at every
//               call the real reference decode loop ever makes.
//   legacy_off  cot=="off" — see file header.
//   base        vocab id of scores[0] (doc 30 #1: the engine returns only the
//               lm_head rows a stage samples from). eos_id/legal_lo/legal_hi/
//               history stay ABSOLUTE ids; this function shifts them. A draw
//               from the returned scores is a relative index — add `base`.
static inline void yue2_distribution(std::vector<float> & scores, const Yue2SamplingParams & sp, int64_t eos_id,
                                     int64_t legal_lo, int64_t legal_hi, const std::vector<int32_t> & history,
                                     int64_t step, bool legacy_off, int64_t base = 0) {
    const int64_t V = (int64_t) scores.size();
    eos_id   -= base;
    legal_lo -= base;
    legal_hi -= base;

    // legal-token mask (additive, TRUE -inf) + EOS whitelist + min_tokens re-block.
    for (int64_t v = 0; v < V; v++) {
        const bool legal = (v >= legal_lo && v < legal_hi) || v == eos_id;
        if (!legal) {
            scores[(size_t) v] = -INFINITY;
        }
    }
    if (step < (int64_t) sp.min_tokens && eos_id >= 0 && eos_id < V) {
        scores[(size_t) eos_id] = -INFINITY;
    }

    // window_penalty: an exact integer frequency count (scatter_add of ones)
    // over history[-penalty_window:], then alpha=penalty**freq applied by a
    // sign-selected multiply/divide. Ids with freq==0 are untouched (alpha=1
    // is a no-op) — skip them rather than touch every one of 184704 entries.
    if (sp.repetition_penalty != 1.0f && !history.empty() && sp.penalty_window > 0) {
        const int64_t n     = (int64_t) history.size();
        const int64_t start = n > sp.penalty_window ? n - sp.penalty_window : 0;
        std::unordered_map<int32_t, int> freq;
        for (int64_t i = start; i < n; i++) {
            freq[history[(size_t) i]]++;
        }
        for (const auto & kv : freq) {
            const int64_t id = (int64_t) kv.first - base;
            if (id < 0 || id >= V) {
                continue;
            }
            float & s = scores[(size_t) id];
            if (!std::isfinite(s)) {
                continue;  // -inf * alpha (alpha finite, >0) stays -inf: a no-op, skip the pow
            }
            const float alpha = std::pow(sp.repetition_penalty, (float) kv.second);
            s                 = yue2_maybe_bf16(s < 0.0f ? s * alpha : s / alpha, legacy_off);
        }
    }

    if (sp.temperature == 0.0f) {
        return;  // greedy short-circuit -- caller does an unfiltered argmax; top-k/top-p never run
    }
    if (sp.temperature != 1.0f) {
        for (int64_t v = 0; v < V; v++) {
            float & s = scores[(size_t) v];
            if (std::isfinite(s)) {
                s = yue2_maybe_bf16(s / sp.temperature, legacy_off);
            }
        }
    }

    // top-k: threshold = k-th largest; STRICT less-than mask (a tie at the
    // threshold survives -- more than top_k entries can remain).
    const int64_t k = sp.top_k > 0 ? std::min<int64_t>(sp.top_k, V) : V;
    float         threshold = -INFINITY;
    if (k < V) {
        std::vector<float> sel;
        sel.reserve((size_t) std::min<int64_t>(k, std::max<int64_t>(0, legal_hi - legal_lo)));
        const int64_t lo = std::max<int64_t>(0, legal_lo);
        const int64_t hi = std::min<int64_t>(V, legal_hi);
        for (int64_t v = lo; v < hi; v++) {
            if (std::isfinite(scores[(size_t) v])) sel.push_back(scores[(size_t) v]);
        }
        if (eos_id >= 0 && eos_id < V && (eos_id < lo || eos_id >= hi) &&
            std::isfinite(scores[(size_t) eos_id])) {
            sel.push_back(scores[(size_t) eos_id]);
        }
        if ((int64_t) sel.size() >= k) {
            std::nth_element(sel.begin(), sel.begin() + (size_t) (k - 1), sel.end(), std::greater<float>());
            threshold = sel[(size_t) (k - 1)];
        }
    }
    if (std::isfinite(threshold)) {
        for (int64_t v = 0; v < V; v++) {
            if (scores[(size_t) v] < threshold) {
                scores[(size_t) v] = -INFINITY;
            }
        }
    }

    // top-p: sort descending, softmax, cumulative mass EXCLUDING the current
    // entry (`cumsum - probabilities > top_p`, sampling.py:48-49), force-keep
    // the top-1 (top-3 under legacy_off) regardless of cumulative mass.
    if (sp.top_p < 1.0f) {
        // Masked entries have zero softmax mass and remain -inf after top-p.
        // Sorting them wastes nearly all of the work for a 184,704-token
        // vocabulary when top-k leaves only a small set of candidates.
        std::vector<int64_t> idx;
        idx.reserve((size_t) (sp.top_k > 0 ? std::min<int64_t>(sp.top_k, V) : V));
        for (int64_t v = 0; v < V; v++) {
            if (std::isfinite(scores[(size_t) v])) idx.push_back(v);
        }
        if (idx.empty()) return;
        // Reference sort is UNSTABLE among exact ties (torch's own
        // `stable=False` default) — 02-fixture-schema.md §9's tied-group
        // policy explicitly allows any deterministic tie-break here as long
        // as the retained COUNT/score MULTISET within a tied group still
        // matches, never the specific ids. A stable sort by descending
        // score (ties broken by ascending id) is one such valid choice.
        std::stable_sort(idx.begin(), idx.end(),
                          [&](int64_t a, int64_t b) { return scores[(size_t) a] > scores[(size_t) b]; });
        const double max_v = (double) scores[(size_t) idx[0]];
        const int64_t n_candidates = (int64_t) idx.size();
        std::vector<double> probs((size_t) n_candidates);
        double               sum = 0.0;
        for (int64_t i = 0; i < n_candidates; i++) {
            const float  sv = scores[(size_t) idx[(size_t) i]];
            const double p  = std::isfinite(sv) ? std::exp((double) sv - max_v) : 0.0;
            probs[(size_t) i] = p;
            sum += p;
        }
        const int64_t keep_floor = legacy_off ? 3 : 1;
        double        cum        = 0.0;
        for (int64_t i = 0; i < n_candidates; i++) {
            const double before = cum;  // cumulative mass BEFORE this entry
            cum += sum > 0.0 ? probs[(size_t) i] / sum : 0.0;
            const bool removed = (i >= keep_floor) && (before > (double) sp.top_p);
            if (removed) {
                scores[(size_t) idx[(size_t) i]] = -INFINITY;
            }
        }
    }
}

// ── The same chain over survivors only (the decode loops' fast path) ───────
//
// yue2_distribution + yue2_sample_draw touch every entry of the row several
// times per step (mask, top-k selection over a full copy, threshold, top-p
// gather, the draw's scan): 0.5 ms a composing step and 2 ms a planning step
// on a 5090 (2026-09-28), 14% and 26% of those stages. This does the same
// arithmetic in the same order but visits the row twice (a size-k heap for the
// k-th largest score, then a gather of the survivors) and runs top-p and the
// draw over the survivors. Survivors come out in ascending id, which is the
// order both the reference top-p tie-break and the draw walk, so the token
// drawn for a seed is the one yue2_distribution + yue2_sample_draw would draw
// (YUE2_SAMPLER_CHECK=1 runs both and says so if they ever differ).
//
// `scores` is the caller's scratch row and is modified (mask and penalty land
// in it). Temperature 0 (greedy) is not handled here: callers keep the full
// path for it.
static inline void yue2_distribution_survivors(float * scores, int64_t V, const Yue2SamplingParams & sp,
                                               int64_t eos_id, int64_t legal_lo, int64_t legal_hi,
                                               const std::vector<int32_t> & history, int64_t step, bool legacy_off,
                                               int64_t base, std::vector<int32_t> * ids, std::vector<float> * vals) {
    eos_id   -= base;
    legal_lo -= base;
    legal_hi -= base;
    ids->clear();
    vals->clear();

    // Legal mask: only the entries outside [lo,hi) can change.
    const int64_t lo = std::max<int64_t>(0, std::min<int64_t>(V, legal_lo));
    const int64_t hi = std::max<int64_t>(lo, std::min<int64_t>(V, legal_hi));
    for (int64_t v = 0; v < lo; v++) if (v != eos_id) scores[v] = -INFINITY;
    for (int64_t v = hi; v < V; v++) if (v != eos_id) scores[v] = -INFINITY;
    if (step < (int64_t) sp.min_tokens && eos_id >= 0 && eos_id < V) scores[eos_id] = -INFINITY;

    if (sp.repetition_penalty != 1.0f && !history.empty() && sp.penalty_window > 0) {
        const int64_t n     = (int64_t) history.size();
        const int64_t start = n > sp.penalty_window ? n - sp.penalty_window : 0;
        std::unordered_map<int32_t, int> freq;
        for (int64_t i = start; i < n; i++) freq[history[(size_t) i]]++;
        for (const auto & kv : freq) {
            const int64_t id = (int64_t) kv.first - base;
            if (id < 0 || id >= V) continue;
            float & s = scores[id];
            if (!std::isfinite(s)) continue;
            const float alpha = std::pow(sp.repetition_penalty, (float) kv.second);
            s                 = yue2_maybe_bf16(s < 0.0f ? s * alpha : s / alpha, legacy_off);
        }
    }

    // Temperature is applied on the fly, with the reference's own arithmetic.
    const bool  tempered = sp.temperature != 1.0f;
    const float temp     = sp.temperature;
    auto        final_of = [&](int64_t v) {
        const float s = scores[v];
        return tempered && std::isfinite(s) ? yue2_maybe_bf16(s / temp, legacy_off) : s;
    };
    // Every finite entry is in [lo,hi) or is the end token, so these two
    // spans are the whole candidate set.
    auto for_each_candidate = [&](auto && fn) {
        if (eos_id >= 0 && eos_id < lo) fn(eos_id);
        for (int64_t v = lo; v < hi; v++) fn(v);
        if (eos_id >= hi && eos_id < V) fn(eos_id);
    };

    // top-k threshold: the k-th largest finite score, duplicates counted,
    // exactly what nth_element over every finite score picks.
    const int64_t k         = sp.top_k > 0 ? sp.top_k : V;
    float         threshold = -INFINITY;
    if (k < V) {
        std::vector<float> heap;  // min-heap of the k largest so far
        heap.reserve((size_t) k);
        int64_t n_finite = 0;
        for_each_candidate([&](int64_t v) {
            const float s = final_of(v);
            if (!std::isfinite(s)) return;
            n_finite++;
            if ((int64_t) heap.size() < k) {
                heap.push_back(s);
                std::push_heap(heap.begin(), heap.end(), std::greater<float>());
            } else if (s > heap.front()) {
                std::pop_heap(heap.begin(), heap.end(), std::greater<float>());
                heap.back() = s;
                std::push_heap(heap.begin(), heap.end(), std::greater<float>());
            }
        });
        if (n_finite >= k) threshold = heap.front();
    }
    for_each_candidate([&](int64_t v) {
        const float s = final_of(v);
        if (std::isfinite(s) && !(s < threshold)) {
            ids->push_back((int32_t) v);
            vals->push_back(s);
        }
    });

    if (sp.top_p < 1.0f && !ids->empty()) {
        const size_t        n = ids->size();
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), (size_t) 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return (*vals)[a] > (*vals)[b]; });
        const double        max_v = (double) (*vals)[order[0]];
        std::vector<double> probs(n);
        double              sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            probs[i] = std::exp((double) (*vals)[order[i]] - max_v);
            sum += probs[i];
        }
        const size_t      keep_floor = legacy_off ? 3 : 1;
        std::vector<bool> removed(n, false);
        double            cum = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double before = cum;
            cum += sum > 0.0 ? probs[i] / sum : 0.0;
            if (i >= keep_floor && before > (double) sp.top_p) removed[order[i]] = true;
        }
        size_t w = 0;
        for (size_t i = 0; i < n; i++) {
            if (removed[i]) continue;
            (*ids)[w]  = (*ids)[i];
            (*vals)[w] = (*vals)[i];
            w++;
        }
        ids->resize(w);
        vals->resize(w);
    }
}

// yue2_sample_draw over the survivors: same max, same exp/sum order, same
// walk. Returns a row-relative index, like yue2_sample_draw.
static inline int64_t yue2_draw_survivors(const std::vector<int32_t> & ids, const std::vector<float> & vals,
                                          std::mt19937_64 & rng) {
    if (ids.empty()) return 0;  // fully masked row: yue2_sample_draw's argmax fallback lands on 0 too
    double max_v    = -INFINITY;
    size_t arg_best = 0;
    for (size_t i = 0; i < vals.size(); i++) {
        if (vals[i] > max_v) {
            max_v    = vals[i];
            arg_best = i;
        }
    }
    std::vector<double> p(vals.size());
    double              sum = 0.0;
    for (size_t i = 0; i < vals.size(); i++) {
        p[i] = std::exp((double) vals[i] - max_v);
        sum += p[i];
    }
    if (!(sum > 0.0)) return ids[arg_best];
    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    double       acc = 0.0;
    for (size_t i = 0; i < p.size(); i++) {
        acc += p[i];
        if (acc > u) return ids[i];
    }
    return ids[arg_best];
}

// YUE2_SAMPLER_CHECK=1: run the full reference path beside the fast one on
// every step and log any disagreement. A test switch, not a production knob.
static inline bool yue2_sampler_check_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("YUE2_SAMPLER_CHECK");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}

// ── Drawing an actual token (production decode loop / --generate) ──────────
//
// Sampled ids are never fixture-diffable regardless of implementation
// (03-reference-numerics.md §5 — PyTorch's own multinomial kernel isn't
// portable), so any deterministic seeded draw is equally "correct" for a
// user-facing seed. Reuses mm3_sample_top_k's own tail (std::mt19937_64 +
// ascending inverse-CDF walk), in spirit, over whatever `distribution()`
// left as survivors (finite entries only).

static inline int64_t yue2_sample_argmax(const std::vector<float> & scores) {
    int64_t best  = 0;
    float   bestv = scores.empty() ? 0.0f : scores[0];
    for (size_t v = 1; v < scores.size(); v++) {
        if (scores[v] > bestv) {
            bestv = scores[v];
            best  = (int64_t) v;
        }
    }
    return best;
}

static inline int64_t yue2_sample_draw(const std::vector<float> & scores, std::mt19937_64 & rng) {
    const int64_t V = (int64_t) scores.size();
    if (V <= 0) {
        return 0;
    }
    double  max_v    = -INFINITY;
    int64_t arg_best = 0;
    std::vector<int64_t> candidates;
    candidates.reserve(128);
    for (int64_t v = 0; v < V; v++) {
        const float score = scores[(size_t) v];
        if (score > max_v) {
            max_v    = score;
            arg_best = v;
        }
        if (std::isfinite(score)) candidates.push_back(v);
    }
    if (!std::isfinite(max_v)) {
        return arg_best;  // fully-masked row -- degenerate, fall back to argmax
    }
    std::vector<double> p;
    p.reserve(candidates.size());
    double               sum = 0.0;
    for (int64_t v : candidates) {
        const double e = std::exp((double) scores[(size_t) v] - max_v);
        p.push_back(e);
        sum += e;
    }
    if (!(sum > 0.0)) {
        return arg_best;
    }
    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    double       acc = 0.0;
    for (size_t i = 0; i < candidates.size(); i++) {
        acc += p[i];
        if (acc > u) {
            return candidates[i];
        }
    }
    return arg_best;
}

// P(eos) under the post-chain distribution of a full row: the semantic
// stage's end_threshold test, with its own float/double arithmetic.
static inline double yue2_end_prob_row(const std::vector<float> & s, int64_t eos_rel) {
    if (eos_rel < 0 || eos_rel >= (int64_t) s.size() || !std::isfinite(s[(size_t) eos_rel])) return -1.0;
    float mx = -INFINITY;
    for (float v : s) if (std::isfinite(v) && v > mx) mx = v;
    double z = 0.0;
    for (float v : s) if (std::isfinite(v)) z += std::exp((double) (v - mx));
    return std::exp((double) (s[(size_t) eos_rel] - mx)) / z;
}

// The same over the survivors (ascending id, every one finite).
static inline double yue2_end_prob_survivors(const std::vector<int32_t> & ids, const std::vector<float> & vals,
                                             int64_t eos_rel) {
    size_t at = ids.size();
    for (size_t i = 0; i < ids.size(); i++) if (ids[i] == eos_rel) at = i;
    if (at == ids.size()) return -1.0;
    float mx = -INFINITY;
    for (float v : vals) if (v > mx) mx = v;
    double z = 0.0;
    for (float v : vals) z += std::exp((double) (v - mx));
    return std::exp((double) (vals[at] - mx)) / z;
}

// One token from one logits row, for the plan and semantic decode loops.
// `row` is scratch (modified). The survivors of the chain stay in `*ids` /
// `*vals` (ascending id, final scores); the greedy path leaves them empty.
// end_threshold > 0: when the chain leaves P(eos) at or above it, return -1
// without drawing (the rng does not advance), as the semantic stage's
// threshold end always has.
static inline int64_t yue2_sample_row(float * row, int64_t V, const Yue2SamplingParams & sp, int64_t eos_id,
                                      int64_t legal_lo, int64_t legal_hi, const std::vector<int32_t> & history,
                                      int64_t step, bool legacy_off, int64_t base, std::mt19937_64 & rng,
                                      std::vector<int32_t> * ids, std::vector<float> * vals,
                                      float end_threshold = 0.0f) {
    if (sp.temperature == 0.0f) {
        std::vector<float> s(row, row + V);
        yue2_distribution(s, sp, eos_id, legal_lo, legal_hi, history, step, legacy_off, base);
        ids->clear();
        vals->clear();
        if (end_threshold > 0.0f && yue2_end_prob_row(s, eos_id - base) >= (double) end_threshold) return -1;
        return yue2_sample_argmax(s);
    }
    const bool         check = yue2_sampler_check_enabled();
    std::vector<float> ref;
    std::mt19937_64    ref_rng;
    if (check) {
        ref.assign(row, row + V);
        ref_rng = rng;
    }
    yue2_distribution_survivors(row, V, sp, eos_id, legal_lo, legal_hi, history, step, legacy_off, base, ids, vals);
    const bool    stop = end_threshold > 0.0f && yue2_end_prob_survivors(*ids, *vals, eos_id - base) >= (double) end_threshold;
    const int64_t tok  = stop ? -1 : yue2_draw_survivors(*ids, *vals, rng);
    if (check) {
        static std::atomic<long long> checked{ 0 }, differed{ 0 };
        yue2_distribution(ref, sp, eos_id, legal_lo, legal_hi, history, step, legacy_off, base);
        const bool    ref_stop = end_threshold > 0.0f && yue2_end_prob_row(ref, eos_id - base) >= (double) end_threshold;
        const int64_t want     = ref_stop ? -1 : yue2_sample_draw(ref, ref_rng);
        if (want != tok) {
            differed++;
            fprintf(stderr, "[YuE2-SamplerCheck] MISMATCH at step %lld: fast %lld, reference %lld\n", (long long) step,
                    (long long) tok, (long long) want);
        }
        if (++checked % 2000 == 0) {
            fprintf(stderr, "[YuE2-SamplerCheck] %lld draws checked, %lld differed\n", checked.load(), differed.load());
        }
    }
    return tok;
}
