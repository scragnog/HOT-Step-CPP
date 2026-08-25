#pragma once
// minimax/mm3-sample.h — the MiniMax-Music3 top-k multinomial sampler.
//
// HOT-Step file (does not exist upstream). Included by minimax/mm3-depth-graph.h
// and minimax/mm3-ar-loop.h, both of which reach tools/hot-step-server.cpp only
// through the single minimax/mm3-server.h hook include.
//
// SCOPE: one function. Both AR-stage samplers — the global LM's semantic code and
// the depth decoder's seven acoustic codes — call the SAME reference helper
// (`_sample_top_k` in the diffusers `encoders.py` at commit dafe3733), so it lives
// once, here, instead of twice in two graph files.
//
// ── The reference, line by line ──────────────────────────────────────────────
//
//   values    = nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
//   top_k     = min(50, values.shape[-1])
//   threshold = topk(values, top_k).values[..., -1]        # the k-th LARGEST
//   values    = values.masked_fill(values < threshold, -inf)
//   probs     = nan_to_num(softmax(values), nan=0.0)
//   probs     = probs / probs.sum().clamp_min(1e-12)
//   return multinomial(probs, 1, generator)
//
// Three details that are easy to lose:
//
// 1. THERE IS NO TEMPERATURE. The softmax is over the raw (guided) logits. The
//    checkpoint's recipe exposes exactly two AR sampling knobs — cfg 1.5 and
//    top-k 50 — and both are fixed metadata (`mm3.ar.*`), not user parameters.
//
// 2. `-inf` BECOMES `-1e9` BEFORE THE TOP-K, NOT AFTER. That matters only when
//    fewer than k entries are finite: then the k-th largest is -1e9, nothing is
//    filtered, and the -1e9 entries still contribute exp(-1e9 - max) = 0. So the
//    result is the same as filtering them — but reproducing the order keeps the
//    degenerate case from diverging.
//
// 3. THE COMPARISON IS STRICTLY LESS. Values EQUAL to the threshold survive, so a
//    tie at the boundary keeps more than k candidates. This is observable in the
//    fixtures: `lm_i*_guided_logits` has 50..55 finite entries, not exactly 50.
//
// ── What cannot be reproduced, and what we do instead ────────────────────────
//
// `torch.multinomial` with a torch Generator is not reproducible outside torch —
// not the Mersenne draw, not the CDF walk order. So a seeded C++ run does NOT
// reproduce the reference's code sequence, and never can. Parity is therefore
// validated on LOGITS with the sampler bypassed (`forced_*`), exactly as the depth
// decoder increment did. Here the draw is a plain inverse-CDF walk in ascending
// index order over a std::mt19937_64 — deterministic for a given seed, which is
// what a user-facing `seed` parameter actually needs to mean.

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

// Sample one index from `logits[0, n)` under the reference recipe above.
//
//   scratch   optional caller-owned buffer, reused across calls to keep the AR
//             loop allocation-free; may be null.
//   returns   an index into `logits`, always in [0, n). Never negative: a fully
//             degenerate distribution (all zero mass) falls back to the argmax.
static int64_t mm3_sample_top_k(const float * logits, int64_t n, int top_k, std::mt19937_64 & rng,
                                std::vector<float> * scratch = nullptr) {
    if (n <= 0) {
        return 0;
    }
    std::vector<float>   local;
    std::vector<float> & v = scratch ? *scratch : local;
    v.resize((size_t) n);

    // nan_to_num, and the argmax fallback in the same sweep.
    int64_t arg_best = 0;
    float   val_best = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        float x = logits[i];
        if (std::isnan(x)) {
            x = -1e9f;
        } else if (std::isinf(x)) {
            x = x > 0.0f ? 1e9f : -1e9f;
        }
        v[(size_t) i] = x;
        if (x > val_best) {
            val_best = x;
            arg_best = i;
        }
    }

    int64_t k = top_k > 0 ? (int64_t) top_k : n;
    if (k > n) {
        k = n;
    }

    // The k-th largest. nth_element over a copy: the sampler runs 25 times a
    // second over 16385 candidates, so an O(n) selection matters and a full sort
    // does not pay for itself.
    float threshold = -INFINITY;
    if (k < n) {
        std::vector<float> sel(v);
        std::nth_element(sel.begin(), sel.begin() + (size_t) (k - 1), sel.end(), std::greater<float>());
        threshold = sel[(size_t) (k - 1)];
    }

    // softmax over the survivors, shifted by their max for stability.
    float max_v = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold && v[(size_t) i] > max_v) {
            max_v = v[(size_t) i];
        }
    }
    if (!std::isfinite(max_v)) {
        return arg_best;
    }

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold) {
            const double p = std::exp((double) v[(size_t) i] - (double) max_v);
            sum += p;
            v[(size_t) i] = (float) p;
        } else {
            v[(size_t) i] = 0.0f;
        }
    }
    if (!(sum > 0.0)) {
        return arg_best;
    }

    // Inverse-CDF walk in ascending index order.
    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    double       acc = 0.0;
    for (int64_t i = 0; i < n; i++) {
        acc += (double) v[(size_t) i];
        if (acc > u) {
            return i;
        }
    }
    return arg_best;
}


// ── semantic-stream sampling knobs (2026-08-25) ─────────────────────────────
//
// The reference recipe stays the DEFAULT: temperature 1, top-k from the
// checkpoint metadata, no top-p, no penalty — at those values every branch
// below is inert and a render is bit-identical to the pre-knob build (gated by
// hash at introduction). The knobs exist because adapters sharpen the semantic
// code distribution and memorising adapters emit verbatim code loops; the
// penalty modes are a port of the ACE LM's (pipeline-lm.cpp), whose tuning was
// measured, with the time constants RESCALED: ACE codes run at 5 Hz, MM3
// semantic frames at 25 fps, so ACE's window 64 (~13 s) is ~320 frames here
// and DRY's min_len 6 (~1.2 s) is ~15 frames (0.6 s, the value ACE measured
// as loop-free). Copying the ACE numbers literally would penalise 0.12 s of
// similarity and chew up sustained textures.
//
// ONLY the global LM's semantic draw takes these. The depth decoder keeps the
// reference sampler: its seven codes are per-frame texture, loops do not live
// there, and perturbing it re-opens the timbre question the acoustic loss
// just closed.

enum MM3RepMode { MM3_REP_PRESENCE = 0, MM3_REP_FREQUENCY = 1, MM3_REP_DRY = 2 };

#define MM3_REP_FREQ_MAX_COUNT 8
#define MM3_REP_DRY_MAX_EXP    8
#define MM3_REP_DRY_MAX_MATCH  64

struct MM3SamplerKnobs {
    float       temperature = 1.0f;   // divides the guided logits; 1 = reference
    int         top_k       = 0;      // 0 = the checkpoint's mm3.ar.top_k (50)
    float       top_p       = 0.0f;   // 0 or >=1 = off; else nucleus over the top-k survivors
    float       rep_penalty = 1.0f;   // 1 = off. Shared 1.0-1.5 slider, ACE semantics
    int         rep_window  = 320;    // ~12.8 s of semantic frames at 25 fps
    int         rep_mode    = MM3_REP_DRY;
    float       dry_base    = 1.75f;
    int         dry_min_len = 15;     // ~0.6 s — ACE's measured loop-free point, rescaled

    bool any_active() const {
        return temperature != 1.0f || top_k > 0 || (top_p > 0.0f && top_p < 1.0f) || rep_penalty > 1.0f;
    }
};

static int mm3_rep_mode_from_name(const std::string & name) {
    if (name == "frequency") {
        return MM3_REP_FREQUENCY;
    }
    if (name == "presence") {
        return MM3_REP_PRESENCE;
    }
    if (!name.empty() && name != "dry") {
        fprintf(stderr, "[MM3-Rep] WARNING unknown rep mode \"%s\", using \"dry\"\n", name.c_str());
    }
    return MM3_REP_DRY;
}

// Candidate layout contract (mm3-ar-loop.h): index 0 is EOS, index j+1 is
// semantic code j. EOS is NEVER penalised — punishing the ending for having
// been emitted before is not a thing, and the history holds codes, not EOS.
// `hist` is the emitted semantic code stream for THIS take, oldest first.
static void mm3_apply_rep_penalty(float * cand, int64_t ncand, const int32_t * hist, int64_t n_hist,
                                  const MM3SamplerKnobs & k) {
    if (k.rep_penalty <= 1.0f || n_hist <= 0 || k.rep_window <= 0) {
        return;
    }
    const int64_t start = n_hist > k.rep_window ? n_hist - k.rep_window : 0;

    if (k.rep_mode == MM3_REP_DRY) {
        // Penalise ONLY a code that would extend a verbatim recent cycle,
        // exponentially in the matched suffix length. Port of ACE's DRY
        // including the slider mapping: (penalty-1)*8, so 1.1 lands on
        // llama.cpp's canonical 0.8 multiplier.
        const int   min_len    = k.dry_min_len > 1 ? k.dry_min_len : 1;
        const float multiplier = (k.rep_penalty - 1.0f) * 8.0f;
        const float base       = k.dry_base > 1.0f ? k.dry_base : 1.75f;
        std::unordered_map<int32_t, int> best_match;
        for (int64_t j = start + 1; j < n_hist; j++) {
            int L = 0;
            while (L < MM3_REP_DRY_MAX_MATCH && j - 1 - L >= start && hist[j - 1 - L] == hist[n_hist - 1 - L]) {
                L++;
                if (n_hist - 1 - L < start) {
                    break;
                }
            }
            if (L < min_len) {
                continue;
            }
            auto it = best_match.find(hist[j]);
            if (it == best_match.end() || L > it->second) {
                best_match[hist[j]] = L;
            }
        }
        for (const auto & kv : best_match) {
            if ((int64_t) kv.first + 1 >= ncand || kv.first < 0) {
                continue;
            }
            int   e   = kv.second - min_len;
            float sub = multiplier * powf(base, (float) (e < MM3_REP_DRY_MAX_EXP ? e : MM3_REP_DRY_MAX_EXP));
            cand[(size_t) (kv.first + 1)] -= sub;
        }
        return;
    }

    if (k.rep_mode == MM3_REP_FREQUENCY) {
        std::unordered_map<int32_t, int> counts;
        for (int64_t i = start; i < n_hist; i++) {
            counts[hist[i]]++;
        }
        for (const auto & kv : counts) {
            if ((int64_t) kv.first + 1 >= ncand || kv.first < 0) {
                continue;
            }
            int     c = kv.second < MM3_REP_FREQ_MAX_COUNT ? kv.second : MM3_REP_FREQ_MAX_COUNT;
            float   p = powf(k.rep_penalty, (float) c);
            float & l = cand[(size_t) (kv.first + 1)];
            l = l > 0.0f ? l / p : l * p;
        }
        return;
    }

    std::unordered_set<int32_t> seen;
    for (int64_t i = start; i < n_hist; i++) {
        seen.insert(hist[i]);
    }
    for (int32_t c : seen) {
        if ((int64_t) c + 1 >= ncand || c < 0) {
            continue;
        }
        float & l = cand[(size_t) (c + 1)];
        l = l > 0.0f ? l / k.rep_penalty : l * k.rep_penalty;
    }
}

// The knobbed semantic draw. At default knobs this reduces EXACTLY to
// mm3_sample_top_k(logits, n, default_top_k, rng, scratch) — same float path,
// same draw — which is the bit-parity contract.
static int64_t mm3_sample_knobbed(float * logits, int64_t n, int default_top_k,
                                  const MM3SamplerKnobs & k, std::mt19937_64 & rng,
                                  std::vector<float> * scratch = nullptr) {
    const int eff_k = k.top_k > 0 ? k.top_k : default_top_k;
    if (k.temperature != 1.0f && k.temperature > 0.0f) {
        for (int64_t i = 0; i < n; i++) {
            if (std::isfinite(logits[i])) {
                logits[i] /= k.temperature;
            }
        }
    }
    if (!(k.top_p > 0.0f && k.top_p < 1.0f)) {
        return mm3_sample_top_k(logits, n, eff_k, rng, scratch);
    }
    // Nucleus over the top-k survivors: run the reference top-k filter into a
    // probability vector, then keep the smallest prefix of descending-prob
    // candidates whose mass reaches top_p, and draw within it.
    std::vector<float>   local;
    std::vector<float> & v = scratch ? *scratch : local;
    v.assign((size_t) n, 0.0f);
    // Reuse the reference machinery for nan handling + threshold by calling the
    // same steps: cheapest correct route is a softmax over the k survivors here.
    std::vector<std::pair<float, int64_t>> surv;
    surv.reserve((size_t) (eff_k + 8));
    {
        std::vector<float> sel;
        sel.reserve((size_t) n);
        for (int64_t i = 0; i < n; i++) {
            float x = logits[i];
            if (std::isnan(x)) {
                x = -1e9f;
            } else if (std::isinf(x)) {
                x = x > 0.0f ? 1e9f : -1e9f;
            }
            v[(size_t) i] = x;
            sel.push_back(x);
        }
        int64_t kk = eff_k > 0 && (int64_t) eff_k < n ? (int64_t) eff_k : n;
        float   threshold = -INFINITY;
        if (kk < n) {
            std::nth_element(sel.begin(), sel.begin() + (size_t) (kk - 1), sel.end(), std::greater<float>());
            threshold = sel[(size_t) (kk - 1)];
        }
        for (int64_t i = 0; i < n; i++) {
            if (v[(size_t) i] >= threshold) {
                surv.emplace_back(v[(size_t) i], i);
            }
        }
    }
    std::sort(surv.begin(), surv.end(),
              [](const std::pair<float, int64_t> & a, const std::pair<float, int64_t> & b) {
                  return a.first > b.first || (a.first == b.first && a.second < b.second);
              });
    double max_v = (double) surv.front().first;
    double sum   = 0.0;
    std::vector<double> probs;
    probs.reserve(surv.size());
    for (const auto & sv : surv) {
        double p = std::exp((double) sv.first - max_v);
        probs.push_back(p);
        sum += p;
    }
    if (!(sum > 0.0)) {
        return surv.front().second;
    }
    double  cum  = 0.0;
    size_t  keep = probs.size();
    for (size_t i = 0; i < probs.size(); i++) {
        cum += probs[i] / sum;
        if (cum >= (double) k.top_p) {
            keep = i + 1;
            break;
        }
    }
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; i++) {
        kept_sum += probs[i];
    }
    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * kept_sum;
    double       acc = 0.0;
    for (size_t i = 0; i < keep; i++) {
        acc += probs[i];
        if (acc > u) {
            return surv[i].second;
        }
    }
    return surv[keep - 1].second;
}
