#pragma once
// minimax/mm3-ar-loop.h — the MiniMax-Music3 autoregressive planning loop.
//
// HOT-Step file (does not exist upstream). Included only by minimax/mm3-server.h,
// which is itself the single hook include into tools/hot-step-server.cpp.
//
// SCOPE (increment 5b): the loop that turns a tokenised prompt into the
// [F, 8, 4096] block of per-frame hidden states the condition encoder consumes,
// plus the RVQ codes that produced them. This is stage 1 of the pipeline end to
// end: prefill -> {sample semantic, depth-decode 7 acoustic, feed back} x F.
// What is still missing for a full render is the orchestration above it — chunk
// windowing, condition encoding, flow sampling, vocoding — all of which already
// exist as validated modules.
//
// ── The loop, from the reference (diffusers `encoders.py` @ dafe3733) ─────────
//
//   text_embeds = embed_tokens(text_ids)                  # [2, T, H]
//   out         = lm(inputs_embeds=text_embeds, use_cache=True)
//   last_hidden = out.last_hidden_state[:, -1]            # [2, H]
//
//   for i in range(max_frames + 1):
//       logits  = lm_head(last_hidden).float()            # [2, V]
//       logits  = logits.masked_fill(vocab_mask, -inf)    # keep semantic + EOS
//       guided  = uncond + (cond - uncond) * 1.5
//       thresh  = topk(cond, 50).values[..., -1]
//       guided  = guided.masked_fill(cond < thresh, -inf)
//       guided  = guided.masked_fill(vocab_mask, -inf)    # CFG on two -inf -> NaN
//       sampled = sample_top_k(guided)                    # top-50 AGAIN, then multinomial
//       if sampled == 151670: break                       # EOS
//       codes, depth_hidden = depth_decode(last_hidden, sampled - 151675)
//       if i > 0:
//           frame_hiddens.append(cat(last_hidden[:1], depth_hidden))
//           if len(frame_hiddens) >= max_frames: break
//       feedback    = embed_audio_frame(codes)            # [2, 1, H]
//       out         = lm(inputs_embeds=feedback, past_key_values=..., use_cache=True)
//       last_hidden = out.last_hidden_state[:, -1]
//
// ── The five things that are easy to get wrong ───────────────────────────────
//
// 1. ITERATION 0 IS FED BACK BUT NOT EMITTED. Its codes are sampled, depth-decoded
//    and pushed into the LM's history, and then its hidden states are DISCARDED.
//    Emitted frame j is iteration j+1. The fixtures encode this twice over:
//    `codes_semantic_all` has 301 entries for 300 emitted frames, and
//    `codes_semantic_emitted == codes_semantic_all[1:]`. Get this off by one and
//    the audio is a frame out of step with its own conditioning — subtle, and it
//    would survive every per-module parity check.
//
// 2. THE MAX-FRAMES BREAK COMES BEFORE THE FEEDBACK. The final emitted frame never
//    runs a decode step, so the last iteration has no feedback embedding at all.
//    (The dump for that iteration carries zeros, not stale data.)
//
// 3. TOP-K IS A DOUBLE FILTER, AND THE TWO PASSES RANK BY DIFFERENT THINGS. The
//    first keeps the CONDITIONAL row's top 50 (ties survive, so 50..55 candidates
//    in practice); the second, inside the sampler, keeps the top 50 of what is
//    left ranked by the GUIDED value. Collapsing them into one filter changes the
//    candidate set. `mm3-sample.h` owns the second pass.
//
// 4. THE UNCONDITIONAL ROW IS NOT AN EMPTY PROMPT. It is the conditional token
//    sequence with everything between index 0 and the final two replaced by
//    token 151654, so it keeps the same length and the same RoPE positions.
//    `mm3-tokenizer.h` owns that; the loop just prefills both rows.
//
// 5. THE VOCAB MASK IS PART OF THE CONTRACT, NOT A SAFETY NET. Only the 16384
//    semantic codes and the EOS token may be sampled; the other 183615 vocab
//    entries are masked to -inf on BOTH rows before CFG. This loop never
//    materialises them: it gathers the 16385 candidate logits straight out of the
//    head output, which is arithmetically identical (every masked entry would be
//    -inf, hence below any threshold and outside every top-k) and turns a
//    400000-element sweep per frame into a 32770-element one.
//
// ── Sampling determinism ─────────────────────────────────────────────────────
//
// `torch.multinomial` with a torch Generator cannot be reproduced in C++, so a
// seeded run here does NOT reproduce the reference's code sequence. Parity is
// validated with the sampler bypassed: `forced_semantic` / `forced_acoustic` feed
// the fixture's own codes so the LM and the depth decoder see exactly the token
// sequence the reference saw, and the LOGITS are then comparable. The seeded path
// is validated separately for determinism (same seed twice -> identical codes),
// range, and absence of NaN. Same protocol the depth-decoder increment used.
//
// ── Validation, measured 2026-08-13 (RTX 5090, f16 GGUF) ─────────────────────
//
// THE OFF-BY-ONE IS PROVEN, NOT ASSERTED. A forced 301-iteration replay of the
// fixture's own codes produced a [300, 8, 4096] block; its first 200 frames
// against `cond_in_w0.bin` (the condition encoder's real input for window 0):
//
//     corr 0.9998494   rel RMSE 1.737e-2   (per layer 1.43e-2 .. 2.02e-2, flat)
//
// 1.74e-2 is, to three digits, the flow DiT's independently measured bf16-dump
// floor — so the whole stage-1 chain (prefill -> 301 x {LM step, depth decode,
// feedback}) reproduces the reference to within the capture's own precision.
// Re-running the SAME comparison one frame out of alignment gives 8.50e-1, a 49x
// worse fit: the emission indexing is right, and it is the kind of bug that would
// otherwise pass every per-module check.
//
// Sampled path (seed 1234, 30 frames): identical codes across two runs, different
// codes for a different seed, all semantic codes in [0, 16384), all acoustic in
// [0, 1024), zero non-finite logits. A 7500-frame request hit EOS naturally at
// frame 1200 and stopped there, exercising the stop token.
//
// Cost per emitted frame, steady state over 1200 frames:
//     LM decode step   15.26 ms
//     depth decode      9.20 ms   (7 codebook graphs; matches the standalone
//                                  increment's 9.2 ms exactly)
//     host (mask, CFG, double top-k, sampling)   0.22 ms
//     ------------------------------------------------
//     total            24.70 ms/frame
//
// At 25 fps that is 0.62 s of wall clock per second of audio — 1.6x faster than
// realtime for the planning stage, against the reference's 539 ms/frame on CPU.
// The depth decoder is 37 % of it despite being 7 % of the parameters, for the
// reason its own header gives: seven sweeps of 0.6 B weights over <=8 tokens is
// launch-bound, not bandwidth-bound. That is where the next speed lever is.

#include "mm3-align.h"
#include "mm3-depth-graph.h"
#include "mm3-lm-graph.h"
#include "mm3-model.h"
#include "mm3-sample.h"
#include "mm3-tokenizer.h"
#include "../yue2/yue2-guide.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// Per-iteration parity capture. Only the first `dump_iters` iterations are kept —
// the blocks are ~370 kB each and the loop runs up to 9000 times.
struct MM3ArDump {
    std::vector<float> last_hidden;   // [2, H]      rows [cond, uncond]
    std::vector<float> sem_logits;    // [2, SV]     RAW head logits over the semantic slice
    std::vector<float> guided;        // [SV]        post-CFG, post-double-filter (-inf where masked)
    std::vector<float> feedback;      // [2, H]      rows identical; zeros if no feedback ran
    std::vector<float> depth_hidden;  // [NC, H]     conditional row only
};

struct MM3ArResult {
    // The seed this take actually drew from: `MM3ArOptions::seed + take index`.
    // Recorded because an ensemble render has to be able to say which seed
    // produced which song — otherwise a good take cannot be reproduced.
    uint64_t seed        = 0;
    int64_t n_iterations = 0;  // AR steps actually run (emitted frames + 1, unless EOS)
    int64_t n_frames     = 0;  // emitted frames F
    int64_t hidden_dim   = 0;  // H
    int64_t n_codebooks  = 0;  // 8 = 1 semantic + 7 acoustic
    int64_t sem_vocab    = 0;  // SV
    bool    eos_hit      = false;
    // EOS trace (MM3ArOptions::eos_trace), one entry per iteration, take 0 only:
    //   eos_p_cond   softmax of the raw CONDITIONAL candidate row at EOS
    //   eos_p_final  what the sampler would draw: CFG-guided, masked by the
    //                conditional top-k, then top-k'd again (mm3_sample_top_k)
    //   eos_rank     EOS's rank in the conditional row (0 = argmax)
    //   eos_topk_ok  1 when EOS survived the conditional top-k mask
    // Exists to answer "does the adapter still predict an ending at the true
    // end of a real track" under forced replay, without touching sampling.
    std::vector<float>   eos_p_cond, eos_p_final;
    std::vector<int32_t> eos_rank, eos_topk_ok;
    // LRC text derived from the LM's own decode attention (mm3-align.h). Empty
    // when alignment was off, the track was instrumental, or alignment failed —
    // never a reason to fail the run.
    std::string lrc;

    std::vector<int32_t> semantic_all;    // [I]        includes the un-emitted iteration 0
    std::vector<int32_t> acoustic_all;    // [I, NC]
    std::vector<float>   frame_hiddens;   // [F, NC+1, H] layer-major: LM hidden then 7 depth hiddens
    std::vector<float>   prefill_hidden;  // [2, H]
    std::vector<MM3ArDump> dumps;

    // ── Forward-math probes, for comparing across take counts ────────────────
    // Takes share one batched forward, so "is take t still computing its own
    // song?" cannot be answered from the CODES: the top-k multinomial is
    // chaotic, and a 1-ulp difference flips one draw and then diverges forever.
    // These are sampled BEFORE the RNG is consulted.
    //
    // TRAP, PAID FOR ONCE: `prefill_sum` is a sum over 4096 signed activations
    // that very nearly cancel, so it is NOT an equality test. Two rows agreeing
    // to corr 0.9994 against the reference can have sums 35.1 and 17.6 — the
    // cancellation amplifies a 1e-3 per-element wobble into tens. Use it only
    // as a cheap "did this change at all" tripwire; judge correctness with
    // correlation / relative RMSE against the fixtures instead
    // (server/scripts/check-mm3-ensemble.mjs).
    double  prefill_sum  = 0.0;  // sum of this take's prefill hidden state
    double  iter0_max    = 0.0;  // largest iteration-0 candidate logit (cond row)
    int64_t iter0_argmax = -1;   // and which candidate it was

    // diagnostics
    int64_t nonfinite_logits = 0;  // candidate logits that were NaN/+inf across the whole run
    double  prefill_ms       = 0.0;
    double  lm_ms            = 0.0;  // decode steps only
    double  depth_ms         = 0.0;
    int64_t depth_steps      = 0;  // shared batch calls, independent of take 0's length
    double  host_ms          = 0.0;  // masking, CFG, top-k, sampling
    double  total_ms         = 0.0;
    int64_t lm_steps         = 0;
};

struct MM3ArOptions {
    int64_t  max_frames = 300;
    uint64_t seed       = 42;

    // Semantic-stream sampling knobs (mm3-sample.h). Defaults = the reference
    // recipe, bit-identical to the pre-knob build.
    MM3SamplerKnobs knobs;

    // Ensemble takes: how many independent songs to decode in lockstep from
    // this one prompt (mm3-model.h). 1 is the single-song path and every shape
    // in the loop reduces to what it always was. Take t is seeded `seed + t`.
    // Clamped to the row budget by mm3_ar_plan_takes, which logs when it bites.
    int      n_takes    = 1;

    // Forced-replay parity mode. Both arrays are indexed by ITERATION (so entry 0
    // is the un-emitted iteration 0), and `forced_len` caps the loop.
    const int32_t * forced_semantic = nullptr;  // [forced_len]
    const int32_t * forced_acoustic = nullptr;  // [forced_len, NC]
    int64_t         forced_len      = 0;
    // Soft forcing (semantic-only). < 0 = the hard pin. >= 0: the forced code
    // gets this added to its CFG-guided logit (before temperature) and the
    // normal sampler draws; 0 = free, large = the pin. A -1 entry in
    // forced_semantic is a free iteration under either mode. EOS stays masked
    // inside the forced span, so the length is still the forced length.
    float           force_bias      = -1.0f;

    // Source guide (HOTSTEP-1090): YuE2's Follow-source mix on the semantic
    // draw. The guide holds the RVQ encoder's top-K codes per frame and their
    // log-softmax (`ace-train mm3-codes --guide-k`, Y2GD layout). Each guided
    // iteration mixes p = (1 - s) p_LM + s p_enc over the candidates
    // (yue2_guide_apply), so the LM picks among the source's near-synonyms
    // instead of being pinned to the argmax. Guide frame j is emitted frame j;
    // iteration 0 reads frame 0, the same warm-up duplicate as `.codes` row 0.
    // Length = guide->T frames, EOS masked. Exclusive with forced_semantic.
    const Yue2Guide * guide      = nullptr;
    Yue2GuideParams   guide_p;               // scale, k, outside; the rest unused
    const float *     guide_gate = nullptr;  // [guide->T] multiplies scale; null = 1
    std::string       guide_trace;           // per-frame JSON for take 0; empty = none
    // Probes on guided frames only (HOTSTEP-1090 round 9), both off by default.
    // depth_greedy: the depth decoder takes argmax acoustic codes, like the
    // launder, instead of sampling them.
    // state_fix: bit 1 = the depth decoder's input, bit 2 = emitted stream 0.
    // After the draw of code c, the conditional hidden row is stepped along
    // d(ln p(c))/dh through the head until c is in its top 10 (mm3_state_fix).
    // The LM's feedback and KV cache never see the moved row.
    bool              depth_greedy = false;
    int               state_fix    = 0;
    // guide_product >= 0 replaces the probability mix with a product rule on the
    // candidates only: a * ln p_enc + ln p_LM. < 0 = off (the mix).
    float             guide_product = -1.0f;
    // State blend: [blend_frames, NC+1, H], the emitted hiddens of frame f move
    // blend_amount of the way toward frame f of this block (frames past its end
    // are left alone). Null = off. Needs collect_hiddens.
    const float *     blend        = nullptr;
    int64_t           blend_frames = 0;
    float             blend_amount = 0.0f;

    int64_t dump_iters      = 0;
    bool    collect_hiddens = true;
    // Record per-iteration EOS statistics into MM3ArResult (see there). Costs
    // two O(NCAND) passes per step on the host; off by default.
    bool    eos_trace       = false;
    // Forced replay normally stops at the end of the forced frames. With this
    // on it hands over to free sampling there and runs to EOS or max_frames:
    // the model continues someone else's plan (lever 4b, 2026-09-08: the base
    // finishes an adapter's song; the same KV was built from the forced codes,
    // so nothing is spliced). Needs full forcing (both code streams).
    bool    forced_continue = false;

    // Lyric timestamps: capture the alignment heads' decode attention and emit
    // LRC into MM3ArResult::lrc. Needs the tokenizer to turn the lyric token
    // ids back into the text lrc_align() groups into lines, so both are set
    // together or not at all.
    bool                  want_lrc = false;
    bool                  lrc_eos_only = false;
    bool                  stop_after_first_eos = false;
    const MM3Tokenizer *  tok      = nullptr;

    // Called after every emitted frame. Cheap; used for server-side progress.
    std::function<void(int64_t /*frames*/, int64_t /*max_frames*/)> on_frame;

    /** THE STREAMING HOOK. Called after every emitted frame, once that frame's
     *  hiddens are already appended to MM3ArResult::frame_hiddens — so the
     *  callback may read [0, frames) of the block and act on it.
     *
     *  This is what lets mm3-pipeline.h condition, denoise and vocode a window
     *  while the planner is still working on later frames, instead of waiting
     *  for the whole plan. It runs ON THIS THREAD, between AR steps: it is not
     *  concurrency, it is interleaving, and heavy work inside it directly
     *  delays the next frame.
     *
     *  Return false to abort the run, having written the reason into *err —
     *  distinct from `should_cancel`, which is a user cancel and reports
     *  MM3_ERR_CANCELLED. Unset (the default) is the plain batch path. */
    std::function<bool(int64_t /*frames*/, std::string * /*err*/)> on_frame_ready;

    /** The ensemble form of the hook above. Takes decode in lockstep, but they
     *  do NOT all finish together — a take that hits EOS freezes while the
     *  others carry on — so the callback gets the per-take frame counts rather
     *  than one number. `frames[t]` is how many frames take t has emitted and
     *  may be read from `outs[t].frame_hiddens`.
     *
     *  Set by the K > 1 path only. When both are set the take-aware one wins;
     *  at K == 1 `on_frame_ready` is called instead, so mm3-pipeline.h's
     *  existing single-song interleaving is untouched. */
    std::function<bool(const int64_t * /*frames per take*/, int /*takes*/, std::string * /*err*/)>
        on_takes_frame_ready;

    // Returns true to abort. Polled once per AR iteration — the finest grain
    // this loop has, and the only one that matters: at ~25 frames of real audio
    // per second of wall clock, a 60 s song is thousands of poll points. On
    // abort mm3_ar_plan() returns false with *err == MM3_ERR_CANCELLED so the
    // caller can tell a user cancel from a real failure.
    std::function<bool()> should_cancel;

    // Runtime LM LoRA (mm3-lm-adapter.h). nullptr = base model. Applied to the
    // graph before prefill; the graph invalidates its cached slots only when
    // (adapter, scales) actually changed.
    const MM3LmAdapter * lm_adapter = nullptr;
    MM3LmAdapterScales   lm_adapter_scales = {};
    // Strength along the song (MM3LmAdapterSchedule), a multiplier on
    // lm_adapter_scales.global. Needs a runtime adapter: merge mode has baked
    // the delta into the weights, so there is nothing to turn down.
    MM3LmAdapterSchedule lm_adapter_sched = {};
    // Plan dial, 0..1: how much the adapter plans the song. Below 1 every take
    // gets a base pair (same prompt, delta masked off), guidance runs per
    // model, the semantic draw samples base + plan * (adapter - base), and the
    // drawn code feeds all four rows. The depth decoder still reads the
    // adapter rows, so the adapter sings whatever is planned. 1 (or no runtime
    // adapter) = no base rows: today's graph and today's RNG draws.
    float                lm_adapter_plan  = 1.0f;

    // The adapter whose SOFT-PROMPT halves (artist token, trained KV prefix)
    // apply. Usually the same object as lm_adapter, and deliberately a separate
    // field because in MERGE mode lm_adapter is null — the low-rank delta has
    // been folded into the weights, but a token and a prefix cannot be folded
    // into anything and still have to reach the graph.
    const MM3LmAdapter * lm_soft = nullptr;
};

// The sentinel a cancelled run reports. Compared by value, not by prefix, so a
// genuine error can never be mistaken for a cancel.
#define MM3_ERR_CANCELLED "cancelled"

static MM3LmGraph g_mm3_lm;

// Whether the plan dial actually adds base rows: it needs a runtime delta to
// mask (merge mode has none), and stands down for a soft prompt.
// ponytail: a soft prompt (artist token, KV prefix) would need a per-row
// attention mask to keep off the base rows; no MM3 adapter ships one yet, so
// the dial is ignored for them. Add the mask when one does.
static bool mm3_plan_rows_on(float plan, const MM3LmAdapter * ad, const MM3LmAdapter * soft) {
    const bool has_soft = soft && (soft->has_artist_token() || soft->has_prefix());
    return plan < 1.0f && ad && !has_soft;
}

// ── Runtime LM adapter cache ────────────────────────────────────────────────
//
// One adapter resident at a time, keyed by (path, mtime) — the working set is
// "the adapter being auditioned", not a zoo. Guarded by g_mm3_mutex like every
// other MM3 global (loaded/dropped on the GPU worker; unload paths hold the
// same mutex). The graph's borrowed pointer is cleared BEFORE the buffer is
// freed — mm3_lm_set_adapter also invalidates the cached graphs that bake the
// old tensors' addresses as constants.
static MM3LmAdapter * g_mm3_lm_adapter = nullptr;

static void mm3_lm_adapter_drop() {
    if (g_mm3_lm_adapter) {
        mm3_lm_set_adapter(&g_mm3_lm, nullptr, MM3LmAdapterScales{});
        // The soft-prompt halves are borrowed from the SAME object, so they go
        // at the same moment — an adapter switch must never leave a stale token
        // or prefix installed (artist_token_clear()'s lesson, one file over).
        mm3_lm_set_soft(&g_mm3_lm, nullptr);
        mm3_lm_adapter_free(g_mm3_lm_adapter);
        g_mm3_lm_adapter = nullptr;
    }
}

// ── State fix (MM3ArOptions::state_fix) ─────────────────────────────────────
// Host copy of the head's semantic rows as stored (possibly quantized), so a
// row costs one dequantize instead of a device round trip. ~71 MB at q8_0.
// The hidden row is post-norm and the head has no bias, so logits = W h.
struct MM3HeadRows {
    const ggml_tensor *  t = nullptr;
    std::vector<uint8_t> raw;
    size_t               nb1 = 0;
    int64_t              H   = 0;
};
static MM3HeadRows g_mm3_head_rows;

static bool mm3_head_rows_load(const MM3Model & m, std::string * err) {
    const ggml_tensor * w = m.lm.output;
    if (g_mm3_head_rows.t == w) {
        return true;
    }
    if (!w || (w->type != GGML_TYPE_F32 && !ggml_get_type_traits(w->type)->to_float)) {
        if (err) *err = "state fix: the LM head type cannot be read on the host";
        return false;
    }
    const int64_t OFF = (int64_t) m.lm_cfg.semantic_vocab_offset, SV = (int64_t) m.lm_cfg.semantic_vocab_size;
    g_mm3_head_rows.raw.resize((size_t) SV * w->nb[1]);
    ggml_backend_tensor_get(w, g_mm3_head_rows.raw.data(), (size_t) OFF * w->nb[1], (size_t) SV * w->nb[1]);
    g_mm3_head_rows.t   = w;
    g_mm3_head_rows.nb1 = w->nb[1];
    g_mm3_head_rows.H   = w->ne[0];
    return true;
}

static void mm3_head_row(int64_t code, float * dst) {
    const MM3HeadRows & r = g_mm3_head_rows;
    const uint8_t *     p = r.raw.data() + (size_t) code * r.nb1;
    if (r.t->type == GGML_TYPE_F32) {
        memcpy(dst, p, (size_t) r.H * sizeof(float));
    } else {
        ggml_get_type_traits(r.t->type)->to_float(p, dst, r.H);
    }
}

// Move the conditional hidden row h [H] until semantic code c is in the top
// `rank` of its own logits. Works over S = the row's top 50 plus c, the codes
// that carry the mass. Logits are linear in h, so each step is exact along its
// direction d = W[c] - sum_{j in S} p_j W[j] (the gradient of ln p(c) on S),
// sized to land c just above the current rank-th competitor. Returns the steps
// used (0 = already there) or -1 when the cap ran out; *rel_dh = |dh| / |h|.
static int mm3_state_fix(float * h, int64_t H, const float * sem_logits, int64_t SV, int32_t c, int rank,
                         int max_steps, float * rel_dh) {
    static std::vector<int32_t> S;
    static std::vector<float>   W, l, d, r, tmp;
    S.resize((size_t) SV);
    for (int64_t j = 0; j < SV; j++) S[(size_t) j] = (int32_t) j;
    const int64_t top = std::min<int64_t>(50, SV);
    std::nth_element(S.begin(), S.begin() + top, S.end(),
                     [&](int32_t a, int32_t b) { return sem_logits[a] > sem_logits[b]; });
    S.resize((size_t) top);
    if (std::find(S.begin(), S.end(), c) == S.end()) S.push_back(c);
    const size_t n  = S.size();
    const size_t ci = (size_t) (std::find(S.begin(), S.end(), c) - S.begin());
    W.resize(n * (size_t) H);
    l.resize(n);
    r.resize(n);
    d.resize((size_t) H);
    for (size_t j = 0; j < n; j++) {
        mm3_head_row(S[j], W.data() + j * (size_t) H);
        l[j] = sem_logits[S[j]];
    }
    double h2 = 0.0, dh2 = 0.0;
    for (int64_t i = 0; i < H; i++) h2 += (double) h[i] * h[i];
    std::vector<float> dh((size_t) H, 0.0f);
    int                steps = 0;
    for (;; steps++) {
        // The rank-th best competitor, and whether c already beats it.
        tmp.clear();
        size_t thr_j = 0;
        for (size_t j = 0; j < n; j++) if (j != ci) tmp.push_back(l[j]);
        const size_t k = std::min<size_t>((size_t) rank, tmp.size()) - 1;
        std::nth_element(tmp.begin(), tmp.begin() + (ptrdiff_t) k, tmp.end(), std::greater<float>());
        const float thr = tmp[k];
        for (size_t j = 0; j < n; j++) if (j != ci && l[j] == thr) { thr_j = j; break; }
        if (l[ci] >= thr) break;
        if (steps >= max_steps) { steps = -1; break; }
        float mx = -INFINITY;
        for (size_t j = 0; j < n; j++) mx = std::max(mx, l[j]);
        double z = 0.0;
        for (size_t j = 0; j < n; j++) z += std::exp((double) (l[j] - mx));
        for (int64_t i = 0; i < H; i++) d[(size_t) i] = W[ci * (size_t) H + (size_t) i];
        for (size_t j = 0; j < n; j++) {
            const float pj = (float) (std::exp((double) (l[j] - mx)) / z);
            const float * wj = W.data() + j * (size_t) H;
            for (int64_t i = 0; i < H; i++) d[(size_t) i] -= pj * wj[i];
        }
        for (size_t j = 0; j < n; j++) {
            const float * wj = W.data() + j * (size_t) H;
            double        s  = 0.0;
            for (int64_t i = 0; i < H; i++) s += (double) wj[i] * d[(size_t) i];
            r[j] = (float) s;
        }
        const float rel = r[ci] - r[thr_j];
        if (!(rel > 1e-6f)) { steps = -1; break; }
        const float eta = (thr - l[ci] + 0.25f) / rel;
        for (int64_t i = 0; i < H; i++) {
            h[i] += eta * d[(size_t) i];
            dh[(size_t) i] += eta * d[(size_t) i];
        }
        for (size_t j = 0; j < n; j++) l[j] += eta * r[j];
    }
    for (int64_t i = 0; i < H; i++) dh2 += (double) dh[(size_t) i] * dh[(size_t) i];
    *rel_dh = h2 > 0.0 ? (float) std::sqrt(dh2 / h2) : 0.0f;
    return steps;
}

// Plan K songs at once. `cond_ids` / `uncond_ids` are the two prefill rows,
// shared by every take — takes diverge only through sampling.
//
// `outs` is an array of `opt.n_takes` results. Take t is seeded `opt.seed + t`
// and draws from its OWN generator, so the takes are genuinely different songs,
// not one song rendered K times. They decode in LOCKSTEP: every take produces
// frame N in the same iteration, which is what lets the streaming hook dispatch
// window k for all of them together.
//
// At n_takes == 1 every shape, every buffer offset and every RNG draw below is
// what the single-song loop always did — that reduction is the contract, and
// `mm3-ar-ensemble.mjs` checks it by rendering the same seed both ways and
// comparing codes.
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_ar_plan_takes(const MM3Model & m, const int32_t * cond_ids, const int32_t * uncond_ids,
                              int64_t n_prompt, const MM3ArOptions & opt, MM3ArResult * outs, std::string * err) {
    const MM3LmConfig & c   = m.lm_cfg;
    const int64_t       H   = (int64_t) c.embedding_length;
    const int64_t       V   = (int64_t) c.vocab_size;
    const int64_t       SV  = (int64_t) c.semantic_vocab_size;
    const int64_t       NC  = (int64_t) c.num_codebooks - 1;
    const int64_t       AV  = (int64_t) c.acoustic_vocab_size;
    const int64_t       OFF = (int64_t) c.semantic_vocab_offset;
    const int64_t       EOS = (int64_t) c.eos_audio;
    const float         CFG = mm3_ar_cfg_scale(c);
    // 1 for a guidance-distilled checkpoint (mm3-model.h): the LM and depth
    // graphs then evaluate the conditional row only, and every "uncond" pointer
    // below aliases the conditional one so the blend stays the identity it
    // already is at CFG 1.0.
    const int           P    = mm3_cfg_rows(c);
    const int           TOPK = c.ar_top_k > 0 ? (int) c.ar_top_k : 50;
    // Ensemble takes, clamped to what ggml's matrix-vector kernels can serve
    // (mm3-model.h). Clamping rather than failing: a caller asking for 8 takes
    // on a CFG-pair checkpoint gets 4 and a log line, not an error.
    const float         plan     = opt.lm_adapter_plan;
    const bool          plan_mix = mm3_plan_rows_on(plan, opt.lm_adapter, opt.lm_soft);
    const int           K        = mm3_clamp_takes(c, opt.n_takes, plan_mix);
    if (K != opt.n_takes && opt.n_takes > 1) {
        fprintf(stderr, "[MM3-AR] %d takes requested; the row budget allows %d%s — rendering %d\n", opt.n_takes,
                mm3_max_takes(c, plan_mix), plan_mix ? " with the plan dial's base rows" : "", K);
    }

    if (n_prompt <= 0) {
        if (err) {
            *err = "the prompt tokenised to zero tokens";
        }
        return false;
    }
    if (c.max_prompt_tokens > 0 && n_prompt > (int64_t) c.max_prompt_tokens) {
        if (err) {
            *err = "the prompt is " + std::to_string((long long) n_prompt) + " tokens; the checkpoint's limit is " +
                   std::to_string(c.max_prompt_tokens);
        }
        return false;
    }
    if (EOS < 0 || EOS >= V || OFF + SV > V) {
        if (err) {
            *err = "the LM vocabulary metadata does not cover the semantic range and EOS";
        }
        return false;
    }

    int64_t max_frames = opt.max_frames;
    if (max_frames <= 0) {
        if (err) {
            *err = "max_frames must be positive";
        }
        return false;
    }
    if (c.max_audio_frames > 0 && max_frames > (int64_t) c.max_audio_frames) {
        max_frames = (int64_t) c.max_audio_frames;
    }
    const bool forced = opt.forced_semantic != nullptr;
    // Forced replay feeds ONE fixture's code sequence, which has no meaning
    // spread across independently-sampling takes. Refuse rather than silently
    // apply it to take 0 alone.
    if (K > 1 && forced) {
        if (err) {
            *err = "forced replay is a single-take path; set n_takes = 1";
        }
        return false;
    }
    // The per-iteration dumps ARE allowed at K > 1, and they capture TAKE 0.
    // That is deliberate: comparing take 0's forward against the reference
    // fixture at each take count is how the ensemble proves that widening the
    // batch does not disturb the math (check-mm3-ensemble.mjs).
    // SEMANTIC-ONLY forcing: forced_semantic without forced_acoustic. The
    // semantic code (content and structure -- what happens when) is pinned to
    // the real audio, while the depth decoder SAMPLES the 7 acoustic codebooks
    // (timbre -- what it sounds like) from its own distribution.
    //
    // This exists because fully-aligned conditioning was measured to be the
    // wrong training signal for a style adapter: it describes the target so
    // completely that the base DiT can render it without the LoRA learning
    // anything, and the LoRA then contributes nothing at inference, where
    // conditioning comes from the LM instead. Sampling the acoustic half keeps
    // the conditioning anchored in time but leaves TIMBRE unspecified, so the
    // adapter has to store it. It also keeps the acoustic codes on the LM's own
    // manifold, which is what inference will actually feed the DiT.
    const bool forced_ac_given = forced && opt.forced_acoustic != nullptr;
    if (forced) {
        if (opt.forced_len <= 0) {
            if (err) {
                *err = "forced replay needs forced_semantic and a positive forced_len";
            }
            return false;
        }
        if (opt.forced_continue && !forced_ac_given) {
            if (err) {
                *err = "forced_continue needs both forced_semantic and forced_acoustic";
            }
            return false;
        }
        if (!opt.forced_continue && opt.forced_len - 1 < max_frames) {
            max_frames = opt.forced_len - 1;  // entry 0 is the un-emitted iteration
        }
        if (max_frames <= 0) {
            if (err) {
                *err = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
            }
            return false;
        }
    }
    const bool guided = opt.guide != nullptr;
    if (guided) {
        if (forced) {
            if (err) {
                *err = "the source guide and forced_semantic are mutually exclusive";
            }
            return false;
        }
        if (opt.guide->T <= 0 || opt.guide->K <= 0 ||
            opt.guide->ids.size() != (size_t) (opt.guide->T * opt.guide->K)) {
            if (err) {
                *err = "the source guide is empty or malformed";
            }
            return false;
        }
        max_frames = std::min(max_frames, opt.guide->T);
        if (opt.state_fix && !mm3_head_rows_load(m, err)) {
            return false;
        }
    } else if (opt.state_fix || opt.depth_greedy) {
        if (err) {
            *err = "state_fix and depth_greedy act on guided frames and need a source guide";
        }
        return false;
    }
    std::vector<float> sf_rel;
    std::vector<int>   sf_steps;
    // Soft or free frames resample the semantic code, so a pinned acoustic
    // replay for that frame would describe a code that was not drawn.
    bool soft_force = forced && opt.force_bias >= 0.0f;
    for (int64_t i = 0; forced && i < opt.forced_len; i++) {
        const int32_t fs = opt.forced_semantic[i];
        if (fs < -1 || (int64_t) fs >= SV) {
            if (err) {
                *err = "forced semantic code " + std::to_string(fs) + " at iteration " + std::to_string((long long) i) +
                       " is outside [0, " + std::to_string((long long) SV) + ") and is not -1 (free)";
            }
            return false;
        }
        soft_force = soft_force || fs == -1;
    }
    if (soft_force && forced_ac_given) {
        if (err) {
            *err = "lm_force_bias and free (-1) frames need semantic-only forcing (no forced_acoustic)";
        }
        return false;
    }
    // Logit scale under soft forcing, take 0, logged once at the end: the
    // guided top-1 minus top-2 margin, the gap from top-1 down to the forced
    // code, and how often the draw agreed with the source.
    // fb_in50: forced frames whose source code survived the conditional
    // top-k (the LM's own candidates), read before the bias re-admits it.
    std::vector<float> fb_margin, fb_gap;
    int64_t            fb_agree = 0, fb_biased = 0, fb_forced = 0, fb_in50 = 0;
    // Guide trace, take 0, one entry per guided iteration:
    //   gap       LM top-1 minus the guide's top-1, CFG'd logits before any filter
    //   mass      LM probability on the candidates, inside its own top-k
    //   overlap   1 when any candidate survived the LM's top-k
    //   rank      the drawn code's place in the candidate list, -1 = outside
    //   surprise  -ln p of the drawn code under the LM's full CFG'd row (nats)
    //   scale     the mix scale actually used (gate applied)
    std::vector<float>   gd_gap, gd_mass, gd_surprise, gd_scale;
    std::vector<int32_t> gd_overlap, gd_rank;
    std::vector<float>   gd_pre;
    int64_t              gd_fallback = 0;  // frames where the LM's top-k dropped every candidate

    // KV budget: prompt + one iteration per emitted frame + the un-emitted first
    // one, plus slack so the last decode step never sits exactly on the boundary.
    // Must be set BEFORE prepare: the graph builder decides per layer
    // whether to materialise attention, and a cached graph is reused.
    //
    // LRC capture is a POST-HOC REPLAY by default (mm3_lm_lrc_replay): the
    // decode runs pure flash — bit-identical to a no-LRC render — and the
    // alignment attention is recomputed afterwards from the sampled codes, at
    // a few seconds per song instead of ~+41 % on every decode step.
    // MM3_LRC_LIVE=1 restores the old all-manual live capture — the
    // validation path (same codes through both must give the same LRC) and
    // the fallback if replay alignment ever misbehaves.
    static const bool lrc_live = [] {
        const char * e = std::getenv("MM3_LRC_LIVE");
        return e && e[0] && e[0] != '0';
    }();
    {
        // Live capture reads row 0's attention, i.e. take 0's. The default
        // post-hoc replay has no such limit and runs per take at the tail.
        const bool want = opt.want_lrc && opt.tok != nullptr && lrc_live && K == 1;
        if (g_mm3_lm.align_capture != want) {
            g_mm3_lm.align_capture = want;
            mm3_lm_free(&g_mm3_lm);   // graphs encode the choice — rebuild them
        }
    }
    // Head slice: production always wants it (the sampler reads nothing
    // outside the EOS+semantic span). Baked into the cached graphs, so the
    // first generation after boot pays one slot rebuild; the mm3-lm-probe
    // parity tool builds its own MM3LmGraph and keeps the full head.
    if (!g_mm3_lm.head_slice) {
        g_mm3_lm.head_slice = true;
        mm3_lm_free_slot(&g_mm3_lm.prefill);
        mm3_lm_free_slot(&g_mm3_lm.decode);
        mm3_lm_free_slot(&g_mm3_lm.replay);
    }
    // Plan dial (mm3_plan_rows_on, settled above with K).
    if (plan < 1.0f && !plan_mix) {
        fprintf(stderr, "[MM3-AR] plan dial %.2f ignored: %s\n", (double) plan,
                !opt.lm_adapter ? "no runtime LM adapter (none loaded, or merge mode)"
                                : "the adapter carries a soft prompt (artist token or KV prefix)");
    }
    // Rows of the adapter half; the base half starts here.
    const size_t KP = (size_t) K * (size_t) P;
    // Both graph sets bake rows() into their shapes, so K has to be settled
    // before either is prepared. Kept in lockstep on purpose: the depth decoder
    // reads the LM's hidden rows straight out of the same buffer.
    mm3_lm_set_takes(&g_mm3_lm, K, plan_mix);
    if (plan_mix) {
        fprintf(stderr, "[MM3-AR] plan dial %.2f: base rows active (%d take%s, %zu adapter + %zu base LM rows)\n",
                (double) plan, K, K == 1 ? "" : "s", KP, KP);
    }
    mm3_depth_set_takes(&g_mm3_depth, K);
    // BEFORE prepare: a trained prefix occupies real KV columns, so the cache
    // has to be sized with them in it (mm3_lm_prepare adds them to the ask).
    mm3_lm_set_soft(&g_mm3_lm, opt.lm_soft);
    if (!mm3_lm_prepare(m, &g_mm3_lm, n_prompt + max_frames + 2, err)) {
        return false;
    }
    // What the graphs will actually compute per step: [hlo, hlo+hn) of the
    // vocabulary. Mirrors the slot builder's choice exactly (same helper,
    // same flag), so the readback buffers below are sized to match.
    int64_t hlo = 0, hn = V;
    if (!g_mm3_lm.head_slice || !mm3_lm_head_slice_span(c, &hlo, &hn)) {
        hlo = 0;
        hn  = V;
    }

    // ── Alignment probe (MM3_ALIGN_DUMP=1) ───────────────────────────────────
    //
    // Discovery instrumentation for lyric timestamps. MM3's flow DiT has no
    // cross-attention and never sees lyrics, so ACE's DiT-cross-attention
    // technique has no analogue here. The one place lyric tokens and audio
    // frames coexist is THIS loop: every decode step attends over the whole
    // prompt, so slicing the lyric columns out of each step's attention builds
    // a [head][lyric_token][frame] matrix directly — the exact shape lrc_align()
    // already consumes. Frames are 25 fps, so frame f is 40 ms with no
    // latent-rate conversion (cleaner than the ACE path).
    //
    // WHICH heads carry monotonic alignment is unknown for MM3 (ACE's are
    // hardcoded per architecture and were found empirically), so this dumps ALL
    // of them for offline scoring. Discovery only — never on in a normal run.
    const bool align_dump = g_mm3_lm.dump_attn;
    int64_t    lyr0 = -1, lyr1 = -1;
    // The span is needed by ALL consumers — the discovery dump, live capture,
    // and the post-hoc replay — so it is located whenever any of them will run.
    if (align_dump || g_mm3_lm.align_capture || (opt.want_lrc && opt.tok != nullptr)) {
        for (int64_t i = 0; i < n_prompt; i++) {
            if (cond_ids[i] == (int32_t) m.lm_cfg.tok_lyrics_start) lyr0 = i + 1;
            if (cond_ids[i] == (int32_t) m.lm_cfg.tok_lyrics_end)   lyr1 = i;
        }
        if (lyr0 < 0 || lyr1 <= lyr0) {
            fprintf(stderr, "[MM3-Align] No lyric span in the prompt (instrumental?) — alignment disabled\n");
        } else {
            fprintf(stderr, "[MM3-Align] Lyric span = prompt tokens [%lld, %lld) = %lld tokens\n",
                    (long long) lyr0, (long long) lyr1, (long long) (lyr1 - lyr0));
        }
    }
    std::vector<float> align_acc;    // [layer][head][token] per frame, appended
    int64_t            align_frames = 0;

    // ── Production alignment capture ─────────────────────────────────────────
    //
    // Same signal as the probe, but only the three heads that carry it, and
    // accumulated straight into lrc_align()'s [head][token][frame] layout so no
    // transpose is needed at the end. Off for instrumentals (nothing to align)
    // and while the discovery dump runs (that owns the graph).
    const bool    lrc_on  = g_mm3_lm.align_capture && !align_dump && lyr0 >= 0 && lyr1 > lyr0;
    const int64_t lrc_tok = lrc_on ? (lyr1 - lyr0) : 0;
    std::vector<std::vector<float>> lrc_rows;   // [head*token] -> per-frame series
    if (lrc_on) {
        lrc_rows.assign((size_t) (MM3_ALIGN_N_HEADS * lrc_tok), std::vector<float>());
        for (auto & r : lrc_rows) {
            r.reserve((size_t) max_frames);
        }
        fprintf(stderr, "[MM3-LRC] Capturing %d head(s) over %lld lyric tokens\n", MM3_ALIGN_N_HEADS,
                (long long) lrc_tok);
    }


    for (int t = 0; t < K; t++) {
        outs[t]             = MM3ArResult{};
        outs[t].hidden_dim  = H;
        outs[t].n_codebooks = NC + 1;
        outs[t].sem_vocab   = SV;
        outs[t].seed        = opt.seed + (uint64_t) t;
    }
    // Take 0's result also carries the SHARED stage timings and diagnostic
    // counters — the LM and depth work is one batched pass, so attributing it
    // per take would be double counting. `out` keeps the name the accumulators
    // below already use.
    MM3ArResult * out = &outs[0];

    // Cut to the batch ceiling rather than to what this run uses: the readback
    // helpers size their transfers from the graph, so a ceiling-sized buffer
    // can never be the thing that overruns.
    // The plan dial's base rows fit too: mm3_max_takes halves K for them.
    std::vector<float> hidden((size_t) (H * MM3_MAX_BATCH_ROWS));
    std::vector<float> logits((size_t) (hn * MM3_MAX_BATCH_ROWS));
    std::vector<float> feedback((size_t) H);

    // Row of take t: conditional, and unconditional. At P == 1 (a
    // guidance-distilled checkpoint) the graphs never build an unconditional
    // row, so it ALIASES the conditional one and the CFG blend stays the
    // identity it already is at scale 1.0. Every read of `hidden` / `logits`
    // below goes through these, which is what makes the body indifferent to
    // both the row count and the take count.
    const auto row_c = [&](int t) { return (size_t) (t * P); };
    const auto row_u = [&](int t) { return (size_t) (P > 1 ? t * P + 1 : t * P); };

    // The single-take parity surfaces — prefill_hidden and the per-iteration
    // dumps — were written against a literal [2, H] block. At P == 1 there is
    // no second row to give them, so mirror the conditional one into place and
    // keep them meaningful. Only ever needed at K == 1, which is the only K
    // those surfaces are reachable at.
    // Not with base rows: row 1 is then the base model's conditional row.
    const auto mirror_uncond = [&]() {
        if (P > 1 || K != 1 || plan_mix) {
            return;
        }
        memcpy(hidden.data() + H, hidden.data(), (size_t) H * sizeof(float));
        memcpy(logits.data() + hn, logits.data(), (size_t) hn * sizeof(float));
    };

    // The adapter strength schedule. Takes decode in lockstep, so one shared
    // multiplier per iteration is exact for every take. Iteration `it` is
    // song time it / 25 s; the decode at the end of iteration it makes the
    // logits iteration it + 1 samples from, so that is the time it is set for.
    const bool sched_on = opt.lm_adapter && opt.lm_adapter_sched.active();
    if (opt.lm_adapter_sched.active() && !opt.lm_adapter) {
        fprintf(stderr, "[MM3-AR] adapter schedule ignored: no runtime LM adapter (none loaded, or merge mode)\n");
    }
    float      sched_cur   = -1.0f;
    const auto sched_apply = [&](int64_t it) {
        const float mult = opt.lm_adapter_sched.at((float) it / 25.0f);
        if (mult == sched_cur) {
            return;
        }
        sched_cur             = mult;
        MM3LmAdapterScales sc = opt.lm_adapter_scales;
        sc.global *= mult;
        // Zero is the base itself: no delta in the graph, and no delta cost.
        mm3_lm_set_adapter(&g_mm3_lm, mult > 0.0f ? opt.lm_adapter : nullptr, sc);
        fprintf(stderr, "[MM3-AR] adapter schedule: x%.2f from %.1f s\n", mult, (double) it / 25.0);
    };

    const auto t_start = std::chrono::steady_clock::now();
    {
        const auto t0 = std::chrono::steady_clock::now();
        if (sched_on) {
            sched_apply(0);
        } else {
            mm3_lm_set_adapter(&g_mm3_lm, opt.lm_adapter, opt.lm_adapter_scales);
        }
        if (!mm3_lm_prefill(m, &g_mm3_lm, cond_ids, uncond_ids, n_prompt, hidden.data(), logits.data(), err)) {
            return false;
        }
        out->prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    mirror_uncond();
    // Every take prefills the SAME prompt, so their rows are identical here —
    // they only diverge once sampling starts. Recorded per take anyway so the
    // parity tooling can read outs[t] without knowing that.
    for (int t = 0; t < K; t++) {
        const float * r0 = hidden.data() + row_c(t) * (size_t) H;
        const float * r1 = hidden.data() + row_u(t) * (size_t) H;
        outs[t].prefill_hidden.assign(r0, r0 + H);
        outs[t].prefill_hidden.insert(outs[t].prefill_hidden.end(), r1, r1 + H);
        double ps = 0.0;
        for (int64_t i = 0; i < H; i++) {
            ps += (double) r0[i];
        }
        outs[t].prefill_sum = ps;
    }


    // Candidate layout: index 0 is EOS, indices 1..SV are semantic codes 0..SV-1.
    // Everything else in the 200000-entry vocabulary is masked and never gathered.
    const int64_t      NCAND = SV + 1;
    std::vector<float> cand_cond((size_t) NCAND);
    std::vector<float> cand_unc((size_t) NCAND);
    std::vector<float> cand_guided((size_t) NCAND);
    std::vector<float> sel_scratch;
    std::vector<float> samp_scratch;

    // One generator per take, seeded seed + t. This is the entire reason the
    // takes are different songs: the batched forward is shared, the DRAW is not.
    std::vector<std::mt19937_64> rngs;
    rngs.reserve((size_t) K);
    for (int t = 0; t < K; t++) {
        rngs.emplace_back(opt.seed + (uint64_t) t);
    }
    // Take-major, matching what mm3_lm_decode and the depth decoder expect.
    std::vector<int32_t>       ac_rows((size_t) (K * NC));
    std::vector<int32_t>       sem_tok((size_t) K);
    std::vector<int32_t>       sem_code((size_t) K);
    std::vector<int64_t>       frames_done((size_t) K, 0);
    std::vector<char>          active((size_t) K, 1);
    int                       first_eos_take = -1;
    std::vector<MM3DepthFrame> frames((size_t) K);
    std::vector<float>              depth_in;              // state fix: the depth decoder's copy of `hidden`
    std::vector<std::vector<float>> fixed((size_t) K);     // state fix: each take's moved conditional row
    std::vector<char>               fixed_ok((size_t) K, 0);

    for (int t = 0; t < K; t++) {
        outs[t].semantic_all.reserve((size_t) (max_frames + 1));
        outs[t].acoustic_all.reserve((size_t) ((max_frames + 1) * NC));
        if (opt.collect_hiddens) {
            // Reserved WHOLE up front: the streaming hook reads this block while
            // the planner is still appending to it, and a reallocation under a
            // reader is silent garbage audio. mm3-pipeline.h relies on it.
            outs[t].frame_hiddens.reserve((size_t) (max_frames * (NC + 1) * H));
        }
    }

    for (int64_t it = 0; it <= max_frames; it++) {
        if (forced && it >= opt.forced_len && !opt.forced_continue) {
            break;
        }
        // Under forced_continue the iterations past the forced frames sample freely.
        const bool    forced_it = forced && it < opt.forced_len;
        const int32_t fs_it     = forced_it ? opt.forced_semantic[it] : -1;
        // Biased or free: sampled normally (rep penalty, knobs) with EOS masked.
        const bool    soft_it   = forced_it && (opt.force_bias >= 0.0f || fs_it < 0);
        // Guide frame: iteration 0 is un-emitted and reads frame 0, like the
        // warm-up row of a .codes file; iteration it >= 1 emits frame it - 1.
        const int64_t gf        = it > 0 ? it - 1 : 0;
        const bool    guide_it  = guided && gf < opt.guide->T;
        const auto t_host0 = std::chrono::steady_clock::now();

        // ── sample this iteration's semantic code, once per take ──
        //
        // The forward that produced `logits` was ONE batched pass; everything
        // from here to the depth call is per-take host work on its own rows and
        // its own generator. That split is the whole ensemble: shared compute,
        // independent draws.
        for (int t = 0; t < K; t++) {
            if (!active[t]) {
                continue;
            }
            // ── gather the candidate logits, both of this take's rows ──
            // Rows are indexed relative to hlo — 0 for the full head, the span
            // start under the head slice.
            const float * lrow_c = logits.data() + row_c(t) * (size_t) hn;
            const float * lrow_u = logits.data() + row_u(t) * (size_t) hn;
            auto          fix    = [&](float x) -> float {
                if (std::isnan(x) || (std::isinf(x) && x > 0.0f)) {
                    out->nonfinite_logits++;
                    return -INFINITY;
                }
                return x;
            };
            cand_cond[0] = fix(lrow_c[EOS - hlo]);
            cand_unc[0]  = fix(lrow_u[EOS - hlo]);
            for (int64_t j = 0; j < SV; j++) {
                cand_cond[(size_t) (j + 1)] = fix(lrow_c[OFF - hlo + j]);
                cand_unc[(size_t) (j + 1)]  = fix(lrow_u[OFF - hlo + j]);
            }
            // Plan dial: mix each row with the base model's, base + plan *
            // (adapter - base). Guidance is linear, so mixing cond and uncond
            // here IS mixing the two guided rows, and the conditional top-k
            // below then ranks by the mixed conditional row: the base's own
            // filter at 0, the adapter's at 1. -inf on either side stays -inf.
            if (plan_mix) {
                const float * brow_c = logits.data() + (row_c(t) + KP) * (size_t) hn;
                const float * brow_u = logits.data() + (row_u(t) + KP) * (size_t) hn;
                // At 0 the base row is the answer outright: the adapter has no
                // vote, so an adapter -inf (or a clamped NaN) cannot mask a code
                // the base allows.
                const auto    mix    = [&](float a, float b) -> float {
                    if (plan == 0.0f) {
                        return b;
                    }
                    return (std::isinf(a) || std::isinf(b)) ? -INFINITY : b + plan * (a - b);
                };
                cand_cond[0] = mix(cand_cond[0], fix(brow_c[EOS - hlo]));
                cand_unc[0]  = mix(cand_unc[0], fix(brow_u[EOS - hlo]));
                for (int64_t j = 0; j < SV; j++) {
                    cand_cond[(size_t) (j + 1)] = mix(cand_cond[(size_t) (j + 1)], fix(brow_c[OFF - hlo + j]));
                    cand_unc[(size_t) (j + 1)]  = mix(cand_unc[(size_t) (j + 1)], fix(brow_u[OFF - hlo + j]));
                }
            }

            if (it == 0) {
                // Before CFG, before top-k, before the draw: the raw conditional
                // candidate row. Every take prefills the same prompt, so at
                // iteration 0 these MUST agree across takes — and if they agree
                // across take counts too, the batch width is not perturbing the
                // forward at all.
                double  best = -INFINITY;
                int64_t bi   = -1;
                for (int64_t i = 0; i < NCAND; i++) {
                    if ((double) cand_cond[(size_t) i] > best) {
                        best = (double) cand_cond[(size_t) i];
                        bi   = i;
                    }
                }
                outs[t].iter0_max    = best;
                outs[t].iter0_argmax = bi;
            }

            // ── CFG, then the first top-k filter (ranked by the CONDITIONAL row) ──
            for (int64_t i = 0; i < NCAND; i++) {
                const float u           = cand_unc[(size_t) i];
                cand_guided[(size_t) i] = u + (cand_cond[(size_t) i] - u) * CFG;
            }
            // Read before the conditional top-k below can mask it out.
            const float fs_guided = soft_it && fs_it >= 0 ? cand_guided[(size_t) (1 + fs_it)] : 0.0f;
            if (guide_it) {
                gd_pre = cand_guided;  // per take: the fallback below reads it too
            }
            {
                int64_t k = TOPK < NCAND ? (int64_t) TOPK : NCAND;
                if (k < 1) {
                    k = 1;
                }
                float threshold = -INFINITY;
                if (k < NCAND) {
                    sel_scratch = cand_cond;
                    std::nth_element(sel_scratch.begin(), sel_scratch.begin() + (size_t) (k - 1), sel_scratch.end(),
                                     std::greater<float>());
                    threshold = sel_scratch[(size_t) (k - 1)];
                }
                for (int64_t i = 0; i < NCAND; i++) {
                    // Strictly less: ties at the threshold survive, which is why the
                    // reference dumps carry 50..55 finite entries and not exactly 50.
                    if (cand_cond[(size_t) i] < threshold) {
                        cand_guided[(size_t) i] = -INFINITY;
                    }
                }
                if (opt.eos_trace && t == 0) {
                    // Raw conditional softmax at EOS, and EOS's rank in that row.
                    double  cmax = -INFINITY;
                    int64_t rank = 0;
                    for (int64_t i = 0; i < NCAND; i++) {
                        const double x = (double) cand_cond[(size_t) i];
                        if (x > cmax) {
                            cmax = x;
                        }
                        if (i > 0 && x > (double) cand_cond[0]) {
                            rank++;
                        }
                    }
                    double csum = 0.0;
                    for (int64_t i = 0; i < NCAND; i++) {
                        const double x = (double) cand_cond[(size_t) i];
                        if (std::isfinite(x)) {
                            csum += std::exp(x - cmax);
                        }
                    }
                    const float p_cond = std::isfinite((double) cand_cond[0]) && csum > 0.0
                                             ? (float) (std::exp((double) cand_cond[0] - cmax) / csum)
                                             : 0.0f;
                    // What the draw would see: the second top-k over the guided row
                    // (mm3_sample_top_k), softmax over its survivors.
                    float p_final = 0.0f;
                    {
                        int64_t k2 = TOPK > 0 ? (int64_t) TOPK : NCAND;
                        if (k2 > NCAND) {
                            k2 = NCAND;
                        }
                        float thr2 = -INFINITY;
                        if (k2 < NCAND) {
                            sel_scratch = cand_guided;
                            std::nth_element(sel_scratch.begin(), sel_scratch.begin() + (size_t) (k2 - 1),
                                             sel_scratch.end(), std::greater<float>());
                            thr2 = sel_scratch[(size_t) (k2 - 1)];
                        }
                        double gmax = -INFINITY;
                        for (int64_t i = 0; i < NCAND; i++) {
                            const float g = cand_guided[(size_t) i];
                            if (g >= thr2 && std::isfinite(g) && (double) g > gmax) {
                                gmax = (double) g;
                            }
                        }
                        double gsum = 0.0;
                        for (int64_t i = 0; i < NCAND; i++) {
                            const float g = cand_guided[(size_t) i];
                            if (g >= thr2 && std::isfinite(g)) {
                                gsum += std::exp((double) g - gmax);
                            }
                        }
                        const float g0 = cand_guided[0];
                        if (std::isfinite(g0) && g0 >= thr2 && gsum > 0.0) {
                            p_final = (float) (std::exp((double) g0 - gmax) / gsum);
                        }
                    }
                    outs[0].eos_p_cond.push_back(p_cond);
                    outs[0].eos_p_final.push_back(p_final);
                    outs[0].eos_rank.push_back((int32_t) rank);
                    outs[0].eos_topk_ok.push_back(std::isfinite(cand_guided[0]) ? 1 : 0);
                }
            }

            // ── soft forcing: pull toward the source code, never end early ──
            if (soft_it) {
                if (t == 0) {
                    float top1 = -INFINITY, top2 = -INFINITY;
                    for (int64_t i = 1; i < NCAND; i++) {
                        const float g = cand_guided[(size_t) i];
                        if (g > top1) {
                            top2 = top1;
                            top1 = g;
                        } else if (g > top2) {
                            top2 = g;
                        }
                    }
                    if (std::isfinite(top2)) {
                        fb_margin.push_back(top1 - top2);
                    }
                    if (fs_it >= 0 && std::isfinite(fs_guided)) {
                        fb_gap.push_back(top1 - fs_guided);
                    }
                    if (fs_it >= 0) {
                        fb_forced++;
                        fb_in50 += std::isfinite(cand_guided[(size_t) (1 + fs_it)]) ? 1 : 0;
                    }
                }
                // A positive bias re-admits the source code if the conditional
                // top-k dropped it; at 0 the frame is left exactly free.
                if (fs_it >= 0 && opt.force_bias > 0.0f) {
                    cand_guided[(size_t) (1 + fs_it)] = fs_guided + opt.force_bias;
                }
                cand_guided[0] = -INFINITY;
            }

            // ── source guide: let the LM choose among the encoder's candidates ──
            // On the row the conditional top-k has already filtered; the mix
            // re-admits any candidate that filter dropped.
            int64_t         guide_pick = -1;
            const int32_t * gids       = nullptr;
            int64_t         gk         = 0;
            if (guide_it) {
                const Yue2Guide & g = *opt.guide;
                gk                  = opt.guide_p.k > 0 ? std::min<int64_t>(opt.guide_p.k, g.K) : g.K;
                gids                = g.ids.data() + (size_t) (gf * g.K);
                Yue2GuideParams gp  = opt.guide_p;
                if (opt.guide_gate) {
                    gp.scale *= opt.guide_gate[gf];
                }
                if (t == 0) {
                    double mx = -INFINITY;
                    for (int64_t i = 0; i < NCAND; i++) {
                        mx = std::max(mx, (double) cand_guided[(size_t) i]);
                    }
                    double z = 0.0, zc = 0.0;
                    for (int64_t i = 0; i < NCAND; i++) {
                        if (std::isfinite(cand_guided[(size_t) i])) {
                            z += std::exp((double) cand_guided[(size_t) i] - mx);
                        }
                    }
                    bool ov = false;
                    for (int64_t j = 0; j < gk; j++) {
                        const float x = cand_guided[(size_t) (1 + gids[j])];
                        if (std::isfinite(x)) {
                            zc += std::exp((double) x - mx);
                            ov = true;
                        }
                    }
                    float top1 = -INFINITY;
                    for (int64_t i = 1; i < NCAND; i++) {
                        top1 = std::max(top1, gd_pre[(size_t) i]);
                    }
                    gd_mass.push_back(z > 0.0 ? (float) (zc / z) : 0.0f);
                    gd_overlap.push_back(ov ? 1 : 0);
                    gd_gap.push_back(std::min(100.0f, top1 - gd_pre[(size_t) (1 + gids[0])]));
                    gd_scale.push_back(gp.scale);
                }
                if (opt.guide_product >= 0.0f) {
                    // Product rule (probe): candidates only, a * ln p_enc + ln p_LM,
                    // the LM's side from its CFG'd logits before the top-k filter.
                    const float * glp = g.lp.data() + (size_t) (gf * g.K);
                    std::fill(cand_guided.begin(), cand_guided.end(), -INFINITY);
                    for (int64_t j = 0; j < gk; j++) {
                        const size_t v = (size_t) (1 + gids[j]);
                        cand_guided[v] = opt.guide_product * glp[j] + gd_pre[v];
                    }
                } else {
                    yue2_guide_apply(cand_guided.data(), NCAND, 0, 1, NCAND, 0, g, gp, gf);
                }
                // One survivor (scale 1, k 1) is taken without a draw, so the
                // shared RNG stays where the hard pin leaves it and the depth
                // decoder samples the same acoustic codes: the parity gate.
                int64_t n_fin = 0;
                for (int64_t i = 0; i < NCAND && n_fin < 2; i++) {
                    if (std::isfinite(cand_guided[(size_t) i])) {
                        n_fin++;
                        guide_pick = i;
                    }
                }
                if (n_fin != 1) {
                    guide_pick = -1;
                }
                // Scale 0 with candidates only leaves the LM's own logits on the
                // candidates, so when its top-k dropped all of them nothing is
                // left and the sampler would fall back to EOS. Re-admit them at
                // their CFG'd logits from before that filter: the LM still picks
                // its favourite among the source's candidates.
                if (n_fin == 0) {
                    for (int64_t j = 0; j < gk; j++) {
                        cand_guided[(size_t) (1 + gids[j])] = gd_pre[(size_t) (1 + gids[j])];
                    }
                    if (t == 0) {
                        gd_fallback++;
                    }
                }
            }

            // ── repetition penalty on the guided candidates (knobs at
            // defaults: no-op). History = this take's emitted codes. ──
            if ((!forced_it || soft_it) && opt.knobs.rep_penalty > 1.0f) {
                mm3_apply_rep_penalty(cand_guided.data(), NCAND, outs[t].semantic_all.data(),
                                      (int64_t) outs[t].semantic_all.size(), opt.knobs);
            }

            // ── sample (or replay) ──
            if (forced_it && !soft_it) {
                sem_code[(size_t) t] = fs_it;  // range-checked at entry
            } else {
                const int64_t idx = guide_pick >= 0 ? guide_pick
                                    : opt.knobs.any_active()
                                        ? mm3_sample_knobbed(cand_guided.data(), NCAND, TOPK, opt.knobs,
                                                             rngs[(size_t) t], &samp_scratch)
                                        : mm3_sample_top_k(cand_guided.data(), NCAND, TOPK, rngs[(size_t) t],
                                                           &samp_scratch);
                if (idx == 0) {
                    // EOS. This take is finished; the others carry on. Its rows
                    // keep being computed (the batch shape is fixed for the run)
                    // but nothing is sampled from or emitted for them again, so
                    // an early EOS costs the ensemble nothing but the rows it
                    // was already paying for.
                    active[t]        = 0;
                    outs[t].eos_hit  = true;
                    continue;
                }
                sem_code[(size_t) t] = (int32_t) (idx - 1);
                if (soft_it && fs_it >= 0 && t == 0) {
                    fb_biased++;
                    fb_agree += sem_code[0] == fs_it ? 1 : 0;
                }
                if (guide_it && t == 0) {
                    int32_t rank = -1;
                    for (int64_t j = 0; j < gk && rank < 0; j++) {
                        rank = gids[j] == sem_code[0] ? (int32_t) j : -1;
                    }
                    double mx = -INFINITY, z = 0.0;
                    for (const float x : gd_pre) {
                        mx = std::max(mx, (double) x);
                    }
                    for (const float x : gd_pre) {
                        z += std::isfinite(x) ? std::exp((double) x - mx) : 0.0;
                    }
                    const double lp = (double) gd_pre[(size_t) idx] - mx - std::log(z);
                    gd_rank.push_back(rank);
                    gd_surprise.push_back((float) std::min(100.0, -lp));
                }
            }
        }
        out->host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();

        if (opt.stop_after_first_eos) {
            // Every row has sampled. Same-step ties use the original take
            // index; the winner has exactly the same draws as a full batch.
            for (int t = 0; t < K; t++) {
                if (outs[t].eos_hit && outs[t].n_frames > 0) {
                    first_eos_take = t;
                    break;
                }
            }
            if (first_eos_take >= 0) {
                fprintf(stderr, "[MM3-AR] First EOS: take %d at %lld frames; stopping the candidate batch\n",
                        first_eos_take, (long long) outs[first_eos_take].n_frames);
                break;
            }
        }
        bool any_active = false;
        for (int t = 0; t < K; t++) {
            any_active = any_active || active[t] != 0;
        }
        if (!any_active) {
            break;
        }

        // A frozen take still occupies its rows, so give it a code the graph can
        // embed. Nothing is read back for it, so the value only has to be legal.
        for (int t = 0; t < K; t++) {
            if (!active[t]) {
                sem_code[(size_t) t] = 0;
            }
        }

        // ── state fix (probe): make the conditional state agree with the
        // drawn code, for the depth decoder and/or emitted stream 0 only.
        // The unconditional row is left alone. ──
        const float * depth_rows = hidden.data();
        std::fill(fixed_ok.begin(), fixed_ok.end(), (char) 0);
        if (guide_it && opt.state_fix) {
            if (opt.state_fix & 1) {
                depth_in = hidden;
                depth_rows = depth_in.data();
            }
            for (int t = 0; t < K; t++) {
                if (!active[t]) {
                    continue;
                }
                std::vector<float> & fr = fixed[(size_t) t];
                const float *        hc = hidden.data() + row_c(t) * (size_t) H;
                fr.assign(hc, hc + H);
                float     rel = 0.0f;
                const int st  = mm3_state_fix(fr.data(), H, logits.data() + row_c(t) * (size_t) hn + (OFF - hlo), SV,
                                              sem_code[(size_t) t], 10, 8, &rel);
                if (opt.state_fix & 1) {
                    memcpy(depth_in.data() + row_c(t) * (size_t) H, fr.data(), (size_t) H * sizeof(float));
                }
                fixed_ok[(size_t) t] = (opt.state_fix & 2) ? 1 : 0;
                if (t == 0) {
                    sf_rel.push_back(rel);
                    sf_steps.push_back(st);
                }
            }
        }

        // ── depth decoder: the seven acoustic codes and their hidden states ──
        // Semantic-only forcing keeps the RNG alive so the depth decoder samples
        // the acoustic codebooks; full forcing passes null and replays them.
        const bool      forced_ac_it = forced_ac_given && it < opt.forced_len;
        const int32_t * forced_ac = forced_ac_it ? opt.forced_acoustic + it * NC : nullptr;
        const bool      greedy_it = forced_ac_it || (opt.depth_greedy && guide_it);
        if (!mm3_depth_decode_takes(m, depth_rows, sem_code.data(), forced_ac, frames.data(), err,
                                    greedy_it ? nullptr : rngs.data(), TOPK)) {
            return false;
        }
        out->depth_ms += frames[0].ms;
        out->depth_steps++;
        for (int t = 0; t < K; t++) {
            if (frames[(size_t) t].n_codes != (int) NC) {
                if (err) {
                    *err = "the depth decoder returned " + std::to_string(frames[(size_t) t].n_codes) +
                           " codes, expected " + std::to_string((long long) NC);
                }
                return false;
            }
        }

        for (int t = 0; t < K; t++) {
            if (!active[t]) {
                continue;
            }
            outs[t].semantic_all.push_back(sem_code[(size_t) t]);
            for (int64_t i = 0; i < NC; i++) {
                outs[t].acoustic_all.push_back(frames[(size_t) t].codes[i]);
            }
            outs[t].n_iterations++;
        }

        const bool dumping = (int64_t) out->dumps.size() < opt.dump_iters;
        if (dumping) {
            // Single-take path only (guarded at entry), so the [2, H] / [2, SV]
            // shapes these carry are still literally true.
            // Through row_c/row_u: at P == 1 with base rows, row 1 is the base.
            const float * lrow_c = logits.data() + row_c(0) * (size_t) hn;
            const float * lrow_u = logits.data() + row_u(0) * (size_t) hn;
            MM3ArDump     d;
            d.last_hidden.assign(hidden.begin() + (ptrdiff_t) (row_c(0) * (size_t) H),
                                 hidden.begin() + (ptrdiff_t) ((row_c(0) + 1) * (size_t) H));
            d.last_hidden.insert(d.last_hidden.end(), hidden.begin() + (ptrdiff_t) (row_u(0) * (size_t) H),
                                 hidden.begin() + (ptrdiff_t) ((row_u(0) + 1) * (size_t) H));
            d.sem_logits.resize((size_t) (2 * SV));
            memcpy(d.sem_logits.data(), lrow_c + (OFF - hlo), (size_t) SV * sizeof(float));
            memcpy(d.sem_logits.data() + SV, lrow_u + (OFF - hlo), (size_t) SV * sizeof(float));
            d.guided.assign(cand_guided.begin() + 1, cand_guided.end());
            d.feedback.assign((size_t) (2 * H), 0.0f);
            d.depth_hidden = frames[0].hiddens;
            out->dumps.push_back(std::move(d));
        }

        // ── emit (iteration 0 is fed back but never emitted) ──
        if (it > 0) {
            for (int t = 0; t < K; t++) {
                if (!active[t]) {
                    continue;
                }
                if (opt.collect_hiddens) {
                    const float * hc = fixed_ok[(size_t) t] ? fixed[(size_t) t].data()
                                                            : hidden.data() + row_c(t) * (size_t) H;
                    outs[t].frame_hiddens.insert(outs[t].frame_hiddens.end(), hc, hc + H);
                    outs[t].frame_hiddens.insert(outs[t].frame_hiddens.end(), frames[(size_t) t].hiddens.begin(),
                                                 frames[(size_t) t].hiddens.end());
                    // State blend: every stream of this frame moves toward the
                    // matching frame of another block (a launder's). Output
                    // only; the LM and its feedback never see it.
                    const int64_t f = outs[t].n_frames;
                    if (opt.blend && f < opt.blend_frames) {
                        const int64_t FH = (NC + 1) * H;
                        float *       dst = outs[t].frame_hiddens.data() + (size_t) (f * FH);
                        const float * src = opt.blend + (size_t) (f * FH);
                        const float   a   = opt.blend_amount;
                        for (int64_t i = 0; i < FH; i++) {
                            dst[i] += a * (src[i] - dst[i]);
                        }
                    }
                }
                outs[t].n_frames++;
                frames_done[(size_t) t] = outs[t].n_frames;
                if (outs[t].n_frames >= max_frames) {
                    // Note 2: no feedback, no decode step, for the last frame.
                    // Freezing rather than breaking is what lets a take that
                    // reaches the cap early stop while the others continue.
                    active[t] = 0;
                }
            }
            // Progress reports the LEADING take: they run in lockstep, so this
            // is the frame the batch as a whole has reached.
            int64_t lead = 0;
            for (int t = 0; t < K; t++) {
                lead = frames_done[(size_t) t] > lead ? frames_done[(size_t) t] : lead;
            }
            if (opt.on_frame) {
                opt.on_frame(lead, max_frames);
            }
            // After on_frame (progress first, so a long dispatch is not
            // mistaken for a stalled planner) and before the cancel poll, so a
            // cancel during a dispatched window is still caught this iteration.
            if (opt.on_takes_frame_ready) {
                if (!opt.on_takes_frame_ready(frames_done.data(), K, err)) {
                    return false;
                }
            } else if (opt.on_frame_ready && K == 1) {
                if (!opt.on_frame_ready(frames_done[0], err)) {
                    return false;
                }
            }
            if (opt.should_cancel && opt.should_cancel()) {
                if (err) {
                    *err = MM3_ERR_CANCELLED;
                }
                return false;
            }
            bool still = false;
            for (int t = 0; t < K; t++) {
                still = still || active[t] != 0;
            }
            if (!still) {
                break;
            }
        }

        // ── feed back and advance ──
        // Frozen takes feed back their last frame's codes. Their rows are never
        // read again, so the content is irrelevant — but the KV cache advances
        // for every row together, so they must feed back SOMETHING.
        for (int t = 0; t < K; t++) {
            sem_tok[(size_t) t] = sem_code[(size_t) t] + (int32_t) OFF;
            for (int64_t i = 0; i < NC; i++) {
                ac_rows[(size_t) (t * NC + i)] = frames[(size_t) t].codes[i] + (int32_t) (i * AV);
            }
        }
        if (sched_on) {
            sched_apply(it + 1);
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (!mm3_lm_decode(m, &g_mm3_lm, sem_tok.data(), ac_rows.data(), hidden.data(), logits.data(),
                               feedback.data(), err)) {
                return false;
            }
            out->lm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            out->lm_steps++;
        }
        mirror_uncond();
        if (lrc_on) {
            std::vector<float> slice;
            for (int hh = 0; hh < MM3_ALIGN_N_HEADS; hh++) {
                const MM3AlignHead & ah = MM3_ALIGN_HEADS[hh];
                if (!mm3_lm_read_attn_slice(m.lm_cfg, g_mm3_lm.decode, (size_t) ah.layer, lyr0, lyr1, slice)) {
                    continue;
                }
                // slice is [Nh, n_cols] — take this head's row out of it.
                const size_t base = (size_t) ah.head * (size_t) lrc_tok;
                if (base + (size_t) lrc_tok > slice.size()) {
                    continue;
                }
                for (int64_t t = 0; t < lrc_tok; t++) {
                    lrc_rows[(size_t) (hh * lrc_tok + t)].push_back(slice[base + (size_t) t]);
                }
            }
        }
        if (align_dump && lyr0 >= 0 && lyr1 > lyr0) {
            std::vector<float> slice;
            for (size_t L = 0; L < g_mm3_lm.decode.attn_scores.size(); L++) {
                if (mm3_lm_read_attn_slice(m.lm_cfg, g_mm3_lm.decode, L, lyr0, lyr1, slice)) {
                    align_acc.insert(align_acc.end(), slice.begin(), slice.end());
                }
            }
            align_frames++;
        }
        if (dumping) {
            MM3ArDump & d = out->dumps.back();
            // The reference dumps [2, H]; both rows are the same vector because
            // the sampled codes are shared across the CFG pair.
            memcpy(d.feedback.data(), feedback.data(), (size_t) H * sizeof(float));
            memcpy(d.feedback.data() + H, feedback.data(), (size_t) H * sizeof(float));
        }
    }

    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_start).count();
    for (int t = 0; t < K; t++) {
        outs[t].total_ms = total_ms;
    }
    if (soft_force) {
        auto median = [](std::vector<float> & v) -> float {
            if (v.empty()) {
                return NAN;
            }
            std::nth_element(v.begin(), v.begin() + (ptrdiff_t) (v.size() / 2), v.end());
            return v[v.size() / 2];
        };
        const float gap_med = median(fb_gap);
        float       gap_p90 = NAN;
        if (!fb_gap.empty()) {
            const size_t i = (size_t) (0.9 * (double) (fb_gap.size() - 1));
            std::nth_element(fb_gap.begin(), fb_gap.begin() + (ptrdiff_t) i, fb_gap.end());
            gap_p90 = fb_gap[i];
        }
        fprintf(stderr,
                "[MM3-AR] Soft forcing: bias %.2f, %lld biased frames, %lld free; agree %.1f%%; "
                "median guided top-1 margin %.2f, median top-1 minus source %.2f, p90 %.2f; source in top-%d %.1f%%\n",
                opt.force_bias, (long long) fb_biased, (long long) (fb_margin.size() - (size_t) fb_biased),
                fb_biased ? 100.0 * (double) fb_agree / (double) fb_biased : 0.0, median(fb_margin), gap_med, gap_p90,
                TOPK, fb_forced ? 100.0 * (double) fb_in50 / (double) fb_forced : 0.0);
    }
    if (!sf_rel.empty()) {
        std::vector<float> rs = sf_rel;
        std::sort(rs.begin(), rs.end());
        size_t ok = 0, moved = 0, big = 0;
        double st = 0.0;
        for (size_t i = 0; i < sf_steps.size(); i++) {
            ok += sf_steps[i] >= 0 ? 1 : 0;
            moved += sf_steps[i] > 0 ? 1 : 0;
            st += sf_steps[i] > 0 ? sf_steps[i] : 0;
            big += sf_rel[i] > 0.3f ? 1 : 0;
        }
        fprintf(stderr,
                "[MM3-AR] State fix (%s%s%s): %zu frames, %.1f%% moved, %.1f%% reached top-10, mean steps %.2f; "
                "|dh|/|h| median %.3f p90 %.3f max %.3f; above 0.3 on %.1f%%%s\n",
                opt.state_fix & 1 ? "depth" : "", opt.state_fix == 3 ? "+" : "", opt.state_fix & 2 ? "stream0" : "",
                rs.size(), 100.0 * (double) moved / (double) rs.size(), 100.0 * (double) ok / (double) rs.size(),
                moved ? st / (double) moved : 0.0, (double) rs[rs.size() / 2], (double) rs[(size_t) (0.9 * (double) (rs.size() - 1))],
                (double) rs.back(), 100.0 * (double) big / (double) rs.size(), opt.depth_greedy ? ", depth greedy" : "");
    } else if (opt.depth_greedy && guided) {
        fprintf(stderr, "[MM3-AR] Depth decoder greedy on guided frames\n");
    }
    if (guided && !gd_rank.empty()) {
        const size_t n   = gd_rank.size();
        size_t       a1 = 0, ak = 0, ov = 0;
        double       ms = 0.0;
        for (size_t i = 0; i < n; i++) {
            a1 += gd_rank[i] == 0 ? 1 : 0;
            ak += gd_rank[i] >= 0 ? 1 : 0;
            ov += gd_overlap[i] ? 1 : 0;
            ms += (double) gd_mass[i];
        }
        std::vector<float> gs = gd_gap;
        std::nth_element(gs.begin(), gs.begin() + (ptrdiff_t) (gs.size() / 2), gs.end());
        fprintf(stderr,
                "[MM3-AR] Guide: scale %.2f%s k %lld outside %.1f, %zu frames; agree top-1 %.1f%%, any candidate "
                "%.1f%%; LM top-%d overlaps candidates %.1f%%; mean LM mass on candidates %.3f; median gap %.2f\n",
                (double) opt.guide_p.scale, opt.guide_gate ? " (gated)" : "",
                (long long) (opt.guide_p.k > 0 ? std::min<int64_t>(opt.guide_p.k, opt.guide->K) : opt.guide->K),
                (double) opt.guide_p.outside, n, 100.0 * (double) a1 / (double) n, 100.0 * (double) ak / (double) n,
                TOPK, 100.0 * (double) ov / (double) n, ms / (double) n, (double) gs[gs.size() / 2]);
        if (opt.guide_product >= 0.0f) {
            fprintf(stderr, "[MM3-AR] Guide rule: product, a %.2f (scale and outside unused)\n",
                    (double) opt.guide_product);
        }
        if (gd_fallback) {
            fprintf(stderr, "[MM3-AR] Guide fallback: the LM's top-%d held no candidate on %lld frames (%.1f%%); "
                            "re-admitted at their unfiltered logits\n",
                    TOPK, (long long) gd_fallback, 100.0 * (double) gd_fallback / (double) n);
        }
        if (!opt.guide_trace.empty()) {
            FILE * f = hs_fopen(opt.guide_trace, "wb");
            if (!f) {
                fprintf(stderr, "[MM3-AR] cannot write guide trace %s\n", opt.guide_trace.c_str());
            } else {
                const auto arr_f = [&](const char * name, const std::vector<float> & v, bool last) {
                    fprintf(f, "\"%s\":[", name);
                    for (size_t i = 0; i < v.size(); i++) {
                        fprintf(f, i ? ",%.4g" : "%.4g", (double) v[i]);
                    }
                    fprintf(f, last ? "]" : "],");
                };
                const auto arr_i = [&](const char * name, const std::vector<int32_t> & v) {
                    fprintf(f, "\"%s\":[", name);
                    for (size_t i = 0; i < v.size(); i++) {
                        fprintf(f, i ? ",%d" : "%d", v[i]);
                    }
                    fprintf(f, "],");
                };
                // Entry 0 is the un-emitted iteration; entry j >= 1 is frame j - 1.
                fprintf(f, "{\"fps\":25,\"first_entry\":\"warm-up\",\"top_k\":%d,", TOPK);
                arr_i("rank", gd_rank);
                arr_i("overlap", gd_overlap);
                arr_f("gap", gd_gap, false);
                arr_f("mass", gd_mass, false);
                arr_f("surprise", gd_surprise, false);
                arr_f("scale", gd_scale, true);
                fprintf(f, "}\n");
                fclose(f);
            }
        }
    }

    // A take with no frames is a failed render, not a short one — and because
    // the batch is one pass there is no partial result worth salvaging.
    for (int t = 0; t < K; t++) {
        if (outs[t].n_frames == 0) {
            if (err) {
                const std::string which = K > 1 ? " (take " + std::to_string(t) + ")" : "";
                *err = outs[t].eos_hit
                           ? "the LM emitted EOS on the first iteration; zero audio frames were generated" + which
                           : "zero audio frames were generated" + which;
            }
            return false;
        }
    }

    if (lrc_on && out->n_frames > 1 && !lrc_rows.empty() && !lrc_rows[0].empty()) {
        const int   n_fr = (int) lrc_rows[0].size();
        const float dur =
            m.lm_cfg.frame_rate > 0 ? (float) out->n_frames / (float) m.lm_cfg.frame_rate : 0.0f;
        std::vector<float> flat;
        flat.reserve(lrc_rows.size() * (size_t) n_fr);
        bool ragged = false;
        for (const auto & r : lrc_rows) {
            if ((int) r.size() != n_fr) {
                ragged = true;
                break;
            }
            flat.insert(flat.end(), r.begin(), r.end());
        }
        if (ragged) {
            // One head short of a frame would silently skew the whole matrix;
            // refuse rather than align against a misshapen one.
            fprintf(stderr, "[MM3-LRC] Ragged capture — skipping alignment\n");
        } else {
            const std::vector<int> ids(cond_ids + lyr0, cond_ids + lyr1);
            const auto             t_lrc = std::chrono::steady_clock::now();
            out->lrc = mm3_align_build_lrc(flat, ids, opt.tok->bpe, n_fr, dur);
            const double lrc_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lrc).count();
            fprintf(stderr, "[MM3-LRC] %s (%d frames, %.1f s, %.0f ms)\n",
                    out->lrc.empty() ? "no alignment produced" : "LRC built", n_fr, dur, lrc_ms);
        }
    }

    // ── LRC via post-hoc replay (the default path) ───────────────────────────
    // The decode above ran pure flash; recompute the alignment attention from
    // the sampled codes now, while the LM is still resident. Failure only costs
    // the LRC, never the render.
    // Runs ONCE PER TAKE: the replay is teacher-forced from a take's own codes,
    // so each take gets timestamps for the song it actually sang. It resets
    // kv_pos, which is why it can only run here, after the batched decode is
    // finished with the cache.
    if (opt.want_lrc && opt.tok != nullptr && !g_mm3_lm.align_capture && !align_dump && lyr0 >= 0 && lyr1 > lyr0) {
        // The replay runs the whole plan under ONE adapter setting. A schedule
        // left the graph at its tail strength; align under the body's instead,
        // which is where the lyrics are. The tail's timestamps are approximate.
        if (sched_on) {
            mm3_lm_set_adapter(&g_mm3_lm, opt.lm_adapter, opt.lm_adapter_scales);
        }
        for (int t = 0; t < K; t++) {
            MM3ArResult * o = &outs[t];
            // Selection is already known, before any alignment work.
            if ((opt.lrc_eos_only && !o->eos_hit) || (first_eos_take >= 0 && t != first_eos_take)) {
                continue;
            }
            if (o->n_frames <= 1) {
                continue;
            }
            // Column count must match what the live path would have captured: one
            // per decode step — every iteration that pushed codes also fed back,
            // except the last when the max-frames break fired before its feedback.
            const int64_t n_steps = o->eos_hit ? o->n_iterations : o->n_iterations - 1;
            if (n_steps <= 1) {
                continue;
            }
            std::vector<float> flat;
            std::string        rerr;
            const auto         t_lrc = std::chrono::steady_clock::now();
            if (mm3_lm_lrc_replay(m, &g_mm3_lm, cond_ids, n_prompt, o->semantic_all.data(), o->acoustic_all.data(),
                                  n_steps, lyr0, lyr1, &flat, &rerr)) {
                const float dur = m.lm_cfg.frame_rate > 0 ? (float) o->n_frames / (float) m.lm_cfg.frame_rate : 0.0f;
                const std::vector<int> ids(cond_ids + lyr0, cond_ids + lyr1);
                o->lrc              = mm3_align_build_lrc(flat, ids, opt.tok->bpe, (int) n_steps, dur);
                const double lrc_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lrc).count();
                fprintf(stderr, "[MM3-LRC] %s via replay%s (%lld frames, %.1f s, %.0f ms)\n",
                        o->lrc.empty() ? "no alignment produced" : "LRC built",
                        K > 1 ? (" take " + std::to_string(t)).c_str() : "", (long long) n_steps, dur, lrc_ms);
            } else {
                fprintf(stderr, "[MM3-LRC] Replay failed (%s) — no LRC for this render\n", rerr.c_str());
            }
        }
    }

    if (align_dump && align_frames > 0 && !align_acc.empty()) {
        // Flat binary: a small ASCII header line, then f32 in
        // [frame][layer][head][token] order. Layout is written down rather than
        // implied so the offline scorer cannot silently misread it.
        const int64_t n_tok    = lyr1 - lyr0;
        const int64_t n_layers = (int64_t) g_mm3_lm.decode.attn_scores.size();
        const int64_t n_heads  = n_layers > 0 && align_frames > 0
            ? (int64_t) (align_acc.size() / (size_t) (align_frames * n_layers * n_tok)) : 0;
        const char *  path     = std::getenv("MM3_ALIGN_FILE");
        std::string   out_path = path && path[0] ? path : "mm3-align-dump.bin";
        FILE *        f        = fopen(out_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[MM3-Align] Could not open %s for writing\n", out_path.c_str());
        } else {
            // v2 adds the lyric token IDs, written as int32 immediately after
            // the header and before the float payload. Without them the dump
            // can only be scored on TOKEN index, and comparing "fraction
            // through tokens" against a reference's "fraction through words" is
            // not a like-for-like test — BPE tokens per word vary, so the two
            // curves differ even for a perfect alignment. The ids let a token
            // be resolved to its text (via tokenizer.ggml.tokens in the LM
            // GGUF) and lined up with real words.
            char hdr[256];
            const int n = snprintf(hdr, sizeof(hdr),
                                   "MM3ALIGN2 frames=%lld layers=%lld heads=%lld tokens=%lld "
                                   "order=frame,layer,head,token fps=%u ids=int32\n",
                                   (long long) align_frames, (long long) n_layers, (long long) n_heads,
                                   (long long) n_tok, m.lm_cfg.frame_rate);
            fwrite(hdr, 1, (size_t) n, f);
            fwrite(cond_ids + lyr0, sizeof(int32_t), (size_t) n_tok, f);
            fwrite(align_acc.data(), sizeof(float), align_acc.size(), f);
            fclose(f);
            fprintf(stderr,
                    "[MM3-Align] Wrote %s — %lld frames x %lld layers x %lld heads x %lld lyric tokens "
                    "(%.1f MB)\n",
                    out_path.c_str(), (long long) align_frames, (long long) n_layers, (long long) n_heads,
                    (long long) n_tok, (double) (align_acc.size() * sizeof(float)) / 1e6);
        }
    }

    if (K > 1) {
        // Per-take frame counts, because an early EOS makes them differ and the
        // difference decides how much audio each take is worth.
        std::string per;
        for (int t = 0; t < K; t++) {
            per += (t ? ", " : "") + std::to_string((long long) outs[t].n_frames) +
                   (outs[t].eos_hit ? " (EOS)" : "");
        }
        fprintf(stderr, "[MM3-AR] %d takes in one batched pass — frames per take: %s\n", K, per.c_str());
    }
    fprintf(stderr,
            "[MM3-AR] %lld frames (%lld iterations%s) in %.0f ms — prefill %.0f, LM %.0f (%lld steps, %.1f ms/step), "
            "depth %.0f (%lld batch steps, %.1f ms/frame), host %.0f\n",
            (long long) out->n_frames, (long long) out->n_iterations, out->eos_hit ? ", EOS" : "", out->total_ms,
            out->prefill_ms, out->lm_ms, (long long) out->lm_steps,
            out->lm_steps ? out->lm_ms / (double) out->lm_steps : 0.0, out->depth_ms,
            (long long) out->depth_steps,
            out->depth_steps ? out->depth_ms / (double) out->depth_steps : 0.0, out->host_ms);
    // ── Non-finite logits: warn, or refuse ──────────────────────────────────
    //
    // `fix()` clamps a NaN/+inf candidate to -inf so the draw stays well
    // defined. That is the right local behaviour and it is why a run like this
    // still finishes -- but finishing is not the same as being worth anything.
    //
    // The observed failure is not a scatter of bad values: it is a whole
    // candidate row going non-finite on EVERY iteration, so the count comes out
    // as an exact multiple of (SV + 1) per iteration. One dead row means the CFG
    // blend has nothing to blend; two means the plan is pure noise. Either way
    // the caller was being handed a "successful" generation built out of -inf,
    // with a stderr line as the only clue.
    //
    // So: below one full row's worth per iteration, warn and continue -- that is
    // a regime nothing has ever been observed in, and clamping may well be
    // enough. At or above it, fail with the cause named. The overwhelmingly
    // likely cause is an LM adapter whose factors overflowed the f16 store
    // (mm3-lm-adapter.h now refuses those at load), so the message says so.
    const int64_t candidates_per_row = SV + 1;
    const int64_t row_equivalents    = out->n_iterations * (int64_t) K * (int64_t) P;
    const int64_t budget             = candidates_per_row * row_equivalents;
    // The plan dial's base rows are counted by fix() too, so the PERCENTAGE is
    // over all of them; the fatal threshold below stays one dead row per take
    // per step, because a dead row on either side poisons the mixed row.
    const int64_t counted = budget * (plan_mix ? 2 : 1);
    if (out->nonfinite_logits) {
        fprintf(stderr, "[MM3-AR] WARNING: %lld non-finite candidate logits were clamped to -inf (%.1f%% of %lld)\n",
                (long long) out->nonfinite_logits, counted ? 100.0 * (double) out->nonfinite_logits / (double) counted : 0.0,
                (long long) counted);
    }
    if (budget > 0 && out->nonfinite_logits * (int64_t) P >= budget) {
        char buf[352];
        snprintf(buf, sizeof(buf),
                 "the LM produced non-finite logits for %lld of %lld candidates - at least one whole CFG row is dead "
                 "on every step, so this plan is noise. If an LM adapter is loaded, that checkpoint has diverged; "
                 "try a different one or the pristine base.",
                 (long long) out->nonfinite_logits, (long long) budget);
        fprintf(stderr, "[MM3-AR] FATAL: %s\n", buf);
        if (err) {
            *err = buf;
        }
        return false;
    }
    return true;
}

// Plan ONE song — the shape every existing caller was written against, and the
// path a normal single-track render still takes. Forces the graphs back to one
// take, so a K > 1 render followed by a K = 1 render rebuilds rather than
// silently reusing wide shapes.
static bool mm3_ar_plan(const MM3Model & m, const int32_t * cond_ids, const int32_t * uncond_ids, int64_t n_prompt,
                        const MM3ArOptions & opt, MM3ArResult * out, std::string * err) {
    MM3ArOptions one = opt;
    one.n_takes      = 1;
    one.on_takes_frame_ready = nullptr;
    return mm3_ar_plan_takes(m, cond_ids, uncond_ids, n_prompt, one, out, err);
}
