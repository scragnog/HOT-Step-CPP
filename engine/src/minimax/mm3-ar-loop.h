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

    int64_t dump_iters      = 0;
    bool    collect_hiddens = true;

    // Lyric timestamps: capture the alignment heads' decode attention and emit
    // LRC into MM3ArResult::lrc. Needs the tokenizer to turn the lyric token
    // ids back into the text lrc_align() groups into lines, so both are set
    // together or not at all.
    bool                  want_lrc = false;
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
};

// The sentinel a cancelled run reports. Compared by value, not by prefix, so a
// genuine error can never be mistaken for a cancel.
#define MM3_ERR_CANCELLED "cancelled"

static MM3LmGraph g_mm3_lm;

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
        mm3_lm_adapter_free(g_mm3_lm_adapter);
        g_mm3_lm_adapter = nullptr;
    }
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
    const int           K = mm3_clamp_takes(c, opt.n_takes);
    if (K != opt.n_takes && opt.n_takes > 1) {
        fprintf(stderr, "[MM3-AR] %d takes requested; the row budget allows %d — rendering %d\n", opt.n_takes,
                mm3_max_takes(c), K);
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
        if (opt.forced_len - 1 < max_frames) {
            max_frames = opt.forced_len - 1;  // entry 0 is the un-emitted iteration
        }
        if (max_frames <= 0) {
            if (err) {
                *err = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
            }
            return false;
        }
    }

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
    // Both graph sets bake rows() into their shapes, so K has to be settled
    // before either is prepared. Kept in lockstep on purpose: the depth decoder
    // reads the LM's hidden rows straight out of the same buffer.
    mm3_lm_set_takes(&g_mm3_lm, K);
    mm3_depth_set_takes(&g_mm3_depth, K);
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
    const auto mirror_uncond = [&]() {
        if (P > 1 || K != 1) {
            return;
        }
        memcpy(hidden.data() + H, hidden.data(), (size_t) H * sizeof(float));
        memcpy(logits.data() + hn, logits.data(), (size_t) hn * sizeof(float));
    };

    const auto t_start = std::chrono::steady_clock::now();
    {
        const auto t0 = std::chrono::steady_clock::now();
        mm3_lm_set_adapter(&g_mm3_lm, opt.lm_adapter, opt.lm_adapter_scales);
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
    std::vector<MM3DepthFrame> frames((size_t) K);

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
        if (forced && it >= opt.forced_len) {
            break;
        }
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
            }

            // ── repetition penalty on the guided candidates (knobs at
            // defaults: no-op). History = this take's emitted codes. ──
            if (!forced && opt.knobs.rep_penalty > 1.0f) {
                mm3_apply_rep_penalty(cand_guided.data(), NCAND, outs[t].semantic_all.data(),
                                      (int64_t) outs[t].semantic_all.size(), opt.knobs);
            }

            // ── sample (or replay) ──
            if (forced) {
                const int32_t fs = opt.forced_semantic[it];
                if (fs < 0 || (int64_t) fs >= SV) {
                    if (err) {
                        *err = "forced semantic code " + std::to_string(fs) + " at iteration " +
                               std::to_string((long long) it) + " is outside [0, " + std::to_string((long long) SV) +
                               ")";
                    }
                    return false;
                }
                sem_code[(size_t) t] = fs;
            } else {
                const int64_t idx = opt.knobs.any_active()
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
            }
        }
        out->host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();

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

        // ── depth decoder: the seven acoustic codes and their hidden states ──
        // Semantic-only forcing keeps the RNG alive so the depth decoder samples
        // the acoustic codebooks; full forcing passes null and replays them.
        const int32_t * forced_ac = forced_ac_given ? opt.forced_acoustic + it * NC : nullptr;
        if (!mm3_depth_decode_takes(m, hidden.data(), sem_code.data(), forced_ac, frames.data(), err,
                                    forced_ac_given ? nullptr : rngs.data(), TOPK)) {
            return false;
        }
        out->depth_ms += frames[0].ms;
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
            const float * lrow_c = logits.data();
            const float * lrow_u = logits.data() + hn;
            MM3ArDump     d;
            d.last_hidden.assign(hidden.begin(), hidden.begin() + 2 * H);
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
                    const float * hc = hidden.data() + row_c(t) * (size_t) H;
                    outs[t].frame_hiddens.insert(outs[t].frame_hiddens.end(), hc, hc + H);
                    outs[t].frame_hiddens.insert(outs[t].frame_hiddens.end(), frames[(size_t) t].hiddens.begin(),
                                                 frames[(size_t) t].hiddens.end());
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
        for (int t = 0; t < K; t++) {
            MM3ArResult * o = &outs[t];
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
            "depth %.0f (%.1f ms/frame), host %.0f\n",
            (long long) out->n_frames, (long long) out->n_iterations, out->eos_hit ? ", EOS" : "", out->total_ms,
            out->prefill_ms, out->lm_ms, (long long) out->lm_steps,
            out->lm_steps ? out->lm_ms / (double) out->lm_steps : 0.0, out->depth_ms,
            out->n_iterations ? out->depth_ms / (double) out->n_iterations : 0.0, out->host_ms);
    if (out->nonfinite_logits) {
        fprintf(stderr, "[MM3-AR] WARNING: %lld non-finite candidate logits were clamped to -inf\n",
                (long long) out->nonfinite_logits);
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
