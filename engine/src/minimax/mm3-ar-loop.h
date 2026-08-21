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

// Plan one song. `cond_ids` / `uncond_ids` are the two prefill rows.
//
// Not thread-safe: the caller serialises (mm3-server.h holds g_mm3_mutex).
static bool mm3_ar_plan(const MM3Model & m, const int32_t * cond_ids, const int32_t * uncond_ids, int64_t n_prompt,
                        const MM3ArOptions & opt, MM3ArResult * out, std::string * err) {
    const MM3LmConfig & c  = m.lm_cfg;
    const int64_t       H  = (int64_t) c.embedding_length;
    const int64_t       V  = (int64_t) c.vocab_size;
    const int64_t       SV = (int64_t) c.semantic_vocab_size;
    const int64_t       NC = (int64_t) c.num_codebooks - 1;
    const int64_t       AV = (int64_t) c.acoustic_vocab_size;
    const int64_t       OFF  = (int64_t) c.semantic_vocab_offset;
    const int64_t       EOS  = (int64_t) c.eos_audio;
    const float         CFG  = c.ar_cfg_scale > 0.0f ? c.ar_cfg_scale : 1.5f;
    const int           TOPK = c.ar_top_k > 0 ? (int) c.ar_top_k : 50;

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
        const bool want = opt.want_lrc && opt.tok != nullptr && lrc_live;
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


    *out             = MM3ArResult{};
    out->hidden_dim  = H;
    out->n_codebooks = NC + 1;
    out->sem_vocab   = SV;

    std::vector<float> hidden((size_t) (H * MM3_LM_CFG_ROWS));
    std::vector<float> logits((size_t) (hn * MM3_LM_CFG_ROWS));
    std::vector<float> feedback((size_t) H);

    const auto t_start = std::chrono::steady_clock::now();
    {
        const auto t0 = std::chrono::steady_clock::now();
        mm3_lm_set_adapter(&g_mm3_lm, opt.lm_adapter, opt.lm_adapter_scales);
        if (!mm3_lm_prefill(m, &g_mm3_lm, cond_ids, uncond_ids, n_prompt, hidden.data(), logits.data(), err)) {
            return false;
        }
        out->prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    out->prefill_hidden.assign(hidden.begin(), hidden.end());

    // Candidate layout: index 0 is EOS, indices 1..SV are semantic codes 0..SV-1.
    // Everything else in the 200000-entry vocabulary is masked and never gathered.
    const int64_t      NCAND = SV + 1;
    std::vector<float> cand_cond((size_t) NCAND);
    std::vector<float> cand_unc((size_t) NCAND);
    std::vector<float> cand_guided((size_t) NCAND);
    std::vector<float> sel_scratch;
    std::vector<float> samp_scratch;

    std::mt19937_64      rng(opt.seed);
    std::vector<int32_t> ac_rows((size_t) NC);
    MM3DepthFrame        frame;

    out->semantic_all.reserve((size_t) (max_frames + 1));
    out->acoustic_all.reserve((size_t) ((max_frames + 1) * NC));
    if (opt.collect_hiddens) {
        out->frame_hiddens.reserve((size_t) (max_frames * (NC + 1) * H));
    }

    for (int64_t it = 0; it <= max_frames; it++) {
        if (forced && it >= opt.forced_len) {
            break;
        }
        const auto t_host0 = std::chrono::steady_clock::now();

        // ── gather the candidate logits, both rows ──
        // Rows are indexed relative to hlo — 0 for the full head, the span
        // start under the head slice.
        const float * lrow_c = logits.data();
        const float * lrow_u = logits.data() + hn;
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

        // ── CFG, then the first top-k filter (ranked by the CONDITIONAL row) ──
        for (int64_t i = 0; i < NCAND; i++) {
            const float u          = cand_unc[(size_t) i];
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

        // ── sample (or replay) ──
        int32_t semantic;
        if (forced) {
            semantic = opt.forced_semantic[it];
            if (semantic < 0 || (int64_t) semantic >= SV) {
                if (err) {
                    *err = "forced semantic code " + std::to_string(semantic) + " at iteration " +
                           std::to_string((long long) it) + " is outside [0, " + std::to_string((long long) SV) + ")";
                }
                return false;
            }
        } else {
            const int64_t idx = mm3_sample_top_k(cand_guided.data(), NCAND, TOPK, rng, &samp_scratch);
            if (idx == 0) {
                out->eos_hit = true;
                out->host_ms +=
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();
                break;
            }
            semantic = (int32_t) (idx - 1);
        }
        out->host_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_host0).count();

        // ── depth decoder: the seven acoustic codes and their hidden states ──
        // Semantic-only forcing keeps the RNG alive so the depth decoder samples
        // the acoustic codebooks; full forcing passes null and replays them.
        const int32_t * forced_ac = forced_ac_given ? opt.forced_acoustic + it * NC : nullptr;
        if (!mm3_depth_decode_frame(m, hidden.data(), hidden.data() + H, semantic, forced_ac, &frame, err,
                                    forced_ac_given ? nullptr : &rng, TOPK)) {
            return false;
        }
        out->depth_ms += frame.ms;
        if (frame.n_codes != (int) NC) {
            if (err) {
                *err = "the depth decoder returned " + std::to_string(frame.n_codes) + " codes, expected " +
                       std::to_string((long long) NC);
            }
            return false;
        }

        out->semantic_all.push_back(semantic);
        for (int64_t i = 0; i < NC; i++) {
            out->acoustic_all.push_back(frame.codes[i]);
        }
        out->n_iterations++;

        const bool dumping = (int64_t) out->dumps.size() < opt.dump_iters;
        if (dumping) {
            MM3ArDump d;
            d.last_hidden.assign(hidden.begin(), hidden.end());
            d.sem_logits.resize((size_t) (2 * SV));
            memcpy(d.sem_logits.data(), lrow_c + (OFF - hlo), (size_t) SV * sizeof(float));
            memcpy(d.sem_logits.data() + SV, lrow_u + (OFF - hlo), (size_t) SV * sizeof(float));
            d.guided.assign(cand_guided.begin() + 1, cand_guided.end());
            d.feedback.assign((size_t) (2 * H), 0.0f);
            d.depth_hidden = frame.hiddens;
            out->dumps.push_back(std::move(d));
        }

        // ── emit (iteration 0 is fed back but never emitted) ──
        if (it > 0) {
            if (opt.collect_hiddens) {
                out->frame_hiddens.insert(out->frame_hiddens.end(), hidden.begin(), hidden.begin() + H);
                out->frame_hiddens.insert(out->frame_hiddens.end(), frame.hiddens.begin(), frame.hiddens.end());
            }
            out->n_frames++;
            if (opt.on_frame) {
                opt.on_frame(out->n_frames, max_frames);
            }
            if (opt.should_cancel && opt.should_cancel()) {
                if (err) {
                    *err = MM3_ERR_CANCELLED;
                }
                return false;
            }
            if (out->n_frames >= max_frames) {
                break;  // note 2: no feedback, no decode step, for the last frame
            }
        }

        // ── feed back and advance ──
        for (int64_t i = 0; i < NC; i++) {
            ac_rows[(size_t) i] = frame.codes[i] + (int32_t) (i * AV);
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            if (!mm3_lm_decode(m, &g_mm3_lm, semantic + (int32_t) OFF, ac_rows.data(), hidden.data(), logits.data(),
                               feedback.data(), err)) {
                return false;
            }
            out->lm_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            out->lm_steps++;
        }
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

    out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();

    if (out->n_frames == 0) {
        if (err) {
            *err = out->eos_hit ? "the LM emitted EOS on the first iteration; zero audio frames were generated"
                                : "zero audio frames were generated";
        }
        return false;
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
    if (opt.want_lrc && opt.tok != nullptr && !g_mm3_lm.align_capture && !align_dump && lyr0 >= 0 && lyr1 > lyr0 &&
        out->n_frames > 1) {
        // Column count must match what the live path would have captured: one
        // per decode step — every iteration that pushed codes also fed back,
        // except the last when the max-frames break fired before its feedback.
        const int64_t n_steps = out->eos_hit ? out->n_iterations : out->n_iterations - 1;
        if (n_steps > 1) {
            std::vector<float> flat;
            std::string        rerr;
            const auto         t_lrc = std::chrono::steady_clock::now();
            if (mm3_lm_lrc_replay(m, &g_mm3_lm, cond_ids, n_prompt, out->semantic_all.data(),
                                  out->acoustic_all.data(), n_steps, lyr0, lyr1, &flat, &rerr)) {
                const float dur =
                    m.lm_cfg.frame_rate > 0 ? (float) out->n_frames / (float) m.lm_cfg.frame_rate : 0.0f;
                const std::vector<int> ids(cond_ids + lyr0, cond_ids + lyr1);
                out->lrc = mm3_align_build_lrc(flat, ids, opt.tok->bpe, (int) n_steps, dur);
                const double lrc_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lrc).count();
                fprintf(stderr, "[MM3-LRC] %s via replay (%lld frames, %.1f s, %.0f ms)\n",
                        out->lrc.empty() ? "no alignment produced" : "LRC built", (long long) n_steps, dur, lrc_ms);
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
