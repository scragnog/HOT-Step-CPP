// pipeline-lm.cpp: ACE-Step LM pipeline implementation
//
// Wraps Qwen3 LM for caption enrichment and audio code generation.
// Supports GGML, TRT (raw NvInfer), and TRT-LLM (Executor) backends.

#include "pipeline-lm.h"

#include "bpe.h"
#include "metadata-fsm.h"
#include "model-store.h"
#include "prompt.h"
#include "qwen3-lm.h"
#include "sampling.h"
#include "timer.h"

#ifdef HOT_STEP_TRT
#include "lm-trt.h"
#endif

#ifdef HOT_STEP_TRTLLM
#include "lm-trtllm.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <filesystem>

struct AceLm {
    ModelStore * store;
    AceLmParams  params;
    ModelKey     lm_key;

    // Speculative decoding: lightweight draft model (0.6B)
    // Loaded once, kept resident, bypasses ModelStore.
    Qwen3LM    draft;
    bool       draft_loaded;

#ifdef HOT_STEP_TRT
    // TRT LM (alternative to GGML path)
    LmTrt      lm_trt;
    bool       use_trt = false;
#endif

#ifdef HOT_STEP_TRTLLM
    // TRT-LLM Executor (high-performance alternative)
    LmTrtLlm   lm_trtllm;
    bool        use_trtllm = false;
#endif
};

// ── Dispatch wrappers: GGML vs TRT ─────────────────────────────────────────
// These allow the pipeline to call a single API regardless of backend.
// When TRT is active, the GGML model pointer is unused (nullptr).

#ifdef HOT_STEP_TRT
static bool s_use_trt = false;  // set during ace_lm_load, read during generate
static LmTrt * s_trt_ctx = nullptr;
#endif

#ifdef HOT_STEP_TRTLLM
static bool s_use_trtllm = false;
static LmTrtLlm * s_trtllm_ctx = nullptr;
#endif

// Safe vocab_size accessor: works with either GGML (m != null) or TRT (m == null)
static inline int lm_vocab_size(Qwen3LM * m) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) return LM_TRT_VOCAB;
#endif
    return m->cfg.vocab_size;
}

// Safe partial head check: TRT never has a partial GPU head (slicing is in C++)
static inline bool lm_has_partial_head(Qwen3LM * m) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) return false;
#endif
    return m && m->lm_head_phase2 != NULL;
}

static inline void lm_reset_kv(Qwen3LM * m, int kv_set) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) { lm_trt_reset_kv(s_trt_ctx, kv_set); return; }
#endif
    qw3lm_reset_kv(m, kv_set);
}

static inline void lm_copy_kv(Qwen3LM * m, int src, int dst) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) { lm_trt_copy_kv(s_trt_ctx, src, dst); return; }
#endif
    qw3lm_copy_kv(m, src, dst);
}

static inline void lm_forward(Qwen3LM * m, const int * tokens, int n, int kv_set, float * logits) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) { lm_trt_forward(s_trt_ctx, tokens, n, kv_set, logits, 0); return; }
#endif
    qw3lm_forward(m, tokens, n, kv_set, logits);
}

static inline void lm_forward_batch(Qwen3LM * m, const int * tokens, const int * sets,
                                     int N, float * logits, int lm_offset = 0, int lm_count = 0) {
#ifdef HOT_STEP_TRT
    if (s_use_trt) {
        // TRT: partial_offset maps to lm_offset
        lm_trt_forward_batch(s_trt_ctx, tokens, sets, N, logits, lm_offset);
        return;
    }
#endif
    qw3lm_forward_batch(m, tokens, sets, N, logits, lm_offset, lm_count);
}

// Batched Phase 1: N text generations with shared prompt, different seeds.
// No CFG. Each element gets its own FSM state and RNG.
// Returns N generated text strings.
static std::vector<std::string> generate_phase1_batch(Qwen3LM *                m,
                                                      BPETokenizer *           bpe,
                                                      const std::vector<int> & prompt_tokens,
                                                      int                      max_new_tokens,
                                                      float                    temperature,
                                                      float                    top_p,
                                                      int                      top_k,
                                                      uint32_t                 base_seed,
                                                      int                      N,
                                                      MetadataFSM *            fsm_template,
                                                      bool                     lyrics_mode,
                                                      float                    cfg_scale         = 1.0f,
                                                      const std::vector<int> * uncond_tokens     = nullptr,
                                                      bool                     stop_at_reasoning = false,
                                                      bool (*cancel)(void *)                     = nullptr,
                                                      void * cancel_data                         = nullptr) {
    int  V       = lm_vocab_size(m);
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens && !uncond_tokens->empty();

    // KV sets: cond [0..N-1], uncond [N..2N-1] if CFG
    for (int i = 0; i < N; i++) {
        lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            lm_reset_kv(m, N + i);
        }
    }

    // Prefill cond once, set 0, copy to 1..N-1
    Timer              t_prefill;
    std::vector<float> prefill_logits(V);
    lm_forward(m, prompt_tokens.data(), (int) prompt_tokens.size(), 0, prefill_logits.data());
    for (int i = 1; i < N; i++) {
        lm_copy_kv(m, 0, i);
    }

    // Prefill uncond once, set N, copy to N+1..2N-1
    std::vector<float> prefill_logits_uncond(V);
    if (use_cfg) {
        lm_forward(m, uncond_tokens->data(), (int) uncond_tokens->size(), N, prefill_logits_uncond.data());
        for (int i = 1; i < N; i++) {
            lm_copy_kv(m, N, N + i);
        }
    }

    fprintf(stderr, "[LM-Phase1] Prefill %.0fms, %zu tokens, N=%d, CFG=%.2f\n", t_prefill.ms(), prompt_tokens.size(), N,
            cfg_scale);

    // Per-element state
    struct P1Seq {
        std::mt19937     rng;
        MetadataFSM      fsm;
        std::vector<int> gen_tokens;
        int              last_token;
        bool             codes_phase;
        bool             done;
    };

    std::vector<P1Seq> seqs(N);

    // Sample first token from shared prefill logits
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed(base_seed + i);
        if (fsm_template) {
            seqs[i].fsm = *fsm_template;
        }
        seqs[i].codes_phase = false;
        seqs[i].done        = false;

        std::vector<float> lg(prefill_logits);
        if (use_cfg) {
            for (int v = 0; v < V; v++) {
                lg[v] = prefill_logits_uncond[v] + cfg_scale * (lg[v] - prefill_logits_uncond[v]);
            }
        }
        if (fsm_template && fsm_template->enabled) {
            seqs[i].fsm.apply_mask(lg.data());
        }

        int tok = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else {
            if (fsm_template && fsm_template->enabled) {
                seqs[i].fsm.update(tok);
            }
            if (tok == TOKEN_THINK_END) {
                seqs[i].codes_phase = true;
                if (stop_at_reasoning) {
                    seqs[i].done = true;
                }
            }
            seqs[i].gen_tokens.push_back(tok);
        }
        seqs[i].last_token = tok;
    }

    // KV set arrays + merged CFG arrays
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i]   = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode
    Timer              t_decode;
    std::vector<float> logits_cond(V * N);
    std::vector<float> logits_uncond(V * N);
    std::vector<int>   tokens(N);

    // CFG: single forward with 2*N (cond + uncond)
    int                N2 = use_cfg ? 2 * N : N;
    std::vector<int>   tokens_2n(N2), sets_2n(N2);
    std::vector<float> logits_2n((size_t) V * N2);
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            sets_2n[i]     = cond_sets[i];
            sets_2n[N + i] = uncond_sets[i];
        }
    }

    int n_active = N;
    for (int i = 0; i < N; i++) {
        if (seqs[i].done) {
            n_active--;
        }
    }

    for (int step = 0; step < max_new_tokens && n_active > 0; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[LM-Phase1] Cancelled at step %d\n", step);
            return {};
        }
        for (int i = 0; i < N; i++) {
            tokens[i] = seqs[i].last_token;
        }

        if (use_cfg) {
            // Single batched forward: cond[0..N-1] + uncond[N..2N-1]
            for (int i = 0; i < N; i++) {
                tokens_2n[i]     = tokens[i];
                tokens_2n[N + i] = tokens[i];
            }
            lm_forward_batch(m, tokens_2n.data(), sets_2n.data(), N2, logits_2n.data());
            memcpy(logits_cond.data(), logits_2n.data(), (size_t) V * N * sizeof(float));
            memcpy(logits_uncond.data(), logits_2n.data() + (size_t) V * N, (size_t) V * N * sizeof(float));
        } else {
            lm_forward_batch(m, tokens.data(), cond_sets.data(), N, logits_cond.data());
        }

        for (int i = 0; i < N; i++) {
            if (seqs[i].done) {
                continue;
            }

            float * lc = logits_cond.data() + (size_t) i * V;

            // CFG combine
            if (use_cfg) {
                float * lu = logits_uncond.data() + (size_t) i * V;
                for (int v = 0; v < V; v++) {
                    lc[v] = lu[v] + cfg_scale * (lc[v] - lu[v]);
                }
            }

            // FSM mask (before </think>)
            if (fsm_template && seqs[i].fsm.enabled && !seqs[i].codes_phase) {
                seqs[i].fsm.apply_mask(lc);
            }

            // After </think>: audio code constraint unless lyrics_mode
            if (seqs[i].codes_phase && !lyrics_mode) {
                for (int v = TOKEN_IM_END + 1; v < AUDIO_CODE_BASE; v++) {
                    lc[v] = -1e9f;
                }
            }

            int tok;
            if (seqs[i].codes_phase && !lyrics_mode) {
                // Restricted sampling: only [TOKEN_IM_END..V)
                int V_eff = V - TOKEN_IM_END;
                tok = sample_top_k_p(lc + TOKEN_IM_END, V_eff, temperature, top_p, top_k, seqs[i].rng) + TOKEN_IM_END;
            } else {
                tok = sample_top_k_p(lc, V, temperature, top_p, top_k, seqs[i].rng);
            }

            if (tok == TOKEN_IM_END) {
                seqs[i].done = true;
                n_active--;
            } else {
                if (seqs[i].fsm.enabled && !seqs[i].codes_phase) {
                    seqs[i].fsm.update(tok);
                }
                if (tok == TOKEN_THINK_END && !seqs[i].codes_phase) {
                    seqs[i].codes_phase = true;
                    if (stop_at_reasoning) {
                        seqs[i].gen_tokens.push_back(tok);
                        seqs[i].done = true;
                        n_active--;
                        continue;
                    }
                }
                seqs[i].gen_tokens.push_back(tok);
            }
            seqs[i].last_token = tok;
        }

        if ((step + 1) % 100 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[LM-Phase1] Step %d, %d active, %.1f tok/s\n", step + 1, n_active,
                    (double) (step + 1) * N / elapsed);
        }
    }

    fprintf(stderr, "[LM-Phase1] Decode %.0fms\n", t_decode.ms());

    // Decode tokens to text
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = bpe_decode(*bpe, seqs[i].gen_tokens);
        fprintf(stderr, "[LM-Phase1 Batch%d] seed=%u, %zu tokens\n", i, base_seed + i, seqs[i].gen_tokens.size());
    }
    return results;
}

// Batched Phase 2: N sequences with potentially different prompts.
// aces.size() == N: each element gets its own lyrics/metadata.
// aces.size() == 1: single prompt replicated for all N (prefill once, copy KV).
// Returns N code strings. Seeds = base_seed + 0, 1, ..., N-1.
static std::vector<std::string> run_phase2_batch(Qwen3LM *                      m,
                                                 BPETokenizer &                 bpe,
                                                 const std::vector<AcePrompt> & aces,
                                                 float                          temperature,
                                                 float                          top_p,
                                                 int                            top_k,
                                                 uint32_t                       base_seed,
                                                 int                            N,
                                                 float                          cfg_scale,
                                                 float                          cfg_cutoff_ratio,
                                                 const char *                   negative_prompt,
                                                 bool                           use_batch_cfg,
                                                 bool (*cancel)(void *),
                                                 void * cancel_data) {
    int  V             = lm_vocab_size(m);
    bool use_cfg       = cfg_scale > 1.0f;
    bool shared_prompt = ((int) aces.size() == 1);

    // Build per-element prompts
    std::vector<std::vector<int>> prompts(N), unconds(N);
    int                           max_tokens = 0;
    for (int i = 0; i < N; i++) {
        const AcePrompt & a   = shared_prompt ? aces[0] : aces[i];
        std::string       cot = build_cot_yaml(a);
        if (i == 0) {
            fprintf(stderr, "[LM-Phase2] N=%d, CoT[0]:\n%s", N, cot.c_str());
        }
        prompts[i] = build_lm_prompt_with_cot(bpe, a, cot);
        if (use_cfg) {
            unconds[i] = build_lm_prompt_uncond_with_cot(bpe, a, negative_prompt);
        }
        int mt = (int) (a.duration * 5) + 100;
        if (mt > max_tokens) {
            max_tokens = mt;
        }
    }
    fprintf(stderr, "[LM-Phase2] max_tokens: %d, CFG: %.2f, seeds: %u..%u\n", max_tokens, cfg_scale, base_seed,
            base_seed + N - 1);
    if (cfg_cutoff_ratio < 1.0f) {
        fprintf(stderr, "[LM-Phase2] CFG cutoff: ratio=%.2f (CFG for first %d/%d tokens)\n",
                cfg_cutoff_ratio, (int)(max_tokens * cfg_cutoff_ratio), max_tokens);
    }

    // Reset all KV sets: cond [0..N-1], uncond [N..2N-1]
    for (int i = 0; i < N; i++) {
        lm_reset_kv(m, i);
    }
    if (use_cfg) {
        for (int i = 0; i < N; i++) {
            lm_reset_kv(m, N + i);
        }
    }

    // Prefill: if shared prompt, prefill once + copy KV. Otherwise prefill each.
    Timer                           t_prefill;
    std::vector<std::vector<float>> prefill_logits_vec(N, std::vector<float>(V));

    if (shared_prompt) {
        lm_forward(m, prompts[0].data(), (int) prompts[0].size(), 0, prefill_logits_vec[0].data());
        for (int i = 1; i < N; i++) {
            lm_copy_kv(m, 0, i);
            prefill_logits_vec[i] = prefill_logits_vec[0];
        }
    } else {
        for (int i = 0; i < N; i++) {
            lm_forward(m, prompts[i].data(), (int) prompts[i].size(), i, prefill_logits_vec[i].data());
        }
    }

    // Prefill uncond
    std::vector<std::vector<float>> prefill_logits_uncond_vec(N, std::vector<float>(V));
    if (use_cfg) {
        if (shared_prompt) {
            lm_forward(m, unconds[0].data(), (int) unconds[0].size(), N, prefill_logits_uncond_vec[0].data());
            for (int i = 1; i < N; i++) {
                lm_copy_kv(m, N, N + i);
                prefill_logits_uncond_vec[i] = prefill_logits_uncond_vec[0];
            }
        } else {
            for (int i = 0; i < N; i++) {
                lm_forward(m, unconds[i].data(), (int) unconds[i].size(), N + i,
                              prefill_logits_uncond_vec[i].data());
            }
        }
    }

    double prefill_ms = t_prefill.ms();
    fprintf(stderr, "[LM-Phase2] Prefill %.0fms (%s)\n", prefill_ms,
            shared_prompt ? "shared, 1 cond + 1 uncond" : "individual, N cond + N uncond");

    // Per-sequence state
    struct BatchSeq {
        std::mt19937     rng;
        std::vector<int> audio_codes;
        int              last_token;
        bool             done;
    };

    std::vector<BatchSeq> seqs(N);

    // Sample first token from per-element prefill logits (N different seeds)
    for (int i = 0; i < N; i++) {
        seqs[i].rng.seed(base_seed + i);
        seqs[i].done = false;

        std::vector<float> lg(prefill_logits_vec[i]);  // copy
        if (use_cfg) {
            float * lu = prefill_logits_uncond_vec[i].data();
            for (int v = 0; v < V; v++) {
                lg[v] = lu[v] + cfg_scale * (lg[v] - lu[v]);
            }
        }
        // Only audio codes + EOS (codes_phase = true from start)
        for (int v = 0; v < AUDIO_CODE_BASE; v++) {
            if (v != TOKEN_IM_END) {
                lg[v] = -1e9f;
            }
        }

        int tok            = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, seqs[i].rng);
        seqs[i].last_token = tok;

        if (tok == TOKEN_IM_END) {
            seqs[i].done = true;
        } else if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            seqs[i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
        }
    }

    // KV set arrays for batched forward
    std::vector<int> cond_sets(N), uncond_sets(N);
    for (int i = 0; i < N; i++) {
        cond_sets[i]   = i;
        uncond_sets[i] = N + i;
    }

    // Batched decode loop.
    // partial head: pre-extracted contiguous tensor for [TOKEN_IM_END..V) rows.
    // When unavailable (alloc failed): full vocab, slightly more compute, same result.
    Timer t_decode;
    bool  partial     = lm_has_partial_head(m);
    int   out_V       = partial ? (V - TOKEN_IM_END) : V;
    int   lm_offset   = partial ? TOKEN_IM_END : 0;
    int   lm_count    = partial ? (V - TOKEN_IM_END) : 0;
    int   eos_idx     = partial ? 0 : TOKEN_IM_END;
    int   code_offset = partial ? (AUDIO_CODE_BASE - TOKEN_IM_END) : AUDIO_CODE_BASE;

    // Pre-allocate batched arrays for the maximum possible size (N or 2*N for CFG)
    int                max_N2 = use_cfg ? 2 * N : N;
    std::vector<int>   batch_tokens(max_N2);
    std::vector<int>   batch_sets(max_N2);
    std::vector<float> batch_logits((size_t) out_V * max_N2);

    // This array maps the compact "active" index back to the original sequence index (0 to N-1)
    std::vector<int> active_to_orig(N);

    // Tiny array for CPU sampling (EOS token + Audio Codes) to prevent sorting 150,000 text logits
    int                compact_V = AUDIO_CODE_COUNT + 1;
    std::vector<float> compact_logits(compact_V);

    int n_active = N;
    for (int i = 0; i < N; i++) {
        if (seqs[i].done) {
            n_active--;
        }
    }

    for (int step = 0; step < max_tokens && n_active > 0; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[LM-Phase2] Cancelled at step %d\n", step);
            return {};
        }
        int current_active = 0;

        // CFG cutoff: stop doing CFG after this ratio of steps
        bool do_cfg_this_step = use_cfg && (cfg_cutoff_ratio >= 1.0f || step < (int)(max_tokens * cfg_cutoff_ratio));

        // 1. DYNAMIC COMPACTION: Loop through all N sequences, but only gather the active ones!
        for (int i = 0; i < N; i++) {
            if (!seqs[i].done) {
                active_to_orig[current_active] = i;  // Remember that this slot belongs to sequence 'i'

                if (do_cfg_this_step) {
                    // Place the Cond token/set in the first half
                    batch_tokens[current_active] = seqs[i].last_token;
                    batch_sets[current_active]   = cond_sets[i];

                    // Place the Uncond token/set exactly n_active elements later
                    batch_tokens[n_active + current_active] = seqs[i].last_token;
                    batch_sets[n_active + current_active]   = uncond_sets[i];
                } else {
                    batch_tokens[current_active] = seqs[i].last_token;
                    batch_sets[current_active]   = cond_sets[i];
                }
                current_active++;
            }
        }

        // 2. FORWARD PASS: GPU only computes attention for n_active sequences
        if (do_cfg_this_step && !use_batch_cfg) {
            // Two separate N=1 forwards (cond, then uncond).
            // Workaround for backends where batched multi-sequence attention
            // produces wrong results (e.g. ROCm/gfx1201). Same logit layout.
            lm_forward_batch(m, batch_tokens.data(), batch_sets.data(), n_active, batch_logits.data(), lm_offset,
                                lm_count);
            lm_forward_batch(m, batch_tokens.data() + n_active, batch_sets.data() + n_active, n_active,
                                batch_logits.data() + (size_t) n_active * out_V, lm_offset, lm_count);
        } else {
            int actual_batch_size = do_cfg_this_step ? (2 * n_active) : n_active;
            lm_forward_batch(m, batch_tokens.data(), batch_sets.data(), actual_batch_size, batch_logits.data(),
                                lm_offset, lm_count);
        }

        // 3. TARGETED CFG & LOGIT EXTRACTION
        for (int a = 0; a < n_active; a++) {
            int orig_i = active_to_orig[a];  // Map back to original sequence object

            // Pointer to the conditional logits for THIS active sequence
            float * lc = batch_logits.data() + (size_t) a * out_V;

            if (do_cfg_this_step) {
                // Pointer to the unconditional logits (offset by n_active)
                float * lu = batch_logits.data() + (size_t) (n_active + a) * out_V;

                // Targeted CFG Math: Only apply it to EOS + Audio Codes. Skip the 150,000 text tokens!
                lc[eos_idx] = lu[eos_idx] + cfg_scale * (lc[eos_idx] - lu[eos_idx]);  // EOS token
                for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                    int idx = code_offset + c;
                    lc[idx] = lu[idx] + cfg_scale * (lc[idx] - lu[idx]);
                }
            }

            // Extract ONLY the valid target tokens into the tiny compact array
            compact_logits[0] = lc[eos_idx];
            for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                compact_logits[c + 1] = lc[code_offset + c];
            }

            // CPU samples instantly because it only has to sort ~2049 items instead of 150,000+
            int compact_tok =
                sample_top_k_p(compact_logits.data(), compact_V, temperature, top_p, top_k, seqs[orig_i].rng);

            // Map the sampled index back to global vocabulary ID
            int tok = (compact_tok == 0) ? TOKEN_IM_END : (AUDIO_CODE_BASE + compact_tok - 1);

            seqs[orig_i].last_token = tok;

            if (tok == TOKEN_IM_END) {
                seqs[orig_i].done = true;
            } else {
                seqs[orig_i].audio_codes.push_back(tok - AUDIO_CODE_BASE);
            }
        }

        // 4. UPDATE ACTIVE COUNT for the next loop iteration
        int next_active_count = 0;
        int total_codes       = 0;
        for (int i = 0; i < N; i++) {
            if (!seqs[i].done) {
                next_active_count++;
            }
            total_codes += (int) seqs[i].audio_codes.size();
        }
        n_active = next_active_count;

        if ((step + 1) % 50 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[LM-Phase2] Step %d, %d active, %d total codes, %.1f tok/s\n", step + 1, n_active,
                    total_codes, (double) (step + 1) * N / elapsed);
        }
    }

    double decode_ms = t_decode.ms();
    fprintf(stderr, "[LM-Phase2] Decode %.0fms\n", decode_ms);

    // Build results
    std::vector<std::string> results(N);
    for (int i = 0; i < N; i++) {
        results[i] = codes_to_string(seqs[i].audio_codes);
        fprintf(stderr, "[LM-Phase2 Batch%d] seed=%u, %zu codes\n", i, base_seed + i, seqs[i].audio_codes.size());
    }
    return results;
}

// Speculative decoding: Phase 2 audio code generation with draft model.
// draft = 0.6B model (fast, no CFG), target = 4B model (verifies with CFG).
// N=1 only (no batched spec decode). Returns 1-element vector.
static std::vector<std::string> run_phase2_speculative(Qwen3LM *      target,
                                                        Qwen3LM *      draft,
                                                        BPETokenizer & bpe,
                                                        const AcePrompt & ace,
                                                        float          temperature,
                                                        float          top_p,
                                                        int            top_k,
                                                        uint32_t       seed,
                                                        float          cfg_scale,
                                                        float          cfg_cutoff_ratio,
                                                        const char *   negative_prompt,
                                                        bool (*cancel)(void *),
                                                        void * cancel_data) {
    int  V             = lm_vocab_size(target);
    bool use_cfg       = cfg_scale > 1.0f;

    // Build prompts
    std::string       cot    = build_cot_yaml(ace);
    std::vector<int>  prompt = build_lm_prompt_with_cot(bpe, ace, cot);
    int max_tokens = (int)(ace.duration * 5) + 100;

    fprintf(stderr, "[LM-Phase2] Speculative decode: draft=%dL/%d, target=%dL/%d, max_tokens=%d, CFG=%.2f\n",
            draft->cfg.n_layers, draft->cfg.hidden_size,
            target->cfg.n_layers, target->cfg.hidden_size,
            max_tokens, cfg_scale);

    // Reset KV caches: target cond=0, uncond=1; draft cond=0
    lm_reset_kv(target, 0);
    if (use_cfg) lm_reset_kv(target, 1);
    qw3lm_reset_kv(draft, 0);  // draft stays on GGML always

    // Prefill target (cond)
    Timer t_prefill;
    std::vector<float> prefill_logits(V);
    lm_forward(target, prompt.data(), (int) prompt.size(), 0, prefill_logits.data());

    // Prefill target (uncond)
    std::vector<float> prefill_logits_uncond(V);
    if (use_cfg) {
        std::vector<int> uncond = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);
        lm_forward(target, uncond.data(), (int) uncond.size(), 1, prefill_logits_uncond.data());
    }

    // Prefill draft (cond only, no CFG)
    std::vector<float> draft_prefill_logits(V);
    qw3lm_forward(draft, prompt.data(), (int) prompt.size(), 0, draft_prefill_logits.data());

    fprintf(stderr, "[LM-Phase2] Prefill %.0fms (target cond+uncond + draft)\n", t_prefill.ms());

    // Sample first token from target's CFG-combined prefill logits
    std::mt19937 rng(seed);
    std::vector<int> audio_codes;
    int last_token;

    {
        std::vector<float> lg(prefill_logits);
        if (use_cfg) {
            for (int v = 0; v < V; v++) {
                lg[v] = prefill_logits_uncond[v] + cfg_scale * (lg[v] - prefill_logits_uncond[v]);
            }
        }
        for (int v = 0; v < AUDIO_CODE_BASE; v++) {
            if (v != TOKEN_IM_END) lg[v] = -1e9f;
        }
        int tok = sample_top_k_p(lg.data(), V, temperature, top_p, top_k, rng);
        last_token = tok;
        if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            audio_codes.push_back(tok - AUDIO_CODE_BASE);
        }
    }

    if (last_token == TOKEN_IM_END) {
        fprintf(stderr, "[LM-Phase2] EOS on first token\n");
        return { codes_to_string(audio_codes) };
    }

    // Partial LM head setup (same for both models)
    bool  t_partial   = lm_has_partial_head(target);
    int   t_out_V     = t_partial ? (V - TOKEN_IM_END) : V;
    int   t_lm_offset = t_partial ? TOKEN_IM_END : 0;
    int   t_lm_count  = t_partial ? (V - TOKEN_IM_END) : 0;
    int   t_eos_idx     = t_partial ? 0 : TOKEN_IM_END;
    int   t_code_offset = t_partial ? (AUDIO_CODE_BASE - TOKEN_IM_END) : AUDIO_CODE_BASE;

    int compact_V = AUDIO_CODE_COUNT + 1;

    // Adaptive K parameters
    int   K     = 8;
    int   K_MAX = 12;     // K_MIN effectively 6 via initial value; K only increases
    float alpha = 0.7f;   // rolling draft-match rate (not including correction tokens)

    // Statistics
    int total_matched     = 0;  // draft tokens that agreed with target
    int total_committed   = 0;  // tokens committed (matched + corrections)
    int total_drafted     = 0;
    int total_iterations  = 0;
    int total_target_fwd  = 0;
    double total_draft_ms = 0;
    double total_verify_ms = 0;

    Timer t_decode;

    while ((int) audio_codes.size() < max_tokens) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[LM-Phase2] Cancelled\n");
            return {};
        }

        // Save KV positions for rollback
        int target_saved_pos  = target->kv_pos[0];
        int draft_saved_pos   = draft->kv_pos[0];

        // ── 1. DRAFT: generate K tokens with 0.6B (no CFG) ──
        Timer t_draft_phase;
        int draft_last = last_token;
        std::vector<int>   draft_tokens(K);
        std::vector<float> draft_logits_buf(V);  // full vocab (qw3lm_forward always returns V)
        int actual_K = K;

        for (int k = 0; k < K; k++) {
            qw3lm_forward(draft, &draft_last, 1, 0, draft_logits_buf.data());

            // Extract compact logits (EOS + audio codes) from full vocab
            std::vector<float> compact(compact_V);
            compact[0] = draft_logits_buf[TOKEN_IM_END];
            for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                compact[c + 1] = draft_logits_buf[AUDIO_CODE_BASE + c];
            }

            // Use a copy of rng state that we DON'T advance — draft sampling
            // uses a throwaway rng. Target verification uses the real rng.
            std::mt19937 draft_rng(rng() + k);
            int compact_tok = sample_top_k_p(compact.data(), compact_V, temperature, top_p, top_k, draft_rng);
            int tok = (compact_tok == 0) ? TOKEN_IM_END : (AUDIO_CODE_BASE + compact_tok - 1);

            draft_tokens[k] = tok;
            draft_last = tok;

            if (tok == TOKEN_IM_END) {
                // Draft predicted EOS — truncate to k+1
                actual_K = k + 1;
                break;
            }
        }
        total_drafted += actual_K;
        total_draft_ms += t_draft_phase.ms();

        // ── 2. VERIFY: forward all K draft tokens through target (single pass) ──
        Timer t_verify_phase;
        std::vector<float> target_cond_logits((size_t) t_out_V * actual_K);
        qw3lm_forward_verify(target, draft_tokens.data(), actual_K, 0, target_cond_logits.data(),
                             t_lm_offset, t_lm_count);

        // Target unconditional forward (for CFG)
        std::vector<float> target_uncond_logits;
        bool do_cfg = use_cfg && (cfg_cutoff_ratio >= 1.0f ||
                      (int) audio_codes.size() < (int)(max_tokens * cfg_cutoff_ratio));
        if (do_cfg) {
            target_uncond_logits.resize((size_t) t_out_V * actual_K);
            qw3lm_forward_verify(target, draft_tokens.data(), actual_K, 1, target_uncond_logits.data(),
                                 t_lm_offset, t_lm_count);
            total_target_fwd += 2;
        } else {
            total_target_fwd += 1;
        }
        total_verify_ms += t_verify_phase.ms();

        // ── 3. ACCEPT/REJECT: compare draft vs target ──
        int n_matched   = 0;  // draft tokens that agreed with target
        int n_committed = 0;  // total tokens to commit (matched + 1 correction)
        bool hit_eos = false;

        for (int k = 0; k < actual_K; k++) {
            float * lc = target_cond_logits.data() + (size_t) k * t_out_V;

            // CFG combine for this position
            if (do_cfg) {
                float * lu = target_uncond_logits.data() + (size_t) k * t_out_V;
                lc[t_eos_idx] = lu[t_eos_idx] + cfg_scale * (lc[t_eos_idx] - lu[t_eos_idx]);
                for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                    int idx = t_code_offset + c;
                    lc[idx] = lu[idx] + cfg_scale * (lc[idx] - lu[idx]);
                }
            }

            // Compact sampling (EOS + audio codes)
            std::vector<float> compact(compact_V);
            compact[0] = lc[t_eos_idx];
            for (int c = 0; c < AUDIO_CODE_COUNT; c++) {
                compact[c + 1] = lc[t_code_offset + c];
            }

            int compact_tok = sample_top_k_p(compact.data(), compact_V, temperature, top_p, top_k, rng);
            int target_tok  = (compact_tok == 0) ? TOKEN_IM_END : (AUDIO_CODE_BASE + compact_tok - 1);

            if (target_tok == draft_tokens[k]) {
                // ACCEPT: draft and target agree
                n_matched++;
                n_committed++;
                if (target_tok == TOKEN_IM_END) {
                    hit_eos = true;
                    break;
                }
                audio_codes.push_back(target_tok - AUDIO_CODE_BASE);
                if ((int) audio_codes.size() >= max_tokens) break;
            } else {
                // REJECT: use target's correction token, stop
                n_committed++;  // correction token counts for KV advance, NOT for alpha
                if (target_tok == TOKEN_IM_END) {
                    hit_eos = true;
                    break;
                }
                audio_codes.push_back(target_tok - AUDIO_CODE_BASE);
                break;
            }
        }

        total_matched += n_matched;
        total_committed += n_committed;
        total_iterations++;

        // Update last_token for next iteration
        if (!audio_codes.empty()) {
            last_token = AUDIO_CODE_BASE + audio_codes.back();
        } else {
            last_token = TOKEN_IM_END;
        }

        // ── 4. KV CACHE ROLLBACK ──
        // Target processed actual_K tokens but only n_committed are valid
        target->kv_pos[0] = target_saved_pos + n_committed;
        if (use_cfg) {
            target->kv_pos[1] = target_saved_pos + n_committed;
        }
        // Draft: rollback to match
        draft->kv_pos[0] = draft_saved_pos + n_committed;

        // ── 5. ADAPTIVE K ──
        // Track match rate (not including correction tokens)
        float match_rate = (float) n_matched / actual_K;
        alpha = 0.9f * alpha + 0.1f * match_rate;
        if (alpha > 0.85f) {
            K = std::min(K + 1, K_MAX);
        }
        // Don't decrease K below K_MIN — better to waste some draft than degenerate

        // Check for EOS
        if (hit_eos) break;

        // Progress logging
        if (total_iterations % 20 == 0) {
            double elapsed = t_decode.ms() / 1000.0;
            float avg_match = total_drafted > 0 ? (float) total_matched / total_drafted : 0;
            fprintf(stderr, "[LM-Phase2] %d codes, K=%d, α=%.2f, match=%.0f%%, %.1f tok/s, draft=%.0f/verify=%.0fms\n",
                    (int) audio_codes.size(), K, alpha, avg_match * 100,
                    (double) audio_codes.size() / elapsed,
                    total_draft_ms / total_iterations, total_verify_ms / total_iterations);
        }
    }

    double decode_ms = t_decode.ms();
    float avg_match = total_drafted > 0 ? (float) total_matched / total_drafted : 0;
    float tokens_per_iter = total_iterations > 0 ? (float) audio_codes.size() / total_iterations : 0;
    fprintf(stderr, "[LM-Phase2] Speculative decode: %zu codes in %.0fms (%.1f tok/s)\n",
            audio_codes.size(), decode_ms, (double) audio_codes.size() / (decode_ms / 1000.0));
    fprintf(stderr, "[LM-Phase2]   iterations=%d, drafted=%d, matched=%d (%.0f%%), per_iter=%.1f\n",
            total_iterations, total_drafted, total_matched, avg_match * 100, tokens_per_iter);
    fprintf(stderr, "[LM-Phase2]   target_fwd=%d (vs %zu non-spec), final K=%d\n",
            total_target_fwd, audio_codes.size(), K);
    fprintf(stderr, "[LM-Phase2]   avg draft=%.1fms, avg verify=%.1fms per iteration\n",
            total_draft_ms / std::max(1, total_iterations),
            total_verify_ms / std::max(1, total_iterations));

    return { codes_to_string(audio_codes) };
}

// Public API

void ace_lm_default_params(AceLmParams * p) {
    p->model_path       = NULL;
    p->max_seq          = 8192;
    p->max_batch        = 1;
    p->use_fsm          = true;
    p->use_fa           = true;
    p->use_batch_cfg    = true;
    p->clamp_fp16       = false;
    p->draft_model_path = NULL;
}

AceLm * ace_lm_load(ModelStore * store, const AceLmParams * params) {
    if (!store || !params || !params->model_path) {
        fprintf(stderr, "[Ace-LM] ERROR: store and model_path are required\n");
        return NULL;
    }

    AceLm * ctx = new AceLm();
    ctx->store  = store;
    ctx->params = *params;

    // KV sets sized for worst case: CFG needs 2x batch.
    ctx->lm_key.kind          = MODEL_LM;
    ctx->lm_key.path          = params->model_path;
    ctx->lm_key.max_seq       = params->max_seq;
    ctx->lm_key.n_kv_sets     = 2 * params->max_batch;
    ctx->lm_key.adapter_path  = "";
    ctx->lm_key.adapter_scale = 1.0f;

    fprintf(stderr, "[Ace-LM] Ready: path=%s, max_seq=%d, max_batch=%d, fa=%s, fsm=%s\n", params->model_path,
            params->max_seq, params->max_batch, params->use_fa ? "yes" : "no", params->use_fsm ? "yes" : "no");
    if (!params->use_batch_cfg) {
        fprintf(stderr, "[Ace-LM] Batched CFG disabled (split N=1 forwards)\n");
    }
    if (params->clamp_fp16) {
        fprintf(stderr, "[Ace-LM] FP16 clamp enabled\n");
    }

    // ── TRT LM detection ────────────────────────────────────────────────────
    // If model_path points to an ONNX export directory (e.g. models/onnx/lm-4B/),
    // use TRT inference instead of GGML.
#ifdef HOT_STEP_TRT
    {
        std::string model_dir(params->model_path);
        std::string onnx_path = model_dir + "/lm_full.onnx";
        std::string engine_path = model_dir + "/lm_full.engine";

        FILE* probe = fopen(onnx_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
            fprintf(stderr, "[Ace-LM] ONNX model found at %s — using TRT backend\n", onnx_path.c_str());

            // Build engine if it doesn't exist
            FILE* engine_probe = fopen(engine_path.c_str(), "rb");
            if (engine_probe) {
                fclose(engine_probe);
                fprintf(stderr, "[Ace-LM] TRT engine found, skipping build\n");
            } else {
                fprintf(stderr, "[Ace-LM] TRT engine not found, building...\n");
                if (!lm_trt_build(onnx_path.c_str(), engine_path.c_str(),
                                  params->max_seq)) {
                    fprintf(stderr, "[Ace-LM] ERROR: TRT engine build failed\n");
                    delete ctx;
                    return NULL;
                }
            }

            // Load engine + refit weights from ONNX
            int n_kv = 2 * params->max_batch;
            if (!lm_trt_load(&ctx->lm_trt, engine_path.c_str(), onnx_path.c_str(),
                             params->max_seq, n_kv)) {
                fprintf(stderr, "[Ace-LM] ERROR: TRT engine load failed\n");
                delete ctx;
                return NULL;
            }

            ctx->use_trt = true;
            fprintf(stderr, "[Ace-LM] TRT LM ready: %d KV sets, max_seq=%d\n",
                    n_kv, params->max_seq);
        }
    }
#endif

    // ── TRT-LLM Executor detection ──────────────────────────────────────────
    // If model_path/trtllm-engine-* exists, use TRT-LLM Executor instead of
    // GGML or raw TRT. The Executor handles KV cache, attention kernels, and
    // scheduling internally — no manual buffer management needed.
#ifdef HOT_STEP_TRTLLM
    if (!ctx->use_trt) {  // Don't override raw TRT if it's already active
        std::string model_dir(params->model_path);
        // Look for a trtllm-engine directory (match any suffix like -RTX5090)
        std::string trtllm_dir;
        try {
            for (auto& entry : std::filesystem::directory_iterator(model_dir)) {
                if (entry.is_directory()) {
                    std::string name = entry.path().filename().string();
                    if (name.find("trtllm-engine") == 0) {
                        // Check for rank0.engine (the actual engine file)
                        auto engine_file = entry.path() / "rank0.engine";
                        if (std::filesystem::exists(engine_file)) {
                            trtllm_dir = entry.path().string();
                            break;
                        }
                    }
                }
            }
        } catch (...) {
            // Filesystem errors are non-fatal
        }

        if (!trtllm_dir.empty()) {
            fprintf(stderr, "[Ace-LM] TRT-LLM engine found at %s\n", trtllm_dir.c_str());
            if (lm_trtllm_load(&ctx->lm_trtllm, trtllm_dir.c_str(), params->max_seq)) {
                ctx->use_trtllm = true;
                fprintf(stderr, "[Ace-LM] TRT-LLM Executor ready: max_seq=%d\n", params->max_seq);
            } else {
                fprintf(stderr, "[Ace-LM] WARNING: TRT-LLM init failed, falling back to GGML\n");
            }
        }
    }
#endif

    // Speculative decoding: load draft model if path provided
    ctx->draft_loaded = false;
    if (params->draft_model_path && params->draft_model_path[0]) {
        fprintf(stderr, "[Ace-LM] Loading draft model for speculative decode: %s\n", params->draft_model_path);
        Timer t_draft;
        // Draft model: small KV cache (2048 max), 1 set (no CFG on draft)
        if (qw3lm_load(&ctx->draft, params->draft_model_path, 2048, 1)) {
            ctx->draft.use_flash_attn = params->use_fa;
            ctx->draft.clamp_fp16     = params->clamp_fp16;
            // Build partial LM head for Phase 2 audio codes
            qw3lm_build_partial_head(&ctx->draft, TOKEN_IM_END);
            ctx->draft_loaded = true;
            fprintf(stderr, "[Ace-LM] Draft model loaded: %dL H=%d, %.0fms\n",
                    ctx->draft.cfg.n_layers, ctx->draft.cfg.hidden_size, t_draft.ms());
        } else {
            fprintf(stderr, "[Ace-LM] WARNING: failed to load draft model, speculative decode disabled\n");
        }
    }

    return ctx;
}

int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                lm_batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    bool (*cancel)(void *),
                    void * cancel_data,
                    int    mode) {
    if (!ctx || !req || !out || lm_batch_size < 1) {
        return -1;
    }
    if (lm_batch_size > ctx->params.max_batch) {
        fprintf(stderr, "[Ace-LM] ERROR: lm_batch_size %d > max_batch %d\n", lm_batch_size, ctx->params.max_batch);
        return -1;
    }
    if (req->caption.empty()) {
        fprintf(stderr, "[Ace-LM] ERROR: caption is empty\n");
        return -1;
    }

    // Set TRT dispatch state for this generate call
#ifdef HOT_STEP_TRT
    s_use_trt = ctx->use_trt;
    s_trt_ctx = ctx->use_trt ? &ctx->lm_trt : nullptr;
#endif
#ifdef HOT_STEP_TRTLLM
    s_use_trtllm = ctx->use_trtllm;
    s_trtllm_ctx = ctx->use_trtllm ? &ctx->lm_trtllm : nullptr;
#endif

    // Acquire GPU LM from the store (GGML path). Skip when using TRT or TRT-LLM.
    Qwen3LM * model = nullptr;
    std::unique_ptr<ModelHandle> lm_guard;

    bool skip_ggml = false;
#ifdef HOT_STEP_TRT
    if (ctx->use_trt) skip_ggml = true;
#endif
#ifdef HOT_STEP_TRTLLM
    if (ctx->use_trtllm) skip_ggml = true;
#endif

    if (!skip_ggml) {
        model = store_require_lm(ctx->store, ctx->lm_key);
        if (!model) {
            fprintf(stderr, "[Ace-LM] ERROR: store_require_lm failed\n");
            return -1;
        }
        lm_guard = std::make_unique<ModelHandle>(ctx->store, model);
    }

    // Runtime flags: safe to set on every require (cache-hit or fresh load).
    // These are GGML-specific; skip when TRT is active.
    if (model) {
        if (!ctx->params.use_fa) {
            model->use_flash_attn = false;
        }
        model->clamp_fp16 = ctx->params.clamp_fp16;

        // Fresh load only: allocate the partial LM head for phase2 audio codes.
        // Contiguous GPU tensor instead of ggml_view_2d on quantized weights.
        // Cached on the model itself, freed by qw3lm_free when the store evicts.
        if (!model->lm_head_buf) {
            qw3lm_build_partial_head(model, TOKEN_IM_END);
        }
    }

    // Vocab size: from GGML model or TRT/TRT-LLM constant
    int vocab_size;
    if (model) {
        vocab_size = model->cfg.vocab_size;
    }
#ifdef HOT_STEP_TRT
    else if (ctx->use_trt) {
        vocab_size = LM_TRT_VOCAB;
    }
#endif
#ifdef HOT_STEP_TRTLLM
    else if (ctx->use_trtllm) {
        vocab_size = LM_TRTLLM_VOCAB;
    }
#endif
    else {
        vocab_size = LM_TRT_VOCAB; // fallback, shouldn't happen
    }

    // CPU-resident tokenizer and FSM template. Owned by the store, never
    // evicted. FSM must be copied before mutation since the template is shared.
    BPETokenizer * bpe = store_bpe(ctx->store, ctx->params.model_path);
    if (!bpe) {
        fprintf(stderr, "[Ace-LM] ERROR: store_bpe failed\n");
        return -1;
    }

    MetadataFSM * fsm_template = nullptr;
    if (ctx->params.use_fsm) {
        fsm_template = store_fsm(ctx->store, ctx->params.model_path, vocab_size);
        if (!fsm_template) {
            fprintf(stderr, "[Ace-LM] ERROR: store_fsm failed\n");
            return -1;
        }
    }

    // Local mutable FSM for this call. A copy is mandatory: force_field and
    // apply_mask mutate state that must not bleed across requests.
    MetadataFSM local_fsm;
    if (fsm_template) {
        local_fsm = *fsm_template;
    }

    Timer t_total;

    // mt19937 consumes the low 32 bits of lm_seed (resolved by caller).
    uint32_t seed = (uint32_t) req->lm_seed;

    // Resolve DiT seed (pass through to output for synth pipeline)
    long long dit_seed = req->seed;
    if (dit_seed < 0) {
        std::random_device rd;
        dit_seed = (int64_t) rd();
    }

    // Generation params from request
    float        temperature = req->lm_temperature;
    float        top_p       = req->lm_top_p;
    int          top_k       = req->lm_top_k;
    float        cfg_scale   = req->lm_cfg_scale;
    const char * neg_prompt  = req->lm_negative_prompt.c_str();

    // Copy request -> AcePrompt (internal LLM struct)
    AcePrompt ace      = {};
    ace.caption        = req->caption;
    ace.lyrics         = req->lyrics;
    ace.duration       = req->duration;
    ace.bpm            = req->bpm;
    ace.keyscale       = req->keyscale;
    ace.timesignature  = req->timesignature;
    ace.vocal_language = req->vocal_language;

    bool user_has_codes = !req->audio_codes.empty();
    bool need_lyrics    = ace.lyrics.empty();
    bool has_all_metas  = (ace.bpm > 0 && ace.duration > 0 && !ace.keyscale.empty() && !ace.timesignature.empty());
    bool need_fill      = need_lyrics || !has_all_metas;
    bool skip_codes     = (mode == LM_MODE_INSPIRE || mode == LM_MODE_FORMAT);

    std::vector<int>       prompt;
    std::vector<AcePrompt> aces;

    // ONE path: fill what's missing, then generate codes.
    // JSON is the instruction. Empty field = "fill it". Filled = "don't touch".
    if (user_has_codes && !skip_codes) {
        fprintf(stderr, "[LM-Generate] audio_codes present, skip LM\n");
    } else if (skip_codes || need_fill || req->use_cot_caption) {
        // inspire/format modes always run Phase 1 with their own instruction.
        // generate mode uses the inspire instruction when lyrics are empty.
        if (mode == LM_MODE_INSPIRE || (mode == LM_MODE_GENERATE && need_lyrics)) {
            std::string sys      = std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION + "\n";
            std::string user_msg = ace.caption;
            if (ace.lyrics == "[Instrumental]") {
                user_msg += "\n\ninstrumental: true";
            } else {
                user_msg += "\n\ninstrumental: false";
            }
            // Include vocal language in the text prompt so the LM has a
            // direct signal (the FSM also constrains it, but can desync
            // if the LM generates unexpected fields like "genres:").
            if (!ace.vocal_language.empty() && ace.vocal_language != "unknown") {
                user_msg += "\nlanguage: " + ace.vocal_language;
            }
            prompt = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
        } else if (mode == LM_MODE_FORMAT) {
            std::string sys      = std::string("# Instruction\n") + LM_FORMAT_INSTRUCTION + "\n";
            std::string user_msg = "# Caption\n" + ace.caption + "\n\n# Lyric\n" + ace.lyrics;
            prompt               = build_custom_prompt(*bpe, sys.c_str(), user_msg.c_str());
        } else {
            prompt = build_lm_prompt(*bpe, ace);
        }
        std::vector<int> uncond;

        // inspire/format always generate lyrics. generate mode: only when lyrics are empty.
        bool gen_lyrics = need_lyrics || skip_codes;

        // Disable CFG for ANY textual expansion (lyrics OR CoT reasoning),
        // as CFG distorts text logits and forces premature newlines.
        float fill_cfg   = (gen_lyrics || req->use_cot_caption) ? 1.0f : cfg_scale;
        float fill_top_p = top_p;
        int   fill_top_k = top_k;

        if (fill_cfg > 1.0f) {
            uncond = build_lm_prompt_uncond(*bpe, ace, neg_prompt);
        }

        local_fsm.reset();
        MetadataFSM * active_fsm = nullptr;

        if (ctx->params.use_fsm) {
            // FSM constrains CoT metadata (bpm/dur/key/lang/tsig).
            // CAPTION_VALUE is free-form (only blocks audio codes).
            // Lyrics after </think> are unconstrained.
            // Force user-provided values into the KV cache so the LM
            // generates lyrics and codes conditioned on the right metadata.

            // Caption lock (use_cot_caption=false): skip the caption zone
            // in CoT, the user-provided caption stays untouched and the LM
            // sees it via the user prompt block.
            // Inspire mode regenerates the caption from scratch, so the
            // lock is ignored there.
            local_fsm.skip_caption = !req->use_cot_caption && (mode != LM_MODE_INSPIRE);

            if (ace.bpm > 0) {
                local_fsm.force_field(*bpe, MetadataFSM::BPM_VALUE, std::to_string(ace.bpm));
            }
            if (ace.duration > 0) {
                local_fsm.force_field(*bpe, MetadataFSM::DURATION_VALUE, std::to_string((int) ace.duration));
            }
            if (!ace.keyscale.empty()) {
                local_fsm.force_field(*bpe, MetadataFSM::KEYSCALE_VALUE, ace.keyscale);
            }
            if (!ace.vocal_language.empty() && ace.vocal_language != "unknown") {
                local_fsm.force_field(*bpe, MetadataFSM::LANGUAGE_VALUE, ace.vocal_language);
            }
            if (!ace.timesignature.empty()) {
                local_fsm.force_field(*bpe, MetadataFSM::TIMESIG_VALUE, ace.timesignature);
            }
            active_fsm = &local_fsm;
        }

        const char * mode_name = skip_codes ? (mode == LM_MODE_INSPIRE ? "inspire" : "format") : "fill";
        fprintf(stderr, "[LM-Generate] mode=%s lyrics=%s metas=%s | %zu tokens, CFG: %.2f, N=%d\n", mode_name,
                gen_lyrics ? "generate" : "keep", has_all_metas ? "complete" : "fill gaps", prompt.size(), fill_cfg,
                lm_batch_size);

        auto phase1_texts = generate_phase1_batch(model, bpe, prompt, 2048, temperature, fill_top_p, fill_top_k, seed,
                                                  lm_batch_size, active_fsm, gen_lyrics, fill_cfg,
                                                  uncond.empty() ? nullptr : &uncond, !gen_lyrics, cancel, cancel_data);
        if (phase1_texts.empty()) {
            return -1;
        }

        // inspire mode: empty base so the LM output overwrites everything.
        // format/generate: gap fill, user metadata preserved.
        AcePrompt parse_base = (mode == LM_MODE_INSPIRE) ? AcePrompt{} : ace;

        // Inspire ignores the caption lock end to end: sampling regenerates
        // and parsing accepts. Other modes honor the user flag.
        bool parse_use_cot_caption = (mode == LM_MODE_INSPIRE) ? true : req->use_cot_caption;
        parse_phase1_into_aces(phase1_texts, parse_base, aces, seed, mode_name, gen_lyrics, parse_use_cot_caption);

        // Caption preservation: the LM may enrich the user caption, but
        // never silently delete it. If the merge ended up with an empty
        // caption while the request had one, restore the request value.
        if (!ace.caption.empty()) {
            for (auto & a : aces) {
                if (a.caption.empty()) {
                    a.caption = ace.caption;
                }
            }
        }

        int n_kv_reset = (fill_cfg > 1.0f) ? 2 * lm_batch_size : lm_batch_size;
        for (int i = 0; i < n_kv_reset; i++) {
            lm_reset_kv(model, i);
        }
    }

    if (aces.empty()) {
        aces = { ace };
    }

    // Debug: dump tokens/logits
    if (!user_has_codes && (dump_logits || dump_tokens)) {
        std::string cot        = build_cot_yaml(aces[0]);
        auto        dbg_prompt = build_lm_prompt_with_cot(*bpe, aces[0], cot);

        if (dump_tokens) {
            FILE * f = fopen(dump_tokens, "w");
            if (f) {
                for (size_t j = 0; j < dbg_prompt.size(); j++) {
                    fprintf(f, "%s%d", j ? "," : "", dbg_prompt[j]);
                }
                fprintf(f, "\n");
                fclose(f);
                fprintf(stderr, "[LM-Debug] Tokens -> %s (%zu)\n", dump_tokens, dbg_prompt.size());
            }
        }
        if (dump_logits) {
            std::vector<float> dbg_logits(vocab_size);
            lm_forward(model, dbg_prompt.data(), (int) dbg_prompt.size(), 0, dbg_logits.data());
            FILE * f = fopen(dump_logits, "wb");
            if (f) {
                fwrite(dbg_logits.data(), sizeof(float), vocab_size, f);
                fclose(f);
                fprintf(stderr, "[LM-Debug] Logits -> %s (%d floats, argmax=%d)\n", dump_logits, vocab_size,
                        (int) (std::max_element(dbg_logits.begin(), dbg_logits.end()) - dbg_logits.begin()));
            }
            lm_reset_kv(model, 0);
        }
    }

    // Phase 2: generate audio codes (skip for inspire/format modes)
    std::vector<std::string> batch_codes(lm_batch_size);
    if (skip_codes) {
        fprintf(stderr, "[LM-Generate] %s mode, no audio code generation\n",
                mode == LM_MODE_INSPIRE ? "Inspire" : "Format");
    } else if (!user_has_codes) {
        // Speculative decode: draft model available + batch_size=1
        if (ctx->draft_loaded && lm_batch_size == 1) {
            const char * neg = (neg_prompt && neg_prompt[0]) ? neg_prompt : nullptr;
            batch_codes = run_phase2_speculative(model, &ctx->draft, *bpe, aces[0],
                                                  temperature, top_p, top_k, seed, cfg_scale,
                                                  req->lm_cfg_cutoff_ratio, neg, cancel, cancel_data);
        } else {
            batch_codes = run_phase2_batch(model, *bpe, aces, temperature, top_p, top_k, seed, lm_batch_size, cfg_scale,
                                           req->lm_cfg_cutoff_ratio,
                                           neg_prompt, ctx->params.use_batch_cfg, cancel, cancel_data);
        }
        if (batch_codes.empty()) {
            return -1;
        }
    } else {
        fprintf(stderr, "[LM-Generate] User audio_codes present, no code generation\n");
    }

    // Write N output requests
    for (int b = 0; b < lm_batch_size; b++) {
        out[b]                = *req;
        const AcePrompt & a   = aces[b < (int) aces.size() ? b : 0];
        out[b].caption        = a.caption;
        out[b].lyrics         = a.lyrics;
        out[b].bpm            = a.bpm;
        out[b].duration       = a.duration;
        out[b].keyscale       = a.keyscale;
        out[b].timesignature  = a.timesignature;
        out[b].vocal_language = a.vocal_language;
        if (!batch_codes[b].empty()) {
            out[b].audio_codes = batch_codes[b];
        }
        out[b].seed          = dit_seed + b;
        out[b].lm_seed       = req->lm_seed + b;
        out[b].lm_batch_size = 1;  // each output is a standalone enriched request

        // Backend-driven flag: report what was actually applied. Inspire
        // always regenerates the caption regardless of the input lock,
        // so the response reflects the effective state and the UI toggle
        // updates itself to match reality.
        if (mode == LM_MODE_INSPIRE) {
            out[b].use_cot_caption = true;
        }
    }

    fprintf(stderr, "[Ace-LM] Total %.0fms | seed=%lld\n", t_total.ms(), dit_seed);
    return 0;
}

void ace_lm_free(AceLm * ctx) {
    if (!ctx) {
        return;
    }
#ifdef HOT_STEP_TRT
    if (ctx->use_trt) {
        lm_trt_free(&ctx->lm_trt);
        ctx->use_trt = false;
        s_use_trt = false;
        s_trt_ctx = nullptr;
        fprintf(stderr, "[Ace-LM] TRT LM freed\n");
    }
#endif
#ifdef HOT_STEP_TRTLLM
    if (ctx->use_trtllm) {
        lm_trtllm_free(&ctx->lm_trtllm);
        ctx->use_trtllm = false;
        s_use_trtllm = false;
        s_trtllm_ctx = nullptr;
        fprintf(stderr, "[Ace-LM] TRT-LLM Executor freed\n");
    }
#endif
    if (ctx->draft_loaded) {
        qw3lm_free(&ctx->draft);
        ctx->draft_loaded = false;
        fprintf(stderr, "[Ace-LM] Draft model freed\n");
    }
    delete ctx;
}

const ModelKey * ace_lm_lm_key(const AceLm * ctx) {
    return ctx ? &ctx->lm_key : nullptr;
}
