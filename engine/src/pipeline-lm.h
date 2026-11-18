#pragma once
// pipeline-lm.h: ACE-Step LM pipeline
//
// A lightweight context bound to a ModelStore. On each generate call the
// pipeline acquires the Qwen3 LM, BPE tokenizer and FSM template from the
// store and releases the LM through RAII before returning.

#include "request.h"
#include "task-types.h"

struct AceLm;
struct ModelStore;

struct AceLmParams {
    const char * model_path;     // LM GGUF (required)
    int          max_seq;        // KV cache length
    int          max_batch;      // max lm_batch_size for generate
    bool         use_fsm;        // constrained decoding
    bool         use_fa;         // flash attention
    bool         use_batch_cfg;  // batch cond+uncond in one forward
    bool         clamp_fp16;     // clamp hidden states to FP16 range
    const char * draft_model_path;  // draft LM for speculative decode (NULL = disabled)

    // Planner-LM runtime LoRA (local HOT-Step feature). NULL/empty = none.
    const char * adapter_path;   // resolved PEFT dir or .safetensors path
    float        adapter_scale;  // user strength (× alpha/r at load)
};

void ace_lm_default_params(AceLmParams * p);

// Build a lightweight LM context bound to a ModelStore. The GPU model is
// acquired per request, never owned by the context. NULL on invalid input.
AceLm * ace_lm_load(ModelStore * store, const AceLmParams * params);

// Enrich request with metadata, lyrics, audio codes.
// out[lm_batch_size] allocated by caller, filled with enriched copies of req.
// mode: LM_MODE_GENERATE (full), LM_MODE_INSPIRE (no codes), LM_MODE_FORMAT (no codes).
// dump_logits/dump_tokens: debug output paths (NULL to disable).
// cancel/cancel_data: abort callback, polled between tokens. NULL = never cancel.
// Returns 0 on success, -1 on error or cancellation.
int ace_lm_generate(AceLm *            ctx,
                    const AceRequest * req,
                    int                lm_batch_size,
                    AceRequest *       out,
                    const char *       dump_logits,
                    const char *       dump_tokens,
                    bool (*cancel)(void *) = nullptr,
                    void * cancel_data     = nullptr,
                    int    mode            = LM_MODE_GENERATE);

// Mixed batch: several requests (each lm_batch_size songs) in one LM pass,
// generate mode only. Every song gets the tokens it would get run alone, seed
// for seed. Requests must agree on the LM and its sampling settings (see
// ace_lm_batch_mismatch) and carry no audio_codes; lm_seed must be resolved.
// out holds the sum of lm_batch_size, in request order.
int ace_lm_generate_multi(AceLm *            ctx,
                          const AceRequest * reqs,
                          int                n_reqs,
                          AceRequest *       out,
                          bool (*cancel)(void *) = nullptr,
                          void * cancel_data     = nullptr);

// Name of the first LM setting that differs between two requests, or NULL
// when they can share one mixed batch.
const char * ace_lm_batch_mismatch(const AceRequest * a, const AceRequest * b);

void ace_lm_free(AceLm * ctx);

// Read the LM ModelKey the context builds for store_require_lm. Used by
// test-model-store to verify both ace_lm and ace_understand resolve to the
// same LM instance when their params are propagated consistently.
struct ModelKey;
const ModelKey * ace_lm_lm_key(const AceLm * ctx);
