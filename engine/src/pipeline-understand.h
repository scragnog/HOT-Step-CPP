#pragma once
// pipeline-understand.h: ACE-Step reverse pipeline (audio -> metadata)
//
// A lightweight context bound to a ModelStore. Each generate call acquires
// the VAE encoder, FSQ tokenizer and LM from the store in sequence, with
// RAII release between stages. Audio -> latents -> codes -> LM -> metadata.

#include "request.h"

struct AceUnderstand;
struct ModelStore;

struct AceUnderstandParams {
    const char * model_path;   // LM GGUF (required, unless dump_dir set for tok-only mode)
    const char * dit_path;     // DiT GGUF (required, has FSQ codebook)
    const char * vae_path;     // VAE GGUF (required, has encoder)
    const char * dump_dir;     // dump tok_latents + tok_codes (NULL = disabled)
    int          max_seq;      // KV cache length
    int          max_batch;    // must match ace_lm max_batch so the LM ModelKey is identical
                               // and the two pipelines share one LM instance
    bool         use_fsm;      // constrained decoding
    bool         use_fa;       // flash attention
    int          vae_chunk;    // latent frames per tile
    int          vae_overlap;  // overlap frames per side
};

void ace_understand_default_params(AceUnderstandParams * p);

// Build a lightweight understand context bound to a ModelStore. Validates
// paths and builds ModelKeys. GPU modules are acquired per request, never
// owned by the context. NULL on invalid input.
AceUnderstand * ace_understand_load(ModelStore * store, const AceUnderstandParams * params);

// Run the understand pipeline.
// src_audio: interleaved stereo 48kHz [L0,R0,L1,R1,...]. Required.
// src_len: samples per channel.
// req: sampling params (temperature, top_p, top_k, seed).
// out: filled with caption, lyrics, metadata, audio_codes, DiT defaults.
// cancel/cancel_data: abort callback, polled between tokens. NULL = never cancel.
// Returns 0 on success, -1 on error or cancellation.
int ace_understand_generate(AceUnderstand *    ctx,
                            const float *      src_audio,
                            int                src_len,
                            const AceRequest * req,
                            AceRequest *       out,
                            bool (*cancel)(void *) = nullptr,
                            void * cancel_data     = nullptr);

void ace_understand_free(AceUnderstand * ctx);

// Read the LM ModelKey the context builds for store_require_lm. Used by
// test-model-store to verify both ace_understand and ace_lm resolve to the
// same LM instance; a drift here silently doubles VRAM under --keep-loaded.
struct ModelKey;
const ModelKey * ace_understand_lm_key(const AceUnderstand * ctx);
