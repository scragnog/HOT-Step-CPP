#pragma once
// pipeline-understand.h: ACE-Step reverse pipeline (audio -> metadata)
//
// Audio -> VAE encode -> FSQ tokenize -> LM understand -> metadata + lyrics.
// Or: audio_codes from request -> LM understand -> metadata + lyrics.

#include "request.h"

struct AceUnderstand;
struct Qwen3LM;
struct BPETokenizer;

struct AceUnderstandParams {
    const char * model_path;   // LM GGUF (required, unless shared_model or dump_dir set)
    const char * dit_path;     // DiT GGUF (required for audio input, has FSQ codebook)
    const char * vae_path;     // VAE GGUF (required for audio input, has encoder)
    const char * dump_dir;     // dump tok_latents + tok_codes (NULL = disabled)
    int          max_seq;      // KV cache length (default: 8192)
    bool         use_fsm;      // constrained decoding (default: true)
    bool         use_fa;       // flash attention (default: true)
    int          vae_chunk;    // latent frames per tile (default: 256)
    int          vae_overlap;  // overlap frames per side (default: 64)

    // shared LM from pipeline-lm (NULL = load own copy from model_path)
    Qwen3LM *      shared_model;
    BPETokenizer * shared_bpe;
};

void ace_understand_default_params(AceUnderstandParams * p);

// Load models. dit_path and vae_path are optional (only needed for audio input).
// NULL on failure.
AceUnderstand * ace_understand_load(const AceUnderstandParams * params);

// Run the understand pipeline.
// src_audio: interleaved stereo 48kHz [L0,R0,L1,R1,...], or NULL for codes-only mode.
// src_len: samples per channel (0 if no audio).
// req: sampling params (temperature, top_p, top_k, seed). In codes-only mode,
//      req->audio_codes must be filled.
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
