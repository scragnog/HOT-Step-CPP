#pragma once
// yue2/sheetsage-model.h — SheetSage2 GGUF loader: config, encoder weights,
// decoder weights, one shared backend residency.
//
// HOT-Step file (no acestep.cpp / upstream analog). Loads models/yue2/
// sheetsage2-<type>.gguf (arch "sheetsage2", engine/tools/convert-sheetsage2.py)
// into ggml tensors: everything the encoder graph (sheetsage-encoder.h, this
// lane) and the decoder graph (sheetsage-decoder.h, a DIFFERENT lane) need, in
// ONE backend buffer — encoder and decoder must be resident together for a
// transcribe() call, so there is no "load encoder, free, load decoder" split.
// The decoder tensors are loaded HERE, by this file, so the decoder lane only
// has to write the compute graph.
//
// AUTHORITY: docs/plans/yue2/20-sheetsage2-model-pin.md ("MODEL PIN" — §0-§4.6
// are this file's contract; §5-§6 explain WHY the numbers are what they are
// but carry nothing this loader needs to re-derive). Where this file and that
// document disagree, the document wins. KV/tensor names below were read
// directly from engine/tools/convert-sheetsage2.py's add_meta()/
// build_encoder()/build_decoder() functions — that converter's own OUTPUT is
// the ground truth for what a real sheetsage2-*.gguf actually contains, and
// the file this loader was built and checked against
// (models/yue2/sheetsage2-f16.gguf) was produced by that exact converter.
//
// ── ENCODER: REUSE, NOT A FORK ──────────────────────────────────────────────
//
// This file does NOT reimplement the mel frontend, the ConvNext subsampling
// stack, or the Conformer block — it reuses yue2-mert.h's structs
// (Yue2MertConfig, Yue2MertWeights' Yue2MertSubBlock/Yue2MertCnxLayer/
// Yue2MertBlock) and free functions (yue2_mert_mel, yue2_mert_sub_build,
// yue2_mert_detail::mert_block, Yue2MertEncGraph, yue2_mert_encode_chunk)
// COMPLETELY UNMODIFIED. Nothing in yue2-mert.h needed to change for this to
// work, because none of the reused functions are hard-wired to 21 blocks or
// to hidden_state_index — that restriction lives only inside
// yue2_mert_load()/yue2_mert_validate_config(), and THIS FILE CALLS NEITHER
// of those (they are locked to arch "yue2-tok" and the "yue2tok.mert.*" KV
// prefix, a different GGUF). Instead this loader builds a *bare*
// `Yue2MertModel` by hand — own WeightCtx (below), `sheetsage2.*` KVs,
// `enc.*` tensor names, `block_count = 24` (none dropped, doc 20 §1 item 1)
// — and hands it to the reused compute functions, which only ever read
// `Yue2MertModel::cfg` / `::w` / `::blocks_loaded` / `::wctx.buffer` (the last
// one purely as an identity token for the encoder graph's cache check, see
// "Yue2MertModel view" below).
//
// SheetSage2 needs all 25 hidden states (embedding + every one of the 24
// block outputs, doc 20 §2.5) where the YuE2 semantic tokenizer needed only
// one (`hidden_states[20]`) — sheetsage-encoder.h gets every state by tapping
// every block through the SAME `Yue2MertEncGraph::tap_blocks` mechanism the
// MERT parity probe (`--mert-block-parity`) already exercises today, not a
// fork of the graph builder.
//
// ── DECODER ─────────────────────────────────────────────────────────────────
//
// Loaded here (this lane); the compute graph is sheetsage-decoder.h's job (a
// different lane), so `SheetSageDecoderWeights` below is effectively that
// lane's input contract — names/shapes are exactly what
// convert-sheetsage2.py::build_decoder() writes, nothing inferred. See doc 20
// §3 for the decoder's own math (post-LN BART-style, 4 separate q/k/v/out
// projections per attention, tied token embedding / output projection stored
// once as `dec.tok_embd`).
//
// ── TENSOR LAYOUT CONVENTION ────────────────────────────────────────────────
//
// Same as yue2-mert.h / convert-yue2-tok.py: `ne = (in, out)` for every 2-D
// matmul weight (gguf reverses the checkpoint's row-major numpy shape on
// write), 1-D tensors unchanged, the kernel-2 resampling convs kept 3-D
// `[k, in, out]`. `Yue2Loader::req(name, e0, e1, ...)` takes ne dims in that
// already-reversed order, so every `req()` call below reads left-to-right as
// "(in, out)" or "(k, in, out)" — cross-checked against
// convert-sheetsage2.py's own `expect=` shape assertions (which are in numpy/
// torch order, the OPPOSITE convention) tensor by tensor while this file was
// written.
//
// ── FINGERPRINT ─────────────────────────────────────────────────────────────
//
// `sheetsage2.vocab_fingerprint` is read back into `SheetSageConfig::
// vocab_fingerprint` and exposed, never recomputed here — recomputing it
// needs the full vocabulary build (structure/chord/duration label tables),
// which is the tokenizer lane's job (sheetsage-tokens.h, doc 20 §4.6 /
// doc 21). This loader only proves the KV round-tripped.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "yue2-mert.h"  // Yue2MertConfig/Weights/Model + reused mel/subsampling/block code
#include "yue2-model.h"  // Yue2Loader, GGUFModel, WeightCtx, gf_get_*/yue2_get_i32_arr, backend_init/_release

// ── Config ──────────────────────────────────────────────────────────────────

// The decoder half — BART-style, but NOT transformers.BartDecoder (doc 20 §3:
// port ai-toolkit's plain-torch rewrite). Every field below is read from a
// `sheetsage2.decoder.*` KV; the "MUST" comments are enforced in
// sheetsage_validate_config() below, not merely documented.
struct SheetSageDecoderConfig {
    uint32_t block_count             = 0;  // 6
    uint32_t embedding_length        = 0;  // 512
    uint32_t feed_forward_length     = 0;  // 2048
    uint32_t head_count              = 0;  // 8
    uint32_t key_length              = 0;  // 64
    uint32_t value_length            = 0;  // 64 (== key_length)
    float    layer_norm_eps          = 0.0f;  // 1e-5 (PyTorch default; doc 20 §3.2, post-LN)
    uint32_t position_offset         = 0;      // 2 -- position id fed to embed_positions is (t + 2)
    uint32_t max_position_embeddings = 0;      // 5122 == max_output_seq_len(5120) + position_offset(2)
    float    embed_scale             = 0.0f;   // 1.0 -- BartConfig scale_embedding=False (doc 20 §3.1)
    bool     qkv_fused               = true;   // MUST read false: four SEPARATE q/k/v/out projections
    std::string norm_style;                    // MUST read "post" (doc 20 §3.2)
    std::string activation;                    // "gelu_erf" -- exact erf, doc 20 §3.2
    bool     cross_attn_kv_static = false;      // MUST read true: cross K/V computed once per window,
                                                 // reused verbatim every decode step (doc 20 §3.4)
    bool     cross_attn_masked    = true;       // MUST read false: no mask over padded memory frames,
                                                 // ever, at any step (doc 20 §3.4)
};

// Top-level sheetsage2.* config not already covered by Yue2MertConfig (which
// carries the shared mel/subsampling/Conformer fields, populated from
// `sheetsage2.*` KVs by sheetsage_parse_config() below — see the file header
// for why that struct, not a fresh one, is reused for the encoder body).
struct SheetSageConfig {
    float    window_seconds     = 0.0f;  // 300.0 -- the ALWAYS-fully-padded window (doc 20 §1.2)
    uint32_t vocab_size         = 0;     // 31678
    uint32_t time_hz            = 0;     // 100
    uint32_t max_output_seq_len = 0;     // 5120 -- a TOTAL sequence-length cap (prefix + generated),
                                          // not a per-window token budget (doc 20 §3.6 point 5);
                                          // the pipeline/decoder lanes own that arithmetic, not this file
    uint32_t token_pad = 0, token_sos = 0, token_eos = 0, token_out = 0;  // 0, 1, 2, 3

    // Doc 20 §4.6: baked SHA-256[:16] of a 13-key vocabulary payload. Exposed
    // for the tokenizer lane to recompute-and-compare, never recomputed here
    // (see file header).
    std::string vocab_fingerprint;

    float overlap_seconds_default   = 0.0f;  // 200.0 -- pipeline-lane informational (window plan)
    float lookahead_seconds_default = 0.0f;  // 100.0 -- pipeline-lane informational

    uint32_t hidden_state_count    = 0;  // 25 = 1 (post-subsampling embedding) + 24 (block outputs)
    uint32_t projection_out_length = 0;  // 512 -- encoder_projection's output width (decoder memory)

    SheetSageDecoderConfig dec;
};

// ── Weights ─────────────────────────────────────────────────────────────────

// Encoder head: softmax(layer_weight)-weighted sum over all 25 hidden states,
// then a biased Linear(1024,512) to the decoder's memory width (doc 20 §2.5).
// `layer_w` is the RAW parameter — softmax happens at RUNTIME (a fixed,
// input-independent 25-element softmax; sheetsage-encoder.h caches it rather
// than recomputing it every window, since it never changes).
struct SheetSageEncoderHead {
    ggml_tensor * layer_w = nullptr;  // [25]          nn.Parameter(zeros(25)); softmax(dim=0) at use
    ggml_tensor * proj_w  = nullptr;  // [1024, 512]   encoder_projection.weight, biased
    ggml_tensor * proj_b  = nullptr;  // [512]
};

// One decoder layer (doc 20 §3.2-§3.4). POST-LN (norm after the residual add,
// not before — a pre-LN implementation silently normalizes the wrong tensor).
// Self-attention and cross-attention are both four SEPARATE biased
// Linear(512,512) projections, never fused, same convention as the encoder
// (doc 20 §2.3/§3.2). Residual order, for the graph lane:
//
//   h, self_kv  = self_attn(x, causal_mask, past=self_cache)
//   x           = self_attn_norm(x + h)
//   h, cross_kv = cross_attn(x, memory, past=cross_cache, cache_static=True)
//   x           = cross_attn_norm(x + h)
//   x           = final_norm(x + fc2(gelu_erf(fc1(x))))
struct SheetSageDecoderBlock {
    ggml_tensor * self_attn_norm_w = nullptr, * self_attn_norm_b = nullptr;
    ggml_tensor * self_attn_q_w = nullptr, * self_attn_q_b = nullptr;
    ggml_tensor * self_attn_k_w = nullptr, * self_attn_k_b = nullptr;
    ggml_tensor * self_attn_v_w = nullptr, * self_attn_v_b = nullptr;
    ggml_tensor * self_attn_o_w = nullptr, * self_attn_o_b = nullptr;  // self_attn_output.*

    ggml_tensor * cross_attn_norm_w = nullptr, * cross_attn_norm_b = nullptr;  // encoder_attn_layer_norm
    ggml_tensor * cross_attn_q_w = nullptr, * cross_attn_q_b = nullptr;
    ggml_tensor * cross_attn_k_w = nullptr, * cross_attn_k_b = nullptr;
    ggml_tensor * cross_attn_v_w = nullptr, * cross_attn_v_b = nullptr;
    ggml_tensor * cross_attn_o_w = nullptr, * cross_attn_o_b = nullptr;  // cross_attn_output.*

    ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;      // fc1: 512 -> 2048
    ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;  // fc2: 2048 -> 512
    ggml_tensor * final_norm_w = nullptr, * final_norm_b = nullptr;
};

struct SheetSageDecoderWeights {
    // == output_projection.weight, bit-identical (tied, doc 20 §3.5) — the
    // converter writes it ONCE; the decoder graph must use this same tensor
    // for both the embedding lookup (ggml_get_rows) and the final logits
    // projection (a transposed matmul), never load a second copy.
    ggml_tensor * tok_embd = nullptr;  // [512, 31678]

    // Learned lookup table, read verbatim by row index (t + position_offset)
    // — NOT a sinusoid formula recomputed at runtime (doc 20 §3.1, same
    // "NATIVE regardless of size" treatment as YuE2's own latent_pos_embed).
    ggml_tensor * pos_embd = nullptr;  // [512, 5122]

    ggml_tensor * norm_embd_w = nullptr, * norm_embd_b = nullptr;  // layernorm_embedding, [512]

    std::vector<SheetSageDecoderBlock> blk;  // 6
};

// ── "exact" load option (2026-09-16 precision investigation) ────────────────
//
// AUTHORITY: docs/plans/yue2/23-sheetsage2-progress.md "Encoder precision
// investigation (2026-09-16)". Ground truth measured there: the ggml CPU
// backend reproduces the CUDA per-block rel-L2 curve almost exactly on
// long-b/window-0001 (hidden_00 3.1e-4 rising to ~1.2e-2..1.6e-2 by block
// 14-23), so this is NOT CUDA reduction order — it is `ggml_mul_mat`
// quantizing the F32 activation side to match an F16/BF16-native weight
// (src0) for the dot-product kernel (CPU vec_dot_type; cuBLAS/cuDNN's F16
// GEMM path on CUDA), exactly the mechanism sheetsage-model.h's own
// `blk23_attn_f32` comment already names for block 23's LoRA-outlier
// out_proj. `exact=true` generalizes that SAME loader trick (Yue2Loader::
// force_f32 — upcast the tensor to F32 in the backend buffer at load time,
// no ggml source change, works from a plain F16 GGUF) to every
// weight-bearing matmul in the ENCODER body: subsampling ConvNeXt
// pw_up/pw_down and every block's ffn1/ffn2 up+down and attn q/k/v/o.
// (`conv_pw1`/`conv_pw2` are the two kernel-1 "conv as matmul" tensors —
// included too, same mechanism.) The decoder is untouched: G2 (teacher-forced
// argmax) already passes under the fast path and generalizing this there is
// out of this investigation's scope.
//
// The attention math itself needs NO such switch: `yue2_mert_detail::
// mert_attn()` (yue2-mert.h) has never used `ggml_flash_attn_ext` for this
// encoder — QK^T and AV are `ggml_mul_mat` between two F32 activation
// tensors (q/k/v are always the F32 output of a mul_mat+bias-add, never
// stored in a reduced type), and the softmax is the plain, always-exact
// `ggml_soft_max_ext(..., /*mask*/ nullptr, ...)` manual path. So "use the
// f32 soft_max path instead of flash" is already this file's only behavior;
// there was no flash-attention path in the encoder to opt out of.
//
// Cost: every NATIVE-policy encoder tensor doubles from F16/BF16 to F32 in
// VRAM. Measured total (doc 23): see the progress doc's cost row.
struct SheetSageModelLoadOptions {
    bool exact = false;  // default false: today's fast path, unchanged.
};

struct SheetSageModel {
    std::string     path;
    std::string     arch;     // must be "sheetsage2"
    std::string     license;
    SheetSageConfig cfg;

    // Encoder body: reused yue2-mert.h types verbatim (see file header). Only
    // `cfg`, `w.sub`, `w.blk`, `blocks_loaded` and `mel` (built below via
    // yue2_mert_build_mel_plan) are ever populated or read — `wctx`/
    // `backend`/`cpu_backend`/`backend_ref` on THIS sub-object are NOT a real
    // second residency: `wctx.buffer` is set to the SAME pointer as this
    // model's own `wctx.buffer` purely so Yue2MertEncGraph's cache-identity
    // check (`g->weights_token == m.wctx.buffer`) works, and
    // `backend_ref` is left false so sheetsage_model_free() never
    // double-frees through it. yue2_mert_free() must NEVER be called on this
    // member — see sheetsage_model_free() below.
    Yue2MertModel mert;

    SheetSageEncoderHead    head;  // layer_weight softmax input + projection
    SheetSageDecoderWeights dec;   // loaded here; consumed by sheetsage-decoder.h (a different lane)

    WeightCtx      wctx        = {};  // the ONE backend buffer for enc.* + dec.* + head tensors
    bool           backend_ref = false;
    ggml_backend_t backend     = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    size_t         vram        = 0;
    double         load_ms     = 0.0;

    // Smallest public accessor added for the pipeline lane (doc 23 "Phase 3b"):
    // records whether THIS load actually ran with SheetSageModelLoadOptions::
    // exact (the encoder matmul-weights-forced-F32 precision fix above), so a
    // caller holding only a `const SheetSageModel&` (sheetsage-pipeline.h's
    // sheetsage_transcribe(), which never sees load_opt) can fail fast instead
    // of silently running fast-path precision when the caller's own
    // SheetSageTranscribeOptions::exact asked for the exact path. Read-only
    // record of a load-time decision -- setting this field directly does
    // nothing; only sheetsage_model_load() may write it.
    bool exact = false;

    std::map<std::string, ggml_tensor *> tmap;  // introspection only
};

// ── Config parsing / validation ─────────────────────────────────────────────

// `mc` gets the shared mel/subsampling/Conformer fields (same struct
// yue2-mert.h's own loader fills, different KV prefix: "sheetsage2." instead
// of "yue2tok.mert."). `c` gets everything sheetsage2-specific.
static void sheetsage_parse_config(const GGUFModel & gf, SheetSageConfig * c, Yue2MertConfig * mc) {
    mc->sample_rate           = gf_get_u32(gf, "sheetsage2.sample_rate");
    mc->n_fft                 = gf_get_u32(gf, "sheetsage2.mel.n_fft");
    mc->win_length            = gf_get_u32(gf, "sheetsage2.mel.win_length");
    mc->hop_length            = gf_get_u32(gf, "sheetsage2.mel.hop_length");
    mc->num_mel_bins          = gf_get_u32(gf, "sheetsage2.mel.num_mel_bins");
    mc->num_freq_bins         = gf_get_u32(gf, "sheetsage2.mel.num_freq_bins");
    mc->frame_rate            = gf_get_f32(gf, "sheetsage2.mel.frame_rate");
    mc->samples_per_frame     = gf_get_u32(gf, "sheetsage2.mel.samples_per_frame");
    mc->minimum_input_samples = gf_get_u32(gf, "sheetsage2.mel.minimum_input_samples");
    mc->spectrogram_power     = gf_get_f32(gf, "sheetsage2.mel.spectrogram_power");
    mc->db_amin               = gf_get_f32(gf, "sheetsage2.mel.db_amin");
    mc->db_multiplier         = gf_get_f32(gf, "sheetsage2.mel.db_multiplier");
    mc->db_top_db_clamp       = gf_get_bool(gf, "sheetsage2.mel.db_top_db_clamp");
    mc->drop_last_mel_frame   = gf_get_bool(gf, "sheetsage2.mel.drop_last_mel_frame");
    mc->mel_std_floor         = gf_get_f32(gf, "sheetsage2.mel.mel_std_floor");

    mc->sub_block_count      = gf_get_u32(gf, "sheetsage2.subsampling.block_count");
    mc->sub_in_channels      = gf_get_u32(gf, "sheetsage2.subsampling.input_channels");
    mc->sub_channels         = yue2_get_i32_arr(gf, "sheetsage2.subsampling.channels");
    mc->sub_depths           = yue2_get_i32_arr(gf, "sheetsage2.subsampling.depths");
    mc->sub_strides          = yue2_get_i32_arr(gf, "sheetsage2.subsampling.strides");
    mc->sub_resample_kernel  = gf_get_u32(gf, "sheetsage2.subsampling.resample_kernel");
    mc->sub_resample_padding = gf_get_bool(gf, "sheetsage2.subsampling.resample_padding");
    mc->sub_convnext_kernel  = gf_get_u32(gf, "sheetsage2.subsampling.convnext_kernel");
    mc->sub_convnext_padding = gf_get_u32(gf, "sheetsage2.subsampling.convnext_padding");
    mc->sub_ln_eps           = gf_get_f32(gf, "sheetsage2.subsampling.layer_norm_epsilon");
    mc->sub_grn_eps          = gf_get_f32(gf, "sheetsage2.subsampling.grn_epsilon");

    mc->embedding_length    = gf_get_u32(gf, "sheetsage2.encoder.embedding_length");
    mc->feed_forward_length = gf_get_u32(gf, "sheetsage2.encoder.feed_forward_length");
    mc->block_count         = gf_get_u32(gf, "sheetsage2.encoder.block_count");
    // Doc 20 §1 item 1: ALL 24 blocks, none dropped — there is no single
    // "the" hidden state the way yue2-tok's hidden_states[20] is, so
    // hidden_state_index has no real meaning here; set to block_count for a
    // harmless, self-consistent value (nothing in the reused compute path
    // reads it — that restriction lives only in yue2_mert_validate_config(),
    // never called by this loader).
    mc->block_count_upstream = mc->block_count;
    mc->hidden_state_index   = mc->block_count;
    mc->head_count           = gf_get_u32(gf, "sheetsage2.encoder.attention.head_count");
    mc->key_length           = gf_get_u32(gf, "sheetsage2.encoder.attention.key_length");
    mc->ln_eps               = gf_get_f32(gf, "sheetsage2.encoder.layer_norm_epsilon");
    mc->conv_dw_kernel       = gf_get_u32(gf, "sheetsage2.encoder.conv_depthwise_kernel_size");
    mc->conv_dw_padding      = gf_get_u32(gf, "sheetsage2.encoder.conv_depthwise_padding");

    mc->rope_freq_base        = gf_get_f32(gf, "sheetsage2.rope.freq_base");
    mc->rope_dim              = gf_get_u32(gf, "sheetsage2.rope.dimension_count");
    mc->rope_inv_freq_in_file = gf_get_bool(gf, "sheetsage2.rope.inv_freq_in_file");

    c->window_seconds     = gf_get_f32(gf, "sheetsage2.window_seconds");
    c->vocab_size         = gf_get_u32(gf, "sheetsage2.vocab_size");
    c->time_hz            = gf_get_u32(gf, "sheetsage2.time_hz");
    c->max_output_seq_len = gf_get_u32(gf, "sheetsage2.max_output_seq_len");
    c->token_pad          = gf_get_u32(gf, "sheetsage2.token.pad");
    c->token_sos          = gf_get_u32(gf, "sheetsage2.token.sos");
    c->token_eos          = gf_get_u32(gf, "sheetsage2.token.eos");
    c->token_out          = gf_get_u32(gf, "sheetsage2.token.out");
    c->vocab_fingerprint  = gf_get_str(gf, "sheetsage2.vocab_fingerprint");

    c->overlap_seconds_default   = gf_get_f32(gf, "sheetsage2.window.overlap_seconds_default");
    c->lookahead_seconds_default = gf_get_f32(gf, "sheetsage2.window.lookahead_seconds_default");

    c->hidden_state_count    = gf_get_u32(gf, "sheetsage2.encoder.hidden_state_count");
    c->projection_out_length = gf_get_u32(gf, "sheetsage2.encoder.projection_out_length");

    c->dec.block_count             = gf_get_u32(gf, "sheetsage2.decoder.block_count");
    c->dec.embedding_length        = gf_get_u32(gf, "sheetsage2.decoder.embedding_length");
    c->dec.feed_forward_length     = gf_get_u32(gf, "sheetsage2.decoder.feed_forward_length");
    c->dec.head_count              = gf_get_u32(gf, "sheetsage2.decoder.attention.head_count");
    c->dec.key_length              = gf_get_u32(gf, "sheetsage2.decoder.attention.key_length");
    c->dec.value_length            = gf_get_u32(gf, "sheetsage2.decoder.attention.value_length");
    c->dec.layer_norm_eps          = gf_get_f32(gf, "sheetsage2.decoder.layer_norm_epsilon");
    c->dec.position_offset         = gf_get_u32(gf, "sheetsage2.decoder.position_offset");
    c->dec.max_position_embeddings = gf_get_u32(gf, "sheetsage2.decoder.max_position_embeddings");
    c->dec.embed_scale             = gf_get_f32(gf, "sheetsage2.decoder.embed_scale");
    c->dec.qkv_fused               = gf_get_bool(gf, "sheetsage2.decoder.attention.qkv_fused");
    c->dec.norm_style              = gf_get_str(gf, "sheetsage2.decoder.norm_style");
    c->dec.activation              = gf_get_str(gf, "sheetsage2.decoder.activation");
    c->dec.cross_attn_kv_static    = gf_get_bool(gf, "sheetsage2.decoder.cross_attn_kv_static");
    c->dec.cross_attn_masked       = gf_get_bool(gf, "sheetsage2.decoder.cross_attn_masked");
}

// Structural sanity, not taste — every check corresponds to something doc 20
// pins at runtime. A file that fails one of these produces plausible-looking,
// wrong output.
static void sheetsage_validate_config(const SheetSageConfig & c, const Yue2MertConfig & mc,
                                       std::vector<std::string> * errs) {
    auto bad = [&](const std::string & msg) {
        if (errs->size() < 24) {
            errs->push_back(msg);
        }
    };
    if (mc.sample_rate != 24000) {
        bad("sheetsage2.sample_rate is not 24000");
    }
    if (mc.block_count != 24) {
        bad("sheetsage2.encoder.block_count != 24 (doc 20 §1 item 1: ALL 24 blocks, none dropped)");
    }
    if (c.hidden_state_count != mc.block_count + 1) {
        bad("sheetsage2.encoder.hidden_state_count != block_count+1 (doc 20 §2.5: embedding + every block output)");
    }
    if (c.projection_out_length == 0) {
        bad("sheetsage2.encoder.projection_out_length is 0 or missing");
    }
    if (mc.rope_inv_freq_in_file) {
        bad("sheetsage2.rope.inv_freq_in_file is true -- inv_freq is persistent=False upstream and MUST be built "
            "(doc 20 §2.2)");
    }
    if (mc.rope_dim != mc.key_length) {
        bad("sheetsage2.rope.dimension_count != encoder.attention.key_length (MERT rotates the full head)");
    }
    if (mc.db_top_db_clamp) {
        bad("sheetsage2.mel.db_top_db_clamp is true -- doc 20 §1.3 says top_db=None, no clamp");
    }
    if (!mc.drop_last_mel_frame) {
        bad("sheetsage2.mel.drop_last_mel_frame is false -- doc 20 §1.3 drops mel[..., :-1]");
    }
    if (mc.sub_resample_padding) {
        bad("sheetsage2.subsampling.resample_padding is true -- doc 20 §2.1 says NO padding");
    }
    if (mc.sub_resample_kernel != 2) {
        bad("sheetsage2.subsampling.resample_kernel != 2");
    }
    if (mc.sub_channels.size() != mc.sub_block_count || mc.sub_depths.size() != mc.sub_block_count ||
        mc.sub_strides.size() != mc.sub_block_count) {
        bad("subsampling channels/depths/strides arrays disagree with block_count");
    }
    if (!mc.sub_channels.empty() && (uint32_t) mc.sub_channels.back() != mc.embedding_length) {
        bad("the last subsampling channel count != encoder.embedding_length");
    }
    if (mc.sub_in_channels != mc.num_mel_bins) {
        bad("subsampling.input_channels != mel.num_mel_bins");
    }
    if (mc.sub_ln_eps == mc.ln_eps) {
        // Trap, same one yue2-mert.h flags: 1e-6 (subsampling) vs 1e-5
        // (Conformer). Equal means a converter regression collapsed them.
        bad("subsampling layer_norm_epsilon == encoder layer_norm_epsilon (doc 20 §2.1/§2.3: 1e-6 vs 1e-5)");
    }
    if (mc.head_count == 0 || mc.key_length == 0 || mc.head_count * mc.key_length != mc.embedding_length) {
        bad("encoder attention head_count * key_length != embedding_length");
    }

    if (c.dec.block_count != 6) {
        bad("sheetsage2.decoder.block_count != 6");
    }
    if (c.dec.qkv_fused) {
        bad("sheetsage2.decoder.attention.qkv_fused is true -- doc 20 §3.2 says four SEPARATE projections");
    }
    if (c.dec.norm_style != "post") {
        bad("sheetsage2.decoder.norm_style is not 'post' -- doc 20 §3.2 is a post-LN decoder");
    }
    if (!c.dec.cross_attn_kv_static) {
        bad("sheetsage2.decoder.cross_attn_kv_static is false -- doc 20 §3.4 computes cross K/V once per "
            "window and reuses it verbatim every step");
    }
    if (c.dec.cross_attn_masked) {
        bad("sheetsage2.decoder.cross_attn_masked is true -- doc 20 §3.4: no mask over padded memory frames, ever");
    }
    if (c.dec.head_count == 0 || c.dec.key_length == 0 ||
        c.dec.head_count * c.dec.key_length != c.dec.embedding_length) {
        bad("decoder attention head_count * key_length != embedding_length");
    }
    if (c.dec.max_position_embeddings != c.max_output_seq_len + c.dec.position_offset) {
        bad("decoder.max_position_embeddings != max_output_seq_len + position_offset "
            "(doc 20 §3.1: [5122,512] = [5120+2,512])");
    }
    if (c.vocab_size == 0) {
        bad("sheetsage2.vocab_size is 0 or missing");
    }
    if (c.vocab_fingerprint.empty()) {
        bad("sheetsage2.vocab_fingerprint is missing");
    }
}

// ── Loader ──────────────────────────────────────────────────────────────────

// Never calls yue2_mert_free() on `m->mert` — that would call wctx_free() on
// `m->mert.wctx`, which shares its `.buffer` pointer with `m->wctx` (see the
// SheetSageModel field comment above) and would double-free it. A plain
// struct reset is safe: `Yue2MertModel` owns no resources through its OWN
// destructor path when `wctx.ctx == nullptr` and `backend_ref == false`,
// both of which sheetsage_model_load() below guarantees for this member.
static void sheetsage_model_free(SheetSageModel * m) {
    wctx_free(&m->wctx);
    m->mert = Yue2MertModel{};
    m->head = SheetSageEncoderHead{};
    m->dec  = SheetSageDecoderWeights{};
    m->tmap.clear();
    m->vram  = 0;
    m->exact = false;
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
}

// Load a sheetsage2-<type>.gguf: config, all 24 encoder blocks + mel/
// subsampling + layer_weight + projection, and the full 6-layer decoder.
// One backend buffer for everything (encoder + decoder + head), since a
// transcribe() call needs both halves resident together.
static bool sheetsage_model_load(SheetSageModel * m, const std::string & path, std::string * err,
                                  const SheetSageModelLoadOptions & load_opt = {}) {
    const auto t0 = std::chrono::steady_clock::now();
    sheetsage_model_free(m);
    m->path = path;

    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    m->arch    = gf_get_str(gf, "general.architecture");
    m->license = gf_get_str(gf, "general.license");
    if (m->arch != "sheetsage2") {
        if (err) {
            *err = "architecture is '" + m->arch + "', expected 'sheetsage2' (" + path + ")";
        }
        gf_close(&gf);
        return false;
    }

    sheetsage_parse_config(gf, &m->cfg, &m->mert.cfg);
    std::vector<std::string> errs;
    sheetsage_validate_config(m->cfg, m->mert.cfg, &errs);
    if (!errs.empty()) {
        if (err) {
            *err = "sheetsage2 config rejected: " + errs[0] +
                   (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        return false;
    }

    const Yue2MertConfig & mc = m->mert.cfg;
    const int64_t          NB = (int64_t) mc.sub_block_count;   // 3
    const int64_t          NL = (int64_t) mc.block_count;       // 24
    const int64_t          DL = (int64_t) m->cfg.dec.block_count;  // 6

    BackendPair bp = backend_init("SheetSage2");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->backend_ref = true;

    // 1039 tensors total in the shipped GGUF (doc 20 §4.2's accounting); a
    // little headroom costs nothing.
    wctx_init(&m->wctx, 1039 + 16);
    Yue2Loader ld{ &m->wctx, &gf, &m->tmap, &errs };

    // ---- encoder: mel frontend + ConvNext subsampling (doc 20 §1.3/§2.1) ---
    const int64_t NM = (int64_t) mc.num_mel_bins;
    const int64_t NF = (int64_t) mc.num_freq_bins;
    m->mert.w.mel_window = ld.req("enc.mel.stft_window", (int64_t) mc.n_fft);
    m->mert.w.mel_fb     = ld.req("enc.mel.filterbank", NM, NF);  // ne = (mel, freq)
    m->mert.w.mel_mean   = ld.req("enc.mel.mean", NM);
    m->mert.w.mel_std    = ld.req("enc.mel.std", NM);

    std::vector<int64_t> chans;
    chans.push_back((int64_t) mc.sub_in_channels);
    for (int64_t b = 0; b < NB; b++) {
        chans.push_back((int64_t) mc.sub_channels[(size_t) b]);
    }
    m->mert.w.sub.assign((size_t) NB, Yue2MertSubBlock{});
    for (int64_t b = 0; b < NB && errs.empty(); b++) {
        const int64_t cin      = chans[(size_t) b];
        const int64_t cout     = chans[(size_t) (b + 1)];
        const int64_t s        = (int64_t) mc.sub_strides[(size_t) b];
        const bool    identity = (cin == cout && s == 1);
        Yue2MertSubBlock & sb  = m->mert.w.sub[(size_t) b];
        char               nm[96];

        if (!identity) {
            snprintf(nm, sizeof(nm), "enc.sub.%d.down_norm.weight", (int) b);
            sb.down_norm_w = ld.req(nm, cin);
            snprintf(nm, sizeof(nm), "enc.sub.%d.down_norm.bias", (int) b);
            sb.down_norm_b = ld.req(nm, cin);
            snprintf(nm, sizeof(nm), "enc.sub.%d.down_conv.weight", (int) b);
            sb.down_conv_w = ld.req(nm, (int64_t) mc.sub_resample_kernel, cin, cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.down_conv.bias", (int) b);
            sb.down_conv_b = ld.req(nm, cout);
        }

        const int64_t depth = (int64_t) mc.sub_depths[(size_t) b];
        sb.cnx.assign((size_t) depth, Yue2MertCnxLayer{});
        for (int64_t l = 0; l < depth && errs.empty(); l++) {
            Yue2MertCnxLayer & cl = sb.cnx[(size_t) l];
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.dw.weight", (int) b, (int) l);
            cl.dw_w = ld.req(nm, (int64_t) mc.sub_convnext_kernel, cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.dw.bias", (int) b, (int) l);
            cl.dw_b = ld.req(nm, cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.norm.weight", (int) b, (int) l);
            cl.norm_w = ld.req(nm, cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.norm.bias", (int) b, (int) l);
            cl.norm_b = ld.req(nm, cout);
            // Doc 23 "exact" mode: pw_up/pw_down are the subsampling stack's
            // only weight-bearing matmuls (NATIVE policy, doc 20 §4.2) and
            // feed hidden_00 (the post-subsampling embedding) BEFORE any
            // Conformer block runs — the measured 3.1e-4 rel-L2 floor at
            // hidden_00 originates here, not in the blocks.
            ld.force_f32 = load_opt.exact;
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.pw_up.weight", (int) b, (int) l);
            cl.pw_up_w = ld.req(nm, cout, 4 * cout);
            ld.force_f32 = false;
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.pw_up.bias", (int) b, (int) l);
            cl.pw_up_b = ld.req(nm, 4 * cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.grn.weight", (int) b, (int) l);
            cl.grn_w = ld.req(nm, 4 * cout);
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.grn.bias", (int) b, (int) l);
            cl.grn_b = ld.req(nm, 4 * cout);
            ld.force_f32 = load_opt.exact;
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.pw_down.weight", (int) b, (int) l);
            cl.pw_down_w = ld.req(nm, 4 * cout, cout);
            ld.force_f32 = false;
            snprintf(nm, sizeof(nm), "enc.sub.%d.cnx.%d.pw_down.bias", (int) b, (int) l);
            cl.pw_down_b = ld.req(nm, cout);
        }
    }

    // ---- encoder: 24 Conformer blocks, none dropped ------------------------
    if (errs.empty()) {
        const int64_t H = (int64_t) mc.embedding_length;
        const int64_t F = (int64_t) mc.feed_forward_length;
        m->mert.w.blk.assign((size_t) NL, Yue2MertBlock{});
        for (int64_t n = 0; n < NL && errs.empty(); n++) {
            Yue2MertBlock & bk = m->mert.w.blk[(size_t) n];
            char            nm[96];
#define SS2_ENC_T(field, fmtstr, ...)                                       \
    do {                                                                    \
        snprintf(nm, sizeof(nm), fmtstr, (int) n);                          \
        bk.field = ld.req(nm, __VA_ARGS__);                                 \
    } while (0)
            SS2_ENC_T(ffn1_norm_w, "enc.blk.%d.ffn1_norm.weight", H);
            SS2_ENC_T(ffn1_norm_b, "enc.blk.%d.ffn1_norm.bias", H);
            // "exact" mode (doc 23): force every weight-bearing matmul in the
            // block to F32, the same loader trick as block 23's attn special
            // case below, generalized. Biases are always F32 already
            // (policy, doc 20 §4.4) so leaving force_f32 on for them is
            // harmless; toggled off between weight/bias pairs anyway for
            // clarity at the call site.
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(ffn1_up_w, "enc.blk.%d.ffn1_up.weight", H, F);
            ld.force_f32 = false;
            SS2_ENC_T(ffn1_up_b, "enc.blk.%d.ffn1_up.bias", F);
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(ffn1_down_w, "enc.blk.%d.ffn1_down.weight", F, H);
            ld.force_f32 = false;
            SS2_ENC_T(ffn1_down_b, "enc.blk.%d.ffn1_down.bias", H);
            SS2_ENC_T(ffn2_norm_w, "enc.blk.%d.ffn2_norm.weight", H);
            SS2_ENC_T(ffn2_norm_b, "enc.blk.%d.ffn2_norm.bias", H);
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(ffn2_up_w, "enc.blk.%d.ffn2_up.weight", H, F);
            ld.force_f32 = false;
            SS2_ENC_T(ffn2_up_b, "enc.blk.%d.ffn2_up.bias", F);
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(ffn2_down_w, "enc.blk.%d.ffn2_down.weight", F, H);
            ld.force_f32 = false;
            SS2_ENC_T(ffn2_down_b, "enc.blk.%d.ffn2_down.bias", H);
            SS2_ENC_T(attn_norm_w, "enc.blk.%d.attn_norm.weight", H);
            SS2_ENC_T(attn_norm_b, "enc.blk.%d.attn_norm.bias", H);

            // Block 23's attention projections carry doc 20 §5.3's flagged
            // outlier LoRA merge: out_proj rel-L2 15.2 / max|delta| 44.2
            // against a base weight whose own max magnitude is ~0.99 (v_proj
            // rel-L2 3.97 is the next-largest). Every other block, and every
            // non-attention tensor in THIS block, differs from base by no
            // more than ordinary rounding (§5.2) — this is the one place in
            // the encoder where an F16-stored value moves far enough from
            // its base magnitude that the file's own F16 rounding plus a
            // second, in-matmul F16 narrowing (see yue2-model.h's
            // Yue2Loader::force_f32 comment, the same fix yue2-tok-head.h
            // already applies for an identical reason) compound into G1's
            // long-b failures (rel-L2 up to 1.90e-2 against the 1.0e-2 line,
            // isolated to windows overlapping this block's own extreme
            // weights). force_f32 upcasts the stored F16 values to F32
            // losslessly at load time and routes attn_q/k/v/o's mul_mat
            // through the CUDA F32 (cublasSgemm) path instead of an F16
            // src0 path; it does NOT recover precision the GGUF's own F16
            // storage already discarded (that needs an F32 weight file,
            // outside this lane's converter). Cost: 4 * 1024x1024 tensors
            // promoted from F16 to F32, +8 MB VRAM.
            // load_opt.exact forces this for EVERY block; block 23 forces it
            // unconditionally regardless of load_opt (kept from before this
            // option existed — its LoRA-outlier out_proj is bad enough on
            // its own, per §5.3, to warrant the promotion even on the fast
            // path).
            const bool blk23_attn_f32 = (n == NL - 1);
            if (blk23_attn_f32 || load_opt.exact) {
                ld.force_f32 = true;
            }
            SS2_ENC_T(attn_q_w, "enc.blk.%d.attn_q.weight", H, H);
            SS2_ENC_T(attn_q_b, "enc.blk.%d.attn_q.bias", H);
            SS2_ENC_T(attn_k_w, "enc.blk.%d.attn_k.weight", H, H);
            SS2_ENC_T(attn_k_b, "enc.blk.%d.attn_k.bias", H);
            SS2_ENC_T(attn_v_w, "enc.blk.%d.attn_v.weight", H, H);
            SS2_ENC_T(attn_v_b, "enc.blk.%d.attn_v.bias", H);
            SS2_ENC_T(attn_o_w, "enc.blk.%d.attn_output.weight", H, H);
            SS2_ENC_T(attn_o_b, "enc.blk.%d.attn_output.bias", H);
            if (blk23_attn_f32 || load_opt.exact) {
                ld.force_f32 = false;
            }
            SS2_ENC_T(conv_norm_w, "enc.blk.%d.conv_norm.weight", H);
            SS2_ENC_T(conv_norm_b, "enc.blk.%d.conv_norm.bias", H);
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(conv_pw1_w, "enc.blk.%d.conv_pw1.weight", H, 2 * H);
            ld.force_f32 = false;
            SS2_ENC_T(conv_dw_w, "enc.blk.%d.conv_dw.weight", (int64_t) mc.conv_dw_kernel, H);
            SS2_ENC_T(conv_dw_norm_w, "enc.blk.%d.conv_dw_norm.weight", H);
            SS2_ENC_T(conv_dw_norm_b, "enc.blk.%d.conv_dw_norm.bias", H);
            ld.force_f32 = load_opt.exact;
            SS2_ENC_T(conv_pw2_w, "enc.blk.%d.conv_pw2.weight", H, H);
            ld.force_f32 = false;
            SS2_ENC_T(final_norm_w, "enc.blk.%d.final_norm.weight", H);
            SS2_ENC_T(final_norm_b, "enc.blk.%d.final_norm.bias", H);
#undef SS2_ENC_T
        }
        m->mert.blocks_loaded = errs.empty();
    }

    // ---- encoder head: layer_weight (raw) + projection (doc 20 §2.5) -------
    if (errs.empty()) {
        const int64_t H = (int64_t) mc.embedding_length;
        const int64_t P = (int64_t) m->cfg.projection_out_length;
        m->head.layer_w = ld.req("enc.layer_w", (int64_t) m->cfg.hidden_state_count);
        // proj_w feeds 06_memory directly (the decisive gate) — include it
        // in "exact" mode for the same reason as every block's weights above.
        ld.force_f32   = load_opt.exact;
        m->head.proj_w = ld.req("enc.proj.weight", H, P);
        ld.force_f32   = false;
        m->head.proj_b = ld.req("enc.proj.bias", P);
    }

    // ---- decoder: loaded here, graph lives in sheetsage-decoder.h ----------
    if (errs.empty()) {
        const int64_t D  = (int64_t) m->cfg.dec.embedding_length;
        const int64_t FF = (int64_t) m->cfg.dec.feed_forward_length;
        const int64_t V  = (int64_t) m->cfg.vocab_size;
        const int64_t MP = (int64_t) m->cfg.dec.max_position_embeddings;
        m->dec.tok_embd    = ld.req("dec.tok_embd", D, V);
        m->dec.pos_embd    = ld.req("dec.pos_embd", D, MP);
        m->dec.norm_embd_w = ld.req("dec.norm_embd.weight", D);
        m->dec.norm_embd_b = ld.req("dec.norm_embd.bias", D);
        m->dec.blk.assign((size_t) DL, SheetSageDecoderBlock{});
        for (int64_t bi = 0; bi < DL && errs.empty(); bi++) {
            SheetSageDecoderBlock & db = m->dec.blk[(size_t) bi];
            char                    nm[96];
#define SS2_DEC_T(field, fmtstr, ...)                                       \
    do {                                                                    \
        snprintf(nm, sizeof(nm), fmtstr, (int) bi);                         \
        db.field = ld.req(nm, __VA_ARGS__);                                 \
    } while (0)
            SS2_DEC_T(self_attn_norm_w, "dec.blk.%d.self_attn_norm.weight", D);
            SS2_DEC_T(self_attn_norm_b, "dec.blk.%d.self_attn_norm.bias", D);
            SS2_DEC_T(self_attn_q_w, "dec.blk.%d.self_attn_q.weight", D, D);
            SS2_DEC_T(self_attn_q_b, "dec.blk.%d.self_attn_q.bias", D);
            SS2_DEC_T(self_attn_k_w, "dec.blk.%d.self_attn_k.weight", D, D);
            SS2_DEC_T(self_attn_k_b, "dec.blk.%d.self_attn_k.bias", D);
            SS2_DEC_T(self_attn_v_w, "dec.blk.%d.self_attn_v.weight", D, D);
            SS2_DEC_T(self_attn_v_b, "dec.blk.%d.self_attn_v.bias", D);
            SS2_DEC_T(self_attn_o_w, "dec.blk.%d.self_attn_output.weight", D, D);
            SS2_DEC_T(self_attn_o_b, "dec.blk.%d.self_attn_output.bias", D);
            SS2_DEC_T(cross_attn_norm_w, "dec.blk.%d.cross_attn_norm.weight", D);
            SS2_DEC_T(cross_attn_norm_b, "dec.blk.%d.cross_attn_norm.bias", D);
            SS2_DEC_T(cross_attn_q_w, "dec.blk.%d.cross_attn_q.weight", D, D);
            SS2_DEC_T(cross_attn_q_b, "dec.blk.%d.cross_attn_q.bias", D);
            SS2_DEC_T(cross_attn_k_w, "dec.blk.%d.cross_attn_k.weight", D, D);
            SS2_DEC_T(cross_attn_k_b, "dec.blk.%d.cross_attn_k.bias", D);
            SS2_DEC_T(cross_attn_v_w, "dec.blk.%d.cross_attn_v.weight", D, D);
            SS2_DEC_T(cross_attn_v_b, "dec.blk.%d.cross_attn_v.bias", D);
            SS2_DEC_T(cross_attn_o_w, "dec.blk.%d.cross_attn_output.weight", D, D);
            SS2_DEC_T(cross_attn_o_b, "dec.blk.%d.cross_attn_output.bias", D);
            SS2_DEC_T(ffn_up_w, "dec.blk.%d.ffn_up.weight", D, FF);
            SS2_DEC_T(ffn_up_b, "dec.blk.%d.ffn_up.bias", FF);
            SS2_DEC_T(ffn_down_w, "dec.blk.%d.ffn_down.weight", FF, D);
            SS2_DEC_T(ffn_down_b, "dec.blk.%d.ffn_down.bias", D);
            SS2_DEC_T(final_norm_w, "dec.blk.%d.final_norm.weight", D);
            SS2_DEC_T(final_norm_b, "dec.blk.%d.final_norm.bias", D);
#undef SS2_DEC_T
        }
    }

    if (!errs.empty()) {
        if (err) {
            *err = errs[0] + (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        sheetsage_model_free(m);
        return false;
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the SheetSage2 weights";
        }
        gf_close(&gf);
        sheetsage_model_free(m);
        return false;
    }
    gf_close(&gf);  // safe: wctx_alloc has copied everything to the backend

    m->vram = m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;

    // Wire the reused Yue2MertModel view (see the SheetSageModel field
    // comment above): identity token only, no second residency, and
    // yue2_mert_free() is NEVER called on m->mert.
    m->mert.wctx.buffer = m->wctx.buffer;
    m->mert.arch        = "sheetsage2";
    if (!yue2_mert_build_mel_plan(&m->mert, err)) {
        sheetsage_model_free(m);
        return false;
    }

    m->exact   = load_opt.exact;
    m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr,
            "[SheetSage2] Loaded %s: %zu tensors, %.1f MB, enc_blocks=%zu dec_blocks=%zu, %.0f ms%s\n",
            yue2_basename(path).c_str(), m->tmap.size(), (double) m->vram / (1024.0 * 1024.0), m->mert.w.blk.size(),
            m->dec.blk.size(), m->load_ms, load_opt.exact ? " [exact: encoder matmul weights forced F32]" : "");
    return true;
}
