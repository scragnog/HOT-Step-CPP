#pragma once
// yue2/yue2-mert.h — MERT-v2-FullSong front half: GGUF loader, mel frontend,
// ConvNext subsampling stack, RoPE tables.
//
// HOT-Step file (no acestep.cpp analog). This is the audio side of
// Mothersuperior's real-audio semantic tokenizer: the chain that turns a mono
// 24 kHz waveform into the 25 Hz, 1024-wide features the tokenizer head reads.
//
//   audio -> mono 24 kHz -> [mel frontend] -> [ConvNext subsampling] ->
//            [21 Conformer blocks] -> hidden_states[20] -> interpolate to T25 ->
//            per-track instance norm -> 8-layer head -> argmax -> code
//
// THIS FILE COVERS everything on that line except the head: the mel frontend,
// the ConvNext subsampling stack, the RoPE tables, all 21 Conformer blocks,
// the interpolation to T25 and the per-track instance norm. yue2_mert_encode()
// is the whole chain, audio in and the head's input out. The head itself is
// yue2-tok-head.h; its weights are loaded from the same GGUF by a separate
// loader, and this file never touches them.
//
// Gated by yue2-probe --mert-front-parity (mel/subsampling/RoPE) and
// --mert-block-parity (the blocks, the whole-track chain, and the end-to-end
// code agreement). Numbers live in those modes' headers, not here.
//
// AUTHORITY: docs/plans/yue2/12-tokenizer-oracle-pin.md. That document was
// produced by RUNNING the reference implementation, not by reading it, and it
// pins every tensor name, shape, epsilon and axis order below. Where this file
// and that document disagree, the document wins. Section references (§3.3,
// §3.4, §3.6 …) point into it.
//
// Weights come from engine/tools/convert-yue2-tok.py: one GGUF, arch
// "yue2-tok", 886 tensors (783 mert.* + 103 tok.*). Every tensor name and
// shape below was checked against the real file (models/yue2/yue2-tok-f16.gguf)
// with the `gguf` python package, not taken on trust from the converter source.
//
// ── SIX TRAPS THE PIN ESTABLISHED AT RUNTIME. DO NOT REGRESS THEM. ──────────
//
// 1. The STFT window and the mel filterbank SHIP AS CHECKPOINT BUFFERS
//    (mert.mel.stft_window [2048], mert.mel.filterbank [1025,128], pin §3.2).
//    Do not construct a Hann window. Do not reproduce a Slaney/HTK triangular
//    bank. Load them. The same goes for mert.mel.{mean,std}: those are the
//    checkpoint's FIXED dataset statistics, not per-clip statistics.
//
// 2. amplitude_to_dB is `10 * log10(max(x, 1e-10))` and NOTHING ELSE
//    (pin §3.3). torchaudio's AmplitudeToDB normally also applies a `top_db`
//    clamp relative to the running maximum; MERT constructs it with
//    top_db=None, so there is no clamp and no coupling between frames. A port
//    that "helpfully" adds whisper-style dynamic-range clamping (which is what
//    engine/src/moss/moss-mel.h does — see REUSE NOTE below) is wrong.
//
// 3. THE LAST MEL FRAME IS DROPPED (`mel[..., :-1]`, pin §3.3). center=True
//    gives S//hop + 1 frames; after the drop, exactly S//hop, i.e. 3000 for a
//    30 s chunk and 3000/4 = 750 = 25 Hz out. Off by one here shifts the whole
//    time axis by 10 ms and every downstream gate fails by a mile.
//
// 4. THE RESAMPLING CONV HAS KERNEL 2 AND NO PADDING (pin §3.4), so
//    out_len = floor((L - 2)/stride) + 1, which is L/2 only for even L. A
//    27.8 s tail gives 2783 -> 1391 -> 695 mel/sub frames, and 695 is what the
//    reference produced. Assuming L/2 costs a frame on every odd-length tail.
//
// 5. GlobalResponseNorm NORMALISES OVER THE TIME AXIS OF THE WHOLE CHUNK
//    (pin §3.4). GRN is therefore sequence-length dependent, and padding
//    contaminates it — which is why the reference batches only equal-length
//    30 s chunks and runs every short tail ALONE. yue2_mert_chunk_plan() below
//    encodes that rule; a port that zero-pads a tail up to 30 s gets different
//    features for the ENTIRE tail, not just its padded region.
//
// 6. TWO DIFFERENT LayerNorm EPSILONS. Everything inside the subsampling stack
//    is 1e-6 (`subsampling_layer_norm_eps`); everything inside the Conformer
//    blocks is 1e-5 (`layer_norm_eps`). They are one keystroke apart and a
//    port that uses one value throughout still runs.
//
// Plus one that belongs to the RoPE tables (pin §3.6): positions RESTART AT 0
// in every 30 s chunk. MERT carries no absolute song position anywhere, which
// is exactly what makes the chunked extraction legitimate.
//
// ── REUSE NOTE: why this file carries its own STFT and mel ──────────────────
//
// Two mel/STFT implementations already exist in engine/src/. Neither can be
// driven to MERT's parameters, so this is a deliberate third one, not an
// oversight:
//
//   * engine/src/moss/moss-mel.h — Whisper's frontend. Its filterbank is built
//     in code (Slaney scale AND Slaney area normalisation, moss-mel.h:97-155)
//     where MERT ships an unnormalised HTK bank as a weight; its dB stage is a
//     global-maximum clamp plus a (log+4)/4 rescale (moss-mel.h:226-237) where
//     MERT is a bare 10*log10; and its window is hardcoded. Every one of those
//     is a hardcoded constant, not a parameter. There is nothing to drive.
//
//   * engine/src/supersep-stft.h — a real parameterised STFT, but it ZERO-pads
//     the edges (supersep-stft.h:130-143 substitutes 0.0f for out-of-range
//     samples) where torch's center=True uses pad_mode="reflect", and it builds
//     its own Hann window (supersep-stft.h:79). Reflect vs zero padding changes
//     the first and last ~4 frames of every chunk — and with 30 s chunking
//     there is a seam every 750 output frames, so this is not a negligible
//     edge effect.
//
// The FFT below is a plain iterative radix-2 in double. n_fft = 2048 is a pure
// power of two, so none of moss-mel.h's awkwardness (n_fft = 400 = 2^4 * 25,
// forcing a recursive split with a naive-DFT base case) applies here. Double
// precision is deliberately MORE accurate than the reference, which runs the
// STFT in fp32 — the residual lands far below the fixture bars either way, and
// this keeps the header free of vendor dependencies (yue2-probe links exactly
// yyjson + ggml; engine/CMakeLists.txt:654-656 and the ace-train comment at
// :684-689 both rely on engine/src/yue2/ staying dependency-free).
//
// ── LAYOUT CONVENTION ───────────────────────────────────────────────────────
//
// Activations in the graph are ne = [C, T]: channel fastest, time slowest.
// That is byte-for-byte torch's row-major [B, T, C], and it is also byte-for-
// byte the fixtures' own row-major [T, C] dumps — 01_mel.f32 is [3000,128] and
// 02_subsampled.f32 is [750,1024], both of which read back with no transpose.
// The convs want the opposite order (ggml_im2col reads time along ne0), so
// every conv is bracketed by a transpose, exactly mirroring the reference's
// own Transpose modules (modeling_mert2.py:72-74).
//
// ── PRECISION, AND THE ONE PLACE THE FIXTURES ARE NOT FP32 ─────────────────
//
// The reference frontend runs in float32 with autocast explicitly DISABLED
// inside (modeling_mert2.py:65), so there is no bf16 anywhere in this half of
// the chain. Measured against the stage fixtures by yue2-probe
// --mert-front-parity:
//
//     mel                   6.78e-07 rel-L2
//     rope cos / sin        2.4e-08 / 3.0e-08
//     subsampling stack     2.05e-06   vs a TRUE fp32 reference
//
// TRAP, and it is the fixtures' rather than ours: 02_subsampled.f32 is NOT a
// pure fp32 dump. stage_fixtures.py sets no TF32 flags, and PyTorch's defaults
// are asymmetric — `cuda.matmul.allow_tf32` is False but `cudnn.allow_tf32` is
// TRUE — so all 14 Conv1d calls in the subsampling stack ran on cuDNN at
// TF32's 10-bit mantissa. Re-running the same torch code on the same mel
// reproduces the shipped fixture BIT-EXACTLY with cudnn.allow_tf32=True and
// moves by 5.593e-04 with it False, which is exactly the number this port
// scores against the shipped file. So a ~5.6e-4 result on that stage is the
// reference's rounding, not a port defect; see the gate discussion in
// yue2-probe.cpp's --mert-front-parity header.
//
// The other slack is the GGUF's own storage type — in yue2-tok-f16.gguf the
// two pointwise matmul weights per ConvNext layer are F16, measured at 2.50e-4
// rel-L2 through the stack. yue2_mert_weights_are_f32() reports which regime
// is loaded so a caller can state the right bar instead of guessing.
//
// OUR side's TF32: ggml does not route these shapes through a TF32 cuBLAS path
// (turning it on moves the result by less than the fixture's own noise), but a
// caller that wants the guarantee should call yue2_tok_disable_tf32()
// (yue2-tok-head.h) or yue2_vae_enc_disable_tf32() (yue2-vae-encode.h) BEFORE
// the first ggml_backend_cuda_init — the driver reads NVIDIA_TF32_OVERRIDE
// when the context is created, so a later call is a silent no-op. This header
// deliberately does not define a third copy of that one-liner.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "yue2-model.h"  // WeightCtx/Yue2Loader/backend machinery, gguf-weights.h

// ── Config ──────────────────────────────────────────────────────────────────

// Every field is read from the GGUF's yue2tok.mert.* KV block; nothing here is
// hardcoded architecture knowledge. The converter wrote these from
// MERT-v2-FullSong/config.json after cross-checking each one
// (convert-yue2-tok.py:740-762), so a config drift shows up as a load error
// rather than as silently wrong audio features.
struct Yue2MertConfig {
    // frontend
    uint32_t sample_rate      = 0;  // 24000
    uint32_t n_fft            = 0;  // 2048
    uint32_t win_length       = 0;  // 2048 (== n_fft; no centring of a shorter window)
    uint32_t hop_length       = 0;  // 240
    uint32_t num_mel_bins     = 0;  // 128
    uint32_t num_freq_bins    = 0;  // 1025 (onesided)
    float    frame_rate       = 0.0f;   // 25.0
    uint32_t samples_per_frame = 0;     // 960 = 24000/25 = hop*4
    uint32_t minimum_input_samples = 0; // 1025 = n_fft/2 + 1: reflect padding needs it
    float    spectrogram_power = 0.0f;  // 2.0 (power, not magnitude)
    float    db_amin           = 0.0f;  // 1e-10
    float    db_multiplier     = 0.0f;  // 10.0
    bool     db_top_db_clamp   = false; // MUST be false — trap 2
    bool     drop_last_mel_frame = false;  // MUST be true — trap 3
    float    mel_std_floor     = 0.0f;  // 1e-5, as clamp_min(std), NOT std + eps

    // subsampling
    uint32_t             sub_block_count   = 0;  // 3
    uint32_t             sub_in_channels   = 0;  // 128 (= num_mel_bins)
    std::vector<int32_t> sub_channels;           // [128, 512, 1024]
    std::vector<int32_t> sub_depths;             // [3, 4, 5]
    std::vector<int32_t> sub_strides;            // [1, 2, 2]
    uint32_t             sub_resample_kernel = 0;  // 2
    bool                 sub_resample_padding = true;  // MUST be false — trap 4
    uint32_t             sub_convnext_kernel  = 0;     // 7
    uint32_t             sub_convnext_padding = 0;     // 3
    float                sub_ln_eps           = 0.0f;  // 1e-6 — trap 6
    float                sub_grn_eps          = 0.0f;  // 1e-6

    // Conformer stack (loaded here, run elsewhere)
    uint32_t embedding_length     = 0;  // 1024
    uint32_t feed_forward_length  = 0;  // 4096
    uint32_t block_count          = 0;  // 21 — what the GGUF carries
    uint32_t block_count_upstream = 0;  // 24 — what MERT ships
    uint32_t hidden_state_index   = 0;  // 20 — the block we stop at (pin §3.1)
    uint32_t head_count           = 0;  // 16
    uint32_t key_length           = 0;  // 64 = 1024/16
    float    ln_eps               = 0.0f;  // 1e-5 — trap 6
    uint32_t conv_dw_kernel       = 0;  // 31
    uint32_t conv_dw_padding      = 0;  // 15

    // RoPE
    float    rope_freq_base = 0.0f;  // 10000
    uint32_t rope_dim       = 0;     // 64 (== key_length)
    bool     rope_inv_freq_in_file = true;  // MUST be false: persistent=False, we build it
};

// ── Weights ─────────────────────────────────────────────────────────────────

// One ConvNext layer. Residual wraps the WHOLE pair (pin §3.4):
//     out = h + pointwise_block(depthwise_block(h))
// The depthwise output is NOT separately residual-added.
struct Yue2MertCnxLayer {
    ggml_tensor * dw_w      = nullptr;  // [7, C]      depthwise Conv1d k=7 pad=3 groups=C
    ggml_tensor * dw_b      = nullptr;  // [C]
    ggml_tensor * norm_w    = nullptr;  // [C]         LayerNorm eps 1e-6
    ggml_tensor * norm_b    = nullptr;  // [C]
    ggml_tensor * pw_up_w   = nullptr;  // [C, 4C]     Linear C -> 4C
    ggml_tensor * pw_up_b   = nullptr;  // [4C]
    ggml_tensor * grn_w     = nullptr;  // [4C]        GlobalResponseNorm scale
    ggml_tensor * grn_b     = nullptr;  // [4C]
    ggml_tensor * pw_down_w = nullptr;  // [4C, C]     Linear 4C -> C
    ggml_tensor * pw_down_b = nullptr;  // [C]
};

// One ConvNext block: optional resampling, then `depth` ConvNext layers.
// Block 0 is 128 -> 128 at stride 1, so its resampling_layer is nn.Identity
// upstream and the GGUF carries NO down_* tensors for it at all.
struct Yue2MertSubBlock {
    ggml_tensor * down_norm_w = nullptr;  // [Cin]            LayerNorm eps 1e-6
    ggml_tensor * down_norm_b = nullptr;  // [Cin]
    ggml_tensor * down_conv_w = nullptr;  // [2, Cin, Cout]   Conv1d k=2 stride=s pad=0
    ggml_tensor * down_conv_b = nullptr;  // [Cout]
    std::vector<Yue2MertCnxLayer> cnx;
};

// One Conformer block (pin §3.5). Not run by this file; loaded because the
// names are pinned and splitting the loader would invite a second, divergent
// copy of it. Residual structure, for whoever writes the graph:
//     h = h + 0.5 * ffn1(ffn1_norm(h))
//     h = attn(attn_norm(h), rope) + h
//     h = conv(h) + h
//     h = h + 0.5 * ffn2(ffn2_norm(h))
//     h = final_norm(h)     <- NOT a residual; the block OUTPUT is normed,
//                              and hidden_states[20] is post THIS for block 20
struct Yue2MertBlock {
    ggml_tensor * ffn1_norm_w = nullptr, * ffn1_norm_b = nullptr;
    ggml_tensor * ffn1_up_w   = nullptr, * ffn1_up_b   = nullptr;   // 1024 -> 4096
    ggml_tensor * ffn1_down_w = nullptr, * ffn1_down_b = nullptr;   // 4096 -> 1024
    ggml_tensor * ffn2_norm_w = nullptr, * ffn2_norm_b = nullptr;
    ggml_tensor * ffn2_up_w   = nullptr, * ffn2_up_b   = nullptr;
    ggml_tensor * ffn2_down_w = nullptr, * ffn2_down_b = nullptr;
    ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    // FOUR SEPARATE projections with bias. MERT does NOT fuse qkv (unlike the
    // tokenizer head, pin §1.5). Do not concatenate them.
    ggml_tensor * attn_q_w = nullptr, * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr, * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr, * attn_v_b = nullptr;
    ggml_tensor * attn_o_w = nullptr, * attn_o_b = nullptr;
    ggml_tensor * conv_norm_w    = nullptr, * conv_norm_b = nullptr;
    ggml_tensor * conv_pw1_w     = nullptr;  // [1024, 2048] -> GLU(dim=1), NO bias
    ggml_tensor * conv_dw_w      = nullptr;  // [31, 1024]   depthwise, NO bias
    ggml_tensor * conv_dw_norm_w = nullptr, * conv_dw_norm_b = nullptr;
    ggml_tensor * conv_pw2_w     = nullptr;  // [1024, 1024] NO bias
    ggml_tensor * final_norm_w   = nullptr, * final_norm_b = nullptr;
};

struct Yue2MertWeights {
    ggml_tensor * mel_window = nullptr;  // [2048]       shipped STFT window — trap 1
    ggml_tensor * mel_fb     = nullptr;  // [128, 1025]  ne0=mel, ne1=freq
    ggml_tensor * mel_mean   = nullptr;  // [128]
    ggml_tensor * mel_std    = nullptr;  // [128]
    std::vector<Yue2MertSubBlock> sub;
    std::vector<Yue2MertBlock>    blk;   // empty unless loaded with want_blocks
};

// Host-side copy of everything the CPU frontend needs, pulled back out of the
// backend buffer once at load. The frontend is CPU-only by design (there is no
// FFT op in ggml, and the reference runs this stage in fp32 with autocast off
// anyway), so it cannot read the weights where they live.
struct Yue2MertMelPlan {
    std::vector<float>   window;   // [n_fft]
    std::vector<float>   fb_mt;    // [n_mel][n_freq] — TRANSPOSED from the GGUF
    std::vector<float>   mean;     // [n_mel]
    std::vector<float>   inv_std;  // [n_mel] = 1 / max(std, 1e-5), precomputed
    std::vector<int32_t> fb_lo;    // [n_mel] first non-zero freq bin
    std::vector<int32_t> fb_hi;    // [n_mel] one past the last non-zero freq bin
};

struct Yue2MertModel {
    std::string     path;
    std::string     arch;     // must be "yue2-tok"
    std::string     license;
    Yue2MertConfig  cfg;
    Yue2MertWeights w;
    Yue2MertMelPlan mel;

    bool           blocks_loaded = false;
    bool           backend_ref   = false;
    ggml_backend_t backend       = nullptr;
    ggml_backend_t cpu_backend   = nullptr;
    WeightCtx      wctx          = {};
    size_t         vram          = 0;
    double         load_ms       = 0.0;

    std::map<std::string, ggml_tensor *> tmap;  // introspection only
};

// ── Config parsing / validation ─────────────────────────────────────────────

static void yue2_mert_parse_config(const GGUFModel & gf, Yue2MertConfig * c) {
    c->sample_rate           = gf_get_u32(gf, "yue2tok.mert.sample_rate");
    c->n_fft                 = gf_get_u32(gf, "yue2tok.mert.n_fft");
    c->win_length            = gf_get_u32(gf, "yue2tok.mert.win_length");
    c->hop_length            = gf_get_u32(gf, "yue2tok.mert.hop_length");
    c->num_mel_bins          = gf_get_u32(gf, "yue2tok.mert.num_mel_bins");
    c->num_freq_bins         = gf_get_u32(gf, "yue2tok.mert.num_freq_bins");
    c->frame_rate            = gf_get_f32(gf, "yue2tok.mert.frame_rate");
    c->samples_per_frame     = gf_get_u32(gf, "yue2tok.mert.samples_per_frame");
    c->minimum_input_samples = gf_get_u32(gf, "yue2tok.mert.minimum_input_samples");
    c->spectrogram_power     = gf_get_f32(gf, "yue2tok.mert.spectrogram_power");
    c->db_amin               = gf_get_f32(gf, "yue2tok.mert.db_amin");
    c->db_multiplier         = gf_get_f32(gf, "yue2tok.mert.db_multiplier");
    c->db_top_db_clamp       = gf_get_bool(gf, "yue2tok.mert.db_top_db_clamp");
    c->drop_last_mel_frame   = gf_get_bool(gf, "yue2tok.mert.drop_last_mel_frame");
    c->mel_std_floor         = gf_get_f32(gf, "yue2tok.mert.mel_std_floor");

    c->sub_block_count      = gf_get_u32(gf, "yue2tok.mert.subsampling.block_count");
    c->sub_in_channels      = gf_get_u32(gf, "yue2tok.mert.subsampling.input_channels");
    c->sub_channels         = yue2_get_i32_arr(gf, "yue2tok.mert.subsampling.channels");
    c->sub_depths           = yue2_get_i32_arr(gf, "yue2tok.mert.subsampling.depths");
    c->sub_strides          = yue2_get_i32_arr(gf, "yue2tok.mert.subsampling.strides");
    c->sub_resample_kernel  = gf_get_u32(gf, "yue2tok.mert.subsampling.resample_kernel");
    c->sub_resample_padding = gf_get_bool(gf, "yue2tok.mert.subsampling.resample_padding");
    c->sub_convnext_kernel  = gf_get_u32(gf, "yue2tok.mert.subsampling.convnext_kernel");
    c->sub_convnext_padding = gf_get_u32(gf, "yue2tok.mert.subsampling.convnext_padding");
    c->sub_ln_eps           = gf_get_f32(gf, "yue2tok.mert.subsampling.layer_norm_epsilon");
    c->sub_grn_eps          = gf_get_f32(gf, "yue2tok.mert.subsampling.grn_epsilon");

    c->embedding_length     = gf_get_u32(gf, "yue2tok.mert.embedding_length");
    c->feed_forward_length  = gf_get_u32(gf, "yue2tok.mert.feed_forward_length");
    c->block_count          = gf_get_u32(gf, "yue2tok.mert.block_count");
    c->block_count_upstream = gf_get_u32(gf, "yue2tok.mert.block_count_upstream");
    c->hidden_state_index   = gf_get_u32(gf, "yue2tok.mert.hidden_state_index");
    c->head_count           = gf_get_u32(gf, "yue2tok.mert.attention.head_count");
    c->key_length           = gf_get_u32(gf, "yue2tok.mert.attention.key_length");
    c->ln_eps               = gf_get_f32(gf, "yue2tok.mert.attention.layer_norm_epsilon");
    c->conv_dw_kernel       = gf_get_u32(gf, "yue2tok.mert.conv_depthwise_kernel_size");
    c->conv_dw_padding      = gf_get_u32(gf, "yue2tok.mert.conv_depthwise_padding");

    c->rope_freq_base        = gf_get_f32(gf, "yue2tok.mert.rope.freq_base");
    c->rope_dim              = gf_get_u32(gf, "yue2tok.mert.rope.dimension_count");
    c->rope_inv_freq_in_file = gf_get_bool(gf, "yue2tok.mert.rope.inv_freq_in_file");
}

// Structural sanity, not taste. Every check below corresponds to something the
// pin fixed at runtime; a file that fails one of these would produce features
// that look plausible and gate at chance.
static void yue2_mert_validate_config(const Yue2MertConfig & c, std::vector<std::string> * errs) {
    auto bad = [&](const std::string & m) {
        if (errs->size() < 24) {
            errs->push_back(m);
        }
    };
    if (c.sample_rate != 24000) {
        bad("yue2tok.mert.sample_rate is not 24000");
    }
    if (c.n_fft == 0 || (c.n_fft & (c.n_fft - 1)) != 0) {
        bad("yue2tok.mert.n_fft is not a power of two (the FFT here is radix-2 only)");
    }
    if (c.win_length != c.n_fft) {
        // torchaudio would centre a shorter window inside the n_fft frame. MERT
        // never does, and this file does not implement that path.
        bad("yue2tok.mert.win_length != n_fft — a shorter window would need centring, unimplemented");
    }
    if (c.hop_length == 0 || c.num_mel_bins == 0) {
        bad("yue2tok.mert.hop_length / num_mel_bins missing");
    }
    if (c.num_freq_bins != c.n_fft / 2 + 1) {
        bad("yue2tok.mert.num_freq_bins != n_fft/2 + 1 (onesided)");
    }
    if (c.spectrogram_power != 2.0f) {
        bad("yue2tok.mert.spectrogram_power != 2.0 — this frontend is a POWER spectrogram");
    }
    if (c.db_top_db_clamp) {
        bad("yue2tok.mert.db_top_db_clamp is true — the pin (§3.3) says top_db=None, no clamp");
    }
    if (!c.drop_last_mel_frame) {
        bad("yue2tok.mert.drop_last_mel_frame is false — the pin (§3.3) drops mel[..., :-1]");
    }
    if (c.sub_resample_padding) {
        bad("yue2tok.mert.subsampling.resample_padding is true — the pin (§3.4) says NO padding");
    }
    if (c.sub_resample_kernel != 2) {
        bad("yue2tok.mert.subsampling.resample_kernel != 2");
    }
    if (c.sub_channels.size() != c.sub_block_count || c.sub_depths.size() != c.sub_block_count ||
        c.sub_strides.size() != c.sub_block_count) {
        bad("subsampling channels/depths/strides arrays disagree with block_count");
    }
    if (!c.sub_channels.empty() && (uint32_t) c.sub_channels.back() != c.embedding_length) {
        bad("the last subsampling channel count != embedding_length (the stack must land at the Conformer width)");
    }
    if (c.sub_in_channels != c.num_mel_bins) {
        bad("subsampling.input_channels != num_mel_bins");
    }
    if (c.sub_ln_eps == c.ln_eps) {
        // Trap 6: they are 1e-6 and 1e-5. Equal means the converter (or a hand
        // edit) collapsed them, and the subsampling stack would run at the
        // Conformer epsilon.
        bad("subsampling layer_norm_epsilon == the Conformer layer_norm_epsilon — the pin (§3.4/§3.5) says 1e-6 vs 1e-5");
    }
    if (c.block_count != c.hidden_state_index + 1) {
        bad("block_count != hidden_state_index + 1 — hidden_states[k] is the OUTPUT of layers[k] (pin §3.1)");
    }
    if (c.head_count == 0 || c.key_length == 0 || c.head_count * c.key_length != c.embedding_length) {
        bad("attention head_count * key_length != embedding_length");
    }
    if (c.rope_dim != c.key_length) {
        bad("rope.dimension_count != attention.key_length (MERT rotates the full head)");
    }
    if (c.rope_inv_freq_in_file) {
        bad("rope.inv_freq_in_file is true — inv_freq is persistent=False upstream and MUST be built (pin §3.2)");
    }
}

// ── Loader ──────────────────────────────────────────────────────────────────

static void yue2_mert_free(Yue2MertModel * m) {
    wctx_free(&m->wctx);
    m->w  = Yue2MertWeights{};
    m->mel = Yue2MertMelPlan{};
    m->tmap.clear();
    m->vram          = 0;
    m->blocks_loaded = false;
    if (m->backend_ref) {
        backend_release(m->backend, m->cpu_backend);
        m->backend     = nullptr;
        m->cpu_backend = nullptr;
        m->backend_ref = false;
    }
}

// Pull an F32 tensor back out of the (possibly GPU) weight buffer.
static bool yue2_mert_readback(const ggml_tensor * t, std::vector<float> * out, const char * what,
                               std::string * err) {
    if (!t) {
        if (err) {
            *err = std::string("MERT tensor missing: ") + what;
        }
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        if (err) {
            *err = std::string("MERT tensor '") + what + "' is not F32 (type " + std::to_string((int) t->type) +
                   "); convert-yue2-tok.py forces every frontend buffer to F32";
        }
        return false;
    }
    out->resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get((ggml_tensor *) t, out->data(), 0, ggml_nbytes(t));
    return true;
}

// Build the host-side mel plan: the shipped window verbatim, the filterbank
// TRANSPOSED to [mel][freq] with its non-zero span per mel bin, and 1/std
// precomputed with the clamp_min applied.
//
// Transposing is a cache decision, not a correctness one. The GGUF keeps the
// checkpoint's own [n_freq, n_mel] orientation (convert-yue2-tok.py:769-772
// explains why: the reference does `spectrum.transpose(-1,-2) @ fb`), which
// makes the inner loop over frequency stride by 128 floats.
//
// Skipping the zero span is exact, not an approximation: the bank is 98.5%
// zeros, the power spectrum is non-negative, and `x + 0.0f == x` for every
// x != -0.0f, so the retained terms are summed in the same order with the same
// result. It turns 131k multiply-adds per frame into about 8k.
static bool yue2_mert_build_mel_plan(Yue2MertModel * m, std::string * err) {
    const Yue2MertConfig & c  = m->cfg;
    const int64_t          NF = (int64_t) c.num_freq_bins;
    const int64_t          NM = (int64_t) c.num_mel_bins;

    std::vector<float> fb, mean, sd;
    if (!yue2_mert_readback(m->w.mel_window, &m->mel.window, "mert.mel.stft_window", err) ||
        !yue2_mert_readback(m->w.mel_fb, &fb, "mert.mel.filterbank", err) ||
        !yue2_mert_readback(m->w.mel_mean, &mean, "mert.mel.mean", err) ||
        !yue2_mert_readback(m->w.mel_std, &sd, "mert.mel.std", err)) {
        return false;
    }

    m->mel.fb_mt.assign((size_t) (NM * NF), 0.0f);
    m->mel.fb_lo.assign((size_t) NM, 0);
    m->mel.fb_hi.assign((size_t) NM, 0);
    for (int64_t mi = 0; mi < NM; mi++) {
        int64_t lo = NF, hi = 0;
        for (int64_t f = 0; f < NF; f++) {
            const float v = fb[(size_t) (f * NM + mi)];  // GGUF ne = (n_mel, n_freq)
            m->mel.fb_mt[(size_t) (mi * NF + f)] = v;
            if (v != 0.0f) {
                if (f < lo) {
                    lo = f;
                }
                hi = f + 1;
            }
        }
        if (lo > hi) {
            lo = 0;
            hi = 0;
        }
        m->mel.fb_lo[(size_t) mi] = (int32_t) lo;
        m->mel.fb_hi[(size_t) mi] = (int32_t) hi;
    }

    m->mel.mean = mean;
    m->mel.inv_std.resize((size_t) NM);
    for (int64_t mi = 0; mi < NM; mi++) {
        // clamp_min on the STD, then reciprocal. NOT (std + eps) — pin §3.3.
        m->mel.inv_std[(size_t) mi] = 1.0f / std::max(sd[(size_t) mi], c.mel_std_floor);
    }
    return true;
}

// Load mert.* out of a yue2-tok GGUF. `want_blocks` also pulls the 21
// Conformer blocks (about 2.1 GiB); the frontend and subsampling stack are
// about 46 MiB on their own, which is all the front-half probe needs.
static bool yue2_mert_load(Yue2MertModel * m, const std::string & path, bool want_blocks, std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    yue2_mert_free(m);
    m->path = path;

    GGUFModel gf;
    if (!gf_load(&gf, path.c_str())) {
        if (err) {
            *err = "cannot open " + path;
        }
        return false;
    }
    m->arch    = gf_get_str(gf, "general.architecture");
    m->license = gf_get_str(gf, "yue2tok.license_attribution");
    if (m->arch != "yue2-tok") {
        if (err) {
            *err = "architecture is '" + m->arch + "', expected 'yue2-tok' (" + path + ")";
        }
        gf_close(&gf);
        return false;
    }

    yue2_mert_parse_config(gf, &m->cfg);
    std::vector<std::string> errs;
    yue2_mert_validate_config(m->cfg, &errs);
    if (!errs.empty()) {
        if (err) {
            *err = "yue2-tok config rejected: " + errs[0] +
                   (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        return false;
    }

    const Yue2MertConfig & c  = m->cfg;
    const int64_t          NB = (int64_t) c.sub_block_count;
    const int64_t          NL = (int64_t) c.block_count;

    BackendPair bp = backend_init("YuE2-MERT");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->backend_ref = true;

    // 4 frontend + per sub-block (4 resample + 10 per ConvNext layer)
    // + per Conformer block 31.
    int budget = 4;
    for (int64_t b = 0; b < NB; b++) {
        budget += 4 + 10 * c.sub_depths[(size_t) b];
    }
    if (want_blocks) {
        budget += (int) (31 * NL);
    }
    wctx_init(&m->wctx, budget + 16);
    Yue2Loader ld{ &m->wctx, &gf, &m->tmap, &errs };

    const int64_t NM = (int64_t) c.num_mel_bins;
    const int64_t NF = (int64_t) c.num_freq_bins;
    m->w.mel_window  = ld.req("mert.mel.stft_window", (int64_t) c.n_fft);
    m->w.mel_fb      = ld.req("mert.mel.filterbank", NM, NF);  // ne = (mel, freq)
    m->w.mel_mean    = ld.req("mert.mel.mean", NM);
    m->w.mel_std     = ld.req("mert.mel.std", NM);

    // channels = [num_mel_bins] + subsampling_channels, so block i maps
    // chans[i] -> chans[i+1]. Block 0 is 128->128 at stride 1: identity
    // resampling, and NO down_* tensors exist for it.
    std::vector<int64_t> chans;
    chans.push_back((int64_t) c.sub_in_channels);
    for (int64_t b = 0; b < NB; b++) {
        chans.push_back((int64_t) c.sub_channels[(size_t) b]);
    }
    m->w.sub.assign((size_t) NB, Yue2MertSubBlock{});
    for (int64_t b = 0; b < NB && errs.empty(); b++) {
        const int64_t cin  = chans[(size_t) b];
        const int64_t cout = chans[(size_t) (b + 1)];
        const int64_t s    = (int64_t) c.sub_strides[(size_t) b];
        const bool    identity = (cin == cout && s == 1);
        Yue2MertSubBlock & sb  = m->w.sub[(size_t) b];
        char               nm[96];

        if (!identity) {
            snprintf(nm, sizeof(nm), "mert.sub.%d.down_norm.weight", (int) b);
            sb.down_norm_w = ld.req(nm, cin);
            snprintf(nm, sizeof(nm), "mert.sub.%d.down_norm.bias", (int) b);
            sb.down_norm_b = ld.req(nm, cin);
            snprintf(nm, sizeof(nm), "mert.sub.%d.down_conv.weight", (int) b);
            sb.down_conv_w = ld.req(nm, (int64_t) c.sub_resample_kernel, cin, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.down_conv.bias", (int) b);
            sb.down_conv_b = ld.req(nm, cout);
        }

        const int64_t depth = (int64_t) c.sub_depths[(size_t) b];
        sb.cnx.assign((size_t) depth, Yue2MertCnxLayer{});
        for (int64_t l = 0; l < depth && errs.empty(); l++) {
            Yue2MertCnxLayer & cl = sb.cnx[(size_t) l];
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.dw.weight", (int) b, (int) l);
            cl.dw_w = ld.req(nm, (int64_t) c.sub_convnext_kernel, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.dw.bias", (int) b, (int) l);
            cl.dw_b = ld.req(nm, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.norm.weight", (int) b, (int) l);
            cl.norm_w = ld.req(nm, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.norm.bias", (int) b, (int) l);
            cl.norm_b = ld.req(nm, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.pw_up.weight", (int) b, (int) l);
            cl.pw_up_w = ld.req(nm, cout, 4 * cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.pw_up.bias", (int) b, (int) l);
            cl.pw_up_b = ld.req(nm, 4 * cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.grn.weight", (int) b, (int) l);
            cl.grn_w = ld.req(nm, 4 * cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.grn.bias", (int) b, (int) l);
            cl.grn_b = ld.req(nm, 4 * cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.pw_down.weight", (int) b, (int) l);
            cl.pw_down_w = ld.req(nm, 4 * cout, cout);
            snprintf(nm, sizeof(nm), "mert.sub.%d.cnx.%d.pw_down.bias", (int) b, (int) l);
            cl.pw_down_b = ld.req(nm, cout);
        }
    }

    if (want_blocks && errs.empty()) {
        const int64_t H = (int64_t) c.embedding_length;
        const int64_t F = (int64_t) c.feed_forward_length;
        m->w.blk.assign((size_t) NL, Yue2MertBlock{});
        for (int64_t n = 0; n < NL && errs.empty(); n++) {
            Yue2MertBlock & b = m->w.blk[(size_t) n];
            char            nm[96];
#define YUE2_MERT_T(field, fmtstr, ...)                                     \
    do {                                                                     \
        snprintf(nm, sizeof(nm), fmtstr, (int) n);                           \
        b.field = ld.req(nm, __VA_ARGS__);                                   \
    } while (0)
            YUE2_MERT_T(ffn1_norm_w, "mert.blk.%d.ffn1_norm.weight", H);
            YUE2_MERT_T(ffn1_norm_b, "mert.blk.%d.ffn1_norm.bias", H);
            YUE2_MERT_T(ffn1_up_w, "mert.blk.%d.ffn1_up.weight", H, F);
            YUE2_MERT_T(ffn1_up_b, "mert.blk.%d.ffn1_up.bias", F);
            YUE2_MERT_T(ffn1_down_w, "mert.blk.%d.ffn1_down.weight", F, H);
            YUE2_MERT_T(ffn1_down_b, "mert.blk.%d.ffn1_down.bias", H);
            YUE2_MERT_T(ffn2_norm_w, "mert.blk.%d.ffn2_norm.weight", H);
            YUE2_MERT_T(ffn2_norm_b, "mert.blk.%d.ffn2_norm.bias", H);
            YUE2_MERT_T(ffn2_up_w, "mert.blk.%d.ffn2_up.weight", H, F);
            YUE2_MERT_T(ffn2_up_b, "mert.blk.%d.ffn2_up.bias", F);
            YUE2_MERT_T(ffn2_down_w, "mert.blk.%d.ffn2_down.weight", F, H);
            YUE2_MERT_T(ffn2_down_b, "mert.blk.%d.ffn2_down.bias", H);
            YUE2_MERT_T(attn_norm_w, "mert.blk.%d.attn_norm.weight", H);
            YUE2_MERT_T(attn_norm_b, "mert.blk.%d.attn_norm.bias", H);
            YUE2_MERT_T(attn_q_w, "mert.blk.%d.attn_q.weight", H, H);
            YUE2_MERT_T(attn_q_b, "mert.blk.%d.attn_q.bias", H);
            YUE2_MERT_T(attn_k_w, "mert.blk.%d.attn_k.weight", H, H);
            YUE2_MERT_T(attn_k_b, "mert.blk.%d.attn_k.bias", H);
            YUE2_MERT_T(attn_v_w, "mert.blk.%d.attn_v.weight", H, H);
            YUE2_MERT_T(attn_v_b, "mert.blk.%d.attn_v.bias", H);
            YUE2_MERT_T(attn_o_w, "mert.blk.%d.attn_output.weight", H, H);
            YUE2_MERT_T(attn_o_b, "mert.blk.%d.attn_output.bias", H);
            YUE2_MERT_T(conv_norm_w, "mert.blk.%d.conv_norm.weight", H);
            YUE2_MERT_T(conv_norm_b, "mert.blk.%d.conv_norm.bias", H);
            YUE2_MERT_T(conv_pw1_w, "mert.blk.%d.conv_pw1.weight", H, 2 * H);
            YUE2_MERT_T(conv_dw_w, "mert.blk.%d.conv_dw.weight", (int64_t) c.conv_dw_kernel, H);
            YUE2_MERT_T(conv_dw_norm_w, "mert.blk.%d.conv_dw_norm.weight", H);
            YUE2_MERT_T(conv_dw_norm_b, "mert.blk.%d.conv_dw_norm.bias", H);
            YUE2_MERT_T(conv_pw2_w, "mert.blk.%d.conv_pw2.weight", H, H);
            YUE2_MERT_T(final_norm_w, "mert.blk.%d.final_norm.weight", H);
            YUE2_MERT_T(final_norm_b, "mert.blk.%d.final_norm.bias", H);
#undef YUE2_MERT_T
        }
        m->blocks_loaded = errs.empty();
    }

    if (!errs.empty()) {
        if (err) {
            *err = errs[0] + (errs.size() > 1 ? " (+" + std::to_string(errs.size() - 1) + " more)" : "");
        }
        gf_close(&gf);
        yue2_mert_free(m);
        return false;
    }

    if (!wctx_alloc(&m->wctx, m->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the MERT weights";
        }
        gf_close(&gf);
        yue2_mert_free(m);
        return false;
    }
    gf_close(&gf);  // safe: wctx_alloc has copied everything to the backend

    m->vram = m->wctx.buffer ? ggml_backend_buffer_get_size(m->wctx.buffer) : 0;
    if (!yue2_mert_build_mel_plan(m, err)) {
        yue2_mert_free(m);
        return false;
    }
    m->load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[YuE2-MERT] Loaded %s: %zu tensors, %.1f MB, blocks=%s, %.0f ms\n", yue2_basename(path).c_str(),
            m->tmap.size(), (double) m->vram / (1024.0 * 1024.0), m->blocks_loaded ? "yes" : "no", m->load_ms);
    return true;
}

// True when every ConvNext matmul weight is stored F32. In yue2-tok-f16.gguf
// the two pointwise weights per layer are F16 — pure storage rounding, about
// 2.07e-4 rel-L2 on the weights themselves — so the subsampled-output bar has
// to be relaxed for that file. Callers print which regime they are in rather
// than quietly picking a tolerance.
static bool yue2_mert_weights_are_f32(const Yue2MertModel & m) {
    for (const auto & sb : m.w.sub) {
        for (const auto & cl : sb.cnx) {
            if (!cl.pw_up_w || !cl.pw_down_w) {
                return false;
            }
            if (cl.pw_up_w->type != GGML_TYPE_F32 || cl.pw_down_w->type != GGML_TYPE_F32) {
                return false;
            }
        }
    }
    return true;
}

// ── Mel frontend (CPU, fp32 semantics, double arithmetic) ───────────────────

namespace yue2_mert_detail {

// Iterative radix-2 Cooley-Tukey. n_fft is 2048 — a pure power of two — so
// there is no odd-size base case to worry about (contrast moss-mel.h, which
// needs a naive-DFT fallback because Whisper's n_fft is 400 = 2^4 * 25).
struct Fft {
    int                 n = 0;
    std::vector<int>    rev;
    std::vector<double> tw_r, tw_i;  // exp(-2*pi*i*j/n), j in [0, n/2)

    void init(int n_) {
        if (n == n_) {
            return;
        }
        n = n_;
        rev.assign((size_t) n, 0);
        int bits = 0;
        while ((1 << bits) < n) {
            bits++;
        }
        for (int i = 0; i < n; i++) {
            int r = 0;
            for (int b = 0; b < bits; b++) {
                if (i & (1 << b)) {
                    r |= 1 << (bits - 1 - b);
                }
            }
            rev[(size_t) i] = r;
        }
        const double pi = 3.14159265358979323846;
        tw_r.assign((size_t) (n / 2), 0.0);
        tw_i.assign((size_t) (n / 2), 0.0);
        for (int j = 0; j < n / 2; j++) {
            const double a  = -2.0 * pi * (double) j / (double) n;
            tw_r[(size_t) j] = std::cos(a);
            tw_i[(size_t) j] = std::sin(a);
        }
    }

    // In-place. re/im are n long; on return they hold the full spectrum.
    void run(double * re, double * im) const {
        for (int i = 0; i < n; i++) {
            const int r = rev[(size_t) i];
            if (i < r) {
                std::swap(re[i], re[r]);
                std::swap(im[i], im[r]);
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; j++) {
                    const double wr = tw_r[(size_t) (j * step)];
                    const double wi = tw_i[(size_t) (j * step)];
                    const int    a  = i + j;
                    const int    b  = a + half;
                    const double xr = re[b] * wr - im[b] * wi;
                    const double xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }
};

// torch's F.pad(mode="reflect") index map: period 2S-2, no repetition of the
// edge sample. Valid for |i| < S, which is guaranteed because the frontend
// refuses inputs shorter than minimum_input_samples = n_fft/2 + 1.
static inline int64_t reflect_idx(int64_t i, int64_t S) {
    if (S <= 1) {
        return 0;
    }
    const int64_t period = 2 * S - 2;
    int64_t       j      = i % period;
    if (j < 0) {
        j += period;
    }
    return j < S ? j : period - j;
}

}  // namespace yue2_mert_detail

// How many mel frames a clip of `n_samples` produces, AFTER the dropped last
// frame. center=True gives n/hop + 1; the drop leaves exactly n/hop.
static inline int64_t yue2_mert_mel_frames(const Yue2MertConfig & c, int64_t n_samples) {
    if (n_samples < (int64_t) c.minimum_input_samples) {
        return 0;
    }
    return n_samples / (int64_t) c.hop_length;
}

// The full frontend (pin §3.3), on one chunk:
//
//   spectrum = |stft(wav, n_fft=2048, hop=240, window=<shipped>,
//                    center=True, pad_mode="reflect", onesided=True)|^2
//   mel      = spectrum^T @ fb                       # [F, 128]
//   mel      = 10 * log10(max(mel, 1e-10))           # top_db=None: NO clamp
//   mel      = mel[:-1]                              # DROP the last frame
//   mel      = (mel - mel_mean) / max(mel_std, 1e-5)
//
// `out` is resized to T_mel * num_mel_bins, row-major [T_mel, 128] — the exact
// on-disk layout of the 01_mel.f32 fixture, and also the ne = [128, T] ggml
// layout the subsampling graph takes as input. No transpose anywhere.
//
// Arithmetic runs in double and rounds once at the end. The reference is fp32
// (autocast explicitly disabled, modeling_mert2.py:65), so this is strictly
// more accurate than the target, not differently accurate.
static bool yue2_mert_mel(const Yue2MertModel & m, const float * pcm, int64_t n_samples, std::vector<float> * out,
                          int64_t * out_frames, std::string * err) {
    const Yue2MertConfig & c = m.cfg;
    if (n_samples < (int64_t) c.minimum_input_samples) {
        if (err) {
            *err = "clip is " + std::to_string(n_samples) + " samples; MERT needs at least " +
                   std::to_string(c.minimum_input_samples) + " (reflect padding is n_fft/2 wide)";
        }
        return false;
    }
    const int64_t NFFT = (int64_t) c.n_fft;
    const int64_t HOP  = (int64_t) c.hop_length;
    const int64_t NF   = (int64_t) c.num_freq_bins;
    const int64_t NM   = (int64_t) c.num_mel_bins;
    const int64_t PAD  = NFFT / 2;

    const int64_t T_all  = n_samples / HOP + 1;  // center=True frame count
    const int64_t T_keep = T_all - 1;            // trap 3: drop the last frame
    if (T_keep <= 0) {
        if (err) {
            *err = "clip too short to produce a mel frame after the dropped last frame";
        }
        return false;
    }

    yue2_mert_detail::Fft fft;
    fft.init((int) NFFT);

    std::vector<double> re((size_t) NFFT), im((size_t) NFFT), pw((size_t) NF);
    out->assign((size_t) (T_keep * NM), 0.0f);

    const float * win   = m.mel.window.data();
    const float * fb    = m.mel.fb_mt.data();
    const int32_t * lo  = m.mel.fb_lo.data();
    const int32_t * hi  = m.mel.fb_hi.data();
    const float * mean  = m.mel.mean.data();
    const float * istd  = m.mel.inv_std.data();
    const double  amin  = (double) c.db_amin;
    const double  mult  = (double) c.db_multiplier;

    for (int64_t t = 0; t < T_keep; t++) {
        // Frame t is centred on sample t*hop: it covers padded indices
        // [t*hop, t*hop + n_fft), i.e. original indices t*hop - n_fft/2 + j.
        const int64_t base = t * HOP - PAD;
        for (int64_t j = 0; j < NFFT; j++) {
            const int64_t src = base + j;
            const int64_t idx = (src >= 0 && src < n_samples)
                                    ? src
                                    : yue2_mert_detail::reflect_idx(src, n_samples);
            re[(size_t) j] = (double) pcm[idx] * (double) win[j];
            im[(size_t) j] = 0.0;
        }
        fft.run(re.data(), im.data());
        for (int64_t f = 0; f < NF; f++) {
            const double r = re[(size_t) f];
            const double i = im[(size_t) f];
            pw[(size_t) f] = r * r + i * i;  // power=2.0
        }
        float * dst = out->data() + (size_t) (t * NM);
        for (int64_t mi = 0; mi < NM; mi++) {
            const float * row = fb + (size_t) (mi * NF);
            double        acc = 0.0;
            for (int32_t f = lo[mi]; f < hi[mi]; f++) {
                acc += pw[(size_t) f] * (double) row[f];
            }
            const double db = mult * std::log10(std::max(acc, amin));
            dst[mi]         = (float) ((db - (double) mean[mi]) * (double) istd[mi]);
        }
    }
    if (out_frames) {
        *out_frames = T_keep;
    }
    return true;
}

// ── RoPE tables (pin §3.6) ──────────────────────────────────────────────────
//
//   inv_freq[i] = 1 / base^(2i/head_dim)     for i in 0..head_dim/2-1
//   angles[t]   = cat(t*inv_freq, t*inv_freq)   -- DUPLICATED, not interleaved
//   cos/sin     = angles.cos(), angles.sin()
//
// Half-split (GPT-NeoX) convention: element d pairs with d + head_dim/2, never
// with d+1. `inv_freq` is persistent=False upstream, so it is NOT in the
// checkpoint and has to be built here — and it is built in FP32 there
// (modeling_mert2.py:130-134 constructs it under torch.device("cpu") in
// float32, and _apply() restores float32 after any .half()), so this function
// deliberately does its arithmetic in float rather than double. At t up to a
// few thousand the angle is large enough that a double-precision inv_freq,
// rounded differently in the last bit, would move cos() by ~1e-5.
//
// POSITIONS RESTART AT 0 IN EVERY CHUNK. There is no absolute song position
// anywhere in MERT; `positions = self.embed_positions(hidden)` is computed
// once per forward from the post-subsampling length.
//
// `cos`/`sin` are resized to T * rope_dim, row-major [T, rope_dim] — the
// 03_rope_{cos,sin}.f32 fixture layout.
static void yue2_mert_rope_tables(const Yue2MertConfig & c, int64_t T, std::vector<float> * cos_out,
                                  std::vector<float> * sin_out) {
    const int64_t D    = (int64_t) c.rope_dim;
    const int64_t half = D / 2;
    std::vector<float> inv((size_t) half);
    for (int64_t i = 0; i < half; i++) {
        const float e   = (float) (2 * i) / (float) D;
        inv[(size_t) i] = 1.0f / std::pow((float) c.rope_freq_base, e);
    }
    cos_out->assign((size_t) (T * D), 0.0f);
    sin_out->assign((size_t) (T * D), 0.0f);
    for (int64_t t = 0; t < T; t++) {
        float * cr = cos_out->data() + (size_t) (t * D);
        float * sr = sin_out->data() + (size_t) (t * D);
        for (int64_t i = 0; i < half; i++) {
            const float a = (float) t * inv[(size_t) i];
            const float cv = std::cos(a);
            const float sv = std::sin(a);
            cr[i] = cv;
            cr[i + half] = cv;   // cat((freqs, freqs), dim=-1)
            sr[i] = sv;
            sr[i + half] = sv;
        }
    }
}

// ── Chunking (pin §3.7) ─────────────────────────────────────────────────────

struct Yue2MertChunk {
    int64_t start = 0;    // sample offset into the mono 24 kHz track
    int64_t length = 0;   // samples
    bool    is_tail = false;  // shorter than a full chunk: MUST run alone
};

// The reference splits at exactly 30 s with no overlap and no crossfade,
// DROPS a final chunk shorter than one second, and runs every short tail as
// its own forward pass. That last rule is not an optimisation detail: GRN
// normalises over the time axis of whatever tensor it is given (trap 5), so
// padding a tail up to 30 s to batch it changes the features for the whole
// tail. RoPE also restarts at 0 in each chunk, which is what makes splitting
// legitimate at all.
static std::vector<Yue2MertChunk> yue2_mert_chunk_plan(const Yue2MertConfig & c, int64_t n_samples) {
    const int64_t CH  = (int64_t) c.sample_rate * 30;
    const int64_t MIN = (int64_t) c.sample_rate;  // a tail under 1 s is DROPPED
    std::vector<Yue2MertChunk> out;
    for (int64_t s = 0; s < n_samples; s += CH) {
        const int64_t len = std::min(CH, n_samples - s);
        if (len < MIN) {
            break;
        }
        out.push_back({ s, len, len < CH });
    }
    return out;
}

// ── ConvNext subsampling stack (ggml graph) ─────────────────────────────────

// Output frame count of the whole stack for a given mel length. Trap 4: the
// resampling conv is kernel 2 with NO padding, so this is
// floor((L-2)/stride)+1 per block, which is L/2 only when L is even.
static int64_t yue2_mert_sub_out_len(const Yue2MertConfig & c, int64_t T_mel) {
    int64_t L = T_mel;
    for (size_t b = 0; b < c.sub_strides.size(); b++) {
        const int64_t s = (int64_t) c.sub_strides[b];
        const int64_t k = (int64_t) c.sub_resample_kernel;
        const int64_t cin  = (b == 0) ? (int64_t) c.sub_in_channels : (int64_t) c.sub_channels[b - 1];
        const int64_t cout = (int64_t) c.sub_channels[b];
        if (cin == cout && s == 1) {
            continue;  // nn.Identity resampling; the ConvNext layers preserve length
        }
        if (L < k) {
            return 0;
        }
        L = (L - k) / s + 1;
    }
    return L;
}

#define YUE2_MERT_SUB_MAX_NODES 4096

struct Yue2MertSubGraph {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       cpu_backend = nullptr;
    bool                 backend_ref = false;
    ggml_backend_sched_t sched       = nullptr;
    WeightCtx            prep        = {};
    const void *         weights_token = nullptr;  // the model's weight buffer

    ggml_tensor * grn_eps = nullptr;  // [1,1] F32 constant, shared by every GRN

    ggml_context * gctx   = nullptr;
    uint8_t *      gbuf   = nullptr;
    ggml_cgraph *  graph  = nullptr;
    ggml_tensor *  input  = nullptr;   // [128, T_mel]
    ggml_tensor *  output = nullptr;   // [1024, T_out]
    int64_t        graph_T = 0;
};

namespace yue2_mert_detail {

// LayerNorm over ne0 (channels), then affine. w and b are 1-D [C], which
// ggml_mul/ggml_add broadcast across ne1 without a reshape.
static ggml_tensor * ln(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// Conv1d over TIME. w [K, IC, OC], x [T, IC] (ne0 = time!), -> [T_out, OC].
// Explicit F32 im2col, never ggml_conv_1d — same rationale as
// yue2_vae_conv1d (yue2-vae-graph.h:382-402): ggml_conv_1d forces an F16
// im2col and this stage's reference runs in fp32 with autocast off.
static ggml_tensor * conv1d_t(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad,
                              int stride) {
    ggml_tensor * col = ggml_im2col(ctx, w, x, /*s0*/ stride, /*s1*/ 0, pad, 0, /*d0*/ 1, 0, /*is_2D*/ false,
                                    GGML_TYPE_F32);  // [IC*K, OL, 1, 1]
    ggml_tensor * y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                   ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]));  // [OL, OC]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// Depthwise Conv1d(C, C, K, padding=pad, groups=C). w [K, C] from the GGUF,
// x [T, C] (ne0 = time), -> [T, C].
//
// This is ggml_conv_1d_dw's own decomposition (ggml.c:4569-4585) with ONE
// change: the im2col destination type is forced to F32. The stock helper picks
// F16 for an F32 kernel, which would quietly halve the precision of a stage
// whose reference runs in fp32.
static ggml_tensor * conv1d_dw_t(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int pad) {
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    const int64_t K = w->ne[0];
    ggml_tensor * w3 = ggml_reshape_3d(ctx, w, K, 1, C);              // [K, 1, C]
    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, T, 1, C, 1);           // [T, 1, C, 1]
    ggml_tensor * col =
        ggml_im2col(ctx, w3, x4, /*s0*/ 1, /*s1*/ 0, pad, 0, /*d0*/ 1, 0, /*is_2D*/ false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, col, w3);                     // [OL, 1, C]
    y               = ggml_reshape_2d(ctx, y, y->ne[0], C);           // [OL, C]
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// GlobalResponseNorm (ConvNeXt-V2), on x with ne = [C, T]:
//
//   magnitude  = ||x||_2 over TIME (dim=1 in the reference's [B,T,C])   -> [C]
//   normalized = magnitude / (mean_over_channels(magnitude) + 1e-6)
//   out        = weight * (x * normalized) + bias + x
//
// TRAP 5 lives here: the L2 is over the whole chunk's time axis, so GRN
// couples every frame and is sequence-length dependent. The reduction needs
// x transposed, because ggml_sum_rows only sums ne0.
static ggml_tensor * grn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, ggml_tensor * eps1) {
    const int64_t C  = x->ne[0];
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x));            // [T, C]
    ggml_tensor * ss = ggml_sum_rows(ctx, ggml_sqr(ctx, xt));             // [1, C]
    ggml_tensor * g  = ggml_sqrt(ctx, ggml_cont(ctx, ggml_transpose(ctx, ss)));  // [C, 1]
    ggml_tensor * mean = ggml_scale(ctx, ggml_sum_rows(ctx, g), 1.0f / (float) C);  // [1, 1]
    ggml_tensor * n    = ggml_div(ctx, g, ggml_add(ctx, mean, eps1));     // [C, 1], broadcast over T
    ggml_tensor * y    = ggml_mul(ctx, ggml_mul(ctx, x, n), w);
    return ggml_add(ctx, ggml_add(ctx, y, b), x);
}

// One ConvNext layer: out = h + pointwise(depthwise(h)). x is [C, T].
static ggml_tensor * cnx_layer(ggml_context * ctx, const Yue2MertCnxLayer & L, ggml_tensor * x, int dw_pad,
                               float ln_eps, ggml_tensor * eps1) {
    ggml_tensor * skip = x;

    // depthwise_block: Transpose -> Conv1d(k=7,pad=3,groups=C) -> Transpose
    ggml_tensor * h = ggml_cont(ctx, ggml_transpose(ctx, x));       // [T, C]
    h               = conv1d_dw_t(ctx, L.dw_w, L.dw_b, h, dw_pad);  // [T, C]
    h               = ggml_cont(ctx, ggml_transpose(ctx, h));       // [C, T]

    // pointwise_block: LayerNorm -> Linear(C,4C) -> GELU -> GRN -> Linear(4C,C)
    h = ln(ctx, h, L.norm_w, L.norm_b, ln_eps);
    h = ggml_mul_mat(ctx, L.pw_up_w, h);                            // [4C, T]
    h = ggml_add(ctx, h, L.pw_up_b);
    h = ggml_gelu_erf(ctx, h);  // exact erf, NOT the tanh approximation
    h = grn(ctx, h, L.grn_w, L.grn_b, eps1);
    h = ggml_mul_mat(ctx, L.pw_down_w, h);                          // [C, T]
    h = ggml_add(ctx, h, L.pw_down_b);

    return ggml_add(ctx, skip, h);
}

}  // namespace yue2_mert_detail

// The whole subsampling stack. `mel` is [128, T_mel] (ne0 = mel bin),
// returns [1024, T_out]. `grn_eps` is the shared [1,1] F32 constant; it is
// passed in rather than read off a graph struct so the encoder graph
// (yue2_mert_enc_build, below) can reuse this builder verbatim instead of
// growing a second, divergent copy of the stack.
static ggml_tensor * yue2_mert_sub_build(ggml_context * ctx, const Yue2MertModel & m, ggml_tensor * grn_eps,
                                         ggml_tensor * mel) {
    using namespace yue2_mert_detail;
    const Yue2MertConfig & c = m.cfg;
    ggml_tensor *          x = mel;

    for (size_t b = 0; b < m.w.sub.size(); b++) {
        const Yue2MertSubBlock & sb = m.w.sub[b];
        if (sb.down_conv_w) {
            // resampling_layer: LayerNorm(eps 1e-6) -> Transpose ->
            //                   Conv1d(k=2, stride=s, NO padding) -> Transpose
            x = ln(ctx, x, sb.down_norm_w, sb.down_norm_b, c.sub_ln_eps);
            x = ggml_cont(ctx, ggml_transpose(ctx, x));                                  // [T, Cin]
            x = conv1d_t(ctx, sb.down_conv_w, sb.down_conv_b, x, /*pad*/ 0,
                         (int) c.sub_strides[b]);                                         // [T', Cout]
            x = ggml_cont(ctx, ggml_transpose(ctx, x));                                  // [Cout, T']
        }
        for (const auto & cl : sb.cnx) {
            x = cnx_layer(ctx, cl, x, (int) c.sub_convnext_padding, c.sub_ln_eps, grn_eps);
        }
    }
    return x;
}

static void yue2_mert_sub_free_graph(Yue2MertSubGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx    = nullptr;
    g->gbuf    = nullptr;
    g->graph   = nullptr;
    g->input   = nullptr;
    g->output  = nullptr;
    g->graph_T = 0;
}

static void yue2_mert_sub_graph_free(Yue2MertSubGraph * g) {
    yue2_mert_sub_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->grn_eps       = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

// One derived tensor: the GRN epsilon, as a [1,1] F32 so it can be added to
// the [1,1] channel-mean inside the graph. A literal cannot go in a graph
// context built with no_alloc, and hoisting eps out of the denominator would
// change the arithmetic.
static bool yue2_mert_sub_prepare(const Yue2MertModel & m, Yue2MertSubGraph * g, std::string * err) {
    const void * token = m.wctx.buffer;
    if (g->sched && g->weights_token == token) {
        return true;
    }
    yue2_mert_sub_graph_free(g);

    BackendPair bp = backend_init("YuE2-MERT-sub");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    wctx_init(&g->prep, 2);
    g->grn_eps = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(g->grn_eps, "mert.sub.grn_eps");
    auto eps = std::make_unique<float[]>(1);
    eps[0]   = m.cfg.sub_grn_eps;
    g->prep.pending.push_back({ g->grn_eps, eps.get(), sizeof(float), 0 });
    g->prep.staging.push_back(std::move(eps));
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the MERT subsampling constants";
        }
        yue2_mert_sub_graph_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, YUE2_MERT_SUB_MAX_NODES * 2);
    g->weights_token = token;
    return true;
}

static bool yue2_mert_sub_ensure_graph(const Yue2MertModel & m, Yue2MertSubGraph * g, int64_t T_mel,
                                       std::string * err) {
    if (g->gctx && g->graph_T == T_mel) {
        return true;
    }
    yue2_mert_sub_free_graph(g);

    const size_t ctx_bytes = ggml_tensor_overhead() * YUE2_MERT_SUB_MAX_NODES + ggml_graph_overhead_custom(
                                                                                   YUE2_MERT_SUB_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MERT subsampling graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MERT subsampling graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) m.cfg.num_mel_bins, T_mel);
    ggml_set_name(g->input, "mert_mel_in");
    ggml_set_input(g->input);

    g->output = yue2_mert_sub_build(ctx, m, g->grn_eps, g->input);
    ggml_set_name(g->output, "mert_sub_out");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, YUE2_MERT_SUB_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf  = nullptr;
        g->graph = nullptr;
        if (err) {
            *err = "MERT subsampling graph allocation failed (out of VRAM?) for T_mel=" + std::to_string(T_mel);
        }
        return false;
    }

    g->gctx    = ctx;
    g->graph_T = T_mel;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr, "[YuE2-MERT] Subsample graph: T_mel=%lld -> %lld frames, %d nodes, %d splits, compute %.0f MB\n",
            (long long) T_mel, (long long) g->output->ne[1], ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// Run the ConvNext subsampling stack over one chunk's mel.
//
// `mel` is T_mel * num_mel_bins floats, row-major [T_mel, 128] — exactly what
// yue2_mert_mel() writes. `out` is resized to T_out * embedding_length,
// row-major [T_out, 1024], which is also the 02_subsampled.f32 layout.
//
// ONE CHUNK AT A TIME, ALWAYS. Trap 5: GRN reduces over this tensor's whole
// time axis, so batching two chunks (or padding a tail out to full length)
// changes the result for every frame, not just the padding. The caller runs
// yue2_mert_chunk_plan() and calls this once per chunk.
//
// Not thread-safe: the caller serialises (same contract as yue2_vae_decode).
static bool yue2_mert_subsample(const Yue2MertModel & m, Yue2MertSubGraph * g, const float * mel, int64_t T_mel,
                                std::vector<float> * out, int64_t * out_frames, std::string * err) {
    if (T_mel <= 0) {
        if (err) {
            *err = "T_mel must be > 0";
        }
        return false;
    }
    if (yue2_mert_sub_out_len(m.cfg, T_mel) <= 0) {
        if (err) {
            *err = "T_mel=" + std::to_string(T_mel) + " is too short for the subsampling stack";
        }
        return false;
    }
    if (!yue2_mert_sub_prepare(m, g, err)) {
        return false;
    }
    if (!yue2_mert_sub_ensure_graph(m, g, T_mel, err)) {
        return false;
    }
    ggml_backend_tensor_set(g->input, mel, 0, ggml_nbytes(g->input));
    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MERT subsampling graph compute failed";
        }
        return false;
    }
    const int64_t C = g->output->ne[0];
    const int64_t T = g->output->ne[1];
    out->resize((size_t) (C * T));
    ggml_backend_tensor_get(g->output, out->data(), 0, (size_t) (C * T) * sizeof(float));
    if (out_frames) {
        *out_frames = T;
    }
    return true;
}

// ── Conformer blocks (pin §3.5) ─────────────────────────────────────────────
//
// 21 identical pre-norm blocks, run on the subsampling stack's output at
// ne = [1024, T_sub]. The residual structure, verbatim from
// MERT-v2-FullSong/modeling_mert2.py::ConformerBlock.forward:
//
//   h = h + 0.5 * ffn1(ffn1_layer_norm(h))       macaron half-step
//   h = attn(attn_layer_norm(h), (cos, sin)) + h 16 heads, RoPE, NO mask
//   h = conv_module(h) + h
//   h = h + 0.5 * ffn2(ffn2_layer_norm(h))       macaron half-step
//   h = final_layer_norm(h)                      NOT a residual
//
// `hidden_states[k]` is the value returned by block k — i.e. POST
// final_layer_norm — so hidden_states[20] is simply what block 20 returns
// (pin §3.1; the GGUF's hidden_state_index KV says 20 and block_count says 21,
// and yue2_mert_validate_config already refuses a file where those disagree).
//
// ── GELU: DETERMINED, NOT ASSUMED ──────────────────────────────────────────
//
// yue2-tok-head.h documents a genuine departure — the tokenizer HEAD's fixture
// matches the TANH approximation, because nn.TransformerEncoderLayer routes
// through PyTorch's fused encoder kernel, which hardcodes tanh regardless of
// the activation="gelu" string.
//
// MERT is NOT that. Its blocks are hand-written modules, and all three GELU
// sites in MERT-v2-FullSong/modeling_mert2.py are the plain API:
//
//     FeedForward.forward     : self.w_2(F.gelu(self.w_1(x)))      (line 209)
//     ConvolutionModule       : nn.GELU()                          (line 224)
//     ConvNextLayer pointwise : nn.GELU()                          (line 98)
//
// F.gelu and nn.GELU() both default to approximate='none', the EXACT erf form,
// and there is no fused path here that could override them.
//
// That is still an argument from source, so the flavour is a runtime switch
// (Yue2MertEncodeOptions::gelu_tanh) and yue2-probe's --mert-block-parity runs
// BOTH and prints both columns. Measured against the TRUE fp32 block fixtures
// (real-30s, F32 GGUF, TF32 off), rel-L2:
//
//     block00    erf 1.718e-06   tanh 3.939e-04     229x
//     block10    erf 2.259e-06   tanh 3.931e-03    1740x
//     block20    erf 3.261e-06   tanh 2.801e-03     859x
//
// Erf wins at every depth by two to three orders of magnitude, and the tanh
// error GROWS with depth while the erf error stays flat — the signature of a
// systematic per-block bias rather than rounding. Source and fixtures agree
// here. The head remains the one place in this port where they do not.
//
// ── RoPE ────────────────────────────────────────────────────────────────────
//
// ggml_rope_ext with GGML_ROPE_TYPE_NEOX is exactly MERT's convention: NEOX
// rotates element i against element i + n_dims/2 (ggml.h's own "[ccccssss]"
// diagram), which is rotate_half — cat((-second, first)) — and its theta is
// pos * freq_base^(-2i/n_dims), which is inv_freq. The standalone
// yue2_mert_rope_tables() above stays as the gated reference for the tables
// themselves (03_rope_{cos,sin}.f32, 2.4e-8); the graph does not feed it in,
// it lets the rope op rebuild the same angles on the device.
//
// POSITIONS RESTART AT 0 IN EVERY CHUNK. The positions tensor is filled with
// 0..T_sub-1 per chunk and never carries a song offset — pin §3.6, and the
// reason 30 s chunking is legitimate at all.

#define YUE2_MERT_ENC_MAX_NODES 16384

namespace yue2_mert_detail {

// MERT's own GELU. Default (tanh_approx=false) is the exact erf form, which is
// what F.gelu/nn.GELU do with approximate='none' — see the section above.
static ggml_tensor * mert_gelu(ggml_context * ctx, ggml_tensor * x, bool tanh_approx) {
    return tanh_approx ? ggml_gelu(ctx, x) : ggml_gelu_erf(ctx, x);
}

// FeedForward: w_2(gelu(w_1(x))), 1024 -> 4096 -> 1024, both with bias. The
// 0.5 macaron factor is applied by the caller, on the residual branch.
static ggml_tensor * mert_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * up_w, ggml_tensor * up_b,
                              ggml_tensor * dn_w, ggml_tensor * dn_b, bool tanh_gelu) {
    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, up_w, x), up_b);
    h               = mert_gelu(ctx, h, tanh_gelu);
    return ggml_add(ctx, ggml_mul_mat(ctx, dn_w, h), dn_b);
}

// SelfAttention. Four SEPARATE projections with bias (pin §3.5) — MERT does
// not fuse qkv, unlike the head. RoPE on q and k only, never v. Then SDPA with
// is_causal=False and no mask of any kind: every frame sees every other frame,
// scale 1/sqrt(head_dim).
//
// `h` is the pre-normed input, ne [D, T]. Returns the out_proj result, ne
// [D, T]; the caller adds the residual.
static ggml_tensor * mert_attn(ggml_context * ctx, const Yue2MertConfig & c, const Yue2MertBlock & w, ggml_tensor * h,
                               ggml_tensor * pos) {
    const int64_t D  = (int64_t) c.embedding_length;
    const int64_t HD = (int64_t) c.key_length;
    const int64_t NH = (int64_t) c.head_count;
    const int64_t T  = h->ne[1];

    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_q_w, h), w.attn_q_b);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_k_w, h), w.attn_k_b);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, w.attn_v_w, h), w.attn_v_b);

    // [D, T] -> [head_dim, head, frame], which is the layout ggml_rope_ext
    // wants (positions index ne2).
    q = ggml_reshape_3d(ctx, q, HD, NH, T);
    k = ggml_reshape_3d(ctx, k, HD, NH, T);
    q = ggml_rope_ext(ctx, q, pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, (int) HD, GGML_ROPE_TYPE_NEOX, 0, c.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, t, head]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v3 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, NH, T), 1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);  // [t_k, t_q, head]
    kq = ggml_soft_max_ext(ctx, kq, /*mask*/ nullptr, 1.0f / std::sqrt((float) HD), /*max_bias*/ 0.0f);

    ggml_tensor * kqv = ggml_mul_mat(ctx, v3, kq);           // [hd, t, head]
    kqv               = ggml_permute(ctx, kqv, 0, 2, 1, 3);  // [hd, head, t]
    ggml_tensor * att = ggml_cont_2d(ctx, kqv, D, T);        // [D, T]
    return ggml_add(ctx, ggml_mul_mat(ctx, w.attn_o_w, att), w.attn_o_b);
}

// ConvolutionModule (pin §3.5). `x` is the block state, ne [D, T]; the
// LayerNorm is INSIDE this function (conv_module(h) = conv_block(layer_norm(h)))
// and the residual is added by the caller.
//
// The reference's Transpose modules exist because torch Conv1d wants [B, C, T].
// In this file's [C, T] layout the two kernel-1 convs are plain matmuls and
// need no transpose at all; only the depthwise k=31 conv does, because
// ggml_im2col reads time along ne0.
//
// GLU SPLIT ORDER — the coin flip the pin calls out explicitly: nn.GLU(dim=1)
// on [B, 2048, T] returns first_half * sigmoid(second_half). The FIRST 1024
// channels are the linear branch, the SECOND 1024 are the gate. Swapping them
// produces plausible-looking, wrong features.
//
// The three convs here have NO bias.
static ggml_tensor * mert_conv_module(ggml_context * ctx, const Yue2MertConfig & c, const Yue2MertBlock & w,
                                      ggml_tensor * x, bool tanh_gelu) {
    const int64_t D = x->ne[0];
    const int64_t T = x->ne[1];

    ggml_tensor * h = ln(ctx, x, w.conv_norm_w, w.conv_norm_b, c.ln_eps);
    h               = ggml_mul_mat(ctx, w.conv_pw1_w, h);  // Conv1d(D, 2D, k=1) == matmul -> [2D, T]

    const size_t  es   = ggml_element_size(h);
    ggml_tensor * lin  = ggml_cont(ctx, ggml_view_2d(ctx, h, D, T, h->nb[1], 0));
    ggml_tensor * gate = ggml_cont(ctx, ggml_view_2d(ctx, h, D, T, h->nb[1], (size_t) D * es));
    h                  = ggml_mul(ctx, lin, ggml_sigmoid(ctx, gate));  // [D, T]

    ggml_tensor * ht = ggml_cont(ctx, ggml_transpose(ctx, h));  // [T, D]
    ht               = conv1d_dw_t(ctx, w.conv_dw_w, /*bias*/ nullptr, ht, (int) c.conv_dw_padding);
    h                = ggml_cont(ctx, ggml_transpose(ctx, ht));  // [D, T]

    h = ln(ctx, h, w.conv_dw_norm_w, w.conv_dw_norm_b, c.ln_eps);
    h = mert_gelu(ctx, h, tanh_gelu);
    return ggml_mul_mat(ctx, w.conv_pw2_w, h);  // Conv1d(D, D, k=1) == matmul -> [D, T]
}

// One Conformer block. `x` is [D, T]; the return value is what
// hidden_states[k] holds for this block.
static ggml_tensor * mert_block(ggml_context * ctx, const Yue2MertConfig & c, const Yue2MertBlock & w, ggml_tensor * x,
                                ggml_tensor * pos, bool tanh_gelu) {
    const float eps = c.ln_eps;  // 1e-5 here, 1e-6 in the subsampling stack — trap 6

    ggml_tensor * f1 = mert_ffn(ctx, ln(ctx, x, w.ffn1_norm_w, w.ffn1_norm_b, eps), w.ffn1_up_w, w.ffn1_up_b,
                                w.ffn1_down_w, w.ffn1_down_b, tanh_gelu);
    x = ggml_add(ctx, x, ggml_scale(ctx, f1, 0.5f));

    x = ggml_add(ctx, mert_attn(ctx, c, w, ln(ctx, x, w.attn_norm_w, w.attn_norm_b, eps), pos), x);

    x = ggml_add(ctx, mert_conv_module(ctx, c, w, x, tanh_gelu), x);

    ggml_tensor * f2 = mert_ffn(ctx, ln(ctx, x, w.ffn2_norm_w, w.ffn2_norm_b, eps), w.ffn2_up_w, w.ffn2_up_b,
                                w.ffn2_down_w, w.ffn2_down_b, tanh_gelu);
    x = ggml_add(ctx, x, ggml_scale(ctx, f2, 0.5f));

    // final_layer_norm: NOT a residual. The block OUTPUT is normed, and that is
    // what lands in hidden_states.
    return ln(ctx, x, w.final_norm_w, w.final_norm_b, eps);
}

}  // namespace yue2_mert_detail

// ── Encoder graph: mel -> subsampling -> 21 blocks, ONE graph per chunk ─────
//
// Everything from the mel spectrogram to hidden_states[20] lives in a single
// cached graph, built once per (T_mel, gelu flavour, tap set) and reused. A
// full-length song presents exactly two shapes — 3000 mel frames for every
// 30 s chunk and one odd tail — so a 4-minute track pays for two graph builds
// and nothing more. Rebuilding per block, or per chunk, would dominate the
// runtime: the build allocates and schedules ~1900 nodes.
struct Yue2MertEncGraph {
    ggml_backend_t       backend       = nullptr;
    ggml_backend_t       cpu_backend   = nullptr;
    bool                 backend_ref   = false;
    ggml_backend_sched_t sched         = nullptr;
    WeightCtx            prep          = {};
    const void *         weights_token = nullptr;

    ggml_tensor * grn_eps = nullptr;  // [1,1] F32, shared by every GRN in the sub stack

    ggml_context * gctx   = nullptr;
    uint8_t *      gbuf   = nullptr;
    ggml_cgraph *  graph  = nullptr;
    ggml_tensor *  input  = nullptr;  // [n_mel, T_mel]  F32
    ggml_tensor *  pos    = nullptr;  // [T_sub]         I32, always 0..T_sub-1
    ggml_tensor *  sub    = nullptr;  // [1024, T_sub]   subsampling output (optional tap)
    ggml_tensor *  output = nullptr;  // [1024, T_sub]   hidden_states[20]

    // Per-block taps, parallel to tap_blocks. Empty in production; the parity
    // mode asks for blocks 0, 1, 10, 19 and 20.
    std::vector<int>           tap_blocks;
    std::vector<ggml_tensor *> taps;

    int64_t          graph_T_mel     = 0;
    bool             graph_gelu_tanh = false;
    bool             graph_want_sub  = false;
    std::vector<int> graph_taps;
};

static void yue2_mert_enc_free_graph(Yue2MertEncGraph * g) {
    if (g->gctx) {
        if (g->sched) {
            ggml_backend_sched_reset(g->sched);
        }
        ggml_free(g->gctx);
        free(g->gbuf);
    }
    g->gctx   = nullptr;
    g->gbuf   = nullptr;
    g->graph  = nullptr;
    g->input  = nullptr;
    g->pos    = nullptr;
    g->sub    = nullptr;
    g->output = nullptr;
    g->taps.clear();
    g->graph_T_mel = 0;
    g->graph_taps.clear();
}

static void yue2_mert_enc_graph_free(Yue2MertEncGraph * g) {
    yue2_mert_enc_free_graph(g);
    if (g->sched) {
        ggml_backend_sched_free(g->sched);
        g->sched = nullptr;
    }
    wctx_free(&g->prep);
    g->grn_eps       = nullptr;
    g->weights_token = nullptr;
    if (g->backend_ref) {
        backend_release(g->backend, g->cpu_backend);
        g->backend     = nullptr;
        g->cpu_backend = nullptr;
        g->backend_ref = false;
    }
}

static bool yue2_mert_enc_prepare(const Yue2MertModel & m, Yue2MertEncGraph * g, std::string * err) {
    const void * token = m.wctx.buffer;
    if (g->sched && g->weights_token == token) {
        return true;
    }
    yue2_mert_enc_graph_free(g);

    BackendPair bp = backend_init("YuE2-MERT-enc");
    g->backend     = bp.backend;
    g->cpu_backend = bp.cpu_backend;
    g->backend_ref = true;

    wctx_init(&g->prep, 2);
    g->grn_eps = ggml_new_tensor_2d(g->prep.ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_name(g->grn_eps, "mert.enc.grn_eps");
    auto eps = std::make_unique<float[]>(1);
    eps[0]   = m.cfg.sub_grn_eps;
    g->prep.pending.push_back({ g->grn_eps, eps.get(), sizeof(float), 0 });
    g->prep.staging.push_back(std::move(eps));
    if (!wctx_alloc(&g->prep, g->backend)) {
        if (err) {
            *err = "backend buffer allocation failed for the MERT encoder constants";
        }
        yue2_mert_enc_graph_free(g);
        return false;
    }

    g->sched         = backend_sched_new(bp, YUE2_MERT_ENC_MAX_NODES * 2);
    g->weights_token = token;
    return true;
}

static bool yue2_mert_enc_ensure_graph(const Yue2MertModel & m, Yue2MertEncGraph * g, int64_t T_mel, bool gelu_tanh,
                                       bool want_sub, std::string * err) {
    if (g->gctx && g->graph_T_mel == T_mel && g->graph_gelu_tanh == gelu_tanh && g->graph_want_sub == want_sub &&
        g->graph_taps == g->tap_blocks) {
        return true;
    }
    yue2_mert_enc_free_graph(g);

    if (!m.blocks_loaded) {
        if (err) {
            *err = "the MERT Conformer blocks are not loaded — call yue2_mert_load(want_blocks=true)";
        }
        return false;
    }
    const int64_t T_sub = yue2_mert_sub_out_len(m.cfg, T_mel);
    if (T_sub <= 0) {
        if (err) {
            *err = "T_mel=" + std::to_string(T_mel) + " is too short for the subsampling stack";
        }
        return false;
    }

    const size_t ctx_bytes = ggml_tensor_overhead() * (YUE2_MERT_ENC_MAX_NODES + 256) +
                             ggml_graph_overhead_custom(YUE2_MERT_ENC_MAX_NODES, false);
    g->gbuf = (uint8_t *) malloc(ctx_bytes);
    if (!g->gbuf) {
        if (err) {
            *err = "out of host memory allocating the MERT encoder graph context";
        }
        return false;
    }
    ggml_init_params ip  = { ctx_bytes, g->gbuf, /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        free(g->gbuf);
        g->gbuf = nullptr;
        if (err) {
            *err = "ggml_init failed for the MERT encoder graph context";
        }
        return false;
    }

    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) m.cfg.num_mel_bins, T_mel);
    ggml_set_name(g->input, "mert_enc_mel");
    ggml_set_input(g->input);

    g->pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_sub);
    ggml_set_name(g->pos, "mert_enc_pos");
    ggml_set_input(g->pos);

    ggml_tensor * x = yue2_mert_sub_build(ctx, m, g->grn_eps, g->input);
    if (want_sub) {
        g->sub = x;
        ggml_set_name(g->sub, "mert_enc_sub");
        ggml_set_output(g->sub);
    }

    g->taps.assign(g->tap_blocks.size(), nullptr);
    for (size_t n = 0; n < m.w.blk.size(); n++) {
        x = yue2_mert_detail::mert_block(ctx, m.cfg, m.w.blk[n], x, g->pos, gelu_tanh);
        for (size_t ti = 0; ti < g->tap_blocks.size(); ti++) {
            if (g->tap_blocks[ti] >= 0 && (size_t) g->tap_blocks[ti] == n) {
                char nm[48];
                snprintf(nm, sizeof(nm), "mert_enc_block%02d", (int) n);
                ggml_set_name(x, nm);
                ggml_set_output(x);
                g->taps[ti] = x;
            }
        }
    }
    g->output = x;  // hidden_states[20]: the last block's own return value
    ggml_set_name(g->output, "mert_enc_hidden");
    ggml_set_output(g->output);

    g->graph = ggml_new_graph_custom(ctx, YUE2_MERT_ENC_MAX_NODES, false);
    ggml_build_forward_expand(g->graph, g->output);
    if (g->sub) {
        ggml_build_forward_expand(g->graph, g->sub);
    }
    for (ggml_tensor * t : g->taps) {
        if (t) {
            ggml_build_forward_expand(g->graph, t);
        }
    }

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, g->graph)) {
        ggml_free(ctx);
        free(g->gbuf);
        g->gbuf   = nullptr;
        g->graph  = nullptr;
        g->output = nullptr;
        if (err) {
            *err = "MERT encoder graph allocation failed (out of VRAM?) for T_mel=" + std::to_string(T_mel);
        }
        return false;
    }

    g->gctx            = ctx;
    g->graph_T_mel     = T_mel;
    g->graph_gelu_tanh = gelu_tanh;
    g->graph_want_sub  = want_sub;
    g->graph_taps      = g->tap_blocks;
    const size_t compute_bytes = ggml_backend_sched_get_buffer_size(g->sched, g->backend);
    fprintf(stderr,
            "[YuE2-MERT] Encoder graph: T_mel=%lld -> %lld frames, %zu blocks, %d nodes, %d splits, compute %.0f MB\n",
            (long long) T_mel, (long long) T_sub, m.w.blk.size(), ggml_graph_n_nodes(g->graph),
            ggml_backend_sched_get_n_splits(g->sched), (double) compute_bytes / (1024.0 * 1024.0));
    return true;
}

// Run one chunk's mel through the subsampling stack and all 21 Conformer
// blocks. `mel` is row-major [T_mel, n_mel]; `out` is resized to
// T_sub * 1024, row-major [T_sub, 1024] — the hidden_20.f32 fixture layout.
//
// ONE CHUNK AT A TIME, ALWAYS (trap 5). GRN reduces over the whole time axis
// of whatever it is handed, so batching two chunks or padding a tail up to
// 30 s changes the features for every frame of it, not just the padding.
//
// Not thread-safe: one cached graph per Yue2MertEncGraph.
static bool yue2_mert_encode_chunk(const Yue2MertModel & m, Yue2MertEncGraph * g, const float * mel, int64_t T_mel,
                                   bool gelu_tanh, std::vector<float> * out, int64_t * out_frames,
                                   std::vector<float> * sub_out, std::vector<std::vector<float>> * taps_out,
                                   std::string * err) {
    if (T_mel <= 0) {
        if (err) {
            *err = "T_mel must be > 0";
        }
        return false;
    }
    if (!yue2_mert_enc_prepare(m, g, err)) {
        return false;
    }
    if (!yue2_mert_enc_ensure_graph(m, g, T_mel, gelu_tanh, sub_out != nullptr, err)) {
        return false;
    }

    const int64_t T_sub = g->output->ne[1];
    ggml_backend_tensor_set(g->input, mel, 0, ggml_nbytes(g->input));

    // Positions restart at 0 in every chunk (pin §3.6).
    std::vector<int32_t> pos((size_t) T_sub);
    for (int64_t t = 0; t < T_sub; t++) {
        pos[(size_t) t] = (int32_t) t;
    }
    ggml_backend_tensor_set(g->pos, pos.data(), 0, pos.size() * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(g->sched, g->graph) != GGML_STATUS_SUCCESS) {
        if (err) {
            *err = "MERT encoder graph compute failed";
        }
        return false;
    }

    const int64_t C = g->output->ne[0];
    out->resize((size_t) (C * T_sub));
    ggml_backend_tensor_get(g->output, out->data(), 0, out->size() * sizeof(float));
    if (sub_out && g->sub) {
        sub_out->resize((size_t) (g->sub->ne[0] * g->sub->ne[1]));
        ggml_backend_tensor_get(g->sub, sub_out->data(), 0, sub_out->size() * sizeof(float));
    }
    if (taps_out) {
        taps_out->assign(g->taps.size(), {});
        for (size_t i = 0; i < g->taps.size(); i++) {
            if (!g->taps[i]) {
                continue;
            }
            (*taps_out)[i].resize((size_t) (g->taps[i]->ne[0] * g->taps[i]->ne[1]));
            ggml_backend_tensor_get(g->taps[i], (*taps_out)[i].data(), 0, (*taps_out)[i].size() * sizeof(float));
        }
    }
    if (out_frames) {
        *out_frames = T_sub;
    }
    return true;
}

// ── Whole-track encode (pin §3.7 + ar_prep.py's instance norm) ──────────────

struct Yue2MertEncodeOptions {
    // MERT's own GELU flavour. false = exact erf, which is what the source
    // says and what the block fixtures confirm. See "GELU: DETERMINED, NOT
    // ASSUMED" above before flipping this.
    bool gelu_tanh = false;

    // The reference's production path rounds the interpolated features to
    // float16 before the instance norm (interpolate(...).half() in
    // prep_real.py). It is part of the reference chain, not an optimisation,
    // and the head was trained on features that went through it — but the FP32
    // oracle (docs/plans/yue2/13-tokenizer-fp32-fixtures.md) deliberately drops
    // it, so its feat25.f32/featnorm.f32 are pure fp32 while stages-30s's
    // 05_head_input.f32 has it ON. Default off; the parity mode sets it per
    // fixture set. Measured cost either way: 0.03-0.13% of frames change
    // argmax (13-tokenizer-fp32-fixtures.md, codes_f16store.i32).
    bool fp16_feature_store = false;

    // Block indices whose output to also return, for bisection. Empty in
    // production — every tap is a live tensor the allocator cannot reuse plus
    // an extra device-to-host copy per chunk.
    std::vector<int> tap_blocks;

    bool want_hidden_raw = false;  // keep the pre-interpolation [T_raw, 1024]
    bool want_feat25     = false;  // keep the post-interpolation, pre-norm [T25, 1024]
    bool want_subsample  = false;  // keep the concatenated subsampling output
};

struct Yue2MertEncodeResult {
    std::vector<float> feat;  // [T25, 1024] instance-normed — the head's input
    int64_t            T25 = 0;

    std::vector<float>              hidden_raw;  // [T_raw, 1024] if want_hidden_raw
    int64_t                         T_raw = 0;
    std::vector<float>              feat25;     // [T25, 1024]   if want_feat25
    std::vector<float>              subsample;  // [T_raw, 1024] if want_subsample
    std::vector<std::vector<float>> taps;       // parallel to Yue2MertEncodeOptions::tap_blocks

    std::vector<Yue2MertChunk> chunks;
    double                     mel_ms   = 0.0;
    double                     graph_ms = 0.0;
    double                     post_ms  = 0.0;
    double                     total_ms = 0.0;
};

namespace yue2_mert_detail {

// Python's round(): half to EVEN, not half away from zero. T25 is
// round(n_samples / sample_rate * frame_rate), and a .5 lands on it whenever
// the track is an exact odd multiple of 20 ms, which is not rare.
static inline int64_t round_half_even(double v) {
    const double f = std::floor(v);
    const double d = v - f;
    int64_t      r = (int64_t) f;
    if (d > 0.5) {
        r += 1;
    } else if (d == 0.5) {
        r += (r & 1) ? 1 : 0;  // ties go to the even neighbour
    }
    return r;
}

// torch's F.interpolate(mode="linear", align_corners=False) along time, on a
// [1, C, T_src] tensor. `src`/`dst` are row-major [T, C] here, which is the
// same bytes on both sides of the reference's .T dance.
//
// The mapping is torch's area_pixel_compute_source_index with align_corners
// false: real = scale*(i + 0.5) - 0.5, CLAMPED AT ZERO (not reflected), then a
// two-tap lerp with the right neighbour clamped to T_src-1. This is not a
// no-op to skip: pin §3.7 measures 2945 raw frames becoming 2946, so it runs
// on every track that is not an exact multiple of 30 s.
static void interp_linear_time(const float * src, int64_t T_src, int64_t C, int64_t T_dst, std::vector<float> * dst) {
    dst->assign((size_t) (T_dst * C), 0.0f);
    if (T_src <= 0 || T_dst <= 0) {
        return;
    }
    const double scale = (double) T_src / (double) T_dst;
    for (int64_t i = 0; i < T_dst; i++) {
        double real = scale * ((double) i + 0.5) - 0.5;
        if (real < 0.0) {
            real = 0.0;
        }
        int64_t      i0  = (int64_t) std::floor(real);
        const double lam = real - (double) i0;
        if (i0 > T_src - 1) {
            i0 = T_src - 1;
        }
        const int64_t i1 = std::min<int64_t>(i0 + 1, T_src - 1);
        const float * a  = src + i0 * C;
        const float * b  = src + i1 * C;
        float *       o  = dst->data() + i * C;
        const float   w1 = (float) lam;
        const float   w0 = 1.0f - w1;
        for (int64_t k = 0; k < C; k++) {
            o[k] = w0 * a[k] + w1 * b[k];
        }
    }
}

}  // namespace yue2_mert_detail

// The full audio -> head-input chain (pin §3.7 plus ar_prep.py's instance
// norm):
//
//   chunk at 30 s, dropping a tail under 1 s
//   per chunk: mel (CPU, fp32) -> subsampling -> 21 Conformer blocks
//   concatenate hidden_states[20] over chunks, in chunk order
//   interpolate along time to T25 = round(S / sample_rate * frame_rate)
//   [optional fp16 round trip]
//   per-channel instance norm over the WHOLE track:
//       (x - mean_over_time) / (std_over_time + 1e-5)
//
// `pcm` is mono at the model's sample rate (24 kHz), `n_samples` long.
// `out->feat` is resized to T25 * 1024, row-major [T25, 1024] — exactly what
// yue2_tok_head_predict() reads, and exactly the featnorm.f32 fixture layout.
//
// THREE THINGS THAT ARE NOT INTERCHANGEABLE WITH SOMETHING SIMPLER:
//
//  * Full chunks and short tails go through the same per-chunk path, but never
//    the same forward. GRN and the convolutions would otherwise see another
//    chunk's frames (trap 5).
//  * The raw concatenation is generally one or two frames SHORT of T25, and
//    the interpolation closes that gap at sub-frame resolution. Padding or
//    truncating instead shifts the whole time axis.
//  * The instance norm is a whole-track statistic. It cannot be streamed
//    without either two passes or accepting a normalisation the head was not
//    trained for. This function does the two passes.
static bool yue2_mert_encode(const Yue2MertModel & m, Yue2MertEncGraph * g, const float * pcm, int64_t n_samples,
                             const Yue2MertEncodeOptions & opt, Yue2MertEncodeResult * out, std::string * err) {
    using namespace yue2_mert_detail;
    const auto t_start = std::chrono::steady_clock::now();

    if (!m.blocks_loaded) {
        if (err) {
            *err = "the MERT Conformer blocks are not loaded — call yue2_mert_load(want_blocks=true)";
        }
        return false;
    }
    const Yue2MertConfig & c = m.cfg;
    const int64_t          C = (int64_t) c.embedding_length;

    *out        = Yue2MertEncodeResult{};
    out->chunks = yue2_mert_chunk_plan(c, n_samples);
    if (out->chunks.empty()) {
        if (err) {
            *err = "audio is " + std::to_string(n_samples) +
                   " samples; the reference drops every chunk under one second, leaving nothing to encode";
        }
        return false;
    }

    g->tap_blocks = opt.tap_blocks;
    out->taps.assign(opt.tap_blocks.size(), {});

    std::vector<float>              hidden;  // [T_raw, 1024], chunks concatenated in order
    std::vector<float>              sub_all;
    std::vector<float>              mel, chunk_out, chunk_sub;
    std::vector<std::vector<float>> chunk_taps;
    int64_t                         T_raw = 0;

    for (const Yue2MertChunk & ch : out->chunks) {
        int64_t    T_mel = 0;
        const auto t0    = std::chrono::steady_clock::now();
        if (!yue2_mert_mel(m, pcm + ch.start, ch.length, &mel, &T_mel, err)) {
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        out->mel_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        int64_t T_sub = 0;
        if (!yue2_mert_encode_chunk(m, g, mel.data(), T_mel, opt.gelu_tanh, &chunk_out, &T_sub,
                                    opt.want_subsample ? &chunk_sub : nullptr,
                                    opt.tap_blocks.empty() ? nullptr : &chunk_taps, err)) {
            return false;
        }
        out->graph_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

        hidden.insert(hidden.end(), chunk_out.begin(), chunk_out.end());
        if (opt.want_subsample) {
            sub_all.insert(sub_all.end(), chunk_sub.begin(), chunk_sub.end());
        }
        for (size_t i = 0; i < out->taps.size() && i < chunk_taps.size(); i++) {
            out->taps[i].insert(out->taps[i].end(), chunk_taps[i].begin(), chunk_taps[i].end());
        }
        T_raw += T_sub;
    }

    const auto t_post = std::chrono::steady_clock::now();
    out->T_raw        = T_raw;
    if (opt.want_subsample) {
        out->subsample = std::move(sub_all);
    }

    // T25 comes from the FULL input length, including any sub-second tail the
    // chunk planner dropped — prep_real.py computes it from len(m24), not from
    // the frames it actually encoded.
    const int64_t T25 = round_half_even((double) n_samples / (double) c.sample_rate * (double) c.frame_rate);
    if (T25 <= 0) {
        if (err) {
            *err = "T25 computed as " + std::to_string(T25);
        }
        return false;
    }
    out->T25 = T25;

    std::vector<float> feat25;
    interp_linear_time(hidden.data(), T_raw, C, T25, &feat25);

    if (opt.fp16_feature_store) {
        for (float & v : feat25) {
            v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
        }
    }

    // Per-channel instance norm over the whole track. Two passes, in double:
    // the reference is float32 numpy (pairwise-summed), so this is strictly
    // more accurate, not differently accurate. Note `std + 1e-5`, NOT
    // clamp_min — the mel frontend uses clamp_min and this one does not, and
    // they are one keystroke apart.
    out->feat.assign((size_t) (T25 * C), 0.0f);
    std::vector<double> mean((size_t) C, 0.0), var((size_t) C, 0.0);
    for (int64_t t = 0; t < T25; t++) {
        const float * row = feat25.data() + t * C;
        for (int64_t k = 0; k < C; k++) {
            mean[(size_t) k] += (double) row[k];
        }
    }
    for (int64_t k = 0; k < C; k++) {
        mean[(size_t) k] /= (double) T25;
    }
    for (int64_t t = 0; t < T25; t++) {
        const float * row = feat25.data() + t * C;
        for (int64_t k = 0; k < C; k++) {
            const double d = (double) row[k] - mean[(size_t) k];
            var[(size_t) k] += d * d;
        }
    }
    std::vector<double> inv((size_t) C);
    for (int64_t k = 0; k < C; k++) {
        const double sd = std::sqrt(var[(size_t) k] / (double) T25);  // population std, ddof=0
        inv[(size_t) k] = 1.0 / (sd + 1e-5);
    }
    for (int64_t t = 0; t < T25; t++) {
        const float * row = feat25.data() + t * C;
        float *       dst = out->feat.data() + t * C;
        for (int64_t k = 0; k < C; k++) {
            dst[k] = (float) (((double) row[k] - mean[(size_t) k]) * inv[(size_t) k]);
        }
    }

    if (opt.want_hidden_raw) {
        out->hidden_raw = std::move(hidden);
    }
    if (opt.want_feat25) {
        out->feat25 = std::move(feat25);
    }

    out->post_ms  = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_post).count();
    out->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    return true;
}
