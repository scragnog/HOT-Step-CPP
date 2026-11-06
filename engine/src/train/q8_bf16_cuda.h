#pragma once

#include <cstdint>

// q8_0 -> BF16 for the LM trainer's --weights bf16 weight stage (lm-ckpt.h).
// The HOT-ggml fork has no direct quant -> BF16 copy, so the in-graph cast goes
// q8_0 -> F32 -> BF16 and writes a full F32 copy of every projection on the
// way. This kernel writes the BF16 values straight away, with the same
// arithmetic: d (F16) widened to float, times the int8 (exact, 19 bits at
// most), rounded to BF16 nearest-even. Bit-identical to the two ggml copies.
namespace hs_q8_bf16_cuda {

// --pissa-fold: the stage holds W - s B0 A0 instead of W. A0 is the ggml
// [in, r] tensor (A0[k * in + i]), B0 the [r, out] one (B0[o * r + k]), both
// F16 or both F32. A null A0 means no fold for that tensor.
struct Fold {
    const void * A0    = nullptr;
    const void * B0    = nullptr;
    bool         f16   = false;
    int          r     = 0;
    float        scale = 0.0f;
};

// Converts `count` contiguous q8_0 tensors of [ne0, ne1] (ne0 a multiple of
// 32) into BF16 on the current device, folding where fold[i].A0 is set (fold
// may be null), then synchronizes. Returns nullptr on success, else the CUDA
// error string.
const char * convert(const void * const * src, void * const * dst, const int64_t * ne0, const int64_t * ne1,
                     const Fold * fold, int count);

} // namespace hs_q8_bf16_cuda
