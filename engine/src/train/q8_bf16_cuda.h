#pragma once

#include <cstdint>

// q8_0 -> BF16 for the LM trainer's --weights bf16 weight stage (lm-ckpt.h).
// The HOT-ggml fork has no direct quant -> BF16 copy, so the in-graph cast goes
// q8_0 -> F32 -> BF16 and writes a full F32 copy of every projection on the
// way. This kernel writes the BF16 values straight away, with the same
// arithmetic: d (F16) widened to float, times the int8 (exact, 19 bits at
// most), rounded to BF16 nearest-even. Bit-identical to the two ggml copies.
namespace hs_q8_bf16_cuda {

// Converts `count` contiguous q8_0 tensors (n[i] elements each, a multiple of
// 32) into BF16 on the current device, then synchronizes. Returns nullptr on
// success, else the CUDA error string.
const char * convert(const void * const * src, void * const * dst, const int64_t * n, int count);

} // namespace hs_q8_bf16_cuda
