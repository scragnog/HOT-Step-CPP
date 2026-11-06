#include "q8_bf16_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace hs_q8_bf16_cuda {
namespace {

// ggml's block_q8_0: an F16 scale and 32 int8 values, 34 bytes.
struct BlockQ8_0 {
    __half d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 layout");

// One element per thread. Same steps as ggml-cuda's dequantize_q8_0 followed
// by cpy_scalar_contiguous<float, nv_bfloat16>: float d, float q, q *= d,
// __float2bfloat16.
__global__ void q8_0_to_bf16(const BlockQ8_0 * src, __nv_bfloat16 * dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const BlockQ8_0 & b = src[i / 32];
    const float       d = b.d;
    float             v = b.qs[i % 32];
    v *= d;
    dst[i] = __float2bfloat16(v);
}

__device__ __forceinline__ float ld(const __half * p, int64_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float ld(const float * p, int64_t i) { return p[i]; }

// The same dequantized value minus scale * (B0 A0)[i, o], summed over k in
// order, in float, then rounded to BF16 once. A thread owns column i for
// kRows consecutive rows o, so each A0 value is loaded once per kRows outputs.
constexpr int kRows = 8;

template <typename T>
__global__ void q8_0_fold_to_bf16(const BlockQ8_0 * src, __nv_bfloat16 * dst, int64_t ne0, int64_t ne1,
                                  const T * A0, const T * B0, int r, float scale) {
    const int64_t i  = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t o0 = (int64_t) blockIdx.y * kRows;
    if (i >= ne0) {
        return;
    }
    const int nr = (int) min((int64_t) kRows, ne1 - o0);
    float     acc[kRows];
    for (int j = 0; j < kRows; j++) {
        acc[j] = 0.0f;
    }
    for (int k = 0; k < r; k++) {
        const float a = ld(A0, (int64_t) k * ne0 + i);
        for (int j = 0; j < kRows; j++) {
            if (j < nr) {
                acc[j] += ld(B0, (o0 + j) * r + k) * a;
            }
        }
    }
    for (int j = 0; j < nr; j++) {
        const int64_t     e = (o0 + j) * ne0 + i;
        const BlockQ8_0 & b = src[e / 32];
        const float       d = b.d;
        float             v = b.qs[e % 32];
        v *= d;
        dst[e] = __float2bfloat16(v - scale * acc[j]);
    }
}

} // namespace

const char * convert(const void * const * src, void * const * dst, const int64_t * ne0, const int64_t * ne1,
                     const Fold * fold, int count) {
    constexpr int kThreads = 256;
    for (int i = 0; i < count; i++) {
        const int64_t      n      = ne0[i] * ne1[i];
        const unsigned int blocks = (unsigned int) ((n + kThreads - 1) / kThreads);
        const BlockQ8_0 *  s      = (const BlockQ8_0 *) src[i];
        __nv_bfloat16 *    d      = (__nv_bfloat16 *) dst[i];
        const Fold *       f      = fold && fold[i].A0 ? &fold[i] : nullptr;
        const dim3         fgrid((unsigned int) ((ne0[i] + kThreads - 1) / kThreads),
                                 (unsigned int) ((ne1[i] + kRows - 1) / kRows));
        if (!f) {
            q8_0_to_bf16<<<blocks, kThreads>>>(s, d, n);
        } else if (f->f16) {
            q8_0_fold_to_bf16<__half><<<fgrid, kThreads>>>(s, d, ne0[i], ne1[i], (const __half *) f->A0,
                                                          (const __half *) f->B0, f->r, f->scale);
        } else {
            q8_0_fold_to_bf16<float><<<fgrid, kThreads>>>(s, d, ne0[i], ne1[i], (const float *) f->A0,
                                                         (const float *) f->B0, f->r, f->scale);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) {
        e = cudaStreamSynchronize(nullptr);
    }
    return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}

} // namespace hs_q8_bf16_cuda
