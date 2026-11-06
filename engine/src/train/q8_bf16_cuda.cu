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

} // namespace

const char * convert(const void * const * src, void * const * dst, const int64_t * n, int count) {
    constexpr int kThreads = 256;
    for (int i = 0; i < count; i++) {
        const int64_t blocks = (n[i] + kThreads - 1) / kThreads;
        q8_0_to_bf16<<<(unsigned int) blocks, kThreads>>>((const BlockQ8_0 *) src[i], (__nv_bfloat16 *) dst[i], n[i]);
    }
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) {
        e = cudaStreamSynchronize(nullptr);
    }
    return e == cudaSuccess ? nullptr : cudaGetErrorString(e);
}

} // namespace hs_q8_bf16_cuda
