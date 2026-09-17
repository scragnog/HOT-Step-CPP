#include "convrot_cuda_api.h"

#include "../../engine/src/convrot.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace aitk_reference {
namespace {

constexpr size_t kAlign = 256;
constexpr int kThreads = 256;
__device__ __constant__ float kH4[16] = {1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1};

struct Workspace {
    char *base;
    size_t left;

    void *take(size_t bytes, size_t alignment = kAlign) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(base);
        const uintptr_t aligned = (address + alignment - 1) & ~(alignment - 1);
        const size_t skip = static_cast<size_t>(aligned - address);
        if (skip > left || bytes > left - skip) return nullptr;
        base = reinterpret_cast<char *>(aligned + bytes);
        left -= skip + bytes;
        return reinterpret_cast<void *>(aligned);
    }
};

size_t padded_rows(const ConvRotCudaShape &s) { return (static_cast<size_t>(s.rows) + 31u) / 32u * 32u; }
size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return 0;
    return a * b;
}
size_t bytes_for(const ConvRotCudaShape &s, bool backward) {
    if (!convrot_cuda_validate_shape(s)) return 0;
    const size_t pr = padded_rows(s), in = static_cast<size_t>(s.input_width), out = static_cast<size_t>(s.output_width);
    size_t total = 0;
    auto add = [&total](size_t count, size_t element) {
        const size_t bytes = checked_mul(count, element);
        if (!bytes || total > std::numeric_limits<size_t>::max() - bytes - kAlign) { total = 0; return; }
        total += kAlign - 1 + bytes;
    };
    if (!backward) {
        add(checked_mul(pr, in), sizeof(int8_t));
        add(checked_mul(pr, out), sizeof(int32_t));
        add(pr, sizeof(float));
    } else {
        if (s.use_bf16) {
            add(checked_mul(out, in), sizeof(__nv_bfloat16));
            add(checked_mul(s.rows, out), sizeof(__nv_bfloat16));
            add(checked_mul(s.rows, in), sizeof(__nv_bfloat16));
            if (s.rotation != 1) add(checked_mul(s.rotation, s.rotation), sizeof(__nv_bfloat16));
            if (s.rotation != 1) add(checked_mul(s.rows, in), sizeof(__nv_bfloat16));
        } else {
            add(checked_mul(out, in), sizeof(float));
        }
    }
    return total;
}

ConvRotCudaStatus cuda_status(cudaError_t status) { return status == cudaSuccess ? ConvRotCudaStatus::success : ConvRotCudaStatus::cuda_error; }
ConvRotCudaStatus cublas_status(cublasStatus_t status) { return status == CUBLAS_STATUS_SUCCESS ? ConvRotCudaStatus::success : ConvRotCudaStatus::cublas_error; }

__device__ __forceinline__ float bf16_cast(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float div_full(float x, float y) {
    float value;
    asm("div.full.f32 %0, %1, %2;" : "=f"(value) : "f"(x), "f"(y));
    return value;
}
__device__ __forceinline__ int round_even(float x) {
    const float lo = floorf(x), fraction = x - lo;
    int result = static_cast<int>(lo);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
    return result;
}

__global__ void rotate_rows(float *out, const float *in, int rows, int width, int group, bool use_bf16) {
    const int groups = width / group, block = static_cast<int>(blockIdx.x), row = block / groups, group_index = block % groups;
    if (row >= rows) return;
    extern __shared__ float values[];
    const size_t base = static_cast<size_t>(row) * width + static_cast<size_t>(group_index) * group;
    for (int i = threadIdx.x; i < group; i += blockDim.x) values[i] = use_bf16 ? bf16_cast(in[base + i]) : in[base + i];
    __syncthreads();
    for (int stride = 1; stride < group; stride *= 4) {
        const int block_width = stride * 4;
        for (int offset = threadIdx.x; offset < group; offset += blockDim.x) {
            const int base_group = (offset / block_width) * block_width, off = offset % block_width;
            if (off < stride) {
                float *p = values + base_group + off;
                const float x0 = p[0], x1 = p[stride], x2 = p[2 * stride], x3 = p[3 * stride];
                p[0] = x0 + x1 + x2 - x3; p[stride] = x0 + x1 - x2 + x3;
                p[2 * stride] = x0 - x1 + x2 + x3; p[3 * stride] = -x0 + x1 + x2 + x3;
            }
        }
        __syncthreads();
    }
    const float norm = rsqrtf(static_cast<float>(group));
    for (int i = threadIdx.x; i < group; i += blockDim.x) {
        const float value = values[i] * norm;
        out[base + i] = use_bf16 ? bf16_cast(value) : value;
    }
}

__global__ void quantize_rows(const float *rotated, int8_t *codes, float *scales, int rows, int width) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float partial[kThreads];
    float local = 0;
    for (int k = threadIdx.x; k < width; k += blockDim.x) local = fmaxf(local, fabsf(rotated[static_cast<size_t>(row) * width + k]));
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) { if (threadIdx.x < step) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + step]); __syncthreads(); }
    const float scale = partial[0] > 0 ? div_full(partial[0], 127.0f) : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
        const int q = max(-127, min(127, round_even(div_full(rotated[static_cast<size_t>(row) * width + k], scale))));
        codes[static_cast<size_t>(row) * width + k] = static_cast<int8_t>(q);
    }
}

__global__ void forward_epilogue(const int32_t *acc, const float *scales, const float *weight_scales, const float *bias, float *out, int rows, int width, bool use_bf16) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(rows) * width) return;
    const int row = static_cast<int>(index / width), col = static_cast<int>(index % width);
    const float b = bias ? (use_bf16 ? bf16_cast(bias[col]) : bias[col]) : 0.0f;
    const float scaled = __fmul_rn(scales[row], weight_scales[col]);
    const float value = __fmaf_rn(static_cast<float>(acc[index]), scaled, b);
    out[index] = use_bf16 ? bf16_cast(value) : value;
}

__global__ void dequant_weights(const int8_t *codes, const float *scales, float *out, __nv_bfloat16 *out_bf16, int width, size_t total, bool use_bf16) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const float scale = use_bf16 ? bf16_cast(scales[index / width]) : scales[index / width];
    const float value = static_cast<float>(codes[index]) * scale;
    if (use_bf16) out_bf16[index] = __float2bfloat16(value); else out[index] = value;
}

__global__ void cast_bf16(const float *in, __nv_bfloat16 *out, size_t count) { const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; if (i < count) out[i] = __float2bfloat16(in[i]); }
__global__ void cast_float(const __nv_bfloat16 *in, float *out, size_t count) { const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; if (i < count) out[i] = __bfloat162float(in[i]); }

__global__ void make_hadamard(__nv_bfloat16 *out, int group) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x, total = static_cast<size_t>(group) * group;
    if (index >= total) return;
    const int r = static_cast<int>(index / group), c = static_cast<int>(index % group);
    float sign = 1.0f;
    for (int shift = 0; shift < 16; shift += 2) sign *= kH4[((r >> shift) & 3) * 4 + ((c >> shift) & 3)];
    out[index] = __float2bfloat16(sign / sqrtf(static_cast<float>(group)));
}

class StreamGuard {
public:
    StreamGuard(cublasHandle_t handle, cudaStream_t stream) : handle_(handle) {
        if (cublasGetStream(handle_, &old_) == CUBLAS_STATUS_SUCCESS) {
            status_ = cublasSetStream(handle_, stream);
        } else status_ = CUBLAS_STATUS_INTERNAL_ERROR;
    }
    ~StreamGuard() { if (status_ == CUBLAS_STATUS_SUCCESS) cublasSetStream(handle_, old_); }
    cublasStatus_t status() const { return status_; }
private:
    cublasHandle_t handle_; cudaStream_t old_ = nullptr; cublasStatus_t status_;
};

class MathGuard {
public:
    explicit MathGuard(cublasHandle_t handle) : handle_(handle) {
        if (cublasGetMathMode(handle_, &old_) == CUBLAS_STATUS_SUCCESS) status_ = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
        else status_ = CUBLAS_STATUS_INTERNAL_ERROR;
    }
    ~MathGuard() { if (status_ == CUBLAS_STATUS_SUCCESS) cublasSetMathMode(handle_, old_); }
    cublasStatus_t status() const { return status_; }
private:
    cublasHandle_t handle_; cublasMath_t old_{}; cublasStatus_t status_;
};

class PointerModeGuard {
public:
    explicit PointerModeGuard(cublasHandle_t handle) : handle_(handle) {
        status_ = cublasGetPointerMode(handle_, &old_);
        if (status_ == CUBLAS_STATUS_SUCCESS) status_ = cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    ~PointerModeGuard() { if (status_ == CUBLAS_STATUS_SUCCESS) cublasSetPointerMode(handle_, old_); }
    cublasStatus_t status() const { return status_; }
private:
    cublasHandle_t handle_;
    cublasPointerMode_t old_ = CUBLAS_POINTER_MODE_HOST;
    cublasStatus_t status_;
};

} // namespace

bool convrot_cuda_validate_shape(const ConvRotCudaShape &s) {
    return s.rows > 0 && s.rows <= 24576 && s.input_width > 0 && s.input_width <= 8192 &&
        s.output_width > 0 && s.output_width <= 16384 && s.input_width % 16 == 0 &&
        s.output_width % 8 == 0 && (s.input_width >= 128 || s.output_width % 16 == 0) &&
        s.rotation > 0 && s.rotation <= 4096 && (s.rotation == 1 || convrot_is_pow4(s.rotation)) &&
        s.input_width % s.rotation == 0;
}

size_t convrot_cuda_forward_workspace_bytes(const ConvRotCudaShape &s) { return bytes_for(s, false); }
size_t convrot_cuda_input_backward_workspace_bytes(const ConvRotCudaShape &s) { return bytes_for(s, true); }

ConvRotCudaStatus convrot_cuda_forward(const ConvRotCudaShape &s, const float *input, const int8_t *weights, const float *weight_scales, const float *bias, float *rotated, int8_t *activation_codes, float *activation_scales, float *output, void *workspace, size_t workspace_bytes, cudaStream_t stream, cublasHandle_t handle) {
    if (!convrot_cuda_validate_shape(s) || !input || !weights || !weight_scales || !rotated || !activation_codes || !activation_scales || !output || !workspace || !handle) return ConvRotCudaStatus::invalid_argument;
    const size_t need = convrot_cuda_forward_workspace_bytes(s); if (workspace_bytes < need) return ConvRotCudaStatus::insufficient_workspace;
    Workspace w{static_cast<char *>(workspace), workspace_bytes};
    auto *codes = static_cast<int8_t *>(w.take(padded_rows(s) * s.input_width * sizeof(int8_t)));
    auto *acc = static_cast<int32_t *>(w.take(padded_rows(s) * s.output_width * sizeof(int32_t)));
    auto *scales = static_cast<float *>(w.take(padded_rows(s) * sizeof(float)));
    if (!codes || !acc || !scales) return ConvRotCudaStatus::insufficient_workspace;
    StreamGuard guard(handle, stream); if (guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    MathGuard math_guard(handle); if (math_guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    PointerModeGuard pointer_guard(handle); if (pointer_guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    const size_t padded_codes = padded_rows(s) * s.input_width;
    cudaError_t ce = cudaMemsetAsync(codes, 0, padded_codes * sizeof(int8_t), stream); if (ce != cudaSuccess) return cuda_status(ce);
    rotate_rows<<<s.rows * (s.input_width / s.rotation), kThreads, static_cast<size_t>(s.rotation) * sizeof(float), stream>>>(rotated, input, s.rows, s.input_width, s.rotation, s.use_bf16); ce = cudaGetLastError(); if (ce != cudaSuccess) return cuda_status(ce);
    quantize_rows<<<s.rows, kThreads, 0, stream>>>(rotated, codes, scales, s.rows, s.input_width); ce = cudaGetLastError(); if (ce != cudaSuccess) return cuda_status(ce);
    ce = cudaMemcpy2DAsync(activation_codes, static_cast<size_t>(s.input_width) * sizeof(int8_t), codes, static_cast<size_t>(s.input_width) * sizeof(int8_t), static_cast<size_t>(s.input_width) * sizeof(int8_t), s.rows, cudaMemcpyDeviceToDevice, stream); if (ce != cudaSuccess) return cuda_status(ce);
    ce = cudaMemcpyAsync(activation_scales, scales, static_cast<size_t>(s.rows) * sizeof(float), cudaMemcpyDeviceToDevice, stream); if (ce != cudaSuccess) return cuda_status(ce);
    const int32_t one = 1, zero = 0; cublasStatus_t cs = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, s.output_width, static_cast<int>(padded_rows(s)), s.input_width, &one, weights, CUDA_R_8I, s.input_width, codes, CUDA_R_8I, s.input_width, &zero, acc, CUDA_R_32I, s.output_width, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT); if (cs != CUBLAS_STATUS_SUCCESS) return cublas_status(cs);
    forward_epilogue<<<static_cast<unsigned>((static_cast<size_t>(s.rows) * s.output_width + kThreads - 1) / kThreads), kThreads, 0, stream>>>(acc, scales, weight_scales, bias, output, s.rows, s.output_width, s.use_bf16); ce = cudaGetLastError(); return cuda_status(ce);
}

ConvRotCudaStatus convrot_cuda_input_backward(const ConvRotCudaShape &s, const int8_t *weights, const float *weight_scales, const float *gradient, float *input_gradient, void *workspace, size_t workspace_bytes, cudaStream_t stream, cublasHandle_t handle) {
    if (!convrot_cuda_validate_shape(s) || !weights || !weight_scales || !gradient || !input_gradient || !workspace || !handle) return ConvRotCudaStatus::invalid_argument;
    const size_t need = convrot_cuda_input_backward_workspace_bytes(s); if (workspace_bytes < need) return ConvRotCudaStatus::insufficient_workspace;
    Workspace w{static_cast<char *>(workspace), workspace_bytes};
    float *weights_fp32 = nullptr; __nv_bfloat16 *weights_bf16 = nullptr, *gradient_bf16 = nullptr, *dx_bf16 = nullptr, *hadamard = nullptr, *rotated_dx = nullptr;
    if (s.use_bf16) { weights_bf16 = static_cast<__nv_bfloat16 *>(w.take(static_cast<size_t>(s.output_width) * s.input_width * sizeof(__nv_bfloat16))); gradient_bf16 = static_cast<__nv_bfloat16 *>(w.take(static_cast<size_t>(s.rows) * s.output_width * sizeof(__nv_bfloat16))); dx_bf16 = static_cast<__nv_bfloat16 *>(w.take(static_cast<size_t>(s.rows) * s.input_width * sizeof(__nv_bfloat16))); if (s.rotation != 1) { hadamard = static_cast<__nv_bfloat16 *>(w.take(static_cast<size_t>(s.rotation) * s.rotation * sizeof(__nv_bfloat16))); rotated_dx = static_cast<__nv_bfloat16 *>(w.take(static_cast<size_t>(s.rows) * s.input_width * sizeof(__nv_bfloat16))); } } else weights_fp32 = static_cast<float *>(w.take(static_cast<size_t>(s.output_width) * s.input_width * sizeof(float)));
    if ((s.use_bf16 && (!weights_bf16 || !gradient_bf16 || !dx_bf16 || (s.rotation != 1 && (!hadamard || !rotated_dx)))) || (!s.use_bf16 && !weights_fp32)) return ConvRotCudaStatus::insufficient_workspace;
    StreamGuard guard(handle, stream); if (guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    MathGuard math_guard(handle); if (math_guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    PointerModeGuard pointer_guard(handle); if (pointer_guard.status() != CUBLAS_STATUS_SUCCESS) return ConvRotCudaStatus::cublas_error;
    const size_t weight_count = static_cast<size_t>(s.output_width) * s.input_width, grad_count = static_cast<size_t>(s.rows) * s.output_width, dx_count = static_cast<size_t>(s.rows) * s.input_width;
    dequant_weights<<<static_cast<unsigned>((weight_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(weights, weight_scales, weights_fp32, weights_bf16, s.input_width, weight_count, s.use_bf16); cudaError_t ce = cudaGetLastError(); if (ce != cudaSuccess) return cuda_status(ce);
    const void *g_ptr = gradient; if (s.use_bf16) { cast_bf16<<<static_cast<unsigned>((grad_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(gradient, gradient_bf16, grad_count); ce = cudaGetLastError(); if (ce != cudaSuccess) return cuda_status(ce); g_ptr = gradient_bf16; }
    const void *w_ptr = s.use_bf16 ? static_cast<const void *>(weights_bf16) : static_cast<const void *>(weights_fp32); void *dx_ptr = s.use_bf16 ? static_cast<void *>(dx_bf16) : static_cast<void *>(input_gradient); const cudaDataType_t type = s.use_bf16 ? CUDA_R_16BF : CUDA_R_32F; const float one = 1, zero = 0;
    cublasStatus_t cs = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.input_width, s.rows, s.output_width, &one, w_ptr, type, s.input_width, g_ptr, type, s.output_width, &zero, dx_ptr, type, s.input_width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); if (cs != CUBLAS_STATUS_SUCCESS) return cublas_status(cs);
    if (s.use_bf16 && s.rotation != 1) { make_hadamard<<<static_cast<unsigned>((static_cast<size_t>(s.rotation) * s.rotation + kThreads - 1) / kThreads), kThreads, 0, stream>>>(hadamard, s.rotation); ce = cudaGetLastError(); if (ce != cudaSuccess) return cuda_status(ce); cs = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, s.rotation, s.rows * (s.input_width / s.rotation), s.rotation, &one, hadamard, CUDA_R_16BF, s.rotation, dx_bf16, CUDA_R_16BF, s.rotation, &zero, rotated_dx, CUDA_R_16BF, s.rotation, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); if (cs != CUBLAS_STATUS_SUCCESS) return cublas_status(cs); cast_float<<<static_cast<unsigned>((dx_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(rotated_dx, input_gradient, dx_count); } else if (s.use_bf16) { cast_float<<<static_cast<unsigned>((dx_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(dx_bf16, input_gradient, dx_count); } else { rotate_rows<<<s.rows * (s.input_width / s.rotation), kThreads, static_cast<size_t>(s.rotation) * sizeof(float), stream>>>(input_gradient, input_gradient, s.rows, s.input_width, s.rotation, false); }
    ce = cudaGetLastError(); return cuda_status(ce);
}

const char *convrot_cuda_status_string(ConvRotCudaStatus status) { switch (status) { case ConvRotCudaStatus::success: return "success"; case ConvRotCudaStatus::invalid_argument: return "invalid_argument"; case ConvRotCudaStatus::insufficient_workspace: return "insufficient_workspace"; case ConvRotCudaStatus::cuda_error: return "cuda_error"; case ConvRotCudaStatus::cublas_error: return "cublas_error"; case ConvRotCudaStatus::unsupported: return "unsupported"; } return "unknown"; }

} // namespace aitk_reference
