#pragma once

// Stream-ordered ConvRot8 operation API. All tensor pointers are device
// pointers owned by the caller. The API never allocates device memory and
// never synchronizes the device; the caller owns the workspace for the full
// lifetime of the submitted work and must use an exclusive cuBLAS handle.

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>

namespace aitk_reference {

enum class ConvRotCudaStatus : int {
    success = 0,
    invalid_argument = 1,
    insufficient_workspace = 2,
    cuda_error = 3,
    cublas_error = 4,
    unsupported = 5,
};

struct ConvRotCudaShape {
    int rows;
    int input_width;
    int output_width;
    int rotation;
    bool use_bf16;
};

// Inputs and outputs use contiguous row-major storage. `weights` is
// [output_width,input_width] int8 and `weight_scales` is one FP32 scale per
// output row. Activation codes in workspace are padded internally to a row
// count divisible by 32 for the cuBLAS int8 GEMM; the public tensors remain
// unpadded. FP32 trainable/gradient storage stays FP32; use_bf16 only selects
// the reference BF16 casts and GEMM path.
// bias may be null for YuE2's bias-free projections. Stream, pointer mode and
// math mode on the caller's exclusive handle are restored after submission.
// Shape support covers up to the 24,576-token context; callers must budget
// the reported workspace rather than silently shorten sequences.
bool convrot_cuda_validate_shape(const ConvRotCudaShape &shape);
size_t convrot_cuda_forward_workspace_bytes(const ConvRotCudaShape &shape);
size_t convrot_cuda_input_backward_workspace_bytes(const ConvRotCudaShape &shape);

ConvRotCudaStatus convrot_cuda_forward(
    const ConvRotCudaShape &shape,
    const float *input,
    const int8_t *weights,
    const float *weight_scales,
    const float *bias,
    float *rotated,
    int8_t *activation_codes,
    float *activation_scales,
    float *output,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    cublasHandle_t cublas);

ConvRotCudaStatus convrot_cuda_input_backward(
    const ConvRotCudaShape &shape,
    const int8_t *weights,
    const float *weight_scales,
    const float *gradient,
    float *input_gradient,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    cublasHandle_t cublas);

const char *convrot_cuda_status_string(ConvRotCudaStatus status);

} // namespace aitk_reference
