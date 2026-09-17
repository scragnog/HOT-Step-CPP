// Standalone CUDA ConvRot8 reference executable.
// Contract matches convrot_probe.cpp; this file intentionally has no engine
// or Python runtime dependency. It is a reference implementation, not a
// production training kernel.
#include "../../engine/src/convrot.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status, const char *what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
void check_cublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
}
#define CUDA_CHECK(x) check_cuda((x), #x)
#define CUBLAS_CHECK(x) check_cublas((x), #x)

template<class T> class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void allocate(size_t count) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(T))); count_ = count; }
    T *get() const { return ptr_; }
    size_t size() const { return count_; }
private:
    T *ptr_ = nullptr;
    size_t count_ = 0;
};

class CublasHandle {
public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() { if (handle_) cublasDestroy(handle_); }
    CublasHandle(const CublasHandle &) = delete;
    cublasHandle_t get() const { return handle_; }
private:
    cublasHandle_t handle_ = nullptr;
};

template<class T> std::vector<T> read_array(std::istream &input, size_t count) {
    std::vector<T> value(count);
    if (!input.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(count * sizeof(T)))) throw std::runtime_error("Truncated fixture");
    return value;
}
template<class T> void write_array(std::ostream &output, const std::vector<T> &value) {
    output.write(reinterpret_cast<const char *>(value.data()), static_cast<std::streamsize>(value.size() * sizeof(T)));
}

__device__ __forceinline__ float bf16_cast(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float triton_div(float x, float y) {
    // Match pinned triton-windows 3.5's emitted PTX, including quantization ties.
    float value;
    asm("div.full.f32 %0, %1, %2;" : "=f"(value) : "f"(x), "f"(y));
    return value;
}
__device__ __forceinline__ int round_even_device(float x) {
    const float lo = floorf(x);
    const float fraction = x - lo;
    int result = static_cast<int>(lo);
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
    return result;
}

// One block owns one row/group. Shared memory implements the same radix-4
// regular Hadamard transform as convrot_transform_group, then casts BF16 at
// the contract boundaries. Applying this kernel again is the inverse.
__global__ void rotate_groups(float *out, const float *in, int rows, int width, int group, bool use_bf16) {
    const int groups_per_row = width / group;
    const int block = static_cast<int>(blockIdx.x);
    const int row = block / groups_per_row;
    const int group_index = block % groups_per_row;
    if (row >= rows) return;
    extern __shared__ float values[];
    const size_t base = static_cast<size_t>(row) * width + static_cast<size_t>(group_index) * group;
    for (int i = threadIdx.x; i < group; i += blockDim.x) values[i] = use_bf16 ? bf16_cast(in[base + i]) : in[base + i];
    __syncthreads();
    for (int stride = 1; stride < group; stride *= 4) {
        const int block_width = stride * 4;
        for (int base_group = 0; base_group < group; base_group += block_width) {
            for (int off = threadIdx.x; off < stride; off += blockDim.x) {
                float *p = values + base_group + off;
                const float x0 = p[0], x1 = p[stride], x2 = p[2 * stride], x3 = p[3 * stride];
                p[0] = x0 + x1 + x2 - x3;
                p[stride] = x0 + x1 - x2 + x3;
                p[2 * stride] = x0 - x1 + x2 + x3;
                p[3 * stride] = -x0 + x1 + x2 + x3;
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
    __shared__ float partial[256];
    float local = 0.0f;
    for (int k = threadIdx.x; k < width; k += blockDim.x) local = fmaxf(local, fabsf(rotated[static_cast<size_t>(row) * width + k]));
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + step]);
        __syncthreads();
    }
    const float scale = partial[0] > 0.0f ? triton_div(partial[0], 127.0f) : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
        const int q = max(-127, min(127, round_even_device(triton_div(rotated[static_cast<size_t>(row) * width + k], scale))));
        codes[static_cast<size_t>(row) * width + k] = static_cast<int8_t>(q);
    }
}

__global__ void forward_epilogue(const int32_t *acc, const float *scales, const float *weight_scales, const float *bias, float *out, int rows, int width, bool use_bf16) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<size_t>(rows) * width) return;
    const int row = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const float bias_cast = use_bf16 ? bf16_cast(bias[col]) : bias[col];
    const float scaled = __fmul_rn(scales[row], weight_scales[col]);
    const float value = __fmaf_rn(static_cast<float>(acc[index]), scaled, bias_cast);
    out[index] = use_bf16 ? bf16_cast(value) : value;
}

__global__ void dequant_weights(const int8_t *codes, const float *scales, float *out, __nv_bfloat16 *out_bf16, int out_width, size_t total, bool use_bf16) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / out_width);
    const float scale = use_bf16 ? bf16_cast(scales[row]) : scales[row];
    const float value = static_cast<float>(codes[index]) * scale;
    if (use_bf16) out_bf16[index] = __float2bfloat16(value);
    else out[index] = value;
}

__global__ void cast_gradient(const float *in, __nv_bfloat16 *out, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] = __float2bfloat16(in[index]);
}

__global__ void bfloat_to_float(const __nv_bfloat16 *in, float *out, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] = __bfloat162float(in[index]);
}

void launch_check() { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: convrot_cuda INPUT.bin OUTPUT.bin");
        if (std::strcmp(argv[1], argv[2]) == 0) throw std::runtime_error("Input and output must differ");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("Could not open input");
        const auto h = read_array<uint32_t>(input, 6);
        if (h[0] != 0x314b5441u || h[1] == 0 || h[1] > 1024 || h[2] == 0 || h[2] > 8192 || h[3] == 0 || h[3] > 16384 || h[5] > 1 || (h[4] != 1 && !convrot_is_pow4(static_cast<int>(h[4]))) || h[4] > 4096 || h[2] % h[4] != 0)
            throw std::runtime_error("Invalid fixture header");
        const int rows = static_cast<int>(h[1]), in = static_cast<int>(h[2]), out = static_cast<int>(h[3]), rotation = static_cast<int>(h[4]);
        const int padded_rows = (rows + 31) / 32 * 32;
        const bool use_bf16 = h[5] != 0;
        auto x = read_array<float>(input, static_cast<size_t>(rows) * in);
        auto w = read_array<int8_t>(input, static_cast<size_t>(out) * in);
        auto scales = read_array<float>(input, out);
        auto gradient = read_array<float>(input, static_cast<size_t>(rows) * out);
        auto bias = read_array<float>(input, out);
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing fixture data");
        for (float value : x) if (!std::isfinite(value)) throw std::runtime_error("Nonfinite activation");
        std::ifstream existing(argv[2], std::ios::binary);
        if (existing.good()) throw std::runtime_error("Output exists; choose a new path");

        const size_t x_count = static_cast<size_t>(rows) * in, y_count = static_cast<size_t>(rows) * out;
        const size_t padded_x_count = static_cast<size_t>(padded_rows) * in, padded_y_count = static_cast<size_t>(padded_rows) * out;
        DeviceBuffer<float> d_x(x_count), d_rotated(padded_x_count), d_scales(padded_rows), d_weight_scales(out), d_bias(out), d_gradient(y_count), d_output(y_count), d_dx(x_count), d_final_dx(x_count), d_w_float(static_cast<size_t>(out) * in);
        DeviceBuffer<int8_t> d_codes(padded_x_count), d_weights(static_cast<size_t>(out) * in);
        DeviceBuffer<int32_t> d_acc(padded_y_count);
        DeviceBuffer<__nv_bfloat16> d_w_bf16(static_cast<size_t>(out) * in), d_gradient_bf16(y_count), d_dx_bf16(x_count);
        CUDA_CHECK(cudaMemcpy(d_x.get(), x.data(), x_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weights.get(), w.data(), w.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight_scales.get(), scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gradient.get(), gradient.data(), y_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias.get(), bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = 256;
        CUDA_CHECK(cudaMemset(d_rotated.get(), 0, padded_x_count * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_codes.get(), 0, padded_x_count * sizeof(int8_t)));
        rotate_groups<<<rows * (in / rotation), threads, static_cast<size_t>(rotation) * sizeof(float)>>>(d_rotated.get(), d_x.get(), rows, in, rotation, use_bf16);
        launch_check();
        quantize_rows<<<rows, threads>>>(d_rotated.get(), d_codes.get(), d_scales.get(), rows, in);
        launch_check();

        CublasHandle blas;
        // Pedantic mode prevents TF32 tensor-core contraction in the FP32
        // backward diagnostic; BF16 mode remains an explicit BF16 GEMM.
        CUBLAS_CHECK(cublasSetMathMode(blas.get(), CUBLAS_DEFAULT_MATH));
        const int32_t one_i = 1, zero_i = 0;
        CUBLAS_CHECK(cublasGemmEx(blas.get(), CUBLAS_OP_T, CUBLAS_OP_N, out, padded_rows, in, &one_i, d_weights.get(), CUDA_R_8I, in, d_codes.get(), CUDA_R_8I, in, &zero_i, d_acc.get(), CUDA_R_32I, out, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        forward_epilogue<<<static_cast<unsigned>((y_count + threads - 1) / threads), threads>>>(d_acc.get(), d_scales.get(), d_weight_scales.get(), d_bias.get(), d_output.get(), rows, out, use_bf16);
        launch_check();

        const unsigned total_weights = static_cast<unsigned>((static_cast<size_t>(out) * in + threads - 1) / threads);
        dequant_weights<<<total_weights, threads, 0, 0>>>(d_weights.get(), d_weight_scales.get(), d_w_float.get(), d_w_bf16.get(), in, static_cast<size_t>(out) * in, use_bf16);
        launch_check();
        if (use_bf16) {
            cast_gradient<<<static_cast<unsigned>((y_count + threads - 1) / threads), threads>>>(d_gradient.get(), d_gradient_bf16.get(), y_count);
            launch_check();
        }
        const float one_f = 1.0f, zero_f = 0.0f;
        const void *w_ptr = use_bf16 ? static_cast<const void *>(d_w_bf16.get()) : static_cast<const void *>(d_w_float.get());
        const void *g_ptr = use_bf16 ? static_cast<const void *>(d_gradient_bf16.get()) : static_cast<const void *>(d_gradient.get());
        const cudaDataType_t dtype = use_bf16 ? CUDA_R_16BF : CUDA_R_32F;
        void *dx_ptr = use_bf16 ? static_cast<void *>(d_dx_bf16.get()) : static_cast<void *>(d_dx.get());
        CUBLAS_CHECK(cublasGemmEx(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, in, rows, out, &one_f, w_ptr, dtype, in, g_ptr, dtype, out, &zero_f, dx_ptr, dtype, in, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        launch_check();
        if (use_bf16) {
            // The STE contract casts the GEMM result before the inverse rotation.
            bfloat_to_float<<<static_cast<unsigned>((x_count + threads - 1) / threads), threads>>>(d_dx_bf16.get(), d_dx.get(), x_count);
            launch_check();
        }
        if (use_bf16 && rotation != 1) {
            // The reference differentiates a BF16 matmul rotation. Preserve its
            // accumulation path: radix-4 is mathematically equal but differs at ties.
            std::vector<float> h_float;
            if (!convrot_build_h_data(rotation, h_float)) throw std::runtime_error("Invalid Hadamard size");
            DeviceBuffer<float> d_h_float(h_float.size());
            DeviceBuffer<__nv_bfloat16> d_h(h_float.size()), d_rotated_dx(x_count);
            CUDA_CHECK(cudaMemcpy(d_h_float.get(), h_float.data(), h_float.size()*sizeof(float), cudaMemcpyHostToDevice));
            cast_gradient<<<static_cast<unsigned>((h_float.size()+threads-1)/threads), threads>>>(d_h_float.get(), d_h.get(), h_float.size());
            launch_check();
            CUBLAS_CHECK(cublasGemmEx(blas.get(), CUBLAS_OP_N, CUBLAS_OP_N, rotation,
                rows*(in/rotation), rotation, &one_f, d_h.get(), CUDA_R_16BF, rotation,
                d_dx_bf16.get(), CUDA_R_16BF, rotation, &zero_f, d_rotated_dx.get(), CUDA_R_16BF,
                rotation, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            bfloat_to_float<<<static_cast<unsigned>((x_count+threads-1)/threads), threads>>>(d_rotated_dx.get(), d_final_dx.get(), x_count);
            launch_check();
        } else {
            rotate_groups<<<rows * (in / rotation), threads, static_cast<size_t>(rotation) * sizeof(float)>>>(d_final_dx.get(), d_dx.get(), rows, in, rotation, use_bf16);
            launch_check();
        }

        std::vector<float> rotated(x_count), output(y_count), input_gradient(x_count), activation_scales(rows);
        std::vector<int8_t> activation_codes(x_count);
        CUDA_CHECK(cudaMemcpy(rotated.data(), d_rotated.get(), x_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(activation_codes.data(), d_codes.get(), x_count * sizeof(int8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(activation_scales.data(), d_scales.get(), rows * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), y_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(input_gradient.data(), d_final_dx.get(), x_count * sizeof(float), cudaMemcpyDeviceToHost));
        std::ofstream result(argv[2], std::ios::binary);
        if (!result) throw std::runtime_error("Could not write output");
        write_array(result, rotated); write_array(result, activation_codes); write_array(result, activation_scales); write_array(result, output); write_array(result, input_gradient);
        if (!result) throw std::runtime_error("Could not write output");
        std::cout << "ConvRot8 CUDA fixture completed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
