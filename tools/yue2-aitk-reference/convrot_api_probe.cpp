#include "convrot_cuda_api.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> std::vector<T> read_array(std::istream &in, size_t count) {
    std::vector<T> result(count);
    if (!in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count * sizeof(T)))) throw std::runtime_error("Truncated fixture");
    return result;
}
template<class T> void write_array(std::ostream &out, const std::vector<T> &data) {
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
}
void cuda_check(cudaError_t status, const char *what) { if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }

int main(int argc, char **argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("Usage: convrot_api_probe INPUT.bin OUTPUT.bin [--no-bias]");
        const bool no_bias = argc == 4 && std::string(argv[3]) == "--no-bias";
        if (argc == 4 && !no_bias) throw std::runtime_error("Unknown option");
        std::ifstream existing(argv[2], std::ios::binary);
        if (existing.good()) throw std::runtime_error("Output exists; choose a new path");
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error("Could not open input");
        const auto h = read_array<uint32_t>(in, 6);
        const aitk_reference::ConvRotCudaShape shape{static_cast<int>(h[1]), static_cast<int>(h[2]), static_cast<int>(h[3]), static_cast<int>(h[4]), h[5] != 0};
        if (h[0] != 0x314b5441u || h[5] > 1 || !aitk_reference::convrot_cuda_validate_shape(shape)) throw std::runtime_error("Invalid fixture header");
        const size_t x_count = static_cast<size_t>(shape.rows) * shape.input_width, y_count = static_cast<size_t>(shape.rows) * shape.output_width;
        auto x = read_array<float>(in, x_count); auto weights = read_array<int8_t>(in, static_cast<size_t>(shape.output_width) * shape.input_width); auto scales = read_array<float>(in, shape.output_width); auto gradient = read_array<float>(in, y_count); auto bias = read_array<float>(in, shape.output_width);
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing fixture data");
        cudaStream_t stream = nullptr; cublasHandle_t handle = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate failed");
        if (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS ||
            cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Could not set caller handle modes");
        const auto check_handle = [&] {
            cublasPointerMode_t pointer_mode;
            cublasMath_t math_mode;
            cudaStream_t restored_stream;
            if (cublasGetPointerMode(handle, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
                cublasGetMathMode(handle, &math_mode) != CUBLAS_STATUS_SUCCESS ||
                cublasGetStream(handle, &restored_stream) != CUBLAS_STATUS_SUCCESS ||
                pointer_mode != CUBLAS_POINTER_MODE_DEVICE || math_mode != CUBLAS_PEDANTIC_MATH || restored_stream != nullptr)
                throw std::runtime_error("Caller handle state was not restored");
        };
        auto destroy = [&] { if (handle) cublasDestroy(handle); if (stream) cudaStreamDestroy(stream); };
        float *dx = nullptr, *d_weight_scales = nullptr, *d_activation_scales = nullptr, *d_bias = nullptr, *d_gradient = nullptr, *d_rotated = nullptr, *d_output = nullptr, *d_input_gradient = nullptr; int8_t *d_weights = nullptr, *d_codes = nullptr; void *workspace = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&dx), x_count * sizeof(float)), "cudaMalloc input");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_weights), weights.size() * sizeof(int8_t)), "cudaMalloc weights");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_weight_scales), scales.size() * sizeof(float)), "cudaMalloc scales");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_activation_scales), shape.rows * sizeof(float)), "cudaMalloc activation scales");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_bias), shape.output_width * sizeof(float)), "cudaMalloc bias");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_gradient), y_count * sizeof(float)), "cudaMalloc gradient");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_rotated), x_count * sizeof(float)), "cudaMalloc rotated");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_codes), x_count * sizeof(int8_t)), "cudaMalloc codes");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_output), y_count * sizeof(float)), "cudaMalloc output");
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&d_input_gradient), x_count * sizeof(float)), "cudaMalloc input gradient");
        const size_t forward_bytes = aitk_reference::convrot_cuda_forward_workspace_bytes(shape), backward_bytes = aitk_reference::convrot_cuda_input_backward_workspace_bytes(shape), workspace_bytes = forward_bytes > backward_bytes ? forward_bytes : backward_bytes;
        constexpr size_t sentinel_bytes = 512;
        cuda_check(cudaMalloc(&workspace, workspace_bytes + sentinel_bytes), "cudaMalloc workspace");
        cuda_check(cudaMemset(workspace, 0xA5, workspace_bytes + sentinel_bytes), "initialize workspace sentinel");
        cuda_check(cudaMemcpyAsync(dx, x.data(), x_count * sizeof(float), cudaMemcpyHostToDevice, stream), "copy input");
        cuda_check(cudaMemcpyAsync(d_weights, weights.data(), weights.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream), "copy weights");
        cuda_check(cudaMemcpyAsync(d_weight_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy scales");
        cuda_check(cudaMemcpyAsync(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy bias");
        cuda_check(cudaMemcpyAsync(d_gradient, gradient.data(), gradient.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy gradient");
        auto status = aitk_reference::convrot_cuda_forward(shape, dx, d_weights, d_weight_scales, no_bias ? nullptr : d_bias, d_rotated, d_codes, d_activation_scales, d_output, workspace, workspace_bytes, stream, handle);
        if (status != aitk_reference::ConvRotCudaStatus::success) throw std::runtime_error(aitk_reference::convrot_cuda_status_string(status));
        check_handle();
        status = aitk_reference::convrot_cuda_input_backward(shape, d_weights, d_weight_scales, d_gradient, d_input_gradient, workspace, workspace_bytes, stream, handle);
        if (status != aitk_reference::ConvRotCudaStatus::success) throw std::runtime_error(aitk_reference::convrot_cuda_status_string(status));
        check_handle();
        cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        std::vector<uint8_t> sentinel(sentinel_bytes);
        cuda_check(cudaMemcpy(sentinel.data(), static_cast<const uint8_t *>(workspace) + workspace_bytes, sentinel_bytes, cudaMemcpyDeviceToHost), "copy workspace sentinel");
        for (uint8_t byte : sentinel) if (byte != 0xA5) throw std::runtime_error("workspace boundary overwritten");
        std::vector<float> rotated(x_count), output(y_count), input_gradient(x_count), activation_scales(shape.rows); std::vector<int8_t> activation_codes(x_count);
        cuda_check(cudaMemcpy(rotated.data(), d_rotated, x_count * sizeof(float), cudaMemcpyDeviceToHost), "copy rotated"); cuda_check(cudaMemcpy(activation_codes.data(), d_codes, x_count * sizeof(int8_t), cudaMemcpyDeviceToHost), "copy codes"); cuda_check(cudaMemcpy(activation_scales.data(), d_activation_scales, shape.rows * sizeof(float), cudaMemcpyDeviceToHost), "copy activation scales"); cuda_check(cudaMemcpy(output.data(), d_output, y_count * sizeof(float), cudaMemcpyDeviceToHost), "copy output"); cuda_check(cudaMemcpy(input_gradient.data(), d_input_gradient, x_count * sizeof(float), cudaMemcpyDeviceToHost), "copy gradient");
        std::ofstream out(argv[2], std::ios::binary); if (!out) throw std::runtime_error("Could not open output"); write_array(out, rotated); write_array(out, activation_codes); write_array(out, activation_scales); write_array(out, output); write_array(out, input_gradient);
        cudaFree(dx); cudaFree(d_weights); cudaFree(d_weight_scales); cudaFree(d_activation_scales); cudaFree(d_bias); cudaFree(d_gradient); cudaFree(d_rotated); cudaFree(d_codes); cudaFree(d_output); cudaFree(d_input_gradient); cudaFree(workspace); destroy();
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
