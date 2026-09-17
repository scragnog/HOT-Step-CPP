// Derived from bitsandbytes 0.49.2 csrc/kernels.cu,
// f0e6ca31b32c4744a9cee4e31610b25796cbf778.
// The complete upstream MIT licence accompanies this file.
#include "adamw8bit_cuda.h"
#include "adamw8bit_default_maps.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace aitk_adamw8bit {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ uint8_t quantize_midpoint(float value, const float *codebook) {
    int low = 0;
    int high = 255;
    while (high - low > 1) {
        const int middle = (low + high) / 2;
        if (value > codebook[middle]) low = middle;
        else high = middle;
    }
    return static_cast<uint8_t>(value > (codebook[low] + codebook[high]) * .5f ? high : low);
}

__global__ void adamw_fp32_kernel(
    float *parameter, const float *gradient, float *moment1, float *moment2, int elements, AdamW8bitCudaConfig config
) {
    const int index = blockIdx.x * kThreads + threadIdx.x;
    if (index >= elements) return;

    const float grad = gradient[index];
    if (!isfinite(grad)) {
        moment1[index] = 0;
        moment2[index] = 0;
        return;
    }

    const float state1 = moment1[index] * config.beta1 + grad * (1.f - config.beta1);
    const float state2 = moment2[index] * config.beta2 + grad * grad * (1.f - config.beta2);
    moment1[index] = state1;
    moment2[index] = state2;

    const float correction1 = 1.f - __powf(config.beta1, config.step);
    const float correction2 = sqrtf(1.f - __powf(config.beta2, config.step));
    parameter[index] += __fdividef(-config.learning_rate * correction2, correction1) *
                        __fdividef(state1, sqrtf(state2) + correction2 * config.eps);
    if (config.weight_decay > 0.f) parameter[index] *= 1.f - config.learning_rate * config.weight_decay;
}

__global__ void adamw_uint8_kernel(
    float *parameter, const float *gradient, uint8_t *state1_codes, uint8_t *state2_codes,
    float *absmax1, float *absmax2, int elements, AdamW8bitCudaConfig config,
    const float *map1, const float *map2
) {
    const int thread = threadIdx.x;
    const int index = blockIdx.x * kThreads + thread;
    const bool valid = index < elements;
    const float grad = valid ? gradient[index] : 0.f;

    __shared__ float reduced1[kThreads];
    __shared__ float reduced2[kThreads];
    float state1 = 0.f;
    float state2 = 0.f;

    if (valid && isfinite(grad)) {
        state1 = map1[state1_codes[index]] * absmax1[blockIdx.x];
        state2 = map2[state2_codes[index]] * absmax2[blockIdx.x];
        state1 = state1 * config.beta1 + grad * (1.f - config.beta1);
        state2 = state2 * config.beta2 + grad * grad * (1.f - config.beta2);
    }

    reduced1[thread] = fabsf(state1);
    reduced2[thread] = fabsf(state2);
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride >>= 1) {
        if (thread < stride) {
            reduced1[thread] = fmaxf(reduced1[thread], reduced1[thread + stride]);
            reduced2[thread] = fmaxf(reduced2[thread], reduced2[thread + stride]);
        }
        __syncthreads();
    }

    const float scale1 = reduced1[0];
    const float scale2 = reduced2[0];
    if (thread == 0) {
        absmax1[blockIdx.x] = scale1;
        absmax2[blockIdx.x] = scale2;
    }
    __syncthreads();

    if (!valid) return;
    if (!isfinite(grad)) {
        state1_codes[index] = quantize_midpoint(0.f, map1);
        state2_codes[index] = quantize_midpoint(0.f, map2);
        return;
    }

    const float correction1 = 1.f - __powf(config.beta1, config.step);
    const float correction2 = sqrtf(1.f - __powf(config.beta2, config.step));
    parameter[index] += __fdividef(-config.learning_rate * correction2, correction1) *
                        __fdividef(state1, sqrtf(state2) + correction2 * config.eps);
    if (config.weight_decay > 0.f) parameter[index] *= 1.f - config.learning_rate * config.weight_decay;

    uint8_t code1 = quantize_midpoint(__fdividef(state1, scale1), map1);
    const uint8_t code2 = quantize_midpoint(__fdividef(state2, scale2), map2);
    if (signbit(map1[code1]) != signbit(state1)) code1 = uint8_t(int(code1) + (state1 > 0.f ? 1 : -1));
    state1_codes[index] = code1;
    state2_codes[index] = code2;
}

bool cuda_ok(cudaError_t error) {
    return error == cudaSuccess;
}

struct DeviceBuffer {
    void *pointer = nullptr;

    ~DeviceBuffer() {
        if (pointer) cudaFree(pointer);
    }

    bool allocate(size_t bytes) {
        return cuda_ok(cudaMalloc(&pointer, bytes));
    }
};

bool compare_floats(
    const Array &expected, const std::vector<float> &actual, float tolerance,
    const std::string &name, CudaReplayReport *report, std::ostringstream &out
) {
    size_t mismatches = 0;
    double maximum = 0;
    if (expected.dtype != 1 || expected.bytes.size() != actual.size() * sizeof(float)) {
        mismatches = std::max<size_t>(1, actual.size());
    } else {
        for (size_t index = 0; index < actual.size(); ++index) {
            const float wanted = f32(expected, index);
            const float observed = actual[index];
            if (!std::isfinite(wanted) || !std::isfinite(observed)) {
                ++mismatches;
                maximum = std::numeric_limits<double>::infinity();
                continue;
            }
            const double difference = std::abs(double(wanted) - observed);
            maximum = std::max(maximum, difference);
            if (difference > tolerance) ++mismatches;
        }
    }
    report->max_abs = std::max(report->max_abs, maximum);
    report->mismatches += mismatches;
    out << name << ": max_abs=" << maximum << " mismatches=" << mismatches << " tol=" << tolerance << '\n';
    return mismatches == 0;
}

bool compare_codes(
    const Array &expected, const std::vector<uint8_t> &actual,
    const std::string &name, CudaReplayReport *report, std::ostringstream &out
) {
    size_t mismatches = 0;
    if (expected.dtype != 2 || expected.bytes.size() != actual.size()) {
        mismatches = std::max<size_t>(1, actual.size());
    } else {
        for (size_t index = 0; index < actual.size(); ++index) {
            if (expected.bytes[index] != actual[index]) ++mismatches;
        }
    }
    report->mismatches += mismatches;
    out << name << ": mismatches=" << mismatches << " exact\n";
    return mismatches == 0;
}

template <typename T>
bool copy_device_to_host(std::vector<T> &output, const T *input) {
    return cuda_ok(cudaMemcpy(output.data(), input, output.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

bool fixture_has_only_finite_floats(const Fixture &fixture) {
    for (const Array &array : fixture.arrays) {
        if (array.dtype != 1) continue;
        if (array.bytes.size() % sizeof(float)) return false;
        for (size_t index = 0; index < array.bytes.size() / sizeof(float); ++index) {
            if (!std::isfinite(f32(array, index))) return false;
        }
    }
    return true;
}

} // namespace

bool adamw8bit_build_dynamic_map(bool signed_map, float output[256]) {
    if (!output) return false;
    std::vector<float> values;
    for (int exponent = 0; exponent < 7; ++exponent) {
        const int count = signed_map ? (1 << exponent) + 1 : (1 << (exponent + 1)) + 1;
        for (int index = 0; index < count - 1; ++index) {
            const float low = .1f + .9f * index / float(count - 1);
            const float high = .1f + .9f * (index + 1) / float(count - 1);
            const float value = powf(10.f, float(-6 + exponent)) * (low + high) * .5f;
            values.push_back(value);
            if (signed_map) values.push_back(-value);
        }
    }
    values.push_back(0.f);
    values.push_back(1.f);
    while (values.size() < 256) values.push_back(0.f);
    std::sort(values.begin(), values.end());
    if (values.size() != 256) return false;
    for (int index = 0; index < 256; ++index) output[index] = values[index];
    return true;
}

bool adamw8bit_initialize_reference_maps(
    const float qmap1[256], const float qmap2[256], float output1[256], float output2[256]
) {
    if (!qmap1 || !qmap2 || !output1 || !output2) return false;
    for (int index = 0; index < 256; ++index) {
        if (!std::isfinite(qmap1[index]) || !std::isfinite(qmap2[index]) ||
            (index && (qmap1[index] < qmap1[index - 1] || qmap2[index] < qmap2[index - 1]))) {
            return false;
        }
        output1[index] = qmap1[index];
        output2[index] = qmap2[index];
    }
    return true;
}

bool adamw8bit_initialize_default_maps(float output1[256], float output2[256]) {
    if (!output1 || !output2) return false;
    adamw8bit_default_maps_from_bits(output1, output2);
    return adamw8bit_initialize_reference_maps(output1, output2, output1, output2);
}

bool adamw8bit_cuda_validate(const AdamW8bitCudaConfig &config, const AdamW8bitCudaState &state) {
    if (!config.elements || config.elements > size_t(INT_MAX) || config.step <= 0 ||
        !state.parameter || !state.gradient || !std::isfinite(config.beta1) ||
        !std::isfinite(config.beta2) || !std::isfinite(config.eps) ||
        !std::isfinite(config.learning_rate) || !std::isfinite(config.weight_decay) ||
        config.beta1 < 0 || config.beta1 >= 1 || config.beta2 < 0 || config.beta2 >= 1 ||
        config.eps <= 0 || config.learning_rate < 0) {
        return false;
    }
    return config.elements < 4096
        ? state.state1_fp32 && state.state2_fp32
        : state.state1_u8 && state.state2_u8 && state.absmax1 && state.absmax2 && state.qmap1 && state.qmap2;
}

CudaOptimizerStatus adamw8bit_step_cuda(
    const AdamW8bitCudaConfig &config, const AdamW8bitCudaState &state, cudaStream_t stream
) {
    if (!adamw8bit_cuda_validate(config, state)) return CudaOptimizerStatus::invalid_argument;
    const int elements = static_cast<int>(config.elements);
    const int blocks = 1 + (elements - 1) / kThreads; // avoids n + 255 overflow at INT_MAX
    if (elements < 4096) {
        adamw_fp32_kernel<<<blocks, kThreads, 0, stream>>>(
            state.parameter, state.gradient, state.state1_fp32, state.state2_fp32, elements, config
        );
    } else {
        adamw_uint8_kernel<<<blocks, kThreads, 0, stream>>>(
            state.parameter, state.gradient, state.state1_u8, state.state2_u8,
            state.absmax1, state.absmax2, elements, config, state.qmap1, state.qmap2
        );
    }
    return cuda_ok(cudaPeekAtLastError()) ? CudaOptimizerStatus::success : CudaOptimizerStatus::cuda_error;
}

const char *adamw8bit_cuda_status_string(CudaOptimizerStatus status) {
    switch (status) {
    case CudaOptimizerStatus::success: return "success";
    case CudaOptimizerStatus::invalid_argument: return "invalid_argument";
    case CudaOptimizerStatus::cuda_error: return "cuda_error";
    case CudaOptimizerStatus::unsupported: return "unsupported";
    }
    return "unknown";
}

bool replay_cuda(const Fixture &fixture, float tolerance, CudaReplayReport *report) {
    if (!report || tolerance < 0) return false;
    *report = {};
    if (!fixture_has_only_finite_floats(fixture)) {
        report->detail = "fixture contains non-finite floating-point data\n";
        report->mismatches = 1;
        report->max_abs = std::numeric_limits<double>::infinity();
        return false;
    }

    const Array &reference_map1 = get(fixture, "step_1_large_qmap1");
    const Array &reference_map2 = get(fixture, "step_1_large_qmap2");
    if (reference_map1.dtype != 1 || reference_map2.dtype != 1 ||
        reference_map1.bytes.size() != 1024 || reference_map2.bytes.size() != 1024) {
        return false;
    }
    float map1[256], map2[256], default_map1[256], default_map2[256];
    for (int index = 0; index < 256; ++index) {
        map1[index] = f32(reference_map1, index);
        map2[index] = f32(reference_map2, index);
    }
    if (!adamw8bit_initialize_reference_maps(map1, map2, map1, map2)) return false;
    if (!adamw8bit_initialize_default_maps(default_map1, default_map2)) return false;

    DeviceBuffer device_map1, device_map2;
    if (!device_map1.allocate(sizeof(map1)) || !device_map2.allocate(sizeof(map2)) ||
        !cuda_ok(cudaMemcpy(device_map1.pointer, map1, sizeof(map1), cudaMemcpyHostToDevice)) ||
        !cuda_ok(cudaMemcpy(device_map2.pointer, map2, sizeof(map2), cudaMemcpyHostToDevice))) {
        return false;
    }

    bool passed = true;
    std::ostringstream output;
    output << "codebooks: default constants and fixture-derived maps directly checked\n";
    for (const char *side : {"small", "large"}) {
        const std::string suffix(side);
        const size_t elements = get(fixture, "parameter_" + suffix + "_initial").bytes.size() / sizeof(float);
        const size_t blocks = (elements + kThreads - 1) / kThreads;
        DeviceBuffer parameter, gradient, state1, state2, absmax1, absmax2;
        if (!parameter.allocate(elements * sizeof(float)) || !gradient.allocate(elements * sizeof(float)) ||
            !cuda_ok(cudaMemcpy(parameter.pointer, get(fixture, "parameter_" + suffix + "_initial").bytes.data(),
                                elements * sizeof(float), cudaMemcpyHostToDevice))) {
            return false;
        }
        if (elements < 4096) {
            if (!state1.allocate(elements * sizeof(float)) || !state2.allocate(elements * sizeof(float)) ||
                !cuda_ok(cudaMemset(state1.pointer, 0, elements * sizeof(float))) ||
                !cuda_ok(cudaMemset(state2.pointer, 0, elements * sizeof(float)))) return false;
        } else {
            if (!state1.allocate(elements) || !state2.allocate(elements) ||
                !absmax1.allocate(blocks * sizeof(float)) || !absmax2.allocate(blocks * sizeof(float)) ||
                !cuda_ok(cudaMemset(state1.pointer, 0, elements)) || !cuda_ok(cudaMemset(state2.pointer, 0, elements)) ||
                !cuda_ok(cudaMemset(absmax1.pointer, 0, blocks * sizeof(float))) ||
                !cuda_ok(cudaMemset(absmax2.pointer, 0, blocks * sizeof(float)))) return false;
        }

        for (int step = 1; step <= 5; ++step) {
            if (!cuda_ok(cudaMemcpy(
                    gradient.pointer, get(fixture, "gradient_" + suffix + "_step_" + std::to_string(step - 1)).bytes.data(),
                    elements * sizeof(float), cudaMemcpyHostToDevice
                ))) return false;
            AdamW8bitCudaConfig config;
            config.elements = elements;
            config.step = step;
            AdamW8bitCudaState state;
            state.parameter = static_cast<float *>(parameter.pointer);
            state.gradient = static_cast<const float *>(gradient.pointer);
            state.qmap1 = static_cast<const float *>(device_map1.pointer);
            state.qmap2 = static_cast<const float *>(device_map2.pointer);
            if (elements < 4096) {
                state.state1_fp32 = static_cast<float *>(state1.pointer);
                state.state2_fp32 = static_cast<float *>(state2.pointer);
            } else {
                state.state1_u8 = static_cast<uint8_t *>(state1.pointer);
                state.state2_u8 = static_cast<uint8_t *>(state2.pointer);
                state.absmax1 = static_cast<float *>(absmax1.pointer);
                state.absmax2 = static_cast<float *>(absmax2.pointer);
            }
            if (adamw8bit_step_cuda(config, state, 0) != CudaOptimizerStatus::success ||
                !cuda_ok(cudaDeviceSynchronize())) return false;

            std::vector<float> parameters(elements);
            if (!copy_device_to_host(parameters, static_cast<float *>(parameter.pointer))) return false;
            passed &= compare_floats(
                get(fixture, "parameter_" + suffix + "_after_step_" + std::to_string(step)), parameters,
                tolerance, "parameter_" + suffix + "_after_step_" + std::to_string(step), report, output
            );
            const std::string prefix = "step_" + std::to_string(step) + "_" + suffix;
            if (elements < 4096) {
                std::vector<float> moment1(elements), moment2(elements);
                if (!copy_device_to_host(moment1, static_cast<float *>(state1.pointer)) ||
                    !copy_device_to_host(moment2, static_cast<float *>(state2.pointer))) return false;
                passed &= compare_floats(get(fixture, prefix + "_state1"), moment1, tolerance, prefix + "_state1", report, output);
                passed &= compare_floats(get(fixture, prefix + "_state2"), moment2, tolerance, prefix + "_state2", report, output);
            } else {
                std::vector<uint8_t> codes1(elements), codes2(elements);
                std::vector<float> scales1(blocks), scales2(blocks), copied_map1(256), copied_map2(256);
                if (!copy_device_to_host(codes1, static_cast<uint8_t *>(state1.pointer)) ||
                    !copy_device_to_host(codes2, static_cast<uint8_t *>(state2.pointer)) ||
                    !copy_device_to_host(scales1, static_cast<float *>(absmax1.pointer)) ||
                    !copy_device_to_host(scales2, static_cast<float *>(absmax2.pointer)) ||
                    !copy_device_to_host(copied_map1, static_cast<float *>(device_map1.pointer)) ||
                    !copy_device_to_host(copied_map2, static_cast<float *>(device_map2.pointer))) return false;
                passed &= compare_codes(get(fixture, prefix + "_state1"), codes1, prefix + "_state1", report, output);
                passed &= compare_codes(get(fixture, prefix + "_state2"), codes2, prefix + "_state2", report, output);
                passed &= compare_floats(get(fixture, prefix + "_absmax1"), scales1, tolerance, prefix + "_absmax1", report, output);
                passed &= compare_floats(get(fixture, prefix + "_absmax2"), scales2, tolerance, prefix + "_absmax2", report, output);
                passed &= compare_floats(get(fixture, prefix + "_qmap1"), copied_map1, 0, prefix + "_qmap1", report, output);
                passed &= compare_floats(get(fixture, prefix + "_qmap2"), copied_map2, 0, prefix + "_qmap2", report, output);
                passed &= compare_floats(get(fixture, prefix + "_qmap1"), std::vector<float>(default_map1, default_map1 + 256), 0, prefix + "_default_qmap1", report, output);
                passed &= compare_floats(get(fixture, prefix + "_qmap2"), std::vector<float>(default_map2, default_map2 + 256), 0, prefix + "_default_qmap2", report, output);
            }
        }
    }
    report->detail = output.str();
    return passed;
}

} // namespace aitk_adamw8bit
