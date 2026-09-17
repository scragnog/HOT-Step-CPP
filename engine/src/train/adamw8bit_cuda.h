#pragma once

// CUDA translation of bitsandbytes 0.49.2 csrc/kernels.cu.
// Derived from f0e6ca31b32c4744a9cee4e31610b25796cbf778; see the full MIT notice
// in adamw8bit_cuda_LICENSE.txt.

#include "adamw8bit_cpu.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace aitk_adamw8bit {

enum class CudaOptimizerStatus : int {
    success,
    invalid_argument,
    cuda_error,
    unsupported,
};

struct AdamW8bitCudaConfig {
    size_t elements = 0;
    int step = 0;
    float beta1 = .9f;
    float beta2 = .999f;
    float eps = 1e-6f;
    float learning_rate = 1e-4f;
    float weight_decay = 1e-4f;
};

// All buffers are caller-owned device allocations. Below 4096 elements the
// FP32 moments are required; from 4096 the uint8 state, absmax and maps are
// required. This operation allocates nothing and never device-synchronizes.
struct AdamW8bitCudaState {
    float *parameter = nullptr;
    const float *gradient = nullptr;
    float *state1_fp32 = nullptr;
    float *state2_fp32 = nullptr;
    uint8_t *state1_u8 = nullptr;
    uint8_t *state2_u8 = nullptr;
    float *absmax1 = nullptr;
    float *absmax2 = nullptr;
    const float *qmap1 = nullptr;
    const float *qmap2 = nullptr;
};

bool adamw8bit_cuda_validate(const AdamW8bitCudaConfig &config, const AdamW8bitCudaState &state);
CudaOptimizerStatus adamw8bit_step_cuda(
    const AdamW8bitCudaConfig &config, const AdamW8bitCudaState &state, cudaStream_t stream
);
const char *adamw8bit_cuda_status_string(CudaOptimizerStatus status);

// Diagnostic translation of create_dynamic_map. It is not an exact fixture
// initializer. For exact parity, explicitly supply reference-derived qmaps.
bool adamw8bit_build_dynamic_map(bool signed_map, float output[256]);

// Initializes the pinned bitsandbytes 0.49.2 maps with no fixture-file dependency.
bool adamw8bit_initialize_default_maps(float output1[256], float output2[256]);

// Validates sorted, finite reference maps and copies them to caller storage.
// This function does not construct or prove the codebooks independently.
bool adamw8bit_initialize_reference_maps(
    const float qmap1[256], const float qmap2[256], float output1[256], float output2[256]
);

struct CudaReplayReport {
    double max_abs = 0;
    size_t mismatches = 0;
    std::string detail;
};

bool replay_cuda(const Fixture &fixture, float atol, CudaReplayReport *report);

} // namespace aitk_adamw8bit
