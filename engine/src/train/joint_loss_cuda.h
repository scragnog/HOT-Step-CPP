#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace yue2_joint_loss_cuda {

enum class Status : int { success = 0, invalid_argument = 1, cuda_error = 2 };

// adapted/base/targets/gradient are chunk-local caller buffers. per_ce/per_kl
// are total_positions-sized caller buffers; callers may submit several chunks
// before reduce(). The caller must prevalidate target ids against vocab.
Status forward_chunk(const float *adapted, const float *base, const uint32_t *targets, float *gradient,
                     float *per_ce, float *per_kl, std::size_t position_offset, std::size_t position_count,
                     std::size_t total_positions, std::size_t vocab, float kl_weight, cudaStream_t stream);
Status reduce(const float *per_ce, const float *per_kl, std::size_t total_positions,
              float *ce, float *kl, cudaStream_t stream);
const char *status_string(Status status);

} // namespace yue2_joint_loss_cuda
