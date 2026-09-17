#pragma once
#include "yue2-aitk-backend.h"

// Chunked full-vocabulary AR CE + KL head helper for the native AITK trainer.
// This is a staging seam only. It assumes yue2-aitk-model.h and
// joint_loss_cuda.h are installed beside it in engine/src/train/.
//
// Each chunk owns temporary GGML metadata/buffer lifetime. The frozen ConvRot
// head tensors are borrowed from Yue2AitkConvRotLinear; they are never copied
// or marked trainable. The CUDA loss kernel owns the adapted-logit gradient,
// including kl_weight, so callers must not multiply the returned hidden
// gradient by the KL weight again.

#include "yue2-aitk-model.h"
#include "joint_loss_cuda.h"
#include "../../ggml/include/ggml-alloc.h"
#include "../../ggml/include/ggml-cuda.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>

namespace yue2_aitk_head_loss {

constexpr std::size_t kArChunk = 128;
constexpr std::size_t kVocab = 184704;

enum class Status : int {
    success = 0,
    invalid_argument,
    unsupported_backend,
    allocation_failure,
    graph_failure,
    cuda_failure,
};

struct Request {
    ggml_backend_t backend = nullptr; // borrowed; must be CUDA
    const Yue2AitkConvRotLinear * head = nullptr; // borrowed frozen lm_head
    const float * adapted_hidden = nullptr; // host [positions, hidden], BF16-valued
    const float * base_hidden = nullptr;    // host [positions, hidden], BF16-valued
    const uint32_t * targets = nullptr;     // host [positions], prevalidated
    std::size_t positions = 0;
    std::size_t hidden = 0;
    float kl_weight = 0.2f;
    uint16_t * adapted_hidden_grad_bf16 = nullptr; // host [positions, hidden]
    float * ce_sum = nullptr; // raw sum over target positions
    float * kl_sum = nullptr; // raw sum over target positions
};

inline const char * status_string(Status status) {
    switch (status) {
    case Status::success: return "success";
    case Status::invalid_argument: return "invalid_argument";
    case Status::unsupported_backend: return "unsupported_backend";
    case Status::allocation_failure: return "allocation_failure";
    case Status::graph_failure: return "graph_failure";
    case Status::cuda_failure: return "cuda_failure";
    }
    return "unknown";
}

inline Status fail(Status status, std::string * error, const char * message) {
    if (error) *error = message;
    return status;
}

struct Temporary {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Temporary() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};

inline uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest, ties to even, matching the value boundary used by
    // the reference BF16 model rather than truncating the hidden gradient.
    const uint32_t round = 0x7fffU + ((bits >> 16) & 1U);
    bits += round;
    return static_cast<uint16_t>(bits >> 16);
}

inline bool validate_nodes(ggml_backend_t backend, const ggml_cgraph * graph, std::string * error) {
    if (!graph) return false;
    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(graph)); ++i) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(graph), i);
        if (node && node->op != GGML_OP_NONE && !ggml_backend_supports_op(backend, node)) {
            if (error) *error = "CUDA backend does not support a head-loss graph node";
            return false;
        }
    }
    return true;
}

struct CudaDeviceGuard {
    int previous = -1;
    bool switched = false;
    bool valid = false;

    explicit CudaDeviceGuard(ggml_backend_t backend) {
        if (!backend || !yue2_aitk_is_cuda(backend)) return;
        const ggml_backend_dev_t device = ggml_backend_get_device(backend);
        const char * name = device ? ggml_backend_dev_name(device) : nullptr;
        if (!name || std::strncmp(name, "CUDA", 4) != 0) return;
        char * end = nullptr;
        const long ordinal = std::strtol(name + 4, &end, 10);
        if (end == name + 4 || *end != '\0' || ordinal < 0 || ordinal > std::numeric_limits<int>::max()) return;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != static_cast<int>(ordinal)) {
            if (cudaSetDevice(static_cast<int>(ordinal)) != cudaSuccess) return;
            switched = true;
        }
        valid = true;
    }

    ~CudaDeviceGuard() {
        if (switched) (void) cudaSetDevice(previous);
    }
};

inline Status compute(const Request & request, std::string * error = nullptr) {
    if (!request.backend || !request.head || !request.adapted_hidden || !request.base_hidden ||
        !request.targets || !request.adapted_hidden_grad_bf16 || !request.ce_sum || !request.kl_sum ||
        !request.positions || !request.hidden || request.hidden > std::numeric_limits<int64_t>::max() ||
        request.positions > std::numeric_limits<int>::max() || !std::isfinite(request.kl_weight) ||
        request.kl_weight < 0.0f || request.head->rows != static_cast<int64_t>(kVocab) ||
        request.head->cols != static_cast<int64_t>(request.hidden) || !request.head->weight_i8 ||
        !request.head->scales_f32 || !request.head->weight_i8->data || !request.head->scales_f32->data ||
        request.head->weight_i8->flags & GGML_TENSOR_FLAG_PARAM ||
        request.head->scales_f32->flags & GGML_TENSOR_FLAG_PARAM) {
        return fail(Status::invalid_argument, error, "invalid head-loss request");
    }
    if (!yue2_aitk_is_cuda(request.backend))
        return fail(Status::unsupported_backend, error, "head loss requires the CUDA GGML backend");
    if (request.positions > std::numeric_limits<std::size_t>::max() / request.hidden ||
        request.positions > std::numeric_limits<std::size_t>::max() / kVocab)
        return fail(Status::invalid_argument, error, "head-loss request size overflows");
    for (std::size_t i = 0; i < request.positions; ++i) {
        if (request.targets[i] >= kVocab) return fail(Status::invalid_argument, error, "head-loss target is outside vocabulary");
        for (std::size_t j = 0; j < request.hidden; ++j) {
            if (!std::isfinite(request.adapted_hidden[i * request.hidden + j]) ||
                !std::isfinite(request.base_hidden[i * request.hidden + j]))
                return fail(Status::invalid_argument, error, "head-loss hidden input is nonfinite");
        }
    }
    CudaDeviceGuard device_guard(request.backend);
    if (!device_guard.valid)
        return fail(Status::unsupported_backend, error, "unable to select the GGML CUDA backend device");

    std::vector<float> chunk_grad(kArChunk * request.hidden);
    std::vector<float> per_ce(request.positions), per_kl(request.positions);
    *request.ce_sum = 0.0f;
    *request.kl_sum = 0.0f;

    for (std::size_t offset = 0; offset < request.positions; offset += kArChunk) {
        const std::size_t count = (request.positions - offset < kArChunk) ?
            request.positions - offset : kArChunk;
        Temporary tmp;
        ggml_init_params params{};
        params.mem_size = 2 * ggml_graph_overhead_custom(64, false) +
                          ggml_tensor_overhead() * 64 + 4096;
        params.no_alloc = true;
        tmp.ctx = ggml_init(params);
        if (!tmp.ctx) return fail(Status::allocation_failure, error, "head-loss GGML context allocation failed");

        ggml_tensor * adapted = ggml_new_tensor_2d(tmp.ctx, GGML_TYPE_F32, request.hidden, count);
        ggml_tensor * base = ggml_new_tensor_2d(tmp.ctx, GGML_TYPE_F32, request.hidden, count);
        ggml_tensor * targets = ggml_new_tensor_1d(tmp.ctx, GGML_TYPE_I32, count);
        ggml_tensor * adapted_logits = ggml_convrot8(tmp.ctx, request.head->weight_i8, adapted,
            request.head->scales_f32, nullptr, request.head->rotation, true);
        ggml_tensor * base_logits = ggml_convrot8(tmp.ctx, request.head->weight_i8, base,
            request.head->scales_f32, nullptr, request.head->rotation, true);
        ggml_tensor * logit_grad = ggml_new_tensor_2d(tmp.ctx, GGML_TYPE_F32, kVocab, count);
        ggml_tensor * hidden_grad = ggml_convrot8_back(tmp.ctx, logit_grad, adapted_logits);
        ggml_tensor * per_ce_t = ggml_new_tensor_1d(tmp.ctx, GGML_TYPE_F32, request.positions);
        ggml_tensor * per_kl_t = ggml_new_tensor_1d(tmp.ctx, GGML_TYPE_F32, request.positions);
        if (!adapted || !base || !targets || !adapted_logits || !base_logits || !logit_grad ||
            !hidden_grad || !per_ce_t || !per_kl_t)
            return fail(Status::allocation_failure, error, "head-loss GGML tensor allocation failed");

        ggml_cgraph * forward = ggml_new_graph_custom(tmp.ctx, 64, false);
        ggml_cgraph * backward = ggml_new_graph_custom(tmp.ctx, 64, false);
        if (!forward || !backward) return fail(Status::graph_failure, error, "head-loss graph allocation failed");
        ggml_build_forward_expand(forward, adapted_logits);
        ggml_build_forward_expand(forward, base_logits);
        ggml_build_forward_expand(backward, hidden_grad);
        if (!validate_nodes(request.backend, forward, error)) return Status::graph_failure;
        if (!validate_nodes(request.backend, backward, error)) return Status::graph_failure;
        tmp.buffer = ggml_backend_alloc_ctx_tensors(tmp.ctx, request.backend);
        if (!tmp.buffer) return fail(Status::allocation_failure, error, "head-loss backend buffer allocation failed");
        ggml_backend_tensor_set(adapted, request.adapted_hidden + offset * request.hidden, 0, ggml_nbytes(adapted));
        ggml_backend_tensor_set(base, request.base_hidden + offset * request.hidden, 0, ggml_nbytes(base));
        ggml_backend_tensor_set(targets, request.targets + offset, 0, ggml_nbytes(targets));
        if (ggml_backend_graph_compute(request.backend, forward) != GGML_STATUS_SUCCESS) {
            return fail(Status::graph_failure, error, "head-loss forward graph failed");
        }
        ggml_backend_synchronize(request.backend);

        auto * adapted_ptr = static_cast<const float *>(adapted_logits->data);
        auto * base_ptr = static_cast<const float *>(base_logits->data);
        auto * target_ptr = static_cast<const uint32_t *>(targets->data);
        auto * gradient_ptr = static_cast<float *>(logit_grad->data);
        auto * ce_ptr = static_cast<float *>(per_ce_t->data);
        auto * kl_ptr = static_cast<float *>(per_kl_t->data);
        if (!adapted_ptr || !base_ptr || !target_ptr || !gradient_ptr || !ce_ptr || !kl_ptr)
            return fail(Status::graph_failure, error, "head-loss CUDA tensor data is unavailable");
        const auto loss_status = yue2_joint_loss_cuda::forward_chunk(
            adapted_ptr, base_ptr, target_ptr, gradient_ptr, ce_ptr, kl_ptr,
            offset, count, request.positions, kVocab, request.kl_weight, nullptr);
        if (loss_status != yue2_joint_loss_cuda::Status::success)
            return fail(Status::cuda_failure, error, yue2_joint_loss_cuda::status_string(loss_status));
        if (cudaStreamSynchronize(nullptr) != cudaSuccess)
            return fail(Status::cuda_failure, error, "head-loss CUDA stream synchronization failed");

        ggml_backend_tensor_get(per_ce_t, per_ce.data() + offset, offset * sizeof(float), count * sizeof(float));
        ggml_backend_tensor_get(per_kl_t, per_kl.data() + offset, offset * sizeof(float), count * sizeof(float));
        *request.ce_sum += [&] { float value = 0.0f; for (std::size_t i = 0; i < count; ++i) value += per_ce[offset + i]; return value; }();
        *request.kl_sum += [&] { float value = 0.0f; for (std::size_t i = 0; i < count; ++i) value += per_kl[offset + i]; return value; }();

        if (ggml_backend_graph_compute(request.backend, backward) != GGML_STATUS_SUCCESS) {
            return fail(Status::graph_failure, error, "head-loss backward graph failed");
        }
        ggml_backend_synchronize(request.backend);
        ggml_backend_tensor_get(hidden_grad, chunk_grad.data(), 0, count * request.hidden * sizeof(float));
        for (std::size_t i = 0; i < count * request.hidden; ++i)
            request.adapted_hidden_grad_bf16[offset * request.hidden + i] = f32_to_bf16(chunk_grad[i]);
    }
    return Status::success;
}

} // namespace yue2_aitk_head_loss
