#pragma once
#include "yue2-aitk-backend.h"

// Staged YuE2 optimizer ownership layer. Copy beside the native training
// sources before integration; this file intentionally does not edit engine/.
#include "adamw8bit_cuda.h"
#include "../../ggml/include/ggml-backend.h"
#include "../../ggml/include/ggml.h"
#include "../../ggml/include/ggml-cuda.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yue2_aitk {

struct ParameterSpec {
    std::string name;
    ggml_tensor * parameter = nullptr;
    ggml_tensor * gradient = nullptr;
};

struct StepConfig {
    float learning_rate = 1e-4f;
    float beta1 = .9f;
    float beta2 = .999f;
    float eps = 1e-6f;
    float weight_decay = 1e-4f;
};

class CudaBuffer {
public:
    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer &) = delete;
    CudaBuffer & operator=(const CudaBuffer &) = delete;
    CudaBuffer(CudaBuffer && other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) { other.ptr_ = nullptr; other.bytes_ = 0; }
    CudaBuffer & operator=(CudaBuffer && other) noexcept { if (this != &other) { reset(); ptr_ = other.ptr_; bytes_ = other.bytes_; other.ptr_ = nullptr; other.bytes_ = 0; } return *this; }
    ~CudaBuffer() { reset(); }

    void allocate(size_t bytes) { reset(); bytes_ = bytes; if (bytes && cudaMalloc(&ptr_, bytes) != cudaSuccess) { bytes_ = 0; throw std::runtime_error("YuE2 optimizer cudaMalloc failed"); } }
    void zero(cudaStream_t stream) { if (ptr_ && cudaMemsetAsync(ptr_, 0, bytes_, stream) != cudaSuccess) throw std::runtime_error("YuE2 optimizer cudaMemsetAsync failed"); }
    void * get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    void reset() noexcept { if (ptr_) cudaFree(ptr_); ptr_ = nullptr; bytes_ = 0; }
private:
    void * ptr_ = nullptr;
    size_t bytes_ = 0;
};

struct HostStateSnapshot {
    std::vector<std::string> names;
    std::vector<size_t> elements;
    int step = 0;
    std::vector<std::vector<float>> parameters;
    std::vector<std::vector<float>> state1_fp32;
    std::vector<std::vector<float>> state2_fp32;
    std::vector<std::vector<uint8_t>> state1_u8;
    std::vector<std::vector<uint8_t>> state2_u8;
    std::vector<std::vector<float>> absmax1;
    std::vector<std::vector<float>> absmax2;
    // State slots 3 and 4 are carried only by the version-2 (LmOptim) resume
    // format: Prodigy's s and x0 buffers. The AdamW8bit optimizer never fills
    // them, and the v1 writer refuses snapshots that do.
    std::vector<std::vector<float>> state3_fp32;
    std::vector<std::vector<float>> state4_fp32;
    std::vector<std::vector<uint8_t>> state3_u8;
    std::vector<std::vector<uint8_t>> state4_u8;
    std::vector<std::vector<float>> absmax3;
    std::vector<std::vector<float>> absmax4;
};

class DeviceGuard {
public:
    explicit DeviceGuard(int target) {
        if (cudaGetDevice(&previous_) != cudaSuccess || cudaSetDevice(target) != cudaSuccess) throw std::runtime_error("YuE2 optimizer CUDA device selection failed");
        active_ = true;
    }
    DeviceGuard(const DeviceGuard &) = delete;
    ~DeviceGuard() { if (active_) cudaSetDevice(previous_); }
private:
    int previous_ = 0;
    bool active_ = false;
};

class Optimizer {
public:
    Optimizer(ggml_backend_t backend, int device_index, std::vector<ParameterSpec> specs)
        : backend_(backend), device_index_(device_index), specs_(std::move(specs)) {
        validate_specs(specs_);
        if (!backend_) throw std::invalid_argument("YuE2 optimizer requires a GGML backend");
        validate_backend_specs();
        DeviceGuard device(device_index_);
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) throw std::runtime_error("YuE2 optimizer cudaStreamCreate failed");
        try { slots_.reserve(specs_.size()); for (const auto & spec : specs_) slots_.emplace_back(make_slot(spec)); }
        catch (...) { cudaStreamDestroy(stream_); stream_ = nullptr; throw; }
    }
    Optimizer(const Optimizer &) = delete;
    Optimizer & operator=(const Optimizer &) = delete;
    ~Optimizer() {
        int previous = 0;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        if (cudaSetDevice(device_index_) == cudaSuccess) {
            slots_.clear();
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (have_previous) cudaSetDevice(previous);
    }

    int step() const { return step_; }
    cudaStream_t stream() const { return stream_; }

    // Validates every parameter and gradient before changing any state. The
    // caller must synchronize GGML's backend before entry; this wrapper does
    // not impose a backend-wide synchronization. GGML owns tensor storage and
    // the native step sees those device pointers directly.
    void step_once(const StepConfig & cfg) {
        DeviceGuard device(device_index_);
        validate_specs(specs_);
        validate_config(cfg);
        validate_backend_specs();
        if (step_ >= INT_MAX) throw std::overflow_error("YuE2 optimizer step counter exhausted");
        std::vector<aitk_adamw8bit::AdamW8bitCudaState> states;
        states.reserve(slots_.size());
        for (size_t i = 0; i < specs_.size(); ++i) states.push_back(make_state(i));
        std::vector<aitk_adamw8bit::AdamW8bitCudaConfig> configs;
        configs.reserve(slots_.size());
        for (size_t i = 0; i < states.size(); ++i) {
            configs.push_back(make_config(slots_[i].elements, cfg, step_ + 1));
            if (!aitk_adamw8bit::adamw8bit_cuda_validate(configs.back(), states[i])) throw std::invalid_argument("YuE2 optimizer native state validation failed");
        }
        for (size_t i = 0; i < states.size(); ++i) {
            const auto status = aitk_adamw8bit::adamw8bit_step_cuda(configs[i], states[i], stream_);
            if (status != aitk_adamw8bit::CudaOptimizerStatus::success) throw std::runtime_error(aitk_adamw8bit::adamw8bit_cuda_status_string(status));
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) throw std::runtime_error("YuE2 optimizer step CUDA synchronization failed");
        ++step_;
    }

    HostStateSnapshot capture() const {
        DeviceGuard device(device_index_);
        validate_specs(specs_);
        validate_backend_specs();
        ggml_backend_synchronize(backend_);
        HostStateSnapshot out; out.step = step_;
        out.parameters.resize(slots_.size()); out.state1_fp32.resize(slots_.size()); out.state2_fp32.resize(slots_.size());
        out.state1_u8.resize(slots_.size()); out.state2_u8.resize(slots_.size()); out.absmax1.resize(slots_.size()); out.absmax2.resize(slots_.size());
        out.state3_fp32.resize(slots_.size()); out.state4_fp32.resize(slots_.size()); out.state3_u8.resize(slots_.size()); out.state4_u8.resize(slots_.size()); out.absmax3.resize(slots_.size()); out.absmax4.resize(slots_.size());
        for (size_t i = 0; i < slots_.size(); ++i) {
            const auto & s = slots_[i]; out.names.push_back(specs_[i].name); out.elements.push_back(s.elements);
            out.parameters[i].resize(s.elements); copy_to_host(out.parameters[i], specs_[i].parameter->data);
            if (s.quantized) { out.state1_u8[i].resize(s.elements); out.state2_u8[i].resize(s.elements); out.absmax1[i].resize(s.blocks); out.absmax2[i].resize(s.blocks); copy_to_host(out.state1_u8[i], s.state1.get()); copy_to_host(out.state2_u8[i], s.state2.get()); copy_to_host(out.absmax1[i], s.absmax1.get()); copy_to_host(out.absmax2[i], s.absmax2.get()); }
            else { out.state1_fp32[i].resize(s.elements); out.state2_fp32[i].resize(s.elements); copy_to_host(out.state1_fp32[i], s.state1.get()); copy_to_host(out.state2_fp32[i], s.state2.get()); }
        }
        return out;
    }

    void restore(const HostStateSnapshot & snap) {
        DeviceGuard device(device_index_);
        validate_specs(specs_);
        validate_backend_specs();
        if (snap.step < 0 || snap.step > INT_MAX || snap.names.size() != slots_.size() || snap.elements.size() != slots_.size() || snap.parameters.size() != slots_.size() || snap.state1_fp32.size() != slots_.size() || snap.state2_fp32.size() != slots_.size() || snap.state1_u8.size() != slots_.size() || snap.state2_u8.size() != slots_.size() || snap.absmax1.size() != slots_.size() || snap.absmax2.size() != slots_.size() || snap.state3_fp32.size() != slots_.size() || snap.state4_fp32.size() != slots_.size() || snap.state3_u8.size() != slots_.size() || snap.state4_u8.size() != slots_.size() || snap.absmax3.size() != slots_.size() || snap.absmax4.size() != slots_.size()) throw std::invalid_argument("YuE2 optimizer snapshot metadata mismatch");
        // Complete validation happens before the first device copy. A malformed
        // later slot must never leave an earlier parameter partially restored.
        for (size_t i = 0; i < slots_.size(); ++i) {
            const auto & s = slots_[i];
            if (snap.names[i] != specs_[i].name || snap.elements[i] != s.elements || snap.parameters[i].size() != s.elements) throw std::invalid_argument("YuE2 optimizer snapshot shape/name mismatch");
            if (!snap.state3_fp32[i].empty() || !snap.state4_fp32[i].empty() || !snap.state3_u8[i].empty() || !snap.state4_u8[i].empty() || !snap.absmax3[i].empty() || !snap.absmax4[i].empty()) throw std::invalid_argument("YuE2 optimizer snapshot carries LmOptim state; resume with the matching --optimizer");
            if (s.quantized) {
                if (snap.state1_u8[i].size() != s.elements || snap.state2_u8[i].size() != s.elements || snap.absmax1[i].size() != s.blocks || snap.absmax2[i].size() != s.blocks) throw std::invalid_argument("YuE2 quantized snapshot state mismatch");
                if (!snap.state1_fp32[i].empty() || !snap.state2_fp32[i].empty()) throw std::invalid_argument("YuE2 quantized snapshot has FP32 state");
                validate_float_vector(snap.parameters[i], "parameter");
                validate_nonnegative_vector(snap.absmax1[i], "absmax");
                validate_nonnegative_vector(snap.absmax2[i], "absmax");
            } else {
                if (snap.state1_fp32[i].size() != s.elements || snap.state2_fp32[i].size() != s.elements) throw std::invalid_argument("YuE2 FP32 snapshot state mismatch");
                if (!snap.state1_u8[i].empty() || !snap.state2_u8[i].empty() || !snap.absmax1[i].empty() || !snap.absmax2[i].empty()) throw std::invalid_argument("YuE2 FP32 snapshot has quantized state");
                validate_float_vector(snap.parameters[i], "parameter");
                validate_float_vector(snap.state1_fp32[i], "state");
                validate_nonnegative_vector(snap.state2_fp32[i], "second moment");
            }
        }
        ggml_backend_synchronize(backend_);
        for (size_t i = 0; i < slots_.size(); ++i) { auto & s = slots_[i]; copy_from_device(snap.parameters[i], specs_[i].parameter->data); if (s.quantized) { copy_from_device(snap.state1_u8[i], s.state1.get()); copy_from_device(snap.state2_u8[i], s.state2.get()); copy_from_device(snap.absmax1[i], s.absmax1.get()); copy_from_device(snap.absmax2[i], s.absmax2.get()); } else { copy_from_device(snap.state1_fp32[i], s.state1.get()); copy_from_device(snap.state2_fp32[i], s.state2.get()); } }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) throw std::runtime_error("YuE2 optimizer restore synchronization failed"); step_ = snap.step;
    }

private:
    struct Slot { size_t elements = 0, blocks = 0; bool quantized = false; CudaBuffer state1, state2, absmax1, absmax2, qmap1, qmap2; };
    static void validate_float_vector(const std::vector<float> & values, const char * label) { for (float value : values) if (!std::isfinite(value)) throw std::invalid_argument(std::string("YuE2 optimizer snapshot has non-finite ") + label); }
    static void validate_nonnegative_vector(const std::vector<float> & values, const char * label) { for (float value : values) if (!std::isfinite(value) || value < 0) throw std::invalid_argument(std::string("YuE2 optimizer snapshot has invalid ") + label); }
    void validate_backend_specs() const {
        const ggml_backend_dev_t device = ggml_backend_get_device(backend_);
        if (!yue2_aitk_is_cuda(backend_) || !device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) throw std::invalid_argument("YuE2 optimizer requires a CUDA GPU GGML backend");
        for (const auto & s : specs_) {
            if (!s.parameter->buffer || !s.gradient->buffer) throw std::invalid_argument("YuE2 optimizer tensor has no backend buffer");
            const auto parameter_type = ggml_backend_buffer_get_type(s.parameter->buffer);
            const auto gradient_type = ggml_backend_buffer_get_type(s.gradient->buffer);
            if (!ggml_backend_supports_buft(backend_, parameter_type) || !ggml_backend_supports_buft(backend_, gradient_type) || ggml_backend_buft_get_device(parameter_type) != device || ggml_backend_buft_get_device(gradient_type) != device) throw std::invalid_argument("YuE2 optimizer tensor buffer is not on its CUDA GGML backend");
            cudaPointerAttributes parameter_attributes{}, gradient_attributes{};
            if (cudaPointerGetAttributes(&parameter_attributes, s.parameter->data) != cudaSuccess || cudaPointerGetAttributes(&gradient_attributes, s.gradient->data) != cudaSuccess || parameter_attributes.type != cudaMemoryTypeDevice || gradient_attributes.type != cudaMemoryTypeDevice || parameter_attributes.device != device_index_ || gradient_attributes.device != device_index_) throw std::invalid_argument("YuE2 optimizer tensor pointer is not on the requested CUDA device");
        }
    }
    static void validate_specs(const std::vector<ParameterSpec> & specs) { if (specs.empty()) throw std::invalid_argument("YuE2 optimizer requires parameters"); std::unordered_set<std::string> names; std::unordered_set<const void *> parameters; for (const auto & s : specs) { if (s.name.empty() || !names.insert(s.name).second || !s.parameter || !s.gradient || !s.parameter->data || !s.gradient->data || !parameters.insert(s.parameter->data).second || s.parameter->type != GGML_TYPE_F32 || s.gradient->type != GGML_TYPE_F32 || !ggml_is_contiguous(s.parameter) || !ggml_is_contiguous(s.gradient) || ggml_nelements(s.parameter) != ggml_nelements(s.gradient)) throw std::invalid_argument("invalid YuE2 optimizer parameter specification"); for (int d = 0; d < GGML_MAX_DIMS; ++d) if (s.parameter->ne[d] != s.gradient->ne[d]) throw std::invalid_argument("YuE2 optimizer parameter/gradient shape mismatch"); } }
    static void validate_config(const StepConfig & c) { if (!std::isfinite(c.learning_rate) || !std::isfinite(c.beta1) || !std::isfinite(c.beta2) || !std::isfinite(c.eps) || !std::isfinite(c.weight_decay) || !(c.learning_rate > 0) || !(c.beta1 >= 0 && c.beta1 < 1) || !(c.beta2 >= 0 && c.beta2 < 1) || !(c.eps > 0) || c.weight_decay < 0) throw std::invalid_argument("invalid YuE2 optimizer config"); }
    Slot make_slot(const ParameterSpec & spec) { Slot s; s.elements = ggml_nelements(spec.parameter); s.quantized = s.elements >= 4096; s.blocks = (s.elements + 255) / 256; const size_t state_bytes = s.quantized ? s.elements * sizeof(uint8_t) : s.elements * sizeof(float); s.state1.allocate(state_bytes); s.state2.allocate(state_bytes); if (s.quantized) { s.absmax1.allocate(s.blocks*sizeof(float)); s.absmax2.allocate(s.blocks*sizeof(float)); s.qmap1.allocate(256*sizeof(float)); s.qmap2.allocate(256*sizeof(float)); float q1[256], q2[256]; if (!aitk_adamw8bit::adamw8bit_initialize_default_maps(q1,q2)) throw std::runtime_error("YuE2 optimizer default map initialization failed"); if (cudaMemcpy(s.qmap1.get(),q1,sizeof(q1),cudaMemcpyHostToDevice)!=cudaSuccess || cudaMemcpy(s.qmap2.get(),q2,sizeof(q2),cudaMemcpyHostToDevice)!=cudaSuccess) throw std::runtime_error("YuE2 optimizer map upload failed"); } if (cudaMemset(s.state1.get(),0,state_bytes)!=cudaSuccess || cudaMemset(s.state2.get(),0,state_bytes)!=cudaSuccess) throw std::runtime_error("YuE2 optimizer state initialization failed"); if (s.quantized && (cudaMemset(s.absmax1.get(),0,s.blocks*sizeof(float))!=cudaSuccess || cudaMemset(s.absmax2.get(),0,s.blocks*sizeof(float))!=cudaSuccess)) throw std::runtime_error("YuE2 optimizer absmax initialization failed"); return s; }
    aitk_adamw8bit::AdamW8bitCudaState make_state(size_t i) const { const auto & spec=specs_[i]; const auto & s=slots_[i]; aitk_adamw8bit::AdamW8bitCudaState st{}; st.parameter=static_cast<float *>(spec.parameter->data); st.gradient=static_cast<const float *>(spec.gradient->data); if(s.quantized){st.state1_u8=static_cast<uint8_t *>(s.state1.get());st.state2_u8=static_cast<uint8_t *>(s.state2.get());st.absmax1=static_cast<float *>(s.absmax1.get());st.absmax2=static_cast<float *>(s.absmax2.get());st.qmap1=static_cast<const float *>(s.qmap1.get());st.qmap2=static_cast<const float *>(s.qmap2.get());}else{st.state1_fp32=static_cast<float *>(s.state1.get());st.state2_fp32=static_cast<float *>(s.state2.get());} return st; }
    static aitk_adamw8bit::AdamW8bitCudaConfig make_config(size_t n, const StepConfig & c, int step) { aitk_adamw8bit::AdamW8bitCudaConfig x; x.elements=n; x.step=step; x.learning_rate=c.learning_rate;x.beta1=c.beta1;x.beta2=c.beta2;x.eps=c.eps;x.weight_decay=c.weight_decay; return x; }
    template<class T> void copy_to_host(std::vector<T> & out, const void * src) const { if (cudaMemcpyAsync(out.data(),src,out.size()*sizeof(T),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess) throw std::runtime_error("YuE2 optimizer state capture copy failed"); if (cudaStreamSynchronize(stream_)!=cudaSuccess) throw std::runtime_error("YuE2 optimizer capture synchronization failed"); }
    template<class T> void copy_from_device(const std::vector<T> & in, void * dst) { if (cudaMemcpyAsync(dst,in.data(),in.size()*sizeof(T),cudaMemcpyHostToDevice,stream_)!=cudaSuccess) throw std::runtime_error("YuE2 optimizer restore copy failed"); }
    ggml_backend_t backend_ = nullptr; int device_index_ = 0; cudaStream_t stream_ = nullptr; int step_ = 0; std::vector<ParameterSpec> specs_; std::vector<Slot> slots_;
};

} // namespace yue2_aitk
