#pragma once

// Persistent native-v1 YuE2 AR/NAR fused-LoRA state. This owns the two
// parameter collections and their persistent F32 gradient slots, but keeps
// optimizer state and update policy outside the export contract.

#include "yue2-aitk-block-executor.h"
#include "yue2-aitk-native-adapter-io.h"
#include "yue2-aitk-lora.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class Yue2AitkTrainState {
public:
    struct NamedTensors { std::string name; ggml_tensor * parameter; ggml_tensor * gradient; };
    static constexpr int kLayers = 28;
    static constexpr int kSites = 4;
    static constexpr int kFactors = 2;
    static constexpr uint32_t kDefaultSeed = 0x59414531u;

    Yue2AitkTrainState() = default;
    ~Yue2AitkTrainState() { reset(); }
    Yue2AitkTrainState(const Yue2AitkTrainState &) = delete;
    Yue2AitkTrainState & operator=(const Yue2AitkTrainState &) = delete;

    bool initialize(ggml_backend_t backend, uint32_t seed = kDefaultSeed, std::string * error = nullptr, bool cursor = false,
                    int64_t rank = 32, float alpha = 32.0f) {
        reset();
        if (!backend) return fail(error, "backend is null");
        if (rank < 1 || rank > 65536) return fail(error, "rank must be within [1, 65536]");
        if (!std::isfinite(alpha) || alpha <= 0.0f) return fail(error, "alpha must be finite and positive");
        ggml_init_params params{}; params.mem_size = 1024*ggml_tensor_overhead()+4096; params.no_alloc = true;
        ctx_ = ggml_init(params); if (!ctx_) return fail(error, "failed to create train-state context");
        backend_ = backend; seed_ = seed; init_policy_ = "native-v1"; rank_ = rank; alpha_ = alpha;
        const Yue2AitkDims dims{2048, 2048, 1024, 6144};
        if (!yue2_aitk_make_expert_adapters(ctx_, dims, kLayers, rank, alpha, &ar_, "ar") ||
            !yue2_aitk_make_expert_adapters(ctx_, dims, kLayers, rank, alpha, &nar_, "nar")) {
            reset(); return fail(error, "failed to allocate AR/NAR adapters");
        }
        slots_.reserve(2u * kLayers * kSites * kFactors);
        if (!build_slots(nar_, "diffusion_model", 0, error) || !build_slots(ar_, "text_encoders", 1, error)) { reset(); return false; }
        if (cursor) {
            auto * parameter=ggml_new_tensor_2d(ctx_,GGML_TYPE_F32,2048,2048);
            auto * gradient=ggml_new_tensor_2d(ctx_,GGML_TYPE_F32,2048,2048);
            cursor_slot_=slots_.size();
            slots_.push_back({"cursor_head.weight",parameter,gradient,2048,2048,-1,std::vector<float>(2048*2048,0)});
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_) { reset(); return fail(error, "failed to allocate train-state backend buffer"); }
        std::mt19937 rng(seed_);
        for (Slot & slot : slots_) {
            std::fill(slot.host.begin(), slot.host.end(), 0.0f);
            if (slot.factor == 0) {
                const float bound = 1.0f / std::sqrt(static_cast<float>(slot.cols));
                std::uniform_real_distribution<float> dist(-bound, bound);
                for (float & value : slot.host) value = dist(rng);
            }
            if (slot.factor == -1) for(size_t i=0;i<size_t(slot.rows);++i) slot.host[i*size_t(slot.cols)+i]=1.0f;
            ggml_backend_tensor_set(slot.parameter, slot.host.data(), 0, ggml_nbytes(slot.parameter));
            ggml_backend_tensor_memset(slot.gradient, 0, 0, ggml_nbytes(slot.gradient));
        }
        initialized_ = true;
        return true;
    }

    void reset() {
        initialized_ = false; slots_.clear(); ar_ = {}; nar_ = {}; cursor_slot_=kInvalid;
        if (buffer_) { ggml_backend_buffer_free(buffer_); buffer_ = nullptr; }
        if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
        backend_ = nullptr;
    }

    bool initialized() const { return initialized_; }
    uint32_t seed() const { return seed_; }
    const std::string & initialization_policy() const { return init_policy_; }
    int64_t rank() const { return rank_; }
    float alpha() const { return alpha_; }
    const Yue2AitkExpertAdapters & ar_adapters() const { return ar_; }
    const Yue2AitkExpertAdapters & nar_adapters() const { return nar_; }
    ggml_tensor * cursor_head() const { return cursor_slot_==kInvalid?nullptr:slots_[cursor_slot_].parameter; }
    ggml_tensor * cursor_gradient() const { return cursor_slot_==kInvalid?nullptr:slots_[cursor_slot_].gradient; }
    void clear_cursor_gradient() {
        if(auto * gradient=cursor_gradient()) ggml_backend_tensor_memset(gradient,0,0,ggml_nbytes(gradient));
    }
    std::vector<NamedTensors> named_tensors() const {
        std::vector<NamedTensors> result; result.reserve(slots_.size());
        for (const auto & slot : slots_) result.push_back({slot.name,slot.parameter,slot.gradient});
        return result;
    }

    // Upload the eight block gradients into their persistent F32 slots. The
    // tensors remain available for a later optimizer, but this class performs
    // no update itself.
    bool upload_gradients(bool nar, int layer, const Yue2AitkBlockBackwardHost & g, std::string * error = nullptr) {
        if (!initialized_ || layer < 0 || layer >= kLayers) return fail(error, "train state is not initialized or layer is invalid");
        const std::array<const std::vector<float> *, 8> values = {{&g.qkv_dA,&g.qkv_dB,&g.output_dA,&g.output_dB,&g.gate_up_dA,&g.gate_up_dB,&g.down_dA,&g.down_dB}};
        for (int factor = 0; factor < kFactors; ++factor) for (int site = 0; site < kSites; ++site) {
            const size_t index = slot_index_[nar ? 0 : 1][layer][site][factor];
            if (index == kInvalid || values[site*2+factor]->size() != slots_[index].host.size()) return fail(error, "gradient shape mismatch");
            for (float value : *values[site*2+factor]) if (!std::isfinite(value)) return fail(error,"non-finite adapter gradient");
        }
        for (int factor = 0; factor < kFactors; ++factor) for (int site = 0; site < kSites; ++site) {
            Slot & slot = slots_[slot_index_[nar ? 0 : 1][layer][site][factor]];
            ggml_backend_tensor_set(slot.gradient, values[site*2+factor]->data(), 0, ggml_nbytes(slot.gradient));
        }
        return true;
    }

    // Default recipe uses one joint parameter group. The host reduction uses
    // double accumulation; its rounding must be measured against Torch's norm.
    bool clip_gradients(float max_norm, double * norm, std::string * error=nullptr) {
        if (!initialized_ || !(max_norm>0) || !std::isfinite(max_norm)) return fail(error,"invalid clipping request");
        double squared=0;
        for (auto & slot : slots_) {
            ggml_backend_tensor_get(slot.gradient,slot.host.data(),0,ggml_nbytes(slot.gradient));
            for (float value : slot.host) {
                if (!std::isfinite(value)) return fail(error,"non-finite gradient before clipping");
                squared+=double(value)*double(value);
            }
        }
        const double length=std::sqrt(squared);
        if (norm) *norm=length;
        const float scale=(std::min)(1.0f,max_norm/(float(length)+1e-6f));
        if (scale<1.0f) for (auto & slot : slots_) {
            for (float & value : slot.host) value*=scale;
            ggml_backend_tensor_set(slot.gradient,slot.host.data(),0,ggml_nbytes(slot.gradient));
        }
        return true;
    }

    // Upload every slot's current host gradient (clipped by clip_gradients)
    // into F32 tensors of matching shape. The LmOptim path uses this to fill
    // its persistent accumulators, which live in a buffer this class does not
    // own and must never be treated as its own gradient buffer.
    bool fill_host_gradients(std::vector<ggml_tensor *> & targets, std::string * error = nullptr) const {
        if (!initialized_ || targets.size() != slots_.size()) return fail(error, "gradient target count does not match slot count");
        for (size_t i = 0; i < slots_.size(); ++i) {
            const Slot & slot = slots_[i];
            ggml_tensor * target = targets[i];
            if (!target || target->type != GGML_TYPE_F32 || !ggml_is_contiguous(target)) return fail(error, "gradient target must be contiguous F32");
            for (int d = 0; d < GGML_MAX_DIMS; ++d) if (target->ne[d] != slot.parameter->ne[d]) return fail(error, "gradient target shape mismatch");
            ggml_backend_tensor_set(target, slot.host.data(), 0, ggml_nbytes(target));
        }
        return true;
    }

    // Refreshes host F32 snapshots and invokes the installed 448-tensor BF16
    // writer. The writer enforces exact names/shapes and no-overwrite publish.
    bool export_snapshot(const char * output_path, int64_t steps, std::string * error = nullptr, const char * ar_path = nullptr, const char * nar_path = nullptr, const std::string & trigger = {}, float caption_dropout = 0.0f) {
        if (!initialized_) return fail(error, "train state is not initialized");
        std::vector<Yue2AitkF32Matrix> factors; factors.reserve(slots_.size());
        for (Slot & slot : slots_) {
            if (slot.factor == -1) continue; // auxiliary training head is resume-only
            ggml_backend_tensor_get(slot.parameter, slot.host.data(), 0, ggml_nbytes(slot.parameter));
            factors.push_back({slot.name, slot.rows, slot.cols, slot.host.data()});
        }
        if (!yue2_aitk_write_fused_lora(factors, rank_, alpha_, steps, output_path)) return fail(error, "fused-LoRA export failed");
        if ((ar_path || nar_path) && !yue2_aitk_write_native_split(factors, rank_, alpha_, steps, ar_path, nar_path, trigger, caption_dropout)) return fail(error, "native AR/NAR export failed");
        return true;
    }

private:
    static constexpr size_t kInvalid = static_cast<size_t>(-1);
    struct Slot {
        std::string name;
        ggml_tensor * parameter = nullptr;
        ggml_tensor * gradient = nullptr;
        int64_t rows = 0, cols = 0;
        int factor = 0;
        std::vector<float> host;
    };
    ggml_context * ctx_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    Yue2AitkExpertAdapters ar_, nar_;
    std::vector<Slot> slots_;
    std::array<std::array<std::array<std::array<size_t, kFactors>, kSites>, kLayers>, 2> slot_index_{};
    uint32_t seed_ = kDefaultSeed;
    std::string init_policy_ = "native-v1";
    bool initialized_ = false;
    size_t cursor_slot_ = kInvalid;
    int64_t rank_ = 32;
    float alpha_ = 32.0f;

    static bool fail(std::string * error, const char * message) { if (error) *error = message; return false; }
    static bool fail(std::string * error, const std::string & message) { if (error) *error = message; return false; }
    static const Yue2AitkFusedLora & site(const Yue2AitkExpertAdapters & e, int layer, int s) {
        const Yue2AitkLayerAdapters & l = e.layers[static_cast<size_t>(layer)];
        return s == 0 ? l.qkv : s == 1 ? l.output : s == 2 ? l.gate_up : l.down;
    }
    bool build_slots(const Yue2AitkExpertAdapters & expert, const char * prefix, int expert_index, std::string * error) {
        static const char * names[kSites] = {"self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"};
        for (int layer=0; layer<kLayers; ++layer) for (int s=0; s<kSites; ++s) {
            const Yue2AitkFusedLora & site_ref = site(expert,layer,s);
            for (int factor=0; factor<kFactors; ++factor) {
                ggml_tensor * parameter = factor == 0 ? site_ref.a : site_ref.b;
                if (!parameter) return fail(error, "null adapter parameter");
                const int64_t rows = parameter->ne[1], cols = parameter->ne[0];
                ggml_tensor * gradient = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, cols, rows);
                if (!gradient) return fail(error, "failed to allocate persistent gradient");
                const std::string name = std::string(prefix) + ".model.layers." + std::to_string(layer) + "." + names[s] + ".lora_" + (factor == 0 ? "A" : "B") + ".weight";
                Slot slot{name,parameter,gradient,rows,cols,factor,std::vector<float>(static_cast<size_t>(rows*cols),0.0f)};
                slot_index_[expert_index][layer][s][factor] = slots_.size(); slots_.push_back(std::move(slot));
            }
        }
        return true;
    }
};
