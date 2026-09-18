#pragma once

// Runtime LoRA factors for the CUDA ConvRot base. Native split AR/NAR exports
// are uploaded separately; the INT8 base weights are never rewritten.

#include "yue2-adapter.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

enum Yue2ConvRotSite { YUE2_CR_Q = 0, YUE2_CR_K, YUE2_CR_V, YUE2_CR_O,
                       YUE2_CR_GATE, YUE2_CR_UP, YUE2_CR_DOWN, YUE2_CR_SITE_COUNT };

struct Yue2ConvRotLora {
    ggml_tensor * a = nullptr; // [input, rank]
    ggml_tensor * b = nullptr; // [rank, output]
    float scale = 0.0f;
};

class Yue2ConvRotAdapter {
public:
    Yue2ConvRotAdapter() = default;
    ~Yue2ConvRotAdapter() { reset(); }
    Yue2ConvRotAdapter(const Yue2ConvRotAdapter &) = delete;
    Yue2ConvRotAdapter & operator=(const Yue2ConvRotAdapter &) = delete;

    const Yue2ConvRotLora & site(int layer, Yue2ConvRotSite site) const {
        return sites_[(size_t) layer][(size_t) site];
    }
    bool nar() const { return nar_; }
    size_t bytes() const { return buffer_ ? ggml_backend_buffer_get_size(buffer_) : 0; }

    void reset() {
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (ctx_) ggml_free(ctx_);
        buffer_ = nullptr;
        ctx_ = nullptr;
        sites_ = {};
    }

    bool load(const Yue2AdapterSpec & spec, ggml_backend_t backend, std::string * err) {
        reset();
        STFile st = {};
        if (!st_open(&st, spec.path.c_str())) return fail(err, "cannot open ConvRot adapter: " + spec.path);
        auto reject = [&](const std::string & why) {
            st_close(&st);
            reset();
            return fail(err, why + ": " + spec.path);
        };
        const Yue2AdapterMeta md = yue2_adapter_read_meta(st);
        if (md.format == "yue2-ar-lora-v1") nar_ = false;
        else if (md.format == "yue2-nar-lora-v1") nar_ = true;
        else return reject("ConvRot requires a native split AR or NAR adapter");
        if (md.rank <= 0 || md.rank > 4096 || !std::isfinite(md.alpha) || md.alpha <= 0 ||
            st.entries.size() != 28u * 7u * 2u || !backend)
            return reject("invalid ConvRot adapter inventory or metadata");
        ggml_init_params ip = { ggml_tensor_overhead() * (28u * 7u * 2u + 16u), nullptr, true };
        ctx_ = ggml_init(ip);
        if (!ctx_) return reject("cannot allocate ConvRot adapter tensor metadata");

        struct Pending { ggml_tensor * tensor; const STEntry * entry; };
        std::vector<Pending> pending;
        pending.reserve(28u * 7u * 2u);
        static const char * names[] = { "attn_q", "attn_k", "attn_v", "attn_output",
                                        "ffn_gate", "ffn_up", "ffn_down" };
        for (int layer = 0; layer < 28; ++layer) {
            for (int s = 0; s < YUE2_CR_SITE_COUNT; ++s) {
                const std::string module = "yue2.blk." + std::to_string(layer) + "." +
                    (nar_ ? "nar_" : "") + names[s];
                const STEntry * a = st_find(st, (module + ".lora_A.weight").c_str());
                const STEntry * b = st_find(st, (module + ".lora_B.weight").c_str());
                const int64_t input = s == YUE2_CR_DOWN ? 6144 : 2048;
                const int64_t output = s == YUE2_CR_K || s == YUE2_CR_V ? 1024 :
                                       s == YUE2_CR_GATE || s == YUE2_CR_UP ? 6144 : 2048;
                if (!a || !b || a->n_dims != 2 || b->n_dims != 2 ||
                    a->shape[0] != md.rank || a->shape[1] != input ||
                    b->shape[0] != output || b->shape[1] != md.rank ||
                    a->dtype != b->dtype ||
                    (a->dtype != "BF16" && a->dtype != "F16" && a->dtype != "F32"))
                    return reject("ConvRot adapter factor shape or dtype mismatch at " + module);
                const size_t elem = a->dtype == "F32" ? 4u : 2u;
                if (!yue2a_extent_ok(st, *a, md.rank * input, elem) ||
                    !yue2a_extent_ok(st, *b, md.rank * output, elem))
                    return reject("ConvRot adapter payload extent is invalid at " + module);
                const ggml_type type = a->dtype == "F32" ? GGML_TYPE_F32 :
                                       a->dtype == "F16" ? GGML_TYPE_F16 : GGML_TYPE_BF16;
                Yue2ConvRotLora & slot = sites_[(size_t) layer][(size_t) s];
                slot.a = ggml_new_tensor_2d(ctx_, type, input, md.rank);
                slot.b = ggml_new_tensor_2d(ctx_, type, md.rank, output);
                if (!slot.a || !slot.b) return reject("cannot allocate ConvRot adapter tensor at " + module);
                const Yue2LoraGroup group = s <= YUE2_CR_O ? YUE2_GRP_ATTN : YUE2_GRP_FFN;
                slot.scale = (md.alpha / (float) md.rank) *
                    yue2_adapter_effective(spec.scales, group, layer, 28);
                if (!std::isfinite(slot.scale)) return reject("nonfinite ConvRot adapter scale");
                ggml_set_name(slot.a, (module + ".lora_A.weight").c_str());
                ggml_set_name(slot.b, (module + ".lora_B.weight").c_str());
                pending.push_back({slot.a, a});
                pending.push_back({slot.b, b});
            }
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buffer_) return reject("cannot allocate ConvRot adapter backend buffer");
        for (const Pending & p : pending)
            ggml_backend_tensor_set(p.tensor, st_data(st, *p.entry), 0, ggml_nbytes(p.tensor));
        st_close(&st);
        return true;
    }

private:
    static bool fail(std::string * err, const std::string & why) {
        if (err) *err = why;
        return false;
    }
    bool nar_ = false;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::array<std::array<Yue2ConvRotLora, YUE2_CR_SITE_COUNT>, 28> sites_{};
};
