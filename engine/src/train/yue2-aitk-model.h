#pragma once

// Native AI Toolkit YuE2 base-weight upload seam. This is deliberately
// separate from the Legacy GGUF YuE2 loader. It preserves ConvRot's raw I8
// rows and F32 per-output scales in GGML tensors; it never makes a full F32
// copy of the base model.
// Consumes the validated installed checkpoint header through yue2-aitk-checkpoint.h.

#include "yue2-aitk-checkpoint.h"
#include "safetensors.h"
#include "../../ggml/include/ggml.h"
#include "../../ggml/include/ggml-backend.h"
#include "../../ggml/include/ggml-alloc.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Yue2AitkConvRotLinear {
    ggml_tensor * weight_i8 = nullptr; // [in, out] GGML view of safetensors [out, in]
    ggml_tensor * scales_f32 = nullptr; // [out] GGML view of safetensors [out, 1]
    int rotation = 256;
    int64_t rows = 0;
    int64_t cols = 0;
    // The tokenizer's companion decoder adapter (apply_companion): a frozen
    // low-rank delta [in, r] / [r, out] on decoder blocks, or a dense F32
    // replacement weight [in, out] (llm2vae, which the companion replaces
    // outright and whose ConvRot weight cannot be patched in place).
    ggml_tensor * frozen_a = nullptr;
    ggml_tensor * frozen_b = nullptr;
    ggml_tensor * dense = nullptr;
};

struct Yue2AitkLayerWeights {
    Yue2AitkConvRotLinear qkv;
    Yue2AitkConvRotLinear output;
    Yue2AitkConvRotLinear gate_up;
    Yue2AitkConvRotLinear down;
};

struct Yue2AitkExpertWeights {
    std::vector<Yue2AitkLayerWeights> layers;
    ggml_tensor * final_norm = nullptr;
};

struct Yue2AitkModelStats {
    size_t convrot_linears = 0;
    size_t embedding_pending = 0;
    size_t ordinary_bf16_tensors = 0;
    size_t raw_i8_bytes = 0;
    size_t raw_scale_bytes = 0;
    size_t ordinary_bf16_bytes = 0;
};

// The embedding is ConvRot-marked in the real checkpoint, but Comfy's loader
// first dequantizes it into an ordinary embedding. Keep that policy outside
// this upload seam. A future callback may create an allocated GGML tensor from
// the validated raw record; nullptr intentionally leaves it pending.
using Yue2AitkEmbeddingLoader = bool (*)(const Yue2AitkCheckpointRecord &, ggml_tensor *,
                                         ggml_backend_t, std::string *);

class Yue2AitkModel {
public:
    Yue2AitkModel() = default;
    ~Yue2AitkModel() { reset(); }
    Yue2AitkModel(const Yue2AitkModel &) = delete;
    Yue2AitkModel & operator=(const Yue2AitkModel &) = delete;

    bool load(const char * checkpoint_path, ggml_backend_t backend,
              Yue2AitkEmbeddingLoader embedding_loader = nullptr,
              std::string * error = nullptr) {
        if (!prepare(checkpoint_path, embedding_loader, error)) return false;
        if (!backend) { reset(); return fail(error, "GGML backend is null"); }
        backend_ = backend;
        return upload(error);
    }

    // Validate and construct only GGML metadata. No backend allocation or payload copy occurs.
    bool prepare(const char * checkpoint_path, Yue2AitkEmbeddingLoader embedding_loader = nullptr,
                 std::string * error = nullptr) {
        reset();
        if (!checkpoint_path || !*checkpoint_path) return fail(error, "checkpoint path is empty");
        std::string why;
        checkpoint_ = std::make_unique<Yue2AitkCheckpoint>();
        if (!checkpoint_->open(checkpoint_path, &why)) { reset(); return fail(error, why); }
        if (checkpoint_->records().size() != 229) {
            reset();
            return fail(error, "validated ConvRot inventory is not 229 records");
        }
        bool have_embedding = false;
        for (const auto & record : checkpoint_->records())
            if (record.name == "text_encoders.model.embed_tokens") have_embedding = true;
        if (!have_embedding) {
            reset();
            return fail(error, "validated checkpoint has no AR embedding record");
        }
        embedding_loader_ = embedding_loader;
        stats_.embedding_pending = 1;
        const size_t ordinary_count = required_ordinary_names().size();
        const size_t descriptor_count = 2 * 228 + ordinary_count + (embedding_loader ? 1 : 0) + 32;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * descriptor_count + 4096;
        params.no_alloc = true;
        ctx_ = ggml_init(params);
        if (!ctx_) {
            reset();
            return fail(error, "failed to allocate GGML metadata context");
        }
        ar_.layers.resize(28); nar_.layers.resize(28);
        if (!load_convrot_sites(error) || !load_ordinary(error)) { reset(); return false; }
        if (embedding_loader) {
            embedding_tensor_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_BF16, 2048, 184704);
            if (!embedding_tensor_) { reset(); return fail(error, "failed creating embedding metadata tensor"); }
            ggml_set_name(embedding_tensor_, "text_encoders.model.embed_tokens");
        }
        return true;
    }

    // Allocate the owned model buffer and upload raw checkpoint bytes. The embedding
    // callback runs only after its allocated tensor is part of that buffer.
    bool upload(std::string * error = nullptr) {
        if (!ctx_ || !backend_) return fail(error, "model metadata is not prepared with a backend");
        if (buffer_) return fail(error, "model payload has already been uploaded");
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
        if (!buffer_) { reset(); return fail(error, "failed to allocate native base GGML buffer"); }
        if (!upload_all(error)) { reset(); return false; }
        if (embedding_loader_) {
            const Yue2AitkCheckpointRecord * embedding = nullptr;
            for (const auto & record : checkpoint_->records())
                if (record.name == "text_encoders.model.embed_tokens") embedding = &record;
            if (!embedding || !embedding_loader_(*embedding, embedding_tensor_, backend_, error)) {
                reset(); return false;
            }
            stats_.embedding_pending = 0;
        }
        return true;
    }

    // Mothersuperior's companion decoder adapter (nar_lora_joint_v9, the pair
    // of the v9 tokenizer head that makes our training codes): its block LoRA
    // becomes a frozen branch on every decoder linear (q/k/v and gate/up fused
    // block-diagonally to match the fused sites), llm2vae takes its full
    // weight through a dense path, and vae2llm (weight, bias) and the llm2vae
    // bias are overwritten. Nothing here is a parameter; gradients still flow
    // through it to the layers below. The engine merges the same file under
    // every generation, so adapters train on the decoder they render with.
    bool apply_companion(const char * path, std::string * error) {
        auto fail = [&](const std::string & m) { if (error) *error = "companion: " + m; return false; };
        if (!backend_ || nar_.layers.empty()) return fail("model is not loaded");
        STFile st = {};
        if (!st_open(&st, path)) return fail(std::string("cannot open ") + path);
        struct Closer { STFile * s; ~Closer() { st_close(s); } } closer{&st};
        auto entry = [&](const std::string & name) -> const STEntry * {
            for (const auto & e : st.entries) if (e.name == name && e.dtype == "F32") return &e;
            return nullptr;
        };
        auto f32 = [&](const STEntry * e) { return (const float *) st_data(st, *e); };
        const int L = (int) nar_.layers.size();
        ggml_init_params ip = { (size_t) (L * 8 + 4) * ggml_tensor_overhead(), nullptr, true };
        companion_ctx_ = ggml_init(ip);
        if (!companion_ctx_) return fail("context allocation failed");
        struct Site { const char * fused; std::vector<const char *> parts; int64_t in, out; };
        struct Built { ggml_tensor * a; ggml_tensor * b; std::vector<float> ha, hb; };
        std::vector<Built> built;
        auto site = [&](int i, Yue2AitkConvRotLinear * dst, std::vector<std::pair<const char *, int64_t>> parts, int64_t in) -> bool {
            // parts: (companion module, its output width), in fused-row order.
            // Each part keeps its own rank: a companion with q at 64 and k/v at
            // 32 must not copy q's A with k's row count.
            int64_t r_total = 0, out_total = 0;
            std::vector<int64_t> ranks;
            for (auto & p : parts) {
                const STEntry * a = entry("layers." + std::to_string(i) + "." + p.first + ".lora_A");
                if (!a || a->n_dims != 2 || a->shape[1] != in) return false;
                ranks.push_back(a->shape[0]); r_total += a->shape[0]; out_total += p.second;
            }
            if (out_total != dst->rows || in != dst->cols) return false;
            Built b;
            b.a = ggml_new_tensor_2d(companion_ctx_, GGML_TYPE_F32, in, r_total);
            b.b = ggml_new_tensor_2d(companion_ctx_, GGML_TYPE_F32, r_total, out_total);
            b.ha.assign((size_t) (in * r_total), 0.0f);
            b.hb.assign((size_t) (r_total * out_total), 0.0f);
            int64_t r0 = 0, o0 = 0;
            for (size_t pi = 0; pi < parts.size(); ++pi) {
                auto &        p = parts[pi];
                const int64_t r = ranks[pi];
                const std::string base = "layers." + std::to_string(i) + "." + p.first;
                const STEntry * ea = entry(base + ".lora_A"), * eb = entry(base + ".lora_B");
                if (!ea || !eb || eb->n_dims != 2 || eb->shape[0] != p.second || eb->shape[1] != r) return false;
                const float * A = f32(ea), * B = f32(eb);
                // A [r, in] rows stack; B [out, r] lands block-diagonally.
                std::copy(A, A + r * in, b.ha.begin() + r0 * in);
                for (int64_t o = 0; o < p.second; ++o)
                    for (int64_t k = 0; k < r; ++k) b.hb[(size_t) ((o0 + o) * r_total + r0 + k)] = B[o * r + k];
                r0 += r; o0 += p.second;
            }
            dst->frozen_a = b.a; dst->frozen_b = b.b;
            built.push_back(std::move(b));
            return true;
        };
        for (int i = 0; i < L; ++i) {
            auto & ly = nar_.layers[(size_t) i];
            if (!site(i, &ly.qkv, {{"nar_self_attn.q_proj", 2048}, {"nar_self_attn.k_proj", 1024}, {"nar_self_attn.v_proj", 1024}}, 2048) ||
                !site(i, &ly.output, {{"nar_self_attn.o_proj", 2048}}, 2048) ||
                !site(i, &ly.gate_up, {{"nar_mlp.gate_proj", 6144}, {"nar_mlp.up_proj", 6144}}, 2048) ||
                !site(i, &ly.down, {{"nar_mlp.down_proj", 2048}}, 6144))
                return fail("layer " + std::to_string(i) + ": block LoRA missing or misshapen");
        }
        const STEntry * l2v = entry("llm2vae.weight");
        if (!l2v || l2v->n_dims != 2 || l2v->shape[0] != llm2vae_.rows || l2v->shape[1] != llm2vae_.cols) return fail("llm2vae.weight missing or misshapen");
        llm2vae_.dense = ggml_new_tensor_2d(companion_ctx_, GGML_TYPE_F32, llm2vae_.cols, llm2vae_.rows);
        companion_buffer_ = ggml_backend_alloc_ctx_tensors(companion_ctx_, backend_);
        if (!companion_buffer_) return fail("backend buffer allocation failed");
        for (auto & b : built) {
            ggml_backend_tensor_set(b.a, b.ha.data(), 0, ggml_nbytes(b.a));
            ggml_backend_tensor_set(b.b, b.hb.data(), 0, ggml_nbytes(b.b));
        }
        ggml_backend_tensor_set(llm2vae_.dense, f32(l2v), 0, ggml_nbytes(llm2vae_.dense));
        // Ordinary flow-head tensors: overwrite in place, in their own type.
        for (auto [ours, theirs] : { std::pair<const char *, const char *>{"model.diffusion_model.vae2llm.weight", "vae2llm.weight"},
                                     {"model.diffusion_model.vae2llm.bias", "vae2llm.bias"}, {"model.diffusion_model.llm2vae.bias", "llm2vae.bias"} }) {
            ggml_tensor * t = ordinary(ours);
            const STEntry * e = entry(theirs);
            if (!t || !e || (int64_t) ((e->data_end - e->data_start) / 4) != ggml_nelements(t)) return fail(std::string(theirs) + " missing or misshapen");
            const float * v = f32(e);
            const int64_t n = ggml_nelements(t);
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(t, v, 0, ggml_nbytes(t));
            } else if (t->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> h((size_t) n);
                for (int64_t k = 0; k < n; ++k) h[(size_t) k] = ggml_fp32_to_bf16(v[k]);
                ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
            } else {
                return fail(std::string(ours) + " has an unexpected type");
            }
        }
        companion_ = true;
        return true;
    }
    bool companion() const { return companion_; }

    void reset() {
        if (companion_buffer_) ggml_backend_buffer_free(companion_buffer_);
        companion_buffer_ = nullptr;
        if (companion_ctx_) ggml_free(companion_ctx_);
        companion_ctx_ = nullptr;
        companion_ = false;
        if (buffer_) ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
        if (ctx_) ggml_free(ctx_);
        ctx_ = nullptr;
        backend_ = nullptr;
        ar_ = Yue2AitkExpertWeights{};
        nar_ = Yue2AitkExpertWeights{};
        lm_head_ = Yue2AitkConvRotLinear{};
        llm2vae_ = Yue2AitkConvRotLinear{};
        time0_ = Yue2AitkConvRotLinear{};
        time2_ = Yue2AitkConvRotLinear{};
        ordinary_.clear();
        embedding_tensor_ = nullptr;
        embedding_loader_ = nullptr;
        stats_ = Yue2AitkModelStats{};
        checkpoint_.reset();
    }

    const Yue2AitkExpertWeights & ar() const { return ar_; }
    const Yue2AitkExpertWeights & nar() const { return nar_; }
    const Yue2AitkConvRotLinear & lm_head() const { return lm_head_; }
    const Yue2AitkConvRotLinear & llm2vae() const { return llm2vae_; }
    const Yue2AitkConvRotLinear & time0() const { return time0_; }
    const Yue2AitkConvRotLinear & time2() const { return time2_; }
    ggml_tensor * ordinary(const char * name) const {
        if (!name) return nullptr;
        for (const auto & item : ordinary_) if (item.name == name) return item.tensor;
        return nullptr;
    }
    ggml_tensor * embedding() const { return embedding_tensor_; }
    bool embedding_dequantization_pending() const { return stats_.embedding_pending != 0; }
    const Yue2AitkModelStats & stats() const { return stats_; }
    ggml_context * context() const { return ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }

private:
    struct Ordinary {
        std::string name;
        ggml_tensor * tensor = nullptr;
    };

    std::unique_ptr<Yue2AitkCheckpoint> checkpoint_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_t backend_ = nullptr; // borrowed; caller owns backend lifetime
    ggml_backend_buffer_t buffer_ = nullptr;
    Yue2AitkExpertWeights ar_, nar_;
    std::vector<Ordinary> ordinary_;
    ggml_tensor * embedding_tensor_ = nullptr;
    Yue2AitkModelStats stats_{};
    Yue2AitkEmbeddingLoader embedding_loader_ = nullptr;

    static bool fail(std::string * error, const std::string & text) {
        if (error) *error = text;
        return false;
    }

    static std::vector<std::string> required_ordinary_names() {
        std::vector<std::string> names;
        for (const char * prefix : {"text_encoders.model.layers.", "model.diffusion_model.model.layers."})
            for (int i = 0; i < 28; ++i) {
                std::string p = std::string(prefix) + std::to_string(i) + ".";
                names.push_back(p + "input_layernorm.weight");
                names.push_back(p + "post_attention_layernorm.weight");
                names.push_back(p + "self_attn.q_norm.weight");
                names.push_back(p + "self_attn.k_norm.weight");
            }
        names.insert(names.end(), {"text_encoders.model.norm.weight", "model.diffusion_model.model.norm.weight",
            "model.diffusion_model.latent_pos_embed.pe", "model.diffusion_model.vae2llm.weight",
            "model.diffusion_model.vae2llm.bias", "model.diffusion_model.llm2vae.bias",
            "model.diffusion_model.time_embedder.mlp.0.bias", "model.diffusion_model.time_embedder.mlp.2.bias"});
        return names;
    }

    static bool ordinary_shape(const std::string & name, const STEntry & e) {
        if (e.dtype != "BF16") return false;
        if (name.find(".input_layernorm.weight") != std::string::npos || name.find(".post_attention_layernorm.weight") != std::string::npos) return e.n_dims == 1 && e.shape[0] == 2048;
        if (name.find(".self_attn.q_norm.weight") != std::string::npos || name.find(".self_attn.k_norm.weight") != std::string::npos) return e.n_dims == 1 && e.shape[0] == 128;
        if (name == "text_encoders.model.norm.weight" || name == "model.diffusion_model.model.norm.weight" || name == "model.diffusion_model.vae2llm.bias" || name == "model.diffusion_model.time_embedder.mlp.0.bias" || name == "model.diffusion_model.time_embedder.mlp.2.bias") return e.n_dims == 1 && e.shape[0] == 2048;
        if (name == "model.diffusion_model.llm2vae.bias") return e.n_dims == 1 && e.shape[0] == 64;
        if (name == "model.diffusion_model.latent_pos_embed.pe") return e.n_dims == 2 && e.shape[0] == 24576 && e.shape[1] == 2048;
        if (name == "model.diffusion_model.vae2llm.weight") return e.n_dims == 2 && e.shape[0] == 2048 && e.shape[1] == 64;
        return false;
    }

    static int site_for(const std::string & name) {
        if (name.find(".self_attn.qkv_proj") != std::string::npos) return 0;
        if (name.find(".self_attn.o_proj") != std::string::npos) return 1;
        if (name.find(".mlp.gate_up_proj") != std::string::npos) return 2;
        if (name.find(".mlp.down_proj") != std::string::npos) return 3;
        return -1;
    }

    bool add_convrot(const Yue2AitkCheckpointRecord & record, Yue2AitkConvRotLinear * out,
                     std::string * error) {
        if (!ctx_ || record.rows <= 0 || record.cols <= 0) return fail(error, "unsupported ConvRot record shape");
        out->weight_i8 = ggml_new_tensor_2d(ctx_, GGML_TYPE_I8, record.cols, record.rows);
        out->scales_f32 = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, record.rows);
        if (!out->weight_i8 || !out->scales_f32) return fail(error, "failed creating ConvRot metadata tensors");
        ggml_set_name(out->weight_i8, (record.name + ".weight_i8").c_str());
        ggml_set_name(out->scales_f32, (record.name + ".scales_f32").c_str());
        out->rows = record.rows; out->cols = record.cols; out->rotation = 256;
        stats_.convrot_linears++;
        stats_.raw_i8_bytes += record.weight_bytes;
        stats_.raw_scale_bytes += record.scale_bytes;
        return true;
    }

    bool load_convrot_sites(std::string * error) {
        for (const auto & record : checkpoint_->records()) {
            if (record.name == "text_encoders.model.embed_tokens") continue; // explicit dequant seam
            int site = site_for(record.name);
            Yue2AitkConvRotLinear * dst = nullptr;
            Yue2AitkExpertWeights * expert = nullptr;
            if (record.name.rfind("text_encoders.model.layers.", 0) == 0) expert = &ar_;
            else if (record.name.rfind("model.diffusion_model.model.layers.", 0) == 0) expert = &nar_;
            else if (record.name == "text_encoders.model.lm_head") { if (!add_convrot(record, &lm_head_, error)) return false; continue; }
            else if (record.name == "model.diffusion_model.llm2vae") { if (!add_convrot(record, &llm2vae_, error)) return false; continue; }
            else if (record.name == "model.diffusion_model.time_embedder.mlp.0") { if (!add_convrot(record, &time0_, error)) return false; continue; }
            else if (record.name == "model.diffusion_model.time_embedder.mlp.2") { if (!add_convrot(record, &time2_, error)) return false; continue; }
            else return fail(error, "unexpected validated ConvRot record: " + record.name);
            if (!expert || site < 0) return fail(error, "ConvRot record is not a supported native site: " + record.name);
            size_t dot = record.name.find(".model.layers.");
            size_t layer_start = dot == std::string::npos ? std::string::npos : dot + 14;
            size_t layer_end = record.name.find('.', layer_start);
            if (layer_start == std::string::npos || layer_end == std::string::npos) return fail(error, "invalid layer name: " + record.name);
            int layer = std::stoi(record.name.substr(layer_start, layer_end - layer_start));
            if (layer < 0 || layer >= 28) return fail(error, "layer index out of range");
            switch (site) { case 0: dst = &expert->layers[layer].qkv; break; case 1: dst = &expert->layers[layer].output; break; case 2: dst = &expert->layers[layer].gate_up; break; default: dst = &expert->layers[layer].down; break; }
            if (!add_convrot(record, dst, error)) return false;
        }
        return stats_.convrot_linears == 228 ? true : fail(error, "native ConvRot upload count is not 228");
    }

    bool load_ordinary(std::string * error) {
        for (const std::string & name : required_ordinary_names()) {
            const STEntry * ep = st_find(checkpoint_->mapped_file(), name.c_str());
            if (!ep) return fail(error, "required ordinary BF16 tensor is missing: " + name);
            const STEntry & e = *ep;
            if (!ordinary_shape(name, e)) return fail(error, "ordinary BF16 tensor has wrong dtype/shape: " + name);
            if (e.n_dims < 1 || e.n_dims > 2) return fail(error, "unsupported ordinary BF16 rank: " + e.name);
            ggml_tensor * tensor = e.n_dims == 1 ? ggml_new_tensor_1d(ctx_, GGML_TYPE_BF16, e.shape[0])
                                                   : ggml_new_tensor_2d(ctx_, GGML_TYPE_BF16, e.shape[1], e.shape[0]);
            if (!tensor) return fail(error, "failed creating ordinary BF16 tensor: " + e.name);
            ggml_set_name(tensor, e.name.c_str());
            ordinary_.push_back({ name, tensor });
            stats_.ordinary_bf16_tensors++;
            stats_.ordinary_bf16_bytes += ggml_nbytes(tensor);
            if (name == "text_encoders.model.norm.weight") ar_.final_norm = tensor;
            if (name == "model.diffusion_model.model.norm.weight") nar_.final_norm = tensor;
        }
        return true;
    }

    bool upload_convrot(const Yue2AitkCheckpointRecord & record, const Yue2AitkConvRotLinear & line, std::string * error) {
        const STEntry * w = st_find(checkpoint_->mapped_file(), (record.name + ".weight").c_str());
        const STEntry * s = st_find(checkpoint_->mapped_file(), (record.name + ".weight_scale").c_str());
        if (!w || !s) return fail(error, "validated ConvRot payload disappeared: " + record.name);
        ggml_backend_tensor_set(line.weight_i8, st_data(checkpoint_->mapped_file(), *w), 0, ggml_nbytes(line.weight_i8));
        ggml_backend_tensor_set(line.scales_f32, st_data(checkpoint_->mapped_file(), *s), 0, ggml_nbytes(line.scales_f32));
        return true;
    }

    bool upload_all(std::string * error) {
        for (const auto & record : checkpoint_->records()) {
            if (record.name == "text_encoders.model.embed_tokens") continue;
            int site = site_for(record.name); const Yue2AitkConvRotLinear * line = nullptr;
            if (record.name == "text_encoders.model.lm_head") line = &lm_head_;
            else if (record.name == "model.diffusion_model.llm2vae") line = &llm2vae_;
            else if (record.name == "model.diffusion_model.time_embedder.mlp.0") line = &time0_;
            else if (record.name == "model.diffusion_model.time_embedder.mlp.2") line = &time2_;
            else {
                bool nar = record.name.rfind("model.diffusion_model.model.layers.", 0) == 0;
                size_t p = record.name.find(".model.layers.") + 14; size_t q = record.name.find('.', p);
                int layer = std::stoi(record.name.substr(p, q - p));
                const auto & l = (nar ? nar_ : ar_).layers[layer];
                line = site == 0 ? &l.qkv : site == 1 ? &l.output : site == 2 ? &l.gate_up : &l.down;
            }
            if (!upload_convrot(record, *line, error)) return false;
        }
        for (const auto & ordinary : ordinary_) {
            const STEntry * e = st_find(checkpoint_->mapped_file(), ordinary.name.c_str());
            if (!e) return fail(error, "ordinary tensor disappeared: " + ordinary.name);
            ggml_backend_tensor_set(ordinary.tensor, st_data(checkpoint_->mapped_file(), *e), 0, ggml_nbytes(ordinary.tensor));
        }
        return true;
    }

    Yue2AitkConvRotLinear lm_head_, llm2vae_, time0_, time2_;
    ggml_context * companion_ctx_ = nullptr;
    ggml_backend_buffer_t companion_buffer_ = nullptr;
    bool companion_ = false;
};
