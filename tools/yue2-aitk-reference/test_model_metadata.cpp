#include "../../engine/src/train/yue2-aitk-model.h"
#include "../../engine/ggml/include/ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool check(bool value, const char *what) {
    if (!value) std::cerr << "FAIL: " << what << '\n';
    return value;
}

bool empty_after_reset(const Yue2AitkModel &model) {
    const Yue2AitkModelStats &stats = model.stats();
    return model.context() == nullptr && model.buffer() == nullptr && model.ar().layers.empty() &&
           model.nar().layers.empty() && model.ar().final_norm == nullptr &&
           model.nar().final_norm == nullptr && model.lm_head().weight_i8 == nullptr &&
           model.llm2vae().weight_i8 == nullptr && model.time0().weight_i8 == nullptr &&
           model.time2().weight_i8 == nullptr && model.embedding() == nullptr &&
           !model.embedding_dequantization_pending() && stats.convrot_linears == 0 &&
           stats.embedding_pending == 0 && stats.ordinary_bf16_tensors == 0 &&
           stats.raw_i8_bytes == 0 && stats.raw_scale_bytes == 0 && stats.ordinary_bf16_bytes == 0;
}

bool compare_tensor_chunks(const ggml_tensor *tensor, const void *expected, size_t bytes) {
    if (!tensor || !expected || ggml_nbytes(tensor) != bytes) return false;
    constexpr size_t kChunk = 8u << 20;
    std::vector<uint8_t> actual((std::min)(kChunk, bytes));
    const auto *source = static_cast<const uint8_t *>(expected);
    for (size_t offset = 0; offset < bytes; ) {
        const size_t amount = (std::min)(kChunk, bytes - offset);
        ggml_backend_tensor_get(tensor, actual.data(), offset, amount);
        if (std::memcmp(actual.data(), source + offset, amount) != 0) return false;
        offset += amount;
    }
    return true;
}

bool compare_named_tensor(const Yue2AitkModel &model, const STFile &file,
                          const char *name, const ggml_tensor *tensor) {
    const STEntry *entry = st_find(file, name);
    return entry && compare_tensor_chunks(tensor, st_data(file, *entry), entry->data_end - entry->data_start);
}

bool run_upload_cpu(const char *checkpoint_path) {
    bool ok = true;
    ggml_backend_t backend = ggml_backend_cpu_init();
    ok = check(backend != nullptr, "CPU backend initializes") && ok;
    if (!backend) return false;
    Yue2AitkModel model;
    std::string error;
    ok = check(model.load(checkpoint_path, backend, nullptr, &error), "CPU upload succeeds") && ok;
    if (!ok) std::cerr << "upload error: " << error << '\n';
    Yue2AitkCheckpoint verifier;
    std::string verify_error;
    ok = check(verifier.open(checkpoint_path, &verify_error), "validated checkpoint opens for upload verification") && ok;
    if (ok) {
        const STFile &file = verifier.mapped_file();
        const auto compare_site = [&](const char *prefix, int layer, const char *site,
                                      const Yue2AitkConvRotLinear &line) {
            const std::string base = std::string(prefix) + ".model.layers." + std::to_string(layer) + "." + site;
            const std::string weight_name = base + ".weight";
            const std::string scale_name = base + ".weight_scale";
            ok = check(compare_named_tensor(model, file, weight_name.c_str(), line.weight_i8),
                       "uploaded ConvRot I8 bytes match") && ok;
            ok = check(compare_named_tensor(model, file, scale_name.c_str(), line.scales_f32),
                       "uploaded ConvRot scale bytes match") && ok;
        };
        for (int layer = 0; layer < 28; ++layer) {
            const auto &ar = model.ar().layers[layer];
            const auto &nar = model.nar().layers[layer];
            for (const auto &site : {std::pair<const char *, const Yue2AitkConvRotLinear *>("self_attn.qkv_proj", &ar.qkv),
                                     {"self_attn.o_proj", &ar.output}, {"mlp.gate_up_proj", &ar.gate_up},
                                     {"mlp.down_proj", &ar.down}})
                compare_site("text_encoders", layer, site.first, *site.second);
            for (const auto &site : {std::pair<const char *, const Yue2AitkConvRotLinear *>("self_attn.qkv_proj", &nar.qkv),
                                     {"self_attn.o_proj", &nar.output}, {"mlp.gate_up_proj", &nar.gate_up},
                                     {"mlp.down_proj", &nar.down}})
                compare_site("model.diffusion_model", layer, site.first, *site.second);
        }
        ok = check(compare_named_tensor(model, file, "text_encoders.model.lm_head.weight", model.lm_head().weight_i8),
                   "uploaded LM head I8 bytes match") && ok;
        ok = check(compare_named_tensor(model, file, "text_encoders.model.lm_head.weight_scale", model.lm_head().scales_f32),
                   "uploaded LM head scales match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.llm2vae.weight", model.llm2vae().weight_i8),
                   "uploaded llm2vae I8 bytes match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.llm2vae.weight_scale", model.llm2vae().scales_f32),
                   "uploaded llm2vae scales match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.time_embedder.mlp.0.weight", model.time0().weight_i8),
                   "uploaded time0 I8 bytes match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.time_embedder.mlp.0.weight_scale", model.time0().scales_f32),
                   "uploaded time0 scales match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.time_embedder.mlp.2.weight", model.time2().weight_i8),
                   "uploaded time2 I8 bytes match") && ok;
        ok = check(compare_named_tensor(model, file, "model.diffusion_model.time_embedder.mlp.2.weight_scale", model.time2().scales_f32),
                   "uploaded time2 scales match") && ok;
        for (const char *name : {"text_encoders.model.norm.weight", "model.diffusion_model.model.norm.weight",
                                 "model.diffusion_model.latent_pos_embed.pe", "model.diffusion_model.vae2llm.weight",
                                 "model.diffusion_model.vae2llm.bias", "model.diffusion_model.llm2vae.bias",
                                 "model.diffusion_model.time_embedder.mlp.0.bias", "model.diffusion_model.time_embedder.mlp.2.bias"})
            ok = check(compare_named_tensor(model, file, name, model.ordinary(name)),
                       "uploaded ordinary bytes match") && ok;
        for (int layer = 0; layer < 28; ++layer) {
            for (const char *expert : {"text_encoders", "model.diffusion_model"}) {
                const std::string p = std::string(expert) + ".model.layers." + std::to_string(layer) + ".";
                for (const char *suffix : {"input_layernorm.weight", "post_attention_layernorm.weight",
                                           "self_attn.q_norm.weight", "self_attn.k_norm.weight"}) {
                    const std::string name = p + suffix;
                    ok = check(compare_named_tensor(model, file, name.c_str(), model.ordinary(name.c_str())),
                               "uploaded layer norm bytes match") && ok;
                }
            }
        }
        const ggml_backend_buffer_t buffer_before = model.buffer();
        ok = check(!model.upload(&error) && model.buffer() == buffer_before,
                   "repeated upload fails without changing buffer") && ok;
    }
    model.reset();
    ok = check(empty_after_reset(model), "upload reset clears state") && ok;
    ggml_backend_free(backend);
    return ok;
}
}

int main(int argc, char **argv) {
    if (argc == 3 && std::string(argv[1]) == "--upload-cpu")
        return run_upload_cpu(argv[2]) ? 0 : 1;
    if (argc != 2) {
        std::cerr << "usage: test_model_metadata.exe CHECKPOINT.safetensors\n"
                  << "   or: test_model_metadata.exe --upload-cpu CHECKPOINT.safetensors\n";
        return 2;
    }
    bool ok = true;
    Yue2AitkModel model;
    std::string error;
    ok = check(model.prepare(argv[1], nullptr, &error), "actual checkpoint prepares") && ok;
    if (ok) {
        const Yue2AitkModelStats &stats = model.stats();
        ok = check(stats.convrot_linears == 228, "228 ConvRot linears") && ok;
        ok = check(stats.ordinary_bf16_tensors == 232, "232 ordinary BF16 tensors") && ok;
        ok = check(stats.embedding_pending == 1 && model.embedding_dequantization_pending(),
                   "embedding remains pending") && ok;
        ok = check(model.ar().layers.size() == 28 && model.nar().layers.size() == 28,
                   "both experts have 28 layers") && ok;
        ok = check(model.ar().layers[0].qkv.rows == 4096 && model.ar().layers[0].qkv.cols == 2048 &&
                   model.nar().layers[0].qkv.rows == 4096 && model.nar().layers[0].qkv.cols == 2048,
                   "AR/NAR QKV dimensions") && ok;
        ok = check(model.lm_head().rows == 184704 && model.lm_head().cols == 2048,
                   "LM head dimensions") && ok;
        ok = check(model.time0().rows == 2048 && model.time0().cols == 256 &&
                   model.time2().rows == 2048 && model.time2().cols == 2048,
                   "time embedding dimensions") && ok;
        ok = check(model.ordinary("text_encoders.model.norm.weight") != nullptr &&
                   model.ordinary("model.diffusion_model.latent_pos_embed.pe") != nullptr &&
                   model.ordinary("missing.tensor") == nullptr,
                   "ordinary tensor lookup") && ok;
    } else {
        std::cerr << "prepare error: " << error << '\n';
    }

    model.reset();
    ok = check(empty_after_reset(model), "reset clears metadata/context/buffer/counts") && ok;
    error.clear();
    ok = check(model.prepare(argv[1], nullptr, &error), "repeat prepare after reset") && ok;
    model.reset();
    ok = check(empty_after_reset(model), "second reset clears state") && ok;

    error.clear();
    ok = check(!model.prepare("__missing_yue2_checkpoint__.safetensors", nullptr, &error),
               "missing checkpoint fails cleanly") && ok;
    ok = check(empty_after_reset(model), "missing checkpoint leaves clean state") && ok;

    if (!ok) return 1;
    std::cout << "YuE2 model metadata contract checks passed\n";
    return 0;
}
