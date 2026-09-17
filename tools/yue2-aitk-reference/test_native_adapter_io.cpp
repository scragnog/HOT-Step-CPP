#include "../../_experiments/aitk-port/staging/yue2-aitk-native-adapter-io.h"
#include "../../engine/src/yue2/yue2-adapter.h"
#include "../../engine/vendor/yyjson/yyjson.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Owned {
    std::string name;
    std::vector<float> values;
};

bool expect(bool value, const char * message) {
    if (!value) std::cerr << "FAIL: " << message << '\n';
    return value;
}

bool loader_names_accept(const char * format) {
    const bool nar = std::string(format) == "nar";
    const std::string prefix = std::string("yue2.blk.0.") + (nar ? "nar_" : "");
    const char * targets[] = {"attn_q", "attn_k", "ffn_gate"};
    for (const char * target : targets) {
        const Yue2LoraTarget mapped = yue2_lora_target(prefix + target);
        if (mapped.why != YUE2_LR_OK || mapped.gguf_name.empty()) return false;
        if (nar != (mapped.family == YUE2_FAM_NAR)) return false;
    }
    return true;
}

std::vector<Yue2AitkF32Matrix> inventory(int64_t rank, std::vector<Owned> * owned) {
    owned->clear();
    std::vector<Yue2AitkF32Matrix> result;
    int expert_index = 0;
    for (const char * expert : {"diffusion_model", "text_encoders"}) {
        int site_index = 0;
        for (int layer = 0; layer < 28; ++layer) {
            for (const char * site : {"self_attn.qkv_proj", "self_attn.o_proj",
                                      "mlp.gate_up_proj", "mlp.down_proj"}) {
                const int64_t input = std::string(site) == "mlp.down_proj" ? 6144 : 2048;
                const int64_t output = std::string(site) == "self_attn.qkv_proj" ? 4096 :
                    std::string(site) == "mlp.gate_up_proj" ? 12288 : 2048;
                const std::string base = std::string(expert) + ".model.layers." +
                    std::to_string(layer) + "." + site + ".lora_";
                const float a_value = 0.25f + 0.5f * expert_index + 0.01f * site_index;
                owned->push_back({base + "A.weight", std::vector<float>(static_cast<size_t>(rank * input), a_value)});
                std::vector<float> b_values(static_cast<size_t>(output * rank));
                const float expert_base = 100.0f * expert_index;
                for (int64_t row = 0; row < output; ++row) {
                    float row_value = 60.0f;
                    if (std::string(site) == "self_attn.qkv_proj") {
                        row_value = row < 2048 ? 10.0f : row < 3072 ? 20.0f : 30.0f;
                    } else if (std::string(site) == "mlp.gate_up_proj") {
                        row_value = row < 6144 ? 40.0f : 50.0f;
                    } else if (std::string(site) == "mlp.down_proj") {
                        row_value = 70.0f;
                    }
                    for (int64_t col = 0; col < rank; ++col) {
                        b_values[static_cast<size_t>(row * rank + col)] = expert_base + row_value;
                    }
                }
                owned->push_back({base + "B.weight", std::move(b_values)});
                result.push_back({owned->at(owned->size() - 2).name, rank, input, owned->at(owned->size() - 2).values.data()});
                result.push_back({owned->back().name, output, rank, owned->back().values.data()});
                ++site_index;
            }
        }
        ++expert_index;
    }
    return result;
}

uint64_t u64(const unsigned char * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

bool check_header(const std::filesystem::path & path, const char * format,
                  size_t expected_tensors) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    unsigned char bytes[8]{};
    in.read(reinterpret_cast<char *>(bytes), 8);
    if (in.gcount() != 8) return false;
    const uint64_t n = u64(bytes);
    std::string json(static_cast<size_t>(n), '\0');
    in.read(json.data(), static_cast<std::streamsize>(n));
    if (in.gcount() != static_cast<std::streamsize>(n)) return false;
    yyjson_doc * doc = yyjson_read(json.data(), json.size(), 0);
    if (!doc) return false;
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md = yyjson_obj_get(root, "__metadata__");
    yyjson_val * fmt = md ? yyjson_obj_get(md, "format") : nullptr;
    yyjson_val * layout = md ? yyjson_obj_get(md, "yue2_adapter_layout") : nullptr;
    bool ok = yyjson_is_obj(root) && yyjson_is_obj(md) && yyjson_is_str(fmt) &&
        std::string(yyjson_get_str(fmt)) == std::string("yue2-") + format + "-lora-v1" &&
        yyjson_is_str(layout) && std::string(yyjson_get_str(layout)) == "native_split_v1" &&
        yyjson_obj_size(root) == expected_tensors + 1;
    // Spot-check the native names and dimensions; the complete inventory count
    // above catches dropped/duplicated split factors.
    const std::string prefix = std::string("yue2.blk.0.") + (std::string(format) == "nar" ? "nar_" : "");
    const std::string names[] = {prefix + "attn_q.lora_A.weight", prefix + "attn_q.lora_B.weight",
                                prefix + "attn_k.lora_B.weight", prefix + "ffn_gate.lora_B.weight"};
    const int64_t rows[] = {2, 2048, 1024, 6144};
    const int64_t cols[] = {2048, 2, 2, 2};
    std::vector<std::pair<uint64_t, uint64_t>> offsets;
    for (const std::string & name : names) {
        yyjson_val * t = yyjson_obj_get(root, name.c_str());
        yyjson_val * shape = t ? yyjson_obj_get(t, "shape") : nullptr;
        yyjson_val * off = t ? yyjson_obj_get(t, "data_offsets") : nullptr;
        yyjson_val * dtype = t ? yyjson_obj_get(t, "dtype") : nullptr;
        ok = ok && t && yyjson_is_obj(t) && shape && yyjson_arr_size(shape) == 2 &&
            yyjson_get_sint(yyjson_arr_get(shape, 0)) == rows[offsets.size()] &&
            yyjson_get_sint(yyjson_arr_get(shape, 1)) == cols[offsets.size()] &&
            dtype && yyjson_is_str(dtype) && std::string(yyjson_get_str(dtype)) == "BF16" &&
            off && yyjson_arr_size(off) == 2;
        if (off && yyjson_arr_size(off) == 2) offsets.emplace_back(
            yyjson_get_uint(yyjson_arr_get(off, 0)), yyjson_get_uint(yyjson_arr_get(off, 1)));
    }
    yyjson_doc_free(doc);
    if (!ok || offsets.size() != 4) return false;
    in.clear();
    in.seekg(static_cast<std::streamoff>(8u + n + offsets[0].first), std::ios::beg);
    uint16_t a_bits = 0;
    in.read(reinterpret_cast<char *>(&a_bits), sizeof(a_bits));
    in.seekg(static_cast<std::streamoff>(8u + n + offsets[1].first), std::ios::beg);
    uint16_t b_bits = 0;
    in.read(reinterpret_cast<char *>(&b_bits), sizeof(b_bits));
    // The first B rows deliberately differ by fused source slice and expert.
    // This catches a q/k/v or AR/NAR offset swap that a uniform fixture would
    // incorrectly accept.
    in.clear();
    in.seekg(static_cast<std::streamoff>(8u + n + offsets[2].first), std::ios::beg);
    uint16_t k_bits = 0;
    in.read(reinterpret_cast<char *>(&k_bits), sizeof(k_bits));
    in.clear();
    in.seekg(static_cast<std::streamoff>(8u + n + offsets[3].first), std::ios::beg);
    uint16_t gate_bits = 0;
    in.read(reinterpret_cast<char *>(&gate_bits), sizeof(gate_bits));
    const bool ar = std::string(format) == "ar";
    return in.good() && a_bits == (ar ? 0x3f40 : 0x3e80) &&
           b_bits == static_cast<uint16_t>((ar ? 0x42dc : 0x4120)) &&
           k_bits == static_cast<uint16_t>((ar ? 0x42f0 : 0x41a0)) &&
           gate_bits == static_cast<uint16_t>((ar ? 0x430c : 0x4220)) &&
           offsets[0].second - offsets[0].first == 2u * 2048u * 2u &&
           offsets[1].second - offsets[1].first == 2048u * 2u * 2u;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path out(argv[1]);
    std::error_code ec;
    if (std::filesystem::exists(out, ec) || !std::filesystem::create_directory(out, ec) || ec) return 2;
    std::vector<Owned> owned;
    const auto factors = inventory(2, &owned);
    const auto ar = out / "native-ar.safetensors";
    const auto nar = out / "native-nar.safetensors";
    bool ok = true;
    ok = expect(factors.size() == 448, "fused inventory has 448 factors") && ok;
    ok = expect(loader_names_accept("ar") && loader_names_accept("nar"), "native loader accepts exported site names") && ok;
    ok = expect(yue2_aitk_write_native_split(factors, 2, 2.0f, 7, ar.string().c_str(), nar.string().c_str()), "split export succeeds") && ok;
    ok = expect(check_header(ar, "ar", 28u * 7u * 2u), "AR header has native split inventory") && ok;
    ok = expect(check_header(nar, "nar", 28u * 7u * 2u), "NAR header has native split inventory") && ok;
    ok = expect(!yue2_aitk_write_native_split(factors, 2, 2.0f, 7, ar.string().c_str(), nar.string().c_str()), "existing outputs are refused") && ok;
    if (!ok) return 1;
    std::cout << "native YuE2 adapter split header checks passed; artifacts preserved in " << out << '\n';
    return 0;
}
