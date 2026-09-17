#include "../../engine/src/train/yue2-aitk-adapter-io.h"
#include "../../engine/vendor/yyjson/yyjson.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Case {
    const char *expert;
    int         layer;
    const char *site;
    bool        a;
};

std::vector<Yue2AitkF32Matrix> complete_inventory(int64_t rank, std::vector<float> *storage) {
    storage->assign(393216, 1.004f); // rounds upward in BF16 (largest factor: gate/up B [12288, 32])
    std::vector<Yue2AitkF32Matrix> out;
    out.reserve(448);
    for (const char *expert : {"diffusion_model", "text_encoders"}) {
        for (int layer = 0; layer < 28; ++layer) {
            for (const char *site : {"self_attn.qkv_proj", "self_attn.o_proj",
                                     "mlp.gate_up_proj", "mlp.down_proj"}) {
                const int64_t input = std::string(site) == "mlp.down_proj" ? 6144 : 2048;
                const int64_t output = std::string(site) == "self_attn.qkv_proj" ? 4096 :
                                       std::string(site) == "mlp.gate_up_proj" ? 12288 : 2048;
                const std::string prefix = std::string(expert) + ".model.layers." +
                                            std::to_string(layer) + "." + site + ".lora_";
                out.push_back({prefix + "A.weight", rank, input, storage->data()});
                out.push_back({prefix + "B.weight", output, rank, storage->data()});
            }
        }
    }
    return out;
}

bool check(bool condition, const char *message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

bool rejected(const std::vector<Yue2AitkF32Matrix> &factors, const char *path) {
    return !yue2_aitk_write_fused_lora(factors, 32, 32.0f, 250, path);
}

uint64_t read_u64_le(const unsigned char *p) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; --i) result = (result << 8) | p[i];
    return result;
}

bool verify_file(const std::filesystem::path &path,
                 const std::vector<Yue2AitkF32Matrix> &factors) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    unsigned char length_bytes[8]{};
    in.read(reinterpret_cast<char *>(length_bytes), sizeof(length_bytes));
    if (in.gcount() != 8) return false;
    const uint64_t header_size = read_u64_le(length_bytes);
    std::string header(static_cast<size_t>(header_size), '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (in.gcount() != static_cast<std::streamsize>(header.size())) return false;
    yyjson_doc *doc = yyjson_read(header.data(), header.size(), 0);
    if (!doc) return false;
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *metadata = yyjson_obj_get(root, "__metadata__");
    const auto metadata_is = [&](const char *key, const char *expected) {
        yyjson_val *value = metadata ? yyjson_obj_get(metadata, key) : nullptr;
        return value && yyjson_is_str(value) && std::string(yyjson_get_str(value)) == expected;
    };
    bool parsed = yyjson_is_obj(root) && yyjson_is_obj(metadata) &&
                  metadata_is("format", "yue2-aitk-fused-lora-v1") &&
                  metadata_is("rank", "32") && metadata_is("alpha", "32") &&
                  metadata_is("steps", "250");
    uint64_t payload_bytes = 0;
    uint64_t cursor = 0;
    for (const Yue2AitkF32Matrix &factor : factors) {
        yyjson_val *tensor = yyjson_obj_get(root, factor.name.c_str());
        yyjson_val *shape = tensor ? yyjson_obj_get(tensor, "shape") : nullptr;
        yyjson_val *offsets = tensor ? yyjson_obj_get(tensor, "data_offsets") : nullptr;
        yyjson_val *dtype = tensor ? yyjson_obj_get(tensor, "dtype") : nullptr;
        const uint64_t bytes = static_cast<uint64_t>(factor.rows) * static_cast<uint64_t>(factor.cols) * 2u;
        parsed = parsed && tensor && yyjson_is_obj(tensor) && yyjson_is_str(dtype) &&
                 std::string(yyjson_get_str(dtype)) == "BF16" && shape && yyjson_is_arr(shape) &&
                 yyjson_arr_size(shape) == 2 && yyjson_get_sint(yyjson_arr_get(shape, 0)) == factor.rows &&
                 yyjson_get_sint(yyjson_arr_get(shape, 1)) == factor.cols && offsets &&
                 yyjson_is_arr(offsets) && yyjson_arr_size(offsets) == 2 &&
                 yyjson_get_uint(yyjson_arr_get(offsets, 0)) == cursor &&
                 yyjson_get_uint(yyjson_arr_get(offsets, 1)) == cursor + bytes;
        cursor += bytes;
        payload_bytes += static_cast<uint64_t>(factor.rows) * static_cast<uint64_t>(factor.cols) * 2u;
    }
    parsed = parsed && yyjson_obj_size(root) == factors.size() + 1;
    const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(path));
    yyjson_doc_free(doc);
    if (!parsed || file_size != 8u + header_size + payload_bytes) return false;
    unsigned char first_bf16[2]{};
    in.read(reinterpret_cast<char *>(first_bf16), 2);
    // 1.004f rounds to 1.0078125 in BF16: the fixed little-endian bytes 80 3f.
    return in.gcount() == 2 && first_bf16[0] == 0x81 && first_bf16[1] == 0x3f;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: test_adapter_io.exe NEW_OUTPUT_DIRECTORY\n";
        return 2;
    }
    const std::filesystem::path output_dir(argv[1]);
    std::error_code ec;
    if (std::filesystem::exists(output_dir, ec) ||
        !std::filesystem::create_directory(output_dir, ec) || ec) {
        std::cerr << "refusing non-new output directory: " << output_dir << '\n';
        return 2;
    }
    const std::filesystem::path output = output_dir / "adapter.safetensors";

    std::vector<float> storage;
    std::vector<Yue2AitkF32Matrix> valid = complete_inventory(32, &storage);
    bool ok = true;
    ok = check(valid.size() == 448, "complete inventory has 448 tensors") && ok;
    ok = check(yue2_aitk_write_fused_lora(valid, 32, 32.0f, 250, output.string().c_str()),
               "valid complete inventory writes") && ok;
    ok = check(verify_file(output, valid), "output is a complete BF16 safetensors file") && ok;
    ok = check(!yue2_aitk_write_fused_lora(valid, 32, 32.0f, 250, output.string().c_str()),
               "existing output is refused") && ok;

    const std::filesystem::path bad = output.string() + ".bad";
    auto missing = valid;
    missing.pop_back();
    ok = check(rejected(missing, bad.string().c_str()), "missing tensor is rejected") && ok;
    auto duplicate = valid;
    duplicate.back().name = duplicate.front().name;
    ok = check(rejected(duplicate, bad.string().c_str()), "duplicate tensor is rejected") && ok;
    auto bad_dims = valid;
    bad_dims.front().cols = 2047;
    ok = check(rejected(bad_dims, bad.string().c_str()), "invalid dimensions are rejected") && ok;
    auto bad_name = valid;
    bad_name.front().name += ".foreign";
    ok = check(rejected(bad_name, bad.string().c_str()), "foreign or missing name is rejected") && ok;
    auto null_data = valid;
    null_data.front().data = nullptr;
    ok = check(rejected(null_data, bad.string().c_str()), "null data is rejected") && ok;
    ok = check(!std::filesystem::exists(bad), "rejected writes create no bad output") && ok;
    { std::ofstream sentinel(bad, std::ios::binary); sentinel << "keep"; }
    ok = check(rejected(valid, bad.string().c_str()), "pre-existing bad output is rejected") && ok;

    if (!ok) return 1;
    std::cout << "YuE2 AI Toolkit adapter I/O contract checks passed; artifacts preserved in "
              << output_dir << '\n';
    return 0;
}
