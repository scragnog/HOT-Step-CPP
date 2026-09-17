#include "../../engine/src/train/yue2-aitk-prepare.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace fs = std::filesystem;

static bool write_bytes(const fs::path & p, const std::vector<uint8_t> & bytes) {
    std::ofstream f(p, std::ios::binary);
    return f && (bytes.empty() || f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) && f.flush();
}
static bool write_f32(const fs::path & p, size_t frames, bool finite) {
    std::vector<uint8_t> bytes(frames * yue2_aitk::kLatentChannels * sizeof(float));
    for (size_t i = 0; i < frames * yue2_aitk::kLatentChannels; ++i) {
        float value = finite ? (1.0f + static_cast<float>(i) * 0.001f) : std::numeric_limits<float>::quiet_NaN();
        std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(value));
    }
    return write_bytes(p, bytes);
}
static bool write_i32(const fs::path & p, size_t frames, int32_t first = 7) {
    std::vector<uint8_t> bytes(frames * sizeof(int32_t));
    for (size_t i = 0; i < frames; ++i) { int32_t value = first + static_cast<int32_t>(i); std::memcpy(bytes.data() + i * sizeof(value), &value, sizeof(value)); }
    return write_bytes(p, bytes);
}
static bool no_manifest(const fs::path & dir) { return !fs::exists(dir / "dataset.json"); }

int main(int argc, char ** argv) {
    if (argc != 2) { std::cerr << "usage: test_yue2_aitk_prepare NEW_OUTPUT_DIR\n"; return 2; }
    const fs::path root = fs::u8path(argv[1]); std::error_code ec;
    if (fs::exists(root, ec) || ec || !fs::create_directories(root, ec) || ec) return 2;
    const fs::path input = root / "input"; fs::create_directories(input, ec);
    const fs::path source = input / "native.json", base = input / "base.gguf", tok = input / "tokenizer.gguf";
    const fs::path latent = input / "latent.f32", codes = input / "codes.i32";
    if (!write_bytes(source, {'{','}','\n'}) || !write_bytes(base, {1,2,3,4}) || !write_bytes(tok, {5,6,7}) || !write_f32(latent, 4, true) || !write_i32(codes, 4)) return 1;
    yue2_aitk::PrepareRequest req;
    req.output_dir = root / "prepared"; req.source_manifest = source; req.base_checkpoint = base;
    req.verified_contract = "native-yue2-preprocess-v1+tokenize-v1+prefix-v1"; req.models = {{"tokenizer", tok}};
    yue2_aitk::VerifiedNativeItem item; item.id = "fixture-0"; item.latent_file = latent; item.semantic_file = codes; item.frames = 4;
    item.prompt.retained_prefix_ids = {151643, 10, 151849}; item.prompt.dropped_prefix_ids = {151643, 11, 151849}; item.prompt.abc_ids = {101, 102}; req.items.push_back(item);
    std::string error;
    if (!yue2_aitk::prepare_dataset(req, &error)) { std::cerr << error << "\n"; return 1; }
    yue2_aitk::Dataset loaded;
    if (!yue2_aitk::read_dataset((req.output_dir / "dataset.json").u8string(), &loaded, &error) || loaded.items.size() != 1) return 1;
    yue2_aitk::sha256::digest original_hash, copied_hash;
    if (!yue2_aitk::sha256::file(source, original_hash, &error) || !yue2_aitk::sha256::file(req.output_dir / "source-manifest.json", copied_hash, &error) || original_hash.hex() != copied_hash.hex()) return 1;
    if (yue2_aitk::prepare_dataset(req, &error)) return 1;
    auto expect_reject = [&](const char * name, auto mutate) {
        auto bad = req; bad.output_dir = root / name; mutate(bad); std::string why;
        return !yue2_aitk::prepare_dataset(bad, &why) && no_manifest(bad.output_dir);
    };
    if (!expect_reject("bad-nul", [](auto & r) { r.items[0].id = std::string("bad\0id", 6); }) ||
        !expect_reject("bad-nonfinite", [](auto & r) { r.items[0].latent_file = r.source_manifest.parent_path() / "bad.f32"; write_f32(r.items[0].latent_file, 4, false); }) ||
        !expect_reject("bad-codec", [](auto & r) { r.items[0].semantic_file = r.source_manifest.parent_path() / "bad.i32"; write_i32(r.items[0].semantic_file, 4, 40000); }) ||
        !expect_reject("bad-provenance", [](auto & r) { r.models.clear(); })) return 1;
    std::cout << "prepare fixture accepted valid input and rejected malformed cases\n"; return 0;
}
