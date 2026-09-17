#include "../../engine/src/train/yue2-aitk-native-import.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool bytes(const fs::path & p, const std::string & s) {
    std::ofstream f(p, std::ios::binary); return f && bool(f.write(s.data(), static_cast<std::streamsize>(s.size()))) && bool(f.flush());
}
static bool latent(const fs::path & p, size_t n) {
    std::ofstream f(p, std::ios::binary); for (size_t i = 0; i < n * 64; ++i) { float v = 0.01f * float(i + 1); if (!f.write(reinterpret_cast<const char *>(&v), sizeof(v))) return false; } return bool(f.flush());
}
static bool codes(const fs::path & p, size_t n) {
    std::ofstream f(p, std::ios::binary); for (size_t i = 0; i < n; ++i) { int32_t v = int32_t(i + 7); if (!f.write(reinterpret_cast<const char *>(&v), sizeof(v))) return false; } return bool(f.flush());
}
static std::string manifest(const std::string & latent_name = "latents/item.f32", const std::string & code_name = "codes/item.i32",
                            const std::string & fps = "25.0", const std::string & dim = "64", const std::string & abc = "K:C\\nC D E") {
    return std::string("{\"format\":\"yue2-preprocess-v1\",\"caption_format\":\"ace-sidecar\",\"frame_rate\":") + fps
        + ",\"latent_dim\":" + dim + ",\"latent_layout\":\"frame_major\",\"latent_index\":\"t * latent_dim + c\",\"latent_dtype\":\"f32\",\"sources\":[{\"name\":\"item\",\"source\":\"item.wav\",\"caption\":\"bright\",\"lyrics\":\"[VERSE]\\nhello\",\"latents\":\"" + latent_name
        + "\",\"codec_ids\":\"" + code_name + "\",\"frames\":4,\"abc\":\"" + abc + "\"}]}\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "usage: test_yue2_aitk_native_import NEW_OUTPUT_ROOT TOKENIZER_GGUF_OR_DIR\n"; return 2; }
    const fs::path root = fs::u8path(argv[1]), input = root / "input"; std::error_code ec;
    if (fs::exists(root, ec) || ec || !fs::create_directories(input / "latents", ec) || ec || !fs::create_directories(input / "codes", ec) || ec) return 2;
    const fs::path source = input / "yue2_preprocess.json", latent_path = input / "latents/item.f32", code_path = input / "codes/item.i32";
    const fs::path base = input / "rawConvRot.safetensors", model = input / "sheetsage.gguf";
    if (!latent(latent_path, 4) || !codes(code_path, 4) || !bytes(base, "base") || !bytes(model, "model") || !bytes(input / "item.wav", "audio") ||
        !bytes(source, manifest())) return 1;
    yue2_aitk_native_import::Request req;
    req.legacy_manifest = source; req.raw_convrot_checkpoint = base; req.tokenizer_gguf_or_dir = fs::u8path(argv[2]); req.output_dir = root / "valid";
    req.models = {{"vae", model}, {"semantic", model}, {"sheetsage", model}};
    std::string error;
    if (!yue2_aitk_native_import::prepare_from_legacy(req, &error)) { std::cerr << "valid rejected: " << error << "\n"; return 1; }
    if (!fs::exists(req.output_dir / "dataset.json") || yue2_aitk_native_import::prepare_from_legacy(req, &error)) return 1;
    yue2_aitk::Dataset prepared;
    if (!yue2_aitk::read_dataset((req.output_dir / "dataset.json").u8string(), &prepared, &error)) {
        std::cerr << "output dataset rejected: " << error << '\n'; return 1;
    }
    BPETokenizer tokenizer{};
    const bool loaded = req.tokenizer_gguf_or_dir.extension() == ".gguf"
        ? yue2_tokenizer_load_from_gguf(&tokenizer, req.tokenizer_gguf_or_dir.u8string())
        : yue2_tokenizer_load_from_dir(&tokenizer, req.tokenizer_gguf_or_dir.u8string());
    if (!loaded) return 1;
    const auto normalized = yue2_token_prefixes(&tokenizer, "bright", "[Verse]\nhello", YUE2_COT_FULL, nullptr);
    const std::vector<int32_t> expected(normalized.begin(), normalized.end());
    if (prepared.items.size() != 1 || prepared.items[0].prompt.retained_prefix_ids != expected) {
        std::cerr << "caption section normalization mismatch\n"; return 1;
    }
    auto reject = [&](const char * name, const std::string & json) {
        const fs::path m = input / (std::string(name) + ".json"), out = root / name;
        if (!bytes(m, json)) return false; auto bad = req; bad.legacy_manifest = m; bad.output_dir = out;
        return !yue2_aitk_native_import::prepare_from_legacy(bad, &error) && !fs::exists(out / "dataset.json");
    };
    if (!reject("bad-fps", manifest("latents/item.f32", "codes/item.i32", "24.0")) ||
        !reject("bad-dim", manifest("latents/item.f32", "codes/item.i32", "25.0", "128")) ||
        !reject("bad-traversal", manifest("../item.f32")) ||
        !reject("bad-abc", manifest("latents/item.f32", "codes/item.i32", "25.0", "64", ""))) return 1;
    std::cout << "native raw import fixture accepted valid input and rejected malformed cases\n"; return 0;
}
