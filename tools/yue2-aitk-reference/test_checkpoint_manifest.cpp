#include "../../engine/src/train/yue2-aitk-checkpoint.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

namespace {

void write_fixture(const std::filesystem::path & path, const std::string & json,
                   const std::vector<uint8_t> & payload) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint64_t n = json.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    out.write(reinterpret_cast<const char *>(payload.data()),
              static_cast<std::streamsize>(payload.size()));
    assert(out.good());
}

void expect_reject(const std::filesystem::path & path) {
    Yue2AitkCheckpoint checkpoint;
    std::string error;
    assert(!checkpoint.open(path.string().c_str(), &error));
    assert(!error.empty());
}

void expect_reject_contains(const std::filesystem::path & path, const char * text) {
    Yue2AitkCheckpoint checkpoint;
    std::string error;
    assert(!checkpoint.open(path.string().c_str(), &error));
    assert(error.find(text) != std::string::npos);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_checkpoint_manifest NEW_OUTPUT_DIRECTORY [CHECKPOINT]\n";
        return 2;
    }
    const auto root = std::filesystem::path(argv[1]);
    if (std::filesystem::exists(root)) {
        std::cerr << "refusing existing output directory: " << root.string() << "\n";
        return 2;
    }
    std::error_code ec;
    if (!std::filesystem::create_directories(root, ec) || ec) {
        std::cerr << "cannot create output directory: " << root.string() << "\n";
        return 2;
    }

    write_fixture(root / "missing.safetensors",
                  R"({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}})", {0});
    expect_reject(root / "missing.safetensors");

    write_fixture(root / "size.safetensors",
                  R"({"x":{"dtype":"I8","shape":[2],"data_offsets":[0,1]}})", {0});
    expect_reject_contains(root / "size.safetensors", "byte size mismatch");

    write_fixture(root / "overlap.safetensors",
                  R"({"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}})", {0});
    expect_reject_contains(root / "overlap.safetensors", "overlapping tensor offsets");

    write_fixture(root / "rank-five.safetensors",
                  R"({"x":{"dtype":"I8","shape":[1,1,1,1,1],"data_offsets":[0,1]}})", {0});
    expect_reject_contains(root / "rank-five.safetensors", "rank exceeds four");
    write_fixture(root / "duplicate-field.safetensors",
                  R"({"x":{"dtype":"I8","dtype":"U8","shape":[1],"data_offsets":[0,1]}})", {0});
    expect_reject_contains(root / "duplicate-field.safetensors", "duplicate tensor field");
    write_fixture(root / "negative-offset.safetensors",
                  R"({"x":{"dtype":"I8","shape":[1],"data_offsets":[-1,0]}})", {0});
    expect_reject_contains(root / "negative-offset.safetensors", "invalid data_offsets");
    write_fixture(root / "unknown-field.safetensors",
                  R"({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1],"unexpected":true}})", {0});
    expect_reject_contains(root / "unknown-field.safetensors", "unknown tensor field");

    {
        std::ofstream out(root / "overflow-header.safetensors", std::ios::binary);
        const uint64_t n = (std::numeric_limits<uint64_t>::max)();
        out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    }
    expect_reject_contains(root / "overflow-header.safetensors", "header length exceeds file");

    {
        std::ofstream out(root / "truncated-json.safetensors", std::ios::binary);
        const uint64_t n = 4;
        out.write(reinterpret_cast<const char *>(&n), sizeof(n));
        out.write("{\"x", 3);
        out.put(' ');
    }
    expect_reject_contains(root / "truncated-json.safetensors", "invalid safetensors header JSON");

    write_fixture(root / "duplicate-name.safetensors",
                  R"({"x":{"dtype":"I8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"I8","shape":[1],"data_offsets":[1,2]}})", {0, 0});
    {
        Yue2AitkCheckpoint checkpoint;
        std::string error;
        assert(!checkpoint.open((root / "duplicate-name.safetensors").string().c_str(), &error));
        assert(error.find("duplicate") != std::string::npos || error.find("invalid") != std::string::npos);
    }

    if (argc == 3) {
        Yue2AitkCheckpoint checkpoint;
        std::string error;
        if (!checkpoint.open(argv[2], &error)) {
            std::cerr << "real checkpoint rejected: " << error << "\n";
            return 1;
        }
        assert(checkpoint.records().size() == 229);
        std::cout << "real checkpoint: " << checkpoint.records().size()
                  << " ConvRot records\n";
    }
    std::cout << "checkpoint manifest malformed-fixture tests passed; outputs preserved at "
              << root.string() << "\n";
    return 0;
}
