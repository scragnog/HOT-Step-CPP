#include "../../engine/src/train/yue2-aitk-embedding-dequant.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x324D4245; // EBM2

template <typename T>
bool read_exact(std::ifstream &input, T *values, size_t count) {
    input.read(reinterpret_cast<char *>(values), static_cast<std::streamsize>(sizeof(T) * count));
    return input.good() || input.gcount() == static_cast<std::streamsize>(sizeof(T) * count);
}
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: yue2_aitk_embedding_probe.exe INPUT OUTPUT\n";
        return 2;
    }
    const std::filesystem::path input_path(argv[1]);
    const std::filesystem::path output_path(argv[2]);
    if (std::filesystem::exists(output_path)) {
        std::cerr << "refusing existing output\n";
        return 2;
    }
    std::ifstream input(input_path, std::ios::binary);
    if (!input) return 2;
    uint32_t header[6]{};
    if (!read_exact(input, header, 6) || header[0] != kMagic || header[1] == 0 || header[2] == 0 ||
        header[3] == 0 || header[4] >= header[1] || header[5] == 0 ||
        static_cast<uint64_t>(header[4]) + header[5] > header[1]) {
        return 2;
    }
    const int64_t rows = header[1], cols = header[2], first_row = header[4], count = header[5];
    if (cols > (1u << 20) || static_cast<uint64_t>(rows) * cols > (1ull << 31)) return 2;
    std::vector<int8_t> qdata(static_cast<size_t>(rows) * cols);
    std::vector<float> scales(rows);
    if (!read_exact(input, qdata.data(), qdata.size()) || !read_exact(input, scales.data(), scales.size())) return 2;
    input.peek();
    if (!input.eof()) return 2;
    std::vector<uint16_t> output(static_cast<size_t>(count) * cols);
    const Yue2AitkConvRotEmbeddingView view{qdata.data(), scales.data(), rows, cols, static_cast<int>(header[3])};
    if (!yue2_aitk_dequantize_embedding_rows_bf16(view, first_row, count, output.data(), output.size())) return 2;
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) return 2;
    out.write(reinterpret_cast<const char *>(output.data()),
              static_cast<std::streamsize>(output.size() * sizeof(uint16_t)));
    return out.good() ? 0 : 1;
}
