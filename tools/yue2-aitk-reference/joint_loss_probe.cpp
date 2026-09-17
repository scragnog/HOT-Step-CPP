#include "joint_loss_cpu.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <limits>

static bool read_u64(std::ifstream &f, uint64_t &x) { x = 0; f.read(reinterpret_cast<char *>(&x), sizeof x); return bool(f); }

int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "usage: joint_loss_probe INPUT OUTPUT\n"; return 2; }
    if (std::filesystem::exists(argv[2])) { std::cerr << "refusing to overwrite output\n"; return 2; }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) { std::cerr << "cannot open input\n"; return 2; }
    uint64_t n = 0, v = 0, chunk = 0;
    if (!read_u64(in, n) || !read_u64(in, v) || !read_u64(in, chunk) || !n || !v || !chunk ||
        n > std::numeric_limits<size_t>::max() || v > std::numeric_limits<size_t>::max() ||
        n > std::numeric_limits<size_t>::max() / v || n * v > 1000000000ULL) {
        std::cerr << "invalid or truncated dimensions\n"; return 2;
    }
    const size_t elements = static_cast<size_t>(n * v);
    const uint64_t expected = 24ULL + n * v * sizeof(float) * 2ULL + n * sizeof(uint32_t);
    if (expected < 24 || std::filesystem::file_size(argv[1]) < expected) { std::cerr << "truncated payload\n"; return 2; }
    std::vector<float> adapted(elements), base(elements); std::vector<unsigned> targets(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(adapted.data()), adapted.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(base.data()), base.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(targets.data()), targets.size() * sizeof(unsigned));
    if (!in) { std::cerr << "truncated input\n"; return 2; }
    try {
        const auto result = yue2_joint_loss::loss_and_gradient(adapted, base, targets, static_cast<size_t>(n), static_cast<size_t>(v), static_cast<size_t>(chunk));
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<const char *>(&result.ce), sizeof result.ce);
        out.write(reinterpret_cast<const char *>(&result.kl), sizeof result.kl);
        out.write(reinterpret_cast<const char *>(result.gradient.data()), result.gradient.size() * sizeof(float));
        return out ? 0 : 2;
    } catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 3; }
}
