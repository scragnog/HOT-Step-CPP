#include "adamw8bit_cuda.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: optimizer_cuda_probe FIXTURE.aitkopt");
        std::ifstream input(argv[1], std::ios::binary);
        const auto fixture = aitk_adamw8bit::read_fixture(input);

        aitk_adamw8bit::CudaReplayReport report{};
        const bool passed = aitk_adamw8bit::replay_cuda(fixture, 2e-5f, &report);
        std::cout << report.detail
                  << "max_abs=" << report.max_abs
                  << " total_mismatches=" << report.mismatches << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
