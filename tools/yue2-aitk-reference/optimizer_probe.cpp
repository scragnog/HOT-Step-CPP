#include "adamw8bit_cpu.h"
#include <fstream>
#include <iostream>

int main(int argc, char ** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::runtime_error("Usage: optimizer_probe FIXTURE.aitkopt [--replay]");
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error("could not open optimizer fixture");
        const auto f = aitk_adamw8bit::read_fixture(in);
        if (f.arrays.empty()) throw std::runtime_error("optimizer fixture contains no arrays");
        std::cout << "AdamW8bit fixture valid: " << f.arrays.size() << " arrays\n";
        for (const auto & a : f.arrays)
            std::cout << "  " << a.name << " (dtype " << a.dtype << ", " << a.bytes.size() << " bytes)\n";
        if (argc == 3 && std::string(argv[2]) == "--replay") {
            const double error = aitk_adamw8bit::replay_fp32(f);
            std::cout << "CPU formula replay max parameter abs error: " << error << "\n";
            std::cerr << "Diagnostic only: state/codebook quantization comparison is pending full archive assertions.\n";
            return 0;
        }
        std::cerr << "CPU AdamW8bit update unavailable: bitsandbytes CUDA kernel source is required; "
                     "no FP32 AdamW substitution was run.\n";
        return 2;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
