// A small binary fixture runner. No GPU, model loading, or external library required.
#include "convrot_cpu.h"
#include <fstream>
#include <iostream>

template<class T> std::vector<T> read_array(std::istream& input, size_t count) {
    std::vector<T> value(count);
    if (!input.read(reinterpret_cast<char*>(value.data()), count*sizeof(T)))
        throw std::runtime_error("Truncated fixture");
    return value;
}
template<class T> void write_array(std::ostream& output, const std::vector<T>& value) {
    output.write(reinterpret_cast<const char*>(value.data()), value.size()*sizeof(T));
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: convrot_probe INPUT.bin OUTPUT.bin");
        if (std::strcmp(argv[1], argv[2]) == 0) throw std::runtime_error("Input and output must differ");
        std::ifstream input(argv[1], std::ios::binary);
        const auto h = read_array<uint32_t>(input, 6);
        if (h[0] != 0x314b5441u || h[1] == 0 || h[1] > 1024 || h[2] == 0 || h[2] > 8192 ||
            h[3] == 0 || h[3] > 16384 || h[5] > 1)
            throw std::runtime_error("Invalid fixture header");
        auto x = read_array<float>(input, size_t(h[1])*h[2]);
        auto w = read_array<int8_t>(input, size_t(h[3])*h[2]);
        auto s = read_array<float>(input, h[3]);
        auto g = read_array<float>(input, size_t(h[1])*h[3]);
        auto b = read_array<float>(input, h[3]);
        if (input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing fixture data");
        const auto r = aitk_reference::convrot8(x,w,s,g,b,h[1],h[2],h[3],h[4],h[5] != 0);
        std::ifstream existing(argv[2], std::ios::binary);
        if (existing.good()) throw std::runtime_error("Output exists; choose a new path");
        std::ofstream output(argv[2], std::ios::binary);
        write_array(output,r.rotated);
        write_array(output,r.activation_codes);
        write_array(output,r.activation_scales);
        write_array(output,r.output);
        write_array(output,r.input_gradient);
        if (!output) throw std::runtime_error("Could not write output");
        std::cout << "ConvRot8 fixture completed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
