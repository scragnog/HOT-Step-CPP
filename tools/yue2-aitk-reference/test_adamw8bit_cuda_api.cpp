#include "adamw8bit_cuda.h"
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    std::ifstream in(argv[1], std::ios::binary);
    const auto fixture = aitk_adamw8bit::read_fixture(in);
    const auto &first = aitk_adamw8bit::get(fixture, "step_1_large_qmap1");
    const auto &second = aitk_adamw8bit::get(fixture, "step_1_large_qmap2");
    float source1[256], source2[256], copied1[256], copied2[256], default1[256], default2[256];
    for (int i = 0; i < 256; ++i) { source1[i] = aitk_adamw8bit::f32(first, i); source2[i] = aitk_adamw8bit::f32(second, i); }
    if (!aitk_adamw8bit::adamw8bit_initialize_reference_maps(source1, source2, copied1, copied2)) return 2;
    for (int i = 0; i < 256; ++i) if (copied1[i] != source1[i] || copied2[i] != source2[i]) return 3;
    if (!aitk_adamw8bit::adamw8bit_initialize_default_maps(default1, default2)) return 7;
    for (int i = 0; i < 256; ++i) if (default1[i] != source1[i] || default2[i] != source2[i]) return 8;
    aitk_adamw8bit::AdamW8bitCudaConfig config{}; config.elements = 4095; config.step = 1;
    aitk_adamw8bit::AdamW8bitCudaState state{}; float value = 0;
    state.parameter = &value; state.gradient = &value; state.state1_fp32 = &value; state.state2_fp32 = &value;
    if (!aitk_adamw8bit::adamw8bit_cuda_validate(config, state)) return 4;
    config.elements = 4096;
    if (aitk_adamw8bit::adamw8bit_cuda_validate(config, state)) return 5;
    uint8_t code = 0; state.state1_u8 = &code; state.state2_u8 = &code; state.absmax1 = &value; state.absmax2 = &value; state.qmap1 = copied1; state.qmap2 = copied2;
    if (!aitk_adamw8bit::adamw8bit_cuda_validate(config, state)) return 6;
    std::cout << "reference codebook copy and 4095/4096 state contract passed\n";
    return 0;
}
