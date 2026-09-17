#pragma once
// Scalar oracle for the ConvRot8 CUDA training contract, not a production backend.
// Source contract: ai-toolkit e65c4d0, toolkit/util/convrot_quant.py.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "../../engine/src/convrot.h"

namespace aitk_reference {
inline float bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits |= 0x00400000u;
    } else {
        bits += 0x7fffu + ((bits >> 16) & 1u);
    }
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

inline int round_even(float x) {
    // Independent of the process floating-point rounding mode.
    float lo = std::floor(x);
    const float fraction = x - lo;
    int result = static_cast<int>(lo);
    if (fraction > 0.5f || (fraction == 0.5f && result % 2 != 0)) ++result;
    return result;
}

struct ConvRotResult {
    std::vector<float> rotated, activation_scales, output, input_gradient;
    std::vector<int8_t> activation_codes;
};

inline ConvRotResult convrot8(const std::vector<float>& input,
                             const std::vector<int8_t>& weights,
                             const std::vector<float>& weight_scales,
                             const std::vector<float>& gradient,
                             const std::vector<float>& bias,
                             int rows, int in, int out, int rotation, bool use_bf16) {
    if (rows <= 0 || in <= 0 || out <= 0 || in > 65536 ||
        (rotation != 1 && !convrot_is_pow4(rotation)) || rotation > 4096 ||
        in % rotation != 0 || input.size() != size_t(rows) * in ||
        weights.size() != size_t(out) * in || weight_scales.size() != size_t(out) ||
        gradient.size() != size_t(rows) * out || bias.size() != size_t(out))
        throw std::invalid_argument("Invalid ConvRot8 fixture shape");
    const auto cast = [use_bf16](float x) { return use_bf16 ? bf16(x) : x; };
    ConvRotResult r;
    r.rotated = input;
    for (float& x : r.rotated) {
        if (!std::isfinite(x)) throw std::invalid_argument("Nonfinite activation");
        x = cast(x);
    }
    if (rotation != 1) convrot_transform_rows(r.rotated.data(), rows, in, rotation);
    for (float& x : r.rotated) x = cast(x);
    r.activation_scales.resize(rows);
    r.activation_codes.resize(input.size());
    for (int m = 0; m < rows; ++m) {
        float amax = 0;
        for (int k = 0; k < in; ++k) amax = std::max(amax, std::fabs(r.rotated[size_t(m)*in+k]));
        const float scale = amax > 0 ? amax / 127.0f : 1.0f;
        r.activation_scales[m] = scale;
        for (int k = 0; k < in; ++k)
            r.activation_codes[size_t(m)*in+k] = static_cast<int8_t>(
                std::clamp(round_even(r.rotated[size_t(m)*in+k]/scale), -127, 127));
    }
    r.output.resize(size_t(rows)*out);
    for (int m = 0; m < rows; ++m) {
        for (int n = 0; n < out; ++n) {
            int32_t sum = 0;
            for (int k = 0; k < in; ++k)
                sum += int(r.activation_codes[size_t(m)*in+k]) * int(weights[size_t(n)*in+k]);
            // Triton CUDA epilogue order: integer accumulator * (activation scale * weight scale).
            float y = float(sum) * (r.activation_scales[m] * weight_scales[n]);
            r.output[size_t(m)*out+n] = cast(y + cast(bias[n]));
        }
    }
    // STE backward casts scales before multiplying codes, then casts the product.
    // This differs from casting one FP32 dequantized weight at the end.
    r.input_gradient.resize(input.size());
    for (int m = 0; m < rows; ++m) {
        for (int k = 0; k < in; ++k) {
            float sum = 0;
            for (int n = 0; n < out; ++n) {
                const float w = cast(float(weights[size_t(n)*in+k]) * cast(weight_scales[n]));
                sum += cast(gradient[size_t(m)*out+n]) * w;
            }
            r.input_gradient[size_t(m)*in+k] = cast(sum);
        }
    }
    if (rotation != 1) convrot_transform_rows(r.input_gradient.data(), rows, in, rotation);
    for (float& x : r.input_gradient) x = cast(x);
    return r;
}
} // namespace aitk_reference
