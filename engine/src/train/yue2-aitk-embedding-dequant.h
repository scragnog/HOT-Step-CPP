#pragma once

// Bounded CPU dequantization for YuE2 ConvRot embedding rows. The checkpoint
// stores rotated int8 rows and one FP32 scale per row. Each selected row is
// dequantized, unrotated with the regular-Hadamard convention, and emitted as
// BF16. Scratch is one FP32 row, so callers can process arbitrarily large
// embedding tables in row chunks without a full-table transient.

#include "st-write.h"
#include "yue2-aitk-checkpoint.h"
#include "../../ggml/include/ggml-backend.h"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <vector>

struct Yue2AitkConvRotEmbeddingView {
    const int8_t *qdata = nullptr; // row-major [rows, cols], rotated basis
    const float   *scales = nullptr; // one FP32 scale per row
    int64_t       rows = 0;
    int64_t       cols = 0;
    int          rotation = 256;
};

namespace yue2_aitk_embedding_detail {

inline bool is_pow4(int value) {
    if (value < 4) return false;
    while (value > 1) {
        if (value % 4 != 0) return false;
        value /= 4;
    }
    return true;
}

inline void transform_group(float *values, int size) {
    for (int stride = 1; stride < size; stride *= 4) {
        const int block = stride * 4;
        for (int base = 0; base < size; base += block) {
            for (int offset = 0; offset < stride; ++offset) {
                float *p = values + base + offset;
                const float x0 = p[0], x1 = p[stride], x2 = p[2 * stride], x3 = p[3 * stride];
                p[0] = x0 + x1 + x2 - x3;
                p[stride] = x0 + x1 - x2 + x3;
                p[2 * stride] = x0 - x1 + x2 + x3;
                p[3 * stride] = -x0 + x1 + x2 + x3;
            }
        }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(size));
    for (int i = 0; i < size; ++i) values[i] *= norm;
}

inline void transform_rows(float *values, int64_t rows, int64_t cols, int rotation) {
    for (int64_t row = 0; row < rows; ++row) {
        float *data = values + row * cols;
        for (int64_t col = 0; col < cols; col += rotation)
            transform_group(data + col, rotation);
    }
}

inline bool fail(std::string *error, const char *message) {
    if (error) *error = message;
    return false;
}

} // namespace yue2_aitk_embedding_detail

// `out_bf16` receives count * view.cols BF16 words in row-major order. Only
// [first_row, first_row + count) is read; no output is produced on invalid
// arguments. The caller owns both buffers and keeps them live for the call.
inline bool yue2_aitk_dequantize_embedding_rows_bf16(
    const Yue2AitkConvRotEmbeddingView &view,
    int64_t first_row,
    int64_t count,
    uint16_t *out_bf16,
    size_t out_words) {
    if (!view.qdata || !view.scales || !out_bf16 || view.rows <= 0 || view.cols <= 0 ||
        view.cols > 1 << 20 || first_row < 0 || count <= 0 || first_row > view.rows - count ||
        !yue2_aitk_embedding_detail::is_pow4(view.rotation) || view.rotation > 4096 ||
        view.cols % view.rotation != 0 || static_cast<uint64_t>(view.rows) >
            static_cast<uint64_t>(SIZE_MAX) / static_cast<uint64_t>(view.cols) ||
        static_cast<uint64_t>(count) > static_cast<uint64_t>(out_words) /
            static_cast<uint64_t>(view.cols)) {
        return false;
    }
    for (int64_t row = 0; row < count; ++row) {
        if (!std::isfinite(view.scales[first_row + row])) return false;
    }
    std::vector<float> scratch(static_cast<size_t>(view.cols));
    for (int64_t row = 0; row < count; ++row) {
        const int64_t source_row = first_row + row;
        const float scale = view.scales[source_row];
        const int8_t *source = view.qdata + source_row * view.cols;
        for (int64_t col = 0; col < view.cols; ++col) {
            scratch[static_cast<size_t>(col)] = static_cast<float>(source[col]) * scale;
        }
        yue2_aitk_embedding_detail::transform_rows(scratch.data(), 1, view.cols, view.rotation);
        uint16_t *destination = out_bf16 + row * view.cols;
        for (int64_t col = 0; col < view.cols; ++col) {
            destination[col] = stw_f32_to_bf16(scratch[static_cast<size_t>(col)]);
        }
    }
    return true;
}

// Yue2AitkModel callback: preserve bounded host memory while filling the
// already-allocated BF16 embedding tensor in backend-visible row chunks.
inline bool yue2_aitk_load_embedding_bf16(
    const Yue2AitkCheckpointRecord &record,
    ggml_tensor *embedding,
    ggml_backend_t backend,
    std::string *error = nullptr) {
    if (!backend || !embedding || record.name != "text_encoders.model.embed_tokens" ||
        record.rows != 184704 || record.cols != 2048 || !record.weight_i8 ||
        !record.row_scales_f32 || embedding->type != GGML_TYPE_BF16 ||
        embedding->ne[0] != record.cols || embedding->ne[1] != record.rows) {
        return yue2_aitk_embedding_detail::fail(error, "invalid YuE2 embedding callback contract");
    }
    constexpr int64_t kChunkRows = 256;
    std::vector<uint16_t> chunk(static_cast<size_t>(kChunkRows) * record.cols);
    const Yue2AitkConvRotEmbeddingView view{record.weight_i8, record.row_scales_f32,
                                            record.rows, record.cols, 256};
    for (int64_t first = 0; first < record.rows; first += kChunkRows) {
        const int64_t count = (std::min)(kChunkRows, record.rows - first);
        if (!yue2_aitk_dequantize_embedding_rows_bf16(view, first, count,
                                                       chunk.data(), chunk.size())) {
            return yue2_aitk_embedding_detail::fail(error, "invalid YuE2 embedding payload");
        }
        ggml_backend_tensor_set(embedding, chunk.data(),
                                 static_cast<size_t>(first) * record.cols * sizeof(uint16_t),
                                 static_cast<size_t>(count) * record.cols * sizeof(uint16_t));
    }
    return true;
}
