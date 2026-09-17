#include "../../engine/src/train/yue2-aitk-lora.h"
#include "../../engine/ggml/include/ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static ggml_context *make_context(size_t bytes) {
    ggml_init_params params{};
    params.mem_size = bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    return ggml_init(params);
}

static void fill_tensor(ggml_tensor *tensor, float seed) {
    float *data = ggml_get_data_f32(tensor);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) data[i] = seed + 0.001f * static_cast<float>(i % 17);
}

int main() {
    const Yue2AitkDims dims{8, 8, 4, 12};
    ggml_context *ctx = make_context(16u << 20);
    assert(ctx);
    Yue2AitkExpertAdapters adapters;
    assert(yue2_aitk_make_expert_adapters(ctx, dims, 2, 2, 32.0f, &adapters));
    assert(adapters.layers.size() == 2);
    assert(adapters.params.size() == 16); // four fused sites, A/B each, per layer
    assert(yue2_aitk_unique_param_count(adapters) == adapters.params.size());
    Yue2AitkExpertAdapters nar;
    assert(yue2_aitk_make_expert_adapters(ctx, dims, 2, 2, 32.0f, &nar, "nar"));
    assert(std::strcmp(ggml_get_name(adapters.params[0]), ggml_get_name(nar.params[0])) != 0);
    const auto &layer = adapters.layers[0];
    assert(layer.qkv.output_width == dims.q_width + 2 * dims.kv_width);
    assert(layer.gate_up.output_width == 2 * dims.feed_forward);
    assert(layer.qkv.a != layer.gate_up.a);
    assert(layer.qkv.b != layer.gate_up.b);
    assert(std::fabs(layer.qkv.scale - 16.0f) < 1e-6f);

    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dims.hidden, 3);
    assert(x);
    ggml_tensor *q = yue2_aitk_fused_delta(ctx, x, layer.qkv, 0, dims.q_width);
    ggml_tensor *k = yue2_aitk_fused_delta(ctx, x, layer.qkv, dims.q_width, dims.kv_width);
    ggml_tensor *v = yue2_aitk_fused_delta(ctx, x, layer.qkv, dims.q_width + dims.kv_width, dims.kv_width);
    assert(q && k && v);
    assert(q->ne[0] == dims.q_width && k->ne[0] == dims.kv_width && v->ne[0] == dims.kv_width);
    assert(q->src[0] != nullptr && k->src[0] != nullptr && v->src[0] != nullptr);

    // The three slices share the same A node. A graph backward therefore
    // accumulates their contributions into one gradient tensor, unlike three
    // independently allocated Legacy A factors.
    assert(q->src[0]->src[1]->src[0] == layer.qkv.a);
    assert(k->src[0]->src[1]->src[0] == layer.qkv.a);
    assert(v->src[0]->src[1]->src[0] == layer.qkv.a);
    assert(q->src[0]->src[0]->view_src == layer.qkv.b);
    assert(k->src[0]->src[0]->view_src == layer.qkv.b);
    assert(v->src[0]->src[0]->view_src == layer.qkv.b);

    // Compute a scalar loss through all three slices and verify that the
    // shared-A gradient equals the sum of three independent A gradients.
    fill_tensor(layer.qkv.a, 0.2f);
    fill_tensor(layer.qkv.b, -0.1f);
    fill_tensor(x, 0.3f);
    ggml_cgraph *shared_graph = ggml_new_graph_custom(ctx, 4096, true);
    ggml_tensor *shared_loss = ggml_add(ctx, ggml_add(ctx, ggml_sum(ctx, q), ggml_sum(ctx, k)), ggml_sum(ctx, v));
    ggml_set_loss(shared_loss);
    ggml_build_forward_expand(shared_graph, shared_loss);
    ggml_build_backward_expand(ctx, shared_graph, nullptr);
    ggml_graph_reset(shared_graph);
    assert(ggml_graph_compute_with_ctx(ctx, shared_graph, 1) == GGML_STATUS_SUCCESS);
    ggml_tensor *shared_grad = ggml_graph_get_grad(shared_graph, layer.qkv.a);
    assert(shared_grad && shared_grad->data);

    ggml_context *ref_ctx = make_context(16u << 20);
    ggml_tensor *ref_x = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, dims.hidden, 3);
    ggml_tensor *ref_a[3]{};
    ggml_tensor *ref_b[3]{};
    const int64_t widths[3] = {dims.q_width, dims.kv_width, dims.kv_width};
    const int64_t offsets[3] = {0, dims.q_width, dims.q_width + dims.kv_width};
    for (int i = 0; i < 3; ++i) {
        ref_a[i] = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, dims.hidden, 2);
        ref_b[i] = ggml_new_tensor_2d(ref_ctx, GGML_TYPE_F32, 2, widths[i]);
        ggml_set_param(ref_a[i]);
        ggml_set_param(ref_b[i]);
        std::memcpy(ggml_get_data_f32(ref_a[i]), ggml_get_data_f32(layer.qkv.a), ggml_nbytes(ref_a[i]));
        const float *src_b = ggml_get_data_f32(layer.qkv.b) + offsets[i] * 2;
        std::memcpy(ggml_get_data_f32(ref_b[i]), src_b, ggml_nbytes(ref_b[i]));
    }
    std::memcpy(ggml_get_data_f32(ref_x), ggml_get_data_f32(x), ggml_nbytes(ref_x));
    ggml_tensor *ref_sum = nullptr;
    for (int i = 0; i < 3; ++i) {
        ggml_tensor *ref_delta = ggml_scale(ref_ctx, ggml_mul_mat(ref_ctx, ref_b[i], ggml_mul_mat(ref_ctx, ref_a[i], ref_x)), layer.qkv.scale);
        ggml_tensor *term = ggml_sum(ref_ctx, ref_delta);
        ref_sum = ref_sum ? ggml_add(ref_ctx, ref_sum, term) : term;
    }
    ggml_cgraph *ref_graph = ggml_new_graph_custom(ref_ctx, 4096, true);
    ggml_set_loss(ref_sum);
    ggml_build_forward_expand(ref_graph, ref_sum);
    ggml_build_backward_expand(ref_ctx, ref_graph, nullptr);
    ggml_graph_reset(ref_graph);
    assert(ggml_graph_compute_with_ctx(ref_ctx, ref_graph, 1) == GGML_STATUS_SUCCESS);
    const float *shared_values = ggml_get_data_f32(ggml_graph_get_grad(shared_graph, layer.qkv.a));
    std::vector<float> reference(static_cast<size_t>(ggml_nelements(layer.qkv.a)), 0.0f);
    for (int i = 0; i < 3; ++i) {
        const float *part = ggml_get_data_f32(ggml_graph_get_grad(ref_graph, ref_a[i]));
        for (size_t j = 0; j < reference.size(); ++j) reference[j] += part[j];
    }
    for (size_t j = 0; j < reference.size(); ++j) {
        assert(std::fabs(shared_values[j] - reference[j]) < 1e-5f);
        const int64_t input_index = j % dims.hidden, rank_index = j / dims.hidden;
        float sum_b = 0.0f, sum_x = 0.0f;
        for (int64_t o = 0; o < layer.qkv.output_width; ++o)
            sum_b += ggml_get_data_f32(layer.qkv.b)[o * 2 + rank_index];
        for (int64_t t = 0; t < 3; ++t) sum_x += ggml_get_data_f32(x)[t * dims.hidden + input_index];
        const float analytic = layer.qkv.scale * sum_b * sum_x;
        assert(std::fabs(analytic) > 0.01f);
        assert(std::fabs(shared_values[j] - analytic) < 1e-4f);
    }
    ggml_free(ref_ctx);
    ggml_free(ctx);
    std::puts("fused LoRA allocation and shared-A graph checks passed");
    return 0;
}
