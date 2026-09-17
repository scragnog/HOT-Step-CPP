#include "../../engine/ggml/include/ggml.h"
#include "../../engine/ggml/include/ggml-alloc.h"
#include "../../engine/ggml/include/ggml-backend.h"
#include "../../engine/ggml/include/ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "../../engine/ggml/include/ggml-cuda.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static float bf16_round(float x) {
    uint32_t bits; std::memcpy(&bits, &x, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u); bits &= 0xffff0000u;
    float y; std::memcpy(&y, &bits, sizeof(y)); return y;
}

static int run(ggml_backend_t backend) {
    ggml_init_params ip{}; ip.mem_size = 8u << 20; ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip); if (!ctx) return 2;
    ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    ggml_tensor * x = ggml_view_2d(ctx, storage, 4, 2, storage->nb[1], 0);
    ggml_tensor * dy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_param(storage);
    ggml_tensor * rounded = ggml_bf16_round(ctx, x);
    ggml_tensor * branch_dy = ggml_add(ctx, dy, dy);
    ggml_tensor * loss = ggml_sum(ctx, ggml_mul(ctx, rounded, branch_dy));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss); ggml_set_loss(loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
            ggml_free(ctx); return 9;
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return 3; }
    const float x_data[16] = {0.0f, 1.0f, -1.0f, 0.003913879f, 1.00390625f, -1.00390625f, 0.5f, -0.5f,
                              0.25f, -0.25f, 2.0f, -2.0f, 0.75f, -0.75f, 4.0f, -4.0f};
    const float dy_data[8] = {1.0f, 1.00390625f, -1.00390625f, 0.003913879f, 2.0f, -2.0f, 0.5f, -0.5f};
    ggml_backend_tensor_set(storage, x_data, 0, ggml_nbytes(storage));
    ggml_backend_tensor_set(dy, dy_data, 0, ggml_nbytes(dy));
    ggml_graph_reset(graph);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 4;
    std::vector<float> got(8), got_grad(16); ggml_backend_tensor_get(rounded, got.data(), 0, ggml_nbytes(rounded));
    ggml_tensor * grad = ggml_graph_get_grad(graph, storage);
    if (!grad) return 5; ggml_backend_tensor_get(grad, got_grad.data(), 0, ggml_nbytes(grad));
    const float expected_x[8] = {x_data[0], x_data[1], x_data[2], x_data[3], x_data[8], x_data[9], x_data[10], x_data[11]};
    for (size_t i = 0; i < got.size(); ++i) {
        const size_t storage_i = (i / 4) * 8 + i % 4;
        if (got[i] != bf16_round(expected_x[i]) || got_grad[storage_i] != bf16_round(2.0f * dy_data[i])) return 6;
    }
    for (size_t i = 4; i < 8; ++i) if (got_grad[i] != 0.0f) return 6;
    for (size_t i = 12; i < 16; ++i) if (got_grad[i] != 0.0f) return 6;
    ggml_backend_buffer_free(buffer); ggml_free(ctx); std::puts("bf16 boundary graph forward/backward OK"); return 0;
}

int main(int argc, char ** argv) {
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    ggml_backend_t backend = cuda ? nullptr : ggml_backend_cpu_init();
#ifdef GGML_USE_CUDA
    if (cuda) backend = ggml_backend_cuda_init(0);
#else
    if (cuda) { std::fputs("CUDA backend not compiled", stderr); return 7; }
#endif
    if (!backend) return 8; const int rc = run(backend); ggml_backend_free(backend); return rc;
}
