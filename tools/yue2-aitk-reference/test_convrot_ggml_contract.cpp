// CPU-only contract test for the GGML ConvRot8 graph integration.
//
// This deliberately does not execute ConvRot arithmetic.  It checks graph
// metadata, source retention, autodiff ownership, and CPU capability routing.
// The --invalid-rotation, --noncontiguous, and --trainable-frozen modes are
// negative subprocess modes: each is expected to terminate in GGML_ASSERT.
//
// Validation command (after the ConvRot patch has been applied and GGML CPU
// libraries are available):
//   test_convrot_ggml_contract.exe
// Expected result: `convrot ggml contract: PASS`.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#ifdef _WIN32
#include <crtdbg.h>
#endif

namespace {

const char * expected_assertion = nullptr;
void assertion_result(const char * message) {
    std::fprintf(stderr, "%s\n", message);
    // Exit directly from GGML's callback: negative tests need no crash dialog.
    std::exit(expected_assertion && std::strstr(message, expected_assertion) ? 86 : 98);
}

struct Fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * scales = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * dy = nullptr;
};

Fixture make_fixture(bool with_bias) {
    ggml_init_params p = {};
    p.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, true) + 4096;
    p.no_alloc = true;

    Fixture f;
    f.ctx = ggml_init(p);
    if (!f.ctx) return f;
    f.weight = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I8, 16, 8);
    f.x = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 16, 3);
    f.scales = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 8);
    f.bias = with_bias ? ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 8) : nullptr;
    f.dy = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 8, 3);
    return f;
}

bool check(bool value, const char * what) {
    if (!value) std::fprintf(stderr, "contract failure: %s\n", what);
    return value;
}

int32_t op_i32(const ggml_tensor * tensor, int index) {
    int32_t value = 0;
    std::memcpy(&value, tensor->op_params + index * (int) sizeof(int32_t), sizeof(value));
    return value;
}

bool positive_contract() {
    Fixture f = make_fixture(true);
    if (!check(f.ctx && f.weight && f.x && f.scales && f.bias && f.dy, "fixture allocation")) return false;

    ggml_tensor * forward = ggml_convrot8(f.ctx, f.weight, f.x, f.scales, f.bias, 4, false);
    if (!check(forward && forward->op == GGML_OP_CONVROT8, "forward op")) return false;
    if (!check(forward->type == GGML_TYPE_F32 && forward->ne[0] == 8 && forward->ne[1] == 3,
               "forward shape/type")) return false;
    if (!check(forward->src[0] == f.weight && forward->src[1] == f.x &&
               forward->src[2] == f.scales && forward->src[3] == f.bias,
               "forward source slots")) return false;
    if (!check(op_i32(forward, 0) == 4 && op_i32(forward, 1) == 0,
               "forward op params")) return false;

    ggml_tensor * back = ggml_convrot8_back(f.ctx, f.dy, forward);
    if (!check(back && back->op == GGML_OP_CONVROT8_BACK, "back op")) return false;
    if (!check(back->type == GGML_TYPE_F32 && back->ne[0] == 16 && back->ne[1] == 3,
               "back shape/type")) return false;
    if (!check(back->src[0] == f.weight && back->src[1] == f.dy && back->src[2] == f.scales &&
               back->src[3] == nullptr && back->src[4] == nullptr,
               "back retains weight/dy/scales only")) return false;
    if (!check(op_i32(back, 0) == 4 && op_i32(back, 1) == 0,
               "back inherited params")) return false;

    ggml_set_param(f.x);
    ggml_tensor * loss = ggml_sum(f.ctx, forward);
    ggml_set_loss(loss);
    ggml_cgraph * graph = ggml_new_graph_custom(f.ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(f.ctx, graph, nullptr);

    if (!check(ggml_graph_get_grad(graph, f.x) != nullptr, "activation gradient exists")) return false;
    if (!check(ggml_graph_get_grad(graph, f.weight) == nullptr &&
               ggml_graph_get_grad(graph, f.scales) == nullptr &&
               ggml_graph_get_grad(graph, f.bias) == nullptr,
               "only activation receives gradient")) return false;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!check(cpu != nullptr, "CPU backend init")) return false;
    const bool forward_cpu = ggml_backend_supports_op(cpu, forward);
    const bool back_cpu = ggml_backend_supports_op(cpu, back);
    ggml_backend_free(cpu);
    ggml_free(f.ctx);
    return check(!forward_cpu && !back_cpu, "CPU explicitly rejects ConvRot8");
}

[[noreturn]] void negative_invalid_rotation() {
    Fixture f = make_fixture(false);
    expected_assertion = "ggml_convrot8_valid_rotation(rotation_size)";
    ggml_convrot8(f.ctx, f.weight, f.x, f.scales, nullptr, 3, false);
    std::fprintf(stderr, "negative mode unexpectedly returned\n");
    std::exit(99);
}

[[noreturn]] void negative_noncontiguous() {
    Fixture f = make_fixture(false);
    ggml_tensor * padded = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 17, 3);
    ggml_tensor * x_view = ggml_view_2d(f.ctx, padded, 16, 3, padded->nb[1], 0);
    expected_assertion = "ggml_is_contiguous(x_f32)";
    ggml_convrot8(f.ctx, f.weight, x_view, f.scales, nullptr, 4, false);
    std::fprintf(stderr, "negative mode unexpectedly returned\n");
    std::exit(99);
}

[[noreturn]] void negative_trainable_frozen() {
    Fixture f = make_fixture(false);
    ggml_set_param(f.weight);
    expected_assertion = "ConvRot8 weight/scales/bias are frozen";
    ggml_convrot8(f.ctx, f.weight, f.x, f.scales, nullptr, 4, false);
    std::fprintf(stderr, "negative mode unexpectedly returned\n");
    std::exit(99);
}

} // namespace

int main(int argc, char ** argv) {
    ggml_set_abort_callback(assertion_result);
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    if (argc == 2 && std::strcmp(argv[1], "--invalid-rotation") == 0) negative_invalid_rotation();
    if (argc == 2 && std::strcmp(argv[1], "--noncontiguous") == 0) negative_noncontiguous();
    if (argc == 2 && std::strcmp(argv[1], "--trainable-frozen") == 0) negative_trainable_frozen();
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--invalid-rotation|--noncontiguous|--trainable-frozen]\n", argv[0]);
        return 2;
    }
    if (!positive_contract()) return 1;
    std::puts("convrot ggml contract: PASS");
    return 0;
}
