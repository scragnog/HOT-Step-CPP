#include "../../engine/ggml/include/ggml.h"
#include "../../engine/ggml/include/ggml-backend.h"
#include "../../engine/ggml/include/ggml-cuda.h"
#include "../../engine/src/train/yue2-aitk-optimizer.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

struct Buffers {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * p_small = nullptr;
    ggml_tensor * g_small = nullptr;
    ggml_tensor * p_large = nullptr;
    ggml_tensor * g_large = nullptr;
    Buffers() = default;
    Buffers(const Buffers &) = delete;
    Buffers & operator=(const Buffers &) = delete;
    Buffers(Buffers && other) noexcept : ctx(other.ctx), buffer(other.buffer), p_small(other.p_small), g_small(other.g_small), p_large(other.p_large), g_large(other.g_large) { other.ctx = nullptr; other.buffer = nullptr; }
    ~Buffers() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};

Buffers make_buffers(ggml_backend_t backend) {
    ggml_init_params params{}; params.mem_size = 4u << 20; params.no_alloc = true;
    Buffers b; b.ctx = ggml_init(params); if (!b.ctx) throw std::runtime_error("ggml_init failed");
    b.p_small = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 4095);
    b.g_small = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 4095);
    b.p_large = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 4097);
    b.g_large = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 4097);
    if (!b.p_small || !b.g_small || !b.p_large || !b.g_large) throw std::runtime_error("tensor construction failed");
    b.buffer = ggml_backend_alloc_ctx_tensors(b.ctx, backend); if (!b.buffer) throw std::runtime_error("backend allocation failed");
    std::vector<float> small(4095), large(4097), zeros(4097, 0.0f);
    for (size_t i = 0; i < small.size(); ++i) small[i] = .01f * float(int(i % 17) - 8);
    for (size_t i = 0; i < large.size(); ++i) large[i] = .01f * float(int(i % 29) - 14);
    ggml_backend_tensor_set(b.p_small, small.data(), 0, ggml_nbytes(b.p_small));
    ggml_backend_tensor_set(b.p_large, large.data(), 0, ggml_nbytes(b.p_large));
    ggml_backend_tensor_set(b.g_small, zeros.data(), 0, ggml_nbytes(b.g_small));
    ggml_backend_tensor_set(b.g_large, zeros.data(), 0, ggml_nbytes(b.g_large));
    return b;
}

void set_gradients(const Buffers & b, int step) {
    std::vector<float> small(4095), large(4097);
    for (size_t i = 0; i < small.size(); ++i) small[i] = step == 0 ? 0.0f : .001f * float(int((i + 3 * step) % 19) - 9);
    for (size_t i = 0; i < large.size(); ++i) large[i] = step == 0 ? 0.0f : .001f * float(int((i + 5 * step) % 23) - 11);
    ggml_backend_tensor_set(b.g_small, small.data(), 0, ggml_nbytes(b.g_small));
    ggml_backend_tensor_set(b.g_large, large.data(), 0, ggml_nbytes(b.g_large));
}

template<typename T> bool all_zero(const std::vector<T> & v) { for (const T x : v) if (x != T{}) return false; return true; }
int fail(const char * message) { std::fprintf(stderr, "optimizer owner probe: %s\n", message); return 3; }
bool same(const yue2_aitk::HostStateSnapshot & a, const yue2_aitk::HostStateSnapshot & b) {
    return a.names == b.names && a.elements == b.elements && a.step == b.step && a.parameters == b.parameters &&
        a.state1_fp32 == b.state1_fp32 && a.state2_fp32 == b.state2_fp32 && a.state1_u8 == b.state1_u8 &&
        a.state2_u8 == b.state2_u8 && a.absmax1 == b.absmax1 && a.absmax2 == b.absmax2;
}

int run(ggml_backend_t backend) {
    Buffers b = make_buffers(backend);
    std::vector<yue2_aitk::ParameterSpec> specs = {{"small", b.p_small, b.g_small}, {"large", b.p_large, b.g_large}};
    yue2_aitk::StepConfig cfg; cfg.learning_rate = 1e-3f; cfg.weight_decay = 0.0f;
    yue2_aitk::Optimizer first(backend, 0, specs);
    const auto initial = first.capture();
    set_gradients(b, 0); first.step_once(cfg);
    const auto zero = first.capture();
    if (zero.step != 1) return fail("zero-gradient step counter is not one");
    if (!all_zero(zero.state1_fp32[0]) || !all_zero(zero.state2_fp32[0])) return fail("zero-gradient FP32 moments are not zero");
    if (!all_zero(zero.absmax1[1]) || !all_zero(zero.absmax2[1])) return fail("zero-gradient quantized absmax is not zero");
    if (zero.parameters != initial.parameters) return fail("zero-gradient parameters changed despite zero weight decay");
    if (zero.state1_u8[1].empty() || zero.state2_u8[1].empty()) return fail("zero-gradient quantized codes were not captured");
    std::printf("zero-gradient captured codes: signed=%u unsigned=%u; decoded moments are zero because absmax=0\n", unsigned(zero.state1_u8[1][0]), unsigned(zero.state2_u8[1][0]));

    yue2_aitk::HostStateSnapshot checkpoint;
    for (int step = 1; step < 5; ++step) {
        set_gradients(b, step); first.step_once(cfg);
        if (step == 2) checkpoint = first.capture();
    }
    const auto final_first = first.capture();
    if (final_first.step != 5) return fail("final step counter is not five");

    yue2_aitk::Optimizer resumed(backend, 0, specs);
    resumed.restore(checkpoint);
    for (int step = 3; step < 5; ++step) { set_gradients(b, step); resumed.step_once(cfg); }
    if (!same(final_first, resumed.capture())) return fail("capture/restore continuation diverged");

    auto bad = checkpoint; bad.names[1] = "wrong"; bad.parameters[0][0] += 1.0f;
    const auto before_bad = resumed.capture(); bool rejected = false;
    try { resumed.restore(bad); } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) return fail("malformed name snapshot was accepted");
    if (!same(before_bad, resumed.capture())) return fail("malformed name snapshot mutated state");
    auto negative_moment = checkpoint; negative_moment.state2_fp32[0][0] = -1.0f;
    rejected = false;
    try { resumed.restore(negative_moment); } catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) return fail("negative second moment snapshot was accepted");
    if (!same(before_bad, resumed.capture())) return fail("negative second moment snapshot mutated state");
    std::puts("optimizer owner state roundtrip passed: 4095/4097, zero gradient, continuation, prevalidated rejection");
    return 0;
}

} // namespace

int main() {
    try {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) { std::fputs("CUDA backend unavailable\n", stderr); return 77; }
        const int rc = run(backend); ggml_backend_free(backend); return rc;
    } catch (const std::exception & e) { std::fprintf(stderr, "optimizer owner probe: %s\n", e.what()); return 2; }
}
