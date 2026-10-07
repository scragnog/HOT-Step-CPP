// yue2-nar-op-bench.cpp -- how close do the generic ggml Metal kernels run to the hardware at the
// shapes of the YuE2 NAR (generation) forward?  No model needed, only ggml.
//
//   mul_mat   : quantized (q8_0) / f16 weights x f32 activations  (qkv, o, gate-up, down; T tokens)
//   flash_attn: ggml_flash_attn_ext at the NAR attention shape (Nh=16, Nkv=8, D=128, Tq=T, Tk=prefix+T)
//
// Prints ms and TFLOPS (2*K*N*T for mul_mat, 4*T*Tk*D*Nh for attention). Compare with the ~6-7 TFLOPS
// simdgroup-mma ceiling measured by mma-ceiling.mm.
//
// Usage: yue2-nar-op-bench [--t N] [--tk N] [--iters N]    (defaults T=9536, Tk=20992, iters=5)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
std::mt19937 rng(12345u);

void fill_f32(std::vector<float> & v, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    for (auto & x : v) x = u(rng);
}

// copy float data into a tensor of any (also quantized) type
void set_tensor_from_f32(ggml_tensor * t, const std::vector<float> & f) {
    const int64_t ne0 = t->ne[0];
    const int64_t rows = ggml_nelements(t) / ne0;
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(f.size());
        ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> q(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, rows, ne0, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    }
}

double time_graph(ggml_backend_t be, ggml_cgraph * g, int iters) {
    double ms = 0;
    for (int it = 0; it < iters + 1; ++it) { // first iteration compiles pipelines
        const auto t0 = clk::now();
        if (ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) return -1;
        ggml_backend_synchronize(be);
        const auto t1 = clk::now();
        if (it > 0) ms += std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    }
    return ms;
}

ggml_context * new_ctx() {
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false) + 4096;
    p.no_alloc = true;
    return ggml_init(p);
}

void bench_mm(ggml_backend_t be, const char * name, ggml_type wt, int64_t K, int64_t N, int64_t T, int iters) {
    ggml_context * ctx = new_ctx();
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wt, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { std::printf("[%-10s %-5s] alloc failed\n", name, ggml_type_name(wt)); ggml_free(ctx); return; }
    std::vector<float> fw((size_t) K*N), fx((size_t) K*T);
    fill_f32(fw, -1.0f, 1.0f); fill_f32(fx, -1.0f, 1.0f);
    set_tensor_from_f32(w, fw); set_tensor_from_f32(x, fx);
    const double ms = time_graph(be, g, iters);
    const double tf = 2.0*K*N*T / (ms*1e-3) / 1e12;
    std::printf("[%-10s %-5s] K=%5lld N=%6lld T=%5lld | %8.2f ms  %5.2f TFLOPS\n", name, ggml_type_name(wt),
                (long long) K, (long long) N, (long long) T, ms, tf);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}

void bench_fa(ggml_backend_t be, ggml_type kvt, bool with_mask, int64_t T, int64_t Tk, int iters) {
    const int64_t D = 128, Nh = 16, Nkv = 8;
    ggml_context * ctx = new_ctx();
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, Nh, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, kvt, D, Tk, Nkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, kvt, D, Tk, Nkv, 1);
    ggml_tensor * m = with_mask ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Tk, GGML_PAD(T, 64)) : nullptr;
    ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
    if (!ggml_backend_supports_op(be, y)) {
        std::printf("[flash_attn %-4s mask=%d] not supported by the backend\n", ggml_type_name(kvt), (int) with_mask);
        ggml_free(ctx); return;
    }
    ggml_cgraph * g = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { std::printf("[flash_attn] alloc failed\n"); ggml_free(ctx); return; }
    std::vector<float> fq((size_t) D*T*Nh), fk((size_t) D*Tk*Nkv);
    fill_f32(fq, -1.0f, 1.0f); fill_f32(fk, -1.0f, 1.0f);
    set_tensor_from_f32(q, fq); set_tensor_from_f32(k, fk); set_tensor_from_f32(v, fk);
    if (m) {
        std::vector<ggml_fp16_t> z((size_t) ggml_nelements(m), ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(m, z.data(), 0, z.size()*sizeof(ggml_fp16_t));
    }
    const double ms = time_graph(be, g, iters);
    const double tf = 4.0*T*Tk*D*Nh / (ms*1e-3) / 1e12;
    std::printf("[flash_attn %-4s mask=%d] Tq=%5lld Tk=%6lld Nh=%lld Nkv=%lld D=%lld | %8.2f ms  %5.2f TFLOPS\n",
                ggml_type_name(kvt), (int) with_mask, (long long) T, (long long) Tk, (long long) Nh, (long long) Nkv,
                (long long) D, ms, tf);
    ggml_backend_buffer_free(buf); ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    int64_t T = 9536, Tk = 20992; int iters = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--t"))     T = std::atoll(argv[i + 1]);
        if (!std::strcmp(argv[i], "--tk"))    Tk = std::atoll(argv[i + 1]);
        if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[i + 1]);
    }
    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev) { std::puts("yue2-nar-op-bench: SKIP (no GPU device)"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!be) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    std::printf("yue2-nar-op-bench: T=%lld Tk=%lld iters=%d\n", (long long) T, (long long) Tk, iters);

    for (ggml_type wt : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        bench_mm(be, "qkv",     wt, 2048,  4096, T, iters);
        bench_mm(be, "o",       wt, 2048,  2048, T, iters);
        bench_mm(be, "gate-up", wt, 2048, 12288, T, iters);
        bench_mm(be, "down",    wt, 6144,  2048, T, iters);
    }
    for (ggml_type kvt : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        bench_fa(be, kvt, true,  T, Tk, iters);
        bench_fa(be, kvt, false, T, Tk, iters);
    }
    ggml_backend_free(be);
    return 0;
}
