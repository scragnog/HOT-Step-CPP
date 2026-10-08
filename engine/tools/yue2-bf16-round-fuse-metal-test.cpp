// yue2-bf16-round-fuse-metal-test.cpp -- bit-exactness gate for the new
// BF16_ROUND epilogue fusion in kernel_bin_fuse (ggml-metal.metal /
// ggml-metal-ops.cpp's ggml_metal_op_bin): when the node right after an
// ADD/SUB/MUL(/DIV) is a single-consumer ggml_bf16_round() of its result,
// the Metal backend now absorbs it into the same kernel dispatch instead of
// a separate kernel_unary_impl pass over the whole tensor.
//
// This is purely an execution-scheduling change on the Metal backend -- the
// ggml graph itself still has two separate nodes (needed for autodiff), and
// bf16_round_cast<T> is the exact same round-to-nearest-even cast whether
// applied inline or as its own kernel -- so fused-Metal, unfused-Metal and
// CPU are all expected to agree bit-for-bit (see kernel_bin_fuse_impl's own
// comment on FC_bin_rnd, and the pre-existing BF16_ROUND cast comment in
// ggml-metal.metal).
//
// Cases cover: single ADD/SUB/MUL + round (yue2-aitk-graph.h's round(add),
// round(sub) in the RoPE join, and mul_bf16's round(mul)); a 3-way ADD
// chain + round (exercises n_fuse>1 together with the round epilogue); a
// row-broadcast shape (src1 has 1 row, exercises kernel_bin_fuse's other
// dispatch path, "dst_row[...]" not "dst_ptr[...]"); and a multi-consumer
// case where the add's result is read by a second branch too, so
// ggml_can_fuse_ext's single-use check must correctly refuse to fuse --
// this must still produce the right answer via the un-fused fallback.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

enum class Op { add, sub, mul };

struct Case {
    const char * name;
    Op op;
    int64_t ne0, ne1;       // shape of a/b (and the add-chain case's extra operands)
    bool broadcast;         // b is [ne0,1] instead of [ne0,ne1]
    int  chain;             // >1: chain this many same-op terms before round()
    bool multi_consumer;    // also read the pre-round result elsewhere
};

const Case kCases[] = {
    {"add-round",            Op::add, 2048, 37, false, 1, false},
    {"sub-round",             Op::sub, 128,  9,  false, 1, false},
    {"mul-round",             Op::mul, 6144, 11, false, 1, false},
    {"add-chain3-round",      Op::add, 64,   5,  false, 3, false},
    {"add-round-broadcast",   Op::add, 2048, 5,  true,  1, false},
    {"add-round-multi-use",   Op::add, 96,   7,  false, 1, true},
};

std::mt19937 rng(0xF00D1234u);
std::vector<float> random_vec(size_t n) {
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    std::vector<float> v(n);
    for (float & x : v) x = dist(rng);
    return v;
}

// Builds: result = round(chain of `case_.chain` `case_.op` terms).
// With multi_consumer, also returns an extra_sum = ggml_sum(pre-round result)
// so the pre-round tensor has a second consumer and cannot be fused away.
struct Built {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> operands; // a, b[, c, d...]
    ggml_tensor * result = nullptr;
    ggml_tensor * extra_sum = nullptr;
    ~Built() { if (ctx) ggml_free(ctx); }
};

bool build(const Case & c, Built * out, std::string * error) {
    ggml_init_params p{}; p.mem_size = 1u*1024u*1024u; p.no_alloc = true;
    out->ctx = ggml_init(p);
    if (!out->ctx) { *error = "ggml_init failed"; return false; }
    ggml_context * ctx = out->ctx;

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
    out->operands.push_back(a);
    ggml_tensor * acc = a;
    const int terms = c.broadcast ? 1 : std::max(1, c.chain);
    for (int i = 0; i < terms; ++i) {
        ggml_tensor * b = c.broadcast
            ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, 1)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne0, c.ne1);
        out->operands.push_back(b);
        switch (c.op) {
            case Op::add: acc = ggml_add(ctx, acc, b); break;
            case Op::sub: acc = ggml_sub(ctx, acc, b); break;
            case Op::mul: acc = ggml_mul(ctx, acc, b); break;
        }
        if (!acc) { *error = "bin op build failed"; return false; }
    }
    if (c.multi_consumer) {
        out->extra_sum = ggml_sum(ctx, acc);
        if (!out->extra_sum) { *error = "sum build failed"; return false; }
    }
    out->result = ggml_bf16_round(ctx, acc);
    if (!out->result) { *error = "bf16_round build failed"; return false; }

    out->graph = ggml_new_graph(ctx);
    if (!out->graph) { *error = "graph alloc failed"; return false; }
    ggml_build_forward_expand(out->graph, out->result);
    if (out->extra_sum) ggml_build_forward_expand(out->graph, out->extra_sum);
    for (ggml_tensor * t : out->operands) ggml_set_input(t);
    ggml_set_output(out->result);
    if (out->extra_sum) ggml_set_output(out->extra_sum);
    return true;
}

struct Result { std::vector<float> result, extra_sum; };

bool run(ggml_backend_t backend, const Case & c, const std::vector<std::vector<float>> & inputs,
         Result * out, std::string * error) {
    Built b;
    if (!build(c, &b, error)) return false;
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, b.graph)) { *error = "alloc failed"; ggml_gallocr_free(alloc); return false; }
    for (size_t i = 0; i < b.operands.size(); ++i) {
        ggml_backend_tensor_set(b.operands[i], inputs[i].data(), 0, ggml_nbytes(b.operands[i]));
    }
    if (ggml_backend_graph_compute(backend, b.graph) != GGML_STATUS_SUCCESS) { *error = "compute failed"; ggml_gallocr_free(alloc); return false; }
    out->result.resize(ggml_nelements(b.result));
    ggml_backend_tensor_get(b.result, out->result.data(), 0, ggml_nbytes(b.result));
    if (b.extra_sum) {
        out->extra_sum.resize(ggml_nelements(b.extra_sum));
        ggml_backend_tensor_get(b.extra_sum, out->extra_sum.data(), 0, ggml_nbytes(b.extra_sum));
    }
    ggml_gallocr_free(alloc);
    return true;
}

bool equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0);
}

// ggml_sum's reduction order isn't guaranteed to match between the CPU and
// Metal backends (this project's own bit-exactness discipline is reserved
// for ops that document a specific, matched summation order -- see e.g.
// ggml_compute_forward_out_prod_f32's comment; plain ggml_sum makes no such
// promise on either backend). The multi-consumer case only uses extra_sum
// to prove the pre-round tensor still gets computed correctly when fusion
// is (correctly) refused, so a loose tolerance is the right bar here, not
// bit-exactness -- unlike `result`, which stays memcmp-exact throughout.
bool close(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        const float tol = 1e-3f * std::max(1.0f, std::fabs(b[i]));
        if (diff > tol) return false;
    }
    return true;
}

} // namespace

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu_dev || !cpu_dev) { std::puts("yue2-bf16-round-fuse-metal-test: SKIP (no Metal/CPU device)"); return 0; }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    bool all_ok = true;
    for (const Case & c : kCases) {
        // Build once just to learn the operand count/shapes for input generation.
        Built probe; std::string err;
        if (!build(c, &probe, &err)) { std::fprintf(stderr, "[%s] probe: %s\n", c.name, err.c_str()); all_ok = false; continue; }
        std::vector<std::vector<float>> inputs;
        for (ggml_tensor * t : probe.operands) inputs.push_back(random_vec(ggml_nelements(t)));

        Result gpu_res, cpu_res;
        bool ok = run(gpu, c, inputs, &gpu_res, &err);
        if (!ok) { std::fprintf(stderr, "[%s] gpu: %s\n", c.name, err.c_str()); all_ok = false; continue; }
        ok = run(cpu, c, inputs, &cpu_res, &err);
        if (!ok) { std::fprintf(stderr, "[%s] cpu: %s\n", c.name, err.c_str()); all_ok = false; continue; }

        const bool result_ok = equal(gpu_res.result, cpu_res.result);
        const bool sum_ok = close(gpu_res.extra_sum, cpu_res.extra_sum);
        ok = result_ok && sum_ok;
        if (!ok) {
            std::fprintf(stderr, "[%s] mismatch (result_ok=%d sum_ok=%d)\n", c.name, result_ok, sum_ok);
        }
        std::printf("[%-22s] %s\n", c.name, ok ? "PASS" : "FAIL");
        all_ok &= ok;
    }

    std::puts(all_ok ? "yue2-bf16-round-fuse-metal-test: ALL PASS" : "yue2-bf16-round-fuse-metal-test: FAIL");
    return all_ok ? 0 : 1;
}
