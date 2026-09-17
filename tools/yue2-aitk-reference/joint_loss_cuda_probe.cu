#include "joint_loss_cuda.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace fs = std::filesystem;
static bool read_u64(std::ifstream &f, uint64_t &value) {
    value = 0; f.read(reinterpret_cast<char *>(&value), sizeof value); return bool(f);
}
static bool checked_count(uint64_t n, uint64_t v, size_t &count) {
    if (!n || !v || n > std::numeric_limits<size_t>::max() / v) return false;
    count = static_cast<size_t>(n * v); return count <= 1000000000ULL;
}
int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "usage: joint_loss_cuda_probe INPUT OUTPUT\n"; return 2; }
    const fs::path input(argv[1]), output(argv[2]);
    if (fs::exists(output)) { std::cerr << "refusing to overwrite output\n"; return 2; }
    std::ifstream in(input, std::ios::binary);
    if (!in) { std::cerr << "cannot open input\n"; return 2; }
    uint64_t n64 = 0, v64 = 0, chunk64 = 0;
    if (!read_u64(in, n64) || !read_u64(in, v64) || !read_u64(in, chunk64) || !chunk64) { std::cerr << "truncated header\n"; return 2; }
    size_t elements = 0;
    if (!checked_count(n64, v64, elements) || n64 > std::numeric_limits<size_t>::max() || v64 > std::numeric_limits<size_t>::max() ||
        chunk64 > std::numeric_limits<size_t>::max()) { std::cerr << "invalid dimensions\n"; return 2; }
    const size_t n = static_cast<size_t>(n64), v = static_cast<size_t>(v64), chunk = static_cast<size_t>(chunk64);
    const uint64_t expected = 24ULL + uint64_t(elements) * sizeof(float) * 2ULL + n64 * sizeof(uint32_t);
    if (expected < 24 || fs::file_size(input) < expected) { std::cerr << "truncated payload\n"; return 2; }
    std::vector<float> adapted(elements), base(elements); std::vector<uint32_t> targets(n);
    in.read(reinterpret_cast<char *>(adapted.data()), adapted.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(base.data()), base.size() * sizeof(float));
    in.read(reinterpret_cast<char *>(targets.data()), targets.size() * sizeof(uint32_t));
    if (!in) { std::cerr << "truncated payload\n"; return 2; }
    for (uint32_t target : targets) if (target >= v) { std::cerr << "target outside vocabulary\n"; return 2; }
    float *da = nullptr, *db = nullptr, *dg = nullptr, *dce = nullptr, *dkl = nullptr; uint32_t *dt = nullptr;
    cudaStream_t stream = nullptr;
    auto fail = [&](const char *where) { std::cerr << where << ": " << cudaGetErrorString(cudaGetLastError()) << "\n"; return 3; };
    if (cudaStreamCreate(&stream) != cudaSuccess) return fail("stream");
    const size_t per_bytes = n * sizeof(float), chunk_rows = (chunk < n ? chunk : n), chunk_tensor_bytes = chunk_rows * v * sizeof(float);
    if (cudaMalloc(&da, chunk_tensor_bytes) != cudaSuccess || cudaMalloc(&db, chunk_tensor_bytes) != cudaSuccess ||
        cudaMalloc(&dg, chunk_tensor_bytes) != cudaSuccess || cudaMalloc(&dt, chunk_rows * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&dce, per_bytes) != cudaSuccess || cudaMalloc(&dkl, per_bytes) != cudaSuccess) return fail("alloc");
    if (cudaMemsetAsync(dce, 0, per_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(dkl, 0, per_bytes, stream) != cudaSuccess) return fail("initialize losses");
    std::vector<float> gradient(elements);
    for (size_t offset = 0; offset < n; offset += chunk) {
        const size_t count = chunk < n - offset ? chunk : n - offset;
        if (cudaMemcpyAsync(da, adapted.data() + offset * v, count * v * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(db, base.data() + offset * v, count * v * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(dt, targets.data() + offset, count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("upload chunk");
        if (yue2_joint_loss_cuda::forward_chunk(da, db, dt, dg, dce, dkl, offset, count, n, v, 0.2f, stream) != yue2_joint_loss_cuda::Status::success) return fail("forward");
        if (cudaMemcpyAsync(gradient.data() + offset * v, dg, count * v * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return fail("download gradient");
        if (count == n - offset) break; // Avoid offset overflow for oversized fixture chunk values.
    }
    float *dce_total = nullptr, *dkl_total = nullptr;
    if (cudaMalloc(&dce_total, sizeof(float)) != cudaSuccess || cudaMalloc(&dkl_total, sizeof(float)) != cudaSuccess) return fail("loss alloc");
    if (yue2_joint_loss_cuda::reduce(dce, dkl, n, dce_total, dkl_total, stream) != yue2_joint_loss_cuda::Status::success) return fail("reduce");
    float ce = 0, kl = 0;
    if (cudaMemcpyAsync(&ce, dce_total, sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&kl, dkl_total, sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return fail("download losses");
    // Per-chunk gradient copies were queued above; this final sync completes them.
    if (cudaStreamSynchronize(stream) != cudaSuccess) return fail("sync");
    std::ofstream out(output, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) { std::cerr << "cannot create output\n"; return 2; }
    const double ced = ce, kld = kl;
    out.write(reinterpret_cast<const char *>(&ced), sizeof ced); out.write(reinterpret_cast<const char *>(&kld), sizeof kld);
    out.write(reinterpret_cast<const char *>(gradient.data()), elements * sizeof(float));
    if (!out) return 2;
    cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dt); cudaFree(dce); cudaFree(dkl); cudaFree(dce_total); cudaFree(dkl_total); cudaStreamDestroy(stream);
    return 0;
}
