// mma-ceiling.mm (v2) -- what can this GPU actually do, and how much does occupancy matter?
//
// Part A: fragments loaded from threadgroup memory every iteration, like the real kernels
//         (4 A + 4 B fragments -> 16 mma per iteration, all accumulators stored at the end):
//           ld_f32  : float x float -> float
//           ld_h2f  : half  x half  -> float   (what kernel_mul_mm uses)
//         swept over threads/threadgroup and threadgroup-memory size (the memory is only
//         *reserved* to limit how many threadgroups fit on a core -- the kernel touches 1 KB).
// Part B: scalar FMA, 16 independent chains per thread (fp32, half2), results always stored.
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation engine/tools/mma-ceiling.mm -o /tmp/mma-ceiling && /tmp/mma-ceiling

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>

static NSString * const kSrc = @R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void ld_f32(device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],
                   threadgroup float * tile [[threadgroup(0)]],
                   uint tid [[thread_position_in_grid]], uint tl [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    if (tl < 512) tile[tl] = 0.01f * float((tl * 7 + tid) % 13);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c[16];
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) c[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint it = 0; it < iters; ++it) {
        simdgroup_float8x8 a[4], b[4];
        const short o = (it & 3) * 8;
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 4; ++i) {
            simdgroup_load(a[i], tile + o + i * 8, 64);
            simdgroup_load(b[i], tile + 256 + o + i * 8, 64);
        }
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 4; ++i)
            _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(c[i*4 + j], a[i], b[j], c[i*4 + j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) simdgroup_store(c[i], out + (tid / 32) * 64 * 16 + i * 64, 8);
}

kernel void ld_h2f(device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]],
                   threadgroup half * tile [[threadgroup(0)]],
                   uint tid [[thread_position_in_grid]], uint tl [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    if (tl < 512) tile[tl] = half(0.01f * float((tl * 7 + tid) % 13));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c[16];
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) c[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint it = 0; it < iters; ++it) {
        simdgroup_half8x8 a[4], b[4];
        const short o = (it & 3) * 8;
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 4; ++i) {
            simdgroup_load(a[i], tile + o + i * 8, 64);
            simdgroup_load(b[i], tile + 256 + o + i * 8, 64);
        }
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 4; ++i)
            _Pragma("clang loop unroll(full)") for (short j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(c[i*4 + j], a[i], b[j], c[i*4 + j]);
    }
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) simdgroup_store(c[i], out + (tid / 32) * 64 * 16 + i * 64, 8);
}

kernel void fma_f32(device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    float acc[16];
    const float m = 1.0f - (tid % 5) * 1e-7f, a = 1e-7f * (1 + tid % 3);
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) acc[i] = 0.001f * i;
    for (uint it = 0; it < iters; ++it)
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) acc[i] = fma(acc[i], m, a);
    float s = 0; _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) s += acc[i];
    out[tid] = s;
}

kernel void fma_h2(device float * out [[buffer(0)]], constant uint & iters [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    half2 acc[16];
    const half2 m = half2(half(1.0f - (tid % 5) * 1e-3f)), a = half2(half(1e-3f * (1 + tid % 3)));
    _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) acc[i] = half2(half(0.001f * i));
    for (uint it = 0; it < iters; ++it)
        _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) acc[i] = fma(acc[i], m, a);
    half2 s = 0; _Pragma("clang loop unroll(full)") for (short i = 0; i < 16; ++i) s += acc[i];
    out[tid] = float(s.x) + float(s.y);
}
)MSL";

static double run(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib, const char * name,
                  uint32_t groups, uint32_t tg_threads, uint32_t iters, uint32_t smem, bool verbose = false) {
    NSError * err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { std::printf("%s: pipeline failed: %s\n", name, err.localizedDescription.UTF8String); return 0; }
    if (verbose) std::printf("  (%s: maxThreads/TG %lu, staticTG %lu)\n", name, (unsigned long) pso.maxTotalThreadsPerThreadgroup, (unsigned long) pso.staticThreadgroupMemoryLength);
    if (tg_threads > pso.maxTotalThreadsPerThreadgroup) return -1;
    id<MTLBuffer> out = [dev newBufferWithLength:256u << 20 options:MTLResourceStorageModeShared];
    double best = 1e30;
    for (int rep = 0; rep < 3; ++rep) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc setBytes:&iters length:4 atIndex:1];
        if (smem) [enc setThreadgroupMemoryLength:smem atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg_threads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (rep > 0) best = std::min(best, (double) (cb.GPUEndTime - cb.GPUStartTime));
    }
    return best;
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { std::printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        std::printf("device: %s (max TG memory %lu B)\n", dev.name.UTF8String, (unsigned long) dev.maxThreadgroupMemoryLength);

        const uint32_t total_threads = 1u << 20, iters = 1000;
        std::printf("\nPart A: mma, fragments loaded from threadgroup memory (16 mma per iteration)\n");
        std::printf("%-8s %8s %9s %10s %10s\n", "kernel", "TG thr", "TG mem B", "ms", "TFLOPS");
        const uint32_t smems[] = {2048, 8192, 12288, 16384, 22016, 24576, 32000};
        const uint32_t tgs[] = {128, 256, 512};
        for (const char * name : {"ld_f32", "ld_h2f"}) {
            for (uint32_t tg : tgs) {
                for (uint32_t sm : smems) {
                    const uint32_t groups = total_threads / tg;
                    const double t = run(dev, q, lib, name, groups, tg, iters, sm);
                    if (t < 0) continue;
                    const double flops = 16.0 * 1024.0 * (double) (total_threads / 32) * iters;
                    std::printf("%-8s %8u %9u %10.2f %10.2f\n", name, tg, sm, t * 1e3, flops / t / 1e12);
                }
            }
        }
        std::printf("\nPart B: scalar FMA, 16 chains per thread\n");
        {
            const double t = run(dev, q, lib, "fma_f32", total_threads / 256, 256, iters * 2, 0, true);
            std::printf("fma_f32  %8.2f ms  %6.2f TFLOPS\n", t * 1e3, 16.0 * 2.0 * total_threads * (iters * 2.0) / t / 1e12);
            const double t2 = run(dev, q, lib, "fma_h2", total_threads / 256, 256, iters * 2, 0, true);
            std::printf("fma_h2   %8.2f ms  %6.2f TFLOPS (half2 counted as 2 lanes)\n", t2 * 1e3, 16.0 * 2.0 * 2.0 * total_threads * (iters * 2.0) / t2 / 1e12);
        }
    }
    return 0;
}
