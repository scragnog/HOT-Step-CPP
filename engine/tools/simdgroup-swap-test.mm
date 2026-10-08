// simdgroup-swap-test.mm -- is simdgroup_multiply_accumulate bit-symmetric under operand swap?
//
// Question (Opus session 3, step 1): the dQ kernel computes S = Q.K^T, the KV kernel
// computes S^T = K.Q^T (same d-order, same 8x8 blocks, f32 accumulate, zero init).
// Is S[i][j] bit-equal to S^T[j][i] on Apple GPUs? If yes, the KV kernel can write dS
// and a light dQ kernel can read it with no change to the numerics of the dQ path.
//
// Standalone (Metal + Foundation only, no ggml). Build + run:
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation \
//       simdgroup-swap-test.mm -o /tmp/simdgroup-swap-test && /tmp/simdgroup-swap-test
//
// Compiled twice (fastMathEnabled YES / NO), each over several data classes
// (normal, wide-exponent, denormals, +-0, attention-like). Result is a bitwise
// compare of S vs transpose(St) over ~1M outputs per class. Exit 0 = all bit-equal.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cmath>

static const int D = 128;      // head dim
static const int TILES = 16384; // 8x8 outputs per tile -> ~1M outputs per class

static NSString *kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
constant int D = 128;
// one simdgroup (32 threads) per threadgroup = one 8x8 tile pair
kernel void swap_test(device const float *Q [[buffer(0)]],
                      device const float *K [[buffer(1)]],
                      device float *S  [[buffer(2)]],
                      device float *St [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]]) {
    device const float *q = Q + (ulong)tg*8*D;   // 8 x D row-major
    device const float *k = K + (ulong)tg*8*D;
    simdgroup_float8x8 a = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    simdgroup_float8x8 b = make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for (int d = 0; d < D; d += 8) {
        simdgroup_float8x8 qd, kdT, kd, qdT;
        simdgroup_load(qd,  q, D, ulong2(d,0), false);
        simdgroup_load(kdT, k, D, ulong2(d,0), true);   // K^T block (d x j)
        simdgroup_multiply_accumulate(a, qd, kdT, a);   // S  = Q . K^T
        simdgroup_load(kd,  k, D, ulong2(d,0), false);
        simdgroup_load(qdT, q, D, ulong2(d,0), true);
        simdgroup_multiply_accumulate(b, kd, qdT, b);   // St = K . Q^T
    }
    simdgroup_store(a, S  + (ulong)tg*64, 8);
    simdgroup_store(b, St + (ulong)tg*64, 8);
}
)";

static float denorm(std::mt19937 &r) {
    uint32_t m = r() & 0x007fffffu, s = (r() & 1u) << 31;
    uint32_t u = s | m; float f; memcpy(&f, &u, 4); return f;
}

static void fill(std::vector<float> &v, int cls, std::mt19937 &r) {
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::uniform_int_distribution<int> E(-8, 8);
    for (auto &x : v) {
        switch (cls) {
        case 0: x = U(r); break;                                  // normal
        case 1: x = ldexpf(U(r), E(r)); break;                    // wide exponent
        case 2: x = (r() % 3 == 0) ? denorm(r) : U(r) * 1e-30f; break; // denormal-heavy
        case 3: { unsigned t = r() % 4; x = t == 0 ? 0.f : t == 1 ? -0.f : U(r); break; } // +-0
        default: x = U(r) * (r() % 16 == 0 ? 8.f : 0.5f); break;  // attention-like
        }
    }
}

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("device: %s\n", dev.name.UTF8String);
        const char *names[] = {"normal", "wide-exp", "denormal", "+-0", "attn-like"};
        long total_bad = 0;
        for (int fm = 1; fm >= 0; --fm) {
            MTLCompileOptions *o = [MTLCompileOptions new];
            o.fastMathEnabled = fm;
            NSError *err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:o error:&err];
            if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 2; }
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"swap_test"] error:&err];
            for (int cls = 0; cls < 5; ++cls) {
                std::mt19937 r(1234 + cls);
                std::vector<float> hq((size_t)TILES*8*D), hk((size_t)TILES*8*D);
                fill(hq, cls, r); fill(hk, cls, r);
                id<MTLBuffer> bq = [dev newBufferWithBytes:hq.data() length:hq.size()*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bk = [dev newBufferWithBytes:hk.data() length:hk.size()*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bs = [dev newBufferWithLength:(size_t)TILES*64*4 options:MTLResourceStorageModeShared];
                id<MTLBuffer> bt = [dev newBufferWithLength:(size_t)TILES*64*4 options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
                [e setBuffer:bs offset:0 atIndex:2]; [e setBuffer:bt offset:0 atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(TILES,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const uint32_t *S = (const uint32_t *)bs.contents, *T = (const uint32_t *)bt.contents;
                long bad = 0, nan = 0; long first = -1;
                for (long t = 0; t < TILES; ++t)
                    for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) {
                        uint32_t a = S[t*64 + i*8 + j], b = T[t*64 + j*8 + i];
                        if (a != b) { if (std::isnan(*(float*)&a) && std::isnan(*(float*)&b)) { ++nan; continue; } if (first < 0) first = t*64+i*8+j; ++bad; }
                    }
                printf("fastmath=%d  %-9s  outputs=%ld  mismatches=%ld%s\n", fm, names[cls], (long)TILES*64, bad, bad ? "  <-- DIFF" : "");
                if (first >= 0) printf("   first at idx %ld: S=%08x St=%08x\n", first, S[first], T[(first/64)*64 + ((first%8)*8) + ((first%64)/8)]);
                total_bad += bad;
            }
        }
        printf(total_bad ? "RESULT: NOT bit-symmetric (%ld mismatches)\n" : "RESULT: ALL BIT-EQUAL\n", total_bad);
        return total_bad ? 1 : 0;
    }
}
