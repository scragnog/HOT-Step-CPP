// Self-check for the LM trainers' attention-mask builders (train/lm-common.h
// lm_causal_mask_fill, train/lm-data.h lm_causal_mask{,_prefix}): the row-fill
// versions must write the SAME BYTES as the per-element loop they replaced, in
// F32 and in F16 (what --attn flash uploads). Host only, no GPU, no models.
// From engine\build\test-standalone in a VS x64 prompt:
//   cl /std:c++17 /EHsc /O2 /utf-8 /I..\..\src /I..\..\ggml\include /I..\..\vendor\yyjson /I..
//      ..\..\tests\test-lm-causal-mask.cpp ..\ggml\src\Release\ggml-base.lib ..\Release\yyjson.lib
#include "train/lm-data.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

void jl(const char *, ...) {}  // lm-common.h declares it; ace-train.cpp defines it

// The pre-2026-11-06 lm_causal_mask_prefix, verbatim.
static void ref_mask(int n_pfx, int pfx_lo, int S, int n_prompt, std::vector<float> * out) {
    const size_t n_kv = (size_t) n_pfx + (size_t) S;
    out->assign(n_kv * (size_t) S, 0.0f);
    float * m = out->data();
    for (int i = 0; i < S; i++) {
        float *    row      = m + (size_t) i * n_kv;
        const bool sees_pfx = (i >= n_prompt);
        for (int j = 0; j < n_pfx; j++) {
            row[j] = (sees_pfx && j >= pfx_lo) ? 0.0f : -INFINITY;
        }
        for (int t = 0; t < S; t++) {
            row[(size_t) n_pfx + (size_t) t] = (t <= i) ? 0.0f : -INFINITY;
        }
    }
}

int main() {
    int fail = 0, cases = 0;
    const int shapes[][4] = {  // n_pfx, pfx_lo, S, n_prompt
        { 0, 0, 1, 0 }, { 0, 0, 7, 0 }, { 0, 0, 1253, 0 }, { 16, 0, 33, 0 }, { 40, 12, 50, 12 },
        { 40, 0, 50, 60 }, { 5, 9, 20, 3 }, { 300, 1183, 1500, 1183 }, { 8, -3, 17, -1 },
    };
    for (const auto & sh : shapes) {
        const int n_pfx = sh[0], pfx_lo = sh[1], S = sh[2], n_prompt = sh[3];
        std::vector<float> ref, got;
        ref_mask(n_pfx, pfx_lo, S, n_prompt, &ref);
        if (n_pfx == 0 && pfx_lo == 0 && n_prompt == 0) {
            lm_causal_mask(S, &got);
        } else {
            lm_causal_mask_prefix(n_pfx, pfx_lo, S, n_prompt, &got);
        }
        // F16: what lm_mask_set used to upload (per-element conversion of ref).
        std::vector<ggml_fp16_t> ref16(ref.size()), got16(ref.size());
        for (size_t i = 0; i < ref.size(); i++) ref16[i] = ggml_fp32_to_fp16(ref[i]);
        lm_causal_mask_fill(got16.data(), n_pfx, pfx_lo, S, n_prompt, ggml_fp32_to_fp16(0.0f),
                            ggml_fp32_to_fp16(-INFINITY));
        const bool ok32 = got.size() == ref.size() && !memcmp(got.data(), ref.data(), ref.size() * sizeof(float));
        const bool ok16 = !memcmp(got16.data(), ref16.data(), ref16.size() * sizeof(ggml_fp16_t));
        if (!ok32 || !ok16) {
            fail++;
            fprintf(stderr, "FAIL n_pfx %d pfx_lo %d S %d n_prompt %d: f32 %s f16 %s\n", n_pfx, pfx_lo, S, n_prompt,
                    ok32 ? "ok" : "DIFF", ok16 ? "ok" : "DIFF");
        }
        cases++;
    }
    printf("lm causal mask: %d/%d shapes byte-identical (f32 + f16)\n", cases - fail, cases);
    return fail ? 1 : 0;
}
