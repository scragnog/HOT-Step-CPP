#pragma once
// minimax/mm3-vram-fit.h - does an MM3 job's planning stage fit the card?
//
// HOT-Step file. Pure arithmetic with no ggml or server state, so
// engine/tests/test-mm3-vram-fit.cpp can check it without a GPU. The caller is
// mm3_synth_worker (mm3-job.h), which measures the budget and decides when the
// gate applies at all (CUDA only, or MM3_VRAM_BUDGET_GB).
//
// The ladder, cheapest change to the audio first:
//   1. the request as sent;
//   2. the planner's KV cache at Q8_0 (half the bytes; a different plan);
//   3. one song wanted (stop_after_first_eos): fewer ending candidates per
//      round, down to 1. Rounds still advance, so endings are still sought.
// Nothing else is shrunk. Frames are never capped: a capped MM3 plan loses its
// ending, and with endings required it is dropped anyway.

#include <cstddef>

// What stage 1 holds on top of its weights and KV cache: LM and depth compute
// buffers, cuBLAS workspace, allocator slack. Measured 0.118 GiB on a 4090
// (Q8_0 LM, 1 take, Q8_0 KV, 7680 positions, staged /mm3/synth); 0.5 GiB
// leaves room for more rows and a runtime LM adapter.
#define MM3_FIT_COMPUTE_BYTES ((size_t) 512 * 1024 * 1024)

struct MM3VramFit {
    bool   ok    = true;   // false = refuse the job
    bool   q8    = false;  // effective KV type
    int    takes = 1;      // effective candidates per round
    size_t need  = 0;      // bytes the chosen (or, refused, the last tried) shape wants
};

// Stage-1 bytes for `rows` LM rows. `kv_row_f16` is one row's f16 KV cache at
// this job's context; Q8_0 is 34 bytes per 32 values against f16's 64, the
// same rounding as mm3_kv_bytes_estimate.
static inline size_t mm3_fit_need(size_t weights, size_t kv_row_f16, int rows, bool q8) {
    const size_t kv = kv_row_f16 * (size_t) rows;
    return weights + (q8 ? kv / 64 * 34 : kv) + MM3_FIT_COMPUTE_BYTES;
}

// `rows_per_take`: CFG rows, doubled by the plan dial's base rows.
// `can_q8`: false when MM3_LM_KV_TYPE pins the cache to f16.
static inline MM3VramFit mm3_vram_fit(size_t budget, size_t weights, size_t kv_row_f16, int rows_per_take,
                                      int takes, bool q8, bool can_q8, bool one_song) {
    MM3VramFit f;
    f.q8    = q8;
    f.takes = takes < 1 ? 1 : takes;
    f.need  = mm3_fit_need(weights, kv_row_f16, rows_per_take * f.takes, f.q8);
    if (f.need <= budget) {
        return f;
    }
    if (can_q8 && !f.q8) {
        f.q8   = true;
        f.need = mm3_fit_need(weights, kv_row_f16, rows_per_take * f.takes, true);
        if (f.need <= budget) {
            return f;
        }
    }
    while (one_song && f.takes > 1) {
        f.takes--;
        f.need = mm3_fit_need(weights, kv_row_f16, rows_per_take * f.takes, f.q8);
        if (f.need <= budget) {
            return f;
        }
    }
    f.ok = false;
    return f;
}
