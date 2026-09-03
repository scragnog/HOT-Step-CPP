#pragma once
// lm-prefix-runtime.h — seed a KV set with a trained prefix (prefix tuning) at
// inference. Included by qwen3-lm.h AFTER struct Qwen3LM is complete.
//
// A prefix is n learned K/V columns per layer that every query attends over. At
// inference that is literally a pre-populated KV cache: write P_k/P_v into
// columns [0, n) of a set and start the real tokens at column n. Two contracts
// from train/lm-prefix.h must hold here or the run is silently wrong:
//
//   1. P_k goes in RAW. It was trained post-QK-norm and WITHOUT RoPE (a prefix
//      has no position); the forward ropes only the window's own K.
//   2. RoPE positions of the real tokens stay 0..S-1, exactly as trained. So
//      kv_pos starts at n but positions are computed from kv_pos - kv_base,
//      while kv_rows and the mask stay ABSOLUTE so the prefix columns are read.
//
// Layout: the trainer's [Nkv*D, n] column has heads contiguous within a column
// (index d + h*D); the cache is [D, S_max, Nkv] F16 with heads as the SLOWEST
// axis. So the write is per head: a contiguous [D, n] block at byte offset
// h * nb[2], transposed on the host.
//
// Seeded ONLY where the caller asks (Phase-2 cond sets). Phase 1 writes the CoT
// and never trained with a prefix; the CFG uncond branch must stay clean or the
// guidance cannot steer toward the artist.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

static bool qw3lm_has_prefix(const Qwen3LM * m) {
    return m && m->lora && m->lora->pfx_n > 0 && !m->lora->pfx_k.empty();
}

// Returns the number of prefix columns written (0 = nothing to seed).
static int qw3lm_seed_prefix(Qwen3LM * m, int kv_set) {
    if (!qw3lm_has_prefix(m)) {
        return 0;
    }
    const LMLora *        L   = m->lora;
    const Qwen3LMConfig & c   = m->cfg;
    const int             D   = c.head_dim;
    const int             Nkv = c.n_kv_heads;
    const int             n   = L->pfx_n;
    const int64_t         row = (int64_t) Nkv * D;

    if (L->pfx_row != row) {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "[LM-Prefix] row %lld in the adapter != model Nkv*D %lld — prefix NOT applied\n",
                    (long long) L->pfx_row, (long long) row);
        }
        return 0;
    }
    if (n >= c.max_seq_len) {
        fprintf(stderr, "[LM-Prefix] n=%d does not fit the KV cache (max_seq %d) — prefix NOT applied\n", n,
                c.max_seq_len);
        return 0;
    }

    std::vector<ggml_fp16_t> blk((size_t) n * (size_t) D);
    for (int l = L->pfx_lo; l < L->pfx_hi && l < c.n_layers; l++) {
        const std::vector<float> & pk = L->pfx_k[(size_t) l];
        const std::vector<float> & pv = L->pfx_v[(size_t) l];
        if (pk.size() != (size_t) n * (size_t) row || pv.size() != pk.size()) {
            continue;  // malformed layer: the loader already warned
        }
        ggml_tensor * ck = m->kv_k[kv_set][l];
        ggml_tensor * cv = m->kv_v[kv_set][l];
        for (int which = 0; which < 2; which++) {
            const std::vector<float> & src = which ? pv : pk;
            ggml_tensor *              dst = which ? cv : ck;
            for (int h = 0; h < Nkv; h++) {
                // [D, n] block for head h: column col, dims d
                for (int col = 0; col < n; col++) {
                    const float * s = src.data() + (size_t) col * (size_t) row + (size_t) h * D;
                    ggml_fp16_t * d = blk.data() + (size_t) col * D;
                    for (int i = 0; i < D; i++) {
                        d[i] = ggml_fp32_to_fp16(s[i]);
                    }
                }
                ggml_backend_tensor_set(dst, blk.data(), (size_t) h * dst->nb[2], blk.size() * sizeof(ggml_fp16_t));
            }
        }
    }
    m->kv_pos[kv_set]  = n;
    m->kv_base[kv_set] = n;
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[LM-Prefix] seeded %d columns over layers [%d,%d) into kv set %d\n", n, L->pfx_lo,
                L->pfx_hi, kv_set);
    }
    return n;
}
