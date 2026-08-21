#pragma once
// minimax/mm3-ar-cache.h — the MM3 AR cache: one slot of stage-1 output.
//
// HOT-Step file (does not exist upstream).
//
// ── What it is ──────────────────────────────────────────────────────────────
//
// Holds the previous render's frame-hidden block so a render that changes only
// FLOW-side settings can skip the AR stage outright. That is roughly half a
// render's wall clock on this machine (AR 72-88 s vs flow 59-101 s on a 200 s
// song), plus the staged LM load and free either side of it.
//
// ── Why the hiddens and not the codes ───────────────────────────────────────
//
// The flow DiT never sees the AR codes. Its real input is the
// [F, num_codebooks, embedding_length] hidden block the condition encoder
// consumes (mm3-pipeline.h). That is exactly why the MM3 Plank
// (forced_semantic / forced_acoustic) does NOT save time: pinning the codes
// still re-runs every per-frame forward pass to regenerate these hiddens.
// Caching the hiddens is the thing that actually skips the work. The two
// features are complementary — the plank is portable and reproducible across
// restarts, this is fast and in-process.
//
// ── One slot, borrowed, never copied ────────────────────────────────────────
//
// On a hit the pipeline reads this block through a const pointer for the whole
// call; on a miss the finished run's own block is MOVED in. Host RAM therefore
// holds exactly one copy at a time — 128 KB per frame at f32, so ~600 MB for a
// 200 s song. That size is the whole reason this is opt-in (`reuse_ar`) and a
// single slot rather than an LRU.
//
// ── Lifetime ────────────────────────────────────────────────────────────────
//
// This is HOST memory, so it deliberately does NOT hang off mm3_unload(): that
// runs after every generation when "keep models loaded" is off, and hooking it
// there would drop the cache before the next render could ever hit it. The
// slot is dropped explicitly instead — by POST /mm3/unload, by a model-role
// change, and by the next miss.
//
// Guarded by g_mm3_mutex like every other MM3 global.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct MM3ArCache {
    /** The full AR-affecting input set as one exact string, compared verbatim.
     *  "" = empty slot. Built by mm3_ar_cache_key() (mm3-job.h), which is where
     *  the correctness argument for this whole feature lives. */
    std::string          key;
    int64_t              frames = 0;
    /** [frames, num_codebooks, embedding_length], layer-major. */
    std::vector<float>   hiddens;
    // Stage-1 byproducts, cached so a hit is indistinguishable from a fresh
    // plan downstream (plank capture, x-lrc-text, the job detail).
    std::vector<int32_t> semantic_all;
    std::vector<int32_t> acoustic_all;
    std::string          lrc;
    bool                 eos_hit = false;
    /** How many renders this slot has served — logged, and worth watching:
     *  a cache that never reaches 1 means something is keying wrong. */
    int64_t              hits = 0;
};

static MM3ArCache g_mm3_ar_cache;

static size_t mm3_ar_cache_bytes() {
    return g_mm3_ar_cache.hiddens.size() * sizeof(float) +
           (g_mm3_ar_cache.semantic_all.size() + g_mm3_ar_cache.acoustic_all.size()) * sizeof(int32_t);
}

/** Drop the slot and actually give the memory back — move-assigning from a
 *  fresh instance destroys the old vectors rather than merely clearing them,
 *  which would keep the capacity. */
static void mm3_ar_cache_clear(const char * why) {
    if (g_mm3_ar_cache.key.empty() && g_mm3_ar_cache.hiddens.empty()) {
        return;
    }
    fprintf(stderr, "[MM3-ARCache] dropped %.0f MB (%s)\n", (double) mm3_ar_cache_bytes() / 1048576.0,
            why ? why : "cleared");
    g_mm3_ar_cache = MM3ArCache{};
}
