// mm3_vram_fit (minimax/mm3-vram-fit.h): the MM3 planning-stage VRAM ladder.
// No GPU or models. Exit 0 = every case passed.
#include "../src/minimax/mm3-vram-fit.h"

#include <cstdio>

static const size_t GiB = (size_t) 1 << 30;
static const size_t MiB = (size_t) 1 << 20;
// One f16 KV row at 7680 positions: 36 layers x 8 KV heads x 128 x K+V x 2 bytes
// per position (147,456 bytes). The GH #324 log's failed allocation was 6 of these.
static const size_t ROW_7680 = (size_t) 7680 * 147456;
static_assert((size_t) 7680 * 147456 * 6 == 6794772480ull, "row bytes must match the GH #324 allocation");

static int fails = 0;

static void expect(const char * name, const MM3VramFit & f, bool ok, bool q8, int takes) {
    const bool pass = f.ok == ok && (!ok || (f.q8 == q8 && f.takes == takes));
    printf("%s: %s - ok=%d q8=%d takes=%d need=%.2f GiB\n", pass ? "PASS" : "FAIL", name, f.ok, f.q8, f.takes,
           (double) f.need / (double) GiB);
    fails += pass ? 0 : 1;
}

int main() {
    // GH #324: GTX 1080, 5.50 GiB of LM + depth, 7.50 GiB free, one song with
    // 3 ending candidates (CFG pair each), f16 asked. Only Q8_0 + 1 fits.
    const size_t w = (size_t) (5.50 * (double) GiB), b = (size_t) (7.50 * (double) GiB);
    expect("GH #324, one song", mm3_vram_fit(b, w, ROW_7680, 2, 3, false, true, true), true, true, 1);
    // q8 + 2 and f16 + 1 must both miss, or the ladder would stop early.
    if (mm3_fit_need(w, ROW_7680, 4, true) <= b || mm3_fit_need(w, ROW_7680, 2, false) <= b) {
        printf("FAIL: q8 + 2 or f16 + 1 fits the GH #324 budget\n");
        fails++;
    }
    // Same card, three variations asked: candidates are not cut, so refuse.
    expect("GH #324, 3 variations", mm3_vram_fit(b, w, ROW_7680, 2, 3, false, true, false), false, false, 0);
    // MM3_LM_KV_TYPE=f16 pins the cache: one song still has nowhere to go.
    expect("GH #324, KV pinned f16", mm3_vram_fit(b, w, ROW_7680, 2, 3, false, false, true), false, false, 0);
    // Budget 6.0: even Q8_0 + 1 (6.62 GiB) does not fit.
    expect("budget 6.0", mm3_vram_fit(6 * GiB, w, ROW_7680, 2, 3, false, true, true), false, false, 0);
    // 13 Nov 4090 receipt (HOTSTEP-1197): one song, 2 candidates, plan dial
    // below 1 (4 rows per take), f16 KV, Q8_0 LM + depth 9.70 GiB. 24,564 MiB
    // card with a 1,038 MiB idle baseline. Completed; must come back unchanged.
    const size_t b4090 = (size_t) (24564 - 1038) * MiB, w4090 = (size_t) (9.70 * (double) GiB);
    expect("4090 f16, 13 Nov", mm3_vram_fit(b4090, w4090, ROW_7680, 4, 2, false, true, true), true, false, 2);
    // A q8 request that fits stays as sent.
    expect("q8 asked, fits", mm3_vram_fit(b4090, w4090, ROW_7680, 4, 2, true, true, true), true, true, 2);
    printf("%s\n", fails ? "MM3 VRAM FIT: FAIL" : "MM3 VRAM FIT: PASS");
    return fails ? 1 : 0;
}
