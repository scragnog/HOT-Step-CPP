// GH #367 follow-up: a semantic draw outside the codec range ends the stream.
// The sampler's fallback for a row with no survivors is index 0, which on a
// full head (ConvRot, imatrix; base 0) is text token 0. Uses the pipeline's
// own mapping and sampler. No GPU, no models.
#include "yue2/yue2-pipeline.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    int  fail  = 0;
    auto check = [&](bool ok, const char * what) {
        if (!ok) {
            printf("FAIL %s\n", what);
            fail++;
        }
    };
    const int64_t lo = YUE2_CODEC_OFFSET, hi = YUE2_CODEC_OFFSET + YUE2_CODEC_SIZE;

    check(yue2_semantic_draw_id(-1, 0) == YUE2_MUSIC_END, "threshold end");
    check(yue2_semantic_draw_id(0, 0) == YUE2_MUSIC_END, "full head index 0");
    check(yue2_semantic_draw_id(0, YUE2_SEM_HEAD_LO) == YUE2_MUSIC_END, "window index 0");
    check(yue2_semantic_draw_id(YUE2_MUSIC_END, 0) == YUE2_MUSIC_END, "full head MUSIC_END");
    check(yue2_semantic_draw_id(lo, 0) == lo, "full head first codec");
    check(yue2_semantic_draw_id(hi - 1, 0) == hi - 1, "full head last codec");
    check(yue2_semantic_draw_id(hi, 0) == YUE2_MUSIC_END, "full head past the codec range");
    check(yue2_semantic_draw_id(lo - YUE2_SEM_HEAD_LO, YUE2_SEM_HEAD_LO) == lo, "window first codec");

    // The sampler itself on a fully masked full-head row: it falls back to
    // index 0, and the mapping must end the stream instead of emitting a codec.
    Yue2SamplingParams sp;
    sp.temperature = 0.85f;
    sp.top_k       = 50;
    sp.top_p       = 0.9f;
    std::vector<float>   row((size_t) hi + 8, -INFINITY);
    std::vector<int32_t> history, ids;
    std::vector<float>   vals;
    std::mt19937_64      rng(823445542);
    const int64_t rel = yue2_sample_row(row.data(), (int64_t) row.size(), sp, YUE2_MUSIC_END, lo, hi, history, 0,
                                        false, 0, rng, &ids, &vals);
    check(yue2_semantic_draw_id(rel, 0) == YUE2_MUSIC_END, "masked full-head row ends the stream");

    // And a normal full-head row still draws its codec.
    std::vector<float> ok_row((size_t) hi + 8, 0.0f);
    ok_row[(size_t) lo + 7] = 40.0f;
    const int64_t rel2 = yue2_sample_row(ok_row.data(), (int64_t) ok_row.size(), sp, YUE2_MUSIC_END, lo, hi, history,
                                         0, false, 0, rng, &ids, &vals);
    check(yue2_semantic_draw_id(rel2, 0) == lo + 7, "full-head row draws its codec");

    printf(fail ? "yue2-semantic-draw-test: FAIL (%d)\n" : "yue2-semantic-draw-test: PASS\n", fail);
    return fail ? 1 : 0;
}
