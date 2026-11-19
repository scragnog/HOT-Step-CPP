// Isolated CPU fixture for the CTC aligner: the trainer parity gate on
// yue2_cursor_align, and the star lattice the align endpoint uses.
//
// No engine build, no model, no audio — the aligner's second half is pure CPU
// arithmetic on a log-probability matrix, so a synthetic matrix is enough to pin
// both behaviours:
//
//   g++ -O2 -std=c++17 -o /tmp/test-yue2-ctc tools/yue2-ctc-align/test_yue2_ctc_align.cpp
//   /tmp/test-yue2-ctc
#include "../../engine/src/yue2/yue2-ctc-align.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int failures = 0;

static bool check(bool value, const char * label) {
    if (!value) {
        fprintf(stderr, "FAIL: %s\n", label);
        failures++;
    }
    return value;
}

// A realistic per-frame log-probability row: one confident letter, blank
// implausible (somebody is singing), everything else a weak possibility.
static void frame(std::vector<float> & lp, int t, int C, int loud) {
    double p[16];
    double sum = 0.0;
    for (int c = 0; c < C; c++) {
        p[c] = (c == 0) ? 0.01 : 0.02;
        sum += p[c];
    }
    if (loud > 0 && loud < C) {
        sum -= p[loud];
        p[loud] = 0.9;
        sum += p[loud];
    }
    for (int c = 0; c < C; c++) {
        lp[(size_t) t * C + c] = (float) std::log(p[c] / sum);
    }
}

// The synthetic matrix the golden rows below were recorded from: four fixed
// seeds, 60 frames, eight labels. Reproducing the LCG exactly is what makes the
// comparison a gate rather than a suggestion.
static std::vector<float> synthetic(unsigned seed, int T, int C) {
    std::vector<float> lp((size_t) T * C);
    unsigned           s = seed;
    for (size_t i = 0; i < lp.size(); i++) {
        s = s * 1664525u + 1013904223u;
        lp[i] = (float) (((s >> 8) % 1000) / 1000.0 + 0.001);
    }
    for (int t = 0; t < T; t++) {
        double sum = 0.0;
        for (int c = 0; c < C; c++) sum += lp[(size_t) t * C + c];
        for (int c = 0; c < C; c++) lp[(size_t) t * C + c] = (float) std::log(lp[(size_t) t * C + c] / sum);
    }
    return lp;
}

struct Row {
    double start, end, score, c0, c1;
};

// Recorded from the header BEFORE the star lattice existed. A change to
// yue2_cursor_align's arithmetic has to show up here, not in a retrained
// adapter: these rows are the AR trainer's `cursor_words` and the file's header
// says they are gated against torchaudio's own output.
static void trainer_gate() {
    static const Row golden[4][2] = {
        { { 0.000000, 0.140000, 0.177023, 0, 2 }, { 0.140000, 0.660000, 0.155663, 3, 5 } },
        { { 0.040000, 0.260000, 0.188746, 0, 2 }, { 0.400000, 1.040000, 0.161582, 3, 5 } },
        { { 0.000000, 0.160000, 0.154528, 0, 2 }, { 0.420000, 0.980000, 0.170528, 3, 5 } },
        { { 0.060000, 0.660000, 0.154144, 0, 2 }, { 0.920000, 1.200000, 0.171866, 3, 5 } },
    };
    const int                 T = 60, C = 8;
    std::vector<std::string>  labels = { "-", "a", "h", "i", "o", "s", "t", "x" };

    for (unsigned seed = 1; seed <= 4; seed++) {
        std::vector<float> lp = synthetic(seed, T, C);
        std::vector<float> rows;
        std::string        err;
        const bool ok = yue2_cursor_align(lp.data(), T, C, (double) T * 0.02, labels, "hi\nox", &rows, &err);
        check(ok, "yue2_cursor_align returns rows");
        if (!check(rows.size() == 10, "two words, five floats each")) {
            continue;
        }
        for (int w = 0; w < 2; w++) {
            const float * got  = rows.data() + w * 5;
            const Row &   want = golden[seed - 1][w];
            char          label[96];
            snprintf(label, sizeof label, "seed %u word %d is unchanged", seed, w);
            check(std::fabs(got[0] - want.start) < 1e-5f && std::fabs(got[1] - want.end) < 1e-5f &&
                      std::fabs(got[2] - want.score) < 1e-5f && std::fabs(got[3] - want.c0) < 1e-5f &&
                      std::fabs(got[4] - want.c1) < 1e-5f,
                  label);
        }
    }
}

// Two lines, "hi" then "om". With `ad_lib`, frames 8..23 are a clear sung "ah"
// that the sheet does not contain; without it the whole utterance is the lyric.
static std::vector<float> scenario(int T, int C, bool ad_lib) {
    std::vector<float> lp((size_t) T * C);
    for (int t = 0; t < T; t++) {
        int loud;
        if (ad_lib) {
            if (t < 4) loud = 2;         // h
            else if (t < 8) loud = 3;    // i
            else if (t < 24) loud = 1;   // a — the ad-lib, not in the lyrics
            else if (t < 32) loud = 4;   // o
            else loud = 5;               // m
        } else {
            if (t < 10) loud = 2;        // h
            else if (t < 20) loud = 3;   // i
            else if (t < 30) loud = 4;   // o
            else loud = 5;               // m
        }
        frame(lp, t, C, loud);
    }
    return lp;
}

static void star_lattice() {
    const int                T = 40, C = 6;
    std::vector<std::string> labels = { "-", "a", "h", "i", "o", "m" };
    const double             fps    = 0.02;
    std::vector<float>       lp     = scenario(T, C, /*ad_lib=*/true);

    std::vector<float> plain;
    std::string        err;
    if (!check(yue2_cursor_align(lp.data(), T, C, T * fps, labels, "hi\nom", &plain, &err), "plain alignment runs")) {
        return;
    }
    std::vector<float> starred;
    if (!check(yue2_cursor_align_star(lp.data(), T, C, T * fps, labels, "hi\nom", YUE2_ALIGN_STAR_MARGIN, &starred, &err),
               "star alignment runs")) {
        return;
    }
    if (!check(plain.size() == 10 && starred.size() == 10, "two words either way")) {
        return;
    }

    // The ad-lib starts at frame 8 (0.16 s), the real second line at frame 24
    // (0.48 s). blank is implausible on all of it — somebody is singing — so the
    // plain lattice's cheapest filler is to hold "o" and "m" over the ad-lib and
    // call the second line done. That is the failure the star exists for.
    printf("plain: word 1 %.2f-%.2f  word 2 %.2f-%.2f\n", plain[0], plain[1], plain[5], plain[6]);
    printf("star:  word 1 %.2f-%.2f  word 2 %.2f-%.2f\n", starred[0], starred[1], starred[5], starred[6]);

    check(plain[5] < 0.30f, "the plain lattice stretches word 2 into the ad-lib");
    check(starred[5] > 0.44f, "the star lattice starts word 2 on its own line");
    check(starred[5] - plain[5] > 0.20f, "the two lattices really do differ here");
    check(starred[1] <= 8 * fps + 1e-4f, "and word 1 still ends where its lyric does");

    // On audio that IS the lyric there is nothing to absorb, so the star must
    // leave the words where the trainer-path lattice had them.
    std::vector<float> clean = scenario(T, C, /*ad_lib=*/false);
    std::vector<float> clean_plain, clean_star;
    if (!check(yue2_cursor_align(clean.data(), T, C, T * fps, labels, "hi\nom", &clean_plain, &err), "clean plain")) {
        return;
    }
    if (!check(yue2_cursor_align_star(clean.data(), T, C, T * fps, labels, "hi\nom", YUE2_ALIGN_STAR_MARGIN, &clean_star,
                                      &err),
               "clean star")) {
        return;
    }
    if (!check(clean_plain.size() == 10 && clean_star.size() == 10, "clean: two words either way")) {
        return;
    }
    for (size_t i = 0; i < 10; i++) {
        if (i % 5 >= 3) continue;  // the times; the char offsets are the words' own
        check(std::fabs(clean_plain[i] - clean_star[i]) <= 2 * fps + 1e-4,
              "a lyric-only passage is aligned the same by both lattices");
    }
}

int main() {
    trainer_gate();
    star_lattice();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("yue2 ctc: trainer gate and star lattice OK\n");
    return 0;
}
