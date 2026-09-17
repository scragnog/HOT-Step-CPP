// CPU-only contract fixture for the YuE2 cursor binding seam.
// Build/run is intentionally owned by the root agent.
#include "../../engine/src/train/yue2-aitk-cursor-data.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace yue2_aitk;

static bool expect(bool ok, const char * what) {
    if (!ok) std::cerr << "FAIL: " << what << "\n";
    return ok;
}

int main() {
    // The lyric stream represents an accented/multibyte lyric after the
    // tokenizer's UTF-8 codepoint accounting.  Ends deliberately cross token
    // boundaries, so the test exercises overlap rather than ASCII offsets.
    const std::vector<int32_t> full_prefix = {151643, 10, 11, 12, 13, 151848, 151849, 151850};
    const std::vector<int32_t> off_prefix  = {151643, 10, 11, 12, 13, 151848, 151849};
    const std::vector<int64_t> full_ends = {1, 2, 4, 6};
    const std::vector<int64_t> off_ends  = {1, 2, 4, 6};
    const std::vector<float> words = {
        0.0f, 0.08f, 0.99f, 0.0f, 2.0f,
        0.08f, 0.20f, 0.98f, 2.0f, 6.0f,
    };
    CursorTargets full, off;
    bool ok = true;
    ok &= expect(bind_cursor_targets(full_prefix, 2, full_ends, 6, 5, words, &full), "full binding");
    ok &= expect(bind_cursor_targets(off_prefix, 2, off_ends, 6, 5, words, &off), "off binding");
    ok &= expect(full.bound && full.j0 == 3 && full.L == 4 && full.nF == 5, "full geometry");
    ok &= expect(off.bound && off.j0 == 3 && off.L == full.L, "off geometry");
    std::vector<CursorTokenRange> ranges;
    cursor_ranges(full, &ranges);
    ok &= expect(ranges.size() == 5 && ranges[0].first == 0 && ranges[0].last == 2, "uniform frame range");
    ok &= expect(ranges[2].first == 2 && ranges[2].last == 4, "multibyte overlap range");

    CursorTargets stale;
    auto bad_ends = full_ends; bad_ends.back() = 5;
    ok &= expect(!bind_cursor_targets(full_prefix, 2, bad_ends, 6, 5, words, &stale), "stale lyric identity rejection");
    auto bad_words = words; bad_words[8] = 7.0f;
    ok &= expect(!bind_cursor_targets(full_prefix, 2, full_ends, 6, 5, bad_words, &stale), "stale span rejection");

    CursorMetadata instrumental;
    instrumental.present = false; instrumental.enabled = false; instrumental.instrumental = true;
    ok &= expect(instrumental.instrumental && !instrumental.present, "instrumental cursor exemption");
    CursorMetadata schema1;
    ok &= expect(!schema1.present && schema1.full_frame_ranges.empty(), "cursor-off schema1 compatibility");
    return ok ? 0 : 1;
}

