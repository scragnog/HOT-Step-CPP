#pragma once

// CPU-only self-check for the AR sequence layouts the joint trainer builds:
// the tuned target set, --ar-targets base, and the condition-dropout prefix
// variants, including the unconditional off prompt that has no ABC bracket.
// No model, no CUDA. `ace-train yue2-batch-check`.
#include "yue2-aitk-batch.h"

#include <cstdio>
#include <string>
#include <vector>

static int yue2_aitk_batch_check_main() {
    using namespace yue2_aitk;
    int failures = 0;
    const auto check = [&](bool ok, const char * what) { if (!ok) { ++failures; std::fprintf(stderr, "[yue2-batch-check] FAIL %s\n", what); } };
    const int32_t kAbcStart = 151847;
    SongInput song;
    song.semantic_tokens = {5, 6, 7, 8};
    song.latents.assign(4 * kLatentChannels, 0.25f);
    PromptInput p;
    p.retained_prefix_ids = {kEod, 10, 11, kAbcStart};
    p.dropped_prefix_ids = {kEod, 20, kAbcStart};
    p.abc_ids = {100, 101, 102};
    p.retained_nolyrics_prefix_ids = {kEod, 30, kAbcStart}; p.dropped_nolyrics_prefix_ids = {kEod, 31, kAbcStart};
    p.retained_notext_prefix_ids = {kEod, 40, kAbcStart};   p.dropped_notext_prefix_ids = {kEod, 41, kAbcStart};
    p.retained_uncond_prefix_ids = {kEod, 50, kAbcStart};   p.dropped_uncond_prefix_ids = {kEod, 51};
    const FrameRange whole{0, 4};
    std::string error;
    Batch b;

    // Tuned, sheet kept: every suffix token is a target.
    p.retain_abc = true; p.condition = PromptInput::Condition::kept;
    check(build(p, song, whole, 0, &b, &error), "tuned/kept builds");
    const std::vector<int32_t> tuned_kept = {100, 101, 102, kAbcEnd, kMusicStart, 5 + kCodecOffset, 6 + kCodecOffset, 7 + kCodecOffset, 8 + kCodecOffset, kMusicEnd};
    check(b.ar.target_ids == tuned_kept, "tuned/kept targets");
    check(b.ar.prediction_positions.size() == tuned_kept.size() && b.ar.prediction_positions[0] == p.retained_prefix_ids.size() - 1, "tuned/kept positions");

    // Base, sheet kept: MUSIC_START is not a target; ABC_END and MUSIC_END are.
    check(build(p, song, whole, 0, &b, &error, true), "base/kept builds");
    const std::vector<int32_t> base_kept = {100, 101, 102, kAbcEnd, 5 + kCodecOffset, 6 + kCodecOffset, 7 + kCodecOffset, 8 + kCodecOffset, kMusicEnd};
    check(b.ar.target_ids == base_kept, "base/kept targets");
    check(b.ar.prediction_positions.size() == base_kept.size() && b.ar.prediction_positions[3] == 6 && b.ar.prediction_positions[4] == 8, "base/kept positions skip MUSIC_START's row");
    check(b.ar.input_ids.size() == p.retained_prefix_ids.size() + tuned_kept.size(), "base/kept input unchanged");

    // Base, sheet dropped: off prompt already carries ABC_END and MUSIC_START.
    p.retain_abc = false;
    check(build(p, song, whole, 0, &b, &error, true), "base/dropped builds");
    const std::vector<int32_t> base_dropped = {5 + kCodecOffset, 6 + kCodecOffset, 7 + kCodecOffset, 8 + kCodecOffset, kMusicEnd};
    check(b.ar.target_ids == base_dropped, "base/dropped targets");
    check(b.ar.prediction_positions.size() == 5 && b.ar.prediction_positions[0] == p.dropped_prefix_ids.size() - 1 + 2, "base/dropped positions");

    // Tuned, sheet dropped: unchanged from before the flag.
    check(build(p, song, whole, 0, &b, &error), "tuned/dropped builds");
    const std::vector<int32_t> tuned_dropped = {kAbcEnd, kMusicStart, 5 + kCodecOffset, 6 + kCodecOffset, 7 + kCodecOffset, 8 + kCodecOffset, kMusicEnd};
    check(b.ar.target_ids == tuned_dropped, "tuned/dropped targets");

    // Condition variants select their own prefix; the bracket stays except uncond/off.
    p.retain_abc = true; p.condition = PromptInput::Condition::nolyrics;
    check(build(p, song, whole, 0, &b, &error), "nolyrics builds");
    check(b.ar.prefix_ids == p.retained_nolyrics_prefix_ids && b.ar.target_ids == tuned_kept, "nolyrics/kept layout");
    check(b.nar.ar.prefix_ids == p.retained_nolyrics_prefix_ids, "nolyrics conditions the NAR the same way");
    p.condition = PromptInput::Condition::notext; p.retain_abc = false;
    check(build(p, song, whole, 0, &b, &error), "notext builds");
    check(b.ar.prefix_ids == p.dropped_notext_prefix_ids && b.ar.target_ids == tuned_dropped, "notext/dropped layout");
    p.condition = PromptInput::Condition::uncond; p.retain_abc = true;
    check(build(p, song, whole, 0, &b, &error), "uncond/full builds");
    check(b.ar.prefix_ids == p.retained_uncond_prefix_ids && b.ar.target_ids == tuned_kept, "uncond/full keeps the ABC bracket");
    p.retain_abc = false;
    check(build(p, song, whole, 0, &b, &error), "uncond/off builds");
    const std::vector<int32_t> uncond_off = {kMusicStart, 5 + kCodecOffset, 6 + kCodecOffset, 7 + kCodecOffset, 8 + kCodecOffset, kMusicEnd};
    check(b.ar.prefix_ids == p.dropped_uncond_prefix_ids && b.ar.target_ids == uncond_off, "uncond/off has no ABC bracket");
    check(build(p, song, whole, 0, &b, &error, true), "uncond/off base builds");
    check(b.ar.target_ids == base_dropped, "uncond/off base drops only MUSIC_START");

    // A dataset without the variants refuses a condition draw instead of training on the wrong prompt.
    PromptInput old = p; old.retained_uncond_prefix_ids.clear();
    check(!build(old, song, whole, 0, &b, &error), "missing variants refused");

    if (failures) { std::fprintf(stderr, "[yue2-batch-check] %d failure(s)\n", failures); return 1; }
    std::printf("[yue2-batch-check] OK\n");
    return 0;
}
