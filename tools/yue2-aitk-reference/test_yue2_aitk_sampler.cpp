#include "../../engine/src/train/yue2-aitk-sampler.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

int main(int argc, char ** argv) {
    yue2_aitk::SongInput song; song.semantic_tokens.resize(2000); song.latents.resize(2000 * 64);
    for (size_t i = 0; i < song.semantic_tokens.size(); ++i) song.semantic_tokens[i] = int32_t(i % 32768);
    for (size_t i = 0; i < song.latents.size(); ++i) song.latents[i] = .03137f * float(int(i % 13) - 6);
    yue2_aitk::PromptInput prompt; prompt.retained_prefix_ids = {1, 2, 3}; prompt.dropped_prefix_ids = {4, 5}; prompt.abc_ids = {9, 10};
    yue2_aitk::Yue2NativeSampler first(20260917); const auto schedule = first.sigmoid_schedule(32); const auto one = first.sample(song, prompt, 1500, schedule);
    if (one.crop.end - one.crop.start > 1500 || one.batch.nar.latents.size() != (one.crop.end - one.crop.start) * 64 || one.noisy_bf16.size() != one.noise_f32.size()) return 2;
    for (size_t i = 0; i < one.target_f32.size(); ++i) if (one.target_f32[i] != yue2_aitk::bf16_round_f32(one.noise_f32[i] - one.clean_f32[i])) return 3;
    if (argc > 1) {
        std::ofstream dump(argv[1], std::ios::binary | std::ios::trunc);
        if (!dump) return 9;
        dump << std::setprecision(9) << "{\"crop_start\":" << one.crop.start << ",\"crop_end\":" << one.crop.end
             << ",\"timestep\":" << one.timestep << ",\"clean_bf16\":[";
        for (size_t i = 0; i < one.clean_f32.size(); ++i) dump << (i ? "," : "") << one.clean_f32[i];
        dump << "],\"noise_bf16\":[";
        for (size_t i = 0; i < one.noise_f32.size(); ++i) dump << (i ? "," : "") << one.noise_f32[i];
        dump << "],\"target_bf16\":[";
        for (size_t i = 0; i < one.target_f32.size(); ++i) dump << (i ? "," : "") << one.target_f32[i];
        dump << "],\"noisy_f32_mixed\":[";
        for (size_t i = 0; i < one.noisy_f32.size(); ++i) dump << (i ? "," : "") << one.noisy_f32[i];
        dump << "],\"noisy_bf16\":[";
        for (size_t i = 0; i < one.noisy_bf16.size(); ++i) dump << (i ? "," : "") << one.noisy_bf16[i];
        dump << "]}\n";
        if (!dump) return 10;
    }
    (void) first.rng().normal01(); // leave an odd Box-Muller spare in the serialized stream
    const std::string checkpoint = first.export_rng_state();
    const auto two = first.sample(song, prompt, 1500, schedule);
    yue2_aitk::Yue2NativeSampler resumed(1); if (!resumed.import_rng_state(checkpoint)) return 4;
    const std::string before_invalid = resumed.export_rng_state(); if (resumed.import_rng_state("Y2SAMPLER1 malformed")) return 6; if (resumed.export_rng_state() != before_invalid) return 7;
    const auto two_resumed = resumed.sample(song, prompt, 1500, schedule);
    if (two.crop.start != two_resumed.crop.start || two.crop.end != two_resumed.crop.end || two.abc_retained != two_resumed.abc_retained || two.timestep != two_resumed.timestep || two.noise_f32 != two_resumed.noise_f32) return 5;
    // Timestep draws are with replacement; the serialized cursor may exceed
    // the schedule length without implying an epoch transition.
    for (int i = 0; i < 40; ++i) (void) resumed.sample(song, prompt, 1500, schedule);
    const std::string long_cursor = resumed.export_rng_state();
    yue2_aitk::Yue2NativeSampler long_cursor_copy(2); if (long_cursor.empty() || !long_cursor_copy.import_rng_state(long_cursor)) return 8;
    const auto next_epoch = resumed.sigmoid_schedule(8); (void) resumed.sample(song, prompt, 1500, next_epoch);
    std::puts("YuE2 native sampler crop/dropout/sigmoid/Box-Muller resume fixture passed"); return 0;
}
