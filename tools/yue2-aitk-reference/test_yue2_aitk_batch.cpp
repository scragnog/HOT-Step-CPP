// Deterministic target positions, ending policy and invalid input contracts.
#include "../../engine/src/train/yue2-aitk-batch.h"

static_assert(yue2_aitk::kCodecOffset == 151853, "pinned codec offset changed");
static_assert(yue2_aitk::kLatentChannels == 64, "pinned latent width changed");

int main() {
    yue2_aitk::PromptInput prompt;
    prompt.retained_prefix_ids = {151643, 151847};
    prompt.dropped_prefix_ids = {151643, 151847};
    prompt.abc_ids = {17, 18};
    yue2_aitk::SongInput song;
    song.semantic_tokens = {1, 2};
    song.latents.resize(2 * yue2_aitk::kLatentChannels);
    yue2_aitk::Batch batch;
    std::string error;
    if (!yue2_aitk::build(prompt, song, {0, 1}, 1, &batch, &error)) return 1;
    // Truncated AR has no artificial MUSIC_END; NAR conditioning always has it.
    if (batch.ar.appended_music_end || !batch.nar.ar.appended_music_end) return 2;
    if (batch.ar.target_ids.size() != 5 || batch.ar.prediction_positions.front() != 1) return 3;
    if (batch.ar.target_mask[0] != 0 || batch.ar.target_mask[1] != 1) return 4;
    if (batch.nar.semantic_tokens.size() != 1 || batch.nar.latents.size() != yue2_aitk::kLatentChannels) return 5;
    // A whole-song NAR crop forces full AR supervision even if the caller's
    // resolved limit is smaller, matching get_noise_prediction's whole_song branch.
    if (!yue2_aitk::build(prompt, song, {0, 2}, 1, &batch, &error)) return 6;
    if (!batch.ar.appended_music_end || batch.ar.target_ids.size() != 7) return 7;
    prompt.retain_abc = false;
    prompt.dropped_prefix_ids = {7, 8, 9};
    if (!yue2_aitk::build(prompt, song, {1, 2}, 99, &batch, &error)) return 8;
    if (batch.ar.target_ids != std::vector<int32_t>({151848,151851,151854,151855,151852})) return 9;
    if (batch.ar.prediction_positions != std::vector<size_t>({2,3,4,5,6})) return 10;
    if (batch.nar.ar.target_ids != std::vector<int32_t>({151848,151851,151855,151852})) return 11;
    song.semantic_tokens[0] = 32768;
    if (yue2_aitk::build(prompt, song, {0,1}, 0, &batch, &error)) return 12;
    song.semantic_tokens[0] = 1;
    song.latent_channels = 32;
    song.latents.resize(64);
    if (yue2_aitk::build(prompt, song, {0,1}, 0, &batch, &error)) return 13;
    return 0;
}
