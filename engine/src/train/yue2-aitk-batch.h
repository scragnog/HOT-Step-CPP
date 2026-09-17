#pragma once

// Deterministic batch construction for the pinned AI Toolkit YuE2 recipe.
// Reference: extensions_built_in/audio_models/yue2/yue2_model.py:447-456
// (_ar_inputs), :515-576 (AR/NAR windowing and detached KV), and
// src/tokenizer.py:43-62 (prefix and ABC/music tail). This file deliberately
// accepts ABC-dropout and NAR-crop decisions from its caller; it has no RNG.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "yue2-aitk-cursor-data.h"

namespace yue2_aitk {

// src/model.py:25-38 in the pinned reference.
constexpr int32_t kAbcEnd = 151848;
constexpr int32_t kMusicStart = 151851;
constexpr int32_t kMusicEnd = 151852;
constexpr int32_t kCodecOffset = 151853;
constexpr int32_t kEod = 151643;
constexpr int32_t kVocabSize = 184704;
constexpr int32_t kCodecSize = 32768;
constexpr size_t kLatentChannels = 64;

struct FrameRange {
    size_t start = 0;
    size_t end = 0; // exclusive
};

struct SongInput {
    // One semantic/codec token per latent frame, as produced by the reference
    // latent cache. Latents are flattened frame-major [frames, 64].
    std::vector<int32_t> semantic_tokens;
    std::vector<float> latents;
    size_t latent_channels = kLatentChannels;
    std::string style;
    std::string lyrics;
    bool instrumental = false;
};

struct PromptInput {
    // These are already tokenizer outputs. retained_prefix_ids is the
    // prefix_head_ids(..., cot=selected mode) result; dropped_prefix_ids is
    // prefix_head_ids(..., cot="off") from yue2_model.py:540-545.
    std::vector<int32_t> retained_prefix_ids;
    std::vector<int32_t> dropped_prefix_ids;
    std::vector<int32_t> abc_ids;
    bool retain_abc = true; // stochastic decision is made by the caller
    // Optional exact cursor binding.  When absent, schema-1 training has no
    // cursor term and keeps the historical behavior.
    CursorMetadata cursor;
};

struct ArSequence {
    std::vector<int32_t> prefix_ids;
    std::vector<int32_t> suffix_ids; // shifted teacher-forced targets
    std::vector<int32_t> input_ids;  // prefix embeddings followed by suffix embeddings
    std::vector<int32_t> target_ids; // one target for every prediction row
    std::vector<size_t> prediction_positions; // shifted rows in input_ids
    std::vector<uint8_t> target_mask; // one at prediction_positions, zero elsewhere
    bool appended_music_end = false;
};

struct NarCondition {
    FrameRange frames;
    std::vector<int32_t> semantic_tokens;
    std::vector<float> latents;
    ArSequence ar;
};

struct Batch {
    ArSequence ar;
    NarCondition nar;
    bool abc_retained = false;
};

inline bool fail(std::string * error, const char * message) {
    if (error) *error = message;
    return false;
}

inline bool validate_song(const SongInput & song, std::string * error) {
    if (song.latent_channels != kLatentChannels) return fail(error, "YuE2 latents require exactly 64 channels");
    if (song.semantic_tokens.empty()) return fail(error, "semantic token sequence is empty");
    if (song.semantic_tokens.size() > std::numeric_limits<size_t>::max() / song.latent_channels)
        return fail(error, "latent frame count overflows the flattened buffer");
    if (song.latents.size() != song.semantic_tokens.size() * song.latent_channels)
        return fail(error, "semantic and latent frame counts are misaligned");
    for (int32_t token : song.semantic_tokens)
        if (token < 0 || token >= kCodecSize) return fail(error, "raw codec token is outside the codec vocabulary");
    return true;
}

inline bool validate_range(FrameRange range, size_t frames, std::string * error) {
    if (range.start >= range.end || range.end > frames)
        return fail(error, "NAR crop is outside the aligned song frames");
    return true;
}

inline ArSequence make_ar_sequence(const PromptInput & prompt,
                                   const std::vector<int32_t> & song_tokens,
                                   bool append_music_end) {
    ArSequence out;
    out.prefix_ids = prompt.retain_abc ? prompt.retained_prefix_ids : prompt.dropped_prefix_ids;
    if (prompt.retain_abc)
        out.suffix_ids.insert(out.suffix_ids.end(), prompt.abc_ids.begin(), prompt.abc_ids.end());
    out.suffix_ids.push_back(kAbcEnd);
    out.suffix_ids.push_back(kMusicStart);
    for (int32_t token : song_tokens) out.suffix_ids.push_back(token + kCodecOffset);
    if (append_music_end) {
        out.suffix_ids.push_back(kMusicEnd);
        out.appended_music_end = true;
    }
    out.input_ids.reserve(out.prefix_ids.size() + out.suffix_ids.size());
    out.input_ids.insert(out.input_ids.end(), out.prefix_ids.begin(), out.prefix_ids.end());
    out.input_ids.insert(out.input_ids.end(), out.suffix_ids.begin(), out.suffix_ids.end());
    out.target_ids = out.suffix_ids;
    if (out.prefix_ids.empty()) return out;
    // _ar_losses slices hidden[-n-1:-1], so the first prediction row is the
    // final prefix position, followed by each preceding suffix position.
    out.prediction_positions.reserve(out.suffix_ids.size());
    out.target_mask.assign(out.input_ids.size(), 0);
    for (size_t i = 0; i < out.suffix_ids.size(); ++i) {
        const size_t position = out.prefix_ids.size() - 1 + i;
        out.prediction_positions.push_back(position);
        out.target_mask[position] = 1;
    }
    return out;
}

// ar_token_limit is an explicit caller choice; zero means the reference's
// ar_max_tokens <= 0 case (the complete song). Otherwise the caller supplies
// the already-resolved limit. The reference uses min(total, ar_max_tokens), and appends
// MUSIC_END only when that limit reaches the complete song (yue2_model.py:552-565).
inline bool build(const PromptInput & prompt, const SongInput & song,
                  FrameRange nar_range, size_t ar_token_limit, Batch * out,
                  std::string * error = nullptr) {
    if (!out) return fail(error, "output batch is null");
    if (!validate_song(song, error)) return false;
    if (!validate_range(nar_range, song.semantic_tokens.size(), error)) return false;
    const auto & prefix = prompt.retain_abc ? prompt.retained_prefix_ids : prompt.dropped_prefix_ids;
    if (prefix.empty()) return fail(error, "selected prompt prefix is empty");
    for (int32_t id : prefix)
        if (id < 0 || id >= kVocabSize) return fail(error, "prompt prefix token is outside the vocabulary");
    if (prompt.retain_abc) {
        for (int32_t id : prompt.abc_ids)
            if (id < 0 || id >= kVocabSize) return fail(error, "ABC token is outside the vocabulary");
    }
    // The Python path truncates only for a non-whole-song NAR window. Its
    // ar_max_tokens <= 0 case means the complete song; positive values clamp.
    const bool nar_whole_song = nar_range.start == 0 && nar_range.end == song.semantic_tokens.size();
    if (nar_whole_song) ar_token_limit = song.semantic_tokens.size();
    else if (ar_token_limit == 0) ar_token_limit = song.semantic_tokens.size();
    else if (ar_token_limit > song.semantic_tokens.size()) ar_token_limit = song.semantic_tokens.size();
    const bool ar_full = ar_token_limit == song.semantic_tokens.size();
    std::vector<int32_t> ar_tokens(song.semantic_tokens.begin(), song.semantic_tokens.begin() + ar_token_limit);
    std::vector<int32_t> nar_tokens(song.semantic_tokens.begin() + nar_range.start,
                                    song.semantic_tokens.begin() + nar_range.end);
    out->ar = make_ar_sequence(prompt, ar_tokens, ar_full);
    out->nar.frames = nar_range;
    out->nar.semantic_tokens = nar_tokens;
    const size_t offset = nar_range.start * song.latent_channels;
    const size_t count = (nar_range.end - nar_range.start) * song.latent_channels;
    out->nar.latents.assign(song.latents.begin() + offset, song.latents.begin() + offset + count);
    // Reference NAR conditioning always passes end_token=True for its crop
    // (yue2_model.py:549-550), including a crop that is not the true song end.
    out->nar.ar = make_ar_sequence(prompt, nar_tokens, true);
    out->abc_retained = prompt.retain_abc;
    return true;
}

} // namespace yue2_aitk
