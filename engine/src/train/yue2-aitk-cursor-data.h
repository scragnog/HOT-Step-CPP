#pragma once

// Optional YuE2 lyric-cursor binding data.  This header is intentionally
// tokenizer-neutral: the tokenizer stage supplies the exact lyric-token end
// offsets and the prefix/head lengths after it has verified its own hashes.
// Old schema-1 records leave CursorMetadata::present false and keep the
// ordinary AR/NAR loss path unchanged.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace yue2_aitk {

struct CursorTargets {
    bool bound = false;
    int64_t j0 = 0; // first lyric-token column in the prefix embedding
    int64_t L = 0;  // number of lyric-token columns
    int64_t nF = 0; // number of codec frames carrying targets
    std::vector<float> T; // [nF, L], frame-contiguous
    std::string why;
};

struct CursorTokenRange {
    int32_t first = 0;
    int32_t last = 0; // exclusive; empty range is allowed for instrumental frames
};

struct CursorMetadata {
    bool present = false;
    bool enabled = true;
    bool instrumental = false;
    int64_t lyric_codepoints = 0;
    std::string lyrics_sha256;
    std::string tokenizer_sha256;
    // [word, start_seconds, end_seconds, confidence, char_begin, char_end]
    // is represented as five values per word, matching cursor_words.bin.
    std::vector<float> words5;
    int64_t full_head_tokens = 0;
    int64_t off_head_tokens = 0;
    std::vector<int64_t> full_lyric_token_end_codepoints;
    std::vector<int64_t> off_lyric_token_end_codepoints;
    CursorTargets full;
    CursorTargets off;
    int64_t j0_full = 0;
    int64_t j0_off = 0;
    int64_t L = 0;
    std::vector<CursorTokenRange> full_frame_ranges;
    std::vector<CursorTokenRange> off_frame_ranges;
};

using CursorBinding = CursorMetadata;

namespace cursor_detail {
inline bool cursor_fail(CursorTargets * out, const char * msg) {
    if (out) { out->bound = false; out->why = msg; out->T.clear(); }
    return false;
}

inline bool finite_words(const std::vector<float> & words5, int64_t lyric_codepoints,
                         std::string * why) {
    if (words5.empty() || words5.size() % 5 != 0) { if (why) *why = "cursor words must be nonempty [n,5]"; return false; }
    const size_t n = words5.size() / 5;
    for (size_t i = 0; i < n; ++i) {
        const float start = words5[i * 5 + 0], end = words5[i * 5 + 1];
        const float score = words5[i * 5 + 2];
        const float c0f = words5[i * 5 + 3], c1f = words5[i * 5 + 4];
        if (!std::isfinite(start) || !std::isfinite(end) || !std::isfinite(score) ||
            !std::isfinite(c0f) || !std::isfinite(c1f) || start < 0.0f || end < start ||
            c0f < 0.0f || c1f < c0f || c1f > static_cast<float>(lyric_codepoints) ||
            std::floor(c0f) != c0f || std::floor(c1f) != c1f)
            { if (why) *why = "cursor word span is invalid"; return false; }
        if (i && start < words5[(i - 1) * 5]) { if (why) *why = "cursor word starts are not sorted"; return false; }
    }
    return true;
}

inline bool token_offsets(const std::vector<int64_t> & ends, int64_t head, int64_t lyric_codepoints,
                          int64_t prefix_size, std::string * why) {
    if (head < 0 || ends.empty() || 1 + head + static_cast<int64_t>(ends.size()) > prefix_size ||
        ends.back() != lyric_codepoints) { if (why) *why = "cursor tokenizer offsets do not cover the lyrics"; return false; }
    int64_t previous = 0;
    for (int64_t end : ends) {
        if (end < previous || end < 0 || end > lyric_codepoints) { if (why) *why = "cursor token offsets are not monotonic"; return false; }
        previous = end;
    }
    return true;
}
}

// Bind one exact tokenizer prefix.  `token_end_codepoints` are decoded UTF-8
// codepoint ends for lyric tokens only.  `prefix_size` includes the leading
// BOS/EOD slot, so j0 is 1 + head_tokens as in ar_lora_cursor.py.
inline bool bind_cursor_targets(const std::vector<int32_t> & prefix_ids,
                                int64_t head_tokens,
                                const std::vector<int64_t> & token_end_codepoints,
                                int64_t lyric_codepoints, int64_t n_frames,
                                const std::vector<float> & words5,
                                CursorTargets * out, int64_t max_frames = 0) {
    if (!out) return false;
    *out = CursorTargets{};
    std::string why;
    if (n_frames <= 0 || lyric_codepoints <= 0 || !cursor_detail::finite_words(words5, lyric_codepoints, &why) ||
        !cursor_detail::token_offsets(token_end_codepoints, head_tokens, lyric_codepoints,
                                      static_cast<int64_t>(prefix_ids.size()), &why))
        return cursor_detail::cursor_fail(out, why.empty() ? "invalid cursor binding input" : why.c_str());
    const int64_t L = static_cast<int64_t>(token_end_codepoints.size());
    const int64_t nF = max_frames > 0 ? std::min(n_frames, max_frames) : n_frames;
    if (nF <= 0 || static_cast<uint64_t>(L) > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(nF))
        return cursor_detail::cursor_fail(out, "cursor target size overflows");
    const size_t n_words = words5.size() / 5;
    std::vector<std::vector<int32_t>> tokens(n_words);
    for (size_t w = 0; w < n_words; ++w) {
        const int64_t c0 = static_cast<int64_t>(words5[w * 5 + 3]);
        const int64_t c1 = static_cast<int64_t>(words5[w * 5 + 4]);
        for (int64_t k = 0, start = 0; k < L; ++k) {
            start = k ? token_end_codepoints[static_cast<size_t>(k - 1)] : 0;
            if (start < c1 && token_end_codepoints[static_cast<size_t>(k)] > c0)
                tokens[w].push_back(static_cast<int32_t>(k));
        }
        if (tokens[w].empty()) {
            const auto it = std::lower_bound(token_end_codepoints.begin(), token_end_codepoints.end(), c0);
            const int64_t k = std::min<int64_t>(L - 1, static_cast<int64_t>(it - token_end_codepoints.begin()));
            tokens[w].push_back(static_cast<int32_t>(std::max<int64_t>(0, k)));
        }
    }
    std::vector<float> starts(n_words);
    for (size_t w = 0; w < n_words; ++w) starts[w] = words5[w * 5];
    out->T.assign(static_cast<size_t>(nF * L), 0.0f);
    for (int64_t frame = 0; frame < nF; ++frame) {
        const float seconds = static_cast<float>(frame) / 25.0f;
        const size_t w = static_cast<size_t>(std::max<int64_t>(0, std::min<int64_t>(static_cast<int64_t>(n_words) - 1,
            static_cast<int64_t>(std::upper_bound(starts.begin(), starts.end(), seconds) - starts.begin()) - 1)));
        const float value = 1.0f / static_cast<float>(tokens[w].size());
        for (int32_t k : tokens[w]) out->T[static_cast<size_t>(frame * L + k)] = value;
    }
    out->bound = true;
    out->j0 = 1 + head_tokens;
    out->L = L;
    out->nF = nF;
    return true;
}

inline void cursor_ranges(const CursorTargets & targets, std::vector<CursorTokenRange> * out) {
    if (!out) return;
    out->assign(static_cast<size_t>(std::max<int64_t>(0, targets.nF)), CursorTokenRange{});
    for (int64_t frame = 0; frame < targets.nF; ++frame) {
        int64_t first = targets.L, last = 0;
        for (int64_t token = 0; token < targets.L; ++token) {
            if (targets.T[static_cast<size_t>(frame * targets.L + token)] > 0.0f) {
                first = std::min(first, token); last = std::max(last, token + 1);
            }
        }
        (*out)[static_cast<size_t>(frame)] = {static_cast<int32_t>(first == targets.L ? 0 : first), static_cast<int32_t>(last)};
    }
}

} // namespace yue2_aitk
