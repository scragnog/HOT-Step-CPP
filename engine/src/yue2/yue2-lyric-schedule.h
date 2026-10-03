#pragma once
// yue2-lyric-schedule.h: C6, a lyric schedule tied to the score clock.
//
// The caller (server/src/services/backends/yue2/lyricSchedule.ts) sends one
// entry per score section that has a sung lyric block: the section's first
// Vocal note time on the score clock, the block's codepoint span in the
// request lyrics and, optionally, the section's codepoint span in the request
// ABC. While the semantic stage composes frame t, a section whose first note
// lies more than lead_sec after t is hidden from the codec stream: its prompt
// rows get an additive attention bias (soft, e.g. -4) or -inf (a hard mask).
// behind >= 0 also hides sections more than `behind` sections before the
// current one. Lyric blocks the caller could not time are never touched.
//
// The bias only changes which prompt rows the codec queries attend to. The
// prompt's own prefill is causal and unmasked, so later prompt rows still
// carry information about the hidden ones; this is a soft timing control,
// not a guarantee. Frame 0's logits come from the prefill and are unmasked.

#include "yue2-lm-graph.h"   // Yue2MaskSpan
#include "yue2-request.h"    // Yue2LyricSchedule
#include "yue2-tokenizer.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Prompt rows [a, b) of one section, in prefix positions.
struct Yue2ScheduleRows {
    std::vector<std::vector<std::pair<int64_t, int64_t>>> rows;  // per section
};

// Byte offset of codepoint `cp` in UTF-8 `s`; -1 if past the end.
static inline int64_t yue2_cp_to_byte(const std::string & s, int64_t cp) {
    int64_t n = 0;
    size_t  i = 0;
    while (n < cp && i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        i += c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        n++;
    }
    return n == cp && i <= s.size() ? (int64_t) i : -1;
}

// Token rows of `ids` (placed at prefix position `base`) whose bytes overlap
// [b0, b1) of the text they encode. Token bytes come from decoding each id
// alone; byte-level BPE makes those concatenate to the exact text.
static inline bool yue2_rows_for_bytes(const BPETokenizer * tok, const std::vector<int> & ids, int64_t base,
                                       size_t text_bytes, int64_t b0, int64_t b1,
                                       std::pair<int64_t, int64_t> * out, std::string * err) {
    int64_t off = 0, first = -1, last = -1;
    for (size_t i = 0; i < ids.size(); i++) {
        const int64_t len = (int64_t) yue2_bpe_decode(tok, { (int32_t) ids[i] }).size();
        if (off < b1 && off + len > b0) {
            if (first < 0) first = (int64_t) i;
            last = (int64_t) i;
        }
        off += len;
    }
    if (off != (int64_t) text_bytes) {
        if (err) *err = "lyric_schedule: token bytes do not add up to the prompt text";
        return false;
    }
    *out = first < 0 ? std::make_pair(base, base) : std::make_pair(base + first, base + last + 1);
    return true;
}

// Map every section's spans to rows of the semantic prefix built by
// yue2_token_prefixes(tok, style, lyrics, cot, &abc_ids): [EOD] + text + [ABC_START] + abc + ...
static inline bool yue2_schedule_rows(const BPETokenizer * tok, const Yue2LyricSchedule & sc, const std::string & style,
                                      const std::string & lyrics, Yue2Cot cot, const std::string & abc_text,
                                      const std::vector<int> & abc_ids, Yue2ScheduleRows * out, std::string * err) {
    const std::string text      = yue2_assemble_text(style, lyrics, cot);
    const std::vector<int> tids = yue2_bpe_encode(tok, text);
    const int64_t lyric_at      = (int64_t) (text.size() - lyrics.size() - 1);  // lyrics sit before the final "\n"
    const int64_t abc_base      = 1 + (int64_t) tids.size() + 1;                 // after EOD, text, ABC_START
    out->rows.assign(sc.sections.size(), {});
    for (size_t k = 0; k < sc.sections.size(); k++) {
        const Yue2LyricSection & s = sc.sections[k];
        const int64_t l0 = yue2_cp_to_byte(lyrics, s.lyric_c0), l1 = yue2_cp_to_byte(lyrics, s.lyric_c1);
        if (l0 < 0 || l1 < 0 || l1 <= l0) {
            if (err) *err = "lyric_schedule: section " + std::to_string(k) + " lyric span is outside the lyrics";
            return false;
        }
        std::pair<int64_t, int64_t> r;
        if (!yue2_rows_for_bytes(tok, tids, 1, text.size(), lyric_at + l0, lyric_at + l1, &r, err)) return false;
        out->rows[k].push_back(r);
        if (sc.abc && s.abc_c0 >= 0) {
            const int64_t a0 = yue2_cp_to_byte(abc_text, s.abc_c0), a1 = yue2_cp_to_byte(abc_text, s.abc_c1);
            if (a0 < 0 || a1 < 0 || a1 <= a0) {
                if (err) *err = "lyric_schedule: section " + std::to_string(k) + " abc span is outside the score";
                return false;
            }
            if (!yue2_rows_for_bytes(tok, abc_ids, abc_base, abc_text.size(), a0, a1, &r, err)) return false;
            out->rows[k].push_back(r);
        }
    }
    return true;
}

// Rows hidden while composing the frame at t_sec. Sections are in score order.
static inline void yue2_schedule_spans(const Yue2LyricSchedule & sc, const Yue2ScheduleRows & rows, double t_sec,
                                       std::vector<Yue2MaskSpan> * out) {
    out->clear();
    int64_t cur = -1;  // last section whose first note has passed
    for (size_t k = 0; k < sc.sections.size(); k++) {
        if (sc.sections[k].start_sec <= t_sec) cur = (int64_t) k;
    }
    for (size_t k = 0; k < sc.sections.size(); k++) {
        const bool future = sc.sections[k].start_sec - sc.lead_sec > t_sec;
        const bool stale  = sc.behind >= 0 && (int64_t) k < cur - sc.behind;
        if (!future && !stale) continue;
        for (const auto & r : rows.rows[k]) {
            if (r.second > r.first) out->push_back({ r.first, r.second, sc.bias });
        }
    }
}
