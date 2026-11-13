#pragma once
// minimax/mm3-align.h — lyric→frame alignment for MiniMax-Music3 (LRC output)
//
// HOT-Step file (does not exist upstream). Included by minimax/mm3-ar-loop.h.
//
// ── Why this is not the ACE mechanism ───────────────────────────────────────
//
// ACE reads alignment out of its DiT's CROSS-attention between lyric-encoder
// tokens and audio frames. MM3 has no analogue: its flow DiT has no cross
// attention and never sees lyrics at all — it is conditioned by channel
// concatenation from the condition encoder, which only ever sees LM frame
// hiddens. The lyrics exist in exactly one place, the LM prompt, so the only
// lyric↔audio contact surface is the LM's own decode attention.
//
// That turns out to be enough. Every AR decode step attends over the whole
// prompt, so taking the lyric columns of each step's attention builds a
// [head][token][frame] matrix directly — the exact shape lrc_align() consumes,
// and at 25 fps frame f is 40 ms with no latent-rate conversion (cleaner than
// ACE, which aligns against latent frames).
//
// ── The heads, and how they were chosen ─────────────────────────────────────
//
// Screened over three clips of different genre, seed and lyrics (MM3_ALIGN_DUMP,
// see the mm3-backend skill). Two properties disqualify a head:
//
//   * it never moves. 55 % of all 1152 heads sit permanently on one structural
//     token, and a naive "is it monotonic" test rates those PERFECT (the peak
//     never goes backwards because it never goes anywhere). Heads are therefore
//     ranked on TRAVEL (fraction of frames off the modal token) times rank
//     correlation with time.
//   * it only works on one clip. The best head on a single clip did not survive
//     the other two, so heads are ranked by their WORST clip, never the mean.
//
// Three heads clear that bar. They are few, but lrc_align()'s bidirectional
// consensus denoising is built to extract a path from exactly this kind of
// noisy, partial evidence.
//
// ── The lead offset ─────────────────────────────────────────────────────────
//
// Measured errors skew consistently NEGATIVE: attention reaches a lyric token
// ~0.6–0.8 s before that word is audible. That is expected rather than a bug —
// the LM attends to a token as it begins GENERATING the content, and the audio
// for those codes lands later. It is a systematic offset, so it is corrected
// as one instead of being left as apparent error.

#include "../lrc-alignment.h"
#include "mm3-model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// (layer, heads) pairs whose decode attention tracks the lyric. Same shape as
// ACE's AlignmentConfig, kept local because it indexes the LM's SELF-attention
// rather than a DiT cross-attention — a different tensor with different
// semantics that happens to have the same use.
struct MM3AlignHead {
    int layer;
    int head;
};

static const MM3AlignHead MM3_ALIGN_HEADS[] = {
    { 12, 27 },
    { 19,  7 },
    { 24, 29 },
};
static const int MM3_ALIGN_N_HEADS = (int) (sizeof(MM3_ALIGN_HEADS) / sizeof(MM3_ALIGN_HEADS[0]));

// Seconds to ADD to every emitted timestamp, correcting the LM's generation
// lead (see header). Measured 0.6–0.8 s across clips; 0.7 is the midpoint.
#define MM3_ALIGN_LEAD_SEC 0.70f

// Does this layer need its attention scores materialised? Only these layers pay
// the manual (non-flash) attention cost — 3 of 36, so roughly a twelfth of the
// penalty a graph-wide switch would cost.
static inline bool mm3_align_layer_needed(int layer) {
    for (int i = 0; i < MM3_ALIGN_N_HEADS; i++) {
        if (MM3_ALIGN_HEADS[i].layer == layer) {
            return true;
        }
    }
    return false;
}

// Deepest layer any alignment head reads. The LRC replay pass (mm3-lm-graph.h)
// runs blocks 0..this and stops — layers above it feed only the logits, which
// the replay never needs.
static inline int mm3_align_max_layer(void) {
    int mx = 0;
    for (int i = 0; i < MM3_ALIGN_N_HEADS; i++) {
        if (MM3_ALIGN_HEADS[i].layer > mx) {
            mx = MM3_ALIGN_HEADS[i].layer;
        }
    }
    return mx;
}

// Incremental detokenisation of the lyric span, matching what lrc_align()
// expects: token_texts[i] is the text token i ADDS to the running string, so
// concatenating them reproduces the lyric. Mirrors the ACE implementation in
// pipeline-synth-ops.cpp (a BPE token is byte-encoded, so each codepoint maps
// back through byte2str to one raw byte).
static void mm3_align_token_texts(const BPETokenizer & bpe, const std::vector<int> & ids,
                                  std::vector<std::string> * out) {
    out->assign(ids.size(), std::string());
    std::string prev_full;
    std::string full;
    for (size_t i = 0; i < ids.size(); i++) {
        const int pid = ids[i];
        if (pid >= 0 && pid < bpe.n_vocab) {
            const std::string & s = bpe.id_to_str[(size_t) pid];
            for (size_t ci = 0; ci < s.size();) {
                int       adv = 0;
                const int cp  = utf8_codepoint(s.c_str() + ci, &adv);
                bool      hit = false;
                for (int by = 0; by < 256; by++) {
                    int       a2  = 0;
                    const int cp2 = utf8_codepoint(bpe.byte2str[by].c_str(), &a2);
                    if (cp2 == cp && (int) bpe.byte2str[by].size() == adv) {
                        full += (char) (unsigned char) by;
                        hit = true;
                        break;
                    }
                }
                if (!hit) {
                    full += '?';
                }
                ci += (size_t) (adv > 0 ? adv : 1);
            }
        }
        (*out)[i] = full.size() > prev_full.size() ? full.substr(prev_full.size()) : std::string();
        prev_full = full;
    }
}

// Shift every [mm:ss.cc] stamp in an LRC body by `delta` seconds, clamped at 0.
// Applied after alignment rather than to the frame indices so the correction
// stays visible and adjustable in one place.
static std::string mm3_align_shift_lrc(const std::string & lrc, float delta) {
    if (delta == 0.0f) {
        return lrc;
    }
    std::string out;
    out.reserve(lrc.size());
    for (size_t i = 0; i < lrc.size();) {
        // A stamp is exactly "[m+:ss.cc]" at the start of a line.
        if (lrc[i] == '[' && (i == 0 || lrc[i - 1] == '\n')) {
            const size_t close = lrc.find(']', i);
            if (close != std::string::npos && close - i >= 7) {
                const std::string body  = lrc.substr(i + 1, close - i - 1);
                const size_t      colon = body.find(':');
                const size_t      dot   = body.find('.');
                if (colon != std::string::npos && dot != std::string::npos && dot > colon) {
                    const int mm = atoi(body.substr(0, colon).c_str());
                    const int ss = atoi(body.substr(colon + 1, dot - colon - 1).c_str());
                    const int cc = atoi(body.substr(dot + 1).c_str());
                    float     t  = (float) (mm * 60 + ss) + (float) cc / 100.0f + delta;
                    if (t < 0.0f) {
                        t = 0.0f;
                    }
                    const int total_cc = (int) (t * 100.0f + 0.5f);
                    char      stamp[32];
                    snprintf(stamp, sizeof(stamp), "[%02d:%02d.%02d]", total_cc / 6000,
                             (total_cc / 100) % 60, total_cc % 100);
                    out += stamp;
                    i = close + 1;
                    continue;
                }
            }
        }
        out += lrc[i++];
    }
    return out;
}

// ── Direct alignment (the path that is actually used) ───────────────────────
//
// lrc_align() was tried first and produced timings compressed into the first
// half of the song — every line early and bunched — while the same captured
// attention run through a plain monotonic DTW lands within ~1 s of Whisper.
// Sweeping its violence_level from 2.0 to 0.0 changed nothing, so the
// difference is not its consensus thresholding. Rather than keep guessing at
// the internals of an algorithm shaped around ACE's 7 dedicated cross-attention
// heads, MM3 does the small, legible thing: average the heads, DTW, group by
// line. lrc_align() stays untouched for ACE.

// Monotonic DTW over cost[token][frame]. Returns, per frame, the token index —
// each step either stays on the current token or advances exactly one, which is
// what makes the result a legal reading order rather than a best-match-per-frame
// scatter.
static void mm3_align_dtw(const std::vector<float> & cost, int n_tok, int n_fr, std::vector<int> * path) {
    const float INF = 1e30f;
    std::vector<float>   acc((size_t) n_tok * n_fr, INF);
    std::vector<uint8_t> back((size_t) n_tok * n_fr, 0);

    acc[0] = cost[0];
    for (int f = 1; f < n_fr; f++) {
        acc[(size_t) f] = acc[(size_t) f - 1] + cost[(size_t) f];
    }
    for (int t = 1; t < n_tok; t++) {
        for (int f = 1; f < n_fr; f++) {
            const size_t i    = (size_t) t * n_fr + f;
            const float  stay = acc[i - 1];
            const float  adv  = acc[i - (size_t) n_fr - 1];
            if (stay <= adv) {
                acc[i] = stay + cost[i];
            } else {
                acc[i]  = adv + cost[i];
                back[i] = 1;
            }
        }
    }
    path->assign((size_t) n_fr, 0);
    int t = n_tok - 1, f = n_fr - 1;
    while (f > 0) {
        (*path)[(size_t) f] = t;
        if (back[(size_t) t * n_fr + f] && t > 0) {
            t--;
        }
        f--;
    }
    (*path)[0] = t;
}

// Is this a PROMPT-MACHINERY tag rather than a song section?
//
// _normalize_lyrics prepends "[start]" to every lyric body, and substitutes the
// literal "[instrumental]" when the caller sends no lyrics (mm3-request.h). Those
// are prompt scaffolding, not structure a listener can hear, so they are the only
// bracket lines dropped from the LRC. Real song tags ("[verse 1]", "[chorus]",
// "[drop]") are kept: they sit in the aligned span like any other token, and the
// UI draws them as the marker row above the waveform exactly as it does for ACE.
//
// Tags arrive lowercased (mm3_lower_tags), so no case folding is needed.
static bool mm3_align_is_sentinel_tag(const std::string & body) {
    return body == "[start]" || body == "[end]" || body == "[instrumental]";
}

static std::string mm3_align_stamp(float sec) {
    if (sec < 0.0f) {
        sec = 0.0f;
    }
    const int cc = (int) (sec * 100.0f + 0.5f);
    char      buf[32];
    snprintf(buf, sizeof(buf), "[%02d:%02d.%02d]", cc / 6000, (cc / 100) % 60, cc % 100);
    return buf;
}

// Build the LRC from captured attention.
//
//   scores    [head][token][frame], heads in MM3_ALIGN_HEADS order
//   ids       the lyric token ids (the same span the scores cover)
//   duration  rendered audio length in seconds
//
// Returns an empty string on any failure — LRC is a garnish, and a run must
// never fail because alignment did.
static std::string mm3_align_build_lrc(const std::vector<float> & scores, const std::vector<int> & ids,
                                       const BPETokenizer & bpe, int n_frames, float duration) {
    const int n_tokens = (int) ids.size();
    if (n_tokens <= 1 || n_frames <= 1 || scores.empty()) {
        return std::string();
    }
    if (scores.size() != (size_t) MM3_ALIGN_N_HEADS * (size_t) n_tokens * (size_t) n_frames) {
        fprintf(stderr, "[MM3-LRC] score buffer is %zu, expected %d*%d*%d — skipping\n", scores.size(),
                MM3_ALIGN_N_HEADS, n_tokens, n_frames);
        return std::string();
    }

    std::vector<std::string> texts;
    mm3_align_token_texts(bpe, ids, &texts);

    // Consensus: normalise each head over tokens (so a confident head does not
    // simply outvote the others), then average.
    std::vector<float> avg((size_t) n_tokens * n_frames, 0.0f);
    for (int h = 0; h < MM3_ALIGN_N_HEADS; h++) {
        const float * src = scores.data() + (size_t) h * n_tokens * n_frames;
        for (int f = 0; f < n_frames; f++) {
            float sum = 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                sum += src[(size_t) t * n_frames + f];
            }
            const float inv = sum > 1e-9f ? 1.0f / sum : 0.0f;
            for (int t = 0; t < n_tokens; t++) {
                avg[(size_t) t * n_frames + f] += src[(size_t) t * n_frames + f] * inv;
            }
        }
    }

    std::vector<float> cost((size_t) n_tokens * n_frames);
    for (size_t i = 0; i < cost.size(); i++) {
        cost[i] = -logf(avg[i] / (float) MM3_ALIGN_N_HEADS + 1e-9f);
    }

    std::vector<int> path;
    mm3_align_dtw(cost, n_tokens, n_frames, &path);

    // Onset frame per token = the first frame the path reaches it.
    std::vector<int> onset((size_t) n_tokens, -1);
    for (int f = 0; f < n_frames; f++) {
        const int t = path[(size_t) f];
        if (t >= 0 && t < n_tokens && onset[(size_t) t] < 0) {
            onset[(size_t) t] = f;
        }
    }

    // Group tokens into lines on the newline the tokenizer emits, and stamp each
    // line with its first token's onset (plus the generation-lead correction).
    const float spf = n_frames > 0 ? duration / (float) n_frames : 0.0f;
    std::string out;
    std::string line;
    int         line_first = 0;
    auto        flush      = [&](void) {
        // Trim; skip blank lines and the prompt-machinery tags. Section tags
        // ("[verse 1]", "[chorus]") ARE emitted -- they align like any other
        // token and become the marker row above the waveform.
        size_t a = line.find_first_not_of(" \t\r\n");
        size_t b = line.find_last_not_of(" \t\r\n");
        if (a != std::string::npos && b != std::string::npos) {
            const std::string body = line.substr(a, b - a + 1);
            if (!body.empty() && !mm3_align_is_sentinel_tag(body)) {
                int fr = onset[(size_t) line_first];
                if (fr < 0) {
                    fr = 0;
                }
                out += mm3_align_stamp((float) fr * spf + MM3_ALIGN_LEAD_SEC) + body + "\n";
            }
        }
        line.clear();
    };

    for (int t = 0; t < n_tokens; t++) {
        const std::string & tx = texts[(size_t) t];
        for (size_t ci = 0; ci < tx.size(); ci++) {
            if (tx[ci] == '\n') {
                flush();
                line_first = t + 1 < n_tokens ? t + 1 : t;
            } else {
                if (line.empty()) {
                    line_first = t;
                }
                line += tx[ci];
            }
        }
    }
    flush();

    if (out.empty()) {
        fprintf(stderr, "[MM3-LRC] no lines produced\n");
    }
    return out;
}

// ── Base handoff cue (MM3ArOptions::handoff_cue) ─────────────────────────────
//
// Where a capped adapter take is cut before the base model plays it out. The
// rule was written before any data and not tuned afterwards: align the take in
// full and again as if it had stopped 50 s before the cap; a sheet line is HELD
// when the two alignments agree within 2 s and the cut one places it at least
// 2 s before its own end. Cue = the last held line + 25 s, none past 20 s before
// the cap. At the app's 300 s cap that is "cut at 250 s, held by 248 s, no cue
// past 280 s".
//
// The cut alignment needs no second pass. Decode attention is causal, so the
// first C columns of the full replay are what a replay of the first C frames
// would capture; only the DTW, which must end on the last token at the last
// frame, sees the shorter matrix differently. Timestamps are parsed back out of
// the LRC text, so the cue carries the same 0.01 s rounding a script reading
// the LRC gets.
struct MM3LrcLine {
    double t;
};

static std::vector<MM3LrcLine> mm3_align_sheet_lines(const std::string & lrc) {
    std::vector<MM3LrcLine> rows;
    size_t                  i = 0;
    while (i < lrc.size()) {
        size_t e = lrc.find('\n', i);
        if (e == std::string::npos) {
            e = lrc.size();
        }
        const std::string l = lrc.substr(i, e - i);
        i                   = e + 1;
        const size_t close  = l.find(']');
        const size_t colon  = l.find(':');
        if (l.empty() || l[0] != '[' || close == std::string::npos || colon == std::string::npos || colon > close) {
            continue;
        }
        std::string text = l.substr(close + 1);
        const size_t a = text.find_first_not_of(" \t\r");
        const size_t b = text.find_last_not_of(" \t\r");
        text = a == std::string::npos ? std::string() : text.substr(a, b - a + 1);
        // Section tags ("[chorus]") and blank lines are not sheet lines.
        if (text.empty() || (text.front() == '[' && text.back() == ']')) {
            continue;
        }
        rows.push_back({ strtod(l.substr(1, colon - 1).c_str(), nullptr) * 60.0 +
                         strtod(l.substr(colon + 1, close - colon - 1).c_str(), nullptr) });
    }
    return rows;
}

// The rule on two LRCs. Returns the cut frame (cue at 25 fps), or -1 = no cue.
static int64_t mm3_align_handoff_rule(const std::string & full_lrc, const std::string & cut_lrc, double cut_sec,
                                      double cap_sec, double * cue_sec) {
    const std::vector<MM3LrcLine> full = mm3_align_sheet_lines(full_lrc);
    const std::vector<MM3LrcLine> cut  = mm3_align_sheet_lines(cut_lrc);
    int                           L    = -1;
    for (size_t i = 0; i < std::min(full.size(), cut.size()); i++) {
        if (std::fabs(full[i].t - cut[i].t) <= 2.0 && cut[i].t <= cut_sec - 2.0) {
            L = (int) i;
        }
    }
    if (L < 0) {
        return -1;
    }
    const double c = full[(size_t) L].t + 25.0;
    if (c > cap_sec - 20.0) {
        return -1;
    }
    if (cue_sec) {
        *cue_sec = c;
    }
    return (int64_t) std::floor(c * 25.0 + 0.5);
}

// From a full replay's [head][token][frame] scores (n_fr columns, n_fr frames
// at 25 fps) of a take that ran to `cap_frames`. Writes the full LRC too.
static int64_t mm3_align_handoff_cue(const std::vector<float> & scores, const std::vector<int> & ids,
                                     const BPETokenizer & bpe, int n_fr, int64_t cap_frames, std::string * full_lrc,
                                     double * cue_sec) {
    const int64_t C = cap_frames - 50 * 25;
    const int     T = (int) ids.size();
    if (C <= 1 || n_fr <= C || scores.size() != (size_t) MM3_ALIGN_N_HEADS * T * n_fr) {
        return -1;
    }
    *full_lrc = mm3_align_build_lrc(scores, ids, bpe, n_fr, (float) n_fr / 25.0f);
    std::vector<float> head((size_t) MM3_ALIGN_N_HEADS * T * C);
    for (size_t r = 0; r < (size_t) MM3_ALIGN_N_HEADS * T; r++) {
        memcpy(head.data() + r * C, scores.data() + r * n_fr, (size_t) C * sizeof(float));
    }
    const std::string cut_lrc = mm3_align_build_lrc(head, ids, bpe, (int) C, (float) C / 25.0f);
    return mm3_align_handoff_rule(*full_lrc, cut_lrc, (double) C / 25.0, (double) cap_frames / 25.0, cue_sec);
}
