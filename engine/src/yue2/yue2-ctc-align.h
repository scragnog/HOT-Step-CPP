#pragma once
// CTC forced alignment + the cursor-word grouping, as torchaudio and
// cursor_prep.py do them. No ggml, no model: this is the half of the MMS_FA
// aligner (docs/plans/yue2/17-mms-fa-port.md §2) that runs on the CPU after the
// acoustic model has produced log-probabilities, and it is gated on its own
// against the twelve `cursor_words.npy` files those exact functions wrote.
//
// Reference, and what each piece reproduces:
//
//   torchaudio.functional.forced_align(log_probs [T, C], targets [N], blank=0)
//     -> per-frame aligned symbol + that symbol's log-prob at that frame.
//     CTC Viterbi over the 2N+1 extended target (blank, t0, blank, t1, ...,
//     blank). Transitions: stay; from s-1; from s-2 only when s is a
//     non-blank that differs from s-2 (the CTC "repeat needs a blank" rule).
//     Ends at the last symbol or the last blank, whichever scores higher.
//
//   torchaudio.functional.merge_tokens(tokens, scores, blank=0)
//     -> runs of the same symbol collapse to one span [start, end) with the
//     MEAN of the per-frame scores; blank spans are dropped. Because the
//     lattice forces a blank between identical consecutive characters, the
//     number of surviving spans equals the number of target characters.
//
//   cursor_prep.py:22-29
//     targets = every word's characters concatenated, NO separators; a word
//     contributes only the characters in the label set; a word with none is
//     skipped (keep[]). Per kept word: start = its first span's start,
//     end = its last span's end, both times frame_index * frame_seconds where
//     frame_seconds = audio_seconds / T; score = mean of its spans' (already
//     exp'd) scores. Char offsets are into the ORIGINAL lyrics string, in
//     codepoints, from `words_of`.
//
// Tie-breaking in the backtrack is the one place two correct implementations
// can legitimately differ by a frame; the acceptance gate is stated in
// seconds for that reason.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct Yue2CtcSpan {
    int    token = 0;   // label index (never blank in the merged output)
    int    start = 0;   // frame, inclusive
    int    end   = 0;   // frame, exclusive
    double score = 0.0; // mean over the span of exp(log-prob) — cursor_prep exp's before merging
};

// log_probs: T rows of C floats, row-major ([T, C]). targets: N label indices,
// none equal to `blank`. Returns false when T < the minimum path length
// (N + number of repeated adjacent targets), which torchaudio raises on.
static bool yue2_ctc_forced_align(const float * log_probs, int64_t T, int C, const std::vector<int> & targets,
                                  int blank, std::vector<int> * path_tok, std::vector<float> * path_lp) {
    const int64_t N = (int64_t) targets.size();
    const int64_t S = 2 * N + 1;
    if (T <= 0 || N < 0) {
        return false;
    }
    auto sym = [&](int64_t s) -> int { return (s % 2 == 0) ? blank : targets[(size_t) (s / 2)]; };

    const double NEG = -1e30;
    std::vector<double>  alpha((size_t) (T * S), NEG);
    std::vector<int8_t>  from((size_t) (T * S), 0);  // 0 = stay, 1 = s-1, 2 = s-2
    auto A = [&](int64_t t, int64_t s) -> double & { return alpha[(size_t) (t * S + s)]; };
    auto F = [&](int64_t t, int64_t s) -> int8_t & { return from[(size_t) (t * S + s)]; };
    auto LP = [&](int64_t t, int s) -> double { return (double) log_probs[(size_t) (t * C + s)]; };

    A(0, 0) = LP(0, blank);
    if (S > 1) {
        A(0, 1) = LP(0, sym(1));
    }
    for (int64_t t = 1; t < T; t++) {
        // Only states reachable both from the start and to the end matter, but
        // a full sweep is cheap at these sizes (T ~ 11k, S ~ 2k) and simpler.
        for (int64_t s = 0; s < S; s++) {
            double best = A(t - 1, s);
            int8_t arg  = 0;
            if (s >= 1 && A(t - 1, s - 1) > best) {
                best = A(t - 1, s - 1);
                arg  = 1;
            }
            if (s >= 2 && sym(s) != blank && sym(s) != sym(s - 2) && A(t - 1, s - 2) > best) {
                best = A(t - 1, s - 2);
                arg  = 2;
            }
            if (best <= NEG / 2) {
                continue;
            }
            A(t, s) = best + LP(t, sym(s));
            F(t, s) = arg;
        }
    }
    int64_t s = S - 1;
    if (S >= 2 && A(T - 1, S - 2) > A(T - 1, S - 1)) {
        s = S - 2;
    }
    if (A(T - 1, s) <= NEG / 2) {
        return false;  // no valid path: T too short for N (+ repeats)
    }
    path_tok->assign((size_t) T, blank);
    path_lp->assign((size_t) T, 0.0f);
    for (int64_t t = T - 1; t >= 0; t--) {
        (*path_tok)[(size_t) t] = sym(s);
        (*path_lp)[(size_t) t]  = (float) LP(t, sym(s));
        if (t > 0) {
            s -= F(t, s);
        }
    }
    return true;
}

// merge_tokens, with cursor_prep's exp() applied to the scores first.
static std::vector<Yue2CtcSpan> yue2_ctc_merge_tokens(const std::vector<int> & tok, const std::vector<float> & lp,
                                                      int blank) {
    std::vector<Yue2CtcSpan> out;
    const int64_t            T = (int64_t) tok.size();
    int64_t                  i = 0;
    while (i < T) {
        int64_t j = i + 1;
        while (j < T && tok[(size_t) j] == tok[(size_t) i]) {
            j++;
        }
        if (tok[(size_t) i] != blank) {
            double s = 0.0;
            for (int64_t k = i; k < j; k++) {
                s += std::exp((double) lp[(size_t) k]);
            }
            out.push_back({ tok[(size_t) i], (int) i, (int) j, s / (double) (j - i) });
        }
        i = j;
    }
    return out;
}

// cursor_prep.py's words_of(): (char0, char1, normalised word) per word, char
// offsets in CODEPOINTS into the lyrics string, skipping [Section] lines.
// Normalisation: lowercase, U+2019 -> ', keep [a-z'], drop if no letter.
struct Yue2LyricWord {
    int64_t     c0 = 0, c1 = 0;  // codepoint span in the original lyrics
    std::string norm;            // [a-z'] only
};

static std::vector<Yue2LyricWord> yue2_cursor_words_of(const std::string & lyrics) {
    // Walk codepoints, tracking line starts and whitespace runs.
    std::vector<Yue2LyricWord> out;
    // Split into lines on '\n' (byte-safe: '\n' never occurs inside a UTF-8 sequence).
    int64_t cp_pos = 0;  // codepoint index of the current line start
    size_t  b      = 0;
    while (b <= lyrics.size()) {
        size_t e = lyrics.find('\n', b);
        if (e == std::string::npos) {
            e = lyrics.size();
        }
        const std::string line = lyrics.substr(b, e - b);
        // is it a [tag] line?  ^\s*\[.*\]\s*$
        {
            size_t l = 0, r = line.size();
            while (l < r && (line[l] == ' ' || line[l] == '\t' || line[l] == '\r')) l++;
            while (r > l && (line[r - 1] == ' ' || line[r - 1] == '\t' || line[r - 1] == '\r')) r--;
            const bool tag = (r > l + 1) && line[l] == '[' && line[r - 1] == ']';
            if (!tag) {
                // words: \S+ runs, with codepoint offsets
                int64_t cp = 0;  // codepoint index within the line
                size_t  i  = 0;
                while (i < line.size()) {
                    // skip whitespace
                    auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; };
                    while (i < line.size() && is_ws((unsigned char) line[i])) { i++; cp++; }
                    if (i >= line.size()) break;
                    const int64_t w0 = cp;
                    std::string   norm;
                    while (i < line.size() && !is_ws((unsigned char) line[i])) {
                        const unsigned char c = (unsigned char) line[i];
                        size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
                        len = std::min(len, line.size() - i);
                        if (len == 1) {
                            char lc = (char) ((c >= 'A' && c <= 'Z') ? c + 32 : c);
                            if ((lc >= 'a' && lc <= 'z') || lc == '\'') norm.push_back(lc);
                        } else if (len == 3 && c == 0xE2 && (unsigned char) line[i + 1] == 0x80 &&
                                   (unsigned char) line[i + 2] == 0x99) {
                            norm.push_back('\'');  // U+2019 -> '
                        }
                        i += len;
                        cp++;
                    }
                    bool has_letter = false;
                    for (char ch : norm) has_letter |= (ch >= 'a' && ch <= 'z');
                    if (has_letter) {
                        out.push_back({ cp_pos + w0, cp_pos + cp, norm });
                    }
                }
            }
        }
        // advance: codepoints in this line + 1 for '\n'
        int64_t n = 0;
        for (unsigned char c : line) if ((c & 0xC0) != 0x80) n++;
        cp_pos += n + 1;
        if (e == lyrics.size()) break;
        b = e + 1;
    }
    return out;
}

// The whole of cursor_prep.py after the model: words -> (start_s, end_s, score,
// char0, char1) rows, five floats per kept word, the file the trainer reads.
// `labels` is the CTC label list in index order (labels[0] == the blank "-").
static bool yue2_cursor_align(const float * log_probs, int64_t T, int C, double audio_seconds,
                              const std::vector<std::string> & labels, const std::string & lyrics,
                              std::vector<float> * rows5, std::string * err) {
    std::vector<int> label_of(256, -1);
    for (size_t i = 1; i < labels.size(); i++) {
        if (labels[i].size() == 1) {
            label_of[(unsigned char) labels[i][0]] = (int) i;
        }
    }
    const std::vector<Yue2LyricWord> ws = yue2_cursor_words_of(lyrics);
    std::vector<std::vector<int>>    toks(ws.size());
    std::vector<size_t>              keep;
    std::vector<int>                 flat;
    for (size_t i = 0; i < ws.size(); i++) {
        for (char ch : ws[i].norm) {
            const int l = label_of[(unsigned char) ch];
            if (l >= 0) toks[i].push_back(l);
        }
        if (!toks[i].empty()) {
            keep.push_back(i);
            flat.insert(flat.end(), toks[i].begin(), toks[i].end());
        }
    }
    if (flat.empty()) {
        *err = "no alignable characters in the lyrics";
        return false;
    }
    std::vector<int>   ptok;
    std::vector<float> plp;
    if (!yue2_ctc_forced_align(log_probs, T, C, flat, /*blank=*/0, &ptok, &plp)) {
        *err = "forced alignment has no valid path (audio too short for the text?)";
        return false;
    }
    const std::vector<Yue2CtcSpan> spans = yue2_ctc_merge_tokens(ptok, plp, 0);
    if (spans.size() != flat.size()) {
        *err = "span count " + std::to_string(spans.size()) + " != target length " + std::to_string(flat.size());
        return false;
    }
    const double fps = audio_seconds / (double) T;  // seconds per frame, cursor_prep's `fps`
    rows5->clear();
    size_t k = 0;
    for (size_t i : keep) {
        const size_t n = toks[i].size();
        double       sc = 0.0;
        for (size_t j = 0; j < n; j++) sc += spans[k + j].score;
        rows5->push_back((float) (spans[k].start * fps));
        rows5->push_back((float) (spans[k + n - 1].end * fps));
        rows5->push_back((float) (sc / (double) n));
        rows5->push_back((float) ws[i].c0);
        rows5->push_back((float) ws[i].c1);
        k += n;
    }
    return true;
}
