#pragma once
// minimax/mm3-request.h — MM3 synth request parsing + prompt assembly.
//
// HOT-Step file (does not exist upstream). The seam between "what a caller
// asked for" (caption, lyrics, duration, seed) and "the exact byte string the
// checkpoint was trained on". Everything here is text work — no GGML, no VRAM,
// no GPU — so it is cheap and safe to call on a cold server. That is what makes
// POST /mm3/tokenize-check a useful pre-flight for the UI.
//
// ── The template (checkpoint contract) ──────────────────────────────────────
//
// diffusers @ dafe3733, modular_pipelines/minimax_music3/encoders.py:218-221:
//
//     text = (
//         f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
//         f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
//     )
//
// which expands to, verbatim (verified byte-for-byte against the fixture
// D:\Ace-Step-Latest\mm3-weights\fixtures\tok_prompt_template.txt):
//
//     <|im_start|><|caption_start|>CAPTION<|caption_end|><|lyrics_start|>[start]
//     LYRICS<|lyrics_end|><|im_end|><|audio_start|>
//
// No spaces, no newlines anywhere except the ones _normalize_lyrics itself
// emits. `[start]\n` is produced by _normalize_lyrics, NOT by the template, and
// the lyric body carries no trailing newline before <|lyrics_end|>.
// encoders.py:34-35 says it outright: "even whitespace-level changes to the
// assembled prompt change the generated audio."
//
// ── Text hygiene, replicated ────────────────────────────────────────────────
//
// _clean_caption (encoders.py:56-79) — the caption goes in as an opaque string,
// but the reference first strips the markdown an LLM-written caption tends to
// carry:
//   1. `<|anything|>` -> "first is rest" (or the bare inner text if it is one
//      word). This is a defusing rule: a caption containing a literal special
//      token would otherwise tokenise as a control token and corrupt the frame
//      structure.
//   2. per line: strip a leading ATX heading (`^\s{0,3}#{1,6}\s+`), a leading
//      bullet (`^\s*[*+-]\s+`, then `^\s*\*\s+` again), unwrap `**bold**`
//      repeatedly until it stops changing, unwrap `*italic*` (with a
//      not-preceded-by-`*` / not-followed-by-`*` guard), then rstrip.
//   3. blank out horizontal rules (`^\s*[-*_]{3,}\s*$`, multiline).
//   4. delete every "• " and every run of exactly four spaces.
//   5. collapse runs of 2+ newlines to one.
//
// _normalize_lyrics (encoders.py:82-93) — the lyric side is a real reformat:
//   1. per line: if the line STARTS with one or more `[tag]` (optionally
//      separated by spaces/tabs), the line is replaced by just those tags —
//      any text sharing the line with a leading tag is DROPPED. Lines with no
//      leading tag pass through untouched.
//   2. "] " -> "]\n" and " [" -> "\n[" — every tag ends up alone on its line.
//   3. " ^ " -> "\n" — the checkpoint's inline line-break marker.
//   4. lowercase the inside of every `[...]`.
//   5. prepend "[start]\n".
//
// The two `re.sub` calls that use lookbehind/lookahead (`*italic*`) and the
// MULTILINE horizontal rule are hand-rolled here: std::regex's ECMAScript
// grammar has no lookbehind, and Python's `\s` spanning newlines under
// re.MULTILINE makes the rule pattern subtly non-line-local. Both hand-rolled
// forms are documented at their definition with the equivalence argument.
//
// ── Where we deliberately differ ────────────────────────────────────────────
//
// The reference REQUIRES non-empty lyrics (encoders.py:210-211,
// `check_inputs`); it has no instrumental path at all. HOT-Step accepts an
// empty `lyrics` and substitutes the literal `[instrumental]` as the lyric
// body, giving `[start]\n[instrumental]`. `[Instrumental]` is a documented
// section tag in MiniMax's own skill README, so this stays inside the trained
// vocabulary rather than inventing one — but it IS a HOT-Step decision, not a
// reference behaviour. The caption is expected to say the piece is instrumental
// too (see .claude/skills/mm3-captioning/SKILL.md).

#include "mm3-lm-adapter.h"
#include "mm3-model.h"
#include "mm3-pipeline.h"
#include "mm3-tokenizer.h"

#include "yyjson.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// encoders.py:44-45. Both are checkpoint limits, not our policy.
#define MM3_MAX_PROMPT_TOKENS 5000
#define MM3_MAX_AUDIO_FRAMES  9000

// The five template literals, encoders.py:36-39.
#define MM3_TPL_IM_START      "<|im_start|>"
#define MM3_TPL_IM_END        "<|im_end|>"
#define MM3_TPL_CAPTION_START "<|caption_start|>"
#define MM3_TPL_CAPTION_END   "<|caption_end|>"
#define MM3_TPL_LYRICS_START  "<|lyrics_start|>"
#define MM3_TPL_LYRICS_END    "<|lyrics_end|>"
#define MM3_TPL_AUDIO_START   "<|audio_start|>"

// What we put in the lyric slot when the caller sends none. See the header
// note — this is HOT-Step policy, the reference simply rejects empty lyrics.
#define MM3_INSTRUMENTAL_LYRIC "[instrumental]"

// ── Small text primitives ───────────────────────────────────────────────────

// Python's `\s` inside a single line. `\n` is excluded because every caller
// below has already split on newlines; keeping it out is what makes the
// hand-rolled rules line-local.
static inline bool mm3_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

static std::string mm3_rstrip(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(0, e);
}

static std::string mm3_strip(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (mm3_is_space(s[b]) || s[b] == '\n')) {
        b++;
    }
    while (e > b && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(b, e - b);
}

static bool mm3_str_blank(const std::string & s) {
    for (char c : s) {
        if (!mm3_is_space(c) && c != '\n') {
            return false;
        }
    }
    return true;
}

// Python `str.split("\n")` — a trailing newline DOES yield an empty last field.
// The reference uses this form for the lyrics (encoders.py:85).
static std::vector<std::string> mm3_split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == '\n') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// Python `str.splitlines()` — the reference uses THIS form for the caption
// (encoders.py:65), and the difference from split("\n") is load-bearing: a
// caption ending in a newline must not gain a trailing empty line, because the
// join below would then leave a trailing '\n' that survives the \n{2,} collapse
// and lands inside <|caption_start|>...<|caption_end|>. Verified against the
// reference: modelling this as split("\n") produced exactly that one-byte
// divergence on every caption with a trailing newline.
//
// Boundaries handled: "\n", "\r\n", "\r". Python also splits on \v, \f, \x1c-\x1e,
// \x85 and U+2028/9; those never appear in a caption and are left alone.
static std::vector<std::string> mm3_splitlines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    size_t                   i     = 0;
    while (i < s.size()) {
        if (s[i] == '\n' || s[i] == '\r') {
            out.push_back(s.substr(start, i - start));
            if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') {
                i++;
            }
            start = i + 1;
        }
        i++;
    }
    if (start < s.size()) {
        out.push_back(s.substr(start));
    }
    return out;
}

static std::string mm3_join_lines(const std::vector<std::string> & v) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            out += '\n';
        }
        out += v[i];
    }
    return out;
}

static void mm3_replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0) {
            out += to;
            i += from.size();
        } else {
            out += s[i];
            i++;
        }
    }
    s.swap(out);
}

// ── _clean_caption ──────────────────────────────────────────────────────────

// `_SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)` (encoders.py:52,57-62).
// Pattern `<\|([^|]*)\|>`: the inner run may not contain '|', so the scan is a
// literal "<|" then everything up to the first '|', which must be followed by
// '>'. Replacement: inner.strip(); split on the first whitespace RUN; two
// fields -> "a is b", otherwise the stripped inner verbatim.
static std::string mm3_rewrite_special_tags(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in[i] == '<' && i + 1 < in.size() && in[i + 1] == '|') {
            size_t j = i + 2;
            while (j < in.size() && in[j] != '|') {
                j++;
            }
            if (j + 1 < in.size() && in[j] == '|' && in[j + 1] == '>') {
                const std::string inner = mm3_strip(in.substr(i + 2, j - (i + 2)));
                size_t            k     = 0;
                while (k < inner.size() && !mm3_is_space(inner[k]) && inner[k] != '\n') {
                    k++;
                }
                if (k < inner.size()) {
                    size_t r = k;
                    while (r < inner.size() && (mm3_is_space(inner[r]) || inner[r] == '\n')) {
                        r++;
                    }
                    if (r < inner.size()) {
                        out += inner.substr(0, k) + " is " + inner.substr(r);
                    } else {
                        // Trailing whitespace only: str.split(None, 1) yields ONE
                        // field, so the "a is b" branch does not fire.
                        out += inner.substr(0, k);
                    }
                } else {
                    out += inner;
                }
                i = j + 2;
                continue;
            }
        }
        out += in[i];
        i++;
    }
    return out;
}

// `^\s{0,3}#{1,6}\s+` -> "" (encoders.py:66).
static void mm3_strip_heading(std::string & line) {
    size_t i = 0;
    while (i < line.size() && i < 3 && mm3_is_space(line[i])) {
        i++;
    }
    size_t h = 0;
    while (i + h < line.size() && h < 6 && line[i + h] == '#') {
        h++;
    }
    if (h == 0) {
        return;
    }
    size_t j = i + h;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;  // `\s+` needs at least one
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

// `^\s*[*+-]\s+` -> "" (encoders.py:67), and the redundant `^\s*\*\s+` at
// encoders.py:68 (a strict subset — replicated anyway so a future divergence
// upstream is a one-line diff here).
static void mm3_strip_bullet(std::string & line, const char * marks) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    if (i >= line.size() || !strchr(marks, line[i])) {
        return;
    }
    size_t j = i + 1;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

// One pass of `re.sub(r"\*\*([^*]+)\*\*", r"\1", line)` — leftmost,
// non-overlapping, inner run non-empty and '*'-free.
static std::string mm3_unwrap_bold_once(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') {
            size_t j = i + 2;
            while (j < s.size() && s[j] != '*') {
                j++;
            }
            if (j > i + 2 && j + 1 < s.size() && s[j] == '*' && s[j + 1] == '*') {
                out += s.substr(i + 2, j - (i + 2));
                i = j + 2;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// `re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)` (encoders.py:74),
// hand-rolled because ECMAScript std::regex has no lookbehind.
//
// Equivalence argument: `[^*\n]+` can never cross a '*', so once the greedy run
// stops the only candidate closer is the character it stopped on. Shrinking the
// run therefore cannot produce a different match, and neither guard can be
// satisfied by backtracking — a failed attempt at position i is a hard failure
// at position i, exactly as this loop treats it. The lookbehind reads the
// ORIGINAL string (Python's does too, even across a previous substitution), so
// the check is against `s`, not against the output built so far.
static std::string mm3_unwrap_italic(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && (i == 0 || s[i - 1] != '*')) {
            size_t j = i + 1;
            while (j < s.size() && s[j] != '*' && s[j] != '\n') {
                j++;
            }
            if (j > i + 1 && j < s.size() && s[j] == '*' && (j + 1 >= s.size() || s[j + 1] != '*')) {
                out += s.substr(i + 1, j - (i + 1));
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// `^\s*[-*_]{3,}\s*$` (encoders.py:77).
static bool mm3_is_hrule(const std::string & line) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    size_t n = 0;
    while (i + n < line.size() && (line[i + n] == '-' || line[i + n] == '*' || line[i + n] == '_')) {
        n++;
    }
    if (n < 3) {
        return false;
    }
    size_t j = i + n;
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    return j == line.size();
}

// encoders.py:56-79, in order.
static std::string mm3_clean_caption(const std::string & caption) {
    std::string text = mm3_rewrite_special_tags(caption);

    std::vector<std::string> lines_out;
    for (std::string line : mm3_splitlines(text)) {
        mm3_strip_heading(line);
        mm3_strip_bullet(line, "*+-");
        mm3_strip_bullet(line, "*");
        while (line.find("**") != std::string::npos) {
            const std::string updated = mm3_unwrap_bold_once(line);
            if (updated == line) {
                break;
            }
            line = updated;
        }
        line = mm3_unwrap_italic(line);
        lines_out.push_back(mm3_rstrip(line));
    }

    // The horizontal-rule sub is MULTILINE with a `\s` that also matches '\n',
    // so in Python it can swallow the newline(s) BEFORE a rule line. It can
    // never swallow the one after (`$` must land at a line end), and any extra
    // blank line it would have eaten is removed by the `\n{2,}` collapse three
    // lines below regardless. Blanking the line in place is therefore
    // equivalent on the collapsed output.
    for (auto & line : lines_out) {
        if (mm3_is_hrule(line)) {
            line.clear();
        }
    }
    text = mm3_join_lines(lines_out);

    mm3_replace_all(text, "\xE2\x80\xA2 ", "");  // "• " (U+2022 + space)
    mm3_replace_all(text, "    ", "");           // exactly four spaces

    // `re.sub(r"\n{2,}", "\n", text)`
    std::string collapsed;
    collapsed.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\n') {
            size_t j = i;
            while (j < text.size() && text[j] == '\n') {
                j++;
            }
            collapsed += '\n';
            i = j;
        } else {
            collapsed += text[i];
            i++;
        }
    }
    return collapsed;
}

// ── _normalize_lyrics ───────────────────────────────────────────────────────

// `_LEADING_TAGS_RE = r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)"` (encoders.py:53).
// Returns true and writes group(1).strip() when the line starts with one or
// more bracketed tags; false leaves the line alone.
static bool mm3_leading_tags(const std::string & line, std::string * out) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        i++;
    }
    const size_t first = i;
    size_t       last  = i;  // end of the last consumed `[tag]` + trailing [ \t]*
    int          n     = 0;
    while (i < line.size() && line[i] == '[') {
        size_t j = i + 1;
        while (j < line.size() && line[j] != ']') {
            j++;
        }
        if (j >= line.size() || j == i + 1) {
            break;  // unterminated, or `[]` (the `[^\]]+` is one-or-more)
        }
        i = j + 1;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            i++;
        }
        last = i;
        n++;
    }
    if (n == 0) {
        return false;
    }
    *out = mm3_strip(line.substr(first, last - first));
    return true;
}

// `re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)`
// (encoders.py:92). ASCII-only lowering: every tag in MiniMax's documented
// vocabulary is ASCII, and Python's Unicode casefold on a non-ASCII tag would
// be a behaviour we cannot replicate without ICU. Non-ASCII bytes pass through.
static std::string mm3_lower_tags(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '[') {
            size_t j = i + 1;
            while (j < s.size() && s[j] != ']') {
                j++;
            }
            if (j < s.size() && j > i + 1) {
                out += '[';
                for (size_t k = i + 1; k < j; k++) {
                    out += (char) std::tolower((unsigned char) s[k]);
                }
                out += ']';
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

// encoders.py:82-93, in order. Always returns a string starting "[start]\n".
static std::string mm3_normalize_lyrics(const std::string & lyrics) {
    std::vector<std::string> out;
    for (const std::string & line : mm3_split_lines(lyrics)) {
        std::string tags;
        out.push_back(mm3_leading_tags(line, &tags) ? tags : line);
    }
    std::string text = mm3_join_lines(out);
    mm3_replace_all(text, "] ", "]\n");
    mm3_replace_all(text, " [", "\n[");
    mm3_replace_all(text, " ^ ", "\n");
    text = mm3_lower_tags(text);
    return "[start]\n" + text;
}

// ── Assembly ────────────────────────────────────────────────────────────────

// encoders.py:218-221. `lyrics` empty/whitespace-only -> the instrumental
// substitution documented at the top of this file.
static std::string mm3_assemble_prompt(const std::string & caption, const std::string & lyrics,
                                       bool * out_instrumental = nullptr) {
    const bool instrumental = mm3_str_blank(lyrics);
    if (out_instrumental) {
        *out_instrumental = instrumental;
    }
    return std::string(MM3_TPL_IM_START) + MM3_TPL_CAPTION_START + mm3_clean_caption(caption) + MM3_TPL_CAPTION_END +
           MM3_TPL_LYRICS_START + mm3_normalize_lyrics(instrumental ? std::string(MM3_INSTRUMENTAL_LYRIC) : lyrics) +
           MM3_TPL_LYRICS_END + MM3_TPL_IM_END + MM3_TPL_AUDIO_START;
}

// ── The wire request ────────────────────────────────────────────────────────

// Everything POST /mm3/synth accepts, resolved against the checkpoint's own
// defaults. `gen` is ready to hand to mm3_generate() once a cancel hook is
// attached.
struct MM3SynthRequest {
    std::string caption;
    std::string lyrics;
    bool        instrumental = false;

    double  duration   = 0.0;  // as asked, seconds
    int64_t max_frames = 0;    // = min(round(duration * frame_rate), 9000)
    int64_t seed_in    = -1;   // as asked (-1 = random)
    int64_t ar_seed_in = -1;   // as asked (-1 = tied to the resolved seed)
    int     wav_bits   = 16;
    // Emit LRC lyric timestamps from the LM's alignment heads (mm3-align.h).
    // Ignored for instrumentals — there is nothing to align.
    bool    want_lrc   = false;

    // ── MM3 Plank ──────────────────────────────────────────────────────────
    // Capture the AR stage's output codes so a later render can replay them.
    // Zero cost when false (the default): the codes already exist in memory,
    // this only decides whether they are serialised onto the job.
    bool    get_ar_codes = false;

    // ── AR cache (mm3-job.h) ───────────────────────────────────────────────
    // Reuse the previous run's frame-hidden block when every AR-affecting
    // input is byte-identical, and skip stage 1 outright. UNLIKE the plank
    // above this IS a speedup — a large one (AR is roughly half the render) —
    // because the condition encoder's actual input is the hidden block, not
    // the codes. Costs one block of host RAM (128 KB per frame; ~600 MB for a
    // 200 s song), which is why it is opt-in rather than always on.
    bool    reuse_ar = false;

    // ── AR cache, on disk (mm3-hiddens-file.h) ─────────────────────────────
    // The same block as `reuse_ar` above, written to and read from a
    // `.mm3hiddens` file so a pinned plan survives a restart. Loading one
    // PRIMES the in-memory slot rather than going round it, so every downstream
    // decision a hit makes — skip the LM load, skip the staged handover, hand
    // back the cached codes and LRC — is made identically here.
    //
    // The file carries the cache key with it and a model-side mismatch is
    // refused, which is the whole reason this is safe to expose: a block made
    // under a different LM quant has the same SHAPE and is numerically
    // meaningless. See mm3-hiddens-file.h for what is refused and what warns.
    std::string forced_frame_hiddens_file;
    bool        save_frame_hiddens = false;
    std::string frame_hiddens_save_path;

    // ── Streaming (mm3-job.h + GET /mm3/stream) ────────────────────────────
    // Emit each window's PCM as soon as it is vocoded and cropped, so a caller
    // can start playing before the render finishes. OPT-IN per request: it
    // moves the vocoder inside the flow loop (mm3-pipeline.h MM3ChunkCb) and
    // buffers the emitted chunks in host RAM until a reader drains them, and a
    // plain render should pay for neither. The finished WAV is produced and
    // delivered exactly as it is today — streaming is an ADDITIONAL output,
    // never a replacement.
    bool    stream = false;
    /** Ensemble takes: how many DIFFERENT songs to render from this one prompt
     *  in a single batched AR pass (mm3-ar-loop.h). 1 = an ordinary render.
     *  Clamped to what the checkpoint's row budget allows — 4 with a CFG pair —
     *  rather than rejected, so a caller that asks for more gets fewer songs
     *  and a log line, not an error. Take t uses seed + t. */
    int     takes  = 1;

    // Replay previously-captured codes instead of sampling them. Both must be
    // present together, with acoustic == semantic * 7. Entry 0 is the
    // un-emitted iteration, so I codes render I-1 frames (mm3-ar-loop.h).
    //
    // NOTE this does NOT make the render faster. The AR loop still executes
    // every per-frame forward pass; forcing fixes which TOKENS come out, not
    // how much compute runs. Its value is reproducibility — an identical
    // semantic bed to A/B flow-stage solver/scheduler/guidance against.
    std::vector<int32_t> forced_semantic;  // [I]
    std::vector<int32_t> forced_acoustic;  // [I * 7], flat, iteration-major

    // ── Runtime LM LoRA (mm3-lm-adapter.h) ─────────────────────────────────
    // Path to a PEFT LM adapter (SimpleTuner language_model checkpoints).
    // Empty = base model. The group scales mirror the 2026-08-20 ablation
    // dials: attention carries plan/genre, MLP carries vocal identity (the
    // ear-validated production setting is attn 1.0 / mlp 0.5); the depth
    // thirds are advanced dials (halving `late` destabilised termination).
    std::string        lm_adapter;
    MM3LmAdapterScales lm_adapter_scales = {};
    // "runtime" (default) = low-rank deltas in-graph, live dials, +28 %/step
    // at r256. "merge" = fold scale·B·A into the resident weights once
    // (mm3-lm-merge.h) — zero per-step cost, scale changes re-merge.
    std::string        lm_adapter_mode = "runtime";

    std::string prompt;  // the assembled template
    int64_t     n_tokens = 0;

    MM3GenRequest gen;
};

// Read one JSON field with a type check that fails LOUDLY: a caller that sends
// `"seed": "42"` must be told, not silently given the default.
static bool mm3_req_num(yyjson_val * root, const char * key, double * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_num(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a number";
        }
        return false;
    }
    // yyjson_get_num, NOT yyjson_get_real: a JSON integer literal is stored as
    // uint/sint and yyjson_get_real returns 0.0 for it, so `"duration": 10`
    // would read as zero. Caught by the first smoke test.
    *out     = yyjson_get_num(v);
    *present = true;
    return true;
}

static bool mm3_req_str(yyjson_val * root, const char * key, std::string * out, bool * present, std::string * err) {
    *present       = false;
    yyjson_val * v = root ? yyjson_obj_get(root, key) : nullptr;
    if (!v || yyjson_is_null(v)) {
        return true;
    }
    if (!yyjson_is_str(v)) {
        if (err) {
            *err = std::string("\"") + key + "\" must be a string";
        }
        return false;
    }
    out->assign(yyjson_get_str(v), yyjson_get_len(v));
    *present = true;
    return true;
}

// Parse + assemble. Does NOT tokenise (that needs the LM GGUF header) — call
// mm3_request_tokenize() next.
//
// Contract, with types and defaults:
//   caption    string, REQUIRED, non-blank
//   lyrics     string, optional, "" -> instrumental
//   duration   number (seconds), REQUIRED unless max_frames is given
//   max_frames integer, optional escape hatch; wins over duration
//   seed       integer, default -1 (= draw one from std::random_device)
//   cfg_flow   number, default = the checkpoint's flow.cfg_scale (1.7)
//   steps      integer, default = the checkpoint's flow.steps (30)
//   get_wav_bits integer in {16, 24, 32}, default 16
//   get_lrc      bool, default false — lyric timestamps (LRC) from the
//                alignment heads; costs the manual attention path on 3 of
//                36 LM layers, and is a no-op on an instrumental
//   infer_method       string, default "" — Lua solver plugin name (ACE's field
//                      name, shared registry). "" = native Euler.
//   scheduler          string, default "" — Lua scheduler plugin. "" = native
//                      inverted-linspace sigmas.
//   guidance_mode      string, default "" — Lua guidance plugin, or "apg" for
//                      the native APG path. "" = plain CFG, MM3's own.
//   flow_shift         number in (0, 20], default 1.0 — timestep warp handed to
//                      a scheduler plugin. Ignored without one.
//   apg_norm_threshold number in [0, 100], default 2.5
//   plugin_params      object {"pluginName:key": value}, default {}
//
//   The five plugin fields are ALL optional and all default to the native
//   path, which is the one the parity fixtures cover. See mm3-plugins.h.
static bool mm3_parse_synth_request(const MM3Model & m, yyjson_val * root, MM3SynthRequest * out, std::string * err) {
    *out = MM3SynthRequest{};

    bool present = false;
    if (!mm3_req_str(root, "caption", &out->caption, &present, err)) {
        return false;
    }
    if (!present || mm3_str_blank(out->caption)) {
        if (err) {
            *err = "\"caption\" is required and must be a non-empty string";
        }
        return false;
    }
    if (!mm3_req_str(root, "lyrics", &out->lyrics, &present, err)) {
        return false;
    }

    const int64_t frame_rate = m.lm_cfg.frame_rate ? (int64_t) m.lm_cfg.frame_rate : 25;
    int64_t       cap        = (int64_t) m.lm_cfg.max_audio_frames;
    if (cap <= 0 || cap > MM3_MAX_AUDIO_FRAMES) {
        cap = MM3_MAX_AUDIO_FRAMES;
    }

    double  dur = 0.0;
    bool    have_dur = false;
    if (!mm3_req_num(root, "duration", &dur, &have_dur, err)) {
        return false;
    }
    double  mf       = 0.0;
    bool    have_mf  = false;
    if (!mm3_req_num(root, "max_frames", &mf, &have_mf, err)) {
        return false;
    }
    if (have_mf) {
        out->max_frames = (int64_t) llround(mf);
        out->duration   = (double) out->max_frames / (double) frame_rate;
    } else if (have_dur) {
        if (!(dur > 0.0)) {
            if (err) {
                *err = "\"duration\" must be greater than 0 seconds";
            }
            return false;
        }
        out->duration   = dur;
        out->max_frames = (int64_t) llround(dur * (double) frame_rate);
    } else {
        if (err) {
            *err = "\"duration\" (seconds) is required";
        }
        return false;
    }
    if (out->max_frames < 1) {
        out->max_frames = 1;
    }
    if (out->max_frames > cap) {
        out->max_frames = cap;  // min(duration * 25, 9000)
    }

    double  seed_d  = -1.0;
    bool    have_seed = false;
    if (!mm3_req_num(root, "seed", &seed_d, &have_seed, err)) {
        return false;
    }
    out->seed_in = have_seed ? (int64_t) llround(seed_d) : -1;
    uint64_t seed;
    if (out->seed_in < 0) {
        std::random_device rd;
        seed = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
    } else {
        seed = (uint64_t) out->seed_in;
    }

    // ar_seed — optional, -1 (the default) ties the AR stage to `seed`. A
    // random `seed` therefore still gives a random plan; splitting them is a
    // deliberate act. Note the asymmetry: an absent ar_seed resolves to the
    // seed that was actually drawn, so the AR cache key stays stable across a
    // re-render of the same take.
    double ar_seed_d   = -1.0;
    bool   have_ar_seed = false;
    if (!mm3_req_num(root, "ar_seed", &ar_seed_d, &have_ar_seed, err)) {
        return false;
    }
    out->ar_seed_in       = have_ar_seed ? (int64_t) llround(ar_seed_d) : -1;
    const uint64_t ar_seed = out->ar_seed_in >= 0 ? (uint64_t) out->ar_seed_in : seed;

    double cfg = m.synth_cfg.flow.cfg_scale > 0.0f ? (double) m.synth_cfg.flow.cfg_scale : 1.7;
    if (!mm3_req_num(root, "cfg_flow", &cfg, &present, err)) {
        return false;
    }
    if (!(cfg > 0.0) || cfg > 100.0) {
        if (err) {
            *err = "\"cfg_flow\" must be in (0, 100]";
        }
        return false;
    }

    double steps = m.synth_cfg.flow.steps ? (double) m.synth_cfg.flow.steps : 30.0;
    if (!mm3_req_num(root, "steps", &steps, &present, err)) {
        return false;
    }
    const int64_t nsteps = (int64_t) llround(steps);
    if (nsteps < 1 || nsteps > 1000) {
        if (err) {
            *err = "\"steps\" must be in 1..1000";
        }
        return false;
    }

    double bits = 16.0;
    if (!mm3_req_num(root, "get_wav_bits", &bits, &present, err)) {
        return false;
    }
    out->wav_bits = (int) llround(bits);
    if (out->wav_bits != 16 && out->wav_bits != 24 && out->wav_bits != 32) {
        if (err) {
            *err = "\"get_wav_bits\" must be 16, 24 or 32";
        }
        return false;
    }

    // get_lrc — optional bool. Absent/false keeps the cheaper all-flash path.
    {
        yyjson_val * v = yyjson_obj_get(root, "get_lrc");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) {
                if (err) {
                    *err = "\"get_lrc\" must be a boolean";
                }
                return false;
            }
            out->want_lrc = yyjson_get_bool(v);
        }
    }

    // ── MM3 Plank: AR code capture ──
    {
        yyjson_val * v = yyjson_obj_get(root, "get_ar_codes");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) {
                if (err) {
                    *err = "\"get_ar_codes\" must be a boolean";
                }
                return false;
            }
            out->get_ar_codes = yyjson_get_bool(v);
        }
    }

    // ── Streaming ──
    {
        yyjson_val * v = yyjson_obj_get(root, "stream");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) {
                if (err) {
                    *err = "\"stream\" must be a boolean";
                }
                return false;
            }
            out->stream = yyjson_get_bool(v);
        }
    }

    // ── Ensemble takes ──
    {
        yyjson_val * v = yyjson_obj_get(root, "takes");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_int(v)) {
                if (err) {
                    *err = "\"takes\" must be an integer";
                }
                return false;
            }
            const int64_t t = yyjson_get_sint(v);
            if (t < 1 || t > MM3_MAX_BATCH_ROWS) {
                if (err) {
                    *err = "\"takes\" must be between 1 and " + std::to_string(MM3_MAX_BATCH_ROWS);
                }
                return false;
            }
            out->takes = (int) t;
        }
    }

    // ── AR cache: reuse_ar ──
    {
        yyjson_val * v = yyjson_obj_get(root, "reuse_ar");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_bool(v)) {
                if (err) {
                    *err = "\"reuse_ar\" must be a boolean";
                }
                return false;
            }
            out->reuse_ar = yyjson_get_bool(v);
        }
    }

    // ── AR cache on disk: replay from / save to a .mm3hiddens file ──
    {
        yyjson_val * v = yyjson_obj_get(root, "forced_frame_hiddens_file");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_str(v)) {
                if (err) {
                    *err = "\"forced_frame_hiddens_file\" must be a string path";
                }
                return false;
            }
            out->forced_frame_hiddens_file = yyjson_get_str(v);
        }
        yyjson_val * sv = yyjson_obj_get(root, "save_frame_hiddens");
        if (sv && !yyjson_is_null(sv)) {
            if (!yyjson_is_bool(sv)) {
                if (err) {
                    *err = "\"save_frame_hiddens\" must be a boolean";
                }
                return false;
            }
            out->save_frame_hiddens = yyjson_get_bool(sv);
        }
        yyjson_val * pv = yyjson_obj_get(root, "frame_hiddens_save_path");
        if (pv && !yyjson_is_null(pv)) {
            if (!yyjson_is_str(pv)) {
                if (err) {
                    *err = "\"frame_hiddens_save_path\" must be a string path";
                }
                return false;
            }
            out->frame_hiddens_save_path = yyjson_get_str(pv);
        }
        // A save with nowhere to put it is a caller bug, and silently dropping
        // it would look exactly like a save that worked.
        if (out->save_frame_hiddens && out->frame_hiddens_save_path.empty()) {
            if (err) {
                *err = "\"save_frame_hiddens\" needs \"frame_hiddens_save_path\"";
            }
            return false;
        }
        // The plank is parsed further down, so the "not both at once" check has
        // to live there rather than here.
    }

    // ── Runtime LM LoRA ──
    {
        yyjson_val * v = yyjson_obj_get(root, "lm_adapter");
        if (v && !yyjson_is_null(v)) {
            if (!yyjson_is_str(v)) {
                if (err) {
                    *err = "\"lm_adapter\" must be a string path";
                }
                return false;
            }
            out->lm_adapter = yyjson_get_str(v);
        }
        yyjson_val * mv = yyjson_obj_get(root, "lm_adapter_mode");
        if (mv && !yyjson_is_null(mv)) {
            if (!yyjson_is_str(mv)) {
                if (err) {
                    *err = "\"lm_adapter_mode\" must be \"runtime\" or \"merge\"";
                }
                return false;
            }
            const std::string mode = yyjson_get_str(mv);
            if (mode != "runtime" && mode != "merge") {
                if (err) {
                    *err = "\"lm_adapter_mode\" must be \"runtime\" or \"merge\", got \"" + mode + "\"";
                }
                return false;
            }
            out->lm_adapter_mode = mode;
        }
        struct {
            const char * key;
            float *      dst;
        } dials[] = {
            { "lm_adapter_scale",       &out->lm_adapter_scales.global },
            { "lm_adapter_scale_attn",  &out->lm_adapter_scales.attn   },
            { "lm_adapter_scale_mlp",   &out->lm_adapter_scales.mlp    },
            { "lm_adapter_scale_early", &out->lm_adapter_scales.early  },
            { "lm_adapter_scale_mid",   &out->lm_adapter_scales.mid    },
            { "lm_adapter_scale_late",  &out->lm_adapter_scales.late   },
        };
        for (auto & d : dials) {
            yyjson_val * sv = yyjson_obj_get(root, d.key);
            if (sv && !yyjson_is_null(sv)) {
                if (!yyjson_is_num(sv)) {
                    if (err) {
                        *err = std::string("\"") + d.key + "\" must be a number";
                    }
                    return false;
                }
                float f = (float) yyjson_get_num(sv);
                if (!(f >= -4.0f && f <= 4.0f)) {  // NaN fails this too
                    if (err) {
                        *err = std::string("\"") + d.key + "\" out of range [-4, 4]";
                    }
                    return false;
                }
                *d.dst = f;
            }
        }
    }

    // ── MM3 Plank: AR replay ──
    // Both arrays or neither. A half-specified replay is a caller bug worth a
    // 400, not a silent fall back to sampling — the whole point of asking for
    // replay is that the codes are pinned.
    {
        yyjson_val * sem_v  = yyjson_obj_get(root, "forced_semantic");
        yyjson_val * ac_v   = yyjson_obj_get(root, "forced_acoustic");
        const bool   has_sem = sem_v && !yyjson_is_null(sem_v);
        const bool   has_ac  = ac_v  && !yyjson_is_null(ac_v);

        if (has_sem != has_ac) {
            if (err) {
                *err = "\"forced_semantic\" and \"forced_acoustic\" must be provided together";
            }
            return false;
        }
        if (has_sem) {
            if (!yyjson_is_arr(sem_v) || !yyjson_is_arr(ac_v)) {
                if (err) {
                    *err = "\"forced_semantic\" and \"forced_acoustic\" must be arrays of integers";
                }
                return false;
            }
            size_t       idx, imax;
            yyjson_val * item;
            yyjson_arr_foreach(sem_v, idx, imax, item) {
                if (!yyjson_is_int(item)) {
                    if (err) {
                        *err = "\"forced_semantic\" must be an array of integers";
                    }
                    return false;
                }
                out->forced_semantic.push_back((int32_t) yyjson_get_int(item));
            }
            yyjson_arr_foreach(ac_v, idx, imax, item) {
                if (!yyjson_is_int(item)) {
                    if (err) {
                        *err = "\"forced_acoustic\" must be an array of integers";
                    }
                    return false;
                }
                out->forced_acoustic.push_back((int32_t) yyjson_get_int(item));
            }

            const int64_t NCB      = 7;  // acoustic codebooks per frame
            const int64_t expected = (int64_t) out->forced_semantic.size() * NCB;
            if ((int64_t) out->forced_acoustic.size() != expected) {
                if (err) {
                    char buf[160];
                    snprintf(buf, sizeof(buf),
                             "\"forced_acoustic\" has %zu entries, expected %lld (= %zu frames * %lld codebooks)",
                             out->forced_acoustic.size(), (long long) expected,
                             out->forced_semantic.size(), (long long) NCB);
                    *err = buf;
                }
                return false;
            }
            // Entry 0 is the un-emitted iteration, so I codes render I-1 frames.
            // mm3-ar-loop.h clamps max_frames to this itself, but doing it here
            // too keeps `duration` (which the flow stage and the job report both
            // read) honest about what will actually come out.
            if (out->forced_semantic.size() < 2) {
                if (err) {
                    *err = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
                }
                return false;
            }
            out->max_frames = (int64_t) out->forced_semantic.size() - 1;
            out->duration   = (double) out->max_frames / (double) frame_rate;
        }
        // A plank AND a hidden-block file is ambiguous rather than harmful: the
        // plank pins the codes and then re-derives the hiddens from them, the
        // file supplies the hiddens outright, so one of the two would silently
        // win (the file, since it skips the AR stage the codes would drive).
        // Say so instead of picking one behind the caller's back.
        if (!out->forced_frame_hiddens_file.empty() && !out->forced_semantic.empty()) {
            if (err) {
                *err = "\"forced_frame_hiddens_file\" and \"forced_semantic\" are mutually exclusive "
                       "(the hidden block already holds the plan the codes would rebuild)";
            }
            return false;
        }
    }

    // ── Sampler plugins (mm3-plugins.h) ──
    // Field names are ACE's on purpose: one UI control set, one wire vocabulary,
    // and a request that can be moved between backends without translation.
    // ALL OPTIONAL. Absent or empty == the native, parity-proven flow loop, so
    // every existing caller keeps the arithmetic the fixtures were built on.
    MM3PluginSel plug;
    if (!mm3_req_str(root, "infer_method", &plug.solver, &present, err)) {
        return false;
    }
    if (!mm3_req_str(root, "scheduler", &plug.scheduler, &present, err)) {
        return false;
    }
    if (!mm3_req_str(root, "guidance_mode", &plug.guidance, &present, err)) {
        return false;
    }

    // MM3's own scheduler is hardcoded shift=1 upstream, so the default
    // reproduces it. Only consulted when a scheduler plugin is named.
    double shift = 1.0;
    if (!mm3_req_num(root, "flow_shift", &shift, &present, err)) {
        return false;
    }
    if (!(shift > 0.0) || shift > 20.0) {
        if (err) {
            *err = "\"flow_shift\" must be in (0, 20]";
        }
        return false;
    }
    plug.shift = (float) shift;

    double apg_thr = 2.5;
    if (!mm3_req_num(root, "apg_norm_threshold", &apg_thr, &present, err)) {
        return false;
    }
    if (apg_thr < 0.0 || apg_thr > 100.0) {
        if (err) {
            *err = "\"apg_norm_threshold\" must be in [0, 100]";
        }
        return false;
    }
    plug.apg_norm_threshold = (float) apg_thr;

    // Declared plugin params: {"pluginName:key": value}. Same shape and same
    // coercion as parse_server_fields() in hot-step-server.cpp — a mismatch
    // would make the SAME UI control mean different things per backend.
    {
        yyjson_val * pp_obj = yyjson_obj_get(root, "plugin_params");
        if (pp_obj && !yyjson_is_null(pp_obj)) {
            if (!yyjson_is_obj(pp_obj)) {
                if (err) {
                    *err = "\"plugin_params\" must be an object";
                }
                return false;
            }
            yyjson_obj_iter it;
            yyjson_obj_iter_init(pp_obj, &it);
            yyjson_val * k;
            while ((k = yyjson_obj_iter_next(&it))) {
                yyjson_val * v = yyjson_obj_iter_get_val(k);
                std::string  vs;
                if (yyjson_is_str(v)) {
                    vs = yyjson_get_str(v);
                } else if (yyjson_is_real(v)) {
                    vs = std::to_string(yyjson_get_real(v));
                } else if (yyjson_is_int(v)) {
                    vs = std::to_string(yyjson_get_int(v));
                } else if (yyjson_is_bool(v)) {
                    vs = yyjson_get_bool(v) ? "true" : "false";
                }
                plug.params[std::string(yyjson_get_str(k))] = vs;
            }
        }
    }

    out->prompt = mm3_assemble_prompt(out->caption, out->lyrics, &out->instrumental);

    out->gen            = MM3GenRequest{};
    out->gen.prompt     = out->prompt;
    out->gen.max_frames = out->max_frames;
    out->gen.seed       = seed;
    out->gen.ar_seed_set = true;   // always resolved past here, tied or not
    out->gen.ar_seed     = ar_seed;
    out->gen.steps      = (int) nsteps;
    out->gen.cfg_flow   = (float) cfg;
    out->gen.plugins    = plug;
    // MM3 Plank replay. MM3GenRequest holds these by value, and the job worker
    // takes the whole MM3SynthRequest by value too, so the copy chain is safe —
    // no pointer into this parse's storage survives the handoff.
    if (!out->forced_semantic.empty()) {
        out->gen.forced_semantic = out->forced_semantic;
        out->gen.forced_acoustic = out->forced_acoustic;
    }
    // LM adapter scales travel by value; the adapter POINTER is resolved and
    // attached by the job layer (mm3-job.h), which owns the loaded adapter's
    // lifetime — a path string must never be dereferenced this early because
    // parse runs on the HTTP thread and load must run on the GPU worker.
    out->gen.lm_adapter_scales = out->lm_adapter_scales;

    // Sampling knobs parse LAST, after the finalize above REBUILDS out->gen
    // from scratch (out->gen = MM3GenRequest{}). Parsed any earlier they are
    // silently wiped — which is exactly what shipped, and what the identical-
    // output-at-any-penalty report was.
    {
        double dv = 0.0;
        if (!mm3_req_num(root, "lm_temperature", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_temperature = (float) (dv < 0.05 ? 0.05 : dv > 4.0 ? 4.0 : dv); }
        if (!mm3_req_num(root, "lm_top_k", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_top_k = (int) (dv < 0 ? 0 : dv > 16385 ? 16385 : dv); }
        if (!mm3_req_num(root, "lm_top_p", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_top_p = (float) (dv < 0.0 ? 0.0 : dv > 1.0 ? 1.0 : dv); }
        if (!mm3_req_num(root, "lm_rep_penalty", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_rep_penalty = (float) (dv < 1.0 ? 1.0 : dv > 2.0 ? 2.0 : dv); }
        if (!mm3_req_num(root, "lm_rep_window", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_rep_window = (int) (dv < 0 ? 0 : dv > 4000 ? 4000 : dv); }
        if (!mm3_req_num(root, "lm_dry_base", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_dry_base = (float) (dv < 1.05 ? 1.05 : dv > 4.0 ? 4.0 : dv); }
        if (!mm3_req_num(root, "lm_dry_min_len", &dv, &present, err)) { return false; }
        if (present) { out->gen.lm_dry_min_len = (int) (dv < 1 ? 1 : dv > 200 ? 200 : dv); }
        if (!mm3_req_str(root, "lm_rep_mode", &out->gen.lm_rep_mode, &present, err)) { return false; }
    }

    return true;
}

// Tokenise the assembled prompt and enforce the 5,000-token checkpoint limit
// (encoders.py:224-227). Header-only GGUF read — works on a cold server.
// The error names the ACTUAL count so a UI can tell the user how much to cut.
static bool mm3_request_tokenize(const MM3Model & m, MM3Tokenizer * tok, MM3SynthRequest * req, std::string * err) {
    if (!mm3_tokenizer_load(m, tok, err)) {
        return false;
    }
    mm3_tokenizer_encode(*tok, req->prompt, &req->gen.ids_cond);
    req->n_tokens = (int64_t) req->gen.ids_cond.size();
    if (req->n_tokens < 3) {
        if (err) {
            *err = "the assembled prompt tokenises to fewer than 3 tokens";
        }
        return false;
    }
    if (req->n_tokens > MM3_MAX_PROMPT_TOKENS) {
        if (err) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "the assembled prompt is %lld tokens; the checkpoint limit is %d "
                     "(shorten the caption or the lyrics by about %lld tokens)",
                     (long long) req->n_tokens, MM3_MAX_PROMPT_TOKENS,
                     (long long) (req->n_tokens - MM3_MAX_PROMPT_TOKENS));
            *err = buf;
        }
        return false;
    }
    mm3_tokenizer_uncond(m.lm_cfg, req->gen.ids_cond, &req->gen.ids_uncond);
    return true;
}
