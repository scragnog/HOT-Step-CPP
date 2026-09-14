#pragma once
// train/yue2-sidecar.h — the ACE dataset sidecar parser, in ONE place.
//
// HOT-Step file, TRAINING-SIDE. `<stem>.txt` beside the source audio, in
// HOT-Step's own Option-A sidecar format:
//
//     caption: Acoustic rock and pop punk blend across a steady drum beat, ...
//     genre: Garage Rock
//     bpm: 121
//     key: D Major
//     ...
//     lyrics:
//     [Verse 1]
//     This is a public service announcement, this is only a test
//     ...
//
// The canonical parser is server/src/services/training/sidecarIO.ts:58-82 and
// its ONE non-obvious rule is copied exactly:
//
//     ONCE `lyrics:` STARTS, EVERY SUBSEQUENT LINE IS LYRICS — INCLUDING LINES
//     CONTAINING COLONS.
//
// That is safe because the writer always emits lyrics last (sidecarIO.ts:147-162).
// Getting it wrong puts `bpm: 121` and the whole lyric sheet inside the style
// string and leaves [Lyrics] empty, which trains and is wrong — the worst
// combination, and the reason this is transcribed rather than approximated.
//
// A line starting with '[' is NEVER a key, so a `[Verse 1]` heading inside a
// value cannot open a new field.
//
// ── Why this is its own header ─────────────────────────────────────────────
//
// Two callers need the identical rules and neither includes the other:
//
//   * train/yue2-preprocess-run.h  — `--caption-mode ace` writes `caption` and
//     `lyrics` per source and per clip into yue2_preprocess.json.
//   * train/yue2-ar-train-run.h    — `--sidecars` reads them at training time
//     when the manifest predates that mode.
//
// Two copies of a parser whose failure mode is "trains at full speed on a
// mangled prefix" is exactly the drift docs/plans/yue2/14-ar-lora-contract.md
// §5.2 warns about. There is one implementation; both sides call it.
//
// ── What it does NOT do ────────────────────────────────────────────────────
//
// It returns the descriptive `caption` prose and the `lyrics` sheet, and
// nothing else. `genre`, `bpm`, `key` and `signature` are parsed only so they
// cannot leak into the caption by opening a continuation line — they are never
// returned. A YuE2 style prompt is a description, not a field dump; pasting
// "bpm: 121 | key: D Major" into [Tags] is the field-noise problem, not a
// richer prompt.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Parse an Option-A sidecar body. Returns true when at least one of the two
// interesting fields was found. `caption` and `lyrics` are always assigned
// (possibly empty), so a caller need not clear them first. `meta_out`, when
// given, receives every parsed field (lower-cased keys, trimmed values) —
// `genre`, `bpm`, `key` included — for the style TEMPLATE below, which is
// the one legitimate consumer: prose, not a field dump.
static bool yue2_sidecar_parse_meta(const std::string & text, std::string * caption, std::string * lyrics,
                                    std::map<std::string, std::string> * meta_out) {
    caption->clear();
    lyrics->clear();
    if (text.empty()) {
        return false;
    }
    size_t i = 0;
    if (text.size() >= 3 && (unsigned char) text[0] == 0xEF && (unsigned char) text[1] == 0xBB &&
        (unsigned char) text[2] == 0xBF) {
        i = 3;  // UTF-8 BOM
    }
    std::string                        cur_key;
    std::vector<std::string>           cur_lines;
    std::map<std::string, std::string> meta;
    auto flush = [&]() {
        if (!cur_key.empty()) {
            std::string v;
            for (size_t k = 0; k < cur_lines.size(); k++) {
                if (k) {
                    v += '\n';
                }
                v += cur_lines[k];
            }
            // trim
            size_t b = v.find_first_not_of(" \t\r\n");
            size_t e = v.find_last_not_of(" \t\r\n");
            meta[cur_key] = (b == std::string::npos) ? std::string() : v.substr(b, e - b + 1);
        }
        cur_key.clear();
        cur_lines.clear();
    };
    while (i <= text.size()) {
        size_t nl = text.find('\n', i);
        if (nl == std::string::npos) {
            nl = text.size();
        }
        std::string line = text.substr(i, nl - i);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        i = nl + 1;
        if (cur_key == "lyrics") {
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            cur_lines.push_back(line);
            if (nl >= text.size()) {
                break;
            }
            continue;
        }
        std::string ls = line;
        ls.erase(0, ls.find_first_not_of(" \t") == std::string::npos ? ls.size() : ls.find_first_not_of(" \t"));
        const size_t colon = ls.find(':');
        if (colon != std::string::npos && !ls.empty() && ls[0] != '[') {
            std::string key = ls.substr(0, colon);
            size_t      ke  = key.find_last_not_of(" \t");
            key             = (ke == std::string::npos) ? std::string() : key.substr(0, ke + 1);
            for (char & ch : key) {
                ch = (char) tolower((unsigned char) ch);
            }
            // A known-field whitelist, like sidecarIO.ts's KNOWN_SIDECAR_KEYS.
            // Without it a caption that wraps onto a second line containing a
            // colon would open a bogus field and silently truncate the style.
            static const char * fields[] = { "caption", "genre",     "bpm",        "key",
                                             "mood",    "instruments", "vocals",   "language",
                                             "tags",    "style",     "duration",   "custom_tag",
                                             "repeat",  "prompt_override", "title", "artist",
                                             "album",   "year",      "energy",     "time_signature",
                                             "signature", "is_instrumental", "trigger", "tag_position",
                                             "lyrics" };
            bool known = false;
            for (const char * fkey : fields) {
                if (key == fkey) {
                    known = true;
                    break;
                }
            }
            if (known) {
                flush();
                cur_key = key;
                std::string rest = ls.substr(colon + 1);
                size_t      rb   = rest.find_first_not_of(" \t");
                cur_lines.push_back(rb == std::string::npos ? std::string() : rest.substr(rb));
                if (nl >= text.size()) {
                    break;
                }
                continue;
            }
        }
        if (!cur_key.empty()) {
            cur_lines.push_back(ls);
        }
        if (nl >= text.size()) {
            break;
        }
    }
    flush();
    auto get = [&](const char * k) {
        auto it = meta.find(k);
        return it == meta.end() ? std::string() : it->second;
    };
    *caption = get("caption");
    *lyrics  = get("lyrics");
    if (meta_out) {
        *meta_out = meta;
    }
    return !caption->empty() || !lyrics->empty();
}

static bool yue2_sidecar_parse(const std::string & text, std::string * caption, std::string * lyrics) {
    return yue2_sidecar_parse_meta(text, caption, lyrics, nullptr);
}

// ── The style string, in upstream's TRAINING TEMPLATE ──────────────────────
//
// Mothersuperior's AR/NAR recipes address the artist as
//
//     <trigger>, in the style of <trigger>. <caption>. <genre>, <bpm> BPM, key of <key>.
//
// (make_layout.py / the trainer README's "xyzq, in the style of xyzq." form,
// with the genre/BPM/key tail from the sidecar as prose, joined by ", " and
// closed with a full stop; whitespace collapsed). Measured 2026-09-14
// (_LISTENING/2026-09-14/RESULTS.md, "the template x adapter x seed grid"):
// the SAME adapter ends and sounds like the artist under this template and
// caps under the bare "<trigger>, <caption>" — the template is a first-class
// control, so both trainers build it here and the app must compose the
// identical string at generation. `bare` keeps the old behaviour for
// adapters trained before this existed.
//
// An empty caption yields the trigger alone (upstream's train.py:192 rule);
// an empty trigger yields the caption plus tail with no "in the style of".
// --style-template's two legal spellings; anything else is a typo that would
// otherwise silently train the bare template (the != "bare" test).
static const char * yue2_style_template_check(const char * v) {
    if (v && (!strcmp(v, "upstream") || !strcmp(v, "bare"))) {
        return v;
    }
    fprintf(stderr, "--style-template must be `upstream` or `bare` (got \"%s\")\n", v ? v : "");
    exit(2);
}

static std::string yue2_style_string(const std::string & trigger, const std::string & caption,
                                     const std::string & genre, const std::string & bpm, const std::string & key,
                                     bool upstream_template) {
    auto squash = [](const std::string & s) {
        std::string o;
        bool        sp = false;
        for (char c : s) {
            const bool ws = (c == ' ' || c == '\t' || c == '\r' || c == '\n');
            if (ws) {
                sp = true;
                continue;
            }
            if (sp && !o.empty()) {
                o += ' ';
            }
            sp = false;
            o += c;
        }
        return o;
    };
    const std::string cap = squash(caption);
    if (!upstream_template) {
        if (cap.empty()) {
            return trigger;
        }
        if (trigger.empty()) {
            return cap;
        }
        return trigger + ", " + cap;
    }
    if (cap.empty()) {
        return trigger;
    }
    std::string tail;
    auto        bit = [&](const std::string & s) {
        if (s.empty()) {
            return;
        }
        if (!tail.empty()) {
            tail += ", ";
        }
        tail += s;
    };
    bit(squash(genre));
    if (!squash(bpm).empty()) {
        bit(squash(bpm) + " BPM");
    }
    if (!squash(key).empty()) {
        bit("key of " + squash(key));
    }
    if (!tail.empty()) {
        tail += ".";
    }
    std::string head = trigger.empty() ? std::string() : trigger + ", in the style of " + trigger + ". ";
    return squash(head + cap + " " + tail);
}

// Replace the last extension with ".txt" (data.py's `file.with_suffix(".txt")`).
static std::string yue2_sidecar_path(const std::string & audio) {
    const size_t dot   = audio.find_last_of('.');
    const size_t slash = audio.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return audio + ".txt";
    }
    return audio.substr(0, dot) + ".txt";
}
