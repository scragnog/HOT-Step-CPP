#pragma once
// yue2/sheetsage-events.h — SheetSage2 token stream -> decoded events -> the
// non-ABC export files (beat/chord/key/structure/melody LAB, events.tsv,
// notation/song_*.txt, notation/song_melody.mid).
//
// HOT-Step file (no acestep.cpp analog). Pure C++, no tensors, no exceptions
// across the pipeline boundary (every entry point returns bool and writes
// *err on failure, matching sheetsage-tokens.h / sheetsage-grammar.h style).
//
// AUTHORITY: docs/plans/yue2/21-sheetsage2-symbolic-pin.md §4-§6 (this file's
// scope), §7 EXCLUDED (ABC/notation rendering is a different lane — ABC text
// itself, `score.abc`/`score.browser.abc`, and anything requiring the
// notation module's `Score` object such as `playback.json`'s measure map are
// NOT built here; see the NOT IMPLEMENTED note above
// yue2_sheet_write_exports()). Where this file and doc 21 disagree, doc 21
// wins; where doc 21 and the actual shipped Python disagree, the Python
// wins — one confirmed instance of that is noted at
// yue2_sheet_seconds_to_tick() below. Python sources cited as
// <file>:<lines> under
// K:\yue2\.cache\huggingface-sheetsage\modules\transformers_modules\SheetSage2\,
// except the MIDI byte format, which is pretty_midi 0.2.10's own write()
// (K:\yue2\.venv-sheetsage\Lib\site-packages\pretty_midi\pretty_midi.py) and
// the mido 1.x MIDI-file writer it delegates to
// (...\Lib\site-packages\mido\midifiles\{midifiles,meta}.py) — SheetSage2
// never touches MIDI bytes itself, so those two files are the real authority
// for notation/song_melody.mid, cross-checked byte-for-byte against
// fixtures/short-a/21_exports/notation/song_melody.mid (see this file's
// test, engine/tests/sheetsage/test_events.cpp).
//
// Depends on sheetsage-tokens.h for the vocabulary (SheetSage2Tokens,
// yue2_sheet_token_type, the label tables) — that file existed and was
// complete by the time this one was written, so there is no private shim
// table here; every token lookup below goes through sheetsage-tokens.h.
// Stitching (doc 21 §4: sliding_window_plan, build_overlap_prefix_tokens,
// stitched_window_events, global_subbeat) is NOT this lane — this file reads
// an ALREADY-STITCHED event list (20_stitched_events.json, or the engine's
// own equivalent once the pipeline lane produces one) and turns it into
// export rows; it never assembles one itself.

#include "sheetsage-tokens.h"

#include "../../vendor/yyjson/yyjson.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════
// §1. Event data model (doc 21 §5.1's decoded event dict, plus the extra
//     fields stitching adds — doc 21 §4.3 — which this file reads but never
//     computes).
// ═════════════════════════════════════════════════════════════════════════

// One decoded melody note (_decode_field's "melody" case,
// tokenization_sheetsage2.py:507-529). `end_time` is set only on a STITCHED
// event's note (stitched_window_events, generation_sheetsage2.py:515-560) —
// a window-level decode (decode_sequence's own output, before stitching)
// never has it, matching 12_events.json's melody notes having no "end_time"
// key (cross-checked against the fixtures: window-level events carry
// {pitch,track,duration_bin,duration_steps} only).
struct Yue2SheetMelodyNote {
    int32_t pitch = 0;
    int32_t track = 0;
    int32_t duration_bin = 0;
    int32_t duration_steps = 0;
    bool has_end_time = false;
    double end_time = 0.0;
};

// One decoded event (decode_sequence's per-event dict, tokenization_sheetsage2.py:625-634,
// plus stitched_window_events's extra fields for a STITCHED event).
// `tokens_by_field` is kept per doc 21 §5's schema-order fields
// (timestamp,rhythm,structure,key,chord,melody == Yue2SheetField order) —
// only fields that survived the "drop empty entries" step (tokenization_sheetsage2.py:597-599)
// are non-empty, mirroring the reference dict's actual keys.
struct Yue2SheetEvent {
    int32_t subbeat = 0;

    std::array<std::vector<int32_t>, 6> tokens_by_field;  // indexed by Yue2SheetField

    bool has_timestamp = false;
    double timestamp = 0.0;

    bool has_rhythm = false;
    bool rhythm_has_meter = false;
    int32_t rhythm_meter_num = 0;
    int32_t rhythm_meter_den = 0;
    bool rhythm_has_eighth = false;
    int32_t rhythm_eighth = 0;

    bool has_structure = false;
    std::string structure;

    bool has_key = false;
    std::string key;

    bool has_chord = false;
    std::string chord;

    bool has_melody = false;
    std::vector<Yue2SheetMelodyNote> melody;

    // Stitching-only fields (stitched_window_events, generation_sheetsage2.py:515-560).
    // Never set by yue2_sheet_decode_sequence(); only by the stitched-events
    // JSON reader below.
    bool has_time = false;
    double time = 0.0;
    bool has_window_index = false;
    int32_t window_index = 0;
    bool has_window_start = false;
    double window_start = 0.0;
    bool has_source_subbeat = false;
    int32_t source_subbeat = 0;
    bool has_global_subbeat = false;
    int32_t global_subbeat = 0;
};

// decode_sequence()'s return value (tokenization_sheetsage2.py:640-645).
struct Yue2SheetDecoded {
    std::string schema_version;
    std::vector<std::string> prompts;  // canonical schema order (doc 21 §1.3)
    std::vector<Yue2SheetEvent> events;
    bool has_eos = false;
};

inline int yue2_sheet_field_index(const char * name) {
    static const char * order[6] = {"timestamp", "rhythm", "structure", "key", "chord", "melody"};
    for (int i = 0; i < 6; ++i) {
        if (std::strcmp(order[i], name) == 0) return i;
    }
    return -1;
}

// ═════════════════════════════════════════════════════════════════════════
// §2. Prompt handling: normalize_prompts (tokenization_sheetsage2.py:233-259),
//     scoped to exactly what decode_sequence's strict check needs.
// ═════════════════════════════════════════════════════════════════════════

namespace yue2_sheet_events_detail {

// sampling_group per V1_TASKS (schema_sheetsage2.py:28-37 / doc 21 §1.3).
// Only used for the mutual-exclusion check inside normalize_prompts — dead
// in practice for FULL_TASK_PROMPTS (doc 21 §1.3's note), kept for fidelity.
inline const char * prompt_sampling_group(const std::string & name) {
    if (name == "timestamp") return "timestamp";
    if (name == "downbeat_meter") return "rhythm";
    if (name == "structure") return "structure";
    if (name == "key") return "key";
    if (name == "chord_majmin") return "chord";
    if (name == "chord_full") return "chord";
    if (name == "melody_vocal") return "melody";
    if (name == "melody_full") return "melody";
    return "";
}

// Schema-order index of a prompt name (used by normalize_prompts's sort key,
// tokenization_sheetsage2.py:245).
inline int prompt_schema_index(const SheetSage2Tokens & vocab, const std::string & name) {
    for (int i = 0; i < SheetSage2Tokens::N_PROMPT_TASKS; ++i) {
        if (vocab.prompt_names[i] == name) return i;
    }
    return -1;
}

}  // namespace yue2_sheet_events_detail

// normalize_prompts(prompts) (tokenization_sheetsage2.py:233-259): dedupe,
// sort by canonical schema order, reject unknown names or two names sharing
// a sampling_group. Returns false with *err set on any of those.
inline bool yue2_sheet_normalize_prompts(const SheetSage2Tokens & vocab,
                                          const std::vector<std::string> & prompts,
                                          std::vector<std::string> * out,
                                          std::string * err) {
    std::vector<std::string> names;
    for (const std::string & raw : prompts) {
        std::string name = raw;
        // strip() + optional "<|name|>" wrapper (:237-239) — decode_sequence
        // never feeds a wrapped/whitespace-padded name (token_to_prompt
        // returns a bare name), so this is a straight passthrough here.
        if (vocab.prompt_name_to_id.find(name) == vocab.prompt_name_to_id.end()) {
            if (err) *err = "Unknown prompt: '" + name + "'";
            return false;
        }
        if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
    }
    std::stable_sort(names.begin(), names.end(), [&](const std::string & a, const std::string & b) {
        return yue2_sheet_events_detail::prompt_schema_index(vocab, a) <
               yue2_sheet_events_detail::prompt_schema_index(vocab, b);
    });
    std::unordered_map<std::string, std::string> selected_group;
    for (const std::string & name : names) {
        const char * group = yue2_sheet_events_detail::prompt_sampling_group(name);
        auto it = selected_group.find(group);
        if (it != selected_group.end() && it->second != name) {
            if (err) *err = "Prompts '" + it->second + "' and '" + name +
                             "' are mutually exclusive within sampling group '" + group + "'";
            return false;
        }
        selected_group[group] = name;
    }
    if (names.empty()) {
        if (err) *err = "At least one task prompt is required";
        return false;
    }
    if (out) *out = names;
    return true;
}

// ═════════════════════════════════════════════════════════════════════════
// §3. decode_sequence (tokenization_sheetsage2.py:536-645, doc 21 §5.1) and
//     decode_generated_tokens's recoverable-error wrapper
//     (generation_sheetsage2.py:281-299, doc 21 §5.4).
// ═════════════════════════════════════════════════════════════════════════

// SheetSage2Tokenizer._decode_field (tokenization_sheetsage2.py:481-534,
// doc 21 §5.3), applied to one field's already-collected token list.
// `prompts` is unused here (the reference threads it through but only the
// appended-token-block fallback reads it, and v1 has none — doc 21 §5.2's
// note) — kept as a parameter for signature fidelity, ignored otherwise.
inline bool yue2_sheet_decode_field(const SheetSage2Tokens & vocab, Yue2SheetField field,
                                     const std::vector<int32_t> & toks, Yue2SheetEvent * out,
                                     std::string * err) {
    switch (field) {
        case Yue2SheetField::Timestamp: {
            int32_t time_id = toks[0] - SheetSage2Tokens::TIME_TOKEN_START;
            out->has_timestamp = true;
            out->timestamp = yue2_sheet_time_seconds(time_id);
            return true;
        }
        case Yue2SheetField::Rhythm: {
            out->has_rhythm = true;
            for (int32_t tok : toks) {
                Yue2SheetTokenType t = yue2_sheet_token_type(tok);
                if (t == Yue2SheetTokenType::Meter) {
                    auto mp = yue2_sheet_meter_pair(vocab, tok - SheetSage2Tokens::METER_TOKEN_START);
                    out->rhythm_has_meter = true;
                    out->rhythm_meter_num = mp.first;
                    out->rhythm_meter_den = mp.second;
                } else if (t == Yue2SheetTokenType::EighthPosition) {
                    out->rhythm_has_eighth = true;
                    out->rhythm_eighth = tok - SheetSage2Tokens::EIGHTH_POSITION_TOKEN_START;
                }
            }
            return true;
        }
        case Yue2SheetField::Structure: {
            out->has_structure = true;
            out->structure = yue2_sheet_structure_label(vocab, toks[0] - SheetSage2Tokens::STRUCTURE_TOKEN_START);
            return true;
        }
        case Yue2SheetField::Key: {
            out->has_key = true;
            out->key = yue2_sheet_key_label(toks[0] - SheetSage2Tokens::KEY_TOKEN_START);
            return true;
        }
        case Yue2SheetField::Chord: {
            out->has_chord = true;
            Yue2SheetTokenType t = yue2_sheet_token_type(toks[0]);
            if (t == Yue2SheetTokenType::ChordMajmin) {
                out->chord = yue2_sheet_majmin_chord_label(vocab, toks[0] - SheetSage2Tokens::MAJMIN_CHORD_TOKEN_START);
            } else {
                out->chord = yue2_sheet_full_chord_label(vocab, toks[0] - SheetSage2Tokens::FULL_CHORD_TOKEN_START);
            }
            return true;
        }
        case Yue2SheetField::Melody: {
            out->has_melody = true;
            size_t i = 0;
            while (i < toks.size()) {
                int32_t pitch_id = toks[i] - SheetSage2Tokens::PITCH_TOKEN_START;
                int32_t duration_bin = 0;
                if (i + 1 < toks.size() && yue2_sheet_token_type(toks[i + 1]) == Yue2SheetTokenType::Duration) {
                    duration_bin = toks[i + 1] - SheetSage2Tokens::DURATION_TOKEN_START;
                    i += 2;
                } else {
                    i += 1;
                }
                Yue2SheetMelodyNote note;
                auto pt = yue2_sheet_pitch_track(pitch_id);
                note.pitch = pt.first;
                note.track = pt.second;
                note.duration_bin = duration_bin;
                note.duration_steps = yue2_sheet_duration_steps(vocab, duration_bin);
                out->melody.push_back(note);
            }
            return true;
        }
        default:
            if (err) *err = "internal: unknown field in yue2_sheet_decode_field";
            return false;
    }
}

// decode_sequence(tokens, strict) (tokenization_sheetsage2.py:536-645, doc
// 21 §5.1). `tokens` is passed by value since the reference mutates a local
// copy (trailing-pad strip); the vocab reference must already be built and
// fingerprint-checked (sheetsage-tokens.h) — this function does not call
// yue2_sheet_tokens_build() itself.
inline bool yue2_sheet_decode_sequence(const SheetSage2Tokens & vocab, std::vector<int32_t> tokens, bool strict,
                                        Yue2SheetDecoded * out, std::string * err) {
    while (!tokens.empty() && tokens.back() == SheetSage2Tokens::PAD) tokens.pop_back();
    if (tokens.empty() || tokens[0] != SheetSage2Tokens::SOS) {
        if (err) *err = "sequence must begin with <|sos|>";
        return false;
    }

    int out_index = -1;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == SheetSage2Tokens::OUT) {
            out_index = (int) i;
            break;
        }
    }
    if (out_index < 0) {
        if (err) *err = "sequence is missing <|out|>";
        return false;
    }

    std::vector<std::string> prompts;
    for (int i = 1; i < out_index; ++i) {
        int32_t tok = tokens[i];
        bool found = false;
        for (const auto & kv : vocab.prompt_name_to_id) {
            if (kv.second == tok) {
                prompts.push_back(kv.first);
                found = true;
                break;
            }
        }
        if (!found) {
            if (err) *err = "token " + std::to_string(tok) + " is not a prompt token";
            return false;
        }
    }
    // token_to_prompt scans prompt_to_id in TABLE (schema) order
    // (tokenization_sheetsage2.py:264-269); the unordered_map scan above can
    // land in a different order, but there is at most one id match per
    // token so the result is identical regardless of scan order.
    if (strict) {
        std::vector<std::string> normalized;
        if (!yue2_sheet_normalize_prompts(vocab, prompts, &normalized, nullptr) || normalized != prompts) {
            if (err) *err = "prompt tokens are not in canonical schema order";
            return false;
        }
    }
    bool active_field[6] = {false, false, false, false, false, false};
    for (const std::string & name : prompts) {
        for (int i = 0; i < SheetSage2Tokens::N_PROMPT_TASKS; ++i) {
            if (vocab.prompt_names[i] == name) {
                // task->output_field, V1_TASKS (schema_sheetsage2.py:28-37):
                // every task's output_field equals its sampling_group name
                // except chord_majmin/chord_full->"chord" and
                // melody_vocal/melody_full->"melody" — reuse the same table.
                const char * group = yue2_sheet_events_detail::prompt_sampling_group(name);
                int fi = yue2_sheet_field_index(group);
                if (fi >= 0) active_field[fi] = true;
            }
        }
    }

    std::vector<Yue2SheetEvent> events;
    size_t position = (size_t) out_index + 1;
    int32_t current_step = 0;
    bool saw_eos = false;
    while (position < tokens.size()) {
        int32_t token = tokens[position];
        if (token == SheetSage2Tokens::EOS) {
            saw_eos = true;
            position += 1;
            break;
        }
        if (yue2_sheet_token_type(token) != Yue2SheetTokenType::SubbeatShift) {
            if (err) *err = "event at token index " + std::to_string(position) + " has no subbeat shift";
            return false;
        }
        int32_t shift = 0;
        while (position < tokens.size() && yue2_sheet_token_type(tokens[position]) == Yue2SheetTokenType::SubbeatShift) {
            shift += yue2_sheet_token_to_subbeat_shift(tokens[position]);
            position += 1;
        }
        current_step += shift;

        std::array<std::vector<int32_t>, 6> tokens_by_field;
        while (position < tokens.size()) {
            token = tokens[position];
            Yue2SheetTokenType tt = yue2_sheet_token_type(token);
            if (tt == Yue2SheetTokenType::SubbeatShift || token == SheetSage2Tokens::EOS) break;
            Yue2SheetField field = yue2_sheet_token_output_field(tt);
            int fi = (field == Yue2SheetField::None) ? -1 : (int) field;
            if (fi < 0) {
                if (err) {
                    *err = "token " + std::to_string(token) + " has no field in schema v1";
                }
                return false;
            }
            if (strict && !active_field[fi]) {
                static const char * field_names[6] = {"timestamp", "rhythm", "structure", "key", "chord", "melody"};
                if (err) *err = std::string("token ") + std::to_string(token) +
                                 " belongs to inactive output field '" + field_names[fi] + "'";
                return false;
            }
            tokens_by_field[fi].push_back(token);
            position += 1;
        }

        bool any_field = false;
        for (const auto & v : tokens_by_field) any_field = any_field || !v.empty();
        if (!any_field) {
            if (strict) {
                if (err) *err = "empty event at subbeat " + std::to_string(current_step);
                return false;
            }
            continue;
        }

        if (strict) {
            for (int fi = 0; fi < 6; ++fi) {
                const auto & toks = tokens_by_field[fi];
                if (toks.empty()) continue;
                std::vector<Yue2SheetTokenType> types;
                for (int32_t t : toks) types.push_back(yue2_sheet_token_type(t));
                if (fi == 0 /* timestamp */) {
                    if (types.size() != 1 || types[0] != Yue2SheetTokenType::Time) {
                        if (err) *err = "timestamp event must contain exactly one time token";
                        return false;
                    }
                } else if (fi == 1 /* rhythm */) {
                    bool ok_eighth_only = types.size() == 1 && types[0] == Yue2SheetTokenType::EighthPosition;
                    bool ok_meter_eighth = types.size() == 2 && types[0] == Yue2SheetTokenType::Meter &&
                                            types[1] == Yue2SheetTokenType::EighthPosition;
                    if (!ok_eighth_only && !ok_meter_eighth) {
                        if (err) *err = "invalid rhythm payload";
                        return false;
                    }
                } else if (fi == 2 || fi == 3 || fi == 4 /* structure, key, chord */) {
                    if (toks.size() != 1) {
                        static const char * field_names[6] = {"timestamp", "rhythm", "structure", "key", "chord", "melody"};
                        if (err) *err = std::string("field '") + field_names[fi] + "' must contain exactly one token";
                        return false;
                    }
                } else if (fi == 5 /* melody */) {
                    size_t idx = 0;
                    while (idx < types.size()) {
                        if (types[idx] != Yue2SheetTokenType::Pitch) {
                            if (err) *err = "melody payload must contain pitch tokens with optional duration";
                            return false;
                        }
                        if (idx + 1 < types.size() && types[idx + 1] == Yue2SheetTokenType::Duration) idx += 2;
                        else idx += 1;
                    }
                }
            }
        }

        Yue2SheetEvent ev;
        ev.subbeat = current_step;
        ev.tokens_by_field = tokens_by_field;
        for (int fi = 0; fi < 6; ++fi) {
            if (tokens_by_field[fi].empty()) continue;
            if (!yue2_sheet_decode_field(vocab, (Yue2SheetField) fi, tokens_by_field[fi], &ev, err)) return false;
        }
        events.push_back(std::move(ev));
    }

    if (strict && !saw_eos) {
        if (err) *err = "sequence is missing <|eos|>";
        return false;
    }
    if (strict && position != tokens.size()) {
        if (err) *err = "non-padding tokens follow <|eos|>";
        return false;
    }
    if (out) {
        out->schema_version = "v1";
        out->prompts = prompts;
        out->events = std::move(events);
        out->has_eos = saw_eos;
    }
    return true;
}

// decode_generated_tokens(tokenizer, tokens, song_id, window_index)
// (generation_sheetsage2.py:281-299, doc 21 §5.4). Only the two named
// substrings are recoverable (retried with strict=false); everything else
// is a hard failure. `recovered` is set true when the non-strict retry ran.
inline bool yue2_sheet_decode_generated_tokens(const SheetSage2Tokens & vocab, const std::vector<int32_t> & tokens,
                                                Yue2SheetDecoded * out, bool * recovered, std::string * err) {
    std::string strict_err;
    if (yue2_sheet_decode_sequence(vocab, tokens, /*strict=*/true, out, &strict_err)) {
        if (recovered) *recovered = false;
        return true;
    }
    bool is_recoverable = strict_err.find("empty event at subbeat") != std::string::npos ||
                           strict_err.find("belongs to inactive output field") != std::string::npos;
    if (!is_recoverable) {
        if (err) *err = strict_err;
        return false;
    }
    if (recovered) *recovered = true;
    return yue2_sheet_decode_sequence(vocab, tokens, /*strict=*/false, out, err);
}

// ═════════════════════════════════════════════════════════════════════════
// §4. event_time_map (generation_sheetsage2.py:302-328, doc 21 §5.3).
//     Utility only — stitching (which is the only caller in the reference)
//     is NOT this lane, so nothing in this file's own export path or test
//     exercises this function; it is provided complete and to-spec because
//     doc 19/the task brief named it as required reading, and because a
//     stitching lane consuming this file will need it verbatim.
// ═════════════════════════════════════════════════════════════════════════

inline double yue2_sheet_np_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return std::nan("");
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

struct Yue2SheetTimeMap {
    bool has_anchors = false;
    std::vector<double> steps;  // strictly increasing (built from a dict: last-value-wins per key, §8)
    std::vector<double> times;
    double step_seconds = 0.125;
    double target_seconds = 0.0;

    double local_time(double step) const {
        if (!has_anchors) {
            return std::min(target_seconds, std::max(0.0, step * 0.125));
        }
        if (step <= steps.front()) {
            double v = times.front() + (step - steps.front()) * step_seconds;
            return std::min(target_seconds, std::max(0.0, v));
        }
        if (step >= steps.back()) {
            double v = times.back() + (step - steps.back()) * step_seconds;
            return std::min(target_seconds, std::max(0.0, v));
        }
        // np.interp: locate the bracketing pair and linearly interpolate.
        size_t hi = (size_t) (std::upper_bound(steps.begin(), steps.end(), step) - steps.begin());
        size_t lo = hi - 1;
        double t0 = steps[lo], t1 = steps[hi], v0 = times[lo], v1 = times[hi];
        if (t1 == t0) return v0;
        return v0 + (step - t0) * (v1 - v0) / (t1 - t0);
    }
};

inline Yue2SheetTimeMap yue2_sheet_event_time_map(const std::vector<Yue2SheetEvent> & decoded_events,
                                                   double target_seconds) {
    Yue2SheetTimeMap map;
    map.target_seconds = target_seconds;
    // dict(anchors).items(): a repeated subbeat keeps the LAST value seen for
    // that key (§8) — replicate with an ordered map that overwrites on
    // re-insert, then take the result in ascending-subbeat order.
    std::vector<std::pair<int32_t, double>> anchors;
    for (const Yue2SheetEvent & e : decoded_events) {
        if (!e.has_timestamp) continue;
        bool replaced = false;
        for (auto & a : anchors) {
            if (a.first == e.subbeat) {
                a.second = e.timestamp;
                replaced = true;
                break;
            }
        }
        if (!replaced) anchors.push_back({e.subbeat, e.timestamp});
    }
    if (anchors.empty()) {
        map.has_anchors = false;
        return map;
    }
    std::sort(anchors.begin(), anchors.end(), [](auto & a, auto & b) { return a.first < b.first; });
    map.has_anchors = true;
    for (auto & a : anchors) {
        map.steps.push_back((double) a.first);
        map.times.push_back(a.second);
    }
    if (anchors.size() >= 2) {
        std::vector<double> ratios;
        for (size_t i = 1; i < anchors.size(); ++i) {
            double dt = map.times[i] - map.times[i - 1];
            double ds = std::max(map.steps[i] - map.steps[i - 1], 1.0);
            ratios.push_back(dt / ds);
        }
        double ss = yue2_sheet_np_median(ratios);
        map.step_seconds = (std::isfinite(ss) && ss > 0.0) ? ss : 0.125;
    } else {
        map.step_seconds = 0.125;
    }
    return map;
}

// ═════════════════════════════════════════════════════════════════════════
// §5. field_text (generation_sheetsage2.py:331-349) — events.tsv's third
//     column, and the exact "; "/","/":" formatting rhythm_events.lab does
//     NOT use (rhythm_events.lab uses real JSON — §8 below).
// ═════════════════════════════════════════════════════════════════════════

inline std::string yue2_sheet_python_repr_double(double v);  // forward decl, §7

inline std::string yue2_sheet_field_text(const Yue2SheetEvent & e) {
    std::vector<std::string> parts;
    if (e.has_timestamp) parts.push_back("timestamp=" + yue2_sheet_python_repr_double(e.timestamp));
    if (e.has_rhythm) {
        std::vector<std::string> kv;
        if (e.rhythm_has_meter) {
            kv.push_back("meter:(" + std::to_string(e.rhythm_meter_num) + ", " + std::to_string(e.rhythm_meter_den) + ")");
        }
        if (e.rhythm_has_eighth) kv.push_back("eighth_position:" + std::to_string(e.rhythm_eighth));
        std::string s = "rhythm=";
        for (size_t i = 0; i < kv.size(); ++i) { if (i) s += ","; s += kv[i]; }
        parts.push_back(s);
    }
    if (e.has_structure) parts.push_back("structure=" + e.structure);
    if (e.has_key) parts.push_back("key=" + e.key);
    if (e.has_chord) parts.push_back("chord=" + e.chord);
    if (e.has_melody) {
        std::string s = "melody=[";
        for (size_t i = 0; i < e.melody.size(); ++i) {
            if (i) s += ",";
            const Yue2SheetMelodyNote & n = e.melody[i];
            s += "pitch=" + std::to_string(n.pitch) + ":track=" + std::to_string(n.track) +
                 ":dur_bin=" + std::to_string(n.duration_bin) + ":dur_steps=" + std::to_string(n.duration_steps);
        }
        s += "]";
        parts.push_back(s);
    }
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += "; ";
        out += parts[i];
    }
    return out;
}

// ═════════════════════════════════════════════════════════════════════════
// §6. Stitched-events JSON reader (structure only — 20_stitched_events.json
//     is produced by a different lane; this file only consumes it). Uses
//     yyjson (immutable/DOM reader).
// ═════════════════════════════════════════════════════════════════════════

namespace yue2_sheet_events_detail {

inline bool read_melody_note(yyjson_val * nv, Yue2SheetMelodyNote * out, std::string * err) {
    yyjson_val * pitch = yyjson_obj_get(nv, "pitch");
    yyjson_val * track = yyjson_obj_get(nv, "track");
    yyjson_val * dbin = yyjson_obj_get(nv, "duration_bin");
    yyjson_val * dsteps = yyjson_obj_get(nv, "duration_steps");
    if (!pitch || !track || !dbin || !dsteps) {
        if (err) *err = "melody note missing a required field";
        return false;
    }
    out->pitch = (int32_t) yyjson_get_int(pitch);
    out->track = (int32_t) yyjson_get_int(track);
    out->duration_bin = (int32_t) yyjson_get_int(dbin);
    out->duration_steps = (int32_t) yyjson_get_int(dsteps);
    yyjson_val * et = yyjson_obj_get(nv, "end_time");
    if (et) {
        out->has_end_time = true;
        out->end_time = yyjson_get_num(et);
    }
    return true;
}

inline bool read_event_values(yyjson_val * values, Yue2SheetEvent * ev, std::string * err) {
    if (!values || !yyjson_is_obj(values)) return true;  // absent "values" -> no fields (shouldn't happen)
    yyjson_val * ts = yyjson_obj_get(values, "timestamp");
    if (ts) { ev->has_timestamp = true; ev->timestamp = yyjson_get_num(ts); }
    yyjson_val * rh = yyjson_obj_get(values, "rhythm");
    if (rh && yyjson_is_obj(rh)) {
        ev->has_rhythm = true;
        yyjson_val * meter = yyjson_obj_get(rh, "meter");
        if (meter && yyjson_arr_size(meter) == 2) {
            ev->rhythm_has_meter = true;
            ev->rhythm_meter_num = (int32_t) yyjson_get_int(yyjson_arr_get(meter, 0));
            ev->rhythm_meter_den = (int32_t) yyjson_get_int(yyjson_arr_get(meter, 1));
        }
        yyjson_val * eighth = yyjson_obj_get(rh, "eighth_position");
        if (eighth) { ev->rhythm_has_eighth = true; ev->rhythm_eighth = (int32_t) yyjson_get_int(eighth); }
    }
    yyjson_val * st = yyjson_obj_get(values, "structure");
    if (st) { ev->has_structure = true; ev->structure = yyjson_get_str(st) ? yyjson_get_str(st) : ""; }
    yyjson_val * key = yyjson_obj_get(values, "key");
    if (key) { ev->has_key = true; ev->key = yyjson_get_str(key) ? yyjson_get_str(key) : ""; }
    yyjson_val * chord = yyjson_obj_get(values, "chord");
    if (chord) { ev->has_chord = true; ev->chord = yyjson_get_str(chord) ? yyjson_get_str(chord) : ""; }
    yyjson_val * melody = yyjson_obj_get(values, "melody");
    if (melody && yyjson_is_arr(melody)) {
        ev->has_melody = true;
        size_t idx, max;
        yyjson_val * nv;
        yyjson_arr_foreach(melody, idx, max, nv) {
            Yue2SheetMelodyNote note;
            if (!read_melody_note(nv, &note, err)) return false;
            ev->melody.push_back(note);
        }
    }
    return true;
}

}  // namespace yue2_sheet_events_detail

// Parses a JSON document whose root is an array of stitched-event objects
// (20_stitched_events.json's exact shape: result["events"] from
// stitched_window_events, generation_sheetsage2.py:515-560 / doc 21 §4.3).
// `text` is the whole file's bytes (caller reads the file); this function
// does no I/O itself.
inline bool yue2_sheet_parse_stitched_events(const std::string & text, std::vector<Yue2SheetEvent> * out,
                                              std::string * err) {
    yyjson_doc * doc = yyjson_read(text.data(), text.size(), 0);
    if (!doc) {
        if (err) *err = "failed to parse stitched-events JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!root || !yyjson_is_arr(root)) {
        if (err) *err = "stitched-events JSON root is not an array";
        yyjson_doc_free(doc);
        return false;
    }
    std::vector<Yue2SheetEvent> events;
    size_t idx, max;
    yyjson_val * ev_json;
    bool ok = true;
    yyjson_arr_foreach(root, idx, max, ev_json) {
        if (!yyjson_is_obj(ev_json)) continue;
        Yue2SheetEvent ev;
        yyjson_val * subbeat = yyjson_obj_get(ev_json, "subbeat");
        if (subbeat) ev.subbeat = (int32_t) yyjson_get_int(subbeat);
        yyjson_val * time = yyjson_obj_get(ev_json, "time");
        if (time) { ev.has_time = true; ev.time = yyjson_get_num(time); }
        yyjson_val * wi = yyjson_obj_get(ev_json, "window_index");
        if (wi) { ev.has_window_index = true; ev.window_index = (int32_t) yyjson_get_int(wi); }
        yyjson_val * ws = yyjson_obj_get(ev_json, "window_start");
        if (ws) { ev.has_window_start = true; ev.window_start = yyjson_get_num(ws); }
        yyjson_val * ss = yyjson_obj_get(ev_json, "source_subbeat");
        if (ss) { ev.has_source_subbeat = true; ev.source_subbeat = (int32_t) yyjson_get_int(ss); }
        yyjson_val * gs = yyjson_obj_get(ev_json, "global_subbeat");
        if (gs) { ev.has_global_subbeat = true; ev.global_subbeat = (int32_t) yyjson_get_int(gs); }
        if (!yue2_sheet_events_detail::read_event_values(yyjson_obj_get(ev_json, "values"), &ev, err)) {
            ok = false;
            break;
        }
        events.push_back(std::move(ev));
    }
    yyjson_doc_free(doc);
    if (!ok) return false;
    if (out) *out = std::move(events);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════
// §7. Python-repr-compatible double formatting. The exported LAB/TSV/TXT
//     files write every float via Python's plain str(x)/rows_text
//     (exports_sheetsage2.py:15-16), which for a float is exactly
//     repr(float) (CPython 3's str==repr for float). CPython's float repr
//     is dtoa mode 0 (shortest round-trip digit string) formatted by
//     Python/pystrtod.c's format_float_short with code 'r': fixed notation
//     unless decpt<=-4 or decpt>16, always at least one digit after the
//     point. std::to_chars's shortest round-trip digit generator (Ryu, in
//     the MSVC STL) produces the same digit string for the same double as
//     CPython's dtoa (both are correct "shortest round-trip decimal"
//     algorithms; when the shortest representation is unique — the case for
//     every value observed in the fixtures used to build this file — any
//     two correct implementations must agree). What is NOT reused from
//     to_chars is its own choice of fixed-vs-scientific format: verified by
//     probe (repr(0.0001) is "0.0001" in Python, but MSVC's to_chars with no
//     explicit format picks "1e-04") that to_chars's own formatting
//     heuristic differs from CPython's; only the scientific-mode DIGITS are
//     reused, and this function re-applies CPython's fixed/scientific
//     threshold and layout by hand.
// ═════════════════════════════════════════════════════════════════════════

namespace yue2_sheet_events_detail {

inline std::string build_fixed(bool neg, const std::string & digits, int decpt) {
    std::string out;
    int n = (int) digits.size();
    if (decpt <= 0) {
        out = "0.";
        out.append((size_t) (-decpt), '0');
        out += digits;
    } else if (decpt >= n) {
        out = digits;
        out.append((size_t) (decpt - n), '0');
        out += ".0";
    } else {
        out = digits.substr(0, (size_t) decpt) + "." + digits.substr((size_t) decpt);
    }
    if (neg) out = "-" + out;
    return out;
}

inline std::string build_scientific(bool neg, const std::string & digits, int exp10) {
    std::string out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    out += "e";
    out += (exp10 >= 0 ? "+" : "-");
    char buf[16];
    std::snprintf(buf, sizeof buf, "%02d", std::abs(exp10));
    out += buf;
    if (neg) out = "-" + out;
    return out;
}

}  // namespace yue2_sheet_events_detail

inline std::string yue2_sheet_python_repr_double(double v) {
    using namespace yue2_sheet_events_detail;
    bool neg = std::signbit(v);
    double av = std::fabs(v);

    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, av, std::chars_format::scientific);
    std::string s(buf, res.ptr);

    size_t i = 0;
    std::string digits;
    digits += s[i++];
    if (i < s.size() && s[i] == '.') {
        ++i;
        while (i < s.size() && s[i] != 'e') digits += s[i++];
    }
    // s[i] == 'e'
    ++i;
    bool eneg = (s[i] == '-');
    if (s[i] == '+' || s[i] == '-') ++i;
    int exp10 = std::stoi(s.substr(i));
    if (eneg) exp10 = -exp10;
    int decpt = exp10 + 1;

    bool use_exp = (decpt <= -4) || (decpt > 16);
    return use_exp ? build_scientific(neg, digits, exp10) : build_fixed(neg, digits, decpt);
}

// ═════════════════════════════════════════════════════════════════════════
// §8. Row/text writers. rows_text()/write_rows() (exports_sheetsage2.py:15-20):
//     "\t".join(str(x) for x in row) + "\n" per row, no trailing blank line
//     beyond the final row's own "\n"; atomic_write_text's newline="\n"
//     means LF only, no CRLF translation (io_sheetsage2.py:55-61).
// ═════════════════════════════════════════════════════════════════════════

struct Yue2SheetCell {
    enum class Kind { Int, Real, Str } kind;
    long long i = 0;
    double d = 0.0;
    std::string s;
    static Yue2SheetCell Int(long long v) { Yue2SheetCell c; c.kind = Kind::Int; c.i = v; return c; }
    static Yue2SheetCell Real(double v) { Yue2SheetCell c; c.kind = Kind::Real; c.d = v; return c; }
    static Yue2SheetCell Str(std::string v) { Yue2SheetCell c; c.kind = Kind::Str; c.s = std::move(v); return c; }
};
using Yue2SheetRow = std::vector<Yue2SheetCell>;

inline std::string yue2_sheet_cell_text(const Yue2SheetCell & c) {
    switch (c.kind) {
        case Yue2SheetCell::Kind::Int: return std::to_string(c.i);
        case Yue2SheetCell::Kind::Real: return yue2_sheet_python_repr_double(c.d);
        case Yue2SheetCell::Kind::Str: return c.s;
    }
    return "";
}

inline std::string yue2_sheet_rows_text(const std::vector<Yue2SheetRow> & rows) {
    std::string out;
    for (const Yue2SheetRow & row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i) out += '\t';
            out += yue2_sheet_cell_text(row[i]);
        }
        out += '\n';
    }
    return out;
}

// The one place a raw (non-rows_text) JSON value is written to a LAB file:
// rhythm_events.lab's second column is json.dumps(event.values.get("rhythm",
// {})) — compact form (separators ", "/": ", i.e. WITH spaces — that is
// json.dumps's default when indent=None, exports_sheetsage2.py:133-134) —
// key order "meter" then "eighth_position" (dict insertion order from
// _decode_field's rhythm case, tokenization_sheetsage2.py:485-492, which
// only ever inserts meter before eighth_position when both are present,
// since the grammar never allows the reverse — doc 21 §5.1's rhythm payload
// shape check).
inline std::string yue2_sheet_rhythm_json(const Yue2SheetEvent & e) {
    if (!e.rhythm_has_meter && !e.rhythm_has_eighth) return "{}";
    std::string out = "{";
    bool first = true;
    if (e.rhythm_has_meter) {
        out += "\"meter\": [" + std::to_string(e.rhythm_meter_num) + ", " + std::to_string(e.rhythm_meter_den) + "]";
        first = false;
    }
    if (e.rhythm_has_eighth) {
        if (!first) out += ", ";
        out += "\"eighth_position\": " + std::to_string(e.rhythm_eighth);
    }
    out += "}";
    return out;
}

// ═════════════════════════════════════════════════════════════════════════
// §9. Exports: events -> rows (exports_sheetsage2.py, doc 21 §6). All
//     functions here take an ALREADY-STITCHED event list (every event has
//     has_time == true) and the song's duration in seconds
//     (pipeline_sheetsage2.py:89, `len(audio)/SAMPLE_RATE` — the fixtures'
//     manifest.json "duration_seconds").
// ═════════════════════════════════════════════════════════════════════════

// interval_rows(events, field, duration) (exports_sheetsage2.py:23-27, doc
// 21 §6.5). `field` selects which of structure/key/chord to read.
struct Yue2SheetIntervalRow { double start; double end; std::string value; };

inline std::vector<Yue2SheetIntervalRow> yue2_sheet_interval_rows(const std::vector<Yue2SheetEvent> & events,
                                                                   Yue2SheetField field, double duration) {
    std::vector<Yue2SheetIntervalRow> rows;
    for (const Yue2SheetEvent & e : events) {
        const std::string * v = nullptr;
        if (field == Yue2SheetField::Structure && e.has_structure) v = &e.structure;
        else if (field == Yue2SheetField::Key && e.has_key) v = &e.key;
        else if (field == Yue2SheetField::Chord && e.has_chord) v = &e.chord;
        if (v) rows.push_back({e.time, 0.0, *v});
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i].end = (i + 1 < rows.size()) ? rows[i + 1].start : duration;
    }
    std::vector<Yue2SheetIntervalRow> kept;
    for (auto & r : rows) if (r.end > r.start) kept.push_back(r);
    return kept;
}

// rhythm_rows(events) (exports_sheetsage2.py:30-44, doc 21 §6.2). Exact
// integer arithmetic per doc 21 §6.6 — no floats anywhere in the grid check.
// Returns false with *err set on the reference's two ValueErrors (off-grid /
// outside-meter), with the reference's own message text.
struct Yue2SheetBeatRow { double time; int32_t beat_id; int32_t numerator; int32_t denominator; };

inline bool yue2_sheet_rhythm_rows(const std::vector<Yue2SheetEvent> & events, std::vector<Yue2SheetBeatRow> * out,
                                    std::string * err) {
    std::vector<Yue2SheetBeatRow> rows;
    bool have_meter = false;
    int32_t meter_num = 0, meter_den = 0;
    for (const Yue2SheetEvent & e : events) {
        if (e.has_rhythm && e.rhythm_has_meter) {
            have_meter = true;
            meter_num = e.rhythm_meter_num;
            meter_den = e.rhythm_meter_den;
        }
        if (e.has_rhythm && e.rhythm_has_eighth && have_meter) {
            long long numerator = (long long) e.rhythm_eighth * (long long) meter_den;
            if (numerator % 8 != 0) {
                if (err) {
                    *err = "Eighth position " + std::to_string(e.rhythm_eighth) + " is off the " +
                           std::to_string(meter_num) + "/" + std::to_string(meter_den) + " beat grid";
                }
                return false;
            }
            long long position = numerator / 8;
            if (!(position >= 0 && position < meter_num)) {
                if (err) {
                    *err = "Eighth position " + std::to_string(e.rhythm_eighth) + " is outside meter (" +
                           std::to_string(meter_num) + ", " + std::to_string(meter_den) + ")";
                }
                return false;
            }
            rows.push_back({e.time, (int32_t) position + 1, meter_num, meter_den});
        }
    }
    if (out) *out = std::move(rows);
    return true;
}

// The abc_beats padding tail (exports_sheetsage2.py:141-152, doc 21 §6.2),
// run against the RAW (unclipped) melody notes' end times for `end`.
// Requires len(beats)>=2 and a positive period, else false with the
// reference's own error text.
inline bool yue2_sheet_pad_beats(const std::vector<Yue2SheetBeatRow> & beats, double duration,
                                  const std::vector<std::array<double, 2>> & raw_note_spans /* [start,end] */,
                                  std::vector<Yue2SheetBeatRow> * out, std::string * err) {
    if (beats.size() < 2) {
        if (err) *err = "At least two decoded beats are required for ABC";
        return false;
    }
    std::vector<Yue2SheetBeatRow> abc_beats(beats);
    size_t tail_n = std::min<size_t>(9, beats.size());
    std::vector<double> tail_times;
    for (size_t i = beats.size() - tail_n; i < beats.size(); ++i) tail_times.push_back(beats[i].time);
    std::vector<double> diffs;
    for (size_t i = 1; i < tail_times.size(); ++i) diffs.push_back(tail_times[i] - tail_times[i - 1]);
    double period = yue2_sheet_np_median(diffs);
    if (!(period > 0.0)) {
        if (err) *err = "Decoded beats must increase in time";
        return false;
    }
    double end = duration;
    for (const auto & span : raw_note_spans) end = std::max(end, span[1]);
    while (abc_beats.back().time < end - 1e-6) {
        Yue2SheetBeatRow prev = abc_beats.back();
        Yue2SheetBeatRow next;
        next.time = prev.time + period;
        next.beat_id = (prev.beat_id % prev.numerator) + 1;
        next.numerator = prev.numerator;
        next.denominator = prev.denominator;
        abc_beats.push_back(next);
    }
    if (out) *out = std::move(abc_beats);
    return true;
}

// Clip interval rows to the abc_beats time domain (exports_sheetsage2.py:
// 156-161, doc 21 §6.5's "only for the ABC path" clip) — used for
// notation/song_{chords,keys,structures}.txt.
inline std::vector<Yue2SheetIntervalRow> yue2_sheet_clip_to_beats(const std::vector<Yue2SheetIntervalRow> & rows,
                                                                   const std::vector<Yue2SheetBeatRow> & abc_beats) {
    double lo = abc_beats.front().time, hi = abc_beats.back().time;
    std::vector<Yue2SheetIntervalRow> out;
    for (const auto & r : rows) {
        if (r.end > lo && r.start < hi) out.push_back({std::max(lo, r.start), std::min(hi, r.end), r.value});
    }
    return out;
}

// Raw melody notes (§6.1, exports_sheetsage2.py:94-102): [start,end,pitch,track],
// polyphonic, sorted lexicographically. Requires every melody note on a
// stitched event to carry end_time (guaranteed by the stitching lane).
struct Yue2SheetRawNote { double start; double end; int32_t pitch; int32_t track; };

inline std::vector<Yue2SheetRawNote> yue2_sheet_raw_melody_notes(const std::vector<Yue2SheetEvent> & events,
                                                                  double duration) {
    std::vector<Yue2SheetRawNote> notes;
    for (const Yue2SheetEvent & e : events) {
        if (!e.has_melody) continue;
        double start = e.time;
        for (const Yue2SheetMelodyNote & n : e.melody) {
            double end = std::min(duration, n.has_end_time ? n.end_time : start);
            if (end > start) notes.push_back({start, end, n.pitch, n.track});
        }
    }
    std::sort(notes.begin(), notes.end(), [](const Yue2SheetRawNote & a, const Yue2SheetRawNote & b) {
        if (a.start != b.start) return a.start < b.start;
        if (a.end != b.end) return a.end < b.end;
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        return a.track < b.track;
    });
    return notes;
}

// notation_notes(notes) (§6.3, exports_sheetsage2.py:59-70): per-track
// monophonic clip (end trimmed to the next onset on the SAME track), drop
// degenerate results, then re-sort the combined (both tracks) list.
inline std::vector<Yue2SheetRawNote> yue2_sheet_notation_notes(const std::vector<Yue2SheetRawNote> & notes) {
    std::vector<Yue2SheetRawNote> result;
    for (int track = 0; track <= 1; ++track) {
        std::vector<Yue2SheetRawNote> ordered;
        for (const auto & n : notes) if (n.track == track) ordered.push_back(n);
        std::sort(ordered.begin(), ordered.end(), [](const Yue2SheetRawNote & a, const Yue2SheetRawNote & b) {
            if (a.start != b.start) return a.start < b.start;
            if (a.pitch != b.pitch) return a.pitch < b.pitch;
            return a.end < b.end;
        });
        for (size_t i = 0; i < ordered.size(); ++i) {
            Yue2SheetRawNote note = ordered[i];
            if (i + 1 < ordered.size() && note.end > ordered[i + 1].start + 1e-6) {
                note.end = ordered[i + 1].start;
            }
            if (note.end > note.start + 1e-6) result.push_back(note);
        }
    }
    std::sort(result.begin(), result.end(), [](const Yue2SheetRawNote & a, const Yue2SheetRawNote & b) {
        if (a.start != b.start) return a.start < b.start;
        if (a.end != b.end) return a.end < b.end;
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        return a.track < b.track;
    });
    return result;
}

// ═════════════════════════════════════════════════════════════════════════
// §10. notation/song_melody.mid — a real Standard MIDI File byte-identical
//     to pretty_midi 0.2.10's write() output for
//     `PrettyMIDI(resolution=960)` with two named instruments ("Vocal"
//     track 0, "Ins" track 1, program 0, velocity 100 — _midi(), non-paper
//     path, exports_sheetsage2.py:47-56 / doc 21 §6.4), containing ONLY
//     these note-on/note-off events (no pitch bends, no control changes —
//     _midi() never emits either). SheetSage2 itself never touches MIDI
//     bytes; the byte format below is pretty_midi's write() delegating to
//     mido's MidiFile.save(), read directly from both packages' source
//     (file header comment) since neither is documented in the SheetSage2
//     pin. `_midi()` ALWAYS creates both instruments, even with zero notes
//     (doc 21 §6.4's "always both, even if empty" note) — callers must pass
//     both note lists, empty or not.
//
// tick(seconds): pretty_midi.PrettyMIDI.time_to_tick() (pretty_midi.py:
// 1012-1044). For a freshly-constructed (never-read-from-file) PrettyMIDI
// object, `self.__tick_to_time` stays the 1-element list `[0]` set in
// __init__ (pretty_midi.py:115) — write() never calls _update_tick_to_time,
// and every call to time_to_tick() (both for note starts/ends in write())
// therefore always takes the "tick == len(__tick_to_time)" extrapolation
// branch (pretty_midi.py:1029-1037): `tick = 0 + (time-0)/final_tick_scale`,
// `return int(round(tick))`. Python's builtin round() on a float is
// round-half-to-even (banker's rounding) — NOT round-half-away-from-zero.
// *** DISCREPANCY vs 21-sheetsage2-symbolic-pin.md §6.4, which describes
// this as "nearest tick, exact .5 ties resolved toward the higher tick":
// that is not what the code does. The doc's own worked numbers (237 ticks
// for 0.1234375s, etc.) never land on an exact .5 tie, so nothing in doc 21
// actually depends on the tie-break rule it states. Per this task's
// standing rule (Python wins over the pin doc on disagreement), this file
// implements round-half-to-even via std::nearbyint() under the process's
// default FE_TONEAREST rounding mode, which IS IEEE-754 round-to-nearest,
// ties-to-even — the same rule as Python's round(). ***
// ═════════════════════════════════════════════════════════════════════════

// 60.0/(120.0*960.0) in double, computed in the same operation order as
// pretty_midi's `self._tick_scales = [(0, 60.0/(initial_tempo*self.resolution))]`
// (pretty_midi.py:113, initial_tempo=120.0 default, resolution=960).
inline double yue2_sheet_midi_tick_scale() { return 60.0 / (120.0 * 960.0); }

inline int64_t yue2_sheet_seconds_to_tick(double seconds) {
    double tick = seconds / yue2_sheet_midi_tick_scale();
    return (int64_t) std::nearbyint(tick);  // FE_TONEAREST default == round-half-to-even, matching Python round()
}

namespace yue2_sheet_events_detail {

inline void midi_write_var_len(std::vector<uint8_t> & out, uint32_t value) {
    uint8_t bytes[5];
    int n = 0;
    do {
        bytes[n++] = (uint8_t) (value & 0x7f);
        value >>= 7;
    } while (value);
    for (int i = n - 1; i >= 0; --i) {
        uint8_t b = bytes[i];
        if (i != 0) b |= 0x80;
        out.push_back(b);
    }
}

inline void midi_write_chunk(std::vector<uint8_t> & out, const char name[4], const std::vector<uint8_t> & data) {
    out.insert(out.end(), name, name + 4);
    uint32_t len = (uint32_t) data.size();
    out.push_back((uint8_t) (len >> 24));
    out.push_back((uint8_t) (len >> 16));
    out.push_back((uint8_t) (len >> 8));
    out.push_back((uint8_t) (len));
    out.insert(out.end(), data.begin(), data.end());
}

inline void midi_write_meta(std::vector<uint8_t> & out, uint8_t type_byte, const std::vector<uint8_t> & data) {
    out.push_back(0xff);
    out.push_back(type_byte);
    midi_write_var_len(out, (uint32_t) data.size());
    out.insert(out.end(), data.begin(), data.end());
}

// One raw event queued for a track, absolute tick, with the (tick,
// secondary_weight) ordering key mido's event_compare implements
// (pretty_midi.py:1286-1329). `has_secondary=false` mirrors a message type
// absent from `secondary_sort` (e.g. track_name): such an event compares
// EQUAL at matching ticks, so a stable sort preserves its insertion-order
// position relative to anything else at the same tick, exactly like
// Python's `event1.time - event2.time` fallback (which returns 0 whenever
// either type is missing from the dict, not just when both are present but
// unequal).
struct MidiRawEvent {
    int64_t tick;
    bool has_secondary;
    int64_t weight;
    size_t insertion_index;
    std::vector<uint8_t> bytes;   // full status+data bytes (no delta, no running-status omission yet)
    uint8_t status_byte;          // 0 for meta events (never running-status-compressed)
    bool is_meta;
};

inline void midi_stable_sort(std::vector<MidiRawEvent> & events) {
    std::stable_sort(events.begin(), events.end(), [](const MidiRawEvent & a, const MidiRawEvent & b) {
        if (a.tick != b.tick) return a.tick < b.tick;
        if (a.has_secondary && b.has_secondary) return a.weight < b.weight;
        return false;  // matches event_compare's fallback of 0 (equal) -> stable order wins
    });
}

// Serializes one already-ordered event list (absolute ticks) into an MTrk
// chunk, converting to delta times and applying running-status compression
// exactly as mido's write_track() does (midifiles.py:238-277) — a channel
// message (status < 0xf0, which every message this file emits satisfies)
// omits its status byte when it matches the immediately preceding channel
// message's status byte; any meta message resets that state.
inline std::vector<uint8_t> midi_serialize_track(std::vector<MidiRawEvent> events) {
    // Append the canonical end-of-track meta event: delta 1 tick after the
    // last event (or tick 0 if the track has no other events), matching
    // pretty_midi's own `track[-1].time + 1` (pretty_midi.py:1377-1378,
    // 1439-1440) followed by mido's fix_end_of_track() re-wrap
    // (midifiles/tracks.py:84-103), which is a no-op on the byte value here
    // since there is only ever one end_of_track message going in.
    int64_t last_tick = events.empty() ? 0 : events.back().tick;
    // events is already tick-sorted by construction (midi_stable_sort ran
    // before this call); the max is therefore just the last element's tick
    // EXCEPT ties can reorder within a tick, so take a true max defensively.
    for (const auto & e : events) last_tick = std::max(last_tick, e.tick);
    MidiRawEvent eot;
    eot.tick = last_tick + 1;
    eot.has_secondary = true;
    eot.weight = 11LL * 256 * 256;
    eot.insertion_index = events.size();
    eot.is_meta = true;
    eot.status_byte = 0;
    midi_write_meta(eot.bytes, 0x2f, {});
    events.push_back(eot);

    std::vector<uint8_t> data;
    int64_t running_tick = 0;
    int running_status = -1;  // -1 == none
    for (const MidiRawEvent & e : events) {
        int64_t delta = e.tick - running_tick;
        running_tick = e.tick;
        midi_write_var_len(data, (uint32_t) delta);
        if (e.is_meta) {
            data.insert(data.end(), e.bytes.begin(), e.bytes.end());
            running_status = -1;
        } else if (running_status == e.status_byte) {
            // Omit the status byte (skip the first byte of e.bytes).
            data.insert(data.end(), e.bytes.begin() + 1, e.bytes.end());
        } else {
            data.insert(data.end(), e.bytes.begin(), e.bytes.end());
            running_status = e.status_byte;
        }
    }
    std::vector<uint8_t> chunk;
    midi_write_chunk(chunk, "MTrk", data);
    return chunk;
}

}  // namespace yue2_sheet_events_detail

// Builds a two-instrument-track (plus timing track) Standard MIDI File, byte
// for byte matching pretty_midi's write() for `_midi(clean)`
// (exports_sheetsage2.py:47-56 non-paper path). `vocal_notes`/`ins_notes`
// are this MIDI's two tracks in that fixed order (track 0 "Vocal", track 1
// "Ins"); either or both may be empty.
inline std::vector<uint8_t> yue2_sheet_write_midi(const std::vector<Yue2SheetRawNote> & vocal_notes,
                                                   const std::vector<Yue2SheetRawNote> & ins_notes) {
    using namespace yue2_sheet_events_detail;

    std::vector<uint8_t> file;
    // MThd: format=1, ntrks=3 (timing + Vocal + Ins), division=960.
    {
        std::vector<uint8_t> hdr;
        auto put16 = [&](int v) { hdr.push_back((uint8_t) (v >> 8)); hdr.push_back((uint8_t) v); };
        put16(1);
        put16(3);
        put16(960);
        midi_write_chunk(file, "MThd", hdr);
    }

    // Timing track: set_tempo(500000) + time_signature(4,4,24,8), both at
    // tick 0, then end_of_track (pretty_midi.py:1332-1379 for a
    // PrettyMIDI with no time_signature_changes and one tick_scale).
    {
        std::vector<MidiRawEvent> events;
        MidiRawEvent tempo;
        tempo.tick = 0;
        tempo.has_secondary = true;
        tempo.weight = 1LL * 256 * 256;
        tempo.insertion_index = 0;
        tempo.is_meta = true;
        tempo.status_byte = 0;
        midi_write_meta(tempo.bytes, 0x51, {0x07, 0xa1, 0x20});  // 500000 = 0x07A120
        events.push_back(tempo);

        MidiRawEvent ts;
        ts.tick = 0;
        ts.has_secondary = true;
        ts.weight = 2LL * 256 * 256;
        ts.insertion_index = 1;
        ts.is_meta = true;
        ts.status_byte = 0;
        midi_write_meta(ts.bytes, 0x58, {4, 2, 24, 8});  // 4/4 (denominator exponent log2(4)=2), 24 clocks/click, 8
        events.push_back(ts);

        midi_stable_sort(events);
        std::vector<uint8_t> chunk = midi_serialize_track(events);
        file.insert(file.end(), chunk.begin(), chunk.end());
    }

    // Channel assignment (pretty_midi.py:1380-1397): channels 0..15 minus 9,
    // in order; instrument n gets channels[n % 15]. Track 0 (Vocal) -> ch 0,
    // track 1 (Ins) -> ch 1 (neither is the drum channel).
    const std::vector<Yue2SheetRawNote> * tracks[2] = {&vocal_notes, &ins_notes};
    const char * names[2] = {"Vocal", "Ins"};
    const int channels[2] = {0, 1};
    for (int t = 0; t < 2; ++t) {
        std::vector<MidiRawEvent> events;
        size_t ins_idx = 0;

        MidiRawEvent name_ev;
        name_ev.tick = 0;
        name_ev.has_secondary = false;  // track_name has no secondary_sort entry
        name_ev.weight = 0;
        name_ev.insertion_index = ins_idx++;
        name_ev.is_meta = true;
        name_ev.status_byte = 0;
        std::vector<uint8_t> name_bytes(names[t], names[t] + std::strlen(names[t]));
        midi_write_meta(name_ev.bytes, 0x03, name_bytes);
        events.push_back(name_ev);

        MidiRawEvent prog;
        prog.tick = 0;
        prog.has_secondary = true;
        prog.weight = 6LL * 256 * 256;
        prog.insertion_index = ins_idx++;
        prog.is_meta = false;
        prog.status_byte = (uint8_t) (0xc0 | channels[t]);
        prog.bytes = {prog.status_byte, 0x00};  // program 0
        events.push_back(prog);

        for (const Yue2SheetRawNote & note : *tracks[t]) {
            int64_t on_tick = yue2_sheet_seconds_to_tick(note.start);
            int64_t off_tick = yue2_sheet_seconds_to_tick(note.end);
            uint8_t status = (uint8_t) (0x90 | channels[t]);

            MidiRawEvent on_ev;
            on_ev.tick = on_tick;
            on_ev.has_secondary = true;
            on_ev.weight = 10LL * 256 * 256 + (int64_t) note.pitch * 256 + 100;  // velocity always 100
            on_ev.insertion_index = ins_idx++;
            on_ev.is_meta = false;
            on_ev.status_byte = status;
            on_ev.bytes = {status, (uint8_t) note.pitch, 100};
            events.push_back(on_ev);

            MidiRawEvent off_ev;
            off_ev.tick = off_tick;
            off_ev.has_secondary = true;
            off_ev.weight = 10LL * 256 * 256 + (int64_t) note.pitch * 256 + 0;  // note_on velocity 0 == note off
            off_ev.insertion_index = ins_idx++;
            off_ev.is_meta = false;
            off_ev.status_byte = status;
            off_ev.bytes = {status, (uint8_t) note.pitch, 0};
            events.push_back(off_ev);
        }

        midi_stable_sort(events);
        std::vector<uint8_t> chunk = midi_serialize_track(events);
        file.insert(file.end(), chunk.begin(), chunk.end());
    }

    return file;
}

// ═════════════════════════════════════════════════════════════════════════
// §11. Events JSON writer — structural comparison only (per-window test
//     against 12_events.json). Uses yyjson's mutable-document builder; type
//     mapping mirrors Python's json.dumps: subbeat/token ids/pitch/track/
//     duration_bin/duration_steps/eighth_position/meter entries as JSON
//     integers, timestamp as a JSON real, structure/key/chord as JSON
//     strings. Field/key order does not matter for a structural compare.
// ═════════════════════════════════════════════════════════════════════════

inline yyjson_mut_val * yue2_sheet_event_to_json(yyjson_mut_doc * doc, const Yue2SheetEvent & e) {
    yyjson_mut_val * obj = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_int(doc, obj, "subbeat", e.subbeat);

    yyjson_mut_val * tbf = yyjson_mut_obj(doc);
    static const char * field_names[6] = {"timestamp", "rhythm", "structure", "key", "chord", "melody"};
    for (int fi = 0; fi < 6; ++fi) {
        if (e.tokens_by_field[fi].empty()) continue;
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        for (int32_t tok : e.tokens_by_field[fi]) yyjson_mut_arr_add_int(doc, arr, tok);
        yyjson_mut_obj_add_val(doc, tbf, field_names[fi], arr);
    }
    yyjson_mut_obj_add_val(doc, obj, "tokens_by_field", tbf);

    yyjson_mut_val * values = yyjson_mut_obj(doc);
    if (e.has_timestamp) yyjson_mut_obj_add_real(doc, values, "timestamp", e.timestamp);
    if (e.has_rhythm) {
        yyjson_mut_val * rh = yyjson_mut_obj(doc);
        if (e.rhythm_has_meter) {
            yyjson_mut_val * meter = yyjson_mut_arr(doc);
            yyjson_mut_arr_add_int(doc, meter, e.rhythm_meter_num);
            yyjson_mut_arr_add_int(doc, meter, e.rhythm_meter_den);
            yyjson_mut_obj_add_val(doc, rh, "meter", meter);
        }
        if (e.rhythm_has_eighth) yyjson_mut_obj_add_int(doc, rh, "eighth_position", e.rhythm_eighth);
        yyjson_mut_obj_add_val(doc, values, "rhythm", rh);
    }
    if (e.has_structure) yyjson_mut_obj_add_strcpy(doc, values, "structure", e.structure.c_str());
    if (e.has_key) yyjson_mut_obj_add_strcpy(doc, values, "key", e.key.c_str());
    if (e.has_chord) yyjson_mut_obj_add_strcpy(doc, values, "chord", e.chord.c_str());
    if (e.has_melody) {
        yyjson_mut_val * arr = yyjson_mut_arr(doc);
        for (const Yue2SheetMelodyNote & n : e.melody) {
            yyjson_mut_val * nv = yyjson_mut_obj(doc);
            yyjson_mut_obj_add_int(doc, nv, "pitch", n.pitch);
            yyjson_mut_obj_add_int(doc, nv, "track", n.track);
            yyjson_mut_obj_add_int(doc, nv, "duration_bin", n.duration_bin);
            yyjson_mut_obj_add_int(doc, nv, "duration_steps", n.duration_steps);
            if (n.has_end_time) yyjson_mut_obj_add_real(doc, nv, "end_time", n.end_time);
            yyjson_mut_arr_add_val(arr, nv);
        }
        yyjson_mut_obj_add_val(doc, values, "melody", arr);
    }
    yyjson_mut_obj_add_val(doc, obj, "values", values);
    return obj;
}

inline std::string yue2_sheet_decoded_to_json_text(const Yue2SheetDecoded & decoded) {
    yyjson_mut_doc * doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "schema_version", decoded.schema_version.c_str());
    yyjson_mut_val * prompts = yyjson_mut_arr(doc);
    for (const std::string & p : decoded.prompts) yyjson_mut_arr_add_strcpy(doc, prompts, p.c_str());
    yyjson_mut_obj_add_val(doc, root, "prompts", prompts);
    yyjson_mut_val * events = yyjson_mut_arr(doc);
    for (const Yue2SheetEvent & e : decoded.events) yyjson_mut_arr_add_val(events, yue2_sheet_event_to_json(doc, e));
    yyjson_mut_obj_add_val(doc, root, "events", events);
    yyjson_mut_obj_add_bool(doc, root, "has_eos", decoded.has_eos);
    const char * s = yyjson_mut_write(doc, 0, nullptr);
    std::string out = s ? s : "";
    if (s) free((void *) s);
    yyjson_mut_doc_free(doc);
    return out;
}

// ═════════════════════════════════════════════════════════════════════════
// §12. Top-level export orchestrator. Builds every file this lane owns from
//     21_exports's relname list (doc 19/22): beat.lab, downbeat.lab,
//     chord.lab, key.lab, structure.lab, melody_full.lab, melody_vocal.lab,
//     melody_instrumental.lab, rhythm_events.lab, events.tsv,
//     notation/song_beats.txt, notation/song_chords.txt,
//     notation/song_keys.txt, notation/song_structures.txt,
//     notation/song_melody.mid.
//
// NOT IMPLEMENTED: playback.json and score.abc/score.browser.abc.
// score.abc is explicitly the other (notation/§7) lane's output. playback.json
// (midi_sheetsage2.py:build_playback) needs (a) mir_eval.chord.encode() to
// turn a chord LABEL back into pitch classes (chord_pitches(), no Harte
// chord-grammar parser exists anywhere in this codebase yet) and, whenever
// ABC succeeded, (b) the notation lane's `Score` object (score.beats for
// downbeat cut points, score.measures for the measure map) — both are
// genuinely outside doc 21 §5-6's scope. This function only returns the 15
// files above; a caller that also wants playback.json needs a follow-up
// lane for (a)+(b). See this file's test report for exactly which fixtures
// this was and wasn't checked against.
// ═════════════════════════════════════════════════════════════════════════

struct Yue2SheetExportFile { std::string relpath; std::vector<uint8_t> bytes; };

inline bool yue2_sheet_write_exports(const std::vector<Yue2SheetEvent> & events, double duration,
                                      std::vector<Yue2SheetExportFile> * out, std::string * err) {
    std::vector<Yue2SheetExportFile> files;
    auto add_text = [&](const char * relpath, const std::string & text) {
        Yue2SheetExportFile f;
        f.relpath = relpath;
        f.bytes.assign(text.begin(), text.end());
        files.push_back(std::move(f));
    };

    // ---- Always computed, independent of the beat/ABC try-block (§ above events.tsv) ----
    std::vector<Yue2SheetRawNote> raw_notes = yue2_sheet_raw_melody_notes(events, duration);

    std::vector<Yue2SheetRow> events_tsv_rows;
    events_tsv_rows.push_back({Yue2SheetCell::Str("time"), Yue2SheetCell::Str("global_subbeat"), Yue2SheetCell::Str("fields")});
    for (const Yue2SheetEvent & e : events) {
        events_tsv_rows.push_back({Yue2SheetCell::Real(e.time), Yue2SheetCell::Int(e.global_subbeat),
                                    Yue2SheetCell::Str(yue2_sheet_field_text(e))});
    }
    add_text("events.tsv", yue2_sheet_rows_text(events_tsv_rows));

    std::vector<Yue2SheetRow> melody_full_rows;
    for (const auto & n : raw_notes) {
        melody_full_rows.push_back({Yue2SheetCell::Real(n.start), Yue2SheetCell::Real(n.end),
                                     Yue2SheetCell::Int(n.pitch), Yue2SheetCell::Int(n.track)});
    }
    add_text("melody_full.lab", yue2_sheet_rows_text(melody_full_rows));
    for (int track = 0; track <= 1; ++track) {
        std::vector<Yue2SheetRow> rows;
        for (const auto & n : raw_notes) {
            if (n.track != track) continue;
            rows.push_back({Yue2SheetCell::Real(n.start), Yue2SheetCell::Real(n.end), Yue2SheetCell::Int(n.pitch)});
        }
        add_text(track == 0 ? "melody_vocal.lab" : "melody_instrumental.lab", yue2_sheet_rows_text(rows));
    }

    for (auto field_pair : {std::make_pair(Yue2SheetField::Chord, "chord.lab"),
                             std::make_pair(Yue2SheetField::Key, "key.lab"),
                             std::make_pair(Yue2SheetField::Structure, "structure.lab")}) {
        std::vector<Yue2SheetIntervalRow> rows = yue2_sheet_interval_rows(events, field_pair.first, duration);
        std::vector<Yue2SheetRow> text_rows;
        for (auto & r : rows) text_rows.push_back({Yue2SheetCell::Real(r.start), Yue2SheetCell::Real(r.end), Yue2SheetCell::Str(r.value)});
        add_text(field_pair.second, yue2_sheet_rows_text(text_rows));
    }

    std::vector<Yue2SheetRow> rhythm_events_rows;
    for (const Yue2SheetEvent & e : events) {
        if (!(e.has_rhythm || e.has_timestamp)) continue;
        rhythm_events_rows.push_back({Yue2SheetCell::Real(e.time), Yue2SheetCell::Str(yue2_sheet_rhythm_json(e))});
    }
    add_text("rhythm_events.lab", yue2_sheet_rows_text(rhythm_events_rows));

    // ---- The beat/notation try-block (exports_sheetsage2.py:136-177 up to,
    //      but not including, generate_abc_from_data — §7, not this lane) ----
    std::vector<Yue2SheetBeatRow> beats;
    if (!yue2_sheet_rhythm_rows(events, &beats, err)) {
        if (out) *out = std::move(files);
        return true;  // matches the reference: an earlier-stage failure here
                       // still yields the events.tsv/melody/*.lab/*.lab files
                       // above; it just means abc_error would be set and the
                       // beat/notation files are skipped. Not a hard error
                       // for THIS function's contract.
    }
    std::vector<Yue2SheetRow> beat_rows, downbeat_rows;
    for (auto & r : beats) {
        beat_rows.push_back({Yue2SheetCell::Real(r.time), Yue2SheetCell::Int(r.beat_id), Yue2SheetCell::Int(r.numerator), Yue2SheetCell::Int(r.denominator)});
        if (r.beat_id == 1) downbeat_rows.push_back({Yue2SheetCell::Real(r.time)});
    }
    add_text("beat.lab", yue2_sheet_rows_text(beat_rows));
    add_text("downbeat.lab", yue2_sheet_rows_text(downbeat_rows));

    std::vector<std::array<double, 2>> raw_spans;
    for (auto & n : raw_notes) raw_spans.push_back({n.start, n.end});
    std::vector<Yue2SheetBeatRow> abc_beats;
    std::string pad_err;
    if (!yue2_sheet_pad_beats(beats, duration, raw_spans, &abc_beats, &pad_err)) {
        if (out) *out = std::move(files);
        return true;  // same rationale as above: reference-side abc_error, not a hard failure here
    }
    std::vector<Yue2SheetRow> song_beats_rows;
    for (auto & r : abc_beats) {
        song_beats_rows.push_back({Yue2SheetCell::Real(r.time), Yue2SheetCell::Int(r.beat_id), Yue2SheetCell::Int(r.numerator), Yue2SheetCell::Int(r.denominator)});
    }
    add_text("notation/song_beats.txt", yue2_sheet_rows_text(song_beats_rows));

    for (auto field_pair : {std::make_pair(Yue2SheetField::Chord, "notation/song_chords.txt"),
                             std::make_pair(Yue2SheetField::Key, "notation/song_keys.txt"),
                             std::make_pair(Yue2SheetField::Structure, "notation/song_structures.txt")}) {
        auto rows = yue2_sheet_clip_to_beats(yue2_sheet_interval_rows(events, field_pair.first, duration), abc_beats);
        std::vector<Yue2SheetRow> text_rows;
        for (auto & r : rows) text_rows.push_back({Yue2SheetCell::Real(r.start), Yue2SheetCell::Real(r.end), Yue2SheetCell::Str(r.value)});
        add_text(field_pair.second, yue2_sheet_rows_text(text_rows));
    }

    if (yue2_sheet_interval_rows(events, Yue2SheetField::Key, duration).empty()) {
        // "No key was decoded; cannot construct a keyed ABC score" — matches
        // the reference, but notation/song_melody.mid is written BEFORE this
        // check in the reference (exports_sheetsage2.py:162 vs :164-166), so
        // this function differs from the reference by stopping one file
        // earlier here; no fixture in this lane's test set exercises this
        // branch (every fixture has at least one key interval).
        if (out) *out = std::move(files);
        return true;
    }

    std::vector<Yue2SheetRawNote> clean = yue2_sheet_notation_notes(raw_notes);
    std::vector<Yue2SheetRawNote> vocal, ins;
    for (auto & n : clean) (n.track == 0 ? vocal : ins).push_back(n);
    std::vector<uint8_t> midi = yue2_sheet_write_midi(vocal, ins);
    Yue2SheetExportFile mid_file;
    mid_file.relpath = "notation/song_melody.mid";
    mid_file.bytes = std::move(midi);
    files.push_back(std::move(mid_file));

    if (out) *out = std::move(files);
    return true;
}
