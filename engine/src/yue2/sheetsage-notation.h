#pragma once
// yue2/sheetsage-notation.h — SheetSage2 notation stage: exported
// beats/chords/keys/structures/melody rows -> validated two-voice ABC text.
//
// HOT-Step file (no acestep.cpp analog). Pure C++, no tensors, no GGML: this
// is the very last leg of the SheetSage2 pipeline (doc 19 "Pipeline" bullet),
// the part that turns already-decoded, already-exported musical data into the
// ABC text the YuE2 AR/NAR trainers splice into a `cot=full` prefix. It is a
// direct, function-for-function port of `notation_sheetsage2.py`'s in-memory
// entry point:
//
//   generate_abc_from_data(melody_midi, beats, chords, keys, structures)
//     = build_rebuilt_abc_score_from_data(...)  [_parse_* + _assemble_abc_score]
//     -> score_to_abc(score)
//
// SCOPE. This file starts from the *export-stage output*, not from the
// stitched decode events: `melody_midi` = the fixed Vocal/Ins MIDI bytes
// produced by `exports_sheetsage2.py`'s `_midi(notation_notes(notes))`;
// `beats` = the padded `abc_beats` rows; `chords`/`keys`/`structures` = the
// beat-domain-clipped interval rows. Building those five inputs from the
// stitched event list (exports_sheetsage2.py §6 of the pin doc) is a
// different lane's job (sheetsage-pipeline.h, doc 19 phase 5) — this file
// only implements notation_sheetsage2.py's own logic (pin doc §7), which is
// exactly what `generate_abc_from_data` receives and exactly what the test
// harness in engine/tests/sheetsage/test_notation.cpp constructs from the
// `21_exports/notation/song_{beats,chords,keys,structures}.txt` +
// `song_melody.mid` fixture files — confirmed byte-for-byte against the real
// `generate_abc_from_data()` by feeding those same five fixture inputs
// through the actual Python (see the workflow's verification note; short-a
// and short-b reproduce `22_abc.txt` exactly, long-a and long-b reproduce
// `24_abc_error.txt` exactly, straight from the oracle).
//
// AUTHORITY: docs/plans/yue2/21-sheetsage2-symbolic-pin.md §6-§8 (the
// exports-stage output shapes and the notation algorithm), cross-checked
// directly against the shipped module at
// K:\yue2\.cache\huggingface-sheetsage\modules\transformers_modules\SheetSage2\
// {notation,exports,midi}_sheetsage2.py (read in full for this port — every
// function below cites its exact source line range). Where this file and the
// pin doc disagree, the doc wins; where the doc and the Python disagree, the
// Python (as actually read here) wins — flagged in the workflow's return
// value per file-map instructions, not silently "corrected" here.
//
// ERROR CONVENTION. Matches the rest of this lane's siblings
// (sheetsage-tokens.h, sheetsage-grammar.h): no C++ exceptions, anywhere,
// including internally. Every function that can fail (i.e. every function
// whose Python original can `raise`) returns `bool` and writes a message
// through a `std::string* err` out-param on failure; the ONE public entry
// point, `yue2_sheet_notation_generate()`, always returns the
// `Yue2SheetAbcResult{ok, abc, error}` struct the workflow brief specifies —
// `ok=false` with `error` set is a normal, expected return for a source
// whose ABC genuinely fails to render (pin doc §7.1: "transcription success
// and ABC-generation success are decoupled" — this is the soft-failure
// contract doc 19's decisions section calls for), not a programming error.
//
// ROUNDING/SEMANTICS TRAPS THIS FILE DELIBERATELY REPRODUCES (pin doc §8):
//   1. `Q:1/4={round(...)}` uses Python's round-half-to-even, not C's
//      round-half-away-from-zero (§8 item 1) — see round_half_even_ll().
//   2. `np.searchsorted(..., side="left")` ties stay in the EARLIER subbeat
//      (§8 item 2) — implemented as `std::lower_bound` (first boundary >= t),
//      not "nearest".
//   3. `np.linspace(start, end, 5)[:-1]` is `start + k*(end-start)/4.0` for
//      k=0..3, a single multiply-divide per point, not an accumulating loop
//      (§8 item 6) — see build_grid().
//   4. `(note - 60) // 12` is Python FLOOR division, not C++ truncating `/`
//      (not spelled out in §8 itself, but load-bearing for negative octaves
//      in note_to_abc() — verified against notation_sheetsage2.py:685-691).
//   5. Exact integer arithmetic belongs to the exports stage's rhythm_rows
//      (§8 item 7), not this file — nothing here does beat-grid arithmetic
//      in fractions; that machinery lives upstream of the rows this file
//      consumes.
//   6. Stable sort wherever Python's `sorted()` is used on a tie-possible key
//      (§8 item 8) — see notes_to_arr()'s std::stable_sort.
//
// NOT PORTED (deliberately, per pin doc §7.9-§7.10):
//   - The FILE interface (read_beats/read_chords/read_keys/read_structures,
//     build_rebuilt_abc_score, generate_abc_from_exports, preflight_exports,
//     abc_paths_from_melody) — the pipeline itself never uses it; the test
//     harness reads the fixture files itself and calls the in-memory API.
//   - validate_serialized_abc() (notation_sheetsage2.py:1321-1482) — the
//     pin doc's own words: "a C++ port that faithfully implements §7.2-§7.8
//     will automatically satisfy it" and it is "worth porting as a test, not
//     part of the production render path". This port instead tracks
//     Z-compressibility directly during rendering (see the `plain_rest` flag
//     in render_voice_measure()) rather than re-deriving it by re-parsing
//     the rendered text with `_MUSIC_ELEMENT_RE` — provably equivalent here
//     because this renderer only ever emits the same prefix/note/duration/
//     tie alternation the regex parses, with no gaps, by construction.
//     G4/G5/G6 (doc 19) are scoped to the final ABC text, which this omission
//     does not affect. Open item for a future pass, not a gap in this file's
//     correctness.
//   - numpy's fixed-width `<U16`/`<U64` truncation on key_arr/chord_arr
//     (exports_sheetsage2.py's `_fill_intervals(..., dtype=...)` call sites).
//     `std::string` here is unbounded; the vocabularies these arrays ever
//     hold (pin doc §1.5 full chord labels, ABC-converted forms, §1.7 key
//     strings) are all far short of 16/64 chars, so the truncation is
//     unreachable in practice and reproducing it would only add risk.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────

// One row of a beats/chords/keys/structures export, exactly as a
// tab-separated line of the `21_exports/notation/song_*.txt` fixtures (or
// the reference's own in-memory `rows` lists) splits into columns. Beats
// rows carry 3 or 4 columns (time, beat_id, num[, den]); chords/keys/
// structures rows carry exactly 3 (start, end, value). Mirrors what
// `_row_entries`/`_parse_beats` etc. (notation_sheetsage2.py:220-320) accept.
using Yue2SheetAbcRow = std::vector<std::string>;

struct Yue2SheetAbcResult {
    bool ok = false;
    std::string abc;
    std::string error;
};

// The whole notation stage, in one call. `melody_midi_bytes` is a Standard
// MIDI File (the bytes `exports_sheetsage2.py`'s `midi_bytes(_midi(clean))`
// would produce and `notation/song_melody.mid` in the fixtures holds
// verbatim) with exactly the fixed two-instrument ("Vocal", "Ins") layout
// pin doc §6.4 documents. `melody_only` is always false for the port (pin
// doc §2 — the port never requests `melody_only`); the parameter exists so a
// future caller matches the reference's own signature exactly.
inline Yue2SheetAbcResult yue2_sheet_notation_generate(
    const std::string & melody_midi_bytes,
    const std::vector<Yue2SheetAbcRow> & beat_rows,
    const std::vector<Yue2SheetAbcRow> & chord_rows,
    const std::vector<Yue2SheetAbcRow> & key_rows,
    const std::vector<Yue2SheetAbcRow> & structure_rows,
    bool melody_only = false);

// ─────────────────────────────────────────────────────────────────────────
// Implementation detail
// ─────────────────────────────────────────────────────────────────────────

namespace yue2_sheet_notation_detail {

constexpr int kSubbeatDivision = 4;  // SUBBEAT_DIVISION, notation_sheetsage2.py:20

// ---- tiny string helpers -------------------------------------------------

inline bool is_py_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

inline std::string strip(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && is_py_space(s[a])) ++a;
    while (b > a && is_py_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline std::string to_lower(const std::string & s) {
    std::string out = s;
    for (char & c : out) c = (char)std::tolower((unsigned char)c);
    return out;
}

inline bool try_parse_double(const std::string & s, double & out) {
    std::string t = strip(s);
    if (t.empty()) return false;
    try {
        size_t pos = 0;
        out = std::stod(t, &pos);
        return pos == t.size();
    } catch (...) {
        return false;
    }
}

inline bool try_parse_int(const std::string & s, int & out) {
    std::string t = strip(s);
    if (t.empty()) return false;
    try {
        size_t pos = 0;
        long v = std::stol(t, &pos);
        if (pos != t.size()) return false;
        out = (int)v;
        return true;
    } catch (...) {
        return false;
    }
}

inline std::string fmt6(double v) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(6);
    oss << v;
    return oss.str();
}

// Best-effort Python `repr()` of a list of strings/ints, used only in
// diagnostic/error text on code paths none of the committed fixtures
// exercise (malformed input rows; the meter_conflict="reject" branch, which
// the pipeline never selects — pin doc §7 open question 5). Not byte-exact
// against CPython's repr in every corner case (no quote-escaping); good
// enough for a human reading a rare error.
inline std::string py_repr_row(const Yue2SheetAbcRow & row) {
    std::string s = "[";
    for (size_t i = 0; i < row.size(); ++i) {
        if (i) s += ", ";
        s += "'" + row[i] + "'";
    }
    s += "]";
    return s;
}

template <typename T>
inline std::string py_repr_list(const std::vector<T> & v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

// ---- round-half-to-even (pin doc §8 item 1) ------------------------------

// Python's builtin round(x) with no ndigits: nearest integer, ties to even.
// std::round/naive (int)(x+0.5) are round-half-away-from-zero and disagree
// on an exact .5 — this file's only call site is score_to_abc()'s tempo
// line, none of the committed fixtures land on an exact tie (pin doc §"Open
// questions" item 3), but the rule is implemented properly regardless.
inline long long round_half_even_ll(double x) {
    double f = std::floor(x);
    double diff = x - f;
    long long fl = (long long)f;
    if (diff < 0.5) return fl;
    if (diff > 0.5) return fl + 1;
    return (fl % 2 == 0) ? fl : fl + 1;
}

// Python floor division `a // b` for b > 0 (only use in this file:
// note_to_abc's `(note - 60) // 12`, notation_sheetsage2.py:685).
inline int floordiv(int a, int b) {
    int q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) --q;
    return q;
}

// ---- BeatEvent / Measure / RebuiltAbcScore (mirrors the Python dataclasses)

struct BeatEvent {
    double time = 0.0;
    int beat_id = 0;
    int declared_numerator = 0;
    int denominator = 0;
    int line_no = 0;  // BeatEvent, notation_sheetsage2.py:41-47
};

struct Measure {
    int index = 0;
    int start_beat = 0, end_beat = 0;
    int numerator = 0, denominator = 0;
    bool pickup = false, partial = false, inferred = false;
    // 0 means Python's `None` (numerator/denominator are always positive
    // when actually set, so 0 is an unambiguous "unset" sentinel — see the
    // `abc_numerator`/`abc_denominator` properties, notation_sheetsage2.py:77-82).
    int notated_numerator = 0;
    int notated_denominator = 0;
    bool pad_before = false;

    int start_t() const { return start_beat * kSubbeatDivision; }
    int end_t() const { return end_beat * kSubbeatDivision; }
    int abc_numerator() const { return notated_numerator ? notated_numerator : numerator; }
    int abc_denominator() const { return notated_denominator ? notated_denominator : denominator; }
};

struct RebuiltAbcScore {
    std::vector<BeatEvent> beats;
    std::vector<Measure> measures;
    std::vector<double> subbeat_times;
    std::vector<double> subbeat_quarters;
    std::vector<int> subbeat_denominators;
    std::vector<std::string> key_arr;    // per-subbeat, filled (see fill_intervals)
    std::vector<std::string> chord_arr;  // per-subbeat, filled
    std::vector<std::pair<int, std::string>> structure_events;  // (subbeat, label)
    std::map<std::string, std::vector<int32_t>> voice_arrs;     // "Vocal", "Ins"
    std::vector<std::string> diagnostics;
    int subbeat_div = kSubbeatDivision;
};

struct MeasureGroup {
    std::vector<Measure> measures;
    std::vector<std::string> structure_labels;
    bool meter_changed = false;
    bool key_changed = false;
};

using IntervalRow = std::tuple<double, double, std::string>;  // (start, end, value)

// ---- row parsing / validation (notation_sheetsage2.py:200-320) ----------

// `_row_entries` (:220-226). Just a column-count check; `entries` here is
// the input rows themselves (line_no is 1-based position, matching
// `enumerate(rows, 1)`).
inline bool row_entries(const std::vector<Yue2SheetAbcRow> & rows, const std::string & source,
                         std::vector<Yue2SheetAbcRow> & out, std::string * err) {
    out.clear();
    out.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() < 3) {
            if (err) *err = source + ":" + std::to_string(i + 1) + ": expected at least 3 columns";
            return false;
        }
        out.push_back(rows[i]);
    }
    return true;
}

// `_parse_beats` (:229-262).
inline bool parse_beats(const std::vector<Yue2SheetAbcRow> & entries, const std::string & path,
                         std::vector<BeatEvent> & out, std::string * err) {
    out.clear();
    for (size_t i = 0; i < entries.size(); ++i) {
        int line_no = (int)i + 1;
        const Yue2SheetAbcRow & row = entries[i];
        std::string meter_text = row[2];
        std::string num_text, den_text;
        if (row.size() >= 4) {
            num_text = meter_text;
            den_text = row[3];
        } else {
            size_t slash = meter_text.find('/');
            if (slash != std::string::npos) {
                num_text = meter_text.substr(0, slash);
                den_text = meter_text.substr(slash + 1);
            } else {
                num_text = meter_text;
                den_text = "4";
            }
        }
        double time = 0.0;
        int beat_id = 0, declared_numerator = 0, denominator = 0;
        bool parsed = try_parse_double(row[0], time) && try_parse_int(row[1], beat_id) &&
                      try_parse_int(num_text, declared_numerator) && try_parse_int(den_text, denominator);
        if (!parsed) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": invalid beat row " + py_repr_row(row);
            return false;
        }
        if (beat_id < 1) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": beat ID must be positive";
            return false;
        }
        if (declared_numerator < 1) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": meter numerator must be positive";
            return false;
        }
        if (denominator < 1 || (denominator & (denominator - 1))) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": meter denominator must be a positive power of two";
            return false;
        }
        if (!out.empty() && time <= out.back().time) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": beat times must be strictly increasing";
            return false;
        }
        BeatEvent b;
        b.time = time;
        b.beat_id = beat_id;
        b.declared_numerator = declared_numerator;
        b.denominator = denominator;
        b.line_no = line_no;
        out.push_back(b);
    }
    if (out.size() < 2) {
        if (err) *err = path + ": at least two beat events are required";
        return false;
    }
    return true;
}

// forward decls needed by the row parsers below (chord/key spelling)
inline bool pitch_class(const std::string & root, int & pc, char & letter, std::string & accidental, std::string * err);
inline bool portable_pitch_name(const std::string & root, bool preserve_double, std::string & out, std::string * err);
inline bool chord_symbol_to_abc(const std::string & chord_in, bool & has_value, std::string & text_out, std::string * err);
inline bool key_symbol_to_abc(const std::string & key_in, std::string & out, std::string * err);

// `_parse_chords` (:269-281). Keeps the RAW label (only validates via
// chord_symbol_to_abc; the conversion happens again, on demand, at render
// time — matches the reference exactly, this is not redundant work removal).
inline bool parse_chords(const std::vector<Yue2SheetAbcRow> & entries, const std::string & path,
                          std::vector<IntervalRow> & out, std::string * err) {
    out.clear();
    bool have_prev = false;
    double previous_end = 0.0;
    for (size_t i = 0; i < entries.size(); ++i) {
        int line_no = (int)i + 1;
        const Yue2SheetAbcRow & row = entries[i];
        double start = 0.0, end = 0.0;
        if (!try_parse_double(row[0], start) || !try_parse_double(row[1], end)) {
            if (err) *err = "could not convert string to float: '" + (try_parse_double(row[0], start) ? row[1] : row[0]) + "'";
            return false;
        }
        std::string chord = strip(row[2]);
        if (end <= start) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": chord end must be after start";
            return false;
        }
        if (have_prev && start < previous_end - 1e-6) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": overlapping chord intervals";
            return false;
        }
        bool has_value;
        std::string discard;
        if (!chord_symbol_to_abc(chord, has_value, discard, err)) return false;
        out.push_back({start, end, chord});
        previous_end = end;
        have_prev = true;
    }
    return true;
}

// `_parse_keys` (:288-302). Normalizes into ABC key spelling immediately —
// this IS the call site that turns "C:major"-style decode values into ABC
// key strings; everything downstream only ever sees the normalized form.
inline bool parse_keys(const std::vector<Yue2SheetAbcRow> & entries, const std::string & path,
                        std::vector<IntervalRow> & out, std::string * err) {
    out.clear();
    bool have_prev = false;
    double previous_end = 0.0;
    for (size_t i = 0; i < entries.size(); ++i) {
        int line_no = (int)i + 1;
        const Yue2SheetAbcRow & row = entries[i];
        double start = 0.0, end = 0.0;
        if (!try_parse_double(row[0], start) || !try_parse_double(row[1], end)) {
            if (err) *err = "could not convert string to float: '" + (try_parse_double(row[0], start) ? row[1] : row[0]) + "'";
            return false;
        }
        std::string key = strip(row[2]);
        if (end <= start) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": key end must be after start";
            return false;
        }
        if (have_prev && start < previous_end - 1e-6) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": overlapping key intervals";
            return false;
        }
        std::string normalized;
        if (!key_symbol_to_abc(key, normalized, err)) return false;
        out.push_back({start, end, normalized});
        previous_end = end;
        have_prev = true;
    }
    if (out.empty()) {
        if (err) *err = path + ": at least one key interval is required";
        return false;
    }
    return true;
}

// `_parse_structures` (:309-320). No value normalization here.
inline bool parse_structures(const std::vector<Yue2SheetAbcRow> & entries, const std::string & path,
                              std::vector<IntervalRow> & out, std::string * err) {
    out.clear();
    bool have_prev = false;
    double previous_end = 0.0;
    for (size_t i = 0; i < entries.size(); ++i) {
        int line_no = (int)i + 1;
        const Yue2SheetAbcRow & row = entries[i];
        double start = 0.0, end = 0.0;
        if (!try_parse_double(row[0], start) || !try_parse_double(row[1], end)) {
            if (err) *err = "could not convert string to float: '" + (try_parse_double(row[0], start) ? row[1] : row[0]) + "'";
            return false;
        }
        std::string label = strip(row[2]);
        if (end <= start) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": structure end must be after start";
            return false;
        }
        if (have_prev && start < previous_end - 1e-6) {
            if (err) *err = path + ":" + std::to_string(line_no) + ": overlapping structure intervals";
            return false;
        }
        out.push_back({start, end, label});
        previous_end = end;
        have_prev = true;
    }
    return true;
}

// ---- chord/key spelling (notation_sheetsage2.py:108-197, 563-710) -------

inline const std::array<std::string, 12> & sharp_pitch_names() {
    static const std::array<std::string, 12> t = {"C", "C#", "D", "D#", "E", "F",
                                                    "F#", "G", "G#", "A", "A#", "B"};
    return t;
}
inline const std::array<std::string, 12> & flat_pitch_names() {
    static const std::array<std::string, 12> t = {"C", "Db", "D", "Eb", "E", "F",
                                                    "Gb", "G", "Ab", "A", "Bb", "B"};
    return t;
}
inline const std::string & letters() {
    static const std::string s = "CDEFGAB";
    return s;
}
inline int natural_pitch_class(char letter) {
    switch (letter) {
        case 'C': return 0;
        case 'D': return 2;
        case 'E': return 4;
        case 'F': return 5;
        case 'G': return 7;
        case 'A': return 9;
        case 'B': return 11;
        default: return -1;
    }
}

inline const std::unordered_map<std::string, std::string> & quality_to_abc() {
    static const std::unordered_map<std::string, std::string> t = {
        {"maj", ""},     {"min", "m"},        {"dim", "dim"},   {"aug", "aug"},
        {"7", "7"},      {"maj7", "maj7"},    {"min7", "m7"},   {"dim7", "dim7"},
        {"hdim7", "m7b5"}, {"sus4", "sus4"},  {"sus2", "sus2"}, {"maj6", "6"},
        {"min6", "m6"},  {"sus4(b7)", "7sus4"}, {"minmaj7", "m(maj7)"},
    };
    return t;
}

inline const std::unordered_map<std::string, int> & key_signature_accidentals() {
    static const std::unordered_map<std::string, int> t = {
        {"C", 0},  {"G", 1},  {"D", 2},  {"A", 3},  {"E", 4},   {"B", 5},   {"F#", 6},  {"C#", 7},
        {"F", -1}, {"Bb", -2}, {"Eb", -3}, {"Ab", -4}, {"Db", -5}, {"Gb", -6}, {"Cb", -7},
        {"Am", 0}, {"Em", 1}, {"Bm", 2}, {"F#m", 3}, {"C#m", 4}, {"G#m", 5}, {"D#m", 6}, {"A#m", 7},
        {"Dm", -1}, {"Gm", -2}, {"Cm", -3}, {"Fm", -4}, {"Bbm", -5}, {"Ebm", -6}, {"Abm", -7},
    };
    return t;
}

// _KEY_RELATIVE_PITCH_NAMES, notation_sheetsage2.py:181-197. Keyed by signed
// accidental count (-7..7), 12 entries indexed by pitch class 0=C..11=B.
inline const std::map<int, std::array<std::string, 12>> & key_relative_pitch_names() {
    static const std::map<int, std::array<std::string, 12>> t = {
        {7,  {"B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"}},
        {6,  {"B#", "C#", "C##", "D#", "E",   "E#", "F#", "F##", "G#", "G##", "A#", "B"}},
        {5,  {"B#", "C#", "C##", "D#", "E",   "E#", "F#", "F##", "G#", "A",   "A#", "B"}},
        {4,  {"B#", "C#", "D",   "D#", "E",   "E#", "F#", "F##", "G#", "A",   "A#", "B"}},
        {3,  {"B#", "C#", "D",   "D#", "E",   "E#", "F#", "G",   "G#", "A",   "A#", "B"}},
        {2,  {"C",  "C#", "D",   "D#", "E",   "E#", "F#", "G",   "G#", "A",   "A#", "B"}},
        {1,  {"C",  "C#", "D",   "D#", "E",   "F",  "F#", "G",   "G#", "A",   "A#", "B"}},
        {0,  {"C",  "C#", "D",   "D#", "E",   "F",  "F#", "G",   "G#", "A",   "Bb", "B"}},
        {-1, {"C",  "C#", "D",   "Eb", "E",   "F",  "F#", "G",   "G#", "A",   "Bb", "B"}},
        {-2, {"C",  "C#", "D",   "Eb", "E",   "F",  "F#", "G",   "Ab", "A",   "Bb", "B"}},
        {-3, {"C",  "Db", "D",   "Eb", "E",   "F",  "F#", "G",   "Ab", "A",   "Bb", "B"}},
        {-4, {"C",  "Db", "D",   "Eb", "E",   "F",  "Gb", "G",   "Ab", "A",   "Bb", "B"}},
        {-5, {"C",  "Db", "D",   "Eb", "E",   "F",  "Gb", "G",   "Ab", "A",   "Bb", "Cb"}},
        {-6, {"C",  "Db", "D",   "Eb", "Fb",  "F",  "Gb", "G",   "Ab", "A",   "Bb", "Cb"}},
        {-7, {"C",  "Db", "D",   "Eb", "Fb",  "F",  "Gb", "G",   "Ab", "Bbb", "Bb", "Cb"}},
    };
    return t;
}

// `_pitch_class` (:563-570). `_ROOT_RE = ^[A-G](#{0,2}|b{0,2})$` fullmatch.
inline bool pitch_class(const std::string & root, int & pc, char & letter, std::string & accidental, std::string * err) {
    if (root.empty() || root[0] < 'A' || root[0] > 'G') {
        if (err) *err = "Invalid pitch spelling '" + root + "'";
        return false;
    }
    letter = root[0];
    std::string acc = root.substr(1);
    if (!acc.empty()) {
        bool all_sharp = acc.find_first_not_of('#') == std::string::npos;
        bool all_flat = acc.find_first_not_of('b') == std::string::npos;
        if ((!all_sharp && !all_flat) || acc.size() > 2) {
            if (err) *err = "Invalid pitch spelling '" + root + "'";
            return false;
        }
    }
    accidental = acc;
    int offset = 0;
    if (!acc.empty()) offset = (acc[0] == '#') ? (int)acc.size() : -(int)acc.size();
    pc = ((natural_pitch_class(letter) + offset) % 12 + 12) % 12;
    return true;
}

// `portable_pitch_name` (:573-578).
inline bool portable_pitch_name(const std::string & root, bool preserve_double, std::string & out, std::string * err) {
    int pc = 0;
    char letter = 0;
    std::string accidental;
    if (!pitch_class(root, pc, letter, accidental, err)) return false;
    if (preserve_double || accidental.size() <= 1) {
        out = root;
        return true;
    }
    out = (accidental[0] == '#') ? sharp_pitch_names()[pc] : flat_pitch_names()[pc];
    return true;
}

inline bool looks_like_root_spelling(const std::string & s) {
    int pc = 0;
    char letter = 0;
    std::string accidental;
    return pitch_class(s, pc, letter, accidental, nullptr);
}

// `_bass_degree_to_pitch` (:581-603). `_BASS_DEGREE_RE =
// ^(?P<accidental>#{0,2}|b{0,2})(?P<degree>[1-9]|1[0-3])$`.
inline bool parse_bass_degree_text(const std::string & s, std::string & accidental, int & degree) {
    size_t i = 0;
    while (i < s.size() && (s[i] == '#' || s[i] == 'b')) ++i;
    if (i > 2) return false;
    std::string acc = s.substr(0, i);
    if (!acc.empty()) {
        for (char c : acc)
            if (c != acc[0]) return false;
    }
    std::string rest = s.substr(i);
    if (rest.empty()) return false;
    for (char c : rest)
        if (!std::isdigit((unsigned char)c)) return false;
    int val = std::stoi(rest);
    if (rest.size() == 1) {
        if (val < 1 || val > 9) return false;
    } else if (rest.size() == 2) {
        if (val < 10 || val > 13) return false;
    } else {
        return false;
    }
    accidental = acc;
    degree = val;
    return true;
}

inline bool bass_degree_to_pitch(const std::string & root, const std::string & degree_text, std::string & out, std::string * err) {
    if (looks_like_root_spelling(degree_text)) {
        return portable_pitch_name(degree_text, /*preserve_double=*/true, out, err);
    }
    std::string degree_accidental;
    int degree = 0;
    if (!parse_bass_degree_text(degree_text, degree_accidental, degree)) {
        if (err) *err = "Invalid chord bass degree '" + degree_text + "'";
        return false;
    }
    int root_pc = 0;
    char root_letter = 0;
    std::string root_accidental;
    if (!pitch_class(root, root_pc, root_letter, root_accidental, err)) return false;

    static const int scale_semitones[7] = {0, 2, 4, 5, 7, 9, 11};
    int interval = scale_semitones[(degree - 1) % 7] + 12 * ((degree - 1) / 7);
    int sharp_count = 0, flat_count = 0;
    for (char c : degree_accidental) {
        if (c == '#') ++sharp_count;
        else if (c == 'b') ++flat_count;
    }
    interval += sharp_count - flat_count;
    int target_pc = ((root_pc + interval) % 12 + 12) % 12;

    int root_letter_index = (int)letters().find(root_letter);
    int target_letter_index = (root_letter_index + degree - 1) % 7;
    char target_letter = letters()[target_letter_index];
    int natural_pc = natural_pitch_class(target_letter);
    int raw = target_pc - natural_pc + 6;
    int difference = ((raw % 12 + 12) % 12) - 6;
    if (difference >= -2 && difference <= 2) {
        static const char * acc_text[5] = {"bb", "b", "", "#", "##"};
        out = std::string(1, target_letter) + acc_text[difference + 2];
        return true;
    }
    bool use_sharp = (root_accidental.find('#') != std::string::npos) || (degree_accidental.find('#') != std::string::npos);
    out = use_sharp ? sharp_pitch_names()[target_pc] : flat_pitch_names()[target_pc];
    return true;
}

// `chord_symbol_to_abc` (:606-625). Returns has_value=false for N/X/? (no
// annotation — matches Python returning None).
inline bool chord_symbol_to_abc(const std::string & chord_in, bool & has_value, std::string & text_out, std::string * err) {
    std::string chord = strip(chord_in);
    if (chord == "N" || chord == "X" || chord == "?") {
        has_value = false;
        text_out.clear();
        return true;
    }
    size_t colon = chord.find(':');
    if (colon == std::string::npos) {
        if (err) *err = "Chord '" + chord + "' is missing the ':' quality separator";
        return false;
    }
    std::string root = chord.substr(0, colon);
    std::string descriptor = chord.substr(colon + 1);
    std::string quality, bass_degree;
    bool has_bass = false;
    size_t slash = descriptor.find('/');
    if (slash != std::string::npos) {
        quality = descriptor.substr(0, slash);
        bass_degree = descriptor.substr(slash + 1);
        has_bass = true;
    } else {
        quality = descriptor;
    }
    auto it = quality_to_abc().find(quality);
    if (it == quality_to_abc().end()) {
        if (err) *err = "Unsupported chord quality '" + quality + "' in '" + chord + "'; refusing to rewrite it as major";
        return false;
    }
    std::string chord_root;
    if (!portable_pitch_name(root, /*preserve_double=*/true, chord_root, err)) return false;
    std::string text = chord_root + it->second;
    if (has_bass && !bass_degree.empty()) {
        std::string bass;
        if (!bass_degree_to_pitch(root, bass_degree, bass, err)) return false;
        text += "/" + bass;
    }
    has_value = true;
    text_out = text;
    return true;
}

// `key_symbol_to_abc` (:628-649).
inline bool key_symbol_to_abc(const std::string & key_in, std::string & out, std::string * err) {
    std::string key = strip(key_in);
    std::string root, mode;
    size_t colon = key.find(':');
    if (colon != std::string::npos) {
        root = key.substr(0, colon);
        mode = key.substr(colon + 1);
        if (mode != "major" && mode != "minor") {
            if (err) *err = "Unsupported key mode '" + mode + "' in '" + key + "'";
            return false;
        }
    } else if (!key.empty() && key.back() == 'm') {
        root = key.substr(0, key.size() - 1);
        mode = "minor";
    } else {
        root = key;
        mode = "major";
    }
    int root_pc = 0;
    char root_letter = 0;
    std::string accidental;
    if (!pitch_class(root, root_pc, root_letter, accidental, err)) return false;
    std::string base;
    if (!portable_pitch_name(root, /*preserve_double=*/false, base, err)) return false;
    std::string candidate = base + (mode == "minor" ? "m" : "");
    const auto & table = key_signature_accidentals();
    if (table.count(candidate)) {
        out = candidate;
        return true;
    }
    bool use_flat = accidental.find('b') != std::string::npos;
    const auto & names = use_flat ? flat_pitch_names() : sharp_pitch_names();
    candidate = names[root_pc] + (mode == "minor" ? "m" : "");
    if (!table.count(candidate)) {
        const auto & fallback = use_flat ? sharp_pitch_names() : flat_pitch_names();
        candidate = fallback[root_pc] + (mode == "minor" ? "m" : "");
    }
    if (!table.count(candidate)) {
        if (err) *err = "Cannot encode portable ABC key for '" + key + "'";
        return false;
    }
    out = candidate;
    return true;
}

// `get_key_accidentals` (:652-661). Returns a 7-entry per-letter array
// (index by "CDEFGAB".index(letter)), values in {-1,0,1}.
inline bool get_key_accidentals(const std::string & key, std::array<int, 7> & out, std::string * err) {
    auto it = key_signature_accidentals().find(key);
    if (it == key_signature_accidentals().end()) {
        if (err) *err = "Unsupported ABC key signature '" + key + "'";
        return false;
    }
    int count = it->second;
    out.fill(0);
    const std::string order = count > 0 ? "FCGDAEB" : "BEADGCF";
    for (int i = 0; i < std::abs(count); ++i) {
        size_t idx = letters().find(order[(size_t)i]);
        out[idx] = count > 0 ? 1 : -1;
    }
    return true;
}

// `note_to_abc` (:664-710). `measure_accidentals` is keyed by letter index
// (0=C..6=B) and mutated in place, exactly matching the Python dict.
inline bool note_to_abc(int note, const std::array<int, 7> & key_accidentals, std::map<int, int> & measure_accidentals,
                         std::string & out, std::string * err) {
    int accidental_count = 0;
    for (int a : key_accidentals) accidental_count += a;
    auto it = key_relative_pitch_names().find(accidental_count);
    if (it == key_relative_pitch_names().end()) {
        if (err) *err = "Unsupported key signature accidental count " + std::to_string(accidental_count);
        return false;
    }
    int pc = ((note % 12) + 12) % 12;
    const std::string & pitch_name = it->second[(size_t)pc];
    char letter = pitch_name[0];
    std::string accidental = pitch_name.substr(1);
    int accidental_number;
    if (accidental == "") accidental_number = 0;
    else if (accidental == "#") accidental_number = 1;
    else if (accidental == "##") accidental_number = 2;
    else if (accidental == "b") accidental_number = -1;
    else accidental_number = -2;  // "bb"

    int octave = floordiv(note - 60, 12);
    if (pc == 11 && accidental_number == -1) ++octave;
    else if (pc == 0 && accidental_number == 1) --octave;

    int scale_index = (int)letters().find(letter);
    int current_accidental = measure_accidentals.count(scale_index) ? measure_accidentals[scale_index]
                                                                     : key_accidentals[(size_t)scale_index];
    std::string accidental_text;
    if (current_accidental != accidental_number) {
        measure_accidentals[scale_index] = accidental_number;
        static const std::unordered_map<int, std::string> txt = {
            {-2, "__"}, {-1, "_"}, {0, "="}, {1, "^"}, {2, "^^"},
        };
        accidental_text = txt.at(accidental_number);
    }
    std::string letter_text;
    if (octave > 0) {
        letter_text = std::string(1, (char)std::tolower((unsigned char)letter));
        if (octave > 1) letter_text += std::string((size_t)(octave - 1), '\'');
    } else if (octave < 0) {
        letter_text = std::string(1, letter) + std::string((size_t)(-octave), ',');
    } else {
        letter_text = std::string(1, letter);
    }
    out = accidental_text + letter_text;
    return true;
}

// ---- MIDI parsing (subset: enough to read our own fixed-layout output) --

struct MidiNote {
    int pitch = 0;
    double start = 0.0, end = 0.0;
};
struct MidiInstrument {
    std::string name;
    bool is_drum = false;
    std::vector<MidiNote> notes;
};
struct ParsedMidi {
    std::vector<MidiInstrument> instruments;
};

inline uint32_t midi_u32be(const uint8_t * p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
inline uint16_t midi_u16be(const uint8_t * p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}
inline uint64_t midi_read_vlq(const uint8_t *& p, const uint8_t * end) {
    uint64_t value = 0;
    for (int i = 0; i < 10 && p < end; ++i) {
        uint8_t b = *p++;
        value = (value << 7) | (uint64_t)(b & 0x7F);
        if (!(b & 0x80)) break;
    }
    return value;
}

struct MidiRawEvent {
    uint64_t tick = 0;
    enum Kind { NOTE_ON, NOTE_OFF, META_TEMPO, META_NAME } kind = NOTE_OFF;
    int channel = 0, pitch = 0, velocity = 0;
    double tempo_us = 500000.0;
    std::string text;
};

// A minimal Standard MIDI File reader: delta-time VLQ + running status,
// channel note on/off (velocity-0 note-on treated as note-off, standard MIDI
// convention), meta Set Tempo (0x51) and Track Name (0x03), sysex/other meta
// skipped by length. This covers exactly what `_classify_melody_tracks` +
// `_notes_to_arr` need (pin doc §7.4) and what our own fixed-layout melody
// MIDI (pin doc §6.4: two instruments named "Vocal"/"Ins", program 0,
// velocity always 100) ever contains — it is not a general SMF reader.
inline bool midi_parse_track_events(const uint8_t * p, const uint8_t * end, std::vector<MidiRawEvent> & out) {
    uint64_t abs_tick = 0;
    uint8_t running_status = 0;
    while (p < end) {
        uint64_t delta = midi_read_vlq(p, end);
        abs_tick += delta;
        if (p >= end) break;
        uint8_t status;
        if (*p & 0x80) {
            status = *p++;
        } else {
            status = running_status;
        }
        if (status == 0xFF) {
            running_status = 0;
            if (p >= end) break;
            uint8_t type = *p++;
            uint64_t len = midi_read_vlq(p, end);
            if ((uint64_t)(end - p) < len) return false;
            const uint8_t * data = p;
            p += len;
            if (type == 0x51 && len == 3) {
                double us = (double)(((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | (uint32_t)data[2]);
                out.push_back({abs_tick, MidiRawEvent::META_TEMPO, 0, 0, 0, us, ""});
            } else if (type == 0x03) {
                out.push_back({abs_tick, MidiRawEvent::META_NAME, 0, 0, 0, 500000.0, std::string((const char *)data, (size_t)len)});
            }
            if (type == 0x2F) break;  // end of track
        } else if (status == 0xF0 || status == 0xF7) {
            running_status = 0;
            uint64_t len = midi_read_vlq(p, end);
            if ((uint64_t)(end - p) < len) return false;
            p += len;
        } else if (status >= 0x80 && status <= 0xEF) {
            running_status = status;
            uint8_t hi = status & 0xF0;
            uint8_t ch = status & 0x0F;
            if (hi == 0x80 || hi == 0x90) {
                if (end - p < 2) return false;
                uint8_t pitch = *p++;
                uint8_t vel = *p++;
                auto kind = (hi == 0x80 || vel == 0) ? MidiRawEvent::NOTE_OFF : MidiRawEvent::NOTE_ON;
                out.push_back({abs_tick, kind, (int)ch, (int)pitch, (int)vel, 0.0, ""});
            } else if (hi == 0xA0 || hi == 0xB0 || hi == 0xE0) {
                if (end - p < 2) return false;
                p += 2;
            } else if (hi == 0xC0 || hi == 0xD0) {
                if (end - p < 1) return false;
                p += 1;
            } else {
                return false;
            }
        } else {
            return false;  // stray data byte with no preceding status; malformed
        }
    }
    return true;
}

struct MidiTempoBreak {
    uint64_t tick;
    double time;
    double tempo_us;
};

inline bool midi_parse_bytes(const std::string & bytes, ParsedMidi & out, std::string * err) {
    out.instruments.clear();
    const uint8_t * base = reinterpret_cast<const uint8_t *>(bytes.data());
    const uint8_t * p = base;
    const uint8_t * end = base + bytes.size();
    if (end - p < 14 || std::memcmp(p, "MThd", 4) != 0) {
        if (err) *err = "Invalid MIDI file: missing MThd header";
        return false;
    }
    p += 4;
    uint32_t header_len = midi_u32be(p);
    p += 4;
    p += 2;  // format
    uint16_t ntracks = midi_u16be(p);
    p += 2;
    uint16_t division = midi_u16be(p);
    p += 2;
    if (header_len > 6) p += (header_len - 6);
    if (division & 0x8000) {
        if (err) *err = "Invalid MIDI file: SMPTE division is not supported";
        return false;
    }
    int ppq = division & 0x7FFF;
    if (ppq <= 0) {
        if (err) *err = "Invalid MIDI file: non-positive ticks-per-quarter";
        return false;
    }

    std::vector<std::vector<MidiRawEvent>> tracks;
    std::vector<std::pair<uint64_t, double>> tempo_changes;
    for (int i = 0; i < ntracks; ++i) {
        if (end - p < 8 || std::memcmp(p, "MTrk", 4) != 0) {
            if (err) *err = "Invalid MIDI file: missing MTrk header";
            return false;
        }
        p += 4;
        uint32_t tlen = midi_u32be(p);
        p += 4;
        const uint8_t * tstart = p;
        const uint8_t * tend = p + tlen;
        if (tend > end) {
            if (err) *err = "Invalid MIDI file: truncated track";
            return false;
        }
        std::vector<MidiRawEvent> events;
        if (!midi_parse_track_events(tstart, tend, events)) {
            if (err) *err = "Invalid MIDI file: malformed track " + std::to_string(i);
            return false;
        }
        for (auto & e : events)
            if (e.kind == MidiRawEvent::META_TEMPO) tempo_changes.push_back({e.tick, e.tempo_us});
        tracks.push_back(std::move(events));
        p = tend;
    }

    std::stable_sort(tempo_changes.begin(), tempo_changes.end(),
                      [](const std::pair<uint64_t, double> & a, const std::pair<uint64_t, double> & b) {
                          return a.first < b.first;
                      });
    std::vector<MidiTempoBreak> tmap;
    {
        double time = 0.0;
        uint64_t prev_tick = 0;
        double prev_tempo = 500000.0;
        if (tempo_changes.empty() || tempo_changes[0].first != 0) tmap.push_back({0, 0.0, 500000.0});
        for (auto & tc : tempo_changes) {
            double dt = (double)(tc.first - prev_tick) * (prev_tempo / 1e6) / (double)ppq;
            time += dt;
            tmap.push_back({tc.first, time, tc.second});
            prev_tick = tc.first;
            prev_tempo = tc.second;
        }
    }
    auto tick_to_time = [&](uint64_t tick) -> double {
        size_t lo = 0, hi = tmap.size();
        while (lo + 1 < hi) {
            size_t mid = (lo + hi) / 2;
            if (tmap[mid].tick <= tick) lo = mid; else hi = mid;
        }
        const MidiTempoBreak & bp = tmap[lo];
        return bp.time + (double)(tick - bp.tick) * (bp.tempo_us / 1e6) / (double)ppq;
    };

    for (auto & events : tracks) {
        std::string track_name;
        std::map<int, MidiInstrument> by_channel;
        std::map<std::pair<int, int>, std::vector<uint64_t>> open_notes;  // (channel,pitch) -> stack of start ticks
        for (auto & e : events) {
            if (e.kind == MidiRawEvent::META_NAME) {
                track_name = e.text;
                continue;
            }
            if (e.kind == MidiRawEvent::META_TEMPO) continue;
            auto key = std::make_pair(e.channel, e.pitch);
            if (e.kind == MidiRawEvent::NOTE_ON) {
                open_notes[key].push_back(e.tick);
            } else {  // NOTE_OFF
                auto it = open_notes.find(key);
                if (it != open_notes.end() && !it->second.empty()) {
                    uint64_t start_tick = it->second.back();
                    it->second.pop_back();
                    double start = tick_to_time(start_tick);
                    double endt = tick_to_time(e.tick);
                    if (endt > start) {
                        MidiInstrument & inst = by_channel[e.channel];
                        inst.is_drum = (e.channel == 9);
                        inst.notes.push_back({e.pitch, start, endt});
                    }
                }
            }
        }
        for (auto & kv : by_channel) {
            kv.second.name = track_name;
            out.instruments.push_back(kv.second);
        }
    }
    return true;
}

// `_classify_melody_tracks` (:516-537).
struct MidiClassified {
    std::vector<MidiInstrument> vocal, ins;
};
inline bool classify_melody_tracks(const ParsedMidi & midi, MidiClassified & out, std::string * err) {
    out.vocal.clear();
    out.ins.clear();
    std::vector<MidiInstrument> unknown;
    for (const auto & inst : midi.instruments) {
        if (inst.is_drum) continue;
        std::string name = to_lower(strip(inst.name));
        if (name.find("vocal") != std::string::npos) {
            out.vocal.push_back(inst);
        } else if (name.find("ins") != std::string::npos || name.find("instrument") != std::string::npos) {
            out.ins.push_back(inst);
        } else if (!inst.notes.empty()) {
            unknown.push_back(inst);
        }
    }
    if (!unknown.empty()) {
        if (out.vocal.empty() && out.ins.empty() && unknown.size() == 1) {
            out.ins.push_back(unknown[0]);
        } else {
            std::string names;
            for (size_t i = 0; i < unknown.size(); ++i) {
                if (i) names += ", ";
                names += "'" + (unknown[i].name.empty() ? std::string("<unnamed>") : unknown[i].name) + "'";
            }
            if (err) *err = "Cannot map non-empty melody track(s) [" + names + "] to fixed Vocal/Ins voices";
            return false;
        }
    }
    return true;
}

// ---- measures / grid / quantization (notation_sheetsage2.py:323-560) ----

template <typename T>
inline T mode_with_first_tiebreak(const std::vector<T> & values) {
    std::unordered_map<T, int> counts;
    for (const T & v : values) ++counts[v];
    int maximum = 0;
    for (auto & kv : counts) maximum = std::max(maximum, kv.second);
    for (const T & v : values)
        if (counts[v] == maximum) return v;
    return values.empty() ? T() : values[0];  // unreachable for non-empty input
}

// `infer_measures` (:329-448).
inline bool infer_measures(const std::vector<BeatEvent> & beats, std::vector<Measure> & measures_out,
                            std::vector<std::string> & diagnostics_out, std::string * err) {
    measures_out.clear();
    diagnostics_out.clear();
    std::vector<int> downbeat_indices;
    for (int i = 0; i < (int)beats.size(); ++i)
        if (beats[(size_t)i].beat_id == 1) downbeat_indices.push_back(i);
    if (downbeat_indices.empty()) {
        if (err) *err = "No downbeat (beat ID 1) exists in the beat lab";
        return false;
    }
    struct Span { int start, end; bool pickup, partial; };
    std::vector<Span> spans;
    if (downbeat_indices.front() > 0) spans.push_back({0, downbeat_indices.front(), true, false});
    for (size_t i = 0; i + 1 < downbeat_indices.size(); ++i)
        spans.push_back({downbeat_indices[i], downbeat_indices[i + 1], false, false});
    if (downbeat_indices.back() < (int)beats.size() - 1)
        spans.push_back({downbeat_indices.back(), (int)beats.size() - 1, false, true});
    if (spans.empty()) {
        if (err) *err = "No positive-length measure exists between downbeats";
        return false;
    }

    for (size_t mi = 0; mi < spans.size(); ++mi) {
        const Span & sp = spans[mi];
        std::vector<BeatEvent> events(beats.begin() + sp.start, beats.begin() + sp.end);
        int beat_count = (int)events.size();
        if (beat_count < 1) {
            if (err) *err = "Measure " + std::to_string(mi) + ": empty downbeat span";
            return false;
        }
        std::vector<int> ids;
        for (auto & e : events) ids.push_back(e.beat_id);
        bool consecutive = true;
        for (int k = 0; k < beat_count; ++k)
            if (ids[(size_t)k] != ids[0] + k) { consecutive = false; break; }
        if (!consecutive) {
            if (err) {
                *err = "Measure " + std::to_string(mi) + " (beat rows " + std::to_string(events.front().line_no) +
                       "-" + std::to_string(events.back().line_no) + "): non-consecutive beat IDs " + py_repr_list(ids);
            }
            return false;
        }
        if (!sp.pickup && ids[0] != 1) {
            if (err) *err = "Measure " + std::to_string(mi) + ": full measure does not start at beat ID 1";
            return false;
        }
        std::vector<int> denominators, declared_numerators;
        for (auto & e : events) { denominators.push_back(e.denominator); declared_numerators.push_back(e.declared_numerator); }
        int denominator = mode_with_first_tiebreak(denominators);
        int declared_numerator = mode_with_first_tiebreak(declared_numerators);
        bool numerator_conflict = false;
        for (int v : declared_numerators) if (v != beat_count) numerator_conflict = true;
        bool denominator_conflict = false;
        for (int v : denominators) if (v != denominator) denominator_conflict = true;
        bool all_same_declared = true;
        for (int v : declared_numerators) if (v != declared_numerators[0]) all_same_declared = false;
        bool pad_final_partial = sp.partial && all_same_declared && !denominator_conflict && declared_numerator >= beat_count;
        bool inferred = sp.pickup || sp.partial || numerator_conflict || denominator_conflict;
        bool unresolved_numerator_conflict = numerator_conflict && !pad_final_partial && !sp.pickup;
        (void)unresolved_numerator_conflict;  // meter_conflict="reject" is never used by this port (pin doc §7 open q.5)
        if (pad_final_partial && declared_numerator > beat_count) {
            diagnostics_out.push_back("measure " + std::to_string(mi) + ": padded final " + std::to_string(beat_count) +
                                       "/" + std::to_string(denominator) + " span to declared " + std::to_string(declared_numerator) +
                                       "/" + std::to_string(denominator) + " with trailing rest");
        } else if (numerator_conflict) {
            diagnostics_out.push_back("measure " + std::to_string(mi) + ": inferred " + std::to_string(beat_count) + "/" +
                                       std::to_string(denominator) + " from downbeat span; declared numerators were " +
                                       py_repr_list(declared_numerators));
        }
        if (denominator_conflict) {
            diagnostics_out.push_back("measure " + std::to_string(mi) + ": placed denominator " + std::to_string(denominator) +
                                       " at the measure boundary; row declarations were " + py_repr_list(denominators));
        }
        Measure m;
        m.index = (int)mi;
        m.start_beat = sp.start;
        m.end_beat = sp.end;
        m.numerator = beat_count;
        m.denominator = denominator;
        m.pickup = sp.pickup;
        m.partial = sp.partial;
        m.inferred = inferred;
        m.notated_numerator = pad_final_partial ? declared_numerator : beat_count;
        m.notated_denominator = 0;
        m.pad_before = false;
        measures_out.push_back(m);
    }
    if (measures_out.size() >= 2) {
        Measure & first = measures_out[0];
        Measure & following = measures_out[1];
        double first_duration = (double)first.numerator / (double)first.denominator;
        double following_duration = (double)following.abc_numerator() / (double)following.abc_denominator();
        if (first_duration < following_duration) {
            first.inferred = true;
            first.notated_numerator = following.abc_numerator();
            first.notated_denominator = following.abc_denominator();
            first.pad_before = true;
            diagnostics_out.push_back("measure 0: padded leading " + std::to_string(first.numerator) + "/" +
                                       std::to_string(first.denominator) + " span to " + std::to_string(following.abc_numerator()) +
                                       "/" + std::to_string(following.abc_denominator()) + " with preceding rest");
        }
    }
    return true;
}

struct Grid {
    std::vector<double> subbeat_times;
    std::vector<double> subbeat_quarters;
    std::vector<int> subbeat_denominators;
};

// `_build_grid` (:451-479). np.linspace(start,end,5)[:-1] reproduced as the
// exact multiply-divide formula, not an accumulating loop (pin doc §8.6).
inline bool build_grid(const std::vector<BeatEvent> & beats, const std::vector<Measure> & measures, Grid & out, std::string * err) {
    int n = (int)beats.size() - 1;
    std::vector<int> interval_denominators((size_t)std::max(n, 0), 0);
    for (const auto & m : measures)
        for (int i = m.start_beat; i < m.end_beat; ++i) interval_denominators[(size_t)i] = m.denominator;
    for (int v : interval_denominators) {
        if (v == 0) {
            if (err) *err = "Downbeat spans do not cover every beat interval";
            return false;
        }
    }
    out.subbeat_times.clear();
    out.subbeat_denominators.clear();
    out.subbeat_quarters.clear();
    out.subbeat_quarters.push_back(0.0);
    double current_quarter = 0.0;
    for (int i = 0; i < n; ++i) {
        double start = beats[(size_t)i].time, end = beats[(size_t)i + 1].time;
        int denom = interval_denominators[(size_t)i];
        for (int k = 0; k < kSubbeatDivision; ++k) {
            double t = start + (double)k * (end - start) / (double)kSubbeatDivision;
            out.subbeat_times.push_back(t);
        }
        for (int k = 0; k < kSubbeatDivision; ++k) out.subbeat_denominators.push_back(denom);
        double quarter_step = 4.0 / (double)denom / (double)kSubbeatDivision;
        for (int k = 0; k < kSubbeatDivision; ++k) {
            current_quarter += quarter_step;
            out.subbeat_quarters.push_back(current_quarter);
        }
    }
    out.subbeat_times.push_back(beats.back().time);
    out.subbeat_denominators.push_back(interval_denominators.empty() ? 0 : interval_denominators.back());
    return true;
}

inline std::vector<double> subbeat_boundaries(const std::vector<double> & times) {
    std::vector<double> b;
    if (times.size() < 2) return b;
    b.resize(times.size() - 1);
    for (size_t i = 0; i + 1 < times.size(); ++i) b[i] = (times[i] + times[i + 1]) / 2.0;
    return b;
}

// `_quantize_time` (:486-487). searchsorted(..., side="left"): count of
// boundaries strictly < t (pin doc §8.2) = std::lower_bound's index.
inline int quantize_time(double t, const std::vector<double> & boundaries) {
    return (int)(std::lower_bound(boundaries.begin(), boundaries.end(), t) - boundaries.begin());
}

// `_fill_intervals` (:490-504).
inline bool fill_intervals(const std::vector<IntervalRow> & rows, const std::vector<double> & subbeat_times,
                            const std::string & default_val, std::vector<std::string> & out, std::string * err) {
    out.assign(subbeat_times.size(), default_val);
    std::vector<double> boundaries = subbeat_boundaries(subbeat_times);
    for (const auto & row : rows) {
        double start = std::get<0>(row), end = std::get<1>(row);
        const std::string & value = std::get<2>(row);
        int s = quantize_time(start, boundaries);
        int e = quantize_time(end, boundaries);
        s = std::max(0, std::min(s, (int)out.size() - 1));
        e = std::max(0, std::min(e, (int)out.size() - 1));
        if (e <= s) {
            if (err) *err = "Interval " + fmt6(start) + "-" + fmt6(end) + " (" + value + ") is shorter than the ABC subbeat grid";
            return false;
        }
        for (int i = s; i < e; ++i) out[(size_t)i] = value;
    }
    if (out.size() > 1) out.back() = out[out.size() - 2];
    return true;
}

// `_structure_events` (:507-513).
inline std::vector<std::pair<int, std::string>> structure_events(const std::vector<IntervalRow> & rows,
                                                                   const std::vector<double> & subbeat_times) {
    std::vector<double> boundaries = subbeat_boundaries(subbeat_times);
    std::vector<std::pair<int, std::string>> events;
    for (const auto & row : rows) {
        double start = std::get<0>(row);
        const std::string & label = std::get<2>(row);
        int t = quantize_time(start, boundaries);
        t = std::max(0, std::min(t, (int)subbeat_times.size() - 1));
        events.push_back({t, label});
    }
    return events;
}

// `_notes_to_arr` (:540-560).
inline bool notes_to_arr(std::vector<MidiNote> notes, const std::vector<double> & subbeat_times, const std::string & voice_id,
                          std::vector<int32_t> & out, std::string * err) {
    out.assign(subbeat_times.size(), 0);
    std::vector<double> boundaries = subbeat_boundaries(subbeat_times);
    std::stable_sort(notes.begin(), notes.end(), [](const MidiNote & a, const MidiNote & b) {
        if (a.start != b.start) return a.start < b.start;
        if (a.end != b.end) return a.end < b.end;
        return a.pitch < b.pitch;
    });
    for (const auto & note : notes) {
        int start_t = quantize_time(note.start, boundaries);
        int end_t = quantize_time(note.end, boundaries);
        start_t = std::max(0, std::min(start_t, (int)out.size() - 1));
        end_t = std::max(0, std::min(end_t, (int)out.size() - 1));
        if (end_t <= start_t) {
            if (err) {
                *err = voice_id + ": MIDI note pitch=" + std::to_string(note.pitch) + " at " + fmt6(note.start) + "-" +
                       fmt6(note.end) + " cannot be represented on the decoded subbeat grid";
            }
            return false;
        }
        bool overlap = false;
        for (int i = start_t; i < end_t; ++i)
            if (out[(size_t)i] != 0) { overlap = true; break; }
        if (overlap) {
            if (err) *err = voice_id + ": overlapping quantized melody notes at subbeats " + std::to_string(start_t) + ":" + std::to_string(end_t);
            return false;
        }
        int32_t sustain = note.pitch * 2 + 2;
        for (int i = start_t; i < end_t; ++i) out[(size_t)i] = sustain;
        out[(size_t)start_t] = sustain + 1;
    }
    return true;
}

// ---- unit denominator / tempo / duration splitting (:798-913) -----------

inline bool abc_unit_denominator(const RebuiltAbcScore & score, long long & out, std::string * err) {
    long long l = 1;
    for (const auto & m : score.measures) {
        l = std::lcm(l, (long long)m.denominator * score.subbeat_div);
        l = std::lcm(l, (long long)m.abc_denominator() * score.subbeat_div);
    }
    if (l > 1024) {
        if (err) *err = "Required ABC unit length 1/" + std::to_string(l) + " is unreasonably small";
        return false;
    }
    out = l;
    return true;
}

inline long long measure_actual_units(const Measure & m, long long unit_denominator) {
    return (long long)m.numerator * unit_denominator / m.denominator;
}
inline long long measure_abc_units(const Measure & m, long long unit_denominator) {
    return (long long)m.abc_numerator() * unit_denominator / m.abc_denominator();
}
inline long long measure_padding_units(const Measure & m, long long unit_denominator) {
    return measure_abc_units(m, unit_denominator) - measure_actual_units(m, unit_denominator);
}

inline bool duration_units(const RebuiltAbcScore & score, int start_t, int end_t, long long unit_denominator,
                            long long & out, std::string * err) {
    long long units = 0;
    for (int i = start_t; i < end_t; ++i) {
        long long divisor = (long long)score.subbeat_denominators[(size_t)i] * score.subbeat_div;
        if (unit_denominator % divisor != 0) {
            if (err) *err = "ABC L:1/" + std::to_string(unit_denominator) + " cannot express a 1/" + std::to_string(divisor) + " subbeat exactly";
            return false;
        }
        units += unit_denominator / divisor;
    }
    out = units;
    return true;
}

inline bool estimate_tempo(const RebuiltAbcScore & score, double & out, std::string * err) {
    double seconds = score.subbeat_times.back() - score.subbeat_times.front();
    double quarter_notes = score.subbeat_quarters.back() - score.subbeat_quarters.front();
    if (seconds <= 0 || quarter_notes <= 0) {
        if (err) *err = "Cannot estimate tempo from a zero-duration score";
        return false;
    }
    out = quarter_notes / seconds * 60.0;
    return true;
}

inline bool continues_pitch(int value, int next_value) {
    if (value <= 0) return false;
    int pitch = value / 2 - 1;
    return next_value == pitch * 2 + 2;
}
inline bool same_note_segment(int value, int next_value) {
    if (value == 0) return next_value == 0;
    int pitch = value / 2 - 1;
    return next_value == pitch * 2 + 2;
}

inline const std::vector<int> & supported_duration_units() {
    static const std::vector<int> v = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48};
    return v;
}

inline bool split_duration_units(int duration, std::vector<int> & out, std::string * err) {
    out.clear();
    if (duration <= 0) {
        if (err) *err = "Cannot serialize non-positive duration " + std::to_string(duration);
        return false;
    }
    int remaining = duration;
    const auto & supported = supported_duration_units();
    while (remaining > 0) {
        if (std::find(supported.begin(), supported.end(), remaining) != supported.end()) {
            out.push_back(remaining);
            break;
        }
        int chunk = -1;
        for (int v : supported)
            if (v < remaining && v > chunk) chunk = v;
        if (chunk == -1) {
            if (err) *err = "Duration " + std::to_string(duration) + " cannot be split into representable ABC values";
            return false;
        }
        out.push_back(chunk);
        remaining -= chunk;
    }
    return true;
}

inline std::string duration_text(int n) { return n == 1 ? std::string() : std::to_string(n); }

inline bool render_duration_tokens(const std::string & prefix, const std::string & note_text, int duration, bool tie_out,
                                    std::vector<std::string> & out, std::string * err) {
    std::vector<int> chunks;
    if (!split_duration_units(duration, chunks, err)) return false;
    out.clear();
    for (size_t i = 0; i < chunks.size(); ++i) {
        bool continues = (note_text != "z") && (i + 1 < chunks.size() || tie_out);
        out.push_back((i == 0 ? prefix : std::string()) + note_text + duration_text(chunks[i]) + (continues ? "-" : ""));
    }
    return true;
}

// ---- one measure / one voice-group / grouping / header (:916-1162) ------

struct RenderedMeasure {
    std::string text;
    bool compressible_full_rest = false;
};

// `_render_voice_measure` (:916-1018). Tracks Z-compressibility directly
// (see the file header's "NOT PORTED" note) instead of re-parsing the
// rendered text with `_MUSIC_ELEMENT_RE`: a measure is a compressible full
// rest iff every segment this function emits has an empty prefix (no
// quoted chord, no inline key change) and note_text == "z" — which is
// exactly `_is_compressible_full_rest`'s condition, checked as we go rather
// than after the fact.
inline bool render_voice_measure(const RebuiltAbcScore & score, const std::string & voice_id, const Measure & measure,
                                  long long unit_denominator, RenderedMeasure & out, std::string * err) {
    auto voice_it = score.voice_arrs.find(voice_id);
    static const std::vector<int32_t> empty_voice;
    const std::vector<int32_t> & voice = (voice_it != score.voice_arrs.end()) ? voice_it->second : empty_voice;
    bool show_chords = (voice_id == "Vocal");
    std::map<int, int> measure_accidentals;
    std::string current_key = score.key_arr[(size_t)measure.start_t()];
    std::array<int, 7> key_accidentals{};
    if (!get_key_accidentals(current_key, key_accidentals, err)) return false;

    std::vector<std::string> parts;
    bool compressible = true;
    auto emit = [&](const std::string & prefix, const std::string & note_text, long long duration, bool tie_out) -> bool {
        if (!prefix.empty() || note_text != "z") compressible = false;
        std::vector<std::string> toks;
        if (!render_duration_tokens(prefix, note_text, (int)duration, tie_out, toks, err)) return false;
        for (auto & t : toks) parts.push_back(t);
        return true;
    };

    long long padding = measure_padding_units(measure, unit_denominator);
    if (padding < 0) {
        if (err) *err = "Measure " + std::to_string(measure.index) + ": notated meter is shorter than its decoded span";
        return false;
    }
    long long leading_padding = measure.pad_before ? padding : 0;
    long long trailing_padding = measure.pad_before ? 0 : padding;

    int t = measure.start_t();
    int end_t = measure.end_t();
    while (t < end_t) {
        int cp_note = end_t, cp_key = end_t, cp_chord = end_t;
        for (int probe = t + 1; probe < end_t; ++probe) {
            if (!same_note_segment((int)voice[(size_t)t], (int)voice[(size_t)probe])) { cp_note = probe; break; }
        }
        for (int probe = t + 1; probe < end_t; ++probe) {
            if (score.key_arr[(size_t)probe] != score.key_arr[(size_t)(probe - 1)]) { cp_key = probe; break; }
        }
        if (show_chords) {
            for (int probe = t + 1; probe < end_t; ++probe) {
                if (score.chord_arr[(size_t)probe] != score.chord_arr[(size_t)(probe - 1)]) { cp_chord = probe; break; }
            }
        }
        int next_t = std::min({end_t, cp_note, cp_key, cp_chord});

        std::string prefix;
        const std::string & key = score.key_arr[(size_t)t];
        if (t > measure.start_t() && key != current_key) {
            current_key = key;
            if (!get_key_accidentals(current_key, key_accidentals, err)) return false;
            measure_accidentals.clear();
            prefix += "[K:" + current_key + "]";
        }
        if (show_chords && (t == measure.start_t() || score.chord_arr[(size_t)t] != score.chord_arr[(size_t)(t - 1)])) {
            const std::string & chord = score.chord_arr[(size_t)t];
            bool has_value = false;
            std::string chord_text;
            if (!chord_symbol_to_abc(chord, has_value, chord_text, err)) return false;
            if (has_value) prefix += "\"" + chord_text + "\"";
        }

        int value = (int)voice[(size_t)t];
        std::string note_text;
        if (value == 0) {
            note_text = "z";
        } else {
            if (!note_to_abc(value / 2 - 1, key_accidentals, measure_accidentals, note_text, err)) return false;
        }
        long long duration = 0;
        if (!duration_units(score, t, next_t, unit_denominator, duration, err)) return false;

        if (t == measure.start_t() && leading_padding) {
            if (value == 0 && prefix.empty()) {
                duration += leading_padding;
            } else {
                if (!emit("", "z", leading_padding, false)) return false;
            }
            leading_padding = 0;
        }
        if (value == 0 && next_t == end_t && trailing_padding) {
            duration += trailing_padding;
            trailing_padding = 0;
        }
        if (duration <= 0) {
            if (err) *err = "Non-positive ABC duration at subbeats " + std::to_string(t) + ":" + std::to_string(next_t);
            return false;
        }
        bool tie_out = (value > 0) && (next_t < (int)voice.size()) && continues_pitch(value, (int)voice[(size_t)next_t]);
        if (!emit(prefix, note_text, duration, tie_out)) return false;
        t = next_t;
    }
    if (leading_padding != 0) {
        if (err) *err = "Measure " + std::to_string(measure.index) + ": leading rest padding was not serialized";
        return false;
    }
    if (trailing_padding) {
        if (!emit("", "z", trailing_padding, false)) return false;
    }
    std::string text;
    for (auto & p : parts) text += p;
    out.text = text;
    out.compressible_full_rest = compressible;
    return true;
}

// `_render_voice_group` (:1038-1069): coalesce runs of compressible-full-rest
// measures into "Z" / "Zn" (n>=2; a run of exactly 1 is plain "Z", never "Z1").
inline bool render_voice_group(const RebuiltAbcScore & score, const std::string & voice_id, const std::vector<Measure> & measures,
                                long long unit_denominator, std::string & out, std::string * err) {
    std::vector<RenderedMeasure> rendered;
    rendered.reserve(measures.size());
    for (const auto & m : measures) {
        RenderedMeasure rm;
        if (!render_voice_measure(score, voice_id, m, unit_denominator, rm, err)) return false;
        rendered.push_back(rm);
    }
    std::string result;
    size_t index = 0;
    while (index < rendered.size()) {
        if (!rendered[index].compressible_full_rest) {
            result += rendered[index].text + "|";
            ++index;
            continue;
        }
        size_t end = index + 1;
        while (end < rendered.size() && rendered[end].compressible_full_rest) ++end;
        size_t count = end - index;
        result += "Z" + (count > 1 ? std::to_string(count) : std::string()) + "|";
        index = end;
    }
    out = result;
    return true;
}

inline std::string sanitize_structure_label(const std::string & value) {
    std::istringstream iss(value);
    std::string word, out;
    while (iss >> word) {
        if (!out.empty()) out += " ";
        out += word;
    }
    return out;
}

// `_measure_groups` (:1076-1121).
inline std::vector<MeasureGroup> measure_groups(const RebuiltAbcScore & score) {
    const Measure & first_measure = score.measures[0];
    std::pair<int, int> active_meter = {first_measure.abc_numerator(), first_measure.abc_denominator()};
    std::string active_key = score.key_arr[(size_t)first_measure.start_t()];
    std::string active_structure;
    std::vector<MeasureGroup> groups;
    for (const auto & measure : score.measures) {
        std::pair<int, int> meter = {measure.abc_numerator(), measure.abc_denominator()};
        const std::string & key = score.key_arr[(size_t)measure.start_t()];
        bool meter_changed = (meter != active_meter);
        bool key_changed = (key != active_key);
        std::vector<std::string> new_labels;
        for (const auto & te : score.structure_events) {
            int t = te.first;
            if (!(measure.start_t() <= t && t < measure.end_t())) continue;
            std::string clean = sanitize_structure_label(te.second);
            if (!clean.empty() && clean != active_structure) {
                new_labels.push_back(clean);
                active_structure = clean;
            }
        }
        bool start_group = groups.empty() || groups.back().measures.size() >= 4 || meter_changed || key_changed || !new_labels.empty();
        if (start_group) {
            MeasureGroup g;
            g.measures.push_back(measure);
            g.structure_labels = new_labels;
            g.meter_changed = meter_changed;
            g.key_changed = key_changed;
            groups.push_back(g);
        } else {
            groups.back().measures.push_back(measure);
        }
        active_meter = meter;
        active_key = score.key_arr[(size_t)(measure.end_t() - 1)];
    }
    return groups;
}

// `score_to_abc` (:1124-1162), minus the `validate_serialized_abc()` self-check
// (file header "NOT PORTED" note).
inline bool score_to_abc(const RebuiltAbcScore & score, std::string & out, std::string * err) {
    long long unit_denominator = 0;
    if (!abc_unit_denominator(score, unit_denominator, err)) return false;
    const Measure & first_measure = score.measures[0];
    const std::string & first_key = score.key_arr[(size_t)first_measure.start_t()];

    double tempo_f = 0.0;
    if (!estimate_tempo(score, tempo_f, err)) return false;
    long long tempo = round_half_even_ll(tempo_f);

    std::vector<std::string> lines;
    lines.push_back("X:1");
    lines.push_back("T:");
    lines.push_back("M:" + std::to_string(first_measure.abc_numerator()) + "/" + std::to_string(first_measure.abc_denominator()));
    lines.push_back("L:1/" + std::to_string(unit_denominator));
    lines.push_back("Q:1/4=" + std::to_string(tempo));
    lines.push_back("V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"");
    lines.push_back("V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"");
    lines.push_back("K:" + first_key);

    for (const auto & group : measure_groups(score)) {
        for (const auto & label : group.structure_labels) lines.push_back("% " + label);
        const Measure & first_group_measure = group.measures[0];
        static const char * voice_ids[2] = {"Vocal", "Ins"};
        for (const char * voice_id : voice_ids) {
            lines.push_back(std::string("V: ") + voice_id);
            if (group.meter_changed) {
                lines.push_back("M:" + std::to_string(first_group_measure.abc_numerator()) + "/" +
                                 std::to_string(first_group_measure.abc_denominator()));
            }
            if (group.key_changed) {
                lines.push_back("K:" + score.key_arr[(size_t)first_group_measure.start_t()]);
            }
            std::string voice_text;
            if (!render_voice_group(score, voice_id, group.measures, unit_denominator, voice_text, err)) return false;
            lines.push_back(voice_text);
        }
    }
    std::string text;
    for (auto & l : lines) { text += l; text += "\n"; }
    out = text;
    return true;
}

// `_assemble_abc_score` (:761-795), given already-parsed rows and MIDI.
inline bool assemble_abc_score(const ParsedMidi & midi, const std::vector<BeatEvent> & beats,
                                const std::vector<IntervalRow> & keys, const std::vector<IntervalRow> & structures,
                                const std::vector<IntervalRow> & chords, bool melody_only, RebuiltAbcScore & out,
                                std::string * err) {
    std::vector<Measure> measures;
    std::vector<std::string> diagnostics;
    if (!infer_measures(beats, measures, diagnostics, err)) return false;
    Grid grid;
    if (!build_grid(beats, measures, grid, err)) return false;
    MidiClassified classified;
    if (!classify_melody_tracks(midi, classified, err)) return false;

    std::map<std::string, std::vector<int32_t>> voice_arrs;
    {
        std::vector<MidiNote> vocal_notes;
        for (const auto & inst : classified.vocal)
            for (const auto & n : inst.notes) vocal_notes.push_back(n);
        std::vector<int32_t> arr;
        if (!notes_to_arr(vocal_notes, grid.subbeat_times, "Vocal", arr, err)) return false;
        voice_arrs["Vocal"] = std::move(arr);
    }
    {
        std::vector<MidiNote> ins_notes;
        for (const auto & inst : classified.ins)
            for (const auto & n : inst.notes) ins_notes.push_back(n);
        std::vector<int32_t> arr;
        if (!notes_to_arr(ins_notes, grid.subbeat_times, "Ins", arr, err)) return false;
        voice_arrs["Ins"] = std::move(arr);
    }

    if (keys.empty()) {
        if (err) *err = "at least one key interval is required";  // defensive; parse_keys already enforces this
        return false;
    }
    std::vector<std::string> key_arr;
    if (!fill_intervals(keys, grid.subbeat_times, std::get<2>(keys[0]), key_arr, err)) return false;

    std::vector<std::string> chord_arr;
    if (melody_only) {
        chord_arr.assign(grid.subbeat_times.size(), "N");
    } else {
        if (!fill_intervals(chords, grid.subbeat_times, "N", chord_arr, err)) return false;
    }

    out.beats = beats;
    out.measures = measures;
    out.subbeat_times = grid.subbeat_times;
    out.subbeat_quarters = grid.subbeat_quarters;
    out.subbeat_denominators = grid.subbeat_denominators;
    out.key_arr = key_arr;
    out.chord_arr = chord_arr;
    out.structure_events = structure_events(structures, grid.subbeat_times);
    out.voice_arrs = voice_arrs;
    out.diagnostics = diagnostics;
    out.subbeat_div = kSubbeatDivision;
    return true;
}

}  // namespace yue2_sheet_notation_detail

// ---- public entry point --------------------------------------------------

inline Yue2SheetAbcResult yue2_sheet_notation_generate(
    const std::string & melody_midi_bytes,
    const std::vector<Yue2SheetAbcRow> & beat_rows,
    const std::vector<Yue2SheetAbcRow> & chord_rows,
    const std::vector<Yue2SheetAbcRow> & key_rows,
    const std::vector<Yue2SheetAbcRow> & structure_rows,
    bool melody_only) {
    using namespace yue2_sheet_notation_detail;
    Yue2SheetAbcResult result;
    std::string err;

    std::vector<Yue2SheetAbcRow> beat_entries;
    if (!row_entries(beat_rows, "beats", beat_entries, &err)) { result.error = err; return result; }
    std::vector<BeatEvent> beats;
    if (!parse_beats(beat_entries, "beats", beats, &err)) { result.error = err; return result; }

    std::vector<Yue2SheetAbcRow> key_entries;
    if (!row_entries(key_rows, "keys", key_entries, &err)) { result.error = err; return result; }
    std::vector<IntervalRow> keys;
    if (!parse_keys(key_entries, "keys", keys, &err)) { result.error = err; return result; }

    std::vector<Yue2SheetAbcRow> structure_entries;
    if (!row_entries(structure_rows, "structures", structure_entries, &err)) { result.error = err; return result; }
    std::vector<IntervalRow> structures;
    if (!parse_structures(structure_entries, "structures", structures, &err)) { result.error = err; return result; }

    std::vector<IntervalRow> chords;
    if (!melody_only) {
        std::vector<Yue2SheetAbcRow> chord_entries;
        if (!row_entries(chord_rows, "chords", chord_entries, &err)) { result.error = err; return result; }
        if (!parse_chords(chord_entries, "chords", chords, &err)) { result.error = err; return result; }
    }

    ParsedMidi midi;
    if (!midi_parse_bytes(melody_midi_bytes, midi, &err)) { result.error = err; return result; }

    RebuiltAbcScore score;
    if (!assemble_abc_score(midi, beats, keys, structures, chords, melody_only, score, &err)) {
        result.error = err;
        return result;
    }
    std::string abc;
    if (!score_to_abc(score, abc, &err)) { result.error = err; return result; }

    result.ok = true;
    result.abc = abc;
    return result;
}
