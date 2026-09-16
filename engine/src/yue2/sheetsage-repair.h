#pragma once
// yue2/sheetsage-repair.h — deterministic post-failure repair pass for the
// SheetSage2 notation stage.
//
// HOT-Step file (no acestep.cpp analog; no upstream Python analog either —
// the reference just fails these sources outright, doc 21 §7.1). Rob's
// decision (2026-09-16): on a real dataset, a handful of sources hit one of
// a small, well-understood family of notation errors — a duplicated beat ID
// inside one measure, a non-increasing beat time, a chord/key interval too
// short for the subbeat grid, or a melody note that can't land on the grid —
// and the pre-existing contract simply drops the whole source (abc_error,
// trains cot=off forever). A repaired sheet is better training conditioning
// than none, so this file re-renders after dropping exactly the offending
// row/note, up to a few rounds, and reports what it dropped. Every other
// notation failure (an unencodable chord quality, no key at all, an
// unmappable melody track, ...) is untouched — this file recognizes exactly
// four error shapes and refuses to guess at anything else.
//
// SCOPE. This file is invoked ONLY after sheetsage-notation.h's own
// `yue2_sheet_notation_generate()` has already failed once on the untouched
// inputs — sheetsage-pipeline.h always tries that first call itself (see its
// own comment), so G4/G6 byte-exactness on the pinned fixtures never passes
// through this file: a fixture that is SUPPOSED to succeed on the first
// render never reaches here, and a fixture that is supposed to reproduce a
// specific `abc_error` string only reaches here if a future test happens to
// feed it a source shaped like one of the four families below (none of the
// doc 19/22 fixtures are).
//
// FAMILIES (doc 21 §7.1's AbcRebuildError hierarchy; error TEXT is the only
// interface this port's notation stage exposes for "which typed error", the
// same string-matching convention doc 21 §5.4 already documents for the
// decoder's own recoverable-vs-fatal split):
//   (a) BeatGridError "Measure N (beat rows R1-R2): non-consecutive beat IDs
//       [...]" -> a single sweep of the WHOLE beat list, segmented into
//       measures by downbeat (beat_id==1) the same way infer_measures()
//       itself does, dropping the LATER row of every duplicate beat_id in
//       EVERY measure in one round (not just the one measure named in the
//       error) -- real files can carry several bad measures, and fixing one
//       row per round hit the round cap on real data.
//   (b) BeatGridError "beats:LINE: beat times must be strictly increasing"
//       -> a single sweep of the WHOLE beat list, keeping a row only if its
//       time is strictly greater than the last KEPT row's time, dropping
//       every offending row in one round (a run of several equal/decreasing
//       timestamps all go in this one pass, keeping the first of the run).
//   (c) AbcRebuildError "Interval S-E (VALUE) is shorter than the ABC
//       subbeat grid" (raised by fill_intervals for a chord or key row; see
//       pin doc §7.3 — structure never reaches this specific check, but the
//       lookup here tries structure_rows too for completeness) -> drop that
//       one interval row. No explicit "extend the previous interval" step is
//       needed: fill_intervals (sheetsage-notation.h) already fills every
//       subbeat cell from whichever row claims it, in row order, and simply
//       removing the too-short row leaves its span uncovered until the
//       previous row's own end — which IS the extension the task asks for.
//   (d) MelodyVoiceError "VOICE: MIDI note pitch=P at S-E cannot be
//       represented on the decoded subbeat grid" -> drop that one note from
//       its voice's raw-note list (VOICE is "Vocal" or "Ins", matching
//       sheetsage-notation.h's own notes_to_arr() call sites) before the
//       melody MIDI bytes are rebuilt for the retry.
//
// Every other AbcRebuildError/BeatGridError/ChordSymbolError/MelodyVoiceError
// text — "overlapping quantized melody notes", "Cannot encode portable ABC
// key", "No key was decoded", an unmappable melody track, etc — matches none
// of the four regexes below and the loop stops on the FIRST round that finds
// nothing to fix, exactly the pre-existing soft-failure behavior.

#include "sheetsage-events.h"
#include "sheetsage-notation.h"

#include <cmath>
#include <cstdlib>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ── Public API ───────────────────────────────────────────────────────────────

// Everything yue2_sheet_notation_generate() needs, PLUS the pre-MIDI raw
// melody notes (sheetsage-pipeline.h's own `vocal`/`ins` split, before
// yue2_sheet_write_midi() bakes them into bytes) so family (d) can drop one
// note and rebuild the MIDI for the retry. Mutated in place across rounds.
struct Yue2SheetRepairInputs {
    std::vector<Yue2SheetAbcRow>  beat_rows, chord_rows, key_rows, structure_rows;
    std::vector<Yue2SheetRawNote> vocal_notes, ins_notes;
};

struct Yue2SheetRepairOutcome {
    Yue2SheetAbcResult result;    // final render attempt (ok/abc/error)
    std::string        repaired;  // "" iff no repair round ever changed anything
};

namespace yue2_sheet_repair_detail {

constexpr int kMaxRepairRounds = 4;

inline bool parse_double(const std::string & s, double & out) {
    try {
        size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

// Family (a). Gated on sheetsage-notation.h's infer_measures() error text
// ("Measure " + idx + " (beat rows " + first_line + "-" + last_line + "):
// non-consecutive beat IDs " + py_repr_list(ids)), but — unlike an earlier
// version of this function — does NOT limit the fix to the one measure named
// in that error: a real file can carry several bad measures, and fixing them
// one row per round hit the round cap on real data (2026-09-16 follow-up).
// Instead this re-derives infer_measures()'s own downbeat segmentation
// (pin doc §7.2: a span starts at each row where beat_id==1, plus a leading
// pickup span before the first downbeat and a trailing partial span after
// the last) over the WHOLE beat_rows list and, within every span
// independently, drops the LATER row of each duplicate beat_id — one sweep,
// one round, regardless of how many measures are affected.
inline bool repair_dup_beat_id(const std::string & error, std::vector<Yue2SheetAbcRow> & beat_rows,
                                std::string & note) {
    static const std::regex re(R"(Measure \d+ \(beat rows \d+-\d+\): non-consecutive beat IDs \[[0-9, ]*\])");
    if (!std::regex_search(error, re)) {
        return false;  // not this family — refuse to guess
    }
    const int n = (int) beat_rows.size();
    std::vector<int> ids(n, 0);
    for (int i = 0; i < n; ++i) {
        if (beat_rows[(size_t) i].size() < 2) {
            return false;  // malformed row — refuse to guess rather than mis-segment
        }
        ids[(size_t) i] = std::atoi(beat_rows[(size_t) i][1].c_str());
    }
    std::vector<int> downbeats;
    for (int i = 0; i < n; ++i) {
        if (ids[(size_t) i] == 1) {
            downbeats.push_back(i);
        }
    }
    std::vector<std::pair<int, int>> spans;  // [start, end)
    if (downbeats.empty()) {
        spans.push_back({ 0, n });
    } else {
        if (downbeats.front() > 0) {
            spans.push_back({ 0, downbeats.front() });
        }
        for (size_t k = 0; k + 1 < downbeats.size(); ++k) {
            spans.push_back({ downbeats[k], downbeats[k + 1] });
        }
        spans.push_back({ downbeats.back(), n });
    }

    std::vector<int> drop_idx;  // 0-based absolute indices into beat_rows
    int              measures_touched = 0;
    for (const auto & span : spans) {
        std::unordered_map<int, int> seen;
        bool                          span_touched = false;
        for (int i = span.first; i < span.second; ++i) {
            const int id = ids[(size_t) i];
            if (seen.find(id) != seen.end()) {
                drop_idx.push_back(i);
                span_touched = true;
            } else {
                seen[id] = 1;
            }
        }
        if (span_touched) {
            measures_touched++;
        }
    }
    if (drop_idx.empty()) {
        return false;
    }
    std::sort(drop_idx.begin(), drop_idx.end(), std::greater<int>());
    for (int idx : drop_idx) {
        beat_rows.erase(beat_rows.begin() + idx);
    }
    note = "beats: dropped " + std::to_string(drop_idx.size()) + " dup-beat-id row" +
           (drop_idx.size() == 1 ? "" : "s") + " across " + std::to_string(measures_touched) + " measure" +
           (measures_touched == 1 ? "" : "s");
    return true;
}

// Family (b). Gated on parse_beats()'s error text ("beats:" + line_no + ":
// beat times must be strictly increasing"), but — same 2026-09-16 follow-up
// as family (a) above — fixes every offending row in ONE pass over the whole
// list rather than the one row named in the error: walk beat_rows in order,
// keep a row only if its time is strictly greater than the last KEPT row's
// time (never the last SEEN row), so a run of several equal/decreasing
// timestamps in a row all get dropped in this single round, keeping the
// first of the run — exactly the task's own "keep the first of equal times".
inline bool repair_non_increasing_time(const std::string & error, std::vector<Yue2SheetAbcRow> & beat_rows,
                                        std::string & note) {
    static const std::regex re(R"(^beats:\d+: beat times must be strictly increasing$)");
    if (!std::regex_search(error, re)) {
        return false;  // not this family — refuse to guess
    }
    std::vector<int> drop_idx;
    double            last_kept = -1.0;
    bool              have_last = false;
    for (int i = 0; i < (int) beat_rows.size(); ++i) {
        if (beat_rows[(size_t) i].empty()) {
            return false;  // malformed row — refuse to guess
        }
        double t = 0.0;
        if (!parse_double(beat_rows[(size_t) i][0], t)) {
            return false;
        }
        if (have_last && t <= last_kept) {
            drop_idx.push_back(i);
        } else {
            last_kept = t;
            have_last = true;
        }
    }
    if (drop_idx.empty()) {
        return false;
    }
    std::sort(drop_idx.begin(), drop_idx.end(), std::greater<int>());
    for (int idx : drop_idx) {
        beat_rows.erase(beat_rows.begin() + idx);
    }
    note = "beats: dropped " + std::to_string(drop_idx.size()) + " non-increasing-time row" +
           (drop_idx.size() == 1 ? "" : "s");
    return true;
}

// Family (c). Matches fill_intervals()'s error text exactly ("Interval " +
// fmt6(start) + "-" + fmt6(end) + " (" + value + ") is shorter than the ABC
// subbeat grid"). Tries key_rows first (fill_intervals is called for keys
// before chords in yue2_sheet_notation_generate()), then chord_rows, then
// structure_rows (never actually raised for structure per pin doc §7.3, kept
// for symmetry/future-proofing only).
inline bool repair_short_interval(const std::string & error, std::vector<Yue2SheetAbcRow> & chord_rows,
                                   std::vector<Yue2SheetAbcRow> & key_rows,
                                   std::vector<Yue2SheetAbcRow> & structure_rows, std::string & note) {
    static const std::regex re(R"(^Interval ([0-9.]+)-([0-9.]+) \((.*)\) is shorter than the ABC subbeat grid$)");
    std::smatch m;
    if (!std::regex_search(error, m, re)) {
        return false;
    }
    double start = 0.0, end = 0.0;
    if (!parse_double(m[1].str(), start) || !parse_double(m[2].str(), end)) {
        return false;
    }
    const std::string value = m[3].str();

    auto try_drop = [&](std::vector<Yue2SheetAbcRow> & rows, const char * field_name) -> bool {
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].size() < 3) {
                continue;
            }
            double rs = 0.0, re_ = 0.0;
            if (!parse_double(rows[i][0], rs) || !parse_double(rows[i][1], re_)) {
                continue;
            }
            if (std::fabs(rs - start) < 1e-3 && std::fabs(re_ - end) < 1e-3 && rows[i][2] == value) {
                rows.erase(rows.begin() + (long) i);
                note = std::string(field_name) + ": dropped short interval " + std::to_string(start) + "-" +
                       std::to_string(end) + " (" + value + ")";
                return true;
            }
        }
        return false;
    };
    if (try_drop(key_rows, "keys")) {
        return true;
    }
    if (try_drop(chord_rows, "chords")) {
        return true;
    }
    if (try_drop(structure_rows, "structures")) {
        return true;
    }
    return false;
}

// Family (d). Matches notes_to_arr()'s error text exactly (voice_id + ":
// MIDI note pitch=" + pitch + " at " + fmt6(start) + "-" + fmt6(end) + "
// cannot be represented on the decoded subbeat grid"). The note's start/end
// here are the MIDI-round-tripped (tick-quantized) times, which can differ
// slightly from the raw pre-MIDI Yue2SheetRawNote this function mutates — so
// match by exact pitch + nearest start/end rather than requiring equality.
inline bool repair_ungriddable_note(const std::string & error, std::vector<Yue2SheetRawNote> & vocal_notes,
                                     std::vector<Yue2SheetRawNote> & ins_notes, std::string & note) {
    static const std::regex re(
        R"(^(Vocal|Ins): MIDI note pitch=(\d+) at ([0-9.]+)-([0-9.]+) cannot be represented on the decoded subbeat grid$)");
    std::smatch m;
    if (!std::regex_search(error, m, re)) {
        return false;
    }
    const std::string voice = m[1].str();
    const int         pitch = std::atoi(m[2].str().c_str());
    double            start = 0.0, end = 0.0;
    if (!parse_double(m[3].str(), start) || !parse_double(m[4].str(), end)) {
        return false;
    }
    std::vector<Yue2SheetRawNote> & notes = (voice == "Vocal") ? vocal_notes : ins_notes;
    int                              best = -1;
    double                           best_dist = 1e18;
    for (size_t i = 0; i < notes.size(); ++i) {
        if (notes[i].pitch != pitch) {
            continue;
        }
        const double dist = std::fabs(notes[i].start - start) + std::fabs(notes[i].end - end);
        if (dist < best_dist) {
            best_dist = dist;
            best      = (int) i;
        }
    }
    if (best < 0) {
        return false;
    }
    notes.erase(notes.begin() + best);
    note = "melody: dropped 1 " + voice + " note (pitch " + std::to_string(pitch) + ")";
    return true;
}

}  // namespace yue2_sheet_repair_detail

// Renders once; on failure, tries the four families above against the
// CURRENT error text, in order, applying the first one that matches and
// re-rendering, for up to kMaxRepairRounds rounds (a family can recur — e.g.
// two separate non-consecutive-beat-ID measures in one source). Stops early
// the first round no family matches. `inputs` is consumed (mutated in place
// across rounds; pass a copy if the caller still needs the originals).
//
// `outcome.repaired` is empty iff no round ever changed anything (in
// particular: always empty when the very first render already succeeds, the
// task's own contract for the field). It is populated whenever at least one
// repair was actually applied, even if every subsequent attempt still fails
// — a caller that only wants to know "did the FINAL render succeed" should
// still check `outcome.result.ok`; `repaired` is "what did we try", not "did
// it end up working".
inline Yue2SheetRepairOutcome yue2_sheet_notation_repair_and_generate(Yue2SheetRepairInputs inputs,
                                                                       bool melody_only = false) {
    using namespace yue2_sheet_repair_detail;
    Yue2SheetRepairOutcome outcome;

    auto build_midi = [&]() -> std::string {
        std::vector<uint8_t> midi = yue2_sheet_write_midi(inputs.vocal_notes, inputs.ins_notes);
        return std::string(midi.begin(), midi.end());
    };

    std::string melody_midi_bytes = build_midi();
    outcome.result = yue2_sheet_notation_generate(melody_midi_bytes, inputs.beat_rows, inputs.chord_rows,
                                                   inputs.key_rows, inputs.structure_rows, melody_only);
    if (outcome.result.ok) {
        return outcome;  // first render already succeeded: repaired stays "" (task contract)
    }

    std::vector<std::string> notes;
    for (int round = 0; round < kMaxRepairRounds; ++round) {
        const std::string & error = outcome.result.error;
        std::string          note;
        const bool changed = repair_dup_beat_id(error, inputs.beat_rows, note) ||
                              repair_non_increasing_time(error, inputs.beat_rows, note) ||
                              repair_short_interval(error, inputs.chord_rows, inputs.key_rows,
                                                     inputs.structure_rows, note) ||
                              repair_ungriddable_note(error, inputs.vocal_notes, inputs.ins_notes, note);
        if (!changed) {
            break;  // outside the four families: remains a soft failure, exactly as before this file existed
        }
        notes.push_back(note);
        melody_midi_bytes = build_midi();
        outcome.result = yue2_sheet_notation_generate(melody_midi_bytes, inputs.beat_rows, inputs.chord_rows,
                                                       inputs.key_rows, inputs.structure_rows, melody_only);
        if (outcome.result.ok) {
            break;
        }
    }

    if (!notes.empty()) {
        std::string joined;
        for (size_t i = 0; i < notes.size(); ++i) {
            if (i) {
                joined += "; ";
            }
            joined += notes[i];
        }
        outcome.repaired = joined;
    }
    return outcome;
}
