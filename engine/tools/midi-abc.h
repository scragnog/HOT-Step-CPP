#pragma once
// midi-abc.h: a MuScriptor transcription as a YuE2 two-voice ABC lead sheet
//
// Feeds SheetSage2's own notation stage (yue2/sheetsage-notation.h via
// sheetsage-repair.h), the writer YuE2 cover scores come from, with:
//   beats   the resolved beat grid (midi-grid.h), beat 1 on each downbeat
//   Vocal   MuScriptor's "voice" notes; Ins one melodic instrument (the
//           busiest, or the caller's choice), each made monophonic (top note)
//   key     Krumhansl-Schmuckler on the duration-weighted pitch classes
//   chords  per half bar, triad/seventh templates on the other instruments,
//           or the caller's chord segments (ChordNet) snapped to the same grid
// Notes are snapped to the notation's 4-per-beat subbeat grid first, so the
// renderer never meets a note it cannot place. Drums and other instruments
// are dropped: YuE2's ABC has two voices.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "midi-grid.h"
#include "yue2/sheetsage-repair.h"

struct AbcNote {
    double onset, offset;  // seconds of the audio
    int    pitch, program;
    bool   drum;
};

// A chord segment in seconds of the audio, "Root:quality[/degree]" (ChordNet's labels)
struct AbcChord {
    std::string label;
    double      onset, offset;
};

struct AbcResult {
    bool        ok = false;
    std::string abc, error, repaired, key;
    int         vocal_notes = 0, ins_notes = 0, ins_program = -1, chords = 0;
};

#define ABC_VOICE_PROGRAM 52

// lead-sheet Ins candidates: not drums, the voice or bass
static bool abc_melodic(int program, int vocal_program = ABC_VOICE_PROGRAM) {
    return program != vocal_program && !(program >= 32 && program <= 39) && program != 43 && program < 128;
}

static const char * ABC_SHARP[12] = { "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B" };
static const char * ABC_FLAT[12]  = { "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B" };

// Krumhansl-Kessler profiles
static const double ABC_MAJOR[12] = { 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88 };
static const double ABC_MINOR[12] = { 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17 };

static double abc_corr(const double * a, const double * b) {
    double ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < 12; i++) { ma += a[i] / 12; mb += b[i] / 12; }
    for (int i = 0; i < 12; i++) {
        sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb);
    }
    return saa > 0 && sbb > 0 ? sab / sqrt(saa * sbb) : 0;
}

// "Bb:major" / "F#:minor"; *flats tells the chord spelling which way to lean
static std::string abc_key(const double * chroma, bool * flats) {
    int    best = 0;
    bool   minor = false;
    double best_r = -2;
    for (int root = 0; root < 12; root++) {
        double rot[12];
        for (int i = 0; i < 12; i++) rot[i] = chroma[(root + i) % 12];
        double r = abc_corr(rot, ABC_MAJOR);
        if (r > best_r) { best_r = r; best = root; minor = false; }
        r = abc_corr(rot, ABC_MINOR);
        if (r > best_r) { best_r = r; best = root; minor = true; }
    }
    // flat keys: F Bb Eb Ab Db Gb major, D G C F Bb Eb minor
    const int major_root = minor ? (best + 3) % 12 : best;
    *flats = major_root == 5 || major_root == 10 || major_root == 3 || major_root == 8 || major_root == 1 || major_root == 6;
    return std::string(*flats ? ABC_FLAT[best] : ABC_SHARP[best]) + (minor ? ":minor" : ":major");
}

// Best chord label for a chroma vector, "N" when too little is sounding.
static std::string abc_chord(const double * chroma, double bass_pc_weight, int bass_pc, bool flats) {
    struct Q { const char * name; int iv[4]; int n; };
    static const Q QS[] = {
        { "maj", { 0, 4, 7 }, 3 }, { "min", { 0, 3, 7 }, 3 }, { "7", { 0, 4, 7, 10 }, 4 },
        { "maj7", { 0, 4, 7, 11 }, 4 }, { "min7", { 0, 3, 7, 10 }, 4 }, { "dim", { 0, 3, 6 }, 3 },
    };
    double total = 0;
    for (int i = 0; i < 12; i++) total += chroma[i];
    if (total <= 1e-9) return "N";
    double best = -1;
    int    best_root = 0;
    const Q * best_q = &QS[0];
    for (int root = 0; root < 12; root++) {
        for (const Q & q : QS) {
            double in = 0;
            for (int k = 0; k < q.n; k++) in += chroma[(root + q.iv[k]) % 12];
            // share of the energy on chord tones, a small bonus for fewer tones
            // (so a plain triad wins unless the seventh is really there) and
            // for the root being in the bass
            double s = in / total - 0.04 * (q.n - 3) + (root == bass_pc ? 0.15 * bass_pc_weight : 0);
            if (s > best) { best = s; best_root = root; best_q = &q; }
        }
    }
    if (best < 0.6) return "N";
    return std::string(flats ? ABC_FLAT[best_root] : ABC_SHARP[best_root]) + ":" + best_q->name;
}

// A ChordNet label as abc_chord writes labels: "Root:quality", the root spelt
// for the key, an inversion's bass as a note name. The qualities the server's
// lead-sheet parser cannot read are reduced to the chord tones it can:
// 9, 11, 13 -> 7; maj9 -> maj7; min9 -> min7; hdim7, dim7 -> dim; sus4(b7) -> sus4.
// "N", or anything unreadable, is "N".
static std::string abc_chordnet_label(const std::string & label, bool flats) {
    static const char * KEEP[] = { "maj", "min", "7", "maj7", "min7", "dim", "aug", "sus2", "sus4" };
    static const struct { const char * from, * to; } REDUCE[] = {
        { "9", "7" }, { "11", "7" }, { "13", "7" }, { "maj9", "maj7" }, { "min9", "min7" },
        { "hdim7", "dim" }, { "dim7", "dim" }, { "sus4(b7)", "sus4" },
    };
    static const struct { const char * deg; int semis; } BASS[] = { { "2", 2 }, { "b3", 3 }, { "3", 4 }, { "5", 7 }, { "b7", 10 } };
    static const int LETTER[7] = { 9, 11, 0, 2, 4, 5, 7 };  // A..G
    const size_t colon = label.find(':');
    if (colon == std::string::npos || colon < 1 || colon > 2 || label[0] < 'A' || label[0] > 'G') return "N";
    int root = LETTER[label[0] - 'A'];
    if (colon == 2) {
        if (label[1] == '#') root++;
        else if (label[1] == 'b') root--;
        else return "N";
    }
    root = (root + 12) % 12;
    std::string q = label.substr(colon + 1), bass;
    const size_t slash = q.find('/');
    if (slash != std::string::npos) {
        bass = q.substr(slash + 1);
        q    = q.substr(0, slash);
    }
    bool known = false;
    for (const char * k : KEEP) known |= q == k;
    for (const auto & r : REDUCE)
        if (q == r.from) { q = r.to; known = true; }
    if (!known) return "N";
    const char * const * names = flats ? ABC_FLAT : ABC_SHARP;
    std::string out = std::string(names[root]) + ":" + q;
    if (!bass.empty()) {
        for (const auto & b : BASS)
            if (bass == b.deg) return out + "/" + names[(root + b.semis) % 12];
        return "N";
    }
    return out;
}

// `grid` is the resolved grid the MIDI was written on (onset delay included);
// ins_program < 0 picks the busiest melodic instrument. vocal_program is the
// program of the Vocal voice. chords, when given, replace the template chord
// estimate: each segment's ends are snapped to the subbeat grid.
static AbcResult midi_to_yue2_abc(const std::vector<AbcNote> & notes_in, const BeatGrid & grid, double duration,
                                  int ins_program, bool melody_only, int vocal_program = ABC_VOICE_PROGRAM,
                                  const std::vector<AbcChord> * chords = nullptr) {
    AbcResult out;
    const double delay = grid.onset_delay;
    std::vector<AbcNote> notes;
    for (AbcNote n : notes_in) {
        if (n.drum) continue;
        n.onset -= delay;
        n.offset -= delay;
        notes.push_back(n);
    }
    if (notes.empty()) { out.error = "No pitched notes to write"; return out; }
    double first = INFINITY, last = 0;
    for (auto & n : notes) { first = std::min(first, n.onset); last = std::max(last, n.offset); }

    // beats over the whole song, beat 1 on the downbeats
    const double period = 60.0 / grid.bpm;
    std::vector<double> beats = grid.beats;
    if (beats.size() < 2) {  // a typed-in tempo: lay its beats out from the downbeat
        beats.clear();
        for (double t = grid.first_downbeat - period * ceil((grid.first_downbeat - std::min(first, 0.0)) / period);
             t < std::max(duration, last) + period; t += period) beats.push_back(t);
    }
    while (beats.front() > first + 1e-6) beats.insert(beats.begin(), beats.front() - (beats[1] - beats[0]));
    while (beats.back() < last - 1e-6) beats.push_back(beats.back() + (beats.back() - beats[beats.size() - 2]));
    const int bpb  = grid.beats_per_bar > 0 ? grid.beats_per_bar : 4;
    const int down = (int) grid_nearest(beats, grid.first_downbeat);
    std::vector<Yue2SheetBeatRow> rows;
    for (size_t i = 0; i < beats.size(); i++) {
        rows.push_back({ beats[i], (int) grid_pymod((double) i - down, bpb) + 1, bpb, 4 });
    }
    std::vector<std::array<double, 2>> spans;
    for (auto & n : notes) spans.push_back({ n.onset, n.offset });
    std::vector<Yue2SheetBeatRow> padded;
    if (!yue2_sheet_pad_beats(rows, std::max(duration - delay, last), spans, &padded, &out.error)) return out;

    // the notation's subbeat boundaries: 4 per beat, linear within each beat
    std::vector<double> sub;
    for (size_t i = 0; i + 1 < padded.size(); i++) {
        for (int k = 0; k < 4; k++) sub.push_back(padded[i].time + k * (padded[i + 1].time - padded[i].time) / 4.0);
    }
    sub.push_back(padded.back().time);
    auto snap = [&](double t) {
        size_t i = std::lower_bound(sub.begin(), sub.end(), t) - sub.begin();
        if (i == 0) return sub.front();
        if (i == sub.size()) return sub.back();
        return (t - sub[i - 1] <= sub[i] - t) ? sub[i - 1] : sub[i];  // ties to the earlier one
    };

    // Ins: the busiest melodic instrument unless one was asked for
    if (ins_program < 0) {
        std::vector<int> count(128, 0);
        for (auto & n : notes) if (abc_melodic(n.program, vocal_program)) count[n.program]++;
        int best = (int) (std::max_element(count.begin(), count.end()) - count.begin());
        ins_program = count[best] > 0 ? best : -1;
    }
    out.ins_program = ins_program;

    std::vector<Yue2SheetRawNote> raw;
    for (auto & n : notes) {
        const int track = n.program == vocal_program ? 0 : n.program == ins_program ? 1 : -1;
        if (track < 0) continue;
        double s = snap(n.onset), e = snap(n.offset);
        if (e > s + 1e-9) raw.push_back({ s, e, n.pitch, track });
    }
    std::vector<Yue2SheetRawNote> clean = yue2_sheet_notation_notes(raw);  // monophonic, top note kept
    Yue2SheetRepairInputs in;
    for (auto & n : clean) (n.track == 0 ? in.vocal_notes : in.ins_notes).push_back(n);
    out.vocal_notes = (int) in.vocal_notes.size();
    out.ins_notes   = (int) in.ins_notes.size();

    for (auto & r : padded) {
        in.beat_rows.push_back({ yue2_sheet_cell_text(Yue2SheetCell::Real(r.time)), std::to_string(r.beat_id),
                                 std::to_string(r.numerator), std::to_string(r.denominator) });
    }
    const double t0 = padded.front().time, t1 = padded.back().time;
    auto cell = [](double v) { return yue2_sheet_cell_text(Yue2SheetCell::Real(v)); };

    // key over everything pitched, weighted by duration
    double chroma[12] = { 0 };
    for (auto & n : notes) chroma[((n.pitch % 12) + 12) % 12] += std::min(n.offset - n.onset, 4.0);
    bool flats = false;
    out.key = abc_key(chroma, &flats);
    in.key_rows.push_back({ cell(t0), cell(t1), out.key });

    // chords per half bar (a whole bar in odd meters) from the accompaniment
    if (!melody_only && chords) {
        for (const AbcChord & c : *chords) {
            const double s = std::max(t0, std::min(t1, snap(c.onset - delay)));
            const double e = std::max(t0, std::min(t1, snap(c.offset - delay)));
            if (e <= s) continue;
            const std::string label = abc_chordnet_label(c.label, flats);
            if (in.chord_rows.empty() && s > t0) in.chord_rows.push_back({ cell(t0), cell(s), "N" });
            if (!in.chord_rows.empty() && in.chord_rows.back()[2] == label) in.chord_rows.back()[1] = cell(e);
            else in.chord_rows.push_back({ cell(s), cell(e), label });
        }
        if (in.chord_rows.empty()) in.chord_rows.push_back({ cell(t0), cell(t1), "N" });
        else if (in.chord_rows.back()[1] != cell(t1)) in.chord_rows.push_back({ in.chord_rows.back()[1], cell(t1), "N" });
        for (auto & r : in.chord_rows) if (r[2] != "N") out.chords++;
    } else if (!melody_only) {
        const size_t span = bpb % 2 == 0 ? bpb / 2 : bpb;
        // spans start on the downbeats; the pickup before the first is one span
        std::vector<size_t> starts = { 0 };
        for (size_t b = (padded[0].beat_id - 1 == 0 ? 0 : (size_t) (bpb - padded[0].beat_id + 1)); b + 1 < padded.size(); b += span) {
            if (b > 0) starts.push_back(b);
        }
        starts.push_back(padded.size() - 1);
        std::string prev;
        for (size_t k = 0; k + 1 < starts.size(); k++) {
            const double s = padded[starts[k]].time, e = padded[starts[k + 1]].time;
            double c[12] = { 0 }, low_w = 0;
            int    low = 128;
            for (auto & n : notes) {
                if (n.program == vocal_program) continue;
                const double ov = std::min(e, n.offset) - std::max(s, n.onset);
                if (ov <= 0) continue;
                c[((n.pitch % 12) + 12) % 12] += ov;
                if (n.pitch < low) { low = n.pitch; low_w = 1; }
            }
            std::string label = abc_chord(c, low_w, low < 128 ? low % 12 : -1, flats);
            if (!in.chord_rows.empty() && label == prev) in.chord_rows.back()[1] = cell(e);
            else in.chord_rows.push_back({ cell(s), cell(e), label });
            prev = label;
        }
        for (auto & r : in.chord_rows) if (r[2] != "N") out.chords++;
    }

    Yue2SheetRepairOutcome res = yue2_sheet_notation_repair_and_generate(in, melody_only);
    out.ok       = res.result.ok;
    out.abc      = res.result.abc;
    out.error    = res.result.error;
    out.repaired = res.repaired;
    return out;
}
