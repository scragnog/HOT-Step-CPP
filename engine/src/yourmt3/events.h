#pragma once
// yourmt3/events.h: YourMT3+ token ids to notes in seconds, as MuLaCover's
// melody transcriber does it (YourMT3 utils/event_codec.py, event2note.py and
// note2event.py, Apache-2.0; see LICENSE). All integer work and float64 times:
//   1. ids to events with the mt3 codec of task mc13_full_plus_256
//      (PAD, EOS, UNK, 206 shifts, 128 pitches, 2 velocities, tie,
//      128 programs, 128 drums: 596 ids);
//   2. per segment, the tie section (notes still sounding from the segment
//      before) and the timed events, ticks at 100 per second from the
//      segment's start;
//   3. per channel, the segments merged in order: a note left sounding by one
//      segment that the next segment's tie section does not carry gets a
//      note-off at the next segment's start time;
//   4. note events to notes (fix long notes, short notes, overlaps), then the
//      13 channels mixed into one list.
// Upstream counts decoding errors and drops the offending event; so does this.
// Every sort is stable, as Python's is, so equal keys keep upstream's order.

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace yourmt3 {

enum EvType { EV_PAD, EV_EOS, EV_UNK, EV_SHIFT, EV_PITCH, EV_VELOCITY, EV_TIE, EV_PROGRAM, EV_DRUM };
static const char * const EV_NAME[] = { "PAD", "EOS", "UNK", "shift", "pitch", "velocity", "tie", "program", "drum" };
constexpr int    N_SHIFT = 206, TPS = 100, VOCAL_PROGRAM = 100, DRUM_PRG = 128;
constexpr double MIN_OFFSET = 0.01;  // MINIMUM_OFFSET_SEC / MINIMUM_OFFSET_TIME

struct Event { int type, value; };

// codec.decode_event_index; ids outside the vocabulary are UNK-free errors upstream, never emitted by argmax
static Event decode_event(int id) {
    static const struct { int type, n; } R[] = { { EV_PAD, 1 },   { EV_EOS, 1 },      { EV_UNK, 1 },      { EV_SHIFT, N_SHIFT },
                                                 { EV_PITCH, 128 }, { EV_VELOCITY, 2 }, { EV_TIE, 1 },      { EV_PROGRAM, 128 },
                                                 { EV_DRUM, 128 } };
    for (const auto & r : R) {
        if (id < r.n) return { r.type, id };
        id -= r.n;
    }
    return { -1, id };
}

struct NoteEvent {
    bool   is_drum;
    int    program;
    double time;  // NaN for tie note events
    int    velocity, pitch;
};

struct Note {
    bool   is_drum;
    int    program;
    double onset, offset;
    int    pitch, velocity;
};

struct SegmentEvents {
    std::vector<NoteEvent>           events, ties;
    std::vector<std::pair<int, int>> last_activity;  // (program, pitch), a list as upstream keeps it
    double                           start_time = 0;
    int                              errors     = 0;
};

// Segment start in seconds, as upstream computes it: index * frames / sample_rate.
static double segment_start(int index, int frames, int sr) { return (double) ((long long) index * frames) / (double) sr; }

// event2note_event(events, start_time, sort=True, tps=100)
static SegmentEvents segment_events(const int * tok, int n, double start_time) {
    SegmentEvents s;
    s.start_time = start_time;
    int program = -1, velocity = -1, tie_at = -1;
    for (int i = 0; i < n; i++) {
        const Event e = decode_event(tok[i]);
        if (e.type == EV_TIE) { tie_at = i; break; }
        if (e.type == EV_SHIFT) break;
        if (e.type == EV_PROGRAM) program = e.value;
        else if (e.type == EV_PITCH) {
            if (program < 0) { s.errors++; continue; }  // Err/Missing prg in tie
            s.ties.push_back({ false, program, NAN, 1, e.value });
            s.last_activity.push_back({ program, e.value });
        }
    }
    if (tie_at < 0) {  // Err/Missing tie: nothing from this segment, activity included
        s.ties.clear();
        s.last_activity.clear();
        s.errors++;
        return s;
    }
    // Python round(): nearest, ties to even (nearbyint in the default rounding mode)
    const long long start_tick = (long long) std::nearbyint(start_time * TPS);
    long long       tick       = start_tick;
    for (int i = tie_at + 1; i < n; i++) {
        const Event e = decode_event(tok[i]);
        switch (e.type) {
            case EV_SHIFT:
                if (e.value <= 0 || e.value > 1000) s.errors++;
                else tick = start_tick + e.value;
                break;
            case EV_DRUM: s.events.push_back({ true, DRUM_PRG, (double) tick / TPS, 1, e.value }); break;
            case EV_PROGRAM: program = e.value; break;
            case EV_VELOCITY: velocity = e.value; break;
            case EV_PITCH: {
                if (program < 0 || velocity < 0) { s.errors++; break; }
                const std::pair<int, int> key = { program, e.value };
                if (velocity > 0) s.last_activity.push_back(key);
                else {
                    auto it = std::find(s.last_activity.begin(), s.last_activity.end(), key);
                    if (it == s.last_activity.end()) { s.errors++; break; }  // note off without note on
                    s.last_activity.erase(it);
                }
                s.events.push_back({ false, program, (double) tick / TPS, velocity, e.value });
                break;
            }
            case EV_EOS: i = n; break;
            case EV_PAD:
            case EV_UNK: break;
            default: s.errors++; break;  // a second tie, or an unknown id
        }
    }
    std::stable_sort(s.events.begin(), s.events.end(), [](const NoteEvent & a, const NoteEvent & b) {
        if (a.time != b.time) return a.time < b.time;
        if (a.is_drum != b.is_drum) return a.is_drum < b.is_drum;
        if (a.program != b.program) return a.program < b.program;
        if (a.velocity != b.velocity) return a.velocity < b.velocity;
        return a.pitch < b.pitch;
    });
    std::stable_sort(s.ties.begin(), s.ties.end(), [](const NoteEvent & a, const NoteEvent & b) {
        if (a.is_drum != b.is_drum) return a.is_drum < b.is_drum;
        if (a.program != b.program) return a.program < b.program;
        return a.pitch < b.pitch;
    });
    return s;
}

static bool note_less(const Note & a, const Note & b) {  // (onset, is_drum, program, velocity, pitch)
    if (a.onset != b.onset) return a.onset < b.onset;
    if (a.is_drum != b.is_drum) return a.is_drum < b.is_drum;
    if (a.program != b.program) return a.program < b.program;
    if (a.velocity != b.velocity) return a.velocity < b.velocity;
    return a.pitch < b.pitch;
}

// validate_notes(fix=True)
static void validate_notes(std::vector<Note> & notes) {
    for (Note & n : notes) {
        if (n.onset > n.offset) n.offset = std::max(n.offset, n.onset + MIN_OFFSET);
        else if (!n.is_drum && n.offset - n.onset < 0.01) n.offset = n.onset + MIN_OFFSET;
    }
}

// trim_overlapping_notes(sort=True). Per (pitch, program, is_drum), sorted by
// onset: an offset past the next onset is cut back to it, zero-length notes go.
// Equal sort keys only ever meet inside one group, so the group order upstream
// takes from a Python set cannot show in the result.
static std::vector<Note> trim_overlapping(const std::vector<Note> & notes) {
    if (notes.size() <= 1) return notes;
    std::vector<std::vector<Note>> groups;
    for (const Note & n : notes) {
        auto g = std::find_if(groups.begin(), groups.end(), [&](const std::vector<Note> & v) {
            return v[0].pitch == n.pitch && v[0].program == n.program && v[0].is_drum == n.is_drum;
        });
        if (g == groups.end()) groups.push_back({ n });
        else g->push_back(n);
    }
    std::vector<Note> out;
    for (auto & g : groups) {
        std::stable_sort(g.begin(), g.end(), [](const Note & a, const Note & b) { return a.onset < b.onset; });
        for (size_t i = 1; i < g.size(); i++)
            if (g[i - 1].offset > g[i].onset) g[i - 1].offset = g[i].onset;
        for (const Note & n : g)
            if (n.onset < n.offset) out.push_back(n);
    }
    std::stable_sort(out.begin(), out.end(), note_less);
    return out;
}

// note_event2note(events, tie_note_events=None, sort, fix_offset, trim_overlap)
static std::vector<Note> events_to_notes(std::vector<NoteEvent> evs, int * errors) {
    std::stable_sort(evs.begin(), evs.end(), [](const NoteEvent & a, const NoteEvent & b) {
        if (a.time != b.time) return a.time < b.time;
        if (a.is_drum != b.is_drum) return a.is_drum < b.is_drum;
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        if (a.velocity != b.velocity) return a.velocity < b.velocity;
        return a.program < b.program;
    });
    std::vector<Note>      notes;
    std::vector<NoteEvent> active;  // an insertion-ordered dict keyed (pitch, program)
    auto find = [&](const NoteEvent & e) {
        return std::find_if(active.begin(), active.end(), [&](const NoteEvent & a) { return a.pitch == e.pitch && a.program == e.program; });
    };
    for (const NoteEvent & e : evs) {
        if (e.is_drum) {
            if (e.velocity == 1) notes.push_back({ true, DRUM_PRG, e.time, e.time + MIN_OFFSET, e.pitch, 1 });
        } else if (e.velocity == 1) {
            auto it = find(e);
            if (it != active.end()) {
                notes.push_back({ false, it->program, it->time, e.time, it->pitch, it->velocity });
                active.erase(it);
            }
            active.push_back(e);
        } else if (e.velocity == 0) {
            auto it = find(e);
            if (it == active.end()) { (*errors)++; continue; }  // Err/onset not found
            notes.push_back({ false, it->program, it->time, e.time, it->pitch, it->velocity });
            active.erase(it);
        }
    }
    for (const NoteEvent & a : active)
        if (a.velocity == 1) notes.push_back({ false, a.program, a.time, a.time + MIN_OFFSET, a.pitch, 1 });
    for (Note & n : notes)
        if (n.offset - n.onset > 10) n.offset = n.onset + MIN_OFFSET;  // Err/long note > 10s
    std::stable_sort(notes.begin(), notes.end(), note_less);
    validate_notes(notes);
    return trim_overlapping(notes);
}

// merge_zipped_note_events_and_ties_to_notes: one channel's segments in order
static std::vector<Note> merge_segments(const std::vector<SegmentEvents> & segs, int * errors) {
    std::vector<NoteEvent> merged;
    for (size_t k = 0; k < segs.size(); k++) {
        const SegmentEvents & s = segs[k];
        *errors += s.errors;
        if (k > 0)
            for (const auto & pla : segs[k - 1].last_activity) {
                const bool tied = std::any_of(s.ties.begin(), s.ties.end(),
                                              [&](const NoteEvent & t) { return t.program == pla.first && t.pitch == pla.second; });
                if (!tied) {
                    merged.push_back({ false, pla.first, s.start_time, 0, pla.second });
                    (*errors)++;  // Err/merging segment tie
                }
            }
        merged.insert(merged.end(), s.events.begin(), s.events.end());
    }
    return events_to_notes(std::move(merged), errors);
}

// mix_notes: channels chained, sorted with offset as the last key, trimmed, validated
static std::vector<Note> mix_notes(const std::vector<std::vector<Note>> & channels) {
    std::vector<Note> all;
    for (const auto & c : channels) all.insert(all.end(), c.begin(), c.end());
    std::stable_sort(all.begin(), all.end(), [](const Note & a, const Note & b) {
        if (note_less(a, b)) return true;
        if (note_less(b, a)) return false;
        return a.offset < b.offset;
    });
    all = trim_overlapping(all);
    validate_notes(all);
    return all;
}

// tokens[seg * n_ch + ch], each stream of any length: every channel's segments
// merged, then the channels mixed. Returns the decoding error count.
static int tokens_to_notes(const std::vector<std::vector<int>> & tokens, int n_seg, int n_ch, int frames, int sr,
                           std::vector<Note> & out) {
    int errors = 0;
    std::vector<std::vector<Note>> channels(n_ch);
    for (int ch = 0; ch < n_ch; ch++) {
        std::vector<SegmentEvents> segs;
        for (int k = 0; k < n_seg; k++) {
            const std::vector<int> & t = tokens[(size_t) k * n_ch + ch];
            segs.push_back(segment_events(t.data(), (int) t.size(), segment_start(k, frames, sr)));
        }
        channels[ch] = merge_segments(segs, &errors);
    }
    out = mix_notes(channels);
    return errors;
}

// upstream's selections (MuLaCover symbolic.py): melody is program 100 and not a drum
static std::vector<Note> select_vocal(const std::vector<Note> & n) {
    std::vector<Note> v;
    for (const Note & x : n)
        if (!x.is_drum && x.program == VOCAL_PROGRAM) v.push_back(x);
    return v;
}
static std::vector<Note> select_drums(const std::vector<Note> & n) {
    std::vector<Note> v;
    for (const Note & x : n)
        if (x.is_drum) v.push_back(x);
    return v;
}

}  // namespace yourmt3
