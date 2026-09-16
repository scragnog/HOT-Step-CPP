#pragma once
// yue2/sheetsage-stitch.h — SheetSage2 sliding-window plan, the cross-window
// overlap prefix, and whole-song event stitching. This is the pure-logic
// pipeline lane: no GGML, no tensors, no I/O of its own.
//
// HOT-Step file (no acestep.cpp analog). A literal port of four Python
// functions in `generation_sheetsage2.py`/`pipeline_sheetsage2.py`:
//   - sliding_window_plan            (pipeline_sheetsage2.py:32-51)
//   - build_overlap_prefix_tokens    (generation_sheetsage2.py:435-512)
//     + its helpers: active_context_before (:398-416), apply_prefix_context
//       (:419-432), set_event_local_timestamp (:392-395), a decoded event's
//       own refresh-from-tokens step, subbeat_shift_to_tokens (already in
//       sheetsage-tokens.h), and encode_decoded_sequence
//       (tokenization_sheetsage2.py:647-663)
//   - stitched_window_events         (generation_sheetsage2.py:515-560)
//
// AUTHORITY: docs/plans/yue2/21-sheetsage2-symbolic-pin.md §4 (this file's
// whole scope). Where this file and that document disagree, the document
// wins (and the Python under it wins over the document — one confirmed gap
// is noted at yue2_sheet_stitch_window() below: doc 21 §4.3's pseudocode
// never restates the `eps` value for its own accept-window comparison, only
// §4.2's `build_overlap_prefix_tokens` does (`eps = 1e-4`). Both functions
// live in the same Python file and compare the same kind of quantity (a
// window-local/absolute-time difference against a `1e-4`-scale window
// boundary), so this port reuses `1e-4` for both — cross-checked empirically
// against every fixture's `window_plan[i].accepted_events` count in
// `engine/tests/sheetsage/test_stitch.cpp`; report to Rob/the next agent if
// that ever needs to be pinned to the literal source instead of inferred).
//
// Depends on sheetsage-tokens.h (vocabulary, subbeat_shift_to_tokens) and
// sheetsage-events.h (Yue2SheetEvent/Yue2SheetDecoded, decode_field,
// normalize_prompts, event_time_map, the stitched-events JSON reader) — both
// already complete; this file adds no private shim for anything they already
// provide. It does NOT decode a raw token stream itself: window-level event
// lists are supplied by the caller (from sheetsage-events.h's decode, or —
// as the standalone test does — loaded directly from a window's own
// `12_events.json`, doc 22's fixture tree), and this file's own output is
// itself another list of (already-decoded) `Yue2SheetEvent`s, ready to feed
// export/notation.

#include "sheetsage-events.h"
#include "sheetsage-tokens.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════
// §1. sliding_window_plan (pipeline_sheetsage2.py:32-51, doc 21 §4.1).
// ═════════════════════════════════════════════════════════════════════════

struct Yue2SheetWindow {
    int32_t window_index = 0;
    double start = 0.0;
    double end = 0.0;
    double accept_start = 0.0;
    double accept_end = 0.0;
    // Always numerically equal to accept_start (doc 21 §4.2's note) — kept
    // as its own field for fidelity with the reference's own dict shape and
    // because build_overlap_prefix_tokens's signature takes it by that name.
    double prefix_end = 0.0;
    // None only for the last window (no next window to hand off to).
    bool has_generation_stop = false;
    double generation_stop = 0.0;  // window-LOCAL seconds (0 = window start)
};

// Builds the window plan for a song of `duration` seconds. Defaults are the
// "default" preset's only configuration this port uses (window=300s,
// overlap=200s, lookahead=100s — doc 19/21 throughout); the "paper" preset
// is a separate, unported mode (doc 21 §4.1's note). Returns false (with
// *err set) only on the reference's own precondition
// `0 <= lookahead_seconds <= overlap_seconds < window_seconds` — never
// exercised by any fixture (all use the defaults), included for fidelity to
// the Python's implicit assert.
inline bool yue2_sheet_sliding_window_plan(double duration, std::vector<Yue2SheetWindow> * out, std::string * err,
                                            double window_seconds = 300.0, double overlap_seconds = 200.0,
                                            double lookahead_seconds = 100.0) {
    if (!(lookahead_seconds >= 0.0 && lookahead_seconds <= overlap_seconds && overlap_seconds < window_seconds)) {
        if (err) *err = "sliding_window_plan: requires 0 <= lookahead_seconds <= overlap_seconds < window_seconds";
        return false;
    }
    const double hop = window_seconds - overlap_seconds;
    // pipeline_sheetsage2.py:32-51's own "last window" epsilon (doc 21 §4.1's
    // worked example resolves a 305.9s song's boundary check with exactly
    // this magnitude: `300 >= 305.899999` is False, `305.9 >= 305.899999` is
    // True).
    const double kLastWindowEps = 1e-6;

    double start = 0.0, accepted = 0.0;
    std::vector<Yue2SheetWindow> result;
    for (int32_t idx = 0;; ++idx) {
        const bool last = (start + window_seconds >= duration - kLastWindowEps);
        const double accept_end = last ? duration : (start + window_seconds - lookahead_seconds);

        Yue2SheetWindow w;
        w.window_index = idx;
        w.start = start;
        w.end = std::min(duration, start + window_seconds);
        w.accept_start = accepted;
        w.accept_end = accept_end;
        w.prefix_end = accepted;
        w.has_generation_stop = !last;
        w.generation_stop = last ? 0.0 : (window_seconds - lookahead_seconds);
        result.push_back(w);

        if (last) break;
        accepted = accept_end;
        start = std::min(start + hop, duration - window_seconds);
    }
    if (out) *out = std::move(result);
    return true;
}

// ═════════════════════════════════════════════════════════════════════════
// §2. Decoder-loop stop contract (doc 19/21 §3.4, doc 22 "Surprise 3"). Not
//     executable code — the actual masked-argmax decode loop lives in the
//     GGML pipeline lane, not here — but this is the exact contract that
//     lane must implement, and it is load-bearing for how §3/§4 fit
//     together, so it is pinned here rather than left to be rediscovered.
//
// A window's own decode loop runs against ONE `Yue2SheetWindow` (`w`):
//
//   1. Natural stop: the grammar-masked argmax picks <|eos|>
//      (`yue2_sheet_grammar_update` returns Finished). Always legal to check
//      first; happens on every window's own model-chosen ending, regardless
//      of `w.has_generation_stop`.
//
//   2. Stop-time cutoff — ONLY when `w.has_generation_stop` is true (i.e.
//      NOT the last window): after each ACCEPTED (post-mask) token is run
//      through `yue2_sheet_grammar_update`, check whether that token is a
//      Time-type token whose decoded seconds (`yue2_sheet_time_seconds`) is
//      `>= w.generation_stop`. If so, the loop does NOT run another forward
//      pass — it force-appends a SYNTHETIC `<|eos|>` directly to the output
//      token list and marks the sample finished, WITHOUT ever running that
//      synthetic `<|eos|>` through `yue2_sheet_grammar_update`. This is why
//      two of the four original fixtures (`short-a`/window-0000,
//      `long-a`/window-0000 — both non-last windows) have
//      `len(08_generated_ids) == decoder_steps_per_window + 1`: the trailing
//      id is the synthetic EOS with no captured decode step. The other two
//      (`short-b`/window-0000, `long-a`/window-0001 — both the LAST window
//      of their song, `has_generation_stop == false`) have no cutoff at all,
//      so if/when EOS appears it is a normal, captured decode step (case 1).
//      A window's own generated token count is therefore NOT reliably
//      `decoder_steps_per_window` or `+1` from `w.has_generation_stop`
//      alone — it also depends on whether the model happened to emit a Time
//      token past the cutoff before naturally choosing EOS itself; check the
//      actual token stream instead of assuming either shape (doc 22's own
//      warning, restated here since it is this contract's direct
//      consequence).
//
//   3. Length limit: regardless of 1/2, the loop stops once
//      `current_length >= 5120` (`max_output_seq_len`); `current_length`
//      starts at `len(prefix)` (the 8-token header plus, for window >= 1,
//      every overlap-prefix token from §4 below) — NOT at 0 — so a window
//      with a long overlap prefix has correspondingly less budget for newly
//      generated tokens. Not exercised by any fixture on disk (doc 21 §3.4).
//
// Once a window's own full token stream (prefix + generated [+ synthetic
// eos]) is decoded into events (sheetsage-events.h), every one of those
// events — INCLUDING the ones that came from the prefix, re-decoded — is
// handed to `yue2_sheet_stitch_window` below and subjected to the same
// accept-range filter as anything else; prefix-derived events are expected
// to fall before `w.accept_start` and be dropped there, not treated
// specially (doc 21 §4.2's note, confirmed on every multi-window fixture:
// e.g. `long-a` window-0001's own `12_events.json` starts with subbeat-0
// events reproducing its overlap prefix's own content, decoded again).
// ═════════════════════════════════════════════════════════════════════════

// ═════════════════════════════════════════════════════════════════════════
// §3. Overlap-prefix construction (generation_sheetsage2.py:392-512, doc 21
//     §4.2).
// ═════════════════════════════════════════════════════════════════════════

// round-half-to-even (Python's `round()`, doc 21 §8 item 1). `std::nearbyint`
// under the default FE_TONEAREST rounding mode is round-half-to-even on
// every platform this project targets (MSVC/x64, IEEE-754) — this is an
// explicit, named helper rather than a bare cast specifically because C++'s
// usual integer-cast truncation and `std::round`'s round-half-away-from-zero
// both disagree with Python on an exact `.5` tie (doc 21 §8 item 1's
// warning).
inline long long yue2_sheet_round_half_even(double x) { return (long long) std::nearbyint(x); }

// set_event_local_timestamp(event, tokenizer, local_time)
// (generation_sheetsage2.py:392-395). Overwrites the event's OWN timestamp
// token in place; caller must refresh .values afterward (this function only
// touches tokens_by_field, matching the reference's own split between this
// function and refresh_event_values).
inline void yue2_sheet_set_event_local_timestamp(Yue2SheetEvent & event, double local_time) {
    long long time_id = yue2_sheet_round_half_even(local_time * double(SheetSage2Tokens::TIME_HZ));
    time_id = std::max<long long>(0, std::min<long long>(SheetSage2Tokens::N_TIME_TOKENS - 1, time_id));
    event.tokens_by_field[size_t(Yue2SheetField::Timestamp)] =
        std::vector<int32_t>{int32_t(SheetSage2Tokens::TIME_TOKEN_START + time_id)};
}

// Recomputes an event's decoded `.values` (has_timestamp/timestamp,
// has_rhythm/..., has_structure/structure, ...) purely from its current
// `tokens_by_field`, discarding whatever was there before. This is the
// reference's `refresh_event_values(event, tokenizer, prompts)` — the
// `prompts` parameter is unused by `_decode_field` for schema v1 (doc 21
// §5.2's note, already applied the same way in sheetsage-events.h's
// `yue2_sheet_decode_field`), so it is not threaded through here either.
inline bool yue2_sheet_refresh_event_values(const SheetSage2Tokens & vocab, Yue2SheetEvent & event, std::string * err) {
    event.has_timestamp = false;
    event.timestamp = 0.0;
    event.has_rhythm = false;
    event.rhythm_has_meter = false;
    event.rhythm_meter_num = 0;
    event.rhythm_meter_den = 0;
    event.rhythm_has_eighth = false;
    event.rhythm_eighth = 0;
    event.has_structure = false;
    event.structure.clear();
    event.has_key = false;
    event.key.clear();
    event.has_chord = false;
    event.chord.clear();
    event.has_melody = false;
    event.melody.clear();
    for (int fi = 0; fi < 6; ++fi) {
        if (event.tokens_by_field[size_t(fi)].empty()) continue;
        if (!yue2_sheet_decode_field(vocab, Yue2SheetField(fi), event.tokens_by_field[size_t(fi)], &event, err)) {
            return false;
        }
    }
    return true;
}

// The four fields active_context_before tracks (generation_sheetsage2.py:
// 398-416): the most recently seen structure/key/chord TOKEN LISTS
// (independently — whichever of the three last had a token, in stitched-
// history order) and the most recently seen METER token specifically (not
// the whole rhythm payload — only the meter half of it, doc 21 §4.2).
struct Yue2SheetPrefixContext {
    bool has_structure = false;
    std::vector<int32_t> structure_tokens;
    bool has_key = false;
    std::vector<int32_t> key_tokens;
    bool has_chord = false;
    std::vector<int32_t> chord_tokens;
    bool has_meter = false;
    int32_t meter_token = 0;
};

// active_context_before(events, tokenizer, time_abs) (generation_sheetsage2.
// py:398-416, doc 21 §4.2). `events` must be in ascending-time (stitched-
// history append) order — the "last-seen wins" walk below relies on that,
// exactly like the reference's own forward scan. Only events with
// `event.time <= time_abs + 1e-6` are considered.
inline Yue2SheetPrefixContext yue2_sheet_active_context_before(const std::vector<Yue2SheetEvent> & stitched_events_so_far,
                                                                double time_abs) {
    Yue2SheetPrefixContext ctx;
    const double eps = 1e-6;
    for (const Yue2SheetEvent & e : stitched_events_so_far) {
        if (!(e.time <= time_abs + eps)) continue;
        const auto & structure_toks = e.tokens_by_field[size_t(Yue2SheetField::Structure)];
        if (!structure_toks.empty()) {
            ctx.has_structure = true;
            ctx.structure_tokens = structure_toks;
        }
        const auto & key_toks = e.tokens_by_field[size_t(Yue2SheetField::Key)];
        if (!key_toks.empty()) {
            ctx.has_key = true;
            ctx.key_tokens = key_toks;
        }
        const auto & chord_toks = e.tokens_by_field[size_t(Yue2SheetField::Chord)];
        if (!chord_toks.empty()) {
            ctx.has_chord = true;
            ctx.chord_tokens = chord_toks;
        }
        if (e.rhythm_has_meter) {
            for (int32_t tok : e.tokens_by_field[size_t(Yue2SheetField::Rhythm)]) {
                if (yue2_sheet_token_type(tok) == Yue2SheetTokenType::Meter) {
                    ctx.has_meter = true;
                    ctx.meter_token = tok;
                    break;
                }
            }
        }
    }
    return ctx;
}

// apply_prefix_context(event, context, tokenizer, prompts)
// (generation_sheetsage2.py:419-432, doc 21 §4.2). Applied ONLY to the very
// first rebased prefix event. Splices missing structure/key/chord straight
// in; for rhythm, only prepends a meter token onto an EXISTING
// eighth-position-only payload (never fabricates a rhythm field from
// nothing) — meter must precede eighth_position within one rhythm payload
// (doc 21 §5.1's strict shape rule), hence "prepend", not "append".
inline bool yue2_sheet_apply_prefix_context(const SheetSage2Tokens & vocab, Yue2SheetEvent & event,
                                             const Yue2SheetPrefixContext & ctx, std::string * err) {
    if (event.tokens_by_field[size_t(Yue2SheetField::Structure)].empty() && ctx.has_structure) {
        event.tokens_by_field[size_t(Yue2SheetField::Structure)] = ctx.structure_tokens;
    }
    if (event.tokens_by_field[size_t(Yue2SheetField::Key)].empty() && ctx.has_key) {
        event.tokens_by_field[size_t(Yue2SheetField::Key)] = ctx.key_tokens;
    }
    if (event.tokens_by_field[size_t(Yue2SheetField::Chord)].empty() && ctx.has_chord) {
        event.tokens_by_field[size_t(Yue2SheetField::Chord)] = ctx.chord_tokens;
    }
    auto & rhythm_toks = event.tokens_by_field[size_t(Yue2SheetField::Rhythm)];
    bool has_meter_tok = false, has_eighth_tok = false;
    for (int32_t tok : rhythm_toks) {
        Yue2SheetTokenType tt = yue2_sheet_token_type(tok);
        if (tt == Yue2SheetTokenType::Meter) has_meter_tok = true;
        if (tt == Yue2SheetTokenType::EighthPosition) has_eighth_tok = true;
    }
    if (has_eighth_tok && !has_meter_tok && ctx.has_meter) {
        rhythm_toks.insert(rhythm_toks.begin(), ctx.meter_token);
    }
    return yue2_sheet_refresh_event_values(vocab, event, err);
}

// prompt_prefix(prompts) (tokenization_sheetsage2.py:271-277, doc 21 §2):
// [sos, *(prompt_to_id[name] for name in normalize_prompts(prompts)), out].
inline bool yue2_sheet_prompt_prefix(const SheetSage2Tokens & vocab, const std::vector<std::string> & prompts,
                                      std::vector<int32_t> * out, std::string * err) {
    std::vector<std::string> normalized;
    if (!yue2_sheet_normalize_prompts(vocab, prompts, &normalized, err)) return false;
    std::vector<int32_t> tokens;
    tokens.push_back(SheetSage2Tokens::SOS);
    for (const std::string & name : normalized) tokens.push_back(vocab.prompt_name_to_id.at(name));
    tokens.push_back(SheetSage2Tokens::OUT);
    if (out) *out = std::move(tokens);
    return true;
}

// encode_decoded_sequence(decoded) (tokenization_sheetsage2.py:647-663, doc
// 21 §4.2). `decoded.events` must be in non-decreasing `.subbeat` order —
// enforced here (the reference raises on a decrease); no EOS is appended
// (has_eos is always false for a continuation prefix in this port's usage,
// but the field is still respected for fidelity).
inline bool yue2_sheet_encode_decoded_sequence(const SheetSage2Tokens & vocab, const Yue2SheetDecoded & decoded,
                                                std::vector<int32_t> * out, std::string * err) {
    std::vector<int32_t> tokens;
    if (!yue2_sheet_prompt_prefix(vocab, decoded.prompts, &tokens, err)) return false;

    int32_t previous_step = 0;
    for (size_t i = 0; i < decoded.events.size(); ++i) {
        const Yue2SheetEvent & e = decoded.events[i];
        if (e.subbeat < previous_step) {
            if (err) *err = "encode_decoded_sequence: event subbeats are not non-decreasing";
            return false;
        }
        const std::vector<int32_t> shift_tokens = yue2_sheet_subbeat_shift_to_tokens(e.subbeat - previous_step);
        tokens.insert(tokens.end(), shift_tokens.begin(), shift_tokens.end());
        previous_step = e.subbeat;
        for (int fi = 0; fi < 6; ++fi) {
            const auto & field_toks = e.tokens_by_field[size_t(fi)];
            tokens.insert(tokens.end(), field_toks.begin(), field_toks.end());
        }
    }
    if (decoded.has_eos) tokens.push_back(SheetSage2Tokens::EOS);
    if (out) *out = std::move(tokens);
    return true;
}

// The overlap prefix build's result. `ok == false` means the reference's own
// `(None, None, None)` — no anchor to rebase against in [window_start,
// prefix_end), so the caller should skip the prefix entirely for this
// window and decode it exactly like window 0 (fresh 8-token header,
// base_subbeat == 0). Not exercised by any fixture on disk as of this
// writing (every window >= 1 across short-a/short-b/long-a/long-b/long-c
// finds an anchor) — flagged here rather than guessed at.
struct Yue2SheetOverlapPrefix {
    bool ok = false;
    Yue2SheetDecoded prefix_decoded;
    std::vector<int32_t> prefix_ids;
    int32_t base_subbeat = 0;
};

// build_overlap_prefix_tokens(stitched_events, tokenizer, prompts,
// window_start, prefix_end) (generation_sheetsage2.py:435-512, doc 21 §4.2).
// `stitched_events_so_far` = every stitched event from windows BEFORE this
// one (NOT the whole song's final stitched list — at the point the
// reference pipeline calls this, later windows haven't been decoded yet),
// in ascending-time order.
inline bool yue2_sheet_build_overlap_prefix_tokens(const SheetSage2Tokens & vocab,
                                                    const std::vector<Yue2SheetEvent> & stitched_events_so_far,
                                                    const std::vector<std::string> & prompts, double window_start,
                                                    double prefix_end, Yue2SheetOverlapPrefix * out, std::string * err) {
    const double eps = 1e-4;  // generation_sheetsage2.py:436 (doc 21 §4.2)

    std::vector<Yue2SheetEvent> source;
    for (const Yue2SheetEvent & e : stitched_events_so_far) {
        if (e.time >= window_start - eps && e.time < prefix_end - eps) source.push_back(e);
    }
    std::stable_sort(source.begin(), source.end(), [](const Yue2SheetEvent & a, const Yue2SheetEvent & b) {
        if (a.global_subbeat != b.global_subbeat) return a.global_subbeat < b.global_subbeat;
        return a.time < b.time;
    });

    int first = -1;
    for (size_t i = 0; i < source.size(); ++i) {
        if (source[i].has_timestamp || source[i].has_rhythm) {
            first = (int) i;
            break;
        }
    }
    if (first < 0) {
        if (out) *out = Yue2SheetOverlapPrefix{};  // ok == false: (None, None, None)
        return true;
    }
    source.erase(source.begin(), source.begin() + first);

    const int32_t base_subbeat = source.front().global_subbeat;
    const Yue2SheetPrefixContext context = yue2_sheet_active_context_before(stitched_events_so_far, source.front().time);

    std::vector<Yue2SheetEvent> prefix_events;
    prefix_events.reserve(source.size());
    for (const Yue2SheetEvent & e : source) {
        Yue2SheetEvent clone = e;
        clone.subbeat = std::max(0, e.global_subbeat - base_subbeat);
        if (clone.has_timestamp) {
            yue2_sheet_set_event_local_timestamp(clone, e.time - window_start);
        }
        if (!yue2_sheet_refresh_event_values(vocab, clone, err)) return false;
        prefix_events.push_back(std::move(clone));
    }
    if (!yue2_sheet_apply_prefix_context(vocab, prefix_events.front(), context, err)) return false;

    Yue2SheetDecoded decoded;
    decoded.schema_version = "v1";
    decoded.prompts = prompts;
    decoded.events = prefix_events;
    decoded.has_eos = false;

    std::vector<int32_t> ids;
    if (!yue2_sheet_encode_decoded_sequence(vocab, decoded, &ids, err)) return false;

    // Guardrail (pipeline_sheetsage2.py:102-107, default preset only): a
    // prefix this large would leave the decode loop almost no budget before
    // hitting the 5120-token cap. Hard failure, not a silent skip, for the
    // preset this port uses. Not exercised by any fixture (the largest
    // observed prefix is long-b's 3226 tokens, window-0007).
    if (ids.size() >= size_t(5120 - 128)) {
        if (err) *err = "Overlap prefix fills the context; reduce overlap_seconds";
        return false;
    }

    if (out) {
        out->ok = true;
        out->prefix_decoded = std::move(decoded);
        out->prefix_ids = std::move(ids);
        out->base_subbeat = base_subbeat;
    }
    return true;
}

// ═════════════════════════════════════════════════════════════════════════
// §4. stitched_window_events (generation_sheetsage2.py:515-560, doc 21 §4.3).
// ═════════════════════════════════════════════════════════════════════════

// Appends this window's own accepted, clipped, rebased events onto `*out`.
// `window_events` is the window's OWN full decoded event list (prefix events
// re-decoded included, per §2's contract above — `window.subbeat` values are
// window-LOCAL, starting at 0 right after that window's own `<|out|>`).
// `target_seconds` is this window's own (possibly truncated) length,
// `w.end - w.start` (doc 21 §4.1's worked example: window 1 of the 305.9s
// song gets `target_seconds = min(300, 300) = 300`, i.e. its own full
// window length — this is exactly `w.end - w.start` for every window,
// truncated or not). `base_subbeat` is 0 for window 0, or the third return
// value of `yue2_sheet_build_overlap_prefix_tokens` for window i's OWN
// prefix (i.e. the value produced when THIS window's prefix was built, not
// the next one's).
//
// eps: see this file's top-of-file note — doc 21 §4.3's own pseudocode does
// not restate a literal eps value for its accept-window check the way §4.2
// does; this port reuses §4.2's `1e-4`, cross-checked against every
// fixture's `accepted_events` count in test_stitch.cpp.
inline void yue2_sheet_stitch_window(const std::vector<Yue2SheetEvent> & window_events, const Yue2SheetWindow & w,
                                      double song_duration, int32_t base_subbeat, std::vector<Yue2SheetEvent> * out) {
    const double eps = 1e-4;
    const double target_seconds = w.end - w.start;
    const Yue2SheetTimeMap tmap = yue2_sheet_event_time_map(window_events, target_seconds);

    for (const Yue2SheetEvent & e : window_events) {
        const double local_time = tmap.local_time(double(e.subbeat));
        const double abs_time = w.start + local_time;
        if (!(abs_time >= w.accept_start - eps && abs_time < w.accept_end - eps)) continue;
        if (!(abs_time < song_duration - eps)) continue;

        Yue2SheetEvent clone = e;
        clone.time = std::min(song_duration, std::max(0.0, abs_time));
        clone.has_time = true;
        clone.window_index = w.window_index;
        clone.has_window_index = true;
        clone.window_start = w.start;
        clone.has_window_start = true;
        clone.source_subbeat = e.subbeat;
        clone.has_source_subbeat = true;
        clone.global_subbeat = base_subbeat + e.subbeat;
        clone.has_global_subbeat = true;
        if (clone.has_timestamp) clone.timestamp = clone.time;

        for (Yue2SheetMelodyNote & note : clone.melody) {
            const double local_end = tmap.local_time(double(e.subbeat) + double(note.duration_steps));
            double end_time = w.start + local_end;
            end_time = std::min(song_duration, std::max(0.0, end_time));
            end_time = std::max(end_time, clone.time + 0.04);
            note.has_end_time = true;
            note.end_time = end_time;
        }

        out->push_back(std::move(clone));
    }
}

// Final whole-song sort (generation_sheetsage2.py:154-155, doc 21 §4.3): run
// ONCE after every window has been stitched (not per window). Ascending by
// (time, global_subbeat).
inline void yue2_sheet_sort_stitched_events(std::vector<Yue2SheetEvent> * events) {
    std::stable_sort(events->begin(), events->end(), [](const Yue2SheetEvent & a, const Yue2SheetEvent & b) {
        if (a.time != b.time) return a.time < b.time;
        return a.global_subbeat < b.global_subbeat;
    });
}
