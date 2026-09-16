#pragma once
// yue2/sheetsage-grammar.h — PromptGrammarState: the per-step allowed-token
// mask and the state machine that advances it, for SheetSage2's grammar-
// constrained greedy decode.
//
// HOT-Step file (no acestep.cpp analog). A literal port of
// `PromptGrammarState` (`generation_sheetsage2.py:12-116`) plus the priming
// replay `constrained_prompt_generate_batch` runs before its own decode loop
// (`:162-177`). There is no sampling anywhere in SheetSage2 decode — masked
// `argmax`, always — so this file's only job is producing that mask and
// keeping it in sync with the accepted token stream.
//
// AUTHORITY: docs/plans/yue2/21-sheetsage2-symbolic-pin.md §3 (grammar) and
// §4.2 (why priming matters for window >= 1). Where this file and that
// document disagree, the document wins (and the Python under it wins over
// the document). Section references below (§3.1, §3.2, §3.3, §3.4, §4.2)
// point into doc 21; line numbers point into
// K:\yue2\.cache\huggingface-sheetsage\modules\transformers_modules\SheetSage2\generation_sheetsage2.py.
//
// Depends only on sheetsage-tokens.h's token_type()/block-boundary constants
// — the grammar never needs the label tables (structure names, chord
// strings, ...), only which block a token id falls in.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "sheetsage-tokens.h"

// ─────────────────────────────────────────────────────────────────────────
// State (§3.1 / __init__, generation_sheetsage2.py:13-20).
// ─────────────────────────────────────────────────────────────────────────

struct Yue2SheetGrammarState {
    enum class Incomplete : int8_t { None, RhythmAfterMeter, MelodyAfterPitch };

    // Count of fully-closed events so far. The reference tracks it but never
    // reads it back (doc 21 §3.1) — kept here purely for debug/telemetry.
    int32_t generated_events = 0;
    bool in_shift = true;
    int32_t shift_run = 0;
    int32_t payload_count = 0;
    // Yue2SheetField value of the last field *type* completed in the current
    // event, or -1 before any field token this event (FIELD_TO_INDEX,
    // generation_sheetsage2.py:9).
    int32_t last_field_index = -1;
    Incomplete incomplete = Incomplete::None;
    // Set once update() sees <|eos|>. Mirrors the task's "finished flag";
    // the reference itself has no such field, it just returns a bool from
    // update() — kept here too so callers can query state after the fact.
    bool finished = false;
};

// Result of one update()/prime() step. EOS is not an error — Finished is the
// ordinary, expected way a window's decode ends. Error means the reference's
// own `raise RuntimeError("Unexpected prompt token type ...")` case
// (generation_sheetsage2.py:115) was hit: a token type reached update() that
// no legal mask could have produced. This never fires on a correct grammar
// port fed a real model's masked-argmax output; it exists so the port never
// throws across the pipeline boundary when something upstream is wrong.
enum class Yue2SheetGrammarStep : int8_t { Continue, Finished, Error };

inline Yue2SheetGrammarState yue2_sheet_grammar_init() { return Yue2SheetGrammarState{}; }

namespace yue2_sheet_grammar_detail {
inline void set_range(std::vector<bool> & allowed, int32_t start, int32_t end) {
    for (int32_t i = start; i < end; ++i) allowed[size_t(i)] = true;
}

// _allow_field_starts (generation_sheetsage2.py:22-42). Each field is gated
// by whether it has ALREADY been completed this event (last_field_index
// strictly less than that field's index) — except melody, which stays legal
// even after a note has already been placed this event (`<=`, not `<`:
// doc 21 §3.2's note on why this one line differs from every other).
//
// Hard-wired to chord_full/melody_full (not chord_majmin/melody_vocal) by
// construction — doc 21 §3's note on this; FULL_TASK_PROMPTS never asks for
// the other two, so this matches the reference exactly for the only prompt
// set the port ever uses.
inline void allow_field_starts(const Yue2SheetGrammarState & st, std::vector<bool> & allowed) {
    using T = SheetSage2Tokens;
    const int32_t lf = st.last_field_index;
    if (lf < int32_t(Yue2SheetField::Timestamp)) set_range(allowed, T::TIME_TOKEN_START, T::TIME_TOKEN_END);
    if (lf < int32_t(Yue2SheetField::Rhythm)) {
        set_range(allowed, T::METER_TOKEN_START, T::METER_TOKEN_END);
        set_range(allowed, T::EIGHTH_POSITION_TOKEN_START, T::EIGHTH_POSITION_TOKEN_END);
    }
    if (lf < int32_t(Yue2SheetField::Structure)) set_range(allowed, T::STRUCTURE_TOKEN_START, T::STRUCTURE_TOKEN_END);
    if (lf < int32_t(Yue2SheetField::Key)) set_range(allowed, T::KEY_TOKEN_START, T::KEY_TOKEN_END);
    if (lf < int32_t(Yue2SheetField::Chord)) set_range(allowed, T::FULL_CHORD_TOKEN_START, T::FULL_CHORD_TOKEN_END);
    if (lf <= int32_t(Yue2SheetField::Melody)) set_range(allowed, T::PITCH_TOKEN_START, T::PITCH_TOKEN_END);  // <=, not <
}
}  // namespace yue2_sheet_grammar_detail

// allowed() (§3.2 / generation_sheetsage2.py:44-69). Later steps only ever
// ADD True bits on top of earlier ones — never clear — and the two
// `incomplete` branches RETURN EARLY, but only after the can_end/shift bits
// above them were already set: a step right after a `meter` token, for
// example, legally allows EOS, a subbeat shift, AND an `eighth_position`
// token all at once. Do not restructure this into an early "return only the
// incomplete-field mask" — that would silently forbid legal continuations
// (doc 21 §3.2's explicit warning).
inline void yue2_sheet_grammar_allowed(const Yue2SheetGrammarState & st, std::vector<bool> & allowed) {
    using T = SheetSage2Tokens;
    allowed.assign(size_t(T::N_TOKENS), false);

    const bool can_end = st.payload_count > 0;
    if (can_end) allowed[size_t(T::EOS)] = true;
    if (st.payload_count > 0 || st.in_shift) {
        if (st.shift_run < 4) {
            yue2_sheet_grammar_detail::set_range(allowed, T::SUBBEAT_SHIFT_TOKEN_START, T::SUBBEAT_SHIFT_TOKEN_END);
        }
    }

    if (st.incomplete == Yue2SheetGrammarState::Incomplete::RhythmAfterMeter) {
        yue2_sheet_grammar_detail::set_range(allowed, T::EIGHTH_POSITION_TOKEN_START, T::EIGHTH_POSITION_TOKEN_END);
        return;
    }
    if (st.incomplete == Yue2SheetGrammarState::Incomplete::MelodyAfterPitch) {
        yue2_sheet_grammar_detail::set_range(allowed, T::DURATION_TOKEN_START, T::DURATION_TOKEN_END);
        yue2_sheet_grammar_detail::set_range(allowed, T::PITCH_TOKEN_START, T::PITCH_TOKEN_END);
        return;
    }

    yue2_sheet_grammar_detail::allow_field_starts(st, allowed);
}

// update() (§3.3 / generation_sheetsage2.py:71-116). Advances state after one
// accepted token. Event-boundary subtlety: the "close out the event" block
// only fires on the FIRST shift token of a run (`!in_shift` guards it) — a
// run of several consecutive shift tokens (chaining shifts > 256 subbeats)
// closes the event exactly once, on entering the run, not once per token.
inline Yue2SheetGrammarStep yue2_sheet_grammar_update(Yue2SheetGrammarState & st, int32_t token, std::string * err) {
    const Yue2SheetTokenType tt = yue2_sheet_token_type(token);

    if (token == SheetSage2Tokens::EOS) {
        st.finished = true;
        return Yue2SheetGrammarStep::Finished;
    }

    if (tt == Yue2SheetTokenType::SubbeatShift) {
        if (!st.in_shift && st.payload_count > 0) {
            st.generated_events += 1;
            st.payload_count = 0;
            st.last_field_index = -1;
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
        }
        st.in_shift = true;
        st.shift_run += 1;
        return Yue2SheetGrammarStep::Continue;
    }

    st.in_shift = false;
    st.shift_run = 0;
    st.payload_count += 1;
    switch (tt) {
        case Yue2SheetTokenType::Time:
            st.last_field_index = int32_t(Yue2SheetField::Timestamp);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        case Yue2SheetTokenType::Meter:
            st.last_field_index = int32_t(Yue2SheetField::Rhythm);
            st.incomplete = Yue2SheetGrammarState::Incomplete::RhythmAfterMeter;
            break;
        case Yue2SheetTokenType::EighthPosition:
            st.last_field_index = int32_t(Yue2SheetField::Rhythm);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        case Yue2SheetTokenType::Structure:
            st.last_field_index = int32_t(Yue2SheetField::Structure);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        case Yue2SheetTokenType::Key:
            st.last_field_index = int32_t(Yue2SheetField::Key);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        case Yue2SheetTokenType::ChordFull:
            st.last_field_index = int32_t(Yue2SheetField::Chord);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        case Yue2SheetTokenType::Pitch:
            st.last_field_index = int32_t(Yue2SheetField::Melody);
            st.incomplete = Yue2SheetGrammarState::Incomplete::MelodyAfterPitch;
            break;
        case Yue2SheetTokenType::Duration:
            st.last_field_index = int32_t(Yue2SheetField::Melody);
            st.incomplete = Yue2SheetGrammarState::Incomplete::None;
            break;
        default:
            // generation_sheetsage2.py:114-115: `raise RuntimeError(f"Unexpected
            // prompt token type {token_type!r}")`. Never hit given the mask this
            // file produces (chord_majmin/melody_vocal/prompt/pad/sos/out/invalid
            // token types are never allowed mid-body) — treat as an assertion,
            // not a recoverable case, and report it rather than throw.
            if (err) *err = "sheetsage2 grammar: unexpected token type for token " + std::to_string(token);
            return Yue2SheetGrammarStep::Error;
    }
    return Yue2SheetGrammarStep::Continue;
}

// Priming replay for windows >= 1 (§3.1's "Priming for windows after the
// first" / §4.2; constrained_prompt_generate_batch, generation_sheetsage2.py:
// 162-177). The caller passes THIS window's own prefix ids exactly as fed to
// the decoder (07_prompt_prefix_ids in the fixtures): the fixed 8-token
// header for window 0 (nothing follows <|out|>, so this is a no-op and the
// state stays at the fresh row from yue2_sheet_grammar_init()), or the
// header plus the re-encoded overlap-prefix events for window >= 1 (doc 21
// §4.2's build_overlap_prefix_tokens output). Mirrors the reference exactly,
// including stripping one trailing <|eos|> if present before locating
// <|out|> — a defensive no-op for these fixtures, since neither
// tokenizer.prompt_prefix() nor build_overlap_prefix_tokens() ever appends
// an eos to a continuation prefix, but cheap and faithful to keep.
inline Yue2SheetGrammarStep yue2_sheet_grammar_prime(Yue2SheetGrammarState & st,
                                                      const std::vector<int32_t> & prefix_ids,
                                                      std::string * err) {
    const int32_t * begin = prefix_ids.data();
    size_t n = prefix_ids.size();
    if (n > 0 && begin[n - 1] == SheetSage2Tokens::EOS) --n;

    size_t out_index = n;
    for (size_t i = 0; i < n; ++i) {
        if (begin[i] == SheetSage2Tokens::OUT) { out_index = i; break; }
    }
    if (out_index == n) {
        if (err) *err = "sheetsage2 grammar prime: prefix has no <|out|> token";
        return Yue2SheetGrammarStep::Error;
    }

    for (size_t i = out_index + 1; i < n; ++i) {
        Yue2SheetGrammarStep step = yue2_sheet_grammar_update(st, begin[i], err);
        if (step != Yue2SheetGrammarStep::Continue) return step;
    }
    return Yue2SheetGrammarStep::Continue;
}
