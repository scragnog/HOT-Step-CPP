#pragma once
// yue2/sheetsage-pipeline.h — SheetSage2 end to end: mono 24 kHz audio in,
// ABC text (or a typed soft-failure) out.
//
// HOT-Step file (no acestep.cpp analog). Composes every other lane in this
// directory into the one call doc 19's "Pipeline" bullet asks for:
// `sheetsage_transcribe()`. Owns no math of its own beyond the window-plan/
// decode-loop orchestration and the tiny glue that turns a stitched event
// list into `sheetsage-notation.h`'s five in-memory inputs — every actual
// computation (encode, decode step, grammar mask, event decode, stitching,
// notation) is delegated to the lane that already owns it.
//
// AUTHORITY: docs/plans/yue2/23-sheetsage2-progress.md (state, hard rules,
// the encoder precision section) and docs/plans/yue2/19-sheetsage2-cot-
// training.md §Decisions (the soft-failure contract, the G6 gate) and
// docs/plans/yue2/22-sheetsage2-fixtures.md. `sheetsage-stitch.h`'s own §2
// ("Decoder-loop stop contract") is the literal spec this file's decode loop
// implements — read that comment block before touching the loop below.
// K:/yue2/.cache/huggingface-sheetsage/modules/transformers_modules/SheetSage2/
// {pipeline_sheetsage2.py,generation_sheetsage2.py} is the Python reference
// for the loop shape (`constrained_prompt_generate_batch`, `generate`, the
// `stop_time` branch) this file ports.
//
// ── ERROR CONVENTION (this file's own addition to the lane's shared style) ──
//
// Every OTHER file in this lane returns `bool`+`*err` for anything that can
// fail, with no notion of a "soft" failure baked into the return type itself
// — sheetsage-notation.h is the one exception, returning `Yue2SheetAbcResult
// {ok,abc,error}` directly because doc 19's decisions section requires ABC
// rendering to fail per-source without aborting a batch job. This file's
// public entry point, `sheetsage_transcribe()`, follows BOTH conventions at
// once, deliberately: the function's own `bool` return is reserved for a
// precondition/setup failure so severe that no `SheetSageTranscribeResult`
// could be produced at all (null/empty audio, a model that was never loaded,
// an `exact` mismatch between what the caller asked for and how the model
// was actually loaded, the vocabulary's own fingerprint self-check failing,
// or `sliding_window_plan`'s own precondition) — these are bugs in the
// CALLER, not in any one source's audio, and *err (the function's own
// out-param) carries the message. Once the function has committed to
// attempting a transcription, it ALWAYS returns `true` from then on and
// routes every subsequent failure — a decode-side hard error (a grammar
// state machine reaching an impossible token, a ggml compute failure) or a
// notation-stage soft failure (the reference's own "shorter than the ABC
// subbeat grid" class of failure) — into `SheetSageTranscribeResult::{ok,
// error}` instead, exactly as the task brief specifies: "a notation failure
// returns ok=false with the reference's error text; a decode-side hard error
// returns ok=false with a typed message." Never throws past this function.

#include "sheetsage-decoder.h"
#include "sheetsage-encoder.h"
#include "sheetsage-events.h"
#include "sheetsage-grammar.h"
#include "sheetsage-model.h"
#include "sheetsage-notation.h"
#include "sheetsage-repair.h"
#include "sheetsage-stitch.h"
#include "sheetsage-tokens.h"

#include "backend.h"  // BackendPair-style reg/proc-address lookup, for SheetSageTranscribeOptions::threads

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// ── Public API ───────────────────────────────────────────────────────────────

struct SheetSageTranscribeOptions {
    // Must equal the SheetSageModelLoadOptions::exact the caller loaded `m`
    // with (sheetsage-model.h's `m.exact`, the accessor this file's own
    // sibling lane added — see that struct's comment). This file has no
    // per-window precision lever of its own (sheetsage-encoder.h's file
    // header: "there is no per-encode 'exact' switch here, deliberately...
    // a LOAD-TIME decision"), so a caller that passes exact=true against a
    // fast-loaded model gets a fast-fail setup error here rather than a
    // silently-wrong transcription. Default false: match whatever the model
    // was loaded with by leaving this false only when the model itself was
    // loaded fast (this file also accepts exact=false against an
    // exact-loaded model -- running MORE precisely than requested is never
    // an error, only the reverse is).
    bool exact = false;

    // > 0: apply via the same ggml_backend_reg_get_proc_address("ggml_backend_
    // set_n_threads") lookup backend.h's own cpu_backend_new() uses, to both
    // m.backend and (if distinct) m.cpu_backend, before doing any work. <= 0
    // (the default): leave the backend's thread count exactly as
    // backend_init() configured it at load time (GGML_N_THREADS env var or
    // the physical-core heuristic) -- this is an override for the OFFLINE
    // yue2-sheet cache stage (doc 23's "run on the CPU backend" recommendation
    // wants explicit control independent of the environment the batch job
    // happens to inherit), not a per-call tuning knob with a default that
    // does anything.
    int threads = 0;

    // Keep every hidden state / intermediate encoder tensor per window (see
    // SheetSageEncodeOptions::want_hidden_states). Off (default) for
    // production/offline caching -- this file's own diagnostics
    // (SheetSageTranscribeResult::window_ids/stitched/timings) already cover
    // everything the yue2-probe --sheetsage-transcribe mode needs without it.
    bool want_hidden_states = false;

    // Apply sheetsage-repair.h's deterministic repair pass when the FIRST
    // notation render fails (doc 23's "Repair" note, task 2026-09-16). On by
    // default: a repaired sheet is better training conditioning than none.
    // The first render is always attempted untouched regardless of this flag
    // -- G4/G6 byte-exactness on the pinned fixtures never goes through the
    // repair path either way (see sheetsage-repair.h's own header). Set false
    // (yue2-sheet's `--no-repair`) to get the pre-repair behavior exactly:
    // one render, and any failure is a plain soft failure.
    bool repair = true;
};

struct SheetSageTranscribeResult {
    bool        ok = false;
    std::string abc;
    std::string error;

    // What sheetsage-repair.h changed en route to `abc`/`error` above, or ""
    // if the first (untouched) render already succeeded or `opt.repair` was
    // false. See sheetsage-repair.h's own Yue2SheetRepairOutcome::repaired.
    std::string repaired;

    // Diagnostics -- always populated up to the point of the failure, even
    // when ok is false, so a caller (or the probe) can see how far the
    // pipeline got.
    int                                windows = 0;
    std::vector<std::vector<int32_t>> window_ids;  // per window: prefix ++ generated (+ synthetic eos)
    std::vector<Yue2SheetEvent>       stitched;    // whole-song stitched events, final sort applied
    double ms_encode   = 0.0;
    double ms_decode   = 0.0;
    double ms_notation = 0.0;
};

// ── Implementation detail ───────────────────────────────────────────────────

namespace yue2_sheet_pipeline_detail {

// FULL_TASK_PROMPTS (doc 19/21: "(timestamp, downbeat_meter, structure, key,
// chord_full, melody_full)" -- the only prompt set this port ever uses, per
// sheetsage-grammar.h's own note that allow_field_starts is hard-wired to
// chord_full/melody_full by construction).
inline const std::vector<std::string> & full_task_prompts() {
    static const std::vector<std::string> v = {
        "timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full",
    };
    return v;
}

// Same reg/proc-address pattern as backend.h's own cpu_backend_new() (that
// function is `static` there, private to that TU's own callers, so this is a
// small duplicate rather than a cross-file dependency on a private symbol).
inline void set_backend_threads(ggml_backend_t backend, int n_threads) {
    if (!backend || n_threads <= 0) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return;
    }
    auto set_fn =
        (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) {
        set_fn(backend, n_threads);
    }
}

// The notation-stage input glue: reproduces exactly the "beat/notation
// try-block" sheetsage-events.h's yue2_sheet_write_exports() runs
// (exports_sheetsage2.py:136-177) but keeps every value as an in-memory
// Yue2SheetAbcRow instead of round-tripping through that function's text
// files -- same underlying calls (yue2_sheet_rhythm_rows/_pad_beats/
// _interval_rows/_clip_to_beats/_raw_melody_notes/_notation_notes/
// _write_midi), same error text on the reference's own ValueErrors, same
// "No key was decoded" message on the no-key branch. `ok=false` here is
// ALWAYS a soft failure in the sense doc 19 means it (this exact source's
// ABC does not exist), never a caller bug -- the caller (sheetsage_transcribe
// below) maps it straight into SheetSageTranscribeResult{ok=false, error}.
struct NotationInputs {
    bool                    ok = false;
    std::string             error;
    std::string             melody_midi_bytes;
    std::vector<Yue2SheetAbcRow> beat_rows, chord_rows, key_rows, structure_rows;
    // Pre-MIDI raw notes, split by voice (the SAME split that produced
    // melody_midi_bytes above) -- exposed so sheetsage-repair.h's family (d)
    // can drop one note and rebuild the MIDI bytes on a retry, without this
    // file duplicating yue2_sheet_notation_notes()'s own vocal/ins split.
    std::vector<Yue2SheetRawNote> vocal_notes, ins_notes;
};

inline NotationInputs build_notation_inputs(const std::vector<Yue2SheetEvent> & events, double duration) {
    NotationInputs out;

    std::vector<Yue2SheetBeatRow> beats;
    std::string                   err;
    if (!yue2_sheet_rhythm_rows(events, &beats, &err)) {
        out.error = err;
        return out;
    }

    std::vector<Yue2SheetRawNote> raw_notes = yue2_sheet_raw_melody_notes(events, duration);
    std::vector<std::array<double, 2>> raw_spans;
    raw_spans.reserve(raw_notes.size());
    for (const auto & n : raw_notes) {
        raw_spans.push_back({ n.start, n.end });
    }

    std::vector<Yue2SheetBeatRow> abc_beats;
    if (!yue2_sheet_pad_beats(beats, duration, raw_spans, &abc_beats, &err)) {
        out.error = err;
        return out;
    }

    for (const auto & r : abc_beats) {
        out.beat_rows.push_back({ yue2_sheet_cell_text(Yue2SheetCell::Real(r.time)),
                                   yue2_sheet_cell_text(Yue2SheetCell::Int(r.beat_id)),
                                   yue2_sheet_cell_text(Yue2SheetCell::Int(r.numerator)),
                                   yue2_sheet_cell_text(Yue2SheetCell::Int(r.denominator)) });
    }

    auto build_interval_rows = [&](Yue2SheetField field) {
        std::vector<Yue2SheetAbcRow> rows;
        for (const auto & r : yue2_sheet_clip_to_beats(yue2_sheet_interval_rows(events, field, duration), abc_beats)) {
            rows.push_back({ yue2_sheet_cell_text(Yue2SheetCell::Real(r.start)),
                              yue2_sheet_cell_text(Yue2SheetCell::Real(r.end)), r.value });
        }
        return rows;
    };
    out.chord_rows     = build_interval_rows(Yue2SheetField::Chord);
    out.key_rows       = build_interval_rows(Yue2SheetField::Key);
    out.structure_rows = build_interval_rows(Yue2SheetField::Structure);

    if (yue2_sheet_interval_rows(events, Yue2SheetField::Key, duration).empty()) {
        // exports_sheetsage2.py's own message (doc 21 §6, verbatim in
        // sheetsage-events.h's yue2_sheet_write_exports() comment).
        out.error = "No key was decoded; cannot construct a keyed ABC score";
        return out;
    }

    std::vector<Yue2SheetRawNote> clean = yue2_sheet_notation_notes(raw_notes);
    std::vector<Yue2SheetRawNote> vocal, ins;
    for (const auto & n : clean) {
        (n.track == 0 ? vocal : ins).push_back(n);
    }
    std::vector<uint8_t> midi = yue2_sheet_write_midi(vocal, ins);
    out.melody_midi_bytes.assign(midi.begin(), midi.end());
    out.vocal_notes = std::move(vocal);
    out.ins_notes   = std::move(ins);

    out.ok = true;
    return out;
}

}  // namespace yue2_sheet_pipeline_detail

// The whole pipeline, one call: pad/window audio -> encode -> grammar-masked
// greedy decode (prefill + step, overlap-primed on windows >= 1) -> per-window
// event decode -> stitch -> notation. `m` must already be loaded
// (sheetsage_model_load()); this file never loads or frees it.
inline bool sheetsage_transcribe(const SheetSageModel & m, const float * mono24, int64_t n,
                                  const SheetSageTranscribeOptions & opt, SheetSageTranscribeResult * out,
                                  std::string * err) {
    using namespace yue2_sheet_pipeline_detail;
    *out = SheetSageTranscribeResult{};

    if (!mono24 || n <= 0) {
        if (err) *err = "sheetsage_transcribe: mono24/n must be a non-empty audio buffer";
        return false;
    }
    if (!m.mert.blocks_loaded || m.dec.blk.empty() || !m.backend) {
        if (err) *err = "sheetsage_transcribe: model is not loaded (call sheetsage_model_load() first)";
        return false;
    }
    if (opt.exact && !m.exact) {
        if (err) {
            *err = "sheetsage_transcribe: SheetSageTranscribeOptions.exact=true but the model was loaded without "
                   "SheetSageModelLoadOptions.exact=true (doc 23: exact is a load-time-only precision decision)";
        }
        return false;
    }
    if (opt.threads > 0) {
        set_backend_threads(m.backend, opt.threads);
        if (m.cpu_backend && m.cpu_backend != m.backend) {
            set_backend_threads(m.cpu_backend, opt.threads);
        }
    }

    SheetSage2Tokens vocab;
    std::string      terr;
    if (!yue2_sheet_tokens_build(vocab, &terr)) {
        if (err) *err = "sheetsage_transcribe: vocabulary build failed: " + terr;
        return false;
    }

    const double sr       = (double) m.mert.cfg.sample_rate;
    const double duration = (double) n / sr;

    std::vector<Yue2SheetWindow> windows;
    if (!yue2_sheet_sliding_window_plan(duration, &windows, &terr, (double) m.cfg.window_seconds,
                                        (double) m.cfg.overlap_seconds_default,
                                        (double) m.cfg.lookahead_seconds_default)) {
        if (err) *err = "sheetsage_transcribe: " + terr;
        return false;
    }

    out->windows = (int) windows.size();
    out->window_ids.resize(windows.size());

    SheetSageEncoderGraph       enc_graph;
    std::vector<Yue2SheetEvent> stitched_so_far;  // grows across windows; also this call's running history
    const int64_t               window_samples = sheetsage_window_samples(m);
    bool                         hard_failed    = false;

    for (size_t wi = 0; wi < windows.size() && !hard_failed; wi++) {
        const Yue2SheetWindow & w = windows[wi];

        // ── slice + encode ──
        const int64_t start_idx = (int64_t) std::llround(w.start * sr);
        const int64_t avail     = std::max<int64_t>(0, std::min<int64_t>(window_samples, n - start_idx));
        const float * win_pcm   = (start_idx >= 0 && start_idx < n) ? (mono24 + start_idx) : nullptr;

        auto t0 = std::chrono::steady_clock::now();
        SheetSageEncodeOptions enc_opt;
        enc_opt.want_hidden_states = opt.want_hidden_states;
        SheetSageEncodeResult enc_res;
        if (!sheetsage_encode_window(m, &enc_graph, win_pcm, avail, enc_opt, &enc_res, &terr)) {
            out->ok    = false;
            out->error = "encoder: " + terr;
            hard_failed = true;
            break;
        }
        out->ms_encode += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        // ── prefix: window 0 gets the fixed task-prompt header; windows >= 1
        // get the overlap prefix rebuilt from every stitched event so far, or
        // fall back to a fresh header when no anchor exists (doc 21 §4.2's
        // (None,None,None) case -- never exercised by any fixture, handled
        // anyway per sheetsage-events.h's own note on Yue2SheetOverlapPrefix)
        std::vector<int32_t> prefix_ids;
        int32_t               base_subbeat = 0;
        if (wi == 0) {
            if (!yue2_sheet_prompt_prefix(vocab, full_task_prompts(), &prefix_ids, &terr)) {
                out->ok = false; out->error = "prompt_prefix: " + terr; hard_failed = true; break;
            }
        } else {
            Yue2SheetOverlapPrefix overlap;
            if (!yue2_sheet_build_overlap_prefix_tokens(vocab, stitched_so_far, full_task_prompts(), w.start,
                                                        w.prefix_end, &overlap, &terr)) {
                out->ok = false; out->error = "overlap_prefix: " + terr; hard_failed = true; break;
            }
            if (overlap.ok) {
                prefix_ids   = overlap.prefix_ids;
                base_subbeat = overlap.base_subbeat;
            } else if (!yue2_sheet_prompt_prefix(vocab, full_task_prompts(), &prefix_ids, &terr)) {
                out->ok = false; out->error = "prompt_prefix: " + terr; hard_failed = true; break;
            }
        }

        Yue2SheetGrammarState state = yue2_sheet_grammar_init();
        if (yue2_sheet_grammar_prime(state, prefix_ids, &terr) == Yue2SheetGrammarStep::Error) {
            out->ok = false; out->error = "grammar prime: " + terr; hard_failed = true; break;
        }

        // ── decode loop: prefill, then grammar-masked greedy step-by-step,
        // per sheetsage-stitch.h's §2 stop contract ──
        SheetSageDecKvCache cache;
        if (!sheetsage_dec_kv_cache_alloc(m, (int64_t) m.cfg.max_output_seq_len, enc_res.T_sub, &cache, &terr)) {
            out->ok = false; out->error = "decoder kv cache: " + terr; hard_failed = true; break;
        }

        auto td0 = std::chrono::steady_clock::now();
        std::vector<int32_t> generated_ids;
        std::vector<int64_t> lp = { (int64_t) prefix_ids.size() - 1 };
        SheetSageDecodeResult step_res;
        bool ok = sheetsage_dec_prefill(m, cache, enc_res.memory.data(), enc_res.T_sub, prefix_ids, lp, &step_res,
                                        &terr);
        if (!ok) {
            out->ok = false; out->error = "decoder prefill: " + terr;
            sheetsage_dec_kv_cache_free(&cache);
            hard_failed = true;
            break;
        }

        int64_t            current_length = (int64_t) prefix_ids.size();
        bool               grammar_error  = false;
        std::vector<bool>  allowed;  // yue2_sheet_grammar_allowed() re-assign()s this every step; hoisted to avoid
                                      // reallocating a ~31678-bit vector on every decode step of every window
        for (;;) {
            yue2_sheet_grammar_allowed(state, allowed);
            // masked argmax over step_res.logits (V floats)
            int32_t accepted = -1;
            float   best     = -INFINITY;
            for (int64_t v = 0; v < step_res.V; v++) {
                if (!allowed[(size_t) v]) continue;
                if (step_res.logits[(size_t) v] > best) {
                    best     = step_res.logits[(size_t) v];
                    accepted = (int32_t) v;
                }
            }
            if (accepted < 0) {
                terr = "sheetsage2 decode: no token is allowed under the current grammar state (mask is all-false)";
                grammar_error = true;
                break;
            }

            Yue2SheetGrammarStep gstep = yue2_sheet_grammar_update(state, accepted, &terr);
            generated_ids.push_back(accepted);
            current_length++;
            if (gstep == Yue2SheetGrammarStep::Error) {
                grammar_error = true;
                break;
            }
            if (gstep == Yue2SheetGrammarStep::Finished) {
                break;  // natural EOS
            }

            if (w.has_generation_stop && yue2_sheet_token_type(accepted) == Yue2SheetTokenType::Time) {
                const int32_t time_id = accepted - SheetSage2Tokens::TIME_TOKEN_START;
                if (yue2_sheet_time_seconds(time_id) >= w.generation_stop) {
                    generated_ids.push_back(SheetSage2Tokens::EOS);  // synthetic eos, never run through update()
                    current_length++;
                    break;
                }
            }
            if (current_length >= (int64_t) m.cfg.max_output_seq_len) {
                break;  // length limit, doc 19 §3.6 point 5 / stitch.h §2 point 3
            }

            std::vector<float> next_logits;
            ok = sheetsage_dec_step(m, cache, accepted, &next_logits, &terr);
            if (!ok) {
                break;
            }
            step_res.logits = std::move(next_logits);
        }
        sheetsage_dec_kv_cache_free(&cache);
        out->ms_decode += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();

        if (!ok) {
            out->ok = false; out->error = "decoder step: " + terr; hard_failed = true; break;
        }
        if (grammar_error) {
            out->ok = false; out->error = "grammar: " + terr; hard_failed = true; break;
        }

        std::vector<int32_t> full_tokens = prefix_ids;
        full_tokens.insert(full_tokens.end(), generated_ids.begin(), generated_ids.end());
        out->window_ids[wi] = full_tokens;

        Yue2SheetDecoded decoded;
        bool              recovered = false;
        if (!yue2_sheet_decode_generated_tokens(vocab, full_tokens, &decoded, &recovered, &terr)) {
            out->ok = false; out->error = "decode_sequence: " + terr; hard_failed = true; break;
        }

        yue2_sheet_stitch_window(decoded.events, w, duration, base_subbeat, &stitched_so_far);
    }

    if (hard_failed) {
        out->stitched = stitched_so_far;
        sheetsage_enc_graph_free(&enc_graph);
        return true;  // committed to an attempt; this is the soft/decode-side-hard-error path (see file header)
    }

    yue2_sheet_sort_stitched_events(&stitched_so_far);
    out->stitched = stitched_so_far;
    sheetsage_enc_graph_free(&enc_graph);

    auto tn0 = std::chrono::steady_clock::now();
    yue2_sheet_pipeline_detail::NotationInputs ni = yue2_sheet_pipeline_detail::build_notation_inputs(stitched_so_far,
                                                                                                       duration);
    if (!ni.ok) {
        out->ok    = false;
        out->error = ni.error;
        out->ms_notation =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tn0).count();
        return true;
    }

    Yue2SheetAbcResult abc_res;
    std::string        repaired;
    if (opt.repair) {
        Yue2SheetRepairInputs ri;
        ri.beat_rows      = ni.beat_rows;
        ri.chord_rows     = ni.chord_rows;
        ri.key_rows       = ni.key_rows;
        ri.structure_rows = ni.structure_rows;
        ri.vocal_notes    = ni.vocal_notes;
        ri.ins_notes      = ni.ins_notes;
        Yue2SheetRepairOutcome ro =
            yue2_sheet_notation_repair_and_generate(std::move(ri), /*melody_only=*/false);
        abc_res  = ro.result;
        repaired = ro.repaired;
    } else {
        abc_res = yue2_sheet_notation_generate(ni.melody_midi_bytes, ni.beat_rows, ni.chord_rows, ni.key_rows,
                                                ni.structure_rows, /*melody_only=*/false);
    }
    out->ms_notation = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tn0).count();

    out->ok       = abc_res.ok;
    out->abc      = abc_res.abc;
    out->error    = abc_res.error;
    out->repaired = repaired;
    return true;
}
