#pragma once
// sheetsage-refined.h: SheetSage2-Refined window decode.
//
// Port of the MIT reference (github.com/anime-song/SheetSage2-Refined):
// grammar.py TokenGrammar (prompt-aware: chord_detail = bass then
// root x quality, key or key_mode), transcribe.py frame_table, and
// engine.py BatchDecoder._forward's rescoring and mode bias, for one window
// at a time with plain greedy decoding.
//
//   window pcm -> encoder (unpadded, + head layer mixes, sheetsage-encoder.h)
//              -> heads (sheetsage-heads.h) -> memory + bar features, frame tables
//              -> decoder prefill + greedy steps, logits += mode bias + rescoring,
//                 masked by the grammar
//
// Cross-attention needs no padding mask: the KV cache is sized to the
// window's real frame count, which is what the reference's -inf bias over
// the padding frames computes.
//
// Gate: yue2-probe --sheetsage-refined-transcribe against a reference decode
// (.tmp oracle scripts; see the converter's header for the pinned revision).

#include "sheetsage-decoder.h"
#include "sheetsage-encoder.h"
#include "sheetsage-heads.h"
#include "sheetsage-tokens.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

// Blocks appended after SheetSage2's 31678 tokens (vocab.py).
struct SheetSageRefinedTokens {
    static constexpr int32_t CHORD_BASS_START   = SheetSage2Tokens::N_TOKENS;  // 31678, 12 bass pitch classes
    static constexpr int32_t CHORD_BASS_END     = CHORD_BASS_START + 12;
    static constexpr int32_t N_QUALITIES        = 62;
    static constexpr int32_t NO_CHORD           = 12 * N_QUALITIES;  // 744: offset of the N chord
    static constexpr int32_t CHORD_DETAIL_START = CHORD_BASS_END;    // 31690, 12 * quality + root, N last
    static constexpr int32_t CHORD_DETAIL_END   = CHORD_DETAIL_START + NO_CHORD + 1;
    static constexpr int32_t N_MODES            = 9;
    static constexpr int32_t NO_KEY             = 12 * N_MODES;  // 108
    static constexpr int32_t KEY_MODE_START     = CHORD_DETAIL_END;  // 32435, 12 * mode + tonic, N last
    static constexpr int32_t KEY_MODE_END       = KEY_MODE_START + NO_KEY + 1;
    static constexpr int32_t SWING_START        = KEY_MODE_END;  // reserved, never generated
    static constexpr int32_t N_TOKENS           = SWING_START + 32;  // 32576

    static constexpr int32_t PROMPT_CHORD_DETAIL = SheetSage2Tokens::PROMPT_TOKEN_START + 8;
    static constexpr int32_t PROMPT_KEY_MODE     = SheetSage2Tokens::PROMPT_TOKEN_START + 9;
};

// vocab.prompt_prefix for the refined prompt set, in field order:
// timestamp, downbeat_meter, structure, key | key_mode, chord_detail, melody_full.
inline std::vector<int32_t> sheetsage_refined_prompt_prefix(bool key_mode) {
    using T = SheetSage2Tokens;
    const int32_t p = T::PROMPT_TOKEN_START;
    return { T::SOS, p + 0, p + 1, p + 2, key_mode ? SheetSageRefinedTokens::PROMPT_KEY_MODE : p + 3,
             SheetSageRefinedTokens::PROMPT_CHORD_DETAIL, p + 7, T::OUT };
}

// ── grammar (grammar.py) ────────────────────────────────────────────────────

struct SheetSageRefinedGrammar {
    enum Class : int8_t { EOS_T, SHIFT_T, TIME_T, METER_T, EIGHTH_T, STRUCTURE_T, KEY_T, CHORD_T, PITCH_T,
                          DURATION_T, BASS_T, DETAIL_T, OTHER };
    enum Incomplete : int8_t { NONE, AFTER_METER, AFTER_PITCH, AFTER_BASS };
    static constexpr int8_t FIELD_OF_CLASS[] = { -1, -1, 0, 1, 1, 2, 3, 4, 5, 5, 4, 4, -1 };

    std::vector<int8_t> cls;  // [N_TOKENS]

    bool       in_shift   = true;
    int32_t    shift_run  = 0;
    int32_t    payload    = 0;
    int32_t    last_field = -1;
    Incomplete incomplete = NONE;

    explicit SheetSageRefinedGrammar(bool key_mode) {
        using T = SheetSage2Tokens;
        using R = SheetSageRefinedTokens;
        cls.assign((size_t) R::N_TOKENS, OTHER);
        auto set = [&](int32_t a, int32_t b, Class c) {
            for (int32_t i = a; i < b; i++) {
                cls[(size_t) i] = c;
            }
        };
        cls[(size_t) T::EOS] = EOS_T;
        set(T::SUBBEAT_SHIFT_TOKEN_START, T::SUBBEAT_SHIFT_TOKEN_END, SHIFT_T);
        set(T::TIME_TOKEN_START, T::TIME_TOKEN_END, TIME_T);
        set(T::METER_TOKEN_START, T::METER_TOKEN_END, METER_T);
        set(T::EIGHTH_POSITION_TOKEN_START, T::EIGHTH_POSITION_TOKEN_END, EIGHTH_T);
        set(T::STRUCTURE_TOKEN_START, T::STRUCTURE_TOKEN_END, STRUCTURE_T);
        if (key_mode) {
            set(R::KEY_MODE_START, R::KEY_MODE_END, KEY_T);
        } else {
            set(T::KEY_TOKEN_START, T::KEY_TOKEN_END, KEY_T);
        }
        set(R::CHORD_BASS_START, R::CHORD_BASS_END, BASS_T);
        set(R::CHORD_DETAIL_START, R::CHORD_DETAIL_START + R::NO_CHORD, DETAIL_T);
        cls[(size_t) (R::CHORD_DETAIL_START + R::NO_CHORD)] = CHORD_T;
        set(T::PITCH_TOKEN_START, T::PITCH_TOKEN_END, PITCH_T);
        set(T::DURATION_TOKEN_START, T::DURATION_TOKEN_END, DURATION_T);
    }

    // Which classes may come next (TokenGrammar.allowed).
    void allowed_classes(bool out[OTHER + 1]) const {
        const bool free = incomplete == NONE;
        const bool open = incomplete == AFTER_BASS;
        out[EOS_T]       = payload > 0 && !open;
        out[SHIFT_T]     = (payload > 0 || in_shift) && shift_run < 4 && !open;
        out[TIME_T]      = free && last_field < 0;
        out[METER_T]     = free && last_field < 1;
        out[EIGHTH_T]    = incomplete == AFTER_METER || (free && last_field < 1);
        out[STRUCTURE_T] = free && last_field < 2;
        out[KEY_T]       = free && last_field < 3;
        out[CHORD_T]     = free && last_field < 4;
        out[PITCH_T]     = incomplete == AFTER_PITCH || (free && last_field <= 5);
        out[DURATION_T]  = incomplete == AFTER_PITCH;
        out[BASS_T]      = free && last_field < 4;
        out[DETAIL_T]    = open;
        out[OTHER]       = false;
    }

    // TokenGrammar.update for one token (finishing is the caller's business).
    void update(int32_t token) {
        const Class c        = (Class) cls[(size_t) token];
        const bool  is_shift = c == SHIFT_T;
        const bool  is_field = c != SHIFT_T && c != EOS_T;
        if (is_shift && !in_shift && payload > 0) {  // a new event
            payload    = 0;
            last_field = -1;
            incomplete = NONE;
        }
        if (is_field) {
            payload++;
            last_field = FIELD_OF_CLASS[c];
            incomplete = c == METER_T ? AFTER_METER : c == PITCH_T ? AFTER_PITCH : c == BASS_T ? AFTER_BASS : NONE;
            in_shift   = false;
            shift_run  = 0;
        } else if (is_shift) {
            in_shift = true;
            shift_run++;
        }
    }

    void prime(const std::vector<int32_t> & prefix) {
        size_t i = 0;
        while (i < prefix.size() && prefix[i] != SheetSage2Tokens::OUT) {
            i++;
        }
        for (i++; i < prefix.size(); i++) {
            update(prefix[i]);
        }
    }
};

// ── frame tables (transcribe.py frame_table) ────────────────────────────────

struct SheetSageRefinedTables {
    int64_t            T = 0;
    std::vector<float> root, bass;    // [T, 12]
    std::vector<float> quality;       // [T, 744]
    std::vector<float> key;           // [T, 24]
    std::vector<float> meter;         // [T, 16]
    std::vector<float> key_boundary;  // [T]
    std::vector<float> downbeat;      // [T]
};

namespace ss2r_tables {

// Row softmax of [T, n] logits, keeping the first `keep` columns.
inline std::vector<float> softmax_rows(const std::vector<float> & x, int64_t T, int n, int keep) {
    std::vector<float> p((size_t) (T * keep));
    for (int64_t t = 0; t < T; t++) {
        const float * r  = x.data() + t * n;
        float         mx = r[0];
        for (int i = 1; i < n; i++) {
            mx = std::max(mx, r[i]);
        }
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            s += std::exp((double) (r[i] - mx));
        }
        for (int i = 0; i < keep; i++) {
            p[(size_t) (t * keep + i)] = (float) (std::exp((double) (r[i] - mx)) / s);
        }
    }
    return p;
}

// Mean over the next `span` frames, the last frame repeated past the end.
inline std::vector<float> lookahead(const std::vector<float> & p, int64_t T, int n, int span) {
    std::vector<float> m((size_t) (T * n), 0.0f);
    for (int64_t t = 0; t < T; t++) {
        for (int j = 0; j < span; j++) {
            const float * r = p.data() + std::min<int64_t>(t + j, T - 1) * n;
            for (int i = 0; i < n; i++) {
                m[(size_t) (t * n + i)] += r[i];
            }
        }
        for (int i = 0; i < n; i++) {
            m[(size_t) (t * n + i)] /= (float) span;
        }
    }
    return m;
}

inline void log_clamped(std::vector<float> & v) {
    for (float & x : v) {
        x = std::log(std::max(x, 1e-6f));
    }
}

// Logit of the largest probability within `reach` frames, clamped to [0.02, 0.98].
inline std::vector<float> peak_logit(const std::vector<float> & p, int64_t T, int reach) {
    std::vector<float> out((size_t) T);
    for (int64_t t = 0; t < T; t++) {
        float mx = 0.0f;
        for (int64_t u = std::max<int64_t>(0, t - reach); u <= std::min<int64_t>(T - 1, t + reach); u++) {
            mx = std::max(mx, p[(size_t) u]);
        }
        mx               = std::min(std::max(mx, 0.02f), 0.98f);
        out[(size_t) t] = std::log(mx) - std::log(1.0f - mx);
    }
    return out;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace ss2r_tables

// The tables the rescoring terms read, from one window's head outputs
// (unchunked GRU, as the reference's rescoring heads run).
inline void sheetsage_refined_tables(const SheetSageRefinedHeads & rh, const SheetSageRefinedHeadOutputs & ho,
                                     SheetSageRefinedTables * tab) {
    using namespace ss2r_tables;
    const int64_t T = ho.T;
    *tab            = SheetSageRefinedTables{};
    tab->T          = T;
    auto la         = [&](const char * term, int fallback) {
        auto it = rh.lookahead.find(term);
        return it == rh.lookahead.end() ? fallback : it->second;
    };
    auto rc = [&](const char * term, int fallback) {
        auto it = rh.reach.find(term);
        return it == rh.reach.end() ? fallback : it->second;
    };
    const auto & chord  = ho.full.at("chord");
    const auto & detail = ho.full.at("detail");
    const auto & multi  = ho.full.at("multi");
    const auto & bar0   = ho.full.at("bar0");
    const auto & bar1   = ho.full.at("bar1");

    tab->root = lookahead(softmax_rows(chord.at("root"), T, 13, 12), T, 12, la("root", 12));
    log_clamped(tab->root);
    tab->bass = lookahead(softmax_rows(chord.at("bass"), T, 13, 12), T, 12, la("bass", 12));
    log_clamped(tab->bass);

    const int          ND = SheetSageRefinedTokens::NO_CHORD;
    std::vector<float> q  = lookahead(softmax_rows(detail.at("detail"), T, ND + 1, ND), T, ND, la("quality", 12));
    tab->quality.resize(q.size());
    for (int64_t t = 0; t < T; t++) {
        float roots[12] = {};
        for (int i = 0; i < ND; i++) {
            roots[i % 12] += q[(size_t) (t * ND + i)];
        }
        for (int i = 0; i < ND; i++) {
            tab->quality[(size_t) (t * ND + i)] =
                std::log(std::max(q[(size_t) (t * ND + i)], 1e-6f)) - std::log(std::max(roots[i % 12], 1e-6f));
        }
    }

    tab->key = lookahead(softmax_rows(multi.at("key"), T, 24, 24), T, 24, la("key", 50));
    log_clamped(tab->key);

    // Bar heads averaged as HeadMean does: mean softmax, renormalised; mean sigmoid.
    std::vector<float> b0 = softmax_rows(bar0.at("bar_beats"), T, 16, 16);
    std::vector<float> b1 = softmax_rows(bar1.at("bar_beats"), T, 16, 16);
    for (int64_t t = 0; t < T; t++) {
        double s = 0.0;
        for (int i = 0; i < 16; i++) {
            const size_t k = (size_t) (t * 16 + i);
            b0[k]          = std::max(0.5f * (b0[k] + b1[k]), 1e-9f);
            s += b0[k];
        }
        for (int i = 0; i < 16; i++) {
            b0[(size_t) (t * 16 + i)] = (float) (b0[(size_t) (t * 16 + i)] / s);
        }
    }
    tab->meter = lookahead(b0, T, 16, la("meter", 8));
    log_clamped(tab->meter);

    std::vector<float> kb((size_t) T), db((size_t) T);
    for (int64_t t = 0; t < T; t++) {
        kb[(size_t) t] = sigmoid(multi.at("key_boundary")[(size_t) t]);
        const float p  = 0.5f * (sigmoid(bar0.at("downbeat")[(size_t) t]) + sigmoid(bar1.at("downbeat")[(size_t) t]));
        db[(size_t) t] = std::min(std::max(p, 1e-6f), 1.0f - 1e-6f);
    }
    tab->key_boundary = peak_logit(kb, T, rc("key_boundary", 25));
    tab->downbeat     = peak_logit(db, T, rc("downbeat", 2));
}

// ── rescoring (engine.py _forward) ──────────────────────────────────────────

struct SheetSageRefinedRescorer {
    const SheetSageRefinedHeads &  rh;
    const SheetSageRefinedTables & tab;
    bool                           key_mode;
    int64_t                        frames;  // the reference engine's frame capacity (one 300 s window)

    int64_t now           = 0;
    bool    key_seen      = false;
    int32_t active_meter  = -1;  // meter index, -1 none
    bool    meter_changed = false;
    std::vector<float> mode_bias;  // [N_MODES], empty unless key_mode

    float w(const char * term) const {
        auto it = rh.rescore_weight.find(term);
        return it == rh.rescore_weight.end() ? 0.0f : it->second;
    }

    SheetSageRefinedRescorer(const SheetSageRefinedHeads & h, const SheetSageRefinedTables & t, bool km, int64_t f) :
        rh(h),
        tab(t),
        key_mode(km),
        frames(f) {
        if (key_mode && rh.mode_tau != 0.0f && rh.mode_counts.size() == SheetSageRefinedTokens::N_MODES) {
            double sum = 0.0;
            for (float c : rh.mode_counts) {
                sum += c;
            }
            for (float c : rh.mode_counts) {
                mode_bias.push_back((float) (-(double) rh.mode_tau * std::log((double) c / sum)));
            }
        }
    }

    void prime(const std::vector<int32_t> & prefix) {
        using T = SheetSage2Tokens;
        int32_t last_time = -1;
        for (int32_t tok : prefix) {
            if (tok >= T::TIME_TOKEN_START && tok < T::TIME_TOKEN_END) {
                last_time = tok - T::TIME_TOKEN_START;
            }
            key_seen = key_seen || (tok >= T::KEY_TOKEN_START && tok < T::KEY_TOKEN_END);
            if (tok >= T::METER_TOKEN_START && tok < T::METER_TOKEN_END) {
                active_meter = tok - T::METER_TOKEN_START;
            }
        }
        // int(times[-1] * time_hz) * 25 // time_hz, in the reference's float64.
        now = last_time < 0 ? 0 :
                              std::min<int64_t>((int64_t) ((double) last_time / 100.0 * 100.0) * 25 / 100, frames - 1);
        meter_changed = false;
    }

    // Meter index -> (numerator, denominator), vocab.METERS order.
    static int32_t meter_beats(int32_t m) { return std::min(m / 6 + 1, 16) - 1; }

    static int32_t meter_unit(int32_t m) { return 1 << (m % 6); }

    void apply(float * logits, bool after_meter) const {
        using T = SheetSage2Tokens;
        using R = SheetSageRefinedTokens;
        if (!mode_bias.empty()) {
            for (int m = 0; m < R::N_MODES; m++) {
                for (int t = 0; t < 12; t++) {
                    logits[R::KEY_MODE_START + m * 12 + t] += mode_bias[(size_t) m];
                }
            }
        }
        if (now >= tab.T) {  // past the heads' frames: the reference's zero-filled table rows
            if (w("meter") != 0.0f && after_meter && meter_changed) {
                for (int32_t e = 1; e < T::N_EIGHTH_POSITIONS; e++) {
                    logits[T::EIGHTH_POSITION_TOKEN_START + e] = -INFINITY;
                }
            }
            return;
        }
        const int ND = R::NO_CHORD;
        if (const float wr = w("root")) {
            const float * r = tab.root.data() + now * 12;
            for (int i = 0; i < ND; i++) {
                logits[R::CHORD_DETAIL_START + i] += wr * r[i % 12];
            }
        }
        if (const float wq = w("quality")) {
            const float * q = tab.quality.data() + now * ND;
            for (int i = 0; i < ND; i++) {
                logits[R::CHORD_DETAIL_START + i] += wq * q[i];
            }
        }
        if (const float wb = w("bass")) {
            const float * b = tab.bass.data() + now * 12;
            for (int i = 0; i < 12; i++) {
                logits[R::CHORD_BASS_START + i] += wb * b[i];
            }
        }
        if (!key_mode) {  // the key terms move the `key` prompt's tokens only
            if (const float wk = w("key")) {
                const float * k = tab.key.data() + now * 24;
                for (int i = 0; i < 24; i++) {
                    logits[T::KEY_TOKEN_START + i] += wk * k[i];
                }
            }
            if (const float wkb = w("key_boundary")) {
                if (key_seen) {
                    for (int i = 0; i < 24; i++) {
                        logits[T::KEY_TOKEN_START + i] += wkb * tab.key_boundary[(size_t) now];
                    }
                }
            }
        }
        const float wd = w("downbeat");
        if (wd != 0.0f) {
            logits[T::EIGHTH_POSITION_TOKEN_START] += wd * tab.downbeat[(size_t) now];
        }
        if (const float wm = w("meter")) {
            const float * table   = tab.meter.data() + now * 16;
            const int32_t active  = std::max(active_meter, 0);
            const float   current = table[meter_beats(active)];
            for (int32_t m = 0; m < T::N_METER_PAIRS; m++) {
                const bool change = active_meter >= 0 && m != active;
                if (change && meter_unit(m) == meter_unit(active)) {
                    logits[T::METER_TOKEN_START + m] += wm * (table[meter_beats(m)] - current);
                }
                if (change && wd != 0.0f) {
                    logits[T::METER_TOKEN_START + m] += wd * tab.downbeat[(size_t) now];
                }
            }
            if (after_meter && meter_changed) {
                for (int32_t e = 1; e < T::N_EIGHTH_POSITIONS; e++) {
                    logits[T::EIGHTH_POSITION_TOKEN_START + e] = -INFINITY;
                }
            }
        }
    }

    void update(int32_t tok) {
        using T = SheetSage2Tokens;
        if (tok >= T::TIME_TOKEN_START && tok < T::TIME_TOKEN_END) {
            const double seconds = (double) (tok - T::TIME_TOKEN_START) / 100.0;
            now                  = std::min<int64_t>(std::max<int64_t>((int64_t) (seconds * 25.0), 0), frames - 1);
        }
        key_seen = key_seen || (tok >= T::KEY_TOKEN_START && tok < T::KEY_TOKEN_END);
        if (tok >= T::METER_TOKEN_START && tok < T::METER_TOKEN_END) {
            const int32_t m = tok - T::METER_TOKEN_START;
            meter_changed   = active_meter >= 0 && m != active_meter;
            active_meter    = m;
        }
    }
};

// ── one window ──────────────────────────────────────────────────────────────

struct SheetSageRefinedWindowResult {
    std::vector<int32_t> tokens;  // prefix + generated, always ending in <eos>
    const char *         ended = "running";  // eos | stop | limit
    int64_t              frames = 0;         // real encoder frames of the window
    double               ms_encode = 0.0, ms_heads = 0.0, ms_decode = 0.0;
};

// Decode one window of mono 24 kHz audio (`pcm`, `n` samples, at most one
// window long) from `prefix` (ending in <out>, or an overlap prefix), stopping
// on <eos>, on a time token at or after `stop_seconds`, or at the context limit.
inline bool sheetsage_refined_decode_window(const SheetSageModel & m, SheetSageEncoderGraph * g, const float * pcm,
                                            int64_t n, const std::vector<int32_t> & prefix, double stop_seconds,
                                            bool key_mode, SheetSageRefinedWindowResult * out, std::string * err) {
    using T = SheetSage2Tokens;
    *out    = SheetSageRefinedWindowResult{};
    if (!m.refined) {
        if (err) {
            *err = "sheetsage_refined_decode_window: model was loaded without the refined overlay";
        }
        return false;
    }
    const SheetSageRefinedHeads & rh = m.rheads;

    auto                   t0 = std::chrono::steady_clock::now();
    SheetSageEncodeOptions eo;
    eo.pad_to_window     = false;
    eo.extra_mix_weights = sheetsage_refined_mix_weights(rh);
    SheetSageEncodeResult er;
    if (!sheetsage_encode_window(m, g, pcm, n, eo, &er, err)) {
        return false;
    }
    auto t1        = std::chrono::steady_clock::now();
    out->ms_encode = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out->frames    = er.T_sub;

    std::vector<const float *> mixes;
    for (const auto & v : er.extra_mixes) {
        mixes.push_back(v.data());
    }
    SheetSageRefinedHeadOutputs ho;
    sheetsage_refined_run_heads(rh, mixes, er.T_sub, &ho);
    sheetsage_refined_add_bar_features(rh, ho.bar_facts, ho.T, er.memory.data(), er.T_sub);
    SheetSageRefinedTables tab;
    sheetsage_refined_tables(rh, ho, &tab);
    auto t2       = std::chrono::steady_clock::now();
    out->ms_heads = std::chrono::duration<double, std::milli>(t2 - t1).count();

    const int64_t frames = sheetsage_window_samples(m) / (int64_t) m.mert.cfg.samples_per_frame;  // 7500
    SheetSageRefinedGrammar  gr(key_mode);
    SheetSageRefinedRescorer rs(rh, tab, key_mode, frames);
    gr.prime(prefix);
    rs.prime(prefix);

    SheetSageDecKvCache cache;
    if (!sheetsage_dec_kv_cache_alloc(m, (int64_t) m.cfg.max_output_seq_len, er.T_sub, &cache, err)) {
        return false;
    }
    SheetSageDecodeResult res;
    std::vector<int64_t>  lp = { (int64_t) prefix.size() - 1 };
    bool ok = sheetsage_dec_prefill(m, cache, er.memory.data(), er.T_sub, prefix, lp, &res, err);
    out->tokens = prefix;
    const int64_t max_len = (int64_t) m.cfg.max_output_seq_len;
    std::vector<float> logits;
    bool               allow[SheetSageRefinedGrammar::OTHER + 1];
    while (ok) {
        logits = std::move(res.logits);
        rs.apply(logits.data(), gr.incomplete == SheetSageRefinedGrammar::AFTER_METER);
        gr.allowed_classes(allow);
        int32_t best   = -1;
        float   best_v = -INFINITY;
        for (int32_t v = 0; v < (int32_t) logits.size(); v++) {
            if (allow[gr.cls[(size_t) v]] && logits[(size_t) v] > best_v) {
                best_v = logits[(size_t) v];
                best   = v;
            }
        }
        if (best < 0) {
            if (err) {
                *err = "sheetsage2-refined decode: no token allowed (grammar mask is empty)";
            }
            ok = false;
            break;
        }
        gr.update(best);
        rs.update(best);
        out->tokens.push_back(best);
        if (best == T::EOS) {
            out->ended = "eos";
            break;
        }
        if (best >= T::TIME_TOKEN_START && best < T::TIME_TOKEN_END &&
            (double) (best - T::TIME_TOKEN_START) / 100.0 >= stop_seconds) {
            out->ended = "stop";
            break;
        }
        if ((int64_t) out->tokens.size() >= max_len) {
            out->ended = "limit";
            break;
        }
        ok = sheetsage_dec_step(m, cache, best, &res.logits, err);
    }
    sheetsage_dec_kv_cache_free(&cache);
    if (ok && out->tokens.back() != T::EOS) {
        out->tokens.push_back(T::EOS);
    }
    out->ms_decode = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    return ok;
}
