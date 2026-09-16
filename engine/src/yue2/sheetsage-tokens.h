#pragma once
// yue2/sheetsage-tokens.h — SheetSage2 tokenizer vocabulary: block layout,
// id<->label tables, token_type() classification, and the vocabulary
// fingerprint self-check.
//
// HOT-Step file (no acestep.cpp analog). SheetSage2 has no vocabulary file on
// disk — the 31,678 token ids are a pure function of the schema/label/duration
// constants below, exactly as `SheetSage2Tokenizer.__init__` builds them
// (`tokenization_sheetsage2.py:80-187`). This file reproduces that
// construction in C++ so nothing Python runs at cache or inference time
// (doc 19 decision 1).
//
// AUTHORITY: docs/plans/yue2/21-sheetsage2-symbolic-pin.md §0-§1 (vocabulary),
// cross-checked directly against the shipped module at
// K:\yue2\.cache\huggingface-sheetsage\modules\transformers_modules\SheetSage2\
// {tokenization,schema,labels,durations}_sheetsage2.py. Where this file and
// that document disagree, the document wins (and the Python under it wins
// over the document — see the doc's own note). Section references (§1.2,
// §1.5, §1.8 ...) point into doc 21.
//
// THE FINGERPRINT IS THE GATE. `tokenizer_fingerprint` in the HF repo's
// config.json is "5ba3325af0344c7f" (doc 21 §0). yue2_sheet_tokens_build()
// recomputes it from these tables at build time (§1.8's exact canonical-JSON
// recipe: json.dumps(payload, sort_keys=True, separators=(",",":")), UTF-8,
// SHA-256, first 16 hex chars) and refuses to report success if it doesn't
// match — that means the vocabulary tables below are wrong before a single
// tensor or a single decode step is touched. No exceptions cross this
// boundary; every entry point here returns a bool and writes *err on failure.
//
// Every range below is [start, end), half-open, in the exact order
// `tokenization_sheetsage2.py:103-187` computes them. token_type() below
// mirrors `SheetSage2Tokenizer.token_type` (`:665-694`) range-for-range.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// windows.h (pulled in transitively via backend.h -- e.g. once this header
// and any GGML backend code share a translation unit, as sheetsage-
// pipeline.h/yue2-probe.cpp now do) #defines OUT (and IN/OPTIONAL) as empty
// SAL-annotation macros, in force for the rest of the translation unit from
// wherever windows.h was first included. `SheetSage2Tokens::OUT` below --
// and every `T::OUT`/`SheetSage2Tokens::OUT` reference in sheetsage-events.h/
// sheetsage-grammar.h/sheetsage-pipeline.h -- names doc 21's `<|out|>` token,
// never the SAL keyword; left alone it macro-expands to nothing and breaks
// `static constexpr int32_t OUT = 3;` into a syntax error the moment this
// header lands in the same TU as windows.h. This was invisible while these
// symbolic-lane files only ever built standalone (engine/tests/sheetsage/,
// no windows.h in that TU at all) and surfaced only once sheetsage-
// pipeline.h/yue2-probe.cpp needed both the symbolic lane and the GGML
// backend together.
#ifdef OUT
#undef OUT
#endif

// ─────────────────────────────────────────────────────────────────────────
// §A. A small, self-contained SHA-256 (no exceptions, no external deps).
//     There is no sha256 helper anywhere else in this tree (checked:
//     engine/src/train/yue2-preprocess-run.h and yue2-ar-train-run.h both
//     note they use FNV-1a / a non-cryptographic hash instead, not sha256).
//     This is the standard FIPS 180-4 block algorithm; verified against the
//     NIST test vectors ("abc" and the empty string) in
//     yue2_sha256_self_test() below, which the table builder runs once.
// ─────────────────────────────────────────────────────────────────────────

namespace yue2_sheet_sha256_detail {

inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

struct Sha256State {
    uint32_t h[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
};

inline void sha256_compress(Sha256State & st, const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
    };
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[i*4]) << 24) | (uint32_t(block[i*4+1]) << 16) |
               (uint32_t(block[i*4+2]) << 8) | uint32_t(block[i*4+3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=st.h[0],b=st.h[1],c=st.h[2],d=st.h[3],e=st.h[4],f=st.h[5],g=st.h[6],h=st.h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + k[i] + w[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st.h[0]+=a; st.h[1]+=b; st.h[2]+=c; st.h[3]+=d;
    st.h[4]+=e; st.h[5]+=f; st.h[6]+=g; st.h[7]+=h;
}

}  // namespace yue2_sheet_sha256_detail

// Returns the lowercase hex SHA-256 digest (64 chars) of `data`.
inline std::string yue2_sha256_hex(const std::string & data) {
    using namespace yue2_sheet_sha256_detail;
    Sha256State st;
    uint64_t bit_len = uint64_t(data.size()) * 8;
    std::vector<uint8_t> msg(data.begin(), data.end());
    msg.push_back(0x80);
    while (msg.size() % 64 != 56) msg.push_back(0x00);
    for (int i = 7; i >= 0; --i) msg.push_back(uint8_t((bit_len >> (i*8)) & 0xff));
    for (size_t off = 0; off < msg.size(); off += 64) sha256_compress(st, &msg[off]);
    static const char * hexd = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (int i = 0; i < 8; ++i) {
        uint32_t v = st.h[i];
        for (int j = 0; j < 4; ++j) {
            uint8_t byte = uint8_t((v >> (24 - j*8)) & 0xff);
            out[i*8 + j*2]     = hexd[byte >> 4];
            out[i*8 + j*2 + 1] = hexd[byte & 0xf];
        }
    }
    return out;
}

// FIPS 180-4 test vectors. Run once by yue2_sheet_tokens_build(); a failure
// here means the sha256 implementation itself is broken, not the vocabulary.
inline bool yue2_sha256_self_test(std::string * err) {
    std::string h1 = yue2_sha256_hex("");
    std::string h2 = yue2_sha256_hex("abc");
    if (h1 != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        if (err) *err = "sha256 self-test failed on empty string: got " + h1;
        return false;
    }
    if (h2 != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        if (err) *err = "sha256 self-test failed on \"abc\": got " + h2;
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// §B. Block layout constants (doc 21 §1.2). All [start,end) half-open.
//     Fixed for schema "v1", audio_length_seconds=300.0, time_hz=100 — the
//     only configuration this port ever uses (doc 19/21 assume it
//     throughout; there is no runtime schema switch).
// ─────────────────────────────────────────────────────────────────────────

struct SheetSage2Tokens {
    // Specials (tokenization_sheetsage2.py:97-101)
    static constexpr int32_t PAD = 0;
    static constexpr int32_t SOS = 1;
    static constexpr int32_t BOS = SOS;  // bos_token is a plain alias of sos_token
    static constexpr int32_t EOS = 2;
    static constexpr int32_t OUT = 3;

    // Prompt block: 256-wide reserved capacity, only 8 ids ever assigned
    // (doc 21 §1.2/§1.3). Dead ids [12,260) are never legitimately produced.
    static constexpr int32_t PROMPT_CAPACITY      = 256;
    static constexpr int32_t PROMPT_TOKEN_START   = 4;
    static constexpr int32_t N_PROMPT_TASKS        = 8;  // V1_TASKS (schema_sheetsage2.py:28-37)
    static constexpr int32_t PROMPT_TASKS_END      = PROMPT_TOKEN_START + N_PROMPT_TASKS;  // 12
    static constexpr int32_t PROMPT_TOKEN_END      = PROMPT_TOKEN_START + PROMPT_CAPACITY; // 260

    static constexpr int32_t MAX_SUBBEAT_SHIFT          = 256;
    static constexpr int32_t SUBBEAT_SHIFT_TOKEN_START  = PROMPT_TOKEN_END;                       // 260
    static constexpr int32_t N_SUBBEAT_SHIFT_TOKENS     = MAX_SUBBEAT_SHIFT + 1;                  // 257
    static constexpr int32_t SUBBEAT_SHIFT_TOKEN_END    = SUBBEAT_SHIFT_TOKEN_START + N_SUBBEAT_SHIFT_TOKENS; // 517

    static constexpr double  AUDIO_LENGTH_SECONDS = 300.0;
    static constexpr int32_t TIME_HZ              = 100;
    static constexpr int32_t N_TIME_TOKENS        = 30000;  // round(300.0 * 100)
    static constexpr int32_t TIME_TOKEN_START     = SUBBEAT_SHIFT_TOKEN_END;              // 517
    static constexpr int32_t TIME_TOKEN_END       = TIME_TOKEN_START + N_TIME_TOKENS;     // 30517

    static constexpr int32_t N_METER_NUMERATORS   = 32;   // 1..32
    static constexpr int32_t N_METER_DENOMINATORS = 6;    // (1,2,4,8,16,32)
    static constexpr int32_t N_METER_PAIRS        = N_METER_NUMERATORS * N_METER_DENOMINATORS; // 192
    static constexpr int32_t METER_TOKEN_START    = TIME_TOKEN_END;                       // 30517
    static constexpr int32_t METER_TOKEN_END      = METER_TOKEN_START + N_METER_PAIRS;    // 30709

    static constexpr int32_t N_EIGHTH_POSITIONS          = 256;
    static constexpr int32_t EIGHTH_POSITION_TOKEN_START = METER_TOKEN_END;                              // 30709
    static constexpr int32_t EIGHTH_POSITION_TOKEN_END   = EIGHTH_POSITION_TOKEN_START + N_EIGHTH_POSITIONS; // 30965

    static constexpr int32_t N_STRUCTURE_LABELS   = 23;
    static constexpr int32_t STRUCTURE_TOKEN_START = EIGHTH_POSITION_TOKEN_END;                    // 30965
    static constexpr int32_t STRUCTURE_TOKEN_END   = STRUCTURE_TOKEN_START + N_STRUCTURE_LABELS;   // 30988

    static constexpr int32_t N_KEY_TOKENS      = 24;  // 12 tonics x 2 modes
    static constexpr int32_t KEY_TOKEN_START   = STRUCTURE_TOKEN_END;              // 30988
    static constexpr int32_t KEY_TOKEN_END     = KEY_TOKEN_START + N_KEY_TOKENS;   // 31012

    static constexpr int32_t N_MAJMIN_CHORDS          = 25;  // "N" + 12 maj + 12 min
    static constexpr int32_t MAJMIN_CHORD_TOKEN_START = KEY_TOKEN_END;                              // 31012
    static constexpr int32_t MAJMIN_CHORD_TOKEN_END   = MAJMIN_CHORD_TOKEN_START + N_MAJMIN_CHORDS; // 31037

    static constexpr int32_t N_FULL_CHORDS          = 361;  // doc 21 §1.5
    static constexpr int32_t FULL_CHORD_TOKEN_START = MAJMIN_CHORD_TOKEN_END;                    // 31037
    static constexpr int32_t FULL_CHORD_TOKEN_END   = FULL_CHORD_TOKEN_START + N_FULL_CHORDS;    // 31398

    static constexpr int32_t N_PITCH_TOKENS    = 256;  // 128 MIDI pitches x 2 melody tracks
    static constexpr int32_t PITCH_TOKEN_START = FULL_CHORD_TOKEN_END;                // 31398
    static constexpr int32_t PITCH_TOKEN_END   = PITCH_TOKEN_START + N_PITCH_TOKENS;  // 31654

    static constexpr int32_t N_DURATION_TOKENS    = 24;
    static constexpr int32_t DURATION_TOKEN_START = PITCH_TOKEN_END;                          // 31654
    static constexpr int32_t DURATION_TOKEN_END   = DURATION_TOKEN_START + N_DURATION_TOKENS; // 31678

    // v1's appended_token_blocks is empty (schema_sheetsage2.py:41-52), so the
    // vocabulary ends exactly at DURATION_TOKEN_END.
    static constexpr int32_t N_TOKENS = DURATION_TOKEN_END;  // 31678, == config.json vocab_size

    // ---- Dynamic tables (built by yue2_sheet_tokens_build) ----
    std::array<std::string, N_PROMPT_TASKS> prompt_names;         // schema order (V1_TASKS)
    std::unordered_map<std::string, int32_t> prompt_name_to_id;   // name -> token id

    std::array<std::pair<int32_t, int32_t>, N_METER_PAIRS> meter_pairs;  // id -> (numerator, denominator)
    std::unordered_map<int64_t, int32_t> meter_pair_to_id;               // (num<<32|den) -> id, for encode direction

    std::array<std::string, N_STRUCTURE_LABELS> structure_labels;

    std::array<std::string, N_MAJMIN_CHORDS> majmin_chord_labels;

    std::array<std::string, N_FULL_CHORDS> full_chord_labels;
    std::unordered_map<std::string, int32_t> full_chord_to_id;  // exact-string lookup (tokenization_sheetsage2.py:158-160)

    std::array<int32_t, N_DURATION_TOKENS> duration_templates;  // subbeat units (durations_sheetsage2.py:4-11)

    std::array<std::string, 6> event_field_order;  // schema v1: timestamp,rhythm,structure,key,chord,melody

    std::string fingerprint;  // first 16 hex chars of the SHA-256, must equal "5ba3325af0344c7f"
};

// Pinned constant from config.json / doc 21 §0. The table builder must
// reproduce this from the tables alone, with zero runtime input.
inline const char * yue2_sheet_expected_fingerprint() { return "5ba3325af0344c7f"; }

// ─────────────────────────────────────────────────────────────────────────
// §C. Table construction (tokenization_sheetsage2.py:__init__ + module-level
//     constants in the same file plus schema_sheetsage2.py/labels_sheetsage2.py/
//     durations_sheetsage2.py). Pure, deterministic, no I/O.
// ─────────────────────────────────────────────────────────────────────────

namespace yue2_sheet_detail {

// CHROMATIC_SHARPS (tokenization_sheetsage2.py:16-29) — sharp spelling only.
inline const std::array<std::string, 12> & chromatic_sharps() {
    static const std::array<std::string, 12> v = {
        "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
    };
    return v;
}

// FULL_CHORD_QUALITIES (:31-47), exact order.
inline const std::array<std::string, 15> & full_chord_qualities() {
    static const std::array<std::string, 15> v = {
        "maj","min","dim","aug","maj7","min7","7","hdim7","dim7",
        "minmaj7","sus2","sus4","sus4(b7)","maj6","min6",
    };
    return v;
}

// FULL_CHORD_INVERSIONS (:49-55) — only these 5 qualities carry inversions.
inline const std::vector<std::string> & full_chord_inversions_for(const std::string & quality) {
    static const std::vector<std::string> maj  = {"/2","/3","/5"};
    static const std::vector<std::string> min_ = {"/2","/b3","/5"};
    static const std::vector<std::string> maj7 = {"/3","/5","/7"};
    static const std::vector<std::string> min7 = {"/b3","/5","/b7"};
    static const std::vector<std::string> sev  = {"/3","/5","/b7"};
    static const std::vector<std::string> none;
    if (quality == "maj")  return maj;
    if (quality == "min")  return min_;
    if (quality == "maj7") return maj7;
    if (quality == "min7") return min7;
    if (quality == "7")    return sev;
    return none;
}

// build_full_chord_vocabulary() (tokenization_sheetsage2.py:58-65): quality
// outer, root middle, inversion inner, plain-root last.
inline void build_full_chord_labels(std::array<std::string, SheetSage2Tokens::N_FULL_CHORDS> & out) {
    size_t i = 0;
    out[i++] = "N";
    for (const std::string & quality : full_chord_qualities()) {
        const std::vector<std::string> & inversions = full_chord_inversions_for(quality);
        for (const std::string & root : chromatic_sharps()) {
            for (size_t inv_i = 0; inv_i <= inversions.size(); ++inv_i) {
                const std::string & inversion = (inv_i < inversions.size()) ? inversions[inv_i] : std::string();
                out[i++] = root + ":" + quality + inversion;
            }
        }
    }
    // count check mirrors doc 21 §1.5: 5 qualities x 12 roots x 4 + 10 x 12 x 1 + 1 = 361
}

// majmin_chord_labels (tokenization_sheetsage2.py:147-151).
inline void build_majmin_chord_labels(std::array<std::string, SheetSage2Tokens::N_MAJMIN_CHORDS> & out) {
    size_t i = 0;
    out[i++] = "N";
    for (const std::string & root : chromatic_sharps()) out[i++] = root + ":maj";
    for (const std::string & root : chromatic_sharps()) out[i++] = root + ":min";
}

// STRUCTURE_LABELS (labels_sheetsage2.py:8-10 — the SECOND module-level
// assignment; the first at :4-5 is dead due to plain Python name rebinding).
inline void build_structure_labels(std::array<std::string, SheetSage2Tokens::N_STRUCTURE_LABELS> & out) {
    static const char * labels[SheetSage2Tokens::N_STRUCTURE_LABELS] = {
        "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus",
        "interlude", "fade-out", "loop", "rap", "preshot", "irregular", "instrumental",
        "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
        "development", "variation", "pre-outro",
    };
    for (int i = 0; i < SheetSage2Tokens::N_STRUCTURE_LABELS; ++i) out[i] = labels[i];
}

// DURATION_TEMPLATES (durations_sheetsage2.py:4-11), subbeat units.
inline void build_duration_templates(std::array<int32_t, SheetSage2Tokens::N_DURATION_TOKENS> & out) {
    static const int32_t v[SheetSage2Tokens::N_DURATION_TOKENS] = {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
        96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
    };
    for (int i = 0; i < SheetSage2Tokens::N_DURATION_TOKENS; ++i) out[i] = v[i];
}

// meter_pairs (tokenization_sheetsage2.py:74-75, :125-129): numerator outer
// (1..32), denominator inner (1,2,4,8,16,32).
inline void build_meter_pairs(std::array<std::pair<int32_t, int32_t>, SheetSage2Tokens::N_METER_PAIRS> & out) {
    static const int32_t denominators[SheetSage2Tokens::N_METER_DENOMINATORS] = {1, 2, 4, 8, 16, 32};
    int idx = 0;
    for (int32_t numerator = 1; numerator <= SheetSage2Tokens::N_METER_NUMERATORS; ++numerator) {
        for (int d = 0; d < SheetSage2Tokens::N_METER_DENOMINATORS; ++d) {
            out[idx++] = {numerator, denominators[d]};
        }
    }
}

// Canonical-JSON string for the fingerprint payload (tokenization_sheetsage2.py:
// _compute_fingerprint, :200-222; doc 21 §1.8). Field order is alphabetical by
// key, separators "," and ":" with no spaces, ints unquoted, strings
// double-quoted, and `300.0` (not `300`) for audio_length_seconds to match
// Python's json.dumps(300.0) repr exactly. This function was cross-checked
// byte-for-byte against a standalone Python re-derivation of the same payload
// (json.dumps(payload, sort_keys=True, separators=(",",":"))), which reproduced
// "5ba3325af0344c7f" independently of this C++ implementation.
inline std::string json_string(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += char(c);
                }
        }
    }
    out += '"';
    return out;
}

template <typename Container>
inline std::string json_string_array(const Container & items) {
    std::string out = "[";
    bool first = true;
    for (const auto & s : items) {
        if (!first) out += ',';
        first = false;
        out += json_string(s);
    }
    out += ']';
    return out;
}

inline std::string canonical_fingerprint_json(const SheetSage2Tokens & t) {
    std::string out = "{";
    out += "\"appended_token_blocks\":[]";  // v1 declares none (schema_sheetsage2.py:41-52)
    out += ",\"audio_length_seconds\":300.0";  // matches Python's json.dumps(300.0)
    out += ",\"duration_templates\":[";
    for (int i = 0; i < SheetSage2Tokens::N_DURATION_TOKENS; ++i) {
        if (i) out += ',';
        out += std::to_string(t.duration_templates[i]);
    }
    out += "]";
    out += ",\"event_field_order\":" + json_string_array(t.event_field_order);
    out += ",\"full_chord_labels\":" + json_string_array(t.full_chord_labels);
    out += ",\"majmin_chord_labels\":" + json_string_array(t.majmin_chord_labels);
    out += ",\"meter_pairs\":[";
    for (int i = 0; i < SheetSage2Tokens::N_METER_PAIRS; ++i) {
        if (i) out += ',';
        out += "[" + std::to_string(t.meter_pairs[i].first) + "," + std::to_string(t.meter_pairs[i].second) + "]";
    }
    out += "]";
    out += ",\"n_tokens\":" + std::to_string(SheetSage2Tokens::N_TOKENS);
    out += ",\"prompt_capacity\":" + std::to_string(SheetSage2Tokens::PROMPT_CAPACITY);
    out += ",\"prompt_names\":" + json_string_array(t.prompt_names);
    out += ",\"schema_version\":\"v1\"";
    out += ",\"structure_labels\":" + json_string_array(t.structure_labels);
    out += ",\"time_hz\":" + std::to_string(SheetSage2Tokens::TIME_HZ);
    out += "}";
    return out;
}

}  // namespace yue2_sheet_detail

// Builds every table and checks the fingerprint. Returns false (with *err
// set) if the sha256 self-test or the fingerprint check fails — a hard bug
// in this file, never a runtime/data condition. `out` is always fully
// populated either way (callers that only need the tables, e.g. a fixture
// dump tool, may ignore a false return at their own risk; the grammar/decode
// code paths must treat it as fatal).
inline bool yue2_sheet_tokens_build(SheetSage2Tokens & out, std::string * err) {
    if (!yue2_sha256_self_test(err)) return false;

    // Prompt names: V1_TASKS in schema order (schema_sheetsage2.py:28-37).
    static const char * prompt_names[SheetSage2Tokens::N_PROMPT_TASKS] = {
        "timestamp", "downbeat_meter", "structure", "key",
        "chord_majmin", "chord_full", "melody_vocal", "melody_full",
    };
    for (int i = 0; i < SheetSage2Tokens::N_PROMPT_TASKS; ++i) {
        out.prompt_names[i] = prompt_names[i];
        out.prompt_name_to_id[prompt_names[i]] = SheetSage2Tokens::PROMPT_TOKEN_START + i;
    }

    yue2_sheet_detail::build_meter_pairs(out.meter_pairs);
    for (int i = 0; i < SheetSage2Tokens::N_METER_PAIRS; ++i) {
        int64_t key = (int64_t(out.meter_pairs[i].first) << 32) | uint32_t(out.meter_pairs[i].second);
        out.meter_pair_to_id[key] = i;
    }

    yue2_sheet_detail::build_structure_labels(out.structure_labels);
    yue2_sheet_detail::build_majmin_chord_labels(out.majmin_chord_labels);
    yue2_sheet_detail::build_full_chord_labels(out.full_chord_labels);
    for (int i = 0; i < SheetSage2Tokens::N_FULL_CHORDS; ++i) {
        out.full_chord_to_id[out.full_chord_labels[i]] = i;
    }
    yue2_sheet_detail::build_duration_templates(out.duration_templates);

    static const char * event_field_order[6] = {
        "timestamp", "rhythm", "structure", "key", "chord", "melody",
    };
    for (int i = 0; i < 6; ++i) out.event_field_order[i] = event_field_order[i];

    std::string json = yue2_sheet_detail::canonical_fingerprint_json(out);
    std::string digest = yue2_sha256_hex(json);
    out.fingerprint = digest.substr(0, 16);

    if (out.fingerprint != yue2_sheet_expected_fingerprint()) {
        if (err) {
            *err = "sheetsage2 vocabulary fingerprint mismatch: expected " +
                   std::string(yue2_sheet_expected_fingerprint()) + ", got " + out.fingerprint;
        }
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// §D. token_type() classification (SheetSage2Tokenizer.token_type,
//     tokenization_sheetsage2.py:665-694) and the output-field map
//     (_token_output_field, :464-479). Pure functions of the fixed schema-v1
//     ranges above — no table instance needed.
// ─────────────────────────────────────────────────────────────────────────

enum class Yue2SheetTokenType : int8_t {
    Pad, Sos, Eos, Out,
    Prompt,
    SubbeatShift, Time, Meter, EighthPosition, Structure, Key,
    ChordMajmin, ChordFull, Pitch, Duration,
    Invalid,  // ids [12,260) (dead prompt capacity) or >= N_TOKENS: never legitimately produced
};

inline Yue2SheetTokenType yue2_sheet_token_type(int32_t token) {
    using T = SheetSage2Tokens;
    // Order follows tokenization_sheetsage2.py:665-694; since every range
    // below is disjoint, the order has no effect on the result, only on
    // fidelity to the reference scan.
    if (token == T::PAD) return Yue2SheetTokenType::Pad;
    if (token == T::SOS) return Yue2SheetTokenType::Sos;
    if (token == T::EOS) return Yue2SheetTokenType::Eos;
    if (token == T::OUT) return Yue2SheetTokenType::Out;
    if (token >= T::PROMPT_TOKEN_START && token < T::PROMPT_TASKS_END) return Yue2SheetTokenType::Prompt;
    if (token >= T::SUBBEAT_SHIFT_TOKEN_START && token < T::SUBBEAT_SHIFT_TOKEN_END) return Yue2SheetTokenType::SubbeatShift;
    if (token >= T::TIME_TOKEN_START && token < T::TIME_TOKEN_END) return Yue2SheetTokenType::Time;
    if (token >= T::METER_TOKEN_START && token < T::METER_TOKEN_END) return Yue2SheetTokenType::Meter;
    if (token >= T::EIGHTH_POSITION_TOKEN_START && token < T::EIGHTH_POSITION_TOKEN_END) return Yue2SheetTokenType::EighthPosition;
    if (token >= T::STRUCTURE_TOKEN_START && token < T::STRUCTURE_TOKEN_END) return Yue2SheetTokenType::Structure;
    if (token >= T::KEY_TOKEN_START && token < T::KEY_TOKEN_END) return Yue2SheetTokenType::Key;
    if (token >= T::MAJMIN_CHORD_TOKEN_START && token < T::MAJMIN_CHORD_TOKEN_END) return Yue2SheetTokenType::ChordMajmin;
    if (token >= T::FULL_CHORD_TOKEN_START && token < T::FULL_CHORD_TOKEN_END) return Yue2SheetTokenType::ChordFull;
    if (token >= T::PITCH_TOKEN_START && token < T::PITCH_TOKEN_END) return Yue2SheetTokenType::Pitch;
    if (token >= T::DURATION_TOKEN_START && token < T::DURATION_TOKEN_END) return Yue2SheetTokenType::Duration;
    return Yue2SheetTokenType::Invalid;  // includes the dead [12,260) gap — doc 21 §1.2
}

// FIELD_TO_INDEX order (generation_sheetsage2.py:9) — also used directly by
// sheetsage-grammar.h.
enum class Yue2SheetField : int8_t {
    Timestamp = 0, Rhythm = 1, Structure = 2, Key = 3, Chord = 4, Melody = 5,
    None = -1,
};

// _token_output_field's built-in map (tokenization_sheetsage2.py:464-479).
// v1 has no appended blocks, so there is no fallback branch to port.
inline Yue2SheetField yue2_sheet_token_output_field(Yue2SheetTokenType t) {
    switch (t) {
        case Yue2SheetTokenType::Time:            return Yue2SheetField::Timestamp;
        case Yue2SheetTokenType::Meter:            return Yue2SheetField::Rhythm;
        case Yue2SheetTokenType::EighthPosition:   return Yue2SheetField::Rhythm;
        case Yue2SheetTokenType::Structure:        return Yue2SheetField::Structure;
        case Yue2SheetTokenType::Key:               return Yue2SheetField::Key;
        case Yue2SheetTokenType::ChordMajmin:       return Yue2SheetField::Chord;
        case Yue2SheetTokenType::ChordFull:         return Yue2SheetField::Chord;
        case Yue2SheetTokenType::Pitch:             return Yue2SheetField::Melody;
        case Yue2SheetTokenType::Duration:          return Yue2SheetField::Melody;
        default:                                    return Yue2SheetField::None;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// §E. id<->label helpers for the other lanes (decode/notation/pipeline).
//     Not consumed by the grammar itself (which only needs token_type() and
//     the block boundaries), but built here since this is the one file that
//     owns the tables (doc 19's file map).
// ─────────────────────────────────────────────────────────────────────────

inline const std::string & yue2_sheet_structure_label(const SheetSage2Tokens & t, int32_t id) {
    return t.structure_labels[id];
}

inline const std::string & yue2_sheet_majmin_chord_label(const SheetSage2Tokens & t, int32_t id) {
    return t.majmin_chord_labels[id];
}

inline const std::string & yue2_sheet_full_chord_label(const SheetSage2Tokens & t, int32_t id) {
    return t.full_chord_labels[id];
}

// Returns -1 if the label is not in the table (tokenization_sheetsage2.py's
// full_chord_to_id.get() returns None in that case; callers here check -1
// the same way Python callers check `is None`).
inline int32_t yue2_sheet_full_chord_id(const SheetSage2Tokens & t, const std::string & label) {
    auto it = t.full_chord_to_id.find(label);
    return it == t.full_chord_to_id.end() ? -1 : it->second;
}

// key: key_id = mode_id*12 + tonic_id (tokenization_sheetsage2.py:328-336,
// :495-498). Returns "{tonic}:{major|minor}".
inline std::string yue2_sheet_key_label(int32_t key_id) {
    const bool minor = key_id >= 12;
    const int32_t tonic_id = key_id % 12;
    return yue2_sheet_detail::chromatic_sharps()[tonic_id] + ":" + (minor ? "minor" : "major");
}

// pitch: pitch_id -> (pitch 0..127, track 0/1) (tokenization_sheetsage2.py:523-524).
inline std::pair<int32_t, int32_t> yue2_sheet_pitch_track(int32_t pitch_id) {
    return {pitch_id % 128, int32_t(pitch_id >= 128)};
}

inline int32_t yue2_sheet_duration_steps(const SheetSage2Tokens & t, int32_t bin) {
    return t.duration_templates[bin];
}

inline const std::pair<int32_t, int32_t> & yue2_sheet_meter_pair(const SheetSage2Tokens & t, int32_t id) {
    return t.meter_pairs[id];
}

// Returns -1 if unsupported (meter_to_id lookup miss, tokenization_sheetsage2.py:308-312).
inline int32_t yue2_sheet_meter_id(const SheetSage2Tokens & t, int32_t numerator, int32_t denominator) {
    int64_t key = (int64_t(numerator) << 32) | uint32_t(denominator);
    auto it = t.meter_pair_to_id.find(key);
    return it == t.meter_pair_to_id.end() ? -1 : it->second;
}

inline double yue2_sheet_time_seconds(int32_t time_id) {
    return double(time_id) / double(SheetSage2Tokens::TIME_HZ);
}

// subbeat_shift_to_tokens (tokenization_sheetsage2.py:279-288): the loop
// condition is `shift > max_subbeat_shift` (strict), so a shift of exactly
// 256 emits ONE token at the max value, not a chained max+zero pair.
inline std::vector<int32_t> yue2_sheet_subbeat_shift_to_tokens(int32_t shift) {
    std::vector<int32_t> tokens;
    while (shift > SheetSage2Tokens::MAX_SUBBEAT_SHIFT) {
        tokens.push_back(SheetSage2Tokens::SUBBEAT_SHIFT_TOKEN_START + SheetSage2Tokens::MAX_SUBBEAT_SHIFT);
        shift -= SheetSage2Tokens::MAX_SUBBEAT_SHIFT;
    }
    tokens.push_back(SheetSage2Tokens::SUBBEAT_SHIFT_TOKEN_START + shift);
    return tokens;
}

inline int32_t yue2_sheet_token_to_subbeat_shift(int32_t token) {
    return token - SheetSage2Tokens::SUBBEAT_SHIFT_TOKEN_START;
}
