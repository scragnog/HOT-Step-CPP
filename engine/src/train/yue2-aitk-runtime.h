#pragma once

// Public, CUDA-independent seam for the YuE2 joint trainer. The parser and
// status contract are shared by ace-train and the CUDA runner; model loading
// and the training loop remain in yue2-aitk-runtime.cpp.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <unordered_set>
#include <cstdlib>
#include <cmath>

namespace yue2_aitk_runtime {

struct Config {
    std::string checkpoint;
    std::string dataset;
    std::string output;
    std::string resume;
    std::int32_t steps = 0;
    std::int32_t save_every = 250;
    std::int32_t pause_at = 0; // absolute step; 0 disables preview pauses
    float cursor_weight = 0.08f;
    bool cursor_weight_explicit = false;
    std::uint64_t seed = 0;
    std::int32_t cuda_index = 0;
    bool jsonl = true;
    std::int32_t rank = 32;
    float alpha = 32.0f;
    // adamw keeps the native CUDA AdamW8bit optimizer; prodigy and muon run
    // through the shared ggml-graph LmOptim (same optimizer the Legacy
    // trainers use) with the trainer's external clip as the only clipper.
    std::string optimizer = "adamw";
    float lr = 1e-4f;
    std::int32_t warmup = 0;
    float weight_decay = 1e-4f;
    float prodigy_d0 = 1e-6f;
    float muon_lr_scale = 1.0f;
    std::int32_t muon_ns_steps = 5;
    // 0 disables the target-loss stop; with it set, a windowed composite loss
    // at or below the target finishes the run early (steps remain the cap).
    float target_loss = 0.0f;
    std::int32_t target_loss_window = 20;
    // Planner (AR) objective: weight of the KL term that anchors the adapted
    // planner to the frozen base (AITK's ar_kl_weight), and the probability
    // that a cot=full example is trained WITHOUT its lead sheet so the same
    // adapter serves cot=off prompts (AITK's abc_dropout). Defaults match what
    // the runtime hardcoded before they were flags.
    float kl_weight = 0.2f;
    float abc_dropout = 0.5f;
    // AdamW only: the planner's learning rate as a multiple of --lr. The
    // decoder (NAR) half always trains at --lr itself. 1.0 = one rate for
    // both halves, which is what every run before this flag did.
    float planner_lr_scale = 1.0f;
};

enum class ParseResult { ok, help, error };

inline void usage(FILE * out) {
    std::fprintf(out,
        "ace-train yue2-joint-train --checkpoint <ConvRot.safetensors> "
        "--dataset <schema1-manifest.json> --output <new-run-dir> "
        "--steps N --save-every N --seed N --device CUDA0 [--resume <record>] [--pause-at N] "
        "[--cursor-weight 0.08 (0 disables lyric timing)] [--rank N] [--alpha F] "
        "[--optimizer adamw|prodigy|muon] [--lr F] [--warmup N] [--weight-decay F] "
        "[--prodigy-d0 F] [--muon-lr-scale F] [--muon-ns-steps N] "
        "[--target-loss F (0 disables)] [--target-loss-window N] "
        "[--kl-weight 0.2] [--abc-dropout 0.5] [--planner-lr-scale 1.0 (adamw only)]\n");
}

namespace detail {
inline bool decimal_u64(const char * text, std::uint64_t * out) {
    if (!text || !*text || !out) return false;
    std::uint64_t value = 0;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const std::uint64_t digit = *p - '0';
        if (value > ((std::numeric_limits<std::uint64_t>::max)() - digit) / 10u) return false;
        value = value * 10u + digit;
    }
    *out = value;
    return true;
}

inline bool decimal_i32(const char * text, std::int32_t * out) {
    std::uint64_t value = 0;
    if (!decimal_u64(text, &value) || value > 0x7fffffffULL || !out) return false;
    *out = static_cast<std::int32_t>(value);
    return true;
}

inline bool device(const char * text, std::int32_t * index) {
    if (!text || !index) return false;
    std::string value(text);
    if (value.size() < 5 || (value[0] != 'C' && value[0] != 'c') ||
        (value[1] != 'U' && value[1] != 'u') || (value[2] != 'D' && value[2] != 'd') ||
        (value[3] != 'A' && value[3] != 'a')) return false;
    const char * suffix = value.c_str() + 4;
    if (*suffix == ':') ++suffix;
    return decimal_i32(suffix, index);
}

inline bool value(const char * option, int argc, char ** argv, int * cursor,
                  std::string * out, std::string * error) {
    if (*cursor + 1 >= argc) {
        if (error) *error = std::string(option) + " needs a value";
        return false;
    }
    *out = argv[++*cursor];
    if (out->empty()) {
        if (error) *error = std::string(option) + " cannot be empty";
        return false;
    }
    return true;
}

// strtof with full-consumption and finiteness checks; the caller bounds the
// range because float parsing accepts any finite magnitude.
inline bool finite_float(const char * text, float * out) {
    if (!text || !*text || !out) return false;
    char * end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end || !std::isfinite(value)) return false;
    *out = value;
    return true;
}
} // namespace detail

inline ParseResult parse(int argc, char ** argv, Config * config, std::string * error) {
    if (!config || argc < 1) { if (error) *error = "invalid parser arguments"; return ParseResult::error; }
    Config parsed;
    std::unordered_set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--help") || !std::strcmp(arg, "-h")) return ParseResult::help;
        if (!seen.insert(arg).second) {
            if (error) *error = std::string("duplicate option: ") + arg;
            return ParseResult::error;
        }
        if (!std::strcmp(arg, "--checkpoint")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.checkpoint, error)) return ParseResult::error;
        } else if (!std::strcmp(arg, "--dataset")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.dataset, error)) return ParseResult::error;
        } else if (!std::strcmp(arg, "--output")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.output, error)) return ParseResult::error;
        } else if (!std::strcmp(arg, "--resume")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.resume, error)) return ParseResult::error;
        } else if (!std::strcmp(arg, "--steps")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.steps)) { if (error) *error = "--steps must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--save-every")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.save_every)) { if (error) *error = "--save-every must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--cursor-weight")) {
            std::string text; if(!detail::value(arg,argc,argv,&i,&text,error))return ParseResult::error;
            char * end=nullptr; const float weight=std::strtof(text.c_str(),&end);
            if(end==text.c_str() || *end || !std::isfinite(weight) || weight<0 || weight>10) {
                if(error)*error="--cursor-weight must be finite and within [0,10]"; return ParseResult::error;
            }
            parsed.cursor_weight=weight; parsed.cursor_weight_explicit=true;
        } else if (!std::strcmp(arg, "--pause-at")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.pause_at)) { if (error) *error = "--pause-at must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--seed")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_u64(value_text.c_str(), &parsed.seed)) { if (error) *error = "--seed must be an unsigned decimal integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--device")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::device(value_text.c_str(), &parsed.cuda_index)) { if (error) *error = "--device must be CUDA0 or CUDA:0"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--rank")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.rank)) { if (error) *error = "--rank must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--alpha")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.alpha)) { if (error) *error = "--alpha must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--optimizer")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.optimizer, error)) return ParseResult::error;
            if (parsed.optimizer != "adamw" && parsed.optimizer != "prodigy" && parsed.optimizer != "muon") {
                if (error) *error = "--optimizer must be adamw, prodigy or muon";
                return ParseResult::error;
            }
        } else if (!std::strcmp(arg, "--lr")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.lr)) { if (error) *error = "--lr must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--warmup")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.warmup)) { if (error) *error = "--warmup must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--weight-decay")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.weight_decay)) { if (error) *error = "--weight-decay must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--prodigy-d0")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.prodigy_d0)) { if (error) *error = "--prodigy-d0 must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--muon-lr-scale")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.muon_lr_scale)) { if (error) *error = "--muon-lr-scale must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--muon-ns-steps")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.muon_ns_steps)) { if (error) *error = "--muon-ns-steps must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--target-loss")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.target_loss)) { if (error) *error = "--target-loss must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--target-loss-window")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.target_loss_window)) { if (error) *error = "--target-loss-window must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--kl-weight")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.kl_weight)) { if (error) *error = "--kl-weight must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--abc-dropout")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.abc_dropout)) { if (error) *error = "--abc-dropout must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--planner-lr-scale")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.planner_lr_scale)) { if (error) *error = "--planner-lr-scale must be a finite number"; return ParseResult::error; }
        } else {
            if (error) *error = std::string("unknown option: ") + arg;
            return ParseResult::error;
        }
    }
    if (parsed.checkpoint.empty() || parsed.dataset.empty() || parsed.output.empty() ||
        parsed.steps <= 0 || parsed.save_every <= 0 || parsed.pause_at > parsed.steps) {
        if (error) *error = "--checkpoint, --dataset, --output, --steps > 0, and --save-every > 0 are required";
        return ParseResult::error;
    }
    *config = std::move(parsed);
    return ParseResult::ok;
}

// SIGINT is observed only at step boundaries; it never interrupts a CUDA
// graph or checkpoint write.
void yue2_aitk_install_sigint_handler();
bool yue2_aitk_cancel_requested();
void yue2_aitk_clear_cancel();

// Returns 0 on completion, 130 at a safe cancellation boundary, and 1 on a
// runtime/data/model error. The implementation emits JSONL before CUDA
// allocation and sets NVIDIA_TF32_OVERRIDE=0 before touching CUDA.
int yue2_aitk_run(const Config & config, std::string * error);

} // namespace yue2_aitk_runtime
