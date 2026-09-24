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
    bool alpha_explicit = false;
    // "lora" (A/B pairs, the only value before 2026-09-22) or "lokr" (kron
    // factors, lokr-apply.h). Under lokr, --rank is unused and --alpha is the
    // LoKr alpha, defaulting to --lokr-dim (scale 1) when not given. The DiT
    // defaults (dim 512 / factor 6) are a trap at YuE2's dims: larger than a
    // rank-64 LoRA. dim 32 / factor 8 is 14 MB.
    std::string adapter_type = "lora";
    std::int32_t lokr_dim = 32;
    std::int32_t lokr_factor = 8;
    // adamw keeps the native CUDA AdamW8bit optimizer; prodigy and muon run
    // through the shared ggml-graph LmOptim (same optimizer the Legacy
    // trainers use) with the trainer's external clip as the only clipper.
    // adamw-lm is AdamW on that LmOptim path (fp32 moments, the LmOptim
    // warmup+cosine schedule): a different implementation, not a refactor,
    // and the one the update modifiers below can act on.
    std::string optimizer = "adamw";
    // Cautious update mask (lm-optim.h): needs an LmOptim optimizer.
    bool cautious = false;
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
    // 0 disables the AR-KL stop; with it set, the run finishes once the
    // windowed mean of ar_kl (planner divergence from the frozen base) reaches
    // the target. Unlike ar_ce it means the same thing for every artist:
    // likeness starts near 1.25, planner damage near 1.9. Shares the window.
    float target_kl = 0.0f;
    // How the KL stop reads the noisy per-step ar_kl. "mean": the trailing
    // mean over target_loss_window, which lags the trend by half a window.
    // "trend": a least-squares line through the last kKlTrendWindow steps,
    // read at the current step, so there is no lag (2026-09-22). The UI draws
    // the same line.
    std::string target_kl_mode = "mean";
    // Planner (AR) objective: weight of the KL term that anchors the adapted
    // planner to the frozen base (AITK's ar_kl_weight), and the probability
    // that a cot=full example is trained WITHOUT its lead sheet so the same
    // adapter serves cot=off prompts (AITK's abc_dropout). Defaults match what
    // the runtime hardcoded before they were flags.
    float kl_weight = 0.2f;
    float abc_dropout = 0.5f;
    // Probability that an example trains with its style reduced to the
    // trigger alone (the legacy AR trainer's --caption-dropout; 0.5 is the
    // measured recipe, arms 137-140 of 2026-09-15). Needs a dataset prepared
    // with the trigger-only prefixes. 0 = off, byte-identical to before.
    float caption_dropout = 0.0f;
    // AdamW only: the planner's learning rate as a multiple of --lr. The
    // decoder (NAR) half always trains at --lr itself. 1.0 = one rate for
    // both halves, which is what every run before this flag did.
    float planner_lr_scale = 1.0f;
    // The decoder (NAR) half's rate as a multiple of --lr, the mirror of
    // planner_lr_scale. LoKr under Prodigy overcooks the NAR long before the
    // AR reaches its KL target (greenday ear test, 2026-09-22: AR 200 + NAR 100
    // beat any single checkpoint). 1.0 = unchanged.
    float nar_lr_scale = 1.0f;
    // With --target-kl: instead of ending the run when the planner reaches its
    // KL, freeze the planner there and train the decoder alone for this many
    // more steps (--steps stays the cap). The KL checkpoint is still written,
    // so the old stop point is always on disk. 0 = stop at the KL, as before.
    std::int32_t nar_extra_steps = 0;
    // With --resume and --nar-extra-steps: freeze the planner at the resumed
    // step (the checkpoint's planner is the one kept), then train the decoder
    // alone. This is how a planner stop decided OUTSIDE the trainer (a plan
    // sweep on the checkpoints) is applied: resume the last good checkpoint
    // with this flag.
    bool freeze_planner_now = false;
    // Decoder drift meter: at every checkpoint, how far the decoder's flow
    // prediction has moved from the base decoder's on a fixed probe set, as a
    // relative squared error. Written to train.jsonl and the checkpoint's
    // meters.json. Off = no probe cost and byte-identical runs.
    bool nar_drift = false;
    // With --resume: restore the checkpoint, take the decoder meters (drift
    // and reconstruction), write them to the checkpoint's meters.json, exit.
    // Scores checkpoints of a finished run; --output is a scratch directory.
    bool meter_only = false;
    // Spike guard. A step whose pre-clip gradient norm exceeds spike_factor x
    // the median of the last 50 applied steps skips its update (0 = off; it
    // arms once 20 norms are in the window, and the window restarts when the
    // planner freezes, because the decoder's norm alone is ~10x smaller).
    // spike_stop skips within spike_stop_window steps end the run there: the
    // weights on disk are the last pre-spike state (0 = never stop).
    float spike_factor = 0.0f;
    std::int32_t spike_stop = 0;
    std::int32_t spike_stop_window = 20;
    // Decoder stop (needs --nar-drift, planner frozen): at a checkpoint, if
    // the reconstruction meter improved by less than recon_stop (a fraction,
    // 0.005 = half a percent) over the last recon_stop_window checkpoints,
    // the decoder is done: stop there. 0 = off. Ear-checked 2026-09-24
    // (Steel Panther 300/425/500: subtle, diminishing returns past the knee).
    float recon_stop = 0.0f;
    std::int32_t recon_stop_window = 3;
    // With --resume: start the reconstruction window empty instead of the
    // record's. A refinement pass resumes a run that may have ENDED on the
    // recon stop; its old readings would fire the stop again at once.
    bool recon_reset = false;
    // Planner refinement (with --resume): ignore the record's planner freeze
    // and train both halves on. The KL stop (--target-kl) is the ceiling.
    bool unfreeze_planner = false;
    // Save a checkpoint each time the KL stop reading crosses the next
    // multiple of this (0.1 = a rung every 0.1 KL), tagged kl_mark. The
    // rungs are what a listener compares to pick the planner's stop.
    float kl_checkpoint_every = 0.0f;
    // With --kl-checkpoint-every: end the segment after each rung checkpoint
    // (event "paused"), so the server can render that rung's previews and
    // resume, as it does for step-cadence preview pauses.
    bool pause_on_kl_mark = false;
    // Planner refinement pacing. --refine-warmup N: the rate ramps from zero
    // over N steps after the unfreeze (the unfreeze step rides in the resume
    // record). --rung-adaptive-lr: when a reading jumps more than one rung
    // since the last, the rate halves (floor 0.05x) and the multiplier rides
    // in the record; the run settles to about one rung per interval.
    std::int32_t refine_warmup = 0;
    bool rung_adaptive_lr = false;
};

enum class ParseResult { ok, help, error };

inline void usage(FILE * out) {
    std::fprintf(out,
        "ace-train yue2-joint-train --checkpoint <ConvRot.safetensors> "
        "--dataset <schema1-manifest.json> --output <new-run-dir> "
        "--steps N --save-every N --seed N --device CUDA0 [--resume <record>] [--pause-at N] "
        "[--cursor-weight 0.08 (0 disables lyric timing)] [--rank N] [--alpha F] "
        "[--adapter-type lora|lokr] [--lokr-dim 32] [--lokr-factor 8] "
        "[--optimizer adamw|adamw-lm|prodigy|muon] [--cautious] [--lr F] [--warmup N] [--weight-decay F] "
        "[--prodigy-d0 F] [--muon-lr-scale F] [--muon-ns-steps N] "
        "[--target-loss F (0 disables)] [--target-kl F (0 disables)] [--target-loss-window N] [--target-kl-mode mean|trend] "
        "[--kl-weight 0.2] [--abc-dropout 0.5] [--caption-dropout 0] [--planner-lr-scale 1.0 (not muon)] [--nar-lr-scale 1.0 (not muon)] "
        "[--nar-extra-steps N (with --target-kl: freeze the planner at its KL, train the decoder N more steps)] "
        "[--nar-drift (log the decoder's drift from base and its reconstruction error at every checkpoint)] "
        "[--meter-only (with --resume: write the checkpoint's meters.json and exit)] "
        "[--recon-stop F (planner frozen: stop when the reconstruction meter improves under F over --recon-stop-window 3 checkpoints)] [--recon-reset (with --resume: empty window)] "
        "[--unfreeze-planner (with --resume: train the planner on past its freeze)] [--kl-checkpoint-every 0.1 (a checkpoint at each KL rung)] "
        "[--freeze-planner-now (with --resume and --nar-extra-steps: freeze the planner at the resumed step)] "
        "[--spike-factor F (skip updates above F x median gradient norm; 0 = off)] [--spike-stop N (stop after N skips)] [--spike-stop-window 20]\n");
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
            parsed.alpha_explicit = true;
        } else if (!std::strcmp(arg, "--adapter-type")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.adapter_type, error)) return ParseResult::error;
            if (parsed.adapter_type != "lora" && parsed.adapter_type != "lokr") {
                if (error) *error = "--adapter-type must be lora or lokr";
                return ParseResult::error;
            }
        } else if (!std::strcmp(arg, "--lokr-dim")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.lokr_dim)) { if (error) *error = "--lokr-dim must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--lokr-factor")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.lokr_factor)) { if (error) *error = "--lokr-factor must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--optimizer")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.optimizer, error)) return ParseResult::error;
            if (parsed.optimizer != "adamw" && parsed.optimizer != "adamw-lm" && parsed.optimizer != "prodigy" && parsed.optimizer != "muon") {
                if (error) *error = "--optimizer must be adamw, adamw-lm, prodigy or muon";
                return ParseResult::error;
            }
        } else if (!std::strcmp(arg, "--cautious")) {
            parsed.cautious = true;
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
        } else if (!std::strcmp(arg, "--target-kl")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.target_kl)) { if (error) *error = "--target-kl must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--target-kl-mode")) {
            if (!detail::value(arg, argc, argv, &i, &parsed.target_kl_mode, error)) return ParseResult::error;
            if (parsed.target_kl_mode != "mean" && parsed.target_kl_mode != "trend") { if (error) *error = "--target-kl-mode must be mean or trend"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--target-loss-window")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.target_loss_window)) { if (error) *error = "--target-loss-window must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--kl-weight")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.kl_weight)) { if (error) *error = "--kl-weight must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--abc-dropout")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.abc_dropout)) { if (error) *error = "--abc-dropout must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--caption-dropout")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.caption_dropout)) { if (error) *error = "--caption-dropout must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--planner-lr-scale")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.planner_lr_scale)) { if (error) *error = "--planner-lr-scale must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--nar-lr-scale")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.nar_lr_scale)) { if (error) *error = "--nar-lr-scale must be a finite number"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--freeze-planner-now")) {
            parsed.freeze_planner_now = true;
        } else if (!std::strcmp(arg, "--recon-stop")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.recon_stop) || parsed.recon_stop < 0.0f || parsed.recon_stop >= 1.0f) { if (error) *error = "--recon-stop must be a fraction in [0, 1)"; return ParseResult::error; }
            if (parsed.recon_stop > 0.0f) parsed.nar_drift = true;
        } else if (!std::strcmp(arg, "--recon-reset")) {
            parsed.recon_reset = true;
        } else if (!std::strcmp(arg, "--unfreeze-planner")) {
            parsed.unfreeze_planner = true;
        } else if (!std::strcmp(arg, "--pause-on-kl-mark")) {
            parsed.pause_on_kl_mark = true;
        } else if (!std::strcmp(arg, "--rung-adaptive-lr")) {
            parsed.rung_adaptive_lr = true;
        } else if (!std::strcmp(arg, "--refine-warmup")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::decimal_i32(text.c_str(), &parsed.refine_warmup) || parsed.refine_warmup < 0) { if (error) *error = "--refine-warmup must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--kl-checkpoint-every")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.kl_checkpoint_every) || parsed.kl_checkpoint_every < 0.0f) { if (error) *error = "--kl-checkpoint-every must be a finite number >= 0"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--recon-stop-window")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::decimal_i32(text.c_str(), &parsed.recon_stop_window) || parsed.recon_stop_window < 1) { if (error) *error = "--recon-stop-window must be a positive integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--spike-factor")) {
            std::string text; if (!detail::value(arg, argc, argv, &i, &text, error) ||
                !detail::finite_float(text.c_str(), &parsed.spike_factor) || parsed.spike_factor < 0.0f) { if (error) *error = "--spike-factor must be a finite number >= 0"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--spike-stop")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.spike_stop)) { if (error) *error = "--spike-stop must be a nonnegative integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--spike-stop-window")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.spike_stop_window) || parsed.spike_stop_window < 1) { if (error) *error = "--spike-stop-window must be a positive integer"; return ParseResult::error; }
        } else if (!std::strcmp(arg, "--nar-drift")) {
            parsed.nar_drift = true;
        } else if (!std::strcmp(arg, "--meter-only")) {
            parsed.meter_only = true; parsed.nar_drift = true;
        } else if (!std::strcmp(arg, "--nar-extra-steps")) {
            std::string value_text; if (!detail::value(arg, argc, argv, &i, &value_text, error) ||
                !detail::decimal_i32(value_text.c_str(), &parsed.nar_extra_steps)) { if (error) *error = "--nar-extra-steps must be a nonnegative integer"; return ParseResult::error; }
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
