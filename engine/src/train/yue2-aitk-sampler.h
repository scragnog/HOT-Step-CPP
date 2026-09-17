#pragma once

// Native YuE2 sampling reference. RNG state is intentionally independent of
// Torch's generator: this stream is versioned and serializable for exact native
// resume, but it does not claim Torch bit parity.
#include "yue2-aitk-batch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iomanip>
#include <iterator>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yue2_aitk {

class NativeRng {
public:
    explicit NativeRng(uint64_t seed = 0) : engine_(seed) {}
    double uniform01() { constexpr double scale = 4503599627370496.0; return (double(engine_() >> 12) + 0.5) / scale; }
    double normal01() { if (has_spare_) { has_spare_ = false; return spare_; } double u1 = uniform01(), u2 = uniform01(); const double radius = std::sqrt(-2.0 * std::log(u1)); const double angle = 6.2831853071795864769 * u2; spare_ = radius * std::sin(angle); has_spare_ = true; return radius * std::cos(angle); }
    size_t uniform_index(size_t exclusive) { if (!exclusive || exclusive > std::numeric_limits<uint64_t>::max()) throw std::invalid_argument("invalid native RNG range"); const uint64_t bound = static_cast<uint64_t>(exclusive); const uint64_t threshold = (uint64_t(0) - bound) % bound; uint64_t draw = 0; do { draw = engine_(); } while (draw < threshold); return static_cast<size_t>(draw % bound); }
    template<class T> void shuffle(std::vector<T> & values) { for (size_t i = values.size(); i > 1; --i) std::swap(values[i - 1], values[uniform_index(i)]); }
    std::string export_state() const { std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<double>::max_digits10) << "Y2RNG1 " << (has_spare_ ? 1 : 0) << ' ' << spare_ << ' ' << engine_; return out.str(); }
    bool import_state(const std::string & state) { if (state.size() > 65536 || state.rfind("Y2RNG1 ", 0) != 0) return false; std::istringstream in(state); in.imbue(std::locale::classic()); std::string magic; int spare = 0; double spare_candidate = 0; std::mt19937_64 engine_candidate; if (!(in >> magic >> spare >> spare_candidate >> engine_candidate) || magic != "Y2RNG1" || (spare != 0 && spare != 1) || !std::isfinite(spare_candidate)) return false; in >> std::ws; if (!in.eof()) return false; engine_ = engine_candidate; has_spare_ = spare != 0; spare_ = spare_candidate; return true; }
    bool has_spare() const { return has_spare_; }
private:
    std::mt19937_64 engine_;
    bool has_spare_ = false;
    double spare_ = 0.0;
    double spare_value_ = 0.0;
};

struct SampledBatch {
    Batch batch;
    FrameRange crop;
    bool abc_retained = false;
    float timestep = 0.0f;
    float timestep_bf16 = 0.0f;
    std::vector<float> clean_f32, noise_f32, target_f32, noisy_f32, noisy_bf16;
};

inline float bf16_round_f32(float value) { uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits)); bits += 0x7fffu + ((bits >> 16) & 1u); bits &= 0xffff0000u; std::memcpy(&value, &bits, sizeof(bits)); return value; }

class Yue2NativeSampler {
public:
    explicit Yue2NativeSampler(uint64_t seed) : rng_(seed) {}
    NativeRng & rng() { return rng_; }
    std::vector<float> sigmoid_schedule(size_t count) { if (!count) throw std::invalid_argument("timestep schedule is empty"); std::vector<float> values; values.reserve(count); for (size_t i = 0; i < count; ++i) values.push_back(float((1.0 - (1.0 / (1.0 + std::exp(-rng_.normal01())))) * 1000.0)); std::sort(values.begin(), values.end(), std::greater<float>()); schedule_ = values; schedule_cursor_ = 0; return values; }
    SampledBatch sample(const SongInput & song, const PromptInput & prompt, size_t train_window_frames, const std::vector<float> & timesteps, float abc_dropout = .5f, size_t ar_token_limit = 0, size_t eligible_timestep_count = 0) {
        if (timesteps.empty() || train_window_frames > 1500 || !(abc_dropout >= 0 && abc_dropout <= 1)) throw std::invalid_argument("invalid native sampler configuration");
        const size_t eligible = eligible_timestep_count ? eligible_timestep_count : timesteps.size();
        if (!eligible || eligible > timesteps.size()) throw std::invalid_argument("invalid eligible timestep count");
        for (float timestep : timesteps) if (!std::isfinite(timestep) || timestep < 0 || timestep > 1000) throw std::invalid_argument("timestep is outside [0,1000]");
        if (schedule_.empty()) schedule_ = timesteps; else if (schedule_ != timesteps) throw std::invalid_argument("timestep schedule changed without a new epoch");
        for (float value : song.latents) if (!std::isfinite(value)) throw std::invalid_argument("song latent is nonfinite");
        const size_t frames = song.semantic_tokens.size(); FrameRange crop{0, frames}; if (train_window_frames && frames > train_window_frames) { crop.end = train_window_frames; crop.start = rng_.uniform_index(frames - train_window_frames + 1); crop.end += crop.start; }
        PromptInput selected = prompt; selected.retain_abc = prompt.retain_abc && rng_.uniform01() >= abc_dropout;
        const size_t timestep_index = rng_.uniform_index(eligible);
        ++schedule_cursor_;
        SampledBatch out; out.crop = crop; out.abc_retained = selected.retain_abc; out.timestep = timesteps[timestep_index]; out.timestep_bf16 = bf16_round_f32(out.timestep / 1000.0f);
        std::string error; if (!build(selected, song, crop, ar_token_limit, &out.batch, &error)) throw std::invalid_argument(error);
        // YuE2 cached latents and get_noise are cast to the recipe training
        // dtype before add_noise. This reference models the BF16 recipe:
        // clean/noise round at ingress, target subtracts those rounded values,
        // and add_noise promotes the BF16 operands through the F32 timestep
        // expression before the model-input BF16 cast.
        out.clean_f32.resize(out.batch.nar.latents.size()); out.noise_f32.resize(out.clean_f32.size()); out.target_f32.resize(out.clean_f32.size()); out.noisy_f32.resize(out.clean_f32.size()); out.noisy_bf16.resize(out.clean_f32.size());
        for (size_t i = 0; i < out.clean_f32.size(); ++i) { out.clean_f32[i] = bf16_round_f32(out.batch.nar.latents[i]); out.noise_f32[i] = bf16_round_f32(float(rng_.normal01())); out.target_f32[i] = bf16_round_f32(out.noise_f32[i] - out.clean_f32[i]); const float t01 = out.timestep / 1000.0f; out.noisy_f32[i] = (1.0f - t01) * out.clean_f32[i] + t01 * out.noise_f32[i]; out.noisy_bf16[i] = bf16_round_f32(out.noisy_f32[i]); }
        return out;
    }
    // Toolkit references: sigmoid draw/sort is custom_flowmatch_sampler.py:121-131;
    // training selects indices from the scheduler in BaseSDTrainProcess.py:1289-1320;
    // YuE2 crop uses random.randint in yue2_model.py:336-341 and ABC dropout at :540.
    // The cursor counts draws from the current schedule.  Timesteps are sampled
    // with replacement, so it is deliberately not bounded by schedule_.size().
    // A new schedule is an explicit caller operation (sigmoid_schedule), not an
    // implicit epoch boundary.
    std::string export_rng_state() const { if (schedule_.empty() || schedule_.size() > kMaxScheduleCount) return {}; std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(std::numeric_limits<double>::max_digits10) << "Y2SAMPLER1 " << schedule_cursor_ << ' ' << schedule_.size(); for (float value : schedule_) out << ' ' << value; const std::string rng = rng_.export_state(); out << ' ' << rng.size() << ' ' << rng; const std::string result = out.str(); return result.size() <= kMaxStateBytes ? result : std::string(); }
    bool import_rng_state(const std::string & state) { if (state.empty() || state.size() > kMaxStateBytes) return false; std::istringstream in(state); in.imbue(std::locale::classic()); std::string magic; uint64_t cursor = 0, count = 0, rng_size = 0; if (!(in >> magic >> cursor >> count) || magic != "Y2SAMPLER1" || count == 0 || count > kMaxScheduleCount) return false; if (count > (std::numeric_limits<size_t>::max)() / sizeof(float)) return false; std::vector<float> schedule(static_cast<size_t>(count)); for (float & value : schedule) if (!(in >> value) || !std::isfinite(value) || value < 0 || value > 1000) return false; if (!(in >> rng_size) || rng_size > kMaxStateBytes) return false; in >> std::ws; std::string rng((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); if (rng.size() != rng_size) return false; NativeRng candidate = rng_; if (!candidate.import_state(rng)) return false; schedule_ = std::move(schedule); schedule_cursor_ = cursor; rng_ = std::move(candidate); return true; }
private:
    static constexpr uint64_t kMaxScheduleCount = 8192;
    static constexpr size_t kMaxStateBytes = 65536;
    NativeRng rng_;
    std::vector<float> schedule_;
    uint64_t schedule_cursor_ = 0;
};

} // namespace yue2_aitk
