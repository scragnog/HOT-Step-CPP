#pragma once
// Dettato 2.0 drum inference, ported from KobiMusic's Apache-2.0 implementation.

#include "drum-features.h"
#include "drum-ops.h"
#include "yyjson.h"

#include <array>
#include <functional>
#include <filesystem>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

namespace dettato::drums {

inline constexpr int fps = 50;
inline constexpr int first_key = 27, drum_keys = 61;
inline constexpr int first_timpani_key = 41, timpani_keys = 13;

enum class Branch { kit, orchestral };
inline const char * branch_name(Branch b) { return b == Branch::kit ? "drum-kit" : "orchestral"; }

struct Hit {
    int frame;
    int key;
    float velocity;

    double onset() const { return double(frame) / fps; }
    double offset() const { return (double(frame) + 5) / fps; }
    int midi_velocity() const {
        const double quantized = std::round(std::clamp(double(velocity), 0.05, 1.0) * 1000.0) / 1000.0;
        return int(std::round(quantized * 127.0));
    }
};

struct Probabilities {
    int frames = 0;
    double router = 0;
    std::vector<float> kit, kit_vel, drum, timpani, vel;
};

struct Result {
    bool skipped = false;  // Source rate below 40 kHz.
    std::optional<double> router;  // Empty for skipped or too-short audio.
    Branch branch = Branch::kit;
    std::vector<Hit> hits;
};

namespace detail {

inline constexpr int width = 88, band_hidden = 22, tower_hidden = 18;
inline constexpr const char * stem_spec = "c16,r,r,p22,c32,r,r,r,p12,c56,r,r,r,p12,r,r,r";
inline const std::array<std::vector<int>, 5> families{{
    {35, 36}, {37, 38, 39, 40}, {42, 44, 46, 51, 53, 59},
    {41, 43, 45, 47, 48, 50, 49, 52, 55, 57}, {54, 69, 70, 82}
}};

struct Thresholds {
    float kit, orchestral, timpani, router;
};

inline Thresholds load_config(const std::filesystem::path & path) {
    std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
        yyjson_read_file(path.string().c_str(), 0, nullptr, nullptr), yyjson_doc_free);
    if (!doc) throw std::runtime_error("cannot read drum config");
    auto * root = yyjson_doc_get_root(doc.get());
    auto * kit = yyjson_obj_get(root, "drum_kit");
    auto * orch = yyjson_obj_get(root, "orchestral");
    auto * adapter = yyjson_obj_get(orch, "adapter");
    auto * blocks = yyjson_obj_get(kit, "temporal");
    auto * tb = yyjson_arr_get(blocks, 0);
    auto * gru = yyjson_arr_get(blocks, 1);
    auto * attn = yyjson_arr_get(blocks, 2);
    auto eq = [](yyjson_val * obj, const char * key, int want) {
        auto * v = yyjson_obj_get(obj, key);
        return yyjson_is_int(v) && yyjson_get_int(v) == want;
    };
    auto str = [](yyjson_val * obj, const char * key, const char * want) {
        return yyjson_equals_str(yyjson_obj_get(obj, key), want);
    };
    auto keys = [](yyjson_val * arr, int n, int first) {
        if (yyjson_arr_size(arr) != size_t(n)) return false;
        for (int i = 0; i < n; ++i) if (yyjson_get_int(yyjson_arr_get(arr, i)) != first + i) return false;
        return true;
    };
    if (!str(root, "model_type", "dettato-drums") || !eq(root, "format_version", 1) ||
        !eq(root, "sample_rate", sample_rate) || !eq(root, "min_source_rate", min_source_rate) ||
        !eq(root, "feature_fps", 100) || !eq(root, "output_fps", fps) ||
        !keys(yyjson_obj_get(root, "keys"), drum_keys, first_key) ||
        !keys(yyjson_obj_get(root, "timpani_keys"), timpani_keys, first_timpani_key) ||
        !str(kit, "stem", stem_spec) || !str(kit, "collapse", "fgru") || !str(kit, "act", "silu") ||
        !eq(kit, "w", width) || !eq(kit, "collapse_m", band_hidden) || !eq(kit, "tower_h", tower_hidden) ||
        yyjson_arr_size(blocks) != 3 || !str(tb, "t", "tb") || !eq(tb, "n", 1) ||
        (yyjson_obj_get(tb, "k") && !eq(tb, "k", 15)) || (yyjson_obj_get(tb, "d") && !eq(tb, "d", 1)) ||
        !str(gru, "t", "gru") || !eq(gru, "h", 32) || !str(attn, "t", "attn") ||
        !eq(attn, "da", 32) || !eq(attn, "heads", 2) || !eq(orch, "n_models", 3) || !eq(adapter, "d", 192) ||
        !eq(adapter, "h", 128) || !eq(adapter, "layers", 2) || !eq(adapter, "heads", 4) || !eq(adapter, "cq_d", 128)) {
        throw std::runtime_error("unsupported Dettato drum architecture");
    }
    auto * towers = yyjson_obj_get(kit, "towers");
    if (yyjson_arr_size(towers) != families.size()) throw std::runtime_error("invalid drum towers");
    for (size_t i = 0; i < families.size(); ++i) {
        const auto & family = families[i];
        auto * row = yyjson_arr_get(towers, i);
        if (yyjson_arr_size(row) != family.size()) throw std::runtime_error("invalid drum tower keys");
        for (size_t j = 0; j < family.size(); ++j) {
            if (yyjson_get_int(yyjson_arr_get(row, j)) != family[j]) throw std::runtime_error("invalid drum tower key");
        }
    }
    auto * thresholds = yyjson_obj_get(root, "thresholds");
    auto read = [&](const char * key) {
        auto * v = yyjson_obj_get(thresholds, key);
        const double t = yyjson_get_num(v);
        if (!yyjson_is_num(v) || !(t >= 0 && t <= 1)) throw std::runtime_error("invalid drum threshold");
        return float(t);
    };
    return {read("drum_kit"), read("orchestral_drum"), read("timpani"), read("router")};
}

struct ResidualConv {
    Conv dw, pw;

    ResidualConv(const WeightFile & file, const std::string & name, int c) :
        dw(file, name + ".dw", c, c, 3, 3, 1, true), pw(file, name + ".pw1", c, c, 1, 1) {
        dw.fold_norm(file, name + ".bn1");
        pw.fold_norm(file, name + ".bn2");
    }

    void forward(const Cpu & cpu, Image & x) const {
        auto y = pw(cpu, dw(cpu, x, true));
        add(x.data, y.data);
    }
};

struct StemStage {
    Conv conv;
    std::vector<ResidualConv> before_pool, after_pool;
    int pool_time;

    StemStage(const WeightFile & f, const std::string & stem, int index, int in, int out, int n, int time,
              int tail = 0) :
        conv(f, stem + std::to_string(index), in, out, 3, 3), pool_time(time) {
        conv.fold_norm(f, stem + std::to_string(index + 1));
        for (int i = 0; i < n; ++i) before_pool.emplace_back(f, stem + std::to_string(index + 3 + i), out);
        for (int i = 0; i < tail; ++i) after_pool.emplace_back(f, stem + std::to_string(index + 4 + n + i), out);
    }

    Image forward(const Cpu & cpu, const Image & x) const {
        auto y = conv(cpu, x, true);
        for (const auto & layer : before_pool) layer.forward(cpu, y);
        y = pool(y, pool_time, 2);
        for (const auto & layer : after_pool) layer.forward(cpu, y);
        return y;
    }
};

struct Tower {
    BiGru gru;
    Linear head;

    Tower(const WeightFile & f, const std::string & name, int index) :
        gru(f, name + ".towers." + std::to_string(index), width, tower_hidden),
        head(f, name + ".heads." + std::to_string(index), width + 2 * tower_hidden, int(families[index].size())) {}
};

inline void activate(Data & x) {
    for (float & v : x) v = silu(v);
}

struct KitOutput {
    Data states, onset, velocity;
};

struct DrumNet {
    StemStage s1, s2, s3;
    BiGru band, temporal;
    Linear collapse, t_pw, t_out, qkv, proj, on, vel1, vel2;
    Data t_dw, t_bias;
    Norm t_norm, song_norm;
    std::vector<Tower> towers;

    DrumNet(const WeightFile & f, const std::string & name) :
        s1(f, name + ".stem.net.", 0, 1, 16, 2, 2),
        s2(f, name + ".stem.net.", 6, 16, 32, 3, 1),
        s3(f, name + ".stem.net.", 13, 32, 56, 3, 1, 3),
        band(f, name + ".collapse.rnn", 56, band_hidden),
        temporal(f, name + ".temporal.1.rnn", width, 32),
        collapse(f, name + ".collapse.proj", 4 * band_hidden, width),
        t_pw(f, name + ".temporal.0.pw", width, width, {width, width, 1}),
        t_out(f, name + ".temporal.1.out", 64, width),
        qkv(f, name + ".temporal.2.qkv", width, 96), proj(f, name + ".temporal.2.o", 32, width),
        on(f, name + ".on.rest", width, drum_keys),
        vel1(f, name + ".vel.0", width, 16), vel2(f, name + ".vel.1", 16, drum_keys),
        t_dw(f.load(name + ".temporal.0.dw.weight", {width, 1, 15})),
        t_bias(f.load(name + ".temporal.0.dw.bias", {width})),
        t_norm(f, name + ".temporal.1.norm", width), song_norm(f, name + ".temporal.2.n", width) {
        const std::string bn = name + ".temporal.0.bn";
        const auto gain = f.load(bn + ".weight", {width});
        const auto b = f.load(bn + ".bias", {width});
        const auto mu = f.load(bn + ".running_mean", {width});
        const auto var = f.load(bn + ".running_var", {width});
        for (int c = 0; c < width; ++c) {
            const float scale = gain[c] / std::sqrt(var[c] + 1e-5f);
            t_bias[c] = (t_bias[c] - mu[c]) * scale + b[c];
            for (int j = 0; j < 15; ++j) t_dw[c * 15 + j] *= scale;
        }
        for (int i = 0; i < int(families.size()); ++i) towers.emplace_back(f, name + ".on", i);
    }

    Data embed(const Cpu & cpu, const float * data, int frames) const {
        Image x(1, frames, mel_bands);
        for (int t = 0; t < frames; ++t) {
            for (int i = 0; i < mel_bands; ++i) x.row(0, t)[i] = (data[size_t(t) * feature_bands + i] + 5.0f) / 3.0f;
        }
        auto y = s3.forward(cpu, s2.forward(cpu, s1.forward(cpu, x)));
        Data seq(size_t(y.t) * y.f * y.c);
        for (int t = 0; t < y.t; ++t) {
            for (int c = 0; c < y.c; ++c) {
                for (int f = 0; f < y.f; ++f) seq[(size_t(t) * y.f + f) * y.c + c] = y.row(c, t)[f];
            }
        }
        const auto scan = band(cpu, seq, y.f, y.t);
        constexpr int h = band_hidden;
        Data summary(size_t(y.t) * 4 * h);
        for (int t = 0; t < y.t; ++t) {
            const float * begin = scan.data() + size_t(t) * y.f * 2 * h;
            const float * end = begin + (y.f - 1) * 2 * h;
            float * row = summary.data() + size_t(t) * 4 * h;
            std::copy_n(end, h, row);
            std::copy_n(begin + h, h, row + h);
            for (int i = 0; i < 2 * h; ++i) {
                float sum = 0;
                for (int f = 0; f < y.f; ++f) sum += begin[f * 2 * h + i];
                row[2 * h + i] = sum / float(y.f);
            }
        }
        return collapse(cpu, summary.data(), y.t);
    }

    KitOutput forward(const Cpu & cpu, const Data & features, int frames) const {
        // Even tile boundaries keep the 2x pool on the song grid. The margins
        // exceed the stem's temporal receptive field. Tiles are independent, so
        // they run side by side, each on a one-thread backend: the stem's ops
        // are too small for ggml's own threads to pay.
        constexpr int chunk = 512, margin = 64;
        const int n = frames / 2;
        Data h(size_t(n) * width);
        cpu.parallel((frames + chunk - 1) / chunk, [&](int first, int last) {
            const Cpu one(1);
            for (int i = first; i < last; ++i) {
                const int s = i * chunk;
                const int a = std::max(0, s - margin);
                const int end = std::min(s + chunk, frames);
                const int b = std::min(end + margin, frames);
                const auto x = embed(one, features.data() + size_t(a) * feature_bands, b - a);
                std::copy_n(x.data() + size_t((s - a) / 2) * width, size_t((end - s) / 2) * width,
                            h.data() + size_t(s / 2) * width);
            }
        });
        Data conv(h.size());
        cpu.parallel(n, [&](int first, int end) {
            for (int t = first; t < end; ++t) {
                float * dst = conv.data() + size_t(t) * width;
                for (int c = 0; c < width; ++c) {
                    float sum = t_bias[c];
                    for (int k = std::max(0, 7 - t); k < std::min(15, n + 7 - t); ++k) {
                        sum += t_dw[c * 15 + k] * h[size_t(t + k - 7) * width + c];
                    }
                    dst[c] = silu(sum);
                }
            }
        });
        add(h, t_pw(cpu, conv.data(), n));
        Data norm = h;
        t_norm.layer(norm, n);
        const auto scan = temporal(cpu, norm, n);
        add(h, t_out(cpu, scan.data(), n));
        norm = h;
        song_norm.layer(norm, n);
        add(h, attention(cpu, norm, n, qkv, proj, 2));

        Data onset = on(cpu, h.data(), n);
        constexpr int tw = 2 * tower_hidden;
        Data tower_input(size_t(n) * (tw + width));
        for (size_t i = 0; i < towers.size(); ++i) {
            const auto hidden = towers[i].gru(cpu, h, n);
            for (int t = 0; t < n; ++t) {
                float * row = tower_input.data() + size_t(t) * (tw + width);
                std::copy_n(hidden.data() + size_t(t) * tw, tw, row);
                std::copy_n(h.data() + size_t(t) * width, width, row + tw);
            }
            const auto logits = towers[i].head(cpu, tower_input.data(), n);
            const auto & keys = families[i];
            for (int t = 0; t < n; ++t) {
                for (size_t k = 0; k < keys.size(); ++k) {
                    onset[size_t(t) * drum_keys + keys[k] - first_key] = logits[size_t(t) * keys.size() + k];
                }
            }
        }
        const auto v = vel1(cpu, h.data(), n);
        return {std::move(h), std::move(onset), vel2(cpu, v.data(), n)};
    }
};

struct CqBranch {
    static constexpr int channels = 32, dim = 128;
    Conv c1, c2, c3;
    Norm n1, n2, n3;
    Linear proj;

    CqBranch(const WeightFile & f, const std::string & name) :
        c1(f, name + ".c1", 3, 24, 5, 7), c2(f, name + ".c2", 24, channels, 3, 5, 2),
        c3(f, name + ".c3", channels, channels, 3, 5), n1(f, name + ".n1", 24),
        n2(f, name + ".n2", channels), n3(f, name + ".n3", channels),
        proj(f, name + ".proj", channels * cq_bands, dim) {}

    Data forward(const Cpu & cpu, const Data & feats, int frames) const {
        double sum = 0;
        float minimum = std::numeric_limits<float>::infinity();
        for (int t = 0; t < frames; ++t) {
            for (int i = 0; i < cq_bands; ++i) {
                const float v = feats[size_t(t) * feature_bands + mel_bands + i];
                sum += v;
                minimum = std::min(minimum, v);
            }
        }
        const float mean = float(sum / (double(frames) * cq_bands));
        Image x(3, frames, cq_bands);
        constexpr int shifts[] = {0, 14, 24};
        for (int c = 0; c < 3; ++c) {
            for (int t = 0; t < frames; ++t) {
                const auto * row = feats.data() + size_t(t) * feature_bands + mel_bands;
                for (int i = 0; i < cq_bands; ++i) {
                    const int f = i + shifts[c];
                    x.row(c, t)[i] = ((f < cq_bands ? row[f] : minimum) - mean) / 4.0f;
                }
            }
        }
        auto h = c1(cpu, x);
        x = Image(0, 0, 0);
        n1.group_silu(cpu, h, 4);
        auto y = c2(cpu, h);
        h = Image(0, 0, 0);
        n2.group_silu(cpu, y, 4);
        auto z = c3(cpu, y);
        n3.group_silu(cpu, z, 4);
        add(z.data, y.data);
        y = Image(0, 0, 0);
        constexpr int block = 256;
        Data rows(size_t(block) * channels * cq_bands), out(size_t(z.t) * dim);
        for (int s = 0; s < z.t; s += block) {
            const int count = std::min(z.t - s, block);
            for (int t = 0; t < count; ++t) {
                for (int c = 0; c < channels; ++c) {
                    std::copy_n(z.row(c, s + t), cq_bands, rows.data() + (size_t(t) * channels + c) * cq_bands);
                }
            }
            auto values = proj(cpu, rows.data(), count);
            activate(values);
            std::copy(values.begin(), values.end(), out.data() + size_t(s) * dim);
        }
        return out;
    }
};

struct Expert {
    static constexpr int state = 256, nin = width + CqBranch::dim + drum_keys;
    DrumNet base;
    CqBranch cq;
    Linear inp, qkv, ao, delta, timp, gate1, gate2;
    BiGru rnn1, rnn2;
    Norm norm;

    Expert(const WeightFile & f, const std::string & name) :
        base(f, name + ".base"), cq(f, name + ".cq"),
        inp(f, name + ".inp", nin, 192), qkv(f, name + ".qkv", state, 3 * state),
        ao(f, name + ".ao", state, state), delta(f, name + ".delta", state, drum_keys),
        timp(f, name + ".timp", state, timpani_keys), gate1(f, name + ".gate.0", state, 64),
        gate2(f, name + ".gate.2", 64, 1), rnn1(f, name + ".rnn", 192, 128),
        rnn2(f, name + ".rnn", state, 128, 1), norm(f, name + ".ln", state) {}

    float forward(const Cpu & cpu, const Data & feats, int frames, Probabilities & sum) const {
        auto kit = base.forward(cpu, feats, frames);
        auto q = cq.forward(cpu, feats, frames);
        const int n = frames / 2;
        Data rows(size_t(n) * nin);
        for (int t = 0; t < n; ++t) {
            float * row = rows.data() + size_t(t) * nin;
            std::copy_n(kit.states.data() + size_t(t) * width, width, row);
            std::copy_n(q.data() + size_t(t) * CqBranch::dim, CqBranch::dim, row + width);
            for (int k = 0; k < drum_keys; ++k) {
                row[width + CqBranch::dim + k] = cpu_gru::sigmoid(kit.onset[size_t(t) * drum_keys + k]);
            }
        }
        auto a = inp(cpu, rows.data(), n);
        activate(a);
        auto r = rnn2(cpu, rnn1(cpu, a, n), n);
        add(r, attention(cpu, r, n, qkv, ao, 4));
        norm.layer(r, n);
        std::array<float, state> pooled{};
        for (int c = 0; c < state; ++c) {
            double total = 0;
            for (int t = 0; t < n; ++t) total += r[size_t(t) * state + c];
            pooled[c] = float(total / n);
        }
        auto g = gate1(cpu, pooled.data(), 1);
        activate(g);
        const auto correction = delta(cpu, r.data(), n);
        const auto timpani = timp(cpu, r.data(), n);
        for (size_t i = 0; i < correction.size(); ++i) {
            sum.drum[i] += cpu_gru::sigmoid(kit.onset[i] + correction[i]);
            sum.vel[i] += cpu_gru::sigmoid(kit.velocity[i]);
        }
        for (size_t i = 0; i < timpani.size(); ++i) sum.timpani[i] += cpu_gru::sigmoid(timpani[i]);
        return cpu_gru::sigmoid(gate2(cpu, g.data(), 1)[0]);
    }
};

template <class Add>
inline void peaks(const Data & data, int frames, int lanes, float threshold, Add add_hit) {
    std::array<int, drum_keys> last;
    last.fill(-3);
    for (int t = 0; t < frames; ++t) {
        const float * row = data.data() + size_t(t) * lanes;
        const float * prev = t == 0 ? row : row - lanes;
        const float * next = t == frames - 1 ? row : row + lanes;
        for (int k = 0; k < lanes; ++k) {
            if (t - last[k] >= 3 && row[k] >= threshold && row[k] >= prev[k] && row[k] >= next[k]) {
                add_hit(t, k);
                last[k] = t;
            }
        }
    }
}

}  // namespace detail

// Initialize GGML's CPU backend registry before inference.
class Model {
    detail::Thresholds thresholds_;
    detail::DrumNet kit_;
    std::array<detail::Expert, 3> experts_;

    Model(detail::Thresholds thresholds, const detail::WeightFile & f) :
        thresholds_(thresholds), kit_(f, "drum_kit"),
        experts_{{{f, "orchestral.0"}, {f, "orchestral.1"}, {f, "orchestral.2"}}} {}

public:
    explicit Model(const std::filesystem::path & dir) :
        Model(detail::load_config(dir / "config.json"),
              detail::WeightFile((dir / "model.safetensors").string().c_str())) {}

    Branch choose(double router) const { return router < thresholds_.router ? Branch::kit : Branch::orchestral; }

    Probabilities probabilities(const std::vector<float> & feats, int threads = 4,
                                const std::function<void()> & network_done = {}) const {
        if (feats.size() % feature_bands != 0) throw std::invalid_argument("invalid drum features shape");
        Probabilities p;
        p.frames = int(feats.size() / (2 * feature_bands));
        if (p.frames == 0) return p;
        // The kit network and the three experts are independent, and one network stops scaling at about four
        // threads, so they run side by side, each on its own CPU backend with a share of the threads. No op's result
        // depends on its thread count and the experts are summed in their fixed order: the output is bit for bit a
        // serial run's.
        const size_t n = size_t(p.frames) * drum_keys;
        const int share = std::max(1, threads / int(1 + experts_.size()));
        std::array<Probabilities, std::tuple_size_v<decltype(experts_)>> part;
        std::array<float, std::tuple_size_v<decltype(experts_)>> gate{};
        detail::KitOutput kit;
        std::mutex lock;
        std::exception_ptr error;
        const auto job = [&](auto fn) {
            return [&, fn] {
                try {
                    detail::Cpu cpu(share);
                    fn(cpu);
                    std::lock_guard<std::mutex> g(lock);
                    if (network_done && !error) network_done();
                } catch (...) {
                    std::lock_guard<std::mutex> g(lock);
                    if (!error) error = std::current_exception();
                }
            };
        };
        std::vector<std::thread> pool;
        pool.emplace_back(job([&](const detail::Cpu & cpu) { kit = kit_.forward(cpu, feats, 2 * p.frames); }));
        for (size_t e = 0; e < experts_.size(); ++e) {
            pool.emplace_back(job([&, e](const detail::Cpu & cpu) {
                part[e].drum.resize(n);
                part[e].vel.resize(n);
                part[e].timpani.resize(size_t(p.frames) * timpani_keys);
                gate[e] = experts_[e].forward(cpu, feats, 2 * p.frames, part[e]);
            }));
        }
        for (auto & t : pool) t.join();
        if (error) std::rethrow_exception(error);
        p.kit.resize(kit.onset.size());
        p.kit_vel.resize(kit.velocity.size());
        std::transform(kit.onset.begin(), kit.onset.end(), p.kit.begin(), cpu_gru::sigmoid);
        std::transform(kit.velocity.begin(), kit.velocity.end(), p.kit_vel.begin(), cpu_gru::sigmoid);
        p.drum.resize(n);
        p.vel.resize(n);
        p.timpani.resize(size_t(p.frames) * timpani_keys);
        for (size_t e = 0; e < experts_.size(); ++e) {
            p.router += gate[e];
            detail::add(p.drum, part[e].drum);
            detail::add(p.vel, part[e].vel);
            detail::add(p.timpani, part[e].timpani);
        }
        p.router /= double(experts_.size());
        for (auto * tensor : {&p.drum, &p.vel, &p.timpani}) {
            for (float & v : *tensor) v /= float(experts_.size());
        }
        return p;
    }

    std::vector<Hit> decode(const Probabilities & p, Branch branch) const {
        std::vector<Hit> hits;
        const bool kit = branch == Branch::kit;
        const auto & onset = kit ? p.kit : p.drum;
        const auto & vel = kit ? p.kit_vel : p.vel;
        std::array<std::vector<int>, timpani_keys> seen;
        detail::peaks(onset, p.frames, drum_keys, kit ? thresholds_.kit : thresholds_.orchestral, [&](int t, int k) {
            const int key = first_key + k;
            hits.push_back({t, key, vel[size_t(t) * drum_keys + k]});
            if (!kit && key >= first_timpani_key && key < first_timpani_key + timpani_keys) {
                seen[key - first_timpani_key].push_back(t);
            }
        });
        if (kit) return hits;
        detail::peaks(p.timpani, p.frames, timpani_keys, thresholds_.timpani, [&](int t, int k) {
            const auto & frames = seen[k];
            auto it = std::lower_bound(frames.begin(), frames.end(), t - 2);
            for (; it != frames.end() && *it <= t + 2; ++it) {
                // Upstream compares seconds in double precision.
                if (std::abs(double(t) / fps - double(*it) / fps) <= 0.04) return;
            }
            hits.push_back({t, first_timpani_key + k, std::clamp(p.timpani[size_t(t) * timpani_keys + k], 0.3f, 1.0f)});
        });
        std::stable_sort(hits.begin(), hits.end(), [](const Hit & a, const Hit & b) { return a.frame < b.frame; });
        return hits;
    }

    Result transcribe(const float * mono, size_t n, int source_rate, int threads = 4,
                      const std::function<void(double)> & progress = {}) const {
        Result out;
        out.skipped = source_rate < min_source_rate;
        if (out.skipped) return out;
        // Count completed stages: features, kit, each orchestral expert, then decode.
        int completed = 0;
        const auto stage_done = [&] {
            if (progress) progress(++completed / double(3 + experts_.size()));
        };
        const auto f = features(mono, n, source_rate, threads);
        stage_done();
        const auto p = probabilities(f, threads, stage_done);
        if (p.frames == 0) return out;
        out.router = p.router;
        out.branch = choose(p.router);
        out.hits = decode(p, out.branch);
        stage_done();
        return out;
    }
};

}  // namespace dettato::drums
