// Validate the native Dettato drum stage against a slice 1 CPU capture.
// Part of HOT-Step CPP. MIT license.

#include "dettato/drums.h"
#include "wav.h"

#include <fstream>
#include <map>

namespace {

namespace drums = dettato::drums;
namespace fs = std::filesystem;
using Floats = std::vector<float>;

template <class T>
std::vector<T> read_binary(const fs::path & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto length = f.tellg();
    if (length < 0 || size_t(length) % sizeof(T)) throw std::runtime_error("invalid file " + path.string());
    std::vector<T> data(size_t(length) / sizeof(T));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(data.data()), length)) throw std::runtime_error("cannot read " + path.string());
    return data;
}

struct Audio {
    int rate = 0;
    Floats mono;
};

Audio read_audio(const fs::path & file) {
    const auto wav = read_binary<uint8_t>(file);
    int n = 0;
    Audio audio;
    std::unique_ptr<float, decltype(&std::free)> stereo(
        read_wav_buf(wav.data(), wav.size(), &n, &audio.rate), std::free);
    if (!stereo || n <= 0 || audio.rate <= 0) throw std::runtime_error("cannot read WAV " + file.string());
    audio.mono.resize(n);
    for (int i = 0; i < n; ++i) audio.mono[i] = 0.5f * (stereo.get()[2 * i] + stereo.get()[2 * i + 1]);
    return audio;
}

double number(yyjson_val * v) {
    if (!yyjson_is_num(v) || !std::isfinite(yyjson_get_num(v))) throw std::runtime_error("invalid oracle number");
    return yyjson_get_num(v);
}

int velocity(double v) {
    const double quantized = std::round(std::clamp(v, 0.05, 1.0) * 1000.0) / 1000.0;
    return int(std::round(127.0 * quantized));
}

using Key = std::pair<int, int>;

bool check_hits(const char * mode, const char * branch, const std::vector<drums::Hit> & hits,
                yyjson_val * reference) {
    auto * onsets = yyjson_obj_get(reference, "onset");
    auto * keys = yyjson_obj_get(reference, "key");
    auto * velocities = yyjson_obj_get(reference, "velocity");
    const size_t count = yyjson_arr_size(keys);
    if (!yyjson_is_arr(keys) || !yyjson_is_arr(onsets) || !yyjson_is_arr(velocities) ||
        yyjson_arr_size(onsets) != count || yyjson_arr_size(velocities) != count) {
        throw std::runtime_error("invalid oracle drum decode");
    }
    std::map<Key, int> wanted;
    for (size_t i = 0; i < count; ++i) {
        const double onset = number(yyjson_arr_get(onsets, i));
        const int frame = int(std::lround(onset * drums::fps));
        const int key = int(number(yyjson_arr_get(keys, i)));
        wanted[{frame, key}] = velocity(number(yyjson_arr_get(velocities, i)));
    }
    size_t matched = 0;
    int max_velocity = 0;
    for (const auto & h : hits) {
        const auto it = wanted.find({h.frame, h.key});
        if (it == wanted.end()) continue;
        ++matched;
        max_velocity = std::max(max_velocity, std::abs(it->second - h.midi_velocity()));
        wanted.erase(it);
    }
    const double f1 = count + hits.size() == 0 ? 1.0 : 2.0 * double(matched) / double(count + hits.size());
    const bool ok = f1 >= 0.99 && max_velocity <= 2;
    std::printf("%s %s: %zu/%zu/%zu matched/native/oracle, F1 %.5f, velocity max %d %s\n",
                mode, branch, matched, hits.size(), count, f1, max_velocity, ok ? "PASS" : "FAIL");
    return ok;
}

float max_gap(const Floats & a, const Floats & b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float gap = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
        gap = std::max(gap, std::abs(a[i] - b[i]));
    }
    return gap;
}

bool check_output(const drums::Model & model, const drums::Probabilities & p, yyjson_val * reference,
                  const char * mode) {
    const auto branch = model.choose(p.router);
    auto * decode = yyjson_obj_get(reference, "drum_decode");
    const double router = number(yyjson_obj_get(reference, "drum_router"));
    const bool same = yyjson_equals_str(yyjson_obj_get(decode, "model"), drums::branch_name(branch));
    const bool ok = same && std::isfinite(p.router) && std::abs(p.router - router) <= 0.02;
    std::printf("%s router: %.8f vs %.8f, %s %s\n", mode, p.router, router, drums::branch_name(branch),
                ok ? "PASS" : "FAIL");
    auto * branches = yyjson_obj_get(reference, "drum_branches");
    const bool kit = check_hits(mode, "drum-kit", model.decode(p, drums::Branch::kit),
                                yyjson_obj_get(branches, "drum-kit"));
    const bool orch = check_hits(mode, "orchestral", model.decode(p, drums::Branch::orchestral),
                                 yyjson_obj_get(branches, "orchestral"));
    return ok && kit && orch;
}

bool check_tensors(const drums::Probabilities & p, const fs::path & capture) {
    const std::pair<const char *, const Floats *> tensors[] = {
        {"kit", &p.kit}, {"kit_vel", &p.kit_vel}, {"drum", &p.drum}, {"timpani", &p.timpani}, {"vel", &p.vel}
    };
    bool ok = true;
    for (const auto & [name, actual] : tensors) {
        const auto expected = read_binary<float>(capture / (std::string("drums.") + name + ".bin"));
        const float gap = max_gap(*actual, expected);
        const bool pass = gap < 0.001f;
        std::printf("fed %-7s max abs %.7g %s\n", name, double(gap), pass ? "PASS" : "FAIL");
        ok = pass && ok;
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "usage: dettato-drums-test MODEL_DIR FIXTURE.wav CAPTURE/cpu [both|fed|audio] [threads]\n");
        return 2;
    }
    try {
        const std::string mode = argc >= 5 ? argv[4] : "both";
        const int threads = argc >= 6 ? std::stoi(argv[5]) : 4;
        if ((mode != "both" && mode != "fed" && mode != "audio") || threads < 1) {
            throw std::invalid_argument("invalid validation mode or thread count");
        }
        const fs::path capture = argv[3];
        const auto audio = read_audio(argv[2]);
        std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
            yyjson_read_file((capture / "result.json").string().c_str(), 0, nullptr, nullptr), yyjson_doc_free);
        if (!doc) throw std::runtime_error("cannot read oracle result.json");
        auto * ref = yyjson_doc_get_root(doc.get());
        const bool skipped = yyjson_is_str(yyjson_obj_get(ref, "drum_info"));

        const drums::Model model(argv[1]);
        std::printf("%s (%d Hz)\n", fs::path(argv[2]).filename().string().c_str(), audio.rate);
        if (audio.rate < drums::min_source_rate || skipped) {
            const auto p = model.transcribe(audio.mono.data(), audio.mono.size(), audio.rate, threads);
            const bool ok = skipped && p.skipped && !p.router && p.hits.empty();
            std::printf("source rate skip %s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        ggml_backend_load_all();
        const auto gold_features = read_binary<float>(capture / "drums.features.bin");
        bool ok = true;
        if (mode != "audio") {
            const auto p = model.probabilities(gold_features, threads);
            const bool tensors = check_tensors(p, capture);
            ok = check_output(model, p, ref, "fed") && tensors;
        }
        if (mode != "fed") {
            const auto features = drums::features(audio.mono.data(), audio.mono.size(), audio.rate, threads);
            const float gap = max_gap(features, gold_features);
            std::printf("audio features max abs %.7g\n", double(gap));
            const auto p = model.probabilities(features, threads);
            const bool audio_ok = check_output(model, p, ref, "audio");
            for (const auto * tensor : {&p.kit, &p.kit_vel, &p.drum, &p.timpani, &p.vel}) {
                if (!std::all_of(tensor->begin(), tensor->end(), [](float v) { return std::isfinite(v); })) {
                    throw std::runtime_error("non-finite drum probability from audio");
                }
            }
            ok = std::isfinite(gap) && audio_ok && ok;
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "dettato-drums-test: %s\n", e.what());
        return 1;
    }
}
