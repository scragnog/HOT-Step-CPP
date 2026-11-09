// HeartCodec encoder: waveform to RVQ codes through heartmula/codec-encode.h, with
// stage-by-stage parity against heartcodec-encoder-reference.py fixtures and an
// optional decode of the codes back to audio. No Python runtime.
//
//   heartcodec-encode-probe --models DIR (--fixtures CASE | --audio FILE)
//       [--compare CASE2] [--codes-out FILE.json] [--decode OUT.wav] [--seed N]
//       [--threads N] [--device gpu] [--quant F16] [--cancel-after N]
//
// --fixtures reads CASE/input.npy and sample_rate.npy, checks every stage of chunk 0
// against CASE, then the whole clip's codes against CASE/codes.npy (match rate per
// codebook). --compare names a second case (e.g. the F16-weight reference) whose
// stages and codes are reported too. CPU unless --device gpu. --decode renders the
// codes through the S3 codec (RVQ, flow, scalar decoder in DIR) with seeded noise.
#include "heartmula/codec-encode.h"

#include "audio-io.h"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;
using namespace heartcodec;

struct Npy {
    std::vector<int64_t> shape;
    std::string type;
    std::vector<uint8_t> data;
    explicit Npy(const fs::path & path) {
        std::ifstream in(path, std::ios::binary);
        require(bool(in), "cannot open " + path.string());
        char magic[8]; in.read(magic, 8);
        require(in && std::memcmp(magic, "\x93NUMPY", 6) == 0 && magic[6] == 1, "unsupported NPY " + path.string());
        uint16_t length = 0; in.read(reinterpret_cast<char *>(&length), 2);
        std::string header(length, ' '); in.read(header.data(), length);
        std::smatch m;
        require(std::regex_search(header, m, std::regex("'descr':\\s*'([^']+)'")), "missing NPY dtype");
        type = m[1];
        require(type == "<f4" || type == "<i8", "expected f32 or i64 NPY: " + path.string());
        require(header.find("'fortran_order': False") != std::string::npos, "Fortran-order NPY");
        require(std::regex_search(header, m, std::regex("'shape':\\s*\\(([^)]*)\\)")), "missing NPY shape");
        std::string dims = m[1]; std::replace(dims.begin(), dims.end(), ',', ' ');
        std::istringstream numbers(dims);
        size_t count = 1;
        for (int64_t n; numbers >> n;) { shape.push_back(n); count *= size_t(n); }
        data.resize(count * (type == "<i8" ? 8 : 4));
        in.read(reinterpret_cast<char *>(data.data()), std::streamsize(data.size()));
        require(bool(in), "truncated NPY " + path.string());
    }
    template<class T> std::vector<T> values() const {
        require(sizeof(T) == (type == "<i8" ? 8u : 4u), "NPY dtype mismatch");
        std::vector<T> v(data.size() / sizeof(T));
        std::memcpy(v.data(), data.data(), data.size());
        return v;
    }
};

struct Stats { double max_abs = 0, rel = 0; };

static Stats diff(const float * got, const float * ref, size_t n) {
    double square = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = double(got[i]) - ref[i];
        maximum = std::max(maximum, std::abs(d)); square += d * d; norm += double(ref[i]) * ref[i];
    }
    return {maximum, std::sqrt(square / std::max(norm, 1e-30))};
}

// Relative-L2 bounds: F16-exact WavLM/Whisper and the DSP stages are compute-order
// noise; MuEncoder and query carry F16 weight rounding, which PyTorch's own
// F16-weight run puts at 2.2e-3 (semantic) and 2.9e-4 (query) relative.
static double bound(const std::string & stage, bool f16_reference) {
    if (stage == "audio_24k" || stage == "chunk_24k" || stage == "wav_16k") return 1e-5;
    // torch's float32 STFT floors near-silent mel bins (below about -82 dB) at its own
    // rounding noise; a float64 FFT, here and in numpy, does not. Pure tones expose
    // it (3.4e-4 relative on the synthetic case's tone channel); music stays near 4e-6.
    if (stage == "mu_mel") return 1e-3;
    if (stage == "whisper_mel") return 1e-5;
    if (stage == "feature_whisper" || stage == "feature_wavlm") return 1e-4;
    if (f16_reference) return 1e-4;
    return stage == "query" ? 1e-3 : 5e-3;
}

static void write_codes(const fs::path & path, const std::vector<int64_t> & codes, int64_t frames) {
    std::ofstream out(path);
    out << "{\"frames\": " << frames << ", \"codebooks\": 8, \"frame_rate\": 12.5, \"codes\": [";
    for (int64_t t = 0; t < frames; ++t) {
        out << (t ? "," : "") << '[';
        for (int q = 0; q < 8; ++q) out << (q ? "," : "") << codes[size_t(q * frames + t)];
        out << ']';
    }
    out << "]}\n";
    require(bool(out), "cannot write " + path.string());
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "heartcodec-encode-probe --models DIR (--fixtures CASE | --audio FILE) [--compare CASE2]\n"
                             "  [--codes-out FILE.json] [--decode OUT.wav] [--seed N] [--threads N] [--device gpu] [--quant F16] [--cancel-after N]\n";
                return 0;
            }
            require(key.rfind("--", 0) == 0 && i + 1 < argc, "expected --key value");
            args[key] = argv[++i];
        }
        require(args.count("--models") && (args.count("--fixtures") != args.count("--audio")),
                "usage: heartcodec-encode-probe --models DIR (--fixtures CASE | --audio FILE)");
        const std::string models = args.at("--models");
        const bool gpu = args.count("--device") && args.at("--device") == "gpu";
        if (!gpu) {
#ifdef _WIN32
            _putenv_s("GGML_BACKEND", "CPU");
#else
            setenv("GGML_BACKEND", "CPU", 1);
#endif
        }
        if (args.count("--threads")) {
#ifdef _WIN32
            _putenv_s("GGML_N_THREADS", args.at("--threads").c_str());
#else
            setenv("GGML_N_THREADS", args.at("--threads").c_str(), 1);
#endif
        }
        BackendPair bp = backend_init("HeartCodec encoder");
        ggml_backend_t backend = bp.backend;
        require(gpu == (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU),
                gpu ? "no GPU backend" : "CPU backend not selected");
        std::cout << "backend=" << ggml_backend_name(backend) << std::endl;

        std::vector<float> planar;
        int channels = 0, rate = 0;
        int64_t samples = 0;
        if (args.count("--fixtures")) {
            Npy input(fs::path(args.at("--fixtures")) / "input.npy");
            require(input.shape.size() == 2, "input.npy must be [channels, samples]");
            channels = int(input.shape[0]); samples = input.shape[1]; planar = input.values<float>();
            rate = int(Npy(fs::path(args.at("--fixtures")) / "sample_rate.npy").values<int64_t>().at(0));
        } else {
            int T = 0, sr = 0;
            float * audio = audio_read(args.at("--audio").c_str(), &T, &sr);
            require(audio != nullptr && T > 0 && sr > 0, "cannot read " + args.at("--audio"));
            channels = 2; samples = T; rate = sr;
            planar.assign(audio, audio + size_t(2) * T);
            free(audio);
        }
        std::cout << "audio channels=" << channels << " samples=" << samples << " rate=" << rate << std::endl;

        std::vector<std::pair<std::string, bool>> cases;
        if (args.count("--fixtures")) cases.push_back({args.at("--fixtures"), false});
        if (args.count("--compare")) cases.push_back({args.at("--compare"), true});
        bool pass = true;
        EncodeTrace trace;
        if (!cases.empty()) trace = [&](const std::string & stage, int chunk, int ch, const std::vector<float> & v, const std::vector<int64_t> &) {
            if (chunk != 0) return;
            for (size_t c = 0; c < cases.size(); ++c) {
                const fs::path file = fs::path(cases[c].first) / (stage + ".npy");
                if (!fs::exists(file)) {  // e.g. per-channel wavlm/whisper: the fixtures hold them unpooled
                    if (ch <= 0) std::cout << "skip    " << stage << " (no fixture in " << cases[c].first << ")" << std::endl;
                    return;
                }
                Npy ref(file);
                auto r = ref.values<float>();
                const size_t rows = ch < 0 ? 1 : size_t(ref.shape.at(0)), per = r.size() / rows;
                require(per == v.size(), stage + ": reference shape mismatch");
                auto s = diff(v.data(), r.data() + size_t(std::max(ch, 0)) * per, per);
                const bool f16 = cases[c].second;
                const bool ok = s.rel <= bound(stage, f16);
                if (!f16) pass = pass && ok;
                std::cout << std::setprecision(4) << (f16 ? "vs-f16w " : "vs-fp32 ") << std::left << std::setw(17) << stage
                          << " ch=" << std::setw(2) << ch << " max_abs=" << std::setw(11) << s.max_abs
                          << " rel_l2=" << std::setw(11) << s.rel << (f16 ? "" : (ok ? " PASS" : " FAIL")) << std::endl;
            }
        };
        // --cancel-after N: report cancelled from the Nth cancel check on (exit 3).
        int64_t checks = 0, cancel_after = args.count("--cancel-after") ? std::stoll(args.at("--cancel-after")) : -1;
        Cancel cancelled;
        if (cancel_after >= 0) cancelled = [&] { return ++checks > cancel_after; };
        auto result = encode(encoder_paths(models, args.count("--quant") ? args.at("--quant") : "F16"),
                             backend, planar, channels, samples, rate, cancelled, trace);
        std::cout << "frames=" << result.frames << " chunks=" << result.chunks;
        for (const auto & [k, s] : result.seconds) std::cout << ' ' << k << '=' << std::setprecision(4) << s << 's';
        std::cout << std::endl;
        for (const auto & [dir, f16] : cases) {
            Npy ref(fs::path(dir) / "codes.npy");
            require(ref.shape.size() == 2 && ref.shape[0] == 8, "codes.npy must be [8, frames]");
            const bool same = ref.shape[1] == result.frames;
            std::cout << (f16 ? "vs-f16w " : "vs-fp32 ") << "codes frames=" << ref.shape[1]
                      << (same ? " (same)" : " (MISMATCH)") << " match=";
            if (!same) { pass = pass && f16; std::cout << '\n'; continue; }
            auto codes = ref.values<int64_t>();
            double all = 0;
            for (int q = 0; q < 8; ++q) {
                int64_t hit = 0;
                for (int64_t t = 0; t < result.frames; ++t) hit += codes[size_t(q * result.frames + t)] == result.codes[size_t(q * result.frames + t)];
                double rate_q = double(hit) / double(result.frames); all += rate_q / 8;
                std::cout << (q ? "," : "") << std::fixed << std::setprecision(4) << rate_q;
            }
            std::cout << " mean=" << all << std::defaultfloat << std::endl;
        }
        if (args.count("--codes-out")) write_codes(args.at("--codes-out"), result.codes, result.frames);
        if (args.count("--decode")) {
            Weights rvq, flow, scalar;
            rvq.load((fs::path(models) / "heartcodec-rvq-F32.gguf").string(), "rvq", backend);
            flow.load((fs::path(models) / "heartcodec-flow-F16.gguf").string(), "flow", backend);
            scalar.load((fs::path(models) / "heartcodec-scalar-decoder-F16.gguf").string(), "scalar-decoder", backend);
            std::mt19937_64 rng(args.count("--seed") ? std::stoull(args.at("--seed")) : 42);
            std::normal_distribution<float> normal;
            Noise noise = [&](size_t n) { std::vector<float> v(n); for (auto & x : v) x = normal(rng); return v; };
            auto t0 = std::chrono::steady_clock::now();
            auto wav = decode(rvq, flow, scalar, backend, result.codes, result.frames, noise);
            std::cout << "decode seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << std::endl;
            const int T = int(wav.size() / 2);
            require(audio_write_wav(args.at("--decode").c_str(), wav.data(), T, 48000), "cannot write " + args.at("--decode"));
            std::cout << "wrote " << args.at("--decode") << " samples=" << T << std::endl;
        }
        backend_release(bp.backend, bp.cpu_backend);
        std::cout << (cases.empty() ? "DONE" : pass ? "PASS" : "FAIL") << std::endl;
        return pass ? 0 : 1;
    } catch (const Cancelled &) {
        std::cerr << "cancelled\n"; return 3;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n'; return 2;
    }
}
