// MM3 refiner CLAP audio-tower parity against the slice-1 torch fixtures.
// Every window directory under --fixtures that holds clap_vector.npy is run from
// its 44.1 kHz waveform through mm3_clap_encode. Each stage is gated on its own
// bound from fixtures.json "tolerances", fed the reference's input where it has
// one: the 48 kHz resampled audio on "clap_processor", the log-mel input_features
// computed from the fixture's own 48 kHz audio on "clap_processor", and the pooled
// vector from our own chain on "clap". The mel of OUR resampled audio is gated on
// its own, wider "clap_full_chain_mel": torchaudio resamples in float32, and its
// 1-2 ulp noise reaches 4e-3 dB in the near-silent top mel bins (see mm3-refiner-clap.h).
// The CLAP stage does not depend on the bridge step count, so only steps-1
// windows run unless --all is given. GGML_BACKEND=CPU forces the CPU backend.
#include "minimax/mm3-refiner-clap.h"

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

static void require(bool ok, const std::string & what) {
    if (!ok) {
        throw std::runtime_error(what);
    }
}

struct Npy {
    std::vector<int64_t> shape;
    std::vector<float>   data;
};

static Npy load_npy(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + path.string());
    char magic[8];
    in.read(magic, 8);
    require(in && std::memcmp(magic, "\x93NUMPY", 6) == 0 && (magic[6] == 1 || magic[6] == 2), "bad NPY " + path.string());
    uint32_t len = 0;
    for (int b = 0; b < (magic[6] == 1 ? 2 : 4); b++) {
        len |= uint32_t(uint8_t(in.get())) << (8 * b);
    }
    std::string header(len, ' ');
    in.read(header.data(), len);
    std::smatch m;
    require(std::regex_search(header, m, std::regex("'descr':\\s*'<f4'")) &&
                std::regex_search(header, m, std::regex("'fortran_order':\\s*False")) &&
                std::regex_search(header, m, std::regex("'shape':\\s*\\(([^)]*)\\)")),
            "expected C-order <f4 NPY: " + path.string());
    std::string dims = m[1];
    std::replace(dims.begin(), dims.end(), ',', ' ');
    std::istringstream ss(dims);
    Npy                r;
    size_t             count = 1;
    for (int64_t n; ss >> n;) {
        r.shape.push_back(n);
        count *= size_t(n);
    }
    r.data.resize(count);
    in.read(reinterpret_cast<char *>(r.data.data()), std::streamsize(count * 4));
    require(bool(in) && in.peek() == EOF, "NPY payload size mismatch: " + path.string());
    return r;
}

static void save_npy(const fs::path & path, const std::vector<float> & v, const std::vector<int64_t> & shape) {
    std::string dims;
    for (auto n : shape) {
        dims += std::to_string(n) + ", ";
    }
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
    header.append((16 - ((10 + header.size() + 1) % 16)) % 16, ' ');
    header += '\n';
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    uint16_t len = uint16_t(header.size());
    out.write(reinterpret_cast<const char *>(&len), 2);
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * 4));
    require(bool(out), "cannot save " + path.string());
}

struct Metrics {
    double max_abs = 0, rel = 0, cosine = 1;
};

static Metrics measure(const float * got, const float * ref, size_t n) {
    double  diff = 0, rr = 0, gg = 0, gr = 0;
    Metrics m;
    for (size_t i = 0; i < n; i++) {
        require(std::isfinite(got[i]) && std::isfinite(ref[i]), "non-finite value");
        double d = double(got[i]) - ref[i];
        m.max_abs = std::max(m.max_abs, std::abs(d));
        diff += d * d;
        rr += double(ref[i]) * ref[i];
        gg += double(got[i]) * got[i];
        gr += double(got[i]) * ref[i];
    }
    m.rel    = std::sqrt(diff) / std::max(std::sqrt(rr), 1e-30);
    m.cosine = rr == 0 && gg == 0 ? 1.0 : (rr == 0 || gg == 0 ? 0.0 : gr / std::sqrt(rr * gg));
    return m;
}

struct Bound {
    double max_abs = 0, rel = 0, cosine = 0;
    bool   ok(const Metrics & m) const { return m.max_abs <= max_abs && m.rel <= rel && m.cosine >= cosine; }
};

static Bound read_bound(yyjson_val * tolerances, const char * stage) {
    yyjson_val * t = yyjson_obj_get(tolerances, stage);
    Bound        b{ yyjson_get_num(yyjson_obj_get(t, "max_abs")), yyjson_get_num(yyjson_obj_get(t, "relative_rmse")),
                    yyjson_get_num(yyjson_obj_get(t, "cosine")) };
    require(b.max_abs > 0 && b.rel > 0 && b.cosine > 0, std::string("tolerances.") + stage + " missing from the contract");
    return b;
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        bool                               all = false;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--all") {
                all = true;
            } else if (i + 1 < argc) {
                args[argv[i]] = argv[i + 1];
                i++;
            }
        }
        if (!args.count("--clap") || !args.count("--fixtures") || !args.count("--contract")) {
            std::cout << "mm3-refiner-clap-probe --clap CLAP.gguf --contract fixtures.json --fixtures DIR\n"
                         "  [--out DIR] [--all]\n"
                         "Runs every window directory under DIR holding clap_vector.npy (steps-1 only unless --all)\n"
                         "and gates the 48 kHz processor audio and input_features on tolerances.clap_processor and\n"
                         "the pooled vector on tolerances.clap. --out writes the engine's clap_vector and\n"
                         "clap_processor_input_features per window for tools/mm3-refiner/check-parity.py.\n"
                         "Exit 0 pass, 1 numerical fail, 2 error.\n";
            return 2;
        }
        // Tolerances come from the frozen contract, never from the command line.
        std::ifstream cf(args.at("--contract"), std::ios::binary);
        std::string   contract((std::istreambuf_iterator<char>(cf)), {});
        yyjson_doc *  doc = yyjson_read(contract.data(), contract.size(), 0);
        require(doc != nullptr, "cannot parse " + args.at("--contract"));
        yyjson_val * tol = yyjson_obj_get(yyjson_doc_get_root(doc), "tolerances");
        const Bound  proc = read_bound(tol, "clap_processor"), chain = read_bound(tol, "clap_full_chain_mel"),
                     vec  = read_bound(tol, "clap");
        yyjson_doc_free(doc);

        std::vector<fs::path> windows;
        for (auto & e : fs::recursive_directory_iterator(args.at("--fixtures"))) {
            if (!e.is_regular_file() || e.path().filename() != "clap_vector.npy") {
                continue;
            }
            const fs::path dir = e.path().parent_path();
            if (all || dir.parent_path().filename() == "steps-1") {
                windows.push_back(dir);
            }
        }
        std::sort(windows.begin(), windows.end());
        require(!windows.empty(), "no window directories with clap_vector.npy under " + args.at("--fixtures"));

        Mm3Clap     clap;
        std::string err;
        require(mm3_clap_load(&clap, args.at("--clap"), &err), err);
        std::cout << "backend=" << ggml_backend_name(clap.backend) << " windows=" << windows.size()
                  << " processor(atol=" << proc.max_abs << " rtol=" << proc.rel << " cos=" << proc.cosine
                  << ") vector(atol=" << vec.max_abs << " rtol=" << vec.rel << " cos=" << vec.cosine << ")\n";

        const fs::path root = fs::absolute(args.at("--fixtures"));
        bool           pass = true;
        Metrics        worst_aud, worst_mel, worst_feat, worst_vec;
        worst_aud.cosine = worst_mel.cosine = worst_feat.cosine = worst_vec.cosine = 1;
        auto fold = [](Metrics & w, const Metrics & m) {
            w.max_abs = std::max(w.max_abs, m.max_abs);
            w.rel     = std::max(w.rel, m.rel);
            w.cosine  = std::min(w.cosine, m.cosine);
        };
        std::cout << std::setprecision(4);
        for (const fs::path & dir : windows) {
            const std::string name = fs::relative(fs::absolute(dir), root).generic_string();
            Npy               wav  = load_npy(dir / "waveform.npy");
            Npy               raud = load_npy(dir / "clap_processor_audio.npy");
            Npy               rfea = load_npy(dir / "clap_processor_input_features.npy");
            Npy               rvec = load_npy(dir / "clap_vector.npy");
            require(wav.shape.size() >= 2 && wav.shape[wav.shape.size() - 2] == 2 && wav.data.size() == size_t(2 * wav.shape.back()),
                    name + ": waveform must be one stereo clip");
            require(rfea.shape == std::vector<int64_t>({ 1, 1, MM3_CLAP_FRAMES, MM3_CLAP_MELS }) && rvec.shape == std::vector<int64_t>({ 1, MM3_CLAP_PROJ }),
                    name + ": CLAP fixture shape is not [1,1,1001,64] / [1,512]");

            Mm3ClapResult r;
            auto          t0 = std::chrono::steady_clock::now();
            require(mm3_clap_encode(&clap, wav.data.data(), wav.shape.back(), &r, &err), name + ": " + err);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            require(r.audio_48k.size() == raud.data.size(),
                    name + ": 48 kHz length " + std::to_string(r.audio_48k.size()) + " vs " + std::to_string(raud.data.size()));

            Metrics ma = measure(r.audio_48k.data(), raud.data.data(), raud.data.size());
            Metrics mf = measure(r.features.data(), rfea.data.data(), rfea.data.size());
            Metrics mv = measure(r.vector.data(), rvec.data.data(), rvec.data.size());
            // The mel front end alone, fed the fixture's own 48 kHz audio instead of ours.
            std::vector<float> iso;
            require(mm3_clap_features(raud.data, &iso, &err), name + ": " + err);
            Metrics mi = measure(iso.data(), rfea.data.data(), rfea.data.size());
            const bool ok = proc.ok(ma) && proc.ok(mi) && chain.ok(mf) && vec.ok(mv);
            pass          = pass && ok;
            fold(worst_aud, ma);
            fold(worst_mel, mi);
            fold(worst_feat, mf);
            fold(worst_vec, mv);
            std::cout << std::left << std::setw(26) << name << std::right << " ms=" << std::setw(4) << std::fixed
                      << std::setprecision(0) << ms << std::defaultfloat << std::setprecision(4)
                      << " | audio48k n=" << raud.data.size() << " max_abs=" << ma.max_abs
                      << " | mel(ref audio) max_abs=" << mi.max_abs << " rel=" << mi.rel
                      << " | vec max_abs=" << mv.max_abs << " rel=" << mv.rel << " cos=" << std::setprecision(9)
                      << mv.cosine << std::setprecision(4) << " | mel(own audio) max_abs=" << mf.max_abs << " rel=" << mf.rel
                      << (ok ? " PASS" : " FAIL") << "\n";
            if (args.count("--out")) {
                fs::path out = fs::path(args.at("--out")) / name;
                fs::create_directories(out);
                save_npy(out / "clap_vector.npy", r.vector, rvec.shape);
                save_npy(out / "clap_processor_input_features.npy", r.features, rfea.shape);
            }
        }
        auto line = [](const char * what, const Metrics & w) {
            std::cout << "  " << std::left << std::setw(16) << what << std::right << " max_abs=" << std::setprecision(4)
                      << w.max_abs << " rel=" << w.rel << " cos=" << std::setprecision(9) << w.cosine << "\n";
        };
        std::cout << "worst over " << windows.size() << " windows:\n";
        line("audio48k", worst_aud);
        line("mel(ref audio)", worst_mel);
        line("vector", worst_vec);
        line("mel(own audio)", worst_feat);
        std::cout << (pass ? "PASS" : "FAIL") << " clap parity, " << windows.size()
                  << " windows x (48 kHz audio, mel on reference audio, mel on own audio, vector)\n";
        mm3_clap_free(&clap);
        return pass ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
