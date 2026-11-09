// MM3 refiner MERT-v1-95M parity against the slice-1 torch fixtures.
// Every window directory under --fixtures that holds mert_normalized.npy is run
// from its 44.1 kHz waveform and DAV frame count through mm3_mert_encode, and
// all 13 normalised states are gated on fixtures.json "tolerances.mert".
// GGML_BACKEND=CPU forces the CPU backend.
#include "minimax/mm3-refiner-mert.h"

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

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) {
            args[argv[i]] = argv[i + 1];
        }
        if (!args.count("--mert") || !args.count("--refiner") || !args.count("--fixtures") || !args.count("--contract")) {
            std::cout << "mm3-refiner-mert-probe --mert MERT.gguf --refiner REFINER.gguf --contract fixtures.json\n"
                         "  --fixtures DIR [--out DIR]\n"
                         "Runs every window directory under DIR holding mert_normalized.npy and gates all 13\n"
                         "normalised states on tolerances.mert. --out writes the engine's mert_normalized per\n"
                         "window for tools/mm3-refiner/check-parity.py. Exit 0 pass, 1 numerical fail, 2 error.\n";
            return 2;
        }
        // Tolerances come from the frozen contract, never from the command line.
        std::ifstream      cf(args.at("--contract"), std::ios::binary);
        std::string        contract((std::istreambuf_iterator<char>(cf)), {});
        yyjson_doc *       doc = yyjson_read(contract.data(), contract.size(), 0);
        require(doc != nullptr, "cannot parse " + args.at("--contract"));
        yyjson_val * tol   = yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(doc), "tolerances"), "mert");
        const double atol  = yyjson_get_num(yyjson_obj_get(tol, "max_abs"));
        const double rtol  = yyjson_get_num(yyjson_obj_get(tol, "relative_rmse"));
        const double ctol  = yyjson_get_num(yyjson_obj_get(tol, "cosine"));
        yyjson_doc_free(doc);
        require(atol > 0 && rtol > 0 && ctol > 0, "tolerances.mert missing from the contract");

        std::vector<fs::path> windows;
        for (auto & e : fs::recursive_directory_iterator(args.at("--fixtures"))) {
            if (e.is_regular_file() && e.path().filename() == "mert_normalized.npy") {
                windows.push_back(e.path().parent_path());
            }
        }
        std::sort(windows.begin(), windows.end());
        require(!windows.empty(), "no window directories with mert_normalized.npy under " + args.at("--fixtures"));

        Mm3Mert     mert;
        std::string err;
        require(mm3_mert_load(&mert, args.at("--mert"), args.at("--refiner"), &err), err);
        std::cout << "backend=" << ggml_backend_name(mert.backend) << " windows=" << windows.size() << " atol=" << atol
                  << " rtol=" << rtol << " cos=" << ctol << "\n";

        const fs::path root = fs::absolute(args.at("--fixtures"));
        bool           pass = true;
        Metrics        worst[MM3_MERT_STATES];
        for (auto & w : worst) {
            w.cosine = 1;
        }
        for (const fs::path & dir : windows) {
            const std::string name = fs::relative(fs::absolute(dir), root).generic_string();
            Npy               wav  = load_npy(dir / "waveform.npy");
            Npy               dav  = load_npy(dir / "dav_latents.npy");
            Npy               ref  = load_npy(dir / "mert_normalized.npy");
            Npy               rint = load_npy(dir / "mert_interpolated.npy");
            Npy               raud = load_npy(dir / "mert_processor_audio.npy");
            Npy               rinp = load_npy(dir / "mert_processor_input_values.npy");
            require(wav.shape.size() >= 2 && wav.shape[wav.shape.size() - 2] == 2 && wav.data.size() == size_t(2 * wav.shape.back()),
                    name + ": waveform must be one stereo clip");
            require(dav.shape.size() == 3 && dav.shape[1] == 128, name + ": dav_latents must be [1,128,L]");
            const int64_t L = dav.shape[2];
            require(ref.shape == std::vector<int64_t>({ 1, MM3_MERT_STATES, L, MM3_MERT_DIM }) && rint.shape == ref.shape,
                    name + ": MERT fixture shape is not [1,13,L,768]");

            Mm3MertResult r;
            auto          t0 = std::chrono::steady_clock::now();
            require(mm3_mert_encode(&mert, wav.data.data(), wav.shape.back(), L, &r, &err), name + ": " + err);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            require(r.audio_24k.size() == raud.data.size() && r.input_values.size() == rinp.data.size(),
                    name + ": 24 kHz length " + std::to_string(r.audio_24k.size()) + " vs " + std::to_string(raud.data.size()));

            Metrics ra = measure(r.audio_24k.data(), raud.data.data(), raud.data.size());
            Metrics ri = measure(r.input_values.data(), rinp.data.data(), rinp.data.size());
            std::cout << std::setprecision(4) << name << " L=" << L << " mert_frames=" << r.mert_frames << " ms=" << ms
                      << " resample max_abs=" << ra.max_abs << " cos=" << std::setprecision(9) << ra.cosine
                      << std::setprecision(4) << " input_values max_abs=" << ri.max_abs << "\n";
            const size_t per = size_t(L * MM3_MERT_DIM);
            for (int s = 0; s < MM3_MERT_STATES; s++) {
                Metrics n  = measure(r.normalized.data() + s * per, ref.data.data() + s * per, per);
                Metrics in = measure(r.interpolated.data() + s * per, rint.data.data() + s * per, per);
                bool    ok = n.max_abs <= atol && n.rel <= rtol && n.cosine >= ctol;
                pass       = pass && ok;
                worst[s].max_abs = std::max(worst[s].max_abs, n.max_abs);
                worst[s].rel     = std::max(worst[s].rel, n.rel);
                worst[s].cosine  = std::min(worst[s].cosine, n.cosine);
                std::cout << "  layer=" << std::setw(2) << s << std::setprecision(4) << " max_abs=" << n.max_abs
                          << " rel_rmse=" << n.rel << " cosine=" << std::setprecision(9) << n.cosine
                          << std::setprecision(4) << " | raw max_abs=" << in.max_abs << " rel_rmse=" << in.rel
                          << (ok ? " PASS" : " FAIL") << "\n";
            }
            if (args.count("--out")) {
                fs::path out = fs::path(args.at("--out")) / name;
                fs::create_directories(out);
                save_npy(out / "mert_normalized.npy", r.normalized, ref.shape);
            }
        }
        std::cout << "worst over " << windows.size() << " windows (normalised states):\n";
        for (int s = 0; s < MM3_MERT_STATES; s++) {
            std::cout << "  layer=" << std::setw(2) << s << std::setprecision(4) << " max_abs=" << worst[s].max_abs
                      << " rel_rmse=" << worst[s].rel << " cosine=" << std::setprecision(9) << worst[s].cosine << "\n";
        }
        std::cout << (pass ? "PASS" : "FAIL") << " mert parity, " << windows.size() << " windows x 13 states\n";
        mm3_mert_free(&mert);
        return pass ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
