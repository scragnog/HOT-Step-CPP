// MM3 refiner DiT and bridge parity against the slice-1 torch fixtures.
// Every window directory under --fixtures that holds dit_forward.npy is run from
// its DAV latents, normalised MERT states and CLAP vector through
// mm3_dit_normalize, mm3_dit_bridge and mm3_dit_denormalize. Gates, from
// fixtures.json "tolerances": normalized_latents on the bridge input, dit on the
// t=1 forward and every velocity, bridge on every Euler state and the final
// latents. The engine bridge runs free (never teacher-forced), so drift
// accumulates as it would in use. --repeat runs each bridge twice and requires
// bit-identical latents. --mert-dir swaps in engine MERT states for an
// end-to-end conditioning check. GGML_BACKEND=CPU forces the CPU backend.
#include "minimax/mm3-refiner-dit.h"

#include <cstring>
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
    void   worst_of(const Metrics & o) {
        max_abs = std::max(max_abs, o.max_abs);
        rel     = std::max(rel, o.rel);
        cosine  = std::min(cosine, o.cosine);
    }
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

struct Gate {
    double atol = 0, rtol = 0, ctol = 0;
    bool   ok(const Metrics & m) const { return m.max_abs <= atol && m.rel <= rtol && m.cosine >= ctol; }
};

static std::ostream & operator<<(std::ostream & os, const Metrics & m) {
    return os << std::setprecision(4) << "max_abs=" << m.max_abs << " rel_rmse=" << m.rel << " cos=" << std::setprecision(9)
              << m.cosine << std::setprecision(4);
}

static std::string npy_name(const char * stem, int k) {
    char b[32];
    snprintf(b, sizeof b, "%s-%02d.npy", stem, k);
    return b;
}

int main(int argc, char ** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) {
            args[argv[i]] = argv[i + 1];
        }
        if (!args.count("--refiner") || !args.count("--fixtures") || !args.count("--contract")) {
            std::cout << "mm3-refiner-dit-probe --refiner REFINER.gguf --contract fixtures.json --fixtures DIR\n"
                         "  [--repeat 1] [--out DIR] [--mert-dir DIR]\n"
                         "Runs every window directory under DIR holding dit_forward.npy through the bridge and\n"
                         "gates the forward, every velocity and state, and the final latents on fixtures.json.\n"
                         "--repeat 1 reruns each bridge and requires bit-identical latents. --out writes the\n"
                         "engine's refined_latents per window. --mert-dir reads each window's mert_normalized.npy\n"
                         "from DIR/<window> (mm3-refiner-mert-probe --out) so the bridge runs on engine MERT.\n"
                         "Exit 0 pass, 1 numerical fail, 2 error.\n";
            return 2;
        }
        // Tolerances come from the frozen contract, never from the command line.
        std::ifstream cf(args.at("--contract"), std::ios::binary);
        std::string   contract((std::istreambuf_iterator<char>(cf)), {});
        yyjson_doc *  doc = yyjson_read(contract.data(), contract.size(), 0);
        require(doc != nullptr, "cannot parse " + args.at("--contract"));
        auto gate = [&](const char * stage) {
            yyjson_val * t = yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(doc), "tolerances"), stage);
            Gate         g{ yyjson_get_num(yyjson_obj_get(t, "max_abs")), yyjson_get_num(yyjson_obj_get(t, "relative_rmse")),
                    yyjson_get_num(yyjson_obj_get(t, "cosine")) };
            require(g.atol > 0 && g.rtol > 0 && g.ctol > 0, std::string("tolerances.") + stage + " missing from the contract");
            return g;
        };
        const Gate g_norm = gate("normalized_latents"), g_dit = gate("dit"), g_bridge = gate("bridge");
        yyjson_doc_free(doc);
        const bool repeat = args.count("--repeat") && args.at("--repeat") != "0";

        std::vector<fs::path> windows;
        for (auto & e : fs::recursive_directory_iterator(args.at("--fixtures"))) {
            if (e.is_regular_file() && e.path().filename() == "dit_forward.npy") {
                windows.push_back(e.path().parent_path());
            }
        }
        std::sort(windows.begin(), windows.end());
        require(!windows.empty(), "no window directories with dit_forward.npy under " + args.at("--fixtures"));

        Mm3RefinerDit dit;
        std::string   err;
        require(mm3_dit_load(&dit, args.at("--refiner"), &err), err);
        std::cout << "backend=" << ggml_backend_name(dit.backend) << " windows=" << windows.size()
                  << " repeat=" << (repeat ? "yes" : "no") << "\n";

        const fs::path                               root = fs::absolute(args.at("--fixtures"));
        bool                                         pass = true, all_repeat = true;
        std::map<int, std::vector<Metrics>>          worst_state, worst_vel;  // steps -> per step index
        std::map<int, Metrics>                       worst_final;
        Metrics                                      worst_fwd, worst_norm;
        double                                       total_ms = 0;
        for (const fs::path & dir : windows) {
            const std::string name  = fs::relative(fs::absolute(dir), root).generic_string();
            Npy               dav   = load_npy(dir / "dav_latents.npy");
            Npy               nref  = load_npy(dir / "normalized_latents.npy");
            Npy               mert  = load_npy(args.count("--mert-dir") ? fs::path(args.at("--mert-dir")) / name / "mert_normalized.npy"
                                                                    : dir / "mert_normalized.npy");
            Npy               clap  = load_npy(dir / "clap_vector.npy");
            Npy               sch   = load_npy(dir / "schedule.npy");
            Npy               fwd   = load_npy(dir / "dit_forward.npy");
            Npy               fin   = load_npy(dir / "refined_latents.npy");
            require(dav.shape.size() == 3 && dav.shape[0] == 1 && dav.shape[1] == MM3_DIT_LATENT, name + ": dav_latents must be [1,128,L]");
            const int64_t L = dav.shape[2];
            require(nref.shape == std::vector<int64_t>({ 1, L, MM3_DIT_LATENT }) && fwd.shape == nref.shape, name + ": latents not [1,L,128]");
            require(mert.shape == std::vector<int64_t>({ 1, MM3_DIT_STATES, L, MM3_DIT_COND }), name + ": mert_normalized not [1,13,L,768]");
            require(clap.shape == std::vector<int64_t>({ 1, MM3_DIT_STYLE }), name + ": clap_vector not [1,512]");
            require(fin.shape == dav.shape && sch.shape.size() == 1 && sch.shape[0] >= 2, name + ": refined_latents or schedule shape");
            const int                steps = int(sch.shape[0] - 1);
            const std::vector<float> s     = mm3_dit_schedule(steps);
            const bool               s_bit = std::memcmp(s.data(), sch.data.data(), s.size() * 4) == 0;

            std::vector<float> x0 = mm3_dit_normalize(&dit, dav.data.data(), L);
            Metrics            mn = measure(x0.data(), nref.data.data(), x0.size());
            bool               ok = s_bit && g_norm.ok(mn);
            worst_norm.worst_of(mn);

            auto & ws = worst_state[steps];
            auto & wv = worst_vel[steps];
            ws.resize(size_t(steps));
            wv.resize(size_t(steps));
            std::vector<std::string> lines;
            Metrics                  mf;
            auto on_step = [&](int k, const float * state, const float * vel) {
                Npy     rs = load_npy(dir / npy_name("step", k + 1));
                Npy     rv = load_npy(dir / npy_name("velocity", k));
                Metrics a  = measure(vel, rv.data.data(), rv.data.size());
                Metrics b  = measure(state, rs.data.data(), rs.data.size());
                if (k == 0) {
                    mf = measure(vel, fwd.data.data(), fwd.data.size());
                    worst_fwd.worst_of(mf);
                    ok = ok && g_dit.ok(mf);
                }
                const bool sok = g_dit.ok(a) && g_bridge.ok(b);
                ok             = ok && sok;
                wv[size_t(k)].worst_of(a);
                ws[size_t(k)].worst_of(b);
                std::ostringstream os;
                os << "  step=" << std::setw(2) << k << " velocity " << a << " | state " << b << (sok ? "" : " FAIL");
                lines.push_back(os.str());
                return true;
            };
            std::vector<float> out;
            auto               t0 = std::chrono::steady_clock::now();
            require(mm3_dit_bridge(&dit, x0.data(), mert.data.data(), clap.data.data(), L, steps, &out, on_step, &err), name + ": " + err);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            total_ms += ms;
            std::vector<float> lat = mm3_dit_denormalize(&dit, out.data(), L);
            Metrics            mfin = measure(lat.data(), fin.data.data(), fin.data.size());
            ok                      = ok && g_bridge.ok(mfin);
            worst_final[steps].worst_of(mfin);

            std::string rep = "";
            if (repeat) {
                std::vector<float> again;
                require(mm3_dit_bridge(&dit, x0.data(), mert.data.data(), clap.data.data(), L, steps, &again, nullptr, &err), name + ": " + err);
                const bool same = again.size() == out.size() && std::memcmp(again.data(), out.data(), out.size() * 4) == 0;
                all_repeat      = all_repeat && same;
                ok              = ok && same;
                rep             = same ? " repeat=bit-identical" : " repeat=DIFFERS";
            }
            pass = pass && ok;
            std::cout << name << " L=" << L << " steps=" << steps << " ms=" << std::setprecision(5) << ms
                      << " schedule=" << (s_bit ? "bit-exact" : "DIFFERS") << " normalized " << mn << "\n"
                      << "  forward(t=1) " << mf << "\n";
            for (auto & l : lines) {
                std::cout << l << "\n";
            }
            std::cout << "  final " << mfin << rep << (ok ? " PASS" : " FAIL") << "\n";
            if (args.count("--out")) {
                fs::path o = fs::path(args.at("--out")) / name;
                fs::create_directories(o);
                save_npy(o / "refined_latents.npy", lat, fin.shape);
            }
        }
        std::cout << "summary over " << windows.size() << " windows, " << std::setprecision(5) << total_ms / 1000.0
                  << " s bridge time\n  normalized " << worst_norm << "\n  forward(t=1) " << worst_fwd << "\n";
        for (auto & [steps, ws] : worst_state) {
            std::cout << "  steps=" << steps << " worst per step (velocity | state):\n";
            for (size_t k = 0; k < ws.size(); k++) {
                std::cout << "    step=" << std::setw(2) << k << " " << worst_vel[steps][k] << " | " << ws[k] << "\n";
            }
            std::cout << "  steps=" << steps << " final " << worst_final[steps]
                      << (g_bridge.ok(worst_final[steps]) ? " within bridge tolerance" : " OUTSIDE bridge tolerance") << "\n";
        }
        if (repeat) {
            std::cout << "repeat: " << (all_repeat ? "every window bit-identical" : "MISMATCH") << "\n";
        }
        std::cout << (pass ? "PASS" : "FAIL") << " dit parity, " << windows.size() << " windows\n";
        mm3_dit_free(&dit);
        return pass ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
