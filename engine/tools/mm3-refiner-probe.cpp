// MM3 latent refiner, whole pipeline: parity against the slice-1 torch fixtures and
// the behavioural checks the integration promises. Links acestep-core, so it runs
// the real ModelStore and the four real modules.
//
//   parity   one or more clips end to end from fixtures-v1 (input.npy and the clip's
//            CLAP vectors, which stand in for the style producer): window starts,
//            overlap weights, DAV latents per window, stitched latents, raw decoded
//            audio, loudness gain, final audio. Gates are the "tolerances" in
//            fixtures.json; the loudness gain is gated at 0.01 dB.
//   behave   bypass, no-style, missing and corrupt files, cancel, silence, odd tail,
//            mono and stereo, 44.1 and 48 kHz, mix endpoints, seams, parameter
//            bounds, eviction under both store policies, repeat-run residency.
//   direct   one dense-attention run of --seconds of audio (the limit probe).
//
// GGML_BACKEND=CPU forces the CPU backend. Exit 0 pass, 1 fail, 2 error.
#include "minimax/mm3-refiner.h"

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

// float32 or float64 C-order NPY, read as float.
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
    const bool  f4 = std::regex_search(header, m, std::regex("'descr':\\s*'<f4'"));
    const bool  f8 = std::regex_search(header, m, std::regex("'descr':\\s*'<f8'"));
    const bool  i8 = std::regex_search(header, m, std::regex("'descr':\\s*'<i8'"));
    require((f4 || f8 || i8) && std::regex_search(header, m, std::regex("'fortran_order':\\s*False")) &&
                std::regex_search(header, m, std::regex("'shape':\\s*\\(([^)]*)\\)")),
            "expected C-order <f4, <f8 or <i8 NPY: " + path.string());
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
    if (f4) {
        in.read(reinterpret_cast<char *>(r.data.data()), std::streamsize(count * 4));
    } else {
        std::vector<char> raw(count * 8);
        in.read(raw.data(), std::streamsize(raw.size()));
        for (size_t i = 0; i < count; i++) {
            if (f8) {
                double d;
                std::memcpy(&d, raw.data() + i * 8, 8);
                r.data[i] = float(d);
            } else {
                int64_t v;
                std::memcpy(&v, raw.data() + i * 8, 8);
                r.data[i] = float(v);  // sample starts: exact below 2^24 is not needed, compared as int64 below
            }
        }
    }
    require(bool(in), "short NPY payload: " + path.string());
    return r;
}

static std::vector<int64_t> load_npy_i64(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot open " + path.string());
    in.seekg(8);
    uint16_t len = 0;
    in.read(reinterpret_cast<char *>(&len), 2);
    in.seekg(10 + len);
    std::vector<int64_t> v;
    int64_t              x;
    while (in.read(reinterpret_cast<char *>(&x), 8)) {
        v.push_back(x);
    }
    return v;
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

static std::map<std::string, std::string> parse_args(int argc, char ** argv) {
    std::map<std::string, std::string> args;
    for (int i = 2; i + 1 < argc; i += 2) {
        args[argv[i]] = argv[i + 1];
    }
    return args;
}

static const std::vector<std::string> ALL_CLIPS = { "boombap-s1001", "boombap-s1002", "cinematic-s1001", "cinematic-s1002",
                                                    "folk-s1001",    "folk-s1002",    "soulpop-s1001",   "soulpop-s1002",
                                                    "synthwave-s1001", "synthwave-s1002", "short",        "odd" };

static Mm3RefinerPaths paths_in(const std::string & dir) {
    return { dir + "/mm3-enc-f16.gguf", dir + "/mm3-refiner-mert-f32.gguf", dir + "/mm3-refiner-refiner-f32.gguf",
             dir + "/mm3-voc-f16.gguf" };
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// The fixture CLAP vector of the clip's window w at the given steps.
static Mm3StyleFn fixture_style(const fs::path & steps_dir) {
    return [steps_dir](int w, const float *, int64_t, float * out, std::string * err) {
        char name[32];
        snprintf(name, sizeof name, "window-%02d", w);
        try {
            Npy v = load_npy(steps_dir / name / "clap_vector.npy");
            std::memcpy(out, v.data.data(), 512 * sizeof(float));
            return true;
        } catch (const std::exception & e) {
            *err = e.what();
            return false;
        }
    };
}

// ── parity ──────────────────────────────────────────────────────────────────

static int cmd_parity(int argc, char ** argv) {
    auto args = parse_args(argc, argv);
    if (!args.count("--contract") || !args.count("--fixtures") || !args.count("--models")) {
        std::cout << "mm3-refiner-probe parity --contract fixtures.json --fixtures DIR --models DIR\n"
                     "  [--clips a,b|all] [--steps 8] [--policy strict|never]\n";
        return 2;
    }
    std::ifstream cf(args.at("--contract"), std::ios::binary);
    std::string   contract((std::istreambuf_iterator<char>(cf)), {});
    yyjson_doc *  doc = yyjson_read(contract.data(), contract.size(), 0);
    require(doc != nullptr, "cannot parse the contract");
    auto gate = [&](const char * stage) {
        yyjson_val * t = yyjson_obj_get(yyjson_obj_get(yyjson_doc_get_root(doc), "tolerances"), stage);
        Gate         g{ yyjson_get_num(yyjson_obj_get(t, "max_abs")), yyjson_get_num(yyjson_obj_get(t, "relative_rmse")),
                yyjson_get_num(yyjson_obj_get(t, "cosine")) };
        require(g.atol > 0 && g.rtol > 0 && g.ctol > 0, std::string("tolerances.") + stage + " missing from the contract");
        return g;
    };
    const Gate g_dav = gate("dav"), g_bridge = gate("bridge"), g_wave = gate("waveform"), g_overlap = gate("overlap");
    yyjson_doc_free(doc);

    std::vector<std::string> clips;
    const std::string        want = args.count("--clips") ? args.at("--clips") : "all";
    if (want == "all") {
        clips = ALL_CLIPS;
    } else {
        std::stringstream ss(want);
        for (std::string c; std::getline(ss, c, ',');) {
            clips.push_back(c);
        }
    }
    const int         steps = args.count("--steps") ? std::stoi(args.at("--steps")) : 8;
    ModelStore *      store = store_create(args.count("--policy") && args.at("--policy") == "never" ? EVICT_NEVER : EVICT_STRICT);
    const Mm3RefinerPaths paths = paths_in(args.at("--models"));
    bool pass = true;
    Metrics w_dav, w_lat, w_raw, w_fin, w_ovl;
    double  worst_gain = 0, total_ms = 0;
    for (const std::string & clip : clips) {
        const fs::path cdir  = fs::path(args.at("--fixtures")) / clip;
        const fs::path sdir  = cdir / ("steps-" + std::to_string(steps));
        Npy            input = load_npy(cdir / "input.npy");
        require(input.shape.size() == 2 && input.shape[0] == 2, clip + ": input.npy not [2,N]");
        const int64_t n = input.shape[1];

        Mm3RefinerParams p;
        p.steps   = steps;
        p.capture = true;
        Mm3RefinerResult r;
        std::string      err;
        std::map<int, double> phase_ms;
        auto                  t_phase = std::chrono::steady_clock::now();
        int                   last    = -1;
        Mm3RefinerHooks       hooks;
        hooks.progress = [&](Mm3RefinerPhase ph, int, int) {
            if (int(ph) != last) {
                if (last >= 0) {
                    phase_ms[last] += ms_since(t_phase);
                }
                t_phase = std::chrono::steady_clock::now();
                last    = int(ph);
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = mm3_refiner_run(store, paths, p, input.data.data(), n, 44100, fixture_style(sdir), hooks, &r, &err);
        const double ms = ms_since(t0);
        total_ms += ms;
        require(ok, clip + ": " + err);

        bool clip_ok = true;
        std::ostringstream os;
        // windows: plan and per-window blend
        Mm3RefinerPlan plan;
        const int64_t  padded = (n + 511) / 512 * 512;
        require(mm3_refiner_plan(padded, p, &plan, &err) == 0, err);
        const std::vector<int64_t> starts = load_npy_i64(sdir / "window_starts.npy");
        const bool                 starts_ok = starts == plan.starts;
        clip_ok                              = clip_ok && starts_ok;
        Metrics m_dav, m_ovl;
        for (size_t w = 0; w < plan.starts.size(); w++) {
            char name[32];
            snprintf(name, sizeof name, "window-%02zu", w);
            Npy dav = load_npy(sdir / name / "dav_latents.npy");
            require(dav.data.size() == r.window_dav[w].size(), clip + ": DAV latent size differs in " + name);
            m_dav.worst_of(measure(r.window_dav[w].data(), dav.data.data(), dav.data.size()));
            if (!plan.single()) {
                Npy           ov        = load_npy(sdir / name / "overlap.npy");
                const int64_t available = std::min<int64_t>(plan.window_samples / 512, plan.frames - plan.starts[w] / 512);
                std::vector<float> b    = mm3_refiner_blend(plan, w, available);
                require(int64_t(ov.data.size()) == available, clip + ": overlap length differs in " + name);
                m_ovl.worst_of(measure(b.data(), ov.data.data(), b.size()));
            }
        }
        Npy lat = load_npy(sdir / "latents.npy");
        require(lat.data.size() == r.latents.size(), clip + ": stitched latent size differs");
        Metrics m_lat = measure(r.latents.data(), lat.data.data(), lat.data.size());
        Npy     dec   = load_npy(sdir / "decoded.npy");
        require(dec.data.size() == r.raw_audio.size(), clip + ": decoded size differs");
        Metrics m_raw = measure(r.raw_audio.data(), dec.data.data(), dec.data.size());
        Npy     fin   = load_npy(sdir / "final_audio.npy");
        require(fin.data.size() == r.audio.size(), clip + ": final audio size differs");
        Metrics m_fin = measure(r.audio.data(), fin.data.data(), fin.data.size());

        // the loudness gain torch applied is recorded in the manifest
        double gain_ref = 0;
        {
            std::ifstream mf(cdir / "manifest.json", std::ios::binary);
            std::string   mj((std::istreambuf_iterator<char>(mf)), {});
            yyjson_doc *  md = yyjson_read(mj.data(), mj.size(), 0);
            require(md != nullptr, clip + ": manifest.json");
            size_t idx, max;
            yyjson_val * e;
            yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(md), "wav"), idx, max, e) {
                if (std::string(yyjson_get_str(yyjson_obj_get(e, "path"))) == "steps-" + std::to_string(steps) + "/final.wav") {
                    gain_ref = yyjson_get_num(yyjson_obj_get(e, "gain_db"));
                }
            }
            yyjson_doc_free(md);
        }
        const double dgain = std::abs(r.gain_db - gain_ref);
        worst_gain         = std::max(worst_gain, dgain);
        clip_ok = clip_ok && g_dav.ok(m_dav) && g_bridge.ok(m_lat) && g_wave.ok(m_raw) && g_wave.ok(m_fin) && dgain <= 0.01 &&
                  (plan.single() || g_overlap.ok(m_ovl));
        w_dav.worst_of(m_dav);
        w_lat.worst_of(m_lat);
        w_raw.worst_of(m_raw);
        w_fin.worst_of(m_fin);
        w_ovl.worst_of(m_ovl);
        pass = pass && clip_ok;
        std::cout << clip << " steps=" << steps << " n=" << n << " windows=" << plan.starts.size() << " frames=" << plan.frames
                  << " ms=" << std::setprecision(6) << ms << " (load+encode/mert/bridge/decode phases " << std::setprecision(5);
        for (auto & [ph, t] : phase_ms) {
            std::cout << " " << t;
        }
        std::cout << ")\n  starts " << (starts_ok ? "match" : "DIFFER") << "  dav " << m_dav << "\n  overlap " << m_ovl
                  << "\n  stitched latents " << m_lat << "\n  raw decoded " << m_raw << "\n  final " << m_fin
                  << "\n  gain_db engine=" << std::setprecision(6) << r.gain_db << " torch=" << gain_ref << " delta=" << dgain
                  << " lufs_in=" << r.lufs_in << " lufs_out=" << r.lufs_out << (clip_ok ? "  PASS" : "  FAIL") << "\n";
    }
    std::cout << "worst over " << clips.size() << " clips: dav " << w_dav << "\n  overlap " << w_ovl << "\n  stitched latents "
              << w_lat << "\n  raw decoded " << w_raw << "\n  final " << w_fin << "\n  gain delta " << worst_gain << " dB, "
              << std::setprecision(6) << total_ms / 1000.0 << " s total\n";
    std::cout << "store after the runs: modules=" << store_gpu_module_count(store) << " vram=" << store_vram_bytes(store) << "\n";
    store_free(store);
    std::cout << (pass ? "PASS" : "FAIL") << " mm3 refiner end-to-end parity, " << clips.size() << " clips\n";
    return pass ? 0 : 1;
}

// ── behave ──────────────────────────────────────────────────────────────────

struct Tests {
    int pass = 0, fail = 0;
    void check(bool ok, const std::string & what) {
        (ok ? pass : fail)++;
        std::cout << (ok ? "  PASS " : "  FAIL ") << what << "\n";
    }
};

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

static int cmd_behave(int argc, char ** argv) {
    auto args = parse_args(argc, argv);
    if (!args.count("--fixtures") || !args.count("--models") || !args.count("--scratch")) {
        std::cout << "mm3-refiner-probe behave --fixtures DIR --models DIR --scratch DIR\n";
        return 2;
    }
    const fs::path sdir  = fs::path(args.at("--fixtures")) / "boombap-s1001" / "steps-8";
    Npy            clip  = load_npy(fs::path(args.at("--fixtures")) / "boombap-s1001" / "input.npy");
    const int64_t  total = clip.shape[1];
    const Mm3RefinerPaths good = paths_in(args.at("--models"));
    const Mm3StyleFn      style = fixture_style(sdir);
    Tests                 t;

    auto crop = [&](int64_t start, int64_t n) {  // planar [2][n] from the clip
        std::vector<float> v((size_t) (2 * n));
        for (int ch = 0; ch < 2; ch++) {
            std::memcpy(v.data() + ch * n, clip.data.data() + ch * total + start, (size_t) n * sizeof(float));
        }
        return v;
    };
    auto run = [&](ModelStore * store, const Mm3RefinerPaths & paths, const Mm3RefinerParams & p, const std::vector<float> & in,
                   int sr, const Mm3StyleFn & sf, const Mm3RefinerHooks & hooks, Mm3RefinerResult * r, std::string * err) {
        return mm3_refiner_run(store, paths, p, in.data(), (int64_t) in.size() / 2, sr, sf, hooks, r, err);
    };

    ModelStore * strict = store_create(EVICT_STRICT);
    const std::vector<float> in8 = crop(1000, 8 * 44100);  // 8 s, not hop aligned at the start of nothing: 352800 = 689.06 hops

    std::cout << "bypass:\n";
    {
        Mm3RefinerParams p;
        p.mix = 0.0f;
        Mm3RefinerResult r;
        std::string      err;
        Mm3RefinerPaths  bogus{ "x/enc", "x/mert", "x/refiner", "x/voc" };
        const bool       ok = run(strict, bogus, p, in8, 44100, nullptr, {}, &r, &err);
        t.check(ok && r.audio == in8 && store_gpu_module_count(strict) == 0, "mix=0 returns the input bit for bit and loads nothing (missing files, no style)");
        std::vector<float> in48;
        {
            int     n48 = 0;
            float * rs  = audio_resample(in8.data(), (int) (in8.size() / 2), 44100, 48000, 2, &n48);
            in48.assign(rs, rs + 2 * (int64_t) n48);
            free(rs);
        }
        Mm3RefinerResult r48;
        const bool       ok48 = run(strict, bogus, p, in48, 48000, nullptr, {}, &r48, &err);
        t.check(ok48 && r48.audio == in48 && r48.sample_rate == 48000 && r48.n == (int64_t) in48.size() / 2, "mix=0 at 48 kHz: same samples, rate and length back");
    }

    std::cout << "bounds:\n";
    {
        Mm3RefinerPlan plan;
        std::string    err;
        auto           code = [&](int64_t total_samples, Mm3RefinerParams p) { return mm3_refiner_plan(total_samples, p, &plan, &err); };
        Mm3RefinerParams p;
        t.check(code(441000, p) == 0 && plan.window_samples == 1323008 && plan.single(), "default 30 s window rounds to 2584 hops (1,323,008 samples)");
        p = {};
        t.check(code(2000000, p) == 0 && plan.overlap_samples == 88064 && plan.starts == std::vector<int64_t>({ 0, 676864 }), "default 2 s overlap rounds to 172 hops; 2,000,000 samples: starts 0 and the hop-floored last window 676,864");
        t.check(code(3000000, p) == 0 && plan.starts == std::vector<int64_t>({ 0, 1234944, 1676800 }), "3,000,000 samples: stride 1,234,944 then the hop-floored last window 1,676,800");
        p       = {};
        p.steps = 0;
        t.check(code(441000, p) == 400, "steps 0 rejected");
        p.steps = 65;
        t.check(code(441000, p) == 400, "steps 65 rejected");
        p        = {};
        p.window_seconds = -1;
        t.check(code(441000, p) == 400, "negative window rejected");
        p                 = {};
        p.overlap_seconds = 30;
        t.check(code(5000000, p) == 400, "overlap equal to the window rejected");
        p     = {};
        p.mix = 1.5f;
        t.check(code(441000, p) == 400, "mix 1.5 rejected");
        t.check(code(22000, {}) == 400, "audio shorter than 0.5 s rejected");
        p                = {};
        p.window_seconds = 0;
        t.check(code(5168 * 512, p) == 0 && plan.single(), "direct mode accepts 5168 frames (60 s)");
        t.check(code(5169 * 512, p) == 413, "direct mode refuses 5169 frames with 413");
        // A large nonzero window picks the same dense attention, so it is bounded the same way.
        const int64_t s61 = 5255 * 512;  // the 61 s clip
        p                = {};
        p.window_seconds = 120;
        t.check(code(s61, p) == 413, "window=120 on a 61 s clip: 413, like window=0");
        p.window_seconds = 120;
        t.check(code(441000 / 512 * 512, p) == 0 && plan.single(), "window=120 on a 10 s clip is fine (the sequence is the clip)");
        p.window_seconds = 61;
        t.check(code(10 * 5255 * 512, p) == 413, "window=61 on a long clip: 413 (window itself past the limit)");
        p.window_seconds = 60;
        t.check(code(2 * 5168 * 512, p) == 0 && !plan.single() && plan.window_samples == 5168 * 512, "window=60 rounds to 5168 hops, exactly the limit: accepted on a long clip");
        p                = {};
        p.window_seconds = 1e300;
        t.check(code(441000 / 512 * 512, p) == 0 && plan.single(), "window=1e300 on a 10 s clip: no overflow, one window");
        t.check(code(s61, p) == 413, "window=1e300 on a 61 s clip: 413");
        p.overlap_seconds = 1e300;
        t.check(code(441000 / 512 * 512, p) == 400, "window=1e300 with overlap=1e300: 400 (overlap not shorter)");
        p                 = {};
        p.overlap_seconds = 1e300;
        t.check(code(5000000, p) == 400, "overlap=1e300: 400, no overflow");
        p                = {};
        p.window_seconds = 1e308;
        t.check(code(s61, p) == 413, "window=1e308 (hop count overflows a double): 413");
        p                 = {};
        p.window_seconds  = 4.0;
        p.overlap_seconds = 1.0;
        const int64_t tot = 12 * 44100 / 512 * 512;
        bool hops = true;
        for (int64_t s : plan.starts) {
            hops = hops && s % 512 == 0;
        }
        t.check(code(tot, p) == 0 && plan.starts.size() >= 4 && hops && plan.starts.back() + plan.window_samples <= tot,
                "4 s windows with 1 s overlap: starts are hop multiples and the last window ends inside the clip");
    }

    std::cout << "seams (overlap-add on constant windows):\n";
    {
        Mm3RefinerParams p;
        p.window_seconds  = 4.0;
        p.overlap_seconds = 1.0;
        Mm3RefinerPlan plan;
        std::string    err;
        const int64_t  tot = 689 * 512;
        require(mm3_refiner_plan(tot, p, &plan, &err) == 0, err);
        Mm3RefinerStitch   st(plan);
        const int64_t      L = plan.window_samples / 512;
        for (size_t w = 0; w < plan.starts.size(); w++) {
            std::vector<float> gen((size_t) (128 * L), float(w + 1));
            st.add(w, gen.data(), L);
        }
        const bool ok = st.finish(&err);
        bool       monotone = ok, ends = ok;
        const float * c0 = st.latents.data();  // channel 0
        for (int64_t i = 1; ok && i < plan.frames; i++) {
            monotone = monotone && c0[i] >= c0[i - 1] - 1e-6f;
        }
        ends = ok && c0[0] == 1.0f && std::abs(c0[plan.frames - 1] - float(plan.starts.size())) < 1e-6f;
        t.check(ok && monotone && ends, "constant windows 1..N stitch to a monotone ramp, no gaps, first window untouched at the start (" + std::to_string(plan.starts.size()) + " windows)");
        Mm3RefinerStitch gap(plan);
        t.check(!gap.finish(&err) && err.find("uncovered") != std::string::npos, "a missing window is reported as uncovered frames");
    }

    std::cout << "errors:\n";
    {
        Mm3RefinerParams p;
        Mm3RefinerResult r;
        std::string      err;
        t.check(!run(strict, good, p, in8, 44100, nullptr, {}, &r, &err) && err.find("style") != std::string::npos && store_gpu_module_count(strict) == 0,
                "no style source: explicit error before any model loads (" + err + ")");
        const char * names[4] = { "DAV encoder", "MERT", "refiner", "decoder" };
        for (int i = 0; i < 4; i++) {
            Mm3RefinerPaths bad = good;
            std::string *   slot[4] = { &bad.enc, &bad.mert, &bad.refiner, &bad.voc };
            *slot[i]                = (fs::path(args.at("--scratch")) / "does-not-exist.gguf").string();
            const bool ok           = run(strict, bad, p, in8, 44100, style, {}, &r, &err);
            t.check(!ok && err.find(names[i]) == 0 && err.find("does-not-exist") != std::string::npos && store_gpu_module_count(strict) == 0,
                    std::string("missing ") + names[i] + " file named in the error, nothing left resident (" + err + ")");
        }
        // corrupt: the first 64 KB of the refiner and of the vocoder
        for (int which = 0; which < 2; which++) {
            const std::string src = which == 0 ? good.refiner : good.voc;
            const fs::path    cut = fs::path(args.at("--scratch")) / (which == 0 ? "corrupt-refiner.gguf" : "corrupt-voc.gguf");
            {
                std::ifstream in(src, std::ios::binary);
                std::string   head(65536, '\0');
                in.read(head.data(), 65536);
                std::ofstream(cut, std::ios::binary).write(head.data(), 65536);
            }
            Mm3RefinerPaths bad = good;
            (which == 0 ? bad.refiner : bad.voc) = cut.string();
            const bool ok = run(strict, bad, p, in8, 44100, style, {}, &r, &err);
            t.check(!ok && !err.empty() && store_gpu_module_count(strict) == 0, std::string("truncated ") + (which == 0 ? "refiner" : "vocoder") + " GGUF fails cleanly (" + err + ")");
        }
    }

    std::cout << "cancel:\n";
    {
        Mm3RefinerParams p;
        Mm3RefinerResult r;
        std::string      err;
        int              forwards = 0;
        Mm3RefinerHooks  hooks;
        hooks.progress  = [&](Mm3RefinerPhase ph, int, int) { forwards += ph == Mm3RefinerPhase::BRIDGE; };
        hooks.cancelled = [&]() { return forwards >= 6; };
        const auto t0   = std::chrono::steady_clock::now();
        const bool ok   = run(strict, good, p, in8, 44100, style, hooks, &r, &err);
        const double ms = ms_since(t0);
        t.check(!ok && err == "cancelled" && store_gpu_module_count(strict) == 0 && store_vram_bytes(strict) == 0 && forwards == 6,
                "cancel inside the bridge stops after step 6 of 32 and releases the DiT (" + std::to_string(forwards) + " forwards, " + std::to_string(ms) + " ms)");
        hooks.cancelled = [&]() { return true; };
        t.check(!run(strict, good, p, in8, 44100, style, hooks, &r, &err) && err == "cancelled" && store_gpu_module_count(strict) == 0, "cancel before the first phase loads nothing");
        Mm3RefinerResult again;
        t.check(run(strict, good, p, in8, 44100, style, {}, &again, &err) && all_finite(again.audio), "a run after a cancel completes");
    }

    std::cout << "audio shapes:\n";
    {
        Mm3RefinerParams p;
        p.steps = 8;
        std::string err;
        // silence, 2 s
        {
            std::vector<float> zeros((size_t) (2 * 88200), 0.0f);
            Mm3RefinerResult   r;
            const bool         ok = run(strict, good, p, zeros, 44100, style, {}, &r, &err);
            float              peak = 0;
            for (float x : r.audio) {
                peak = std::max(peak, std::abs(x));
            }
            t.check(ok && all_finite(r.audio) && r.audio.size() == zeros.size() && peak <= 1.0f && r.gain_db == 0.0,
                    "silence: finite output, same length, no loudness gain applied (peak " + std::to_string(peak) + ")");
        }
        // non-hop-aligned tail, stereo and dual mono
        const int64_t      odd = 60001;
        std::vector<float> st  = crop(5000, odd);
        Mm3RefinerResult   rs;
        t.check(run(strict, good, p, st, 44100, style, {}, &rs, &err) && rs.n == odd && rs.audio.size() == st.size() && all_finite(rs.audio),
                "stereo, 60,001 samples (not a hop multiple): length preserved, finite");
        std::vector<float> mono((size_t) (2 * odd));
        std::memcpy(mono.data(), st.data(), (size_t) odd * 4);
        std::memcpy(mono.data() + odd, st.data(), (size_t) odd * 4);
        Mm3RefinerResult rm;
        const bool       okm = run(strict, good, p, mono, 44100, style, {}, &rm, &err);
        double           dlr = 0;
        for (int64_t i = 0; okm && i < odd; i++) {
            dlr = std::max(dlr, (double) std::abs(rm.audio[(size_t) i] - rm.audio[(size_t) (odd + i)]));
        }
        t.check(okm && dlr == 0.0, "mono input (duplicated channels) comes back as identical channels, max L-R " + std::to_string(dlr));
        // 48 kHz in and out, mix endpoints and a blend
        std::vector<float> in48;
        int                n48 = 0;
        {
            float * rs48 = audio_resample(in8.data(), (int) (in8.size() / 2), 44100, 48000, 2, &n48);
            in48.assign(rs48, rs48 + 2 * (int64_t) n48);
            free(rs48);
        }
        Mm3RefinerResult w48, h48;
        p.mix = 1.0f;
        const bool ok1 = run(strict, good, p, in48, 48000, style, {}, &w48, &err);
        p.mix          = 0.5f;
        const bool ok2 = run(strict, good, p, in48, 48000, style, {}, &h48, &err);
        double     dev = 0;
        for (size_t i = 0; ok1 && ok2 && i < in48.size(); i++) {
            dev = std::max(dev, (double) std::abs(h48.audio[i] - (0.5f * w48.audio[i] + 0.5f * in48[i])));
        }
        t.check(ok1 && ok2 && w48.sample_rate == 48000 && w48.n == n48 && w48.audio.size() == in48.size() && all_finite(w48.audio),
                "48 kHz input: resampled in and out, rate " + std::to_string(w48.sample_rate) + " and length " + std::to_string(w48.n) + " back");
        t.check(ok1 && ok2 && dev < 2e-7, "mix=0.5 equals 0.5*refined + 0.5*input (max deviation " + std::to_string(dev) + ")");
    }

    std::cout << "residency:\n";
    {
        Mm3RefinerParams p;
        p.steps = 4;
        Mm3RefinerResult r;
        std::string      err;
        t.check(run(strict, good, p, in8, 44100, style, {}, &r, &err) && store_gpu_module_count(strict) == 0 && store_vram_bytes(strict) == 0,
                "EVICT_STRICT: all four modules unloaded after a successful run");
        ModelStore * keep = store_create(EVICT_NEVER);
        const bool   ran_keep = run(keep, good, p, in8, 44100, style, {}, &r, &err);
        const size_t resident = store_vram_bytes(keep);
        t.check(ran_keep && store_gpu_module_count(keep) == 4 && resident > 0,
                "EVICT_NEVER: the four modules stay resident (" + std::to_string(resident >> 20) + " MB)");
        Mm3RefinerResult r2;
        t.check(run(keep, good, p, in8, 44100, style, {}, &r2, &err) && store_vram_bytes(keep) == resident && r2.audio == r.audio,
                "a second run reuses them: same residency, identical output");
        int       still = 0;
        const int freed = store_evict_all(keep, &still);
        t.check(freed == 4 && still == 0 && store_vram_bytes(keep) == 0, "store_evict_all frees all four");
        t.check(run(keep, good, p, in8, 44100, style, {}, &r2, &err) && r2.audio == r.audio, "reload after eviction reproduces the output");
        store_free(keep);
    }
    store_free(strict);
    std::cout << (t.fail ? "FAIL" : "PASS") << " mm3 refiner behaviour, " << t.pass << " passed, " << t.fail << " failed\n";
    return t.fail ? 1 : 0;
}

// ── direct ──────────────────────────────────────────────────────────────────

static int cmd_direct(int argc, char ** argv) {
    auto args = parse_args(argc, argv);
    if (!args.count("--fixtures") || !args.count("--models")) {
        std::cout << "mm3-refiner-probe direct --fixtures DIR --models DIR [--seconds 60] [--steps 8]\n";
        return 2;
    }
    const double  seconds = args.count("--seconds") ? std::stod(args.at("--seconds")) : 60.0;
    const int     steps   = args.count("--steps") ? std::stoi(args.at("--steps")) : 8;
    const int64_t n       = (int64_t) (seconds * 44100.0);
    Npy           a = load_npy(fs::path(args.at("--fixtures")) / "boombap-s1001" / "input.npy");
    Npy           b = load_npy(fs::path(args.at("--fixtures")) / "boombap-s1002" / "input.npy");
    std::vector<float> in((size_t) (2 * n));
    for (int ch = 0; ch < 2; ch++) {  // boombap-s1001 then s1002, cut to length
        for (int64_t i = 0; i < n; i++) {
            const int64_t k = i < a.shape[1] ? i : i - a.shape[1];
            in[(size_t) (ch * n + i)] = i < a.shape[1] ? a.data[(size_t) (ch * a.shape[1] + k)] : b.data[(size_t) (ch * b.shape[1] + std::min<int64_t>(k, b.shape[1] - 1))];
        }
    }
    ModelStore *     store = store_create(EVICT_STRICT);
    Mm3RefinerParams p;
    p.steps          = steps;
    p.window_seconds = 0;
    Mm3RefinerResult r;
    std::string      err;
    Mm3StyleFn       style = fixture_style(fs::path(args.at("--fixtures")) / "boombap-s1001" / "steps-8");
    const auto       t0    = std::chrono::steady_clock::now();
    const bool       ok    = mm3_refiner_run(store, paths_in(args.at("--models")), p, in.data(), n, 44100, [&](int, const float * pl, int64_t nn, float * o, std::string * e) { return style(0, pl, nn, o, e); }, {}, &r, &err);
    std::cout << "direct " << seconds << " s, " << (n + 511) / 512 << " frames, steps " << steps << ": " << (ok ? "ok" : "FAILED " + err) << ", "
              << ms_since(t0) / 1000.0 << " s, finite=" << (ok && all_finite(r.audio)) << ", lufs_in=" << r.lufs_in << " lufs_out=" << r.lufs_out << "\n";
    store_free(store);
    return ok && all_finite(r.audio) ? 0 : 1;
}

int main(int argc, char ** argv) {
    try {
        const std::string cmd = argc > 1 ? argv[1] : "";
        if (cmd == "parity") {
            return cmd_parity(argc, argv);
        }
        if (cmd == "behave") {
            return cmd_behave(argc, argv);
        }
        if (cmd == "direct") {
            return cmd_direct(argc, argv);
        }
        std::cout << "mm3-refiner-probe parity|behave|direct ... (run a command with no options for its usage)\n";
        return 2;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
