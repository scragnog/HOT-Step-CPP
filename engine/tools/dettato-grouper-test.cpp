// Validate the native Dettato grouper against a slice 1 capture and its exact-PCEN reference.
// Part of HOT-Step CPP. MIT license.
//
// usage: dettato-grouper-test WEIGHTS_DIR FIXTURE_DIR
//   WEIGHTS_DIR holds dettato/ (the decode thresholds) and dettato-grouper/. FIXTURE_DIR holds cpu/ (the oracle) and
//   cpu-exact/ (tools/dettato-reference/exact_grouper.py: upstream's grouper with PCEN in float64, fed cpu/'s features).
// Stages:
//   gather   note features from the captured sub-run maps and taps at the oracle's notes vs grouper.meta / feat / fsc
//   fed      the network in exact-PCEN mode on the captured features and audio vs cpu-exact: probs, emb, win.*
//   decode   that prediction through the host decode vs cpu-exact's groups (same track after best matching)
//   float32  the shipping mode (PCEN through a float32 FFT) on the same inputs vs the oracle: reported, not gated; the
//            oracle's grouping hangs on MKL's FFT rounding (exact_grouper.py writes the same numbers for upstream)
// Gates: tools/dettato-reference/tolerances.json "grouper" and "fed_tensors". Exit 1 when any gate fails.

#include "dettato/grouper.h"

#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
namespace gr = dettato::grouper;
namespace tr = dettato::transcriber;

template <class T> std::vector<T> read_binary(const fs::path & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto    length = f.tellg();
    if (length < 0 || size_t(length) % sizeof(T)) throw std::runtime_error("cannot read " + path.string());
    std::vector<T> data(size_t(length) / sizeof(T));
    f.seekg(0);
    if (length > 0 && !f.read(reinterpret_cast<char *>(data.data()), length)) throw std::runtime_error("cannot read " + path.string());
    return data;
}

void need(bool cond, const char * what) {
    if (!cond) throw std::runtime_error(std::string("capture mismatch: ") + what);
}

std::vector<double> numbers(yyjson_val * arr) {
    need(yyjson_is_arr(arr), "oracle field is not an array");
    std::vector<double> v;
    size_t              i, n;
    yyjson_val *        x;
    yyjson_arr_foreach(arr, i, n, x) v.push_back(yyjson_get_num(x));
    return v;
}

bool gate(const char * what, bool ok) {
    std::printf("  %-52s %s\n", what, ok ? "PASS" : "FAIL");
    return ok;
}

using Doc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
Doc json(const fs::path & p) {
    Doc d(yyjson_read_file(p.string().c_str(), 0, nullptr, nullptr), yyjson_doc_free);
    if (!d) throw std::runtime_error("cannot read " + p.string());
    return d;
}

std::vector<std::vector<int>> groups_of(yyjson_val * arr, std::vector<int> * inst = nullptr) {
    std::vector<std::vector<int>> g;
    size_t                        i, n;
    yyjson_val *                  item;
    yyjson_arr_foreach(arr, i, n, item) {
        if (inst) inst->push_back((int) yyjson_get_int(yyjson_arr_get(item, 0)));
        std::vector<int> ids;
        for (double v : numbers(yyjson_arr_get(item, 1))) ids.push_back((int) v);
        g.push_back(ids);
    }
    return g;
}

// share of the reference's grouped notes in the same track after best matching (replay_grouper.py), and whether
// every reference track with >= 10% of them keeps its instrument
struct Agree { double same = 1, coverage = 1; bool names = true; };
Agree agree(int n, const std::vector<std::vector<int>> & ref, const std::vector<int> & ref_inst,
            const std::vector<std::vector<int>> & got, const std::vector<int> & got_inst) {
    std::vector<int> la(n, -1), lb(n, -1);
    for (size_t i = 0; i < ref.size(); i++)
        for (int id : ref[i]) la[id] = (int) i;
    for (size_t i = 0; i < got.size(); i++)
        for (int id : got[i]) lb[id] = (int) i;
    dettato::Matrix M(std::max<size_t>(ref.size(), 1), dettato::Vec(std::max<size_t>(got.size(), 1), 0.0));
    int in_ref = 0, both = 0;
    for (int i = 0; i < n; i++) {
        in_ref += la[i] >= 0;
        if (la[i] >= 0 && lb[i] >= 0) {
            both++;
            M[la[i]][lb[i]] -= 1;
        }
    }
    Agree  a;
    double s = 0;
    std::map<int, int> mp;
    for (auto [r, c] : dettato::assignment(M)) {
        s -= M[r][c];
        mp[r] = c;
    }
    a.same     = in_ref ? s / in_ref : 1.0;
    a.coverage = in_ref ? (double) both / in_ref : 1.0;
    for (size_t i = 0; i < ref.size(); i++)
        if (in_ref && ref[i].size() >= 0.10 * in_ref) {
            auto it = mp.find((int) i);
            if (it == mp.end() || it->second >= (int) got.size() || got_inst[it->second] != ref_inst[i]) a.names = false;
        }
    return a;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dettato-grouper-test WEIGHTS_DIR FIXTURE_DIR\n");
        return 2;
    }
    try {
        const fs::path weights = argv[1], fix = argv[2], cap = fix / "cpu", ex = fix / "cpu-exact";
        Doc  res = json(cap / "result.json");
        bool ok  = true;

        // ---- gather from the captured sub-run
        tr::Model tm;
        if (!tr::read_config(tm, (weights / "dettato" / "config.json").string())) return 1;
        std::vector<float> maps[tr::N_HEADS_OUT];
        for (int h = 0; h < tr::N_HEADS_OUT; h++) maps[h] = read_binary<float>(cap / ("sub.maps." + std::string(tr::HEAD_NAMES[h]) + ".bin"));
        const int  T = (int) (maps[0].size() / tr::N_KEYS);
        const auto H = read_binary<ggml_fp16_t>(cap / "sub.tap.hidden.bin"), S = read_binary<ggml_fp16_t>(cap / "sub.tap.stem.bin");
        need(H.size() == (size_t) T * tr::N_KEYS * tr::C && S.size() == H.size(), "sub.tap.* are not T x 88 x 368");
        yyjson_val * tn = yyjson_obj_get(yyjson_doc_get_root(res.get()), "transcriber_notes");
        const auto   on = numbers(yyjson_obj_get(tn, "onset")), off = numbers(yyjson_obj_get(tn, "offset")),
                   pi = numbers(yyjson_obj_get(tn, "pitch"));
        std::vector<std::array<double, 3>> in;
        for (size_t i = 0; i < on.size(); i++) in.push_back({ on[i], std::max(off[i], on[i] + 0.03), pi[i] });
        const gr::Features f = gr::gather(maps, T, H.data(), S.data(), gr::assign_slots(in, tr::decode_notes(maps, T, tm.th)));
        const auto meta = read_binary<float>(cap / "grouper.meta.bin");
        const auto feat = read_binary<int8_t>(cap / "grouper.feat.bin");
        const auto fsc  = read_binary<ggml_fp16_t>(cap / "grouper.fsc.bin");
        {
            const bool same_n = meta.size() == f.meta.size() && feat.size() == f.feat.size() && fsc.size() == f.fsc.size();
            double     dm = 0;
            size_t     codes_off = 0, codes_far = 0, sc_off = 0;
            if (same_n) {
                for (size_t i = 0; i < meta.size(); i++) dm = std::isfinite(meta[i]) && std::isfinite(f.meta[i]) ? std::max(dm, (double) std::fabs(meta[i] - f.meta[i])) : INFINITY;
                for (size_t i = 0; i < feat.size(); i++) {
                    const int d = std::abs((int) feat[i] - (int) f.feat[i]);
                    codes_off += d > 0;
                    codes_far += d > 1;
                }
                for (size_t i = 0; i < fsc.size(); i++) {
                    const int d = std::abs((int) fsc[i] - (int) f.fsc[i]);  // same sign: float16 steps
                    sc_off += d > 1 || !std::isfinite(ggml_fp16_to_fp32(fsc[i])) || !std::isfinite(ggml_fp16_to_fp32(f.fsc[i]));
                }
            }
            std::printf("gather: %d notes (oracle %zu), meta max abs %.3g, int8 codes off %zu (beyond +-1: %zu) of %zu, scales beyond one step %zu\n",
                        f.n, meta.size() / gr::N_META, dm, codes_off, codes_far, feat.size(), sc_off);
            ok = gate("gather: same notes", same_n) && ok;
            ok = gate("gather: meta max abs < 1e-3", same_n && dm < 1e-3) && ok;
            ok = gate("gather: int8 codes within +-1", same_n && codes_far == 0) && ok;
            ok = gate("gather: scales within one float16 step", same_n && sc_off == 0) && ok;
        }
        if (meta.empty()) {
            std::printf("no notes: nothing to group\n%s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }

        // ---- the network on the oracle's own features and audio
        gr::Features fo;
        fo.n    = (int) (meta.size() / gr::N_META);
        fo.meta = meta;
        fo.feat = feat;
        fo.fsc  = fsc;
        Doc        tj = json(cap / "tensors.json");
        const bool retuned = yyjson_obj_get(yyjson_doc_get_root(tj.get()), "x16_retuned") != nullptr;
        const auto y = read_binary<float>(cap / (retuned ? "x16_retuned.bin" : "x16.bin"));
        gr::Model gm;
        if (!gr::load_model(gm, (weights / "dettato-grouper").string())) return 1;
        auto quiet = [](int, int) {};
        Doc  rex   = json(ex / "result.json");
        std::vector<int> ref_inst, ora_inst;
        const auto       ref = groups_of(yyjson_obj_get(yyjson_doc_get_root(rex.get()), "groups"), &ref_inst);
        const auto       ora = groups_of(yyjson_obj_get(yyjson_doc_get_root(res.get()), "groups"), &ora_inst);
        for (int mode = 0; mode < 2; mode++) {
            gm.exact_pcen = mode == 0;
            const double      t0 = tr::now_ms();
            gr::Prediction32  p  = gr::predict(gm, y, fo, quiet);
            const double      t1 = tr::now_ms();
            const fs::path    rd = mode == 0 ? ex : cap;
            auto gap32 = [&](const char * name, const std::vector<float> & v) {
                const auto r = read_binary<float>(rd / name);
                if (r.size() != v.size()) return (double) INFINITY;
                double d = 0;
                for (size_t i = 0; i < r.size(); i++) d = std::isfinite(r[i]) && std::isfinite(v[i]) ? std::max(d, (double) std::fabs(r[i] - v[i])) : INFINITY;
                return d;
            };
            auto gap16 = [&](const char * name, const std::vector<ggml_fp16_t> & v) {
                const auto r = read_binary<ggml_fp16_t>(rd / name);
                if (r.size() != v.size()) return (double) INFINITY;
                double d = 0;
                for (size_t i = 0; i < r.size(); i++) {
                    const float a = ggml_fp16_to_fp32(r[i]), b = ggml_fp16_to_fp32(v[i]);
                    d = std::isfinite(a) && std::isfinite(b) ? std::max(d, (double) std::fabs(a - b)) : INFINITY;
                }
                return d;
            };
            const auto wc = read_binary<int64_t>(rd / "grouper.win.count.bin");
            bool layout = wc.size() == p.win_count.size();
            for (size_t i = 0; layout && i < wc.size(); i++) layout = wc[i] == p.win_count[i];
            const double gp = gap32("grouper.probs.bin", p.probs), ge = gap32("grouper.emb.bin", p.emb), gh = gap16("grouper.h.bin", p.h);
            const double gwp = gap16("grouper.win.probs.bin", p.win_probs), gwe = gap16("grouper.win.emb.bin", p.win_emb);
            const auto groups = dettato::decode(gr::to_decode(gm, p, fo.n), [&] {
                std::vector<dettato::Note> nn;
                for (int i = 0; i < fo.n; i++) nn.push_back({ meta[(size_t) i * gr::N_META], meta[(size_t) i * gr::N_META + 1], std::nearbyint(meta[(size_t) i * gr::N_META + 2]) });
                return nn;
            }(), gm.head, gm.opt);
            std::vector<std::vector<int>> got;
            std::vector<int>              got_inst;
            for (auto & g : groups) {
                got.push_back(g.notes);
                got_inst.push_back(g.instrument);
            }
            const Agree a = agree(fo.n, mode == 0 ? ref : ora, mode == 0 ? ref_inst : ora_inst, got, got_inst);
            std::printf("%s (%s, %.0f ms: front %.0f, lanes %.0f, notes %.0f): %zu windows, probs %.3g, emb %.3g, h %.3g, win.probs %.3g, win.emb %.3g; "
                        "%zu groups, same track %.4f, coverage %.4f, names %s\n",
                        mode == 0 ? "fed, exact PCEN vs cpu-exact" : "float32 PCEN vs oracle", ggml_backend_name(gm.be), t1 - t0,
                        gm.ms_front, gm.ms_lanes, gm.ms_notes, p.win_count.size(), gp, ge, gh, gwp, gwe, groups.size(), a.same, a.coverage,
                        a.names ? "same" : "differ");
            gm.ms_front = gm.ms_lanes = gm.ms_notes = 0;
            if (mode == 0) {
                ok = gate("fed: same window layout", layout) && ok;
                ok = gate("fed: probs and win.probs max abs < 1e-3", gp < 1e-3 && gwp < 1e-3) && ok;
                ok = gate("fed: emb and win.emb max abs < 1e-3", ge < 1e-3 && gwe < 1e-3) && ok;
                ok = gate("decode: every note in the same track", a.same == 1.0 && a.coverage == 1.0) && ok;
                ok = gate("decode: same instruments", a.names && got_inst.size() == ref_inst.size()) && ok;
            }
        }
        gr::free_model(gm);
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
