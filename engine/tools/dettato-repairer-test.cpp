// Validate the native Dettato evidence repairer against a slice 1 repairer capture (capture.py --repairer).
// Part of HOT-Step CPP. MIT license.
//
// usage: dettato-repairer-test WEIGHTS_DIR CAPTURE/cpu-repair
//   WEIGHTS_DIR holds dettato/ (the transcriber's decode thresholds) and dettato-evidence/.
// Stages, each against the capture:
//   decode    notes decoded from the captured all-window maps = result.repair.input_notes
//   input     the sorted repairer notes = repair.k.*
//   evidence  rows from the captured maps and repair.hidden vs repair.ev
//   fed       the network and decode on the captured evidence: P(keep), moved onsets and ends, insertion proposals,
//             and the repaired notes
//   chain     the same from the port's own evidence: repaired notes against result.repair.output
// Gates: tools/dettato-reference/tolerances.json, "repairer". Exit 1 when any gate fails.

#include "dettato/repairer.h"

#include <filesystem>
#include <fstream>

namespace {

namespace fs = std::filesystem;
namespace rp = dettato::repairer;
namespace tr = dettato::transcriber;

template <class T> std::vector<T> read_binary(const fs::path & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    const auto    length = f.tellg();
    if (length < 0 || size_t(length) % sizeof(T)) throw std::runtime_error("cannot read " + path.string());
    std::vector<T> data(size_t(length) / sizeof(T));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(data.data()), length)) throw std::runtime_error("cannot read " + path.string());
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
    yyjson_arr_foreach(arr, i, n, x) {
        need(yyjson_is_num(x), "oracle array holds a non-number");
        v.push_back(yyjson_get_num(x));
    }
    return v;
}

bool gate(const char * what, bool ok) {
    std::printf("  %-44s %s\n", what, ok ? "PASS" : "FAIL");
    return ok;
}

struct NoteGate {
    double f1 = 0, median_on_ms = 0, off_within = 0;
    size_t matched = 0, a = 0, b = 0;
    bool   ok = false;
};

// compare.py match(): one-to-one nearest onset per pitch within 10 ms; then the transcriber's note gates
NoteGate note_gate(const std::vector<double> & a_on, const std::vector<double> & a_off, const std::vector<int> & a_p,
                   const std::vector<double> & b_on, const std::vector<double> & b_off, const std::vector<int> & b_p) {
    constexpr double tol = 0.010;
    std::map<int, std::vector<int>> ka, kb;
    for (size_t i = 0; i < a_p.size(); i++) ka[a_p[i]].push_back((int) i);
    for (size_t i = 0; i < b_p.size(); i++) kb[b_p[i]].push_back((int) i);
    std::vector<double> on_err, off_err;
    for (auto & [key, ia] : ka) {
        auto it = kb.find(key);
        if (it == kb.end()) continue;
        std::vector<int> ib = it->second;
        std::stable_sort(ia.begin(), ia.end(), [&](int x, int y) { return a_on[x] < a_on[y]; });
        std::stable_sort(ib.begin(), ib.end(), [&](int x, int y) { return b_on[x] < b_on[y]; });
        std::vector<double> bo;
        for (int j : ib) bo.push_back(b_on[j]);
        std::vector<char> used(ib.size(), 0);
        for (int i : ia) {
            const int j = (int) (std::lower_bound(bo.begin(), bo.end(), a_on[i]) - bo.begin());
            int       best = -1;
            double    bd   = tol + 1e-9;
            for (int c : { j - 1, j, j + 1 })
                if (c >= 0 && c < (int) ib.size() && !used[c] && std::fabs(bo[c] - a_on[i]) < bd) {
                    best = c;
                    bd   = std::fabs(bo[c] - a_on[i]);
                }
            if (best < 0) continue;
            used[best] = 1;
            on_err.push_back(bd);
            off_err.push_back(std::fabs(b_off[ib[best]] - a_off[i]));
        }
    }
    NoteGate g;
    g.matched = on_err.size();
    g.a       = a_p.size();
    g.b       = b_p.size();
    g.f1      = g.a + g.b == 0 ? 1.0 : 2.0 * g.matched / (double) (g.a + g.b);
    if (!on_err.empty()) {
        std::sort(on_err.begin(), on_err.end());
        const size_t h = on_err.size() / 2;
        g.median_on_ms = 1000 * (on_err.size() % 2 ? on_err[h] : 0.5 * (on_err[h - 1] + on_err[h]));
        size_t w       = 0;
        for (double e : off_err) w += e <= 0.020;
        g.off_within = (double) w / off_err.size();
    } else {
        g.off_within = 1.0;
    }
    g.ok = g.f1 >= 0.99 && g.median_on_ms <= 1.0 && g.off_within >= 0.99;
    return g;
}

bool check_output(const char * label, const rp::Repaired & r, yyjson_val * out) {
    const auto on = numbers(yyjson_obj_get(out, "onset")), off = numbers(yyjson_obj_get(out, "offset")),
               pd = numbers(yyjson_obj_get(out, "pitch"));
    need(off.size() == on.size() && pd.size() == on.size(), "output columns differ in length");
    std::vector<int> p;
    for (double x : pd) p.push_back((int) x);
    size_t           exact = 0;
    if (on.size() == r.onset.size())
        for (size_t i = 0; i < on.size(); i++)
            exact += on[i] == r.onset[i] && off[i] == r.offset[i] && p[i] == r.pitch[i];
    const NoteGate g = note_gate(on, off, p, r.onset, r.offset, r.pitch);
    std::printf("%s: %d transcribed, %d kept, %d dropped, %d inserted; oracle %zu notes; identical rows %zu\n", label, r.transcribed,
                r.kept, r.dropped, r.n_inserted, on.size(), exact);
    std::printf("  notes F1 %.5f (%zu/%zu/%zu matched/port/oracle), median onset %.3f ms, offsets within 20 ms %.4f\n", g.f1,
                g.matched, g.b, g.a, g.median_on_ms, g.off_within);
    return gate("notes: F1 >= 0.99, onset <= 1 ms, offsets 99%", g.ok);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: dettato-repairer-test WEIGHTS_DIR CAPTURE/cpu-repair\n");
        return 2;
    }
    try {
        const fs::path weights = argv[1], cap = argv[2];
        std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
            yyjson_read_file((cap / "result.json").string().c_str(), 0, nullptr, nullptr), yyjson_doc_free);
        if (!doc) throw std::runtime_error("cannot read oracle result.json");
        yyjson_val * rep = yyjson_obj_get(yyjson_doc_get_root(doc.get()), "repair");
        if (!yyjson_is_obj(rep)) throw std::runtime_error("not a repairer capture (result.repair missing)");
        bool ok = true;

        // ---- decode from the captured maps
        tr::Model tm;
        if (!tr::read_config(tm, (weights / "dettato" / "config.json").string())) return 1;
        std::vector<float> maps[tr::N_HEADS_OUT];
        for (int h = 0; h < tr::N_HEADS_OUT; h++) maps[h] = read_binary<float>(cap / ("maps." + std::string(tr::HEAD_NAMES[h]) + ".bin"));
        for (auto & mp : maps) need(!mp.empty() && mp.size() == maps[0].size() && mp.size() % tr::N_KEYS == 0, "maps.* are not six T x 88 arrays");
        const int  T      = (int) (maps[0].size() / tr::N_KEYS);
        const auto hidden = read_binary<ggml_fp16_t>(cap / "repair.hidden.bin");
        if (hidden.size() != (size_t) T * tr::N_KEYS * rp::C_H) throw std::runtime_error("repair.hidden has the wrong size");
        const auto notes = tr::decode_notes(maps, T, tm.th);
        {
            yyjson_val * in = yyjson_obj_get(rep, "input_notes");
            const auto   on = numbers(yyjson_obj_get(in, "onset")), off = numbers(yyjson_obj_get(in, "offset")),
                       p = numbers(yyjson_obj_get(in, "pitch")), cf = numbers(yyjson_obj_get(in, "conf")),
                       sl = numbers(yyjson_obj_get(in, "slot"));
            need(off.size() == on.size() && p.size() == on.size() && cf.size() == on.size() && sl.size() == on.size(),
                 "input_notes columns differ in length");
            size_t same = 0;
            if (on.size() == notes.size())
                for (size_t i = 0; i < on.size(); i++)
                    same += std::fabs(on[i] - notes[i].onset) < 1e-9 && std::fabs(off[i] - notes[i].offset) < 1e-9 &&
                            p[i] == notes[i].pitch && sl[i] == notes[i].slot && (float) cf[i] == notes[i].conf;
            std::printf("decode: %zu notes, oracle %zu, identical %zu\n", notes.size(), on.size(), same);
            ok = gate("decode: the oracle's input notes exactly", same == on.size() && same == notes.size()) && ok;
        }

        // ---- sorted input notes
        std::vector<int> order;
        const auto       k = rp::input_notes(notes, order);
        {
            const auto kp = read_binary<uint8_t>(cap / "repair.k.pitch.bin"), kv = read_binary<uint8_t>(cap / "repair.k.vel.bin");
            const auto kon = read_binary<float>(cap / "repair.k.on.bin"), koff = read_binary<float>(cap / "repair.k.off.bin");
            need(kv.size() == kp.size() && kon.size() == kp.size() && koff.size() == kp.size(), "repair.k.* differ in length");
            size_t     same = 0;
            if (kp.size() == k.size())
                for (size_t i = 0; i < k.size(); i++) same += kp[i] == k[i].pitch && kv[i] == k[i].vel && kon[i] == k[i].on && koff[i] == k[i].off;
            std::printf("input: %zu sorted notes, identical %zu of %zu\n", k.size(), same, kp.size());
            ok = gate("input: repair.k.* exactly", same == kp.size() && same == k.size()) && ok;
        }

        // ---- evidence
        const auto ref_ev = read_binary<float>(cap / "repair.ev.bin");
        if (ref_ev.size() != notes.size() * rp::N_EV) throw std::runtime_error("repair.ev has the wrong size");
        double     t0 = tr::now_ms();
        const auto ev = rp::evidence(maps, T, hidden.data(), notes);
        const double ms_ev = tr::now_ms() - t0;
        {
            float  map_gap = 0;
            size_t flips = 0, codes = 0;
            int    max_code = 0;
            bool   finite = true;
            for (size_t i = 0; i < notes.size(); i++)
                for (int j = 0; j < rp::N_EV; j++) {
                    const float a = ev[i * rp::N_EV + j], b = ref_ev[i * rp::N_EV + j];
                    finite = finite && std::isfinite(a);
                    if (j < rp::N_MAPF + 2) {
                        map_gap = std::max(map_gap, std::fabs(a - b));
                    } else {
                        const int d = (int) std::lround(std::fabs(a - b) * rp::Q);
                        codes++;
                        flips += d != 0;
                        max_code = std::max(max_code, d);
                    }
                }
            std::printf("evidence: %.1f ms; map numbers max abs %.3g; hidden codes off by one %zu of %zu (%.4f%%), max %d\n", ms_ev,
                        map_gap, flips, codes, 100.0 * flips / std::max<size_t>(codes, 1), max_code);
            ok = gate("evidence: map numbers within 1e-3", finite && map_gap < 1e-3f) && ok;
            ok = gate("evidence: hidden codes within +-1, <= 0.1% off", max_code <= 1 && flips * 1000 <= codes) && ok;
        }

        // ---- fed: the network on the captured evidence
        rp::Model m;
        if (!rp::load_model(m, (weights / "dettato-evidence").string())) return 1;
        std::vector<float> evs(k.size() * rp::N_EV);
        for (size_t i = 0; i < k.size(); i++) std::copy_n(&ref_ev[(size_t) order[i] * rp::N_EV], rp::N_EV, &evs[i * rp::N_EV]);
        t0                = tr::now_ms();
        const auto e      = rp::edits(m, k, evs.data());
        const double ms_e = tr::now_ms() - t0;
        {
            const auto ran = read_binary<uint8_t>(cap / "repair.window_ran.bin");
            const auto pk = read_binary<float>(cap / "repair.pk.bin"), on_r = read_binary<float>(cap / "repair.on_r.bin"),
                       off_r = read_binary<float>(cap / "repair.off_r.bin");
            need(ran.size() == k.size() && pk.size() == k.size() && on_r.size() == k.size() && off_r.size() == k.size(),
                 "repair.window_ran / pk / on_r / off_r do not hold one value per note");
            size_t same_ran = 0, keep_flip = 0, on_moved = 0, off_moved = 0;
            float  pk_gap = 0;
            double move_ms = 0;
            bool   finite = true;
            for (size_t i = 0; i < k.size(); i++) {
                const bool r = !std::isnan(e.pk[i]);
                same_ran += r == (ran[i] != 0);
                if (!r || !ran[i]) continue;
                finite = finite && std::isfinite(e.pk[i]) && std::isfinite(e.on_r[i]) && std::isfinite(e.off_r[i]) &&
                         std::isfinite(pk[i]) && std::isfinite(on_r[i]) && std::isfinite(off_r[i]);
                pk_gap = std::max(pk_gap, (float) std::fabs(e.pk[i] - pk[i]));
                keep_flip += (e.pk[i] >= m.keep_th) != (pk[i] >= m.keep_th);
                on_moved += std::fabs(e.on_r[i] - on_r[i]) > 1e-4;   // a different 5 ms onset bin
                off_moved += std::fabs(e.off_r[i] - off_r[i]) > 1e-4;
                move_ms = std::max({ move_ms, 1000 * std::fabs(e.on_r[i] - on_r[i]), 1000 * std::fabs(e.off_r[i] - off_r[i]) });
            }
            std::printf("fed: %d windows, %.1f ms; P(keep) max abs %.3g; keep decisions flipped %zu; onset bin moved %zu, end moved %zu (of %zu), largest move %.2f ms\n",
                        e.windows_run, ms_e, pk_gap, keep_flip, on_moved, off_moved, k.size(), move_ms);
            ok = gate("fed: same windows ran", same_ran == k.size() && ran.size() == k.size()) && ok;
            ok = gate("fed: P(keep) within 1e-3", finite && pk_gap < 1e-3f) && ok;
            ok = gate("fed: onsets and ends in the same bin, >= 99%", (on_moved + off_moved) * 100 <= 2 * k.size()) && ok;

            // insertion proposals, in the same order (note by note, slot by slot)
            yyjson_val * ins = yyjson_obj_get(rep, "insertions");
            size_t       n_ref = yyjson_arr_size(ins), same = 0;
            float        s_gap = 0;
            bool         s_finite = true;  // a NaN score would slip past std::max, so both sides must be finite
            need(yyjson_is_arr(ins), "result.repair.insertions is not an array");
            if (n_ref == e.ins.size())
                for (size_t i = 0; i < n_ref; i++) {
                    const auto x = numbers(yyjson_arr_get(ins, i));
                    need(x.size() == 4, "an insertion proposal is not (onset, end, pitch, score)");
                    const auto & y = e.ins[i];
                    s_finite = s_finite && std::isfinite(x[3]) && std::isfinite(y.score) && std::isfinite(y.on) && std::isfinite(y.off);
                    same += std::fabs(x[0] - y.on) < 1e-4 && std::fabs(x[1] - y.off) < 1e-4 && (int) x[2] == y.pitch;
                    s_gap = std::max(s_gap, (float) std::fabs(x[3] - y.score));
                }
            std::printf("fed: %zu insertion proposals, oracle %zu, same onset/end/pitch %zu, score max abs %.3g\n", e.ins.size(), n_ref,
                        same, s_gap);
            ok = gate("fed: insertion scores within 1e-3", n_ref == e.ins.size() && s_finite && s_gap < 1e-3f) && ok;
            ok = gate("fed: insertion placements, >= 99% the same", same * 100 >= 99 * n_ref) && ok;
        }
        ok = check_output("fed", rp::apply(m, k, e), yyjson_obj_get(rep, "output")) && ok;

        // ---- chain: the port's own evidence
        std::vector<float> evc(k.size() * rp::N_EV);
        for (size_t i = 0; i < k.size(); i++) std::copy_n(&ev[(size_t) order[i] * rp::N_EV], rp::N_EV, &evc[i * rp::N_EV]);
        ok = check_output("chain", rp::apply(m, k, rp::edits(m, k, evc.data())), yyjson_obj_get(rep, "output")) && ok;

        rp::free_model(m);
        std::printf("%s\n", ok ? "REPAIRER PASS" : "REPAIRER FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "dettato-repairer-test: %s\n", ex.what());
        return 1;
    }
}
