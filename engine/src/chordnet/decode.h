#pragma once
// chordnet/decode.h: chord templates, per-frame log observations and the
// unrestricted Viterbi decode (transition penalty 30) of Music X Lab's
// ISMIR 2019 chord recogniser as MuLaCover bundles it (MIT). The arithmetic
// is float32 in numpy's order, so near-ties resolve the same way.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "chordnet/model.h"

namespace chordnet {

struct Template {
    std::string name;
    int         v[6];  // triad, bass, seventh, ninth, eleventh, thirteenth
};

// The TSV written by convert-chordnet.py --vocab-out; '#' lines are comments.
static bool load_templates(const std::string & path, std::vector<Template> & out) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "[chordnet] cannot read %s\n", path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Template t;
        if (!std::getline(ss, t.name, '\t')) return false;
        for (int &x : t.v)
            if (!(ss >> x)) {
                fprintf(stderr, "[chordnet] bad template line: %s\n", line.c_str());
                return false;
            }
        out.push_back(t);
    }
    if (out.empty() || out[0].name != "N") {
        fprintf(stderr, "[chordnet] %s: the first template must be N\n", path.c_str());
        return false;
    }
    return true;
}

struct Decoded {
    std::vector<std::string> names;   // usable templates, decode order
    std::vector<int>         label;   // [T] index into names
    std::vector<float>       score;   // [T][names] Viterbi scores, for margins
};

// probs: six [T][n] fold means. Templates whose class exceeds a head are
// dropped; the bass head reserves class 0 for no chord.
static Decoded viterbi(const std::vector<Template> & tpl, const std::vector<float> probs[NHEAD], int T,
                       float penalty = 30.0f) {
    Decoded d;
    std::vector<std::array<int, 6>> ch;
    for (const Template & t : tpl) {
        bool ok = true;
        for (int i = 0; i < 6; i++) ok = ok && t.v[i] < HEAD_DIM[i];
        if (!ok) continue;
        d.names.push_back(t.name);
        std::array<int, 6> a;
        for (int i = 0; i < 6; i++) a[i] = t.v[i];
        a[1] += 1;
        ch.push_back(a);
    }
    const int C = (int) ch.size();
    if (T == 0) return d;
    std::vector<float> obs((size_t) T * C);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < C; j++) {
            float o = std::log(probs[0][(size_t) t * HEAD_DIM[0] + ch[j][0]]);
            o += std::log(probs[1][(size_t) t * HEAD_DIM[1] + ch[j][1]]);
            for (int s = 0; s < 4; s++)
                if (ch[j][s + 2] >= 0) o += std::log(probs[s + 2][(size_t) t * HEAD_DIM[s + 2] + ch[j][s + 2]]);
            obs[(size_t) t * C + j] = o;
        }
    d.score.assign((size_t) T * C, 0.0f);
    std::vector<int> best(T), prev((size_t) T * C);
    float * sc = d.score.data();
    for (int j = 0; j < C; j++) sc[j] = (j ? -INFINITY : 0.0f) + obs[j];
    auto argmax = [&](const float * r) {
        int b = 0;
        for (int j = 1; j < C; j++)
            if (r[j] > r[b]) b = j;
        return b;
    };
    best[0] = argmax(sc);
    for (int t = 1; t < T; t++) {
        const float * same = sc + (size_t) (t - 1) * C;
        float *       cur  = sc + (size_t) t * C;
        const float   diff = same[best[t - 1]] - penalty;
        for (int j = 0; j < C; j++) {
            const bool keep = same[j] > diff;
            cur[j] = std::max(diff, same[j]) + obs[(size_t) t * C + j];
            prev[(size_t) t * C + j] = keep ? j : best[t - 1];
        }
        best[t] = argmax(cur);
    }
    d.label.assign(T, 0);
    d.label[T - 1] = best[T - 1];
    for (int t = T - 2; t >= 0; t--) d.label[t] = prev[(size_t) (t + 1) * C + d.label[t + 1]];
    return d;
}

struct Segment {
    std::string label;
    double      onset, offset;
};

static std::vector<Segment> segments(const std::vector<std::string> & labels) {
    const double fs = (double) HOP / SR;
    std::vector<Segment> out;
    size_t start = 0;
    for (size_t f = 0; f < labels.size(); f++)
        if (f + 1 == labels.size() || labels[f + 1] != labels[f]) {
            out.push_back({ labels[f], start * fs, (f + 1) * fs });
            start = f + 1;
        }
    return out;
}

}  // namespace chordnet
