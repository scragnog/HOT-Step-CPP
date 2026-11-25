#pragma once
// dettato/repairer.h: dettato's evidence repairer (evidence/features.py,
// evidence/model.py and evidence/repair.py of dettato 2.0.0, 4.9M parameters;
// see LICENSE here). It edits the transcriber's notes: drops the ones it judges
// wrong, moves onsets and ends, and proposes missed notes.
//
// Host side, in upstream's precision:
//   - the 769-number evidence row of every note: 31 numbers from the six
//     all-window maps, the transcriber's hidden state at the onset and its mean
//     over the note (torch's double-accumulated cumsum, numpy's pairwise float32
//     mean and std), standardised and quantised to int8 steps of 1/32, with the
//     map numbers rounded through float16;
//   - the note features of each 8 s window (Fourier terms, neighbour context, the
//     comb-filter beat grid) and the decode of the model's outputs into edits.
// Network on the ggml CPU backend, one graph per window, as upstream runs it
// (batch of one, so the padding mask is all true and drops out). The CPU is the
// fastest place for it: 837 notes in 11 windows took 70 ms there against 334 ms
// on CUDA and 2.0 s on Vulkan (graph set-up dominates), and Vulkan's matmuls
// ignore the strict-F32 hint (P(keep) 1.3e-3 off; upstream's own CPU/CUDA spread
// on identical inputs is 7e-7). The rotary embedding
// over continuous positions (onset seconds and pitch) uses host cos/sin tables
// and a row-permuted copy of the q and k weights whose output is the pair swap
// (-x[2i+1], x[2i]). Every matmul carries GGML_HINT_STRICT_F32.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "dettato/transcriber.h"  // backend, safetensors Loader, mm(), Note, N_KEYS, FRAME_S

namespace dettato::repairer {

using transcriber::Note;
using transcriber::N_KEYS;
using transcriber::PITCH_LO;

constexpr int D = 192, LAYERS = 9, HEADS = 6, HD = D / HEADS, FF = 576, N_EV = 769, C_H = 368, N_MAPF = 31;
constexpr int ROPE_T = 10, ROPE_P = 4, ROPE_USED = 2 * (ROPE_T + ROPE_P);
constexpr int N_CONT = 24 + 24 + 24 + 2 + 8 + 28;
constexpr int N_DON = 201, N_DR = 161, N_DUR = 128, N_INS_P = 49, N_INS_T = 401, INS_P_MAX = 24, N_SLOT = 2;
constexpr int N_INS = 1 + N_INS_P + N_INS_T + N_DUR;                     // 579
constexpr int O_KEEP = 0, O_DON = 1, O_DUR = O_DON + N_DON, O_INS = O_DUR + N_DR, N_OUT = O_INS + N_SLOT * N_INS;  // 1521
constexpr float  L_WIN = 8.0f, DON_STEP = 0.005f, DON_MAX = 0.5f, DR_STEP = 0.05f, DR_MAX = 4.0f, INS_T_STEP = 0.01f,
                 INS_T_MIN = -1.0f, Q = 32.0f;
constexpr int    MAX_WIN = 4096, GRAPH_NODES = 4096;
constexpr double ONSET_TH = 0.25, DEDUP_S = 0.03;
constexpr float  TWO_PI = (float) (2.0 * M_PI);  // a Python float times a float32 tensor: the float32 of 2 pi

struct Model {
    BackendPair                          bp    = {};       // the shared pair (released at free); the stage runs on bp.cpu_backend
    ggml_backend_sched_t                 sched = nullptr;
    ggml_context *                       ctx   = nullptr;
    ggml_backend_buffer_t                buf   = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::vector<uint8_t>                 meta;
    double                               keep_th = 0.4, insert_th = 0.6;
    std::vector<float> fourier_on, fourier_ld, rope_w, grid_p, dur_cent;  // wavelengths, rotary rates, periods, duration bins
    std::vector<float> dur_w;                                              // [N_DUR][N_DUR], _dur_matrix
    bool               shown = false;
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            fprintf(stderr, "[dettato] missing repairer tensor %s\n", n.c_str());
            exit(1);
        }
        return it->second;
    }
};

// torch.linspace(start, end, n) in float32 (the CPU kernel's scalar steps), then exp
static std::vector<float> exp_linspace(double lo, double hi, int n) {
    const float s = (float) std::log(lo), e = (float) std::log(hi), st = (e - s) / (float) (n - 1);
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) v[i] = std::exp(i < n / 2 ? s + st * (float) i : e - st * (float) (n - i - 1));
    return v;
}

// ---------------------------------------------------------------- config and weights
static bool read_config(Model & m, const std::string & path) {
    yyjson_doc * doc = yyjson_read_file(path.c_str(), 0, nullptr, nullptr);
    if (!doc) {
        fprintf(stderr, "[dettato] cannot read %s\n", path.c_str());
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md   = yyjson_obj_get(root, "model");
    auto         num  = [](yyjson_val * o, const char * k, double fb) {
        yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
        return v && yyjson_is_num(v) ? yyjson_get_num(v) : fb;
    };
    const bool ok = md && yyjson_equals_str(yyjson_obj_get(root, "stage"), "evidence") && num(md, "d", 0) == D &&
                    num(md, "layers", 0) == LAYERS && num(md, "heads", 0) == HEADS && num(md, "ff_mult", 0) == 3 &&
                    num(md, "n_ev", 0) == N_EV && num(root, "window_s", 0) == L_WIN;
    if (ok) {
        m.keep_th   = num(root, "keep_th", 0.4);
        m.insert_th = num(root, "insert_th", 0.6);
    } else {
        fprintf(stderr, "[dettato] %s: not the dettato 2.0 evidence repairer configuration\n", path.c_str());
    }
    yyjson_doc_free(doc);
    return ok;
}

static bool load_model(Model & m, const std::string & dir) {
    if (!read_config(m, dir + "/config.json")) return false;
    transcriber::Loader L;
    if (!st_open(&L.st, (dir + "/model.safetensors").c_str())) return false;
    std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, std::vector<float>>>> pend;
    auto add  = [&](const std::string & n, std::vector<int64_t> ne, std::vector<float> v) { pend.push_back({ n, { std::move(ne), std::move(v) } }); };
    auto copy = [](const float * p, size_t n) { return std::vector<float>(p, p + n); };
    auto lin  = [&](const std::string & src, const std::string & dst, int in, int out, bool bias) {
        add(dst + ".w", { in, out }, copy(L.get(src + ".weight", { out, in }), (size_t) in * out));
        if (bias) add(dst + ".b", { out }, copy(L.get(src + ".bias", { out }), out));
    };
    // rows giving the rotary pair swap of x = W h: out[2i] = -x[2i + 1], out[2i + 1] = x[2i] (i < 14 per head), else 0
    auto rot = [](const float * w) {
        std::vector<float> r((size_t) D * D, 0.0f);
        for (int h = 0; h < HEADS; h++)
            for (int d = 0; d < ROPE_USED; d++) {
                const int   src = h * HD + (d % 2 ? d - 1 : d + 1), dst = h * HD + d;
                const float sg  = d % 2 ? 1.0f : -1.0f;
                for (int i = 0; i < D; i++) r[(size_t) dst * D + i] = sg * w[(size_t) src * D + i];
            }
        return r;
    };

    add("emb.inst", { D, 130 }, copy(L.get("emb.inst.weight", { 130, D }), (size_t) 130 * D));
    add("emb.pitch", { D, 256 }, copy(L.get("emb.pitch.weight", { 256, D }), (size_t) 256 * D));
    add("emb.vel", { D, 16 }, copy(L.get("emb.vel.weight", { 16, D }), (size_t) 16 * D));
    lin("emb.cont.0", "cont0", N_CONT, D, true);
    lin("emb.cont.2", "cont2", D, D, true);
    lin("emb.evm.0", "evm0", N_EV, D, true);
    lin("emb.evm.2", "evm2", D, D, true);
    add("emb.norm", { D }, copy(L.get("emb.norm.w", { D }), D));
    for (int l = 0; l < LAYERS; l++) {
        const std::string p = "bb.blocks." + std::to_string(l), o = "b" + std::to_string(l);
        add(o + ".n1", { D }, copy(L.get(p + ".n1.w", { D }), D));
        add(o + ".n2", { D }, copy(L.get(p + ".n2.w", { D }), D));
        const float * wq = L.get(p + ".att.q.weight", { D, D }), * wkv = L.get(p + ".att.kv.weight", { 2 * D, D });
        add(o + ".q", { D, D }, copy(wq, (size_t) D * D));
        add(o + ".qr", { D, D }, rot(wq));
        add(o + ".k", { D, D }, copy(wkv, (size_t) D * D));
        add(o + ".kr", { D, D }, rot(wkv));
        add(o + ".v", { D, D }, copy(wkv + (size_t) D * D, (size_t) D * D));
        add(o + ".o", { D, D }, copy(L.get(p + ".att.o.weight", { D, D }), (size_t) D * D));
        const float * w1 = L.get(p + ".ff.w1.weight", { 2 * FF, D });
        add(o + ".w1a", { D, FF }, copy(w1, (size_t) FF * D));
        add(o + ".w1b", { D, FF }, copy(w1 + (size_t) FF * D, (size_t) FF * D));
        add(o + ".w2", { FF, D }, copy(L.get(p + ".ff.w2.weight", { D, FF }), (size_t) D * FF));
    }
    add("bb.norm", { D }, copy(L.get("bb.norm.w", { D }), D));
    {  // keep, don, dur and both insertion slots as one [D, N_OUT] head
        std::vector<float> w, b;
        auto cat = [&](const std::string & p, int n) {
            const float * pw = L.get(p + ".weight", { n, D }), * pb = L.get(p + ".bias", { n });
            w.insert(w.end(), pw, pw + (size_t) n * D);
            b.insert(b.end(), pb, pb + n);
        };
        cat("keep", 1);
        cat("don", N_DON);
        cat("dur", N_DR);
        for (int s = 0; s < N_SLOT; s++) cat("ins." + std::to_string(s), N_INS);
        add("head.w", { D, N_OUT }, w);
        add("head.b", { N_OUT }, b);
    }
    for (auto & e : L.st.entries)
        if (!L.used.count(e.name)) {
            fprintf(stderr, "[dettato] unexpected tensor %s in the repairer file\n", e.name.c_str());
            L.ok = false;
        }
    st_close(&L.st);
    if (!L.ok) return false;

    m.bp = backend_init("Dettato");
    ggml_init_params ip = { (pend.size() + 8) * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, const std::vector<float> *>> up;
    for (auto & p : pend) {
        auto &        ne = p.second.first;
        ggml_tensor * d  = ggml_new_tensor(m.ctx, GGML_TYPE_F32, (int) ne.size(), ne.data());
        ggml_set_name(d, p.first.c_str());
        m.t[p.first] = d;
        up.push_back({ d, &p.second.second });
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.bp.cpu_backend);
    if (!m.buf) {
        fprintf(stderr, "[dettato] repairer weight buffer allocation failed\n");
        return false;
    }
    for (auto & u : up) ggml_backend_tensor_set(u.first, u.second->data(), 0, ggml_nbytes(u.first));
    m.sched = backend_sched_new({ m.bp.cpu_backend, m.bp.cpu_backend, false }, GRAPH_NODES);
    m.meta.resize(GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false));

    m.fourier_on = exp_linspace(0.01, 32.0, 12);
    m.fourier_ld = exp_linspace(0.04, 40.0, 12);
    m.grid_p     = exp_linspace(0.18, 1.5, 72);
    for (auto [n, lo, hi] : { std::tuple{ ROPE_T, 0.02, 40.0 }, std::tuple{ ROPE_P, 2.0, 96.0 } })
        for (float lam : exp_linspace(lo, hi, n)) m.rope_w.push_back(TWO_PI / lam);
    {  // DUR_CENT: numpy float64 geometric bin centres, stored float32
        const double lo = std::log(0.01), hi = std::log(16.0), st = (hi - lo) / N_DUR;
        std::vector<double> e(N_DUR + 1);
        for (int i = 0; i <= N_DUR; i++) e[i] = std::exp(i == N_DUR ? hi : i * st + lo);
        m.dur_cent.resize(N_DUR);
        for (int i = 0; i < N_DUR; i++) m.dur_cent[i] = (float) std::sqrt(e[i] * e[i + 1]);
        m.dur_w.assign((size_t) N_DUR * N_DUR, 0.0f);
        for (int i = 0; i < N_DUR; i++) {
            const float tol = std::max(0.2f * m.dur_cent[i], 0.05f);
            for (int j = 0; j < N_DUR; j++) m.dur_w[(size_t) i * N_DUR + j] = std::fabs(m.dur_cent[j] - m.dur_cent[i]) <= tol;
        }
    }
    return true;
}

static void free_model(Model & m) {
    if (m.sched) ggml_backend_sched_free(m.sched);
    if (m.buf) ggml_backend_buffer_free(m.buf);
    if (m.ctx) ggml_free(m.ctx);
    if (m.bp.backend) backend_release(m.bp.backend, m.bp.cpu_backend);
    m = Model();
}

// ---------------------------------------------------------------- evidence (features.py)
// numpy's float32 pairwise sum (contiguous)
static float np_sum(const float * a, int n) {
    if (n < 8) {
        float r = 0.0f;
        for (int i = 0; i < n; i++) r += a[i];
        return r;
    }
    if (n <= 128) {
        float r[8];
        for (int j = 0; j < 8; j++) r[j] = a[j];
        int i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; j++) r[j] += a[i + j];
        float s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += a[i];
        return s;
    }
    int n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

// _std_q of one C_H row: int8 codes / Q into q, returns log(sd) (float32)
static float std_q(const float * x, float * q) {
    const float mu = np_sum(x, C_H) / (float) C_H;
    float       d[C_H];
    for (int i = 0; i < C_H; i++) d[i] = (x[i] - mu) * (x[i] - mu);
    const float sd = std::sqrt(np_sum(d, C_H) / (float) C_H) + 1e-6f;
    for (int i = 0; i < C_H; i++) {
        const float z = std::min(3.96f, std::max(-3.96f, (x[i] - mu) / sd));
        q[i]          = (float) (int8_t) std::nearbyint(z * Q) / Q;
    }
    return std::log(sd);
}

// The evidence rows (N x N_EV, row-aligned with notes) from the all-window maps [6][T * 88] and the all-window
// hidden tap [T][88][C_H] (float16), for the notes the transcriber decoded from those maps.
static std::vector<float> evidence(const std::vector<float> * maps, int T, const ggml_fp16_t * hidden, const std::vector<Note> & notes) {
    const int          N = (int) notes.size();
    std::vector<float> ev((size_t) N * N_EV), at((size_t) N * C_H), mean((size_t) N * C_H);
    std::vector<int>   Lk(N), A(N), B(N);
    auto clampi = [](long long v, long long lo, long long hi) { return (int) std::min(std::max(v, lo), hi); };
    for (int i = 0; i < N; i++) {
        Lk[i] = notes[i].pitch - PITCH_LO;
        A[i]  = clampi((long long) std::nearbyint(notes[i].onset / transcriber::FRAME_S), 0, T - 1);
        B[i]  = clampi((long long) std::nearbyint(notes[i].offset / transcriber::FRAME_S), A[i] + 1, T);
    }
    // map numbers (map_evidence / _map_feats)
    auto at_ = [&](const std::vector<float> & p, int t, int l) { return p[(size_t) std::min(std::max(t, 0), T - 1) * N_KEYS + l]; };
    for (int i = 0; i < N; i++) {
        const int   s2 = notes[i].slot == 2, l = Lk[i], a = A[i], b = B[i], span = std::max(b - a, 1);
        const auto &on = maps[s2 ? 3 : 0], &fr = maps[s2 ? 4 : 1], &off = maps[s2 ? 5 : 2];
        const auto &o_on = maps[s2 ? 0 : 3], &o_fr = maps[s2 ? 1 : 4];
        float *     f = &ev[(size_t) i * N_EV];
        int         k = 0;
        for (int d = -2; d <= 2; d++) f[k++] = at_(on, a + d, l);
        for (int j = 0; j < 8; j++) f[k++] = at_(fr, a + (j * span) / 8, l);
        for (int d : { 1, 3, 6, 12 }) f[k++] = at_(fr, b - 1 + d, l);
        float mx = -1.0f;
        for (int d = 1; d <= 5; d++) mx = std::max(mx, a - d >= 0 ? at_(fr, a - d, l) : 0.0f);
        f[k++] = mx;
        float om = 0.0f, opos = 0.0f;
        for (int t = a + 1, j = 0; t < std::min(b + 12, T); t++, j++) {
            const float v = off[(size_t) t * N_KEYS + l];
            if (j == 0 || v > om) {
                om   = v;
                opos = (float) std::min((j + 1) / (double) span, 2.0);
            }
        }
        f[k++] = om;
        f[k++] = opos;
        f[k++] = at_(off, b - 1, l);
        f[k++] = std::max({ at_(o_on, a - 1, l), at_(o_on, a, l), at_(o_on, a + 1, l) });
        f[k++] = at_(o_fr, a, l);
        float ms = at_(o_fr, a, l);
        for (int j = 1; j < 4; j++) ms += at_(o_fr, a + (j * span) / 4, l);
        f[k++] = ms / 4.0f;
        f[k++] = s2 ? 1.0f : 0.0f;
        for (int dl : { -1, 1, -12, 12, 19, 24 }) {
            const int l2 = l + dl, lc = std::min(std::max(l2, 0), N_KEYS - 1);
            const float v = std::max({ at_(maps[0], a - 1, lc), at_(maps[0], a, lc), at_(maps[0], a + 1, lc) });
            f[k++] = l2 >= 0 && l2 < N_KEYS ? v : 0.0f;
        }
    }
    // hidden state at the onset and its mean over [A, B) (gather_hidden): per lane, torch's cumsum (double
    // accumulator, float32 results), then float32 differences and division
    std::vector<std::vector<int>> by_lane(N_KEYS);
    for (int i = 0; i < N; i++) by_lane[Lk[i]].push_back(i);
    std::vector<int> lanes;
    for (int l = 0; l < N_KEYS; l++)
        if (!by_lane[l].empty()) lanes.push_back(l);
    const int nthreads = std::max(1, std::min({ 8, (int) std::thread::hardware_concurrency(), (int) lanes.size() }));
    std::vector<std::thread> pool;
    for (int ti = 0; ti < nthreads; ti++)
        pool.emplace_back([&, ti]() {
            std::vector<float>  cs((size_t) (T + 1) * C_H);
            std::vector<double> acc(C_H);
            for (size_t li = ti; li < lanes.size(); li += nthreads) {
                const int l = lanes[li];
                std::fill(acc.begin(), acc.end(), 0.0);
                std::fill(cs.begin(), cs.begin() + C_H, 0.0f);
                for (int t = 0; t < T; t++) {
                    const ggml_fp16_t * h = hidden + ((size_t) t * N_KEYS + l) * C_H;
                    float *             c = &cs[(size_t) (t + 1) * C_H];
                    for (int j = 0; j < C_H; j++) c[j] = (float) (acc[j] += (double) ggml_fp16_to_fp32(h[j]));
                }
                for (int i : by_lane[l]) {
                    const ggml_fp16_t * h  = hidden + ((size_t) A[i] * N_KEYS + l) * C_H;
                    const float         dn = (float) std::max(B[i] - A[i], 1);
                    for (int j = 0; j < C_H; j++) {
                        at[(size_t) i * C_H + j]   = ggml_fp16_to_fp32(h[j]);
                        mean[(size_t) i * C_H + j] = (cs[(size_t) B[i] * C_H + j] - cs[(size_t) A[i] * C_H + j]) / dn;
                    }
                }
            }
        });
    for (auto & t : pool) t.join();
    // _assemble: map numbers and both log spreads through float16, then the two code vectors
    auto h16 = [](float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); };
    for (int i = 0; i < N; i++) {
        float *     f  = &ev[(size_t) i * N_EV];
        const float la = std_q(&at[(size_t) i * C_H], f + N_MAPF + 2);
        const float lm = std_q(&mean[(size_t) i * C_H], f + N_MAPF + 2 + C_H);
        f[N_MAPF]      = la;
        f[N_MAPF + 1]  = lm;
        for (int k = 0; k < N_MAPF + 2; k++) f[k] = h16(f[k]);
    }
    return ev;
}

// ---------------------------------------------------------------- notes in and out (repair.py)
struct KNote {  // NOTE_DT: inst is always 0 (pitched)
    int   pitch = 0, vel = 0;
    float on = 0, off = 0;
};

// np.round(x, 4): x * 1e4, rint, / 1e4
static double np_round4(double x) { return std::nearbyint(x * 1e4) / 1e4; }

// Python's round(x, 4): the exact binary value rounded half to even (fma decides the tie side exactly)
static double py_round4(double x) {
    double f = std::floor(x * 1e4);
    if (std::fma(x, 1e4, -f) < 0) f -= 1;
    else if (std::fma(x, 1e4, -(f + 1)) >= 0) f += 1;
    const double e = std::fma(x, 1e4, -(f + 0.5));
    const double r = e > 0 || (e == 0 && std::fmod(f, 2.0) != 0) ? f + 1 : f;
    return r / 1e4;
}

// input_notes: the repairer's notes sorted by (onset, pitch); order[i] = the transcriber note of k[i]
static std::vector<KNote> input_notes(const std::vector<Note> & res, std::vector<int> & order) {
    const int          n = (int) res.size();
    std::vector<KNote> k(n);
    for (int i = 0; i < n; i++) {
        const double on = np_round4(res[i].onset), off = np_round4(res[i].offset);
        const double x  = ((double) res[i].conf - ONSET_TH) / (1.0 - ONSET_TH);
        k[i].pitch      = res[i].pitch;
        k[i].vel        = (int) std::min(127.0, std::max(1.0, std::nearbyint(1 + 126 * x)));
        k[i].on         = (float) on;
        k[i].off        = (float) std::max(off, on + 1e-3);
    }
    order.resize(n);
    for (int i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return k[a].on != k[b].on ? k[a].on < k[b].on : k[a].pitch < k[b].pitch;
    });
    std::vector<KNote> s(n);
    for (int i = 0; i < n; i++) s[i] = k[order[i]];
    return s;
}

// ---------------------------------------------------------------- one window's note features (NoteEmbed's cont input)
// on, dur relative to the window; out [n][N_CONT]
static void note_features(const Model & m, const std::vector<float> & on, const std::vector<float> & dur, const std::vector<int> & pitch,
                          float * out) {
    const int n = (int) on.size();
    auto fourier = [](float x, const std::vector<float> & lam, float * o) {
        const int k = (int) lam.size();
        for (int i = 0; i < k; i++) {
            const float a = TWO_PI * x / lam[i];
            o[i]          = std::sin(a);
            o[k + i]      = std::cos(a);
        }
    };
    auto sl = [](float x) { return (x > 0 ? 1.0f : x < 0 ? -1.0f : 0.0f) * std::log1p(std::fabs(x) * 20.0f); };
    const float big = 10.0f;
    // grid_features: comb filter over 72 periods, two strongest separated by more than 3 bins
    const int           NP = (int) m.grid_p.size();
    std::vector<float>  R(NP), phase(NP);
    for (int p = 0; p < NP; p++) {
        double zr = 0, zi = 0;
        for (int i = 0; i < n; i++) {
            const float a = TWO_PI * on[i] / m.grid_p[p];
            zr += std::cos(a);
            zi += std::sin(a);
        }
        const float fr = (float) zr, fi = (float) zi;
        R[p]     = std::sqrt(fr * fr + fi * fi) / (float) std::max(n, 1);
        phase[p] = std::atan2(fi, fr);
    }
    int   kb[2];
    float Rm[72];
    std::copy(R.begin(), R.end(), Rm);
    for (int it = 0; it < 2; it++) {
        int k = 0;
        for (int p = 1; p < NP; p++)
            if (Rm[p] > Rm[k]) k = p;
        kb[it] = k;
        for (int p = 0; p < NP; p++)
            if (std::abs(p - k) <= 3) Rm[p] = -1.0f;
    }
    static const int SUB[6] = { 1, 2, 3, 4, 6, 8 };
    for (int i = 0; i < n; i++) {
        float *     o  = out + (size_t) i * N_CONT;
        const float ld = std::log(std::max(dur[i], 1e-4f)), off = on[i] + dur[i];
        fourier(on[i], m.fourier_on, o);
        fourier(ld, m.fourier_ld, o + 24);
        fourier(off, m.fourier_on, o + 48);
        o[72] = (ld + 3.0f) / 3.0f;
        o[73] = std::min(dur[i], 8.0f) / 2.0f;
        // context_features
        float prev_on = big, prev_off_v = 0, next_on = big;
        int   chord = 0, dens = 0, same_near = 0;
        bool  have_prev = false;
        for (int j = 0; j < n; j++) {
            const float dt = on[j] - on[i], adt = std::fabs(dt);
            const bool  sk = pitch[j] == pitch[i] && j != i;
            if (sk && dt <= 0 && -dt < prev_on) {
                prev_on    = -dt;
                prev_off_v = on[j] + dur[j];
                have_prev  = true;
            }
            if (sk && dt > 0 && dt < next_on) next_on = dt;
            chord += adt < 0.035f;
            dens += adt < 0.5f;
            same_near += adt < 0.25f && sk;
        }
        const float prev_gap_off = have_prev && prev_on < big ? on[i] - prev_off_v : big;
        const float next_overlap = next_on < big ? off - (on[i] + next_on) : -big;
        float *     c = o + 74;
        c[0] = sl(prev_on);
        c[1] = sl(prev_gap_off);
        c[2] = sl(next_on);
        c[3] = sl(next_overlap);
        c[4] = std::log1p((float) chord);
        c[5] = std::log1p((float) dens);
        c[6] = std::log1p((float) dens);  // every note is instrument 0: same_inst is the whole window
        c[7] = std::log1p((float) same_near);
        float * g = o + 82;
        for (int it = 0; it < 2; it++) {
            const float Pk = m.grid_p[kb[it]], base = TWO_PI * on[i] / Pk - phase[kb[it]];
            for (int s : SUB) {
                *g++ = std::sin((float) s * base);
                *g++ = std::cos((float) s * base);
            }
            *g++ = R[kb[it]];
            *g++ = std::log(Pk);
        }
    }
}

// ---------------------------------------------------------------- network
static ggml_tensor * rms(ggml_context * c, ggml_tensor * x, ggml_tensor * w) { return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), w); }

static ggml_tensor * lin(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    return ggml_add(c, transcriber::mm(c, m.get(p + ".w"), x), m.get(p + ".b"));
}

// one window: [N_OUT, n]
static ggml_tensor * build(ggml_context * c, const Model & m, ggml_tensor * pitch, ggml_tensor * vel, ggml_tensor * cont,
                           ggml_tensor * ev, ggml_tensor * cs, ggml_tensor * sn) {
    using transcriber::mm;
    const int64_t n  = cont->ne[1];
    ggml_tensor * x  = ggml_add(c, ggml_get_rows(c, m.get("emb.pitch"), pitch), ggml_view_1d(c, m.get("emb.inst"), D, 0));
    x                = ggml_add(c, x, ggml_get_rows(c, m.get("emb.vel"), vel));
    x                = ggml_add(c, x, lin(c, m, ggml_silu(c, lin(c, m, cont, "cont0")), "cont2"));
    x                = ggml_add(c, x, lin(c, m, ggml_silu(c, lin(c, m, ev, "evm0")), "evm2"));
    x                = rms(c, x, m.get("emb.norm"));
    auto rope = [&](ggml_tensor * a, ggml_tensor * ar) {  // [D, n] -> [HD, HEADS, n]
        a  = ggml_reshape_3d(c, a, HD, HEADS, n);
        ar = ggml_reshape_3d(c, ar, HD, HEADS, n);
        return ggml_add(c, ggml_mul(c, a, cs), ggml_mul(c, ar, sn));
    };
    for (int l = 0; l < LAYERS; l++) {
        const std::string p = "b" + std::to_string(l);
        ggml_tensor *     h = rms(c, x, m.get(p + ".n1"));
        ggml_tensor *     q = rope(mm(c, m.get(p + ".q"), h), mm(c, m.get(p + ".qr"), h));
        ggml_tensor *     k = rope(mm(c, m.get(p + ".k"), h), mm(c, m.get(p + ".kr"), h));
        ggml_tensor *     v = ggml_reshape_3d(c, mm(c, m.get(p + ".v"), h), HD, HEADS, n);
        ggml_tensor *     q4 = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // [HD, n, HEADS]
        ggml_tensor *     k4 = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor *     vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));  // [n, HD, HEADS]
        ggml_tensor *     kq = ggml_soft_max_ext(c, mm(c, k4, q4), nullptr, 1.0f / std::sqrt((float) HD), 0.0f);
        ggml_tensor *     o  = ggml_cont(c, ggml_permute(c, mm(c, vt, kq), 0, 2, 1, 3));  // [HD, HEADS, n]
        x                    = ggml_add(c, x, mm(c, m.get(p + ".o"), ggml_reshape_2d(c, o, D, n)));
        h                    = rms(c, x, m.get(p + ".n2"));
        ggml_tensor * f      = ggml_mul(c, ggml_silu(c, mm(c, m.get(p + ".w1a"), h)), mm(c, m.get(p + ".w1b"), h));
        x                    = ggml_add(c, x, mm(c, m.get(p + ".w2"), f));
    }
    return lin(c, m, rms(c, x, m.get("bb.norm")), "head");
}

// k: one window's notes (on relative to the window start, float32), ev: their rows -> out [n][N_OUT]
static void forward(Model & m, const std::vector<float> & on, const std::vector<float> & dur, const std::vector<int> & pitch,
                    const std::vector<int> & vel, const float * ev, std::vector<float> & out) {
    const int          n = (int) on.size();
    std::vector<float> cont((size_t) n * N_CONT), cs((size_t) n * HD), sn((size_t) n * HD);
    note_features(m, on, dur, pitch, cont.data());
    for (int i = 0; i < n; i++)
        for (int d = 0; d < HD; d++) {
            const int   a   = d / 2;
            const float ang = d < ROPE_USED ? (a < ROPE_T ? on[i] : (float) pitch[i]) * m.rope_w[a] : 0.0f;
            cs[(size_t) i * HD + d] = d < ROPE_USED ? std::cos(ang) : 1.0f;
            sn[(size_t) i * HD + d] = d < ROPE_USED ? std::sin(ang) : 0.0f;
        }
    std::vector<int32_t> pi(n), vi(n);
    for (int i = 0; i < n; i++) {
        pi[i] = pitch[i];
        vi[i] = vel[i] / 8;
    }
    ggml_init_params ip = { m.meta.size(), m.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    tp = ggml_new_tensor_1d(c, GGML_TYPE_I32, n), * tv = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_tensor *    tc = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_CONT, n), * te = ggml_new_tensor_2d(c, GGML_TYPE_F32, N_EV, n);
    ggml_tensor *    tcs = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, 1, n), * tsn = ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, 1, n);
    for (ggml_tensor * t : { tp, tv, tc, te, tcs, tsn }) ggml_set_input(t);
    ggml_tensor * y = build(c, m, tp, tv, tc, te, tcs, tsn);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(m.sched);
    for (ggml_tensor * t : { tp, tv, tc, te, tcs, tsn }) ggml_backend_sched_set_tensor_backend(m.sched, t, m.bp.cpu_backend);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) {
        fprintf(stderr, "[dettato] repairer graph allocation failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(tp, pi.data(), 0, ggml_nbytes(tp));
    ggml_backend_tensor_set(tv, vi.data(), 0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tc, cont.data(), 0, ggml_nbytes(tc));
    ggml_backend_tensor_set(te, ev, 0, ggml_nbytes(te));
    ggml_backend_tensor_set(tcs, cs.data(), 0, ggml_nbytes(tcs));
    ggml_backend_tensor_set(tsn, sn.data(), 0, ggml_nbytes(tsn));
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(m.sched, gf))) {
        fprintf(stderr, "[dettato] repairer graph compute failed\n");
        exit(1);
    }
    out.resize((size_t) n * N_OUT);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    if (!m.shown) {  // once: graph splits and any node the scheduler left off the compute backend
        int nodes = 0, off = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE) continue;
            nodes++;
            if (ggml_backend_sched_get_tensor_backend(m.sched, t) != m.bp.cpu_backend) off++;
        }
        fprintf(stderr, "[dettato] PLACEMENT repair nodes=%d splits=%d off %s=%d\n", nodes, ggml_backend_sched_get_n_splits(m.sched),
                ggml_backend_name(m.bp.cpu_backend), off);
        m.shown = true;
    }
    ggml_free(c);
}

// ---------------------------------------------------------------- edits (Repairer.candidates, EvidenceRepairer.edits)
struct Insertion {
    double on, off;
    int    pitch;
    float  score;
};

struct Edits {
    std::vector<double>    pk, on_r, off_r;  // NaN: the note's window passed through unchanged
    std::vector<Insertion> ins;              // every proposal inside its window
    int                    windows_run = 0;
};

static void softmax(const float * x, int n, float * p) {
    float mx = x[0];
    for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += p[i] = std::exp(x[i] - mx);
    for (int i = 0; i < n; i++) p[i] /= s;
}

// argmax of a zero-padded box sum of width 2 half + 1 (first maximum)
static int box_argmax(const float * p, int n, int half) {
    int   best = 0;
    float bv   = -1.0f;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = std::max(0, i - half); j <= std::min(n - 1, i + half); j++) s += p[j];
        if (s > bv) {
            bv   = s;
            best = i;
        }
    }
    return best;
}

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// k sorted (input_notes), ev rows in k's order
static Edits edits(Model & m, const std::vector<KNote> & k, const float * ev) {
    const int n = (int) k.size();
    Edits     e;
    e.pk.assign(n, NAN);
    e.on_r.assign(n, NAN);
    e.off_r.assign(n, NAN);
    if (!n) return e;
    float end = k[0].off;
    for (auto & x : k) end = std::max(end, x.off);
    const long long nw = (long long) std::ceil(((double) end + L_WIN) / L_WIN);
    std::vector<float> out, p(N_INS_T), pd(N_DUR);
    for (long long w = 0; w < nw; w++) {
        const double t0 = (double) w * L_WIN;
        auto lb = [&](double t) { return (int) (std::lower_bound(k.begin(), k.end(), t, [](const KNote & a, double v) { return (double) a.on < v; }) - k.begin()); };
        const int lo = lb(t0), hi = lb(t0 + L_WIN), cnt = hi - lo;
        if (cnt <= 0 || cnt > MAX_WIN) continue;
        std::vector<float> on(cnt), dur(cnt);
        std::vector<int>   pitch(cnt), vel(cnt);
        for (int i = 0; i < cnt; i++) {
            const KNote & x = k[lo + i];
            on[i]    = (float) ((double) x.on - t0);
            dur[i]   = std::max(x.off - x.on, 0.0f);
            pitch[i] = x.pitch;
            vel[i]   = x.vel;
        }
        forward(m, on, dur, pitch, vel, ev + (size_t) lo * N_EV, out);
        e.windows_run++;
        for (int i = 0; i < cnt; i++) {
            const float * o  = &out[(size_t) i * N_OUT];
            const float   pk = sigmoid(o[O_KEEP]);
            softmax(o + O_DON, N_DON, p.data());
            const int     don_b = box_argmax(p.data(), N_DON, 10);
            softmax(o + O_DUR, N_DR, p.data());
            const int     rb    = box_argmax(p.data(), N_DR, 3);
            const float   dur_k = std::max(dur[i], 1e-4f) * std::exp((float) rb * DR_STEP - DR_MAX);
            const float   on_k  = on[i] + ((float) don_b * DON_STEP - DON_MAX);
            const float   off_k = on_k + dur_k;
            const bool    ok    = on_k >= 0.0f && on_k < L_WIN;
            e.pk[lo + i]    = ok ? (double) pk : -1.0;
            e.on_r[lo + i]  = (double) on_k + t0;
            e.off_r[lo + i] = (double) off_k + t0;
            for (int s = 0; s < N_SLOT; s++) {
                const float * q  = o + O_INS + s * N_INS;
                const float   pe = sigmoid(q[0]);
                int           ip = 0;
                for (int j = 1; j < N_INS_P; j++)
                    if (q[1 + j] > q[1 + ip]) ip = j;
                softmax(q + 1 + N_INS_P, N_INS_T, p.data());
                const int it = box_argmax(p.data(), N_INS_T, 5);
                softmax(q + 1 + N_INS_P + N_INS_T, N_DUR, pd.data());
                int   idd = 0;
                float bv  = 0;
                for (int j = 0; j < N_DUR; j++) {
                    float acc = 0.0f;
                    for (int a = 0; a < N_DUR; a++) acc += pd[a] * m.dur_w[(size_t) a * N_DUR + j];
                    if (j == 0 || acc > bv) {
                        bv  = acc;
                        idd = j;
                    }
                }
                const float on_i  = on[i] + ((float) it * INS_T_STEP + INS_T_MIN);
                const float off_i = on_i + m.dur_cent[idd];
                if (on_i >= 0.0f && on_i < L_WIN)
                    e.ins.push_back({ (double) on_i + t0, (double) off_i + t0, std::min(127, std::max(0, pitch[i] + ip - INS_P_MAX)), pe });
            }
        }
    }
    return e;
}

// ---------------------------------------------------------------- apply (EvidenceRepairer.apply)
struct Repaired {
    std::vector<double> onset, offset;  // seconds, rounded to 0.1 ms
    std::vector<int>    pitch;
    std::vector<char>   inserted;
    std::vector<float>  score;
    int                 transcribed = 0, kept = 0, dropped = 0, n_inserted = 0;
};

static Repaired apply(const Model & m, const std::vector<KNote> & k, const Edits & e) {
    struct Row { double on, off; int pitch; float score; int kind; };
    std::vector<Row> c;
    for (size_t j = 0; j < k.size(); j++) {
        if (std::isnan(e.pk[j])) c.push_back({ (double) k[j].on, (double) k[j].off, k[j].pitch, 1.0f, 0 });
        else if (e.pk[j] >= m.keep_th) c.push_back({ e.on_r[j], e.off_r[j], k[j].pitch, (float) e.pk[j], 0 });
    }
    for (auto & x : e.ins)
        if ((double) x.score >= m.insert_th) c.push_back({ x.on, x.off, x.pitch, x.score, 1 });
    // _safe_dedup_idx: an insertion near a kept note of the same pitch goes; insertions dedup among themselves
    std::map<int, std::vector<double>> kp;
    for (auto & r : c)
        if (!r.kind) kp[r.pitch].push_back(r.on);
    for (auto & v : kp) std::sort(v.second.begin(), v.second.end());
    std::vector<int> ins_ok;
    for (int q = 0; q < (int) c.size(); q++) {
        if (!c[q].kind) continue;
        bool ok = true;
        auto it = kp.find(c[q].pitch);
        if (it != kp.end()) {
            const auto & v = it->second;
            const int    i = (int) (std::lower_bound(v.begin(), v.end(), c[q].on) - v.begin());
            for (int r : { i - 1, i })
                if (r >= 0 && r < (int) v.size() && std::fabs(v[r] - c[q].on) < DEDUP_S) ok = false;
        }
        if (ok) ins_ok.push_back(q);
    }
    {  // _dedup_idx over the surviving insertions: sorted by (key, onset), neighbours within DEDUP_S keep the higher score
        std::vector<int> o(ins_ok.size());
        for (size_t i = 0; i < o.size(); i++) o[i] = (int) i;
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) {
            const Row & x = c[ins_ok[a]], & y = c[ins_ok[b]];
            return x.pitch != y.pitch ? x.pitch < y.pitch : x.on < y.on;
        });
        std::vector<char> keep(o.size(), 1);
        for (size_t j = 0; j + 1 < o.size(); j++) {
            const Row & a = c[ins_ok[o[j]]], & b = c[ins_ok[o[j + 1]]];
            if (a.pitch != b.pitch || !(b.on - a.on < DEDUP_S) || !keep[j]) continue;
            if (a.score >= b.score) keep[j + 1] = 0;
            else keep[j] = 0;
        }
        std::vector<int> s;
        for (size_t j = 0; j < o.size(); j++)
            if (keep[j]) s.push_back(ins_ok[o[j]]);
        std::sort(s.begin(), s.end());
        ins_ok.swap(s);
    }
    std::vector<char> take(c.size(), 0);
    for (size_t q = 0; q < c.size(); q++) take[q] = !c[q].kind;
    for (int q : ins_ok) take[q] = 1;
    Repaired r;
    r.transcribed = (int) k.size();
    for (size_t q = 0; q < c.size(); q++) {
        if (!take[q]) continue;
        r.onset.push_back(py_round4(c[q].on));
        r.offset.push_back(py_round4(std::max(c[q].off, c[q].on + 0.01)));
        r.pitch.push_back(c[q].pitch);
        r.inserted.push_back((char) c[q].kind);
        r.score.push_back(c[q].score);
        (c[q].kind ? r.n_inserted : r.kept)++;
    }
    r.dropped = r.transcribed - r.kept;
    return r;
}

// The whole stage (EvidenceRepairer.from_run): the transcriber's notes decoded from the all-window maps, the maps
// and the all-window hidden tap -> repaired notes (on the same time axis).
static Repaired repair(Model & m, const std::vector<Note> & notes, const std::vector<float> * maps, int T, const ggml_fp16_t * hidden) {
    const std::vector<float> ev = evidence(maps, T, hidden, notes);
    std::vector<int>         order;
    const std::vector<KNote> k = input_notes(notes, order);
    std::vector<float>       evs((size_t) k.size() * N_EV);
    for (size_t i = 0; i < k.size(); i++) std::copy_n(&ev[(size_t) order[i] * N_EV], N_EV, &evs[i * N_EV]);
    return apply(m, k, edits(m, k, evs.data()));
}

}  // namespace dettato::repairer
