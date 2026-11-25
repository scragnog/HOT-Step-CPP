#pragma once
// dettato/grouper.h: dettato's grouper network (grouper/model.py, NoteNet, 2.9M parameters), its audio front end
// (grouper/front.py), the note features it reads from the transcriber (grouper/core.py: assign_slots, gather) and the
// windowed prediction (Grouper.predict) of dettato 2.0.0; see LICENSE here. The decode is grouper-decode.h.
//
// Host side, in upstream's precision where a value is rounded:
//   - the note features: 12 scalars per note from the every-4 s sub-run maps, and three 368-number taps (hidden state
//     at the onset and over the note, stem output over the first 200 ms) quantised to int8 with a float16 scale;
//   - per 24 s window: the 16-bit audio, the harmonic front end (the transcriber's HCQT on 25.5 s), PCEN and the
//     per-window standardisation; the harmonic demixing (HDemix) and channel profile (Prof) readers; the note
//     rasters; the window's tap standardisation; the note-wise reads of the lane trunk.
// PCEN smooths every channel with a 200-tap filter through a float32 FFT, as upstream does (here pocketfft). Wherever
// the power sits far below the row's peak the value is that FFT's rounding noise, and the network was trained on it:
// on the oracle's inputs an exact filter moves the window probabilities 2 to 4 times further from upstream on average
// than numpy's float32 pocketfft does. No FFT reproduces upstream's noise itself (MKL's, fed torch's float32 exp), so
// grouping that hangs on it differs between any two FFTs, upstream's own CPU and CUDA included. exact_pcen computes the
// filter exactly instead (float64): the deterministic mode the network is checked in (tools/dettato-reference).
// Two graphs per window: the lane trunk (stem conv as one matmul over host patches, six lane blocks, pitch attention
// after blocks 2 and 5) and the note transformer (token sum, seven blocks with rotary positions on onset time, heads).
// They run on CUDA when that is the compute backend and on the CPU otherwise: Vulkan's matmuls ignore the strict-F32
// hint (window probabilities 3e-3 off, a fifth of t1's notes regrouped); Metal is unmeasured. PitchMix is a 63-tap
// depthwise conv along the lanes with the residual folded into the centre tap; gammas fold into the last projection.
// Every matmul carries GGML_HINT_STRICT_F32.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "dettato/grouper-decode.h"
#include "dettato/transcriber.h"  // backend, Loader, mm(), dwconv(), HCQT front end, Average, Note
#include "pocketfft_hdronly.h"

namespace dettato::grouper {

using transcriber::N_KEYS;
using transcriber::PITCH_LO;

constexpr int D = 176, DEPTH = 7, HEADS = 4, HDIM = D / HEADS, FF = 352, EMB = 32, K = 73;
constexpr int HD_H = 10, HD_K = 10, HD_SUS = 4, HD_T = HD_K + HD_SUS, HD_NF = HD_H * HD_T + HD_H, HD_HID = 160;
constexpr int LC = 128, LDEPTH = 6, LANES = 96, TS = 8, LHEADS = 4, LHD = LC / LHEADS, PK = 63, PK_HALF = PK / 2;
constexpr int S15 = 368, S15_HID = 256, N_TAP = 3, N_CH = transcriber::N_CH, N_BINS = transcriber::N_BINS;
constexpr int N_PMIX = 14, NE = 425;  // extended log-frequency axis of the long window (288 + 137 harmonic shifts)
constexpr int SPAN_CAP = 50, STEM_SPAN = 10;
constexpr double PAD0 = 0.5, PAD1 = 1.0, SLOT_TOL = 0.1, DEDUP = 0.03;
constexpr int    GRAPH_NODES = 8192;
static const int PMIX_OFF[N_PMIX] = { -2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31 };
static const int HD_KF[HD_K]      = { -2, 0, 2, 4, 8, 12, 16, 24, 32, 48 };
static const int LDIL[4]          = { 1, 2, 4, 8 };
static const bool LATTN[LDEPTH]   = { false, false, true, false, false, true };  // attn_at = {5, 2}

struct Model {
    BackendPair                          bp    = {};
    ggml_backend_t                       be    = nullptr;  // where the graphs run: bp.backend if CUDA, else bp.cpu_backend
    ggml_backend_sched_t                 sched = nullptr;
    ggml_context *                       ctx   = nullptr;
    ggml_backend_buffer_t                buf   = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::vector<uint8_t>                 meta;
    transcriber::Front                   front;
    std::vector<std::string>             names;     // the 73 instruments, upstream's spelling
    std::vector<int>                     programs;  // their General MIDI programs
    std::vector<Layer>                   head;      // the track head (heads.th), float64, applied by the decode
    DecodeOptions                        opt;
    float                                pi[HD_H] = {};  // softplus(hd.prior)
    double                               window_s = 24.0, window_hop = 0.5;
    int                                  max_notes = 768, batch = 4;
    bool                                 exact_pcen = false;  // validation only: the PCEN filter in float64
    bool                                 shown[2] = { false, false };
    double ms_front = 0, ms_lanes = 0, ms_notes = 0;
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            fprintf(stderr, "[dettato] missing grouper tensor %s\n", n.c_str());
            exit(1);
        }
        return it->second;
    }
};

// ---------------------------------------------------------------- config and weights
static bool read_config(Model & m, const std::string & path) {
    yyjson_doc * doc = yyjson_read_file(path.c_str(), 0, nullptr, nullptr);
    if (!doc) {
        fprintf(stderr, "[dettato] cannot read %s\n", path.c_str());
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md = yyjson_obj_get(root, "model"), * dc = yyjson_obj_get(root, "decode");
    auto num = [](yyjson_val * o, const char * k, double fb) {
        yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
        return v && yyjson_is_num(v) ? yyjson_get_num(v) : fb;
    };
    auto flag = [](yyjson_val * o, const char * k, bool fb) {
        yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
        return v && yyjson_is_bool(v) ? yyjson_get_bool(v) : fb;
    };
    yyjson_val * ins = yyjson_obj_get(root, "instruments"), * prg = yyjson_obj_get(root, "programs"),
               * taps = md ? yyjson_obj_get(md, "s15_taps") : nullptr;
    bool ok = md && dc && yyjson_equals_str(yyjson_obj_get(root, "stage"), "grouper") && num(md, "d", 0) == D &&
              num(md, "depth", 0) == DEPTH && num(md, "heads", 0) == HEADS && num(md, "ff", 0) == FF && num(md, "emb", 0) == EMB &&
              num(md, "hd", 0) == HD_H && num(md, "hd_sus", 0) == HD_SUS && num(md, "lanes", 0) == LC &&
              num(md, "lane_depth", 0) == LDEPTH && num(md, "lane_ts", 0) == TS && num(md, "lane_attn", 0) == 2 &&
              num(md, "lane_exp", 0) == 2 && num(md, "n_lanes", 0) == LANES && num(md, "s15_hid", 0) == S15_HID &&
              num(md, "s15_dim", 0) == S15 && num(md, "pos_scale", 0) == 10.0 && num(root, "sample_rate", 0) == transcriber::SR &&
              taps && yyjson_arr_size(taps) == N_TAP && ins && yyjson_arr_size(ins) == K && prg && yyjson_arr_size(prg) == K;
    for (int i = 0; ok && i < N_TAP; i++) ok = yyjson_get_num(yyjson_arr_get(taps, i)) == i;
    if (ok) {
        for (int i = 0; i < K; i++) {
            const char * s = yyjson_get_str(yyjson_arr_get(ins, i));
            m.names.push_back(s ? s : "");
            m.programs.push_back((int) yyjson_get_num(yyjson_arr_get(prg, i)));
        }
        m.window_s   = num(root, "window_s", 24.0);
        m.window_hop = num(root, "window_hop", 0.5);
        m.max_notes  = (int) num(root, "max_notes", 768);
        DecodeOptions & o = m.opt;
        o.tau = num(dc, "tau", o.tau);
        o.tau2 = num(dc, "tau2", o.tau2);
        o.lambda = num(dc, "lam", o.lambda);
        o.min_notes = (int) num(dc, "min_notes", o.min_notes);
        o.refine = (int) num(dc, "refine", o.refine);
        o.alpha = num(dc, "alpha", o.alpha);
        o.expected_credit = flag(dc, "ec", o.expected_credit);
        o.merge = flag(dc, "merge", o.merge);
        o.absorb = num(dc, "absorb", o.absorb);
        o.absorb_fraction = num(dc, "absorb_frac", o.absorb_fraction);
        // the window layout this port implements (24 s windows, 25.5 s of audio)
        ok = m.window_s == 24.0 && m.window_hop > 0 && m.window_hop <= 1 && m.max_notes > 0;  // a hop of 0 would never advance
    }
    if (!ok) fprintf(stderr, "[dettato] %s: not the dettato 2.0 grouper configuration\n", path.c_str());
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
    auto lin  = [&](const std::string & src, const std::string & dst, int in, int out) {
        add(dst + ".w", { in, out }, copy(L.get(src + ".weight", { out, in }), (size_t) in * out));
        add(dst + ".b", { out }, copy(L.get(src + ".bias", { out }), out));
    };
    auto ln = [&](const std::string & src, const std::string & dst, int n, const char * w = "weight", const char * b = "bias") {
        add(dst + ".w", { n }, copy(L.get(src + "." + w, { n }), n));
        add(dst + ".b", { n }, copy(L.get(src + "." + b, { n }), n));
    };
    // depthwise kernels channel-fastest, [taps][LC] (transcriber::dwconv)
    auto chan_last = [](const float * p, int taps) {
        std::vector<float> v((size_t) taps * LC);
        for (int c = 0; c < LC; c++)
            for (int k = 0; k < taps; k++) v[(size_t) k * LC + c] = p[(size_t) c * taps + k];
        return v;
    };
    // a linear layer whose output rows are scaled by g (a gamma folded in)
    auto lin_g = [&](const std::string & src, const std::string & dst, int in, int out, const float * g) {
        const float * w = L.get(src + ".weight", { out, in }), * b = L.get(src + ".bias", { out });
        std::vector<float> wf((size_t) in * out), bf(out);
        for (int o = 0; o < out; o++) {
            for (int i = 0; i < in; i++) wf[(size_t) o * in + i] = w[(size_t) o * in + i] * g[o];
            bf[o] = b[o] * g[o];
        }
        add(dst + ".w", { in, out }, wf);
        add(dst + ".b", { out }, bf);
    };

    // ---- note-level readers and embeddings
    lin("hd.enc.0", "hd0", HD_NF, HD_HID);
    lin("hd.enc.2", "hd2", HD_HID, D);
    {
        const float * pr = L.get("hd.prior", { HD_H });
        for (int h = 0; h < HD_H; h++) m.pi[h] = pr[h] > 20.0f ? pr[h] : log1pf(expf(pr[h]));  // F.softplus
    }
    lin("prof.net.0", "prof0", 4 * N_CH, 64);
    lin("prof.net.2", "prof2", 64, D);
    ln("s15_ln", "s15ln", S15);
    lin("s15_net.0", "s15a", N_TAP * S15, S15_HID);
    lin("s15_net.2", "s15b", S15_HID, D);
    add("pitch_emb", { D, 128 }, copy(L.get("pitch_emb.weight", { 128, D }), (size_t) 128 * D));
    lin("feat.0", "feat0", 11, 64);
    lin("feat.2", "feat2", 64, D);
    ln("ln_in", "ln_in", D);
    ln("ln_out", "ln_out", D);
    lin("heads.inst", "inst", D, K);
    lin("heads.emb", "emb", D, EMB);
    L.get("heads.keep.weight", { 1, D });  // the keep score: not read by the pipeline
    L.get("heads.keep.bias", { 1 });
    for (int i : { 0, 2, 4 }) {  // the track head, float64 for the decode
        const int     in = i == 0 ? D + 6 : D, out = i == 4 ? K : D;
        const float * w = L.get("heads.th." + std::to_string(i) + ".weight", { out, in });
        const float * b = L.get("heads.th." + std::to_string(i) + ".bias", { out });
        Layer         l;
        l.bias.assign(b, b + out);
        l.weights.assign(out, Vec(in));
        for (int o = 0; o < out; o++)
            for (int j = 0; j < in; j++) l.weights[o][j] = w[(size_t) o * in + j];
        m.head.push_back(std::move(l));
    }
    for (int i = 0; i < DEPTH; i++) {
        const std::string p = "blocks." + std::to_string(i), o = "b" + std::to_string(i);
        ln(p + ".ln1", o + ".ln1", D);
        ln(p + ".ln2", o + ".ln2", D);
        lin(p + ".qkv", o + ".qkv", D, 3 * D);
        {  // rows giving rotate_half(q), rotate_half(k): out[d] = -x[d + 22] (d < 22), x[d - 22] (d >= 22)
            const float * wq = L.get(p + ".qkv.weight", { 3 * D, D }), * bq = L.get(p + ".qkv.bias", { 3 * D });
            std::vector<float> wr((size_t) 2 * D * D), br(2 * D);
            const int          half = HDIM / 2;
            for (int part = 0; part < 2; part++)
                for (int h = 0; h < HEADS; h++)
                    for (int d = 0; d < HDIM; d++) {
                        const int   src = part * D + h * HDIM + (d < half ? d + half : d - half), dst = part * D + h * HDIM + d;
                        const float sg  = d < half ? -1.0f : 1.0f;
                        for (int i2 = 0; i2 < D; i2++) wr[(size_t) dst * D + i2] = sg * wq[(size_t) src * D + i2];
                        br[dst] = sg * bq[src];
                    }
            add(o + ".rot.w", { D, 2 * D }, wr);
            add(o + ".rot.b", { 2 * D }, br);
        }
        lin(p + ".o", o + ".o", D, D);
        lin(p + ".ff.0", o + ".ff0", D, FF);
        lin(p + ".ff.2", o + ".ff2", FF, D);
    }

    // ---- lane trunk
    {
        const float * w = L.get("lanes.stem.weight", { LC, N_CH, TS, 3 });  // rows c * 24 + kt * 3 + kf, as the host patches
        add("stem.w", { N_CH * TS * 3, LC }, copy(w, (size_t) LC * N_CH * TS * 3));
        add("stem.b", { LC }, copy(L.get("lanes.stem.bias", { LC }), LC));
        add("occ.w", { 4, LC }, copy(L.get("lanes.occ.weight", { LC, 4, 1, 1 }), (size_t) LC * 4));
        add("occ.b", { LC }, copy(L.get("lanes.occ.bias", { LC }), LC));
    }
    for (int i = 0; i < LDEPTH; i++) {
        const std::string p = "lanes.blocks." + std::to_string(i), o = "l" + std::to_string(i);
        const float * kt = L.get(p + ".dw_t.weight", { LC, 1, 7, 1 }), * bt = L.get(p + ".dw_t.bias", { LC });
        const float * taps = L.get(p + ".pm.taps", { LC, N_PMIX });
        add(o + ".kt", { LC, 1, 1, 7 }, chan_last(kt, 7));
        std::vector<float> kp((size_t) LC * PK, 0.0f), bp((size_t) LC * LANES);
        for (int c = 0; c < LC; c++) {
            kp[(size_t) c * PK + PK_HALF] = 1.0f;
            for (int k = 0; k < N_PMIX; k++) kp[(size_t) c * PK + PK_HALF + PMIX_OFF[k]] = taps[c * N_PMIX + k];
            for (int l = 0; l < LANES; l++) {  // the time conv's bias through y + PitchMix(y)
                double s = 1.0;
                for (int k = 0; k < N_PMIX; k++)
                    if (l + PMIX_OFF[k] >= 0 && l + PMIX_OFF[k] < LANES) s += taps[c * N_PMIX + k];
                bp[(size_t) l * LC + c] = (float) (bt[c] * s);
            }
        }
        add(o + ".kp", { LC, 1, PK, 1 }, chan_last(kp.data(), PK));
        add(o + ".bp", { LC, LANES, 1, 1 }, bp);
        ln(p + ".norm", o + ".norm", LC, "w", "b");
        add(o + ".w1", { LC, 2 * LC }, copy(L.get(p + ".pw1.weight", { 2 * LC, LC, 1, 1 }), (size_t) 2 * LC * LC));
        add(o + ".b1", { 2 * LC }, copy(L.get(p + ".pw1.bias", { 2 * LC }), 2 * LC));
        const float * g  = L.get(p + ".gamma", { LC });
        const float * w2 = L.get(p + ".pw2.weight", { LC, 2 * LC, 1, 1 }), * b2 = L.get(p + ".pw2.bias", { LC });
        std::vector<float> w2f((size_t) LC * 2 * LC), b2f(LC);
        for (int oc = 0; oc < LC; oc++) {
            for (int ic = 0; ic < 2 * LC; ic++) w2f[(size_t) oc * 2 * LC + ic] = w2[(size_t) oc * 2 * LC + ic] * g[oc];
            b2f[oc] = b2[oc] * g[oc];
        }
        add(o + ".w2", { 2 * LC, LC }, w2f);
        add(o + ".b2", { LC }, b2f);
        if (LATTN[i]) {
            const std::string q = "lanes.attn." + std::to_string(i), a = "a" + std::to_string(i);
            ln(q + ".ln", a + ".ln", LC);
            lin(q + ".qkv", a + ".qkv", LC, 3 * LC);
            lin_g(q + ".out", a + ".out", LC, LC, L.get(q + ".gamma", { LC }));
            const float * rb = L.get(q + ".rel_bias", { LHEADS, 2 * LANES - 1 });  // mask[h][q][k] = rel_bias[h][k - q + 95]
            std::vector<float> mk((size_t) LHEADS * LANES * LANES);
            for (int h = 0; h < LHEADS; h++)
                for (int qi = 0; qi < LANES; qi++)
                    for (int ki = 0; ki < LANES; ki++)
                        mk[((size_t) h * LANES + qi) * LANES + ki] = rb[h * (2 * LANES - 1) + ki - qi + LANES - 1];
            add(a + ".mask", { LANES, LANES, LHEADS, 1 }, mk);
        }
    }
    lin("lanes.read", "read", 2 * LC, D);

    for (auto & e : L.st.entries)
        if (!L.used.count(e.name)) {
            fprintf(stderr, "[dettato] unexpected tensor %s in the grouper file\n", e.name.c_str());
            L.ok = false;
        }
    st_close(&L.st);
    if (!L.ok) return false;

    m.bp = backend_init("Dettato");
    const bool cuda = strncmp(ggml_backend_name(m.bp.backend), "CUDA", 4) == 0;
    m.be = cuda ? m.bp.backend : m.bp.cpu_backend;
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
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.be);
    if (!m.buf) {
        fprintf(stderr, "[dettato] grouper weight buffer allocation failed\n");
        return false;
    }
    for (auto & u : up) ggml_backend_tensor_set(u.first, u.second->data(), 0, ggml_nbytes(u.first));
    m.sched = cuda ? backend_sched_new(m.bp, GRAPH_NODES) : backend_sched_new({ m.bp.cpu_backend, m.bp.cpu_backend, false }, GRAPH_NODES);
    m.meta.resize(GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false));
    m.front = transcriber::make_front();
    return true;
}

static void free_model(Model & m) {
    if (m.sched) ggml_backend_sched_free(m.sched);
    if (m.buf) ggml_backend_buffer_free(m.buf);
    if (m.ctx) ggml_free(m.ctx);
    if (m.bp.backend) backend_release(m.bp.backend, m.bp.cpu_backend);
    m = Model();
}

// ---------------------------------------------------------------- note features (assign_slots, gather)
constexpr int N_META = 12;

struct Features {
    int                      n = 0;
    std::vector<float>       meta;  // [n][12]: onset, offset, pitch, slot, 8 map readings
    std::vector<int8_t>      feat;  // [n][3][368]
    std::vector<ggml_fp16_t> fsc;   // [n][3]
};

// Input notes (onset, offset, pitch) + the transcriber's own notes -> [onset, max(offset, onset + 0.02), pitch, slot 0/1],
// sorted by (onset, pitch); slot of the nearest same-pitch transcriber note within 0.1 s, else 0.
static std::vector<std::array<double, 4>> assign_slots(const std::vector<std::array<double, 3>> & in,
                                                       const std::vector<transcriber::Note> & own) {
    const size_t        n = in.size();
    std::vector<double> slot(n, 0.0);
    std::map<int, std::vector<int>> by_pitch;
    for (size_t j = 0; j < own.size(); j++) by_pitch[own[j].pitch].push_back((int) j);
    for (size_t i = 0; i < n; i++) {
        auto it = by_pitch.find((int) in[i][2]);
        if (it == by_pitch.end() || (double) it->first != in[i][2]) continue;
        int    best = -1;
        double bd   = 0;
        for (int j : it->second) {  // argmin, first of equals
            const double d = std::fabs(in[i][0] - own[j].onset);
            if (best < 0 || d < bd) {
                best = j;
                bd   = d;
            }
        }
        slot[i] = bd <= SLOT_TOL ? own[best].slot - 1.0 : 0.0;
    }
    std::vector<int> o(n);
    for (size_t i = 0; i < n; i++) o[i] = (int) i;
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return in[a][0] != in[b][0] ? in[a][0] < in[b][0] : in[a][2] < in[b][2]; });
    std::vector<std::array<double, 4>> out(n);
    for (size_t i = 0; i < n; i++) out[i] = { in[o[i]][0], std::max(in[o[i]][1], in[o[i]][0] + 0.02), in[o[i]][2], slot[o[i]] };
    return out;
}

// The sub-run maps [6][T][88] and float16 taps [T][88][368] at the notes (assign_slots output).
static Features gather(const std::vector<float> * P, int T, const ggml_fp16_t * H, const ggml_fp16_t * S,
                       const std::vector<std::array<double, 4>> & nt) {
    constexpr int C = transcriber::C;
    Features      f;
    f.n = (int) nt.size();
    f.meta.resize((size_t) f.n * N_META);
    f.feat.resize((size_t) f.n * N_TAP * C);
    f.fsc.resize((size_t) f.n * N_TAP);
    const double hop = transcriber::FRAME_S;
    // torch's cumsum over time (double accumulator, float32 out) of a map's lane: cs[t] = sum of the first t frames
    std::map<std::pair<int, int>, std::vector<float>> cums;
    auto cs = [&](int head, int lane) -> const std::vector<float> & {
        auto & v = cums[{ head, lane }];
        if (v.empty()) {
            v.resize(T + 1);
            double a = 0;
            v[0]     = 0.0f;
            for (int t = 0; t < T; t++) v[t + 1] = (float) (a += P[head][(size_t) t * N_KEYS + lane]);
        }
        return v;
    };
    for (int i = 0; i < f.n; i++) {
        const int  lane = std::clamp((int) std::nearbyint(nt[i][2]) - PITCH_LO, 0, N_KEYS - 1);
        const int  t0   = (int) std::clamp((long long) std::nearbyint(nt[i][0] / hop), 0LL, (long long) T - 1);
        const int  te   = (int) std::min((long long) T, std::max((long long) std::ceil(nt[i][1] / hop), (long long) t0 + 1));
        auto at  = [&](int h, int t) { return P[h][(size_t) std::clamp(t, 0, T - 1) * N_KEYS + lane]; };
        auto mx3 = [&](int h, int t) { return std::max({ at(h, t - 1), at(h, t), at(h, t + 1) }); };
        auto spm = [&](int h) {
            const std::vector<float> & c = cs(h, lane);
            return (c[te] - c[t0]) / (float) std::max(te - t0, 1);
        };
        float * mt = &f.meta[(size_t) i * N_META];
        // heads: 0 onset, 1 frame, 2 offset, 3 onset2, 4 frame2, 5 offset2
        const float row[N_META] = { (float) nt[i][0], (float) nt[i][1], (float) nt[i][2], (float) nt[i][3],
                                    mx3(0, t0), mx3(3, t0), at(1, t0), at(4, t0), spm(1), spm(4), mx3(2, te - 1), mx3(5, te - 1) };
        std::copy(row, row + N_META, mt);
        float X[N_TAP][C];
        auto  rows = [&](const ggml_fp16_t * M, int t, float * dst, bool add) {
            const ggml_fp16_t * src = M + ((size_t) t * N_KEYS + lane) * C;
            for (int c = 0; c < C; c++) dst[c] = add ? dst[c] + ggml_fp16_to_fp32(src[c]) : ggml_fp16_to_fp32(src[c]);
        };
        rows(H, t0, X[0], false);
        for (int k = 1; k < N_TAP; k++) {  // float32 sum over [t0, min(te, t0 + cap)), then / count
            const ggml_fp16_t * M   = k == 1 ? H : S;
            const int           lim = std::min(te, t0 + (k == 1 ? SPAN_CAP : STEM_SPAN));
            std::fill(X[k], X[k] + C, 0.0f);
            for (int t = t0; t < lim; t++) rows(M, t, X[k], true);
            const float cnt = (float) std::max(lim - t0, 1);
            for (int c = 0; c < C; c++) X[k][c] /= cnt;
        }
        for (int k = 0; k < N_TAP; k++) {
            float mx = 0.0f;
            for (int c = 0; c < C; c++) mx = std::max(mx, std::fabs(X[k][c]));
            const float sc = std::max(mx, 1e-6f) / 127.0f;
            for (int c = 0; c < C; c++)
                f.feat[((size_t) i * N_TAP + k) * C + c] = (int8_t) std::clamp(std::nearbyint(X[k][c] / sc), -127.0f, 127.0f);
            f.fsc[(size_t) i * N_TAP + k] = ggml_fp32_to_fp16(sc);
        }
    }
    return f;
}

// ---------------------------------------------------------------- windows (song_windows, _Song.window)
struct Win {
    double           t0;
    std::vector<int> sel;
};

// meta's onsets (sorted) -> windows: a fixed hop; a window holding more than nmax notes is followed by one starting at
// its first uncovered note. Window starts stay float32-exact (multiples of 3 s or float32 onsets).
static std::vector<Win> song_windows(const std::vector<float> & on, double W, double hop, int nmax) {
    std::vector<Win> out;
    const int        n = (int) on.size();
    if (!n) return out;
    const double end = on[n - 1];
    double       t0  = 0.0;
    auto ss = [&](double v) { return (int) (std::lower_bound(on.begin(), on.end(), (float) v) - on.begin()); };
    for (;;) {
        const int a = ss(t0), b = ss(t0 + W);
        Win       w{ t0, {} };
        for (int i = a; i < std::min(b, a + nmax); i++) w.sel.push_back(i);
        if (!w.sel.empty()) out.push_back(w);
        if (b >= n && a + nmax >= b) break;
        t0 = b - a > nmax ? std::max(t0 + 0.25 * hop, (double) on[a + nmax / 2]) : t0 + hop;
        if (t0 > end) break;
    }
    return out;
}

// ---------------------------------------------------------------- front end (front.py Front, pcen, standardise)
// One window's audio (16-bit steps) -> X [N_CH][Tf][N_BINS] (PCEN'd, standardised) and P [NE][Tf] (linear power).
static void front_end(const Model & m, const std::vector<float> & audio, int Tf, std::vector<float> & X, std::vector<float> & P) {
    X.resize((size_t) N_CH * Tf * N_BINS);
    P.resize((size_t) NE * Tf);
    transcriber::hcqt(m.front, audio.data(), X.data(), (int) audio.size(), Tf, P.data());
    // PCEN per (channel, bin): M = causal 200-tap smoothing of E = exp(X) with taps s (1 - s)^j
    constexpr float s = 0.025f, eps = 1e-6f, alpha = 0.98f, delta = 2.0f, r = 0.5f;
    constexpr int   KP = 200;
    const int       ncol = N_CH * N_BINS, nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    std::vector<float> E((size_t) ncol * Tf), M((size_t) ncol * Tf);
    auto cols = [&](auto fn) { transcriber::Average::frames(0, ncol, [&](int c0, int c1) { for (int col = c0; col < c1; col++) fn(col); }); };
    cols([&](int col) {
        const float * x = &X[(size_t) (col / N_BINS) * Tf * N_BINS + col % N_BINS];
        for (int t = 0; t < Tf; t++) E[(size_t) col * Tf + t] = expf(x[(size_t) t * N_BINS]);
    });
    if (m.exact_pcen) {  // float64 recursion with the exact geometric taps
        const double tail = s * std::pow(1.0 - s, KP);
        cols([&](int col) {
            double acc = 0;
            for (int t = 0; t < Tf; t++) {
                const float * e = &E[(size_t) col * Tf];
                acc = (1.0 - s) * acc + (double) s * e[t] - (t >= KP ? tail * e[t - KP] : 0.0);
                M[(size_t) col * Tf + t] = (float) std::max(acc, 0.0);
            }
        });
    } else {  // irfft(rfft(E, n) * rfft(k, n), n)[:T], float32, as torch.fft (pocketfft here)
        size_t n = 1;
        while (n < (size_t) Tf + KP) n <<= 1;
        const size_t nf = n / 2 + 1;
        std::vector<float> buf((size_t) ncol * n, 0.0f), kt(n, 0.0f);
        std::vector<std::complex<float>> spec((size_t) ncol * nf), ks(nf);
        for (int j = 0; j < KP; j++) kt[j] = s * powf(1.0f - s, (float) j);
        for (int col = 0; col < ncol; col++) std::copy_n(&E[(size_t) col * Tf], Tf, &buf[(size_t) col * n]);
        const pocketfft::stride_t sr1 = { (ptrdiff_t) sizeof(float) }, sc1 = { (ptrdiff_t) sizeof(std::complex<float>) };
        const pocketfft::stride_t srn = { (ptrdiff_t) (n * sizeof(float)), (ptrdiff_t) sizeof(float) };
        const pocketfft::stride_t scn = { (ptrdiff_t) (nf * sizeof(std::complex<float>)), (ptrdiff_t) sizeof(std::complex<float>) };
        pocketfft::r2c({ n }, sr1, sc1, { 0 }, pocketfft::FORWARD, kt.data(), ks.data(), 1.0f);
        pocketfft::r2c({ (size_t) ncol, n }, srn, scn, { 1 }, pocketfft::FORWARD, buf.data(), spec.data(), 1.0f, nth);
        cols([&](int col) {
            for (size_t k = 0; k < nf; k++) {  // complex64 product, (ac - bd, ad + bc)
                std::complex<float> & a = spec[(size_t) col * nf + k];
                const float ar = a.real(), ai = a.imag(), br = ks[k].real(), bi = ks[k].imag();
                a = { ar * br - ai * bi, ar * bi + ai * br };
            }
        });
        pocketfft::c2r({ (size_t) ncol, n }, scn, srn, { 1 }, pocketfft::BACKWARD, spec.data(), buf.data(), 1.0f / (float) n, nth);
        for (int col = 0; col < ncol; col++)
            for (int t = 0; t < Tf; t++) M[(size_t) col * Tf + t] = std::max(buf[(size_t) col * n + t], 0.0f);
    }
    cols([&](int col) {
        float * x = &X[(size_t) (col / N_BINS) * Tf * N_BINS + col % N_BINS];
        for (int t = 0; t < Tf; t++) {
            const float e = E[(size_t) col * Tf + t], mm = M[(size_t) col * Tf + t];
            const float out = powf(e / powf(eps + mm, alpha) + delta, r) - sqrtf(delta);
            x[(size_t) t * N_BINS] = logf(out + 1e-4f);
        }
    });
    // standardise every channel over its active region (the top 12 log units)
    for (int c = 0; c < N_CH; c++) {
        float *      x  = &X[(size_t) c * Tf * N_BINS];
        const size_t nn = (size_t) Tf * N_BINS;
        float        mx = -INFINITY;
        for (size_t i = 0; i < nn; i++) mx = std::max(mx, x[i]);
        const float th = mx - 12.0f;
        double      cnt = 0, sum = 0, sq = 0;
        for (size_t i = 0; i < nn; i++)
            if (x[i] > th) {
                cnt++;
                sum += x[i];
            }
        cnt             = std::max(cnt, 1.0);
        const float mu  = (float) (sum / cnt);
        for (size_t i = 0; i < nn; i++)
            if (x[i] > th) sq += (double) (x[i] - mu) * (double) (x[i] - mu);
        const float sd = std::max((float) std::sqrt(sq / cnt), 0.5f);
        for (size_t i = 0; i < nn; i++) x[i] = (x[i] - mu) / sd;
    }
}

struct Frames { int on, off, lane; };

// note_frames: 10 ms frames of the window's audio (which starts PAD0 early)
static Frames note_frames(const float * n, int Tf) {
    Frames    f;
    f.on   = (int) std::clamp((long long) std::nearbyintf((n[0] + 0.5f) * 100.0f), 0LL, (long long) Tf - 1);
    f.off  = (int) std::max(std::min((long long) std::nearbyintf((n[1] + 0.5f) * 100.0f), (long long) Tf), (long long) f.on + 1);
    f.lane = std::clamp((int) std::nearbyintf(n[2]) - PITCH_LO, 0, N_KEYS - 1);
    return f;
}

// HDemix.raw: per note, the log share-power of its 10 harmonics at 10 attack offsets and 4 sustain points -> [n][150]
static void hdemix(const Model & m, const std::vector<float> & P, int Tf, const std::vector<float> & notes, int n, float * out) {
    const int T1 = Tf + 1;
    std::vector<int> hs(HD_H);
    for (int h = 0; h < HD_H; h++) hs[h] = (int) std::nearbyint(36 * std::log2(h + 1.0));
    std::vector<Frames> fr(n);
    std::vector<int>    bins((size_t) n * HD_H);
    std::vector<float>  valid((size_t) n * HD_H);
    std::vector<float>  O((size_t) NE * T1, 0.0f);
    for (int i = 0; i < n; i++) {
        fr[i] = note_frames(&notes[(size_t) i * N_META], Tf);
        for (int h = 0; h < HD_H; h++) {
            const int b = 3 * fr[i].lane + 36 + hs[h];  // 3 lane - lo + shift
            valid[(size_t) i * HD_H + h] = b < NE ? 1.0f : 0.0f;
            bins[(size_t) i * HD_H + h]  = std::clamp(b, 0, NE - 1);
        }
    }
    // occupancy: every note's claim pi on its comb bins over its frames (scatter_add order, then a cumsum over time)
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < n; i++)
            for (int h = 0; h < HD_H; h++) {
                const float w = m.pi[h] * valid[(size_t) i * HD_H + h];
                O[(size_t) bins[(size_t) i * HD_H + h] * T1 + (pass ? fr[i].off : fr[i].on)] += pass ? -w : w;
            }
    for (int b = 0; b < NE; b++) {
        float * o = &O[(size_t) b * T1];
        double  a = 0;
        for (int t = 0; t < T1; t++) o[t] = (float) (a += o[t]);
    }
    static const float frac[HD_SUS] = { 0.25f, 0.5f, 0.75f, 0.95f };
    for (int i = 0; i < n; i++) {
        int tk[HD_T];
        for (int k = 0; k < HD_K; k++) tk[k] = fr[i].on + HD_KF[k];
        const float dur = (float) (fr[i].off - fr[i].on);
        for (int k = 0; k < HD_SUS; k++) tk[HD_K + k] = fr[i].on + (int) (dur * frac[k]);
        float lE[HD_H][HD_T], mx = -INFINITY;
        for (int k = 0; k < HD_T; k++) tk[k] = std::clamp(tk[k], 0, Tf - 1);
        for (int h = 0; h < HD_H; h++) {
            const int   b = bins[(size_t) i * HD_H + h];
            const float v = valid[(size_t) i * HD_H + h], own = m.pi[h];
            for (int k = 0; k < HD_T; k++) {
                const float act   = (tk[k] >= fr[i].on && tk[k] < fr[i].off) ? 1.0f : 0.0f;
                const float Og    = O[(size_t) b * T1 + tk[k]];
                const float share = own / std::max(Og + own * (1.0f - act), 1e-6f);
                lE[h][k]          = logf(P[(size_t) b * Tf + tk[k]] * std::min(share, 1.0f) + 1e-7f) * v;
                mx                = std::max(mx, lE[h][k]);
            }
        }
        float * o = out + (size_t) i * HD_NF;
        for (int h = 0; h < HD_H; h++)
            for (int k = 0; k < HD_T; k++) o[h * HD_T + k] = std::max(lE[h][k] - mx, -20.0f) / 10.0f;
        for (int h = 0; h < HD_H; h++) o[HD_H * HD_T + h] = valid[(size_t) i * HD_H + h];
    }
}

// Prof: mean and std of every channel at the note's bin and the next one up, over the note -> [n][36]
static void profile(const std::vector<float> & X, int Tf, const std::vector<float> & notes, int n, float * out) {
    std::map<int, std::vector<float>> cs, cs2;  // per (channel, bin): torch's cumsum over time (double, float32 out)
    auto col = [&](int c, int f, bool sq) -> const std::vector<float> & {
        auto & v = (sq ? cs2 : cs)[c * N_BINS + f];
        if (v.empty()) {
            v.resize(Tf + 1);
            double a = 0;
            v[0]     = 0.0f;
            for (int t = 0; t < Tf; t++) {
                const float x = X[((size_t) c * Tf + t) * N_BINS + f];
                v[t + 1]      = (float) (a += sq ? x * x : x);
            }
        }
        return v;
    };
    for (int i = 0; i < n; i++) {
        const Frames fr = note_frames(&notes[(size_t) i * N_META], Tf);
        const float  Ln = (float) std::max(fr.off - fr.on, 1);
        float *      o  = out + (size_t) i * 4 * N_CH;
        for (int off = 0; off < 2; off++) {
            const int fb = std::min(3 * fr.lane + off, N_BINS - 1);
            for (int c = 0; c < N_CH; c++) {
                const std::vector<float> & a = col(c, fb, false), & b = col(c, fb, true);
                const float m1 = (a[fr.off] - a[fr.on]) / Ln, m2 = (b[fr.off] - b[fr.on]) / Ln;
                o[off * 2 * N_CH + c]        = m1;
                o[off * 2 * N_CH + N_CH + c] = std::sqrt(std::max(m2 - m1 * m1, 0.0f) + 1e-4f);
            }
        }
    }
}

struct LaneFrames { int on, off, lane; };

// rasters' frames on the lane trunk's 80 ms grid
static LaneFrames lane_frames(const float * n, int Tq) {
    LaneFrames f;
    f.on   = (int) std::clamp((long long) ((n[0] + 0.5f) * 12.5f), 0LL, (long long) Tq - 1);
    f.off  = (int) std::max(std::min((long long) ((n[1] + 0.5f) * 12.5f), (long long) Tq), (long long) f.on + 1);
    f.lane = std::clamp((int) std::nearbyintf(n[2]) - PITCH_LO, 0, LANES - 1);
    return f;
}

// note rasters -> R [Tq][LANES][4] (onsets, sustain, onset confidence, second-voice onsets)
static void rasters(const std::vector<float> & notes, int n, int Tq, float * R) {
    std::vector<float> sus((size_t) (Tq + 1) * LANES, 0.0f);
    std::fill(R, R + (size_t) Tq * LANES * 4, 0.0f);
    std::vector<LaneFrames> fr(n);
    for (int i = 0; i < n; i++) fr[i] = lane_frames(&notes[(size_t) i * N_META], Tq);
    for (int i = 0; i < n; i++) R[((size_t) fr[i].on * LANES + fr[i].lane) * 4 + 0] += 1.0f;
    for (int i = 0; i < n; i++) sus[(size_t) fr[i].on * LANES + fr[i].lane] += 1.0f;
    for (int i = 0; i < n; i++) sus[(size_t) fr[i].off * LANES + fr[i].lane] += -1.0f;
    for (int i = 0; i < n; i++) {
        const float * nn = &notes[(size_t) i * N_META];
        R[((size_t) fr[i].on * LANES + fr[i].lane) * 4 + 2] += nn[3] > 0.5f ? nn[5] : nn[4];
    }
    for (int i = 0; i < n; i++) R[((size_t) fr[i].on * LANES + fr[i].lane) * 4 + 3] += notes[(size_t) i * N_META + 3] > 0.5f ? 1.0f : 0.0f;
    for (int l = 0; l < LANES; l++) {
        double a = 0;
        for (int t = 0; t < Tq; t++) R[((size_t) t * LANES + l) * 4 + 1] = (float) (a += sus[(size_t) t * LANES + l]);
    }
}

// the stem conv's input as patches: [Tq][LANES][N_CH * 8 * 3], rows c * 24 + kt * 3 + kf (kernel (8, 3), stride (8, 3), pad (0, 1))
static void patches(const std::vector<float> & X, int Tf, int Tq, float * out) {
    constexpr int KP = N_CH * TS * 3;
    for (int t = 0; t < Tq; t++)
        for (int p = 0; p < LANES; p++) {
            float * o = out + ((size_t) t * LANES + p) * KP;
            for (int c = 0; c < N_CH; c++)
                for (int kt = 0; kt < TS; kt++)
                    for (int kf = 0; kf < 3; kf++) {
                        const int f = 3 * p - 1 + kf, tt = t * TS + kt;
                        o[c * TS * 3 + kt * 3 + kf] = (f >= 0 && f < N_BINS && tt < Tf) ? X[((size_t) c * Tf + tt) * N_BINS + f] : 0.0f;
                    }
        }
}

// ---------------------------------------------------------------- graphs
static ggml_tensor * linear(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    return ggml_add(c, transcriber::mm(c, m.get(p + ".w"), x), m.get(p + ".b"));
}
static ggml_tensor * layer_norm(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p, float eps) {
    x = ggml_rms_norm(c, ggml_sub(c, x, ggml_mean(c, x)), eps);
    return ggml_add(c, ggml_mul(c, x, m.get(p + ".w")), m.get(p + ".b"));
}
static ggml_tensor * mlp(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & a, const std::string & b) {
    return linear(c, m, ggml_gelu_erf(c, linear(c, m, x, a)), b);
}

// Multi-head attention over dim 1. q, k, v: [nh * dh, L, N] (strided views allowed) -> [nh * dh, L, N]
static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, int nh, int dh) {
    using transcriber::mm;
    const int64_t L = q->ne[1], N = q->ne[2];
    auto heads = [&](ggml_tensor * x) { return ggml_view_4d(c, x, dh, nh, L, N, dh * sizeof(float), x->nb[1], x->nb[2], 0); };
    ggml_tensor * q4 = ggml_cont(c, ggml_permute(c, heads(q), 0, 2, 1, 3));
    ggml_tensor * k4 = ggml_cont(c, ggml_permute(c, heads(k), 0, 2, 1, 3));
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, heads(v), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_soft_max_ext(c, mm(c, k4, q4), mask, 1.0f / sqrtf((float) dh), 0.0f);
    ggml_tensor * o  = ggml_cont(c, ggml_permute(c, mm(c, vt, kq), 0, 2, 1, 3));
    return ggml_reshape_3d(c, o, (int64_t) nh * dh, L, N);
}

// the lane trunk: patches [216, LANES, Tq, B], rasters [4, LANES, Tq, B] -> Y [LC, LANES, Tq, B]
static ggml_tensor * build_lanes(ggml_context * c, const Model & m, ggml_tensor * pt, ggml_tensor * rs) {
    using transcriber::mm;
    ggml_tensor * y = ggml_add(c, mm(c, m.get("stem.w"), pt), m.get("stem.b"));
    y               = ggml_add(c, y, ggml_add(c, mm(c, m.get("occ.w"), rs), m.get("occ.b")));
    for (int i = 0; i < LDEPTH; i++) {
        const std::string p = "l" + std::to_string(i);
        const int         d = LDIL[i % 4];
        ggml_tensor *     z = transcriber::dwconv(c, m.get(p + ".kt"), y, 1, 0, 3 * d, d);
        z = ggml_add(c, transcriber::dwconv(c, m.get(p + ".kp"), z, 1, PK_HALF, 0, 1), m.get(p + ".bp"));
        z = layer_norm(c, m, z, p + ".norm", 1e-6f);
        z = ggml_gelu_erf(c, ggml_add(c, mm(c, m.get(p + ".w1"), z), m.get(p + ".b1")));
        y = ggml_add(c, y, ggml_add(c, mm(c, m.get(p + ".w2"), z), m.get(p + ".b2")));
        if (LATTN[i]) {  // the lanes of every frame attend to each other
            const std::string a  = "a" + std::to_string(i);
            const int64_t     Tq = y->ne[2], B = y->ne[3];
            ggml_tensor *     h  = ggml_reshape_3d(c, layer_norm(c, m, y, a + ".ln", 1e-5f), LC, LANES, Tq * B);
            ggml_tensor *     qkv = linear(c, m, h, a + ".qkv");
            auto part = [&](int k) { return ggml_view_3d(c, qkv, LC, qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], (size_t) k * LC * sizeof(float)); };
            ggml_tensor * o = linear(c, m, attention(c, part(0), part(1), part(2), m.get(a + ".mask"), LHEADS, LHD), a + ".out");
            y = ggml_add(c, y, ggml_reshape_4d(c, o, LC, LANES, Tq, B));
        }
    }
    return y;
}

struct NoteIn { ggml_tensor * pitch, * scal, * hd, * prof, * read, * s15, * cs, * sn; };
struct NoteOut { ggml_tensor * inst, * emb, * h; };

// one window's notes -> inst [K, n], emb [EMB, n] (before normalisation), h [D, n]
static NoteOut build_notes(ggml_context * c, const Model & m, const NoteIn & in) {
    using transcriber::mm;
    const int64_t n = in.scal->ne[1];
    ggml_tensor * x = ggml_add(c, ggml_get_rows(c, m.get("pitch_emb"), in.pitch), mlp(c, m, in.scal, "feat0", "feat2"));
    ggml_tensor * xa = mlp(c, m, in.hd, "hd0", "hd2");
    xa               = ggml_add(c, xa, mlp(c, m, in.prof, "prof0", "prof2"));
    xa               = ggml_add(c, xa, linear(c, m, in.read, "read"));
    x                = ggml_add(c, x, xa);
    ggml_tensor * f  = ggml_reshape_2d(c, layer_norm(c, m, in.s15, "s15ln", 1e-5f), N_TAP * S15, n);
    x                = ggml_add(c, x, mlp(c, m, f, "s15a", "s15b"));
    x                = layer_norm(c, m, x, "ln_in", 1e-5f);
    for (int i = 0; i < DEPTH; i++) {
        const std::string p   = "b" + std::to_string(i);
        ggml_tensor *     h   = layer_norm(c, m, x, p + ".ln1", 1e-5f);
        ggml_tensor *     qkv = linear(c, m, h, p + ".qkv");
        ggml_tensor *     rot = linear(c, m, h, p + ".rot");
        auto part = [&](ggml_tensor * t, int k) { return ggml_view_3d(c, t, HDIM, HEADS, n, HDIM * sizeof(float), t->nb[1], (size_t) k * D * sizeof(float)); };
        ggml_tensor * q = ggml_add(c, ggml_mul(c, part(qkv, 0), in.cs), ggml_mul(c, part(rot, 0), in.sn));
        ggml_tensor * k = ggml_add(c, ggml_mul(c, part(qkv, 1), in.cs), ggml_mul(c, part(rot, 1), in.sn));
        ggml_tensor * v = ggml_view_3d(c, qkv, D, n, 1, qkv->nb[1], qkv->nb[1] * n, 2 * D * sizeof(float));
        ggml_tensor * a = attention(c, ggml_reshape_3d(c, q, D, n, 1), ggml_reshape_3d(c, k, D, n, 1), v, nullptr, HEADS, HDIM);
        x               = ggml_add(c, x, linear(c, m, ggml_reshape_2d(c, a, D, n), p + ".o"));
        x               = ggml_add(c, x, mlp(c, m, layer_norm(c, m, x, p + ".ln2", 1e-5f), p + ".ff0", p + ".ff2"));
    }
    x = layer_norm(c, m, x, "ln_out", 1e-5f);
    return { linear(c, m, x, "inst"), linear(c, m, x, "emb"), x };
}

static void report_placement(Model & m, ggml_cgraph * gf, const char * label) {
    int nodes = 0, off = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
            t->op == GGML_OP_TRANSPOSE)
            continue;
        nodes++;
        if (ggml_backend_sched_get_tensor_backend(m.sched, t) != m.be) off++;
    }
    fprintf(stderr, "[dettato] PLACEMENT %s nodes=%d splits=%d off %s=%d\n", label, nodes, ggml_backend_sched_get_n_splits(m.sched),
            ggml_backend_name(m.be), off);
}

static void alloc_and_check(Model & m, ggml_cgraph * gf, const char * label) {
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) {
        fprintf(stderr, "[dettato] %s graph allocation failed\n", label);
        exit(1);
    }
}

// B windows: patches [B][Tq][LANES][216], rasters [B][Tq][LANES][4] -> Y [B][Tq][LANES][LC]
static void run_lanes(Model & m, const float * pt, const float * rs, int Tq, int B, float * Y) {
    ggml_init_params ip = { m.meta.size(), m.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    tp = ggml_new_tensor_4d(c, GGML_TYPE_F32, N_CH * TS * 3, LANES, Tq, B);
    ggml_tensor *    tr = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, LANES, Tq, B);
    ggml_set_input(tp);
    ggml_set_input(tr);
    ggml_tensor * y = build_lanes(c, m, tp, tr);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(m.sched);
    ggml_backend_sched_set_tensor_backend(m.sched, tp, m.be);
    ggml_backend_sched_set_tensor_backend(m.sched, tr, m.be);
    alloc_and_check(m, gf, "grouper lanes");
    ggml_backend_tensor_set(tp, pt, 0, ggml_nbytes(tp));
    ggml_backend_tensor_set(tr, rs, 0, ggml_nbytes(tr));
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(m.sched, gf))) {
        fprintf(stderr, "[dettato] grouper lanes graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(y, Y, 0, ggml_nbytes(y));
    if (!m.shown[0]) {
        report_placement(m, gf, "grouper-lanes");
        m.shown[0] = true;
    }
    ggml_free(c);
}

// the lane trunk read at every note: Y at its onset frame and averaged over its frames -> [n][256]
static void lane_read(const float * Y, int Tq, const std::vector<float> & notes, int n, float * out) {
    std::map<int, std::vector<float>> cs;  // per lane: [Tq + 1][LC] cumsum over time (double, float32 out)
    for (int i = 0; i < n; i++) {
        const LaneFrames fr = lane_frames(&notes[(size_t) i * N_META], Tq);
        auto &           v  = cs[fr.lane];
        if (v.empty()) {
            v.assign((size_t) (Tq + 1) * LC, 0.0f);
            std::vector<double> a(LC, 0.0);
            for (int t = 0; t < Tq; t++)
                for (int ch = 0; ch < LC; ch++) v[(size_t) (t + 1) * LC + ch] = (float) (a[ch] += Y[((size_t) t * LANES + fr.lane) * LC + ch]);
        }
        const float   len = (float) std::max(fr.off - fr.on, 1);
        float *       o   = out + (size_t) i * 2 * LC;
        const float * yo  = &Y[((size_t) fr.on * LANES + fr.lane) * LC];
        for (int ch = 0; ch < LC; ch++) {
            o[ch]      = yo[ch];
            o[LC + ch] = (v[(size_t) fr.off * LC + ch] - v[(size_t) fr.on * LC + ch]) / len;
        }
    }
}

// one window's tokens and transformer: host inputs -> probs [n][K] (softmax), emb [n][EMB] (unit), h [n][D]
static void run_notes(Model & m, const std::vector<float> & notes, int n, const float * hd, const float * prof, const float * read,
                      const float * s15, std::vector<float> & probs, std::vector<float> & emb, std::vector<float> & hid) {
    std::vector<int32_t> pitch(n);
    std::vector<float>   scal((size_t) n * 11), cs((size_t) n * HDIM), sn((size_t) n * HDIM);
    float                freq[HDIM / 2];
    for (int j = 0; j < HDIM / 2; j++) freq[j] = 1.0f / powf(10000.0f, (float) j / (float) (HDIM / 2));
    for (int i = 0; i < n; i++) {
        const float * nn = &notes[(size_t) i * N_META];
        pitch[i]         = std::clamp((int) std::nearbyintf(nn[2]), 0, 127);
        float * s        = &scal[(size_t) i * 11];
        s[0]             = logf(std::max(nn[1] - nn[0], 0.01f));
        for (int k = 0; k < 9; k++) s[1 + k] = nn[3 + k];
        s[10]           = nn[2] / 64.0f - 1.0f;
        const float pos = nn[0] * 10.0f;
        for (int d = 0; d < HDIM; d++) {
            const float ang = pos * freq[d % (HDIM / 2)];
            cs[(size_t) i * HDIM + d] = cosf(ang);
            sn[(size_t) i * HDIM + d] = sinf(ang);
        }
    }
    ggml_init_params ip = { m.meta.size(), m.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    NoteIn in;
    in.pitch = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    in.scal  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 11, n);
    in.hd    = ggml_new_tensor_2d(c, GGML_TYPE_F32, HD_NF, n);
    in.prof  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4 * N_CH, n);
    in.read  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * LC, n);
    in.s15   = ggml_new_tensor_3d(c, GGML_TYPE_F32, S15, N_TAP, n);
    in.cs    = ggml_new_tensor_3d(c, GGML_TYPE_F32, HDIM, 1, n);
    in.sn    = ggml_new_tensor_3d(c, GGML_TYPE_F32, HDIM, 1, n);
    ggml_tensor * ins[] = { in.pitch, in.scal, in.hd, in.prof, in.read, in.s15, in.cs, in.sn };
    for (ggml_tensor * t : ins) ggml_set_input(t);
    NoteOut o = build_notes(c, m, in);
    for (ggml_tensor * t : { o.inst, o.emb, o.h }) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    ggml_backend_sched_reset(m.sched);
    for (ggml_tensor * t : ins) ggml_backend_sched_set_tensor_backend(m.sched, t, m.be);
    alloc_and_check(m, gf, "grouper notes");
    const void * src[] = { pitch.data(), scal.data(), hd, prof, read, s15, cs.data(), sn.data() };
    for (int k = 0; k < 8; k++) ggml_backend_tensor_set(ins[k], src[k], 0, ggml_nbytes(ins[k]));
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(m.sched, gf))) {
        fprintf(stderr, "[dettato] grouper notes graph compute failed\n");
        exit(1);
    }
    std::vector<float> logits((size_t) n * K);
    probs.resize((size_t) n * K);
    emb.resize((size_t) n * EMB);
    hid.resize((size_t) n * D);
    ggml_backend_tensor_get(o.inst, logits.data(), 0, ggml_nbytes(o.inst));
    ggml_backend_tensor_get(o.emb, emb.data(), 0, ggml_nbytes(o.emb));
    ggml_backend_tensor_get(o.h, hid.data(), 0, ggml_nbytes(o.h));
    if (!m.shown[1]) {
        report_placement(m, gf, "grouper-notes");
        m.shown[1] = true;
    }
    ggml_free(c);
    for (int i = 0; i < n; i++) {  // torch.softmax and F.normalize, float32
        float * l = &logits[(size_t) i * K], mx = -INFINITY, sum = 0.0f;
        for (int k = 0; k < K; k++) mx = std::max(mx, l[k]);
        for (int k = 0; k < K; k++) sum += (probs[(size_t) i * K + k] = expf(l[k] - mx));
        for (int k = 0; k < K; k++) probs[(size_t) i * K + k] /= sum;
        float * e  = &emb[(size_t) i * EMB];
        double  ss = 0;
        for (int k = 0; k < EMB; k++) ss += (double) e[k] * e[k];
        const float nrm = std::max((float) std::sqrt(ss), 1e-12f);
        for (int k = 0; k < EMB; k++) e[k] /= nrm;
    }
}

// ---------------------------------------------------------------- prediction over the song (Grouper.predict)
struct Prediction32 {  // the capture's float32 / float16 roundings, before the decode's float64
    std::vector<float>       probs, emb;   // [n][K], [n][EMB]
    std::vector<ggml_fp16_t> h;            // [n][D]
    std::vector<double>      win_t0;
    std::vector<int>         win_count, win_sel;
    std::vector<ggml_fp16_t> win_emb, win_probs;
};

// y: the (retuned) 16 kHz audio the transcriber heard; f: the note features. progress(done, total) per window batch.
template <typename Progress>
static Prediction32 predict(Model & m, const std::vector<float> & y, const Features & f, Progress progress) {
    using transcriber::SR;
    const double W = m.window_s, hop = W * m.window_hop;
    const int    n = f.n;
    Prediction32 out;
    std::vector<double> acc((size_t) n * K, 0.0), eacc((size_t) n * EMB, 0.0), wsum(n, 0.0), best(n, -1.0);
    out.h.assign((size_t) n * D, ggml_fp32_to_fp16(0.0f));
    std::vector<float> on(n);
    for (int i = 0; i < n; i++) on[i] = f.meta[(size_t) i * N_META];
    const std::vector<Win> jobs = song_windows(on, W, hop, m.max_notes);
    // _Song: the audio in 16-bit steps
    std::vector<int16_t> a16(y.size());
    for (size_t i = 0; i < y.size(); i++) a16[i] = (int16_t) std::clamp(std::nearbyintf(y[i] * 32768.0f), -32768.0f, 32767.0f);
    const int Ls = (int) std::nearbyint((PAD0 + W + PAD1) * SR), Tf = Ls / transcriber::HOP, Tq = (Tf - TS) / TS + 1;
    constexpr int KP = N_CH * TS * 3;
    for (size_t b0 = 0; b0 < jobs.size(); b0 += m.batch) {
        const int B = (int) std::min((size_t) m.batch, jobs.size() - b0);
        struct Item { std::vector<int> ids; std::vector<float> notes, X, P; };
        std::vector<Item>  items(B);
        std::vector<float> pt((size_t) B * Tq * LANES * KP), rs((size_t) B * Tq * LANES * 4), Y((size_t) B * Tq * LANES * LC);
        double             t0ms = transcriber::now_ms();
        std::vector<std::thread> pool;
        for (int k = 0; k < B; k++) pool.emplace_back([&, k]() {  // _Song.window (a window with under 4 notes repeats them up to 4)
            const Win & w  = jobs[b0 + k];
            Item &      it = items[k];
            it.ids         = w.sel;
            for (size_t j = 0; it.ids.size() < 4; j++) it.ids.push_back(w.sel[j % w.sel.size()]);
            it.notes.resize(it.ids.size() * N_META);
            for (size_t j = 0; j < it.ids.size(); j++) {
                float * d = &it.notes[j * N_META];
                std::copy_n(&f.meta[(size_t) it.ids[j] * N_META], N_META, d);
                d[0] = d[0] - (float) w.t0;
                d[1] = std::min(d[1] - (float) w.t0, (float) (W + PAD1));
            }
            std::vector<float> audio(Ls, 0.0f);
            const long long    s0 = (long long) std::nearbyint((w.t0 - PAD0) * SR);
            for (long long i = std::max(s0, 0LL); i < std::min(s0 + Ls, (long long) a16.size()); i++) audio[i - s0] = a16[i] / 32768.0f;
            front_end(m, audio, Tf, it.X, it.P);
            patches(it.X, Tf, Tq, &pt[(size_t) k * Tq * LANES * KP]);
            rasters(it.notes, (int) it.ids.size(), Tq, &rs[(size_t) k * Tq * LANES * 4]);
        });
        for (auto & th : pool) th.join();
        double t1ms = transcriber::now_ms();
        run_lanes(m, pt.data(), rs.data(), Tq, B, Y.data());
        double t2ms = transcriber::now_ms();
        for (int k = 0; k < B; k++) {
            const Win & w  = jobs[b0 + k];
            Item &      it = items[k];
            const int   nw = (int) it.ids.size();
            std::vector<float> hd((size_t) nw * HD_NF), pr((size_t) nw * 4 * N_CH), rd((size_t) nw * 2 * LC), s15((size_t) nw * N_TAP * S15);
            hdemix(m, it.P, Tf, it.notes, nw, hd.data());
            profile(it.X, Tf, it.notes, nw, pr.data());
            lane_read(&Y[(size_t) k * Tq * LANES * LC], Tq, it.notes, nw, rd.data());
            // the taps (int8 x float16 scale, stored float16 by the collate), standardised over the window's notes
            for (int j = 0; j < nw; j++)
                for (int t = 0; t < N_TAP; t++) {
                    const float sc = ggml_fp16_to_fp32(f.fsc[(size_t) it.ids[j] * N_TAP + t]);
                    for (int ch = 0; ch < S15; ch++) {
                        const float v = (float) f.feat[((size_t) it.ids[j] * N_TAP + t) * S15 + ch] * sc;
                        s15[((size_t) j * N_TAP + t) * S15 + ch] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
                    }
                }
            for (int t = 0; t < N_TAP; t++)
                for (int ch = 0; ch < S15; ch++) {
                    double s = 0, q = 0;
                    for (int j = 0; j < nw; j++) s += s15[((size_t) j * N_TAP + t) * S15 + ch];
                    const float mu = (float) (s / nw);
                    for (int j = 0; j < nw; j++) {
                        const float d = s15[((size_t) j * N_TAP + t) * S15 + ch] - mu;
                        q += (double) d * d;
                    }
                    const float sd = std::max((float) std::sqrt(q / nw), 1e-3f);
                    for (int j = 0; j < nw; j++) {
                        float & v = s15[((size_t) j * N_TAP + t) * S15 + ch];
                        v         = (v - mu) / sd;
                    }
                }
            std::vector<float> p, e, hh;
            run_notes(m, it.notes, nw, hd.data(), pr.data(), rd.data(), s15.data(), p, e, hh);
            // average into the song: weights towards the window centre, h from the most central window
            const int mm = (int) w.sel.size();
            out.win_t0.push_back(w.t0);
            out.win_count.push_back(mm);
            for (int j = 0; j < mm; j++) {
                const int   i  = w.sel[j];
                const float wt = std::clamp(1.0f - std::fabs((on[i] - (float) w.t0) / (float) W - 0.5f) * 2.0f, 0.05f, 1.0f);
                for (int q = 0; q < K; q++) acc[(size_t) i * K + q] += p[(size_t) j * K + q] * wt;
                for (int q = 0; q < EMB; q++) eacc[(size_t) i * EMB + q] += e[(size_t) j * EMB + q] * wt;
                wsum[i] += wt;
                if (wt > best[i]) {
                    for (int q = 0; q < D; q++) out.h[(size_t) i * D + q] = ggml_fp32_to_fp16(hh[(size_t) j * D + q]);
                    best[i] = wt;
                }
                out.win_sel.push_back(i);
                for (int q = 0; q < EMB; q++) out.win_emb.push_back(ggml_fp32_to_fp16(e[(size_t) j * EMB + q]));
                for (int q = 0; q < K; q++) out.win_probs.push_back(ggml_fp32_to_fp16(p[(size_t) j * K + q]));
            }
        }
        m.ms_front += t1ms - t0ms;
        m.ms_lanes += t2ms - t1ms;
        m.ms_notes += transcriber::now_ms() - t2ms;
        progress((int) (b0 + B), (int) jobs.size());
    }
    out.probs.resize((size_t) n * K);
    out.emb.resize((size_t) n * EMB);
    for (int i = 0; i < n; i++) {
        const double ws = std::max(wsum[i], 1e-9);
        for (int q = 0; q < K; q++) out.probs[(size_t) i * K + q] = (float) (acc[(size_t) i * K + q] / ws);
        double nrm = 0;
        for (int q = 0; q < EMB; q++) nrm += eacc[(size_t) i * EMB + q] * eacc[(size_t) i * EMB + q];
        nrm = std::max(std::sqrt(nrm), 1e-9);
        for (int q = 0; q < EMB; q++) out.emb[(size_t) i * EMB + q] = (float) (eacc[(size_t) i * EMB + q] / nrm);
    }
    return out;
}

// the decode's view of a prediction (grouper-decode.h)
static Prediction to_decode(const Model & m, const Prediction32 & p, int n) {
    Prediction d;
    d.window_seconds = m.window_s;
    d.probability.assign(n, Vec(K));
    d.embedding.assign(n, Vec(EMB));
    d.hidden.assign(n, Vec(D));
    for (int i = 0; i < n; i++) {
        for (int q = 0; q < K; q++) d.probability[i][q] = p.probs[(size_t) i * K + q];
        for (int q = 0; q < EMB; q++) d.embedding[i][q] = p.emb[(size_t) i * EMB + q];
        for (int q = 0; q < D; q++) d.hidden[i][q] = ggml_fp16_to_fp32(p.h[(size_t) i * D + q]);
    }
    size_t pos = 0;
    for (size_t w = 0; w < p.win_t0.size(); w++) {
        Window win;
        win.start = p.win_t0[w];
        for (int j = 0; j < p.win_count[w]; j++, pos++) {
            win.indices.push_back(p.win_sel[pos]);
            Vec e(EMB), pr(K);
            for (int q = 0; q < EMB; q++) e[q] = ggml_fp16_to_fp32(p.win_emb[pos * EMB + q]);
            for (int q = 0; q < K; q++) pr[q] = ggml_fp16_to_fp32(p.win_probs[pos * K + q]);
            win.embedding.push_back(std::move(e));
            win.probability.push_back(std::move(pr));
        }
        d.windows.push_back(std::move(win));
    }
    return d;
}

// ---------------------------------------------------------------- tracks (Grouper.group, _tracks)
struct TrackNote { double onset, offset; int pitch, velocity; };
struct Track {
    int                    instrument;  // index into Model::names / programs
    std::vector<TrackNote> notes;
};

// Groups -> tracks, the largest first; same-pitch notes within 30 ms in one track become one note; times x r.
static std::vector<Track> tracks(const Features & f, const std::vector<Group> & groups, double r) {
    std::vector<double> on(f.n), off(f.n);
    std::vector<int>    pitch(f.n), vel(f.n);
    for (int i = 0; i < f.n; i++) {
        const float * mt = &f.meta[(size_t) i * N_META];
        on[i]            = (double) mt[0] * r;
        off[i]           = (double) mt[1] * r;
        pitch[i]         = (int) std::nearbyint(mt[2]);
        vel[i]           = transcriber::velocity(mt[3] > 0.5f ? mt[5] : mt[4]);
    }
    std::vector<int> order(groups.size());
    for (size_t g = 0; g < groups.size(); g++) order[g] = (int) g;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return groups[a].notes.size() > groups[b].notes.size(); });
    std::vector<Track> out;
    for (int g : order) {
        std::vector<int> o = groups[g].notes;
        std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return pitch[a] != pitch[b] ? pitch[a] < pitch[b] : on[a] < on[b]; });
        std::vector<int> keep;
        int              last = -1;
        for (int i : o) {
            if (last >= 0 && pitch[i] == pitch[last] && on[i] - on[last] <= DEDUP) continue;
            keep.push_back(i);
            last = i;
        }
        std::stable_sort(keep.begin(), keep.end(), [&](int a, int b) { return on[a] < on[b]; });
        Track t{ groups[g].instrument, {} };
        for (int i : keep) t.notes.push_back({ on[i], std::max(off[i], on[i] + 0.02), pitch[i], vel[i] });
        out.push_back(std::move(t));
    }
    return out;
}

}  // namespace dettato::grouper
