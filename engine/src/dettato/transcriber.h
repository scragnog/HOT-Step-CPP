#pragma once
// dettato/transcriber.h: dettato's transcriber network (transcriber/model.py,
// LaneNet, 15.3M parameters) on any ggml backend, and the windowed run over a
// recording (transcriber/core.py, Transcriber.run). See LICENSE here.
//
// Weights come from upstream's model.safetensors and config.json unmodified.
// At load:
//   - BatchNorm folds into the convolution before it (stem n1, n2, n3) or into
//     the pointwise conv after it (the lane blocks' norm). The stem's inorm and
//     up_norm stay as a per-channel scale and shift: each sits before a
//     zero-padded convolution, where a fold would change the borders.
//   - Each lane block's gamma folds into pw2 and each attention gamma into its
//     output projection.
//   - PitchMix becomes a 63-tap depthwise conv along pitch (offsets -31..31)
//     with the residual folded into the centre tap. The time conv's bias,
//     passed through PitchMix's zero-padded taps, becomes a [C, P] bias.
//   - The time-axis attention's rotary embedding uses host cos/sin tables in
//     torch's float32 steps and a row-permuted copy of the q and k weights
//     whose output is rotate_half(q), rotate_half(k).
//   - Each bidirectional GRU gets a block-diagonal W_hh with gate-major rows
//     [r_f r_r z_f z_r n_f n_r] over the state [h_f; h_r], so both directions
//     advance in one matmul per step; b_hr and b_hz fold into the input gates.
// Two graphs: the stem (HCQT -> 368 x 400 x 88 per window, one window at a time) and
// the trunk (lane blocks, axial attention, GRUs unrolled over 200 steps, the
// 20 ms refine blocks and the heads). Every matmul carries
// GGML_HINT_STRICT_F32. LayerNorm is two-pass, rms_norm(x - mean(x)).

#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "backend.h"
#include "dettato/transcriber-host.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "safetensors.h"
#include "yyjson.h"

namespace dettato::transcriber {

constexpr int C = 368, MID = 64, C2 = 2 * C, NH = 8, DH = C / NH, HID = 192, DEPTH = 15, N_REFINE = 2, KT = 7;
constexpr int N_PMIX = 14, PK = 63, PK_HALF = PK / 2, T_TRUNK = FW / 2;
constexpr int GRAPH_NODES = 32768;
static const int PMIX_OFF[N_PMIX] = { -2, -1, 1, 2, -12, 12, -19, 19, -24, 24, -28, 28, -31, 31 };
static const int DILATION[4]      = { 1, 2, 4, 8 };

struct Model {
    BackendPair                          bp    = {};
    ggml_backend_sched_t                 sched = nullptr;
    ggml_context *                       ctx   = nullptr;
    ggml_backend_buffer_t                buf   = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::vector<uint8_t>                 meta;
    Front                                front;
    DecodeTh                             th;
    int                                  stem_batch = 1, trunk_batch = 8;  // stem im2col: ~0.5 GB per window
    bool                                 skip_gru = false;  // profiling only: the GRUs become identities
    bool                                 shown[2] = { false, false };
    double ms_hcqt = 0, ms_stem = 0, ms_trunk = 0, ms_accum = 0;
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            fprintf(stderr, "[dettato] missing tensor %s\n", n.c_str());
            exit(1);
        }
        return it->second;
    }
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- config
static bool read_config(Model & m, const std::string & path) {
    yyjson_doc * doc = yyjson_read_file(path.c_str(), 0, nullptr, nullptr);
    if (!doc) {
        fprintf(stderr, "[dettato] cannot read %s\n", path.c_str());
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * md   = yyjson_obj_get(root, "model");
    yyjson_val * th   = yyjson_obj_get(root, "thresholds");
    auto         num  = [](yyjson_val * o, const char * k, double fb) {
        yyjson_val * v = o ? yyjson_obj_get(o, k) : nullptr;
        return v && yyjson_is_num(v) ? yyjson_get_num(v) : fb;
    };
    bool ok = md && th && yyjson_equals_str(yyjson_obj_get(root, "stage"), "transcriber");
    // the architecture this port implements; anything else is refused
    ok = ok && num(md, "c", 0) == C && num(md, "depth", 0) == DEPTH && num(md, "exp", 0) == 2 && num(md, "kt", 0) == KT &&
         num(md, "mid", 0) == MID && num(md, "t_stride", 0) == 2 && num(md, "attn_heads", 0) == NH &&
         num(md, "gru_hidden", 0) == HID && num(md, "n_fft", 0) == N_FFT && num(md, "refine", 0) == N_REFINE &&
         num(md, "n_lanes", 0) == N_KEYS && num(root, "sample_rate", 0) == SR;
    yyjson_val * pm = md ? yyjson_obj_get(md, "pmix") : nullptr;
    ok = ok && pm && yyjson_arr_size(pm) == N_PMIX;
    for (int i = 0; ok && i < N_PMIX; i++) ok = yyjson_get_num(yyjson_arr_get(pm, i)) == PMIX_OFF[i];
    if (ok) {
        m.th.on_th         = num(th, "on_th", 0.25);
        m.th.off_th        = num(th, "off_th", 0.3);
        m.th.fr_th         = num(th, "fr_th", 0.4);
        m.th.fr_low_frames = (int) num(th, "fr_low_frames", 2);
        m.th.on_nms        = (int) num(th, "on_nms", 0);
        m.th.v2_th         = num(th, "v2_th", 0.3);
    } else {
        fprintf(stderr, "[dettato] %s: not the dettato 2.0 transcriber configuration\n", path.c_str());
    }
    yyjson_doc_free(doc);
    return ok;
}

// ---------------------------------------------------------------- weights
struct Loader {
    STFile                st = {};
    bool                  ok = true;
    std::set<std::string> used;
    const float * get(const std::string & name, std::vector<int64_t> shape) {
        used.insert(name);
        const STEntry * e = st_find(st, name.c_str());
        bool good = e && e->dtype == "F32" && e->n_dims == (int) shape.size();
        for (size_t i = 0; good && i < shape.size(); i++) good = e->shape[i] == shape[i];
        if (!good) {
            fprintf(stderr, "[dettato] %s: missing or not F32 of the expected shape\n", name.c_str());
            ok = false;
            static std::vector<float> zero(1 << 20);
            return zero.data();
        }
        return (const float *) st_data(st, *e);
    }
};

struct BN { std::vector<float> s, t; };

static BN batch_norm(Loader & L, const std::string & p, int n) {
    const float * w = L.get(p + ".weight", { n }), * b = L.get(p + ".bias", { n });
    const float * mu = L.get(p + ".running_mean", { n }), * var = L.get(p + ".running_var", { n });
    BN r;
    r.s.resize(n);
    r.t.resize(n);
    for (int i = 0; i < n; i++) {
        r.s[i] = (float) ((double) w[i] / std::sqrt((double) var[i] + 1e-5));
        r.t[i] = (float) ((double) b[i] - (double) mu[i] * r.s[i]);
    }
    return r;
}

static bool load_model(Model & m, const std::string & dir) {
    if (!read_config(m, dir + "/config.json")) return false;
    Loader L;
    if (!st_open(&L.st, (dir + "/model.safetensors").c_str())) return false;

    std::vector<std::pair<std::string, std::pair<std::vector<int64_t>, std::vector<float>>>> pend;
    auto add = [&](const std::string & n, std::vector<int64_t> ne, std::vector<float> v) {
        pend.push_back({ n, { std::move(ne), std::move(v) } });
    };
    auto copy = [](const float * p, size_t n) { return std::vector<float>(p, p + n); };
    // depthwise kernels channel-fastest, [taps][C], as the CWHN conv reads them (dwconv below)
    auto chan_last = [](const float * p, int taps) {
        std::vector<float> v((size_t) taps * C);
        for (int c = 0; c < C; c++)
            for (int k = 0; k < taps; k++) v[(size_t) k * C + c] = p[(size_t) c * taps + k];
        return v;
    };

    // ---- stem
    {
        BN in = batch_norm(L, "stem.inorm", N_CH);
        add("stem.in_s", { 1, 1, N_CH, 1 }, in.s);
        add("stem.in_t", { 1, 1, N_CH, 1 }, in.t);
        auto conv = [&](const std::string & p, const std::string & bn, int ci, int co) {
            const float * w = L.get(p + ".weight", { co, ci, 3, 3 }), * b = L.get(p + ".bias", { co });
            BN            n = batch_norm(L, bn, co);
            std::vector<float> wf((size_t) co * ci * 9), bf(co);
            for (int o = 0; o < co; o++) {
                for (int i = 0; i < ci * 9; i++) wf[(size_t) o * ci * 9 + i] = w[(size_t) o * ci * 9 + i] * n.s[o];
                bf[o] = b[o] * n.s[o] + n.t[o];
            }
            add(p + ".w", { 3, 3, ci, co }, wf);
            add(p + ".b", { 1, 1, co, 1 }, bf);
        };
        conv("stem.c1", "stem.n1", N_CH, MID);
        conv("stem.c2", "stem.n2", MID, MID);
        conv("stem.down", "stem.n3", MID, C);
    }
    // ---- trunk
    add("down.k", { C, 1, 1, 3 }, chan_last(L.get("down.weight", { C, 1, 3, 1 }), 3));
    add("down.b", { C }, copy(L.get("down.bias", { C }), C));
    {
        BN u = batch_norm(L, "up_norm", C);
        add("up.s", { C }, u.s);
        add("up.t", { C }, u.t);
    }
    auto lane_block = [&](const std::string & p) {
        const float * kt = L.get(p + ".dw_t.weight", { C, 1, KT, 1 }), * bt = L.get(p + ".dw_t.bias", { C });
        const float * taps = L.get(p + ".pmix.taps", { C, N_PMIX });
        BN            n   = batch_norm(L, p + ".norm", C);
        const float * w1  = L.get(p + ".pw1.weight", { C2, C, 1, 1 }), * b1 = L.get(p + ".pw1.bias", { C2 });
        const float * w2  = L.get(p + ".pw2.weight", { C, C2, 1, 1 }), * b2 = L.get(p + ".pw2.bias", { C });
        const float * g   = L.get(p + ".gamma", { C });
        add(p + ".kt", { C, 1, 1, KT }, chan_last(kt, KT));
        std::vector<float> kp((size_t) C * PK, 0.0f), bp((size_t) C * N_KEYS);
        for (int c = 0; c < C; c++) {
            kp[(size_t) c * PK + PK_HALF] = 1.0f;
            for (int k = 0; k < N_PMIX; k++) kp[(size_t) c * PK + PK_HALF + PMIX_OFF[k]] = taps[c * N_PMIX + k];
            for (int i = 0; i < N_KEYS; i++) {
                double s = 1.0;
                for (int k = 0; k < N_PMIX; k++)
                    if (i + PMIX_OFF[k] >= 0 && i + PMIX_OFF[k] < N_KEYS) s += taps[c * N_PMIX + k];
                bp[(size_t) i * C + c] = (float) (bt[c] * s);
            }
        }
        add(p + ".kp", { C, 1, PK, 1 }, chan_last(kp.data(), PK));
        add(p + ".bp", { C, N_KEYS, 1, 1 }, bp);
        // pw1(norm(z)) = (W1 diag(s)) z + (W1 t + b1)
        std::vector<float> w1f((size_t) C2 * C), b1f(C2);
        for (int o = 0; o < C2; o++) {
            double acc = b1[o];
            for (int i = 0; i < C; i++) {
                w1f[(size_t) o * C + i] = w1[(size_t) o * C + i] * n.s[i];
                acc += (double) w1[(size_t) o * C + i] * n.t[i];
            }
            b1f[o] = (float) acc;
        }
        add(p + ".w1", { C, C2 }, w1f);
        add(p + ".b1", { C2 }, b1f);
        std::vector<float> w2f((size_t) C * C2), b2f(C);
        for (int o = 0; o < C; o++) {
            for (int i = 0; i < C2; i++) w2f[(size_t) o * C2 + i] = w2[(size_t) o * C2 + i] * g[o];
            b2f[o] = b2[o] * g[o];
        }
        add(p + ".w2", { C2, C }, w2f);
        add(p + ".b2", { C }, b2f);
    };
    auto layer_norm_w = [&](const std::string & p) {
        add(p + ".ln.w", { C }, copy(L.get(p + ".ln.weight", { C }), C));
        add(p + ".ln.b", { C }, copy(L.get(p + ".ln.bias", { C }), C));
    };
    auto attn = [&](const std::string & p, bool pitch) {
        layer_norm_w(p);
        const float * wq = L.get(p + ".qkv.weight", { 3 * C, C }), * bq = L.get(p + ".qkv.bias", { 3 * C });
        add(p + ".qkv.w", { C, 3 * C }, copy(wq, (size_t) 3 * C * C));
        add(p + ".qkv.b", { 3 * C }, copy(bq, 3 * C));
        const float * wo = L.get(p + ".out.weight", { C, C }), * bo = L.get(p + ".out.bias", { C });
        const float * g  = L.get(p + ".gamma", { C });
        std::vector<float> wof((size_t) C * C), bof(C);
        for (int o = 0; o < C; o++) {
            for (int i = 0; i < C; i++) wof[(size_t) o * C + i] = wo[(size_t) o * C + i] * g[o];
            bof[o] = bo[o] * g[o];
        }
        add(p + ".out.w", { C, C }, wof);
        add(p + ".out.b", { C }, bof);
        if (pitch) {  // mask[h][q][k] = rel_bias[h][k - q + 127]
            const float * rb = L.get(p + ".rel_bias", { NH, 255 });
            std::vector<float> mk((size_t) NH * N_KEYS * N_KEYS);
            for (int h = 0; h < NH; h++)
                for (int q = 0; q < N_KEYS; q++)
                    for (int k = 0; k < N_KEYS; k++) mk[((size_t) h * N_KEYS + q) * N_KEYS + k] = rb[h * 255 + k - q + 127];
            add(p + ".mask", { N_KEYS, N_KEYS, NH, 1 }, mk);
        } else {  // rows giving rotate_half(q), rotate_half(k): out[d] = -x[d + 23] (d < 23), x[d - 23] (d >= 23)
            std::vector<float> wr((size_t) 2 * C * C), br(2 * C);
            const int half = DH / 2;
            for (int part = 0; part < 2; part++)
                for (int hd = 0; hd < NH; hd++)
                    for (int d = 0; d < DH; d++) {
                        const int  src = part * C + hd * DH + (d < half ? d + half : d - half);
                        const int  dst = part * C + hd * DH + d;
                        const float sg = d < half ? -1.0f : 1.0f;
                        for (int i = 0; i < C; i++) wr[(size_t) dst * C + i] = sg * wq[(size_t) src * C + i];
                        br[dst] = sg * bq[src];
                    }
            add(p + ".rot.w", { C, 2 * C }, wr);
            add(p + ".rot.b", { 2 * C }, br);
        }
    };
    auto gru = [&](const std::string & p) {
        layer_norm_w(p);
        const std::string g = p + ".gru.";
        const float * wif = L.get(g + "weight_ih_l0", { 3 * HID, C }), * wir = L.get(g + "weight_ih_l0_reverse", { 3 * HID, C });
        const float * whf = L.get(g + "weight_hh_l0", { 3 * HID, HID }), * whr = L.get(g + "weight_hh_l0_reverse", { 3 * HID, HID });
        const float * bif = L.get(g + "bias_ih_l0", { 3 * HID }), * bir = L.get(g + "bias_ih_l0_reverse", { 3 * HID });
        const float * bhf = L.get(g + "bias_hh_l0", { 3 * HID }), * bhr = L.get(g + "bias_hh_l0_reverse", { 3 * HID });
        add(p + ".wih_f", { C, 3 * HID }, copy(wif, (size_t) 3 * HID * C));
        add(p + ".wih_r", { C, 3 * HID }, copy(wir, (size_t) 3 * HID * C));
        std::vector<float> bf(3 * HID), br(3 * HID);
        for (int i = 0; i < 3 * HID; i++) {  // b_hr, b_hz into the input gates; b_hn stays in the step
            bf[i] = i < 2 * HID ? bif[i] + bhf[i] : bif[i];
            br[i] = i < 2 * HID ? bir[i] + bhr[i] : bir[i];
        }
        add(p + ".bih_f", { 3 * HID }, bf);
        add(p + ".bih_r", { 3 * HID }, br);
        std::vector<float> whh((size_t) 6 * HID * 2 * HID, 0.0f), bhn(2 * HID);
        for (int gate = 0; gate < 3; gate++)
            for (int dir = 0; dir < 2; dir++)
                for (int i = 0; i < HID; i++) {
                    const float * src = (dir ? whr : whf) + (size_t) (gate * HID + i) * HID;
                    float *       dst = &whh[(size_t) (gate * 2 * HID + dir * HID + i) * 2 * HID + dir * HID];
                    memcpy(dst, src, HID * sizeof(float));
                }
        for (int i = 0; i < HID; i++) {
            bhn[i]       = bhf[2 * HID + i];
            bhn[HID + i] = bhr[2 * HID + i];
        }
        add(p + ".whh", { 2 * HID, 6 * HID }, whh);
        add(p + ".bhn", { 2 * HID }, bhn);
        add(p + ".out.w", { 2 * HID, C }, copy(L.get(p + ".out.weight", { C, 2 * HID }), (size_t) 2 * HID * C));
        add(p + ".out.b", { C }, copy(L.get(p + ".out.bias", { C }), C));
    };
    int li = 0;
    for (int i = 0; i < DEPTH; i++) {
        lane_block("layers." + std::to_string(li++));
        if ((i + 1) % 5 == 0) {
            attn("layers." + std::to_string(li++), true);
            attn("layers." + std::to_string(li++), false);
            gru("layers." + std::to_string(li++));
        }
    }
    for (int r = 0; r < N_REFINE; r++) lane_block("refine." + std::to_string(r));
    add("head.w", { C, N_HEADS_OUT }, copy(L.get("head.weight", { N_HEADS_OUT, C, 1, 1 }), (size_t) N_HEADS_OUT * C));
    add("head.b", { N_HEADS_OUT }, copy(L.get("head.bias", { N_HEADS_OUT }), N_HEADS_OUT));
    {  // rotary tables over the 200 trunk frames, torch's float32 steps
        std::vector<float> cs((size_t) C * T_TRUNK), sn((size_t) C * T_TRUNK);
        const int          half = DH / 2;
        for (int t = 0; t < T_TRUNK; t++)
            for (int d = 0; d < C; d++) {
                const int   j    = (d % DH) % half;
                const float freq = 1.0f / std::pow(10000.0f, (float) j / (float) half);
                const float ang  = (float) t * freq;
                cs[(size_t) t * C + d] = std::cos(ang);
                sn[(size_t) t * C + d] = std::sin(ang);
            }
        add("rope.cos", { C, T_TRUNK }, cs);
        add("rope.sin", { C, T_TRUNK }, sn);
    }
    // every F32 tensor of the file must have been read (the I64 BatchNorm counters are not parameters)
    for (auto & e : L.st.entries)
        if (e.dtype == "F32" && !L.used.count(e.name)) {
            fprintf(stderr, "[dettato] unexpected tensor %s in the transcriber file\n", e.name.c_str());
            L.ok = false;
        }
    if (!L.ok) {
        st_close(&L.st);
        return false;
    }
    st_close(&L.st);

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
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, m.bp.backend);
    if (!m.buf) {
        fprintf(stderr, "[dettato] weight buffer allocation failed\n");
        return false;
    }
    for (auto & u : up) ggml_backend_tensor_set(u.first, u.second->data(), 0, ggml_nbytes(u.first));
    m.sched = backend_sched_new(m.bp, GRAPH_NODES);
    m.meta.resize(GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false));
    m.front = make_front();
    return true;
}

static void free_model(Model & m) {
    if (m.sched) ggml_backend_sched_free(m.sched);
    if (m.buf) ggml_backend_buffer_free(m.buf);
    if (m.ctx) ggml_free(m.ctx);
    if (m.bp.backend) backend_release(m.bp.backend, m.bp.cpu_backend);
    m = Model();
}

// ---------------------------------------------------------------- graph pieces
static ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(c, w, x);
    ggml_mul_mat_set_hint(r, GGML_HINT_STRICT_F32);
    return r;
}
static ggml_tensor * linear(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    return ggml_add(c, mm(c, m.get(p + ".w"), x), m.get(p + ".b"));
}
static ggml_tensor * layer_norm(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    x = ggml_rms_norm(c, ggml_sub(c, x, ggml_mean(c, x)), 1e-5f);
    return ggml_add(c, ggml_mul(c, x, m.get(p + ".ln.w")), m.get(p + ".ln.b"));
}

// Depthwise conv of x [C, P, T, B] (channels contiguous) -> [C, P', T', B]. k is stored [C, 1, KW, KH]
// (channel-fastest); the permute presents it as [KW, KH, 1, C] with the strides the CWHN kernels index.
static ggml_tensor * dwconv(ggml_context * c, ggml_tensor * k, ggml_tensor * x, int s1, int p0, int p1, int d1) {
    ggml_tensor * y = ggml_conv_2d_dw_direct(c, ggml_permute(c, k, 3, 2, 0, 1), ggml_permute(c, x, 2, 0, 1, 3), 1, s1, p0, p1, 1, d1);
    return ggml_permute(c, y, 1, 2, 0, 3);
}

static ggml_tensor * lane_block(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p, int dt) {
    ggml_tensor * y = dwconv(c, m.get(p + ".kt"), x, 1, 0, (KT / 2) * dt, dt);
    y               = ggml_add(c, dwconv(c, m.get(p + ".kp"), y, 1, PK_HALF, 0, 1), m.get(p + ".bp"));
    ggml_tensor * h = ggml_gelu_erf(c, ggml_add(c, mm(c, m.get(p + ".w1"), y), m.get(p + ".b1")));
    return ggml_add(c, x, ggml_add(c, mm(c, m.get(p + ".w2"), h), m.get(p + ".b2")));
}

// Multi-head attention over dim 1. q, k, v: [C, L, N] (strided views allowed). -> [C, L, N]
static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) {
    const int64_t L = q->ne[1], N = q->ne[2];
    auto heads = [&](ggml_tensor * x) { return ggml_view_4d(c, x, DH, NH, L, N, DH * sizeof(float), x->nb[1], x->nb[2], 0); };
    ggml_tensor * q4 = ggml_cont(c, ggml_permute(c, heads(q), 0, 2, 1, 3));
    ggml_tensor * k4 = ggml_cont(c, ggml_permute(c, heads(k), 0, 2, 1, 3));
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, heads(v), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_soft_max_ext(c, mm(c, k4, q4), mask, 1.0f / sqrtf((float) DH), 0.0f);
    ggml_tensor * o  = ggml_cont(c, ggml_permute(c, mm(c, vt, kq), 0, 2, 1, 3));
    return ggml_reshape_3d(c, o, C, L, N);
}

static ggml_tensor * qkv_part(ggml_context * c, ggml_tensor * qkv, int part) {
    return ggml_view_3d(c, qkv, C, qkv->ne[1], qkv->ne[2], qkv->nb[1], qkv->nb[2], (size_t) part * C * sizeof(float));
}

// x [C, P, T, B]: attention across the 88 lanes of every frame, learned relative bias per head
static ggml_tensor * attn_pitch(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    const int64_t P = x->ne[1], N = x->ne[2] * x->ne[3];
    ggml_tensor * h   = ggml_reshape_3d(c, layer_norm(c, m, x, p), C, P, N);
    ggml_tensor * qkv = linear(c, m, h, p + ".qkv");
    ggml_tensor * o   = attention(c, qkv_part(c, qkv, 0), qkv_part(c, qkv, 1), qkv_part(c, qkv, 2), m.get(p + ".mask"));
    o                 = linear(c, m, o, p + ".out");
    return ggml_add(c, x, ggml_reshape_4d(c, o, C, P, x->ne[2], x->ne[3]));
}

// x [C, P, T, B]: attention across the frames of every lane, rotary positions
static ggml_tensor * attn_time(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p) {
    const int64_t P = x->ne[1], T = x->ne[2], B = x->ne[3];
    ggml_tensor * xt  = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)), C, T, P * B);
    ggml_tensor * h   = layer_norm(c, m, xt, p);
    ggml_tensor * qkv = linear(c, m, h, p + ".qkv");
    ggml_tensor * rot = linear(c, m, h, p + ".rot");
    ggml_tensor * cs = m.get("rope.cos"), * sn = m.get("rope.sin");
    ggml_tensor * q = ggml_add(c, ggml_mul(c, qkv_part(c, qkv, 0), cs), ggml_mul(c, qkv_part(c, rot, 0), sn));
    ggml_tensor * k = ggml_add(c, ggml_mul(c, qkv_part(c, qkv, 1), cs), ggml_mul(c, qkv_part(c, rot, 1), sn));
    ggml_tensor * o = linear(c, m, attention(c, q, k, qkv_part(c, qkv, 2), nullptr), p + ".out");
    o               = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, o, C, T, P, B), 0, 2, 1, 3));
    return ggml_add(c, x, o);
}

struct GruInputs {
    ggml_tensor *              h0 = nullptr, * rev = nullptr, * idx = nullptr;
    std::vector<ggml_tensor *> pin;  // graph tensors that must live on the compute backend
};

// x [C, P, T, B]: bidirectional GRU along time for every lane, both directions in one step
static ggml_tensor * lane_gru(ggml_context * c, const Model & m, ggml_tensor * x, const std::string & p, GruInputs & in,
                              ggml_cgraph * gf) {
    const int64_t P = x->ne[1], T = x->ne[2], B = x->ne[3], N = P * B;
    ggml_tensor * xg = ggml_cont(c, ggml_permute(c, layer_norm(c, m, x, p), 0, 1, 3, 2));  // [C, P, B, T]
    ggml_tensor * xr = ggml_get_rows(c, ggml_reshape_2d(c, xg, C * N, T), in.rev);         // time reversed
    ggml_tensor * gf_ = ggml_add(c, mm(c, m.get(p + ".wih_f"), ggml_reshape_3d(c, xg, C, N, T)), m.get(p + ".bih_f"));
    ggml_tensor * gr_ = ggml_add(c, mm(c, m.get(p + ".wih_r"), ggml_reshape_3d(c, xr, C, N, T)), m.get(p + ".bih_r"));
    ggml_tensor * G  = ggml_concat(c, ggml_reshape_4d(c, gf_, HID, 3, N, T), ggml_reshape_4d(c, gr_, HID, 3, N, T), 0);  // [2H, 3, N, T]
    ggml_tensor * O  = ggml_new_tensor_2d(c, GGML_TYPE_F32, HID, 2 * N * T);  // rows (dir, p, t, b) = [2H, P, T, B]
    in.pin.push_back(O);  // every row is written by exactly one set_rows below
    ggml_tensor * h  = in.h0;
    ggml_tensor * whh = m.get(p + ".whh"), * bhn = m.get(p + ".bhn");
    const size_t  row = sizeof(float), H2 = 2 * HID;
    for (int s = 0; s < T; s++) {
        const size_t  gs  = (size_t) s * G->nb[3];
        ggml_tensor * gh  = mm(c, whh, h);  // [6H, N]
        ggml_tensor * grz = ggml_view_2d(c, G, 2 * H2, N, G->nb[2], gs);
        ggml_tensor * gn  = ggml_view_2d(c, G, H2, N, G->nb[2], gs + 2 * H2 * row);
        ggml_tensor * rz  = ggml_sigmoid(c, ggml_add(c, grz, ggml_view_2d(c, gh, 2 * H2, N, gh->nb[1], 0)));
        ggml_tensor * r   = ggml_view_2d(c, rz, H2, N, rz->nb[1], 0);
        ggml_tensor * z   = ggml_view_2d(c, rz, H2, N, rz->nb[1], H2 * row);
        ggml_tensor * hn  = ggml_add(c, ggml_view_2d(c, gh, H2, N, gh->nb[1], 2 * H2 * row), bhn);
        ggml_tensor * n   = ggml_tanh(c, ggml_add(c, gn, ggml_mul(c, r, hn)));
        h                 = ggml_add(c, n, ggml_mul(c, z, ggml_sub(c, h, n)));
        O = ggml_set_rows(c, O, ggml_reshape_2d(c, h, HID, 2 * N), ggml_view_1d(c, in.idx, 2 * N, (size_t) s * 2 * N * in.idx->nb[0]));
        ggml_build_forward_expand(gf, O);
    }
    ggml_tensor * y = linear(c, m, ggml_reshape_4d(c, O, H2, P, T, B), p + ".out");
    return ggml_add(c, x, y);
}

// 2-D conv as im2col + one strict-F32 matmul (ggml_conv_2d's layout): k [KW, KH, IC, OC], x [W, H, IC, N] -> [W', H', OC, N].
// On the 5090 the stem took 0.47 s this way against 1.21 s through ggml_conv_2d_direct (90 s song, 47 windows).
static ggml_tensor * conv2d(ggml_context * c, ggml_tensor * k, ggml_tensor * x, int s0, int s1, int p0, int p1) {
    ggml_tensor * im = ggml_im2col(c, k, x, s0, s1, p0, p1, 1, 1, true, GGML_TYPE_F32);  // [IC*KH*KW, W', H', N]
    const int64_t OW = im->ne[1], OH = im->ne[2], N = im->ne[3], OC = k->ne[3];
    ggml_tensor * r = mm(c, ggml_reshape_2d(c, im, im->ne[0], OW * OH * N), ggml_reshape_2d(c, k, im->ne[0], OC));  // [W'H'N, OC]
    return ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, r, OW, OH, N, OC), 0, 1, 3, 2));
}

// stem: HCQT [N_BINS, T10, N_CH, B] -> [C, 88, FW, B] (the trunk's input and the stem tap)
static ggml_tensor * build_stem(ggml_context * c, const Model & m, ggml_tensor * x) {
    x = ggml_add(c, ggml_mul(c, x, m.get("stem.in_s")), m.get("stem.in_t"));
    x = ggml_gelu_erf(c, ggml_add(c, conv2d(c, m.get("stem.c1.w"), x, 1, 1, 1, 1), m.get("stem.c1.b")));
    x = ggml_gelu_erf(c, ggml_add(c, conv2d(c, m.get("stem.c2.w"), x, 1, 1, 1, 1), m.get("stem.c2.b")));
    x = ggml_add(c, conv2d(c, m.get("stem.down.w"), x, 3, 2, 1, 1), m.get("stem.down.b"));  // [96, 400, C, B]
    x = ggml_view_4d(c, x, N_KEYS, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
    return ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
}

struct TrunkOut { ggml_tensor * probs = nullptr, * hidden = nullptr; };

// trunk: x0 [C, 88, FW, B] -> probs [88, FW, 6, B], hidden [C, 88, FW, B]
static TrunkOut build_trunk(ggml_context * c, const Model & m, ggml_tensor * x0, GruInputs & gi, ggml_cgraph * gf) {
    ggml_tensor * h = ggml_add(c, dwconv(c, m.get("down.k"), x0, 2, 0, 1, 1), m.get("down.b"));
    int li = 0;
    for (int i = 0; i < DEPTH; i++) {
        h = lane_block(c, m, h, "layers." + std::to_string(li++), DILATION[i % 4]);
        if ((i + 1) % 5 == 0) {
            h = attn_pitch(c, m, h, "layers." + std::to_string(li++));
            h = attn_time(c, m, h, "layers." + std::to_string(li++));
            const std::string g = "layers." + std::to_string(li++);
            if (!m.skip_gru) h = lane_gru(c, m, h, g, gi, gf);
        }
    }
    const int64_t P = h->ne[1], T = h->ne[2], B = h->ne[3];
    h = ggml_reshape_4d(c, ggml_repeat_4d(c, ggml_reshape_4d(c, h, C * P, 1, T, B), C * P, 2, T, B), C, P, 2 * T, B);
    h = ggml_add(c, ggml_mul(c, ggml_add(c, h, x0), m.get("up.s")), m.get("up.t"));
    for (int r = 0; r < N_REFINE; r++) h = lane_block(c, m, h, "refine." + std::to_string(r), 1);
    TrunkOut o;
    o.hidden = h;
    ggml_tensor * pr = ggml_sigmoid(c, linear(c, m, h, "head"));  // [6, 88, FW, B]
    o.probs = ggml_cont(c, ggml_permute(c, pr, 2, 0, 1, 3));
    return o;
}

static void report_placement(const char * label, Model & m, ggml_cgraph * gf) {
    std::map<std::string, int> per, off;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
            t->op == GGML_OP_TRANSPOSE)
            continue;
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(m.sched, t);
        per[b ? ggml_backend_name(b) : "unassigned"]++;
        if (b != m.bp.backend) off[ggml_op_desc(t)]++;
    }
    fprintf(stderr, "[dettato] PLACEMENT %-6s nodes=%d splits=%d copies=%d |", label, n, ggml_backend_sched_get_n_splits(m.sched),
            ggml_backend_sched_get_n_copies(m.sched));
    for (auto & kv : per) fprintf(stderr, " %s=%d", kv.first.c_str(), kv.second);
    fprintf(stderr, " | off %s:", ggml_backend_name(m.bp.backend));
    if (off.empty()) fprintf(stderr, " none");
    for (auto & kv : off) fprintf(stderr, " %s x%d", kv.first.c_str(), kv.second);
    fprintf(stderr, "\n");
}

static void compute(Model & m, ggml_cgraph * gf, const char * label) {
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(m.sched, gf))) {
        fprintf(stderr, "[dettato] %s graph compute failed\n", label);
        exit(1);
    }
}

// hc [B][N_CH][T10][N_BINS] -> stem [B][FW][88][C]
static void run_stem(Model & m, const float * hc, int B, float * out) {
    ggml_init_params ip = { m.meta.size(), m.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    x  = ggml_new_tensor_4d(c, GGML_TYPE_F32, N_BINS, T10, N_CH, B);
    ggml_set_input(x);
    ggml_tensor * y = build_stem(c, m, x);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    ggml_backend_sched_reset(m.sched);
    ggml_backend_sched_set_tensor_backend(m.sched, x, m.bp.backend);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) {
        fprintf(stderr, "[dettato] stem graph allocation failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(x, hc, 0, ggml_nbytes(x));
    compute(m, gf, "stem");
    ggml_backend_tensor_get(y, out, 0, ggml_nbytes(y));
    if (!m.shown[0]) {
        report_placement("stem", m, gf);
        m.shown[0] = true;
    }
    ggml_free(c);
}

// stem [B][FW][88][C] -> probs [B][6][FW][88], hidden [B][FW][88][C] (when hidden != nullptr)
static void run_trunk(Model & m, const float * stem, int B, float * probs, float * hidden) {
    ggml_init_params ip = { m.meta.size(), m.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    x  = ggml_new_tensor_4d(c, GGML_TYPE_F32, C, N_KEYS, FW, B);
    ggml_set_input(x);
    const int N = N_KEYS * B;
    GruInputs gi;
    gi.h0  = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2 * HID, N);
    gi.rev = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_TRUNK);
    gi.idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, (int64_t) 2 * N * T_TRUNK);
    for (ggml_tensor * t : { gi.h0, gi.rev, gi.idx }) ggml_set_input(t);
    gi.pin = { x };
    if (!m.skip_gru) gi.pin.insert(gi.pin.end(), { gi.h0, gi.rev, gi.idx });
    TrunkOut o = build_trunk(c, m, x, gi, gf);
    ggml_set_output(o.probs);
    if (hidden) ggml_set_output(o.hidden);
    ggml_build_forward_expand(gf, o.probs);
    if (hidden) ggml_build_forward_expand(gf, o.hidden);
    ggml_backend_sched_reset(m.sched);
    for (ggml_tensor * t : gi.pin) ggml_backend_sched_set_tensor_backend(m.sched, t, m.bp.backend);
    if (!ggml_backend_sched_alloc_graph(m.sched, gf)) {
        fprintf(stderr, "[dettato] trunk graph allocation failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(x, stem, 0, ggml_nbytes(x));
    if (!m.skip_gru) {
        std::vector<float> zero((size_t) 2 * HID * N, 0.0f);
        ggml_backend_tensor_set(gi.h0, zero.data(), 0, ggml_nbytes(gi.h0));
        std::vector<int32_t> rev(T_TRUNK);
        for (int t = 0; t < T_TRUNK; t++) rev[t] = T_TRUNK - 1 - t;
        ggml_backend_tensor_set(gi.rev, rev.data(), 0, ggml_nbytes(gi.rev));
        std::vector<int64_t> idx((size_t) 2 * N * T_TRUNK);
        for (int s = 0; s < T_TRUNK; s++)
            for (int b = 0; b < B; b++)
                for (int p = 0; p < N_KEYS; p++)
                    for (int dir = 0; dir < 2; dir++) {
                        const int t = dir ? T_TRUNK - 1 - s : s;
                        idx[(size_t) s * 2 * N + 2 * (p + N_KEYS * b) + dir] = dir + 2 * (p + (int64_t) N_KEYS * (t + (int64_t) T_TRUNK * b));
                    }
        ggml_backend_tensor_set(gi.idx, idx.data(), 0, ggml_nbytes(gi.idx));
    }
    compute(m, gf, "trunk");
    ggml_backend_tensor_get(o.probs, probs, 0, ggml_nbytes(o.probs));
    if (hidden) ggml_backend_tensor_get(o.hidden, hidden, 0, ggml_nbytes(o.hidden));
    if (!m.shown[1]) {
        report_placement("trunk", m, gf);
        m.shown[1] = true;
    }
    ggml_free(c);
}

// ---------------------------------------------------------------- the windowed run (Transcriber.run)
// all: every window (every 2 s), its maps and, with all_hidden, the hidden tap (the repairer's input).
// sub: the windows starting on a multiple of 4 s, maps and, with sub_taps, the hidden and stem taps (the grouper's).
struct RunResult {
    Average all, sub;
    int     T = 0, n_windows = 0;
};

// progress(done, total) after every trunk batch
template <typename Progress>
static void run(Model & m, const std::vector<float> & y, bool all_hidden, bool sub_taps, RunResult & r, Progress progress) {
    const long long          n  = (long long) y.size();
    const Layout             L  = layout(n);
    const std::vector<float> wt = taper();
    r.T         = L.T;
    r.n_windows = (int) L.starts.size();
    r.all.init(L.T, C, all_hidden, false);
    r.sub.init(L.T, C, sub_taps, sub_taps);
    const bool   need_hidden = all_hidden || sub_taps;
    const int    Bt = std::max(1, m.trunk_batch), Bs = std::max(1, std::min(m.stem_batch, Bt));
    const size_t n_hc = (size_t) N_CH * T10 * N_BINS, n_st = (size_t) FW * N_KEYS * C, n_pr = (size_t) N_HEADS_OUT * FW * N_KEYS;
    std::vector<float> hc((size_t) Bt * n_hc), st((size_t) Bt * n_st), pr((size_t) Bt * n_pr), hid(need_hidden ? (size_t) Bt * n_st : 0);
    const int nthreads = std::max(1, std::min(Bt, (int) std::thread::hardware_concurrency()));
    for (int b0 = 0; b0 < r.n_windows; b0 += Bt) {
        const int B  = std::min(Bt, r.n_windows - b0);
        double    t0 = now_ms();
        std::vector<std::thread> pool;
        for (int ti = 0; ti < std::min(nthreads, B); ti++)
            pool.emplace_back([&, ti]() {
                std::vector<float> xw(WIN);
                for (int k = ti; k < B; k += nthreads) {
                    const long long s0 = L.starts[b0 + k];
                    for (int i = 0; i < WIN; i++) {
                        const long long j = s0 + i - PAD;
                        xw[i] = (j >= 0 && j < n) ? y[(size_t) j] : 0.0f;
                    }
                    hcqt(m.front, xw.data(), &hc[(size_t) k * n_hc]);
                }
            });
        for (auto & t : pool) t.join();
        double t1 = now_ms();
        for (int s0 = 0; s0 < B; s0 += Bs) run_stem(m, &hc[(size_t) s0 * n_hc], std::min(Bs, B - s0), &st[(size_t) s0 * n_st]);
        double t2 = now_ms();
        run_trunk(m, st.data(), B, pr.data(), need_hidden ? hid.data() : nullptr);
        double t3 = now_ms();
        for (int k = 0; k < B; k++) {
            const int s0 = L.starts[b0 + k], g0 = frame_offset(s0);
            const int a = std::max(0, g0), b = std::min(L.T, g0 + FW);
            if (b <= a) continue;
            const float * hk = need_hidden ? &hid[(size_t) k * n_st] : nullptr;
            r.all.add(a, b, g0, &wt[a - g0], &pr[(size_t) k * n_pr], hk, nullptr);
            if (s0 % SUB_STEP == 0) r.sub.add(a, b, g0, &wt[a - g0], &pr[(size_t) k * n_pr], hk, &st[(size_t) k * n_st]);
        }
        m.ms_hcqt += t1 - t0;
        m.ms_stem += t2 - t1;
        m.ms_trunk += t3 - t2;
        m.ms_accum += now_ms() - t3;
        progress(b0 + B, r.n_windows);
    }
    double t4 = now_ms();
    r.all.finish();
    r.sub.finish();
    m.ms_accum += now_ms() - t4;
}

}  // namespace dettato::transcriber
