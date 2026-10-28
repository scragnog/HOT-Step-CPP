#pragma once
// sheetsage-heads.h: SheetSage2-Refined frame heads, CPU.
//
// Port of heads.py's FrameHead / bar_facts / HeadMean / BarFeatures from the
// MIT reference (github.com/anime-song/SheetSage2-Refined). The heads read
// learned mixes of the encoder's 25 hidden states; the mixes are computed in
// the encoder graph (sheetsage-encoder.h, SheetSageEncodeOptions::
// extra_mix_weights) and handed in here as [T,1024] row-major buffers.
// Everything after the mix runs here in fp32:
//
//   x = gelu(inp(concat_j LayerNorm(mix_j)))           [T, hidden]
//   x = x + gelu(conv_k(x)), k = 0,1                   grouped Conv1d, 32 ch/group, kernel 9, pad 4
//   x = BiGRU(x), 2 layers, hidden/2 per direction     optionally chunked (heads.py CHUNK)
//   out = linear heads on x
//
// Why CPU: GGML has no GRU op, and the heads are tiny next to the encoder (a
// 300 s window is 7500 frames). The input linears are the only sizeable
// matmuls; they and the recurrences are split across threads.
//
// The weights come from the overlay GGUF written by
// engine/tools/convert-sheetsage2-refined.py (hd.<key>.<source name>, all F32).
// Gate: yue2-probe --sheetsage-refined-heads against
// engine/tools/sheetsage2-refined-oracle.py dumps.

#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

struct SheetSageHeadLinear {
    int                n_out = 0, n_in = 0;
    std::vector<float> w;  // [n_out, n_in] row-major (torch layout)
    std::vector<float> b;  // [n_out]
};

struct SheetSageHeadGru {  // one layer, one direction (torch gate order r, z, n)
    SheetSageHeadLinear ih;  // [3H, in]
    SheetSageHeadLinear hh;  // [3H, H]
};

struct SheetSageHead {
    std::string key;
    int         hidden = 0, mixes = 0, gru_layers = 0, binary = 0;

    std::vector<float>  mix_softmax;  // [mixes, 25]
    std::vector<float>  norm_w, norm_b;
    SheetSageHeadLinear inp;
    SheetSageHeadLinear conv[2];  // w as [hidden, 32, 9]
    std::vector<SheetSageHeadGru> gru;  // [layer * 2 + dir]
    std::map<std::string, SheetSageHeadLinear> outs;  // root, bass, tones, key, binary, bar, detail
};

// The two bar heads, the BarFeatures projection and the three rescoring heads.
struct SheetSageRefinedHeads {
    std::vector<SheetSageHead> heads;  // bar0, bar1, chord, detail, multi
    SheetSageHeadLinear        bar_proj;  // [512, 43]
    int chunk_core = 1500, chunk_context = 375;

    // Decode-time rescoring (config.json "rescoring", transcribe.py
    // FRAME_LOOKAHEAD / FRAME_REACH) and the key_mode bias.
    std::map<std::string, float> rescore_weight;  // downbeat, meter, root, bass, quality, key, key_boundary
    std::map<std::string, int>   lookahead;       // root, bass, quality, key, meter
    std::map<std::string, int>   reach;           // key_boundary, downbeat
    float                        mode_tau = 0.0f;
    std::vector<float>           mode_counts;     // per theory.MODES

    const SheetSageHead * find(const std::string & key) const {
        for (const SheetSageHead & h : heads) {
            if (h.key == key) {
                return &h;
            }
        }
        return nullptr;
    }
};

// heads.py BINARY / BAR names, in output order.
static const char * const SS2R_BINARY[] = { "chord_boundary", "key_boundary", "beat", "downbeat", "group" };
static const struct {
    const char * name;
    int          n;
} SS2R_BAR[] = { { "bar_beats", 16 }, { "beats_left", 16 }, { "bar_groups", 8 } };
static const int SS2R_BAR_FEATURES = 43;
static const int SS2R_STATES       = 25;
static const int SS2R_WIDTH        = 1024;

// ── loading ─────────────────────────────────────────────────────────────────

static bool ss2r_tensor(ggml_context * ctx, const std::string & name, std::vector<float> * out,
                        std::vector<int64_t> * shape, std::string * err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        *err = "missing tensor " + name;
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        *err = name + " is not F32";
        return false;
    }
    out->assign((const float *) t->data, (const float *) t->data + ggml_nelements(t));
    if (shape) {  // torch order (outermost first)
        shape->clear();
        for (int d = ggml_n_dims(t) - 1; d >= 0; d--) {
            shape->push_back(t->ne[d]);
        }
    }
    return true;
}

static bool ss2r_linear(ggml_context * ctx, const std::string & prefix, SheetSageHeadLinear * l,
                        std::string * err) {
    std::vector<int64_t> s;
    if (!ss2r_tensor(ctx, prefix + ".weight", &l->w, &s, err) || !ss2r_tensor(ctx, prefix + ".bias", &l->b, nullptr, err)) {
        return false;
    }
    l->n_out = (int) s[0];
    l->n_in  = (int) (l->w.size() / (size_t) l->n_out);
    return true;
}

static bool ss2r_gguf_u32(gguf_context * g, const std::string & key, int * out) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return false;
    }
    *out = (int) gguf_get_val_u32(g, id);
    return true;
}

static bool sheetsage_refined_heads_load(const std::string & path, SheetSageRefinedHeads * out, std::string * err) {
    *out                    = SheetSageRefinedHeads{};
    ggml_context *     ctx  = nullptr;
    gguf_init_params   gp   = { /*no_alloc*/ false, &ctx };
    gguf_context *     g    = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        *err = "cannot read " + path;
        return false;
    }
    bool ok = false;
    do {
        const int64_t arch_id = gguf_find_key(g, "general.architecture");
        if (arch_id < 0 || std::string(gguf_get_val_str(g, arch_id)) != "sheetsage2r") {
            *err = path + " is not a sheetsage2r (SheetSage2-Refined) GGUF";
            break;
        }
        ss2r_gguf_u32(g, "sheetsage2r.heads.chunk_core", &out->chunk_core);
        ss2r_gguf_u32(g, "sheetsage2r.heads.chunk_context", &out->chunk_context);
        for (const char * t : { "downbeat", "meter", "root", "bass", "quality", "key", "key_boundary" }) {
            const int64_t id = gguf_find_key(g, (std::string("sheetsage2r.rescore.") + t + ".weight").c_str());
            if (id >= 0) {
                out->rescore_weight[t] = gguf_get_val_f32(g, id);
            }
            int n = 0;
            if (ss2r_gguf_u32(g, std::string("sheetsage2r.rescore.") + t + ".lookahead", &n)) {
                out->lookahead[t] = n;
            }
            if (ss2r_gguf_u32(g, std::string("sheetsage2r.rescore.") + t + ".reach", &n)) {
                out->reach[t] = n;
            }
        }
        {
            const int64_t tau = gguf_find_key(g, "sheetsage2r.key_mode.tau");
            const int64_t cnt = gguf_find_key(g, "sheetsage2r.key_mode.counts");
            if (tau >= 0) {
                out->mode_tau = gguf_get_val_f32(g, tau);
            }
            if (cnt >= 0 && gguf_get_arr_type(g, cnt) == GGUF_TYPE_FLOAT32) {
                const float * d = (const float *) gguf_get_arr_data(g, cnt);
                out->mode_counts.assign(d, d + gguf_get_arr_n(g, cnt));
            }
        }
        if (!ss2r_linear(ctx, "bar.proj", &out->bar_proj, err)) {
            break;
        }
        if (out->bar_proj.n_in != SS2R_BAR_FEATURES) {
            *err = "bar.proj expects 43 bar facts";
            break;
        }
        bool head_ok = true;
        for (const char * key : { "bar0", "bar1", "chord", "detail", "multi" }) {
            SheetSageHead h;
            h.key            = key;
            const std::string kp = std::string("sheetsage2r.head.") + key;
            if (!ss2r_gguf_u32(g, kp + ".hidden", &h.hidden) || !ss2r_gguf_u32(g, kp + ".mixes", &h.mixes) ||
                !ss2r_gguf_u32(g, kp + ".gru_layers", &h.gru_layers) || !ss2r_gguf_u32(g, kp + ".binary", &h.binary)) {
                *err     = std::string("missing head KVs for ") + key;
                head_ok  = false;
                break;
            }
            const std::string p = std::string("hd.") + key;
            std::vector<float> mix;
            if (!ss2r_tensor(ctx, p + ".mix", &mix, nullptr, err) || (int) mix.size() != h.mixes * SS2R_STATES ||
                !ss2r_tensor(ctx, p + ".norm.weight", &h.norm_w, nullptr, err) ||
                !ss2r_tensor(ctx, p + ".norm.bias", &h.norm_b, nullptr, err) || !ss2r_linear(ctx, p + ".inp", &h.inp, err) ||
                !ss2r_linear(ctx, p + ".convs.0", &h.conv[0], err) || !ss2r_linear(ctx, p + ".convs.1", &h.conv[1], err)) {
                if (err->empty()) {
                    *err = p + ".mix has the wrong size";
                }
                head_ok = false;
                break;
            }
            h.mix_softmax.resize(mix.size());
            for (int j = 0; j < h.mixes; j++) {  // softmax over the 25 states, in double
                const float * r  = mix.data() + j * SS2R_STATES;
                double        mx = r[0], sum = 0.0;
                for (int s = 1; s < SS2R_STATES; s++) {
                    mx = std::max(mx, (double) r[s]);
                }
                for (int s = 0; s < SS2R_STATES; s++) {
                    sum += std::exp((double) r[s] - mx);
                }
                for (int s = 0; s < SS2R_STATES; s++) {
                    h.mix_softmax[j * SS2R_STATES + s] = (float) (std::exp((double) r[s] - mx) / sum);
                }
            }
            for (int l = 0; l < h.gru_layers && head_ok; l++) {
                for (int d = 0; d < 2 && head_ok; d++) {
                    const std::string sfx = std::to_string(l) + (d ? "_reverse" : "");
                    SheetSageHeadGru  gr;
                    std::vector<int64_t> s;
                    head_ok = ss2r_tensor(ctx, p + ".rnn.weight_ih_l" + sfx, &gr.ih.w, &s, err) &&
                              ss2r_tensor(ctx, p + ".rnn.bias_ih_l" + sfx, &gr.ih.b, nullptr, err);
                    gr.ih.n_out = (int) s[0];
                    gr.ih.n_in  = (int) s[1];
                    head_ok     = head_ok && ss2r_tensor(ctx, p + ".rnn.weight_hh_l" + sfx, &gr.hh.w, &s, err) &&
                              ss2r_tensor(ctx, p + ".rnn.bias_hh_l" + sfx, &gr.hh.b, nullptr, err);
                    gr.hh.n_out = (int) s[0];
                    gr.hh.n_in  = (int) s[1];
                    h.gru.push_back(std::move(gr));
                }
            }
            for (const char * o : { "root", "bass", "tones", "key", "binary", "bar", "detail" }) {
                const std::string name = p + "." + o + ".weight";
                if (!ggml_get_tensor(ctx, name.c_str())) {
                    continue;
                }
                SheetSageHeadLinear l;
                if (!ss2r_linear(ctx, p + "." + o, &l, err)) {
                    head_ok = false;
                    break;
                }
                h.outs[o] = std::move(l);
            }
            if (!head_ok) {
                break;
            }
            out->heads.push_back(std::move(h));
        }
        ok = head_ok;
    } while (false);
    gguf_free(g);
    if (ctx) {
        ggml_free(ctx);
    }
    return ok;
}

// ── math ────────────────────────────────────────────────────────────────────

static int ss2r_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return (int) std::max(1u, std::min(n, 32u));
}

template <class F> static void ss2r_parallel(int n, F fn) {
    const int nt = std::min(n, ss2r_threads());
    if (nt <= 1) {
        for (int i = 0; i < n; i++) {
            fn(i);
        }
        return;
    }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) {
        th.emplace_back([&, t] {
            for (int i = t; i < n; i += nt) {
                fn(i);
            }
        });
    }
    for (std::thread & x : th) {
        x.join();
    }
}

static inline float ss2r_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

static inline float ss2r_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// y[t] = W x[t] + b for T rows; x [T, n_in], y [T, n_out].
static void ss2r_linear_rows(const SheetSageHeadLinear & l, const float * x, int64_t T, float * y) {
    const int64_t block = 16;
    ss2r_parallel((int) ((T + block - 1) / block), [&](int bi) {
        const int64_t t0 = bi * block, t1 = std::min(T, t0 + block);
        for (int64_t t = t0; t < t1; t++) {
            const float * xr = x + t * l.n_in;
            float *       yr = y + t * l.n_out;
            for (int o = 0; o < l.n_out; o++) {
                const float * wr  = l.w.data() + (size_t) o * l.n_in;
                float         acc = 0.0f;
                for (int i = 0; i < l.n_in; i++) {
                    acc += wr[i] * xr[i];
                }
                yr[o] = acc + l.b[o];
            }
        }
    });
}

// One GRU direction over x [n, in_dim] -> y columns [off, off+H) of a [n, 2H] buffer.
static void ss2r_gru_dir(const SheetSageHeadGru & g, const float * gx, int64_t n, bool reverse, float * y,
                         int64_t y_stride, int64_t y_off) {
    const int H = g.hh.n_in;
    std::vector<float> h((size_t) H, 0.0f), gh((size_t) 3 * H);
    for (int64_t s = 0; s < n; s++) {
        const int64_t t  = reverse ? n - 1 - s : s;
        const float * gi = gx + t * 3 * H;  // W_ih x + b_ih, precomputed
        for (int o = 0; o < 3 * H; o++) {
            const float * wr  = g.hh.w.data() + (size_t) o * H;
            float         acc = 0.0f;
            for (int i = 0; i < H; i++) {
                acc += wr[i] * h[(size_t) i];
            }
            gh[(size_t) o] = acc + g.hh.b[(size_t) o];
        }
        float * yr = y + t * y_stride + y_off;
        for (int i = 0; i < H; i++) {
            const float r  = ss2r_sigmoid(gi[i] + gh[(size_t) i]);
            const float z  = ss2r_sigmoid(gi[H + i] + gh[(size_t) (H + i)]);
            const float nn = std::tanh(gi[2 * H + i] + r * gh[(size_t) (2 * H + i)]);
            h[(size_t) i]  = (1.0f - z) * nn + z * h[(size_t) i];
            yr[i]          = h[(size_t) i];
        }
    }
}

// Full bidirectional multi-layer GRU on x [n, hidden] -> [n, hidden].
static void ss2r_bigru(const SheetSageHead & hd, const float * x, int64_t n, std::vector<float> * out) {
    const int H2 = hd.hidden;
    std::vector<float> cur(x, x + n * H2), next((size_t) (n * H2));
    for (int l = 0; l < hd.gru_layers; l++) {
        const SheetSageHeadGru & gf = hd.gru[(size_t) (l * 2)];
        const SheetSageHeadGru & gb = hd.gru[(size_t) (l * 2 + 1)];
        std::vector<float> gxf((size_t) (n * gf.ih.n_out)), gxb((size_t) (n * gb.ih.n_out));
        ss2r_linear_rows(gf.ih, cur.data(), n, gxf.data());
        ss2r_linear_rows(gb.ih, cur.data(), n, gxb.data());
        std::thread tb([&] { ss2r_gru_dir(gb, gxb.data(), n, true, next.data(), H2, H2 / 2); });
        ss2r_gru_dir(gf, gxf.data(), n, false, next.data(), H2, 0);
        tb.join();
        cur.swap(next);
    }
    *out = std::move(cur);
}

// heads.py FrameHead.forward. mixes[j] -> [T, 1024]; outputs keyed like the reference.
static void sheetsage_head_forward(const SheetSageHead & hd, const std::vector<const float *> & mixes, int64_t T,
                                   bool chunked, int chunk_core, int chunk_context,
                                   std::map<std::string, std::vector<float>> * outputs) {
    const int     W  = SS2R_WIDTH;
    const int     M  = hd.mixes;
    const int     Hd = hd.hidden;
    std::vector<float> cat((size_t) (T * M * W));
    ss2r_parallel((int) T, [&](int t) {
        for (int j = 0; j < M; j++) {
            const float * xr = mixes[(size_t) j] + (int64_t) t * W;
            float *       yr = cat.data() + ((int64_t) t * M + j) * W;
            double        mean = 0.0, var = 0.0;
            for (int i = 0; i < W; i++) {
                mean += xr[i];
            }
            mean /= W;
            for (int i = 0; i < W; i++) {
                const double d = xr[i] - mean;
                var += d * d;
            }
            const float inv = (float) (1.0 / std::sqrt(var / W + 1e-5));
            for (int i = 0; i < W; i++) {
                yr[i] = ((float) (xr[i] - mean)) * inv * hd.norm_w[(size_t) i] + hd.norm_b[(size_t) i];
            }
        }
    });

    std::vector<float> x((size_t) (T * Hd));
    ss2r_linear_rows(hd.inp, cat.data(), T, x.data());
    for (float & v : x) {
        v = ss2r_gelu(v);
    }
    cat.clear();
    cat.shrink_to_fit();

    // Two residual grouped convs: groups of 32 channels, kernel 9, zero pad 4.
    const int G = 32, K = 9, P = 4;
    std::vector<float> y((size_t) (T * Hd));
    for (int c = 0; c < 2; c++) {
        const SheetSageHeadLinear & cv = hd.conv[c];
        ss2r_parallel((int) T, [&](int t) {
            for (int o = 0; o < Hd; o++) {
                const int     g0  = (o / G) * G;
                const float * wr  = cv.w.data() + (size_t) o * G * K;
                float         acc = cv.b[(size_t) o];
                for (int k = 0; k < K; k++) {
                    const int64_t tt = (int64_t) t + k - P;
                    if (tt < 0 || tt >= T) {
                        continue;
                    }
                    const float * xr = x.data() + tt * Hd + g0;
                    for (int i = 0; i < G; i++) {
                        acc += wr[i * K + k] * xr[i];
                    }
                }
                y[(size_t) t * Hd + o] = acc;
            }
        });
        for (size_t i = 0; i < x.size(); i++) {
            x[i] += ss2r_gelu(y[i]);
        }
    }

    // BiGRU, chunked as heads.py _rnn_chunked when the clip is long enough.
    std::vector<float> r((size_t) (T * Hd));
    if (chunked && T > (int64_t) chunk_core + 2 * chunk_context) {
        std::vector<int64_t> starts;
        for (int64_t s = 0; s < T; s += chunk_core) {
            starts.push_back(s);
        }
        ss2r_parallel((int) starts.size(), [&](int pi) {
            const int64_t s  = starts[(size_t) pi];
            const int64_t lo = std::max<int64_t>(s - chunk_context, 0);
            const int64_t hi = std::min<int64_t>(s + chunk_core + chunk_context, T);
            std::vector<float> part;
            ss2r_bigru(hd, x.data() + lo * Hd, hi - lo, &part);
            const int64_t end = std::min<int64_t>(s + chunk_core, T);
            std::memcpy(r.data() + s * Hd, part.data() + (s - lo) * Hd, (size_t) ((end - s) * Hd) * sizeof(float));
        });
    } else {
        ss2r_bigru(hd, x.data(), T, &r);
    }

    outputs->clear();
    for (const auto & kv : hd.outs) {
        std::vector<float> o((size_t) (T * kv.second.n_out));
        ss2r_linear_rows(kv.second, r.data(), T, o.data());
        const int n = kv.second.n_out;
        auto split = [&](const std::string & name, int off, int cnt) {
            std::vector<float> & dst = (*outputs)[name];
            dst.resize((size_t) (T * cnt));
            for (int64_t t = 0; t < T; t++) {
                std::memcpy(dst.data() + t * cnt, o.data() + t * n + off, (size_t) cnt * sizeof(float));
            }
        };
        if (kv.first == "binary") {
            for (int i = 0; i < n; i++) {
                split(SS2R_BINARY[i], i, 1);
            }
        } else if (kv.first == "bar") {
            int off = 0;
            for (const auto & b : SS2R_BAR) {
                split(b.name, off, b.n);
                off += b.n;
            }
        } else {
            (*outputs)[kv.first] = std::move(o);
        }
    }
}

// heads.py bar_facts: [T, 43] probabilities from one bar head's outputs.
static void sheetsage_bar_facts(const std::map<std::string, std::vector<float>> & out, int64_t T,
                                std::vector<float> * facts) {
    facts->assign((size_t) (T * SS2R_BAR_FEATURES), 0.0f);
    for (int64_t t = 0; t < T; t++) {
        float * f = facts->data() + t * SS2R_BAR_FEATURES;
        int     c = 0;
        for (const auto & b : SS2R_BAR) {
            const float * l  = out.at(b.name).data() + t * b.n;
            float         mx = l[0];
            for (int i = 1; i < b.n; i++) {
                mx = std::max(mx, l[i]);
            }
            double sum = 0.0;
            for (int i = 0; i < b.n; i++) {
                sum += std::exp((double) (l[i] - mx));
            }
            for (int i = 0; i < b.n; i++) {
                f[c++] = (float) (std::exp((double) (l[i] - mx)) / sum);
            }
        }
        for (const char * n : { "beat", "downbeat", "group" }) {
            f[c++] = ss2r_sigmoid(out.at(n)[(size_t) t]);
        }
    }
}

// The mixes the heads need, in the order sheetsage_refined_run_heads() reads
// them: for each head, its `mixes` rows of softmax(mix) over the 25 states.
static std::vector<std::vector<float>> sheetsage_refined_mix_weights(const SheetSageRefinedHeads & rh) {
    std::vector<std::vector<float>> w;
    for (const SheetSageHead & h : rh.heads) {
        for (int j = 0; j < h.mixes; j++) {
            w.emplace_back(h.mix_softmax.begin() + j * SS2R_STATES, h.mix_softmax.begin() + (j + 1) * SS2R_STATES);
        }
    }
    return w;
}

struct SheetSageRefinedHeadOutputs {
    int64_t T = 0;
    std::map<std::string, std::map<std::string, std::vector<float>>> full;     // per head, unchunked GRU
    std::map<std::string, std::map<std::string, std::vector<float>>> chunked;  // bar heads, chunked GRU
    std::vector<float> bar_facts;  // [T, 43], mean of bar0/bar1 chunked
};

// Runs every head on one window. `mixes` follows sheetsage_refined_mix_weights() order.
static void sheetsage_refined_run_heads(const SheetSageRefinedHeads & rh, const std::vector<const float *> & mixes,
                                        int64_t T, SheetSageRefinedHeadOutputs * out) {
    *out   = SheetSageRefinedHeadOutputs{};
    out->T = T;
    // Every (head, chunked?) pass is independent; the unchunked GRUs step
    // through all T frames one by one, so they run side by side.
    struct Job {
        const SheetSageHead *      h;
        std::vector<const float *> mixes;
        bool                       chunked;
        std::map<std::string, std::vector<float>> * dst;
    };
    std::vector<Job> jobs;
    size_t           mi = 0;
    for (const SheetSageHead & h : rh.heads) {
        std::vector<const float *> hm(mixes.begin() + (int64_t) mi, mixes.begin() + (int64_t) (mi + h.mixes));
        mi += (size_t) h.mixes;
        jobs.push_back({ &h, hm, false, &out->full[h.key] });
        if (h.key == "bar0" || h.key == "bar1") {
            jobs.push_back({ &h, hm, true, &out->chunked[h.key] });
        }
    }
    std::vector<std::thread> th;
    for (Job & j : jobs) {
        th.emplace_back([&rh, &j, T] {
            sheetsage_head_forward(*j.h, j.mixes, T, j.chunked, rh.chunk_core, rh.chunk_context, j.dst);
        });
    }
    for (std::thread & t : th) {
        t.join();
    }
    std::vector<std::vector<float>> facts;
    for (const char * k : { "bar0", "bar1" }) {
        if (out->chunked.count(k)) {
            facts.emplace_back();
            sheetsage_bar_facts(out->chunked[k], T, &facts.back());
        }
    }
    out->bar_facts.assign((size_t) (T * SS2R_BAR_FEATURES), 0.0f);
    for (const std::vector<float> & f : facts) {
        for (size_t i = 0; i < f.size(); i++) {
            out->bar_facts[i] += f[i] / (float) facts.size();
        }
    }
}

// BarFeatures.forward: memory[t] += W facts[t] + b for t < min(T_mem, T_facts).
static void sheetsage_refined_add_bar_features(const SheetSageRefinedHeads & rh, const std::vector<float> & facts,
                                               int64_t T_facts, float * memory, int64_t T_mem) {
    const int64_t T = std::min(T_mem, T_facts);
    const int     D = rh.bar_proj.n_out;
    std::vector<float> add((size_t) (T * D));
    ss2r_linear_rows(rh.bar_proj, facts.data(), T, add.data());
    for (size_t i = 0; i < add.size(); i++) {
        memory[i] += add[i];
    }
}
