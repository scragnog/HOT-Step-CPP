// YourMT3+ native GGML spike (HOTSTEP-1178).
//
// Runs the Perceiver-TF encoder from a captured pre-encoder output, the
// multi-channel projection, and the 13-stream T5 decoder (prefill, one cached
// step, then the full greedy decode) on one segment batch, and compares each
// point with the CPU reference capture. Also reports backend placement, a
// per-op profile and timings. Written from the capture contract and the model
// inventory; not a port of the reference source.
//
// usage: spike --self-test
//        spike <model.gguf> <capture-dir> [--cpu] [--tf32] [--profile] [--reps N]
//              [--dec-from-capture] [--no-eos-stop]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// ---- fixed model shape (inventory) ----
static const int D_ENC = 128, N_LAT = 26, N_HEAD_E = 8, DH_E = 16, ROT_DIMS = 4;
static const int N_EXPERT = 8, TOPK = 2, D_FF_E = 512;
static const int D_DEC = 512, N_HEAD_D = 6, DH_D = 64, D_FF_D = 1024, N_LAYER_D = 8;
static const int N_CH = 13, VOCAB = 596, MAX_TOK = 256;

static bool g_strict = true;  // GGML_HINT_STRICT_F32 on every matmul (TF32 otherwise)
static bool g_exact_gelu = false;  // composite tanh GELU; ggml-cpu's ggml_gelu reads an fp16 table

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::vector<float> read_f32(const std::string & path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    f.read((char *) v.data(), n * sizeof(float));
    if ((size_t) f.gcount() != n * sizeof(float)) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    return v;
}

// ---- weights ----
struct Model {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::map<std::string, std::vector<float>> host;  // small tensors needed on the host
    ggml_tensor * get(const std::string & n) {
        auto it = t.find(n);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); }
        return it->second;
    }
};

// Rows of a query projection reordered so that its output is rotate_half(q):
// out[2i] = -q[2i+1], out[2i+1] = q[2i] for each head. Exact (permutation and sign).
static void swap_rows(const float * w, float * o, int rows, int cols) {
    for (int r = 0; r < rows; r += 2) {
        for (int c = 0; c < cols; ++c) {
            o[(size_t) r * cols + c]       = -w[(size_t) (r + 1) * cols + c];
            o[(size_t) (r + 1) * cols + c] =  w[(size_t) r * cols + c];
        }
    }
}

static void load_model(Model & m, const char * path, ggml_backend_t be) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "gguf load failed\n"); exit(1); }
    const int n = (int) gguf_get_n_tensors(g);
    const size_t data_off = gguf_get_data_offset(g);
    std::ifstream f(path, std::ios::binary);

    auto read_tensor = [&](const char * name, std::vector<float> & out) {
        int64_t id = gguf_find_tensor(g, name);
        if (id < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        out.resize(ggml_nelements(mt));
        f.seekg(data_off + gguf_get_tensor_offset(g, id));
        f.read((char *) out.data(), ggml_nbytes(mt));
    };

    // device context: every gguf tensor plus stacked experts and swapped query weights
    const size_t n_extra = 3 * 12 + 2 * 15;
    ggml_init_params ip = { (n + n_extra + 16) * ggml_tensor_overhead(), nullptr, true };
    m.ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> pending;

    for (int i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        std::string s = name;
        if (s.find(".experts.") != std::string::npos) continue;  // stacked below
        ggml_tensor * mt = ggml_get_tensor(meta, name);
        ggml_tensor * dt = ggml_dup_tensor(m.ctx, mt);
        ggml_set_name(dt, name);
        m.t[s] = dt;
        std::vector<float> data;
        read_tensor(name, data);
        if (s.find("rotary_emb") != std::string::npos) m.host[s] = data;
        pending.push_back({dt, std::move(data)});
    }

    // MoE experts -> [in, out, N_EXPERT] stacks for mul_mat_id
    auto stack = [&](const std::string & pre, const char * w, int in, int out) {
        ggml_tensor * dt = ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, in, out, N_EXPERT);
        std::string sname = pre + "mlp." + w + "_stack";
        ggml_set_name(dt, sname.c_str());
        std::vector<float> all((size_t) in * out * N_EXPERT);
        for (int e = 0; e < N_EXPERT; ++e) {
            std::vector<float> one;
            std::string nm = pre + "mlp.experts." + std::to_string(e) + "." + w + ".weight";
            read_tensor(nm.c_str(), one);
            memcpy(all.data() + (size_t) e * in * out, one.data(), one.size() * sizeof(float));
        }
        m.t[sname] = dt;
        pending.push_back({dt, std::move(all)});
    };
    auto swapped_q = [&](const std::string & pre) {
        std::vector<float> w, b;
        read_tensor((pre + "query.weight").c_str(), w);
        read_tensor((pre + "query.bias").c_str(), b);
        std::vector<float> ws(w.size()), bs(b.size());
        swap_rows(w.data(), ws.data(), D_ENC, D_ENC);
        swap_rows(b.data(), bs.data(), D_ENC, 1);
        ggml_tensor * tw = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, D_ENC, D_ENC);
        ggml_tensor * tb = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, D_ENC);
        m.t[pre + "query_swap.weight"] = tw;
        m.t[pre + "query_swap.bias"] = tb;
        pending.push_back({tw, std::move(ws)});
        pending.push_back({tb, std::move(bs)});
    };
    for (int b = 0; b < 3; ++b) {
        std::string bp = "encoder.blocks." + std::to_string(b) + ".";
        swapped_q(bp + "spectral_cross_attention.self.");
        for (const char * kind : {"local_transformer", "temporal_transformer"}) {
            for (int l = 0; l < 2; ++l) {
                std::string lp = bp + kind + "." + std::to_string(l) + ".";
                stack(lp, "w1", D_ENC, D_FF_E);
                stack(lp, "gate", D_ENC, D_FF_E);
                stack(lp, "w2", D_FF_E, D_ENC);
                swapped_q(lp + "attention.self.");
            }
        }
    }

    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, be);
    for (auto & p : pending) ggml_backend_tensor_set(p.first, p.second.data(), 0, ggml_nbytes(p.first));
    gguf_free(g);
    ggml_free(meta);
}

// ---- graph helpers ----
static ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(c, w, x);
    if (g_strict) ggml_mul_mat_set_hint(r, GGML_HINT_STRICT_F32);
    return r;
}
static ggml_tensor * mm_id(ggml_context * c, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) {
    ggml_tensor * r = ggml_mul_mat_id(c, w, x, ids);
    // ggml-cuda copies the node's op_params into the per-expert slices, so the
    // strict hint reaches cuBLAS there too (op_params[1] is unused by MUL_MAT_ID otherwise).
    if (g_strict) r->op_params[1] = GGML_HINT_STRICT_F32;
    return r;
}
static ggml_tensor * linear(ggml_context * c, Model & m, ggml_tensor * x, const std::string & p, bool bias = true) {
    ggml_tensor * y = mm(c, m.get(p + ".weight"), x);
    if (bias) y = ggml_add(c, y, m.get(p + ".bias"));
    return y;
}
static ggml_tensor * layer_norm(ggml_context * c, Model & m, ggml_tensor * x, const std::string & p) {
    // Two-pass: centre first, then RMS. ggml-cuda's ggml_norm takes the variance as
    // E[x^2] - mean^2 in one pass, which cancels badly when |mean| >> std and
    // misses the 1e-3 gate on the encoder.
    x = ggml_rms_norm(c, ggml_sub(c, x, ggml_mean(c, x)), 1e-5f);
    return ggml_add(c, ggml_mul(c, x, m.get(p + ".weight")), m.get(p + ".bias"));
}

// Multi-head attention over the leading axis. q [D, Lq, N], k/v [D, Lk, N].
static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               int n_head, int dh, float scale) {
    const int64_t Lq = q->ne[1], Lk = k->ne[1], N = q->ne[2];
    ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, dh, n_head, Lq, N), 0, 2, 1, 3);  // [dh, Lq, H, N]
    ggml_tensor * k4 = ggml_permute(c, ggml_reshape_4d(c, k, dh, n_head, Lk, N), 0, 2, 1, 3);  // [dh, Lk, H, N]
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, v, dh, n_head, Lk, N), 1, 2, 0, 3));  // [Lk, dh, H, N]
    ggml_tensor * kq = mm(c, ggml_cont(c, k4), ggml_cont(c, q4));     // [Lk, Lq, H, N]
    kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
    ggml_tensor * o = mm(c, vt, kq);                                   // [dh, Lq, H, N]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));                  // [dh, H, Lq, N]
    return ggml_reshape_3d(c, o, dh * n_head, Lq, N);
}

// Queries with partial RoPE: q*cos + rotate_half(q)*sin, rotate_half from the
// row-swapped projection. Tables are [D_ENC, L] (cos 1 / sin 0 on unrotated dims).
static ggml_tensor * rope_query(ggml_context * c, Model & m, ggml_tensor * x, const std::string & p,
                                ggml_tensor * cos_t, ggml_tensor * sin_t) {
    ggml_tensor * q  = linear(c, m, x, p + "query");
    ggml_tensor * qs = linear(c, m, x, p + "query_swap");
    return ggml_add(c, ggml_mul(c, q, cos_t), ggml_mul(c, qs, sin_t));
}

// router probabilities kept for the near-tie probe (layer prefix, tensor)
static std::vector<std::pair<std::string, ggml_tensor *>> g_router;

static ggml_tensor * moe(ggml_context * c, Model & m, ggml_tensor * x, const std::string & p) {
    const int64_t M = x->ne[1] * x->ne[2];
    ggml_tensor * x2 = ggml_reshape_2d(c, x, D_ENC, M);
    ggml_tensor * probs = ggml_soft_max(c, mm(c, m.get(p + "mlp.gate.weight"), x2));  // [E, M]
    ggml_set_output(probs);
    g_router.push_back({p, probs});
    ggml_tensor * ids = ggml_argsort_top_k(c, probs, TOPK);                            // [K, M]
    ggml_tensor * w = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, N_EXPERT, M), ids); // [1, K, M]
    w = ggml_reshape_2d(c, w, TOPK, M);
    w = ggml_div(c, w, ggml_sum_rows(c, w));
    ggml_tensor * xe = ggml_reshape_3d(c, x2, D_ENC, 1, M);
    ggml_tensor * up = mm_id(c, m.get(p + "mlp.w1_stack"), xe, ids);    // [F, K, M]
    ggml_tensor * gt = mm_id(c, m.get(p + "mlp.gate_stack"), xe, ids);
    ggml_tensor * h = ggml_mul(c, ggml_silu(c, up), gt);
    ggml_tensor * dn = mm_id(c, m.get(p + "mlp.w2_stack"), h, ids);    // [D, K, M]
    dn = ggml_mul(c, dn, ggml_reshape_3d(c, w, 1, TOPK, M));
    ggml_tensor * e0 = ggml_view_2d(c, dn, D_ENC, M, dn->nb[2], 0);
    ggml_tensor * e1 = ggml_view_2d(c, dn, D_ENC, M, dn->nb[2], dn->nb[1]);
    ggml_tensor * out = ggml_add(c, e0, e1);
    return ggml_reshape_3d(c, out, D_ENC, x->ne[1], x->ne[2]);
}

// self-attention layer: x [D, L, N]
static ggml_tensor * perceiver_layer(ggml_context * c, Model & m, ggml_tensor * x, const std::string & p,
                                     ggml_tensor * cos_t, ggml_tensor * sin_t) {
    std::string a = p + "attention.self.";
    ggml_tensor * h = layer_norm(c, m, x, a + "layernorm1");
    ggml_tensor * q = rope_query(c, m, h, a, cos_t, sin_t);
    ggml_tensor * k = linear(c, m, h, a + "key");
    ggml_tensor * v = linear(c, m, h, a + "value");
    ggml_tensor * o = attention(c, q, k, v, N_HEAD_E, DH_E, 1.0f / sqrtf((float) DH_E));
    ggml_tensor * att = ggml_add(c, linear(c, m, o, p + "attention.output.dense"), x);
    return ggml_add(c, moe(c, m, layer_norm(c, m, att, p + "layernorm"), p), att);
}

struct RopeTabs { ggml_tensor *cos_lat, *sin_lat, *cos_tmp, *sin_tmp; std::vector<float> h[4]; };

// cos/sin tables [D_ENC, L], computed in float as the reference does.
static void rope_table(const std::vector<float> & freqs, bool pixel, int L, std::vector<float> & cs, std::vector<float> & sn) {
    cs.assign((size_t) D_ENC * L, 1.0f);
    sn.assign((size_t) D_ENC * L, 0.0f);
    for (int s = 0; s < L; ++s) {
        float pos;
        if (pixel) {
            // torch.linspace(-1, 1, L) in float32: symmetric halves
            const float step = 2.0f / (float) (L - 1);
            pos = s < L / 2 ? -1.0f + step * (float) s : 1.0f - step * (float) (L - 1 - s);
        } else {
            pos = (float) s;
        }
        for (int hd = 0; hd < N_HEAD_E; ++hd) {
            for (int d = 0; d < ROT_DIMS; ++d) {
                const float th = pos * freqs[d / 2];
                cs[(size_t) s * D_ENC + hd * DH_E + d] = cosf(th);
                sn[(size_t) s * D_ENC + hd * DH_E + d] = sinf(th);
            }
        }
    }
}

// Encoder (+ multi-channel projection). pre: [C=128, F=128, T, B] as captured.
static void build_encoder(ggml_context * c, Model & m, ggml_tensor * pre, RopeTabs & rt,
                          ggml_tensor ** enc_out, ggml_tensor ** pdec_out) {
    const int64_t T = pre->ne[2], B = pre->ne[3], N = T * B;
    // attention to channel: tokens are the 128 conv channels, features the 128 bins
    ggml_tensor * kv = ggml_cont(c, ggml_permute(c, pre, 1, 0, 2, 3));   // [F, C, T, B]
    kv = ggml_reshape_3d(c, kv, D_ENC, 128, N);
    ggml_tensor * lat = m.get("encoder.latent_array.latents");          // [D, K]
    ggml_tensor * h = ggml_repeat_4d(c, lat, D_ENC, N_LAT, N, 1);
    h = ggml_reshape_3d(c, h, D_ENC, N_LAT, N);
    for (int b = 0; b < 3; ++b) {
        std::string bp = "encoder.blocks." + std::to_string(b) + ".";
        std::string sp = bp + "spectral_cross_attention.";
        ggml_tensor * hq = layer_norm(c, m, h, sp + "self.layernorm1");
        ggml_tensor * hk = layer_norm(c, m, kv, sp + "self.layernorm2");
        ggml_tensor * q = rope_query(c, m, hq, sp + "self.", rt.cos_lat, rt.sin_lat);
        ggml_tensor * k = linear(c, m, hk, sp + "self.key");
        ggml_tensor * v = linear(c, m, hk, sp + "self.value");
        ggml_tensor * o = attention(c, q, k, v, N_HEAD_E, DH_E, 1.0f / sqrtf((float) DH_E));
        h = ggml_add(c, linear(c, m, o, sp + "output.dense"), h);
        for (int l = 0; l < 2; ++l)
            h = perceiver_layer(c, m, h, bp + "local_transformer." + std::to_string(l) + ".", rt.cos_lat, rt.sin_lat);
        ggml_tensor * ht = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, h, D_ENC, N_LAT, T, B), 0, 2, 1, 3));
        ht = ggml_reshape_3d(c, ht, D_ENC, T, N_LAT * B);                 // [D, T, K*B]
        for (int l = 0; l < 2; ++l)
            ht = perceiver_layer(c, m, ht, bp + "temporal_transformer." + std::to_string(l) + ".", rt.cos_tmp, rt.sin_tmp);
        h = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, ht, D_ENC, T, N_LAT, B), 0, 2, 1, 3));
        h = ggml_reshape_3d(c, h, D_ENC, N_LAT, N);
    }
    *enc_out = ggml_reshape_4d(c, h, D_ENC, N_LAT, T, B);
    // pairs of latents per stream: [256, 13, T, B] -> [256, T, 13, B] -> linear 256->512
    ggml_tensor * g = ggml_reshape_4d(c, h, 2 * D_ENC, N_CH, T, B);
    g = ggml_cont(c, ggml_permute(c, g, 0, 2, 1, 3));
    *pdec_out = mm(c, m.get("pre_decoder.0.linear.weight"), g);         // [512, T, 13, B]
}

// ---- decoder state ----
struct DecState {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * kc[N_LAYER_D], * vc[N_LAYER_D];   // self cache [64, MAX_TOK, 6, S]
    ggml_tensor * ck[N_LAYER_D], * cv[N_LAYER_D];   // cross K [64, T, 6, S], V^T [T, 64, 6, S]
    int S = 0, T = 0;
};

static void dec_state_init(DecState & st, ggml_backend_t be, int S, int T) {
    ggml_init_params ip = { 4 * N_LAYER_D * ggml_tensor_overhead() + 1024, nullptr, true };
    st.ctx = ggml_init(ip);
    st.S = S; st.T = T;
    for (int l = 0; l < N_LAYER_D; ++l) {
        st.kc[l] = ggml_new_tensor_4d(st.ctx, GGML_TYPE_F32, DH_D, MAX_TOK, N_HEAD_D, S);
        st.vc[l] = ggml_new_tensor_4d(st.ctx, GGML_TYPE_F32, DH_D, MAX_TOK, N_HEAD_D, S);
        st.ck[l] = ggml_new_tensor_4d(st.ctx, GGML_TYPE_F32, DH_D, T, N_HEAD_D, S);
        st.cv[l] = ggml_new_tensor_4d(st.ctx, GGML_TYPE_F32, T, DH_D, N_HEAD_D, S);
    }
    st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, be);
}

static std::string dl(int l, const char * rest) {
    return "decoder.decoder.block." + std::to_string(l) + ".layer." + rest;
}

// cross-attention K/V from the projected encoder states (once per segment batch)
static void build_cross_kv(ggml_context * c, ggml_cgraph * gf, Model & m, DecState & st, ggml_tensor * pdec) {
    const int T = st.T, S = st.S;
    ggml_tensor * e = ggml_reshape_3d(c, pdec, D_DEC, T, S);
    for (int l = 0; l < N_LAYER_D; ++l) {
        ggml_tensor * k = mm(c, m.get(dl(l, "1.EncDecAttention.k.weight")), e);   // [384, T, S]
        ggml_tensor * v = mm(c, m.get(dl(l, "1.EncDecAttention.v.weight")), e);
        k = ggml_permute(c, ggml_reshape_4d(c, k, DH_D, N_HEAD_D, T, S), 0, 2, 1, 3);  // [64, T, 6, S]
        v = ggml_permute(c, ggml_reshape_4d(c, v, DH_D, N_HEAD_D, T, S), 1, 2, 0, 3);  // [T, 64, 6, S]
        ggml_build_forward_expand(gf, ggml_cpy(c, k, st.ck[l]));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, st.cv[l]));
    }
}

// T5 gelu_new: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
static ggml_tensor * gelu_tanh(ggml_context * c, ggml_tensor * x) {
    if (!g_exact_gelu) return ggml_gelu(c, x);
    ggml_tensor * x3 = ggml_mul(c, ggml_mul(c, x, x), x);
    ggml_tensor * in = ggml_scale(c, ggml_add(c, x, ggml_scale(c, x3, 0.044715f)), 0.7978845608028654f);
    ggml_tensor * hx = ggml_scale(c, x, 0.5f);
    return ggml_add(c, hx, ggml_mul(c, hx, ggml_tanh(c, in)));
}

static ggml_tensor * rms(ggml_context * c, Model & m, ggml_tensor * x, const std::string & w) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), m.get(w));
}

// One decoder step for every stream at past length P. tok [S] i32, pos [1] i32.
static ggml_tensor * build_dec_step(ggml_context * c, ggml_cgraph * gf, Model & m, DecState & st,
                                    ggml_tensor * tok, ggml_tensor * pos, int P) {
    const int S = st.S, T = st.T, Lk = P + 1;
    ggml_tensor * x = ggml_get_rows(c, m.get("embed_tokens.weight"), tok);                 // [512, S]
    x = ggml_add(c, x, ggml_get_rows(c, m.get("decoder.decoder.additive_pe.weight"), pos));
    for (int l = 0; l < N_LAYER_D; ++l) {
        // self attention (T5: no score scaling); one new token, so no mask is needed
        ggml_tensor * h = rms(c, m, x, dl(l, "0.layer_norm.weight"));
        ggml_tensor * q = mm(c, m.get(dl(l, "0.SelfAttention.q.weight")), h);
        ggml_tensor * k = mm(c, m.get(dl(l, "0.SelfAttention.k.weight")), h);
        ggml_tensor * v = mm(c, m.get(dl(l, "0.SelfAttention.v.weight")), h);
        ggml_tensor * kslot = ggml_view_4d(c, st.kc[l], DH_D, 1, N_HEAD_D, S, st.kc[l]->nb[1], st.kc[l]->nb[2], st.kc[l]->nb[3], (size_t) P * st.kc[l]->nb[1]);
        ggml_tensor * vslot = ggml_view_4d(c, st.vc[l], DH_D, 1, N_HEAD_D, S, st.vc[l]->nb[1], st.vc[l]->nb[2], st.vc[l]->nb[3], (size_t) P * st.vc[l]->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_4d(c, k, DH_D, 1, N_HEAD_D, S), kslot));
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_4d(c, v, DH_D, 1, N_HEAD_D, S), vslot));
        ggml_tensor * K = ggml_view_4d(c, st.kc[l], DH_D, Lk, N_HEAD_D, S, st.kc[l]->nb[1], st.kc[l]->nb[2], st.kc[l]->nb[3], 0);
        ggml_tensor * V = ggml_view_4d(c, st.vc[l], DH_D, Lk, N_HEAD_D, S, st.vc[l]->nb[1], st.vc[l]->nb[2], st.vc[l]->nb[3], 0);
        ggml_tensor * q4 = ggml_reshape_4d(c, q, DH_D, 1, N_HEAD_D, S);
        ggml_tensor * kq = ggml_soft_max(c, mm(c, K, q4));                           // [Lk, 1, 6, S]
        ggml_tensor * vt = ggml_cont(c, ggml_permute(c, V, 1, 0, 2, 3));            // [Lk, 64, 6, S]
        ggml_tensor * o = mm(c, vt, kq);                                             // [64, 1, 6, S]
        o = ggml_reshape_2d(c, o, DH_D * N_HEAD_D, S);
        x = ggml_add(c, x, mm(c, m.get(dl(l, "0.SelfAttention.o.weight")), o));
        // cross attention
        h = rms(c, m, x, dl(l, "1.layer_norm.weight"));
        q4 = ggml_reshape_4d(c, mm(c, m.get(dl(l, "1.EncDecAttention.q.weight")), h), DH_D, 1, N_HEAD_D, S);
        kq = ggml_soft_max(c, mm(c, st.ck[l], q4));                                 // [T, 1, 6, S]
        o = ggml_reshape_2d(c, mm(c, st.cv[l], kq), DH_D * N_HEAD_D, S);
        x = ggml_add(c, x, mm(c, m.get(dl(l, "1.EncDecAttention.o.weight")), o));
        // gated-gelu FF (tanh gelu)
        h = rms(c, m, x, dl(l, "2.layer_norm.weight"));
        ggml_tensor * f = ggml_mul(c, gelu_tanh(c, mm(c, m.get(dl(l, "2.DenseReluDense.wi_0.weight")), h)),
                                   mm(c, m.get(dl(l, "2.DenseReluDense.wi_1.weight")), h));
        x = ggml_add(c, x, mm(c, m.get(dl(l, "2.DenseReluDense.wo.weight")), f));
    }
    (void) T;
    x = rms(c, m, x, "decoder.decoder.final_layer_norm.weight");
    x = ggml_scale(c, x, 1.0f / sqrtf((float) D_DEC));
    return mm(c, m.get("lm_head.lm_head.weight"), x);                                 // [596, S]
}

// ---- placement / profile ----
static void report_placement(const char * label, ggml_backend_sched_t sched, ggml_cgraph * gf) {
    std::map<std::string, int> per_be;
    std::map<std::string, int> off_gpu;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
            t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) continue;
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
        std::string bn = b ? ggml_backend_name(b) : "NULL";
        per_be[bn]++;
        if (bn.find("CUDA") == std::string::npos) off_gpu[std::string(ggml_op_desc(t))]++;
    }
    printf("placement %-10s nodes=%d splits=%d copies=%d |", label, n, ggml_backend_sched_get_n_splits(sched),
           ggml_backend_sched_get_n_copies(sched));
    for (auto & kv : per_be) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf(" | off-GPU:");
    if (off_gpu.empty()) printf(" none");
    for (auto & kv : off_gpu) printf(" %s x%d", kv.first.c_str(), kv.second);
    printf("\n");
}

struct Prof { double last = 0; std::map<std::string, std::pair<double, int>> by_op; };
static bool prof_cb(ggml_tensor * t, bool ask, void * ud) {
    Prof * p = (Prof *) ud;
    if (ask) return true;
    const double n = now_ms();
    auto & e = p->by_op[ggml_op_desc(t)];
    e.first += n - p->last;
    e.second++;
    p->last = n;
    return true;
}
static void print_prof(const char * label, Prof & p) {
    std::vector<std::pair<double, std::string>> v;
    double tot = 0;
    for (auto & kv : p.by_op) { v.push_back({kv.second.first, kv.first}); tot += kv.second.first; }
    std::sort(v.rbegin(), v.rend());
    printf("profile %s (per-node sync, total %.2f ms):", label, tot);
    for (size_t i = 0; i < v.size() && i < 10; ++i)
        printf(" %s %.2f (%d)", v[i].second.c_str(), v[i].first, p.by_op[v[i].second].second);
    printf("\n");
}

// ---- comparison ----
static double max_abs(const float * a, const float * b, size_t n, size_t * at = nullptr) {
    double m = 0; size_t w = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { if (at) *at = i; return INFINITY; }
        const double d = fabs((double) a[i] - (double) b[i]);
        if (d > m) { m = d; w = i; }
    }
    if (at) *at = w;
    return m;
}
static void argmax_rows(const float * x, int rows, int cols, std::vector<int> & out) {
    out.resize(rows);
    for (int r = 0; r < rows; ++r) {
        const float * p = x + (size_t) r * cols;
        out[r] = (int) (std::max_element(p, p + cols) - p);
    }
}
static void row(const char * point, double err, int ok_tok, int n_tok) {
    const bool pass = err < 1e-3 && ok_tok == n_tok;
    if (n_tok > 0) printf("PARITY | %-26s | max abs %.3e | argmax %d/%d | %s\n", point, err, ok_tok, n_tok, pass ? "PASS" : "MISS");
    else printf("PARITY | %-26s | max abs %.3e | - | %s\n", point, err, pass ? "PASS" : "MISS");
}

int main(int argc, char ** argv) {
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        // the gate must fail on any non-finite value, wherever it sits
        const float nan = NAN, inf = INFINITY;
        const float z[3] = {0, 0, 0};
        const float cases[4][3] = {{nan, 0, 0}, {0, nan, 0}, {0, 0, inf}, {0, 0.0005f, 0}};
        const bool want_fail[4] = {true, true, true, false};
        int bad = 0;
        for (int i = 0; i < 4; ++i) {
            const bool fails = !(max_abs(cases[i], z, 3) < 1e-3) || !(max_abs(z, cases[i], 3) < 1e-3);
            if (fails != want_fail[i]) { printf("self-test case %d wrong\n", i); bad++; }
        }
        printf("self-test: %s\n", bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }
    if (argc < 3) { fprintf(stderr, "usage: spike <model.gguf> <capture-dir> [--cpu] [--tf32] [--profile] [--reps N]\n"); return 2; }
    const std::string cap = argv[2];
    bool use_cpu = false, profile = false, dec_from_capture = false, no_eos_stop = false;
    int reps = 3;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--cpu")) use_cpu = true;
        else if (!strcmp(argv[i], "--tf32")) g_strict = false;
        else if (!strcmp(argv[i], "--profile")) profile = true;
        else if (!strcmp(argv[i], "--dec-from-capture")) dec_from_capture = true;
        else if (!strcmp(argv[i], "--no-eos-stop")) no_eos_stop = true;  // time the 256-token worst case
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    const char * be_dir = getenv("SPIKE_BACKEND_DIR");
    if (be_dir) ggml_backend_load_all_from_path(be_dir); else ggml_backend_load_all();

    const double t_load0 = now_ms();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = use_cpu ? nullptr : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t main_be = gpu ? gpu : cpu;
    g_exact_gelu = gpu == nullptr;
    printf("backend %s, matmul %s\n", ggml_backend_name(main_be), g_strict ? "strict f32" : "default (TF32 where ggml-cuda uses it)");

    Model m;
    load_model(m, argv[1], main_be);
    const double t_load = now_ms() - t_load0;
    printf("load: %.1f ms (gguf read + upload, backend init)\n", t_load);

    // captures
    const int B = 8, T = 110, S = B * N_CH;
    std::vector<float> pre = read_f32(cap + "/mt3.b0.pre-encoder.f32", (size_t) B * T * 128 * 128);
    std::vector<float> ref_enc = read_f32(cap + "/mt3.b0.encoder.f32", (size_t) B * T * N_LAT * D_ENC);
    std::vector<float> ref_pdec = read_f32(cap + "/mt3.b0.pre-decoder.f32", (size_t) B * N_CH * T * D_DEC);
    std::vector<float> ref_pre_lg = read_f32(cap + "/mt3.b0.prefill.logits.f32", (size_t) S * VOCAB);
    std::vector<float> ref_step_lg = read_f32(cap + "/mt3.b0.cached-step.logits.f32", (size_t) S * VOCAB);

    // rope tables
    RopeTabs rt;
    std::vector<float> cl, sl, ct, st_;
    rope_table(m.host.at("encoder.rotary_emb_latent.freqs"), true, N_LAT, cl, sl);
    rope_table(m.host.at("encoder.rotary_emb_temporal.freqs"), false, T, ct, st_);
    {   // the SCA module shares the latent table only if their freqs agree
        auto & a = m.host.at("encoder.rotary_emb_sca.freqs");
        auto & b = m.host.at("encoder.rotary_emb_latent.freqs");
        if (a != b) { fprintf(stderr, "sca and latent rope freqs differ\n"); return 1; }
    }

    ggml_backend_t bes[2] = { main_be, cpu };
    const int n_be = gpu ? 2 : 1;
    ggml_backend_sched_t sched = ggml_backend_sched_new(bes, nullptr, n_be, 65536, false, true);

    DecState ds;
    dec_state_init(ds, main_be, S, T);

    // ---------------- encoder + projection + cross K/V ----------------
    const size_t meta_sz = 65536 * ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false);
    std::vector<uint8_t> meta_buf(meta_sz);
    std::vector<float> enc_out((size_t) B * T * N_LAT * D_ENC), pdec_out((size_t) B * N_CH * T * D_DEC);
    double t_enc_best = 1e30, t_enc_first = 0;
    Prof penc;
    for (int rep = 0; rep < reps + (profile ? 1 : 0); ++rep) {
        const bool prof_run = profile && rep == reps;
        const double t0 = now_ms();   // graph build, alloc, upload, compute
        ggml_init_params ip = { meta_sz, meta_buf.data(), true };
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, false);
        ggml_tensor * x = ggml_new_tensor_4d(c, GGML_TYPE_F32, 128, 128, T, B); ggml_set_input(x);
        rt.cos_lat = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, N_LAT); ggml_set_input(rt.cos_lat);
        rt.sin_lat = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, N_LAT); ggml_set_input(rt.sin_lat);
        rt.cos_tmp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, T); ggml_set_input(rt.cos_tmp);
        rt.sin_tmp = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, T); ggml_set_input(rt.sin_tmp);
        ggml_tensor *enc, *pdec;
        g_router.clear();
        build_encoder(c, m, x, rt, &enc, &pdec);
        ggml_set_output(enc); ggml_set_output(pdec);
        ggml_build_forward_expand(gf, enc);
        ggml_build_forward_expand(gf, pdec);
        build_cross_kv(c, gf, m, ds, pdec);
        ggml_backend_sched_reset(sched);
        if (prof_run) { penc.last = now_ms(); ggml_backend_sched_set_eval_callback(sched, prof_cb, &penc); }
        else ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
        ggml_backend_tensor_set(x, pre.data(), 0, ggml_nbytes(x));
        ggml_backend_tensor_set(rt.cos_lat, cl.data(), 0, ggml_nbytes(rt.cos_lat));
        ggml_backend_tensor_set(rt.sin_lat, sl.data(), 0, ggml_nbytes(rt.sin_lat));
        ggml_backend_tensor_set(rt.cos_tmp, ct.data(), 0, ggml_nbytes(rt.cos_tmp));
        ggml_backend_tensor_set(rt.sin_tmp, st_.data(), 0, ggml_nbytes(rt.sin_tmp));
        if (prof_run) penc.last = now_ms();
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_synchronize(main_be);
        const double dt = now_ms() - t0;
        if (rep == 0) {
            t_enc_first = dt;
            report_placement("encoder", sched, gf);
            ggml_backend_tensor_get(enc, enc_out.data(), 0, ggml_nbytes(enc));
            ggml_backend_tensor_get(pdec, pdec_out.data(), 0, ggml_nbytes(pdec));
            // near-tie probe: smallest gap between the 2nd and 3rd router probability
            int li = 0;
            for (auto & r : g_router) {
                const bool temporal = r.first.find("temporal") != std::string::npos;
                const int64_t M = r.second->ne[1];
                std::vector<float> pr((size_t) N_EXPERT * M);
                ggml_backend_tensor_get(r.second, pr.data(), 0, pr.size() * sizeof(float));
                float best = 1; int64_t bi = 0; int ties = 0;
                for (int64_t i = 0; i < M; ++i) {
                    float v[N_EXPERT]; memcpy(v, &pr[(size_t) i * N_EXPERT], sizeof(v));
                    std::sort(v, v + N_EXPERT, std::greater<float>());
                    const float gap = v[1] - v[2];
                    if (gap < 1e-5f) {
                        ties++;
                        int64_t b2, t2, k2;
                        if (temporal) { t2 = i % T; k2 = (i / T) % N_LAT; b2 = i / (T * N_LAT); }
                        else { k2 = i % N_LAT; t2 = (i / N_LAT) % T; b2 = i / (N_LAT * T); }
                        if (getenv("SPIKE_TIES")) {
                            int ord[N_EXPERT];
                            for (int e = 0; e < N_EXPERT; ++e) ord[e] = e;
                            const float * pe = &pr[(size_t) i * N_EXPERT];
                            std::sort(ord, ord + N_EXPERT, [&](int a, int b) { return pe[a] > pe[b]; });
                            printf("    tie layer %d b=%lld t=%lld k=%lld gap %.2e top3 e%d %.7f e%d %.7f e%d %.7f\n", li,
                                   (long long) b2, (long long) t2, (long long) k2, gap,
                                   ord[0], pe[ord[0]], ord[1], pe[ord[1]], ord[2], pe[ord[2]]);
                        }
                    }
                    if (gap < best) { best = gap; bi = i; }
                }
                int64_t b_, t_, k_;
                if (temporal) { t_ = bi % T; k_ = (bi / T) % N_LAT; b_ = bi / (T * N_LAT); }
                else { k_ = bi % N_LAT; t_ = (bi / N_LAT) % T; b_ = bi / (N_LAT * T); }
                printf("  router %2d %-44s min 2nd-3rd gap %.2e at b=%lld t=%lld k=%lld, gaps<1e-5: %d\n", li++,
                       r.first.c_str() + 15, best, (long long) b_, (long long) t_, (long long) k_, ties);
            }
        }
        if (!prof_run) t_enc_best = std::min(t_enc_best, dt);
        ggml_free(c);
    }
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    if (profile) print_prof("encoder+proj+crossKV", penc);

    size_t at = 0;
    double e_enc = max_abs(enc_out.data(), ref_enc.data(), enc_out.size(), &at);
    row("encoder", e_enc, 0, 0);
    {   // error spread: elements over the gate and the latent tokens they sit in
        size_t over = 0; std::map<size_t, int> toks; double sum = 0;
        for (size_t i = 0; i < enc_out.size(); ++i) {
            const double d = fabs((double) enc_out[i] - ref_enc[i]);
            sum += d;
            if (d >= 1e-3) { over++; toks[i / D_ENC]++; }
        }
        printf("  encoder: mean abs %.3e, %zu of %zu elements >= 1e-3, in %zu of %zu (b,t,k) tokens\n",
               sum / enc_out.size(), over, enc_out.size(), toks.size(), enc_out.size() / D_ENC);
        int shown = 0;
        for (auto & kv : toks) {
            if (shown++ >= 6) break;
            const size_t tk = kv.first;
            printf("    token b=%zu t=%zu k=%zu: %d elements\n", tk / (N_LAT * T), (tk / N_LAT) % T, tk % N_LAT, kv.second);
        }
        if (e_enc >= 1e-3) printf("  worst at flat %zu: got %.6f ref %.6f\n", at, enc_out[at], ref_enc[at]);
    }
    double e_pd = max_abs(pdec_out.data(), ref_pdec.data(), pdec_out.size(), &at);
    row("pre-decoder", e_pd, 0, 0);

    if (dec_from_capture) {
        // rebuild the cross K/V from the captured pre-decoder states, isolating the decoder
        ggml_init_params ip = { meta_sz, meta_buf.data(), true };
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, false);
        ggml_tensor * pd = ggml_new_tensor_4d(c, GGML_TYPE_F32, D_DEC, T, N_CH, B); ggml_set_input(pd);
        build_cross_kv(c, gf, m, ds, pd);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(pd, ref_pdec.data(), 0, ggml_nbytes(pd));
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_free(c);
        printf("decoder input: captured pre-decoder states\n");
    } else {
        printf("decoder input: this run's pre-decoder states\n");
    }

    // ---------------- decoder ----------------
    // tokens: [batch][segment][channel][len] in the capture; parse the one batch
    std::vector<std::vector<int>> ref_tok(S);
    {
        std::ifstream f(cap + "/tokens-by-batch-channel.json");
        std::string js((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        // depth-3 arrays are the per-stream token lists, in [segment][channel] order
        int depth = 0, idx = -1; std::string num;
        for (char ch : js) {
            if (ch == '[') { depth++; if (depth == 4) idx++; continue; }
            if (ch == ']' || ch == ',') {
                if (!num.empty() && depth == 4) ref_tok[idx].push_back(atoi(num.c_str()));
                num.clear();
                if (ch == ']') depth--;
                continue;
            }
            if ((ch >= '0' && ch <= '9') || ch == '-') num += ch;
        }
        if (idx != S - 1) { fprintf(stderr, "token json: %d streams\n", idx + 1); return 1; }
    }

    std::vector<int> cur(S, 0);           // BOS
    std::vector<int> unfinished(S, 1);
    std::vector<std::vector<int>> got(S);
    std::vector<float> lg((size_t) S * VOCAB);
    double t_dec_total = 0;
    int steps = 0;
    Prof pdec_prof;
    std::vector<double> step_ms;
    const double t_loop0 = now_ms();
    for (int P = 0; P < MAX_TOK; ++P) {
        const double t0 = now_ms();   // graph build, reset, alloc, compute, readback, argmax
        ggml_init_params ip = { meta_sz, meta_buf.data(), true };
        ggml_context * c = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, false);
        ggml_tensor * tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, S); ggml_set_input(tok);
        ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(pos);
        ggml_tensor * logits = build_dec_step(c, gf, m, ds, tok, pos, P);
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
        ggml_backend_sched_reset(sched);
        const bool prof_step = profile && P == 1;
        if (prof_step) ggml_backend_sched_set_eval_callback(sched, prof_cb, &pdec_prof);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }
        ggml_backend_tensor_set(tok, cur.data(), 0, S * sizeof(int));
        int32_t p32 = P;
        ggml_backend_tensor_set(pos, &p32, 0, sizeof(p32));
        if (prof_step) pdec_prof.last = now_ms();
        ggml_backend_sched_graph_compute(sched, gf);
        ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
        std::vector<int> am;
        argmax_rows(lg.data(), S, VOCAB, am);
        const double dt = now_ms() - t0;
        if (prof_step) ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
        if (P == 0) report_placement("dec-step", sched, gf);
        if (!prof_step) { t_dec_total += dt; step_ms.push_back(dt); }
        ggml_free(c);
        steps++;
        cur = am;
        if (P == 0) {
            std::vector<int> ra; argmax_rows(ref_pre_lg.data(), S, VOCAB, ra);
            int ok = 0; for (int s = 0; s < S; ++s) ok += am[s] == ra[s];
            row("prefill logits", max_abs(lg.data(), ref_pre_lg.data(), lg.size()), ok, S);
            // the reference cached step is fed the argmax of ITS prefill logits
            cur = ra;
        } else if (P == 1) {
            std::vector<int> ra; argmax_rows(ref_step_lg.data(), S, VOCAB, ra);
            int ok = 0; for (int s = 0; s < S; ++s) ok += am[s] == ra[s];
            row("cached step logits", max_abs(lg.data(), ref_step_lg.data(), lg.size()), ok, S);
        }
        // reference loop: the prefill token is kept as is; later tokens are padded
        // once a stream has emitted EOS, and the loop ends when every stream has
        bool any = P == 0;
        for (int s = 0; s < S; ++s) {
            got[s].push_back(unfinished[s] ? am[s] : 0);
            if (P > 0 && unfinished[s] && am[s] == 1) unfinished[s] = 0;
            any |= unfinished[s] != 0;
        }
        if (!any && !no_eos_stop) break;
    }
    const double t_loop = now_ms() - t_loop0;
    int tok_ok = 0;
    for (int s = 0; s < S; ++s) tok_ok += got[s] == ref_tok[s];
    printf("PARITY | %-26s | streams equal %d/%d | len %zu vs ref %zu | %s\n", "full greedy decode", tok_ok, S,
           got[0].size(), ref_tok[0].size(), tok_ok == S ? "PASS" : "MISS");
    if (profile) print_prof("decoder step P=1", pdec_prof);

    std::sort(step_ms.begin(), step_ms.end());
    printf("TIME | encoder+proj+crossKV per batch of %d: best %.2f ms, first %.2f ms\n", B, t_enc_best, t_enc_first);
    printf("TIME | decode: %d steps, %zu timed (profiled step excluded), sum %.2f ms, median step %.3f ms, min %.3f, max %.3f"
           " (each: graph build, reset, alloc, compute, logits readback, argmax)\n",
           steps, step_ms.size(), t_dec_total, step_ms[step_ms.size() / 2], step_ms.front(), step_ms.back());
    printf("TIME | decode loop wall %.2f ms for %d steps%s\n", t_loop, steps,
           profile ? " (includes one per-node-synced profile step and parity checks; time without --profile)"
                   : " (includes parity checks)");

    // decode-step cost at the far end of the cache, for the 256-token worst case
    {
        for (int P : {127, 255}) {
            for (int r = 0; r < 3; ++r) {
                const double t0 = now_ms();
                ggml_init_params ip = { meta_sz, meta_buf.data(), true };
                ggml_context * c = ggml_init(ip);
                ggml_cgraph * gf = ggml_new_graph_custom(c, 65536, false);
                ggml_tensor * tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, S); ggml_set_input(tok);
                ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1); ggml_set_input(pos);
                ggml_tensor * logits = build_dec_step(c, gf, m, ds, tok, pos, P);
                ggml_set_output(logits);
                ggml_build_forward_expand(gf, logits);
                ggml_backend_sched_reset(sched);
                ggml_backend_sched_alloc_graph(sched, gf);
                ggml_backend_tensor_set(tok, cur.data(), 0, S * sizeof(int));
                int32_t p32 = P;
                ggml_backend_tensor_set(pos, &p32, 0, sizeof(p32));
                ggml_backend_sched_graph_compute(sched, gf);
                ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
                std::vector<int> am;
                argmax_rows(lg.data(), S, VOCAB, am);
                const double dt = now_ms() - t0;
                ggml_free(c);
                if (r == 2) { printf("TIME | decode step at P=%d: %.3f ms\n", P, dt); }
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(ds.buf); ggml_free(ds.ctx);
    ggml_backend_buffer_free(m.buf); ggml_free(m.ctx);
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}
