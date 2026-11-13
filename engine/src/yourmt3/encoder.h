#pragma once
// yourmt3/encoder.h: the YourMT3+ conv pre-encoder (three residual 3x3 blocks,
// eval BatchNorm, ReLU, 1x2 average pooling over frequency) and the
// Perceiver-TF encoder (26 latents, three blocks of spectral cross-attention,
// two local and two temporal self-attention layers with a top-2 MoE of eight
// SiLU-gated experts, partial RoPE on the queries) on any ggml backend.
// Written from the capture contract, the inventory and the 6b spike
// (tools/lead-sheet-port-spike), whose numerics rules apply:
//   - every matmul carries GGML_HINT_STRICT_F32 (TF32 moved the encoder by ~1);
//   - LayerNorm is two-pass, rms_norm(x - mean(x)): ggml-cuda's one-pass
//     variance flipped near-tied top-2 router choices;
//   - partial RoPE rotates the first 4 of each head's 16 dims, queries only,
//     with positions linspace(-1, 1) over latents and 0..T-1 over time.
// The pre-encoder's F32 convolutions run ggml-cuda's fp32 direct kernel.

#include <chrono>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "yourmt3/features.h"
#include "yourmt3/weights.h"

namespace yourmt3 {

constexpr int GRAPH_NODES = 8192;

struct Encoder {
    Weights              w;
    BackendPair          bp    = {};
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
    std::vector<float>   cos_lat, sin_lat, cos_tmp, sin_tmp;  // [D_ENC][L]
    bool                 placement_shown[2] = { false, false };
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// cos/sin tables [L][D_ENC] as float, cos 1 / sin 0 on the unrotated dims.
static void rope_table(const std::vector<float> & freqs, bool pixel, int L, std::vector<float> & cs, std::vector<float> & sn) {
    cs.assign((size_t) D_ENC * L, 1.0f);
    sn.assign((size_t) D_ENC * L, 0.0f);
    for (int s = 0; s < L; ++s) {
        float pos = (float) s;
        if (pixel) {  // torch.linspace(-1, 1, L) in float32: symmetric halves
            const float step = 2.0f / (float) (L - 1);
            pos = s < L / 2 ? -1.0f + step * (float) s : 1.0f - step * (float) (L - 1 - s);
        }
        for (int hd = 0; hd < N_HEAD_E; ++hd)
            for (int d = 0; d < ROT_DIMS; ++d) {
                const float th = pos * freqs[d / 2];
                cs[(size_t) s * D_ENC + hd * DH_E + d] = std::cos(th);
                sn[(size_t) s * D_ENC + hd * DH_E + d] = std::sin(th);
            }
    }
}

static bool encoder_init(Encoder & e, const std::string & gguf_path) {
    e.bp = backend_init("YourMT3");
    if (!load_weights(e.w, gguf_path, e.bp.backend)) {
        backend_release(e.bp.backend, e.bp.cpu_backend);
        return false;
    }
    e.sched = backend_sched_new(e.bp, GRAPH_NODES);
    e.meta.resize(GRAPH_NODES * ggml_tensor_overhead() + ggml_graph_overhead_custom(GRAPH_NODES, false));
    rope_table(e.w.freqs_lat, true, N_LAT, e.cos_lat, e.sin_lat);
    rope_table(e.w.freqs_tmp, false, NFRAME, e.cos_tmp, e.sin_tmp);
    return true;
}

static void encoder_free(Encoder & e) {
    if (e.sched) ggml_backend_sched_free(e.sched);
    e.sched = nullptr;
    free_weights(e.w);
    backend_release(e.bp.backend, e.bp.cpu_backend);
    e.bp = {};
}

// ---- graph helpers ----
static ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(c, w, x);
    ggml_mul_mat_set_hint(r, GGML_HINT_STRICT_F32);
    return r;
}
// ggml-cuda copies the node's op_params into the per-expert slices, so the
// strict hint reaches cuBLAS there too (op_params[1] is otherwise unused by MUL_MAT_ID).
static ggml_tensor * mm_id(ggml_context * c, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) {
    ggml_tensor * r = ggml_mul_mat_id(c, w, x, ids);
    r->op_params[1] = GGML_HINT_STRICT_F32;
    return r;
}
static ggml_tensor * linear(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & p) {
    return ggml_add(c, mm(c, w.get(p + ".weight"), x), w.get(p + ".bias"));
}
static ggml_tensor * layer_norm(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & p) {
    x = ggml_rms_norm(c, ggml_sub(c, x, ggml_mean(c, x)), 1e-5f);
    return ggml_add(c, ggml_mul(c, x, w.get(p + ".weight")), w.get(p + ".bias"));
}

// Multi-head attention over the leading axis. q [D, Lq, N], k/v [D, Lk, N].
static ggml_tensor * attention(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {
    const int64_t Lq = q->ne[1], Lk = k->ne[1], N = q->ne[2];
    ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, DH_E, N_HEAD_E, Lq, N), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(c, ggml_reshape_4d(c, k, DH_E, N_HEAD_E, Lk, N), 0, 2, 1, 3);
    ggml_tensor * vt = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, v, DH_E, N_HEAD_E, Lk, N), 1, 2, 0, 3));
    ggml_tensor * kq = ggml_soft_max_ext(c, mm(c, ggml_cont(c, k4), ggml_cont(c, q4)), nullptr, 1.0f / sqrtf((float) DH_E), 0.0f);
    ggml_tensor * o  = ggml_cont(c, ggml_permute(c, mm(c, vt, kq), 0, 2, 1, 3));  // [dh, H, Lq, N]
    return ggml_reshape_3d(c, o, D_ENC, Lq, N);
}

// q*cos + rotate_half(q)*sin, rotate_half from the row-swapped projection.
static ggml_tensor * rope_query(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & p,
                                ggml_tensor * cs, ggml_tensor * sn) {
    return ggml_add(c, ggml_mul(c, linear(c, w, x, p + "query"), cs), ggml_mul(c, linear(c, w, x, p + "query_swap"), sn));
}

static ggml_tensor * moe(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & p) {
    const int64_t M  = x->ne[1] * x->ne[2];
    ggml_tensor * x2 = ggml_reshape_2d(c, x, D_ENC, M);
    ggml_tensor * pr = ggml_soft_max(c, mm(c, w.get(p + "mlp.gate.weight"), x2));      // [E, M]
    ggml_tensor * id = ggml_argsort_top_k(c, pr, TOPK);                                 // [K, M]
    ggml_tensor * g  = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, pr, 1, N_EXPERT, M), id), TOPK, M);
    g                = ggml_div(c, g, ggml_sum_rows(c, g));
    ggml_tensor * xe = ggml_reshape_3d(c, x2, D_ENC, 1, M);
    ggml_tensor * h  = ggml_mul(c, ggml_silu(c, mm_id(c, w.get(p + "mlp.w1_stack"), xe, id)),
                                mm_id(c, w.get(p + "mlp.gate_stack"), xe, id));        // [F, K, M]
    ggml_tensor * dn = ggml_mul(c, mm_id(c, w.get(p + "mlp.w2_stack"), h, id), ggml_reshape_3d(c, g, 1, TOPK, M));
    ggml_tensor * o  = ggml_add(c, ggml_view_2d(c, dn, D_ENC, M, dn->nb[2], 0), ggml_view_2d(c, dn, D_ENC, M, dn->nb[2], dn->nb[1]));
    return ggml_reshape_3d(c, o, D_ENC, x->ne[1], x->ne[2]);
}

static ggml_tensor * perceiver_layer(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & p,
                                     ggml_tensor * cs, ggml_tensor * sn) {
    const std::string a = p + "attention.self.";
    ggml_tensor *     h = layer_norm(c, w, x, a + "layernorm1");
    ggml_tensor *     o = attention(c, rope_query(c, w, h, a, cs, sn), linear(c, w, h, a + "key"), linear(c, w, h, a + "value"));
    ggml_tensor *     r = ggml_add(c, linear(c, w, o, p + "attention.output.dense"), x);
    return ggml_add(c, moe(c, w, layer_norm(c, w, r, p + "layernorm"), p), r);
}

// spec ne [NBIN, NFRAME, 1, B] -> [C=128, F=128, T, B] (torch (B, T, F, C))
static ggml_tensor * build_pre_encoder(ggml_context * c, const Weights & w, ggml_tensor * x) {
    for (int b = 0; b < 3; b++) {
        const std::string p  = "pre_encoder.0.blocks." + std::to_string(b) + ".";
        auto              bn = [&](ggml_tensor * h, const char * n) {
            return ggml_add(c, ggml_mul(c, h, w.get(p + n + ".scale")), w.get(p + n + ".shift"));
        };
        ggml_tensor * h = ggml_relu(c, bn(ggml_conv_2d_direct(c, w.get(p + "conv1.weight"), x, 1, 1, 1, 1, 1, 1), "bn1"));
        h               = bn(ggml_conv_2d_direct(c, w.get(p + "conv2.weight"), h, 1, 1, 1, 1, 1, 1), "bn2");
        ggml_tensor * s = x;
        if (b == 0)
            s = ggml_add(c, ggml_conv_2d_direct(c, w.get(p + "shortcut.weight"), x, 1, 1, 0, 0, 1, 1),
                         ggml_reshape_4d(c, w.get(p + "shortcut.bias"), 1, 1, N_CONV_CH, 1));
        x = ggml_pool_2d(c, ggml_relu(c, ggml_add(c, h, s)), GGML_OP_POOL_AVG, 2, 1, 2, 1, 0, 0);
    }
    return ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
}

struct Rope { ggml_tensor *cl, *sl, *ct, *st; };

// pre [C=128, F=128, T, B] -> encoder states [D_ENC, N_LAT, T, B] (torch (B, T, 26, 128))
static ggml_tensor * build_encoder(ggml_context * c, const Weights & w, ggml_tensor * pre, const Rope & r) {
    const int64_t T = pre->ne[2], B = pre->ne[3], N = T * B;
    // attention to channel: the 128 conv channels are the tokens, the 128 bins their features
    ggml_tensor * kv = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, pre, 1, 0, 2, 3)), D_ENC, N_CONV_CH, N);
    ggml_tensor * h  = ggml_reshape_3d(c, ggml_repeat_4d(c, w.get("encoder.latent_array.latents"), D_ENC, N_LAT, N, 1), D_ENC, N_LAT, N);
    for (int b = 0; b < N_BLOCK; b++) {
        const std::string bp = "encoder.blocks." + std::to_string(b) + ".";
        const std::string sp = bp + "spectral_cross_attention.";
        ggml_tensor *     hk = layer_norm(c, w, kv, sp + "self.layernorm2");
        ggml_tensor *     q  = rope_query(c, w, layer_norm(c, w, h, sp + "self.layernorm1"), sp + "self.", r.cl, r.sl);
        ggml_tensor *     o  = attention(c, q, linear(c, w, hk, sp + "self.key"), linear(c, w, hk, sp + "self.value"));
        h = ggml_add(c, linear(c, w, o, sp + "output.dense"), h);  // query residual
        for (int l = 0; l < N_SUB; l++) h = perceiver_layer(c, w, h, bp + "local_transformer." + std::to_string(l) + ".", r.cl, r.sl);
        ggml_tensor * ht = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, h, D_ENC, N_LAT, T, B), 0, 2, 1, 3));
        ht = ggml_reshape_3d(c, ht, D_ENC, T, N_LAT * B);
        for (int l = 0; l < N_SUB; l++) ht = perceiver_layer(c, w, ht, bp + "temporal_transformer." + std::to_string(l) + ".", r.ct, r.st);
        h = ggml_reshape_3d(c, ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, ht, D_ENC, T, N_LAT, B), 0, 2, 1, 3)), D_ENC, N_LAT, N);
    }
    return ggml_reshape_4d(c, h, D_ENC, N_LAT, T, B);
}

// Backend placement of a scheduled graph: node and split counts, ops off the main backend.
static void report_placement(const char * label, ggml_backend_sched_t sched, ggml_cgraph * gf, ggml_backend_t main_be) {
    std::map<std::string, int> per_be, off;
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
            t->op == GGML_OP_TRANSPOSE)
            continue;
        ggml_backend_t b  = ggml_backend_sched_get_tensor_backend(sched, t);
        std::string    bn = b ? ggml_backend_name(b) : "unassigned";
        per_be[bn]++;
        if (b != main_be) off[ggml_op_desc(t)]++;
    }
    printf("PLACEMENT %-12s nodes=%d splits=%d copies=%d |", label, n, ggml_backend_sched_get_n_splits(sched),
           ggml_backend_sched_get_n_copies(sched));
    for (auto & kv : per_be) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf(" | off %s:", ggml_backend_name(main_be));
    if (off.empty()) printf(" none");
    for (auto & kv : off) printf(" %s x%d", kv.first.c_str(), kv.second);
    printf("\n");
}

// One graph from `in` (shape ne) through `build`; returns the output. Placement printed once per kind.
template <typename Build>
static std::vector<float> run_graph(Encoder & e, int kind, const char * label, const int64_t ne[4], const float * in,
                                    Build build) {
    ggml_init_params ip = { e.meta.size(), e.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    x  = ggml_new_tensor_4d(c, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_input(x);
    Rope r = {};
    ggml_tensor * out = build(c, x, r);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_reset(e.sched);
    // The scheduler pins graph inputs to the CPU, and the weightless ops that
    // read them (the channel-token permute and the SCA LayerNorm statistics)
    // would follow: 6 splits. Inputs go straight to the compute backend instead.
    for (ggml_tensor * t : { x, r.cl, r.sl, r.ct, r.st })
        if (t) ggml_backend_sched_set_tensor_backend(e.sched, t, e.bp.backend);
    if (!ggml_backend_sched_alloc_graph(e.sched, gf)) {
        fprintf(stderr, "[yourmt3] %s graph allocation failed\n", label);
        exit(1);
    }
    ggml_backend_tensor_set(x, in, 0, ggml_nbytes(x));
    if (r.cl) {
        ggml_backend_tensor_set(r.cl, e.cos_lat.data(), 0, ggml_nbytes(r.cl));
        ggml_backend_tensor_set(r.sl, e.sin_lat.data(), 0, ggml_nbytes(r.sl));
        ggml_backend_tensor_set(r.ct, e.cos_tmp.data(), 0, ggml_nbytes(r.ct));
        ggml_backend_tensor_set(r.st, e.sin_tmp.data(), 0, ggml_nbytes(r.st));
    }
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(e.sched, gf))) {
        fprintf(stderr, "[yourmt3] %s graph compute failed\n", label);
        exit(1);
    }
    std::vector<float> y((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
    if (!e.placement_shown[kind]) {
        report_placement(label, e.sched, gf, e.bp.backend);
        e.placement_shown[kind] = true;
    }
    ggml_free(c);
    return y;
}

// spectrogram [B][NFRAME][NBIN] -> pre-encoder [B][NFRAME][128][128]
static std::vector<float> pre_encode(Encoder & e, const float * spec, int B) {
    const int64_t ne[4] = { NBIN, NFRAME, 1, B };
    return run_graph(e, 0, "pre-encoder", ne, spec,
                     [&](ggml_context * c, ggml_tensor * x, Rope &) { return build_pre_encoder(c, e.w, x); });
}

// pre-encoder [B][NFRAME][128][128] -> encoder [B][NFRAME][N_LAT][D_ENC]
static std::vector<float> encode(Encoder & e, const float * pre, int B) {
    const int64_t ne[4] = { N_CONV_CH, 128, NFRAME, B };
    return run_graph(e, 1, "encoder", ne, pre, [&](ggml_context * c, ggml_tensor * x, Rope & r) {
        r.cl = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, N_LAT);
        r.sl = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, N_LAT);
        r.ct = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, NFRAME);
        r.st = ggml_new_tensor_2d(c, GGML_TYPE_F32, D_ENC, NFRAME);
        for (ggml_tensor * t : { r.cl, r.sl, r.ct, r.st }) ggml_set_input(t);
        return build_encoder(c, e.w, x, r);
    });
}

// Mono audio at any integer rate through the encoder, in batches of BATCH
// segments. Keeps every intermediate (validation reads them); times each stage.
struct FrontHalf {
    std::vector<float> wav;   // [n_seg][SEG] padded 16 kHz
    std::vector<float> spec;  // [n_seg][NFRAME][NBIN]
    std::vector<float> pre;   // [n_seg][NFRAME][128][128]
    std::vector<float> enc;   // [n_seg][NFRAME][N_LAT][D_ENC]
    int    n_seg = 0;
    double ms_resample = 0, ms_spec = 0, ms_pre = 0, ms_enc = 0;
};

static void front_half(Encoder & e, const std::vector<float> & mono, int sr, FrontHalf & f, bool keep = true) {
    double t0 = now_ms();
    f.wav = pad_segments(resample(mono, sr, SR), &f.n_seg);
    double t1 = now_ms();
    f.spec = spectrogram(f.wav.data(), f.n_seg, e.w.window.data());
    double t2 = now_ms();
    const size_t n_pre = (size_t) NFRAME * 128 * 128, n_enc = (size_t) NFRAME * N_LAT * D_ENC;
    if (keep) {
        f.pre.resize(f.n_seg * n_pre);
        f.enc.resize(f.n_seg * n_enc);
    }
    for (int b0 = 0; b0 < f.n_seg; b0 += BATCH) {
        const int B = std::min(BATCH, f.n_seg - b0);
        double    a = now_ms();
        std::vector<float> pre = pre_encode(e, &f.spec[(size_t) b0 * NFRAME * NBIN], B);
        double    m = now_ms();
        std::vector<float> enc = encode(e, pre.data(), B);
        double    z = now_ms();
        f.ms_pre += m - a;
        f.ms_enc += z - m;
        if (keep) {
            std::copy(pre.begin(), pre.end(), f.pre.begin() + (size_t) b0 * n_pre);
            std::copy(enc.begin(), enc.end(), f.enc.begin() + (size_t) b0 * n_enc);
        }
    }
    f.ms_resample = t1 - t0;
    f.ms_spec     = t2 - t1;
}

}  // namespace yourmt3
