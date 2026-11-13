#pragma once
// yourmt3/decoder.h: the YourMT3+ multi-channel projection and 13-stream T5
// decoder, with greedy decoding to upstream's stop rule
// (t5mod_helper.task_cond_dec_generate, Apache-2.0; see LICENSE).
//
//   projection  encoder [D_ENC, 26, T, B] -> pairs of latents per channel
//               [256, 13, T, B] -> one shared linear to [512, T, 13, B]
//   cross K/V   per layer from the projected states, once per batch
//   step        token embedding + learned position, then 8 T5 layers (RMS
//               norm eps 1e-6, unscaled self- and cross-attention over 6 heads
//               of 64, gated tanh-GELU feed-forward), final norm, x / sqrt(512),
//               LM head. One new token per stream against a KV cache.
//   greedy      BOS (0) for every stream; the prefill token is kept as it is;
//               later tokens of a stream that has emitted EOS (1) are PAD (0);
//               the batch stops when every stream has, or at 256 tokens.
// The 104 streams of a full batch are segment-major: stream = segment * 13 + channel.
// The 6b spike's numerics rules apply (strict F32 matmuls; tanh GELU from basic
// ops on the CPU backend, whose ggml_gelu reads an fp16 table).

#include <cstring>
#include <vector>

#include "yourmt3/encoder.h"

namespace yourmt3 {

constexpr int N_HEAD_D = 6, DH_D = 64, MAX_TOK = 256, BOS = 0, EOS = 1, PAD = 0;

struct Decoder {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         kc[N_LAYER_D] = {}, * vc[N_LAYER_D] = {};  // self cache [64, MAX_TOK, 6, S]
    ggml_tensor *         ck[N_LAYER_D] = {}, * cv[N_LAYER_D] = {};  // cross K [64, T, 6, S], V^T [T, 64, 6, S]
    int                   S = 0;
    bool                  placement_shown[2] = { false, false };
};

static void decoder_free(Decoder & d) {
    if (d.buf) ggml_backend_buffer_free(d.buf);
    if (d.ctx) ggml_free(d.ctx);
    d.buf = nullptr;
    d.ctx = nullptr;
    d.S   = 0;
}

// Cache tensors for S streams on the compute backend (reallocated when S changes).
static bool decoder_alloc(Decoder & d, Encoder & e, int S) {
    if (d.S == S) return true;
    decoder_free(d);
    ggml_init_params ip = { 4 * N_LAYER_D * ggml_tensor_overhead() + 1024, nullptr, true };
    d.ctx = ggml_init(ip);
    for (int l = 0; l < N_LAYER_D; l++) {
        d.kc[l] = ggml_new_tensor_4d(d.ctx, GGML_TYPE_F32, DH_D, MAX_TOK, N_HEAD_D, S);
        d.vc[l] = ggml_new_tensor_4d(d.ctx, GGML_TYPE_F32, DH_D, MAX_TOK, N_HEAD_D, S);
        d.ck[l] = ggml_new_tensor_4d(d.ctx, GGML_TYPE_F32, DH_D, NFRAME, N_HEAD_D, S);
        d.cv[l] = ggml_new_tensor_4d(d.ctx, GGML_TYPE_F32, NFRAME, DH_D, N_HEAD_D, S);
    }
    d.buf = ggml_backend_alloc_ctx_tensors(d.ctx, e.bp.backend);
    if (!d.buf) {
        fprintf(stderr, "[yourmt3] decoder cache allocation failed (%d streams)\n", S);
        decoder_free(d);
        return false;
    }
    d.S = S;
    return true;
}

static std::string dec_name(int l, const char * rest) { return "decoder.decoder.block." + std::to_string(l) + ".layer." + rest; }

static ggml_tensor * t5_norm(ggml_context * c, const Weights & w, ggml_tensor * x, const std::string & name) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), w.get(name));
}

// T5 gelu_new: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
static ggml_tensor * gelu_tanh(ggml_context * c, ggml_tensor * x, bool composite) {
    if (!composite) return ggml_gelu(c, x);
    ggml_tensor * x3 = ggml_mul(c, ggml_mul(c, x, x), x);
    ggml_tensor * in = ggml_scale(c, ggml_add(c, x, ggml_scale(c, x3, 0.044715f)), 0.7978845608028654f);
    ggml_tensor * hx = ggml_scale(c, x, 0.5f);
    return ggml_add(c, hx, ggml_mul(c, hx, ggml_tanh(c, in)));
}

static bool backend_is_cpu(ggml_backend_t be) {
    return ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Allocate, upload the inputs, compute. Inputs go to the compute backend (no splits).
static void dec_compute(Encoder & e, Decoder & d, int kind, const char * label, ggml_cgraph * gf,
                        const std::vector<std::pair<ggml_tensor *, const void *>> & inputs) {
    ggml_backend_sched_reset(e.sched);
    for (auto & in : inputs) ggml_backend_sched_set_tensor_backend(e.sched, in.first, e.bp.backend);
    if (!ggml_backend_sched_alloc_graph(e.sched, gf)) {
        fprintf(stderr, "[yourmt3] %s graph allocation failed\n", label);
        exit(1);
    }
    for (auto & in : inputs) ggml_backend_tensor_set(in.first, in.second, 0, ggml_nbytes(in.first));
    if (!backend_compute_ok(ggml_backend_sched_graph_compute(e.sched, gf))) {
        fprintf(stderr, "[yourmt3] %s graph compute failed\n", label);
        exit(1);
    }
    if (!d.placement_shown[kind]) {
        report_placement(label, e.sched, gf, e.bp.backend);
        d.placement_shown[kind] = true;
    }
}

// Projection and cross K/V for B segments. `in` is the encoder output
// [B][T][26][128], or with from_pdec the projected states [B][13][T][512].
// Returns the projected states (empty with from_pdec).
static std::vector<float> decoder_prepare(Encoder & e, Decoder & d, const float * in, int B, bool from_pdec) {
    if (!decoder_alloc(d, e, B * N_CH)) exit(1);
    ggml_init_params ip = { e.meta.size(), e.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    const int        T  = NFRAME, S = d.S;
    ggml_tensor *    x  = from_pdec ? ggml_new_tensor_4d(c, GGML_TYPE_F32, D_DEC, T, N_CH, B)
                                    : ggml_new_tensor_4d(c, GGML_TYPE_F32, D_ENC, N_LAT, T, B);
    ggml_set_input(x);
    ggml_tensor * pd = x;
    if (!from_pdec) {
        ggml_tensor * g = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, x, 2 * D_ENC, N_CH, T, B), 0, 2, 1, 3));
        pd = mm(c, e.w.get("pre_decoder.0.linear.weight"), g);  // [512, T, 13, B]
        ggml_set_output(pd);
        ggml_build_forward_expand(gf, pd);
    }
    ggml_tensor * st = ggml_reshape_3d(c, pd, D_DEC, T, S);
    for (int l = 0; l < N_LAYER_D; l++) {
        ggml_tensor * k = mm(c, e.w.get(dec_name(l, "1.EncDecAttention.k.weight")), st);  // [384, T, S]
        ggml_tensor * v = mm(c, e.w.get(dec_name(l, "1.EncDecAttention.v.weight")), st);
        k = ggml_permute(c, ggml_reshape_4d(c, k, DH_D, N_HEAD_D, T, S), 0, 2, 1, 3);    // [64, T, 6, S]
        v = ggml_permute(c, ggml_reshape_4d(c, v, DH_D, N_HEAD_D, T, S), 1, 2, 0, 3);    // [T, 64, 6, S]
        ggml_build_forward_expand(gf, ggml_cpy(c, k, d.ck[l]));
        ggml_build_forward_expand(gf, ggml_cpy(c, v, d.cv[l]));
    }
    dec_compute(e, d, 0, "pre-decoder+crossKV", gf, { { x, in } });
    std::vector<float> out;
    if (!from_pdec) {
        out.resize((size_t) ggml_nelements(pd));
        ggml_backend_tensor_get(pd, out.data(), 0, ggml_nbytes(pd));
    }
    ggml_free(c);
    return out;
}

// One step for every stream at past length P: logits [S][VOCAB].
static void decoder_step(Encoder & e, Decoder & d, const int * tok, int P, std::vector<float> & logits) {
    const int        S = d.S, Lk = P + 1;
    const bool       composite = backend_is_cpu(e.bp.backend);
    ggml_init_params ip = { e.meta.size(), e.meta.data(), true };
    ggml_context *   c  = ggml_init(ip);
    ggml_cgraph *    gf = ggml_new_graph_custom(c, GRAPH_NODES, false);
    ggml_tensor *    ti = ggml_new_tensor_1d(c, GGML_TYPE_I32, S);
    ggml_tensor *    pi = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_input(ti);
    ggml_set_input(pi);
    const Weights & w = e.w;
    ggml_tensor *   x = ggml_add(c, ggml_get_rows(c, w.get("embed_tokens.weight"), ti),
                                 ggml_get_rows(c, w.get("decoder.decoder.additive_pe.weight"), pi));  // [512, S]
    for (int l = 0; l < N_LAYER_D; l++) {
        ggml_tensor * kc = d.kc[l], * vc = d.vc[l];
        ggml_tensor * h  = t5_norm(c, w, x, dec_name(l, "0.layer_norm.weight"));
        ggml_tensor * q  = mm(c, w.get(dec_name(l, "0.SelfAttention.q.weight")), h);
        ggml_tensor * k  = mm(c, w.get(dec_name(l, "0.SelfAttention.k.weight")), h);
        ggml_tensor * v  = mm(c, w.get(dec_name(l, "0.SelfAttention.v.weight")), h);
        ggml_tensor * ks = ggml_view_4d(c, kc, DH_D, 1, N_HEAD_D, S, kc->nb[1], kc->nb[2], kc->nb[3], (size_t) P * kc->nb[1]);
        ggml_tensor * vs = ggml_view_4d(c, vc, DH_D, 1, N_HEAD_D, S, vc->nb[1], vc->nb[2], vc->nb[3], (size_t) P * vc->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_4d(c, k, DH_D, 1, N_HEAD_D, S), ks));
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_4d(c, v, DH_D, 1, N_HEAD_D, S), vs));
        ggml_tensor * K  = ggml_view_4d(c, kc, DH_D, Lk, N_HEAD_D, S, kc->nb[1], kc->nb[2], kc->nb[3], 0);
        ggml_tensor * V  = ggml_view_4d(c, vc, DH_D, Lk, N_HEAD_D, S, vc->nb[1], vc->nb[2], vc->nb[3], 0);
        ggml_tensor * kq = ggml_soft_max(c, mm(c, K, ggml_reshape_4d(c, q, DH_D, 1, N_HEAD_D, S)));  // [Lk, 1, 6, S]
        ggml_tensor * o  = mm(c, ggml_cont(c, ggml_permute(c, V, 1, 0, 2, 3)), kq);              // [64, 1, 6, S]
        x = ggml_add(c, x, mm(c, w.get(dec_name(l, "0.SelfAttention.o.weight")), ggml_reshape_2d(c, o, DH_D * N_HEAD_D, S)));
        h  = t5_norm(c, w, x, dec_name(l, "1.layer_norm.weight"));
        q  = mm(c, w.get(dec_name(l, "1.EncDecAttention.q.weight")), h);
        kq = ggml_soft_max(c, mm(c, d.ck[l], ggml_reshape_4d(c, q, DH_D, 1, N_HEAD_D, S)));      // [T, 1, 6, S]
        o  = mm(c, d.cv[l], kq);
        x  = ggml_add(c, x, mm(c, w.get(dec_name(l, "1.EncDecAttention.o.weight")), ggml_reshape_2d(c, o, DH_D * N_HEAD_D, S)));
        h  = t5_norm(c, w, x, dec_name(l, "2.layer_norm.weight"));
        ggml_tensor * f = ggml_mul(c, gelu_tanh(c, mm(c, w.get(dec_name(l, "2.DenseReluDense.wi_0.weight")), h), composite),
                                   mm(c, w.get(dec_name(l, "2.DenseReluDense.wi_1.weight")), h));
        x = ggml_add(c, x, mm(c, w.get(dec_name(l, "2.DenseReluDense.wo.weight")), f));
    }
    x = ggml_scale(c, t5_norm(c, w, x, "decoder.decoder.final_layer_norm.weight"), 1.0f / sqrtf((float) D_DEC));
    ggml_tensor * lg = mm(c, w.get("lm_head.lm_head.weight"), x);  // [596, S]
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);
    const int32_t p32 = P;
    dec_compute(e, d, 1, "dec-step", gf, { { ti, tok }, { pi, &p32 } });
    logits.resize((size_t) S * VOCAB);
    ggml_backend_tensor_get(lg, logits.data(), 0, ggml_nbytes(lg));
    ggml_free(c);
}

struct DecodeOut {
    std::vector<std::vector<int>> tok;         // [S][steps]
    std::vector<float>            lg_prefill;  // [S][VOCAB], the first two steps' logits
    std::vector<float>            lg_step;
    int                           steps = 0;
    bool                          finite = true;
};

// Greedy decode of the prepared batch (task_cond_dec_generate, max_length 256).
static void decoder_greedy(Encoder & e, Decoder & d, DecodeOut & o) {
    const int S = d.S;
    o.tok.assign(S, {});
    o.steps  = 0;
    o.finite = true;
    std::vector<int>   cur(S, BOS), unfinished(S, 1);
    std::vector<float> lg;
    for (int P = 0; P < MAX_TOK; P++) {
        decoder_step(e, d, cur.data(), P, lg);
        if (P == 0) o.lg_prefill = lg;
        if (P == 1) o.lg_step = lg;
        bool any = P == 0;
        for (int s = 0; s < S; s++) {
            const float * r = &lg[(size_t) s * VOCAB];
            int           a = 0;
            for (int v = 0; v < VOCAB; v++) {
                if (!std::isfinite(r[v])) o.finite = false;
                if (r[v] > r[a]) a = v;  // torch.argmax: the first maximum
            }
            cur[s] = a;
            o.tok[s].push_back(unfinished[s] ? a : PAD);
            if (P > 0 && unfinished[s] && a == EOS) unfinished[s] = 0;
            any |= unfinished[s] != 0;
        }
        o.steps++;
        if (!any) break;
    }
}

}  // namespace yourmt3
