#pragma once
// yourmt3/weights.h: the YourMT3+ GGUF written by engine/tools/convert-yourmt3.py
// (torch names, f32). The tensor set is checked name for name and shape for
// shape against the fixed architecture, decoder side included, then uploaded
// to the compute backend. Load-time derived tensors:
//   *.mlp.{w1,gate,w2}_stack    the eight experts stacked for mul_mat_id
//   *.query_swap.{weight,bias}  query rows permuted so the output is rotate_half(q)
//   pre_encoder.*.bn*.{scale,shift}  eval BatchNorm folded as ATen does
// The STFT window and the rotary frequencies stay on the host.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

namespace yourmt3 {

constexpr int D_ENC = 128, N_LAT = 26, N_HEAD_E = 8, DH_E = 16, ROT_DIMS = 4, N_EXPERT = 8, TOPK = 2, D_FF_E = 512;
constexpr int N_CONV_CH = 128, N_BLOCK = 3, N_SUB = 2;
constexpr int D_DEC = 512, N_LAYER_D = 8, D_ATT_D = 384, D_FF_D = 1024, VOCAB = 596, N_POS_D = 1034, N_CH = 13;
constexpr long long N_VALUES = 47081606, N_PARAMS = 47078022;

struct Weights {
    ggml_context *                       ctx = nullptr;
    ggml_backend_buffer_t                buf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::vector<float>                   window;                  // [2048] torch.hann_window, periodic
    std::vector<float>                   freqs_lat, freqs_tmp;    // [2] each; the SCA freqs equal freqs_lat
    ggml_tensor * get(const std::string & n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            fprintf(stderr, "[yourmt3] missing tensor %s\n", n.c_str());
            exit(1);
        }
        return it->second;
    }
};

static void free_weights(Weights & w) {
    if (w.buf) ggml_backend_buffer_free(w.buf);
    if (w.ctx) ggml_free(w.ctx);
    w.buf = nullptr;
    w.ctx = nullptr;
    w.t.clear();
}

// Every tensor the GGUF must hold, with its torch shape.
static std::map<std::string, std::vector<int64_t>> expected_tensors() {
    std::map<std::string, std::vector<int64_t>> e;
    const std::vector<int64_t> v128 = { 128 }, m128 = { 128, 128 };
    e["spectrogram.stft.window"] = { 2048 };
    for (int b = 0; b < 3; b++) {
        const std::string p = "pre_encoder.0.blocks." + std::to_string(b) + ".";
        for (const char * bn : { "bn1.", "bn2." })
            for (const char * s : { "weight", "bias", "running_mean", "running_var" }) e[p + bn + s] = v128;
        e[p + "conv1.weight"] = { 128, b ? 128 : 1, 3, 3 };
        e[p + "conv2.weight"] = { 128, 128, 3, 3 };
        if (!b) {
            e[p + "shortcut.weight"] = { 128, 1, 1, 1 };
            e[p + "shortcut.bias"]   = v128;
        }
    }
    e["encoder.latent_array.latents"] = { N_LAT, D_ENC };
    for (const char * r : { "sca", "latent", "temporal" }) e[std::string("encoder.rotary_emb_") + r + ".freqs"] = { 2 };
    auto attn = [&](const std::string & p) {  // p ends in "self."
        for (const char * l : { "query", "key", "value" }) {
            e[p + l + ".weight"] = m128;
            e[p + l + ".bias"]   = v128;
        }
    };
    for (int b = 0; b < N_BLOCK; b++) {
        const std::string bp = "encoder.blocks." + std::to_string(b) + ".";
        const std::string sp = bp + "spectral_cross_attention.";
        for (const char * n : { "layernorm1.", "layernorm2." }) {
            e[sp + "self." + n + "weight"] = v128;
            e[sp + "self." + n + "bias"]   = v128;
        }
        attn(sp + "self.");
        e[sp + "output.dense.weight"] = m128;
        e[sp + "output.dense.bias"]   = v128;
        for (const char * kind : { "local_transformer.", "temporal_transformer." })
            for (int l = 0; l < N_SUB; l++) {
                const std::string lp = bp + kind + std::to_string(l) + ".";
                e[lp + "attention.self.layernorm1.weight"] = v128;
                e[lp + "attention.self.layernorm1.bias"]   = v128;
                attn(lp + "attention.self.");
                e[lp + "attention.output.dense.weight"] = m128;
                e[lp + "attention.output.dense.bias"]   = v128;
                e[lp + "layernorm.weight"]              = v128;
                e[lp + "layernorm.bias"]                = v128;
                e[lp + "mlp.gate.weight"]               = { N_EXPERT, D_ENC };
                for (int x = 0; x < N_EXPERT; x++) {
                    const std::string xp = lp + "mlp.experts." + std::to_string(x) + ".";
                    e[xp + "w1.weight"]   = { D_FF_E, D_ENC };
                    e[xp + "gate.weight"] = { D_FF_E, D_ENC };
                    e[xp + "w2.weight"]   = { D_ENC, D_FF_E };
                }
            }
    }
    e["pre_decoder.0.linear.weight"]        = { D_DEC, 2 * D_ENC };
    e["embed_tokens.weight"]                = { VOCAB, D_DEC };
    e["lm_head.lm_head.weight"]             = { VOCAB, D_DEC };
    e["decoder.decoder.additive_pe.weight"] = { N_POS_D, D_DEC };
    e["decoder.decoder.final_layer_norm.weight"] = { D_DEC };
    for (int l = 0; l < N_LAYER_D; l++) {
        const std::string p = "decoder.decoder.block." + std::to_string(l) + ".layer.";
        for (const char * a : { "0.SelfAttention.", "1.EncDecAttention." }) {
            for (const char * q : { "q", "k", "v" }) e[p + a + q + ".weight"] = { D_ATT_D, D_DEC };
            e[p + a + "o.weight"] = { D_DEC, D_ATT_D };
        }
        for (const char * n : { "0", "1", "2" }) e[p + n + ".layer_norm.weight"] = { D_DEC };
        e[p + "2.DenseReluDense.wi_0.weight"] = { D_FF_D, D_DEC };
        e[p + "2.DenseReluDense.wi_1.weight"] = { D_FF_D, D_DEC };
        e[p + "2.DenseReluDense.wo.weight"]   = { D_DEC, D_FF_D };
    }
    return e;
}

// Rows of a query projection reordered so that its output is rotate_half(q):
// out[2i] = -q[2i+1], out[2i+1] = q[2i]. Exact (a permutation and a sign).
static void swap_rows(const float * w, float * o, int rows, int cols) {
    for (int r = 0; r < rows; r += 2)
        for (int c = 0; c < cols; ++c) {
            o[(size_t) r * cols + c]       = -w[(size_t) (r + 1) * cols + c];
            o[(size_t) (r + 1) * cols + c] = w[(size_t) r * cols + c];
        }
}

static bool load_weights(Weights & w, const std::string & path, ggml_backend_t be) {
    ggml_context *   host = nullptr;
    gguf_init_params gp   = { /*no_alloc*/ false, &host };
    gguf_context *   g    = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        fprintf(stderr, "[yourmt3] cannot read %s\n", path.c_str());
        return false;
    }
    // the tensor set, exactly
    const auto exp = expected_tensors();
    bool       ok  = true;
    long long  n_values = 0;
    const int  n = (int) gguf_get_n_tensors(g);
    for (int i = 0; i < n; i++) {
        const char *  name = gguf_get_tensor_name(g, i);
        ggml_tensor * t    = ggml_get_tensor(host, name);
        auto          it   = exp.find(name);
        bool          good = it != exp.end() && t->type == GGML_TYPE_F32 && ggml_n_dims(t) <= (int) it->second.size();
        if (good)
            for (size_t d = 0; d < 4; d++) {  // torch shape is reversed ne
                const size_t k = it->second.size();
                good &= t->ne[d] == (d < k ? it->second[k - 1 - d] : 1);
            }
        if (!good) {
            fprintf(stderr, "[yourmt3] %s: unexpected name, type or shape\n", name);
            ok = false;
        }
        n_values += ggml_nelements(t);
    }
    if ((size_t) n != exp.size() || n_values != N_VALUES) {
        fprintf(stderr, "[yourmt3] %s: %d tensors, %lld values; expected %zu and %lld\n", path.c_str(), n, n_values,
                exp.size(), N_VALUES);
        ok = false;
    }
    const int64_t kp = gguf_find_key(g, "yourmt3.parameters");
    if (kp < 0 || gguf_get_kv_type(g, kp) != GGUF_TYPE_UINT32 || gguf_get_val_u32(g, kp) != N_PARAMS) {
        fprintf(stderr, "[yourmt3] %s: parameter count missing or not %lld\n", path.c_str(), N_PARAMS);
        ok = false;
    }
    if (!ok) {
        gguf_free(g);
        ggml_free(host);
        return false;
    }
    auto hdata = [&](const std::string & name) { return (const float *) ggml_get_tensor(host, name.c_str())->data; };
    auto hvec  = [&](const std::string & name, size_t k) { const float * p = hdata(name); return std::vector<float>(p, p + k); };
    w.window    = hvec("spectrogram.stft.window", 2048);
    w.freqs_lat = hvec("encoder.rotary_emb_latent.freqs", 2);
    w.freqs_tmp = hvec("encoder.rotary_emb_temporal.freqs", 2);
    if (hvec("encoder.rotary_emb_sca.freqs", 2) != w.freqs_lat) {
        fprintf(stderr, "[yourmt3] SCA and latent rotary frequencies differ\n");
        gguf_free(g);
        ggml_free(host);
        return false;
    }

    // device tensors: everything except the individual experts, plus the derived ones
    const size_t n_extra = 2 * N_BLOCK * N_SUB * 3 + 2 * (N_BLOCK + 2 * N_BLOCK * N_SUB) + 4 * 3;
    ggml_init_params ip = { (n + n_extra + 16) * ggml_tensor_overhead(), nullptr, true };
    w.ctx = ggml_init(ip);
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> pending;
    auto add = [&](const std::string & name, std::vector<int64_t> ne, std::vector<float> data) {
        ggml_tensor * dt = ggml_new_tensor(w.ctx, GGML_TYPE_F32, (int) ne.size(), ne.data());
        ggml_set_name(dt, name.c_str());
        w.t[name] = dt;
        pending.push_back({ dt, std::move(data) });
    };
    for (int i = 0; i < n; i++) {
        const std::string s = gguf_get_tensor_name(g, i);
        if (s.find(".experts.") != std::string::npos || (s.rfind("pre_encoder.", 0) == 0 && s.find(".bn") != std::string::npos))
            continue;
        ggml_tensor * ht = ggml_get_tensor(host, s.c_str());
        add(s, { ht->ne[0], ht->ne[1], ht->ne[2], ht->ne[3] }, hvec(s, (size_t) ggml_nelements(ht)));
    }
    for (int b = 0; b < 3; b++)
        for (const char * bn : { "bn1", "bn2" }) {
            const std::string p = "pre_encoder.0.blocks." + std::to_string(b) + "." + bn + ".";
            const float * wt = hdata(p + "weight"), * bs = hdata(p + "bias");
            const float * mu = hdata(p + "running_mean"), * var = hdata(p + "running_var");
            std::vector<float> sc(N_CONV_CH), sh(N_CONV_CH);
            for (int c = 0; c < N_CONV_CH; c++) {
                const float invstd = 1.0f / std::sqrt(var[c] + 1e-5f);
                sc[c] = invstd * wt[c];
                sh[c] = bs[c] - mu[c] * sc[c];
            }
            add(p + "scale", { 1, 1, N_CONV_CH, 1 }, sc);
            add(p + "shift", { 1, 1, N_CONV_CH, 1 }, sh);
        }
    auto stack = [&](const std::string & pre, const char * nm, int in, int out) {
        std::vector<float> all((size_t) in * out * N_EXPERT);
        for (int x = 0; x < N_EXPERT; x++)
            memcpy(all.data() + (size_t) x * in * out, hdata(pre + "mlp.experts." + std::to_string(x) + "." + nm + ".weight"),
                   (size_t) in * out * sizeof(float));
        add(pre + "mlp." + nm + "_stack", { in, out, N_EXPERT }, std::move(all));
    };
    auto swapped_q = [&](const std::string & pre) {
        std::vector<float> ws((size_t) D_ENC * D_ENC), bs(D_ENC);
        swap_rows(hdata(pre + "query.weight"), ws.data(), D_ENC, D_ENC);
        swap_rows(hdata(pre + "query.bias"), bs.data(), D_ENC, 1);
        add(pre + "query_swap.weight", { D_ENC, D_ENC }, std::move(ws));
        add(pre + "query_swap.bias", { D_ENC }, std::move(bs));
    };
    for (int b = 0; b < N_BLOCK; b++) {
        const std::string bp = "encoder.blocks." + std::to_string(b) + ".";
        swapped_q(bp + "spectral_cross_attention.self.");
        for (const char * kind : { "local_transformer", "temporal_transformer" })
            for (int l = 0; l < N_SUB; l++) {
                const std::string lp = bp + kind + "." + std::to_string(l) + ".";
                stack(lp, "w1", D_ENC, D_FF_E);
                stack(lp, "gate", D_ENC, D_FF_E);
                stack(lp, "w2", D_FF_E, D_ENC);
                swapped_q(lp + "attention.self.");
            }
    }
    gguf_free(g);
    ggml_free(host);
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, be);
    if (!w.buf) {
        fprintf(stderr, "[yourmt3] weight buffer allocation failed\n");
        free_weights(w);
        return false;
    }
    for (auto & p : pending) ggml_backend_tensor_set(p.first, p.second.data(), 0, ggml_nbytes(p.first));
    return true;
}

}  // namespace yourmt3
