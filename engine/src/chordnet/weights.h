#pragma once
// chordnet/weights.h: the five ChordNet folds from one f32 GGUF written by
// engine/tools/convert-chordnet.py (tensor names fold<i>.<torch name>).
// Loaded into host memory: ChordNet runs on the CPU backend only.

#include <cstdio>
#include <string>

#include "ggml.h"
#include "gguf.h"

namespace chordnet {

constexpr int NFOLD = 5;
constexpr int NCONV = 10;
static const char * const CONV_NAMES[NCONV] = { "conv1a", "conv1b", "conv1c", "conv2a", "conv2b",
                                                "conv2c", "conv3a", "conv3b", "conv4a", "conv4b" };

struct Fold {
    ggml_tensor * conv_w[NCONV];  // [3, 3, IC, OC]
    ggml_tensor * conv_b[NCONV];  // [OC], checked but unused: InstanceNorm cancels it
    ggml_tensor * w_ih[2];        // [240, 384], forward and reverse
    ggml_tensor * b_ih[2];        // [384]
    ggml_tensor * b_hh[2];        // [384]
    const float * w_hh[2];        // [384][96] row-major
    const float * fc_w;           // [100][192]
    const float * fc_b;           // [100]
};

struct Weights {
    ggml_context * ctx  = nullptr;
    gguf_context * gguf = nullptr;
    Fold           fold[NFOLD];
};

static void free_weights(Weights & w) {
    if (w.gguf) gguf_free(w.gguf);
    if (w.ctx) ggml_free(w.ctx);
    w.gguf = nullptr;
    w.ctx  = nullptr;
}

static bool load_weights(Weights & w, const std::string & path) {
    gguf_init_params gp = { /*no_alloc*/ false, &w.ctx };
    w.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!w.gguf) {
        fprintf(stderr, "[chordnet] cannot read %s\n", path.c_str());
        return false;
    }
    bool ok = true;
    auto get = [&](int f, const std::string & name, int64_t n0, int64_t n1, int64_t n2, int64_t n3) -> ggml_tensor * {
        const std::string full = "fold" + std::to_string(f) + "." + name;
        ggml_tensor * t = ggml_get_tensor(w.ctx, full.c_str());
        if (!t || t->type != GGML_TYPE_F32 || t->ne[0] != n0 || t->ne[1] != n1 || t->ne[2] != n2 || t->ne[3] != n3) {
            fprintf(stderr, "[chordnet] %s: missing or wrong shape/type\n", full.c_str());
            ok = false;
        }
        return t;
    };
    static const int IC[NCONV] = { 1, 16, 16, 16, 32, 32, 32, 64, 64, 80 };
    static const int OC[NCONV] = { 16, 16, 16, 32, 32, 32, 64, 64, 80, 80 };
    for (int f = 0; f < NFOLD && ok; f++) {
        Fold & d = w.fold[f];
        for (int i = 0; i < NCONV; i++) {
            const std::string p = std::string("audio_feature_block.") + CONV_NAMES[i];
            d.conv_w[i] = get(f, p + ".weight", 3, 3, IC[i], OC[i]);
            d.conv_b[i] = get(f, p + ".bias", OC[i], 1, 1, 1);
        }
        for (int r = 0; r < 2; r++) {
            const std::string s = r ? "_reverse" : "";
            d.w_ih[r] = get(f, "lstm1.weight_ih_l0" + s, 240, 384, 1, 1);
            d.b_ih[r] = get(f, "lstm1.bias_ih_l0" + s, 384, 1, 1, 1);
            d.b_hh[r] = get(f, "lstm1.bias_hh_l0" + s, 384, 1, 1, 1);
            ggml_tensor * hh = get(f, "lstm1.weight_hh_l0" + s, 96, 384, 1, 1);
            d.w_hh[r] = hh ? (const float *) hh->data : nullptr;
        }
        ggml_tensor * fw = get(f, "final_fc1.weight", 192, 100, 1, 1);
        ggml_tensor * fb = get(f, "final_fc1.bias", 100, 1, 1, 1);
        d.fc_w = fw ? (const float *) fw->data : nullptr;
        d.fc_b = fb ? (const float *) fb->data : nullptr;
    }
    if (!ok) free_weights(w);
    return ok;
}

}  // namespace chordnet
