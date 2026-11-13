#pragma once
// chordnet/model.h: one ChordNet fold. The CNN and the LSTM input projections
// run as a GGML graph on the CPU backend (conv_2d_direct tiles its im2col, so
// a four-minute input does not need a full im2col buffer); the recurrence,
// the six heads and their softmax are plain loops.
//
// CNN: ten 3x3 convolutions, each followed by InstanceNorm2d (no affine,
// eps 1e-5, variance taken after centring) and SELU, with max pooling over
// frequency 3, 3, 4. The convolution biases are not added: the norm's
// centring removes any per-channel constant, so they cancel exactly. Adding
// them only costs precision: on a quiet input (conv1a std ~1e-4 beside biases
// up to 0.53) the float32 rounding of the sum, scaled by 1/sqrt(eps), moved
// the CNN output by ~1e-3. The last two convolutions are unpadded in frequency, so
// 252 bins end as 3; the output row is channel-major (c * 3 + f).

#include <cmath>
#include <cstring>
#include <vector>

#include "chordnet/features.h"
#include "chordnet/weights.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace chordnet {

constexpr int CNN_DIM = 240, HID = 96, GATES = 4 * HID, LSTM_DIM = 2 * HID, NHEAD = 6;
static const int HEAD_DIM[NHEAD] = { 73, 13, 4, 4, 3, 3 };

struct FoldOut {
    std::vector<float> cnn;          // [T][240]
    std::vector<float> lstm;         // [T][192]
    std::vector<float> head[NHEAD];  // [T][HEAD_DIM]
};

static void selu_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void *) {
    const float  L = 1.0507009873554804934193349852946f;
    const float  LA = (float) (1.6732632423543772848170429916717 * 1.0507009873554804934193349852946);
    const int64_t n = ggml_nelements(dst), per = (n + nth - 1) / nth;
    const int64_t i0 = ith * per, i1 = std::min(n, i0 + per);
    const float * x = (const float *) a->data;
    float *       y = (float *) dst->data;
    for (int64_t i = i0; i < i1; i++) y[i] = x[i] > 0.0f ? x[i] * L : std::expm1(x[i]) * LA;
}

// InstanceNorm2d over each channel's W x H plane: centre, then rms_norm.
static ggml_tensor * instance_norm(ggml_context * c, ggml_tensor * x) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
    ggml_tensor * x2 = ggml_reshape_2d(c, x, W * H, C);
    x2 = ggml_rms_norm(c, ggml_sub(c, x2, ggml_mean(c, x2)), 1e-5f);
    return ggml_reshape_3d(c, x2, W, H, C);
}

static ggml_tensor * conv_block(ggml_context * c, const Fold & f, int i, ggml_tensor * x) {
    const int pad_f = i >= 8 ? 0 : 1;
    ggml_tensor * y = ggml_conv_2d_direct(c, f.conv_w[i], x, 1, 1, pad_f, 1, 1, 1);
    return ggml_map_custom1_inplace(c, instance_norm(c, y), selu_op, GGML_N_TASKS_MAX, nullptr);
}

// x: [T][252] crop (from_cnn = false) or [T][240] CNN output (from_cnn = true).
// Fills out.cnn (unless from_cnn) and the two input projections
// proj[r] = W_ih x + b_ih + b_hh as [T][384].
static void run_graph(ggml_backend_t be, const Fold & f, const float * x, int T, bool from_cnn, FoldOut & out,
                      std::vector<float> proj[2]) {
    ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * in = from_cnn ? ggml_new_tensor_2d(c, GGML_TYPE_F32, CNN_DIM, T)
                                : ggml_new_tensor_3d(c, GGML_TYPE_F32, NFEAT, T, 1);
    ggml_set_input(in);
    ggml_tensor * feat = in;
    if (!from_cnn) {
        ggml_tensor * h = in;
        for (int i = 0; i < NCONV; i++) {
            h = conv_block(c, f, i, h);
            if (i == 2 || i == 5) h = ggml_pool_2d(c, h, GGML_OP_POOL_MAX, 3, 1, 3, 1, 0, 0);
            if (i == 7) h = ggml_pool_2d(c, h, GGML_OP_POOL_MAX, 4, 1, 4, 1, 0, 0);
        }
        // [3, T, 80] -> [3, 80, T] -> [240, T]
        feat = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, h, 0, 2, 1, 3)), CNN_DIM, T);
        ggml_set_output(feat);
        ggml_build_forward_expand(gf, feat);
    }
    ggml_tensor * p[2];
    for (int r = 0; r < 2; r++) {
        p[r] = ggml_add(c, ggml_add(c, ggml_mul_mat(c, f.w_ih[r], feat), f.b_ih[r]), f.b_hh[r]);
        ggml_set_output(p[r]);
        ggml_build_forward_expand(gf, p[r]);
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    ggml_gallocr_alloc_graph(ga, gf);
    memcpy(in->data, x, ggml_nbytes(in));
    ggml_backend_graph_compute(be, gf);
    if (!from_cnn) out.cnn.assign((const float *) feat->data, (const float *) feat->data + (size_t) T * CNN_DIM);
    for (int r = 0; r < 2; r++) proj[r].assign((const float *) p[r]->data, (const float *) p[r]->data + (size_t) T * GATES);
    ggml_gallocr_free(ga);
    ggml_free(c);
}

static double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Bidirectional LSTM, PyTorch gate order i, f, g, o, zero initial state.
static void run_lstm(const Fold & f, const std::vector<float> proj[2], int T, std::vector<float> & out) {
    out.assign((size_t) T * LSTM_DIM, 0.0f);
    for (int r = 0; r < 2; r++) {
        const float * W = f.w_hh[r];
        std::vector<double> h(HID, 0.0), cs(HID, 0.0), g(GATES);
        for (int s = 0; s < T; s++) {
            const int t = r ? T - 1 - s : s;
            const float * px = &proj[r][(size_t) t * GATES];
            for (int j = 0; j < GATES; j++) {
                double acc = px[j];
                const float * w = W + (size_t) j * HID;
                for (int k = 0; k < HID; k++) acc += (double) w[k] * h[k];
                g[j] = acc;
            }
            for (int k = 0; k < HID; k++) {
                const double ig = sigm(g[k]), fg = sigm(g[HID + k]), gg = std::tanh(g[2 * HID + k]), og = sigm(g[3 * HID + k]);
                cs[k] = fg * cs[k] + ig * gg;
                h[k]  = og * std::tanh(cs[k]);
                out[(size_t) t * LSTM_DIM + r * HID + k] = (float) h[k];
            }
        }
    }
}

// final_fc1 then one softmax per head: [T][192] -> six [T][n]
static void run_heads(const Fold & f, const std::vector<float> & lstm, int T, FoldOut & out) {
    for (int hd = 0; hd < NHEAD; hd++) out.head[hd].assign((size_t) T * HEAD_DIM[hd], 0.0f);
    double z[100];
    for (int t = 0; t < T; t++) {
        const float * x = &lstm[(size_t) t * LSTM_DIM];
        for (int o = 0; o < 100; o++) {
            double acc = f.fc_b[o];
            const float * w = f.fc_w + (size_t) o * LSTM_DIM;
            for (int k = 0; k < LSTM_DIM; k++) acc += (double) w[k] * x[k];
            z[o] = acc;
        }
        int off = 0;
        for (int hd = 0; hd < NHEAD; hd++) {
            const int n = HEAD_DIM[hd];
            double mx = z[off], sum = 0.0;
            for (int i = 1; i < n; i++) mx = std::max(mx, z[off + i]);
            for (int i = 0; i < n; i++) sum += std::exp(z[off + i] - mx);
            for (int i = 0; i < n; i++) out.head[hd][(size_t) t * n + i] = (float) (std::exp(z[off + i] - mx) / sum);
            off += n;
        }
    }
}

}  // namespace chordnet
