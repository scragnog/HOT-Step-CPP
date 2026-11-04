#pragma once
#include "codec-flow.h"

namespace heartcodec {

inline ggml_tensor * scalar_conv(ggml_context * c, const Weights & w, const std::string & p,
                                 ggml_tensor * x, int dilation = 1, bool causal = true) {
    int pad = int(w.get(p + ".weight")->ne[0] - 1) * dilation;
    if (!causal || !pad) return conv(c, w, p, x, pad / 2, dilation);
    // Causal = symmetric im2col padding with the last `pad` outputs dropped.
    // Metal has no left PAD (GH #329), so no ggml_pad_ext here.
    auto y = conv(c, w, p, x, pad, dilation);
    return ggml_cont(c, ggml_view_2d(c, y, x->ne[0], y->ne[1], y->nb[1], 0));
}

// relu(x) - slope * relu(-x), with the slope read in-graph (no host sync).
inline ggml_tensor * prelu(ggml_context * c, const Weights & w, const std::string & p, ggml_tensor * x) {
    auto weight = w.get(p + ".weight");
    require(ggml_nelements(weight) == 1 && weight->type == GGML_TYPE_F32, p + ": expected scalar F32 PReLU");
    return ggml_sub(c, ggml_relu(c, x), ggml_mul(c, ggml_relu(c, ggml_neg(c, x)), weight));
}

inline ggml_tensor * scalar_graph(Graph & g, const Weights & w, ggml_tensor * input) {
    auto c = g.ctx;
    const std::string p = "scalar_model.decoder.";
    auto x = scalar_conv(c, w, p + "0", input, 1, false);
    constexpr int strides[] = {5, 4, 4, 4, 3};
    constexpr int dilations[] = {1, 3, 5, 7, 9};
    for (int b = 0; b < 5; ++b) {
        std::string bp = p + std::to_string(b + 1);
        auto kernel = w.get(bp + ".up_conv.layer.weight");
        int stride = strides[b];
        require(kernel->ne[0] == 2 * stride && kernel->ne[2] == x->ne[1], bp + ": invalid transposed convolution");
        // Causal transposed convolution has no symmetric padding. Trim only
        // its final stride, exactly as the reference ConvTranspose1d wrapper.
        x = ggml_conv_transpose_1d(c, kernel, x, stride, 0, 1);
        x = ggml_cont(c, ggml_view_2d(c, x, x->ne[0] - stride, x->ne[1], x->nb[1], 0));
        if (auto bias = w.optional(bp + ".up_conv.layer.bias"))
            x = ggml_add(c, x, ggml_reshape_2d(c, bias, 1, bias->ne[0]));
        for (int r = 0; r < 5; ++r) {
            std::string rp = bp + ".convs." + std::to_string(r);
            auto skip = x;
            x = scalar_conv(c, w, rp + ".conv1", x, dilations[r]);
            x = prelu(c, w, rp + ".activation1", x);
            x = scalar_conv(c, w, rp + ".conv2", x);
            x = prelu(c, w, rp + ".activation2", x);
            x = ggml_add(c, skip, x);
        }
    }
    // PostProcessor repeats every sample twice before its causal convolution.
    int64_t time = x->ne[0], channels = x->ne[1];
    x = ggml_reshape_3d(c, x, 1, time, channels);
    x = ggml_repeat(c, x, ggml_new_tensor_3d(c, GGML_TYPE_F32, 2, time, channels));
    x = ggml_reshape_2d(c, x, 2 * time, channels);
    x = scalar_conv(c, w, p + "6.conv", x);
    x = prelu(c, w, p + "6.activation", x);
    return scalar_conv(c, w, p + "7", x);
}

// Time chunks bound the im2col workspace (3.1 GiB for a whole 744-frame chunk on
// CUDA, about 1 GiB at 186). Only decoder.0 looks ahead (3 frames, symmetric
// padding); everything after it is causal, and the receptive field is about 50
// latent frames (36 of them from block 1's dilated units at 5 samples a frame).
// With 64 frames of left context and 3 of right, each kept sample sees exactly
// the inputs the whole-sequence graph gives it.
constexpr int64_t scalar_chunk_frames = 186, scalar_left_context = 64, scalar_right_context = 3;

// Input is the flow's [time,256], split into two independent 128-wide channels.
// Output is planar stereo [2,time*1920] at 48 kHz.
inline std::vector<float> scalar_decode(const Weights & w, ggml_backend_t backend,
                                        const std::vector<float> & latent, int64_t frames,
                                        const Cancel & cancelled = {}, int64_t chunk = scalar_chunk_frames) {
    require(frames > 0 && latent.size() == size_t(frames * 256) && chunk > 0, "invalid scalar latent shape");
    std::vector<float> result;
    for (int ch = 0; ch < 2; ++ch) {
        for (int64_t s = 0; s < frames; s += chunk) {
            check_cancel(cancelled);
            const int64_t e = std::min(frames, s + chunk);
            const int64_t a = std::max<int64_t>(0, s - scalar_left_context);
            const int64_t b = std::min(frames, e + scalar_right_context), n = b - a;
            std::vector<float> planar(size_t(n * 128));
            for (int d = 0; d < 128; ++d) for (int64_t t = 0; t < n; ++t) {
                // torch.round uses ties-to-even, including negative half-integers.
                planar[size_t(d * n + t)] = std::nearbyint(9.0f * latent[size_t((a + t) * 256 + ch * 128 + d)]) / 9.0f;
            }
            Graph g(backend, 4096);  // 25 residual units of ~31 nodes plus up-convs
            auto input = g.input(GGML_TYPE_F32, {n, 128}, planar.data());
            auto piece = g.run(scalar_graph(g, w, input));
            require(piece.size() == size_t(n * 1920), "unexpected scalar waveform length");
            result.insert(result.end(), piece.begin() + (s - a) * 1920, piece.begin() + (e - a) * 1920);
        }
    }
    return result;
}

} // namespace heartcodec
