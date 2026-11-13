#pragma once
// HeartCodec encoder frontends: resampling, mel spectrograms and the MuEncoder,
// WavLM and Whisper graphs of heartlib's HeartCodecFeatureExtractors.
//
// Activations are [features, time] (ne0 = features) except inside convolutions,
// which run time-major [time, channels] as codec-flow.h does. The DSP runs on the
// host in double where torch runs float32, so it sits at or below torch's own
// rounding. On CPU every weight runs as F32 (the S3 parity baseline); on a GPU
// only 2D matrices keep their stored F16.
#include "heartmula/codec-flow.h"

#include <cstring>
#include <complex>
#include <numeric>

namespace heartcodec {

constexpr double enc_pi = 3.14159265358979323846;
constexpr int64_t enc_chunk_samples = 714240;           // 29.76 s at 24 kHz
constexpr int64_t enc_tail = 240;                        // alignment tail per chunk
constexpr int64_t enc_frames = enc_chunk_samples / 960;  // 744 at 25 Hz
constexpr int64_t enc_samples_16k = (enc_chunk_samples + enc_tail) * 2 / 3;  // 476320

// ── Weights ─────────────────────────────────────────────────────────────────

// Loads one encoder-pack GGUF into `w`. Weight-norm pairs fold into `<p>.weight`
// and BatchNorm statistics into `<bn>.scale` / `<bn>.shift` (eps 1e-5), both F32.
// A [1, in, out] pointwise kernel loads as the matrix [in, out].
inline void load_encoder_part(Weights & w, const std::string & path, const std::string & role, ggml_backend_t backend) {
    require(!w.storage.ctx, "weights already loaded");
    GGUFModel file{};
    require(gf_load(&file, path.c_str()), "cannot load " + path);
    struct Close { GGUFModel * f; ~Close() { gf_close(f); } } close{&file};
    auto str = [&](const char * key) -> std::string {
        int64_t i = gguf_find_key(file.gguf, key);
        require(i >= 0 && gguf_get_kv_type(file.gguf, i) == GGUF_TYPE_STRING, path + ": missing " + key);
        return gguf_get_val_str(file.gguf, i);
    };
    require(str("general.architecture") == "heartcodec-" + role, path + ": architecture mismatch");
    require(str("heartmula.component") == role, path + ": component mismatch");
    require(str("heartmula.variant") == "shared", path + ": encoder variant must be shared");
    int64_t version = gguf_find_key(file.gguf, "heartmula.format_version");
    require(version >= 0 && gguf_get_kv_type(file.gguf, version) == GGUF_TYPE_UINT32 &&
            gguf_get_val_u32(file.gguf, version) == 1, path + ": unsupported format version");
    w.fingerprint = str("heartmula.base_fingerprint");
    w.dependencies = str("heartmula.dependencies");
    w.config = str("heartmula.config_json");
    require(yue2_aitk::sha256::bytes(w.config.data(), w.config.size()).hex() == str("heartmula.config_sha256"),
            path + ": config hash mismatch");
    using Json = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
    Json dep(yyjson_read(w.dependencies.data(), w.dependencies.size(), 0), yyjson_doc_free);
    require(bool(dep), path + ": invalid dependencies JSON");
    auto hash = yyjson_obj_get(yyjson_doc_get_root(dep.get()), "rvq_sha256");
    require(yyjson_is_str(hash) && yyjson_get_len(hash) == 64, path + ": missing RVQ hash");
    w.rvq_hash = yyjson_get_str(hash);

    auto device = ggml_backend_get_device(backend);
    const bool gpu = device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU;
    const int64_t count = gguf_get_n_tensors(file.gguf);
    wctx_init(&w.storage, int(count));
    auto meta = [&](const std::string & name) {
        auto t = ggml_get_tensor(file.meta, name.c_str());
        require(t != nullptr, path + ": missing tensor " + name);
        return t;
    };
    auto host = [&](const std::string & name) {
        auto t = meta(name);
        const uint8_t * raw = file.mapping + file.data_offset + gguf_get_tensor_offset(file.gguf, gguf_find_tensor(file.gguf, name.c_str()));
        std::vector<float> v(size_t(ggml_nelements(t)));
        if (t->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(raw), v.data(), ggml_nelements(t));
        else { require(t->type == GGML_TYPE_F32, path + ": unsupported precision for " + name); std::memcpy(v.data(), raw, v.size() * 4); }
        return v;
    };
    auto add = [&](const std::string & name, int dims, const int64_t * ne, std::vector<float> data) {
        auto dst = ggml_new_tensor(w.storage.ctx, GGML_TYPE_F32, dims, ne);
        ggml_set_name(dst, name.c_str());
        require(size_t(ggml_nelements(dst)) == data.size(), name + ": folded size mismatch");
        auto buf = std::make_unique<float[]>(data.size());
        std::copy(data.begin(), data.end(), buf.get());
        w.storage.pending.push_back({dst, buf.get(), ggml_nbytes(dst), 0});
        w.storage.staging.push_back(std::move(buf));
        w.tensors[name] = dst;
    };
    auto ends = [](const std::string & s, const std::string & tail) {
        return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
    };
    const std::string g_tail = ".parametrizations.weight.original0", v_tail = ".parametrizations.weight.original1";
    for (int64_t i = 0; i < count; ++i) {
        const std::string name = gguf_get_tensor_name(file.gguf, i);
        ggml_tensor * src = meta(name);
        if (src->type == GGML_TYPE_I64 || ends(name, g_tail) || ends(name, ".running_var") ||
            ends(name, ".num_batches_tracked")) continue;  // consumed with their partner or unused
        if (ends(name, v_tail)) {
            // w = g * v / ||v||, the norm taken over every axis but g's.
            const std::string prefix = name.substr(0, name.size() - v_tail.size());
            auto g = meta(prefix + g_tail);
            int axis = -1;
            for (int a = 0; a < 4; ++a) if (g->ne[a] != 1) { require(axis < 0 && g->ne[a] == src->ne[a], name + ": weight-norm axis"); axis = a; }
            require(axis >= 0, name + ": scalar weight-norm gain");
            auto v = host(name), gv = host(prefix + g_tail);
            std::vector<double> norm(size_t(src->ne[axis]), 0.0);
            int64_t stride = 1;
            for (int a = 0; a < axis; ++a) stride *= src->ne[a];
            auto index = [&](size_t e) { return size_t((int64_t(e) / stride) % src->ne[axis]); };
            for (size_t e = 0; e < v.size(); ++e) norm[index(e)] += double(v[e]) * v[e];
            for (size_t e = 0; e < v.size(); ++e) v[e] = float(double(gv[index(e)]) * v[e] / std::sqrt(norm[index(e)]));
            add(prefix + ".weight", ggml_n_dims(src), src->ne, std::move(v));
        } else if (ends(name, ".running_mean")) {
            const std::string bn = name.substr(0, name.size() - 13);
            auto mean = host(name), var = host(bn + ".running_var"), gain = host(bn + ".weight"), bias = host(bn + ".bias");
            std::vector<float> scale(mean.size()), shift(mean.size());
            for (size_t c = 0; c < mean.size(); ++c) {
                double s = double(gain[c]) / std::sqrt(double(var[c]) + 1e-5);
                scale[c] = float(s); shift[c] = float(double(bias[c]) - double(mean[c]) * s);
            }
            add(bn + ".scale", 1, src->ne, std::move(scale));
            add(bn + ".shift", 1, src->ne, std::move(shift));
        } else {
            int64_t ne[4] = {src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
            int dims = ggml_n_dims(src);
            if (name.find("pointwise_conv") != std::string::npos) {
                require(dims == 3 && ne[0] == 1, name + ": expected a [1, in, out] kernel");
                ne[0] = ne[1]; ne[1] = ne[2]; ne[2] = 1; dims = 2;
            }
            // BatchNorm gains and biases fold above; skip them when their stats exist.
            if ((ends(name, ".weight") || ends(name, ".bias"))) {
                const std::string base = name.substr(0, name.rfind('.'));
                if (gguf_find_tensor(file.gguf, (base + ".running_mean").c_str()) >= 0) continue;
            }
            const bool matrix = dims == 2 && ends(name, ".weight");
            if (gpu && matrix && src->type == GGML_TYPE_F16) w.tensors[name] = gf_load_tensor(&w.storage, file, name, ne, dims);
            else add(name, dims, ne, host(name));
        }
    }
    require(wctx_alloc(&w.storage, backend), path + ": allocation failed");
}

// ── DSP ─────────────────────────────────────────────────────────────────────

// torchaudio's sinc_interp_hann resampler (width 6, rolloff 0.99). `cached` builds
// the kernel as transforms.Resample does (float64, cast to float32); otherwise as
// functional.resample on float32 audio, in float32. Planar [channels][length].
inline std::vector<float> resample(const std::vector<float> & x, int channels, int64_t length,
                                   int64_t from, int64_t to, bool cached, int64_t * out_length) {
    require(from > 0 && to > 0 && channels > 0 && x.size() == size_t(channels * length), "invalid resample input");
    const int64_t gcd = std::gcd(from, to), o = from / gcd, n = to / gcd;
    const double base = double(std::min(o, n)) * 0.99;
    const int64_t width = int64_t(std::ceil(6.0 * double(o) / base)), taps = 2 * width + o;
    std::vector<float> kernel(size_t(n * taps));
    for (int64_t j = 0; j < n; ++j) for (int64_t k = 0; k < taps; ++k) {
        float value;
        if (cached) {
            double t = double(float(-double(j)) / float(n)) + double(k - width) / double(o);
            t = std::clamp(t * base, -6.0, 6.0);
            double window = std::cos(t * enc_pi / 6.0 / 2.0);
            window *= window;
            t *= enc_pi;
            double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
            value = float(sinc * (window * (base / double(o))));
        } else {
            float t = float(-j) / float(n) + float(k - width) / float(o);
            t = std::clamp(t * float(base), -6.0f, 6.0f);
            float window = std::cos(t * float(enc_pi) / 6.0f / 2.0f);
            window = window * window;
            t *= float(enc_pi);
            float sinc = t == 0.0f ? 1.0f : std::sin(t) / t;
            value = sinc * (window * float(base / double(o)));
        }
        kernel[size_t(j * taps + k)] = value;
    }
    const int64_t target = (n * length + o - 1) / o, blocks = length / o + 1;
    std::vector<float> out(size_t(channels * target));
    for (int c = 0; c < channels; ++c) {
        const float * src = x.data() + c * length;
        float * dst = out.data() + c * target;
        for (int64_t b = 0; b < blocks; ++b) for (int64_t j = 0; j < n; ++j) {
            const int64_t i = b * n + j;
            if (i >= target) break;
            double acc = 0;
            const float * kj = kernel.data() + j * taps;
            for (int64_t k = 0; k < taps; ++k) {
                int64_t s = b * o + k - width;
                if (s >= 0 && s < length) acc += double(src[s]) * kj[k];
            }
            dst[i] = float(acc);
        }
    }
    *out_length = target;
    return out;
}

// Centered (reflect-padded) windowed one-sided power STFT: [frames][n_fft/2+1],
// frames = n / hop + 1. Radix-2 FFT for powers of two, a direct DFT otherwise.
inline std::vector<double> power_stft(const float * x, int64_t n, int n_fft, int hop,
                                      const std::vector<double> & window, int64_t * frames_out) {
    const int pad = n_fft / 2, bins = n_fft / 2 + 1;
    require(n > pad, "audio shorter than half an FFT window");
    const int64_t frames = n / hop + 1;
    std::vector<double> padded(size_t(n + 2 * pad));
    for (int64_t i = 0; i < n + 2 * pad; ++i) {
        int64_t s = i - pad;
        if (s < 0) s = -s;
        if (s >= n) s = 2 * (n - 1) - s;
        padded[size_t(i)] = x[s];
    }
    const bool pow2 = (n_fft & (n_fft - 1)) == 0;
    std::vector<std::complex<double>> twiddle(static_cast<size_t>(n_fft));
    for (int k = 0; k < n_fft; ++k) twiddle[size_t(k)] = std::polar(1.0, -2.0 * enc_pi * k / n_fft);
    std::vector<int> reversed(size_t(n_fft), 0);
    if (pow2) for (int i = 0, bits = 0; i < n_fft; ++i) {
        if (i == 0) { while ((1 << bits) < n_fft) ++bits; }
        int r = 0;
        for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        reversed[size_t(i)] = r;
    }
    std::vector<double> power(size_t(frames * bins));
    std::vector<std::complex<double>> buf(static_cast<size_t>(n_fft));
    for (int64_t f = 0; f < frames; ++f) {
        const double * frame = padded.data() + f * hop;
        if (pow2) {
            for (int i = 0; i < n_fft; ++i) buf[size_t(reversed[size_t(i)])] = frame[i] * window[size_t(i)];
            for (int len = 2; len <= n_fft; len <<= 1) {
                const int step = n_fft / len;
                for (int s = 0; s < n_fft; s += len) for (int k = 0; k < len / 2; ++k) {
                    auto u = buf[size_t(s + k)], v = buf[size_t(s + k + len / 2)] * twiddle[size_t(k * step)];
                    buf[size_t(s + k)] = u + v; buf[size_t(s + k + len / 2)] = u - v;
                }
            }
            for (int k = 0; k < bins; ++k) power[size_t(f * bins + k)] = std::norm(buf[size_t(k)]);
        } else {
            for (int k = 0; k < bins; ++k) {
                std::complex<double> acc = 0;
                for (int i = 0; i < n_fft; ++i) acc += frame[i] * window[size_t(i)] * twiddle[size_t((int64_t(k) * i) % n_fft)];
                power[size_t(f * bins + k)] = std::norm(acc);
            }
        }
    }
    *frames_out = frames;
    return power;
}

// MusicFM mel for MuEncoder: power mel (2048/240, 128 bins, the GGUF's own Hann
// window and filterbank), 10 log10, last frame dropped, (x - mean) / std.
// Returns time-major [128][frames] (ne0 = time) for the 2D convolutions.
// The MuEncoder mel's Hann window and filterbank, read off the device once.
struct MuMelBuffers {
    std::vector<double> window;
    std::vector<float>  fb;
};

inline MuMelBuffers muencoder_mel_buffers(const Weights & w) {
    const std::string p = "encoder.feature_extractors.muencoder.preprocessor_melspec_2048.mel_stft.";
    auto fetch = [&](const std::string & name) {
        auto t = w.get(name);
        require(t->type == GGML_TYPE_F32, name + ": expected F32");
        std::vector<float> v(size_t(ggml_nelements(t)));
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
        return v;
    };
    auto window_f = fetch(p + "spectrogram.window"), fb = fetch(p + "mel_scale.fb");
    require(window_f.size() == 2048 && fb.size() == size_t(1025 * 128), "MuEncoder mel buffers have unexpected shapes");
    return { std::vector<double>(window_f.begin(), window_f.end()), std::move(fb) };
}

// Host-only, so chunks can run on worker threads (codec-encode.h).
inline std::vector<float> muencoder_mel(const MuMelBuffers & b, const float * x, int64_t n, double mean, double std_dev,
                                        int64_t * frames_out) {
    const std::vector<double> & window = b.window;
    const std::vector<float> &  fb     = b.fb;
    int64_t frames = 0;
    auto power = power_stft(x, n, 2048, 240, window, &frames);
    const int64_t kept = frames - 1;
    std::vector<float> mel(size_t(128 * kept));
    for (int64_t t = 0; t < kept; ++t) for (int m = 0; m < 128; ++m) {
        double acc = 0;
        for (int f = 0; f < 1025; ++f) acc += power[size_t(t * 1025 + f)] * fb[size_t(f * 128 + m)];
        float db = 10.0f * std::log10(std::max(float(acc), 1e-10f));
        mel[size_t(m * kept + t)] = float((double(db) - mean) / std_dev);
    }
    *frames_out = kept;
    return mel;
}

inline std::vector<float> muencoder_mel(const Weights & w, const float * x, int64_t n, double mean, double std_dev,
                                        int64_t * frames_out) {
    return muencoder_mel(muencoder_mel_buffers(w), x, n, mean, std_dev, frames_out);
}

// transformers' mel_filter_bank(201, 80, 0, 8000, 16000, slaney, slaney): [201][80].
inline std::vector<double> whisper_filters() {
    auto to_mel = [](double f) { return f >= 1000.0 ? 15.0 + std::log(f / 1000.0) * (27.0 / std::log(6.4)) : 3.0 * f / 200.0; };
    auto to_hz = [](double m) { return m >= 15.0 ? 1000.0 * std::exp(std::log(6.4) / 27.0 * (m - 15.0)) : 200.0 * m / 3.0; };
    const int bins = 201, mels = 80;
    std::vector<double> edge(mels + 2), fft(bins), filters(size_t(bins * mels));
    const double lo = to_mel(0.0), hi = to_mel(8000.0);
    for (int i = 0; i < mels + 2; ++i) edge[size_t(i)] = to_hz(lo + (hi - lo) * i / (mels + 1));
    for (int i = 0; i < bins; ++i) fft[size_t(i)] = 8000.0 * i / (bins - 1);
    for (int f = 0; f < bins; ++f) for (int m = 0; m < mels; ++m) {
        double down = -(edge[size_t(m)] - fft[size_t(f)]) / (edge[size_t(m + 1)] - edge[size_t(m)]);
        double up = (edge[size_t(m + 2)] - fft[size_t(f)]) / (edge[size_t(m + 2)] - edge[size_t(m + 1)]);
        double enorm = 2.0 / (edge[size_t(m + 2)] - edge[size_t(m)]);
        filters[size_t(f * mels + m)] = double(float(std::max(0.0, std::min(down, up)) * enorm));
    }
    return filters;
}

// WhisperFeatureExtractor (torch path, do_normalize off): zero-pad to 30 s, 400/160
// Hann STFT, last frame dropped, slaney mel, log10, floor at max - 8, (x + 4) / 4.
// Returns time-major [80][3000].
inline std::vector<float> whisper_mel(const float * x, int64_t n) {
    require(n <= 480000, "Whisper input longer than 30 s");
    std::vector<float> padded(480000, 0.0f);
    std::copy_n(x, n, padded.data());
    std::vector<double> window(400);
    for (int i = 0; i < 400; ++i) window[size_t(i)] = double(float(0.5 - 0.5 * std::cos(2.0 * enc_pi * i / 400.0)));
    int64_t frames = 0;
    auto power = power_stft(padded.data(), 480000, 400, 160, window, &frames);
    static const std::vector<double> filters = whisper_filters();
    const int64_t kept = frames - 1;
    std::vector<float> mel(size_t(80 * kept));
    float peak = -INFINITY;
    for (int64_t t = 0; t < kept; ++t) for (int m = 0; m < 80; ++m) {
        double acc = 0;
        for (int f = 0; f < 201; ++f) acc += double(float(power[size_t(t * 201 + f)])) * filters[size_t(f * 80 + m)];
        float v = std::log10(std::max(float(acc), 1e-10f));
        mel[size_t(m * kept + t)] = v;
        peak = std::max(peak, v);
    }
    for (auto & v : mel) v = (std::max(v, peak - 8.0f) + 4.0f) / 4.0f;
    return mel;
}

// ── Graph helpers ───────────────────────────────────────────────────────────

inline ggml_tensor * layer_norm(ggml_context * c, const Weights & w, const std::string & p, ggml_tensor * x) {
    x = ggml_norm(c, x, 1e-5f);
    return ggml_add(c, ggml_mul(c, x, w.get(p + ".weight")), w.get(p + ".bias"));
}

// Time-major [time, in] convolution with stride; kernel [K, in, out].
inline ggml_tensor * conv_strided(ggml_context * c, ggml_tensor * kernel, ggml_tensor * x, int stride, int pad) {
    require(kernel->ne[1] == x->ne[1], "convolution width mismatch");
    auto col = ggml_im2col(c, kernel, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);
    return ggml_mul_mat(c, ggml_reshape_2d(c, col, col->ne[0], col->ne[1]),
                        ggml_reshape_2d(c, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
}

// Depthwise time-major [time, ch] convolution, kernel [K, 1, ch], F32 im2col.
inline ggml_tensor * conv_depthwise(ggml_context * c, ggml_tensor * kernel, ggml_tensor * x, int pad) {
    const int64_t time = x->ne[0], ch = x->ne[1], k = kernel->ne[0];
    require(kernel->ne[1] == 1 && kernel->ne[2] == ch, "depthwise kernel shape");
    auto col = ggml_im2col(c, kernel, ggml_reshape_4d(c, x, time, 1, ch, 1), 1, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);
    auto y = ggml_sum_rows(c, ggml_mul(c, col, ggml_reshape_3d(c, kernel, k, 1, ch)));
    return ggml_reshape_2d(c, y, y->ne[1], ch);
}

// Grouped time-major convolution, kernel [K, in/groups, out].
inline ggml_tensor * conv_grouped(ggml_context * c, ggml_tensor * kernel, ggml_tensor * x, int groups, int pad) {
    const int64_t per = x->ne[1] / groups;
    require(kernel->ne[1] == per && kernel->ne[2] == x->ne[1], "grouped kernel shape");
    ggml_tensor * y = nullptr;
    for (int g = 0; g < groups; ++g) {
        auto xg = ggml_view_2d(c, x, x->ne[0], per, x->nb[1], size_t(g) * per * x->nb[1]);
        auto kg = ggml_view_3d(c, kernel, kernel->ne[0], per, per, kernel->nb[1], kernel->nb[2], size_t(g) * per * kernel->nb[2]);
        auto part = conv_strided(c, kg, xg, 1, pad);
        y = y ? ggml_concat(c, y, part, 1) : part;
    }
    return y;
}

inline ggml_tensor * transpose(ggml_context * c, ggml_tensor * x) { return ggml_cont(c, ggml_transpose(c, x)); }

// q, k, v [head_dim, heads, time]; optional additive bias [time_k, time_q, heads].
inline ggml_tensor * attend(ggml_context * c, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale,
                            ggml_tensor * bias = nullptr) {
    const int64_t hd = q->ne[0], heads = q->ne[1], tq = q->ne[2];
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    auto kq = ggml_mul_mat(c, k, q);
    if (bias) kq = ggml_soft_max(c, ggml_add(c, ggml_scale(c, kq, scale), bias));
    else kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
    auto vt = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));
    auto out = ggml_permute(c, ggml_mul_mat(c, vt, kq), 0, 2, 1, 3);
    return ggml_cont_2d(c, out, hd * heads, tq);
}

// Mean of adjacent time pairs: [d, 2n] -> [d, n].
inline ggml_tensor * pair_mean(ggml_context * c, ggml_tensor * x, int64_t pairs) {
    auto even = ggml_view_2d(c, x, x->ne[0], pairs, 2 * x->nb[1], 0);
    auto odd = ggml_view_2d(c, x, x->ne[0], pairs, 2 * x->nb[1], x->nb[1]);
    return ggml_scale(c, ggml_add(c, even, odd), 0.5f);
}

inline ggml_tensor * positions(Graph & g, int64_t n) {
    std::vector<int32_t> pos(static_cast<size_t>(n));
    std::iota(pos.begin(), pos.end(), 0);
    return g.input(GGML_TYPE_I32, {n}, pos.data());
}

// ── MuEncoder ───────────────────────────────────────────────────────────────

// Conv2dSubsampling on one time window of the mel, [time, 128] -> [1024, time / 4].
inline ggml_tensor * muencoder_frontend(Graph & g, const Weights & w, ggml_tensor * mel) {
    auto c = g.ctx;
    const std::string p = "encoder.feature_extractors.muencoder.conv.";
    auto bn = [&](const std::string & q, ggml_tensor * x) {
        int64_t ch = x->ne[2];
        return ggml_add(c, ggml_mul(c, x, ggml_reshape_3d(c, w.get(q + ".scale"), 1, 1, ch)),
                        ggml_reshape_3d(c, w.get(q + ".shift"), 1, 1, ch));
    };
    // ggml_conv_2d with an F32 im2col (ggml's own rounds the columns to F16).
    auto conv2 = [&](const std::string & q, ggml_tensor * x, int stride) {
        auto k = w.get(q + ".weight");
        auto col = ggml_im2col(c, k, x, stride, stride, 1, 1, 1, 1, true, GGML_TYPE_F32);
        auto y = ggml_mul_mat(c, ggml_reshape_2d(c, col, col->ne[0], col->ne[3] * col->ne[2] * col->ne[1]),
                              ggml_reshape_2d(c, k, k->ne[0] * k->ne[1] * k->ne[2], k->ne[3]));
        y = ggml_reshape_4d(c, y, col->ne[1], col->ne[2], col->ne[3], k->ne[3]);
        y = ggml_cont(c, ggml_permute(c, y, 0, 1, 3, 2));
        return ggml_add(c, y, ggml_reshape_3d(c, w.get(q + ".bias"), 1, 1, y->ne[2]));
    };
    auto x = ggml_reshape_4d(c, mel, mel->ne[0], mel->ne[1], 1, 1);
    for (int b = 0; b < 2; ++b) {
        const std::string q = p + "conv." + std::to_string(b) + ".";
        auto out = bn(q + "bn2", conv2(q + "conv2", ggml_relu(c, bn(q + "bn1", conv2(q + "conv1", x, 2))), 1));
        x = ggml_relu(c, ggml_add(c, bn(q + "bn3", conv2(q + "conv3", x, 2)), out));
    }
    // [time, 32, 512] -> [32 * 512, time], feature index channel * 32 + band.
    const int64_t time = x->ne[0];
    x = ggml_cont(c, ggml_permute(c, x, 2, 0, 1, 3));
    return linear(c, w, p + "linear", ggml_reshape_2d(c, x, x->ne[0] * x->ne[1], time));
}

inline ggml_tensor * ffn(ggml_context * c, const Weights & w, const std::string & p, ggml_tensor * x, bool swish) {
    auto h = linear(c, w, p + ".intermediate_dense", x);
    h = swish ? ggml_silu(c, h) : ggml_gelu_erf(c, h);
    return linear(c, w, p + ".output_dense", h);
}

// Wav2Vec2-Conformer layers 0..10 (rotary on the layer input, before q/k).
// Returns [1024, 2T]: hidden_states[2] then hidden_states[11].
inline ggml_tensor * muencoder_conformer(Graph & g, const Weights & w, ggml_tensor * x) {
    auto c = g.ctx;
    const int64_t time = x->ne[1];
    auto pos = positions(g, time);
    ggml_tensor * acoustic = nullptr;
    for (int l = 0; l <= 10; ++l) {
        const std::string p = "encoder.feature_extractors.muencoder.conformer.layers." + std::to_string(l) + ".";
        x = ggml_add(c, x, ggml_scale(c, ffn(c, w, p + "ffn1", layer_norm(c, w, p + "ffn1_layer_norm", x), true), 0.5f));
        auto h = layer_norm(c, w, p + "self_attn_layer_norm", x);
        auto rotated = ggml_rope_ext(c, ggml_reshape_3d(c, h, 64, 16, time), pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0,
                                     10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        rotated = ggml_reshape_2d(c, rotated, 1024, time);
        auto q = ggml_reshape_3d(c, linear(c, w, p + "self_attn.linear_q", rotated), 64, 16, time);
        auto k = ggml_reshape_3d(c, linear(c, w, p + "self_attn.linear_k", rotated), 64, 16, time);
        auto v = ggml_reshape_3d(c, linear(c, w, p + "self_attn.linear_v", h), 64, 16, time);
        x = ggml_add(c, x, linear(c, w, p + "self_attn.linear_out", attend(c, q, k, v, 0.125f)));
        // Convolution module: pointwise GLU, depthwise 31, BatchNorm, swish, pointwise.
        const std::string m = p + "conv_module.";
        h = ggml_mul_mat(c, w.get(m + "pointwise_conv1.weight"), layer_norm(c, w, m + "layer_norm", x));
        auto values = ggml_cont(c, ggml_view_2d(c, h, 1024, time, h->nb[1], 0));
        auto gate = ggml_cont(c, ggml_view_2d(c, h, 1024, time, h->nb[1], 1024 * ggml_element_size(h)));
        h = transpose(c, ggml_mul(c, values, ggml_sigmoid(c, gate)));
        h = conv_depthwise(c, w.get(m + "depthwise_conv.weight"), h, 15);
        h = ggml_add(c, ggml_mul(c, h, ggml_reshape_2d(c, w.get(m + "batch_norm.scale"), 1, 1024)),
                     ggml_reshape_2d(c, w.get(m + "batch_norm.shift"), 1, 1024));
        h = transpose(c, ggml_silu(c, h));
        x = ggml_add(c, x, ggml_mul_mat(c, w.get(m + "pointwise_conv2.weight"), h));
        x = ggml_add(c, x, ggml_scale(c, ffn(c, w, p + "ffn2", layer_norm(c, w, p + "ffn2_layer_norm", x), true), 0.5f));
        x = layer_norm(c, w, p + "final_layer_norm", x);
        if (l == 1) acoustic = x;
    }
    return ggml_concat(c, acoustic, x, 1);
}

// Output frames per frontend window and the context each side; 4 output frames
// (16 mel frames) cover the two stride-2 residual blocks' receptive field.
constexpr int64_t mu_window = 128, mu_margin = 4;

// One channel's 24 kHz chunk (with tail) -> acoustic and semantic [1024, 744] each.
inline void muencoder_run(const Weights & w, ggml_backend_t backend, const float * audio, double mean, double std_dev,
                          std::vector<float> & acoustic, std::vector<float> & semantic, const Cancel & cancelled,
                          std::vector<float> * mel_out = nullptr, const std::vector<float> * mel_in = nullptr) {
    int64_t mel_frames = enc_frames * 4;
    std::vector<float> computed;
    if (!mel_in) {
        computed = muencoder_mel(w, audio, (enc_chunk_samples + enc_tail) / 960 * 960, mean, std_dev, &mel_frames);
        require(mel_frames == enc_frames * 4, "unexpected MuEncoder mel length");
    }
    const std::vector<float> & mel = mel_in ? *mel_in : computed;
    if (mel_out) *mel_out = mel;
    std::vector<float> hidden(size_t(1024 * enc_frames));
    for (int64_t s = 0; s < enc_frames; s += mu_window) {
        check_cancel(cancelled);
        const int64_t e = std::min(enc_frames, s + mu_window);
        const int64_t a = std::max<int64_t>(0, s - mu_margin), b = std::min(enc_frames, e + mu_margin);
        const int64_t n = (b - a) * 4;
        std::vector<float> piece(size_t(n * 128));
        for (int m = 0; m < 128; ++m) std::copy_n(mel.data() + m * mel_frames + a * 4, n, piece.data() + m * n);
        Graph g(backend, 256);
        auto out = g.run(muencoder_frontend(g, w, g.input(GGML_TYPE_F32, {n, 128}, piece.data())));
        require(out.size() == size_t(1024 * (b - a)), "unexpected MuEncoder frontend length");
        std::copy(out.begin() + (s - a) * 1024, out.begin() + (e - a) * 1024, hidden.begin() + s * 1024);
    }
    check_cancel(cancelled);
    Graph g(backend, 4096);
    auto both = g.run(muencoder_conformer(g, w, g.input(GGML_TYPE_F32, {1024, enc_frames}, hidden.data())));
    acoustic.assign(both.begin(), both.begin() + 1024 * enc_frames);
    semantic.assign(both.begin() + 1024 * enc_frames, both.end());
}

// ── WavLM ───────────────────────────────────────────────────────────────────

// WavLM relative-position bucket (320 buckets, max distance 800) in torch's float32.
inline int32_t wavlm_bucket(int64_t relative) {
    int32_t bucket = relative > 0 ? 160 : 0;
    const int64_t a = relative < 0 ? -relative : relative;
    if (a < 80) return bucket + int32_t(a);
    float v = std::log(float(a) / 80.0f) / float(std::log(10.0)) * 80.0f;
    return bucket + std::min<int32_t>(int32_t(80.0f + v), 159);
}

// Seven-stage waveform convolution: 16 kHz [n] -> [512, n / 320] at 50 Hz.
inline ggml_tensor * wavlm_frontend(Graph & g, const Weights & w, const std::vector<float> & audio) {
    const std::string p = "encoder.feature_extractors.wavlm_encoder.feature_extractor.conv_layers.";
    auto c = g.ctx;
    auto x = g.input(GGML_TYPE_F32, {int64_t(audio.size()), 1}, audio.data());
    constexpr int strides[] = {5, 2, 2, 2, 2, 2, 2};
    for (int l = 0; l < 7; ++l) {
        const std::string q = p + std::to_string(l) + ".";
        x = conv_strided(c, w.get(q + "conv.weight"), x, strides[l], 0);
        if (l == 0) {  // GroupNorm with one group per channel: normalize over time.
            x = ggml_norm(c, x, 1e-5f);
            x = ggml_add(c, ggml_mul(c, x, ggml_reshape_2d(c, w.get(q + "layer_norm.weight"), 1, 512)),
                         ggml_reshape_2d(c, w.get(q + "layer_norm.bias"), 1, 512));
        }
        x = ggml_gelu_erf(c, x);
    }
    return transpose(c, x);
}

// One channel's 16 kHz audio -> the mean of hidden states 6..9 at 50 Hz [768, 1488],
// then pair-averaged to [768, 744].
inline std::vector<float> wavlm_run(const Weights & w, ggml_backend_t backend, const std::vector<float> & audio) {
    const std::string p = "encoder.feature_extractors.wavlm_encoder.";
    Graph g(backend, 4096);
    auto c = g.ctx;
    auto extract = wavlm_frontend(g, w, audio);
    const int64_t time = extract->ne[1];
    require(time == 2 * enc_frames, "WavLM frame count does not match the 24 kHz alignment contract");
    auto x = linear(c, w, p + "feature_projection.projection", layer_norm(c, w, p + "feature_projection.layer_norm", extract));
    auto pos = conv_grouped(c, w.get(p + "encoder.pos_conv_embed.conv.weight"), transpose(c, x), 16, 64);
    pos = ggml_view_2d(c, pos, time, 768, pos->nb[1], 0);
    pos = ggml_gelu_erf(c, ggml_add(c, pos, ggml_reshape_2d(c, w.get(p + "encoder.pos_conv_embed.conv.bias"), 1, 768)));
    x = layer_norm(c, w, p + "encoder.layer_norm", ggml_add(c, x, transpose(c, pos)));
    // Layer 0's bucket embedding, shared ungated by every layer: [time_k, time_q, 12].
    std::vector<int32_t> buckets(size_t(time * time));
    for (int64_t tq = 0; tq < time; ++tq) for (int64_t tk = 0; tk < time; ++tk)
        buckets[size_t(tq * time + tk)] = wavlm_bucket(tk - tq);
    auto rows = ggml_get_rows(c, w.get(p + "encoder.layers.0.attention.rel_attn_embed.weight"),
                              g.input(GGML_TYPE_I32, {time * time}, buckets.data()));
    auto bias = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, rows, 12, time, time), 2, 0, 1, 3));
    ggml_tensor * sum = nullptr;
    for (int l = 0; l <= 8; ++l) {
        const std::string q = p + "encoder.layers." + std::to_string(l) + ".";
        const std::string a = q + "attention.";
        // Gate per head and query: sigmoid of two 4-wide sums of an 8-wide projection.
        auto gp = linear(c, w, a + "gru_rel_pos_linear", ggml_reshape_3d(c, x, 64, 12, time));
        gp = ggml_sigmoid(c, ggml_sum_rows(c, ggml_reshape_4d(c, gp, 4, 2, 12, time)));
        auto ga = ggml_cont(c, ggml_view_3d(c, gp, 1, 12, time, gp->nb[2], gp->nb[3], 0));
        auto gb = ggml_cont(c, ggml_view_3d(c, gp, 1, 12, time, gp->nb[2], gp->nb[3], gp->nb[1]));
        // gate_a * (gate_b * const - 1) + 2, then [12, time] -> [1, time, 12] to scale the bias rows.
        auto constant = ggml_reshape_3d(c, w.get(a + "gru_rel_pos_const"), 1, 12, 1);
        auto gate = ggml_scale_bias(c, ggml_mul(c, ga, ggml_scale_bias(c, ggml_mul(c, gb, constant), 1.0f, -1.0f)), 1.0f, 2.0f);
        gate = ggml_reshape_3d(c, transpose(c, ggml_reshape_2d(c, ggml_cont(c, gate), 12, time)), 1, time, 12);
        auto qh = ggml_reshape_3d(c, linear(c, w, a + "q_proj", x), 64, 12, time);
        auto kh = ggml_reshape_3d(c, linear(c, w, a + "k_proj", x), 64, 12, time);
        auto vh = ggml_reshape_3d(c, linear(c, w, a + "v_proj", x), 64, 12, time);
        auto attn = linear(c, w, a + "out_proj", attend(c, qh, kh, vh, 0.125f, ggml_mul(c, bias, gate)));
        x = layer_norm(c, w, q + "layer_norm", ggml_add(c, x, attn));
        x = layer_norm(c, w, q + "final_layer_norm", ggml_add(c, x, ffn(c, w, q + "feed_forward", x, false)));
        if (l >= 5) sum = sum ? ggml_add(c, sum, x) : x;
    }
    return g.run(pair_mean(c, ggml_scale(c, sum, 0.25f), enc_frames));
}

// ── Whisper ─────────────────────────────────────────────────────────────────

// Whisper encoder over the padded 30 s window; the first 1488 positions are kept and
// pair-averaged: [768, 744].
inline std::vector<float> whisper_run(const Weights & w, ggml_backend_t backend, const std::vector<float> & mel) {
    const std::string p = "encoder.feature_extractors.whisper_encoder.";
    require(mel.size() == size_t(80 * 3000), "Whisper mel must be [80, 3000]");
    Graph g(backend, 4096);
    auto c = g.ctx;
    auto x = g.input(GGML_TYPE_F32, {3000, 80}, mel.data());
    x = ggml_gelu_erf(c, conv(c, w, p + "conv1", x, 1));
    x = conv_strided(c, w.get(p + "conv2.weight"), x, 2, 1);
    x = ggml_gelu_erf(c, ggml_add(c, x, ggml_reshape_2d(c, w.get(p + "conv2.bias"), 1, 768)));
    x = ggml_add(c, transpose(c, x), w.get(p + "embed_positions.weight"));
    for (int l = 0; l < 12; ++l) {
        const std::string q = p + "layers." + std::to_string(l) + ".";
        auto h = layer_norm(c, w, q + "self_attn_layer_norm", x);
        auto qh = ggml_reshape_3d(c, linear(c, w, q + "self_attn.q_proj", h), 64, 12, 1500);
        auto kh = ggml_reshape_3d(c, linear(c, w, q + "self_attn.k_proj", h), 64, 12, 1500);
        auto vh = ggml_reshape_3d(c, linear(c, w, q + "self_attn.v_proj", h), 64, 12, 1500);
        x = ggml_add(c, x, linear(c, w, q + "self_attn.out_proj", attend(c, qh, kh, vh, 0.125f)));
        h = ggml_gelu_erf(c, linear(c, w, q + "fc1", layer_norm(c, w, q + "final_layer_norm", x)));
        x = ggml_add(c, x, linear(c, w, q + "fc2", h));
    }
    x = layer_norm(c, w, p + "layer_norm", x);
    return g.run(pair_mean(c, x, enc_frames));
}

} // namespace heartcodec
