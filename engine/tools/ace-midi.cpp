// ace-midi.cpp: MuScriptor audio->MIDI transcription (GGML port)
//
// Native port of the MuScriptor transcription model (Kyutai & Mirelo,
// arXiv:2607.08168, code MIT, weights CC BY-NC 4.0). Decoder-only causal
// transformer with mel-spectrogram prefix conditioning; MT3 event vocab.
// Design + validation plan: docs/plans/muscriptor-cpp-port.md.
//
// Phase 1 (this file, current state): weight loading + transformer prefill
// graph + logit-parity selftest against the Python oracle dumps produced by
// tools/ace-midi-validate.py.
//
//   ace-midi --model <dir> --validate <dir>
//     <dir>/model.safetensors + config.json ; validation dir with
//     prefix.bin / logits_bos.bin / manifest.json
//
// Later phases add: mel frontend, chunked greedy decode with KV cache +
// prelude forcing, note-event decode, MIDI writer, JSONL streaming.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend.h"
#include "ggml.h"
#include "safetensors.h"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct MidiConfig {
    int dim        = 768;
    int num_heads  = 12;
    int num_layers = 14;
    int card       = 1393;
    int head_dim() const { return dim / num_heads; }
    int ffn_dim() const { return 4 * dim; }
    int bos_id() const { return card; }  // "initial token" = card
};

static int json_int_field(const char * json, const char * key, int fb) {
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char * p = strstr(json, needle);
    if (!p) return fb;
    p = strchr(p + strlen(needle), ':');
    if (!p) return fb;
    return atoi(p + 1);
}

static bool load_config(MidiConfig * c, const std::string & dir) {
    std::string path = dir + "/config.json";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot open %s\n", path.c_str());
        return false;
    }
    std::string j(65536, 0);
    size_t n = fread(j.data(), 1, j.size() - 1, f);
    fclose(f);
    j.resize(n);
    c->dim        = json_int_field(j.c_str(), "dim", c->dim);
    c->num_heads  = json_int_field(j.c_str(), "num_heads", c->num_heads);
    c->num_layers = json_int_field(j.c_str(), "num_layers", c->num_layers);
    c->card       = json_int_field(j.c_str(), "card", c->card);
    fprintf(stderr, "[ace-midi] config: dim=%d heads=%d layers=%d card=%d\n",
            c->dim, c->num_heads, c->num_layers, c->card);
    return true;
}

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct MidiLayer {
    ggml_tensor * norm1_w, * norm1_b;
    ggml_tensor * in_proj;    // [dim, 3*dim] (ggml: ne0=in)
    ggml_tensor * out_proj;   // [dim, dim]
    ggml_tensor * norm2_w, * norm2_b;
    ggml_tensor * ffn1;       // [dim, 4*dim]
    ggml_tensor * ffn2;       // [4*dim, dim]
};

// Constants fixed by the upstream model (transcription_model.py)
#define MIDI_SAMPLE_RATE   16000
#define MIDI_CHUNK_SAMPLES 80000   // 5 s
#define MIDI_N_FFT         2048
#define MIDI_HOP           160     // 100 Hz frame rate
#define MIDI_N_MELS        512
#define MIDI_MEL_FRAMES    501     // 1 + 80000/160 (center=True)
#define MIDI_MAX_GEN       2000    // max tokens per chunk
#define MIDI_EOS_ID        1

struct MidiModel {
    MidiConfig             cfg;
    ggml_context *         wctx = nullptr;
    ggml_backend_buffer_t  wbuf = nullptr;
    ggml_tensor *          emb;         // [dim, card+1]
    std::vector<MidiLayer> layers;
    ggml_tensor *          out_norm_w, * out_norm_b;
    ggml_tensor *          head;        // [dim, card]

    ggml_backend_t       backend, cpu_backend;
    ggml_backend_sched_t sched;

    // KV cache (batch=1): per layer K [D, max_seq, H], V [max_seq, D, H], f32
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
    int                        max_seq = 0;

    // Host-side copies for CPU input assembly / mel frontend
    std::vector<float> emb_host;     // [card+1, dim] (row-major, incl. BOS row)
    std::vector<float> mel_window;   // [2048]
    std::vector<float> mel_fb;       // [1025, 512] row-major
    std::vector<float> mel_proj_w;   // [dim, 512] row-major (torch [out,in])
    std::vector<float> mel_proj_b;   // [dim]
    std::vector<float> ds_null_emb;  // dataset_name embed row 1 (None cond)
    std::vector<float> ig_null_emb;  // instrument_group embed row 1 (None cond)
};

// Create a ggml tensor mirroring a safetensors entry (torch [out,in] row-major
// -> ggml [in, out]) and upload its data, converting BF16/F16 -> F32.
static ggml_tensor * load_tensor(MidiModel * m, ggml_context * ctx, const STFile & st,
                                 const std::string & name, int64_t ne0, int64_t ne1) {
    (void) m;
    const STEntry * e = st_find(st, name.c_str());
    if (!e) {
        fprintf(stderr, "[ace-midi] FATAL: missing tensor %s\n", name.c_str());
        exit(1);
    }
    // torch shape is [out, in] row-major -> ggml [ne0=in, ne1=out], same memory
    ggml_tensor * t = ne1 > 0
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1)
        : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    ggml_set_name(t, name.c_str());
    return t;
}

static void upload_tensor(MidiModel * m, const STFile & st, ggml_tensor * t) {
    const STEntry * e = st_find(st, t->name);
    size_t n = ggml_nelements(t);
    // verify element count matches
    int64_t st_n = 1;
    for (int i = 0; i < e->n_dims; i++) st_n *= e->shape[i];
    if ((int64_t) n != st_n) {
        fprintf(stderr, "[ace-midi] FATAL: %s shape mismatch (st=%lld ggml=%zu)\n",
                t->name, (long long) st_n, n);
        exit(1);
    }
    const void * src = st_data(st, *e);
    if (e->dtype == "F32") {
        ggml_backend_tensor_set(t, src, 0, n * 4);
    } else if (e->dtype == "BF16") {
        std::vector<float> tmp(n);
        const uint16_t * s = (const uint16_t *) src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t) s[i] << 16;
            memcpy(&tmp[i], &bits, 4);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, n * 4);
    } else if (e->dtype == "F16") {
        std::vector<float> tmp(n);
        const ggml_fp16_t * s = (const ggml_fp16_t *) src;
        for (size_t i = 0; i < n; i++) tmp[i] = ggml_fp16_to_fp32(s[i]);
        ggml_backend_tensor_set(t, tmp.data(), 0, n * 4);
    } else {
        fprintf(stderr, "[ace-midi] FATAL: %s unsupported dtype %s\n", t->name, e->dtype.c_str());
        exit(1);
    }
}

// Published checkpoints use the legacy multi-codebook key layout for the
// embedding and head (emb.0.* / linears.0.*) — same remap as the Python
// loader's _remap_single_codebook_keys.
static std::string resolve_key(const STFile & st, const std::string & canonical, const std::string & legacy) {
    if (st_find(st, canonical.c_str())) return canonical;
    if (st_find(st, legacy.c_str())) return legacy;
    return canonical;  // load_tensor will report it missing
}

static bool load_model(MidiModel * m, const std::string & dir) {
    if (!load_config(&m->cfg, dir)) return false;
    const MidiConfig & c = m->cfg;

    STFile st;
    std::string wpath = dir + "/model.safetensors";
    if (!st_open(&st, wpath.c_str())) return false;

    BackendPair bp = backend_init("MIDI");
    m->backend     = bp.backend;
    m->cpu_backend = bp.cpu_backend;
    m->sched       = backend_sched_new(bp, 8192);

    int n_tensors = 3 /*emb, out_norm w/b*/ + 1 /*head*/ + c.num_layers * 8;
    ggml_init_params ip = { (size_t) n_tensors * ggml_tensor_overhead() + 4096, NULL, true };
    m->wctx = ggml_init(ip);

    const std::string emb_key  = resolve_key(st, "emb.weight", "emb.0.weight");
    const std::string head_key = resolve_key(st, "linear.weight", "linears.0.weight");
    m->emb        = load_tensor(m, m->wctx, st, emb_key, c.dim, c.card + 1);
    m->out_norm_w = load_tensor(m, m->wctx, st, "out_norm.weight", c.dim, 0);
    m->out_norm_b = load_tensor(m, m->wctx, st, "out_norm.bias", c.dim, 0);
    m->head       = load_tensor(m, m->wctx, st, head_key, c.dim, c.card);

    m->layers.resize(c.num_layers);
    for (int l = 0; l < c.num_layers; l++) {
        char base[96];
        snprintf(base, sizeof(base), "transformer.layers.%d.", l);
        MidiLayer & L = m->layers[l];
        L.norm1_w  = load_tensor(m, m->wctx, st, std::string(base) + "norm1.weight", c.dim, 0);
        L.norm1_b  = load_tensor(m, m->wctx, st, std::string(base) + "norm1.bias", c.dim, 0);
        L.in_proj  = load_tensor(m, m->wctx, st, std::string(base) + "self_attn.in_proj_weight", c.dim, 3 * c.dim);
        L.out_proj = load_tensor(m, m->wctx, st, std::string(base) + "self_attn.out_proj.weight", c.dim, c.dim);
        L.norm2_w  = load_tensor(m, m->wctx, st, std::string(base) + "norm2.weight", c.dim, 0);
        L.norm2_b  = load_tensor(m, m->wctx, st, std::string(base) + "norm2.bias", c.dim, 0);
        L.ffn1     = load_tensor(m, m->wctx, st, std::string(base) + "linear1.weight", c.dim, c.ffn_dim());
        L.ffn2     = load_tensor(m, m->wctx, st, std::string(base) + "linear2.weight", c.ffn_dim(), c.dim);
    }

    m->wbuf = ggml_backend_alloc_ctx_tensors(m->wctx, m->backend);
    if (!m->wbuf) {
        fprintf(stderr, "[ace-midi] FATAL: weight buffer alloc failed\n");
        return false;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(m->wctx); t; t = ggml_get_next_tensor(m->wctx, t)) {
        upload_tensor(m, st, t);
    }

    // Host-side copies: token embeddings (input assembly), mel frontend
    // weights, and the null class-conditioner rows. Conditioner tokenize
    // maps None -> -1, +1 in tokenize, +1 again in forward => row 1.
    auto read_host = [&](const char * name, std::vector<float> & out) {
        const STEntry * e = st_find(st, name);
        if (!e) {
            fprintf(stderr, "[ace-midi] FATAL: missing tensor %s\n", name);
            exit(1);
        }
        int64_t n = 1;
        for (int i = 0; i < e->n_dims; i++) n *= e->shape[i];
        out.resize((size_t) n);
        const void * src = st_data(st, *e);
        if (e->dtype == "F32") {
            memcpy(out.data(), src, (size_t) n * 4);
        } else {
            const uint16_t * s = (const uint16_t *) src;
            for (int64_t i = 0; i < n; i++) {
                if (e->dtype == "BF16") {
                    uint32_t bits = (uint32_t) s[i] << 16;
                    memcpy(&out[i], &bits, 4);
                } else {
                    out[i] = ggml_fp16_to_fp32((ggml_fp16_t) s[i]);
                }
            }
        }
    };
    read_host(emb_key.c_str(), m->emb_host);
    read_host("condition_provider.conditioners.self_wav.mel_spec_transform.spectrogram.window", m->mel_window);
    read_host("condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb", m->mel_fb);
    read_host("condition_provider.conditioners.self_wav.output_proj.weight", m->mel_proj_w);
    read_host("condition_provider.conditioners.self_wav.output_proj.bias", m->mel_proj_b);
    {
        std::vector<float> tmp;
        read_host("condition_provider.conditioners.dataset_name.embed.weight", tmp);
        m->ds_null_emb.assign(tmp.begin() + c.dim, tmp.begin() + 2 * c.dim);  // row 1
        read_host("condition_provider.conditioners.instrument_group.embed.weight", tmp);
        m->ig_null_emb.assign(tmp.begin() + c.dim, tmp.begin() + 2 * c.dim);  // row 1
    }

    // KV cache: prefix (mel 501 + 2 class conds) + BOS + tie prompt + max gen
    m->max_seq = MIDI_MEL_FRAMES + 2 + 1 + 300 + MIDI_MAX_GEN;
    {
        const int D = c.head_dim(), H = c.num_heads;
        ggml_init_params kp = { (size_t) c.num_layers * 2 * ggml_tensor_overhead() + 4096, NULL, true };
        m->kv_ctx = ggml_init(kp);
        m->kv_k.resize(c.num_layers);
        m->kv_v.resize(c.num_layers);
        for (int l = 0; l < c.num_layers; l++) {
            m->kv_k[l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F32, D, m->max_seq, H);
            m->kv_v[l] = ggml_new_tensor_3d(m->kv_ctx, GGML_TYPE_F32, m->max_seq, D, H);
            char nm[32];
            snprintf(nm, sizeof(nm), "kv_k_%d", l);
            ggml_set_name(m->kv_k[l], nm);
            snprintf(nm, sizeof(nm), "kv_v_%d", l);
            ggml_set_name(m->kv_v[l], nm);
        }
        m->kv_buf = ggml_backend_alloc_ctx_tensors(m->kv_ctx, m->backend);
        if (!m->kv_buf) {
            fprintf(stderr, "[ace-midi] FATAL: KV cache alloc failed\n");
            return false;
        }
    }

    fprintf(stderr, "[ace-midi] loaded %d layers (%.1f MB weights, %.1f MB KV cache)\n",
            c.num_layers, (double) ggml_backend_buffer_get_size(m->wbuf) / 1e6,
            (double) ggml_backend_buffer_get_size(m->kv_buf) / 1e6);
    st_close(&st);
    return true;
}

// ---------------------------------------------------------------------------
// Sinusoidal positions (transformer.py create_sin_embedding: cat([cos, sin]),
// exponent i/(half_dim - 1), max_period 10000)
// ---------------------------------------------------------------------------

static void add_sin_pos(float * x, int T, int dim, int pos0) {
    int half = dim / 2;
    for (int t = 0; t < T; t++) {
        double pos = (double) (pos0 + t);
        for (int i = 0; i < half; i++) {
            double phase = pos / pow(10000.0, (double) i / (double) (half - 1));
            x[(size_t) t * dim + i]        += (float) cos(phase);
            x[(size_t) t * dim + half + i] += (float) sin(phase);
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill forward: input embeddings [dim, T] -> logits [card, T]
// ---------------------------------------------------------------------------

static ggml_tensor * build_layer_norm(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_mul(ctx, x, w);
    return ggml_add(ctx, x, b);
}

// Forward T tokens at cache position n_past; writes K/V into the cache and
// reads back the LAST position's logits. n_past=0 with T>1 is the prefill;
// T=1 with n_past>0 is a decode step. Caller advances n_past by T afterwards.
static void forward_tokens(MidiModel * m, const float * input, int T, int n_past, float * logits_last) {
    const MidiConfig & c = m->cfg;
    const int H = c.num_heads, D = c.head_dim();
    const int n_kv = n_past + T;

    ggml_init_params ip = { ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, T);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor * x = inp;
    for (int l = 0; l < c.num_layers; l++) {
        MidiLayer & L = m->layers[l];

        // --- causal self-attention with KV cache ---
        ggml_tensor * h   = build_layer_norm(ctx, x, L.norm1_w, L.norm1_b);
        ggml_tensor * qkv = ggml_mul_mat(ctx, L.in_proj, h);  // [3*dim, T]

        ggml_tensor * q = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) c.dim * 4);
        ggml_tensor * v = ggml_view_2d(ctx, qkv, c.dim, T, qkv->nb[1], (size_t) 2 * c.dim * 4);

        // packed layout per token is [h, d] (rearrange "(p h d)")
        ggml_tensor * k3 = ggml_reshape_3d(ctx, ggml_cont(ctx, k), D, H, T);
        ggml_tensor * v3 = ggml_reshape_3d(ctx, ggml_cont(ctx, v), D, H, T);

        // append current K rows: cache K layout [D, max_seq, H], slice dim1 [n_past, n_past+T)
        ggml_tensor * kc = m->kv_k[l];
        ggml_tensor * k_dst = ggml_view_3d(ctx, kc, D, T, H, kc->nb[1], kc->nb[2],
                                           (size_t) n_past * kc->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3), k_dst));

        // append current V rows: cache V layout [max_seq, D, H], slice dim0 [n_past, n_past+T)
        ggml_tensor * vc = m->kv_v[l];
        ggml_tensor * v_dst = ggml_view_3d(ctx, vc, T, D, H, vc->nb[1], vc->nb[2],
                                           (size_t) n_past * vc->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3), v_dst));

        ggml_tensor * Q = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q), D, H, T), 0, 2, 1, 3);  // [D, T, H]
        ggml_tensor * K = ggml_view_3d(ctx, kc, D, n_kv, H, kc->nb[1], kc->nb[2], 0);   // [D, n_kv, H]
        ggml_tensor * V = ggml_view_3d(ctx, vc, n_kv, D, H, vc->nb[1], vc->nb[2], 0);   // [n_kv, D, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                       // [n_kv, T, H]
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) D));
        kq = ggml_diag_mask_inf(ctx, kq, n_past);                         // causal, bottom-right aligned
        kq = ggml_soft_max(ctx, kq);

        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);                     // [D, T, H]
        ggml_tensor * att = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [D, H, T]
        att = ggml_reshape_2d(ctx, att, c.dim, T);
        att = ggml_mul_mat(ctx, L.out_proj, att);

        x = ggml_add(ctx, x, att);

        // --- FFN (exact GELU, matching torch F.gelu default) ---
        ggml_tensor * f = build_layer_norm(ctx, x, L.norm2_w, L.norm2_b);
        f = ggml_mul_mat(ctx, L.ffn1, f);
        f = ggml_gelu_erf(ctx, f);
        f = ggml_mul_mat(ctx, L.ffn2, f);
        x = ggml_add(ctx, x, f);
    }

    x = build_layer_norm(ctx, x, m->out_norm_w, m->out_norm_b);
    ggml_tensor * logits = ggml_mul_mat(ctx, m->head, x);  // [card, T]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(m->sched);
    if (!ggml_backend_sched_alloc_graph(m->sched, gf)) {
        fprintf(stderr, "[ace-midi] FATAL: graph alloc failed\n");
        exit(1);
    }
    ggml_backend_tensor_set(inp, input, 0, (size_t) c.dim * T * 4);
    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ace-midi] FATAL: graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(logits, logits_last, (size_t) (T - 1) * logits->nb[1], (size_t) c.card * 4);

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// Mel frontend (conditioners.py MelSpectrogramConditioner, torchaudio-equiv):
// magnitude STFT (2048/160, periodic hann from ckpt, center reflect pad) ->
// HTK mel fb from ckpt -> log(+1e-6) -> output_proj -> zero masked frames.
// ---------------------------------------------------------------------------

static void fft_radix2(float * re, float * im, int n) {
    // exact twiddle table (per stage), computed once — the naive multiplicative
    // twiddle recurrence drifts enough to visibly perturb the log-mel
    static std::vector<float> tw_re, tw_im;
    static int tw_n = 0;
    if (tw_n != n) {
        tw_re.assign((size_t) n, 0.0f);
        tw_im.assign((size_t) n, 0.0f);
        for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
            for (int j = 0; j < len / 2; j++) {
                double ang = -2.0 * 3.14159265358979323846 * j / len;
                tw_re[(size_t) base + j] = (float) cos(ang);
                tw_im[(size_t) base + j] = (float) sin(ang);
            }
        }
        tw_n = n;
    }

    // bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2, base = 0; len <= n; len <<= 1, base += len >> 2) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                int   a = i + j, b = i + j + len / 2;
                float cr = tw_re[(size_t) base + j], ci = tw_im[(size_t) base + j];
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
}

// Compute the full conditioning prefix for one 5 s chunk:
// [mel 501 | dataset_name(None) | instrument_group(None)] rows of dim floats.
// n_samples is the unpadded chunk length (masks trailing mel frames).
static std::vector<float> compute_prefix(MidiModel * m, const float * wav, int n_samples) {
    const MidiConfig & c = m->cfg;
    const int pad = MIDI_N_FFT / 2;
    const int n   = MIDI_CHUNK_SAMPLES;

    // reflect-padded chunk (zero-pad the tail to 5 s first, like F.pad)
    std::vector<float> padded(n + 2 * pad, 0.0f);
    auto sample_at = [&](int i) -> float {
        // reflect at both edges of the zero-padded 80000-sample chunk
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
        return (i >= 0 && i < n_samples) ? wav[i] : 0.0f;
    };
    for (int i = 0; i < n + 2 * pad; i++) padded[i] = sample_at(i - pad);

    const int T = MIDI_MEL_FRAMES;
    std::vector<float> prefix((size_t) (T + 2) * c.dim, 0.0f);

    // frames masked at index >= n_samples/160.0 (length_to_mask semantics)
    const double frame_limit = (double) n_samples / (double) MIDI_HOP;

    std::vector<float>  re(MIDI_N_FFT), im(MIDI_N_FFT), logmel(MIDI_N_MELS);
    std::vector<double> melacc(MIDI_N_MELS);
    for (int f = 0; f < T; f++) {
        if ((double) f >= frame_limit) continue;  // masked -> zero row

        const float * frame = padded.data() + (size_t) f * MIDI_HOP;
        for (int i = 0; i < MIDI_N_FFT; i++) {
            re[i] = frame[i] * m->mel_window[i];
            im[i] = 0.0f;
        }
        fft_radix2(re.data(), im.data(), MIDI_N_FFT);

        // magnitude (power=1.0) -> mel -> log (double accumulation)
        const int n_freq = MIDI_N_FFT / 2 + 1;
        for (int mm = 0; mm < MIDI_N_MELS; mm++) melacc[mm] = 0.0;
        for (int k = 0; k < n_freq; k++) {
            double mag = sqrt((double) re[k] * re[k] + (double) im[k] * im[k]);
            if (mag == 0.0) continue;
            const float * fbrow = m->mel_fb.data() + (size_t) k * MIDI_N_MELS;
            for (int mm = 0; mm < MIDI_N_MELS; mm++) melacc[mm] += mag * fbrow[mm];
        }
        for (int mm = 0; mm < MIDI_N_MELS; mm++) logmel[mm] = logf((float) melacc[mm] + 1e-6f);

        // output_proj: [dim, 512] @ logmel + bias
        float * out = prefix.data() + (size_t) f * c.dim;
        for (int o = 0; o < c.dim; o++) {
            const float * wrow = m->mel_proj_w.data() + (size_t) o * MIDI_N_MELS;
            double acc = m->mel_proj_b[o];
            for (int i = 0; i < MIDI_N_MELS; i++) acc += (double) wrow[i] * logmel[i];
            out[o] = (float) acc;
        }
    }

    // class conditioner rows (always the None/null class at inference)
    memcpy(prefix.data() + (size_t) T * c.dim, m->ds_null_emb.data(), (size_t) c.dim * 4);
    memcpy(prefix.data() + (size_t) (T + 1) * c.dim, m->ig_null_emb.data(), (size_t) c.dim * 4);
    return prefix;
}

// ---------------------------------------------------------------------------
// Greedy chunk decode: prefill [prefix | BOS | prompt] then argmax steps.
// Emits every accepted token (prompt tokens included, EOS excluded) via cb.
// ---------------------------------------------------------------------------

static void greedy_argmax_range(const float * logits, int n_valid, int * out) {
    int best = 0;
    for (int i = 1; i < n_valid; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    *out = best;
}

template <typename TokenCb>
static void decode_chunk(MidiModel * m, const std::vector<float> & prefix,
                         const std::vector<int> & prompt, int max_gen, TokenCb cb) {
    const MidiConfig & c = m->cfg;
    const int n_valid = c.card < 1393 ? c.card : 1393;  // logits[1393:] masked upstream
    const int T_prefix = (int) (prefix.size() / c.dim);

    // prefill input: prefix + BOS + prompt tokens, sinusoidal positions from 0
    int T0 = T_prefix + 1 + (int) prompt.size();
    std::vector<float> input((size_t) T0 * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    for (size_t i = 0; i < prompt.size(); i++) {
        memcpy(input.data() + prefix.size() + (i + 1) * c.dim,
               m->emb_host.data() + (size_t) prompt[i] * c.dim, (size_t) c.dim * 4);
        cb(prompt[i]);  // teacher-forced tokens flow through the stream
    }
    add_sin_pos(input.data(), T0, c.dim, 0);

    std::vector<float> logits(c.card);
    forward_tokens(m, input.data(), T0, 0, logits.data());
    int n_past = T0;

    int tok;
    greedy_argmax_range(logits.data(), n_valid, &tok);

    std::vector<float> step(c.dim);
    for (int i = (int) prompt.size(); i < max_gen; i++) {
        if (tok == MIDI_EOS_ID) return;
        cb(tok);
        if (n_past + 1 > m->max_seq) {
            fprintf(stderr, "[ace-midi] WARNING: KV cache full at %d tokens\n", n_past);
            return;
        }
        memcpy(step.data(), m->emb_host.data() + (size_t) tok * c.dim, (size_t) c.dim * 4);
        add_sin_pos(step.data(), 1, c.dim, n_past);
        forward_tokens(m, step.data(), 1, n_past, logits.data());
        n_past++;
        greedy_argmax_range(logits.data(), n_valid, &tok);
    }
}

// ---------------------------------------------------------------------------
// MT3 event vocabulary (tokenizer/notes.py build_event_vocab, max_shift 1001):
// 0-2 PAD/EOS/UNK | 3-1003 shift | 1004-1131 pitch | 1132-1133 velocity |
// 1134 tie | 1135-1264 program(0-129) | 1265-1392 drum
// ---------------------------------------------------------------------------

enum EvType { EV_SPECIAL, EV_SHIFT, EV_PITCH, EV_VELOCITY, EV_TIE, EV_PROGRAM, EV_DRUM };
struct Ev { EvType type; int value; };

static Ev vocab_decode(int id) {
    if (id < 3)     return { EV_SPECIAL, id };
    if (id < 1004)  return { EV_SHIFT, id - 3 };
    if (id < 1132)  return { EV_PITCH, id - 1004 };
    if (id < 1134)  return { EV_VELOCITY, id - 1132 };
    if (id == 1134) return { EV_TIE, 0 };
    if (id < 1265)  return { EV_PROGRAM, id - 1135 };
    if (id < 1393)  return { EV_DRUM, id - 1265 };
    return { EV_SPECIAL, 2 };
}
static int tok_program(int program) { return 1135 + program; }
static int tok_pitch(int pitch) { return 1004 + pitch; }
#define TOK_TIE 1134

#define DRUM_PROGRAM 128
#define MIN_NOTE_DUR 0.01
#define FRAME_RATE   100

// MT3_FULL_PLUS named groups: representative (first) program -> group name.
// The model always emits the representative program of a group (mt3.py).
static const char * instrument_for_program(int program) {
    switch (program) {
        case 0:  return "acoustic_piano";
        case 2:  return "electric_piano";
        case 8:  return "chromatic_percussion";
        case 16: return "organ";
        case 24: return "acoustic_guitar";
        case 26: return "clean_electric_guitar";
        case 29: return "distorted_electric_guitar";
        case 32: return "acoustic_bass";
        case 33: return "electric_bass";
        case 40: return "violin";
        case 41: return "viola";
        case 42: return "cello";
        case 43: return "contrabass";
        case 46: return "orchestral_harp";
        case 47: return "timpani";
        case 48: return "string_ensemble";
        case 50: return "synth_strings";
        case 52: return "voice";
        case 55: return "orchestra_hit";
        case 56: return "trumpet";
        case 57: return "trombone";
        case 58: return "tuba";
        case 60: return "french_horn";
        case 61: return "brass_section";
        case 64: return "soprano_and_alto_sax";
        case 66: return "tenor_sax";
        case 67: return "baritone_sax";
        case 68: return "oboe";
        case 69: return "english_horn";
        case 70: return "bassoon";
        case 71: return "clarinet";
        case 72: return "flutes";
        case 80: return "synth_lead";
        case 88: return "synth_pad";
        case DRUM_PROGRAM: return "drums";
        default: return nullptr;  // caller formats "program_<n>"
    }
}

// ---------------------------------------------------------------------------
// OpenNoteTracker — 1:1 port of events.py:96-231, the single state machine
// for both event decoding and prelude forcing.
// ---------------------------------------------------------------------------

struct NoteAction {
    enum Kind { START, END, DRUM_HIT } kind;
    int    program;  // rep program (unused for DRUM_HIT)
    int    pitch;
    double time;
};

struct OpenNoteTracker {
    // (program,pitch) -> onset, insertion-ordered like a Python dict
    std::vector<std::pair<std::pair<int, int>, double>> open;

    double seek_time = 0.0, next_seek_time = -1.0;  // <0 == None
    int    start_tick = 0, tick_state = 0;
    int    program = -1, velocity = -1;             // -1 == None
    bool   in_prologue = true, skip_rest = false, chunk_started = false;
    std::vector<std::pair<int, int>> tie_set;

    bool open_has(std::pair<int, int> key) const {
        for (auto & e : open) if (e.first == key) return true;
        return false;
    }
    void open_erase(std::pair<int, int> key) {
        for (size_t i = 0; i < open.size(); i++) {
            if (open[i].first == key) { open.erase(open.begin() + (long) i); return; }
        }
    }
    bool tie_has(std::pair<int, int> key) const {
        for (auto & e : tie_set) if (e == key) return true;
        return false;
    }

    std::vector<NoteAction> end_all(double time) {
        std::vector<NoteAction> a;
        for (auto & e : open) a.push_back({ NoteAction::END, e.first.first, e.first.second, time });
        open.clear();
        return a;
    }

    std::vector<NoteAction> feed_boundary(double seek, double next_seek /* <0 == None */) {
        std::vector<NoteAction> actions;
        if (chunk_started && in_prologue) actions = end_all(seek_time);
        seek_time      = seek;
        next_seek_time = next_seek;
        start_tick     = (int) llround(seek * FRAME_RATE);
        tick_state     = start_tick;
        program        = -1;
        velocity       = -1;
        in_prologue    = true;
        skip_rest      = false;
        tie_set.clear();
        chunk_started  = true;
        return actions;
    }

    std::vector<NoteAction> feed(int token) {
        Ev event = vocab_decode(token);

        if (in_prologue) {
            if (event.type == EV_TIE) {
                in_prologue = false;
                velocity    = -1;
                std::vector<NoteAction> actions;
                std::vector<std::pair<std::pair<int, int>, double>> kept;
                for (auto & e : open) {
                    if (tie_has(e.first)) kept.push_back(e);
                    else actions.push_back({ NoteAction::END, e.first.first, e.first.second, seek_time });
                }
                open = kept;
                return actions;
            }
            if (event.type == EV_SHIFT) {
                in_prologue = false;
                skip_rest   = true;
                return end_all(seek_time);
            }
            if (event.type == EV_PROGRAM) {
                program = event.value;
            } else if (event.type == EV_PITCH && program >= 0) {
                tie_set.push_back({ program, event.value });
            }
            return {};
        }

        if (skip_rest) return {};

        if (event.type == EV_SHIFT) {
            if (event.value > 0) tick_state = start_tick + event.value;
        } else if (event.type == EV_PROGRAM) {
            program = event.value;
        } else if (event.type == EV_VELOCITY) {
            velocity = event.value;
        } else if (event.type == EV_DRUM) {
            double time = (double) tick_state / FRAME_RATE;
            if (next_seek_time < 0 || time < next_seek_time) {
                return { { NoteAction::DRUM_HIT, DRUM_PROGRAM, event.value, time } };
            }
        } else if (event.type == EV_PITCH) {
            if (program < 0 || velocity < 0) return {};
            double time = (double) tick_state / FRAME_RATE;
            if (next_seek_time >= 0 && time >= next_seek_time) return {};
            std::pair<int, int> key = { program, event.value };
            std::vector<NoteAction> actions;
            if (open_has(key)) {
                open_erase(key);
                actions.push_back({ NoteAction::END, key.first, key.second, time });
            }
            if (velocity > 0) {
                open.push_back({ key, time });
                actions.push_back({ NoteAction::START, key.first, key.second, time });
            }
            return actions;
        }
        return {};
    }

    std::vector<NoteAction> finish() {
        if (chunk_started && in_prologue) return end_all(seek_time);
        std::vector<NoteAction> a;
        for (auto & e : open) a.push_back({ NoteAction::END, e.first.first, e.first.second, e.second + MIN_NOTE_DUR });
        open.clear();
        return a;
    }

    // sorted (program,pitch) pairs currently held open (for tie prompts)
    std::vector<std::pair<int, int>> open_keys() const {
        std::vector<std::pair<int, int>> ks;
        for (auto & e : open) ks.push_back(e.first);
        std::sort(ks.begin(), ks.end());
        return ks;
    }
};

// mt3.py tie_section_token_ids: program token once per run of pitches, then tie
static std::vector<int> tie_section_tokens(const std::vector<std::pair<int, int>> & open_keys) {
    std::vector<int> tokens;
    int prog_state = -1;
    for (auto & k : open_keys) {
        if (k.first != prog_state) {
            tokens.push_back(tok_program(k.first));
            prog_state = k.first;
        }
        tokens.push_back(tok_pitch(k.second));
    }
    tokens.push_back(TOK_TIE);
    return tokens;
}

// ---------------------------------------------------------------------------
// Note assembly + cleanup (tokenizer/notes.py validate/trim) + MIDI writer
// (utils/midi.py + note_event2midi — mido-compatible type-1 SMF)
// ---------------------------------------------------------------------------

struct MidiNote {
    bool   is_drum;
    int    program;   // DRUM_PROGRAM for drums
    double onset, offset;
    int    pitch;
};

static void sort_notes_vec(std::vector<MidiNote> & notes) {
    std::stable_sort(notes.begin(), notes.end(), [](const MidiNote & a, const MidiNote & b) {
        if (a.onset != b.onset) return a.onset < b.onset;
        if (a.is_drum != b.is_drum) return !a.is_drum;
        if (a.program != b.program) return a.program < b.program;
        if (a.pitch != b.pitch) return a.pitch < b.pitch;
        return a.offset < b.offset;
    });
}

static void validate_notes_fix(std::vector<MidiNote> & notes) {
    for (auto & n : notes) {
        // matches validate_notes(fix=True): onset>offset -> max(offset, onset+0.01)
        // which is always onset+0.01 in that branch; short non-drum notes padded
        if (n.onset > n.offset) n.offset = n.onset + MIN_NOTE_DUR;
        else if (!n.is_drum && n.offset - n.onset < 0.01) n.offset = n.onset + MIN_NOTE_DUR;
    }
}

static std::vector<MidiNote> trim_overlapping(std::vector<MidiNote> notes) {
    if (notes.size() <= 1) return notes;
    // group by (program, pitch, is_drum); iterate groups in first-appearance order
    std::vector<MidiNote> out;
    std::vector<std::pair<std::pair<int, int>, bool>> seen;
    for (auto & n : notes) {
        std::pair<std::pair<int, int>, bool> ch = { { n.program, n.pitch }, n.is_drum };
        bool dup = false;
        for (auto & s : seen) if (s == ch) { dup = true; break; }
        if (dup) continue;
        seen.push_back(ch);

        std::vector<MidiNote> group;
        for (auto & g : notes) {
            if (g.program == n.program && g.pitch == n.pitch && g.is_drum == n.is_drum) group.push_back(g);
        }
        std::stable_sort(group.begin(), group.end(), [](const MidiNote & a, const MidiNote & b) { return a.onset < b.onset; });
        for (size_t i = 1; i < group.size(); i++) {
            if (group[i - 1].offset > group[i].onset) group[i - 1].offset = group[i].onset;
        }
        for (auto & g : group) if (g.onset < g.offset) out.push_back(g);
    }
    sort_notes_vec(out);
    return out;
}

// SMF helpers (mido-compatible byte layout, no running status)
static void put_be32(std::vector<uint8_t> & b, uint32_t v) {
    b.push_back((uint8_t) (v >> 24)); b.push_back((uint8_t) (v >> 16));
    b.push_back((uint8_t) (v >> 8));  b.push_back((uint8_t) v);
}
static void put_varlen(std::vector<uint8_t> & b, uint32_t v) {
    uint8_t buf[4];
    int n = 0;
    buf[n++] = (uint8_t) (v & 0x7f);
    while (v >>= 7) buf[n++] = (uint8_t) ((v & 0x7f) | 0x80);
    while (n--) b.push_back(buf[n]);
}

struct MidiEvent {  // one channel/meta message at an absolute tick
    double time;
    bool   is_drum;
    int    program;   // track key
    int    velocity;  // 1 = on, 0 = off (pre-writer semantics)
    int    pitch;
};

// events sorted like sort_note_events: (time, is_drum, program, velocity, pitch)
static void sort_midi_events(std::vector<MidiEvent> & evs) {
    std::stable_sort(evs.begin(), evs.end(), [](const MidiEvent & a, const MidiEvent & b) {
        if (a.time != b.time) return a.time < b.time;
        if (a.is_drum != b.is_drum) return !a.is_drum;
        if (a.program != b.program) return a.program < b.program;
        if (a.velocity != b.velocity) return a.velocity < b.velocity;
        return a.pitch < b.pitch;
    });
}

// Serialize notes to a type-1 SMF (480 tpb, 120 bpm, velocity 100), matching
// note_event2midi: per-program named tracks, channels 0-8,10-15 by first
// appearance (overflow shares 15), drums on 9 with +0.01 s synthetic offs.
static std::vector<uint8_t> write_midi(std::vector<MidiNote> notes) {
    const int    TPB = 480;
    const int    TEMPO = 500000;
    const int    VEL = 100;

    validate_notes_fix(notes);
    notes = trim_overlapping(notes);

    // note2note_event + drum offsets (writer adds them at time+0.01)
    std::vector<MidiEvent> evs;
    for (auto & n : notes) {
        evs.push_back({ n.onset, n.is_drum, n.is_drum ? DRUM_PROGRAM : n.program, 1, n.pitch });
        if (!n.is_drum) evs.push_back({ n.offset, false, n.program, 0, n.pitch });
    }
    sort_midi_events(evs);
    {
        std::vector<MidiEvent> drum_offs;
        for (auto & e : evs) {
            if (e.is_drum) drum_offs.push_back({ e.time + 0.01, true, DRUM_PROGRAM, 0, e.pitch });
        }
        for (auto & d : drum_offs) evs.push_back(d);
        sort_midi_events(evs);
    }

    // per-program tracks
    struct Track { std::vector<uint8_t> bytes; int last_tick = 0; int channel = 0; };
    std::vector<int>   track_order;               // program keys, first appearance
    std::vector<Track> tracks;
    std::vector<int>   avail_channels;
    for (int i = 0; i < 9; i++) avail_channels.push_back(i);
    for (int i = 10; i < 16; i++) avail_channels.push_back(i);

    auto track_for = [&](const MidiEvent & e) -> Track & {
        int key = (e.is_drum || e.program == DRUM_PROGRAM) ? DRUM_PROGRAM : e.program;
        for (size_t i = 0; i < track_order.size(); i++) {
            if (track_order[i] == key) return tracks[i];
        }
        track_order.push_back(key);
        tracks.emplace_back();
        Track & t = tracks.back();
        int gm_program;
        std::string name;
        if (key == DRUM_PROGRAM) {
            t.channel  = 9;
            gm_program = 0;
            name       = "drums";
        } else {
            if (!avail_channels.empty()) {
                t.channel = avail_channels.front();
                avail_channels.erase(avail_channels.begin());
            } else {
                t.channel = 15;
            }
            gm_program = key;
            const char * inm = instrument_for_program(key);
            if (inm) {
                name = inm;
                for (auto & ch : name) if (ch == '_') ch = ' ';  // program_names uses spaces
            } else {
                char buf[32];
                snprintf(buf, sizeof(buf), "program %d", key);
                name = buf;
            }
        }
        // track_name meta + program_change, both at delta 0
        put_varlen(t.bytes, 0);
        t.bytes.push_back(0xff); t.bytes.push_back(0x03);
        put_varlen(t.bytes, (uint32_t) name.size());
        t.bytes.insert(t.bytes.end(), name.begin(), name.end());
        put_varlen(t.bytes, 0);
        t.bytes.push_back((uint8_t) (0xc0 | t.channel));
        t.bytes.push_back((uint8_t) gm_program);
        return t;
    };

    for (auto & e : evs) {
        // mido second2tick + round (banker's) — nearbyint matches
        int tick = (int) nearbyint(e.time * 1e6 / TEMPO * TPB);
        Track & t = track_for(e);
        int delta = tick - t.last_tick;
        t.last_tick = tick;
        put_varlen(t.bytes, (uint32_t) (delta < 0 ? 0 : delta));
        t.bytes.push_back((uint8_t) ((e.velocity > 0 ? 0x90 : 0x80) | t.channel));
        t.bytes.push_back((uint8_t) e.pitch);
        t.bytes.push_back((uint8_t) (e.velocity > 0 ? VEL : 0));
    }

    // assemble file: meta track (set_tempo) + program tracks, each + end_of_track
    std::vector<uint8_t> out;
    out.insert(out.end(), { 'M', 'T', 'h', 'd' });
    put_be32(out, 6);
    out.push_back(0); out.push_back(1);                                  // format 1
    uint16_t ntrks = (uint16_t) (1 + tracks.size());
    out.push_back((uint8_t) (ntrks >> 8)); out.push_back((uint8_t) ntrks);
    out.push_back((uint8_t) (TPB >> 8)); out.push_back((uint8_t) TPB);

    auto emit_track = [&](const std::vector<uint8_t> & body) {
        out.insert(out.end(), { 'M', 'T', 'r', 'k' });
        put_be32(out, (uint32_t) (body.size() + 4));                     // + end_of_track
        out.insert(out.end(), body.begin(), body.end());
        out.insert(out.end(), { 0x00, 0xff, 0x2f, 0x00 });
    };
    {
        std::vector<uint8_t> meta;
        put_varlen(meta, 0);
        meta.push_back(0xff); meta.push_back(0x51); meta.push_back(0x03);
        meta.push_back((uint8_t) (TEMPO >> 16)); meta.push_back((uint8_t) (TEMPO >> 8)); meta.push_back((uint8_t) TEMPO);
        emit_track(meta);
    }
    for (auto & t : tracks) emit_track(t.bytes);
    return out;
}

// ---------------------------------------------------------------------------
// Validation mode (Phase 1): logit parity vs the Python oracle
// ---------------------------------------------------------------------------

static std::vector<float> read_f32_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot open %s\n", path.c_str());
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v((size_t) sz / 4);
    size_t got = fread(v.data(), 4, v.size(), f);
    fclose(f);
    if (got != v.size()) {
        fprintf(stderr, "[ace-midi] short read on %s\n", path.c_str());
        exit(1);
    }
    return v;
}

// Parity gate: strict 1e-3 default is calibrated for the fp32 CPU backend
// (oracle is fp32 CPU torch). GPU backends drift ~5e-3 from TF32/accumulation
// order while still matching argmax — pass a looser --tol there; the real GPU
// acceptance test is greedy token-stream parity (Phase 2).
static int run_validate(MidiModel * m, const std::string & vdir, double tol) {
    const MidiConfig & c = m->cfg;

    std::vector<float> prefix = read_f32_file(vdir + "/prefix.bin");
    std::vector<float> ref    = read_f32_file(vdir + "/logits_bos.bin");
    if ((int) ref.size() != c.card) {
        fprintf(stderr, "[ace-midi] logits_bos.bin size %zu != card %d\n", ref.size(), c.card);
        return 1;
    }
    int T_prefix = (int) (prefix.size() / c.dim);
    int T        = T_prefix + 1;
    fprintf(stderr, "[ace-midi] validate: prefix %d frames + BOS\n", T_prefix);

    // Assemble input: [prefix | BOS] + sinusoidal positions
    std::vector<float> input((size_t) T * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    add_sin_pos(input.data(), T, c.dim, 0);

    std::vector<float> logits(c.card);
    int64_t t0 = ggml_time_ms();
    forward_tokens(m, input.data(), T, 0, logits.data());
    fprintf(stderr, "[ace-midi] prefill (%d tokens): %lld ms\n", T, (long long) (ggml_time_ms() - t0));

    // Compare
    double max_abs = 0, max_rel = 0;
    int    argmax_cpp = 0, argmax_ref = 0;
    for (int i = 0; i < c.card; i++) {
        double d = fabs((double) logits[i] - (double) ref[i]);
        if (d > max_abs) max_abs = d;
        double r = d / (fabs((double) ref[i]) + 1e-6);
        if (r > max_rel) max_rel = r;
        if (logits[i] > logits[argmax_cpp]) argmax_cpp = i;
        if (ref[i] > ref[argmax_ref]) argmax_ref = i;
    }
    printf("max_abs_diff = %.6g\nmax_rel_diff = %.6g\n", max_abs, max_rel);
    printf("argmax: cpp=%d (%.4f)  ref=%d (%.4f)\n", argmax_cpp, logits[argmax_cpp], argmax_ref, ref[argmax_ref]);

    bool pass = max_abs < tol && argmax_cpp == argmax_ref;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Full transcription: chunk loop + prelude forcing + event decode.
// Emits NoteStart/NoteEnd events (indexed, instrument-named) via callbacks
// and returns the assembled note list for the MIDI writer.
// ---------------------------------------------------------------------------

struct NoteEventOut {
    bool   is_start;
    int    index;       // NoteStart index / NoteEnd's start index
    int    pitch;
    double time;
    int    program;     // rep program (DRUM_PROGRAM for drums)
};

struct TranscribeResult {
    std::vector<NoteEventOut> events;
    std::vector<MidiNote>     notes;
};

static std::string instrument_name(int program) {
    const char * n = instrument_for_program(program);
    if (n) return n;
    char buf[32];
    snprintf(buf, sizeof(buf), "program_%d", program);
    return buf;
}

// jsonl: stream events + progress to stdout as they decode (demo-site UX)
static TranscribeResult transcribe_wav(MidiModel * m, const float * wav, int n_samples, bool jsonl) {
    TranscribeResult res;
    OpenNoteTracker  tracker;

    // open (program,pitch) -> (event index, onset) for pairing + note assembly
    std::vector<std::pair<std::pair<int, int>, std::pair<int, double>>> open_ev;
    int next_index = 0;

    auto handle_actions = [&](const std::vector<NoteAction> & actions) {
        for (auto & a : actions) {
            if (a.kind == NoteAction::END) {
                std::pair<int, int> key = { a.program, a.pitch };
                for (size_t i = 0; i < open_ev.size(); i++) {
                    if (open_ev[i].first == key) {
                        int    idx   = open_ev[i].second.first;
                        double onset = open_ev[i].second.second;
                        open_ev.erase(open_ev.begin() + (long) i);
                        res.events.push_back({ false, idx, a.pitch, a.time, a.program });
                        res.notes.push_back({ false, a.program, onset, a.time, a.pitch });
                        if (jsonl) {
                            printf("{\"type\":\"note_end\",\"index\":%d,\"time\":%.6f}\n", idx, a.time);
                            fflush(stdout);
                        }
                        break;
                    }
                }
            } else if (a.kind == NoteAction::START) {
                int idx = next_index++;
                open_ev.push_back({ { a.program, a.pitch }, { idx, a.time } });
                res.events.push_back({ true, idx, a.pitch, a.time, a.program });
                if (jsonl) {
                    printf("{\"type\":\"note_start\",\"index\":%d,\"pitch\":%d,\"time\":%.6f,\"instrument\":\"%s\"}\n",
                           idx, a.pitch, a.time, instrument_name(a.program).c_str());
                    fflush(stdout);
                }
            } else {  // DRUM_HIT: instantaneous start/end pair
                int idx = next_index++;
                res.events.push_back({ true, idx, a.pitch, a.time, DRUM_PROGRAM });
                res.events.push_back({ false, idx, a.pitch, a.time + MIN_NOTE_DUR, DRUM_PROGRAM });
                res.notes.push_back({ true, DRUM_PROGRAM, a.time, a.time + MIN_NOTE_DUR, a.pitch });
                if (jsonl) {
                    printf("{\"type\":\"note_start\",\"index\":%d,\"pitch\":%d,\"time\":%.6f,\"instrument\":\"drums\"}\n",
                           idx, a.pitch, a.time);
                    printf("{\"type\":\"note_end\",\"index\":%d,\"time\":%.6f}\n", idx, a.time + MIN_NOTE_DUR);
                    fflush(stdout);
                }
            }
        }
    };

    const int n_chunks = (n_samples + MIDI_CHUNK_SAMPLES - 1) / MIDI_CHUNK_SAMPLES;
    if (jsonl) {
        printf("{\"type\":\"progress\",\"completed\":0,\"total\":%d}\n", n_chunks);
        fflush(stdout);
    }

    for (int ci = 0; ci < n_chunks; ci++) {
        double seek      = ci * 5.0;
        double next_seek = (ci + 1 < n_chunks) ? (ci + 1) * 5.0 : -1.0;
        int    c_samples = n_samples - ci * MIDI_CHUNK_SAMPLES;
        if (c_samples > MIDI_CHUNK_SAMPLES) c_samples = MIDI_CHUNK_SAMPLES;

        // boundary settles the tracker BEFORE the tie prompt is read
        handle_actions(tracker.feed_boundary(seek, next_seek));
        std::vector<int> prompt;
        if (ci > 0) prompt = tie_section_tokens(tracker.open_keys());

        std::vector<float> prefix = compute_prefix(m, wav + (size_t) ci * MIDI_CHUNK_SAMPLES, c_samples);

        int64_t t0 = ggml_time_ms();
        int     n_tok = 0;
        decode_chunk(m, prefix, prompt, MIDI_MAX_GEN, [&](int tok) {
            n_tok++;
            handle_actions(tracker.feed(tok));
        });
        fprintf(stderr, "[ace-midi] chunk %d/%d: %d tokens, %lld ms\n",
                ci + 1, n_chunks, n_tok, (long long) (ggml_time_ms() - t0));

        if (jsonl) {
            printf("{\"type\":\"progress\",\"completed\":%d,\"total\":%d}\n", ci + 1, n_chunks);
            fflush(stdout);
        }
    }
    handle_actions(tracker.finish());
    return res;
}

// Phase 2 validation: C++ mel prefix vs oracle prefix.bin, then greedy
// token-stream parity vs tokens_ref.json.
static int run_validate_decode(MidiModel * m, const std::string & vdir) {
    const MidiConfig & c = m->cfg;

    std::vector<float> wav        = read_f32_file(vdir + "/wav.bin");
    std::vector<float> prefix_ref = read_f32_file(vdir + "/prefix.bin");

    // 1. mel-prefix parity
    std::vector<float> prefix = compute_prefix(m, wav.data(), (int) wav.size());
    if (prefix.size() != prefix_ref.size()) {
        fprintf(stderr, "[ace-midi] prefix size mismatch: cpp %zu vs ref %zu\n", prefix.size(), prefix_ref.size());
        return 1;
    }
    double mel_max_abs = 0;
    for (size_t i = 0; i < prefix.size(); i++) {
        double d = fabs((double) prefix[i] - (double) prefix_ref[i]);
        if (d > mel_max_abs) mel_max_abs = d;
    }
    printf("prefix_max_abs_diff = %.6g\n", mel_max_abs);

    // 2. greedy token-stream parity (exact-match territory in fp32)
    std::vector<int> ref_tokens;
    {
        FILE * f = fopen((vdir + "/tokens_ref.json").c_str(), "rb");
        if (!f) {
            fprintf(stderr, "[ace-midi] cannot open tokens_ref.json\n");
            return 1;
        }
        std::string j(1 << 20, 0);
        size_t n = fread(j.data(), 1, j.size() - 1, f);
        fclose(f);
        j.resize(n);
        for (const char * p = j.c_str(); *p; p++) {
            if (*p >= '0' && *p <= '9') {
                ref_tokens.push_back(atoi(p));
                while (*p >= '0' && *p <= '9') p++;
            }
        }
    }

    std::vector<int> tokens;
    int64_t t0 = ggml_time_ms();
    decode_chunk(m, prefix, {}, 256, [&](int t) { tokens.push_back(t); });
    int64_t dt = ggml_time_ms() - t0;

    int first_div = -1;
    size_t n_cmp = tokens.size() < ref_tokens.size() ? tokens.size() : ref_tokens.size();
    for (size_t i = 0; i < n_cmp; i++) {
        if (tokens[i] != ref_tokens[i]) { first_div = (int) i; break; }
    }
    printf("tokens: cpp=%zu ref=%zu first_divergence=%d (%lld ms, %.1f tok/s)\n",
           tokens.size(), ref_tokens.size(), first_div,
           (long long) dt, tokens.empty() ? 0.0 : 1000.0 * (double) tokens.size() / (double) dt);
    if (first_div >= 0) {
        printf("  at %d: cpp=%d ref=%d\n", first_div, tokens[first_div], ref_tokens[first_div]);
    }

    bool pass = first_div < 0 && tokens.size() == ref_tokens.size();
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Phase 3 validation: multi-chunk transcription (prelude forcing) — compare
// note events vs events_ref15.json and MIDI bytes vs ref15.mid.
static int run_validate_midi(MidiModel * m, const std::string & vdir) {
    std::vector<float> wav = read_f32_file(vdir + "/wav15.bin");
    TranscribeResult   res = transcribe_wav(m, wav.data(), (int) wav.size(), false);

    // parse events_ref15.json (flat array of {type,index,pitch?,time,instrument?})
    struct RefEv { bool is_start; int index; int pitch; double time; };
    std::vector<RefEv> ref;
    {
        FILE * f = fopen((vdir + "/events_ref15.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "[ace-midi] cannot open events_ref15.json\n"); return 1; }
        std::string j(1 << 22, 0);
        size_t n = fread(j.data(), 1, j.size() - 1, f);
        fclose(f);
        j.resize(n);
        const char * p = j.c_str();
        while ((p = strstr(p, "\"type\":")) != nullptr) {
            RefEv e = {};
            const char * close = strchr(p, '}');
            // json.dumps may put a space after ':' — search within the object
            const char * ts = strstr(p, "start");
            e.is_start = ts != nullptr && (!close || ts < close);
            const char * q = strstr(p, "\"index\":");
            e.index = q ? atoi(q + 8) : -1;
            q = strstr(p, "\"pitch\":");
            e.pitch = (q && close && q < close) ? atoi(q + 8) : -1;
            q = strstr(p, "\"time\":");
            e.time = q ? atof(q + 7) : -1;
            ref.push_back(e);
            p = close ? close : p + 1;
        }
    }

    int mismatches = 0;
    size_t n_cmp = res.events.size() < ref.size() ? res.events.size() : ref.size();
    for (size_t i = 0; i < n_cmp; i++) {
        const NoteEventOut & a = res.events[i];
        const RefEv & b = ref[i];
        bool ok = a.is_start == b.is_start && a.index == b.index && fabs(a.time - b.time) < 1e-5
               && (!b.is_start || a.pitch == b.pitch);
        if (!ok && mismatches++ < 5) {
            printf("  event %zu: cpp{%s idx=%d pitch=%d t=%.4f} ref{%s idx=%d pitch=%d t=%.4f}\n", i,
                   a.is_start ? "start" : "end", a.index, a.pitch, a.time,
                   b.is_start ? "start" : "end", b.index, b.pitch, b.time);
        }
    }
    printf("events: cpp=%zu ref=%zu mismatches=%d\n", res.events.size(), ref.size(), mismatches);

    // MIDI byte comparison
    std::vector<uint8_t> mid = write_midi(res.notes);
    std::vector<uint8_t> ref_mid;
    {
        FILE * f = fopen((vdir + "/ref15.mid").c_str(), "rb");
        if (!f) { fprintf(stderr, "[ace-midi] cannot open ref15.mid\n"); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        ref_mid.resize((size_t) sz);
        size_t got = fread(ref_mid.data(), 1, ref_mid.size(), f);
        fclose(f);
        (void) got;
    }
    int first_byte_div = -1;
    size_t nb = mid.size() < ref_mid.size() ? mid.size() : ref_mid.size();
    for (size_t i = 0; i < nb; i++) {
        if (mid[i] != ref_mid[i]) { first_byte_div = (int) i; break; }
    }
    printf("midi: cpp=%zu bytes ref=%zu bytes first_divergence=%d\n", mid.size(), ref_mid.size(), first_byte_div);
    if (first_byte_div >= 0) {
        printf("  at %d: cpp=%02x ref=%02x\n", first_byte_div, mid[first_byte_div], ref_mid[first_byte_div]);
    }

    bool pass = mismatches == 0 && res.events.size() == ref.size()
             && first_byte_div < 0 && mid.size() == ref_mid.size();
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Transcribe a raw f32 mono 16 kHz file to MIDI (+ optional JSONL streaming)
static int run_transcribe_raw(MidiModel * m, const std::string & wav_path,
                              const std::string & out_path, bool jsonl) {
    std::vector<float> wav = read_f32_file(wav_path);
    TranscribeResult   res = transcribe_wav(m, wav.data(), (int) wav.size(), jsonl);
    std::vector<uint8_t> mid = write_midi(res.notes);
    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot write %s\n", out_path.c_str());
        return 1;
    }
    fwrite(mid.data(), 1, mid.size(), f);
    fclose(f);
    if (jsonl) {
        printf("{\"type\":\"done\",\"notes\":%zu,\"midi_bytes\":%zu}\n", res.notes.size(), mid.size());
        fflush(stdout);
    }
    fprintf(stderr, "[ace-midi] wrote %s (%zu notes, %zu bytes)\n", out_path.c_str(), res.notes.size(), mid.size());
    return 0;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_dir, validate_dir, validate_decode_dir, validate_midi_dir;
    std::string raw_path, out_path = "out.mid";
    std::string device = "cpu";
    bool   jsonl = false;
    double tol   = 1e-3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate") && i + 1 < argc) validate_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate-decode") && i + 1 < argc) validate_decode_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate-midi") && i + 1 < argc) validate_midi_dir = argv[++i];
        else if (!strcmp(argv[i], "--transcribe-raw") && i + 1 < argc) raw_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--jsonl")) jsonl = true;
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
    }
    if (model_dir.empty()) {
        fprintf(stderr,
                "ace-midi (Phase 3) — MuScriptor GGML port\n"
                "usage: ace-midi --model <dir> <mode>\n"
                "  modes:\n"
                "    --transcribe-raw <f32.bin> [--out out.mid] [--jsonl]   raw mono 16 kHz f32 -> MIDI\n"
                "    --validate <dir>          logit parity vs oracle [--tol <x>]\n"
                "    --validate-decode <dir>   mel + greedy token parity vs oracle\n"
                "    --validate-midi <dir>     multi-chunk events + MIDI bytes vs oracle\n"
                "  --device cpu|auto|<name>  backend (default cpu — see note below)\n"
                "\n"
                "  NOTE: cpu is the default because CUDA TF32 matmul noise destabilizes\n"
                "  this model's greedy decode (chunks run to the 2000-token cap without\n"
                "  EOS). GPU support is tracked in docs/plans/muscriptor-cpp-port.md §7.\n");
        return 2;
    }

    // Default to the validated CPU backend unless explicitly overridden.
    // GGML_BACKEND env (read by backend_init) still wins if the user set it.
    if (!getenv("GGML_BACKEND") && device != "auto") {
#ifdef _WIN32
        _putenv_s("GGML_BACKEND", device == "cpu" ? "CPU" : device.c_str());
#else
        setenv("GGML_BACKEND", device == "cpu" ? "CPU" : device.c_str(), 1);
#endif
    }

    ggml_time_init();
    MidiModel m;
    if (!load_model(&m, model_dir)) return 1;

    if (!validate_dir.empty()) return run_validate(&m, validate_dir, tol);
    if (!validate_decode_dir.empty()) return run_validate_decode(&m, validate_decode_dir);
    if (!validate_midi_dir.empty()) return run_validate_midi(&m, validate_midi_dir);
    if (!raw_path.empty()) return run_transcribe_raw(&m, raw_path, out_path, jsonl);
    fprintf(stderr, "[ace-midi] no mode given\n");
    return 2;
}
