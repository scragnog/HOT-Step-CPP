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

#include "audio-io.h"
#include "backend.h"
#include "beat-this.h"
#include "ggml.h"
#include "midi-abc.h"
#include "midi-grid.h"
#include "safetensors.h"
#include "chordnet/inference.h"
#include "train/yue2-aitk-sha256.h"
#include "yourmt3/encoder.h"
#include "yyjson.h"

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

    // KV cache (batch=1): per layer K [D, max_seq, H], V [max_seq, D, H].
    // F32 on CPU (byte-exact oracle parity); F16 on GPU — the f32->f16 cpy
    // kernels are the llama.cpp-exercised path (the f32->f32 non-contiguous
    // cpy miswrites on CUDA), and f16 KV matches upstream's GPU autocast.
    ggml_context *             kv_ctx = nullptr;
    ggml_backend_buffer_t      kv_buf = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
    ggml_type                  kv_type = GGML_TYPE_F32;
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
        m->kv_type = (bp.backend != bp.cpu_backend) ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_init_params kp = { (size_t) c.num_layers * 2 * ggml_tensor_overhead() + 4096, NULL, true };
        m->kv_ctx = ggml_init(kp);
        m->kv_k.resize(c.num_layers);
        m->kv_v.resize(c.num_layers);
        for (int l = 0; l < c.num_layers; l++) {
            m->kv_k[l] = ggml_new_tensor_3d(m->kv_ctx, m->kv_type, D, m->max_seq, H);
            m->kv_v[l] = ggml_new_tensor_3d(m->kv_ctx, m->kv_type, m->max_seq, D, H);
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

// Debug probe (env ACE_MIDI_PROBE=<file-prefix>): on the first T==1 step,
// dump layer-0 intermediates so backends can be diffed op by op.
struct ProbeSlot { const char * name; ggml_tensor * t; };
static std::vector<ProbeSlot> g_probe_slots;
static bool g_probe_armed = false;

// Forward T tokens at cache position n_past; writes K/V into the cache and
// reads back the logits of the last n_out positions ([n_out][card]). n_past=0
// with T>1 is the prefill; T=1 with n_past>0 is a decode step; T>1 with
// n_past>0 is a speculative verify block. Caller advances n_past afterwards.
static void forward_tokens(MidiModel * m, const float * input, int T, int n_past, float * logits_last,
                           int n_out = 1) {
    const MidiConfig & c = m->cfg;
    const int H = c.num_heads, D = c.head_dim();
    const int n_kv = n_past + T;

    ggml_init_params ip = { ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dim, T);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);

    // A verify block needs a causal mask over a non-empty cache, which
    // diag_mask_inf gets wrong on CUDA (below). Pass it explicitly instead,
    // the soft_max_ext path llama.cpp runs every batch through.
    ggml_tensor * kq_mask = nullptr;
    if (T > 1 && n_past > 0) {
        kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, T);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_set_input(kq_mask);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);

    static int step_calls = 0;
    const bool probe = getenv("ACE_MIDI_PROBE") && T == 1 && step_calls == 0;
    if (T == 1) step_calls++;
    g_probe_slots.clear();
    auto probe_add = [&](const char * name, ggml_tensor * t) {
        if (probe) { ggml_set_output(t); g_probe_slots.push_back({ name, t }); }
    };

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
        // append current K rows: cache K layout [D, max_seq, H], slice dim1
        ggml_tensor * kc = m->kv_k[l];
        ggml_tensor * k_dst = ggml_view_3d(ctx, kc, D, T, H, kc->nb[1], kc->nb[2],
                                           (size_t) n_past * kc->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3), k_dst));

        // append current V rows into the transposed cache [max_seq, D, H].
        // Write via the llama.cpp idiom — 2-D transposed src into a 2-D
        // strided view — the one scatter pattern the CUDA cpy kernel is
        // known-good for. The earlier 3-D single-element-per-row view write
        // silently corrupted rows on CUDA (heads >= 1 got stale VRAM), which
        // wrecked every decode step; CPU was unaffected. Probe-verified.
        ggml_tensor * vc = m->kv_v[l];
        ggml_tensor * v_dst = ggml_view_2d(ctx, vc, T, D * H, vc->nb[1],
                                           (size_t) n_past * vc->nb[0]);
        ggml_tensor * v2 = ggml_cont(ctx, v);  // [dim, T]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_transpose(ctx, v2), v_dst));

        ggml_tensor * Q = ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q), D, H, T), 0, 2, 1, 3);  // [D, T, H]
        ggml_tensor * K = ggml_view_3d(ctx, kc, D, n_kv, H, kc->nb[1], kc->nb[2], 0);   // [D, n_kv, H]
        ggml_tensor * V = ggml_view_3d(ctx, vc, n_kv, D, H, vc->nb[1], vc->nb[2], 0);   // [n_kv, D, H]

        ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                       // [n_kv, T, H]
        kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) D));
        // Causal mask (bottom-right aligned). For T=1 a single query row
        // attends the whole cache, so the mask is a mathematical no-op —
        // skip it: the legacy diag_mask_inf CUDA kernel miscomputes with
        // n_past > 0, which silently corrupted every decode step on GPU.
        if (kq_mask) {
            kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f, 0.0f);
        } else {
            if (T > 1) kq = ggml_diag_mask_inf(ctx, kq, n_past);
            kq = ggml_soft_max(ctx, kq);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx, V, kq);                     // [D, T, H]
        ggml_tensor * att = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // [D, H, T]
        att = ggml_reshape_2d(ctx, att, c.dim, T);
        att = ggml_mul_mat(ctx, L.out_proj, att);

        if (l == 0) {
            probe_add("h", h);
            probe_add("qkv", qkv);
            probe_add("K", K);
            probe_add("V", V);
            probe_add("kqsoft", kq);
            probe_add("att", att);
        }

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
    if (kq_mask) {
        // query row t sees cache rows 0..n_past+t
        std::vector<float> mask((size_t) n_kv * T, 0.0f);
        for (int t = 0; t < T; t++) {
            for (int j = n_past + t + 1; j < n_kv; j++) mask[(size_t) t * n_kv + j] = -INFINITY;
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * 4);
    }
    if (ggml_backend_sched_graph_compute(m->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[ace-midi] FATAL: graph compute failed\n");
        exit(1);
    }
    ggml_backend_tensor_get(logits, logits_last, (size_t) (T - n_out) * logits->nb[1], (size_t) n_out * c.card * 4);

    if (probe) {
        const char * prefix_env = getenv("ACE_MIDI_PROBE");
        for (auto & s : g_probe_slots) {
            char path[512];
            snprintf(path, sizeof(path), "%s_%s.bin", prefix_env, s.name);
            std::vector<uint8_t> buf(ggml_nbytes(s.t));
            ggml_backend_tensor_get(s.t, buf.data(), 0, buf.size());
            FILE * f = fopen(path, "wb");
            if (f) { fwrite(buf.data(), 1, buf.size(), f); fclose(f); }
            fprintf(stderr, "[probe] %s: ne=[%lld,%lld,%lld] -> %s\n", s.name,
                    (long long) s.t->ne[0], (long long) s.t->ne[1], (long long) s.t->ne[2], path);
        }
    }

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

static int n_valid_tokens(const MidiConfig & c) { return c.card < 1393 ? c.card : 1393; }  // logits[1393:] masked upstream

// Token embeddings with sinusoidal positions pos0.. into out[n][dim]
static void embed_tokens(MidiModel * m, const int * toks, int n, int pos0, float * out) {
    const int dim = m->cfg.dim;
    for (int i = 0; i < n; i++) {
        memcpy(out + (size_t) i * dim, m->emb_host.data() + (size_t) toks[i] * dim, (size_t) dim * 4);
    }
    add_sin_pos(out, n, dim, pos0);
}

// Prefill [prefix | BOS | prompt] (sinusoidal positions from 0); returns the
// first greedy token and sets *n_past to the prefill length.
static int prefill_chunk(MidiModel * m, const std::vector<float> & prefix, const std::vector<int> & prompt,
                         int * n_past) {
    const MidiConfig & c = m->cfg;
    const int T_prefix = (int) (prefix.size() / c.dim);
    int T0 = T_prefix + 1 + (int) prompt.size();
    std::vector<float> input((size_t) T0 * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    for (size_t i = 0; i < prompt.size(); i++) {
        memcpy(input.data() + prefix.size() + (i + 1) * c.dim,
               m->emb_host.data() + (size_t) prompt[i] * c.dim, (size_t) c.dim * 4);
    }
    add_sin_pos(input.data(), T0, c.dim, 0);

    std::vector<float> logits(c.card);
    forward_tokens(m, input.data(), T0, 0, logits.data());
    *n_past = T0;
    int tok;
    greedy_argmax_range(logits.data(), n_valid_tokens(c), &tok);
    return tok;
}

template <typename TokenCb>
static void decode_chunk(MidiModel * m, const std::vector<float> & prefix,
                         const std::vector<int> & prompt, int max_gen, TokenCb cb) {
    const MidiConfig & c = m->cfg;
    const int n_valid = n_valid_tokens(c);
    for (int t : prompt) cb(t);  // teacher-forced tokens flow through the stream

    int n_past;
    int tok = prefill_chunk(m, prefix, prompt, &n_past);
    std::vector<float> logits(c.card);
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

// Greedy speculative decoding (upstream MuScriptor PR #95): the draft model
// proposes up to k tokens one at a time, the target checks them all in one
// block and keeps them up to the first disagreement plus its own token there.
// Every emitted token is the target's argmax, so the stream matches
// decode_chunk up to block-vs-scalar float rounding. Rejected K/V rows need
// no undo: n_past moves back and the next write overwrites them.
#define MIDI_MAX_DRAFT_K 4  // upstream: K=8 flipped a target argmax on MPS fp16

struct SpecStats { long proposed = 0, accepted = 0, blocks = 0; };

template <typename TokenCb>
static void decode_chunk_spec(MidiModel * m, const std::vector<float> & prefix, MidiModel * d,
                              const std::vector<float> & d_prefix, int k_max, const std::vector<int> & prompt,
                              int max_gen, SpecStats * st, TokenCb cb) {
    const MidiConfig & c = m->cfg;
    const int n_valid = n_valid_tokens(c);
    for (int t : prompt) cb(t);

    int n_past = 0, d_past = 0;
    int cur = prefill_chunk(m, prefix, prompt, &n_past);
    prefill_chunk(d, d_prefix, prompt, &d_past);

    const int limit = max_gen - (int) prompt.size();  // tokens decode_chunk would emit at most
    int emitted = 0;
    std::vector<float> d_logits(d->cfg.card), step(d->cfg.dim);
    std::vector<int>   block;
    std::vector<float> blk_in, blk_logits;

    while (true) {
        if (cur == MIDI_EOS_ID) return;
        cb(cur);
        if (++emitted >= limit) return;

        int k = k_max;
        if (k > limit - emitted) k = limit - emitted;
        if (k > m->max_seq - n_past - 1) k = m->max_seq - n_past - 1;
        if (k > d->max_seq - d_past - 1) k = d->max_seq - d_past - 1;
        if (k < 1) {
            fprintf(stderr, "[ace-midi] WARNING: KV cache full at %d tokens\n", n_past);
            return;
        }

        // draft: feed cur, then each proposal, k steps
        block.assign(1, cur);
        for (int i = 0; i < k; i++) {
            embed_tokens(d, &block[i], 1, d_past + i, step.data());
            forward_tokens(d, step.data(), 1, d_past + i, d_logits.data());
            int p;
            greedy_argmax_range(d_logits.data(), n_valid, &p);
            block.push_back(p);
        }

        // target: [cur, p1..pk] in one block -> its tokens t0..tk
        blk_in.resize((size_t) (k + 1) * c.dim);
        blk_logits.resize((size_t) (k + 1) * c.card);
        embed_tokens(m, block.data(), k + 1, n_past, blk_in.data());
        forward_tokens(m, blk_in.data(), k + 1, n_past, blk_logits.data(), k + 1);
        int a = 0, t_next = 0;
        for (int i = 0; i <= k; i++) {
            greedy_argmax_range(blk_logits.data() + (size_t) i * c.card, n_valid, &t_next);
            if (i == k || t_next != block[i + 1]) break;
            a++;
        }
        st->blocks++;
        st->proposed += k;
        st->accepted += a;

        // keep cur + a accepted proposals in both caches
        n_past += a + 1;
        if (a == k) {  // draft never saw pk
            embed_tokens(d, &block[k], 1, d_past + k, step.data());
            forward_tokens(d, step.data(), 1, d_past + k, d_logits.data());
        }
        d_past += a + 1;

        for (int i = 1; i <= a; i++) {
            if (block[i] == MIDI_EOS_ID) return;
            cb(block[i]);
            if (++emitted >= limit) return;
        }
        cur = t_next;
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

// Grid timing for write_midi (upstream notes_to_midi): note times are beats
// from tick 0, not seconds.
struct MidiTiming {
    std::vector<std::pair<int, int>> tempo;          // (tick, microseconds per beat)
    int                              beats_per_bar = 0;  // 0: no time signature
    double                           offset_s      = 0;  // where the recording's 0 s plays
};

// Serialize notes to a type-1 SMF (480 tpb, velocity 100), matching
// note_event2midi: per-program named tracks, channels 0-8,10-15 by first
// appearance (overflow shares 15), drums on 9 with synthetic offs.
// Without timing: seconds at 120 bpm, drum offs +0.01 s, tempo only in the
// meta track (the byte-exact v0.2 layout --validate-midi checks). With it:
// beats, the tempo map in every track (MuseScore ignores a note-less
// conductor track), a time signature, drum offs +1/32 beat.
static std::vector<uint8_t> write_midi(std::vector<MidiNote> notes, const MidiTiming * tm = nullptr) {
    const int    TPB = 480;
    const int    TEMPO = 500000;
    const int    VEL = 100;
    const double drum_len = tm ? 1.0 / 32 : 0.01;
    const std::vector<std::pair<int, int>> tempo = tm ? tm->tempo : std::vector<std::pair<int, int>>{ { 0, TEMPO } };

    if (!tm) {  // the grid path cleans the notes in seconds, before converting them
        validate_notes_fix(notes);
        notes = trim_overlapping(notes);
    }

    // note2note_event + drum offsets (writer adds them at time + drum_len)
    std::vector<MidiEvent> evs;
    for (auto & n : notes) {
        evs.push_back({ n.onset, n.is_drum, n.is_drum ? DRUM_PROGRAM : n.program, 1, n.pitch });
        if (!n.is_drum) evs.push_back({ n.offset, false, n.program, 0, n.pitch });
    }
    sort_midi_events(evs);
    {
        std::vector<MidiEvent> drum_offs;
        for (auto & e : evs) {
            if (e.is_drum) drum_offs.push_back({ e.time + drum_len, true, DRUM_PROGRAM, 0, e.pitch });
        }
        for (auto & d : drum_offs) evs.push_back(d);
        sort_midi_events(evs);
    }

    // per-program tracks
    struct Track { std::vector<uint8_t> bytes; int last_tick = 0; int channel = 0; size_t tempo_i = 0; };
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

    // set_tempo at an absolute tick; on a shared tick it precedes the notes
    auto put_tempo = [&](std::vector<uint8_t> & b, int & last_tick, std::pair<int, int> tc) {
        put_varlen(b, (uint32_t) (tc.first - last_tick));
        last_tick = tc.first;
        b.push_back(0xff); b.push_back(0x51); b.push_back(0x03);
        b.push_back((uint8_t) (tc.second >> 16)); b.push_back((uint8_t) (tc.second >> 8)); b.push_back((uint8_t) tc.second);
    };

    for (auto & e : evs) {
        // mido second2tick + round (banker's) — nearbyint matches
        int tick = (int) nearbyint(tm ? e.time * TPB : e.time * 1e6 / TEMPO * TPB);
        Track & t = track_for(e);
        while (tm && t.tempo_i < tempo.size() && tempo[t.tempo_i].first <= tick) put_tempo(t.bytes, t.last_tick, tempo[t.tempo_i++]);
        int delta = tick - t.last_tick;
        t.last_tick = tick;
        put_varlen(t.bytes, (uint32_t) (delta < 0 ? 0 : delta));
        t.bytes.push_back((uint8_t) ((e.velocity > 0 ? 0x90 : 0x80) | t.channel));
        t.bytes.push_back((uint8_t) e.pitch);
        t.bytes.push_back((uint8_t) (e.velocity > 0 ? VEL : 0));
    }
    for (auto & t : tracks) {
        while (tm && t.tempo_i < tempo.size()) put_tempo(t.bytes, t.last_tick, tempo[t.tempo_i++]);
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
        if (tm && tm->beats_per_bar > 0) {  // n/4, mido's defaults for the click fields
            put_varlen(meta, 0);
            meta.insert(meta.end(), { 0xff, 0x58, 0x04, (uint8_t) tm->beats_per_bar, 0x02, 0x18, 0x08 });
        }
        if (tm && tm->offset_s != 0.0) {  // upstream's marker: how far the bars delayed the music
            char text[64];
            int  n = snprintf(text, sizeof(text), "muscriptor:bar_offset=%.4f", tm->offset_s);
            put_varlen(meta, 0);
            meta.push_back(0xff); meta.push_back(0x06);
            put_varlen(meta, (uint32_t) n);
            meta.insert(meta.end(), text, text + n);
        }
        int last = 0;
        for (auto & tc : tempo) put_tempo(meta, last, tc);
        emit_track(meta);
    }
    for (auto & t : tracks) emit_track(t.bytes);
    return out;
}

// ---------------------------------------------------------------------------
// Grid-aware export (upstream utils/midi.py notes_to_midi): onset-delay shift,
// tempo map / bar alignment, optional quantize, gap fill, leading-bar trim.
// ---------------------------------------------------------------------------

struct GridOut {
    BeatGrid grid;
    Timeline timeline;
    double   offset_s;   // MIDI seconds before the recording's 0 s
};

// onto a step-beat grid, `strength` of the way; a note that rounds to nothing
// keeps one step
static std::vector<MidiNote> quantized_notes(std::vector<MidiNote> notes, double step, double strength) {
    auto snap = [&](double t) { return nearbyint(t / step) * step; };
    for (auto & n : notes) {
        double on = snap(n.onset), off = std::max(snap(n.offset), on + step);
        n.onset  += strength * (on - n.onset);
        n.offset += strength * (off - n.offset);
    }
    return trim_overlapping(notes);
}

// per instrument, hold a note to the next onset when the gap is <= max_gap beats
static void gaps_filled(std::vector<MidiNote> & notes, double max_gap) {
    const double eps = 1e-6;
    std::vector<std::vector<double>> onsets(130);  // sorted onsets per program (vocab 0-129)
    for (auto & n : notes) if (!n.is_drum) onsets[n.program].push_back(n.onset);
    for (auto & v : onsets) std::sort(v.begin(), v.end());
    for (auto & n : notes) {
        if (n.is_drum) continue;
        const auto & v  = onsets[n.program];
        auto         it = std::upper_bound(v.begin(), v.end(), n.onset + eps);
        if (it != v.end() && *it - n.offset > 0 && *it - n.offset <= max_gap + eps) n.offset = *it;
    }
}

static std::vector<uint8_t> write_midi_grid(std::vector<MidiNote> notes, const BeatGrid * detected,
                                            const GridExport & o, GridOut * out) {
    validate_notes_fix(notes);
    notes = trim_overlapping(notes);
    std::vector<double> onsets;
    for (auto & n : notes) onsets.push_back(n.onset);

    BeatGrid     g     = grid_resolve(detected, o, onsets);
    const double delay = g.onset_delay;
    double       earliest = -delay;
    for (auto & n : notes) {
        n.onset -= delay;
        n.offset -= delay;
        earliest = std::min(earliest, n.onset);
    }
    Timeline tl = grid_timeline(g, earliest);
    for (auto & n : notes) {
        n.onset  = tl.beats_at(n.onset);
        n.offset = tl.beats_at(n.offset);
    }
    if (o.quantize) {
        if (g.subdivision) notes = quantized_notes(notes, 1.0 / g.subdivision, o.strength);
        else fprintf(stderr, "[ace-midi] not quantizing: no beat subdivision to snap the notes to\n");
    }
    if (o.fill_gaps > 0) gaps_filled(notes, o.fill_gaps);
    if (o.trim_leading && !notes.empty()) {
        const int bar = g.beats_per_bar ? g.beats_per_bar : 1;
        double first = INFINITY;
        for (auto & n : notes) first = std::min(first, n.onset);
        const double empty = std::max(0.0, floor((first + 1e-6) / bar)) * bar;
        tl.shift(-empty);
        for (auto & n : notes) { n.onset -= empty; n.offset -= empty; }
    }

    MidiTiming tm;
    tm.tempo         = tl.tempo_changes(480);
    tm.beats_per_bar = g.beats_per_bar;
    tm.offset_s      = -tl.seconds_at(0.0);
    if (out) *out = { g, tl, tm.offset_s };
    return write_midi(notes, &tm);
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
// draft (optional): speculative decoding with draft_k proposals per block
static TranscribeResult transcribe_wav(MidiModel * m, const float * wav, int n_samples, bool jsonl,
                                       MidiModel * draft = nullptr, int draft_k = 0) {
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

        const float * cwav = wav + (size_t) ci * MIDI_CHUNK_SAMPLES;
        std::vector<float> prefix = compute_prefix(m, cwav, c_samples);

        int64_t t0 = ggml_time_ms();
        int     n_tok = 0;
        auto    on_tok = [&](int tok) {
            n_tok++;
            handle_actions(tracker.feed(tok));
        };
        if (draft) {
            SpecStats st;
            decode_chunk_spec(m, prefix, draft, compute_prefix(draft, cwav, c_samples), draft_k, prompt,
                              MIDI_MAX_GEN, &st, on_tok);
            fprintf(stderr, "[ace-midi] chunk %d/%d: %d tokens, %lld ms (draft accepted %ld/%ld in %ld blocks)\n",
                    ci + 1, n_chunks, n_tok, (long long) (ggml_time_ms() - t0), st.accepted, st.proposed, st.blocks);
        } else {
            decode_chunk(m, prefix, prompt, MIDI_MAX_GEN, on_tok);
            fprintf(stderr, "[ace-midi] chunk %d/%d: %d tokens, %lld ms\n",
                    ci + 1, n_chunks, n_tok, (long long) (ggml_time_ms() - t0));
        }

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

// Debug: teacher-force the oracle's reference tokens and dump every step's
// full logits. Running this on two backends and diffing the dumps separates
// "bounded numeric noise flipping near-ties" from "compute-path bug".
static int run_dump_logits(MidiModel * m, const std::string & vdir, const std::string & out_file) {
    const MidiConfig & c = m->cfg;
    std::vector<float> prefix = read_f32_file(vdir + "/prefix.bin");

    std::vector<int> ref_tokens;
    {
        FILE * f = fopen((vdir + "/tokens_ref.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "[ace-midi] cannot open tokens_ref.json\n"); return 1; }
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

    FILE * out = fopen(out_file.c_str(), "wb");
    if (!out) { fprintf(stderr, "[ace-midi] cannot write %s\n", out_file.c_str()); return 1; }

    const int T_prefix = (int) (prefix.size() / c.dim);
    int T0 = T_prefix + 1;
    std::vector<float> input((size_t) T0 * c.dim);
    memcpy(input.data(), prefix.data(), prefix.size() * 4);
    memcpy(input.data() + prefix.size(), m->emb_host.data() + (size_t) c.bos_id() * c.dim, (size_t) c.dim * 4);
    add_sin_pos(input.data(), T0, c.dim, 0);

    std::vector<float> logits(c.card);
    forward_tokens(m, input.data(), T0, 0, logits.data());
    fwrite(logits.data(), 4, (size_t) c.card, out);
    int n_past = T0;

    std::vector<float> step(c.dim);
    for (size_t i = 0; i < ref_tokens.size(); i++) {   // teacher-forced walk
        memcpy(step.data(), m->emb_host.data() + (size_t) ref_tokens[i] * c.dim, (size_t) c.dim * 4);
        add_sin_pos(step.data(), 1, c.dim, n_past);
        forward_tokens(m, step.data(), 1, n_past, logits.data());
        n_past++;
        fwrite(logits.data(), 4, (size_t) c.card, out);
    }
    fclose(out);
    fprintf(stderr, "[ace-midi] dumped %zu x %d logits to %s\n", ref_tokens.size() + 1, c.card, out_file.c_str());
    return 0;
}

// Load WAV/MP3 (any rate, mono/stereo) as mono float32 at its own rate
// (channel mean, as the Python pipeline).
static std::vector<float> load_audio_mono(const std::string & path, int * sr) {
    int T = 0;
    float * planar = audio_read(path.c_str(), &T, sr);  // [L: T][R: T]
    if (!planar || T <= 0) {
        fprintf(stderr, "[ace-midi] cannot read audio: %s\n", path.c_str());
        exit(1);
    }
    std::vector<float> mono((size_t) T);
    for (int i = 0; i < T; i++) mono[i] = 0.5f * (planar[i] + planar[(size_t) T + i]);
    free(planar);
    return mono;
}

static std::vector<float> resample_16k(std::vector<float> mono, int sr) {
    const size_t n_in = mono.size();
    if (sr != MIDI_SAMPLE_RATE) {
        int n_out = 0;
        float * r = audio_resample(mono.data(), (int) n_in, sr, MIDI_SAMPLE_RATE, 1, &n_out);
        if (!r || n_out <= 0) {
            fprintf(stderr, "[ace-midi] resample %d -> %d failed\n", sr, MIDI_SAMPLE_RATE);
            exit(1);
        }
        mono.assign(r, r + n_out);
        free(r);
    }
    fprintf(stderr, "[ace-midi] audio: %.1f s @ %d Hz -> %zu samples @ 16 kHz\n",
            (double) n_in / sr, sr, mono.size());
    return mono;
}

static bool write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[ace-midi] cannot write %s\n", path.c_str());
        return false;
    }
    fwrite(data, 1, n, f);
    fclose(f);
    return true;
}

// beats.json: the tracked beats (re-export reruns the grid from them) and the
// written grid's beats and bar lines in seconds of the audio, for the piano
// roll. audio_shift_s maps MIDI seconds (from tick 0) back to the audio.
static bool write_beats_json(const std::string & path, const BeatTrack & tr, double duration, const GridOut & go) {
    std::string j = "{\"tracked_beats\":[";
    char        buf[64];
    auto num = [&](double v, bool comma) { snprintf(buf, sizeof(buf), comma ? ",%.6f" : "%.6f", v); j += buf; };
    for (size_t i = 0; i < tr.beats.size(); i++) num(tr.beats[i], i > 0);
    j += "],\"tracked_downbeats\":[";
    for (size_t i = 0; i < tr.downbeats.size(); i++) num(tr.downbeats[i], i > 0);
    const BeatGrid & g = go.grid;
    const double delay = g.onset_delay;
    snprintf(buf, sizeof(buf), "],\"duration\":%.6f,\"grid\":{", duration);
    j += buf;
    snprintf(buf, sizeof(buf), "\"bpm\":%.4f,", g.bpm); j += buf;
    snprintf(buf, sizeof(buf), "\"beats_per_bar\":%d,", g.beats_per_bar); j += buf;
    snprintf(buf, sizeof(buf), "\"steady\":%s,", g.steady ? "true" : "false"); j += buf;
    snprintf(buf, sizeof(buf), "\"onset_delay\":%.6f,", delay); j += buf;
    snprintf(buf, sizeof(buf), "\"subdivision\":%d,", g.subdivision); j += buf;
    snprintf(buf, sizeof(buf), "\"offset_s\":%.6f,", go.offset_s); j += buf;
    snprintf(buf, sizeof(buf), "\"audio_shift_s\":%.6f,", delay - go.offset_s); j += buf;
    // notes were moved by -delay before the timeline, so audio = seconds_at(p) + delay
    std::string beats = "\"beats\":[", bars = "\"bars\":[";
    const int   p0 = (int) ceil(go.timeline.beats_at(-delay)), p1 = (int) floor(go.timeline.beats_at(duration - delay));
    for (int p = p0; p <= p1; p++) {
        snprintf(buf, sizeof(buf), "%.4f", go.timeline.seconds_at(p) + delay);
        beats += (p > p0 ? "," : "") + std::string(buf);
        if (g.beats_per_bar && grid_pymod(p, g.beats_per_bar) == 0) bars += (bars.back() == '[' ? "" : ",") + std::string(buf);
    }
    j += beats + "]," + bars + "]}}\n";
    return write_file(path, j.data(), j.size());
}

static bool run_beats(const std::string & model_dir, const std::vector<float> & mono, int sr, BeatTrack * tr) {
    BeatThis bt;
    if (!bt_load(&bt, model_dir + "/model.safetensors")) {
        fprintf(stderr, "[ace-midi] cannot load the beat model from %s\n", model_dir.c_str());
        return false;
    }
    int64_t t0 = ggml_time_ms();
    *tr = bt_track(&bt, mono.data(), (int) mono.size(), sr);
    bt_free(&bt);
    fprintf(stderr, "[ace-midi] beats: %zu beats, %zu downbeats (%lld ms)\n", tr->beats.size(), tr->downbeats.size(),
            (long long) (ggml_time_ms() - t0));
    return true;
}

struct AbcOpts {
    std::string out;               // YuE2 ABC lead sheet path; empty = none
    int         ins_program = -1;  // the Ins voice's instrument; -1 = busiest melodic one
    bool        melody_only = false;
};

// The transcription as a YuE2 lead sheet on the same grid as the MIDI (midi-abc.h)
static void write_abc(const std::vector<MidiNote> & notes, const GridOut & go, double duration, const AbcOpts & ao) {
    std::vector<AbcNote> an;
    for (auto & n : notes) an.push_back({ n.onset, n.offset, n.pitch, n.program, n.is_drum });
    AbcResult r = midi_to_yue2_abc(an, go.grid, duration, ao.ins_program, ao.melody_only);
    if (r.ok) write_file(ao.out, r.abc.data(), r.abc.size());
    fprintf(stderr, "[ace-midi] abc: %s (vocal %d notes, ins %s %d notes, key %s, %d chords%s%s)\n",
            r.ok ? "ok" : r.error.c_str(), r.vocal_notes, r.ins_program < 0 ? "none" : instrument_name(r.ins_program).c_str(), r.ins_notes,
            r.key.c_str(), r.chords, r.repaired.empty() ? "" : ", repaired: ", r.repaired.c_str());
    std::string err;
    for (char c : r.error) err += (c == '"' || c == '\\') ? std::string("\\") + c : (c == '\n' ? std::string(" ") : std::string(1, c));
    printf("{\"type\":\"abc\",\"ok\":%s,\"vocal_notes\":%d,\"ins_notes\":%d,\"ins\":\"%s\",\"key\":\"%s\",\"chords\":%d,\"error\":\"%s\"}\n",
           r.ok ? "true" : "false", r.vocal_notes, r.ins_notes, r.ins_program < 0 ? "none" : instrument_name(r.ins_program).c_str(), r.key.c_str(),
           r.chords, err.c_str());
    fflush(stdout);
}

// Notes -> MIDI on the detected grid, plus beats.json (and the ABC lead
// sheet). A failed detection writes the placeholder grid (120 BPM, no
// meter), as upstream's best-effort.
static std::vector<uint8_t> grid_midi(const std::vector<MidiNote> & notes, const BeatTrack & tr, double duration,
                                      const GridExport & opts, const std::string & beats_out,
                                      const AbcOpts & ao = {}) {
    BeatGrid g;
    bool     ok = grid_detect(tr.beats, tr.downbeats, &g);
    if (!ok) fprintf(stderr, "[ace-midi] only %zu beats detected, need %d; writing 120 BPM\n", tr.beats.size(), GRID_MIN_BEATS);
    GridOut go;
    std::vector<uint8_t> mid = write_midi_grid(notes, ok ? &g : nullptr, opts, &go);
    if (!beats_out.empty()) write_beats_json(beats_out, tr, duration, go);
    if (!ao.out.empty()) write_abc(notes, go, duration, ao);
    return mid;
}

// Transcribe an audio file (WAV/MP3, or raw f32 mono 16 kHz with is_raw).
// With a beat model the MIDI gets tempo, meter and bar lines.
static int run_transcribe(MidiModel * m, const std::string & audio_path, const std::string & out_path, bool jsonl,
                          bool is_raw, const std::string & beat_model, const GridExport & opts,
                          const std::string & beats_out, const AbcOpts & ao, MidiModel * draft, int draft_k) {
    std::vector<float> wav;
    BeatTrack          tr;
    if (is_raw) {
        wav = read_f32_file(audio_path);
        if (!beat_model.empty() && !run_beats(beat_model, wav, MIDI_SAMPLE_RATE, &tr)) return 1;
    } else {
        int                sr   = 0;
        std::vector<float> mono = load_audio_mono(audio_path, &sr);
        if (!beat_model.empty() && !run_beats(beat_model, mono, sr, &tr)) return 1;
        wav = resample_16k(std::move(mono), sr);
    }
    TranscribeResult     res = transcribe_wav(m, wav.data(), (int) wav.size(), jsonl, draft, draft_k);
    std::vector<uint8_t> mid = beat_model.empty()
        ? write_midi(res.notes)
        : grid_midi(res.notes, tr, (double) wav.size() / MIDI_SAMPLE_RATE, opts, beats_out, ao);
    if (!write_file(out_path, mid.data(), mid.size())) return 1;
    if (jsonl) {
        printf("{\"type\":\"done\",\"notes\":%zu,\"midi_bytes\":%zu}\n", res.notes.size(), mid.size());
        fflush(stdout);
    }
    fprintf(stderr, "[ace-midi] wrote %s (%zu notes, %zu bytes)\n", out_path.c_str(), res.notes.size(), mid.size());
    return 0;
}

// The numbers in a JSON array `"key":[...]` (our own beats.json only)
static std::vector<double> json_numbers(const std::string & j, const char * key) {
    std::vector<double> v;
    size_t p = j.find(std::string("\"") + key + "\":[");
    if (p == std::string::npos) return v;
    const char * s = j.c_str() + p + strlen(key) + 4;
    while (*s && *s != ']') {
        char * end;
        double x = strtod(s, &end);
        if (end == s) { s++; continue; }
        v.push_back(x);
        s = end;
    }
    return v;
}

static std::string read_text(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char        buf[65536];
    size_t      n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

// Re-export: the streamed note events (--jsonl output) + beats.json -> MIDI
// with new export options. No transcription model needed.
static int run_export(const std::string & events_path, const std::string & beats_in, const std::string & out_path,
                      const GridExport & opts, const std::string & beats_out, const AbcOpts & ao) {
    std::string ev = read_text(events_path), bj = read_text(beats_in);
    if (ev.empty() || bj.empty()) {
        fprintf(stderr, "[ace-midi] cannot read %s or %s\n", events_path.c_str(), beats_in.c_str());
        return 1;
    }
    auto field = [](const std::string & line, const char * key, std::string * str) -> double {
        size_t p = line.find(std::string("\"") + key + "\":");
        if (p == std::string::npos) return -1;
        const char * s = line.c_str() + p + strlen(key) + 3;
        if (str) {
            const char * q = strchr(s + 1, '"');
            *str = q ? std::string(s + 1, q) : "";
            return 0;
        }
        return atof(s);
    };
    auto program_for = [](const std::string & name, bool * drum) {
        *drum = name == "drums";
        if (*drum) return DRUM_PROGRAM;
        for (int p = 0; p < DRUM_PROGRAM; p++) {
            const char * n = instrument_for_program(p);
            if (n && name == n) return p;
        }
        return name.rfind("program_", 0) == 0 ? atoi(name.c_str() + 8) : 0;
    };

    // pair note_start / note_end by index, as transcribe_wav assembles notes
    std::vector<MidiNote> notes;
    std::vector<std::pair<int, MidiNote>> open;
    size_t pos = 0;
    while (pos < ev.size()) {
        size_t      nl   = ev.find('\n', pos);
        std::string line = ev.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos              = nl == std::string::npos ? ev.size() : nl + 1;
        std::string type, inst;
        field(line, "type", &type);
        const int index = (int) field(line, "index", nullptr);
        if (type == "note_start") {
            field(line, "instrument", &inst);
            MidiNote n;
            n.program = program_for(inst, &n.is_drum);
            n.pitch   = (int) field(line, "pitch", nullptr);
            n.onset = n.offset = field(line, "time", nullptr);
            open.push_back({ index, n });
        } else if (type == "note_end") {
            for (size_t i = 0; i < open.size(); i++) {
                if (open[i].first != index) continue;
                open[i].second.offset = field(line, "time", nullptr);
                notes.push_back(open[i].second);
                open.erase(open.begin() + (long) i);
                break;
            }
        }
    }
    BeatTrack tr;
    tr.beats     = json_numbers(bj, "tracked_beats");
    tr.downbeats = json_numbers(bj, "tracked_downbeats");
    const size_t dp       = bj.find("\"duration\":");
    const double duration = dp == std::string::npos ? 0.0 : atof(bj.c_str() + dp + 11);

    std::vector<uint8_t> mid = grid_midi(notes, tr, duration, opts, beats_out, ao);
    if (!write_file(out_path, mid.data(), mid.size())) return 1;
    fprintf(stderr, "[ace-midi] exported %s (%zu notes, %zu bytes)\n", out_path.c_str(), notes.size(), mid.size());
    return 0;
}

// ---------------------------------------------------------------------------
// ChordNet chords (lead-sheet source): --chords and --validate-chords.
// CPU only; needs no MuScriptor model. Engine code: src/chordnet/.
// ---------------------------------------------------------------------------

static int chord_threads() { return backend_cpu_n_threads(); }  // GGML_N_THREADS overrides

static bool chord_load(const std::string & model, const std::string & vocab, chordnet::Weights & w,
                       std::vector<chordnet::Template> & tpl) {
    if (model.empty() || vocab.empty()) {
        fprintf(stderr, "[ace-midi] chord modes need --chord-model <gguf> and --chord-vocab <tsv>\n");
        return false;
    }
    return chordnet::load_weights(w, model) && chordnet::load_templates(vocab, tpl);
}

// Segments in seconds, no beat rounding: [[label, onset, offset], ...]
static int run_chords(const std::string & model, const std::string & vocab, const std::string & audio,
                      const std::string & out_path) {
    chordnet::Weights               w;
    std::vector<chordnet::Template> tpl;
    if (!chord_load(model, vocab, w, tpl)) return 1;
    int                sr   = 0;
    std::vector<float> mono = load_audio_mono(audio, &sr);
    chordnet::Result   r;
    const int threads = chord_threads();
    chordnet::transcribe(w, tpl, mono, sr, threads, r);
    chordnet::free_weights(w);
    char buf[512];
    snprintf(buf, sizeof buf, "{\"tuning\": %.17g, \"frames\": %d, \"segments\": [", r.tuning, r.T);
    std::string j = buf;
    for (size_t i = 0; i < r.segs.size(); i++) {
        snprintf(buf, sizeof buf, "%s[\"%s\", %.17g, %.17g]", i ? ", " : "", r.segs[i].label.c_str(), r.segs[i].onset,
                 r.segs[i].offset);
        j += buf;
    }
    j += "]}\n";
    if (!write_file(out_path, j.data(), j.size())) return 1;
    fprintf(stderr, "[ace-midi] chords: %zu segments, %.1f s audio, %d threads: features %.0f ms, model %.0f ms, decode %.0f ms\n",
            r.segs.size(), (double) mono.size() / sr, threads, r.ms_features, r.ms_model, r.ms_decode);
    return 0;
}

// max |a - b|; a size mismatch or any non-finite value fails (INFINITY)
static double parity_err(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    }
    return m;
}

static int g_parity_fail = 0;

// One PARITY row: chained from the source audio, fed from the stage's captured input.
static void parity_row(const std::string & cap, const std::string & point, double chained, double fed, double gate,
                       const char * unit = "") {
    const bool ok = chained < gate && (std::isnan(fed) || fed < gate);
    if (!ok) g_parity_fail++;
    char f[32];
    if (std::isnan(fed)) snprintf(f, sizeof f, "-");
    else snprintf(f, sizeof f, "%.3g%s", fed, unit);
    printf("PARITY %-16s %-20s chained=%.3g%s fed=%s gate<%g%s %s\n", cap.c_str(), point.c_str(), chained, unit, f, gate,
           unit, ok ? "PASS" : "FAIL");
}

static std::vector<std::string> json_strings(yyjson_val * arr) {
    std::vector<std::string> out;
    size_t       i, n;
    yyjson_val * v;
    yyjson_arr_foreach(arr, i, n, v) out.push_back(yyjson_get_str(v) ? yyjson_get_str(v) : "");
    return out;
}

// Frames whose label differs; the first one's Viterbi score margin (this
// port's label minus the reference label, in this port's scores) to *detail.
static double label_miss(const std::vector<std::string> & mine, const std::vector<std::string> & ref,
                         const chordnet::Decoded & d, std::string * detail) {
    if (mine.size() != ref.size()) return INFINITY;
    int miss = 0;
    for (size_t t = 0; t < mine.size(); t++) {
        if (mine[t] == ref[t]) continue;
        if (!miss++) {
            const size_t C = d.names.size();
            size_t       a = 0, b = 0;
            for (size_t k = 0; k < C; k++) {
                if (d.names[k] == mine[t]) a = k;
                if (d.names[k] == ref[t]) b = k;
            }
            char buf[256];
            snprintf(buf, sizeof buf, "first miss frame %zu: %s over %s, score margin %.6g", t, mine[t].c_str(),
                     ref[t].c_str(), (double) d.score[t * C + a] - d.score[t * C + b]);
            *detail = buf;
        }
    }
    return miss;
}

// Labels must match one for one (else INFINITY); then the max onset/offset error in seconds.
static double segment_err(const std::vector<chordnet::Segment> & a, yyjson_val * ref) {
    if (yyjson_arr_size(ref) != a.size()) return INFINITY;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        yyjson_val * s   = yyjson_arr_get(ref, i);
        const char * lab = yyjson_get_str(yyjson_arr_get(s, 0));
        if (!lab || a[i].label != lab) return INFINITY;
        m = std::max({ m, std::fabs(a[i].onset - yyjson_get_num(yyjson_arr_get(s, 1))),
                       std::fabs(a[i].offset - yyjson_get_num(yyjson_arr_get(s, 2))) });
    }
    return m;
}

static int run_validate_chords(const std::string & model, const std::string & vocab, std::string dir) {
    using namespace chordnet;
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
    const std::string     cap = dir.substr(dir.find_last_of("/\\") + 1);
    Weights               w;
    std::vector<Template> tpl;
    if (!chord_load(model, vocab, w, tpl)) return 1;
    const int threads = chord_threads();

    yyjson_doc * man = yyjson_read_file((dir + "/manifest.json").c_str(), 0, nullptr, nullptr);
    yyjson_doc * run = yyjson_read_file((dir + "/run.json").c_str(), 0, nullptr, nullptr);
    yyjson_doc * lab = yyjson_read_file((dir + "/chord-frame-labels.json").c_str(), 0, nullptr, nullptr);
    yyjson_doc * seg = yyjson_read_file((dir + "/chords-seconds.json").c_str(), 0, nullptr, nullptr);
    yyjson_val * tun = man ? yyjson_obj_get(yyjson_doc_get_root(man), "tuning_bins") : nullptr;
    std::string  audio;
    if (run) {
        yyjson_val * cmd = yyjson_obj_get(yyjson_doc_get_root(run), "command");
        for (size_t i = 0; i + 1 < yyjson_arr_size(cmd); i++) {
            const char * a = yyjson_get_str(yyjson_arr_get(cmd, i));
            const char * b = yyjson_get_str(yyjson_arr_get(cmd, i + 1));
            if (a && b && !strcmp(a, "--audio")) audio = b;
        }
    }
    auto done = [&]() {
        yyjson_doc_free(man);
        yyjson_doc_free(run);
        yyjson_doc_free(lab);
        yyjson_doc_free(seg);
        free_weights(w);
    };
    // A capture without chord tensors is skipped only when its manifest records
    // the upstream rejection that stopped it; anything else missing fails.
    yyjson_doc * tns = yyjson_read_file((dir + "/tensors.json").c_str(), 0, nullptr, nullptr);
    bool has_chord = false;
    if (tns) {
        size_t       i, n;
        yyjson_val * k, * v;
        yyjson_obj_foreach(yyjson_doc_get_root(tns), i, n, k, v) has_chord |= !strncmp(yyjson_get_str(k), "chord.", 6);
        yyjson_doc_free(tns);
    }
    yyjson_val * mroot  = man ? yyjson_doc_get_root(man) : nullptr;
    const char * status = yyjson_get_str(yyjson_obj_get(mroot, "status"));
    yyjson_val * skips  = yyjson_obj_get(mroot, "skips");
    if (man && tns && !has_chord && status && !strcmp(status, "captured-with-upstream-rejection") && yyjson_arr_size(skips) > 0) {
        printf("PARITY %-16s upstream rejected it before ChordNet (%s): skipped\n", cap.c_str(),
               yyjson_get_str(yyjson_arr_get(skips, 0)) ? yyjson_get_str(yyjson_arr_get(skips, 0)) : "?");
        done();
        return 0;
    }
    std::string missing;
    if (!man) missing += " manifest.json";
    if (!tns) missing += " tensors.json";
    if (!has_chord) missing += " chord.* tensors";
    if (!tun || !yyjson_is_num(tun)) missing += " tuning_bins";
    if (!lab) missing += " chord-frame-labels.json";
    if (!seg) missing += " chords-seconds.json";
    if (audio.empty()) missing += " run.json --audio";
    if (!missing.empty()) {
        printf("PARITY %-16s FAIL: capture incomplete or unreadable:%s\n", cap.c_str(), missing.c_str());
        done();
        return 1;
    }
    const double ref_tuning = yyjson_get_num(tun);
    auto         rd = [&](const std::string & name) { return read_f32_file(dir + "/chord." + name + ".f32"); };
    const double NaN = std::nan("");
    const double G   = 1e-3;

    // chained: the source audio through every stage
    int                sr   = 0;
    std::vector<float> mono = load_audio_mono(audio, &sr);
    Result             c;
    transcribe(w, tpl, mono, sr, threads, c);

    const std::vector<float> wav = rd("waveform-22050"), hyb = rd("hybrid-cqt"), mag = rd("cqt-magnitude");
    const std::vector<float> x0  = rd("fold0.crop-18-270");
    parity_row(cap, "waveform-22050", parity_err(c.wav, wav), NaN, G);
    parity_row(cap, "tuning", std::fabs(c.tuning - ref_tuning), std::fabs(estimate_tuning(wav) - ref_tuning), G);
    int                      Tf      = 0;
    const std::vector<float> hyb_fed = hybrid_cqt(wav, ref_tuning, &Tf);
    parity_row(cap, "hybrid-cqt", parity_err(c.mag, hyb), parity_err(hyb_fed, hyb), G);
    std::vector<float> mag_fed(hyb.size());
    for (size_t i = 0; i < hyb.size(); i++) mag_fed[i] = std::fabs(hyb[i]);
    parity_row(cap, "cqt-magnitude", parity_err(c.mag, mag), parity_err(mag_fed, mag), G);
    const int T = (int) (mag.size() / NBINS);
    parity_row(cap, "crop-18-270", parity_err(c.x, x0), parity_err(crop(mag, T), x0), G);

    ggml_backend_t be = cpu_backend(threads);
    for (int k = 0; k < NFOLD; k++) {
        const std::string        f = "fold" + std::to_string(k);
        const std::vector<float> x = rd(f + ".crop-18-270"), cnn = rd(f + ".cnn"), lstm = rd(f + ".lstm");
        FoldOut                  fed, tmp;
        std::vector<float>       proj[2];
        run_graph(be, w.fold[k], x.data(), T, false, fed, proj);
        parity_row(cap, f + ".cnn", parity_err(c.fold[k].cnn, cnn), parity_err(fed.cnn, cnn), G);
        run_graph(be, w.fold[k], cnn.data(), T, true, tmp, proj);
        run_lstm(w.fold[k], proj, T, fed.lstm);
        parity_row(cap, f + ".lstm", parity_err(c.fold[k].lstm, lstm), parity_err(fed.lstm, lstm), G);
        run_heads(w.fold[k], lstm, T, fed);
        for (int h = 0; h < NHEAD; h++) {
            const std::string p = f + ".head" + std::to_string(h);
            const std::vector<float> ref = rd(p);
            parity_row(cap, p, parity_err(c.fold[k].head[h], ref), parity_err(fed.head[h], ref), G);
        }
    }
    ggml_backend_free(be);

    FoldOut ref_heads[NFOLD];
    for (int k = 0; k < NFOLD; k++)
        for (int h = 0; h < NHEAD; h++) ref_heads[k].head[h] = rd("fold" + std::to_string(k) + ".head" + std::to_string(h));
    std::vector<float> mean_fed[NHEAD], mean_ref[NHEAD];
    fold_mean(ref_heads, T, mean_fed);
    for (int h = 0; h < NHEAD; h++) {
        mean_ref[h] = rd("mean.head" + std::to_string(h));
        parity_row(cap, "mean.head" + std::to_string(h), parity_err(c.mean[h], mean_ref[h]), parity_err(mean_fed[h], mean_ref[h]), G);
    }

    const std::vector<std::string> ref_labels = json_strings(yyjson_doc_get_root(lab));
    const Decoded                  dfed       = viterbi(tpl, mean_ref, T);
    std::string                    dc, df;
    const double                   mc = label_miss(frame_labels(c.dec), ref_labels, c.dec, &dc);
    const double                   mf = label_miss(frame_labels(dfed), ref_labels, dfed, &df);
    parity_row(cap, "viterbi-labels", mc, mf, 0.5, " frames");
    if (!dc.empty()) printf("       chained %s\n", dc.c_str());
    if (!df.empty()) printf("       fed %s\n", df.c_str());
    yyjson_val * sref = yyjson_doc_get_root(seg);
    parity_row(cap, "segments", segment_err(c.segs, sref), segment_err(segments(ref_labels), sref), 1e-6, " s");
    printf("TIMING %-16s %.2f s audio, %d threads: features %.0f ms, model %.0f ms, decode %.0f ms\n", cap.c_str(),
           (double) mono.size() / sr, threads, c.ms_features, c.ms_model, c.ms_decode);
    done();
    printf("chord parity %s: %d failing row(s)\n", g_parity_fail ? "FAIL" : "PASS", g_parity_fail);
    return g_parity_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------
// YourMT3+ front half (lead-sheet source): --validate-yourmt3 and
// --yourmt3-encode. Audio to encoder states only; the decoder comes in a
// later stage. Backend from --device (auto = best). Engine code: src/yourmt3/.
// ---------------------------------------------------------------------------

static double span_err(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    }
    return m;
}

static void yourmt3_timing(const char * label, const yourmt3::FrontHalf & f, double seconds, const char * be) {
    printf("TIMING %-16s %.2f s audio, %d segments, %s: resample %.0f ms, spectrogram %.0f ms, pre-encoder %.0f ms, "
           "encoder %.0f ms\n",
           label, seconds, f.n_seg, be, f.ms_resample, f.ms_spec, f.ms_pre, f.ms_enc);
}

// Audio through the encoder; prints placement and stage times, writes nothing.
static int run_yourmt3_encode(const std::string & model, const std::string & audio) {
    const double t0 = yourmt3::now_ms();
    yourmt3::Encoder e;
    if (!yourmt3::encoder_init(e, model)) return 1;
    const double t_load = yourmt3::now_ms() - t0;
    int                sr   = 0;
    std::vector<float> mono = load_audio_mono(audio, &sr);
    const double       t1   = yourmt3::now_ms();
    yourmt3::FrontHalf f;
    yourmt3::front_half(e, mono, sr, f, false);
    printf("TIMING load %.0f ms (GGUF read, upload, backend init); front half %.0f ms\n", t_load, yourmt3::now_ms() - t1);
    yourmt3_timing("encode", f, (double) mono.size() / sr, ggml_backend_name(e.bp.backend));
    yourmt3::encoder_free(e);
    return 0;
}

// One capture file, checked against tensors.json (byte count and sha256).
static std::vector<float> capture_tensor(const std::string & dir, yyjson_val * tensors, const std::string & key,
                                         std::string * missing) {
    yyjson_val * t     = yyjson_obj_get(tensors, key.c_str());
    const char * file  = yyjson_get_str(yyjson_obj_get(t, "file"));
    const char * sha   = yyjson_get_str(yyjson_obj_get(t, "sha256"));
    const size_t bytes = (size_t) yyjson_get_uint(yyjson_obj_get(t, "bytes"));
    std::vector<float> v;
    if (!file || !sha || !bytes || bytes % 4) {
        *missing += " " + key + " (tensors.json entry)";
        return v;
    }
    FILE * fp = fopen((dir + "/" + file).c_str(), "rb");
    if (fp) {
        v.resize(bytes / 4);
        const size_t got = fread(v.data(), 1, bytes, fp);
        const bool   eof = fgetc(fp) == EOF;
        fclose(fp);
        if (got != bytes || !eof || yue2_aitk::sha256::bytes(v.data(), bytes).hex() != sha) {
            *missing += std::string(" ") + file + " (size or sha256 differs)";
            v.clear();
        }
    } else {
        *missing += std::string(" ") + file;
    }
    return v;
}

// dump (optional): this run's chained and fed outputs per batch, for offline checks.
static int run_validate_yourmt3(const std::string & model, std::string dir, const std::string & dump) {
    using namespace yourmt3;
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
    const std::string cap = dir.substr(dir.find_last_of("/\\") + 1);
    yyjson_doc *      man = yyjson_read_file((dir + "/manifest.json").c_str(), 0, nullptr, nullptr);
    yyjson_doc *      run = yyjson_read_file((dir + "/run.json").c_str(), 0, nullptr, nullptr);
    yyjson_doc *      tns = yyjson_read_file((dir + "/tensors.json").c_str(), 0, nullptr, nullptr);
    yyjson_val *      troot = tns ? yyjson_doc_get_root(tns) : nullptr;
    auto              done  = [&]() {
        yyjson_doc_free(man);
        yyjson_doc_free(run);
        yyjson_doc_free(tns);
    };
    std::string audio;
    if (run) {
        yyjson_val * cmd = yyjson_obj_get(yyjson_doc_get_root(run), "command");
        for (size_t i = 0; i + 1 < yyjson_arr_size(cmd); i++) {
            const char * a = yyjson_get_str(yyjson_arr_get(cmd, i));
            const char * b = yyjson_get_str(yyjson_arr_get(cmd, i + 1));
            if (a && b && !strcmp(a, "--audio")) audio = b;
        }
    }
    // Batches are every index any mt3.b<k>.* key names; all of 0..max must be complete.
    int  n_batch = 0;
    bool any_mt3 = false;
    if (troot) {
        size_t       i, n;
        yyjson_val * key, * val;
        yyjson_obj_foreach(troot, i, n, key, val) {
            const char * ks = yyjson_get_str(key);
            if (!ks || strncmp(ks, "mt3.", 4)) continue;
            any_mt3 = true;
            if (!strncmp(ks, "mt3.b", 5)) n_batch = std::max(n_batch, atoi(ks + 5) + 1);
        }
    }
    // Skipped only when upstream rejected the input before YourMT3 ran: a
    // recorded rejection, no YourMT3 timing and no mt3.* entry at all. The
    // reference rejections (no BPM, no melody) come after YourMT3 and fail here.
    yyjson_val * mroot   = man ? yyjson_doc_get_root(man) : nullptr;
    const char * status  = yyjson_get_str(yyjson_obj_get(mroot, "status"));
    yyjson_val * skips   = yyjson_obj_get(mroot, "skips");
    const bool   mt3_ran = yyjson_obj_get(yyjson_obj_get(mroot, "timings_seconds"), "yourmt3_load_and_inference") != nullptr;
    if (man && tns && !any_mt3 && !mt3_ran && status && !strcmp(status, "captured-with-upstream-rejection") &&
        yyjson_arr_size(skips) > 0) {
        printf("PARITY %-16s upstream rejected it before YourMT3 (%s): skipped\n", cap.c_str(),
               yyjson_get_str(yyjson_arr_get(skips, 0)) ? yyjson_get_str(yyjson_arr_get(skips, 0)) : "?");
        done();
        return 0;
    }
    std::string missing;
    if (!man) missing += " manifest.json";
    if (!tns) missing += " tensors.json";
    if (!n_batch) missing += " mt3.b0.* tensors";
    if (audio.empty()) missing += " run.json --audio";
    static const char * POINT[4] = { "waveform-padded", "linear-spectrogram", "pre-encoder", "encoder" };
    std::vector<std::vector<float>> ref((size_t) n_batch * 4);
    for (int k = 0; k < n_batch; k++)
        for (int p = 0; p < 4; p++)
            ref[k * 4 + p] = capture_tensor(dir, troot, "mt3.b" + std::to_string(k) + "." + POINT[p], &missing);
    if (!missing.empty()) {
        printf("PARITY %-16s FAIL: capture incomplete or unreadable:%s\n", cap.c_str(), missing.c_str());
        done();
        return 1;
    }

    Encoder e;
    if (!encoder_init(e, model)) {
        done();
        return 1;
    }
    // chained: the source audio through every stage
    int                sr   = 0;
    std::vector<float> mono = load_audio_mono(audio, &sr);
    FrontHalf          c;
    front_half(e, mono, sr, c);
    // The capture must hold every batch of this audio, each with its exact segment count.
    const int want_batches = (c.n_seg + BATCH - 1) / BATCH;
    if (n_batch != want_batches) {
        printf("PARITY %-16s FAIL: capture has %d batch(es), the source audio makes %d (%d segments)\n", cap.c_str(),
               n_batch, want_batches, c.n_seg);
        g_parity_fail++;
    }
    const double NaN = std::nan("");
    const double G   = 1e-3;
    const size_t per[4] = { (size_t) SEG, (size_t) NFRAME * NBIN, (size_t) NFRAME * 128 * 128, (size_t) NFRAME * N_LAT * D_ENC };
    const std::vector<float> * chained[4] = { &c.wav, &c.spec, &c.pre, &c.enc };
    for (int k = 0; k < n_batch; k++) {
        const int B = std::min(BATCH, c.n_seg - k * BATCH);
        bool shape_ok = B > 0;
        for (int p = 0; p < 4; p++) shape_ok &= ref[k * 4 + p].size() == (size_t) std::max(B, 0) * per[p];
        if (!shape_ok) {
            printf("PARITY %-16s FAIL: batch %d is not the %d segment(s) this audio gives it\n", cap.c_str(), k,
                   std::max(B, 0));
            g_parity_fail++;
            continue;
        }
        // fed: each stage from the captured input of that stage
        const std::vector<float> spec = spectrogram(ref[k * 4].data(), B, e.w.window.data());
        const std::vector<float> pre  = pre_encode(e, ref[k * 4 + 1].data(), B);
        const std::vector<float> enc  = encode(e, ref[k * 4 + 2].data(), B);
        const std::vector<float> * fed[4] = { nullptr, &spec, &pre, &enc };
        if (!dump.empty())
            for (int p = 1; p < 4; p++) {
                const std::string b = dump + "/b" + std::to_string(k) + ".";
                const size_t      n = ref[k * 4 + p].size();
                if (!write_file(b + "chained." + POINT[p] + ".f32", chained[p]->data() + (size_t) k * BATCH * per[p], n * 4) ||
                    !write_file(b + "fed." + POINT[p] + ".f32", fed[p]->data(), n * 4))
                    g_parity_fail++;
            }
        for (int p = 0; p < 4; p++) {
            const std::vector<float> & r = ref[k * 4 + p];
            const double ch = span_err(chained[p]->data() + (size_t) k * BATCH * per[p], r.data(), r.size());
            parity_row(cap, "b" + std::to_string(k) + "." + POINT[p], ch, fed[p] ? span_err(fed[p]->data(), r.data(), r.size()) : NaN, G);
        }
    }
    yourmt3_timing(cap.c_str(), c, (double) mono.size() / sr, ggml_backend_name(e.bp.backend));
    encoder_free(e);
    done();
    printf("yourmt3 parity %s: %d failing row(s)\n", g_parity_fail ? "FAIL" : "PASS", g_parity_fail);
    return g_parity_fail ? 1 : 0;
}

// ---------------------------------------------------------------------------

/** Release everything load_model() allocated, in reverse order.
 *
 *  Order matters: the scheduler references both backends, and each buffer is
 *  owned by the backend it was allocated on, so the backends go last. Every
 *  ggml_* free here tolerates null, and the handles are cleared so a second
 *  call is harmless. */
static void free_model(MidiModel * m) {
    if (!m) return;
    if (m->sched)  { ggml_backend_sched_free(m->sched);   m->sched  = nullptr; }
    if (m->kv_buf) { ggml_backend_buffer_free(m->kv_buf); m->kv_buf = nullptr; }
    if (m->kv_ctx) { ggml_free(m->kv_ctx);                m->kv_ctx = nullptr; }
    if (m->wbuf)   { ggml_backend_buffer_free(m->wbuf);   m->wbuf   = nullptr; }
    if (m->wctx)   { ggml_free(m->wctx);                  m->wctx   = nullptr; }
    // backend.h refcounts the shared backend pair, and already handles the
    // case where the best backend IS the CPU one and both handles are the
    // same pointer. Calling ggml_backend_free directly would bypass the
    // refcount and double free.
    backend_release(m->backend, m->cpu_backend);
    m->backend     = nullptr;
    m->cpu_backend = nullptr;
}

int main(int argc, char ** argv) {
    std::string model_dir, validate_dir, validate_decode_dir, validate_midi_dir;
    std::string audio_path, out_path = "out.mid";
    std::string dump_dir;
    std::string device = "auto";
    std::string beat_model, beats_out, export_events, beats_in, track_path, draft_dir;
    int         draft_k = 2;  // upstream default
    GridExport  opts;
    AbcOpts     abc;
    bool   jsonl  = false;
    bool   is_raw = false;
    double tol    = 1e-3;
    std::string chord_model, chord_vocab, chord_audio, chord_validate;
    std::string mt3_model, mt3_validate, mt3_audio, mt3_dump;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self-test-grid")) return grid_self_test();
        else if (!strcmp(argv[i], "--beat-model") && i + 1 < argc) beat_model = argv[++i];
        else if (!strcmp(argv[i], "--beats-out") && i + 1 < argc) beats_out = argv[++i];
        else if (!strcmp(argv[i], "--export") && i + 1 < argc) export_events = argv[++i];
        else if (!strcmp(argv[i], "--beats-in") && i + 1 < argc) beats_in = argv[++i];
        else if (!strcmp(argv[i], "--track") && i + 1 < argc) track_path = argv[++i];
        else if (!strcmp(argv[i], "--abc-out") && i + 1 < argc) abc.out = argv[++i];
        else if (!strcmp(argv[i], "--abc-ins") && i + 1 < argc) abc.ins_program = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--abc-melody-only")) abc.melody_only = true;
        else if (!strcmp(argv[i], "--bpm") && i + 1 < argc) opts.bpm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beats-per-bar") && i + 1 < argc) opts.beats_per_bar = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--first-downbeat") && i + 1 < argc) { opts.has_downbeat = true; opts.first_downbeat = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--tempo-map")) opts.tempo_map = 1;
        else if (!strcmp(argv[i], "--fixed-tempo")) opts.tempo_map = 0;
        else if (!strcmp(argv[i], "--beat-scale") && i + 1 < argc) opts.beat_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quantize")) opts.quantize = true;
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc) opts.subdivision = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quantize-strength") && i + 1 < argc) opts.strength = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fill-gaps") && i + 1 < argc) opts.fill_gaps = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trim-leading-bars")) opts.trim_leading = true;
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--draft-model") && i + 1 < argc) draft_dir = argv[++i];
        else if (!strcmp(argv[i], "--speculative-k") && i + 1 < argc) draft_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--validate") && i + 1 < argc) validate_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate-decode") && i + 1 < argc) validate_decode_dir = argv[++i];
        else if (!strcmp(argv[i], "--validate-midi") && i + 1 < argc) validate_midi_dir = argv[++i];
        else if (!strcmp(argv[i], "--transcribe") && i + 1 < argc) audio_path = argv[++i];
        else if (!strcmp(argv[i], "--transcribe-raw") && i + 1 < argc) { audio_path = argv[++i]; is_raw = true; }
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--jsonl")) jsonl = true;
        else if (!strcmp(argv[i], "--tol") && i + 1 < argc) tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
        else if (!strcmp(argv[i], "--chords") && i + 1 < argc) chord_audio = argv[++i];
        else if (!strcmp(argv[i], "--validate-chords") && i + 1 < argc) chord_validate = argv[++i];
        else if (!strcmp(argv[i], "--chord-model") && i + 1 < argc) chord_model = argv[++i];
        else if (!strcmp(argv[i], "--chord-vocab") && i + 1 < argc) chord_vocab = argv[++i];
        else if (!strcmp(argv[i], "--yourmt3-model") && i + 1 < argc) mt3_model = argv[++i];
        else if (!strcmp(argv[i], "--validate-yourmt3") && i + 1 < argc) mt3_validate = argv[++i];
        else if (!strcmp(argv[i], "--yourmt3-encode") && i + 1 < argc) mt3_audio = argv[++i];
        else if (!strcmp(argv[i], "--yourmt3-dump") && i + 1 < argc) mt3_dump = argv[++i];
    }
    {
        const double l2 = opts.beat_scale > 0 ? log2(opts.beat_scale) : 0.5;
        if (l2 != floor(l2)) {
            fprintf(stderr, "[ace-midi] --beat-scale must be a power of two, not %g\n", opts.beat_scale);
            return 2;
        }
    }
    if (!track_path.empty()) {  // beats only: no transcription model
        if (beat_model.empty() || beats_out.empty()) {
            fprintf(stderr, "[ace-midi] --track needs --beat-model <dir> and --beats-out <beats.json>\n");
            return 2;
        }
        ggml_time_init();
        int                sr   = 0;
        std::vector<float> mono = load_audio_mono(track_path, &sr);
        BeatTrack          tr;
        if (!run_beats(beat_model, mono, sr, &tr)) return 1;
        grid_midi({}, tr, (double) mono.size() / sr, opts, beats_out);
        return 0;
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
    if (!mt3_validate.empty() || !mt3_audio.empty()) {
        if (mt3_model.empty()) {
            fprintf(stderr, "[ace-midi] YourMT3 modes need --yourmt3-model <gguf>\n");
            return 2;
        }
        ggml_time_init();
        return mt3_validate.empty() ? run_yourmt3_encode(mt3_model, mt3_audio) : run_validate_yourmt3(mt3_model, mt3_validate, mt3_dump);
    }
    if (!chord_validate.empty()) return run_validate_chords(chord_model, chord_vocab, chord_validate);
    if (!chord_audio.empty()) {
        if (out_path == "out.mid") out_path = "chords.json";
        return run_chords(chord_model, chord_vocab, chord_audio, out_path);
    }
    if (!export_events.empty()) {
        if (beats_in.empty()) {
            fprintf(stderr, "[ace-midi] --export needs --beats-in <beats.json>\n");
            return 2;
        }
        return run_export(export_events, beats_in, out_path, opts, beats_out, abc);
    }
    if (model_dir.empty()) {
        fprintf(stderr,
                "ace-midi — MuScriptor GGML port, with Beat This! tempo and bars\n"
                "usage: ace-midi --model <dir> <mode>\n"
                "  modes:\n"
                "    --transcribe <audio.wav|mp3> [--out out.mid] [--jsonl]  audio -> MIDI\n"
                "    --transcribe-raw <f32.bin>   same, raw mono 16 kHz f32 input\n"
                "    --draft-model <dir> [--speculative-k <1-4>]   speculative decoding: a\n"
                "                              smaller checkpoint proposes k tokens (default 2)\n"
                "                              that the main model checks in one pass\n"
                "    --validate <dir>          logit parity vs oracle [--tol <x>]\n"
                "    --validate-decode <dir>   mel + greedy token parity vs oracle\n"
                "    --validate-midi <dir>     multi-chunk events + MIDI bytes vs oracle\n"
                "  --device auto|cpu|<name>  backend (default auto = best available;\n"
                "                            cpu is the byte-exact oracle-parity path)\n"
                "  tempo and bars (MuScriptor v0.4 export options):\n"
                "    --beat-model <dir>        Beat This! weights (model.safetensors): detect\n"
                "                              tempo, meter and downbeats, write them into the MIDI\n"
                "    --beats-out <beats.json>  tracked beats + the written grid in audio seconds\n"
                "    --bpm <x> | --beats-per-bar <n> | --first-downbeat <s>   override the grid\n"
                "    --tempo-map | --fixed-tempo   follow the beats / one tempo (default: follow\n"
                "                              when the beats don't fit one tempo)\n"
                "    --beat-scale <2|0.5|...>  beats per detected beat\n"
                "    --quantize [--grid <steps per beat>] [--quantize-strength <0..1>]\n"
                "    --fill-gaps <beats>       hold notes over short rests\n"
                "    --trim-leading-bars       drop empty bars before the first note\n"
                "    --abc-out <score.abc> [--abc-ins <program>] [--abc-melody-only]\n"
                "                              YuE2 lead sheet: voice -> Vocal, one instrument -> Ins,\n"
                "                              estimated key and chords, on the same grid\n"
                "  ace-midi --track <audio> --beat-model <dir> --beats-out <beats.json>\n"
                "                              beats and grid only, no transcription model\n"
                "  ace-midi --export <events.jsonl> --beats-in <beats.json> [options] --out x.mid\n"
                "                              re-export a transcription, no model needed\n"
                "  ace-midi --self-test-grid   grid maths on synthetic beats\n"
                "  ace-midi --chords <audio> --chord-model <gguf> --chord-vocab <tsv> [--out chords.json]\n"
                "                              ChordNet chord segments in seconds, CPU (GGML_N_THREADS)\n"
                "  ace-midi --validate-chords <capture dir> --chord-model <gguf> --chord-vocab <tsv>\n"
                "                              ChordNet parity against a reference capture\n"
                "  ace-midi --validate-yourmt3 <capture dir> --yourmt3-model <gguf> [--device cpu]\n"
                "           [--yourmt3-dump <dir>]  YourMT3+ front half (waveform, spectrogram, pre-encoder,\n"
                "                              encoder) against a reference capture; the dump keeps\n"
                "                              this run's chained and fed outputs\n"
                "  ace-midi --yourmt3-encode <audio> --yourmt3-model <gguf> [--device cpu]\n"
                "                              audio through the YourMT3+ encoder: placement and times only\n");
        return 2;
    }

    if (!draft_dir.empty() && (draft_k < 1 || draft_k > MIDI_MAX_DRAFT_K)) {
        fprintf(stderr, "[ace-midi] --speculative-k must be 1-%d, not %d\n", MIDI_MAX_DRAFT_K, draft_k);
        return 2;
    }

    ggml_time_init();
    MidiModel m;
    if (!load_model(&m, model_dir)) return 1;
    MidiModel draft;
    if (!draft_dir.empty()) {
        if (!load_model(&draft, draft_dir)) { free_model(&m); return 1; }
        // cards differ (small 1393, large 1395); only the emittable range must match
        if (n_valid_tokens(draft.cfg) != n_valid_tokens(m.cfg)) {
            fprintf(stderr, "[ace-midi] draft vocabulary (%d) differs from the model's (%d)\n",
                    n_valid_tokens(draft.cfg), n_valid_tokens(m.cfg));
            free_model(&draft);
            free_model(&m);
            return 2;
        }
        fprintf(stderr, "[ace-midi] speculative decoding: draft %s, k=%d\n", draft_dir.c_str(), draft_k);
    }
    MidiModel * draft_p = draft_dir.empty() ? nullptr : &draft;

    // Every mode used to `return run_*(...)` straight out of main, leaving the
    // scheduler, backends and buffers alive. On Metal the device lives in a
    // static vector whose destructor runs during __cxa_finalize_ranges and
    // asserts GGML_ASSERT([rsets->data count] == 0) inside
    // ggml_metal_rsets_free, which aborts with SIGABRT. Transcription had
    // already finished and written its MIDI by then, so the abnormal exit made
    // the Node side report FAILED and throw away a good result (issues #102,
    // #86). Free before returning.
    int rc = 2;
    if      (!validate_dir.empty())        rc = run_validate(&m, validate_dir, tol);
    else if (!validate_decode_dir.empty()) rc = run_validate_decode(&m, validate_decode_dir);
    else if (!validate_midi_dir.empty())   rc = run_validate_midi(&m, validate_midi_dir);
    else if (!dump_dir.empty())            rc = run_dump_logits(&m, dump_dir, out_path);
    else if (!audio_path.empty())          rc = run_transcribe(&m, audio_path, out_path, jsonl, is_raw, beat_model, opts, beats_out, abc,
                                                          draft_p, draft_k);
    else fprintf(stderr, "[ace-midi] no mode given\n");

    if (draft_p) free_model(draft_p);
    free_model(&m);
    return rc;
}
