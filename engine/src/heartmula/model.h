#pragma once
// heartmula/model.h: HeartMuLa global (3B) and local (300M) LM weights.
//
// HOT-Step file. Loads the two LM GGUFs written by engine/tools/convert-heartmula.py
// (engine/docs/HEARTMULA_CONTRACT.md, "GGUF metadata"). Both are torchtune
// llama3_2 decoders with Identity embeddings/outputs (heartlib modeling_heartmula.py):
//
//   global  28 layers, 24 q heads / 8 kv heads x 128, k/v width 1024, ctx 8192
//   local    3 layers,  8 q heads / 4 kv heads x 384, k/v width 1536, ctx 8
//
// Both use SwiGLU 8192, RMSNorm eps 1e-5, rope base 5e5 with Llama-3 scaling
// (factor 32, low 1, high 4, original context 8192), and torchtune's interleaved
// RoPE (pairs 2i, 2i+1), which is ggml's NORMAL mode. The global file also holds
// text_embeddings, audio_embeddings (8 x 8197 rows, codebook k at k * 8197),
// codebook0_head, muq_linear and unconditional_text_embedding; the local file holds
// projection and audio_head.0..6 (already split and transposed to [8197, 3072]).
// The local decode reads the global's audio_embeddings, so both load together.

#include <string>  // backend.h uses std::wstring without including it

#include "backend.h"
#include "gguf-weights.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define HEARTMULA_FORMAT_VERSION 1
#define HEARTMULA_CODEBOOKS      8
#define HEARTMULA_AUDIO_VOCAB    8197
#define HEARTMULA_COLUMNS        9  // 8 audio codebooks + text, heartlib's _parallel_number

struct HeartMuLaLmConfig {
    int   n_layer   = 0;
    int   n_embd    = 3072;
    int   n_head    = 0;
    int   n_head_kv = 0;
    int   head_dim  = 0;
    int   n_ff      = 8192;
    int   n_ctx     = 0;
    float rms_eps   = 1e-5f;
    float rope_base = 500000.0f;
};

static HeartMuLaLmConfig heartmula_global_config() {
    HeartMuLaLmConfig c;
    c.n_layer   = 28;
    c.n_head    = 24;
    c.n_head_kv = 8;
    c.head_dim  = 128;
    c.n_ctx     = 8192;
    return c;
}

static HeartMuLaLmConfig heartmula_local_config() {
    HeartMuLaLmConfig c;
    c.n_layer   = 3;
    c.n_head    = 8;
    c.n_head_kv = 4;
    c.head_dim  = 384;
    c.n_ctx     = HEARTMULA_CODEBOOKS;
    return c;
}

// torchtune Llama3ScaledRoPE.apply_scaling as ggml freq factors: ggml divides
// each pair's base frequency by ff[i], so ff = freq / scaled_freq.
static std::vector<float> heartmula_rope_freq_factors(int head_dim, float base) {
    const double scale = 32.0, low = 1.0, high = 4.0, old_ctx = 8192.0;
    const double low_wavelen = old_ctx / low, high_wavelen = old_ctx / high;
    std::vector<float> ff((size_t) head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        const float  freq    = 1.0f / powf(base, (float) (2 * i) / (float) head_dim);
        const double wavelen = 2.0 * 3.14159265358979323846 / freq;
        double       scaled  = freq;
        if (wavelen > low_wavelen) {
            scaled = freq / scale;
        } else if (wavelen >= high_wavelen) {
            const double smooth = (old_ctx / wavelen - low) / (high - low);
            scaled              = (1.0 - smooth) * freq / scale + smooth * freq;
        }
        ff[(size_t) i] = (float) ((double) freq / (double) (float) scaled);
    }
    return ff;
}

struct HeartMuLaLayer {
    ggml_tensor * sa_norm  = nullptr;
    ggml_tensor * q        = nullptr;
    ggml_tensor * k        = nullptr;
    ggml_tensor * v        = nullptr;
    ggml_tensor * o        = nullptr;
    ggml_tensor * mlp_norm = nullptr;
    ggml_tensor * w1       = nullptr;  // gate
    ggml_tensor * w2       = nullptr;  // down
    ggml_tensor * w3       = nullptr;  // up
};

struct HeartMuLaStack {
    HeartMuLaLmConfig           cfg;
    std::vector<HeartMuLaLayer> layers;
    ggml_tensor *               norm         = nullptr;
    ggml_tensor *               freq_factors = nullptr;  // F32 [head_dim / 2]
};

struct HeartMuLaModel {
    BackendPair bp          = {};
    bool        backend_ref = false;

    HeartMuLaStack global;
    HeartMuLaStack local;

    ggml_tensor * text_embd    = nullptr;  // [3072, 128256]
    ggml_tensor * audio_embd   = nullptr;  // [3072, 8 * 8197]
    ggml_tensor * uncond_embd  = nullptr;  // [3072, 1]
    ggml_tensor * muq_bias     = nullptr;  // [3072]; zero MuQ input projects to this
    ggml_tensor * c0_head      = nullptr;  // [3072, 8197]
    ggml_tensor * projection   = nullptr;  // [3072, 3072]
    ggml_tensor * audio_head[HEARTMULA_CODEBOOKS - 1] = {};

    WeightCtx   wctx_global = {};
    WeightCtx   wctx_local  = {};
    std::string variant;
    std::string base_fingerprint;
    bool        loaded = false;
};

static void heartmula_model_free(HeartMuLaModel * m) {
    wctx_free(&m->wctx_global);
    wctx_free(&m->wctx_local);
    if (m->backend_ref) {
        backend_release(m->bp.backend, m->bp.cpu_backend);
    }
    *m = HeartMuLaModel{};
}

namespace heartmula_model_detail {

inline bool fail(std::string * err, const std::string & msg) {
    if (err) {
        *err = msg;
    }
    return false;
}

// Header identity checks for one LM file; the full catalogue checks are S6's.
inline bool check_header(const GGUFModel & gf, const char * path, const char * arch, std::string * err) {
    const std::string got = gf_get_str(gf, "general.architecture");
    if (got != arch) {
        return fail(err, std::string(path) + ": architecture is '" + got + "', expected '" + arch + "'");
    }
    if (gf_get_u32(gf, "heartmula.format_version") != HEARTMULA_FORMAT_VERSION) {
        return fail(err, std::string(path) + ": unsupported heartmula.format_version");
    }
    if (!*gf_get_str(gf, "heartmula.base_fingerprint")) {
        return fail(err, std::string(path) + ": heartmula.base_fingerprint is missing");
    }
    return true;
}

inline bool has_all(const GGUFModel & gf, const std::vector<std::string> & names, const char * path,
                    std::string * err) {
    for (const auto & n : names) {
        if (gguf_find_tensor(gf.gguf, n.c_str()) < 0) {
            return fail(err, std::string(path) + ": tensor '" + n + "' is missing");
        }
    }
    return true;
}

inline std::vector<std::string> stack_names(const std::string & prefix, int n_layer) {
    std::vector<std::string> out;
    for (int i = 0; i < n_layer; i++) {
        const std::string p = prefix + ".layers." + std::to_string(i) + ".";
        for (const char * s : { "sa_norm.scale", "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight",
                                "attn.output_proj.weight", "mlp_norm.scale", "mlp.w1.weight", "mlp.w2.weight",
                                "mlp.w3.weight" }) {
            out.push_back(p + s);
        }
    }
    out.push_back(prefix + ".norm.scale");
    return out;
}

// Computed, not stored: staged like a converted weight so it rides the same buffer.
inline ggml_tensor * stage_freq_factors(WeightCtx * w, int head_dim, float base, const std::string & name) {
    const std::vector<float> ff = heartmula_rope_freq_factors(head_dim, base);
    ggml_tensor *            t  = ggml_new_tensor_1d(w->ctx, GGML_TYPE_F32, (int64_t) ff.size());
    ggml_set_name(t, name.c_str());
    auto buf = std::make_unique<float[]>(ff.size());
    std::copy(ff.begin(), ff.end(), buf.get());
    w->pending.push_back({ t, buf.get(), ff.size() * sizeof(float), 0 });
    w->staging.push_back(std::move(buf));
    return t;
}

inline void load_stack(WeightCtx * w, const GGUFModel & gf, const std::string & prefix, HeartMuLaStack * s) {
    s->layers.resize((size_t) s->cfg.n_layer);
    for (int i = 0; i < s->cfg.n_layer; i++) {
        const std::string p = prefix + ".layers." + std::to_string(i) + ".";
        HeartMuLaLayer &  l = s->layers[(size_t) i];
        l.sa_norm           = gf_load_tensor_f32(w, gf, p + "sa_norm.scale");
        l.q                 = gf_load_tensor(w, gf, p + "attn.q_proj.weight");
        l.k                 = gf_load_tensor(w, gf, p + "attn.k_proj.weight");
        l.v                 = gf_load_tensor(w, gf, p + "attn.v_proj.weight");
        l.o                 = gf_load_tensor(w, gf, p + "attn.output_proj.weight");
        l.mlp_norm          = gf_load_tensor_f32(w, gf, p + "mlp_norm.scale");
        l.w1                = gf_load_tensor(w, gf, p + "mlp.w1.weight");
        l.w2                = gf_load_tensor(w, gf, p + "mlp.w2.weight");
        l.w3                = gf_load_tensor(w, gf, p + "mlp.w3.weight");
    }
    s->norm         = gf_load_tensor_f32(w, gf, prefix + ".norm.scale");
    s->freq_factors = stage_freq_factors(w, s->cfg.head_dim, s->cfg.rope_base, prefix + ".rope_freq_factors");
}

inline bool check_shapes(const HeartMuLaStack & s, const char * path, std::string * err) {
    const int64_t H = s.cfg.n_embd, q = (int64_t) s.cfg.n_head * s.cfg.head_dim,
                  kv = (int64_t) s.cfg.n_head_kv * s.cfg.head_dim;
    for (const auto & l : s.layers) {
        for (const ggml_tensor * t : { l.q->ne[0] != H || l.q->ne[1] != q ? l.q : nullptr,
                                       l.k->ne[1] != kv ? l.k : nullptr, l.v->ne[1] != kv ? l.v : nullptr,
                                       l.o->ne[0] != q ? l.o : nullptr, l.w1->ne[1] != s.cfg.n_ff ? l.w1 : nullptr,
                                       l.w2->ne[0] != s.cfg.n_ff ? l.w2 : nullptr }) {
            if (t) {
                return fail(err, std::string(path) + ": tensor '" + ggml_get_name(t) +
                                     "' does not match the expected HeartMuLa config");
            }
        }
    }
    return true;
}

}  // namespace heartmula_model_detail

// Load global + local. Both must share the variant and base fingerprint.
static bool heartmula_model_load(HeartMuLaModel * m, const char * global_path, const char * local_path,
                                 std::string * err) {
    using namespace heartmula_model_detail;
    heartmula_model_free(m);
    GGUFModel gg = {}, gl = {};
    if (!gf_load(&gg, global_path)) {
        return fail(err, std::string("cannot open ") + global_path);
    }
    if (!gf_load(&gl, local_path)) {
        gf_close(&gg);
        return fail(err, std::string("cannot open ") + local_path);
    }
    auto close = [&]() {
        gf_close(&gg);
        gf_close(&gl);
    };
    if (!check_header(gg, global_path, "heartmula-global", err) ||
        !check_header(gl, local_path, "heartmula-local", err)) {
        close();
        return false;
    }
    m->variant          = gf_get_str(gg, "heartmula.variant");
    m->base_fingerprint = gf_get_str(gg, "heartmula.base_fingerprint");
    if (m->variant != gf_get_str(gl, "heartmula.variant") ||
        m->base_fingerprint != gf_get_str(gl, "heartmula.base_fingerprint")) {
        close();
        return fail(err, std::string(local_path) + ": base_fingerprint differs from " + global_path);
    }

    m->global.cfg = heartmula_global_config();
    m->local.cfg  = heartmula_local_config();

    std::vector<std::string> gnames = stack_names("backbone", m->global.cfg.n_layer);
    for (const char * n : { "text_embeddings.weight", "audio_embeddings.weight", "unconditional_text_embedding.weight",
                            "muq_linear.bias", "codebook0_head.weight" }) {
        gnames.push_back(n);
    }
    std::vector<std::string> lnames = stack_names("decoder", m->local.cfg.n_layer);
    lnames.push_back("projection.weight");
    for (int i = 0; i < HEARTMULA_CODEBOOKS - 1; i++) {
        lnames.push_back("audio_head." + std::to_string(i) + ".weight");
    }
    if (!has_all(gg, gnames, global_path, err) || !has_all(gl, lnames, local_path, err)) {
        close();
        return false;
    }

    m->bp          = backend_init("HeartMuLa");
    m->backend_ref = true;

    wctx_init(&m->wctx_global, (int) gnames.size() + 1);
    load_stack(&m->wctx_global, gg, "backbone", &m->global);
    m->text_embd   = gf_load_tensor(&m->wctx_global, gg, "text_embeddings.weight");
    m->audio_embd  = gf_load_tensor(&m->wctx_global, gg, "audio_embeddings.weight");
    m->uncond_embd = gf_load_tensor_f32(&m->wctx_global, gg, "unconditional_text_embedding.weight");
    m->muq_bias    = gf_load_tensor_f32(&m->wctx_global, gg, "muq_linear.bias");
    m->c0_head     = gf_load_tensor(&m->wctx_global, gg, "codebook0_head.weight");

    wctx_init(&m->wctx_local, (int) lnames.size() + 1);
    load_stack(&m->wctx_local, gl, "decoder", &m->local);
    m->projection = gf_load_tensor(&m->wctx_local, gl, "projection.weight");
    for (int i = 0; i < HEARTMULA_CODEBOOKS - 1; i++) {
        m->audio_head[i] = gf_load_tensor(&m->wctx_local, gl, "audio_head." + std::to_string(i) + ".weight");
    }

    const bool shapes = check_shapes(m->global, global_path, err) && check_shapes(m->local, local_path, err) &&
                        m->audio_embd->ne[1] == (int64_t) HEARTMULA_CODEBOOKS * HEARTMULA_AUDIO_VOCAB &&
                        m->c0_head->ne[1] == HEARTMULA_AUDIO_VOCAB &&
                        m->audio_head[0]->ne[1] == HEARTMULA_AUDIO_VOCAB;
    if (!shapes) {
        close();
        heartmula_model_free(m);
        return err && !err->empty() ? false : fail(err, "HeartMuLa embedding/head shapes are unexpected");
    }
    const bool ok = wctx_alloc(&m->wctx_global, m->bp.backend) && wctx_alloc(&m->wctx_local, m->bp.backend);
    close();
    if (!ok) {
        heartmula_model_free(m);
        return fail(err, "HeartMuLa weight allocation failed (out of VRAM?)");
    }
    m->loaded = true;
    return true;
}

// ── MuLaCover cover modules ─────────────────────────────────────────────────
// The global GGUF written by engine/tools/convert-mulacover.py also holds the cover
// modules of backbone layers 20..27 (upstream peft_layers; full modules, not LoRA):
// a symbolic adaptor (melody 256, drums 128 and chords 12 to 24 each, fusion 72 to
// 512, two Llama layers at 512 with 8 heads x 64, out to 3072), gated adaptive cross
// attention (main attention 3072 as 8 heads x 384, gate attention 12 with one head,
// x/context gates 3072 to 12, final gate 12 to 1) and the tag MLP (Qwen 1024 to
// 6144, split into scale and shift); plus qwen_linear (1024 to 3072) for the
// prompt's style slot. Every attention uses torchtune's Llama-3 scaled RoPE (base
// 5e5, factor 32) at its own head width, so the loader stages three freq-factor
// vectors. The graphs are in heartmula/mulacover.h and heartmula/lm-graph.h.

#define HEARTMULA_COVER_FIRST     20
#define HEARTMULA_COVER_LAYERS    8
#define HEARTMULA_COVER_MAX_STEPS 5000  // upstream MAX_SYMBOLIC_LENGTH, sixteenth-note rows
#define HEARTMULA_COVER_STYLE_DIM 1024  // Qwen3-Embedding-0.6B
#define HEARTMULA_COVER_ADAPT_DIM 512
#define HEARTMULA_COVER_ADAPT_HD  64   // adaptor: 8 heads x 64
#define HEARTMULA_COVER_CA_HD     384  // cross attention: 8 heads x 384
#define HEARTMULA_COVER_GATE_DIM  12   // gate attention: one head x 12
#define HEARTMULA_COVER_HEADS     8

struct HeartMuLaCoverLayer {
    ggml_tensor *  melody_w = nullptr, * melody_b = nullptr;  // adaptor.pianoroll_proj
    ggml_tensor *  drum_w = nullptr, * drum_b = nullptr;      // adaptor.drum_pianoroll_proj
    ggml_tensor *  chord_w = nullptr, * chord_b = nullptr;    // adaptor.chord_proj
    ggml_tensor *  fusion_w = nullptr, * fusion_b = nullptr;
    HeartMuLaLayer adaptor[2];
    ggml_tensor *  adaptor_norm = nullptr;
    ggml_tensor *  out_w = nullptr, * out_b = nullptr;
    ggml_tensor *  ca_norm = nullptr;                                                  // cross_attn.norm
    ggml_tensor *  ca_q = nullptr, * ca_k = nullptr, * ca_v = nullptr, * ca_o = nullptr;  // cross_attn.attn
    ggml_tensor *  gate_q = nullptr, * gate_k = nullptr, * gate_v = nullptr, * gate_o = nullptr;  // cross_attn.gate
    ggml_tensor *  x_gate_w = nullptr, * x_gate_b = nullptr;
    ggml_tensor *  ctx_gate_w = nullptr, * ctx_gate_b = nullptr;
    ggml_tensor *  gate_w = nullptr, * gate_b = nullptr;  // cross_attn.gate_proj
    ggml_tensor *  tag_w = nullptr, * tag_b = nullptr;    // tag_mlp_layers.N.mlp
};

struct HeartMuLaCover {
    std::vector<HeartMuLaCoverLayer> layers;  // [i] is backbone layer HEARTMULA_COVER_FIRST + i
    ggml_tensor *                    qwen_w = nullptr, * qwen_b = nullptr;
    ggml_tensor *                    ff_adapt = nullptr, * ff_ca = nullptr, * ff_gate = nullptr;  // RoPE freq factors
    WeightCtx                        wctx   = {};
    bool                             loaded = false;
};

static void heartmula_cover_free(HeartMuLaCover * c) {
    wctx_free(&c->wctx);
    *c = HeartMuLaCover{};
}

namespace heartmula_model_detail {

// ne0 = input width, ne1 = output width; ne1 == 0 marks a 1-D vector (scale or bias).
struct CoverSpec {
    std::string    name;
    int64_t        ne0, ne1;
    ggml_tensor ** dst;
};

inline std::vector<CoverSpec> cover_specs(HeartMuLaCover * c) {
    std::vector<CoverSpec> s;
    auto vec    = [&](const std::string & n, int64_t w, ggml_tensor ** d) { s.push_back({ n, w, 0, d }); };
    auto linear = [&](const std::string & n, int64_t in, int64_t out, ggml_tensor ** w, ggml_tensor ** b) {
        s.push_back({ n + ".weight", in, out, w });
        if (b) vec(n + ".bias", out, b);
    };
    c->layers.resize(HEARTMULA_COVER_LAYERS);
    for (int i = 0; i < HEARTMULA_COVER_LAYERS; i++) {
        HeartMuLaCoverLayer & l   = c->layers[(size_t) i];
        const std::string     idx = std::to_string(HEARTMULA_COVER_FIRST + i);
        const std::string     a = "peft_layers." + idx + ".adaptor.", x = "peft_layers." + idx + ".cross_attn.";
        linear(a + "pianoroll_proj", 256, 24, &l.melody_w, &l.melody_b);
        linear(a + "drum_pianoroll_proj", 128, 24, &l.drum_w, &l.drum_b);
        linear(a + "chord_proj", 12, 24, &l.chord_w, &l.chord_b);
        linear(a + "fusion_proj", 72, 512, &l.fusion_w, &l.fusion_b);
        for (int j = 0; j < 2; j++) {
            const std::string d  = a + "decoder.layers." + std::to_string(j) + ".";
            HeartMuLaLayer &  dl = l.adaptor[j];
            vec(d + "sa_norm.scale", 512, &dl.sa_norm);
            linear(d + "attn.q_proj", 512, 512, &dl.q, nullptr);
            linear(d + "attn.k_proj", 512, 512, &dl.k, nullptr);
            linear(d + "attn.v_proj", 512, 512, &dl.v, nullptr);
            linear(d + "attn.output_proj", 512, 512, &dl.o, nullptr);
            vec(d + "mlp_norm.scale", 512, &dl.mlp_norm);
            linear(d + "mlp.w1", 512, 2048, &dl.w1, nullptr);
            linear(d + "mlp.w2", 2048, 512, &dl.w2, nullptr);
            linear(d + "mlp.w3", 512, 2048, &dl.w3, nullptr);
        }
        vec(a + "decoder.norm.scale", 512, &l.adaptor_norm);
        linear(a + "out_proj", 512, 3072, &l.out_w, &l.out_b);
        vec(x + "norm.scale", 3072, &l.ca_norm);
        linear(x + "attn.q_proj", 3072, 3072, &l.ca_q, nullptr);
        linear(x + "attn.k_proj", 3072, 3072, &l.ca_k, nullptr);
        linear(x + "attn.v_proj", 3072, 3072, &l.ca_v, nullptr);
        linear(x + "attn.output_proj", 3072, 3072, &l.ca_o, nullptr);
        linear(x + "gate.q_proj", 12, 12, &l.gate_q, nullptr);
        linear(x + "gate.k_proj", 12, 12, &l.gate_k, nullptr);
        linear(x + "gate.v_proj", 12, 12, &l.gate_v, nullptr);
        linear(x + "gate.output_proj", 12, 12, &l.gate_o, nullptr);
        linear(x + "x_gate_proj", 3072, 12, &l.x_gate_w, &l.x_gate_b);
        linear(x + "context_gate_proj", 3072, 12, &l.ctx_gate_w, &l.ctx_gate_b);
        linear(x + "gate_proj", 12, 1, &l.gate_w, &l.gate_b);
        linear("tag_mlp_layers." + idx + ".mlp", 1024, 6144, &l.tag_w, &l.tag_b);
    }
    linear("qwen_linear", 1024, 3072, &c->qwen_w, &c->qwen_b);
    return s;
}

// Name, shape and dtype from the header: vectors F32/F16/BF16 (widened to F32 on load),
// matrices F32/F16/BF16 or a quant type whose block divides the row.
inline bool check_cover_spec(const GGUFModel & gf, const CoverSpec & s, const char * path, std::string * err) {
    const ggml_tensor * t = gguf_find_tensor(gf.gguf, s.name.c_str()) >= 0 ? ggml_get_tensor(gf.meta, s.name.c_str())
                                                                            : nullptr;
    if (!t) {
        return fail(err, std::string(path) + ": tensor '" + s.name + "' is missing");
    }
    const int64_t ne1 = s.ne1 ? s.ne1 : 1;
    if (t->ne[0] != s.ne0 || t->ne[1] != ne1 || t->ne[2] != 1 || t->ne[3] != 1) {
        return fail(err, std::string(path) + ": tensor '" + s.name + "' has shape [" + std::to_string(t->ne[0]) +
                             ", " + std::to_string(t->ne[1]) + "], expected [" + std::to_string(s.ne0) + ", " +
                             std::to_string(ne1) + "]");
    }
    const bool plain = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    const bool quant = s.ne1 && ggml_is_quantized(t->type) && t->ne[0] % ggml_blck_size(t->type) == 0;
    if (!plain && !quant) {
        return fail(err, std::string(path) + ": tensor '" + s.name + "' has unsupported type " +
                             ggml_type_name(t->type));
    }
    return true;
}

}  // namespace heartmula_model_detail

// Load the cover modules from MuLaCover's global GGUF onto the backend `m` already holds.
// `m` must be that same global (base fingerprint and variant match). Every name, shape and
// dtype is checked against the header before anything is allocated; the error names the
// first bad tensor.
static bool heartmula_cover_load(HeartMuLaCover * c, const HeartMuLaModel & m, const char * global_path,
                                 std::string * err) {
    using namespace heartmula_model_detail;
    heartmula_cover_free(c);
    if (!m.loaded) {
        return fail(err, "load the MuLaCover LM before its cover modules");
    }
    GGUFModel gf = {};
    if (!gf_load(&gf, global_path)) {
        return fail(err, std::string("cannot open ") + global_path);
    }
    if (!check_header(gf, global_path, "heartmula-global", err)) {
        gf_close(&gf);
        return false;
    }
    if (gf_get_str(gf, "heartmula.base_fingerprint") != m.base_fingerprint ||
        gf_get_str(gf, "heartmula.variant") != m.variant) {
        gf_close(&gf);
        return fail(err, std::string(global_path) + ": not the global file of the loaded " + m.variant + " model");
    }
    const std::vector<CoverSpec> specs = cover_specs(c);
    for (const auto & s : specs) {
        if (!check_cover_spec(gf, s, global_path, err)) {
            gf_close(&gf);
            heartmula_cover_free(c);
            return false;
        }
    }
    wctx_init(&c->wctx, (int) specs.size() + 3);
    for (const auto & s : specs) {
        *s.dst = s.ne1 ? gf_load_tensor(&c->wctx, gf, s.name) : gf_load_tensor_f32(&c->wctx, gf, s.name);
    }
    c->ff_adapt = stage_freq_factors(&c->wctx, HEARTMULA_COVER_ADAPT_HD, 500000.0f, "cover.adaptor.rope_freq_factors");
    c->ff_ca    = stage_freq_factors(&c->wctx, HEARTMULA_COVER_CA_HD, 500000.0f, "cover.cross_attn.rope_freq_factors");
    c->ff_gate  = stage_freq_factors(&c->wctx, HEARTMULA_COVER_GATE_DIM, 500000.0f, "cover.gate.rope_freq_factors");
    const bool ok = wctx_alloc(&c->wctx, m.bp.backend);
    gf_close(&gf);
    if (!ok) {
        heartmula_cover_free(c);
        return fail(err, "MuLaCover cover weight allocation failed (out of VRAM?)");
    }
    c->loaded = true;
    return true;
}
