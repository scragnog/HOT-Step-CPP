#pragma once
// heartmula/mulacover.h: MuLaCover's prompt, symbolic conditioning and context cache.
//
// HOT-Step file. Port of upstream mulacover (pinned e27fa7fa) pipeline.py _build_prompt
// and modeling.py SymbolicAdaptor / CrossAttentionLayer; the per-step half (tag MLP,
// GACA, style slot) is lm-graph.h note G. Weights: model.h HeartMuLaCover.
//
// A. INPUT. Rolls on the source's sixteenth-note grid, as upstream symbolic.py
//    to_tensors writes them and the server builds them: melody [steps][256] (pitch * 2
//    onset, pitch * 2 + 1 sustain), drums [steps][128], chords [steps][12] chroma, and
//    pad [steps] (upstream context_mask, 1 = padding row). At most 5000 steps. Key
//    positions are the row index 0..steps-1, with no BPM scaling.
//
// B. CFG. On the unconditioned row every roll value is -1 (upstream _mask_condition)
//    and so is the tag-MLP style input; the padding mask is shared by both rows.
//
// C. CONTEXT ONCE PER REQUEST. Every cover layer's adaptor reads the same rolls, and
//    layer N's context is the running sum of adaptor outputs 20..N. Nothing depends on
//    the frame, so one graph computes all eight contexts and writes each layer's roped
//    cross-attention and gate K/V into the graph's cover cache (F16, as the backbone KV).
//    Upstream recomputes them every frame; the values are the same.
//
// D. ADAPTOR. torchtune llama3_2 at 512: two layers, 8 heads x 64, SwiGLU 2048, RoPE at
//    the row index, final norm. Upstream passes a key-padding mask and no cache, so the
//    attention is bidirectional, not causal. Flash on GPU (K/V cast to F16).

#include "heartmula/lm-graph.h"
#include "heartmula/tokenizer.h"

#include <chrono>

// pipeline.py MuLaCoverGenConfig (checked against the GGUF's gen_config by the converter)
#define MULACOVER_TAG_START  128021
#define MULACOVER_TAG_END    128022
#define MULACOVER_MUQ_START  128023
#define MULACOVER_MUQ_END    128024
#define MULACOVER_QWEN_START 128025
#define MULACOVER_QWEN_END   128026

// pipeline.py _sanitize_parameters defaults
#define MULACOVER_DEFAULT_CFG         1.5f
#define MULACOVER_DEFAULT_TEMPERATURE 1.0f
#define MULACOVER_DEFAULT_TOPK        250

struct MuLaCoverPrompt {
    std::vector<int32_t> ids;  // text column
    int64_t              qwen_idx = 0, muq_idx = 0;
    std::string          tags;  // as the style encoder reads them (stripped, lowercased, unwrapped)
};

// pipeline.py preprocess + _build_prompt:
//   [tag_start] tags [tag_end] [qwen_start] 0 [qwen_end] [muq_start] 0 [muq_end] [bos] lyrics [eos]
static MuLaCoverPrompt mulacover_prompt(const HeartMuLaTokenizer & t, const std::string & tags,
                                        const std::string & lyrics) {
    static const char * ws    = " \t\r\n\v\f";
    auto                strip = [](std::string s) {
        s.erase(0, std::min(s.size(), s.find_first_not_of(ws)));
        s.erase(s.find_last_not_of(ws) + 1);
        return s;
    };
    MuLaCoverPrompt p;
    p.tags = heartmula_tokenizer_detail::lowercase(t, strip(tags));
    if (p.tags.size() >= 11 && p.tags.compare(0, 5, "<tag>") == 0 &&
        p.tags.compare(p.tags.size() - 6, 6, "</tag>") == 0) {
        p.tags = strip(p.tags.substr(5, p.tags.size() - 11));
    }
    const auto tag = heartmula_tokenizer_encode(t, p.tags, false);
    const auto lyr = heartmula_tokenizer_encode(t, heartmula_tokenizer_detail::lowercase(t, lyrics), false);
    p.ids.push_back(MULACOVER_TAG_START);
    p.ids.insert(p.ids.end(), tag.begin(), tag.end());
    p.ids.push_back(MULACOVER_TAG_END);
    p.qwen_idx = (int64_t) p.ids.size() + 1;
    p.muq_idx  = (int64_t) p.ids.size() + 4;
    for (int32_t id : { MULACOVER_QWEN_START, 0, MULACOVER_QWEN_END, MULACOVER_MUQ_START, 0, MULACOVER_MUQ_END }) {
        p.ids.push_back(id);
    }
    p.ids.push_back(t.bos_id);
    p.ids.insert(p.ids.end(), lyr.begin(), lyr.end());
    p.ids.push_back(t.eos_id);
    return p;
}

struct HeartMuLaCoverInput {
    int64_t              steps = 0;
    std::vector<float>   melody, drums, chords;  // [steps][256], [steps][128], [steps][12]
    std::vector<uint8_t> pad;                    // [steps], 1 = padding row
    std::vector<float>   style;                  // [1024] normalised Qwen vector (zeros for empty tags)
};

// Note B: the adaptor's input per CFG row, [B][steps][width].
static std::vector<float> heartmula_cover_rows(const std::vector<float> & roll, int B) {
    std::vector<float> out(roll);
    if (B == 2) {
        out.resize(roll.size() * 2, -1.0f);
    }
    return out;
}

// One adaptor layer (note D), x [512, steps, B].
static ggml_tensor * mulacover_adaptor_layer(ggml_context * ctx, const HeartMuLaLayer & w, const HeartMuLaCover & c,
                                             ggml_tensor * x, ggml_tensor * pos, ggml_tensor * mask, bool flash) {
    const int64_t S = x->ne[1], B = x->ne[2], D = HEARTMULA_COVER_ADAPT_HD, Nh = HEARTMULA_COVER_HEADS;
    ggml_tensor * n = heartmula_rms(ctx, x, w.sa_norm, 1e-5f);
    ggml_tensor * q = heartmula_rope_dim(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.q, n), D, Nh, S, B), pos, c.ff_adapt);
    ggml_tensor * k = heartmula_rope_dim(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.k, n), D, Nh, S, B), pos, c.ff_adapt);
    ggml_tensor * v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, w.v, n), D, Nh, S, B);
    q               = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [D, S, Nh, B]
    k               = ggml_permute(ctx, k, 0, 2, 1, 3);
    v               = ggml_permute(ctx, v, 0, 2, 1, 3);
    const float   scale = 1.0f / sqrtf((float) D);
    ggml_tensor * a;
    if (flash) {
        a = ggml_flash_attn_ext(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F16), ggml_cast(ctx, v, GGML_TYPE_F16), mask, scale,
                                0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32);
    } else {
        a = heartmula_attn_manual(ctx, q, ggml_cont(ctx, k), ggml_cont(ctx, v), mask, scale);
    }
    x                 = ggml_add(ctx, x, ggml_mul_mat(ctx, w.o, ggml_reshape_3d(ctx, a, D * Nh, S, B)));
    ggml_tensor * n2  = heartmula_rms(ctx, x, w.mlp_norm, 1e-5f);
    ggml_tensor * mlp = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, w.w1, n2)), ggml_mul_mat(ctx, w.w3, n2));
    return ggml_add(ctx, x, ggml_mul_mat(ctx, w.w2, mlp));
}

// Bind a cover to `g` (after heartmula_lm_reserve, which sets its CFG rows): run the
// context graph once (note C), keep its K/V, and arm the step graphs (lm-graph.h note G)
// with the style vector and the prompt's style slot `qwen_pos`. Optional `contexts`
// receives each cover layer's context [B][steps][3072], for parity.
static bool heartmula_cover_bind(const HeartMuLaModel & m, const HeartMuLaCover & c, HeartMuLaLmGraph * g,
                                 const HeartMuLaCoverInput & in, int64_t qwen_pos,
                                 std::vector<std::vector<float>> * contexts, std::string * err) {
    auto fail = [&](const std::string & e) {
        heartmula_cover_unbind(g);
        if (err) *err = "MuLaCover: " + e;
        return false;
    };
    heartmula_cover_unbind(g);
    const int64_t S = in.steps, B = g->B;
    if (!c.loaded || !g->kv_buf) return fail("load the cover modules and reserve the LM first");
    if (S < 1 || S > HEARTMULA_COVER_MAX_STEPS) {
        return fail("the lead sheet is " + std::to_string((long long) S) + " sixteenth notes; 1 to " +
                    std::to_string(HEARTMULA_COVER_MAX_STEPS) + " are allowed");
    }
    if (in.melody.size() != (size_t) S * 256 || in.drums.size() != (size_t) S * 128 ||
        in.chords.size() != (size_t) S * 12 || in.pad.size() != (size_t) S ||
        in.style.size() != HEARTMULA_COVER_STYLE_DIM) {
        return fail("roll, mask or style sizes do not match the step count");
    }
    if (std::all_of(in.pad.begin(), in.pad.end(), [](uint8_t p) { return p != 0; })) {
        return fail("every lead-sheet row is padding");
    }

    const int64_t CD = HEARTMULA_COVER_CA_HD, GD = HEARTMULA_COVER_GATE_DIM, Nh = HEARTMULA_COVER_HEADS;
    ggml_init_params ip = { (size_t) (4 * HEARTMULA_COVER_LAYERS) * ggml_tensor_overhead() + 1024, nullptr, true };
    g->cover_ctx        = ggml_init(ip);
    for (int l = 0; l < HEARTMULA_COVER_LAYERS; l++) {
        g->ca_k.push_back(ggml_new_tensor_4d(g->cover_ctx, GGML_TYPE_F16, CD, S, Nh, B));
        g->ca_v.push_back(ggml_new_tensor_4d(g->cover_ctx, GGML_TYPE_F16, CD, S, Nh, B));
        g->gate_k.push_back(ggml_new_tensor_4d(g->cover_ctx, GGML_TYPE_F16, GD, S, 1, B));
        g->gate_v.push_back(ggml_new_tensor_4d(g->cover_ctx, GGML_TYPE_F16, GD, S, 1, B));
    }
    g->cover_buf = ggml_backend_alloc_ctx_tensors(g->cover_ctx, m.bp.backend);
    if (!g->cover_buf) return fail("context cache allocation failed (out of VRAM?)");

    HeartMuLaSlot  s;
    ggml_cgraph *  gf  = nullptr;
    ggml_context * ctx = heartmula_slot_begin(&s, m.bp, &gf);
    ggml_tensor *  mel = heartmula_input(ctx, GGML_TYPE_F32, 256, S, B, "mc_melody");
    ggml_tensor *  dr  = heartmula_input(ctx, GGML_TYPE_F32, 128, S, B, "mc_drums");
    ggml_tensor *  ch  = heartmula_input(ctx, GGML_TYPE_F32, 12, S, B, "mc_chords");
    ggml_tensor *  pos = heartmula_input(ctx, GGML_TYPE_I32, S, 1, 1, "mc_pos");
    ggml_tensor *  msk = heartmula_input(ctx, GGML_TYPE_F16, S, S, 1, "mc_mask");

    auto linear = [&](ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) { return ggml_add(ctx, ggml_mul_mat(ctx, w, x), b); };
    std::vector<ggml_tensor *> outs;
    ggml_tensor *              context = nullptr;
    for (int l = 0; l < HEARTMULA_COVER_LAYERS; l++) {
        const HeartMuLaCoverLayer & w = c.layers[(size_t) l];
        ggml_tensor *               x = ggml_concat(ctx, ggml_concat(ctx, linear(w.melody_w, w.melody_b, mel),
                                                                     linear(w.drum_w, w.drum_b, dr), 0),
                                                    linear(w.chord_w, w.chord_b, ch), 0);  // [72, S, B]
        x = linear(w.fusion_w, w.fusion_b, x);
        for (const HeartMuLaLayer & al : w.adaptor) {
            x = mulacover_adaptor_layer(ctx, al, c, x, pos, msk, g->flash);
        }
        x       = linear(w.out_w, w.out_b, heartmula_rms(ctx, x, w.adaptor_norm, 1e-5f));  // [H, S, B]
        context = context ? ggml_add(ctx, context, x) : x;
        if (contexts) {
            ggml_set_output(context);
            outs.push_back(context);
        }
        auto store = [&](ggml_tensor * y, int64_t d, int64_t heads, ggml_tensor * ff, ggml_tensor * dst) {
            y = ggml_reshape_4d(ctx, y, d, heads, S, B);
            if (ff) y = heartmula_rope_dim(ctx, y, pos, ff);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_permute(ctx, y, 0, 2, 1, 3), dst));
        };
        store(ggml_mul_mat(ctx, w.ca_k, context), CD, Nh, c.ff_ca, g->ca_k[(size_t) l]);
        store(ggml_mul_mat(ctx, w.ca_v, context), CD, Nh, nullptr, g->ca_v[(size_t) l]);
        ggml_tensor * cg = linear(w.ctx_gate_w, w.ctx_gate_b, context);  // [12, S, B]
        store(ggml_mul_mat(ctx, w.gate_k, cg), GD, 1, c.ff_gate, g->gate_k[(size_t) l]);
        store(ggml_mul_mat(ctx, w.gate_v, cg), GD, 1, nullptr, g->gate_v[(size_t) l]);
    }
    if (!heartmula_slot_alloc(&s, gf, err, "MuLaCover context")) {
        heartmula_slot_free(&s);
        return fail(err ? *err : "context graph allocation failed");
    }

    const auto            m_rows = heartmula_cover_rows(in.melody, (int) B), d_rows = heartmula_cover_rows(in.drums, (int) B),
                          c_rows = heartmula_cover_rows(in.chords, (int) B);
    std::vector<int32_t>  kpos((size_t) S);
    std::vector<uint16_t> mask((size_t) (S * S));
    for (int64_t j = 0; j < S; j++) kpos[(size_t) j] = (int32_t) j;
    for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) mask[(size_t) (i * S + j)] = ggml_fp32_to_fp16(in.pad[(size_t) j] ? -INFINITY : 0.0f);
    }
    ggml_backend_tensor_set(mel, m_rows.data(), 0, m_rows.size() * sizeof(float));
    ggml_backend_tensor_set(dr, d_rows.data(), 0, d_rows.size() * sizeof(float));
    ggml_backend_tensor_set(ch, c_rows.data(), 0, c_rows.size() * sizeof(float));
    ggml_backend_tensor_set(pos, kpos.data(), 0, kpos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(msk, mask.data(), 0, mask.size() * sizeof(uint16_t));
    int       cpu_nodes = 0;
    const int n_nodes   = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        cpu_nodes += m.bp.cpu_backend != m.bp.backend &&
                     ggml_backend_sched_get_tensor_backend(s.sched, ggml_graph_node(gf, i)) == m.bp.cpu_backend;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = ggml_backend_sched_graph_compute(s.sched, gf) == GGML_STATUS_SUCCESS;
    if (ok && contexts) {
        contexts->assign(outs.size(), {});
        for (size_t l = 0; l < outs.size(); l++) {
            (*contexts)[l].resize((size_t) ggml_nelements(outs[l]));
            ggml_backend_tensor_get(outs[l], (*contexts)[l].data(), 0, ggml_nbytes(outs[l]));
        }
    }
    fprintf(stderr, "[MuLaCover] context: %lld steps x %lld rows, %d nodes, %d splits, %d on CPU, %.1f ms\n",
            (long long) S, (long long) B, n_nodes, ggml_backend_sched_get_n_splits(s.sched), cpu_nodes,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    heartmula_slot_free(&s);
    if (!ok) return fail("context graph compute failed");

    g->cover     = &c;
    g->cover_len = S;
    g->qwen_pos  = qwen_pos;
    g->cover_pad = in.pad;
    g->qwen      = in.style;
    g->style     = in.style;
    if (B == 2) g->style.resize((size_t) (2 * HEARTMULA_COVER_STYLE_DIM), -1.0f);  // note B
    return true;
}
