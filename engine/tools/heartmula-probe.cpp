// HeartMuLa bring-up probe: tokenizer (CPU), tiny RoPE/GQA/sampler fixtures and
// global+local LM logit parity against engine/tools/heartmula-reference.py output.
#include "heartmula/local-graph.h"
#include "heartmula/tokenizer.h"

#include <cstring>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>

using JsonDoc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

static std::vector<int32_t> tokenize(const HeartMuLaTokenizer & t, yyjson_val * root) {
    const auto * text = yyjson_obj_get(root, "text");
    if (!yyjson_is_str(text)) { throw std::runtime_error("text must be a string"); }
    const auto * mode_value = yyjson_obj_get(root, "mode");
    if (mode_value && !yyjson_is_str(mode_value)) { throw std::runtime_error("mode must be a string"); }
    const std::string mode = mode_value ? yyjson_get_str(mode_value) : "raw";
    const std::string input(yyjson_get_str(text), yyjson_get_len(text));
    const auto * add = yyjson_obj_get(root, "add_special_tokens");
    if (add && !yyjson_is_bool(add)) { throw std::runtime_error("add_special_tokens must be a bool"); }
    if (mode == "raw") { return heartmula_tokenizer_encode(t, input, !add || yyjson_get_bool(add)); }
    if (mode == "tags" || mode == "lyrics") {
        if (add && !yyjson_get_bool(add)) { throw std::runtime_error("prompt modes always add BOS/EOS"); }
        return heartmula_tokenizer_prompt(t, input, mode == "tags");
    }
    throw std::runtime_error("mode must be raw, tags or lyrics");
}

static void print_ids(const std::vector<int32_t> & ids) {
    std::cout << "{\"ids\":[";
    for (size_t i = 0; i < ids.size(); ++i) { if (i) { std::cout << ','; } std::cout << ids[i]; }
    std::cout << "]}\n";
}

// ── .npy fixtures ───────────────────────────────────────────────────────────

struct Npy {
    std::vector<int64_t> shape;
    std::vector<double>  v;  // every dtype widened; fixtures are small or read once
    int64_t size() const { return (int64_t) v.size(); }
};

static Npy read_npy(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    char magic[8];
    f.read(magic, 8);
    if (std::string(magic + 1, 5) != "NUMPY") { throw std::runtime_error(path + ": not .npy"); }
    uint32_t hlen = 0;
    if (magic[6] == 1) { uint16_t h; f.read((char *) &h, 2); hlen = h; } else { f.read((char *) &hlen, 4); }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    const auto field = [&](const char * key) {
        const size_t at = header.find(key);
        if (at == std::string::npos) { throw std::runtime_error(path + ": header lacks " + key); }
        return at + strlen(key);
    };
    const size_t d = header.find('\'', field("'descr':")) + 1;
    const std::string descr = header.substr(d, header.find('\'', d) - d);
    const size_t fo = field("'fortran_order':");
    if (header.find("True", fo) < header.find(',', fo)) { throw std::runtime_error(path + ": Fortran order"); }
    Npy out;
    const size_t sp = header.find('(', field("'shape':")) + 1, se = header.find(')', sp);
    int64_t n = 1;
    for (std::string dims = header.substr(sp, se - sp); !dims.empty();) {
        const size_t c = dims.find(',');
        const std::string tok = dims.substr(0, c);
        if (tok.find_first_not_of(' ') != std::string::npos) { out.shape.push_back(std::stoll(tok)); n *= out.shape.back(); }
        dims = c == std::string::npos ? "" : dims.substr(c + 1);
    }
    out.v.resize((size_t) n);
    const auto read_all = [&](auto tag) {
        std::vector<decltype(tag)> raw((size_t) n);
        f.read((char *) raw.data(), (std::streamsize) (raw.size() * sizeof(tag)));
        for (int64_t i = 0; i < n; i++) { out.v[(size_t) i] = (double) raw[(size_t) i]; }
    };
    if (descr == "<f4") { read_all(float{}); }
    else if (descr == "<i8") { read_all(int64_t{}); }
    else if (descr == "<i4") { read_all(int32_t{}); }
    else if (descr == "|b1") { read_all(uint8_t{}); }
    else { throw std::runtime_error(path + ": unsupported dtype " + descr); }
    if (!f) { throw std::runtime_error(path + ": truncated"); }
    return out;
}

static std::vector<float> as_f32(const Npy & a) { return std::vector<float>(a.v.begin(), a.v.end()); }

static constexpr double M_PI_DOUBLE = 3.14159265358979323846;

struct Diff { double max_abs = 0, rel_l2 = 0; };

static Diff diff(const float * got, const double * want, int64_t n) {
    double num = 0, den = 0;
    Diff   d;
    for (int64_t i = 0; i < n; i++) {
        const double e = (double) got[i] - want[i];
        d.max_abs      = std::max(d.max_abs, std::fabs(e));
        num += e * e;
        den += want[i] * want[i];
    }
    d.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
    return d;
}

template <typename T> static int64_t argmax(const T * x, int64_t n) { return (int64_t) (std::max_element(x, x + n) - x); }

template <typename T> static double cross_entropy(const T * x, int64_t n, int64_t target) {
    const double top = (double) *std::max_element(x, x + n);
    double       sum = 0;
    for (int64_t i = 0; i < n; i++) { sum += std::exp((double) x[i] - top); }
    return std::log(sum) + top - (double) x[target];
}

// No silent CPU fallback: count graph nodes the scheduler placed on the CPU backend.
static void report_placement(const char * name, const HeartMuLaSlot & s, const BackendPair & bp, int * bad) {
    int cpu = 0;
    const int n = ggml_graph_n_nodes(s.graph);
    for (int i = 0; i < n; i++) {
        ggml_tensor * t = ggml_graph_node(s.graph, i);
        if (bp.cpu_backend != bp.backend && ggml_backend_sched_get_tensor_backend(s.sched, t) == bp.cpu_backend) {
            if (cpu++ < 5) { fprintf(stderr, "  CPU node %s (%s)\n", t->name, ggml_op_desc(t)); }
        }
    }
    printf("placement %-8s nodes=%d splits=%d cpu_nodes=%d backend=%s\n", name, n,
           ggml_backend_sched_get_n_splits(s.sched), cpu, ggml_backend_name(bp.backend));
    *bad += cpu;
}

// ── --tiny: torchtune RoPE + GQA attention, heartlib sample_topk ─────────────

static int run_tiny(const std::string & dir) {
    BackendPair bp   = backend_init("HeartMuLa-tiny");
    int         fail = 0;
    for (const bool global : { true, false }) {
        const std::string name = global ? "global" : "local";
        HeartMuLaStack    st;
        st.cfg = global ? heartmula_global_config() : heartmula_local_config();
        const int64_t D = st.cfg.head_dim, Nh = st.cfg.n_head, Nkv = st.cfg.n_head_kv;
        const Npy q = read_npy(dir + "/" + name + "_q.npy"), kv = read_npy(dir + "/" + name + "_kv.npy"),
                  pos = read_npy(dir + "/" + name + "_pos.npy"), out = read_npy(dir + "/" + name + "_out.npy"),
                  qr = read_npy(dir + "/" + name + "_q_rope.npy");
        const int64_t S = pos.size(), B = 2;

        ggml_init_params wp = { 4 * ggml_tensor_overhead(), nullptr, true };
        ggml_context *   wc = ggml_init(wp);
        st.freq_factors     = ggml_new_tensor_1d(wc, GGML_TYPE_F32, D / 2);
        ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wc, bp.backend);
        const auto ff = heartmula_rope_freq_factors((int) D, st.cfg.rope_base);
        ggml_backend_tensor_set(st.freq_factors, ff.data(), 0, ff.size() * sizeof(float));

        for (const bool flash : { false, true }) {
            if (flash && (!global || !bp.has_gpu)) { continue; }
            HeartMuLaSlot  s;
            ggml_cgraph *  gf  = nullptr;
            ggml_context * ctx = heartmula_slot_begin(&s, bp, &gf);
            ggml_tensor * in_q  = heartmula_input(ctx, GGML_TYPE_F32, Nh * D, S, B, "q");
            ggml_tensor * in_kv = heartmula_input(ctx, GGML_TYPE_F32, 2 * Nkv * D, S, B, "kv");
            ggml_tensor * in_p  = heartmula_input(ctx, GGML_TYPE_I32, S, 1, 1, "pos");
            ggml_tensor * in_m  = heartmula_input(ctx, GGML_TYPE_F16, S, S, 1, "mask");
            const auto part = [&](size_t off) {
                ggml_tensor * x = ggml_view_3d(ctx, in_kv, Nkv * D, S, B, in_kv->nb[1], in_kv->nb[2], off);
                return ggml_reshape_4d(ctx, ggml_cont(ctx, x), D, Nkv, S, B);
            };
            ggml_tensor * qq = heartmula_rope(ctx, ggml_reshape_4d(ctx, in_q, D, Nh, S, B), in_p, st);
            ggml_tensor * kk = heartmula_rope(ctx, part(0), in_p, st);
            ggml_tensor * vv = part((size_t) (Nkv * D) * sizeof(float));
            ggml_tensor * kc = ggml_cont(ctx, ggml_permute(ctx, kk, 0, 2, 1, 3));
            ggml_tensor * vc = ggml_cont(ctx, ggml_permute(ctx, vv, 0, 2, 1, 3));
            ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, qq, 0, 2, 1, 3));
            const float   scale = 1.0f / sqrtf((float) D);
            ggml_tensor * att;
            if (flash) {
                // Production flash reads the F16 cache, so this does too.
                att = ggml_flash_attn_ext(ctx, q4, ggml_cast(ctx, kc, GGML_TYPE_F16), ggml_cast(ctx, vc, GGML_TYPE_F16),
                                          in_m, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(att, GGML_PREC_F32);
            } else {
                att = heartmula_attn_manual(ctx, q4, kc, vc, in_m, scale);
            }
            ggml_set_output(qq);
            ggml_set_output(att);
            ggml_build_forward_expand(gf, qq);
            ggml_build_forward_expand(gf, att);
            std::string err;
            if (!heartmula_slot_alloc(&s, gf, &err, "tiny")) { throw std::runtime_error(err); }
            const auto qf = as_f32(q), kvf = as_f32(kv);
            std::vector<int32_t>  p32(pos.v.begin(), pos.v.end());
            std::vector<uint16_t> mask((size_t) (S * S));
            for (int64_t i = 0; i < S; i++) {
                for (int64_t j = 0; j < S; j++) { mask[(size_t) (i * S + j)] = ggml_fp32_to_fp16(j <= i ? 0.0f : -INFINITY); }
            }
            ggml_backend_tensor_set(in_q, qf.data(), 0, qf.size() * 4);
            ggml_backend_tensor_set(in_kv, kvf.data(), 0, kvf.size() * 4);
            ggml_backend_tensor_set(in_p, p32.data(), 0, p32.size() * 4);
            ggml_backend_tensor_set(in_m, mask.data(), 0, mask.size() * 2);
            ggml_backend_sched_graph_compute(s.sched, s.graph);
            std::vector<float> gq((size_t) ggml_nelements(qq)), ga((size_t) ggml_nelements(att));
            ggml_backend_tensor_get(qq, gq.data(), 0, gq.size() * 4);
            ggml_backend_tensor_get(att, ga.data(), 0, ga.size() * 4);
            const Diff dq = diff(gq.data(), qr.v.data(), qr.size()), da = diff(ga.data(), out.v.data(), out.size());
            // Float32 rotation angles at position 8191 are only good to ~1e-3 rad in
            // torch too, so judge both against float64 RoPE from torchtune's formula.
            std::vector<double> exact((size_t) qr.size());
            for (int64_t i = 0; i < D / 2; i++) {
                const double freq = 1.0 / std::pow(500000.0, 2.0 * (double) i / (double) D), wl = 2 * M_PI_DOUBLE / freq;
                const double sm   = (8192.0 / wl - 1.0) / 3.0;
                const double nf   = wl < 2048 ? freq : wl > 8192 ? freq / 32 : (1 - sm) * freq / 32 + sm * freq;
                for (int64_t b = 0; b < B; b++) {
                    for (int64_t t = 0; t < S; t++) {
                        const double a = pos.v[(size_t) t] * nf, c = std::cos(a), sn = std::sin(a);
                        for (int64_t h = 0; h < Nh; h++) {
                            const size_t o  = (size_t) (((b * S + t) * Nh + h) * D + 2 * i);
                            const double x0 = q.v[o], x1 = q.v[o + 1];
                            exact[o]        = x0 * c - x1 * sn;
                            exact[o + 1]    = x1 * c + x0 * sn;
                        }
                    }
                }
            }
            std::vector<float> torch_rope = as_f32(qr);
            const Diff ours_x = diff(gq.data(), exact.data(), qr.size()), torch_x = diff(torch_rope.data(), exact.data(), qr.size());
            // Manual attention is F32 throughout; flash reads F16 K/V.
            const bool ok = ours_x.max_abs <= 2.0 * torch_x.max_abs && da.rel_l2 < (flash ? 1e-3 : 1e-4);
            fail += !ok;
            printf("tiny %-6s %-6s rope vs torch max %.2e rel %.2e (vs exact: ours %.2e, torch %.2e) | attn max %.2e "
                   "rel %.2e  %s\n", name.c_str(), flash ? "flash" : "manual", dq.max_abs, dq.rel_l2, ours_x.max_abs,
                   torch_x.max_abs, da.max_abs, da.rel_l2, ok ? "PASS" : "FAIL");
            heartmula_slot_free(&s);
        }
        ggml_backend_buffer_free(wb);
        ggml_free(wc);
    }
    const Npy logits = read_npy(dir + "/sample_logits.npy"), draws = read_npy(dir + "/sample_draws.npy"),
              codes = read_npy(dir + "/sample_codes.npy");
    const int64_t  V  = logits.shape[1];
    const auto     lf = as_f32(logits), df = as_f32(draws);
    int            mism = 0;
    HeartMuLaDraws dr;
    for (int64_t r = 0; r < logits.shape[0]; r++) {
        dr.injected = df.data() + r * V;
        mism += heartmula_sample_topk(lf.data() + r * V, V, 50, 0.9f, dr) != (int32_t) codes.v[(size_t) r];
    }
    printf("tiny sampler %lld rows, mismatches %d  %s\n", (long long) logits.shape[0], mism, mism ? "FAIL" : "PASS");
    fail += mism != 0;
    backend_release(bp.backend, bp.cpu_backend);
    printf("TINY FIXTURES: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

// ── --lm: prefill + incremental c0..c7 logits vs heartmula-reference.py ──────

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static int run_lm(const std::string & fix, const std::string & variant, const std::string & gpath,
                  const std::string & lpath, const std::string & tok_path, int64_t chunk, int repeat_frames) {
    const int64_t     V = HEARTMULA_AUDIO_VOCAB, K = HEARTMULA_CODEBOOKS;
    const std::string base = fix + "/lm/" + variant;
    HeartMuLaModel    m;
    std::string       err;
    auto t0 = std::chrono::steady_clock::now();
    if (!heartmula_model_load(&m, gpath.c_str(), lpath.c_str(), &err)) { throw std::runtime_error(err); }
    printf("load %.0f ms, variant %s\n", ms_since(t0), m.variant.c_str());

    const Npy tokens = read_npy(base + "/f16/frame0/input_tokens.npy"), mask = read_npy(base + "/f16/frame0/input_mask.npy"),
              muq = read_npy(base + "/f16/prompt/muq_idx.npy"), teacher = read_npy(base + "/f16/teacher_codes.npy");
    const int64_t T = tokens.shape[1];
    std::vector<HeartMuLaPosition> prompt((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        for (int c = 0; c < HEARTMULA_COLUMNS; c++) {
            prompt[(size_t) i].id[c]   = (int32_t) tokens.v[(size_t) (i * HEARTMULA_COLUMNS + c)];
            prompt[(size_t) i].mask[c] = mask.v[(size_t) (i * HEARTMULA_COLUMNS + c)] != 0;
        }
    }
    if (!tok_path.empty()) {
        // The C++ tokenizer must rebuild the reference prompt's text column exactly.
        HeartMuLaTokenizer tk;
        if (!heartmula_tokenizer_load(&tk, tok_path.c_str(), &err)) { throw std::runtime_error(err); }
        auto         ids    = heartmula_tokenizer_prompt(tk, "Piano, ambient", true);
        const size_t muq_at = ids.size();
        ids.push_back(0);
        const auto lyr = heartmula_tokenizer_prompt(tk, "[verse]\nA quiet morning\nA distant light", false);
        ids.insert(ids.end(), lyr.begin(), lyr.end());
        bool same = (int64_t) ids.size() == T && (int64_t) muq_at == (int64_t) muq.v[0];
        for (int64_t i = 0; same && i < T; i++) { same = ids[(size_t) i] == prompt[(size_t) i].id[HEARTMULA_COLUMNS - 1]; }
        printf("prompt %lld tokens, muq_idx %lld, C++ tokenizer %s\n", (long long) T, (long long) muq.v[0],
               same ? "matches" : "DIFFERS");
        if (!same) { return 1; }
    }

    HeartMuLaLmGraph    g;
    HeartMuLaLocalGraph lg;
    // Frame 4 is the sparse long-position probe at 4096.
    if (!heartmula_lm_reserve(m, &g, 2, 4097, &err) || !heartmula_local_prepare(m, &lg, 2, &err)) {
        throw std::runtime_error(err);
    }
    std::vector<float> hidden((size_t) (2 * m.global.cfg.n_embd)), c0((size_t) (2 * V)), raw((size_t) (K * 2 * V));
    int    fail = 0, placement_bad = 0;
    // CE of one teacher code is noisy per entry, so it is gated over all 80 rows.
    double worst_ratio = 0, ce_ours = 0, ce_torch = 0, ce_ours_max = 0, ce_torch_max = 0;
    int    flips_ours = 0, flips_torch = 0;
    for (int f = 0; f < 5; f++) {
        const std::string fdir = "/frame" + std::to_string(f);
        t0 = std::chrono::steady_clock::now();
        if (f == 0) {
            if (!heartmula_lm_prefill(m, &g, prompt.data(), T, (int64_t) muq.v[0], hidden.data(), c0.data(), &err, chunk)) {
                throw std::runtime_error(err);
            }
        } else {
            const Npy ft = read_npy(base + "/f16" + fdir + "/input_tokens.npy"),
                      fm = read_npy(base + "/f16" + fdir + "/input_mask.npy"),
                      fp = read_npy(base + "/f16" + fdir + "/input_pos.npy");
            HeartMuLaPosition p;
            for (int c = 0; c < HEARTMULA_COLUMNS; c++) { p.id[c] = (int32_t) ft.v[(size_t) c]; p.mask[c] = fm.v[(size_t) c] != 0; }
            const int64_t position = (int64_t) fp.v[0];
            if (!heartmula_lm_step(m, &g, &p, &position, 1, -1, hidden.data(), c0.data(), &err)) { throw std::runtime_error(err); }
        }
        const double  tg = ms_since(t0);
        const int64_t tf = std::min(f, 3);
        const auto    teacher_pick = [&](int k, const float *, int) { return (int32_t) teacher.v[(size_t) (tf * K + k)]; };
        int32_t       codes[HEARTMULA_CODEBOOKS];
        t0 = std::chrono::steady_clock::now();
        if (!heartmula_local_frame(m, &lg, hidden.data(), c0.data(), teacher_pick, codes, raw.data(), &err)) {
            throw std::runtime_error(err);
        }
        printf("frame %d (pos %s): global %.1f ms, local %.1f ms\n", f, f == 4 ? "4096" : "seq", tg, ms_since(t0));
        if (f == 1) {
            report_placement("prefill", g.prefill.graph ? g.prefill : g.decode, m.bp, &placement_bad);
            report_placement("decode", g.decode, m.bp, &placement_bad);
            report_placement("local1", lg.slot[1], m.bp, &placement_bad);
            report_placement("local2", lg.slot[2], m.bp, &placement_bad);
        }
        for (int k = 0; k < K; k++) {
            const std::string kd  = fdir + "/c" + std::to_string(k) + "/raw.npy";
            const Npy         r32 = read_npy(base + "/fp32" + kd), r16 = read_npy(base + "/f16" + kd);
            const float *     got = raw.data() + (size_t) k * 2 * V;
            const auto        t16 = as_f32(r16);
            const Diff ours = diff(got, r32.v.data(), 2 * V), drift = diff(t16.data(), r32.v.data(), 2 * V);
            const int64_t target    = (int64_t) teacher.v[(size_t) (tf * K + k)];
            bool          argmax_ok = true;
            double        ce_gap = 0, torch_gap = 0;
            for (int r = 0; r < 2; r++) {
                // A flip is only a near-tie if the fp32 top-2 margin is inside torch's own F16 drift.
                const double * w  = r32.v.data() + r * V;
                const int64_t  a  = argmax(w, V);
                double         m2 = -INFINITY;
                for (int64_t i = 0; i < V; i++) { if (i != a) { m2 = std::max(m2, w[i]); } }
                const bool flip = argmax(got + r * V, V) != a;
                flips_ours += flip;
                flips_torch += argmax(t16.data() + r * V, V) != a;
                argmax_ok &= !flip || w[a] - m2 <= 2.0 * drift.max_abs;
                const double ce32 = cross_entropy(r32.v.data() + r * V, V, target);
                ce_gap    = std::max(ce_gap, std::fabs(cross_entropy(got + r * V, V, target) - ce32));
                torch_gap = std::max(torch_gap, std::fabs(cross_entropy(t16.data() + r * V, V, target) - ce32));
            }
            ce_ours += ce_gap;
            ce_torch += torch_gap;
            ce_ours_max  = std::max(ce_ours_max, ce_gap);
            ce_torch_max = std::max(ce_torch_max, torch_gap);
            // Every tolerance comes from PyTorch's own FP32-vs-F16 drift on the same logits.
            const double ratio = ours.rel_l2 / std::max(drift.rel_l2, 1e-12);
            const bool   ok    = argmax_ok && ratio <= 2.0 && ours.max_abs <= 2.0 * std::max(drift.max_abs, 1e-3);
            worst_ratio = std::max(worst_ratio, ratio);
            fail += !ok;
            printf("  c%d vs fp32: max %.4f rel %.2e dCE %.4f | torch f16: max %.4f rel %.2e dCE %.4f | ratio %.2f "
                   "argmax %s %s\n", k, ours.max_abs, ours.rel_l2, ce_gap, drift.max_abs, drift.rel_l2, torch_gap, ratio,
                   argmax_ok ? "ok" : "DIFF", ok ? "PASS" : "FAIL");
        }
    }
    printf("placement: %d CPU nodes\n", placement_bad);
    printf("argmax flips vs fp32 (near-ties only): ours %d, torch f16 %d of %lld rows %s\n", flips_ours, flips_torch,
           (long long) (5 * K * 2), flips_ours <= flips_torch ? "PASS" : "FAIL");
    fail += flips_ours > flips_torch;

    if (repeat_frames > 0) {
        // Same seed, same song: sample the frames twice from the same prompt.
        std::vector<int32_t> runs[2];
        double               per_frame = 0;
        for (auto & run : runs) {
            HeartMuLaDraws draws(1234);
            const auto     pick = heartmula_sampling_chooser(1.5f, 50, 1.0f, &draws);
            bool           ok   = heartmula_lm_reserve(m, &g, 2, T + repeat_frames, &err) &&
                        heartmula_lm_prefill(m, &g, prompt.data(), T, (int64_t) muq.v[0], hidden.data(), c0.data(), &err, chunk);
            t0 = std::chrono::steady_clock::now();
            for (int f = 0; ok && f < repeat_frames; f++) {
                int32_t codes[HEARTMULA_CODEBOOKS];
                ok = heartmula_local_frame(m, &lg, hidden.data(), c0.data(), pick, codes, nullptr, &err);
                run.insert(run.end(), codes, codes + K);
                HeartMuLaPosition p;
                for (int c = 0; c < K; c++) { p.id[c] = codes[c]; p.mask[c] = true; }
                const int64_t position = T + f;
                ok = ok && heartmula_lm_step(m, &g, &p, &position, 1, -1, hidden.data(), c0.data(), &err);
            }
            if (!ok) { throw std::runtime_error(err); }
            per_frame = ms_since(t0) / repeat_frames;
        }
        const bool same = runs[0] == runs[1];
        printf("seeded repeat: %d frames x 2 %s, %.2f ms/frame (%.1f frames/s; realtime is 12.5)\n", repeat_frames,
               same ? "identical" : "DIFFERENT", per_frame, 1000.0 / per_frame);
        fail += !same;
    }
    const bool ce_ok = ce_ours <= ce_torch && ce_ours_max <= 2.0 * ce_torch_max;
    {
        // 240 s budget, measured: 3000 frames + a 512-token prompt, full-chunk prefill.
        const int64_t n = 3000 + 512;
        if (!heartmula_lm_reserve(m, &g, 2, n, &err) ||
            !heartmula_lm_build(m, &g, &g.prefill, HEARTMULA_PREFILL_CHUNK, heartmula_bucket(n), &err) ||
            !heartmula_lm_build(m, &g, &g.decode, 1, heartmula_bucket(n), &err)) {
            throw std::runtime_error(err);
        }
        const auto mib = [](size_t b) { return (double) b / (1024.0 * 1024.0); };
        size_t local = 0;
        for (int i = 1; i < K; i++) { local += ggml_backend_sched_get_buffer_size(lg.slot[i].sched, m.bp.backend); }
        printf("240 s budget: weights global %.0f + local %.0f MiB, KV %.0f MiB (%lld positions), workspace prefill "
               "%.0f / decode %.0f / local %.0f MiB\n", mib(ggml_backend_buffer_get_size(m.wctx_global.buffer)),
               mib(ggml_backend_buffer_get_size(m.wctx_local.buffer)), mib(ggml_backend_buffer_get_size(g.kv_buf)),
               (long long) g.n_ctx, mib(ggml_backend_sched_get_buffer_size(g.prefill.sched, m.bp.backend)),
               mib(ggml_backend_sched_get_buffer_size(g.decode.sched, m.bp.backend)), mib(local));
    }
    printf("worst rel-L2 ratio to torch F16 drift: %.2f; |dCE| vs fp32 mean/max: ours %.4f/%.4f, torch f16 %.4f/%.4f %s\n",
           worst_ratio, ce_ours / (5 * K), ce_ours_max, ce_torch / (5 * K), ce_torch_max, ce_ok ? "PASS" : "FAIL");
    fail += !ce_ok;
    heartmula_local_free(&lg);
    heartmula_lm_free(&g);
    heartmula_model_free(&m);
    const bool ok = !fail && !placement_bad;
    printf("LM PARITY %s: 5 frames x %lld codebooks x 2 rows, variant %s, prefill chunk %lld\n", ok ? "PASS" : "FAIL",
           (long long) K, variant.c_str(), (long long) chunk);
    return ok ? 0 : 1;
}

// ── --adapter: runtime LoRA parity (heartmula-adapter-reference.py) and cache rules ──

static int run_adapter(const std::string & fix, const std::string & variant, const std::string & gpath,
                       const std::string & lpath, const std::string & adapter_path, const std::string & ref,
                       const std::string & negatives) {
    const int64_t     V = HEARTMULA_AUDIO_VOCAB;
    const std::string base = fix + "/lm/" + variant;
    HeartMuLaModel    m;
    std::string       err;
    if (!heartmula_model_load(&m, gpath.c_str(), lpath.c_str(), &err)) { throw std::runtime_error(err); }
    const Npy tokens = read_npy(base + "/f16/frame0/input_tokens.npy"), mask = read_npy(base + "/f16/frame0/input_mask.npy"),
              muq = read_npy(base + "/f16/prompt/muq_idx.npy");
    const int64_t T = tokens.shape[1];
    std::vector<HeartMuLaPosition> prompt((size_t) T);
    for (int64_t i = 0; i < T; i++) {
        for (int c = 0; c < HEARTMULA_COLUMNS; c++) {
            prompt[(size_t) i].id[c]   = (int32_t) tokens.v[(size_t) (i * HEARTMULA_COLUMNS + c)];
            prompt[(size_t) i].mask[c] = mask.v[(size_t) (i * HEARTMULA_COLUMNS + c)] != 0;
        }
    }
    HeartMuLaPosition step;
    {
        const Npy ft = read_npy(base + "/f16/frame1/input_tokens.npy"), fm = read_npy(base + "/f16/frame1/input_mask.npy");
        for (int c = 0; c < HEARTMULA_COLUMNS; c++) { step.id[c] = (int32_t) ft.v[(size_t) c]; step.mask[c] = fm.v[(size_t) c] != 0; }
    }
    const int64_t step_pos = (int64_t) read_npy(base + "/f16/frame1/input_pos.npy").v[0];

    HeartMuLaLmGraph g;
    int              fail = 0;
    const auto       check = [&](bool ok, const std::string & what) {
        printf("  %-62s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
        fail += !ok;
    };
    // Prefill then one cached decode step; the logits are the two rows' codebook-0 values.
    const auto run = [&](std::vector<float> * l0, std::vector<float> * l1) {
        std::vector<float> hidden((size_t) (2 * m.global.cfg.n_embd));
        l0->assign((size_t) (2 * V), 0.0f);
        l1->assign((size_t) (2 * V), 0.0f);
        if (!heartmula_lm_reserve(m, &g, 2, T + 8, &err) ||
            !heartmula_lm_prefill(m, &g, prompt.data(), T, (int64_t) muq.v[0], hidden.data(), l0->data(), &err) ||
            !heartmula_lm_step(m, &g, &step, &step_pos, 1, -1, hidden.data(), l1->data(), &err)) {
            throw std::runtime_error(err);
        }
    };
    const auto same = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    };

    HeartMuLaAdapter ad;
    if (!heartmula_adapter_load(&ad, m, adapter_path.c_str(), &err)) { throw std::runtime_error(err); }
    printf("adapter %s: rank %d alpha %.1f id %llu, %s over %s\n", adapter_path.c_str(), ad.rank, ad.alpha,
           (unsigned long long) ad.id, ad.variant.c_str(), m.variant.c_str());
    std::vector<float> b0, b1, a0, a1, z0, z1, r0, r1, w0, w1;
    run(&b0, &b1);  // no adapter bound

    heartmula_lm_set_adapter(&g, &ad, 0.0f);
    run(&z0, &z1);
    check(same(z0, b0) && same(z1, b1), "strength 0 logits are bitwise the base logits");

    heartmula_lm_set_adapter(&g, &ad, 1.0f);
    run(&a0, &a1);
    heartmula_lm_set_adapter(&g, &ad, 0.0f);
    run(&w0, &w1);
    check(same(w0, b0) && same(w1, b1), "adapter on, then strength 0: graphs and KV rebuilt, base logits");
    heartmula_lm_set_adapter(&g, nullptr, 1.0f);
    run(&w0, &w1);
    check(same(w0, b0) && same(w1, b1), "adapter detached: base logits");

    // A switch mid-sequence drops the KV rows and the built graphs; nothing is reused.
    heartmula_lm_set_adapter(&g, &ad, 1.0f);
    run(&r0, &r1);
    check(same(r0, a0) && same(r1, a1), "re-attached at strength 1: same logits as the first run");
    heartmula_lm_set_adapter(&g, &ad, 0.5f);
    heartmula_lm_sync_adapter(&g);
    size_t rows = 0;
    for (uint8_t v : g.valid) { rows += v; }
    check(rows == 0 && !g.prefill.graph && !g.decode.graph, "strength change drops KV rows and built graphs");
    heartmula_lm_set_adapter(&g, &ad, 1.0f);
    const uint64_t old_id = ad.id;
    if (!heartmula_adapter_load(&ad, m, adapter_path.c_str(), &err)) { throw std::runtime_error(err); }
    run(&r0, &r1);
    check(ad.id != old_id && same(r0, a0) && same(r1, a1), "reload takes a new id, graphs rebuilt, same logits");

    // Parity: torch FP32 is truth, torch F16 drift sets the tolerance (as run_lm does).
    const Npy t32[2] = { read_npy(ref + "/fp32/frame0_c0.npy"), read_npy(ref + "/fp32/frame1_c0.npy") };
    const Npy t16[2] = { read_npy(ref + "/f16/frame0_c0.npy"), read_npy(ref + "/f16/frame1_c0.npy") };
    const Npy tb[2]  = { read_npy(base + "/fp32/frame0/c0/raw.npy"), read_npy(base + "/fp32/frame1/c0/raw.npy") };
    const std::vector<float> * ours[2] = { &a0, &a1 };
    double worst = 0;
    for (int f = 0; f < 2; f++) {
        const float * got    = ours[f]->data();
        const Diff    d_ours = diff(got, t32[f].v.data(), 2 * V);
        const auto    h16    = as_f32(t16[f]);
        const Diff    drift  = diff(h16.data(), t32[f].v.data(), 2 * V);
        const auto    b32    = as_f32(tb[f]);
        const Diff    effect = diff(b32.data(), t32[f].v.data(), 2 * V);  // what the adapter changes
        bool argmax_ok = true;
        for (int r = 0; r < 2; r++) {
            const double * w  = t32[f].v.data() + r * V;
            const int64_t  a  = argmax(w, V);
            double         m2 = -INFINITY;
            for (int64_t i = 0; i < V; i++) { if (i != a) { m2 = std::max(m2, w[i]); } }
            argmax_ok &= argmax(got + r * V, V) == a || w[a] - m2 <= 2.0 * drift.max_abs;
        }
        const double ratio = d_ours.rel_l2 / std::max(drift.rel_l2, 1e-12);
        const bool   ok    = argmax_ok && ratio <= 2.0 && d_ours.max_abs <= 2.0 * std::max(drift.max_abs, 1e-3);
        worst = std::max(worst, ratio);
        printf("  frame %d c0 vs torch fp32: max %.4f rel %.2e | torch f16: max %.4f rel %.2e | ratio %.2f | adapter "
               "effect rel %.2e\n", f, d_ours.max_abs, d_ours.rel_l2, drift.max_abs, drift.rel_l2, ratio, effect.rel_l2);
        check(ok && effect.rel_l2 > 20.0 * d_ours.rel_l2,
              f == 0 ? "adapter-on prefill logits match PyTorch" : "adapter-on cached decode logits match PyTorch");
    }

    // Wrong files are refused with a reason; the good file is refused over a different model.
    if (!negatives.empty()) {
        for (const char * name : { "bad-variant", "bad-fingerprint", "bad-component", "bad-target" }) {
            HeartMuLaAdapter bad;
            err.clear();
            const bool loaded = heartmula_adapter_load(&bad, m, (negatives + "/" + name + ".gguf").c_str(), &err);
            printf("    %s: %s\n", name, err.c_str());
            check(!loaded && !err.empty() && !bad.loaded(), std::string("rejects ") + name);
        }
    }
    if (!negatives.empty()) {  // rank 1 gives a 1-D A to ggml_n_dims; it is valid and must load
        HeartMuLaAdapter one;
        err.clear();
        check(heartmula_adapter_load(&one, m, (negatives + "/ok-rank1.gguf").c_str(), &err) && one.rank == 1, "loads a valid rank-1 adapter");
        heartmula_adapter_free(&one);
    }
    {
        const std::string variant0 = m.variant, fp0 = m.base_fingerprint;  // identity is read before the file checks
        m.variant = variant0 == "3B" ? "3B-happy-new-year" : "3B";
        HeartMuLaAdapter bad;
        err.clear();
        check(!heartmula_adapter_load(&bad, m, adapter_path.c_str(), &err) && err.find("variant") != std::string::npos,
              "rejects the adapter over the other LM variant");
        m.variant          = variant0;
        m.base_fingerprint = std::string(64, '0');
        err.clear();
        check(!heartmula_adapter_load(&bad, m, adapter_path.c_str(), &err) && err.find("fingerprint") != std::string::npos,
              "rejects the adapter over a different base fingerprint");
        m.base_fingerprint = fp0;
    }
    heartmula_adapter_free(&ad);
    heartmula_lm_free(&g);
    heartmula_model_free(&m);
    printf("ADAPTER PARITY %s: backbone q/k/v/output LoRA, variant %s, worst rel-L2 ratio to torch F16 drift %.2f\n",
           fail ? "FAIL" : "PASS", variant.c_str(), worst);
    return fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    try {
        std::string path, mode = "raw", text, fixtures, tiny, lm, variant = "3B", global_path, local_path,
                    adapter, adapter_ref, negatives;
        int64_t chunk = HEARTMULA_PREFILL_CHUNK;
        int repeat_frames = 0;
        bool stdin_mode = false, have_text = false, action = false, add_bos = true;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::runtime_error("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--tokenizer") { path = value(); }
            else if (arg == "--tokenize") { action = true; }
            else if (arg == "--mode") { mode = value(); }
            else if (arg == "--text") { text = value(); have_text = true; }
            else if (arg == "--stdin") { stdin_mode = true; }
            else if (arg == "--no-special-tokens") { add_bos = false; }
            else if (arg == "--tokenizer-check") { fixtures = value(); action = true; }
            else if (arg == "--tiny") { tiny = value(); }
            else if (arg == "--lm") { lm = value(); }
            else if (arg == "--variant") { variant = value(); }
            else if (arg == "--global") { global_path = value(); }
            else if (arg == "--local") { local_path = value(); }
            else if (arg == "--adapter") { adapter = value(); }
            else if (arg == "--adapter-ref") { adapter_ref = value(); }
            else if (arg == "--negatives") { negatives = value(); }
            else if (arg == "--chunk") { chunk = std::stoll(value()); }
            else if (arg == "--repeat") { repeat_frames = std::stoi(value()); }
            else if (arg == "--help") {
                std::cout << "heartmula-probe --tokenize --tokenizer FILE "
                             "[--mode raw|tags|lyrics --text TEXT [--no-special-tokens] | --stdin]\n"
                             "heartmula-probe --tokenizer FILE --tokenizer-check CASES.json\n"
                             "heartmula-probe --tiny DIR   (heartmula-tiny-fixtures.py output)\n"
                             "heartmula-probe --lm FIXTURES --global G.gguf --local L.gguf [--variant 3B]\n"
                             "    [--tokenizer FILE] [--chunk N] [--repeat FRAMES]\n"
                             "heartmula-probe --lm FIXTURES --global G.gguf --local L.gguf --adapter A.gguf\n"
                             "    --adapter-ref DIR [--negatives DIR]   (heartmula-adapter-reference.py output)\n"
                             "--stdin reads JSON lines with text, mode and add_special_tokens.\n";
                return 0;
            } else { throw std::runtime_error("unknown option: " + arg); }
        }
        if (!tiny.empty()) { return run_tiny(tiny); }
        if (!lm.empty()) {
            if (global_path.empty() || local_path.empty() || chunk < 1) {
                throw std::runtime_error("--lm needs --global and --local");
            }
            if (!adapter.empty()) {
                if (adapter_ref.empty()) { throw std::runtime_error("--adapter needs --adapter-ref"); }
                return run_adapter(lm, variant, global_path, local_path, adapter, adapter_ref, negatives);
            }
            return run_lm(lm, variant, global_path, local_path, path, chunk, repeat_frames);
        }
        if (!action || path.empty() || (stdin_mode && have_text) ||
            (!fixtures.empty() && (stdin_mode || have_text))) {
            throw std::runtime_error("invalid options; use --help");
        }
        HeartMuLaTokenizer t;
        std::string error;
        if (!heartmula_tokenizer_load(&t, path.c_str(), &error)) { throw std::runtime_error(error); }
        if (!fixtures.empty()) {
            JsonDoc doc(yyjson_read_file(fixtures.c_str(), 0, nullptr, nullptr), yyjson_doc_free);
            if (!doc) { throw std::runtime_error("cannot read fixture JSON"); }
            auto * cases = yyjson_obj_get(yyjson_doc_get_root(doc.get()), "cases");
            if (!yyjson_is_arr(cases) || !yyjson_arr_size(cases)) { throw std::runtime_error("empty fixture cases"); }
            size_t index, count, mismatches = 0;
            yyjson_val * row;
            yyjson_arr_foreach(cases, index, count, row) {
                const auto ids = tokenize(t, row);
                auto * expected = yyjson_obj_get(row, "ids");
                bool match = yyjson_is_arr(expected) && yyjson_arr_size(expected) == ids.size();
                for (size_t j = 0; match && j < ids.size(); ++j) {
                    auto * id = yyjson_arr_get(expected, j);
                    match = yyjson_is_int(id) && yyjson_get_int(id) == ids[j];
                }
                if (!match) { ++mismatches; std::cerr << "mismatch at case " << index << '\n'; }
            }
            std::cout << "TOKENIZER FIXTURES: " << (mismatches ? "FAIL" : "PASS")
                      << "; cases=" << yyjson_arr_size(cases) << "; mismatches=" << mismatches << '\n';
            return mismatches ? 1 : 0;
        }
        if (stdin_mode) {
            std::string line;
            while (std::getline(std::cin, line)) {
                JsonDoc doc(yyjson_read(line.data(), line.size(), 0), yyjson_doc_free);
                if (!doc) { throw std::runtime_error("invalid JSON input line"); }
                print_ids(tokenize(t, yyjson_doc_get_root(doc.get())));
            }
        } else if (have_text) {
            if (mode == "raw") { print_ids(heartmula_tokenizer_encode(t, text, add_bos)); }
            else if ((mode == "tags" || mode == "lyrics") && add_bos) {
                print_ids(heartmula_tokenizer_prompt(t, text, mode == "tags"));
            } else { throw std::runtime_error("invalid mode/special-token options"); }
        } else { throw std::runtime_error("provide --text, --stdin or --tokenizer-check"); }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "heartmula-probe: " << e.what() << '\n';
        return 1;
    }
}
