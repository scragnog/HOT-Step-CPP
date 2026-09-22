#pragma once

// Small deterministic integration check: real AR/NAR adapter allocation,
// optimizer steps and checkpoint readers/writers, without model weights.
#include "yue2-aitk-lora.h"
template<class Context, class State, class Init, class Save, class Load, class Free>
static bool yue2_optim_check_half(const char * half, const std::string & dir, const Yue2Model & m,
                                 Init init, Save save, Load load, Free free_ctx) {
    for (const char * name : { "adamw", "prodigy", "muon" }) {
        Yue2OptimConfig cfg;
        cfg.optimizer = name;
        cfg.prodigy_d0 = 2e-5f;
        Context a, b;
        std::string err;
        auto fail = [&](const std::string & why) {
            fprintf(stderr, "[yue2-optim-check] FAIL %s/%s: %s\n", half, name, why.c_str());
            free_ctx(&a); free_ctx(&b);
            return false;
        };
        if (!init(m, cfg, 42, &a, &err) || !init(m, cfg, 99, &b, &err)) return fail(err);
        for (auto * c : { &a, &b }) {
            c->opt.base_lr = cfg.optimizer == "prodigy" ? 1.0f : 1e-3f;
            c->opt.lr_floor = 1.0f;
            c->opt.warmup_steps = 0;
            c->opt.total_steps = 8;
        }
        if (cfg.optimizer == "muon" && a.opt.n_muon == 0) return fail("no Muon tensors");
        auto step = [&](Context & c, int k) {
            for (size_t j = 0; j < c.opt.acc.size(); ++j) {
                auto * g = c.opt.acc[j];
                std::vector<float> v((size_t) ggml_nelements(g));
                for (size_t i = 0; i < v.size(); ++i)
                    v[i] = 0.01f * std::sin((float) (i + j * 7 + k * 3));
                ggml_backend_tensor_set(g, v.data(), 0, v.size() * sizeof(float));
            }
            LmStepStats stats{};
            return lm_optim_step(&c.opt, c.osched, &stats) && std::isfinite(stats.lr) && stats.lr > 0;
        };
        if (!step(a, 0) || !step(a, 1)) return fail("optimizer step");
        State st{};
        st.rank = 16; st.alpha = 16; st.n_layers = 1;
        st.n_params = (int32_t) a.params.size(); st.steps_done = 2;
        st.opt_step = a.opt.opt_step; st.opt_iter = a.opt.opt_iter;
        const std::string file = dir + "/" + half + "-" + name + ".bin";
        if (!save(file, st, a, &err)) return fail(err);
        State got{};
        if (!load(file, st, &got, &b, &err)) return fail(err);
        if (cfg.optimizer == "adamw") {
            // The v2 on-disk headers have fixed widths, independent of C++
            // struct padding. Remove v3's 48-byte optimizer record to exercise
            // the legacy reader against its original tensor-block layout.
            const size_t prefix = !strcmp(half, "ar") ? 116 : 108;
            FILE * f = hs_fopen(file.c_str(), "rb");
            if (!f) return fail("open legacy fixture");
            fseek(f, 0, SEEK_END);
            const long bytes = ftell(f);
            rewind(f);
            std::vector<uint8_t> raw((size_t) bytes);
            const bool read_ok = fread(raw.data(), 1, raw.size(), f) == raw.size();
            fclose(f);
            if (!read_ok || raw.size() < prefix + 48) return fail("read legacy fixture");
            raw.erase(raw.begin() + prefix, raw.begin() + prefix + 48);
            const uint32_t version = 2;
            memcpy(raw.data() + 8, &version, sizeof(version));
            const std::string legacy = file + ".v2";
            f = hs_fopen(legacy.c_str(), "wb");
            if (!f) return fail("create legacy fixture");
            const bool write_ok = fwrite(raw.data(), 1, raw.size(), f) == raw.size();
            fclose(f);
            if (!write_ok || !load(legacy, st, &got, &b, &err)) return fail("v2 restore: " + err);
        }
        if (!step(a, 2) || !step(b, 2)) return fail("resumed step");
        auto ta = yue2_optim_state_tensors(a.opt), tb = yue2_optim_state_tensors(b.opt);
        ta.insert(ta.end(), a.params.begin(), a.params.end());
        tb.insert(tb.end(), b.params.begin(), b.params.end());
        if (ta.size() != tb.size() || a.opt.prodigy_d != b.opt.prodigy_d ||
            a.opt.prodigy_r != b.opt.prodigy_r || a.opt.opt_step != b.opt.opt_step ||
            a.opt.opt_iter != b.opt.opt_iter) return fail("state counters differ");
        for (size_t j = 0; j < ta.size(); ++j) {
            std::vector<float> va((size_t) ggml_nelements(ta[j])), vb(va.size());
            ggml_backend_tensor_get(ta[j], va.data(), 0, va.size() * sizeof(float));
            ggml_backend_tensor_get(tb[j], vb.data(), 0, vb.size() * sizeof(float));
            if (memcmp(va.data(), vb.data(), va.size() * sizeof(float))) return fail("resume tensor differs");
            for (float v : va) if (!std::isfinite(v)) return fail("nonfinite state");
        }
        // Never reinterpret a checkpoint under another optimizer.
        const std::string selected = b.opt.optimizer;
        b.opt.optimizer = selected == "adamw" ? "prodigy" : "adamw";
        if (load(file, st, &got, &b, &err)) return fail("optimizer mismatch accepted");
        b.opt.optimizer = selected;
        if (selected == "muon") {
            b.opt.muon.lr_scale *= 2;
            if (load(file, st, &got, &b, &err)) return fail("Muon setting mismatch accepted");
        }
        fprintf(stderr, "[yue2-optim-check] PASS %s/%s: finite updates, bit-exact resume, mismatch rejected\n", half, name);
        free_ctx(&a); free_ctx(&b);
    }
    return true;
}

// Per-parameter LR groups, which is what --planner-lr-scale rides on: a
// parameter at lr_mul 0.5 must move EXACTLY half as far as the same parameter
// at 1.0, given the same gradients and the same step, and a parameter left at
// 1.0 must not move at all differently for having a scaled neighbour. Muon is
// refused by validation (one rate per shape bucket), so it is not checked here.
template<class Context, class Init, class Free>
static bool yue2_optim_check_lr_mul(const Yue2Model & m, Init init, Free free_ctx) {
    for (const char * name : { "adamw", "prodigy" }) {
        Yue2OptimConfig cfg;
        cfg.optimizer = name;
        cfg.prodigy_d0 = 1e-2f;
        Context a, b;
        std::string err;
        auto fail = [&](const std::string & why) {
            fprintf(stderr, "[yue2-optim-check] FAIL lr_mul/%s: %s\n", name, why.c_str());
            free_ctx(&a); free_ctx(&b);
            return false;
        };
        // Same seed both sides: the only difference is the multiplier.
        if (!init(m, cfg, 42, &a, &err) || !init(m, cfg, 42, &b, &err)) return fail(err);
        for (auto * c : { &a, &b }) {
            c->opt.base_lr = cfg.optimizer == "prodigy" ? 1.0f : 1e-1f;
            c->opt.lr_floor = 1.0f;
            c->opt.warmup_steps = 0;
            c->opt.total_steps = 8;
        }
        if (a.params.size() < 2) return fail("too few parameters to split");
        for (size_t j = 0; j < b.params.size(); j += 2)
            if (!lm_optim_set_lr_mul(&b.opt, b.params[j], 0.5f)) return fail("set_lr_mul rejected a parameter");
        auto read = [](Context & c, std::vector<std::vector<float>> * out) {
            out->resize(c.params.size());
            for (size_t j = 0; j < c.params.size(); ++j) {
                (*out)[j].resize((size_t) ggml_nelements(c.params[j]));
                ggml_backend_tensor_get(c.params[j], (*out)[j].data(), 0, (*out)[j].size() * sizeof(float));
            }
        };
        std::vector<std::vector<float>> a0, b0, a1, b1;
        read(a, &a0); read(b, &b0);
        for (size_t j = 0; j < a0.size(); ++j)
            if (a0[j] != b0[j]) return fail("same seed produced different initial weights");
        for (auto * c : { &a, &b })
            for (size_t j = 0; j < c->opt.acc.size(); ++j) {
                auto * g = c->opt.acc[j];
                std::vector<float> v((size_t) ggml_nelements(g));
                for (size_t i = 0; i < v.size(); ++i) v[i] = 0.01f * std::sin((float) (i + j * 7));
                ggml_backend_tensor_set(g, v.data(), 0, v.size() * sizeof(float));
            }
        LmStepStats sa{}, sb{};
        if (!lm_optim_step(&a.opt, a.osched, &sa) || !lm_optim_step(&b.opt, b.osched, &sb))
            return fail("optimizer step");
        read(a, &a1); read(b, &b1);
        // Every delta is rounded into an fp32 parameter, so one element's
        // halving is only good to ~1 ulp of that parameter's value (0.4% on a
        // delta of 1e-6). Compare the SUMS instead: rounding cancels, and a
        // multiplier that was ignored entirely still shows up as 0.5.
        double sum_err = 0.0, sum_ref = 0.0;
        for (size_t j = 0; j < a1.size(); ++j) {
            const bool scaled = (j % 2) == 0;
            for (size_t i = 0; i < a1[j].size(); ++i) {
                const double da = (double) a1[j][i] - (double) a0[j][i];
                const double db = (double) b1[j][i] - (double) b0[j][i];
                if (!std::isfinite(da) || !std::isfinite(db)) return fail("nonfinite update");
                if (!scaled) {
                    // An unscaled parameter must be untouched by its neighbour's
                    // multiplier — which for Prodigy also means the global d did
                    // not move, since d is estimated before any scaling.
                    if (a1[j][i] != b1[j][i]) return fail("unscaled parameter changed with a scaled neighbour");
                    continue;
                }
                sum_err += std::fabs(da * 0.5 - db);
                sum_ref += std::fabs(da);
            }
        }
        if (sum_ref <= 0.0) return fail("scaled parameters did not move at all");
        const double rel = sum_err / sum_ref;  // 0.5 if the multiplier is ignored
        if (rel > 1e-3) return fail("lr_mul 0.5 did not halve the update (relative error " + std::to_string(rel) + ")");
        fprintf(stderr, "[yue2-optim-check] PASS lr_mul/%s: 0.5x halves the update, neighbours untouched\n", name);
        free_ctx(&a); free_ctx(&b);
    }
    return true;
}

// Update modifiers (lm-optim.h, 2026-09-22). Two runs from the same seed and
// the same gradients, weight decay 0 so every delta is the pure update:
//
//   unfused : AdamW through the unfused graph must match the fused
//             ggml_opt_step_adamw to fp32 rounding (summed |delta| rel 1e-4).
//   cautious: for adamw / prodigy / muon, an element is KEPT iff its plain
//             update opposes its gradient (delta_off * g < 0) and is then
//             scaled by 1/clamp(mean(mask), 1e-3, 1); every other element must
//             not move at all. The mean is per tensor, except a Muon bucket,
//             where it is per bucket. Compared as sums, like lr_mul above.
template<class Context, class Init, class Free>
static bool yue2_optim_check_modifiers(const Yue2Model & m, Init init, Free free_ctx) {
    struct Case { const char * name; const char * optimizer; bool cautious; bool unfused; };
    const Case cases[] = {
        { "unfused/adamw",  "adamw",   false, true  },
        { "cautious/adamw", "adamw",   true,  false },
        { "cautious/prodigy", "prodigy", true, false },
        { "cautious/muon",  "muon",    true,  false },
    };
    for (const Case & cs : cases) {
        Yue2OptimConfig cfg;
        cfg.optimizer = cs.optimizer;
        cfg.prodigy_d0 = 1e-2f;
        Context a, b;
        std::string err;
        auto fail = [&](const std::string & why) {
            fprintf(stderr, "[yue2-optim-check] FAIL %s: %s\n", cs.name, why.c_str());
            free_ctx(&a); free_ctx(&b);
            return false;
        };
        if (!init(m, cfg, 42, &a, &err) || !init(m, cfg, 42, &b, &err)) return fail(err);
        for (auto * c : { &a, &b }) {
            c->opt.base_lr = cfg.optimizer == "prodigy" ? 1.0f : 1e-1f;
            c->opt.lr_floor = 1.0f;
            c->opt.warmup_steps = 0;
            c->opt.total_steps = 8;
            c->opt.weight_decay = 0.0f;
            c->opt.muon.wd = 0.0f;
        }
        // The cautious cases compare the unfused graph against itself with
        // the mask on, so `a` takes the unfused form too; the unfused case
        // is the one that compares unfused (b) against fused (a).
        b.opt.adamw_unfused = cs.unfused || cs.cautious;
        a.opt.adamw_unfused = cs.cautious;
        if (cfg.optimizer == "muon" && a.opt.n_muon == 0) return fail("no Muon tensors");
        auto read = [](Context & c, std::vector<std::vector<float>> * out) {
            out->resize(c.params.size());
            for (size_t j = 0; j < c.params.size(); ++j) {
                (*out)[j].resize((size_t) ggml_nelements(c.params[j]));
                ggml_backend_tensor_get(c.params[j], (*out)[j].data(), 0, (*out)[j].size() * sizeof(float));
            }
        };
        auto set_grads = [&](Context & c, int phase, std::vector<std::vector<float>> * keep) {
            for (size_t j = 0; j < c.opt.acc.size(); ++j) {
                auto * g = c.opt.acc[j];
                std::vector<float> v((size_t) ggml_nelements(g));
                for (size_t i = 0; i < v.size(); ++i) {
                    const float g1 = 0.01f * std::sin((float) (i * 3 + j * 7 + 1)) + 0.002f * std::cos((float) (i + 2 * j));
                    // Phase 2: on roughly half the elements a SMALL reversal, so
                    // the momentum keeps the old sign there and the update
                    // opposes the gradient — the case the mask exists for. On
                    // the first Adam step the update is exactly sign(g) and
                    // nothing could ever be masked, hence the warm-up.
                    v[i] = phase == 1 ? g1 : (((i + j) % 3) ? g1 : -0.05f * g1);
                }
                ggml_backend_tensor_set(g, v.data(), 0, v.size() * sizeof(float));
                if (keep) (*keep)[j] = v;
            }
        };
        std::vector<std::vector<float>> a0, b0, a1, b1, grads(a.params.size());
        LmStepStats sa{}, sb{};
        if (cs.cautious) {
            // Warm-up step with the mask OFF on both sides, so the two runs
            // enter the measured step with identical weights and moments.
            set_grads(a, 1, nullptr); set_grads(b, 1, nullptr);
            if (!lm_optim_step(&a.opt, a.osched, &sa) || !lm_optim_step(&b.opt, b.osched, &sb)) return fail("warm-up step");
            b.opt.cautious = true;
        }
        read(a, &a0); read(b, &b0);
        for (size_t j = 0; j < a0.size(); ++j) if (a0[j] != b0[j]) return fail("the two runs diverged before the measured step");
        set_grads(a, cs.cautious ? 2 : 1, &grads); set_grads(b, cs.cautious ? 2 : 1, nullptr);
        if (!lm_optim_step(&a.opt, a.osched, &sa) || !lm_optim_step(&b.opt, b.osched, &sb)) return fail("optimizer step");
        read(a, &a1); read(b, &b1);
        // Per-parameter scale group: the tensor itself, or its Muon bucket.
        std::vector<int> group(a.params.size());
        for (size_t j = 0; j < group.size(); ++j) group[j] = (int) j;
        int next_group = (int) group.size();
        if (cfg.optimizer == "muon")
            for (const LmMuonBucket & bk : a.opt.muon_buckets) { for (int j : bk.idx) group[(size_t) j] = next_group; ++next_group; }
        std::vector<double> kept(next_group, 0.0), total(next_group, 0.0);
        for (size_t j = 0; j < a1.size(); ++j)
            for (size_t i = 0; i < a1[j].size(); ++i) {
                const double d = (double) a1[j][i] - (double) a0[j][i];
                total[group[j]] += 1.0;
                if (d * (double) grads[j][i] < 0.0) kept[group[j]] += 1.0;
            }
        double sum_err = 0.0, sum_ref = 0.0, moved_masked = 0.0;
        size_t n_kept = 0, n_masked = 0;
        for (size_t j = 0; j < a1.size(); ++j) {
            const double mean = total[group[j]] > 0 ? kept[group[j]] / total[group[j]] : 0.0;
            const double scale = 1.0 / std::min(1.0, std::max(1e-3, mean));
            for (size_t i = 0; i < a1[j].size(); ++i) {
                const double da = (double) a1[j][i] - (double) a0[j][i];
                const double db = (double) b1[j][i] - (double) b0[j][i];
                if (!std::isfinite(da) || !std::isfinite(db)) return fail("nonfinite update");
                double pred = da;
                if (cs.cautious) {
                    const bool keep = da * (double) grads[j][i] < 0.0;
                    pred = keep ? da * scale : 0.0;
                    if (keep) ++n_kept; else { ++n_masked; moved_masked += std::fabs(db); }
                }
                sum_err += std::fabs(pred - db);
                sum_ref += std::fabs(pred);
            }
        }
        if (sum_ref <= 0.0) return fail("parameters did not move at all");
        const double rel = sum_err / sum_ref;
        const double bar = cs.cautious ? 1e-3 : 1e-4;
        if (rel > bar) return fail("update differs from the predicted one (relative error " + std::to_string(rel) + ")");
        if (cs.cautious && (n_kept == 0 || n_masked == 0)) return fail("cautious mask was all-ones or all-zeros; the check is vacuous");
        if (cs.cautious && moved_masked > 0.0) return fail("a masked element moved");
        fprintf(stderr, "[yue2-optim-check] PASS %s: %s (relative error %.2e)\n", cs.name,
                cs.cautious ? (std::to_string(n_kept) + " kept / " + std::to_string(n_masked) + " masked, masked elements exactly still").c_str()
                            : "unfused graph matches ggml_opt_step_adamw", rel);
        free_ctx(&a); free_ctx(&b);
    }
    return true;
}

// LoKr sites for the joint trainer: the graph lokr_apply_delta() builds
// (through yue2_aitk_make_expert_adapters_lokr's factorization, order and
// scale choices) must equal a host-materialized kron(w1, w2) x on every
// fused site, both monolithic and factorized w2, and the native-split
// exporter's row slicing must pick out exactly the q/k/v and gate/up rows.
// A disagreement here loads fine and computes something else.
static bool yue2_lokr_check(ggml_backend_t backend) {
    // Small YuE2-shaped dims: qkv out 32, o 16x16, gate_up out 48, down in 24.
    const Yue2AitkDims dims{16, 16, 8, 24};
    const int factor = 4;
    bool all_ok = true;
    for (int dim : { 2, 4 }) {  // 2 -> factorized w2 on the wide sites, 4 -> monolithic everywhere
        const float alpha = 1.5f * dim;
        ggml_init_params ip = { 256 * ggml_tensor_overhead(), nullptr, true };
        ggml_context * pctx = ggml_init(ip);
        Yue2AitkExpertAdapters ad;
        std::string why;
        if (!yue2_aitk_lokr_slices_ok(dims, factor, &why) ||
            !yue2_aitk_make_expert_adapters_lokr(pctx, dims, 1, dim, factor, alpha, &ad, "ar")) {
            fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d: cannot allocate sites (%s)\n", dim, why.c_str());
            ggml_free(pctx); return false;
        }
        ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, backend);
        if (!pbuf) { fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d: buffer\n", dim); ggml_free(pctx); return false; }
        // Deterministic, non-trivial fills (w2 / w2_b are zero-init in the
        // trainer; here they must be non-zero or the check proves nothing).
        auto fill = [](ggml_tensor * t, int salt) {
            std::vector<float> v((size_t) ggml_nelements(t));
            for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(0.37f * (float) i + (float) salt) + 0.05f * (float) salt;
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
            return v;
        };
        std::vector<float> host_w1, host_w2, host_w2a, host_w2b;
        int n_mono = 0, n_fact = 0;
        double worst = 0.0;
        for (int s = 0; s < kYue2AitkSites; ++s) {
            const Yue2AitkFusedLora & site = yue2_aitk_site(ad.layers[0], s);
            const LokrApplySite & k = site.lokr;
            host_w1 = fill(k.w1, 1 + s);
            if (k.mono) { host_w2 = fill(k.w2, 11 + s); ++n_mono; }
            else { host_w2a = fill(k.w2_a, 21 + s); host_w2b = fill(k.w2_b, 31 + s); ++n_fact; }
            const int64_t in = site.input_width, out = site.output_width, S = 3;
            // Host W2 [out_k, in_n] (torch order), monolithic or w2_a @ w2_b.
            std::vector<double> W2((size_t) (k.out_k * k.in_n), 0.0);
            for (int64_t kk = 0; kk < k.out_k; ++kk) for (int64_t n = 0; n < k.in_n; ++n) {
                if (k.mono) W2[(size_t) (kk * k.in_n + n)] = host_w2[(size_t) (kk * k.in_n + n)];
                else for (int64_t r = 0; r < dim; ++r) W2[(size_t) (kk * k.in_n + n)] += (double) host_w2a[(size_t) (kk * dim + r)] * (double) host_w2b[(size_t) (r * k.in_n + n)];
            }
            // Full delta W = scale * kron(w1, W2): [out, in], row l*out_k+kk, col m*in_n+n.
            std::vector<double> W((size_t) (out * in), 0.0);
            for (int64_t l = 0; l < k.out_l; ++l) for (int64_t m = 0; m < k.in_m; ++m)
                for (int64_t kk = 0; kk < k.out_k; ++kk) for (int64_t n = 0; n < k.in_n; ++n)
                    W[(size_t) ((l * k.out_k + kk) * in + (m * k.in_n + n))] =
                        (double) k.scale * (double) host_w1[(size_t) (l * k.in_m + m)] * W2[(size_t) (kk * k.in_n + n)];
            // Graph: x [in, S] -> delta [out, S].
            ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * gctx = ggml_init(gp);
            ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, in, S);
            ggml_tensor * d = ggml_reshape_2d(gctx, lokr_apply_delta(gctx, x, k), out, S);
            ggml_cgraph * gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, d);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            std::vector<float> hx((size_t) (in * S));
            for (size_t i = 0; i < hx.size(); ++i) hx[i] = std::cos(0.11f * (float) i + (float) s);
            bool ok = ggml_gallocr_alloc_graph(ga, gf);
            if (ok) { ggml_backend_tensor_set(x, hx.data(), 0, hx.size() * sizeof(float)); ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS; }
            std::vector<float> hd((size_t) (out * S));
            if (ok) ggml_backend_tensor_get(d, hd.data(), 0, hd.size() * sizeof(float));
            ggml_gallocr_free(ga); ggml_free(gctx);
            if (!ok) { fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d site %d: graph\n", dim, s); all_ok = false; continue; }
            double ref_mag = 0.0, err = 0.0;
            for (int64_t t = 0; t < S; ++t) for (int64_t o = 0; o < out; ++o) {
                double y = 0.0;
                for (int64_t i = 0; i < in; ++i) y += W[(size_t) (o * in + i)] * (double) hx[(size_t) (t * in + i)];
                ref_mag = std::max(ref_mag, std::fabs(y));
                err = std::max(err, std::fabs(y - (double) hd[(size_t) (t * out + o)]));
            }
            const double rel = ref_mag > 0.0 ? err / ref_mag : 1.0;
            worst = std::max(worst, rel);
            if (rel > 1e-5) { fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d site %d: matvec vs materialized kron rel %.3e\n", dim, s, rel); all_ok = false; }
            // Export slicing: rows [r0, r1) of W must be kron(w1[r0/out_k : r1/out_k], W2).
            const int64_t q = dims.q_width, kv = dims.kv_width, ff = dims.feed_forward;
            std::vector<std::pair<int64_t, int64_t>> ranges;
            if (s == 0) ranges = { {0, q}, {q, kv}, {q + kv, kv} };
            else if (s == 2) ranges = { {0, ff}, {ff, ff} };
            else ranges = { {0, out} };
            for (const auto & r : ranges) {
                if (r.first % k.out_k != 0 || r.second % k.out_k != 0) { fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d site %d: slice not on out_k\n", dim, s); all_ok = false; continue; }
                const int64_t l0 = r.first / k.out_k, l1 = (r.first + r.second) / k.out_k;
                for (int64_t l = l0; l < l1; ++l) for (int64_t kk = 0; kk < k.out_k; ++kk) for (int64_t i = 0; i < in; ++i) {
                    const int64_t m = i / k.in_n, n = i % k.in_n;
                    const double want = (double) k.scale * (double) host_w1[(size_t) (l * k.in_m + m)] * W2[(size_t) (kk * k.in_n + n)];
                    const double have = W[(size_t) ((r.first + (l - l0) * k.out_k + kk) * in + i)];
                    if (want != have) { fprintf(stderr, "[yue2-optim-check] FAIL lokr/dim%d site %d: slice row mismatch\n", dim, s); all_ok = false; l = l1; kk = k.out_k; break; }
                }
            }
        }
        if (all_ok) fprintf(stderr, "[yue2-optim-check] PASS lokr/dim%d: %d monolithic + %d factorized sites, matvec vs materialized kron max rel %.3e (bar 1e-5), q/k/v and gate/up row slices exact\n", dim, n_mono, n_fact, worst);
        ggml_backend_buffer_free(pbuf); ggml_free(pctx);
    }
    return all_ok;
}

static int yue2_optim_check_main(const std::string & dir, const char * backend_name = "CPU") {
    ggml_time_init();
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) return 1;
    pm_mkdir_p(dir);
    Yue2Model m;
    m.backend = backend;
    m.cpu_backend = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU
                        ? backend : ggml_backend_init_by_name("CPU", nullptr);
    if (!m.cpu_backend) { ggml_backend_free(backend); return 1; }
    m.lm_cfg.embedding_length = 32;
    m.lm_cfg.feed_forward_length = 64;
    m.lm_cfg.head_count = 2;
    m.lm_cfg.head_count_kv = 1;
    m.lm_cfg.key_length = 16;
    m.lm_cfg.latent_dim = 8;
    const bool nar = yue2_optim_check_half<Yue2NtTrainCtx, Yue2NtCkptState>("nar", dir, m,
        [](const Yue2Model & model, const Yue2OptimConfig & cfg, uint64_t seed, Yue2NtTrainCtx * c, std::string * err) {
            return yue2_nt_train_ctx_init(model, 1, 16, 16, YUE2_NT_ATTN_MLP_PROJ, seed, 0, 1, 1, "check", c, err, &cfg);
        }, yue2_nt_ckpt_save, yue2_nt_ckpt_load, yue2_nt_train_ctx_free);
    const bool ar = nar && yue2_optim_check_half<Yue2AtTrainCtx, Yue2AtCkptState>("ar", dir, m,
        [](const Yue2Model & model, const Yue2OptimConfig & cfg, uint64_t seed, Yue2AtTrainCtx * c, std::string * err) {
            return yue2_at_train_ctx_init(model, 1, 16, 16, YUE2_AT_T_ATTN_MLP, seed, 0, 1, 1, "check", c, err, true, &cfg);
        }, yue2_at_ckpt_save, yue2_at_ckpt_load, yue2_at_train_ctx_free);
    const bool mul = ar && yue2_optim_check_lr_mul<Yue2NtTrainCtx>(m,
        [](const Yue2Model & model, const Yue2OptimConfig & cfg, uint64_t seed, Yue2NtTrainCtx * c, std::string * err) {
            return yue2_nt_train_ctx_init(model, 1, 16, 16, YUE2_NT_ATTN_MLP_PROJ, seed, 0, 1, 1, "check", c, err, &cfg);
        }, yue2_nt_train_ctx_free);
    const bool mods = mul && yue2_optim_check_modifiers<Yue2NtTrainCtx>(m,
        [](const Yue2Model & model, const Yue2OptimConfig & cfg, uint64_t seed, Yue2NtTrainCtx * c, std::string * err) {
            return yue2_nt_train_ctx_init(model, 1, 16, 16, YUE2_NT_ATTN_MLP_PROJ, seed, 0, 1, 1, "check", c, err, &cfg);
        }, yue2_nt_train_ctx_free);
    const bool lokr = mods && yue2_lokr_check(backend);
    if (m.cpu_backend != backend) ggml_backend_free(m.cpu_backend);
    ggml_backend_free(backend);
    return lokr ? 0 : 1;
}
