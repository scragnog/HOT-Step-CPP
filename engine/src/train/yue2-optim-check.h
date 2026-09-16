#pragma once

// Small deterministic integration check: real AR/NAR adapter allocation,
// optimizer steps and checkpoint readers/writers, without model weights.
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
    if (m.cpu_backend != backend) ggml_backend_free(m.cpu_backend);
    ggml_backend_free(backend);
    return ar ? 0 : 1;
}
