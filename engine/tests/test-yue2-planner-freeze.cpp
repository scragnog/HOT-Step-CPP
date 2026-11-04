// The planner freeze the held-out gate reuses (GH #316), on the real train
// state and the real LmOptim update, CPU backend: after one live step has
// given every moment a value, a frozen step leaves every planner slot (the
// text_encoders.* adapters, LoRA / DoRA / LoKr alike) bit-identical while the
// decoder's slots move, under adamw-lm, prodigy and muon. The native 8-bit
// AdamW is a CUDA kernel and is not built here; the hold it relies on is the
// same write-back after the update.
// Build (VS x64 prompt, from engine/tests; the exe needs build\Release's ggml
// DLLs on PATH):
//   cl /std:c++17 /EHsc /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN /DGGML_MAX_NAME=128 /DYUE2_AITK_CUDA_RUNTIME=0
//      /I..\src /I..\src\train /I..\ggml\include /I..\vendor\yyjson
//      test-yue2-planner-freeze.cpp ..\vendor\yyjson\yyjson.c
//      /link ..\build\ggml\src\Release\ggml.lib ..\build\ggml\src\Release\ggml-base.lib
static void jl(const char * fmt, ...) { (void) fmt; }

#include "yue2-aitk-train-state.h"
#include "lm-optim.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return false; } } while (0)

static std::vector<float> read(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static bool one(ggml_backend_t backend, const char * adapter, const char * optimizer) {
    const bool lokr = !std::strcmp(adapter, "lokr"), dora = !std::strcmp(adapter, "dora");
    Yue2AitkTrainState state; std::string error;
    // Rank 16 / LoKr dim 16: small on the CPU, and at Muon's min_dim, so Muon
    // takes the factors itself instead of handing them all to AdamW.
    CHECK(state.initialize(backend, 7, &error, false, 16, 16.0f, lokr ? 16 : 0, lokr ? 8 : 0, dora));
    const auto named = state.named_tensors();
    // Fresh LoRA B factors are zero; nonzero weights everywhere make weight
    // decay visible on any slot that is not held.
    std::mt19937 rng(11); std::normal_distribution<float> n01(0.0f, 0.02f);
    for (const auto & p : named) { std::vector<float> w((size_t) ggml_nelements(p.parameter)); for (float & x : w) x = n01(rng); ggml_backend_tensor_set(p.parameter, w.data(), 0, ggml_nbytes(p.parameter)); }

    LmOptim o; o.optimizer = !std::strcmp(optimizer, "adamw-lm") ? "adamw" : optimizer;
    o.lr_floor = 1.0f; o.total_steps = 1; o.warmup_steps = 0; o.weight_decay = 0.1f; o.grad_clip = 0.0f; o.base_lr = !std::strcmp(optimizer, "prodigy") ? 1.0f : 1e-3f;
    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * sc = ggml_init(sp);
    o.t_lossgrad = ggml_new_tensor_1d(sc, GGML_TYPE_F32, 1); o.t_adamw = ggml_new_tensor_1d(sc, GGML_TYPE_F32, 7);
    o.t_clip = ggml_new_tensor_1d(sc, GGML_TYPE_F32, 1); o.t_eps = ggml_new_tensor_1d(sc, GGML_TYPE_F32, 1); o.t_gnorm2 = ggml_new_tensor_1d(sc, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t sb = ggml_backend_alloc_ctx_tensors(sc, backend);
    const float one_f = 1.0f, eps = 1e-6f;
    ggml_backend_tensor_set(o.t_lossgrad, &one_f, 0, 4); ggml_backend_tensor_set(o.t_clip, &one_f, 0, 4); ggml_backend_tensor_set(o.t_eps, &eps, 0, 4);
    std::vector<ggml_tensor *> params; for (const auto & p : named) params.push_back(p.parameter);
    std::string lm_err;
    CHECK(lm_optim_init(&o, params, backend, &lm_err));
    ggml_backend_t backends[1] = { backend }; ggml_backend_buffer_type_t bufts[1] = { ggml_backend_get_default_buffer_type(backend) };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, std::max(16384, o.est_nodes), false, true);
    CHECK(sched);

    // The joint step's tail (yue2-aitk-joint-step.h): zero the planner's
    // gradients if frozen, clip, fill the accumulators, update, hold.
    const auto step = [&](bool frozen, bool hold = true) -> bool {
        for (const auto & p : named) { std::vector<float> g((size_t) ggml_nelements(p.gradient)); for (float & x : g) x = n01(rng); ggml_backend_tensor_set(p.gradient, g.data(), 0, ggml_nbytes(p.gradient)); }
        if (frozen) state.zero_planner_gradients();
        double norm = 0.0;
        CHECK(state.clip_gradients(1.0f, &norm, &error));
        CHECK(state.fill_device_gradients(o.acc, &error));
        LmStepStats stats{};
        CHECK(lm_optim_step(&o, sched, &stats));
        if (frozen && hold) state.hold_planner();
        ggml_backend_synchronize(backend);
        return true;
    };
    CHECK(step(false));  // live: every moment is now nonzero
    state.freeze_planner();
    std::vector<std::vector<float>> before; for (const auto & p : named) before.push_back(read(p.parameter));
    CHECK(step(true)); CHECK(step(true));
    size_t planner = 0, decoder = 0, decoder_moved = 0;
    for (size_t i = 0; i < named.size(); ++i) {
        const auto after = read(named[i].parameter);
        if (named[i].name.rfind("diffusion_model", 0) == 0) { ++decoder; decoder_moved += after != before[i]; }
        else { ++planner; if (after != before[i]) { std::fprintf(stderr, "FAIL %s/%s: frozen planner slot %s moved\n", adapter, optimizer, named[i].name.c_str()); return false; } }
    }
    CHECK(planner > 0 && decoder > 0 && decoder_moved == decoder);
    // Control: the same frozen step without the hold moves the planner
    // (momentum and weight decay), so the hold is what kept it still.
    CHECK(step(true, false));
    size_t drifted = 0;
    for (size_t i = 0; i < named.size(); ++i)
        if (named[i].name.rfind("diffusion_model", 0) != 0) drifted += read(named[i].parameter) != before[i];
    CHECK(drifted == planner);
    std::printf("PASS %-4s x %-8s planner %zu slots held (all drift without the hold), decoder %zu slots moved, %d on Muon\n",
                adapter, optimizer, planner, decoder_moved, o.n_muon);
    ggml_backend_sched_free(sched); lm_optim_free(&o); ggml_backend_buffer_free(sb); ggml_free(sc);
    return true;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) { std::fprintf(stderr, "no CPU backend\n"); return 1; }
    bool ok = true;
    for (const char * adapter : {"lora", "dora", "lokr"})
        for (const char * optimizer : {"adamw-lm", "prodigy", "muon"}) ok = one(cpu, adapter, optimizer) && ok;
    ggml_backend_free(cpu);
    std::puts(ok ? "PASS test-yue2-planner-freeze" : "FAIL test-yue2-planner-freeze");
    return ok ? 0 : 1;
}
