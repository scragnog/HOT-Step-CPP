#include <string>
#include "adapter-merge.h"
#include "minimax/mm3-lm-adapter.h"
#include "train/st-write.h"
#include "train/lm-pissa.h"
#include <chrono>
#include <filesystem>
#include <stdexcept>

static void jl(const char *, ...) {}

static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void write(const std::string & path, const std::vector<STWTensor> & tensors) {
    check(st_write_file(path.c_str(), tensors, {{"format", "pt"}}, STW_F32), "write safetensors");
}
int main(int argc, char ** argv) {
  try {
    if (argc > 1) {
        if (std::string(argv[1]) == "--fingerprint" && argc == 4) {
            STFile st; check(st_open(&st, argv[2]), "open residual for fingerprint");
            const auto fp = pissa_residual_fingerprint(st); st_close(&st);
            FILE * out = hs_fopen(argv[3], "wb"); check(out != nullptr, "fingerprint output");
            const bool written = fwrite(fp.data(), sizeof(float), fp.size(), out) == fp.size();
            const bool closed = fclose(out) == 0;
            check(written && closed, "write fingerprint bytes"); return 0;
        }
        if (std::string(argv[1]) == "--load" && (argc == 4 || argc == 5)) {
            const int runs = argc == 5 ? atoi(argv[4]) : 5;
            check(runs > 0 && runs <= 100, "benchmark run count");
            for (int i = 0; i < runs; ++i) {
                std::string err;
                auto start = std::chrono::steady_clock::now();
                auto * ad = mm3_lm_adapter_load(argv[2], &err, argv[3]);
                check(ad != nullptr, err.c_str());
                printf("adapter load ms=%.3f\n", std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now()-start).count());
                mm3_lm_adapter_free(ad);
            }
            return 0;
        }
        // Paired read benchmark, alternating arms after one warm-up.
        for (int i = 0; i < 9; ++i) {
            PissaResidualMeta m; std::vector<PissaResidualSite> sites; std::string err;
            const bool hash = i % 2 == 0;
            auto start = std::chrono::steady_clock::now();
            check(pissa_residual_read(argv[1], &m, &sites, &err, hash), err.c_str());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
            printf("residual read fingerprint=%d ms=%.3f bytes=%lld\n", hash, ms, pissa_file_size(argv[1]));
        }
        return 0;
    }
    check(pissa_xxh64("", 0) == 0xef46db3751d8e999ull, "XXH64 empty vector");
    check(pissa_xxh64("abc", 3) == 0x44bc2cf5ad770999ull, "XXH64 abc vector");
    uint8_t bytes[100]; for (int i = 0; i < 100; ++i) bytes[i] = uint8_t(i);
    check(pissa_xxh64(bytes, 31) == 0xc346d2b59b4d8ee1ull, "XXH64 short boundary vector");
    check(pissa_xxh64(bytes, 32) == 0xcbf59c5116ff32b4ull, "XXH64 block boundary vector");
    check(pissa_xxh64(bytes, 100) == 0x6ac1e58032166597ull, "XXH64 multi-block vector");
    const auto dir = std::filesystem::current_path() / ("pissa-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(dir);
    const std::string base = (dir / "base.gguf").u8string();
    const std::string residual = pissa_residual_path(base, 1);
    const std::string adapter = (dir / "adapter_model.safetensors").u8string();
    float rm[] = {1, 1, 9, 2, 0, 1, 0, 100, 0.5f, 0.5f};
    float a[] = {1, 2}, b[] = {3, 4}, delta[] = {0.1f, 0.2f};
    float pm[] = {1, 1, 0, 1, 0, 100}, marker = 4;
    auto residual_write = [&] { write(residual, {{"pissa.meta", {10}, rm},
        {"pissa.L0.self_attn.q_proj.A0", {1,2}, a}, {"pissa.L0.self_attn.q_proj.B0", {2,1}, b}}); };
    residual_write();
    STFile st; check(st_open(&st, residual.c_str()), "open residual");
    auto fingerprint = pissa_residual_fingerprint(st); st_close(&st);
    // Exercise the trainer's residual writer, including its F16 storage.
    PissaResidualMeta generated; generated.rank = 1; generated.q = 9;
    generated.iters = 2; generated.layer_hi = 1; generated.base_size = 100;
    PissaResidualSite site; site.site = "self_attn.q_proj"; site.in = site.out = 2;
    site.A0 = {1, 2}; site.B0 = {3, 4};
    std::string err;
    check(lm_pissa_residual_write(residual, generated, {site}, &err), err.c_str());
    check(st_open(&st, residual.c_str()), "open generated residual");
    fingerprint = pissa_residual_fingerprint(st); st_close(&st);
    PissaResidualMeta reread; std::vector<PissaResidualSite> sites;
    check(pissa_residual_read(residual, &reread, &sites, &err, true), err.c_str());
    check(fingerprint == reread.fingerprint, "generated/read fingerprint differs");
    // Export the real delta path, using the identity captured at init even
    // after the file has been replaced. No model weights or GPU are needed.
    LmLora bank; bank.rank = 1; bank.layer_hi = 1; bank.pissa = true;
    bank.pissa_residual = pissa_residual_name(base, 1); bank.pissa_base_size = 100;
    bank.pissa_fingerprint = fingerprint;
    bank.ctx = ggml_init({4096, nullptr, true});
    auto & pair = bank.layers[0].p[0];
    pair.A = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, 2, 1);
    pair.B = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, 1, 2);
    pair.A0 = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, 2, 1);
    pair.B0 = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, 1, 2);
    ggml_backend_load_all();
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "CPU backend");
    bank.buf = ggml_backend_alloc_ctx_tensors(bank.ctx, cpu);
    check(bank.buf != nullptr, "CPU tensor allocation");
    for (auto * t : {pair.A, pair.B, pair.A0, pair.B0}) ggml_backend_tensor_set(t, a, 0, sizeof(a));
    Qwen3LM model{}; bank.model = &model; bank.scale = 0.25f;
    LmPissaCacheSite cached; cached.in = cached.out = 2; cached.frac = 0.5;
    cached.A0 = {1, 2}; cached.B0 = {3, 4};
    const auto cache = (dir / "init.pissa").u8string();
    check(lm_pissa_cache_write(cache, 1, 9, 2, 0, 1, {cached}), "write SVD cache fixture");
    std::filesystem::remove(residual);
    LmPissaStats stats;
    check(lm_pissa_init(&bank, nullptr, 8, 2, &stats, &err, cache, residual, 100), err.c_str());
    check(pissa_residual_read(residual, &reread, &sites, &err, true), err.c_str());
    fingerprint = reread.fingerprint;
    check(bank.pissa_fingerprint == fingerprint && !bank.pissa_residual.empty(), "generating init lost identity");
    check(lm_pissa_init(&bank, nullptr, 8, 2, &stats, &err, cache, residual, 100), err.c_str());
    check(bank.pissa_fingerprint == fingerprint, "reading init changed identity at non-unit scale");
    check(lm_pissa_cache_write(cache, 1, 9, 1, 0, 1, {cached}), "write fallback cache fixture");
    check(lm_pissa_init(&bank, nullptr, 8, 1, &stats, &err, cache, residual, 100), err.c_str());
    check(bank.pissa_residual.empty() && bank.pissa_fingerprint.empty(), "fallback selected delta export");
    check(pissa_residual_read(residual, &reread, &sites, &err, true), err.c_str());
    check(reread.fingerprint == fingerprint, "fallback overwrote an existing residual");
    bank.pissa_residual = pissa_residual_name(base, 1); bank.pissa_fingerprint = fingerprint;
    const auto exported = (dir / "exported").u8string();
    LmExportMeta export_meta; export_meta.lm_path = base;
    LmExportResult result;
    a[1] = -a[1]; residual_write();
    check(lm_export_pissa_delta(bank, {}, export_meta, exported, &result, &err), err.c_str());
    check(st_open(&st, (exported+"/adapter_model.safetensors").c_str()), "open delta export");
    const auto * fp = st_find(st, "hot_step.pissa.fingerprint");
    check(fp && fp->dtype == "F32" && fp->shape[0] == 5, "F16 export lost fingerprint dtype/shape");
    check(memcmp(st_data(st, *fp), fingerprint.data(), 20) == 0, "export re-hashed replaced file");
    st_close(&st);
    auto * exported_ad = mm3_lm_adapter_load((exported+"/adapter_model.safetensors").c_str(), &err, base.c_str());
    check(exported_ad == nullptr, "real delta export accepted replacement residual");
    ggml_backend_buffer_free(bank.buf); ggml_free(bank.ctx); ggml_backend_free(cpu);
    a[1] = -a[1];
    residual_write();
    check(st_open(&st, residual.c_str()), "restore test residual");
    fingerprint = pissa_residual_fingerprint(st); st_close(&st);
    const std::string prefix = "language_model.model.layers.0.self_attn.q_proj";
    std::vector<STWTensor> tensors = {{prefix+".lora_A.weight", {1,2}, delta},
        {prefix+".lora_B.weight", {2,1}, delta}, {"hot_step.param_method", {1}, &marker},
        {"hot_step.pissa.meta", {6}, pm}, {"hot_step.pissa.fingerprint", {5}, fingerprint.data()}};
    write(adapter, tensors);
    auto load = [&] { err.clear(); return mm3_lm_adapter_load(adapter.c_str(), &err, base.c_str()); };
    auto * ad = load(); check(ad != nullptr, err.c_str()); mm3_lm_adapter_free(ad);
    // Same dimensions, metadata, filename and length, different frozen basis.
    a[1] = -a[1]; residual_write();
    ad = load();
    if (ad) { mm3_lm_adapter_free(ad); throw std::runtime_error("mismatched residual ACCEPTED"); }
    check(err.find(adapter) != std::string::npos && err.find(residual) != std::string::npos,
          "mismatch error must name both files");
    check(err.find("retrain") != std::string::npos, "mismatch error needs recovery guidance");
    // Legacy adapters deliberately retain the former checks.
    tensors.pop_back(); write(adapter, tensors);
    ad = load(); check(ad != nullptr, err.c_str()); mm3_lm_adapter_free(ad);
    // A present but malformed fingerprint must never fall back to legacy.
    tensors.push_back({"hot_step.pissa.fingerprint", {4}, fingerprint.data()}); write(adapter, tensors);
    ad = load(); check(ad == nullptr, "malformed fingerprint accepted");
    std::filesystem::remove_all(dir);
    puts("PASS PiSSA residual: generated/read init, retained export identity, mismatch, legacy and malformed fingerprint on CPU");
    return 0;
  } catch (const std::exception & e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
