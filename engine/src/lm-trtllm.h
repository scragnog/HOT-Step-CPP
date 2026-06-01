#pragma once
// lm-trtllm.h — TRT-LLM Executor wrapper for Qwen3 4B LM
//
// Dual-mode architecture:
//   1. Full Decode: Executor runs its own decode loop (fastest).
//      LogitsPostProcessor applies FSM/audio masking on-GPU.
//   2. Single-Step Logits: For CFG — enqueue maxTokens=1 with
//      returnGenerationLogits, extract logits, combine on CPU.
//
// The Executor manages KV cache (paged), attention kernels (fused),
// and scheduling internally. No manual KV buffer management needed.

#ifdef HOT_STEP_TRTLLM

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tensorrt_llm/executor/executor.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace tle = tensorrt_llm::executor;

// ── Constants ────────────────────────────────────────────────────────────────

#define LM_TRTLLM_VOCAB        217204
#define LM_TRTLLM_MAX_SEQ      8192

// ── Plugin registration (dynamic load) ───────────────────────────────────────
// initTrtLlmPlugins must be called before Executor creation to register
// custom ops (GPTAttention, PagedKVCache, etc.) into the TensorRT PluginRegistry.
// We load it dynamically to avoid pulling the full DLL dependency chain at startup.

static bool s_trtllm_plugins_registered = false;

static bool register_trtllm_plugins() {
    if (s_trtllm_plugins_registered) return true;

#ifdef _WIN32
    HMODULE hMod = GetModuleHandleA("nvinfer_plugin_tensorrt_llm.dll");
    if (!hMod) {
        // Find the engine exe directory to locate DLL search paths
        char exePath[MAX_PATH];
        GetModuleFileNameA(NULL, exePath, MAX_PATH);
        std::string exeDir(exePath);
        auto slash = exeDir.find_last_of("\\/");
        if (slash != std::string::npos) exeDir = exeDir.substr(0, slash);

        // The plugin DLL depends on:
        //   tensorrt_llm.dll    → in engine/trtllm-libs/
        //   nvinfer_10.dll      → in engine/deps/tensorrt_libs/
        // We need BOTH directories in the DLL search order.
        // Use SetDefaultDllDirectories + AddDllDirectory for multi-dir support.
        std::string trtllmDir = exeDir + "\\..\\..\\trtllm-libs";
        std::string trtDir    = exeDir + "\\..\\..\\deps\\tensorrt_libs";

        // Enable AddDllDirectory-based search
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

        DLL_DIRECTORY_COOKIE cookie1 = nullptr, cookie2 = nullptr;

        // Convert to wide strings for AddDllDirectory
        auto toWide = [](const std::string& s) -> std::wstring {
            int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
            std::wstring ws(len, 0);
            MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &ws[0], len);
            return ws;
        };

        std::wstring wTrtllm = toWide(trtllmDir);
        std::wstring wTrt    = toWide(trtDir);
        cookie1 = AddDllDirectory(wTrtllm.c_str());
        cookie2 = AddDllDirectory(wTrt.c_str());

        if (cookie1) fprintf(stderr, "[LM-TRTLLM] Added DLL dir: %s\n", trtllmDir.c_str());
        if (cookie2) fprintf(stderr, "[LM-TRTLLM] Added DLL dir: %s\n", trtDir.c_str());

        // Load with full path, using the expanded search dirs for transitive deps
        std::string fullPath = trtllmDir + "\\nvinfer_plugin_tensorrt_llm.dll";
        hMod = LoadLibraryExA(fullPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

        if (!hMod) {
            // Try bare name as last resort
            hMod = LoadLibraryExA("nvinfer_plugin_tensorrt_llm.dll", NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        }

        // Clean up directory cookies (DLLs already loaded, deps resolved)
        if (cookie1) RemoveDllDirectory(cookie1);
        if (cookie2) RemoveDllDirectory(cookie2);
        // Restore default DLL search behavior
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    }
    if (!hMod) {
        DWORD err = GetLastError();
        fprintf(stderr, "[LM-TRTLLM] Cannot load nvinfer_plugin_tensorrt_llm.dll (error %lu)\n", err);
        return false;
    }

    typedef bool (*InitPluginsFn)(void*, const char*);
    auto fn = (InitPluginsFn)GetProcAddress(hMod, "initTrtLlmPlugins");
    if (!fn) {
        fprintf(stderr, "[LM-TRTLLM] Cannot find initTrtLlmPlugins in plugin DLL\n");
        return false;
    }

    bool ok = fn(nullptr, "tensorrt_llm");
    if (ok) {
        s_trtllm_plugins_registered = true;
        fprintf(stderr, "[LM-TRTLLM] TRT-LLM plugins registered\n");
    } else {
        fprintf(stderr, "[LM-TRTLLM] initTrtLlmPlugins returned false\n");
    }
    return ok;
#else
    fprintf(stderr, "[LM-TRTLLM] Plugin registration not implemented on this platform\n");
    return false;
#endif
}

// ── LmTrtLlm context ────────────────────────────────────────────────────────

struct LmTrtLlm {
    std::unique_ptr<tle::Executor> executor;

    // Config (read from engine config.json at load time)
    int vocab_size  = LM_TRTLLM_VOCAB;
    int max_seq_len = LM_TRTLLM_MAX_SEQ;

    // Stats
    int64_t load_time_ms = 0;
    bool    loaded       = false;
};

// ── Init / Shutdown ──────────────────────────────────────────────────────────

inline bool lm_trtllm_load(
    LmTrtLlm*  ctx,
    const char* engine_dir,
    int         max_seq_len = LM_TRTLLM_MAX_SEQ
) {
    auto t0 = std::chrono::steady_clock::now();

    std::filesystem::path engine_path(engine_dir);
    if (!std::filesystem::exists(engine_path)) {
        fprintf(stderr, "[LM-TRTLLM] Engine dir not found: %s\n", engine_dir);
        return false;
    }

    ctx->max_seq_len = max_seq_len;

    try {
        // Register TRT-LLM custom plugins (GPTAttention, PagedKVCache, etc.)
        // Must happen before engine deserialization.
        if (!register_trtllm_plugins()) {
            fprintf(stderr, "[LM-TRTLLM] Plugin registration failed — cannot load engine\n");
            return false;
        }

        // Executor config: single GPU, greedy/sampling, gather logits
        tle::ExecutorConfig exec_config(/*maxBeamWidth=*/1);

        // KV cache: use up to 70% of free GPU memory
        tle::KvCacheConfig kv_config;
        kv_config.setFreeGpuMemoryFraction(0.7f);
        exec_config.setKvCacheConfig(kv_config);

        // Enable generation logits gathering (required for CFG single-step mode)
        exec_config.setGatherGenerationLogits(true);

        fprintf(stderr, "[LM-TRTLLM] Creating Executor from %s ...\n", engine_dir);

        ctx->executor = std::make_unique<tle::Executor>(
            engine_path,
            tle::ModelType::kDECODER_ONLY,
            exec_config
        );

        if (!ctx->executor->canEnqueueRequests()) {
            fprintf(stderr, "[LM-TRTLLM] Executor created but cannot enqueue requests\n");
            return false;
        }

        ctx->loaded = true;
        auto t1 = std::chrono::steady_clock::now();
        ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        fprintf(stderr, "[LM-TRTLLM] Executor ready in %lld ms (vocab=%d, max_seq=%d)\n",
                (long long)ctx->load_time_ms, ctx->vocab_size, ctx->max_seq_len);
        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "[LM-TRTLLM] Executor init failed: %s\n", e.what());
        return false;
    }
}

inline void lm_trtllm_free(LmTrtLlm* ctx) {
    if (ctx->executor) {
        try {
            ctx->executor->shutdown();
        } catch (...) {}
        ctx->executor.reset();
    }
    ctx->loaded = false;
    fprintf(stderr, "[LM-TRTLLM] Shutdown complete\n");
}

// ── Full Decode Mode ─────────────────────────────────────────────────────────
//
// Let the Executor run its own decode loop with fused kernels.
// LogitsPostProcessor applies masking on-GPU before sampling.
// Returns generated token IDs (excluding prompt).
//
// This is the FASTEST path — no host↔device round-trips per token.

inline std::vector<int32_t> lm_trtllm_generate(
    LmTrtLlm*             ctx,
    const int*             prompt_tokens,
    int                    n_prompt,
    int                    max_new_tokens,
    float                  temperature,
    float                  top_p,
    int                    top_k,
    uint64_t               seed,
    int                    end_id,
    tle::LogitsPostProcessor mask_fn = nullptr
) {
    if (!ctx->loaded) return {};

    // Build token vector
    tle::VecTokens input(prompt_tokens, prompt_tokens + n_prompt);

    // Sampling config
    tle::SamplingConfig sampling(/*beamWidth=*/1);
    sampling.setTopK(top_k > 0 ? std::optional<tle::SizeType32>(top_k) : std::nullopt);
    sampling.setTopP(top_p > 0.0f && top_p < 1.0f ? std::optional<tle::FloatType>(top_p) : std::nullopt);
    sampling.setTemperature(temperature > 0.0f ? std::optional<tle::FloatType>(temperature) : std::nullopt);
    sampling.setSeed(std::optional<tle::RandomSeedType>(seed));

    // Output config — no logits needed in full decode mode
    tle::OutputConfig output_config;
    output_config.excludeInputFromOutput = true;

    // Build request
    tle::Request request(
        input,
        /*maxTokens=*/max_new_tokens,
        /*streaming=*/false,
        sampling,
        output_config,
        /*endId=*/std::optional<tle::SizeType32>(end_id)
    );

    // Attach logits post-processor if provided (for FSM/audio masking)
    if (mask_fn) {
        request.setLogitsPostProcessor(mask_fn);
    }

    try {
        auto t0 = std::chrono::steady_clock::now();

        auto request_id = ctx->executor->enqueueRequest(std::move(request));

        // Wait for completion (30s timeout — generous for long generations)
        auto responses = ctx->executor->awaitResponses(
            request_id,
            std::chrono::milliseconds(300000)  // 5 min for very long sequences
        );

        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        if (responses.empty()) {
            fprintf(stderr, "[LM-TRTLLM] No response (timeout)\n");
            return {};
        }

        auto& resp = responses[0];
        if (resp.hasError()) {
            fprintf(stderr, "[LM-TRTLLM] Error: %s\n", resp.getErrorMsg().c_str());
            return {};
        }

        auto result = resp.getResult();
        auto& output_tokens = result.outputTokenIds[0]; // beam 0

        fprintf(stderr, "[LM-TRTLLM] Generated %zu tokens in %lld ms (%.1f tok/s)\n",
                output_tokens.size(), (long long)ms,
                output_tokens.size() * 1000.0 / ms);

        return output_tokens;

    } catch (const std::exception& e) {
        fprintf(stderr, "[LM-TRTLLM] Generate failed: %s\n", e.what());
        return {};
    }
}

// ── Single-Step Logits Mode ──────────────────────────────────────────────────
//
// For CFG: enqueue the FULL token sequence (prompt + all generated so far)
// with maxTokens=1 and returnGenerationLogits=true. The Executor does a
// single forward pass and returns the logits for the next token.
//
// Caller extracts logits, does CFG combination, samples, and re-enqueues.
// This has per-token overhead but still uses fused attention kernels.
//
// Returns true on success, logits_out filled with [vocab_size] floats.

inline bool lm_trtllm_forward_logits(
    LmTrtLlm*    ctx,
    const int*    token_ids,
    int           n_tokens,
    float*        logits_out,    // [vocab_size] host output
    float         temperature,
    uint64_t      seed
) {
    if (!ctx->loaded) return false;

    tle::VecTokens input(token_ids, token_ids + n_tokens);

    // Sampling: temperature=1 to get undistorted logits
    // (the Executor will sample a token internally but we ignore it)
    tle::SamplingConfig sampling(/*beamWidth=*/1);
    sampling.setTemperature(1.0f);  // No temperature distortion on logits
    sampling.setTopK(1);            // Greedy — fastest, we don't use the sampled token
    sampling.setSeed(seed);

    // Output config: return generation logits
    tle::OutputConfig output_config;
    output_config.returnGenerationLogits = true;
    output_config.excludeInputFromOutput = true;

    tle::Request request(
        input,
        /*maxTokens=*/1,
        /*streaming=*/false,
        sampling,
        output_config
    );

    try {
        auto request_id = ctx->executor->enqueueRequest(std::move(request));

        auto responses = ctx->executor->awaitResponses(
            request_id,
            std::chrono::milliseconds(30000)
        );

        if (responses.empty()) {
            fprintf(stderr, "[LM-TRTLLM] Logits: no response (timeout)\n");
            return false;
        }

        auto& resp = responses[0];
        if (resp.hasError()) {
            fprintf(stderr, "[LM-TRTLLM] Logits error: %s\n", resp.getErrorMsg().c_str());
            return false;
        }

        auto result = resp.getResult();

        // Extract generation logits: shape [beamSize=1, maxTokens=1, vocabSizePadded]
        if (!result.generationLogits.has_value()) {
            fprintf(stderr, "[LM-TRTLLM] No generation logits returned "
                    "(engine built without --gather_generation_logits?)\n");
            return false;
        }

        const auto& logits_tensor = result.generationLogits.value();

        // The tensor data is on host (Executor copies it for us)
        // Shape: [1, 1, vocabSizePadded] — we want the last dim
        auto shape = logits_tensor.getShape();
        int64_t vocab_padded = shape[shape.size() - 1];
        int copy_count = (int)std::min((int64_t)ctx->vocab_size, vocab_padded);

        const float* src = static_cast<const float*>(logits_tensor.getData());
        memcpy(logits_out, src, copy_count * sizeof(float));

        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "[LM-TRTLLM] Forward logits failed: %s\n", e.what());
        return false;
    }
}

// ── Batch Logits Mode ────────────────────────────────────────────────────────
//
// For CFG: enqueue cond + uncond as two separate requests, await both.
// Returns logits for each in parallel. Faster than two sequential calls
// because the Executor can schedule them concurrently.
//
// cond_tokens/uncond_tokens: full token sequences (prompt + generated so far)
// logits_cond/logits_uncond: [vocab_size] host outputs

inline bool lm_trtllm_forward_logits_cfg(
    LmTrtLlm*    ctx,
    const int*    cond_tokens,
    int           n_cond,
    const int*    uncond_tokens,
    int           n_uncond,
    float*        logits_cond,     // [vocab_size] host output
    float*        logits_uncond,   // [vocab_size] host output
    uint64_t      seed
) {
    if (!ctx->loaded) return false;

    // Build both requests
    tle::SamplingConfig sampling(1);
    sampling.setTemperature(1.0f);
    sampling.setTopK(1);
    sampling.setSeed(seed);

    tle::OutputConfig output_config;
    output_config.returnGenerationLogits = true;
    output_config.excludeInputFromOutput = true;

    tle::VecTokens cond_input(cond_tokens, cond_tokens + n_cond);
    tle::VecTokens uncond_input(uncond_tokens, uncond_tokens + n_uncond);

    tle::Request cond_req(cond_input, 1, false, sampling, output_config);
    tle::Request uncond_req(uncond_input, 1, false, sampling, output_config);

    try {
        // Enqueue both — Executor can schedule them concurrently
        auto cond_id   = ctx->executor->enqueueRequest(std::move(cond_req));
        auto uncond_id = ctx->executor->enqueueRequest(std::move(uncond_req));

        // Await both
        std::vector<tle::IdType> ids = {cond_id, uncond_id};
        auto all_responses = ctx->executor->awaitResponses(
            ids, std::chrono::milliseconds(30000)
        );

        // all_responses[0] = cond responses, all_responses[1] = uncond responses
        if (all_responses.size() < 2) {
            fprintf(stderr, "[LM-TRTLLM] CFG: expected 2 response groups, got %zu\n",
                    all_responses.size());
            return false;
        }

        // Extract logits from each
        auto extract = [&](const std::vector<tle::Response>& resps, float* out) -> bool {
            if (resps.empty() || resps[0].hasError()) {
                if (!resps.empty())
                    fprintf(stderr, "[LM-TRTLLM] CFG error: %s\n", resps[0].getErrorMsg().c_str());
                return false;
            }
            auto result = resps[0].getResult();
            if (!result.generationLogits.has_value()) {
                fprintf(stderr, "[LM-TRTLLM] CFG: no logits returned\n");
                return false;
            }
            const auto& tensor = result.generationLogits.value();
            auto shape = tensor.getShape();
            int64_t vocab_padded = shape[shape.size() - 1];
            int copy_count = (int)std::min((int64_t)ctx->vocab_size, vocab_padded);
            memcpy(out, static_cast<const float*>(tensor.getData()), copy_count * sizeof(float));
            return true;
        };

        if (!extract(all_responses[0], logits_cond)) return false;
        if (!extract(all_responses[1], logits_uncond)) return false;

        return true;

    } catch (const std::exception& e) {
        fprintf(stderr, "[LM-TRTLLM] CFG forward failed: %s\n", e.what());
        return false;
    }
}

#endif // HOT_STEP_TRTLLM
