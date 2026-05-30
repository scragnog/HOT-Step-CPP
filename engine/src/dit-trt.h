#pragma once
// dit-trt.h — TensorRT native API wrapper for DiT inference + LoRA refitting
//
// Uses TRT directly (not ORT) because:
//  - IRefitter is required for runtime LoRA adapter switching
//  - ORT's TRT EP doesn't expose kREFIT_IDENTICAL or IRefitter API
//
// Architecture:
//  BUILD (once per GPU arch):
//    ONNX → TRT engine with kREFIT_IDENTICAL | kSTRIP_PLAN | kFP16
//    → ~50MB stripped engine file (weights stored separately in ONNX)
//
//  RUNTIME:
//    Load engine → Refit with base weights from ONNX → Run
//    On adapter switch: refit with merged weights (~0.5s)

#ifdef HOT_STEP_TRT

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <chrono>

// TRT headers
#include "NvInfer.h"
#include "NvOnnxParser.h"

// ── TRT Logger ──────────────────────────────────────────────────────────────

class DitTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Skip INFO/VERBOSE unless debugging
        if (severity > Severity::kWARNING) return;
        const char* prefix = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: prefix = "[TRT-INTERNAL] "; break;
            case Severity::kERROR:          prefix = "[TRT-ERROR] ";    break;
            case Severity::kWARNING:        prefix = "[TRT-WARN] ";     break;
            default: break;
        }
        fprintf(stderr, "%s%s\n", prefix, msg);
    }
};

// ── DitTrt context ──────────────────────────────────────────────────────────

struct DitTrt {
    // TRT objects
    nvinfer1::IRuntime*            runtime  = nullptr;
    nvinfer1::ICudaEngine*         engine   = nullptr;
    nvinfer1::IExecutionContext*   context  = nullptr;

    // I/O tensor indices (resolved once at load time)
    // Inputs:  input_latents[B,T,192], enc_hidden[B,S,2048], t[B], t_r[B]
    // Outputs: velocity[B,T,64]
    int idx_input_latents = -1;
    int idx_enc_hidden    = -1;
    int idx_t             = -1;
    int idx_t_r           = -1;
    int idx_velocity      = -1;

    // Device buffers (allocated lazily, resized as needed)
    void*  d_input_latents = nullptr;
    void*  d_enc_hidden    = nullptr;
    void*  d_t             = nullptr;
    void*  d_t_r           = nullptr;
    void*  d_velocity      = nullptr;
    size_t buf_input_latents_bytes = 0;
    size_t buf_enc_hidden_bytes    = 0;
    size_t buf_velocity_bytes      = 0;

    // ONNX path (needed for weight refitting)
    std::string onnx_path;

    // Base weight cache (BF16 host memory, keyed by TRT weight name)
    // Populated on first load; used to revert adapter changes
    std::unordered_map<std::string, std::vector<uint16_t>> base_weights;

    // Current adapter state
    std::string current_adapter;  // empty = base model
    std::mutex  refit_mutex;

    // Logger
    DitTrtLogger logger;

    // Stats
    int64_t build_time_ms = 0;
    int64_t load_time_ms  = 0;
};

// ── Engine build (once per GPU architecture) ────────────────────────────────

// Build a TRT engine from ONNX and serialize to disk.
// Returns true on success.  engine_path will be created/overwritten.
//
// This is slow (5-30 minutes) but only needs to run once per GPU arch.
// The engine is built with:
//   - kREFIT_IDENTICAL: allows weight refitting with zero inference penalty
//   - kSTRIP_PLAN: strips weights from engine file (~50MB vs ~5GB)
//   - kFP16: FP16 inference
inline bool dit_trt_build(
    const char* onnx_path,
    const char* engine_path,
    int         device_id = 0
) {
    DitTrtLogger logger;

    fprintf(stderr, "[DiT-TRT] Building engine from %s ...\n", onnx_path);
    fprintf(stderr, "[DiT-TRT] This will take 5-30 minutes (first run only).\n");
    auto t0 = std::chrono::steady_clock::now();

    // Set CUDA device
    cudaSetDevice(device_id);

    // Create builder
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        fprintf(stderr, "[DiT-TRT] Failed to create TRT builder\n");
        return false;
    }

    // Create network — standard EXPLICIT_BATCH, NO STRONGLY_TYPED.
    // STRONGLY_TYPED: TRT honors the per-tensor dtypes from the ONNX graph.
    // The dynamo-exported bf16_mixed ONNX has bf16 trunk + fp32 islands.
    // TRT runs bf16 tensor cores for matmuls and fp32 for norms/residuals.
    uint32_t net_flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
      | 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = builder->createNetworkV2(net_flags);
    if (!network) {
        fprintf(stderr, "[DiT-TRT] Failed to create network\n");
        delete builder;
        return false;
    }
    fprintf(stderr, "[DiT-TRT] STRONGLY_TYPED network (bf16_mixed from dynamo)\n");

    // Parse ONNX
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_path,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[DiT-TRT] ONNX parse failed\n");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Builder config
    auto config = builder->createBuilderConfig();

    // STRONGLY_TYPED + TF32: graph types are authoritative.
    // TF32 accelerates fp32 island ops on tensor cores.
    // No FP16/BF16 builder flags — STRONGLY_TYPED forbids them.
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
    fprintf(stderr, "[DiT-TRT] STRONGLY_TYPED + TF32 (bf16_mixed from dynamo ONNX)\n");

    // Enable refittable engine (zero perf penalty with IDENTICAL)
    config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
    config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    fprintf(stderr, "[DiT-TRT] kREFIT_IDENTICAL + kSTRIP_PLAN enabled\n");

    // Workspace (4 GB should be plenty for DiT)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               4ULL << 30);

    // Optimization profile with dynamic shapes
    auto profile = builder->createOptimizationProfile();

    // input_latents: [B, T, 192]
    //   T ranges: min=64, opt=2048, max=8192
    profile->setDimensions("input_latents",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 64, 192));
    profile->setDimensions("input_latents",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 2048, 192));
    profile->setDimensions("input_latents",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 8192, 192));

    // enc_hidden: [B, S, 2048]
    //   S ranges: min=64, opt=512, max=2048
    profile->setDimensions("enc_hidden",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 64, 2048));
    profile->setDimensions("enc_hidden",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2048));
    profile->setDimensions("enc_hidden",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(2, 2048, 2048));

    // t: [B]
    profile->setDimensions("t",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
    profile->setDimensions("t",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {1}});
    profile->setDimensions("t",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {2}});

    // t_r: [B]
    profile->setDimensions("t_r",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
    profile->setDimensions("t_r",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {1}});
    profile->setDimensions("t_r",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {2}});

    config->addOptimizationProfile(profile);

    // Build serialized engine
    auto serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        fprintf(stderr, "[DiT-TRT] Engine build failed\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Write to disk
    FILE* f = fopen(engine_path, "wb");
    if (!f) {
        fprintf(stderr, "[DiT-TRT] Cannot write to %s\n", engine_path);
        delete serialized;
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }
    fwrite(serialized->data(), 1, serialized->size(), f);
    fclose(f);

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    fprintf(stderr, "[DiT-TRT] Engine saved to %s (%zu bytes, %.1f min)\n",
            engine_path, serialized->size(), ms / 60000.0);

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

// ── Engine load + base weight refit ─────────────────────────────────────────

// Load a pre-built TRT engine and refit with base weights from the ONNX file.
// The ONNX path is needed because kSTRIP_PLAN engines have no weights embedded.
inline bool dit_trt_load(
    DitTrt*     ctx,
    const char* engine_path,
    const char* onnx_path,
    int         device_id = 0
) {
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(device_id);
    ctx->onnx_path = onnx_path;

    // Read engine file
    FILE* f = fopen(engine_path, "rb");
    if (!f) {
        fprintf(stderr, "[DiT-TRT] Cannot open engine %s\n", engine_path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    size_t engine_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> engine_data(engine_size);
    fread(engine_data.data(), 1, engine_size, f);
    fclose(f);

    // Deserialize
    ctx->runtime = nvinfer1::createInferRuntime(ctx->logger);
    if (!ctx->runtime) {
        fprintf(stderr, "[DiT-TRT] Failed to create TRT runtime\n");
        return false;
    }

    ctx->engine = ctx->runtime->deserializeCudaEngine(
        engine_data.data(), engine_size);
    if (!ctx->engine) {
        fprintf(stderr, "[DiT-TRT] Failed to deserialize engine\n");
        return false;
    }

    fprintf(stderr, "[DiT-TRT] Engine loaded (%zu bytes)\n", engine_size);

    // Refit with base weights from ONNX
    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) {
        fprintf(stderr, "[DiT-TRT] Failed to create refitter\n");
        return false;
    }

    // Use parser refitter to auto-load weights from ONNX
    // TRT 10: refitFromFile takes only the path (no severity arg)
    auto parser_refitter = nvonnxparser::createParserRefitter(*refitter, ctx->logger);
    if (!parser_refitter->refitFromFile(onnx_path)) {
        fprintf(stderr, "[DiT-TRT] Parser refit from ONNX failed\n");
        delete parser_refitter;
        delete refitter;
        return false;
    }

    if (!refitter->refitCudaEngine()) {
        fprintf(stderr, "[DiT-TRT] Engine refit failed\n");
        delete parser_refitter;
        delete refitter;
        return false;
    }

    // Cache base weights for later adapter revert
    // TRT 10: use getAllWeights (not getAll with LayerRole)
    int32_t num_weights = refitter->getAllWeights(0, nullptr);
    if (num_weights > 0) {
        std::vector<const char*> names(num_weights);
        refitter->getAllWeights(num_weights, names.data());

        for (int32_t i = 0; i < num_weights; i++) {
            auto w = refitter->getNamedWeights(names[i]);
            if ((w.type == nvinfer1::DataType::kBF16 || w.type == nvinfer1::DataType::kHALF) && w.count > 0) {
                const uint16_t* data = static_cast<const uint16_t*>(w.values);
                ctx->base_weights[names[i]] =
                    std::vector<uint16_t>(data, data + w.count);
            }
        }
        fprintf(stderr, "[DiT-TRT] Cached %zu base weight tensors for refit\n",
                ctx->base_weights.size());
    }

    delete parser_refitter;
    delete refitter;

    // Create execution context
    ctx->context = ctx->engine->createExecutionContext();
    if (!ctx->context) {
        fprintf(stderr, "[DiT-TRT] Failed to create execution context\n");
        return false;
    }

    // Resolve I/O tensor indices and log dtypes
    int num_io = ctx->engine->getNbIOTensors();
    for (int i = 0; i < num_io; i++) {
        const char* name = ctx->engine->getIOTensorName(i);
        auto dtype = ctx->engine->getTensorDataType(name);
        auto mode = ctx->engine->getTensorIOMode(name);
        const char* dtype_str = "unknown";
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:  dtype_str = "fp32"; break;
            case nvinfer1::DataType::kHALF:   dtype_str = "fp16"; break;
            case nvinfer1::DataType::kBF16:   dtype_str = "bf16"; break;
            case nvinfer1::DataType::kINT32:  dtype_str = "int32"; break;
            case nvinfer1::DataType::kINT8:   dtype_str = "int8"; break;
            case nvinfer1::DataType::kBOOL:   dtype_str = "bool"; break;
            default: break;
        }
        const char* io_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        fprintf(stderr, "[DiT-TRT] IO[%d] %-20s %s  %s\n", i, name, io_str, dtype_str);
        
        if (std::string(name) == "input_latents") ctx->idx_input_latents = i;
        else if (std::string(name) == "enc_hidden") ctx->idx_enc_hidden = i;
        else if (std::string(name) == "t")          ctx->idx_t = i;
        else if (std::string(name) == "t_r")        ctx->idx_t_r = i;
        else if (std::string(name) == "velocity")   ctx->idx_velocity = i;
    }

    if (ctx->idx_input_latents < 0 || ctx->idx_enc_hidden < 0 ||
        ctx->idx_t < 0 || ctx->idx_t_r < 0 || ctx->idx_velocity < 0) {
        fprintf(stderr, "[DiT-TRT] Missing I/O tensors!\n");
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();
    ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Load + refit complete (%lld ms)\n",
            (long long)ctx->load_time_ms);
    ctx->current_adapter.clear();

    return true;
}

// ── LoRA adapter refitting ──────────────────────────────────────────────────

// Refit engine with LoRA adapter deltas.
// `deltas` maps TRT weight names to merged weight data (W_base + delta).
// Each value should point to BF16 data with the same element count as base.
//
// Returns refit time in milliseconds.
inline int64_t dit_trt_refit_adapter(
    DitTrt*     ctx,
    const std::string& adapter_name,
    const std::unordered_map<std::string, const void*>& merged_weights
) {
    std::lock_guard<std::mutex> lock(ctx->refit_mutex);
    auto t0 = std::chrono::steady_clock::now();

    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) {
        fprintf(stderr, "[DiT-TRT] Failed to create refitter for adapter\n");
        return -1;
    }

    int updated = 0;
    for (const auto& [name, data] : merged_weights) {
        auto it = ctx->base_weights.find(name);
        if (it == ctx->base_weights.end()) {
            fprintf(stderr, "[DiT-TRT] Warning: weight '%s' not in base cache\n",
                    name.c_str());
            continue;
        }

        nvinfer1::Weights w;
        w.type   = nvinfer1::DataType::kBF16;
        w.values = data;
        w.count  = static_cast<int64_t>(it->second.size());

        if (!refitter->setNamedWeights(name.c_str(), w)) {
            fprintf(stderr, "[DiT-TRT] Failed to set weight '%s'\n",
                    name.c_str());
        } else {
            updated++;
        }
    }

    if (!refitter->refitCudaEngine()) {
        fprintf(stderr, "[DiT-TRT] Adapter refit failed\n");
        delete refitter;
        return -1;
    }

    delete refitter;
    ctx->current_adapter = adapter_name;

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Adapter '%s' applied: %d weights refitted (%lld ms)\n",
            adapter_name.c_str(), updated, (long long)ms);
    return ms;
}

// Revert to base weights (remove adapter)
inline int64_t dit_trt_refit_base(DitTrt* ctx) {
    std::lock_guard<std::mutex> lock(ctx->refit_mutex);
    if (ctx->current_adapter.empty()) return 0;  // already base

    auto t0 = std::chrono::steady_clock::now();

    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) return -1;

    for (const auto& [name, data] : ctx->base_weights) {
        nvinfer1::Weights w;
        w.type   = nvinfer1::DataType::kBF16;
        w.values = data.data();
        w.count  = static_cast<int64_t>(data.size());
        refitter->setNamedWeights(name.c_str(), w);
    }

    bool ok = refitter->refitCudaEngine();
    delete refitter;

    if (!ok) {
        fprintf(stderr, "[DiT-TRT] Base refit failed!\n");
        return -1;
    }

    ctx->current_adapter.clear();

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr, "[DiT-TRT] Reverted to base weights (%lld ms)\n",
            (long long)ms);
    return ms;
}

// ── Single forward pass ─────────────────────────────────────────────────────

// Run one DiT forward pass (one diffusion timestep).
// All pointers are GPU device memory.
//
// input_latents: [N, T, 192] fp16
// enc_hidden:    [N, S, 2048] fp16
// t:             [N] fp32
// t_r:           [N] fp32
// velocity_out:  [N, T, 64] fp16 (output)
inline bool dit_trt_forward(
    DitTrt*     ctx,
    const void* input_latents,  // GPU, fp16 [N, T, 192]
    const void* enc_hidden,     // GPU, fp16 [N, S, 2048]
    const float* t,             // GPU, fp32 [N]
    const float* t_r,           // GPU, fp32 [N]
    int N, int T, int S,
    void* velocity_out,         // GPU, fp16 [N, T, 64]
    cudaStream_t stream = nullptr
) {
    auto* context = ctx->context;

    // Set input shapes
    const char* input_latents_name = ctx->engine->getIOTensorName(ctx->idx_input_latents);
    const char* enc_hidden_name    = ctx->engine->getIOTensorName(ctx->idx_enc_hidden);
    const char* t_name             = ctx->engine->getIOTensorName(ctx->idx_t);
    const char* t_r_name           = ctx->engine->getIOTensorName(ctx->idx_t_r);
    const char* velocity_name      = ctx->engine->getIOTensorName(ctx->idx_velocity);

    context->setInputShape(input_latents_name, nvinfer1::Dims3(N, T, 192));
    context->setInputShape(enc_hidden_name,    nvinfer1::Dims3(N, S, 2048));
    context->setInputShape(t_name,             nvinfer1::Dims{1, {N}});
    context->setInputShape(t_r_name,           nvinfer1::Dims{1, {N}});

    // Set tensor addresses (TRT 10: setTensorAddress takes void*, cast away const)
    context->setTensorAddress(input_latents_name, const_cast<void*>(input_latents));
    context->setTensorAddress(enc_hidden_name,    const_cast<void*>(enc_hidden));
    context->setTensorAddress(t_name,             const_cast<void*>(static_cast<const void*>(t)));
    context->setTensorAddress(t_r_name,           const_cast<void*>(static_cast<const void*>(t_r)));
    context->setTensorAddress(velocity_name,      velocity_out);

    // Enqueue on stream
    bool ok = context->enqueueV3(stream ? stream : 0);
    if (!ok) {
        fprintf(stderr, "[DiT-TRT] enqueueV3 failed\n");
    }
    return ok;
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

inline void dit_trt_free(DitTrt* ctx) {
    if (ctx->d_input_latents) { cudaFree(ctx->d_input_latents); ctx->d_input_latents = nullptr; }
    if (ctx->d_enc_hidden)    { cudaFree(ctx->d_enc_hidden);    ctx->d_enc_hidden = nullptr; }
    if (ctx->d_t)             { cudaFree(ctx->d_t);             ctx->d_t = nullptr; }
    if (ctx->d_t_r)           { cudaFree(ctx->d_t_r);           ctx->d_t_r = nullptr; }
    if (ctx->d_velocity)      { cudaFree(ctx->d_velocity);      ctx->d_velocity = nullptr; }
    // TRT 10: use delete instead of destroy()
    if (ctx->context)         { delete ctx->context;            ctx->context = nullptr; }
    if (ctx->engine)          { delete ctx->engine;             ctx->engine = nullptr; }
    if (ctx->runtime)         { delete ctx->runtime;            ctx->runtime = nullptr; }
    ctx->base_weights.clear();
}

#endif // HOT_STEP_TRT
