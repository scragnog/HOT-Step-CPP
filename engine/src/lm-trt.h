#pragma once
// lm-trt.h — TensorRT runtime for Qwen3 4B autoregressive LM
//
// Single full-vocab engine. Phase 2 audio-code logit slicing
// is handled in C++ (offset = 151645), not in the ONNX graph.
//
// KV Cache Strategy:
//   Pre-allocate [batch, n_kv_heads, max_seq_len, head_dim] per layer.
//   Double-buffer: past_kv → engine → present_kv, then swap pointers.
//   kv_pos[set] tracks the valid length per KV set.
//
// Build (once per GPU arch):
//   ONNX → TRT engine with kREFIT_IDENTICAL | kSTRIP_PLAN
//   → ~50-100MB stripped engine file (weights in ONNX sidecar)
//
// Runtime:
//   Load engine → Refit with base weights → Run

#ifdef HOT_STEP_TRT

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <chrono>
#include <algorithm>

#include "NvInfer.h"
#include "NvOnnxParser.h"

// ── Constants ───────────────────────────────────────────────────────────────

#define LM_TRT_MAX_LAYERS   36
#define LM_TRT_MAX_KV_SETS  32   // Matches QW3LM_MAX_KV_SETS
#define LM_TRT_N_KV_HEADS    8
#define LM_TRT_HEAD_DIM    128
#define LM_TRT_HIDDEN     2560
#define LM_TRT_VOCAB    217204

// Phase 2 partial vocab: logits[offset:] = audio codes
#define LM_TRT_PARTIAL_OFFSET 151645

// ── TRT Logger (shared with dit-trt.h if both included) ────────────────────

#ifndef HOT_STEP_TRT_LOGGER_DEFINED
#define HOT_STEP_TRT_LOGGER_DEFINED
class LmTrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
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
#endif

// ── LmTrt context ──────────────────────────────────────────────────────────

struct LmTrt {
    // TRT objects
    nvinfer1::IRuntime*            runtime  = nullptr;
    nvinfer1::ICudaEngine*         engine   = nullptr;
    nvinfer1::IExecutionContext*   context  = nullptr;

    // Config
    int n_layers    = LM_TRT_MAX_LAYERS;
    int n_kv_heads  = LM_TRT_N_KV_HEADS;
    int head_dim    = LM_TRT_HEAD_DIM;
    int vocab_size  = LM_TRT_VOCAB;
    int max_seq_len = 8192;

    // KV cache: single-buffer GPU memory per kv_set per layer
    // Each buffer: [1, n_kv_heads, max_seq_len, head_dim] in bf16
    // After each forward, new KV tokens are appended at kv_pos offset.
    void*  d_kv_key[LM_TRT_MAX_KV_SETS][LM_TRT_MAX_LAYERS] = {};
    void*  d_kv_val[LM_TRT_MAX_KV_SETS][LM_TRT_MAX_LAYERS] = {};

    // Scratch buffers for present KV output (new tokens only)
    // Size: [1, n_kv_heads, max_profile_seq, head_dim] bf16
    // These receive the TRT present_key/value outputs, then get
    // cudaMemcpy'd into the main KV buffer at kv_pos.
    void*  d_present_key_scratch[LM_TRT_MAX_LAYERS] = {};
    void*  d_present_val_scratch[LM_TRT_MAX_LAYERS] = {};

    int    kv_pos[LM_TRT_MAX_KV_SETS] = {};
    int    n_kv_sets = 0;

    // Scratch GPU buffers for inputs/outputs
    void*  d_input_ids    = nullptr;  // [max_batch, max_seq] int64
    void*  d_position_ids = nullptr;  // [max_batch, max_seq] int64
    void*  d_attn_mask    = nullptr;  // [max_batch, max_total] int64
    void*  d_logits       = nullptr;  // [max_batch, max_seq, vocab] fp32

    // ONNX path (for weight refitting)
    std::string onnx_path;

    // Base weight cache for adapter refit
    std::unordered_map<std::string, std::vector<uint16_t>> base_weights;
    std::string current_adapter;
    std::mutex  refit_mutex;
    std::unordered_set<std::string> weights_transposed;

    // CUDA stream for inference
    cudaStream_t stream = nullptr;

    // Pre-cached tensor name strings (computed once at load, avoids snprintf per forward)
    char tn_past_key   [LM_TRT_MAX_LAYERS][64];
    char tn_past_val   [LM_TRT_MAX_LAYERS][64];
    char tn_present_key[LM_TRT_MAX_LAYERS][64];
    char tn_present_val[LM_TRT_MAX_LAYERS][64];

    // Logger
    LmTrtLogger logger;

    // Stats
    int64_t build_time_ms = 0;
    int64_t load_time_ms  = 0;
};

// ── KV buffer helpers ──────────────────────────────────────────────────────

static inline size_t lm_trt_kv_buf_bytes(const LmTrt* ctx) {
    // [1, n_kv_heads, max_seq_len, head_dim] in bf16
    return (size_t)1 * ctx->n_kv_heads * ctx->max_seq_len * ctx->head_dim * 2;
}

// Size of present KV scratch buffer (max profile seq_len = 2048)
static inline size_t lm_trt_present_scratch_bytes(const LmTrt* ctx) {
    return (size_t)1 * ctx->n_kv_heads * 2048 * ctx->head_dim * 2;
}


// ── Engine build ───────────────────────────────────────────────────────────

inline bool lm_trt_build(
    const char* onnx_path,
    const char* engine_path,
    int         max_seq_len = 8192,
    int         device_id = 0
) {
    LmTrtLogger logger;

    fprintf(stderr, "[LM-TRT] Building engine from %s ...\n", onnx_path);
    fprintf(stderr, "[LM-TRT] This will take 10-60 minutes (first run only).\n");
    auto t0 = std::chrono::steady_clock::now();

    cudaSetDevice(device_id);

    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) {
        fprintf(stderr, "[LM-TRT] Failed to create TRT builder\n");
        return false;
    }

    // STRONGLY_TYPED: TRT honors per-tensor dtypes from dynamo ONNX
    uint32_t net_flags = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)
      | 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = builder->createNetworkV2(net_flags);
    if (!network) {
        fprintf(stderr, "[LM-TRT] Failed to create network\n");
        delete builder;
        return false;
    }
    fprintf(stderr, "[LM-TRT] STRONGLY_TYPED network (bf16_mixed from dynamo)\n");

    // Parse ONNX
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnx_path,
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "[LM-TRT] ONNX parse failed\n");
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Builder config
    auto config = builder->createBuilderConfig();

    // TF32 for fp32 island acceleration; no FP16/BF16 flags (STRONGLY_TYPED)
    config->setFlag(nvinfer1::BuilderFlag::kTF32);
    fprintf(stderr, "[LM-TRT] STRONGLY_TYPED + TF32\n");

    // Builder optimization level 5 (TRT-LLM default) — exhaustive kernel search
    config->setBuilderOptimizationLevel(5);
    fprintf(stderr, "[LM-TRT] Optimization level 5\n");

    // Refittable engine (for adapter/LoRA support)
    config->setFlag(nvinfer1::BuilderFlag::kREFIT_IDENTICAL);
    config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
    fprintf(stderr, "[LM-TRT] kREFIT_IDENTICAL + kSTRIP_PLAN enabled\n");

    // Workspace — 8GB for 4B model (lots of large matmuls)
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               8ULL << 30);

    // ── Optimization profiles ──────────────────────────────────────────────
    //
    // Single profile with wide ranges. TRT will auto-tune.
    //
    // Inputs:
    //   input_ids:      [batch, seq_len]                        int64
    //   position_ids:   [batch, seq_len]                        int64
    //   attention_mask: [batch, total_len]                      int64
    //   past_key_N:     [batch, n_kv, past_seq_len, head_dim]   bf16
    //   past_value_N:   [batch, n_kv, past_seq_len, head_dim]   bf16
    //

    auto profile = builder->createOptimizationProfile();

    const int nkv = LM_TRT_N_KV_HEADS;
    const int hd  = LM_TRT_HEAD_DIM;

    // input_ids: [1, S]  (batch is fixed at 1 in the ONNX export)
    // opt=1: autoregressive decode (1 token at a time) is the hot path.
    // Prefill (S=300-700) uses the same profile but at non-optimal dimensions.
    profile->setDimensions("input_ids",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions("input_ids",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 1));
    profile->setDimensions("input_ids",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 2048));

    // position_ids: [1, S]
    profile->setDimensions("position_ids",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions("position_ids",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 1));
    profile->setDimensions("position_ids",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, 2048));

    // attention_mask: [1, total_len]  (total = past + seq)
    profile->setDimensions("attention_mask",
        nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
    profile->setDimensions("attention_mask",
        nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(1, 512));
    profile->setDimensions("attention_mask",
        nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(1, max_seq_len));

    // KV cache tensors: [1, nkv, past_seq, hd]
    // past_seq_len min=1 (TRT can't do 0-length dims in some cases)
    for (int l = 0; l < LM_TRT_MAX_LAYERS; l++) {
        char kname[64], vname[64];
        snprintf(kname, sizeof(kname), "past_key_%d", l);
        snprintf(vname, sizeof(vname), "past_value_%d", l);

        profile->setDimensions(kname,
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, nkv, 1, hd));
        profile->setDimensions(kname,
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, nkv, 512, hd));
        profile->setDimensions(kname,
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, nkv, max_seq_len, hd));

        profile->setDimensions(vname,
            nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, nkv, 1, hd));
        profile->setDimensions(vname,
            nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, nkv, 512, hd));
        profile->setDimensions(vname,
            nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, nkv, max_seq_len, hd));
    }

    config->addOptimizationProfile(profile);

    // Build serialized engine
    fprintf(stderr, "[LM-TRT] Building serialized engine...\n");
    auto serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        fprintf(stderr, "[LM-TRT] Engine build failed\n");
        delete config;
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    // Write to disk
    FILE* f = fopen(engine_path, "wb");
    if (!f) {
        fprintf(stderr, "[LM-TRT] Cannot write to %s\n", engine_path);
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

    fprintf(stderr, "[LM-TRT] Engine saved to %s (%zu bytes, %.1f min)\n",
            engine_path, serialized->size(), ms / 60000.0);

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true;
}

// ── Engine load + base weight refit ─────────────────────────────────────────

inline bool lm_trt_load(
    LmTrt*      ctx,
    const char* engine_path,
    const char* onnx_path,
    int         max_seq_len = 8192,
    int         n_kv_sets   = 4,
    int         device_id   = 0
) {
    auto t0 = std::chrono::steady_clock::now();
    cudaSetDevice(device_id);

    ctx->onnx_path   = onnx_path;
    ctx->max_seq_len = max_seq_len;
    ctx->n_kv_sets   = std::min(n_kv_sets, (int)LM_TRT_MAX_KV_SETS);

    // Read engine file
    FILE* f = fopen(engine_path, "rb");
    if (!f) {
        fprintf(stderr, "[LM-TRT] Cannot open engine %s\n", engine_path);
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
        fprintf(stderr, "[LM-TRT] Failed to create TRT runtime\n");
        return false;
    }

    ctx->engine = ctx->runtime->deserializeCudaEngine(
        engine_data.data(), engine_size);
    if (!ctx->engine) {
        fprintf(stderr, "[LM-TRT] Failed to deserialize engine\n");
        return false;
    }
    fprintf(stderr, "[LM-TRT] Engine loaded (%zu bytes)\n", engine_size);

    // Refit with base weights from ONNX
    auto refitter = nvinfer1::createInferRefitter(*ctx->engine, ctx->logger);
    if (!refitter) {
        fprintf(stderr, "[LM-TRT] Failed to create refitter\n");
        return false;
    }

    auto parser_refitter = nvonnxparser::createParserRefitter(*refitter, ctx->logger);
    if (!parser_refitter->refitFromFile(onnx_path)) {
        fprintf(stderr, "[LM-TRT] Parser refit from ONNX failed\n");
        delete parser_refitter;
        delete refitter;
        return false;
    }

    if (!refitter->refitCudaEngine()) {
        fprintf(stderr, "[LM-TRT] Engine refit failed\n");
        delete parser_refitter;
        delete refitter;
        return false;
    }

    // Cache base weights for adapter revert
    int32_t num_weights = refitter->getAllWeights(0, nullptr);
    if (num_weights > 0) {
        std::vector<const char*> names(num_weights);
        refitter->getAllWeights(num_weights, names.data());

        for (int32_t i = 0; i < num_weights; i++) {
            auto w = refitter->getNamedWeights(names[i]);
            if ((w.type == nvinfer1::DataType::kBF16 ||
                 w.type == nvinfer1::DataType::kHALF) && w.count > 0) {
                const uint16_t* data = static_cast<const uint16_t*>(w.values);
                ctx->base_weights[names[i]] =
                    std::vector<uint16_t>(data, data + w.count);
            }
        }
        fprintf(stderr, "[LM-TRT] Cached %zu base weight tensors for refit\n",
                ctx->base_weights.size());
    }

    delete parser_refitter;
    delete refitter;

    // Create execution context
    ctx->context = ctx->engine->createExecutionContext();
    if (!ctx->context) {
        fprintf(stderr, "[LM-TRT] Failed to create execution context\n");
        return false;
    }

    // Log I/O tensors with names, modes, and shapes
    int num_io = ctx->engine->getNbIOTensors();
    fprintf(stderr, "[LM-TRT] Engine has %d I/O tensors:\n", num_io);
    for (int i = 0; i < num_io; i++) {
        const char* name = ctx->engine->getIOTensorName(i);
        auto mode = ctx->engine->getTensorIOMode(name);
        const char* modeStr = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        auto dims = ctx->engine->getTensorShape(name);
        char dimStr[256] = {};
        int off = 0;
        for (int d = 0; d < dims.nbDims; d++) {
            off += snprintf(dimStr + off, sizeof(dimStr) - off, "%s%lld",
                           d > 0 ? "×" : "", (long long)dims.d[d]);
        }
        // Only print first 5 and last 5 for brevity (skip middle if >10)
        if (num_io <= 20 || i < 5 || i >= num_io - 5) {
            fprintf(stderr, "  [%d] %-6s %-30s  [%s]\n", i, modeStr, name, dimStr);
        } else if (i == 5) {
            fprintf(stderr, "  ... (%d more tensors) ...\n", num_io - 10);
        }
    }

    // Allocate KV cache buffers (single-buffer per set/layer)
    size_t kv_bytes = lm_trt_kv_buf_bytes(ctx);
    fprintf(stderr, "[LM-TRT] Allocating KV cache: %d sets × %d layers × 1 buffer × %.1f MB\n",
            ctx->n_kv_sets, ctx->n_layers,
            kv_bytes / (1024.0 * 1024.0));

    for (int s = 0; s < ctx->n_kv_sets; s++) {
        for (int l = 0; l < ctx->n_layers; l++) {
            cudaMalloc(&ctx->d_kv_key[s][l], kv_bytes);
            cudaMalloc(&ctx->d_kv_val[s][l], kv_bytes);
        }
        ctx->kv_pos[s] = 0;
    }

    // Present KV scratch buffers (shared across all KV sets, one per layer)
    // Size: [1, n_kv_heads, 2048, head_dim] bf16 — matches max profile seq_len
    size_t present_bytes = lm_trt_present_scratch_bytes(ctx);
    for (int l = 0; l < ctx->n_layers; l++) {
        cudaMalloc(&ctx->d_present_key_scratch[l], present_bytes);
        cudaMalloc(&ctx->d_present_val_scratch[l], present_bytes);
    }

    // Allocate scratch buffers for inputs/outputs
    // input_ids, position_ids: [1, max_seq=2048] int64 — matches profile max
    size_t ids_bytes = 1 * 2048 * sizeof(int64_t);
    cudaMalloc(&ctx->d_input_ids,    ids_bytes);
    cudaMalloc(&ctx->d_position_ids, ids_bytes);

    // attention_mask: [1, max_seq_len] int64
    // Pre-filled with all 1s since causal masking is inside the ONNX graph.
    // This avoids per-forward host→device copies of the mask.
    size_t mask_bytes = 1 * max_seq_len * sizeof(int64_t);
    cudaMalloc(&ctx->d_attn_mask, mask_bytes);
    {
        std::vector<int64_t> ones(max_seq_len, 1);
        cudaMemcpy(ctx->d_attn_mask, ones.data(), mask_bytes, cudaMemcpyHostToDevice);
    }

    // logits: [1, max_seq=2048, vocab] fp32 — must match profile max seq_len
    size_t logits_bytes = 1ULL * 2048 * ctx->vocab_size * sizeof(float);
    cudaMalloc(&ctx->d_logits, logits_bytes);

    // Zero all KV cache buffers — first forward reads past_len=1 of
    // uninitialized memory if kv_pos=0 (profile min=1 workaround)
    for (int s = 0; s < ctx->n_kv_sets; s++) {
        for (int l = 0; l < ctx->n_layers; l++) {
            cudaMemset(ctx->d_kv_key[s][l], 0, kv_bytes);
            cudaMemset(ctx->d_kv_val[s][l], 0, kv_bytes);
        }
    }

    // Create CUDA stream for inference
    cudaStreamCreate(&ctx->stream);

    size_t total_gpu_mb = (
        ctx->n_kv_sets * ctx->n_layers * 2 * kv_bytes +  // 2 buffers (K+V) per set/layer
        ctx->n_layers * 2 * present_bytes +               // present scratch
        ids_bytes * 2 + mask_bytes + logits_bytes
    ) / (1024 * 1024);

    auto t1 = std::chrono::steady_clock::now();
    ctx->load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    fprintf(stderr, "[LM-TRT] Load complete: %lld ms, ~%zu MB GPU\n",
            (long long)ctx->load_time_ms, total_gpu_mb);
    ctx->current_adapter.clear();

    // Pre-cache tensor name strings for the forward pass hot loop
    for (int l = 0; l < ctx->n_layers; l++) {
        snprintf(ctx->tn_past_key[l],    64, "past_key_%d", l);
        snprintf(ctx->tn_past_val[l],    64, "past_value_%d", l);
        snprintf(ctx->tn_present_key[l], 64, "present_key_%d", l);
        snprintf(ctx->tn_present_val[l], 64, "present_value_%d", l);
    }

    return true;
}

// ── KV cache management ────────────────────────────────────────────────────

inline void lm_trt_reset_kv(LmTrt* ctx, int kv_set) {
    ctx->kv_pos[kv_set] = 0;
    // No need to zero: kv_pos tracks valid range
}

inline void lm_trt_reset_all_kv(LmTrt* ctx) {
    for (int s = 0; s < ctx->n_kv_sets; s++) {
        lm_trt_reset_kv(ctx, s);
    }
}

inline void lm_trt_copy_kv(LmTrt* ctx, int src, int dst) {
    // Only copy the valid portion (kv_pos tokens, not entire max_seq_len)
    int pos = ctx->kv_pos[src];
    if (pos <= 0) { ctx->kv_pos[dst] = 0; return; }
    size_t valid_bytes = (size_t)ctx->n_kv_heads * pos * ctx->head_dim * 2;
    for (int l = 0; l < ctx->n_layers; l++) {
        cudaMemcpy(ctx->d_kv_key[dst][l], ctx->d_kv_key[src][l],
                   valid_bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ctx->d_kv_val[dst][l], ctx->d_kv_val[src][l],
                   valid_bytes, cudaMemcpyDeviceToDevice);
    }
    ctx->kv_pos[dst] = pos;
}

// ── Forward pass ───────────────────────────────────────────────────────────

// Run one LM forward pass (prefill or decode).
// token_ids:  host array of token IDs, length n_tokens
// kv_set:     which KV cache set to use
// logits_out: host buffer for output logits [vocab_size] or [partial_vocab]
// partial_offset: if > 0, only copy logits[offset:] (Phase 2)
// sync:       if true, synchronize stream before returning (default).
//             Set false for batched CFG to defer sync to caller.
//
// Returns true on success.
inline bool lm_trt_forward(
    LmTrt*       ctx,
    const int*   token_ids,     // host, length n_tokens
    int          n_tokens,
    int          kv_set,
    float*       logits_out,    // host output
    int          partial_offset = 0,  // 0 = full vocab, >0 = slice
    bool         sync = true
) {
    int kv_pos  = ctx->kv_pos[kv_set];
    int kv_len  = kv_pos + n_tokens;  // total sequence length after this forward
    int batch   = 1;

    auto* context = ctx->context;
    cudaStream_t s = ctx->stream;

    // Prepare input_ids: [1, n_tokens] int64
    // Use stack for small decode (≤16 tokens), heap for prefill
    if (n_tokens <= 16) {
        int64_t ids[16];
        for (int i = 0; i < n_tokens; i++) ids[i] = token_ids[i];
        cudaMemcpyAsync(ctx->d_input_ids, ids,
                        n_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice, s);
    } else {
        std::vector<int64_t> ids(n_tokens);
        for (int i = 0; i < n_tokens; i++) ids[i] = token_ids[i];
        cudaMemcpyAsync(ctx->d_input_ids, ids.data(),
                        n_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice, s);
    }

    // Prepare position_ids: [1, n_tokens] int64
    if (n_tokens <= 16) {
        int64_t pos[16];
        for (int i = 0; i < n_tokens; i++) pos[i] = kv_pos + i;
        cudaMemcpyAsync(ctx->d_position_ids, pos,
                        n_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice, s);
    } else {
        std::vector<int64_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; i++) pos[i] = kv_pos + i;
        cudaMemcpyAsync(ctx->d_position_ids, pos.data(),
                        n_tokens * sizeof(int64_t),
                        cudaMemcpyHostToDevice, s);
    }

    // attention_mask: already pre-filled with all 1s on device at load time

    // Set input shapes
    context->setInputShape("input_ids",      nvinfer1::Dims2(batch, n_tokens));
    context->setInputShape("position_ids",   nvinfer1::Dims2(batch, n_tokens));
    context->setInputShape("attention_mask",  nvinfer1::Dims2(batch, kv_len));

    // Set input tensor addresses
    context->setTensorAddress("input_ids",     ctx->d_input_ids);
    context->setTensorAddress("position_ids",  ctx->d_position_ids);
    context->setTensorAddress("attention_mask", ctx->d_attn_mask);

    // Set KV cache input/output shapes and addresses
    // past_seq_len for first forward (kv_pos=0) needs special handling:
    // TRT profile min is 1, so we use 1 even when kv_pos=0 and fill with zeros
    int past_len = std::max(kv_pos, 1);
    nvinfer1::Dims4 past_kv_shape(batch, ctx->n_kv_heads, past_len, ctx->head_dim);

    // Present KV output shape: [1, n_kv_heads, n_tokens, head_dim]
    // (new ONNX export outputs only the new tokens, not full history)

    for (int l = 0; l < ctx->n_layers; l++) {
        // Use pre-cached names (no snprintf in hot loop)
        context->setInputShape(ctx->tn_past_key[l], past_kv_shape);
        context->setInputShape(ctx->tn_past_val[l], past_kv_shape);

        // Past: read from the single KV buffer
        context->setTensorAddress(ctx->tn_past_key[l], ctx->d_kv_key[kv_set][l]);
        context->setTensorAddress(ctx->tn_past_val[l], ctx->d_kv_val[kv_set][l]);

        // Present: write to scratch buffers (NOT back to the KV buffer)
        context->setTensorAddress(ctx->tn_present_key[l], ctx->d_present_key_scratch[l]);
        context->setTensorAddress(ctx->tn_present_val[l], ctx->d_present_val_scratch[l]);
    }

    // Logits output
    context->setTensorAddress("logits", ctx->d_logits);

    // Execute (no validation in release — we checked at load time)
    bool ok = context->enqueueV3(s);
    if (!ok) {
        fprintf(stderr, "[LM-TRT] enqueueV3 failed\n");
        return false;
    }

    // Append present KV into the single buffer at kv_pos.
    // present output: [1, n_kv_heads, n_tokens, head_dim] bf16
    // Must be copied into kv_buf[head_h][kv_pos:kv_pos+n_tokens][:]
    //
    // The KV buffer is contiguous: [n_kv_heads, max_seq_len, head_dim]
    // We need to insert n_tokens rows at position kv_pos for each head.
    // Since the head stride is max_seq_len * head_dim, and we're inserting
    // at a different offset per head, we need per-head copies.
    //
    // present output is [n_kv_heads, n_tokens, head_dim] contiguous.
    // Destination in KV: head_h * max_seq * hd + kv_pos * hd
    size_t hd_bytes = (size_t)ctx->head_dim * 2;  // bf16
    size_t row_bytes = (size_t)n_tokens * hd_bytes;
    size_t kv_head_stride = (size_t)ctx->max_seq_len * hd_bytes;
    size_t present_head_stride = (size_t)n_tokens * hd_bytes;

    for (int l = 0; l < ctx->n_layers; l++) {
        char* dst_k = (char*)ctx->d_kv_key[kv_set][l];
        char* dst_v = (char*)ctx->d_kv_val[kv_set][l];
        char* src_k = (char*)ctx->d_present_key_scratch[l];
        char* src_v = (char*)ctx->d_present_val_scratch[l];

        for (int h = 0; h < ctx->n_kv_heads; h++) {
            size_t dst_off = h * kv_head_stride + (size_t)kv_pos * hd_bytes;
            size_t src_off = h * present_head_stride;
            cudaMemcpyAsync(dst_k + dst_off, src_k + src_off, row_bytes,
                           cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(dst_v + dst_off, src_v + src_off, row_bytes,
                           cudaMemcpyDeviceToDevice, s);
        }
    }

    ctx->kv_pos[kv_set] = kv_len;

    // Copy logits to host (async on the same stream)
    if (partial_offset > 0) {
        // Phase 2: only copy logits[last_token, offset:]
        int out_vocab = ctx->vocab_size - partial_offset;
        size_t offset_bytes = (size_t)(n_tokens - 1) * ctx->vocab_size + partial_offset;
        cudaMemcpyAsync(logits_out,
                        (float*)ctx->d_logits + offset_bytes,
                        out_vocab * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
    } else {
        // Full vocab: copy last token's logits
        size_t offset = (size_t)(n_tokens - 1) * ctx->vocab_size;
        cudaMemcpyAsync(logits_out,
                        (float*)ctx->d_logits + offset,
                        ctx->vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost, s);
    }

    // Only sync if caller requests it (batched CFG defers sync)
    if (sync) {
        cudaStreamSynchronize(s);
    }

    return true;
}

// ── Batched forward (CFG decode) ───────────────────────────────────────────

// Batched single-token decode for CFG.
// Each element has its own kv_set with potentially different kv_pos.
// token_ids: [N] tokens (one per batch element)
// kv_sets:   [N] kv set indices
// logits_out: [N, vocab_size] or [N, partial_vocab] host buffer
//
// NOTE: TRT requires all batch elements to have the same past_seq_len
// within a single enqueue. If kv_pos values differ, we run N separate
// forwards. If they're all equal, we batch into one call.
inline bool lm_trt_forward_batch(
    LmTrt*       ctx,
    const int*   token_ids,     // host, [N]
    const int*   kv_sets,       // [N] kv set indices
    int          N,
    float*       logits_out,    // host, [N, vocab] or [N, partial_vocab]
    int          partial_offset = 0
) {
    int out_vocab = partial_offset > 0 ? ctx->vocab_size - partial_offset
                                       : ctx->vocab_size;

    // Run each element, deferring sync until the last one.
    // This halves the number of GPU stalls for CFG (N=2).
    for (int i = 0; i < N; i++) {
        bool is_last = (i == N - 1);
        if (!lm_trt_forward(ctx, &token_ids[i], 1, kv_sets[i],
                            logits_out + i * out_vocab,
                            partial_offset, /*sync=*/is_last)) {
            return false;
        }
    }
    return true;
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

inline void lm_trt_free(LmTrt* ctx) {
    // Sync all pending work before tearing down TRT resources.
    // Without this, GGML's CUDA backend may crash after TRT cleanup
    // because TRT's stream/context destruction can corrupt shared CUDA state.
    if (ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
    }
    cudaDeviceSynchronize();

    // Free KV cache (single-buffer)
    for (int s = 0; s < LM_TRT_MAX_KV_SETS; s++) {
        for (int l = 0; l < LM_TRT_MAX_LAYERS; l++) {
            if (ctx->d_kv_key[s][l]) { cudaFree(ctx->d_kv_key[s][l]); ctx->d_kv_key[s][l] = nullptr; }
            if (ctx->d_kv_val[s][l]) { cudaFree(ctx->d_kv_val[s][l]); ctx->d_kv_val[s][l] = nullptr; }
        }
    }

    // Free present KV scratch buffers
    for (int l = 0; l < LM_TRT_MAX_LAYERS; l++) {
        if (ctx->d_present_key_scratch[l]) { cudaFree(ctx->d_present_key_scratch[l]); ctx->d_present_key_scratch[l] = nullptr; }
        if (ctx->d_present_val_scratch[l]) { cudaFree(ctx->d_present_val_scratch[l]); ctx->d_present_val_scratch[l] = nullptr; }
    }

    // Free scratch buffers
    if (ctx->d_input_ids)    { cudaFree(ctx->d_input_ids);    ctx->d_input_ids = nullptr; }
    if (ctx->d_position_ids) { cudaFree(ctx->d_position_ids); ctx->d_position_ids = nullptr; }
    if (ctx->d_attn_mask)    { cudaFree(ctx->d_attn_mask);    ctx->d_attn_mask = nullptr; }
    if (ctx->d_logits)       { cudaFree(ctx->d_logits);       ctx->d_logits = nullptr; }

    // TRT objects
    if (ctx->context) { delete ctx->context; ctx->context = nullptr; }
    if (ctx->engine)  { delete ctx->engine;  ctx->engine = nullptr; }
    if (ctx->runtime) { delete ctx->runtime; ctx->runtime = nullptr; }

    // CUDA stream
    if (ctx->stream) { cudaStreamDestroy(ctx->stream); ctx->stream = nullptr; }

    ctx->base_weights.clear();
}

#endif // HOT_STEP_TRT
