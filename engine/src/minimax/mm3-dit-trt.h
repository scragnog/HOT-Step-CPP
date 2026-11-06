#pragma once
// Native MM3 DiT forward runtime. Sampling, CFG and stitching remain in MM3.
// The prepared engine is immutable; cache misses refit the selected GGUF weights.
#include "mm3-model.h"
#include "yyjson.h"

#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#ifdef HOT_STEP_TRT
#    include <cuda_runtime_api.h>
#    include <NvInfer.h>
#    include <NvOnnxParser.h>
#    include <atomic>
#    include <thread>
#endif
#include "../trt-runtime-probe.h"

// ── Files ───────────────────────────────────────────────────────────────────
//
// What a user downloads (Model Manager, Hugging Face):
//   <models>/mm3/mm3-dit-trt.onnx          the DiT graph, weightless
//   <models>/mm3/mm3-dit-trt.engine.json   GGUF-name -> engine-weight manifest
// What this machine builds once per GPU + TensorRT version:
//   <models>/mm3/mm3-trt-cache/base-<key>.engine
// and then refits per selected DiT GGUF + adapter stack:
//   <models>/mm3/mm3-trt-cache/<hash>.engine
// A prebuilt engine can still be supplied for development through
// MM3_DIT_TRT_ENGINE (its manifest is <engine>.json) or by dropping
// mm3-dit-trt.engine next to the ONNX; those take precedence over building.

static std::filesystem::path mm3_trt_dir(const MM3Model & m) {
    return std::filesystem::path(m.models_dir) / "mm3";
}

static std::string mm3_trt_onnx_path(const MM3Model & m) {
    return (mm3_trt_dir(m) / "mm3-dit-trt.onnx").string();
}

// A prebuilt engine, if one was supplied. Empty when the engine must be built.
static std::string mm3_trt_prebuilt_engine_path(const MM3Model & m) {
    const char * path = std::getenv("MM3_DIT_TRT_ENGINE");
    if (path && *path) {
        return path;
    }
    std::error_code ec;
    const auto      legacy = (mm3_trt_dir(m) / "mm3-dit-trt.engine").string();
    return std::filesystem::is_regular_file(legacy, ec) ? legacy : std::string();
}

static std::string mm3_trt_manifest_path(const MM3Model & m) {
    const char * path = std::getenv("MM3_DIT_TRT_ENGINE");
    if (path && *path) {
        return std::string(path) + ".json";
    }
    return (mm3_trt_dir(m) / "mm3-dit-trt.engine.json").string();
}

// Opt-in FP8 MLP engine (not ear-validated): request field dit_precision "fp8"
// (MM3Model::dit_fp8), or MM3_DIT_TRT_FP8=1 as a dev override that forces it on,
// builds a separate base engine whose ff_in/ff_out GEMMs (blocks 2..33) run in FP8
// E4M3, and keys the refit cache apart. Activation scales come from a calibration
// file made by tools/mm3-trt-export/fp8_lab.py (the server ships one and passes its
// path as dit_fp8_calib); weight scales are computed at refit.
static bool mm3_trt_fp8(const MM3Model & m) {
    const char * v = std::getenv("MM3_DIT_TRT_FP8");
    return m.dit_fp8 || (v && *v == '1');
}

static std::string mm3_trt_fp8_calib_path(const MM3Model & m) {
    return m.dit_fp8_calib.empty() ? (mm3_trt_dir(m) / "mm3-dit-trt.fp8.json").string() : m.dit_fp8_calib;
}

// Everything /mm3/props needs to explain why TensorRT is or is not usable,
// tier by tier: build -> runtime DLLs -> downloaded assets -> CUDA device.
struct MM3TrtStatus {
    bool        supported = false;   // compiled with HOT_STEP_TRT
    bool        cuda      = false;   // a CUDA device and the CUDA ggml backend
    int         sm        = 0;       // device compute capability as major*10+minor
    bool        rt_nvinfer          = false;
    bool        rt_parser           = false;
    bool        rt_builder_resource = false;
    bool        onnx      = false;
    bool        manifest  = false;
    bool        engine    = false;   // a usable engine exists (prebuilt or built for this key)
    bool        available = false;
    bool        needs_build = false; // available, but the first render builds the engine
    std::string reason;
};

#ifdef HOT_STEP_TRT
static std::filesystem::path mm3_trt_base_engine_path(const MM3Model & m, int device);
#endif

static MM3TrtStatus mm3_trt_status(const MM3Model & m) {
    MM3TrtStatus s;
#ifdef HOT_STEP_TRT
    s.supported = true;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        s.reason = "TensorRT requires an available NVIDIA CUDA device";
        return s;
    }
    int device = 0;
    if (m.backend) {
        const char * backend_name = ggml_backend_name(m.backend);
        if (!backend_name || sscanf(backend_name, "CUDA%d", &device) != 1) {
            s.reason = "TensorRT requires the CUDA engine backend";
            return s;
        }
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        s.sm = properties.major * 10 + properties.minor;
    }
    s.cuda = true;
    const auto rt = hot_step_trt_runtime_probe(s.sm);
    s.rt_nvinfer          = rt.nvinfer;
    s.rt_parser           = rt.parser;
    s.rt_builder_resource = rt.builder_resource;
    std::error_code ec;
    const auto      prebuilt = mm3_trt_prebuilt_engine_path(m);
    s.onnx     = std::filesystem::is_regular_file(mm3_trt_onnx_path(m), ec);
    s.manifest = std::filesystem::is_regular_file(mm3_trt_manifest_path(m), ec);
    s.engine   = !prebuilt.empty();
    if (!s.rt_nvinfer) {
        s.reason = "TensorRT runtime (nvinfer_10.dll) is not installed; get it from the Model Manager";
        return s;
    }
    if (!s.manifest) {
        s.reason = "The MM3 TensorRT weight manifest is not installed; get it from the Model Manager";
        return s;
    }
    if (mm3_trt_fp8(m) && !std::filesystem::is_regular_file(mm3_trt_fp8_calib_path(m), ec)) {
        s.reason = "TensorRT FP8 needs its calibration file " + mm3_trt_fp8_calib_path(m);
        return s;
    }
    if (!s.engine) {
        // No prebuilt engine: we need the ONNX plus the parser and builder
        // resources to make one, unless a base engine for this GPU already exists.
        s.engine = std::filesystem::is_regular_file(mm3_trt_base_engine_path(m, device), ec);
        if (!s.engine) {
            if (!s.onnx) {
                s.reason = "The MM3 TensorRT DiT graph (mm3-dit-trt.onnx) is not installed; get it from the Model Manager";
                return s;
            }
            if (!s.rt_parser) {
                s.reason = "TensorRT ONNX parser (nvonnxparser_10.dll) is not installed; get it from the Model Manager";
                return s;
            }
            if (!s.rt_builder_resource) {
                s.reason = "TensorRT builder resources for this GPU (nvinfer_builder_resource_sm" +
                           std::to_string(s.sm) + "_10.dll) are not installed; get them from the Model Manager";
                return s;
            }
            s.needs_build = true;
        }
    }
    s.available = true;
#else
    (void) m;
    s.reason = "This engine build does not include TensorRT";
#endif
    return s;
}

// Host copy of the last refitted engine plan read from mm3-trt-cache. With
// keep-loaded off the runtime (and its GPU memory) is dropped after every job,
// so the next job would read the ~4.5 GB file again: ~3 s warm, and 35-45 s
// when the read ran at ~130 MB/s (seen 2026-11-05). Holding the bytes leaves
// only deserialize + context (~0.7 s) per job, and no VRAM is held between
// jobs. Keyed by path + size + mtime; the file name already hashes the whole
// refit identity. HOST memory, so like the AR slot it is not dropped by
// mm3_unload() (that runs after every job) but by POST /mm3/unload, a model
// selection change and a switch to the GGML renderer. Guarded by g_mm3_mutex.
// ponytail: one slot, no RAM-size check; add one if low-RAM users run TensorRT.
struct MM3TrtPlanCache {
    std::string             key;
    std::unique_ptr<char[]> bytes;
    size_t                  size = 0;
};

static MM3TrtPlanCache g_mm3_trt_plan;

static void mm3_trt_plan_drop(const char * why) {
    if (!g_mm3_trt_plan.bytes) {
        return;
    }
    fprintf(stderr, "[MM3-TRT] Dropped %.0f MB host engine plan (%s)\n", double(g_mm3_trt_plan.size) / 1048576.0,
            why);
    g_mm3_trt_plan = MM3TrtPlanCache{};
}

static bool mm3_trt_available(const MM3Model & m, std::string * reason) {
    const auto s = mm3_trt_status(m);
    if (reason) {
        *reason = s.reason;
    }
    return s.available;
}

#ifdef HOT_STEP_TRT
namespace mm3_trt {
static void check(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void cuda_check(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Logger : nvinfer1::ILogger {
    void log(Severity severity, const char * message) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[MM3-TRT] %s\n", message);
        }
    }
};

struct Buffer {
    float * data = nullptr;

    void allocate(size_t count) { cuda_check(cudaMalloc((void **) &data, count * sizeof(float)), "allocate inputs"); }

    ~Buffer() {
        if (data) {
            cudaFree(data);
        }
    }
};

struct DeviceScope {
    int previous = 0;

    explicit DeviceScope(int device) {
        cuda_check(cudaGetDevice(&previous), "get device");
        cuda_check(cudaSetDevice(device), "set device");
    }

    ~DeviceScope() { cudaSetDevice(previous); }
};

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return uint16_t((bits >> 16) | 0x40U);
    }
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

static uint64_t fnv1a(const std::string & text) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}
}  // namespace mm3_trt

// The base engine is what the ONNX builds into on this GPU with this TensorRT.
// Engines are locked to both, so the key is TensorRT version + device model +
// ONNX identity. Driver version is deliberately left out: TensorRT engines
// survive driver updates, and rebuilding for one would cost minutes for nothing.
static std::filesystem::path mm3_trt_base_engine_path(const MM3Model & m, int device) {
    std::ostringstream identity;
    identity << "mm3-trt-base-v1\n" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '\n';
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        identity << properties.name << ':' << properties.major << ':' << properties.minor << '\n';
    }
    std::error_code ec;
    const auto      onnx = std::filesystem::absolute(mm3_trt_onnx_path(m)).lexically_normal();
    identity << onnx.string() << '\n' << std::filesystem::file_size(onnx, ec) << '\n'
             << std::filesystem::last_write_time(onnx, ec).time_since_epoch().count() << '\n';
    const bool fp8 = mm3_trt_fp8(m);
    if (fp8) {
        identity << "fp8-mlp-v2\n";
    }
    std::ostringstream name;
    name << (fp8 ? "base-fp8-" : "base-") << std::hex << mm3_trt::fnv1a(identity.str()) << ".engine";
    return mm3_trt_dir(m) / "mm3-trt-cache" / name.str();
}

namespace mm3_trt {

class Runtime final : public MM3DitRuntime {
    Logger                                       logger;
    int                                          device = 0;
    cudaStream_t                                 stream = nullptr;
    std::unique_ptr<nvinfer1::IRuntime>          runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    Buffer                                       dx, dc, dt, drc, drs, dy;
    std::vector<float>                           basis, condition, bx, bc, fourier, cosine, sine, velocity;
    int                                          cached_L       = 0;
    int                                          uploaded_B     = 0;
    float                                        uploaded_gate  = -1.0f;
    size_t                                       resident_bytes = 0;

    std::filesystem::path cache_path(const MM3Model & m, const std::string & source, const std::string & manifest) {
        // File identity matches the model/adapter cache convention elsewhere in
        // MM3. Sampler, duration, seed and conditioning deliberately stay out.
        std::ostringstream identity;
        auto               add_file = [&](const std::string & name) {
            const auto path = std::filesystem::absolute(name).lexically_normal();
            identity << path.string() << '\n'
                     << std::filesystem::file_size(path) << '\n'
                     << std::filesystem::last_write_time(path).time_since_epoch().count() << '\n';
        };
        identity << "mm3-trt-refit-cache-v1\n"
                 << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '\n';
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device), "identify GPU");
        int driver = 0;
        cuda_check(cudaDriverGetVersion(&driver), "identify driver");
        identity << properties.name << ':' << properties.major << ':' << properties.minor << ':' << driver << '\n';
        add_file(source);
        add_file(manifest);
        add_file(m.role_file[MM3_R_DIT].path);
        if (mm3_trt_fp8(m)) {
            identity << "fp8-mlp-v2\n";
            add_file(mm3_trt_fp8_calib_path(m));
        }
        identity << m.rest_adapter_desc << '\n';
        // Keep the exact requested scale too: rest_adapter_desc is formatted
        // for display and rounds floats to six decimal places.
        for (const char * key : { "MM3_ADAPTER", "MM3_ADAPTER_SCALE" }) {
            const char * value = std::getenv(key);
            identity << key << '=' << (value ? value : "") << '\n';
        }
        size_t begin = 0;
        while (begin < m.rest_adapter_desc.size()) {
            const auto end   = m.rest_adapter_desc.find("; ", begin);
            const auto entry = m.rest_adapter_desc.substr(begin, end == std::string::npos ? end : end - begin);
            const auto at    = entry.rfind('@');
            check(at != std::string::npos, "Invalid merged adapter identity");
            const std::filesystem::path adapter(entry.substr(0, at));
            const bool                  directory = std::filesystem::is_directory(adapter);
            const auto                  weights   = directory ? adapter / "adapter_model.safetensors" : adapter;
            const auto                  config = (directory ? adapter : adapter.parent_path()) / "adapter_config.json";
            add_file(weights.string());
            identity << "adapter_config=" << config.string() << '\n';
            if (std::filesystem::is_regular_file(config)) {
                add_file(config.string());
                // Alpha is a merge input. Hash the small config contents as well
                // as metadata, including when a tool preserves its timestamp.
                std::ifstream config_file(config, std::ios::binary);
                check(bool(config_file), "Cannot read adapter configuration for cache identity");
                identity << config_file.rdbuf() << '\n';
            } else {
                identity << "absent\n";
            }
            if (end == std::string::npos) {
                break;
            }
            begin = end + 2;
        }
        std::ostringstream name;
        name << std::hex << fnv1a(identity.str()) << ".engine";
        return mm3_trt_dir(m) / "mm3-trt-cache" / name.str();
    }

    // Build the base engine from the weightless ONNX. One shared B1..2 profile
    // over L 3..689 (the largest window the sampler ever asks for), general
    // kREFIT because the refit source (a GGUF, an adapter stack) differs from
    // whatever the ONNX carries. A heartbeat keeps the server's stall watchdog
    // informed: on a slow GPU this is minutes.
    // The shipped ONNX is weightless: its initializers point at an external
    // data file that is never downloaded. Every weight is refit from the GGUF
    // after the build, so the bytes only have to be well-formed BF16 that
    // TensorRT cannot fold (no zeros) or deduplicate (no two tensors alike).
    // The pattern is a pure function of the byte offset, the same one
    // tools/mm3-trt-export/make_placeholder_data.py writes; a weak hash here
    // would repeat every 256 KB and walk straight into TensorRT's
    // "weights of same values but of different counts" error.
    static void write_placeholder_data(const std::filesystem::path & path, uint64_t total_bytes) {
        check(total_bytes % 2 == 0, "Placeholder data size must be even");
        check(std::filesystem::space(path.parent_path()).available > total_bytes + 256ULL * 1024 * 1024,
              "Insufficient disk space to stage the MM3 TensorRT build (" +
                  std::to_string(total_bytes / (1024 * 1024)) + " MiB needed)");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        check(bool(out), "Cannot create " + path.string());
        const size_t          chunk_elems = 32ULL * 1024 * 1024;  // 64 MiB per write
        std::vector<uint16_t> buf(chunk_elems);
        for (uint64_t k = 0, total_elems = total_bytes / 2; k < total_elems; k += chunk_elems) {
            const size_t n = size_t(std::min<uint64_t>(chunk_elems, total_elems - k));
#    pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < int64_t(n); ++i) {
                uint32_t h = uint32_t(k + uint64_t(i));
                h ^= h >> 16;
                h *= 0x7FEB352Du;
                h ^= h >> 15;
                h *= 0x846CA68Bu;
                h ^= h >> 16;
                uint16_t bits = uint16_t(0x3C00u + ((h >> 7) & 0x1FFu));
                bits |= uint16_t(((h >> 31) & 1u) << 15);
                buf[size_t(i)] = bits;  // little-endian u16 at byte offset 2k
            }
            out.write(reinterpret_cast<const char *>(buf.data()), std::streamsize(n * 2));
        }
        out.close();
        check(bool(out), "Cannot write " + path.string());
    }

    // Engine names of the MLP weights (ff_in, ff_out) that run in FP8: blocks
    // 2..33. The first and last two blocks stay BF16; with them in FP8 the
    // worst window's latent corr vs BF16 was 0.9974, without 0.9994.
    static std::set<std::string> ff_names(const std::string & manifest_path) {
        yyjson_read_err                                         json_error{};
        std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
            yyjson_read_file(manifest_path.c_str(), 0, nullptr, &json_error), yyjson_doc_free);
        check(bool(doc), "Cannot read TensorRT weight manifest");
        std::set<std::string> names;
        size_t                index, count;
        yyjson_val *          entry;
        yyjson_arr_foreach(yyjson_obj_get(yyjson_doc_get_root(doc.get()), "weights"), index, count, entry) {
            const char * gguf   = yyjson_get_str(yyjson_obj_get(entry, "gguf_name"));
            const char * engine = yyjson_get_str(yyjson_obj_get(entry, "engine_name"));
            int block = -1;
            if (gguf && engine && sscanf(gguf, "dit.blk.%d.", &block) == 1 && block >= 2 && block <= 33 &&
                (strstr(gguf, ".ffn_in.weight") || strstr(gguf, ".ffn_out.weight"))) {
                names.insert(engine);
            }
        }
        check(names.size() == 64, "FP8: expected 64 MLP weights (blocks 2..33) in the manifest");
        return names;
    }

    // FP8 E4M3 on the MLP GEMMs named in `ff`. Weight: Q/DQ per
    // output channel with scale fp8w.<name>. Activation: Q/DQ per tensor with
    // scale fp8a.<name>. Both scales are named weights the refit sets (the
    // placeholders here are never used). The GLU feeding ff_out (value *
    // silu(gate), BF16 in the ONNX) is recomputed in FP32. Norms, attention,
    // qkv/out projections and the first/last layers stay BF16.
    static void insert_fp8(nvinfer1::INetworkDefinition &      net,
                           const std::set<std::string> &       ff,
                           std::deque<std::vector<uint16_t>> & keep) {
        using namespace nvinfer1;
        std::map<ITensor *, std::vector<std::pair<ILayer *, int>>> users;
        std::map<ITensor *, ILayer *>                              maker;
        std::vector<ILayer *>                                      weights;
        for (int i = 0; i < net.getNbLayers(); ++i) {
            ILayer * l = net.getLayer(i);
            for (int j = 0; j < l->getNbInputs(); ++j) {
                if (ITensor * t = l->getInput(j)) {
                    users[t].push_back({ l, j });
                }
            }
            for (int j = 0; j < l->getNbOutputs(); ++j) {
                maker[l->getOutput(j)] = l;
            }
            if (l->getType() == LayerType::kCONSTANT && ff.count(l->getName())) {
                weights.push_back(l);
            }
        }
        check(weights.size() == ff.size(), "FP8: MLP weights not found in the parsed graph");
        auto sole_user = [&](ITensor * t, LayerType type) {
            const auto & u = users[t];
            check(u.size() == 1 && u[0].first->getType() == type, "FP8: unexpected MLP graph shape");
            return u[0];
        };
        auto scale = [&](const std::string & name, Dims dims) -> ITensor & {
            keep.emplace_back(size_t(dims.nbDims ? dims.d[0] : 1), bf16(1.0f / 448));
            Weights w{ DataType::kBF16, keep.back().data(), int64_t(keep.back().size()) };
            IConstantLayer * c = net.addConstant(dims, w);
            check(c && net.setWeightsName(w, name.c_str()), "FP8: cannot add scale " + name);
            return *c->getOutput(0);
        };
        for (ILayer * c : weights) {
            const std::string name     = c->getName();
            ITensor *         w        = c->getOutput(0);
            ILayer *          shuffle  = sole_user(w, LayerType::kSHUFFLE).first;
            const auto [mm, slot]      = sole_user(shuffle->getOutput(0), LayerType::kMATRIX_MULTIPLY);
            const Dims        wd       = w->getDimensions();
            check(slot == 1 && wd.nbDims == 2, "FP8: unexpected MLP weight layout");
            Dims channels{};
            channels.nbDims = 1;
            channels.d[0]   = wd.d[1];
            ITensor & sw    = scale("fp8w." + name, channels);
            auto *    qw    = net.addQuantize(*w, sw, DataType::kFP8);
            qw->setAxis(1);
            auto * dw = net.addDequantize(*qw->getOutput(0), sw, DataType::kBF16);
            dw->setAxis(1);
            shuffle->setInput(0, *dw->getOutput(0));

            ITensor * a = mm->getInput(0);
            if (wd.d[0] != 2048) {  // ff_out: its input is value * silu(gate)
                ILayer * glu  = maker[a];
                ILayer * silu = glu ? maker[glu->getInput(1)] : nullptr;
                check(glu && silu && glu->getType() == LayerType::kELEMENTWISE &&
                          silu->getType() == LayerType::kELEMENTWISE,
                      "FP8: unexpected GLU graph shape");
                ITensor * g  = net.addCast(*silu->getInput(0), DataType::kFLOAT)->getOutput(0);
                ITensor * s  = net.addActivation(*g, ActivationType::kSIGMOID)->getOutput(0);
                ITensor * gs = net.addElementWise(*g, *s, ElementWiseOperation::kPROD)->getOutput(0);
                ITensor * v  = net.addCast(*glu->getInput(0), DataType::kFLOAT)->getOutput(0);
                a            = net.addElementWise(*v, *gs, ElementWiseOperation::kPROD)->getOutput(0);
                // Myelin rejects a matmul whose operand scales differ in type
                // (FP32 activation, BF16 weight), so quantize from BF16.
                a = net.addCast(*a, DataType::kBF16)->getOutput(0);
            }
            ITensor & sa = scale("fp8a." + name, Dims{});
            auto *    qa = net.addQuantize(*a, sa, DataType::kFP8);
            auto *    da = net.addDequantize(*qa->getOutput(0), sa, DataType::kBF16);
            mm->setInput(0, *da->getOutput(0));
        }
    }

    void build(const std::string & onnx_path, const std::string & manifest_path, const std::filesystem::path & out,
               bool fp8) {
        const auto start = std::chrono::steady_clock::now();
        fprintf(stderr, "[MM3-TRT] Building the MM3 DiT engine for this GPU from %s (one time, a few minutes)\n",
                onnx_path.c_str());
        // Stage the external data file the graph names, unless one is already
        // there (a developer's full export, say). Ours is removed afterwards.
        std::filesystem::path staged;
        {
            yyjson_read_err                                         json_error{};
            std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
                yyjson_read_file(manifest_path.c_str(), 0, nullptr, &json_error), yyjson_doc_free);
            check(bool(doc), "Cannot read TensorRT weight manifest");
            auto data = yyjson_obj_get(yyjson_doc_get_root(doc.get()), "onnx_data");
            check(yyjson_is_obj(data), "Manifest predates the weightless ONNX (no onnx_data); update it from the Model Manager");
            const char *   location = yyjson_get_str(yyjson_obj_get(data, "location"));
            const uint64_t bytes    = yyjson_get_uint(yyjson_obj_get(data, "bytes"));
            check(location && *location && bytes > 0, "Invalid onnx_data entry in the TensorRT manifest");
            const auto data_path = std::filesystem::path(onnx_path).parent_path() / location;
            std::error_code ec;
            if (!std::filesystem::is_regular_file(data_path, ec)) {
                const auto t0 = std::chrono::steady_clock::now();
                write_placeholder_data(data_path, bytes);
                staged = data_path;
                fprintf(stderr, "[MM3-TRT] Staged %.1f GiB of placeholder weights in %.0f s\n",
                        double(bytes) / (1024.0 * 1024 * 1024),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        struct Unstage {
            std::filesystem::path path;

            ~Unstage() {
                if (!path.empty()) {
                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                }
            }
        } unstage{ staged };
        std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        check(bool(builder), "Cannot create TensorRT builder");
        const uint32_t flags = 1U << uint32_t(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
        check(bool(network), "Cannot create TensorRT network");
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        check(bool(parser), "Cannot create TensorRT ONNX parser");
        check(parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kWARNING)),
              "Cannot parse the MM3 DiT ONNX graph");
        std::deque<std::vector<uint16_t>> keep;  // FP8 scale constants live until the build ends
        if (fp8) {
            insert_fp8(*network, ff_names(manifest_path), keep);
        }
        std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        check(bool(config), "Cannot create TensorRT builder config");
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
        config->setFlag(nvinfer1::BuilderFlag::kREFIT);
        auto * profile = builder->createOptimizationProfile();
        check(profile != nullptr, "Cannot create TensorRT profile");
        using Sel = nvinfer1::OptProfileSelector;
        check(profile->setDimensions("x", Sel::kMIN, nvinfer1::Dims3(1, 128, 3)) &&
                  profile->setDimensions("x", Sel::kOPT, nvinfer1::Dims3(2, 128, 689)) &&
                  profile->setDimensions("x", Sel::kMAX, nvinfer1::Dims3(2, 128, 689)) &&
                  profile->setDimensions("cond", Sel::kMIN, nvinfer1::Dims3(1, 2048, 3)) &&
                  profile->setDimensions("cond", Sel::kOPT, nvinfer1::Dims3(2, 2048, 689)) &&
                  profile->setDimensions("cond", Sel::kMAX, nvinfer1::Dims3(2, 2048, 689)) &&
                  profile->setDimensions("timestep_fourier", Sel::kMIN, nvinfer1::Dims2(1, 256)) &&
                  profile->setDimensions("timestep_fourier", Sel::kOPT, nvinfer1::Dims2(2, 256)) &&
                  profile->setDimensions("timestep_fourier", Sel::kMAX, nvinfer1::Dims2(2, 256)) &&
                  profile->setDimensions("rope_cos", Sel::kMIN, nvinfer1::Dims2(4, 32)) &&
                  profile->setDimensions("rope_cos", Sel::kOPT, nvinfer1::Dims2(690, 32)) &&
                  profile->setDimensions("rope_cos", Sel::kMAX, nvinfer1::Dims2(690, 32)) &&
                  profile->setDimensions("rope_sin", Sel::kMIN, nvinfer1::Dims2(4, 32)) &&
                  profile->setDimensions("rope_sin", Sel::kOPT, nvinfer1::Dims2(690, 32)) &&
                  profile->setDimensions("rope_sin", Sel::kMAX, nvinfer1::Dims2(690, 32)),
              "Cannot set the TensorRT profile");
        check(config->addOptimizationProfile(profile) >= 0, "Cannot add the TensorRT profile");

        std::atomic<bool> done{ false };
        std::thread       heartbeat([&] {
            int tick = 0;
            while (!done.load()) {
                for (int i = 0; i < 30 && !done.load(); i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                if (done.load()) {
                    break;
                }
                const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                fprintf(stderr, "[MM3-TRT] Engine build in progress (%.0f s elapsed)\n", elapsed);
                fflush(stderr);
                (void) ++tick;
            }
        });
        std::unique_ptr<nvinfer1::IHostMemory> plan;
        try {
            plan.reset(builder->buildSerializedNetwork(*network, *config));
        } catch (...) {
            done.store(true);
            heartbeat.join();
            throw;
        }
        done.store(true);
        heartbeat.join();
        check(bool(plan) && plan->size() > 0, "TensorRT engine build failed");

        std::filesystem::create_directories(out.parent_path());
        check(std::filesystem::space(out.parent_path()).available > plan->size() + 256ULL * 1024 * 1024,
              "Insufficient disk space for the MM3 TensorRT engine");
        const auto temporary =
            out.string() + ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(static_cast<const char *>(plan->data()), std::streamsize(plan->size()));
            output.close();
            check(bool(output), "Cannot write the MM3 TensorRT engine");
        }
        std::error_code ec;
        if (!std::filesystem::exists(out)) {
            std::filesystem::rename(temporary, out, ec);
            check(!ec, "Cannot place the MM3 TensorRT engine: " + ec.message());
        } else {
            std::filesystem::remove(temporary, ec);
        }
        fprintf(stderr, "[MM3-TRT] Built %s (%zu bytes) in %.0f s\n", out.string().c_str(), size_t(plan->size()),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }

    // `keep` serves the plan from / stores it in g_mm3_trt_plan. The read is
    // unbuffered 64 MB chunks into an uninitialised buffer: std::ifstream into
    // a zero-filled vector took 3.2 s for the 4.5 GB plan.
    bool deserialize(const std::filesystem::path & path, bool keep = false) {
        const auto      t0 = std::chrono::steady_clock::now();
        std::error_code ec;
        const auto      size = std::filesystem::file_size(path, ec);
        if (ec || size == 0) {
            return false;
        }
        const auto        mtime = std::filesystem::last_write_time(path, ec);
        const std::string key =
            path.string() + '|' + std::to_string(size) + '|' + std::to_string(mtime.time_since_epoch().count());
        const bool hit = keep && !ec && g_mm3_trt_plan.bytes && g_mm3_trt_plan.key == key;
        std::unique_ptr<char[]> local;
        if (!hit) {
            if (keep) {
                mm3_trt_plan_drop("superseded");
            }
#    ifdef _WIN32
            std::FILE * file = _wfopen(path.c_str(), L"rb");
#    else
            std::FILE * file = std::fopen(path.c_str(), "rb");
#    endif
            if (!file) {
                return false;
            }
            std::setvbuf(file, nullptr, _IONBF, 0);
            local.reset(new char[size]);
            size_t done = 0;
            while (done < size) {
                const size_t got = std::fread(local.get() + done, 1, std::min<size_t>(size - done, 64u << 20), file);
                if (got == 0) {
                    break;
                }
                done += got;
            }
            std::fclose(file);
            if (done != size) {
                return false;
            }
        }
        const char * data = hit ? g_mm3_trt_plan.bytes.get() : local.get();
        const auto   t1   = std::chrono::steady_clock::now();
        engine.reset(runtime->deserializeCudaEngine(data, size));
        fprintf(stderr, "[MM3-TRT] Engine plan %s %.0f ms (%.2f GB), deserialize %.0f ms\n",
                hit ? "from host copy" : "file read", std::chrono::duration<double, std::milli>(t1 - t0).count(),
                double(size) / 1073741824.0,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
        if (!engine) {
            if (hit) {
                mm3_trt_plan_drop("rejected by TensorRT");
            }
            return false;
        }
        if (keep && !hit && !ec) {
            g_mm3_trt_plan = MM3TrtPlanCache{ key, std::move(local), size_t(size) };
        }
        return true;
    }

    void save_cache(const std::filesystem::path & path) {
        std::filesystem::path temporary;
        try {
            std::filesystem::create_directories(path.parent_path());
            std::unique_ptr<nvinfer1::IHostMemory> bytes(engine->serialize());
            check(bool(bytes), "Cannot serialize refitted engine");
            check(std::filesystem::space(path.parent_path()).available > bytes->size() + 256ULL * 1024 * 1024,
                  "Insufficient space for native engine cache");
            temporary =
                path.string() + ".tmp." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(static_cast<const char *>(bytes->data()), std::streamsize(bytes->size()));
            output.close();
            check(bool(output), "Cannot write native engine cache");
            // The deterministic destination is immutable. A racing preparation
            // can win without invalidating this process's already-loaded model.
            if (!std::filesystem::exists(path)) {
                std::filesystem::rename(temporary, path);
            } else {
                std::filesystem::remove(temporary);
            }
            fprintf(stderr, "[MM3-TRT] Saved refitted weight cache: %s\n", path.string().c_str());
        } catch (const std::exception & e) {
            std::error_code ec;
            if (!temporary.empty()) {
                std::filesystem::remove(temporary, ec);
            }
            fprintf(stderr, "[MM3-TRT] Weight cache unavailable: %s (rendering remains available)\n", e.what());
        }
    }

    // The DiT as a CPU refit source. The rest load stages it only when an
    // MM3_ADAPTER merge needs it; a cache hit never reads it.
    struct StagedDit {
        WeightCtx w = {};
        ~StagedDit() { wctx_free(&w); }
    };

    static void stage_dit(const MM3Model & m, WeightCtx * w) {
        const std::string & path = m.role_file[MM3_R_DIT].path;
        GGUFModel           gf   = {};
        check(gf_load(&gf, path.c_str()), "Cannot open DiT GGUF for the TensorRT refit: " + path);
        std::vector<std::string> errs;
        MM3DitWeights            unused;
        wctx_init(w, 10 + int(m.synth_cfg.dit.block_count) * 10);
        MM3Loader ld{ w, &gf, nullptr, &errs };
        mm3_load_dit_tensors(m.synth_cfg.dit, &unused, ld);
        const bool ok = errs.empty() && wctx_alloc(w, m.cpu_backend);
        gf_close(&gf);
        check(ok, errs.empty() ? std::string("CPU allocation failed for TensorRT DiT refit sources") : errs[0]);
    }

    void refit(const MM3Model & m, const std::string & manifest_path) {
        StagedDit      staged;
        ggml_context * dit = m.wctx_dit_cpu.ctx;
        if (!dit || !ggml_get_tensor(dit, "dit.proj_in.weight")) {
            stage_dit(m, &staged.w);
            dit = staged.w.ctx;
        }
        yyjson_read_err                                         json_error{};
        std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
            yyjson_read_file(manifest_path.c_str(), 0, nullptr, &json_error), yyjson_doc_free);
        check(bool(doc), "Cannot read TensorRT weight manifest");
        auto root = yyjson_doc_get_root(doc.get());
        check(yyjson_get_int(yyjson_obj_get(root, "version")) == 1, "Unsupported TensorRT manifest version");
        const char * contract = yyjson_get_str(yyjson_obj_get(root, "contract"));
        check(contract && std::string(contract) == "mm3-dit-bf16-fp32-rope-v1", "Wrong TensorRT model contract");
        auto weights = yyjson_obj_get(root, "weights");
        check(yyjson_is_arr(weights) && yyjson_arr_size(weights) == 368, "Expected all 368 DiT parameters in manifest");
        std::unique_ptr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
        check(bool(refitter), "Engine is not refittable");
        std::set<std::string>              seen_gguf, seen_engine;
        std::vector<std::vector<uint16_t>> storage;
        storage.reserve(368 + 2 * 72);
        // FP8 engine: per-tensor activation amax from the calibration file,
        // for the GEMMs the build quantized.
        std::map<std::string, float> act_amax;
        if (mm3_trt_fp8(m)) {
            const auto ff = ff_names(manifest_path);
            const auto                                              path = mm3_trt_fp8_calib_path(m);
            std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> calib(
                yyjson_read_file(path.c_str(), 0, nullptr, &json_error), yyjson_doc_free);
            check(bool(calib), "Cannot read FP8 calibration " + path);
            auto         amax = yyjson_obj_get(yyjson_doc_get_root(calib.get()), "act_amax");
            size_t       i, n;
            yyjson_val * key;
            yyjson_val * value;
            yyjson_obj_foreach(amax, i, n, key, value) {
                if (ff.count(yyjson_get_str(key))) {
                    // A zero, negative or non-numeric amax gives a zero scale,
                    // and the quantize then writes inf/NaN into a cached plan.
                    const double v = yyjson_get_num(value);
                    check(yyjson_is_num(value) && std::isfinite(v) && v > 0.0,
                          std::string("FP8 calibration has a bad act_amax for ") + yyjson_get_str(key));
                    act_amax[yyjson_get_str(key)] = float(v);
                }
            }
            check(act_amax.size() == ff.size(), "FP8 calibration is missing MLP inputs");
        }
        size_t       index, count;
        yyjson_val * entry;
        yyjson_arr_foreach(weights, index, count, entry) {
            const char * source          = yyjson_get_str(yyjson_obj_get(entry, "gguf_name"));
            const char * target          = yyjson_get_str(yyjson_obj_get(entry, "engine_name"));
            auto         transpose_value = yyjson_obj_get(entry, "transpose");
            check(source && target && yyjson_is_bool(transpose_value), "Invalid weight mapping");
            check(seen_gguf.insert(source).second && seen_engine.insert(target).second, "Duplicate weight mapping");
            auto tensor = ggml_get_tensor(dit, source);
            check(tensor != nullptr, std::string("Missing DiT tensor: ") + source);
            const size_t n         = size_t(ggml_nelements(tensor));
            const auto   prototype = refitter->getWeightsPrototype(target);
            check(prototype.type == nvinfer1::DataType::kBF16 && prototype.count == int64_t(n),
                  std::string("Engine weight shape/type mismatch: ") + target);
            std::vector<float> values(n);
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, values.data(), 0, n * 4);
            } else if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(n);
                ggml_backend_tensor_get(tensor, half.data(), 0, n * 2);
                ggml_fp16_to_fp32_row(half.data(), values.data(), int64_t(n));
            } else {
                // A quantized DiT (Q8_0 is the common download) is widened
                // row by row through ggml's own dequantizer, so the engine
                // renders exactly the weights the GGML path would have used.
                const auto * traits = ggml_get_type_traits(tensor->type);
                check(traits && traits->to_float && ggml_is_contiguous(tensor),
                      std::string("Cannot dequantize DiT tensor for TensorRT: ") + source);
                std::vector<uint8_t> raw(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
                const int64_t rows     = ggml_nrows(tensor);
                const int64_t per_row  = tensor->ne[0];
                const size_t  row_size = ggml_row_size(tensor->type, per_row);
#    pragma omp parallel for schedule(static)
                for (int64_t r = 0; r < rows; ++r) {
                    traits->to_float(raw.data() + size_t(r) * row_size, values.data() + size_t(r) * size_t(per_row),
                                     per_row);
                }
            }
            storage.emplace_back(n);
            auto & dest = storage.back();
            if (yyjson_get_bool(transpose_value)) {
                check(tensor->ne[2] == 1 && tensor->ne[3] == 1, "Only matrix weights can be transposed");
                const int64_t in = tensor->ne[0], out = tensor->ne[1];
#    pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < in; ++i) {
                    for (int64_t o = 0; o < out; ++o) {
                        dest[size_t(i * out + o)] = bf16(values[size_t(o * in + i)]);
                    }
                }
                const auto act = act_amax.find(target);
                if (act != act_amax.end()) {
                    // FP8 scales: weight amax per output channel (of the merged
                    // weights, so adapters are covered), activation amax / 448.
                    auto & sw = storage.emplace_back(size_t(out));
#    pragma omp parallel for schedule(static)
                    for (int64_t o = 0; o < out; ++o) {
                        float amax = 1e-8f;
                        for (int64_t i = 0; i < in; ++i) {
                            amax = std::max(amax, std::fabs(values[size_t(o * in + i)]));
                        }
                        sw[size_t(o)] = bf16(amax / 448.0f);
                    }
                    auto & sa = storage.emplace_back(1, bf16(act->second / 448.0f));
                    check(refitter->setNamedWeights(("fp8w." + std::string(target)).c_str(),
                                                    { nvinfer1::DataType::kBF16, sw.data(), int64_t(out) }) &&
                              refitter->setNamedWeights(("fp8a." + std::string(target)).c_str(),
                                                        { nvinfer1::DataType::kBF16, sa.data(), 1 }),
                          std::string("Cannot refit FP8 scales of ") + target);
                }
            } else {
#    pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < int64_t(n); ++i) {
                    dest[size_t(i)] = bf16(values[size_t(i)]);
                }
            }
            check(refitter->setNamedWeights(target, { nvinfer1::DataType::kBF16, dest.data(), int64_t(n) }),
                  std::string("Cannot refit ") + target);
        }
        for (auto tensor = ggml_get_first_tensor(dit); tensor; tensor = ggml_get_next_tensor(dit, tensor)) {
            const std::string name = ggml_get_name(tensor);
            if (name != "dit.rope_inv_freq" && name != "dit.time_fourier.weight") {
                check(seen_gguf.count(name) == 1, "Unmapped DiT parameter: " + name);
            }
        }
        check(refitter->getMissingWeights(0, nullptr) == 0, "Incomplete refit weights");
        check(refitter->refitCudaEngine(), "DiT weight refit failed");
    }
    // Dev only: MM3_DIT_TRT_DUMP=<dir> writes every forward as call_<n>.bin
    // (int32 L, B; float t, gate; int32 has_cond; x [128,L]; cond [L,2048] if
    // present; velocity [B,128,L]) for tools/mm3-trt-export/fp8_lab.py, which
    // calibrates FP8 activation scales and compares renderers on identical input.
    void dump(const float * x, const float * cond, float gate, float t, int L, int B) {
        static const char * dir = std::getenv("MM3_DIT_TRT_DUMP");
        static int          n   = 0;
        if (!dir || !*dir) {
            return;
        }
        char name[32];
        snprintf(name, sizeof(name), "call_%05d.bin", n++);
        std::ofstream f(std::filesystem::path(dir) / name, std::ios::binary);
        const int32_t head[2] = { L, B }, has = cond != nullptr;
        const float   tg[2]   = { t, gate };
        f.write((const char *) head, 8).write((const char *) tg, 8).write((const char *) &has, 4);
        f.write((const char *) x, std::streamsize(size_t(128) * L * 4));
        if (cond) {
            f.write((const char *) cond, std::streamsize(size_t(2048) * L * 4));
        }
        f.write((const char *) velocity.data(), std::streamsize(velocity.size() * 4));
    }

  public:
    size_t gpu_bytes() const override { return resident_bytes; }

    ~Runtime() override {
        // CUDA current-device state belongs to each host thread.
        int old = 0;
        cudaGetDevice(&old);
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
        }
        context.reset();
        engine.reset();
        runtime.reset();
        for (Buffer * b : { &dx, &dc, &dt, &drc, &drs, &dy }) {
            if (b->data) {
                cudaFree(b->data);
            }
            b->data = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
        }
        cudaSetDevice(old);
    }

    void load(const MM3Model & m) {
        const auto   start        = std::chrono::steady_clock::now();
        const char * backend_name = ggml_backend_name(m.backend);
        check(backend_name && sscanf(backend_name, "CUDA%d", &device) == 1, "TensorRT requires a CUDA backend");
        DeviceScope  device_scope(device);
        const auto & c = m.synth_cfg.dit;
        check(c.block_count == 36 && c.embedding_length == 2048 && c.head_count == 32 && c.head_dim == 64 &&
                  c.in_channels == 128 && c.condition_dim == 2048 && c.ff_inner == 8192 && c.rope_dim == 32 &&
                  c.rope_theta == 10000 && c.fourier_dim == 256 && c.rope_type == "neox" &&
                  c.glu_order == "value_gate" && c.concat_channels == 2304 && c.layer_norm_eps == 1e-5f &&
                  !c.attn_bias && c.timestep_token_prepended && c.pre_post_conv_residual,
              "Selected GGUF does not match the native MM3 DiT architecture");
        size_t free_before = 0, total = 0;
        cuda_check(cudaMemGetInfo(&free_before, &total), "query GPU memory");
        const auto manifest = mm3_trt_manifest_path(m);
        check(std::filesystem::is_regular_file(manifest), "Missing MM3 TensorRT weight manifest: " + manifest);
        // Base engine: a supplied prebuilt one wins; otherwise the one built for
        // this GPU + TensorRT from the ONNX, building it now if absent.
        std::string path = mm3_trt_prebuilt_engine_path(m);
        if (path.empty()) {
            const auto base = mm3_trt_base_engine_path(m, device);
            if (!std::filesystem::is_regular_file(base)) {
                const auto onnx = mm3_trt_onnx_path(m);
                check(std::filesystem::is_regular_file(onnx), "Missing MM3 TensorRT DiT graph: " + onnx);
                const auto rt = hot_step_trt_runtime_probe(0);
                check(rt.parser, "TensorRT ONNX parser (nvonnxparser_10.dll) is not installed");
                build(onnx, manifest, base, mm3_trt_fp8(m));
            }
            path = base.string();
        }
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        check(bool(file), "Cannot open MM3 TensorRT engine: " + path);
        const auto length = file.tellg();
        check(length > 0, "Empty TensorRT engine");
        // Engine weights plus context/input headroom. Never request a WDDM spill.
        check(free_before > size_t(length) + 1024ULL * 1024 * 1024,
              "Insufficient dedicated GPU memory for MM3 TensorRT");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        check(bool(runtime), "Cannot create TensorRT runtime");
        const auto prepared  = cache_path(m, path, manifest);
        bool       cache_hit = std::filesystem::is_regular_file(prepared) && deserialize(prepared, /*keep=*/true);
        if (!cache_hit) {
            // A rejected file belongs to our derived cache, not the supplied
            // engine. Remove it so successful refitting can repair the cache.
            std::error_code ec;
            std::filesystem::remove(prepared, ec);
            deserialize(path);
        }
        check(bool(engine), "Cannot load TensorRT engine; prepare it for this GPU and TensorRT version");
        check(engine->getNbOptimizationProfiles() == 1 && engine->getNbIOTensors() == 6,
              "Wrong MM3 TensorRT engine interface");
        for (const char * name : { "x", "cond", "timestep_fourier", "rope_cos", "rope_sin", "output" }) {
            check(engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT, "Expected F32 engine interface");
        }
        const auto loaded = std::chrono::steady_clock::now();
        if (!cache_hit) {
            refit(m, manifest);
        }
        const auto refitted = std::chrono::steady_clock::now();
        if (!cache_hit) {
            save_cache(prepared);
        } else {
            fprintf(stderr, "[MM3-TRT] Reused refitted weight cache: %s\n", prepared.string().c_str());
        }
        const auto setup = std::chrono::steady_clock::now();
        basis.resize(128);
        check(m.synth.dit.time_fourier->type == GGML_TYPE_F32, "Fourier basis must remain F32");
        ggml_backend_tensor_get(m.synth.dit.time_fourier, basis.data(), 0, basis.size() * 4);
        cuda_check(cudaStreamCreate(&stream), "create stream");
        context.reset(engine->createExecutionContext());
        check(bool(context), "Cannot create TensorRT context");
        check(context->setOptimizationProfileAsync(0, stream), "Cannot select TensorRT profile");
        dx.allocate(2 * 128 * 689);
        dc.allocate(2 * 2048 * 689);
        dt.allocate(512);
        drc.allocate(690 * 32);
        drs.allocate(690 * 32);
        dy.allocate(2 * 128 * 689);
        check(context->setTensorAddress("x", dx.data) && context->setTensorAddress("cond", dc.data) &&
                  context->setTensorAddress("timestep_fourier", dt.data) &&
                  context->setTensorAddress("rope_cos", drc.data) && context->setTensorAddress("rope_sin", drs.data) &&
                  context->setTensorAddress("output", dy.data),
              "Cannot bind TensorRT inputs");
        cuda_check(cudaStreamSynchronize(stream), "finish setup");
        fprintf(stderr, "[MM3-TRT] Context and buffers %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup).count());
        size_t free_after = 0;
        cuda_check(cudaMemGetInfo(&free_after, &total), "query resident memory");
        resident_bytes = free_before > free_after ? free_before - free_after : size_t(length);
        fprintf(
            stderr,
            "[MM3-TRT] Loaded cached engine in %.0f ms, refit selected DiT weights in %.0f ms, %.1f MiB GPU resident\n",
            std::chrono::duration<double, std::milli>(loaded - start).count(),
            std::chrono::duration<double, std::milli>(refitted - loaded).count(), double(resident_bytes) / 1048576);
    }

    bool run(const float * x,
             const float * cond,
             float         gate,
             float         t,
             int64_t       length,
             float *       out_c,
             float *       out_u,
             std::string * err) override {
        try {
            DeviceScope scope(device);
            check(length >= 3 && length <= 689 && std::isfinite(t) && std::isfinite(gate),
                  "Invalid MM3 TensorRT forward shape or timestep");
            const int    L = int(length), B = out_u ? 2 : 1;
            const size_t N = size_t(128) * L, C = size_t(2048) * L;
            const bool   shape_changed    = cached_L != L;
            const float  effective_gate   = B == 2 ? 1.0f : gate;
            const bool   upload_condition = cond || shape_changed || uploaded_B != B || uploaded_gate != effective_gate;
            if (cached_L != L) {
                condition.clear();
                cached_L = L;
            }
            if (cond) {
                condition.resize(C);
                for (int p = 0; p < L; ++p) {
                    for (int c = 0; c < 2048; ++c) {
                        condition[size_t(c) * L + p] = cond[size_t(p) * 2048 + c];
                    }
                }
            }
            check(condition.size() == C, "No condition supplied for this TensorRT window");
            bx.resize(B * N);
            if (upload_condition) {
                bc.assign(B * C, 0);
            }
            fourier.resize(B * 256);
            velocity.resize(B * N);
            for (int b = 0; b < B; ++b) {
                std::copy_n(x, N, bx.data() + b * N);
            }
            if (upload_condition) {
                for (size_t i = 0; i < C; ++i) {
                    bc[i] = condition[i] * effective_gate;
                }
            }
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < 128; ++j) {
                    const double angle         = 2.0 * 3.14159265358979323846 * double(t) * double(basis[j]);
                    fourier[b * 256 + j]       = float(std::cos(angle));
                    fourier[b * 256 + 128 + j] = float(std::sin(angle));
                }
            }
            if (shape_changed) {
                cosine.resize(size_t(L + 1) * 32);
                sine.resize(cosine.size());
                for (int p = 0; p <= L; ++p) {
                    for (int j = 0; j < 16; ++j) {
                        const double angle         = p / std::pow(10000.0, double(2 * j) / 32);
                        cosine[size_t(p) * 32 + j] = cosine[size_t(p) * 32 + j + 16] = float(std::cos(angle));
                        sine[size_t(p) * 32 + j] = sine[size_t(p) * 32 + j + 16] = float(std::sin(angle));
                    }
                }
            }
            check(context->setInputShape("x", nvinfer1::Dims3(B, 128, L)) &&
                      context->setInputShape("cond", nvinfer1::Dims3(B, 2048, L)) &&
                      context->setInputShape("timestep_fourier", nvinfer1::Dims2(B, 256)) &&
                      context->setInputShape("rope_cos", nvinfer1::Dims2(L + 1, 32)) &&
                      context->setInputShape("rope_sin", nvinfer1::Dims2(L + 1, 32)),
                  "TensorRT profile does not support this window");
            const auto output_shape = context->getTensorShape("output");
            check(output_shape.nbDims == 3 && output_shape.d[0] == B && output_shape.d[1] == 128 &&
                      output_shape.d[2] == L,
                  "TensorRT output shape does not match the allocated velocity buffer");
            auto upload = [&](Buffer & buffer, const std::vector<float> & values) {
                cuda_check(
                    cudaMemcpyAsync(buffer.data, values.data(), values.size() * 4, cudaMemcpyHostToDevice, stream),
                    "upload inputs");
            };
            upload(dx, bx);
            if (upload_condition) {
                upload(dc, bc);
            }
            upload(dt, fourier);
            if (shape_changed) {
                upload(drc, cosine);
                upload(drs, sine);
            }
            check(context->enqueueV3(stream), "TensorRT DiT forward failed");
            cuda_check(cudaMemcpyAsync(velocity.data(), dy.data, velocity.size() * 4, cudaMemcpyDeviceToHost, stream),
                       "read velocity");
            cuda_check(cudaStreamSynchronize(stream), "DiT forward sync");
            for (float v : velocity) {
                if (!std::isfinite(v)) {
                    throw std::runtime_error("Non-finite TensorRT velocity");
                }
            }
            uploaded_B    = B;
            uploaded_gate = effective_gate;
            dump(x, cond, gate, t, L, B);
            std::copy_n(velocity.data(), N, out_c);
            if (out_u) {
                std::copy_n(velocity.data() + N, N, out_u);
            }
            return true;
        } catch (const std::exception & e) {
            if (err) {
                *err = e.what();
            }
            return false;
        }
    }
};
}  // namespace mm3_trt
#endif

static bool mm3_trt_prepare(const MM3Model & m, std::string * err) {
    if (m.dit_runtime) {
        return true;
    }
#ifdef HOT_STEP_TRT
    try {
        auto value = std::make_shared<mm3_trt::Runtime>();
        value->load(m);
        m.dit_runtime = std::move(value);
        return true;
    } catch (const std::exception & e) {
        if (err) {
            *err = e.what();
        }
        return false;
    }
#else
    if (err) {
        *err = "TensorRT is not available in this engine build";
    }
    return false;
#endif
}

static bool mm3_trt_run(const MM3Model & m,
                        const float *    x,
                        const float *    cond,
                        float            gate,
                        float            t,
                        int64_t          L,
                        float *          out_c,
                        float *          out_u,
                        std::string *    err) {
    if (!mm3_trt_prepare(m, err)) {
        return false;
    }
    const bool ok = m.dit_runtime->run(x, cond, gate, t, L, out_c, out_u, err);
    if (!ok) {
        m.dit_runtime.reset();
    }
    return ok;
}
