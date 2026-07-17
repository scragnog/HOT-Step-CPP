#pragma once
// backend.h: shared GGML backend initialization
//
// All modules use the same pattern: load all backends, pick best GPU,
// keep CPU as fallback. This avoids duplicating init logic across
// qwen3.h, qwen3-lm.h, cond.h, dit.h, vae.h.

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

struct BackendPair {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;
    bool           has_gpu;
};

// Cached backend state (shared across all modules in the same binary).
// Must be `inline` (not `static`) so every TU that includes this header
// shares a single instance. `static` would give each TU its own copy,
// breaking the refcount guard and causing duplicate ggml_backend_load_all
// calls (see GitHub #49).
inline BackendPair g_backend_cache = {};
inline int         g_backend_refs  = 0;

// Physical core count heuristic (logical / 2 for HT/SMT).
// Used for GGML CPU thread count: GEMM shares SIMD units across hyperthreads,
// so one thread per physical core is optimal.
static int backend_cpu_n_threads(void) {
    const char * env = std::getenv("GGML_N_THREADS");
    if (env) {
        int n = atoi(env);
        if (n > 0) return n;
    }
    int n = (int) std::thread::hardware_concurrency() / 2;
    return n > 0 ? n : 1;
}

// Standalone CPU backend via Registry API (DL-safe, no ggml-cpu.h needed).
// Sets thread count via proc address since ggml_backend_cpu_device_init_backend
// ignores its params string and always defaults to GGML_DEFAULT_N_THREADS (4).
// Returns NULL on failure.
static ggml_backend_t cpu_backend_new(int n_threads) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t     cpu     = NULL;
    if (cpu_dev) {
        cpu = ggml_backend_dev_init(cpu_dev, NULL);
    }
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    if (!cpu) {
        return NULL;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : NULL;
    if (reg) {
        auto set_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_fn) {
            set_fn(cpu, n_threads);
        }
    }
    return cpu;
}

// Filter + dedup for ggml log lines.
// 1) DEBUG-level messages are dropped entirely. The CUDA graph capture path
//    logs "CUDA Graph id N reused" / warmup lines on virtually every compute —
//    78% of a real session log. Set HOTSTEP_GGML_DEBUG=1 to pass them through
//    when debugging the CUDA graph layer itself.
// 2) Remaining consecutive lines that differ only in digits are collapsed into
//    one line + a single "[Dedup]" summary. Digit-insensitive comparison
//    matters: alternating graph ids ("id 3 reused" / "id 4 reused") defeated
//    the old exact-match dedup AND re-emitted its summary on every alternation
//    — the dedup itself became log spam.
static void acestep_ggml_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    static const bool debug_pass = std::getenv("HOTSTEP_GGML_DEBUG") != nullptr;
    if (level == GGML_LOG_LEVEL_DEBUG && !debug_pass) {
        return;
    }

    static char last_norm[256] = { 0 };
    static int  count          = 0;

    // Normalize digit runs to '#' so lines differing only in numbers compare
    // equal ("id 3 reused" == "id 47 reused").
    char   norm[256];
    size_t j      = 0;
    bool   in_num = false;
    for (size_t i = 0; text[i] && j < sizeof(norm) - 1; i++) {
        char c = text[i];
        if (c >= '0' && c <= '9') {
            if (!in_num) {
                norm[j++] = '#';
                in_num    = true;
            }
        } else {
            norm[j++] = c;
            in_num    = false;
        }
    }
    norm[j] = 0;

    if (count > 0 && strcmp(norm, last_norm) == 0) {
        count++;
        return;
    }

    if (count > 1) {
        fprintf(stderr, "[Dedup] previous message repeated %d times\n", count);
    }

    fputs(text, stderr);
    strncpy(last_norm, norm, sizeof(last_norm) - 1);
    last_norm[sizeof(last_norm) - 1] = 0;
    count                            = 1;
    fflush(stderr);
}

// Initialize backends: load all available (CUDA, Metal, Vulkan...),
// pick the best one, keep CPU as fallback.
// label: log prefix, e.g. "DiT", "VAE", "LM"
// Subsequent calls reuse the same backend (single VMM pool).
static BackendPair backend_init(const char * label) {
    static bool log_installed = false;
    if (!log_installed) {
        ggml_log_set(acestep_ggml_log, nullptr);
        log_installed = true;
    }

    if (g_backend_refs > 0) {
        g_backend_refs++;
        fprintf(stderr, "[Load] %s backend: %s (shared)\n", label, ggml_backend_name(g_backend_cache.backend));
        return g_backend_cache;
    }

    ggml_backend_load_all();
    BackendPair bp = {};

    // GGML_BACKEND env var: force a specific device instead of auto-best.
    // Device names: CUDA0, Vulkan0, CPU, BLAS (see ggml_backend_dev_name).
    const char * force_backend = std::getenv("GGML_BACKEND");
    if (force_backend) {
        bp.backend = ggml_backend_init_by_name(force_backend, nullptr);
        if (!bp.backend) {
            fprintf(stderr, "[Load] FATAL: GGML_BACKEND=%s not found. Available:", force_backend);
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            }
            fprintf(stderr, "\n");
            exit(1);
        }
    } else {
        bp.backend = ggml_backend_init_best();
    }
    if (!bp.backend) {
        fprintf(stderr, "[Load] FATAL: no backend available\n");
        exit(1);
    }
    bool best_is_cpu = (strcmp(ggml_backend_name(bp.backend), "CPU") == 0);
    int  n_threads   = backend_cpu_n_threads();
    if (best_is_cpu) {
        ggml_backend_free(bp.backend);
        bp.backend     = cpu_backend_new(n_threads);
        bp.cpu_backend = bp.backend;
    } else {
        bp.cpu_backend = cpu_backend_new(n_threads);
    }
    if (!bp.cpu_backend) {
        fprintf(stderr, "[Load] FATAL: failed to init CPU backend\n");
        exit(1);
    }
    bp.has_gpu = !best_is_cpu;

#ifdef _WIN32
    // Diagnose why CUDA backend didn't load (when ggml-cuda.dll is present).
    // Try loading each dependency DLL individually and report the exact Windows
    // error for each failure. This makes missing-DLL issues self-diagnosing
    // instead of requiring back-and-forth with the user.
    if (best_is_cpu) {
        wchar_t exe_path[MAX_PATH];
        GetModuleFileNameW(NULL, exe_path, MAX_PATH);
        std::wstring dir(exe_path);
        auto pos = dir.find_last_of(L'\\');
        if (pos != std::wstring::npos) dir = dir.substr(0, pos);

        bool cuda_dll = GetFileAttributesW((dir + L"\\ggml-cuda.dll").c_str()) != INVALID_FILE_ATTRIBUTES;
        bool vulkan_dll = GetFileAttributesW((dir + L"\\ggml-vulkan.dll").c_str()) != INVALID_FILE_ATTRIBUTES;

        if (cuda_dll) {
            fprintf(stderr, "[Load] WARNING: ggml-cuda.dll found but CUDA backend did not load.\n");
            fprintf(stderr, "[Load]   Diagnosing dependency chain...\n");

            // DLLs that ggml-cuda.dll imports (load order matters: base first).
            // Names use narrow strings for logging, wide strings for LoadLibrary.
            struct DllCheck { const wchar_t * wname; const char * name; bool required; };
            DllCheck deps[] = {
                { L"ggml-base.dll",        "ggml-base.dll",        true  },
                { L"cublas64_13.dll",       "cublas64_13.dll",      true  },
                { L"cublasLt64_13.dll",     "cublasLt64_13.dll",    true  },
                { L"VCRUNTIME140.dll",      "VCRUNTIME140.dll",     true  },
                { L"VCRUNTIME140_1.dll",    "VCRUNTIME140_1.dll",   true  },
                { L"MSVCP140.dll",          "MSVCP140.dll",         true  },
            };

            bool any_failed = false;
            for (auto & dep : deps) {
                // Check if file exists in engine dir
                std::wstring full = dir + L"\\" + dep.wname;
                bool on_disk = GetFileAttributesW(full.c_str()) != INVALID_FILE_ATTRIBUTES;

                // Try loading (searches exe dir + system PATH)
                HMODULE h = LoadLibraryW(dep.wname);
                if (h) {
                    FreeLibrary(h);
                } else {
                    DWORD err = GetLastError();
                    wchar_t msg_buf[512] = {};
                    FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                   NULL, err, 0, msg_buf, 512, NULL);
                    // Trim trailing \r\n
                    size_t len = wcslen(msg_buf);
                    while (len > 0 && (msg_buf[len-1] == L'\n' || msg_buf[len-1] == L'\r'))
                        msg_buf[--len] = 0;
                    fprintf(stderr, "[Load]   FAIL: %s — %s (error %lu: %ls)\n",
                            dep.name,
                            on_disk ? "file exists but won't load" : "NOT FOUND in engine dir",
                            (unsigned long) err, msg_buf);
                    any_failed = true;
                }
            }

            // Now try ggml-cuda.dll itself to get its specific error
            HMODULE hcuda = LoadLibraryW(L"ggml-cuda.dll");
            if (hcuda) {
                fprintf(stderr, "[Load]   NOTE: ggml-cuda.dll loads OK now but GGML didn't pick it up.\n");
                fprintf(stderr, "[Load]   This may be a GGML backend registration issue.\n");
                FreeLibrary(hcuda);
            } else {
                DWORD err = GetLastError();
                wchar_t msg_buf[512] = {};
                FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                               NULL, err, 0, msg_buf, 512, NULL);
                size_t len = wcslen(msg_buf);
                while (len > 0 && (msg_buf[len-1] == L'\n' || msg_buf[len-1] == L'\r'))
                    msg_buf[--len] = 0;
                fprintf(stderr, "[Load]   ggml-cuda.dll load error %lu: %ls\n",
                        (unsigned long) err, msg_buf);
            }

            if (any_failed) {
                fprintf(stderr, "[Load]   Fix: Settings -> Model Manager -> CUDA Runtime -> Download\n");
                fprintf(stderr, "[Load]   Or re-download the CUDA release ZIP and extract all DLLs.\n");
            } else {
                fprintf(stderr, "[Load]   All dependency DLLs loaded OK individually.\n");
                fprintf(stderr, "[Load]   This may be a driver issue. Check: nvidia-smi\n");
                fprintf(stderr, "[Load]   Minimum driver: 560.xx for CUDA 12.8, 570.xx for CUDA 13.x\n");
            }
        } else if (vulkan_dll) {
            // Vulkan build but Vulkan didn't load — different diagnosis
            fprintf(stderr, "[Load] WARNING: ggml-vulkan.dll found but Vulkan backend did not load.\n");
            fprintf(stderr, "[Load]   Check that your GPU drivers include Vulkan support.\n");
        }
    }
#endif

    fprintf(stderr, "[Load] %s backend: %s (CPU threads: %d)\n", label, ggml_backend_name(bp.backend), n_threads);

    g_backend_cache = bp;
    g_backend_refs  = 1;
    return bp;
}

// Release a backend reference. Frees GPU + CPU backends when refcount hits 0.
static void backend_release(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    if (g_backend_refs <= 0) {
        return;
    }
    g_backend_refs--;
    if (g_backend_refs == 0) {
        if (backend && backend != cpu_backend) {
            ggml_backend_free(backend);
        }
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
        g_backend_cache = {};
    }
}

// Create a scheduler from a backend pair.
// max_nodes: graph size hint (4096 for small models, 8192 for large)
// When a GPU is present, use its host buffer type for the CPU backend.
// Pinned memory lets the scheduler keep more ops on GPU instead of
// falling back to CPU with plain malloc.
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { NULL, NULL };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : NULL;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        fprintf(stderr, "[Load] FATAL: failed to create scheduler\n");
        exit(1);
    }
    return sched;
}
